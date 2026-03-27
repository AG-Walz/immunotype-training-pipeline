import torch.nn.functional as F
from torch.nn import *
import torch
import math
from torch_geometric.nn import BatchNorm, TransformerConv, HeteroConv



class PositionalEncoding(Module):
    # adapted from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x


class SequenceEncoder(Module):
    def __init__(
        self,
        embedding_dim,
        dim_ff,
        n_heads,
        n_layers,
        dropout=0.1, 
        device='cuda:0',
    ):
        super(SequenceEncoder, self).__init__()       
        encoder_layer = TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            batch_first=True,
            dim_feedforward=dim_ff,
            dropout=dropout
        )
        
        self.encoder = TransformerEncoder(
            encoder_layer=encoder_layer, 
            num_layers=n_layers,
            norm=LayerNorm(embedding_dim),
        ).to(device)
        self.bn = BatchNorm(embedding_dim, allow_single_element=True).to(device)

    def forward(self, x, src_mask):
        x = self.encoder(x, src_key_padding_mask=src_mask)
        x = torch.select(x, dim=1, index=0)
        x = F.leaky_relu(self.bn(x))
        return x


class GNN(Module):
    def __init__(
        self,
        vocab_size=45,
        embedding_dim=128,
        dim_ff_enc_pep=128,
        n_heads_enc_pep=8,
        n_layers_enc_pep=6,
        dim_ff_enc_mhc=1024,
        n_heads_enc_mhc=8,
        n_layers_enc_mhc=6,
        n_heads_conv=8,
        n_layers_conv=2,
        dim_out_conv=32,
        dropout=0.1,
        act=F.leaky_relu,
        # gumbel_init=torch.rand(1, 1),
        device1='cuda:0',
        device2='cuda:1',
        ):
        super(GNN, self).__init__()

        self.device1 = device1
        self.device2 = device2
        self.act = act

        dim = n_heads_conv * dim_out_conv

        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim).to(device1)
        self.positional_encoding = PositionalEncoding(embedding_dim).to(device1)

        # Peptide
        self.encoder = ModuleDict({
            'peptide': ModuleList([
                SequenceEncoder(embedding_dim, dim_ff_enc_pep, n_heads_enc_pep, n_layers_enc_pep, dropout, device1),
                Linear(embedding_dim, dim).to(device1)
            ]),
            'mhc': ModuleList([
                SequenceEncoder(embedding_dim, dim_ff_enc_mhc, n_heads_enc_mhc, n_layers_enc_mhc, dropout, device1),
                Linear(embedding_dim, dim * 2).to(device1)
            ])
        })
        self.bn_conv = ModuleList([ModuleDict({
            'peptide': BatchNorm(dim, allow_single_element=True).to(device2),
            'mhc': BatchNorm(dim * 2, allow_single_element=True).to(device2),
        }) for _ in range(n_layers_conv + 1)])
        
        self.conv = ModuleList([
            HeteroConv({
                ('peptide', 'determines', 'mhc'): TransformerConv((dim, dim * 2), dim_out_conv, heads=n_heads_conv, dropout=dropout).to(device2),
                ('mhc', 'influences', 'mhc'): TransformerConv(dim * 2, dim_out_conv, heads=n_heads_conv, dropout=dropout).to(device2),
                ('peptide', 'influences', 'peptide'): TransformerConv(dim, dim_out_conv, heads=n_heads_conv, dropout=dropout).to(device2),
            }, aggr='cat')
            for _ in range(n_layers_conv)
        ])
        
        self.fc = Linear(dim * 2, 1).to(device2)

    def forward(self, batch):
        x_dict = batch.x_dict
        edge_index_dict = batch.edge_index_dict
        
        x_dict = {k: v.to(self.device1) for k, v in x_dict.items()}
        edge_index_dict = {k: v.to(self.device2) for k, v in edge_index_dict.items()}

        # peptide & mhc encoding
        for nodes in ['peptide', 'mhc']:
            x = x_dict[nodes]
            src_mask = x == 0
            x = self.embedding(x)
            x = self.positional_encoding(x)
            x = self.encoder[nodes][0](x, src_mask) # encoder
            x_dict[nodes] = self.encoder[nodes][1](x) # linear

        if self.device1 != self.device2:
            x_dict = {k: v.to(self.device2) for k, v in x_dict.items()}

        # conv
        for bn, conv in zip(self.bn_conv[:-1], self.conv):
            x_res = x_dict.copy()
            x_dict = {k: self.act(bn[k](v)) for k, v in x_dict.items()}
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: v + x_res[k] for k, v in x_dict.items()}

        x_dict = {k: self.act(self.bn_conv[-1][k](v)) for k, v in x_dict.items()}
        
        # dense out
        x = self.fc(x_dict['mhc'])
        return x
