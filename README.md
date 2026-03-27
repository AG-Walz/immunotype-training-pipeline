# immunotype-training-pipeline

Training pipeline for [immunotype](https://github.com/AG-Walz/immunotype), a deep learning ensemble model for HLA class I typing from immunopeptidomics data.

## Overview

The pipeline is split into sequential notebooks:

| Notebook | Description |
|----------|-------------|
| `00_data_processing` | Data processing and allele normalization for MHC, BA, EL, and PCI databases |
| `01_pretrain_mhc_autoencoder` | Pretrain a transformer autoencoder on MHC protein sequences |
| `02_pretrain_ba` | Pretrain the GNN on binding affinity (BA) data |
| `03_pretrain_el` | Pretrain the GNN on eluted ligand (EL) data |
| `04_in_silico_db_train` | Train on in-silico generated donor samples |
| `05_pci_db_train` | Fine-tune and evaluate on PCI-DB+ with cross-validation |

## Setup

```bash
pip install -r requirements.txt
```
