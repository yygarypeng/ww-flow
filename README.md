# ww-flow

Flow-based reconstruction of $H \to WW^\ast \to \ell\nu\ell\nu$ kinematics with an invertible neural network.

Contact: [Yuan-Yen Peng](mailto:yuan-yen.peng@cern.ch)

## Overview

This repository trains a conditional INN to map between reconstructed event observables and truth-level $W$ boson kinematics.

The current model uses two stages:

1. A physics block converts between $W$ bosons and neutrinos using lepton four-vectors and the massless-neutrino constraint.
2. A conditional normalizing flow models the remaining under-constrained degrees of freedom with latent variables.

The training setup is now config-driven via [config.yaml](config.yaml), not hard-coded in [train.py](train.py).

## Current Setup

### Default paths

- Input HDF5: `/root/data/danning_h5/ypeng/mc20_qe_v4_recotruth_merged.h5`
- Log root: `/root/work/ww-flow/logs/`
- Project name: `hww_inn_regressor`

These values are defined in [config.yaml](config.yaml).

### Default training configuration

| Parameter | Current value | Notes |
|-----------|---------------|-------|
| `batch_size` | `256` | Set in [config.yaml](config.yaml) |
| `epochs` | `1024` | Max epochs |
| `learning_rate` | `5e-5` | Used by `AdamW` |
| `num_blocks` | `30` | Number of FrEIA `AllInOneBlock`s |
| `obs_dim` | `10` | First 10 reco features are supervised observables |
| `lack_dim` | `4` | Latent dimension |
| early stopping patience | `128` | Set in [train.py](train.py) |
| validation fraction | `0.05` | Set in [train.py](train.py) |
| test fraction | `0.05` | Set in [train.py](train.py) |

### Active loss weights

| Loss | Weight |
|------|--------|
| `L_x` | `1.0` |
| `L_y` | `1.0` |
| `L_z` | `1.0` |
| `L_pad` | `1.0` |
| `L_W` | `1.0` |
| `L_higgs` | `0.5` |
| `L_neu_mass` | `0.0` |
| `L_x_huber` | `0.0` |

## Data Layout

The training code uses INN-style notation, so the usual supervised-learning input/target naming is inverted relative to a standard regressor.

- Reco features from [load_data.py](load_data.py) are returned first and stored as `llvv`
- Truth targets are returned second and stored as `ww`
- In [train.py](train.py), `X = ww` and `Y = llvv`

### Reco feature tensor `Y`

`Y` has 32 features per event:

- 10 observed variables used as `y`
  - positive lepton: `px, py, pz, E`
  - negative lepton: `px, py, pz, E`
  - MET: `px, py`
- 22 conditioning variables used as `c`
  - leading 3 jets: `3 x (px, py, pz, E)` = 12
  - dilepton system: `px, py, pz, E` = 4
  - angular features: `deta(l1,l2), dphi(ll,met), dphi(l1,met), dphi(l2,met), dphi(l1,l2), dr(l1,l2)` = 6

### Truth tensor `X`

The raw truth tensor has 16 features per event:

- 8 trained features
  - `W+`: `px, py, pz, m`
  - `W-`: `px, py, pz, m`
- 8 auxiliary neutrino features kept for scaling and the physics block
  - neutrino and anti-neutrino: `px, py, pz, m=0`

Training uses only the first 8 truth features as the direct INN input.

### Effective model dimensions

With the current config and loader:

- `x_dim = 8`
- `inputs_dim = 32`
- `y_dim = 10`
- `c_dim = 22`
- `z_dim = 4`
- `internal_dim = y_dim + z_dim = 14`
- `input_pad = internal_dim - x_dim = 6`

## Model Architecture

### [model.py](model.py)

`INN` combines:

- `WtoNeutrinoBlock` from [layers.py](layers.py)
  - forward direction: $(W, \ell) \to \nu$
  - reverse direction: $(\nu, \ell) \to W$
- a FrEIA `GraphINN`
  - 30 conditional `AllInOneBlock`s by default
  - one FrEIA condition node carrying the 22 conditioning features

`INNLightningModule` wraps the model for PyTorch Lightning training.

### Conditioning network

The active subnet constructor is `CondNet` in [layers.py](layers.py). It embeds:

- the INN half-state
- three jet tokens
- one dilepton token
- one angular-feature token

It then refines the INN token with 5 stacked cross-attention blocks before producing coupling parameters.

### Losses

The training step in [model.py](model.py) uses:

- `L_x`: MMD between reconstructed and true $W$ kinematics, excluding the 2 explicit $W$ masses
- `L_y`: Huber loss on the observed 10 reco variables
- `L_z`: MMD between predicted latent representation and sampled latent target
- `L_pad`: mean absolute padding penalty
- `L_W`: MMD on the two reconstructed $W$ masses
- `L_higgs`: Huber loss toward $m_H = 125$ GeV
- `L_neu_mass`: neutrino mass consistency monitor
- `L_x_huber`: optional Huber monitor on reconstructed $W$ kinematics

Loss implementations live in [losses.py](losses.py).

## File Guide

### Core training files

- [train.py](train.py): main training entry point, logger setup, checkpointing, early stopping
- [config.yaml](config.yaml): runtime configuration for data path, logging path, and hyperparameters
- [model.py](model.py): INN and Lightning module
- [layers.py](layers.py): physics block, conditioning network, archived custom coupling layers
- [losses.py](losses.py): MMD, Higgs-mass, and neutrino-mass losses
- [data_module.py](data_module.py): standardization and train/val/test dataloaders
- [load_data.py](load_data.py): HDF5 loading and feature construction
- [physics.py](physics.py): helper kinematic functions used during feature building

### Analysis and utilities

- [visualize.ipynb](visualize.ipynb): notebook for post-training checks and plots
- [ohbboosting.py](ohbboosting.py): ROOT-based Lorentz-boost utilities for helicity-basis studies
- [record](record): local notes/logs
- [logs/](logs/): checkpoints and CSV logs

## Training

1. Update the paths or hyperparameters in [config.yaml](config.yaml) if needed.
2. Start training:

```bash
python train.py
```

To enable Weights & Biases logging:

```bash
python train.py --wandb
```

### Important runtime behavior

- If training starts with `train=True` and the checkpoint directory already exists, [train.py](train.py) deletes the whole project log directory before launching a fresh run.
- The trainer uses GPU automatically when `torch.cuda.is_available()` is true.
- Checkpoints monitor `val_loss` and keep only the best model.

## Notes

- [ohbboosting.py](ohbboosting.py) is not part of the main training path.
- The archived coupling classes in [layers.py](layers.py) are retained for reference; the active flow uses FrEIA blocks.
- The HDF5 loader concatenates all top-level categories and removes rows containing NaN or infinite values.

## References

- INN formulation: [Analyzing Inverse Problems with Invertible Neural Networks](https://arxiv.org/abs/1808.04730)
- MMD loss: [A Kernel Two-Sample Test](https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf)
- Boosted basis motivation: [Testing Bell inequalities in Higgs boson decays](https://arxiv.org/abs/2106.01377)
