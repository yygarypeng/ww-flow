# ww-flow

Flow-based reconstruction tools for studying
`H -> WW* -> l nu l nu` kinematics with conditional invertible neural
networks.

The current training target is the truth lepton angular information in the
two W-boson rest frames. The model learns a conditional normalizing flow from
truth-level angular variables to reconstructed event observables, then samples
the inverse direction for generative reconstruction.

Contact: [Yuan-Yen Peng](mailto:yuan-yen.peng@cern.ch)

## Overview

This repository contains a PyTorch Lightning training pipeline for a
conditional INN based on FrEIA `AllInOneBlock` layers. The workflow is:

1. Load reconstructed and truth objects from an HDF5 ntuple.
2. Build compact reconstructed observables and conditioning features.
3. Build truth lepton-angle targets in the W rest frames.
4. Standardize inputs and targets.
5. Train a conditional INN with MMD, Huber, and padding-consistency losses.
6. Use notebooks and plotting helpers for post-training validation.

Runtime options are controlled through [`config.yaml`](config.yaml).

## Repository Layout

| File | Purpose |
|------|---------|
| [`train.py`](train.py) | Main training entry point, logging, checkpointing, and early stopping |
| [`config.yaml`](config.yaml) | Data path, output path, model dimensions, and hyperparameters |
| [`model.py`](model.py) | FrEIA INN and PyTorch Lightning module |
| [`layers.py`](layers.py) | Attention-based conditional subnet used inside coupling blocks |
| [`load_data.py`](load_data.py) | HDF5 loading and feature construction |
| [`data_module.py`](data_module.py) | Dataset standardization and train/validation/test dataloaders |
| [`losses.py`](losses.py) | MMD loss implementation |
| [`physics.py`](physics.py) | Basic kinematic helper functions |
| [`ohbboosting.py`](ohbboosting.py) | ROOT-based boosts into W rest frames and angular observables |
| [`eval.py`](eval.py) | RMSE and R2 helpers for angular validation |
| [`plottingtools.py`](plottingtools.py) | ATLAS-style histogram and residual plotting helpers |
| [`visualize.ipynb`](visualize.ipynb) | Interactive post-training checks and plotting |
| [`run_train.sh`](run_train.sh) | Example background training launcher |

## Current Configuration

Default values are defined in [`config.yaml`](config.yaml).

| Parameter | Default | Notes |
|-----------|---------|-------|
| `data_path` | `/root/data/danning_h5/ypeng/mc20_qe_v4_recotruth_merged.h5` | Input HDF5 file |
| `project_name` | `hww_inn_regressor-lep` | Log/checkpoint directory name |
| `ckpt_path` | `/root/work/ww-flow/logs/` | Log root |
| `batch_size` | `256` | Training batch size |
| `epochs` | `1024` | Maximum training epochs |
| `learning_rate` | `1.0e-4` | AdamW learning rate |
| `num_blocks` | `8` | Number of FrEIA coupling blocks |
| `obs_dim` | `8` | Observed lepton variables predicted as `y` |
| `lack_dim` | `8` | Latent/sampled dimension `z` |
| `num_workers` | `6` | DataLoader worker count |
| `persistent_workers` | `true` | DataLoader worker persistence |
| `pin_memory` | `true` | DataLoader pinned-memory transfer |
| `prefetch_factor` | `4` | DataLoader prefetch factor when workers are enabled |

Active loss weights:

| Loss | Weight | Meaning |
|------|--------|---------|
| `L_x` | `10.0` | MMD loss on generated target variables |
| `L_x_huber` | `0.5` | Huber reconstruction loss on target variables |
| `L_y` | `0.1` | Huber loss on observed reco variables |
| `L_z` | `10.0` | MMD loss encouraging Gaussian latent behavior |
| `L_pad` | `1.0` | Padding reconstruction consistency |
| `L_pad_noise` | `0.1` | Noisy-padding consistency |
| `L_x_gen` | `0.0` | Monitoring-only generated-target Huber loss |

## Data Layout

`load_data.load_data(data_path)` returns two arrays:

| Name in code | Shape | Meaning |
|--------------|-------|---------|
| `train_obj` / `llvv` | `(N, 21)` | Reconstructed observables and conditioning features |
| `target_obj` / `ww` | `(N, 6)` | Truth lepton-angle targets in W rest frames |

Rows containing NaN or infinite values are removed after all categories are
concatenated.

### Reconstructed Feature Tensor

The first `obs_dim = 8` columns are the directly predicted observed variables:

| Columns | Variables |
|---------|-----------|
| `0:4` | positive lepton `px`, `py`, `eta`, `log(E)` |
| `4:8` | negative lepton `px`, `py`, `eta`, `log(E)` |

The remaining `c_dim = 13` columns are conditioning variables:

| Columns | Variables |
|---------|-----------|
| `8:10` | MET `px`, `py` |
| `10:14` | leading jet `px`, `py`, `pz`, `log(E)` |
| `14:18` | subleading jet `px`, `py`, `pz`, `log(E)` |
| `18:21` | dilepton high-level features: `deta(l1,l2)`, `dphi(l1,l2)`, `m_ll` |

The conditional subnet in [`layers.py`](layers.py) expects this exact
13-dimensional conditioning layout.

### Target Tensor

The target tensor has 6 columns:

| Columns | Variables |
|---------|-----------|
| `0` | positive lepton `theta / pi` in the W rest frame |
| `1:3` | positive lepton `sin(phi)`, `cos(phi)` |
| `3` | negative lepton `theta / pi` in the W rest frame |
| `4:6` | negative lepton `sin(phi)`, `cos(phi)` |

The `phi` target is encoded as sine and cosine to avoid a discontinuity at the
angular boundary.

### Effective Model Dimensions

With the current defaults:

| Dimension | Value |
|-----------|-------|
| `x_dim` | `6` |
| `inputs_dim` | `21` |
| `y_dim` | `8` |
| `c_dim` | `13` |
| `z_dim` | `8` |
| `internal_dim = y_dim + z_dim` | `16` |
| `input_pad = internal_dim - x_dim` | `10` |

## Installation

This project is a research codebase and does not currently ship a locked
environment file. Install the core dependencies in a Python environment with
GPU-compatible PyTorch if training on CUDA.

Required Python packages include:

- `numpy`
- `h5py`
- `torch`
- `pytorch-lightning`
- `scikit-learn`
- `PyYAML`
- `FrEIA`
- `matplotlib`
- `mplhep`

The boosting utilities in [`ohbboosting.py`](ohbboosting.py) also require
PyROOT, because they use `ROOT.TLorentzVector` and `ROOT.TVector3`.

## Training

1. Edit [`config.yaml`](config.yaml) if the data path, log directory, or
   hyperparameters should change.
2. Run training:

```bash
python train.py
```

To enable Weights & Biases logging:

```bash
python train.py --wandb
```

The helper script launches the same training job in the background with CPU
affinity and writes output to `record`:

```bash
bash run_train.sh
```

Important runtime behavior:

- If the configured checkpoint directory already exists, `train.py` removes it
  before starting a fresh training run.
- The trainer uses GPU automatically when `torch.cuda.is_available()` is true.
- Checkpointing monitors `val_loss` and keeps only the best checkpoint.
- CSV logs are written under `logs/<project_name>/log/`.

## Evaluation And Plotting

Use [`eval.py`](eval.py) for basic RMSE and R2 calculations on predicted versus
truth arrays. Use [`plottingtools.py`](plottingtools.py) and
[`visualize.ipynb`](visualize.ipynb) for histogram, residual, and ratio-panel
checks.

The plotting utilities use ATLAS-style `mplhep` formatting and support optional
save paths for generated figures.

## Notes

- The HDF5 loader concatenates all top-level categories in the input file.
- Jet conditioning currently uses only the leading and subleading jets.
- Jet slots with exactly zero four-vectors are masked in the attention-based
  conditional subnet.
- Generated logs, checkpoints, Python bytecode, and local records are ignored
  by [`.gitignore`](.gitignore).

## References

- INN formulation: [Analyzing Inverse Problems with Invertible Neural Networks](https://arxiv.org/abs/1808.04730)
- MMD loss: [A Kernel Two-Sample Test](https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf)
- Boosted-basis motivation: [Testing Bell inequalities in Higgs boson decays](https://arxiv.org/abs/2106.01377)

## License

This repository is distributed under the terms of the [LICENSE](LICENSE) file.
