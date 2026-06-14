# ww-flow

Flow-based reconstruction tools for studying
$H \to WW^\ast \to \ell\nu\ell\nu$ kinematics with conditional invertible neural
networks.

The `main` branch targets truth-level W-boson and neutrino kinematics. It uses
a physics-motivated reversible block to convert between W-boson and neutrino
representations, then models the remaining under-constrained degrees of
freedom with a conditional normalizing flow.

Contact: [Yuan-Yen Peng](mailto:yuan-yen.peng@cern.ch)

## Overview

This repository contains a PyTorch Lightning training pipeline for a
conditional INN based on FrEIA `AllInOneBlock` layers. The workflow is:

1. Load reconstructed and truth objects from an HDF5 ntuple.
2. Build reconstructed event observables and conditioning features.
3. Build truth W-boson and neutrino targets.
4. Standardize inputs and targets.
5. Train a physics-assisted conditional INN with MMD, Huber, padding, and
   physics-motivated losses.
6. Use the notebook for post-training validation.

Runtime options are controlled through [`config.yaml`](config.yaml).

## Repository Layout

| File | Purpose |
|------|---------|
| [`train.py`](train.py) | Main training entry point, logging, checkpointing, and early stopping |
| [`config.yaml`](config.yaml) | Data path, output path, model dimensions, and hyperparameters |
| [`model.py`](model.py) | Physics-assisted FrEIA INN and PyTorch Lightning module |
| [`layers.py`](layers.py) | W-to-neutrino reversible block and attention-based conditional subnet |
| [`losses.py`](losses.py) | MMD, Higgs-mass, and neutrino-mass loss functions |
| [`load_data.py`](load_data.py) | HDF5 loading and feature construction |
| [`data_module.py`](data_module.py) | Dataset standardization and train/validation/test dataloaders |
| [`physics.py`](physics.py) | Basic kinematic helper functions |
| [`ohbboosting.py`](ohbboosting.py) | ROOT-based boosts for helicity-basis studies |
| [`visualize.ipynb`](visualize.ipynb) | Interactive post-training checks and plots |

## Current Configuration

Default values are defined in [`config.yaml`](config.yaml).

| Parameter | Default | Notes |
|-----------|---------|-------|
| `data_path` | `/root/data/danning_h5/ypeng/mc20_qe_v4_recotruth_merged.h5` | Input HDF5 file |
| `project_name` | `hww_inn_regressor` | Log/checkpoint directory name |
| `ckpt_path` | `/root/work/ww-flow/logs/` | Log root |
| `batch_size` | `512` | Training batch size |
| `epochs` | `1024` | Maximum training epochs |
| `learning_rate` | `5.0e-5` | AdamW learning rate |
| `num_blocks` | `16` | Number of FrEIA coupling blocks |
| `obs_dim` | `10` | Observed reco variables predicted as `y` |
| `lack_dim` | `6` | Latent/sampled dimension `z` |
| `num_workers` | `2` | DataLoader worker count |
| `persistent_workers` | `false` | DataLoader worker persistence |
| `pin_memory` | `true` | DataLoader pinned-memory transfer |
| `prefetch_factor` | `2` | DataLoader prefetch factor when workers are enabled |

Active loss weights:

| Loss | Weight | Meaning |
|------|--------|---------|
| `L_x` | `50.0` | MMD loss on reconstructed W/neutrino target variables |
| `L_y` | `10.0` | Huber loss on observed reco variables |
| `L_z` | `10.0` | MMD loss encouraging Gaussian latent behavior |
| `L_pad` | `10.0` | Padding reconstruction consistency |
| `L_pad_noise` | `10.0` | Noisy-padding inverse consistency |
| `L_x_gen` | `0.0` | Monitoring-only generated-target Huber loss |
| `L_W` | `10.0` | MMD loss on reconstructed W masses |
| `L_higgs` | `5.0` | Higgs-mass consistency loss |
| `L_neu_mass` | `0.0` | Neutrino-mass consistency monitor |
| `L_x_huber` | `0.5` | Huber reconstruction loss on W/neutrino targets |

## Data Layout

`load_data.load_data(data_path)` returns two arrays:

| Name in code | Shape | Meaning |
|--------------|-------|---------|
| `train_obj` / `llvv` | `(N, 32)` | Reconstructed observables and conditioning features |
| `target_obj` / `ww` | `(N, 16)` | Raw truth W-boson and neutrino target variables |

Rows containing NaN or infinite values are removed after all categories are
concatenated.

### Reconstructed Feature Tensor

The first `obs_dim = 10` columns are the directly predicted observed variables:

| Columns | Variables |
|---------|-----------|
| `0:4` | positive lepton `px`, `py`, `pz`, `E` |
| `4:8` | negative lepton `px`, `py`, `pz`, `E` |
| `8:10` | MET `px`, `py` |

The remaining `c_dim = 22` columns are conditioning variables:

| Columns | Variables |
|---------|-----------|
| `10:22` | leading three jets, each as `px`, `py`, `pz`, `E` |
| `22:26` | dilepton system `px`, `py`, `pz`, `E` |
| `26:32` | angular features: `deta(l1,l2)`, `dphi(ll,met)`, `dphi(l1,met)`, `dphi(l2,met)`, `dphi(l1,l2)`, `dr(l1,l2)` |

### Target Tensor

The raw target tensor has 16 columns:

| Columns | Variables |
|---------|-----------|
| `0:3` | truth `W+` momentum `px`, `py`, `pz` |
| `3:6` | truth `W-` momentum `px`, `py`, `pz` |
| `6:8` | truth `W+` and `W-` masses |
| `8:11` | truth neutrino momentum `px`, `py`, `pz` |
| `11:14` | truth anti-neutrino momentum `px`, `py`, `pz` |
| `14:16` | massless neutrino placeholders |

The physics block uses the first 8 target variables and the observed lepton
features to convert between W-boson and neutrino representations.
Training passes `X[..., :8]` into the INN while retaining the full 16-column
target scaler for the physics block and physical losses.

### Effective Model Dimensions

With the current defaults:

| Dimension | Value |
|-----------|-------|
| `x_dim` | `8` |
| `inputs_dim` | `32` |
| `y_dim` | `10` |
| `c_dim` | `22` |
| `z_dim` | `6` |
| `internal_dim = y_dim + z_dim` | `16` |
| `input_pad = internal_dim - x_dim` | `8` |

## Model Architecture

The active model in [`model.py`](model.py) combines two pieces:

| Component | Role |
|-----------|------|
| `WtoNeutrinoBlock` | Reversible physics block that maps W-boson variables and lepton observables to neutrino variables, and back |
| FrEIA `GraphINN` | Conditional normalizing flow over the standardized internal representation |

The conditional flow uses an attention-based `CondNet` in [`layers.py`](layers.py).
It embeds the INN half-state plus condition tokens for the three leading jets,
the dilepton system, and angular high-level features.

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

Important runtime behavior:

- If the configured checkpoint directory already exists, `train.py` removes it
  before starting a fresh training run.
- The trainer uses GPU automatically when `torch.cuda.is_available()` is true.
- Checkpointing monitors `val_loss` and keeps only the best checkpoint.
- CSV logs are written under `logs/<project_name>/log/`.

## Notes

- The HDF5 loader concatenates all top-level categories in the input file.
- Jet conditioning uses the leading three jet slots.
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
