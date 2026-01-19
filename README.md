# ww-flow

Flow-based model to reconstruct W boson rest frames using Invertible Neural Networks (INN).

contact: [Yuan-Yen Peng](yuan-yen.peng@cern.ch)

## Overview

This project implements a machine learning method to reconstruct full lepton info in the W boson rest frames in $H \to WW^\ast \to \ell\nu\ell\nu$ using flow-based models. The challenge of reconstruction is the hardly detectable neutrinos. The model learns to map from observed particle momenta (leptons, missing energy, jets) in reco-level MC20 data to the true W boson four-momenta in their rest frames.

## Project Structure

### Core Files

#### [data_module.py](data_module.py)
**Purpose**: PyTorch Lightning data handling and preprocessing

**Key Components**:
- `ArrayDataset_StdNorm`: Custom Dataset class that standardizes input and output data using `StandardScaler`
  - Normalizes X and Y independently
  - Returns normalized tensors for training
  
- `WBosonDataModule`: PyTorch Lightning DataModule for efficient data loading
  - Splits data into train/validation/test sets (80%/10%/10%)
  - Configurable batch size (default: 512)
  - Multi-worker data loading with automatic CPU optimization
  - Stores scalers for later inverse transformation

**How it works**:
1. Receives raw numpy arrays X ($WW^\ast$) and Y ($\ell\ell\nu\nu$)
2. Creates standardized dataset using StandardScaler
3. Automatically splits into train/val/test dataloaders
4. Handles batch creation and memory-efficient loading

---

#### [load_data.py](load_data.py)
**Purpose**: Load and preprocess HDF5 data (organized from QE-v4 sample) files

**Key Functions**:
- `load_particles_from_h5()`: Generic HDF5 loader
  - Recursively reads datasets and attributes from H5 file structure
  - Handles both datasets and groups
  - Returns nested dictionary structure matching file hierarchy

- `load_data()`: Main data preprocessing function
  - Loads particle data (leptons, missing energy, jets) from H5
  - Collects `truth-level' (parton-level) $W$ boson four-vectors from simulation
  - Constructs training features X and target variables Y
  - Returns concatenated numpy arrays across all data categories

**Data Organization**:
(The X and Y arrays are not as typical NN, in INN setting we have input and target swapped)
- Target Y: W boson momenta and masses and additional neutrino scaling variables
- Input features Y: Lepton momenta (px, py, pz, E), missing energy, jet properties

---

#### [layers.py](layers.py)
**Purpose**: Custom neural network building blocks

**Key Components**:
- `DenseDropoutBlock`: Pre-activation dense layer
  - LayerNorm $\to$ SiLU $\to$ Linear $\to$ Dropout
  - Prevents activation saturation through pre-normalization
  
- `ResidualBlock`: Residual connection with two dense blocks
  - Handles dimension changes via projection layer when needed
  - Enables deeper networks without gradient vanishing
  - Two stacked `DenseDropoutBlock`s with residual connection

- `WtoNeutrinoBlock`: Physics-informed layer (partial definition shown)
  - Incorporates physical constraints for $W \to \ell\nu$ decays
  - Takes scaler objects for denormalization

- `SNet, TNet, AffineCouplingBlock, Permutation`: construct invetiable coupling layers
  - Archived, not used in final model
  - Used FrEIA GLOW blocks instead

---

#### [model.py](model.py)
**Purpose**: Main invertible neural network architecture

**Key Classes**:
- `INN`: Core invertible neural network
  - Combines physics block with normalizing flow
  - Uses GLOW coupling blocks from FrEIA library
  - Forward pass: Input $\to$ Physics block $\to$ Flow network $\to$ Output
  - Reverse pass: Sample from latent space $\to$ Reconstruct input
  
  **Architecture**:
  - Input x (interested obs): 8D ($W$ momenta + masses)
  - Pad to internal dimension (y_dim + z_dim)
  - Series of GLOW coupling blocks with random permutations
  - Outputs y (observed obs) and z (latent sample from $\mathcal{N}_\text{z\_dim}(0, I)$) variables

- `INNLightningModule`: PyTorch Lightning wrapper
  - Implements training/validation loops
  - Computes multi-component loss function:
    - `L_y`: L1 loss between predicted and true observables
    - `L_z`: MMD loss matching latent distributions
    - `L_x`: MMD loss for reconstructed inputs
    - `L_pad`: Penalty for non-zero paddings
    - `L_higgs`: Physics loss for Higgs mass
    - `L_neu_mass`: Physics loss for neutrino mass (monitor purpose)
    - `L_y_mmd`: Optional MMD on observables (monitor purpose)
  - Uses Adam optimizer with configurable learning rate

**How it works**:
1. Forward pass (simulation): Encode observed data through physics-aware flow
2. Reverse pass (sampling): Sample from latent space and decode to input space
3. Loss combines reconstruction and distribution matching objectives

---

#### [train.py](train.py)
**Purpose**: Training script with hyperparameter configuration

**Key Configuration**:
- `BATCH_SIZE`: 512
- `EPOCHS`: 512 (with early stopping)
- `LEARNING_RATE`: 1e-5
- `NUM_BLOCKS`: 10 (GLOW coupling blocks)
- `LACK_DIM`: 6 (latent dimension)
- `LOSS_WEIGHTS`: Weights for each loss component

**Main Workflow**:
1. Load data from HDF5 file
2. Extract input ($WW^\ast$) and target ($\ell\ell\nu\nu$) arrays
3. Initialize data module and compute scalers
4. Create INN model with computed dimensions
5. Setup PyTorch Lightning trainer with:
   - GPU acceleration
   - Model checkpointing (saves best validation model)
   - Early stopping (patience=32 epochs)
   - CSV logging
6. Train model on training set, validate on validation set
7. Return data module for later evaluation

---

#### [utilities.py](utilities.py)
**Purpose**: Loss functions and utility functions

**Key Loss Functions**:
- `mmd_loss()`: Maximum Mean Discrepancy loss
  - Measures distribution difference between two sets
  - Uses multiple RBF kernels with adaptive bandwidth
  - Bandwidth automatically scaled by median distance
  - Returns MMD$^2$ estimate
  - Used to match latent distributions between predicted and true

- `higgs_loss()`: Physics-informed loss for Higgs mass reconstruction
  - Encourages correct Higgs mass (125 GeV) reconstruction

- `neu_mass_loss()`: Physics-informed loss for neutrino mass
  - Constrains neutrino mass reconstruction
  - Uses scaler objects for proper scale

**Note**: These losses are weighted components of the total training loss

---

#### [ohbboosting.py](ohbboosting.py)
**Purpose**: Lorentz boost operations (boost to orthonormal helicity basis) for kinematic reconstruction

**Key Class**:
- `Booster`: Implements relativistic kinematics
  - `_boost_to_rest_frame()`: Boost 4-vectors to rest frame
  - `_construct_basis()`: Create orthogonal basis in Higgs rest frame
  - `_map_to_basis()`: Project momenta onto basis vectors
  - Uses ROOT TLorentzVector for physics calculations
  - Enables multi-processing for batch operations

**How it works**:
- Boosts particle momenta from lab frame to $W$/$H$ rest frames
- Constructs physics-inspired coordinate basis
- Maps reconstructed momenta to meaningful kinematic variables

---

#### [visualize.ipynb](visualize.ipynb)
**Purpose**: Jupyter notebook for analyzing and visualizing results

**Typical contents**:
- Load trained model and test data
- Generate predictions on test set
- Plot reconstructed vs true kinematic variables
- Analyze mass distributions and correlations
- Diagnostic plots for training convergence

---

### Additional Files
- `record`: Training record/log file
- `logs/`: Directory storing training logs and checkpoints

## Workflow

### Training
```bash
python train.py  # Trains INN model from scratch
```

1. Loads data from HDF5 file
2. Preprocesses with standard scaling
3. Splits into train/val/test
4. Trains INN model with multi-loss objective
5. Saves best checkpoint based on validation loss
6. Uses early stopping if no improvement

### Inference
```python
from train import main
from model import INNLightningModule

# Load trained model
model = INNLightningModule.load_from_checkpoint("checkpoint.ckpt")

# Sample from latent space
z_sample = torch.randn(batch_size, z_dim)
cond = conditioning_variables
reconstructed_input, _ = model.forward(z_sample, cond, reverse=True)
```

## Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `x_dim` | 8 | Input dimension ($W$ momenta + masses + neutrinos (scaling purpose, not engaged in training)) |
| `y_dim` | 10 | Observables $WW^\ast$ decayed products (leptons, missing energy) |
| `z_dim` | 6 | Latent dimension (under-constrained DoF $\leftarrow$ neutrinos) |
| `c_dim` | Variable | Conditioning dimension (jets) |
| `input_dim` | Variable | Input dimension (leptons, missing energy, jets) |
| `internal_dim` | Variable | the dimension of the internal layers |
| `num_blocks` | 10 | Number of GLOW coupling blocks |
| `batch_size` | 512 | Training batch size |
| `learning_rate` | 1e-5 | Adam optimizer learning rate |

## Dependencies

- PyTorch
- PyTorch Lightning
- FrEIA (Flow-based Invertible modules)
- scikit-learn (StandardScaler)
- numpy, h5py
- ROOT (for physics calculations)

## References

- GLOW coupling blocks: [Glow: Generative Flow using Invertible 1x1 Convolutions](https://arxiv.org/abs/1805.07722)
- Fundamental INN structure: [Analyzing Inverse Problems with Invertible Neural Networks](https://arxiv.org/abs/1808.04730v3)
- MMD loss: [A Kernel Two-Sample Test](https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf)
- Boosted basis: [Testing Bell inequalities in Higgs boson decays](https://arxiv.org/abs/2106.01377)
