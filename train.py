import os
import glob

import numpy as np

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from model import INNLightningModule

from data_module import WBosonDataModule
import load_data as data


# ====== Hyperparameters ======
BATCH_SIZE = 512
EPOCHS = 512 # maximum epochs (use large enough), have put the early stopping callback
LEARNING_RATE = 1e-5
NUM_BLOCKS = 6
LACK_DIM = 6 # number of lacking constraints
LOSS_WEIGHTS = {
    "L_y": 10.0,
	"L_z": 100.0, 
	"L_x": 200.0, 
	"L_pad": 1.0,
	# complementary losses 
	"L_higgs": 0.1,
	"L_y_mmd": 0.0,
	"L_neu_mass": 0.0
} # weights for different loss components


# ===== Pathes ======
# data_path = "/root/data/mc20_truth_v4_SM.h5"
data_path = "/root/data/danning_h5/ypeng/mc20_qe_v4_recotruth_merged.h5"
project_name = "hww_inn_regressor"
ckpt_path = glob.glob(f"/root/work/ww-flow/logs/{project_name}")
    

# ===== Main ======
def main(train=True):
	if train is True:
		if len(ckpt_path) > 0:
			print(f"Found existing checkpoint at {ckpt_path[0]}, deleting entire folder...")
			os.system(f"rm -rf {ckpt_path[0]}")
		else:
			print("No existing checkpoint found, starting fresh...")
	else:
		print("Evaluation mode, loading checkpoints...")

	torch.set_default_dtype(torch.float32)
	torch.set_float32_matmul_precision("medium") # "high" is more accurate but slower
	llvv, ww = data.load_data(data_path) # llvv, WW
	X = ww.astype(np.float32)
	Y = llvv.astype(np.float32)
	dm = WBosonDataModule(
		X, Y,
		batch_size=BATCH_SIZE,
		val_frac=0.1,
		test_frac=0.1,
	)
	dm.setup()
	ww_scaler = dm.std_ds.scaler_X
	lvlv_scaler = dm.std_ds.scaler_Y
	del dm
	dm = WBosonDataModule(
			X[..., :8], Y[..., :],
			batch_size=BATCH_SIZE,
			val_frac=0.1,
			test_frac=0.1,
		)
	ww_dim = X[..., :8].shape[1] # (N, 8) # 
	#  only access lep+- and met; others are used as cond variables
	inputs_dim = Y[..., :].shape[1]
	y_dim = 10 # dimension of y (observed variables)
	c_dim =inputs_dim - y_dim  # conditioning dimension
	lack_dim = LACK_DIM # number of lacking constraints
	assert lack_dim % 2 == 0, "lack_dim must be even"
	# https://arxiv.org/abs/1808.04730

	if train is True:
		print("Starting training...")
		print(f"x_dim: {ww_dim}, inputs_dim: {inputs_dim}, y_dim: {y_dim}, z_dim: {lack_dim}, c_dim: {c_dim}")
		print(f"internal_dim: {y_dim + lack_dim}")
		print(f"Expected subnet in_dim during forward pass: {(y_dim + lack_dim) // 2} + {c_dim} = {(y_dim + lack_dim) // 2 + c_dim}")
		model = INNLightningModule(
			x_dim=ww_dim, inputs_dim=inputs_dim, 
			internal_dim=y_dim + lack_dim,
			y_dim=y_dim, z_dim=lack_dim, c_dim=c_dim,
			ww_scaler=ww_scaler, lvlv_scaler=lvlv_scaler,
			num_blocks=NUM_BLOCKS, lr=LEARNING_RATE,
			loss_weights=LOSS_WEIGHTS
		)

		ckpt = ModelCheckpoint(
			monitor="val_loss", 
			mode="min", 
			save_top_k=1, # only save the best model
			filename="reg-{epoch:02d}-{val_loss:.2f}"
		)
		early_stopping = EarlyStopping(
			monitor="val_loss",
			patience=32,
			mode="min",
			verbose=False
		)
		trainer = pl.Trainer(
			max_epochs=EPOCHS, 
			accelerator="gpu", 
			devices="auto", 
			callbacks=[ckpt, early_stopping], 
			logger=CSVLogger("logs", name=project_name),
		)
		trainer.fit(model, dm)
	else:
		print("Loading model from checkpoint for evaluation... return datamodule")
		return dm # use the same random sample for splitting data in datamodule

if __name__ == "__main__":
    from time import time
    start = time()
    main()
    end = time()
    print(f"Total time: {end - start:.2f} seconds")
    print("Done!")