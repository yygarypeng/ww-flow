import os
import argparse
import yaml
import numpy as np

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from model import INNLightningModule

from data_module import WBosonDataModule
import load_data as data


# ====== Load config ======
def load_config(config_path="config.yaml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# ===== Main ======
def main(train=True):
	# ---------- load config ----------
	_cfg = load_config()
	_param = _cfg["parameters"]
	BATCH_SIZE = _param["batch_size"]
	EPOCHS = _param["epochs"]
	LEARNING_RATE = _param["learning_rate"]
	LOSS_WEIGHTS = _param["loss_weights"]
	NUM_BLOCKS = _param["num_blocks"]
	OBS_DIM = _param["obs_dim"]
	LACK_DIM = _param["lack_dim"]
	NUM_WORKERS = _param.get("num_workers", 0)
	PERSISTENT_WORKERS = _param.get("persistent_workers", False)
	PIN_MEMORY = _param.get("pin_memory", torch.cuda.is_available())
	PREFETCH_FACTOR = _param.get("prefetch_factor", 2)

	data_path = _cfg["paths"]["data_path"]
	project_name = _cfg["paths"]["project_name"]
	ckpt_path = _cfg["paths"]["ckpt_path"] + project_name

	if train is True:
		if os.path.exists(ckpt_path):
			print(f"Found existing checkpoint at {ckpt_path}, deleting entire folder...")
			os.system(f"rm -rf {ckpt_path}")
		else:
			print("No existing checkpoint found, starting fresh...")
	else:
		print("Evaluation mode, loading checkpoints...")

	torch.set_default_dtype(torch.float32)
	torch.set_float32_matmul_precision("medium") # "high" is more accurate but slower
	llvv, ww = data.load_data(data_path) # llvv, WW
	X = ww.astype(np.float32)
	Y = llvv.astype(np.float32)
	# get transformation info (del later) 
	dm = WBosonDataModule(
		X, Y,
		batch_size=BATCH_SIZE,
		val_frac=0.05,
		test_frac=0.01,
		num_workers=NUM_WORKERS,
		persistent_workers=PERSISTENT_WORKERS,
		pin_memory=PIN_MEMORY,
		prefetch_factor=PREFETCH_FACTOR,
	)
	dm.setup()
	ww_scaler = dm.std_ds.scaler_X
	lvlv_scaler = dm.std_ds.scaler_Y
	ww_dim = X.shape[1]
	inputs_dim = Y.shape[1]
	y_dim = OBS_DIM # dimension of y (observed variables)
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
			loss_weights=LOSS_WEIGHTS,
		)

		ckpt = ModelCheckpoint(
			monitor="val_loss", 
			mode="min", 
			save_top_k=1, # only save the best model
			filename="reg-{epoch:02d}-{val_loss:.2f}"
		)

		early_stopping = EarlyStopping(
			monitor="val_loss",
			patience=128,
			mode="min",
			verbose=False
		)

		csv_logger = CSVLogger(
			save_dir=ckpt_path,
			name="log",
		)

		if args.wandb:
			wandb_logger = WandbLogger(
				project=project_name,
				name="log",
				save_dir=ckpt_path,
				log_model=True,
			)

			wandb_logger.watch(model, log="all", log_freq=500, log_graph=False)

		trainer = pl.Trainer(
			max_epochs=EPOCHS,
			accelerator="gpu" if torch.cuda.is_available() else "cpu",
			devices=1 if torch.cuda.is_available() else None,
			callbacks=[ckpt, early_stopping],
			logger=[csv_logger, wandb_logger] if args.wandb else [csv_logger],
			log_every_n_steps=500,
			enable_progress_bar=True
		)

		trainer.fit(model, datamodule=dm)
		if args.wandb:
			wandb_logger.experiment.finish()
	else:
		print("Loading model from checkpoint for evaluation... return datamodule")
		return dm # use the same random sample for splitting data in datamodule

if __name__ == "__main__":
    from time import time
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--wandb', '-w', action='store_true', help='Enable wandb logging and training mode')
    args = argparser.parse_args()
    
    start = time()
    main()
    end = time()
    print(f"Total time: {end - start:.2f} seconds")
    print("Done!")
