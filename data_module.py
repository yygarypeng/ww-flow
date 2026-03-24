import os
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as L
from sklearn.preprocessing import StandardScaler

class ArrayDataset_StdNorm(Dataset):
    def __init__(self, X, Y):
        super().__init__()
        self.scaler_X = StandardScaler()
        self.scaler_Y = StandardScaler()
        
        self.X = torch.as_tensor(self.scaler_X.fit_transform(X), dtype=torch.float32)
        self.Y = torch.as_tensor(self.scaler_Y.fit_transform(Y), dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class WBosonDataModule(L.LightningDataModule):
    def __init__(
        self,
        X,
        Y,
        seed=114,
        batch_size=512,
        val_frac=0.1,
        test_frac=0.1,
        num_workers=0,
        persistent_workers=False,
        pin_memory=True,
        prefetch_factor=2,
    ):
        super().__init__()
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.seed = seed
        # Set reasonable default for num_workers
        if num_workers is None:
            # Use 80% of available cores to prevent system overload
            self.num_workers = max(1, int(os.cpu_count() * 0.8))
            print(f"Setting num_workers to {self.num_workers}")
        else:
            self.num_workers = num_workers
            print(f"Using {self.num_workers} workers in data loading.")
            
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers

    def setup(self, stage=None):
        # the `stage` argument is used to specify which stage of the training process is being set up
        # it'll be used when calling the trainer.fit(), etc.
        ds = ArrayDataset_StdNorm(self.X, self.Y)
        self.std_ds = ds # save the standardized dataset for later use (e.g. inverse transform)
        
        n = len(ds)
        n_val = int(self.val_frac * n)
        n_test = int(self.test_frac * n)
        n_train = n - n_val - n_test
        
        self.train_ds, self.val_ds, self.test_ds = random_split(
            ds, [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(114)
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )

    def test_dataloader(self):
        if self.test_ds is None:
            return None

        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )
