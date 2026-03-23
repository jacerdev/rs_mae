from dataclasses import asdict
import lightning as L
import logging
from torch.utils.data import DataLoader

from datasets import ReBENDataset
from utils.config import ReBENDataset_Cfg


class ReBENDataModule(L.LightningDataModule):
    def __init__(self, dataset_cfg: ReBENDataset_Cfg,
                 train_transform,
                 val_transform,
                 batch_size,
                 num_workers,
                 prefetch_factor):
        super().__init__()

        self.save_hyperparameters()
        self.dataset_cfg = dataset_cfg
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.train_ds= self.val_ds = None

    def setup(self, stage=None):
        def _create_dataset(split: str, transform=None):
            return ReBENDataset(**{**asdict(self.dataset_cfg)},
                                split=split, transform=transform, return_mask=False)
    
        if stage == "fit" or stage is None:
            self.train_ds = _create_dataset("train", transform=self.train_transform)
            self.val_ds = _create_dataset("validation", transform=self.val_transform)
            logging.info(f"Train set length: {len(self.train_ds)}, Val set length: {len(self.val_ds)}")

    def _create_dataloader(self, dataset, shuffle):        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.num_workers > 0,
            pin_memory=True)

    def train_dataloader(self): return self._create_dataloader(self.train_ds, shuffle=True)
    def val_dataloader(self): return self._create_dataloader(self.val_ds, shuffle=False)
