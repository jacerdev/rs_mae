from dataclasses import asdict
import lightning as L
import logging
from torch.utils.data import DataLoader

class SegmentationDataModule(L.LightningDataModule):
    def __init__(self, dataset_class, dataset_cfg,
                 train_transform,
                 val_transform,
                 batch_size,
                 num_workers,
                 prefetch_factor):
        super().__init__()

        self.save_hyperparameters(ignore=['dataset_class'])
        self.dataset_class = dataset_class
        self.dataset_cfg = dataset_cfg
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.train_ds= self.val_ds = self.test_ds = None

    def _create_dataset(self, split: str, transform=None):
        dataset_kwargs = {**asdict(self.dataset_cfg)}
        dataset_kwargs.update(grid_sampling=False) if split != "train" else None
        return self.dataset_class(**dataset_kwargs,
                            split=split, transform=transform)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_ds = self._create_dataset("train", transform=self.train_transform)
            self.val_ds = self._create_dataset("validation", transform=self.val_transform)
            logging.info(f"Train set size: [{len(self.train_ds)}, {tuple(self.train_ds[0][0].shape)}], Val set size: [{len(self.val_ds)}, {tuple(self.val_ds[0][0].shape)}]")
        else:
            raise ValueError(f"Invalid stage: {stage}. Only 'fit' is supported.")

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

    def on_train_epoch_start(self):
        if hasattr(self.train_ds, "shift_grid"):
            self.train_ds.shift_grid(self.trainer.current_epoch)
