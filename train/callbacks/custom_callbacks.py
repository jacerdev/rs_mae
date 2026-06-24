import lightning as L
import logging
from pathlib import Path
import subprocess


class TrainingState:
    _instance = None
    epoch = 0
    def __new__(cls):
        if not cls._instance:
            cls._instance = super(TrainingState, cls).__new__(cls)
        return cls._instance

TRAINING_STATE = TrainingState()

class TrainingStateCallback(L.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        TRAINING_STATE.epoch = trainer.current_epoch


class CheckpointSyncCallback(L.Callback):
    def __init__(self, checkpoints_dir, sync_root="/content/drive/My Drive/saved_checkpoints/"):
        self.checkpoints_dir = Path(checkpoints_dir)
        self.dest_dir = Path(sync_root) / self.checkpoints_dir.name

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        self.dest_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            "rsync", "-av", "--include=*/", "--include=*.ckpt", "--exclude=*",
            f"{self.checkpoints_dir}/", f"{self.dest_dir}/"
        ], check=False)

class DebuggingCallback(L.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        logging.debug(f"Starting training epoch {trainer.current_epoch}...")

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if batch_idx == 0:
            logging.debug(f"Starting batch 0 of training epoch {trainer.current_epoch}...")

    def on_train_epoch_end(self, trainer, pl_module):
        logging.debug(f"Finished training epoch {trainer.current_epoch}.")

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        logging.debug(f"Saving checkpoint at epoch {trainer.current_epoch}...")

    def on_validation_epoch_start(self, trainer, pl_module):
        logging.debug(f"Starting validation epoch {trainer.current_epoch}...")

    def on_validation_epoch_end(self, trainer, pl_module):
        logging.debug(f"Finished validation epoch {trainer.current_epoch}.")
