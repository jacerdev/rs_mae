import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import hydra
from hydra.utils import instantiate
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.plugins.io import AsyncCheckpointIO
import logging
from omegaconf import DictConfig
from pathlib import Path
import torch
import wandb

from datasets import ReBEN7kDataset, BurnScarsDataset, Sen1Floods11Dataset, C2SegDataset, CashewPlantationDataset
from models.segmentation.segmentation_vit import load_pretrained_encoder_weights
from train.callbacks import TrainingStateCallback
from train.lightning_modules import SegmentationDataModule, SegmentationViTModule
from utils.config import SegmentationViT_Cfg, ReBEN7kDataset_Cfg, BurnScarsDataset_Cfg, Sen1Floods11Dataset_Cfg, C2SegDataset_Cfg, CashewPlantationDataset_Cfg


@hydra.main(version_base="1.3", config_path="../configs", config_name="finetune")
def main(cfg: DictConfig):
    L.seed_everything(cfg.seed, workers=True)
    is_fast_run = cfg.get("fast_run", False)
    
    # Logger
    wandb_logger = WandbLogger(
        project=cfg.wandb.project, 
        name=cfg.wandb.run_name,
        save_dir=cfg.wandb.outdir,
        id=cfg.wandb.run_id,
        resume="allow" if cfg.wandb.run_id else None)

    # Dataset
    train_transform = A.Compose([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=1.0),
                                 ToTensorV2()])
    val_transform = A.Compose([ToTensorV2()])

    match cfg.dataset_name:
        case "reben_7k":
            dataset_class = ReBEN7kDataset
            dataset_class_cfg = ReBEN7kDataset_Cfg
            train_transform = A.Compose([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=1.0),
                                         A.PadIfNeeded(min_height=cfg.img_size, min_width=cfg.img_size,
                                         border_mode=cv2.BORDER_REFLECT), ToTensorV2()])
            val_transform = A.Compose([A.PadIfNeeded(min_height=cfg.img_size, min_width=cfg.img_size,
                                       border_mode=cv2.BORDER_REFLECT), ToTensorV2()])
        case "burn_scars":
            dataset_class = BurnScarsDataset
            dataset_class_cfg = BurnScarsDataset_Cfg

        case "sen1floods11":
            dataset_class = Sen1Floods11Dataset
            dataset_class_cfg = Sen1Floods11Dataset_Cfg

        case "cashew_plantation":
            dataset_class = CashewPlantationDataset
            dataset_class_cfg = CashewPlantationDataset_Cfg

        case name if name.startswith("c2seg"):
            dataset_class = C2SegDataset
            dataset_class_cfg = C2SegDataset_Cfg
            train_transform = A.Compose([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=1.0),
                                         A.RandomCrop(cfg.img_size, cfg.img_size, p=1.0), ToTensorV2()])
        case _:
            raise ValueError(f"Unknown dataset name: {cfg.dataset_name}")

    # Data
    data_module = SegmentationDataModule(
        dataset_class=dataset_class,
        dataset_cfg=dataset_class_cfg(**cfg.dataset),
        train_transform=train_transform,
        val_transform=val_transform,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        prefetch_factor=cfg.prefetch_factor)

    # Model
    model_cfg = SegmentationViT_Cfg(
        **{**dict(cfg.model), "norm_layer": instantiate(cfg.model.norm_layer)}
    )
    model = SegmentationViTModule(
        model_cfg=model_cfg,
        base_lr=cfg.base_lr,
        warmup_frac=cfg.warmup_frac,
        layer_lr_decay=cfg.layer_lr_decay,
        weight_decay=cfg.weight_decay,
        bands=cfg.dataset.bands, dataset_name=cfg.dataset_name)

    # Callbacks
    if dataset_class == C2SegDataset:
        monitor, mode, filename = "val_loss", "min", "seg-{epoch:02d}-{val_loss:.4f}"
    else:
        monitor, mode, filename = "val_iou", "max", "seg-{epoch:02d}-{val_iou:.4f}"
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(monitor=monitor, mode=mode, save_last=True, save_top_k=5,
                                          filename=filename)
    early_stopping = EarlyStopping(monitor=monitor, mode=mode, patience=cfg.early_stop_patience)
    callbacks = [checkpoint_callback, lr_monitor, early_stopping, TrainingStateCallback()]
    if Path("/content/drive/My Drive").exists():
        from train.callbacks import CheckpointSyncCallback
        checkpoints_dir = Path(cfg.wandb.outdir) / cfg.wandb.project
        callbacks.append(CheckpointSyncCallback(checkpoints_dir))

    # Trainer
    trainer = L.Trainer(
        fast_dev_run=is_fast_run,
        max_epochs=cfg.max_epochs,
        accelerator="auto",
        devices="auto",
        strategy="ddp" if torch.cuda.device_count() > 1 else "auto",
        logger=wandb_logger,
        callbacks=callbacks,
        plugins=[AsyncCheckpointIO()],
        precision=cfg.precision,
        log_every_n_steps=cfg.log_every_n_steps,
        gradient_clip_val=cfg.gradient_clip_val,
        accumulate_grad_batches=cfg.accumulate_grad_batches)

    # Checkpoint
    resume_path = None
    if cfg.wandb.run_id:
        logging.info(f"WandB logs resuming from run id {cfg.wandb.run_id} under name {cfg.wandb.run_name}")
        last_ckpt_path = Path(cfg.wandb.outdir) / cfg.wandb.project / cfg.wandb.run_id / "checkpoints" / "last.ckpt"
        if last_ckpt_path.exists():
            logging.info(f"Found last checkpoint at {last_ckpt_path}")
            resume_path = str(last_ckpt_path)
        else:
            logging.info(f"No last checkpoint found at {last_ckpt_path}")

    # Pretrained MAE weights
    elif cfg.ckpt_export_path:
        actual_path = cfg.ckpt_export_path
        if actual_path.endswith(".txt"):
            logging.info(f"Reading checkpoint path from text file: {actual_path}")
            with open(actual_path, "r") as f:
                actual_path = f.read().strip()
                
        load_pretrained_encoder_weights(model.model, actual_path, bands=cfg.dataset.bands)

    else:
        raise ValueError("Training from scratch ?!")

    # Train
    exit_code = 1
    try:
        # re-seed before training to ensure deterministic sample order regardless of model initialization
        L.seed_everything(cfg.seed, workers=True)
        trainer.fit(model, datamodule=data_module, ckpt_path=resume_path)
        exit_code = 0
    finally:
        wandb.finish(exit_code=exit_code)

if __name__ == "__main__":
    import os
    os.environ["WANDB_ASYNC_UPLOADS"] = "true"
    torch.set_float32_matmul_precision('high')
    
    main()
