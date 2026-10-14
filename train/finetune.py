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
import re
import torch
import wandb

from datasets import ReBEN7kDataset, C2SegDataset, CashewPlantationDataset, MMHunanDataset
from models.segmentation.segmentation_vit import load_pretrained_encoder_weights
from test.benchmark import run_benchmark
from train.callbacks import TrainingStateCallback
from train.lightning_modules import SegmentationDataModule, SegmentationViTModule
from utils.config import SegmentationViT_Cfg, ReBEN7kDataset_Cfg, C2SegDataset_Cfg, CashewPlantationDataset_Cfg, MMHunanDataset_Cfg
from utils.utils import log_run_to_json, get_checkpoints_dir, find_best_ckpt


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
    
    run_id = wandb_logger.experiment.id
    if isinstance(run_id, str):
        log_run_to_json(
            experiment_name=cfg.experiment_name,
            seed=cfg.seed,
            dataset_name=cfg.dataset_name,
            run_name=cfg.wandb.run_name, # or wandb_logger.experiment.name
            run_id=run_id,
            json_path=cfg.runs_json if not cfg.get("is_dummy", False) else "artifacts/runs_dummy.json"
        )

    # Dataset
    train_transform = A.Compose([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=1.0),
                                 ToTensorV2()], seed=cfg.seed)
    val_transform = A.Compose([ToTensorV2()], seed=cfg.seed)

    match cfg.dataset_name:
        case "reben_7k":
            dataset_class = ReBEN7kDataset
            dataset_class_cfg = ReBEN7kDataset_Cfg
            train_transform = A.Compose([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=1.0),
                                         A.PadIfNeeded(min_height=cfg.img_size, min_width=cfg.img_size,
                                         border_mode=cv2.BORDER_REFLECT), ToTensorV2()], seed=cfg.seed)
            val_transform = A.Compose([A.PadIfNeeded(min_height=cfg.img_size, min_width=cfg.img_size,
                                       border_mode=cv2.BORDER_REFLECT), ToTensorV2()], seed=cfg.seed)
        case "cashew_plantation":
            dataset_class = CashewPlantationDataset
            dataset_class_cfg = CashewPlantationDataset_Cfg

        case "mmhunan":
            dataset_class = MMHunanDataset
            dataset_class_cfg = MMHunanDataset_Cfg

        case name if name.startswith("c2seg"):
            dataset_class = C2SegDataset
            dataset_class_cfg = C2SegDataset_Cfg
            train_transform = A.Compose([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=1.0),
                                         A.RandomCrop(cfg.img_size, cfg.img_size, p=1.0), ToTensorV2()], seed=cfg.seed)
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
    lr_monitor = LearningRateMonitor(logging_interval='step')
    # Tracks the peaks (val_iou)
    best_checkpoint_callback = ModelCheckpoint(monitor="val_iou", mode="max", save_last=False, save_top_k=1,
                                          filename="seg-{epoch:02d}-{val_iou:.4f}")
    # Tracks the true fully annealed end state
    annealed_checkpoint_callback = ModelCheckpoint(monitor="epoch", mode="max", save_last=False, save_top_k=1,
                                                   filename="fully_annealed")
                                                   
    early_stopping = EarlyStopping(monitor="val_iou", mode="max", patience=cfg.early_stop_patience)
    callbacks = [best_checkpoint_callback, annealed_checkpoint_callback, lr_monitor, early_stopping, TrainingStateCallback()]
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
    else:
        pretrained_exp_key = f"{cfg.experiment_name}-{cfg.seed}"
        logging.info(f"Fetching pretrained checkpoint for experiment: {pretrained_exp_key}")
        
        try:
            ckpt_dir = get_checkpoints_dir(pretrained_exp_key, "reben", cfg.wandb.outdir, wandb_project=cfg.wandb.project, json_path=cfg.runs_json if not cfg.get("is_dummy", False) else "artifacts/runs_dummy.json")
            actual_path = find_best_ckpt(ckpt_dir)
            
            if not actual_path:
                raise FileNotFoundError(f"Could not find a valid checkpoint in {ckpt_dir}")
                
            logging.info(f"Found pretrained checkpoint at: {actual_path}")
            load_pretrained_encoder_weights(model.model, str(actual_path), bands=cfg.dataset.bands, pretrained_model_bands=cfg.pretrained_model_bands)
            
        except Exception as e:
            logging.error(f"Failed to load pretrained weights from runs.json: {e}")
            raise e

    # Train
    exit_code = 1
    try:
        # re-seed before training to ensure deterministic sample order regardless of model initialization
        L.seed_everything(cfg.seed, workers=True)
        trainer.fit(model, datamodule=data_module, ckpt_path=resume_path)
        
        # Benchmarking Peak Checkpoint
        best_ckpt_path = best_checkpoint_callback.best_model_path
        logging.info(f"Benchmarking BEST checkpoint: {best_ckpt_path}")
        results = run_benchmark(cfg, ckpt_path=best_ckpt_path, results_dir=f"results/{cfg.sensor}/results_peak")
        table = wandb.Table(columns=["Metric", "Value"])
        match = re.search(r"epoch=(\d+)", best_ckpt_path)
        best_epoch = int(match.group(1)) if match else "unknown"
        table.add_data("Epoch", best_epoch)
        for k, v in results.items(): table.add_data(k, v)
        wandb.log({"test/results_peak": table})

        # Benchmarking Fully Annealed Checkpoint
        annealed_ckpt_path = annealed_checkpoint_callback.best_model_path
        logging.info(f"Benchmarking FULLY ANNEALED checkpoint: {annealed_ckpt_path}")
        results_annealed = run_benchmark(cfg, ckpt_path=annealed_ckpt_path, results_dir=f"results/{cfg.sensor}/results_annealed")
        table_annealed = wandb.Table(columns=["Metric", "Value"])
        table_annealed.add_data("Epoch", trainer.current_epoch - 1)
        for k, v in results_annealed.items(): table_annealed.add_data(k, v)
        wandb.log({"test/results_annealed": table_annealed})

        exit_code = 0
    finally:
        wandb.finish(exit_code=exit_code)

if __name__ == "__main__":
    import os
    os.environ["WANDB_ASYNC_UPLOADS"] = "true"
    torch.set_float32_matmul_precision('high')
    
    main()
