import albumentations as A
from albumentations.pytorch import ToTensorV2
import hydra
from hydra.utils import instantiate
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.plugins.io import AsyncCheckpointIO
import logging
from omegaconf import DictConfig
from pathlib import Path
import torch
import wandb

from train.callbacks import TrainingStateCallback
from train.lightning_modules import ReBENDataModule, MAEModule
from utils.config import ReBENDataset_Cfg, MaskedAutoencoderViT_Cfg
from utils.utils import log_run_to_json


@hydra.main(version_base="1.3", config_path="../configs", config_name="pretrain")
def main(cfg: DictConfig):
    L.seed_everything(cfg.seed, workers=True)
    
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
            dataset_name="reben",
            run_name=cfg.wandb.run_name, # or wandb_logger.experiment.name
            run_id=run_id,
            json_path=cfg.runs_json if not cfg.get("is_dummy", False) else "artifacts/runs_dummy.json"
        )

    # Experiment
    masker_kwargs, target_kwargs, loss_kwargs = {}, {}, {}
    
    match cfg.experiment_name:
        case "baseline" | "baseline_200ep" | "baseline_full_200ep":
            pass
        case "feat_guided" | "hog_target":
            pass
        case "curriculum_ndi" | "curriculum_hog" | "curriculum_ndi_sigmae" | "curriculum_feat_unif":
            masker_kwargs = {"saliency_extractor": instantiate(cfg.saliency_extractor)}
        case "fourier_loss":
            pass
        case "hog_weighted":
            loss_kwargs = {"saliency_extractor": instantiate(cfg.saliency_extractor)}
        case "latent_target":
            target_kwargs = {"embed_dim": cfg.model.embed_dim}
        case "segmentation_target":
            cfg.dataset["return_mask"] = "clc"
        case "glcm_loss":
            pass
        case _:
            raise ValueError("Experiment not recognized.")

    masker = instantiate(cfg.masker, **masker_kwargs)
    target = instantiate(cfg.target, **target_kwargs)
    loss = instantiate(cfg.loss, **loss_kwargs)

    # Data
    train_transform = A.Compose([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=1.0),
                                 A.RandomCrop(cfg.img_size, cfg.img_size, p=1.0), ToTensorV2()], seed=cfg.seed)
    val_transform = A.Compose([A.CenterCrop(cfg.img_size, cfg.img_size), ToTensorV2()], seed=cfg.seed)
    
    data_module = ReBENDataModule(
        dataset_cfg=ReBENDataset_Cfg(**cfg.dataset),
        train_transform=train_transform,
        val_transform=val_transform,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        prefetch_factor=cfg.prefetch_factor)

    # Model
    model_cfg = MaskedAutoencoderViT_Cfg(
        masker=masker,
        target=target,
        loss=loss,
        **{**dict(cfg.model), "norm_layer": instantiate(cfg.model.norm_layer)}
    )
    model = MAEModule(
        model_cfg=model_cfg,
        mask_ratio=cfg.mask_ratio,
        base_lr=cfg.base_lr,
        warmup_frac=cfg.warmup_frac,
        weight_decay=cfg.weight_decay,
        bands=cfg.bands)

    # Callbacks
    # checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min", save_last=True, save_top_k=1,
    checkpoint_callback = ModelCheckpoint(monitor="epoch", mode="max", save_last=True, save_top_k=1,
                                        filename="mae-{epoch:02d}-{val_loss:.4f}")
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Trainer
    trainer = L.Trainer(
        fast_dev_run=cfg.get("fast_run", False),
        max_epochs=cfg.max_epochs,
        accelerator="auto",
        devices="auto",
        strategy="ddp" if torch.cuda.device_count() > 1 else "auto",
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor, TrainingStateCallback()],
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
    elif cfg.resume_ckpt_path:
        logging.info(f"Resuming Lightning training from checkpoint: {cfg.resume_ckpt_path}")
        resume_path = cfg.resume_ckpt_path
    
    # Train
    exit_code = 1
    try:
        # XXX initialize with a pretrained MAE checkpoint XXX
        init_ckpt_path = cfg.paths.get("init_ckpt_path", None)
        if init_ckpt_path and not resume_path:
            from models.backbones.mae_vit import load_pretrained_mae_weights
            logging.warning(f"=== LOADING weights from {init_ckpt_path} ===")
            load_pretrained_mae_weights(model.model, init_ckpt_path, cfg.bands, cfg.bands)
        # XXX

        # re-seed before training to ensure deterministic sample order regardless of model initialization
        L.seed_everything(cfg.seed, workers=True)
        trainer.fit(model, datamodule=data_module, ckpt_path=resume_path)
            
        # Export the best checkpoint path
        best_model_path = checkpoint_callback.best_model_path
        logging.info(f"Best model path: {best_model_path}")
        exit_code = 0
    finally:
        wandb.finish(exit_code=exit_code)

if __name__ == "__main__":
    import os
    os.environ["WANDB_ASYNC_UPLOADS"] = "true"
    torch.set_float32_matmul_precision('high')
    
    main()
