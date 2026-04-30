from dataclasses import asdict
import lightning as L
import logging
import math
import torch
import torch.nn as nn
import torchvision.utils as vutils
import torchvision.transforms.functional as TF
from torchmetrics.classification import MulticlassAccuracy, JaccardIndex, F1Score

from models.segmentation import SegmentationViT
from utils.config import SegmentationViT_Cfg
from utils.spatial_ops import windowed_inference, windowed_evaluation
from utils.visualization import prepare_segmentation_visuals

REFERENCE_BATCH_SIZE = 256
DECODER_LR_SCALER = 10 # uninitialized_weights_lr = pretrained_weights_lr * DECODER_LR_SCALER

class SegmentationViTModule(L.LightningModule):
    def __init__(self, model_cfg: SegmentationViT_Cfg,
                 base_lr,
                 warmup_frac,
                 layer_lr_decay,
                 weight_decay,
                 bands=None, dataset_name=None):
        super().__init__()

        self.save_hyperparameters()
        self.model_cfg = model_cfg
        self.base_lr = base_lr
        self.warmup_frac = warmup_frac
        self.layer_lr_decay = layer_lr_decay
        self.weight_decay = weight_decay

        self.bands = bands
        self.dataset_name = dataset_name

        self.model = SegmentationViT(**asdict(self.model_cfg))

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        #self.train_acc = MulticlassAccuracy(num_classes=model_cfg.num_classes, ignore_index=-1, average='micro')
        self.val_acc = MulticlassAccuracy(num_classes=model_cfg.num_classes, ignore_index=-1, average='micro')
        
        #self.train_iou = JaccardIndex(task="multiclass", num_classes=model_cfg.num_classes, ignore_index=-1)
        self.val_iou = JaccardIndex(task="multiclass", num_classes=model_cfg.num_classes, ignore_index=-1)
        
        #self.train_f1 = F1Score(task="multiclass", num_classes=model_cfg.num_classes, ignore_index=-1, average='macro')
        self.val_f1 = F1Score(task="multiclass", num_classes=model_cfg.num_classes, ignore_index=-1, average='macro')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        imgs, masks = batch
        preds = self.model(imgs) # (B, num_classes, H, W)
        masks = masks.long()
        
        loss = self.loss_fn(preds, masks)
        #self.train_acc(preds, masks)
        #self.train_iou(preds, masks)
        #self.train_f1(preds, masks)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        #self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        #self.log("train_iou", self.train_iou, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        #self.log("train_f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        
        if batch_idx == 0:
            self._log_predictions({"imgs": imgs, "preds": preds, "masks": masks}, "train")
            
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, masks = batch # imgs could be larger than img_size
        masks = masks.long()
        _, _, H, W = imgs.shape
        input_size = self.model.img_size

        if H <= input_size and W <= input_size:
            preds = self.model(imgs)
            loss = self.loss_fn(preds, masks)
            for torchmetric in [self.val_acc, self.val_iou, self.val_f1]:
                torchmetric.update(preds, masks)
        else:
            preds_list = []
            for sample in imgs:
                pred = windowed_inference(model=self.model, image=sample,
                                            window_size=input_size,
                                            stride=input_size // 2,
                                            num_classes=self.model.num_classes,
                                            batch_size=self.trainer.datamodule.batch_size,
                                            merger_device='cpu')
                preds_list.append(pred)
            preds = torch.stack(preds_list, dim=0) # (B, num_classes, H, W) on CPU
            
            loss = windowed_evaluation(preds=preds, masks=masks, input_size=input_size,
                                   batch_size=self.trainer.datamodule.batch_size,
                                   loss_fn=self.loss_fn,
                                   torchmetrics=[self.val_acc, self.val_iou, self.val_f1])

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_acc", self.val_acc, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_iou", self.val_iou, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_f1", self.val_f1, on_epoch=True, prog_bar=True, sync_dist=True)
        
        if batch_idx == 0:
            self._log_predictions({"imgs": imgs, "preds": preds, "masks": masks}, "val")

    def on_before_optimizer_step(self, optimizer):
        norms = L.pytorch.utilities.grad_norm(self, norm_type=2)
        self.log_dict(norms)
        
    def configure_optimizers(self):
        effective_batch_size = self.trainer.datamodule.batch_size
        if self.trainer.world_size > 1:
            effective_batch_size *= self.trainer.world_size
        if self.trainer.accumulate_grad_batches > 1:
            effective_batch_size *= self.trainer.accumulate_grad_batches
            
        scaled_backbone_lr = self.base_lr * effective_batch_size / REFERENCE_BATCH_SIZE
        scaled_decoder_lr = scaled_backbone_lr * DECODER_LR_SCALER
        logging.info(f"Effective batch size: {effective_batch_size} | Scaled Backbone LR: {scaled_backbone_lr:.6f} | Scaled Decoder LR: {scaled_decoder_lr:.6f}")
        
        # retrieve the custom parameter groups
        param_groups = get_parameter_groups_llrd(self.model, weight_decay=self.weight_decay,
                                backbone_lr=scaled_backbone_lr, decoder_lr=scaled_decoder_lr,
                                layer_lr_decay=self.layer_lr_decay)
        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999))

        # scheduler
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
        #                                         factor=self.reduce_lr_factor, patience=self.reduce_lr_patience, threshold=self.reduce_lr_threshold,
        #                                         min_lr=1e-6)

        # return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss",
        #                                                 "interval": "epoch", "frequency": 1}}

        # scheduler (linear warmup + cosine annealing)
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(self.warmup_frac * total_steps)

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            
            # cosine decay
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def _log_predictions(self, outputs, split):
        imgs, preds, masks = outputs["imgs"].detach(), outputs["preds"].detach(), outputs["masks"].detach()
        
        B, C, H, W = imgs.shape

        if H > 768 or W > 768:
            crop_size = min(768, H, W)
            imgs = TF.center_crop(imgs, output_size=crop_size)
            masks = TF.center_crop(masks, output_size=crop_size)
            preds = TF.center_crop(preds, output_size=crop_size)
            num_samples = 1
        elif H > 256 or W > 256:
            num_samples = min(B, 2)
        else:
            num_samples = min(B, 5)
        
        selected_imgs = imgs[:num_samples]
        selected_masks = masks[:num_samples]
        selected_preds = preds[:num_samples]

        img_vis, gt_mask_vis, pred_mask_vis = prepare_segmentation_visuals(
            selected_imgs, selected_masks, selected_preds, 
            bands=self.bands, dataset_name=self.dataset_name)
        
        rows = [img_vis, gt_mask_vis, pred_mask_vis]
        combined = torch.cat(rows, dim=0)
        
        grid = vutils.make_grid(combined, nrow=num_samples, normalize=False)
        tag = f"predictions / {split}"
        self.logger.log_image(key=tag, images=[grid], step=self.global_step)


def get_parameter_groups_llrd(model, weight_decay, backbone_lr, decoder_lr, layer_lr_decay): # model is a SegmentationViT
    """Creates parameter groups with Layer-wise Learning Rate Decay (LLRD)"""
    param_groups_by_name = {}
    param_groups_by_var = {}

    num_vit_blocks = len(model.encoder.vit.blocks)

    for param_name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # determine if weight decay should be applied
        if len(param.shape) == 1 or param_name.endswith(".bias") or "pos_embed" in param_name or "cls_token" in param_name:
            group_type = "no_decay"
            applied_weight_decay = 0.0
        else:
            group_type = "decay"
            applied_weight_decay = weight_decay

        # layer-wise lr for ViT backbone
        if param_name.startswith("encoder.vit."):
            if param_name.startswith("encoder.vit.patch_embed") or "cls_token" in param_name or "pos_embed" in param_name:
                layer_idx = 0  # stem / embeddings
            elif param_name.startswith("encoder.vit.blocks."):
                block_idx = int(param_name.split("encoder.vit.blocks.")[1].split(".")[0])
                layer_idx = block_idx + 1
            else:
                layer_idx = num_vit_blocks + 1  # final norm layer

            # lr decayed for earlier layers
            lr_scale = layer_lr_decay ** (num_vit_blocks + 1 - layer_idx)
            param_lr = backbone_lr * lr_scale
            group_name = f"{group_type}_backbone_layer_{layer_idx}"

        # adapter layers
        elif param_name.startswith("encoder.res_adapters."):
            param_lr = backbone_lr
            group_name = f"{group_type}_res_adapters"

        # decoder layers
        else:
            param_lr = decoder_lr
            group_name = f"{group_type}_decoder"

        # append parameter group
        if group_name not in param_groups_by_name:
            param_groups_by_name[group_name] = {
                "weight_decay": applied_weight_decay,
                "lr": param_lr,
                "params": [],
            }
            param_groups_by_var[group_name] = {
                "weight_decay": applied_weight_decay,
                "lr": param_lr,
                "params": [],
            }

        param_groups_by_name[group_name]["params"].append(param_name)
        param_groups_by_var[group_name]["params"].append(param)

    for group_name, group_dict in param_groups_by_name.items():
        logging.debug(f"Group: {group_name} | LR: {group_dict['lr']:.2e} | WD: {group_dict['weight_decay']} | Params: {len(group_dict['params'])}")

    return list(param_groups_by_var.values())