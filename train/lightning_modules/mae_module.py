from dataclasses import asdict
import lightning as L
import logging
import math
import torch
import torchvision.utils as vutils

from models.backbones import MaskedAutoencoderViT
from models.strategies.target import HOGNDITarget
from models.strategies.loss import FFTEnhancedLoss
from utils.config import MaskedAutoencoderViT_Cfg
from utils.spatial_ops import unpatchify
from utils.visualization import quant_norm, generate_hog_image, generate_saliency_grid


REFERENCE_BATCH_SIZE = 256

class MAEModule(L.LightningModule):
    def __init__(self, model_cfg: MaskedAutoencoderViT_Cfg,
                 mask_ratio,
                 base_lr,
                 warmup_frac,
                 weight_decay,
                 bands=None):
        super().__init__()

        self.save_hyperparameters()
        self.model_cfg = model_cfg
        self.mask_ratio = mask_ratio
        self.base_lr = base_lr
        self.warmup_frac = warmup_frac
        self.weight_decay = weight_decay
        self.bands = bands

        self.model = MaskedAutoencoderViT(**asdict(self.model_cfg), use_decoder=True)

        self._log_recon_fn = self._log_hog_ndi_reconstructions if isinstance(self.model.target, HOGNDITarget) \
                             else self._log_reconstructions

    def forward(self, x):
        return self.model(x, mask_ratio=self.mask_ratio)

    def training_step(self, batch, batch_idx):
        loss, pred, mask, individual_losses = self.model(batch, mask_ratio=self.mask_ratio)
        
        # log individual losses
        if isinstance(self.model.target, HOGNDITarget): keys = ["train_hog_loss", "train_ndi_loss"]
        elif isinstance(self.model.loss, FFTEnhancedLoss): keys = ["train_mse_loss", "train_amp_loss", "train_phase_loss"]
        else: keys = []
        for k, v in zip(keys, individual_losses): self.log(k, v, on_step=True, on_epoch=True, sync_dist=True)

        # log main loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        if batch_idx == 0:
            self._log_recon_fn({"imgs": batch, "pred": pred, "mask": mask}, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, pred, mask, individual_losses = self.model(batch, mask_ratio=self.mask_ratio, phase="val")
        
        # log individual losses
        if isinstance(self.model.target, HOGNDITarget): keys = ["val_hog_loss", "val_ndi_loss"]
        elif isinstance(self.model.loss, FFTEnhancedLoss): keys = ["val_mse_loss", "val_amp_loss", "val_phase_loss"]
        else: keys = []
        for k, v in zip(keys, individual_losses): self.log(k, v, on_epoch=True, sync_dist=True)

        # log main loss
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        
        if batch_idx == 0:
            self._log_recon_fn({"imgs": batch, "pred": pred, "mask": mask}, "val")

    def on_before_optimizer_step(self, optimizer):
        norms = L.pytorch.utilities.grad_norm(self, norm_type=2)
        self.log_dict(norms)
        
    def configure_optimizers(self):
        effective_batch_size = self.trainer.datamodule.batch_size
        if self.trainer.world_size > 1:
            effective_batch_size *= self.trainer.world_size
        if self.trainer.accumulate_grad_batches > 1:
            effective_batch_size *= self.trainer.accumulate_grad_batches
            
        scaled_lr = self.base_lr * effective_batch_size / REFERENCE_BATCH_SIZE
        logging.info(f"Effective batch size: {effective_batch_size} | Scaled LR: {scaled_lr:.6f}")

        optimizer = torch.optim.AdamW(self.parameters(), lr=scaled_lr, betas=(0.9, 0.95), weight_decay=self.weight_decay)

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

    def _log_reconstructions(self, outputs, split):
        imgs, pred, mask = outputs["imgs"].detach(), outputs["pred"].detach(), outputs["mask"].detach() 
        B, _, H, W = imgs.shape

        raw_imgs = imgs.clone()
        if self.model.normalize:
            imgs = (imgs - self.model.mean) / self.model.std
        imgs = torch.nan_to_num(imgs, nan=0.0)

        # select random samples
        num_samples = min(B, 5)
        indices = torch.randperm(B, generator=torch.Generator().manual_seed(self.current_epoch % 5))[:num_samples]
        imgs, raw_imgs, pred, mask = imgs[indices], raw_imgs[indices], pred[indices], mask[indices]

        p = self.model.patch_embed.patch_size[0] # square patch
        C = self.model.target.out_features // (p ** 2)
        h, w = H // p, W // p

        mask_expanded = mask.reshape(num_samples, h, w).repeat_interleave(p, dim=1).repeat_interleave(p, dim=2).unsqueeze(1) # (num_samples, 1, H, W)
        
        # input (masked image)
        masked_imgs = imgs * (1 - mask_expanded)

        # target
        target = self.model.target(imgs).detach()
        target = unpatchify(target, p, H, W, C)

        # output
        recon_raw = unpatchify(pred, p, H, W, C)
        recon_raw = target * (1 - mask_expanded) + recon_raw * mask_expanded
 
        # output restored
        recon_restored = self.model.target.restore(pred, gt_imgs=imgs)
        if recon_restored is not None:
            recon_restored = unpatchify(recon_restored, p, H, W, C)
            recon_restored = imgs * (1 - mask_expanded) + recon_restored * mask_expanded

        # log
        band_combinations = {"RGB": ["B04", "B03", "B02"], "NIR": ["B07", "B06", "B05"], "SWIR": ["B11"]*3}
        band_indices = {name: [self.bands.index(b) for b in bands] for name, bands
                        in band_combinations.items()} if self.bands is not None else {"":[0,1,2]}

        for name, bands_idx in band_indices.items():
            is_curriculum_masker = hasattr(self.model, "masker") and type(self.model.masker).__name__ == "CurriculumMasker"
            is_weighted_loss = hasattr(self.model, "loss") and type(self.model.loss).__name__ == "SaliencyWeightedLoss"
            
            if is_curriculum_masker or is_weighted_loss:
                extractor = self.model.masker.saliency_extractor if is_curriculum_masker else self.model.loss.saliency_extractor
                saliency = extractor(raw_imgs).detach()
                saliency_grid_tensors = []
                for i in range(num_samples):
                    s_overlay = generate_saliency_grid(saliency[i], image=quant_norm(imgs[i:i+1, bands_idx, :, :])[0].permute(1, 2, 0).cpu().numpy())
                    saliency_grid_tensors.append(torch.from_numpy(s_overlay).float().permute(2, 0, 1) / 255.0)

                rows = [quant_norm(masked_imgs[:, bands_idx, :, :]), # input
                        torch.stack(saliency_grid_tensors).to(imgs.device), # saliency grid
                        quant_norm(imgs[:, bands_idx, :, :])] # unmasked input
            else:
                rows = [quant_norm(masked_imgs[:, bands_idx, :, :]), # input
                        quant_norm(imgs[:, bands_idx, :, :]), # unmasked input
                        quant_norm(target[:, bands_idx, :, :]), # target
                        quant_norm(recon_raw[:, bands_idx, :, :])] # output
            if recon_restored is not None:
                rows.append(quant_norm(recon_restored[:, bands_idx, :, :])) # restored output
            combined = torch.cat(rows, dim=0)
            
            grid = vutils.make_grid(combined, nrow=num_samples, normalize=False)
            tag = f"reconstructions_{name} / {split}"
            self.logger.log_image(key=tag, images=[grid], step=self.global_step)

    def _log_hog_ndi_reconstructions(self, outputs, split):
        imgs, pred, mask = outputs["imgs"].detach(), outputs["pred"].detach(), outputs["mask"].detach()
        B, _, H, W = imgs.shape

        raw_imgs = imgs.clone()
        if getattr(self.model, "normalize", False):
            imgs = (imgs - self.model.mean) / self.model.std
        imgs = torch.nan_to_num(imgs, nan=0.0)

        num_samples = min(B, 5)
        indices = torch.randperm(B, generator=torch.Generator().manual_seed(self.current_epoch % 5))[:num_samples]
        imgs, raw_imgs, pred, mask = imgs[indices], raw_imgs[indices], pred[indices], mask[indices]
        
        p = self.model.patch_embed.patch_size[0]
        h, w = H // p, W // p

        mask_expanded = mask.reshape(num_samples, h, w).repeat_interleave(p, dim=1).repeat_interleave(p, dim=2).unsqueeze(1)  # (num_samples, 1, H, W)
        
        # target
        hog_target, ndi_target = [t.detach() for t in self.model.target(imgs, raw_imgs=raw_imgs)]
        hog_target, ndi_target = self.model.target.restore(torch.cat([hog_target, ndi_target], dim=-1), gt_imgs=raw_imgs)
        ndi_target = unpatchify(ndi_target, p, H, W, HOGNDITarget.NDI_CHANNELS)
        
        pool = self.model.target.pool
        nbins = self.model.target.nbins
        n = p // pool
        hog_target = hog_target.reshape(num_samples, h, w, -1, n, n, nbins).permute(0, 3, 6, 1, 4, 2, 5) # (num_samples, C, nbins, h, n, w, n)
        hog_target = hog_target.reshape(num_samples, -1, nbins, h*n, w*n) # (num_samples, C, nbins, hn, wn)

        # output
        hog_pred, ndi_pred = self.model.target.restore(pred, gt_imgs=raw_imgs)
        ndi_recon = unpatchify(ndi_pred, p, H, W, HOGNDITarget.NDI_CHANNELS)
        ndi_recon = ndi_target * (1 - mask_expanded) + ndi_recon * mask_expanded

        mask_expanded_hog = mask.reshape(num_samples, h, w).repeat_interleave(n, dim=1).repeat_interleave(n, dim=2).unsqueeze(1).unsqueeze(1) # (num_samples, 1, 1, hn, wn)
        hog_recon = hog_pred.reshape(num_samples, h, w, -1, n, n, nbins).permute(0, 3, 6, 1, 4, 2, 5) # (num_samples, C, nbins, h, n, w, n)
        hog_recon = hog_recon.reshape(num_samples, -1, nbins, h*n, w*n) # (num_samples, C, nbins, hn, wn)
        hog_recon = hog_target * (1 - mask_expanded_hog) + hog_recon * mask_expanded_hog

        # log
        true_color_bands = ["B04", "B03", "B02"]
        band_indices = [self.bands.index(b) for b in true_color_bands if b
                        in self.bands] if self.bands is not None else [0, 1, 2]
                
        # NDI
        imgs = quant_norm(imgs[:, band_indices, :, :])
        combined = torch.cat([imgs * (1 - mask_expanded),
                            imgs,
                            quant_norm(ndi_target + 1),
                            quant_norm(ndi_recon)], dim=0)
        grid = vutils.make_grid(combined, nrow=num_samples, normalize=False)
        tag = f"reconstructions_ndi / {split}"
        self.logger.log_image(key=tag, images=[grid], step=self.global_step)
        
        # HOG
        if self.model.target.hog_out_channels == 1: band_indices = [0]
        hog_recon_list, hog_target_list, hog_recon_overlayed_list, scale_factor = [], [], [], 3
        imgs = torch.nn.functional.interpolate(imgs, scale_factor=scale_factor, mode='bilinear', align_corners=False)
        mask_expanded = torch.nn.functional.interpolate(mask_expanded.float(), scale_factor=scale_factor, mode='nearest')
        for i in range(num_samples):
            hog_target_list.append(torch.from_numpy(generate_hog_image(hog_target[i, band_indices], cell_size=pool*scale_factor).transpose(2,0,1)/255).to(imgs.device))
            hog_recon_list.append(torch.from_numpy(generate_hog_image(hog_recon[i, band_indices], cell_size=pool*scale_factor).transpose(2,0,1)/255).to(imgs.device))
            hog_recon_overlayed_list.append(torch.from_numpy(generate_hog_image(hog_recon[i, band_indices], image=imgs[i]).transpose(2,0,1)/255).to(imgs.device))

        hog_recon = torch.stack(hog_recon_list)
        hog_target = torch.stack(hog_target_list)
        #hog_recon = hog_recon * torch.cat([hog_recon, 1 - torch.abs(hog_recon - 0.5)*2, 1 - hog_recon], dim=1)
        #hog_target = hog_target * torch.cat([hog_target, 1 - torch.abs(hog_target - 0.5)*2, 1 - hog_target], dim=1)
        hog_recon_overlayed = torch.stack(hog_recon_overlayed_list)
        combined = torch.cat([imgs * (1 - mask_expanded),
                            hog_target.repeat(1, 3, 1, 1),
                            hog_recon.repeat(1, 3, 1, 1),
                            hog_recon_overlayed], dim=0)
        grid = vutils.make_grid(combined, nrow=num_samples, normalize=False)
        tag = f"reconstructions_hog / {split}"
        self.logger.log_image(key=tag, images=[grid], step=self.global_step)
