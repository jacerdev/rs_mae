import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from train.callbacks import TRAINING_STATE
from models.strategies.loss.base import ReconstructionLoss
from utils.features import DifferentiableGLCM




class MaskedGLCMLoss(nn.Module, ReconstructionLoss):
    """Computes a combined GLCM and MSE loss for Masked Autoencoders over grouped tokens"""
    def __init__(self, in_chans, patch_size, group_tokens=3, stride_tokens=2,
                 bins=32, min_val=-3.0, max_val=3.0, bandwidth=None, chunk_size=-1,
                 alpha=1.0, warmup_start_epoch=0, warmup_epochs=0, **kwargs):
        nn.Module.__init__(self)
        ReconstructionLoss.__init__(self, **kwargs)
        
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.group_tokens = group_tokens
        self.stride_tokens = stride_tokens
        self.chunk_size = chunk_size
        self.alpha = alpha
        self.warmup_start_epoch = warmup_start_epoch
        self.warmup_epochs = warmup_epochs
        
        self.glcm = DifferentiableGLCM(bins=bins, min_val=min_val, max_val=max_val, bandwidth=bandwidth)
        
    def __call__(self, pred, target, mask, phase="", **kwargs):
        B, L, D = pred.shape
        
        loss_mse_full = (pred - target) ** 2
        loss_mse_full = loss_mse_full.mean(dim=-1)
        loss_mse = (loss_mse_full * mask).sum() / mask.sum().clamp(min=1)
        
        # cosine warmup scaling
        if phase == "validation":
            scale = 1.0 
        else:
            epoch = TRAINING_STATE.epoch
            if epoch < self.warmup_start_epoch:
                scale = 0.0
            else:
                if self.warmup_epochs <= 0:
                    scale = 1.0
                else:
                    progress = min(1.0, (epoch - self.warmup_start_epoch) / self.warmup_epochs)
                    scale = (1.0 - math.cos(progress * math.pi)) / 2.0
                    
        effective_alpha = self.alpha * scale
        
        if effective_alpha == 0:
            return loss_mse, {"mse_loss": loss_mse.detach(), "glcm_loss": torch.tensor(0.0, device=pred.device)}
            
        S = int(math.sqrt(L))
        p = self.patch_size
        C = self.in_chans
        
        pred_img = pred.view(B, S, S, C, p, p).permute(0, 3, 1, 4, 2, 5).reshape(B, C, S * p, S * p)
        target_img = target.view(B, S, S, C, p, p).permute(0, 3, 1, 4, 2, 5).reshape(B, C, S * p, S * p)
        
        mask_grid = mask.view(B, 1, S, S).float()
        
        block_mask = F.max_pool2d(mask_grid, kernel_size=self.group_tokens, stride=self.stride_tokens) > 0
        block_mask_flat = block_mask.view(-1)
        
        gp = p * self.group_tokens
        pixel_stride = p * self.stride_tokens

        pred_blocks = F.unfold(pred_img,   kernel_size=gp, stride=pixel_stride).transpose(1, 2).reshape(-1, C * gp * gp)
        target_blocks = F.unfold(target_img, kernel_size=gp, stride=pixel_stride).transpose(1, 2).reshape(-1, C * gp * gp)
        
        pred_m = pred_blocks[block_mask_flat]
        target_m = target_blocks[block_mask_flat]
        
        M = pred_m.shape[0]
        if M == 0:
            return loss_mse, {"mse_loss": loss_mse.detach(), "glcm_loss": torch.tensor(0.0, device=pred.device)}
        
        # Fold channels into the batch dim: (M*C, gp, gp)
        pred_spatial   = pred_m.view(M * C, gp, gp)
        target_spatial = target_m.view(M * C, gp, gp)

        pred_spatial   = torch.clamp(pred_spatial,   self.glcm.min_val, self.glcm.max_val)
        target_spatial = torch.clamp(target_spatial, self.glcm.min_val, self.glcm.max_val)

        H_pred = self.glcm(pred_spatial,   chunk_size=self.chunk_size)  # (M*C, K, K)
        H_targ = self.glcm(target_spatial, chunk_size=self.chunk_size)  # (M*C, K, K)

        loss_glcm = F.mse_loss(H_pred, H_targ, reduction='mean')
        
        total_loss = loss_mse + effective_alpha * loss_glcm
        
        loss_components = {
            "mse_loss":  loss_mse.detach(),
            "glcm_loss": loss_glcm.detach(),
        }
        
        return total_loss, loss_components

    def get_texture_error_map(self, pred_img, target_img):
        """computes the spatial GLCM MSE error map between pred and target for non-overlapping windows."""
        B, C, H, W = pred_img.shape
        gp = self.group_tokens * self.patch_size

        num_windows_h = H // gp
        num_windows_w = W // gp

        pred_blocks   = pred_img.unfold(2, gp, gp).unfold(3, gp, gp)   # (B, C, nH, nW, gp, gp)
        target_blocks = target_img.unfold(2, gp, gp).unfold(3, gp, gp)

        # flatten windows and fold channels into batch: (B*N*C, gp, gp)
        # unfold gives (B, C, nH, nW, gp, gp) -> permute to (B, nH, nW, C, gp, gp) so
        # the flat index order is [b, window, channel], matching the view(B, N, C) below
        N = num_windows_h * num_windows_w
        pred_blocks   = pred_blocks.permute(0, 2, 3, 1, 4, 5).contiguous().view(B * N * C, gp, gp)
        target_blocks = target_blocks.permute(0, 2, 3, 1, 4, 5).contiguous().view(B * N * C, gp, gp)

        pred_blocks   = torch.clamp(pred_blocks,   self.glcm.min_val, self.glcm.max_val)
        target_blocks = torch.clamp(target_blocks, self.glcm.min_val, self.glcm.max_val)

        H_pred = self.glcm(pred_blocks,   chunk_size=self.chunk_size)  # (B*N*C, K, K)
        H_targ = self.glcm(target_blocks, chunk_size=self.chunk_size)

        # MSE per window, averaged over channels: (B*N*C, K, K) -> (B, N)
        mse_per_patch = F.mse_loss(H_pred, H_targ, reduction='none').mean(dim=(1, 2))  # (B*N*C,)
        mse_per_patch = mse_per_patch.view(B, N, C).mean(dim=2)                        # (B, N) mean over C
        mse_grid = mse_per_patch.view(B, num_windows_h, num_windows_w)

        # interpolate up to full image size
        error_map = F.interpolate(mse_grid.unsqueeze(1), size=(H, W), mode='nearest').squeeze(1)

        return error_map
