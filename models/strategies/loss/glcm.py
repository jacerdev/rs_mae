import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from train.callbacks import TRAINING_STATE
from models.strategies.loss.base import ReconstructionLoss


class DifferentiableGLCM(nn.Module):
    """
    Differentiable Gray Level Co-occurrence Matrix (GLCM) using Kernel Density Estimation.
    Computes Omni-directional (Horizontal, Vertical, Diagonals) and Symmetric GLCM.
    """
    def __init__(self, bins=32, min_val=-3.0, max_val=3.0, bandwidth=None):
        super().__init__()
        self.bins = bins
        self.min_val = min_val
        self.max_val = max_val
        
        self.L = (max_val - min_val) / bins
        self.W = bandwidth if bandwidth is not None else self.L / 2.0
        
        centers = torch.linspace(min_val + self.L / 2, max_val - self.L / 2, bins)
        self.register_buffer("mu", centers.view(1, -1, 1)) # (1, K, 1)

    def _get_probabilities(self, I):
        term1 = torch.sigmoid((I - self.mu + self.L / 2) / self.W)
        term2 = torch.sigmoid((I - self.mu - self.L / 2) / self.W)
        return term1 - term2

    def compute_histogram(self, I1, I2):
        B, N = I1.shape
        if N == 0:
            return torch.zeros((B, self.bins, self.bins), device=I1.device)
            
        I1_exp, I2_exp = I1.unsqueeze(1), I2.unsqueeze(1)
        P1, P2 = self._get_probabilities(I1_exp), self._get_probabilities(I2_exp)
        
        H = torch.bmm(P1, P2.transpose(1, 2)) / N
        H = H / (H.sum(dim=(1, 2), keepdim=True) + 1e-8)
        return H

    def forward(self, x, chunk_size=512):
        """
        x: (B, H, W) spatial patches
        Returns: Single Omni-directional, Symmetric GLCM of shape (B, K, K)
        """
        B, H_dim, W_dim = x.shape
        
        if str(chunk_size).lower() == 'auto' or chunk_size == -1:
            if x.device.type == 'cuda':
                free_memory, _ = torch.cuda.mem_get_info(x.device)
                # memory overhead calculation (we reserve a safe 80% of available VRAM to determine the max chunk size)
                dynamic_chunk_size = int((free_memory * 0.8) / (162 * 1024))
                chunk_size = max(1024, min(B, dynamic_chunk_size))
            else:
                chunk_size = 32768
                
        if isinstance(chunk_size, (int, float)) and chunk_size > 0 and B > chunk_size:
            H_list = []
            for i in range(0, B, int(chunk_size)):
                chunk = x[i:i+int(chunk_size)]
                if chunk.requires_grad:
                    h = checkpoint(self._forward_single_chunk, chunk, use_reentrant=False)
                else:
                    h = self._forward_single_chunk(chunk)
                H_list.append(h)
            return torch.cat(H_list, dim=0)
        else:
            return self._forward_single_chunk(x)

    def _forward_single_chunk(self, x):
        B, H, W = x.shape
        K = self.bins
        
        # Compute probabilities for the enitre patch once
        x_flat = x.view(B, -1).unsqueeze(1) # (B, 1, H*W)
        P_flat = self._get_probabilities(x_flat) # (B, K, H*W)
        P = P_flat.view(B, K, H, W)
        
        # Slice the probability map for the 4 directions and compute histograms
        # horizontal
        P1_h = P[:, :, :, :-1].reshape(B, K, -1)
        P2_h = P[:, :, :, 1:].reshape(B, K, -1)
        H_h = torch.bmm(P1_h, P2_h.transpose(1, 2)) / max(1, P1_h.shape[-1])
        
        # verticl
        P1_v = P[:, :, :-1, :].reshape(B, K, -1)
        P2_v = P[:, :, 1:, :].reshape(B, K, -1)
        H_v = torch.bmm(P1_v, P2_v.transpose(1, 2)) / max(1, P1_v.shape[-1])
        
        # diagonal
        P1_d1 = P[:, :, :-1, :-1].reshape(B, K, -1)
        P2_d1 = P[:, :, 1:, 1:].reshape(B, K, -1)
        H_d1 = torch.bmm(P1_d1, P2_d1.transpose(1, 2)) / max(1, P1_d1.shape[-1])
        
        # diagonal 2
        P1_d2 = P[:, :, 1:, :-1].reshape(B, K, -1)
        P2_d2 = P[:, :, :-1, 1:].reshape(B, K, -1)
        H_d2 = torch.bmm(P1_d2, P2_d2.transpose(1, 2)) / max(1, P1_d2.shape[-1])
        
        # pool all directional pairs together
        H_total = H_h + H_v + H_d1 + H_d2
        
        # add transpose to enforce symmetry (counts reverse directions)
        H_sym = (H_total + H_total.transpose(1, 2)) / 2.0
        
        # normalize back to 1.0 probability distribution
        H_sym = H_sym / (H_sym.sum(dim=(1, 2), keepdim=True) + 1e-8)
        
        return H_sym

class MaskedGLCMLoss(nn.Module, ReconstructionLoss):
    """
    Computes a combined GLCM and MSE loss for Masked Autoencoders over grouped tokens.
    """
    def __init__(self, 
                 in_chans: int,
                 patch_size: int,
                 group_tokens: int = 3,  # Adjusted default
                 stride_tokens: int = 2, # Controls window overlap
                 bins: int = 32,
                 min_val: float = -3.0,
                 max_val: float = 3.0,
                 bandwidth: float = None,
                 chunk_size = 'auto',
                 alpha: float = 1.0,
                 warmup_start_epoch: int = 0,
                 warmup_epochs: int = 0,
                 **kwargs):
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
        
        # Stride logic injected here for the token mask
        block_mask = F.max_pool2d(mask_grid, kernel_size=self.group_tokens, stride=self.stride_tokens) > 0 
        block_mask_flat = block_mask.view(-1)
        
        gp = p * self.group_tokens 
        pixel_stride = p * self.stride_tokens # Pixel stride matches token stride
        
        # Stride logic injected here for the pixel windows
        pred_blocks = F.unfold(pred_img, kernel_size=gp, stride=pixel_stride).transpose(1, 2).reshape(-1, C * gp * gp)
        target_blocks = F.unfold(target_img, kernel_size=gp, stride=pixel_stride).transpose(1, 2).reshape(-1, C * gp * gp)
        
        pred_m = pred_blocks[block_mask_flat]
        target_m = target_blocks[block_mask_flat]
        
        M = pred_m.shape[0]
        if M == 0:
            return loss_mse, {"mse_loss": loss_mse.detach(), "glcm_loss": torch.tensor(0.0, device=pred.device)}
            
        pred_spatial = pred_m.view(M * C, gp, gp)
        target_spatial = target_m.view(M * C, gp, gp)
        
        pred_spatial = torch.clamp(pred_spatial, self.glcm.min_val, self.glcm.max_val)
        target_spatial = torch.clamp(target_spatial, self.glcm.min_val, self.glcm.max_val)
        
        H_pred = self.glcm(pred_spatial, chunk_size=self.chunk_size)
        H_targ = self.glcm(target_spatial, chunk_size=self.chunk_size)
        
        loss_glcm = F.mse_loss(H_pred, H_targ, reduction='mean')
        
        total_loss = loss_mse + effective_alpha * loss_glcm
        
        loss_components = {
            "mse_loss": loss_mse.detach(),
            "glcm_loss": loss_glcm.detach()
            #"glcm_loss": (loss_glcm * effective_alpha).detach()
        }
        
        return total_loss, loss_components

    def get_texture_error_map(self, pred_img, target_img):
        """Computes the spatial GLCM MSE error map between pred and target for 16x16 non-overlapping windows."""
        B, C, H, W = pred_img.shape
        gp = self.group_tokens * self.patch_size # usually 16
        
        # Extract non-overlapping 16x16 patches
        # (B, C, H, W) -> unfold -> (B, C, num_windows_h, num_windows_w, gp, gp)
        num_windows_h = H // gp
        num_windows_w = W // gp
        
        pred_blocks = pred_img.unfold(2, gp, gp).unfold(3, gp, gp).reshape(-1, gp, gp)
        target_blocks = target_img.unfold(2, gp, gp).unfold(3, gp, gp).reshape(-1, gp, gp)
        
        pred_blocks = torch.clamp(pred_blocks, self.glcm.min_val, self.glcm.max_val)
        target_blocks = torch.clamp(target_blocks, self.glcm.min_val, self.glcm.max_val)
        
        # Compute GLCM
        H_pred = self.glcm(pred_blocks, chunk_size=self.chunk_size)
        H_targ = self.glcm(target_blocks, chunk_size=self.chunk_size)
        
        # Compute MSE per patch
        mse = F.mse_loss(H_pred, H_targ, reduction='none')
        mse_per_patch = mse.mean(dim=(1, 2)) # mean over K, K bins
        
        # Reshape to spatial grid
        mse_grid = mse_per_patch.view(B, C, num_windows_h, num_windows_w)
        
        # Average over channels to get one error scalar per window
        error_map = mse_grid.mean(dim=1) # (B, num_windows_h, num_windows_w)
        
        # Interpolate up to full image size (B, H, W)
        error_map = F.interpolate(error_map.unsqueeze(1), size=(H, W), mode='nearest').squeeze(1)
        
        return error_map
