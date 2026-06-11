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
    Reference: 'Focus on Texture: Rethinking Pre-training in Masked Autoencoders for Medical Image Classification'
    """
    def __init__(self, bins=32, min_val=-3.0, max_val=3.0, bandwidth=None):
        super().__init__()
        self.bins = bins
        self.min_val = min_val
        self.max_val = max_val
        
        # L is the length of bins
        self.L = (max_val - min_val) / bins
        
        # W is the bandwidth, controls smoothness
        self.W = bandwidth if bandwidth is not None else self.L / 2.0
        
        # mu is the center of each bin
        centers = torch.linspace(min_val + self.L / 2, max_val - self.L / 2, bins)
        self.register_buffer("mu", centers.view(1, -1, 1)) # (1, K, 1)

    def _get_probabilities(self, I):
        """
        Computes KDE-based probabilities P for each intensity bin using a sigmoid kernel.
        I: (B, 1, N)
        Returns: (B, K, N)
        """
        term1 = torch.sigmoid((I - self.mu + self.L / 2) / self.W)
        term2 = torch.sigmoid((I - self.mu - self.L / 2) / self.W)
        return term1 - term2

    def compute_histogram(self, I1, I2):
        """
        Computes the joint histogram H(I1, I2).
        I1, I2: (B, N) flattened adjacent pixels
        Returns: (B, K, K) differentiable GLCM matrix
        """
        B, N = I1.shape
        if N == 0:
            return torch.zeros((B, self.bins, self.bins), device=I1.device)
            
        I1_exp = I1.unsqueeze(1) # (B, 1, N)
        I2_exp = I2.unsqueeze(1) # (B, 1, N)
        
        P1 = self._get_probabilities(I1_exp) # (B, K, N)
        P2 = self._get_probabilities(I2_exp) # (B, K, N)
        
        # H(I1, I2) = (1/N) * P1 @ P2^T
        H = torch.bmm(P1, P2.transpose(1, 2)) / N
        
        # Normalize to ensure it sums to 1.0 (valid probability distribution)
        H = H / (H.sum(dim=(1, 2), keepdim=True) + 1e-8)
        
        return H

    def forward(self, x, chunk_size=1024):
        """
        x: (B, H, W) spatial patches
        Returns: Horizontal and Vertical GLCMs, each of shape (B, K, K)
        """
        B, H_dim, W_dim = x.shape
        
        # Process in chunks to prevent OOM on large batches
        if B > chunk_size:
            H_h_list, H_v_list = [], []
            for i in range(0, B, chunk_size):
                h, v = self.forward(x[i:i+chunk_size], chunk_size=B+1) # disable chunking for recursive call
                H_h_list.append(h)
                H_v_list.append(v)
            return torch.cat(H_h_list, dim=0), torch.cat(H_v_list, dim=0)
        
        # Horizontal (drop last column vs drop first column)
        I1_h = x[:, :, :-1].reshape(B, -1)
        I2_h = x[:, :, 1:].reshape(B, -1)
        H_h = self.compute_histogram(I1_h, I2_h)
        
        # Vertical (drop last row vs drop first row)
        I1_v = x[:, :-1, :].reshape(B, -1)
        I2_v = x[:, 1:, :].reshape(B, -1)
        H_v = self.compute_histogram(I1_v, I2_v)
        
        return H_h, H_v

class MaskedGLCMLoss(nn.Module, ReconstructionLoss):
    """
    Computes a combined GLCM and MSE loss for Masked Autoencoders.
    Reconstructs the spatial patch and matches the statistical texture matrices.
    """
    def __init__(self, 
                 patch_size: int,
                 in_chans: int = 4,
                 bins: int = 32,
                 min_val: float = -3.0,
                 max_val: float = 3.0,
                 bandwidth: float = None,
                 alpha: float = 1.0, # MSE weight
                 beta: float = 1.0,  # GLCM weight
                 warmup_start_epoch: int = 0,
                 warmup_epochs: int = 0,
                 **kwargs):
        nn.Module.__init__(self)
        ReconstructionLoss.__init__(self, **kwargs)
        
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.alpha = alpha
        self.beta = beta
        self.warmup_start_epoch = warmup_start_epoch
        self.warmup_epochs = warmup_epochs
        
        self.glcm = DifferentiableGLCM(bins=bins, min_val=min_val, max_val=max_val, bandwidth=bandwidth)
        
    def __call__(self, pred, target, mask, phase="", **kwargs):
        """
        pred: (B, L, D) where D = patch_size**2 * in_chans
        target: (B, L, D)
        mask: (B, L) where 1 indicates masked patch
        """
        B, L, D = pred.shape
        
        # standard pixel-wise MSE Loss
        loss_mse_full = (pred - target) ** 2
        loss_mse_full = loss_mse_full.mean(dim=-1) # (B, L)
        loss_mse = (loss_mse_full * mask).sum() / mask.sum().clamp(min=1)
        
        # scaling for GLCM weight
        if phase == "validation":
            scale = 1.0 # evaluate full loss dynamically
        else:
            epoch = TRAINING_STATE.epoch
            if epoch < self.warmup_start_epoch:
                scale = 0.0
            else:
                if self.warmup_epochs <= 0:
                    scale = 1.0
                else:
                    # Cosine ramp-up for smooth gradient injection
                    progress = min(1.0, (epoch - self.warmup_start_epoch) / self.warmup_epochs)
                    scale = (1.0 - math.cos(progress * math.pi)) / 2.0
                    
        effective_beta = self.beta * scale
        
        # Extract masked patches only for GLCM
        mask_flat = mask.view(-1).bool()
        pred_m = pred.reshape(-1, D)[mask_flat]   # (M, D)
        target_m = target.reshape(-1, D)[mask_flat] # (M, D)
        
        M = pred_m.shape[0]
        if M == 0 or effective_beta == 0:
            return self.alpha * loss_mse, {"mse_loss": loss_mse.detach(), "glcm_loss": torch.tensor(0.0, device=pred.device)}
            
        # Reshape to spatial dimensions: (M * C, p, p)
        # We treat each channel of each patch as an independent image for GLCM
        p = self.patch_size
        C = self.in_chans
        
        pred_spatial = pred_m.reshape(M * C, p, p)
        target_spatial = target_m.reshape(M * C, p, p)
        
        # clamp extreme values so KDE bins don't completely miss them
        pred_spatial = torch.clamp(pred_spatial, self.glcm.min_val, self.glcm.max_val)
        target_spatial = torch.clamp(target_spatial, self.glcm.min_val, self.glcm.max_val)
        
        # Compute GLCMs
        # use gradient checkpointing to save VRAM
        if pred_spatial.requires_grad:
            # Checkpoint the glcm forward pass
            H_h_pred, H_v_pred = checkpoint(self.glcm, pred_spatial, use_reentrant=False)
        else:
            H_h_pred, H_v_pred = self.glcm(pred_spatial)
            
        H_h_targ, H_v_targ = self.glcm(target_spatial)
        
        # Compute MSE between GLCM matrices
        loss_glcm_h = F.mse_loss(H_h_pred, H_h_targ, reduction='mean')
        loss_glcm_v = F.mse_loss(H_v_pred, H_v_targ, reduction='mean')
        loss_glcm = (loss_glcm_h + loss_glcm_v) / 2.0
        
        # Total Loss
        total_loss = self.alpha * loss_mse + effective_beta * loss_glcm
        
        loss_components = {
            "mse_loss": loss_mse.detach(),
            "glcm_loss": loss_glcm.detach()
        }
        
        return total_loss, loss_components
