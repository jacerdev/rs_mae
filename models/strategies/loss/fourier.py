import math
import torch
import torch.nn.functional as F

from models.strategies.loss.base import ReconstructionLoss
from models.strategies.loss.mse import MaskedMSELoss
from train.callbacks import TRAINING_STATE
from utils.spatial_ops import unpatchify


class FFTEnhancedLoss(ReconstructionLoss):
    def __init__(self, patch_size=16, lambda_amp=0.1, lambda_phase=0.5, norm="ortho", warmup_epochs=20, **kwargs):
        super().__init__(**kwargs)

        self.mse_loss_fn = MaskedMSELoss(**kwargs)
        self.patch_size = patch_size
        self.base_lambda_amp = lambda_amp
        self.base_lambda_phase = lambda_phase
        self.warmup_epochs = warmup_epochs
        self.norm = norm


    def __call__(self, pred, target, mask, **kwargs):
        # Spatial pixel loss (MSE)
        mse_total, _ = self.mse_loss_fn(pred, target, mask, **kwargs)
        
        # Fourier domain loss
        B, N, dim = pred.shape
        p = self.patch_size
        C = dim // (p * p)
        h = int(math.sqrt(N))
        H = W = h * p

        mask_expanded = mask.unsqueeze(-1) # Shape becomes (B, N, 1)
        pred_full = pred * mask_expanded + target * (1 - mask_expanded) # inject ground truth for visible patches
        pred_img = unpatchify(pred_full, p, H, W, C)
        target_img = unpatchify(target, p, H, W, C)
        
        with torch.amp.autocast('cuda', enabled=False): # force float32 to bypass the cuFFT power-of-2 requirement
            # 2D real FFT (for speed) on the full image
            fft_p = torch.fft.rfft2(pred_img.to(torch.float32), norm=self.norm)
            fft_t = torch.fft.rfft2(target_img.to(torch.float32), norm=self.norm)
            
            # amplitude loss
            log_amp_p = torch.log1p(torch.abs(fft_p)) # log to compress dynamic range
            log_amp_t = torch.log1p(torch.abs(fft_t))
            amp_total = F.l1_loss(log_amp_p, log_amp_t)
            
            # phase loss
            phase_diff = torch.angle(fft_p) - torch.angle(fft_t)
            #phase_dist = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff)).abs()
            #phase_total = phase_dist.mean() 
            phase_total = (1 - torch.cos(phase_diff)).mean() # avoid wrap-around issues (circular phase), grads of 1-cos are smoother at 0
        
        # Calculate current weights based on warmup
        scale = min(1.0, TRAINING_STATE.epoch / max(1, self.warmup_epochs))
        curr_lambda_amp = self.base_lambda_amp * scale
        curr_lambda_phase = self.base_lambda_phase * scale

        # Total loss
        total_loss = mse_total + (curr_lambda_amp * amp_total) + (curr_lambda_phase * phase_total)
        
        return total_loss, (mse_total, amp_total, phase_total)
