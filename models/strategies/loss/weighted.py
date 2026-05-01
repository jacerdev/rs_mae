import torch
import torch.nn as nn

from models.strategies.loss.base import ReconstructionLoss
from train.callbacks import TRAINING_STATE


class SaliencyWeightedLoss(nn.Module, ReconstructionLoss):
    def __init__(self, saliency_extractor, patch_size=8, warmup_epochs=20, saliency_scale=0.5, **kwargs):
        nn.Module.__init__(self)
        ReconstructionLoss.__init__(self, **kwargs)
        self.saliency_extractor = saliency_extractor
        self.patch_size = patch_size
        self.warmup_epochs = warmup_epochs

        self.saliency_scale = saliency_scale
        

    def __call__(self, pred, target, mask, **kwargs):
        B, N, dim = pred.shape

        if "raw_imgs" not in kwargs:
            raise ValueError("raw_imgs must be provided for SaliencyWeightedLoss")
        raw_imgs = kwargs["raw_imgs"]
        
        # extract saliency and normalize
        with torch.no_grad():
            saliency = self.saliency_extractor(raw_imgs)  # (B, h_grid, w_grid)
            
            # min-max normalization per image
            s_flat = saliency.view(B, -1)
            s_min = s_flat.quantile(0.05, dim=1, keepdim=True)  # robust min
            s_max = s_flat.quantile(0.95, dim=1, keepdim=True)  # robust max
            s_norm = (s_flat - s_min) / (s_max - s_min + 1e-6)
            s_norm = s_norm.clamp(0, 1)  # shape: (B, N), range [0, 1]
            
        # scale starts at 0 and goes to 1 linearly over warmup_epochs
        scale = min(1.0, TRAINING_STATE.epoch / max(1, self.warmup_epochs))
        patch_weights = 1.0 + (self.saliency_scale * scale * s_norm)
        # Normalize weights so the mean is 1.0, preserving global learning rate
        patch_weights = patch_weights / patch_weights.mean(dim=-1, keepdim=True)

        mse_per_patch = (pred - target).pow(2).mean(dim=-1)  # (B, N)
        weighted_mse = mse_per_patch * patch_weights * mask  # (B, N)
        
        # Average over masked patches
        total_loss = weighted_mse.sum() / mask.sum().clamp(min=1)

        return total_loss, (total_loss,)
