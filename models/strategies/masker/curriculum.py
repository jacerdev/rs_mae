import torch
import torch.nn as nn

from models.strategies.masker.base import Masker
from train.callbacks import TRAINING_STATE


class CurriculumMasker(nn.Module, Masker):
    """
    Curriculum Masker that transitions from Easy to Random to Hard tasks based on saliency and epoch.
    """
    def __init__(self, saliency_extractor, max_epochs, easy_end=0.1, hard_start=0.6, min_w=-0.5, max_w=0.7, **kwargs):
        nn.Module.__init__(self)
        Masker.__init__(self, **kwargs)
        self.saliency_extractor = saliency_extractor
        self.max_epochs = max_epochs

        self.min_w = min_w
        self.max_w = max_w
        self.easy_end = easy_end
        self.hard_start = hard_start


    def compute_masking_score(self, saliency, epoch, max_epochs):
        """
        Higher score patches are to be masked.
        
        Args:
            saliency: Tensor of shape (B, N) representing the richness/saliency of each patch.
            epoch: Current training epoch.
            max_epochs: Total number of training epochs E.
            
        Returns:
            scores: Tensor of shape (B, N) containing the masking scores.
        """
        # noise uniformly sampled from range [0,1]
        noise = torch.rand_like(saliency)
        
        # weights
        gamma = epoch / max_epochs
        if gamma < self.easy_end: # Easy phase: saliency weight = min_w -> 0
            alpha = (1 - (gamma / self.easy_end)) * self.min_w
        elif gamma < self.hard_start: # Random phase: saliency weight = 0
            alpha = 0.0
        else: # Hard phase: saliency weight = 0 -> max_w
            alpha = ((gamma - self.hard_start) / (1.0 - self.hard_start)) * self.max_w
        
        return alpha * saliency + (1.0 - abs(alpha)) * noise

    def get_normalized_saliency(self, imgs):
        """Extracts and normalizes saliency to [0, 1] range."""
        saliency = self.saliency_extractor(imgs)
        
        # Min-Max Normalization per sample
        s_flat = saliency.view(saliency.shape[0], -1)
        s_min = s_flat.min(dim=1, keepdim=True)[0].view(-1, 1, 1)
        s_max = s_flat.max(dim=1, keepdim=True)[0].view(-1, 1, 1)
        
        normalized = (saliency - s_min) / (s_max - s_min + 1e-6)
        return normalized

    def forward(self, x, mask_ratio=0.75, phase="curriculum", raw_imgs=None, **kwargs):
        if phase == "curriculum":
            if raw_imgs is None:
                raise ValueError("raw_imgs must be provided for CurriculumMasker in curriculum phase")

            saliency = self.get_normalized_saliency(raw_imgs).flatten(1) # (B, N)
            scores = self.compute_masking_score(saliency, TRAINING_STATE.epoch, self.max_epochs)
        
        else: # val: random masking
            B, N, D = x.shape
            scores = torch.rand(B, N, device=x.device)
        
        return Masker.__call__(self, x, mask_ratio, scores, **kwargs)


if __name__ == "__main__":
    B, N, D = 2, 4, 3
    x = torch.arange(B * N * D, dtype=torch.float32).reshape(B, N, D)
    
    # mock saliency for 4 patches
    saliency = torch.tensor([[0.1, 0.9, 0.2, 0.8], [0.5, 0.5, 0.5, 0.5]])
    
    # Use a lambda to return the mock saliency
    masking = CurriculumMasker(saliency_extractor=lambda _: saliency, max_epochs=10)
    
    # Test batch 0 saliency: [0.1, 0.9, 0.2, 0.8]
    TRAINING_STATE.epoch = 0
    _, _, ids_keep, _ = masking(x, mask_ratio=0.5, raw_imgs=x)
    print(f"Epoch 0  (Easy) - Keep indices (expected 1, 3): {ids_keep[0].tolist()}")

    TRAINING_STATE.epoch = 5
    _, _, ids_keep, _ = masking(x, mask_ratio=0.5, raw_imgs=x)
    print(f"Epoch 5  (Rand) - Keep indices: {ids_keep[0].tolist()}")

    TRAINING_STATE.epoch = 10
    _, _, ids_keep, _ = masking(x, mask_ratio=0.5, raw_imgs=x)
    print(f"Epoch 10 (Hard) - Keep indices (expected 0, 2): {ids_keep[0].tolist()}")
