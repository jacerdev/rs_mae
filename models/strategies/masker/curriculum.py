import torch
import torch.nn as nn

from models.strategies.masker.base import Masker


class CurriculumMasker(nn.Module, Masker):
    """
    Curriculum Masker that transitions from Easy to Random to Hard tasks based on saliency and epoch.
    """
    def __init__(self, saliency_fn, num_epochs):
        nn.Module.__init__(self)
        Masker.__init__(self)
        self.saliency_fn = saliency_fn
        self.num_epochs = num_epochs
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def compute_masking_score(self, saliency, epoch, num_epochs):
        """
        Compute the masking score based on saliency and current epoch.
        Higher score patches are to be masked.
        
        Args:
            saliency: Tensor of shape (B, N) representing the richness/saliency of each patch.
            epoch: Current training epoch.
            num_epochs: Total number of training epochs E.
            
        Returns:
            scores: Tensor of shape (B, N) containing the masking scores.
        """
        # α = 2.e/E − 1, sweeps from -1 to 1
        alpha = 2.0 * epoch / num_epochs - 1.0
        
        # noise uniformly sampled from range [0,1]
        noise = torch.rand_like(saliency)
        
        # score = α. saliency + (1- |α|).noise
        score = alpha * saliency + (1.0 - abs(alpha)) * noise
        return score

    def get_normalized_saliency(self, imgs):
        """Extracts and normalizes saliency to [0, 1] range."""
        saliency = self.saliency_fn(imgs)
        
        # Min-Max Normalization per sample
        s_flat = saliency.view(saliency.shape[0], -1)
        s_min = s_flat.min(dim=1, keepdim=True)[0].view(-1, 1, 1)
        s_max = s_flat.max(dim=1, keepdim=True)[0].view(-1, 1, 1)
        
        normalized = (saliency - s_min) / (s_max - s_min + 1e-6)
        return normalized

    def forward(self, x, mask_ratio=0.75, **kwargs):
        B, N, D = x.shape
        imgs = kwargs.pop('imgs', x)
        
        saliency = self.get_normalized_saliency(imgs).flatten(1) # (B, N)
        scores = self.compute_masking_score(saliency, self.epoch, self.num_epochs)
        
        return Masker.__call__(self, x, mask_ratio, scores, **kwargs)


if __name__ == "__main__":
    B, N, D = 2, 4, 3
    x = torch.arange(B * N * D, dtype=torch.float32).reshape(B, N, D)
    
    # mock saliency for 4 patches
    saliency = torch.tensor([[0.1, 0.9, 0.2, 0.8], [0.5, 0.5, 0.5, 0.5]])
    
    # Use a lambda to return the mock saliency
    masking = CurriculumMasker(saliency_fn=lambda _: saliency, num_epochs=10)
    
    # Test batch 0 saliency: [0.1, 0.9, 0.2, 0.8]
    masking.set_epoch(0)
    _, _, ids_keep, _ = masking(x, mask_ratio=0.5)
    print(f"Epoch 0  (Easy) - Keep indices (expected 1, 3): {ids_keep[0].tolist()}")

    masking.set_epoch(5)
    _, _, ids_keep, _ = masking(x, mask_ratio=0.5)
    print(f"Epoch 5  (Rand) - Keep indices: {ids_keep[0].tolist()}")

    masking.set_epoch(10)
    _, _, ids_keep, _ = masking(x, mask_ratio=0.5)
    print(f"Epoch 10 (Hard) - Keep indices (expected 0, 2): {ids_keep[0].tolist()}")
