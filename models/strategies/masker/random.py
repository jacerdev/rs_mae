import torch

from models.strategies.masker.base import Masker


class RandomMasker(Masker):
    def __call__(self, x, mask_ratio=0.75, **kwargs):
        B, N, D = x.shape
        noise = torch.rand(B, N, device=x.device) # Random scores per token
        return super().__call__(x, mask_ratio, noise, **kwargs)

