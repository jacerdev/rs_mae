import torch

from models.strategies.masker.base import Masker


class RandomMasker(Masker):
    def __call__(self, x, mask_ratio=0.75, **kwargs):
        B, N, D = x.shape
        noise = torch.rand(B, N, device=x.device) # Random scores per token
        return super().__call__(x, mask_ratio, noise, **kwargs)


if __name__ == "__main__":
    B, N, D = 2, 4, 3
    x = torch.arange(B * N * D, dtype=torch.float32).reshape(B, N, D)

    # fixed noise
    noise = torch.tensor([[0.5, 0.7, 0.1, 0.8]])

    masking = RandomMasker(mask_ratio=0.5)
    x_visible, mask, ids_keep, ids_restore = masking(x)#, fixed_noise=torch.tensor([[0.5, 0.7, 0.1, 0.8]]))

    print("ids_shuffle (top-k indices):", torch.topk(noise, k=2, dim=1)[1])
    print("ids_keep (sorted):", ids_keep)
    print("x:", x)
    print("x_visible:", x_visible)