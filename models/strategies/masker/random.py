# Reference: https://github.com/facebookresearch/mae

import torch

from models.strategies.masker.base import Masker


class RandomMasker(Masker):
    def mask(self, x, mask_ratio=0.75, **kwargs):
        B, N, D = x.shape
        N_keep = int(N * (1 - mask_ratio))

        noise = torch.rand(B, N, device=x.device) # Random scores per token
        # Sort and take bottom N_keep as visible
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :N_keep]
        x_visible = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, N], device=x.device)
        mask[:, :N_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_visible, mask, ids_keep, ids_restore

    def unmask(self, x_visible, ids_restore, mask_token):
        B, N_keep, D = x_visible.shape
        N = ids_restore.shape[1]
        mask_tokens = mask_token.repeat(B, N - N_keep, 1)
        x = torch.cat([x_visible, mask_tokens], dim=1)
        x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D))  # unshuffle

        return x


if __name__ == "__main__":
    B, N, D = 2, 4, 3
    x = torch.arange(B * N * D, dtype=torch.float32).reshape(B, N, D)

    # fixed noise
    noise = torch.tensor([[0.5, 0.7, 0.1, 0.8]])

    masking = RandomMasker(mask_ratio=0.5)
    x_visible, ids_keep = masking(x)#, fixed_noise=torch.tensor([[0.5, 0.7, 0.1, 0.8]]))

    print("ids_shuffle (top-k indices):", torch.topk(noise, k=2, dim=1)[1])
    print("ids_keep (sorted):", ids_keep)
    print("x:", x)
    print("x_visible:", x_visible)