# Reference: https://github.com/facebookresearch/mae

import numpy as np
import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    # (B, C, H, W) -> (B, H/patch_size * W/patch_size, embed_dim)
    def __init__(self, patch_size=16, in_channels=3, embed_dim=384, norm_layer=None):
        super().__init__()
        patch_size = tuple(patch_size) if hasattr(patch_size, '__iter__') else (patch_size, patch_size)
        self.patch_size = patch_size

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H % self.patch_size[0] == 0 and W % self.patch_size[0] == 0, f"Input image height and width must be divisible by patch size {self.patch_size}, got ({H},{W})."
        #assert H == self.img_size[0] and W == self.img_size[1], f"Input image size doesn't match model. expected: ({self.img_size[0]}*{self.img_size[1]}), got ({H}*{W})."

        x = self.proj(x) # (B, embed_dim, H_grid, W_grid)
        x = x.flatten(2).transpose(1, 2) # (B, H_grid * W_grid, embed_dim)
        x = self.norm(x)
        return x

class SineCosinePosEmbedding(nn.Module):
    def __init__(self, embed_dim, grid_size : int, cls_token=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.cls_token = cls_token

        pos_embed_tensor = self.create_sincos_pos_embed(embed_dim, grid_size, cls_token) # (N, embed_dim)
        self.register_buffer('pos_embed_tensor', torch.tensor(pos_embed_tensor, dtype=torch.float32))

    @staticmethod
    def create_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
        grid_h = np.arange(grid_size, dtype=np.float32)
        grid_w = np.arange(grid_size, dtype=np.float32)
        grid = np.stack(np.meshgrid(grid_w, grid_h), axis=0) # shape: (2, H_grid, W_grid)

        #grid = grid.reshape([2, 1, grid_size, grid_size]) # (!)

        def get_1d_sincos_pos_embed_from_grid(dim, pos, idx_range=14):
            assert dim % 2 == 0
            omega = np.arange(dim // 2, dtype=np.float32) * 2. / dim
            omega = 1. / 10000**omega

            out = pos.reshape(-1)[:, None] * omega[None, :] # (N, dim/2)
            emb = np.concatenate([np.sin(out), np.cos(out)], axis=1)
            return emb # (N, dim)

        emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0], idx_range=grid_size) # (H_grid * W_grid, embed_dim/2)
        emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1], idx_range=grid_size) # (H_grid * W_grid, embed_dim/2)
        pos_embed = np.concatenate([emb_h, emb_w], axis=1) # (H_grid * W_grid, embed_dim)

        if cls_token: pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)

        return pos_embed

    def forward(self, x, ids_keep=None): # ids_keep without CLS
        if ids_keep is not None:
            if self.cls_token:
                ids_keep = torch.cat([torch.zeros(x.shape[0], 1, dtype=ids_keep.dtype, device=ids_keep.device),
                                       ids_keep + 1], dim=1)  # (B, N_keep+1)
            ids_keep_exp = ids_keep.unsqueeze(-1).expand(-1, -1, self.embed_dim)  # (B, N_keep (+1 if cls_token), D)
            pos_visible = torch.gather(self.pos_embed_tensor.unsqueeze(0).expand(x.shape[0], -1, -1), dim=1, index=ids_keep_exp)  # (B, N_keep, D)
            return x + pos_visible
        else:
            return x + self.pos_embed_tensor.unsqueeze(0)



