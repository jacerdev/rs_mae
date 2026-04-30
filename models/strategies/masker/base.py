
# Reference: https://github.com/facebookresearch/mae

import torch

class Masker():
    def __init__(self, **kwargs):
        pass

    def __call__(self, x, mask_ratio, scores, **kwargs):
        """
        :param x: Tensor of shape (B, N, D), where N is the number of patches/tokens.
        :param mask_ratio: float
        :param scores: Tensor of shape (B, N) containing the masking scores.
        :param kwargs: additional arguments
        :return:
            x: (B, N_visible, D) modified tokens.
            mask: (B, N) binary mask
            ids_keep: (B, N_visible) indices of visible tokens.
            ids_restore: (B, N) indices to restore original order.
        """
        B, N, D = x.shape
        N_keep = int(N * (1 - mask_ratio))

        # Sort and take bottom N_keep as visible (highest scores are masked)
        ids_shuffle = torch.argsort(scores, dim=1)
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