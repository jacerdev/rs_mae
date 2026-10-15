import torch.nn as nn

from models.strategies.target.base import TargetGenerator
from utils.features import HOGLayer
from utils.spatial_ops import z_normalize, z_denormalize


class HOGTarget(nn.Module, TargetGenerator):
    def __init__(self, patch_size, bands, pool, nbins=9, use_max_gradient=False, gaussian_window=None,
                 normalize=True, norm_pix=True, **kwargs):
        nn.Module.__init__(self)

        assert patch_size % pool == 0, "patch_size must be divisible by pool"
        self.pool = pool
        self.nbins = nbins
        self.norm_pix = norm_pix
        self.use_max_gradient = use_max_gradient
        self.hog_out_channels = 1 if use_max_gradient else len(bands)

        self._hog_feats = (patch_size // pool)**2 * self.hog_out_channels * nbins

        TargetGenerator.__init__(self, patch_size=patch_size,
                                out_features=self._hog_feats, **kwargs)

        self.hog = HOGLayer(nbins=nbins, pool=pool, in_channels=len(bands), use_max_gradient=use_max_gradient,
                            gaussian_window=gaussian_window, normalize=normalize)

    def forward(self, imgs, **kwargs):
        """
        imgs: (B, C, H, W) normalized imagery
        returns: hog_tokens of shape [B, N, (p//pool)^2 * hog_out_channels * nbins]
        """
        hog_tokens = self._get_raw_hog_targets(imgs)

        if self.norm_pix:
            hog_tokens = z_normalize(hog_tokens)

        return hog_tokens

    def restore(self, pred, **kwargs):
        """Reverse the target Z-score normalization for visualization.
        pred: (B, N, D_out)
        """
        if self.norm_pix:
            gt_imgs = kwargs["gt_imgs"]
            raw_hog = self._get_raw_hog_targets(gt_imgs)
            pred = z_denormalize(pred, raw_hog)

        return pred

    def _get_raw_hog_targets(self, imgs):
        """Generate un-normalized HOG features"""
        hog_grid = self.hog(imgs)
        B, C, nb, hn, wn = hog_grid.shape
        n = self.patch_size // self.pool
        h, w = hn // n, wn // n
        hog_tokens = hog_grid.reshape(B, C, nb, h, n, w, n)
        hog_tokens = hog_tokens.permute(0, 3, 5, 1, 2, 4, 6)
        hog_tokens = hog_tokens.reshape(B, h * w, C * nb * n * n)
        return hog_tokens  # [B, N, C*nb*n²]

