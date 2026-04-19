from models.strategies.target.base import TargetGenerator
from utils.spatial_ops import patchify, z_normalize, z_denormalize


class NormalizedPixelTarget(TargetGenerator):
    def __init__(self, patch_size, **kwargs):
        in_chans = kwargs["in_chans"]
        out_features = in_chans * patch_size ** 2
        super().__init__(patch_size=patch_size, out_features=out_features, **kwargs)

    def __call__(self, imgs, **kwargs):
        x = patchify(imgs, self.patch_size)
        x_norm = z_normalize(x)
        return x_norm

    def restore(self, pred, **kwargs):
        gt_imgs = kwargs["gt_imgs"]

        x = patchify(gt_imgs, self.patch_size)
        pred_denormed = z_denormalize(pred, x)
        return pred_denormed