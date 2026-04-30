from models.strategies.target.base import TargetGenerator
from utils.spatial_ops import patchify


class IdentityTarget(TargetGenerator):
    def __init__(self, patch_size, **kwargs):
        in_chans = kwargs["in_chans"]
        out_features = in_chans * patch_size ** 2
        super().__init__(patch_size=patch_size, out_features=out_features, **kwargs)

    def __call__(self, x, **kwargs):
        return patchify(x, self.patch_size)

    def restore(self, pred, **kwargs):
        return None