
class TargetGenerator:
    def __init__(self, patch_size, out_features, **kwargs):
        if out_features is None or patch_size is None: raise ValueError("patch_size and out_features cannot be None")
        self.out_features = out_features
        self.patch_size = patch_size

    def __call__(self, imgs):
        """
        imgs: (B, C, H, W)
        returns: targets: (B, N, D_out) -> Ready for Loss calculation
        """
        raise NotImplementedError

    def restore(self, pred, **kwargs):
        """
        Reverse the target transformation for visualization.
        pred: (B, N, D_out) - model predictions
        gt_imgs: (B, C, H, W) - the param `imgs` that was passed to __call__ (optional depending on target)
        returns: (B, N, D_out) - restored predictions in original range
        """
        # return None if no post-processing is needed
        raise NotImplementedError