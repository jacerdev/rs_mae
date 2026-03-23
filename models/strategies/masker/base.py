
import torch

class Masker():
    def __init__(self, **kwargs):
        pass

    def mask(self, x, mask_ratio, **kwargs):
        """
        :param x: Tensor of shape (B, N, D), where N is the number of patches/tokens.
        :param mask_ratio: float
        :param kwargs: additional arguments
        :return:
            x: (B, N_visible, D) modified tokens.
            mask: (B, N) binary mask
            ids_keep: (B, N_visible) indices of visible tokens.
            ids_restore: (B, N) indices to restore original order.
        """
        raise NotImplementedError

    def unmask(self, x_visible, ids_restore, mask_token):

        raise NotImplementedError