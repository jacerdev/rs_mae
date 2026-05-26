import torch
import torch.nn.functional as F
from models.strategies.target.base import TargetGenerator
from utils.spatial_ops import patchify

class SegmentationTarget(TargetGenerator):
    def __init__(self, patch_size, num_classes, **kwargs):
        out_features = num_classes * patch_size ** 2
        self.num_classes = num_classes
        super().__init__(patch_size=patch_size, out_features=out_features, **kwargs)

    def __call__(self, imgs, gt_masks=None, **kwargs):
        if gt_masks is None:
            raise ValueError("gt_masks must be provided to SegmentationTarget. Ensure it is passed through mae_vit.py kwargs.")
            
        gt_masks = gt_masks.unsqueeze(1) # (B, 1, H, W)
        
        target = patchify(gt_masks.float(), self.patch_size) # (B, N, p*p*C)
        return target.long()

    def restore(self, pred, **kwargs):
        return None # just use utils.visualization.prepare_segmentation_visuals()

class MaskedCrossEntropyLoss(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, pred, target, mask, **kwargs):
        """
        pred: (B, N, p*p*num_classes)
        target: (B, N, p*p) - class indices
        mask: (B, N) - 1 is masked, 0 is keep
        """        
        B, N, D = pred.shape
        P = target.shape[-1]
        C = D // P

        pred = pred.reshape(B * N * P, C)
        target = target.reshape(B * N * P)

        loss = F.cross_entropy(pred, target, reduction='none', ignore_index=-1)
        loss = loss.view(B, N, P).mean(dim=-1)  # (B, N)

        loss = (loss * mask).sum() / mask.sum().clamp(min=1)
        return loss, {"ce_loss": loss.detach()}

