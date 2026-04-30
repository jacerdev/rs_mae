from models.strategies.loss.base import ReconstructionLoss


class MaskedMSELoss(ReconstructionLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, pred, target, mask, **kwargs):
        targets = target if isinstance(target, tuple) else (target,)
        sizes = [t.shape[-1] for t in targets]
        pred_parts = pred.split(sizes, dim=-1)

        losses = [
            ((p - t).pow(2).mean(dim=-1) * mask).sum() / mask.sum().clamp(min=1)
            for p, t in zip(pred_parts, targets)
        ]
        return sum(losses), tuple(losses)
