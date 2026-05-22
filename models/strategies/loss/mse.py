from models.strategies.loss.base import ReconstructionLoss


class MaskedMSELoss(ReconstructionLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, pred, target, mask, **kwargs):
        """
        If `target` is a tuple, `pred` is split along the last dimension to match each
        target component, and an independent MSE is computed per component. Otherwise,
        a single MSE is computed.
        """
        targets = target if isinstance(target, tuple) else (target,)
        sizes = [t.shape[-1] for t in targets]
        pred_parts = pred.split(sizes, dim=-1)

        losses = [
            ((p - t).pow(2).mean(dim=-1) * mask).sum() / mask.sum().clamp(min=1)
            for p, t in zip(pred_parts, targets)
        ]

        if len(losses) == 1: loss_components = {"mse_loss": losses[0].detach()}
        else: loss_components = {f"mse_loss_{i}": l.detach() for i, l in enumerate(losses)}

        return sum(losses), loss_components
