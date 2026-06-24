class ReconstructionLoss:
    def __init__(self, **kwargs):
        pass

    def __call__(self, pred, target, mask, **kwargs):
        """
        Args:
            pred: Tensor (B, N, D)
            target: Tensor (B, N, D)
            mask: Tensor (B, N), 1 = masked, 0 = visible
            **kwargs: Extra context (e.g., latent features) for specialized losses
        Returns:
            scalar total loss, dict of individual losses for logging
        """
        raise NotImplementedError