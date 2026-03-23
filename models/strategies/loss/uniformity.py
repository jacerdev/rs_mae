import torch

from models.strategies.loss.base import ReconstructionLoss


class UniformityEnhancedMAELoss(ReconstructionLoss):
    def __init__(self, recon_loss_obj: ReconstructionLoss, lambda_unif=1.0, **kwargs):
        super().__init__(**kwargs)
        self.recon_loss = recon_loss_obj
        self.lambda_unif = lambda_unif

    def __call__(self, pred, target, mask, **kwargs):
        """
        pred: (B, N, D)
        target: (B, N, D)
        mask: (B, N)
        **kwargs: expects 'latent' (B, N_enc, D_enc)
        """
        latent = kwargs.get('latent')

        loss_mae = self.recon_loss(pred, target, mask)
        
        if latent is not None:
             # use [CLS] token for uniformity
             features = latent[:, 0, :]
             loss_unif = uniformity_loss(features)
             return loss_mae + self.lambda_unif * loss_unif
            
        return loss_mae


def uniformity_loss(features):
    """
    features: Tensor (B, D), unmasked features
    Returns:
        scalar loss
    """
    features = torch.nn.functional.normalize(features, dim=-1)

    # shuffle to get negative pairs
    perm = torch.randperm(features.size(0), device=features.device)
    features_neg = features[perm]

    sim = (features * features_neg).sum(dim=-1)  # (B,)
    return (sim ** 2).mean()
