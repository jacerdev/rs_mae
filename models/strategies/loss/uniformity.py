import torch.nn.functional as F

from models.strategies.loss.base import ReconstructionLoss
from models.strategies.loss import MaskedMSELoss


class UniformityEnhancedMAELoss(ReconstructionLoss):
    def __init__(self, lambda_unif=0.01, base_loss=None, **kwargs):
        super().__init__(**kwargs)
        self.recon_loss = base_loss if base_loss is not None else MaskedMSELoss()
        self.lambda_unif = lambda_unif
        
    def __call__(self, pred, target, mask, **kwargs):
        """
        pred: (B, N, D)
        target: (B, N, D)
        mask: (B, N)
        kwargs: expects 'latent' (B, N_enc, D_enc)
        """
        loss_mae, loss_components = self.recon_loss(pred, target, mask, **kwargs)
        
        # the paper uses the [CLS] token (index 0) after the encoder's layer norm
        latent = kwargs['latent']
        features = latent[:, 0, :]
        loss_unif = uniformity_loss(features)
        
        loss_components["uniformity_loss"] = loss_unif.detach()
        return loss_mae + self.lambda_unif * loss_unif, loss_components
        
def uniformity_loss(features):
    """
    features: Tensor (B, D), typically the [CLS] token
    Returns:
        mean of squared pairwise similarities
    """
    features = F.normalize(features, dim=-1)
    
    # compute the full pairwise cosine similarity matrix (self-similarity is constant so zero gradient)
    sim_matrix = features @ features.T

    return (sim_matrix ** 2).mean()