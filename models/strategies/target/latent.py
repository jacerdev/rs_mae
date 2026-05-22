import copy
import torch
import torch.nn as nn

from models.strategies.loss import MaskedMSELoss
from models.strategies.target.base import TargetGenerator
from utils.spatial_ops import patchify, z_normalize, z_denormalize


class LatentTarget(nn.Module, TargetGenerator):
    """
    TargetGenerator that combines original pixel patches and latent features
    from a momentum encoder as targets.
    """
    def __init__(self, patch_size, in_chans, embed_dim, m=0.996, norm_pix=True, **kwargs):
        nn.Module.__init__(self)

        self.m = m
        self.norm_pix = norm_pix
        self._pixel_feats = in_chans * patch_size ** 2
        self._latent_feats = embed_dim

        TargetGenerator.__init__(self, patch_size=patch_size, 
                                 out_features=self._pixel_feats + self._latent_feats, **kwargs)

        # to be populated in set_encoder
        self.patch_embed = None
        self.pos_embed = None
        self.blocks = None
        self.norm = None
        self.cls_token = None

    def set_encoder(self, online_encoder):
        """clone the encoder components from the online model and disable gradients"""
        self.patch_embed = copy.deepcopy(online_encoder.patch_embed)
        self.pos_embed = copy.deepcopy(online_encoder.pos_embed)
        self.blocks = copy.deepcopy(online_encoder.blocks)
        self.norm = copy.deepcopy(online_encoder.norm)
        
        if online_encoder.cls_token is not None:
            self.cls_token = copy.deepcopy(online_encoder.cls_token)

        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def _update_ema(self, online_encoder):
        """update EMA parameters from the online encoder"""
        submodules = [
            (self.patch_embed, online_encoder.patch_embed),
            (self.pos_embed, online_encoder.pos_embed),
            (self.blocks, online_encoder.blocks),
            (self.norm, online_encoder.norm)
        ]
        
        for target_m, online_m in submodules:
            for param_k, param_q in zip(target_m.parameters(), online_m.parameters()):
                param_k.mul_(self.m).add_(param_q, alpha=1.0 - self.m) # target = m * target + (1-m) * online
                
        if self.cls_token is not None and online_encoder.cls_token is not None:
            self.cls_token.mul_(self.m).add_(online_encoder.cls_token, alpha=1.0 - self.m)

    def forward(self, imgs, online_encoder=None, **kwargs):
        """
        imgs: (B, C, H, W) normalized imagery
        returns: Tuple of (pixel_tokens, latent_tokens)
        """
        if online_encoder is not None and online_encoder.training:
            self._update_ema(online_encoder)

        # prepare latent targets
        with torch.no_grad():
            x = self.patch_embed(imgs)
            
            if self.cls_token is not None:
                cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
                x = torch.cat((cls_tokens, x), dim=1)
                
            x = self.pos_embed(x, None)  # None for ids_keep so no masking
            
            for blk in self.blocks:
                x = blk(x)
            x = self.norm(x)
            
            if self.cls_token is not None: x = x[:, 1:, :] # discard cls_token
        latent_tokens = x

        # prepare pixel targets
        pixel_tokens = patchify(imgs, self.patch_size)
        if self.norm_pix:
            pixel_tokens = z_normalize(pixel_tokens)

        return pixel_tokens, latent_tokens

    def restore(self, pred, **kwargs):
        """split prediction back into pixel and latent parts"""
        pixel_pred = pred[..., :self._pixel_feats]
        latent_pred = pred[..., self._pixel_feats:]
        
        if self.norm_pix:
            gt_imgs = kwargs["gt_imgs"]
            raw_pixel_tokens = patchify(gt_imgs, self.patch_size)
            pixel_pred = z_denormalize(pixel_pred, raw_pixel_tokens)
                
        return pixel_pred, latent_pred


class LatentTargetLoss(MaskedMSELoss):
    """wrapper for MaskedMSELoss to return individual losses as a dict"""
    def __call__(self, pred, target, mask, **kwargs):
        total_loss, loss_components = super().__call__(pred, target, mask, **kwargs)
        
        return total_loss, {
            "pixel_loss": loss_components["mse_loss_0"],
            "latent_loss": loss_components["mse_loss_1"]
        }
