# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# --------------------------------------------------------
# Modified to allow for configurable masking strategy, reconstruction target, and loss design.
# Original code: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import logging
import torch
import torch.nn as nn

from models.backbones.layers.attention import SelfAttentionBlock
from models.backbones.layers.embedding import PatchEmbedding, SineCosinePosEmbedding
from models.strategies.loss import MaskedMSELoss
from models.strategies.masker import RandomMasker
from models.strategies.target import IdentityTarget


class MaskedAutoencoderViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 masker=None, target=None, loss=None,
                 cls_token=True,
                 drop_path_rate=0., # vit-s 0.0, vit-b 0.1
                 use_decoder=True,
                 normalize=False, bands=None):
        super().__init__()
        
        self.use_decoder = use_decoder
        self.normalize = normalize
        
        if self.normalize:
            from datasets.reben import ReBENDataset
            mean_tensor = torch.tensor([ReBENDataset.BAND_STATS["mean"][b] for b in bands], dtype=torch.float32)
            std_tensor = torch.tensor([ReBENDataset.BAND_STATS["std"][b] for b in bands], dtype=torch.float32)
            self.register_buffer("mean", mean_tensor.view(1, -1, 1, 1), persistent=False)
            self.register_buffer("std", std_tensor.view(1, -1, 1, 1), persistent=False)

        # Encoder
        # Patch Embed -> Add CLS & Pos Embed -> Self-Attention Blocks -> Layer Norm
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if cls_token else None

        self.patch_embed = PatchEmbedding(patch_size, in_chans, embed_dim)
        self.pos_embed = SineCosinePosEmbedding(embed_dim, img_size // patch_size, cls_token=cls_token)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)] # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            SelfAttentionBlock(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, path_drop=dpr[i])
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        logging.debug(f"Encoder params: {sum(p.numel() for n,p in self.named_parameters() if not n.startswith('decoder')) / 1e6:.3f} M")

        if self.use_decoder:
            # Strategies
            self.masker = masker if masker is not None else RandomMasker()
            self.target = target if target is not None else IdentityTarget(patch_size=patch_size, in_chans=in_chans)
            self.loss = loss if loss is not None else MaskedMSELoss()
           
            # Decoder
            # Linear Projection to Decoder dim -> Add Mask & Pos Embed -> Self-Attention Blocks -> Layer Norm -> Linear Prediction
            self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

            self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
            self.decoder_pos_embed = SineCosinePosEmbedding(decoder_embed_dim, img_size // patch_size, cls_token=cls_token)
            self.decoder_blocks = nn.ModuleList([
                SelfAttentionBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for _ in range(decoder_depth)])
            self.decoder_norm = norm_layer(decoder_embed_dim)

            self.decoder_pred = nn.Linear(decoder_embed_dim, self.target.out_features, bias=True)

            logging.debug(f"Decoder params: {sum(p.numel() for n,p in self.named_parameters() if n.startswith('decoder')) / 1e6:.3f} M")

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        if self.cls_token is not None:
            torch.nn.init.normal_(self.cls_token, std=.02)
        if self.use_decoder:
            torch.nn.init.normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, x, mask_ratio, **kwargs):
        x = self.patch_embed(x)
        
        if self.use_decoder and mask_ratio > 0:
            x, mask, ids_keep, ids_restore = self.masker(x, mask_ratio, **kwargs) # pass kwargs for saliency-based masking
        else:
            B, N, D = x.shape
            ids_keep = None # passing None makes SineCosinePosEmbedding bypass gather
            mask = torch.zeros(B, N, device=x.device)
            ids_restore = torch.arange(N, device=x.device).unsqueeze(0).expand(B, N)

        # append cls token
        if self.cls_token is not None:
            cls_token = self.cls_token
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        x = self.pos_embed(x, ids_keep)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return (x, mask, ids_restore) if self.use_decoder else x

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)

        if self.cls_token is not None:
            x_ = x[:, 1:, :]
            x_ = self.masker.unmask(x_, ids_restore, self.mask_token)
            x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        else:
            x = self.masker.unmask(x, ids_restore, self.mask_token)

        x = self.decoder_pos_embed(x)

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)

        if self.cls_token is not None: # remove cls token
            x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask, **kwargs):
        """
        imgs: [B, C_in, H, W]
        pred: [B, N, p*p*C_out]
        mask: [B, N], 0 is keep, 1 is remove,
        """
        target = self.target(imgs, **kwargs)
        return self.loss(pred, target, mask, **kwargs)

    def forward(self, imgs, mask_ratio=0.75, **kwargs):
        raw_imgs = torch.nan_to_num(imgs.detach(), nan=0.0, posinf=0.0, neginf=0.0)

        if self.normalize:
            imgs = (imgs - self.mean) / self.std
        imgs = torch.nan_to_num(imgs, nan=0.0, posinf=0.0, neginf=0.0)

        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio, raw_imgs=raw_imgs, **kwargs)
        pred = self.forward_decoder(latent, ids_restore)
        
        loss, extra_losses = self.forward_loss(imgs, pred, mask, latent=latent, raw_imgs=raw_imgs)
        return loss, pred, mask, extra_losses


def load_pretrained_mae_weights(model, ckpt_path, bands):
    """
    Load pretrained MAE weights (encoder + decoder) into a MaskedAutoencoderViT instance.
    Handles matching of specific spectral bands using patch_embed.proj.weight slicing.
    """
    logging.info(f"Loading pretrained full MAE weights from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]

    # drop pos_embed keys because they're fixed 2D sine-cosine embeddings
    # MAEModule wraps the model, so in the checkpoint keys start with "model."
    raw_state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()
                      if k.startswith("model.") and "pos_embed" not in k}

    # slice patch_embed.proj.weight to match the downstream model's expected bands
    PRETRAINED_MODEL_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
    if bands is not None and list(bands) != PRETRAINED_MODEL_BANDS:
        if set(bands).issubset(PRETRAINED_MODEL_BANDS):
            band_indices = [PRETRAINED_MODEL_BANDS.index(b) for b in bands]
            logging.info(f"Slicing patch_embed.proj.weight: {band_indices}")
            raw_state_dict["patch_embed.proj.weight"] = raw_state_dict["patch_embed.proj.weight"][:, band_indices, :, :]
        else:
            logging.warning(f"At least one band is unknown in {bands}")

    # preemptively strip shape mismatches
    target_state = model.state_dict()
    valid_state_dict = {}

    for k, v in raw_state_dict.items():
        if k in target_state:
            if v.shape == target_state[k].shape:
                valid_state_dict[k] = v
            else:
                logging.warning(f"Shape mismatch dropping '{k}': target={target_state[k].shape} vs ckpt={v.shape}")
        else:
            valid_state_dict[k] = v 

    missing, unexpected = model.load_state_dict(valid_state_dict, strict=False)

    logging.debug(f"Missing keys: {missing}")
    logging.debug(f"Unexpected keys: {unexpected}")
