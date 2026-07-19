from segmentation_models_pytorch.base.heads import SegmentationHead
from segmentation_models_pytorch.base.model import SegmentationModel
from segmentation_models_pytorch.decoders.fpn.decoder import FPNDecoder
import torch
import torch.nn as nn
import logging

from models.backbones import MaskedAutoencoderViT


class MAEPyramidEncoder(nn.Module):
    """
    SMP-compatible ViT Encoder wrapper.
    Extracts intermediate features from specific transformer blocks specfied by `out_indices`
    """
    def __init__(self, **mae_kwargs):
        super().__init__()

        self.grid_size = mae_kwargs['img_size'] // mae_kwargs['patch_size'] # Assume square image and patch
        self.embed_dim = mae_kwargs['embed_dim']

        # instantiate encoder
        self.vit = MaskedAutoencoderViT(**mae_kwargs, use_decoder=False,
                                        masker=None, target=None, loss=None,
                                        decoder_embed_dim=None, decoder_depth=None, decoder_num_heads=None)

        # blocks to extract features from: ~1/4, ~1/2, ~3/4, and final block
        depth = mae_kwargs['depth']
        self.out_indices = [max(0, depth // 4 - 1),
                            max(0, depth // 2 - 1),
                            max(0, depth * 3 // 4 - 1),
                            depth - 1]
        # SMP interface
        self._out_channels = [mae_kwargs["in_chans"]] + [self.embed_dim]*4

        # rescale modules following Li et al. (2021b): exact ops per scale to produce
        # standard 1/4, 1/8, 1/16, 1/32 pyramid. Scale is patch_size-dependent:
        scales = [mae_kwargs['patch_size'] / s for s in [4.0, 8.0, 16.0, 32.0]]

        def _make_adapter(scale):
            D = self.embed_dim
            if scale >= 4.0: # not used for p=8
                return nn.Sequential(
                    nn.ConvTranspose2d(D, D, kernel_size=2, stride=2),
                    nn.GroupNorm(32, D),
                    nn.GELU(),
                    nn.ConvTranspose2d(D, D, kernel_size=2, stride=2),
                )
            elif scale == 2.0:
                return nn.ConvTranspose2d(D, D, kernel_size=2, stride=2)
            elif scale == 1.0:
                return nn.Identity()
            elif scale == 0.5:
                return nn.MaxPool2d(kernel_size=2, stride=2)
            else:
                # scale < 0.5 (e.g. 0.25 for p=8 last block): conv downsample then maxpool
                # It is not established in the literature how to downsample by 4x, as the baseline only downsamples by 2x using max pooling.
                # We custom-design this 4x downsampling by combining a stride-2 convolution with the literature's standard max pooling.
                return nn.Sequential(
                    nn.Conv2d(D, D, kernel_size=2, stride=2),
                    nn.GroupNorm(32, D),
                    nn.GELU(),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )

        self.res_adapters = nn.ModuleList([_make_adapter(s) for s in scales])

    @property
    def out_channels(self):
        return self._out_channels
        
    def forward(self, x):
        if self.vit.normalize:
            x = (x - self.vit.mean) / self.vit.std
        x = torch.nan_to_num(x, nan=0.0)

        B, C, H, W = x.shape
        
        x_emb = self.vit.patch_embed(x)
        
        if self.vit.cls_token is not None:
            cls_tokens = self.vit.cls_token.expand(B, -1, -1)
            x_emb = torch.cat((cls_tokens, x_emb), dim=1)
            
        x_emb = self.vit.pos_embed(x_emb)
        
        # extract features at out_indices
        features = [x]
        curr_x = x_emb
        for i, blk in enumerate(self.vit.blocks):
            curr_x = blk(curr_x)
            
            if i in self.out_indices:
                idx = self.out_indices.index(i)
                feat = self.vit.norm(curr_x)
                 # drop CLS token
                if self.vit.cls_token is not None:
                    feat = feat[:, 1:]
                    
                # reshape to 2D spatial features and apply scale operation to create a pyramid for FPN
                feat = feat.transpose(1, 2).contiguous()
                feat = feat.view(B, self.embed_dim, self.grid_size, self.grid_size)
                feat = self.res_adapters[idx](feat)
                features.append(feat)
                
        return features

class SegmentationViT(SegmentationModel):
    def __init__(self, num_classes, pyramid_channels=256, segmentation_channels=128,
                 **mae_kwargs):
        super().__init__()

        self.img_size = mae_kwargs['img_size']
        self.num_classes = num_classes

        self.encoder = MAEPyramidEncoder(**mae_kwargs)
        logging.debug(f"Encoder params: {sum(p.numel() for p in self.encoder.parameters()) / 1e6} M")

        self.decoder = FPNDecoder(
            encoder_channels=self.encoder.out_channels,
            encoder_depth=4,
            pyramid_channels=pyramid_channels,
            segmentation_channels=segmentation_channels,
        )
        logging.debug(f"FPN Decoder params: {sum(p.numel() for p in self.decoder.parameters()) / 1e6} M")
        
        self.segmentation_head = SegmentationHead(
            in_channels=segmentation_channels, 
            out_channels=num_classes,
            activation=None,
            kernel_size=3,
            upsampling=4  # FPN P2 is always at H/4 resolution
        )
        logging.debug(f"Segmentation head params: {sum(p.numel() for p in self.segmentation_head.parameters()) / 1e6} M")
        
        self.classification_head = None # pure segmentation task
        
        self.name = "vit-mae-custom"
        self.initialize()
    
    def forward(self, x):
        features = self.encoder(x)
        decoder_output = self.decoder(*features)
        masks = self.segmentation_head(decoder_output)
        return masks

def load_pretrained_encoder_weights(model, ckpt_path, bands): # model is a SegmentationViT
    logging.info(f"Loading pretrained MAE weights from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]

    # drop pos_embed keys because they're fixed 2D sine-cosine embeddings
    raw_state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()
                      if k.startswith("model.") and "pos_embed" not in k}

    # slice patch_embed.proj.weight to match the downstream model's expected bands
    PRETRAINED_MODEL_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
    if list(bands) != PRETRAINED_MODEL_BANDS:
        if set(bands).issubset(PRETRAINED_MODEL_BANDS):
            band_indices = [PRETRAINED_MODEL_BANDS.index(b) for b in bands]
            logging.info(f"Slicing patch_embed.proj.weight: {band_indices}")
            raw_state_dict["patch_embed.proj.weight"] = raw_state_dict["patch_embed.proj.weight"][:, band_indices, :, :]
        else:
            logging.warning(f"At least one band is unknown in {bands}")

    # remap to the SegmentationViT's encoder structure
    adjusted_state_dict = {f"encoder.vit.{k}": v for k, v in raw_state_dict.items()}

    # preemptively strip shape mismatches
    target_state = model.state_dict()
    valid_state_dict = {}

    for k, v in adjusted_state_dict.items():
        if k in target_state:
            if v.shape == target_state[k].shape:
                valid_state_dict[k] = v
            else:
                logging.warning(f"Shape mismatch dropping '{k}': target={target_state[k].shape} vs ckpt={v.shape}")
        else:
            valid_state_dict[k] = v # Let strict=False deal with extra MAE parameters (decoder..)

    missing, unexpected = model.load_state_dict(valid_state_dict, strict=False)

    logging.debug(f"Missing keys: {missing}")
    logging.debug(f"Unexpected keys: {unexpected}")