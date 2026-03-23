from segmentation_models_pytorch.base.heads import SegmentationHead
from segmentation_models_pytorch.base.model import SegmentationModel
from segmentation_models_pytorch.decoders.fpn.decoder import FPNDecoder
import torch
import torch.nn as nn
import logging

from models.backbones import MaskedAutoencoderViT
from models.backbones.layers.embedding import PatchEmbedding


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

        # Rescale modules to create a 1/2, 1/4, 1/8, 1/16 pyramid
        # ViT-Adapter's pure interpolation and norm from "Vision Transformer Adapter for Dense Predictions" (https://arxiv.org/abs/2205.08534)
        self.res_adapters = nn.ModuleList([
            nn.Sequential( # 14x14 -> 56x56
                nn.Upsample(scale_factor=4.0, mode='bilinear', align_corners=False),
                nn.BatchNorm2d(self.embed_dim)
            ),
            nn.Sequential( # 14x14 -> 28x28
                nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=False),
                nn.BatchNorm2d(self.embed_dim)
            ),
            nn.Sequential( # 14x14
                nn.Identity(),
                nn.BatchNorm2d(self.embed_dim)
            ),
            nn.Sequential( # 14x14 -> 7x7
                nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                nn.BatchNorm2d(self.embed_dim)
            )
        ])

    @property
    def out_channels(self):
        return self._out_channels
        
    def forward(self, x):
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
            upsampling=mae_kwargs['patch_size'] // 4 # FPN output is of scale patch_size/4 scale
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