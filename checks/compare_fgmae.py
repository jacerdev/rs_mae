import sys
from pathlib import Path
PROJECT_PATH = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_PATH))

import torch
import numpy as np

from sandbox.fg_mae.models_mae_2 import mae_vit_small_patch16 as official_model_fn
from models.backbones import MaskedAutoencoderViT as custom_model_fn
from models.strategies.target import HOGNDITarget
from models.strategies.masker import RandomMasker
from models.strategies.loss import MaskedMSELoss

def compare():
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    img_size = 224
    patch_size = 16
    in_chans = 13
    bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]
    
    # Official Model
    official_model = official_model_fn(in_chans=in_chans, norm_pix_loss=True, hog_norm=True)
    official_model.to(device)
    official_model.eval()

    # My Model
    masker = RandomMasker()
    custom_gen = HOGNDITarget(patch_size=patch_size, bands=bands, pool=8, nbins=9,
                               gaussian_window=16, normalize=True, norm_pix=True, bands_mean=[0.0]*13, bands_std=[1.0]*13)
    loss_fn = MaskedMSELoss()
    custom_model = custom_model_fn(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=384,
        depth=12,
        num_heads=6,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        masker=masker,
        target=custom_gen,
        loss=loss_fn
    )
    custom_model.to(device)
    custom_model.eval()

    # snyc weights for a fair comparison
    sync_weights(official_model, custom_model)
    
    # Input
    x = torch.rand(5, in_chans, img_size, img_size).to(device)

    # Official Forward
    torch.manual_seed(seed)
    with torch.no_grad():
        official_loss, official_mask, official_hog_pred, official_hog_target, official_ndi_pred, official_ndi_target = official_model(x, mask_ratio=0.7)

    # Custom Forward
    torch.manual_seed(seed)
    with torch.no_grad():
        custom_loss, custom_pred, custom_mask, custom_loss_components = custom_model(x, mask_ratio=0.7)
        # custom_pred has both concatenated
        hog_out_dim = custom_model.target._hog_feats
        custom_hog_pred = custom_pred[..., :hog_out_dim]
        custom_ndi_pred = custom_pred[..., hog_out_dim:]

    diff_mask = torch.abs(official_mask - custom_mask).sum().item()
    print(f"Mask Comparison: {diff_mask}")

    custom_hog_target = custom_model.target.hog(x)
    diff_hog_target = torch.abs(official_hog_target - custom_hog_target).sum().item()
    print(f"HOG Target Diff: {diff_hog_target:.2e}")

    diff_hog_pred = torch.abs(official_hog_pred - custom_hog_pred).max().item()
    print(f"HOG Prediction Max Diff: {diff_hog_pred:.2e}")
    diff_ndi_pred = torch.abs(official_ndi_pred - custom_ndi_pred).max().item()
    print(f"NDI Prediction Max Diff: {diff_ndi_pred:.2e}")

    #print(f"Official Loss: {official_loss.item():.6f}")
    #print(f"Custom Loss:   {custom_loss.item():.6f}")
    print(f"Difference:    {abs(official_loss - custom_loss).item():.2e}")

def sync_weights(official_model, custom_model):
    with torch.no_grad():
        # Encoder
        custom_model.cls_token.copy_(official_model.cls_token)
        custom_model.pos_embed.pos_embed_tensor.copy_(official_model.pos_embed[0])
        custom_model.patch_embed.proj.weight.copy_(official_model.patch_embed.proj.weight)
        custom_model.patch_embed.proj.bias.copy_(official_model.patch_embed.proj.bias)
        
        for i in range(12):
            custom_model.blocks[i].norm1.weight.copy_(official_model.blocks[i].norm1.weight)
            custom_model.blocks[i].norm1.bias.copy_(official_model.blocks[i].norm1.bias)
            custom_model.blocks[i].attn.qkv.weight.copy_(official_model.blocks[i].attn.qkv.weight)
            custom_model.blocks[i].attn.qkv.bias.copy_(official_model.blocks[i].attn.qkv.bias)
            custom_model.blocks[i].attn.proj.weight.copy_(official_model.blocks[i].attn.proj.weight)
            custom_model.blocks[i].attn.proj.bias.copy_(official_model.blocks[i].attn.proj.bias)
            custom_model.blocks[i].norm2.weight.copy_(official_model.blocks[i].norm2.weight)
            custom_model.blocks[i].norm2.bias.copy_(official_model.blocks[i].norm2.bias)
            custom_model.blocks[i].mlp.fc1.weight.copy_(official_model.blocks[i].mlp.fc1.weight)
            custom_model.blocks[i].mlp.fc1.bias.copy_(official_model.blocks[i].mlp.fc1.bias)
            custom_model.blocks[i].mlp.fc2.weight.copy_(official_model.blocks[i].mlp.fc2.weight)
            custom_model.blocks[i].mlp.fc2.bias.copy_(official_model.blocks[i].mlp.fc2.bias)
        
        custom_model.norm.weight.copy_(official_model.norm.weight)
        custom_model.norm.bias.copy_(official_model.norm.bias)

        # Decoder
        custom_model.decoder_embed.weight.copy_(official_model.decoder_embed.weight)
        custom_model.decoder_embed.bias.copy_(official_model.decoder_embed.bias)
        custom_model.mask_token.copy_(official_model.mask_token)
        custom_model.decoder_pos_embed.pos_embed_tensor.copy_(official_model.decoder_pos_embed[0])
        
        for i in range(8):
            custom_model.decoder_blocks[i].norm1.weight.copy_(official_model.decoder_blocks[i].norm1.weight)
            custom_model.decoder_blocks[i].norm1.bias.copy_(official_model.decoder_blocks[i].norm1.bias)
            custom_model.decoder_blocks[i].attn.qkv.weight.copy_(official_model.decoder_blocks[i].attn.qkv.weight)
            custom_model.decoder_blocks[i].attn.qkv.bias.copy_(official_model.decoder_blocks[i].attn.qkv.bias)
            custom_model.decoder_blocks[i].attn.proj.weight.copy_(official_model.decoder_blocks[i].attn.proj.weight)
            custom_model.decoder_blocks[i].attn.proj.bias.copy_(official_model.decoder_blocks[i].attn.proj.bias)
            custom_model.decoder_blocks[i].norm2.weight.copy_(official_model.decoder_blocks[i].norm2.weight)
            custom_model.decoder_blocks[i].norm2.bias.copy_(official_model.decoder_blocks[i].norm2.bias)
            custom_model.decoder_blocks[i].mlp.fc1.weight.copy_(official_model.decoder_blocks[i].mlp.fc1.weight)
            custom_model.decoder_blocks[i].mlp.fc1.bias.copy_(official_model.decoder_blocks[i].mlp.fc1.bias)
            custom_model.decoder_blocks[i].mlp.fc2.weight.copy_(official_model.decoder_blocks[i].mlp.fc2.weight)
            custom_model.decoder_blocks[i].mlp.fc2.bias.copy_(official_model.decoder_blocks[i].mlp.fc2.bias)
        
        custom_model.decoder_norm.weight.copy_(official_model.decoder_norm.weight)
        custom_model.decoder_norm.bias.copy_(official_model.decoder_norm.bias)

        # Official has self.decoder_pred and self.decoder_pred2
        # Custom has single self.decoder_pred
        custom_model.decoder_pred.weight.copy_(torch.cat([official_model.decoder_pred.weight, official_model.decoder_pred2.weight], dim=0))
        custom_model.decoder_pred.bias.copy_(torch.cat([official_model.decoder_pred.bias, official_model.decoder_pred2.bias], dim=0))

if __name__ == "__main__":
    compare()


