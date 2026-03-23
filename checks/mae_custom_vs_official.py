import sys
from pathlib import Path
import torch
import torch.nn as nn
from functools import partial

PROJECT_PATH = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_PATH))
TEST_PATH = PROJECT_PATH / "sandbox"
sys.path.append(str(TEST_PATH))
FACEBOOK_MAE_PATH = PROJECT_PATH / "sandbox" / "mae"
sys.path.append(str(FACEBOOK_MAE_PATH))

from mae.models_mae import MaskedAutoencoderViT as official_mae
from models.backbones.mae_vit import MaskedAutoencoderViT as custom_mae
from models.strategies.target import NormalizedPixelTarget

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_vit_large(mae_cls, **kwargs):
    return mae_cls(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

def main():
    print(f"Running sanity check on {device}...")
    img = torch.randn(2, 3, 224, 224).to(device)
    
    checkpoint_path = "/mnt/ssd/mae_visualize_vit_large.pth" # https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_large.pth
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

    # Official Model
    off_model = init_vit_large(official_mae, in_chans=3, norm_pix_loss=True).to(device).eval()
    off_model.load_state_dict(state_dict)

    with torch.no_grad():
        torch.manual_seed(42)
        o_latent, o_mask, o_ids = off_model.forward_encoder(img, mask_ratio=0.75)
        torch.manual_seed(42)
        o_loss, o_pred, _ = off_model(img, mask_ratio=0.75)
        
        torch.manual_seed(42)
        o_latent_nomask, _, o_ids_nomask = off_model.forward_encoder(img, mask_ratio=0.0)

    # Free memory
    del off_model
    torch.cuda.empty_cache()

    # Custom Model (Full)
    cust_model_full = init_vit_large(
        custom_mae, in_chans=3, use_decoder=True, 
        target=NormalizedPixelTarget(patch_size=16, in_chans=3)
    ).to(device).eval()
    
    if state_dict:
        new_state_dict = {}
        for k, v in state_dict.items():
            if k == "pos_embed": new_state_dict["pos_embed.pos_embed_tensor"] = v.squeeze(0)
            elif k == "decoder_pos_embed": new_state_dict["decoder_pos_embed.pos_embed_tensor"] = v.squeeze(0)
            else: new_state_dict[k] = v
        cust_model_full.load_state_dict(new_state_dict, strict=True)

    with torch.no_grad():
        torch.manual_seed(42)
        c_latent, c_mask, c_ids = cust_model_full.forward_encoder(img, mask_ratio=0.75)
        torch.manual_seed(42)
        c_loss, c_pred, _ = cust_model_full(img, mask_ratio=0.75)

    print("Test 1: Encoder Output (mask_ratio=0.75)")
    assert (o_latent - c_latent).abs().max().item() < 1e-5
    
    print("Test 2: Full Forward Pass (Reconstruction & Loss)")
    assert (o_pred - c_pred).abs().max().item() < 1e-5

    # Free memory
    del cust_model_full
    torch.cuda.empty_cache()

    # Custom Model (Encoder Only)
    cust_model_enc = init_vit_large(
        custom_mae, in_chans=3, use_decoder=False
    ).to(device).eval()

    if state_dict:
        new_state_dict_enc = {}
        for k, v in state_dict.items():
            if k == "pos_embed": 
                new_state_dict_enc["pos_embed.pos_embed_tensor"] = v.squeeze(0)
            elif "decoder" not in k and "mask_token" not in k:
                new_state_dict_enc[k] = v
        cust_model_enc.load_state_dict(new_state_dict_enc, strict=True)

    with torch.no_grad():
        torch.manual_seed(42)
        c_enc_latent = cust_model_enc.forward_encoder(img, mask_ratio=0.0)

        # official MAE always randomly shuffles its tokens (even with mask_ratio=0)
        # custom's `use_decoder=False` implementation explicitly preserves the spatial order
        # to compare, unshuffle official latent tokens back
        B, L, D = o_latent_nomask.shape
        # Separate CLS token and spatial tokens
        o_cls = o_latent_nomask[:, :1, :]
        o_spatial = o_latent_nomask[:, 1:, :]

        # Unshuffle spatial tokens using the restore indices returned by the official encoder
        o_spatial_unshuffled = torch.gather(o_spatial, dim=1, index=o_ids_nomask.unsqueeze(-1).expand(-1, -1, D))
        # Combine back
        o_latent_nomask_unshuffled = torch.cat([o_cls, o_spatial_unshuffled], dim=1)

        enc_latent_diff = (o_latent_nomask_unshuffled - c_enc_latent).abs().max().item()

    print("Test 3: Encoder-Only Mode (use_decoder=False, mask_ratio=0.0)")
    assert enc_latent_diff < 1e-5

    print("\n✅")

if __name__ == "__main__":
    main()
