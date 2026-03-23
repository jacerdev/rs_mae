import math
import numpy as np
from pytorch_toolbelt.inference.tiles import TileMerger
from pytorch_toolbelt.inference.tiles import compute_pyramid_patch_weight_loss
import torch
from tqdm import tqdm


@torch.inference_mode()
def sliding_window_inference(model, image, window_size, stride, num_classes, batch_size, merger_device=None):
    """ 
    Batched sliding window inference with dynamic overlap distribution

    Instead of padding images that are not divisible by the stride, the window positions
    are recalculated to distribute the extra pixels across all overlaps
    """
    if merger_device is None: merger_device = image.device
    device = image.device
    _, h, w = image.shape
    
    # number of tiles needed
    nw = math.ceil((w - window_size) / stride) + 1
    nh = math.ceil((h - window_size) / stride) + 1

    # distribute the extra pixels across steps i.e. increase overlap instead of padding
    x_coords = np.linspace(0, w - window_size, nw, dtype=int)
    y_coords = np.linspace(0, h - window_size, nh, dtype=int)
    crops = []
    for y in y_coords:
        for x in x_coords:
            crops.append((x, y, window_size, window_size))

    merger = TileMerger(
        image_shape=(h, w),
        channels=num_classes,
        weight=compute_pyramid_patch_weight_loss(window_size, window_size)[0],
        device=merger_device
    )
    model.eval()
    
    crops_iter = range(0, len(crops), batch_size)
    if len(crops_iter) > 20: crops_iter = tqdm(crops_iter)
    for i in crops_iter:
        batch_crops = crops[i : i + batch_size]
        batch_tiles = [image[:, y:y+window_size, x:x+window_size] for x, y, _, _ in batch_crops]
        batch_tensor = torch.stack(batch_tiles).to(device)

        logits = model(batch_tensor)
        if str(merger_device) == "cpu": logits = logits.cpu()
        merger.integrate_batch(logits, np.array(batch_crops))

    return merger.merge()
    
def patchify(imgs, p):
    # (B, C, H, W) -> (B, N, D)
    B, C, H, W = imgs.shape
    assert H % p == 0 and W % p == 0

    h, w = H // p, W // p
    x = imgs.reshape(shape=(B, C, h, p, w, p))
    x = torch.einsum('bchpwq->bhwpqc', x)
    x = x.reshape(shape=(B, h * w, p ** 2 * C))
    return x

def unpatchify(tokens, p, H, W, C):
    # (B, N, D) -> (B, C, H, W)
    h, w = H // p, W // p  # grid size
    assert h * w == tokens.shape[1]

    tokens = tokens.reshape(shape=(tokens.shape[0], h, w, p, p, C))
    tokens = torch.einsum('bhwpqc->bchpwq', tokens)
    imgs = tokens.reshape(shape=(tokens.shape[0], C, H, W))
    return imgs

def z_normalize(x):
    """Applies per-patch Z-score normalization"""
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True)
    return (x - mean) / (var + 1e-6).sqrt()

def z_denormalize(pred, raw_target):
    """Reverses per-patch Z-score normalization using stats from the raw target"""
    mean = raw_target.mean(dim=-1, keepdim=True)
    var = raw_target.var(dim=-1, keepdim=True)
    return pred * (var + 1e-6).sqrt() + mean
