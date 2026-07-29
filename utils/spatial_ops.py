import math
import numpy as np
from pytorch_toolbelt.inference.tiles import TileMerger
from pytorch_toolbelt.inference.tiles import compute_pyramid_patch_weight_loss
import torch


@torch.inference_mode()
def windowed_inference(model, image, window_size, stride, num_classes, batch_size, merger_device=None):
    """ 
    Batched sliding window inference with dynamic overlap distribution.
    Requires batched (B, C, H, W) inputs.
    
    Instead of padding images that are not divisible by the stride, the window positions
    are recalculated to distribute the extra pixels across all overlaps.
    
    Args:
        model: A callable PyTorch model.
        image: The massive input tensor(s) to be stitched, shape (B, C, H, W).
        window_size (int): The spatial size of the crops fed into the model (i.e. model_input_size).
        stride (int): The step size between sliding windows.
        num_classes (int): Number of output channels/classes.
        batch_size (int): The number of `window_size` crops passed to the model simultaneously.
        merger_device: The device to accumulate the stitched prediction on (use "cpu" to prevent GPU OOM).
        
    Note:
        If `merger_device` is "cpu", the final stitched tensor of shape (B, num_classes, H, W) 
        is allocated in system RAM. If it is too massive, process images sequentially 
        instead of batching them.
    """
    if merger_device is None: merger_device = image.device
    device = next(model.parameters()).device
    
    B, C, h, w = image.shape
    
    # number of tiles needed
    nw = math.ceil((w - window_size) / stride) + 1
    nh = math.ceil((h - window_size) / stride) + 1

    # distribute the extra pixels across steps i.e. increase overlap instead of padding
    x_coords = np.linspace(0, w - window_size, nw, dtype=int)
    y_coords = np.linspace(0, h - window_size, nh, dtype=int)
    
    tasks = []
    for b in range(B):
        for y in y_coords:
            for x in x_coords:
                tasks.append((b, x, y, window_size, window_size))

    mergers = [TileMerger(
        image_shape=(h, w),
        channels=num_classes,
        weight=compute_pyramid_patch_weight_loss(window_size, window_size)[0],
        device=merger_device
    ) for _ in range(B)]
    
    model.eval()
    
    for i in range(0, len(tasks), batch_size):
        batch_tasks = tasks[i : i + batch_size]
        batch_tiles = [image[b, :, y:y+window_size, x:x+window_size] for b, x, y, _, _ in batch_tasks]
        batch_tensor = torch.stack(batch_tiles).to(device)

        logits = model(batch_tensor)
        if str(merger_device) == "cpu": logits = logits.cpu()
        
        # Group logits by image index to integrate efficiently
        b_indices = np.array([task[0] for task in batch_tasks])
        for b_idx in np.unique(b_indices):
            # Boolean mask extracting only the crops in this batch belonging to image b_idx
            mask = (b_indices == b_idx)
            b_logits = logits[mask]
            b_crops = np.array([task[1:] for j, task in enumerate(batch_tasks) if mask[j]])
            mergers[b_idx].integrate_batch(b_logits, b_crops)

    merged = torch.stack([m.merge() for m in mergers])
    return merged

def windowed_evaluation(preds, masks, tile_size, eval_batch_size=1, loss_fn=None, torchmetrics=None):
    """
    Evaluates massive prediction tensors by slicing them into memory-safe spatial tiles.
    
    Args:
        preds (torch.Tensor): Logits or predictions tensor, shape (B, C, H, W) or (C, H, W).
        masks (torch.Tensor): Ground truth labels tensor, shape (B, H, W) or (H, W).
        tile_size (int): The maximum side length of the spatial chunks the image will be sliced into.
        eval_batch_size (int): Number of images from the batch (B) to process simultaneously.
        loss_fn (callable, optional): Loss function to compute over the tiles.
        torchmetrics (list, optional): List of torchmetrics to update over the tiles.
        
    Note:
        To prevent Out-Of-Memory (OOM) errors on the GPU, ensure that the total pixels processed 
        at once (tile_size**2 * eval_batch_size) does not exceed the known maximum pixel capacity 
        of the hardware (e.g. model_input_size**2 * inference_batch_size).
    """
    if torchmetrics is None: torchmetrics = []
    device = masks.device
    
    # Ensure preds and masks are compatible with CPU slicing if large
    preds = preds.cpu()
    masks = masks.long().cpu()
        
    B, _, H, W = preds.shape
    h, w = (H // (H // tile_size + 1), W // (W // tile_size + 1))

    losses = []
    for y in range(0, H, h):
        for x in range(0, W, w):
            y_end = min(y + h, H)
            x_end = min(x + w, W)
            
            # Slice across batch dimension to prevent OOM on large B
            for i in range(0, B, eval_batch_size):
                preds_tile = preds[i : i+eval_batch_size, :, y:y_end, x:x_end].to(device)
                masks_tile = masks[i : i+eval_batch_size, y:y_end, x:x_end].to(device)
                
                if loss_fn is not None:
                    losses.append(loss_fn(preds_tile, masks_tile))
                for torchmetric in torchmetrics:
                    torchmetric.update(preds_tile, masks_tile)
                    
    loss = torch.stack(losses).mean() if loss_fn is not None else None
    return loss

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
