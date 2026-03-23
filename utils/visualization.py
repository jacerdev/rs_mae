import cv2
import math
from matplotlib.colors import to_rgb
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

from datasets import ReBENDataset, BurnScarsDataset, Sen1Floods11Dataset, CashewPlantationDataset, C2SegDataset
from utils.utils import CLASSES_TO_SEMANTIC, SEMANTIC_COLORS


def quant_norm(data, lower_quant=0.01, upper_quant=0.99):
    """percentile-based stretching
    processes each sample (image) in the batch independently"""
    is_numpy = isinstance(data, np.ndarray)
    x = data.astype(np.float32, copy=True) if is_numpy else data.cpu().numpy().astype(np.float32)
    np.nan_to_num(x, copy=False, nan=0.0, posinf=1.0, neginf=0.0)

    orig_ndim = x.ndim
    if orig_ndim == 3: x = x[None, ...]

    B = x.shape[0]
    for i in range(B):
        sample = x[i]
        mask = sample != 0  # avoid masked array allocation
        if not mask.any():
            continue

        vals = sample[mask]
        lq, uq = np.quantile(vals, (lower_quant, upper_quant))
        diff = uq - lq

        sample -= lq
        sample /= diff + 1e-8
        np.clip(sample, 0, 1, out=sample)

    if orig_ndim == 3: x = x[0]
    return x if is_numpy else torch.from_numpy(x).to(data.device)

def plot_images(imgs, ncols=None, title=None, img_titles=None, size=7, dataset_name=None):
    """Plot a list of images. if image has 2 dims then it's treated as a segmentation mask"""
    imgs = _ensure_image_list(imgs)
    if ncols is None: ncols = math.ceil(math.sqrt(len(imgs)))

    num_imgs = len(imgs)
    nrows = (num_imgs + ncols - 1) // ncols

    plt.figure(figsize=(ncols*size, nrows*size))    
    for i in range(num_imgs):
        plt.subplot(nrows, ncols, i + 1)
        img = imgs[i]

        #if img.shape[-1] == 1: # if segmentation mask
        #    plot_segmentation_mask(img, dataset_name=dataset_name)
        #else:
        plt.imshow(img)
        if img_titles is not None: plt.title(img_titles[i])

        plt.axis('off')
    plt.tight_layout()
    if title is not None: plt.suptitle(title)
    plt.show()


def plot_segmentation_mask(mask, dataset_name=None):
    if isinstance(mask, torch.Tensor): 
        mask = mask.cpu().numpy()

    mask = mask.squeeze()
    unique_ids = np.unique(mask)

    color_tensor = get_color_tensor(dataset_name=dataset_name, max_id=unique_ids.max())
    id_to_color = {uid: tuple(color_tensor[uid].tolist()) for uid in unique_ids}

    # mask is (H, W). color_tensor[mask] gives (H, W, 3) in [0, 1] range
    mask_tensor = torch.from_numpy(mask).long()
    rgb_mask = (color_tensor[mask_tensor].numpy() * 255).astype(np.uint8)

    # legend
    patches = [mpatches.Patch(color=id_to_color[uid], label=f"{uid}")
               for uid in id_to_color.keys()]

    plt.imshow(rgb_mask, interpolation="nearest")
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.axis("off")

def get_color_tensor(dataset_name=None, max_id=None, cmap_name="tab20"):
    """Creates a tensor of shape (max_class_id + 2, 3) for efficient RGB masking"""
    if dataset_name is None:
        if max_id is None: raise ValueError("max_id must be provided when dataset_name is None")
        cmap_source = plt.get_cmap(cmap_name)
        
        # +2 allows index -1 (ignore_index class) to access the last element safely
        colors = cmap_source(np.linspace(0, 1, max_id + 2))[:, :3]  # drop alpha
        return torch.tensor(colors, dtype=torch.float32)
        
    dataset_name = dataset_name.lower()
    match dataset_name:
        case "reben" | "reben_7k":
            id_to_class = ReBENDataset.CLASSES
        case "burn_scars":
            id_to_class = BurnScarsDataset.CLASSES
        case "sen1floods11":
            id_to_class = Sen1Floods11Dataset.CLASSES
        case "cashew_plantation":
            id_to_class = CashewPlantationDataset.CLASSES
        case "c2seg_augsburg" | "c2seg_berlin" | "c2seg_beijing" | "c2seg_wuhan":
            id_to_class = C2SegDataset.CLASSES
        case _:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
    valid_ids = list(id_to_class.keys())
    max_id = max(valid_ids)
    
    color_tensor = torch.zeros((max_id + 2, 3), dtype=torch.float32) # +2 for ignore_index (:= -1) class
    
    for uid in valid_ids:
        semantic_name = CLASSES_TO_SEMANTIC[id_to_class[uid]]
        hex_color = SEMANTIC_COLORS[semantic_name]
        color_tensor[uid] = torch.tensor(to_rgb(hex_color), dtype=torch.float32)
        
    return color_tensor

def _ensure_image_list(img):
    """
    Ensure input images are a list of (H, W, C) arrays suitable for plotting
    
    img: np.array of shape (H, W), (H, W, 1), (H, W, 3), (B, H, W, 1), (B, H, W, 3) or torch array with channel dim leading
    Returns: np.ndarray of shape (B, H, W, C)
    """
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
        if img.ndim == 4: img = img.transpose(0, 2, 3, 1)
        elif img.ndim == 3 and img.shape[0] <= 3:
            img = img.transpose(1, 2, 0)

    if isinstance(img, np.ndarray):
        if img.ndim == 4: img = list(img)
        elif img.ndim == 3:
            if img.shape[2] > 3: # list of masks
                img = list(img[..., None])
            else:
                img = [img]
        elif img.ndim == 2: img = [img[..., None]]
        else: raise ValueError(f"Unsupported shape: {img.shape}")

    elif isinstance(img, list):
        for i in range(len(img)):
            if isinstance(img[i], torch.Tensor):
                img[i] = img[i].cpu().numpy()
                if img[i].ndim == 3:
                    img[i] = img[i].transpose(1, 2, 0)
                    
            if isinstance(img[i], np.ndarray):
                if img[i].ndim == 2:
                    img[i] = img[i][..., None]
    return img

def plot_saliency_grid(saliency_grid, image=None):
    #fig, ax = plt.subplots(figsize=(6, 6))
    if image is not None:
        img_min = image.min()
        image = (image - img_min) / (image.max() - img_min + 1e-6)
        plt.imshow(image, cmap='gray')
        plt.axis('off')

    im = plt.imshow(saliency_grid, cmap='jet', alpha=0.5 if image is not None else 1.0,
                   extent=[0, image.shape[1], image.shape[0], 0] if image is not None else None)
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Saliency Score")
  
    #plt.grid(True)
    plt.show()

def generate_hog_image(hog_grid: torch.Tensor, image=None, cell_size=16) -> np.ndarray:
    """
    Generate HOG image according to HOG features.
    Args:
        hog_grid: HOG features with shape (C, nbins, h, w) (the input gets averaged over the channel dimension)
        image: image to overlay HOG features on with shape (H, W, 3)
    Returns:
        HOG image with shape (H, W, 1 or 3 if image is provided)
        lines are drawn perpendicular to gradients (parallel to edges)

    Reference: https://mmselfsup.readthedocs.io/en/1.x/_modules/mmselfsup/models/target_generators/hog_generator.html
    """
    if image is not None:
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
            image = image.transpose(1, 2, 0)

    _, nbins, h_grid, w_grid = hog_grid.shape
    if image is not None:
        hog_image = image.copy()
        pool = image.shape[0] // h_grid
        if image.dtype != 'uint8':
            hog_image = (hog_image * 255).astype(np.uint8)
    else:
        pool = cell_size
        hog_image = np.zeros((h_grid * pool, w_grid * pool, 1), dtype=np.uint8)

    cell_gradient = np.nan_to_num(np.abs(np.array(hog_grid.mean(dim=0).detach().cpu()))**1.75, nan=0)
    cell_width = pool // 2
    max_mag = np.array(cell_gradient).max()
    angle_gap = 180 / nbins # bug in the original implementation (should be 180, not 360)

    for x in range(cell_gradient.shape[1]):
        for y in range(cell_gradient.shape[2]):
            cell_grad = cell_gradient[:, x, y]
            cell_grad /= max_mag + 1e-8
            for i, magnitude in enumerate(cell_grad):
                if magnitude < 0.01:
                    continue
                angle = i * angle_gap
                angle_radian = math.radians(angle + 90) # +90 to align with edge instead of gradient
                x1 = cell_width + int(x * pool +
                            magnitude * cell_width * math.cos(angle_radian))
                y1 = cell_width + int(y * pool +
                            magnitude * cell_width * math.sin(angle_radian))
                x2 = cell_width + int(x * pool -
                            magnitude * cell_width * math.cos(angle_radian))
                y2 = cell_width + int(y * pool -
                            magnitude * cell_width * math.sin(angle_radian))

                val = int(255 * math.sqrt(magnitude))
                cv2.line(hog_image, (y1, x1), (y2, x2), thickness=2 if hog_image.shape[0] > 200 else 1,
                        color=(val, val, val) if hog_image.shape[-1] == 3 else val)
    return hog_image
