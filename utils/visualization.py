import cv2
import math
from matplotlib.colors import to_rgb, hsv_to_rgb
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

from datasets import ReBENDataset, BurnScarsDataset, Sen1Floods11Dataset, CashewPlantationDataset, C2SegDataset, MMHunanDataset


def quant_norm(data, lower_quant=0.01, upper_quant=0.99):
    """percentile-based stretching
    processes each sample (image) in the batch independently"""
    is_numpy = isinstance(data, np.ndarray)
    x = data.astype(np.float32, copy=True) if is_numpy else data.cpu().float().numpy()
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
    imgs = _ensure_batch(imgs)
    if ncols is None: ncols = math.ceil(math.sqrt(len(imgs)))

    num_imgs = len(imgs)
    nrows = (num_imgs + ncols - 1) // ncols

    plt.figure(figsize=(ncols*size, nrows*size))    
    for i in range(num_imgs):
        plt.subplot(nrows, ncols, i + 1)
        img = imgs[i]

        if img.shape[-1] == 1: # if segmentation mask
            plot_segmentation_mask(img, dataset_name=dataset_name)
        else:
            plt.imshow(img)
        if img_titles is not None: plt.title(img_titles[i])

        plt.axis('off')
    plt.tight_layout()
    if title is not None: plt.suptitle(title)
    plt.show()

def get_color_tensor(dataset_name=None, max_id=None, cmap_name="tab20"):
    """Creates a tensor of shape (max_class_id + 2, 3) for efficient RGB masking"""
    if dataset_name is None:
        if max_id is None: raise ValueError("max_id must be provided when dataset_name is None")
        
        n = max_id + 2 # +2 allows index -1 (ignore_index class) to access the last element safely

        if cmap_name == "custom":
            colors = np.array([hsv_to_rgb((i / n, 0.75, 0.95)) for i in range(n)])
        else:
            cmap_source = plt.get_cmap(cmap_name)
            colors = cmap_source(np.linspace(0, 1, n))[:, :3]
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
        case "mmhunan":
            id_to_class = MMHunanDataset.CLASSES
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

def prepare_segmentation_visuals(imgs, masks, preds, bands=None, dataset_name=None):
    color_tensor = get_color_tensor(dataset_name) if dataset_name is not None else \
                    get_color_tensor(max_id=preds.shape[1], cmap_name="custom") # num_classes
    color_tensor = color_tensor.to(masks.device)

    # imgs
    true_color_bands = ["B04", "B03", "B02"]
    band_indices = [bands.index(b) for b in true_color_bands if b
                    in bands] if bands is not None else [0, 1, 2]
    imgs_vis = quant_norm(imgs[:, band_indices, :, :])
    
    # gt
    gt_vis = color_tensor[masks].permute(0, 3, 1, 2) # (K, 3, H, W)

    # preds
    pred_classes = torch.argmax(preds, dim=1) # (B, H, W)
    pred_vis = color_tensor[pred_classes].permute(0, 3, 1, 2)

    return imgs_vis, gt_vis, pred_vis

def generate_saliency_grid(saliency_grid, image=None, alpha=0.5, vmin=None, vmax=None):
    if isinstance(saliency_grid, torch.Tensor):
        saliency_grid = saliency_grid.detach().cpu().float().numpy()
    
    # prepare background image
    img_min, img_max = image.min(), image.max()
    img_norm = (image - img_min) / (img_max - img_min + 1e-6)
    if img_norm.ndim == 2: 
        img_norm = np.stack([img_norm] * 3, axis=-1)

    # upscale grid using Nearest Neighbor to keep blocks sharp
    h, w = image.shape[:2]
    saliency_grid = saliency_grid.astype(np.float32)
    saliency_up = cv2.resize(saliency_grid, (w, h), interpolation=cv2.INTER_NEAREST)

    # apply 'jet' colormap
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(norm=norm, cmap='jet')
    saliency_rgb = sm.to_rgba(saliency_up)[:, :, :3]
    # alpha blend
    overlay = (1 - alpha) * img_norm + alpha * saliency_rgb
    
    return (overlay * 255).astype(np.uint8)

def generate_hog_image(hog_grid: torch.Tensor,
                       image: np.ndarray = None,
                       cell_size: int = 16) -> np.ndarray:
    """Render a HOG visualisation from a grid of HOG features.

    The function can either draw the orientation lines onto a blank canvas or
    overlay them on top of a provided image.

    Args:
        hog_grid (torch.Tensor): HOG feature tensor of shape (Channels, nbins, h, w).
            The channels are averaged before rendering.
        image (np.ndarray, optional): image of shape (H, W, C) or (H, W)
            to overlay the HOG lines on.  When provided, cell_size is
            inferred from the image dimensions and the HOG grid size.
            Defaults to None.
        cell_size (int): Side length (in pixels) of each HOG cell used when
            no overlay image is supplied.  Defaults to 16.

    Returns:
        np.ndarray: Rendered HOG image.

        * Shape (H, W, C) (uint8) when *image* is provided.
        * Shape (H, W) (uint8) when drawn on a blank canvas.
    """
    _, nbins, h_grid, w_grid = hog_grid.shape

    if image is not None:
        if isinstance(image, torch.Tensor):
            image = image.cpu().float().numpy()
            if image.ndim == 3:
                image = image.transpose(1, 2, 0)

        pool = image.shape[0] // h_grid
        if image.max() <= 1.0:
            hog_image = (image * 255).astype(np.uint8).copy()
        else:
            hog_image = image.astype(np.uint8).copy()
    else:
        pool = cell_size
        hog_image = np.zeros((h_grid * pool, w_grid * pool, 1), dtype=np.uint8)

    # Average over the channel dimension and move to CPU.
    cell_gradient = hog_grid.mean(dim=0).detach().cpu().float().numpy()  # (nbins, h, w)
    cell_gradient = np.abs(cell_gradient) # **2 # contrast boost
    cell_gradient = np.nan_to_num(cell_gradient, nan=0.0)

    cell_half = pool // 2
    max_mag = cell_gradient.max()
    angle_gap = 180.0 / nbins # HOG bins span [0°, 180°)

    for x in range(h_grid):
        for y in range(w_grid):
            cell_grad = cell_gradient[:, x, y] / (max_mag + 1e-8)
            for i, magnitude in enumerate(cell_grad):
                if magnitude < 0.01:
                    continue
                # +90° so lines are drawn along the edge direction (perpendicular to the gradient)
                angle_radian = math.radians(i * angle_gap + 90)
                cos_a = math.cos(angle_radian)
                sin_a = math.sin(angle_radian)
                cx = x * pool + cell_half
                cy = y * pool + cell_half
                x1 = cx + int(magnitude * cell_half * cos_a)
                y1 = cy + int(magnitude * cell_half * sin_a)
                x2 = cx - int(magnitude * cell_half * cos_a)
                y2 = cy - int(magnitude * cell_half * sin_a)
                intensity = int(255 * math.sqrt(magnitude))
                color = (intensity, intensity, intensity) if hog_image.shape[-1] == 3 else intensity
                cv2.line(hog_image, (y1, x1), (y2, x2), color=color, thickness=2 if hog_image.shape[0] > 200 else 1)
    return hog_image


def _ensure_batch(img):
    """
    Ensure input images are a list of (H, W, C) arrays suitable for plotting
    
    img: np.array of shape (H, W), (H, W, 1), (H, W, 3), (B, H, W, 1), (B, H, W, 3) or torch array with channel dim leading
    Returns: np.ndarray of shape (B, H, W, C)
    """
    if isinstance(img, torch.Tensor):
        img = img.cpu().float().numpy()
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
                img[i] = img[i].cpu().float().numpy()
                if img[i].ndim == 3:
                    img[i] = img[i].transpose(1, 2, 0)
                    
            if isinstance(img[i], np.ndarray):
                if img[i].ndim == 2:
                    img[i] = img[i][..., None]
    return img

def plot_segmentation_mask(mask, dataset_name=None):
    if isinstance(mask, torch.Tensor): 
        mask = mask.cpu().float().numpy()

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


SEMANTIC_COLORS = {
    "unlabeled": "#808080",              # neutral gray
    
    # Human Impact / Infrastructure
    "urban": "#e31a1c",                  # distinct red
    "industrial": "#6a3d9a",             # deep purple
    "street": "#333333",                 # dark charcoal
    "industrial_disturbance": "#4d004b", # very dark purple
    "burn_scar": "#7f0000",              # dark mahogany
    
    # Agriculture (Yellow/Brown Tones)
    "arable": "#fff5d7",                 # pale cream
    "permanent_crops": "#d95f02",        # burnt orange
    "pasture": "#ffff99",                # soft yellow
    "mixed_agriculture": "#fdbf6f",      # light orange-tan
    "agriculture_natural_mix": "#b2df8a",# pale lime
    "agroforestry": "#8c510a",           # deep earth brown
    
    # Natural Vegetation (Green Tones)
    "forest_broadleaf": "#33a02c",       # standard mid-green
    "forest_conifer": "#00441b",         # deep forest green
    "forest_mixed": "#1b7837",           # hunter green
    "natural_grassland": "#c2e699",      # light pastel green
    "heathland": "#78c679",              # medium-light green
    "shrubland": "#41ab5d",              # sage green
    "artificial_vegetation": "#a6d854",  # bright yellow-green
    
    # Water & Coast (Blue/Beige Tones)
    "beach_sand": "#f5f240",             # warm sand
    "wetland_inland": "#4eb3d3",         # bright sky blue
    "wetland_coastal": "#0868ac",        # deep ocean blue
    "water_inland": "#7fcdbb",           # turquoise-cyan
    "water_marine": "#084594",           # navy blue
}

CLASSES_TO_SEMANTIC = {
    # ReBEN
    "Unlabeled": "unlabeled",
    "Urban fabric": "urban",
    "Industrial or commercial units": "industrial",
    "Arable land": "arable",
    "Permanent crops": "permanent_crops",
    "Pastures": "pasture",
    "Complex cultivation patterns": "mixed_agriculture",
    "Land principally occupied by agriculture, with significant areas of natural vegetation": "agriculture_natural_mix",
    "Agro-forestry areas": "agroforestry",
    "Broad-leaved forest": "forest_broadleaf",
    "Coniferous forest": "forest_conifer",
    "Mixed forest": "forest_mixed",
    "Natural grassland and sparsely vegetated areas": "natural_grassland",
    "Moors, heathland and sclerophyllous vegetation": "heathland",
    "Transitional woodland, shrub": "shrubland",
    "Beaches, dunes, sands": "beach_sand",
    "Inland wetlands": "wetland_inland",
    "Coastal wetlands": "wetland_coastal",
    "Inland waters": "water_inland",
    "Marine waters": "water_marine",
    
    # Sen1Floods11
    "Missing data": "unlabeled",
    "No flood": "natural_grassland",
    "Flood": "wetland_coastal",
    
    # BurnScars
    #"Missing data": "unlabeled",
    "Not burned": "natural_grassland",
    "Burn scar": "burn_scar",

    # CashewPlantation
    "No data": "unlabeled",
    "Well-managed plantation": "permanent_crops",
    "Poorly-managed plantation": "agroforestry",
    "Non-plantation": "mixed_agriculture",
    "Residential": "urban",
    "Background_cashew": "agriculture_natural_mix",
    "Uncertain": "arable",

    # C2Seg
    "Background_c2seg": "unlabeled",
    "Surface water": "water_inland",
    "Street": "street",
    "Urban Fabric": "urban",
    "Industrial, commercial and transport": "industrial",
    "Mine, dump, and construction sites": "industrial_disturbance",
    "Artificial, vegetated areas": "artificial_vegetation",
    "Arable Land": "arable",
    "Permanent Crops": "permanent_crops",
    #"Pastures": "pasture",
    "Forests": "forest_mixed",
    "Shrub": "shrubland",
    "Open spaces with no vegetation": "natural_grassland",
    #"Inland wetlands": "wetland_inland",

    # MMHunan
    "Missing data": "unlabeled",
    "Cropland": "arable",
    "Forest": "forest_mixed",
    "Grassland": "natural_grassland",
    "Wetland": "wetland_inland",
    "Water": "water_inland",
    "Unused land": "beach_sand",
    "Built-up area": "urban"
}
