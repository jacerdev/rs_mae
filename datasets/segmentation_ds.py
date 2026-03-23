import numpy as np
import torch
from torch.utils.data import Dataset

from .reben import ReBENDataset
from .data_loaders import load_reben_7k, load_burn_scars, load_sen1floods11, load_cashew_plantation, load_c2seg

class SegmentationDataset(Dataset):
    """stochastic grid-sampling with a translational per-epoch offset (relies on current_epoch being updated at the start of each epoch)"""
    def __init__(self, samples, masks, ids=None, grid_sampling=True, crop_size=None, min_margin=None, transform=None):
        """
        samples (list or np.ndarray): Raw image array(s) (H, W, C).
        masks (list or np.ndarray): Raw mask array(s) (H, W).
        ids (list): List of unique identifiers for each sample.
        grid_sampling: If True, enables stochastic grid cropping, otherwise return samples as is
        crop_size (tuple): Target crop dimensions (Height, Width).
        min_margin: Minimum margin to allow shifting
        """
        self.samples = samples
        self.masks = masks
        self.ids = ids
        self.grid_sampling = grid_sampling
        self.transform = transform

        if not self.grid_sampling:
            self.crop_size = self.current_epoch = self.margin_y = self.margin_x = self.grid_coords = self.offsets = None
            return

        self.crop_size = crop_size

        # precalculate grid coordinates and margins
        H, W = self.samples[0].shape[:2]
        if min_margin is None:
            max_y_crops = H // self.crop_size
            max_x_crops = W // self.crop_size
        else:
            max_y_crops = (H - min_margin) // self.crop_size
            max_x_crops = (W - min_margin) // self.crop_size
        
        self.margin_y = H - (max_y_crops * self.crop_size)
        self.margin_x = W - (max_x_crops * self.crop_size)
        
        # store the top-left anchor coords for every grid cell
        self.grid_coords = []
        for t_idx in range(len(self.samples)):
            for cy in range(max_y_crops):
                for cx in range(max_x_crops):
                    self.grid_coords.append((t_idx, cy * self.crop_size, cx * self.crop_size))

        # allocate shared memory for offsets so multiple workers can sample from the same grid
        self.offsets = torch.zeros((len(self.samples), 2), dtype=torch.int32).share_memory_()
        self.current_epoch = -1
        self.set_epoch(0)

    def set_epoch(self, epoch):
        """Generates grid offset per sample (must be called by the Custom Epoch Hook before every new epoch)"""
        if not self.grid_sampling or self.current_epoch == epoch:
            return

        self.current_epoch = epoch

        for t_idx in range(len(self.samples)):
            # Generate shifting offset applied to the entire grid of this sample
            off_y = np.random.randint(0, self.margin_y + 1)
            off_x = np.random.randint(0, self.margin_x + 1)
            self.offsets[t_idx, 0] = int(off_y)
            self.offsets[t_idx, 1] = int(off_x)

    def __len__(self):
        return len(self.grid_coords) if self.grid_sampling else len(self.samples)

    def __getitem__(self, idx):
        if not self.grid_sampling:
            img = self.samples[idx]
            mask = self.masks[idx]
        else:
            t_idx, base_y, base_x = self.grid_coords[idx]
            off_y, off_x = self.offsets[t_idx].tolist()
            
            y_start = base_y + off_y
            x_start = base_x + off_x
            
            img = self.samples[t_idx][y_start : y_start+self.crop_size , x_start : x_start+self.crop_size]
            mask = self.masks[t_idx][y_start : y_start+self.crop_size , x_start : x_start+self.crop_size]

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img, mask = augmented["image"], augmented["mask"]

        return img, mask

class ReBEN7kDataset(SegmentationDataset):
    CLASSES = ReBENDataset.CLASSES
    BANDS = ReBENDataset.BANDS

    def __init__(self, data_dir, transform=None,
                 crop_size=112, grid_sampling=True, min_margin=None,
                 **ReBENDataset_kwargs):
        
        samples, masks, ids = load_reben_7k(data_dir, **ReBENDataset_kwargs)
        
        super().__init__(
            samples=samples,
            masks=masks,
            ids=ids,
            grid_sampling=grid_sampling,
            crop_size=crop_size,
            min_margin=min_margin,
            transform=transform
        )

class Sen1Floods11Dataset(SegmentationDataset):
    CLASSES = {
        -1: "Missing data",
        0: "No flood",
        1: "Flood"
    }
    BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]

    def __init__(self, data_dir, split_file, split, bands=None, normalize=False, transform=None,
                 crop_size=112, grid_sampling=True, min_margin=None):
        
        samples, masks, ids = load_sen1floods11(data_dir, split_file, split, bands)

        if normalize: samples = _normalize_samples(samples, bands)

        super().__init__(
            samples=samples,
            masks=masks,
            ids=ids,
            grid_sampling=grid_sampling,
            crop_size=crop_size,
            min_margin=min_margin,
            transform=transform
        )
        
class BurnScarsDataset(SegmentationDataset):
    CLASSES = {
        -1: "Missing data",
        0: "Not burned",
        1: "Burn scar"
    }
    BANDS = ["B02", "B03", "B04", "B8A", "B11", "B12"]
    def __init__(self, data_dir, split_file, split, bands=None, normalize=False, transform=None,
                 crop_size=112, grid_sampling=True, min_margin=None):
        
        samples, masks, ids = load_burn_scars(data_dir, split_file, split, bands)

        if normalize: samples = _normalize_samples(samples, bands)

        super().__init__(
            samples=samples,
            masks=masks,
            ids=ids,
            grid_sampling=grid_sampling,
            crop_size=crop_size,
            min_margin=min_margin,
            transform=transform
        )

class CashewPlantationDataset(SegmentationDataset):
    # https://torchgeo.readthedocs.io/en/v0.6.2/api/datasets.html?utm_source=chatgpt.com#benin-cashew-plantations
    CLASSES = {
        -1: "No data",
        0: "Well-managed plantation",
        1: "Poorly-managed plantation",
        2: "Non-plantation",
        3: "Residential",
        4: "Background_cashew",
        5: "Uncertain"
    }
    BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]
    def __init__(self, data_dir, split_file, split, bands=None, normalize=False, transform=None,
                 crop_size=112, grid_sampling=True, min_margin=None):
        
        samples, masks, ids = load_cashew_plantation(data_dir, split_file, split, bands)

        if normalize: samples = _normalize_samples(samples, bands)

        super().__init__(
            samples=samples,
            masks=masks,
            ids=ids,
            grid_sampling=grid_sampling,
            crop_size=crop_size,
            min_margin=min_margin,
            transform=transform
        )

class C2SegDataset(SegmentationDataset):
    # https://openremotesensing.net/wp-content/uploads/2023/04/Datasets_and_Protocols_FINAL.pdf
    CLASSES = { # := originalmask - 1
        -1: "Background_c2seg",
        0: "Surface water",
        1: "Street",
        2: "Urban Fabric",
        3: "Industrial, commercial and transport",
        4: "Mine, dump, and construction sites",
        5: "Artificial, vegetated areas",
        6: "Arable Land",
        7: "Permanent Crops",
        8: "Pastures",
        9: "Forests",
        10: "Shrub",
        11: "Open spaces with no vegetation",
        12: "Inland wetlands",
    }
    BANDS = ["B02", "B03", "B04", "B08"]

    def __init__(self, data_dir, city, split, split_frac=.8, bands=None, normalize=False, transform=None,
                 crop_size=112, grid_sampling=True, min_margin=None):
        
        samples, masks, ids = load_c2seg(data_dir, city=city, split=split, split_frac=split_frac, key='MSI', bands=bands)
        
        if normalize: samples = _normalize_samples(samples, bands)
            
        super().__init__(
            samples=samples,
            masks=masks,
            ids=ids,
            grid_sampling=grid_sampling,
            crop_size=crop_size,
            min_margin=min_margin,
            transform=transform
        )

def _normalize_samples(samples, bands):
    """Normalizes samples in-place using ReBENDataset statistics for given bands"""
    mean = np.array([ReBENDataset.BAND_STATS["mean"][b] for b in bands], dtype=np.float32).reshape(1, 1, -1)
    std  = np.array([ReBENDataset.BAND_STATS["std"][b] for b in bands], dtype=np.float32).reshape(1, 1, -1)
    
    for i, sample in enumerate(samples):
        samples[i] = sample.astype(np.float32, copy=False)
        samples[i] -= mean
        samples[i] /= std
        
    return samples
