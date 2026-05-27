from pathlib import Path
import pandas as pd
import tifffile
import numpy as np
import scipy.io
import h5py

from .reben import ReBENDataset


def load_reben_7k(data_dir, keys_path=None,
                  split_file=None, split=None,
                  align_size=120, bands=None,
                  normalize=False):
    """Loads reben_7k dataset by wrapping ReBENDataset"""
    dataset = ReBENDataset(
        data_dir=data_dir,
        keys_path=keys_path,
        split_file=split_file,
        split=split,
        align_size=align_size,
        bands=bands,
        normalize=normalize,
        transform=None,
        val_to_train_frac=0.0,
        test_to_train_frac=0.0,
        seed=None,
        blocklist_path=None,
        return_mask=True
    )
    
    samples, masks = [], []

    # Load all images and masks into memory
    for i in range(len(dataset)):
        img_tensor, mask_tensor = dataset[i]
        
        # Convert tensors to NumPy
        img = img_tensor.numpy().transpose(1, 2, 0)
        if not normalize: img = img.astype(np.int16)
        mask = mask_tensor.numpy().astype(np.int8)
        
        samples.append(img)
        masks.append(mask)
        
    return samples, masks, dataset.keys

def load_sen1floods11(data_dir, split_file, split, bands=None):
    """Loads Sen1Floods11 dataset samples and masks into memory"""
    IMAGE_SUFFIX = "_S2Hand"
    LABEL_SUFFIX = "_LabelHand"
    ID_LENGTH = 0 # filename has no prefix so we just slice to start
    BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]
    
    data_dir = Path(data_dir)
    df = pd.read_csv(split_file)
    split_ids = set(df[df['split'] == split]['patch_id'].tolist())
    
    img_paths, mask_paths = _get_matched_tifs(data_dir, split_ids, IMAGE_SUFFIX, LABEL_SUFFIX, ID_LENGTH)
    
    samples, masks, ids = [], [], []
    for img_path, mask_path in zip(img_paths, mask_paths):
        img = tifffile.imread(img_path)
        mask = tifffile.imread(mask_path)

        if split != "test" and np.mean(mask != -1) < .35: # ignore_index = -1
            continue

        # Sen1Floods11 stored as C, H, W (transpose to standard H, W, C)
        img = img.transpose(1, 2, 0)
        
        img = _select_bands(img, BANDS, bands)
        
        samples.append(img)
        masks.append(mask.astype(np.int8))
        ids.append(img_path.stem)
        
    return samples, masks, ids

def load_burn_scars(data_dir, split_file, split, bands=None):
    """Loads Burn Scars dataset 512x512 samples and masks into memory"""
    IMAGE_SUFFIX = ".4_merged"
    LABEL_SUFFIX = ".4.mask"
    ID_LENGTH = 17 # e.g. T10SEJ.2018220.v1
    BANDS = ["B02", "B03", "B04", "B8A", "B11", "B12"]
    
    data_dir = Path(data_dir)
    df = pd.read_csv(split_file)
    split_ids = set(df[df['split'] == split]['patch_id'].tolist())
    
    img_paths, mask_paths = _get_matched_tifs(data_dir, split_ids, IMAGE_SUFFIX, LABEL_SUFFIX, ID_LENGTH)
    
    samples, masks, ids = [], [], []
    for img_path, mask_path in zip(img_paths, mask_paths):
        img = tifffile.imread(img_path)
        mask = tifffile.imread(mask_path)

        if split != "test" and np.mean(mask != -1) < .35: # ignore_index = -1
            continue

        # Burn scars natively sits in the 0-1 reflectance range
        img = (img * 10000.0).astype(np.int16) # multiply by 10000 to align with reBEN's scale
        
        img = _select_bands(img, BANDS, bands)
        
        samples.append(img)
        masks.append(mask.astype(np.int8))
        ids.append(img_path.stem)
        
    return samples, masks, ids

def load_cashew_plantation(data_dir, split_file, split, bands=None):
    """Loads m-Cashew-Plantation dataset samples and masks into memory"""
    BANDS = {'B01': 0, 'B02': 1, 'B03': 2, 'B04': 3, 'B05': 4, 'B06': 5, 'B07': 6, 'B08': 7,
            'B8A': 8, 'B09': 9, 'B10': 12, 'B11': 10, 'B12': 11}
    
    data_dir = Path(data_dir)
    df = pd.read_csv(split_file)
    split_ids = df[df['split'] == split]['patch_id'].tolist()
    
    if bands and (invalid := set(bands) - set(BANDS.keys())):
        raise ValueError(f"Requested band(s) {invalid} not found in available bands: {BANDS}")
    target_bands = bands if bands is not None else list(BANDS.keys())

    samples, masks, ids = [], [], []
    for patch_id in split_ids:
        file_path = data_dir / f"{patch_id}.hdf5"
        
        with h5py.File(file_path, "r") as h5file:
            keys = sorted(h5file.keys())
            target_keys = [keys[BANDS[b]] for b in target_bands]
            band_arrays = [np.array(h5file[k]) for k in target_keys]
            image = np.stack(band_arrays, axis=-1)
            mask = np.array(h5file["label"]) - 1 # decrement class ids by 1 to get ignore_index to -1

        if split != "test" and np.mean(mask != -1) < .35: # ignore_index = -1
            continue
        
        samples.append(image)
        masks.append(mask.astype(np.int8))
        ids.append(patch_id)
        
    return samples, masks, ids

def load_c2seg(data_dir, city, split, split_frac=.8, key='MSI', bands=None):
    """Loads one key ('MSI', 'SAR') for one city (augsburg, berlin, beijing, wuhan) and its mask"""
    BANDS = ["B02", "B03", "B04", "B08"]
    data_dir = Path(data_dir)
    city = city.lower()
    
    # German cities are saved with .mat format and scipy.io
    if city in ['augsburg', 'berlin']:
        data_path = data_dir / f'{city}_multimodal.mat'
        mat_data = scipy.io.loadmat(data_path)
        sample = mat_data[key]
        mask = mat_data['label']
        
    # Chinese cities are saved with HDF5 format and h5py
    elif city in ['beijing', 'wuhan']:
        data_path = data_dir / f'{city}.mat'
        mask_path = data_dir / f'{city}_label.mat'
        # HDF5 is (C, H, W). transpose it
        with h5py.File(data_path, 'r') as f:
            sample = np.array(f[key]).transpose()
        with h5py.File(mask_path, 'r') as fl:
            mask = np.array(fl['label']).transpose()
            
    else:
        raise ValueError(f"Unknown city: {city}. Expected one of 'augsburg', 'berlin', 'beijing', 'wuhan'.")

    if key == 'MSI':
        sample = _select_bands(sample, BANDS, bands)
    
    sample = sample.astype(np.int16)
    mask = (mask - 1).astype(np.int8) # decrement class ids by 1 to get ignore_index to -1

    # train/validation split: split vertically (west/east)
    split_idx = int(sample.shape[1] * split_frac)
    if split == 'train':
        sample = sample[:, :split_idx]
        mask = mask[:, :split_idx]
    elif split == 'validation':
        sample = sample[:, split_idx:]
        mask = mask[:, split_idx:]

    return [sample], [mask], [city]

def load_mmhunan(data_dir, split, bands=None):
    """Loads Hunan dataset samples and masks into memory"""
    BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]
    
    data_dir = Path(data_dir) / split
    s2_dir, lc_dir = data_dir / 's2', data_dir / 'lc'
    
    igbp2hunan = np.array([255, 0, 1, 2, 1, 3, 4, 6, 6, 5, 6, 7, 255])
    
    samples, masks, ids = [], [], []
    for img_path in sorted(s2_dir.iterdir()):
        basename = img_path.name
        mask_path = lc_dir / basename.replace("s2_", "lc_", 1)
        
        img = tifffile.imread(img_path)
        lc = tifffile.imread(mask_path)
        
        lc[lc == 255] = 12
        mask = igbp2hunan[lc]
        mask[mask == 255] = -1 # map the resulting 255 ignore index to -1
        mask = mask.astype(np.int8)
        
        if split != "test" and np.mean(mask != -1) < .35: # ignore_index = -1
            continue
        
        img = _select_bands(img, BANDS, bands)
        img = img.astype(np.int16)
        
        samples.append(img)
        masks.append(mask)
        ids.append(img_path.stem)
        
    return samples, masks, ids

def _select_bands(img, available_bands, requested_bands=None):
    """Selects requested bands from a given image array (H, W, C)"""
    if requested_bands is None:
        return img

    if requested_bands and (invalid := set(requested_bands) - set(available_bands)):
        raise ValueError(f"Requested band(s) {invalid} not found in available bands: {available_bands}")
   
    band_indices = [available_bands.index(b) for b in requested_bands]
    return img[..., band_indices]

def _get_matched_tifs(data_dir, split_ids, image_suffix, label_suffix, id_length):
    """Finds matched image and label files for a given list of patch IDs"""
    image_files, label_files = {}, {}
    for file_path in data_dir.glob("*.tif"):
        basename = file_path.name[:-4]
        
        if basename.endswith(image_suffix):
            name_without_suffix = basename[:-len(image_suffix)]
            patch_id = name_without_suffix[-id_length:]                
            if patch_id in split_ids:
                image_files[patch_id] = file_path
                
        elif basename.endswith(label_suffix):
            name_without_suffix = basename[:-len(label_suffix)]
            patch_id = name_without_suffix[-id_length:]                
            if patch_id in split_ids:
                label_files[patch_id] = file_path
                
    valid_ids = sorted(list(set(image_files.keys()).intersection(set(label_files.keys()))))
    return [image_files[pid] for pid in valid_ids], [label_files[pid] for pid in valid_ids]
