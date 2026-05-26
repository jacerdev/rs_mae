import lmdb
import logging
import pandas as pd
from pathlib import Path
import pickle
import random
from safetensors.torch import load as load_safetensors
import torch
from torch.utils.data import Dataset


class ReBENDataset(Dataset):
    # https://git.tu-berlin.de/rsim/fl-lrp/-/blob/main/utils/BENv2_utils.py
    BAND_STATS = { # Z-score normalization
        "mean": {   # 120_bilinear
            "B01": 360.64678955078125,
            "B02": 438.3720703125,
            "B03": 614.0556640625,
            "B04": 588.4096069335938,
            "B05": 942.7476806640625,
            "B06": 1769.8486328125,
            "B07": 2049.475830078125,
            "B08": 2193.2919921875,
            "B09": 2241.10595703125,
            "B11": 1568.2115478515625,
            "B12": 997.715087890625,
            "B8A": 2235.48681640625,
            "VH": -19.352558135986328,
            "VV": -12.643863677978516
        },
        "std": {    # 120_bilinear
            "B01": 563.1734008789062,
            "B02": 607.02685546875,
            "B03": 603.2968139648438,
            "B04": 684.56884765625,
            "B05": 727.5784301757812,
            "B06": 1087.4288330078125,
            "B07": 1261.4302978515625,
            "B08": 1369.3717041015625,
            "B09": 1294.35546875,
            "B11": 1063.9197998046875,
            "B12": 806.8846435546875,
            "B8A": 1342.490478515625,
            "VH": 5.590505599975586,
            "VV": 5.133493900299072,
        }
    }

    # https://bigearth.net/static/documents/Description_BigEarthNet_v2.pdf
    CLASSES = {
        -1: "Unlabeled",
        0: "Urban fabric",
        1: "Industrial or commercial units",
        2: "Arable land",
        3: "Permanent crops",
        4: "Pastures",
        5: "Complex cultivation patterns",
        6: "Land principally occupied by agriculture, with significant areas of natural vegetation",
        7: "Agro-forestry areas",
        8: "Broad-leaved forest",
        9: "Coniferous forest",
        10: "Mixed forest",
        11: "Natural grassland and sparsely vegetated areas",
        12: "Moors, heathland and sclerophyllous vegetation",
        13: "Transitional woodland, shrub",
        14: "Beaches, dunes, sands",
        15: "Inland wetlands",
        16: "Coastal wetlands",
        17: "Inland waters",
        18: "Marine waters"
    }

    CLC_TO_LULC = {
        111: 0,
        112: 0,
        121: 1,
        211: 2,
        212: 2,
        213: 2,
        221: 3,
        222: 3,
        223: 3,
        241: 3,
        231: 4,
        242: 5,
        243: 6,
        244: 7,
        311: 8,
        312: 9,
        313: 10,
        321: 11,
        333: 11,
        322: 12,
        323: 12,
        324: 13,
        331: 14,
        411: 15,
        412: 15,
        421: 16,
        422: 16,
        511: 17,
        512: 17,
        521: 18,
        522: 18,
        523: 18,
        122: -1,
        123: -1,
        124: -1,
        131: -1,
        132: -1,
        133: -1,
        141: -1,
        142: -1,
        332: -1,
        334: -1,
        335: -1,
        423: -1,
        999: -1,
    }

    BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]

    def __init__(self, data_dir, keys_path=None, # keys_path is to save/load keys to avoid scanning the entire LMDB
                 split_file=None, split=None, # split_file is a parquet file containing columns "patch_id", "split"
                 align_size=120, bands=None, # bands defaults to all reBEN bands
                 normalize=True, transform=None, # transform must be an albumentations.Compose
                 val_to_train_frac=0.0, test_to_train_frac=0.0, seed=42, # requires split_file
                 blocklist_path=None, # blocklist_path is a path to a csv/parquet file containing column "patch_id"
                 return_mask=False):
        super().__init__()

        self.data_dir = str(data_dir)
        self.split = split
        self.align_size = tuple(align_size) if hasattr(align_size, '__iter__') else (align_size, align_size)
        self.bands = bands if bands is not None else ReBENDataset.BANDS
        if any(b not in ReBENDataset.BANDS for b in self.bands): raise KeyError(f"At least one requested band is not available in reBEN.")
        self.normalize = normalize
        self.transform = transform
        self.return_mask = return_mask

        # load all tensor keys present in the LMDB
        self.env = None
        keys_path = Path(keys_path) if keys_path is not None else None
        if keys_path is not None and keys_path.exists() and keys_path.stat().st_mtime >= (Path(data_dir) / "data.mdb").stat().st_mtime:
            with keys_path.open("rb") as f:
                available_keys = set(pickle.load(f))
            logging.info(f"Cached {len(available_keys)} keys from {keys_path}")
        else:
            with lmdb.open(self.data_dir, readonly=True, lock=False, readahead=False, meminit=False) as env:
                with env.begin(write=False) as transaction:
                    with transaction.cursor() as cursor:
                        available_keys = set(key.decode('ascii') for key in cursor.iternext(keys=True, values=False))
                        logging.info(f"Loaded {len(available_keys)} keys from LMDB in {self.data_dir}")
            if keys_path is not None:
                with keys_path.open("wb") as f: pickle.dump(available_keys, f, protocol=pickle.HIGHEST_PROTOCOL)

        if self.normalize:
            self.mean = torch.tensor([ReBENDataset.BAND_STATS["mean"][b] for b in self.bands], dtype=torch.float32).view(-1, 1, 1)
            self.std = torch.tensor([ReBENDataset.BAND_STATS["std"][b] for b in self.bands], dtype=torch.float32).view(-1, 1, 1)

        if blocklist_path is not None:
            blocklist_df = (pd.read_csv(blocklist_path, usecols=["patch_id"]) if blocklist_path.endswith(".csv") else
                            pd.read_parquet(blocklist_path, columns=["patch_id"]))
            blocklist_keys = set(blocklist_df["patch_id"].tolist())
            available_keys = available_keys - blocklist_keys
            logging.info(f"Blocked {len(blocklist_keys)} keys from {blocklist_path}")

        if split_file is None or split is None:
            self.keys = sorted([k for k in available_keys if not k.endswith("_reference_map")])
        else:
            # filter for requested split if metadata is provided
            df = (pd.read_csv(split_file, usecols=["split", "patch_id"]) if split_file.endswith(".csv") else
                  pd.read_parquet(split_file, columns=["split", "patch_id"]))
            split_keys = set(df[df['split'] == split]["patch_id"]).intersection(available_keys)
            
            # Perform transfer from val/test set to train set if requested and valid
            if (val_to_train_frac > 0.0 or test_to_train_frac > 0.0) and split in ("train", "validation", "test"):
                if split == "train":
                    val_keys = set(df[df['split'] == "validation"]["patch_id"]).intersection(available_keys)
                    split_keys, _ = _apply_fractional_split(split_keys, val_keys, val_to_train_frac, seed)

                    test_keys = set(df[df['split'] == "test"]["patch_id"]).intersection(available_keys)
                    split_keys, _ = _apply_fractional_split(split_keys, test_keys, test_to_train_frac, seed)
                        
                elif split == "validation":
                    _, split_keys = _apply_fractional_split(set(), split_keys, val_to_train_frac, seed)
                        
                elif split == "test":
                    _, split_keys = _apply_fractional_split(set(), split_keys, test_to_train_frac, seed)
                        
            self.keys = sorted(list(split_keys))

        # lookup table to convert mask classes
        if self.return_mask:
            self.lut = torch.full((1000,), -1, dtype=torch.long) # -1 is the ignore_index
            
            if self.return_mask == "clc":
                # create a contiguous mapping
                valid_clc = sorted([k for k, v in ReBENDataset.CLC_TO_LULC.items() if v != -1])
                clc_to_contiguous = {clc: i for i, clc in enumerate(valid_clc)}
                for k in ReBENDataset.CLC_TO_LULC.keys():
                    self.lut[k] = clc_to_contiguous.get(k, -1) # -1 for ignored classes
            else:
                for k, v in ReBENDataset.CLC_TO_LULC.items():
                    self.lut[k] = v

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        if self.env is None: # lazy loading (very important to avoid multiprocessing errors)
             self.env = lmdb.open(self.data_dir, readonly=True, lock=False, readahead=False, meminit=False)
        
        key_str = self.keys[idx]
        with self.env.begin(write=False) as txn:
            byte_data = txn.get(key_str.encode('ascii'))
            tensor_dict = load_safetensors(byte_data)
        
        # build image stack
        band_tensors = []
        for band in self.bands:
            t = tensor_dict[band].float()
            if t.shape != self.align_size:
                t = torch.nn.functional.interpolate(t.unsqueeze(0).unsqueeze(0),
                        size=self.align_size, mode='bilinear', align_corners=False).squeeze()
            band_tensors.append(t)
        image = torch.stack(band_tensors, dim=0)

        # load segmentation mask if requested
        mask = None
        if self.return_mask:
            mask_key = f"{key_str}_reference_map"
            with self.env.begin(write=False) as txn:
                mask_bytes = txn.get(mask_key.encode('ascii'))
                if mask_bytes is None: raise ValueError(f"Mask missing for {key_str}")
                mask = load_safetensors(mask_bytes)["Data"].long().squeeze()

            # map to LULC Classes and align resolution
            mask = self.lut[mask]
            if mask.shape != self.align_size:
                mask = torch.nn.functional.interpolate(mask.float().unsqueeze(0).unsqueeze(0),
                            size=self.align_size, mode='nearest').squeeze().long()

        # apply transforms
        if self.transform is not None:
            image_np = image.cpu().numpy().transpose(1, 2, 0)
            if self.return_mask and mask is not None:
                mask_np = mask.cpu().numpy()
                augmented = self.transform(image=image_np, mask=mask_np)
                image, mask = augmented["image"], augmented["mask"]
            else:
                image = self.transform(image=image_np)["image"]

        if self.normalize:
            image = (image - self.mean) / self.std
        #image = torch.nan_to_num(image, nan=0.0)###################
        
        return (image, mask) if self.return_mask else image


def _apply_fractional_split(target_keys : set, source_keys : set, frac, seed=42):
    if frac <= 0.0:
        return target_keys, source_keys
    elif frac >= 1.0:
        return target_keys.union(source_keys), set()

    # sort to guarantee determinism
    target_keys = set(target_keys)
    source_keys_list = sorted(list(source_keys))

    # shuffle in-place
    rng = random.Random(seed)
    rng.shuffle(source_keys_list)

    # slice the list
    n_transfer = int(len(source_keys_list) * frac)
    transferred_keys = set(source_keys_list[:n_transfer])
    remaining_source_keys = set(source_keys_list[n_transfer:])

    return target_keys.union(transferred_keys), remaining_source_keys
            

