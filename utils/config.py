from dataclasses import dataclass
from typing import List, Any

# ==============================================================================
# Datasets Configurations
# ==============================================================================

@dataclass
class ReBENDataset_Cfg:
    data_dir: str
    keys_path: str
    split_file: str
    align_size: int
    bands: List[str]
    normalize: bool

    val_to_train_frac: float
    test_to_train_frac: float
    seed: int
    blocklist_path: str

@dataclass
class SegmentationDataset_Cfg:
    grid_sampling: bool
    crop_size: int
    min_margin: int

@dataclass
class ReBEN7kDataset_Cfg(SegmentationDataset_Cfg):
    data_dir: str
    keys_path: str
    split_file: str
    align_size: int
    bands: List[str]
    normalize: bool

@dataclass
class Sen1Floods11Dataset_Cfg(SegmentationDataset_Cfg):
    data_dir: str
    split_file: str
    bands: List[str]
    normalize: bool

@dataclass
class BurnScarsDataset_Cfg(SegmentationDataset_Cfg):
    data_dir: str
    split_file: str
    bands: List[str]
    normalize: bool

@dataclass
class CashewPlantationDataset_Cfg(SegmentationDataset_Cfg):
    data_dir: str
    split_file: str
    bands: List[str]
    normalize: bool

@dataclass
class C2SegDataset_Cfg(SegmentationDataset_Cfg):
    data_dir: str
    city: str
    split_frac: float
    bands: List[str]
    normalize: bool

# ==============================================================================
# Models Configurations
# ==============================================================================

@dataclass
class MaskedAutoencoderViT_Cfg:
    img_size: int

    # Encoder
    patch_size: int
    in_chans: int
    embed_dim: int
    depth: int
    num_heads: int

    # Decoder
    decoder_embed_dim: int
    decoder_depth: int
    decoder_num_heads: int

    # Model Hyperparameters
    mlp_ratio: float
    drop_path_rate: float
    cls_token: bool
    norm_layer: Any
    
    normalize: bool
    bands: List[str]

    # Instantiated Objects
    masker: Any
    target: Any
    loss: Any

@dataclass
class SegmentationViT_Cfg:
    # Segmentation head
    num_classes: int
    pyramid_channels: int
    segmentation_channels: int

    img_size: int

    # Encoder
    patch_size: int
    in_chans: int
    embed_dim: int
    depth: int
    num_heads: int

    # Model Hyperparameters
    mlp_ratio: float
    drop_path_rate: float
    cls_token: bool
    norm_layer: Any
    
    normalize: bool
    bands: List[str]
