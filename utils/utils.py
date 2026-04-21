import json
import lightning as L
import logging
import os
import pandas as pd
from pathlib import Path
import subprocess
import sys
import torch

### hardcoded stuff ###

class CheckpointSyncCallback(L.Callback):
    def __init__(self, checkpoints_dir, sync_root="/content/drive/My Drive/saved_checkpoints/"):
        self.checkpoints_dir = Path(checkpoints_dir)
        self.dest_dir = Path(sync_root) / self.checkpoints_dir.name

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        self.dest_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            "rsync", "-av", "--include=*/", "--include=*.ckpt", "--exclude=*",
            f"{self.checkpoints_dir}/", f"{self.dest_dir}/"
        ], check=False)

class DebuggingCallback(L.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        logging.debug(f"Starting training epoch {trainer.current_epoch}...")

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if batch_idx == 0:
            logging.debug(f"Starting batch 0 of training epoch {trainer.current_epoch}...")

    def on_train_epoch_end(self, trainer, pl_module):
        logging.debug(f"Finished training epoch {trainer.current_epoch}.")

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        logging.debug(f"Saving checkpoint at epoch {trainer.current_epoch}...")

    def on_validation_epoch_start(self, trainer, pl_module):
        logging.debug(f"Starting validation epoch {trainer.current_epoch}...")

    def on_validation_epoch_end(self, trainer, pl_module):
        logging.debug(f"Finished validation epoch {trainer.current_epoch}.")


def load_pretrained_encoder_weights(model, ckpt_path, bands): # model is a SegmentationViT
    logging.info(f"Loading pretrained MAE weights from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]

    # drop pos_embed keys because they're fixed 2D sine-cosine embeddings
    raw_state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()
                      if k.startswith("model.") and "pos_embed" not in k}

    # slice patch_embed.proj.weight to match the downstream model's expected bands
    PRETRAINED_MODEL_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
    if list(bands) != PRETRAINED_MODEL_BANDS:
        if set(bands).issubset(PRETRAINED_MODEL_BANDS):
            band_indices = [PRETRAINED_MODEL_BANDS.index(b) for b in bands]
            logging.info(f"Slicing patch_embed.proj.weight: {band_indices}")
            raw_state_dict["patch_embed.proj.weight"] = raw_state_dict["patch_embed.proj.weight"][:, band_indices, :, :]
        else:
            logging.warning(f"At least one band is unknown in {bands}")

    # remap to the SegmentationViT's encoder structure
    adjusted_state_dict = {f"encoder.vit.{k}": v for k, v in raw_state_dict.items()}

    # preemptively strip shape mismatches
    target_state = model.state_dict()
    valid_state_dict = {}

    for k, v in adjusted_state_dict.items():
        if k in target_state:
            if v.shape == target_state[k].shape:
                valid_state_dict[k] = v
            else:
                logging.warning(f"Shape mismatch dropping '{k}': target={target_state[k].shape} vs ckpt={v.shape}")
        else:
            valid_state_dict[k] = v # Let strict=False deal with extra MAE parameters (decoder..)

    missing, unexpected = model.load_state_dict(valid_state_dict, strict=False)

    logging.debug(f"Missing keys: {missing}")
    logging.debug(f"Unexpected keys: {unexpected}")

def get_parameter_groups_llrd(model, weight_decay, backbone_lr, decoder_lr, layer_lr_decay): # model is a SegmentationViT
    """Creates parameter groups with Layer-wise Learning Rate Decay (LLRD)"""
    param_groups_by_name = {}
    param_groups_by_var = {}

    num_vit_blocks = len(model.encoder.vit.blocks)

    for param_name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # determine if weight decay should be applied
        if len(param.shape) == 1 or param_name.endswith(".bias") or "pos_embed" in param_name or "cls_token" in param_name:
            group_type = "no_decay"
            applied_weight_decay = 0.0
        else:
            group_type = "decay"
            applied_weight_decay = weight_decay

        # layer-wise lr for ViT backbone
        if param_name.startswith("encoder.vit."):
            if param_name.startswith("encoder.vit.patch_embed") or "cls_token" in param_name or "pos_embed" in param_name:
                layer_idx = 0  # stem / embeddings
            elif param_name.startswith("encoder.vit.blocks."):
                block_idx = int(param_name.split("encoder.vit.blocks.")[1].split(".")[0])
                layer_idx = block_idx + 1
            else:
                layer_idx = num_vit_blocks + 1  # final norm layer

            # lr decayed for earlier layers
            lr_scale = layer_lr_decay ** (num_vit_blocks + 1 - layer_idx)
            param_lr = backbone_lr * lr_scale
            group_name = f"{group_type}_backbone_layer_{layer_idx}"

        # adapter layers
        elif param_name.startswith("encoder.res_adapters."):
            param_lr = backbone_lr
            group_name = f"{group_type}_res_adapters"

        # decoder layers
        else:
            param_lr = decoder_lr
            group_name = f"{group_type}_decoder"

        # append parameter group
        if group_name not in param_groups_by_name:
            param_groups_by_name[group_name] = {
                "weight_decay": applied_weight_decay,
                "lr": param_lr,
                "params": [],
            }
            param_groups_by_var[group_name] = {
                "weight_decay": applied_weight_decay,
                "lr": param_lr,
                "params": [],
            }

        param_groups_by_name[group_name]["params"].append(param_name)
        param_groups_by_var[group_name]["params"].append(param)

    for group_name, group_dict in param_groups_by_name.items():
        logging.debug(f"Group: {group_name} | LR: {group_dict['lr']:.2e} | WD: {group_dict['weight_decay']} | Params: {len(group_dict['params'])}")

    return list(param_groups_by_var.values())

def get_checkpoint_path(experiment, dataset, artifacts_dir, wandb_project="RS_MAE-Thesis"):
    workspace_root = Path(__file__).parents[1]
    runs_json_path = workspace_root / "runs.json"
    
    with open(runs_json_path, "r") as f:
        runs_data = json.load(f)
    
    if dataset == "c2seg_wuhan": dataset = "c2seg_beijing"
    elif dataset == "c2seg_berlin": dataset = "c2seg_augsburg"
    run_id = runs_data.get(experiment, {}).get(dataset, {}).get("run_id")
    if not run_id:
        raise ValueError(f"Run ID not found for experiment='{experiment}', dataset='{dataset}' in {runs_json_path}")
    
    ckpt_dir = Path(artifacts_dir) / wandb_project / run_id / "checkpoints"
    
    def _find_best(directory):
        if not directory.exists():
            return None
        
        ckpts = list(directory.glob("seg-*.ckpt")) + list(directory.glob("mae-*.ckpt"))
        if not ckpts:
            return None
        
        best_ckpt, min_loss = None, float('inf')
        for ckpt in ckpts:
            loss_str = ckpt.stem.split("=")[-1]
            loss = float(loss_str)
            if loss < min_loss:
                min_loss = loss
                best_ckpt = ckpt
        
        return best_ckpt

    best_ckpt_path = _find_best(ckpt_dir)
    
    if not best_ckpt_path:
        logging.info(f"Checkpoint not found in {ckpt_dir}. Attempting to rsync from cluster...")
        
        remote_path = f"jacer@gateway.hpc.tu-berlin.de:/scratch/jacer/outputs/training_out/{wandb_project}/{run_id}"
        local_dest = Path(artifacts_dir) / wandb_project
        local_dest.mkdir(parents=True, exist_ok=True)
        try:
            # Clear LD_LIBRARY_PATH to avoid OpenSSL version mismatch in Conda/Miniforge env
            env = os.environ.copy()
            if "LD_LIBRARY_PATH" in env:
                del env["LD_LIBRARY_PATH"]
            subprocess.run(f'rsync -avP --exclude last.ckpt {remote_path} {str(local_dest)}', 
                           shell=True, executable="/bin/bash", check=True, env=env)
            logging.info("rsync completed successfully.")
            best_ckpt_path = _find_best(ckpt_dir)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Could not retrieve checkpoints from cluster for run_id {run_id}. Error: {e}")

    if not best_ckpt_path:
        raise FileNotFoundError(f"Could not find any valid checkpoints in {ckpt_dir} after rsync attempt.")
        
    return str(best_ckpt_path)

def save_results(results, dataset_name, experiment_name, ood_split=None, csv_path="results.csv"):
    """results: dict like {"acc":0.59, "iou":0.38, "f1":0.53, "loss":...}"""
    display_name = (dataset_name + ("_" + ood_split if ood_split else "")) \
        .replace("_", " ").title().strip()

    experiment_col = experiment_name.replace("_", " ").title().strip()

    csv_path = Path(csv_path)

    # load or initialize
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=["Dataset", "Metric"])

    # normalize
    df = df.astype(object)
    df["Dataset"] = df["Dataset"].astype(str).str.strip().replace(r"^\s*$", pd.NA, regex=True).ffill()
    df["Metric"] = df["Metric"].astype(str).str.strip().str.lower()

    # ensure column exists
    if experiment_col not in df.columns:
        df[experiment_col] = None

    # insert / update
    for metric_name, val in results.items():
        if metric_name == "loss":
            continue

        metric_name = metric_name.lower().strip()

        mask = (df["Dataset"] == display_name) & (df["Metric"] == metric_name)

        if mask.any():
            df.loc[mask, experiment_col] = f"{val*100:.2f}"
        else:
            new_row = {
                "Dataset": display_name,
                "Metric": metric_name,
                experiment_col: f"{val*100:.2f}"
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # keep table structured
    metric_order = ["acc", "iou", "f1"]
    df["Metric"] = pd.Categorical(df["Metric"], categories=metric_order, ordered=True)

    df = df.sort_values(by=["Dataset", "Metric"]).reset_index(drop=True)

    # blank repeated dataset names
    df_out = df.copy()
    df_out["Dataset"] = df_out["Dataset"].where(
        df_out["Dataset"] != df_out["Dataset"].shift(), "")

    df_out.to_csv(csv_path, index=False)

def setup_logging(level="info"):
    class ColorFormatter(logging.Formatter):
        COLORS = {
            logging.DEBUG: "\033[37m",
            logging.INFO: "",
            logging.WARNING: "\033[93m",
            logging.ERROR: "\033[31m",
        }
        RESET = "\033[0m"

        def __init__(self, fmt=None, datefmt=None):
            super().__init__(fmt=fmt, datefmt=datefmt)

        def format(self, record):
            msg = super().format(record)
            color = self.COLORS.get(record.levelno, "")
            return f"{color}{msg}{self.RESET}"


    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColorFormatter('[%(asctime)s] %(funcName)s(): %(message)s', datefmt='%H:%M:%S'))

    logger = logging.getLogger()
    logger.handlers.clear()
    logger.addHandler(handler)
    
    if level == "info":
        logger.setLevel(logging.INFO)
    elif level == "debug":
        logger.setLevel(logging.DEBUG)

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
    "beach_sand": "#fef0d9",             # warm sand
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
}
