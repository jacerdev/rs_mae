import json
import lightning as L
import logging
import os
from pathlib import Path
import subprocess
import sys

### hardcoded stuff ###

def get_checkpoint_path(experiment, dataset_name, artifacts_dir, wandb_project="RS_MAE-Thesis"):
    workspace_root = Path(__file__).parents[1]
    runs_json_path = workspace_root / "runs.json"
    
    with open(runs_json_path, "r") as f:
        runs_data = json.load(f)
    
    if dataset_name == "c2seg_wuhan": dataset_name = "c2seg_beijing"
    elif dataset_name == "c2seg_berlin": dataset_name = "c2seg_augsburg"
    run_id = runs_data.get(experiment, {}).get(dataset_name, {}).get("run_id")
    if not run_id:
        raise ValueError(f"Run ID not found for experiment='{experiment}', dataset_name='{dataset_name}' in {runs_json_path}")
    
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
