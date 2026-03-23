import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from dataclasses import asdict
import hydra
from hydra.utils import instantiate
import logging
from omegaconf import DictConfig
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy, JaccardIndex, F1Score
from tqdm import tqdm

from datasets import ReBEN7kDataset, BurnScarsDataset, Sen1Floods11Dataset, C2SegDataset, CashewPlantationDataset
from models.segmentation.segmentation_vit import SegmentationViT
from utils.config import SegmentationViT_Cfg, ReBEN7kDataset_Cfg, BurnScarsDataset_Cfg, Sen1Floods11Dataset_Cfg, C2SegDataset_Cfg, CashewPlantationDataset_Cfg
from utils.tensor_ops import sliding_window_inference
from utils.utils import get_checkpoint_path
from utils.visualization import quant_norm, get_color_tensor, plot_images


class EvaluationDataset(torch.utils.data.Dataset):
    def __init__(self, ds):
        self.ds = ds
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        img, mask = self.ds[idx]
        return self.ds.ids[idx], img, mask


@hydra.main(version_base="1.3", config_path="../configs", config_name="test")
def main(cfg: DictConfig):
    # Dataset
    test_transform = A.Compose([ToTensorV2()])

    match cfg.dataset_name:
        case "reben_7k":
            dataset_class = ReBEN7kDataset
            dataset_class_cfg = ReBEN7kDataset_Cfg
            test_transform = A.Compose([A.PadIfNeeded(min_height=cfg.img_size, min_width=cfg.img_size,
                                         border_mode=cv2.BORDER_REFLECT), ToTensorV2()])
        case "burn_scars":
            dataset_class = BurnScarsDataset
            dataset_class_cfg = BurnScarsDataset_Cfg

        case "sen1floods11":
            dataset_class = Sen1Floods11Dataset
            dataset_class_cfg = Sen1Floods11Dataset_Cfg

        case "cashew_plantation":
            dataset_class = CashewPlantationDataset
            dataset_class_cfg = CashewPlantationDataset_Cfg

        case "c2seg_augsburg" | "c2seg_berlin" | "c2seg_beijing" | "c2seg_wuhan":
            dataset_class = C2SegDataset
            dataset_class_cfg = C2SegDataset_Cfg

        case _:
            raise ValueError(f"Unknown dataset name extracted from wandb.run_name: {cfg.dataset_name}")

    # Data
    dataset_cfg = dataset_class_cfg(**cfg.dataset)
    dataset_kwargs = {**asdict(dataset_cfg), "grid_sampling": False}
    ood_split = cfg.get("ood_split", None)
    test_ds = dataset_class(**dataset_kwargs, split=ood_split if ood_split else "test", transform=test_transform)
    logging.info(f"Test set size: [{len(test_ds)}, {tuple(test_ds[0][0].shape)}]")

    eval_ds = EvaluationDataset(test_ds)
    test_loader = DataLoader(
        eval_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        prefetch_factor=cfg.prefetch_factor,
        persistent_workers=cfg.num_workers > 0,
        pin_memory=True)

    # Cache Check
    cache_dir = Path(cfg.artifacts_dir) / cfg.experiment_name / (cfg.dataset_name + ("_" + ood_split if ood_split else ""))
    use_cache = cache_dir.exists() and any(p.suffix == ".pt" for p in cache_dir.iterdir())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if use_cache:
        logging.info(f"Using cached results from {cache_dir}. Delete them to re-run inference.")
        model = None
    else:
        # Model
        model_cfg = SegmentationViT_Cfg(
            **{**dict(cfg.model), "norm_layer": instantiate(cfg.model.norm_layer)}
        )
        model = SegmentationViT(**asdict(model_cfg))

        # Checkpoint
        ckpt_path = get_checkpoint_path(cfg.experiment_name, cfg.dataset_name, cfg.artifacts_dir)
        try:
            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state_dict = state.get("state_dict", state)
            state_dict = {k.removeprefix("model."): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            logging.info(f"Loaded checkpoint from {ckpt_path}")
        except Exception as e:
            logging.error(f"Failed to load checkpoint from {ckpt_path}: {e}")

        model = model.to(device).eval()

    # Metrics
    num_classes = cfg.num_classes
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    acc  = MulticlassAccuracy(num_classes=num_classes, ignore_index=-1, average='micro').to(device)
    iou  = JaccardIndex(task="multiclass", num_classes=num_classes, ignore_index=-1).to(device)
    f1   = F1Score(task="multiclass", num_classes=num_classes, ignore_index=-1, average='macro').to(device)

    # Inference
    total_loss = 0.0
    cache_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch_idx, (sample_ids, imgs, masks) in tqdm(enumerate(test_loader)):
            imgs  = imgs.to(device)
            masks = masks.long().to(device)
            _, _, H, W = imgs.shape

            input_size = cfg.img_size

            if use_cache:
                batch_preds = []
                for i in range(len(imgs)):
                    sample_id = sample_ids[i]
                    cache_path = cache_dir / f"{sample_id}.pt"
                    pred = torch.load(cache_path, map_location='cpu', weights_only=False)
                    batch_preds.append(pred)
                preds = torch.cat(batch_preds, dim=0) # (B, C, H, W) on CPU
            else:
                if H <= input_size and W <= input_size:
                    preds = model(imgs) # (B, C, H, W) on device
                else:
                    preds_list = []
                    for sample in imgs:
                        pred = sliding_window_inference(model=model, image=sample,
                                                     window_size=input_size,
                                                     stride=input_size // 2,
                                                     num_classes=num_classes,
                                                     batch_size=cfg.batch_size,
                                                     merger_device='cpu')
                        preds_list.append(pred)
                    preds = torch.stack(preds_list, dim=0) # (B, C, H, W) on CPU
                # Save each result
                for i in range(len(imgs)):
                    sample_id = sample_ids[i]
                    torch.save(preds[i:i+1].cpu().clone(), cache_dir / f"{sample_id}.pt")

            # Evaluation
            if H <= input_size and W <= input_size:
                preds = preds.to(device)
                batch_loss = loss_fn(preds, masks).item() if loss_fn else 0.0
                for torchmetric in [acc, iou, f1]:
                    torchmetric.update(preds, masks)
            else:
                batch_loss_tensor = tiled_evaluation(
                    preds=preds, # on CPU
                    masks=masks,
                    input_size=input_size,
                    num_classes=num_classes,
                    batch_size=cfg.batch_size,
                    loss_fn=loss_fn,
                    torchmetrics=[acc, iou, f1]
                )
                batch_loss = batch_loss_tensor.item() if batch_loss_tensor is not None else 0.0

            total_loss += batch_loss

            #imgs_vis, gt_vis, pred_vis = create_visuals(imgs.detach().cpu(), masks.detach().cpu(), preds.detach().cpu(),
            #                                            bands=dataset_cfg.bands, dataset_name=cfg.dataset_name)
            #plot_images(torch.stack([imgs_vis, gt_vis, pred_vis], dim=1).flatten(0,1), ncols=3)

    results = {f"loss": total_loss / len(test_loader),
               f"acc":  acc.compute().item(),
               f"iou":  iou.compute().item(),
               f"f1":   f1.compute().item()}
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    # Save to results.csv
    try:
        display_name = (cfg.dataset_name + ("_" + ood_split if ood_split else "")).replace("_", " ").title()
        experiment_col = cfg.experiment_name.replace("_", " ").title()

        csv_path = Path("results.csv")
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df = df.astype(object) 
            df['Dataset'] = df['Dataset'].ffill()
        else:
            df = pd.DataFrame(columns=["Dataset", "Metric"])

        if experiment_col not in df.columns:
            df[experiment_col] = None
        
        df = df.reset_index(drop=True)

        for metric_name, val in results.items():
            if metric_name == "loss": continue
            mask = (df["Dataset"] == display_name) & (df["Metric"] == metric_name)
            if mask.any():
                df.loc[mask, experiment_col] = f"{val*100.0:.2f}"
            else:
                new_row = {"Dataset": display_name, "Metric": metric_name, experiment_col: f"{val*100.0:.2f}"}
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        # Blank out redundant Dataset names for final formatting
        df_out = df.copy()
        df_out['Dataset'] = df_out['Dataset'].where(df_out['Dataset'] != df_out['Dataset'].shift(), "")
        
        df_out.to_csv(csv_path, index=False)
        logging.info(f"Saved results to results.csv")
    except Exception as e:
        logging.warning(f"Could not save results to results.csv: {e}")

    return results

def tiled_evaluation(preds, masks, input_size, num_classes, batch_size=1,
                   loss_fn=None, torchmetrics=None):
    if torchmetrics is None: torchmetrics = []
    device = masks.device
    
    # Ensure preds and masks are compatible with CPU slicing if large
    preds = preds.cpu()
    masks = masks.long().cpu()
    
    _, _, H, W = preds.shape

    # Tile size logic
    tile_size = int(input_size * batch_size**0.5)
    h, w = (H // (H // tile_size + 1), W // (W // tile_size + 1))

    losses = []
    for y in range(0, H, h):
        for x in range(0, W, w):
            y_end = min(y + h, H)
            x_end = min(x + w, W)
            preds_tile = preds[:, :, y:y_end, x:x_end].to(device)
            masks_tile = masks[:, y:y_end, x:x_end].to(device)
            
            if loss_fn is not None:
                losses.append(loss_fn(preds_tile, masks_tile))
            for torchmetric in torchmetrics:
                torchmetric.update(preds_tile, masks_tile)
                    
    loss = torch.stack(losses).mean() if loss_fn is not None else None
    return loss

def create_visuals(imgs, masks, preds, bands=None, dataset_name=None):
        # get predicted classes
        pred_classes = torch.argmax(preds, dim=1) # (B, H, W)

        true_color_bands = ["B04", "B03", "B02"]
        band_indices = [bands.index(b) for b in true_color_bands if b
                        in bands] if bands is not None else [0, 1, 2]

        imgs_vis = quant_norm(imgs[:, band_indices, :, :])
        
        num_classes = preds.shape[1]
        color_tensor = get_color_tensor(dataset_name) if dataset_name is not None else \
                       get_color_tensor(max_id=num_classes)
        # mask.shape = (K, H, W) ==> color_tensor[mask].shape = (K, H, W, 3)
        gt_vis = color_tensor[masks].permute(0, 3, 1, 2) # (K, 3, H, W)
        pred_vis = color_tensor[pred_classes].permute(0, 3, 1, 2)

        return imgs_vis, gt_vis, pred_vis

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    main()
