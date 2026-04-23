import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from dataclasses import asdict
import hydra
from hydra.utils import instantiate
import logging
from omegaconf import DictConfig
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy, JaccardIndex, F1Score
from tqdm import tqdm

from datasets import ReBEN7kDataset, BurnScarsDataset, Sen1Floods11Dataset, C2SegDataset, CashewPlantationDataset
from models.segmentation.segmentation_vit import SegmentationViT
from utils.config import SegmentationViT_Cfg, ReBEN7kDataset_Cfg, BurnScarsDataset_Cfg, Sen1Floods11Dataset_Cfg, C2SegDataset_Cfg, CashewPlantationDataset_Cfg
from utils.spatial_ops import windowed_inference, windowed_evaluation
from utils.utils import get_checkpoint_path, save_results
from utils.visualization import plot_images, prepare_segmentation_visuals

batch_size = 128
num_workers = 4
prefetch_factor = 2
artifacts_dir = "/mnt/ssd/trained_models/"

class EvaluationDataset(torch.utils.data.Dataset):
    def __init__(self, ds):
        self.ds = ds
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        img, mask = self.ds[idx]
        return self.ds.ids[idx], img, mask


@hydra.main(version_base="1.3", config_path="../configs", config_name="finetune")
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

        case name if name.startswith("c2seg"):
            dataset_class = C2SegDataset
            dataset_class_cfg = C2SegDataset_Cfg

        case _:
            raise ValueError(f"Unknown dataset name extracted from wandb.run_name: {cfg.dataset_name}")

    # Data
    dataset_cfg = dataset_class_cfg(**cfg.dataset)
    ood_split = cfg.get("ood_split", None)
    test_ds = dataset_class(**{**asdict(dataset_cfg), "grid_sampling": False},
                            split=ood_split if ood_split else "test", transform=test_transform)
    logging.info(f"Test set size: [{len(test_ds)}, {tuple(test_ds[0][0].shape)}]")

    eval_ds = EvaluationDataset(test_ds)
    test_loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=num_workers > 0,
        pin_memory=True)

    # Cache Check
    cache_dir = Path(artifacts_dir) / "results" / cfg.experiment_name / (cfg.dataset_name + ("_" + ood_split if ood_split else ""))
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
        ckpt_path = get_checkpoint_path(cfg.experiment_name, cfg.dataset_name, artifacts_dir)
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
                        pred = windowed_inference(model=model, image=sample,
                                                window_size=input_size,
                                                stride=input_size // 2,
                                                num_classes=num_classes,
                                                batch_size=batch_size,
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
                batch_loss_tensor = windowed_evaluation(preds=preds, # on CPU
                                                        masks=masks,
                                                        input_size=input_size,
                                                        batch_size=batch_size,
                                                        loss_fn=loss_fn,
                                                        torchmetrics=[acc, iou, f1])
                batch_loss = batch_loss_tensor.item() if batch_loss_tensor is not None else 0.0

            total_loss += batch_loss

            #imgs_vis, gt_vis, pred_vis = prepare_segmentation_visuals(imgs.detach().cpu(), masks.detach().cpu(), preds.detach().cpu(),
            #                                            bands=dataset_cfg.bands, dataset_name=cfg.dataset_name)
            #plot_images(torch.stack([imgs_vis, gt_vis, pred_vis], dim=1).flatten(0,1), ncols=3)

    results = {f"loss": total_loss / len(test_loader),
               f"Acc":  acc.compute().item(),
               f"IoU":  iou.compute().item(),
               f"F1":   f1.compute().item()}
    
    save_results(results, cfg.dataset_name, cfg.experiment_name, ood_split=ood_split, csv_path="results.csv")
    
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    return results


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    main()
