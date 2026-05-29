import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from dataclasses import asdict
import hydra
from hydra.utils import instantiate
import logging
from omegaconf import DictConfig
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy, JaccardIndex, F1Score
from tqdm import tqdm

from datasets import ReBEN7kDataset, BurnScarsDataset, Sen1Floods11Dataset, C2SegDataset, CashewPlantationDataset, MMHunanDataset
from models.segmentation.segmentation_vit import SegmentationViT
from utils.config import SegmentationViT_Cfg, ReBEN7kDataset_Cfg, BurnScarsDataset_Cfg, Sen1Floods11Dataset_Cfg, C2SegDataset_Cfg, CashewPlantationDataset_Cfg, MMHunanDataset_Cfg
from utils.spatial_ops import windowed_inference, windowed_evaluation
from utils.utils import get_checkpoints_dir, find_best_ckpt, log_per_class_results, log_results
from utils.visualization import plot_images, prepare_segmentation_visuals

PLOT_RESULTS = False

class EvaluationDataset(torch.utils.data.Dataset):
    """Wrapper to add ids to dataset items."""
    def __init__(self, ds):
        self.ds = ds
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        img, mask = self.ds[idx]
        return self.ds.ids[idx], img, mask


def run_benchmark(cfg: DictConfig, ckpt_path: str = None, results_dir="log_results"):
    if ckpt_path is None: # Offline run (local pc)
        batch_size = 16
        num_workers = 2
        prefetch_factor = 2

        exp_key = f"{cfg.experiment_name}-{cfg.seed}"
        ckpt_dir = get_checkpoints_dir(exp_key, cfg.dataset_name, cfg.paths.artifacts_dir, wandb_project=cfg.wandb.project, json_path="runs_dummy.json" if cfg.get("is_dummy", False) else cfg.runs_json)
        ckpt_path = find_best_ckpt(ckpt_dir)
        if not ckpt_path:
            raise FileNotFoundError(f"Could not find checkpoint in {ckpt_dir}")
    else: # online run
        batch_size = 64
        num_workers = cfg.num_workers
        prefetch_factor = cfg.prefetch_factor

    logging.info(f"Loading checkpoint from {ckpt_path}")

    # Data
    test_transform = A.Compose([ToTensorV2()], seed=cfg.seed)
    splits_to_eval = ["test"]

    match cfg.dataset_name:
        case "reben_7k":
            dataset_class = ReBEN7kDataset
            dataset_class_cfg = ReBEN7kDataset_Cfg
            test_transform = A.Compose([A.PadIfNeeded(min_height=cfg.img_size, min_width=cfg.img_size,
                                         border_mode=cv2.BORDER_REFLECT), ToTensorV2()], seed=cfg.seed)
            splits_to_eval = ["test", "ood"]
        case "burn_scars":
            dataset_class = BurnScarsDataset
            dataset_class_cfg = BurnScarsDataset_Cfg
        case "sen1floods11":
            dataset_class = Sen1Floods11Dataset
            dataset_class_cfg = Sen1Floods11Dataset_Cfg
            splits_to_eval = ["test", "bolivia"]
        case "cashew_plantation":
            dataset_class = CashewPlantationDataset
            dataset_class_cfg = CashewPlantationDataset_Cfg
        case "mmhunan":
            dataset_class = MMHunanDataset
            dataset_class_cfg = MMHunanDataset_Cfg
        case name if name.startswith("c2seg"):
            dataset_class = C2SegDataset
            dataset_class_cfg = C2SegDataset_Cfg
            if cfg.dataset_name == "c2seg_beijing": cfg.dataset_name, cfg.dataset.city = "c2seg_wuhan", "wuhan"
            elif cfg.dataset_name == "c2seg_augsburg": cfg.dataset_name, cfg.dataset.city = "c2seg_berlin", "berlin"
        case _:
            raise ValueError(f"Unknown dataset name extracted from wandb.run_name: {cfg.dataset_name}")

    dataset_cfg = dataset_class_cfg(**cfg.dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model_cfg = SegmentationViT_Cfg(
        **{**dict(cfg.model), "norm_layer": instantiate(cfg.model.norm_layer)}
    )
    model = SegmentationViT(**asdict(model_cfg))

    # Load checkpoint
    try:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = state.get("state_dict", state)
        state_dict = {k.removeprefix("model."): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        logging.info(f"Loaded checkpoint from {ckpt_path}")
    except Exception as e:
        logging.error(f"Failed to load checkpoint from {ckpt_path}: {e}")

    model = model.to(device).eval()

    all_results = {}

    for split_name in splits_to_eval:
        logging.info(f"=== Benchmarking split: {split_name} ===")
        
        # Dataloader
        test_ds = dataset_class(**{**asdict(dataset_cfg), "grid_sampling": False},
                                split=split_name, transform=test_transform)
        logging.info(f"Test set size: [{len(test_ds)}, {tuple(test_ds[0][0].shape)}]")

        eval_ds = EvaluationDataset(test_ds)
        test_loader = DataLoader(
            eval_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, prefetch_factor=prefetch_factor,
            persistent_workers=num_workers > 0, pin_memory=True)

        # Metrics
        loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        acc  = MulticlassAccuracy(num_classes=cfg.num_classes, ignore_index=-1, average='micro').to(device)
        iou  = JaccardIndex(task="multiclass", num_classes=cfg.num_classes, ignore_index=-1, average='macro').to(device)
        f1   = F1Score(task="multiclass", num_classes=cfg.num_classes, ignore_index=-1, average='none').to(device)

        total_loss = 0.0

        with torch.no_grad():
            for batch_idx, (sample_ids, imgs, masks) in tqdm(enumerate(test_loader), total=len(test_loader)):
                imgs  = imgs.to(device)
                masks = masks.long().to(device)
                _, _, H, W = imgs.shape
                input_size = cfg.img_size

                # Inference
                if H <= input_size and W <= input_size:
                    preds = model(imgs) # (B, C, H, W) on device
                else:
                    preds_list = []
                    for sample in imgs:
                        pred = windowed_inference(model=model, image=sample,
                                                window_size=input_size,
                                                stride=input_size // 2,
                                                num_classes=cfg.num_classes,
                                                batch_size=batch_size,
                                                merger_device='cpu')
                        preds_list.append(pred)
                    preds = torch.stack(preds_list, dim=0) # (B, C, H, W) on CPU

                # Evaluation
                if H <= input_size and W <= input_size:
                    preds = preds.to(device)
                    batch_loss = loss_fn(preds, masks).item() if loss_fn else 0.0
                    for torchmetric in [acc, iou, f1]:
                        torchmetric.update(preds, masks)
                else:
                    batch_loss_tensor = windowed_evaluation(preds=preds, # on CPU
                                                            masks=masks,
                                                            tile_size=int(input_size * batch_size**0.5),
                                                            loss_fn=loss_fn,
                                                            torchmetrics=[acc, iou, f1])
                    batch_loss = batch_loss_tensor.item() if batch_loss_tensor is not None else 0.0

                total_loss += batch_loss

                if PLOT_RESULTS:
                    imgs_vis, gt_vis, pred_vis = prepare_segmentation_visuals(imgs.detach().cpu(), masks.detach().cpu(), preds.detach().cpu(),
                                                                bands=dataset_cfg.bands, dataset_name=cfg.dataset_name)
                    plot_images(torch.stack([imgs_vis, gt_vis, pred_vis], dim=1).flatten(0,1), ncols=3)
                
        f1_per_class = f1.compute()

        results = {f"loss": total_loss / len(test_loader),
                   f"OA": acc.compute().item(),
                   f"mIoU": iou.compute().item(),
                   f"mF1": f1_per_class.nanmean().item()}


        # per class results
        per_class_results = {}
        for idx, f1_val in enumerate(f1_per_class):
            class_name = dataset_class.CLASSES.get(idx, f"Class {idx}")
            per_class_results[class_name] = f1_val.item()

        # log
        dummy = '_dummy' if cfg.get('is_dummy', False) else ''
        split_suffix = f'_{split_name}' if split_name != 'test' else ''
        display_dataset_name = (cfg.dataset_name + split_suffix).replace("_", " ").title().strip()
        log_results(results, display_dataset_name, cfg.experiment_name,
                    csv_path=f"{results_dir}{dummy}/benchmark_results-{cfg.seed}.csv")
        log_per_class_results(per_class_results, cfg.experiment_name,
                              csv_path=f"{results_dir}{dummy}/{cfg.dataset_name}{split_suffix}-{cfg.seed}.csv")

        # return all results
        split_prefix = f'{split_name}_' if split_name != 'test' else ''
        all_results.update({f"{split_prefix}{k}": v for k, v in results.items()})
        all_results.update({f"{split_prefix}F1 - {class_name}": val for class_name, val in per_class_results.items()})

    return all_results


@hydra.main(version_base="1.3", config_path="../configs", config_name="finetune")
def main(cfg: DictConfig):
    all_results = run_benchmark(cfg)

    from pprint import pprint
    pprint(all_results)

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    main()