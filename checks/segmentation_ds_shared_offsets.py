from pathlib import Path
import sys
PROJECT_PATH = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_PATH))

import numpy as np
from torch.utils.data import DataLoader, get_worker_info
from datasets.segmentation_ds import SegmentationDataset


class TestDataset(SegmentationDataset):
    def __getitem__(self, idx):
        # Make every worker print its own state
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else "Main"
        print(f"[Worker {worker_id}] getitem called for idx {idx}")
        print(f"[Worker {worker_id}] self.offsets:{' '*12} {self.offsets.tolist()}")

        img, mask = super().__getitem__(idx)
        return img, mask, self.offsets

def main():
    # 5 tiles of 512x512
    tiles = [np.zeros((512, 512, 3), dtype=np.float32)] * 5
    masks = [np.zeros((512, 512), dtype=np.int64)] * 5
    
    # crop_size=224 gives a margin of 512-224*2 = 64
    dataset = TestDataset(
        tiles=tiles, masks=masks, crop_size=224, 
        grid_sampling=True, min_margin=None
    )

    print(f"Dataset length: {len(dataset)}")
    
    loader = DataLoader(
        dataset, batch_size=2, num_workers=4, persistent_workers=True
    )
    
    print(f"\n[Epoch 0] Main process self.offsets: {dataset.offsets.tolist()}")
    for batch in loader:
        _, _, worker_offsets = batch
        break
        
    dataset.shift_grid(1)
    
    print(f"\n[Epoch 1] Main process self.offsets: {dataset.offsets.tolist()}")
    for batch in loader:
        _, _, worker_offsets = batch
        break

if __name__ == "__main__":
    main()
