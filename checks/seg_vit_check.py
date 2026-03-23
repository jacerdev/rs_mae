import sys
from pathlib import Path
PROJECT_PATH = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_PATH))
import torch
from models.segmentation import SegmentationViT

def test_vit_smp_forward():
    print("Running sanity check for ViTMAESegmentationModel...")
    
    # Instantiate model
    num_classes = 13
    img_size = 112
    in_chans = 3
    
    model = SegmentationViT(
        num_classes=num_classes,
        img_size=img_size,
        patch_size=8,
        in_chans=in_chans,
        embed_dim=128,
        depth=4,
        pyramid_channels=64,
        segmentation_channels=32,
        use_decoder=False
    )
    model.eval()
    
    # dummy input
    batch_size = 2
    x = torch.randn(batch_size, in_chans, img_size, img_size)
    
    print(f"Input shape: {x.shape}")
    
    # forward pass
    with torch.no_grad():
        output = model(x)
        
    print(f"Output shape: {output.shape}")    
    
    # check intermediate features
    features = model.encoder(x)
    print(f"Number of features in pyramid: {len(features)}")
    for i, f in enumerate(features):
        print(f"  Feature {i} shape: {f.shape}")
        

if __name__ == "__main__":
    try:
        test_vit_smp_forward()
    except Exception as e:
        print(f"\n❌")
        import traceback
        traceback.print_exc()
