import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path so we can import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.strategies.loss.glcm import DifferentiableGLCM

def generate_synthetic_textures(size=32):
    """Generates synthetic texture images normalized to [-0.5, 0.5] range."""
    coords = np.ogrid[:size, :size]
    
    # 1. Checkerboard
    checkerboard = ((coords[0] + coords[1]) % 2).astype(np.float32)
    
    # 2. Horizontal stripes
    h_stripes = np.zeros((size, size), dtype=np.float32)
    h_stripes[:] = (coords[0] % 2)
    
    # 3. Vertical stripes
    v_stripes = np.zeros((size, size), dtype=np.float32)
    v_stripes[:] = (coords[1] % 2)
    
    # Scale from [0, 1] to [-0.5, 0.5] (safely inside the -1 to 1 bins)
    textures = {
        "Checkerboard": torch.from_numpy(checkerboard - 0.5),
        "Horizontal Stripes": torch.from_numpy(h_stripes - 0.5),
        "Vertical Stripes": torch.from_numpy(v_stripes - 0.5)
    }
    return textures

def compute_discrete_glcm(image, bins=8, min_val=-1.0, max_val=1.0, direction="horizontal"):
    """Computes exact discrete GLCM using hard binning for comparison."""
    # Digitize image into bins
    bin_edges = np.linspace(min_val, max_val, bins + 1)
    # Clip to avoid out of bounds
    img_clipped = np.clip(image, min_val, max_val)
    # digitize returns 1-indexed, we want 0-indexed
    binned = np.digitize(img_clipped, bin_edges) - 1
    # Handle values exactly at max_val
    binned[binned == bins] = bins - 1
    
    if direction == "horizontal":
        I1 = binned[:, :-1].flatten()
        I2 = binned[:, 1:].flatten()
    else: # vertical
        I1 = binned[:-1, :].flatten()
        I2 = binned[1:, :].flatten()
        
    # Compute 2D histogram
    H, _, _ = np.histogram2d(I1, I2, bins=bins, range=[[0, bins], [0, bins]])
    # Normalize to probabilities
    return H / (H.sum() + 1e-8)

def main():
    print("--- Differentiable vs Discrete GLCM Verification ---")
    
    bins = 8
    glcm_module = DifferentiableGLCM(
        bins=bins,
        min_val=-1.0,
        max_val=1.0,
        bandwidth=0.15 # controls the soft-binning smoothness
    )
    
    textures = generate_synthetic_textures(size=16)
    fig, axes = plt.subplots(3, 4, figsize=(15, 10))
    
    for idx, (name, img_tensor) in enumerate(textures.items()):
        img_np = img_tensor.numpy()
        x = img_tensor.unsqueeze(0) 
        
        # 1. Compute Differentiable GLCM (Soft)
        with torch.no_grad():
            glcm_h_soft, glcm_v_soft = glcm_module(x)
        h_soft_np = glcm_h_soft[0].numpy()
        v_soft_np = glcm_v_soft[0].numpy()
        
        # 2. Compute Discrete GLCM (Hard)
        h_hard_np = compute_discrete_glcm(img_np, bins, -1.0, 1.0, "horizontal")
        v_hard_np = compute_discrete_glcm(img_np, bins, -1.0, 1.0, "vertical")
        
        print(f"\nTexture: {name}")
        print(f"  Horizontal Soft GLCM Sum: {h_soft_np.sum():.4f}")
        print(f"  Horizontal Hard GLCM Sum: {h_hard_np.sum():.4f}")
        # Compute MSE between Hard and Soft approximations
        mse_h = np.mean((h_soft_np - h_hard_np)**2)
        print(f"  MSE between Soft and Hard Horizontal GLCM: {mse_h:.6f}")
        
        # Plotting
        # Image
        ax_img = axes[idx, 0]
        im = ax_img.imshow(img_np, cmap="gray", vmin=-1.0, vmax=1.0)
        ax_img.set_title(f"{name} Image")
        fig.colorbar(im, ax=ax_img, fraction=0.046, pad=0.04)
        
        # Hard Horizontal
        ax_h1 = axes[idx, 1]
        im_h1 = ax_h1.imshow(h_hard_np, cmap="viridis", vmin=0)
        ax_h1.set_title("Discrete (Hard) GLCM (H)")
        fig.colorbar(im_h1, ax=ax_h1, fraction=0.046, pad=0.04)
        
        # Soft Horizontal
        ax_h2 = axes[idx, 2]
        im_h2 = ax_h2.imshow(h_soft_np, cmap="viridis", vmin=0)
        ax_h2.set_title("Differentiable (Soft) GLCM (H)")
        fig.colorbar(im_h2, ax=ax_h2, fraction=0.046, pad=0.04)
        
        # Soft Vertical
        ax_v = axes[idx, 3]
        im_v = ax_v.imshow(v_soft_np, cmap="viridis", vmin=0)
        ax_v.set_title("Differentiable (Soft) GLCM (V)")
        fig.colorbar(im_v, ax=ax_v, fraction=0.046, pad=0.04)
        
    plt.tight_layout()
    output_png = "checks/glcm_visualization.png"
    plt.savefig(output_png, dpi=150)
    print(f"\n✅ Plot saved successfully to: {os.path.abspath(output_png)}")
    
if __name__ == "__main__":
    main()
