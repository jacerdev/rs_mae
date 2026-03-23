import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class HOGLayer(nn.Module):
    # Copyright (c) OpenMMLab. All rights reserved.
    # --------------------------------------------------------
    # Modified to handle multi-spectral remote sensing imagery
    # Original code: https://mmselfsup.readthedocs.io/en/1.x/_modules/mmselfsup/models/target_generators/hog_generator.html
    # --------------------------------------------------------

    def __init__(self, nbins=9, pool=8, in_channels=10, use_max_gradient=False, gaussian_window=None, normalize=True):
        super().__init__()
        self.nbins = nbins
        self.pool = pool
        self.in_channels = in_channels
        self.use_max_gradient = use_max_gradient
        self.gaussian_window = gaussian_window
        self.normalize = normalize

        # Sobel filters
        weight_x = torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        weight_x = weight_x.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1)
        weight_y = weight_x.transpose(2, 3)
        self.register_buffer('weight_x', weight_x)
        self.register_buffer('weight_y', weight_y)

        # Precompute Gaussian window if specified
        if gaussian_window is not None:
            gkern = self._get_gaussian_kernel(gaussian_window, gaussian_window // 2)
            self.register_buffer('gkern', gkern)

    def _get_gaussian_kernel(self, size, std):
        n = torch.arange(0, size).float()
        n -= n.mean()
        n /= std
        gkern1d = torch.exp(-0.5 * n**2)
        gkern2d = torch.outer(gkern1d, gkern1d)
        return gkern2d / gkern2d.sum()

    @torch.no_grad()
    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.in_channels, "Channel mismatch"

        x_pad = F.pad(x, (1, 1, 1, 1), mode='reflect')
        gx = F.conv2d(x_pad, self.weight_x, bias=None, stride=1, padding=0, groups=C)
        gy = F.conv2d(x_pad, self.weight_y, bias=None, stride=1, padding=0, groups=C)
        norm = torch.stack([gx, gy], dim=-1).norm(dim=-1)
        phase = torch.atan2(gx, gy) # range [-pi, pi]
        phase = phase / math.pi * self.nbins # range [-nbins, nbins]

        if self.use_max_gradient:
            norm, max_idx = norm.max(dim=1, keepdim=True) # (B, 1, H, W)
            phase = torch.gather(phase, 1, max_idx) # (B, 1, H, W)
            out_channels = 1
        else:
            out_channels = C

        hist = torch.zeros((B, out_channels, self.nbins, H, W), dtype=torch.float, device=x.device)
        phase = phase.view(B, out_channels, 1, H, W)
        norm = norm.view(B, out_channels, 1, H, W)

        if self.gaussian_window:
            if H != self.gaussian_window: # assuming squared inputs
                assert H % self.gaussian_window == 0, 'h {} gw {}'.format(H, self.gaussian_window)
                repeat_rate = H // self.gaussian_window
                temp_gkern = self.gkern.repeat([repeat_rate, repeat_rate])
            else:
                temp_gkern = self.gkern
            norm *= temp_gkern

        # scatter-add to bins (hard binning)
        #hist.scatter_add_(2, phase.floor().long() % self.nbins, norm) # phase range [0, nbins]
        
        # Soft binning
        phase_floor = phase.floor()
        # weights (distance to bin center)
        w1 = phase - phase_floor
        w0 = 1.0 - w1
        b0 = phase_floor.long() % self.nbins
        b1 = (b0 + 1) % self.nbins
        # Scatter-add
        hist.scatter_add_(2, b0, norm * w0)
        hist.scatter_add_(2, b1, norm * w1)

        # pool to get grid
        hist = hist.unfold(3, self.pool, self.pool)
        hist = hist.unfold(4, self.pool, self.pool)
        pooled = hist.sum(dim=[-1, -2])

        # normalize over bins
        if self.normalize:
            pooled = F.normalize(pooled, p=2, dim=2)
        
        return pooled # (B, out_channels, nbins, h, w)
    
def get_hog_saliency(hog_features_grid): # (B, C, nbins, h_grid, w_grid)
    """
    SelectiveMAE's saliency for selective masking
    (Don't call it on L2 normalized hog features, otherwise
    noisy features get highest saliency, instead of patches with clean edges)
    """
    saliency = hog_features_grid.sum(dim=(1, 2)) # (B, h_grid, w_grid)
    mean = saliency.mean(dim=(1, 2), keepdim=True) # (B, 1, 1)
    var = saliency.var(dim=(1, 2), keepdim=True)
    out = (saliency - mean) / (var + 1e-6).sqrt()
    return out # (B, h_grid, w_grid)


class NDILayer(nn.Module):
    """
    Extracts NDVI, NDWI, and NDBI from multispectral imagery.
    NDVI = (NIR - R) / (NIR + R) # Normalized Difference Vegetation Index
    NDWI = (G - NIR) / (G + NIR) # Normalized Difference Water Index
    NDBI = (SWIR - NIR) / (SWIR + NIR) # Normalized Difference Built-up Index

    (to be computed on the original radiometric values, not on the normalized ones !)
    """
    def __init__(self, bands):
        super().__init__()
        self.bands = bands
        try:
            self.nir_idx = bands.index("B08")
            self.red_idx = bands.index("B04")
            self.green_idx = bands.index("B03")
            self.swir_idx = bands.index("B11")
        except ValueError as e:
            raise ValueError(f"Required band missing from input bands list: {e}")

    @torch.no_grad()
    def forward(self, x):
        # x: [B, C, H, W]
        nir = x[:, self.nir_idx:self.nir_idx+1]
        red = x[:, self.red_idx:self.red_idx+1]
        green = x[:, self.green_idx:self.green_idx+1]
        swir = x[:, self.swir_idx:self.swir_idx+1]

        ndvi = (nir - red) / (nir + red + 1e-5)
        ndwi = (green - nir) / (green + nir + 1e-5)
        ndbi = (swir - nir) / (swir + nir + 1e-5)

        out = torch.cat([ndvi, ndwi, ndbi], dim=1)  # [B, 3, H, W]
        
        return out




"""
def draw_hog_arrows(hog_grid, image=None, color=(255, 0, 0), thickness=1):
    
    hog_grid: [h_grid, w_grid, nbins] (single channel's hog features)
    image: [H, W, C] (optional background)
    
    if isinstance(hog_grid, torch.Tensor):
        hog_grid = hog_grid.detach().cpu().numpy()
    if image is not None:
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
            image = image.transpose(1, 2, 0)
        
    h_grid, w_grid, nbins = hog_grid.shape
    
    if image is not None:
        out_img = image.copy()
        patch_size = image.shape[0] // h_grid
        if image.dtype != 'uint8': color = (color[0] / 255, color[1] / 255, color[2] / 255)
    else:
        patch_size = 8
        out_img = np.zeros((h_grid * patch_size, w_grid * patch_size, 3), dtype=np.uint8)

    angles = np.linspace(0, np.pi, nbins, endpoint=False)
    scale = patch_size / 2.5

    for r in range(h_grid):
        for c in range(w_grid):
            # patch center
            cx = int(c * patch_size + patch_size // 2)
            cy = int(r * patch_size + patch_size // 2)
            
            for bin_idx in range(nbins):
                mag = hog_grid[r, c, bin_idx]
                
                if mag > 0.05: # assuming L2 normalized hog features
                    dx = mag * math.cos(angles[bin_idx]) * scale
                    dy = mag * math.sin(angles[bin_idx]) * scale
                    # end points for the line (drawing from center)
                    p1 = (int(cx - dx), int(cy - dy))
                    p2 = (int(cx + dx), int(cy + dy))
                    cv2.line(out_img, p1, p2, color, thickness, cv2.LINE_AA)
                    
    return out_img"""