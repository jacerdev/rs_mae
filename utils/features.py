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

        # normalize over bins per grid cell
        if self.normalize:
            pooled = F.normalize(pooled, p=2, dim=2, eps=1e-6)
        
        return pooled # (B, out_channels, nbins, h, w)
    
class HOGSaliency(HOGLayer):
    """
    Saliency extractor based on HOG features.
    Computes HOG features and selectively extracts Min-Max normalized saliency per patch.
    """
    def __init__(self, nbins=9, pool=8, in_channels=10):
        # enforce normalize=False as otherwise noisy features get highest saliency, instead of clean edges
        super().__init__(nbins, pool, in_channels, use_max_gradient=False, gaussian_window=None, normalize=False)

    @torch.no_grad()
    def forward(self, x):
        # x: (B, C, H, W)
        hog_features_grid = super().forward(x) # (B, C, nbins, h_grid, w_grid)
        saliency = hog_features_grid.sum(dim=(1, 2)) # (B, h_grid, w_grid)

        # SelectiveMAE's saliency
        #mean = saliency.mean(dim=(1, 2), keepdim=True) # (B, 1, 1)
        #var = saliency.var(dim=(1, 2), keepdim=True)
        #out = (saliency - mean) / (var + 1e-6).sqrt()

        # Min-max normalization
        #s_flat = saliency.view(saliency.shape[0], -1) 
        #s_min = s_flat.min(dim=1)[0].view(-1, 1, 1)
        #s_max = s_flat.max(dim=1)[0].view(-1, 1, 1)
        #out = (saliency - s_min) / (s_max - s_min + 1e-6)
        
        return saliency # (B, h_grid, w_grid)

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
        # x: [B, C, H, W] should have raw values (not normalized)
            
        nir = x[:, self.nir_idx:self.nir_idx+1]
        red = x[:, self.red_idx:self.red_idx+1]
        green = x[:, self.green_idx:self.green_idx+1]
        swir = x[:, self.swir_idx:self.swir_idx+1]

        ndvi = (nir - red) / (nir + red + 1e-5)
        ndwi = (green - nir) / (green + nir + 1e-5)
        ndbi = (swir - nir) / (swir + nir + 1e-5)

        out = torch.cat([ndvi, ndwi, ndbi], dim=1)  # [B, 3, H, W]
        
        return out

class NDISaliency(NDILayer):
    """
    Saliency extractor based on NDI indices.
    Calculates Coefficient of Variation (std / abs_mean) for each patch
    """
    def __init__(self, patch_size, bands):
        super().__init__(bands)
        self.patch_size = patch_size

    @torch.no_grad()
    def forward(self, x):
        ndi_features = super().forward(x)  # (B, 3, H, W)
        
        # Unfold into patches: (B, 3, h_grid, w_grid, patch_size, patch_size)
        patches = ndi_features.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        
        # std and mean over patches
        mean_abs = patches.abs().mean(dim=(-1, -2))
        std = patches.std(dim=(-1, -2), unbiased=False)
        
        cv = std / (mean_abs + 1e-6)  # (B, 3, h_grid, w_grid)
        
        # average over channels
        saliency = cv.mean(dim=1)  # (B, h_grid, w_grid)
        
        # Min-max normalization
        #s_flat = saliency.view(saliency.shape[0], -1) 
        #s_min = s_flat.min(dim=1)[0].view(-1, 1, 1)
        #s_max = s_flat.max(dim=1)[0].view(-1, 1, 1)
        #out = (saliency - s_min) / (s_max - s_min + 1e-6)

        return saliency  # (B, h_grid, w_grid)



class DifferentiableGLCM(nn.Module):
    """
    Args:
        bins (int): Number of gray-level bins (K). Default: 32.
        min_val (float): Lower clamp value of the input range. Default: -3.0.
        max_val (float): Upper clamp value of the input range. Default:  3.0.
        bandwidth (float | None): Sigmoid bandwidth. Default: half the bin width.

    Input:
        x (Tensor): (B, H, W), a batch of single-channel spatial patches.
                    Values should lie within [min_val, max_val] (not enforced here)

    Output:
        Tensor of (B, K, K), normalized, symmetric GLCM per sample.
    """

    def __init__(self, bins=32, min_val=-3.0, max_val=3.0,
                 bandwidth: float = None):
        super().__init__()
        self.bins    = bins
        self.min_val = min_val
        self.max_val = max_val

        self.L = (max_val - min_val) / bins
        self.W = bandwidth if bandwidth is not None else self.L / 2.0

        centers = torch.linspace(min_val + self.L / 2, max_val - self.L / 2, bins)
        self.register_buffer("mu", centers.view(1, -1, 1))  # (1, K, 1)

    def forward(self, x, chunk_size=-1):
        """ x (Tensor): (B, H, W)
            chunk_size (int): max batch size (-1 disables chunking)
        Returns:
            Tensor: (B, K, K) symmetric normalized GLCM
        """
        B = x.shape[0]

        if chunk_size > 0 and B > chunk_size:
            return torch.cat(
                [self._forward_chunk(x[i:i + chunk_size]) for i in range(0, B, chunk_size)],
                dim=0,
            )
        return self._forward_chunk(x)

    def _soft_bins(self, x):
        """Map pixel values to soft bins. x: (B,1,N) -> (B,K,N)"""
        term1 = torch.sigmoid((x - self.mu + self.L / 2) / self.W)
        term2 = torch.sigmoid((x - self.mu - self.L / 2) / self.W)
        return term1 - term2

    def _forward_chunk(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W = x.shape
        K = self.bins

        P = self._soft_bins(x.view(B, 1, H * W)).view(B, K, H, W)

        # collect co-occurring pixel pairs from all 4 directions in one tensor
        P_left = torch.cat([ # bin probabilities for every source pixel
            P[:, :, :,  :-1].reshape(B, K, -1),  # horizontal left
            P[:, :, :-1,  :].reshape(B, K, -1),  # vertical   down
            P[:, :, :-1, :-1].reshape(B, K, -1), # diagonal   down-right
            P[:, :, 1:,  :-1].reshape(B, K, -1), # diagonal   up-right
        ], dim=2)

        P_right = torch.cat([ # bin probabilities for every neighbor pixel
            P[:, :, :,   1:].reshape(B, K, -1),
            P[:, :, 1:,   :].reshape(B, K, -1),
            P[:, :, 1:,  1:].reshape(B, K, -1),
            P[:, :, :-1, 1:].reshape(B, K, -1),
        ], dim=2)

        N_total = P_left.shape[2]

        # (B, K, N) x (B, N, K) -> (B, K, K)
        H_mat = torch.bmm(P_left, P_right.transpose(1, 2)) / max(1, N_total)

        # symmetrize and normalize to a probability distribution
        H_sym = (H_mat + H_mat.transpose(1, 2)) / 2.0
        H_sym = H_sym / (H_sym.sum(dim=(1, 2), keepdim=True) + 1e-8)

        return H_sym  # (B, K, K)

