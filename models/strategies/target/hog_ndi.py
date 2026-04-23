import torch
import torch.nn as nn
from models.strategies.target.base import TargetGenerator
from utils.features import HOGLayer, NDILayer
from utils.spatial_ops import patchify, z_normalize, z_denormalize


class HOGNDITarget(nn.Module, TargetGenerator):
	"""
	NDI expects input NOT to be Z-score normalised (as in the default ReBENDataset pipeline),
	otherwise the indices are no longer in the calibrated [-1, 1] range.

	(Inherits nn.Module so that HOGLayer/NDILayer's register_buffer
	tensors (Sobel kernels) are correctly moved to the training device)

	MRO: nn.Module comes first so nn.Module.__call__ takes precedence over
	TargetGenerator.__call__, dispatching correctly to forward().
	"""
	NDI_CHANNELS = 3  # NDVI, NDWI, NDBI

	def __init__(self, patch_size, bands, pool, nbins=9, use_max_gradient=False, gaussian_window=None,
				 normalize=True, norm_pix=True, **kwargs):
		nn.Module.__init__(self)

		assert patch_size % pool == 0, "patch_size must be divisible by pool"
		self.pool = pool
		self.nbins = nbins
		self.norm_pix = norm_pix
		self.hog_out_channels = 1 if use_max_gradient else len(bands)

		self._hog_feats = (patch_size // pool)**2 * self.hog_out_channels * nbins
		self._ndi_feats = patch_size**2 * HOGNDITarget.NDI_CHANNELS

		TargetGenerator.__init__(self, patch_size=patch_size,
								out_features=self._hog_feats + self._ndi_feats, **kwargs)

		self.hog = HOGLayer(nbins=nbins, pool=pool, in_channels=len(bands), use_max_gradient=use_max_gradient,
							gaussian_window=gaussian_window, normalize=normalize)

		self.ndi = NDILayer(bands=bands)

	def forward(self, imgs, **kwargs):
		"""
		raw_imgs : (B, C, H, W) passed via kwargs
		returns: Tuple of (hog_tokens, ndi_tokens)
		"""
		raw_imgs = kwargs["raw_imgs"]
		hog_tokens = self._get_raw_hog_targets(imgs) # or raw_imgs, matters only for precision (when using float16..)
		ndi_tokens = self._get_raw_ndi_targets(raw_imgs)

		# Apply per-patch Z-score normalization if enabled
		if self.norm_pix: # destroys the physical meaning
			hog_tokens = z_normalize(hog_tokens)
			ndi_tokens = z_normalize(ndi_tokens)

		return hog_tokens, ndi_tokens

	def restore(self, pred, **kwargs):
		"""Split prediction back into HOG and NDI parts"""
		hog_pred = pred[..., :self._hog_feats]
		ndi_pred = pred[..., self._hog_feats:]
		
		# Reverse the Z-score normalization using the original image statistics
		if self.norm_pix:
			# Recalculate raw targets to extract their original mean and variance
			raw_imgs = kwargs["gt_imgs"]
			raw_hog = self._get_raw_hog_targets(raw_imgs)
			raw_ndi = self._get_raw_ndi_targets(raw_imgs)

			hog_pred = z_denormalize(hog_pred, raw_hog)
			ndi_pred = z_denormalize(ndi_pred, raw_ndi)
            
		return hog_pred, ndi_pred

	def _get_raw_hog_targets(self, imgs):
		"""Helper method to generate un-normalized HOG features"""
		hog_grid = self.hog(imgs)
		B, C, nb, hn, wn = hog_grid.shape
		n = self.patch_size // self.pool
		h, w = hn // n, wn // n
		hog_tokens = hog_grid.reshape(B, C, nb, h, n, w, n)
		#hog_tokens = hog_tokens.permute(0, 3, 5, 1, 4, 6, 2)
		#hog_tokens = hog_tokens.reshape(B, h * w, C * n * n * nb)
		hog_tokens = hog_tokens.permute(0, 3, 5, 1, 2, 4, 6)
		hog_tokens = hog_tokens.reshape(B, h * w, C * nb * n * n)
		return hog_tokens # [B, N, C*n²*nb]

	def _get_raw_ndi_targets(self, raw_imgs):
		"""Helper method to generate NDI features using raw images"""
		ndi_img = self.ndi(raw_imgs)
		ndi_tokens = patchify(ndi_img, self.patch_size) 
		return ndi_tokens # [B, N, p²*3]