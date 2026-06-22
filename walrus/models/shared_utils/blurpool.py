"""Stride-aware anti-aliasing blur for strided convolutions (BlurPool, Zhang 2019).

A strided convolution fuses *filtering* (the kernel) and *downsampling* (the
stride). The subsample step aliases any spatial frequency above the post-stride
Nyquist limit back into lower frequencies, corrupting them. This is the same
artifact that patch jittering scrubs stochastically; ``BlurPoolDynamic`` removes it
deterministically per forward pass.

Implementation note: at Walrus's large, *dynamic* strides (2-8 per axis), the
faithful Zhang formulation (conv at stride 1 -> blur -> subsample) costs ``s**d``x
more conv MACs, which is prohibitive. We instead apply a fixed depthwise low-pass
to the conv *input* (matched to that layer's stride) and leave the existing strided
conv unchanged - i.e. "low-pass then strided subsample". This is cheap (a depthwise
conv), parameter-free, weight-compatible, and preserves spatial sizes so the strided
conv output shape is identical with the flag on or off. It matches the methodology
of the aliasing measurement in
``knowledge_base/walrus_encoder_spectral_antialiasing.md`` (blur the field, then
subsample).
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

_CONV = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}


def _binomial_kernel_1d(taps: int) -> torch.Tensor:
    """Normalized 1D binomial (approx. Gaussian) low-pass of length ``taps``."""
    n = taps - 1
    coeffs = torch.tensor(
        [math.comb(n, i) for i in range(taps)], dtype=torch.float32
    )
    return coeffs / coeffs.sum()


class BlurPoolDynamic(nn.Module):
    """Separable, depthwise, stride-aware binomial low-pass.

    Applied to a conv's input before a strided subsample. Blurs only along axes
    with ``stride > 1`` and ``size > 1``; leaves all other axes (and singleton /
    inflated dims) untouched. The filter width scales with the stride so the cutoff
    tracks the post-stride Nyquist limit (``taps = 2*stride - 1``).

    Parameters
    ----------
    spatial_dims:
        Number of trailing spatial dims (1/2/3). Walrus uses 3.
    pad_mode:
        Padding mode for the same-size blur (``"reflect"`` by default; falls back
        to ``"replicate"`` when an axis is shorter than the padding width).
    """

    def __init__(self, spatial_dims: int = 3, pad_mode: str = "reflect") -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.conv = _CONV[spatial_dims]
        self.pad_mode = pad_mode
        # Lazily-built kernel cache keyed by (taps, device, dtype).
        self._cache: dict = {}

    def _kernel1d(
        self, taps: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        key = (taps, device, dtype)
        k = self._cache.get(key)
        if k is None:
            k = _binomial_kernel_1d(taps).to(device=device, dtype=dtype)
            self._cache[key] = k
        return k

    def forward(self, x: torch.Tensor, stride: Sequence[int]) -> torch.Tensor:
        # x: (N, C, *spatial) with len(spatial) == spatial_dims
        nd = self.spatial_dims
        C = x.shape[1]
        for a in range(nd):
            s = int(stride[a])
            size_a = x.shape[2 + a]
            if s <= 1 or size_a <= 1:
                continue  # nothing to anti-alias on this axis
            taps = 2 * s - 1
            pad = (taps - 1) // 2  # = s - 1, "same" padding
            k = self._kernel1d(taps, x.device, x.dtype)
            kshape = [1] * nd
            kshape[a] = taps
            weight = k.view(1, 1, *kshape).expand(C, 1, *kshape).contiguous()
            # F.pad fills the last dim first; axis a sits at offset 2*(nd-1-a).
            pad_arg = [0] * (2 * nd)
            pos = 2 * (nd - 1 - a)
            mode = self.pad_mode if pad < size_a else "replicate"
            pad_arg[pos] = pad
            pad_arg[pos + 1] = pad
            x = F.pad(x, tuple(pad_arg), mode=mode)
            x = self.conv(x, weight, stride=1, padding=0, groups=C)
        return x
