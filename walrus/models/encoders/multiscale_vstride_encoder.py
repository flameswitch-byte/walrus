"""B5 — multi-scale conv tokenizer (fork of the vstride encoder).

Each strided conv layer is augmented with one or more **coarse** paths alongside the
**medium** (base) path. The coarse paths first *coarse-grain* the input by an integer
factor (conservative average pooling = the finite-volume restriction operator), then
apply a narrow base-kernel conv on the coarsened grid. A base kernel on a 2x-pooled
grid reaches 2x the original receptive field ("an 8x8 view that is truly 16x16") at
1/4 the kernel volume of a dense 16x16 conv, and average pooling is anti-aliased by
construction (unlike dilation, which point-samples and aliases). All paths land on
the SAME token grid, are concatenated over channels, and fused by a 1x1 mix.

Design choices (see knowledge_base/walrus_encoder_multiscale_skip.md):
- The **medium path reuses the base proj1/proj2 at FULL width** -> loads pretrained
  weights; the fork is **bit-exact to the baseline at init** once the 1x1 mix is
  identity-on-medium / zero-on-coarse.
- Coarse paths are **narrow** (width // extra_div) and learned from scratch.
- The **fine** (sub-base) path was dropped: it is information-redundant (a subset of
  the medium window) and only added an inductive bias. Pushing *upward* in support
  (coarse) is the non-redundant direction (cross-boundary context).
- Coarse-graining = **average pooling** (conservative volume average, the physically
  correct restriction for conserved densities; anti-aliased). Richer variants
  (mean+variance moments; Haar wavelet) are future work.
- Fixed linear op (parallel convs + pooling + 1x1 mix) -> rollout-safe.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from the_well.data.datasets import BoundaryCondition
from torch import Tensor

from walrus.models.encoders.vstride_encoder import (
    CONV_FUNCS,
    SpaceBagAdaptiveDVstrideEncoder,
)

POOL_FUNCS = {1: F.avg_pool1d, 2: F.avg_pool2d, 3: F.avg_pool3d}


class MultiScaleSpaceBagVstrideEncoder(SpaceBagAdaptiveDVstrideEncoder):
    """Multi-scale (B5) fork of ``SpaceBagAdaptiveDVstrideEncoder``: medium + coarse
    (pooling-based) paths, fused by a 1x1 mix."""

    def __init__(
        self,
        *args,
        multiscale_layers: Tuple[int, ...] = (1, 2),
        multiscale_extra_div: int = 4,
        multiscale_coarse_pools: Tuple[int, ...] = (2,),
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.multiscale_layers = tuple(multiscale_layers)
        self.multiscale_extra_div = multiscale_extra_div
        # One coarse path per pooling factor (e.g. (2,) -> 16px reach; (2,4) -> 16 & 32).
        self.coarse_pools = tuple(multiscale_coarse_pools)
        self.n_coarse = len(self.coarse_pools)

        conv_class = CONV_FUNCS[self.spatial_dims][0]
        self.pool_func = POOL_FUNCS[self.spatial_dims]

        # Layer 1 coarse paths (medium = self.proj1, full width). Coarse convs use the
        # SAME base kernel as medium (the reach comes from the pooling, not a big
        # kernel). Field-selected at forward time like proj1.
        if 1 in self.multiscale_layers and self.n_coarse:
            extra1 = max(1, self.inner_dim // multiscale_extra_div)
            self.coarse1 = nn.ModuleList(
                conv_class(
                    self.input_dim, extra1, kernel_size=self.base_kernel1, bias=False
                )
                for _ in self.coarse_pools
            )
            self.mix1 = conv_class(
                self.inner_dim + self.n_coarse * extra1,
                self.inner_dim,
                kernel_size=1,
                bias=True,
            )
            self._init_mix_identity(self.mix1, self.inner_dim)

        # Layer 2 coarse paths (medium = self.proj2, full width = output_dim).
        if 2 in self.multiscale_layers and self.n_coarse:
            extra2 = max(1, self.output_dim // multiscale_extra_div)
            self.coarse2 = nn.ModuleList(
                conv_class(
                    self.inner_dim, extra2, kernel_size=self.base_kernel2, bias=False
                )
                for _ in self.coarse_pools
            )
            self.mix2 = conv_class(
                self.output_dim + self.n_coarse * extra2,
                self.output_dim,
                kernel_size=1,
                bias=True,
            )
            self._init_mix_identity(self.mix2, self.output_dim)

    def _init_mix_identity(self, mix: nn.Module, medium_width: int) -> None:
        """Init the 1x1 mix to pass the medium block through unchanged and zero the
        coarse extras -> the layer output equals the baseline at init."""
        with torch.no_grad():
            mix.weight.zero_()
            if mix.bias is not None:
                mix.bias.zero_()
            # weight shape: [out, in, 1, 1, (1)]; medium occupies the first
            # `medium_width` input channels (concat order is [medium, coarse...]).
            idx = (slice(None), slice(0, medium_width)) + (0,) * self.spatial_dims
            mix.weight[idx] = torch.eye(medium_width)

    def _medium_path(self, x, weight, bias, stride):
        """Base (medium) strided conv: valid conv (jitterer pre-pads), singleton-axis
        kernel collapse, optional BlurPool. Identical to the baseline encoder."""
        nd = self.spatial_dims
        w = weight
        st = list(stride)
        for i, dim in enumerate(x.shape[-nd:][::-1], start=1):
            if dim == 1:
                w = w.sum(dim=-i, keepdim=True)
                st[-i] = 1
        xp = x
        if self.anti_aliased_stride and self.blurpool is not None:
            xp = self.blurpool(xp, st)
        return self.conv_func(xp, w, bias, tuple(st), 0)

    def _coarse_path(self, x, bcs, weight, bias, stride, pool_f, k_base):
        """Coarse-grain by ``pool_f`` (conservative average pooling = finite-volume
        restriction), then a base-kernel conv at the reduced stride, padded/cropped so
        the output lands on the SAME token grid as the medium path. Reach in original
        pixels = pool_f * base_kernel."""
        nd = self.spatial_dims
        bcs_local = list(bcs)
        while len(bcs_local) < nd:
            bcs_local = bcs_local + [[2, 2]]

        # Per-axis pool factor (1 on singleton axes).
        pool_k = [pool_f if x.shape[-nd + a] > 1 else 1 for a in range(nd)]
        xc = self.pool_func(
            x, kernel_size=tuple(pool_k), stride=tuple(pool_k), ceil_mode=True
        )

        st = [1] * nd
        for a in range(nd):
            N = x.shape[-nd + a]  # pre-pool size
            if N == 1:
                continue
            s = int(stride[a])
            kb = int(k_base[a])
            target = (N - kb) // s + 1  # medium output size on this axis
            sp = max(1, s // pool_k[a])
            st[a] = sp
            Np = xc.shape[-nd + a]
            ka = int(weight.shape[-nd + a])  # coarse conv kernel (= base, pre-collapse)
            ptot = (target - 1) * sp + ka - Np  # pad (>0) or crop (<0) to hit target
            pos = 2 * (nd - 1 - a)
            cur = xc.shape[-nd + a]
            if ptot > 0:
                left, right = ptot // 2, ptot - ptot // 2
                pad_arg = [0] * (2 * nd)
                pad_arg[pos], pad_arg[pos + 1] = left, right
                periodic = (
                    a < len(bcs_local)
                    and int(bcs_local[a][0]) == BoundaryCondition["PERIODIC"].value
                )
                if periodic and max(left, right) < cur:
                    mode = "circular"
                elif max(left, right) < cur:
                    mode = "reflect"
                else:
                    mode = "replicate"
                xc = F.pad(xc, tuple(pad_arg), mode=mode)
            elif ptot < 0:
                c = -ptot
                cl, cr = c // 2, c - c // 2
                sl = [slice(None)] * xc.ndim
                sl[-nd + a] = slice(cl, xc.shape[-nd + a] - cr)
                xc = xc[tuple(sl)]

        # Singleton-axis kernel collapse (stride already 1 there).
        w = weight
        for i, dim in enumerate(xc.shape[-nd:][::-1], start=1):
            if dim == 1:
                w = w.sum(dim=-i, keepdim=True)
        if self.anti_aliased_stride and self.blurpool is not None:
            xc = self.blurpool(xc, st)
        return self.conv_func(xc, w, bias, tuple(st), 0)

    def _medium_weight1(self, field_indices):
        """proj1 weight with the SpaceBag field selection + scale hack (replicated for
        bit-exactness with the parent)."""
        w = self.proj1.weight[:, field_indices]
        scale = (
            (self.proj1.weight.shape[1] - self.extra_dims)
            / (w.shape[1] - self.extra_dims)
        ) ** 0.5
        w = w.clone()
        w[:, :-2] = w[:, :-2] * scale
        return w

    def forward(
        self, x: Tensor, field_indices: Tensor, bcs=None, metadata=None, **kwargs
    ) -> Tuple[Tensor, Dict[str, Any]]:
        embed_kernel = kwargs["random_kernel"]
        stride1 = tuple(embed_kernel[i][0] for i in range(self.spatial_dims))
        stride2 = tuple(embed_kernel[i][1] for i in range(self.spatial_dims))

        T = x.shape[0]
        x = rearrange(x, "T B ... -> (T B) ...")

        # Layer 1
        w_med1 = self._medium_weight1(field_indices)
        if 1 in self.multiscale_layers and self.n_coarse:
            med = self._medium_path(x, w_med1, None, stride1)
            coarse = [
                self._coarse_path(
                    x, bcs, self.coarse1[j].weight[:, field_indices], None,
                    stride1, pf, self.base_kernel1,
                )
                for j, pf in enumerate(self.coarse_pools)
            ]
            x = self.mix1(torch.cat([med, *coarse], dim=1))
        else:
            x = self._medium_path(x, w_med1, None, stride1)
        x = self.act(self.norm1(x))

        # Layer 2
        if 2 in self.multiscale_layers and self.n_coarse:
            med = self._medium_path(x, self.proj2.weight, self.proj2.bias, stride2)
            coarse = [
                self._coarse_path(
                    x, bcs, self.coarse2[j].weight, None, stride2, pf, self.base_kernel2,
                )
                for j, pf in enumerate(self.coarse_pools)
            ]
            x = self.mix2(torch.cat([med, *coarse], dim=1))
        else:
            x = self._medium_path(x, self.proj2.weight, self.proj2.bias, stride2)
        x = self.act(self.norm2(x))

        x = rearrange(x, "(T B) ... -> T B ...", T=T)
        return x, kwargs
