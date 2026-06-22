"""Two-grid encoder: emit a fine token grid AND a persistent coarse token grid.

Design B of the cross-scale work (see knowledge_base/walrus_crossscale_attention.md).
Unlike Design A (per-block scratch coarse pooled from the fine *features*), this emits
a coarse grid produced by an **un-fused B5-style raw-field tokenizer**: average-pool
the raw input by an integer ``ratio`` (conservative finite-volume restriction), then a
small, dedicated 2-conv stack at the encoder's dynamic stride. Reading the raw field at
low resolution (rather than pooling already-tokenized fine features) lets the coarse
path keep large-scale modes the fine encoder doesn't emphasize, and is not capped by
what survived the fine averaging.

The coarse grid is cropped to exactly ``ceil(Hf / ratio)`` per axis so that
``parent(fine_i) = fine_i // ratio`` holds (exact when ``Hf % ratio == 0``, which the
periodic-roll path in ``isotropic_model`` guards). ``ratio`` is sampled per-forward
(fixed at eval) and returned in ``stage_info`` so the processor blocks and the periodic
roll can stay registered with the fine grid.

The fine path is identical to ``SpaceBagAdaptiveDVstrideEncoder`` (bit-exact, warm
-startable). Only the coarse path and the extra ``stage_info`` keys are new.
"""

from __future__ import annotations

import math
import random
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


class TwoGridSpaceBagVstrideEncoder(SpaceBagAdaptiveDVstrideEncoder):
    """Fork of ``SpaceBagAdaptiveDVstrideEncoder`` that additionally emits a coarse
    token grid from an un-fused raw-field coarse path. ``forward`` returns
    ``(fine, stage_info)`` with ``stage_info["coarse"]`` and
    ``stage_info["coarse_ratio"]`` added."""

    def __init__(
        self,
        *args,
        coarse_ratios: Tuple[int, ...] = (2, 4),
        eval_coarse_ratio: int = 2,
        coarse_extra_div: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.coarse_ratios = tuple(coarse_ratios)
        self.eval_coarse_ratio = eval_coarse_ratio
        # ratio >= 2 guarantees the coarse grid has strictly fewer tokens than fine on
        # every non-singleton axis (coarse_count = prod ceil(Hf/ratio) < prod Hf).
        if min(self.coarse_ratios) < 2 or self.eval_coarse_ratio < 2:
            raise ValueError(
                "coarse_ratios and eval_coarse_ratio must all be >= 2 so the coarse "
                f"grid is strictly coarser than fine; got {self.coarse_ratios}, "
                f"eval={self.eval_coarse_ratio}"
            )
        self._groups = kwargs.get("groups", 12)
        norm_layer = kwargs.get("norm_layer", nn.GroupNorm)
        conv_class = CONV_FUNCS[self.spatial_dims][0]
        self.pool_func = POOL_FUNCS[self.spatial_dims]

        # Dedicated (narrow) raw-field coarse tokenizer. Same base kernels/strides as
        # the fine path (reach comes from the pre-pool, not a wider kernel); output at
        # full hidden width so coarse tokens live in the same space as fine tokens.
        c_inner = max(self._groups, (self.inner_dim // coarse_extra_div))
        c_inner = (c_inner // self._groups) * self._groups  # keep divisible by groups
        self.coarse_proj1 = conv_class(
            self.input_dim, c_inner, kernel_size=self.base_kernel1, bias=False
        )
        self.coarse_norm1 = norm_layer(self._groups, c_inner, affine=True)
        self.coarse_proj2 = conv_class(
            c_inner, self.output_dim, kernel_size=self.base_kernel2, bias=False
        )
        self.coarse_norm2 = norm_layer(self._groups, self.output_dim, affine=True)

    # ------------------------------------------------------------------ coarse path
    def _fit_axis(self, xc: Tensor, axis: int, target: int, periodic: bool) -> Tensor:
        """Center-crop or pad ``xc`` along ``axis`` to exactly ``target`` cells, via a
        single ``index_select`` (rank-agnostic; periodic -> wrap, else -> replicate
        edge). Center-crops when larger; pads symmetrically when smaller."""
        cur = xc.shape[axis]
        if cur == target:
            return xc
        if cur > target:
            start = (cur - target) // 2
            idx = torch.arange(start, start + target, device=xc.device)
        else:
            left = (target - cur) // 2
            base = torch.arange(target, device=xc.device) - left
            idx = base % cur if periodic else base.clamp(0, cur - 1)
        return xc.index_select(axis, idx)

    def _coarse_forward(
        self,
        x: Tensor,
        field_indices: Tensor,
        bcs,
        fine_spatial: Sequence[int],
        random_kernel,
        ratio: int,
    ) -> Tensor:
        """x: (T, B, C, H, W, D) padded raw input. Returns coarse (T, B, C, Hc, Wc, Dc)
        with ``Hc = ceil(Hf / ratio)`` per spatial axis (1 on singleton axes)."""
        nd = self.spatial_dims
        T = x.shape[0]
        x = rearrange(x, "T B ... -> (T B) ...")
        stride1 = tuple(random_kernel[i][0] for i in range(nd))
        stride2 = tuple(random_kernel[i][1] for i in range(nd))

        # Per-axis pool factor (1 on singleton axes -> coarse keeps that axis at 1).
        pool_k = [ratio if x.shape[-nd + a] > 1 else 1 for a in range(nd)]
        xc = self.pool_func(
            x, kernel_size=tuple(pool_k), stride=tuple(pool_k), ceil_mode=True
        )

        # Field-selected first conv (space bag); from-scratch so no weight-match scale.
        w1 = self.coarse_proj1.weight[:, field_indices]
        xc = self.adaptive_conv(xc, w1, None, stride1, (0,) * nd)
        xc = self.act(self.coarse_norm1(xc))
        xc = self.adaptive_conv(xc, self.coarse_proj2.weight, None, stride2, (0,) * nd)
        xc = self.act(self.coarse_norm2(xc))

        # Land on exactly ceil(Hf/ratio) per axis so parent = fine_idx // ratio holds.
        bcs_local = list(bcs) if bcs is not None else []
        for a in range(nd):
            fa = int(fine_spatial[a])
            target = 1 if fa == 1 else max(1, math.ceil(fa / pool_k[a]))
            periodic = (
                a < len(bcs_local)
                and int(bcs_local[a][0]) == BoundaryCondition["PERIODIC"].value
            )
            xc = self._fit_axis(xc, xc.ndim - nd + a, target, periodic)

        return rearrange(xc, "(T B) ... -> T B ...", T=T)

    def forward(
        self, x: Tensor, field_indices: Tensor, bcs=None, metadata=None, **kwargs
    ) -> Tuple[Tensor, Dict[str, Any]]:
        ratio = (
            # torch RNG (not python `random`) so gradient-checkpoint recompute reproduces
            # the same coarse ratio via preserve_rng_state -> no shape-mismatch CheckpointError.
            self.coarse_ratios[int(torch.randint(len(self.coarse_ratios), (1,)).item())]
            if self.training
            else self.eval_coarse_ratio
        )
        # Fine path is the unmodified baseline (raw x preserved for the coarse path).
        fine, stage_info = super().forward(x, field_indices, bcs, metadata, **kwargs)
        coarse = self._coarse_forward(
            x, field_indices, bcs, fine.shape[3:], kwargs["random_kernel"], ratio
        )
        stage_info = dict(stage_info)
        stage_info["coarse"] = coarse
        stage_info["coarse_ratio"] = ratio
        return fine, stage_info
