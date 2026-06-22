"""Two-grid cross-scale spatial attention (Design B).

A persistent coarse token grid (from ``TwoGridSpaceBagVstrideEncoder``) co-evolves with
the fine grid across the processor stack. Each block, MERA / disentangle-first:

    (A) COARSE UPDATE   coarse self-attends (``coarse_window``: global all<->all, or a
                        w^d local window) and -- if ``coarse_coupling == 'bidirectional'``
                        -- also reads its own ratio^d fine children (restriction), with an
                        optional (mean||var) saliency summary (``restrict_moments``).
    (B) FINE UPDATE     each fine token attends jointly (one softmax) to its fixed local
                        ``fine_window^d`` neighborhood on the fine grid AND the coarse
                        grid (prolongation).

Knobs (each axis independent):
  * ``fine_window``   (int) -- fine LOCAL self-attention extent (k^d keys/token).
  * ``coarse_window`` (int w | 'global') -- COARSE SELF-attention locality ONLY:
      'global' -> every coarse token attends every other (carries global reach);
      w        -> coarse attends its centered w^d coarse neighborhood (reach grows
                  with stack depth, CNN-style -- NO single-block global reach).
  * ``prolong_window`` (int p) -- fine reads its parent's centered p^d coarse
      neighborhood and upsamples it back to the fine grid (smooth interpolation /
      prolongation; p=1 == single-parent injection, which can seam at cell edges).
  * ``coarse_coupling`` ('one_way' | 'bidirectional') -- restriction direction:
      'bidirectional' -> coarse ALSO reads its own ratio^d fine children;
      'one_way'       -> coarse refines its raw-field seed only.
  * ``restrict_moments`` (bool) -- when bidirectional, append a per-cell (mean||var)
      summary as ONE extra attendable restriction key/value. var = the sub-grid energy
      pooling discards, so the coarse token can route on "a sharp feature was here"
      (the §10 restriction-failure mitigation) instead of only averaging children.

Cross-scale reads are ALWAYS strictly local (fine<->parent neighborhood,
coarse<->own children) and do NOT depend on ``coarse_window`` -- ``coarse_window``
governs coarse self-attention alone. Per-block cost is O(N_f) + O(N_c^2): genuinely
linear in fine tokens. Global reach is carried by global coarse self-attention.

All attention goes through ONE ``_joint_attention(q, blocks)`` over a list of key/value
"blocks" that are either ``shared`` (same keys for every query, e.g. all-coarse) or
``windowed`` (per-query neighborhood). No SDPA/manual fork.

RoPE: shared ``RotaryEmbedding`` (``freqs_for='pixel'`` -> per-axis linspace(-1,1)), so
fine and coarse positions share one normalized frame; fine.coarse cross phases are
consistent, and the periodic roll in ``isotropic_model`` rolls both grids in registration.
"""

from __future__ import annotations

import itertools
import math
from typing import Callable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from the_well.data.datasets import BoundaryCondition
from timm.layers import DropPath
from torch import Tensor
from torch.nn import init

from ..shared_utils.lr_rope_temporary import RotaryEmbedding, apply_rotary_emb
from ..shared_utils.normalization import RMSGroupNorm


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class TwoGridCrossScaleAttention(nn.Module):
    """Persistent-coarse cross-scale attention; drop-in for the ``space_mixing`` slot of
    a two-grid ``SpaceTimeSplitBlock`` (see module docstring)."""

    def __init__(
        self,
        hidden_dim: int = 256,
        mlp_dim: int | None = None,
        num_heads: int = 8,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        gradient_checkpointing: bool = False,
        norm_layer: Callable = RMSGroupNorm,
        # --- cross-scale specific ---
        fine_window: int = 3,
        coarse_window="global",  # coarse SELF-attn locality: int w | "global"
        prolong_window: int = 3,  # fine reads parent's p^d coarse nbhd; 1 = injection
        coarse_coupling: str = "bidirectional",  # "one_way" | "bidirectional"
        restrict_moments: bool = True,  # add per-cell (mean||var) saliency to restriction
        spatial_dims: int = 3,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim {hidden_dim} not divisible by num_heads {num_heads}"
            )
        if coarse_coupling not in ("one_way", "bidirectional"):
            raise ValueError(
                f"coarse_coupling must be 'one_way' or 'bidirectional', got "
                f"{coarse_coupling!r}"
            )
        self.bidirectional = coarse_coupling == "bidirectional"
        self.restrict_moments = bool(restrict_moments)

        # coarse_window: "global" -> attend all; positive int -> windowed neighborhood.
        if isinstance(coarse_window, str):
            if coarse_window.lower() != "global":
                raise ValueError(
                    f"coarse_window str must be 'global', got {coarse_window!r}"
                )
            self.coarse_global = True
            self.coarse_window = None
        else:
            self.coarse_global = False
            self.coarse_window = int(coarse_window)
            if self.coarse_window < 1:
                raise ValueError(f"coarse_window int must be >= 1, got {coarse_window}")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.mlp_dim = mlp_dim or hidden_dim * 4
        if self.mlp_dim % 2 != 0:
            raise ValueError(f"mlp_dim must be even, got {self.mlp_dim}")
        self.spatial_dims = spatial_dims
        self.fine_window = fine_window
        self.prolong_window = int(prolong_window)
        if self.prolong_window < 1:
            raise ValueError(f"prolong_window must be >= 1, got {prolong_window}")

        # --- fine: fused FF + Q + K_fine + V_fine (mirrors FullAttention) ---
        self.norm_f = norm_layer(num_heads, hidden_dim, affine=True)
        self.fused_dims = (self.mlp_dim, hidden_dim, hidden_dim, hidden_dim)
        self.fused_ff_qkv = nn.Linear(hidden_dim, sum(self.fused_dims))
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)

        # --- coarse self (+ restriction read of fine): Q + K + V ---
        self.norm_c = norm_layer(num_heads, hidden_dim, affine=True)
        self.coarse_qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.cq_norm = nn.LayerNorm(self.head_dim)
        self.ck_norm = nn.LayerNorm(self.head_dim)
        self.coarse_out = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # per-cell (mean || var) sub-grid saliency, attended as 1 extra restriction key/value
        if self.restrict_moments:
            self.restrict_moment_kv = nn.Linear(2 * hidden_dim, 2 * hidden_dim)
            # match the per-head key norm every other key gets (kf/kc/kc2) so the moment
            # logit scale is calibrated against the children in the joint softmax
            self.km_norm = nn.LayerNorm(self.head_dim)

        # --- keys/values from the UPDATED coarse, read by fine (prolongation) ---
        self.norm_cross = norm_layer(num_heads, hidden_dim, affine=True)
        self.cross_kv = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.kc_norm = nn.LayerNorm(self.head_dim)

        # --- outputs (layer-scale init -> block starts near identity) ---
        self.activation = SwiGLU()
        self.ff_out = nn.Linear(self.mlp_dim // 2, hidden_dim)
        self.attn_out = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self._layerscale_init(self.ff_out, layer_scale_init_value)
        self._layerscale_init(self.attn_out, layer_scale_init_value)
        self._layerscale_init(self.coarse_out, layer_scale_init_value)

        self.rotary_emb = RotaryEmbedding(
            self.head_dim // 4, freqs_for="pixel", max_freq=256
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    @staticmethod
    def _layerscale_init(linear: nn.Linear, lsv: float) -> None:
        init.kaiming_uniform_(linear.weight, a=math.sqrt(5) / lsv)
        if linear.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(linear.weight)
            bound = 1 / math.sqrt(fan_in) * lsv
            init.uniform_(linear.bias, -bound, bound)

    def make_rope_learnable(self, per_axis: bool = False) -> None:
        if hasattr(self, "rotary_emb"):
            self.rotary_emb.make_learnable(per_axis)

    # ------------------------------------------------------------------ small helpers
    def _periodic_flags(self, bcs, nd: int) -> List[bool]:
        flags = [False] * nd
        if bcs is None:
            return flags
        try:
            per_dim = bcs[0]
            for a in range(min(nd, len(per_dim))):
                flags[a] = int(per_dim[a][0]) == BoundaryCondition["PERIODIC"].value
        except (TypeError, IndexError, KeyError):
            pass
        return flags

    def _to_heads(self, t: Tensor) -> Tensor:
        return rearrange(t, "b h w d (he c) -> b he h w d c", he=self.num_heads)

    @staticmethod
    def _flat(t: Tensor) -> Tensor:
        # (B, he, A, B2, C2, c) -> (B, he, A*B2*C2, c)
        return rearrange(t, "b he a b2 c2 d -> b he (a b2 c2) d")

    def _neighbors(
        self, grid: Tensor, window: int, periodic: Sequence[bool]
    ) -> Tuple[Tensor, Tensor]:
        """Gather a centered ``window^d`` spatial neighborhood for every cell, via offset
        rolls (works for any trailing feature dims). ``grid``: (B, he, H, W, D, *feat) ->
        neighbors (B, he, H, W, D, K, *feat) and a validity mask (H, W, D, K) bool (False
        where a neighbor falls outside a non-periodic boundary)."""
        nd = 3
        sizes = grid.shape[2:5]
        dev = grid.device
        wins = [window if sizes[a] > 1 else 1 for a in range(nd)]
        ranges = [list(range(-(w // 2), w // 2 + 1)) for w in wins]
        neigh, masks = [], []
        for off in itertools.product(*ranges):
            g = grid
            valid = torch.ones(sizes, device=dev)
            for a, o in enumerate(off):
                if o != 0:
                    g = torch.roll(
                        g, shifts=-o, dims=2 + a
                    )  # neighbor[pos] = grid[pos+o]
                    if not periodic[a]:
                        pos = torch.arange(sizes[a], device=dev) + o
                        ok = (pos >= 0) & (pos < sizes[a])
                        shape = [1, 1, 1]
                        shape[a] = sizes[a]
                        valid = valid * ok.view(shape).float()
            neigh.append(g)
            masks.append(valid)
        out = torch.stack(neigh, dim=5)  # (B, he, H, W, D, K, *feat)
        mask = torch.stack(masks, dim=3).bool()  # (H, W, D, K)
        return out, mask

    @staticmethod
    def _upsample_to_fine(
        t: Tensor, pools: Sequence[int], target: Sequence[int], spatial_axes
    ) -> Tensor:
        """Nearest-upsample a coarse-grid tensor to the fine grid (repeat each coarse cell
        ``pool`` times per axis, then crop to ``target``)."""
        for ax, p, tgt in zip(spatial_axes, pools, target):
            if p > 1:
                t = t.repeat_interleave(p, dim=ax)
            if t.shape[ax] > tgt:
                t = t.narrow(ax, 0, tgt)
        return t

    def _gather_children(
        self, grid: Tensor, ratios: Sequence[int], coarse_sizes: Sequence[int]
    ) -> Tuple[Tensor, Tensor]:
        """Group the fine grid into the non-overlapping ``ratio^d`` block under each
        coarse cell (the restriction stencil). grid: (B, he, Hf, Wf, Df, c) -> children
        (B, he, Hc, Wc, Dc, R, c) and mask (Hc, Wc, Dc, R) (False on zero-pad children
        when Hf isn't an exact multiple of ratio)."""
        valid = grid.new_ones((1, 1, *grid.shape[2:5], 1))
        gx = grid
        for a in range(3):
            ax = 2 + a
            pad = coarse_sizes[a] * ratios[a] - gx.shape[ax]
            if pad > 0:
                gshape = list(gx.shape)
                gshape[ax] = pad
                gx = torch.cat([gx, gx.new_zeros(gshape)], dim=ax)
                vshape = list(valid.shape)
                vshape[ax] = pad
                valid = torch.cat([valid, valid.new_zeros(vshape)], dim=ax)
        rh, rw, rd = ratios
        gx = rearrange(
            gx,
            "b he (hc rh) (wc rw) (dc rd) c -> b he hc wc dc (rh rw rd) c",
            rh=rh,
            rw=rw,
            rd=rd,
        )
        valid = rearrange(
            valid,
            "b he (hc rh) (wc rw) (dc rd) c -> b he hc wc dc (rh rw rd c)",
            rh=rh,
            rw=rw,
            rd=rd,
        )
        return gx, valid[0, 0].bool()

    # ------------------------------------------------------------------ cross blocks
    def _coarse_self_block(self, kc, vc, periodic):
        """Coarse self-attention key/value block. 'global' -> all coarse (shared);
        int w -> centered w^d coarse neighborhood (windowed). This is the ONLY place
        ``coarse_window`` is consumed."""
        if self.coarse_global:
            return ("shared", self._flat(kc), self._flat(vc))
        kc_n, m = self._neighbors(kc, self.coarse_window, periodic)
        vc_n, _ = self._neighbors(vc, self.coarse_window, periodic)
        return ("windowed", kc_n, vc_n, m)

    def _coarse_reads_children_block(self, kf, vf, ratios, coarse_sizes):
        """Restriction: coarse reads its OWN ratio^d fine children (no window). q-grid =
        coarse, so the mask is (Hc,Wc,Dc,R)."""
        kc, m = self._gather_children(kf, ratios, coarse_sizes)  # (B,he,Hc,Wc,Dc,R,c)
        vc, _ = self._gather_children(vf, ratios, coarse_sizes)
        return ("windowed", kc, vc, m)

    def _restriction_moments(self, feat, ratios, coarse_sizes):
        """Per-coarse-cell (mean || var) of the fine features over each cell's ratio^d
        children -- the resolved field plus the sub-grid energy that pooling discards.
        feat: (B, H, W, D, C) -> (B, Hc, Wc, Dc, 2C). Masked so ragged-boundary zero-pad
        children don't bias the statistics."""
        g = rearrange(feat, "b h w d c -> b 1 h w d c")
        ch, m = self._gather_children(g, ratios, coarse_sizes)  # (B,1,Hc,Wc,Dc,R,C)
        ch = ch[:, 0]  # (B,Hc,Wc,Dc,R,C)
        mf = m.to(ch.dtype)[None, ..., None]  # (1,Hc,Wc,Dc,R,1)
        cnt = mf.sum(dim=4).clamp_min(1.0)  # (1,Hc,Wc,Dc,1)
        mean = (ch * mf).sum(dim=4) / cnt  # (B,Hc,Wc,Dc,C)
        var = ((ch - mean.unsqueeze(4)) ** 2 * mf).sum(dim=4) / cnt
        return torch.cat([mean, var], dim=-1)  # (B,Hc,Wc,Dc,2C)

    def _augment_with_moments(self, block, featf, ratios, coarse_sizes, pos_c):
        """Append the (mean||var) saliency summary as ONE extra attendable key/value to a
        children-restriction block, so the coarse query can route on sub-cell sharpness."""
        _, kc, vc, m = block
        moments = self._restriction_moments(featf, ratios, coarse_sizes)
        k_m, v_m = self.restrict_moment_kv(moments).split(self.hidden_dim, dim=-1)
        k_m = apply_rotary_emb(
            pos_c, self.km_norm(self._to_heads(k_m))
        )  # (B,he,Hc,Wc,Dc,c)
        v_m = self._to_heads(v_m)
        kc = torch.cat([kc, k_m.unsqueeze(5)], dim=5)  # (B,he,Hc,Wc,Dc,R+1,c)
        vc = torch.cat([vc, v_m.unsqueeze(5)], dim=5)
        m = torch.cat([m, m.new_ones((*m.shape[:3], 1))], dim=3)  # (Hc,Wc,Dc,R+1)
        return ("windowed", kc, vc, m)

    def _fine_reads_parent_block(self, kc2, vc2, ratios, fine_sizes, periodic):
        """Prolongation: fine reads its parent's centered ``prolong_window^d`` coarse
        neighborhood, upsampled (nearest) to the fine grid. p=1 -> single parent."""
        kc_n, m = self._neighbors(
            kc2, self.prolong_window, periodic
        )  # (B,he,Hc,Wc,Dc,K,c)
        vc_n, _ = self._neighbors(vc2, self.prolong_window, periodic)
        kc_n = self._upsample_to_fine(kc_n, ratios, fine_sizes, (2, 3, 4))
        vc_n = self._upsample_to_fine(vc_n, ratios, fine_sizes, (2, 3, 4))
        m = self._upsample_to_fine(m, ratios, fine_sizes, (0, 1, 2))
        return ("windowed", kc_n, vc_n, m)

    def _joint_attention(self, q: Tensor, blocks: List[tuple]) -> Tensor:
        """One softmax over a list of key/value blocks. q: (B,he,X,Y,Z,c). Each block is
        ('shared', k, v) with k/v (B,he,N,c) [same keys for every query], or
        ('windowed', k, v, mask) with k/v (B,he,X,Y,Z,K,c) and mask (X,Y,Z,K)."""
        scale = self.head_dim**-0.5
        scores = []
        for blk in blocks:
            if blk[0] == "shared":
                s = torch.einsum("bhxyzc,bhnc->bhxyzn", q, blk[1]) * scale
            else:
                s = torch.einsum("bhxyzc,bhxyzkc->bhxyzk", q, blk[1]) * scale
                # Fill in the score's own dtype: under bf16 autocast `s` is bf16 while
                # `q` stays fp32 (off the norm layer), and finfo(fp32).min overflows bf16.
                s = s.masked_fill(
                    ~blk[3].unsqueeze(0).unsqueeze(0), torch.finfo(s.dtype).min
                )
            scores.append(s)
        attn = torch.softmax(torch.cat(scores, dim=-1), dim=-1)
        parts = attn.split([s.shape[-1] for s in scores], dim=-1)
        out = None
        for part, blk in zip(parts, blocks):
            if blk[0] == "shared":
                o = torch.einsum("bhxyzn,bhnc->bhxyzc", part, blk[2])
            else:
                o = torch.einsum("bhxyzk,bhxyzkc->bhxyzc", part, blk[2])
            out = o if out is None else out + o
        return out

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        x: Tensor,
        bcs=None,
        coarse: Tensor = None,
        coarse_ratio: int = 2,
        return_att: bool = False,
    ):
        # x (fine): (B, C, H, W, D);  coarse: (B, C, Hc, Wc, Dc).  B == T*B_orig.
        assert coarse is not None, "TwoGridCrossScaleAttention requires a coarse grid"
        B, C, H, W, D = x.shape
        Hc, Wc, Dc = coarse.shape[-3:]
        nd = self.spatial_dims
        periodic = self._periodic_flags(bcs, nd)
        ratios = [coarse_ratio if (H, W, D)[a] > 1 else 1 for a in range(nd)]
        inp_f, inp_c = x, coarse

        # project fine (FF + q + k + v)
        featf = rearrange(self.norm_f(x), "b c h w d -> b h w d c")
        ff, qf, kf, vf = self.fused_ff_qkv(featf).split(self.fused_dims, dim=-1)
        qf, kf, vf = map(self._to_heads, (qf, kf, vf))
        qf, kf = self.q_norm(qf), self.k_norm(kf)
        pos_f = self.rotary_emb.get_axial_freqs(H, W, D)
        qf, kf = apply_rotary_emb(pos_f, qf), apply_rotary_emb(pos_f, kf)

        # project coarse self (q + k + v)
        featc = rearrange(self.norm_c(coarse), "b c h w d -> b h w d c")
        qc, kc, vc = self.coarse_qkv(featc).split(self.hidden_dim, dim=-1)
        qc, kc, vc = map(self._to_heads, (qc, kc, vc))
        qc, kc = self.cq_norm(qc), self.ck_norm(kc)
        pos_c = self.rotary_emb.get_axial_freqs(Hc, Wc, Dc)
        qc, kc = apply_rotary_emb(pos_c, qc), apply_rotary_emb(pos_c, kc)

        # (A) coarse update: coarse self-attn (global or w^d local) + optional
        # restriction (coarse reads its own ratio^d children).
        blocks_c = [self._coarse_self_block(kc, vc, periodic)]
        if self.bidirectional:
            rblk = self._coarse_reads_children_block(kf, vf, ratios, (Hc, Wc, Dc))
            if self.restrict_moments:
                rblk = self._augment_with_moments(
                    rblk, featf, ratios, (Hc, Wc, Dc), pos_c
                )
            blocks_c.append(rblk)
        c_out = self._joint_attention(qc, blocks_c)
        c_out = rearrange(c_out, "b he hc wc dc c -> b hc wc dc (he c)")
        coarse = inp_c + rearrange(
            self.coarse_out(c_out), "b hc wc dc c -> b c hc wc dc"
        )

        # re-project the UPDATED coarse as keys/values for fine (prolongation)
        featc2 = rearrange(self.norm_cross(coarse), "b c h w d -> b h w d c")
        kc2, vc2 = self.cross_kv(featc2).split(self.hidden_dim, dim=-1)
        kc2, vc2 = map(self._to_heads, (kc2, vc2))
        kc2 = apply_rotary_emb(pos_c, self.kc_norm(kc2))

        # (B) fine update: local fine self-attn (windowed) + parent-neighborhood read.
        kf_n, m_f = self._neighbors(kf, self.fine_window, periodic)
        vf_n, _ = self._neighbors(vf, self.fine_window, periodic)
        blocks_f = [
            ("windowed", kf_n, vf_n, m_f),
            self._fine_reads_parent_block(kc2, vc2, ratios, (H, W, D), periodic),
        ]
        f_out = self._joint_attention(qf, blocks_f)
        f_out = rearrange(f_out, "b he h w d c -> b h w d (he c)")
        f_out = self.attn_out(f_out)
        x = self.drop_path(f_out + self.ff_out(self.activation(ff)))
        x = rearrange(x, "b h w d c -> b c h w d") + inp_f
        return x, coarse, []
