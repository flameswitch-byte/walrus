"""FMM / H-matrix spatial attention (faithful, multi-level — design study §12).

Drop-in replacement for ``FullAttention`` in the ``space_mixing`` slot of a standard
``SpaceTimeSplitBlock`` (single grid: no two-grid encoder plumbing — the coarse levels are
built INTERNALLY by pooling the fine grid). Implements the scale-by-DISTANCE axis (§12.7):
every query is global, but a key's resolution decreases with query–key distance, and each
key is handled at exactly one level (interaction list → telescoping → O(N log N)).

Structure (one joint softmax, no gate — §12.6):

    query q  ──►  NEAR : exact fine self-attention over a centered ``near_radius`` window
             └──►  FAR_ℓ (ℓ = 1..L) : query reads its parent's ANNULUS on coarse level ℓ
                   (Chebyshev distance in [far_inner, far_outer] coarse cells), where level
                   ℓ is the fine grid pooled by ``pool_base**ℓ`` with a LEARNED pooling
                   (option B, §12.6 — untied per level by default).

    out = joint_softmax(q ; [near keys] ∪ [far_1 moment keys] ∪ … ∪ [far_L moment keys])

Why this is faithful FMM and not the single-level ``hetero_multipole`` (§12.7):
  * near-field stays EXACT (high-rank near sources survive — §10.5/§10.7);
  * far-field is pooled, but only the WELL-SEPARATED annulus at each level, each region once
    (the interaction list), so there is no double-counting and cost is O(N log N);
  * one joint softmax → the near/far allocation is data-dependent and LEARNED (no gate).

Option B (learned pooling) is the default: each level's pooling is an ``nn.Linear`` over the
``pool_base**d`` children, so the far-field operator is learned per physics rather than fixed
to the inverse-Laplacian (§12.6). Set ``learned_pool=False`` for the fixed mean-pool (a
weaker, option-A-flavoured control). BC-aware throughout: periodic axes wrap, wall axes clamp
(validity masks), reusing the two-grid neighbour machinery.

Init: layer-scale on ``attn_out``/``ff_out`` → the block starts ≈ identity (residual
dominates), so a flat ``FullAttention`` checkpoint loads cleanly (shared FF/Q/K/V projection
names) and the far-field is learned from there. NOTE: this is a warm *init*, not behavioural
equivalence to flat attention (near-field reach ≠ global) — see §12.11.
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


class FMMAttention(nn.Module):
    """Multi-level FMM / H-matrix attention; drop-in ``space_mixing`` (single grid)."""

    def __init__(
        self,
        hidden_dim: int = 256,
        mlp_dim: int | None = None,
        num_heads: int = 8,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        bias_type: str = "rel",
        max_d: int = 3,
        weight_tied_axes: bool = True,
        gradient_checkpointing: bool = False,
        norm_layer: Callable = RMSGroupNorm,
        # --- FMM specific ---
        near_radius: int = 3,  # fine exact near-field half-width (leaf + adjacent leaves)
        pool_base: int = 2,  # coarsening factor per level (level ℓ pool = pool_base**ℓ)
        far_inner: int = 2,  # annulus inner radius (coarse cells) — well-separated start
        far_outer: int = 3,  # annulus outer radius (coarse cells) — handed to coarser level
        max_levels: int | None = None,  # cap on far levels (None = auto from grid size)
        learned_pool: bool = True,  # option B (learned) vs option A (fixed mean pool)
        tie_levels: bool = False,  # share pooling weights across levels (§12.9: keep False)
        global_mop_up: bool = False,  # add a shared coarsest level for full global reach (finding 3)
        max_token_grid: int | None = None,  # size pooling exactly for a grid this large (finding 2)
        spatial_dims: int = 3,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim {hidden_dim} not divisible by num_heads {num_heads}"
            )
        if far_inner < 1 or far_outer < far_inner:
            raise ValueError(
                f"need 1 <= far_inner <= far_outer, got {far_inner}, {far_outer}"
            )
        if pool_base < 2:
            raise ValueError(f"pool_base must be >= 2, got {pool_base}")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.mlp_dim = mlp_dim or hidden_dim * 4
        if self.mlp_dim % 2 != 0:
            raise ValueError(f"mlp_dim must be even, got {self.mlp_dim}")
        self.max_d = max_d
        self.spatial_dims = spatial_dims
        self.near_radius = int(near_radius)
        self.pool_base = int(pool_base)
        self.far_inner = int(far_inner)
        self.far_outer = int(far_outer)
        self.max_levels = max_levels
        self.learned_pool = bool(learned_pool)
        self.tie_levels = bool(tie_levels)
        self.global_mop_up = bool(global_mop_up)

        # --- projections (FF + Q + K + V), names match FullAttention for warm-start ---
        self.norm1 = norm_layer(num_heads, hidden_dim, affine=True)
        self.fused_dims = (self.mlp_dim, hidden_dim, hidden_dim, hidden_dim)
        self.fused_ff_qkv = nn.Linear(hidden_dim, sum(self.fused_dims))
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)
        self.coarse_k_norm = nn.LayerNorm(self.head_dim)  # pooled (far) keys

        self.activation = SwiGLU()
        self.ff_out = nn.Linear(self.mlp_dim // 2, hidden_dim)
        self.attn_out = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self._layerscale_init(self.ff_out, layer_scale_init_value)
        self._layerscale_init(self.attn_out, layer_scale_init_value)

        # --- learned pooling (option B): per-level Linear over pool_base**d children ----
        # Each level pools the PREVIOUS level by ``pool_base`` (recursive upward pass), so
        # every Linear has the same in-dim and they can be untied per level cheaply. Size the
        # count to avoid unused Linears (finding 2): ``max_levels`` if given, else exactly the
        # number a ``max_token_grid`` needs, else a safe default of 8.
        if max_levels is not None:
            self._max_built_levels = max_levels
        elif max_token_grid is not None:
            self._max_built_levels = max(1, self._natural_level_count(int(max_token_grid)))
        else:
            self._max_built_levels = 8
        # Untied pooling reserves ONE extra slot (index == _max_built_levels) for the global
        # mop-up level; tied pooling reuses slot 0 for everything.
        self._mop_idx = 0 if self.tie_levels else self._max_built_levels
        if self.learned_pool:
            r = self.pool_base ** self.spatial_dims  # children per coarse cell
            if self.tie_levels:
                n_proj = 1
            else:
                n_proj = self._max_built_levels + (1 if self.global_mop_up else 0)
            self.pool_k = nn.ModuleList(
                [nn.Linear(r * self.head_dim, self.head_dim, bias=False) for _ in range(n_proj)]
            )
            self.pool_v = nn.ModuleList(
                [nn.Linear(r * self.head_dim, self.head_dim, bias=False) for _ in range(n_proj)]
            )

        self.rotary_emb = RotaryEmbedding(
            self.head_dim // 4, freqs_for="pixel", max_freq=256
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    # ------------------------------------------------------------------ init / rope
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

    def get_rotary_embedding(self, n, device):
        return self.rotary_emb(n, device=device)

    # ------------------------------------------------------------------ small helpers
    @staticmethod
    def _periodic_flags(bcs, nd: int) -> List[bool]:
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
        return rearrange(t, "b he a b2 c2 d -> b he (a b2 c2) d")

    def _natural_level_count(self, grid: int) -> int:
        """How many far levels a cubic grid of side ``grid`` naturally produces (UNCAPPED) —
        used to size the pooling ModuleList exactly when ``max_token_grid`` is given (finding
        2). Mirrors the level loop in ``_levels`` without the ``_max_built_levels`` cap."""
        if self.max_levels == 0:
            return 0
        min_extent = 2 * self.far_outer + 1
        n, ell = 0, 1
        while True:
            p = self.pool_base ** ell
            if (grid + p - 1) // p < min_extent:
                break
            n += 1
            if (self.far_outer + 1) * p - 1 >= grid:
                break
            ell += 1
        return n

    def _levels(self, sizes: Sequence[int]) -> List[int]:
        """Pool factors p_ℓ = pool_base**ℓ for the far levels. A level is included only if
        its coarse grid is large enough that the ``±far_outer`` annulus window does NOT wrap
        on itself (extent >= 2*far_outer+1) — otherwise periodic wrap aliases offsets (e.g.
        +3 ≡ -1) and double-counts. This bounds the far reach to ~one ring short of the full
        domain (the single longest-wavelength coupling is dropped, v1 — see module note)."""
        if self.max_levels == 0:
            return []  # near-only ablation
        min_extent = 2 * self.far_outer + 1
        diam = max(sizes)
        levels: List[int] = []
        ell = 1
        while True:
            p = self.pool_base ** ell
            coarse_extent = max((s + p - 1) // p for s in sizes if s > 1)
            if coarse_extent < min_extent:
                break
            levels.append(p)
            reach = (self.far_outer + 1) * p - 1  # fine distance this annulus reaches
            if reach >= diam or len(levels) >= self._max_built_levels:
                break
            ell += 1
        return levels

    def _neighbors(
        self, grid: Tensor, radius: int, periodic: Sequence[bool],
        inner: int = 0,
    ) -> Tuple[Tensor, Tensor]:
        """Centered ``(2*radius+1)^d`` neighborhood per cell via offset rolls. If
        ``inner > 0`` keep only the ANNULUS (Chebyshev distance in [inner, radius]).
        Returns neighbors (B,he,H,W,D,K,c) and validity+annulus mask (H,W,D,K)."""
        nd = 3
        sizes = grid.shape[2:5]
        dev = grid.device
        rads = [radius if sizes[a] > 1 else 0 for a in range(nd)]
        ranges = [list(range(-r, r + 1)) for r in rads]
        neigh, masks = [], []
        for off in itertools.product(*ranges):
            cheb = max((abs(o) for o in off), default=0)
            keep = inner <= cheb  # outer already bounded by the ranges
            g = grid
            valid = torch.ones(sizes, device=dev)
            for a, o in enumerate(off):
                if o != 0:
                    g = torch.roll(g, shifts=-o, dims=2 + a)
                    if not periodic[a]:
                        pos = torch.arange(sizes[a], device=dev) + o
                        ok = (pos >= 0) & (pos < sizes[a])
                        shape = [1, 1, 1]
                        shape[a] = sizes[a]
                        valid = valid * ok.view(shape).float()
            if not keep:
                valid = valid * 0.0
            neigh.append(g)
            masks.append(valid)
        out = torch.stack(neigh, dim=5)
        mask = torch.stack(masks, dim=3).bool()
        return out, mask

    @staticmethod
    def _upsample_to_fine(
        t: Tensor, pool: int, target: Sequence[int], spatial_axes
    ) -> Tensor:
        """Repeat each coarse cell ``pool`` times per axis (parent→children alignment),
        then crop to ``target``."""
        for ax, tgt in zip(spatial_axes, target):
            if t.shape[ax] != 1 or tgt == 1:
                if pool > 1:
                    t = t.repeat_interleave(pool, dim=ax)
                if t.shape[ax] > tgt:
                    t = t.narrow(ax, 0, tgt)
        return t

    def _gather_children(self, grid: Tensor, ratio: int) -> Tensor:
        """Group the grid into non-overlapping ``ratio^3`` blocks (zero-padded at ragged
        boundaries AND on singleton axes, so the child count R = ratio**3 is FIXED across
        2D/3D inputs — required because one pooling Linear serves both).
        grid (B,he,H,W,D,c) -> (B,he,Hc,Wc,Dc, ratio**3, c)."""
        sizes = grid.shape[2:5]
        rs = [ratio, ratio, ratio]  # force on all axes -> constant R
        coarse = [(sizes[a] + rs[a] - 1) // rs[a] for a in range(3)]
        gx = grid
        for a in range(3):
            ax = 2 + a
            pad = coarse[a] * rs[a] - gx.shape[ax]
            if pad > 0:
                gshape = list(gx.shape)
                gshape[ax] = pad
                gx = torch.cat([gx, gx.new_zeros(gshape)], dim=ax)
        rh, rw, rd = rs
        gx = rearrange(
            gx,
            "b he (hc rh) (wc rw) (dc rd) c -> b he hc wc dc (rh rw rd) c",
            rh=rh, rw=rw, rd=rd,
        )
        return gx

    def _pool_once(self, kf: Tensor, vf: Tensor, level_idx: int):
        """Pool (kf, vf) by ``pool_base`` into the next coarse level. Option B: learned
        Linear over the children; option A: mean pool."""
        kc_ch = self._gather_children(kf, self.pool_base)  # (B,he,Hc,Wc,Dc,R,c)
        vc_ch = self._gather_children(vf, self.pool_base)
        if self.learned_pool:
            idx = 0 if self.tie_levels else min(level_idx, len(self.pool_k) - 1)
            B, he, Hc, Wc, Dc, R, c = kc_ch.shape
            flat_k = rearrange(kc_ch, "b he hc wc dc r c -> b he hc wc dc (r c)")
            flat_v = rearrange(vc_ch, "b he hc wc dc r c -> b he hc wc dc (r c)")
            kc = self.pool_k[idx](flat_k)
            vc = self.pool_v[idx](flat_v)
        else:
            kc = kc_ch.mean(dim=5)
            vc = vc_ch.mean(dim=5)
        return kc, vc

    def _joint_attention(self, q: Tensor, blocks: List[tuple]) -> Tensor:
        """One softmax over key/value blocks. q (B,he,X,Y,Z,c). Block is
        ('shared', k, v) with k/v (B,he,N,c), or ('windowed', k, v, mask) with k/v
        (B,he,X,Y,Z,K,c) and mask (X,Y,Z,K). (Copied from TwoGridCrossScaleAttention.)"""
        scale = self.head_dim ** -0.5
        scores = []
        for blk in blocks:
            if blk[0] == "shared":
                s = torch.einsum("bhxyzc,bhnc->bhxyzn", q, blk[1]) * scale
            else:
                s = torch.einsum("bhxyzc,bhxyzkc->bhxyzk", q, blk[1]) * scale
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
    def forward(self, x, bcs=None, return_att: bool = False):
        # x (fine): (B, C, H, W, D).  B == T * B_orig.
        B, C, H, W, D = x.shape
        periodic = self._periodic_flags(bcs, 3)
        inp = x.clone()  # defensive (matches FullAttention); no in-place below, but safe

        feat = rearrange(self.norm1(x), "b c h w d -> b h w d c")
        ff, qf, kf, vf = self.fused_ff_qkv(feat).split(self.fused_dims, dim=-1)
        qf, kf, vf = map(self._to_heads, (qf, kf, vf))
        qf, kf = self.q_norm(qf), self.k_norm(kf)
        # FAR path pools the PRE-RoPE keys (finding 1): pooling already-fine-RoPE'd keys
        # would bake each child's fine rotation into the coarse token, which then composes
        # with the coarse rotation below → double-encoded position. Keep a pre-RoPE copy so
        # the coarse keys carry ONLY their coarse-grid rotation.
        kf_pre = kf
        pos_f = self.rotary_emb.get_axial_freqs(H, W, D)
        qf, kf = apply_rotary_emb(pos_f, qf), apply_rotary_emb(pos_f, kf)

        # NEAR block: exact fine self-attention over a centered near_radius window.
        kf_n, m_near = self._neighbors(kf, self.near_radius, periodic)
        vf_n, _ = self._neighbors(vf, self.near_radius, periodic)
        blocks = [("windowed", kf_n, vf_n, m_near)]

        # FAR blocks: recursively pool the PRE-RoPE keys, then each level reads its annulus.
        ck, cv = kf_pre, vf  # current (finest) level keys (pre-RoPE) / values, per head
        for level_idx, pool in enumerate(self._levels((H, W, D))):
            ck, cv = self._pool_once(ck, cv, level_idx)  # pool previous level by pool_base
            Hc, Wc, Dc = ck.shape[2:5]
            pos_c = self.rotary_emb.get_axial_freqs(Hc, Wc, Dc)
            ck_r = apply_rotary_emb(pos_c, self.coarse_k_norm(ck))
            # annulus neighborhood on the coarse grid (distance in [far_inner, far_outer])
            kc_n, m_far = self._neighbors(
                ck_r, self.far_outer, periodic, inner=self.far_inner
            )
            vc_n, _ = self._neighbors(cv, self.far_outer, periodic, inner=self.far_inner)
            # align each fine query to its parent's annulus (repeat by the total pool)
            kc_up = self._upsample_to_fine(kc_n, pool, (H, W, D), (2, 3, 4))
            vc_up = self._upsample_to_fine(vc_n, pool, (H, W, D), (2, 3, 4))
            m_up = self._upsample_to_fine(m_far, pool, (H, W, D), (0, 1, 2))
            blocks.append(("windowed", kc_up, vc_up, m_up))

        # Optional GLOBAL MOP-UP (finding 3): one extra pooling step, attended as a SHARED
        # block (every query sees all its cells, each cell once → alias-free), restoring the
        # longest-wavelength coupling the annulus levels drop (§6.3). Cells sit at coarse
        # positions (RoPE'd once). Introduces a mild, controlled overlap with the finest
        # annuli; matters most in true-3D where only ~1 annulus level fits. Off by default.
        if self.global_mop_up and self.max_levels != 0:
            gk, gv = self._pool_once(ck, cv, self._mop_idx)  # pre-RoPE ck → coarse
            Hg, Wg, Dg = gk.shape[2:5]
            pos_g = self.rotary_emb.get_axial_freqs(Hg, Wg, Dg)
            gk = apply_rotary_emb(pos_g, self.coarse_k_norm(gk))
            blocks.append(("shared", self._flat(gk), self._flat(gv)))

        att = self._joint_attention(qf, blocks)
        att = rearrange(att, "b he h w d c -> b h w d (he c)")
        att_out = self.attn_out(att)
        x = self.drop_path(att_out + self.ff_out(self.activation(ff)))
        x = rearrange(x, "b h w d c -> b c h w d") + inp
        return x, []
