"""Heterogeneous multi-reach spatial attention.

Drop-in replacement for ``FullAttention`` where each head gets a different *reach*:

  - ``n_local``     heads: centered box window of radius ``local_radius``.
  - ``n_strided``   heads: dilated neighborhood (DiNAT form) of radius ``strided_radius``
                           at a per-head stride (one entry of ``strided_strides`` each).
  - ``n_multipole`` heads: cheap long-range via **moment-preserving coarse keys** — each
                           neighborhood is summarized by its multipole moments
                           (monopole ‖ dipole ‖ quadrupole) and the query attends over
                           those coarse tokens (pooled-KV cross attention). See
                           knowledge_base/walrus_attention_design.md §13.2a.
  - ``n_full``      heads: unmasked global attention (exact, data-dependent).

Design (see §13.2 / §13.7 / §13.8):
  * local/strided/full use **masked-dense SDPA** (per-head boolean reach masks, one call).
  * multipole uses a **pooled-KV** path: avg-pool moments → coarse tokens → cross attention.
  * **Configurable allocation**: ``n_local + n_strided + n_multipole + n_full == num_heads``.
    Head order is local | strided | full | multipole; the masked-dense group is the first
    ``M = num_heads - n_multipole`` heads. ``n_strided = n_multipole = 0`` ablates the
    mid-range slot; ``n_full = num_heads`` recovers flat ``FullAttention`` exactly.
  * **Warm-startable**: with ``n_multipole = 0`` the parameters match ``FullAttention``
    (load a flat checkpoint with ``strict=False``); the multipole moment projections only
    exist when ``n_multipole > 0``.
  * **BC-aware masks** (periodic wrap / non-periodic clamp), cached per
    ``(H, W, D, periodic, device)``.
"""

import math
from typing import Callable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import DropPath
from torch import Tensor
from torch.nn import init

from the_well.data.datasets import BoundaryCondition

from ..shared_utils.lr_rope_temporary import RotaryEmbedding, apply_rotary_emb
from ..shared_utils.normalization import RMSGroupNorm
from ..shared_utils.position_biases import RelativePositionBias


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class HeterogeneousAttention(nn.Module):
    def __init__(
        self,
        hidden_dim=768,
        mlp_dim=None,
        num_heads=12,
        n_local=None,
        n_strided=0,
        n_multipole=0,
        n_full=None,
        local_radius=1,
        strided_radius=1,
        strided_strides=(2, 4),
        multipole_pool=4,
        multipole_eval_pool=None,
        multipole_order=2,
        drop_path=0,
        layer_scale_init_value=1e-6,
        bias_type="rel",
        max_d=3,
        weight_tied_axes=True,
        gradient_checkpointing=False,
        norm_layer: Callable = RMSGroupNorm,
    ):
        super().__init__()
        self.mlp_dim = mlp_dim or hidden_dim * 4
        if self.mlp_dim % 2 != 0:
            raise ValueError(
                f"mlp_dim must be divisible by 2, got {self.mlp_dim} instead."
            )
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.max_d = max_d
        self.weight_tied_axes = weight_tied_axes

        # ---- head allocation -------------------------------------------------
        n_local = n_local or 0
        n_strided = n_strided or 0
        n_multipole = n_multipole or 0
        # Default (nothing specified) = all-full = flat FullAttention (warm-start baseline).
        if n_full is None and n_local == 0 and n_strided == 0 and n_multipole == 0:
            n_full = num_heads
        if n_full is None:
            n_full = num_heads - n_local - n_strided - n_multipole
        strided_strides = tuple(int(s) for s in (strided_strides or ()))
        if n_strided == 0:
            strided_strides = ()
        if n_strided != len(strided_strides):
            raise ValueError(
                f"n_strided={n_strided} must equal len(strided_strides)="
                f"{len(strided_strides)} (one stride per strided head)."
            )
        if n_local + n_strided + n_multipole + n_full != num_heads:
            raise ValueError(
                f"head allocation must sum to num_heads={num_heads}: n_local={n_local} "
                f"+ n_strided={n_strided} + n_multipole={n_multipole} + n_full={n_full}."
            )
        if min(n_local, n_strided, n_multipole, n_full) < 0:
            raise ValueError("head counts must be non-negative.")
        self.n_local, self.n_strided = n_local, n_strided
        self.n_multipole, self.n_full = n_multipole, n_full
        self.local_radius = int(local_radius)
        self.strided_radius = int(strided_radius)
        self.strided_strides = strided_strides
        # Variable pooling (mirrors the two-grid coarse_ratios): ``multipole_pool`` may
        # be an int (fixed) or a list of ints (sampled per-forward in training); at eval
        # the fixed ``multipole_eval_pool`` is used (defaults to the coarsest factor).
        if isinstance(multipole_pool, int):
            self.multipole_pools = [int(multipole_pool)]
        else:
            self.multipole_pools = [int(p) for p in multipole_pool]
        if min(self.multipole_pools) < 2:
            raise ValueError("multipole_pool factors must be >= 2.")
        self.multipole_eval_pool = (
            int(multipole_eval_pool)
            if multipole_eval_pool is not None
            else max(self.multipole_pools)
        )
        self.multipole_order = int(multipole_order)

        # Masked-dense heads = local | strided | full (the first M heads). multipole = last.
        # _head_specs MUST match the QKV channel split order of these M heads.
        self._head_specs: List[Tuple[str, int, int]] = (
            [("local", self.local_radius, 1)] * n_local
            + [("strided", self.strided_radius, s) for s in strided_strides]
            + [("full", 0, 0)] * n_full
        )
        self.n_masked = len(self._head_specs)  # == num_heads - n_multipole
        self._mask_cache = {}

        # Multipole moment projections (only when used). nd fixed at max_d for stable
        # param shapes across 2D/3D; singleton axes contribute zero moments.
        if n_multipole > 0:
            nd = max_d
            self._n_moments = 1
            if self.multipole_order >= 1:
                self._n_moments += nd  # dipole
            if self.multipole_order >= 2:
                self._n_moments += nd * (nd + 1) // 2  # quadrupole (symmetric)
            self.mp_key_proj = nn.Linear(self._n_moments * self.head_dim, self.head_dim, bias=False)
            self.mp_val_proj = nn.Linear(self._n_moments * self.head_dim, self.head_dim, bias=False)

        # ---- projections (identical to FullAttention for warm-start) ----------
        self.norm1 = norm_layer(num_heads, hidden_dim, affine=True)
        self.fused_dims = (self.mlp_dim, hidden_dim, hidden_dim, hidden_dim)  # FF, Q, K, V
        self.fused_ff_qkv = nn.Linear(hidden_dim, sum(self.fused_dims))

        self.activation = SwiGLU()
        self.ff_out = nn.Linear(int(self.mlp_dim // 2), hidden_dim)
        init.kaiming_uniform_(
            self.ff_out.weight, a=math.sqrt(5) / layer_scale_init_value
        )
        if self.ff_out.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.ff_out.weight)
            bound = 1 / math.sqrt(fan_in) * layer_scale_init_value
            init.uniform_(self.ff_out.bias, -bound, bound)

        self.attn_out = nn.Linear(hidden_dim, hidden_dim, bias=False)
        init.kaiming_uniform_(
            self.attn_out.weight, a=math.sqrt(5) / layer_scale_init_value
        )
        if self.attn_out.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.attn_out.weight)
            bound = 1 / math.sqrt(fan_in) * layer_scale_init_value
            init.uniform_(self.attn_out.bias, -bound, bound)

        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)

        self.rotary_emb = RotaryEmbedding(
            self.head_dim // 4, freqs_for="pixel", max_freq=256
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    # ------------------------------------------------------------------ rope
    def make_rope_learnable(self, per_axis=False):
        if hasattr(self, "rotary_emb"):
            self.rotary_emb.make_learnable(per_axis)

    def get_rotary_embedding(self, n, device):
        return self.rotary_emb(n, device=device)

    # ------------------------------------------------------------------ masks
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

    def _build_mask(self, H, W, D, periodic, device) -> Tensor:
        """Boolean ``[n_masked, N, N]`` attend-mask (True = key participates), token
        order ``(h w d)`` matching the SDPA rearrange below."""
        sizes = (H, W, D)
        N = H * W * D
        deltas = []
        for a, S in enumerate(sizes):
            idx = torch.arange(S, device=device)
            d = idx[:, None] - idx[None, :]
            if periodic[a] and S > 1:
                d = torch.remainder(d + S // 2, S) - S // 2
            deltas.append(d)

        masks = []
        for typ, radius, stride in self._head_specs:
            if typ == "full":
                masks.append(torch.ones(N, N, dtype=torch.bool, device=device))
                continue
            per_axis = []
            for a in range(3):
                d = deltas[a]
                if typ == "local":
                    aa = d.abs() <= radius
                else:  # strided dilated neighborhood
                    aa = (torch.remainder(d, stride) == 0) & (d.abs() <= radius * stride)
                per_axis.append(aa)
            ah, aw, ad = per_axis
            allowed = (
                ah[:, None, None, :, None, None]
                & aw[None, :, None, None, :, None]
                & ad[None, None, :, None, None, :]
            )
            masks.append(allowed.reshape(N, N))
        return torch.stack(masks, dim=0)

    def _get_mask(self, H, W, D, bcs, device):
        # Masked group all-full -> no mask -> identical to FullAttention on those heads.
        if self.n_local == 0 and self.n_strided == 0:
            return None
        periodic = tuple(self._periodic_flags(bcs, 3))
        key = (H, W, D, periodic, device)
        mask = self._mask_cache.get(key)
        if mask is None:
            mask = self._build_mask(H, W, D, periodic, device)
            self._mask_cache[key] = mask
        return mask

    # ------------------------------------------------------------------ multipole
    def _pick_pool(self) -> int:
        """Per-forward pool factor: sampled from ``multipole_pools`` in training (if more
        than one), fixed to ``multipole_eval_pool`` at eval."""
        if self.training and len(self.multipole_pools) > 1:
            idx = int(torch.randint(len(self.multipole_pools), (1,)).item())
            return self.multipole_pools[idx]
        return self.multipole_eval_pool

    def _coarse_moments(self, t: Tensor, proj: nn.Linear, pool: int):
        """Summarize each ``pool^d`` neighborhood of ``t`` (B, h, H, W, D, c) by its
        multipole moments and project back to head_dim. Returns coarse tokens
        (B, h, Hc, Wc, Dc, c) and the coarse sizes (Hc, Wc, Dc)."""
        B, h, H, W, Dd, c = t.shape
        sizes = (H, W, Dd)
        pools = [min(pool, s) for s in sizes]

        def axis_r(S, p):
            ar = torch.arange(S, device=t.device, dtype=t.dtype)
            if p <= 1:
                return torch.zeros(S, device=t.device, dtype=t.dtype)
            within = ar % p
            return (within - (p - 1) / 2.0) / ((p - 1) / 2.0)  # ~[-1, 1]

        rh, rw, rd = (axis_r(S, p) for S, p in zip(sizes, pools))

        feats = [t]  # monopole
        if self.multipole_order >= 1:
            feats += [
                t * rh.view(1, 1, H, 1, 1, 1),
                t * rw.view(1, 1, 1, W, 1, 1),
                t * rd.view(1, 1, 1, 1, Dd, 1),
            ]
        if self.multipole_order >= 2:
            feats += [
                t * (rh * rh).view(1, 1, H, 1, 1, 1),
                t * (rw * rw).view(1, 1, 1, W, 1, 1),
                t * (rd * rd).view(1, 1, 1, 1, Dd, 1),
                t * (rh.view(H, 1) * rw.view(1, W)).view(1, 1, H, W, 1, 1),
                t * (rh.view(H, 1) * rd.view(1, Dd)).view(1, 1, H, 1, Dd, 1),
                t * (rw.view(W, 1) * rd.view(1, Dd)).view(1, 1, 1, W, Dd, 1),
            ]

        pooled = []
        for f in feats:
            g = rearrange(f, "b h hh ww dd c -> (b h) c hh ww dd")
            g = F.avg_pool3d(g, kernel_size=tuple(pools), ceil_mode=True)
            pooled.append(rearrange(g, "(b h) c hc wc dc -> b h hc wc dc c", b=B))

        Hc, Wc, Dc = pooled[0].shape[2:5]
        stk = torch.stack(pooled, dim=-2)  # b h hc wc dc nm c
        stk = rearrange(stk, "b h hc wc dc nm c -> b h hc wc dc (nm c)")
        return proj(stk), (Hc, Wc, Dc)

    # ------------------------------------------------------------------ forward
    def forward(self, x, bcs, return_att=False):
        # input is (t b) x c x h x w x d
        B, C, H, W, D = x.shape

        input = x.clone()
        x = self.norm1(x)

        fused_ff_qkv = rearrange(x, "b c h w d -> b h w d c")
        ff, q, k, v = self.fused_ff_qkv(fused_ff_qkv).split(self.fused_dims, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b h w d (he c) -> b he h w d c", he=self.num_heads),
            (q, k, v),
        )
        q = self.q_norm(q)
        k = self.k_norm(k)

        pos_fine = self.rotary_emb.get_axial_freqs(H, W, D)
        M = self.n_masked
        att_parts = []

        # --- masked-dense group (local | strided | full) -------------------
        if M > 0:
            qm, km, vm = q[:, :M], k[:, :M], v[:, :M]
            qm, km = apply_rotary_emb(pos_fine, qm), apply_rotary_emb(pos_fine, km)
            qm, km, vm = map(
                lambda t: rearrange(t, "b he h w d c -> b he (h w d) c"), (qm, km, vm)
            )
            mask = self._get_mask(H, W, D, bcs, q.device)
            att_parts.append(F.scaled_dot_product_attention(qm, km, vm, attn_mask=mask))

        # --- multipole group (pooled-KV cross attention) -------------------
        if self.n_multipole > 0:
            qp, kp, vp = q[:, M:], k[:, M:], v[:, M:]
            pool = self._pick_pool()  # same pool for keys and values (shared coarse grid)
            ck, (Hc, Wc, Dc) = self._coarse_moments(kp, self.mp_key_proj, pool)
            cv, _ = self._coarse_moments(vp, self.mp_val_proj, pool)
            pos_coarse = self.rotary_emb.get_axial_freqs(Hc, Wc, Dc)
            qp = apply_rotary_emb(pos_fine, qp)
            ck = apply_rotary_emb(pos_coarse, ck)
            qp = rearrange(qp, "b he h w d c -> b he (h w d) c")
            ck = rearrange(ck, "b he hc wc dc c -> b he (hc wc dc) c")
            cv = rearrange(cv, "b he hc wc dc c -> b he (hc wc dc) c")
            att_parts.append(F.scaled_dot_product_attention(qp, ck, cv))

        att = torch.cat(att_parts, dim=1) if len(att_parts) > 1 else att_parts[0]
        att = rearrange(att, "b he (h w d) c -> b h w d (he c)", h=H, w=W)
        att_out = self.attn_out(att)
        x = self.drop_path(att_out + self.ff_out(self.activation(ff)))
        x = rearrange(x, "b h w d c -> b c h w d") + input

        return x, []
