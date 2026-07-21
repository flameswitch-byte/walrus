"""GPU-optimized FMM / H-matrix attention (fused single-SDPA joint softmax).

Same MATH as ``FMMAttention``, but the joint softmax over [near ∪ far-rings ∪ mop-up] is
executed as ONE ``torch.nn.functional.scaled_dot_product_attention`` (SDPA) call instead of
the manual einsum → softmax → einsum in the base class. SDPA dispatches by device:

    * CUDA  → the fused FlashAttention / memory-efficient kernel (scores + softmax + value
              aggregate fused in-kernel; the N×K score matrix is never written to HBM).
    * CPU   → the math backend, numerically identical to the base class's manual joint
              softmax (up to fp tolerance).

So we verify parity on CPU (math backend) here; the SAME op runs the fused flash kernel on
CUDA later — identical logic, faster. No flags, no separate code path per device.

DESIGN NOTE — why this does NOT also use the coarse-score (no-upsample) far optimization:
  A single fused SDPA needs every key laid out *per query* (q:(…,1,c), kv:(…,Ktot,c)), so the
  far rings must be materialized to fine resolution (via the base class's upsample). The
  coarse-score trick keeps far keys shared-per-parent, which is mutually exclusive with one
  fused softmax. This file therefore trades that MEMORY optimization for KERNEL FUSION (the
  GPU speed win). The coarse-score / streaming path remains the memory-optimal alternative
  for memory-bound 3D (see scratch_fmm_coarse_score.py / scratch_fmm_stream_near.py).

Because it subclasses ``FMMAttention``, __init__, all pooling/neighbor helpers, and the
parameter set are IDENTICAL — a base checkpoint loads with zero missing/extra keys, and the
two forwards can be compared weight-for-weight.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from ..shared_utils.lr_rope_temporary import apply_rotary_emb
from .fmm_attention import FMMAttention


class FMMAttentionFast(FMMAttention):
    """FMM attention with the joint softmax fused into one SDPA (flash on CUDA)."""

    def _sdpa_joint(self, q: Tensor, blocks: List[tuple]) -> Tensor:
        """One fused SDPA over the concatenated key/value blocks. q (B,he,X,Y,Z,c).
        Block is ('windowed', k, v, mask) with k/v (B,he,X,Y,Z,K,c), mask (X,Y,Z,K), or
        ('shared', k, v) with k/v (B,he,N,c) (broadcast to every query)."""
        B, he, X, Y, Z, c = q.shape
        ks, vs, masks = [], [], []
        for blk in blocks:
            if blk[0] == "shared":
                k, v = blk[1], blk[2]                                # (B,he,Ng,c)
                Ng = k.shape[2]
                k = k[:, :, None, None, None].expand(B, he, X, Y, Z, Ng, c)
                v = v[:, :, None, None, None].expand(B, he, X, Y, Z, Ng, c)
                m = torch.ones(X, Y, Z, Ng, dtype=torch.bool, device=q.device)
            else:
                k, v, m = blk[1], blk[2], blk[3]                     # (B,he,X,Y,Z,K,c)
            ks.append(k)
            vs.append(v)
            masks.append(m)
        k = torch.cat(ks, dim=5)                                     # (B,he,X,Y,Z,Ktot,c)
        v = torch.cat(vs, dim=5)
        mask = torch.cat(masks, dim=3)                              # (X,Y,Z,Ktot)
        Ktot = k.shape[5]
        XYZ = X * Y * Z

        # SDPA batch layout: (B·he·XYZ, seq, c). Query length 1 (each token is one query).
        qf = q.reshape(B * he * XYZ, 1, c)
        kf = k.reshape(B * he * XYZ, Ktot, c)
        vf = v.reshape(B * he * XYZ, Ktot, c)
        # mask depends only on (spatial, key); broadcast over B·he, add query-seq dim.
        mf = (
            mask.reshape(XYZ, Ktot)
            .unsqueeze(0)
            .expand(B * he, XYZ, Ktot)
            .reshape(B * he * XYZ, 1, Ktot)
        )
        # SDPA default scale = 1/sqrt(c) = head_dim**-0.5 (matches the base class).
        out = F.scaled_dot_product_attention(qf, kf, vf, attn_mask=mf)  # (…,1,c)
        return out.reshape(B, he, X, Y, Z, c)

    # ------------------------------------------------------------------ forward
    def forward(self, x, bcs=None, return_att: bool = False):
        # Identical to FMMAttention.forward EXCEPT the final aggregation uses _sdpa_joint.
        B, C, H, W, D = x.shape
        periodic = self._periodic_flags(bcs, 3)
        inp = x.clone()

        feat = rearrange(self.norm1(x), "b c h w d -> b h w d c")
        ff, qf, kf, vf = self.fused_ff_qkv(feat).split(self.fused_dims, dim=-1)
        qf, kf, vf = map(self._to_heads, (qf, kf, vf))
        qf, kf = self.q_norm(qf), self.k_norm(kf)
        kf_pre = kf  # pre-RoPE keys for the far pool (finding 1)
        pos_f = self.rotary_emb.get_axial_freqs(H, W, D)
        qf, kf = apply_rotary_emb(pos_f, qf), apply_rotary_emb(pos_f, kf)

        # NEAR block
        kf_n, m_near = self._neighbors(kf, self.near_radius, periodic)
        vf_n, _ = self._neighbors(vf, self.near_radius, periodic)
        blocks = [("windowed", kf_n, vf_n, m_near)]

        # FAR rings (materialized to fine so they can feed one fused SDPA — see design note)
        ck, cv = kf_pre, vf
        for level_idx, pool in enumerate(self._levels((H, W, D))):
            ck, cv = self._pool_once(ck, cv, level_idx)
            Hc, Wc, Dc = ck.shape[2:5]
            pos_c = self.rotary_emb.get_axial_freqs(Hc, Wc, Dc)
            ck_r = apply_rotary_emb(pos_c, self.coarse_k_norm(ck))
            kc_n, m_far = self._neighbors(ck_r, self.far_outer, periodic, inner=self.far_inner)
            vc_n, _ = self._neighbors(cv, self.far_outer, periodic, inner=self.far_inner)
            kc_up = self._upsample_to_fine(kc_n, pool, (H, W, D), (2, 3, 4))
            vc_up = self._upsample_to_fine(vc_n, pool, (H, W, D), (2, 3, 4))
            m_up = self._upsample_to_fine(m_far, pool, (H, W, D), (0, 1, 2))
            blocks.append(("windowed", kc_up, vc_up, m_up))

        # Optional global mop-up (shared block)
        if self.global_mop_up and self.max_levels != 0:
            gk, gv = self._pool_once(ck, cv, self._mop_idx)
            Hg, Wg, Dg = gk.shape[2:5]
            pos_g = self.rotary_emb.get_axial_freqs(Hg, Wg, Dg)
            gk = apply_rotary_emb(pos_g, self.coarse_k_norm(gk))
            blocks.append(("shared", self._flat(gk), self._flat(gv)))

        att = self._sdpa_joint(qf, blocks)                          # ← fused SDPA (flash on CUDA)
        att = rearrange(att, "b he h w d c -> b h w d (he c)")
        att_out = self.attn_out(att)
        x = self.drop_path(att_out + self.ff_out(self.activation(ff)))
        x = rearrange(x, "b h w d c -> b c h w d") + inp
        return x, []
