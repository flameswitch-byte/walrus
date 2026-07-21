"""Path C — memory-optimal FMM / H-matrix attention, upgraded toward Kang et al. (2023).

Combines the two memory/gather optimizations that the single-SDPA `fmm_attention_fast.py`
had to give up, PLUS the two design upgrades the FMA/FMMformer papers endorse:

  * NEAR  — no K-fold gather.
      leaf_near=True  (default, Kang I_0): block-tridiagonal near. Tokens are grouped into
                      leaf blocks of size ``pool_base``; a query attends its own block + the
                      adjacent blocks (block-distance <= 1). This tiles EXACTLY with the far
                      annulus {2,3} (no query-centered/parent-centered seam), and materializes
                      only O(N * 3^d * c) (much less than the query-centered stack O(N*K*c)).
      leaf_near=False : query-centered window, computed by a streaming online rollup on CPU
                      (no stack) or a fused NATTEN kernel on CUDA. Kept as the verified parity
                      anchor vs FMMAttention.
  * FAR   — coarse-score (no upsample). Far logits scored at COARSE resolution (group fine
      queries under their parent); far values applied at coarse resolution. Memory ~
      N/pool^d * K * c per ring instead of the fine upsample O(N*K*c).
  * RANK  — pool_rank>1 (Kang p=4, FMMformer multi-kernel): each coarse cell is summarized by
      ``pool_rank`` learned vectors (multiple multipole moments), so each far cell contributes
      ``pool_rank`` keys. pool_rank=1 recovers the single-summary far.
  * JOIN  — ONE manual softmax over concatenated [near ∪ far-rings ∪ mop-up] LOGITS (logits are
      N*K, cheap; no flash LSE-merge needed).

Peak memory O(N*c) + O(N*K) logits instead of O(N*K*c) — what makes 3D / high-res feasible.

Device/testability: leaf_near + coarse far + manual join runs and is exact on CPU (unit-tested;
the anchor config leaf_near=False, pool_rank=1 is numerically identical to FMMAttention). The
CUDA NATTEN near path (leaf_near=False) is gated behind is_cuda+natten+periodic and documented
as verify-on-GPU. Assumes token grids divisible by pool (powers of two — Walrus uses 16/32/64).
"""

from __future__ import annotations

import itertools
from typing import List, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from ..shared_utils.lr_rope_temporary import apply_rotary_emb
from .fmm_attention import FMMAttention

try:  # fused neighborhood-attention kernels (CUDA); optional
    from natten.functional import na2d_av, na2d_qk, na3d_av, na3d_qk
    _HAS_NATTEN = True
except Exception:  # pragma: no cover
    _HAS_NATTEN = False


class FMMAttentionFused(FMMAttention):
    """No-gather near (leaf/streaming/NATTEN) + coarse-score rank-p far + manual joint softmax."""

    def __init__(self, *args, leaf_near: bool = True, pool_rank: int = 1, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.leaf_near = bool(leaf_near)
        self.pool_rank = int(pool_rank)
        if self.pool_rank < 1:
            raise ValueError("pool_rank must be >= 1")
        if self.pool_rank > 1 and self.tie_levels:
            raise ValueError("tie_levels is incompatible with pool_rank>1 (per-level in-dims differ)")
        # Rebuild pooling projections for rank-p. Level 0 pools the fine grid (p_in=1); deeper
        # levels pool a p-vector grid (p_in=pool_rank). Constant children R = pool_base**d.
        if self.learned_pool and self.pool_rank > 1:
            r = self.pool_base ** self.spatial_dims
            c, p = self.head_dim, self.pool_rank
            n = len(self.pool_k)

            def lin(i):
                in_dim = (r if i == 0 else r * p) * c
                return nn.Linear(in_dim, p * c, bias=False)
            self.pool_k = nn.ModuleList([lin(i) for i in range(n)])
            self.pool_v = nn.ModuleList([lin(i) for i in range(n)])

    # ------------------------------------------------------------------ rank-p pooling
    def _pool_once(self, x_k, x_v, level_idx):
        """Pool by pool_base. Returns coarse (B,he,Hc,Wc,Dc,p,c). Input may be (…,c) (level 0,
        p_in=1) or (…,p,c) (deeper). Overrides the base rank-1 pooling."""
        p = self.pool_rank
        has_p = x_k.dim() == 7

        def do(x, proj):
            if has_p:
                x = rearrange(x, "b he h w d p c -> b he h w d (p c)")
            ch = self._gather_children(x, self.pool_base)          # (…,Hc,Wc,Dc,R,feat)
            if self.learned_pool:
                ch = rearrange(ch, "b he hc wc dc r f -> b he hc wc dc (r f)")
                out = proj(ch)                                     # (…, p*c)
                return rearrange(out, "b he hc wc dc (p c) -> b he hc wc dc p c", p=p)
            # mean pool → single summary, broadcast to p (rank-p only meaningful when learned)
            m = ch.mean(dim=5)                                     # (…,Hc,Wc,Dc,feat)
            if has_p:
                m = rearrange(m, "b he hc wc dc (p c) -> b he hc wc dc p c", p=p).mean(5, keepdim=True)
            else:
                m = m.unsqueeze(5)
            return m.expand(*m.shape[:5], p, m.shape[-1])
        idx = 0 if self.tie_levels else min(level_idx, len(self.pool_k) - 1)
        proj_k = self.pool_k[idx] if self.learned_pool else None
        proj_v = self.pool_v[idx] if self.learned_pool else None
        return do(x_k, proj_k), do(x_v, proj_v)

    # ------------------------------------------------------------------ leaf-aligned near
    def _leaf_gather(self, x, m, sizes):
        """Group into leaf blocks of size m (per non-singleton axis) → (B,he,Hb,Wb,Db,R,c)."""
        mh, mw, md = (m if s > 1 else 1 for s in sizes)
        return rearrange(x, "b he (hb mh) (wb mw) (db md) c -> b he hb wb db (mh mw md) c",
                         mh=mh, mw=mw, md=md), (mh, mw, md)

    def _leaf_near(self, qf, kf, vf, periodic, sizes):
        """Block-tridiagonal near (Kang I_0). Returns near_logits (B,he,H,W,D,K), mask
        (H,W,D,K), and (vf_near, R, mblk) for the value apply. K = 3^d_ns * m^d_ns."""
        m = self.pool_base
        scale = self.head_dim ** -0.5
        kf_blk, mblk = self._leaf_gather(kf, m, sizes)             # (B,he,Hb,Wb,Db,R,c)
        vf_blk, _ = self._leaf_gather(vf, m, sizes)
        qf_blk, _ = self._leaf_gather(qf, m, sizes)
        R = kf_blk.shape[5]
        # ±1 block-neighborhood: fold (R,c) into one feature, gather on the block grid, unfold
        kbf = rearrange(kf_blk, "b he a b2 c2 r c -> b he a b2 c2 (r c)")
        vbf = rearrange(vf_blk, "b he a b2 c2 r c -> b he a b2 c2 (r c)")
        kf_nb, m_nb = self._neighbors(kbf, 1, periodic)            # (B,he,Hb,Wb,Db,Knb,R*c),(Hb,Wb,Db,Knb)
        vf_nb, _ = self._neighbors(vbf, 1, periodic)
        kf_near = rearrange(kf_nb, "b he a b2 c2 knb (r c) -> b he a b2 c2 (knb r) c", r=R)
        vf_near = rearrange(vf_nb, "b he a b2 c2 knb (r c) -> b he a b2 c2 (knb r) c", r=R)
        # score: each of R query-children vs all K=Knb*R neighbor keys
        s = torch.einsum("bhABCrc,bhABCkc->bhABCrk", qf_blk, kf_near) * scale
        mh, mw, md = mblk
        near_logits = rearrange(s, "b he hb wb db (mh mw md) k -> b he (hb mh) (wb mw) (db md) k",
                                mh=mh, mw=mw, md=md)
        # mask: block-neighbor validity, each valid neighbor block → R valid keys; upsample to fine
        mask_blk = m_nb.unsqueeze(-1).expand(*m_nb.shape, R).reshape(*m_nb.shape[:3], -1)  # (Hb,Wb,Db,K)
        mask = self._upsample_to_fine(mask_blk, m, sizes, (0, 1, 2))
        return near_logits, mask, (vf_near, R, mblk)

    def _leaf_near_apply(self, w, ctx, sizes):
        vf_near, R, mblk = ctx
        mh, mw, md = mblk
        wg = rearrange(w, "b he (hb mh) (wb mw) (db md) k -> b he hb wb db (mh mw md) k",
                       mh=mh, mw=mw, md=md)
        og = torch.einsum("bhABCrk,bhABCkc->bhABCrc", wg, vf_near)
        return rearrange(og, "b he hb wb db (mh mw md) c -> b he (hb mh) (wb mw) (db md) c",
                         mh=mh, mw=mw, md=md)

    # ------------------------------------------------------------------ streaming near (anchor)
    def _stream_near_logits(self, qf, kf, periodic, sizes) -> Tuple[Tensor, Tensor]:
        scale = self.head_dim ** -0.5
        rads = [self.near_radius if s > 1 else 0 for s in sizes]
        dev = qf.device
        logits, masks = [], []
        for off in itertools.product(*[range(-r, r + 1) for r in rads]):
            rk = kf
            valid = torch.ones(sizes, device=dev)
            for a, o in enumerate(off):
                if o != 0:
                    rk = torch.roll(rk, shifts=-o, dims=2 + a)
                    if not periodic[a]:
                        pos = torch.arange(sizes[a], device=dev) + o
                        ok = (pos >= 0) & (pos < sizes[a])
                        shape = [1, 1, 1]; shape[a] = sizes[a]
                        valid = valid * ok.view(shape).float()
            logits.append((qf * rk).sum(-1) * scale)
            masks.append(valid)
        return torch.stack(logits, dim=-1), torch.stack(masks, dim=-1).bool()

    def _stream_near_apply(self, w, vf, periodic, sizes) -> Tensor:
        rads = [self.near_radius if s > 1 else 0 for s in sizes]
        acc = torch.zeros_like(vf)
        for k, off in enumerate(itertools.product(*[range(-r, r + 1) for r in rads])):
            rv = vf
            for a, o in enumerate(off):
                if o != 0:
                    rv = torch.roll(rv, shifts=-o, dims=2 + a)
            acc = acc + w[..., k].unsqueeze(-1) * rv
        return acc

    @staticmethod
    def _circular_pad(t, r, spatial_axes) -> Tensor:
        for ax in spatial_axes:
            if t.shape[ax] > 1 and r > 0:
                t = torch.cat([t.narrow(ax, t.shape[ax] - r, r), t, t.narrow(ax, 0, r)], dim=ax)
        return t

    def _natten_near(self, qf, kf, vf, sizes, w=None):
        """Fused NATTEN neighborhood (CUDA, periodic via circular pad). VERIFY ON GPU."""
        ks, r = 2 * self.near_radius + 1, self.near_radius
        is3d = sizes[2] > 1
        qk = na3d_qk if is3d else na2d_qk
        av = na3d_av if is3d else na2d_av
        sq = (lambda t: t if is3d else t.squeeze(4))
        unsq = (lambda t: t if is3d else t.unsqueeze(4))
        sax = (2, 3, 4) if is3d else (2, 3)
        if w is None:
            qp, kp = self._circular_pad(sq(qf), r, sax), self._circular_pad(sq(kf), r, sax)
            lp = qk(qp, kp, kernel_size=ks, dilation=1)
            for ax in sax:
                lp = lp.narrow(ax, r, sizes[ax - 2])
            return unsq(lp) * (self.head_dim ** -0.5)
        wp, vp = self._circular_pad(sq(w), r, sax), self._circular_pad(sq(vf), r, sax)
        op = av(wp, vp, kernel_size=ks, dilation=1)
        for ax in sax:
            op = op.narrow(ax, r, sizes[ax - 2])
        return unsq(op)

    # ------------------------------------------------------------------ rank-p coarse far
    def _far_blocks(self, qf, kf_pre, vf, periodic, sizes):
        H, W, D = sizes
        scale = self.head_dim ** -0.5
        out = []
        ck, cv = kf_pre, vf
        for level_idx, pool in enumerate(self._levels(sizes)):
            ck, cv = self._pool_once(ck, cv, level_idx)            # (B,he,Hc,Wc,Dc,p,c)
            Hc, Wc, Dc, p = ck.shape[2], ck.shape[3], ck.shape[4], ck.shape[5]
            pos_c = self.rotary_emb.get_axial_freqs(Hc, Wc, Dc)
            # RoPE per summary vector (merge p into the head axis so freqs broadcast)
            ck_r = rearrange(self.coarse_k_norm(ck), "b he hc wc dc p c -> b (he p) hc wc dc c")
            ck_r = apply_rotary_emb(pos_c, ck_r)
            ck_r = rearrange(ck_r, "b (he p) hc wc dc c -> b he hc wc dc (p c)", p=p)
            cvf = rearrange(cv, "b he hc wc dc p c -> b he hc wc dc (p c)")
            kc_n, m_far = self._neighbors(ck_r, self.far_outer, periodic, inner=self.far_inner)
            vc_n, _ = self._neighbors(cvf, self.far_outer, periodic, inner=self.far_inner)
            kc_n = rearrange(kc_n, "b he hc wc dc kann (p c) -> b he hc wc dc (kann p) c", p=p)
            vc_n = rearrange(vc_n, "b he hc wc dc kann (p c) -> b he hc wc dc (kann p) c", p=p)
            mask = m_far.unsqueeze(-1).expand(*m_far.shape, p).reshape(*m_far.shape[:3], -1)
            ph, pw, pd = H // Hc, W // Wc, D // Dc
            qg = rearrange(qf, "b he (hc ph) (wc pw) (dc pd) c -> b he hc wc dc (ph pw pd) c",
                           ph=ph, pw=pw, pd=pd)
            s = torch.einsum("bhABCrc,bhABCkc->bhABCrk", qg, kc_n) * scale
            far_logits = rearrange(s, "b he hc wc dc (ph pw pd) k -> b he (hc ph) (wc pw) (dc pd) k",
                                   ph=ph, pw=pw, pd=pd)
            mask_fine = self._upsample_to_fine(mask, pool, sizes, (0, 1, 2))
            out.append((far_logits, mask_fine, vc_n, (ph, pw, pd)))
        return out

    @staticmethod
    def _far_apply(w, vc_n, phpwpd):
        ph, pw, pd = phpwpd
        wg = rearrange(w, "b he (hc ph) (wc pw) (dc pd) k -> b he hc wc dc (ph pw pd) k",
                       ph=ph, pw=pw, pd=pd)
        og = torch.einsum("bhABCrk,bhABCkc->bhABCrc", wg, vc_n)
        return rearrange(og, "b he hc wc dc (ph pw pd) c -> b he (hc ph) (wc pw) (dc pd) c",
                         ph=ph, pw=pw, pd=pd)

    # ------------------------------------------------------------------ forward
    def forward(self, x, bcs=None, return_att: bool = False):
        B, C, H, W, D = x.shape
        sizes = (H, W, D)
        periodic = self._periodic_flags(bcs, 3)
        inp = x.clone()

        feat = rearrange(self.norm1(x), "b c h w d -> b h w d c")
        ff, qf, kf, vf = self.fused_ff_qkv(feat).split(self.fused_dims, dim=-1)
        qf, kf, vf = map(self._to_heads, (qf, kf, vf))
        qf, kf = self.q_norm(qf), self.k_norm(kf)
        kf_pre = kf
        pos_f = self.rotary_emb.get_axial_freqs(H, W, D)
        qf, kf = apply_rotary_emb(pos_f, qf), apply_rotary_emb(pos_f, kf)

        # ---- NEAR ----
        near_ctx = None
        if self.leaf_near:
            near_logits, near_mask, near_ctx = self._leaf_near(qf, kf, vf, periodic, sizes)
        else:
            use_natten = qf.is_cuda and _HAS_NATTEN and all(periodic[a] or sizes[a] == 1 for a in range(3))
            if use_natten:
                near_logits = self._natten_near(qf, kf, vf, sizes)
                near_mask = torch.ones(near_logits.shape[2:], dtype=torch.bool, device=x.device)
            else:
                near_logits, near_mask = self._stream_near_logits(qf, kf, periodic, sizes)

        # ---- FAR (coarse-score, rank-p) ----
        far = self._far_blocks(qf, kf_pre, vf, periodic, sizes)

        # ---- MOP-UP (shared coarsest) ----
        mop = None
        if self.global_mop_up and self.max_levels != 0:
            ck, cv = kf_pre, vf
            n_far = len(self._levels(sizes))
            for i in range(n_far):
                ck, cv = self._pool_once(ck, cv, i)
            # if no far level ran, ck is still the fine (p=1) grid → use the level-0 projection
            mop_level = self._mop_idx if n_far > 0 else 0
            gk, gv = self._pool_once(ck, cv, mop_level)            # (B,he,Hg,Wg,Dg,p,c)
            Hg, Wg, Dg, p = gk.shape[2], gk.shape[3], gk.shape[4], gk.shape[5]
            pos_g = self.rotary_emb.get_axial_freqs(Hg, Wg, Dg)
            gk = rearrange(self.coarse_k_norm(gk), "b he hg wg dg p c -> b (he p) hg wg dg c")
            gk = rearrange(apply_rotary_emb(pos_g, gk), "b (he p) hg wg dg c -> b he (hg wg dg p) c", p=p)
            gv = rearrange(gv, "b he hg wg dg p c -> b he (hg wg dg p) c")
            mop_logits = torch.einsum("bhxyzc,bhnc->bhxyzn", qf, gk) * (self.head_dim ** -0.5)
            mop = (mop_logits, gv)

        # ---- JOINT SOFTMAX ----
        NEG = torch.finfo(near_logits.dtype).min
        parts_logits = [near_logits.masked_fill(~near_mask.unsqueeze(0).unsqueeze(0), NEG)]
        for far_logits, mask, _, _ in far:
            parts_logits.append(far_logits.masked_fill(~mask.unsqueeze(0).unsqueeze(0), NEG))
        if mop is not None:
            parts_logits.append(mop[0])
        w = torch.softmax(torch.cat(parts_logits, dim=-1), dim=-1)
        parts = w.split([t.shape[-1] for t in parts_logits], dim=-1)

        # ---- APPLY ----
        wn = parts[0]
        if self.leaf_near:
            out = self._leaf_near_apply(wn, near_ctx, sizes)
        else:
            use_natten = qf.is_cuda and _HAS_NATTEN and all(periodic[a] or sizes[a] == 1 for a in range(3))
            out = self._natten_near(qf, kf, vf, sizes, w=wn) if use_natten \
                else self._stream_near_apply(wn, vf, periodic, sizes)
        for j, (_, _, vc_n, phpwpd) in enumerate(far):
            out = out + self._far_apply(parts[1 + j], vc_n, phpwpd)
        if mop is not None:
            out = out + torch.einsum("bhxyzn,bhnc->bhxyzc", parts[-1], mop[1])

        out = rearrange(out, "b he h w d c -> b h w d (he c)")
        att_out = self.attn_out(out)
        x = self.drop_path(att_out + self.ff_out(self.activation(ff)))
        x = rearrange(x, "b h w d c -> b c h w d") + inp
        return x, []
