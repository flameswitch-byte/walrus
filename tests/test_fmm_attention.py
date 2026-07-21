"""Unit tests for FMMAttention (faithful multi-level FMM / H-matrix attention, v1).

Covers: forward shape/grad (2D + 3D), wall BC, learned vs mean pooling, level auto-sizing,
warm-start state-dict compatibility with FullAttention, and — the key correctness check —
interaction-list TILING: every (query, fine-key) pair within reach is covered, and almost
never double-counted across the near + per-level far blocks.
"""

import itertools
import unittest

import numpy as np
import torch

from the_well.data.datasets import BoundaryCondition
from walrus.models.spatial_blocks.fmm_attention import FMMAttention
from walrus.models.spatial_blocks.full_attention import FullAttention

P = BoundaryCondition["PERIODIC"].value
NONP = next((bc.value for bc in BoundaryCondition if bc.value != P), P)


def _bcs(per):
    return [[[P if p else NONP, P if p else NONP] for p in per]]


def _coverage(H, W, near_r, pool_base, fi, fo, periodic, max_levels=8):
    """Reference (query, fine-key) coverage COUNT over the near + far blocks, mirroring the
    module's logic: query-centered near window; per-level parent-centered annulus whose
    coarse cells are footprinted back to fine. Returns an (N, N) int count."""
    N = H * W
    cnt = np.zeros((N, N), dtype=int)

    def fidx(h, w):
        return h * W + w

    # near: query-centered fine window of radius near_r
    for qh, qw in itertools.product(range(H), range(W)):
        for dh, dw in itertools.product(range(-near_r, near_r + 1), repeat=2):
            kh, kw = qh + dh, qw + dw
            if periodic[0]:
                kh %= H
            elif not (0 <= kh < H):
                continue
            if periodic[1]:
                kw %= W
            elif not (0 <= kw < W):
                continue
            cnt[fidx(qh, qw), fidx(kh, kw)] += 1

    # far levels (same auto-sizing as FMMAttention._levels)
    levels, ell, diam, min_extent = [], 1, max(H, W), 2 * fo + 1
    while True:
        p = pool_base ** ell
        coarse_extent = max((s + p - 1) // p for s in (H, W) if s > 1)
        if coarse_extent < min_extent:
            break
        levels.append(p)
        if (fo + 1) * p - 1 >= diam or len(levels) >= max_levels:
            break
        ell += 1

    for p in levels:
        Hc, Wc = (H + p - 1) // p, (W + p - 1) // p
        for qh, qw in itertools.product(range(H), range(W)):
            ph, pw = qh // p, qw // p
            for dh, dw in itertools.product(range(-fo, fo + 1), repeat=2):
                if max(abs(dh), abs(dw)) < fi:
                    continue
                ch, cw = ph + dh, pw + dw
                if periodic[0]:
                    ch %= Hc
                elif not (0 <= ch < Hc):
                    continue
                if periodic[1]:
                    cw %= Wc
                elif not (0 <= cw < Wc):
                    continue
                for fh in range(ch * p, min((ch + 1) * p, H)):
                    for fw in range(cw * p, min((cw + 1) * p, W)):
                        cnt[fidx(qh, qw), fidx(fh, fw)] += 1
    return cnt


class TestFMMAttention(unittest.TestCase):
    def _make(self, **kw):
        kw.setdefault("hidden_dim", 256)
        kw.setdefault("num_heads", 8)
        return FMMAttention(**kw)

    def test_forward_2d_shape_and_grad(self):
        m = self._make()
        x = torch.randn(2, 256, 16, 16, 1, requires_grad=True)
        y, _ = m(x, _bcs((True, True)))
        assert y.shape == x.shape and torch.isfinite(y).all()
        y.sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()

    def test_forward_3d(self):
        m = self._make()
        x = torch.randn(2, 256, 8, 8, 8)
        y, _ = m(x, _bcs((True, True, True)))
        assert y.shape == x.shape and torch.isfinite(y).all()

    def test_wall_axis(self):
        m = self._make()
        x = torch.randn(2, 256, 16, 16, 1)
        y, _ = m(x, _bcs((True, False)))  # wall on W
        assert torch.isfinite(y).all()

    def test_mean_pool_variant(self):
        m = self._make(learned_pool=False)
        x = torch.randn(2, 256, 16, 16, 1, requires_grad=True)
        y, _ = m(x, _bcs((True, True)))
        assert torch.isfinite(y).all()
        y.sum().backward()
        assert torch.isfinite(x.grad).all()

    def test_tie_levels(self):
        m = self._make(tie_levels=True)
        x = torch.randn(2, 256, 16, 16, 1)
        assert torch.isfinite(m(x, _bcs((True, True)))[0]).all()

    def test_levels_auto(self):
        m = self._make()  # far_outer=3 -> min coarse extent 2*3+1 = 7
        assert m._levels((16, 16, 1)) == [2]         # p=4 extent 4 < 7 -> excluded
        assert m._levels((32, 32, 1)) == [2, 4]      # p=8 extent 4 < 7 -> excluded
        assert m._levels((8, 8, 1)) == []            # p=2 extent 4 < 7 -> near only

    def test_global_mop_up(self):
        """Shared coarsest level (finding 3): forward + grad in 2D and 3D."""
        for shape, per in (((2, 256, 16, 16, 1), (True, True)),
                           ((2, 256, 8, 8, 8), (True, True, True))):
            m = self._make(global_mop_up=True)
            x = torch.randn(*shape, requires_grad=True)
            y, _ = m(x, _bcs(per))
            assert y.shape == x.shape and torch.isfinite(y).all()
            y.sum().backward()
            assert x.grad is not None and torch.isfinite(x.grad).all()

    def test_fast_parity(self):
        """FMMAttentionFast (fused SDPA) == FMMAttention (manual joint softmax), weight-for-weight."""
        from walrus.models.spatial_blocks.fmm_attention_fast import FMMAttentionFast
        cases = [
            (dict(), (2, 256, 16, 16, 1), (True, True)),
            (dict(), (2, 256, 8, 8, 8), (True, True, True)),
            (dict(global_mop_up=True), (2, 256, 16, 16, 1), (True, True)),
            (dict(learned_pool=False), (2, 256, 16, 16, 1), (True, True)),
            (dict(), (2, 256, 16, 16, 1), (True, False)),  # wall on W
        ]
        for kw, shape, per in cases:
            m = FMMAttention(hidden_dim=256, num_heads=8, max_token_grid=64, **kw).eval()
            mf = FMMAttentionFast(hidden_dim=256, num_heads=8, max_token_grid=64, **kw).eval()
            miss, unexp = mf.load_state_dict(m.state_dict(), strict=True)
            assert not miss and not unexp, (miss, unexp)
            x = torch.randn(*shape)
            with torch.no_grad():
                y, _ = m(x, _bcs(per))
                yf, _ = mf(x, _bcs(per))
            assert torch.allclose(y, yf, atol=1e-4, rtol=1e-4), (kw, (y - yf).abs().max().item())

    def test_fused_parity_anchor(self):
        """FMMAttentionFused ANCHOR config (leaf_near=False, pool_rank=1: streaming near +
        rank-1 coarse far) == base FMMAttention, weight-for-weight."""
        from walrus.models.spatial_blocks.fmm_attention_fused import FMMAttentionFused
        cases = [
            (dict(), (2, 256, 16, 16, 1), (True, True)),
            (dict(), (2, 256, 8, 8, 8), (True, True, True)),
            (dict(global_mop_up=True), (2, 256, 16, 16, 1), (True, True)),
            (dict(), (2, 256, 16, 16, 1), (True, False)),  # wall on W
        ]
        for kw, shape, per in cases:
            m = FMMAttention(hidden_dim=256, num_heads=8, max_token_grid=64, **kw).eval()
            mu = FMMAttentionFused(hidden_dim=256, num_heads=8, max_token_grid=64,
                                   leaf_near=False, pool_rank=1, **kw).eval()
            mu.load_state_dict(m.state_dict(), strict=True)
            x = torch.randn(*shape)
            with torch.no_grad():
                y, _ = m(x, _bcs(per))
                yu, _ = mu(x, _bcs(per))
            assert torch.allclose(y, yu, atol=1e-4, rtol=1e-4), (kw, (y - yu).abs().max().item())

    def test_fused_leaf_and_rank(self):
        """New Kang-style paths (leaf_near, pool_rank>1): forward + grad finite, 2D & 3D."""
        from walrus.models.spatial_blocks.fmm_attention_fused import FMMAttentionFused
        cfgs = [
            dict(leaf_near=True, pool_rank=1),
            dict(leaf_near=True, pool_rank=4),
            dict(leaf_near=False, pool_rank=4),
            dict(leaf_near=True, pool_rank=4, global_mop_up=True),
        ]
        for kw in cfgs:
            for shape, per in (((2, 256, 16, 16, 1), (True, True)),
                               ((2, 256, 8, 8, 8), (True, True, True))):
                m = FMMAttentionFused(hidden_dim=256, num_heads=8, max_token_grid=64, **kw)
                x = torch.randn(*shape, requires_grad=True)
                y, _ = m(x, _bcs(per))
                assert y.shape == x.shape and torch.isfinite(y).all(), kw
                y.sum().backward()
                assert x.grad is not None and torch.isfinite(x.grad).all(), kw

    def test_fused_leaf_near_logits(self):
        """The block-tridiagonal near logits match an explicit brute-force reference."""
        from walrus.models.spatial_blocks.fmm_attention_fused import FMMAttentionFused
        torch.manual_seed(1)
        m = FMMAttentionFused(hidden_dim=64, num_heads=4, max_token_grid=64, leaf_near=True)
        B, he, H, W, c = 1, 4, 8, 8, 16
        qf = torch.randn(B, he, H, W, 1, c)
        kf = torch.randn(B, he, H, W, 1, c)
        vf = torch.randn(B, he, H, W, 1, c)
        near_logits, mask, _ = m._leaf_near(qf, kf, vf, [True, True, True], (H, W, 1))
        # brute force: for query (i,j), block=(i//pb, j//pb); it attends fine (a,b) with
        # block-distance <= 1 (periodic). Check every stored logit hits such a key with q·k.
        pb = m.pool_base
        scale = c ** -0.5
        ref = {}  # (i,j) -> set of scores it should contain
        for i, j in itertools.product(range(H), range(W)):
            bi, bj = i // pb, j // pb
            vals = []
            for a, b in itertools.product(range(H), range(W)):
                if (abs((a // pb) - bi) <= 1 or abs((a // pb) - bi) >= (H // pb) - 1) and \
                   (abs((b // pb) - bj) <= 1 or abs((b // pb) - bj) >= (W // pb) - 1):
                    vals.append(round(float((qf[0, 0, i, j, 0] * kf[0, 0, a, b, 0]).sum() * scale), 3))
            ref[(i, j)] = sorted(vals)
        # the valid logits for each query (sorted) must equal the reference set
        for i, j in [(0, 0), (3, 5), (7, 7)]:
            got = near_logits[0, 0, i, j, 0][mask[i, j, 0]]
            got = sorted(round(float(v), 3) for v in got)
            assert got == ref[(i, j)], (i, j, len(got), len(ref[(i, j)]))

    def test_max_token_grid_sizing(self):
        """finding 2: pooling ModuleList sized to exactly the levels a grid needs."""
        # 64 grid, far_outer=3, pool_base=2 -> levels p=2,4,8 -> 3
        assert len(self._make(max_token_grid=64).pool_k) == 3
        # untied + mop-up reserves one extra slot
        assert len(self._make(max_token_grid=64, global_mop_up=True).pool_k) == 4
        # tied collapses to a single shared projection
        assert len(self._make(max_token_grid=64, tie_levels=True, global_mop_up=True).pool_k) == 1
        # default (no hint) keeps the safe over-allocation
        assert len(self._make().pool_k) == 8

    def test_warm_start_statedict(self):
        """Shared FF/Q/K/V/out projections load from a flat FullAttention checkpoint;
        only the FMM-specific params (pooling, coarse key-norm) are missing."""
        fmm = self._make()
        fa = FullAttention(hidden_dim=256, num_heads=8)
        missing, unexpected = fmm.load_state_dict(fa.state_dict(), strict=False)
        assert not unexpected, f"unexpected keys from flat ckpt: {unexpected}"
        for k in missing:
            assert ("pool_k" in k or "pool_v" in k or "coarse_k_norm" in k), k

    def test_tiling_periodic(self):
        """Interaction-list quality on a 32x32 periodic grid. The telescoping annuli must
        NOT alias (each key covered at most twice — only the thin query-centered-near vs
        parent-centered-far seam), the reach must be translation-invariant, and coverage
        must be substantial (the outermost ring — the weakly-coupled 1/r tail — is dropped
        by design, so reach is < N, v1)."""
        cnt = _coverage(32, 32, near_r=3, pool_base=2, fi=2, fo=3, periodic=(True, True))
        N = cnt.shape[0]
        overlap = (cnt > 1).sum() / cnt.size
        exact_once = (cnt == 1).sum() / max((cnt > 0).sum(), 1)
        per_query_reach = (cnt > 0).sum(axis=1)
        assert cnt.max() <= 2, f"aliasing: a key is covered {cnt.max()}x (>2)"
        assert per_query_reach.min() == per_query_reach.max(), "reach not translation-inv"
        assert per_query_reach.min() > 0.7 * N, f"reach too small: {per_query_reach.min()}/{N}"
        assert exact_once > 0.85, f"too much double-counting: exact_once={exact_once:.3f}"
        assert overlap < 0.10, f"overlap fraction too high: {overlap:.3f}"


if __name__ == "__main__":
    unittest.main()
