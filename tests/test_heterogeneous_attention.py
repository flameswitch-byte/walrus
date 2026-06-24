"""Unit tests for HeterogeneousAttention (multi-reach spatial attention, v1).

Covers: forward shape/grad (2D + 3D), boundary-condition-aware masks, the
``n_strided=0`` ablation, allocation validation, warm-start equivalence to
FullAttention, and per-head mask correctness.

Note: requires the local ``walrus`` tree to be importable (the installed copy must
be recompiled, or run with PYTHONPATH=<repo root>).
"""

import unittest

import torch

from the_well.data.datasets import BoundaryCondition
from walrus.models.spatial_blocks.full_attention import FullAttention
from walrus.models.spatial_blocks.heterogeneous_attention import HeterogeneousAttention

P = BoundaryCondition["PERIODIC"].value
NONP = next(
    (bc.value for bc in BoundaryCondition if bc.value != P), P
)  # any non-periodic code (falls back to P if the enum has only one member)


def _bcs(per):
    """space_mixing bcs format: [batch, n_dims, 2] of boundary codes."""
    return [[[P if p else NONP, P if p else NONP] for p in per]]


class TestHeterogeneousAttention(unittest.TestCase):
    def _make(self, **kw):
        kw.setdefault("hidden_dim", 256)
        kw.setdefault("num_heads", 8)
        return HeterogeneousAttention(**kw)

    def test_forward_2d_shape_and_grad(self):
        m = self._make(n_local=4, n_strided=2, n_full=2, strided_strides=(2, 4))
        x = torch.randn(2, 256, 8, 8, 1, requires_grad=True)
        y, _ = m(x, _bcs((True, True)))
        assert y.shape == x.shape
        assert torch.isfinite(y).all()
        y.sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()

    def test_forward_3d(self):
        m = self._make(n_local=4, n_strided=2, n_full=2, strided_strides=(2, 4))
        x = torch.randn(2, 256, 4, 4, 4)
        y, _ = m(x, _bcs((True, True, True)))
        assert y.shape == x.shape and torch.isfinite(y).all()

    def test_wall_axis(self):
        m = self._make(n_local=4, n_strided=2, n_full=2, strided_strides=(2, 4))
        x = torch.randn(2, 256, 8, 8, 1)
        y, _ = m(x, _bcs((True, False)))  # wall on the W axis
        assert torch.isfinite(y).all()

    def test_n_strided_zero_ablation(self):
        m = self._make(n_local=6, n_strided=0, n_full=2, strided_strides=())
        x = torch.randn(2, 256, 8, 8, 1)
        y, _ = m(x, _bcs((True, True)))
        assert torch.isfinite(y).all()

    def test_allocation_must_sum(self):
        with self.assertRaises(ValueError):
            self._make(n_local=4, n_strided=2, n_full=4, strided_strides=(2, 4))

    def test_strides_must_match_n_strided(self):
        with self.assertRaises(ValueError):
            self._make(n_local=4, n_strided=2, n_full=2, strided_strides=(2,))

    def test_warm_start_equivalence(self):
        """All-full heads + copied flat weights == FullAttention, bit-for-bit."""
        hf = self._make(n_full=8)  # all-full -> mask is None
        fa = FullAttention(hidden_dim=256, num_heads=8)
        missing, unexpected = hf.load_state_dict(fa.state_dict(), strict=False)
        assert not missing and not unexpected
        hf.eval()
        fa.eval()
        with torch.no_grad():
            x = torch.randn(2, 256, 8, 8, 1)
            ya, _ = hf(x, _bcs((True, True)))
            yb, _ = fa(x, _bcs((True, True)))
        assert torch.allclose(ya, yb, atol=1e-5)

    def test_multipole_2d(self):
        m = self._make(n_local=4, n_strided=0, n_multipole=2, n_full=2,
                       strided_strides=(), multipole_pool=4, multipole_order=2)
        x = torch.randn(2, 256, 8, 8, 1, requires_grad=True)
        y, _ = m(x, _bcs((True, True)))
        assert y.shape == x.shape and torch.isfinite(y).all()
        y.sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()

    def test_multipole_3d(self):
        m = self._make(n_local=4, n_strided=0, n_multipole=2, n_full=2,
                       strided_strides=(), multipole_pool=2, multipole_order=2)
        x = torch.randn(2, 256, 4, 4, 4)
        y, _ = m(x, _bcs((True, True, True)))
        assert y.shape == x.shape and torch.isfinite(y).all()

    def test_multipole_order1(self):
        m = self._make(n_local=4, n_strided=0, n_multipole=2, n_full=2,
                       strided_strides=(), multipole_pool=4, multipole_order=1)
        x = torch.randn(2, 256, 8, 8, 1)
        y, _ = m(x, _bcs((True, True)))
        assert torch.isfinite(y).all()

    def test_multipole_variable_pool(self):
        m = self._make(n_local=4, n_strided=0, n_multipole=2, n_full=2,
                       strided_strides=(), multipole_pool=[2, 4], multipole_eval_pool=4)
        x = torch.randn(2, 256, 8, 8, 1)
        m.train()
        for _ in range(3):  # sampling path runs without error
            assert torch.isfinite(m(x, _bcs((True, True)))[0]).all()
        m.eval()
        with torch.no_grad():
            y1, _ = m(x, _bcs((True, True)))
            y2, _ = m(x, _bcs((True, True)))
        assert torch.allclose(y1, y2)  # eval is deterministic (fixed eval pool)

    def test_bad_pool_raises(self):
        with self.assertRaises(ValueError):
            self._make(n_local=4, n_strided=0, n_multipole=2, n_full=2,
                       strided_strides=(), multipole_pool=1)

    def test_full_palette(self):
        """All four head types active in one block."""
        m = self._make(n_local=2, n_strided=2, n_multipole=2, n_full=2,
                       strided_strides=(2, 4), multipole_pool=4)
        x = torch.randn(2, 256, 8, 8, 1, requires_grad=True)
        y, _ = m(x, _bcs((True, True)))
        assert y.shape == x.shape and torch.isfinite(y).all()
        y.sum().backward()
        assert torch.isfinite(x.grad).all()

    def test_mask_correctness(self):
        m = self._make(
            hidden_dim=32, n_local=4, n_strided=2, n_full=2, strided_strides=(2, 4)
        )
        mask = m._build_mask(4, 4, 1, (True, True, False), torch.device("cpu"))
        assert mask.shape == (8, 16, 16)

        def idx(h, w):
            return h * 4 + w

        local, strided2, full = mask[0], mask[4], mask[6]
        # no all-False rows (would NaN the softmax)
        assert mask.any(dim=2).all()
        # local (radius 1): sees adjacent (1,2), not the far corner (3,3)
        assert local[idx(1, 1), idx(1, 2)] and not local[idx(1, 1), idx(3, 3)]
        # strided s=2 (radius 1): sees offset-2 (3,3), not the adjacent (1,2)
        assert strided2[idx(1, 1), idx(3, 3)] and not strided2[idx(1, 1), idx(1, 2)]
        # full: everything
        assert full.all()


if __name__ == "__main__":
    unittest.main()
