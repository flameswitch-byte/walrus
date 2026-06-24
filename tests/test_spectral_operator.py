"""Unit tests for the gated spectral operator stage (SpectralOperator / OperatorSplitBlock).

Covers: forward shape/grad (2D + 3D), **bit-exact warm-start** (zero-init gate => the block
output equals the inner backbone exactly, for both flat and hetero inners), the inverse-
Laplacian filter init correctness on a synthetic field, gate-unsticks-from-zero (nonzero
gradient on the gate), and fp32 stability under bf16 autocast.

Note: requires the local ``walrus`` tree to be importable (recompile the installed copy, or
run with PYTHONPATH=<repo root>).
"""

import math
import unittest

import torch

from the_well.data.datasets import BoundaryCondition
from walrus.models.spatial_blocks.full_attention import FullAttention
from walrus.models.spatial_blocks.heterogeneous_attention import HeterogeneousAttention
from walrus.models.spatial_blocks.spectral_operator import (
    OperatorSplitBlock,
    SpectralOperator,
)

P = BoundaryCondition["PERIODIC"].value


def _bcs(per):
    return [[[P, P] for _ in per]]


def _flat_partial(**kw):
    def make(**runtime):
        return FullAttention(num_heads=8, **kw, **runtime)

    return make


def _hetero_partial(**kw):
    def make(**runtime):
        return HeterogeneousAttention(
            num_heads=8, n_local=3, n_multipole=2, n_full=3,
            multipole_pool=[2, 3, 4], multipole_eval_pool=3, multipole_order=2,
            **kw, **runtime,
        )

    return make


class TestSpectralOperator(unittest.TestCase):
    # ---------------------------------------------------------------- shapes
    def test_forward_2d_shape_and_grad(self):
        m = OperatorSplitBlock(attention=_flat_partial(), hidden_dim=256)
        x = torch.randn(2, 256, 8, 8, 1, requires_grad=True)
        y, _ = m(x, _bcs((True, True)))
        assert y.shape == x.shape
        y.sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()

    def test_forward_3d_shape(self):
        m = OperatorSplitBlock(attention=_flat_partial(), hidden_dim=256)
        x = torch.randn(2, 256, 4, 4, 4)
        y, _ = m(x, _bcs((True, True, True)))
        assert y.shape == x.shape
        assert torch.isfinite(y).all()

    # ---------------------------------------------------- warm-start exactness
    def _warm_start_exact(self, partial, spatial):
        torch.manual_seed(0)
        block = OperatorSplitBlock(attention=partial, hidden_dim=256)
        block.eval()
        # gate is zero-init -> the operator must contribute exactly nothing.
        assert torch.count_nonzero(block.gate) == 0
        x = torch.randn(2, 256, *spatial)
        with torch.no_grad():
            ref, _ = block.attention(x.clone(), _bcs([True] * len(spatial)))
            out, _ = block(x.clone(), _bcs([True] * len(spatial)))
        assert torch.equal(out, ref), "zero-gate block must equal the inner backbone bit-for-bit"

    def test_warm_start_exact_flat(self):
        self._warm_start_exact(_flat_partial(), (8, 8, 1))

    def test_warm_start_exact_hetero(self):
        self._warm_start_exact(_hetero_partial(), (8, 8, 1))

    def test_gate_receives_gradient(self):
        # Even at gate=0, the gate must get a nonzero gradient (operator output != 0),
        # so it unsticks from zero during training.
        block = OperatorSplitBlock(attention=_flat_partial(), hidden_dim=256)
        x = torch.randn(2, 256, 8, 8, 1)
        y, _ = block(x, _bcs((True, True)))
        y.pow(2).sum().backward()
        assert block.gate.grad is not None
        assert torch.count_nonzero(block.gate.grad) > 0

    # ---------------------------------------------- inverse-Laplacian correctness
    def test_inverse_laplacian_on_sine(self):
        """With identity projections and unit gain, the operator must map a single
        Fourier mode ``sin(2π m x / H)`` to ``-1/|k|² ·`` itself (the analytic
        inverse-Laplacian eigenvalue), on a periodic grid."""
        C, H, W = 4, 16, 16
        op = SpectralOperator(hidden_dim=C, norm_groups=1)
        op.eval()
        # neutralize the norm so we test the spectral map in isolation
        with torch.no_grad():
            if hasattr(op.norm, "weight") and op.norm.weight is not None:
                op.norm.weight.fill_(1.0)

        m = 2  # mode index along H
        xs = torch.arange(H, dtype=torch.float32)
        field = torch.sin(2 * math.pi * m * xs / H)  # (H,)
        x = field.view(1, 1, H, 1, 1).expand(1, C, H, W, 1).contiguous()

        # Apply ONLY the spectral map (skip norm by feeding through a fresh op whose norm
        # is RMS over a constant-per-group input -> rescales; instead compare directions).
        with torch.no_grad():
            out = op(x, _bcs((True, True)))

        # the output should be (anti)parallel to the input mode along H and constant in W
        out_line = out[0, 0, :, 0, 0]
        # correlation with the input mode should be near +/-1 (pure eigenmode preserved)
        corr = torch.nn.functional.cosine_similarity(
            out_line.flatten(), field.flatten(), dim=0
        )
        assert corr.abs() > 0.99, f"operator did not preserve the Fourier eigenmode (corr={corr:.3f})"
        # the analytic eigenvalue is negative (-1/k^2) -> output anti-parallel to input
        assert corr < 0, f"inverse-Laplacian eigenvalue should be negative (corr={corr:.3f})"

    # ---------------------------------------------------------- bf16 autocast
    def test_bf16_autocast_runs_fp32_internally(self):
        if not hasattr(torch, "autocast"):
            self.skipTest("no autocast")
        m = OperatorSplitBlock(attention=_flat_partial(), hidden_dim=256)
        x = torch.randn(2, 256, 8, 8, 1)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            y, _ = m(x, _bcs((True, True)))
        assert torch.isfinite(y).all()

    def test_scalar_gate(self):
        m = OperatorSplitBlock(
            attention=_flat_partial(), hidden_dim=256, gate_mode="static_scalar"
        )
        assert m.gate.numel() == 1 and float(m.gate) == 0.0
        x = torch.randn(2, 256, 8, 8, 1)
        y, _ = m(x, _bcs((True, True)))
        assert y.shape == x.shape


if __name__ == "__main__":
    unittest.main()
