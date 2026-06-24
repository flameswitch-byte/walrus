"""Gated spectral (inverse-Laplacian / Leray) operator stage.

Implements the **operator-split block** of ``knowledge_base/walrus_attention_design.md``
§13.2 / §13.4 / §13.10: an attention backbone followed by a *gated* full-width spectral
sub-layer that represents the **elliptic pressure operator** (``p = ∇⁻²(source)``) — which
is *diagonal in Fourier* (``p̂(k) = −source(k)/|k|²``) and therefore exactly representable,
where softmax attention can only approximate it.

Two pieces:

  * :class:`SpectralOperator` — ``norm → in_proj(1×1) → rfftn → per-channel complex filter
    (init ∝ −1/|k|², the inverse-Laplacian) → irfftn → out_proj(1×1)``. Runs in **fp32**
    (FFT is inaccurate/unsupported in bf16). **v1 is periodic** (FFT on every non-singleton
    spatial axis); wall axes (rayleigh) need DCT/DST and are deferred (§13.5) — multipole+local
    is the named FMM fallback there.

  * :class:`OperatorSplitBlock` — a drop-in ``space_mixing`` wrapper that holds an inner
    attention block (``FullAttention`` *or* ``HeterogeneousAttention``, given as a Hydra
    ``_partial_``) and appends the spectral stage as an **additive, gated residual**::

        x, att = inner_attention(x, bcs)            # unchanged backbone
        x = x + gate ⊙ SpectralOperator(x, bcs)     # new stage

    The gate is **zero-init** (per-channel layer-scale by default, §13.3 "soft static mix
    first"), so at init the block is **bit-for-bit identical** to the inner backbone — a
    flat/hetero checkpoint warm-starts exactly (operator params are new; gate = 0). The gate
    still receives gradient (operator output ≠ 0 thanks to the inverse-Laplacian init), so it
    unsticks from zero when the operator helps. A data-dependent physics gate (§13.3) is the
    documented next step; ``gate_mode`` leaves room for it.

This is the same wrapper for both backbones — "build on both flat and hetero and gate it".
"""

import math
from typing import Callable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from ..shared_utils.normalization import RMSGroupNorm


class SpectralOperator(nn.Module):
    """Full-width spectral inverse-Laplacian operator (periodic, fp32).

    ``out = out_proj( iFFT( filter(k) ⊙ FFT( in_proj(norm(x)) ) ) )`` with the filter
    initialized to the per-channel inverse-Laplacian ``−1/|k|²`` (DC mode zeroed).
    """

    def __init__(
        self,
        hidden_dim: int,
        norm_groups: int = 8,
        keep_modes: Optional[int] = None,
        laplace_power: float = 1.0,
        norm_layer: Callable = RMSGroupNorm,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.keep_modes = keep_modes
        self.laplace_power = float(laplace_power)
        self.norm = norm_layer(norm_groups, hidden_dim, affine=True)

        # 1×1 channel mixers (real space): in_proj forms the "source" combination,
        # out_proj distributes the "potential". Identity-init so the stage starts as a
        # clean per-channel inverse-Laplacian once the gate opens.
        self.in_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.eye_(self.in_proj.weight)
        nn.init.eye_(self.out_proj.weight)

        # Per-channel complex gain on the inverse-Laplacian base; init 1 + 0j (a real,
        # purely-even operator). The imaginary part lets the operator learn a non-even
        # correction that a real 1×1 conv cannot represent.
        self.gain_re = nn.Parameter(torch.ones(hidden_dim))
        self.gain_im = nn.Parameter(torch.zeros(hidden_dim))

        self._base_cache: dict = {}

    # ------------------------------------------------------------------ filter
    def _base_filter(
        self, sizes: Tuple[int, int, int], fft_dims: List[int], device, dtype
    ) -> Tensor:
        """Real ``−1/|k|^(2·power)`` base on the transformed grid, DC zeroed.

        ``fft_dims`` index into the channels-last spatial axes ``(1, 2, 3)`` of an
        ``(b, h, w, d, c)`` tensor. Shape: the rfftn output spatial shape, ready to
        broadcast as ``(1, *spatial, 1)``.
        """
        key = (sizes, tuple(fft_dims), self.keep_modes, self.laplace_power, device)
        cached = self._base_cache.get(key)
        if cached is not None:
            return cached

        last = fft_dims[-1]
        k_axes: List[Tensor] = []
        mode_axes: List[Tensor] = []
        for ax in (1, 2, 3):
            S = sizes[ax - 1]
            if ax in fft_dims:
                if ax == last:
                    f = torch.fft.rfftfreq(S, device=device, dtype=torch.float32)
                else:
                    f = torch.fft.fftfreq(S, device=device, dtype=torch.float32)
            else:
                f = torch.zeros(1, device=device, dtype=torch.float32)
            k_axes.append(2.0 * math.pi * f)
            # integer-ish mode index magnitude for keep_modes truncation
            mode_axes.append((f * S).round().abs())

        kh, kw, kd = k_axes
        k2 = (
            (kh**2).view(-1, 1, 1)
            + (kw**2).view(1, -1, 1)
            + (kd**2).view(1, 1, -1)
        )
        base = torch.zeros_like(k2)
        nonzero = k2 > 0
        base[nonzero] = -1.0 / (k2[nonzero] ** self.laplace_power)  # inverse-Laplacian; DC = 0

        if self.keep_modes is not None:
            mh, mw, md = mode_axes
            keep = (
                (mh.view(-1, 1, 1) <= self.keep_modes)
                & (mw.view(1, -1, 1) <= self.keep_modes)
                & (md.view(1, 1, -1) <= self.keep_modes)
            )
            base = base * keep.to(base.dtype)

        base = base.view(1, *base.shape, 1)  # (1, lh, lw, ld, 1)
        self._base_cache[key] = base
        return base

    # ------------------------------------------------------------------ forward
    def forward(self, x: Tensor, bcs=None) -> Tensor:
        # x: (b, c, h, w, d)
        B, C, H, W, D = x.shape
        sizes = (H, W, D)
        in_dtype = x.dtype

        # channels-last spatial axes (1, 2, 3); transform only non-singleton axes
        x = rearrange(self.norm(x), "b c h w d -> b h w d c")
        fft_dims = [a for a in (1, 2, 3) if x.shape[a] > 1]
        if not fft_dims:  # nothing to transform (all spatial size 1) -> no-op
            return torch.zeros(B, C, H, W, D, device=x.device, dtype=in_dtype)

        # fp32 throughout (FFT + complex filter); disable autocast so matmuls stay fp32.
        with torch.autocast(device_type=x.device.type, enabled=False):
            xf32 = x.float()
            src = self.in_proj(xf32)
            hat = torch.fft.rfftn(src, dim=fft_dims, norm="ortho")

            base = self._base_filter(sizes, fft_dims, x.device, torch.float32)
            gain = torch.complex(self.gain_re.float(), self.gain_im.float())
            filt = base.to(hat.dtype) * gain.view(1, 1, 1, 1, C)
            hat = hat * filt

            fft_sizes = [sizes[a - 1] for a in fft_dims]
            out = torch.fft.irfftn(hat, s=fft_sizes, dim=fft_dims, norm="ortho")
            out = self.out_proj(out)

        out = rearrange(out, "b h w d c -> b c h w d")
        return out.to(in_dtype)


class OperatorSplitBlock(nn.Module):
    """``space_mixing`` wrapper: inner attention backbone + gated spectral operator.

    ``attention`` is a Hydra ``_partial_`` for the inner block (``FullAttention`` or
    ``HeterogeneousAttention``); it is instantiated here with the runtime kwargs passed by
    ``SpaceTimeSplitBlock`` (hidden_dim / drop_path / gradient_checkpointing / norm_layer),
    exactly as ``space_mixing`` partials are normally handled.
    """

    def __init__(
        self,
        attention: Callable,
        hidden_dim: int = 768,
        drop_path: float = 0.0,
        gradient_checkpointing: bool = False,
        norm_layer: Callable = RMSGroupNorm,
        norm_groups: int = 8,
        keep_modes: Optional[int] = None,
        laplace_power: float = 1.0,
        gate_mode: str = "static_channel",
    ):
        super().__init__()
        self.attention = attention(
            hidden_dim=hidden_dim,
            drop_path=drop_path,
            gradient_checkpointing=gradient_checkpointing,
            norm_layer=norm_layer,
        )
        self.operator = SpectralOperator(
            hidden_dim=hidden_dim,
            norm_groups=norm_groups,
            keep_modes=keep_modes,
            laplace_power=laplace_power,
            norm_layer=norm_layer,
        )

        # Zero-init gate -> bit-exact warm-start from the inner backbone (§13.3 soft mix).
        self.gate_mode = gate_mode
        if gate_mode == "static_channel":
            self.gate = nn.Parameter(torch.zeros(hidden_dim))
        elif gate_mode == "static_scalar":
            self.gate = nn.Parameter(torch.zeros(1))
        else:
            raise ValueError(
                f"gate_mode must be 'static_channel' or 'static_scalar', got {gate_mode!r}. "
                "(data-dependent physics gate is §13.3, not yet built.)"
            )

    def make_rope_learnable(self, per_axis: bool = False):
        if hasattr(self.attention, "make_rope_learnable"):
            self.attention.make_rope_learnable(per_axis)

    def forward(self, x, bcs, return_att=False):
        x, att = self.attention(x, bcs, return_att=return_att)
        op = self.operator(x, bcs)
        if self.gate_mode == "static_channel":
            g = self.gate.view(1, -1, 1, 1, 1)
        else:
            g = self.gate
        x = x + g * op
        return x, att
