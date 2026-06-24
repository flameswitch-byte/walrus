"""Custom training losses for Walrus.

Currently houses ``SpectralLogMAE`` (roadmap idea 4.A.1): a soft, universally-safe
energy-spectrum penalty that punishes the small-scale blurring an L1 loss is blind to.

Design notes (see knowledge_base/walrus_physics_foundation_roadmap.md, 4.A.1):

* COMPUTED IN NORMALIZED-DELTA SPACE. At the training loss site the prediction and
  target are the per-sample/-channel whitened delta, which keeps each sample O(1).

* RELATIVE FLOOR, NOT ABSOLUTE EPS (critical -- learned the hard way). The log floor
  must scale with each sample's own energy (``rel_floor * max-ring-power incl DC``).
  An absolute ``psd_eps`` is NOT safe even in normalized space: the *field* is O(1) but
  the *per-ring PSD* is not -- high-k rings of smooth / low-k-dominated systems are
  ~0, far below any fixed eps. There ``log(E)`` and its ``1/E`` gradient explode, the
  spectral gradient becomes ~5x the L1 (even at weight 0.02), and with grad clipping
  the update is dominated by the spectral direction -> the main task is STARVED for
  exactly the smooth systems (active_matter, planetswe, rayleigh_benard, shear_flow,
  viscoelastic all froze; acoustic/helmholtz/gray_scott/turbulent kept learning). The
  relative floor brings the smooth-field spectral gradient to ~0.05-0.1x the L1 while
  still penalizing high-k blurring on rough fields thousands-fold.

* LOG, NOT RAW. ``mean | log E_pred - log E_ref |``. The log makes the penalty
  scale-balanced across rings (a factor-2 error in the tiny high-k tail counts the same
  as in the energy-containing low-k modes) and per-sample scale-invariant, so loud
  systems do not dominate the batch gradient the way a raw PSD-MSE would.

* DC RING DROPPED. The k=0 bin is the field mean (zero after whitening) / total energy;
  conservation is a separate, gated concern (idea 4.A.3), not this term's job.

* WINDOWING GATED ON BCs. A plain FFT assumes periodicity; on wall/open domains the
  unmatched edges inject spurious high-k energy (Gibbs). We apply a separable Hann
  window only when the dataset is not all-periodic. The valid (unpadded) region is
  already what the trainer hands us, so the padding Gibbs source does not arise here.
"""

from __future__ import annotations

from typing import Optional

import torch

from the_well.benchmark.metrics.common import Metric
from the_well.data.datasets import BoundaryCondition, WellMetadata


def _hann_window_nd(
    spatial_shape: tuple[int, ...],
    axes_to_window: tuple[bool, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Separable Hann window over the spatial dims, identity on un-windowed axes.

    Returns a tensor broadcastable over ``[..., *spatial, C]`` (spatial dims, trailing
    singleton channel).
    """
    nsd = len(spatial_shape)
    w = torch.ones(spatial_shape, device=device, dtype=dtype)
    for i, (n, do_window) in enumerate(zip(spatial_shape, axes_to_window)):
        if not do_window or n < 2:
            continue
        w1d = torch.hann_window(n, periodic=False, device=device, dtype=dtype)
        shape = [1] * nsd
        shape[i] = n
        w = w * w1d.reshape(shape)
    return w.unsqueeze(-1)  # trailing channel axis


def _radial_psd(
    field: torch.Tensor,
    n_spatial_dims: int,
    n_bins: Optional[int],
    count_eps: float,
) -> torch.Tensor:
    """Isotropic, radially-averaged power spectral density.

    ``field`` is ``[..., s_1, ..., s_nsd, C]`` (channels last, the_well convention).
    Returns ring-averaged power ``[..., n_bins, C]``.

    Uses ``rfftn`` (half spectrum on the last spatial axis). The dropped Hermitian half
    is a per-ring constant factor that cancels in the log-ratio against the reference,
    so it is harmless for this loss; we do not reweight it.
    """
    spatial_dims = tuple(range(-n_spatial_dims - 1, -1))
    spatial_shape = tuple(field.shape[d] for d in spatial_dims)

    fx = torch.fft.rfftn(field, dim=spatial_dims)
    power = fx.real**2 + fx.imag**2  # [..., k_1, ..., k_last_half, C]

    # |k| grid in cycles-per-box (fftfreq * n) so it is resolution-aware.
    ks = []
    for i, n in enumerate(spatial_shape):
        if i < n_spatial_dims - 1:
            freq = torch.fft.fftfreq(n, device=field.device) * n
        else:
            freq = torch.fft.rfftfreq(n, device=field.device) * n
        shape = [1] * n_spatial_dims
        shape[i] = -1
        ks.append(freq.reshape(shape))
    kmag = torch.sqrt(sum(k**2 for k in ks)).flatten()  # [K]

    if n_bins is None:
        # the_well-style default: ~sqrt of the smallest axis, one ring per integer-ish |k|.
        n_bins = max(2, int(min(spatial_shape) ** 0.5))
    edges = torch.linspace(
        0.0, kmag.max().item() + 1e-6, n_bins + 1, device=field.device
    )
    idx = (torch.bucketize(kmag, edges, right=True) - 1).clamp_(0, n_bins - 1)  # [K]

    lead = power.shape[: -n_spatial_dims - 1]
    n_ch = power.shape[-1]
    pflat = power.reshape(*lead, -1, n_ch)  # [..., K, C]

    ring_power = torch.zeros(
        *lead, n_bins, n_ch, device=field.device, dtype=pflat.dtype
    )
    ring_power.index_add_(-2, idx, pflat)
    counts = torch.zeros(n_bins, device=field.device, dtype=pflat.dtype)
    counts.index_add_(0, idx, torch.ones_like(kmag))
    counts = counts.reshape(*([1] * len(lead)), n_bins, 1)
    return ring_power / (counts + count_eps)  # ring-mean power [..., n_bins, C]


class SpectralLogMAE(Metric):
    """Energy-spectrum loss: MAE of the log radially-averaged PSD (idea 4.A.1).

    Subclasses the_well ``Metric`` so it shares the channels-last / spatial-dims
    convention and can double as a validation metric, but it is intended to be added to
    the L1 training loss with its own weight.

    Args:
        psd_eps: additive floor inside the log. Tuned for normalized-delta space (fields
            are O(1)); do not reuse a physical-units value here.
        n_bins: number of |k| rings. None -> ~sqrt(min spatial extent).
        drop_dc: skip the k=0 ring (field mean / total energy).
        window_policy: when to apply the Hann window -- "nonperiodic" (default; only on
            datasets with a non-periodic axis), "always", or "never".
    """

    def __init__(
        self,
        psd_eps: float = 1e-8,
        rel_floor: float = 1e-2,
        n_bins: Optional[int] = None,
        drop_dc: bool = True,
        window_policy: str = "nonperiodic",
    ):
        super().__init__()
        assert window_policy in ("nonperiodic", "always", "never")
        self.psd_eps = psd_eps
        # Floor inside the log, RELATIVE to each sample's full-spectrum energy. This is
        # the critical knob: an absolute floor (psd_eps alone) makes the log and its
        # 1/E gradient explode on smooth / low-k-dominated fields whose non-DC rings
        # are ~0, which (via grad clipping) starves the main task for exactly those
        # systems. ~1e-2 keeps the spectral gradient ~0.05-0.1x the L1 gradient on
        # smooth fields while still penalizing high-k blurring on rough fields. See
        # knowledge_base/walrus_validation_logging.md / spectral failure mode.
        self.rel_floor = rel_floor
        self.n_bins = n_bins
        self.drop_dc = drop_dc
        self.window_policy = window_policy

    def _should_window(self, meta: WellMetadata) -> bool:
        if self.window_policy == "always":
            return True
        if self.window_policy == "never":
            return False
        bcs = getattr(meta, "boundary_condition_types", None)
        if not bcs:
            return False  # unknown -> assume periodic, do not blunt the spectrum
        periodic = BoundaryCondition.PERIODIC.name
        return any(str(bc).upper().split(".")[-1] != periodic for bc in bcs)

    def eval(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        meta: WellMetadata,
        eps: Optional[float] = None,  # accepted for interface parity; not the PSD floor
    ) -> torch.Tensor:
        nsd = meta.n_spatial_dims
        # FFT is fp32-only under AMP; the normalization code uses the same guard.
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            y = y.float()
            if self._should_window(meta):
                spatial_shape = tuple(x.shape[d] for d in range(-nsd - 1, -1))
                w = _hann_window_nd(
                    spatial_shape, (True,) * nsd, x.device, x.dtype
                )
                x = x * w
                y = y * w

            ex_full = _radial_psd(x, nsd, self.n_bins, self.psd_eps)  # incl DC ring
            ey_full = _radial_psd(y, nsd, self.n_bins, self.psd_eps)
            # Relative floor inside the log, scaled by the reference's full-spectrum
            # energy: max ring power over BOTH rings (incl the DC bin) AND channels.
            # - incl DC: the DC-dropped rings of a smooth field are ~0, so a floor from
            #   them (or an absolute psd_eps) does nothing and 1/E explodes.
            # - over channels (per-lead, not per-channel): an identically-zero reference
            #   channel (e.g. a `density: torch.zeros_like` field) has amax==0, which
            #   would collapse a per-channel floor back to psd_eps and re-open the
            #   blowup on that channel. Maxing over channels lets a zeroed channel
            #   inherit the floor from the sample's real energy. Channels are O(1) in
            #   normalized-delta space, so a shared per-sample floor is well-scaled.
            floor = (
                self.rel_floor * ey_full.amax(dim=(-2, -1), keepdim=True)
                + self.psd_eps
            )
            start = 1 if self.drop_dc else 0
            ex = ex_full[..., start:, :]
            ey = ey_full[..., start:, :]

            log_err = (torch.log(ex + floor) - torch.log(ey + floor)).abs()
            # Mean over rings -> per-(lead, channel), matching MAE's [..., C] return so
            # the trainer's .mean() behaves identically.
            return log_err.mean(dim=-2)
