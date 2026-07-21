"""Standalone Poisson sanity check for FMMAttention (design study §12.10).

Trains a tiny stack of attention blocks to apply the periodic inverse-Laplacian
u = (-Δ)⁻¹ f on a 2D torus (the canonical elliptic / global-coupling task — the field
§10.4 says only global attention captures). Compares three space-mixers at matched size:

    full      flat FullAttention            (O(N²) exact global — the reference ceiling)
    fmm       FMMAttention (near + far)      (O(N log N) — the design under test)
    near      FMMAttention (max_levels=0)    (near-only ablation — NO global path)

If `fmm` tracks `full` and both crush `near`, the far-field is carrying the global
elliptic coupling as designed. Run:  python scratch_fmm_poisson.py
"""

import torch
import torch.nn as nn

torch.set_num_threads(8)  # FMM has many small ops; all-core oversubscription thrashes CPU

from walrus.models.spatial_blocks.fmm_attention import FMMAttention
from walrus.models.spatial_blocks.fmm_attention_fused import FMMAttentionFused
from walrus.models.spatial_blocks.full_attention import FullAttention

torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
H = W = 16
C = 48
HEADS = 4
NBLOCKS = 3
BS = 8
STEPS = 500
P = 1  # BoundaryCondition PERIODIC value (matches the_well enum; see test helper)
BCS = [[[P, P], [P, P]]]  # periodic on both axes


def poisson_batch(bs):
    """Random low-pass mean-zero source f and its periodic Poisson solution u."""
    f = torch.randn(bs, 1, H, W, device=DEV)
    fk = torch.fft.rfft2(f)
    ky = torch.fft.fftfreq(H, d=1.0 / H, device=DEV).view(H, 1)
    kx = torch.fft.rfftfreq(W, d=1.0 / W, device=DEV).view(1, W // 2 + 1)
    k2 = (ky ** 2 + kx ** 2)
    lowpass = (k2 <= (8 ** 2)).float()            # keep low modes -> smooth, solvable
    fk = fk * lowpass
    inv = torch.zeros_like(k2)
    inv[k2 > 0] = 1.0 / k2[k2 > 0]                # (-Δ)⁻¹ in Fourier; DC -> 0
    uk = fk * inv
    f = torch.fft.irfft2(fk, s=(H, W))
    u = torch.fft.irfft2(uk, s=(H, W))
    # normalize per-sample so the loss is scale-free
    f = f / f.flatten(1).std(1).view(-1, 1, 1, 1)
    u = u / u.flatten(1).std(1).view(-1, 1, 1, 1)
    return f, u


class Solver(nn.Module):
    def __init__(self, mixer_fn):
        super().__init__()
        self.inp = nn.Linear(1, C)
        self.blocks = nn.ModuleList([mixer_fn() for _ in range(NBLOCKS)])
        self.out = nn.Linear(C, 1)

    def forward(self, f):
        # f: (B,1,H,W) -> tokens (B,C,H,W,1)
        x = self.inp(f.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).unsqueeze(-1)
        for blk in self.blocks:
            x, _ = blk(x, BCS)
        x = x.squeeze(-1).permute(0, 2, 3, 1)
        return self.out(x).permute(0, 3, 1, 2)


def mixers():
    common = dict(hidden_dim=C, num_heads=HEADS, mlp_dim=C * 2)
    fused = dict(**common, near_radius=3, pool_base=2, far_inner=2, far_outer=3,
                 learned_pool=True, max_token_grid=64)
    return {
        "full": lambda: FullAttention(hidden_dim=C, num_heads=HEADS, mlp_dim=C * 2),
        "fmm": lambda: FMMAttention(**common, near_radius=3, pool_base=2,
                                    far_inner=2, far_outer=3, learned_pool=True),
        # upgraded (Kang): leaf-aligned near + rank-4 far
        "fused_leaf_p4": lambda: FMMAttentionFused(**fused, leaf_near=True, pool_rank=4),
        "fused_leaf_p4_mop": lambda: FMMAttentionFused(**fused, leaf_near=True, pool_rank=4,
                                                       global_mop_up=True),
        "near": lambda: FMMAttention(**common, near_radius=3, max_levels=0),
    }


def rel_err(pred, u):
    return (pred - u).flatten(1).norm(dim=1).mean() / u.flatten(1).norm(dim=1).mean()


def train(name, mixer_fn):
    model = Solver(mixer_fn).to(DEV)
    n_param = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    fixed_f, fixed_u = poisson_batch(64)  # held-out eval batch
    for step in range(STEPS):
        f, u = poisson_batch(BS)
        pred = model(f)
        loss = ((pred - u) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 150 == 0 or step == STEPS - 1:
            with torch.no_grad():
                e = rel_err(model(fixed_f), fixed_u).item()
            print(f"  [{name:4s}] step {step:4d}  loss {loss.item():.4f}  rel_err {e:.4f}")
    with torch.no_grad():
        return rel_err(model(fixed_f), fixed_u).item(), n_param


if __name__ == "__main__":
    print(f"device={DEV} grid={H}x{W} C={C} heads={HEADS} blocks={NBLOCKS}")
    results = {}
    for name, fn in mixers().items():
        print(f"--- {name} ---")
        results[name] = train(name, fn)
    print("\n=== final relative L2 error (lower better) ===")
    for name, (e, n) in results.items():
        print(f"  {name:4s}  rel_err={e:.4f}  params={n/1e3:.0f}K")
    print("\nExpect: fmm << near (far-field carries the global elliptic coupling),"
          " and fmm approaching full.")
