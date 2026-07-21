"""Verify the 'score far at coarse resolution (no upsample)' optimization.

Isolates ONE far ring and computes s_far two ways:
  A (current): _neighbors(coarse) -> UPSAMPLE K/V to fine -> score einsum         (materializes K·c at FINE res)
  B (proposed): _neighbors(coarse) -> GROUP queries by parent -> score at COARSE -> ungroup   (K·c stays coarse)
Both produce the same (B,he,H,W,D,K) score tensor. We (1) assert they're numerically equal,
(2) time them, (3) report the size of the big materialized key tensor for each.
Run:  python scratch_fmm_coarse_score.py
"""

import time

import torch
from einops import rearrange

torch.set_num_threads(8)
torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

from walrus.models.spatial_blocks.fmm_attention import FMMAttention

HID, HEADS = 256, 8
C = HID // HEADS
PER = [True, True, True]
WARMUP, REPS = 3, 20


def mb(t):
    return t.numel() * t.element_size() / 1e6


def method_A(m, qf, ck, cv, pool, sizes):
    """Current: gather coarse annulus, UPSAMPLE to fine, then score."""
    kc_n, _ = m._neighbors(ck, m.far_outer, PER, inner=m.far_inner)
    vc_n, _ = m._neighbors(cv, m.far_outer, PER, inner=m.far_inner)
    kc_up = m._upsample_to_fine(kc_n, pool, sizes, (2, 3, 4))       # <-- the fine-res blow-up
    vc_up = m._upsample_to_fine(vc_n, pool, sizes, (2, 3, 4))
    s = torch.einsum("bhxyzc,bhxyzkc->bhxyzk", qf, kc_up)
    return s, mb(kc_up) + mb(vc_up)


def method_B(m, qf, ck, cv, pool, sizes):
    """Proposed: gather coarse annulus, GROUP queries by parent, score at COARSE res, ungroup."""
    H, W, D = sizes
    kc_n, _ = m._neighbors(ck, m.far_outer, PER, inner=m.far_inner)  # (B,he,Hc,Wc,Dc,K,c)
    ph, pw, pd = pool, pool, (pool if D > 1 else 1)
    qf_g = rearrange(qf, "b he (hc ph) (wc pw) (dc pd) c -> b he hc wc dc (ph pw pd) c",
                     ph=ph, pw=pw, pd=pd)                            # group children under parent
    s_g = torch.einsum("bhABCrc,bhABCkc->bhABCrk", qf_g, kc_n)      # score at COARSE res
    s = rearrange(s_g, "b he hc wc dc (ph pw pd) k -> b he (hc ph) (wc pw) (dc pd) k",
                  ph=ph, pw=pw, pd=pd)
    return s, mb(kc_n)  # V mirror is symmetric; report key tensor size


def timeit(fn, *a):
    for _ in range(WARMUP):
        fn(*a)
    if DEV == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(REPS):
        fn(*a)
    if DEV == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / REPS * 1e3


def run(tag, B, grid, spatial_dims, pool=2):
    m = FMMAttention(hidden_dim=HID, num_heads=HEADS, spatial_dims=spatial_dims,
                     max_token_grid=grid).to(DEV)
    if spatial_dims == 3:
        H = W = D = grid; Hc = Wc = Dc = grid // pool
    else:
        H = W = grid; D = 1; Hc = Wc = grid // pool; Dc = 1
    sizes = (H, W, D)
    qf = torch.randn(B, HEADS, H, W, D, C, device=DEV)
    ck = torch.randn(B, HEADS, Hc, Wc, Dc, C, device=DEV)
    cv = torch.randn(B, HEADS, Hc, Wc, Dc, C, device=DEV)

    with torch.no_grad():
        sA, memA = method_A(m, qf, ck, cv, pool, sizes)
        sB, memB = method_B(m, qf, ck, cv, pool, sizes)
        ok = torch.allclose(sA, sB, atol=1e-4)
        tA = timeit(lambda: method_A(m, qf, ck, cv, pool, sizes))
        tB = timeit(lambda: method_B(m, qf, ck, cv, pool, sizes))
    print(f"{tag:14} equal={ok!s:5}  A(upsample) {tA:8.1f}ms {memA:7.0f}MB   "
          f"B(coarse) {tB:8.1f}ms {memB:7.0f}MB   speedup {tA/tB:5.1f}x  mem {memA/memB:4.1f}x")


if __name__ == "__main__":
    print(f"device={DEV}  hidden={HID} heads={HEADS} head_dim={C}\n")
    print("=== 2D ring (pool=2) ===")
    run("2D 32 B=16", 16, 32, 2)
    run("2D 64 B=16", 16, 64, 2)
    print("\n=== 3D ring (pool=2), small batch to keep A in RAM ===")
    run("3D 16 B=2", 2, 16, 3)
    print("\nspeedup>1 → coarse-scoring is faster; mem>1 → it materializes less.")
