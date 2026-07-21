"""Parity + wall-clock: FMMAttentionFast (fused SDPA) vs FMMAttention (manual joint softmax).

(1) PARITY: with identical weights, both forwards must produce the same output (CPU math
    backend == the base class's manual joint softmax; the CUDA flash path is the same op).
(2) TIMING: wall-clock on the current CPU (fused SDPA on CPU is the math backend, so no GPU
    speedup is expected here — the flash win is CUDA-only; this just confirms it's not slower
    on CPU and gives a baseline).
Run:  python scratch_fmm_fast_check.py
"""

import time

import torch
from the_well.data.datasets import BoundaryCondition

from walrus.models.spatial_blocks.full_attention import FullAttention
from walrus.models.spatial_blocks.fmm_attention import FMMAttention
from walrus.models.spatial_blocks.fmm_attention_fast import FMMAttentionFast

torch.set_num_threads(8)
torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
P = BoundaryCondition["PERIODIC"].value
NONP = next((bc.value for bc in BoundaryCondition if bc.value != P), P)
HID, HEADS = 256, 8


def bcs(per):
    return [[[P if p else NONP, P if p else NONP] for p in per]]


def parity():
    print("=== PARITY (max abs diff, must be ~0) ===")
    cfgs = [
        ("2D 16 periodic", dict(), (2, HID, 16, 16, 1), (True, True)),
        ("2D 32 periodic", dict(), (2, HID, 32, 32, 1), (True, True)),
        ("2D 16 wall-W", dict(), (2, HID, 16, 16, 1), (True, False)),
        ("3D 8 periodic", dict(), (2, HID, 8, 8, 8), (True, True, True)),
        ("2D 32 mop-up", dict(global_mop_up=True), (2, HID, 32, 32, 1), (True, True)),
        ("2D 32 mean-pool", dict(learned_pool=False), (2, HID, 32, 32, 1), (True, True)),
        ("2D 32 tie-levels", dict(tie_levels=True), (2, HID, 32, 32, 1), (True, True)),
    ]
    all_ok = True
    for tag, kw, shape, per in cfgs:
        m = FMMAttention(hidden_dim=HID, num_heads=HEADS, max_token_grid=64, **kw).to(DEV).eval()
        mf = FMMAttentionFast(hidden_dim=HID, num_heads=HEADS, max_token_grid=64, **kw).to(DEV).eval()
        miss, unexp = mf.load_state_dict(m.state_dict(), strict=True)  # identical param set
        x = torch.randn(*shape, device=DEV)
        with torch.no_grad():
            y, _ = m(x, bcs(per))
            yf, _ = mf(x, bcs(per))
        d = (y - yf).abs().max().item()
        ok = torch.allclose(y, yf, atol=1e-4, rtol=1e-4)
        all_ok &= ok
        print(f"  {tag:18} max|Δ|={d:.2e}  {'OK' if ok else 'FAIL'}")
    print(f"  → {'ALL PARITY OK' if all_ok else 'PARITY FAILED'}\n")
    return all_ok


def timeit(fn, backward, warm=3, reps=10):
    for _ in range(warm):
        if backward:
            fn().sum().backward()
        else:
            with torch.no_grad():
                fn()
    if DEV == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        if backward:
            fn().sum().backward()
        else:
            with torch.no_grad():
                fn()
    if DEV == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3


def bench(grid=32, B=16):
    print(f"=== WALL-CLOCK  grid={grid}x{grid}  B={B}  device={DEV} ===")
    x = torch.randn(B, HID, grid, grid, 1, device=DEV)
    models = {
        "full": FullAttention(hidden_dim=HID, num_heads=HEADS).to(DEV),
        "fmm (manual)": FMMAttention(hidden_dim=HID, num_heads=HEADS, max_token_grid=64).to(DEV),
        "fmm_fast (SDPA)": FMMAttentionFast(hidden_dim=HID, num_heads=HEADS, max_token_grid=64).to(DEV),
    }
    for name, m in models.items():
        f = timeit(lambda: m(x, bcs((True, True)))[0], backward=False)
        b = timeit(lambda: m(x.clone().requires_grad_(), bcs((True, True)))[0], backward=True)
        print(f"  {name:16}  fwd {f:8.1f} ms   fwd+bwd {b:8.1f} ms")


if __name__ == "__main__":
    print(f"device={DEV}  hidden={HID} heads={HEADS}\n")
    ok = parity()
    bench(32, 16)
    print("\nNote: on CPU, SDPA uses the math backend (no fusion) → fmm_fast ≈ fmm here.")
    print("The flash/mem-efficient speedup appears only on CUDA (run there later).")
