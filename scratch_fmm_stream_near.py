"""Test option 2's mechanism for the NEAR path: streaming online-softmax (no K-fold stack).

A fused FlashAttention/NATTEN kernel avoids materializing the (B,he,N,K,c) neighbor stack by
computing the softmax in a streaming pass over the K offsets. No such kernel is installed here
(and no GPU), so we run the SAME algorithm in pure PyTorch to check:
  (1) correctness  — equals the gather-based near attention,
  (2) peak memory  — O(N·c) instead of O(N·K·c)  ← the thing that makes 3D feasible,
  (3) CPU time     — the honest tradeoff (no kernel fusion → loop overhead).

  A (gather):    _neighbors → (…,K,c) stack → einsum score → softmax → einsum value
  B (streaming): loop offsets, roll → elementwise dot → online-softmax accumulate (no stack)
Run:  python scratch_fmm_stream_near.py
"""

import itertools
import time

import torch

torch.set_num_threads(8)
torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
from walrus.models.spatial_blocks.fmm_attention import FMMAttention

HID, HEADS = 256, 8
C = HID // HEADS
WARMUP, REPS = 3, 15


def mb(*ts):
    return sum(t.numel() * t.element_size() for t in ts) / 1e6


def gather_near(m, qf, kf, vf, r, periodic, sizes):
    kf_n, mask = m._neighbors(kf, r, periodic)          # (B,he,H,W,D,K,c)  ← the big stack
    vf_n, _ = m._neighbors(vf, r, periodic)
    s = torch.einsum("bhxyzc,bhxyzkc->bhxyzk", qf, kf_n) * (C ** -0.5)
    s = s.masked_fill(~mask.unsqueeze(0).unsqueeze(0), torch.finfo(s.dtype).min)
    w = torch.softmax(s, dim=-1)
    out = torch.einsum("bhxyzk,bhxyzkc->bhxyzc", w, vf_n)
    return out, mb(kf_n, vf_n)


def stream_near(qf, kf, vf, r, periodic, sizes):
    """Online-softmax over the window — never builds the (…,K,c) stack."""
    B, he, H, W, D, c = qf.shape
    dev = qf.device
    scale = c ** -0.5
    NEG = -1e9
    rads = [r if s > 1 else 0 for s in sizes]
    m = torch.full((B, he, H, W, D), NEG, device=dev)          # running max
    l = torch.zeros((B, he, H, W, D), device=dev)              # running denom
    acc = torch.zeros_like(qf)                                  # running weighted value (…,c)
    peak = mb(m, l, acc)
    for off in itertools.product(*[range(-x, x + 1) for x in rads]):
        rk, rv = kf, vf
        valid = torch.ones(sizes, device=dev)
        for a, o in enumerate(off):
            if o != 0:
                rk = torch.roll(rk, shifts=-o, dims=2 + a)
                rv = torch.roll(rv, shifts=-o, dims=2 + a)
                if not periodic[a]:
                    pos = torch.arange(sizes[a], device=dev) + o
                    ok = (pos >= 0) & (pos < sizes[a])
                    shape = [1, 1, 1]; shape[a] = sizes[a]
                    valid = valid * ok.view(shape).float()
        sc = (qf * rk).sum(-1) * scale                         # (B,he,H,W,D) — scalar per token
        sc = torch.where(valid.bool()[None, None], sc, torch.full_like(sc, NEG))
        m_new = torch.maximum(m, sc)
        corr = torch.exp(m - m_new)
        p = torch.exp(sc - m_new)
        l = l * corr + p
        acc = acc * corr.unsqueeze(-1) + p.unsqueeze(-1) * rv
        m = m_new
        peak = max(peak, mb(m, l, acc, rk, rv, sc))            # working set at any instant
    return acc / l.unsqueeze(-1), peak


def timeit(fn):
    for _ in range(WARMUP):
        fn()
    if DEV == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(REPS):
        fn()
    if DEV == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / REPS * 1e3


def run(tag, B, grid, sd, r=3):
    m = FMMAttention(hidden_dim=HID, num_heads=HEADS, spatial_dims=sd, max_token_grid=grid).to(DEV)
    if sd == 3:
        H = W = D = grid
    else:
        H = W = grid; D = 1
    sizes = (H, W, D)
    per = [True] * 3
    qf = torch.randn(B, HEADS, H, W, D, C, device=DEV)
    kf = torch.randn(B, HEADS, H, W, D, C, device=DEV)
    vf = torch.randn(B, HEADS, H, W, D, C, device=DEV)
    with torch.no_grad():
        oA, memA = gather_near(m, qf, kf, vf, r, per, sizes)
        oB, memB = stream_near(qf, kf, vf, r, per, sizes)
        ok = torch.allclose(oA, oB, atol=1e-4)
        tA = timeit(lambda: gather_near(m, qf, kf, vf, r, per, sizes))
        tB = timeit(lambda: stream_near(qf, kf, vf, r, per, sizes))
    K = (2 * r + 1) ** (3 if sd == 3 else 2)
    print(f"{tag:12} K={K:3}  equal={ok!s:5}  gather {tA:8.1f}ms {memA:7.0f}MB   "
          f"stream {tB:8.1f}ms {memB:7.0f}MB   time {tA/tB:5.2f}x  mem {memA/memB:5.1f}x")


if __name__ == "__main__":
    print(f"device={DEV}  hidden={HID} heads={HEADS} head_dim={C}  (near-only)\n")
    run("2D 32 B=16", 16, 32, 2)
    run("2D 64 B=16", 16, 64, 2)
    run("3D 16 B=2", 2, 16, 3)
    print("\nmem>1 → streaming materializes less (never builds the K-fold stack).")
    print("time>1 → streaming faster; <1 → gather faster (streaming pays Python-loop overhead,")
    print("         which a real fused CUDA kernel would remove).")
