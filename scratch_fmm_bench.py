"""Wall-clock + FLOP benchmark: FMMAttention vs FullAttention across token-grid sizes.

Answers "does FMM save compute at N tokens?" — separately for (a) attention FLOPs (where
FMM is O(N·K) vs O(N²)) and (b) measured wall-clock (where FMM pays gather/loop overhead).
Run:  python scratch_fmm_bench.py
"""

import time

import torch

from walrus.models.spatial_blocks.full_attention import FullAttention
from walrus.models.spatial_blocks.fmm_attention import FMMAttention

torch.set_num_threads(8)
torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

HID, HEADS, Bsz = 256, 8, 16      # medium dims; B = T·B_orig
GRIDS = [16, 32, 48, 64]
WARMUP, REPS = 3, 15
P = 1
BCS = [[[P, P]]]  # periodic 2D (bcs[0] = per-dim [lo,hi])


def bcs_2d():
    return [[[P, P], [P, P]]]


def make(kind):
    if kind == "full":
        return FullAttention(hidden_dim=HID, num_heads=HEADS).to(DEV)
    # "fmm" now = the upgraded Path-C fused block (leaf near + coarse rank-1 far), the real
    # candidate. Set pool_rank=4 to measure the Kang far-rank cost.
    from walrus.models.spatial_blocks.fmm_attention_fused import FMMAttentionFused
    return FMMAttentionFused(hidden_dim=HID, num_heads=HEADS, pool_base=2, far_inner=2,
                             far_outer=3, learned_pool=True, leaf_near=True, pool_rank=1,
                             max_token_grid=64).to(DEV)


def sync():
    if DEV == "cuda":
        torch.cuda.synchronize()


def time_block(m, x, bcs, backward):
    # warmup
    for _ in range(WARMUP):
        if backward:
            m.zero_grad(set_to_none=True)
            y, _ = m(x, bcs)
            y.sum().backward()
        else:
            with torch.no_grad():
                m(x, bcs)
    sync()
    t0 = time.perf_counter()
    for _ in range(REPS):
        if backward:
            m.zero_grad(set_to_none=True)
            y, _ = m(x, bcs)
            y.sum().backward()
        else:
            with torch.no_grad():
                m(x, bcs)
    sync()
    return (time.perf_counter() - t0) / REPS * 1e3  # ms/iter


def flop_ratio(grid):
    """Attention-only FLOP ratio FMM/full ≈ (keys per query) / N."""
    N = grid * grid
    fmm = make("fmm")
    n_far = len(fmm._levels((grid, grid, 1)))
    near = (2 * 3 + 1) ** 2                 # 49 offsets (2D)
    keys = near * (1 + n_far)               # near + one 49-wide block per ring
    return keys / N, n_far, keys


if __name__ == "__main__":
    print(f"device={DEV}  hidden={HID} heads={HEADS} B={Bsz}  reps={REPS}")
    full_p = sum(p.numel() for p in make("full").parameters()) / 1e3
    fmm_p = sum(p.numel() for p in make("fmm").parameters()) / 1e3
    print(f"params/block: full {full_p:.0f}K   fmm {fmm_p:.0f}K\n")
    hdr = f"{'grid':>5} {'N':>6} {'rings':>5} {'keys/q':>7} {'FLOP FMM/full':>13} " \
          f"{'full fwd':>9} {'fmm fwd':>9} {'fwd x':>6} " \
          f"{'full f+b':>9} {'fmm f+b':>9} {'f+b x':>6}"
    print(hdr)
    print("-" * len(hdr))
    for g in GRIDS:
        x = torch.randn(Bsz, HID, g, g, 1, device=DEV)
        ratio, nfar, keys = flop_ratio(g)
        mf, mm = make("full"), make("fmm")
        ff = time_block(mf, x, bcs_2d(), backward=False)
        fm = time_block(mm, x, bcs_2d(), backward=False)
        bf = time_block(mf, x.clone().requires_grad_(), bcs_2d(), backward=True)
        bm = time_block(mm, x.clone().requires_grad_(), bcs_2d(), backward=True)
        print(f"{g:>5} {g*g:>6} {nfar:>5} {keys:>7} {ratio:>12.3f}x "
              f"{ff:>8.1f}m {fm:>8.1f}m {ff/fm:>5.2f}x "
              f"{bf:>8.1f}m {bm:>8.1f}m {bf/bm:>5.2f}x")
    print("\nFLOP ratio < 1  → FMM does fewer attention FLOPs.")
    print("fwd/f+b x > 1   → FMM is FASTER wall-clock; < 1 → full is faster (overhead).")
