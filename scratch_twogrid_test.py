import math

import torch
from the_well.data.datasets import BoundaryCondition

from walrus.models.spatial_blocks.cross_scale_attention_twogrid import (
    TwoGridCrossScaleAttention,
)
from walrus.models.shared_utils.flexi_utils import choose_kernel_size_deterministic

PER = BoundaryCondition["PERIODIC"].value


def make_bcs(nd, periodic):
    code = PER if periodic else 0
    return [[[code, code] for _ in range(nd)]]


def run(tag, H, W, Dg, ratio, periodic, cwin, coupling, pwin=3, fwin=3, moments=True):
    hidden, heads, B = 128, 8, 2
    blk = TwoGridCrossScaleAttention(
        hidden_dim=hidden, num_heads=heads, fine_window=fwin, coarse_window=cwin,
        prolong_window=pwin, coarse_coupling=coupling, restrict_moments=moments,
        spatial_dims=3,
    ).train()
    x = torch.randn(B, hidden, H, W, Dg, requires_grad=True)
    cs = lambda s: (max(1, math.ceil(s / ratio)) if s > 1 else 1)
    Hc, Wc, Dc = cs(H), cs(W), cs(Dg)
    coarse = torch.randn(B, hidden, Hc, Wc, Dc, requires_grad=True)
    nd = sum(int(s > 1) for s in (H, W, Dg))
    out, cout, _ = blk(x, make_bcs(nd, periodic), coarse=coarse, coarse_ratio=ratio)
    loss = out.float().pow(2).mean() + cout.float().pow(2).mean()
    loss.backward()
    assert out.shape == x.shape and cout.shape == coarse.shape
    assert x.grad.abs().mean().item() > 0 and coarse.grad.abs().mean().item() > 0
    print(
        f"[OK] {tag:9s} fine{tuple(x.shape[2:])} coarse{(Hc,Wc,Dc)} "
        f"cwin={str(cwin):6s} pwin={pwin} coup={coupling:13s} fwin={fwin} | loss {loss.item():.4f}"
    )


print("=== coarse_window (coarse self-attn) x coarse_coupling ===")
for cwin in ("global", 1, 3):
    for coupling in ("bidirectional", "one_way"):
        run("2D", 32, 32, 1, 4, True, cwin, coupling)
print("=== prolong_window: injection (1) vs interp (2,3) ===")
for pwin in (1, 2, 3):
    run("2D pw", 32, 32, 1, 4, True, "global", "bidirectional", pwin=pwin)
print("=== restrict_moments on/off (and odd-size masked stats) ===")
run("2D mom1", 32, 32, 1, 4, True, "global", "bidirectional", moments=True)
run("2D mom0", 32, 32, 1, 4, True, "global", "bidirectional", moments=False)
run("2D modd", 30, 30, 1, 4, True, 3, "bidirectional", moments=True)  # ragged children
run("3D mom ", 16, 16, 16, 4, True, "global", "bidirectional", moments=True)
print("=== removed aliases now REJECTED ===")
for coupling in ("bidirectional_local", "bidirectional_global"):
    try:
        run("2D rej", 32, 32, 1, 4, True, "global", coupling)
        raise SystemExit(f"FAIL: {coupling} should have raised")
    except ValueError:
        print(f"[OK] {coupling!r} correctly rejected")
print("=== edge cases ===")
run("2D wall", 32, 32, 1, 4, False, 3, "bidirectional")
run("2D odd ", 30, 30, 1, 4, True, 3, "bidirectional")      # ceil coarse + ragged children
run("2D r2  ", 32, 32, 1, 2, True, "global", "bidirectional")
run("2D f5  ", 32, 32, 1, 4, True, "global", "bidirectional", fwin=5)
run("3D glob", 16, 16, 16, 4, True, "global", "bidirectional")
run("3D loc ", 16, 16, 16, 4, True, 3, "bidirectional")

print("=== per_axis_tokens lever ===")
for pat in (None, 32, 64, 128):
    try:
        ks = choose_kernel_size_deterministic((256, 256, 1), per_axis_tokens=pat)
        s = ks[0][0] * ks[0][1]
        print(f"  pat={str(pat):4s} -> strides {ks} -> {256//s} fine tok/axis (256 input)")
    except Exception as e:
        print(f"  pat={str(pat):4s} -> {type(e).__name__}: {e}")

print("all good")
