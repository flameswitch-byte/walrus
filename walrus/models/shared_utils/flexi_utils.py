import itertools
import random
from typing import Literal, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from functorch import vmap
from torch import Tensor


def choose_kernel_size_random(kernel_scales_seq):
    """
    Choose a kernel size from kernel_scales_seq with uniform probability
    """
    return random.choices(kernel_scales_seq)[0]


def create_patch_dict(kernel_scales_seq: Tuple[Tuple[int, int], ...]):
    patch_sizes = [p[0] * p[1] for p in kernel_scales_seq]
    # Generate a dictionary mapping patch sizes to partitions
    return dict(zip(patch_sizes, kernel_scales_seq))


def generate_patch_combinations(kernel_scales_seq, spatial_dims):
    """
    Generate all possible patch combinations for a given spatial dimension
    """
    patch_sizes = [p[0] * p[1] for p in kernel_scales_seq]
    if spatial_dims == 1:
        patch_combinations = [
            [
                p,
            ]
            for p in patch_sizes
        ]
    elif spatial_dims == 2:
        patch_combinations = list(itertools.product(patch_sizes, repeat=2))
    elif spatial_dims == 3:
        patch_combinations = list(itertools.product(patch_sizes, repeat=3))
    else:
        raise ValueError("Spatial dimension must be 1, 2 or 3")
    return patch_combinations


def generate_two_conv_combinations(kernel_scales_seq, spatial_dims):
    """
    Generate all possible two layer combinations for a given spatial dimension
    """
    patch_to_partition = create_patch_dict(kernel_scales_seq)
    patch_combinations = generate_patch_combinations(kernel_scales_seq, spatial_dims)
    kernel_scales_seq1 = []
    kernel_scales_seq2 = []
    for patches in patch_combinations:
        temp = []
        temp1 = []
        for p in patches:
            temp.append(patch_to_partition[p][0])
            temp1.append(patch_to_partition[p][1])
        kernel_scales_seq1.append(temp)
        kernel_scales_seq2.append(temp1)
    kernel_scales_seq1 = tuple(set(tuple(tuple(k) for k in kernel_scales_seq1)))
    kernel_scales_seq2 = tuple(set(tuple(tuple(k) for k in kernel_scales_seq2)))

    return kernel_scales_seq1, kernel_scales_seq2


# ``patch -> (stride1, stride2)`` with stride1*stride2 == patch. base_kernel must be >=
# each stride or the strided conv skips pixels. ONLY EVEN strides: odd strides (e.g. 3)
# don't round-trip through the vstride jitterer/decoder (reconstruction comes out short),
# so the soft-target chooser must pick from even-stride patches only -- it degrades to the
# nearest such patch rather than hitting an unsafe one.
_PATCH_DICT = {
    0: (1, 1),
    1: (1, 1),
    2: (2, 1),  # even strides only; 2=(2,1) round-trips (verified on 128-axis @ 64 tokens)
    4: (2, 2),
    8: (4, 2),
    12: (6, 2),
    16: (4, 4),
    24: (6, 4),
    32: (8, 4),
}


def choose_kernel_size_deterministic(
    x_shape: Tuple[int, ...],
    per_axis_tokens: int = None,
    base_kernel: Tuple[Tuple[int, ...], Tuple[int, ...]] = None,
) -> Tuple[Tuple[int, int], ...]:
    """
    Choose the per-axis ``(stride1, stride2)`` tokenizer patch from the image size.

    ``per_axis_tokens``:
      * ``None`` (default) -> LEGACY EXACT behavior, unchanged: 32 tokens/axis for 1D/2D
        and 1-2-axis 3D, 16/axis for true 3D volumes; requires ``axis % target == 0`` and
        ``axis//target`` to be a supported patch (asserts otherwise). The flat baseline
        relies on this path being byte-identical.
      * an int -> SOFT target (resolution lever, robust): for each axis pick the patch
        that (a) divides the axis, (b) has strides <= ``base_kernel`` (no gappy conv), and
        (c) gives a token count CLOSEST to the target. Never crashes (patch 1 always
        valid) and degrades gracefully on axes that can't hit the target exactly -- the
        model handles variable per-axis token counts natively (vstride/FlexiViT).

    ``base_kernel`` = ``(base_kernel1_per_axis, base_kernel2_per_axis)`` (the encoder's
    layer-1/2 kernels). Used only on the soft-target path to exclude patches whose stride
    exceeds the kernel. If omitted, no kernel filtering is applied.
    """
    patch_dict = _PATCH_DICT
    nd = len(x_shape)
    if nd not in (1, 2, 3):
        raise ValueError("Image size must be 1, 2 or 3 dimensions")

    # default (legacy) token target
    if nd == 3:
        H, W, D = x_shape[:3]
        non_singleton_D = int(H != 1) + int(W != 1) + int(D != 1)
        default_target = 512 // 16 if non_singleton_D <= 2 else 256 // 16
    else:
        default_target = 512 // 16
    strict = per_axis_tokens is None
    target = default_target if strict else int(per_axis_tokens)

    def _kbound(axis_idx: int) -> Tuple[int, int]:
        if base_kernel is None:
            return (1 << 30, 1 << 30)
        k1, k2 = base_kernel
        b1 = k1[axis_idx] if axis_idx < len(k1) else (1 << 30)
        b2 = k2[axis_idx] if axis_idx < len(k2) else (1 << 30)
        return (int(b1), int(b2))

    def _pick(axis_idx: int, axis_len: int) -> Tuple[int, int]:
        if axis_len == 1:
            return patch_dict[0]
        if strict:
            assert axis_len % target == 0, (
                f"axis size {axis_len} not divisible by per_axis_tokens {target}"
            )
            p = axis_len // target
            assert p in patch_dict, (
                f"patch {p} (= {axis_len}//{target}) not in patch_dict "
                f"{sorted(patch_dict)}; set per_axis_tokens (soft target) or extend it"
            )
            return patch_dict[p]
        # soft target: patches that divide the axis and fit under the kernel
        b1, b2 = _kbound(axis_idx)
        cands = [
            p
            for p in patch_dict
            if p >= 1
            and axis_len % p == 0
            and patch_dict[p][0] <= b1
            and patch_dict[p][1] <= b2
        ]
        if not cands:  # patch 1 = (1,1) always fits, but guard anyway
            cands = [1]
        # closest token count to target; tie -> smaller patch (finer)
        best = min(cands, key=lambda p: (abs(axis_len // p - target), p))
        return patch_dict[best]

    return tuple(_pick(i, s) for i, s in enumerate(x_shape[:nd]))


InterpolationType = Literal[
    "nearest", "linear", "bilinear", "bicubic", "trilinear", "area", "nearest-exact"
]


def _cache_pinvs(
    kernel_scales_seq: Tuple[Tuple[int, int], ...],
    interpolation: InterpolationType,
    antialias: bool,
    base_kernel_size: Tuple[int, int],
    spatial_dims: int = 2,
) -> dict:
    """
    Calculate and cache pseudo-inverses of resize matrices for all possible kernels
    """

    pinvs = {}
    for ps in kernel_scales_seq:
        pinvs[ps] = _calculate_pinv(base_kernel_size, ps, interpolation, antialias)
    return pinvs


def _resize(
    x: Tensor,
    shape: Tuple[int, int],
    interpolation: InterpolationType,
    antialias: bool,
    spatial_dims: int = 2,
) -> Tensor:
    """
    Resize tensor x to shape using interpolation
    """

    x_resized = F.interpolate(
        x[None, None, ...],
        shape,
        mode=interpolation,
        antialias=antialias,
    )
    return x_resized[0, 0, ...]


def _calculate_pinv(
    old_shape: Tuple[int, int],
    new_shape: Tuple[int, int],
    interpolation: InterpolationType,
    antialias: bool,
) -> Tensor:
    """
    Calculate pseudo-inverse of resize matrix from old_shape to new_shape
    """

    mat = []
    for i in range(np.prod(old_shape)):
        basis_vec = torch.zeros(old_shape)
        basis_vec[np.unravel_index(i, old_shape)] = 1.0
        mat.append(_resize(basis_vec, new_shape, interpolation, antialias).reshape(-1))
    resize_matrix = torch.stack(mat)
    return torch.linalg.pinv(resize_matrix)


def resize_patch_embed(
    patch_embed: Tensor,
    base_kernel_size: Tuple[int, ...],
    new_patch_size: Tuple[int, ...],
    pinvs: dict,
    spatial_dims: int = 2,
):
    """Resize patch_embed to target resolution via pseudo-inverse resizing"""
    # Return original kernel if no resize is necessary
    if base_kernel_size == new_patch_size:
        return patch_embed

    pinv = pinvs[new_patch_size]
    pinv = pinv.to(patch_embed.device)

    def resample_patch_embed(patch_embed: Tensor):
        resampled_kernel = pinv @ patch_embed.reshape(-1)
        if spatial_dims == 1:
            (h,) = new_patch_size
            return rearrange(resampled_kernel, "(h) -> h", h=h)
        elif spatial_dims == 2:
            h, w = new_patch_size
            return rearrange(resampled_kernel, "(h w) -> h w", h=h, w=w)
        else:
            h, w, d = new_patch_size
            return rearrange(resampled_kernel, "(h w d) -> h w d", h=h, w=w, d=d)

    v_resample_patch_embed = vmap(vmap(resample_patch_embed, 0, 0), 1, 1)

    return v_resample_patch_embed(patch_embed)
