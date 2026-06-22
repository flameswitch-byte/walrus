from __future__ import annotations

from typing import Tuple, Union, cast

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from the_well.data.datasets import BoundaryCondition
from torch import Tensor

CONV_FUNCS = {
    1: (nn.ConvTranspose1d, F.conv_transpose1d),
    2: (nn.ConvTranspose2d, F.conv_transpose2d),
    3: (nn.ConvTranspose3d, F.conv_transpose3d),
}

# Normal (non-transposed) convs used by the resize-conv anti-checkerboard upsampler.
NORMAL_CONV_FUNCS = {
    1: (nn.Conv1d, F.conv1d),
    2: (nn.Conv2d, F.conv2d),
    3: (nn.Conv3d, F.conv3d),
}

_INTERP_MODE = {1: "linear", 2: "bilinear", 3: "trilinear"}


class VstrideDecoder(nn.Module):
    def __init__(
        self,
        base_kernel_size1d: Tuple[Tuple[int, int], ...] = ((4, 4),),
        base_kernel_size2d: Tuple[Tuple[int, int], ...] = ((8, 4), (8, 4)),
        base_kernel_size3d: Tuple[Tuple[int, int], ...] = ((4, 4), (4, 4), (4, 4)),
        output_dim: int = 3,
        input_dim: int = 768,  #
        inner_dim: int = 192,  # Dimension of the internal convs - base is outer/4
        spatial_dims: int = 2,
        groups: int = 12,
        learned_pad: bool = True,
    ) -> None:
        super().__init__()

        self.learned_pad = learned_pad
        self.input_dim = input_dim
        self.inner_dim = inner_dim
        self.output_dim = output_dim
        self.base_kernel_size = (
            base_kernel_size2d if spatial_dims == 2 else base_kernel_size3d
        )
        self.spatial_dims = spatial_dims
        self.norm_layer1 = nn.GroupNorm

        # First layer
        self.base_kernel1 = tuple(
            [self.base_kernel_size[i][1] for i in range(self.spatial_dims)]
        )
        self.stride1 = self.base_kernel1

        # Second layer
        self.base_kernel2 = tuple(
            [self.base_kernel_size[i][0] for i in range(self.spatial_dims)]
        )
        self.stride2 = self.base_kernel2

        if self.spatial_dims == 1:
            conv: type[nn.ConvTranspose1d | nn.ConvTranspose2d | nn.ConvTranspose3d] = (
                nn.ConvTranspose1d
            )
            self.conv_func = F.conv_transpose1d
        elif self.spatial_dims == 2:
            conv = nn.ConvTranspose2d
            self.conv_func = F.conv_transpose2d
        elif self.spatial_dims == 3:
            conv = nn.ConvTranspose3d
            self.conv_func = F.conv_transpose3d

        self.proj1 = conv(
            input_dim,
            inner_dim,
            kernel_size=self.base_kernel1,  # type: ignore
            bias=False,
        )

        # Normalization layer after the first convolutional layer
        self.norm1 = self.norm_layer1(groups, inner_dim, affine=True)
        self.act1 = nn.GELU()

        self.proj2 = conv(
            inner_dim,
            output_dim,
            kernel_size=self.base_kernel2,  # type: ignore
        )

    def forward(
        self,
        x: Tensor,
        state_labels,
        bcs=None,
        stage_info=None,
        metadata=None,
    ) -> Union[Tensor, Tuple[Tensor, Tuple[int, int]]]:
        embed_kernel = stage_info["random_kernel"]
        debed_kernel = tuple((b, a) for (a, b) in embed_kernel)

        stride1 = tuple([debed_kernel[i][0] for i in range(self.spatial_dims)])
        stride2 = tuple([debed_kernel[i][1] for i in range(self.spatial_dims)])

        if self.learned_pad:
            # learned padding is taken care of in patch jitterer
            padding1, padding2 = 0, 0
        else:
            padding1 = tuple(
                [
                    int(np.ceil((self.stride1[i] - stride) / 2.0))
                    for i, stride in enumerate(stride1)
                ]
            )  # type: ignore
            padding2 = tuple(
                [
                    int(np.ceil((self.stride2[i] - stride) / 2.0))
                    for i, stride in enumerate(stride2)
                ]
            )  # type: ignore

        padding1 = cast(Tuple[int, ...], padding1)  # type: ignore
        padding2 = cast(Tuple[int, ...], padding2)  # type: ignore

        weight1 = self.proj1.weight
        # x is (T, B, C, H, W, D)
        # state_labels is (C_in)
        T = x.shape[0]
        indims = x.ndim
        # Flatten time
        x = rearrange(x, "T B ... -> (T B) ...")  # T B C H W D -> (T B) C H W D
        x = x.squeeze((-2, -1))  # (T B) C H W D -> (T B) C H [W] [D]

        x = self.conv_func(
            x, weight1, bias=self.proj1.bias, stride=stride1, padding=padding1
        )
        x = self.norm1(x)  # Apply normalization
        x = self.act1(x)  # Apply GELU activation

        weight2 = self.proj2.weight
        x = self.conv_func(
            x,
            weight2[:, state_labels],
            bias=self.proj2.bias[state_labels],  # type: ignore
            stride=stride2,
            padding=padding2,
        )

        # Do twice for 3d/1d
        x = rearrange(x, "(T B) ... -> T B ...", T=T)
        if x.ndim < indims:
            x = x.unsqueeze(-1)
        if x.ndim < indims:
            x = x.unsqueeze(-1)

        return x


class AdaptiveDVstrideDecoder(nn.Module):
    def __init__(
        self,
        base_kernel_size1d: Tuple[Tuple[int, int], ...] = ((4, 4),),
        base_kernel_size2d: Tuple[Tuple[int, int], ...] = ((8, 4), (8, 4)),
        base_kernel_size3d: Tuple[Tuple[int, int], ...] = ((4, 4), (4, 4), (4, 4)),
        output_dim: int = 3,
        input_dim: int = 768,  #
        inner_dim: int = 192,  # Dimension of the internal convs - base is outer/4
        spatial_dims: int = 3,
        groups: int = 12,
        learned_pad: bool = True,
        norm_layer: nn.Module = nn.GroupNorm,
        activation: nn.Module = nn.GELU,
        upsample_mode: str = "transpose",
        resize_conv_kernel: int = 3,
    ) -> None:
        super().__init__()

        self.learned_pad = learned_pad
        self.input_dim = input_dim
        self.inner_dim = inner_dim
        self.output_dim = output_dim
        self.base_kernel_size = (
            base_kernel_size2d if spatial_dims == 2 else base_kernel_size3d
        )
        self.spatial_dims = spatial_dims
        if upsample_mode not in ("transpose", "resize_conv"):
            raise ValueError(
                f"upsample_mode must be 'transpose' or 'resize_conv', got {upsample_mode}"
            )
        self.upsample_mode = upsample_mode

        # First layer
        self.base_kernel1 = tuple(
            [self.base_kernel_size[i][1] for i in range(self.spatial_dims)]
        )
        self.stride1 = self.base_kernel1

        # Second layer
        self.base_kernel2 = tuple(
            [self.base_kernel_size[i][0] for i in range(self.spatial_dims)]
        )
        self.stride2 = self.base_kernel2

        # NOTE: register submodules in the SAME order as the original 2-layer decoder
        # (proj1 -> norm1 -> act -> proj2) so the parameter ORDER is unchanged and
        # AdamW optimizer state from pre-existing checkpoints still aligns on resume
        # (optimizer state loads by position, not name). The resize_conv branch is a
        # new code path with no legacy checkpoints, so its order is free.
        if upsample_mode == "transpose":
            conv_class, self.conv_func = CONV_FUNCS[self.spatial_dims]
            self.proj1 = conv_class(
                input_dim,
                inner_dim,
                kernel_size=self.base_kernel1,  # type: ignore
                bias=False,
            )
            self.norm1 = norm_layer(groups, inner_dim, affine=True)
            self.act = activation()
            self.proj2 = conv_class(
                inner_dim,
                output_dim,
                kernel_size=self.base_kernel2,  # type: ignore
            )
        else:
            # Anti-checkerboard upsampler: interpolate then a normal (non-transposed)
            # conv. Handles Walrus's variable per-axis upscale factors (literal
            # PixelShuffle needs a fixed factor). NOT weight-compatible with the
            # transposed-conv decoder -> requires (fine)training.
            conv_class, self.normal_conv_func = NORMAL_CONV_FUNCS[self.spatial_dims]
            # Refinement kernel: small (default 3) — interpolation does the upsampling,
            # so the conv only refines locally and the big base kernel is unnecessary
            # (and ~s^d more costly at full output resolution). `None`/0 falls back to
            # the base (transpose-equivalent) kernel. Sizing always uses base_kernel*.
            rck1 = (
                (resize_conv_kernel,) * spatial_dims
                if resize_conv_kernel
                else self.base_kernel1
            )
            rck2 = (
                (resize_conv_kernel,) * spatial_dims
                if resize_conv_kernel
                else self.base_kernel2
            )
            self.rc1 = conv_class(
                input_dim,
                inner_dim,
                kernel_size=rck1,  # type: ignore
                bias=False,
            )
            self.rc2 = conv_class(
                inner_dim,
                output_dim,
                kernel_size=rck2,  # type: ignore
            )
            self.norm1 = norm_layer(groups, inner_dim, affine=True)
            self.act = activation()

    def _transpose_out_size(self, in_spatial, bcs, kernel, stride):
        """Output size that ``adaptive_conv_transpose`` would produce, per axis.
        Replicated so resize-conv matches the transpose path exactly (the patch
        jitterer's unjitter crop depends on this size). Singleton axes stay 1."""
        nd = self.spatial_dims
        out = []
        for a in range(nd):
            ind = int(in_spatial[a])
            if ind == 1:
                out.append(1)
                continue
            k = int(kernel[a])
            s = int(stride[a])
            periodic = (
                a < len(bcs) and int(bcs[a][0]) == BoundaryCondition["PERIODIC"].value
            )
            if periodic:
                pad_in = (k - s) // s
                pad_out = k - s
                out.append((ind + 2 * pad_in - 1) * s + k - 2 * pad_out)
            else:
                out.append((ind - 1) * s + k)
        return tuple(out)

    def resize_conv(self, x, bcs, weight, bias, stride, size_kernel):
        """Anti-checkerboard upsample: interpolate to the transpose-equivalent size,
        then a same-pad normal conv. Every output pixel is produced once (no
        overlap-add), so no checkerboard. BC-aware padding (circular for periodic
        axes); singleton / inflated axes collapse the kernel and are not upsampled.

        `size_kernel` is the transpose-equivalent (base) kernel used ONLY to compute
        the interpolation target (so unjitter still matches); the actual refinement
        conv uses `weight`'s own (typically small) kernel — decoupled so a cheap 3x3x3
        refinement can replace the costly base-kernel conv."""
        nd = self.spatial_dims
        in_spatial = list(x.shape[-nd:])
        # Mirror adaptive_conv_transpose: pad bcs out to nd (extra axes non-periodic).
        bcs = list(bcs)
        while len(bcs) < nd:
            bcs = bcs + [[2, 2]]
        target = self._transpose_out_size(in_spatial, bcs, size_kernel, stride)
        if tuple(target) != tuple(in_spatial):
            x = F.interpolate(
                x, size=tuple(target), mode=_INTERP_MODE[nd], align_corners=False
            )
        k = list(weight.shape[-nd:])  # actual refinement kernel (small)
        # Collapse the kernel on singleton axes (matches the encoder/transpose path).
        for a in range(nd):
            if x.shape[-nd + a] == 1 and k[a] > 1:
                weight = weight.sum(dim=-(nd - a), keepdim=True)
                k[a] = 1
        # Same-size padding per axis, BC-aware.
        for a in range(nd):
            ka = k[a]
            if ka == 1:
                continue
            left, right = (ka - 1) // 2, ka // 2
            pos = 2 * (nd - 1 - a)
            pad_arg = [0] * (2 * nd)
            pad_arg[pos] = left
            pad_arg[pos + 1] = right
            cur = x.shape[-nd + a]
            periodic = (
                a < len(bcs) and int(bcs[a][0]) == BoundaryCondition["PERIODIC"].value
            )
            if periodic and max(left, right) < cur:
                mode = "circular"
            elif max(left, right) < cur:
                mode = "reflect"
            else:
                mode = "replicate"
            x = F.pad(x, tuple(pad_arg), mode=mode)
        return self.normal_conv_func(x, weight, bias, stride=1, padding=0)

    def adaptive_conv_transpose(self, x, bcs, weight, bias, stride, padding):
        spatial_dims = x.shape[-self.spatial_dims :]
        stride = list(stride)
        padding = list(padding)
        periodic_padding = [0, 0] * (self.spatial_dims)
        padding_out = [0, 0] * (self.spatial_dims)
        if len(bcs) < self.spatial_dims:
            bcs = bcs + [[2, 2]]
        for i, dim in enumerate(spatial_dims[::-1], start=1):
            if dim == 1:
                weight = weight.mean(dim=-i, keepdim=True)
                stride[-i] = 1
                padding[-i] = 0
            k = weight.shape[-i]  # Shape of the kernel in the given dim
            s = stride[-i]
            pad_in = (
                (k - s) // s
            )  # TODO (mm) - This might get messed up for future non-divisible cases
            pad_out = k - s
            # if dist.get_rank() == 0:
            #     print("wtf is going on", i, dim, bcs)
            # NOTE - if dim = 1, padding is already 0, but this is more explicit.
            if dim != 1 and int(bcs[-i][0]) == BoundaryCondition["PERIODIC"].value:
                # if dist.get_rank() == 0:
                #     print(dist.get_rank(), "periodic!", bcs, spatial_dims)
                periodic_padding[2 * (i - 1)] = pad_in
                periodic_padding[2 * (i - 1) + 1] = pad_in
                padding_out[2 * (i - 1)] = -pad_out
                padding_out[2 * (i - 1) + 1] = -pad_out
        # Now pad and perform the convs
        # if dist.get_rank() == 0:
        #     print(dist.get_rank(), "pre per pad", x.shape, bcs)
        x = F.pad(x, pad=tuple(periodic_padding), mode="circular")
        # if dist.get_rank() == 0:
        #     print(dist.get_rank(), "post per pad", x.shape)
        x = self.conv_func(x, weight, bias, tuple(stride), tuple(padding))
        # if dist.get_rank() == 0:
        #     print(dist.get_rank(), "pre out pad", x.shape)
        x = F.pad(x, pad=padding_out, mode="constant")
        # if dist.get_rank() == 0:
        #     print(dist.get_rank(), "post out pad", x.shape)
        return x

    def forward(
        self,
        x: Tensor,
        state_labels,
        bcs,
        stage_info=None,
        metadata=None,
    ) -> Union[Tensor, Tuple[Tensor, Tuple[int, int]]]:
        embed_kernel = stage_info["random_kernel"]
        debed_kernel = tuple((b, a) for (a, b) in embed_kernel)

        stride1 = tuple([debed_kernel[i][0] for i in range(self.spatial_dims)])
        stride2 = tuple([debed_kernel[i][1] for i in range(self.spatial_dims)])

        if self.learned_pad:
            # learned padding is taken care of in patch jitterer
            padding1 = (0,) * self.spatial_dims
            padding2 = (0,) * self.spatial_dims
        else:
            padding1 = tuple(
                [
                    int(np.ceil((self.stride1[i] - stride) / 2.0))
                    for i, stride in enumerate(stride1)
                ]
            )  # type: ignore
            padding2 = tuple(
                [
                    int(np.ceil((self.stride2[i] - stride) / 2.0))
                    for i, stride in enumerate(stride2)
                ]
            )  # type: ignore

        padding1 = cast(Tuple[int, ...], padding1)  # type: ignore
        padding2 = cast(Tuple[int, ...], padding2)  # type: ignore

        # x is (T, B, C, H, W, D)
        # state_labels is (C_in)
        T = x.shape[0]
        # Flatten time
        x = rearrange(x, "T B ... -> (T B) ...")  # T B C H W D -> (T B) C H W D

        if self.upsample_mode == "resize_conv":
            # Normal-conv layout: out channels are dim 0, so sub-select on dim 0.
            # Sizing uses the base (transpose-equivalent) kernel; the conv uses its
            # own (small) refinement kernel.
            x = self.resize_conv(
                x, bcs, self.rc1.weight, self.rc1.bias, stride1, self.base_kernel1
            )
            x = self.act(self.norm1(x))
            x = self.resize_conv(
                x,
                bcs,
                self.rc2.weight[state_labels],
                self.rc2.bias[state_labels],  # type: ignore
                stride2,
                self.base_kernel2,
            )
        else:
            x = self.adaptive_conv_transpose(
                x,
                bcs,
                self.proj1.weight,
                bias=self.proj1.bias,
                stride=stride1,
                padding=padding1,
            )
            x = self.act(self.norm1(x))  # Apply normalization

            x = self.adaptive_conv_transpose(
                x,
                bcs,
                self.proj2.weight[:, state_labels],
                bias=self.proj2.bias[state_labels],  # type: ignore
                stride=stride2,
                padding=padding2,
            )
        # Do twice for 3d/1d
        x = rearrange(x, "(T B) ... -> T B ...", T=T)
        # if dist.get_rank() == 0:
        #     print(metadata.dataset_name, x.shape)

        return x
