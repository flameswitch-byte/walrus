from functools import partial
from typing import Callable

import torch.nn as nn
from einops import rearrange
from torch.utils.checkpoint import checkpoint

from walrus.models.shared_utils.normalization import RMSGroupNorm


class SpaceTimeSplitBlock(nn.Module):
    """
    Operates similar to standard MHSA -> Inverted Bottleneck but with ConvNext
    block replacing linear part.

    Note: HYDRA is instantiating space_block and time_block as functools.partial functions
    so parameters that aren't shared are pre-set based on the config files.
    """

    def __init__(
        self,
        space_mixing,
        time_mixing,
        channel_mixing,
        hidden_dim=768,
        drop_path=0.0,
        gradient_checkpointing=False,
        causal_in_time=False,
        norm_layer: Callable = RMSGroupNorm,
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.space_mixing = space_mixing(
            hidden_dim=hidden_dim,
            drop_path=drop_path,
            gradient_checkpointing=gradient_checkpointing,
            norm_layer=norm_layer,
        )
        self.time_mixing = time_mixing(
            hidden_dim=hidden_dim,
            drop_path=drop_path,
            gradient_checkpointing=gradient_checkpointing,
            causal_in_time=causal_in_time,
            norm_layer=norm_layer,
        )
        self.channel_mixing = channel_mixing(hidden_dim=hidden_dim)
        self.causal_in_time = causal_in_time

    def make_rope_learnable(self, per_axis=False):
        """
        Make the RoPE learnable in the time mixing module.
        """
        self.time_mixing.make_rope_learnable(per_axis)
        self.space_mixing.make_rope_learnable(per_axis)

    def forward(self, x, bcs, coarse=None, coarse_ratio=None, registers=None, return_att=False):
        # input is t x b x c x h x w
        T, B, C, H, W, D = x.shape
        # Time attention runs on the FINE grid only (coarse is per-frame spatial-global
        # and passes through untouched). See Design B, knowledge_base doc.
        if self.gradient_checkpointing:
            # kwargs seem to need to be passed explicitly
            wrapped_temporal = partial(self.time_mixing, return_att=return_att)
            x, t_att = checkpoint(wrapped_temporal, x, use_reentrant=False)
        else:
            x, t_att = self.time_mixing(x, return_att=return_att)  # Residual in block
        # Temporal handles the rearrange so still is t x b x c x h x w
        x = rearrange(x, "t b c h w d -> (t b) c h w d")

        two_grid = coarse is not None
        if two_grid:
            Tc = coarse.shape[0]
            coarse = rearrange(coarse, "t b c h w d -> (t b) c h w d")

        # Persistent global register tokens (flat-path only; runs alongside FullAttention).
        # Time attention above ran fine-only, so registers pass through it untouched; here
        # they join the spatial attention. Flatten time into batch to match space_mixing.
        has_reg = registers is not None
        if has_reg:
            registers = rearrange(registers, "t b k c -> (t b) k c")

        if self.gradient_checkpointing:
            # kwargs seem to need to be passed explicitly
            if two_grid:
                wrapped_spatial = partial(
                    self.space_mixing,
                    coarse=coarse,
                    coarse_ratio=coarse_ratio,
                    return_att=return_att,
                )
                x, coarse, s_att = checkpoint(
                    wrapped_spatial, x, bcs, use_reentrant=False
                )
            elif has_reg:
                wrapped_spatial = partial(
                    self.space_mixing, registers=registers, return_att=return_att
                )
                x, registers, s_att = checkpoint(
                    wrapped_spatial, x, bcs, use_reentrant=False
                )
            else:
                wrapped_spatial = partial(self.space_mixing, return_att=return_att)
                x, s_att = checkpoint(wrapped_spatial, x, bcs, use_reentrant=False)
        else:
            if two_grid:
                x, coarse, s_att = self.space_mixing(
                    x, bcs, coarse=coarse, coarse_ratio=coarse_ratio,
                    return_att=return_att,
                )
            elif has_reg:
                x, registers, s_att = self.space_mixing(
                    x, bcs, registers=registers, return_att=return_att
                )
            else:
                x, s_att = self.space_mixing(
                    x, bcs, return_att=return_att
                )  # Convnext has the residual in the block
        x = rearrange(x, "(t b) c h w d -> t b c h w d", t=T)
        if two_grid:
            coarse = rearrange(coarse, "(t b) c h w d -> t b c h w d", t=Tc)
        if has_reg:
            registers = rearrange(registers, "(t b) k c -> t b k c", t=T)
        # MLP input is channels last - #TODO redefine as 1x1 conv to avoid reshape
        x = self.channel_mixing(
            x
        )  # Currently set to identity, but needs to be reshaped generally

        att = t_att + s_att if return_att else []
        if two_grid:
            return x, coarse, att
        if has_reg:
            return x, registers, att
        return x, att
