"""Full spatial attention + persistent global register tokens.

A drop-in `space_mixing` fork of ``FullAttention`` (same fine-grid math, byte-identical
when no registers are supplied). If ``registers`` (shape ``(b, K, C)``) are passed, the K
learned global tokens are projected through the SAME q/k/v/ff weights (no new attention
params), attend jointly with the N spatial tokens in ONE softmax, and are returned with a
residual update so ``IsotropicModel`` can carry them across blocks (persistent global
memory / conserved-state / regime accumulator).

Design notes (knowledge_base/walrus_general_addons_pushforward_registers.md):
  - Registers get NO rotary embedding -> they are position-free global slots.
  - They are excluded from the patch-jitter roll (carried as a separate tensor).
  - They are dropped before the decoder (auxiliary; never map back to fields).
  - Enabled via ``model.num_register_tokens > 0``; ``registers=None`` reproduces
    ``FullAttention`` exactly.

Only new parameters in the whole feature live in ``IsotropicModel.register_tokens``
(``K x hidden_dim``); this module adds none.
"""

import torch
import torch.nn.functional as F
from einops import rearrange

from ..shared_utils.lr_rope_temporary import apply_rotary_emb
from .full_attention import FullAttention


class RegisterAttention(FullAttention):
    def forward(self, x, bcs, registers=None, return_att=False):
        # input is (b=(t b)) x c x h x w x d
        B, C, H, W, D = x.shape

        input = x.clone()
        x = self.norm1(x)

        fused_ff_qkv = rearrange(x, "b c h w d -> b h w d c")
        ff, q, k, v = self.fused_ff_qkv(fused_ff_qkv).split(self.fused_dims, dim=-1)

        q, k, v = map(
            lambda t: rearrange(t, "b h w d (he c) -> b he h w d c", he=self.num_heads),
            (q, k, v),
        )
        q = self.q_norm(q)
        k = self.k_norm(k)
        pos_emb = self.rotary_emb.get_axial_freqs(H, W, D)
        q, k = map(lambda t: apply_rotary_emb(pos_emb, t), (q, k))
        q, k, v = map(
            lambda t: rearrange(t, "b he h w d c -> b he (h w d) c"), (q, k, v)
        )

        N = H * W * D
        if registers is not None:
            # Reuse norm1 (RMSGroupNorm over channels): reshape the K registers as a
            # spatial axis so they normalize together (same treatment as fine tokens).
            reg_n = rearrange(
                self.norm1(rearrange(registers, "b k c -> b c k 1 1")),
                "b c k 1 1 -> b k c",
            )
            rff, rq, rk, rv = self.fused_ff_qkv(reg_n).split(self.fused_dims, dim=-1)
            rq, rk, rv = map(
                lambda t: rearrange(t, "b k (he c) -> b he k c", he=self.num_heads),
                (rq, rk, rv),
            )
            rq = self.q_norm(rq)
            rk = self.k_norm(rk)  # deliberately NO rotary embedding -> position-free
            q = torch.cat([q, rq], dim=2)
            k = torch.cat([k, rk], dim=2)
            v = torch.cat([v, rv], dim=2)

        att = F.scaled_dot_product_attention(q, k, v)

        reg_att = None
        if registers is not None:
            att, reg_att = att[:, :, :N], att[:, :, N:]

        att = rearrange(att, "b he (h w d) c -> b h w d (he c)", h=H, w=W)
        att_out = self.attn_out(att)
        x = self.drop_path(att_out + self.ff_out(self.activation(ff)))
        x = rearrange(x, "b h w d c -> b c h w d") + input

        if registers is not None:
            reg_att = rearrange(reg_att, "b he k c -> b k (he c)")
            reg_out = self.attn_out(reg_att)
            registers = registers + self.drop_path(
                reg_out + self.ff_out(self.activation(rff))
            )
            return x, registers, []

        return x, []
