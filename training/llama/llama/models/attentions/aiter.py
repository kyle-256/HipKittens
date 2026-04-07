import torch
import torch.nn as nn
import aiter

_VALID_LAYOUTS = ('bshd', 'sbhd')


class AITERSelfAttention(nn.Module):
    """aiter flash attention wrapper.  Accepts both bshd and sbhd layouts.

    aiter itself requires bshd-contiguous tensors, so the sbhd path transposes
    S/B before and after the kernel call — this is an aiter limitation.
    """

    def __init__(
        self,
        causal=False,
        softmax_scale=None,
        attention_dropout=0.0,
        deterministic=False,
        input_layout='bshd',
    ):
        super().__init__()
        assert input_layout in _VALID_LAYOUTS
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop = nn.Dropout(attention_dropout)
        self.deterministic = deterministic
        self.input_layout = input_layout

    def forward(self, qkv, causal=None, key_padding_mask=None):
        assert qkv.dtype in [torch.float16, torch.bfloat16]
        assert qkv.is_cuda
        causal = self.causal if causal is None else causal
        q, k, v = qkv.unbind(dim=2)

        if self.input_layout == 'sbhd':
            q = q.transpose(0, 1).contiguous()
            k = k.transpose(0, 1).contiguous()
            v = v.transpose(0, 1).contiguous()

        out, _ = aiter.flash_attn_func(
            q, k, v,
            dropout_p=self.drop.p,
            causal=self.causal,
            return_lse=True,
            deterministic=self.deterministic,
        )
        out = out.to(q.dtype).contiguous()

        if self.input_layout == 'sbhd':
            out = out.transpose(0, 1)

        return out


class AITERCrossAttention(nn.Module):
    """aiter flash cross-attention wrapper.  Accepts both bshd and sbhd layouts."""

    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0,
                 deterministic=False, input_layout='bshd'):
        super().__init__()
        assert input_layout in _VALID_LAYOUTS
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop = nn.Dropout(attention_dropout)
        self.deterministic = deterministic
        self.input_layout = input_layout

    def forward(self, q, kv, causal=None, key_padding_mask=None):
        causal = self.causal if causal is None else causal
        k, v = kv.unbind(dim=2)

        if self.input_layout == 'sbhd':
            q = q.transpose(0, 1).contiguous()
            k = k.transpose(0, 1).contiguous()
            v = v.transpose(0, 1).contiguous()

        out, _ = aiter.flash_attn_func(
            q, k, v,
            dropout_p=self.drop.p,
            causal=causal,
            return_lse=True,
            deterministic=self.deterministic,
        )
        out = out.to(q.dtype).contiguous()

        if self.input_layout == 'sbhd':
            out = out.transpose(0, 1)

        return out
