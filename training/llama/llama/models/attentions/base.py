import math
from functools import partial

import torch
import torch.nn as nn
from einops import rearrange, repeat

_VALID_LAYOUTS = ('bshd', 'sbhd')


class SelfAttention(nn.Module):
    """Reference scaled dot product attention.

    Supports bshd and sbhd layouts via einsum index permutation (zero copy).
    """

    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0,
                 input_layout='bshd'):
        super().__init__()
        assert input_layout in _VALID_LAYOUTS
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop = nn.Dropout(attention_dropout)
        self.input_layout = input_layout

    def forward(self, qkv, causal=None, key_padding_mask=None):
        if self.input_layout == 'sbhd':
            seqlen, batch_size = qkv.shape[0], qkv.shape[1]
        else:
            batch_size, seqlen = qkv.shape[0], qkv.shape[1]

        causal = self.causal if causal is None else causal
        q, k, v = qkv.unbind(dim=2)
        softmax_scale = self.softmax_scale or 1.0 / math.sqrt(q.shape[-1])

        if self.input_layout == 'sbhd':
            scores = torch.einsum("tbhd,sbhd->bhts", q, k * softmax_scale)
        else:
            scores = torch.einsum("bthd,bshd->bhts", q, k * softmax_scale)

        if key_padding_mask is not None:
            padding_mask = torch.full(
                (batch_size, seqlen), -10000.0, dtype=scores.dtype, device=scores.device
            )
            padding_mask.masked_fill_(key_padding_mask, 0.0)
            scores = scores + rearrange(padding_mask, "b s -> b 1 1 s")
        if causal:
            causal_mask = torch.triu(
                torch.full((seqlen, seqlen), -10000.0, device=scores.device), 1
            )
            scores = scores + causal_mask.to(dtype=scores.dtype)
        attention = torch.softmax(scores, dim=-1, dtype=v.dtype)
        attention_drop = self.drop(attention)

        if self.input_layout == 'sbhd':
            output = torch.einsum("bhts,sbhd->tbhd", attention_drop, v)
        else:
            output = torch.einsum("bhts,bshd->bthd", attention_drop, v)
        return output


class CrossAttention(nn.Module):
    """Reference scaled dot product cross-attention.  Supports bshd / sbhd."""

    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0,
                 input_layout='bshd'):
        super().__init__()
        assert input_layout in _VALID_LAYOUTS
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop = nn.Dropout(attention_dropout)
        self.input_layout = input_layout

    def forward(self, q, kv, causal=None, key_padding_mask=None):
        if self.input_layout == 'sbhd':
            seqlen_q, batch_size = q.shape[0], q.shape[1]
            seqlen_k = kv.shape[0]
        else:
            batch_size, seqlen_q = q.shape[0], q.shape[1]
            seqlen_k = kv.shape[1]

        causal = self.causal if causal is None else causal
        if kv.shape[3] != q.shape[2]:
            kv = repeat(kv, "... hkv d -> ... (hkv g) d", g=q.shape[2] // kv.shape[3])
        k, v = kv.unbind(dim=2)
        softmax_scale = self.softmax_scale or 1.0 / math.sqrt(q.shape[-1])

        if self.input_layout == 'sbhd':
            scores = torch.einsum("tbhd,sbhd->bhts", q, k * softmax_scale)
        else:
            scores = torch.einsum("bthd,bshd->bhts", q, k * softmax_scale)

        if key_padding_mask is not None:
            padding_mask = torch.full(
                (batch_size, seqlen_k), -10000.0, dtype=scores.dtype, device=scores.device
            )
            padding_mask.masked_fill_(key_padding_mask, 0.0)
            scores = scores + rearrange(padding_mask, "b s -> b 1 1 s")
        if causal:
            row_idx = rearrange(
                torch.arange(seqlen_q, device=q.device, dtype=torch.long), "s -> s 1"
            )
            col_idx = torch.arange(seqlen_k, device=kv.device, dtype=torch.long)
            sk = (
                seqlen_k
                if key_padding_mask is None
                else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
            )
            causal_mask = col_idx > row_idx + sk - seqlen_q
            scores = scores.masked_fill(causal_mask, -10000.0)
        attention = torch.softmax(scores, dim=-1, dtype=v.dtype)
        attention_drop = self.drop(attention)

        if self.input_layout == 'sbhd':
            output = torch.einsum("bhts,sbhd->tbhd", attention_drop, v)
        else:
            output = torch.einsum("bhts,bshd->bthd", attention_drop, v)
        return output
