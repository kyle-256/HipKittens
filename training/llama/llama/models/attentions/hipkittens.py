import torch
import torch.nn as nn
from torch.autograd import Function

import llama.models.attentions.tk_kernel_fwd as tk_kernel_fwd
import llama.models.attentions.tk_kernel_bkwd as tk_kernel_bkwd
import llama.models.attentions.tk_kernel_bkwd_prep as tk_kernel_bkwd_prep

_VALID_LAYOUTS = ('bshd', 'sbhd')


class HipAttnFunction(Function):
    """
    Supports both BSHD (batch, seq, heads, dim) and SBHD (seq, batch, heads, dim)
    layouts natively — the TK kernels handle the coordinate mapping internally,
    so no transpose or data copy is needed.

    Pass sbhd=True as the last positional arg when calling .apply() for SBHD tensors.
    """

    @staticmethod
    def forward(ctx, q, k, v, sbhd=False):
        if sbhd:
            S, B, H, D = q.shape
            HKV = k.shape[2]
        else:
            B, S, H, D = q.shape
            HKV = k.shape[2]

        dev = q.device
        out_dtype = q.dtype

        assert q.is_cuda and k.is_cuda and v.is_cuda
        assert q.device == k.device == v.device
        assert H % HKV == 0

        q = q.to(torch.bfloat16).contiguous()
        k = k.to(torch.bfloat16).contiguous()
        v = v.to(torch.bfloat16).contiguous()

        if sbhd:
            O = torch.empty((S, B, H, D), dtype=torch.bfloat16, device=dev).contiguous()
        else:
            O = torch.empty((B, S, H, D), dtype=torch.bfloat16, device=dev).contiguous()
        L = torch.empty((B, H, 1, S), dtype=torch.float32, device=dev).contiguous()

        if sbhd:
            tk_kernel_fwd.dispatch_fwd_sbhd(q, k, v, O, L)
        else:
            tk_kernel_fwd.dispatch_fwd(q, k, v, O, L)

        ctx.save_for_backward(q, k, v, O, L)
        ctx.sbhd = sbhd
        return O.to(out_dtype)

    @staticmethod
    def backward(ctx, dO_in):
        q, k, v, O, L = ctx.saved_tensors
        sbhd = ctx.sbhd

        if sbhd:
            S, B, H, D = O.shape
            HKV = k.shape[2]
        else:
            B, S, H, D = O.shape
            HKV = k.shape[2]

        dev = dO_in.device
        assert H % HKV == 0

        dO = dO_in.to(torch.bfloat16).contiguous()

        dQ_in = torch.zeros((B, H, S, D), dtype=torch.bfloat16, device=dev).contiguous()
        if sbhd:
            dQ = torch.empty((S, B, H, D), dtype=torch.bfloat16, device=dev).contiguous()
            dK = torch.empty((S, B, HKV, D), dtype=torch.bfloat16, device=dev).contiguous()
            dV = torch.empty((S, B, HKV, D), dtype=torch.bfloat16, device=dev).contiguous()
        else:
            dQ = torch.empty((B, S, H, D), dtype=torch.bfloat16, device=dev).contiguous()
            dK = torch.empty((B, S, HKV, D), dtype=torch.bfloat16, device=dev).contiguous()
            dV = torch.empty((B, S, HKV, D), dtype=torch.bfloat16, device=dev).contiguous()
        delta = torch.empty((B, H, 1, S), dtype=torch.float32, device=dev).contiguous()

        if sbhd:
            tk_kernel_bkwd_prep.dispatch_prep_sbhd(O, dO, delta)
            tk_kernel_bkwd.dispatch_bwd_combined_sbhd(q, k, v, dO, dQ_in, dK, dV, L, delta)
            tk_kernel_bkwd_prep.dispatch_dq_shuffle_sbhd(dQ_in, dQ)
        else:
            tk_kernel_bkwd_prep.dispatch_prep(O, dO, delta)
            tk_kernel_bkwd.dispatch_bwd_combined(q, k, v, dO, dQ_in, dK, dV, L, delta)
            tk_kernel_bkwd_prep.dispatch_dq_shuffle(dQ_in, dQ)

        return dQ.to(dO_in.dtype), dK.to(dO_in.dtype), dV.to(dO_in.dtype), None


class HipSelfAttention(nn.Module):
    """Scaled dot product attention via HipKittens TK kernels.

    Supports both bshd (batch, seq, head, dim) and sbhd (seq, batch, head, dim)
    input layouts natively — zero data copy for either layout.
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
        self.sbhd = (input_layout == 'sbhd')

    def forward(self, qkv, causal=None, key_padding_mask=None):
        """
        qkv: (B, S, 3, H, D) if bshd, (S, B, 3, H, D) if sbhd.
        """
        assert qkv.dtype in [torch.float16, torch.bfloat16]
        assert qkv.is_cuda
        causal = self.causal if causal is None else causal
        q, k, v = qkv.unbind(dim=2)
        out = HipAttnFunction.apply(q, k, v, self.sbhd)
        return out.to(q.dtype).contiguous()


class HipCrossAttention(nn.Module):
    """Cross-attention via HipKittens TK kernels.  Supports bshd / sbhd layouts."""

    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0,
                 deterministic=False, input_layout='bshd'):
        super().__init__()
        assert input_layout in _VALID_LAYOUTS
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop = nn.Dropout(attention_dropout)
        self.deterministic = deterministic
        self.sbhd = (input_layout == 'sbhd')

    def forward(self, q, kv, causal=None, key_padding_mask=None):
        """
        q:  (B, Sq, H, D) if bshd, (Sq, B, H, D) if sbhd.
        kv: (B, Sk, 2, H_k, D) if bshd, (Sk, B, 2, H_k, D) if sbhd.
        """
        causal = self.causal if causal is None else causal
        k, v = kv.unbind(dim=2)
        out = HipAttnFunction.apply(q, k, v, self.sbhd)
        return out.to(q.dtype).contiguous()
