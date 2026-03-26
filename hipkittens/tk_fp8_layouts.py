from . import _tk_fp8_layouts

gemm_rcr = _tk_fp8_layouts.gemm_rcr
gemm_rrr = _tk_fp8_layouts.gemm_rrr
gemm_crr = _tk_fp8_layouts.gemm_crr

__all__ = ["gemm_rcr", "gemm_rrr", "gemm_crr"]
