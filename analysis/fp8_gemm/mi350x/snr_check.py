"""NaN-safe SNR (signal-to-noise ratio) gate for MXFP4 correctness checks.

Bug background (Round 6 deep-LOSE notes, AGENT_PROMPT.md line 261):
    `bench_deep_lose.py` correctness check has a false-OK SNR bug
    (NaN baselines pass `if snr > 25` because NaN compare is False
    but mis-classifies as OK)

The classic mistake:

    snr_db = 10*math.log10(sig_pwr/err_pwr)        # may be NaN if either is NaN
    if snr_db < 25:                                # NaN < 25 is False → not rejected
        reject()

A NaN SNR can arise when:
  * the variant kernel produces all-NaN output (sig of `got`/`got-ref` not finite)
  * sig_pwr is exactly 0 and err_pwr is also 0 → 0/0 = NaN inside log10
  * any input contains Inf/NaN that contaminates the mean (mean of any-NaN tensor is NaN)

`+Inf` SNR is the OPPOSITE: it means err_pwr → 0 (perfect match). That's GOOD.

Provided helpers below give explicit, NaN-safe semantics:

  classify_snr(snr_db, threshold=25.0) -> ("OK"|"FAIL", reason)
  is_snr_ok(snr_db, threshold=25.0)    -> bool   (False on NaN, True on +Inf)

And `compute_snr_db(ref_tensor, got_tensor)` returns either a float (possibly +Inf)
or raises a SnrError describing why the check is invalid (NaN inputs, all-zero ref,
shape mismatch). Callers should treat any SnrError as FAIL — never silently OK.
"""

from __future__ import annotations

import math
from typing import Tuple


class SnrError(Exception):
    """Raised when SNR cannot be computed (NaN inputs, all-zero ref, etc.)."""


def is_snr_ok(snr_db: float, threshold: float = 25.0) -> bool:
    """NaN-safe SNR gate.

    Returns True iff snr_db is a real number (incl. +Inf) and >= threshold.
    NaN snr_db ALWAYS returns False (this is the bug fix).
    -Inf is below any finite threshold so returns False.
    """
    if snr_db is None:
        return False
    try:
        v = float(snr_db)
    except (TypeError, ValueError):
        return False
    if math.isnan(v):
        return False
    # +Inf >= threshold is True, so a perfect match (err_pwr=0 ⇒ snr=+Inf) passes.
    return v >= threshold


def classify_snr(snr_db: float, threshold: float = 25.0) -> Tuple[str, str]:
    """Return ("OK"|"FAIL", reason). Never returns OK on NaN."""
    if snr_db is None:
        return "FAIL", "snr_db is None"
    try:
        v = float(snr_db)
    except (TypeError, ValueError):
        return "FAIL", f"snr_db not numeric: {snr_db!r}"
    if math.isnan(v):
        return "FAIL", "snr_db is NaN (variant likely emitted NaN/garbage)"
    if math.isinf(v) and v < 0:
        return "FAIL", "snr_db is -Inf (no signal in reference)"
    if v < threshold:
        return "FAIL", f"snr_db {v:.2f} < threshold {threshold}"
    return "OK", f"snr_db {v:.2f} >= threshold {threshold}"


def compute_snr_db(ref, got, eps: float = 1e-30) -> float:
    """Compute SNR(dB) between ref and got torch tensors. NaN-safe.

    Raises SnrError if either input contains NaN/Inf or shapes mismatch.
    Returns +Inf on a perfect match (err_pwr == 0 with sig_pwr > 0).
    """
    import torch  # local import; helper is usable without torch installed for is_snr_ok
    if ref.shape != got.shape:
        raise SnrError(f"shape mismatch ref={tuple(ref.shape)} got={tuple(got.shape)}")
    ref_f = ref.float()
    got_f = got.float()
    if not torch.isfinite(ref_f).all().item():
        raise SnrError("reference tensor contains NaN/Inf")
    if not torch.isfinite(got_f).all().item():
        raise SnrError("variant tensor contains NaN/Inf")
    diff = ref_f - got_f
    sig_pwr = (ref_f * ref_f).mean().item()
    err_pwr = (diff * diff).mean().item()
    if sig_pwr <= 0:
        raise SnrError("reference signal power is zero")
    if err_pwr <= 0:
        return float("inf")
    return 10.0 * math.log10(sig_pwr / err_pwr)
