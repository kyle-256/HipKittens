#!/bin/bash
# R52 Dev K — A-side LDS double-buffer at 4096³ occ=1 — REFUTED before bench
#
# This script is INTENTIONALLY a no-op recording the bail-clause math.
# The proposed MXFP8_CRR_ASIDE_DOUBLEBUF=1 variant blows the 160 KB
# per-CU LDS cap on the CRR target (136 KB existing + 32 KB new
# A-prefetch = 168 KB > 160 KB). Per prompt step 6, no GPU bench was
# executed.
#
# Per R49 Dev B's lesson:
#   "don't waste GPU time on a known regression"
#
# See r52k_findings.md for the full analytical refute and the audit
# arithmetic captured below.

set -euo pipefail

cat <<'AUDIT'
========================================================================
R52K LDS pre-bench audit — bail clause triggered
========================================================================
Target shape: 4096^3 CRR (8B Q/O, the prompt's HEADROOM canary)

Existing per-CTA LDS budget (R50C-verified, anchored by build remarks):
  CRR (V2, SCALE_VERSION=2):
    LDS Size [bytes/block]: 139264 = 136 KB
    breakdown:
      As[2][2]  = 4 * ST_v2a (16 KB) = 64 KB
      Bs[2][2]  = 4 * ST_v2  (16 KB) = 64 KB
      scale packs / col-A reencode statics ≈ 8 KB
    free per CTA: 160 - 136 = 24 KB

Proposed extension MXFP8_CRR_ASIDE_DOUBLEBUF=1:
  As[3][2]   adds 2 * ST_v2a (16 KB) = 32 KB
  new total: 136 + 32 = 168 KB
  hardware LDS limit: 160 KB per CU

  168 > 160  ==> kernel will NOT launch
                 (hipErrorInvalidConfiguration / sharedMemPerBlock cap)

Smaller variant As_pf[1] (16 KB add) fits at 152 KB but is provably
ISA-equivalent to baseline (one wave-tile lookahead is what the existing
production K-loop ALREADY issues via global_load_a(As[toc][1], k+1) on
crr_mxfp8_exact_8wave_fastpath.inc:204). See r52k_findings.md §3.

Verdict: REFUTED, no source patch, no GPU bench, no compile.
========================================================================
AUDIT

echo "Bench skipped (bail clause). See r52k_findings.md."
exit 0
