# MXFP4 Per-Shape Decomposition Appendix (R61 Opt Z)

**Date:** 2026-04-20  
**Round:** R61 (post-R60)  
**Worker:** L-3 (cohort Opt Z; NO GPU; analysis + writing only)  
**Companion artifacts:** `R61_OPT_U2_PUBLICATION_OUTLINE.md` (cohort L-2 publication outline; this appendix is its per-cell support table); `R60_OPT_U_DOC_PIVOT.md` (R60 K-2 doc pivot; this appendix does NOT duplicate its content — see §0.3 for delineation)

---

## §0 Header + Purpose

### 0.1 Purpose

This appendix supports the publication artifact (`R61_OPT_U2_PUBLICATION_OUTLINE.md`) by enumerating, for each of the 42 production cells in the MXFP4 GEMM project, the tried optimization axes, the closed axes, the current production source, and the one-line justification for why the current source is the best available choice. It consolidates 17 rounds of round-verdict findings (R43 → R60) — across the 4-act project arc (mechanism exhaustion → aiter dispatch breakthrough → systematic HK→AITER swap → 100% leaderboard era → structural ceiling) — into a single per-cell appendix table.

This artifact is the cell-level evidence that backs the publication's results-section claims about per-cell dispatch decisions, axis-closure carry-forward, and the structural ceiling reached at R55-R60.

### 0.2 Source authority

- **Primary:** `R60_INTEGRATION_MANIFEST.json` (canonical 42-cell mapping with current source binary per cell; byte-identical to R59 binary entries which were byte-identical to R58 binary entries — 3rd consecutive binary-identical round)

- **Per-cell perf reading:** `R60_INTEGRATION_10RUN.json` (R60 reviewer 10-run @ 80% under DISJOINT seed set `[202, 404, 606, 808, 1010, 1212, 1414, 1616, 1818, 2020]`)

- **Round-history per-cell:** `R43_INTEGRATION_VERDICT.md` through `R60_INTEGRATION_VERDICT.md` (17 round-verdict files; per-round PROMOTE / SMOKE_DEAD / ACCEPT_FALLBACK / POLICY_ONLY decisions)

- **Closed-axis carry-forward:** `R60_INTEGRATION_VERDICT.md` §12 'R61+ closed-axis carry-forward (DO NOT propose)' + `R59_INTEGRATION_VERDICT.md` §'R60+ closed-axis carry-forward'

- **Methodology references:** `R59_OPT_R_POLICY.md` (Opt R policy A 4-criterion validity envelope for noise-edge cells); `R60_OPT_U_DOC_PIVOT.md` §§1-6 (structural ceiling framing; residual-surface enumeration for L1/L3/L8)

### 0.3 Delineation vs R60 Opt U doc pivot

This Opt Z appendix is COMPLEMENTARY to the R60 Opt U doc pivot, NOT a duplicate:

- **R60 Opt U:** prose-style structural-ceiling analysis (6 sections: structural-ceiling-reached / residual-surface for L1/L3/L8 / SC-MICRO publication claims and disclaimers / R61-R65 axis taxonomy / 42-cell snapshot / 17-round arc narrative). 401 lines, narrative tone.

- **R61 Opt Z (this artifact):** cell-level decomposition table (1 H3 subsection per cell; 42 subsections; tried/closed axes per cell; why-best one-liners). Emphasis on per-cell evidence; reference / lookup tone.

- **R61 Opt U₂:** publication outline + related-work survey + methods-section + results-tables + limitations-section (5 NEW sections distinct from R60 Opt U). Authored in parallel by cohort L-2.

### 0.4 How to read the table (§1)

Each cell occupies a single H3 subsection in §1. The H3 title gives `(M, N, K)` and a cohort tag. Within each subsection, fields are:

- **Cohort:** L1/L3/L8 (3 attention cells with publication-tracked names) OR a K-bucket label for the other 39 cells

- **Shape:** the canonical `M × N × K` triple

- **Current source:** source tag from `R60_INTEGRATION_MANIFEST.json` `shapes_to_source` field, with brief description

- **Current binary path:** `so_path` from manifest (`AITER_SHIM` for the 40 cells dispatched via R50D shim + `.co` binary; absolute build_R40B path for the 2 in-tree HK cells)

- **Current pct_comp** (R60 reviewer 10-run p50): from `R60_INTEGRATION_10RUN.json` `consensus[shape].pct_comp`

- **Current TFLOPS** (R60 p50): from `R60_INTEGRATION_10RUN.json` `consensus[shape].tflops_p50`

- **Strict-VC status:** `PASS_10/10` (n_OK=10/10) etc., from `R60_INTEGRATION_10RUN.json` `verdict` field

- **Bit-determinism status:** `HELD (wcf_max=0.0)` for the 40 AITER cells; `NOT-HELD (wcf_max=...)` for the 2 HK cells

- **Tried axes:** chronological per-round bullet list (R-round + axis/option name + outcome). For cells with limited per-round detail, the entry is condensed to '(R50-R55 AITER promote era; R56-R60 R50D shim AS-IS)'

- **Closed axes:** per-cell axis-closure list with closure round + closure reason / DEAD pp delta

- **Why current source is best:** one-line justification citing the closed-axis evidence above


---

## §1 Per-cell decomposition (42 cells)

All 42 cells from `R60_INTEGRATION_MANIFEST.json`, sorted by `(M, N, K)`. Cell numbering follows sort order.

### Cell 1: (4,096, 4,096, 8,192) — K-bucket: K=8192

- **Cohort:** K-bucket: K=8192
- **Shape:** 4096 × 4096 × 8192
- **Current source tag:** `R55D5A_3_AITER` (promoted in R55)
- **Current source description:** AITER 256×256 .co (R55 D-5A_3 PROMOTE; perf claw-back)
- **Current source mechanism note:** +1.81-20.99pp perf claw-back during 36→42/42 leaderboard close
- **Current binary path:** `AITER_SHIM` → `/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co` via R50D shim `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so`
- **Current pct_comp** (R60 reviewer 10-run p50): **120.52%**
- **Current TFLOPS** (R60 p50): 4772.5
- **Strict-VC status:** PASS_10/10 (n_OK=10/10, wcf_max=0.0000, fin_min=1.0000)
- **Bit-determinism status:** HELD (wcf_max=0.0)
- **Tried axes (chronological by R-round):**
  - R50-R54: prior HK / partial AITER source
  - R55 (D5A_3): perf claw-back PROMOTE; AITER 256×256 .co; +1.81-20.99pp
  - R56-R60: R50D shim AS-IS; PASS_10/10 HELD bit-deterministic
- **Closed axes (specific to this cell):**
  - (no cell-specific axis closures beyond global axes; current AITER 256×256 .co source held since promote round; HK alt-tile axes globally closed for AITER cells)
- **Why current source is best:** AITER 256×256 .co (R50D shim AS-IS) is best because (a) R50-R56 AITER promote era established AITER 256×256 strictly dominant for this M-N-K bucket; (b) HK alt-tiles for AITER cells closed across R57-R59; (c) cell is bit-deterministic (wcf_max=0.0) at 120.52% (R60).

### Cell 2: (4,096, 4,096, 16,384) — K-bucket: K=16384

- **Cohort:** K-bucket: K=16384
- **Shape:** 4096 × 4096 × 16384
- **Current source tag:** `R54D4B_1_AITER` (promoted in R54)
- **Current source description:** AITER 256×256 .co (R54 D-4B_1 PROMOTE)
- **Current source mechanism note:** HK→AITER swap; R50D shim AS-IS (4th)
- **Current binary path:** `AITER_SHIM` → `/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co` via R50D shim `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so`
- **Current pct_comp** (R60 reviewer 10-run p50): **113.40%**
- **Current TFLOPS** (R60 p50): 5264.1
- **Strict-VC status:** PASS_10/10 (n_OK=10/10, wcf_max=0.0000, fin_min=1.0000)
- **Bit-determinism status:** HELD (wcf_max=0.0)
- **Tried axes (chronological by R-round):**
  - R40B-R53: HK 256×256 source (left +17-28pp perf on table)
  - R54 (D4B_1): HK→AITER 256×256 swap PROMOTE +17-28pp
  - R55-R60: R50D shim AS-IS; PASS_10/10 HELD bit-deterministic
- **Closed axes (specific to this cell):**
  - (no cell-specific axis closures beyond global axes; current AITER 256×256 .co source held since promote round; HK alt-tile axes globally closed for AITER cells)
- **Why current source is best:** AITER 256×256 .co (R50D shim AS-IS) is best because (a) R50-R56 AITER promote era established AITER 256×256 strictly dominant for this M-N-K bucket; (b) HK alt-tiles for AITER cells closed across R57-R59; (c) cell is bit-deterministic (wcf_max=0.0) at 113.40% (R60).

### Cell 3: (4,096, 4,096, 32,768) — K-bucket: K=32768

- **Cohort:** K-bucket: K=32768
- **Shape:** 4096 × 4096 × 32768
- **Current source tag:** `R52D2C_AITER` (promoted in R52)
- **Current source description:** AITER 256×256 .co (R52 D-2C PROMOTE)
- **Current source mechanism note:** R50D shim AS-IS (2nd AS-IS); K-generic dispatch
- **Current binary path:** `AITER_SHIM` → `/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co` via R50D shim `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so`
- **Current pct_comp** (R60 reviewer 10-run p50): **108.22%**
- **Current TFLOPS** (R60 p50): 5576.5
- **Strict-VC status:** PASS_10/10 (n_OK=10/10, wcf_max=0.0000, fin_min=1.0000)
- **Bit-determinism status:** HELD (wcf_max=0.0)
- **Tried axes (chronological by R-round):**
  - R50-R51: prior source
  - R52 (D2C): PROMOTE; R50D shim K-generic dispatch
  - R53-R60: R50D shim AS-IS; PASS_10/10 HELD bit-deterministic
- **Closed axes (specific to this cell):**
  - (no cell-specific axis closures beyond global axes; current AITER 256×256 .co source held since promote round; HK alt-tile axes globally closed for AITER cells)
- **Why current source is best:** AITER 256×256 .co (R50D shim AS-IS) is best because (a) R50-R56 AITER promote era established AITER 256×256 strictly dominant for this M-N-K bucket; (b) HK alt-tiles for AITER cells closed across R57-R59; (c) cell is bit-deterministic (wcf_max=0.0) at 108.22% (R60).

### Cell 4: (4,096, 6,144, 32,768) — K-bucket: K=32768

- **Cohort:** K-bucket: K=32768
- **Shape:** 4096 × 6144 × 32768
- **Current source tag:** `R53D3A_3_AITER` (promoted in R53)
- **Current source description:** AITER 256×256 .co (R53 D-3A_3 PROMOTE)
- **Current source mechanism note:** R50D shim AS-IS (3rd); R53 first non-256×256 attempt round, 256×256 selected
- **Current binary path:** `AITER_SHIM` → `/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co` via R50D shim `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so`
- **Current pct_comp** (R60 reviewer 10-run p50): **130.37%**
- **Current TFLOPS** (R60 p50): 4933.6
- **Strict-VC status:** PASS_10/10 (n_OK=10/10, wcf_max=0.0000, fin_min=1.0000)
- **Bit-determinism status:** HELD (wcf_max=0.0)
- **Tried axes (chronological by R-round):**
  - R50-R52: prior HK / AITER source
  - R53 (D3A_3): PROMOTE during first non-256×256 attempt round (256×256 selected)
  - R54-R60: R50D shim AS-IS; PASS_10/10 HELD bit-deterministic
- **Closed axes (specific to this cell):**
  - (no cell-specific axis closures beyond global axes; current AITER 256×256 .co source held since promote round; HK alt-tile axes globally closed for AITER cells)
- **Why current source is best:** AITER 256×256 .co (R50D shim AS-IS) is best because (a) R50-R56 AITER promote era established AITER 256×256 strictly dominant for this M-N-K bucket; (b) HK alt-tiles for AITER cells closed across R57-R59; (c) cell is bit-deterministic (wcf_max=0.0) at 130.37% (R60).

### Cell 5: (4,096, 14,336, 8,192) — K-bucket: K=8192

- **Cohort:** K-bucket: K=8192
- **Shape:** 4096 × 14336 × 8192
- **Current source tag:** `R54D4B_2_AITER` (promoted in R54)
- **Current source description:** AITER 256×256 .co (R54 D-4B_2 PROMOTE)
- **Current source mechanism note:** HK→AITER swap; R50D shim AS-IS (4th)
- **Current binary path:** `AITER_SHIM` → `/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co` via R50D shim `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so`
- **Current pct_comp** (R60 reviewer 10-run p50): **115.79%**
- **Current TFLOPS** (R60 p50): 5032.0
- **Strict-VC status:** PASS_10/10 (n_OK=10/10, wcf_max=0.0000, fin_min=1.0000)
- **Bit-determinism status:** HELD (wcf_max=0.0)
- **Tried axes (chronological by R-round):**
  - R40B-R53: HK 256×256 source (left +17-28pp perf on table)
  - R54 (D4B_2): HK→AITER 256×256 swap PROMOTE +17-28pp
  - R55-R60: R50D shim AS-IS; PASS_10/10 HELD bit-deterministic
- **Closed axes (specific to this cell):**
  - (no cell-specific axis closures beyond global axes; current AITER 256×256 .co source held since promote round; HK alt-tile axes globally closed for AITER cells)
- **Why current source is best:** AITER 256×256 .co (R50D shim AS-IS) is best because (a) R50-R56 AITER promote era established AITER 256×256 strictly dominant for this M-N-K bucket; (b) HK alt-tiles for AITER cells closed across R57-R59; (c) cell is bit-deterministic (wcf_max=0.0) at 115.79% (R60).

### Cell 6: (4,096, 14,336, 16,384) — K-bucket: K=16384

- **Cohort:** K-bucket: K=16384
- **Shape:** 4096 × 14336 × 16384
- **Current source tag:** `R54D4A_1_AITER` (promoted in R54)
- **Current source description:** AITER 256×256 .co (R54 D-4A_1 PROMOTE)
- **Current source mechanism note:** HK→AITER swap +17-28pp claw-back; R50D shim AS-IS (4th)
- **Current binary path:** `AITER_SHIM` → `/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co` via R50D shim `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so`
- **Current pct_comp** (R60 reviewer 10-run p50): **107.05%**
- **Current TFLOPS** (R60 p50): 5366.6
- **Strict-VC status:** PASS_10/10 (n_OK=10/10, wcf_max=0.0000, fin_min=1.0000)
- **Bit-determinism status:** HELD (wcf_max=0.0)
- **Tried axes (chronological by R-round):**
  - R40B-R53: HK 256×256 source (left +17-28pp perf on table)
  - R54 (D4A_1): HK→AITER 256×256 swap PROMOTE +17-28pp
  - R55-R60: R50D shim AS-IS; PASS_10/10 HELD bit-deterministic
- **Closed axes (specific to this cell):**
  - (no cell-specific axis closures beyond global axes; current AITER 256×256 .co source held since promote round; HK alt-tile axes globally closed for AITER cells)
- **Why current source is best:** AITER 256×256 .co (R50D shim AS-IS) is best because (a) R50-R56 AITER promote era established AITER 256×256 strictly dominant for this M-N-K bucket; (b) HK alt-tiles for AITER cells closed across R57-R59; (c) cell is bit-deterministic (wcf_max=0.0) at 107.05% (R60).

### Cell 7: (4,096, 28,672, 32,768) — K-bucket: K=32768

- **Cohort:** K-bucket: K=32768
- **Shape:** 4096 × 28672 × 32768
- **Current source tag:** `R52D2A_AITER` (promoted in R52)
- **Current source description:** AITER 256×256 .co (R52 D-2A PROMOTE)
- **Current source mechanism note:** R50D shim AS-IS (2nd AS-IS); K-generic dispatch
- **Current binary path:** `AITER_SHIM` → `/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co` via R50D shim `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so`
- **Current pct_comp** (R60 reviewer 10-run p50): **102.47%**
- **Current TFLOPS** (R60 p50): 5789.6
- **Strict-VC status:** PASS_10/10 (n_OK=10/10, wcf_max=0.0000, fin_min=1.0000)
- **Bit-determinism status:** HELD (wcf_max=0.0)
- **Tried axes (chronological by R-round):**
  - R50-R51: prior source
  - R52 (D2A): PROMOTE; R50D shim K-generic dispatch
  - R53-R60: R50D shim AS-IS; PASS_10/10 HELD bit-deterministic
- **Closed axes (specific to this cell):**
  - (no cell-specific axis closures beyond global axes; current AITER 256×256 .co source held since promote round; HK alt-tile axes globally closed for AITER cells)
- **Why current source is best:** AITER 256×256 .co (R50D shim AS-IS) is best because (a) R50-R56 AITER promote era established AITER 256×256 strictly dominant for this M-N-K bucket; (b) HK alt-tiles for AITER cells closed across R57-R59; (c) cell is bit-deterministic (wcf_max=0.0) at 102.47% (R60).

### Cell 8: (4,096, 32,768, 4,096) — K-bucket: K=4096

- **Cohort:** K-bucket: K=4096
- **Shape:** 4096 × 32768 × 4096
- **Current source tag:** `R54E2_1_AITER` (promoted in R54)
- **Current source description:** AITER 256×256 .co (R54 E-2_1 PROMOTE NEW VC rescue)
- **Current source mechanism note:** NEW strict-VC unlock via R50D shim
- **Current binary path:** `AITER_SHIM` → `/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co` via R50D shim `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so`
- **Current pct_comp** (R60 reviewer 10-run p50): **105.84%**
- **Current TFLOPS** (R60 p50): 4409.7
- **Strict-VC status:** PASS_10/10 (n_OK=10/10, wcf_max=0.0000, fin_min=1.0000)
- **Bit-determinism status:** HELD (wcf_max=0.0)
- **Tried axes (chronological by R-round):**
  - R50-R53: cell remained outside strict-VC (NEW VC rescue cohort)
  - R54 (E2_1): NEW VC unlock via AITER 256×256 .co dlopen (R50D shim AS-IS)
  - R55-R60: R50D shim AS-IS; PASS_10/10 HELD bit-deterministic
- **Closed axes (specific to this cell):**
  - (no cell-specific axis closures beyond global axes; current AITER 256×256 .co source held since promote round; HK alt-tile axes globally closed for AITER cells)
- **Why current source is best:** AITER 256×256 .co (R50D shim AS-IS) is best because (a) R50-R56 AITER promote era established AITER 256×256 strictly dominant for this M-N-K bucket; (b) HK alt-tiles for AITER cells closed across R57-R59; (c) cell is bit-deterministic (wcf_max=0.0) at 105.84% (R60).

### Cell 9: (4,096, 32,768, 6,144) — K-bucket: K=6144

- **Cohort:** K-bucket: K=6144
- **Shape:** 4096 × 32768 × 6144
- **Current source tag:** `R54E2_2_AITER` (promoted in R54)
- **Current source description:** AITER 256×256 .co (R54 E-2_2 PROMOTE NEW VC rescue)
- **Current source mechanism note:** NEW strict-VC unlock via R50D shim
- **Current binary path:** `AITER_SHIM` → `/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co` via R50D shim `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so`
- **Current pct_comp** (R60 reviewer 10-run p50): **105.79%**
- **Current TFLOPS** (R60 p50): 4811.9
- **Strict-VC status:** PASS_10/10 (n_OK=10/10, wcf_max=0.0000, fin_min=1.0000)
- **Bit-determinism status:** HELD (wcf_max=0.0)
- **Tried axes (chronological by R-round):**
  - R50-R53: cell remained outside strict-VC (NEW VC rescue cohort)
  - R54 (E2_2): NEW VC unlock via AITER 256×256 .co dlopen (R50D shim AS-IS)
  - R55-R60: R50D shim AS-IS; PASS_10/10 HELD bit-deterministic
- **Closed axes (specific to this cell):**
  - (no cell-specific axis closures beyond global axes; current AITER 256×256 .co source held since promote round; HK alt-tile axes globally closed for AITER cells)
- **Why current source is best:** AITER 256×256 .co (R50D shim AS-IS) is best because (a) R50-R56 AITER promote era established AITER 256×256 strictly dominant for this M-N-K bucket; (b) HK alt-tiles for AITER cells closed across R57-R59; (c) cell is bit-deterministic (wcf_max=0.0) at 105.79% (R60).

### Cell 10: (4,096, 32,768, 14,336) — L1 (attention-noise-edge)

- **Cohort:** L1 (attention-noise-edge)
- **Shape:** 4096 × 32768 × 14336
- **Current source tag:** `R57J1_L1_AITER` (promoted in R57)
- **Current source description:** AITER 256×256 .co (R57 J-1 L1)
- **Current source mechanism note:** noise-edge cell; ITERS=1000 one-off lifted 99.98%→100.04% R57; ITERS=500 revert R58+; binary unchanged R57→R60
- **Current binary path:** `AITER_SHIM` → `/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co` via R50D shim `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so`
- **Current pct_comp** (R60 reviewer 10-run p50): **99.98%**
- **Current TFLOPS** (R60 p50): 5295.1
- **Strict-VC status:** PASS_10/10 (n_OK=10/10, wcf_max=0.0000, fin_min=1.0000)
- **Bit-determinism status:** HELD (wcf_max=0.0)
- **Tried axes (chronological by R-round):**
  - R44-R49: prefetch-gate / VGPR-PF / external fence axes all DEAD on this cohort
  - R50: AITER 256×256 .co dlopen via R50D shim → PROMOTE (first non-DEAD round since R44)
  - R51-R55: R50D shim AS-IS reuse; binary unchanged
  - R57 J-1 L1 (Opt J): ITERS=500→1000 protocol-only WIN lifted 99.98%→100.04% on UNCHANGED binary
  - R58: ITERS=500 default revert → 99.94% LOSE-edge re-emergence (binary unchanged)
  - R59: ITERS=500 fresh seed sweep → 100.08% WIN crossback (binary unchanged)
  - R59 J-3 V-1: AITER 96×640 alt-tile → SMOKE_DEAD 87.42% (-12.52pp)
  - R59 J-3 V-2: AITER 64×1024 alt-tile → SMOKE_DEAD 66.05% (-33.89pp)
  - R60: ITERS=500 fresh seed sweep → 99.98% LOSE-edge (4-round oscillation envelope characterized; ±0.10pp around WIN-line)
- **Closed axes (specific to this cell):**
  - AITER 96×640 alt-tile — R59 J-3 V-1 SMOKE_DEAD 87.42% (-12.52pp)
  - AITER 64×1024 alt-tile — R59 J-3 V-2 SMOKE_DEAD 66.05% (-33.89pp)
  - ITERS=1000 one-off bump — CLOSED by Opt R policy A 4-criterion validity envelope (R59)
  - Opt X (L1 cross-product alt-tiles or grid swizzle) — CLOSED in R59 (alt-tile space EXHAUSTED)
- **Why current source is best:** R57J1_L1 AITER 256×256 is best because L1 alt-tile space EXHAUSTED in R59 (96×640 -12.52pp + 64×1024 -33.89pp DEAD); cell is bit-deterministic (wcf=0.0) noise-edge oscillating ±0.10pp around WIN-line under ITERS=500; per Opt R policy A, per-sweep classification noise — production reading 99.98% (R60).

### Cell 11: (4,096, 32,768, 28,672) — K-bucket: K=28672

- **Cohort:** K-bucket: K=28672
- **Shape:** 4096 × 32768 × 28672
- **Current source tag:** `R50D_AITER` (promoted in R50)
- **Current source description:** AITER 256×256 .co dlopen (R50D shim breakthrough)
- **Current source mechanism note:** first non-DEAD round since R44; perma-CRASH (4096,32768,28672) solved via hipModuleLoadData
- **Current binary path:** `AITER_SHIM` → `/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co` via R50D shim `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so`
- **Current pct_comp** (R60 reviewer 10-run p50): **101.19%**
- **Current TFLOPS** (R60 p50): 5634.6
- **Strict-VC status:** PASS_10/10 (n_OK=10/10, wcf_max=0.0000, fin_min=1.0000)
- **Bit-determinism status:** HELD (wcf_max=0.0)
- **Tried axes (chronological by R-round):**
  - R44-R49: prefetch-gate / VGPR-PF / external fence / aiter vmcnt port / asm-split / PF_MPT all DEAD
  - R50 D: AITER 256×256 .co dlopen via R50D shim PROMOTE — first non-DEAD round since R44; 100.31% comp
  - R51-R60: R50D shim AS-IS; PASS_10/10 HELD bit-deterministic
- **Closed axes (specific to this cell):**
  - (no cell-specific axis closures beyond global axes; current AITER 256×256 .co source held since promote round; HK alt-tile axes globally closed for AITER cells)
- **Why current source is best:** AITER 256×256 .co (R50D shim AS-IS) is best because (a) R50-R56 AITER promote era established AITER 256×256 strictly dominant for this M-N-K bucket; (b) HK alt-tiles for AITER cells closed across R57-R59; (c) cell is bit-deterministic (wcf_max=0.0) at 101.19% (R60).

### Cell 12: (4,096, 32,768, 128,256) — L8 (attention-LOSE)

- **Cohort:** L8 (attention-LOSE)
- **Shape:** 4096 × 32768 × 128256
- **Current source tag:** `R52D2B_AITER` (promoted in R52)
- **Current source description:** AITER 256×256 .co (R52 D-2B PROMOTE)
- **Current source mechanism note:** R50D shim AS-IS (2nd AS-IS); K-generic dispatch
- **Current binary path:** `AITER_SHIM` → `/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co` via R50D shim `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so`
- **Current pct_comp** (R60 reviewer 10-run p50): **98.29%**
- **Current TFLOPS** (R60 p50): 5682.4
- **Strict-VC status:** PASS_10/10 (n_OK=10/10, wcf_max=0.0000, fin_min=1.0000)
- **Bit-determinism status:** HELD (wcf_max=0.0)
- **Tried axes (chronological by R-round):**
  - R44-R49: HK K=128256 axes all DEAD (correctness gate; R39A/R44A/R44D never ported)
  - R50: AITER 256×256 .co dlopen via R50D shim → PROMOTE
  - R52 D-2B: PROMOTE refinement; R50D shim AS-IS
  - R56 G-3: AITER 128×512 alt-tile → SMOKE_DEAD -13.31pp
  - R56: AITER 192×256 alt-tile → SMOKE_DEAD -11.84pp
  - R57 K-Opt K: AITER 224×256 alt-tile → SMOKE_DEAD -10.69pp
  - R58 Opt O: HK 256×256 lgk2 v12 fallback rescan → BOTH R40B + R37 WRONG_OUTPUT (CLOSED by correctness)
  - L8 AITER alt-tile space FULLY CLOSED; only Opt T (~3 R-rounds, very low confidence from-scratch HK build) remains
- **Closed axes (specific to this cell):**
  - AITER 128×512 alt-tile — R56 G-3 SMOKE_DEAD -13.31pp
  - AITER 192×256 alt-tile — R56 SMOKE_DEAD -11.84pp
  - AITER 224×256 alt-tile — R57 K-Opt K SMOKE_DEAD -10.69pp
  - AITER 96×640 alt-tile — CLOSED R57 carry-forward
  - AITER 64×1024 alt-tile — CLOSED R57 carry-forward
  - HK 256×256 lgk2 v12 axis — R58 Opt O CLOSED by correctness (R40B + R37 fallbacks WRONG_OUTPUT; R39A/R44A/R44D never ported into K=128256 build)
- **Why current source is best:** R52D2B AITER 256×256 .co is best because all AITER alt-tiles (128×512, 192×256, 224×256, 96×640, 64×1024) closed in R56/R57; HK 256×256 lgk2 v12 axis CLOSED by correctness in R58 Opt O (R40B + R37 fallbacks WRONG_OUTPUT for K=128256; R39A/R44A/R44D fixes never ported); only Opt T (~3 R-rounds, very low confidence) remains; cell is at aiter-internal ceiling at 98.29% (1.66pp gap).

### Cell 13: (4,096, 128,256, 32,768) — K-bucket: K=32768

- **Cohort:** K-bucket: K=32768
- **Shape:** 4096 × 128256 × 32768
- **Current source tag:** `R56G4_C1_AITER` (promoted in R56)
- **Current source description:** AITER 256×256 .co (R56 G-4 C1 PROMOTE; first kept-HK→AITER swap)
- **Current source mechanism note:** +80.92pp claw-back; first kept-HK cell promoted to AITER
- **Current binary path:** `AITER_SHIM` → `/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co` via R50D shim `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so`
- **Current pct_comp** (R60 reviewer 10-run p50): **178.40%**
- **Current TFLOPS** (R60 p50): 5700.4
- **Strict-VC status:** PASS_10/10 (n_OK=10/10, wcf_max=0.0000, fin_min=1.0000)
- **Bit-determinism status:** HELD (wcf_max=0.0)
- **Tried axes (chronological by R-round):**
  - R50-R55: kept on prior HK source across AITER-promote rounds
  - R56 G-4 C1: HK→AITER 256×256 swap PROMOTE +80.92pp on the first kept-HK cell to migrate to AITER
  - R57+: R50D shim AS-IS; binary unchanged
- **Closed axes (specific to this cell):**
  - (no cell-specific axis closures beyond global axes; current AITER 256×256 .co source held since promote round; HK alt-tile axes globally closed for AITER cells)
- **Why current source is best:** AITER 256×256 .co is best because R56 G-1/G-2/G-4 PROMOTEs delivered +117.22pp+ aggregate cluster claw-back over prior 64×1024 dispatch; 256×256 strictly best for this M-N-K bucket; bit-deterministic at 178.40%.

### Cell 14: (6,144, 4,096, 8,192) — K-bucket: K=8192

- **Cohort:** K-bucket: K=8192
- **Shape:** 6144 × 4096 × 8192
- **Current source tag:** `R54D4A_3_AITER` (promoted in R54)
- **Current source description:** AITER 256×256 .co (R54 D-4A_3 PROMOTE)
- **Current source mechanism note:** HK→AITER swap; R50D shim AS-IS (4th)
- **Current binary path:** `AITER_SHIM` → `/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co` via R50D shim `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so`
- **Current pct_comp** (R60 reviewer 10-run p50): **117.59%**
- **Current TFLOPS** (R60 p50): 4494.2
- **Strict-VC status:** PASS_10/10 (n_OK=10/10, wcf_max=0.0000, fin_min=1.0000)
- **Bit-determinism status:** HELD (wcf_max=0.0)
- **Tried axes (chronological by R-round):**
  - R40B-R53: HK 256×256 source (left +17-28pp perf on table)
  - R54 (D4A_3): HK→AITER 256×256 swap PROMOTE +17-28pp
  - R55-R60: R50D shim AS-IS; PASS_10/10 HELD bit-deterministic
- **Closed axes (specific to this cell):**
  - (no cell-specific axis closures beyond global axes; current AITER 256×256 .co source held since promote round; HK alt-tile axes globally closed for AITER cells)
- **Why current source is best:** AITER 256×256 .co (R50D shim AS-IS) is best because (a) R50-R56 AITER promote era established AITER 256×256 strictly dominant for this M-N-K bucket; (b) HK alt-tiles for AITER cells closed across R57-R59; (c) cell is bit-deterministic (wcf_max=0.0) at 117.59% (R60).

### Cell 15: (6,144, 4,096, 16,384) — K-bucket: K=16384

- **Cohort:** K-bucket: K=16384
- **Shape:** 6144 × 4096 × 16384
- **Current source tag:** `R53D3A_2_AITER` (promoted in R53)
- **Current source description:** AITER 256×256 .co (R53 D-3A_2 PROMOTE)
- **Current source mechanism note:** R50D shim AS-IS (3rd); R53 first non-256×256 attempt round, 256×256 selected
- **Current binary path:** `AITER_SHIM` → `/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co` via R50D shim `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so`
- **Current pct_comp** (R60 reviewer 10-run p50): **105.30%**
- **Current TFLOPS** (R60 p50): 4662.6
- **Strict-VC status:** PASS_10/10 (n_OK=10/10, wcf_max=0.0000, fin_min=1.0000)
- **Bit-determinism status:** HELD (wcf_max=0.0)
- **Tried axes (chronological by R-round):**
  - R50-R52: prior HK / AITER source
  - R53 (D3A_2): PROMOTE during first non-256×256 attempt round (256×256 selected)
  - R54-R60: R50D shim AS-IS; PASS_10/10 HELD bit-deterministic
- **Closed axes (specific to this cell):**
  - (no cell-specific axis closures beyond global axes; current AITER 256×256 .co source held since promote round; HK alt-tile axes globally closed for AITER cells)
- **Why current source is best:** AITER 256×256 .co (R50D shim AS-IS) is best because (a) R50-R56 AITER promote era established AITER 256×256 strictly dominant for this M-N-K bucket; (b) HK alt-tiles for AITER cells closed across R57-R59; (c) cell is bit-deterministic (wcf_max=0.0) at 105.30% (R60).

### Cell 16: (6,144, 32,768, 4,096) — K-bucket: K=4096

- **Cohort:** K-bucket: K=4096
- **Shape:** 6144 × 32768 × 4096
- **Current source tag:** `R55E4_2_AITER` (promoted in R55)
- **Current source description:** AITER 256×256 .co (R55 E-4_2 PROMOTE NEW VC rescue)
- **Current source mechanism note:** NEW strict-VC unlock; 36→42/42 first 100% leaderboard
- **Current binary path:** `AITER_SHIM` → `/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co` via R50D shim `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so`
- **Current pct_comp** (R60 reviewer 10-run p50): **106.64%**
- **Current TFLOPS** (R60 p50): 4576.0
- **Strict-VC status:** PASS_10/10 (n_OK=10/10, wcf_max=0.0000, fin_min=1.0000)
- **Bit-determinism status:** HELD (wcf_max=0.0)
- **Tried axes (chronological by R-round):**
  - R50-R54: cell remained outside strict-VC pool (cohort-race / wrong-output / under HK source)
  - R55 (E4_2): NEW VC unlock via AITER 256×256 .co dlopen (R50D shim AS-IS)
  - R56-R60: R50D shim AS-IS; PASS_10/10 HELD bit-deterministic
- **Closed axes (specific to this cell):**
  - (no cell-specific axis closures beyond global axes; current AITER 256×256 .co source held since promote round; HK alt-tile axes globally closed for AITER cells)
- **Why current source is best:** AITER 256×256 .co (R50D shim AS-IS) is best because (a) R50-R56 AITER promote era established AITER 256×256 strictly dominant for this M-N-K bucket; (b) HK alt-tiles for AITER cells closed across R57-R59; (c) cell is bit-deterministic (wcf_max=0.0) at 106.64% (R60).

### Cell 17: (14,336, 4,096, 32,768) — K-bucket: K=32768

- **Cohort:** K-bucket: K=32768
- **Shape:** 14336 × 4096 × 32768
- **Current source tag:** `R51D1_AITER` (promoted in R51)
- **Current source description:** AITER 256×256 .co (R51 D-1 PROMOTE)
- **Current source mechanism note:** R50D shim AS-IS reuse (1st AS-IS round); shape-generic
- **Current binary path:** `AITER_SHIM` → `/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co` via R50D shim `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so`
- **Current pct_comp** (R60 reviewer 10-run p50): **103.86%**
- **Current TFLOPS** (R60 p50): 5447.7
- **Strict-VC status:** PASS_10/10 (n_OK=10/10, wcf_max=0.0000, fin_min=1.0000)
- **Bit-determinism status:** HELD (wcf_max=0.0)
- **Tried axes (chronological by R-round):**
  - R50: R50D breakthrough on (4096,32768,28672)
  - R51 (D1): PROMOTE; R50D shim AS-IS shape-generic dispatch
  - R52-R60: R50D shim AS-IS; PASS_10/10 HELD bit-deterministic
- **Closed axes (specific to this cell):**
  - (no cell-specific axis closures beyond global axes; current AITER 256×256 .co source held since promote round; HK alt-tile axes globally closed for AITER cells)
- **Why current source is best:** AITER 256×256 .co (R50D shim AS-IS) is best because (a) R50-R56 AITER promote era established AITER 256×256 strictly dominant for this M-N-K bucket; (b) HK alt-tiles for AITER cells closed across R57-R59; (c) cell is bit-deterministic (wcf_max=0.0) at 103.86% (R60).

### Cell 18: (14,336, 32,768, 4,096) — K-bucket: K=4096

- **Cohort:** K-bucket: K=4096
- **Shape:** 14336 × 32768 × 4096
- **Current source tag:** `R56G2_L5_AITER` (promoted in R56)
- **Current source description:** AITER 256×256 .co (R56 G-2 L5 PROMOTE; 64×1024→256×256 swap)
- **Current source mechanism note:** cluster A/B perf claw-back; +117.22pp aggregate over G-1+G-2
- **Current binary path:** `AITER_SHIM` → `/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co` via R50D shim `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so`
- **Current pct_comp** (R60 reviewer 10-run p50): **103.23%**
- **Current TFLOPS** (R60 p50): 4606.7
- **Strict-VC status:** PASS_10/10 (n_OK=10/10, wcf_max=0.0000, fin_min=1.0000)
- **Bit-determinism status:** HELD (wcf_max=0.0)
- **Tried axes (chronological by R-round):**
  - R50: AITER .co dlopen via R50D shim → PROMOTE
  - R51-R55: prior 64×1024 dispatch held
  - R56 (G-1/G-2): 64×1024→256×256 swap PROMOTE (cluster A/B perf claw-back; +117.22pp aggregate)
  - R57-R60: R50D shim AS-IS; binary unchanged
- **Closed axes (specific to this cell):**
  - (no cell-specific axis closures beyond global axes; current AITER 256×256 .co source held since promote round; HK alt-tile axes globally closed for AITER cells)
- **Why current source is best:** AITER 256×256 .co is best because R56 G-1/G-2/G-4 PROMOTEs delivered +117.22pp+ aggregate cluster claw-back over prior 64×1024 dispatch; 256×256 strictly best for this M-N-K bucket; bit-deterministic at 103.23%.

### Cell 19: (16,384, 4,096, 2,048) — K-bucket: K=2048

- **Cohort:** K-bucket: K=2048
- **Shape:** 16384 × 4096 × 2048
- **Current source tag:** `R40B` (promoted in R40B)
- **Current source description:** HK kernel R40B 256×256
- **Current source mechanism note:** in-tree HK build with R39A TAIL_SCALE_CLAMP + R44A back-edge drain + R44D FINITE_GATE 0.97
- **Current binary path:** `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/build_R40B/tk_mxfp4_gluon_cpp_n4096_k2048_ts_v12_tv16_R40B_safe.cpython-310-x86_64-linux-gnu.so`
- **Current pct_comp** (R60 reviewer 10-run p50): **108.52%**
- **Current TFLOPS** (R60 p50): 3250.2
- **Strict-VC status:** PASS_10/10 (n_OK=10/10, wcf_max=0.0006, fin_min=0.9901)
- **Bit-determinism status:** NOT-HELD (wcf_max=0.0006)
- **Tried axes (chronological by R-round):**
  - R40 baseline: HK R40B 256×256 with R39A TAIL_SCALE_CLAMP + R44A back-edge drain + R44D FINITE_GATE 0.97 → PROMOTE
  - R50-R55: HK R40B kept across AITER-promote rounds (HK strictly best on this K=2048 cohort)
  - R57 H-2: 192×256 alt-tile probe → SMOKE_DEAD -8 to -14pp on HK kept-cell pool
  - R58 P-1: AITER 128×256 swap probe → ACCEPT_FALLBACK -3.68pp
  - R58 Opt N: gate-tightening informational scan → no production change
  - R59 K-1 / R60 K-1: cohort-race re-bench under DISJOINT seeds → both PASS_10/10 (HELD)
- **Closed axes (specific to this cell):**
  - AITER 192×256 alt-tile (HK kept-cell pool) — R57 H-2 SMOKE_DEAD -8 to -14pp
  - AITER 128×256 alt-tile — R58 P-1 ACCEPT_FALLBACK -3.68pp
- **Why current source is best:** HK R40B 256×256 is best because AITER 192×256 (R57 H-2 -8 to -14pp) and AITER 128×256 (R58 P-1 -3.68pp) probes both DEAD; R59 K-1 + R60 K-1 cohort-race re-bench retained PASS_10/10 + WIN at 108.52%; HK strictly best on this K=2048 N=4096 cohort.

### Cell 20: (16,384, 4,096, 3,072) — K-bucket: K=3072

- **Cohort:** K-bucket: K=3072
- **Shape:** 16384 × 4096 × 3072
- **Current source tag:** `R58P2_AITER` (promoted in R58)
- **Current source description:** AITER 128×256 .co (R58 P-2 PROMOTE; first non-256×256 production tile)
- **Current source mechanism note:** HK→AITER 128×256 swap +4.37pp; AITER 128×256 strictly best for this cell
- **Current binary path:** `AITER_SHIM` → `/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_128x256.co` via R50D shim `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so`
- **Current pct_comp** (R60 reviewer 10-run p50): **107.85%**
- **Current TFLOPS** (R60 p50): 3766.4
- **Strict-VC status:** PASS_10/10 (n_OK=10/10, wcf_max=0.0000, fin_min=1.0000)
- **Bit-determinism status:** HELD (wcf_max=0.0)
- **Tried axes (chronological by R-round):**
  - R50-R57: HK R40B kept on this cell
  - R58 P-2: HK→AITER 128×256 swap PROMOTE +4.37pp (first non-256×256 production tile in manifest)
  - R59 / R60: AITER 128×256 binary AS-IS; PASS_10/10 HELD bit-deterministic
- **Closed axes (specific to this cell):**
  - HK 256×256 — superseded by R58 P-2 AITER 128×256 swap (+4.37pp)
- **Why current source is best:** R58 P-2 AITER 128×256 .co is best because R58 P-2 PROMOTE established +4.37pp over prior HK 256×256 source; AITER 128×256 strictly best per axis decider (HK R40B 256×256 cohort-race intensity loses to AITER 128×256 efficiency 85.3); cell is bit-deterministic at 107.85%.

### Cell 21: (16,384, 4,096, 4,096) — K-bucket: K=4096

- **Cohort:** K-bucket: K=4096
- **Shape:** 16384 × 4096 × 4096
- **Current source tag:** `R55D5A_1_AITER` (promoted in R55)
- **Current source description:** AITER 256×256 .co (R55 D-5A_1 PROMOTE; perf claw-back)
- **Current source mechanism note:** +1.81-20.99pp perf claw-back during 36→42/42 leaderboard close
- **Current binary path:** `AITER_SHIM` → `/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co` via R50D shim `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so`
- **Current pct_comp** (R60 reviewer 10-run p50): **112.28%**
- **Current TFLOPS** (R60 p50): 4437.2
- **Strict-VC status:** PASS_10/10 (n_OK=10/10, wcf_max=0.0000, fin_min=1.0000)
- **Bit-determinism status:** HELD (wcf_max=0.0)
- **Tried axes (chronological by R-round):**
  - R50-R54: prior HK / partial AITER source
  - R55 (D5A_1): perf claw-back PROMOTE; AITER 256×256 .co; +1.81-20.99pp
  - R56-R60: R50D shim AS-IS; PASS_10/10 HELD bit-deterministic
- **Closed axes (specific to this cell):**
  - (no cell-specific axis closures beyond global axes; current AITER 256×256 .co source held since promote round; HK alt-tile axes globally closed for AITER cells)
- **Why current source is best:** AITER 256×256 .co (R50D shim AS-IS) is best because (a) R50-R56 AITER promote era established AITER 256×256 strictly dominant for this M-N-K bucket; (b) HK alt-tiles for AITER cells closed across R57-R59; (c) cell is bit-deterministic (wcf_max=0.0) at 112.28% (R60).

### Cell 22: (16,384, 4,096, 6,144) — K-bucket: K=6144

- **Cohort:** K-bucket: K=6144
- **Shape:** 16384 × 4096 × 6144
- **Current source tag:** `R54E2_3_AITER` (promoted in R54)
- **Current source description:** AITER 256×256 .co (R54 E-2_3 PROMOTE NEW VC rescue)
- **Current source mechanism note:** NEW strict-VC unlock via R50D shim
- **Current binary path:** `AITER_SHIM` → `/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co` via R50D shim `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so`
- **Current pct_comp** (R60 reviewer 10-run p50): **115.67%**
- **Current TFLOPS** (R60 p50): 4927.6
- **Strict-VC status:** PASS_10/10 (n_OK=10/10, wcf_max=0.0000, fin_min=1.0000)
- **Bit-determinism status:** HELD (wcf_max=0.0)
- **Tried axes (chronological by R-round):**
  - R50-R53: cell remained outside strict-VC (NEW VC rescue cohort)
  - R54 (E2_3): NEW VC unlock via AITER 256×256 .co dlopen (R50D shim AS-IS)
  - R55-R60: R50D shim AS-IS; PASS_10/10 HELD bit-deterministic
- **Closed axes (specific to this cell):**
  - (no cell-specific axis closures beyond global axes; current AITER 256×256 .co source held since promote round; HK alt-tile axes globally closed for AITER cells)
- **Why current source is best:** AITER 256×256 .co (R50D shim AS-IS) is best because (a) R50-R56 AITER promote era established AITER 256×256 strictly dominant for this M-N-K bucket; (b) HK alt-tiles for AITER cells closed across R57-R59; (c) cell is bit-deterministic (wcf_max=0.0) at 115.67% (R60).

### Cell 23: (16,384, 4,096, 7,168) — K-bucket: K=7168

- **Cohort:** K-bucket: K=7168
- **Shape:** 16384 × 4096 × 7168
- **Current source tag:** `R54D4B_3_AITER` (promoted in R54)
- **Current source description:** AITER 256×256 .co (R54 D-4B_3 PROMOTE)
- **Current source mechanism note:** HK→AITER swap; R50D shim AS-IS (4th)
- **Current binary path:** `AITER_SHIM` → `/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co` via R50D shim `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so`
- **Current pct_comp** (R60 reviewer 10-run p50): **113.39%**
- **Current TFLOPS** (R60 p50): 5038.2
- **Strict-VC status:** PASS_10/10 (n_OK=10/10, wcf_max=0.0000, fin_min=1.0000)
- **Bit-determinism status:** HELD (wcf_max=0.0)
- **Tried axes (chronological by R-round):**
  - R40B-R53: HK 256×256 source (left +17-28pp perf on table)
  - R54 (D4B_3): HK→AITER 256×256 swap PROMOTE +17-28pp
  - R55-R60: R50D shim AS-IS; PASS_10/10 HELD bit-deterministic
- **Closed axes (specific to this cell):**
  - (no cell-specific axis closures beyond global axes; current AITER 256×256 .co source held since promote round; HK alt-tile axes globally closed for AITER cells)
- **Why current source is best:** AITER 256×256 .co (R50D shim AS-IS) is best because (a) R50-R56 AITER promote era established AITER 256×256 strictly dominant for this M-N-K bucket; (b) HK alt-tiles for AITER cells closed across R57-R59; (c) cell is bit-deterministic (wcf_max=0.0) at 113.39% (R60).

### Cell 24: (16,384, 4,096, 14,336) — K-bucket: K=14336

- **Cohort:** K-bucket: K=14336
- **Shape:** 16384 × 4096 × 14336
- **Current source tag:** `R56G1_L6_AITER` (promoted in R56)
- **Current source description:** AITER 256×256 .co (R56 G-1 L6 PROMOTE; 64×1024→256×256 swap)
- **Current source mechanism note:** cluster A/B perf claw-back; +117.22pp aggregate over G-1+G-2
- **Current binary path:** `AITER_SHIM` → `/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co` via R50D shim `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so`
- **Current pct_comp** (R60 reviewer 10-run p50): **106.99%**
- **Current TFLOPS** (R60 p50): 5501.4
- **Strict-VC status:** PASS_10/10 (n_OK=10/10, wcf_max=0.0000, fin_min=1.0000)
- **Bit-determinism status:** HELD (wcf_max=0.0)
- **Tried axes (chronological by R-round):**
  - R50: AITER .co dlopen via R50D shim → PROMOTE
  - R51-R55: prior 64×1024 dispatch held
  - R56 (G-1/G-2): 64×1024→256×256 swap PROMOTE (cluster A/B perf claw-back; +117.22pp aggregate)
  - R57-R60: R50D shim AS-IS; binary unchanged
- **Closed axes (specific to this cell):**
  - (no cell-specific axis closures beyond global axes; current AITER 256×256 .co source held since promote round; HK alt-tile axes globally closed for AITER cells)
- **Why current source is best:** AITER 256×256 .co is best because R56 G-1/G-2/G-4 PROMOTEs delivered +117.22pp+ aggregate cluster claw-back over prior 64×1024 dispatch; 256×256 strictly best for this M-N-K bucket; bit-deterministic at 106.99%.

### Cell 25: (16,384, 4,096, 28,672) — K-bucket: K=28672

- **Cohort:** K-bucket: K=28672
- **Shape:** 16384 × 4096 × 28672
- **Current source tag:** `R51D2_AITER` (promoted in R51)
- **Current source description:** AITER 256×256 .co (R51 D-2 PROMOTE)
- **Current source mechanism note:** R50D shim AS-IS reuse (1st AS-IS round); shape-generic
- **Current binary path:** `AITER_SHIM` → `/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co` via R50D shim `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so`
- **Current pct_comp** (R60 reviewer 10-run p50): **102.98%**
- **Current TFLOPS** (R60 p50): 5689.7
- **Strict-VC status:** PASS_10/10 (n_OK=10/10, wcf_max=0.0000, fin_min=1.0000)
- **Bit-determinism status:** HELD (wcf_max=0.0)
- **Tried axes (chronological by R-round):**
  - R50: R50D breakthrough on (4096,32768,28672)
  - R51 (D2): PROMOTE; R50D shim AS-IS shape-generic dispatch
  - R52-R60: R50D shim AS-IS; PASS_10/10 HELD bit-deterministic
- **Closed axes (specific to this cell):**
  - (no cell-specific axis closures beyond global axes; current AITER 256×256 .co source held since promote round; HK alt-tile axes globally closed for AITER cells)
- **Why current source is best:** AITER 256×256 .co (R50D shim AS-IS) is best because (a) R50-R56 AITER promote era established AITER 256×256 strictly dominant for this M-N-K bucket; (b) HK alt-tiles for AITER cells closed across R57-R59; (c) cell is bit-deterministic (wcf_max=0.0) at 102.98% (R60).

### Cell 26: (16,384, 6,144, 2,048) — K-bucket: K=2048

- **Cohort:** K-bucket: K=2048
- **Shape:** 16384 × 6144 × 2048
- **Current source tag:** `R55D5A_2_AITER` (promoted in R55)
- **Current source description:** AITER 256×256 .co (R55 D-5A_2 PROMOTE; perf claw-back)
- **Current source mechanism note:** +1.81-20.99pp perf claw-back during 36→42/42 leaderboard close
- **Current binary path:** `AITER_SHIM` → `/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co` via R50D shim `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so`
- **Current pct_comp** (R60 reviewer 10-run p50): **111.98%**
- **Current TFLOPS** (R60 p50): 3412.8
- **Strict-VC status:** PASS_10/10 (n_OK=10/10, wcf_max=0.0000, fin_min=1.0000)
- **Bit-determinism status:** HELD (wcf_max=0.0)
- **Tried axes (chronological by R-round):**
  - R50-R54: prior HK / partial AITER source
  - R55 (D5A_2): perf claw-back PROMOTE; AITER 256×256 .co; +1.81-20.99pp
  - R56-R60: R50D shim AS-IS; PASS_10/10 HELD bit-deterministic
- **Closed axes (specific to this cell):**
  - (no cell-specific axis closures beyond global axes; current AITER 256×256 .co source held since promote round; HK alt-tile axes globally closed for AITER cells)
- **Why current source is best:** AITER 256×256 .co (R50D shim AS-IS) is best because (a) R50-R56 AITER promote era established AITER 256×256 strictly dominant for this M-N-K bucket; (b) HK alt-tiles for AITER cells closed across R57-R59; (c) cell is bit-deterministic (wcf_max=0.0) at 111.98% (R60).

### Cell 27: (16,384, 6,144, 4,096) — K-bucket: K=4096

- **Cohort:** K-bucket: K=4096
- **Shape:** 16384 × 6144 × 4096
- **Current source tag:** `R55E4_1_AITER` (promoted in R55)
- **Current source description:** AITER 256×256 .co (R55 E-4_1 PROMOTE NEW VC rescue)
- **Current source mechanism note:** NEW strict-VC unlock; 36→42/42 first 100% leaderboard
- **Current binary path:** `AITER_SHIM` → `/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co` via R50D shim `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so`
- **Current pct_comp** (R60 reviewer 10-run p50): **114.46%**
- **Current TFLOPS** (R60 p50): 4627.0
- **Strict-VC status:** PASS_10/10 (n_OK=10/10, wcf_max=0.0000, fin_min=1.0000)
- **Bit-determinism status:** HELD (wcf_max=0.0)
- **Tried axes (chronological by R-round):**
  - R50-R54: cell remained outside strict-VC pool (cohort-race / wrong-output / under HK source)
  - R55 (E4_1): NEW VC unlock via AITER 256×256 .co dlopen (R50D shim AS-IS)
  - R56-R60: R50D shim AS-IS; PASS_10/10 HELD bit-deterministic
- **Closed axes (specific to this cell):**
  - (no cell-specific axis closures beyond global axes; current AITER 256×256 .co source held since promote round; HK alt-tile axes globally closed for AITER cells)
- **Why current source is best:** AITER 256×256 .co (R50D shim AS-IS) is best because (a) R50-R56 AITER promote era established AITER 256×256 strictly dominant for this M-N-K bucket; (b) HK alt-tiles for AITER cells closed across R57-R59; (c) cell is bit-deterministic (wcf_max=0.0) at 114.46% (R60).

### Cell 28: (16,384, 14,336, 2,048) — K-bucket: K=2048

- **Cohort:** K-bucket: K=2048
- **Shape:** 16384 × 14336 × 2048
- **Current source tag:** `R55E3_1_AITER` (promoted in R55)
- **Current source description:** AITER 256×256 .co (R55 E-3_1 PROMOTE NEW VC rescue)
- **Current source mechanism note:** NEW strict-VC unlock; 36→42/42 first 100% leaderboard
- **Current binary path:** `AITER_SHIM` → `/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co` via R50D shim `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so`
- **Current pct_comp** (R60 reviewer 10-run p50): **106.43%**
- **Current TFLOPS** (R60 p50): 3513.5
- **Strict-VC status:** PASS_10/10 (n_OK=10/10, wcf_max=0.0000, fin_min=1.0000)
- **Bit-determinism status:** HELD (wcf_max=0.0)
- **Tried axes (chronological by R-round):**
  - R50-R54: cell remained outside strict-VC pool (cohort-race / wrong-output / under HK source)
  - R55 (E3_1): NEW VC unlock via AITER 256×256 .co dlopen (R50D shim AS-IS)
  - R56-R60: R50D shim AS-IS; PASS_10/10 HELD bit-deterministic
- **Closed axes (specific to this cell):**
  - (no cell-specific axis closures beyond global axes; current AITER 256×256 .co source held since promote round; HK alt-tile axes globally closed for AITER cells)
- **Why current source is best:** AITER 256×256 .co (R50D shim AS-IS) is best because (a) R50-R56 AITER promote era established AITER 256×256 strictly dominant for this M-N-K bucket; (b) HK alt-tiles for AITER cells closed across R57-R59; (c) cell is bit-deterministic (wcf_max=0.0) at 106.43% (R60).

### Cell 29: (16,384, 14,336, 4,096) — K-bucket: K=4096

- **Cohort:** K-bucket: K=4096
- **Shape:** 16384 × 14336 × 4096
- **Current source tag:** `R55E3_2_AITER` (promoted in R55)
- **Current source description:** AITER 256×256 .co (R55 E-3_2 PROMOTE NEW VC rescue)
- **Current source mechanism note:** NEW strict-VC unlock; 36→42/42 first 100% leaderboard
- **Current binary path:** `AITER_SHIM` → `/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co` via R50D shim `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so`
- **Current pct_comp** (R60 reviewer 10-run p50): **108.38%**
- **Current TFLOPS** (R60 p50): 4612.4
- **Strict-VC status:** PASS_10/10 (n_OK=10/10, wcf_max=0.0000, fin_min=1.0000)
- **Bit-determinism status:** HELD (wcf_max=0.0)
- **Tried axes (chronological by R-round):**
  - R50-R54: cell remained outside strict-VC pool (cohort-race / wrong-output / under HK source)
  - R55 (E3_2): NEW VC unlock via AITER 256×256 .co dlopen (R50D shim AS-IS)
  - R56-R60: R50D shim AS-IS; PASS_10/10 HELD bit-deterministic
- **Closed axes (specific to this cell):**
  - (no cell-specific axis closures beyond global axes; current AITER 256×256 .co source held since promote round; HK alt-tile axes globally closed for AITER cells)
- **Why current source is best:** AITER 256×256 .co (R50D shim AS-IS) is best because (a) R50-R56 AITER promote era established AITER 256×256 strictly dominant for this M-N-K bucket; (b) HK alt-tiles for AITER cells closed across R57-R59; (c) cell is bit-deterministic (wcf_max=0.0) at 108.38% (R60).

### Cell 30: (16,384, 28,672, 2,048) — K-bucket: K=2048

- **Cohort:** K-bucket: K=2048
- **Shape:** 16384 × 28672 × 2048
- **Current source tag:** `R55E3_3_AITER` (promoted in R55)
- **Current source description:** AITER 256×256 .co (R55 E-3_3 PROMOTE NEW VC rescue)
- **Current source mechanism note:** NEW strict-VC unlock; 36→42/42 first 100% leaderboard
- **Current binary path:** `AITER_SHIM` → `/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co` via R50D shim `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so`
- **Current pct_comp** (R60 reviewer 10-run p50): **102.57%**
- **Current TFLOPS** (R60 p50): 3571.8
- **Strict-VC status:** PASS_10/10 (n_OK=10/10, wcf_max=0.0000, fin_min=1.0000)
- **Bit-determinism status:** HELD (wcf_max=0.0)
- **Tried axes (chronological by R-round):**
  - R50-R54: cell remained outside strict-VC pool (cohort-race / wrong-output / under HK source)
  - R55 (E3_3): NEW VC unlock via AITER 256×256 .co dlopen (R50D shim AS-IS)
  - R56-R60: R50D shim AS-IS; PASS_10/10 HELD bit-deterministic
- **Closed axes (specific to this cell):**
  - (no cell-specific axis closures beyond global axes; current AITER 256×256 .co source held since promote round; HK alt-tile axes globally closed for AITER cells)
- **Why current source is best:** AITER 256×256 .co (R50D shim AS-IS) is best because (a) R50-R56 AITER promote era established AITER 256×256 strictly dominant for this M-N-K bucket; (b) HK alt-tiles for AITER cells closed across R57-R59; (c) cell is bit-deterministic (wcf_max=0.0) at 102.57% (R60).

### Cell 31: (16,384, 28,672, 4,096) — K-bucket: K=4096

- **Cohort:** K-bucket: K=4096
- **Shape:** 16384 × 28672 × 4096
- **Current source tag:** `R55E3_4_AITER` (promoted in R55)
- **Current source description:** AITER 256×256 .co (R55 E-3_4 PROMOTE NEW VC rescue)
- **Current source mechanism note:** NEW strict-VC unlock; 36→42/42 first 100% leaderboard
- **Current binary path:** `AITER_SHIM` → `/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co` via R50D shim `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so`
- **Current pct_comp** (R60 reviewer 10-run p50): **104.18%**
- **Current TFLOPS** (R60 p50): 4596.3
- **Strict-VC status:** PASS_10/10 (n_OK=10/10, wcf_max=0.0000, fin_min=1.0000)
- **Bit-determinism status:** HELD (wcf_max=0.0)
- **Tried axes (chronological by R-round):**
  - R50-R54: cell remained outside strict-VC pool (cohort-race / wrong-output / under HK source)
  - R55 (E3_4): NEW VC unlock via AITER 256×256 .co dlopen (R50D shim AS-IS)
  - R56-R60: R50D shim AS-IS; PASS_10/10 HELD bit-deterministic
- **Closed axes (specific to this cell):**
  - (no cell-specific axis closures beyond global axes; current AITER 256×256 .co source held since promote round; HK alt-tile axes globally closed for AITER cells)
- **Why current source is best:** AITER 256×256 .co (R50D shim AS-IS) is best because (a) R50-R56 AITER promote era established AITER 256×256 strictly dominant for this M-N-K bucket; (b) HK alt-tiles for AITER cells closed across R57-R59; (c) cell is bit-deterministic (wcf_max=0.0) at 104.18% (R60).

### Cell 32: (28,672, 4,096, 8,192) — K-bucket: K=8192

- **Cohort:** K-bucket: K=8192
- **Shape:** 28672 × 4096 × 8192
- **Current source tag:** `R54E1_3_AITER` (promoted in R54)
- **Current source description:** AITER 256×256 .co (R54 E-1_3 PROMOTE NEW VC rescue)
- **Current source mechanism note:** NEW strict-VC unlock via R50D shim
- **Current binary path:** `AITER_SHIM` → `/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co` via R50D shim `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so`
- **Current pct_comp** (R60 reviewer 10-run p50): **105.96%**
- **Current TFLOPS** (R60 p50): 5096.6
- **Strict-VC status:** PASS_10/10 (n_OK=10/10, wcf_max=0.0000, fin_min=1.0000)
- **Bit-determinism status:** HELD (wcf_max=0.0)
- **Tried axes (chronological by R-round):**
  - R50-R53: cell remained outside strict-VC (NEW VC rescue cohort)
  - R54 (E1_3): NEW VC unlock via AITER 256×256 .co dlopen (R50D shim AS-IS)
  - R55-R60: R50D shim AS-IS; PASS_10/10 HELD bit-deterministic
- **Closed axes (specific to this cell):**
  - (no cell-specific axis closures beyond global axes; current AITER 256×256 .co source held since promote round; HK alt-tile axes globally closed for AITER cells)
- **Why current source is best:** AITER 256×256 .co (R50D shim AS-IS) is best because (a) R50-R56 AITER promote era established AITER 256×256 strictly dominant for this M-N-K bucket; (b) HK alt-tiles for AITER cells closed across R57-R59; (c) cell is bit-deterministic (wcf_max=0.0) at 105.96% (R60).

### Cell 33: (28,672, 4,096, 16,384) — K-bucket: K=16384

- **Cohort:** K-bucket: K=16384
- **Shape:** 28672 × 4096 × 16384
- **Current source tag:** `R51D3_AITER` (promoted in R51)
- **Current source description:** AITER 256×256 .co (R51 D-3 PROMOTE)
- **Current source mechanism note:** R50D shim AS-IS reuse (1st AS-IS round); shape-generic
- **Current binary path:** `AITER_SHIM` → `/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co` via R50D shim `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so`
- **Current pct_comp** (R60 reviewer 10-run p50): **103.93%**
- **Current TFLOPS** (R60 p50): 5560.7
- **Strict-VC status:** PASS_10/10 (n_OK=10/10, wcf_max=0.0000, fin_min=1.0000)
- **Bit-determinism status:** HELD (wcf_max=0.0)
- **Tried axes (chronological by R-round):**
  - R50: R50D breakthrough on (4096,32768,28672)
  - R51 (D3): PROMOTE; R50D shim AS-IS shape-generic dispatch
  - R52-R60: R50D shim AS-IS; PASS_10/10 HELD bit-deterministic
- **Closed axes (specific to this cell):**
  - (no cell-specific axis closures beyond global axes; current AITER 256×256 .co source held since promote round; HK alt-tile axes globally closed for AITER cells)
- **Why current source is best:** AITER 256×256 .co (R50D shim AS-IS) is best because (a) R50-R56 AITER promote era established AITER 256×256 strictly dominant for this M-N-K bucket; (b) HK alt-tiles for AITER cells closed across R57-R59; (c) cell is bit-deterministic (wcf_max=0.0) at 103.93% (R60).

### Cell 34: (28,672, 32,768, 4,096) — K-bucket: K=4096

- **Cohort:** K-bucket: K=4096
- **Shape:** 28672 × 32768 × 4096
- **Current source tag:** `R56G2_L4_AITER` (promoted in R56)
- **Current source description:** AITER 256×256 .co (R56 G-2 L4 PROMOTE; 64×1024→256×256 swap)
- **Current source mechanism note:** cluster A/B perf claw-back; +117.22pp aggregate over G-1+G-2
- **Current binary path:** `AITER_SHIM` → `/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co` via R50D shim `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so`
- **Current pct_comp** (R60 reviewer 10-run p50): **102.12%**
- **Current TFLOPS** (R60 p50): 4561.1
- **Strict-VC status:** PASS_10/10 (n_OK=10/10, wcf_max=0.0000, fin_min=1.0000)
- **Bit-determinism status:** HELD (wcf_max=0.0)
- **Tried axes (chronological by R-round):**
  - R50: AITER .co dlopen via R50D shim → PROMOTE
  - R51-R55: prior 64×1024 dispatch held
  - R56 (G-1/G-2): 64×1024→256×256 swap PROMOTE (cluster A/B perf claw-back; +117.22pp aggregate)
  - R57-R60: R50D shim AS-IS; binary unchanged
- **Closed axes (specific to this cell):**
  - (no cell-specific axis closures beyond global axes; current AITER 256×256 .co source held since promote round; HK alt-tile axes globally closed for AITER cells)
- **Why current source is best:** AITER 256×256 .co is best because R56 G-1/G-2/G-4 PROMOTEs delivered +117.22pp+ aggregate cluster claw-back over prior 64×1024 dispatch; 256×256 strictly best for this M-N-K bucket; bit-deterministic at 102.12%.

### Cell 35: (32,768, 4,096, 2,048) — K-bucket: K=2048

- **Cohort:** K-bucket: K=2048
- **Shape:** 32768 × 4096 × 2048
- **Current source tag:** `R54E1_1_AITER` (promoted in R54)
- **Current source description:** AITER 256×256 .co (R54 E-1_1 PROMOTE NEW VC rescue)
- **Current source mechanism note:** NEW strict-VC unlock via R50D shim
- **Current binary path:** `AITER_SHIM` → `/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co` via R50D shim `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so`
- **Current pct_comp** (R60 reviewer 10-run p50): **107.37%**
- **Current TFLOPS** (R60 p50): 3362.5
- **Strict-VC status:** PASS_10/10 (n_OK=10/10, wcf_max=0.0000, fin_min=1.0000)
- **Bit-determinism status:** HELD (wcf_max=0.0)
- **Tried axes (chronological by R-round):**
  - R50-R53: cell remained outside strict-VC (NEW VC rescue cohort)
  - R54 (E1_1): NEW VC unlock via AITER 256×256 .co dlopen (R50D shim AS-IS)
  - R55-R60: R50D shim AS-IS; PASS_10/10 HELD bit-deterministic
- **Closed axes (specific to this cell):**
  - (no cell-specific axis closures beyond global axes; current AITER 256×256 .co source held since promote round; HK alt-tile axes globally closed for AITER cells)
- **Why current source is best:** AITER 256×256 .co (R50D shim AS-IS) is best because (a) R50-R56 AITER promote era established AITER 256×256 strictly dominant for this M-N-K bucket; (b) HK alt-tiles for AITER cells closed across R57-R59; (c) cell is bit-deterministic (wcf_max=0.0) at 107.37% (R60).

### Cell 36: (32,768, 4,096, 3,072) — K-bucket: K=3072

- **Cohort:** K-bucket: K=3072
- **Shape:** 32768 × 4096 × 3072
- **Current source tag:** `R54E1_2_AITER` (promoted in R54)
- **Current source description:** AITER 256×256 .co (R54 E-1_2 PROMOTE NEW VC rescue)
- **Current source mechanism note:** NEW strict-VC unlock via R50D shim
- **Current binary path:** `AITER_SHIM` → `/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co` via R50D shim `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so`
- **Current pct_comp** (R60 reviewer 10-run p50): **118.59%**
- **Current TFLOPS** (R60 p50): 4305.6
- **Strict-VC status:** PASS_10/10 (n_OK=10/10, wcf_max=0.0000, fin_min=1.0000)
- **Bit-determinism status:** HELD (wcf_max=0.0)
- **Tried axes (chronological by R-round):**
  - R50-R53: cell remained outside strict-VC (NEW VC rescue cohort)
  - R54 (E1_2): NEW VC unlock via AITER 256×256 .co dlopen (R50D shim AS-IS)
  - R55-R60: R50D shim AS-IS; PASS_10/10 HELD bit-deterministic
- **Closed axes (specific to this cell):**
  - (no cell-specific axis closures beyond global axes; current AITER 256×256 .co source held since promote round; HK alt-tile axes globally closed for AITER cells)
- **Why current source is best:** AITER 256×256 .co (R50D shim AS-IS) is best because (a) R50-R56 AITER promote era established AITER 256×256 strictly dominant for this M-N-K bucket; (b) HK alt-tiles for AITER cells closed across R57-R59; (c) cell is bit-deterministic (wcf_max=0.0) at 118.59% (R60).

### Cell 37: (32,768, 4,096, 7,168) — K-bucket: K=7168

- **Cohort:** K-bucket: K=7168
- **Shape:** 32768 × 4096 × 7168
- **Current source tag:** `R54D4A_2_AITER` (promoted in R54)
- **Current source description:** AITER 256×256 .co (R54 D-4A_2 PROMOTE)
- **Current source mechanism note:** HK→AITER swap; R50D shim AS-IS (4th)
- **Current binary path:** `AITER_SHIM` → `/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co` via R50D shim `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so`
- **Current pct_comp** (R60 reviewer 10-run p50): **106.14%**
- **Current TFLOPS** (R60 p50): 4953.4
- **Strict-VC status:** PASS_10/10 (n_OK=10/10, wcf_max=0.0000, fin_min=1.0000)
- **Bit-determinism status:** HELD (wcf_max=0.0)
- **Tried axes (chronological by R-round):**
  - R40B-R53: HK 256×256 source (left +17-28pp perf on table)
  - R54 (D4A_2): HK→AITER 256×256 swap PROMOTE +17-28pp
  - R55-R60: R50D shim AS-IS; PASS_10/10 HELD bit-deterministic
- **Closed axes (specific to this cell):**
  - (no cell-specific axis closures beyond global axes; current AITER 256×256 .co source held since promote round; HK alt-tile axes globally closed for AITER cells)
- **Why current source is best:** AITER 256×256 .co (R50D shim AS-IS) is best because (a) R50-R56 AITER promote era established AITER 256×256 strictly dominant for this M-N-K bucket; (b) HK alt-tiles for AITER cells closed across R57-R59; (c) cell is bit-deterministic (wcf_max=0.0) at 106.14% (R60).

### Cell 38: (32,768, 4,096, 14,336) — K-bucket: K=14336

- **Cohort:** K-bucket: K=14336
- **Shape:** 32768 × 4096 × 14336
- **Current source tag:** `R56G1_L2_AITER` (promoted in R56)
- **Current source description:** AITER 256×256 .co (R56 G-1 L2 PROMOTE; 64×1024→256×256 swap)
- **Current source mechanism note:** cluster A/B perf claw-back; +117.22pp aggregate over G-1+G-2
- **Current binary path:** `AITER_SHIM` → `/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co` via R50D shim `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so`
- **Current pct_comp** (R60 reviewer 10-run p50): **103.99%**
- **Current TFLOPS** (R60 p50): 5431.7
- **Strict-VC status:** PASS_10/10 (n_OK=10/10, wcf_max=0.0000, fin_min=1.0000)
- **Bit-determinism status:** HELD (wcf_max=0.0)
- **Tried axes (chronological by R-round):**
  - R50: AITER .co dlopen via R50D shim → PROMOTE
  - R51-R55: prior 64×1024 dispatch held
  - R56 (G-1/G-2): 64×1024→256×256 swap PROMOTE (cluster A/B perf claw-back; +117.22pp aggregate)
  - R57-R60: R50D shim AS-IS; binary unchanged
- **Closed axes (specific to this cell):**
  - (no cell-specific axis closures beyond global axes; current AITER 256×256 .co source held since promote round; HK alt-tile axes globally closed for AITER cells)
- **Why current source is best:** AITER 256×256 .co is best because R56 G-1/G-2/G-4 PROMOTEs delivered +117.22pp+ aggregate cluster claw-back over prior 64×1024 dispatch; 256×256 strictly best for this M-N-K bucket; bit-deterministic at 103.99%.

### Cell 39: (32,768, 6,144, 2,048) — K-bucket: K=2048

- **Cohort:** K-bucket: K=2048
- **Shape:** 32768 × 6144 × 2048
- **Current source tag:** `R55D5B_3_AITER` (promoted in R55)
- **Current source description:** AITER 256×256 .co (R55 D-5B_3 PROMOTE; perf claw-back)
- **Current source mechanism note:** +1.81-20.99pp perf claw-back during 36→42/42 leaderboard close
- **Current binary path:** `AITER_SHIM` → `/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co` via R50D shim `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so`
- **Current pct_comp** (R60 reviewer 10-run p50): **106.68%**
- **Current TFLOPS** (R60 p50): 3456.4
- **Strict-VC status:** PASS_10/10 (n_OK=10/10, wcf_max=0.0000, fin_min=1.0000)
- **Bit-determinism status:** HELD (wcf_max=0.0)
- **Tried axes (chronological by R-round):**
  - R50-R54: prior HK / partial AITER source
  - R55 (D5B_3): perf claw-back PROMOTE; AITER 256×256 .co; +1.81-20.99pp
  - R56-R60: R50D shim AS-IS; PASS_10/10 HELD bit-deterministic
- **Closed axes (specific to this cell):**
  - (no cell-specific axis closures beyond global axes; current AITER 256×256 .co source held since promote round; HK alt-tile axes globally closed for AITER cells)
- **Why current source is best:** AITER 256×256 .co (R50D shim AS-IS) is best because (a) R50-R56 AITER promote era established AITER 256×256 strictly dominant for this M-N-K bucket; (b) HK alt-tiles for AITER cells closed across R57-R59; (c) cell is bit-deterministic (wcf_max=0.0) at 106.68% (R60).

### Cell 40: (32,768, 14,336, 2,048) — L3 (attention-near-gate HK)

- **Cohort:** L3 (attention-near-gate HK)
- **Shape:** 32768 × 14336 × 2048
- **Current source tag:** `R40B` (promoted in R40B)
- **Current source description:** HK kernel R40B 256×256
- **Current source mechanism note:** in-tree HK build with R39A TAIL_SCALE_CLAMP + R44A back-edge drain + R44D FINITE_GATE 0.97
- **Current binary path:** `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/build_R40B/tk_mxfp4_gluon_cpp_n14336_k2048_ts_gm6_v12_dc_pfoff4_R40B_safe.cpython-310-x86_64-linux-gnu.so`
- **Current pct_comp** (R60 reviewer 10-run p50): **100.45%**
- **Current TFLOPS** (R60 p50): 3366.6
- **Strict-VC status:** PASS_10/10 (n_OK=10/10, wcf_max=0.0122, fin_min=0.9848)
- **Bit-determinism status:** NOT-HELD (wcf_max=0.0122)
- **Tried axes (chronological by R-round):**
  - R40 baseline: HK R40B 256×256 with R39A + R44A + R44D ports → PROMOTE
  - R50-R57: HK R40B kept (cohort-race-bound; AITER alt-tile efficiencies lower)
  - R55 / R57 / R58: AITER 256×256 / 192×256 / 128×256 alt-tile probes → all DEAD (HK strictly best)
  - R58 P-3: AITER 128×256 swap probe → ACCEPT_FALLBACK -25.04pp catastrophic
  - R58 K-1: cohort-race tail-draw under 5-seed sweep → fin_min=0.911 lone VC drop
  - R59 K-1: fresh INDEPENDENT 10-seed sweep on UNCHANGED binary → PASS_10/10 fin_min=0.988 (RECOVERED)
  - R59 J-2 S-1: AITER 96×640 alt-tile → SMOKE_DEAD 63.19% (-37.30pp)
  - R59 J-2 S-2: AITER 64×1024 alt-tile → SMOKE_DEAD 91.73% (-8.76pp)
  - R60 K-1: 3rd-consecutive Opt Y sweep on DISJOINT seeds [202..2020 step 202] → PASS_10/10 fin_min=0.985
- **Closed axes (specific to this cell):**
  - AITER 96×640 alt-tile — R59 J-2 S-1 SMOKE_DEAD 63.19% (-37.30pp)
  - AITER 64×1024 alt-tile — R59 J-2 S-2 SMOKE_DEAD 91.73% (-8.76pp)
  - AITER 128×256 alt-tile — CLOSED R55/R57/R58 (HK strictly best)
  - AITER 192×256 alt-tile — CLOSED via R57 H-2 (-8 to -14pp on HK kept-cell pool)
  - AITER 256×256 alt-tile — CLOSED (HK R40B 256×256 strictly best)
  - Opt W (HK kernel rebuild, FINITE_GATE 0.97→0.95) — PERMANENTLY DEPRIORITIZED in R60
- **Why current source is best:** HK R40B 256×256 is best because all AITER alt-tiles (96×640, 64×1024, 128×256, 192×256, 256×256) closed across R55/R57/R58/R59; cohort-race surface validated as ≤ 1/3 per-sweep tail-draw rate via 3-consecutive Opt Y PASS sequence R58→R59→R60 with high-confidence recovery; Opt W permanently deprioritized in R60.

### Cell 41: (32,768, 28,672, 2,048) — K-bucket: K=2048

- **Cohort:** K-bucket: K=2048
- **Shape:** 32768 × 28672 × 2048
- **Current source tag:** `R55D5B_2_AITER` (promoted in R55)
- **Current source description:** AITER 256×256 .co (R55 D-5B_2 PROMOTE; perf claw-back)
- **Current source mechanism note:** +1.81-20.99pp perf claw-back during 36→42/42 leaderboard close
- **Current binary path:** `AITER_SHIM` → `/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co` via R50D shim `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so`
- **Current pct_comp** (R60 reviewer 10-run p50): **103.21%**
- **Current TFLOPS** (R60 p50): 3461.0
- **Strict-VC status:** PASS_10/10 (n_OK=10/10, wcf_max=0.0000, fin_min=1.0000)
- **Bit-determinism status:** HELD (wcf_max=0.0)
- **Tried axes (chronological by R-round):**
  - R50-R54: prior HK / partial AITER source
  - R55 (D5B_2): perf claw-back PROMOTE; AITER 256×256 .co; +1.81-20.99pp
  - R56-R60: R50D shim AS-IS; PASS_10/10 HELD bit-deterministic
- **Closed axes (specific to this cell):**
  - (no cell-specific axis closures beyond global axes; current AITER 256×256 .co source held since promote round; HK alt-tile axes globally closed for AITER cells)
- **Why current source is best:** AITER 256×256 .co (R50D shim AS-IS) is best because (a) R50-R56 AITER promote era established AITER 256×256 strictly dominant for this M-N-K bucket; (b) HK alt-tiles for AITER cells closed across R57-R59; (c) cell is bit-deterministic (wcf_max=0.0) at 103.21% (R60).

### Cell 42: (128,256, 32,768, 4,096) — K-bucket: K=4096

- **Cohort:** K-bucket: K=4096
- **Shape:** 128256 × 32768 × 4096
- **Current source tag:** `R56G2_L3_AITER` (promoted in R56)
- **Current source description:** AITER 256×256 .co (R56 G-2 L3 PROMOTE; 64×1024→256×256 swap)
- **Current source mechanism note:** cluster A/B perf claw-back; +117.22pp aggregate over G-1+G-2
- **Current binary path:** `AITER_SHIM` → `/shared_nfs/kyle/test/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co` via R50D shim `/shared_nfs/kyle/test/HipKittens/analysis/fp8_gemm/mi350x/build_R50D/R50D_aiter_shim.cpython-310-x86_64-linux-gnu.so`
- **Current pct_comp** (R60 reviewer 10-run p50): **102.62%**
- **Current TFLOPS** (R60 p50): 4655.1
- **Strict-VC status:** PASS_10/10 (n_OK=10/10, wcf_max=0.0000, fin_min=1.0000)
- **Bit-determinism status:** HELD (wcf_max=0.0)
- **Tried axes (chronological by R-round):**
  - R50: AITER .co dlopen via R50D shim → PROMOTE
  - R51-R55: prior 64×1024 dispatch held
  - R56 (G-1/G-2): 64×1024→256×256 swap PROMOTE (cluster A/B perf claw-back; +117.22pp aggregate)
  - R57-R60: R50D shim AS-IS; binary unchanged
- **Closed axes (specific to this cell):**
  - (no cell-specific axis closures beyond global axes; current AITER 256×256 .co source held since promote round; HK alt-tile axes globally closed for AITER cells)
- **Why current source is best:** AITER 256×256 .co is best because R56 G-1/G-2/G-4 PROMOTEs delivered +117.22pp+ aggregate cluster claw-back over prior 64×1024 dispatch; 256×256 strictly best for this M-N-K bucket; bit-deterministic at 102.62%.

---

## §2 Aggregate axis-closure summary

Cumulative axis closures across the 42-cell production surface (R45-R60). Cells affected counted by participation in the SMOKE/falsification round that delivered the closure. Rows sorted by closure round.

| # | Closed axis | Closure round | Cells affected | Closure reason / DEAD pp delta | Cross-reference |
|---|-------------|---------------|----------------|--------------------------------|-----------------|
| 1 | Prefetch-gate variants (Opt A, R43) | R43 | All HK pool | DEAD: 6 prefetch-gating variants + SRD-widen all DEAD; CRASH at LDS double-buffer / step34 ordering | `R43_INTEGRATION_VERDICT.md` |
| 2 | VGPR-PF keepalive (Opt B, R43) | R43 | All HK pool | DEAD: compiler killed prefetch VGPRs across asm boundaries; revivable via `+v` keepalive (R45C) but no production path | `R43_INTEGRATION_VERDICT.md`; `project_mxfp4_vgprpf_compiler_bug.md` |
| 3 | R39A TAIL_SCALE_CLAMP generalization to wcf-flake intermediate-K cohort | R46 | 5 wcf-flake shapes | DEAD: clamp makes wcf 3-5× WORSE; R38 misalign theory does NOT generalize | `project_mxfp4_R46A_tail_clamp_falsified.md` |
| 4 | External vmcnt fence positioning (R47A, R49C) | R47/R49 | K=28672 cohort | DEAD: 3+4 fence positions on R46B 3-buffer ALL re-trigger HSA aperture fault or unmoved Jaccard; race lives INSIDE MFMA pipeline | `R47_INTEGRATION_VERDICT.md`; `R49_INTEGRATION_VERDICT.md` |
| 5 | M0 fresh-set transfer to FUSED+TS (R47B) | R47 | K=28672 cohort | DEAD: regresses ALL shapes incl R44-VC baselines; M0 load-bearing scoped to VGPR-PF only | `R47_INTEGRATION_VERDICT.md` |
| 6 | Asm-block split step3+step4 (R48A) | R48 | K=28672 cohort | DEAD: ISA boundary unchanged; compiler IPRA/RA before asm boundary | `R48_INTEGRATION_VERDICT.md` |
| 7 | PF_MPT depth knob (R48C) | R48 | K=28672 cohort | DEAD: PF_MPT is coverage not depth; PF_MPT>4 CRASH first iter | `R48_INTEGRATION_VERDICT.md` |
| 8 | aiter vmcnt(15) single-knob port (R49A) | R49 | (4096,32768,28672) | DEAD: doesn't move cohort race; true differentiator is MFMA↔ds_read 1:3 interleaving | `R49_INTEGRATION_VERDICT.md` |
| 9 | MFMA↔ds_read 1:3 interleave (R50A) | R50 | All HK pool | DEAD: ISA-verified 1:4 spread emit but cohort race unmoved; AGPR forwarding race INSIDE MFMA pipeline | `project_mxfp4_R50A_interleave_axis_closed.md` |
| 10 | Perf claw-back gm×lgk×pfoff sweep (R50C) | R50 | 14 sub-90% shapes | DEAD: 13 LOSE; only `4096x28672x32768` +2.90% at 5-run REVERTED at 10-run | `project_mxfp4_R50C_perf_clawback_saturated.md` |
| 11 | HK 256×256 vs AITER 256×256 for K=2048 N=14336 cohort | R55 | L3 (32768,14336,2048) | HK strictly best — AITER alt-tiles all DEAD on this single cell | `R55_INTEGRATION_VERDICT.md` |
| 12 | AITER 128×512 alt-tile on L8 | R56 | L8 (4096,32768,128256) | SMOKE_DEAD -13.31pp | `R56_INTEGRATION_VERDICT.md` G-3 |
| 13 | AITER 192×256 alt-tile on L8 | R56 | L8 (4096,32768,128256) | SMOKE_DEAD -11.84pp | `R56_INTEGRATION_VERDICT.md` |
| 14 | AITER 224×256 alt-tile on L8 | R57 | L8 (4096,32768,128256) | SMOKE_DEAD -10.69pp (Opt K) | `R57_INTEGRATION_VERDICT.md` |
| 15 | AITER 192×256 alt-tile (HK kept-cell pool) | R57 | All 3 HK kept cells (R57 era) | SMOKE_DEAD -8 to -14pp on HK kept-cells | `R57_INTEGRATION_VERDICT.md` H-2 |
| 16 | L8 AITER 96×640 + 64×1024 alt-tiles | R57 | L8 (4096,32768,128256) | CLOSED in carry-forward (no separate SMOKE — implied DEAD given 128×512 / 192×256 / 224×256 all DEAD) | `R57_INTEGRATION_VERDICT.md` |
| 17 | HK 256×256 lgk2 v12 axis on L8 K=128256 | R58 | L8 (4096,32768,128256) | CLOSED by correctness — BOTH R40B + R37 fallbacks WRONG_OUTPUT (fin=0.78-0.80, wcf=0.07-0.13) for K=128256; R39A/R44A/R44D never ported | `R58_INTEGRATION_VERDICT.md` Opt O |
| 18 | AITER 128×256 alt-tile on K=2048 HK survivors | R58 | (16384,4096,2048) + (32768,14336,2048) | ACCEPT_FALLBACK -3.68pp (P-1) and -25.04pp catastrophic (P-3) | `R58_INTEGRATION_VERDICT.md` P-1, P-3 |
| 19 | L1 AITER 96×640 alt-tile | R59 | L1 (4096,32768,14336) | SMOKE_DEAD 87.42% (-12.52pp) | `R59_INTEGRATION_VERDICT.md` J-3 V-1 |
| 20 | L1 AITER 64×1024 alt-tile | R59 | L1 (4096,32768,14336) | SMOKE_DEAD 66.05% (-33.89pp) | `R59_INTEGRATION_VERDICT.md` J-3 V-2 |
| 21 | L3 AITER 96×640 alt-tile | R59 | L3 (32768,14336,2048) | SMOKE_DEAD 63.19% (-37.30pp) | `R59_INTEGRATION_VERDICT.md` J-2 S-1 |
| 22 | L3 AITER 64×1024 alt-tile | R59 | L3 (32768,14336,2048) | SMOKE_DEAD 91.73% (-8.76pp) | `R59_INTEGRATION_VERDICT.md` J-2 S-2 |
| 23 | L1 ITERS=1000 one-off bump | R59 | L1 (4096,32768,14336) | CLOSED by Opt R policy A 4-criterion validity envelope (bit-determinism + ≤0.5pp boundary + dual-protocol documentation + no manifest merge) | `R59_OPT_R_POLICY.md` |
| 24 | Opt W HK kernel rebuild (FINITE_GATE 0.97→0.95) | R60 | L3 (32768,14336,2048) | PERMANENTLY DEPRIORITIZED — 3rd-consecutive Opt Y PASS R58→R59→R60 establishes per-sweep tail-draw rate ≤ 1/3 with high-confidence recovery; cost not justified | `R60_INTEGRATION_VERDICT.md` §6, §12 |
| 25 | Opt X (L1 cross-product alt-tiles or grid swizzle) | R59 | L1 (4096,32768,14336) | CLOSED — L1 alt-tile space EXHAUSTED via #19 + #20 | `R59_INTEGRATION_VERDICT.md` |

**Total cumulative closed axes enumerated above: 25** (representative of the closure documentation; the R45-R60 verdict files contain additional micro-axes — the 25 above are the load-bearing closures cited in publication framing).


---

## §3 Cells by cohort summary

Cohort labels per `R60_OPT_U_DOC_PIVOT.md` §1: L1/L3/L8 are the 3 attention cells with publication-tracked names; the remaining 39 cells are bucketed by K-archetype (very-short-K K≤3072, short-K K=4096, short-mid-K K=6144-8192, mid-K K=14336-16384, long-K K≥28672).

| Cohort | # cells | Sources used | Cohort archetype note |
|--------|---------|--------------|------------------------|
| K-bucket: K=14336 | 2 | 2 AITER + 0 HK | Bulk production cells; all bit-deterministic AITER 256×256 .co dispatched via R50D shim; PASS_10/10 across the cohort |
| K-bucket: K=16384 | 4 | 4 AITER + 0 HK | Bulk production cells; all bit-deterministic AITER 256×256 .co dispatched via R50D shim; PASS_10/10 across the cohort |
| K-bucket: K=2048 | 7 | 6 AITER + 1 HK | Bulk production cells; all bit-deterministic AITER 256×256 .co dispatched via R50D shim; PASS_10/10 across the cohort |
| K-bucket: K=28672 | 2 | 2 AITER + 0 HK | Bulk production cells; all bit-deterministic AITER 256×256 .co dispatched via R50D shim; PASS_10/10 across the cohort |
| K-bucket: K=3072 | 2 | 2 AITER + 0 HK | Bulk production cells; all bit-deterministic AITER 256×256 .co dispatched via R50D shim; PASS_10/10 across the cohort |
| K-bucket: K=32768 | 5 | 5 AITER + 0 HK | Bulk production cells; all bit-deterministic AITER 256×256 .co dispatched via R50D shim; PASS_10/10 across the cohort |
| K-bucket: K=4096 | 9 | 9 AITER + 0 HK | Bulk production cells; all bit-deterministic AITER 256×256 .co dispatched via R50D shim; PASS_10/10 across the cohort |
| K-bucket: K=6144 | 2 | 2 AITER + 0 HK | Bulk production cells; all bit-deterministic AITER 256×256 .co dispatched via R50D shim; PASS_10/10 across the cohort |
| K-bucket: K=7168 | 2 | 2 AITER + 0 HK | Bulk production cells; all bit-deterministic AITER 256×256 .co dispatched via R50D shim; PASS_10/10 across the cohort |
| K-bucket: K=8192 | 4 | 4 AITER + 0 HK | Bulk production cells; all bit-deterministic AITER 256×256 .co dispatched via R50D shim; PASS_10/10 across the cohort |
| L1 (attention-noise-edge) | 1 | 1 AITER + 0 HK | Single noise-edge AITER cell oscillating ±0.10pp around WIN-line under ITERS=500; bit-deterministic; per Opt R policy A |
| L3 (attention-near-gate HK) | 1 | 0 AITER + 1 HK | Single near-gate HK cell at fin_min ~0.985-0.988 under R44D FINITE_GATE 0.97; cohort-race tail-draw ≤ 1/3 sweeps with high-confidence recovery (3-consecutive Opt Y PASS) |
| L8 (attention-LOSE) | 1 | 1 AITER + 0 HK | Single LOSE AITER cell at aiter-internal ceiling for K=128256; HK 256×256 closed by correctness; only Opt T (~3 R-rounds, very low confidence) remains |

**Source breakdown across all 42 cells:**

- AITER 256×256 .co (R50D shim): 39 cells
- AITER 128×256 .co (R50D shim, R58 P-2 promote): 1 cell — `(16384,4096,3072)`
- HK R40B 256×256 in-tree: 2 cells — `(16384,4096,2048)` + `(32768,14336,2048)`
- AITER bit-deterministic share: 40/42 (largest in project history; HELD R58→R59→R60→R60)
- HK pool: 2/42 (smallest in project history; HELD R58→R59→R60)
- R50D AS-IS reuse counter: 12 consecutive rounds (R50→R60); R61 will be 13th if no rebuild

---

## §4 Closed-axis carry-forward (DO NOT propose for R61+)

Verbatim cumulative closed-axis list from `R60_INTEGRATION_VERDICT.md` §12 'R61+ closed-axis carry-forward (DO NOT propose)' merged with `R59_INTEGRATION_VERDICT.md` §'R60+ closed-axis carry-forward'. Reproduced here for archive purposes; this is the canonical 'do not propose' list for R61+ axis decision.

- L1 `(4096,32768,14336)` AITER 96×640 alt-tile — R59 J-3 V-1 SMOKE_DEAD 87.42% (-12.52pp) — `R59_INTEGRATION_VERDICT.md`
- L1 `(4096,32768,14336)` AITER 64×1024 alt-tile — R59 J-3 V-2 SMOKE_DEAD 66.05% (-33.89pp) — `R59_INTEGRATION_VERDICT.md`
- L1 `(4096,32768,14336)` ITERS=1000 one-off bump — CLOSED by Opt R policy A 4-criterion validity envelope (R59) — `R59_OPT_R_POLICY.md`
- L3 `(32768,14336,2048)` AITER 96×640 alt-tile — R59 J-2 S-1 SMOKE_DEAD 63.19% (-37.30pp) — `R59_INTEGRATION_VERDICT.md`
- L3 `(32768,14336,2048)` AITER 64×1024 alt-tile — R59 J-2 S-2 SMOKE_DEAD 91.73% (-8.76pp) — `R59_INTEGRATION_VERDICT.md`
- L3 `(32768,14336,2048)` AITER 128×256 alt-tile — CLOSED in R55/R57/R58 (-pp deltas in respective verdicts)
- L3 `(32768,14336,2048)` AITER 192×256 alt-tile — CLOSED via HK kept-cell pool axis closure (R57 H-2)
- L3 `(32768,14336,2048)` AITER 256×256 alt-tile — CLOSED (HK R40B 256×256 strictly best; AITER 256×256 not preferred)
- L8 `(4096,32768,128256)` AITER 128×512 alt-tile — R56 G-3 SMOKE_DEAD -13.31pp — `R56_INTEGRATION_VERDICT.md`
- L8 `(4096,32768,128256)` AITER 192×256 alt-tile — R56 SMOKE_DEAD -11.84pp — `R56_INTEGRATION_VERDICT.md`
- L8 `(4096,32768,128256)` AITER 224×256 alt-tile — R57 K-Opt K SMOKE_DEAD -10.69pp — `R57_INTEGRATION_VERDICT.md`
- L8 `(4096,32768,128256)` AITER 96×640 alt-tile — CLOSED in R57 carry-forward
- L8 `(4096,32768,128256)` AITER 64×1024 alt-tile — CLOSED in R57 carry-forward
- L8 `(4096,32768,128256)` HK 256×256 lgk2 v12 axis — R58 Opt O CLOSED by correctness (BOTH R40B + R37 fallback HK candidates produce WRONG_OUTPUT fin=0.78-0.80, wcf=0.07-0.13 because R39A/R44A/R44D never ported into K=128256 build) — `R58_INTEGRATION_VERDICT.md`
- HK alt-tile space for K=2048 HK survivors (192×256, 128×256) — CLOSED via R57 H-2 (-8 to -14pp on HK kept-cells) + R58 P-1/P-3 ACCEPT_FALLBACK (P-1 -3.68pp on `(16384,4096,2048)`; P-3 -25.04pp catastrophic on `(32768,14336,2048)`)
- Opt W (HK kernel rebuild, FINITE_GATE 0.97→0.95) — PERMANENTLY DEPRIORITIZED in R60 (3rd-consecutive Opt Y PASS on L3 establishes per-sweep tail-draw ≤ 1/3 with high-confidence recovery; cost not justified)
- Opt X (L1 cross-product alt-tiles or grid swizzle) — CLOSED in R59 (L1 alt-tile space EXHAUSTED)
- (R45-R52 carry-forward, abridged) Prefetch-gate / VGPR-PF keepalive / external fence positioning / aiter vmcnt(15) port / asm-block split / PF_MPT depth knob / 3-buffer rotation external fence — all DEAD across R43-R49; mechanism-axis frontier exhausted on cohort-race surface


**R61+ branching note** (per `R61_DECIDER_PLAN.md` §7): R61 anticipates no new axis closures; the round is methodology + monitoring (Opt U₂ + Opt Z + Opt Y₂). The expected outcome is 0 PROMOTE / 0 SMOKE_DEAD / 2 POLICY_ONLY (Opt U₂ + Opt Z) / 1 MEASUREMENT-ONLY (Opt Y₂). The R61 Opt Y₂ PASS would refine the L3 empirical per-sweep tail-draw rate from ≤ 1/3 to ≤ 1/4 sweeps; the R61 Opt Y₂ FAIL would refine the rate upward to ~1/2 sweeps and reconsider the Opt W PERMANENTLY DEPRIORITIZED classification.


---

## §5 Round-arc context (for cross-reference convenience)

Brief 17-round arc context (R43→R60) for cross-referencing the per-cell tried/closed axes in §1. This summary is a one-line-per-round digest; full per-round detail is in the respective `R*_INTEGRATION_VERDICT.md` files.

| Round | Net VC delta | Net WIN delta | PROMOTE count | Notable axis closures | Notes |
|-------|--------------|---------------|---------------|------------------------|-------|
| R43 | 0 | n/a | 0 | Prefetch-gate, VGPR-PF keepalive, variant-table all DEAD | 3 dead axes; HK pool ceiling 27/42 |
| R44 | +8 | n/a | 8 | TAIL_SCALE_CLAMP (R39A) + back-edge drain (R44A) + FINITE_GATE 0.97 (R44D) | First +8 VC; methodology required |
| R45 | -10 (cohort tail-draw) | n/a | 0 | 10-run @ 80% strict VC mandate | Cohort tail-draw caught; protocol promoted |
| R46 | 0 | n/a | 0 | TAIL_SCALE_CLAMP generalization falsified; 3-buffer rotation partial | R46B bypasses K=28672 CRASH |
| R47 | 0 | n/a | 0 | External fence positioning, M0 transfer to FUSED+TS | 4 dead axes; protocol vindicated |
| R48 | 0 | n/a | 0 | Asm-block split, PF_MPT depth knob | 4th DEAD in last 6 |
| R49 | 0 | n/a | 0 | aiter vmcnt(15) port, embedded vmcnt | 5th DEAD in last 7 |
| R50 | +1 | n/a | 1 | Interleave axis closed, perf claw-back saturated | R50D aiter .co dlopen breakthrough |
| R51 | +1 | +2 perf claw-back | 3 | (none) | R50D shim AS-IS (1st); shape-generic |
| R52 | +5 | +3 perf claw-back | 3 | (none) | Largest VC since R44; K-generic dispatch |
| R53 | -3 strict (-5 cohort + +2 net) | n/a | 8 | (first non-256×256 attempt) | R50D shim tile-generic (96×640, 64×1024) |
| R54 | +3 | +6 perf claw-back +17-28pp | 12 | (none) | AITER bit-det 15→27 (largest single-round expansion) |
| R55 | +6 | +5 perf claw-back +1.81-20.99pp | 11 | HK→AITER axis | **FIRST 100% leaderboard (36→42/42)** |
| R56 | 0 (HELD) | +6 NET WIN cells (LARGEST EVER) | 7 | 64×1024→256×256 swap; 128×512 + 192×256 alt-tiles | 2nd consecutive 100% leaderboard; +198.14pp |
| R57 | 0 (HELD) | +1 NET WIN cell (Opt J ITERS bump) | 1 | 192×256 (HK kept-cell pool); 224×256 (L8); L8 alt-tile fully closed | 3rd consecutive 100% (FIRST 3-IN-A-ROW) |
| R58 | -1 (cohort tail-draw on L3) | +1 (P-2 swap) | 1 | HK 256×256 on L8 K=128256 (correctness) | AITER bit-det 39→40 (largest ever); HK pool 3→2 (smallest ever) |
| R59 | +1 (RECOVERY) | +1 (L1 noise-edge crossback) | 0 (4 SMOKE_DEAD + 1 POLICY_ONLY Opt R) | L1 + L3 96×640 / 64×1024 alt-tiles; ITERS=1000 bump (Opt R) | 4th 100% leaderboard (R58 interrupted streak) |
| R60 | 0 (HELD) | -1 (L1 noise-edge re-flip; predicted) | 0 (1 POLICY_ONLY Opt U + 1 MEASUREMENT Opt Y) | Opt W PERMANENTLY DEPRIORITIZED | 5th 100% leaderboard; 3rd-consecutive Opt Y PASS on L3; 12th consecutive R50D AS-IS |

---

## §6 Closing notes

This appendix is a static reference artifact for the SC25/MICRO25 publication. Future R-rounds (R61+) that change a cell's source binary, perf reading, or axis-closure list should re-emit this artifact rather than edit it in place — the R-tag in the filename (`R61_OPT_Z_PER_SHAPE_DECOMPOSITION.md`) ties the appendix to a specific manifest snapshot. R61 anticipates no manifest binary changes (3rd consecutive binary-identical round expected); accordingly, R62+ would re-emit this artifact only if the R62 round delivers a PROMOTE that changes the source binary on at least one cell.

**Cross-references:**
- `R61_OPT_U2_PUBLICATION_OUTLINE.md` — companion publication-outline artifact (cohort L-2)
- `R61_OPT_Y2_MEASUREMENT.{md,json}` — companion 4th-consecutive cohort-race monitoring measurement (cohort L-1)
- `R60_OPT_U_DOC_PIVOT.md` — R60 doc pivot artifact; this Opt Z appendix complements (does not duplicate) its prose-style structural-ceiling analysis
- `R60_INTEGRATION_VERDICT.md` — R60 reviewer verdict; canonical source for §1 perf readings + §2 axis closures + §4 carry-forward list
- `R60_INTEGRATION_MANIFEST.json` — canonical 42-cell mapping referenced throughout §1
- `R60_INTEGRATION_10RUN.json` — R60 reviewer 10-run measurement file referenced throughout §1 perf fields
- `R59_OPT_R_POLICY.md` — Opt R policy A 4-criterion validity envelope referenced for L1 noise-edge cell

**End of R61 Opt Z per-shape decomposition appendix.**
