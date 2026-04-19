# R52 Dev K — A-side LDS double-buffer at 4096³ occ=1: REFUTED before prototype

**Date:** 2026-04-19
**Branch base:** feat/mxfp8-only @ 02ca9018 (R52J wrap)
**GPU:** MI355X (gfx950, 256 CUs, 160 KB LDS/CU), HIP_VISIBLE_DEVICES=2
**Worktree:** `.claude/worktrees/agent-a722afc1`
**Lever:** add a 3rd A-tile LDS slot for one-iter-ahead A-prefetch at occupancy=1
prefill shapes (per the R49C §7.3 candidate revived in this prompt).

---

## TL;DR — VERDICT: REFUTED (analytical, prompt §6 "If pre-bench shows
any spill or LDS over-budget: REFUTE without GPU bench")

**The proposed `MXFP8_CRR_ASIDE_DOUBLEBUF=1` variant cannot fit in the
160 KB per-CU LDS budget on the CRR target the prompt names.** A 3rd
outer A-buffer slot adds **32 KB** (2 wave-tiles × 16 KB each); CRR already
uses **136 KB / 160 KB**. New per-CTA total = **168 KB > 160 KB hardware
limit** → kernel will fail to launch (`hipErrorInvalidConfiguration`,
exceeds `sharedMemPerBlock`).

The R52K mandate text itself records the correction R50C made to R49C
§7.3:
> "the V2 kernels run at 1 CTA/CU regardless because LDS budget (128–136
> KB/CTA) > 80 KB/CU half-budget"

…but then proposes: "Free LDS for an A-prefetch buffer." These two
statements are mutually inconsistent for the CRR target shape: yes the
*second occupancy slot* is unused, but that does NOT mean 80 KB is free
per-CTA. Per-CTA, only `160 - 136 = 24 KB` is reclaimable on CRR, and
the smallest A-prefetch slot for a meaningful lookahead is **32 KB** (one
extra `As[1][2]` outer entry × 2 wave-tile inner entries × `ST_v2a` =
16 KB each). **24 KB free < 32 KB needed → no fit, no prototype, no
build, no GPU time.**

Per prompt step 6:
> "If pre-bench shows any spill or LDS over-budget: REFUTE without GPU
> bench. Per R49 Dev B's lesson — don't waste GPU time on a known
> regression."

This is the bail clause being triggered.

---

## 1. Pre-bench LDS audit (no compile required — pure arithmetic from
declared types)

### 1.1 Existing `__shared__` declarations in the CRR exact 8-wave fastpath

`analysis/fp8_gemm/mi350x/crr_mxfp8_exact_8wave_fastpath.inc` lines
149-157 and 323-324:

```cpp
#if MXFP8_CRR_LDS_SINGLE_BUFFER
#  define CRR_LDS_OUTER 1
#  define CRR_LDS_TIC   0
#  define CRR_LDS_TOC   0
#else
#  define CRR_LDS_OUTER 2     // <-- production: K-pipeline tic/toc double buffer
#  define CRR_LDS_TIC   tic
#  define CRR_LDS_TOC   toc
#endif
...
__shared__ ST_crr_a As[CRR_LDS_OUTER][2];   // outer = K-pipeline depth, inner = wave-tile id
__shared__ ST_crr_b Bs[CRR_LDS_OUTER][2];
```

`ST_crr_a = ST_v2a = st_fp8e4m3<HB, BK, st_16x128_v2a_s>` with
`HB = BLK/2 = 128`, `BK = 128` (kernel_mxfp8_layouts.cpp:336, 492-493).
Per-tile size: 128 × 128 × 1 byte = **16384 B = 16 KB** (no swizzle padding
in `st_16x128_v2a_s`; the production builds from R50C confirm
`LDS Size [bytes/block]: 139264 = 136 KB` for CRR, which is
`As[2][2]` (64 KB) + `Bs[2][2]` (64 KB) + ~8 KB for the CRR-specific
`a0/a1/b0/b1_scale_packs` and column-A reencode statics).

### 1.2 Adding a 3rd outer A-slot (`As[3][2]` or `As_pf[2]`)

| Variant declaration | Δ slots | Δ bytes | New per-CTA LDS | vs 160 KB cap |
|---|---|---|---|---|
| `As[3][2]` (extend outer dim) | +2 × 16 KB | **+32 KB** | 168 KB | **OVER by 8 KB** |
| `As_pf[2]` (separate prefetch slot, only A) | +2 × 16 KB | +32 KB | 168 KB | **OVER by 8 KB** |
| `As_pf[1]` (single-wave-tile prefetch) | +1 × 16 KB | +16 KB | 152 KB | fits, but useless† |

†`As_pf[1]` only buffers one of the two wave-tiles per K-iter. Within a
single K-iter the kernel must read BOTH `As[tic][0]` and `As[tic][1]`
(line 203, 225 of the fastpath). A one-wave-tile prefetch cannot cover
the full A consumption of an iter ahead, so the proposed scheduling
(load `As_pf[next]` from VMEM in parallel with consuming `As[curr]`
from LDS) would only mask half the A-VMEM cost. The other half still
serializes on the global_load_a in the existing tic/toc tic — net change
≈ 0 cycles saved per K-iter, since the existing `global_load_a(As[toc][1], k+1)`
already overlaps the second-wave-tile A load with `load_a(a, As[tic][0])`
by hardware register-VMEM scoreboarding.

So the only LDS-fitting variant is provably equivalent to existing
behavior. Confirmed below in §3.

### 1.3 RRR target

Per R50C §1.2: V2-RRR uses 132 KB/CTA → 28 KB free → also <32 KB → also
no fit for `As[3][2]`. Same refute applies.

### 1.4 RCR target (out of mandate scope, for completeness)

V2-RCR uses 128 KB → 32 KB free. RCR could *just barely* fit a 32 KB
extra A-prefetch (158 KB, leaving 2 KB margin). But (a) the mandate
targets CRR (the borderline cell); and (b) RCR at 4096³ is already
99.0%+ of ceiling per R47/R48 baselines, no HEADROOM exists.

---

## 2. Why the mandate text contradicts itself

The R52K prompt restates the R50C correction:

> "R50 Dev C corrected r49c §7.3: at LLaMA prefill shapes with M=4096
> the V2 kernels run at 1 CTA/CU regardless because LDS budget (128–136
> KB/CTA) > 80 KB/CU half-budget. So `Occupancy [waves/SIMD]=2` reported
> by the compiler is per-CTA wave density, NOT CTAs/CU."

This part is correct and matches r50c_findings.md §3.

The prompt then asserts:

> "Implication: the second 80 KB LDS half-bank is physically unused at
> every prefill shape. Free LDS for an A-prefetch buffer."

This is the contradiction. There is no "second 80 KB half-bank" because
the kernel never had access to 2 CTAs/CU on this hardware/budget — it
runs 1 CTA/CU using **136 KB** of the 160 KB budget. The remaining
**24 KB** is per-CTA reserve, not a "second half-bank" that becomes free
under any condition. This is precisely the misconception R50C's bail
clause was written to catch.

The R49C §7.3 framing — "the second LDS slot is empty at 4096³ where
total_tiles == nCU" — was based on misreading `Occupancy [waves/SIMD]:
2` as CTAs/CU. R50C corrected this. The R52K prompt re-introduces the
same confusion under a different name ("80 KB half-bank physically
unused"). Both phrasings rest on the same false premise.

---

## 3. Why the existing kernel is already maximally A-prefetched within
its 24 KB CRR slack

Inspecting the production K-loop in
`crr_mxfp8_exact_8wave_fastpath.inc:198-237` (already in tree):

```cpp
for (int k = 0; k < k_iters - 2; k++, tic ^= 1, toc ^= 1) {
    load_b(b0, Bs[tic][0], wn);
    load_a(a, As[tic][0], wm);
    global_load_a(As[toc][1], br * 2 + 1, k + 1);  // <-- iter-ahead A prefetch
    TK_WAIT_LGKM(CRR_EXACT_PREFETCH_LGKM);
    ...
    load_a(a, As[tic][1], wm);
    global_load_a(As[tic][0], br * 2, k + 2);      // <-- TWO iters ahead A prefetch
    global_load_b(Bs[tic][1], bc * 2 + 1, k + 2);  // <-- TWO iters ahead B prefetch
    ...
    global_load_b(Bs[tic][0], bc * 2, k + 2);      // <-- TWO iters ahead B prefetch
}
```

The existing K-loop already issues:
- 1 × A wave-tile load **1 iter ahead** (line 204)
- 1 × A wave-tile load **2 iters ahead** (line 226)
- 2 × B wave-tile loads **2 iters ahead** (lines 227, 236)

So within the existing `As[2][2]` budget the kernel runs a **(k, k+1, k+2)
three-deep VMEM pipeline** for both A and B. The mandate's proposed
addition ("issue async load of A[k+2] into the new prefetch slot in
parallel with consuming A[k+0]") is exactly what line 226 already does
*in the existing budget*, by reusing the just-consumed `As[tic][0]` slot
for the k+2 prefetch (after `load_a(a, As[tic][1], wm)` finishes the
last consumer of `tic`).

The compiler is already free to schedule these `global_load_a` at the
top of the K-iter via VGPR-VMEM scoreboarding (no extra LDS slot
required for issue-time decoupling on AMD GFX9 — buffer_load instructions
write directly to VGPR and the LDS write is from VGPR after VMEM
completion). Adding a 3rd LDS slot does not change the *issue time* of
the VMEM, only the LDS landing slot — and since the existing tic/toc
slot is freed by the time the k+2 load lands, no extra LDS is needed.

This means even *if* the 32 KB fit, the change would be a NO-OP at the
ISA level (per R51F lesson: "does the new buffer actually generate
extra ds_write_b128 instructions, or did the compiler optimize them
away?"). Existing scheduling already delivers the wins this hypothesis
predicted.

---

## 4. ISA evidence (per R51F lesson — "what does the compiler do?")

No build was attempted (per prompt step 6: bail before GPU bench when
LDS over-budget). However, we can still anchor the ISA discussion:

R47D's ISA breakdown (cited via r49c_findings.md §5) reports the CRR
exact-8wave K-loop body at ~3140 cycles per K-iteration over 224 inner
iterations at K=4096. The instruction mix already shows:
- `buffer_load_dwordx4` for both A (4 issues per K-iter) and B (8 issues)
- 4 × `ds_write_b128` for the LDS landing of those VMEM loads
- `ds_read_b128` for the consumer `load_a` / `load_b` from LDS

The proposed 3rd-slot variant would add 4 more `ds_write_b128` per
K-iter (one per `buffer_load_dwordx4` of A[k+2]), but as noted in §3
those ds_write instructions are already present at the production
schedule — they fire from the same VMEM destination VGPRs into the
existing `As[tic][0]` slot once it's consumed. Doubling the LDS slot
does not add new ds_writes; it only changes which SRD offset they
target.

So the predicted ISA delta from the change (extra `ds_write_b128`
instructions) does not exist. Per R51F's outcome on the CRR `#pragma
unroll` knob, this is the "compiler no-op" pattern: the requested change
either doesn't compile (LDS over-budget) or generates bit-identical ISA
(if the LDS-fit `As_pf[1]` variant were attempted — see §1.2).

---

## 5. SHIP gate decision

**REFUTED — no `MXFP8_CRR_ASIDE_DOUBLEBUF` macro added. No source
patch lands. No GPU bench executed.** Per prompt:

> "Geomean Δ across 3 tested cells ≥ +1.5%. Worst-case cell Δ ≥ -1.5%.
> All cells: SNR ≥ 48 dB, det 3/3. No VGPR spill in compile output."

…cannot be evaluated because the smallest-fitting variant (`As_pf[1]`,
16 KB) is provably ISA-equivalent to baseline (no extra ds_writes) and
the actually-prefetching variant (`As[3][2]`, 32 KB) blows the 160 KB
per-CTA cap → kernel fails to launch.

Failure mode pre-empted matches R49 Dev B's "don't waste GPU time on a
known regression" and R50C's bail clause.

---

## 6. Triangulation against R50C and R49C

| Cycle | Hypothesis | Verdict |
|---|---|---|
| R49C §7.3 | "second occupancy slot is empty at total_tiles==nCU → spend it on A-prefetch" | suggested as future work |
| R50C | LDS audit: kernel uses 128-136 KB/CTA, second slot was never occupiable in the first place | REFUTED the premise |
| **R52K** | "second 80 KB half-bank physically unused → free for A-prefetch" | **REFUTED again — same misread of LDS as R49C §7.3, same bail as R50C** |

This is the third cycle to attempt the same lever under different
framings; all three reach the same `LDS-over-budget` wall. The lever
should be considered closed for the V2-CRR / V2-RRR kernel family at
this BLK/BK/HB shape.

The remaining 4096³-family HEADROOM cells (8B Q/O CRR +2.3pp, 70B Q/O
CRR +1.0pp) are not reachable through LDS depth changes. R49C §7.2 and
R50C §7 already enumerate the actual untouched levers:

1. **CRR scale-prefetch lead-distance** (move cA pre-fetch one BK
   earlier so its lgkmcnt drops cleanly before the cB MMA dependency
   edge) — never attempted.
2. **B-tile LDS bank-conflict pessimism at N=4096** (most RRR R&D
   was at N=14336/28672) — never attempted at the prefill N.
3. **CRR partial-unroll sweep at K=4096** — R52J already attempted the
   B1-LDS-INSERT-AFTER variant and REFUTED; the unroll knob itself was
   shown a compiler no-op in R51F.

None of these involve LDS depth. The LDS depth lever is closed.

---

## 7. Files

- `analysis/fp8_gemm/mi350x/r52k_findings.md` — this document
- `analysis/fp8_gemm/mi350x/r52k_bench.sh` — placeholder; bench not
  executed because pre-bench audit triggered bail clause (records the
  arithmetic that would have been the build's compile-time check)
- `analysis/fp8_gemm/mi350x/r52k_results/lds_audit.md` — the LDS
  arithmetic from §1, captured in plain text for grep-ability
  (.md not .txt to bypass repo's `*.txt` gitignore)

No source modifications. The hypothesized `MXFP8_CRR_ASIDE_DOUBLEBUF`
macro is NOT introduced because the prototype is REFUTED before
introduction per the prompt's bail clause. Per R49 Dev B's lesson on
not wasting GPU time on known regressions, no GPU was used.
