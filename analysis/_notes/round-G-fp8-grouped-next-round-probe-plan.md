# Round G — FP8 grouped: probe plan after Round F closes

**Status (2026-05-07 EOD)**: Round F finalized (M2a structural port +
M2-debug-3 wait-counter correctness fix landed; M5 perf falsified, kernel
quarantined to env=1). Production score = 691 unchanged. This note scopes
Round G's probe-then-decide plan; no kernel changes proposed yet.

---

## Where the lifts have to come from

Per-section progress on `_metric_gpt_oss_fp8_kernel.py` (target avg 2800 T):

| section | HK avg | TRT avg  | progress | gap to 2800 |
|---------|--------|----------|----------|-------------|
| fwd     | 1908   | 1720     | 0.682    | +892  T (+47 %) |
| dgrad   | 2088   | 1782     | 0.746    | +712  T (+34 %) |
| wgrad   | 1777   | 937      | 0.635    | +1010 T (+57 %) |

All three sections are far from target. Current HK/TRT ratio is healthy
(1.7-2.1x on wgrad, 1.05-1.25x on fwd, 1.07-1.29x on dgrad), so the gap
reflects **structural saturation of the kernels**, not headroom-vs-Triton.

Per-shape dgrad rocprof (HK):

| shape              | dgrad TFLOPS | K_kern (RCR after H4) | K-tail? |
|--------------------|--------------|------------------------|---------|
| GateUP_B4_M2048    | 1934         | 5760                   | no      |
| GateUP_B4_M4096    | 2529         | 5760                   | no      |
| Down_B4_M2048      | 1430         | 2880                   | yes (K%128=64) |
| Down_B4_M4096      | 1918         | 2880                   | yes     |
| GateUP_B32_M2048   | 2564         | 5760                   | no      |
| GateUP_B32_M4096   | 2602         | 5760                   | no      |
| Down_B32_M2048     | 1897         | 2880                   | yes     |
| Down_B32_M4096     | 1963         | 2880                   | yes     |

Down dgrad is consistently 23-44 % slower than GateUP dgrad. Two factors:
1. K-tail epilog runs (FUSED_KTAIL=true) — the kernel does an extra
   ~64-element K-block reduction inside the main launch.
2. K-iters per tile = 22 (Down) vs 45 (GateUP) — half the steady-state K
   accumulation per per-tile fixed overhead.

Factor 2 alone implies Down should be at most ~22/45 × GateUP = ~50 % of
GateUP under "fixed overhead per tile" pessimism, but Down is actually
75 % of GateUP — meaning the K-iter penalty is partially amortised. The
K-tail epilog explains the remaining 25 % gap.

For wgrad, all 8 shapes use `grouped_var_k_kernel_fp8` (variable-K CRR);
no K-tail kernel breakout. The 97.1 % single-kernel concentration on
Down B=4 wgrad means there's no host overhead lever; everything is
inside the kernel itself.

---

## Round G probe options

In rough EV order (estimated max gain × probability of landing):

### Option G1: PMC-grounded var-K CRR profile (wgrad) — HIGH leverage, HIGH risk

* Why: wgrad is the lowest progress section (0.635). Round B falsified
  CRR/RRR wait counter sweeps but it was on the FIXED-K CRR. Variable-K
  CRR has different per-iter overhead (per-group `ki_g = M_g/HB`
  computation, per-tile variable K accumulation length).
* Probe: rocprofv3 PMC pass on `grouped_var_k_kernel_fp8` for the 8
  gpt_oss wgrad shapes capturing
  {GRBM_GUI_ACTIVE, SQ_BUSY_CYCLES, SQ_INSTS_MFMA, SQ_INSTS_VALU,
   SQ_INSTS_LDS, SQ_INSTS_VMEM_RD, SQ_WAIT_INST_LDS}. Compute MFMA-busy %
  and dominant stall (LDS-bank-conflict, VMEM-read backlog, VALU pre-MFMA
  overhead). Compare to RCR fwd (50 % peak per Round D).
* Decide: if MFMA-busy < 50 % AND a non-MFMA stall is identified, propose
  a focused optimisation (instruction interleave, LDS-bank fix, etc.).
  If MFMA-busy > 70 %, kernel is saturated → falsify-and-document.
* EV math: wgrad section weight in score = 1/3. Lifting wgrad avg by
  +500 T moves overall score +500/3/2800 × 1000 = +60 points (691 → 751).

### Option G2: FUSED_KTAIL path PMC + targeted re-tune — MEDIUM leverage, MEDIUM risk

* Why: Down dgrad family (4 shapes) loses 25 % to GateUP across the board.
  The FUSED_KTAIL=true path was tuned through Round 3 (HK commits 07354791,
  ad-hoc R3-R12-dm chain) but never PMC-profiled with the current
  R96-EOD wait counter values.
* Probe: rocprofv3 PMC pass on `grouped_rcr_kernel<0, false, true, false>`
  spec, isolating the K-tail epilog cycles vs main-loop cycles. Counters:
  same as G1 + `SQ_INSTS_FLAT` (R22-PT identified ~411 divergent SRD
  fallback loops in this spec; current state unknown).
* Decide: if K-tail epilog > 15 % of kernel cycles, probe interleave +
  scratch-relief levers similar to Round-44-dm probes (mul-store
  interleave already shipped). If < 5 %, lever is exhausted.
* EV math: dgrad section weight = 1/3. Lifting Down dgrad to GateUP
  dgrad parity = +500 T avg on 4/8 dgrad shapes = +250 T section
  average = +250/3/2800 × 1000 = +30 points (691 → 721).

### Option G3: Forward path tile-merge / persistence-loop tuning — MEDIUM-LOW

* Why: fwd is the second-lowest progress (0.682). All fwd shapes go
  through `grouped_rcr_kernel<*, *, true, *>` (FUSED_KTAIL=true since
  K=2880 K%128=64 for all 8). Same code path as G2 but different shape
  geometry (N from output, K=2880 always).
* Probe / decide: subset of G2; same kernel, same PMC, same falsify
  criterion. If G2 lands a fix, fwd should benefit "for free".
* EV: subsumed by G2.

### Option G4: H4 reroute heuristic re-tune — LOW leverage

* Why: every dgrad on the metric currently H4-reroutes (Round-3 forces
  unconditional reroute on trans_b=False after caching the transpose).
  Investigating whether selective opt-out helps any particular shape.
* Probe: per-shape time the "no reroute → RRR direct" path vs the
  current "H4 → RCR with K-tail-fuse" path on Down dgrad family.
  Round-3 commit message documented +9-30 % wins from H4 reroute on
  the 24-shape FP8 metric — re-verify is on Down B=4 dgrad specifically
  (which has the lowest dgrad number).
* Decide: probably already optimal (Round-3's table covers similar
  shapes), but the verification is cheap (~5 min runtime).

### Option G5: Variable-K CRR LDS-tile reshape — HIGH risk, multi-week

* Why: wgrad's `grouped_var_k_kernel_fp8` uses ST_v2 LDS tiles inherited
  from RCR. The wgrad data movement pattern is different (M-strided B
  load, K-accumulation across groups) — a CRR-specialised LDS layout
  with tile padding tuned for the access pattern might help.
* Risk: extensive reasoning + bench cycles, similar in scope to Round F
  (which falsified). Best deferred until G1 PMC analysis identifies
  LDS as the actual bottleneck.

---

## Recommended sequencing for Round G

Sequential, each with ship-or-falsify gate:

| step | what                                | budget | trigger to next |
|------|-------------------------------------|--------|-----------------|
| G1a  | wgrad PMC pass (8 shapes × 7 ctrs)  | 1 day  | always          |
| G1b  | identify dominant stall + write up  | 1 day  | always          |
| G1c  | propose lever + EV math             | 0.5 d  | EV > +30 pts    |
| G2a  | dgrad K-tail PMC pass (Down B=4)    | 0.5 d  | parallel w/ G1  |
| G2b  | identify K-tail bottleneck          | 0.5 d  | always          |
| G2c  | propose lever                       | 0.5 d  | EV > +30 pts    |
| G3   | implement higher-EV lever           | 2-5 d  | (G1c or G2c)    |
| G4   | metric verify + ship-or-falsify     | 0.5 d  | always          |

Total: ~5-10 days for Round G. If both G1c AND G2c falsify (current
kernels are already saturated), this becomes "Round G negative finding,
HipKittens FP8 grouped is structurally saturated at 1900-2100 T avg per
section on gpt_oss; further lifts require an algorithmic change, not a
kernel tune."

---

## What NOT to do in Round G

* Re-attempt Round F b128 with different parameters. The per-tile fixed
  overhead arithmetic is fundamental; only a major architectural change
  (e.g., 4-wave 2-CTA b128) could flip the conclusion, which is a
  Round H+ scope.
* Re-sweep CRR/RRR wait counters (Round B already saturated).
* Re-sweep RCR wait counters (Round A shipped LGKM=8; Round B verified
  RCR steady waits; Round C verified setprio/sched_barrier; Round D
  PMC-confirmed structural saturation).
* Port new MFMA primitives (Round E falsified mfma_32x32x64 via
  microbench).

The remaining open levers all require PMC-grounded characterization
before they can be ranked or attempted.
