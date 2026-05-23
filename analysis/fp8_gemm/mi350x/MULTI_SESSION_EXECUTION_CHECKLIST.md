# Multi-Session Execution Checklist (R86)

## Per-Session Entry Gates

Before any session:
- [ ] `git log -1` HK turbo + PT outer + PT 3rdparty commit hashes recorded
- [ ] sync.sh push parity verified for both repos
- [ ] chi2811 docker exec `import primus_turbo; print(primus_turbo.__file__)` OK
- [ ] `_smoke_p1_0_rcr_v2.py` runs without GPU error
- [ ] Current bench v2/Triton geomean recorded (baseline)

## Per-Session Exit Gates

Before commit + session end:
- [ ] amdhsa.kernels metadata spill check: no regression vs entry baseline
- [ ] `_smoke_p1_0_rcr_v2.py` SNR ≥ 30 dB on all 10 case
- [ ] bench geomean ≥ entry baseline OR documented regression with explanation
- [ ] HK + PT 3rdparty diff matches (`diff -q` clean)
- [ ] sync.sh push parity (HK commit + PT submodule bump + PT outer commit)
- [ ] CLAUDE.md §5 round log entry added
- [ ] PLAN_V2.md §8 milestone KPI updated if milestone gate passed/failed
- [ ] If session aborted mid-work: branch saved with `[wip]` prefix, NOT pushed

## P1.2 Session Sequence (32×32 rewrite)

### S1 — Fragment types + load primitives (~150 LOC)
Entry: R57 V=104 spill=0 probe holds
Exit: probe extended with rt_32x64_s frags, V<70 A<32 spill=0

### S2 — K-loop body clone with 32×32 (~200 LOC)
Entry: S1 frags + load primitives validated
Exit: `grouped_rcr_kernel_body_pinned_32` compiles, metadata V<200 A<128 spill=0 on FUSED=false

### S3 — FUSED_KTAIL port (~100 LOC)
Entry: S2 base body compiles + smoke FUSED=false bit-eq
Exit: FUSED=true template instance V<200 A<128 spill≤5

### S4 — Dispatcher integration + bench (~50 LOC)
Entry: All 4 v2 template variants spill=0
Exit: 24-shape bench: v2/Triton ≥ 1.024× (R43 baseline), no shape regress > 5%

### S5 — Production routing + cleanup (~30 LOC + ~400 LOC delete)
Entry: S4 perf gate passed
Exit: 16×16 body deleted, 32×32 default, autotune adjusted

## P1.3a Session Sequence (split-K)

### S1 — Dispatcher param plumbing (~50 LOC)
### S2 — K-partition kernel body (~200 LOC)
### S3 — Reduce kernel (~100 LOC)
### S4 — Cross-group B share heuristic (~150 LOC)
### S5 — Bench + autotune integration (~100 LOC)

## Abort Criteria

If any session ends with:
- spill regression > 10
- correctness break that takes > 4 hr to debug
- v2/Triton geomean drop > 5pp vs entry

→ Abort + revert all session changes; root-cause + document failure mode.

After 2 consecutive aborts on same phase: switch to alternative (P1.2 32×32 → BLK 128×128 single-acc OR split-K only).

## Session Dependency Graph

```
R52-R58 foundation (DONE) → P1.2 S1 → S2 → S3 → S4 → S5
                                                       ↓
                                          P1.3a S1 → S2 → S3 → S4 → S5
P2.2 (RRR) starts after P1.2 S4 (wrapper reusable)
P3.2 (CRR) starts after P1.2 S4 (wrapper reusable)
```

## Total Estimated Effort

- P1.2: 5 sessions × 5 hr = 25 hr
- P1.3a: 5 sessions × 5 hr = 25 hr
- P2.2: 5-7 sessions × 5 hr = 30 hr
- P3.2: 3-4 sessions × 5 hr = 17 hr

Total: ~100 hr of focused engineering across 18-21 sessions to fully meet user goals (1.15× Triton + spill=0 + 3% hk_dense).
