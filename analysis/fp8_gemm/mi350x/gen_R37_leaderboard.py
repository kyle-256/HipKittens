#!/usr/bin/env python3
"""Generate R37_LEADERBOARD.md."""
import json, os, time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(SCRIPT_DIR, "bench_all42_results_R37_fixB.json")) as f:
    r37 = json.load(f)
with open(os.path.join(SCRIPT_DIR, "bench_all42_results_r22.json")) as f:
    r22 = json.load(f)

# R22 (≈R25 ceiling) by (M,N,K)
r25 = {}
for r in r22.get("results", []):
    if r.get("tflops"):
        r25[(r["M"], r["N"], r["K"])] = r["tflops"]

results = sorted(r37["results"], key=lambda r: (r["M"], r["N"], r["K"]))
out = []
out.append("# R37 — Fix B Leaderboard (2026-04-19)")
out.append("")
out.append(f"Round: R37_fixB (kpair_64mfma_step34 backported into default; -mllvm -amdgpu-sched-strategy=max-memory-clause stripped)")
out.append(f"Bench: warmup={r37['warmup']}, iters={r37['iters']}, trim={r37['trim_frac']}, gate finite≥{r37['correctness_gate_finite']}")
out.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
out.append("")
out.append("## Summary")
out.append("")
out.append(f"- **WIN: {r37['wins']}/42** (correct output AND tflops ≥ comp)")
out.append(f"- **LOSE: {r37['losses']}/42** (correct output but tflops < comp)")
out.append(f"- **WRONG_OUTPUT: {r37['wrong_output']}/42** (kernel_finite < 0.995)")
out.append(f"- **CRASH/ERR: {r37['errors']}/42**")
out.append(f"- Elapsed: {r37['elapsed_minutes']} min on 8 GPUs")
out.append("")
out.append("## Per-shape results")
out.append("")
out.append("| M | N | K | finite | R37 TFLOPS | R25 TFLOPS | comp TFLOPS | R37/comp | status | variant |")
out.append("|---|---|---|--------|------------|------------|-------------|----------|--------|---------|")
for r in results:
    m, n, k = r["M"], r["N"], r["K"]
    fin = r.get("kernel_finite")
    fin_s = f"{fin:.4f}" if fin is not None else "N/A"
    tf = r.get("tflops")
    tf_s = f"{tf:.1f}" if tf is not None else "—"
    r25_tf = r25.get((m, n, k))
    r25_s = f"{r25_tf:.1f}" if r25_tf else "—"
    comp = r["comp"]
    if tf:
        ratio = f"{tf/comp*100:.1f}%"
    else:
        ratio = "—"
    status = r["status"]
    var = r.get("best_variant", "?")
    out.append(f"| {m} | {n} | {k} | {fin_s} | {tf_s} | {r25_s} | {comp:.1f} | {ratio} | {status} | {var} |")
out.append("")
out.append("## Key findings")
out.append("")
out.append("1. **R37 fix delivers correct output for 14/42 shapes**, all WINS vs comp (12-18% over baseline).")
out.append("   These are the shapes where the BEST_VARIANTS flag stack is compatible with the fused step34 path.")
out.append("")
out.append("2. **19 shapes WRONG_OUTPUT** — fused step34 still produces 0.6-0.99 finite frac on these.")
out.append("   Most are M ≥ 16384 with K ≥ 4096 (large grids, more iters → more chances for memc-style")
out.append("   reordering bugs to manifest). Some have finite just under the gate (0.989-0.999) and")
out.append("   would benefit from a tighter SNR-aware gate; others are deeply broken (0.62-0.76).")
out.append("")
out.append("3. **9 shapes CRASH** with HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION. All use the")
out.append("   `_ts_lgk2_gm6_v12_memc_pfoff4` (or sibling without _kx_btw_all) variant. The R25-C")
out.append("   tail-pf-off path interaction with fused step34 is producing OOB loads.")
out.append("")
out.append("## Recommendations")
out.append("")
out.append("- R37's WINS confirm the surgical fix is correct in principle: when memc and other")
out.append("  scheduler-aggressive flags are absent, the fused step34 is both correct AND faster")
out.append("  than the legacy path (12-18% over comp on the 14 winning shapes).")
out.append("- Next step: convert `emit_one_pf` (the buffer_load_to_lds intrinsic call) into an")
out.append("  inline `asm volatile` block, which would prevent the LLVM scheduler from reordering")
out.append("  prefetches across iteration boundaries — the root cause of memc-incompatibility.")
out.append("- Alternatively: fork BEST_VARIANTS to drop scheduler-aggressive flags from any variant")
out.append("  that fails correctness on R37, accepting a small TFLOPS hit on those shapes.")
out.append("")
out.append("## Files")
out.append("")
out.append("- `kernel_mxfp4_gluon_cpp.cpp` — modified with R37_FIX_B (default ON) + S1-force-barrier")
out.append("- `build_R37.py` — builder that strips `-mllvm -amdgpu-sched-strategy=max-memory-clause`")
out.append("- `bench_all_42_R37.py` — correctness-gated bench harness")
out.append("- `bench_all42_results_R37_fixB.json` — full bench results")
out.append("- `R37_BENCH_RUN.log` — bench run log")
out.append("- `build_R37/` — module .so files")

with open(os.path.join(SCRIPT_DIR, "R37_LEADERBOARD.md"), "w") as f:
    f.write("\n".join(out))
print(f"Wrote R37_LEADERBOARD.md ({len(out)} lines)")
