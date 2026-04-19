#!/usr/bin/env python3
"""Generate R36_LEADERBOARD.md comparing R25 (broken-output) vs R36 (correctness-gated _f34)."""
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def load(path):
    with open(path) as f:
        return json.load(f)

r25 = load(os.path.join(SCRIPT_DIR, "bench_all42_results_R25_FINAL_v2.json"))
r36 = load(os.path.join(SCRIPT_DIR, "bench_all42_results_R36_f34.json"))

# Build maps keyed by (M,N,K)
r25_map = {(r["M"], r["N"], r["K"]): r for r in r25["results"]}
r36_map = {(r["M"], r["N"], r["K"]): r for r in r36["results"]}

shapes = sorted(r25_map.keys())

lines = []
lines.append("# R36 Leaderboard — _f34 Correctness-Gated Re-bench (2026-04-19)\n")
lines.append("## TL;DR — Mechanically Applying `-DFUSED_STEP34=1` to BEST_VARIANTS Stacks BREAKS Everything\n")
lines.append(f"- R25_FINAL_v2 reported **{r25['wins']}/42 WIN**, but every WIN was \"time to wrong output\".")
lines.append(f"- R36 builds `_f34` versions of every BEST_VARIANTS module by appending `-DFUSED_STEP34=1`")
lines.append(f"  to its CPPFLAGS, then runs a correctness gate (constant scale=-4, finite_frac >= 0.995)")
lines.append(f"  before timing. Result: **{r36['wins']}/42 WIN, {r36['losses']}/42 LOSE,")
lines.append(f"  {r36['wrong_output']}/42 WRONG_OUTPUT, {r36['errors']}/42 CRASH/ERR**.\n")
lines.append("- The R35 finding (_f34 produces correct output) was validated only against the")
lines.append("  PLAIN kernel (no R25-C tail-pf-off, no BARRIER_TO_WAITCNT, no K_EXACT). When `-DFUSED_STEP34=1`")
lines.append("  is composed with the production-tuned BEST_VARIANTS flags (R25C_TAIL_PF_OFF_ITERS,")
lines.append("  BARRIER_TO_WAITCNT_ALL, R25C_K_EXACT, etc.), every shape either crashes (memory")
lines.append("  aperture violation) or produces wrong output.\n")
lines.append("**Conclusion:** Fix B (root-cause repair: backport `kpair_64mfma_step34` into the")
lines.append("default code path while preserving R25-C tail-pf logic) is REQUIRED. Fix A (mechanical")
lines.append("`-DFUSED_STEP34=1` per-shape) is unworkable.\n")
lines.append("## Bench Setup\n")
lines.append(f"- WARMUP={r36['warmup']}, ITERS={r36['iters']}, trim={r36['trim_frac']}")
lines.append(f"- Correctness gate: kernel_finite >= {r36['correctness_gate_finite']} on **constant scale=-4**")
lines.append("  inputs (matches `snr_all_42_shapes.py`; bounds the bf16 output range so the gate")
lines.append("  measures *kernel correctness*, not bf16 saturation).")
lines.append("- Timing inputs use random scale [-2,3] (matches existing bench harness).")
lines.append(f"- GPUs: {r36['gpus']}, total {r36['elapsed_minutes']} min.\n")

lines.append("## Per-Shape Results\n")
lines.append("| # | M | N | K | R25 TFLOPS (broken) | R36 _f34 TFLOPS (correct?) | R36 status | finite | comp | R25 vs comp | R36 vs comp | parent variant |")
lines.append("|---|---|---|---|---:|---:|:---|---:|---:|---:|---:|:---|")

regressions = []
wins_r36 = []
wrong_only = []
crashes = []

for i, shape in enumerate(shapes):
    m, n, k = shape
    r25_r = r25_map[shape]
    r36_r = r36_map[shape]
    r25_t = r25_r.get("tflops")
    r36_t = r36_r.get("tflops")
    r25_v = r25_r.get("best_variant", "?")
    comp = r25_r.get("comp")
    fin = r36_r.get("kernel_finite")
    fin_s = f"{fin:.4f}" if fin is not None else "—"
    r36_status = r36_r.get("status", "?")

    r25_ratio = (r25_t / comp * 100) if (r25_t and comp) else None
    r36_ratio = (r36_t / comp * 100) if (r36_t and comp) else None
    r25_ratio_s = f"{r25_ratio:.1f}%" if r25_ratio else "—"
    r36_ratio_s = f"{r36_ratio:.1f}%" if r36_ratio else "—"
    r36_t_s = f"{r36_t:.1f}" if r36_t else "—"
    r25_t_s = f"{r25_t:.1f}" if r25_t else "—"

    lines.append(
        f"| {i+1} | {m} | {n} | {k} | {r25_t_s} | {r36_t_s} | {r36_status} | "
        f"{fin_s} | {comp:.1f} | {r25_ratio_s} | {r36_ratio_s} | `{r25_v}` |"
    )

    if r36_status == "OK":
        if r36_t and comp and r36_t >= comp:
            wins_r36.append((m, n, k, r36_t, comp, r25_t, r25_v))
        if r25_t and r36_t:
            delta_pct = (r36_t - r25_t) / r25_t * 100.0
            if delta_pct < -5:
                regressions.append((m, n, k, r25_t, r36_t, delta_pct, r25_v))
    elif r36_status == "WRONG_OUTPUT":
        wrong_only.append((m, n, k, fin, r25_v))
    else:
        crashes.append((m, n, k, r36_status, r25_v))

lines.append("\n## Summary Counts\n")
lines.append(f"- **WIN**: {r36['wins']}/42 (passed correctness gate AND TFLOPS >= comp)")
lines.append(f"- **LOSE**: {r36['losses']}/42 (passed correctness gate but TFLOPS < comp)")
lines.append(f"- **WRONG_OUTPUT**: {r36['wrong_output']}/42 (kernel_finite < 0.995 even with bounded scales)")
lines.append(f"- **CRASH/ERR**: {r36['errors']}/42 (memory aperture violation or other failure)\n")

if wins_r36:
    lines.append("## R36 Wins (correctness-gated)\n")
    for m, n, k, t, comp, r25t, var in wins_r36:
        lines.append(f"- `{m}x{n}x{k}` — {t:.1f} TFLOPS vs comp {comp:.1f} ({t/comp*100:.1f}%), variant `{var}_f34`")
    lines.append("")
else:
    lines.append("## R36 Wins\n\n(none)\n")

if regressions:
    lines.append("## Worst Regressions (R36 _f34 vs R25, top 5 by Δ%)\n")
    for m, n, k, r25t, r36t, dp, var in sorted(regressions, key=lambda x: x[5])[:5]:
        lines.append(f"- `{m}x{n}x{k}` — R25={r25t:.1f} → R36={r36t:.1f} ({dp:+.1f}%), parent `{var}`")
    lines.append("")
else:
    lines.append("## Worst Regressions\n\n(no R36 result with TFLOPS<R25 since virtually no shape passed correctness)\n")

lines.append("## Failure Mode Breakdown\n")
lines.append("### CRASH / Memory aperture violation\n")
lines.append("These shapes' f34 builds segfault on the first run (HSA memory aperture violation).")
lines.append("Mechanism: when `FUSED_STEP34=1` is set, the entire R25-C tail-pf-off conditional branch")
lines.append("(line 2849-2945 of kernel) is bypassed. `kpair_64mfma_step34` ALWAYS issues prefetches,")
lines.append("which on the last K-iter read past the buffer SRD and trigger the GPU's address fault.\n")
for m, n, k, st, var in crashes:
    lines.append(f"- `{m}x{n}x{k}` — `{var}_f34` ({st})")
lines.append("\n### WRONG_OUTPUT\n")
lines.append("Shapes whose f34 build produces non-finite cells even with bounded constant scale=-4.")
lines.append("Distinct from CRASH because the kernel returns; the corruption manifests as inf/nan in")
lines.append("the output. Likely the same prefetch-past-end issue as CRASH but the OOB read happens")
lines.append("to land in mapped memory and produce garbage rather than fault.\n")
for m, n, k, fin, var in sorted(wrong_only, key=lambda x: -x[3]):
    lines.append(f"- `{m}x{n}x{k}` — `{var}_f34` (finite={fin:.4f})")
lines.append("")

lines.append("## Sanity Check: PLAIN _f34 (no R25-C/BTW/K_EXACT) DOES Work\n")
lines.append("On `32768x4096x2048` (which CRASHES with `ts_lgk2_gm6_v12_memc_pfoff4_f34`):")
lines.append("- Plain `tk_mxfp4_gluon_cpp_n4096_k2048_f34` (just `-DFUSED_STEP34=1`, no other flags)")
lines.append("  produces **finite_frac = 0.9977** at constant scale=-4, passing the correctness gate.")
lines.append("- This confirms the issue is in the **flag composition** (FUSED_STEP34 + R25-C tail-pf,")
lines.append("  + BTW, + K_EXACT), not in FUSED_STEP34 itself.\n")

lines.append("## Recommended Next Steps\n")
lines.append("1. **Fix B from R35 diagnosis (backport)**: rewrite the non-fused step3+step4 path so that")
lines.append("   it uses a fused single-asm-block step3+step4 INTERNALLY (matching `kpair_64mfma_step34`)")
lines.append("   while still respecting R25-C tail-pf-off, BTW, and K_EXACT branching. This is the only")
lines.append("   path that preserves both correctness AND the per-shape tuning that gave the 41/42 result.")
lines.append("2. **Audit `kpair_64mfma_step34` for tail-iter prefetch safety**: it should accept a")
lines.append("   `pf_active` template parameter and skip prefetches on the final iter (mirroring the")
lines.append("   `_r25c_tail_no_pf` branch that the FUSED path currently bypasses).")
lines.append("3. **Re-evaluate the 41/42 claim**: every R31/R32/R33 \"WIN\" was measuring time-to-garbage.")
lines.append("   The TODO/AGENT_PROMPT files should be updated to reflect that the production")
lines.append("   leaderboard is now empty until Fix B lands.\n")

lines.append("## Artifacts\n")
lines.append("- `bench_all_42_correct.py` — correctness-gated bench harness (constant-scale gate)")
lines.append("- `build_R36_f34.py` — parallel builder for `_f34` versions of every BEST_VARIANTS entry")
lines.append("- `bench_all42_results_R36_f34.json` — full results")
lines.append("- `bench_all42_results_R36_f34.log` — bench log")
lines.append("- `R36_F34_BUILD.log` — build log (30 unique builds, all succeeded)")
lines.append("- `R36_F34_BUILD_MANIFEST.json` — built module manifest\n")

out = os.path.join(SCRIPT_DIR, "R36_LEADERBOARD.md")
with open(out, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Wrote {out} ({len(lines)} lines)")
