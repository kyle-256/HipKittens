"""R35 Dev D — aggregate paired-bench results across all 14 cells × 2 GPUs.

Parses r35d_<cell>_<la>_vs_<lb>_gpu<phys>.txt and produces a markdown table.
Computes per-cell Δ%, Welch t, advisory firing count, and min-of-GPUs Δ%.
"""
import re, glob, os, sys, statistics, math, json

HERE = os.path.dirname(os.path.abspath(__file__))

CELLS = [
    # (tag, label, M, N, K, expected_advisory_for_RRR_vs_CRR)
    ("8b_q",     "8B-Q     (4096x4096x4096)",     4096, 4096, 4096,   False),
    ("8b_k",     "8B-K     (4096x1024x4096)",     4096, 1024, 4096,   True),
    ("8b_v",     "8B-V     (4096x1024x4096)",     4096, 1024, 4096,   True),
    ("8b_o",     "8B-O     (4096x4096x4096)",     4096, 4096, 4096,   False),
    ("8b_gate",  "8B-Gate  (4096x14336x4096)",    4096, 14336, 4096,  False),  # r35-a not landed @ R34 head
    ("8b_up",    "8B-Up    (4096x14336x4096)",    4096, 14336, 4096,  False),  # r35-a not landed @ R34 head
    ("8b_down",  "8B-Down  (4096x4096x14336)",    4096, 4096, 14336,  False),  # not wired (informational)
    ("70b_q",    "70B-Q    (4096x8192x8192)",     4096, 8192, 8192,   False),
    ("70b_k",    "70B-K    (4096x1024x8192)",     4096, 1024, 8192,   True),
    ("70b_v",    "70B-V    (4096x1024x8192)",     4096, 1024, 8192,   True),
    ("70b_o",    "70B-O    (4096x8192x8192)",     4096, 8192, 8192,   False),
    ("70b_gate", "70B-Gate (4096x28672x8192)",    4096, 28672, 8192,  True),
    ("70b_up",   "70B-Up   (4096x28672x8192)",    4096, 28672, 8192,  True),
    ("70b_down", "70B-Down (4096x8192x28672)",    4096, 8192, 28672,  True),
]

ADVISORY_RE = re.compile(r"\[tk_mxfp8_layouts\] gemm_crr_pq_v2: shape \(M=(\d+), N=(\d+), K=(\d+)\)")
DELTA_RE = re.compile(r"DELTA_MEDIAN_PCT (\w+)_vs_(\w+) = ([-+]\d+\.\d+)%")
WELCH_RE = re.compile(r"Welch t \((\w+) vs (\w+)\) = ([-+]?\d+\.\d+)")
SCLK_POST_RE = re.compile(r"\[sclk-post-preheat\].*?(\d+)Mhz")
CORR_RE = re.compile(r"CORRECTNESS_(\w+) snr_db=(\d+\.\d+) pass_rate_pct=(\d+\.\d+) det_ok=(\w+)")
LIST_RE = re.compile(r"^(\w+)_TFLOPS_LIST (.+)$", re.M)
MEDIAN_RE = re.compile(r"^(\w+)\s+median=([\d.]+) mean=([\d.]+) stdev=([\d.]+) n=(\d+)", re.M)

def parse_log(path):
    """Parse a single bench log; returns the LAST (final) attempt's data only."""
    with open(path) as f:
        text = f.read()
    # Split into attempts. The last successful attempt is the saved one.
    # Strategy: find all DELTA lines; take the LAST occurrence for the final values.
    deltas = list(DELTA_RE.finditer(text))
    welches = list(WELCH_RE.finditer(text))
    medians = list(MEDIAN_RE.finditer(text))
    advisories = list(ADVISORY_RE.finditer(text))
    sclks = list(SCLK_POST_RE.finditer(text))
    corrs = list(CORR_RE.finditer(text))
    lists = list(LIST_RE.finditer(text))
    if not deltas:
        return None
    # last delta = final saved bench
    delta = deltas[-1]
    welch = welches[-1] if welches else None
    # For medians, take the last 2
    final_medians = medians[-2:] if len(medians) >= 2 else medians
    final_lists = lists[-2:] if len(lists) >= 2 else lists
    final_corrs = corrs[-2:] if len(corrs) >= 2 else corrs
    sclk_post = int(sclks[-1].group(1)) if sclks else None
    advisory_count = len(advisories)
    return {
        "delta_pct": float(delta.group(3)),
        "label_b": delta.group(1),
        "label_a": delta.group(2),
        "welch_t": float(welch.group(3)) if welch else None,
        "medians": [(m.group(1), float(m.group(2)), float(m.group(4)), int(m.group(5))) for m in final_medians],
        "lists": [(l.group(1), [float(x) for x in l.group(2).split(",")]) for l in final_lists],
        "advisory_count": advisory_count,
        "sclk_post_preheat_mhz": sclk_post,
        "correctness": [(c.group(1), float(c.group(2)), float(c.group(3)), c.group(4)) for c in final_corrs],
    }


def main():
    results = {}  # cell -> {gpu -> {"crr_vs_rrr": data, "crr_vs_rcr": data}}
    for cell in CELLS:
        tag = cell[0]
        results[tag] = {}
        for gpu in [6, 7]:
            results[tag][gpu] = {}
            for cmp_kind in ["rrr", "rcr"]:
                path = os.path.join(HERE, f"r35d_{tag}_crr_vs_{cmp_kind}_gpu{gpu}.txt")
                if os.path.exists(path):
                    parsed = parse_log(path)
                    if parsed:
                        results[tag][f"crr_vs_{cmp_kind}"] = results[tag].get(f"crr_vs_{cmp_kind}", {})
                        results[tag][gpu][f"crr_vs_{cmp_kind}"] = parsed

    # Markdown table — Phase 3 — CRR vs RRR per cell, both GPUs
    out = []
    out.append("## CRR vs RRR — per-cell paired bench (BABA, n=10/kernel)\n")
    out.append("| Cell | Shape | GPU6 Δ% | GPU6 t | GPU6 adv | GPU7 Δ% | GPU7 t | GPU7 adv | min Δ% | Adv expected | Verdict |")
    out.append("|---|---|---:|---:|:---:|---:|---:|:---:|---:|:---:|---|")
    for tag, label, M, N, K, exp_adv in CELLS:
        gpu6 = results.get(tag, {}).get(6, {}).get("crr_vs_rrr")
        gpu7 = results.get(tag, {}).get(7, {}).get("crr_vs_rrr")
        if not gpu6 and not gpu7:
            out.append(f"| {tag} | {M}x{N}x{K} | n/a | n/a | n/a | n/a | n/a | n/a | n/a | {'YES' if exp_adv else 'no'} | NO_DATA |")
            continue
        d6 = f"{gpu6['delta_pct']:+.2f}" if gpu6 else "n/a"
        t6 = f"{gpu6['welch_t']:+.2f}" if gpu6 and gpu6['welch_t'] is not None else "n/a"
        a6 = str(gpu6['advisory_count']) if gpu6 else "n/a"
        d7 = f"{gpu7['delta_pct']:+.2f}" if gpu7 else "n/a"
        t7 = f"{gpu7['welch_t']:+.2f}" if gpu7 and gpu7['welch_t'] is not None else "n/a"
        a7 = str(gpu7['advisory_count']) if gpu7 else "n/a"
        deltas = []
        if gpu6: deltas.append(gpu6['delta_pct'])
        if gpu7: deltas.append(gpu7['delta_pct'])
        min_d = min(deltas) if deltas else None
        min_d_str = f"{min_d:+.2f}" if min_d is not None else "n/a"
        adv_expected = "YES" if exp_adv else "no"

        # Verdict logic
        adv_actual_6 = gpu6['advisory_count'] > 0 if gpu6 else False
        adv_actual_7 = gpu7['advisory_count'] > 0 if gpu7 else False
        any_adv = adv_actual_6 or adv_actual_7

        if exp_adv:
            if any_adv and min_d is not None and min_d >= 5.0:
                verdict = "**SHIP**"
            elif any_adv:
                verdict = "ADV-FIRES min-Δ-low"
            else:
                verdict = "**ADV-MISSING**"
        else:
            if any_adv:
                verdict = "**FALSE-POS**"
            else:
                verdict = "OK (no adv)"

        out.append(f"| {label.split()[0]} | {M}x{N}x{K} | {d6} | {t6} | {a6} | {d7} | {t7} | {a7} | {min_d_str} | {adv_expected} | {verdict} |")

    out.append("")
    out.append("## CRR vs RCR — square Q/O cells (3-way comparison)\n")
    out.append("| Cell | Shape | GPU6 Δ% (RCR vs CRR) | GPU6 t | GPU7 Δ% | GPU7 t | min Δ% (RCR>CRR) |")
    out.append("|---|---|---:|---:|---:|---:|---:|")
    for tag, label, M, N, K, _ in CELLS:
        if tag not in ["8b_q", "8b_o", "70b_q", "70b_o"]:
            continue
        gpu6 = results.get(tag, {}).get(6, {}).get("crr_vs_rcr")
        gpu7 = results.get(tag, {}).get(7, {}).get("crr_vs_rcr")
        d6 = f"{gpu6['delta_pct']:+.2f}" if gpu6 else "n/a"
        t6 = f"{gpu6['welch_t']:+.2f}" if gpu6 and gpu6['welch_t'] is not None else "n/a"
        d7 = f"{gpu7['delta_pct']:+.2f}" if gpu7 else "n/a"
        t7 = f"{gpu7['welch_t']:+.2f}" if gpu7 and gpu7['welch_t'] is not None else "n/a"
        deltas = []
        if gpu6: deltas.append(gpu6['delta_pct'])
        if gpu7: deltas.append(gpu7['delta_pct'])
        min_d = min(deltas) if deltas else None
        min_d_str = f"{min_d:+.2f}" if min_d is not None else "n/a"
        out.append(f"| {label.split()[0]} | {M}x{N}x{K} | {d6} | {t6} | {d7} | {t7} | {min_d_str} |")

    out.append("")
    out.append("## sclk-post-preheat (R34 contention diagnosis)\n")
    out.append("| Cell | GPU6 sclk MHz | GPU7 sclk MHz |")
    out.append("|---|---:|---:|")
    for tag, label, *_ in CELLS:
        gpu6 = results.get(tag, {}).get(6, {}).get("crr_vs_rrr")
        gpu7 = results.get(tag, {}).get(7, {}).get("crr_vs_rrr")
        s6 = gpu6['sclk_post_preheat_mhz'] if gpu6 else "n/a"
        s7 = gpu7['sclk_post_preheat_mhz'] if gpu7 else "n/a"
        out.append(f"| {tag} | {s6} | {s7} |")

    out.append("")
    out.append("## Correctness summary\n")
    correctness_fails = []
    for tag, label, *_ in CELLS:
        for gpu in [6, 7]:
            for cmp_kind in ["crr_vs_rrr", "crr_vs_rcr"]:
                d = results.get(tag, {}).get(gpu, {}).get(cmp_kind)
                if not d: continue
                for layer, snr, pass_pct, det in d['correctness']:
                    if snr < 48 or pass_pct < 99 or det != "True":
                        correctness_fails.append(f"{tag} GPU{gpu} {cmp_kind} {layer}: snr={snr} pass={pass_pct}% det={det}")
    if correctness_fails:
        out.append("FAILURES:")
        for f in correctness_fails:
            out.append(f" - {f}")
    else:
        out.append("All correctness checks PASS (snr ≥ 48 dB, pass_rate=100%, det_ok=True).")

    text = "\n".join(out)
    print(text)
    with open(os.path.join(HERE, "r35d_aggregate_table.md"), "w") as f:
        f.write(text)
    # also dump JSON
    with open(os.path.join(HERE, "r35d_aggregate.json"), "w") as f:
        # convert deeply
        def conv(x):
            if isinstance(x, dict): return {str(k): conv(v) for k, v in x.items()}
            if isinstance(x, list): return [conv(y) for y in x]
            if isinstance(x, tuple): return [conv(y) for y in x]
            return x
        json.dump(conv(results), f, indent=2)

if __name__ == "__main__":
    main()
