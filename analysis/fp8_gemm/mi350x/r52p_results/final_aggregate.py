#!/usr/bin/env python3
"""Final aggregation: combine set1 and set3 across all profiled cells."""
import csv, os, statistics, json
ROOT = os.path.dirname(os.path.abspath(__file__))

CELLS = {
    "v2_8B_GateUp":  ("v1_8B_gateup",          "v2_8B_set3",          (4096,14336,4096), "V2 RRR (default), M=4096 N=14336 K=4096 — TARGET HEADROOM cell"),
    "v1_8B_GateUp":  ("v1_8B_gateup_forced",   None,                  (4096,14336,4096), "V1 RRR (forced),  M=4096 N=14336 K=4096 — what R52N analyzed"),
    "v2_70B_QO":     ("v2_70B_QO",             "v2_70B_set3",         (4096, 8192,8192), "V2 RRR (default), M=4096 N=8192 K=8192 — fast comparison cell"),
    "v1_70B_QO":     ("v1_70B_QO_forced",      None,                  (4096, 8192,8192), "V1 RRR (forced),  M=4096 N=8192 K=8192"),
    "v2_70B_GateUp": ("v2_70B_GateUp",         "v2_70B_GateUp_set3",  (4096,28672,8192), "V2 RRR (default), M=4096 N=28672 K=8192 — large N comparison"),
}

def load(dirname):
    by_disp = {}
    timing = {}
    for pmc_dir in sorted(os.listdir(os.path.join(ROOT, dirname))):
        path = os.path.join(ROOT, dirname, pmc_dir, "pmc_counter_collection.csv")
        if not os.path.exists(path): continue
        with open(path) as f:
            for row in csv.DictReader(f):
                did = int(row["Dispatch_Id"])
                cn = row["Counter_Name"]
                cv = float(row["Counter_Value"])
                by_disp.setdefault(did, {})[cn] = cv
                timing.setdefault(did, {})["start"] = int(row["Start_Timestamp"])
                timing[did]["end"] = int(row["End_Timestamp"])
    counters = {}
    durations = []
    for did, cdict in by_disp.items():
        for c, v in cdict.items():
            counters.setdefault(c, []).append(v)
        durations.append(timing[did]["end"] - timing[did]["start"])
    medians = {c: statistics.median(vs) for c, vs in counters.items()}
    medians["__duration_ns_median__"] = statistics.median(durations) if durations else 0
    return medians

results = {}
for cell, (d1, d3, shape, descr) in CELLS.items():
    raw = load(d1)
    if d3 and os.path.exists(os.path.join(ROOT, d3)):
        raw3 = load(d3)
        for k, v in raw3.items():
            if not k.startswith("__"):
                raw[k] = v
    M, N, K = shape
    flops = 2 * M * N * K
    raw["__shape__"] = shape
    raw["__descr__"] = descr
    raw["__tflops__"] = flops / raw["__duration_ns_median__"] / 1000.0 if raw["__duration_ns_median__"] else 0
    results[cell] = raw

# Print key table
print("\n" + "="*120)
print("R52P STALL BREAKDOWN — V2 RRR production (and V1 forced) at multiple cells, MI355X gfx950")
print("="*120)
print()
print(f"{'Cell':<24} | TFLOPS | dur(us) | MfmaUtil | VALUBusy | TCC_hit | TA_DATA_stall_TC | TCP_TA_DATA_stall")
print("-"*120)
for cell, r in results.items():
    M,N,K = r["__shape__"]
    g = r.get("GRBM_GUI_ACTIVE", 1)
    tflops = r["__tflops__"]
    dur = r["__duration_ns_median__"] / 1000
    mfma = r.get("MfmaUtil", 0)
    valu = r.get("VALUBusy", 0)
    hit_pct = 100 * r.get("TCC_HIT_sum", 0) / max(1, r.get("TCC_HIT_sum",0)+r.get("TCC_MISS_sum",0))
    ta_data = r.get("TA_DATA_STALLED_BY_TC_CYCLES_sum", 0) / g
    tcp_data = r.get("TCP_TCP_TA_DATA_STALL_CYCLES_sum", 0) / g
    print(f"{cell:<24} | {tflops:6.0f} | {dur:7.1f} | {mfma:6.2f}%  | {valu:6.2f}%  | {hit_pct:5.2f}% | {ta_data:14.4f}   | {tcp_data:14.4f}")

# Per-shape gap analysis
print("\n" + "="*120)
print("HEADROOM GAP ANALYSIS — V2 8B Gate/Up vs V2 70B Q/O (default production path)")
print("="*120)
v2_8B = results["v2_8B_GateUp"]
v2_70B = results["v2_70B_QO"]

def cmp(label, key, scale=1.0, fmt="{:.2f}", per_grbm=False):
    a = v2_8B.get(key, 0) * scale
    b = v2_70B.get(key, 0) * scale
    if per_grbm:
        a /= v2_8B["GRBM_GUI_ACTIVE"]
        b /= v2_70B["GRBM_GUI_ACTIVE"]
    delta = b - a
    print(f"  {label:<48}: 8B={fmt.format(a):>14}  70B={fmt.format(b):>14}  delta={fmt.format(delta):>14}")

print()
cmp("Achieved TFLOPS",                      "__tflops__")
cmp("MfmaUtil (%)",                         "MfmaUtil")
cmp("VALUBusy (%)",                         "VALUBusy")
cmp("TCC L2 hit rate (%)",                  "TCC_HIT_sum", fmt="{:.2f}")  # raw, not pct - use derived
print()
cmp("TA_DATA_STALLED_BY_TC / GRBM",         "TA_DATA_STALLED_BY_TC_CYCLES_sum", per_grbm=True, fmt="{:.4f}")
cmp("TCP_TA_DATA_STALL / GRBM",             "TCP_TCP_TA_DATA_STALL_CYCLES_sum", per_grbm=True, fmt="{:.4f}")
cmp("TA_ADDR_STALLED_BY_TC / GRBM",         "TA_ADDR_STALLED_BY_TC_CYCLES_sum", per_grbm=True, fmt="{:.4f}")
cmp("TCP_PENDING_STALL / GRBM",             "TCP_PENDING_STALL_CYCLES_sum",     per_grbm=True, fmt="{:.4f}")
cmp("TCP_RFIFO_STALL / GRBM",               "TCP_RFIFO_STALL_CYCLES_sum",       per_grbm=True, fmt="{:.4f}")
cmp("SQ_VMEM_TA_ADDR_FIFO_FULL / GRBM",     "SQ_VMEM_TA_ADDR_FIFO_FULL",        per_grbm=True, fmt="{:.4f}")
cmp("SQ_LDS_BANK_CONFLICT (cyc) / GRBM",    "SQ_LDS_BANK_CONFLICT",             per_grbm=True, fmt="{:.6f}")
cmp("FetchSize (KB)",                       "FetchSize",                                       fmt="{:.0f}")
cmp("TCC_MISS_sum",                         "TCC_MISS_sum",                                    fmt="{:.0f}")
cmp("TCP_PENDING_STALL_sum (raw)",          "TCP_PENDING_STALL_CYCLES_sum",                    fmt="{:.0f}")
cmp("TA_DATA_STALLED_BY_TC_sum (raw)",      "TA_DATA_STALLED_BY_TC_CYCLES_sum",                fmt="{:.0f}")
cmp("TCP_TA_DATA_STALL_sum (raw)",          "TCP_TCP_TA_DATA_STALL_CYCLES_sum",                fmt="{:.0f}")

# Also do 70B Gate/Up (large-N comparison) to test "is large N the cause?"
print("\n" + "="*120)
print("LARGE-N CROSS-CHECK — V2 70B Gate/Up (N=28672) vs V2 70B Q/O (N=8192) vs V2 8B Gate/Up (N=14336)")
print("="*120)
print(f"{'Cell':<20} | TFLOPS | MfmaUtil | TA_DATA_stall_TC/GRBM | TCP_TA_DATA_stall/GRBM | TCP_PEND/GRBM")
print("-"*120)
for label, cell in [("V2 70B QO (N=8192)", "v2_70B_QO"), ("V2 8B Gate/Up (N=14336)", "v2_8B_GateUp"), ("V2 70B Gate/Up (N=28672)", "v2_70B_GateUp")]:
    r = results[cell]
    g = r["GRBM_GUI_ACTIVE"]
    print(f"{label:<24} | {r['__tflops__']:6.0f} | {r.get('MfmaUtil',0):6.2f}%  | {r.get('TA_DATA_STALLED_BY_TC_CYCLES_sum',0)/g:>20.4f}  | {r.get('TCP_TCP_TA_DATA_STALL_CYCLES_sum',0)/g:>20.4f}  | {r.get('TCP_PENDING_STALL_CYCLES_sum',0)/g:>13.4f}")

# Save aggregated to JSON
out = {c: {k: v for k,v in r.items() if not isinstance(v, (list, tuple))} | {"__shape__": list(r["__shape__"])} for c, r in results.items()}
with open(os.path.join(ROOT, "final_aggregated.json"), "w") as f:
    json.dump(out, f, indent=2, default=str)
print(f"\nFinal data saved to {ROOT}/final_aggregated.json")
