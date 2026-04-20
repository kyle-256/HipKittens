#!/usr/bin/env python3
"""R56B aggregator — adapted from R55F/aggregate.py for CRR layout.

Cells: 70B_GateUp (4096x28672x8192), 70B_QO (4096x8192x8192), 70B_Down (4096x8192x28672)
Dtypes: fp8 (crr_exact_8wave_kernel), mxfp8 (crr_exact_8wave_scaled_kernel<true,2>)
Sets: set1 (utilization), set3 (stall attribution)
Layout: CRR
"""
import csv, os, statistics, json, glob

ROOT = os.path.dirname(os.path.abspath(__file__))

KERNEL_TAGS = {
    "mxfp8": ["crr_exact_8wave_scaled_kernel"],
    "fp8":   ["crr_exact_8wave_kernel"],
}
MAX_DURATION_US = 5_000.0  # drop fallback gemm_tail_kernel runs

CELLS = [
    ("70B_GateUp", 4096, 28672, 8192),
    ("70B_QO",     4096,  8192, 8192),
    ("70B_Down",   4096,  8192, 28672),
]
SETS = ["set1", "set3"]
DTYPES = ["fp8", "mxfp8"]


def load_cell(dirname, dtype):
    by_disp = {}
    timing = {}
    knames = set()
    tags = KERNEL_TAGS[dtype]
    full = os.path.join(ROOT, dirname)
    if not os.path.isdir(full):
        return None
    for pmc_dir in sorted(os.listdir(full)):
        pmc_path = os.path.join(full, pmc_dir)
        if not os.path.isdir(pmc_path): continue
        csvs = glob.glob(os.path.join(pmc_path, "**", "*counter_collection.csv"), recursive=True)
        for csvp in csvs:
            with open(csvp) as f:
                rdr = csv.DictReader(f)
                for row in rdr:
                    kn = row.get("Kernel_Name", "")
                    if not any(t in kn for t in tags):
                        continue
                    dur_us = (int(row["End_Timestamp"]) - int(row["Start_Timestamp"])) / 1000.0
                    if dur_us > MAX_DURATION_US:
                        continue
                    did = int(row["Dispatch_Id"])
                    cn = row["Counter_Name"]
                    cv = float(row["Counter_Value"])
                    by_disp.setdefault(did, {})[cn] = cv
                    timing.setdefault(did, {})["start"] = int(row["Start_Timestamp"])
                    timing[did]["end"] = int(row["End_Timestamp"])
                    knames.add(kn)
    counters = {}
    durations = []
    for did, cdict in by_disp.items():
        for c, v in cdict.items():
            counters.setdefault(c, []).append(v)
        durations.append(timing[did]["end"] - timing[did]["start"])
    medians = {c: statistics.median(vs) for c, vs in counters.items()}
    medians["__duration_ns_median__"] = statistics.median(durations) if durations else 0
    medians["__num_dispatches__"] = len(by_disp)
    medians["__kernel_names__"] = sorted(knames)
    return medians


def derive(c, M, N, K):
    flops = 2 * M * N * K
    grbm = c.get("GRBM_GUI_ACTIVE", 0)
    waves = c.get("SQ_WAVES", 0)
    sq_busy = c.get("SQ_BUSY_CYCLES", 0)
    dur_ns = c["__duration_ns_median__"]
    derived = {
        "duration_us": dur_ns / 1000.0,
        "achieved_TFLOPS": flops / dur_ns / 1000.0 if dur_ns else 0,
        "GRBM_GUI_ACTIVE": grbm,
        "SQ_WAVES": waves,
        "SQ_INSTS_VALU": c.get("SQ_INSTS_VALU", 0),
        "SQ_INSTS_MFMA": c.get("SQ_INSTS_MFMA", 0),
        "SQ_INSTS_LDS": c.get("SQ_INSTS_LDS", 0),
        "SQ_INSTS_VMEM_RD": c.get("SQ_INSTS_VMEM_RD", 0),
        "SQ_INSTS_SALU": c.get("SQ_INSTS_SALU", 0),
        "SQ_VALU_MFMA_BUSY_CYCLES": c.get("SQ_VALU_MFMA_BUSY_CYCLES", 0),
        "SQ_BUSY_CYCLES": sq_busy,
        "SQ_WAIT_INST_LDS_cyc4": c.get("SQ_WAIT_INST_LDS", 0),
        "SQ_WAIT_INST_ANY_cyc4": c.get("SQ_WAIT_INST_ANY", 0),
        "SQ_WAIT_ANY_cyc4": c.get("SQ_WAIT_ANY", 0),
        "SQ_LDS_BANK_CONFLICT_cyc": c.get("SQ_LDS_BANK_CONFLICT", 0),
        "SQ_ACTIVE_INST_VALU": c.get("SQ_ACTIVE_INST_VALU", 0),
        "SQ_ACTIVE_INST_VMEM": c.get("SQ_ACTIVE_INST_VMEM", 0),
        "SQ_ACTIVE_INST_LDS": c.get("SQ_ACTIVE_INST_LDS", 0),
        "TCC_HIT_sum": c.get("TCC_HIT_sum", 0),
        "TCC_MISS_sum": c.get("TCC_MISS_sum", 0),
        "TCC_hit_pct": 100 * c.get("TCC_HIT_sum", 0) / max(1, c.get("TCC_HIT_sum", 0) + c.get("TCC_MISS_sum", 0)),
        "FetchSize_KB": c.get("FetchSize", 0),
        "VALUBusy_pct": c.get("VALUBusy", 0),
        "MfmaUtil_pct": c.get("MfmaUtil", 0),
        "MemUnitStalled_pct": c.get("MemUnitStalled", 0),
        "MemUnitBusy_pct": c.get("MemUnitBusy", 0),
        "LDSBankConflict_pct": c.get("LDSBankConflict", 0),
        "ALUStalledByLDS_pct": c.get("ALUStalledByLDS", 0),
        "L2CacheHit_pct": c.get("L2CacheHit", 0),
        # set3 stalls
        "TA_ADDR_STALLED_BY_TC_sum": c.get("TA_ADDR_STALLED_BY_TC_CYCLES_sum", 0),
        "TA_ADDR_STALLED_BY_TD_sum": c.get("TA_ADDR_STALLED_BY_TD_CYCLES_sum", 0),
        "TA_DATA_STALLED_BY_TC_sum": c.get("TA_DATA_STALLED_BY_TC_CYCLES_sum", 0),
        "TCP_TCP_TA_DATA_STALL_sum": c.get("TCP_TCP_TA_DATA_STALL_CYCLES_sum", 0),
        "TCC_BUBBLE_sum": c.get("TCC_BUBBLE_sum", 0),
        "TCC_BUSY_sum": c.get("TCC_BUSY_sum", 0),
        "TCP_PENDING_STALL_sum": c.get("TCP_PENDING_STALL_CYCLES_sum", 0),
        "TCP_RFIFO_STALL_sum": c.get("TCP_RFIFO_STALL_CYCLES_sum", 0),
        "SQ_LDS_DATA_FIFO_FULL": c.get("SQ_LDS_DATA_FIFO_FULL", 0),
        "SQ_VMEM_TA_ADDR_FIFO_FULL": c.get("SQ_VMEM_TA_ADDR_FIFO_FULL", 0),
        # /GRBM normalized
        "TA_ADDR_STALL_TC_per_grbm": c.get("TA_ADDR_STALLED_BY_TC_CYCLES_sum", 0) / grbm if grbm else 0,
        "TA_DATA_STALL_TC_per_grbm": c.get("TA_DATA_STALLED_BY_TC_CYCLES_sum", 0) / grbm if grbm else 0,
        "TCP_TA_DATA_STALL_per_grbm": c.get("TCP_TCP_TA_DATA_STALL_CYCLES_sum", 0) / grbm if grbm else 0,
        "TCP_PENDING_per_grbm": c.get("TCP_PENDING_STALL_CYCLES_sum", 0) / grbm if grbm else 0,
        "TCP_RFIFO_per_grbm": c.get("TCP_RFIFO_STALL_CYCLES_sum", 0) / grbm if grbm else 0,
        "TCC_BUBBLE_per_grbm": c.get("TCC_BUBBLE_sum", 0) / grbm if grbm else 0,
        "TCC_BUSY_per_grbm": c.get("TCC_BUSY_sum", 0) / grbm if grbm else 0,
        "VMEM_FIFO_FULL_per_grbm": c.get("SQ_VMEM_TA_ADDR_FIFO_FULL", 0) / grbm if grbm else 0,
        "LDS_FIFO_FULL_per_grbm": c.get("SQ_LDS_DATA_FIFO_FULL", 0) / grbm if grbm else 0,
        "LDS_BANK_CONFLICT_per_grbm": c.get("SQ_LDS_BANK_CONFLICT", 0) / grbm if grbm else 0,
    }
    return derived


def merge_sets(label, dtype):
    merged = {}
    durations = []
    knames = set()
    for s in SETS:
        d = load_cell(f"{label}_{dtype}_crr_{s}", dtype)
        if d is None:
            continue
        if d.get("__num_dispatches__", 0) == 0 or d.get("__duration_ns_median__", 0) == 0:
            continue
        for k, v in d.items():
            if k.startswith("__"):
                if k == "__duration_ns_median__":
                    durations.append(v)
                elif k == "__kernel_names__":
                    knames.update(v)
                continue
            merged[k] = v
    merged["__duration_ns_median__"] = statistics.median(durations) if durations else 0
    merged["__kernel_names__"] = sorted(knames)
    return merged


results = {}
for label, M, N, K in CELLS:
    results[label] = {"M": M, "N": N, "K": K}
    for dtype in DTYPES:
        merged = merge_sets(label, dtype)
        derived = derive(merged, M, N, K)
        results[label][dtype] = {
            "raw": {k: v for k, v in merged.items() if not k.startswith("__")},
            "derived": derived,
            "kernel_names": merged.get("__kernel_names__", []),
        }


def fmt(v):
    if isinstance(v, float):
        if abs(v) > 1e9: return f"{v:>14.3e}"
        if abs(v) > 100: return f"{v:>14.1f}"
        if abs(v) > 1: return f"{v:>14.3f}"
        return f"{v:>14.5f}"
    return f"{str(v):>14}"

KEYS = [
    "duration_us", "achieved_TFLOPS",
    "GRBM_GUI_ACTIVE", "SQ_BUSY_CYCLES",
    "SQ_INSTS_VALU", "SQ_INSTS_MFMA", "SQ_INSTS_LDS", "SQ_INSTS_VMEM_RD", "SQ_INSTS_SALU",
    "SQ_VALU_MFMA_BUSY_CYCLES",
    "SQ_WAIT_INST_LDS_cyc4", "SQ_WAIT_INST_ANY_cyc4", "SQ_WAIT_ANY_cyc4",
    "SQ_ACTIVE_INST_VALU", "SQ_ACTIVE_INST_VMEM", "SQ_ACTIVE_INST_LDS",
    "TCC_HIT_sum", "TCC_MISS_sum", "TCC_hit_pct", "FetchSize_KB",
    "VALUBusy_pct", "MfmaUtil_pct", "MemUnitStalled_pct", "MemUnitBusy_pct",
    "LDSBankConflict_pct", "ALUStalledByLDS_pct", "L2CacheHit_pct",
    "TA_ADDR_STALL_TC_per_grbm",
    "TA_DATA_STALL_TC_per_grbm",
    "TCP_TA_DATA_STALL_per_grbm",
    "TCP_PENDING_per_grbm",
    "TCP_RFIFO_per_grbm",
    "TCC_BUBBLE_per_grbm",
    "TCC_BUSY_per_grbm",
    "VMEM_FIFO_FULL_per_grbm",
    "LDS_FIFO_FULL_per_grbm",
    "LDS_BANK_CONFLICT_per_grbm",
]

for label, M, N, K in CELLS:
    cell = results[label]
    fp8d = cell["fp8"]["derived"]
    mxd = cell["mxfp8"]["derived"]
    print()
    print("=" * 100)
    print(f"=== {label} M={M} N={N} K={K} (CRR) ===")
    print("=" * 100)
    print(f"{'Metric':<38}{'FP8':>14}{'MXFP8':>14}{'MX-FP8':>14}{'MX/FP8':>14}")
    print("-" * 100)
    for k in KEYS:
        f = fp8d.get(k, 0)
        m = mxd.get(k, 0)
        d = m - f
        r = (m / f) if f else 0
        print(f"{k:<38}{fmt(f)}{fmt(m)}{fmt(d)}{fmt(r)}")
    print(f"  MXFP8 kernels: {cell['mxfp8']['kernel_names']}")
    print(f"  FP8 kernels:   {cell['fp8']['kernel_names']}")

print()
print("=" * 100)
print("=== SUMMARY MXFP8/FP8 ratio per cell (CRR) ===")
print("=" * 100)
print(f"{'Cell':<14}{'K':>8}{'FP8 TFLOPS':>14}{'MX TFLOPS':>14}{'MX/FP8':>10}{'MfmaUtil FP8':>14}{'MfmaUtil MX':>14}{'TCChit FP8':>12}{'TCChit MX':>12}")
for label, M, N, K in CELLS:
    fp8d = results[label]["fp8"]["derived"]
    mxd  = results[label]["mxfp8"]["derived"]
    ratio = (mxd["achieved_TFLOPS"] / fp8d["achieved_TFLOPS"] * 100) if fp8d["achieved_TFLOPS"] else 0
    print(f"{label:<14}{K:>8}{fp8d['achieved_TFLOPS']:>14.1f}{mxd['achieved_TFLOPS']:>14.1f}{ratio:>9.1f}%{fp8d['MfmaUtil_pct']:>14.2f}{mxd['MfmaUtil_pct']:>14.2f}{fp8d['TCC_hit_pct']:>12.2f}{mxd['TCC_hit_pct']:>12.2f}")

with open(os.path.join(ROOT, "aggregated.json"), "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nWrote aggregated.json")
