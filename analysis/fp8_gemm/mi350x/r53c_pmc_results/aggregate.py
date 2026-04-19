#!/usr/bin/env python3
"""R53C aggregator — adapted from R52P. Aggregates rocprofv3 PMC outputs.
Handles new rocprofv3 output naming with hostname subdir + pid prefix."""
import csv, os, sys, statistics, json, glob

ROOT = os.path.dirname(os.path.abspath(__file__))

# Filter only the kernel of interest (rrr_exact_8wave_scaled or crr_exact_8wave_scaled with PRESHUFFLED=true PACK=2)
KERNEL_TAGS = {
    "rrr": "rrr_exact_8wave_scaled_kernel<true, 2>",
    "crr": "crr_exact_8wave_scaled_kernel<true, 2>",
}

CELLS = {
    "rrr_set1": ("rrr", "set1"),
    "rrr_set3": ("rrr", "set3"),
    "crr_set1": ("crr", "set1"),
    "crr_set3": ("crr", "set3"),
}

SHAPE = (4096, 8192, 28672)


def load_cell(dirname, layout):
    by_disp = {}  # dispatch_id -> {counter -> value}
    timing = {}
    knames = set()
    tag = KERNEL_TAGS[layout]
    for pmc_dir in sorted(os.listdir(os.path.join(ROOT, dirname))):
        pmc_path = os.path.join(ROOT, dirname, pmc_dir)
        if not os.path.isdir(pmc_path): continue
        # Find csv file (under host subdir / *_counter_collection.csv)
        csvs = glob.glob(os.path.join(pmc_path, "**", "*counter_collection.csv"), recursive=True)
        for csvp in csvs:
            with open(csvp) as f:
                rdr = csv.DictReader(f)
                for row in rdr:
                    if tag not in row.get("Kernel_Name", ""):
                        continue
                    did = int(row["Dispatch_Id"])
                    cn = row["Counter_Name"]
                    cv = float(row["Counter_Value"])
                    by_disp.setdefault(did, {})[cn] = cv
                    timing.setdefault(did, {})["start"] = int(row["Start_Timestamp"])
                    timing[did]["end"] = int(row["End_Timestamp"])
                    knames.add(row["Kernel_Name"])
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


def derive(c):
    M, N, K = SHAPE
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
        "SQ_INSTS_VMEM_WR": c.get("SQ_INSTS_VMEM_WR", 0),
        "SQ_INSTS_SALU": c.get("SQ_INSTS_SALU", 0),
        "SQ_VALU_MFMA_BUSY_CYCLES": c.get("SQ_VALU_MFMA_BUSY_CYCLES", 0),
        "SQ_BUSY_CYCLES": sq_busy,
        "SQ_WAIT_INST_LDS_cyc4": c.get("SQ_WAIT_INST_LDS", 0),
        "SQ_WAIT_INST_ANY_cyc4": c.get("SQ_WAIT_INST_ANY", 0),
        "SQ_WAIT_ANY_cyc4": c.get("SQ_WAIT_ANY", 0),
        "SQ_LDS_BANK_CONFLICT_cyc": c.get("SQ_LDS_BANK_CONFLICT", 0),
        "TCC_HIT_sum": c.get("TCC_HIT_sum", 0),
        "TCC_MISS_sum": c.get("TCC_MISS_sum", 0),
        "TCC_hit_pct": 100 * c.get("TCC_HIT_sum", 0) / max(1, c.get("TCC_HIT_sum", 0) + c.get("TCC_MISS_sum", 0)),
        "TCP_PENDING_STALL_sum": c.get("TCP_PENDING_STALL_CYCLES_sum", 0),
        "TCP_LFIFO_STALL_sum": c.get("TCP_LFIFO_STALL_CYCLES_sum", 0),
        "TCP_READ_TAGCONFLICT_STALL_sum": c.get("TCP_READ_TAGCONFLICT_STALL_CYCLES_sum", 0),
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
        "TCP_PENDING_STALL_sum_v3": c.get("TCP_PENDING_STALL_CYCLES_sum", 0),
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


# Merge set1+set3 cells into rrr_combined and crr_combined
data = {}
for cellname, (layout, pmcset) in CELLS.items():
    p = os.path.join(ROOT, cellname)
    if not os.path.isdir(p):
        print(f"  missing {cellname}")
        continue
    data[cellname] = load_cell(cellname, layout)

# Merge sets per layout (counters from different sets are different)
combined = {}
for layout in ("rrr", "crr"):
    merged = {}
    for cellname, (lay, _set) in CELLS.items():
        if lay != layout: continue
        if cellname not in data: continue
        d = data[cellname]
        for k, v in d.items():
            if k.startswith("__"):
                if k == "__duration_ns_median__":
                    # take min; we want the actual kernel duration. They should be similar.
                    merged.setdefault(k, []).append(v)
                continue
            merged[k] = v  # later sets overwrite; ok since counters don't overlap meaningfully
    if "__duration_ns_median__" in merged:
        merged["__duration_ns_median__"] = statistics.median(merged["__duration_ns_median__"])
    combined[layout] = merged

# Print comparison table
print("=" * 100)
print(f"{'Metric':<42}{'RRR':>20}{'CRR':>20}{'CRR-RRR':>18}")
print("=" * 100)

derived = {lay: derive(combined[lay]) for lay in combined}

keys = [
    "duration_us", "achieved_TFLOPS",
    "GRBM_GUI_ACTIVE", "SQ_WAVES", "SQ_BUSY_CYCLES",
    "SQ_INSTS_VALU", "SQ_INSTS_MFMA", "SQ_INSTS_LDS", "SQ_INSTS_VMEM_RD", "SQ_INSTS_SALU",
    "SQ_VALU_MFMA_BUSY_CYCLES",
    "SQ_WAIT_INST_LDS_cyc4", "SQ_WAIT_INST_ANY_cyc4", "SQ_WAIT_ANY_cyc4",
    "SQ_LDS_BANK_CONFLICT_cyc",
    "TCC_HIT_sum", "TCC_MISS_sum", "TCC_hit_pct",
    "FetchSize_KB",
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
for k in keys:
    rrr_v = derived.get("rrr", {}).get(k, 0)
    crr_v = derived.get("crr", {}).get(k, 0)
    delta = crr_v - rrr_v
    def fmt(v):
        if isinstance(v, float):
            if abs(v) > 1e9: return f"{v:>20.3e}"
            if abs(v) > 100: return f"{v:>20.1f}"
            if abs(v) > 1: return f"{v:>20.3f}"
            return f"{v:>20.5f}"
        return f"{str(v):>20}"
    print(f"{k:<42}{fmt(rrr_v)}{fmt(crr_v)}{fmt(delta)}")

print("=" * 100)
for lay, d in combined.items():
    print(f"  {lay}: kernels={d.get('__kernel_names__', '?')}")

# JSON dump
out = {
    lay: {"raw": {k: v for k, v in combined[lay].items() if not k.startswith("__")},
          "derived": derived[lay]}
    for lay in combined
}
with open(os.path.join(ROOT, "aggregated.json"), "w") as f:
    json.dump(out, f, indent=2)
print(f"Wrote aggregated.json")
