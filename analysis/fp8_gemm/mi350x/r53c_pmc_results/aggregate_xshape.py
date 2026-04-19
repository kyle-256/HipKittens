#!/usr/bin/env python3
"""R53C cross-shape aggregator: 70B Down (K=28672) vs 70B Q/O (K=8192) for both RRR and CRR.
Highlights what is K-specific vs layout-specific."""
import csv, os, sys, statistics, json, glob

ROOT = os.path.dirname(os.path.abspath(__file__))

KERNEL_TAGS = {
    "rrr": "rrr_exact_8wave_scaled_kernel<true, 2>",
    "crr": "crr_exact_8wave_scaled_kernel<true, 2>",
}

CELLS = {
    # 70B Down (K=28672)
    "Down_rrr_set1": ("Down", "rrr", "set1", "rrr_set1"),
    "Down_rrr_set3": ("Down", "rrr", "set3", "rrr_set3"),
    "Down_crr_set1": ("Down", "crr", "set1", "crr_set1"),
    "Down_crr_set3": ("Down", "crr", "set3", "crr_set3"),
    # 70B Q/O (K=8192)
    "QO_rrr_set1":   ("QO",   "rrr", "set1", "QO_rrr_set1"),
    "QO_rrr_set3":   ("QO",   "rrr", "set3", "QO_rrr_set3"),
    "QO_crr_set1":   ("QO",   "crr", "set1", "QO_crr_set1"),
    "QO_crr_set3":   ("QO",   "crr", "set3", "QO_crr_set3"),
}

SHAPES = {"Down": (4096, 8192, 28672), "QO": (4096, 8192, 8192)}


def load_dir(dirname, layout):
    by_disp = {}; timing = {}
    tag = KERNEL_TAGS[layout]
    for pmc_dir in sorted(os.listdir(os.path.join(ROOT, dirname))):
        pmc_path = os.path.join(ROOT, dirname, pmc_dir)
        if not os.path.isdir(pmc_path): continue
        csvs = glob.glob(os.path.join(pmc_path, "**", "*counter_collection.csv"), recursive=True)
        for csvp in csvs:
            with open(csvp) as f:
                for row in csv.DictReader(f):
                    if tag not in row.get("Kernel_Name", ""): continue
                    did = int(row["Dispatch_Id"])
                    by_disp.setdefault(did, {})[row["Counter_Name"]] = float(row["Counter_Value"])
                    timing.setdefault(did, {})["start"] = int(row["Start_Timestamp"])
                    timing[did]["end"] = int(row["End_Timestamp"])
    counters = {}; durations = []
    for did, cdict in by_disp.items():
        for c, v in cdict.items():
            counters.setdefault(c, []).append(v)
        durations.append(timing[did]["end"] - timing[did]["start"])
    medians = {c: statistics.median(vs) for c, vs in counters.items()}
    medians["__duration_ns_median__"] = statistics.median(durations) if durations else 0
    return medians


def derive(c, shape):
    M, N, K = shape
    flops = 2 * M * N * K
    grbm = c.get("GRBM_GUI_ACTIVE", 0)
    dur_ns = c["__duration_ns_median__"]
    return {
        "duration_us": dur_ns / 1000.0,
        "TFLOPS": flops / dur_ns / 1000.0 if dur_ns else 0,
        "MfmaUtil%": c.get("MfmaUtil", 0),
        "VALUBusy%": c.get("VALUBusy", 0),
        "L2hit%": 100 * c.get("TCC_HIT_sum", 0) / max(1, c.get("TCC_HIT_sum", 0) + c.get("TCC_MISS_sum", 0)),
        "MFMA": c.get("SQ_INSTS_MFMA", 0),
        "VALU": c.get("SQ_INSTS_VALU", 0),
        "LDS_inst": c.get("SQ_INSTS_LDS", 0),
        "SALU": c.get("SQ_INSTS_SALU", 0),
        "VMEM_RD": c.get("SQ_INSTS_VMEM_RD", 0),
        "GRBM": grbm,
        "WAIT_INST_LDS_cyc4": c.get("SQ_WAIT_INST_LDS", 0),
        "WAIT_INST_ANY_cyc4": c.get("SQ_WAIT_INST_ANY", 0),
        "WAIT_ANY_cyc4": c.get("SQ_WAIT_ANY", 0),
        "TA_DATA_TC_perGRBM": c.get("TA_DATA_STALLED_BY_TC_CYCLES_sum", 0) / grbm if grbm else 0,
        "TA_ADDR_TC_perGRBM": c.get("TA_ADDR_STALLED_BY_TC_CYCLES_sum", 0) / grbm if grbm else 0,
        "TCP_PEND_perGRBM": c.get("TCP_PENDING_STALL_CYCLES_sum", 0) / grbm if grbm else 0,
        "TCP_RFIFO_perGRBM": c.get("TCP_RFIFO_STALL_CYCLES_sum", 0) / grbm if grbm else 0,
        "VMEM_FIFO_FULL_perGRBM": c.get("SQ_VMEM_TA_ADDR_FIFO_FULL", 0) / grbm if grbm else 0,
        "TCC_BUSY_perGRBM": c.get("TCC_BUSY_sum", 0) / grbm if grbm else 0,
    }


cells = {}
for cname, (shape, layout, _set, dir_) in CELLS.items():
    p = os.path.join(ROOT, dir_)
    if not os.path.isdir(p):
        continue
    cells[cname] = load_dir(dir_, layout)

# Combine sets per (shape, layout)
combined = {}
for cname, (shape, layout, _set, _dir) in CELLS.items():
    if cname not in cells: continue
    key = f"{shape}_{layout}"
    merged = combined.setdefault(key, {"__duration_ns_median__": []})
    for k, v in cells[cname].items():
        if k == "__duration_ns_median__":
            merged["__duration_ns_median__"].append(v)
        else:
            merged[k] = v
for key in combined:
    durs = combined[key]["__duration_ns_median__"]
    combined[key]["__duration_ns_median__"] = statistics.median(durs) if durs else 0

# Print 4-column table
order = ["Down_rrr", "Down_crr", "QO_rrr", "QO_crr"]
header_shapes = {k: ("70BDown_RRR" if k=="Down_rrr" else "70BDown_CRR" if k=="Down_crr" else "70BQO_RRR" if k=="QO_rrr" else "70BQO_CRR") for k in order}
derived = {}
for k in order:
    if k in combined:
        shape_label = "Down" if k.startswith("Down") else "QO"
        derived[k] = derive(combined[k], SHAPES[shape_label])

print("=" * 120)
hdr = f"{'Metric':<28}"
for k in order:
    hdr += f"{header_shapes[k]:>22}"
print(hdr)
print("=" * 120)
keys = ["duration_us","TFLOPS","MfmaUtil%","VALUBusy%","L2hit%",
        "MFMA","VALU","LDS_inst","SALU","VMEM_RD",
        "WAIT_INST_LDS_cyc4","WAIT_INST_ANY_cyc4","WAIT_ANY_cyc4","GRBM",
        "TA_DATA_TC_perGRBM","TA_ADDR_TC_perGRBM","TCP_PEND_perGRBM","TCP_RFIFO_perGRBM",
        "VMEM_FIFO_FULL_perGRBM","TCC_BUSY_perGRBM"]
for kk in keys:
    line = f"{kk:<28}"
    for k in order:
        v = derived.get(k, {}).get(kk, 0)
        if isinstance(v, float):
            if abs(v) > 1e9: s = f"{v:>22.3e}"
            elif abs(v) > 100: s = f"{v:>22.1f}"
            elif abs(v) > 1: s = f"{v:>22.3f}"
            else: s = f"{v:>22.5f}"
        else:
            s = f"{str(v):>22}"
        line += s
    print(line)

# Per-K-pair normalization (divide by K-pairs = K/16; both RRR and CRR consume K/16 K-pairs total per CTA)
print()
print("Per-K-pair (K-pairs = K/16) normalized counts:")
print(f"{'Metric':<28}", end="")
for k in order:
    print(f"{header_shapes[k]:>22}", end="")
print()
for k in order:
    shape_label = "Down" if k.startswith("Down") else "QO"
    K = SHAPES[shape_label][2]
    # CTA grid for this kernel
    M, N, K_ = SHAPES[shape_label]
    # BLK=256 by default, M_per_CTA=128, N_per_CTA=128 → CTAs = (M/128) * (N/128)
    # K-iterations per CTA = K/16 = K-pairs (for double-pump kernels = K/32 for RRR? need to verify)
    # We just want comparable CRR vs RRR for the same K. Use K/16 as nominal K-pair.
    K_pairs_per_CTA = K // 16
    # Total K-pairs across all CTAs = K_pairs_per_CTA * num_CTAs. SQ counters are per-cell totals.
    waves = (M // 128) * (N // 128) * 16  # 16 waves per CTA (8 wave * 2 wgrp? actually 8 waves per CTA)
    # Just take total counters / (M/128 * N/128 * K/16) = per-K-pair-per-CTA
    num_CTAs = (M // 128) * (N // 128)
    norm = num_CTAs * K_pairs_per_CTA
    derived[k]["_norm"] = norm
print()
for kk in ["MFMA", "VALU", "LDS_inst", "SALU"]:
    line = f"{kk+'/Kpair/CTA':<28}"
    for k in order:
        v = derived.get(k, {}).get(kk, 0)
        norm = derived.get(k, {}).get("_norm", 1)
        per = v / norm if norm else 0
        line += f"{per:>22.3f}"
    print(line)

with open(os.path.join(ROOT, "aggregated_xshape.json"), "w") as f:
    json.dump({k: derived[k] for k in derived}, f, indent=2)
print("\nWrote aggregated_xshape.json")
