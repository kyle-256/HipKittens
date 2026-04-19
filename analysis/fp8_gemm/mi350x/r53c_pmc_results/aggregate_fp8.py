#!/usr/bin/env python3
"""R53C FP8 baseline aggregator. Compares FP8 RRR/CRR at K=8192 vs K=28672."""
import csv, os, statistics, json, glob

ROOT = os.path.dirname(os.path.abspath(__file__))

KERNEL_TAGS = {
    "rrr": "rrr_exact_8wave_kernel",  # FP8 unscaled
    "crr": "crr_exact_8wave_kernel",
}

CELLS = {
    "Down_rrr": ("Down", "rrr", "fp8_Down_rrr_set1"),
    "Down_crr": ("Down", "crr", "fp8_Down_crr_set1"),
    "QO_rrr":   ("QO",   "rrr", "fp8_QO_rrr_set1"),
    "QO_crr":   ("QO",   "crr", "fp8_QO_crr_set1"),
}

SHAPES = {"Down": (4096, 8192, 28672), "QO": (4096, 8192, 8192)}


def load_dir(dirname, layout):
    by_disp = {}; timing = {}
    tag = KERNEL_TAGS[layout]
    for pmc_dir in sorted(os.listdir(os.path.join(ROOT, dirname))):
        pmc_path = os.path.join(ROOT, dirname, pmc_dir)
        if not os.path.isdir(pmc_path): continue
        for csvp in glob.glob(os.path.join(pmc_path, "**", "*counter_collection.csv"), recursive=True):
            with open(csvp) as f:
                for row in csv.DictReader(f):
                    if tag not in row.get("Kernel_Name", ""): continue
                    did = int(row["Dispatch_Id"])
                    by_disp.setdefault(did, {})[row["Counter_Name"]] = float(row["Counter_Value"])
                    timing.setdefault(did, {})["start"] = int(row["Start_Timestamp"])
                    timing[did]["end"] = int(row["End_Timestamp"])
    counters = {}; durations = []
    for did, cdict in by_disp.items():
        for c, v in cdict.items(): counters.setdefault(c, []).append(v)
        durations.append(timing[did]["end"] - timing[did]["start"])
    medians = {c: statistics.median(vs) for c, vs in counters.items()}
    medians["__duration_ns_median__"] = statistics.median(durations) if durations else 0
    return medians


cells = {}
for cname, (shape, layout, dir_) in CELLS.items():
    p = os.path.join(ROOT, dir_)
    if not os.path.isdir(p): continue
    cells[cname] = load_dir(dir_, layout)

print("=" * 100)
print(f"{'Metric':<20}{'FP8 70BDown_RRR':>20}{'FP8 70BDown_CRR':>20}{'FP8 70BQO_RRR':>20}{'FP8 70BQO_CRR':>20}")
print("=" * 100)
def derive(c, shape):
    M,N,K = shape
    flops = 2*M*N*K
    dur_ns = c["__duration_ns_median__"]
    return {
        "duration_us": dur_ns/1000.0,
        "TFLOPS": flops/dur_ns/1000.0 if dur_ns else 0,
        "MfmaUtil%": c.get("MfmaUtil",0),
        "VALUBusy%": c.get("VALUBusy",0),
        "L2hit%": 100*c.get("TCC_HIT_sum",0)/max(1,c.get("TCC_HIT_sum",0)+c.get("TCC_MISS_sum",0)),
        "MFMA": c.get("SQ_INSTS_MFMA",0),
        "VALU": c.get("SQ_INSTS_VALU",0),
        "LDS_inst": c.get("SQ_INSTS_LDS",0),
        "SALU": c.get("SQ_INSTS_SALU",0),
        "VMEM_RD": c.get("SQ_INSTS_VMEM_RD",0),
        "GRBM": c.get("GRBM_GUI_ACTIVE",0),
        "WAIT_INST_LDS": c.get("SQ_WAIT_INST_LDS",0),
        "WAIT_INST_ANY": c.get("SQ_WAIT_INST_ANY",0),
    }

derived = {}
order = ["Down_rrr","Down_crr","QO_rrr","QO_crr"]
for k in order:
    if k in cells:
        shape = "Down" if k.startswith("Down") else "QO"
        derived[k] = derive(cells[k], SHAPES[shape])
keys = ["duration_us","TFLOPS","MfmaUtil%","VALUBusy%","L2hit%","MFMA","VALU","LDS_inst","SALU","WAIT_INST_LDS","WAIT_INST_ANY","GRBM"]
for kk in keys:
    line = f"{kk:<20}"
    for k in order:
        v = derived.get(k,{}).get(kk,0)
        if isinstance(v,float):
            if abs(v)>1e9: s = f"{v:>20.3e}"
            elif abs(v)>100: s = f"{v:>20.1f}"
            elif abs(v)>1: s = f"{v:>20.3f}"
            else: s = f"{v:>20.5f}"
        else: s = f"{str(v):>20}"
        line += s
    print(line)
print()

# Now FP8 baseline ratio analysis
print("FP8 baseline ratio (Down/QO) for same layout — does FP8 scale better at higher K?")
for lay in ("rrr","crr"):
    d_down = derived.get(f"Down_{lay}", {})
    d_qo = derived.get(f"QO_{lay}", {})
    if d_down and d_qo:
        # K=28672/K=8192 = 3.5x flops; ideal would scale exactly 3.5x in time
        ratio_t = d_down["duration_us"] / d_qo["duration_us"]
        ratio_tf = d_down["TFLOPS"] / d_qo["TFLOPS"]
        ratio_util = d_down["MfmaUtil%"] / d_qo["MfmaUtil%"]
        print(f"  FP8 {lay.upper()}: dur ratio Down/QO = {ratio_t:.3f} (ideal=3.5), TFLOPS ratio = {ratio_tf:.3f}, MfmaUtil ratio = {ratio_util:.3f}")
