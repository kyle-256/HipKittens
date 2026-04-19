#!/usr/bin/env python3
"""Aggregate set3 PMC results."""
import csv, os, statistics, json
ROOT = os.path.dirname(os.path.abspath(__file__))

def load_cell(dirname):
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

CELLS = {
    "v2_8B_GateUp": "v2_8B_set3",
    "v2_70B_QO":    "v2_70B_set3",
}
results = {c: load_cell(d) for c, d in CELLS.items()}

# Also add GRBM_GUI_ACTIVE from set1 results
set1 = json.load(open(os.path.join(ROOT, "aggregated.json")))
results["v2_8B_GateUp"]["GRBM_GUI_ACTIVE"] = set1["v2_8B_GateUp_default"]["raw"]["GRBM_GUI_ACTIVE"]
results["v2_70B_QO"]["GRBM_GUI_ACTIVE"] = set1["v2_70B_QO_default"]["raw"]["GRBM_GUI_ACTIVE"]

print(f"\n{'Stall counter (raw)':<45}", end="")
for c in CELLS:
    print(f"{c:>20}", end="")
print()
print("-"*85)

stall_keys = [
    "TA_ADDR_STALLED_BY_TC_CYCLES_sum",
    "TA_ADDR_STALLED_BY_TD_CYCLES_sum",
    "TA_DATA_STALLED_BY_TC_CYCLES_sum",
    "TCP_TCP_TA_DATA_STALL_CYCLES_sum",
    "TCC_BUBBLE_sum",
    "TCC_BUSY_sum",
    "TCP_PENDING_STALL_CYCLES_sum",
    "TCP_RFIFO_STALL_CYCLES_sum",
    "SQ_LDS_DATA_FIFO_FULL",
    "SQ_VMEM_TA_ADDR_FIFO_FULL",
]
for key in stall_keys:
    print(f"{key:<45}", end="")
    for c in CELLS:
        v = results[c].get(key, 0)
        print(f"{v:>20.0f}", end="")
    print()

# Normalize per GRBM cycle
print(f"\n{'Per GRBM_GUI_ACTIVE cycle':<45}", end="")
for c in CELLS:
    print(f"{c:>20}", end="")
print()
print("-"*85)
for key in stall_keys:
    print(f"{key:<45}", end="")
    for c in CELLS:
        v = results[c].get(key, 0) / results[c]["GRBM_GUI_ACTIVE"]
        print(f"{v:>20.4f}", end="")
    print()

# Normalize per total active cycle (TCC_BUSY_sum is summed across 16 channels - normalize differently)
print(f"\n{'Stall ratios':<45}", end="")
for c in CELLS:
    print(f"{c:>20}", end="")
print()
print("-"*85)
for c in CELLS:
    r = results[c]
    grbm = r["GRBM_GUI_ACTIVE"]
print(f"\n{'TA_ADDR_stall_by_TC / TA_ADDR_TD_stall':<45}", end="")
for c in CELLS:
    r = results[c]
    a = r.get("TA_ADDR_STALLED_BY_TC_CYCLES_sum", 0)
    b = r.get("TA_ADDR_STALLED_BY_TD_CYCLES_sum", 0)
    print(f"{a/(a+b)*100 if (a+b)>0 else 0:>20.2f}", end="")
print()

# print TCC_BUBBLE / TCC_BUSY
print(f"{'TCC_BUBBLE / TCC_BUSY (%)':<45}", end="")
for c in CELLS:
    r = results[c]
    bub = r.get("TCC_BUBBLE_sum", 0)
    bus = r.get("TCC_BUSY_sum", 0)
    print(f"{bub/bus*100 if bus>0 else 0:>20.2f}", end="")
print()
