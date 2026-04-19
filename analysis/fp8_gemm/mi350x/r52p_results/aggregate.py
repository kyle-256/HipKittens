#!/usr/bin/env python3
"""Aggregate rocprofv3 PMC outputs into a single counter table per cell."""
import csv, os, sys, statistics, json

CELLS = {
    "v2_8B_GateUp_default":  ("v1_8B_gateup",          "V2 RRR (default), M=4096 N=14336 K=4096"),
    "v1_8B_GateUp_forced":   ("v1_8B_gateup_forced",   "V1 RRR (forced),  M=4096 N=14336 K=4096"),
    "v2_70B_QO_default":     ("v2_70B_QO",             "V2 RRR (default), M=4096 N=8192  K=8192"),
    "v1_70B_QO_forced":      ("v1_70B_QO_forced",      "V1 RRR (forced),  M=4096 N=8192  K=8192"),
}

ROOT = os.path.dirname(os.path.abspath(__file__))

def load_cell(dirname):
    by_disp = {}  # dispatch_id -> {counter -> value}
    timing = {}
    knames = set()
    for pmc_dir in sorted(os.listdir(os.path.join(ROOT, dirname))):
        path = os.path.join(ROOT, dirname, pmc_dir, "pmc_counter_collection.csv")
        if not os.path.exists(path): continue
        with open(path) as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                did = int(row["Dispatch_Id"])
                cn = row["Counter_Name"]
                cv = float(row["Counter_Value"])
                by_disp.setdefault(did, {})[cn] = cv
                timing.setdefault(did, {})["start"] = int(row["Start_Timestamp"])
                timing[did]["end"] = int(row["End_Timestamp"])
                knames.add(row["Kernel_Name"])
    # For each counter, take the median across all dispatches
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

def derive(c, shape):
    """Derive useful breakdowns. shape = (M,N,K)."""
    M,N,K = shape
    flops = 2 * M * N * K
    SIMD_NUM = 256  # MI355X = 8 SE * 32 SIMD = 256 SIMDs, but SQ counters are aggregated per-SE in some cases
    # GFX950 / MI355X actually has more — let's compute from GRBM_GUI_ACTIVE
    grbm = c.get("GRBM_GUI_ACTIVE", 0)
    waves = c.get("SQ_WAVES", 0)
    valu = c.get("SQ_INSTS_VALU", 0)
    mfma = c.get("SQ_INSTS_MFMA", 0)
    lds = c.get("SQ_INSTS_LDS", 0)
    vmem_rd = c.get("SQ_INSTS_VMEM_RD", 0)
    vmem_wr = c.get("SQ_INSTS_VMEM_WR", 0)
    salu = c.get("SQ_INSTS_SALU", 0)
    wait_lds = c.get("SQ_WAIT_INST_LDS", 0)
    wait_any = c.get("SQ_WAIT_INST_ANY", 0)
    sq_busy = c.get("SQ_BUSY_CYCLES", 0)
    mfma_busy = c.get("SQ_VALU_MFMA_BUSY_CYCLES", 0)
    bank_conf = c.get("SQ_LDS_BANK_CONFLICT", 0)
    tcc_hit = c.get("TCC_HIT_sum", 0)
    tcc_miss = c.get("TCC_MISS_sum", 0)
    tcp_pend = c.get("TCP_PENDING_STALL_CYCLES_sum", 0)
    tcp_lfifo = c.get("TCP_LFIFO_STALL_CYCLES_sum", 0)
    tcp_tagcf = c.get("TCP_READ_TAGCONFLICT_STALL_CYCLES_sum", 0)
    fetch_kb = c.get("FetchSize", 0)
    valu_busy_pct = c.get("VALUBusy", 0)
    mfma_util_pct = c.get("MfmaUtil", 0)
    mem_unit_stalled = c.get("MemUnitStalled", 0)
    lds_bank_conf_pct = c.get("LDSBankConflict", 0)

    dur_ns = c["__duration_ns_median__"]
    derived = {
        "duration_us": dur_ns / 1000.0,
        "achieved_TFLOPS": flops / dur_ns / 1000.0 if dur_ns else 0,
        "GRBM_GUI_ACTIVE": grbm,
        "SQ_WAVES": waves,
        "SQ_INSTS_VALU": valu,
        "SQ_INSTS_MFMA": mfma,
        "SQ_INSTS_LDS": lds,
        "SQ_INSTS_VMEM_RD": vmem_rd,
        "SQ_INSTS_VMEM_WR": vmem_wr,
        "SQ_INSTS_SALU": salu,
        "SQ_WAIT_INST_LDS_cyc4": wait_lds,
        "SQ_WAIT_INST_ANY_cyc4": wait_any,
        "SQ_BUSY_CYCLES": sq_busy,
        "SQ_VALU_MFMA_BUSY_CYCLES": mfma_busy,
        "SQ_LDS_BANK_CONFLICT_cyc": bank_conf,
        "TCC_HIT_sum": tcc_hit,
        "TCC_MISS_sum": tcc_miss,
        "TCC_total_req": tcc_hit + tcc_miss,
        "TCC_hit_pct": 100.0 * tcc_hit / (tcc_hit + tcc_miss) if (tcc_hit + tcc_miss) else 0,
        "TCP_PENDING_STALL_sum": tcp_pend,
        "TCP_LFIFO_STALL_sum": tcp_lfifo,
        "TCP_READ_TAGCONFLICT_STALL_sum": tcp_tagcf,
        "FetchSize_KB": fetch_kb,
        "VALUBusy_pct": valu_busy_pct,
        "MfmaUtil_pct": mfma_util_pct,
        "MemUnitStalled_pct": mem_unit_stalled,
        "LDSBankConflict_pct": lds_bank_conf_pct,
        # per-wave normalized
        "wait_lds_cyc_per_wave": wait_lds * 4 / waves if waves else 0,
        "wait_any_cyc_per_wave": wait_any * 4 / waves if waves else 0,
        "mfma_busy_per_wave": mfma_busy / waves if waves else 0,
        "sq_busy_per_wave": sq_busy / waves if waves else 0,
        "wait_lds_pct_of_busy": 100.0 * wait_lds * 4 / sq_busy if sq_busy else 0,
        "wait_any_pct_of_busy": 100.0 * wait_any * 4 / sq_busy if sq_busy else 0,
        "mfma_busy_pct_of_busy": 100.0 * mfma_busy / sq_busy if sq_busy else 0,
        "lds_bank_conf_pct_of_busy": 100.0 * bank_conf / sq_busy if sq_busy else 0,
        "lds_inst_per_mfma": lds / mfma if mfma else 0,
        "vmem_inst_per_mfma": (vmem_rd + vmem_wr) / mfma if mfma else 0,
        "valu_inst_per_mfma": valu / mfma if mfma else 0,
        "salu_inst_per_mfma": salu / mfma if mfma else 0,
    }
    return derived

SHAPES = {
    "v2_8B_GateUp_default":  (4096, 14336, 4096),
    "v1_8B_GateUp_forced":   (4096, 14336, 4096),
    "v2_70B_QO_default":     (4096,  8192, 8192),
    "v1_70B_QO_forced":      (4096,  8192, 8192),
}

print("="*120)
print(f"{'Metric':<42}", end="")
for cell in CELLS:
    short = cell.replace("_default", "_def").replace("_forced", "_frc")
    print(f"{short:>20}", end="")
print()
print("="*120)

results = {}
for cell, (dirname, descr) in CELLS.items():
    if not os.path.exists(os.path.join(ROOT, dirname)):
        continue
    raw = load_cell(dirname)
    der = derive(raw, SHAPES[cell])
    results[cell] = (raw, der, descr)

keys_in_order = [
    "duration_us", "achieved_TFLOPS",
    "GRBM_GUI_ACTIVE", "SQ_WAVES", "SQ_BUSY_CYCLES",
    "SQ_INSTS_VALU", "SQ_INSTS_MFMA", "SQ_INSTS_LDS", "SQ_INSTS_VMEM_RD", "SQ_INSTS_VMEM_WR", "SQ_INSTS_SALU",
    "SQ_VALU_MFMA_BUSY_CYCLES", "mfma_busy_per_wave", "mfma_busy_pct_of_busy",
    "SQ_WAIT_INST_LDS_cyc4", "wait_lds_cyc_per_wave", "wait_lds_pct_of_busy",
    "SQ_WAIT_INST_ANY_cyc4", "wait_any_cyc_per_wave", "wait_any_pct_of_busy",
    "SQ_LDS_BANK_CONFLICT_cyc", "lds_bank_conf_pct_of_busy", "LDSBankConflict_pct",
    "TCC_HIT_sum", "TCC_MISS_sum", "TCC_total_req", "TCC_hit_pct",
    "TCP_PENDING_STALL_sum", "TCP_LFIFO_STALL_sum", "TCP_READ_TAGCONFLICT_STALL_sum",
    "FetchSize_KB",
    "VALUBusy_pct", "MfmaUtil_pct", "MemUnitStalled_pct",
    "lds_inst_per_mfma", "vmem_inst_per_mfma", "valu_inst_per_mfma", "salu_inst_per_mfma",
]
for k in keys_in_order:
    print(f"{k:<42}", end="")
    for cell in CELLS:
        if cell not in results: continue
        v = results[cell][1].get(k, 0)
        if isinstance(v, float):
            if v > 1e9:
                print(f"{v:>20.3e}", end="")
            elif v > 100:
                print(f"{v:>20.1f}", end="")
            elif v > 1:
                print(f"{v:>20.3f}", end="")
            else:
                print(f"{v:>20.5f}", end="")
        else:
            print(f"{str(v):>20}", end="")
    print()

print("="*120)
print("Kernel names dispatched (per cell):")
for cell, (raw, der, descr) in results.items():
    print(f"  {cell}: {raw['__kernel_names__']}")

# Dump JSON for further analysis
out = {cell: {"descr": descr, "raw": {k:v for k,v in raw.items() if not k.startswith('__')}, "derived": der}
       for cell, (raw, der, descr) in results.items()}
with open(os.path.join(ROOT, "aggregated.json"), "w") as f:
    json.dump(out, f, indent=2)
print(f"\nWrote {os.path.join(ROOT, 'aggregated.json')}")
