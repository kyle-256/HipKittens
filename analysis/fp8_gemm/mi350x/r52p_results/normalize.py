#!/usr/bin/env python3
"""Normalize PMC counters per-MFMA-instruction to compare cells with different K."""
import json, os

ROOT = os.path.dirname(os.path.abspath(__file__))
data = json.load(open(os.path.join(ROOT, "aggregated.json")))

cells = list(data.keys())
print(f"{'Per-MFMA metric':<42}", end="")
for c in cells:
    short = c.replace("_default", "").replace("_forced", "f")
    print(f"{short:>20}", end="")
print()
print("-" * (42 + 20*len(cells)))

def per_mfma(cell, key):
    raw = data[cell]["raw"]
    mfma = raw["SQ_INSTS_MFMA"]
    return raw.get(key, 0) / mfma if mfma else 0

def per_wave_cycles(cell, key, scale=4):
    raw = data[cell]["raw"]
    waves = raw["SQ_WAVES"]
    return raw.get(key, 0) * scale / waves if waves else 0

def show(label, fn):
    print(f"{label:<42}", end="")
    for c in cells:
        v = fn(c)
        if isinstance(v, float):
            if v > 1000: print(f"{v:>20.1f}", end="")
            elif v > 1: print(f"{v:>20.4f}", end="")
            else: print(f"{v:>20.6f}", end="")
        else: print(f"{v:>20}", end="")
    print()

show("achieved_TFLOPS",            lambda c: data[c]["derived"]["achieved_TFLOPS"])
show("MfmaUtil_pct",               lambda c: data[c]["derived"]["MfmaUtil_pct"])
show("VALUBusy_pct",               lambda c: data[c]["derived"]["VALUBusy_pct"])
show("",                           lambda c: "")
show("MFMA insts (millions)",      lambda c: data[c]["raw"]["SQ_INSTS_MFMA"]/1e6)
show("LDS insts / MFMA",           lambda c: per_mfma(c, "SQ_INSTS_LDS"))
show("VMEM_RD insts / MFMA",       lambda c: per_mfma(c, "SQ_INSTS_VMEM_RD"))
show("VMEM_WR insts / MFMA",       lambda c: per_mfma(c, "SQ_INSTS_VMEM_WR"))
show("VALU insts / MFMA",          lambda c: per_mfma(c, "SQ_INSTS_VALU"))
show("SALU insts / MFMA",          lambda c: per_mfma(c, "SQ_INSTS_SALU"))
show("",                           lambda c: "")
show("WAIT_INST_LDS cyc / MFMA",   lambda c: per_mfma(c, "SQ_WAIT_INST_LDS_cyc4")*4)
show("WAIT_INST_ANY cyc / MFMA",   lambda c: per_mfma(c, "SQ_WAIT_INST_ANY_cyc4")*4)
show("MFMA_BUSY cyc / MFMA",       lambda c: per_mfma(c, "SQ_VALU_MFMA_BUSY_CYCLES"))
show("",                           lambda c: "")
show("TCC requests / MFMA",        lambda c: (data[c]["raw"]["TCC_HIT_sum"] + data[c]["raw"]["TCC_MISS_sum"])/data[c]["raw"]["SQ_INSTS_MFMA"])
show("TCC hit_pct",                lambda c: data[c]["derived"]["TCC_hit_pct"])
show("TCC misses (millions)",      lambda c: data[c]["raw"]["TCC_MISS_sum"]/1e6)
show("FetchSize KB",               lambda c: data[c]["raw"]["FetchSize"])
show("FetchSize KB / MFMA",        lambda c: data[c]["raw"]["FetchSize"]/data[c]["raw"]["SQ_INSTS_MFMA"])
show("TCP_PENDING_STALL / MFMA",   lambda c: per_mfma(c, "TCP_PENDING_STALL_sum"))
show("",                           lambda c: "")
show("Duration (us)",              lambda c: data[c]["derived"]["duration_us"])
show("Active GUI cycles",          lambda c: data[c]["raw"]["GRBM_GUI_ACTIVE"])
show("SQ_BUSY_CYCLES",             lambda c: data[c]["raw"]["SQ_BUSY_CYCLES"])

# Headroom calculation
print()
print("="*100)
print("DELTA (8B GateUp vs 70B QO, V2 default path):")
v2_8B = data["v2_8B_GateUp_default"]
v2_70B = data["v2_70B_QO_default"]
print(f"  TFLOPS gap: {v2_70B['derived']['achieved_TFLOPS'] - v2_8B['derived']['achieved_TFLOPS']:.1f} ({(v2_70B['derived']['achieved_TFLOPS']/v2_8B['derived']['achieved_TFLOPS']-1)*100:.2f}%)")
print(f"  MfmaUtil gap: {v2_70B['derived']['MfmaUtil_pct'] - v2_8B['derived']['MfmaUtil_pct']:.2f}pp")
print(f"  VALUBusy gap: {v2_70B['derived']['VALUBusy_pct'] - v2_8B['derived']['VALUBusy_pct']:.2f}pp")
print(f"  TCC_hit gap: {v2_70B['derived']['TCC_hit_pct'] - v2_8B['derived']['TCC_hit_pct']:.2f}pp")
print(f"  WAIT_LDS cyc/MFMA: 8B={per_mfma('v2_8B_GateUp_default', 'SQ_WAIT_INST_LDS_cyc4')*4:.4f} vs 70B={per_mfma('v2_70B_QO_default', 'SQ_WAIT_INST_LDS_cyc4')*4:.4f}")
print(f"  WAIT_ANY cyc/MFMA: 8B={per_mfma('v2_8B_GateUp_default', 'SQ_WAIT_INST_ANY_cyc4')*4:.4f} vs 70B={per_mfma('v2_70B_QO_default', 'SQ_WAIT_INST_ANY_cyc4')*4:.4f}")

print()
print("="*100)
print("DELTA (V1 forced vs V2 default at 8B GateUp):")
print(f"  TFLOPS gap: V1={data['v1_8B_GateUp_forced']['derived']['achieved_TFLOPS']:.1f}, V2={v2_8B['derived']['achieved_TFLOPS']:.1f}, diff={(v2_8B['derived']['achieved_TFLOPS']/data['v1_8B_GateUp_forced']['derived']['achieved_TFLOPS']-1)*100:+.2f}% (V2 vs V1)")
print(f"  MfmaUtil: V1={data['v1_8B_GateUp_forced']['derived']['MfmaUtil_pct']:.2f}, V2={v2_8B['derived']['MfmaUtil_pct']:.2f}")
print(f"  V1 VMEM_RD/MFMA: {per_mfma('v1_8B_GateUp_forced', 'SQ_INSTS_VMEM_RD'):.4f} vs V2: {per_mfma('v2_8B_GateUp_default', 'SQ_INSTS_VMEM_RD'):.4f}")
print(f"  V1 VALU/MFMA:    {per_mfma('v1_8B_GateUp_forced', 'SQ_INSTS_VALU'):.4f} vs V2: {per_mfma('v2_8B_GateUp_default', 'SQ_INSTS_VALU'):.4f}")
