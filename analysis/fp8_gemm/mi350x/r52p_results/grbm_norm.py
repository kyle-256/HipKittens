#!/usr/bin/env python3
"""Normalize PMC counters per-GRBM-cycle (active GPU cycle) for true stall attribution."""
import json, os
ROOT = os.path.dirname(os.path.abspath(__file__))
data = json.load(open(os.path.join(ROOT, "aggregated.json")))

# MI355X gfx950: 256 CUs total, 8 SE x 32 CUs/SE?
# Actually MI355X = MI355X has 304 CUs in 8 XCD (per chiplet). Each XCD = 38 CU.
# But these counters are per-SE aggregated.
# The most useful normalization is per-GRBM cycle, which counts the duration of the kernel
# in GPU shader-active cycles. SQ counters are per-SIMD (4 SIMDs / CU).
# Normalize by GRBM_GUI_ACTIVE * SIMD_count.

# For the PCT_OF_BUSY metrics shown in raw output, the divisor is probably wrong.
# Let me just give per-GRBM-cycle ratios (which is what MfmaUtil_pct etc use).

def stat(cell, key):
    return data[cell]["raw"].get(key, 0)

cells = list(data.keys())
print(f"\n{'METRIC (per GRBM_GUI_ACTIVE active cycle)':<55}", end="")
for c in cells:
    s = c.replace("_default","").replace("_forced","")
    print(f"{s:>20}", end="")
print()
print("-"*135)

for label, key in [
    ("SQ_BUSY_CYCLES / GRBM_GUI_ACTIVE",     "SQ_BUSY_CYCLES"),
    ("SQ_VALU_MFMA_BUSY_CYCLES / GRBM_ACTIVE","SQ_VALU_MFMA_BUSY_CYCLES"),
    ("SQ_WAIT_INST_LDS*4 / GRBM_ACTIVE",     "SQ_WAIT_INST_LDS_cyc4"),
    ("SQ_WAIT_INST_ANY*4 / GRBM_ACTIVE",     "SQ_WAIT_INST_ANY_cyc4"),
    ("TCP_PENDING_STALL / GRBM_ACTIVE",      "TCP_PENDING_STALL_sum"),
    ("TCC_MISS_sum / GRBM_ACTIVE",           "TCC_MISS_sum"),
    ("SQ_INSTS_VMEM_RD / GRBM_ACTIVE",       "SQ_INSTS_VMEM_RD"),
    ("SQ_INSTS_LDS / GRBM_ACTIVE",           "SQ_INSTS_LDS"),
    ("SQ_INSTS_VALU / GRBM_ACTIVE",          "SQ_INSTS_VALU"),
    ("SQ_INSTS_SALU / GRBM_ACTIVE",          "SQ_INSTS_SALU"),
]:
    print(f"{label:<55}", end="")
    for c in cells:
        scale = 4 if "_cyc4" in key else 1
        v = stat(c, key) * scale / stat(c, "GRBM_GUI_ACTIVE")
        print(f"{v:>20.4f}", end="")
    print()

# Direct: from the rocprof derived metrics
print(f"\n{'DIRECT DERIVED METRICS':<55}", end="")
for c in cells:
    s = c.replace("_default","").replace("_forced","")
    print(f"{s:>20}", end="")
print()
print("-"*135)
for label in ["MfmaUtil_pct", "VALUBusy_pct", "MemUnitStalled_pct", "LDSBankConflict_pct"]:
    print(f"{label:<55}", end="")
    for c in cells:
        v = data[c]["derived"].get(label, 0)
        print(f"{v:>20.4f}", end="")
    print()

# Compute MFMA pipe theoretical occupancy
# 8 SE * 32 SIMD * MFMA pipe = some number. mfma_busy_cycles is summed across all SIMDs.
# rocm-smi tells us SIMD count. Let me check:
print(f"\n{'INSTRUCTION COMPOSITION (% of all dynamic insts)':<55}", end="")
for c in cells:
    s = c.replace("_default","").replace("_forced","")
    print(f"{s:>20}", end="")
print()
print("-"*135)
for label, key in [("MFMA", "SQ_INSTS_MFMA"), ("VALU (non-MFMA)", "SQ_INSTS_VALU"),
                   ("LDS", "SQ_INSTS_LDS"), ("VMEM_RD", "SQ_INSTS_VMEM_RD"),
                   ("VMEM_WR", "SQ_INSTS_VMEM_WR"), ("SALU", "SQ_INSTS_SALU")]:
    print(f"{label:<55}", end="")
    for c in cells:
        # VALU includes MFMA. Subtract for non-MFMA VALU.
        if key == "SQ_INSTS_VALU":
            v = stat(c, "SQ_INSTS_VALU") - stat(c, "SQ_INSTS_MFMA")
        else:
            v = stat(c, key)
        total = stat(c, "SQ_INSTS_VALU") + stat(c, "SQ_INSTS_LDS") + stat(c, "SQ_INSTS_VMEM_RD") + stat(c, "SQ_INSTS_VMEM_WR") + stat(c, "SQ_INSTS_SALU")
        print(f"{100*v/total:>19.2f}%", end="")
    print()

print(f"\n{'TFLOPS achieved':<55}", end="")
for c in cells:
    print(f"{data[c]['derived']['achieved_TFLOPS']:>20.1f}", end="")
print()
print(f"{'Duration (us)':<55}", end="")
for c in cells:
    print(f"{data[c]['derived']['duration_us']:>20.1f}", end="")
print()
