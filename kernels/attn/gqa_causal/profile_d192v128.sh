#!/bin/bash
# ===========================================================================
# profile_d192v128.sh -- Precise kernel timing via rocprofv3
#
# Uses rocprofv3 --kernel-trace to capture exact GPU kernel durations,
# then extracts attend_ker timings and computes true TFLOPS.
#
# Usage:
#   ./profile_d192v128.sh [N] [B] [H] [H_KV]
#   ./profile_d192v128.sh 4096
#   ./profile_d192v128.sh 8192 16 64 8
#
# Requirements:
#   - ROCm with rocprofv3 installed (/opt/rocm/bin/rocprofv3)
#   - The kernel must be compiled with matching ATTN_N first, or this
#     script will recompile it automatically.
# ===========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
ROCPROFV3="${ROCM_PATH}/bin/rocprofv3"

# Default parameters
N="${1:-4096}"
B="${2:-16}"
H="${3:-64}"
H_KV="${4:-8}"
D_QK=192
D_V=128
CAUSAL=1

echo "============================================================"
echo "  rocprofv3 Kernel Profile: D_QK=${D_QK} D_V=${D_V}"
echo "  N=${N}  B=${B}  H=${H}  H_KV=${H_KV}  causal=${CAUSAL}"
echo "============================================================"

# -------------------------------------------------------------------
# Step 0: Check rocprofv3
# -------------------------------------------------------------------
if [ ! -x "${ROCPROFV3}" ]; then
    echo "ERROR: rocprofv3 not found at ${ROCPROFV3}"
    echo "       Set ROCM_PATH or install ROCm profiling tools."
    exit 1
fi

# -------------------------------------------------------------------
# Step 1: Recompile kernel with correct ATTN_N
# -------------------------------------------------------------------
echo ""
echo ">>> Step 1: Recompiling kernel with ATTN_N=${N} ..."
make -C "${SCRIPT_DIR}" clean >/dev/null 2>&1 || true
make -j -C "${SCRIPT_DIR}" \
    ATTN_N="${N}" \
    ATTN_B="${B}" \
    ATTN_H="${H}" \
    ATTN_H_KV="${H_KV}" \
    ATTN_D_QK="${D_QK}" \
    ATTN_D_V="${D_V}" \
    2>&1 | tail -5
echo "    Compilation done."

# -------------------------------------------------------------------
# Step 2: Create a minimal Python script for profiling
# -------------------------------------------------------------------
PROFILE_PY=$(mktemp /tmp/profile_hk_XXXXXX.py)
cat > "${PROFILE_PY}" << 'PYEOF'
import torch
import sys
import os

sys.path.insert(0, os.environ["PROFILE_DIR"])
import tk_kernel

B     = int(os.environ["P_B"])
N     = int(os.environ["P_N"])
H     = int(os.environ["P_H"])
H_KV  = int(os.environ["P_H_KV"])
D_QK  = int(os.environ.get("P_D_QK", "192"))
D_V   = int(os.environ.get("P_D_V", "128"))

dtype = torch.bfloat16

# Warmup (fewer iterations for profiling)
for _ in range(10):
    out = torch.zeros(B, N, H, D_V, dtype=dtype, device='cuda')
    lse = torch.zeros(B, H, 1, N, dtype=torch.float32, device='cuda')
    q = torch.randn(B, N, H, D_QK, dtype=dtype, device='cuda')
    k = torch.randn(B, N, H_KV, D_QK, dtype=dtype, device='cuda')
    v = torch.randn(B, N, H_KV, D_V, dtype=dtype, device='cuda')
    tk_kernel.dispatch_micro(q, k, v, out, lse)
torch.cuda.synchronize()

# Profiled iterations (just a few -- rocprofv3 captures each kernel launch)
for _ in range(5):
    out = torch.zeros(B, N, H, D_V, dtype=dtype, device='cuda')
    lse = torch.zeros(B, H, 1, N, dtype=torch.float32, device='cuda')
    q = torch.randn(B, N, H, D_QK, dtype=dtype, device='cuda')
    k = torch.randn(B, N, H_KV, D_QK, dtype=dtype, device='cuda')
    v = torch.randn(B, N, H_KV, D_V, dtype=dtype, device='cuda')
    tk_kernel.dispatch_micro(q, k, v, out, lse)
torch.cuda.synchronize()

print("Profiled iterations complete.")
PYEOF

# -------------------------------------------------------------------
# Step 3: Run with rocprofv3 --kernel-trace
# -------------------------------------------------------------------
PROFILE_OUT_DIR=$(mktemp -d /tmp/rocprof_hk_XXXXXX)

echo ""
echo ">>> Step 2: Running with rocprofv3 --kernel-trace ..."
echo "    Output dir: ${PROFILE_OUT_DIR}"

export PROFILE_DIR="${SCRIPT_DIR}"
export P_B="${B}"
export P_N="${N}"
export P_H="${H}"
export P_H_KV="${H_KV}"
export P_D_QK="${D_QK}"
export P_D_V="${D_V}"

${ROCPROFV3} --kernel-trace \
    -o "${PROFILE_OUT_DIR}/results" \
    -- python3 "${PROFILE_PY}" 2>&1 | tail -3

echo "    Profiling done."

# -------------------------------------------------------------------
# Step 4: Parse results from the SQLite database
# -------------------------------------------------------------------
echo ""
echo ">>> Step 3: Parsing kernel trace results ..."

# rocprofv3 outputs results to an SQLite database or CSV.
# Try SQLite first (newer rocprofv3), then fall back to CSV.

PARSE_PY=$(mktemp /tmp/parse_profile_XXXXXX.py)
cat > "${PARSE_PY}" << 'PYEOF2'
import os
import sys
import glob
import json

profile_dir = sys.argv[1]
N    = int(sys.argv[2])
B    = int(sys.argv[3])
H    = int(sys.argv[4])
D_QK = int(sys.argv[5])
D_V  = int(sys.argv[6])
causal = int(sys.argv[7])

# Compute expected FLOPs
total_flops = 2 * B * N * N * H * (D_QK + D_V)
if causal:
    total_flops //= 2

durations_ns = []  # in nanoseconds

# ---- Try SQLite database ----
db_files = glob.glob(os.path.join(profile_dir, "**/*.db"), recursive=True)
if db_files:
    import sqlite3
    for db_file in db_files:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        # List tables to find kernel trace
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        # Common table names in rocprofv3
        kernel_table = None
        for t in tables:
            if 'kernel' in t.lower():
                kernel_table = t
                break

        if kernel_table:
            cursor.execute(f"SELECT * FROM {kernel_table} LIMIT 1")
            columns = [desc[0] for desc in cursor.description]

            # Find the kernel name column and duration column
            name_col = None
            dur_col = None
            start_col = None
            end_col = None
            for c in columns:
                cl = c.lower()
                if 'kernel' in cl and 'name' in cl:
                    name_col = c
                elif cl == 'kernelname' or cl == 'name':
                    name_col = c
                if 'duration' in cl:
                    dur_col = c
                if 'start' in cl and 'time' not in cl.replace('start', ''):
                    start_col = c
                elif cl == 'start':
                    start_col = c
                if 'end' in cl and 'time' not in cl.replace('end', ''):
                    end_col = c
                elif cl == 'end':
                    end_col = c

            if name_col is None:
                name_col = columns[0]

            # Query for attend_ker entries
            if dur_col:
                cursor.execute(
                    f"SELECT {name_col}, {dur_col} FROM {kernel_table} "
                    f"WHERE {name_col} LIKE '%attend_ker%'"
                )
                for row in cursor.fetchall():
                    durations_ns.append(float(row[1]))
            elif start_col and end_col:
                cursor.execute(
                    f"SELECT {name_col}, {start_col}, {end_col} FROM {kernel_table} "
                    f"WHERE {name_col} LIKE '%attend_ker%'"
                )
                for row in cursor.fetchall():
                    durations_ns.append(float(row[2]) - float(row[1]))
            else:
                # Dump all columns for debugging
                print(f"Available columns in {kernel_table}: {columns}")
                cursor.execute(f"SELECT * FROM {kernel_table} LIMIT 5")
                for row in cursor.fetchall():
                    print(row)

        conn.close()

# ---- Try CSV fallback ----
if not durations_ns:
    csv_files = glob.glob(os.path.join(profile_dir, "**/*.csv"), recursive=True)
    for csv_file in csv_files:
        import csv
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Check if this row is an attend_ker invocation
                name_val = ""
                for k in row:
                    if 'kernel' in k.lower() or 'name' in k.lower():
                        name_val = row[k]
                        break
                if 'attend_ker' not in name_val:
                    continue

                # Try to find duration
                dur = None
                for k in row:
                    if 'duration' in k.lower():
                        dur = float(row[k])
                        break
                if dur is None:
                    # Try start/end
                    start_val = end_val = None
                    for k in row:
                        if 'start' in k.lower():
                            start_val = float(row[k])
                        elif 'end' in k.lower():
                            end_val = float(row[k])
                    if start_val is not None and end_val is not None:
                        dur = end_val - start_val
                if dur is not None:
                    durations_ns.append(dur)

# ---- Try JSON fallback ----
if not durations_ns:
    json_files = glob.glob(os.path.join(profile_dir, "**/*.json"), recursive=True)
    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)
        if isinstance(data, list):
            for entry in data:
                name = entry.get("KernelName", entry.get("Name", ""))
                if "attend_ker" in name:
                    dur = entry.get("DurationNs", entry.get("Duration", 0))
                    if dur:
                        durations_ns.append(float(dur))

# ---- Report ----
if not durations_ns:
    print("WARNING: No attend_ker entries found in profiling output.")
    print(f"  Profile dir: {profile_dir}")
    print("  Listing all files:")
    for root, dirs, files in os.walk(profile_dir):
        for fn in files:
            fp = os.path.join(root, fn)
            print(f"    {fp}  ({os.path.getsize(fp)} bytes)")
    sys.exit(1)

# Skip first few (warmup) and compute stats on the rest
# We ran 10 warmup + 5 profiled iterations, so take last 5
if len(durations_ns) > 5:
    durations_ns = durations_ns[-5:]

durations_us = [d / 1e3 for d in durations_ns]
durations_ms = [d / 1e6 for d in durations_ns]

avg_ns = sum(durations_ns) / len(durations_ns)
avg_us = avg_ns / 1e3
avg_ms = avg_ns / 1e6
min_ns = min(durations_ns)
max_ns = max(durations_ns)

tflops_avg = (total_flops / 1e12) / (avg_ns / 1e9)
tflops_best = (total_flops / 1e12) / (min_ns / 1e9)

print()
print("=" * 72)
print(f"  attend_ker Kernel Profile  (N={N}, B={B}, H={H})")
print(f"  D_QK={D_QK}, D_V={D_V}, causal={causal}")
print("=" * 72)
print(f"  Samples:     {len(durations_ns)}")
print(f"  Avg:         {avg_us:.1f} us  ({avg_ms:.4f} ms)")
print(f"  Min:         {min_ns/1e3:.1f} us")
print(f"  Max:         {max_ns/1e3:.1f} us")
print(f"  FLOPs:       {total_flops:.3e}")
print(f"  Avg TFLOPS:  {tflops_avg:.2f}")
print(f"  Best TFLOPS: {tflops_best:.2f}")
print("=" * 72)
print()
print("Per-invocation timings (us):")
for i, d in enumerate(durations_us):
    tf = (total_flops / 1e12) / (d / 1e6)
    print(f"  [{i}]  {d:.1f} us  ->  {tf:.2f} TFLOPS")
PYEOF2

python3 "${PARSE_PY}" "${PROFILE_OUT_DIR}" "${N}" "${B}" "${H}" "${D_QK}" "${D_V}" "${CAUSAL}"

# -------------------------------------------------------------------
# Cleanup
# -------------------------------------------------------------------
rm -f "${PROFILE_PY}" "${PARSE_PY}"
echo ""
echo "Raw profile data preserved at: ${PROFILE_OUT_DIR}"
echo "Done."
