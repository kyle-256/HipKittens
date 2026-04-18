# P23 Step 7 PMC validation harness

Built by **P23 Dev F** (parallel to Devs B/C/E doing Steps 1-4 of the
P23 plan in `memory/project_bf16_rcr_padded_b128_design.md`).

Goal: validate that Step 4 (`shared_to_register::load(col_l)` switched
from `ds_read_b64_tr_b16` to `ds_read_b128` over `st_64x32_padded_b128`)
does not regress LDS bank conflicts on RCR.

## How to invoke

```bash
cd analysis/bf16_gemm/mi350x

# 1. BEFORE (pre-Step 4 baseline) — already captured 2026-04-18:
#    -> pmc_step7/pmc_baseline_st_16x32_s.json
bash pmc_validate_st_64x32_padded.sh baseline

# 2. AFTER (post-Step 4): rebuild .so, then run:
make
bash pmc_validate_st_64x32_padded.sh padded_b128 \
  pmc_step7/pmc_baseline_st_16x32_s.json
```

Exit code 0 = PASS (RCR ratio ≤ 0.05 AND absolute ≤ 1M);
exit 1 = FAIL.

## CRITICAL framing correction (Dev F finding)

The P23 design memo asserts:
> "TK BANK_CONFLICT/MFMA on RCR worst shape = 67.6M / 90.2M
>  (conflict-bound)"

**This is incorrect.** Re-reading P21 Dev C's source data:
- `67.6M` is `SQ_INSTS_LDS` for **CRR** (not BANK_CONFLICT, not RCR).
- TK RCR `(8192, 22016, 4096)` actual `SQ_LDS_BANK_CONFLICT = 0`.
- TK RCR `(4096, 28672, 4096)` actual `SQ_LDS_BANK_CONFLICT = 0`
  (this harness's measurement, 2026-04-18 GPU 6).

The 135.3M figure cited for "CRR worst-RCR-shape" in Dev C is correct
and reproduced by this harness (135,266,304 on `(8192,22016,4096) crr`).

## Implication for Step 4 / Step 7

Since RCR baseline BANK_CONFLICT is already 0, Step 7's "expect drop
from ~67.6M to <1M" is **vacuous on the bank-conflict axis** — there
are no RCR bank conflicts to remove.

What Step 4 actually changes (per Dev C report):
- **Reduces LDS reads per MFMA** from 0.375 → ~0.25 (matches BL).
  This is the lever that closes the 3.7pp gap on RCR worst.
- The `st_64x32_padded_b128` layout's main effect on RCR is fewer/wider
  `ds_read_b128` (vs the current `ds_read_b64_tr_b16`-using path), not
  bank-conflict elimination.

The harness still serves a useful **regression** purpose:
- Catch any new bank conflicts introduced by Step 4.
- Verify CRR (which Step 4 should not touch) stays at 135M.
- Verify MFMA count is unchanged (correctness check).

## Counters captured (gfx950 confirmed)

| Counter | Purpose |
|---|---|
| `SQ_INSTS_VALU_MFMA_BF16` | BF16 MFMA op count (workload size invariant) |
| `GRBM_GUI_ACTIVE` | Active cycles (perf proxy) |
| `SQ_INSTS_LDS` | LDS instruction issues |
| `SQ_LDS_BANK_CONFLICT` | THE key metric — bank conflict events |
| `SQ_WAIT_INST_LDS` | LDS-wait stall cycles |

## Files

- `pmc_validate_st_64x32_padded.sh` — top-level driver (in parent dir)
- `pmc_step7/single_launch.py` — single-launch profile script
- `pmc_step7/pmc.txt` — counter list for rocprofv3
- `pmc_step7/parse.py` — CSV parser + verdict emitter
- `pmc_step7/pmc_baseline_st_16x32_s.json` — captured baseline
  (current `st_16x32_s` etc.) for AFTER comparison
- `pmc_step7/runs_<tag>_<timestamp>/` — per-run rocprofv3 CSVs

## Reused from prior PMC work

- `/tmp/p21_dev_c/run_pmc_all.sh` and `/tmp/p21_dev_c/single_launch.py`
- `/tmp/p21_dev_e/single_launch.py` (P21 Dev E padded-LDS sweep)
- `/tmp/dev_f_run_all.sh` and `/tmp/dev_f_parse.py` (P19 FP8 PMC harness)
