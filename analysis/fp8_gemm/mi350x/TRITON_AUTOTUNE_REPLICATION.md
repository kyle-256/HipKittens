# Triton Autotune Config Mirror (R92)

## Triton Reference (aiter `grouped_matmul_kernel_fp8`)

Triton autotune for gfx950 selects per-shape configs from:
- BLK_M ∈ {128, 256}, BLK_N ∈ {128, 256}, BK ∈ {64, 128}
- num_stages ∈ {2, 3}
- group_m ∈ {1, 2, 4, 8}
- chunk_size ∈ {32, 64, 128}

Per `[[fp8-rrr-attempt-h11_h12]]`: Triton picks chunk_size=32 default for fp8 grouped on gfx950.

## HK v2 Coverage Match

| Triton config | HK v2 coverage |
|---------------|----------------|
| BLK 256×256 BK=128 | YES (current default) |
| BLK 128×128 BK=128 | NO (would need new template) |
| num_stages=2 | YES (As/Bs[2][2] double-buffer) |
| group_m=4 | YES (caller-passed) |
| chunk_size=32 | YES (post-R43) |
| chunk_size=64 | YES (envoverride) |

Missing: smaller BLK. Adding BLK 128×128 single-acc covers Triton's short-K choice.

## Autotune Strategy

Per shape, run all viable configs once, pick best by TFLOPS. Cache cfg→shape mapping (NOT result cache, just config).

Wait — `[[no-cache]]` forbids caches. So autotune-per-call cost = N config × bench overhead. Not viable per call.

Alternative: hand-pick per-shape via offline benchmark, hard-code into dispatcher. PT's `PRIMUS_TURBO_AUTO_TUNE=1` env may handle this once.

## Implementation Path

- Session 1: add BLK 128×128 single-acc kernel (~500 LOC)
- Session 2: dispatcher heuristic shape→config (~50 LOC)
- Session 3: bench all 24 production shapes with both configs, pick winners
- Session 4: hard-code winner table into dispatcher (offline tuned)

Estimated +5-10pp on short-K shapes (qwen_down/qwen_up bench projection).

## Risk

Hard-coded config table won't generalize to new shapes user adds. Need runtime fallback (current BLK 256×256 default if shape not in table).
