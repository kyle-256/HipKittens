# scripts/

Tooling for the FP8 blockwise GEMM kernel work on MI300X.

## Layout

| File | Purpose |
|---|---|
| `_shapes_target.py` | 18 production (M,N,K) shapes × 3 sections (fwd/dgrad/wgrad) + listed Triton baseline + HK capability gate |
| `_bench_blockwise_triton_target_shapes.py` | Re-bench Primus-Turbo Triton across all 54 (shape, section) pairs on THIS MI300X. Caches to `.bench_blockwise_triton_target_shapes_cache.json`. |
| `_metric_blockwise_fp8_target_shapes.py` | Score HK across all 54 pairs. `score = round(mean(min(HK/(1.25×Triton), 1.0)) × 1000)`. Last stdout line is the integer score. |
| `_metric_blockwise_fp8_loser_shapes.py` | Same scoring formula on the focused subset where HK is currently below Triton (~8× per-pair sensitivity vs full). Used by the daemon for targeted tuning rounds. |
| `auto_optimize_blockwise_fp8.py` | Per-round optimization daemon: spawns a Claude session, applies its diff, runs the metric, keeps best score. Writes `_auto_optimize_blockwise_fp8_state.json`. |
| `launch_auto_optimize_blockwise_fp8.sh` | nohup launcher for the daemon. |
| `_task_blockwise_fp8.md` | Daemon task spec (loser-focused). |
| `_task_blockwise_fp8_losers_extra.md` | Extra context appended to the task spec. |
| `_goal_blockwise_fp8.md` | Project north-star: target shapes, current score, exhausted knobs, remaining structural attacks. |

## Common commands

```bash
# Score the kernel
METRIC_TRIALS=5 python3 scripts/_metric_blockwise_fp8_target_shapes.py

# Re-bench Triton (refresh cache before scoring)
python3 scripts/_bench_blockwise_triton_target_shapes.py

# Launch daemon for N rounds
bash scripts/launch_auto_optimize_blockwise_fp8.sh
```

Metric contract: stdout's last non-empty line is an integer score. Compile-fail / correctness-fail return large negative penalties so they're auto-rejected by the outer loop.
