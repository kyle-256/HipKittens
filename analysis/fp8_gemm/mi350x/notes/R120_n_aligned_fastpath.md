# R120 — N-aligned fast path

Dispatcher splits N_MASKED_STORE=false vs true based on `n_aligned = (g.bpc * BLOCK_SIZE == g.n)`.
N_MASKED_STORE=false → no per-store mask check, faster.
N_MASKED_STORE=true → buffer_store with mask, ~3-5% slower.

All 8 prod shapes: N divisible by 256 → n_aligned=true → N_MASKED_STORE=false path.

Edge case: dsv3 N=7168/256=28, qwen N=4096/256=16, gpt_oss N=5760/256=22.5 (not aligned!). Wait gpt_oss up N=5760, 5760/256=22.5 → NOT aligned → N_MASKED_STORE=true.

So gpt_oss takes masked store path. Per `[[bench_24]]`: gpt_oss_up still good perf. Mask overhead absorbed by mfma latency.

N_MASKED_STORE not a lever for current shapes.
