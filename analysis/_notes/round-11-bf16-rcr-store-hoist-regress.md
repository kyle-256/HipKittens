# Round-11 — BF16 RCR fuse store: per-block hoist regresses

## Hypothesis (rejected)

Mirror FP8 **round-59** wedge (+39 % wall on
`gpt_oss_20B-GateUP-B32-M4096` K=2816) into the BF16 RCR fuse epilog:
hoist the per-block `(col+1)*BLOCK_SIZE <= g.n` interior-tile branch out of
`store_c_tile_n_masked` and into the kernel epilog so the compiler can
specialise the interior path on bare `store(...)` and drop the helper's
lane-level OOB-column scan body from VGPR.

```cpp
if ((g.n % BLOCK_SIZE) == 0) {            // launch-uniform aligned (DSV3)
    store(g.c, C_accum[0][0], ...);  // ×4
} else if ((col + 1) * BLOCK_SIZE <= g.n) {
    // round-11 probe: interior tile of misaligned N
    store(g.c, C_accum[0][0], ...);  // ×4 — bit-identical to helper fast-path
} else {
    store_c_tile_n_masked(g.c, C_accum[0][0], ..., g.n);  // ×4
}
```

## Result

`scripts/_metric_grouped_only.py` (3 runs each, HIP_VISIBLE_DEVICES=2):

| state            | score | grpBF16 geomean | grpFP8 geomean |
|------------------|------:|----------------:|---------------:|
| round-10 baseline| 832-835 | 1.0924 | 0.9183 |
| **round-11 hoist** | **825-829** | **1.0705** | 0.9161 |

**−7…−10 score** vs baseline. The regression is concentrated in the
DeepSeek-V3 path (which still takes the unchanged `(g.n % BLOCK_SIZE) == 0`
first branch):

| shape (N, K)              | hk_tflops baseline | hk_tflops hoist | Δ |
|---------------------------|-------------------:|----------------:|------:|
| DSV3-GateUP-B16-M2048 (2048,7168) | 1387.4 | 1333.3 | −3.9 % |
| DSV3-Down-B16-M2048   (7168,2048) | 1228.3 | 1205.2 | −1.9 % |
| DSV3-GateUP-B16-M4096 (2048,7168) | 1427.5 | 1390.7 | −2.6 % |
| DSV3-GateUP-B32-M2048 (2048,7168) | 1407.7 | 1360.0 | −3.4 % |
| DSV3-GateUP-B32-M4096 (2048,7168) | 1432.3 | 1382.4 | −3.5 % |

The misaligned `gpt_oss` shapes (which are the targets of the wedge) saw no
measurable change: GateUP-B4-M2048 0.997 → 0.989, Down-B4-M2048
1.087 → 1.082, GateUP-B4-M4096 1.053 → 1.036.

## Why the FP8 round-59 win does **not** transfer

FP8 round-59 promoted the masked-store branch from a runtime check
inside the helper to a **template parameter** `N_MASKED_STORE` on the
kernel itself, so the compiler emitted two specialised SASS bodies:

```cpp
template <Layout L, int KI_HINT, bool N_MASKED_STORE>
__global__ void grouped_kernel_fp8(...) {
    if constexpr (N_MASKED_STORE) {
        if ((bc + 1) * BLOCK_SIZE <= g.n) store(...);   // interior
        else                              store_n_masked(...);  // tail
    } else {
        store(...);                                     // aligned only
    }
}
```

Two distinct kernels, one per host-side N-alignment dispatch. The
N_MASKED_STORE=false body has no helper symbol in scope at all, so VGPR
falls and the bare-store epilogue tightens.

BF16 already has the **launch-uniform** outer branch
`(g.n % BLOCK_SIZE) == 0` in the kernel epilog, but it is a runtime
branch on the wave-uniform `g.n` (vs FP8's `if constexpr`), and that
already keeps the helper body out of the aligned path's hot loop —
the helper is only inlined into the misaligned branch.

What the round-11 probe added was a **third** epilog branch (per-block
`col+1 vs g.n/BLOCK_SIZE`) layered on top of the existing two. The
DSV3 path still went through the original `if ((g.n % BLOCK_SIZE) == 0)`
branch unchanged, but the compiler perturbed store/branch reordering
in the surrounding epilog (the two later branches grew the basic
block count from 2 to 3 + grew the helper inline body once per fuse
template) and the dominant DSV3 path slowed −2…−4 %.

The misaligned `gpt_oss` path — where the wedge was supposed to
materialise — saw the helper body inlined into the bare-store branch
just like the masked branch, with no observable improvement: the
helper's `if (n1 <= n_limit) store(...)` fast-path forward was
already getting compiled to bare loads/stores in the interior case,
and the extra epilog branch just added BB overhead.

## Conclusion

Don't repeat this hoist. To replicate the FP8 round-59 win in BF16
the wedge would need to be a **template parameter** on the entire
`grouped_kernel<L, KI_HINT, ...>` (mirroring the FP8 N_MASKED_STORE
template arg), launched from a host-side N-alignment dispatch — not a
runtime branch reorder. That's a much larger surgery (every grouped
template instantiation doubles), and given the gpt_oss BF16 cases are
already 1.01–1.09× Triton (8 cases, geomean 1.04) the wedge headroom
is tiny.

`store_c_tile_n_masked` reverted; round-11 ships only this note (no
.so / no .cpp change).

## Round-11 score posture

`grouped_only` baseline still 832–835. The **forward** K-tail fuse
mainline (RCR / RRR via H4 / FP8 RCR / FP8 RRR via H4) is saturated
across both dtypes. Remaining gaps:

* FP8 gpt_oss 8 cases @ 0.81–0.88 (K=2880 K-tail dominant on 2-stage path)
  — wedge would need a fundamental fuse-epilog rewrite (e.g. moving
  to mfma_32x32x64_fp8 native K=64 path, doubling output tile size to
  rt_32x32_s — ~1 round of register layout work + numerical re-derivation).
* BF16 gpt_oss 8 cases @ 1.01–1.09 (already close to BF16 DSV3 1.10–1.17;
  marginal headroom).
* FP8 DSV3 8 cases @ 0.93–1.05 (rule tune forbidden by run policy).

Backward (dB var-K, dA via H4) is correctness-only, doesn't move metric.
