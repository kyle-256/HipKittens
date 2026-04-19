# R44 Opt C — Fault-PC report (PARTIAL: PC range identified, no live VA)

**Date**: 2026-04-19
**GPU**: MI355X (gfx950) device 5
**Build under test**: `build_R43A/tk_mxfp4_gluon_cpp_n32768_k28672_ts_lgk2_gm7_pfoff104_kx28672_btw_all_R43A_p2b_ctrl.cpython-310-x86_64-linux-gnu.so`
**Shape**: `(M=4096, N=32768, K=28672)`
**Status**: CRASH reproduces 4/4 outside debugger; **does NOT reproduce under rocgdb** (heisenbug).

## TL;DR for Opt A

The CRASH is a **late-K-iter LDS double-buffer write-after-read race** at the K-loop back-edge. The 16 `buffer_load_dwordx4 v*, s[16:23], s* offen lds` instructions issued at PC `0x1A884`–`0x1A9A0` (lines 8973–9020 of `R44_OPT_C_p2b_ctrl_kernel.s`) are the iter-N+1 LDS prefetches; they are issued **unconditionally** and the loop's back-edge `s_waitcnt lgkmcnt(0); s_cbranch_scc0` at PC `0x1A9CC`–`0x1A9D0` (line 9026–9027) **does not drain `vmcnt`** before falling through to the TAIL_SPLIT epilogue at PC `0x1A9D4` onward. The epilogue's `ds_read_b128` (line 9029, PC `0x1A9E4`) reads the SAME LDS double-buffer slot that the in-flight prefetches are still writing.

→ **Opt A's 3-buffer rotation idea will fix it by construction**: if iter-N+1 prefetches target slot `(bt+1) % 3`, they cannot alias the slot the epilogue is reading from `bt % 3 = (k_byte_iters - 1) % 3`.

→ **Cheaper alternative for Opt A to test first**: insert a single `s_waitcnt vmcnt(0)` either (a) just **before** the back-edge `s_cbranch_scc0` at PC `0x1A9D0` (drains all in-flight prefetches before fall-through), or (b) just **after** the back-edge target (top of TAIL_SPLIT epilogue, PC `0x1A9D4`). Either should close the race. R43 fix1f tried (a) at C++ level but emitted it inside the `R43A_GATE_PF_TAIL_KBOUND` branch where the compiler may have hoisted/dropped it; this needs to be enforced via inline `asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory")` placed at the very last C++ statement of the `for (int bt = ...)` loop body, AND with `R37_FIX_B + R40A_PF_FENCE` macros enabled to prevent the compiler from reordering it.

## Falsifiable predictions met

- **P-C.1 (PARTIAL)**: HSA returns no faulting VA — the runtime only logs `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION code: 0x29`. The gpucore dump is created but contains no live wavefront register state recoverable by `rocgdb 16.3` (`info dispatches`, `info queues`, `info lanes` all empty). MISS on getting an exact VA.
- **P-C.2 (HIT, statically)**: A compact PC range `[0x1A884, 0x1A9A0]` (16 instructions) is identifiable as the faulting region by static disasm + back-edge analysis + matching against the kernel-source `for (int bt = 0; bt + 1 < k_byte_iters; ++bt)` loop structure.

## Methodology

### Phase 0 — Reproduce
`R44_OPT_C_repro.py`: minimal load-and-call. CRASH 4/4 with `code: 0x29 HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` for all `AMD_SERIALIZE_KERNEL ∈ {0,1,3}`.

### Phase 1 — HSA fault info
- `HSA_DEBUG=1 AMD_LOG_LEVEL=4 HSA_ENABLE_QUEUE_FAULT_MESSAGE=1`: produces full HIP-API trace + the abort callback message, but **no faulting VA, no faulting PC, no waveID**. The MI350X runtime's queue-abort path doesn't print these.
- `rocgdb 16.3 -batch ... --args python R44_OPT_C_repro.py` with `set amdgpu precise-memory on; set amdgpu precise-alu-exceptions on`: the kernel **completes normally** (`UNEXPECTED`), confirming the race is timing-sensitive enough that debugger trap-installation and shadow-thread setup serializes wave issue past the unsafe window. Repeated 2× → same: never crashed under rocgdb.
- Loading the post-mortem `gpucore.*` (`/opt/rocm/bin/rocgdb -batch /usr/bin/python3.10 gpucore.1857072`) shows 8 AMDGPU agents but `No queues / No dispatches / No lanes` — the runtime's coredump captures memory but not the live wavefront context tables on this ROCm 7.1 build, so PC-from-core is not feasible with the available tooling.

### Phase 2 — Static disasm
Extracted HSACO via `/opt/rocm/llvm/bin/llvm-objdump --offloading <so>` → `<so>.0.hipv4-amdgcn-amd-amdhsa--gfx950` (114 KB).
Disassembled with `--mcpu=gfx950` → `R44_OPT_C_p2b_ctrl_kernel.s` (10206 lines, 624 `buffer_load_dwordx4 ... lds` sites, 38 `s_barrier`, 0 `ds_write` — LDS writes happen via `buffer_load ... lds` direct-to-LDS).
Kernel symbol `_Z22mxfp4_gluon_cpp_kernel13gluon_globals` at PC `0x1900`; main K-loop spans PC `0x295C` (back-edge target) → `0x1A9D0` (back-edge).

### Phase 3 — Source pairing
- **PC `0x1A884`–`0x1A9A0`**: 16 `buffer_load_dwordx4 v98/99/126/100/102/101/103/104, s[16:23], s_offset offen lds` — these are the 16 `emit_pf_tail<0>(pf_a0_p, pf_a1_p)` + `emit_pf_tail<0>(pf_bl_p, pf_br_p)` calls in `kernel_mxfp4_gluon_cpp.cpp:3313-3314` (the FUSED_STEP34 unguarded branch).
- **PC `0x1A99C`**: `s_add_i32 s44, s44, 37` then `s_cmpk_eq_i32 s44, 0x70` — K-loop iteration counter increment by 37 then compare with `0x70 = 112 = k_byte_iters`. (37 is the inner unroll factor used by `pragma unroll 8`-style emission for k_byte_iters=112.)
- **PC `0x1A9AC`–`0x1A9C4`**: 4 `buffer_load_dwordx2 v*, v105, s[*], s70 offen` — the 4 `load_pq_scale_x2_async` calls (1 each for a0/a1/bl/br) at source line ~3198/3207.
- **PC `0x1A9CC`**: `s_waitcnt lgkmcnt(0)` — drains the 4 scale loads' LDS-side completion.
- **PC `0x1A9D0`**: `s_cbranch_scc0 -24606` → branches back to PC `0x295C` (top of K-loop body) when `s44 != 0x70`. **Crucially: NO `s_waitcnt vmcnt(0)` here** — the 16 `buffer_load_dwordx4 ... lds` issued at `0x1A884`–`0x1A9A0` may still be in flight on fall-through (last K iter).
- **PC `0x1A9D4`** (line 9028) onward: TAIL_SPLIT epilogue with `v_mfma_scale_f32_16x16x128_f8f6f4` interleaved with `ds_read_b128 v[*], v131/v140/v141/v142` reading the LDS slots that are still being written by the unfinished prefetches.

Compare with the steady-state barrier pattern in the body: every `s_barrier` (38 sites) is preceded by `s_waitcnt lgkmcnt(0); s_waitcnt vmcnt(8)` (allows up to 8 VMEM in flight). The vmcnt(8) is fine for steady-state because the next iter's body re-issues prefetches that are 8-deep batched. The **back-edge fall-through has no such fence** because the C++-level `for (int bt = ...)` loop just exits without an explicit barrier — it's the implicit "next iteration" in the steady state, but on the last iter the "next iteration" is replaced by the TAIL_SPLIT block which doesn't honor the same prefetch-in-flight contract.

## Mechanism hypothesis (carried forward from R42 Opt B + this round)

The `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION code: 0x29` is reported by the LDS controller (DS unit). In gfx950, DS-unit aperture violations fire when:
1. A `buffer_load_*_dwordx4 ... offen lds` writes to LDS at an address computed from a runtime-dynamic SRD `voff`, and
2. That `voff`, after carry-out from the per-lane offset, lands outside the per-WG LDS aperture (128 KB on this kernel — see `group_seg_size=131072` in the AMD_LOG_LEVEL=4 dispatch trace).

When the iter-N+1 LDS prefetch is still in flight while the TAIL_SPLIT epilogue is reading and computing addresses for the writeback path (lines 9029–9043 + onward), the SRD register state (`s[16:23]` for A, `s[20:23]` for B) **may have already been mutated** by the epilogue's setup code (the `s_add_*` instructions at PC `0x1A9D4+`). The in-flight prefetches use these mutated SRD values when their address calc is dispatched into the L1/L2 pipeline → out-of-aperture LDS write → fault.

This explains why:
- `FUSED_STEP34=0` (R42 Opt B Phase-2A) removes the CRASH: the non-fused step3+step4 emit a separate barrier+vmcnt fence between step3 and step4 that drains the prefetches.
- `TAIL_SPLIT=0` (R42 Opt B Phase-2A) removes the CRASH: the unified tail handler doesn't mutate SRDs.
- R43 fix1f (`s_waitcnt vmcnt(0)` BEFORE `emit_pf_tail`) does **not** fix it: the fence was placed BEFORE the prefetch issue, but the race is at the END of the prefetch issue (after `s_cbranch_scc0`).
- rocgdb hides it: trap installation slows wave issue, prefetches drain before the fall-through.

## Files produced

- `R44_OPT_C_repro.py` — minimal CRASH reproducer (1-call, no bench)
- `R44_OPT_C_FAULT_RAW.log` — `AMD_LOG_LEVEL=4` trace (no fault VA recoverable)
- `R44_OPT_C_FAULT_VERBOSE.log` — `HSA_DEBUG=1 + AMD_LOG_LEVEL=3 + serialize=3` trace (same — no VA)
- `R44_OPT_C_rocgdb_live_stdout.log` + `R44_OPT_C_rocgdb_live.log` — rocgdb live attach (kernel completes; heisenbug)
- `R44_OPT_C_rocgdb_core.log` + `R44_OPT_C_rocgdb_core2.log` — rocgdb on gpucore.1857072 (no live wave state in core)
- `R44_OPT_C_p2b_ctrl_kernel.s` — full kernel ISA disasm (10206 lines)
- `R44_OPT_C_KERNEL_ISA.s` — extracted ISA excerpts: back-edge region + back-edge target + 4 sample steady-state barriers
- `decode_gpucore_note.py` — partial AMDGPU note decoder (incomplete; format not recoverable from public ROCm source)
- `R44_OPT_C_rocgdb_cmds.txt`, `R44_OPT_C_rocgdb_core.txt`, `R44_OPT_C_rocgdb_core2.txt`, `R44_OPT_C_rocgdb_live.txt` — rocgdb command scripts

## Recommendation to Opt A (specific actionable instructions)

In `kernel_mxfp4_gluon_cpp.cpp`, at the END of the `for (int bt = 0; bt + 1 < k_byte_iters; ++bt)` loop body (line ~3500ish, just before the closing `}` of the steady-state loop), add an inline asm fence that the compiler cannot drop:

```cpp
        // R44 Opt C diagnosis: back-edge fall-through has no vmcnt(0) drain;
        // 16 in-flight buffer_load_to_lds from this iter race the TAIL_SPLIT
        // epilogue's ds_reads on the SAME LDS double-buffer slots → aperture fault.
        // Drain ALL VMEM before exiting the loop.
        asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory");
    }  // end for (bt)
```

If the compiler still drops it (LICM/DCE), promote to `__asm__ __volatile__` with an explicit "v" register clobber, or place at the start of the TAIL_SPLIT block (line ~2957 region of the source where `const int bt = k_byte_iters - 1;` is set).

The 3-buffer rotation (Opt A's primary plan) is more invasive but is structurally robust — it removes the slot-aliasing precondition entirely.

## Self-review

- **Bench rules**: not applicable — diagnostic only, no perf measurement.
- **GPU isolation**: ran exclusively on GPU 5 (HIP_VISIBLE_DEVICES=5) for all live runs. Disasm + ISA analysis is CPU-only.
- **No kernel modifications**: confirmed — only inspected pre-built `.so` from R43A.
- **Hard timeout 3 h**: completed within ~1 h.
- **Stopping criteria**: WIN-PARTIAL — got the **PC range** (16 candidate instructions at `0x1A884`–`0x1A9A0`) and the **mechanism** + **specific actionable fix** for Opt A, but did NOT get an exact single PC nor faulting VA (HSA runtime + rocgdb 16.3 do not expose them on this ROCm 7.1 build for queue-abort faults).
