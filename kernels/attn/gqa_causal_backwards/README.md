## Commands

```bash
# Symmetric D=128
make ATTN_B=16 ATTN_H=64 ATTN_H_KV=8 ATTN_N=4096
python test_python.py 16 4096 64 8 1

# Asymmetric D_QK=192, D_V=128
make asymmetric ATTN_B=16 ATTN_H=64 ATTN_H_KV=8 ATTN_N=4096 ATTN_D_QK=192 ATTN_D_V=128
python test_python_d192v128.py 16 4096 64 8 1

# ART kernel (D_QK=192, D_V=128) — single fused BWD
THUNDERKITTENS_ROOT=/shared_nfs/kyle/HipKittens3 /opt/rocm/bin/hipcc attn_bkwd_causal_d192v128_art.cpp \
  -DKITTENS_CDNA4 --offload-arch=gfx950 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math -std=c++20 -w \
  -I$THUNDERKITTENS_ROOT/include -I$THUNDERKITTENS_ROOT/prototype \
  $(python3 -m pybind11 --includes) -shared -fPIC -I/opt/rocm/include/hip \
  -DATTN_B=16 -DATTN_H=64 -DATTN_H_KV=8 -DATTN_N=4096 \
  -o tk_kernel_bkwd_art.so
```

---

## Optimization Log (D_QK=192, D_V=128 BWD)

### Target
- FWD: 1200-1300 TFLOPS (achieved: 1292T @ N=16384)
- BWD: 1200 TFLOPS
- All gradients: cos > 0.99 vs float64

### Phase 1: Non-ART kernel (TK auto register allocation)

**Original scalar dQ kernel (git HEAD `324a6263`):**
- Combined dK/dV: 20ms (~1100T) — near-optimal
- Separate dQ (scalar + bf16 atomics): 91ms — 3.2B atomic adds bottleneck
- Total BWD: 111ms = **124 TFLOPS**
- Correctness: dQ cos=0.999872, dK cos=0.999996, dV cos=0.999997

**Q-parallel dQ attempts (all failed):**
- mma_AtB: wrong semantics for dQ (reduces over wrong dim when KV_BLOCK=Q_TILE)
- mma_AB: col_l→row_l conversion broken on gfx950
- reinterpret_cast tile extraction: C++ row-major tiles[h][w] reads wrong tiles
- G::load with per-warp coords: crashes (GROUP op needs uniform coords)
- Global→register load: broken on gfx950
- AGPR aliasing: compiler aliases acc with persistent dQ accumulators

**Key gfx950 bugs discovered:**
1. `rt_32x16_s` (stride 8) produces garbage — use `rt_32x16_4_s` (stride 4)
2. `sv_fl<32>` global→shared load silently loads nothing on 64-lane warps
3. `transpose` broken for 32x32 tiles
4. `store(smem, col_l_tile)` unreliable
5. `amdgpu_num_vgpr(29)` forces VGPR-mode MFMA

**Agent results (10 parallel agents):**
| Agent | Result |
|-------|--------|
| Merge dQ into combined | 193 VGPR spills, not viable |
| dQ scheduling opt | 91→88ms (+3.8%) |
| dKV double-buffer | 20→18.8ms (+4%) |
| 8-warp kernel | 81 VGPR spills, not viable |
| Q-parallel MMA (×5) | All NaN due to AGPR aliasing |
| N=8192 benchmark | FWD 78ms, BWD 710ms |

### Phase 2: ART kernel (Assigned Register Tiles)

**Architecture:**
- Single fused kernel: dK + dV in one pass (dQ separate)
- `__attribute__((amdgpu_num_vgpr(29)))` — compiler uses only 29 VGPRs
- Explicit VGPR/AGPR register range assignments via ART
- WSK=32, DOT_SLICE_QO=16, STEP_QO=64, BLOCK_SIZE_KV=128

**Register layout (512 VGPR + 512 AGPR @ occupancy 1):**
```
AGPRs:
  a[0:95]    dK_j_T  [192×32] col_l  (persistent accumulator)
  a[96:143]  K_j     [32×192] row_l  (reload per iter)
  a[144:175] (free / V overlap area)
  a[176:199] Q_i     [16×192] row_l  (reload per slice)

VGPRs:
  v[0:28]    compiler-managed
  v[29]      -inf constant
  v[30:53]   P/dP/P_bf16/dP_bf16 (overlapping temporaries)
  v[62:93]   dO_i, dO_i_col
  v[94:117]  Q_i_col
  v[118:125] dQ_i_T
  v[126:127] L, delta scalars
  v[128:191] dV_j_T  [128×32] col_l  (persistent accumulator)
  v[192:223] V_j / dS_T / K_col  (overlapping, V reloaded per slice)
```

**Build stats:** 240 VGPRs, 200 AGPRs, 0 spills, 0 scratch

**Bugs found and fixed:**
1. L subtraction commented out during debug — restored
2. Barrier deadlock: `if (!skip) { ... barrier }` — different warps had different skip values. Fix: remove skip guard, let causal mask handle zeroing
3. Causal mask incomplete for WSK=32: needed `q_slice_pos == k_pos + 16` case
4. ds>0 shared memory subtile addressing issues with st_16x32_s swizzle

**Current results (2025-04-15):**
| Metric | Original | ART kernel |
|--------|----------|------------|
| Time (N=1024) | ~26ms | **4.6ms** |
| Time (N=4096) | 111ms | **46ms** |
| TFLOPS (N=4096) | 124T | **296T** |
| dV cos | 0.999997 | **0.974** |
| dK cos | 0.999996 | **0.769** (WIP) |
| dQ | cos=0.999 (scalar) | Not yet in ART |
| Spills | 0 | **0** |

**Remaining work:**
- Fix dK accuracy (Q_i_col address issue for 192-wide tiles)
- Restore dV to >0.99
- dQ as separate kernel pass
- Instruction interleaving (main perf lever: 2-3× expected)
- V load optimization (reduce global reloads)

### Files

| File | Description |
|------|-------------|
| `attn_bkwd_causal_d192v128.cpp` | Original non-ART BWD kernel (scalar dQ + combined dK/dV) |
| `attn_bkwd_causal_d192v128_art.cpp` | **ART-based fused BWD kernel (WIP)** |
| `attn_bkwd_causal.cpp` | Reference 128/128 BWD kernel (3378 lines, fully hand-scheduled) |
| `test_python_d192v128.py` | Test script for original kernel |
| `test_art_bwd.py` | Test script for ART kernel |
| `dq_kernel_insert.cpp` | Experimental Q-parallel dQ (not used) |
| `utils.cpp` | Shared utilities (atomic_pk_add_bf16_with_warpid) |
