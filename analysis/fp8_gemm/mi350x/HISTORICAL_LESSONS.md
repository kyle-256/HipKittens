# Historical Lessons (R102, condensed)

## Memory File References (use these as `[[name]]` links)

### Source-level dead-ends (don't retry)
- `[[lds-ktail-deadend]]` — RRR FUSED_KTAIL LDS coop-load broken
- `[[hk-agpr-deadend]]` — pinned-AGPR rewrite needed for 17-VGPR spill
- `[[fp8-rrr-attempt-h1]]` — B-load hoist 0 register delta
- `[[fp8-rrr-attempt-h2]]` — readfirstlane pin 0 effect
- `[[fp8-rrr-attempt-h4-bn128]]` — H3 pattern on bn128 +6 VGPR
- `[[fp8-rrr-attempt-h5-diag]]` — view-drop 0 delta (spill structural)
- `[[fp8-rrr-attempt-h8]]` — launch_bounds(_,2) ignored on gfx950 8-wave
- `[[fp8-rrr-attempt-h9]]` — 11-site sched_group_barrier regression
- `[[fp8-rrr-attempt-h10]]` — chunk_size sweep ±1% noise
- `[[fp8-rrr-attempt-h11_h12]]` — GRID_MUL sweep, bandwidth-bound ceiling
- `[[fp8-rrr-attempt-h14]]` — physics-bound 25% gap from data volume
- `[[bn128-per-warp-area-not-lever]]` — per-warp area swap insufficient
- `[[fp8-rrr-32x32-flawed-premise]]` — 32×32 wrapper alone won't降 spill
- `[[no-constant-sweep]]` — user mandate: ISA-level justification only
- `[[fp8-rcr-unroll-harmful]]` — RCR_MAIN_UNROLL≠2 active-harmful (R33)
- `[[fp8-rcr-partial-barriers]]` — never remove partial barriers (R39/R79)

### Committed wins (don't undo)
- `[[fp8-crr-agpr-inplace]]` — CRR spill 24→0 via mfma1616128_agpr_inplace
- `[[grouped-rcr-agpr-inplace]]` — FUSED_KTAIL a_kt1 serialization
- `[[fp8-rrr-attempt-h3]]` — H3 B SRD shift commit (HK 2d922b8b)
- `[[fp8-rrr-attempt-h6]]` — H6+H7 single-b0 + barrier (HK b7655e5a)
- `[[bf16-fuse-universal]]` — K_rem==64 fuse_ktail universal gate
- `[[bf16-byte-level-addressing]]` — bf16 per-group shifted gl view
- `[[swizzled-srd-oob-clamp]]` — non-swizzled SRD for byte-level HW clamp
- `[[var-k-per-group-view]]` — wgrad per-group shifted gl + non-swizzled SRD
- `[[fp8-rrr-state-carryover-fault]]` — RRR GPU fault on 2nd call FIXED
- `[[bn128-race-rcr-fix]]` — RCR bn128 race triple-buffer + FUSED gate
- **R43 (this session, HK 56f688fe)** — chunk_size 64→32 Triton match +1.5pp

### Physics/structural (architecture lever required)
- `[[fp8-rrr-attempt-h14]]` — bandwidth ceiling, algorithmic必
- `[[ck-grouped-gemm-reference]]` — CK <5% gap via 4 combined levers

### Process/build cache
- `[[verify-host-arch-first]]` — rocminfo before kernel debug
- `[[fp8-rrr-grad-a-fix-2026-05-19]]` — build cache nuke required
- `[[dual-hk-path]]` — every HK edit must cp to PT 3rdparty
- `[[propagate-changes]]` — sync.sh push without --mirror doesn't propagate deletes

### Tools
- `[[vgpr-spill-elimination-techniques]]` — 4 technique cheat-sheet
- `[[fp8-fwd-gap-not-spill]]` — PMC shows spill <5% lever for fwd gap
