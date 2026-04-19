;; ============================================================================
;; R50A: HipKittens R50A_AITER_INTERLEAVE=1 — kpair_64mfma_step34 NEW BODY
;;       Aiter-style 1:4 spread (1 ds_read per 4 MFMAs) instead of front-loaded.
;;
;; KEY INVARIANTS (vs R44 baseline):
;;   - Same MFMA count (32 in Step3, 32 in Step4)
;;   - Same ds_read count (8 in Step3, 8 in Step4)
;;   - Same MFMA operands (acc, A1×Bl/Br, scales) — UNCHANGED
;;   - Same ds_read targets (nxt_a0_d[0..7] / nxt_bl_d[0..7]) — UNCHANGED
;;   - Same ds_read source addresses (a0_p0/p1, bl_p0/p1) and OFFSETS — UNCHANGED
;;   - Same ASM volatile operand list and clobbers
;;
;; ONLY CHANGE: ds_read INSTRUCTIONS are spread across all 4 rows of Step3/4
;;              instead of being concentrated in Row 0.
;;
;; AITER PATTERN: groups of (2 MFMA + 1 ds_read), repeated for ~32 MFMAs gives ~16 ds_reads.
;; HK has 32 MFMAs + 8 ds_reads per step => ratio 4:1 (one ds_read every 4 MFMAs).
;; Spread plan per Step3 (32 MFMAs, 8 ds_reads):
;;   Row 0 P0: 4 MFMAs + 2 ds_reads (was 4)  — drop %34, %35 from this slot
;;   Row 0 P1: 4 MFMAs + 2 ds_reads (was 4)  — drop %38, %39 from this slot
;;   Row 1 P0: 4 MFMAs + 1 ds_read  (was 0)  — emit %34
;;   Row 1 P1: 4 MFMAs + 1 ds_read  (was 0)  — emit %38
;;   Row 2 P0: 4 MFMAs + 1 ds_read  (was 0)  — emit %35
;;   Row 2 P1: 4 MFMAs + 1 ds_read  (was 0)  — emit %39
;;   Row 3 P0: 4 MFMAs + 0 ds_reads (was 0)
;;   Row 3 P1: 4 MFMAs + 0 ds_reads (was 0)
;;
;; Note: Row 3 still has 0 ds_reads, but the long pure-MFMA stretch is now 8
;; instead of 24 — this should improve VMEM/LDS counter overlap.
;;
;; Same scheme for Step 4 with bl_p0/p1 → %44..%47.
;; ============================================================================

; ════════ STEP 3: A1×Bl (32 MFMAs) + 8 ds_reads (1:4 spread) ════════
; Row 0 Phase 0: 4 MFMAs + 2 ds_reads (was 4 ds_reads)
v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %48, %56, %0,  %72, %74 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 %32, %78 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %48, %57, %1,  %72, %74 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 %33, %78 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %48, %58, %2,  %72, %75 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %48, %59, %3,  %72, %75 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
; Row 0 Phase 1: 4 MFMAs + 2 ds_reads (was 4 ds_reads)
v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %52, %60, %0,  %72, %74 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 %36, %79 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %52, %61, %1,  %72, %74 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 %37, %79 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %52, %62, %2,  %72, %75 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %52, %63, %3,  %72, %75 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
; Row 1 Phase 0: 4 MFMAs + 1 ds_read (was 0 ds_reads)
v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %49, %56, %4,  %72, %74 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %49, %57, %5,  %72, %74 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 %34, %78 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %49, %58, %6,  %72, %75 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %49, %59, %7,  %72, %75 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
; Row 1 Phase 1: 4 MFMAs + 1 ds_read (was 0 ds_reads)
v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %53, %60, %4,  %72, %74 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %53, %61, %5,  %72, %74 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 %38, %79 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %53, %62, %6,  %72, %75 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %53, %63, %7,  %72, %75 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
; Row 2 Phase 0: 4 MFMAs + 1 ds_read (was 0 ds_reads)
v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %50, %56, %8,  %73, %74 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %50, %57, %9,  %73, %74 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 %35, %78 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 %10, %50, %58, %10, %73, %75 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 %11, %50, %59, %11, %73, %75 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
; Row 2 Phase 1: 4 MFMAs + 1 ds_read (was 0 ds_reads)
v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %54, %60, %8,  %73, %74 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %54, %61, %9,  %73, %74 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 %39, %79 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 %10, %54, %62, %10, %73, %75 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 %11, %54, %63, %11, %73, %75 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
; Row 3 P0+P1: 8 PURE MFMAs (no ds_reads, all 8 already issued)
v_mfma_scale_f32_16x16x128_f8f6f4 %12, %51, %56, %12, %73, %74 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 %13, %51, %57, %13, %73, %74 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 %14, %51, %58, %14, %73, %75 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 %15, %51, %59, %15, %73, %75 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 %12, %55, %60, %12, %73, %74 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 %13, %55, %61, %13, %73, %74 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 %14, %55, %62, %14, %73, %75 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 %15, %55, %63, %15, %73, %75 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

; ════════ STEP 4: A1×Br (32 MFMAs) + 8 ds_reads (mirrors Step 3) ════════
; (Same pattern with %16..%31=acc_br, %40..%47=nxt_bl_d, %64..%71=br_*l/h,
;  %76,%77=sbr0,sbr1, %80,%81=bl_p0,bl_p1)

; Row 0 Phase 0: 4 MFMAs + 2 ds_reads
v_mfma_scale_f32_16x16x128_f8f6f4 %16, %48, %64, %16, %72, %76 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 %40, %80 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 %17, %48, %65, %17, %72, %76 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 %41, %80 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 %18, %48, %66, %18, %72, %77 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 %19, %48, %67, %19, %72, %77 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
; Row 0 Phase 1: 4 MFMAs + 2 ds_reads
v_mfma_scale_f32_16x16x128_f8f6f4 %16, %52, %68, %16, %72, %76 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 %44, %81 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 %17, %52, %69, %17, %72, %76 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 %45, %81 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 %18, %52, %70, %18, %72, %77 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 %19, %52, %71, %19, %72, %77 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
; Row 1 Phase 0: 4 MFMAs + 1 ds_read
v_mfma_scale_f32_16x16x128_f8f6f4 %20, %49, %64, %20, %72, %76 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 %21, %49, %65, %21, %72, %76 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 %42, %80 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 %22, %49, %66, %22, %72, %77 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 %23, %49, %67, %23, %72, %77 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
; Row 1 Phase 1: 4 MFMAs + 1 ds_read
v_mfma_scale_f32_16x16x128_f8f6f4 %20, %53, %68, %20, %72, %76 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 %21, %53, %69, %21, %72, %76 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 %46, %81 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 %22, %53, %70, %22, %72, %77 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 %23, %53, %71, %23, %72, %77 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
; Row 2 Phase 0: 4 MFMAs + 1 ds_read
v_mfma_scale_f32_16x16x128_f8f6f4 %24, %50, %64, %24, %73, %76 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 %25, %50, %65, %25, %73, %76 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 %43, %80 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 %26, %50, %66, %26, %73, %77 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 %27, %50, %67, %27, %73, %77 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
; Row 2 Phase 1: 4 MFMAs + 1 ds_read
v_mfma_scale_f32_16x16x128_f8f6f4 %24, %54, %68, %24, %73, %76 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 %25, %54, %69, %25, %73, %76 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 %47, %81 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 %26, %54, %70, %26, %73, %77 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 %27, %54, %71, %27, %73, %77 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
; Row 3 P0+P1: 8 PURE MFMAs (no ds_reads, all 8 already issued)
v_mfma_scale_f32_16x16x128_f8f6f4 %28, %51, %64, %28, %73, %76 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 %29, %51, %65, %29, %73, %76 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 %30, %51, %66, %30, %73, %77 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 %31, %51, %67, %31, %73, %77 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 %28, %55, %68, %28, %73, %76 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 %29, %55, %69, %29, %73, %76 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 %30, %55, %70, %30, %73, %77 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 %31, %55, %71, %31, %73, %77 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
