;; ============================================================================
;; R50A: HipKittens R44 baseline — kpair_64mfma_step34 ASM VOLATILE BODY
;;       (lines 2304-2535 of kernel_mxfp4_gluon_cpp.cpp)
;;       FUSED_STEP34=1 (R44 baseline), R48A_SPLIT_STEP34=0, R48B_MFMA_REORDER=0
;;
;; OBSERVATION:
;; Step3 has 32 MFMAs + 8 ds_reads (writing nxt_a0_d[0..7]).
;; Step4 has 32 MFMAs + 8 ds_reads (writing nxt_bl_d[0..7]).
;; In each step, ALL 8 ds_reads are FRONT-LOADED in Row 0 (Phase 0 + Phase 1).
;; Rows 1, 2, 3 contain 24 pure MFMAs (NO ds_reads).
;;
;; Distribution per Step3:
;;   Row 0 P0 (4 MFMAs + 4 ds_reads, 1:1 interleave) — uses ds_a0[0..3]
;;   Row 0 P1 (4 MFMAs + 4 ds_reads, 1:1 interleave) — uses ds_a0[4..7]
;;   Row 1 (8 MFMAs, no ds_reads)
;;   Row 2 (8 MFMAs, no ds_reads)
;;   Row 3 (8 MFMAs, no ds_reads)
;;   => 8 ds_reads concentrated in first 8 MFMAs (positions 1..8 of 32)
;;   => 24 consecutive pure MFMAs follow
;; ============================================================================
;; STEP 3: A1×Bl (32 MFMAs) + 8 ds_reads for nxt_a0
;; Operands: %0..%15 = acc_bl[0..15] (+a)
;;           %32..%39 = nxt_a0_d[0..7] (=&v output)
;;           %48..%55 = a1_*l/h (v input scratch)
;;           %56..%63 = bl_*l/h (v input)
;;           %72..%75 = sa0,sa1,sbl0,sbl1 scales
;;           %78,%79 = a0_p0, a0_p1 (LDS source addresses)

; Row 0 Phase 0: 4 MFMAs + 4 ds_reads (1:1 interleave)
v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %48, %56, %0,  %72, %74 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 %32, %78 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %48, %57, %1,  %72, %74 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 %33, %78 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %48, %58, %2,  %72, %75 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 %34, %78 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %48, %59, %3,  %72, %75 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 %35, %78 offset:6144
; Row 0 Phase 1: 4 MFMAs + 4 ds_reads (1:1 interleave)
v_mfma_scale_f32_16x16x128_f8f6f4 %0,  %52, %60, %0,  %72, %74 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 %36, %79 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 %1,  %52, %61, %1,  %72, %74 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 %37, %79 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 %2,  %52, %62, %2,  %72, %75 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 %38, %79 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 %3,  %52, %63, %3,  %72, %75 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 %39, %79 offset:6144
; Row 1 P0+P1: 8 PURE MFMAs (no ds_reads) — sa0 odd
v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %49, %56, %4,  %72, %74 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %49, %57, %5,  %72, %74 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %49, %58, %6,  %72, %75 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %49, %59, %7,  %72, %75 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 %4,  %53, %60, %4,  %72, %74 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 %5,  %53, %61, %5,  %72, %74 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 %6,  %53, %62, %6,  %72, %75 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 %7,  %53, %63, %7,  %72, %75 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
; Row 2 P0+P1: 8 PURE MFMAs (no ds_reads) — sa1 even
v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %50, %56, %8,  %73, %74 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %50, %57, %9,  %73, %74 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 %10, %50, %58, %10, %73, %75 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 %11, %50, %59, %11, %73, %75 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 %8,  %54, %60, %8,  %73, %74 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 %9,  %54, %61, %9,  %73, %74 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 %10, %54, %62, %10, %73, %75 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 %11, %54, %63, %11, %73, %75 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
; Row 3 P0+P1: 8 PURE MFMAs (no ds_reads) — sa1 odd
v_mfma_scale_f32_16x16x128_f8f6f4 %12, %51, %56, %12, %73, %74 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 %13, %51, %57, %13, %73, %74 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 %14, %51, %58, %14, %73, %75 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 %15, %51, %59, %15, %73, %75 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 %12, %55, %60, %12, %73, %74 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 %13, %55, %61, %13, %73, %74 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 %14, %55, %62, %14, %73, %75 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 %15, %55, %63, %15, %73, %75 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

; STEP 4 mirrors Step 3 with %16..%31=acc_br, %40..%47=nxt_bl_d, %64..%71=br_*,
;        %76,%77=sbr0,sbr1, %80,%81=bl_p0,bl_p1.
;        Same structure (4 MFMAs + 4 ds_reads in Row 0 P0/P1, then 24 pure MFMAs).
