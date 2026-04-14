	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.text
	.protected	_Z15attend_prep_ker17attn_prep_globals ; -- Begin function _Z15attend_prep_ker17attn_prep_globals
	.globl	_Z15attend_prep_ker17attn_prep_globals
	.p2align	8
	.type	_Z15attend_prep_ker17attn_prep_globals,@function
_Z15attend_prep_ker17attn_prep_globals: ; @_Z15attend_prep_ker17attn_prep_globals
; %bb.0:
	s_load_dwordx8 s[8:15], s[0:1], 0x0
	s_load_dwordx8 s[20:27], s[0:1], 0x30
	s_waitcnt lgkmcnt(0)
	s_load_dword s11, s[0:1], 0x20
	v_lshrrev_b32_e32 v1, 2, v0
	v_and_b32_e32 v39, 15, v0
	v_mov_b32_e32 v20, s8
	v_mov_b32_e32 v21, s9
	s_load_dword s5, s[0:1], 0x50
	s_load_dwordx2 s[8:9], s[0:1], 0x60
	s_load_dword s13, s[0:1], 0x80
	s_load_dwordx4 s[16:19], s[0:1], 0x70
	s_lshl_b32 s0, s4, 6
	v_and_or_b32 v38, v1, 48, s0
	s_mul_i32 s0, s2, s24
	v_add_u32_e32 v1, s0, v38
	v_mul_lo_u32 v1, v1, s26
	v_add_u32_e32 v1, s3, v1
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s0, s26, s5
	s_mul_i32 s1, s22, s24
	v_mul_lo_u32 v4, v1, s5
	s_mul_i32 s1, s0, s1
	v_mov_b32_e32 v2, s20
	v_mov_b32_e32 v3, s21
	v_ashrrev_i32_e32 v5, 31, v4
	s_lshl_b32 s1, s1, 1
	v_lshlrev_b32_e32 v23, 1, v39
	v_and_b32_e32 v22, 48, v0
	v_lshl_add_u64 v[16:17], v[4:5], 1, v[2:3]
	v_mov_b32_e32 v18, s1
	v_mov_b32_e32 v19, 0x20000
	v_mad_u64_u32 v[24:25], s[0:1], v23, s0, v[22:23]
	s_mov_b64 s[20:21], exec
.LBB0_1:                                ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v16
	v_readfirstlane_b32 s5, v17
	v_readfirstlane_b32 s6, v18
	v_readfirstlane_b32 s7, v19
	v_cmp_eq_u64_e32 vcc, s[4:5], v[16:17]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[18:19]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_load_dwordx4 v[12:15], v24, s[4:7], 0 offen
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_1
; %bb.2:
	s_mov_b64 exec, s[20:21]
	s_mov_b64 s[20:21], exec
.LBB0_3:                                ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v16
	v_readfirstlane_b32 s5, v17
	v_readfirstlane_b32 s6, v18
	v_readfirstlane_b32 s7, v19
	v_cmp_eq_u64_e32 vcc, s[4:5], v[16:17]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[18:19]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_load_dwordx4 v[8:11], v24, s[4:7], 0 offen offset:64
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_3
; %bb.4:
	s_mov_b64 exec, s[20:21]
	s_mov_b64 s[20:21], exec
.LBB0_5:                                ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v16
	v_readfirstlane_b32 s5, v17
	v_readfirstlane_b32 s6, v18
	v_readfirstlane_b32 s7, v19
	v_cmp_eq_u64_e32 vcc, s[4:5], v[16:17]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[18:19]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_load_dwordx4 v[4:7], v24, s[4:7], 0 offen offset:128
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_5
; %bb.6:
	s_mov_b64 exec, s[20:21]
	s_mov_b64 s[20:21], exec
.LBB0_7:                                ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v16
	v_readfirstlane_b32 s5, v17
	v_readfirstlane_b32 s6, v18
	v_readfirstlane_b32 s7, v19
	v_cmp_eq_u64_e32 vcc, s[4:5], v[16:17]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[18:19]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_load_dwordx4 v[0:3], v24, s[4:7], 0 offen offset:192
                                        ; implicit-def: $vgpr16_vgpr17_vgpr18_vgpr19
                                        ; implicit-def: $vgpr24_vgpr25
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_7
; %bb.8:
	s_mov_b64 exec, s[20:21]
	s_mul_i32 s0, s2, s12
	v_add_u32_e32 v16, s0, v38
	v_mul_lo_u32 v16, v16, s14
	v_add_u32_e32 v16, s3, v16
	s_mul_i32 s0, s14, s11
	s_mul_i32 s1, s10, s12
	v_mul_lo_u32 v16, v16, s11
	s_mul_i32 s1, s0, s1
	v_ashrrev_i32_e32 v17, 31, v16
	s_lshl_b32 s1, s1, 1
	v_lshl_add_u64 v[28:29], v[16:17], 1, v[20:21]
	v_mov_b32_e32 v30, s1
	v_mov_b32_e32 v31, 0x20000
	v_mad_u64_u32 v[36:37], s[0:1], v23, s0, v[22:23]
	s_mov_b64 s[10:11], exec
.LBB0_9:                                ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v28
	v_readfirstlane_b32 s5, v29
	v_readfirstlane_b32 s6, v30
	v_readfirstlane_b32 s7, v31
	v_cmp_eq_u64_e32 vcc, s[4:5], v[28:29]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[30:31]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_load_dwordx4 v[32:35], v36, s[4:7], 0 offen
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_9
; %bb.10:
	s_mov_b64 exec, s[10:11]
	s_mov_b64 s[10:11], exec
.LBB0_11:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v28
	v_readfirstlane_b32 s5, v29
	v_readfirstlane_b32 s6, v30
	v_readfirstlane_b32 s7, v31
	v_cmp_eq_u64_e32 vcc, s[4:5], v[28:29]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[30:31]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_load_dwordx4 v[24:27], v36, s[4:7], 0 offen offset:64
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_11
; %bb.12:
	s_mov_b64 exec, s[10:11]
	s_mov_b64 s[10:11], exec
.LBB0_13:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v28
	v_readfirstlane_b32 s5, v29
	v_readfirstlane_b32 s6, v30
	v_readfirstlane_b32 s7, v31
	v_cmp_eq_u64_e32 vcc, s[4:5], v[28:29]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[30:31]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_load_dwordx4 v[20:23], v36, s[4:7], 0 offen offset:128
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_13
; %bb.14:
	s_mov_b64 exec, s[10:11]
	s_mov_b64 s[10:11], exec
.LBB0_15:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v28
	v_readfirstlane_b32 s5, v29
	v_readfirstlane_b32 s6, v30
	v_readfirstlane_b32 s7, v31
	v_cmp_eq_u64_e32 vcc, s[4:5], v[28:29]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[30:31]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_load_dwordx4 v[16:19], v36, s[4:7], 0 offen offset:192
                                        ; implicit-def: $vgpr28_vgpr29_vgpr30_vgpr31
                                        ; implicit-def: $vgpr36_vgpr37
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_15
; %bb.16:
	s_mov_b64 exec, s[10:11]
	s_waitcnt vmcnt(7)
	v_alignbit_b32 v12, v12, v12, 16
	s_waitcnt vmcnt(3)
	v_and_b32_e32 v37, 0xffff0000, v35
	v_lshlrev_b32_e32 v36, 16, v35
	v_and_b32_e32 v43, 0xffff0000, v15
	v_lshlrev_b32_e32 v42, 16, v15
	s_waitcnt vmcnt(2)
	v_and_b32_e32 v45, 0xffff0000, v27
	v_lshlrev_b32_e32 v44, 16, v27
	v_and_b32_e32 v49, 0xffff0000, v11
	v_lshlrev_b32_e32 v48, 16, v11
	v_alignbit_b32 v28, v32, v32, 16
	v_and_b32_e32 v31, 0xffff0000, v33
	v_lshlrev_b32_e32 v30, 16, v33
	v_and_b32_e32 v33, 0xffff0000, v34
	v_lshlrev_b32_e32 v32, 16, v34
	v_and_b32_e32 v35, 0xffff0000, v12
	v_lshlrev_b32_e32 v34, 16, v12
	v_and_b32_e32 v41, 0xffff0000, v13
	v_lshlrev_b32_e32 v40, 16, v13
	v_and_b32_e32 v13, 0xffff0000, v14
	v_lshlrev_b32_e32 v12, 16, v14
	v_pk_mul_f32 v[14:15], v[36:37], v[42:43]
	v_and_b32_e32 v37, 0xffff0000, v25
	v_lshlrev_b32_e32 v36, 16, v25
	v_and_b32_e32 v43, 0xffff0000, v26
	v_lshlrev_b32_e32 v42, 16, v26
	v_and_b32_e32 v27, 0xffff0000, v9
	v_lshlrev_b32_e32 v26, 16, v9
	v_and_b32_e32 v47, 0xffff0000, v10
	v_lshlrev_b32_e32 v46, 16, v10
	v_pk_mul_f32 v[10:11], v[44:45], v[48:49]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v51, 0xffff0000, v23
	v_lshlrev_b32_e32 v50, 16, v23
	v_and_b32_e32 v55, 0xffff0000, v7
	v_lshlrev_b32_e32 v54, 16, v7
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v57, 0xffff0000, v19
	v_lshlrev_b32_e32 v56, 16, v19
	v_and_b32_e32 v61, 0xffff0000, v3
	v_lshlrev_b32_e32 v60, 16, v3
	v_and_b32_e32 v53, 0xffff0000, v6
	v_lshlrev_b32_e32 v52, 16, v6
	v_pk_mul_f32 v[6:7], v[50:51], v[54:55]
	v_and_b32_e32 v51, 0xffff0000, v17
	v_lshlrev_b32_e32 v50, 16, v17
	v_and_b32_e32 v55, 0xffff0000, v18
	v_lshlrev_b32_e32 v54, 16, v18
	v_and_b32_e32 v19, 0xffff0000, v1
	v_lshlrev_b32_e32 v18, 16, v1
	v_and_b32_e32 v59, 0xffff0000, v2
	v_lshlrev_b32_e32 v58, 16, v2
	v_pk_mul_f32 v[2:3], v[56:57], v[60:61]
	v_pk_fma_f32 v[10:11], v[36:37], v[26:27], v[10:11]
	v_and_b32_e32 v25, 0xffff0000, v24
	v_lshlrev_b32_e32 v24, 16, v24
	v_and_b32_e32 v9, 0xffff0000, v8
	v_lshlrev_b32_e32 v8, 16, v8
	v_pk_fma_f32 v[10:11], v[42:43], v[46:47], v[10:11]
	v_pk_fma_f32 v[2:3], v[50:51], v[18:19], v[2:3]
	v_and_b32_e32 v17, 0xffff0000, v16
	v_lshlrev_b32_e32 v16, 16, v16
	v_and_b32_e32 v1, 0xffff0000, v0
	v_lshlrev_b32_e32 v0, 16, v0
	v_pk_fma_f32 v[8:9], v[24:25], v[8:9], v[10:11]
	v_pk_fma_f32 v[2:3], v[54:55], v[58:59], v[2:3]
	v_and_b32_e32 v45, 0xffff0000, v21
	v_lshlrev_b32_e32 v44, 16, v21
	v_and_b32_e32 v49, 0xffff0000, v22
	v_lshlrev_b32_e32 v48, 16, v22
	v_and_b32_e32 v23, 0xffff0000, v5
	v_lshlrev_b32_e32 v22, 16, v5
	v_add_f32_e32 v10, v8, v9
	v_pk_fma_f32 v[8:9], v[30:31], v[40:41], v[14:15]
	v_pk_fma_f32 v[0:1], v[16:17], v[0:1], v[2:3]
	v_and_b32_e32 v29, 0xffff0000, v28
	v_lshlrev_b32_e32 v28, 16, v28
	v_pk_fma_f32 v[8:9], v[32:33], v[12:13], v[8:9]
	v_pk_fma_f32 v[6:7], v[44:45], v[22:23], v[6:7]
	v_add_f32_e32 v0, v0, v1
	v_mbcnt_lo_u32_b32 v1, -1, 0
	v_and_b32_e32 v21, 0xffff0000, v20
	v_lshlrev_b32_e32 v20, 16, v20
	v_and_b32_e32 v5, 0xffff0000, v4
	v_lshlrev_b32_e32 v4, 16, v4
	v_pk_fma_f32 v[8:9], v[28:29], v[34:35], v[8:9]
	v_pk_fma_f32 v[6:7], v[48:49], v[52:53], v[6:7]
	v_mbcnt_hi_u32_b32 v1, -1, v1
	v_add_f32_e32 v8, v8, v9
	v_pk_fma_f32 v[4:5], v[20:21], v[4:5], v[6:7]
	v_and_b32_e32 v2, 63, v1
	v_add_f32_e32 v8, v10, v8
	v_add_f32_e32 v4, v4, v5
	v_cmp_gt_u32_e32 vcc, 32, v2
	v_add_f32_e32 v4, v8, v4
	v_add_f32_e32 v0, v4, v0
	v_cndmask_b32_e64 v3, 0, 32, vcc
	v_add_lshl_u32 v3, v3, v1, 2
	ds_bpermute_b32 v3, v3, v0
	v_cmp_gt_u32_e32 vcc, 48, v2
	s_mul_i32 s0, s2, s16
	s_add_i32 s0, s0, s3
	v_cndmask_b32_e64 v2, 0, 16, vcc
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v0, v0, v3
	v_add_lshl_u32 v2, v2, v1, 2
	ds_bpermute_b32 v2, v2, v0
	v_and_or_b32 v1, v1, 64, v39
	v_lshlrev_b32_e32 v1, 2, v1
	s_mul_i32 s0, s0, s18
	s_mul_i32 s0, s0, s13
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v0, v0, v2
	ds_bpermute_b32 v4, v1, v0
	v_add_u32_e32 v0, s0, v38
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshl_add_u64 v[0:1], v[0:1], 2, s[8:9]
	v_lshlrev_b32_e32 v2, 2, v39
	v_mov_b32_e32 v3, 0
	v_lshl_add_u64 v[0:1], v[0:1], 0, v[2:3]
	s_waitcnt lgkmcnt(0)
	global_store_dword v[0:1], v4, off
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z15attend_prep_ker17attn_prep_globals
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 152
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 1
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 62
		.amdhsa_next_free_sgpr 28
		.amdhsa_accum_offset 64
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end0:
	.size	_Z15attend_prep_ker17attn_prep_globals, .Lfunc_end0-_Z15attend_prep_ker17attn_prep_globals
                                        ; -- End function
	.set _Z15attend_prep_ker17attn_prep_globals.num_vgpr, 62
	.set _Z15attend_prep_ker17attn_prep_globals.num_agpr, 0
	.set _Z15attend_prep_ker17attn_prep_globals.numbered_sgpr, 28
	.set _Z15attend_prep_ker17attn_prep_globals.private_seg_size, 0
	.set _Z15attend_prep_ker17attn_prep_globals.uses_vcc, 1
	.set _Z15attend_prep_ker17attn_prep_globals.uses_flat_scratch, 0
	.set _Z15attend_prep_ker17attn_prep_globals.has_dyn_sized_stack, 0
	.set _Z15attend_prep_ker17attn_prep_globals.has_recursion, 0
	.set _Z15attend_prep_ker17attn_prep_globals.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 1524
; TotalNumSgprs: 34
; NumVgprs: 62
; NumAgprs: 0
; TotalNumVgprs: 62
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 4
; VGPRBlocks: 7
; NumSGPRsForWavesPerEU: 34
; NumVGPRsForWavesPerEU: 62
; AccumOffset: 64
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 15
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.protected	_Z21attend_dq_shuffle_ker23attn_dq_shuffle_globals ; -- Begin function _Z21attend_dq_shuffle_ker23attn_dq_shuffle_globals
	.globl	_Z21attend_dq_shuffle_ker23attn_dq_shuffle_globals
	.p2align	8
	.type	_Z21attend_dq_shuffle_ker23attn_dq_shuffle_globals,@function
_Z21attend_dq_shuffle_ker23attn_dq_shuffle_globals: ; @_Z21attend_dq_shuffle_ker23attn_dq_shuffle_globals
; %bb.0:
	s_load_dwordx8 s[16:23], s[0:1], 0x0
	s_load_dwordx8 s[8:15], s[0:1], 0x30
	s_load_dword s5, s[0:1], 0x20
	s_waitcnt lgkmcnt(0)
	s_load_dword s11, s[0:1], 0x50
	s_lshl_b32 s0, s4, 6
	v_lshrrev_b32_e32 v1, 2, v0
	v_and_or_b32 v1, v1, 48, s0
	s_mul_i32 s0, s2, s20
	s_add_i32 s0, s0, s3
	s_mul_i32 s0, s0, s22
	v_add_u32_e32 v4, s0, v1
	s_mul_i32 s0, s18, s20
	v_mul_lo_u32 v4, v4, s5
	s_mul_i32 s0, s0, s22
	v_mov_b32_e32 v2, s16
	v_mov_b32_e32 v3, s17
	v_ashrrev_i32_e32 v5, 31, v4
	s_mul_i32 s0, s0, s5
	v_lshl_add_u64 v[6:7], v[4:5], 1, v[2:3]
	s_lshl_b32 s0, s0, 1
	v_lshlrev_b32_e32 v2, 4, v0
	v_mov_b32_e32 v30, s8
	v_mov_b32_e32 v31, s9
	v_mov_b32_e32 v8, s0
	v_mov_b32_e32 v9, 0x20000
	v_and_b32_e32 v2, 0x3f0, v2
	s_mov_b64 s[8:9], exec
.LBB1_1:                                ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v6
	v_readfirstlane_b32 s5, v7
	v_readfirstlane_b32 s6, v8
	v_readfirstlane_b32 s7, v9
	v_cmp_eq_u64_e32 vcc, s[4:5], v[6:7]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[8:9]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_load_dwordx4 v[26:29], v2, s[4:7], 0 offen
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB1_1
; %bb.2:
	s_mov_b64 exec, s[8:9]
	s_mov_b64 s[8:9], exec
.LBB1_3:                                ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v6
	v_readfirstlane_b32 s5, v7
	v_readfirstlane_b32 s6, v8
	v_readfirstlane_b32 s7, v9
	v_cmp_eq_u64_e32 vcc, s[4:5], v[6:7]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[8:9]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_load_dwordx4 v[22:25], v2, s[4:7], 0 offen offset:1024
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB1_3
; %bb.4:
	s_mov_b64 exec, s[8:9]
	s_mov_b64 s[8:9], exec
.LBB1_5:                                ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v6
	v_readfirstlane_b32 s5, v7
	v_readfirstlane_b32 s6, v8
	v_readfirstlane_b32 s7, v9
	v_cmp_eq_u64_e32 vcc, s[4:5], v[6:7]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[8:9]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_load_dwordx4 v[18:21], v2, s[4:7], 0 offen offset:2048
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB1_5
; %bb.6:
	s_mov_b64 exec, s[8:9]
	s_mov_b64 s[8:9], exec
.LBB1_7:                                ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v6
	v_readfirstlane_b32 s5, v7
	v_readfirstlane_b32 s6, v8
	v_readfirstlane_b32 s7, v9
	v_cmp_eq_u64_e32 vcc, s[4:5], v[6:7]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[8:9]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_load_dwordx4 v[14:17], v2, s[4:7], 0 offen offset:3072
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB1_7
; %bb.8:
	s_mov_b64 exec, s[8:9]
	v_or_b32_e32 v32, 0x1000, v2
	s_mov_b64 s[8:9], exec
.LBB1_9:                                ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v6
	v_readfirstlane_b32 s5, v7
	v_readfirstlane_b32 s6, v8
	v_readfirstlane_b32 s7, v9
	v_cmp_eq_u64_e32 vcc, s[4:5], v[6:7]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[8:9]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_load_dwordx4 v[10:13], v32, s[4:7], 0 offen
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB1_9
; %bb.10:
	s_mov_b64 exec, s[8:9]
	s_mov_b64 s[8:9], exec
.LBB1_11:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v6
	v_readfirstlane_b32 s5, v7
	v_readfirstlane_b32 s6, v8
	v_readfirstlane_b32 s7, v9
	v_cmp_eq_u64_e32 vcc, s[4:5], v[6:7]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[8:9]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_load_dwordx4 v[2:5], v32, s[4:7], 0 offen offset:1024
                                        ; implicit-def: $vgpr6_vgpr7_vgpr8_vgpr9
                                        ; implicit-def: $vgpr32
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB1_11
; %bb.12:
	s_mov_b64 exec, s[8:9]
	s_mul_i32 s2, s2, s12
	v_add_u32_e32 v1, s2, v1
	v_mul_lo_u32 v1, v1, s14
	v_add_u32_e32 v1, s3, v1
	s_waitcnt lgkmcnt(0)
	v_mul_lo_u32 v6, v1, s11
	v_lshlrev_b32_e32 v1, 2, v0
	s_mul_i32 s0, s14, s11
	v_and_b32_e32 v1, 12, v1
	v_ashrrev_i32_e32 v7, 31, v6
	v_mul_lo_u32 v1, v1, s0
	v_lshl_add_u64 v[6:7], v[6:7], 1, v[30:31]
	v_lshrrev_b32_e32 v8, 1, v0
	v_lshrrev_b32_e32 v9, 3, v0
	s_mul_i32 s1, s10, s12
	v_add_u32_e32 v30, s0, v1
	v_and_b32_e32 v8, 16, v8
	v_and_b32_e32 v9, 2, v9
	v_and_b32_e32 v0, 12, v0
	s_mul_i32 s1, s0, s1
	v_add_u32_e32 v31, s0, v30
	v_or3_b32 v0, v9, v0, v8
	s_lshl_b32 s1, s1, 1
	v_add_u32_e32 v32, s0, v31
	v_mov_b32_e32 v8, s1
	v_mov_b32_e32 v9, 0x20000
	v_add_lshl_u32 v36, v1, v0, 1
	v_add_lshl_u32 v35, v30, v0, 1
	v_add_lshl_u32 v34, v31, v0, 1
	v_add_lshl_u32 v33, v32, v0, 1
	s_mov_b64 s[2:3], exec
	s_waitcnt vmcnt(0)
.LBB1_13:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v6
	v_readfirstlane_b32 s5, v7
	v_readfirstlane_b32 s6, v8
	v_readfirstlane_b32 s7, v9
	v_cmp_eq_u64_e32 vcc, s[4:5], v[6:7]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[8:9]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dword v26, v36, s[4:7], 0 offen
                                        ; implicit-def: $vgpr36
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB1_13
; %bb.14:
	s_mov_b64 exec, s[2:3]
	s_mov_b64 s[2:3], exec
.LBB1_15:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v6
	v_readfirstlane_b32 s5, v7
	v_readfirstlane_b32 s6, v8
	v_readfirstlane_b32 s7, v9
	v_cmp_eq_u64_e32 vcc, s[4:5], v[6:7]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[8:9]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dword v27, v35, s[4:7], 0 offen
                                        ; implicit-def: $vgpr35
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB1_15
; %bb.16:
	s_mov_b64 exec, s[2:3]
	s_mov_b64 s[2:3], exec
.LBB1_17:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v6
	v_readfirstlane_b32 s5, v7
	v_readfirstlane_b32 s6, v8
	v_readfirstlane_b32 s7, v9
	v_cmp_eq_u64_e32 vcc, s[4:5], v[6:7]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[8:9]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dword v28, v34, s[4:7], 0 offen
                                        ; implicit-def: $vgpr34
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB1_17
; %bb.18:
	s_mov_b64 exec, s[2:3]
	s_mov_b64 s[2:3], exec
.LBB1_19:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v6
	v_readfirstlane_b32 s5, v7
	v_readfirstlane_b32 s6, v8
	v_readfirstlane_b32 s7, v9
	v_cmp_eq_u64_e32 vcc, s[4:5], v[6:7]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[8:9]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dword v29, v33, s[4:7], 0 offen
                                        ; implicit-def: $vgpr26_vgpr27_vgpr28_vgpr29
                                        ; implicit-def: $vgpr33
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB1_19
; %bb.20:
	s_mov_b64 exec, s[2:3]
	v_or_b32_e32 v26, 32, v0
	v_add_lshl_u32 v29, v1, v26, 1
	v_add_lshl_u32 v28, v30, v26, 1
	v_add_lshl_u32 v27, v31, v26, 1
	v_add_lshl_u32 v26, v32, v26, 1
	s_mov_b64 s[2:3], exec
.LBB1_21:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v6
	v_readfirstlane_b32 s5, v7
	v_readfirstlane_b32 s6, v8
	v_readfirstlane_b32 s7, v9
	v_cmp_eq_u64_e32 vcc, s[4:5], v[6:7]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[8:9]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dword v22, v29, s[4:7], 0 offen
                                        ; implicit-def: $vgpr29
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB1_21
; %bb.22:
	s_mov_b64 exec, s[2:3]
	s_mov_b64 s[2:3], exec
.LBB1_23:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v6
	v_readfirstlane_b32 s5, v7
	v_readfirstlane_b32 s6, v8
	v_readfirstlane_b32 s7, v9
	v_cmp_eq_u64_e32 vcc, s[4:5], v[6:7]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[8:9]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dword v23, v28, s[4:7], 0 offen
                                        ; implicit-def: $vgpr28
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB1_23
; %bb.24:
	s_mov_b64 exec, s[2:3]
	s_mov_b64 s[2:3], exec
.LBB1_25:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v6
	v_readfirstlane_b32 s5, v7
	v_readfirstlane_b32 s6, v8
	v_readfirstlane_b32 s7, v9
	v_cmp_eq_u64_e32 vcc, s[4:5], v[6:7]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[8:9]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dword v24, v27, s[4:7], 0 offen
                                        ; implicit-def: $vgpr27
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB1_25
; %bb.26:
	s_mov_b64 exec, s[2:3]
	s_mov_b64 s[2:3], exec
.LBB1_27:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v6
	v_readfirstlane_b32 s5, v7
	v_readfirstlane_b32 s6, v8
	v_readfirstlane_b32 s7, v9
	v_cmp_eq_u64_e32 vcc, s[4:5], v[6:7]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[8:9]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dword v25, v26, s[4:7], 0 offen
                                        ; implicit-def: $vgpr22_vgpr23_vgpr24_vgpr25
                                        ; implicit-def: $vgpr26
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB1_27
; %bb.28:
	s_mov_b64 exec, s[2:3]
	v_or_b32_e32 v22, 64, v0
	v_add_lshl_u32 v25, v1, v22, 1
	v_add_lshl_u32 v24, v30, v22, 1
	v_add_lshl_u32 v23, v31, v22, 1
	v_add_lshl_u32 v22, v32, v22, 1
	s_mov_b64 s[2:3], exec
.LBB1_29:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v6
	v_readfirstlane_b32 s5, v7
	v_readfirstlane_b32 s6, v8
	v_readfirstlane_b32 s7, v9
	v_cmp_eq_u64_e32 vcc, s[4:5], v[6:7]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[8:9]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dword v18, v25, s[4:7], 0 offen
                                        ; implicit-def: $vgpr25
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB1_29
; %bb.30:
	s_mov_b64 exec, s[2:3]
	s_mov_b64 s[2:3], exec
.LBB1_31:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v6
	v_readfirstlane_b32 s5, v7
	v_readfirstlane_b32 s6, v8
	v_readfirstlane_b32 s7, v9
	v_cmp_eq_u64_e32 vcc, s[4:5], v[6:7]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[8:9]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dword v19, v24, s[4:7], 0 offen
                                        ; implicit-def: $vgpr24
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB1_31
; %bb.32:
	s_mov_b64 exec, s[2:3]
	s_mov_b64 s[2:3], exec
.LBB1_33:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v6
	v_readfirstlane_b32 s5, v7
	v_readfirstlane_b32 s6, v8
	v_readfirstlane_b32 s7, v9
	v_cmp_eq_u64_e32 vcc, s[4:5], v[6:7]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[8:9]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dword v20, v23, s[4:7], 0 offen
                                        ; implicit-def: $vgpr23
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB1_33
; %bb.34:
	s_mov_b64 exec, s[2:3]
	s_mov_b64 s[2:3], exec
.LBB1_35:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v6
	v_readfirstlane_b32 s5, v7
	v_readfirstlane_b32 s6, v8
	v_readfirstlane_b32 s7, v9
	v_cmp_eq_u64_e32 vcc, s[4:5], v[6:7]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[8:9]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dword v21, v22, s[4:7], 0 offen
                                        ; implicit-def: $vgpr18_vgpr19_vgpr20_vgpr21
                                        ; implicit-def: $vgpr22
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB1_35
; %bb.36:
	s_mov_b64 exec, s[2:3]
	v_or_b32_e32 v18, 0x60, v0
	v_add_lshl_u32 v21, v1, v18, 1
	v_add_lshl_u32 v20, v30, v18, 1
	v_add_lshl_u32 v19, v31, v18, 1
	v_add_lshl_u32 v18, v32, v18, 1
	s_mov_b64 s[2:3], exec
.LBB1_37:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v6
	v_readfirstlane_b32 s5, v7
	v_readfirstlane_b32 s6, v8
	v_readfirstlane_b32 s7, v9
	v_cmp_eq_u64_e32 vcc, s[4:5], v[6:7]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[8:9]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dword v14, v21, s[4:7], 0 offen
                                        ; implicit-def: $vgpr21
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB1_37
; %bb.38:
	s_mov_b64 exec, s[2:3]
	s_mov_b64 s[2:3], exec
.LBB1_39:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v6
	v_readfirstlane_b32 s5, v7
	v_readfirstlane_b32 s6, v8
	v_readfirstlane_b32 s7, v9
	v_cmp_eq_u64_e32 vcc, s[4:5], v[6:7]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[8:9]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dword v15, v20, s[4:7], 0 offen
                                        ; implicit-def: $vgpr20
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB1_39
; %bb.40:
	s_mov_b64 exec, s[2:3]
	s_mov_b64 s[2:3], exec
.LBB1_41:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v6
	v_readfirstlane_b32 s5, v7
	v_readfirstlane_b32 s6, v8
	v_readfirstlane_b32 s7, v9
	v_cmp_eq_u64_e32 vcc, s[4:5], v[6:7]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[8:9]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dword v16, v19, s[4:7], 0 offen
                                        ; implicit-def: $vgpr19
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB1_41
; %bb.42:
	s_mov_b64 exec, s[2:3]
	s_mov_b64 s[2:3], exec
.LBB1_43:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v6
	v_readfirstlane_b32 s5, v7
	v_readfirstlane_b32 s6, v8
	v_readfirstlane_b32 s7, v9
	v_cmp_eq_u64_e32 vcc, s[4:5], v[6:7]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[8:9]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dword v17, v18, s[4:7], 0 offen
                                        ; implicit-def: $vgpr14_vgpr15_vgpr16_vgpr17
                                        ; implicit-def: $vgpr18
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB1_43
; %bb.44:
	s_mov_b64 exec, s[2:3]
	v_or_b32_e32 v14, 0x80, v0
	v_add_lshl_u32 v17, v1, v14, 1
	v_add_lshl_u32 v16, v30, v14, 1
	v_add_lshl_u32 v15, v31, v14, 1
	v_add_lshl_u32 v14, v32, v14, 1
	s_mov_b64 s[2:3], exec
.LBB1_45:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v6
	v_readfirstlane_b32 s5, v7
	v_readfirstlane_b32 s6, v8
	v_readfirstlane_b32 s7, v9
	v_cmp_eq_u64_e32 vcc, s[4:5], v[6:7]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[8:9]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dword v10, v17, s[4:7], 0 offen
                                        ; implicit-def: $vgpr17
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB1_45
; %bb.46:
	s_mov_b64 exec, s[2:3]
	s_mov_b64 s[2:3], exec
.LBB1_47:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v6
	v_readfirstlane_b32 s5, v7
	v_readfirstlane_b32 s6, v8
	v_readfirstlane_b32 s7, v9
	v_cmp_eq_u64_e32 vcc, s[4:5], v[6:7]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[8:9]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dword v11, v16, s[4:7], 0 offen
                                        ; implicit-def: $vgpr16
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB1_47
; %bb.48:
	s_mov_b64 exec, s[2:3]
	s_mov_b64 s[2:3], exec
.LBB1_49:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v6
	v_readfirstlane_b32 s5, v7
	v_readfirstlane_b32 s6, v8
	v_readfirstlane_b32 s7, v9
	v_cmp_eq_u64_e32 vcc, s[4:5], v[6:7]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[8:9]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dword v12, v15, s[4:7], 0 offen
                                        ; implicit-def: $vgpr15
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB1_49
; %bb.50:
	s_mov_b64 exec, s[2:3]
	s_mov_b64 s[2:3], exec
.LBB1_51:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v6
	v_readfirstlane_b32 s5, v7
	v_readfirstlane_b32 s6, v8
	v_readfirstlane_b32 s7, v9
	v_cmp_eq_u64_e32 vcc, s[4:5], v[6:7]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[8:9]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dword v13, v14, s[4:7], 0 offen
                                        ; implicit-def: $vgpr10_vgpr11_vgpr12_vgpr13
                                        ; implicit-def: $vgpr14
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB1_51
; %bb.52:
	s_mov_b64 exec, s[2:3]
	v_or_b32_e32 v0, 0xa0, v0
	v_add_lshl_u32 v11, v1, v0, 1
	v_add_lshl_u32 v10, v30, v0, 1
	v_add_lshl_u32 v1, v31, v0, 1
	v_add_lshl_u32 v0, v32, v0, 1
	s_mov_b64 s[2:3], exec
.LBB1_53:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v6
	v_readfirstlane_b32 s5, v7
	v_readfirstlane_b32 s6, v8
	v_readfirstlane_b32 s7, v9
	v_cmp_eq_u64_e32 vcc, s[4:5], v[6:7]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[8:9]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dword v2, v11, s[4:7], 0 offen
                                        ; implicit-def: $vgpr11
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB1_53
; %bb.54:
	s_mov_b64 exec, s[2:3]
	s_mov_b64 s[2:3], exec
.LBB1_55:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v6
	v_readfirstlane_b32 s5, v7
	v_readfirstlane_b32 s6, v8
	v_readfirstlane_b32 s7, v9
	v_cmp_eq_u64_e32 vcc, s[4:5], v[6:7]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[8:9]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dword v3, v10, s[4:7], 0 offen
                                        ; implicit-def: $vgpr10
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB1_55
; %bb.56:
	s_mov_b64 exec, s[2:3]
	s_mov_b64 s[2:3], exec
.LBB1_57:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v6
	v_readfirstlane_b32 s5, v7
	v_readfirstlane_b32 s6, v8
	v_readfirstlane_b32 s7, v9
	v_cmp_eq_u64_e32 vcc, s[4:5], v[6:7]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[8:9]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dword v4, v1, s[4:7], 0 offen
                                        ; implicit-def: $vgpr1
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB1_57
; %bb.58:
	s_mov_b64 exec, s[2:3]
	s_mov_b64 s[0:1], exec
.LBB1_59:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v6
	v_readfirstlane_b32 s5, v7
	v_readfirstlane_b32 s6, v8
	v_readfirstlane_b32 s7, v9
	v_cmp_eq_u64_e32 vcc, s[4:5], v[6:7]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[8:9]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dword v5, v0, s[4:7], 0 offen
                                        ; implicit-def: $vgpr6_vgpr7_vgpr8_vgpr9
                                        ; implicit-def: $vgpr2_vgpr3_vgpr4_vgpr5
                                        ; implicit-def: $vgpr0
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB1_59
; %bb.60:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z21attend_dq_shuffle_ker23attn_dq_shuffle_globals
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 104
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 1
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 37
		.amdhsa_next_free_sgpr 24
		.amdhsa_accum_offset 40
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end1:
	.size	_Z21attend_dq_shuffle_ker23attn_dq_shuffle_globals, .Lfunc_end1-_Z21attend_dq_shuffle_ker23attn_dq_shuffle_globals
                                        ; -- End function
	.set _Z21attend_dq_shuffle_ker23attn_dq_shuffle_globals.num_vgpr, 37
	.set _Z21attend_dq_shuffle_ker23attn_dq_shuffle_globals.num_agpr, 0
	.set _Z21attend_dq_shuffle_ker23attn_dq_shuffle_globals.numbered_sgpr, 24
	.set _Z21attend_dq_shuffle_ker23attn_dq_shuffle_globals.private_seg_size, 0
	.set _Z21attend_dq_shuffle_ker23attn_dq_shuffle_globals.uses_vcc, 1
	.set _Z21attend_dq_shuffle_ker23attn_dq_shuffle_globals.uses_flat_scratch, 0
	.set _Z21attend_dq_shuffle_ker23attn_dq_shuffle_globals.has_dyn_sized_stack, 0
	.set _Z21attend_dq_shuffle_ker23attn_dq_shuffle_globals.has_recursion, 0
	.set _Z21attend_dq_shuffle_ker23attn_dq_shuffle_globals.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 2428
; TotalNumSgprs: 30
; NumVgprs: 37
; NumAgprs: 0
; TotalNumVgprs: 37
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 3
; VGPRBlocks: 4
; NumSGPRsForWavesPerEU: 30
; NumVGPRsForWavesPerEU: 37
; AccumOffset: 40
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 9
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.type	__hip_cuid_7812801c6160aeee,@object ; @__hip_cuid_7812801c6160aeee
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_7812801c6160aeee
__hip_cuid_7812801c6160aeee:
	.byte	0                               ; 0x0
	.size	__hip_cuid_7812801c6160aeee, 1

	.ident	"clang version 20.0.0git (https://github.com/ROCm/llvm-project.git 27682a16360e33e37c4f3cc6adf9a620733f8fe1)"
	.ident	"AMD clang version 20.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.1.0 25425 1b0eada6b0ee93e2e694c8c146d23fca90bc11c5)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_7812801c6160aeee
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .offset:         0
        .size:           152
        .value_kind:     by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 152
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _Z15attend_prep_ker17attn_prep_globals
    .private_segment_fixed_size: 0
    .sgpr_count:     34
    .sgpr_spill_count: 0
    .symbol:         _Z15attend_prep_ker17attn_prep_globals.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     62
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .agpr_count:     0
    .args:
      - .offset:         0
        .size:           104
        .value_kind:     by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 104
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _Z21attend_dq_shuffle_ker23attn_dq_shuffle_globals
    .private_segment_fixed_size: 0
    .sgpr_count:     30
    .sgpr_spill_count: 0
    .symbol:         _Z21attend_dq_shuffle_ker23attn_dq_shuffle_globals.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     37
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
