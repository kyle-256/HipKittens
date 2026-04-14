	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.text
	.protected	_Z32attend_bwd_combined_d192v128_ker34attn_bwd_combined_d192v128_globals ; -- Begin function _Z32attend_bwd_combined_d192v128_ker34attn_bwd_combined_d192v128_globals
	.globl	_Z32attend_bwd_combined_d192v128_ker34attn_bwd_combined_d192v128_globals
	.p2align	8
	.type	_Z32attend_bwd_combined_d192v128_ker34attn_bwd_combined_d192v128_globals,@function
_Z32attend_bwd_combined_d192v128_ker34attn_bwd_combined_d192v128_globals: ; @_Z32attend_bwd_combined_d192v128_ker34attn_bwd_combined_d192v128_globals
; %bb.0:
	s_load_dword s5, s[0:1], 0x20
	s_load_dwordx2 s[6:7], s[0:1], 0x30
	s_load_dwordx4 s[8:11], s[0:1], 0x40
	s_waitcnt lgkmcnt(0)
	s_load_dword s9, s[0:1], 0x50
	s_load_dwordx4 s[24:27], s[0:1], 0xa0
	s_lshl_b32 s11, s3, 1
	s_waitcnt lgkmcnt(0)
	s_lshl_b32 s27, s2, 3
	s_max_i32 s33, s11, 0
	s_lshl_b32 s52, s3, 7
	s_mov_b64 s[14:15], src_shared_base
	s_cmp_lg_u32 0, -1
	s_cselect_b32 s14, 0, 0
	s_cselect_b32 s11, s15, 0
	s_and_b32 s12, s14, 15
	s_and_b32 s15, s14, -16
	s_add_u32 s15, s15, 16
	s_mov_b32 s13, 0
	s_addc_u32 s16, s11, 0
	s_cmp_eq_u64 s[12:13], 0
	s_cselect_b32 s53, s14, s15
	s_cselect_b32 s11, s11, s16
	s_add_u32 s14, s53, 0xc000
	s_addc_u32 s11, s11, 0
	s_and_b32 s12, s14, 15
	s_and_b32 s15, s14, -16
	s_add_u32 s15, s15, 16
	s_addc_u32 s16, s11, 0
	s_cmp_eq_u64 s[12:13], 0
	s_cselect_b32 s70, s14, s15
	s_cselect_b32 s11, s11, s16
	s_add_u32 s14, s70, 0xc000
	s_addc_u32 s11, s11, 0
	s_and_b32 s12, s14, 15
	s_and_b32 s15, s14, -16
	s_add_u32 s15, s15, 16
	s_addc_u32 s16, s11, 0
	s_cmp_eq_u64 s[12:13], 0
	s_cselect_b32 s71, s14, s15
	s_cselect_b32 s11, s11, s16
	s_add_u32 s14, s71, 0x8000
	s_addc_u32 s11, s11, 0
	s_and_b32 s12, s14, 15
	s_and_b32 s15, s14, -16
	s_add_u32 s15, s15, 16
	s_addc_u32 s16, s11, 0
	s_cmp_eq_u64 s[12:13], 0
	s_cselect_b32 s54, s14, s15
	s_cselect_b32 s11, s11, s16
	s_add_u32 s14, s54, 0x1000
	s_addc_u32 s11, s11, 0
	s_and_b32 s12, s14, 15
	s_and_b32 s15, s14, -16
	s_add_u32 s15, s15, 16
	s_addc_u32 s16, s11, 0
	s_cmp_eq_u64 s[12:13], 0
	s_cselect_b32 s56, s14, s15
	s_cselect_b32 s57, s11, s16
	s_add_u32 s11, s56, 0x200
	s_addc_u32 s14, s57, 0
	s_and_b32 s12, s11, 15
	s_and_b32 s15, s11, -16
	s_add_u32 s15, s15, 16
	s_addc_u32 s16, s14, 0
	s_load_dword s25, s[0:1], 0xb0
	s_load_dwordx4 s[28:31], s[0:1], 0x10
	s_load_dwordx2 s[62:63], s[0:1], 0x0
	s_cmp_eq_u64 s[12:13], 0
	s_mul_i32 s8, s4, s8
	s_cselect_b32 s61, s14, s16
	s_cselect_b32 s60, s11, s15
	s_add_i32 s8, s8, s52
	s_mul_i32 s8, s8, s10
	s_add_i32 s8, s8, s2
	s_mul_i32 s8, s8, s9
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s29, s30, s5
	v_lshlrev_b32_e32 v38, 4, v0
	s_mul_i32 s31, s26, s25
	s_mul_i32 s11, s10, s9
	s_ashr_i32 s9, s8, 31
	v_and_b32_e32 v47, 0xc00, v38
	s_lshl_b32 s49, s29, 4
	s_lshl_b32 s48, s31, 4
	s_lshl_b64 s[8:9], s[8:9], 1
	v_lshlrev_b32_e32 v40, 3, v0
	v_lshrrev_b32_e32 v39, 1, v0
	s_add_u32 s36, s6, s8
	v_bfe_u32 v1, v0, 1, 4
	v_and_b32_e32 v2, 8, v40
	v_add_u32_e32 v3, s53, v47
	s_movk_i32 s6, 0x70
	s_addc_u32 s37, s7, s9
	v_and_or_b32 v5, v39, s6, v2
	v_mul_lo_u32 v6, v1, s11
	v_readfirstlane_b32 s7, v3
	s_lshl_b32 s38, s11, 8
	s_mov_b32 s39, 0x110000
	v_add_lshl_u32 v7, v5, v6, 1
	s_mov_b32 m0, s7
	s_movk_i32 s40, 0x17ff
	buffer_load_dwordx4 v7, s[36:39], 0 offen lds
	v_or_b32_e32 v7, 0x1000, v38
	v_lshrrev_b32_e32 v9, 5, v7
	s_movk_i32 s44, 0x1800
	v_cmp_lt_u32_e32 vcc, s40, v7
	v_and_b32_e32 v9, 0xf0, v9
	v_add_u32_e32 v10, 0x7fffff40, v9
	v_cndmask_b32_e64 v8, 0, 16, vcc
	v_cmp_gt_u32_e32 vcc, s44, v7
	v_or_b32_e32 v8, v8, v1
	v_mul_lo_u32 v8, v8, s11
	v_cndmask_b32_e32 v7, v10, v9, vcc
	v_or_b32_e32 v7, v7, v2
	v_add_lshl_u32 v7, v7, v8, 1
	v_add_u32_e32 v8, 0x1000, v3
	v_mov_b32_e32 v4, 0x80
	v_readfirstlane_b32 s8, v8
	s_mov_b32 m0, s8
	s_lshl_b32 s8, s11, 4
	buffer_load_dwordx4 v7, s[36:39], 0 offen lds
	v_add_u32_e32 v6, s8, v6
	v_add_u32_e32 v7, 0x2000, v3
	v_add_u32_e32 v5, v5, v6
	v_readfirstlane_b32 s9, v7
	v_lshl_add_u32 v5, v5, 1, v4
	s_mov_b32 m0, s9
	s_movk_i32 s7, 0xf0
	buffer_load_dwordx4 v5, s[36:39], 0 offen lds
	v_lshrrev_b32_e32 v5, 5, v0
	v_or_b32_e32 v7, 24, v5
	v_mul_lo_u16_e32 v8, 22, v7
	v_lshrrev_b16_e32 v8, 6, v8
	v_and_b32_e32 v8, 12, v8
	v_add_u16_e32 v7, v7, v8
	v_lshlrev_b16_e32 v7, 4, v7
	v_add_u32_e32 v8, 0x3000, v3
	v_and_or_b32 v7, v7, s7, v2
	v_add_u32_e32 v6, s8, v6
	v_readfirstlane_b32 s9, v8
	v_add_lshl_u32 v7, v7, v6, 1
	s_mov_b32 m0, s9
	v_add_u32_e32 v6, s8, v6
	buffer_load_dwordx4 v7, s[36:39], 0 offen lds
	v_or_b32_e32 v7, 0x4000, v38
	v_lshrrev_b32_e32 v8, 9, v7
	v_mul_lo_u16_e32 v9, 43, v8
	v_lshrrev_b16_e32 v9, 7, v9
	v_mul_u32_u24_e32 v7, 0x2aab, v7
	v_and_b32_e32 v9, 12, v9
	v_add_u16_e32 v8, v8, v9
	v_lshrrev_b32_e32 v7, 22, v7
	v_and_or_b32 v7, v7, s6, v1
	v_lshlrev_b16_e32 v8, 4, v8
	v_and_or_b32 v8, v8, s7, v2
	v_mul_lo_u32 v7, v7, s11
	v_add_lshl_u32 v7, v8, v7, 1
	v_add_u32_e32 v8, 0x4000, v3
	s_load_dwordx8 s[72:79], s[0:1], 0x60
	v_readfirstlane_b32 s9, v8
	s_mov_b32 m0, s9
	v_lshrrev_b32_e32 v42, 6, v0
	buffer_load_dwordx4 v7, s[36:39], 0 offen lds
	v_or_b32_e32 v7, 40, v5
	v_mul_lo_u16_e32 v8, 43, v7
	v_lshrrev_b16_e32 v8, 7, v8
	v_and_b32_e32 v8, 12, v8
	v_add_u16_e32 v7, v7, v8
	v_lshlrev_b16_e32 v7, 4, v7
	v_add_u32_e32 v8, 0x5000, v3
	v_and_or_b32 v7, v7, s7, v2
	v_readfirstlane_b32 s9, v8
	v_add_lshl_u32 v7, v7, v6, 1
	s_mov_b32 m0, s9
	v_add_u32_e32 v8, 0x6000, v3
	buffer_load_dwordx4 v7, s[36:39], 0 offen lds
	v_or_b32_e32 v7, 48, v5
	v_mul_lo_u16_e32 v7, 43, v7
	v_lshrrev_b16_e32 v7, 7, v7
	v_and_b32_e32 v7, 12, v7
	v_add_u16_e32 v7, v5, v7
	v_lshlrev_b16_e32 v7, 4, v7
	v_and_or_b32 v7, v7, s7, v2
	v_add_u32_e32 v6, s8, v6
	v_readfirstlane_b32 s9, v8
	v_add_lshl_u32 v7, v7, v6, 1
	s_mov_b32 m0, s9
	v_add_u32_e32 v6, s8, v6
	buffer_load_dwordx4 v7, s[36:39], 0 offen lds
	v_or_b32_e32 v7, 0x7000, v38
	v_lshrrev_b32_e32 v8, 9, v7
	v_mul_lo_u16_e32 v9, 43, v8
	v_lshrrev_b16_e32 v9, 7, v9
	v_mul_u32_u24_e32 v7, 0x2aab, v7
	v_and_b32_e32 v9, 12, v9
	v_add_u16_e32 v8, v8, v9
	v_lshrrev_b32_e32 v7, 22, v7
	v_and_or_b32 v7, v7, s6, v1
	v_lshlrev_b16_e32 v8, 4, v8
	v_and_or_b32 v8, v8, s7, v2
	v_mul_lo_u32 v7, v7, s11
	v_add_lshl_u32 v7, v8, v7, 1
	v_add_u32_e32 v8, 0x7000, v3
	v_lshlrev_b32_e32 v41, 5, v42
	v_readfirstlane_b32 s9, v8
	s_mov_b32 m0, s9
	v_add_u32_e32 v8, 0x8000, v3
	buffer_load_dwordx4 v7, s[36:39], 0 offen lds
	v_or_b32_e32 v7, 64, v5
	v_mul_lo_u16_e32 v7, 43, v7
	v_lshrrev_b16_e32 v7, 7, v7
	v_and_b32_e32 v7, 12, v7
	v_add_u16_e32 v7, v5, v7
	v_lshlrev_b16_e32 v7, 4, v7
	v_and_or_b32 v7, v7, s7, v2
	v_readfirstlane_b32 s9, v8
	v_add_lshl_u32 v7, v7, v6, 1
	s_mov_b32 m0, s9
	v_add_u32_e32 v6, s8, v6
	buffer_load_dwordx4 v7, s[36:39], 0 offen lds
	v_or_b32_e32 v7, 0x48, v5
	v_mul_lo_u16_e32 v8, 43, v7
	v_lshrrev_b16_e32 v8, 7, v8
	v_and_b32_e32 v8, 12, v8
	v_add_u16_e32 v7, v7, v8
	v_lshlrev_b16_e32 v7, 4, v7
	v_add_u32_e32 v8, 0x9000, v3
	v_and_or_b32 v7, v7, s7, v2
	v_readfirstlane_b32 s9, v8
	v_add_lshl_u32 v7, v7, v6, 1
	s_mov_b32 m0, s9
	v_or_b32_e32 v58, s52, v41
	buffer_load_dwordx4 v7, s[36:39], 0 offen lds
	v_or_b32_e32 v7, 0xa000, v38
	v_lshrrev_b32_e32 v8, 9, v7
	v_mul_lo_u16_e32 v9, 43, v8
	v_lshrrev_b16_e32 v9, 7, v9
	v_mul_u32_u24_e32 v7, 0xaaab, v7
	v_and_b32_e32 v9, 12, v9
	v_add_u16_e32 v8, v8, v9
	v_lshrrev_b32_e32 v7, 24, v7
	v_and_or_b32 v1, v7, s6, v1
	v_lshlrev_b16_e32 v7, 4, v8
	v_and_or_b32 v7, v7, s7, v2
	v_mul_lo_u32 v1, v1, s11
	v_add_lshl_u32 v1, v7, v1, 1
	v_add_u32_e32 v7, 0xa000, v3
	v_bfe_u32 v8, v0, 2, 4
	v_readfirstlane_b32 s6, v7
	s_mov_b32 m0, s6
	v_mul_lo_u32 v11, v8, s29
	buffer_load_dwordx4 v1, s[36:39], 0 offen lds
	v_or_b32_e32 v1, 0x58, v5
	v_mul_lo_u16_e32 v5, 43, v1
	v_lshrrev_b16_e32 v5, 7, v5
	v_and_b32_e32 v5, 12, v5
	v_add_u16_e32 v1, v1, v5
	v_lshlrev_b16_e32 v1, 4, v1
	v_and_or_b32 v1, v1, s7, v2
	v_add_u32_e32 v2, s8, v6
	v_add_lshl_u32 v1, v1, v2, 1
	v_add_u32_e32 v2, 0xb000, v3
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v3, s73
	v_readfirstlane_b32 s6, v2
	s_mov_b32 m0, s6
	s_mul_i32 s6, s4, s76
	buffer_load_dwordx4 v1, s[36:39], 0 offen lds
	v_or_b32_e32 v1, 0x1000, v47
	v_cmp_lt_u32_e32 vcc, s40, v1
	v_lshrrev_b32_e32 v6, 5, v1
	v_add_u32_e32 v7, 0x7fffff40, v6
	v_cndmask_b32_e64 v5, 0, 16, vcc
	v_cmp_gt_u32_e32 vcc, s44, v1
	v_add_u32_e32 v1, s6, v58
	v_or_b32_e32 v5, v5, v8
	v_cndmask_b32_e32 v6, v7, v6, vcc
	v_mul_lo_u32 v7, v1, s78
	v_bitop3_b32 v1, v0, v38, 32 bitop3:0x6c
	v_lshrrev_b32_e32 v1, 1, v1
	v_and_b32_e32 v9, 24, v1
	v_lshrrev_b32_e32 v1, 5, v47
	v_or_b32_e32 v10, v9, v1
	v_or_b32_e32 v6, v6, v9
	v_mul_lo_u32 v5, v5, s29
	v_add_lshl_u32 v59, v6, v5, 1
	v_add3_u32 v5, v11, s49, v10
	s_movk_i32 s6, 0x60
	v_lshl_add_u32 v60, v5, 1, v4
	v_and_or_b32 v4, v39, s6, v9
	s_load_dword s36, s[0:1], 0x80
	s_load_dwordx2 s[64:65], s[0:1], 0x90
	s_load_dwordx2 s[34:35], s[0:1], 0x110
	s_load_dwordx8 s[8:15], s[0:1], 0xf0
	s_load_dwordx8 s[16:23], s[0:1], 0x120
	s_load_dwordx2 s[58:59], s[0:1], 0x140
	s_load_dwordx2 s[66:67], s[0:1], 0x150
	v_mad_u64_u32 v[4:5], s[6:7], v8, s31, v[4:5]
	s_waitcnt lgkmcnt(0)
	s_load_dword s15, s[0:1], 0x170
	s_load_dwordx4 s[40:43], s[0:1], 0x160
	s_load_dword s21, s[0:1], 0x1a0
	s_load_dwordx2 s[68:69], s[0:1], 0x180
	s_load_dwordx4 s[44:47], s[0:1], 0x190
	v_lshlrev_b32_e32 v61, 1, v4
	v_add_lshl_u32 v62, v4, s48, 1
	v_add_u32_e32 v4, s2, v7
	s_mul_i32 s11, s78, s36
	s_mul_i32 s13, s74, s76
	v_mul_lo_u32 v4, v4, s36
	v_mov_b32_e32 v2, s72
	v_ashrrev_i32_e32 v5, 31, v4
	v_and_b32_e32 v43, 15, v0
	s_mul_i32 s6, s11, s13
	v_lshl_add_u64 v[34:35], v[4:5], 1, v[2:3]
	s_lshl_b32 s6, s6, 1
	v_mul_lo_u32 v2, v43, s11
	v_and_b32_e32 v46, 48, v0
	v_and_b32_e32 v44, 32, v0
	v_lshrrev_b32_e32 v45, 2, v0
	v_add_lshl_u32 v1, v10, v11, 1
	v_mov_b32_e32 v36, s6
	v_mov_b32_e32 v37, 0x20000
	v_lshl_add_u32 v18, v2, 1, v46
	s_mov_b64 s[36:37], exec
.LBB0_1:                                ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s48, v34
	v_readfirstlane_b32 s49, v35
	v_readfirstlane_b32 s50, v36
	v_readfirstlane_b32 s51, v37
	v_cmp_eq_u64_e32 vcc, s[48:49], v[34:35]
	s_nop 0
	v_cmp_eq_u64_e64 s[6:7], s[50:51], v[36:37]
	s_and_b64 s[6:7], vcc, s[6:7]
	s_and_saveexec_b64 s[6:7], s[6:7]
	buffer_load_dwordx4 v[2:5], v18, s[48:51], 0 offen
	s_xor_b64 exec, exec, s[6:7]
	s_cbranch_execnz .LBB0_1
; %bb.2:
	s_mov_b64 exec, s[36:37]
	s_mov_b64 s[36:37], exec
.LBB0_3:                                ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s48, v34
	v_readfirstlane_b32 s49, v35
	v_readfirstlane_b32 s50, v36
	v_readfirstlane_b32 s51, v37
	v_cmp_eq_u64_e32 vcc, s[48:49], v[34:35]
	s_nop 0
	v_cmp_eq_u64_e64 s[6:7], s[50:51], v[36:37]
	s_and_b64 s[6:7], vcc, s[6:7]
	s_and_saveexec_b64 s[6:7], s[6:7]
	buffer_load_dwordx4 v[6:9], v18, s[48:51], 0 offen offset:64
	s_xor_b64 exec, exec, s[6:7]
	s_cbranch_execnz .LBB0_3
; %bb.4:
	s_mov_b64 exec, s[36:37]
	s_mov_b64 s[36:37], exec
.LBB0_5:                                ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s48, v34
	v_readfirstlane_b32 s49, v35
	v_readfirstlane_b32 s50, v36
	v_readfirstlane_b32 s51, v37
	v_cmp_eq_u64_e32 vcc, s[48:49], v[34:35]
	s_nop 0
	v_cmp_eq_u64_e64 s[6:7], s[50:51], v[36:37]
	s_and_b64 s[6:7], vcc, s[6:7]
	s_and_saveexec_b64 s[6:7], s[6:7]
	buffer_load_dwordx4 v[10:13], v18, s[48:51], 0 offen offset:128
	s_xor_b64 exec, exec, s[6:7]
	s_cbranch_execnz .LBB0_5
; %bb.6:
	s_mov_b64 exec, s[36:37]
	s_mov_b64 s[36:37], exec
.LBB0_7:                                ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s48, v34
	v_readfirstlane_b32 s49, v35
	v_readfirstlane_b32 s50, v36
	v_readfirstlane_b32 s51, v37
	v_cmp_eq_u64_e32 vcc, s[48:49], v[34:35]
	s_nop 0
	v_cmp_eq_u64_e64 s[6:7], s[50:51], v[36:37]
	s_and_b64 s[6:7], vcc, s[6:7]
	s_and_saveexec_b64 s[6:7], s[6:7]
	buffer_load_dwordx4 v[14:17], v18, s[48:51], 0 offen offset:192
	s_xor_b64 exec, exec, s[6:7]
	s_cbranch_execnz .LBB0_7
; %bb.8:
	s_mov_b64 exec, s[36:37]
	v_lshl_add_u32 v48, s11, 5, v18
	s_mov_b64 s[36:37], exec
.LBB0_9:                                ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s48, v34
	v_readfirstlane_b32 s49, v35
	v_readfirstlane_b32 s50, v36
	v_readfirstlane_b32 s51, v37
	v_cmp_eq_u64_e32 vcc, s[48:49], v[34:35]
	s_nop 0
	v_cmp_eq_u64_e64 s[6:7], s[50:51], v[36:37]
	s_and_b64 s[6:7], vcc, s[6:7]
	s_and_saveexec_b64 s[6:7], s[6:7]
	buffer_load_dwordx4 v[18:21], v48, s[48:51], 0 offen
	s_xor_b64 exec, exec, s[6:7]
	s_cbranch_execnz .LBB0_9
; %bb.10:
	s_mov_b64 exec, s[36:37]
	s_mov_b64 s[36:37], exec
.LBB0_11:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s48, v34
	v_readfirstlane_b32 s49, v35
	v_readfirstlane_b32 s50, v36
	v_readfirstlane_b32 s51, v37
	v_cmp_eq_u64_e32 vcc, s[48:49], v[34:35]
	s_nop 0
	v_cmp_eq_u64_e64 s[6:7], s[50:51], v[36:37]
	s_and_b64 s[6:7], vcc, s[6:7]
	s_and_saveexec_b64 s[6:7], s[6:7]
	buffer_load_dwordx4 v[22:25], v48, s[48:51], 0 offen offset:64
	s_xor_b64 exec, exec, s[6:7]
	s_cbranch_execnz .LBB0_11
; %bb.12:
	s_mov_b64 exec, s[36:37]
	s_mov_b64 s[36:37], exec
.LBB0_13:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s48, v34
	v_readfirstlane_b32 s49, v35
	v_readfirstlane_b32 s50, v36
	v_readfirstlane_b32 s51, v37
	v_cmp_eq_u64_e32 vcc, s[48:49], v[34:35]
	s_nop 0
	v_cmp_eq_u64_e64 s[6:7], s[50:51], v[36:37]
	s_and_b64 s[6:7], vcc, s[6:7]
	s_and_saveexec_b64 s[6:7], s[6:7]
	buffer_load_dwordx4 v[26:29], v48, s[48:51], 0 offen offset:128
	s_xor_b64 exec, exec, s[6:7]
	s_cbranch_execnz .LBB0_13
; %bb.14:
	s_mov_b64 exec, s[36:37]
	s_mov_b64 s[36:37], exec
.LBB0_15:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s48, v34
	v_readfirstlane_b32 s49, v35
	v_readfirstlane_b32 s50, v36
	v_readfirstlane_b32 s51, v37
	v_cmp_eq_u64_e32 vcc, s[48:49], v[34:35]
	s_nop 0
	v_cmp_eq_u64_e64 s[6:7], s[50:51], v[36:37]
	s_and_b64 s[6:7], vcc, s[6:7]
	s_and_saveexec_b64 s[6:7], s[6:7]
	buffer_load_dwordx4 v[30:33], v48, s[48:51], 0 offen offset:192
                                        ; implicit-def: $vgpr34_vgpr35_vgpr36_vgpr37
                                        ; implicit-def: $vgpr48
	s_xor_b64 exec, exec, s[6:7]
	s_cbranch_execnz .LBB0_15
; %bb.16:
	s_mov_b64 exec, s[36:37]
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s13, s4, s40
	s_add_i32 s6, s13, s27
	s_mul_i32 s15, s15, s42
	s_lshl_b32 s11, s33, 6
	s_mul_i32 s6, s15, s6
	s_add_i32 s6, s6, s11
	s_ashr_i32 s7, s6, 31
	s_lshl_b64 s[6:7], s[6:7], 2
	s_add_u32 s36, s66, s6
	s_mul_i32 s19, s4, s44
	s_addc_u32 s37, s67, s7
	s_add_i32 s6, s19, s27
	s_mul_i32 s21, s21, s46
	s_mul_i32 s6, s21, s6
	s_add_i32 s6, s6, s11
	v_lshlrev_b32_e32 v34, 2, v0
	s_ashr_i32 s7, s6, 31
	s_mov_b32 m0, s56
	s_movk_i32 s38, 0x100
	v_and_b32_e32 v63, 0xfc, v34
	s_lshl_b64 s[6:7], s[6:7], 2
	buffer_load_dword v63, s[36:39], 0 offen lds
	s_add_u32 s36, s68, s6
	s_mul_i32 s23, s4, s28
	s_addc_u32 s37, s69, s7
	s_add_i32 s6, s23, s11
	s_mul_i32 s6, s6, s30
	s_add_i32 s6, s6, s27
	s_mul_i32 s6, s6, s5
	s_ashr_i32 s7, s6, 31
	s_lshl_b64 s[6:7], s[6:7], 1
	v_add_u32_e32 v64, s70, v47
	s_mov_b32 m0, s60
	s_add_u32 s40, s62, s6
	v_readfirstlane_b32 s6, v64
	v_add_u32_e32 v65, 0x1000, v64
	buffer_load_dword v63, s[36:39], 0 offen lds
	s_addc_u32 s41, s63, s7
	s_lshl_b32 s42, s29, 6
	s_mov_b32 s43, s39
	s_mov_b32 m0, s6
	v_readfirstlane_b32 s6, v65
	v_add_u32_e32 v66, 0x2000, v64
	buffer_load_dwordx4 v1, s[40:43], 0 offen lds
	s_mov_b32 m0, s6
	v_readfirstlane_b32 s6, v66
	s_mul_i32 s29, s4, s24
	buffer_load_dwordx4 v59, s[40:43], 0 offen lds
	s_mov_b32 m0, s6
	s_add_i32 s6, s29, s11
	s_mul_i32 s6, s6, s26
	s_add_i32 s6, s6, s27
	s_mul_i32 s6, s6, s25
	s_ashr_i32 s7, s6, 31
	s_lshl_b64 s[6:7], s[6:7], 1
	v_add_u32_e32 v67, s71, v47
	s_add_u32 s44, s64, s6
	v_readfirstlane_b32 s6, v67
	v_add_u32_e32 v68, 0x1000, v67
	buffer_load_dwordx4 v60, s[40:43], 0 offen lds
	s_addc_u32 s45, s65, s7
	s_lshl_b32 s46, s31, 6
	s_mov_b32 s47, s39
	s_mov_b32 m0, s6
	v_readfirstlane_b32 s6, v68
	s_or_b32 s28, s11, 32
	buffer_load_dwordx4 v61, s[44:47], 0 offen lds
	s_mov_b32 m0, s6
	s_add_i32 s6, s23, s28
	s_mul_i32 s6, s6, s30
	s_add_i32 s6, s6, s27
	s_mul_i32 s6, s6, s5
	s_add_i32 s24, s70, 0x3000
	s_ashr_i32 s7, s6, 31
	s_lshl_b64 s[6:7], s[6:7], 1
	v_add_u32_e32 v69, s24, v47
	s_add_u32 s40, s62, s6
	v_readfirstlane_b32 s6, v69
	v_add_u32_e32 v70, 0x1000, v69
	buffer_load_dwordx4 v62, s[44:47], 0 offen lds
	s_addc_u32 s41, s63, s7
	s_mov_b32 m0, s6
	v_readfirstlane_b32 s6, v70
	v_add_u32_e32 v71, 0x2000, v69
	buffer_load_dwordx4 v1, s[40:43], 0 offen lds
	s_mov_b32 m0, s6
	v_readfirstlane_b32 s6, v71
	buffer_load_dwordx4 v59, s[40:43], 0 offen lds
	s_mov_b32 m0, s6
	s_add_i32 s6, s29, s28
	s_mul_i32 s6, s6, s26
	s_add_i32 s6, s6, s27
	s_mul_i32 s6, s6, s25
	s_add_i32 s24, s71, 0x2000
	s_ashr_i32 s7, s6, 31
	s_lshl_b64 s[6:7], s[6:7], 1
	v_add_u32_e32 v72, s24, v47
	s_add_u32 s44, s64, s6
	v_readfirstlane_b32 s6, v72
	v_add_u32_e32 v73, 0x1000, v72
	buffer_load_dwordx4 v60, s[40:43], 0 offen lds
	s_addc_u32 s45, s65, s7
	s_mov_b32 m0, s6
	v_readfirstlane_b32 s6, v73
	buffer_load_dwordx4 v61, s[44:47], 0 offen lds
	s_mov_b32 m0, s6
	s_mov_b32 s77, 0
	buffer_load_dwordx4 v62, s[44:47], 0 offen lds
	s_cmp_gt_i32 s3, 7
	v_mov_b32_e32 v35, 0
	v_accvgpr_write_b32 a144, 0
	v_accvgpr_write_b32 a145, 0
	v_accvgpr_write_b32 a146, 0
	v_accvgpr_write_b32 a147, 0
	v_accvgpr_write_b32 a148, 0
	v_accvgpr_write_b32 a149, 0
	v_accvgpr_write_b32 a150, 0
	v_accvgpr_write_b32 a151, 0
	v_accvgpr_write_b32 a152, 0
	v_accvgpr_write_b32 a153, 0
	v_accvgpr_write_b32 a154, 0
	v_accvgpr_write_b32 a155, 0
	v_accvgpr_write_b32 a156, 0
	v_accvgpr_write_b32 a157, 0
	v_accvgpr_write_b32 a158, 0
	v_accvgpr_write_b32 a159, 0
	v_accvgpr_write_b32 a128, 0
	v_accvgpr_write_b32 a129, 0
	v_accvgpr_write_b32 a130, 0
	v_accvgpr_write_b32 a131, 0
	v_accvgpr_write_b32 a132, 0
	v_accvgpr_write_b32 a133, 0
	v_accvgpr_write_b32 a134, 0
	v_accvgpr_write_b32 a135, 0
	v_accvgpr_write_b32 a136, 0
	v_accvgpr_write_b32 a137, 0
	v_accvgpr_write_b32 a138, 0
	v_accvgpr_write_b32 a139, 0
	v_accvgpr_write_b32 a140, 0
	v_accvgpr_write_b32 a141, 0
	v_accvgpr_write_b32 a142, 0
	v_accvgpr_write_b32 a143, 0
	v_accvgpr_write_b32 a112, 0
	v_accvgpr_write_b32 a113, 0
	v_accvgpr_write_b32 a114, 0
	v_accvgpr_write_b32 a115, 0
	v_accvgpr_write_b32 a116, 0
	v_accvgpr_write_b32 a117, 0
	v_accvgpr_write_b32 a118, 0
	v_accvgpr_write_b32 a119, 0
	v_accvgpr_write_b32 a120, 0
	v_accvgpr_write_b32 a121, 0
	v_accvgpr_write_b32 a122, 0
	v_accvgpr_write_b32 a123, 0
	v_accvgpr_write_b32 a124, 0
	v_accvgpr_write_b32 a125, 0
	v_accvgpr_write_b32 a126, 0
	v_accvgpr_write_b32 a127, 0
	v_accvgpr_write_b32 a96, 0
	v_accvgpr_write_b32 a97, 0
	v_accvgpr_write_b32 a98, 0
	v_accvgpr_write_b32 a99, 0
	v_accvgpr_write_b32 a100, 0
	v_accvgpr_write_b32 a101, 0
	v_accvgpr_write_b32 a102, 0
	v_accvgpr_write_b32 a103, 0
	v_accvgpr_write_b32 a104, 0
	v_accvgpr_write_b32 a105, 0
	v_accvgpr_write_b32 a106, 0
	v_accvgpr_write_b32 a107, 0
	v_accvgpr_write_b32 a108, 0
	v_accvgpr_write_b32 a109, 0
	v_accvgpr_write_b32 a110, 0
	v_accvgpr_write_b32 a111, 0
	v_accvgpr_write_b32 a80, 0
	v_accvgpr_write_b32 a81, 0
	v_accvgpr_write_b32 a82, 0
	v_accvgpr_write_b32 a83, 0
	v_accvgpr_write_b32 a84, 0
	v_accvgpr_write_b32 a85, 0
	v_accvgpr_write_b32 a86, 0
	v_accvgpr_write_b32 a87, 0
	v_accvgpr_write_b32 a88, 0
	v_accvgpr_write_b32 a89, 0
	v_accvgpr_write_b32 a90, 0
	v_accvgpr_write_b32 a91, 0
	v_accvgpr_write_b32 a92, 0
	v_accvgpr_write_b32 a93, 0
	v_accvgpr_write_b32 a94, 0
	v_accvgpr_write_b32 a95, 0
	v_accvgpr_write_b32 a64, 0
	v_accvgpr_write_b32 a65, 0
	v_accvgpr_write_b32 a66, 0
	v_accvgpr_write_b32 a67, 0
	v_accvgpr_write_b32 a68, 0
	v_accvgpr_write_b32 a69, 0
	v_accvgpr_write_b32 a70, 0
	v_accvgpr_write_b32 a71, 0
	v_accvgpr_write_b32 a72, 0
	v_accvgpr_write_b32 a73, 0
	v_accvgpr_write_b32 a74, 0
	v_accvgpr_write_b32 a75, 0
	v_accvgpr_write_b32 a76, 0
	v_accvgpr_write_b32 a77, 0
	v_accvgpr_write_b32 a78, 0
	v_accvgpr_write_b32 a79, 0
	v_accvgpr_write_b32 a48, 0
	v_accvgpr_write_b32 a49, 0
	v_accvgpr_write_b32 a50, 0
	v_accvgpr_write_b32 a51, 0
	v_accvgpr_write_b32 a52, 0
	v_accvgpr_write_b32 a53, 0
	v_accvgpr_write_b32 a54, 0
	v_accvgpr_write_b32 a55, 0
	v_accvgpr_write_b32 a56, 0
	v_accvgpr_write_b32 a57, 0
	v_accvgpr_write_b32 a58, 0
	v_accvgpr_write_b32 a59, 0
	v_accvgpr_write_b32 a60, 0
	v_accvgpr_write_b32 a61, 0
	v_accvgpr_write_b32 a62, 0
	v_accvgpr_write_b32 a63, 0
	v_accvgpr_write_b32 a32, 0
	v_accvgpr_write_b32 a33, 0
	v_accvgpr_write_b32 a34, 0
	v_accvgpr_write_b32 a35, 0
	v_accvgpr_write_b32 a36, 0
	v_accvgpr_write_b32 a37, 0
	v_accvgpr_write_b32 a38, 0
	v_accvgpr_write_b32 a39, 0
	v_accvgpr_write_b32 a40, 0
	v_accvgpr_write_b32 a41, 0
	v_accvgpr_write_b32 a42, 0
	v_accvgpr_write_b32 a43, 0
	v_accvgpr_write_b32 a44, 0
	v_accvgpr_write_b32 a45, 0
	v_accvgpr_write_b32 a46, 0
	v_accvgpr_write_b32 a47, 0
	v_accvgpr_write_b32 a16, 0
	v_accvgpr_write_b32 a17, 0
	v_accvgpr_write_b32 a18, 0
	v_accvgpr_write_b32 a19, 0
	v_accvgpr_write_b32 a20, 0
	v_accvgpr_write_b32 a21, 0
	v_accvgpr_write_b32 a22, 0
	v_accvgpr_write_b32 a23, 0
	v_accvgpr_write_b32 a24, 0
	v_accvgpr_write_b32 a25, 0
	v_accvgpr_write_b32 a26, 0
	v_accvgpr_write_b32 a27, 0
	v_accvgpr_write_b32 a28, 0
	v_accvgpr_write_b32 a29, 0
	v_accvgpr_write_b32 a30, 0
	v_accvgpr_write_b32 a31, 0
	v_accvgpr_write_b32 a0, 0
	v_accvgpr_write_b32 a1, 0
	v_accvgpr_write_b32 a2, 0
	v_accvgpr_write_b32 a3, 0
	v_accvgpr_write_b32 a4, 0
	v_accvgpr_write_b32 a5, 0
	v_accvgpr_write_b32 a6, 0
	v_accvgpr_write_b32 a7, 0
	v_accvgpr_write_b32 a8, 0
	v_accvgpr_write_b32 a9, 0
	v_accvgpr_write_b32 a10, 0
	v_accvgpr_write_b32 a11, 0
	v_accvgpr_write_b32 a12, 0
	v_accvgpr_write_b32 a13, 0
	v_accvgpr_write_b32 a14, 0
	v_accvgpr_write_b32 a15, 0
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	s_barrier
	s_cbranch_scc1 .LBB0_29
; %bb.17:                               ; %.lr.ph
	v_lshlrev_b32_e32 v36, 6, v0
	v_and_b32_e32 v36, 0x3c0, v36
	v_and_b32_e32 v37, 32, v34
	v_bitop3_b32 v74, v36, v37, v46 bitop3:0x36
	v_and_b32_e32 v37, 16, v0
	v_lshlrev_b32_e32 v47, 5, v0
	s_load_dwordx2 s[6:7], s[0:1], 0xc0
	s_load_dwordx4 s[48:51], s[0:1], 0xd0
	s_load_dword s3, s[0:1], 0xe0
	v_and_b32_e32 v46, 0x200, v38
	v_and_b32_e32 v47, 0x1e0, v47
	v_and_or_b32 v34, v34, 12, v37
	v_or3_b32 v46, v46, v37, v47
	v_and_b32_e32 v37, 0x2c0, v38
	v_lshlrev_b32_e32 v34, 1, v34
	v_mul_u32_u24_e32 v36, 0x1800, v42
	v_bitop3_b32 v77, v34, v44, v37 bitop3:0x36
	v_lshlrev_b32_e32 v34, 10, v42
	v_and_b32_e32 v38, 24, v39
	v_lshlrev_b32_e32 v44, 1, v0
	s_sub_i32 s31, 16, s33
	v_lshlrev_b32_e32 v36, 1, v36
	v_add_u32_e32 v37, s54, v34
	v_or_b32_e32 v39, v47, v38
	v_and_b32_e32 v44, 24, v44
	v_add3_u32 v75, s53, v36, v46
	v_and_b32_e32 v36, 12, v45
	v_xad_u32 v78, v39, v44, v37
	v_and_b32_e32 v37, 0x1f8, v40
	s_abs_i32 s59, s31
	v_xad_u32 v79, v37, v38, s54
	v_add3_u32 v80, s53, v34, v37
	s_waitcnt lgkmcnt(0)
	v_mul_lo_u32 v34, v43, s3
	v_or_b32_e32 v38, v41, v36
	v_cvt_f32_u32_e32 v40, s59
	v_add_lshl_u32 v81, v38, v34, 1
	v_or_b32_e32 v38, 4, v42
	v_lshlrev_b32_e32 v39, 10, v38
	v_add3_u32 v85, s53, v39, v37
	v_lshl_or_b32 v37, v38, 5, v36
	v_add_lshl_u32 v86, v37, v34, 1
	v_rcp_iflag_f32_e32 v34, v40
	s_lshl_b32 s24, s31, 3
	s_mov_b32 s0, 0xffff
	s_add_i32 s35, s24, -1
	v_mul_f32_e32 v34, 0x4f7ffffe, v34
	v_cvt_u32_f32_e32 v34, v34
	s_max_i32 s72, s24, 1
	s_sub_i32 s24, 0, s59
	v_accvgpr_write_b32 a0, 0
	v_readfirstlane_b32 s28, v34
	v_lshlrev_b32_e32 v34, 2, v36
	v_bfi_b32 v2, s0, v2, v2
	v_bfi_b32 v4, s0, v4, v4
	v_bfi_b32 v3, s0, v3, v3
	v_bfi_b32 v5, s0, v5, v5
	v_bfi_b32 v6, s0, v6, v6
	v_bfi_b32 v8, s0, v8, v8
	v_bfi_b32 v7, s0, v7, v7
	v_bfi_b32 v9, s0, v9, v9
	v_bfi_b32 v10, s0, v10, v10
	v_bfi_b32 v12, s0, v12, v12
	v_bfi_b32 v11, s0, v11, v11
	v_bfi_b32 v13, s0, v13, v13
	v_bfi_b32 v14, s0, v14, v14
	v_bfi_b32 v16, s0, v16, v16
	v_bfi_b32 v15, s0, v15, v15
	v_bfi_b32 v17, s0, v17, v17
	v_bfi_b32 v18, s0, v18, v18
	v_bfi_b32 v20, s0, v20, v20
	v_bfi_b32 v19, s0, v19, v19
	v_bfi_b32 v21, s0, v21, v21
	v_bfi_b32 v22, s0, v22, v22
	v_bfi_b32 v24, s0, v24, v24
	v_bfi_b32 v23, s0, v23, v23
	v_bfi_b32 v25, s0, v25, v25
	v_bfi_b32 v26, s0, v26, v26
	v_bfi_b32 v28, s0, v28, v28
	v_bfi_b32 v27, s0, v27, v27
	v_bfi_b32 v29, s0, v29, v29
	v_bfi_b32 v30, s0, v30, v30
	v_bfi_b32 v32, s0, v32, v32
	v_bfi_b32 v31, s0, v31, v31
	v_bfi_b32 v33, s0, v33, v33
	s_mul_i32 s0, s4, s48
	s_mul_i32 s24, s24, s28
	v_lshl_add_u64 v[54:55], s[56:57], 0, v[34:35]
	v_lshl_add_u64 v[56:57], s[60:61], 0, v[34:35]
	v_sub_u32_e32 v34, v36, v43
	v_accvgpr_mov_b32 a15, a0
	s_add_i32 s51, s27, s0
	s_movk_i32 s0, 0x80
	s_mul_hi_u32 s24, s28, s24
	v_sub_u32_e32 v34, v34, v41
	v_accvgpr_mov_b32 a1, a0
	v_accvgpr_mov_b32 a2, a0
	v_accvgpr_mov_b32 a3, a0
	v_accvgpr_mov_b32 a4, a0
	v_accvgpr_mov_b32 a5, a0
	v_accvgpr_mov_b32 a6, a0
	v_accvgpr_mov_b32 a7, a0
	v_accvgpr_mov_b32 a8, a0
	v_accvgpr_mov_b32 a9, a0
	v_accvgpr_mov_b32 a10, a0
	v_accvgpr_mov_b32 a11, a0
	v_accvgpr_mov_b32 a12, a0
	v_accvgpr_mov_b32 a13, a0
	v_accvgpr_mov_b32 a14, a0
	v_accvgpr_mov_b32 a31, a15
	v_accvgpr_mov_b32 a47, a15
	v_accvgpr_mov_b32 a63, a15
	v_accvgpr_mov_b32 a79, a15
	v_accvgpr_mov_b32 a95, a15
	v_accvgpr_mov_b32 a111, a15
	v_accvgpr_mov_b32 a127, a15
	v_accvgpr_mov_b32 a143, a15
	v_accvgpr_mov_b32 a159, a15
	v_add_u32_e32 v76, 32, v58
	s_lshl_b32 s54, s3, 5
	v_add_u32_e32 v82, 4, v81
	v_add_u32_e32 v83, 32, v81
	v_add_u32_e32 v84, 36, v81
	v_cmp_gt_u32_e64 s[0:1], s0, v0
	v_add_u32_e32 v87, 4, v86
	v_add_u32_e32 v88, 32, v86
	v_add_u32_e32 v89, 36, v86
	s_ashr_i32 s73, s31, 31
	s_add_i32 s74, s28, s24
	s_add_i32 s57, s23, 32
	s_add_i32 s61, s29, 32
	s_lshl_b32 s75, s3, 4
	v_subrev_u32_e32 v90, s52, v34
	s_lshl_b32 s76, s31, 6
	s_mov_b32 s24, 0x3dd53b94
	s_mov_b32 s28, 0x3fb8aa3b
	v_mov_b32_e32 v91, 0xff800000
	s_mov_b32 s55, 0x20000
	s_mov_b32 s43, 0x110000
	v_accvgpr_mov_b32 a30, a14
	v_accvgpr_mov_b32 a29, a13
	v_accvgpr_mov_b32 a28, a12
	v_accvgpr_mov_b32 a27, a11
	v_accvgpr_mov_b32 a26, a10
	v_accvgpr_mov_b32 a25, a9
	v_accvgpr_mov_b32 a24, a8
	v_accvgpr_mov_b32 a23, a7
	v_accvgpr_mov_b32 a22, a6
	v_accvgpr_mov_b32 a21, a5
	v_accvgpr_mov_b32 a20, a4
	v_accvgpr_mov_b32 a19, a3
	v_accvgpr_mov_b32 a18, a2
	v_accvgpr_mov_b32 a17, a1
	v_accvgpr_mov_b32 a16, a0
	v_accvgpr_mov_b32 a46, a14
	v_accvgpr_mov_b32 a45, a13
	v_accvgpr_mov_b32 a44, a12
	v_accvgpr_mov_b32 a43, a11
	v_accvgpr_mov_b32 a42, a10
	v_accvgpr_mov_b32 a41, a9
	v_accvgpr_mov_b32 a40, a8
	v_accvgpr_mov_b32 a39, a7
	v_accvgpr_mov_b32 a38, a6
	v_accvgpr_mov_b32 a37, a5
	v_accvgpr_mov_b32 a36, a4
	v_accvgpr_mov_b32 a35, a3
	v_accvgpr_mov_b32 a34, a2
	v_accvgpr_mov_b32 a33, a1
	v_accvgpr_mov_b32 a32, a0
	v_accvgpr_mov_b32 a62, a14
	v_accvgpr_mov_b32 a61, a13
	v_accvgpr_mov_b32 a60, a12
	v_accvgpr_mov_b32 a59, a11
	v_accvgpr_mov_b32 a58, a10
	v_accvgpr_mov_b32 a57, a9
	v_accvgpr_mov_b32 a56, a8
	v_accvgpr_mov_b32 a55, a7
	v_accvgpr_mov_b32 a54, a6
	v_accvgpr_mov_b32 a53, a5
	v_accvgpr_mov_b32 a52, a4
	v_accvgpr_mov_b32 a51, a3
	v_accvgpr_mov_b32 a50, a2
	v_accvgpr_mov_b32 a49, a1
	v_accvgpr_mov_b32 a48, a0
	v_accvgpr_mov_b32 a78, a14
	v_accvgpr_mov_b32 a77, a13
	v_accvgpr_mov_b32 a76, a12
	v_accvgpr_mov_b32 a75, a11
	v_accvgpr_mov_b32 a74, a10
	v_accvgpr_mov_b32 a73, a9
	v_accvgpr_mov_b32 a72, a8
	v_accvgpr_mov_b32 a71, a7
	v_accvgpr_mov_b32 a70, a6
	v_accvgpr_mov_b32 a69, a5
	v_accvgpr_mov_b32 a68, a4
	v_accvgpr_mov_b32 a67, a3
	v_accvgpr_mov_b32 a66, a2
	v_accvgpr_mov_b32 a65, a1
	v_accvgpr_mov_b32 a64, a0
	v_accvgpr_mov_b32 a94, a14
	v_accvgpr_mov_b32 a93, a13
	v_accvgpr_mov_b32 a92, a12
	v_accvgpr_mov_b32 a91, a11
	v_accvgpr_mov_b32 a90, a10
	v_accvgpr_mov_b32 a89, a9
	v_accvgpr_mov_b32 a88, a8
	v_accvgpr_mov_b32 a87, a7
	v_accvgpr_mov_b32 a86, a6
	v_accvgpr_mov_b32 a85, a5
	v_accvgpr_mov_b32 a84, a4
	v_accvgpr_mov_b32 a83, a3
	v_accvgpr_mov_b32 a82, a2
	v_accvgpr_mov_b32 a81, a1
	v_accvgpr_mov_b32 a80, a0
	v_accvgpr_mov_b32 a110, a14
	v_accvgpr_mov_b32 a109, a13
	v_accvgpr_mov_b32 a108, a12
	v_accvgpr_mov_b32 a107, a11
	v_accvgpr_mov_b32 a106, a10
	v_accvgpr_mov_b32 a105, a9
	v_accvgpr_mov_b32 a104, a8
	v_accvgpr_mov_b32 a103, a7
	v_accvgpr_mov_b32 a102, a6
	v_accvgpr_mov_b32 a101, a5
	v_accvgpr_mov_b32 a100, a4
	v_accvgpr_mov_b32 a99, a3
	v_accvgpr_mov_b32 a98, a2
	v_accvgpr_mov_b32 a97, a1
	v_accvgpr_mov_b32 a96, a0
	v_accvgpr_mov_b32 a126, a14
	v_accvgpr_mov_b32 a125, a13
	v_accvgpr_mov_b32 a124, a12
	v_accvgpr_mov_b32 a123, a11
	v_accvgpr_mov_b32 a122, a10
	v_accvgpr_mov_b32 a121, a9
	v_accvgpr_mov_b32 a120, a8
	v_accvgpr_mov_b32 a119, a7
	v_accvgpr_mov_b32 a118, a6
	v_accvgpr_mov_b32 a117, a5
	v_accvgpr_mov_b32 a116, a4
	v_accvgpr_mov_b32 a115, a3
	v_accvgpr_mov_b32 a114, a2
	v_accvgpr_mov_b32 a113, a1
	v_accvgpr_mov_b32 a112, a0
	v_accvgpr_mov_b32 a142, a14
	v_accvgpr_mov_b32 a141, a13
	v_accvgpr_mov_b32 a140, a12
	v_accvgpr_mov_b32 a139, a11
	v_accvgpr_mov_b32 a138, a10
	v_accvgpr_mov_b32 a137, a9
	v_accvgpr_mov_b32 a136, a8
	v_accvgpr_mov_b32 a135, a7
	v_accvgpr_mov_b32 a134, a6
	v_accvgpr_mov_b32 a133, a5
	v_accvgpr_mov_b32 a132, a4
	v_accvgpr_mov_b32 a131, a3
	v_accvgpr_mov_b32 a130, a2
	v_accvgpr_mov_b32 a129, a1
	v_accvgpr_mov_b32 a128, a0
	v_accvgpr_mov_b32 a158, a14
	v_accvgpr_mov_b32 a157, a13
	v_accvgpr_mov_b32 a156, a12
	v_accvgpr_mov_b32 a155, a11
	v_accvgpr_mov_b32 a154, a10
	v_accvgpr_mov_b32 a153, a9
	v_accvgpr_mov_b32 a152, a8
	v_accvgpr_mov_b32 a151, a7
	v_accvgpr_mov_b32 a150, a6
	v_accvgpr_mov_b32 a149, a5
	v_accvgpr_mov_b32 a148, a4
	v_accvgpr_mov_b32 a147, a3
	v_accvgpr_mov_b32 a146, a2
	v_accvgpr_mov_b32 a145, a1
	v_accvgpr_mov_b32 a144, a0
	s_branch .LBB0_19
.LBB0_18:                               ; %._crit_edge2406
                                        ;   in Loop: Header=BB0_19 Depth=1
	s_cmp_eq_u32 s77, s72
	s_cbranch_scc1 .LBB0_29
.LBB0_19:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_21 Depth 2
	s_abs_i32 s37, s77
	s_mul_hi_u32 s40, s37, s74
	s_mul_i32 s41, s40, s59
	s_ashr_i32 s36, s77, 31
	s_sub_i32 s37, s37, s41
	s_xor_b32 s36, s36, s73
	s_add_i32 s41, s40, 1
	s_sub_i32 s44, s37, s59
	s_cmp_ge_u32 s37, s59
	s_cselect_b32 s40, s41, s40
	s_cselect_b32 s37, s44, s37
	s_add_i32 s41, s40, 1
	s_cmp_ge_u32 s37, s59
	s_cselect_b32 s37, s41, s40
	s_xor_b32 s37, s37, s36
	s_sub_i32 s37, s37, s36
	s_mul_i32 s36, s37, s31
	s_sub_i32 s36, s77, s36
	s_add_i32 s36, s36, s33
	s_lshl_b32 s40, s36, 6
	s_add_i32 s36, s37, s51
	s_mul_i32 s36, s36, s50
	s_add_i32 s36, s36, s40
	v_add_u32_e32 v92, s40, v90
	s_lshl_b32 s40, s77, 6
	s_add_i32 s40, s11, s40
	s_mul_i32 s37, s76, s37
	s_mov_b32 s39, s77
	s_mul_i32 s36, s3, s36
	s_sub_i32 s47, s40, s37
	s_mov_b64 s[40:41], 0
	s_mov_b32 s77, 0
	s_mov_b32 s78, 0
	s_branch .LBB0_21
.LBB0_20:                               ;   in Loop: Header=BB0_21 Depth=2
	s_or_b64 exec, exec, s[44:45]
	s_barrier
	v_add_u32_e32 v42, s79, v77
	;;#ASMSTART
	ds_read_b64_tr_b16 v[38:39], v42 offset:0
ds_read_b64_tr_b16 v[40:41], v42 offset:0x100

	;;#ASMEND
	v_permlane16_swap_b32_e64 v34, v36 bound_ctrl:1
	v_permlane16_swap_b32_e64 v35, v37 bound_ctrl:1
	s_add_i32 s78, s78, 1
	s_nop 0
	v_mfma_f32_32x32x16_bf16 a[144:159], v[38:41], v[34:37], a[144:159]
	;;#ASMSTART
	ds_read_b64_tr_b16 v[38:39], v42 offset:0x400
ds_read_b64_tr_b16 v[40:41], v42 offset:0x500

	;;#ASMEND
	s_add_u32 s40, s40, 64
	s_addc_u32 s41, s41, 0
	s_add_i32 s36, s36, s75
	s_add_i32 s77, s77, 16
	s_cmpk_eq_i32 s40, 0x100
	v_mfma_f32_32x32x16_bf16 a[128:143], v[38:41], v[34:37], a[128:143]
	;;#ASMSTART
	ds_read_b64_tr_b16 v[38:39], v42 offset:0x800
ds_read_b64_tr_b16 v[40:41], v42 offset:0x900

	;;#ASMEND
	s_nop 0
	v_mfma_f32_32x32x16_bf16 a[112:127], v[38:41], v[34:37], a[112:127]
	;;#ASMSTART
	ds_read_b64_tr_b16 v[38:39], v42 offset:0xc00
ds_read_b64_tr_b16 v[40:41], v42 offset:0xd00

	;;#ASMEND
	s_nop 0
	v_mfma_f32_32x32x16_bf16 a[96:111], v[38:41], v[34:37], a[96:111]
	;;#ASMSTART
	ds_read_b64_tr_b16 v[38:39], v42 offset:0x1000
ds_read_b64_tr_b16 v[40:41], v42 offset:0x1100

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[42:43], v42 offset:0x1400
ds_read_b64_tr_b16 v[44:45], v42 offset:0x1500

	;;#ASMEND
	s_barrier
	v_mfma_f32_32x32x16_bf16 a[80:95], v[38:41], v[34:37], a[80:95]
	v_mfma_f32_32x32x16_bf16 a[64:79], v[42:45], v[34:37], a[64:79]
	s_cbranch_scc1 .LBB0_27
.LBB0_21:                               ;   Parent Loop BB0_19 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_lshr_b32 s52, s78, 1
	s_and_b32 s37, s78, 1
	s_mul_i32 s44, s52, 0x3000
	s_add_i32 s79, s70, s44
	s_mul_i32 s44, s37, 0x1800
	s_add_i32 s79, s79, s44
	v_add_u32_e32 v93, s79, v74
	;;#ASMSTART
	ds_read_b128 v[34:37], v93 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[38:41], v93 offset:0x400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[42:45], v93 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[46:49], v93 offset:0xc00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[50:53], v93 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[94:97], v93 offset:0x1400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[98:101], v75 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[102:105], v75 offset:0x400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[106:109], v75 offset:0x800

	;;#ASMEND
	s_add_i32 s48, s47, s77
	v_mfma_f32_16x16x32_bf16 a[160:163], v[34:37], v[98:101], 0
	;;#ASMSTART
	ds_read_b128 v[98:101], v75 offset:0xc00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[110:113], v75 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[114:117], v75 offset:0x1400

	;;#ASMEND
	v_mfma_f32_16x16x32_bf16 a[160:163], v[38:41], v[102:105], a[160:163]
	;;#ASMSTART
	ds_read_b128 v[102:105], v75 offset:0x1800

	;;#ASMEND
	s_add_i32 s44, s48, 16
	v_cmp_gt_i32_e32 vcc, s44, v58
	v_mfma_f32_16x16x32_bf16 a[164:167], v[34:37], v[102:105], 0
	;;#ASMSTART
	ds_read_b128 v[34:37], v75 offset:0x1c00

	;;#ASMEND
	s_nop 0
	v_mfma_f32_16x16x32_bf16 a[164:167], v[38:41], v[34:37], a[164:167]
	;;#ASMSTART
	ds_read_b128 v[34:37], v75 offset:0x2000

	;;#ASMEND
	v_mov_b32_e32 v38, 0xff800000
	v_mov_b32_e32 v39, 0xff800000
	v_mfma_f32_16x16x32_bf16 a[160:163], v[42:45], v[106:109], a[160:163]
	v_mov_b32_e32 v40, 0xff800000
	v_mov_b32_e32 v41, 0xff800000
	v_mfma_f32_16x16x32_bf16 a[164:167], v[42:45], v[34:37], a[164:167]
	;;#ASMSTART
	ds_read_b128 v[34:37], v75 offset:0x2400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[42:45], v75 offset:0x2800

	;;#ASMEND
	v_mfma_f32_16x16x32_bf16 a[160:163], v[46:49], v[98:101], a[160:163]
	;;#ASMSTART
	ds_read_b128 v[98:101], v75 offset:0x2c00

	;;#ASMEND
	v_mfma_f32_16x16x32_bf16 a[164:167], v[46:49], v[34:37], a[164:167]
	v_mov_b32_e32 v34, 0xff800000
	v_mov_b32_e32 v35, 0xff800000
	v_mov_b32_e32 v36, 0xff800000
	v_mfma_f32_16x16x32_bf16 a[160:163], v[50:53], v[110:113], a[160:163]
	v_mov_b32_e32 v37, 0xff800000
	v_mfma_f32_16x16x32_bf16 a[164:167], v[50:53], v[42:45], a[164:167]
	v_mfma_f32_16x16x32_bf16 a[160:163], v[94:97], v[114:117], a[160:163]
	v_mfma_f32_16x16x32_bf16 a[164:167], v[94:97], v[98:101], a[164:167]
	s_and_saveexec_b64 s[44:45], vcc
	s_cbranch_execz .LBB0_25
; %bb.22:                               ;   in Loop: Header=BB0_21 Depth=2
	v_lshl_add_u64 v[34:35], v[54:55], 0, s[40:41]
	flat_load_dwordx4 v[34:37], v[34:35]
	s_nop 2
	v_accvgpr_read_b32 v38, a160
	v_accvgpr_read_b32 v42, a164
	v_accvgpr_read_b32 v39, a161
	v_accvgpr_read_b32 v40, a162
	v_accvgpr_read_b32 v41, a163
	v_accvgpr_read_b32 v43, a165
	v_accvgpr_read_b32 v44, a166
	v_accvgpr_read_b32 v45, a167
	v_cmp_lt_i32_e32 vcc, s48, v76
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_pk_mul_f32 v[46:47], v[34:35], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[36:37], s[28:29] op_sel_hi:[1,0]
	v_pk_fma_f32 v[34:35], v[38:39], s[24:25], v[46:47] op_sel_hi:[1,0,1] neg_lo:[0,0,1] neg_hi:[0,0,1]
	v_pk_fma_f32 v[36:37], v[40:41], s[24:25], v[48:49] op_sel_hi:[1,0,1] neg_lo:[0,0,1] neg_hi:[0,0,1]
	v_fma_f32 v38, v42, s24, -v46
	v_fma_f32 v39, v43, s24, -v47
	v_fma_f32 v40, v44, s24, -v48
	v_fma_f32 v41, v45, s24, -v49
	s_and_saveexec_b64 s[48:49], vcc
	s_cbranch_execz .LBB0_24
; %bb.23:                               ;   in Loop: Header=BB0_21 Depth=2
	v_add_u32_e32 v42, s77, v92
	v_add_u32_e32 v43, 3, v42
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[80:81], v43, 3
	v_cmp_lt_i32_e64 s[82:83], v43, 2
	v_cndmask_b32_e64 v34, v34, v91, s[80:81]
	v_cndmask_b32_e64 v35, v35, v91, s[82:83]
	
	;;#ASMEND
	v_add_u32_e32 v42, -13, v42
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[80:81], v43, 1
	v_cmp_lt_i32_e64 s[82:83], v43, 0
	v_cndmask_b32_e64 v36, v36, v91, s[80:81]
	v_cndmask_b32_e64 v37, v37, v91, s[82:83]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[80:81], v42, 3
	v_cmp_lt_i32_e64 s[82:83], v42, 2
	v_cndmask_b32_e64 v38, v38, v91, s[80:81]
	v_cndmask_b32_e64 v39, v39, v91, s[82:83]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[80:81], v42, 1
	v_cmp_lt_i32_e64 s[82:83], v42, 0
	v_cndmask_b32_e64 v40, v40, v91, s[80:81]
	v_cndmask_b32_e64 v41, v41, v91, s[82:83]
	
	;;#ASMEND
.LBB0_24:                               ; %Flow
                                        ;   in Loop: Header=BB0_21 Depth=2
	s_or_b64 exec, exec, s[48:49]
.LBB0_25:                               ; %.preheader2071
                                        ;   in Loop: Header=BB0_21 Depth=2
	s_or_b64 exec, exec, s[44:45]
	s_lshl_b32 s44, s52, 13
	v_min_f32_e32 v34, 0, v34
	v_min_f32_e32 v35, 0, v35
	v_min_f32_e32 v36, 0, v36
	v_min_f32_e32 v37, 0, v37
	v_min_f32_e32 v38, 0, v38
	s_add_i32 s44, s71, s44
	s_lshl_b32 s37, s37, 12
	v_min_f32_e32 v39, 0, v39
	v_min_f32_e32 v40, 0, v40
	v_min_f32_e32 v41, 0, v41
	v_exp_f32_e32 v34, v34
	v_exp_f32_e32 v35, v35
	v_exp_f32_e32 v36, v36
	v_exp_f32_e32 v37, v37
	v_exp_f32_e32 v38, v38
	s_add_i32 s44, s44, s37
	v_exp_f32_e32 v42, v39
	v_exp_f32_e32 v40, v40
	v_exp_f32_e32 v41, v41
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v52, v34, v35
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v39, v36, v37
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v53, v38, v42
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v93, v40, v41
	;;#ASMEND
	v_add_u32_e32 v38, s44, v74
	;;#ASMSTART
	ds_read_b128 v[34:37], v38 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[40:43], v38 offset:0x400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[44:47], v38 offset:0x800

	;;#ASMEND
	s_ashr_i32 s37, s36, 31
	v_mfma_f32_16x16x32_bf16 a[160:163], v[34:37], v[2:5], 0
	v_mfma_f32_16x16x32_bf16 a[164:167], v[34:37], v[18:21], 0
	;;#ASMSTART
	ds_read_b128 v[34:37], v38 offset:0xc00

	;;#ASMEND
	v_mov_b32_e32 v38, v52
	v_mfma_f32_16x16x32_bf16 a[160:163], v[40:43], v[6:9], a[160:163]
	v_mfma_f32_16x16x32_bf16 a[164:167], v[40:43], v[22:25], a[164:167]
	v_lshl_add_u64 v[40:41], v[56:57], 0, s[40:41]
	flat_load_dwordx4 v[48:51], v[40:41]
	v_mov_b32_e32 v40, v53
	v_mfma_f32_16x16x32_bf16 a[160:163], v[44:47], v[10:13], a[160:163]
	s_nop 0
	v_permlane16_swap_b32_e64 v38, v40 bound_ctrl:1
	v_mfma_f32_16x16x32_bf16 a[164:167], v[44:47], v[26:29], a[164:167]
	v_add_u32_e32 v46, s44, v77
	s_lshl_b64 s[44:45], s[36:37], 1
	s_add_u32 s52, s6, s44
	v_mfma_f32_16x16x32_bf16 a[160:163], v[34:37], v[14:17], a[160:163]
	s_addc_u32 s53, s7, s45
	v_mfma_f32_16x16x32_bf16 a[164:167], v[34:37], v[30:33], a[164:167]
	s_nop 5
	v_accvgpr_read_b32 v34, a160
	v_accvgpr_read_b32 v35, a161
	v_accvgpr_read_b32 v36, a162
	v_accvgpr_read_b32 v37, a163
	v_accvgpr_read_b32 v41, a164
	v_accvgpr_read_b32 v42, a165
	v_accvgpr_read_b32 v43, a166
	v_accvgpr_read_b32 v44, a167
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_sub_f32_e32 v34, v34, v48
	v_sub_f32_e32 v35, v35, v49
	v_sub_f32_e32 v36, v36, v50
	v_sub_f32_e32 v37, v37, v51
	v_sub_f32_e32 v41, v41, v48
	v_sub_f32_e32 v42, v42, v49
	v_sub_f32_e32 v43, v43, v50
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v34, v34, v35
	;;#ASMEND
	v_sub_f32_e32 v44, v44, v51
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v35, v36, v37
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v36, v41, v42
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v37, v43, v44
	;;#ASMEND
	;;#ASMSTART
	v_cvt_f32_bf16_e32 v41, v52
	;;#ASMEND
	;;#ASMSTART
	v_cvt_f32_bf16_e32 v42, v34
	;;#ASMEND
	;;#ASMSTART
	v_lshrrev_b32_e32 v43, 16, v52
	;;#ASMEND
	;;#ASMSTART
	v_lshrrev_b32_e32 v34, 16, v34
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cvt_f32_bf16_e32 v43, v43
	;;#ASMEND
	;;#ASMSTART
	v_cvt_f32_bf16_e32 v34, v34
	;;#ASMEND
	v_mul_f32_e32 v41, v42, v41
	v_mul_f32_e32 v34, v34, v43
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v34, v34, v41
	;;#ASMEND
	;;#ASMSTART
	v_cvt_f32_bf16_e32 v41, v39
	;;#ASMEND
	;;#ASMSTART
	v_cvt_f32_bf16_e32 v42, v35
	;;#ASMEND
	;;#ASMSTART
	v_lshrrev_b32_e32 v43, 16, v39
	;;#ASMEND
	;;#ASMSTART
	v_lshrrev_b32_e32 v35, 16, v35
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cvt_f32_bf16_e32 v43, v43
	;;#ASMEND
	;;#ASMSTART
	v_cvt_f32_bf16_e32 v35, v35
	;;#ASMEND
	v_mul_f32_e32 v41, v42, v41
	v_mul_f32_e32 v35, v35, v43
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v35, v35, v41
	;;#ASMEND
	;;#ASMSTART
	v_cvt_f32_bf16_e32 v41, v53
	;;#ASMEND
	;;#ASMSTART
	v_cvt_f32_bf16_e32 v42, v36
	;;#ASMEND
	;;#ASMSTART
	v_lshrrev_b32_e32 v43, 16, v53
	;;#ASMEND
	;;#ASMSTART
	v_lshrrev_b32_e32 v36, 16, v36
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cvt_f32_bf16_e32 v43, v43
	;;#ASMEND
	;;#ASMSTART
	v_cvt_f32_bf16_e32 v36, v36
	;;#ASMEND
	v_mul_f32_e32 v41, v42, v41
	v_mul_f32_e32 v36, v36, v43
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v36, v36, v41
	;;#ASMEND
	;;#ASMSTART
	v_cvt_f32_bf16_e32 v41, v93
	;;#ASMEND
	;;#ASMSTART
	v_cvt_f32_bf16_e32 v42, v37
	;;#ASMEND
	;;#ASMSTART
	v_lshrrev_b32_e32 v43, 16, v93
	;;#ASMEND
	;;#ASMSTART
	v_lshrrev_b32_e32 v37, 16, v37
	;;#ASMEND
	v_permlane16_swap_b32_e64 v39, v93 bound_ctrl:1
	;;#ASMSTART
	v_cvt_f32_bf16_e32 v43, v43
	;;#ASMEND
	;;#ASMSTART
	v_cvt_f32_bf16_e32 v37, v37
	;;#ASMEND
	v_mul_f32_e32 v42, v42, v41
	v_mul_f32_e32 v37, v37, v43
	v_mov_b32_e32 v41, v93
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v37, v37, v42
	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[42:43], v46 offset:0
ds_read_b64_tr_b16 v[44:45], v46 offset:0x100

	;;#ASMEND
                                        ; kill: def $vgpr93 killed $vgpr93
	s_nop 1
	v_mfma_f32_32x32x16_bf16 a[48:63], v[42:45], v[38:41], a[48:63]
	;;#ASMSTART
	ds_read_b64_tr_b16 v[42:43], v46 offset:0x400
ds_read_b64_tr_b16 v[44:45], v46 offset:0x500

	;;#ASMEND
	s_nop 0
	v_mfma_f32_32x32x16_bf16 a[32:47], v[42:45], v[38:41], a[32:47]
	;;#ASMSTART
	ds_read_b64_tr_b16 v[42:43], v46 offset:0x800
ds_read_b64_tr_b16 v[44:45], v46 offset:0x900

	;;#ASMEND
	s_nop 0
	v_mfma_f32_32x32x16_bf16 a[16:31], v[42:45], v[38:41], a[16:31]
	;;#ASMSTART
	ds_read_b64_tr_b16 v[42:43], v46 offset:0xc00
ds_read_b64_tr_b16 v[44:45], v46 offset:0xd00

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v78, v[34:35] offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_write_b64 v78, v[36:37] offset:0x200

	;;#ASMEND
	s_barrier
	;;#ASMSTART
	ds_read_b64_tr_b16 v[50:51], v79 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[46:47], v79 offset:0x400

	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 a[0:15], v[42:45], v[38:41], a[0:15]
	;;#ASMSTART
	ds_read_b64_tr_b16 v[42:43], v79 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[38:39], v79 offset:0xc00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[52:53], v79 offset:0x200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[48:49], v79 offset:0x600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[44:45], v79 offset:0xa00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[40:41], v79 offset:0xe00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[94:95], v80 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[98:99], v80 offset:0x200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[102:103], v80 offset:0x3000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[106:107], v80 offset:0x3200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[110:111], v80 offset:0x6000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[114:115], v80 offset:0x6200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[118:119], v80 offset:0x9000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[122:123], v80 offset:0x9200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[96:97], v80 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[100:101], v80 offset:0x1a00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[104:105], v80 offset:0x4800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[108:109], v80 offset:0x4a00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[112:113], v80 offset:0x7800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[116:117], v80 offset:0x7a00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[120:121], v80 offset:0xa800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[124:125], v80 offset:0xaa00

	;;#ASMEND
	s_nop 0
	v_mfma_f32_16x16x32_bf16 a[160:163], v[94:97], v[50:53], 0
	v_mfma_f32_16x16x32_bf16 a[164:167], v[98:101], v[50:53], 0
	v_mfma_f32_16x16x32_bf16 a[160:163], v[102:105], v[46:49], a[160:163]
	v_mfma_f32_16x16x32_bf16 a[164:167], v[106:109], v[46:49], a[164:167]
	v_mfma_f32_16x16x32_bf16 a[160:163], v[110:113], v[42:45], a[160:163]
	v_mfma_f32_16x16x32_bf16 a[164:167], v[114:117], v[42:45], a[164:167]
	v_mfma_f32_16x16x32_bf16 a[160:163], v[118:121], v[38:41], a[160:163]
	v_mfma_f32_16x16x32_bf16 a[164:167], v[122:125], v[38:41], a[164:167]
	s_nop 6
	v_accvgpr_read_b32 v93, a160
	v_accvgpr_read_b32 v94, a161
	v_mul_f32_e32 v93, 0x3d93cd3a, v93
	v_mul_f32_e32 v94, 0x3d93cd3a, v94
	v_accvgpr_read_b32 v95, a162
	v_accvgpr_read_b32 v96, a163
	v_mul_f32_e32 v95, 0x3d93cd3a, v95
	v_mul_f32_e32 v96, 0x3d93cd3a, v96
	v_accvgpr_read_b32 v97, a164
	v_accvgpr_read_b32 v98, a165
	v_accvgpr_read_b32 v99, a166
	v_accvgpr_read_b32 v100, a167
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v93, v94, v93
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v94, v96, v95
	;;#ASMEND
	v_mul_f32_e32 v97, 0x3d93cd3a, v97
	;;#ASMSTART
	buffer_atomic_pk_add_bf16 v93, v81, s[52:55], 0 offen
	;;#ASMEND
	v_mul_f32_e32 v98, 0x3d93cd3a, v98
	v_mul_f32_e32 v99, 0x3d93cd3a, v99
	v_mul_f32_e32 v100, 0x3d93cd3a, v100
	;;#ASMSTART
	buffer_atomic_pk_add_bf16 v94, v82, s[52:55], 0 offen
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v93, v98, v97
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v94, v100, v99
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	buffer_atomic_pk_add_bf16 v93, v83, s[52:55], 0 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_atomic_pk_add_bf16 v94, v84, s[52:55], 0 offen
	;;#ASMEND
	s_and_saveexec_b64 s[44:45], s[0:1]
	s_cbranch_execz .LBB0_20
; %bb.26:                               ;   in Loop: Header=BB0_21 Depth=2
	;;#ASMSTART
	ds_read_b64_tr_b16 v[94:95], v85 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[98:99], v85 offset:0x200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[102:103], v85 offset:0x3000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[106:107], v85 offset:0x3200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[110:111], v85 offset:0x6000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[114:115], v85 offset:0x6200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[118:119], v85 offset:0x9000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[122:123], v85 offset:0x9200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[96:97], v85 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[100:101], v85 offset:0x1a00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[104:105], v85 offset:0x4800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[108:109], v85 offset:0x4a00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[112:113], v85 offset:0x7800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[116:117], v85 offset:0x7a00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[120:121], v85 offset:0xa800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[124:125], v85 offset:0xaa00

	;;#ASMEND
	s_nop 0
	v_mfma_f32_16x16x32_bf16 a[160:163], v[94:97], v[50:53], 0
	v_mfma_f32_16x16x32_bf16 a[164:167], v[98:101], v[50:53], 0
	v_mfma_f32_16x16x32_bf16 a[160:163], v[102:105], v[46:49], a[160:163]
	v_mfma_f32_16x16x32_bf16 a[164:167], v[106:109], v[46:49], a[164:167]
	v_mfma_f32_16x16x32_bf16 a[160:163], v[110:113], v[42:45], a[160:163]
	v_mfma_f32_16x16x32_bf16 a[164:167], v[114:117], v[42:45], a[164:167]
	v_mfma_f32_16x16x32_bf16 a[160:163], v[118:121], v[38:41], a[160:163]
	s_nop 7
	v_accvgpr_read_b32 v42, a160
	v_accvgpr_read_b32 v43, a161
	v_accvgpr_read_b32 v44, a162
	v_accvgpr_read_b32 v45, a163
	v_mfma_f32_16x16x32_bf16 a[160:163], v[122:125], v[38:41], a[164:167]
	v_mul_f32_e32 v42, 0x3d93cd3a, v42
	v_mul_f32_e32 v38, 0x3d93cd3a, v45
	v_mul_f32_e32 v43, 0x3d93cd3a, v43
	v_mul_f32_e32 v44, 0x3d93cd3a, v44
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v42, v43, v42
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v38, v38, v44
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	buffer_atomic_pk_add_bf16 v42, v86, s[52:55], 0 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_atomic_pk_add_bf16 v38, v87, s[52:55], 0 offen
	;;#ASMEND
	s_nop 2
	v_accvgpr_read_b32 v39, a160
	v_accvgpr_read_b32 v40, a161
	v_mul_f32_e32 v39, 0x3d93cd3a, v39
	v_accvgpr_read_b32 v41, a162
	v_accvgpr_read_b32 v45, a163
	v_mul_f32_e32 v40, 0x3d93cd3a, v40
	v_mul_f32_e32 v41, 0x3d93cd3a, v41
	v_mul_f32_e32 v45, 0x3d93cd3a, v45
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v38, v40, v39
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v39, v45, v41
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	buffer_atomic_pk_add_bf16 v38, v88, s[52:55], 0 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_atomic_pk_add_bf16 v39, v89, s[52:55], 0 offen
	;;#ASMEND
	s_branch .LBB0_20
.LBB0_27:                               ;   in Loop: Header=BB0_19 Depth=1
	s_add_i32 s77, s39, 1
	s_cmp_lg_u32 s39, s35
	s_cbranch_scc0 .LBB0_18
; %bb.28:                               ;   in Loop: Header=BB0_19 Depth=1
	s_abs_i32 s37, s77
	s_mul_hi_u32 s39, s37, s74
	s_mul_i32 s40, s39, s59
	s_ashr_i32 s36, s77, 31
	s_sub_i32 s37, s37, s40
	s_xor_b32 s36, s36, s73
	s_add_i32 s40, s39, 1
	s_sub_i32 s41, s37, s59
	s_cmp_ge_u32 s37, s59
	s_cselect_b32 s39, s40, s39
	s_cselect_b32 s37, s41, s37
	s_add_i32 s40, s39, 1
	s_cmp_ge_u32 s37, s59
	s_cselect_b32 s37, s40, s39
	s_xor_b32 s37, s37, s36
	s_sub_i32 s36, s37, s36
	s_add_i32 s48, s36, s27
	s_mul_i32 s36, s36, s31
	s_sub_i32 s36, s77, s36
	s_add_i32 s36, s36, s33
	s_lshl_b32 s49, s36, 6
	s_add_i32 s36, s49, s23
	s_mul_i32 s36, s36, s30
	s_add_i32 s36, s36, s48
	s_mul_i32 s36, s36, s5
	s_ashr_i32 s37, s36, 31
	s_lshl_b64 s[36:37], s[36:37], 1
	s_add_u32 s40, s62, s36
	v_readfirstlane_b32 s36, v64
	s_addc_u32 s41, s63, s37
	s_mov_b32 m0, s36
	v_readfirstlane_b32 s36, v65
	buffer_load_dwordx4 v1, s[40:43], 0 offen lds
	s_mov_b32 m0, s36
	v_readfirstlane_b32 s36, v66
	buffer_load_dwordx4 v59, s[40:43], 0 offen lds
	s_mov_b32 m0, s36
	s_add_i32 s36, s49, s29
	s_mul_i32 s36, s36, s26
	s_add_i32 s36, s36, s48
	s_mul_i32 s36, s36, s25
	s_ashr_i32 s37, s36, 31
	s_lshl_b64 s[36:37], s[36:37], 1
	s_add_u32 s44, s64, s36
	v_readfirstlane_b32 s36, v67
	buffer_load_dwordx4 v60, s[40:43], 0 offen lds
	s_addc_u32 s45, s65, s37
	s_mov_b32 s47, s43
	s_mov_b32 m0, s36
	v_readfirstlane_b32 s36, v68
	buffer_load_dwordx4 v61, s[44:47], 0 offen lds
	s_mov_b32 m0, s36
	s_add_i32 s36, s49, s57
	s_mul_i32 s36, s36, s30
	s_add_i32 s36, s36, s48
	s_mul_i32 s36, s36, s5
	s_ashr_i32 s37, s36, 31
	s_lshl_b64 s[36:37], s[36:37], 1
	s_add_u32 s40, s62, s36
	v_readfirstlane_b32 s36, v69
	buffer_load_dwordx4 v62, s[44:47], 0 offen lds
	s_addc_u32 s41, s63, s37
	s_mov_b32 m0, s36
	v_readfirstlane_b32 s36, v70
	buffer_load_dwordx4 v1, s[40:43], 0 offen lds
	s_mov_b32 m0, s36
	v_readfirstlane_b32 s36, v71
	buffer_load_dwordx4 v59, s[40:43], 0 offen lds
	s_mov_b32 m0, s36
	s_add_i32 s36, s49, s61
	s_mul_i32 s36, s36, s26
	s_add_i32 s36, s36, s48
	s_mul_i32 s36, s36, s25
	s_ashr_i32 s37, s36, 31
	s_lshl_b64 s[36:37], s[36:37], 1
	s_add_u32 s44, s64, s36
	v_readfirstlane_b32 s36, v72
	buffer_load_dwordx4 v60, s[40:43], 0 offen lds
	s_addc_u32 s45, s65, s37
	s_mov_b32 m0, s36
	v_readfirstlane_b32 s36, v73
	buffer_load_dwordx4 v61, s[44:47], 0 offen lds
	s_mov_b32 m0, s36
	s_add_i32 s36, s48, s13
	s_mul_i32 s36, s15, s36
	s_add_i32 s36, s36, s49
	s_ashr_i32 s37, s36, 31
	s_lshl_b64 s[36:37], s[36:37], 2
	s_add_u32 s36, s66, s36
	buffer_load_dwordx4 v62, s[44:47], 0 offen lds
	s_addc_u32 s37, s67, s37
	s_mov_b32 s39, s43
	s_mov_b32 m0, s56
	s_add_i32 s48, s48, s19
	buffer_load_dword v63, s[36:39], 0 offen lds
	s_mul_i32 s36, s21, s48
	s_add_i32 s36, s36, s49
	s_ashr_i32 s37, s36, 31
	s_lshl_b64 s[36:37], s[36:37], 2
	s_add_u32 s36, s68, s36
	s_addc_u32 s37, s69, s37
	s_mov_b32 m0, s60
	s_nop 0
	buffer_load_dword v63, s[36:39], 0 offen lds
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	s_barrier
	s_branch .LBB0_18
.LBB0_29:                               ; %._crit_edge
	s_mul_i32 s0, s4, s20
	v_add_u32_e32 v1, s0, v58
	v_mul_lo_u32 v1, v1, s22
	v_add_u32_e32 v1, s2, v1
	s_mul_i32 s0, s22, s58
	s_mul_i32 s1, s18, s20
	v_mul_lo_u32 v4, v1, s58
	v_and_b32_e32 v17, 31, v0
	v_lshrrev_b32_e32 v0, 3, v0
	s_mul_i32 s1, s0, s1
	v_mov_b32_e32 v2, s16
	v_mov_b32_e32 v3, s17
	v_ashrrev_i32_e32 v5, 31, v4
	v_and_b32_e32 v16, 4, v0
	s_lshl_b32 s1, s1, 1
	v_mul_lo_u32 v107, v17, s0
	v_accvgpr_read_b32 v0, a48
	v_accvgpr_read_b32 v1, a49
	v_accvgpr_read_b32 v106, a144
	v_accvgpr_read_b32 v105, a145
	v_accvgpr_read_b32 v104, a146
	v_accvgpr_read_b32 v103, a147
	v_accvgpr_read_b32 v102, a148
	v_accvgpr_read_b32 v101, a149
	v_accvgpr_read_b32 v100, a150
	v_accvgpr_read_b32 v99, a151
	v_accvgpr_read_b32 v98, a152
	v_accvgpr_read_b32 v97, a153
	v_accvgpr_read_b32 v96, a154
	v_accvgpr_read_b32 v95, a155
	v_accvgpr_read_b32 v94, a156
	v_accvgpr_read_b32 v93, a157
	v_accvgpr_read_b32 v92, a158
	v_accvgpr_read_b32 v91, a159
	v_accvgpr_read_b32 v90, a128
	v_accvgpr_read_b32 v89, a129
	v_accvgpr_read_b32 v88, a130
	v_accvgpr_read_b32 v87, a131
	v_accvgpr_read_b32 v86, a132
	v_accvgpr_read_b32 v85, a133
	v_accvgpr_read_b32 v84, a134
	v_accvgpr_read_b32 v83, a135
	v_accvgpr_read_b32 v82, a136
	v_accvgpr_read_b32 v81, a137
	v_accvgpr_read_b32 v80, a138
	v_accvgpr_read_b32 v79, a139
	v_accvgpr_read_b32 v78, a140
	v_accvgpr_read_b32 v77, a141
	v_accvgpr_read_b32 v76, a142
	v_accvgpr_read_b32 v75, a143
	v_accvgpr_read_b32 v74, a112
	v_accvgpr_read_b32 v73, a113
	v_accvgpr_read_b32 v72, a114
	v_accvgpr_read_b32 v71, a115
	v_accvgpr_read_b32 v70, a116
	v_accvgpr_read_b32 v69, a117
	v_accvgpr_read_b32 v68, a118
	v_accvgpr_read_b32 v67, a119
	v_accvgpr_read_b32 v66, a120
	v_accvgpr_read_b32 v65, a121
	v_accvgpr_read_b32 v64, a122
	v_accvgpr_read_b32 v63, a123
	v_accvgpr_read_b32 v62, a124
	v_accvgpr_read_b32 v61, a125
	v_accvgpr_read_b32 v60, a126
	v_accvgpr_read_b32 v59, a127
	v_accvgpr_read_b32 v57, a96
	v_accvgpr_read_b32 v56, a97
	v_accvgpr_read_b32 v55, a98
	v_accvgpr_read_b32 v54, a99
	v_accvgpr_read_b32 v53, a100
	v_accvgpr_read_b32 v52, a101
	v_accvgpr_read_b32 v51, a102
	v_accvgpr_read_b32 v50, a103
	v_accvgpr_read_b32 v49, a104
	v_accvgpr_read_b32 v48, a105
	v_accvgpr_read_b32 v47, a106
	v_accvgpr_read_b32 v46, a107
	v_accvgpr_read_b32 v45, a108
	v_accvgpr_read_b32 v44, a109
	v_accvgpr_read_b32 v43, a110
	v_accvgpr_read_b32 v42, a111
	v_accvgpr_read_b32 v41, a80
	v_accvgpr_read_b32 v40, a81
	v_accvgpr_read_b32 v39, a82
	v_accvgpr_read_b32 v38, a83
	v_accvgpr_read_b32 v37, a84
	v_accvgpr_read_b32 v36, a85
	v_accvgpr_read_b32 v35, a86
	v_accvgpr_read_b32 v34, a87
	v_accvgpr_read_b32 v33, a88
	v_accvgpr_read_b32 v32, a89
	v_accvgpr_read_b32 v31, a90
	v_accvgpr_read_b32 v30, a91
	v_accvgpr_read_b32 v29, a92
	v_accvgpr_read_b32 v28, a93
	v_accvgpr_read_b32 v27, a94
	v_accvgpr_read_b32 v26, a95
	v_accvgpr_read_b32 v25, a64
	v_accvgpr_read_b32 v24, a65
	v_accvgpr_read_b32 v23, a66
	v_accvgpr_read_b32 v22, a67
	v_accvgpr_read_b32 v21, a68
	v_accvgpr_read_b32 v20, a69
	v_accvgpr_read_b32 v19, a70
	v_accvgpr_read_b32 v18, a71
	v_accvgpr_read_b32 v12, a72
	v_accvgpr_read_b32 v13, a73
	v_accvgpr_read_b32 v14, a74
	v_accvgpr_read_b32 v15, a75
	v_accvgpr_read_b32 v8, a76
	v_accvgpr_read_b32 v9, a77
	v_accvgpr_read_b32 v10, a78
	v_accvgpr_read_b32 v11, a79
	v_mov_b32_e32 v6, s8
	v_mov_b32_e32 v7, s9
	v_lshl_add_u64 v[2:3], v[4:5], 1, v[2:3]
	v_mov_b32_e32 v4, s1
	v_mov_b32_e32 v5, 0x20000
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v0, v0, v1
	;;#ASMEND
	v_accvgpr_read_b32 v1, a50
	v_add_lshl_u32 v107, v107, v16, 1
	s_mov_b64 s[6:7], exec
	v_accvgpr_read_b32 v108, a51
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v1, v1, v108
	;;#ASMEND
.LBB0_30:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s16, v2
	v_readfirstlane_b32 s17, v3
	v_readfirstlane_b32 s18, v4
	v_readfirstlane_b32 s19, v5
	v_cmp_eq_u64_e32 vcc, s[16:17], v[2:3]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[18:19], v[4:5]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[0:1], v107, s[16:19], 0 offen
                                        ; implicit-def: $vgpr0_vgpr1
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_30
; %bb.31:
	s_mov_b64 exec, s[6:7]
	v_accvgpr_read_b32 v0, a52
	v_accvgpr_read_b32 v1, a53
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v0, v0, v1
	;;#ASMEND
	v_accvgpr_read_b32 v1, a54
	s_mov_b64 s[6:7], exec
	v_accvgpr_read_b32 v108, a55
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v1, v1, v108
	;;#ASMEND
.LBB0_32:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s16, v2
	v_readfirstlane_b32 s17, v3
	v_readfirstlane_b32 s18, v4
	v_readfirstlane_b32 s19, v5
	v_cmp_eq_u64_e32 vcc, s[16:17], v[2:3]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[18:19], v[4:5]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[0:1], v107, s[16:19], 0 offen offset:16
                                        ; implicit-def: $vgpr0_vgpr1
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_32
; %bb.33:
	s_mov_b64 exec, s[6:7]
	v_accvgpr_read_b32 v0, a56
	v_accvgpr_read_b32 v1, a57
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v0, v0, v1
	;;#ASMEND
	v_accvgpr_read_b32 v1, a58
	s_mov_b64 s[6:7], exec
	v_accvgpr_read_b32 v108, a59
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v1, v1, v108
	;;#ASMEND
.LBB0_34:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s16, v2
	v_readfirstlane_b32 s17, v3
	v_readfirstlane_b32 s18, v4
	v_readfirstlane_b32 s19, v5
	v_cmp_eq_u64_e32 vcc, s[16:17], v[2:3]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[18:19], v[4:5]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[0:1], v107, s[16:19], 0 offen offset:32
                                        ; implicit-def: $vgpr0_vgpr1
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_34
; %bb.35:
	s_mov_b64 exec, s[6:7]
	v_accvgpr_read_b32 v0, a60
	v_accvgpr_read_b32 v1, a61
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v0, v0, v1
	;;#ASMEND
	v_accvgpr_read_b32 v1, a62
	s_mov_b64 s[6:7], exec
	v_accvgpr_read_b32 v108, a63
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v1, v1, v108
	;;#ASMEND
.LBB0_36:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s16, v2
	v_readfirstlane_b32 s17, v3
	v_readfirstlane_b32 s18, v4
	v_readfirstlane_b32 s19, v5
	v_cmp_eq_u64_e32 vcc, s[16:17], v[2:3]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[18:19], v[4:5]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[0:1], v107, s[16:19], 0 offen offset:48
                                        ; implicit-def: $vgpr0_vgpr1
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_36
; %bb.37:
	s_mov_b64 exec, s[6:7]
	v_accvgpr_read_b32 v0, a32
	v_accvgpr_read_b32 v1, a33
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v0, v0, v1
	;;#ASMEND
	v_accvgpr_read_b32 v1, a34
	s_mov_b64 s[6:7], exec
	v_accvgpr_read_b32 v108, a35
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v1, v1, v108
	;;#ASMEND
.LBB0_38:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s16, v2
	v_readfirstlane_b32 s17, v3
	v_readfirstlane_b32 s18, v4
	v_readfirstlane_b32 s19, v5
	v_cmp_eq_u64_e32 vcc, s[16:17], v[2:3]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[18:19], v[4:5]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[0:1], v107, s[16:19], 0 offen offset:64
                                        ; implicit-def: $vgpr0_vgpr1
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_38
; %bb.39:
	s_mov_b64 exec, s[6:7]
	v_accvgpr_read_b32 v0, a36
	v_accvgpr_read_b32 v1, a37
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v0, v0, v1
	;;#ASMEND
	v_accvgpr_read_b32 v1, a38
	s_mov_b64 s[6:7], exec
	v_accvgpr_read_b32 v108, a39
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v1, v1, v108
	;;#ASMEND
.LBB0_40:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s16, v2
	v_readfirstlane_b32 s17, v3
	v_readfirstlane_b32 s18, v4
	v_readfirstlane_b32 s19, v5
	v_cmp_eq_u64_e32 vcc, s[16:17], v[2:3]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[18:19], v[4:5]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[0:1], v107, s[16:19], 0 offen offset:80
                                        ; implicit-def: $vgpr0_vgpr1
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_40
; %bb.41:
	s_mov_b64 exec, s[6:7]
	v_accvgpr_read_b32 v0, a40
	v_accvgpr_read_b32 v1, a41
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v0, v0, v1
	;;#ASMEND
	v_accvgpr_read_b32 v1, a42
	s_mov_b64 s[6:7], exec
	v_accvgpr_read_b32 v108, a43
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v1, v1, v108
	;;#ASMEND
.LBB0_42:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s16, v2
	v_readfirstlane_b32 s17, v3
	v_readfirstlane_b32 s18, v4
	v_readfirstlane_b32 s19, v5
	v_cmp_eq_u64_e32 vcc, s[16:17], v[2:3]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[18:19], v[4:5]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[0:1], v107, s[16:19], 0 offen offset:96
                                        ; implicit-def: $vgpr0_vgpr1
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_42
; %bb.43:
	s_mov_b64 exec, s[6:7]
	v_accvgpr_read_b32 v0, a44
	v_accvgpr_read_b32 v1, a45
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v0, v0, v1
	;;#ASMEND
	v_accvgpr_read_b32 v1, a46
	s_mov_b64 s[6:7], exec
	v_accvgpr_read_b32 v108, a47
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v1, v1, v108
	;;#ASMEND
.LBB0_44:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s16, v2
	v_readfirstlane_b32 s17, v3
	v_readfirstlane_b32 s18, v4
	v_readfirstlane_b32 s19, v5
	v_cmp_eq_u64_e32 vcc, s[16:17], v[2:3]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[18:19], v[4:5]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[0:1], v107, s[16:19], 0 offen offset:112
                                        ; implicit-def: $vgpr0_vgpr1
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_44
; %bb.45:
	s_mov_b64 exec, s[6:7]
	v_accvgpr_read_b32 v0, a16
	v_accvgpr_read_b32 v1, a17
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v0, v0, v1
	;;#ASMEND
	v_accvgpr_read_b32 v1, a18
	s_mov_b64 s[6:7], exec
	v_accvgpr_read_b32 v108, a19
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v1, v1, v108
	;;#ASMEND
.LBB0_46:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s16, v2
	v_readfirstlane_b32 s17, v3
	v_readfirstlane_b32 s18, v4
	v_readfirstlane_b32 s19, v5
	v_cmp_eq_u64_e32 vcc, s[16:17], v[2:3]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[18:19], v[4:5]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[0:1], v107, s[16:19], 0 offen offset:128
                                        ; implicit-def: $vgpr0_vgpr1
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_46
; %bb.47:
	s_mov_b64 exec, s[6:7]
	v_accvgpr_read_b32 v0, a20
	v_accvgpr_read_b32 v1, a21
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v0, v0, v1
	;;#ASMEND
	v_accvgpr_read_b32 v1, a22
	s_mov_b64 s[6:7], exec
	v_accvgpr_read_b32 v108, a23
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v1, v1, v108
	;;#ASMEND
.LBB0_48:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s16, v2
	v_readfirstlane_b32 s17, v3
	v_readfirstlane_b32 s18, v4
	v_readfirstlane_b32 s19, v5
	v_cmp_eq_u64_e32 vcc, s[16:17], v[2:3]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[18:19], v[4:5]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[0:1], v107, s[16:19], 0 offen offset:144
                                        ; implicit-def: $vgpr0_vgpr1
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_48
; %bb.49:
	s_mov_b64 exec, s[6:7]
	v_accvgpr_read_b32 v0, a24
	v_accvgpr_read_b32 v1, a25
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v0, v0, v1
	;;#ASMEND
	v_accvgpr_read_b32 v1, a26
	s_mov_b64 s[6:7], exec
	v_accvgpr_read_b32 v108, a27
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v1, v1, v108
	;;#ASMEND
.LBB0_50:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s16, v2
	v_readfirstlane_b32 s17, v3
	v_readfirstlane_b32 s18, v4
	v_readfirstlane_b32 s19, v5
	v_cmp_eq_u64_e32 vcc, s[16:17], v[2:3]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[18:19], v[4:5]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[0:1], v107, s[16:19], 0 offen offset:160
                                        ; implicit-def: $vgpr0_vgpr1
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_50
; %bb.51:
	s_mov_b64 exec, s[6:7]
	v_accvgpr_read_b32 v0, a28
	v_accvgpr_read_b32 v1, a29
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v0, v0, v1
	;;#ASMEND
	v_accvgpr_read_b32 v1, a30
	s_mov_b64 s[6:7], exec
	v_accvgpr_read_b32 v108, a31
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v1, v1, v108
	;;#ASMEND
.LBB0_52:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s16, v2
	v_readfirstlane_b32 s17, v3
	v_readfirstlane_b32 s18, v4
	v_readfirstlane_b32 s19, v5
	v_cmp_eq_u64_e32 vcc, s[16:17], v[2:3]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[18:19], v[4:5]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[0:1], v107, s[16:19], 0 offen offset:176
                                        ; implicit-def: $vgpr0_vgpr1
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_52
; %bb.53:
	s_mov_b64 exec, s[6:7]
	v_accvgpr_read_b32 v0, a0
	v_accvgpr_read_b32 v1, a1
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v0, v0, v1
	;;#ASMEND
	v_accvgpr_read_b32 v1, a2
	s_mov_b64 s[6:7], exec
	v_accvgpr_read_b32 v108, a3
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v1, v1, v108
	;;#ASMEND
.LBB0_54:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s16, v2
	v_readfirstlane_b32 s17, v3
	v_readfirstlane_b32 s18, v4
	v_readfirstlane_b32 s19, v5
	v_cmp_eq_u64_e32 vcc, s[16:17], v[2:3]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[18:19], v[4:5]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[0:1], v107, s[16:19], 0 offen offset:192
                                        ; implicit-def: $vgpr0_vgpr1
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_54
; %bb.55:
	s_mov_b64 exec, s[6:7]
	v_accvgpr_read_b32 v0, a4
	v_accvgpr_read_b32 v1, a5
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v0, v0, v1
	;;#ASMEND
	v_accvgpr_read_b32 v1, a6
	s_mov_b64 s[6:7], exec
	v_accvgpr_read_b32 v108, a7
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v1, v1, v108
	;;#ASMEND
.LBB0_56:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s16, v2
	v_readfirstlane_b32 s17, v3
	v_readfirstlane_b32 s18, v4
	v_readfirstlane_b32 s19, v5
	v_cmp_eq_u64_e32 vcc, s[16:17], v[2:3]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[18:19], v[4:5]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[0:1], v107, s[16:19], 0 offen offset:208
                                        ; implicit-def: $vgpr0_vgpr1
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_56
; %bb.57:
	s_mov_b64 exec, s[6:7]
	v_accvgpr_read_b32 v0, a8
	v_accvgpr_read_b32 v1, a9
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v0, v0, v1
	;;#ASMEND
	v_accvgpr_read_b32 v1, a10
	s_mov_b64 s[6:7], exec
	v_accvgpr_read_b32 v108, a11
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v1, v1, v108
	;;#ASMEND
.LBB0_58:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s16, v2
	v_readfirstlane_b32 s17, v3
	v_readfirstlane_b32 s18, v4
	v_readfirstlane_b32 s19, v5
	v_cmp_eq_u64_e32 vcc, s[16:17], v[2:3]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[18:19], v[4:5]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[0:1], v107, s[16:19], 0 offen offset:224
                                        ; implicit-def: $vgpr0_vgpr1
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_58
; %bb.59:
	s_mov_b64 exec, s[6:7]
	v_accvgpr_read_b32 v0, a12
	v_accvgpr_read_b32 v1, a13
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v0, v0, v1
	;;#ASMEND
	v_accvgpr_read_b32 v1, a14
	s_mov_b64 s[6:7], exec
	v_accvgpr_read_b32 v108, a15
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v1, v1, v108
	;;#ASMEND
.LBB0_60:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s16, v2
	v_readfirstlane_b32 s17, v3
	v_readfirstlane_b32 s18, v4
	v_readfirstlane_b32 s19, v5
	v_cmp_eq_u64_e32 vcc, s[16:17], v[2:3]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[18:19], v[4:5]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[0:1], v107, s[16:19], 0 offen offset:240
                                        ; implicit-def: $vgpr2_vgpr3_vgpr4_vgpr5
                                        ; implicit-def: $vgpr0_vgpr1
                                        ; implicit-def: $vgpr107
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_60
; %bb.61:
	s_mov_b64 exec, s[6:7]
	s_mul_i32 s4, s4, s12
	v_add_u32_e32 v0, s4, v58
	v_mul_lo_u32 v0, v0, s14
	v_add_u32_e32 v0, s2, v0
	v_mul_lo_u32 v0, v0, s34
	s_mul_i32 s0, s14, s34
	s_mul_i32 s1, s10, s12
	v_ashrrev_i32_e32 v1, 31, v0
	s_mul_i32 s1, s0, s1
	v_lshl_add_u64 v[0:1], v[0:1], 1, v[6:7]
	s_lshl_b32 s1, s1, 1
	v_mul_lo_u32 v6, v17, s0
	v_mul_f32_e32 v4, 0x3d93cd3a, v106
	v_mul_f32_e32 v5, 0x3d93cd3a, v105
	v_mul_f32_e32 v102, 0x3d93cd3a, v102
	v_mul_f32_e32 v101, 0x3d93cd3a, v101
	v_mul_f32_e32 v100, 0x3d93cd3a, v100
	v_mul_f32_e32 v99, 0x3d93cd3a, v99
	v_mul_f32_e32 v98, 0x3d93cd3a, v98
	v_mul_f32_e32 v97, 0x3d93cd3a, v97
	v_mul_f32_e32 v96, 0x3d93cd3a, v96
	v_mul_f32_e32 v95, 0x3d93cd3a, v95
	v_mul_f32_e32 v94, 0x3d93cd3a, v94
	v_mul_f32_e32 v93, 0x3d93cd3a, v93
	v_mul_f32_e32 v92, 0x3d93cd3a, v92
	v_mul_f32_e32 v91, 0x3d93cd3a, v91
	v_mul_f32_e32 v90, 0x3d93cd3a, v90
	v_mul_f32_e32 v89, 0x3d93cd3a, v89
	v_mul_f32_e32 v88, 0x3d93cd3a, v88
	v_mul_f32_e32 v87, 0x3d93cd3a, v87
	v_mul_f32_e32 v86, 0x3d93cd3a, v86
	v_mul_f32_e32 v85, 0x3d93cd3a, v85
	v_mul_f32_e32 v84, 0x3d93cd3a, v84
	v_mul_f32_e32 v83, 0x3d93cd3a, v83
	v_mul_f32_e32 v82, 0x3d93cd3a, v82
	v_mul_f32_e32 v81, 0x3d93cd3a, v81
	v_mul_f32_e32 v80, 0x3d93cd3a, v80
	v_mul_f32_e32 v79, 0x3d93cd3a, v79
	v_mul_f32_e32 v78, 0x3d93cd3a, v78
	v_mul_f32_e32 v77, 0x3d93cd3a, v77
	v_mul_f32_e32 v76, 0x3d93cd3a, v76
	v_mul_f32_e32 v75, 0x3d93cd3a, v75
	v_mul_f32_e32 v74, 0x3d93cd3a, v74
	v_mul_f32_e32 v73, 0x3d93cd3a, v73
	v_mul_f32_e32 v72, 0x3d93cd3a, v72
	v_mul_f32_e32 v71, 0x3d93cd3a, v71
	v_mul_f32_e32 v70, 0x3d93cd3a, v70
	v_mul_f32_e32 v69, 0x3d93cd3a, v69
	v_mul_f32_e32 v68, 0x3d93cd3a, v68
	v_mul_f32_e32 v67, 0x3d93cd3a, v67
	v_mul_f32_e32 v66, 0x3d93cd3a, v66
	v_mul_f32_e32 v65, 0x3d93cd3a, v65
	v_mul_f32_e32 v64, 0x3d93cd3a, v64
	v_mul_f32_e32 v63, 0x3d93cd3a, v63
	v_mul_f32_e32 v62, 0x3d93cd3a, v62
	v_mul_f32_e32 v61, 0x3d93cd3a, v61
	v_mul_f32_e32 v60, 0x3d93cd3a, v60
	v_mul_f32_e32 v59, 0x3d93cd3a, v59
	v_mul_f32_e32 v57, 0x3d93cd3a, v57
	v_mul_f32_e32 v56, 0x3d93cd3a, v56
	v_mul_f32_e32 v55, 0x3d93cd3a, v55
	v_mul_f32_e32 v54, 0x3d93cd3a, v54
	v_mul_f32_e32 v53, 0x3d93cd3a, v53
	v_mul_f32_e32 v52, 0x3d93cd3a, v52
	v_mul_f32_e32 v51, 0x3d93cd3a, v51
	v_mul_f32_e32 v50, 0x3d93cd3a, v50
	v_mul_f32_e32 v49, 0x3d93cd3a, v49
	v_mul_f32_e32 v48, 0x3d93cd3a, v48
	v_mul_f32_e32 v47, 0x3d93cd3a, v47
	v_mul_f32_e32 v46, 0x3d93cd3a, v46
	v_mul_f32_e32 v45, 0x3d93cd3a, v45
	v_mul_f32_e32 v44, 0x3d93cd3a, v44
	v_mul_f32_e32 v43, 0x3d93cd3a, v43
	v_mul_f32_e32 v42, 0x3d93cd3a, v42
	v_mul_f32_e32 v41, 0x3d93cd3a, v41
	v_mul_f32_e32 v40, 0x3d93cd3a, v40
	v_mul_f32_e32 v39, 0x3d93cd3a, v39
	v_mul_f32_e32 v38, 0x3d93cd3a, v38
	v_mul_f32_e32 v37, 0x3d93cd3a, v37
	v_mul_f32_e32 v36, 0x3d93cd3a, v36
	v_mul_f32_e32 v35, 0x3d93cd3a, v35
	v_mul_f32_e32 v34, 0x3d93cd3a, v34
	v_mul_f32_e32 v33, 0x3d93cd3a, v33
	v_mul_f32_e32 v32, 0x3d93cd3a, v32
	v_mul_f32_e32 v31, 0x3d93cd3a, v31
	v_mul_f32_e32 v30, 0x3d93cd3a, v30
	v_mul_f32_e32 v29, 0x3d93cd3a, v29
	v_mul_f32_e32 v28, 0x3d93cd3a, v28
	v_mul_f32_e32 v27, 0x3d93cd3a, v27
	v_mul_f32_e32 v26, 0x3d93cd3a, v26
	v_mul_f32_e32 v25, 0x3d93cd3a, v25
	v_mul_f32_e32 v24, 0x3d93cd3a, v24
	v_mul_f32_e32 v23, 0x3d93cd3a, v23
	v_mul_f32_e32 v22, 0x3d93cd3a, v22
	v_mul_f32_e32 v21, 0x3d93cd3a, v21
	v_mul_f32_e32 v20, 0x3d93cd3a, v20
	v_mul_f32_e32 v19, 0x3d93cd3a, v19
	v_mul_f32_e32 v18, 0x3d93cd3a, v18
	v_mul_f32_e32 v12, 0x3d93cd3a, v12
	v_mul_f32_e32 v13, 0x3d93cd3a, v13
	v_mul_f32_e32 v14, 0x3d93cd3a, v14
	v_mul_f32_e32 v15, 0x3d93cd3a, v15
	v_mul_f32_e32 v8, 0x3d93cd3a, v8
	v_mul_f32_e32 v9, 0x3d93cd3a, v9
	v_mul_f32_e32 v10, 0x3d93cd3a, v10
	v_mul_f32_e32 v11, 0x3d93cd3a, v11
	v_mov_b32_e32 v2, s1
	v_mov_b32_e32 v3, 0x20000
	v_add_lshl_u32 v6, v6, v16, 1
	s_mov_b64 s[2:3], exec
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	s_barrier
	v_mul_f32_e32 v104, 0x3d93cd3a, v104
	v_mul_f32_e32 v103, 0x3d93cd3a, v103
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v4, v4, v5
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v5, v104, v103
	;;#ASMEND
.LBB0_62:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v0
	v_readfirstlane_b32 s5, v1
	v_readfirstlane_b32 s6, v2
	v_readfirstlane_b32 s7, v3
	v_cmp_eq_u64_e32 vcc, s[4:5], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[4:5], v6, s[4:7], 0 offen
                                        ; implicit-def: $vgpr4_vgpr5
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_62
; %bb.63:
	s_mov_b64 exec, s[2:3]
	s_mov_b64 s[2:3], exec
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v4, v102, v101
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v5, v100, v99
	;;#ASMEND
.LBB0_64:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v0
	v_readfirstlane_b32 s5, v1
	v_readfirstlane_b32 s6, v2
	v_readfirstlane_b32 s7, v3
	v_cmp_eq_u64_e32 vcc, s[4:5], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[4:5], v6, s[4:7], 0 offen offset:16
                                        ; implicit-def: $vgpr4_vgpr5
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_64
; %bb.65:
	s_mov_b64 exec, s[2:3]
	s_mov_b64 s[2:3], exec
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v4, v98, v97
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v5, v96, v95
	;;#ASMEND
.LBB0_66:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v0
	v_readfirstlane_b32 s5, v1
	v_readfirstlane_b32 s6, v2
	v_readfirstlane_b32 s7, v3
	v_cmp_eq_u64_e32 vcc, s[4:5], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[4:5], v6, s[4:7], 0 offen offset:32
                                        ; implicit-def: $vgpr4_vgpr5
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_66
; %bb.67:
	s_mov_b64 exec, s[2:3]
	s_mov_b64 s[2:3], exec
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v4, v94, v93
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v5, v92, v91
	;;#ASMEND
.LBB0_68:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v0
	v_readfirstlane_b32 s5, v1
	v_readfirstlane_b32 s6, v2
	v_readfirstlane_b32 s7, v3
	v_cmp_eq_u64_e32 vcc, s[4:5], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[4:5], v6, s[4:7], 0 offen offset:48
                                        ; implicit-def: $vgpr4_vgpr5
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_68
; %bb.69:
	s_mov_b64 exec, s[2:3]
	s_mov_b64 s[2:3], exec
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v4, v90, v89
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v5, v88, v87
	;;#ASMEND
.LBB0_70:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v0
	v_readfirstlane_b32 s5, v1
	v_readfirstlane_b32 s6, v2
	v_readfirstlane_b32 s7, v3
	v_cmp_eq_u64_e32 vcc, s[4:5], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[4:5], v6, s[4:7], 0 offen offset:64
                                        ; implicit-def: $vgpr4_vgpr5
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_70
; %bb.71:
	s_mov_b64 exec, s[2:3]
	s_mov_b64 s[2:3], exec
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v4, v86, v85
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v5, v84, v83
	;;#ASMEND
.LBB0_72:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v0
	v_readfirstlane_b32 s5, v1
	v_readfirstlane_b32 s6, v2
	v_readfirstlane_b32 s7, v3
	v_cmp_eq_u64_e32 vcc, s[4:5], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[4:5], v6, s[4:7], 0 offen offset:80
                                        ; implicit-def: $vgpr4_vgpr5
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_72
; %bb.73:
	s_mov_b64 exec, s[2:3]
	s_mov_b64 s[2:3], exec
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v4, v82, v81
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v5, v80, v79
	;;#ASMEND
.LBB0_74:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v0
	v_readfirstlane_b32 s5, v1
	v_readfirstlane_b32 s6, v2
	v_readfirstlane_b32 s7, v3
	v_cmp_eq_u64_e32 vcc, s[4:5], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[4:5], v6, s[4:7], 0 offen offset:96
                                        ; implicit-def: $vgpr4_vgpr5
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_74
; %bb.75:
	s_mov_b64 exec, s[2:3]
	s_mov_b64 s[2:3], exec
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v4, v78, v77
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v5, v76, v75
	;;#ASMEND
.LBB0_76:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v0
	v_readfirstlane_b32 s5, v1
	v_readfirstlane_b32 s6, v2
	v_readfirstlane_b32 s7, v3
	v_cmp_eq_u64_e32 vcc, s[4:5], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[4:5], v6, s[4:7], 0 offen offset:112
                                        ; implicit-def: $vgpr4_vgpr5
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_76
; %bb.77:
	s_mov_b64 exec, s[2:3]
	s_mov_b64 s[2:3], exec
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v4, v74, v73
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v5, v72, v71
	;;#ASMEND
.LBB0_78:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v0
	v_readfirstlane_b32 s5, v1
	v_readfirstlane_b32 s6, v2
	v_readfirstlane_b32 s7, v3
	v_cmp_eq_u64_e32 vcc, s[4:5], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[4:5], v6, s[4:7], 0 offen offset:128
                                        ; implicit-def: $vgpr4_vgpr5
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_78
; %bb.79:
	s_mov_b64 exec, s[2:3]
	s_mov_b64 s[2:3], exec
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v4, v70, v69
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v5, v68, v67
	;;#ASMEND
.LBB0_80:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v0
	v_readfirstlane_b32 s5, v1
	v_readfirstlane_b32 s6, v2
	v_readfirstlane_b32 s7, v3
	v_cmp_eq_u64_e32 vcc, s[4:5], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[4:5], v6, s[4:7], 0 offen offset:144
                                        ; implicit-def: $vgpr4_vgpr5
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_80
; %bb.81:
	s_mov_b64 exec, s[2:3]
	s_mov_b64 s[2:3], exec
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v4, v66, v65
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v5, v64, v63
	;;#ASMEND
.LBB0_82:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v0
	v_readfirstlane_b32 s5, v1
	v_readfirstlane_b32 s6, v2
	v_readfirstlane_b32 s7, v3
	v_cmp_eq_u64_e32 vcc, s[4:5], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[4:5], v6, s[4:7], 0 offen offset:160
                                        ; implicit-def: $vgpr4_vgpr5
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_82
; %bb.83:
	s_mov_b64 exec, s[2:3]
	s_mov_b64 s[2:3], exec
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v4, v62, v61
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v5, v60, v59
	;;#ASMEND
.LBB0_84:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v0
	v_readfirstlane_b32 s5, v1
	v_readfirstlane_b32 s6, v2
	v_readfirstlane_b32 s7, v3
	v_cmp_eq_u64_e32 vcc, s[4:5], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[4:5], v6, s[4:7], 0 offen offset:176
                                        ; implicit-def: $vgpr4_vgpr5
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_84
; %bb.85:
	s_mov_b64 exec, s[2:3]
	s_mov_b64 s[2:3], exec
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v4, v57, v56
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v5, v55, v54
	;;#ASMEND
.LBB0_86:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v0
	v_readfirstlane_b32 s5, v1
	v_readfirstlane_b32 s6, v2
	v_readfirstlane_b32 s7, v3
	v_cmp_eq_u64_e32 vcc, s[4:5], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[4:5], v6, s[4:7], 0 offen offset:192
                                        ; implicit-def: $vgpr4_vgpr5
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_86
; %bb.87:
	s_mov_b64 exec, s[2:3]
	s_mov_b64 s[2:3], exec
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v4, v53, v52
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v5, v51, v50
	;;#ASMEND
.LBB0_88:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v0
	v_readfirstlane_b32 s5, v1
	v_readfirstlane_b32 s6, v2
	v_readfirstlane_b32 s7, v3
	v_cmp_eq_u64_e32 vcc, s[4:5], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[4:5], v6, s[4:7], 0 offen offset:208
                                        ; implicit-def: $vgpr4_vgpr5
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_88
; %bb.89:
	s_mov_b64 exec, s[2:3]
	s_mov_b64 s[2:3], exec
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v4, v49, v48
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v5, v47, v46
	;;#ASMEND
.LBB0_90:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v0
	v_readfirstlane_b32 s5, v1
	v_readfirstlane_b32 s6, v2
	v_readfirstlane_b32 s7, v3
	v_cmp_eq_u64_e32 vcc, s[4:5], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[4:5], v6, s[4:7], 0 offen offset:224
                                        ; implicit-def: $vgpr4_vgpr5
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_90
; %bb.91:
	s_mov_b64 exec, s[2:3]
	s_mov_b64 s[2:3], exec
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v4, v45, v44
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v5, v43, v42
	;;#ASMEND
.LBB0_92:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v0
	v_readfirstlane_b32 s5, v1
	v_readfirstlane_b32 s6, v2
	v_readfirstlane_b32 s7, v3
	v_cmp_eq_u64_e32 vcc, s[4:5], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[4:5], v6, s[4:7], 0 offen offset:240
                                        ; implicit-def: $vgpr4_vgpr5
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_92
; %bb.93:
	s_mov_b64 exec, s[2:3]
	s_mov_b64 s[2:3], exec
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v4, v41, v40
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v5, v39, v38
	;;#ASMEND
.LBB0_94:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v0
	v_readfirstlane_b32 s5, v1
	v_readfirstlane_b32 s6, v2
	v_readfirstlane_b32 s7, v3
	v_cmp_eq_u64_e32 vcc, s[4:5], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[4:5], v6, s[4:7], 0 offen offset:256
                                        ; implicit-def: $vgpr4_vgpr5
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_94
; %bb.95:
	s_mov_b64 exec, s[2:3]
	s_mov_b64 s[2:3], exec
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v4, v37, v36
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v5, v35, v34
	;;#ASMEND
.LBB0_96:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v0
	v_readfirstlane_b32 s5, v1
	v_readfirstlane_b32 s6, v2
	v_readfirstlane_b32 s7, v3
	v_cmp_eq_u64_e32 vcc, s[4:5], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[4:5], v6, s[4:7], 0 offen offset:272
                                        ; implicit-def: $vgpr4_vgpr5
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_96
; %bb.97:
	s_mov_b64 exec, s[2:3]
	s_mov_b64 s[2:3], exec
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v4, v33, v32
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v5, v31, v30
	;;#ASMEND
.LBB0_98:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v0
	v_readfirstlane_b32 s5, v1
	v_readfirstlane_b32 s6, v2
	v_readfirstlane_b32 s7, v3
	v_cmp_eq_u64_e32 vcc, s[4:5], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[4:5], v6, s[4:7], 0 offen offset:288
                                        ; implicit-def: $vgpr4_vgpr5
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_98
; %bb.99:
	s_mov_b64 exec, s[2:3]
	s_mov_b64 s[2:3], exec
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v4, v29, v28
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v5, v27, v26
	;;#ASMEND
.LBB0_100:                              ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v0
	v_readfirstlane_b32 s5, v1
	v_readfirstlane_b32 s6, v2
	v_readfirstlane_b32 s7, v3
	v_cmp_eq_u64_e32 vcc, s[4:5], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[4:5], v6, s[4:7], 0 offen offset:304
                                        ; implicit-def: $vgpr4_vgpr5
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_100
; %bb.101:
	s_mov_b64 exec, s[2:3]
	s_mov_b64 s[2:3], exec
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v4, v25, v24
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v5, v23, v22
	;;#ASMEND
.LBB0_102:                              ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v0
	v_readfirstlane_b32 s5, v1
	v_readfirstlane_b32 s6, v2
	v_readfirstlane_b32 s7, v3
	v_cmp_eq_u64_e32 vcc, s[4:5], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[4:5], v6, s[4:7], 0 offen offset:320
                                        ; implicit-def: $vgpr4_vgpr5
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_102
; %bb.103:
	s_mov_b64 exec, s[2:3]
	s_mov_b64 s[2:3], exec
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v4, v21, v20
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v5, v19, v18
	;;#ASMEND
.LBB0_104:                              ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v0
	v_readfirstlane_b32 s5, v1
	v_readfirstlane_b32 s6, v2
	v_readfirstlane_b32 s7, v3
	v_cmp_eq_u64_e32 vcc, s[4:5], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[4:5], v6, s[4:7], 0 offen offset:336
                                        ; implicit-def: $vgpr4_vgpr5
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_104
; %bb.105:
	s_mov_b64 exec, s[2:3]
	s_mov_b64 s[2:3], exec
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v4, v12, v13
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v5, v14, v15
	;;#ASMEND
.LBB0_106:                              ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v0
	v_readfirstlane_b32 s5, v1
	v_readfirstlane_b32 s6, v2
	v_readfirstlane_b32 s7, v3
	v_cmp_eq_u64_e32 vcc, s[4:5], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[4:5], v6, s[4:7], 0 offen offset:352
                                        ; implicit-def: $vgpr4_vgpr5
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_106
; %bb.107:
	s_mov_b64 exec, s[2:3]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v4, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v5, v10, v11
	;;#ASMEND
	s_mov_b64 s[0:1], exec
.LBB0_108:                              ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s4, v0
	v_readfirstlane_b32 s5, v1
	v_readfirstlane_b32 s6, v2
	v_readfirstlane_b32 s7, v3
	v_cmp_eq_u64_e32 vcc, s[4:5], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[4:5], v6, s[4:7], 0 offen offset:368
                                        ; implicit-def: $vgpr0_vgpr1_vgpr2_vgpr3
                                        ; implicit-def: $vgpr4_vgpr5
                                        ; implicit-def: $vgpr6
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_108
; %bb.109:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z32attend_bwd_combined_d192v128_ker34attn_bwd_combined_d192v128_globals
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 440
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
		.amdhsa_next_free_vgpr 296
		.amdhsa_next_free_sgpr 84
		.amdhsa_accum_offset 128
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
	.size	_Z32attend_bwd_combined_d192v128_ker34attn_bwd_combined_d192v128_globals, .Lfunc_end0-_Z32attend_bwd_combined_d192v128_ker34attn_bwd_combined_d192v128_globals
                                        ; -- End function
	.set _Z32attend_bwd_combined_d192v128_ker34attn_bwd_combined_d192v128_globals.num_vgpr, 126
	.set _Z32attend_bwd_combined_d192v128_ker34attn_bwd_combined_d192v128_globals.num_agpr, 168
	.set _Z32attend_bwd_combined_d192v128_ker34attn_bwd_combined_d192v128_globals.numbered_sgpr, 84
	.set _Z32attend_bwd_combined_d192v128_ker34attn_bwd_combined_d192v128_globals.private_seg_size, 0
	.set _Z32attend_bwd_combined_d192v128_ker34attn_bwd_combined_d192v128_globals.uses_vcc, 1
	.set _Z32attend_bwd_combined_d192v128_ker34attn_bwd_combined_d192v128_globals.uses_flat_scratch, 0
	.set _Z32attend_bwd_combined_d192v128_ker34attn_bwd_combined_d192v128_globals.has_dyn_sized_stack, 0
	.set _Z32attend_bwd_combined_d192v128_ker34attn_bwd_combined_d192v128_globals.has_recursion, 0
	.set _Z32attend_bwd_combined_d192v128_ker34attn_bwd_combined_d192v128_globals.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 15988
; TotalNumSgprs: 90
; NumVgprs: 126
; NumAgprs: 168
; TotalNumVgprs: 296
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 11
; VGPRBlocks: 36
; NumSGPRsForWavesPerEU: 90
; NumVGPRsForWavesPerEU: 296
; AccumOffset: 128
; Occupancy: 1
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 31
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.type	__hip_cuid_db3cb63947998211,@object ; @__hip_cuid_db3cb63947998211
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_db3cb63947998211
__hip_cuid_db3cb63947998211:
	.byte	0                               ; 0x0
	.size	__hip_cuid_db3cb63947998211, 1

	.ident	"clang version 20.0.0git (https://github.com/ROCm/llvm-project.git 27682a16360e33e37c4f3cc6adf9a620733f8fe1)"
	.ident	"AMD clang version 20.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.1.0 25425 1b0eada6b0ee93e2e694c8c146d23fca90bc11c5)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __shm
	.addrsig_sym __hip_cuid_db3cb63947998211
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     168
    .args:
      - .offset:         0
        .size:           440
        .value_kind:     by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 440
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _Z32attend_bwd_combined_d192v128_ker34attn_bwd_combined_d192v128_globals
    .private_segment_fixed_size: 0
    .sgpr_count:     90
    .sgpr_spill_count: 0
    .symbol:         _Z32attend_bwd_combined_d192v128_ker34attn_bwd_combined_d192v128_globals.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     296
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
