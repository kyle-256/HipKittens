	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.text
	.protected	_Z16mxfp4_art_kernel13gluon_globals ; -- Begin function _Z16mxfp4_art_kernel13gluon_globals
	.globl	_Z16mxfp4_art_kernel13gluon_globals
	.p2align	8
	.type	_Z16mxfp4_art_kernel13gluon_globals,@function
_Z16mxfp4_art_kernel13gluon_globals:    ; @_Z16mxfp4_art_kernel13gluon_globals
; %bb.0:
	s_load_dword s8, s[0:1], 0xf8
	s_waitcnt lgkmcnt(0)
	s_add_i32 s3, s8, 7
	s_ashr_i32 s4, s3, 31
	s_ashr_i32 s9, s8, 31
	s_lshr_b32 s4, s4, 29
	s_lshr_b32 s5, s9, 29
	s_add_i32 s3, s3, s4
	s_ashr_i32 s4, s3, 3
	s_add_i32 s3, s8, s5
	s_and_b32 s3, s3, -8
	s_sub_i32 s3, s8, s3
	s_cmp_lg_u32 s3, 0
	s_cselect_b32 s3, s3, 8
	s_ashr_i32 s5, s2, 31
	s_lshr_b32 s5, s5, 29
	s_add_i32 s6, s2, s5
	s_and_b32 s5, s6, -8
	s_sub_i32 s5, s2, s5
	s_cmp_ge_i32 s5, s3
	s_cbranch_scc0 .LBB0_2
; %bb.1:
	s_mul_i32 s2, s3, s4
	s_sub_i32 s3, s5, s3
	s_add_i32 s7, s4, -1
	s_mul_i32 s3, s3, s7
	s_add_i32 s7, s3, s2
	s_ashr_i32 s2, s6, 3
	s_cbranch_execz .LBB0_3
	s_branch .LBB0_4
.LBB0_2:
                                        ; implicit-def: $sgpr7
	s_ashr_i32 s2, s6, 3
.LBB0_3:
	s_mul_i32 s7, s4, s5
.LBB0_4:
	s_add_i32 s2, s7, s2
	s_cmp_ge_i32 s2, s8
	s_cbranch_scc0 .LBB0_6
; %bb.5:
	s_endpgm
.LBB0_6:
	s_load_dwordx2 s[4:5], s[0:1], 0x80
	s_load_dwordx2 s[6:7], s[0:1], 0xc0
                                        ; implicit-def: $vgpr28 : SGPR spill to VGPR lane
	s_load_dwordx2 s[26:27], s[0:1], 0xe0
	v_lshrrev_b32_e32 v2, 6, v0
	v_lshlrev_b32_e32 v2, 10, v2
	v_lshrrev_b32_e32 v10, 7, v0
	s_waitcnt lgkmcnt(0)
	v_writelane_b32 v28, s6, 0
	s_movk_i32 s53, 0x70
	v_lshlrev_b32_e32 v6, 6, v10
	v_writelane_b32 v28, s7, 1
	s_load_dword s27, s[0:1], 0xf0
	s_load_dwordx2 s[30:31], s[0:1], 0x0
	s_load_dwordx2 s[16:17], s[0:1], 0x30
	s_load_dwordx2 s[28:29], s[0:1], 0x50
	s_load_dwordx2 s[6:7], s[0:1], 0x60
	s_load_dwordx2 s[14:15], s[0:1], 0xb0
	s_load_dwordx2 s[12:13], s[0:1], 0x90
	s_ashr_i32 s1, s2, 31
	s_lshr_b32 s3, s1, 25
	s_lshr_b32 s0, s9, 27
	s_add_i32 s3, s2, s3
	s_add_i32 s8, s8, s0
	s_ashr_i32 s5, s3, 7
	s_ashr_i32 s0, s8, 5
	s_lshl_b32 s5, s5, 2
	s_sub_i32 s0, s0, s5
	s_min_i32 s0, s0, 4
	s_abs_i32 s8, s0
	v_cvt_f32_u32_e32 v1, s8
	s_sub_i32 s10, 0, s8
	s_abs_i32 s9, s2
	s_waitcnt lgkmcnt(0)
	v_readfirstlane_b32 s29, v2
	v_rcp_iflag_f32_e32 v1, v1
	s_add_i32 s34, s29, 0x8000
	s_add_i32 s33, s29, 0x4000
	s_add_i32 s35, s34, 0x4000
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	v_lshrrev_b32_e32 v5, 3, v0
	v_bfe_u32 v11, v0, 6, 1
	;;#ASMSTART
	;;#ASMEND
	v_readfirstlane_b32 s11, v1
	s_mul_i32 s10, s10, s11
	s_mul_hi_u32 s10, s11, s10
	s_add_i32 s11, s11, s10
	s_mul_hi_u32 s10, s9, s11
	s_mul_i32 s10, s10, s8
	s_sub_i32 s9, s9, s10
	s_sub_i32 s10, s9, s8
	s_cmp_ge_u32 s9, s8
	s_cselect_b32 s9, s10, s9
	s_sub_i32 s10, s9, s8
	s_cmp_ge_u32 s9, s8
	s_cselect_b32 s9, s10, s9
	s_xor_b32 s9, s9, s1
	s_sub_i32 s50, s9, s1
	s_and_b32 s1, s3, 0xffffff80
	s_sub_i32 s1, s2, s1
	s_xor_b32 s0, s1, s0
	s_ashr_i32 s51, s0, 31
	s_abs_i32 s0, s1
	s_mul_hi_u32 s1, s0, s11
	s_mul_i32 s2, s1, s8
	s_sub_i32 s0, s0, s2
	s_add_i32 s50, s50, s5
	s_add_i32 s2, s1, 1
	s_sub_i32 s3, s0, s8
	s_cmp_ge_u32 s0, s8
	s_cselect_b32 s1, s2, s1
	s_cselect_b32 s0, s3, s0
	s_add_i32 s2, s1, 1
	s_cmp_ge_u32 s0, s8
	v_lshlrev_b32_e32 v1, 4, v0
	s_cselect_b32 s0, s2, s1
	v_bitop3_b32 v4, v1, s53, v0 bitop3:0x48
	v_lshl_or_b32 v24, s50, 8, v6
	s_xor_b32 s52, s0, s51
	v_mad_u64_u32 v[2:3], s[0:1], v5, s28, v[4:5]
	v_ashrrev_i32_e32 v6, 6, v24
	s_lshl_b32 s0, s28, 5
	v_mul_lo_u32 v6, v6, s4
	v_add_u32_e32 v1, s0, v2
	v_or_b32_e32 v5, 0x60, v5
	v_ashrrev_i32_e32 v7, 31, v6
	v_add_u32_e32 v3, s0, v1
	v_mad_u64_u32 v[4:5], s[0:1], v5, s28, v[4:5]
	v_lshl_add_u64 v[6:7], s[6:7], 0, v[6:7]
	s_sub_i32 s49, s52, s51
	v_readfirstlane_b32 s0, v6
	v_or_b32_e32 v6, 0x80, v24
	v_ashrrev_i32_e32 v6, 6, v6
	v_mul_lo_u32 v6, v6, s4
	v_readfirstlane_b32 s1, v7
	v_ashrrev_i32_e32 v7, 31, v6
	v_lshl_add_u64 v[6:7], s[6:7], 0, v[6:7]
	s_lshl_b32 s18, s49, 8
	v_readfirstlane_b32 s8, v6
	v_lshl_or_b32 v6, v11, 6, s18
	v_readfirstlane_b32 s9, v7
	v_ashrrev_i32_e32 v7, 6, v6
	s_mov_b32 s3, 0x110000
	s_mov_b32 s2, -1
	v_mul_lo_u32 v8, v7, s14
	s_mov_b64 s[6:7], s[2:3]
	v_ashrrev_i32_e32 v9, 31, v8
	v_or_b32_e32 v7, 0x80, v6
	s_mov_b64 s[4:5], s[0:1]
	v_lshl_add_u64 v[8:9], s[12:13], 0, v[8:9]
	v_ashrrev_i32_e32 v7, 6, v7
	s_mov_b32 s4, s8
	s_mov_b32 s5, s9
	v_readfirstlane_b32 s15, v8
	s_mov_b64 s[10:11], s[2:3]
	v_mul_lo_u32 v8, v7, s14
	v_readfirstlane_b32 s19, v9
	s_mov_b64 s[8:9], s[0:1]
	v_ashrrev_i32_e32 v9, 31, v8
	s_mov_b32 s8, s15
	v_lshl_add_u64 v[8:9], s[12:13], 0, v[8:9]
	s_mov_b64 s[14:15], s[2:3]
	s_mov_b64 s[12:13], s[0:1]
	s_mov_b32 s12, s16
	s_mul_i32 s16, s18, s28
	s_mov_b32 s13, s17
	s_mov_b32 s17, s16
	s_mov_b32 s18, s29
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_write_b32 a[0], 0
v_accvgpr_write_b32 a[1], 0
v_accvgpr_write_b32 a[2], 0
v_accvgpr_write_b32 a[3], 0

	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_write_b32 a[4], 0
v_accvgpr_write_b32 a[5], 0
v_accvgpr_write_b32 a[6], 0
v_accvgpr_write_b32 a[7], 0

	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_write_b32 a[8], 0
v_accvgpr_write_b32 a[9], 0
v_accvgpr_write_b32 a[10], 0
v_accvgpr_write_b32 a[11], 0

	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_write_b32 a[12], 0
v_accvgpr_write_b32 a[13], 0
v_accvgpr_write_b32 a[14], 0
v_accvgpr_write_b32 a[15], 0

	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_write_b32 a[16], 0
v_accvgpr_write_b32 a[17], 0
v_accvgpr_write_b32 a[18], 0
v_accvgpr_write_b32 a[19], 0

	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_write_b32 a[20], 0
v_accvgpr_write_b32 a[21], 0
v_accvgpr_write_b32 a[22], 0
v_accvgpr_write_b32 a[23], 0

	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_write_b32 a[24], 0
v_accvgpr_write_b32 a[25], 0
v_accvgpr_write_b32 a[26], 0
v_accvgpr_write_b32 a[27], 0

	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_write_b32 a[28], 0
v_accvgpr_write_b32 a[29], 0
v_accvgpr_write_b32 a[30], 0
v_accvgpr_write_b32 a[31], 0

	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_write_b32 a[32], 0
v_accvgpr_write_b32 a[33], 0
v_accvgpr_write_b32 a[34], 0
v_accvgpr_write_b32 a[35], 0

	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_write_b32 a[36], 0
v_accvgpr_write_b32 a[37], 0
v_accvgpr_write_b32 a[38], 0
v_accvgpr_write_b32 a[39], 0

	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_write_b32 a[40], 0
v_accvgpr_write_b32 a[41], 0
v_accvgpr_write_b32 a[42], 0
v_accvgpr_write_b32 a[43], 0

	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_write_b32 a[44], 0
v_accvgpr_write_b32 a[45], 0
v_accvgpr_write_b32 a[46], 0
v_accvgpr_write_b32 a[47], 0

	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_write_b32 a[48], 0
v_accvgpr_write_b32 a[49], 0
v_accvgpr_write_b32 a[50], 0
v_accvgpr_write_b32 a[51], 0

	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_write_b32 a[52], 0
v_accvgpr_write_b32 a[53], 0
v_accvgpr_write_b32 a[54], 0
v_accvgpr_write_b32 a[55], 0

	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_write_b32 a[56], 0
v_accvgpr_write_b32 a[57], 0
v_accvgpr_write_b32 a[58], 0
v_accvgpr_write_b32 a[59], 0

	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_write_b32 a[60], 0
v_accvgpr_write_b32 a[61], 0
v_accvgpr_write_b32 a[62], 0
v_accvgpr_write_b32 a[63], 0

	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_write_b32 a[64], 0
v_accvgpr_write_b32 a[0x41], 0
v_accvgpr_write_b32 a[0x42], 0
v_accvgpr_write_b32 a[0x43], 0

	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_write_b32 a[0x44], 0
v_accvgpr_write_b32 a[0x45], 0
v_accvgpr_write_b32 a[0x46], 0
v_accvgpr_write_b32 a[0x47], 0

	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_write_b32 a[0x48], 0
v_accvgpr_write_b32 a[0x49], 0
v_accvgpr_write_b32 a[0x4a], 0
v_accvgpr_write_b32 a[0x4b], 0

	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_write_b32 a[0x4c], 0
v_accvgpr_write_b32 a[0x4d], 0
v_accvgpr_write_b32 a[0x4e], 0
v_accvgpr_write_b32 a[0x4f], 0

	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_write_b32 a[0x50], 0
v_accvgpr_write_b32 a[0x51], 0
v_accvgpr_write_b32 a[0x52], 0
v_accvgpr_write_b32 a[0x53], 0

	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_write_b32 a[0x54], 0
v_accvgpr_write_b32 a[0x55], 0
v_accvgpr_write_b32 a[0x56], 0
v_accvgpr_write_b32 a[0x57], 0

	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_write_b32 a[0x58], 0
v_accvgpr_write_b32 a[0x59], 0
v_accvgpr_write_b32 a[0x5a], 0
v_accvgpr_write_b32 a[0x5b], 0

	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_write_b32 a[0x5c], 0
v_accvgpr_write_b32 a[0x5d], 0
v_accvgpr_write_b32 a[0x5e], 0
v_accvgpr_write_b32 a[0x5f], 0

	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_write_b32 a[0x60], 0
v_accvgpr_write_b32 a[0x61], 0
v_accvgpr_write_b32 a[0x62], 0
v_accvgpr_write_b32 a[0x63], 0

	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_write_b32 a[0x64], 0
v_accvgpr_write_b32 a[0x65], 0
v_accvgpr_write_b32 a[0x66], 0
v_accvgpr_write_b32 a[0x67], 0

	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_write_b32 a[0x68], 0
v_accvgpr_write_b32 a[0x69], 0
v_accvgpr_write_b32 a[0x6a], 0
v_accvgpr_write_b32 a[0x6b], 0

	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_write_b32 a[0x6c], 0
v_accvgpr_write_b32 a[0x6d], 0
v_accvgpr_write_b32 a[0x6e], 0
v_accvgpr_write_b32 a[0x6f], 0

	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_write_b32 a[0x70], 0
v_accvgpr_write_b32 a[0x71], 0
v_accvgpr_write_b32 a[0x72], 0
v_accvgpr_write_b32 a[0x73], 0

	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_write_b32 a[0x74], 0
v_accvgpr_write_b32 a[0x75], 0
v_accvgpr_write_b32 a[0x76], 0
v_accvgpr_write_b32 a[0x77], 0

	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_write_b32 a[0x78], 0
v_accvgpr_write_b32 a[0x79], 0
v_accvgpr_write_b32 a[0x7a], 0
v_accvgpr_write_b32 a[0x7b], 0

	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_write_b32 a[0x7c], 0
v_accvgpr_write_b32 a[0x7d], 0
v_accvgpr_write_b32 a[0x7e], 0
v_accvgpr_write_b32 a[0x7f], 0

	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_write_b32 a[0x80], 0
v_accvgpr_write_b32 a[0x81], 0
v_accvgpr_write_b32 a[0x82], 0
v_accvgpr_write_b32 a[0x83], 0

	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_write_b32 a[0x84], 0
v_accvgpr_write_b32 a[0x85], 0
v_accvgpr_write_b32 a[0x86], 0
v_accvgpr_write_b32 a[0x87], 0

	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_write_b32 a[0x88], 0
v_accvgpr_write_b32 a[0x89], 0
v_accvgpr_write_b32 a[0x8a], 0
v_accvgpr_write_b32 a[0x8b], 0

	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_write_b32 a[0x8c], 0
v_accvgpr_write_b32 a[0x8d], 0
v_accvgpr_write_b32 a[0x8e], 0
v_accvgpr_write_b32 a[0x8f], 0

	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_write_b32 a[0x90], 0
v_accvgpr_write_b32 a[0x91], 0
v_accvgpr_write_b32 a[0x92], 0
v_accvgpr_write_b32 a[0x93], 0

	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_write_b32 a[0x94], 0
v_accvgpr_write_b32 a[0x95], 0
v_accvgpr_write_b32 a[0x96], 0
v_accvgpr_write_b32 a[0x97], 0

	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_write_b32 a[0x98], 0
v_accvgpr_write_b32 a[0x99], 0
v_accvgpr_write_b32 a[0x9a], 0
v_accvgpr_write_b32 a[0x9b], 0

	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_write_b32 a[0x9c], 0
v_accvgpr_write_b32 a[0x9d], 0
v_accvgpr_write_b32 a[0x9e], 0
v_accvgpr_write_b32 a[0x9f], 0

	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_write_b32 a[0xa0], 0
v_accvgpr_write_b32 a[0xa1], 0
v_accvgpr_write_b32 a[0xa2], 0
v_accvgpr_write_b32 a[0xa3], 0

	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_write_b32 a[0xa4], 0
v_accvgpr_write_b32 a[0xa5], 0
v_accvgpr_write_b32 a[0xa6], 0
v_accvgpr_write_b32 a[0xa7], 0

	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_write_b32 a[0xa8], 0
v_accvgpr_write_b32 a[0xa9], 0
v_accvgpr_write_b32 a[0xaa], 0
v_accvgpr_write_b32 a[0xab], 0

	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_write_b32 a[0xac], 0
v_accvgpr_write_b32 a[0xad], 0
v_accvgpr_write_b32 a[0xae], 0
v_accvgpr_write_b32 a[0xaf], 0

	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_write_b32 a[0xb0], 0
v_accvgpr_write_b32 a[0xb1], 0
v_accvgpr_write_b32 a[0xb2], 0
v_accvgpr_write_b32 a[0xb3], 0

	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_write_b32 a[0xb4], 0
v_accvgpr_write_b32 a[0xb5], 0
v_accvgpr_write_b32 a[0xb6], 0
v_accvgpr_write_b32 a[0xb7], 0

	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_write_b32 a[0xb8], 0
v_accvgpr_write_b32 a[0xb9], 0
v_accvgpr_write_b32 a[0xba], 0
v_accvgpr_write_b32 a[0xbb], 0

	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_write_b32 a[0xbc], 0
v_accvgpr_write_b32 a[0xbd], 0
v_accvgpr_write_b32 a[0xbe], 0
v_accvgpr_write_b32 a[0xbf], 0

	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_write_b32 a[0xc0], 0
v_accvgpr_write_b32 a[0xc1], 0
v_accvgpr_write_b32 a[0xc2], 0
v_accvgpr_write_b32 a[0xc3], 0

	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_write_b32 a[0xc4], 0
v_accvgpr_write_b32 a[0xc5], 0
v_accvgpr_write_b32 a[0xc6], 0
v_accvgpr_write_b32 a[0xc7], 0

	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_write_b32 a[0xc8], 0
v_accvgpr_write_b32 a[0xc9], 0
v_accvgpr_write_b32 a[0xca], 0
v_accvgpr_write_b32 a[0xcb], 0

	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_write_b32 a[0xcc], 0
v_accvgpr_write_b32 a[0xcd], 0
v_accvgpr_write_b32 a[0xce], 0
v_accvgpr_write_b32 a[0xcf], 0

	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_write_b32 a[0xd0], 0
v_accvgpr_write_b32 a[0xd1], 0
v_accvgpr_write_b32 a[0xd2], 0
v_accvgpr_write_b32 a[0xd3], 0

	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_write_b32 a[0xd4], 0
v_accvgpr_write_b32 a[0xd5], 0
v_accvgpr_write_b32 a[0xd6], 0
v_accvgpr_write_b32 a[0xd7], 0

	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_write_b32 a[0xd8], 0
v_accvgpr_write_b32 a[0xd9], 0
v_accvgpr_write_b32 a[0xda], 0
v_accvgpr_write_b32 a[0xdb], 0

	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_write_b32 a[0xdc], 0
v_accvgpr_write_b32 a[0xdd], 0
v_accvgpr_write_b32 a[0xde], 0
v_accvgpr_write_b32 a[0xdf], 0

	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_write_b32 a[0xe0], 0
v_accvgpr_write_b32 a[0xe1], 0
v_accvgpr_write_b32 a[0xe2], 0
v_accvgpr_write_b32 a[0xe3], 0

	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_write_b32 a[0xe4], 0
v_accvgpr_write_b32 a[0xe5], 0
v_accvgpr_write_b32 a[0xe6], 0
v_accvgpr_write_b32 a[0xe7], 0

	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_write_b32 a[0xe8], 0
v_accvgpr_write_b32 a[0xe9], 0
v_accvgpr_write_b32 a[0xea], 0
v_accvgpr_write_b32 a[0xeb], 0

	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_write_b32 a[0xec], 0
v_accvgpr_write_b32 a[0xed], 0
v_accvgpr_write_b32 a[0xee], 0
v_accvgpr_write_b32 a[0xef], 0

	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_write_b32 a[0xf0], 0
v_accvgpr_write_b32 a[0xf1], 0
v_accvgpr_write_b32 a[0xf2], 0
v_accvgpr_write_b32 a[0xf3], 0

	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_write_b32 a[0xf4], 0
v_accvgpr_write_b32 a[0xf5], 0
v_accvgpr_write_b32 a[0xf6], 0
v_accvgpr_write_b32 a[0xf7], 0

	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_write_b32 a[0xf8], 0
v_accvgpr_write_b32 a[0xf9], 0
v_accvgpr_write_b32 a[0xfa], 0
v_accvgpr_write_b32 a[0xfb], 0

	;;#ASMEND
	;;#ASMSTART
	v_accvgpr_write_b32 a[0xfc], 0
v_accvgpr_write_b32 a[0xfd], 0
v_accvgpr_write_b32 a[0xfe], 0
v_accvgpr_write_b32 a[0xff], 0

	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	v_mov_b32_e32 v7, s27
	;;#ASMSTART
	v_mov_b32 v[29], v7
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_add_i32 s37, s29, 0x1000
	s_mov_b32 m0, s18
	s_mov_b32 s18, s37
	buffer_load_dwordx4 v2, s[12:15], s17 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_add_i32 s38, s29, 0x2000
	s_mov_b32 m0, s18
	s_mov_b32 s18, s38
	buffer_load_dwordx4 v1, s[12:15], s17 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_add_i32 s39, s29, 0x3000
	s_mov_b32 m0, s18
	s_mov_b32 s18, s39
	buffer_load_dwordx4 v3, s[12:15], s17 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s18
	s_mov_b32 s18, s34
	buffer_load_dwordx4 v4, s[12:15], s17 offen lds
	s_lshl_b32 s17, s28, 7
	s_add_i32 s54, s16, s17
	s_mov_b32 s17, s54
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_add_i32 s40, s34, 0x1000
	s_mov_b32 m0, s18
	s_mov_b32 s18, s40
	buffer_load_dwordx4 v2, s[12:15], s17 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_add_i32 s41, s34, 0x2000
	s_mov_b32 m0, s18
	s_mov_b32 s18, s41
	buffer_load_dwordx4 v1, s[12:15], s17 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_add_i32 s42, s34, 0x3000
	s_mov_b32 m0, s18
	s_mov_b32 s18, s42
	buffer_load_dwordx4 v3, s[12:15], s17 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s18
	s_bitset1_b32 s16, 7
	buffer_load_dwordx4 v4, s[12:15], s17 offen lds
	s_mov_b32 s17, s33
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_add_i32 s43, s29, 0x5000
	s_mov_b32 m0, s17
	s_mov_b32 s17, s43
	buffer_load_dwordx4 v2, s[12:15], s16 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_add_i32 s44, s29, 0x6000
	s_mov_b32 m0, s17
	s_mov_b32 s17, s44
	buffer_load_dwordx4 v1, s[12:15], s16 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_add_i32 s45, s29, 0x7000
	s_mov_b32 m0, s17
	s_mov_b32 s17, s45
	buffer_load_dwordx4 v3, s[12:15], s16 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s17
	s_mov_b32 s17, s35
	buffer_load_dwordx4 v4, s[12:15], s16 offen lds
	s_add_i32 s16, s54, 0x80
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_add_i32 s46, s34, 0x5000
	s_mov_b32 m0, s17
	s_mov_b32 s17, s46
	buffer_load_dwordx4 v2, s[12:15], s16 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_add_i32 s47, s34, 0x6000
	s_mov_b32 m0, s17
	s_mov_b32 s17, s47
	buffer_load_dwordx4 v1, s[12:15], s16 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_add_i32 s48, s34, 0x7000
	s_mov_b32 m0, s17
	s_mov_b32 s17, s48
	buffer_load_dwordx4 v3, s[12:15], s16 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s17
	s_mov_b32 s9, s19
	buffer_load_dwordx4 v4, s[12:15], s16 offen lds
	s_mov_b64 s[18:19], s[2:3]
	v_readfirstlane_b32 s20, v8
	v_readfirstlane_b32 s21, v9
	s_mov_b64 s[16:17], s[0:1]
	s_mov_b32 s16, s20
	s_mov_b32 s17, s21
	s_mov_b64 s[22:23], s[2:3]
	s_mov_b64 s[20:21], s[0:1]
	s_mov_b32 s20, s30
	v_lshlrev_b32_e32 v7, 13, v11
	v_and_b32_e32 v8, 48, v0
	v_lshlrev_b32_e32 v9, 7, v0
	s_movk_i32 s30, 0x780
	v_and_or_b32 v9, v9, s30, v7
	v_or_b32_e32 v11, 64, v8
	v_lshlrev_b32_e32 v12, 3, v0
	v_or_b32_e32 v13, v9, v11
	v_bitop3_b32 v16, v12, v13, s53 bitop3:0x6c
	v_or_b32_e32 v13, 0x4000, v9
	v_or_b32_e32 v14, v13, v8
	v_or_b32_e32 v13, v13, v11
	v_or_b32_e32 v7, v9, v8
	v_bitop3_b32 v18, v12, v13, s53 bitop3:0x6c
	v_or_b32_e32 v13, 0x8000, v9
	v_or_b32_e32 v9, 0xc000, v9
	v_bitop3_b32 v17, v12, v14, s53 bitop3:0x6c
	v_or_b32_e32 v14, v13, v8
	v_or_b32_e32 v8, v9, v8
	v_bitop3_b32 v21, v12, v8, s53 bitop3:0x6c
	v_or_b32_e32 v8, v9, v11
	v_bitop3_b32 v22, v12, v8, s53 bitop3:0x6c
	v_lshlrev_b32_e32 v8, 12, v0
	s_mov_b32 s30, 0xf030
	v_bitop3_b32 v23, v8, s30, v0 bitop3:0xc8
	v_lshlrev_b32_e32 v8, 18, v10
	v_or_b32_e32 v13, v13, v11
	v_lshl_or_b32 v8, s50, 20, v8
	v_and_b32_e32 v5, 0x1f8, v12
	s_mov_b32 s21, s31
	v_bitop3_b32 v7, v12, v7, s53 bitop3:0x6c
	v_bitop3_b32 v19, v12, v14, s53 bitop3:0x6c
	v_bitop3_b32 v20, v12, v13, s53 bitop3:0x6c
	v_readfirstlane_b32 s30, v8
	s_mov_b32 s31, 0
	;;#ASMSTART
	buffer_load_dwordx2 v[8:9], v5, s[0:3], s31 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[10:11], v5, s[4:7], s31 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[12:13], v5, s[8:11], s31 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[14:15], v5, s[16:19], s31 offen
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	buffer_load_dwordx4 v[62:0x41], v23, s[20:23], s30 offen offset:0
	;;#ASMEND
	s_add_i32 s31, s30, 64
	;;#ASMSTART
	buffer_load_dwordx4 v[0x42:0x45], v23, s[20:23], s31 offen offset:0
	;;#ASMEND
	s_add_i32 s31, s30, 0x10000
	;;#ASMSTART
	buffer_load_dwordx4 v[0x46:0x49], v23, s[20:23], s31 offen offset:0
	;;#ASMEND
	s_add_i32 s31, s30, 0x10040
	;;#ASMSTART
	buffer_load_dwordx4 v[0x4a:0x4d], v23, s[20:23], s31 offen offset:0
	;;#ASMEND
	s_add_i32 s31, s30, 0x20000
	;;#ASMSTART
	buffer_load_dwordx4 v[0x4e:0x51], v23, s[20:23], s31 offen offset:0
	;;#ASMEND
	s_add_i32 s31, s30, 0x20040
	;;#ASMSTART
	buffer_load_dwordx4 v[0x52:0x55], v23, s[20:23], s31 offen offset:0
	;;#ASMEND
	s_add_i32 s31, s30, 0x30000
	;;#ASMSTART
	buffer_load_dwordx4 v[0x56:0x59], v23, s[20:23], s31 offen offset:0
	;;#ASMEND
	s_add_i32 s31, s30, 0x30040
	;;#ASMSTART
	buffer_load_dwordx4 v[0x5a:0x5d], v23, s[20:23], s31 offen offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[30:33], v7 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[34:37], v16 offset:0
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[38:41], v7 offset:0x800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[42:45], v16 offset:0x800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[46:49], v7 offset:0x1000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[50:53], v16 offset:0x1000
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[54:57], v7 offset:0x1800
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[58:61], v16 offset:0x1800
	;;#ASMEND
	s_lshl_b32 s52, s52, 8
	s_lshl_b32 s51, s51, 8
	;;#ASMSTART
	s_waitcnt vmcnt(0) lgkmcnt(0)
	;;#ASMEND
	s_mul_i32 s49, s49, s28
	s_sub_i32 s51, s52, s51
	s_lshl_b32 s31, s49, 8
	s_bitset1_b32 s51, 7
	s_mov_b32 s36, 0
	s_add_i32 s49, s31, 0x100
	s_add_i32 s50, s54, 0x100
	s_mul_i32 s28, s51, s28
	s_mov_b32 s51, 16
	s_movk_i32 s52, 0x1e00
.LBB0_7:                                ; %.preheader.preheader
                                        ; =>This Inner Loop Header: Depth=1
	s_add_i32 s53, s30, s36
	s_add_i32 s75, s53, 0x80000
	;;#ASMSTART
	v_mov_b32 v[166], v8
v_mov_b32 v[167], v9
v_mov_b32 v[168], v10
v_mov_b32 v[169], v11
v_mov_b32 v[158], v12
v_mov_b32 v[159], v13
v_mov_b32 v[160], v14
v_mov_b32 v[161], v15

	;;#ASMEND
	s_add_i32 s57, s52, 0xffffe400
	s_add_i32 s54, s31, s36
	;;#ASMSTART
	buffer_load_dwordx2 v[8:9], v5, s[0:3], s57 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[10:11], v5, s[4:7], s57 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[12:13], v5, s[8:11], s57 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[14:15], v5, s[16:19], s57 offen
	;;#ASMEND
	s_add_i32 s77, s75, 64
	s_add_i32 s82, s75, 0x10000
	s_add_i32 s83, s75, 0x10040
	s_add_i32 s85, s75, 0x20000
	s_add_i32 s89, s75, 0x20040
	s_add_i32 s91, s75, 0x30000
	s_add_i32 s92, s75, 0x30040
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[62:65], v[30:33], a[0:3], v[166], v[158] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[94:97], v19 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[62:65], v[38:41], a[4:7], v[166], v[158] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[98:101], v20 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[62:65], v[46:49], a[8:11], v[166], v[159] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[102:105], v19 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[62:65], v[54:57], a[12:15], v[166], v[159] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[106:109], v20 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[66:69], v[34:37], a[0:3], v[166], v[158] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[110:113], v19 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[66:69], v[42:45], a[4:7], v[166], v[158] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[114:117], v20 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[66:69], v[50:53], a[8:11], v[166], v[159] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[118:121], v19 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:69], v[58:61], a[12:15], v[166], v[159] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[122:125], v20 offset:6144
buffer_load_dwordx4 v[126:129], v23, s[20:23], s75 offen
buffer_load_dwordx4 v[130:133], v23, s[20:23], s77 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[70:73], v[30:33], a[16:19], v[166], v[158] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[70:73], v[38:41], a[20:23], v[166], v[158] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[70:73], v[46:49], a[24:27], v[166], v[159] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[70:73], v[54:57], a[28:31], v[166], v[159] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[74:77], v[34:37], a[16:19], v[166], v[158] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[74:77], v[42:45], a[20:23], v[166], v[158] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[134:137], v23, s[20:23], s82 offen
buffer_load_dwordx4 v[138:141], v23, s[20:23], s83 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[74:77], v[50:53], a[24:27], v[166], v[159] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[74:77], v[58:61], a[28:31], v[166], v[159] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[78:81], v[30:33], a[32:35], v[167], v[158] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[78:81], v[38:41], a[36:39], v[167], v[158] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[78:81], v[46:49], a[40:43], v[167], v[159] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[78:81], v[54:57], a[44:47], v[167], v[159] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[142:145], v23, s[20:23], s85 offen
buffer_load_dwordx4 v[146:149], v23, s[20:23], s89 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[82:85], v[34:37], a[32:35], v[167], v[158] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[82:85], v[42:45], a[36:39], v[167], v[158] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[82:85], v[50:53], a[40:43], v[167], v[159] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[82:85], v[58:61], a[44:47], v[167], v[159] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[86:89], v[30:33], a[48:51], v[167], v[158] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[86:89], v[38:41], a[52:55], v[167], v[158] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[150:153], v23, s[20:23], s91 offen
buffer_load_dwordx4 v[154:157], v23, s[20:23], s92 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[86:89], v[46:49], a[56:59], v[167], v[159] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[86:89], v[54:57], a[60:63], v[167], v[159] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[90:93], v[34:37], a[48:51], v[167], v[158] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[90:93], v[42:45], a[52:55], v[167], v[158] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[90:93], v[50:53], a[56:59], v[167], v[159] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[90:93], v[58:61], a[60:63], v[167], v[159] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[62:65], v[94:97], a[64:67], v[166], v[160] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[62:65], v[102:105], a[68:71], v[166], v[160] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[62:65], v[110:113], a[72:75], v[166], v[161] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[62:65], v[118:121], a[76:79], v[166], v[161] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[66:69], v[98:101], a[64:67], v[166], v[160] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[66:69], v[106:109], a[68:71], v[166], v[160] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[66:69], v[114:117], a[72:75], v[166], v[161] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[66:69], v[122:125], a[76:79], v[166], v[161] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[70:73], v[94:97], a[80:83], v[166], v[160] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[70:73], v[102:105], a[84:87], v[166], v[160] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[70:73], v[110:113], a[88:91], v[166], v[161] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[70:73], v[118:121], a[92:95], v[166], v[161] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[74:77], v[98:101], a[80:83], v[166], v[160] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[74:77], v[106:109], a[84:87], v[166], v[160] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[74:77], v[114:117], a[88:91], v[166], v[161] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[74:77], v[122:125], a[92:95], v[166], v[161] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[78:81], v[94:97], a[96:99], v[167], v[160] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[78:81], v[102:105], a[100:103], v[167], v[160] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[78:81], v[110:113], a[104:107], v[167], v[161] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[78:81], v[118:121], a[108:111], v[167], v[161] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[82:85], v[98:101], a[96:99], v[167], v[160] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[82:85], v[106:109], a[100:103], v[167], v[160] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[82:85], v[114:117], a[104:107], v[167], v[161] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[82:85], v[122:125], a[108:111], v[167], v[161] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[86:89], v[94:97], a[112:115], v[167], v[160] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[86:89], v[102:105], a[116:119], v[167], v[160] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[86:89], v[110:113], a[120:123], v[167], v[161] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[86:89], v[118:121], a[124:127], v[167], v[161] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[90:93], v[98:101], a[112:115], v[167], v[160] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[90:93], v[106:109], a[116:119], v[167], v[160] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[90:93], v[114:117], a[120:123], v[167], v[161] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[90:93], v[122:125], a[124:127], v[167], v[161] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_mov_b32 m0, s29
	s_add_i32 s72, s54, 0x100
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	buffer_load_dwordx4 v2, s[12:15], s72 offen lds
	s_mov_b32 m0, s37
	s_add_i32 s55, s28, s36
	buffer_load_dwordx4 v1, s[12:15], s72 offen lds
	s_mov_b32 m0, s38
	s_add_i32 s74, s55, 0x100
	buffer_load_dwordx4 v3, s[12:15], s72 offen lds
	s_mov_b32 m0, s39
	s_add_i32 s76, s53, 0x80
	buffer_load_dwordx4 v4, s[12:15], s72 offen lds
	s_mov_b32 m0, s34
	s_add_i32 s56, s52, 0xffffe600
	buffer_load_dwordx4 v2, s[12:15], s74 offen lds
	s_mov_b32 m0, s40
	s_add_i32 s80, s53, 0x80080
	buffer_load_dwordx4 v1, s[12:15], s74 offen lds
	s_mov_b32 m0, s41
	s_add_i32 s75, s76, 64
	buffer_load_dwordx4 v3, s[12:15], s74 offen lds
	s_mov_b32 m0, s42
	s_add_i32 s82, s76, 0x10000
	buffer_load_dwordx4 v4, s[12:15], s74 offen lds
	s_add_i32 s85, s76, 0x10040
	s_add_i32 s89, s76, 0x20000
	s_add_i32 s91, s76, 0x20040
	s_add_i32 s72, s76, 0x30000
	s_add_i32 s74, s76, 0x30040
	;;#ASMSTART
	s_waitcnt vmcnt(0)
s_barrier
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[126:129], v[30:33], a[128:131], v[168], v[158] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[126:129], v[38:41], a[132:135], v[168], v[158] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[62:65], v23, s[20:23], s76 offen
buffer_load_dwordx4 v[66:69], v23, s[20:23], s75 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[126:129], v[46:49], a[136:139], v[168], v[159] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[126:129], v[54:57], a[140:143], v[168], v[159] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[130:133], v[34:37], a[128:131], v[168], v[158] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[130:133], v[42:45], a[132:135], v[168], v[158] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[70:73], v23, s[20:23], s82 offen
buffer_load_dwordx4 v[74:77], v23, s[20:23], s85 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[130:133], v[50:53], a[136:139], v[168], v[159] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[130:133], v[58:61], a[140:143], v[168], v[159] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[134:137], v[30:33], a[144:147], v[168], v[158] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[134:137], v[38:41], a[148:151], v[168], v[158] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[78:81], v23, s[20:23], s89 offen
buffer_load_dwordx4 v[82:85], v23, s[20:23], s91 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[134:137], v[46:49], a[152:155], v[168], v[159] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[134:137], v[54:57], a[156:159], v[168], v[159] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[138:141], v[34:37], a[144:147], v[168], v[158] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[138:141], v[42:45], a[148:151], v[168], v[158] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[86:89], v23, s[20:23], s72 offen
buffer_load_dwordx4 v[90:93], v23, s[20:23], s74 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[138:141], v[50:53], a[152:155], v[168], v[159] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[138:141], v[58:61], a[156:159], v[168], v[159] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[142:145], v[30:33], a[160:163], v[169], v[158] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[142:145], v[38:41], a[164:167], v[169], v[158] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[142:145], v[46:49], a[168:171], v[169], v[159] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[142:145], v[54:57], a[172:175], v[169], v[159] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[146:149], v[34:37], a[160:163], v[169], v[158] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[146:149], v[42:45], a[164:167], v[169], v[158] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[146:149], v[50:53], a[168:171], v[169], v[159] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[146:149], v[58:61], a[172:175], v[169], v[159] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[150:153], v[30:33], a[176:179], v[169], v[158] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[150:153], v[38:41], a[180:183], v[169], v[158] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[150:153], v[46:49], a[184:187], v[169], v[159] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[150:153], v[54:57], a[188:191], v[169], v[159] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[154:157], v[34:37], a[176:179], v[169], v[158] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[154:157], v[42:45], a[180:183], v[169], v[158] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[154:157], v[50:53], a[184:187], v[169], v[159] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[154:157], v[58:61], a[188:191], v[169], v[159] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[126:129], v[94:97], a[192:195], v[168], v[160] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v17 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[126:129], v[102:105], a[196:199], v[168], v[160] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v18 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[126:129], v[110:113], a[200:203], v[168], v[161] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v17 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[126:129], v[118:121], a[204:207], v[168], v[161] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v18 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[130:133], v[98:101], a[192:195], v[168], v[160] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v17 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[130:133], v[106:109], a[196:199], v[168], v[160] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v18 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[130:133], v[114:117], a[200:203], v[168], v[161] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v17 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[130:133], v[122:125], a[204:207], v[168], v[161] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v18 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[134:137], v[94:97], a[208:211], v[168], v[160] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[134:137], v[102:105], a[212:215], v[168], v[160] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[134:137], v[110:113], a[216:219], v[168], v[161] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[134:137], v[118:121], a[220:223], v[168], v[161] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[138:141], v[98:101], a[208:211], v[168], v[160] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[138:141], v[106:109], a[212:215], v[168], v[160] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[138:141], v[114:117], a[216:219], v[168], v[161] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[138:141], v[122:125], a[220:223], v[168], v[161] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[142:145], v[94:97], a[224:227], v[169], v[160] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[142:145], v[102:105], a[228:231], v[169], v[160] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[142:145], v[110:113], a[232:235], v[169], v[161] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[142:145], v[118:121], a[236:239], v[169], v[161] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[146:149], v[98:101], a[224:227], v[169], v[160] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[146:149], v[106:109], a[228:231], v[169], v[160] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[146:149], v[114:117], a[232:235], v[169], v[161] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[146:149], v[122:125], a[236:239], v[169], v[161] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[150:153], v[94:97], a[240:243], v[169], v[160] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[150:153], v[102:105], a[244:247], v[169], v[160] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[150:153], v[110:113], a[248:251], v[169], v[161] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[150:153], v[118:121], a[252:255], v[169], v[161] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[154:157], v[98:101], a[240:243], v[169], v[160] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[154:157], v[106:109], a[244:247], v[169], v[160] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[154:157], v[114:117], a[248:251], v[169], v[161] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[154:157], v[122:125], a[252:255], v[169], v[161] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_i32 s72, s80, 64
	s_add_i32 s74, s80, 0x10000
	s_add_i32 s75, s80, 0x10040
	s_add_i32 s76, s80, 0x20000
	s_add_i32 s82, s80, 0x20040
	s_add_i32 s85, s80, 0x30000
	;;#ASMSTART
	s_waitcnt vmcnt(0) lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v[166], v8
v_mov_b32 v[167], v9
v_mov_b32 v[168], v10
v_mov_b32 v[169], v11
v_mov_b32 v[158], v12
v_mov_b32 v[159], v13
v_mov_b32 v[160], v14
v_mov_b32 v[161], v15

	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[8:9], v5, s[0:3], s56 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[10:11], v5, s[4:7], s56 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[12:13], v5, s[8:11], s56 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[14:15], v5, s[16:19], s56 offen
	;;#ASMEND
	s_add_i32 s56, s80, 0x30040
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[62:65], v[30:33], a[0:3], v[166], v[158] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[94:97], v21 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[62:65], v[38:41], a[4:7], v[166], v[158] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[98:101], v22 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[62:65], v[46:49], a[8:11], v[166], v[159] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[102:105], v21 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[62:65], v[54:57], a[12:15], v[166], v[159] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[106:109], v22 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[66:69], v[34:37], a[0:3], v[166], v[158] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[110:113], v21 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[66:69], v[42:45], a[4:7], v[166], v[158] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[114:117], v22 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[66:69], v[50:53], a[8:11], v[166], v[159] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[118:121], v21 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:69], v[58:61], a[12:15], v[166], v[159] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[122:125], v22 offset:6144
buffer_load_dwordx4 v[126:129], v23, s[20:23], s80 offen
buffer_load_dwordx4 v[130:133], v23, s[20:23], s72 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[70:73], v[30:33], a[16:19], v[166], v[158] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[70:73], v[38:41], a[20:23], v[166], v[158] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[70:73], v[46:49], a[24:27], v[166], v[159] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[70:73], v[54:57], a[28:31], v[166], v[159] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[74:77], v[34:37], a[16:19], v[166], v[158] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[74:77], v[42:45], a[20:23], v[166], v[158] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[134:137], v23, s[20:23], s74 offen
buffer_load_dwordx4 v[138:141], v23, s[20:23], s75 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[74:77], v[50:53], a[24:27], v[166], v[159] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[74:77], v[58:61], a[28:31], v[166], v[159] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[78:81], v[30:33], a[32:35], v[167], v[158] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[78:81], v[38:41], a[36:39], v[167], v[158] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[78:81], v[46:49], a[40:43], v[167], v[159] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[78:81], v[54:57], a[44:47], v[167], v[159] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[142:145], v23, s[20:23], s76 offen
buffer_load_dwordx4 v[146:149], v23, s[20:23], s82 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[82:85], v[34:37], a[32:35], v[167], v[158] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[82:85], v[42:45], a[36:39], v[167], v[158] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[82:85], v[50:53], a[40:43], v[167], v[159] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[82:85], v[58:61], a[44:47], v[167], v[159] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[86:89], v[30:33], a[48:51], v[167], v[158] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[86:89], v[38:41], a[52:55], v[167], v[158] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[150:153], v23, s[20:23], s85 offen
buffer_load_dwordx4 v[154:157], v23, s[20:23], s56 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[86:89], v[46:49], a[56:59], v[167], v[159] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[86:89], v[54:57], a[60:63], v[167], v[159] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[90:93], v[34:37], a[48:51], v[167], v[158] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[90:93], v[42:45], a[52:55], v[167], v[158] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[90:93], v[50:53], a[56:59], v[167], v[159] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[90:93], v[58:61], a[60:63], v[167], v[159] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[62:65], v[94:97], a[64:67], v[166], v[160] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[62:65], v[102:105], a[68:71], v[166], v[160] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[62:65], v[110:113], a[72:75], v[166], v[161] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[62:65], v[118:121], a[76:79], v[166], v[161] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[66:69], v[98:101], a[64:67], v[166], v[160] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[66:69], v[106:109], a[68:71], v[166], v[160] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[66:69], v[114:117], a[72:75], v[166], v[161] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[66:69], v[122:125], a[76:79], v[166], v[161] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[70:73], v[94:97], a[80:83], v[166], v[160] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[70:73], v[102:105], a[84:87], v[166], v[160] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[70:73], v[110:113], a[88:91], v[166], v[161] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[70:73], v[118:121], a[92:95], v[166], v[161] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[74:77], v[98:101], a[80:83], v[166], v[160] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[74:77], v[106:109], a[84:87], v[166], v[160] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[74:77], v[114:117], a[88:91], v[166], v[161] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[74:77], v[122:125], a[92:95], v[166], v[161] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[78:81], v[94:97], a[96:99], v[167], v[160] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[78:81], v[102:105], a[100:103], v[167], v[160] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[78:81], v[110:113], a[104:107], v[167], v[161] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[78:81], v[118:121], a[108:111], v[167], v[161] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[82:85], v[98:101], a[96:99], v[167], v[160] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[82:85], v[106:109], a[100:103], v[167], v[160] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[82:85], v[114:117], a[104:107], v[167], v[161] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[82:85], v[122:125], a[108:111], v[167], v[161] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[86:89], v[94:97], a[112:115], v[167], v[160] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[86:89], v[102:105], a[116:119], v[167], v[160] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[86:89], v[110:113], a[120:123], v[167], v[161] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[86:89], v[118:121], a[124:127], v[167], v[161] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[90:93], v[98:101], a[112:115], v[167], v[160] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[90:93], v[106:109], a[116:119], v[167], v[160] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[90:93], v[114:117], a[120:123], v[167], v[161] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[90:93], v[122:125], a[124:127], v[167], v[161] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_i32 s78, s54, 0x180
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_mov_b32 m0, s33
	s_add_i32 s79, s55, 0x180
	buffer_load_dwordx4 v2, s[12:15], s78 offen lds
	s_mov_b32 m0, s43
	s_add_i32 s81, s53, 0x100
	buffer_load_dwordx4 v1, s[12:15], s78 offen lds
	s_mov_b32 m0, s44
	s_add_i32 s58, s52, 0xffffe800
	buffer_load_dwordx4 v3, s[12:15], s78 offen lds
	s_mov_b32 m0, s45
	s_add_i32 s84, s53, 0x80100
	buffer_load_dwordx4 v4, s[12:15], s78 offen lds
	s_mov_b32 m0, s35
	s_add_i32 s56, s81, 64
	buffer_load_dwordx4 v2, s[12:15], s79 offen lds
	s_mov_b32 m0, s46
	s_add_i32 s72, s81, 0x10000
	buffer_load_dwordx4 v1, s[12:15], s79 offen lds
	s_mov_b32 m0, s47
	s_add_i32 s74, s81, 0x10040
	buffer_load_dwordx4 v3, s[12:15], s79 offen lds
	s_mov_b32 m0, s48
	s_add_i32 s75, s81, 0x20000
	buffer_load_dwordx4 v4, s[12:15], s79 offen lds
	s_add_i32 s76, s81, 0x20040
	s_add_i32 s78, s81, 0x30000
	s_add_i32 s79, s81, 0x30040
	;;#ASMSTART
	s_waitcnt vmcnt(0)
s_barrier
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[126:129], v[30:33], a[128:131], v[168], v[158] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[126:129], v[38:41], a[132:135], v[168], v[158] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[62:65], v23, s[20:23], s81 offen
buffer_load_dwordx4 v[66:69], v23, s[20:23], s56 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[126:129], v[46:49], a[136:139], v[168], v[159] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[126:129], v[54:57], a[140:143], v[168], v[159] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[130:133], v[34:37], a[128:131], v[168], v[158] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[130:133], v[42:45], a[132:135], v[168], v[158] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[70:73], v23, s[20:23], s72 offen
buffer_load_dwordx4 v[74:77], v23, s[20:23], s74 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[130:133], v[50:53], a[136:139], v[168], v[159] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[130:133], v[58:61], a[140:143], v[168], v[159] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[134:137], v[30:33], a[144:147], v[168], v[158] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[134:137], v[38:41], a[148:151], v[168], v[158] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[78:81], v23, s[20:23], s75 offen
buffer_load_dwordx4 v[82:85], v23, s[20:23], s76 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[134:137], v[46:49], a[152:155], v[168], v[159] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[134:137], v[54:57], a[156:159], v[168], v[159] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[138:141], v[34:37], a[144:147], v[168], v[158] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[138:141], v[42:45], a[148:151], v[168], v[158] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[86:89], v23, s[20:23], s78 offen
buffer_load_dwordx4 v[90:93], v23, s[20:23], s79 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[138:141], v[50:53], a[152:155], v[168], v[159] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[138:141], v[58:61], a[156:159], v[168], v[159] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[142:145], v[30:33], a[160:163], v[169], v[158] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[142:145], v[38:41], a[164:167], v[169], v[158] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[142:145], v[46:49], a[168:171], v[169], v[159] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[142:145], v[54:57], a[172:175], v[169], v[159] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[146:149], v[34:37], a[160:163], v[169], v[158] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[146:149], v[42:45], a[164:167], v[169], v[158] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[146:149], v[50:53], a[168:171], v[169], v[159] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[146:149], v[58:61], a[172:175], v[169], v[159] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[150:153], v[30:33], a[176:179], v[169], v[158] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[150:153], v[38:41], a[180:183], v[169], v[158] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[150:153], v[46:49], a[184:187], v[169], v[159] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[150:153], v[54:57], a[188:191], v[169], v[159] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[154:157], v[34:37], a[176:179], v[169], v[158] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[154:157], v[42:45], a[180:183], v[169], v[158] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[154:157], v[50:53], a[184:187], v[169], v[159] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[154:157], v[58:61], a[188:191], v[169], v[159] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[126:129], v[94:97], a[192:195], v[168], v[160] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v7 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[126:129], v[102:105], a[196:199], v[168], v[160] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v16 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[126:129], v[110:113], a[200:203], v[168], v[161] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v7 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[126:129], v[118:121], a[204:207], v[168], v[161] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v16 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[130:133], v[98:101], a[192:195], v[168], v[160] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v7 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[130:133], v[106:109], a[196:199], v[168], v[160] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v16 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[130:133], v[114:117], a[200:203], v[168], v[161] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v7 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[130:133], v[122:125], a[204:207], v[168], v[161] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v16 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[134:137], v[94:97], a[208:211], v[168], v[160] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[134:137], v[102:105], a[212:215], v[168], v[160] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[134:137], v[110:113], a[216:219], v[168], v[161] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[134:137], v[118:121], a[220:223], v[168], v[161] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[138:141], v[98:101], a[208:211], v[168], v[160] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[138:141], v[106:109], a[212:215], v[168], v[160] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[138:141], v[114:117], a[216:219], v[168], v[161] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[138:141], v[122:125], a[220:223], v[168], v[161] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[142:145], v[94:97], a[224:227], v[169], v[160] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[142:145], v[102:105], a[228:231], v[169], v[160] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[142:145], v[110:113], a[232:235], v[169], v[161] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[142:145], v[118:121], a[236:239], v[169], v[161] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[146:149], v[98:101], a[224:227], v[169], v[160] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[146:149], v[106:109], a[228:231], v[169], v[160] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[146:149], v[114:117], a[232:235], v[169], v[161] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[146:149], v[122:125], a[236:239], v[169], v[161] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[150:153], v[94:97], a[240:243], v[169], v[160] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[150:153], v[102:105], a[244:247], v[169], v[160] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[150:153], v[110:113], a[248:251], v[169], v[161] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[150:153], v[118:121], a[252:255], v[169], v[161] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[154:157], v[98:101], a[240:243], v[169], v[160] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[154:157], v[106:109], a[244:247], v[169], v[160] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[154:157], v[114:117], a[248:251], v[169], v[161] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[154:157], v[122:125], a[252:255], v[169], v[161] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_i32 s56, s84, 64
	s_add_i32 s72, s84, 0x10000
	s_add_i32 s74, s84, 0x10040
	s_add_i32 s75, s84, 0x20000
	s_add_i32 s76, s84, 0x20040
	s_add_i32 s78, s84, 0x30000
	;;#ASMSTART
	s_waitcnt vmcnt(0) lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v[166], v8
v_mov_b32 v[167], v9
v_mov_b32 v[168], v10
v_mov_b32 v[169], v11
v_mov_b32 v[158], v12
v_mov_b32 v[159], v13
v_mov_b32 v[160], v14
v_mov_b32 v[161], v15

	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[8:9], v5, s[0:3], s58 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[10:11], v5, s[4:7], s58 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[12:13], v5, s[8:11], s58 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[14:15], v5, s[16:19], s58 offen
	;;#ASMEND
	s_add_i32 s58, s84, 0x30040
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[62:65], v[30:33], a[0:3], v[166], v[158] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[94:97], v19 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[62:65], v[38:41], a[4:7], v[166], v[158] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[98:101], v20 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[62:65], v[46:49], a[8:11], v[166], v[159] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[102:105], v19 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[62:65], v[54:57], a[12:15], v[166], v[159] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[106:109], v20 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[66:69], v[34:37], a[0:3], v[166], v[158] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[110:113], v19 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[66:69], v[42:45], a[4:7], v[166], v[158] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[114:117], v20 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[66:69], v[50:53], a[8:11], v[166], v[159] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[118:121], v19 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:69], v[58:61], a[12:15], v[166], v[159] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[122:125], v20 offset:6144
buffer_load_dwordx4 v[126:129], v23, s[20:23], s84 offen
buffer_load_dwordx4 v[130:133], v23, s[20:23], s56 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[70:73], v[30:33], a[16:19], v[166], v[158] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[70:73], v[38:41], a[20:23], v[166], v[158] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[70:73], v[46:49], a[24:27], v[166], v[159] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[70:73], v[54:57], a[28:31], v[166], v[159] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[74:77], v[34:37], a[16:19], v[166], v[158] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[74:77], v[42:45], a[20:23], v[166], v[158] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[134:137], v23, s[20:23], s72 offen
buffer_load_dwordx4 v[138:141], v23, s[20:23], s74 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[74:77], v[50:53], a[24:27], v[166], v[159] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[74:77], v[58:61], a[28:31], v[166], v[159] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[78:81], v[30:33], a[32:35], v[167], v[158] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[78:81], v[38:41], a[36:39], v[167], v[158] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[78:81], v[46:49], a[40:43], v[167], v[159] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[78:81], v[54:57], a[44:47], v[167], v[159] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[142:145], v23, s[20:23], s75 offen
buffer_load_dwordx4 v[146:149], v23, s[20:23], s76 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[82:85], v[34:37], a[32:35], v[167], v[158] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[82:85], v[42:45], a[36:39], v[167], v[158] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[82:85], v[50:53], a[40:43], v[167], v[159] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[82:85], v[58:61], a[44:47], v[167], v[159] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[86:89], v[30:33], a[48:51], v[167], v[158] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[86:89], v[38:41], a[52:55], v[167], v[158] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[150:153], v23, s[20:23], s78 offen
buffer_load_dwordx4 v[154:157], v23, s[20:23], s58 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[86:89], v[46:49], a[56:59], v[167], v[159] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[86:89], v[54:57], a[60:63], v[167], v[159] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[90:93], v[34:37], a[48:51], v[167], v[158] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[90:93], v[42:45], a[52:55], v[167], v[158] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[90:93], v[50:53], a[56:59], v[167], v[159] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[90:93], v[58:61], a[60:63], v[167], v[159] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[62:65], v[94:97], a[64:67], v[166], v[160] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[62:65], v[102:105], a[68:71], v[166], v[160] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[62:65], v[110:113], a[72:75], v[166], v[161] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[62:65], v[118:121], a[76:79], v[166], v[161] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[66:69], v[98:101], a[64:67], v[166], v[160] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[66:69], v[106:109], a[68:71], v[166], v[160] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[66:69], v[114:117], a[72:75], v[166], v[161] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[66:69], v[122:125], a[76:79], v[166], v[161] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[70:73], v[94:97], a[80:83], v[166], v[160] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[70:73], v[102:105], a[84:87], v[166], v[160] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[70:73], v[110:113], a[88:91], v[166], v[161] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[70:73], v[118:121], a[92:95], v[166], v[161] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[74:77], v[98:101], a[80:83], v[166], v[160] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[74:77], v[106:109], a[84:87], v[166], v[160] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[74:77], v[114:117], a[88:91], v[166], v[161] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[74:77], v[122:125], a[92:95], v[166], v[161] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[78:81], v[94:97], a[96:99], v[167], v[160] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[78:81], v[102:105], a[100:103], v[167], v[160] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[78:81], v[110:113], a[104:107], v[167], v[161] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[78:81], v[118:121], a[108:111], v[167], v[161] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[82:85], v[98:101], a[96:99], v[167], v[160] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[82:85], v[106:109], a[100:103], v[167], v[160] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[82:85], v[114:117], a[104:107], v[167], v[161] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[82:85], v[122:125], a[108:111], v[167], v[161] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[86:89], v[94:97], a[112:115], v[167], v[160] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[86:89], v[102:105], a[116:119], v[167], v[160] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[86:89], v[110:113], a[120:123], v[167], v[161] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[86:89], v[118:121], a[124:127], v[167], v[161] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[90:93], v[98:101], a[112:115], v[167], v[160] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[90:93], v[106:109], a[116:119], v[167], v[160] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[90:93], v[114:117], a[120:123], v[167], v[161] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[90:93], v[122:125], a[124:127], v[167], v[161] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_i32 s87, s54, 0x200
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_mov_b32 m0, s29
	s_add_i32 s88, s55, 0x200
	buffer_load_dwordx4 v2, s[12:15], s87 offen lds
	s_mov_b32 m0, s37
	s_add_i32 s90, s53, 0x180
	buffer_load_dwordx4 v1, s[12:15], s87 offen lds
	s_mov_b32 m0, s38
	s_add_i32 s59, s52, 0xffffea00
	buffer_load_dwordx4 v3, s[12:15], s87 offen lds
	s_mov_b32 m0, s39
	s_add_i32 vcc_hi, s53, 0x80180
	buffer_load_dwordx4 v4, s[12:15], s87 offen lds
	s_mov_b32 m0, s34
	s_add_i32 s56, s90, 64
	buffer_load_dwordx4 v2, s[12:15], s88 offen lds
	s_mov_b32 m0, s40
	s_add_i32 s58, s90, 0x10000
	buffer_load_dwordx4 v1, s[12:15], s88 offen lds
	s_mov_b32 m0, s41
	s_add_i32 s72, s90, 0x10040
	buffer_load_dwordx4 v3, s[12:15], s88 offen lds
	s_mov_b32 m0, s42
	s_add_i32 s74, s90, 0x20000
	buffer_load_dwordx4 v4, s[12:15], s88 offen lds
	s_add_i32 s75, s90, 0x20040
	s_add_i32 s76, s90, 0x30000
	s_add_i32 s78, s90, 0x30040
	;;#ASMSTART
	s_waitcnt vmcnt(0)
s_barrier
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[126:129], v[30:33], a[128:131], v[168], v[158] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[126:129], v[38:41], a[132:135], v[168], v[158] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[62:65], v23, s[20:23], s90 offen
buffer_load_dwordx4 v[66:69], v23, s[20:23], s56 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[126:129], v[46:49], a[136:139], v[168], v[159] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[126:129], v[54:57], a[140:143], v[168], v[159] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[130:133], v[34:37], a[128:131], v[168], v[158] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[130:133], v[42:45], a[132:135], v[168], v[158] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[70:73], v23, s[20:23], s58 offen
buffer_load_dwordx4 v[74:77], v23, s[20:23], s72 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[130:133], v[50:53], a[136:139], v[168], v[159] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[130:133], v[58:61], a[140:143], v[168], v[159] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[134:137], v[30:33], a[144:147], v[168], v[158] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[134:137], v[38:41], a[148:151], v[168], v[158] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[78:81], v23, s[20:23], s74 offen
buffer_load_dwordx4 v[82:85], v23, s[20:23], s75 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[134:137], v[46:49], a[152:155], v[168], v[159] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[134:137], v[54:57], a[156:159], v[168], v[159] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[138:141], v[34:37], a[144:147], v[168], v[158] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[138:141], v[42:45], a[148:151], v[168], v[158] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[86:89], v23, s[20:23], s76 offen
buffer_load_dwordx4 v[90:93], v23, s[20:23], s78 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[138:141], v[50:53], a[152:155], v[168], v[159] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[138:141], v[58:61], a[156:159], v[168], v[159] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[142:145], v[30:33], a[160:163], v[169], v[158] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[142:145], v[38:41], a[164:167], v[169], v[158] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[142:145], v[46:49], a[168:171], v[169], v[159] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[142:145], v[54:57], a[172:175], v[169], v[159] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[146:149], v[34:37], a[160:163], v[169], v[158] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[146:149], v[42:45], a[164:167], v[169], v[158] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[146:149], v[50:53], a[168:171], v[169], v[159] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[146:149], v[58:61], a[172:175], v[169], v[159] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[150:153], v[30:33], a[176:179], v[169], v[158] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[150:153], v[38:41], a[180:183], v[169], v[158] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[150:153], v[46:49], a[184:187], v[169], v[159] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[150:153], v[54:57], a[188:191], v[169], v[159] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[154:157], v[34:37], a[176:179], v[169], v[158] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[154:157], v[42:45], a[180:183], v[169], v[158] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[154:157], v[50:53], a[184:187], v[169], v[159] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[154:157], v[58:61], a[188:191], v[169], v[159] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[126:129], v[94:97], a[192:195], v[168], v[160] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v17 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[126:129], v[102:105], a[196:199], v[168], v[160] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v18 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[126:129], v[110:113], a[200:203], v[168], v[161] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v17 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[126:129], v[118:121], a[204:207], v[168], v[161] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v18 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[130:133], v[98:101], a[192:195], v[168], v[160] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v17 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[130:133], v[106:109], a[196:199], v[168], v[160] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v18 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[130:133], v[114:117], a[200:203], v[168], v[161] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v17 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[130:133], v[122:125], a[204:207], v[168], v[161] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v18 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[134:137], v[94:97], a[208:211], v[168], v[160] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[134:137], v[102:105], a[212:215], v[168], v[160] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[134:137], v[110:113], a[216:219], v[168], v[161] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[134:137], v[118:121], a[220:223], v[168], v[161] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[138:141], v[98:101], a[208:211], v[168], v[160] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[138:141], v[106:109], a[212:215], v[168], v[160] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[138:141], v[114:117], a[216:219], v[168], v[161] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[138:141], v[122:125], a[220:223], v[168], v[161] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[142:145], v[94:97], a[224:227], v[169], v[160] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[142:145], v[102:105], a[228:231], v[169], v[160] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[142:145], v[110:113], a[232:235], v[169], v[161] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[142:145], v[118:121], a[236:239], v[169], v[161] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[146:149], v[98:101], a[224:227], v[169], v[160] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[146:149], v[106:109], a[228:231], v[169], v[160] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[146:149], v[114:117], a[232:235], v[169], v[161] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[146:149], v[122:125], a[236:239], v[169], v[161] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[150:153], v[94:97], a[240:243], v[169], v[160] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[150:153], v[102:105], a[244:247], v[169], v[160] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[150:153], v[110:113], a[248:251], v[169], v[161] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[150:153], v[118:121], a[252:255], v[169], v[161] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[154:157], v[98:101], a[240:243], v[169], v[160] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[154:157], v[106:109], a[244:247], v[169], v[160] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[154:157], v[114:117], a[248:251], v[169], v[161] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[154:157], v[122:125], a[252:255], v[169], v[161] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_i32 s56, vcc_hi, 64
	s_add_i32 s58, vcc_hi, 0x10000
	s_add_i32 s72, vcc_hi, 0x10040
	s_add_i32 s74, vcc_hi, 0x20000
	s_add_i32 s75, vcc_hi, 0x20040
	s_add_i32 s78, vcc_hi, 0x30000
	;;#ASMSTART
	s_waitcnt vmcnt(0) lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v[166], v8
v_mov_b32 v[167], v9
v_mov_b32 v[168], v10
v_mov_b32 v[169], v11
v_mov_b32 v[158], v12
v_mov_b32 v[159], v13
v_mov_b32 v[160], v14
v_mov_b32 v[161], v15

	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[8:9], v5, s[0:3], s59 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[10:11], v5, s[4:7], s59 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[12:13], v5, s[8:11], s59 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[14:15], v5, s[16:19], s59 offen
	;;#ASMEND
	s_add_i32 s59, vcc_hi, 0x30040
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[62:65], v[30:33], a[0:3], v[166], v[158] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[94:97], v21 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[62:65], v[38:41], a[4:7], v[166], v[158] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[98:101], v22 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[62:65], v[46:49], a[8:11], v[166], v[159] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[102:105], v21 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[62:65], v[54:57], a[12:15], v[166], v[159] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[106:109], v22 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[66:69], v[34:37], a[0:3], v[166], v[158] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[110:113], v21 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[66:69], v[42:45], a[4:7], v[166], v[158] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[114:117], v22 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[66:69], v[50:53], a[8:11], v[166], v[159] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[118:121], v21 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:69], v[58:61], a[12:15], v[166], v[159] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[122:125], v22 offset:6144
buffer_load_dwordx4 v[126:129], v23, s[20:23], vcc_hi offen
buffer_load_dwordx4 v[130:133], v23, s[20:23], s56 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[70:73], v[30:33], a[16:19], v[166], v[158] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[70:73], v[38:41], a[20:23], v[166], v[158] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[70:73], v[46:49], a[24:27], v[166], v[159] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[70:73], v[54:57], a[28:31], v[166], v[159] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[74:77], v[34:37], a[16:19], v[166], v[158] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[74:77], v[42:45], a[20:23], v[166], v[158] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[134:137], v23, s[20:23], s58 offen
buffer_load_dwordx4 v[138:141], v23, s[20:23], s72 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[74:77], v[50:53], a[24:27], v[166], v[159] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[74:77], v[58:61], a[28:31], v[166], v[159] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[78:81], v[30:33], a[32:35], v[167], v[158] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[78:81], v[38:41], a[36:39], v[167], v[158] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[78:81], v[46:49], a[40:43], v[167], v[159] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[78:81], v[54:57], a[44:47], v[167], v[159] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[142:145], v23, s[20:23], s74 offen
buffer_load_dwordx4 v[146:149], v23, s[20:23], s75 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[82:85], v[34:37], a[32:35], v[167], v[158] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[82:85], v[42:45], a[36:39], v[167], v[158] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[82:85], v[50:53], a[40:43], v[167], v[159] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[82:85], v[58:61], a[44:47], v[167], v[159] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[86:89], v[30:33], a[48:51], v[167], v[158] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[86:89], v[38:41], a[52:55], v[167], v[158] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[150:153], v23, s[20:23], s78 offen
buffer_load_dwordx4 v[154:157], v23, s[20:23], s59 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[86:89], v[46:49], a[56:59], v[167], v[159] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[86:89], v[54:57], a[60:63], v[167], v[159] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[90:93], v[34:37], a[48:51], v[167], v[158] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[90:93], v[42:45], a[52:55], v[167], v[158] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[90:93], v[50:53], a[56:59], v[167], v[159] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[90:93], v[58:61], a[60:63], v[167], v[159] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[62:65], v[94:97], a[64:67], v[166], v[160] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[62:65], v[102:105], a[68:71], v[166], v[160] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[62:65], v[110:113], a[72:75], v[166], v[161] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[62:65], v[118:121], a[76:79], v[166], v[161] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[66:69], v[98:101], a[64:67], v[166], v[160] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[66:69], v[106:109], a[68:71], v[166], v[160] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[66:69], v[114:117], a[72:75], v[166], v[161] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[66:69], v[122:125], a[76:79], v[166], v[161] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[70:73], v[94:97], a[80:83], v[166], v[160] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[70:73], v[102:105], a[84:87], v[166], v[160] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[70:73], v[110:113], a[88:91], v[166], v[161] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[70:73], v[118:121], a[92:95], v[166], v[161] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[74:77], v[98:101], a[80:83], v[166], v[160] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[74:77], v[106:109], a[84:87], v[166], v[160] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[74:77], v[114:117], a[88:91], v[166], v[161] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[74:77], v[122:125], a[92:95], v[166], v[161] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[78:81], v[94:97], a[96:99], v[167], v[160] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[78:81], v[102:105], a[100:103], v[167], v[160] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[78:81], v[110:113], a[104:107], v[167], v[161] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[78:81], v[118:121], a[108:111], v[167], v[161] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[82:85], v[98:101], a[96:99], v[167], v[160] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[82:85], v[106:109], a[100:103], v[167], v[160] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[82:85], v[114:117], a[104:107], v[167], v[161] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[82:85], v[122:125], a[108:111], v[167], v[161] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[86:89], v[94:97], a[112:115], v[167], v[160] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[86:89], v[102:105], a[116:119], v[167], v[160] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[86:89], v[110:113], a[120:123], v[167], v[161] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[86:89], v[118:121], a[124:127], v[167], v[161] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[90:93], v[98:101], a[112:115], v[167], v[160] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[90:93], v[106:109], a[116:119], v[167], v[160] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[90:93], v[114:117], a[120:123], v[167], v[161] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[90:93], v[122:125], a[124:127], v[167], v[161] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_i32 s99, s54, 0x280
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_mov_b32 m0, s33
	s_add_i32 vcc_lo, s55, 0x280
	buffer_load_dwordx4 v2, s[12:15], s99 offen lds
	s_mov_b32 m0, s43
	s_add_i32 s24, s53, 0x200
	buffer_load_dwordx4 v1, s[12:15], s99 offen lds
	s_mov_b32 m0, s44
	s_add_i32 s60, s52, 0xffffec00
	buffer_load_dwordx4 v3, s[12:15], s99 offen lds
	s_mov_b32 m0, s45
	s_add_i32 s25, s53, 0x80200
	buffer_load_dwordx4 v4, s[12:15], s99 offen lds
	s_mov_b32 m0, s35
	s_add_i32 s56, s24, 64
	buffer_load_dwordx4 v2, s[12:15], vcc_lo offen lds
	s_mov_b32 m0, s46
	s_add_i32 s58, s24, 0x10000
	buffer_load_dwordx4 v1, s[12:15], vcc_lo offen lds
	s_mov_b32 m0, s47
	s_add_i32 s59, s24, 0x10040
	buffer_load_dwordx4 v3, s[12:15], vcc_lo offen lds
	s_mov_b32 m0, s48
	s_add_i32 s72, s24, 0x20000
	buffer_load_dwordx4 v4, s[12:15], vcc_lo offen lds
	s_add_i32 s74, s24, 0x20040
	s_add_i32 s75, s24, 0x30000
	s_add_i32 s78, s24, 0x30040
	;;#ASMSTART
	s_waitcnt vmcnt(0)
s_barrier
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[126:129], v[30:33], a[128:131], v[168], v[158] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[126:129], v[38:41], a[132:135], v[168], v[158] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[62:65], v23, s[20:23], s24 offen
buffer_load_dwordx4 v[66:69], v23, s[20:23], s56 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[126:129], v[46:49], a[136:139], v[168], v[159] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[126:129], v[54:57], a[140:143], v[168], v[159] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[130:133], v[34:37], a[128:131], v[168], v[158] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[130:133], v[42:45], a[132:135], v[168], v[158] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[70:73], v23, s[20:23], s58 offen
buffer_load_dwordx4 v[74:77], v23, s[20:23], s59 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[130:133], v[50:53], a[136:139], v[168], v[159] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[130:133], v[58:61], a[140:143], v[168], v[159] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[134:137], v[30:33], a[144:147], v[168], v[158] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[134:137], v[38:41], a[148:151], v[168], v[158] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[78:81], v23, s[20:23], s72 offen
buffer_load_dwordx4 v[82:85], v23, s[20:23], s74 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[134:137], v[46:49], a[152:155], v[168], v[159] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[134:137], v[54:57], a[156:159], v[168], v[159] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[138:141], v[34:37], a[144:147], v[168], v[158] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[138:141], v[42:45], a[148:151], v[168], v[158] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[86:89], v23, s[20:23], s75 offen
buffer_load_dwordx4 v[90:93], v23, s[20:23], s78 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[138:141], v[50:53], a[152:155], v[168], v[159] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[138:141], v[58:61], a[156:159], v[168], v[159] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[142:145], v[30:33], a[160:163], v[169], v[158] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[142:145], v[38:41], a[164:167], v[169], v[158] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[142:145], v[46:49], a[168:171], v[169], v[159] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[142:145], v[54:57], a[172:175], v[169], v[159] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[146:149], v[34:37], a[160:163], v[169], v[158] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[146:149], v[42:45], a[164:167], v[169], v[158] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[146:149], v[50:53], a[168:171], v[169], v[159] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[146:149], v[58:61], a[172:175], v[169], v[159] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[150:153], v[30:33], a[176:179], v[169], v[158] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[150:153], v[38:41], a[180:183], v[169], v[158] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[150:153], v[46:49], a[184:187], v[169], v[159] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[150:153], v[54:57], a[188:191], v[169], v[159] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[154:157], v[34:37], a[176:179], v[169], v[158] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[154:157], v[42:45], a[180:183], v[169], v[158] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[154:157], v[50:53], a[184:187], v[169], v[159] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[154:157], v[58:61], a[188:191], v[169], v[159] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[126:129], v[94:97], a[192:195], v[168], v[160] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v7 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[126:129], v[102:105], a[196:199], v[168], v[160] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v16 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[126:129], v[110:113], a[200:203], v[168], v[161] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v7 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[126:129], v[118:121], a[204:207], v[168], v[161] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v16 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[130:133], v[98:101], a[192:195], v[168], v[160] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v7 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[130:133], v[106:109], a[196:199], v[168], v[160] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v16 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[130:133], v[114:117], a[200:203], v[168], v[161] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v7 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[130:133], v[122:125], a[204:207], v[168], v[161] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v16 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[134:137], v[94:97], a[208:211], v[168], v[160] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[134:137], v[102:105], a[212:215], v[168], v[160] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[134:137], v[110:113], a[216:219], v[168], v[161] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[134:137], v[118:121], a[220:223], v[168], v[161] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[138:141], v[98:101], a[208:211], v[168], v[160] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[138:141], v[106:109], a[212:215], v[168], v[160] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[138:141], v[114:117], a[216:219], v[168], v[161] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[138:141], v[122:125], a[220:223], v[168], v[161] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[142:145], v[94:97], a[224:227], v[169], v[160] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[142:145], v[102:105], a[228:231], v[169], v[160] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[142:145], v[110:113], a[232:235], v[169], v[161] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[142:145], v[118:121], a[236:239], v[169], v[161] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[146:149], v[98:101], a[224:227], v[169], v[160] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[146:149], v[106:109], a[228:231], v[169], v[160] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[146:149], v[114:117], a[232:235], v[169], v[161] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[146:149], v[122:125], a[236:239], v[169], v[161] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[150:153], v[94:97], a[240:243], v[169], v[160] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[150:153], v[102:105], a[244:247], v[169], v[160] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[150:153], v[110:113], a[248:251], v[169], v[161] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[150:153], v[118:121], a[252:255], v[169], v[161] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[154:157], v[98:101], a[240:243], v[169], v[160] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[154:157], v[106:109], a[244:247], v[169], v[160] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[154:157], v[114:117], a[248:251], v[169], v[161] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[154:157], v[122:125], a[252:255], v[169], v[161] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_i32 s24, s25, 64
	s_add_i32 s56, s25, 0x10000
	s_add_i32 s58, s25, 0x10040
	s_add_i32 s59, s25, 0x20000
	s_add_i32 s72, s25, 0x20040
	s_add_i32 s78, s25, 0x30000
	;;#ASMSTART
	s_waitcnt vmcnt(0) lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v[166], v8
v_mov_b32 v[167], v9
v_mov_b32 v[168], v10
v_mov_b32 v[169], v11
v_mov_b32 v[158], v12
v_mov_b32 v[159], v13
v_mov_b32 v[160], v14
v_mov_b32 v[161], v15

	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[8:9], v5, s[0:3], s60 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[10:11], v5, s[4:7], s60 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[12:13], v5, s[8:11], s60 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[14:15], v5, s[16:19], s60 offen
	;;#ASMEND
	s_add_i32 s60, s25, 0x30040
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[62:65], v[30:33], a[0:3], v[166], v[158] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[94:97], v19 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[62:65], v[38:41], a[4:7], v[166], v[158] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[98:101], v20 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[62:65], v[46:49], a[8:11], v[166], v[159] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[102:105], v19 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[62:65], v[54:57], a[12:15], v[166], v[159] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[106:109], v20 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[66:69], v[34:37], a[0:3], v[166], v[158] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[110:113], v19 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[66:69], v[42:45], a[4:7], v[166], v[158] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[114:117], v20 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[66:69], v[50:53], a[8:11], v[166], v[159] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[118:121], v19 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:69], v[58:61], a[12:15], v[166], v[159] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[122:125], v20 offset:6144
buffer_load_dwordx4 v[126:129], v23, s[20:23], s25 offen
buffer_load_dwordx4 v[130:133], v23, s[20:23], s24 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[70:73], v[30:33], a[16:19], v[166], v[158] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[70:73], v[38:41], a[20:23], v[166], v[158] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[70:73], v[46:49], a[24:27], v[166], v[159] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[70:73], v[54:57], a[28:31], v[166], v[159] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[74:77], v[34:37], a[16:19], v[166], v[158] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[74:77], v[42:45], a[20:23], v[166], v[158] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[134:137], v23, s[20:23], s56 offen
buffer_load_dwordx4 v[138:141], v23, s[20:23], s58 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[74:77], v[50:53], a[24:27], v[166], v[159] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[74:77], v[58:61], a[28:31], v[166], v[159] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[78:81], v[30:33], a[32:35], v[167], v[158] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[78:81], v[38:41], a[36:39], v[167], v[158] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[78:81], v[46:49], a[40:43], v[167], v[159] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[78:81], v[54:57], a[44:47], v[167], v[159] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[142:145], v23, s[20:23], s59 offen
buffer_load_dwordx4 v[146:149], v23, s[20:23], s72 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[82:85], v[34:37], a[32:35], v[167], v[158] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[82:85], v[42:45], a[36:39], v[167], v[158] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[82:85], v[50:53], a[40:43], v[167], v[159] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[82:85], v[58:61], a[44:47], v[167], v[159] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[86:89], v[30:33], a[48:51], v[167], v[158] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[86:89], v[38:41], a[52:55], v[167], v[158] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[150:153], v23, s[20:23], s78 offen
buffer_load_dwordx4 v[154:157], v23, s[20:23], s60 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[86:89], v[46:49], a[56:59], v[167], v[159] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[86:89], v[54:57], a[60:63], v[167], v[159] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[90:93], v[34:37], a[48:51], v[167], v[158] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[90:93], v[42:45], a[52:55], v[167], v[158] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[90:93], v[50:53], a[56:59], v[167], v[159] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[90:93], v[58:61], a[60:63], v[167], v[159] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[62:65], v[94:97], a[64:67], v[166], v[160] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[62:65], v[102:105], a[68:71], v[166], v[160] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[62:65], v[110:113], a[72:75], v[166], v[161] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[62:65], v[118:121], a[76:79], v[166], v[161] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[66:69], v[98:101], a[64:67], v[166], v[160] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[66:69], v[106:109], a[68:71], v[166], v[160] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[66:69], v[114:117], a[72:75], v[166], v[161] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[66:69], v[122:125], a[76:79], v[166], v[161] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[70:73], v[94:97], a[80:83], v[166], v[160] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[70:73], v[102:105], a[84:87], v[166], v[160] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[70:73], v[110:113], a[88:91], v[166], v[161] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[70:73], v[118:121], a[92:95], v[166], v[161] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[74:77], v[98:101], a[80:83], v[166], v[160] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[74:77], v[106:109], a[84:87], v[166], v[160] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[74:77], v[114:117], a[88:91], v[166], v[161] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[74:77], v[122:125], a[92:95], v[166], v[161] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[78:81], v[94:97], a[96:99], v[167], v[160] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[78:81], v[102:105], a[100:103], v[167], v[160] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[78:81], v[110:113], a[104:107], v[167], v[161] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[78:81], v[118:121], a[108:111], v[167], v[161] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[82:85], v[98:101], a[96:99], v[167], v[160] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[82:85], v[106:109], a[100:103], v[167], v[160] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[82:85], v[114:117], a[104:107], v[167], v[161] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[82:85], v[122:125], a[108:111], v[167], v[161] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[86:89], v[94:97], a[112:115], v[167], v[160] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[86:89], v[102:105], a[116:119], v[167], v[160] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[86:89], v[110:113], a[120:123], v[167], v[161] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[86:89], v[118:121], a[124:127], v[167], v[161] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[90:93], v[98:101], a[112:115], v[167], v[160] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[90:93], v[106:109], a[116:119], v[167], v[160] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[90:93], v[114:117], a[120:123], v[167], v[161] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[90:93], v[122:125], a[124:127], v[167], v[161] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_i32 s98, s54, 0x300
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_mov_b32 m0, s29
	s_add_i32 s86, s55, 0x300
	buffer_load_dwordx4 v2, s[12:15], s98 offen lds
	s_mov_b32 m0, s37
	s_add_i32 s73, s53, 0x280
	buffer_load_dwordx4 v1, s[12:15], s98 offen lds
	s_mov_b32 m0, s38
	s_add_i32 s62, s52, 0xffffee00
	buffer_load_dwordx4 v3, s[12:15], s98 offen lds
	s_mov_b32 m0, s39
	s_add_i32 s77, s53, 0x80280
	buffer_load_dwordx4 v4, s[12:15], s98 offen lds
	s_mov_b32 m0, s34
	s_add_i32 s24, s73, 64
	buffer_load_dwordx4 v2, s[12:15], s86 offen lds
	s_mov_b32 m0, s40
	s_add_i32 s25, s73, 0x10000
	buffer_load_dwordx4 v1, s[12:15], s86 offen lds
	s_mov_b32 m0, s41
	s_add_i32 s56, s73, 0x10040
	buffer_load_dwordx4 v3, s[12:15], s86 offen lds
	s_mov_b32 m0, s42
	s_add_i32 s58, s73, 0x20000
	buffer_load_dwordx4 v4, s[12:15], s86 offen lds
	s_add_i32 s59, s73, 0x20040
	s_add_i32 s60, s73, 0x30000
	s_add_i32 s86, s73, 0x30040
	;;#ASMSTART
	s_waitcnt vmcnt(0)
s_barrier
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[126:129], v[30:33], a[128:131], v[168], v[158] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[126:129], v[38:41], a[132:135], v[168], v[158] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[62:65], v23, s[20:23], s73 offen
buffer_load_dwordx4 v[66:69], v23, s[20:23], s24 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[126:129], v[46:49], a[136:139], v[168], v[159] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[126:129], v[54:57], a[140:143], v[168], v[159] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[130:133], v[34:37], a[128:131], v[168], v[158] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[130:133], v[42:45], a[132:135], v[168], v[158] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[70:73], v23, s[20:23], s25 offen
buffer_load_dwordx4 v[74:77], v23, s[20:23], s56 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[130:133], v[50:53], a[136:139], v[168], v[159] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[130:133], v[58:61], a[140:143], v[168], v[159] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[134:137], v[30:33], a[144:147], v[168], v[158] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[134:137], v[38:41], a[148:151], v[168], v[158] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[78:81], v23, s[20:23], s58 offen
buffer_load_dwordx4 v[82:85], v23, s[20:23], s59 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[134:137], v[46:49], a[152:155], v[168], v[159] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[134:137], v[54:57], a[156:159], v[168], v[159] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[138:141], v[34:37], a[144:147], v[168], v[158] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[138:141], v[42:45], a[148:151], v[168], v[158] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[86:89], v23, s[20:23], s60 offen
buffer_load_dwordx4 v[90:93], v23, s[20:23], s86 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[138:141], v[50:53], a[152:155], v[168], v[159] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[138:141], v[58:61], a[156:159], v[168], v[159] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[142:145], v[30:33], a[160:163], v[169], v[158] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[142:145], v[38:41], a[164:167], v[169], v[158] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[142:145], v[46:49], a[168:171], v[169], v[159] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[142:145], v[54:57], a[172:175], v[169], v[159] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[146:149], v[34:37], a[160:163], v[169], v[158] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[146:149], v[42:45], a[164:167], v[169], v[158] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[146:149], v[50:53], a[168:171], v[169], v[159] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[146:149], v[58:61], a[172:175], v[169], v[159] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[150:153], v[30:33], a[176:179], v[169], v[158] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[150:153], v[38:41], a[180:183], v[169], v[158] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[150:153], v[46:49], a[184:187], v[169], v[159] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[150:153], v[54:57], a[188:191], v[169], v[159] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[154:157], v[34:37], a[176:179], v[169], v[158] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[154:157], v[42:45], a[180:183], v[169], v[158] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[154:157], v[50:53], a[184:187], v[169], v[159] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[154:157], v[58:61], a[188:191], v[169], v[159] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[126:129], v[94:97], a[192:195], v[168], v[160] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v17 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[126:129], v[102:105], a[196:199], v[168], v[160] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v18 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[126:129], v[110:113], a[200:203], v[168], v[161] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v17 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[126:129], v[118:121], a[204:207], v[168], v[161] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v18 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[130:133], v[98:101], a[192:195], v[168], v[160] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v17 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[130:133], v[106:109], a[196:199], v[168], v[160] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v18 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[130:133], v[114:117], a[200:203], v[168], v[161] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v17 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[130:133], v[122:125], a[204:207], v[168], v[161] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v18 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[134:137], v[94:97], a[208:211], v[168], v[160] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[134:137], v[102:105], a[212:215], v[168], v[160] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[134:137], v[110:113], a[216:219], v[168], v[161] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[134:137], v[118:121], a[220:223], v[168], v[161] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[138:141], v[98:101], a[208:211], v[168], v[160] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[138:141], v[106:109], a[212:215], v[168], v[160] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[138:141], v[114:117], a[216:219], v[168], v[161] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[138:141], v[122:125], a[220:223], v[168], v[161] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[142:145], v[94:97], a[224:227], v[169], v[160] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[142:145], v[102:105], a[228:231], v[169], v[160] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[142:145], v[110:113], a[232:235], v[169], v[161] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[142:145], v[118:121], a[236:239], v[169], v[161] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[146:149], v[98:101], a[224:227], v[169], v[160] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[146:149], v[106:109], a[228:231], v[169], v[160] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[146:149], v[114:117], a[232:235], v[169], v[161] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[146:149], v[122:125], a[236:239], v[169], v[161] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[150:153], v[94:97], a[240:243], v[169], v[160] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[150:153], v[102:105], a[244:247], v[169], v[160] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[150:153], v[110:113], a[248:251], v[169], v[161] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[150:153], v[118:121], a[252:255], v[169], v[161] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[154:157], v[98:101], a[240:243], v[169], v[160] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[154:157], v[106:109], a[244:247], v[169], v[160] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[154:157], v[114:117], a[248:251], v[169], v[161] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[154:157], v[122:125], a[252:255], v[169], v[161] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_i32 s24, s77, 64
	s_add_i32 s25, s77, 0x10000
	s_add_i32 s56, s77, 0x10040
	s_add_i32 s58, s77, 0x20000
	s_add_i32 s59, s77, 0x20040
	s_add_i32 s60, s77, 0x30000
	;;#ASMSTART
	s_waitcnt vmcnt(0) lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v[166], v8
v_mov_b32 v[167], v9
v_mov_b32 v[168], v10
v_mov_b32 v[169], v11
v_mov_b32 v[158], v12
v_mov_b32 v[159], v13
v_mov_b32 v[160], v14
v_mov_b32 v[161], v15

	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[8:9], v5, s[0:3], s62 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[10:11], v5, s[4:7], s62 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[12:13], v5, s[8:11], s62 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[14:15], v5, s[16:19], s62 offen
	;;#ASMEND
	s_add_i32 s62, s77, 0x30040
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[62:65], v[30:33], a[0:3], v[166], v[158] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[94:97], v21 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[62:65], v[38:41], a[4:7], v[166], v[158] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[98:101], v22 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[62:65], v[46:49], a[8:11], v[166], v[159] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[102:105], v21 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[62:65], v[54:57], a[12:15], v[166], v[159] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[106:109], v22 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[66:69], v[34:37], a[0:3], v[166], v[158] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[110:113], v21 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[66:69], v[42:45], a[4:7], v[166], v[158] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[114:117], v22 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[66:69], v[50:53], a[8:11], v[166], v[159] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[118:121], v21 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:69], v[58:61], a[12:15], v[166], v[159] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[122:125], v22 offset:6144
buffer_load_dwordx4 v[126:129], v23, s[20:23], s77 offen
buffer_load_dwordx4 v[130:133], v23, s[20:23], s24 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[70:73], v[30:33], a[16:19], v[166], v[158] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[70:73], v[38:41], a[20:23], v[166], v[158] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[70:73], v[46:49], a[24:27], v[166], v[159] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[70:73], v[54:57], a[28:31], v[166], v[159] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[74:77], v[34:37], a[16:19], v[166], v[158] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[74:77], v[42:45], a[20:23], v[166], v[158] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[134:137], v23, s[20:23], s25 offen
buffer_load_dwordx4 v[138:141], v23, s[20:23], s56 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[74:77], v[50:53], a[24:27], v[166], v[159] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[74:77], v[58:61], a[28:31], v[166], v[159] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[78:81], v[30:33], a[32:35], v[167], v[158] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[78:81], v[38:41], a[36:39], v[167], v[158] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[78:81], v[46:49], a[40:43], v[167], v[159] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[78:81], v[54:57], a[44:47], v[167], v[159] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[142:145], v23, s[20:23], s58 offen
buffer_load_dwordx4 v[146:149], v23, s[20:23], s59 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[82:85], v[34:37], a[32:35], v[167], v[158] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[82:85], v[42:45], a[36:39], v[167], v[158] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[82:85], v[50:53], a[40:43], v[167], v[159] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[82:85], v[58:61], a[44:47], v[167], v[159] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[86:89], v[30:33], a[48:51], v[167], v[158] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[86:89], v[38:41], a[52:55], v[167], v[158] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[150:153], v23, s[20:23], s60 offen
buffer_load_dwordx4 v[154:157], v23, s[20:23], s62 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[86:89], v[46:49], a[56:59], v[167], v[159] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[86:89], v[54:57], a[60:63], v[167], v[159] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[90:93], v[34:37], a[48:51], v[167], v[158] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[90:93], v[42:45], a[52:55], v[167], v[158] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[90:93], v[50:53], a[56:59], v[167], v[159] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[90:93], v[58:61], a[60:63], v[167], v[159] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[62:65], v[94:97], a[64:67], v[166], v[160] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[62:65], v[102:105], a[68:71], v[166], v[160] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[62:65], v[110:113], a[72:75], v[166], v[161] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[62:65], v[118:121], a[76:79], v[166], v[161] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[66:69], v[98:101], a[64:67], v[166], v[160] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[66:69], v[106:109], a[68:71], v[166], v[160] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[66:69], v[114:117], a[72:75], v[166], v[161] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[66:69], v[122:125], a[76:79], v[166], v[161] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[70:73], v[94:97], a[80:83], v[166], v[160] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[70:73], v[102:105], a[84:87], v[166], v[160] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[70:73], v[110:113], a[88:91], v[166], v[161] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[70:73], v[118:121], a[92:95], v[166], v[161] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[74:77], v[98:101], a[80:83], v[166], v[160] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[74:77], v[106:109], a[84:87], v[166], v[160] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[74:77], v[114:117], a[88:91], v[166], v[161] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[74:77], v[122:125], a[92:95], v[166], v[161] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[78:81], v[94:97], a[96:99], v[167], v[160] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[78:81], v[102:105], a[100:103], v[167], v[160] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[78:81], v[110:113], a[104:107], v[167], v[161] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[78:81], v[118:121], a[108:111], v[167], v[161] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[82:85], v[98:101], a[96:99], v[167], v[160] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[82:85], v[106:109], a[100:103], v[167], v[160] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[82:85], v[114:117], a[104:107], v[167], v[161] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[82:85], v[122:125], a[108:111], v[167], v[161] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[86:89], v[94:97], a[112:115], v[167], v[160] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[86:89], v[102:105], a[116:119], v[167], v[160] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[86:89], v[110:113], a[120:123], v[167], v[161] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[86:89], v[118:121], a[124:127], v[167], v[161] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[90:93], v[98:101], a[112:115], v[167], v[160] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[90:93], v[106:109], a[116:119], v[167], v[160] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[90:93], v[114:117], a[120:123], v[167], v[161] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[90:93], v[122:125], a[124:127], v[167], v[161] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_i32 s96, s54, 0x380
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_mov_b32 m0, s33
	s_add_i32 s57, s55, 0x380
	buffer_load_dwordx4 v2, s[12:15], s96 offen lds
	s_mov_b32 m0, s43
	s_add_i32 s83, s53, 0x300
	buffer_load_dwordx4 v1, s[12:15], s96 offen lds
	s_mov_b32 m0, s44
	s_add_i32 s64, s52, 0xfffff000
	buffer_load_dwordx4 v3, s[12:15], s96 offen lds
	s_mov_b32 m0, s45
	s_add_i32 s97, s53, 0x80300
	buffer_load_dwordx4 v4, s[12:15], s96 offen lds
	s_mov_b32 m0, s35
	s_add_i32 s24, s83, 64
	buffer_load_dwordx4 v2, s[12:15], s57 offen lds
	s_mov_b32 m0, s46
	s_add_i32 s25, s83, 0x10000
	buffer_load_dwordx4 v1, s[12:15], s57 offen lds
	s_mov_b32 m0, s47
	s_add_i32 s56, s83, 0x10040
	buffer_load_dwordx4 v3, s[12:15], s57 offen lds
	s_mov_b32 m0, s48
	s_add_i32 s58, s83, 0x20000
	buffer_load_dwordx4 v4, s[12:15], s57 offen lds
	s_add_i32 s62, s83, 0x20040
	s_add_i32 s96, s83, 0x30000
	s_add_i32 s57, s83, 0x30040
	;;#ASMSTART
	s_waitcnt vmcnt(0)
s_barrier
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[126:129], v[30:33], a[128:131], v[168], v[158] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[126:129], v[38:41], a[132:135], v[168], v[158] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[62:65], v23, s[20:23], s83 offen
buffer_load_dwordx4 v[66:69], v23, s[20:23], s24 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[126:129], v[46:49], a[136:139], v[168], v[159] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[126:129], v[54:57], a[140:143], v[168], v[159] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[130:133], v[34:37], a[128:131], v[168], v[158] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[130:133], v[42:45], a[132:135], v[168], v[158] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[70:73], v23, s[20:23], s25 offen
buffer_load_dwordx4 v[74:77], v23, s[20:23], s56 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[130:133], v[50:53], a[136:139], v[168], v[159] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[130:133], v[58:61], a[140:143], v[168], v[159] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[134:137], v[30:33], a[144:147], v[168], v[158] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[134:137], v[38:41], a[148:151], v[168], v[158] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[78:81], v23, s[20:23], s58 offen
buffer_load_dwordx4 v[82:85], v23, s[20:23], s62 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[134:137], v[46:49], a[152:155], v[168], v[159] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[134:137], v[54:57], a[156:159], v[168], v[159] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[138:141], v[34:37], a[144:147], v[168], v[158] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[138:141], v[42:45], a[148:151], v[168], v[158] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[86:89], v23, s[20:23], s96 offen
buffer_load_dwordx4 v[90:93], v23, s[20:23], s57 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[138:141], v[50:53], a[152:155], v[168], v[159] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[138:141], v[58:61], a[156:159], v[168], v[159] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[142:145], v[30:33], a[160:163], v[169], v[158] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[142:145], v[38:41], a[164:167], v[169], v[158] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[142:145], v[46:49], a[168:171], v[169], v[159] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[142:145], v[54:57], a[172:175], v[169], v[159] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[146:149], v[34:37], a[160:163], v[169], v[158] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[146:149], v[42:45], a[164:167], v[169], v[158] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[146:149], v[50:53], a[168:171], v[169], v[159] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[146:149], v[58:61], a[172:175], v[169], v[159] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[150:153], v[30:33], a[176:179], v[169], v[158] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[150:153], v[38:41], a[180:183], v[169], v[158] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[150:153], v[46:49], a[184:187], v[169], v[159] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[150:153], v[54:57], a[188:191], v[169], v[159] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[154:157], v[34:37], a[176:179], v[169], v[158] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[154:157], v[42:45], a[180:183], v[169], v[158] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[154:157], v[50:53], a[184:187], v[169], v[159] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[154:157], v[58:61], a[188:191], v[169], v[159] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[126:129], v[94:97], a[192:195], v[168], v[160] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v7 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[126:129], v[102:105], a[196:199], v[168], v[160] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v16 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[126:129], v[110:113], a[200:203], v[168], v[161] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v7 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[126:129], v[118:121], a[204:207], v[168], v[161] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v16 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[130:133], v[98:101], a[192:195], v[168], v[160] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v7 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[130:133], v[106:109], a[196:199], v[168], v[160] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v16 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[130:133], v[114:117], a[200:203], v[168], v[161] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v7 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[130:133], v[122:125], a[204:207], v[168], v[161] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v16 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[134:137], v[94:97], a[208:211], v[168], v[160] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[134:137], v[102:105], a[212:215], v[168], v[160] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[134:137], v[110:113], a[216:219], v[168], v[161] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[134:137], v[118:121], a[220:223], v[168], v[161] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[138:141], v[98:101], a[208:211], v[168], v[160] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[138:141], v[106:109], a[212:215], v[168], v[160] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[138:141], v[114:117], a[216:219], v[168], v[161] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[138:141], v[122:125], a[220:223], v[168], v[161] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[142:145], v[94:97], a[224:227], v[169], v[160] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[142:145], v[102:105], a[228:231], v[169], v[160] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[142:145], v[110:113], a[232:235], v[169], v[161] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[142:145], v[118:121], a[236:239], v[169], v[161] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[146:149], v[98:101], a[224:227], v[169], v[160] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[146:149], v[106:109], a[228:231], v[169], v[160] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[146:149], v[114:117], a[232:235], v[169], v[161] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[146:149], v[122:125], a[236:239], v[169], v[161] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[150:153], v[94:97], a[240:243], v[169], v[160] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[150:153], v[102:105], a[244:247], v[169], v[160] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[150:153], v[110:113], a[248:251], v[169], v[161] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[150:153], v[118:121], a[252:255], v[169], v[161] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[154:157], v[98:101], a[240:243], v[169], v[160] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[154:157], v[106:109], a[244:247], v[169], v[160] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[154:157], v[114:117], a[248:251], v[169], v[161] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[154:157], v[122:125], a[252:255], v[169], v[161] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_i32 s92, s54, 0x400
	s_add_i32 s94, s55, 0x400
	s_add_i32 s85, s54, 0x480
	s_add_i32 s82, s55, 0x480
	s_add_i32 s84, s54, 0x500
	s_add_i32 s81, s55, 0x500
	s_add_i32 s76, s54, 0x580
	s_add_i32 s80, s55, 0x580
	s_add_i32 s75, s54, 0x600
	s_add_i32 s74, s55, 0x600
	s_add_i32 s72, s54, 0x680
	s_add_i32 s73, s55, 0x680
	s_add_i32 s60, s54, 0x700
	s_add_i32 s59, s55, 0x700
	s_add_i32 s56, s54, 0x780
	s_add_i32 s54, s55, 0x780
	s_add_i32 s24, s97, 64
	s_add_i32 s25, s97, 0x10000
	s_add_i32 s55, s97, 0x10040
	s_add_i32 s57, s97, 0x20000
	s_add_i32 s58, s97, 0x20040
	s_add_i32 s96, s97, 0x30000
	;;#ASMSTART
	s_waitcnt vmcnt(0) lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v[166], v8
v_mov_b32 v[167], v9
v_mov_b32 v[168], v10
v_mov_b32 v[169], v11
v_mov_b32 v[158], v12
v_mov_b32 v[159], v13
v_mov_b32 v[160], v14
v_mov_b32 v[161], v15

	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[8:9], v5, s[0:3], s64 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[10:11], v5, s[4:7], s64 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[12:13], v5, s[8:11], s64 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[14:15], v5, s[16:19], s64 offen
	;;#ASMEND
	s_add_i32 s64, s97, 0x30040
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[62:65], v[30:33], a[0:3], v[166], v[158] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[94:97], v19 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[62:65], v[38:41], a[4:7], v[166], v[158] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[98:101], v20 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[62:65], v[46:49], a[8:11], v[166], v[159] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[102:105], v19 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[62:65], v[54:57], a[12:15], v[166], v[159] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[106:109], v20 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[66:69], v[34:37], a[0:3], v[166], v[158] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[110:113], v19 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[66:69], v[42:45], a[4:7], v[166], v[158] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[114:117], v20 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[66:69], v[50:53], a[8:11], v[166], v[159] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[118:121], v19 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:69], v[58:61], a[12:15], v[166], v[159] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[122:125], v20 offset:6144
buffer_load_dwordx4 v[126:129], v23, s[20:23], s97 offen
buffer_load_dwordx4 v[130:133], v23, s[20:23], s24 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[70:73], v[30:33], a[16:19], v[166], v[158] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[70:73], v[38:41], a[20:23], v[166], v[158] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[70:73], v[46:49], a[24:27], v[166], v[159] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[70:73], v[54:57], a[28:31], v[166], v[159] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[74:77], v[34:37], a[16:19], v[166], v[158] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[74:77], v[42:45], a[20:23], v[166], v[158] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[134:137], v23, s[20:23], s25 offen
buffer_load_dwordx4 v[138:141], v23, s[20:23], s55 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[74:77], v[50:53], a[24:27], v[166], v[159] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[74:77], v[58:61], a[28:31], v[166], v[159] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[78:81], v[30:33], a[32:35], v[167], v[158] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[78:81], v[38:41], a[36:39], v[167], v[158] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[78:81], v[46:49], a[40:43], v[167], v[159] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[78:81], v[54:57], a[44:47], v[167], v[159] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[142:145], v23, s[20:23], s57 offen
buffer_load_dwordx4 v[146:149], v23, s[20:23], s58 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[82:85], v[34:37], a[32:35], v[167], v[158] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[82:85], v[42:45], a[36:39], v[167], v[158] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[82:85], v[50:53], a[40:43], v[167], v[159] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[82:85], v[58:61], a[44:47], v[167], v[159] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[86:89], v[30:33], a[48:51], v[167], v[158] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[86:89], v[38:41], a[52:55], v[167], v[158] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[150:153], v23, s[20:23], s96 offen
buffer_load_dwordx4 v[154:157], v23, s[20:23], s64 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[86:89], v[46:49], a[56:59], v[167], v[159] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[86:89], v[54:57], a[60:63], v[167], v[159] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[90:93], v[34:37], a[48:51], v[167], v[158] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[90:93], v[42:45], a[52:55], v[167], v[158] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[90:93], v[50:53], a[56:59], v[167], v[159] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[90:93], v[58:61], a[60:63], v[167], v[159] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[62:65], v[94:97], a[64:67], v[166], v[160] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[62:65], v[102:105], a[68:71], v[166], v[160] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[62:65], v[110:113], a[72:75], v[166], v[161] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[62:65], v[118:121], a[76:79], v[166], v[161] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[66:69], v[98:101], a[64:67], v[166], v[160] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[66:69], v[106:109], a[68:71], v[166], v[160] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[66:69], v[114:117], a[72:75], v[166], v[161] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[66:69], v[122:125], a[76:79], v[166], v[161] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[70:73], v[94:97], a[80:83], v[166], v[160] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[70:73], v[102:105], a[84:87], v[166], v[160] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[70:73], v[110:113], a[88:91], v[166], v[161] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[70:73], v[118:121], a[92:95], v[166], v[161] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[74:77], v[98:101], a[80:83], v[166], v[160] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[74:77], v[106:109], a[84:87], v[166], v[160] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[74:77], v[114:117], a[88:91], v[166], v[161] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[74:77], v[122:125], a[92:95], v[166], v[161] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[78:81], v[94:97], a[96:99], v[167], v[160] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[78:81], v[102:105], a[100:103], v[167], v[160] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[78:81], v[110:113], a[104:107], v[167], v[161] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[78:81], v[118:121], a[108:111], v[167], v[161] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[82:85], v[98:101], a[96:99], v[167], v[160] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[82:85], v[106:109], a[100:103], v[167], v[160] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[82:85], v[114:117], a[104:107], v[167], v[161] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[82:85], v[122:125], a[108:111], v[167], v[161] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[86:89], v[94:97], a[112:115], v[167], v[160] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[86:89], v[102:105], a[116:119], v[167], v[160] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[86:89], v[110:113], a[120:123], v[167], v[161] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[86:89], v[118:121], a[124:127], v[167], v[161] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[90:93], v[98:101], a[112:115], v[167], v[160] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[90:93], v[106:109], a[116:119], v[167], v[160] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[90:93], v[114:117], a[120:123], v[167], v[161] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[90:93], v[122:125], a[124:127], v[167], v[161] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_mov_b32 m0, s29
	s_add_i32 s95, s53, 0x380
	buffer_load_dwordx4 v2, s[12:15], s92 offen lds
	s_mov_b32 m0, s37
	s_add_i32 s93, s53, 0x80380
	buffer_load_dwordx4 v1, s[12:15], s92 offen lds
	s_mov_b32 m0, s38
	s_add_i32 s24, s95, 64
	buffer_load_dwordx4 v3, s[12:15], s92 offen lds
	s_mov_b32 m0, s39
	s_add_i32 s25, s95, 0x10000
	buffer_load_dwordx4 v4, s[12:15], s92 offen lds
	s_mov_b32 m0, s34
	s_add_i32 s64, s95, 0x10040
	buffer_load_dwordx4 v2, s[12:15], s94 offen lds
	s_mov_b32 m0, s40
	s_add_i32 s96, s95, 0x20000
	buffer_load_dwordx4 v1, s[12:15], s94 offen lds
	s_mov_b32 m0, s41
	s_add_i32 s97, s95, 0x20040
	buffer_load_dwordx4 v3, s[12:15], s94 offen lds
	s_mov_b32 m0, s42
	s_add_i32 s92, s95, 0x30000
	buffer_load_dwordx4 v4, s[12:15], s94 offen lds
	s_add_i32 s94, s95, 0x30040
	;;#ASMSTART
	s_waitcnt vmcnt(0)
s_barrier
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[126:129], v[30:33], a[128:131], v[168], v[158] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[126:129], v[38:41], a[132:135], v[168], v[158] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[62:65], v23, s[20:23], s95 offen
buffer_load_dwordx4 v[66:69], v23, s[20:23], s24 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[126:129], v[46:49], a[136:139], v[168], v[159] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[126:129], v[54:57], a[140:143], v[168], v[159] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[130:133], v[34:37], a[128:131], v[168], v[158] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[130:133], v[42:45], a[132:135], v[168], v[158] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[70:73], v23, s[20:23], s25 offen
buffer_load_dwordx4 v[74:77], v23, s[20:23], s64 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[130:133], v[50:53], a[136:139], v[168], v[159] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[130:133], v[58:61], a[140:143], v[168], v[159] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[134:137], v[30:33], a[144:147], v[168], v[158] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[134:137], v[38:41], a[148:151], v[168], v[158] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[78:81], v23, s[20:23], s96 offen
buffer_load_dwordx4 v[82:85], v23, s[20:23], s97 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[134:137], v[46:49], a[152:155], v[168], v[159] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[134:137], v[54:57], a[156:159], v[168], v[159] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[138:141], v[34:37], a[144:147], v[168], v[158] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[138:141], v[42:45], a[148:151], v[168], v[158] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[86:89], v23, s[20:23], s92 offen
buffer_load_dwordx4 v[90:93], v23, s[20:23], s94 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[138:141], v[50:53], a[152:155], v[168], v[159] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[138:141], v[58:61], a[156:159], v[168], v[159] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[142:145], v[30:33], a[160:163], v[169], v[158] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[142:145], v[38:41], a[164:167], v[169], v[158] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[142:145], v[46:49], a[168:171], v[169], v[159] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[142:145], v[54:57], a[172:175], v[169], v[159] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[146:149], v[34:37], a[160:163], v[169], v[158] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[146:149], v[42:45], a[164:167], v[169], v[158] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[146:149], v[50:53], a[168:171], v[169], v[159] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[146:149], v[58:61], a[172:175], v[169], v[159] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[150:153], v[30:33], a[176:179], v[169], v[158] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[150:153], v[38:41], a[180:183], v[169], v[158] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[150:153], v[46:49], a[184:187], v[169], v[159] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[150:153], v[54:57], a[188:191], v[169], v[159] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[154:157], v[34:37], a[176:179], v[169], v[158] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[154:157], v[42:45], a[180:183], v[169], v[158] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[154:157], v[50:53], a[184:187], v[169], v[159] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[154:157], v[58:61], a[188:191], v[169], v[159] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[126:129], v[94:97], a[192:195], v[168], v[160] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v17 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[126:129], v[102:105], a[196:199], v[168], v[160] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v18 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[126:129], v[110:113], a[200:203], v[168], v[161] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v17 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[126:129], v[118:121], a[204:207], v[168], v[161] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v18 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[130:133], v[98:101], a[192:195], v[168], v[160] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v17 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[130:133], v[106:109], a[196:199], v[168], v[160] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v18 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[130:133], v[114:117], a[200:203], v[168], v[161] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v17 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[130:133], v[122:125], a[204:207], v[168], v[161] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v18 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[134:137], v[94:97], a[208:211], v[168], v[160] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[134:137], v[102:105], a[212:215], v[168], v[160] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[134:137], v[110:113], a[216:219], v[168], v[161] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[134:137], v[118:121], a[220:223], v[168], v[161] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[138:141], v[98:101], a[208:211], v[168], v[160] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[138:141], v[106:109], a[212:215], v[168], v[160] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[138:141], v[114:117], a[216:219], v[168], v[161] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[138:141], v[122:125], a[220:223], v[168], v[161] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[142:145], v[94:97], a[224:227], v[169], v[160] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[142:145], v[102:105], a[228:231], v[169], v[160] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[142:145], v[110:113], a[232:235], v[169], v[161] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[142:145], v[118:121], a[236:239], v[169], v[161] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[146:149], v[98:101], a[224:227], v[169], v[160] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[146:149], v[106:109], a[228:231], v[169], v[160] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[146:149], v[114:117], a[232:235], v[169], v[161] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[146:149], v[122:125], a[236:239], v[169], v[161] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[150:153], v[94:97], a[240:243], v[169], v[160] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[150:153], v[102:105], a[244:247], v[169], v[160] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[150:153], v[110:113], a[248:251], v[169], v[161] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[150:153], v[118:121], a[252:255], v[169], v[161] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[154:157], v[98:101], a[240:243], v[169], v[160] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[154:157], v[106:109], a[244:247], v[169], v[160] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[154:157], v[114:117], a[248:251], v[169], v[161] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[154:157], v[122:125], a[252:255], v[169], v[161] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_i32 s61, s52, 0xfffff200
	s_add_i32 s24, s93, 64
	s_add_i32 s25, s93, 0x10000
	s_add_i32 s92, s93, 0x10040
	s_add_i32 s94, s93, 0x20000
	s_add_i32 s95, s93, 0x20040
	s_add_i32 s96, s93, 0x30000
	s_add_i32 s97, s93, 0x30040
	;;#ASMSTART
	s_waitcnt vmcnt(0) lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v[166], v8
v_mov_b32 v[167], v9
v_mov_b32 v[168], v10
v_mov_b32 v[169], v11
v_mov_b32 v[158], v12
v_mov_b32 v[159], v13
v_mov_b32 v[160], v14
v_mov_b32 v[161], v15

	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[8:9], v5, s[0:3], s61 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[10:11], v5, s[4:7], s61 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[12:13], v5, s[8:11], s61 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[14:15], v5, s[16:19], s61 offen
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[62:65], v[30:33], a[0:3], v[166], v[158] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[94:97], v21 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[62:65], v[38:41], a[4:7], v[166], v[158] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[98:101], v22 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[62:65], v[46:49], a[8:11], v[166], v[159] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[102:105], v21 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[62:65], v[54:57], a[12:15], v[166], v[159] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[106:109], v22 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[66:69], v[34:37], a[0:3], v[166], v[158] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[110:113], v21 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[66:69], v[42:45], a[4:7], v[166], v[158] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[114:117], v22 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[66:69], v[50:53], a[8:11], v[166], v[159] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[118:121], v21 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:69], v[58:61], a[12:15], v[166], v[159] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[122:125], v22 offset:6144
buffer_load_dwordx4 v[126:129], v23, s[20:23], s93 offen
buffer_load_dwordx4 v[130:133], v23, s[20:23], s24 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[70:73], v[30:33], a[16:19], v[166], v[158] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[70:73], v[38:41], a[20:23], v[166], v[158] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[70:73], v[46:49], a[24:27], v[166], v[159] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[70:73], v[54:57], a[28:31], v[166], v[159] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[74:77], v[34:37], a[16:19], v[166], v[158] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[74:77], v[42:45], a[20:23], v[166], v[158] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[134:137], v23, s[20:23], s25 offen
buffer_load_dwordx4 v[138:141], v23, s[20:23], s92 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[74:77], v[50:53], a[24:27], v[166], v[159] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[74:77], v[58:61], a[28:31], v[166], v[159] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[78:81], v[30:33], a[32:35], v[167], v[158] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[78:81], v[38:41], a[36:39], v[167], v[158] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[78:81], v[46:49], a[40:43], v[167], v[159] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[78:81], v[54:57], a[44:47], v[167], v[159] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[142:145], v23, s[20:23], s94 offen
buffer_load_dwordx4 v[146:149], v23, s[20:23], s95 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[82:85], v[34:37], a[32:35], v[167], v[158] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[82:85], v[42:45], a[36:39], v[167], v[158] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[82:85], v[50:53], a[40:43], v[167], v[159] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[82:85], v[58:61], a[44:47], v[167], v[159] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[86:89], v[30:33], a[48:51], v[167], v[158] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[86:89], v[38:41], a[52:55], v[167], v[158] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[150:153], v23, s[20:23], s96 offen
buffer_load_dwordx4 v[154:157], v23, s[20:23], s97 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[86:89], v[46:49], a[56:59], v[167], v[159] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[86:89], v[54:57], a[60:63], v[167], v[159] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[90:93], v[34:37], a[48:51], v[167], v[158] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[90:93], v[42:45], a[52:55], v[167], v[158] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[90:93], v[50:53], a[56:59], v[167], v[159] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[90:93], v[58:61], a[60:63], v[167], v[159] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[62:65], v[94:97], a[64:67], v[166], v[160] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[62:65], v[102:105], a[68:71], v[166], v[160] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[62:65], v[110:113], a[72:75], v[166], v[161] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[62:65], v[118:121], a[76:79], v[166], v[161] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[66:69], v[98:101], a[64:67], v[166], v[160] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[66:69], v[106:109], a[68:71], v[166], v[160] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[66:69], v[114:117], a[72:75], v[166], v[161] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[66:69], v[122:125], a[76:79], v[166], v[161] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[70:73], v[94:97], a[80:83], v[166], v[160] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[70:73], v[102:105], a[84:87], v[166], v[160] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[70:73], v[110:113], a[88:91], v[166], v[161] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[70:73], v[118:121], a[92:95], v[166], v[161] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[74:77], v[98:101], a[80:83], v[166], v[160] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[74:77], v[106:109], a[84:87], v[166], v[160] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[74:77], v[114:117], a[88:91], v[166], v[161] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[74:77], v[122:125], a[92:95], v[166], v[161] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[78:81], v[94:97], a[96:99], v[167], v[160] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[78:81], v[102:105], a[100:103], v[167], v[160] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[78:81], v[110:113], a[104:107], v[167], v[161] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[78:81], v[118:121], a[108:111], v[167], v[161] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[82:85], v[98:101], a[96:99], v[167], v[160] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[82:85], v[106:109], a[100:103], v[167], v[160] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[82:85], v[114:117], a[104:107], v[167], v[161] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[82:85], v[122:125], a[108:111], v[167], v[161] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[86:89], v[94:97], a[112:115], v[167], v[160] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[86:89], v[102:105], a[116:119], v[167], v[160] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[86:89], v[110:113], a[120:123], v[167], v[161] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[86:89], v[118:121], a[124:127], v[167], v[161] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[90:93], v[98:101], a[112:115], v[167], v[160] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[90:93], v[106:109], a[116:119], v[167], v[160] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[90:93], v[114:117], a[120:123], v[167], v[161] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[90:93], v[122:125], a[124:127], v[167], v[161] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_mov_b32 m0, s33
	s_add_i32 s89, s53, 0x400
	buffer_load_dwordx4 v2, s[12:15], s85 offen lds
	s_mov_b32 m0, s43
	s_add_i32 s91, s53, 0x80400
	buffer_load_dwordx4 v1, s[12:15], s85 offen lds
	s_mov_b32 m0, s44
	s_add_i32 s61, s89, 64
	buffer_load_dwordx4 v3, s[12:15], s85 offen lds
	s_mov_b32 m0, s45
	s_add_i32 s24, s89, 0x10000
	buffer_load_dwordx4 v4, s[12:15], s85 offen lds
	s_mov_b32 m0, s35
	s_add_i32 s25, s89, 0x10040
	buffer_load_dwordx4 v2, s[12:15], s82 offen lds
	s_mov_b32 m0, s46
	s_add_i32 s92, s89, 0x20000
	buffer_load_dwordx4 v1, s[12:15], s82 offen lds
	s_mov_b32 m0, s47
	s_add_i32 s93, s89, 0x20040
	buffer_load_dwordx4 v3, s[12:15], s82 offen lds
	s_mov_b32 m0, s48
	s_add_i32 s94, s89, 0x30000
	buffer_load_dwordx4 v4, s[12:15], s82 offen lds
	s_add_i32 s95, s89, 0x30040
	;;#ASMSTART
	s_waitcnt vmcnt(0)
s_barrier
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[126:129], v[30:33], a[128:131], v[168], v[158] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[126:129], v[38:41], a[132:135], v[168], v[158] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[62:65], v23, s[20:23], s89 offen
buffer_load_dwordx4 v[66:69], v23, s[20:23], s61 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[126:129], v[46:49], a[136:139], v[168], v[159] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[126:129], v[54:57], a[140:143], v[168], v[159] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[130:133], v[34:37], a[128:131], v[168], v[158] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[130:133], v[42:45], a[132:135], v[168], v[158] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[70:73], v23, s[20:23], s24 offen
buffer_load_dwordx4 v[74:77], v23, s[20:23], s25 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[130:133], v[50:53], a[136:139], v[168], v[159] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[130:133], v[58:61], a[140:143], v[168], v[159] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[134:137], v[30:33], a[144:147], v[168], v[158] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[134:137], v[38:41], a[148:151], v[168], v[158] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[78:81], v23, s[20:23], s92 offen
buffer_load_dwordx4 v[82:85], v23, s[20:23], s93 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[134:137], v[46:49], a[152:155], v[168], v[159] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[134:137], v[54:57], a[156:159], v[168], v[159] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[138:141], v[34:37], a[144:147], v[168], v[158] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[138:141], v[42:45], a[148:151], v[168], v[158] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[86:89], v23, s[20:23], s94 offen
buffer_load_dwordx4 v[90:93], v23, s[20:23], s95 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[138:141], v[50:53], a[152:155], v[168], v[159] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[138:141], v[58:61], a[156:159], v[168], v[159] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[142:145], v[30:33], a[160:163], v[169], v[158] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[142:145], v[38:41], a[164:167], v[169], v[158] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[142:145], v[46:49], a[168:171], v[169], v[159] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[142:145], v[54:57], a[172:175], v[169], v[159] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[146:149], v[34:37], a[160:163], v[169], v[158] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[146:149], v[42:45], a[164:167], v[169], v[158] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[146:149], v[50:53], a[168:171], v[169], v[159] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[146:149], v[58:61], a[172:175], v[169], v[159] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[150:153], v[30:33], a[176:179], v[169], v[158] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[150:153], v[38:41], a[180:183], v[169], v[158] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[150:153], v[46:49], a[184:187], v[169], v[159] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[150:153], v[54:57], a[188:191], v[169], v[159] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[154:157], v[34:37], a[176:179], v[169], v[158] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[154:157], v[42:45], a[180:183], v[169], v[158] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[154:157], v[50:53], a[184:187], v[169], v[159] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[154:157], v[58:61], a[188:191], v[169], v[159] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[126:129], v[94:97], a[192:195], v[168], v[160] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v7 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[126:129], v[102:105], a[196:199], v[168], v[160] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v16 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[126:129], v[110:113], a[200:203], v[168], v[161] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v7 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[126:129], v[118:121], a[204:207], v[168], v[161] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v16 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[130:133], v[98:101], a[192:195], v[168], v[160] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v7 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[130:133], v[106:109], a[196:199], v[168], v[160] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v16 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[130:133], v[114:117], a[200:203], v[168], v[161] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v7 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[130:133], v[122:125], a[204:207], v[168], v[161] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v16 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[134:137], v[94:97], a[208:211], v[168], v[160] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[134:137], v[102:105], a[212:215], v[168], v[160] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[134:137], v[110:113], a[216:219], v[168], v[161] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[134:137], v[118:121], a[220:223], v[168], v[161] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[138:141], v[98:101], a[208:211], v[168], v[160] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[138:141], v[106:109], a[212:215], v[168], v[160] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[138:141], v[114:117], a[216:219], v[168], v[161] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[138:141], v[122:125], a[220:223], v[168], v[161] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[142:145], v[94:97], a[224:227], v[169], v[160] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[142:145], v[102:105], a[228:231], v[169], v[160] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[142:145], v[110:113], a[232:235], v[169], v[161] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[142:145], v[118:121], a[236:239], v[169], v[161] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[146:149], v[98:101], a[224:227], v[169], v[160] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[146:149], v[106:109], a[228:231], v[169], v[160] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[146:149], v[114:117], a[232:235], v[169], v[161] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[146:149], v[122:125], a[236:239], v[169], v[161] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[150:153], v[94:97], a[240:243], v[169], v[160] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[150:153], v[102:105], a[244:247], v[169], v[160] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[150:153], v[110:113], a[248:251], v[169], v[161] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[150:153], v[118:121], a[252:255], v[169], v[161] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[154:157], v[98:101], a[240:243], v[169], v[160] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[154:157], v[106:109], a[244:247], v[169], v[160] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[154:157], v[114:117], a[248:251], v[169], v[161] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[154:157], v[122:125], a[252:255], v[169], v[161] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_i32 s63, s52, 0xfffff400
	s_add_i32 s96, s91, 64
	s_add_i32 s97, s91, 0x10000
	s_add_i32 s85, s91, 0x10040
	s_add_i32 s82, s91, 0x20000
	s_add_i32 s24, s91, 0x20040
	s_add_i32 s25, s91, 0x30000
	s_add_i32 s61, s91, 0x30040
	;;#ASMSTART
	s_waitcnt vmcnt(0) lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v[166], v8
v_mov_b32 v[167], v9
v_mov_b32 v[168], v10
v_mov_b32 v[169], v11
v_mov_b32 v[158], v12
v_mov_b32 v[159], v13
v_mov_b32 v[160], v14
v_mov_b32 v[161], v15

	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[8:9], v5, s[0:3], s63 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[10:11], v5, s[4:7], s63 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[12:13], v5, s[8:11], s63 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[14:15], v5, s[16:19], s63 offen
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[62:65], v[30:33], a[0:3], v[166], v[158] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[94:97], v19 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[62:65], v[38:41], a[4:7], v[166], v[158] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[98:101], v20 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[62:65], v[46:49], a[8:11], v[166], v[159] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[102:105], v19 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[62:65], v[54:57], a[12:15], v[166], v[159] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[106:109], v20 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[66:69], v[34:37], a[0:3], v[166], v[158] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[110:113], v19 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[66:69], v[42:45], a[4:7], v[166], v[158] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[114:117], v20 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[66:69], v[50:53], a[8:11], v[166], v[159] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[118:121], v19 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:69], v[58:61], a[12:15], v[166], v[159] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[122:125], v20 offset:6144
buffer_load_dwordx4 v[126:129], v23, s[20:23], s91 offen
buffer_load_dwordx4 v[130:133], v23, s[20:23], s96 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[70:73], v[30:33], a[16:19], v[166], v[158] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[70:73], v[38:41], a[20:23], v[166], v[158] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[70:73], v[46:49], a[24:27], v[166], v[159] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[70:73], v[54:57], a[28:31], v[166], v[159] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[74:77], v[34:37], a[16:19], v[166], v[158] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[74:77], v[42:45], a[20:23], v[166], v[158] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[134:137], v23, s[20:23], s97 offen
buffer_load_dwordx4 v[138:141], v23, s[20:23], s85 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[74:77], v[50:53], a[24:27], v[166], v[159] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[74:77], v[58:61], a[28:31], v[166], v[159] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[78:81], v[30:33], a[32:35], v[167], v[158] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[78:81], v[38:41], a[36:39], v[167], v[158] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[78:81], v[46:49], a[40:43], v[167], v[159] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[78:81], v[54:57], a[44:47], v[167], v[159] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[142:145], v23, s[20:23], s82 offen
buffer_load_dwordx4 v[146:149], v23, s[20:23], s24 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[82:85], v[34:37], a[32:35], v[167], v[158] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[82:85], v[42:45], a[36:39], v[167], v[158] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[82:85], v[50:53], a[40:43], v[167], v[159] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[82:85], v[58:61], a[44:47], v[167], v[159] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[86:89], v[30:33], a[48:51], v[167], v[158] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[86:89], v[38:41], a[52:55], v[167], v[158] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[150:153], v23, s[20:23], s25 offen
buffer_load_dwordx4 v[154:157], v23, s[20:23], s61 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[86:89], v[46:49], a[56:59], v[167], v[159] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[86:89], v[54:57], a[60:63], v[167], v[159] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[90:93], v[34:37], a[48:51], v[167], v[158] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[90:93], v[42:45], a[52:55], v[167], v[158] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[90:93], v[50:53], a[56:59], v[167], v[159] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[90:93], v[58:61], a[60:63], v[167], v[159] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[62:65], v[94:97], a[64:67], v[166], v[160] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[62:65], v[102:105], a[68:71], v[166], v[160] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[62:65], v[110:113], a[72:75], v[166], v[161] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[62:65], v[118:121], a[76:79], v[166], v[161] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[66:69], v[98:101], a[64:67], v[166], v[160] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[66:69], v[106:109], a[68:71], v[166], v[160] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[66:69], v[114:117], a[72:75], v[166], v[161] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[66:69], v[122:125], a[76:79], v[166], v[161] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[70:73], v[94:97], a[80:83], v[166], v[160] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[70:73], v[102:105], a[84:87], v[166], v[160] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[70:73], v[110:113], a[88:91], v[166], v[161] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[70:73], v[118:121], a[92:95], v[166], v[161] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[74:77], v[98:101], a[80:83], v[166], v[160] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[74:77], v[106:109], a[84:87], v[166], v[160] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[74:77], v[114:117], a[88:91], v[166], v[161] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[74:77], v[122:125], a[92:95], v[166], v[161] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[78:81], v[94:97], a[96:99], v[167], v[160] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[78:81], v[102:105], a[100:103], v[167], v[160] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[78:81], v[110:113], a[104:107], v[167], v[161] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[78:81], v[118:121], a[108:111], v[167], v[161] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[82:85], v[98:101], a[96:99], v[167], v[160] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[82:85], v[106:109], a[100:103], v[167], v[160] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[82:85], v[114:117], a[104:107], v[167], v[161] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[82:85], v[122:125], a[108:111], v[167], v[161] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[86:89], v[94:97], a[112:115], v[167], v[160] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[86:89], v[102:105], a[116:119], v[167], v[160] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[86:89], v[110:113], a[120:123], v[167], v[161] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[86:89], v[118:121], a[124:127], v[167], v[161] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[90:93], v[98:101], a[112:115], v[167], v[160] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[90:93], v[106:109], a[116:119], v[167], v[160] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[90:93], v[114:117], a[120:123], v[167], v[161] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[90:93], v[122:125], a[124:127], v[167], v[161] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_mov_b32 m0, s29
	s_add_i32 s87, s53, 0x480
	buffer_load_dwordx4 v2, s[12:15], s84 offen lds
	s_mov_b32 m0, s37
	s_add_i32 s90, s53, 0x80480
	buffer_load_dwordx4 v1, s[12:15], s84 offen lds
	s_mov_b32 m0, s38
	s_add_i32 s89, s87, 64
	buffer_load_dwordx4 v3, s[12:15], s84 offen lds
	s_mov_b32 m0, s39
	s_add_i32 s92, s87, 0x10000
	buffer_load_dwordx4 v4, s[12:15], s84 offen lds
	s_mov_b32 m0, s34
	s_add_i32 s93, s87, 0x10040
	buffer_load_dwordx4 v2, s[12:15], s81 offen lds
	s_mov_b32 m0, s40
	s_add_i32 s94, s87, 0x20000
	buffer_load_dwordx4 v1, s[12:15], s81 offen lds
	s_mov_b32 m0, s41
	s_add_i32 s95, s87, 0x20040
	buffer_load_dwordx4 v3, s[12:15], s81 offen lds
	s_mov_b32 m0, s42
	s_add_i32 s63, s87, 0x30000
	buffer_load_dwordx4 v4, s[12:15], s81 offen lds
	s_add_i32 s24, s87, 0x30040
	;;#ASMSTART
	s_waitcnt vmcnt(0)
s_barrier
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[126:129], v[30:33], a[128:131], v[168], v[158] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[126:129], v[38:41], a[132:135], v[168], v[158] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[62:65], v23, s[20:23], s87 offen
buffer_load_dwordx4 v[66:69], v23, s[20:23], s89 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[126:129], v[46:49], a[136:139], v[168], v[159] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[126:129], v[54:57], a[140:143], v[168], v[159] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[130:133], v[34:37], a[128:131], v[168], v[158] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[130:133], v[42:45], a[132:135], v[168], v[158] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[70:73], v23, s[20:23], s92 offen
buffer_load_dwordx4 v[74:77], v23, s[20:23], s93 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[130:133], v[50:53], a[136:139], v[168], v[159] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[130:133], v[58:61], a[140:143], v[168], v[159] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[134:137], v[30:33], a[144:147], v[168], v[158] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[134:137], v[38:41], a[148:151], v[168], v[158] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[78:81], v23, s[20:23], s94 offen
buffer_load_dwordx4 v[82:85], v23, s[20:23], s95 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[134:137], v[46:49], a[152:155], v[168], v[159] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[134:137], v[54:57], a[156:159], v[168], v[159] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[138:141], v[34:37], a[144:147], v[168], v[158] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[138:141], v[42:45], a[148:151], v[168], v[158] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[86:89], v23, s[20:23], s63 offen
buffer_load_dwordx4 v[90:93], v23, s[20:23], s24 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[138:141], v[50:53], a[152:155], v[168], v[159] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[138:141], v[58:61], a[156:159], v[168], v[159] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[142:145], v[30:33], a[160:163], v[169], v[158] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[142:145], v[38:41], a[164:167], v[169], v[158] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[142:145], v[46:49], a[168:171], v[169], v[159] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[142:145], v[54:57], a[172:175], v[169], v[159] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[146:149], v[34:37], a[160:163], v[169], v[158] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[146:149], v[42:45], a[164:167], v[169], v[158] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[146:149], v[50:53], a[168:171], v[169], v[159] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[146:149], v[58:61], a[172:175], v[169], v[159] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[150:153], v[30:33], a[176:179], v[169], v[158] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[150:153], v[38:41], a[180:183], v[169], v[158] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[150:153], v[46:49], a[184:187], v[169], v[159] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[150:153], v[54:57], a[188:191], v[169], v[159] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[154:157], v[34:37], a[176:179], v[169], v[158] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[154:157], v[42:45], a[180:183], v[169], v[158] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[154:157], v[50:53], a[184:187], v[169], v[159] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[154:157], v[58:61], a[188:191], v[169], v[159] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[126:129], v[94:97], a[192:195], v[168], v[160] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v17 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[126:129], v[102:105], a[196:199], v[168], v[160] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v18 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[126:129], v[110:113], a[200:203], v[168], v[161] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v17 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[126:129], v[118:121], a[204:207], v[168], v[161] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v18 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[130:133], v[98:101], a[192:195], v[168], v[160] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v17 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[130:133], v[106:109], a[196:199], v[168], v[160] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v18 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[130:133], v[114:117], a[200:203], v[168], v[161] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v17 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[130:133], v[122:125], a[204:207], v[168], v[161] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v18 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[134:137], v[94:97], a[208:211], v[168], v[160] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[134:137], v[102:105], a[212:215], v[168], v[160] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[134:137], v[110:113], a[216:219], v[168], v[161] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[134:137], v[118:121], a[220:223], v[168], v[161] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[138:141], v[98:101], a[208:211], v[168], v[160] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[138:141], v[106:109], a[212:215], v[168], v[160] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[138:141], v[114:117], a[216:219], v[168], v[161] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[138:141], v[122:125], a[220:223], v[168], v[161] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[142:145], v[94:97], a[224:227], v[169], v[160] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[142:145], v[102:105], a[228:231], v[169], v[160] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[142:145], v[110:113], a[232:235], v[169], v[161] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[142:145], v[118:121], a[236:239], v[169], v[161] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[146:149], v[98:101], a[224:227], v[169], v[160] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[146:149], v[106:109], a[228:231], v[169], v[160] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[146:149], v[114:117], a[232:235], v[169], v[161] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[146:149], v[122:125], a[236:239], v[169], v[161] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[150:153], v[94:97], a[240:243], v[169], v[160] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[150:153], v[102:105], a[244:247], v[169], v[160] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[150:153], v[110:113], a[248:251], v[169], v[161] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[150:153], v[118:121], a[252:255], v[169], v[161] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[154:157], v[98:101], a[240:243], v[169], v[160] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[154:157], v[106:109], a[244:247], v[169], v[160] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[154:157], v[114:117], a[248:251], v[169], v[161] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[154:157], v[122:125], a[252:255], v[169], v[161] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_i32 s65, s52, 0xfffff600
	s_add_i32 s25, s90, 64
	s_add_i32 s61, s90, 0x10000
	s_add_i32 s82, s90, 0x10040
	s_add_i32 s85, s90, 0x20000
	s_add_i32 s91, s90, 0x20040
	s_add_i32 s96, s90, 0x30000
	s_add_i32 s97, s90, 0x30040
	;;#ASMSTART
	s_waitcnt vmcnt(0) lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v[166], v8
v_mov_b32 v[167], v9
v_mov_b32 v[168], v10
v_mov_b32 v[169], v11
v_mov_b32 v[158], v12
v_mov_b32 v[159], v13
v_mov_b32 v[160], v14
v_mov_b32 v[161], v15

	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[8:9], v5, s[0:3], s65 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[10:11], v5, s[4:7], s65 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[12:13], v5, s[8:11], s65 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[14:15], v5, s[16:19], s65 offen
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[62:65], v[30:33], a[0:3], v[166], v[158] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[94:97], v21 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[62:65], v[38:41], a[4:7], v[166], v[158] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[98:101], v22 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[62:65], v[46:49], a[8:11], v[166], v[159] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[102:105], v21 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[62:65], v[54:57], a[12:15], v[166], v[159] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[106:109], v22 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[66:69], v[34:37], a[0:3], v[166], v[158] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[110:113], v21 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[66:69], v[42:45], a[4:7], v[166], v[158] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[114:117], v22 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[66:69], v[50:53], a[8:11], v[166], v[159] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[118:121], v21 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:69], v[58:61], a[12:15], v[166], v[159] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[122:125], v22 offset:6144
buffer_load_dwordx4 v[126:129], v23, s[20:23], s90 offen
buffer_load_dwordx4 v[130:133], v23, s[20:23], s25 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[70:73], v[30:33], a[16:19], v[166], v[158] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[70:73], v[38:41], a[20:23], v[166], v[158] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[70:73], v[46:49], a[24:27], v[166], v[159] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[70:73], v[54:57], a[28:31], v[166], v[159] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[74:77], v[34:37], a[16:19], v[166], v[158] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[74:77], v[42:45], a[20:23], v[166], v[158] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[134:137], v23, s[20:23], s61 offen
buffer_load_dwordx4 v[138:141], v23, s[20:23], s82 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[74:77], v[50:53], a[24:27], v[166], v[159] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[74:77], v[58:61], a[28:31], v[166], v[159] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[78:81], v[30:33], a[32:35], v[167], v[158] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[78:81], v[38:41], a[36:39], v[167], v[158] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[78:81], v[46:49], a[40:43], v[167], v[159] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[78:81], v[54:57], a[44:47], v[167], v[159] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[142:145], v23, s[20:23], s85 offen
buffer_load_dwordx4 v[146:149], v23, s[20:23], s91 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[82:85], v[34:37], a[32:35], v[167], v[158] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[82:85], v[42:45], a[36:39], v[167], v[158] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[82:85], v[50:53], a[40:43], v[167], v[159] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[82:85], v[58:61], a[44:47], v[167], v[159] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[86:89], v[30:33], a[48:51], v[167], v[158] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[86:89], v[38:41], a[52:55], v[167], v[158] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[150:153], v23, s[20:23], s96 offen
buffer_load_dwordx4 v[154:157], v23, s[20:23], s97 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[86:89], v[46:49], a[56:59], v[167], v[159] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[86:89], v[54:57], a[60:63], v[167], v[159] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[90:93], v[34:37], a[48:51], v[167], v[158] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[90:93], v[42:45], a[52:55], v[167], v[158] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[90:93], v[50:53], a[56:59], v[167], v[159] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[90:93], v[58:61], a[60:63], v[167], v[159] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[62:65], v[94:97], a[64:67], v[166], v[160] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[62:65], v[102:105], a[68:71], v[166], v[160] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[62:65], v[110:113], a[72:75], v[166], v[161] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[62:65], v[118:121], a[76:79], v[166], v[161] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[66:69], v[98:101], a[64:67], v[166], v[160] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[66:69], v[106:109], a[68:71], v[166], v[160] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[66:69], v[114:117], a[72:75], v[166], v[161] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[66:69], v[122:125], a[76:79], v[166], v[161] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[70:73], v[94:97], a[80:83], v[166], v[160] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[70:73], v[102:105], a[84:87], v[166], v[160] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[70:73], v[110:113], a[88:91], v[166], v[161] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[70:73], v[118:121], a[92:95], v[166], v[161] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[74:77], v[98:101], a[80:83], v[166], v[160] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[74:77], v[106:109], a[84:87], v[166], v[160] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[74:77], v[114:117], a[88:91], v[166], v[161] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[74:77], v[122:125], a[92:95], v[166], v[161] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[78:81], v[94:97], a[96:99], v[167], v[160] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[78:81], v[102:105], a[100:103], v[167], v[160] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[78:81], v[110:113], a[104:107], v[167], v[161] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[78:81], v[118:121], a[108:111], v[167], v[161] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[82:85], v[98:101], a[96:99], v[167], v[160] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[82:85], v[106:109], a[100:103], v[167], v[160] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[82:85], v[114:117], a[104:107], v[167], v[161] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[82:85], v[122:125], a[108:111], v[167], v[161] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[86:89], v[94:97], a[112:115], v[167], v[160] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[86:89], v[102:105], a[116:119], v[167], v[160] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[86:89], v[110:113], a[120:123], v[167], v[161] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[86:89], v[118:121], a[124:127], v[167], v[161] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[90:93], v[98:101], a[112:115], v[167], v[160] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[90:93], v[106:109], a[116:119], v[167], v[160] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[90:93], v[114:117], a[120:123], v[167], v[161] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[90:93], v[122:125], a[124:127], v[167], v[161] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_mov_b32 m0, s33
	s_add_i32 s79, s53, 0x500
	buffer_load_dwordx4 v2, s[12:15], s76 offen lds
	s_mov_b32 m0, s43
	s_add_i32 s88, s53, 0x80500
	buffer_load_dwordx4 v1, s[12:15], s76 offen lds
	s_mov_b32 m0, s44
	s_add_i32 s84, s79, 64
	buffer_load_dwordx4 v3, s[12:15], s76 offen lds
	s_mov_b32 m0, s45
	s_add_i32 s81, s79, 0x10000
	buffer_load_dwordx4 v4, s[12:15], s76 offen lds
	s_mov_b32 m0, s35
	s_add_i32 s24, s79, 0x10040
	buffer_load_dwordx4 v2, s[12:15], s80 offen lds
	s_mov_b32 m0, s46
	s_add_i32 s63, s79, 0x20000
	buffer_load_dwordx4 v1, s[12:15], s80 offen lds
	s_mov_b32 m0, s47
	s_add_i32 s87, s79, 0x20040
	buffer_load_dwordx4 v3, s[12:15], s80 offen lds
	s_mov_b32 m0, s48
	s_add_i32 s89, s79, 0x30000
	buffer_load_dwordx4 v4, s[12:15], s80 offen lds
	s_add_i32 s92, s79, 0x30040
	;;#ASMSTART
	s_waitcnt vmcnt(0)
s_barrier
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[126:129], v[30:33], a[128:131], v[168], v[158] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[126:129], v[38:41], a[132:135], v[168], v[158] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[62:65], v23, s[20:23], s79 offen
buffer_load_dwordx4 v[66:69], v23, s[20:23], s84 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[126:129], v[46:49], a[136:139], v[168], v[159] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[126:129], v[54:57], a[140:143], v[168], v[159] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[130:133], v[34:37], a[128:131], v[168], v[158] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[130:133], v[42:45], a[132:135], v[168], v[158] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[70:73], v23, s[20:23], s81 offen
buffer_load_dwordx4 v[74:77], v23, s[20:23], s24 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[130:133], v[50:53], a[136:139], v[168], v[159] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[130:133], v[58:61], a[140:143], v[168], v[159] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[134:137], v[30:33], a[144:147], v[168], v[158] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[134:137], v[38:41], a[148:151], v[168], v[158] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[78:81], v23, s[20:23], s63 offen
buffer_load_dwordx4 v[82:85], v23, s[20:23], s87 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[134:137], v[46:49], a[152:155], v[168], v[159] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[134:137], v[54:57], a[156:159], v[168], v[159] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[138:141], v[34:37], a[144:147], v[168], v[158] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[138:141], v[42:45], a[148:151], v[168], v[158] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[86:89], v23, s[20:23], s89 offen
buffer_load_dwordx4 v[90:93], v23, s[20:23], s92 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[138:141], v[50:53], a[152:155], v[168], v[159] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[138:141], v[58:61], a[156:159], v[168], v[159] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[142:145], v[30:33], a[160:163], v[169], v[158] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[142:145], v[38:41], a[164:167], v[169], v[158] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[142:145], v[46:49], a[168:171], v[169], v[159] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[142:145], v[54:57], a[172:175], v[169], v[159] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[146:149], v[34:37], a[160:163], v[169], v[158] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[146:149], v[42:45], a[164:167], v[169], v[158] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[146:149], v[50:53], a[168:171], v[169], v[159] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[146:149], v[58:61], a[172:175], v[169], v[159] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[150:153], v[30:33], a[176:179], v[169], v[158] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[150:153], v[38:41], a[180:183], v[169], v[158] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[150:153], v[46:49], a[184:187], v[169], v[159] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[150:153], v[54:57], a[188:191], v[169], v[159] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[154:157], v[34:37], a[176:179], v[169], v[158] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[154:157], v[42:45], a[180:183], v[169], v[158] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[154:157], v[50:53], a[184:187], v[169], v[159] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[154:157], v[58:61], a[188:191], v[169], v[159] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[126:129], v[94:97], a[192:195], v[168], v[160] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v7 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[126:129], v[102:105], a[196:199], v[168], v[160] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v16 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[126:129], v[110:113], a[200:203], v[168], v[161] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v7 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[126:129], v[118:121], a[204:207], v[168], v[161] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v16 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[130:133], v[98:101], a[192:195], v[168], v[160] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v7 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[130:133], v[106:109], a[196:199], v[168], v[160] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v16 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[130:133], v[114:117], a[200:203], v[168], v[161] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v7 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[130:133], v[122:125], a[204:207], v[168], v[161] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v16 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[134:137], v[94:97], a[208:211], v[168], v[160] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[134:137], v[102:105], a[212:215], v[168], v[160] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[134:137], v[110:113], a[216:219], v[168], v[161] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[134:137], v[118:121], a[220:223], v[168], v[161] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[138:141], v[98:101], a[208:211], v[168], v[160] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[138:141], v[106:109], a[212:215], v[168], v[160] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[138:141], v[114:117], a[216:219], v[168], v[161] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[138:141], v[122:125], a[220:223], v[168], v[161] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[142:145], v[94:97], a[224:227], v[169], v[160] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[142:145], v[102:105], a[228:231], v[169], v[160] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[142:145], v[110:113], a[232:235], v[169], v[161] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[142:145], v[118:121], a[236:239], v[169], v[161] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[146:149], v[98:101], a[224:227], v[169], v[160] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[146:149], v[106:109], a[228:231], v[169], v[160] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[146:149], v[114:117], a[232:235], v[169], v[161] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[146:149], v[122:125], a[236:239], v[169], v[161] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[150:153], v[94:97], a[240:243], v[169], v[160] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[150:153], v[102:105], a[244:247], v[169], v[160] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[150:153], v[110:113], a[248:251], v[169], v[161] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[150:153], v[118:121], a[252:255], v[169], v[161] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[154:157], v[98:101], a[240:243], v[169], v[160] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[154:157], v[106:109], a[244:247], v[169], v[160] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[154:157], v[114:117], a[248:251], v[169], v[161] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[154:157], v[122:125], a[252:255], v[169], v[161] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_i32 s66, s52, 0xfffff800
	s_add_i32 s93, s88, 64
	s_add_i32 s94, s88, 0x10000
	s_add_i32 s95, s88, 0x10040
	s_add_i32 s65, s88, 0x20000
	s_add_i32 s25, s88, 0x20040
	s_add_i32 s61, s88, 0x30000
	s_add_i32 s82, s88, 0x30040
	;;#ASMSTART
	s_waitcnt vmcnt(0) lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v[166], v8
v_mov_b32 v[167], v9
v_mov_b32 v[168], v10
v_mov_b32 v[169], v11
v_mov_b32 v[158], v12
v_mov_b32 v[159], v13
v_mov_b32 v[160], v14
v_mov_b32 v[161], v15

	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[8:9], v5, s[0:3], s66 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[10:11], v5, s[4:7], s66 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[12:13], v5, s[8:11], s66 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[14:15], v5, s[16:19], s66 offen
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[62:65], v[30:33], a[0:3], v[166], v[158] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[94:97], v19 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[62:65], v[38:41], a[4:7], v[166], v[158] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[98:101], v20 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[62:65], v[46:49], a[8:11], v[166], v[159] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[102:105], v19 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[62:65], v[54:57], a[12:15], v[166], v[159] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[106:109], v20 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[66:69], v[34:37], a[0:3], v[166], v[158] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[110:113], v19 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[66:69], v[42:45], a[4:7], v[166], v[158] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[114:117], v20 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[66:69], v[50:53], a[8:11], v[166], v[159] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[118:121], v19 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:69], v[58:61], a[12:15], v[166], v[159] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[122:125], v20 offset:6144
buffer_load_dwordx4 v[126:129], v23, s[20:23], s88 offen
buffer_load_dwordx4 v[130:133], v23, s[20:23], s93 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[70:73], v[30:33], a[16:19], v[166], v[158] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[70:73], v[38:41], a[20:23], v[166], v[158] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[70:73], v[46:49], a[24:27], v[166], v[159] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[70:73], v[54:57], a[28:31], v[166], v[159] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[74:77], v[34:37], a[16:19], v[166], v[158] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[74:77], v[42:45], a[20:23], v[166], v[158] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[134:137], v23, s[20:23], s94 offen
buffer_load_dwordx4 v[138:141], v23, s[20:23], s95 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[74:77], v[50:53], a[24:27], v[166], v[159] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[74:77], v[58:61], a[28:31], v[166], v[159] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[78:81], v[30:33], a[32:35], v[167], v[158] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[78:81], v[38:41], a[36:39], v[167], v[158] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[78:81], v[46:49], a[40:43], v[167], v[159] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[78:81], v[54:57], a[44:47], v[167], v[159] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[142:145], v23, s[20:23], s65 offen
buffer_load_dwordx4 v[146:149], v23, s[20:23], s25 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[82:85], v[34:37], a[32:35], v[167], v[158] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[82:85], v[42:45], a[36:39], v[167], v[158] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[82:85], v[50:53], a[40:43], v[167], v[159] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[82:85], v[58:61], a[44:47], v[167], v[159] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[86:89], v[30:33], a[48:51], v[167], v[158] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[86:89], v[38:41], a[52:55], v[167], v[158] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[150:153], v23, s[20:23], s61 offen
buffer_load_dwordx4 v[154:157], v23, s[20:23], s82 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[86:89], v[46:49], a[56:59], v[167], v[159] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[86:89], v[54:57], a[60:63], v[167], v[159] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[90:93], v[34:37], a[48:51], v[167], v[158] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[90:93], v[42:45], a[52:55], v[167], v[158] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[90:93], v[50:53], a[56:59], v[167], v[159] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[90:93], v[58:61], a[60:63], v[167], v[159] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[62:65], v[94:97], a[64:67], v[166], v[160] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[62:65], v[102:105], a[68:71], v[166], v[160] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[62:65], v[110:113], a[72:75], v[166], v[161] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[62:65], v[118:121], a[76:79], v[166], v[161] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[66:69], v[98:101], a[64:67], v[166], v[160] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[66:69], v[106:109], a[68:71], v[166], v[160] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[66:69], v[114:117], a[72:75], v[166], v[161] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[66:69], v[122:125], a[76:79], v[166], v[161] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[70:73], v[94:97], a[80:83], v[166], v[160] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[70:73], v[102:105], a[84:87], v[166], v[160] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[70:73], v[110:113], a[88:91], v[166], v[161] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[70:73], v[118:121], a[92:95], v[166], v[161] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[74:77], v[98:101], a[80:83], v[166], v[160] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[74:77], v[106:109], a[84:87], v[166], v[160] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[74:77], v[114:117], a[88:91], v[166], v[161] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[74:77], v[122:125], a[92:95], v[166], v[161] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[78:81], v[94:97], a[96:99], v[167], v[160] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[78:81], v[102:105], a[100:103], v[167], v[160] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[78:81], v[110:113], a[104:107], v[167], v[161] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[78:81], v[118:121], a[108:111], v[167], v[161] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[82:85], v[98:101], a[96:99], v[167], v[160] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[82:85], v[106:109], a[100:103], v[167], v[160] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[82:85], v[114:117], a[104:107], v[167], v[161] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[82:85], v[122:125], a[108:111], v[167], v[161] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[86:89], v[94:97], a[112:115], v[167], v[160] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[86:89], v[102:105], a[116:119], v[167], v[160] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[86:89], v[110:113], a[120:123], v[167], v[161] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[86:89], v[118:121], a[124:127], v[167], v[161] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[90:93], v[98:101], a[112:115], v[167], v[160] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[90:93], v[106:109], a[116:119], v[167], v[160] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[90:93], v[114:117], a[120:123], v[167], v[161] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[90:93], v[122:125], a[124:127], v[167], v[161] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_mov_b32 m0, s29
	s_add_i32 s78, s53, 0x580
	buffer_load_dwordx4 v2, s[12:15], s75 offen lds
	s_mov_b32 m0, s37
	s_add_i32 s86, s53, 0x80580
	buffer_load_dwordx4 v1, s[12:15], s75 offen lds
	s_mov_b32 m0, s38
	s_add_i32 s85, s78, 64
	buffer_load_dwordx4 v3, s[12:15], s75 offen lds
	s_mov_b32 m0, s39
	s_add_i32 s90, s78, 0x10000
	buffer_load_dwordx4 v4, s[12:15], s75 offen lds
	s_mov_b32 m0, s34
	s_add_i32 s91, s78, 0x10040
	buffer_load_dwordx4 v2, s[12:15], s74 offen lds
	s_mov_b32 m0, s40
	s_add_i32 s96, s78, 0x20000
	buffer_load_dwordx4 v1, s[12:15], s74 offen lds
	s_mov_b32 m0, s41
	s_add_i32 s97, s78, 0x20040
	buffer_load_dwordx4 v3, s[12:15], s74 offen lds
	s_mov_b32 m0, s42
	s_add_i32 s76, s78, 0x30000
	buffer_load_dwordx4 v4, s[12:15], s74 offen lds
	s_add_i32 s80, s78, 0x30040
	;;#ASMSTART
	s_waitcnt vmcnt(0)
s_barrier
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[126:129], v[30:33], a[128:131], v[168], v[158] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[126:129], v[38:41], a[132:135], v[168], v[158] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[62:65], v23, s[20:23], s78 offen
buffer_load_dwordx4 v[66:69], v23, s[20:23], s85 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[126:129], v[46:49], a[136:139], v[168], v[159] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[126:129], v[54:57], a[140:143], v[168], v[159] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[130:133], v[34:37], a[128:131], v[168], v[158] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[130:133], v[42:45], a[132:135], v[168], v[158] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[70:73], v23, s[20:23], s90 offen
buffer_load_dwordx4 v[74:77], v23, s[20:23], s91 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[130:133], v[50:53], a[136:139], v[168], v[159] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[130:133], v[58:61], a[140:143], v[168], v[159] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[134:137], v[30:33], a[144:147], v[168], v[158] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[134:137], v[38:41], a[148:151], v[168], v[158] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[78:81], v23, s[20:23], s96 offen
buffer_load_dwordx4 v[82:85], v23, s[20:23], s97 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[134:137], v[46:49], a[152:155], v[168], v[159] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[134:137], v[54:57], a[156:159], v[168], v[159] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[138:141], v[34:37], a[144:147], v[168], v[158] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[138:141], v[42:45], a[148:151], v[168], v[158] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[86:89], v23, s[20:23], s76 offen
buffer_load_dwordx4 v[90:93], v23, s[20:23], s80 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[138:141], v[50:53], a[152:155], v[168], v[159] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[138:141], v[58:61], a[156:159], v[168], v[159] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[142:145], v[30:33], a[160:163], v[169], v[158] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[142:145], v[38:41], a[164:167], v[169], v[158] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[142:145], v[46:49], a[168:171], v[169], v[159] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[142:145], v[54:57], a[172:175], v[169], v[159] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[146:149], v[34:37], a[160:163], v[169], v[158] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[146:149], v[42:45], a[164:167], v[169], v[158] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[146:149], v[50:53], a[168:171], v[169], v[159] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[146:149], v[58:61], a[172:175], v[169], v[159] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[150:153], v[30:33], a[176:179], v[169], v[158] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[150:153], v[38:41], a[180:183], v[169], v[158] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[150:153], v[46:49], a[184:187], v[169], v[159] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[150:153], v[54:57], a[188:191], v[169], v[159] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[154:157], v[34:37], a[176:179], v[169], v[158] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[154:157], v[42:45], a[180:183], v[169], v[158] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[154:157], v[50:53], a[184:187], v[169], v[159] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[154:157], v[58:61], a[188:191], v[169], v[159] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[126:129], v[94:97], a[192:195], v[168], v[160] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v17 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[126:129], v[102:105], a[196:199], v[168], v[160] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v18 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[126:129], v[110:113], a[200:203], v[168], v[161] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v17 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[126:129], v[118:121], a[204:207], v[168], v[161] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v18 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[130:133], v[98:101], a[192:195], v[168], v[160] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v17 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[130:133], v[106:109], a[196:199], v[168], v[160] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v18 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[130:133], v[114:117], a[200:203], v[168], v[161] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v17 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[130:133], v[122:125], a[204:207], v[168], v[161] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v18 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[134:137], v[94:97], a[208:211], v[168], v[160] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[134:137], v[102:105], a[212:215], v[168], v[160] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[134:137], v[110:113], a[216:219], v[168], v[161] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[134:137], v[118:121], a[220:223], v[168], v[161] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[138:141], v[98:101], a[208:211], v[168], v[160] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[138:141], v[106:109], a[212:215], v[168], v[160] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[138:141], v[114:117], a[216:219], v[168], v[161] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[138:141], v[122:125], a[220:223], v[168], v[161] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[142:145], v[94:97], a[224:227], v[169], v[160] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[142:145], v[102:105], a[228:231], v[169], v[160] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[142:145], v[110:113], a[232:235], v[169], v[161] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[142:145], v[118:121], a[236:239], v[169], v[161] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[146:149], v[98:101], a[224:227], v[169], v[160] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[146:149], v[106:109], a[228:231], v[169], v[160] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[146:149], v[114:117], a[232:235], v[169], v[161] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[146:149], v[122:125], a[236:239], v[169], v[161] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[150:153], v[94:97], a[240:243], v[169], v[160] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[150:153], v[102:105], a[244:247], v[169], v[160] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[150:153], v[110:113], a[248:251], v[169], v[161] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[150:153], v[118:121], a[252:255], v[169], v[161] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[154:157], v[98:101], a[240:243], v[169], v[160] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[154:157], v[106:109], a[244:247], v[169], v[160] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[154:157], v[114:117], a[248:251], v[169], v[161] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[154:157], v[122:125], a[252:255], v[169], v[161] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_i32 s68, s52, 0xfffffa00
	s_add_i32 s24, s86, 64
	s_add_i32 s63, s86, 0x10000
	s_add_i32 s79, s86, 0x10040
	s_add_i32 s81, s86, 0x20000
	s_add_i32 s84, s86, 0x20040
	s_add_i32 s87, s86, 0x30000
	s_add_i32 s89, s86, 0x30040
	;;#ASMSTART
	s_waitcnt vmcnt(0) lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v[166], v8
v_mov_b32 v[167], v9
v_mov_b32 v[168], v10
v_mov_b32 v[169], v11
v_mov_b32 v[158], v12
v_mov_b32 v[159], v13
v_mov_b32 v[160], v14
v_mov_b32 v[161], v15

	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[8:9], v5, s[0:3], s68 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[10:11], v5, s[4:7], s68 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[12:13], v5, s[8:11], s68 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[14:15], v5, s[16:19], s68 offen
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[62:65], v[30:33], a[0:3], v[166], v[158] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[94:97], v21 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[62:65], v[38:41], a[4:7], v[166], v[158] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[98:101], v22 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[62:65], v[46:49], a[8:11], v[166], v[159] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[102:105], v21 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[62:65], v[54:57], a[12:15], v[166], v[159] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[106:109], v22 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[66:69], v[34:37], a[0:3], v[166], v[158] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[110:113], v21 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[66:69], v[42:45], a[4:7], v[166], v[158] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[114:117], v22 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[66:69], v[50:53], a[8:11], v[166], v[159] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[118:121], v21 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:69], v[58:61], a[12:15], v[166], v[159] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[122:125], v22 offset:6144
buffer_load_dwordx4 v[126:129], v23, s[20:23], s86 offen
buffer_load_dwordx4 v[130:133], v23, s[20:23], s24 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[70:73], v[30:33], a[16:19], v[166], v[158] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[70:73], v[38:41], a[20:23], v[166], v[158] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[70:73], v[46:49], a[24:27], v[166], v[159] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[70:73], v[54:57], a[28:31], v[166], v[159] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[74:77], v[34:37], a[16:19], v[166], v[158] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[74:77], v[42:45], a[20:23], v[166], v[158] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[134:137], v23, s[20:23], s63 offen
buffer_load_dwordx4 v[138:141], v23, s[20:23], s79 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[74:77], v[50:53], a[24:27], v[166], v[159] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[74:77], v[58:61], a[28:31], v[166], v[159] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[78:81], v[30:33], a[32:35], v[167], v[158] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[78:81], v[38:41], a[36:39], v[167], v[158] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[78:81], v[46:49], a[40:43], v[167], v[159] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[78:81], v[54:57], a[44:47], v[167], v[159] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[142:145], v23, s[20:23], s81 offen
buffer_load_dwordx4 v[146:149], v23, s[20:23], s84 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[82:85], v[34:37], a[32:35], v[167], v[158] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[82:85], v[42:45], a[36:39], v[167], v[158] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[82:85], v[50:53], a[40:43], v[167], v[159] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[82:85], v[58:61], a[44:47], v[167], v[159] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[86:89], v[30:33], a[48:51], v[167], v[158] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[86:89], v[38:41], a[52:55], v[167], v[158] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[150:153], v23, s[20:23], s87 offen
buffer_load_dwordx4 v[154:157], v23, s[20:23], s89 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[86:89], v[46:49], a[56:59], v[167], v[159] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[86:89], v[54:57], a[60:63], v[167], v[159] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[90:93], v[34:37], a[48:51], v[167], v[158] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[90:93], v[42:45], a[52:55], v[167], v[158] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[90:93], v[50:53], a[56:59], v[167], v[159] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[90:93], v[58:61], a[60:63], v[167], v[159] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[62:65], v[94:97], a[64:67], v[166], v[160] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[62:65], v[102:105], a[68:71], v[166], v[160] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[62:65], v[110:113], a[72:75], v[166], v[161] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[62:65], v[118:121], a[76:79], v[166], v[161] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[66:69], v[98:101], a[64:67], v[166], v[160] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[66:69], v[106:109], a[68:71], v[166], v[160] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[66:69], v[114:117], a[72:75], v[166], v[161] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[66:69], v[122:125], a[76:79], v[166], v[161] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[70:73], v[94:97], a[80:83], v[166], v[160] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[70:73], v[102:105], a[84:87], v[166], v[160] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[70:73], v[110:113], a[88:91], v[166], v[161] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[70:73], v[118:121], a[92:95], v[166], v[161] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[74:77], v[98:101], a[80:83], v[166], v[160] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[74:77], v[106:109], a[84:87], v[166], v[160] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[74:77], v[114:117], a[88:91], v[166], v[161] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[74:77], v[122:125], a[92:95], v[166], v[161] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[78:81], v[94:97], a[96:99], v[167], v[160] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[78:81], v[102:105], a[100:103], v[167], v[160] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[78:81], v[110:113], a[104:107], v[167], v[161] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[78:81], v[118:121], a[108:111], v[167], v[161] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[82:85], v[98:101], a[96:99], v[167], v[160] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[82:85], v[106:109], a[100:103], v[167], v[160] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[82:85], v[114:117], a[104:107], v[167], v[161] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[82:85], v[122:125], a[108:111], v[167], v[161] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[86:89], v[94:97], a[112:115], v[167], v[160] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[86:89], v[102:105], a[116:119], v[167], v[160] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[86:89], v[110:113], a[120:123], v[167], v[161] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[86:89], v[118:121], a[124:127], v[167], v[161] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[90:93], v[98:101], a[112:115], v[167], v[160] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[90:93], v[106:109], a[116:119], v[167], v[160] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[90:93], v[114:117], a[120:123], v[167], v[161] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[90:93], v[122:125], a[124:127], v[167], v[161] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_mov_b32 m0, s33
	s_add_i32 s77, s53, 0x600
	buffer_load_dwordx4 v2, s[12:15], s72 offen lds
	s_mov_b32 m0, s43
	s_add_i32 s83, s53, 0x80600
	buffer_load_dwordx4 v1, s[12:15], s72 offen lds
	s_mov_b32 m0, s44
	s_add_i32 s92, s77, 64
	buffer_load_dwordx4 v3, s[12:15], s72 offen lds
	s_mov_b32 m0, s45
	s_add_i32 s66, s77, 0x10000
	buffer_load_dwordx4 v4, s[12:15], s72 offen lds
	s_mov_b32 m0, s35
	s_add_i32 s25, s77, 0x10040
	buffer_load_dwordx4 v2, s[12:15], s73 offen lds
	s_mov_b32 m0, s46
	s_add_i32 s61, s77, 0x20000
	buffer_load_dwordx4 v1, s[12:15], s73 offen lds
	s_mov_b32 m0, s47
	s_add_i32 s65, s77, 0x20040
	buffer_load_dwordx4 v3, s[12:15], s73 offen lds
	s_mov_b32 m0, s48
	s_add_i32 s82, s77, 0x30000
	buffer_load_dwordx4 v4, s[12:15], s73 offen lds
	s_add_i32 s88, s77, 0x30040
	;;#ASMSTART
	s_waitcnt vmcnt(0)
s_barrier
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[126:129], v[30:33], a[128:131], v[168], v[158] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[126:129], v[38:41], a[132:135], v[168], v[158] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[62:65], v23, s[20:23], s77 offen
buffer_load_dwordx4 v[66:69], v23, s[20:23], s92 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[126:129], v[46:49], a[136:139], v[168], v[159] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[126:129], v[54:57], a[140:143], v[168], v[159] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[130:133], v[34:37], a[128:131], v[168], v[158] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[130:133], v[42:45], a[132:135], v[168], v[158] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[70:73], v23, s[20:23], s66 offen
buffer_load_dwordx4 v[74:77], v23, s[20:23], s25 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[130:133], v[50:53], a[136:139], v[168], v[159] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[130:133], v[58:61], a[140:143], v[168], v[159] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[134:137], v[30:33], a[144:147], v[168], v[158] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[134:137], v[38:41], a[148:151], v[168], v[158] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[78:81], v23, s[20:23], s61 offen
buffer_load_dwordx4 v[82:85], v23, s[20:23], s65 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[134:137], v[46:49], a[152:155], v[168], v[159] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[134:137], v[54:57], a[156:159], v[168], v[159] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[138:141], v[34:37], a[144:147], v[168], v[158] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[138:141], v[42:45], a[148:151], v[168], v[158] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[86:89], v23, s[20:23], s82 offen
buffer_load_dwordx4 v[90:93], v23, s[20:23], s88 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[138:141], v[50:53], a[152:155], v[168], v[159] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[138:141], v[58:61], a[156:159], v[168], v[159] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[142:145], v[30:33], a[160:163], v[169], v[158] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[142:145], v[38:41], a[164:167], v[169], v[158] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[142:145], v[46:49], a[168:171], v[169], v[159] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[142:145], v[54:57], a[172:175], v[169], v[159] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[146:149], v[34:37], a[160:163], v[169], v[158] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[146:149], v[42:45], a[164:167], v[169], v[158] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[146:149], v[50:53], a[168:171], v[169], v[159] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[146:149], v[58:61], a[172:175], v[169], v[159] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[150:153], v[30:33], a[176:179], v[169], v[158] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[150:153], v[38:41], a[180:183], v[169], v[158] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[150:153], v[46:49], a[184:187], v[169], v[159] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[150:153], v[54:57], a[188:191], v[169], v[159] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[154:157], v[34:37], a[176:179], v[169], v[158] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[154:157], v[42:45], a[180:183], v[169], v[158] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[154:157], v[50:53], a[184:187], v[169], v[159] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[154:157], v[58:61], a[188:191], v[169], v[159] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[126:129], v[94:97], a[192:195], v[168], v[160] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v7 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[126:129], v[102:105], a[196:199], v[168], v[160] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v16 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[126:129], v[110:113], a[200:203], v[168], v[161] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v7 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[126:129], v[118:121], a[204:207], v[168], v[161] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v16 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[130:133], v[98:101], a[192:195], v[168], v[160] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v7 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[130:133], v[106:109], a[196:199], v[168], v[160] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v16 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[130:133], v[114:117], a[200:203], v[168], v[161] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v7 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[130:133], v[122:125], a[204:207], v[168], v[161] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v16 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[134:137], v[94:97], a[208:211], v[168], v[160] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[134:137], v[102:105], a[212:215], v[168], v[160] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[134:137], v[110:113], a[216:219], v[168], v[161] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[134:137], v[118:121], a[220:223], v[168], v[161] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[138:141], v[98:101], a[208:211], v[168], v[160] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[138:141], v[106:109], a[212:215], v[168], v[160] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[138:141], v[114:117], a[216:219], v[168], v[161] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[138:141], v[122:125], a[220:223], v[168], v[161] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[142:145], v[94:97], a[224:227], v[169], v[160] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[142:145], v[102:105], a[228:231], v[169], v[160] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[142:145], v[110:113], a[232:235], v[169], v[161] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[142:145], v[118:121], a[236:239], v[169], v[161] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[146:149], v[98:101], a[224:227], v[169], v[160] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[146:149], v[106:109], a[228:231], v[169], v[160] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[146:149], v[114:117], a[232:235], v[169], v[161] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[146:149], v[122:125], a[236:239], v[169], v[161] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[150:153], v[94:97], a[240:243], v[169], v[160] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[150:153], v[102:105], a[244:247], v[169], v[160] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[150:153], v[110:113], a[248:251], v[169], v[161] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[150:153], v[118:121], a[252:255], v[169], v[161] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[154:157], v[98:101], a[240:243], v[169], v[160] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[154:157], v[106:109], a[244:247], v[169], v[160] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[154:157], v[114:117], a[248:251], v[169], v[161] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[154:157], v[122:125], a[252:255], v[169], v[161] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_i32 s69, s52, 0xfffffc00
	s_add_i32 s93, s83, 64
	s_add_i32 s94, s83, 0x10000
	s_add_i32 s95, s83, 0x10040
	s_add_i32 s75, s83, 0x20000
	s_add_i32 s74, s83, 0x20040
	s_add_i32 s76, s83, 0x30000
	s_add_i32 s78, s83, 0x30040
	;;#ASMSTART
	s_waitcnt vmcnt(0) lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v[166], v8
v_mov_b32 v[167], v9
v_mov_b32 v[168], v10
v_mov_b32 v[169], v11
v_mov_b32 v[158], v12
v_mov_b32 v[159], v13
v_mov_b32 v[160], v14
v_mov_b32 v[161], v15

	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[8:9], v5, s[0:3], s69 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[10:11], v5, s[4:7], s69 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[12:13], v5, s[8:11], s69 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[14:15], v5, s[16:19], s69 offen
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[62:65], v[30:33], a[0:3], v[166], v[158] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[94:97], v19 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[62:65], v[38:41], a[4:7], v[166], v[158] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[98:101], v20 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[62:65], v[46:49], a[8:11], v[166], v[159] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[102:105], v19 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[62:65], v[54:57], a[12:15], v[166], v[159] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[106:109], v20 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[66:69], v[34:37], a[0:3], v[166], v[158] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[110:113], v19 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[66:69], v[42:45], a[4:7], v[166], v[158] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[114:117], v20 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[66:69], v[50:53], a[8:11], v[166], v[159] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[118:121], v19 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:69], v[58:61], a[12:15], v[166], v[159] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[122:125], v20 offset:6144
buffer_load_dwordx4 v[126:129], v23, s[20:23], s83 offen
buffer_load_dwordx4 v[130:133], v23, s[20:23], s93 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[70:73], v[30:33], a[16:19], v[166], v[158] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[70:73], v[38:41], a[20:23], v[166], v[158] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[70:73], v[46:49], a[24:27], v[166], v[159] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[70:73], v[54:57], a[28:31], v[166], v[159] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[74:77], v[34:37], a[16:19], v[166], v[158] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[74:77], v[42:45], a[20:23], v[166], v[158] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[134:137], v23, s[20:23], s94 offen
buffer_load_dwordx4 v[138:141], v23, s[20:23], s95 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[74:77], v[50:53], a[24:27], v[166], v[159] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[74:77], v[58:61], a[28:31], v[166], v[159] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[78:81], v[30:33], a[32:35], v[167], v[158] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[78:81], v[38:41], a[36:39], v[167], v[158] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[78:81], v[46:49], a[40:43], v[167], v[159] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[78:81], v[54:57], a[44:47], v[167], v[159] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[142:145], v23, s[20:23], s75 offen
buffer_load_dwordx4 v[146:149], v23, s[20:23], s74 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[82:85], v[34:37], a[32:35], v[167], v[158] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[82:85], v[42:45], a[36:39], v[167], v[158] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[82:85], v[50:53], a[40:43], v[167], v[159] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[82:85], v[58:61], a[44:47], v[167], v[159] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[86:89], v[30:33], a[48:51], v[167], v[158] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[86:89], v[38:41], a[52:55], v[167], v[158] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[150:153], v23, s[20:23], s76 offen
buffer_load_dwordx4 v[154:157], v23, s[20:23], s78 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[86:89], v[46:49], a[56:59], v[167], v[159] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[86:89], v[54:57], a[60:63], v[167], v[159] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[90:93], v[34:37], a[48:51], v[167], v[158] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[90:93], v[42:45], a[52:55], v[167], v[158] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[90:93], v[50:53], a[56:59], v[167], v[159] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[90:93], v[58:61], a[60:63], v[167], v[159] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[62:65], v[94:97], a[64:67], v[166], v[160] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[62:65], v[102:105], a[68:71], v[166], v[160] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[62:65], v[110:113], a[72:75], v[166], v[161] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[62:65], v[118:121], a[76:79], v[166], v[161] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[66:69], v[98:101], a[64:67], v[166], v[160] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[66:69], v[106:109], a[68:71], v[166], v[160] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[66:69], v[114:117], a[72:75], v[166], v[161] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[66:69], v[122:125], a[76:79], v[166], v[161] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[70:73], v[94:97], a[80:83], v[166], v[160] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[70:73], v[102:105], a[84:87], v[166], v[160] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[70:73], v[110:113], a[88:91], v[166], v[161] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[70:73], v[118:121], a[92:95], v[166], v[161] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[74:77], v[98:101], a[80:83], v[166], v[160] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[74:77], v[106:109], a[84:87], v[166], v[160] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[74:77], v[114:117], a[88:91], v[166], v[161] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[74:77], v[122:125], a[92:95], v[166], v[161] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[78:81], v[94:97], a[96:99], v[167], v[160] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[78:81], v[102:105], a[100:103], v[167], v[160] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[78:81], v[110:113], a[104:107], v[167], v[161] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[78:81], v[118:121], a[108:111], v[167], v[161] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[82:85], v[98:101], a[96:99], v[167], v[160] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[82:85], v[106:109], a[100:103], v[167], v[160] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[82:85], v[114:117], a[104:107], v[167], v[161] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[82:85], v[122:125], a[108:111], v[167], v[161] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[86:89], v[94:97], a[112:115], v[167], v[160] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[86:89], v[102:105], a[116:119], v[167], v[160] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[86:89], v[110:113], a[120:123], v[167], v[161] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[86:89], v[118:121], a[124:127], v[167], v[161] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[90:93], v[98:101], a[112:115], v[167], v[160] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[90:93], v[106:109], a[116:119], v[167], v[160] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[90:93], v[114:117], a[120:123], v[167], v[161] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[90:93], v[122:125], a[124:127], v[167], v[161] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_mov_b32 m0, s29
	s_add_i32 s62, s53, 0x680
	buffer_load_dwordx4 v2, s[12:15], s60 offen lds
	s_mov_b32 m0, s37
	s_add_i32 s58, s53, 0x80680
	buffer_load_dwordx4 v1, s[12:15], s60 offen lds
	s_mov_b32 m0, s38
	s_add_i32 s80, s62, 64
	buffer_load_dwordx4 v3, s[12:15], s60 offen lds
	s_mov_b32 m0, s39
	s_add_i32 s85, s62, 0x10000
	buffer_load_dwordx4 v4, s[12:15], s60 offen lds
	s_mov_b32 m0, s34
	s_add_i32 s90, s62, 0x10040
	buffer_load_dwordx4 v2, s[12:15], s59 offen lds
	s_mov_b32 m0, s40
	s_add_i32 s91, s62, 0x20000
	buffer_load_dwordx4 v1, s[12:15], s59 offen lds
	s_mov_b32 m0, s41
	s_add_i32 s96, s62, 0x20040
	buffer_load_dwordx4 v3, s[12:15], s59 offen lds
	s_mov_b32 m0, s42
	s_add_i32 s97, s62, 0x30000
	buffer_load_dwordx4 v4, s[12:15], s59 offen lds
	s_add_i32 s68, s62, 0x30040
	;;#ASMSTART
	s_waitcnt vmcnt(0)
s_barrier
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[126:129], v[30:33], a[128:131], v[168], v[158] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[126:129], v[38:41], a[132:135], v[168], v[158] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[62:65], v23, s[20:23], s62 offen
buffer_load_dwordx4 v[66:69], v23, s[20:23], s80 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[126:129], v[46:49], a[136:139], v[168], v[159] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[126:129], v[54:57], a[140:143], v[168], v[159] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[130:133], v[34:37], a[128:131], v[168], v[158] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[130:133], v[42:45], a[132:135], v[168], v[158] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[70:73], v23, s[20:23], s85 offen
buffer_load_dwordx4 v[74:77], v23, s[20:23], s90 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[130:133], v[50:53], a[136:139], v[168], v[159] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[130:133], v[58:61], a[140:143], v[168], v[159] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[134:137], v[30:33], a[144:147], v[168], v[158] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[134:137], v[38:41], a[148:151], v[168], v[158] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[78:81], v23, s[20:23], s91 offen
buffer_load_dwordx4 v[82:85], v23, s[20:23], s96 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[134:137], v[46:49], a[152:155], v[168], v[159] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[134:137], v[54:57], a[156:159], v[168], v[159] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[138:141], v[34:37], a[144:147], v[168], v[158] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[138:141], v[42:45], a[148:151], v[168], v[158] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[86:89], v23, s[20:23], s97 offen
buffer_load_dwordx4 v[90:93], v23, s[20:23], s68 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[138:141], v[50:53], a[152:155], v[168], v[159] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[138:141], v[58:61], a[156:159], v[168], v[159] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[142:145], v[30:33], a[160:163], v[169], v[158] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[142:145], v[38:41], a[164:167], v[169], v[158] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[142:145], v[46:49], a[168:171], v[169], v[159] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[142:145], v[54:57], a[172:175], v[169], v[159] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[146:149], v[34:37], a[160:163], v[169], v[158] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[146:149], v[42:45], a[164:167], v[169], v[158] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[146:149], v[50:53], a[168:171], v[169], v[159] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[146:149], v[58:61], a[172:175], v[169], v[159] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[150:153], v[30:33], a[176:179], v[169], v[158] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[150:153], v[38:41], a[180:183], v[169], v[158] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[150:153], v[46:49], a[184:187], v[169], v[159] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[150:153], v[54:57], a[188:191], v[169], v[159] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[154:157], v[34:37], a[176:179], v[169], v[158] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[154:157], v[42:45], a[180:183], v[169], v[158] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[154:157], v[50:53], a[184:187], v[169], v[159] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[154:157], v[58:61], a[188:191], v[169], v[159] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[126:129], v[94:97], a[192:195], v[168], v[160] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v17 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[126:129], v[102:105], a[196:199], v[168], v[160] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v18 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[126:129], v[110:113], a[200:203], v[168], v[161] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v17 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[126:129], v[118:121], a[204:207], v[168], v[161] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v18 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[130:133], v[98:101], a[192:195], v[168], v[160] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v17 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[130:133], v[106:109], a[196:199], v[168], v[160] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v18 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[130:133], v[114:117], a[200:203], v[168], v[161] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v17 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[130:133], v[122:125], a[204:207], v[168], v[161] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v18 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[134:137], v[94:97], a[208:211], v[168], v[160] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[134:137], v[102:105], a[212:215], v[168], v[160] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[134:137], v[110:113], a[216:219], v[168], v[161] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[134:137], v[118:121], a[220:223], v[168], v[161] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[138:141], v[98:101], a[208:211], v[168], v[160] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[138:141], v[106:109], a[212:215], v[168], v[160] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[138:141], v[114:117], a[216:219], v[168], v[161] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[138:141], v[122:125], a[220:223], v[168], v[161] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[142:145], v[94:97], a[224:227], v[169], v[160] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[142:145], v[102:105], a[228:231], v[169], v[160] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[142:145], v[110:113], a[232:235], v[169], v[161] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[142:145], v[118:121], a[236:239], v[169], v[161] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[146:149], v[98:101], a[224:227], v[169], v[160] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[146:149], v[106:109], a[228:231], v[169], v[160] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[146:149], v[114:117], a[232:235], v[169], v[161] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[146:149], v[122:125], a[236:239], v[169], v[161] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[150:153], v[94:97], a[240:243], v[169], v[160] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[150:153], v[102:105], a[244:247], v[169], v[160] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[150:153], v[110:113], a[248:251], v[169], v[161] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[150:153], v[118:121], a[252:255], v[169], v[161] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[154:157], v[98:101], a[240:243], v[169], v[160] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[154:157], v[106:109], a[244:247], v[169], v[160] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[154:157], v[114:117], a[248:251], v[169], v[161] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[154:157], v[122:125], a[252:255], v[169], v[161] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_i32 s67, s52, 0xfffffe00
	s_add_i32 s24, s58, 64
	s_add_i32 s63, s58, 0x10000
	s_add_i32 s79, s58, 0x10040
	s_add_i32 s81, s58, 0x20000
	s_add_i32 s84, s58, 0x20040
	s_add_i32 s86, s58, 0x30000
	s_add_i32 s87, s58, 0x30040
	;;#ASMSTART
	s_waitcnt vmcnt(0) lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v[166], v8
v_mov_b32 v[167], v9
v_mov_b32 v[168], v10
v_mov_b32 v[169], v11
v_mov_b32 v[158], v12
v_mov_b32 v[159], v13
v_mov_b32 v[160], v14
v_mov_b32 v[161], v15

	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[8:9], v5, s[0:3], s67 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[10:11], v5, s[4:7], s67 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[12:13], v5, s[8:11], s67 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[14:15], v5, s[16:19], s67 offen
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[62:65], v[30:33], a[0:3], v[166], v[158] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[94:97], v21 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[62:65], v[38:41], a[4:7], v[166], v[158] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[98:101], v22 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[62:65], v[46:49], a[8:11], v[166], v[159] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[102:105], v21 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[62:65], v[54:57], a[12:15], v[166], v[159] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[106:109], v22 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[66:69], v[34:37], a[0:3], v[166], v[158] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[110:113], v21 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[66:69], v[42:45], a[4:7], v[166], v[158] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[114:117], v22 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[66:69], v[50:53], a[8:11], v[166], v[159] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[118:121], v21 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:69], v[58:61], a[12:15], v[166], v[159] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[122:125], v22 offset:6144
buffer_load_dwordx4 v[126:129], v23, s[20:23], s58 offen
buffer_load_dwordx4 v[130:133], v23, s[20:23], s24 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[70:73], v[30:33], a[16:19], v[166], v[158] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[70:73], v[38:41], a[20:23], v[166], v[158] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[70:73], v[46:49], a[24:27], v[166], v[159] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[70:73], v[54:57], a[28:31], v[166], v[159] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[74:77], v[34:37], a[16:19], v[166], v[158] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[74:77], v[42:45], a[20:23], v[166], v[158] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[134:137], v23, s[20:23], s63 offen
buffer_load_dwordx4 v[138:141], v23, s[20:23], s79 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[74:77], v[50:53], a[24:27], v[166], v[159] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[74:77], v[58:61], a[28:31], v[166], v[159] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[78:81], v[30:33], a[32:35], v[167], v[158] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[78:81], v[38:41], a[36:39], v[167], v[158] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[78:81], v[46:49], a[40:43], v[167], v[159] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[78:81], v[54:57], a[44:47], v[167], v[159] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[142:145], v23, s[20:23], s81 offen
buffer_load_dwordx4 v[146:149], v23, s[20:23], s84 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[82:85], v[34:37], a[32:35], v[167], v[158] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[82:85], v[42:45], a[36:39], v[167], v[158] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[82:85], v[50:53], a[40:43], v[167], v[159] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[82:85], v[58:61], a[44:47], v[167], v[159] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[86:89], v[30:33], a[48:51], v[167], v[158] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[86:89], v[38:41], a[52:55], v[167], v[158] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[150:153], v23, s[20:23], s86 offen
buffer_load_dwordx4 v[154:157], v23, s[20:23], s87 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[86:89], v[46:49], a[56:59], v[167], v[159] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[86:89], v[54:57], a[60:63], v[167], v[159] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[90:93], v[34:37], a[48:51], v[167], v[158] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[90:93], v[42:45], a[52:55], v[167], v[158] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[90:93], v[50:53], a[56:59], v[167], v[159] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[90:93], v[58:61], a[60:63], v[167], v[159] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[62:65], v[94:97], a[64:67], v[166], v[160] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[62:65], v[102:105], a[68:71], v[166], v[160] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[62:65], v[110:113], a[72:75], v[166], v[161] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[62:65], v[118:121], a[76:79], v[166], v[161] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[66:69], v[98:101], a[64:67], v[166], v[160] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[66:69], v[106:109], a[68:71], v[166], v[160] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[66:69], v[114:117], a[72:75], v[166], v[161] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[66:69], v[122:125], a[76:79], v[166], v[161] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[70:73], v[94:97], a[80:83], v[166], v[160] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[70:73], v[102:105], a[84:87], v[166], v[160] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[70:73], v[110:113], a[88:91], v[166], v[161] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[70:73], v[118:121], a[92:95], v[166], v[161] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[74:77], v[98:101], a[80:83], v[166], v[160] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[74:77], v[106:109], a[84:87], v[166], v[160] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[74:77], v[114:117], a[88:91], v[166], v[161] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[74:77], v[122:125], a[92:95], v[166], v[161] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[78:81], v[94:97], a[96:99], v[167], v[160] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[78:81], v[102:105], a[100:103], v[167], v[160] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[78:81], v[110:113], a[104:107], v[167], v[161] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[78:81], v[118:121], a[108:111], v[167], v[161] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[82:85], v[98:101], a[96:99], v[167], v[160] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[82:85], v[106:109], a[100:103], v[167], v[160] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[82:85], v[114:117], a[104:107], v[167], v[161] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[82:85], v[122:125], a[108:111], v[167], v[161] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[86:89], v[94:97], a[112:115], v[167], v[160] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[86:89], v[102:105], a[116:119], v[167], v[160] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[86:89], v[110:113], a[120:123], v[167], v[161] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[86:89], v[118:121], a[124:127], v[167], v[161] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[90:93], v[98:101], a[112:115], v[167], v[160] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[90:93], v[106:109], a[116:119], v[167], v[160] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[90:93], v[114:117], a[120:123], v[167], v[161] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[90:93], v[122:125], a[124:127], v[167], v[161] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_mov_b32 m0, s33
	s_add_i32 s71, s51, -2
	buffer_load_dwordx4 v2, s[12:15], s56 offen lds
	s_mov_b32 m0, s43
	s_add_i32 s55, s53, 0x700
	buffer_load_dwordx4 v1, s[12:15], s56 offen lds
	s_mov_b32 m0, s44
	s_min_u32 s71, s71, 29
	buffer_load_dwordx4 v3, s[12:15], s56 offen lds
	s_mov_b32 m0, s45
	s_add_i32 s57, s53, 0x80700
	buffer_load_dwordx4 v4, s[12:15], s56 offen lds
	s_mov_b32 m0, s35
	s_add_i32 s89, s55, 64
	buffer_load_dwordx4 v2, s[12:15], s54 offen lds
	s_mov_b32 m0, s46
	s_add_i32 s72, s55, 0x10000
	buffer_load_dwordx4 v1, s[12:15], s54 offen lds
	s_mov_b32 m0, s47
	s_add_i32 s73, s55, 0x10040
	buffer_load_dwordx4 v3, s[12:15], s54 offen lds
	s_mov_b32 m0, s48
	s_add_i32 s25, s55, 0x20000
	buffer_load_dwordx4 v4, s[12:15], s54 offen lds
	s_add_i32 s61, s55, 0x20040
	s_add_i32 s65, s55, 0x30000
	s_add_i32 s66, s55, 0x30040
	;;#ASMSTART
	s_waitcnt vmcnt(0)
s_barrier
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[126:129], v[30:33], a[128:131], v[168], v[158] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[126:129], v[38:41], a[132:135], v[168], v[158] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[62:65], v23, s[20:23], s55 offen
buffer_load_dwordx4 v[66:69], v23, s[20:23], s89 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[126:129], v[46:49], a[136:139], v[168], v[159] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[126:129], v[54:57], a[140:143], v[168], v[159] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[130:133], v[34:37], a[128:131], v[168], v[158] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[130:133], v[42:45], a[132:135], v[168], v[158] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[70:73], v23, s[20:23], s72 offen
buffer_load_dwordx4 v[74:77], v23, s[20:23], s73 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[130:133], v[50:53], a[136:139], v[168], v[159] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[130:133], v[58:61], a[140:143], v[168], v[159] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[134:137], v[30:33], a[144:147], v[168], v[158] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[134:137], v[38:41], a[148:151], v[168], v[158] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[78:81], v23, s[20:23], s25 offen
buffer_load_dwordx4 v[82:85], v23, s[20:23], s61 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[134:137], v[46:49], a[152:155], v[168], v[159] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[134:137], v[54:57], a[156:159], v[168], v[159] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[138:141], v[34:37], a[144:147], v[168], v[158] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[138:141], v[42:45], a[148:151], v[168], v[158] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[86:89], v23, s[20:23], s65 offen
buffer_load_dwordx4 v[90:93], v23, s[20:23], s66 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[138:141], v[50:53], a[152:155], v[168], v[159] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[138:141], v[58:61], a[156:159], v[168], v[159] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[142:145], v[30:33], a[160:163], v[169], v[158] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[142:145], v[38:41], a[164:167], v[169], v[158] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[142:145], v[46:49], a[168:171], v[169], v[159] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[142:145], v[54:57], a[172:175], v[169], v[159] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[146:149], v[34:37], a[160:163], v[169], v[158] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[146:149], v[42:45], a[164:167], v[169], v[158] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[146:149], v[50:53], a[168:171], v[169], v[159] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[146:149], v[58:61], a[172:175], v[169], v[159] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[150:153], v[30:33], a[176:179], v[169], v[158] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[150:153], v[38:41], a[180:183], v[169], v[158] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[150:153], v[46:49], a[184:187], v[169], v[159] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[150:153], v[54:57], a[188:191], v[169], v[159] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[154:157], v[34:37], a[176:179], v[169], v[158] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[154:157], v[42:45], a[180:183], v[169], v[158] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[154:157], v[50:53], a[184:187], v[169], v[159] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[154:157], v[58:61], a[188:191], v[169], v[159] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[126:129], v[94:97], a[192:195], v[168], v[160] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v7 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[126:129], v[102:105], a[196:199], v[168], v[160] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v16 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[126:129], v[110:113], a[200:203], v[168], v[161] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v7 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[126:129], v[118:121], a[204:207], v[168], v[161] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v16 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[130:133], v[98:101], a[192:195], v[168], v[160] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v7 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[130:133], v[106:109], a[196:199], v[168], v[160] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v16 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[130:133], v[114:117], a[200:203], v[168], v[161] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v7 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[130:133], v[122:125], a[204:207], v[168], v[161] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v16 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[134:137], v[94:97], a[208:211], v[168], v[160] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[134:137], v[102:105], a[212:215], v[168], v[160] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[134:137], v[110:113], a[216:219], v[168], v[161] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[134:137], v[118:121], a[220:223], v[168], v[161] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[138:141], v[98:101], a[208:211], v[168], v[160] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[138:141], v[106:109], a[212:215], v[168], v[160] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[138:141], v[114:117], a[216:219], v[168], v[161] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[138:141], v[122:125], a[220:223], v[168], v[161] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[142:145], v[94:97], a[224:227], v[169], v[160] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[142:145], v[102:105], a[228:231], v[169], v[160] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[142:145], v[110:113], a[232:235], v[169], v[161] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[142:145], v[118:121], a[236:239], v[169], v[161] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[146:149], v[98:101], a[224:227], v[169], v[160] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[146:149], v[106:109], a[228:231], v[169], v[160] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[146:149], v[114:117], a[232:235], v[169], v[161] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[146:149], v[122:125], a[236:239], v[169], v[161] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[150:153], v[94:97], a[240:243], v[169], v[160] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[150:153], v[102:105], a[244:247], v[169], v[160] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[150:153], v[110:113], a[248:251], v[169], v[161] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[150:153], v[118:121], a[252:255], v[169], v[161] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[154:157], v[98:101], a[240:243], v[169], v[160] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[154:157], v[106:109], a[244:247], v[169], v[160] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[154:157], v[114:117], a[248:251], v[169], v[161] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[154:157], v[122:125], a[252:255], v[169], v[161] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_lshl_b32 s71, s71, 7
	s_add_i32 s77, s57, 64
	s_add_i32 s82, s57, 0x10000
	s_add_i32 s88, s57, 0x10040
	s_add_i32 s92, s57, 0x20000
	s_add_i32 s69, s57, 0x20040
	s_add_i32 s74, s57, 0x30000
	s_add_i32 s75, s57, 0x30040
	;;#ASMSTART
	s_waitcnt vmcnt(0) lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v[166], v8
v_mov_b32 v[167], v9
v_mov_b32 v[168], v10
v_mov_b32 v[169], v11
v_mov_b32 v[158], v12
v_mov_b32 v[159], v13
v_mov_b32 v[160], v14
v_mov_b32 v[161], v15

	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[8:9], v5, s[0:3], s52 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[10:11], v5, s[4:7], s52 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[12:13], v5, s[8:11], s52 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[14:15], v5, s[16:19], s52 offen
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[62:65], v[30:33], a[0:3], v[166], v[158] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[94:97], v19 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[62:65], v[38:41], a[4:7], v[166], v[158] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[98:101], v20 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[62:65], v[46:49], a[8:11], v[166], v[159] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[102:105], v19 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[62:65], v[54:57], a[12:15], v[166], v[159] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[106:109], v20 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[66:69], v[34:37], a[0:3], v[166], v[158] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[110:113], v19 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[66:69], v[42:45], a[4:7], v[166], v[158] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[114:117], v20 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[66:69], v[50:53], a[8:11], v[166], v[159] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[118:121], v19 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:69], v[58:61], a[12:15], v[166], v[159] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[122:125], v20 offset:6144
buffer_load_dwordx4 v[126:129], v23, s[20:23], s57 offen
buffer_load_dwordx4 v[130:133], v23, s[20:23], s77 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[70:73], v[30:33], a[16:19], v[166], v[158] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[70:73], v[38:41], a[20:23], v[166], v[158] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[70:73], v[46:49], a[24:27], v[166], v[159] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[70:73], v[54:57], a[28:31], v[166], v[159] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[74:77], v[34:37], a[16:19], v[166], v[158] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[74:77], v[42:45], a[20:23], v[166], v[158] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[134:137], v23, s[20:23], s82 offen
buffer_load_dwordx4 v[138:141], v23, s[20:23], s88 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[74:77], v[50:53], a[24:27], v[166], v[159] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[74:77], v[58:61], a[28:31], v[166], v[159] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[78:81], v[30:33], a[32:35], v[167], v[158] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[78:81], v[38:41], a[36:39], v[167], v[158] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[78:81], v[46:49], a[40:43], v[167], v[159] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[78:81], v[54:57], a[44:47], v[167], v[159] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[142:145], v23, s[20:23], s92 offen
buffer_load_dwordx4 v[146:149], v23, s[20:23], s69 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[82:85], v[34:37], a[32:35], v[167], v[158] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[82:85], v[42:45], a[36:39], v[167], v[158] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[82:85], v[50:53], a[40:43], v[167], v[159] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[82:85], v[58:61], a[44:47], v[167], v[159] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[86:89], v[30:33], a[48:51], v[167], v[158] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[86:89], v[38:41], a[52:55], v[167], v[158] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[150:153], v23, s[20:23], s74 offen
buffer_load_dwordx4 v[154:157], v23, s[20:23], s75 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[86:89], v[46:49], a[56:59], v[167], v[159] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[86:89], v[54:57], a[60:63], v[167], v[159] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[90:93], v[34:37], a[48:51], v[167], v[158] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[90:93], v[42:45], a[52:55], v[167], v[158] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[90:93], v[50:53], a[56:59], v[167], v[159] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[90:93], v[58:61], a[60:63], v[167], v[159] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[62:65], v[94:97], a[64:67], v[166], v[160] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[62:65], v[102:105], a[68:71], v[166], v[160] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[62:65], v[110:113], a[72:75], v[166], v[161] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[62:65], v[118:121], a[76:79], v[166], v[161] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[66:69], v[98:101], a[64:67], v[166], v[160] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[66:69], v[106:109], a[68:71], v[166], v[160] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[66:69], v[114:117], a[72:75], v[166], v[161] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[66:69], v[122:125], a[76:79], v[166], v[161] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[70:73], v[94:97], a[80:83], v[166], v[160] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[70:73], v[102:105], a[84:87], v[166], v[160] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[70:73], v[110:113], a[88:91], v[166], v[161] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[70:73], v[118:121], a[92:95], v[166], v[161] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[74:77], v[98:101], a[80:83], v[166], v[160] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[74:77], v[106:109], a[84:87], v[166], v[160] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[74:77], v[114:117], a[88:91], v[166], v[161] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[74:77], v[122:125], a[92:95], v[166], v[161] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[78:81], v[94:97], a[96:99], v[167], v[160] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[78:81], v[102:105], a[100:103], v[167], v[160] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[78:81], v[110:113], a[104:107], v[167], v[161] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[78:81], v[118:121], a[108:111], v[167], v[161] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[82:85], v[98:101], a[96:99], v[167], v[160] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[82:85], v[106:109], a[100:103], v[167], v[160] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[82:85], v[114:117], a[104:107], v[167], v[161] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[82:85], v[122:125], a[108:111], v[167], v[161] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[86:89], v[94:97], a[112:115], v[167], v[160] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[86:89], v[102:105], a[116:119], v[167], v[160] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[86:89], v[110:113], a[120:123], v[167], v[161] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[86:89], v[118:121], a[124:127], v[167], v[161] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[90:93], v[98:101], a[112:115], v[167], v[160] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[90:93], v[106:109], a[116:119], v[167], v[160] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[90:93], v[114:117], a[120:123], v[167], v[161] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[90:93], v[122:125], a[124:127], v[167], v[161] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_i32 s59, s71, s49
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_mov_b32 m0, s29
	s_add_i32 s71, s71, s50
	buffer_load_dwordx4 v2, s[12:15], s59 offen lds
	s_mov_b32 m0, s37
	s_add_i32 s64, s53, 0x780
	buffer_load_dwordx4 v1, s[12:15], s59 offen lds
	s_mov_b32 m0, s38
	s_add_i32 s70, s51, -1
	buffer_load_dwordx4 v3, s[12:15], s59 offen lds
	s_mov_b32 m0, s39
	s_add_i32 s76, s64, 64
	buffer_load_dwordx4 v4, s[12:15], s59 offen lds
	s_mov_b32 m0, s34
	s_add_i32 s78, s64, 0x10000
	buffer_load_dwordx4 v2, s[12:15], s71 offen lds
	s_mov_b32 m0, s40
	s_add_i32 s83, s64, 0x10040
	buffer_load_dwordx4 v1, s[12:15], s71 offen lds
	s_mov_b32 m0, s41
	s_add_i32 s93, s64, 0x20000
	s_add_i32 s94, s64, 0x20040
	s_add_i32 s95, s64, 0x30000
	s_add_i32 s60, s64, 0x30040
	buffer_load_dwordx4 v3, s[12:15], s71 offen lds
	s_mov_b32 m0, s42
	s_cmpk_lg_i32 s36, 0x800
	buffer_load_dwordx4 v4, s[12:15], s71 offen lds
	s_cselect_b32 s62, s51, 31
	s_min_u32 s68, s70, 29
	s_add_i32 s53, s53, 0x80780
	;;#ASMSTART
	s_waitcnt vmcnt(0)
s_barrier
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[126:129], v[30:33], a[128:131], v[168], v[158] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[126:129], v[38:41], a[132:135], v[168], v[158] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[62:65], v23, s[20:23], s64 offen
buffer_load_dwordx4 v[66:69], v23, s[20:23], s76 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[126:129], v[46:49], a[136:139], v[168], v[159] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[126:129], v[54:57], a[140:143], v[168], v[159] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[130:133], v[34:37], a[128:131], v[168], v[158] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[130:133], v[42:45], a[132:135], v[168], v[158] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[70:73], v23, s[20:23], s78 offen
buffer_load_dwordx4 v[74:77], v23, s[20:23], s83 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[130:133], v[50:53], a[136:139], v[168], v[159] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[130:133], v[58:61], a[140:143], v[168], v[159] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[134:137], v[30:33], a[144:147], v[168], v[158] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[134:137], v[38:41], a[148:151], v[168], v[158] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[78:81], v23, s[20:23], s93 offen
buffer_load_dwordx4 v[82:85], v23, s[20:23], s94 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[134:137], v[46:49], a[152:155], v[168], v[159] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[134:137], v[54:57], a[156:159], v[168], v[159] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[138:141], v[34:37], a[144:147], v[168], v[158] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[138:141], v[42:45], a[148:151], v[168], v[158] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[86:89], v23, s[20:23], s95 offen
buffer_load_dwordx4 v[90:93], v23, s[20:23], s60 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[138:141], v[50:53], a[152:155], v[168], v[159] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[138:141], v[58:61], a[156:159], v[168], v[159] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[142:145], v[30:33], a[160:163], v[169], v[158] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[142:145], v[38:41], a[164:167], v[169], v[158] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[142:145], v[46:49], a[168:171], v[169], v[159] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[142:145], v[54:57], a[172:175], v[169], v[159] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[146:149], v[34:37], a[160:163], v[169], v[158] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[146:149], v[42:45], a[164:167], v[169], v[158] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[146:149], v[50:53], a[168:171], v[169], v[159] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[146:149], v[58:61], a[172:175], v[169], v[159] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[150:153], v[30:33], a[176:179], v[169], v[158] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[150:153], v[38:41], a[180:183], v[169], v[158] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[150:153], v[46:49], a[184:187], v[169], v[159] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[150:153], v[54:57], a[188:191], v[169], v[159] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[154:157], v[34:37], a[176:179], v[169], v[158] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[154:157], v[42:45], a[180:183], v[169], v[158] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[154:157], v[50:53], a[184:187], v[169], v[159] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[154:157], v[58:61], a[188:191], v[169], v[159] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[126:129], v[94:97], a[192:195], v[168], v[160] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v17 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[126:129], v[102:105], a[196:199], v[168], v[160] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v18 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[126:129], v[110:113], a[200:203], v[168], v[161] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v17 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[126:129], v[118:121], a[204:207], v[168], v[161] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v18 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[130:133], v[98:101], a[192:195], v[168], v[160] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v17 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[130:133], v[106:109], a[196:199], v[168], v[160] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v18 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[130:133], v[114:117], a[200:203], v[168], v[161] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v17 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[130:133], v[122:125], a[204:207], v[168], v[161] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v18 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[134:137], v[94:97], a[208:211], v[168], v[160] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[134:137], v[102:105], a[212:215], v[168], v[160] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[134:137], v[110:113], a[216:219], v[168], v[161] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[134:137], v[118:121], a[220:223], v[168], v[161] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[138:141], v[98:101], a[208:211], v[168], v[160] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[138:141], v[106:109], a[212:215], v[168], v[160] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[138:141], v[114:117], a[216:219], v[168], v[161] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[138:141], v[122:125], a[220:223], v[168], v[161] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[142:145], v[94:97], a[224:227], v[169], v[160] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[142:145], v[102:105], a[228:231], v[169], v[160] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[142:145], v[110:113], a[232:235], v[169], v[161] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[142:145], v[118:121], a[236:239], v[169], v[161] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[146:149], v[98:101], a[224:227], v[169], v[160] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[146:149], v[106:109], a[228:231], v[169], v[160] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[146:149], v[114:117], a[232:235], v[169], v[161] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[146:149], v[122:125], a[236:239], v[169], v[161] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[150:153], v[94:97], a[240:243], v[169], v[160] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[150:153], v[102:105], a[244:247], v[169], v[160] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[150:153], v[110:113], a[248:251], v[169], v[161] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[150:153], v[118:121], a[252:255], v[169], v[161] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[154:157], v[98:101], a[240:243], v[169], v[160] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[154:157], v[106:109], a[244:247], v[169], v[160] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[154:157], v[114:117], a[248:251], v[169], v[161] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[154:157], v[122:125], a[252:255], v[169], v[161] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_lshl_b32 s70, s62, 9
	s_lshl_b32 s68, s68, 7
	s_add_i32 s80, s53, 64
	s_add_i32 s85, s53, 0x10000
	s_add_i32 s90, s53, 0x10040
	s_add_i32 s91, s53, 0x20000
	s_add_i32 s96, s53, 0x20040
	s_add_i32 s97, s53, 0x30000
	s_add_i32 s67, s53, 0x30040
	;;#ASMSTART
	s_waitcnt vmcnt(0) lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	v_mov_b32 v[166], v8
v_mov_b32 v[167], v9
v_mov_b32 v[168], v10
v_mov_b32 v[169], v11
v_mov_b32 v[158], v12
v_mov_b32 v[159], v13
v_mov_b32 v[160], v14
v_mov_b32 v[161], v15

	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[8:9], v5, s[0:3], s70 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[10:11], v5, s[4:7], s70 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[12:13], v5, s[8:11], s70 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[14:15], v5, s[16:19], s70 offen
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[62:65], v[30:33], a[0:3], v[166], v[158] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[94:97], v21 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[62:65], v[38:41], a[4:7], v[166], v[158] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[98:101], v22 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[62:65], v[46:49], a[8:11], v[166], v[159] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[102:105], v21 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[62:65], v[54:57], a[12:15], v[166], v[159] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[106:109], v22 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[66:69], v[34:37], a[0:3], v[166], v[158] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[110:113], v21 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[66:69], v[42:45], a[4:7], v[166], v[158] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[114:117], v22 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[66:69], v[50:53], a[8:11], v[166], v[159] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[118:121], v21 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[66:69], v[58:61], a[12:15], v[166], v[159] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[122:125], v22 offset:6144
buffer_load_dwordx4 v[126:129], v23, s[20:23], s53 offen
buffer_load_dwordx4 v[130:133], v23, s[20:23], s80 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[70:73], v[30:33], a[16:19], v[166], v[158] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[70:73], v[38:41], a[20:23], v[166], v[158] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[70:73], v[46:49], a[24:27], v[166], v[159] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[70:73], v[54:57], a[28:31], v[166], v[159] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[74:77], v[34:37], a[16:19], v[166], v[158] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[74:77], v[42:45], a[20:23], v[166], v[158] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[134:137], v23, s[20:23], s85 offen
buffer_load_dwordx4 v[138:141], v23, s[20:23], s90 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[74:77], v[50:53], a[24:27], v[166], v[159] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[74:77], v[58:61], a[28:31], v[166], v[159] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[78:81], v[30:33], a[32:35], v[167], v[158] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[78:81], v[38:41], a[36:39], v[167], v[158] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[78:81], v[46:49], a[40:43], v[167], v[159] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[78:81], v[54:57], a[44:47], v[167], v[159] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[142:145], v23, s[20:23], s91 offen
buffer_load_dwordx4 v[146:149], v23, s[20:23], s96 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[82:85], v[34:37], a[32:35], v[167], v[158] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[82:85], v[42:45], a[36:39], v[167], v[158] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[82:85], v[50:53], a[40:43], v[167], v[159] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[82:85], v[58:61], a[44:47], v[167], v[159] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[86:89], v[30:33], a[48:51], v[167], v[158] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[86:89], v[38:41], a[52:55], v[167], v[158] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[150:153], v23, s[20:23], s97 offen
buffer_load_dwordx4 v[154:157], v23, s[20:23], s67 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[86:89], v[46:49], a[56:59], v[167], v[159] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[86:89], v[54:57], a[60:63], v[167], v[159] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[90:93], v[34:37], a[48:51], v[167], v[158] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[90:93], v[42:45], a[52:55], v[167], v[158] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[90:93], v[50:53], a[56:59], v[167], v[159] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[90:93], v[58:61], a[60:63], v[167], v[159] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[62:65], v[94:97], a[64:67], v[166], v[160] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[62:65], v[102:105], a[68:71], v[166], v[160] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[62:65], v[110:113], a[72:75], v[166], v[161] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[62:65], v[118:121], a[76:79], v[166], v[161] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[66:69], v[98:101], a[64:67], v[166], v[160] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[66:69], v[106:109], a[68:71], v[166], v[160] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[66:69], v[114:117], a[72:75], v[166], v[161] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[66:69], v[122:125], a[76:79], v[166], v[161] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[70:73], v[94:97], a[80:83], v[166], v[160] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[70:73], v[102:105], a[84:87], v[166], v[160] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[70:73], v[110:113], a[88:91], v[166], v[161] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[70:73], v[118:121], a[92:95], v[166], v[161] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[74:77], v[98:101], a[80:83], v[166], v[160] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[74:77], v[106:109], a[84:87], v[166], v[160] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[74:77], v[114:117], a[88:91], v[166], v[161] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[74:77], v[122:125], a[92:95], v[166], v[161] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[78:81], v[94:97], a[96:99], v[167], v[160] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[78:81], v[102:105], a[100:103], v[167], v[160] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[78:81], v[110:113], a[104:107], v[167], v[161] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[78:81], v[118:121], a[108:111], v[167], v[161] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[82:85], v[98:101], a[96:99], v[167], v[160] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[82:85], v[106:109], a[100:103], v[167], v[160] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[82:85], v[114:117], a[104:107], v[167], v[161] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[82:85], v[122:125], a[108:111], v[167], v[161] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[86:89], v[94:97], a[112:115], v[167], v[160] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[86:89], v[102:105], a[116:119], v[167], v[160] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[86:89], v[110:113], a[120:123], v[167], v[161] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[86:89], v[118:121], a[124:127], v[167], v[161] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[90:93], v[98:101], a[112:115], v[167], v[160] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[90:93], v[106:109], a[116:119], v[167], v[160] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[90:93], v[114:117], a[120:123], v[167], v[161] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[90:93], v[122:125], a[124:127], v[167], v[161] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_i32 s24, s68, s49
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_mov_b32 m0, s33
	s_add_i32 s58, s68, s50
	buffer_load_dwordx4 v2, s[12:15], s24 offen lds
	s_mov_b32 m0, s43
	s_addk_i32 s36, 0x800
	buffer_load_dwordx4 v1, s[12:15], s24 offen lds
	s_mov_b32 m0, s44
	s_add_i32 s51, s51, 16
	buffer_load_dwordx4 v3, s[12:15], s24 offen lds
	s_mov_b32 m0, s45
	s_addk_i32 s52, 0x2000
	buffer_load_dwordx4 v4, s[12:15], s24 offen lds
	s_mov_b32 m0, s35
	s_lshl_b32 s24, s62, 7
	buffer_load_dwordx4 v2, s[12:15], s58 offen lds
	s_mov_b32 m0, s46
	s_add_i32 s24, s24, s30
	buffer_load_dwordx4 v1, s[12:15], s58 offen lds
	s_mov_b32 m0, s47
	s_add_i32 s25, s24, 64
	buffer_load_dwordx4 v3, s[12:15], s58 offen lds
	s_mov_b32 m0, s48
	s_add_i32 s53, s24, 0x10000
	buffer_load_dwordx4 v4, s[12:15], s58 offen lds
	s_add_i32 s54, s24, 0x10040
	s_add_i32 s55, s24, 0x20000
	s_add_i32 s56, s24, 0x20040
	s_add_i32 s57, s24, 0x30000
	s_add_i32 s58, s24, 0x30040
	;;#ASMSTART
	s_waitcnt vmcnt(0)
s_barrier
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[126:129], v[30:33], a[128:131], v[168], v[158] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[126:129], v[38:41], a[132:135], v[168], v[158] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[62:65], v23, s[20:23], s24 offen
buffer_load_dwordx4 v[66:69], v23, s[20:23], s25 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[126:129], v[46:49], a[136:139], v[168], v[159] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[126:129], v[54:57], a[140:143], v[168], v[159] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[130:133], v[34:37], a[128:131], v[168], v[158] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[130:133], v[42:45], a[132:135], v[168], v[158] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[70:73], v23, s[20:23], s53 offen
buffer_load_dwordx4 v[74:77], v23, s[20:23], s54 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[130:133], v[50:53], a[136:139], v[168], v[159] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[130:133], v[58:61], a[140:143], v[168], v[159] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[134:137], v[30:33], a[144:147], v[168], v[158] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[134:137], v[38:41], a[148:151], v[168], v[158] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[78:81], v23, s[20:23], s55 offen
buffer_load_dwordx4 v[82:85], v23, s[20:23], s56 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[134:137], v[46:49], a[152:155], v[168], v[159] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[134:137], v[54:57], a[156:159], v[168], v[159] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[138:141], v[34:37], a[144:147], v[168], v[158] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[138:141], v[42:45], a[148:151], v[168], v[158] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
buffer_load_dwordx4 v[86:89], v23, s[20:23], s57 offen
buffer_load_dwordx4 v[90:93], v23, s[20:23], s58 offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[138:141], v[50:53], a[152:155], v[168], v[159] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[138:141], v[58:61], a[156:159], v[168], v[159] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[142:145], v[30:33], a[160:163], v[169], v[158] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[142:145], v[38:41], a[164:167], v[169], v[158] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[142:145], v[46:49], a[168:171], v[169], v[159] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[142:145], v[54:57], a[172:175], v[169], v[159] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[146:149], v[34:37], a[160:163], v[169], v[158] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[146:149], v[42:45], a[164:167], v[169], v[158] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[146:149], v[50:53], a[168:171], v[169], v[159] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[146:149], v[58:61], a[172:175], v[169], v[159] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[150:153], v[30:33], a[176:179], v[169], v[158] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[150:153], v[38:41], a[180:183], v[169], v[158] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[150:153], v[46:49], a[184:187], v[169], v[159] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[150:153], v[54:57], a[188:191], v[169], v[159] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[154:157], v[34:37], a[176:179], v[169], v[158] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[154:157], v[42:45], a[180:183], v[169], v[158] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[154:157], v[50:53], a[184:187], v[169], v[159] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[154:157], v[58:61], a[188:191], v[169], v[159] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[126:129], v[94:97], a[192:195], v[168], v[160] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v7 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[126:129], v[102:105], a[196:199], v[168], v[160] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v16 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[126:129], v[110:113], a[200:203], v[168], v[161] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v7 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[126:129], v[118:121], a[204:207], v[168], v[161] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v16 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[130:133], v[98:101], a[192:195], v[168], v[160] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v7 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[130:133], v[106:109], a[196:199], v[168], v[160] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v16 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[130:133], v[114:117], a[200:203], v[168], v[161] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v7 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[130:133], v[122:125], a[204:207], v[168], v[161] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v16 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[134:137], v[94:97], a[208:211], v[168], v[160] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[134:137], v[102:105], a[212:215], v[168], v[160] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[134:137], v[110:113], a[216:219], v[168], v[161] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[134:137], v[118:121], a[220:223], v[168], v[161] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[138:141], v[98:101], a[208:211], v[168], v[160] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[138:141], v[106:109], a[212:215], v[168], v[160] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[138:141], v[114:117], a[216:219], v[168], v[161] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[138:141], v[122:125], a[220:223], v[168], v[161] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[142:145], v[94:97], a[224:227], v[169], v[160] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[142:145], v[102:105], a[228:231], v[169], v[160] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[142:145], v[110:113], a[232:235], v[169], v[161] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[142:145], v[118:121], a[236:239], v[169], v[161] op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[146:149], v[98:101], a[224:227], v[169], v[160] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[146:149], v[106:109], a[228:231], v[169], v[160] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[146:149], v[114:117], a[232:235], v[169], v[161] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[146:149], v[122:125], a[236:239], v[169], v[161] op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[150:153], v[94:97], a[240:243], v[169], v[160] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[150:153], v[102:105], a[244:247], v[169], v[160] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[150:153], v[110:113], a[248:251], v[169], v[161] op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[150:153], v[118:121], a[252:255], v[169], v[161] op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[154:157], v[98:101], a[240:243], v[169], v[160] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[154:157], v[106:109], a[244:247], v[169], v[160] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[154:157], v[114:117], a[248:251], v[169], v[161] op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[154:157], v[122:125], a[252:255], v[169], v[161] op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0) lgkmcnt(0)
	;;#ASMEND
	s_cmpk_eq_i32 s36, 0x1000
	s_cbranch_scc0 .LBB0_7
; %bb.8:
	v_readlane_b32 s4, v28, 0
	v_mad_i64_i32 v[2:3], s[0:1], s26, v24, 0
	v_readlane_b32 s5, v28, 1
	v_ashrrev_i32_e32 v7, 31, v6
	v_lshrrev_b32_e32 v1, 2, v0
	v_lshl_add_u64 v[2:3], v[2:3], 1, s[4:5]
	v_lshlrev_b64 v[34:35], 1, v[6:7]
	v_and_b32_e32 v19, 12, v1
	v_and_b32_e32 v18, 15, v0
	v_lshl_add_u64 v[36:37], v[2:3], 0, v[34:35]
	v_mad_u64_u32 v[2:3], s[0:1], v19, s26, v[18:19]
	v_add_u32_e32 v4, s26, v2
	;;#ASMSTART
	v_mov_b32 v0, v[29]
	;;#ASMEND
	v_mov_b32_e32 v133, v24
	;;#ASMSTART
	v_accvgpr_read_b32 v6, a[0]
v_accvgpr_read_b32 v1, a[1]
v_mul_f32 v6, v0, v6
v_mul_f32 v1, v0, v1
v_cvt_pk_bf16_f32 v6, v6, v1
v_accvgpr_read_b32 v8, a[2]
v_accvgpr_read_b32 v1, a[3]
v_mul_f32 v8, v0, v8
v_mul_f32 v1, v0, v1
v_cvt_pk_bf16_f32 v8, v8, v1
v_accvgpr_read_b32 v9, a[4]
v_accvgpr_read_b32 v1, a[5]
v_mul_f32 v9, v0, v9
v_mul_f32 v1, v0, v1
v_cvt_pk_bf16_f32 v9, v9, v1
v_accvgpr_read_b32 v10, a[6]
v_accvgpr_read_b32 v1, a[7]
v_mul_f32 v10, v0, v10
v_mul_f32 v1, v0, v1
v_cvt_pk_bf16_f32 v10, v10, v1
v_accvgpr_read_b32 v11, a[8]
v_accvgpr_read_b32 v1, a[9]
v_mul_f32 v11, v0, v11
v_mul_f32 v1, v0, v1
v_cvt_pk_bf16_f32 v11, v11, v1
v_accvgpr_read_b32 v14, a[10]
v_accvgpr_read_b32 v1, a[11]
v_mul_f32 v14, v0, v14
v_mul_f32 v1, v0, v1
v_cvt_pk_bf16_f32 v14, v14, v1
v_accvgpr_read_b32 v15, a[12]
v_accvgpr_read_b32 v1, a[13]
v_mul_f32 v15, v0, v15
v_mul_f32 v1, v0, v1
v_cvt_pk_bf16_f32 v15, v15, v1
v_accvgpr_read_b32 v16, a[14]
v_accvgpr_read_b32 v1, a[15]
v_mul_f32 v16, v0, v16
v_mul_f32 v1, v0, v1
v_cvt_pk_bf16_f32 v16, v16, v1
v_accvgpr_read_b32 v17, a[16]
v_accvgpr_read_b32 v1, a[17]
v_mul_f32 v17, v0, v17
v_mul_f32 v1, v0, v1
v_cvt_pk_bf16_f32 v17, v17, v1
v_accvgpr_read_b32 v22, a[18]
v_accvgpr_read_b32 v1, a[19]
v_mul_f32 v22, v0, v22
v_mul_f32 v1, v0, v1
v_cvt_pk_bf16_f32 v22, v22, v1
v_accvgpr_read_b32 v23, a[20]
v_accvgpr_read_b32 v1, a[21]
v_mul_f32 v23, v0, v23
v_mul_f32 v1, v0, v1
v_cvt_pk_bf16_f32 v23, v23, v1
v_accvgpr_read_b32 v24, a[22]
v_accvgpr_read_b32 v1, a[23]
v_mul_f32 v24, v0, v24
v_mul_f32 v1, v0, v1
v_cvt_pk_bf16_f32 v24, v24, v1
v_accvgpr_read_b32 v25, a[24]
v_accvgpr_read_b32 v1, a[25]
v_mul_f32 v25, v0, v25
v_mul_f32 v1, v0, v1
v_cvt_pk_bf16_f32 v25, v25, v1
v_accvgpr_read_b32 v26, a[26]
v_accvgpr_read_b32 v1, a[27]
v_mul_f32 v26, v0, v26
v_mul_f32 v1, v0, v1
v_cvt_pk_bf16_f32 v26, v26, v1
v_accvgpr_read_b32 v27, a[28]
v_accvgpr_read_b32 v1, a[29]
v_mul_f32 v27, v0, v27
v_mul_f32 v1, v0, v1
v_cvt_pk_bf16_f32 v27, v27, v1
v_accvgpr_read_b32 v29, a[30]
v_accvgpr_read_b32 v1, a[31]
v_mul_f32 v29, v0, v29
v_mul_f32 v1, v0, v1
v_cvt_pk_bf16_f32 v29, v29, v1
v_accvgpr_read_b32 v30, a[32]
v_accvgpr_read_b32 v1, a[33]
v_mul_f32 v30, v0, v30
v_mul_f32 v1, v0, v1
v_cvt_pk_bf16_f32 v30, v30, v1
v_accvgpr_read_b32 v31, a[34]
v_accvgpr_read_b32 v1, a[35]
v_mul_f32 v31, v0, v31
v_mul_f32 v1, v0, v1
v_cvt_pk_bf16_f32 v31, v31, v1
v_accvgpr_read_b32 v32, a[36]
v_accvgpr_read_b32 v1, a[37]
v_mul_f32 v32, v0, v32
v_mul_f32 v1, v0, v1
v_cvt_pk_bf16_f32 v32, v32, v1
v_accvgpr_read_b32 v33, a[38]
v_accvgpr_read_b32 v1, a[39]
v_mul_f32 v33, v0, v33
v_mul_f32 v1, v0, v1
v_cvt_pk_bf16_f32 v33, v33, v1
v_accvgpr_read_b32 v62, a[40]
v_accvgpr_read_b32 v1, a[41]
v_mul_f32 v62, v0, v62
v_mul_f32 v1, v0, v1
v_cvt_pk_bf16_f32 v62, v62, v1
v_accvgpr_read_b32 v63, a[42]
v_accvgpr_read_b32 v1, a[43]
v_mul_f32 v63, v0, v63
v_mul_f32 v1, v0, v1
v_cvt_pk_bf16_f32 v63, v63, v1
v_accvgpr_read_b32 v64, a[44]
v_accvgpr_read_b32 v1, a[45]
v_mul_f32 v64, v0, v64
v_mul_f32 v1, v0, v1
v_cvt_pk_bf16_f32 v64, v64, v1
v_accvgpr_read_b32 v65, a[46]
v_accvgpr_read_b32 v1, a[47]
v_mul_f32 v65, v0, v65
v_mul_f32 v1, v0, v1
v_cvt_pk_bf16_f32 v65, v65, v1
v_accvgpr_read_b32 v66, a[48]
v_accvgpr_read_b32 v1, a[49]
v_mul_f32 v66, v0, v66
v_mul_f32 v1, v0, v1
v_cvt_pk_bf16_f32 v66, v66, v1
v_accvgpr_read_b32 v70, a[50]
v_accvgpr_read_b32 v1, a[51]
v_mul_f32 v70, v0, v70
v_mul_f32 v1, v0, v1
v_cvt_pk_bf16_f32 v70, v70, v1
v_accvgpr_read_b32 v71, a[52]
v_accvgpr_read_b32 v1, a[53]
v_mul_f32 v71, v0, v71
v_mul_f32 v1, v0, v1
v_cvt_pk_bf16_f32 v71, v71, v1
v_accvgpr_read_b32 v72, a[54]
v_accvgpr_read_b32 v1, a[55]
v_mul_f32 v72, v0, v72
v_mul_f32 v1, v0, v1
v_cvt_pk_bf16_f32 v72, v72, v1
v_accvgpr_read_b32 v73, a[56]
v_accvgpr_read_b32 v1, a[57]
v_mul_f32 v73, v0, v73
v_mul_f32 v1, v0, v1
v_cvt_pk_bf16_f32 v73, v73, v1
v_accvgpr_read_b32 v74, a[58]
v_accvgpr_read_b32 v1, a[59]
v_mul_f32 v74, v0, v74
v_mul_f32 v1, v0, v1
v_cvt_pk_bf16_f32 v74, v74, v1
v_accvgpr_read_b32 v75, a[60]
v_accvgpr_read_b32 v1, a[61]
v_mul_f32 v75, v0, v75
v_mul_f32 v1, v0, v1
v_cvt_pk_bf16_f32 v75, v75, v1
v_accvgpr_read_b32 v76, a[62]
v_accvgpr_read_b32 v1, a[63]
v_mul_f32 v76, v0, v76
v_mul_f32 v1, v0, v1
v_cvt_pk_bf16_f32 v76, v76, v1

	;;#ASMEND
	v_ashrrev_i32_e32 v3, 31, v2
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshlrev_b64 v[0:1], 1, v[2:3]
	v_lshlrev_b64 v[2:3], 1, v[4:5]
	v_add_u32_e32 v4, s26, v4
	v_lshl_add_u64 v[38:39], v[36:37], 0, v[0:1]
	v_lshl_add_u64 v[40:41], v[36:37], 0, v[2:3]
	v_ashrrev_i32_e32 v5, 31, v4
	global_store_short v[38:39], v6, off
	global_store_short_d16_hi v[40:41], v6, off
	v_lshlrev_b64 v[6:7], 1, v[4:5]
	v_add_u32_e32 v4, s26, v4
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshlrev_b64 v[12:13], 1, v[4:5]
	v_lshl_add_u64 v[42:43], v[36:37], 0, v[6:7]
	v_lshl_add_u64 v[44:45], v[36:37], 0, v[12:13]
	v_or_b32_e32 v4, 16, v19
	global_store_short v[42:43], v8, off
	global_store_short_d16_hi v[44:45], v8, off
	global_store_short v[38:39], v9, off offset:32
	global_store_short_d16_hi v[40:41], v9, off offset:32
	global_store_short v[42:43], v10, off offset:32
	global_store_short_d16_hi v[44:45], v10, off offset:32
	global_store_short v[38:39], v11, off offset:64
	global_store_short_d16_hi v[40:41], v11, off offset:64
	global_store_short v[42:43], v14, off offset:64
	global_store_short_d16_hi v[44:45], v14, off offset:64
	global_store_short v[38:39], v15, off offset:96
	global_store_short_d16_hi v[40:41], v15, off offset:96
	global_store_short v[42:43], v16, off offset:96
	global_store_short_d16_hi v[44:45], v16, off offset:96
	v_mad_u64_u32 v[8:9], s[0:1], v4, s26, v[18:19]
	v_add_u32_e32 v10, s26, v8
	v_ashrrev_i32_e32 v9, 31, v8
	v_ashrrev_i32_e32 v11, 31, v10
	v_lshlrev_b64 v[4:5], 1, v[8:9]
	v_lshlrev_b64 v[8:9], 1, v[10:11]
	v_add_u32_e32 v10, s26, v10
	v_ashrrev_i32_e32 v11, 31, v10
	v_lshlrev_b64 v[14:15], 1, v[10:11]
	v_add_u32_e32 v10, s26, v10
	v_ashrrev_i32_e32 v11, 31, v10
	v_lshl_add_u64 v[46:47], v[36:37], 0, v[4:5]
	v_lshl_add_u64 v[48:49], v[36:37], 0, v[8:9]
	v_lshlrev_b64 v[20:21], 1, v[10:11]
	v_or_b32_e32 v10, 32, v19
	global_store_short v[46:47], v17, off
	global_store_short_d16_hi v[48:49], v17, off
	v_lshl_add_u64 v[50:51], v[36:37], 0, v[14:15]
	v_lshl_add_u64 v[52:53], v[36:37], 0, v[20:21]
	v_mad_u64_u32 v[16:17], s[0:1], v10, s26, v[18:19]
	global_store_short v[50:51], v22, off
	global_store_short_d16_hi v[52:53], v22, off
	global_store_short v[46:47], v23, off offset:32
	global_store_short_d16_hi v[48:49], v23, off offset:32
	global_store_short v[50:51], v24, off offset:32
	global_store_short_d16_hi v[52:53], v24, off offset:32
	global_store_short v[46:47], v25, off offset:64
	global_store_short_d16_hi v[48:49], v25, off offset:64
	global_store_short v[50:51], v26, off offset:64
	global_store_short_d16_hi v[52:53], v26, off offset:64
	global_store_short v[46:47], v27, off offset:96
	global_store_short_d16_hi v[48:49], v27, off offset:96
	global_store_short v[50:51], v29, off offset:96
	global_store_short_d16_hi v[52:53], v29, off offset:96
	v_add_u32_e32 v22, s26, v16
	v_add_u32_e32 v24, s26, v22
	v_ashrrev_i32_e32 v17, 31, v16
	v_ashrrev_i32_e32 v23, 31, v22
	v_ashrrev_i32_e32 v25, 31, v24
	v_lshlrev_b64 v[10:11], 1, v[16:17]
	v_lshlrev_b64 v[16:17], 1, v[22:23]
	v_lshlrev_b64 v[22:23], 1, v[24:25]
	v_add_u32_e32 v24, s26, v24
	v_ashrrev_i32_e32 v25, 31, v24
	v_or_b32_e32 v19, 48, v19
	v_lshl_add_u64 v[54:55], v[36:37], 0, v[10:11]
	v_lshl_add_u64 v[56:57], v[36:37], 0, v[16:17]
	v_lshlrev_b64 v[26:27], 1, v[24:25]
	v_mad_u64_u32 v[24:25], s[0:1], v19, s26, v[18:19]
	global_store_short v[54:55], v30, off
	global_store_short_d16_hi v[56:57], v30, off
	v_lshl_add_u64 v[58:59], v[36:37], 0, v[22:23]
	v_lshl_add_u64 v[60:61], v[36:37], 0, v[26:27]
	v_add_u32_e32 v30, s26, v24
	global_store_short v[58:59], v31, off
	global_store_short_d16_hi v[60:61], v31, off
	global_store_short v[54:55], v32, off offset:32
	global_store_short_d16_hi v[56:57], v32, off offset:32
	global_store_short v[58:59], v33, off offset:32
	global_store_short_d16_hi v[60:61], v33, off offset:32
	global_store_short v[54:55], v62, off offset:64
	global_store_short_d16_hi v[56:57], v62, off offset:64
	global_store_short v[58:59], v63, off offset:64
	global_store_short_d16_hi v[60:61], v63, off offset:64
	global_store_short v[54:55], v64, off offset:96
	global_store_short_d16_hi v[56:57], v64, off offset:96
	global_store_short v[58:59], v65, off offset:96
	global_store_short_d16_hi v[60:61], v65, off offset:96
	v_add_u32_e32 v32, s26, v30
	v_ashrrev_i32_e32 v25, 31, v24
	v_ashrrev_i32_e32 v31, 31, v30
	v_ashrrev_i32_e32 v33, 31, v32
	v_lshlrev_b64 v[18:19], 1, v[24:25]
	v_lshlrev_b64 v[24:25], 1, v[30:31]
	v_lshlrev_b64 v[30:31], 1, v[32:33]
	v_add_u32_e32 v32, s26, v32
	v_ashrrev_i32_e32 v33, 31, v32
	v_lshl_add_u64 v[62:63], v[36:37], 0, v[18:19]
	v_lshl_add_u64 v[64:65], v[36:37], 0, v[24:25]
	v_lshlrev_b64 v[32:33], 1, v[32:33]
	global_store_short v[62:63], v66, off
	global_store_short_d16_hi v[64:65], v66, off
	v_lshl_add_u64 v[66:67], v[36:37], 0, v[30:31]
	v_lshl_add_u64 v[68:69], v[36:37], 0, v[32:33]
	global_store_short v[66:67], v70, off
	global_store_short_d16_hi v[68:69], v70, off
	global_store_short v[62:63], v71, off offset:32
	global_store_short_d16_hi v[64:65], v71, off offset:32
	global_store_short v[66:67], v72, off offset:32
	global_store_short_d16_hi v[68:69], v72, off offset:32
	global_store_short v[62:63], v73, off offset:64
	global_store_short_d16_hi v[64:65], v73, off offset:64
	global_store_short v[66:67], v74, off offset:64
	global_store_short_d16_hi v[68:69], v74, off offset:64
	global_store_short v[62:63], v75, off offset:96
	global_store_short_d16_hi v[64:65], v75, off offset:96
	global_store_short v[66:67], v76, off offset:96
	global_store_short_d16_hi v[68:69], v76, off offset:96
	s_mov_b64 s[0:1], 0x100
	;;#ASMSTART
	v_accvgpr_read_b32 v29, a[64]
v_accvgpr_read_b32 v70, a[0x41]
v_accvgpr_read_b32 v71, a[0x42]
v_accvgpr_read_b32 v72, a[0x43]
v_accvgpr_read_b32 v73, a[0x44]
v_accvgpr_read_b32 v74, a[0x45]
v_accvgpr_read_b32 v75, a[0x46]
v_accvgpr_read_b32 v76, a[0x47]
v_accvgpr_read_b32 v77, a[0x48]
v_accvgpr_read_b32 v78, a[0x49]
v_accvgpr_read_b32 v79, a[0x4a]
v_accvgpr_read_b32 v80, a[0x4b]
v_accvgpr_read_b32 v81, a[0x4c]
v_accvgpr_read_b32 v82, a[0x4d]
v_accvgpr_read_b32 v83, a[0x4e]
v_accvgpr_read_b32 v84, a[0x4f]
v_accvgpr_read_b32 v85, a[0x50]
v_accvgpr_read_b32 v86, a[0x51]
v_accvgpr_read_b32 v87, a[0x52]
v_accvgpr_read_b32 v88, a[0x53]
v_accvgpr_read_b32 v89, a[0x54]
v_accvgpr_read_b32 v90, a[0x55]
v_accvgpr_read_b32 v91, a[0x56]
v_accvgpr_read_b32 v92, a[0x57]
v_accvgpr_read_b32 v93, a[0x58]
v_accvgpr_read_b32 v94, a[0x59]
v_accvgpr_read_b32 v95, a[0x5a]
v_accvgpr_read_b32 v96, a[0x5b]
v_accvgpr_read_b32 v97, a[0x5c]
v_accvgpr_read_b32 v98, a[0x5d]
v_accvgpr_read_b32 v99, a[0x5e]
v_accvgpr_read_b32 v100, a[0x5f]
v_accvgpr_read_b32 v101, a[0x60]
v_accvgpr_read_b32 v102, a[0x61]
v_accvgpr_read_b32 v103, a[0x62]
v_accvgpr_read_b32 v104, a[0x63]
v_accvgpr_read_b32 v105, a[0x64]
v_accvgpr_read_b32 v106, a[0x65]
v_accvgpr_read_b32 v107, a[0x66]
v_accvgpr_read_b32 v108, a[0x67]
v_accvgpr_read_b32 v109, a[0x68]
v_accvgpr_read_b32 v110, a[0x69]
v_accvgpr_read_b32 v111, a[0x6a]
v_accvgpr_read_b32 v112, a[0x6b]
v_accvgpr_read_b32 v113, a[0x6c]
v_accvgpr_read_b32 v114, a[0x6d]
v_accvgpr_read_b32 v115, a[0x6e]
v_accvgpr_read_b32 v116, a[0x6f]
v_accvgpr_read_b32 v117, a[0x70]
v_accvgpr_read_b32 v118, a[0x71]
v_accvgpr_read_b32 v119, a[0x72]
v_accvgpr_read_b32 v120, a[0x73]
v_accvgpr_read_b32 v121, a[0x74]
v_accvgpr_read_b32 v122, a[0x75]
v_accvgpr_read_b32 v123, a[0x76]
v_accvgpr_read_b32 v124, a[0x77]
v_accvgpr_read_b32 v125, a[0x78]
v_accvgpr_read_b32 v126, a[0x79]
v_accvgpr_read_b32 v127, a[0x7a]
v_accvgpr_read_b32 v128, a[0x7b]
v_accvgpr_read_b32 v129, a[0x7c]
v_accvgpr_read_b32 v130, a[0x7d]
v_accvgpr_read_b32 v131, a[0x7e]
v_accvgpr_read_b32 v132, a[0x7f]

	;;#ASMEND
	v_lshl_add_u64 v[36:37], v[36:37], 0, s[0:1]
	v_mul_f32_e32 v29, s27, v29
	v_mul_f32_e32 v70, s27, v70
	v_mul_f32_e32 v71, s27, v71
	v_mul_f32_e32 v72, s27, v72
	global_store_short_d16_hi v[38:39], v29, off offset:256
	global_store_short_d16_hi v[40:41], v70, off offset:256
	global_store_short_d16_hi v[42:43], v71, off offset:256
	global_store_short_d16_hi v[44:45], v72, off offset:256
	v_mul_f32_e32 v29, s27, v73
	v_mul_f32_e32 v42, s27, v74
	v_lshl_add_u64 v[38:39], v[36:37], 0, v[0:1]
	v_lshl_add_u64 v[40:41], v[36:37], 0, v[2:3]
	v_mul_f32_e32 v44, s27, v75
	global_store_short_d16_hi v[38:39], v29, off offset:32
	global_store_short_d16_hi v[40:41], v42, off offset:32
	v_lshl_add_u64 v[42:43], v[36:37], 0, v[6:7]
	v_mul_f32_e32 v70, s27, v76
	global_store_short_d16_hi v[42:43], v44, off offset:32
	v_lshl_add_u64 v[44:45], v[36:37], 0, v[12:13]
	v_mul_f32_e32 v29, s27, v77
	global_store_short_d16_hi v[44:45], v70, off offset:32
	v_mul_f32_e32 v70, s27, v78
	v_mul_f32_e32 v71, s27, v79
	v_mul_f32_e32 v72, s27, v80
	global_store_short_d16_hi v[38:39], v29, off offset:64
	global_store_short_d16_hi v[40:41], v70, off offset:64
	global_store_short_d16_hi v[42:43], v71, off offset:64
	global_store_short_d16_hi v[44:45], v72, off offset:64
	v_mul_f32_e32 v29, s27, v81
	v_mul_f32_e32 v70, s27, v82
	v_mul_f32_e32 v71, s27, v83
	v_mul_f32_e32 v72, s27, v84
	global_store_short_d16_hi v[38:39], v29, off offset:96
	global_store_short_d16_hi v[40:41], v70, off offset:96
	global_store_short_d16_hi v[42:43], v71, off offset:96
	global_store_short_d16_hi v[44:45], v72, off offset:96
	v_mul_f32_e32 v29, s27, v85
	v_mul_f32_e32 v38, s27, v86
	v_mul_f32_e32 v39, s27, v87
	v_mul_f32_e32 v40, s27, v88
	global_store_short_d16_hi v[46:47], v29, off offset:256
	global_store_short_d16_hi v[48:49], v38, off offset:256
	global_store_short_d16_hi v[50:51], v39, off offset:256
	global_store_short_d16_hi v[52:53], v40, off offset:256
	v_mul_f32_e32 v29, s27, v89
	v_mul_f32_e32 v42, s27, v90
	v_lshl_add_u64 v[38:39], v[36:37], 0, v[4:5]
	v_lshl_add_u64 v[40:41], v[36:37], 0, v[8:9]
	v_mul_f32_e32 v44, s27, v91
	global_store_short_d16_hi v[38:39], v29, off offset:32
	global_store_short_d16_hi v[40:41], v42, off offset:32
	v_lshl_add_u64 v[42:43], v[36:37], 0, v[14:15]
	v_mul_f32_e32 v46, s27, v92
	global_store_short_d16_hi v[42:43], v44, off offset:32
	v_lshl_add_u64 v[44:45], v[36:37], 0, v[20:21]
	v_mul_f32_e32 v29, s27, v93
	global_store_short_d16_hi v[44:45], v46, off offset:32
	v_mul_f32_e32 v46, s27, v94
	v_mul_f32_e32 v47, s27, v95
	v_mul_f32_e32 v48, s27, v96
	global_store_short_d16_hi v[38:39], v29, off offset:64
	global_store_short_d16_hi v[40:41], v46, off offset:64
	global_store_short_d16_hi v[42:43], v47, off offset:64
	global_store_short_d16_hi v[44:45], v48, off offset:64
	v_mul_f32_e32 v29, s27, v97
	v_mul_f32_e32 v46, s27, v98
	v_mul_f32_e32 v47, s27, v99
	v_mul_f32_e32 v48, s27, v100
	global_store_short_d16_hi v[38:39], v29, off offset:96
	global_store_short_d16_hi v[40:41], v46, off offset:96
	global_store_short_d16_hi v[42:43], v47, off offset:96
	global_store_short_d16_hi v[44:45], v48, off offset:96
	v_mul_f32_e32 v29, s27, v101
	v_mul_f32_e32 v38, s27, v102
	v_mul_f32_e32 v39, s27, v103
	v_mul_f32_e32 v40, s27, v104
	global_store_short_d16_hi v[54:55], v29, off offset:256
	global_store_short_d16_hi v[56:57], v38, off offset:256
	global_store_short_d16_hi v[58:59], v39, off offset:256
	global_store_short_d16_hi v[60:61], v40, off offset:256
	v_mul_f32_e32 v29, s27, v105
	v_mul_f32_e32 v42, s27, v106
	v_lshl_add_u64 v[38:39], v[36:37], 0, v[10:11]
	v_lshl_add_u64 v[40:41], v[36:37], 0, v[16:17]
	v_mul_f32_e32 v44, s27, v107
	global_store_short_d16_hi v[38:39], v29, off offset:32
	global_store_short_d16_hi v[40:41], v42, off offset:32
	v_lshl_add_u64 v[42:43], v[36:37], 0, v[22:23]
	v_mul_f32_e32 v46, s27, v108
	global_store_short_d16_hi v[42:43], v44, off offset:32
	v_lshl_add_u64 v[44:45], v[36:37], 0, v[26:27]
	v_mul_f32_e32 v29, s27, v109
	global_store_short_d16_hi v[44:45], v46, off offset:32
	v_mul_f32_e32 v46, s27, v110
	v_mul_f32_e32 v47, s27, v111
	v_mul_f32_e32 v48, s27, v112
	global_store_short_d16_hi v[38:39], v29, off offset:64
	global_store_short_d16_hi v[40:41], v46, off offset:64
	global_store_short_d16_hi v[42:43], v47, off offset:64
	global_store_short_d16_hi v[44:45], v48, off offset:64
	v_mul_f32_e32 v29, s27, v113
	v_mul_f32_e32 v46, s27, v114
	v_mul_f32_e32 v47, s27, v115
	v_mul_f32_e32 v48, s27, v116
	global_store_short_d16_hi v[38:39], v29, off offset:96
	global_store_short_d16_hi v[40:41], v46, off offset:96
	global_store_short_d16_hi v[42:43], v47, off offset:96
	global_store_short_d16_hi v[44:45], v48, off offset:96
	v_mul_f32_e32 v29, s27, v117
	v_mul_f32_e32 v38, s27, v118
	v_mul_f32_e32 v39, s27, v119
	v_mul_f32_e32 v40, s27, v120
	global_store_short_d16_hi v[62:63], v29, off offset:256
	global_store_short_d16_hi v[64:65], v38, off offset:256
	global_store_short_d16_hi v[66:67], v39, off offset:256
	global_store_short_d16_hi v[68:69], v40, off offset:256
	v_mul_f32_e32 v29, s27, v121
	v_mul_f32_e32 v42, s27, v122
	v_lshl_add_u64 v[38:39], v[36:37], 0, v[18:19]
	v_lshl_add_u64 v[40:41], v[36:37], 0, v[24:25]
	v_mul_f32_e32 v44, s27, v123
	v_mul_f32_e32 v45, s27, v124
	global_store_short_d16_hi v[38:39], v29, off offset:32
	global_store_short_d16_hi v[40:41], v42, off offset:32
	v_lshl_add_u64 v[42:43], v[36:37], 0, v[30:31]
	v_lshl_add_u64 v[36:37], v[36:37], 0, v[32:33]
	v_mul_f32_e32 v29, s27, v125
	global_store_short_d16_hi v[42:43], v44, off offset:32
	global_store_short_d16_hi v[36:37], v45, off offset:32
	v_mul_f32_e32 v44, s27, v126
	v_mul_f32_e32 v45, s27, v127
	v_mul_f32_e32 v46, s27, v128
	global_store_short_d16_hi v[38:39], v29, off offset:64
	global_store_short_d16_hi v[40:41], v44, off offset:64
	global_store_short_d16_hi v[42:43], v45, off offset:64
	global_store_short_d16_hi v[36:37], v46, off offset:64
	v_mul_f32_e32 v29, s27, v129
	v_mul_f32_e32 v44, s27, v130
	v_mul_f32_e32 v45, s27, v131
	v_mul_f32_e32 v46, s27, v132
	global_store_short_d16_hi v[38:39], v29, off offset:96
	global_store_short_d16_hi v[40:41], v44, off offset:96
	global_store_short_d16_hi v[42:43], v45, off offset:96
	global_store_short_d16_hi v[36:37], v46, off offset:96
	v_or_b32_e32 v29, 0x80, v133
	v_mad_i64_i32 v[36:37], s[2:3], s26, v29, 0
	v_lshl_add_u64 v[36:37], v[36:37], 1, s[4:5]
	v_lshl_add_u64 v[34:35], v[36:37], 0, v[34:35]
	;;#ASMSTART
	v_accvgpr_read_b32 v29, a[0x80]
v_accvgpr_read_b32 v36, a[0x81]
v_accvgpr_read_b32 v37, a[0x82]
v_accvgpr_read_b32 v38, a[0x83]
v_accvgpr_read_b32 v44, a[0x84]
v_accvgpr_read_b32 v45, a[0x85]
v_accvgpr_read_b32 v46, a[0x86]
v_accvgpr_read_b32 v47, a[0x87]
v_accvgpr_read_b32 v48, a[0x88]
v_accvgpr_read_b32 v49, a[0x89]
v_accvgpr_read_b32 v50, a[0x8a]
v_accvgpr_read_b32 v51, a[0x8b]
v_accvgpr_read_b32 v52, a[0x8c]
v_accvgpr_read_b32 v53, a[0x8d]
v_accvgpr_read_b32 v54, a[0x8e]
v_accvgpr_read_b32 v55, a[0x8f]
v_accvgpr_read_b32 v56, a[0x90]
v_accvgpr_read_b32 v57, a[0x91]
v_accvgpr_read_b32 v58, a[0x92]
v_accvgpr_read_b32 v59, a[0x93]
v_accvgpr_read_b32 v60, a[0x94]
v_accvgpr_read_b32 v61, a[0x95]
v_accvgpr_read_b32 v62, a[0x96]
v_accvgpr_read_b32 v63, a[0x97]
v_accvgpr_read_b32 v64, a[0x98]
v_accvgpr_read_b32 v65, a[0x99]
v_accvgpr_read_b32 v66, a[0x9a]
v_accvgpr_read_b32 v67, a[0x9b]
v_accvgpr_read_b32 v68, a[0x9c]
v_accvgpr_read_b32 v69, a[0x9d]
v_accvgpr_read_b32 v70, a[0x9e]
v_accvgpr_read_b32 v71, a[0x9f]
v_accvgpr_read_b32 v72, a[0xa0]
v_accvgpr_read_b32 v73, a[0xa1]
v_accvgpr_read_b32 v74, a[0xa2]
v_accvgpr_read_b32 v75, a[0xa3]
v_accvgpr_read_b32 v76, a[0xa4]
v_accvgpr_read_b32 v77, a[0xa5]
v_accvgpr_read_b32 v78, a[0xa6]
v_accvgpr_read_b32 v79, a[0xa7]
v_accvgpr_read_b32 v80, a[0xa8]
v_accvgpr_read_b32 v81, a[0xa9]
v_accvgpr_read_b32 v82, a[0xaa]
v_accvgpr_read_b32 v83, a[0xab]
v_accvgpr_read_b32 v84, a[0xac]
v_accvgpr_read_b32 v85, a[0xad]
v_accvgpr_read_b32 v86, a[0xae]
v_accvgpr_read_b32 v87, a[0xaf]
v_accvgpr_read_b32 v88, a[0xb0]
v_accvgpr_read_b32 v89, a[0xb1]
v_accvgpr_read_b32 v90, a[0xb2]
v_accvgpr_read_b32 v91, a[0xb3]
v_accvgpr_read_b32 v92, a[0xb4]
v_accvgpr_read_b32 v93, a[0xb5]
v_accvgpr_read_b32 v94, a[0xb6]
v_accvgpr_read_b32 v95, a[0xb7]
v_accvgpr_read_b32 v96, a[0xb8]
v_accvgpr_read_b32 v97, a[0xb9]
v_accvgpr_read_b32 v98, a[0xba]
v_accvgpr_read_b32 v99, a[0xbb]
v_accvgpr_read_b32 v100, a[0xbc]
v_accvgpr_read_b32 v101, a[0xbd]
v_accvgpr_read_b32 v102, a[0xbe]
v_accvgpr_read_b32 v103, a[0xbf]

	;;#ASMEND
	s_nop 0
	v_mul_f32_e32 v29, s27, v29
	v_mul_f32_e32 v40, s27, v36
	v_mul_f32_e32 v42, s27, v37
	v_mul_f32_e32 v104, s27, v38
	v_lshl_add_u64 v[36:37], v[34:35], 0, v[0:1]
	v_lshl_add_u64 v[38:39], v[34:35], 0, v[2:3]
	global_store_short_d16_hi v[36:37], v29, off
	global_store_short_d16_hi v[38:39], v40, off
	v_lshl_add_u64 v[40:41], v[34:35], 0, v[6:7]
	global_store_short_d16_hi v[40:41], v42, off
	v_lshl_add_u64 v[42:43], v[34:35], 0, v[12:13]
	v_mul_f32_e32 v29, s27, v44
	v_mul_f32_e32 v44, s27, v45
	v_mul_f32_e32 v45, s27, v46
	v_mul_f32_e32 v46, s27, v47
	global_store_short_d16_hi v[42:43], v104, off
	global_store_short_d16_hi v[36:37], v29, off offset:32
	global_store_short_d16_hi v[38:39], v44, off offset:32
	global_store_short_d16_hi v[40:41], v45, off offset:32
	global_store_short_d16_hi v[42:43], v46, off offset:32
	v_mul_f32_e32 v29, s27, v48
	v_mul_f32_e32 v44, s27, v49
	v_mul_f32_e32 v45, s27, v50
	v_mul_f32_e32 v46, s27, v51
	global_store_short_d16_hi v[36:37], v29, off offset:64
	global_store_short_d16_hi v[38:39], v44, off offset:64
	global_store_short_d16_hi v[40:41], v45, off offset:64
	global_store_short_d16_hi v[42:43], v46, off offset:64
	v_mul_f32_e32 v29, s27, v52
	v_mul_f32_e32 v44, s27, v53
	v_mul_f32_e32 v45, s27, v54
	v_mul_f32_e32 v46, s27, v55
	global_store_short_d16_hi v[36:37], v29, off offset:96
	global_store_short_d16_hi v[38:39], v44, off offset:96
	global_store_short_d16_hi v[40:41], v45, off offset:96
	global_store_short_d16_hi v[42:43], v46, off offset:96
	v_mul_f32_e32 v29, s27, v56
	v_mul_f32_e32 v48, s27, v57
	v_lshl_add_u64 v[44:45], v[34:35], 0, v[4:5]
	v_lshl_add_u64 v[46:47], v[34:35], 0, v[8:9]
	v_mul_f32_e32 v50, s27, v58
	global_store_short_d16_hi v[44:45], v29, off
	global_store_short_d16_hi v[46:47], v48, off
	v_lshl_add_u64 v[48:49], v[34:35], 0, v[14:15]
	v_mul_f32_e32 v52, s27, v59
	global_store_short_d16_hi v[48:49], v50, off
	v_lshl_add_u64 v[50:51], v[34:35], 0, v[20:21]
	global_store_short_d16_hi v[50:51], v52, off
	v_mul_f32_e32 v29, s27, v60
	v_mul_f32_e32 v52, s27, v61
	v_mul_f32_e32 v53, s27, v62
	v_mul_f32_e32 v54, s27, v63
	global_store_short_d16_hi v[44:45], v29, off offset:32
	global_store_short_d16_hi v[46:47], v52, off offset:32
	global_store_short_d16_hi v[48:49], v53, off offset:32
	global_store_short_d16_hi v[50:51], v54, off offset:32
	v_mul_f32_e32 v29, s27, v64
	v_mul_f32_e32 v52, s27, v65
	v_mul_f32_e32 v53, s27, v66
	v_mul_f32_e32 v54, s27, v67
	global_store_short_d16_hi v[44:45], v29, off offset:64
	global_store_short_d16_hi v[46:47], v52, off offset:64
	global_store_short_d16_hi v[48:49], v53, off offset:64
	global_store_short_d16_hi v[50:51], v54, off offset:64
	v_mul_f32_e32 v29, s27, v68
	v_mul_f32_e32 v52, s27, v69
	v_mul_f32_e32 v53, s27, v70
	v_mul_f32_e32 v54, s27, v71
	global_store_short_d16_hi v[44:45], v29, off offset:96
	global_store_short_d16_hi v[46:47], v52, off offset:96
	global_store_short_d16_hi v[48:49], v53, off offset:96
	global_store_short_d16_hi v[50:51], v54, off offset:96
	v_mul_f32_e32 v29, s27, v72
	v_mul_f32_e32 v56, s27, v73
	v_lshl_add_u64 v[52:53], v[34:35], 0, v[10:11]
	v_lshl_add_u64 v[54:55], v[34:35], 0, v[16:17]
	v_mul_f32_e32 v58, s27, v74
	global_store_short_d16_hi v[52:53], v29, off
	global_store_short_d16_hi v[54:55], v56, off
	v_lshl_add_u64 v[56:57], v[34:35], 0, v[22:23]
	v_mul_f32_e32 v60, s27, v75
	global_store_short_d16_hi v[56:57], v58, off
	v_lshl_add_u64 v[58:59], v[34:35], 0, v[26:27]
	global_store_short_d16_hi v[58:59], v60, off
	v_mul_f32_e32 v29, s27, v76
	v_mul_f32_e32 v60, s27, v77
	v_mul_f32_e32 v61, s27, v78
	v_mul_f32_e32 v62, s27, v79
	global_store_short_d16_hi v[52:53], v29, off offset:32
	global_store_short_d16_hi v[54:55], v60, off offset:32
	global_store_short_d16_hi v[56:57], v61, off offset:32
	global_store_short_d16_hi v[58:59], v62, off offset:32
	v_mul_f32_e32 v29, s27, v80
	v_mul_f32_e32 v60, s27, v81
	v_mul_f32_e32 v61, s27, v82
	v_mul_f32_e32 v62, s27, v83
	global_store_short_d16_hi v[52:53], v29, off offset:64
	global_store_short_d16_hi v[54:55], v60, off offset:64
	global_store_short_d16_hi v[56:57], v61, off offset:64
	global_store_short_d16_hi v[58:59], v62, off offset:64
	v_mul_f32_e32 v29, s27, v84
	v_mul_f32_e32 v60, s27, v85
	v_mul_f32_e32 v61, s27, v86
	v_mul_f32_e32 v62, s27, v87
	global_store_short_d16_hi v[52:53], v29, off offset:96
	global_store_short_d16_hi v[54:55], v60, off offset:96
	global_store_short_d16_hi v[56:57], v61, off offset:96
	global_store_short_d16_hi v[58:59], v62, off offset:96
	v_mul_f32_e32 v29, s27, v88
	v_mul_f32_e32 v64, s27, v89
	v_lshl_add_u64 v[60:61], v[34:35], 0, v[18:19]
	v_lshl_add_u64 v[62:63], v[34:35], 0, v[24:25]
	v_mul_f32_e32 v66, s27, v90
	global_store_short_d16_hi v[60:61], v29, off
	global_store_short_d16_hi v[62:63], v64, off
	v_lshl_add_u64 v[64:65], v[34:35], 0, v[30:31]
	v_mul_f32_e32 v68, s27, v91
	global_store_short_d16_hi v[64:65], v66, off
	v_lshl_add_u64 v[66:67], v[34:35], 0, v[32:33]
	global_store_short_d16_hi v[66:67], v68, off
	v_mul_f32_e32 v29, s27, v92
	v_mul_f32_e32 v68, s27, v93
	v_mul_f32_e32 v69, s27, v94
	v_mul_f32_e32 v70, s27, v95
	global_store_short_d16_hi v[60:61], v29, off offset:32
	global_store_short_d16_hi v[62:63], v68, off offset:32
	global_store_short_d16_hi v[64:65], v69, off offset:32
	global_store_short_d16_hi v[66:67], v70, off offset:32
	v_mul_f32_e32 v29, s27, v96
	v_mul_f32_e32 v68, s27, v97
	v_mul_f32_e32 v69, s27, v98
	v_mul_f32_e32 v70, s27, v99
	global_store_short_d16_hi v[60:61], v29, off offset:64
	global_store_short_d16_hi v[62:63], v68, off offset:64
	global_store_short_d16_hi v[64:65], v69, off offset:64
	global_store_short_d16_hi v[66:67], v70, off offset:64
	v_mul_f32_e32 v29, s27, v100
	v_mul_f32_e32 v68, s27, v101
	v_mul_f32_e32 v69, s27, v102
	v_mul_f32_e32 v70, s27, v103
	global_store_short_d16_hi v[60:61], v29, off offset:96
	global_store_short_d16_hi v[62:63], v68, off offset:96
	global_store_short_d16_hi v[64:65], v69, off offset:96
	global_store_short_d16_hi v[66:67], v70, off offset:96
	;;#ASMSTART
	v_accvgpr_read_b32 v29, a[0xc0]
v_accvgpr_read_b32 v68, a[0xc1]
v_accvgpr_read_b32 v69, a[0xc2]
v_accvgpr_read_b32 v70, a[0xc3]
v_accvgpr_read_b32 v71, a[0xc4]
v_accvgpr_read_b32 v72, a[0xc5]
v_accvgpr_read_b32 v73, a[0xc6]
v_accvgpr_read_b32 v74, a[0xc7]
v_accvgpr_read_b32 v75, a[0xc8]
v_accvgpr_read_b32 v76, a[0xc9]
v_accvgpr_read_b32 v77, a[0xca]
v_accvgpr_read_b32 v78, a[0xcb]
v_accvgpr_read_b32 v79, a[0xcc]
v_accvgpr_read_b32 v80, a[0xcd]
v_accvgpr_read_b32 v81, a[0xce]
v_accvgpr_read_b32 v82, a[0xcf]
v_accvgpr_read_b32 v83, a[0xd0]
v_accvgpr_read_b32 v84, a[0xd1]
v_accvgpr_read_b32 v85, a[0xd2]
v_accvgpr_read_b32 v86, a[0xd3]
v_accvgpr_read_b32 v87, a[0xd4]
v_accvgpr_read_b32 v88, a[0xd5]
v_accvgpr_read_b32 v89, a[0xd6]
v_accvgpr_read_b32 v90, a[0xd7]
v_accvgpr_read_b32 v91, a[0xd8]
v_accvgpr_read_b32 v92, a[0xd9]
v_accvgpr_read_b32 v93, a[0xda]
v_accvgpr_read_b32 v94, a[0xdb]
v_accvgpr_read_b32 v95, a[0xdc]
v_accvgpr_read_b32 v96, a[0xdd]
v_accvgpr_read_b32 v97, a[0xde]
v_accvgpr_read_b32 v98, a[0xdf]
v_accvgpr_read_b32 v99, a[0xe0]
v_accvgpr_read_b32 v100, a[0xe1]
v_accvgpr_read_b32 v101, a[0xe2]
v_accvgpr_read_b32 v102, a[0xe3]
v_accvgpr_read_b32 v103, a[0xe4]
v_accvgpr_read_b32 v104, a[0xe5]
v_accvgpr_read_b32 v105, a[0xe6]
v_accvgpr_read_b32 v106, a[0xe7]
v_accvgpr_read_b32 v107, a[0xe8]
v_accvgpr_read_b32 v108, a[0xe9]
v_accvgpr_read_b32 v109, a[0xea]
v_accvgpr_read_b32 v110, a[0xeb]
v_accvgpr_read_b32 v111, a[0xec]
v_accvgpr_read_b32 v112, a[0xed]
v_accvgpr_read_b32 v113, a[0xee]
v_accvgpr_read_b32 v114, a[0xef]
v_accvgpr_read_b32 v115, a[0xf0]
v_accvgpr_read_b32 v116, a[0xf1]
v_accvgpr_read_b32 v117, a[0xf2]
v_accvgpr_read_b32 v118, a[0xf3]
v_accvgpr_read_b32 v119, a[0xf4]
v_accvgpr_read_b32 v120, a[0xf5]
v_accvgpr_read_b32 v121, a[0xf6]
v_accvgpr_read_b32 v122, a[0xf7]
v_accvgpr_read_b32 v123, a[0xf8]
v_accvgpr_read_b32 v124, a[0xf9]
v_accvgpr_read_b32 v125, a[0xfa]
v_accvgpr_read_b32 v126, a[0xfb]
v_accvgpr_read_b32 v127, a[0xfc]
v_accvgpr_read_b32 v128, a[0xfd]
v_accvgpr_read_b32 v129, a[0xfe]
v_accvgpr_read_b32 v130, a[0xff]

	;;#ASMEND
	v_lshl_add_u64 v[34:35], v[34:35], 0, s[0:1]
	v_mul_f32_e32 v29, s27, v29
	v_mul_f32_e32 v68, s27, v68
	v_mul_f32_e32 v69, s27, v69
	v_mul_f32_e32 v70, s27, v70
	global_store_short_d16_hi v[36:37], v29, off offset:256
	global_store_short_d16_hi v[38:39], v68, off offset:256
	global_store_short_d16_hi v[40:41], v69, off offset:256
	global_store_short_d16_hi v[42:43], v70, off offset:256
	v_mul_f32_e32 v29, s27, v71
	v_lshl_add_u64 v[0:1], v[34:35], 0, v[0:1]
	v_mul_f32_e32 v36, s27, v72
	v_mul_f32_e32 v37, s27, v73
	v_mul_f32_e32 v38, s27, v74
	global_store_short_d16_hi v[0:1], v29, off offset:32
	v_lshl_add_u64 v[2:3], v[34:35], 0, v[2:3]
	v_lshl_add_u64 v[6:7], v[34:35], 0, v[6:7]
	v_lshl_add_u64 v[12:13], v[34:35], 0, v[12:13]
	v_mul_f32_e32 v29, s27, v75
	global_store_short_d16_hi v[2:3], v36, off offset:32
	global_store_short_d16_hi v[6:7], v37, off offset:32
	global_store_short_d16_hi v[12:13], v38, off offset:32
	v_mul_f32_e32 v36, s27, v76
	v_mul_f32_e32 v37, s27, v77
	v_mul_f32_e32 v38, s27, v78
	global_store_short_d16_hi v[0:1], v29, off offset:64
	global_store_short_d16_hi v[2:3], v36, off offset:64
	global_store_short_d16_hi v[6:7], v37, off offset:64
	global_store_short_d16_hi v[12:13], v38, off offset:64
	v_mul_f32_e32 v29, s27, v79
	v_mul_f32_e32 v36, s27, v80
	v_mul_f32_e32 v37, s27, v81
	v_mul_f32_e32 v38, s27, v82
	global_store_short_d16_hi v[0:1], v29, off offset:96
	global_store_short_d16_hi v[2:3], v36, off offset:96
	global_store_short_d16_hi v[6:7], v37, off offset:96
	global_store_short_d16_hi v[12:13], v38, off offset:96
	v_mul_f32_e32 v0, s27, v83
	v_mul_f32_e32 v1, s27, v84
	v_mul_f32_e32 v2, s27, v85
	v_mul_f32_e32 v3, s27, v86
	global_store_short_d16_hi v[44:45], v0, off offset:256
	global_store_short_d16_hi v[46:47], v1, off offset:256
	global_store_short_d16_hi v[48:49], v2, off offset:256
	global_store_short_d16_hi v[50:51], v3, off offset:256
	v_mul_f32_e32 v2, s27, v87
	v_lshl_add_u64 v[0:1], v[34:35], 0, v[4:5]
	v_mul_f32_e32 v6, s27, v88
	v_mul_f32_e32 v7, s27, v89
	global_store_short_d16_hi v[0:1], v2, off offset:32
	v_lshl_add_u64 v[2:3], v[34:35], 0, v[8:9]
	v_lshl_add_u64 v[4:5], v[34:35], 0, v[14:15]
	v_mul_f32_e32 v12, s27, v90
	global_store_short_d16_hi v[2:3], v6, off offset:32
	global_store_short_d16_hi v[4:5], v7, off offset:32
	v_lshl_add_u64 v[6:7], v[34:35], 0, v[20:21]
	v_mul_f32_e32 v8, s27, v91
	global_store_short_d16_hi v[6:7], v12, off offset:32
	v_mul_f32_e32 v9, s27, v92
	v_mul_f32_e32 v12, s27, v93
	v_mul_f32_e32 v13, s27, v94
	global_store_short_d16_hi v[0:1], v8, off offset:64
	global_store_short_d16_hi v[2:3], v9, off offset:64
	global_store_short_d16_hi v[4:5], v12, off offset:64
	global_store_short_d16_hi v[6:7], v13, off offset:64
	v_mul_f32_e32 v8, s27, v95
	v_mul_f32_e32 v9, s27, v96
	v_mul_f32_e32 v12, s27, v97
	v_mul_f32_e32 v13, s27, v98
	global_store_short_d16_hi v[0:1], v8, off offset:96
	global_store_short_d16_hi v[2:3], v9, off offset:96
	global_store_short_d16_hi v[4:5], v12, off offset:96
	global_store_short_d16_hi v[6:7], v13, off offset:96
	v_mul_f32_e32 v0, s27, v99
	v_mul_f32_e32 v1, s27, v100
	v_mul_f32_e32 v2, s27, v101
	v_mul_f32_e32 v3, s27, v102
	global_store_short_d16_hi v[52:53], v0, off offset:256
	global_store_short_d16_hi v[54:55], v1, off offset:256
	global_store_short_d16_hi v[56:57], v2, off offset:256
	global_store_short_d16_hi v[58:59], v3, off offset:256
	v_mul_f32_e32 v2, s27, v103
	v_lshl_add_u64 v[0:1], v[34:35], 0, v[10:11]
	v_mul_f32_e32 v4, s27, v104
	global_store_short_d16_hi v[0:1], v2, off offset:32
	v_lshl_add_u64 v[2:3], v[34:35], 0, v[16:17]
	v_mul_f32_e32 v6, s27, v105
	global_store_short_d16_hi v[2:3], v4, off offset:32
	v_lshl_add_u64 v[4:5], v[34:35], 0, v[22:23]
	v_mul_f32_e32 v8, s27, v106
	global_store_short_d16_hi v[4:5], v6, off offset:32
	v_lshl_add_u64 v[6:7], v[34:35], 0, v[26:27]
	global_store_short_d16_hi v[6:7], v8, off offset:32
	v_mul_f32_e32 v8, s27, v107
	v_mul_f32_e32 v9, s27, v108
	v_mul_f32_e32 v10, s27, v109
	v_mul_f32_e32 v11, s27, v110
	global_store_short_d16_hi v[0:1], v8, off offset:64
	global_store_short_d16_hi v[2:3], v9, off offset:64
	global_store_short_d16_hi v[4:5], v10, off offset:64
	global_store_short_d16_hi v[6:7], v11, off offset:64
	v_mul_f32_e32 v8, s27, v111
	v_mul_f32_e32 v9, s27, v112
	v_mul_f32_e32 v10, s27, v113
	v_mul_f32_e32 v11, s27, v114
	global_store_short_d16_hi v[0:1], v8, off offset:96
	global_store_short_d16_hi v[2:3], v9, off offset:96
	global_store_short_d16_hi v[4:5], v10, off offset:96
	global_store_short_d16_hi v[6:7], v11, off offset:96
	v_mul_f32_e32 v0, s27, v115
	v_mul_f32_e32 v1, s27, v116
	v_mul_f32_e32 v2, s27, v117
	v_mul_f32_e32 v3, s27, v118
	global_store_short_d16_hi v[60:61], v0, off offset:256
	global_store_short_d16_hi v[62:63], v1, off offset:256
	global_store_short_d16_hi v[64:65], v2, off offset:256
	global_store_short_d16_hi v[66:67], v3, off offset:256
	v_mul_f32_e32 v2, s27, v119
	v_lshl_add_u64 v[0:1], v[34:35], 0, v[18:19]
	v_mul_f32_e32 v4, s27, v120
	global_store_short_d16_hi v[0:1], v2, off offset:32
	v_lshl_add_u64 v[2:3], v[34:35], 0, v[24:25]
	v_mul_f32_e32 v6, s27, v121
	global_store_short_d16_hi v[2:3], v4, off offset:32
	v_lshl_add_u64 v[4:5], v[34:35], 0, v[30:31]
	v_mul_f32_e32 v8, s27, v122
	global_store_short_d16_hi v[4:5], v6, off offset:32
	v_lshl_add_u64 v[6:7], v[34:35], 0, v[32:33]
	global_store_short_d16_hi v[6:7], v8, off offset:32
	v_mul_f32_e32 v8, s27, v123
	v_mul_f32_e32 v9, s27, v124
	v_mul_f32_e32 v10, s27, v125
	v_mul_f32_e32 v11, s27, v126
	global_store_short_d16_hi v[0:1], v8, off offset:64
	global_store_short_d16_hi v[2:3], v9, off offset:64
	global_store_short_d16_hi v[4:5], v10, off offset:64
	global_store_short_d16_hi v[6:7], v11, off offset:64
	v_mul_f32_e32 v8, s27, v127
	v_mul_f32_e32 v9, s27, v128
	v_mul_f32_e32 v10, s27, v129
	v_mul_f32_e32 v11, s27, v130
	global_store_short_d16_hi v[0:1], v8, off offset:96
	global_store_short_d16_hi v[2:3], v9, off offset:96
	global_store_short_d16_hi v[4:5], v10, off offset:96
	global_store_short_d16_hi v[6:7], v11, off offset:96
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z16mxfp4_art_kernel13gluon_globals
		.amdhsa_group_segment_fixed_size 65536
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 504
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
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 512
		.amdhsa_next_free_sgpr 100
		.amdhsa_accum_offset 256
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
	.size	_Z16mxfp4_art_kernel13gluon_globals, .Lfunc_end0-_Z16mxfp4_art_kernel13gluon_globals
                                        ; -- End function
	.set _Z16mxfp4_art_kernel13gluon_globals.num_vgpr, 256
	.set _Z16mxfp4_art_kernel13gluon_globals.num_agpr, 256
	.set _Z16mxfp4_art_kernel13gluon_globals.numbered_sgpr, 100
	.set _Z16mxfp4_art_kernel13gluon_globals.private_seg_size, 0
	.set _Z16mxfp4_art_kernel13gluon_globals.uses_vcc, 1
	.set _Z16mxfp4_art_kernel13gluon_globals.uses_flat_scratch, 0
	.set _Z16mxfp4_art_kernel13gluon_globals.has_dyn_sized_stack, 0
	.set _Z16mxfp4_art_kernel13gluon_globals.has_recursion, 0
	.set _Z16mxfp4_art_kernel13gluon_globals.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 64572
; TotalNumSgprs: 106
; NumVgprs: 256
; NumAgprs: 256
; TotalNumVgprs: 512
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 65536 bytes/workgroup (compile time only)
; SGPRBlocks: 13
; VGPRBlocks: 63
; NumSGPRsForWavesPerEU: 106
; NumVGPRsForWavesPerEU: 512
; AccumOffset: 256
; Occupancy: 1
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 63
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.type	__hip_cuid_a539bdc6c6af0e7b,@object ; @__hip_cuid_a539bdc6c6af0e7b
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_a539bdc6c6af0e7b
__hip_cuid_a539bdc6c6af0e7b:
	.byte	0                               ; 0x0
	.size	__hip_cuid_a539bdc6c6af0e7b, 1

	.ident	"clang version 20.0.0git (https://github.com/ROCm/llvm-project.git 27682a16360e33e37c4f3cc6adf9a620733f8fe1)"
	.ident	"AMD clang version 20.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.1.0 25425 1b0eada6b0ee93e2e694c8c146d23fca90bc11c5)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_a539bdc6c6af0e7b
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     256
    .args:
      - .offset:         0
        .size:           248
        .value_kind:     by_value
      - .offset:         248
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         252
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         256
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         260
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         262
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         264
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         266
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         268
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         270
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         288
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         296
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         304
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         312
        .size:           2
        .value_kind:     hidden_grid_dims
    .group_segment_fixed_size: 65536
    .kernarg_segment_align: 8
    .kernarg_segment_size: 504
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _Z16mxfp4_art_kernel13gluon_globals
    .private_segment_fixed_size: 0
    .sgpr_count:     106
    .sgpr_spill_count: 2
    .symbol:         _Z16mxfp4_art_kernel13gluon_globals.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     512
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
