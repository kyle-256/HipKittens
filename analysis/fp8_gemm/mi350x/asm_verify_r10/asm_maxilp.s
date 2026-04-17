	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.text
	.protected	_Z22mxfp4_gluon_cpp_kernel13gluon_globals ; -- Begin function _Z22mxfp4_gluon_cpp_kernel13gluon_globals
	.globl	_Z22mxfp4_gluon_cpp_kernel13gluon_globals
	.p2align	8
	.type	_Z22mxfp4_gluon_cpp_kernel13gluon_globals,@function
_Z22mxfp4_gluon_cpp_kernel13gluon_globals: ; @_Z22mxfp4_gluon_cpp_kernel13gluon_globals
; %bb.0:
	s_load_dword s6, s[0:1], 0xf8
	v_mov_b32_e32 v206, v0
	s_waitcnt lgkmcnt(0)
	s_add_i32 s3, s6, 7
	s_ashr_i32 s10, s6, 31
	s_ashr_i32 s4, s3, 31
	s_lshr_b32 s5, s10, 29
	s_lshr_b32 s4, s4, 29
	s_add_i32 s5, s6, s5
	s_add_i32 s3, s3, s4
	s_and_b32 s5, s5, -8
	s_ashr_i32 s4, s3, 3
	s_sub_i32 s3, s6, s5
	s_cmp_lg_u32 s3, 0
	s_cselect_b32 s3, s3, 8
	s_ashr_i32 s5, s2, 31
	s_lshr_b32 s5, s5, 29
	s_add_i32 s7, s2, s5
	s_and_b32 s5, s7, -8
	s_sub_i32 s5, s2, s5
	s_cmp_ge_i32 s5, s3
	s_cbranch_scc0 .LBB0_2
; %bb.1:
	s_sub_i32 s9, s5, s3
	s_add_i32 s11, s4, -1
	s_mul_i32 s8, s3, s4
	s_mul_i32 s9, s9, s11
	s_add_i32 s8, s9, s8
	s_ashr_i32 s2, s7, 3
	s_cbranch_execz .LBB0_3
	s_branch .LBB0_4
.LBB0_2:
                                        ; implicit-def: $sgpr8
	s_ashr_i32 s2, s7, 3
.LBB0_3:
	s_mul_i32 s8, s4, s5
.LBB0_4:
	s_add_i32 s7, s8, s2
	s_cmp_ge_i32 s7, s6
	s_cbranch_scc0 .LBB0_6
; %bb.5:
	s_endpgm
.LBB0_6:
	s_load_dwordx2 s[30:31], s[0:1], 0x0
	s_load_dwordx2 s[8:9], s[0:1], 0x20
	s_load_dwordx2 s[34:35], s[0:1], 0x30
	s_load_dwordx2 s[14:15], s[0:1], 0x50
	s_load_dwordx2 s[4:5], s[0:1], 0x60
	s_load_dwordx2 s[16:17], s[0:1], 0x80
	s_load_dwordx2 s[12:13], s[0:1], 0x90
	s_load_dwordx2 s[18:19], s[0:1], 0xb0
	s_load_dwordx2 s[24:25], s[0:1], 0xc0
	s_load_dwordx2 s[26:27], s[0:1], 0xe0
	s_load_dword s28, s[0:1], 0xf0
	s_waitcnt lgkmcnt(0)
	s_lshr_b32 s9, s10, 28
	s_add_i32 s6, s6, s9
	s_ashr_i32 s6, s6, 4
	v_lshrrev_b32_e32 v1, 6, v206
	v_lshlrev_b32_e32 v0, 10, v1
	s_abs_i32 s1, s7
	v_readfirstlane_b32 s38, v0
	s_add_i32 s45, s38, 0x4000
	s_ashr_i32 s0, s7, 31
	s_add_i32 s39, s38, 0x18000
	s_add_i32 s40, s39, 0x4000
	s_add_i32 s41, s38, 0x10000
	s_add_i32 s42, s41, 0x4000
	s_add_i32 s43, s38, 0x8000
	s_add_i32 s44, s43, 0x4000
	v_lshrrev_b32_e32 v4, 7, v206
	v_lshlrev_b32_e32 v207, 6, v4
	s_mov_b32 s3, 0x110000
	v_mov_b64_e32 v[76:77], s[4:5]
	s_lshr_b32 s4, s0, 26
	s_add_i32 s4, s7, s4
	s_ashr_i32 s5, s4, 6
	s_lshl_b32 s5, s5, 2
	s_sub_i32 s6, s6, s5
	s_min_i32 s6, s6, 4
	s_abs_i32 s9, s6
	s_sub_i32 s10, 0, s9
	s_mov_b32 s2, -1
	v_cvt_f32_u32_e32 v0, s9
	v_lshlrev_b32_e32 v7, 4, v206
	s_movk_i32 s17, 0x70
	v_lshrrev_b32_e32 v6, 3, v206
	v_rcp_iflag_f32_e32 v0, v0
	v_lshlrev_b32_e32 v5, 7, v206
	v_lshlrev_b32_e32 v4, 13, v4
	v_and_b32_e32 v5, 0x780, v5
	v_mul_f32_e32 v0, 0x4f7ffffe, v0
	v_cvt_u32_f32_e32 v0, v0
	v_and_b32_e32 v3, 48, v206
	v_bfe_u32 v2, v206, 6, 1
	v_lshlrev_b32_e32 v34, 3, v206
	v_lshlrev_b32_e32 v78, 6, v2
	v_readfirstlane_b32 s11, v0
	s_mul_i32 s10, s10, s11
	s_mul_hi_u32 s10, s11, s10
	s_add_i32 s11, s11, s10
	s_mul_hi_u32 s10, s1, s11
	s_mul_i32 s10, s10, s9
	s_sub_i32 s1, s1, s10
	s_sub_i32 s10, s1, s9
	s_cmp_ge_u32 s1, s9
	s_cselect_b32 s1, s10, s1
	s_sub_i32 s10, s1, s9
	s_cmp_ge_u32 s1, s9
	s_cselect_b32 s1, s10, s1
	s_xor_b32 s1, s1, s0
	s_sub_i32 s27, s1, s0
	s_add_i32 s27, s27, s5
	v_or_b32_e32 v36, v5, v3
	v_or_b32_e32 v18, v36, v4
	v_bitop3_b32 v14, v34, v18, s17 bitop3:0x6c
	v_or_b32_e32 v18, 64, v18
	v_bitop3_b32 v30, v34, v18, s17 bitop3:0x6c
	s_andn2_b32 s4, s4, 63
	s_sub_i32 s4, s7, s4
	s_abs_i32 s7, s4
	s_mul_hi_u32 s10, s7, s11
	s_mul_i32 s0, s10, s9
	s_sub_i32 s0, s7, s0
	s_sub_i32 s1, s0, s9
	v_and_b32_e32 v110, 0x1f8, v34
	s_add_i32 s11, s10, 1
	s_xor_b32 s4, s4, s6
	s_ashr_i32 s15, s4, 31
	s_cmp_ge_u32 s0, s9
	s_cselect_b32 s4, s11, s10
	s_mov_b32 s29, 7
	s_cselect_b32 s0, s1, s0
	s_add_i32 s1, s4, 1
	s_cmp_ge_u32 s0, s9
	s_cselect_b32 s22, s1, s4
	s_add_i32 s75, s39, 0x7000
	s_movk_i32 s33, 0x1000
	s_lshl_b32 s19, s27, 8
	s_mul_i32 s46, s8, s19
	s_mov_b32 s9, s46
	;;#ASMSTART
	;;#ASMEND
	s_add_i32 s47, s38, 0x2000
	v_or_b32_e32 v0, s19, v207
	v_ashrrev_i32_e32 v0, 6, v0
	v_mad_i64_i32 v[0:1], s[0:1], s16, v0, v[76:77]
	s_nop 0
	v_readfirstlane_b32 s0, v0
	v_bitop3_b32 v0, v7, s17, v206 bitop3:0x48
	v_mad_u64_u32 v[66:67], s[10:11], v6, s8, v[0:1]
	s_lshl_b32 s10, s8, 5
	s_nop 0
	v_add_u32_e32 v67, s10, v66
	v_add_u32_e32 v94, s10, v67
	s_mov_b32 s10, s47
	s_add_i32 s48, s38, 0x3000
	v_readfirstlane_b32 s1, v1
	s_mov_b64 s[6:7], s[2:3]
	s_mov_b64 s[4:5], s[0:1]
	s_mov_b32 s4, s30
	s_bitset1_b32 s19, 7
	s_mul_i32 s49, s8, s19
	s_mov_b32 s20, s49
	s_mov_b32 s5, s38
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s5
	s_mov_b32 s5, s31
	buffer_load_dwordx4 v66, s[4:7], s9 offen lds
	v_or_b32_e32 v1, 0x60, v6
	s_add_i32 s31, s38, 0x1000
	s_mov_b32 s11, s31
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s11
	s_add_i32 s50, s43, 0x1000
	buffer_load_dwordx4 v67, s[4:7], s9 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s10
	v_mad_u64_u32 v[68:69], s[10:11], v1, s8, v[0:1]
	s_mov_b32 s10, s48
	s_add_i32 s51, s43, 0x2000
	s_mov_b32 s8, s43
	buffer_load_dwordx4 v94, s[4:7], s9 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s10
	s_add_i32 s52, s43, 0x3000
	buffer_load_dwordx4 v68, s[4:7], s9 offen lds
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s8
	s_mov_b32 s8, s50
	s_add_i32 s54, s41, 0x2000
	buffer_load_dwordx4 v66, s[4:7], s20 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s8
	s_mov_b32 s8, s51
	s_add_i32 s55, s41, 0x3000
	buffer_load_dwordx4 v67, s[4:7], s20 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s8
	s_mov_b64 s[10:11], s[2:3]
	s_mov_b64 s[8:9], s[0:1]
	s_mov_b32 s8, s34
	s_add_i32 s57, s39, 0x1000
	s_mov_b32 s9, s52
	buffer_load_dwordx4 v94, s[4:7], s20 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s9
	s_xor_b32 s9, s22, s15
	s_sub_i32 s9, s9, s15
	s_add_i32 s58, s39, 0x2000
	buffer_load_dwordx4 v68, s[4:7], s20 offen lds
	v_mad_u64_u32 v[70:71], s[20:21], v6, s14, v[0:1]
	s_lshl_b32 s20, s9, 8
	s_nop 0
	v_or_b32_e32 v80, s20, v78
	s_add_i32 s59, s39, 0x3000
	s_mov_b32 s9, s35
	s_add_i32 s35, s41, 0x1000
	s_mov_b32 s22, s35
	s_mul_i32 s53, s14, s20
	s_mov_b32 s15, s53
	;;#ASMSTART
	;;#ASMEND
	v_or_b32_e32 v6, v5, v4
	v_or_b32_e32 v7, 0xc000, v6
	s_mov_b32 s21, s41
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s21
	s_lshl_b32 s21, s14, 5
	v_add_u32_e32 v69, s21, v70
	v_add_u32_e32 v71, s21, v69
	s_mov_b32 s21, s54
	s_add_i32 s63, s38, 0x5000
	buffer_load_dwordx4 v70, s[8:11], s15 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s22
	v_mad_u64_u32 v[72:73], s[22:23], v1, s14, v[0:1]
	v_lshlrev_b32_e32 v1, 13, v2
	v_or_b32_e32 v35, 0x10000, v1
	v_or_b32_e32 v50, v36, v35
	s_add_i32 s65, s38, 0x6000
	v_or_b32_e32 v1, v1, v5
	v_or_b32_e32 v0, 64, v3
	v_or_b32_e32 v8, v7, v0
	v_bitop3_b32 v73, v34, v8, s17 bitop3:0x6c
	buffer_load_dwordx4 v69, s[8:11], s15 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s21
	s_mov_b32 s21, s55
	v_or_b32_e32 v7, v7, v3
	v_bitop3_b32 v95, v34, v7, s17 bitop3:0x6c
	v_or_b32_e32 v7, 0x8000, v6
	v_or_b32_e32 v8, v7, v0
	v_bitop3_b32 v96, v34, v8, s17 bitop3:0x6c
	buffer_load_dwordx4 v71, s[8:11], s15 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s21
	s_mov_b32 s21, s39
	v_or_b32_e32 v7, v7, v3
	v_bitop3_b32 v97, v34, v7, s17 bitop3:0x6c
	v_or_b32_e32 v7, 0x1c000, v1
	v_or_b32_e32 v8, v7, v0
	v_bitop3_b32 v98, v34, v8, s17 bitop3:0x6c
	buffer_load_dwordx4 v72, s[8:11], s15 offen lds
	s_or_b32 s15, s20, 0x80
	v_or_b32_e32 v2, s15, v78
	v_ashrrev_i32_e32 v2, 6, v2
	v_or_b32_e32 v7, v7, v3
	v_bitop3_b32 v99, v34, v7, s17 bitop3:0x6c
	v_or_b32_e32 v7, 0x18000, v1
	v_or_b32_e32 v8, v7, v0
	v_bitop3_b32 v100, v34, v8, s17 bitop3:0x6c
	v_mov_b32_e32 v78, v80
	v_or_b32_e32 v7, v7, v3
	v_bitop3_b32 v101, v34, v7, s17 bitop3:0x6c
	v_or_b32_e32 v1, 0x14000, v1
	v_or_b32_e32 v7, v1, v0
	v_bitop3_b32 v102, v34, v7, s17 bitop3:0x6c
	v_or_b32_e32 v7, v5, v35
	v_add_u32_e32 v9, v7, v0
	v_lshrrev_b32_e32 v10, 4, v9
	v_bitop3_b32 v104, v10, v9, s17 bitop3:0x6c
	s_mul_i32 s56, s14, s15
	s_mov_b32 s14, s56
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s21
	s_mov_b32 s21, s57
	v_add_u32_e32 v7, v7, v3
	v_lshrrev_b32_e32 v9, 4, v7
	v_bitop3_b32 v105, v9, v7, s17 bitop3:0x6c
	buffer_load_dwordx4 v70, s[8:11], s14 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s21
	s_mov_b32 s21, s58
	v_or_b32_e32 v1, v1, v3
	v_bitop3_b32 v103, v34, v1, s17 bitop3:0x6c
	v_or_b32_e32 v1, 0x4000, v6
	v_or_b32_e32 v8, v1, v3
	v_bitop3_b32 v106, v34, v8, s17 bitop3:0x6c
	buffer_load_dwordx4 v69, s[8:11], s14 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s21
	s_mov_b32 s21, s59
	v_or_b32_e32 v1, v1, v0
	v_bitop3_b32 v107, v34, v1, s17 bitop3:0x6c
	buffer_load_dwordx4 v71, s[8:11], s14 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s21
	s_add_u32 s21, s34, s53
	s_add_u32 s61, s21, 0x100
	s_mov_b32 s21, s45
	v_or_b32_e32 v0, v6, v0
	v_bitop3_b32 v108, v34, v0, s17 bitop3:0x6c
	v_or_b32_e32 v0, v6, v3
	v_bitop3_b32 v109, v34, v0, s17 bitop3:0x6c
	v_lshrrev_b32_e32 v34, 4, v50
	v_bitop3_b32 v46, v34, v50, s17 bitop3:0x6c
	v_add_u32_e32 v50, 64, v50
	v_lshrrev_b32_e32 v51, 4, v50
	v_bitop3_b32 v62, v51, v50, s17 bitop3:0x6c
	buffer_load_dwordx4 v72, s[8:11], s14 offen lds
	s_add_u32 s14, s34, s56
	s_add_u32 s60, s14, 0x100
	s_or_b32 s14, s46, 0x80
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s21
	s_add_u32 s21, s30, s49
	s_add_u32 s62, s21, 0x100
	s_mov_b32 s21, s63
	v_mov_b64_e32 v[0:1], s[12:13]
	v_mad_i64_i32 v[74:75], s[12:13], s18, v2, v[0:1]
	buffer_load_dwordx4 v66, s[4:7], s14 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s21
	s_add_u32 s21, s30, s46
	s_add_u32 s64, s21, 0x100
	s_mov_b32 s21, s65
	s_add_i32 s66, s38, 0x7000
	buffer_load_dwordx4 v67, s[4:7], s14 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s21
	s_mov_b32 s21, s66
	s_add_i32 s67, s43, 0x5000
	buffer_load_dwordx4 v94, s[4:7], s14 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s21
	s_mov_b32 s21, s44
	s_add_i32 s68, s43, 0x6000
	buffer_load_dwordx4 v68, s[4:7], s14 offen lds
	s_add_i32 s14, s49, 0x80
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s21
	s_mov_b32 s21, s67
	s_add_i32 s69, s43, 0x7000
	buffer_load_dwordx4 v66, s[4:7], s14 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s21
	s_mov_b32 s21, s68
	s_add_i32 s70, s41, 0x5000
	buffer_load_dwordx4 v67, s[4:7], s14 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s21
	s_mov_b32 s21, s69
	s_add_i32 s71, s41, 0x6000
	buffer_load_dwordx4 v94, s[4:7], s14 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s21
	s_mov_b32 s21, s42
	s_add_i32 s72, s41, 0x7000
	buffer_load_dwordx4 v68, s[4:7], s14 offen lds
	s_or_b32 s14, s53, 0x80
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s21
	s_mov_b32 s21, s70
	s_add_i32 s73, s39, 0x5000
	buffer_load_dwordx4 v70, s[8:11], s14 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s21
	s_mov_b32 s21, s71
	s_add_i32 s74, s39, 0x6000
	buffer_load_dwordx4 v69, s[8:11], s14 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s21
	s_mov_b32 s21, s72
	s_mov_b64 s[36:37], 0
	buffer_load_dwordx4 v71, s[8:11], s14 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s21
	s_mov_b32 s21, s40
	v_accvgpr_write_b32 a191, 0
	buffer_load_dwordx4 v72, s[8:11], s14 offen lds
	s_add_i32 s14, s56, 0x80
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s21
	s_mov_b32 s21, s73
	v_accvgpr_write_b32 a190, 0
	buffer_load_dwordx4 v70, s[8:11], s14 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s21
	s_mov_b32 s21, s74
	buffer_load_dwordx4 v69, s[8:11], s14 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s21
	s_mov_b32 s21, s75
	buffer_load_dwordx4 v71, s[8:11], s14 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s21
	v_readfirstlane_b32 s21, v74
	buffer_load_dwordx4 v72, s[8:11], s14 offen lds
	s_mov_b64 s[14:15], s[2:3]
	s_mov_b64 s[12:13], s[0:1]
	s_mov_b32 s12, s21
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[2:5], v14 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[6:9], v14 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[10:13], v14 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[14:17], v14 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[18:21], v30 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[22:25], v30 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[26:29], v30 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[30:33], v30 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[34:37], v46 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[38:41], v46 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[42:45], v46 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[46:49], v46 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[50:53], v62 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[54:57], v62 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[58:61], v62 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[62:65], v62 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	scratch_store_dwordx2 off, v[78:79], off ; 8-byte Folded Spill
	v_ashrrev_i32_e32 v78, 6, v80
	v_mad_i64_i32 v[0:1], s[20:21], s18, v78, v[0:1]
	s_nop 0
	v_readfirstlane_b32 s20, v0
	v_or_b32_e32 v0, s19, v207
	v_ashrrev_i32_e32 v0, 6, v0
	v_mad_i64_i32 v[76:77], s[16:17], s16, v0, v[76:77]
	s_nop 0
	v_readfirstlane_b32 s77, v76
	v_accvgpr_write_b32 a189, 0
	s_mov_b64 s[18:19], s[2:3]
	s_mov_b64 s[16:17], s[0:1]
	s_mov_b32 s16, s20
	s_mov_b64 s[22:23], s[2:3]
	s_mov_b64 s[20:21], s[0:1]
	v_accvgpr_write_b32 a188, 0
	v_accvgpr_write_b32 a195, 0
	v_accvgpr_write_b32 a194, 0
	v_accvgpr_write_b32 a193, 0
	v_accvgpr_write_b32 a192, 0
	v_accvgpr_write_b32 a199, 0
	v_accvgpr_write_b32 a198, 0
	v_accvgpr_write_b32 a197, 0
	v_accvgpr_write_b32 a196, 0
	v_accvgpr_write_b32 a203, 0
	v_accvgpr_write_b32 a202, 0
	v_accvgpr_write_b32 a201, 0
	v_accvgpr_write_b32 a200, 0
	v_accvgpr_write_b32 a207, 0
	v_accvgpr_write_b32 a206, 0
	v_accvgpr_write_b32 a205, 0
	v_accvgpr_write_b32 a204, 0
	v_accvgpr_write_b32 a211, 0
	v_accvgpr_write_b32 a210, 0
	v_accvgpr_write_b32 a209, 0
	v_accvgpr_write_b32 a208, 0
	v_accvgpr_write_b32 a215, 0
	v_accvgpr_write_b32 a214, 0
	v_accvgpr_write_b32 a213, 0
	v_accvgpr_write_b32 a212, 0
	v_accvgpr_write_b32 a223, 0
	v_accvgpr_write_b32 a222, 0
	v_accvgpr_write_b32 a221, 0
	v_accvgpr_write_b32 a220, 0
	v_accvgpr_write_b32 a227, 0
	v_accvgpr_write_b32 a226, 0
	v_accvgpr_write_b32 a225, 0
	v_accvgpr_write_b32 a224, 0
	v_accvgpr_write_b32 a231, 0
	v_accvgpr_write_b32 a230, 0
	v_accvgpr_write_b32 a229, 0
	v_accvgpr_write_b32 a228, 0
	v_accvgpr_write_b32 a235, 0
	v_accvgpr_write_b32 a234, 0
	v_accvgpr_write_b32 a233, 0
	v_accvgpr_write_b32 a232, 0
	v_accvgpr_write_b32 a239, 0
	v_accvgpr_write_b32 a238, 0
	v_accvgpr_write_b32 a237, 0
	v_accvgpr_write_b32 a236, 0
	v_accvgpr_write_b32 a243, 0
	v_accvgpr_write_b32 a242, 0
	v_accvgpr_write_b32 a241, 0
	v_accvgpr_write_b32 a240, 0
	v_accvgpr_write_b32 a11, 0
	v_accvgpr_write_b32 a10, 0
	v_accvgpr_write_b32 a9, 0
	v_accvgpr_write_b32 a8, 0
	v_accvgpr_write_b32 a7, 0
	v_accvgpr_write_b32 a6, 0
	v_accvgpr_write_b32 a5, 0
	v_accvgpr_write_b32 a4, 0
	v_accvgpr_write_b32 a3, 0
	v_accvgpr_write_b32 a2, 0
	v_accvgpr_write_b32 a1, 0
	v_accvgpr_write_b32 a0, 0
	v_accvgpr_write_b32 a123, 0
	v_accvgpr_write_b32 a122, 0
	v_accvgpr_write_b32 a121, 0
	v_accvgpr_write_b32 a120, 0
	v_accvgpr_write_b32 a127, 0
	v_accvgpr_write_b32 a126, 0
	v_accvgpr_write_b32 a125, 0
	v_accvgpr_write_b32 a124, 0
	v_accvgpr_write_b32 a131, 0
	v_accvgpr_write_b32 a130, 0
	v_accvgpr_write_b32 a129, 0
	v_accvgpr_write_b32 a128, 0
	v_accvgpr_write_b32 a135, 0
	v_accvgpr_write_b32 a134, 0
	v_accvgpr_write_b32 a133, 0
	v_accvgpr_write_b32 a132, 0
	v_accvgpr_write_b32 a139, 0
	v_accvgpr_write_b32 a138, 0
	v_accvgpr_write_b32 a137, 0
	v_accvgpr_write_b32 a136, 0
	v_accvgpr_write_b32 a143, 0
	v_accvgpr_write_b32 a142, 0
	v_accvgpr_write_b32 a141, 0
	v_accvgpr_write_b32 a140, 0
	v_accvgpr_write_b32 a147, 0
	v_accvgpr_write_b32 a146, 0
	v_accvgpr_write_b32 a145, 0
	v_accvgpr_write_b32 a144, 0
	v_accvgpr_write_b32 a151, 0
	v_accvgpr_write_b32 a150, 0
	v_accvgpr_write_b32 a149, 0
	v_accvgpr_write_b32 a148, 0
	v_accvgpr_write_b32 a155, 0
	v_accvgpr_write_b32 a154, 0
	v_accvgpr_write_b32 a153, 0
	v_accvgpr_write_b32 a152, 0
	v_accvgpr_write_b32 a159, 0
	v_accvgpr_write_b32 a158, 0
	v_accvgpr_write_b32 a157, 0
	v_accvgpr_write_b32 a156, 0
	v_accvgpr_write_b32 a163, 0
	v_accvgpr_write_b32 a162, 0
	v_accvgpr_write_b32 a161, 0
	v_accvgpr_write_b32 a160, 0
	v_accvgpr_write_b32 a171, 0
	v_accvgpr_write_b32 a170, 0
	v_accvgpr_write_b32 a169, 0
	v_accvgpr_write_b32 a168, 0
	v_accvgpr_write_b32 a175, 0
	v_accvgpr_write_b32 a174, 0
	v_accvgpr_write_b32 a173, 0
	v_accvgpr_write_b32 a172, 0
	v_accvgpr_write_b32 a179, 0
	v_accvgpr_write_b32 a178, 0
	v_accvgpr_write_b32 a177, 0
	v_accvgpr_write_b32 a176, 0
	v_accvgpr_write_b32 a183, 0
	v_accvgpr_write_b32 a182, 0
	v_accvgpr_write_b32 a181, 0
	v_accvgpr_write_b32 a180, 0
	v_accvgpr_write_b32 a187, 0
	v_accvgpr_write_b32 a186, 0
	v_accvgpr_write_b32 a185, 0
	v_accvgpr_write_b32 a184, 0
	v_accvgpr_write_b32 a55, 0
	v_accvgpr_write_b32 a54, 0
	v_accvgpr_write_b32 a53, 0
	v_accvgpr_write_b32 a52, 0
	v_accvgpr_write_b32 a59, 0
	v_accvgpr_write_b32 a58, 0
	v_accvgpr_write_b32 a57, 0
	v_accvgpr_write_b32 a56, 0
	v_accvgpr_write_b32 a63, 0
	v_accvgpr_write_b32 a62, 0
	v_accvgpr_write_b32 a61, 0
	v_accvgpr_write_b32 a60, 0
	v_accvgpr_write_b32 a67, 0
	v_accvgpr_write_b32 a66, 0
	v_accvgpr_write_b32 a65, 0
	v_accvgpr_write_b32 a64, 0
	v_accvgpr_write_b32 a71, 0
	v_accvgpr_write_b32 a70, 0
	v_accvgpr_write_b32 a69, 0
	v_accvgpr_write_b32 a68, 0
	v_accvgpr_write_b32 a75, 0
	v_accvgpr_write_b32 a74, 0
	v_accvgpr_write_b32 a73, 0
	v_accvgpr_write_b32 a72, 0
	v_accvgpr_write_b32 a83, 0
	v_accvgpr_write_b32 a82, 0
	v_accvgpr_write_b32 a81, 0
	v_accvgpr_write_b32 a80, 0
	v_accvgpr_write_b32 a87, 0
	v_accvgpr_write_b32 a86, 0
	v_accvgpr_write_b32 a85, 0
	v_accvgpr_write_b32 a84, 0
	v_accvgpr_write_b32 a91, 0
	v_accvgpr_write_b32 a90, 0
	v_accvgpr_write_b32 a89, 0
	v_accvgpr_write_b32 a88, 0
	v_accvgpr_write_b32 a95, 0
	v_accvgpr_write_b32 a94, 0
	v_accvgpr_write_b32 a93, 0
	v_accvgpr_write_b32 a92, 0
	v_accvgpr_write_b32 a99, 0
	v_accvgpr_write_b32 a98, 0
	v_accvgpr_write_b32 a97, 0
	v_accvgpr_write_b32 a96, 0
	v_accvgpr_write_b32 a103, 0
	v_accvgpr_write_b32 a102, 0
	v_accvgpr_write_b32 a101, 0
	v_accvgpr_write_b32 a100, 0
	v_accvgpr_write_b32 a107, 0
	v_accvgpr_write_b32 a106, 0
	v_accvgpr_write_b32 a105, 0
	v_accvgpr_write_b32 a104, 0
	v_accvgpr_write_b32 a111, 0
	v_accvgpr_write_b32 a110, 0
	v_accvgpr_write_b32 a109, 0
	v_accvgpr_write_b32 a108, 0
	v_accvgpr_write_b32 a115, 0
	v_accvgpr_write_b32 a114, 0
	v_accvgpr_write_b32 a113, 0
	v_accvgpr_write_b32 a112, 0
	v_accvgpr_write_b32 a119, 0
	v_accvgpr_write_b32 a118, 0
	v_accvgpr_write_b32 a117, 0
	v_accvgpr_write_b32 a116, 0
	v_accvgpr_write_b32 a15, 0
	v_accvgpr_write_b32 a14, 0
	v_accvgpr_write_b32 a13, 0
	v_accvgpr_write_b32 a12, 0
	v_accvgpr_write_b32 a19, 0
	v_accvgpr_write_b32 a18, 0
	v_accvgpr_write_b32 a17, 0
	v_accvgpr_write_b32 a16, 0
	v_accvgpr_write_b32 a23, 0
	v_accvgpr_write_b32 a22, 0
	v_accvgpr_write_b32 a21, 0
	v_accvgpr_write_b32 a20, 0
	v_accvgpr_write_b32 a27, 0
	v_accvgpr_write_b32 a26, 0
	v_accvgpr_write_b32 a25, 0
	v_accvgpr_write_b32 a24, 0
	v_accvgpr_write_b32 a31, 0
	v_accvgpr_write_b32 a30, 0
	v_accvgpr_write_b32 a29, 0
	v_accvgpr_write_b32 a28, 0
	v_accvgpr_write_b32 a35, 0
	v_accvgpr_write_b32 a34, 0
	v_accvgpr_write_b32 a33, 0
	v_accvgpr_write_b32 a32, 0
	v_accvgpr_write_b32 a39, 0
	v_accvgpr_write_b32 a38, 0
	v_accvgpr_write_b32 a37, 0
	v_accvgpr_write_b32 a36, 0
	v_accvgpr_write_b32 a43, 0
	v_accvgpr_write_b32 a42, 0
	v_accvgpr_write_b32 a41, 0
	v_accvgpr_write_b32 a40, 0
	v_accvgpr_write_b32 a47, 0
	v_accvgpr_write_b32 a46, 0
	v_accvgpr_write_b32 a45, 0
	v_accvgpr_write_b32 a44, 0
	v_accvgpr_write_b32 a51, 0
	v_accvgpr_write_b32 a50, 0
	v_accvgpr_write_b32 a49, 0
	v_accvgpr_write_b32 a48, 0
	v_accvgpr_write_b32 a79, 0
	v_accvgpr_write_b32 a78, 0
	v_accvgpr_write_b32 a77, 0
	v_accvgpr_write_b32 a76, 0
	v_accvgpr_write_b32 a167, 0
	v_accvgpr_write_b32 a166, 0
	v_accvgpr_write_b32 a165, 0
	v_accvgpr_write_b32 a164, 0
	v_accvgpr_write_b32 a219, 0
	v_accvgpr_write_b32 a218, 0
	v_accvgpr_write_b32 a217, 0
	v_accvgpr_write_b32 a216, 0
	v_accvgpr_write_b32 a247, 0
	v_accvgpr_write_b32 a246, 0
	v_accvgpr_write_b32 a245, 0
	v_accvgpr_write_b32 a244, 0
	v_accvgpr_write_b32 a251, 0
	v_readfirstlane_b32 s13, v75
	s_mov_b32 s20, s77
	v_readfirstlane_b32 s17, v1
	v_readfirstlane_b32 s21, v77
	v_accvgpr_write_b32 a250, 0
	v_accvgpr_write_b32 a249, 0
	v_accvgpr_write_b32 a248, 0
	v_accvgpr_write_b32 a255, 0
	v_accvgpr_write_b32 a254, 0
	v_accvgpr_write_b32 a253, 0
	v_accvgpr_write_b32 a252, 0
	s_mov_b32 s76, 0
	;;#ASMSTART
	buffer_load_dwordx2 v[74:75], v110, s[12:15], s76 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[78:79], v110, s[16:19], s76 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[76:77], v110, s[20:23], s76 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[80:81], v110, s[0:3], s76 offen
	;;#ASMEND
.LBB0_7:                                ; =>This Inner Loop Header: Depth=1
	s_add_u32 s76, s46, s36
	s_add_u32 s80, s76, 0x100
	s_mov_b32 m0, s38
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15],  v[2:5], v[34:37], a[12:15],  v80, v78 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[140:143], v101 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19],  v[2:5], v[38:41], a[16:19],  v80, v78 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[144:147], v101 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23],  v[2:5], v[42:45], a[20:23],  v80, v79 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[148:151], v101 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27],  v[2:5], v[46:49], a[24:27],  v80, v79 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[152:155], v101 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15],  v[18:21], v[50:53], a[12:15],  v80, v78 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[156:159], v100 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19],  v[18:21], v[54:57], a[16:19],  v80, v78 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[160:163], v100 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23],  v[18:21], v[58:61], a[20:23],  v80, v79 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[164:167], v100 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27],  v[18:21], v[62:65], a[24:27],  v80, v79 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[168:171], v100 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[6:9], v[34:37], a[28:31],  v80, v78 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[6:9], v[38:41], a[32:35],  v80, v78 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[6:9], v[42:45], a[36:39],  v80, v79 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[6:9], v[46:49], a[40:43],  v80, v79 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[22:25], v[50:53], a[28:31],  v80, v78 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[22:25], v[54:57], a[32:35],  v80, v78 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[22:25], v[58:61], a[36:39],  v80, v79 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[22:25], v[62:65], a[40:43],  v80, v79 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[10:13], v[34:37], a[44:47],  v81, v78 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[10:13], v[38:41], a[48:51],  v81, v78 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[10:13], v[42:45], a[76:79], v81, v79 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[10:13], v[46:49], a[164:167], v81, v79 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[26:29], v[50:53], a[44:47],  v81, v78 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[26:29], v[54:57], a[48:51],  v81, v78 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[26:29], v[58:61], a[76:79], v81, v79 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[26:29], v[62:65], a[164:167], v81, v79 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[14:17], v[34:37], a[216:219], v81, v78 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[14:17], v[38:41], a[244:247], v81, v78 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[14:17], v[42:45], a[248:251], v81, v79 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[14:17], v[46:49], a[252:255], v81, v79 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[30:33], v[50:53], a[216:219], v81, v78 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[30:33], v[54:57], a[244:247], v81, v78 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[30:33], v[58:61], a[248:251], v81, v79 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[30:33], v[62:65], a[252:255], v81, v79 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[2:5], v[140:143], a[52:55], v80, v74 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[90:93], v97 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[2:5], v[144:147], a[56:59], v80, v74 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[112:115], v97 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[2:5], v[148:151], a[60:63], v80, v75 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[116:119], v97 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[2:5], v[152:155], a[64:67], v80, v75 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[120:123], v97 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[18:21], v[156:159], a[52:55], v80, v74 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[124:127], v96 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[18:21], v[160:163], a[56:59], v80, v74 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[128:131], v96 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[18:21], v[164:167], a[60:63], v80, v75 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[132:135], v96 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[18:21], v[168:171], a[64:67], v80, v75 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[136:139], v96 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[6:9], v[140:143], a[68:71], v80, v74 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[6:9], v[144:147], a[72:75], v80, v74 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[6:9], v[148:151], a[80:83], v80, v75 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[6:9], v[152:155], a[84:87], v80, v75 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[22:25], v[156:159], a[68:71], v80, v74 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[22:25], v[160:163], a[72:75], v80, v74 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[22:25], v[164:167], a[80:83], v80, v75 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[22:25], v[168:171], a[84:87], v80, v75 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[10:13], v[140:143], a[88:91], v81, v74 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[10:13], v[144:147], a[92:95], v81, v74 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[10:13], v[148:151], a[96:99], v81, v75 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[10:13], v[152:155], a[100:103], v81, v75 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[26:29], v[156:159], a[88:91], v81, v74 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[26:29], v[160:163], a[92:95], v81, v74 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[26:29], v[164:167], a[96:99], v81, v75 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[26:29], v[168:171], a[100:103], v81, v75 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[14:17], v[140:143], a[104:107], v81, v74 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[14:17], v[144:147], a[108:111], v81, v74 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[14:17], v[148:151], a[112:115], v81, v75 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[14:17], v[152:155], a[116:119], v81, v75 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[30:33], v[156:159], a[104:107], v81, v74 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[30:33], v[160:163], a[108:111], v81, v74 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[30:33], v[164:167], a[112:115], v81, v75 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[30:33], v[168:171], a[116:119], v81, v75 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(16)
s_barrier

	;;#ASMEND
	s_add_u32 s77, s49, s36
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123],  v[90:93], v[34:37], a[120:123],  v76, v78 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[2:5], v106 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127],  v[90:93], v[38:41], a[124:127],  v76, v78 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[6:9], v106 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[90:93], v[42:45], a[128:131],  v76, v79 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[10:13], v106 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[90:93], v[46:49], a[132:135],  v76, v79 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[14:17], v106 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123],  v[124:127], v[50:53], a[120:123],  v76, v78 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[18:21], v107 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127],  v[124:127], v[54:57], a[124:127],  v76, v78 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[22:25], v107 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[124:127], v[58:61], a[128:131],  v76, v79 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[26:29], v107 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[124:127], v[62:65], a[132:135],  v76, v79 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v107 offset:6144

	;;#ASMEND
	buffer_load_dwordx4 v66, s[4:7], s80 offen lds
	s_mov_b32 m0, s31
	s_add_u32 s81, s77, 0x100
	buffer_load_dwordx4 v67, s[4:7], s80 offen lds
	s_mov_b32 m0, s47
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[112:115], v[34:37], a[136:139],  v76, v78 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[112:115], v[38:41], a[140:143],  v76, v78 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[112:115], v[42:45], a[144:147],  v76, v79 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[112:115], v[46:49], a[148:151],  v76, v79 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[128:131], v[50:53], a[136:139],  v76, v78 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[128:131], v[54:57], a[140:143],  v76, v78 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[128:131], v[58:61], a[144:147],  v76, v79 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[128:131], v[62:65], a[148:151],  v76, v79 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_u32 s78, s53, s36
	buffer_load_dwordx4 v94, s[4:7], s80 offen lds
	s_mov_b32 m0, s48
	s_add_u32 s82, s78, 0x100
	buffer_load_dwordx4 v68, s[4:7], s80 offen lds
	s_mov_b32 m0, s43
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[116:119], v[34:37], a[152:155],  v77, v78 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[116:119], v[38:41], a[156:159],  v77, v78 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[116:119], v[42:45], a[160:163], v77, v79 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[116:119], v[46:49], a[168:171], v77, v79 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[132:135], v[50:53], a[152:155],  v77, v78 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[132:135], v[54:57], a[156:159],  v77, v78 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[132:135], v[58:61], a[160:163], v77, v79 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[132:135], v[62:65], a[168:171], v77, v79 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_u32 s79, s56, s36
	buffer_load_dwordx4 v66, s[4:7], s81 offen lds
	s_mov_b32 m0, s50
	s_add_u32 s83, s79, 0x100
	buffer_load_dwordx4 v67, s[4:7], s81 offen lds
	s_mov_b32 m0, s51
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[120:123], v[34:37], a[172:175], v77, v78 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[120:123], v[38:41], a[176:179], v77, v78 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[120:123], v[42:45], a[180:183], v77, v79 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[120:123], v[46:49], a[184:187], v77, v79 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[136:139], v[50:53], a[172:175], v77, v78 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[136:139], v[54:57], a[176:179], v77, v78 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[136:139], v[58:61], a[180:183], v77, v79 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[136:139], v[62:65], a[184:187], v77, v79 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_i32 s84, s33, 0xfffff200
	buffer_load_dwordx4 v94, s[4:7], s81 offen lds
	s_mov_b32 m0, s52
	;;#ASMSTART
	buffer_load_dwordx2 v[88:89], v110, s[0:3], s84 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[82:83], v110, s[20:23], s84 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[86:87], v110, s[16:19], s84 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[84:85], v110, s[12:15], s84 offen
	;;#ASMEND
	s_add_u32 s80, s79, 0x180
	buffer_load_dwordx4 v68, s[4:7], s81 offen lds
	s_mov_b32 m0, s41
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191],  v[90:93], v[140:143], a[188:191],  v76, v74 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v103 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[90:93], v[144:147], a[192:195],  v76, v74 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v103 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[90:93], v[148:151], a[196:199],  v76, v75 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v103 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[90:93], v[152:155], a[200:203],  v76, v75 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v103 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191],  v[124:127], v[156:159], a[188:191],  v76, v74 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v102 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[124:127], v[160:163], a[192:195],  v76, v74 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v102 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[124:127], v[164:167], a[196:199],  v76, v75 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v102 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[124:127], v[168:171], a[200:203],  v76, v75 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[62:65], v102 offset:6144

	;;#ASMEND
	s_add_u32 s81, s78, 0x180
	buffer_load_dwordx4 v70, s[8:11], s82 offen lds
	s_mov_b32 m0, s35
	s_add_i32 s84, s33, 0xfffff400
	buffer_load_dwordx4 v69, s[8:11], s82 offen lds
	s_mov_b32 m0, s54
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[112:115], v[140:143], a[204:207],  v76, v74 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[112:115], v[144:147], a[208:211],  v76, v74 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[112:115], v[148:151], a[212:215],  v76, v75 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[112:115], v[152:155], a[220:223],  v76, v75 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[128:131], v[156:159], a[204:207],  v76, v74 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[128:131], v[160:163], a[208:211],  v76, v74 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[128:131], v[164:167], a[212:215],  v76, v75 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[128:131], v[168:171], a[220:223],  v76, v75 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[78:79], v110, s[12:15], s84 offen
	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v71, s[8:11], s82 offen lds
	s_mov_b32 m0, s55
	s_nop 0
	buffer_load_dwordx4 v72, s[8:11], s82 offen lds
	s_mov_b32 m0, s39
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[116:119], v[140:143], a[224:227],  v77, v74 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[116:119], v[144:147], a[228:231],  v77, v74 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[116:119], v[148:151], a[232:235], v77, v75 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[116:119], v[152:155], a[236:239], v77, v75 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[132:135], v[156:159], a[224:227],  v77, v74 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[132:135], v[160:163], a[228:231],  v77, v74 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[132:135], v[164:167], a[232:235], v77, v75 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[132:135], v[168:171], a[236:239], v77, v75 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_u32 s82, s77, 0x180
	buffer_load_dwordx4 v70, s[8:11], s83 offen lds
	s_mov_b32 m0, s57
	s_nop 0
	buffer_load_dwordx4 v69, s[8:11], s83 offen lds
	s_mov_b32 m0, s58
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[120:123], v[140:143], a[240:243], v77, v74 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[120:123], v[144:147], a[8:11], v77, v74 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[120:123], v[148:151], a[4:7], v77, v75 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[120:123], v[152:155], a[0:3], v77, v75 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[136:139], v[156:159], a[240:243], v77, v74 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[136:139], v[160:163], a[8:11], v77, v74 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[136:139], v[164:167], a[4:7], v77, v75 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[136:139], v[168:171], a[0:3], v77, v75 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[92:93], v110, s[0:3], s84 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[74:75], v110, s[20:23], s84 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[90:91], v110, s[16:19], s84 offen
	;;#ASMEND
	s_add_i32 s84, s33, 0xfffff600
	buffer_load_dwordx4 v71, s[8:11], s83 offen lds
	s_mov_b32 m0, s59
	;;#ASMSTART
	buffer_load_dwordx2 v[76:77], v110, s[20:23], s84 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[80:81], v110, s[12:15], s84 offen
	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v72, s[8:11], s83 offen lds
	s_add_u32 s83, s76, 0x180
	s_mov_b32 m0, s45
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15],  v[2:5], v[34:37], a[12:15],  v88, v86 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[144:147], v99 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19],  v[2:5], v[38:41], a[16:19],  v88, v86 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[148:151], v99 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23],  v[2:5], v[42:45], a[20:23],  v88, v87 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[152:155], v99 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27],  v[2:5], v[46:49], a[24:27],  v88, v87 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[156:159], v99 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15],  v[18:21], v[50:53], a[12:15],  v88, v86 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[160:163], v98 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19],  v[18:21], v[54:57], a[16:19],  v88, v86 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[164:167], v98 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23],  v[18:21], v[58:61], a[20:23],  v88, v87 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[168:171], v98 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27],  v[18:21], v[62:65], a[24:27],  v88, v87 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[172:175], v98 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[6:9], v[34:37], a[28:31],  v88, v86 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[6:9], v[38:41], a[32:35],  v88, v86 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[6:9], v[42:45], a[36:39],  v88, v87 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[6:9], v[46:49], a[40:43],  v88, v87 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[22:25], v[50:53], a[28:31],  v88, v86 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[22:25], v[54:57], a[32:35],  v88, v86 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[22:25], v[58:61], a[36:39],  v88, v87 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[22:25], v[62:65], a[40:43],  v88, v87 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[10:13], v[34:37], a[44:47],  v89, v86 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[10:13], v[38:41], a[48:51],  v89, v86 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[10:13], v[42:45], a[76:79], v89, v87 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[10:13], v[46:49], a[164:167], v89, v87 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[26:29], v[50:53], a[44:47],  v89, v86 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[26:29], v[54:57], a[48:51],  v89, v86 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[26:29], v[58:61], a[76:79], v89, v87 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[26:29], v[62:65], a[164:167], v89, v87 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[14:17], v[34:37], a[216:219], v89, v86 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[14:17], v[38:41], a[244:247], v89, v86 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[14:17], v[42:45], a[248:251], v89, v87 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[14:17], v[46:49], a[252:255], v89, v87 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[30:33], v[50:53], a[216:219], v89, v86 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[30:33], v[54:57], a[244:247], v89, v86 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[30:33], v[58:61], a[248:251], v89, v87 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[30:33], v[62:65], a[252:255], v89, v87 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[2:5], v[144:147], a[52:55], v88, v84 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[112:115], v95 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[2:5], v[148:151], a[56:59], v88, v84 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[116:119], v95 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[2:5], v[152:155], a[60:63], v88, v85 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[120:123], v95 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[2:5], v[156:159], a[64:67], v88, v85 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[124:127], v95 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[18:21], v[160:163], a[52:55], v88, v84 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[128:131], v73 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[18:21], v[164:167], a[56:59], v88, v84 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[132:135], v73 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[18:21], v[168:171], a[60:63], v88, v85 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[136:139], v73 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[18:21], v[172:175], a[64:67], v88, v85 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[140:143], v73 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[6:9], v[144:147], a[68:71], v88, v84 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[6:9], v[148:151], a[72:75], v88, v84 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[6:9], v[152:155], a[80:83], v88, v85 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[6:9], v[156:159], a[84:87], v88, v85 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[22:25], v[160:163], a[68:71], v88, v84 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[22:25], v[164:167], a[72:75], v88, v84 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[22:25], v[168:171], a[80:83], v88, v85 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[22:25], v[172:175], a[84:87], v88, v85 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[10:13], v[144:147], a[88:91], v89, v84 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[10:13], v[148:151], a[92:95], v89, v84 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[10:13], v[152:155], a[96:99], v89, v85 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[10:13], v[156:159], a[100:103], v89, v85 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[26:29], v[160:163], a[88:91], v89, v84 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[26:29], v[164:167], a[92:95], v89, v84 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[26:29], v[168:171], a[96:99], v89, v85 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[26:29], v[172:175], a[100:103], v89, v85 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[14:17], v[144:147], a[104:107], v89, v84 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[14:17], v[148:151], a[108:111], v89, v84 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[14:17], v[152:155], a[112:115], v89, v85 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[14:17], v[156:159], a[116:119], v89, v85 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[30:33], v[160:163], a[104:107], v89, v84 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[30:33], v[164:167], a[108:111], v89, v84 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[30:33], v[168:171], a[112:115], v89, v85 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[30:33], v[172:175], a[116:119], v89, v85 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(16)
s_barrier

	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123],  v[112:115], v[34:37], a[120:123],  v82, v86 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[2:5], v109 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127],  v[112:115], v[38:41], a[124:127],  v82, v86 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[6:9], v109 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[112:115], v[42:45], a[128:131],  v82, v87 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[10:13], v109 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[112:115], v[46:49], a[132:135],  v82, v87 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[14:17], v109 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123],  v[128:131], v[50:53], a[120:123],  v82, v86 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[18:21], v108 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127],  v[128:131], v[54:57], a[124:127],  v82, v86 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[22:25], v108 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[128:131], v[58:61], a[128:131],  v82, v87 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[26:29], v108 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[128:131], v[62:65], a[132:135],  v82, v87 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v108 offset:6144

	;;#ASMEND
	buffer_load_dwordx4 v66, s[4:7], s83 offen lds
	s_mov_b32 m0, s63
	s_nop 0
	buffer_load_dwordx4 v67, s[4:7], s83 offen lds
	s_mov_b32 m0, s65
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[116:119], v[34:37], a[136:139],  v82, v86 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[116:119], v[38:41], a[140:143],  v82, v86 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[116:119], v[42:45], a[144:147],  v82, v87 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[116:119], v[46:49], a[148:151],  v82, v87 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[132:135], v[50:53], a[136:139],  v82, v86 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[132:135], v[54:57], a[140:143],  v82, v86 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[132:135], v[58:61], a[144:147],  v82, v87 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[132:135], v[62:65], a[148:151],  v82, v87 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v94, s[4:7], s83 offen lds
	s_mov_b32 m0, s66
	s_nop 0
	buffer_load_dwordx4 v68, s[4:7], s83 offen lds
	s_mov_b32 m0, s44
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[120:123], v[34:37], a[152:155],  v83, v86 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[120:123], v[38:41], a[156:159],  v83, v86 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[120:123], v[42:45], a[160:163], v83, v87 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[120:123], v[46:49], a[168:171], v83, v87 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[136:139], v[50:53], a[152:155],  v83, v86 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[136:139], v[54:57], a[156:159],  v83, v86 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[136:139], v[58:61], a[160:163], v83, v87 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[136:139], v[62:65], a[168:171], v83, v87 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_u32 s83, s76, 0x200
	buffer_load_dwordx4 v66, s[4:7], s82 offen lds
	s_mov_b32 m0, s67
	s_nop 0
	buffer_load_dwordx4 v67, s[4:7], s82 offen lds
	s_mov_b32 m0, s68
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[124:127], v[34:37], a[172:175], v83, v86 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[124:127], v[38:41], a[176:179], v83, v86 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[124:127], v[42:45], a[180:183], v83, v87 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[124:127], v[46:49], a[184:187], v83, v87 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[140:143], v[50:53], a[172:175], v83, v86 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[140:143], v[54:57], a[176:179], v83, v86 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[140:143], v[58:61], a[180:183], v83, v87 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[140:143], v[62:65], a[184:187], v83, v87 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v94, s[4:7], s82 offen lds
	s_mov_b32 m0, s69
	s_nop 0
	buffer_load_dwordx4 v68, s[4:7], s82 offen lds
	s_mov_b32 m0, s42
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191],  v[112:115], v[144:147], a[188:191],  v82, v84 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v105 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[112:115], v[148:151], a[192:195],  v82, v84 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v105 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[112:115], v[152:155], a[196:199],  v82, v85 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v105 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[112:115], v[156:159], a[200:203],  v82, v85 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v105 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191],  v[128:131], v[160:163], a[188:191],  v82, v84 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v104 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[128:131], v[164:167], a[192:195],  v82, v84 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v104 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[128:131], v[168:171], a[196:199],  v82, v85 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v104 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[128:131], v[172:175], a[200:203],  v82, v85 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[62:65], v104 offset:6144

	;;#ASMEND
	s_add_u32 s82, s77, 0x200
	buffer_load_dwordx4 v70, s[8:11], s81 offen lds
	s_mov_b32 m0, s70
	s_nop 0
	buffer_load_dwordx4 v69, s[8:11], s81 offen lds
	s_mov_b32 m0, s71
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[116:119], v[144:147], a[204:207],  v82, v84 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[116:119], v[148:151], a[208:211],  v82, v84 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[116:119], v[152:155], a[212:215],  v82, v85 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[116:119], v[156:159], a[220:223],  v82, v85 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[132:135], v[160:163], a[204:207],  v82, v84 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[132:135], v[164:167], a[208:211],  v82, v84 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[132:135], v[168:171], a[212:215],  v82, v85 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[132:135], v[172:175], a[220:223],  v82, v85 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v71, s[8:11], s81 offen lds
	s_mov_b32 m0, s72
	s_nop 0
	buffer_load_dwordx4 v72, s[8:11], s81 offen lds
	s_mov_b32 m0, s40
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[120:123], v[144:147], a[224:227],  v83, v84 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[120:123], v[148:151], a[228:231],  v83, v84 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[120:123], v[152:155], a[232:235], v83, v85 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[120:123], v[156:159], a[236:239], v83, v85 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[136:139], v[160:163], a[224:227],  v83, v84 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[136:139], v[164:167], a[228:231],  v83, v84 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[136:139], v[168:171], a[232:235], v83, v85 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[136:139], v[172:175], a[236:239], v83, v85 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_u32 s81, s78, 0x200
	buffer_load_dwordx4 v70, s[8:11], s80 offen lds
	s_mov_b32 m0, s73
	s_nop 0
	buffer_load_dwordx4 v69, s[8:11], s80 offen lds
	s_mov_b32 m0, s74
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[124:127], v[144:147], a[240:243], v83, v84 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[124:127], v[148:151], a[8:11], v83, v84 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[124:127], v[152:155], a[4:7], v83, v85 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[124:127], v[156:159], a[0:3], v83, v85 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[140:143], v[160:163], a[240:243], v83, v84 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[140:143], v[164:167], a[8:11], v83, v84 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[140:143], v[168:171], a[4:7], v83, v85 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[140:143], v[172:175], a[0:3], v83, v85 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[84:85], v110, s[0:3], s84 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[82:83], v110, s[16:19], s84 offen
	;;#ASMEND
	s_add_i32 s84, s33, 0xfffff800
	buffer_load_dwordx4 v71, s[8:11], s80 offen lds
	s_mov_b32 m0, s75
	s_nop 0
	buffer_load_dwordx4 v72, s[8:11], s80 offen lds
	s_mov_b32 m0, s38
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15],  v[2:5], v[34:37], a[12:15],  v92, v90 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[140:143], v101 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19],  v[2:5], v[38:41], a[16:19],  v92, v90 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[144:147], v101 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23],  v[2:5], v[42:45], a[20:23],  v92, v91 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[148:151], v101 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27],  v[2:5], v[46:49], a[24:27],  v92, v91 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[152:155], v101 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15],  v[18:21], v[50:53], a[12:15],  v92, v90 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[156:159], v100 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19],  v[18:21], v[54:57], a[16:19],  v92, v90 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[160:163], v100 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23],  v[18:21], v[58:61], a[20:23],  v92, v91 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[164:167], v100 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27],  v[18:21], v[62:65], a[24:27],  v92, v91 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[168:171], v100 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[6:9], v[34:37], a[28:31],  v92, v90 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[6:9], v[38:41], a[32:35],  v92, v90 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[6:9], v[42:45], a[36:39],  v92, v91 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[6:9], v[46:49], a[40:43],  v92, v91 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[22:25], v[50:53], a[28:31],  v92, v90 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[22:25], v[54:57], a[32:35],  v92, v90 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[22:25], v[58:61], a[36:39],  v92, v91 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[22:25], v[62:65], a[40:43],  v92, v91 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[10:13], v[34:37], a[44:47],  v93, v90 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[10:13], v[38:41], a[48:51],  v93, v90 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[10:13], v[42:45], a[76:79], v93, v91 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[10:13], v[46:49], a[164:167], v93, v91 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[26:29], v[50:53], a[44:47],  v93, v90 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[26:29], v[54:57], a[48:51],  v93, v90 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[26:29], v[58:61], a[76:79], v93, v91 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[26:29], v[62:65], a[164:167], v93, v91 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[14:17], v[34:37], a[216:219], v93, v90 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[14:17], v[38:41], a[244:247], v93, v90 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[14:17], v[42:45], a[248:251], v93, v91 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[14:17], v[46:49], a[252:255], v93, v91 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[30:33], v[50:53], a[216:219], v93, v90 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[30:33], v[54:57], a[244:247], v93, v90 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[30:33], v[58:61], a[248:251], v93, v91 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[30:33], v[62:65], a[252:255], v93, v91 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[2:5], v[140:143], a[52:55], v92, v78 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[86:89], v97 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[2:5], v[144:147], a[56:59], v92, v78 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[112:115], v97 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[2:5], v[148:151], a[60:63], v92, v79 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[116:119], v97 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[2:5], v[152:155], a[64:67], v92, v79 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[120:123], v97 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[18:21], v[156:159], a[52:55], v92, v78 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[124:127], v96 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[18:21], v[160:163], a[56:59], v92, v78 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[128:131], v96 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[18:21], v[164:167], a[60:63], v92, v79 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[132:135], v96 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[18:21], v[168:171], a[64:67], v92, v79 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[136:139], v96 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[6:9], v[140:143], a[68:71], v92, v78 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[6:9], v[144:147], a[72:75], v92, v78 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[6:9], v[148:151], a[80:83], v92, v79 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[6:9], v[152:155], a[84:87], v92, v79 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[22:25], v[156:159], a[68:71], v92, v78 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[22:25], v[160:163], a[72:75], v92, v78 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[22:25], v[164:167], a[80:83], v92, v79 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[22:25], v[168:171], a[84:87], v92, v79 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[10:13], v[140:143], a[88:91], v93, v78 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[10:13], v[144:147], a[92:95], v93, v78 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[10:13], v[148:151], a[96:99], v93, v79 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[10:13], v[152:155], a[100:103], v93, v79 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[26:29], v[156:159], a[88:91], v93, v78 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[26:29], v[160:163], a[92:95], v93, v78 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[26:29], v[164:167], a[96:99], v93, v79 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[26:29], v[168:171], a[100:103], v93, v79 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[14:17], v[140:143], a[104:107], v93, v78 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[14:17], v[144:147], a[108:111], v93, v78 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[14:17], v[148:151], a[112:115], v93, v79 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[14:17], v[152:155], a[116:119], v93, v79 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[30:33], v[156:159], a[104:107], v93, v78 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[30:33], v[160:163], a[108:111], v93, v78 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[30:33], v[164:167], a[112:115], v93, v79 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[30:33], v[168:171], a[116:119], v93, v79 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(16)
s_barrier

	;;#ASMEND
	s_add_u32 s80, s79, 0x200
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123],  v[86:89], v[34:37], a[120:123],  v74, v90 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[2:5], v106 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127],  v[86:89], v[38:41], a[124:127],  v74, v90 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[6:9], v106 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[86:89], v[42:45], a[128:131],  v74, v91 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[10:13], v106 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[86:89], v[46:49], a[132:135],  v74, v91 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[14:17], v106 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123],  v[124:127], v[50:53], a[120:123],  v74, v90 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[18:21], v107 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127],  v[124:127], v[54:57], a[124:127],  v74, v90 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[22:25], v107 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[124:127], v[58:61], a[128:131],  v74, v91 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[26:29], v107 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[124:127], v[62:65], a[132:135],  v74, v91 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v107 offset:6144

	;;#ASMEND
	buffer_load_dwordx4 v66, s[4:7], s83 offen lds
	s_mov_b32 m0, s31
	;;#ASMSTART
	buffer_load_dwordx2 v[92:93], v110, s[0:3], s84 offen
	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v67, s[4:7], s83 offen lds
	s_mov_b32 m0, s47
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[112:115], v[34:37], a[136:139],  v74, v90 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[112:115], v[38:41], a[140:143],  v74, v90 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[112:115], v[42:45], a[144:147],  v74, v91 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[112:115], v[46:49], a[148:151],  v74, v91 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[128:131], v[50:53], a[136:139],  v74, v90 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[128:131], v[54:57], a[140:143],  v74, v90 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[128:131], v[58:61], a[144:147],  v74, v91 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[128:131], v[62:65], a[148:151],  v74, v91 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v94, s[4:7], s83 offen lds
	s_mov_b32 m0, s48
	s_nop 0
	buffer_load_dwordx4 v68, s[4:7], s83 offen lds
	s_mov_b32 m0, s43
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[116:119], v[34:37], a[152:155],  v75, v90 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[116:119], v[38:41], a[156:159],  v75, v90 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[116:119], v[42:45], a[160:163], v75, v91 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[116:119], v[46:49], a[168:171], v75, v91 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[132:135], v[50:53], a[152:155],  v75, v90 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[132:135], v[54:57], a[156:159],  v75, v90 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[132:135], v[58:61], a[160:163], v75, v91 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[132:135], v[62:65], a[168:171], v75, v91 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_u32 s83, s76, 0x280
	buffer_load_dwordx4 v66, s[4:7], s82 offen lds
	s_mov_b32 m0, s50
	s_nop 0
	buffer_load_dwordx4 v67, s[4:7], s82 offen lds
	s_mov_b32 m0, s51
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[120:123], v[34:37], a[172:175], v75, v90 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[120:123], v[38:41], a[176:179], v75, v90 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[120:123], v[42:45], a[180:183], v75, v91 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[120:123], v[46:49], a[184:187], v75, v91 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[136:139], v[50:53], a[172:175], v75, v90 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[136:139], v[54:57], a[176:179], v75, v90 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[136:139], v[58:61], a[180:183], v75, v91 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[136:139], v[62:65], a[184:187], v75, v91 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v94, s[4:7], s82 offen lds
	s_mov_b32 m0, s52
	s_nop 0
	buffer_load_dwordx4 v68, s[4:7], s82 offen lds
	s_mov_b32 m0, s41
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191],  v[86:89], v[140:143], a[188:191],  v74, v78 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v103 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[86:89], v[144:147], a[192:195],  v74, v78 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v103 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[86:89], v[148:151], a[196:199],  v74, v79 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v103 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[86:89], v[152:155], a[200:203],  v74, v79 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v103 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191],  v[124:127], v[156:159], a[188:191],  v74, v78 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v102 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[124:127], v[160:163], a[192:195],  v74, v78 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v102 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[124:127], v[164:167], a[196:199],  v74, v79 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v102 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[124:127], v[168:171], a[200:203],  v74, v79 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[62:65], v102 offset:6144

	;;#ASMEND
	s_add_u32 s82, s77, 0x280
	buffer_load_dwordx4 v70, s[8:11], s81 offen lds
	s_mov_b32 m0, s35
	s_nop 0
	buffer_load_dwordx4 v69, s[8:11], s81 offen lds
	s_mov_b32 m0, s54
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[112:115], v[140:143], a[204:207],  v74, v78 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[112:115], v[144:147], a[208:211],  v74, v78 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[112:115], v[148:151], a[212:215],  v74, v79 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[112:115], v[152:155], a[220:223],  v74, v79 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[128:131], v[156:159], a[204:207],  v74, v78 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[128:131], v[160:163], a[208:211],  v74, v78 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[128:131], v[164:167], a[212:215],  v74, v79 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[128:131], v[168:171], a[220:223],  v74, v79 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v71, s[8:11], s81 offen lds
	s_mov_b32 m0, s55
	s_nop 0
	buffer_load_dwordx4 v72, s[8:11], s81 offen lds
	s_mov_b32 m0, s39
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[116:119], v[140:143], a[224:227],  v75, v78 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[116:119], v[144:147], a[228:231],  v75, v78 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[116:119], v[148:151], a[232:235], v75, v79 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[116:119], v[152:155], a[236:239], v75, v79 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[132:135], v[156:159], a[224:227],  v75, v78 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[132:135], v[160:163], a[228:231],  v75, v78 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[132:135], v[164:167], a[232:235], v75, v79 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[132:135], v[168:171], a[236:239], v75, v79 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_u32 s81, s78, 0x280
	buffer_load_dwordx4 v70, s[8:11], s80 offen lds
	s_mov_b32 m0, s57
	s_nop 0
	buffer_load_dwordx4 v69, s[8:11], s80 offen lds
	s_mov_b32 m0, s58
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[120:123], v[140:143], a[240:243], v75, v78 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[120:123], v[144:147], a[8:11], v75, v78 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[120:123], v[148:151], a[4:7], v75, v79 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[120:123], v[152:155], a[0:3], v75, v79 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[136:139], v[156:159], a[240:243], v75, v78 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[136:139], v[160:163], a[8:11], v75, v78 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[136:139], v[164:167], a[4:7], v75, v79 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[136:139], v[168:171], a[0:3], v75, v79 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[74:75], v110, s[20:23], s84 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[88:89], v110, s[16:19], s84 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[78:79], v110, s[12:15], s84 offen
	;;#ASMEND
	s_add_i32 s84, s33, 0xfffffa00
	buffer_load_dwordx4 v71, s[8:11], s80 offen lds
	s_mov_b32 m0, s59
	;;#ASMSTART
	buffer_load_dwordx2 v[90:91], v110, s[0:3], s84 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[86:87], v110, s[16:19], s84 offen
	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v72, s[8:11], s80 offen lds
	s_mov_b32 m0, s45
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15],  v[2:5], v[34:37], a[12:15],  v84, v82 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[144:147], v99 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19],  v[2:5], v[38:41], a[16:19],  v84, v82 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[148:151], v99 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23],  v[2:5], v[42:45], a[20:23],  v84, v83 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[152:155], v99 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27],  v[2:5], v[46:49], a[24:27],  v84, v83 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[156:159], v99 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15],  v[18:21], v[50:53], a[12:15],  v84, v82 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[160:163], v98 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19],  v[18:21], v[54:57], a[16:19],  v84, v82 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[164:167], v98 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23],  v[18:21], v[58:61], a[20:23],  v84, v83 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[168:171], v98 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27],  v[18:21], v[62:65], a[24:27],  v84, v83 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[172:175], v98 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[6:9], v[34:37], a[28:31],  v84, v82 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[6:9], v[38:41], a[32:35],  v84, v82 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[6:9], v[42:45], a[36:39],  v84, v83 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[6:9], v[46:49], a[40:43],  v84, v83 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[22:25], v[50:53], a[28:31],  v84, v82 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[22:25], v[54:57], a[32:35],  v84, v82 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[22:25], v[58:61], a[36:39],  v84, v83 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[22:25], v[62:65], a[40:43],  v84, v83 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[10:13], v[34:37], a[44:47],  v85, v82 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[10:13], v[38:41], a[48:51],  v85, v82 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[10:13], v[42:45], a[76:79], v85, v83 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[10:13], v[46:49], a[164:167], v85, v83 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[26:29], v[50:53], a[44:47],  v85, v82 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[26:29], v[54:57], a[48:51],  v85, v82 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[26:29], v[58:61], a[76:79], v85, v83 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[26:29], v[62:65], a[164:167], v85, v83 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[14:17], v[34:37], a[216:219], v85, v82 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[14:17], v[38:41], a[244:247], v85, v82 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[14:17], v[42:45], a[248:251], v85, v83 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[14:17], v[46:49], a[252:255], v85, v83 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[30:33], v[50:53], a[216:219], v85, v82 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[30:33], v[54:57], a[244:247], v85, v82 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[30:33], v[58:61], a[248:251], v85, v83 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[30:33], v[62:65], a[252:255], v85, v83 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[2:5], v[144:147], a[52:55], v84, v80 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[112:115], v95 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[2:5], v[148:151], a[56:59], v84, v80 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[116:119], v95 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[2:5], v[152:155], a[60:63], v84, v81 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[120:123], v95 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[2:5], v[156:159], a[64:67], v84, v81 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[124:127], v95 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[18:21], v[160:163], a[52:55], v84, v80 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[128:131], v73 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[18:21], v[164:167], a[56:59], v84, v80 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[132:135], v73 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[18:21], v[168:171], a[60:63], v84, v81 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[136:139], v73 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[18:21], v[172:175], a[64:67], v84, v81 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[140:143], v73 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[6:9], v[144:147], a[68:71], v84, v80 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[6:9], v[148:151], a[72:75], v84, v80 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[6:9], v[152:155], a[80:83], v84, v81 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[6:9], v[156:159], a[84:87], v84, v81 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[22:25], v[160:163], a[68:71], v84, v80 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[22:25], v[164:167], a[72:75], v84, v80 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[22:25], v[168:171], a[80:83], v84, v81 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[22:25], v[172:175], a[84:87], v84, v81 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[10:13], v[144:147], a[88:91], v85, v80 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[10:13], v[148:151], a[92:95], v85, v80 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[10:13], v[152:155], a[96:99], v85, v81 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[10:13], v[156:159], a[100:103], v85, v81 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[26:29], v[160:163], a[88:91], v85, v80 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[26:29], v[164:167], a[92:95], v85, v80 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[26:29], v[168:171], a[96:99], v85, v81 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[26:29], v[172:175], a[100:103], v85, v81 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[14:17], v[144:147], a[104:107], v85, v80 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[14:17], v[148:151], a[108:111], v85, v80 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[14:17], v[152:155], a[112:115], v85, v81 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[14:17], v[156:159], a[116:119], v85, v81 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[30:33], v[160:163], a[104:107], v85, v80 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[30:33], v[164:167], a[108:111], v85, v80 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[30:33], v[168:171], a[112:115], v85, v81 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[30:33], v[172:175], a[116:119], v85, v81 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(16)
s_barrier

	;;#ASMEND
	s_add_u32 s80, s79, 0x280
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123],  v[112:115], v[34:37], a[120:123],  v76, v82 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[2:5], v109 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127],  v[112:115], v[38:41], a[124:127],  v76, v82 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[6:9], v109 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[112:115], v[42:45], a[128:131],  v76, v83 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[10:13], v109 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[112:115], v[46:49], a[132:135],  v76, v83 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[14:17], v109 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123],  v[128:131], v[50:53], a[120:123],  v76, v82 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[18:21], v108 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127],  v[128:131], v[54:57], a[124:127],  v76, v82 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[22:25], v108 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[128:131], v[58:61], a[128:131],  v76, v83 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[26:29], v108 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[128:131], v[62:65], a[132:135],  v76, v83 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v108 offset:6144

	;;#ASMEND
	buffer_load_dwordx4 v66, s[4:7], s83 offen lds
	s_mov_b32 m0, s63
	;;#ASMSTART
	buffer_load_dwordx2 v[84:85], v110, s[12:15], s84 offen
	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v67, s[4:7], s83 offen lds
	s_mov_b32 m0, s65
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[116:119], v[34:37], a[136:139],  v76, v82 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[116:119], v[38:41], a[140:143],  v76, v82 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[116:119], v[42:45], a[144:147],  v76, v83 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[116:119], v[46:49], a[148:151],  v76, v83 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[132:135], v[50:53], a[136:139],  v76, v82 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[132:135], v[54:57], a[140:143],  v76, v82 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[132:135], v[58:61], a[144:147],  v76, v83 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[132:135], v[62:65], a[148:151],  v76, v83 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v94, s[4:7], s83 offen lds
	s_mov_b32 m0, s66
	s_nop 0
	buffer_load_dwordx4 v68, s[4:7], s83 offen lds
	s_mov_b32 m0, s44
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[120:123], v[34:37], a[152:155],  v77, v82 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[120:123], v[38:41], a[156:159],  v77, v82 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[120:123], v[42:45], a[160:163], v77, v83 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[120:123], v[46:49], a[168:171], v77, v83 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[136:139], v[50:53], a[152:155],  v77, v82 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[136:139], v[54:57], a[156:159],  v77, v82 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[136:139], v[58:61], a[160:163], v77, v83 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[136:139], v[62:65], a[168:171], v77, v83 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_u32 s83, s76, 0x300
	buffer_load_dwordx4 v66, s[4:7], s82 offen lds
	s_mov_b32 m0, s67
	s_nop 0
	buffer_load_dwordx4 v67, s[4:7], s82 offen lds
	s_mov_b32 m0, s68
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[124:127], v[34:37], a[172:175], v77, v82 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[124:127], v[38:41], a[176:179], v77, v82 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[124:127], v[42:45], a[180:183], v77, v83 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[124:127], v[46:49], a[184:187], v77, v83 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[140:143], v[50:53], a[172:175], v77, v82 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[140:143], v[54:57], a[176:179], v77, v82 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[140:143], v[58:61], a[180:183], v77, v83 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[140:143], v[62:65], a[184:187], v77, v83 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[82:83], v110, s[20:23], s84 offen
	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v94, s[4:7], s82 offen lds
	s_mov_b32 m0, s69
	s_nop 0
	buffer_load_dwordx4 v68, s[4:7], s82 offen lds
	s_mov_b32 m0, s42
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191],  v[112:115], v[144:147], a[188:191],  v76, v80 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v105 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[112:115], v[148:151], a[192:195],  v76, v80 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v105 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[112:115], v[152:155], a[196:199],  v76, v81 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v105 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[112:115], v[156:159], a[200:203],  v76, v81 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v105 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191],  v[128:131], v[160:163], a[188:191],  v76, v80 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v104 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[128:131], v[164:167], a[192:195],  v76, v80 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v104 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[128:131], v[168:171], a[196:199],  v76, v81 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v104 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[128:131], v[172:175], a[200:203],  v76, v81 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[62:65], v104 offset:6144

	;;#ASMEND
	s_add_u32 s82, s77, 0x300
	buffer_load_dwordx4 v70, s[8:11], s81 offen lds
	s_mov_b32 m0, s70
	s_nop 0
	buffer_load_dwordx4 v69, s[8:11], s81 offen lds
	s_mov_b32 m0, s71
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[116:119], v[144:147], a[204:207],  v76, v80 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[116:119], v[148:151], a[208:211],  v76, v80 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[116:119], v[152:155], a[212:215],  v76, v81 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[116:119], v[156:159], a[220:223],  v76, v81 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[132:135], v[160:163], a[204:207],  v76, v80 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[132:135], v[164:167], a[208:211],  v76, v80 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[132:135], v[168:171], a[212:215],  v76, v81 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[132:135], v[172:175], a[220:223],  v76, v81 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v71, s[8:11], s81 offen lds
	s_mov_b32 m0, s72
	s_nop 0
	buffer_load_dwordx4 v72, s[8:11], s81 offen lds
	s_mov_b32 m0, s40
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[120:123], v[144:147], a[224:227],  v77, v80 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[120:123], v[148:151], a[228:231],  v77, v80 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[120:123], v[152:155], a[232:235], v77, v81 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[120:123], v[156:159], a[236:239], v77, v81 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[136:139], v[160:163], a[224:227],  v77, v80 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[136:139], v[164:167], a[228:231],  v77, v80 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[136:139], v[168:171], a[232:235], v77, v81 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[136:139], v[172:175], a[236:239], v77, v81 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_u32 s81, s78, 0x300
	buffer_load_dwordx4 v70, s[8:11], s80 offen lds
	s_mov_b32 m0, s73
	s_add_u32 s78, s78, 0x380
	buffer_load_dwordx4 v69, s[8:11], s80 offen lds
	s_mov_b32 m0, s74
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[124:127], v[144:147], a[240:243], v77, v80 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[124:127], v[148:151], a[8:11], v77, v80 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[124:127], v[152:155], a[4:7], v77, v81 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[124:127], v[156:159], a[0:3], v77, v81 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[140:143], v[160:163], a[240:243], v77, v80 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[140:143], v[164:167], a[8:11], v77, v80 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[140:143], v[168:171], a[4:7], v77, v81 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[140:143], v[172:175], a[0:3], v77, v81 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v71, s[8:11], s80 offen lds
	s_mov_b32 m0, s75
	s_nop 0
	buffer_load_dwordx4 v72, s[8:11], s80 offen lds
	s_mov_b32 m0, s38
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15],  v[2:5], v[34:37], a[12:15],  v92, v88 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[144:147], v101 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19],  v[2:5], v[38:41], a[16:19],  v92, v88 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[148:151], v101 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23],  v[2:5], v[42:45], a[20:23],  v92, v89 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[152:155], v101 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27],  v[2:5], v[46:49], a[24:27],  v92, v89 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[156:159], v101 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15],  v[18:21], v[50:53], a[12:15],  v92, v88 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[160:163], v100 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19],  v[18:21], v[54:57], a[16:19],  v92, v88 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[164:167], v100 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23],  v[18:21], v[58:61], a[20:23],  v92, v89 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[168:171], v100 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27],  v[18:21], v[62:65], a[24:27],  v92, v89 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[172:175], v100 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[6:9], v[34:37], a[28:31],  v92, v88 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[6:9], v[38:41], a[32:35],  v92, v88 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[6:9], v[42:45], a[36:39],  v92, v89 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[6:9], v[46:49], a[40:43],  v92, v89 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[22:25], v[50:53], a[28:31],  v92, v88 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[22:25], v[54:57], a[32:35],  v92, v88 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[22:25], v[58:61], a[36:39],  v92, v89 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[22:25], v[62:65], a[40:43],  v92, v89 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[10:13], v[34:37], a[44:47],  v93, v88 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[10:13], v[38:41], a[48:51],  v93, v88 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[10:13], v[42:45], a[76:79], v93, v89 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[10:13], v[46:49], a[164:167], v93, v89 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[26:29], v[50:53], a[44:47],  v93, v88 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[26:29], v[54:57], a[48:51],  v93, v88 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[26:29], v[58:61], a[76:79], v93, v89 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[26:29], v[62:65], a[164:167], v93, v89 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[14:17], v[34:37], a[216:219], v93, v88 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[14:17], v[38:41], a[244:247], v93, v88 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[14:17], v[42:45], a[248:251], v93, v89 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[14:17], v[46:49], a[252:255], v93, v89 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[30:33], v[50:53], a[216:219], v93, v88 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[30:33], v[54:57], a[244:247], v93, v88 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[30:33], v[58:61], a[248:251], v93, v89 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[30:33], v[62:65], a[252:255], v93, v89 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[2:5], v[144:147], a[52:55], v92, v78 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[112:115], v97 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[2:5], v[148:151], a[56:59], v92, v78 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[116:119], v97 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[2:5], v[152:155], a[60:63], v92, v79 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[120:123], v97 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[2:5], v[156:159], a[64:67], v92, v79 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[124:127], v97 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[18:21], v[160:163], a[52:55], v92, v78 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[128:131], v96 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[18:21], v[164:167], a[56:59], v92, v78 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[132:135], v96 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[18:21], v[168:171], a[60:63], v92, v79 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[136:139], v96 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[18:21], v[172:175], a[64:67], v92, v79 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[140:143], v96 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[6:9], v[144:147], a[68:71], v92, v78 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[6:9], v[148:151], a[72:75], v92, v78 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[6:9], v[152:155], a[80:83], v92, v79 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[6:9], v[156:159], a[84:87], v92, v79 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[22:25], v[160:163], a[68:71], v92, v78 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[22:25], v[164:167], a[72:75], v92, v78 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[22:25], v[168:171], a[80:83], v92, v79 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[22:25], v[172:175], a[84:87], v92, v79 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[10:13], v[144:147], a[88:91], v93, v78 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[10:13], v[148:151], a[92:95], v93, v78 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[10:13], v[152:155], a[96:99], v93, v79 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[10:13], v[156:159], a[100:103], v93, v79 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[26:29], v[160:163], a[88:91], v93, v78 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[26:29], v[164:167], a[92:95], v93, v78 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[26:29], v[168:171], a[96:99], v93, v79 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[26:29], v[172:175], a[100:103], v93, v79 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[14:17], v[144:147], a[104:107], v93, v78 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[14:17], v[148:151], a[108:111], v93, v78 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[14:17], v[152:155], a[112:115], v93, v79 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[14:17], v[156:159], a[116:119], v93, v79 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[30:33], v[160:163], a[104:107], v93, v78 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[30:33], v[164:167], a[108:111], v93, v78 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[30:33], v[168:171], a[112:115], v93, v79 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[30:33], v[172:175], a[116:119], v93, v79 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(16)
s_barrier

	;;#ASMEND
	s_add_u32 s80, s79, 0x300
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123],  v[112:115], v[34:37], a[120:123],  v74, v88 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[2:5], v106 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127],  v[112:115], v[38:41], a[124:127],  v74, v88 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[6:9], v106 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[112:115], v[42:45], a[128:131],  v74, v89 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[10:13], v106 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[112:115], v[46:49], a[132:135],  v74, v89 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[14:17], v106 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123],  v[128:131], v[50:53], a[120:123],  v74, v88 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[18:21], v107 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127],  v[128:131], v[54:57], a[124:127],  v74, v88 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[22:25], v107 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[128:131], v[58:61], a[128:131],  v74, v89 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[26:29], v107 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[128:131], v[62:65], a[132:135],  v74, v89 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v107 offset:6144

	;;#ASMEND
	buffer_load_dwordx4 v66, s[4:7], s83 offen lds
	s_mov_b32 m0, s31
	s_nop 0
	buffer_load_dwordx4 v67, s[4:7], s83 offen lds
	s_mov_b32 m0, s47
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[116:119], v[34:37], a[136:139],  v74, v88 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[116:119], v[38:41], a[140:143],  v74, v88 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[116:119], v[42:45], a[144:147],  v74, v89 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[116:119], v[46:49], a[148:151],  v74, v89 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[132:135], v[50:53], a[136:139],  v74, v88 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[132:135], v[54:57], a[140:143],  v74, v88 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[132:135], v[58:61], a[144:147],  v74, v89 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[132:135], v[62:65], a[148:151],  v74, v89 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v94, s[4:7], s83 offen lds
	s_mov_b32 m0, s48
	s_nop 0
	buffer_load_dwordx4 v68, s[4:7], s83 offen lds
	s_mov_b32 m0, s43
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[120:123], v[34:37], a[152:155],  v75, v88 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[120:123], v[38:41], a[156:159],  v75, v88 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[120:123], v[42:45], a[160:163], v75, v89 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[120:123], v[46:49], a[168:171], v75, v89 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[136:139], v[50:53], a[152:155],  v75, v88 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[136:139], v[54:57], a[156:159],  v75, v88 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[136:139], v[58:61], a[160:163], v75, v89 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[136:139], v[62:65], a[168:171], v75, v89 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v66, s[4:7], s82 offen lds
	s_mov_b32 m0, s50
	s_nop 0
	buffer_load_dwordx4 v67, s[4:7], s82 offen lds
	s_mov_b32 m0, s51
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[124:127], v[34:37], a[172:175], v75, v88 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[124:127], v[38:41], a[176:179], v75, v88 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[124:127], v[42:45], a[180:183], v75, v89 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[124:127], v[46:49], a[184:187], v75, v89 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[140:143], v[50:53], a[172:175], v75, v88 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[140:143], v[54:57], a[176:179], v75, v88 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[140:143], v[58:61], a[180:183], v75, v89 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[140:143], v[62:65], a[184:187], v75, v89 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v94, s[4:7], s82 offen lds
	s_mov_b32 m0, s52
	s_nop 0
	buffer_load_dwordx4 v68, s[4:7], s82 offen lds
	s_mov_b32 m0, s41
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191],  v[112:115], v[144:147], a[188:191],  v74, v78 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v103 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[112:115], v[148:151], a[192:195],  v74, v78 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v103 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[112:115], v[152:155], a[196:199],  v74, v79 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v103 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[112:115], v[156:159], a[200:203],  v74, v79 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v103 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191],  v[128:131], v[160:163], a[188:191],  v74, v78 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v102 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[128:131], v[164:167], a[192:195],  v74, v78 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v102 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[128:131], v[168:171], a[196:199],  v74, v79 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v102 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[128:131], v[172:175], a[200:203],  v74, v79 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[62:65], v102 offset:6144

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v70, s[8:11], s81 offen lds
	s_mov_b32 m0, s35
	s_nop 0
	buffer_load_dwordx4 v69, s[8:11], s81 offen lds
	s_mov_b32 m0, s54
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[116:119], v[144:147], a[204:207],  v74, v78 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[116:119], v[148:151], a[208:211],  v74, v78 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[116:119], v[152:155], a[212:215],  v74, v79 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[116:119], v[156:159], a[220:223],  v74, v79 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[132:135], v[160:163], a[204:207],  v74, v78 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[132:135], v[164:167], a[208:211],  v74, v78 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[132:135], v[168:171], a[212:215],  v74, v79 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[132:135], v[172:175], a[220:223],  v74, v79 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v71, s[8:11], s81 offen lds
	s_mov_b32 m0, s55
	s_nop 0
	buffer_load_dwordx4 v72, s[8:11], s81 offen lds
	s_mov_b32 m0, s39
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[120:123], v[144:147], a[224:227],  v75, v78 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[120:123], v[148:151], a[228:231],  v75, v78 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[120:123], v[152:155], a[232:235], v75, v79 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[120:123], v[156:159], a[236:239], v75, v79 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[136:139], v[160:163], a[224:227],  v75, v78 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[136:139], v[164:167], a[228:231],  v75, v78 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[136:139], v[168:171], a[232:235], v75, v79 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[136:139], v[172:175], a[236:239], v75, v79 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_u32 s81, s76, 0x380
	buffer_load_dwordx4 v70, s[8:11], s80 offen lds
	s_mov_b32 m0, s57
	s_add_i32 s76, s29, -1
	buffer_load_dwordx4 v69, s[8:11], s80 offen lds
	s_mov_b32 m0, s58
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[124:127], v[144:147], a[240:243], v75, v78 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[124:127], v[148:151], a[8:11], v75, v78 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[124:127], v[152:155], a[4:7], v75, v79 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[124:127], v[156:159], a[0:3], v75, v79 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[140:143], v[160:163], a[240:243], v75, v78 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[140:143], v[164:167], a[8:11], v75, v78 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[140:143], v[168:171], a[4:7], v75, v79 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[140:143], v[172:175], a[0:3], v75, v79 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_min_u32 s76, s76, 0x7d
	buffer_load_dwordx4 v71, s[8:11], s80 offen lds
	s_mov_b32 m0, s59
	s_lshl_b32 s76, s76, 7
	buffer_load_dwordx4 v72, s[8:11], s80 offen lds
	s_mov_b32 m0, s45
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15],  v[2:5], v[34:37], a[12:15],  v90, v86 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[144:147], v99 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19],  v[2:5], v[38:41], a[16:19],  v90, v86 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[148:151], v99 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23],  v[2:5], v[42:45], a[20:23],  v90, v87 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[152:155], v99 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27],  v[2:5], v[46:49], a[24:27],  v90, v87 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[156:159], v99 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15],  v[18:21], v[50:53], a[12:15],  v90, v86 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[160:163], v98 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19],  v[18:21], v[54:57], a[16:19],  v90, v86 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[164:167], v98 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23],  v[18:21], v[58:61], a[20:23],  v90, v87 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[168:171], v98 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27],  v[18:21], v[62:65], a[24:27],  v90, v87 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[172:175], v98 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[6:9], v[34:37], a[28:31],  v90, v86 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[6:9], v[38:41], a[32:35],  v90, v86 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[6:9], v[42:45], a[36:39],  v90, v87 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[6:9], v[46:49], a[40:43],  v90, v87 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[22:25], v[50:53], a[28:31],  v90, v86 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[22:25], v[54:57], a[32:35],  v90, v86 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[22:25], v[58:61], a[36:39],  v90, v87 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[22:25], v[62:65], a[40:43],  v90, v87 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[10:13], v[34:37], a[44:47],  v91, v86 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[10:13], v[38:41], a[48:51],  v91, v86 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[10:13], v[42:45], a[76:79], v91, v87 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[10:13], v[46:49], a[164:167], v91, v87 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[26:29], v[50:53], a[44:47],  v91, v86 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[26:29], v[54:57], a[48:51],  v91, v86 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[26:29], v[58:61], a[76:79], v91, v87 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[26:29], v[62:65], a[164:167], v91, v87 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[14:17], v[34:37], a[216:219], v91, v86 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[14:17], v[38:41], a[244:247], v91, v86 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[14:17], v[42:45], a[248:251], v91, v87 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[14:17], v[46:49], a[252:255], v91, v87 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[30:33], v[50:53], a[216:219], v91, v86 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[30:33], v[54:57], a[244:247], v91, v86 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[30:33], v[58:61], a[248:251], v91, v87 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[30:33], v[62:65], a[252:255], v91, v87 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[2:5], v[144:147], a[52:55], v90, v84 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[112:115], v95 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[2:5], v[148:151], a[56:59], v90, v84 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[116:119], v95 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[2:5], v[152:155], a[60:63], v90, v85 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[120:123], v95 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[2:5], v[156:159], a[64:67], v90, v85 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[124:127], v95 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[18:21], v[160:163], a[52:55], v90, v84 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[128:131], v73 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[18:21], v[164:167], a[56:59], v90, v84 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[132:135], v73 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[18:21], v[168:171], a[60:63], v90, v85 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[136:139], v73 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[18:21], v[172:175], a[64:67], v90, v85 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[140:143], v73 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[6:9], v[144:147], a[68:71], v90, v84 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[6:9], v[148:151], a[72:75], v90, v84 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[6:9], v[152:155], a[80:83], v90, v85 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[6:9], v[156:159], a[84:87], v90, v85 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[22:25], v[160:163], a[68:71], v90, v84 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[22:25], v[164:167], a[72:75], v90, v84 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[22:25], v[168:171], a[80:83], v90, v85 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[22:25], v[172:175], a[84:87], v90, v85 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[10:13], v[144:147], a[88:91], v91, v84 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[10:13], v[148:151], a[92:95], v91, v84 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[10:13], v[152:155], a[96:99], v91, v85 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[10:13], v[156:159], a[100:103], v91, v85 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[26:29], v[160:163], a[88:91], v91, v84 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[26:29], v[164:167], a[92:95], v91, v84 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[26:29], v[168:171], a[96:99], v91, v85 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[26:29], v[172:175], a[100:103], v91, v85 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[14:17], v[144:147], a[104:107], v91, v84 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[14:17], v[148:151], a[108:111], v91, v84 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[14:17], v[152:155], a[112:115], v91, v85 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[14:17], v[156:159], a[116:119], v91, v85 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[30:33], v[160:163], a[104:107], v91, v84 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[30:33], v[164:167], a[108:111], v91, v84 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[30:33], v[168:171], a[112:115], v91, v85 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[30:33], v[172:175], a[116:119], v91, v85 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(16)
s_barrier

	;;#ASMEND
	s_add_u32 s80, s77, 0x380
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123],  v[112:115], v[34:37], a[120:123],  v82, v86 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[2:5], v109 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127],  v[112:115], v[38:41], a[124:127],  v82, v86 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[6:9], v109 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[112:115], v[42:45], a[128:131],  v82, v87 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[10:13], v109 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[112:115], v[46:49], a[132:135],  v82, v87 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[14:17], v109 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123],  v[128:131], v[50:53], a[120:123],  v82, v86 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[18:21], v108 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127],  v[128:131], v[54:57], a[124:127],  v82, v86 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[22:25], v108 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[128:131], v[58:61], a[128:131],  v82, v87 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[26:29], v108 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[128:131], v[62:65], a[132:135],  v82, v87 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v108 offset:6144

	;;#ASMEND
	buffer_load_dwordx4 v66, s[4:7], s81 offen lds
	s_mov_b32 m0, s63
	s_add_u32 s77, s79, 0x380
	buffer_load_dwordx4 v67, s[4:7], s81 offen lds
	s_mov_b32 m0, s65
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[116:119], v[34:37], a[136:139],  v82, v86 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[116:119], v[38:41], a[140:143],  v82, v86 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[116:119], v[42:45], a[144:147],  v82, v87 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[116:119], v[46:49], a[148:151],  v82, v87 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[132:135], v[50:53], a[136:139],  v82, v86 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[132:135], v[54:57], a[140:143],  v82, v86 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[132:135], v[58:61], a[144:147],  v82, v87 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[132:135], v[62:65], a[148:151],  v82, v87 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_i32 s79, s33, 0xfffffc00
	buffer_load_dwordx4 v94, s[4:7], s81 offen lds
	s_mov_b32 m0, s66
	;;#ASMSTART
	buffer_load_dwordx2 v[80:81], v110, s[0:3], s79 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[74:75], v110, s[20:23], s79 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[78:79], v110, s[16:19], s79 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[76:77], v110, s[12:15], s79 offen
	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v68, s[4:7], s81 offen lds
	s_mov_b32 m0, s44
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[120:123], v[34:37], a[152:155],  v83, v86 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[120:123], v[38:41], a[156:159],  v83, v86 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[120:123], v[42:45], a[160:163], v83, v87 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[120:123], v[46:49], a[168:171], v83, v87 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[136:139], v[50:53], a[152:155],  v83, v86 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[136:139], v[54:57], a[156:159],  v83, v86 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[136:139], v[58:61], a[160:163], v83, v87 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[136:139], v[62:65], a[168:171], v83, v87 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v66, s[4:7], s80 offen lds
	s_mov_b32 m0, s67
	s_nop 0
	buffer_load_dwordx4 v67, s[4:7], s80 offen lds
	s_mov_b32 m0, s68
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[124:127], v[34:37], a[172:175], v83, v86 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[124:127], v[38:41], a[176:179], v83, v86 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[124:127], v[42:45], a[180:183], v83, v87 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[124:127], v[46:49], a[184:187], v83, v87 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[140:143], v[50:53], a[172:175], v83, v86 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[140:143], v[54:57], a[176:179], v83, v86 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[140:143], v[58:61], a[180:183], v83, v87 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[140:143], v[62:65], a[184:187], v83, v87 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v94, s[4:7], s80 offen lds
	s_mov_b32 m0, s69
	s_nop 0
	buffer_load_dwordx4 v68, s[4:7], s80 offen lds
	s_mov_b32 m0, s42
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191],  v[112:115], v[144:147], a[188:191],  v82, v84 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v105 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[112:115], v[148:151], a[192:195],  v82, v84 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v105 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[112:115], v[152:155], a[196:199],  v82, v85 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v105 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[112:115], v[156:159], a[200:203],  v82, v85 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v105 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191],  v[128:131], v[160:163], a[188:191],  v82, v84 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v104 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[128:131], v[164:167], a[192:195],  v82, v84 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v104 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[128:131], v[168:171], a[196:199],  v82, v85 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v104 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[128:131], v[172:175], a[200:203],  v82, v85 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[62:65], v104 offset:6144

	;;#ASMEND
	s_add_i32 s80, s33, 0xfffffe00
	buffer_load_dwordx4 v70, s[8:11], s78 offen lds
	s_mov_b32 m0, s70
	;;#ASMSTART
	buffer_load_dwordx2 v[88:89], v110, s[0:3], s80 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[86:87], v110, s[16:19], s80 offen
	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v69, s[8:11], s78 offen lds
	s_mov_b32 m0, s71
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[116:119], v[144:147], a[204:207],  v82, v84 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[116:119], v[148:151], a[208:211],  v82, v84 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[116:119], v[152:155], a[212:215],  v82, v85 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[116:119], v[156:159], a[220:223],  v82, v85 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[132:135], v[160:163], a[204:207],  v82, v84 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[132:135], v[164:167], a[208:211],  v82, v84 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[132:135], v[168:171], a[212:215],  v82, v85 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[132:135], v[172:175], a[220:223],  v82, v85 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v71, s[8:11], s78 offen lds
	s_mov_b32 m0, s72
	s_nop 0
	buffer_load_dwordx4 v72, s[8:11], s78 offen lds
	s_mov_b32 m0, s40
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[120:123], v[144:147], a[224:227],  v83, v84 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[120:123], v[148:151], a[228:231],  v83, v84 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[120:123], v[152:155], a[232:235], v83, v85 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[120:123], v[156:159], a[236:239], v83, v85 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[136:139], v[160:163], a[224:227],  v83, v84 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[136:139], v[164:167], a[228:231],  v83, v84 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[136:139], v[168:171], a[232:235], v83, v85 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[136:139], v[172:175], a[236:239], v83, v85 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v70, s[8:11], s77 offen lds
	s_mov_b32 m0, s73
	s_nop 0
	buffer_load_dwordx4 v69, s[8:11], s77 offen lds
	s_mov_b32 m0, s74
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[124:127], v[144:147], a[240:243], v83, v84 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[124:127], v[148:151], a[8:11], v83, v84 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[124:127], v[152:155], a[4:7], v83, v85 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[124:127], v[156:159], a[0:3], v83, v85 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[140:143], v[160:163], a[240:243], v83, v84 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[140:143], v[164:167], a[8:11], v83, v84 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[140:143], v[168:171], a[4:7], v83, v85 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[140:143], v[172:175], a[0:3], v83, v85 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[82:83], v110, s[20:23], s80 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[84:85], v110, s[12:15], s80 offen
	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v71, s[8:11], s77 offen lds
	s_mov_b32 m0, s75
	s_nop 0
	buffer_load_dwordx4 v72, s[8:11], s77 offen lds
	s_add_u32 s77, s64, s76
	s_sub_u32 s79, s77, s30
	s_mov_b32 m0, s38
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15],  v[2:5], v[34:37], a[12:15],  v80, v78 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[140:143], v101 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19],  v[2:5], v[38:41], a[16:19],  v80, v78 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[144:147], v101 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23],  v[2:5], v[42:45], a[20:23],  v80, v79 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[148:151], v101 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27],  v[2:5], v[46:49], a[24:27],  v80, v79 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[152:155], v101 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15],  v[18:21], v[50:53], a[12:15],  v80, v78 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[156:159], v100 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19],  v[18:21], v[54:57], a[16:19],  v80, v78 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[160:163], v100 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23],  v[18:21], v[58:61], a[20:23],  v80, v79 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[164:167], v100 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27],  v[18:21], v[62:65], a[24:27],  v80, v79 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[168:171], v100 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[6:9], v[34:37], a[28:31],  v80, v78 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[6:9], v[38:41], a[32:35],  v80, v78 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[6:9], v[42:45], a[36:39],  v80, v79 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[6:9], v[46:49], a[40:43],  v80, v79 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[22:25], v[50:53], a[28:31],  v80, v78 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[22:25], v[54:57], a[32:35],  v80, v78 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[22:25], v[58:61], a[36:39],  v80, v79 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[22:25], v[62:65], a[40:43],  v80, v79 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[10:13], v[34:37], a[44:47],  v81, v78 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[10:13], v[38:41], a[48:51],  v81, v78 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[10:13], v[42:45], a[76:79], v81, v79 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[10:13], v[46:49], a[164:167], v81, v79 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[26:29], v[50:53], a[44:47],  v81, v78 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[26:29], v[54:57], a[48:51],  v81, v78 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[26:29], v[58:61], a[76:79], v81, v79 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[26:29], v[62:65], a[164:167], v81, v79 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[14:17], v[34:37], a[216:219], v81, v78 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[14:17], v[38:41], a[244:247], v81, v78 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[14:17], v[42:45], a[248:251], v81, v79 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[14:17], v[46:49], a[252:255], v81, v79 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[30:33], v[50:53], a[216:219], v81, v78 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[30:33], v[54:57], a[244:247], v81, v78 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[30:33], v[58:61], a[248:251], v81, v79 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[30:33], v[62:65], a[252:255], v81, v79 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[2:5], v[140:143], a[52:55], v80, v76 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[90:93], v97 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[2:5], v[144:147], a[56:59], v80, v76 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[112:115], v97 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[2:5], v[148:151], a[60:63], v80, v77 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[116:119], v97 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[2:5], v[152:155], a[64:67], v80, v77 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[120:123], v97 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[18:21], v[156:159], a[52:55], v80, v76 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[124:127], v96 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[18:21], v[160:163], a[56:59], v80, v76 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[128:131], v96 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[18:21], v[164:167], a[60:63], v80, v77 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[132:135], v96 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[18:21], v[168:171], a[64:67], v80, v77 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[136:139], v96 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[6:9], v[140:143], a[68:71], v80, v76 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[6:9], v[144:147], a[72:75], v80, v76 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[6:9], v[148:151], a[80:83], v80, v77 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[6:9], v[152:155], a[84:87], v80, v77 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[22:25], v[156:159], a[68:71], v80, v76 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[22:25], v[160:163], a[72:75], v80, v76 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[22:25], v[164:167], a[80:83], v80, v77 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[22:25], v[168:171], a[84:87], v80, v77 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[10:13], v[140:143], a[88:91], v81, v76 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[10:13], v[144:147], a[92:95], v81, v76 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[10:13], v[148:151], a[96:99], v81, v77 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[10:13], v[152:155], a[100:103], v81, v77 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[26:29], v[156:159], a[88:91], v81, v76 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[26:29], v[160:163], a[92:95], v81, v76 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[26:29], v[164:167], a[96:99], v81, v77 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[26:29], v[168:171], a[100:103], v81, v77 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[14:17], v[140:143], a[104:107], v81, v76 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[14:17], v[144:147], a[108:111], v81, v76 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[14:17], v[148:151], a[112:115], v81, v77 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[14:17], v[152:155], a[116:119], v81, v77 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[30:33], v[156:159], a[104:107], v81, v76 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[30:33], v[160:163], a[108:111], v81, v76 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[30:33], v[164:167], a[112:115], v81, v77 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[30:33], v[168:171], a[116:119], v81, v77 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(16)
s_barrier

	;;#ASMEND
	s_add_u32 s77, s62, s76
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123],  v[90:93], v[34:37], a[120:123],  v74, v78 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[2:5], v106 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127],  v[90:93], v[38:41], a[124:127],  v74, v78 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[6:9], v106 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[90:93], v[42:45], a[128:131],  v74, v79 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[10:13], v106 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[90:93], v[46:49], a[132:135],  v74, v79 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[14:17], v106 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123],  v[124:127], v[50:53], a[120:123],  v74, v78 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[18:21], v107 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127],  v[124:127], v[54:57], a[124:127],  v74, v78 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[22:25], v107 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[124:127], v[58:61], a[128:131],  v74, v79 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[26:29], v107 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[124:127], v[62:65], a[132:135],  v74, v79 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v107 offset:6144

	;;#ASMEND
	buffer_load_dwordx4 v66, s[4:7], s79 offen lds
	s_mov_b32 m0, s31
	s_sub_u32 s78, s77, s30
	buffer_load_dwordx4 v67, s[4:7], s79 offen lds
	s_mov_b32 m0, s47
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[112:115], v[34:37], a[136:139],  v74, v78 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[112:115], v[38:41], a[140:143],  v74, v78 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[112:115], v[42:45], a[144:147],  v74, v79 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[112:115], v[46:49], a[148:151],  v74, v79 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[128:131], v[50:53], a[136:139],  v74, v78 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[128:131], v[54:57], a[140:143],  v74, v78 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[128:131], v[58:61], a[144:147],  v74, v79 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[128:131], v[62:65], a[148:151],  v74, v79 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_u32 s77, s61, s76
	buffer_load_dwordx4 v94, s[4:7], s79 offen lds
	s_mov_b32 m0, s48
	s_sub_u32 s77, s77, s34
	buffer_load_dwordx4 v68, s[4:7], s79 offen lds
	s_mov_b32 m0, s43
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[116:119], v[34:37], a[152:155],  v75, v78 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[116:119], v[38:41], a[156:159],  v75, v78 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[116:119], v[42:45], a[160:163], v75, v79 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[116:119], v[46:49], a[168:171], v75, v79 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[132:135], v[50:53], a[152:155],  v75, v78 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[132:135], v[54:57], a[156:159],  v75, v78 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[132:135], v[58:61], a[160:163], v75, v79 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[132:135], v[62:65], a[168:171], v75, v79 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_u32 s76, s60, s76
	buffer_load_dwordx4 v66, s[4:7], s78 offen lds
	s_mov_b32 m0, s50
	s_sub_u32 s76, s76, s34
	buffer_load_dwordx4 v67, s[4:7], s78 offen lds
	s_mov_b32 m0, s51
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[120:123], v[34:37], a[172:175], v75, v78 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[120:123], v[38:41], a[176:179], v75, v78 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[120:123], v[42:45], a[180:183], v75, v79 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[120:123], v[46:49], a[184:187], v75, v79 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[136:139], v[50:53], a[172:175], v75, v78 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[136:139], v[54:57], a[176:179], v75, v78 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[136:139], v[58:61], a[180:183], v75, v79 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[136:139], v[62:65], a[184:187], v75, v79 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v94, s[4:7], s78 offen lds
	s_mov_b32 m0, s52
	s_nop 0
	buffer_load_dwordx4 v68, s[4:7], s78 offen lds
	s_mov_b32 m0, s41
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191],  v[90:93], v[140:143], a[188:191],  v74, v76 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v103 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[90:93], v[144:147], a[192:195],  v74, v76 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v103 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[90:93], v[148:151], a[196:199],  v74, v77 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v103 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[90:93], v[152:155], a[200:203],  v74, v77 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v103 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191],  v[124:127], v[156:159], a[188:191],  v74, v76 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v102 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[124:127], v[160:163], a[192:195],  v74, v76 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v102 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[124:127], v[164:167], a[196:199],  v74, v77 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v102 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[124:127], v[168:171], a[200:203],  v74, v77 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[62:65], v102 offset:6144

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v70, s[8:11], s77 offen lds
	s_mov_b32 m0, s35
	s_nop 0
	buffer_load_dwordx4 v69, s[8:11], s77 offen lds
	s_mov_b32 m0, s54
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[112:115], v[140:143], a[204:207],  v74, v76 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[112:115], v[144:147], a[208:211],  v74, v76 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[112:115], v[148:151], a[212:215],  v74, v77 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[112:115], v[152:155], a[220:223],  v74, v77 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[128:131], v[156:159], a[204:207],  v74, v76 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[128:131], v[160:163], a[208:211],  v74, v76 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[128:131], v[164:167], a[212:215],  v74, v77 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[128:131], v[168:171], a[220:223],  v74, v77 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v71, s[8:11], s77 offen lds
	s_mov_b32 m0, s55
	s_nop 0
	buffer_load_dwordx4 v72, s[8:11], s77 offen lds
	s_mov_b32 m0, s39
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[116:119], v[140:143], a[224:227],  v75, v76 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[116:119], v[144:147], a[228:231],  v75, v76 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[116:119], v[148:151], a[232:235], v75, v77 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[116:119], v[152:155], a[236:239], v75, v77 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[132:135], v[156:159], a[224:227],  v75, v76 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[132:135], v[160:163], a[228:231],  v75, v76 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[132:135], v[164:167], a[232:235], v75, v77 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[132:135], v[168:171], a[236:239], v75, v77 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v70, s[8:11], s76 offen lds
	s_mov_b32 m0, s57
	s_nop 0
	buffer_load_dwordx4 v69, s[8:11], s76 offen lds
	s_mov_b32 m0, s58
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[120:123], v[140:143], a[240:243], v75, v76 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[120:123], v[144:147], a[8:11], v75, v76 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[120:123], v[148:151], a[4:7], v75, v77 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[120:123], v[152:155], a[0:3], v75, v77 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[136:139], v[156:159], a[240:243], v75, v76 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[136:139], v[160:163], a[8:11], v75, v76 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[136:139], v[164:167], a[4:7], v75, v77 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[136:139], v[168:171], a[0:3], v75, v77 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v71, s[8:11], s76 offen lds
	s_mov_b32 m0, s59
	s_nop 0
	buffer_load_dwordx4 v72, s[8:11], s76 offen lds
	s_min_u32 s76, s29, 0x7d
	s_lshl_b32 s76, s76, 7
	s_add_u32 s77, s64, s76
	s_sub_u32 s79, s77, s30
	s_mov_b32 m0, s45
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15],  v[2:5], v[34:37], a[12:15],  v88, v86 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[140:143], v99 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19],  v[2:5], v[38:41], a[16:19],  v88, v86 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[144:147], v99 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23],  v[2:5], v[42:45], a[20:23],  v88, v87 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[148:151], v99 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27],  v[2:5], v[46:49], a[24:27],  v88, v87 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[152:155], v99 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15],  v[18:21], v[50:53], a[12:15],  v88, v86 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[156:159], v98 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19],  v[18:21], v[54:57], a[16:19],  v88, v86 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[160:163], v98 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23],  v[18:21], v[58:61], a[20:23],  v88, v87 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[164:167], v98 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27],  v[18:21], v[62:65], a[24:27],  v88, v87 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[168:171], v98 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[6:9], v[34:37], a[28:31],  v88, v86 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[6:9], v[38:41], a[32:35],  v88, v86 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[6:9], v[42:45], a[36:39],  v88, v87 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[6:9], v[46:49], a[40:43],  v88, v87 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[22:25], v[50:53], a[28:31],  v88, v86 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[22:25], v[54:57], a[32:35],  v88, v86 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[22:25], v[58:61], a[36:39],  v88, v87 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[22:25], v[62:65], a[40:43],  v88, v87 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[10:13], v[34:37], a[44:47],  v89, v86 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[10:13], v[38:41], a[48:51],  v89, v86 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[10:13], v[42:45], a[76:79], v89, v87 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[10:13], v[46:49], a[164:167], v89, v87 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[26:29], v[50:53], a[44:47],  v89, v86 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[26:29], v[54:57], a[48:51],  v89, v86 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[26:29], v[58:61], a[76:79], v89, v87 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[26:29], v[62:65], a[164:167], v89, v87 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[14:17], v[34:37], a[216:219], v89, v86 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[14:17], v[38:41], a[244:247], v89, v86 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[14:17], v[42:45], a[248:251], v89, v87 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[14:17], v[46:49], a[252:255], v89, v87 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[30:33], v[50:53], a[216:219], v89, v86 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[30:33], v[54:57], a[244:247], v89, v86 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[30:33], v[58:61], a[248:251], v89, v87 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[30:33], v[62:65], a[252:255], v89, v87 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[2:5], v[140:143], a[52:55], v88, v84 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[90:93], v95 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[2:5], v[144:147], a[56:59], v88, v84 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[112:115], v95 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[2:5], v[148:151], a[60:63], v88, v85 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[116:119], v95 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[2:5], v[152:155], a[64:67], v88, v85 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[120:123], v95 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[18:21], v[156:159], a[52:55], v88, v84 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[124:127], v73 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[18:21], v[160:163], a[56:59], v88, v84 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[128:131], v73 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[18:21], v[164:167], a[60:63], v88, v85 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[132:135], v73 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[18:21], v[168:171], a[64:67], v88, v85 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[136:139], v73 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[6:9], v[140:143], a[68:71], v88, v84 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[6:9], v[144:147], a[72:75], v88, v84 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[6:9], v[148:151], a[80:83], v88, v85 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[6:9], v[152:155], a[84:87], v88, v85 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[22:25], v[156:159], a[68:71], v88, v84 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[22:25], v[160:163], a[72:75], v88, v84 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[22:25], v[164:167], a[80:83], v88, v85 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[22:25], v[168:171], a[84:87], v88, v85 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[10:13], v[140:143], a[88:91], v89, v84 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[10:13], v[144:147], a[92:95], v89, v84 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[10:13], v[148:151], a[96:99], v89, v85 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[10:13], v[152:155], a[100:103], v89, v85 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[26:29], v[156:159], a[88:91], v89, v84 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[26:29], v[160:163], a[92:95], v89, v84 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[26:29], v[164:167], a[96:99], v89, v85 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[26:29], v[168:171], a[100:103], v89, v85 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[14:17], v[140:143], a[104:107], v89, v84 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[14:17], v[144:147], a[108:111], v89, v84 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[14:17], v[148:151], a[112:115], v89, v85 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[14:17], v[152:155], a[116:119], v89, v85 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[30:33], v[156:159], a[104:107], v89, v84 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[30:33], v[160:163], a[108:111], v89, v84 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[30:33], v[164:167], a[112:115], v89, v85 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[30:33], v[168:171], a[116:119], v89, v85 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(16)
s_barrier

	;;#ASMEND
	s_add_u32 s77, s62, s76
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123],  v[90:93], v[34:37], a[120:123],  v82, v86 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[2:5], v109 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127],  v[90:93], v[38:41], a[124:127],  v82, v86 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[6:9], v109 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[90:93], v[42:45], a[128:131],  v82, v87 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[10:13], v109 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[90:93], v[46:49], a[132:135],  v82, v87 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[14:17], v109 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123],  v[124:127], v[50:53], a[120:123],  v82, v86 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[18:21], v108 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127],  v[124:127], v[54:57], a[124:127],  v82, v86 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[22:25], v108 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[124:127], v[58:61], a[128:131],  v82, v87 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[26:29], v108 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[124:127], v[62:65], a[132:135],  v82, v87 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v108 offset:6144

	;;#ASMEND
	buffer_load_dwordx4 v66, s[4:7], s79 offen lds
	s_mov_b32 m0, s63
	s_sub_u32 s78, s77, s30
	buffer_load_dwordx4 v67, s[4:7], s79 offen lds
	s_mov_b32 m0, s65
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[112:115], v[34:37], a[136:139],  v82, v86 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[112:115], v[38:41], a[140:143],  v82, v86 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[112:115], v[42:45], a[144:147],  v82, v87 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[112:115], v[46:49], a[148:151],  v82, v87 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[128:131], v[50:53], a[136:139],  v82, v86 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[128:131], v[54:57], a[140:143],  v82, v86 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[128:131], v[58:61], a[144:147],  v82, v87 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[128:131], v[62:65], a[148:151],  v82, v87 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_u32 s77, s61, s76
	buffer_load_dwordx4 v94, s[4:7], s79 offen lds
	s_mov_b32 m0, s66
	s_sub_u32 s77, s77, s34
	buffer_load_dwordx4 v68, s[4:7], s79 offen lds
	s_mov_b32 m0, s44
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[116:119], v[34:37], a[152:155],  v83, v86 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[116:119], v[38:41], a[156:159],  v83, v86 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[116:119], v[42:45], a[160:163], v83, v87 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[116:119], v[46:49], a[168:171], v83, v87 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[132:135], v[50:53], a[152:155],  v83, v86 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[132:135], v[54:57], a[156:159],  v83, v86 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[132:135], v[58:61], a[160:163], v83, v87 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[132:135], v[62:65], a[168:171], v83, v87 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_u32 s76, s60, s76
	buffer_load_dwordx4 v66, s[4:7], s78 offen lds
	s_mov_b32 m0, s67
	s_sub_u32 s76, s76, s34
	buffer_load_dwordx4 v67, s[4:7], s78 offen lds
	s_mov_b32 m0, s68
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[120:123], v[34:37], a[172:175], v83, v86 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[120:123], v[38:41], a[176:179], v83, v86 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[120:123], v[42:45], a[180:183], v83, v87 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[120:123], v[46:49], a[184:187], v83, v87 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[136:139], v[50:53], a[172:175], v83, v86 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[136:139], v[54:57], a[176:179], v83, v86 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[136:139], v[58:61], a[180:183], v83, v87 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[136:139], v[62:65], a[184:187], v83, v87 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_cmpk_lg_i32 s36, 0x3c00
	buffer_load_dwordx4 v94, s[4:7], s78 offen lds
	s_mov_b32 m0, s69
	s_cselect_b32 s80, s33, 0xfe00
	buffer_load_dwordx4 v68, s[4:7], s78 offen lds
	s_mov_b32 m0, s42
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191],  v[90:93], v[140:143], a[188:191],  v82, v84 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v105 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[90:93], v[144:147], a[192:195],  v82, v84 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v105 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[90:93], v[148:151], a[196:199],  v82, v85 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v105 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[90:93], v[152:155], a[200:203],  v82, v85 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v105 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191],  v[124:127], v[156:159], a[188:191],  v82, v84 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v104 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[124:127], v[160:163], a[192:195],  v82, v84 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v104 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[124:127], v[164:167], a[196:199],  v82, v85 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v104 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[124:127], v[168:171], a[200:203],  v82, v85 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[62:65], v104 offset:6144

	;;#ASMEND
	s_add_u32 s36, s36, 0x400
	buffer_load_dwordx4 v70, s[8:11], s77 offen lds
	s_mov_b32 m0, s70
	s_addc_u32 s37, s37, 0
	buffer_load_dwordx4 v69, s[8:11], s77 offen lds
	s_mov_b32 m0, s71
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[112:115], v[140:143], a[204:207],  v82, v84 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[112:115], v[144:147], a[208:211],  v82, v84 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[112:115], v[148:151], a[212:215],  v82, v85 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[112:115], v[152:155], a[220:223],  v82, v85 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[128:131], v[156:159], a[204:207],  v82, v84 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[128:131], v[160:163], a[208:211],  v82, v84 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[128:131], v[164:167], a[212:215],  v82, v85 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[128:131], v[168:171], a[220:223],  v82, v85 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_addk_i32 s33, 0x1000
	buffer_load_dwordx4 v71, s[8:11], s77 offen lds
	s_mov_b32 m0, s72
	s_add_i32 s29, s29, 8
	buffer_load_dwordx4 v72, s[8:11], s77 offen lds
	s_mov_b32 m0, s40
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[116:119], v[140:143], a[224:227],  v83, v84 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[116:119], v[144:147], a[228:231],  v83, v84 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[116:119], v[148:151], a[232:235], v83, v85 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[116:119], v[152:155], a[236:239], v83, v85 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[132:135], v[156:159], a[224:227],  v83, v84 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[132:135], v[160:163], a[228:231],  v83, v84 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[132:135], v[164:167], a[232:235], v83, v85 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[132:135], v[168:171], a[236:239], v83, v85 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_cmpk_eq_i32 s36, 0x4000
	buffer_load_dwordx4 v70, s[8:11], s76 offen lds
	s_mov_b32 m0, s73
	;;#ASMSTART
	buffer_load_dwordx2 v[80:81], v110, s[0:3], s80 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[76:77], v110, s[20:23], s80 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[78:79], v110, s[16:19], s80 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[74:75], v110, s[12:15], s80 offen
	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v69, s[8:11], s76 offen lds
	s_mov_b32 m0, s74
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[120:123], v[140:143], a[240:243], v83, v84 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[120:123], v[144:147], a[8:11], v83, v84 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[120:123], v[148:151], a[4:7], v83, v85 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[120:123], v[152:155], a[0:3], v83, v85 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[136:139], v[156:159], a[240:243], v83, v84 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[136:139], v[160:163], a[8:11], v83, v84 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[136:139], v[164:167], a[4:7], v83, v85 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[136:139], v[168:171], a[0:3], v83, v85 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v71, s[8:11], s76 offen lds
	s_mov_b32 m0, s75
	s_nop 0
	buffer_load_dwordx4 v72, s[8:11], s76 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_cbranch_scc0 .LBB0_7
; %bb.8:
	v_accvgpr_read_b32 v0, a28
	v_accvgpr_read_b32 v1, a29
	v_accvgpr_read_b32 v221, a35
	v_accvgpr_read_b32 v215, a27
	v_pk_mul_f32 v[0:1], v[0:1], s[28:29] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v220, a34
	v_accvgpr_read_b32 v214, a26
	v_accvgpr_read_b32 v213, a25
	v_accvgpr_read_b32 v212, a24
	v_accvgpr_write_b32 a25, v1
	v_accvgpr_write_b32 a24, v0
	v_pk_mul_f32 v[0:1], v[220:221], s[28:29] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v219, a33
	v_accvgpr_read_b32 v218, a32
	v_accvgpr_write_b32 a27, v1
	v_accvgpr_read_b32 v227, a39
	v_accvgpr_write_b32 a26, v0
	v_pk_mul_f32 v[0:1], v[218:219], s[28:29] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v226, a38
	v_accvgpr_read_b32 v2, a30
	v_accvgpr_read_b32 v3, a31
	v_accvgpr_write_b32 a31, v1
	v_accvgpr_write_b32 a30, v0
	v_pk_mul_f32 v[0:1], v[226:227], s[28:29] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v225, a37
	v_accvgpr_read_b32 v224, a36
	v_accvgpr_write_b32 a33, v1
	v_accvgpr_read_b32 v255, a47
	v_accvgpr_write_b32 a32, v0
	v_pk_mul_f32 v[0:1], v[224:225], s[28:29] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v254, a46
	v_accvgpr_write_b32 a35, v1
	v_accvgpr_write_b32 a34, v0
	v_pk_mul_f32 v[0:1], v[254:255], s[28:29] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v253, a45
	v_accvgpr_read_b32 v252, a44
	v_accvgpr_write_b32 a29, v1
	v_accvgpr_read_b32 v237, a51
	v_accvgpr_write_b32 a28, v0
	v_pk_mul_f32 v[0:1], v[252:253], s[28:29] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v236, a50
	v_accvgpr_write_b32 a37, v1
	v_accvgpr_write_b32 a36, v0
	v_pk_mul_f32 v[0:1], v[236:237], s[28:29] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v235, a49
	v_accvgpr_read_b32 v234, a48
	v_accvgpr_write_b32 a39, v1
	v_accvgpr_read_b32 v243, a79
	v_accvgpr_read_b32 v231, a43
	v_accvgpr_write_b32 a38, v0
	v_pk_mul_f32 v[0:1], v[234:235], s[28:29] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v242, a78
	v_accvgpr_read_b32 v230, a42
	v_accvgpr_read_b32 v229, a41
	v_accvgpr_read_b32 v228, a40
	v_accvgpr_write_b32 a43, v1
	v_accvgpr_write_b32 a42, v0
	v_pk_mul_f32 v[0:1], v[242:243], s[28:29] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v241, a77
	v_accvgpr_read_b32 v240, a76
	v_accvgpr_write_b32 a45, v1
	v_accvgpr_read_b32 v251, a219
	v_accvgpr_write_b32 a44, v0
	v_pk_mul_f32 v[0:1], v[240:241], s[28:29] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v250, a218
	v_accvgpr_write_b32 a47, v1
	v_accvgpr_write_b32 a46, v0
	v_pk_mul_f32 v[0:1], v[250:251], s[28:29] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v249, a217
	v_accvgpr_read_b32 v248, a216
	v_accvgpr_write_b32 a41, v1
	v_accvgpr_read_b32 v181, a55
	v_accvgpr_write_b32 a40, v0
	v_pk_mul_f32 v[0:1], v[248:249], s[28:29] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v180, a54
	v_accvgpr_write_b32 a49, v1
	v_accvgpr_read_b32 v177, a59
	v_accvgpr_write_b32 a48, v0
	v_pk_mul_f32 v[0:1], v[180:181], s[28:29] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v176, a58
	v_accvgpr_write_b32 a51, v1
	v_accvgpr_write_b32 a50, v0
	v_pk_mul_f32 v[0:1], v[176:177], s[28:29] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v175, a57
	v_accvgpr_read_b32 v174, a56
	v_accvgpr_read_b32 v179, a53
	v_accvgpr_read_b32 v178, a52
	v_accvgpr_write_b32 a53, v1
	v_accvgpr_read_b32 v165, a71
	v_accvgpr_write_b32 a52, v0
	v_pk_mul_f32 v[0:1], v[174:175], s[28:29] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v164, a70
	v_accvgpr_write_b32 a55, v1
	v_accvgpr_read_b32 v149, a91
	v_accvgpr_write_b32 a54, v0
	v_pk_mul_f32 v[0:1], v[164:165], s[28:29] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v148, a90
	v_accvgpr_write_b32 a59, v1
	v_accvgpr_write_b32 a58, v0
	v_pk_mul_f32 v[0:1], v[148:149], s[28:29] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v201, a15
	v_accvgpr_write_b32 a57, v1
	v_accvgpr_write_b32 a56, v0
	scratch_load_dwordx2 v[0:1], off, off   ; 8-byte Folded Reload
	v_accvgpr_read_b32 v194, a240
	v_accvgpr_read_b32 v46, a192
	v_accvgpr_read_b32 v50, a188
	v_accvgpr_read_b32 v94, a140
	v_accvgpr_read_b32 v141, a99
	v_accvgpr_read_b32 v145, a95
	v_accvgpr_read_b32 v200, a14
	v_accvgpr_read_b32 v195, a241
	v_accvgpr_read_b32 v196, a242
	v_accvgpr_read_b32 v197, a243
	v_accvgpr_read_b32 v48, a194
	v_accvgpr_read_b32 v49, a195
	v_accvgpr_read_b32 v51, a189
	v_accvgpr_read_b32 v90, a144
	v_accvgpr_read_b32 v95, a141
	v_accvgpr_read_b32 v96, a142
	v_accvgpr_read_b32 v97, a143
	v_accvgpr_read_b32 v140, a98
	v_accvgpr_read_b32 v143, a93
	v_accvgpr_read_b32 v142, a92
	v_accvgpr_read_b32 v190, a244
	v_accvgpr_read_b32 v205, a19
	v_pk_mul_f32 v[4:5], v[200:201], s[28:29] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v6, a236
	v_accvgpr_read_b32 v47, a193
	v_accvgpr_read_b32 v92, a146
	v_accvgpr_read_b32 v93, a147
	v_accvgpr_read_b32 v98, a136
	v_accvgpr_read_b32 v139, a97
	v_accvgpr_read_b32 v138, a96
	v_accvgpr_read_b32 v191, a245
	v_accvgpr_read_b32 v247, a167
	v_accvgpr_read_b32 v204, a18
	v_accvgpr_read_b32 v199, a13
	v_accvgpr_read_b32 v198, a12
	v_accvgpr_write_b32 a13, v5
	v_pk_mul_f32 v[226:227], v[230:231], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[230:231], v[228:229], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[228:229], v[142:143], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[216:217], v[140:141], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[142:143], v[96:97], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[140:141], v[94:95], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[96:97], v[50:51], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[94:95], v[48:49], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[196:197], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[194:195], s[28:29] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v197, a11
	v_accvgpr_read_b32 v7, a237
	v_accvgpr_read_b32 v42, a196
	v_accvgpr_read_b32 v52, a190
	v_accvgpr_read_b32 v53, a191
	v_accvgpr_read_b32 v99, a137
	v_accvgpr_read_b32 v137, a103
	v_accvgpr_read_b32 v144, a94
	v_accvgpr_read_b32 v186, a248
	v_accvgpr_read_b32 v246, a166
	v_accvgpr_read_b32 v245, a165
	v_accvgpr_read_b32 v244, a164
	v_accvgpr_write_b32 a12, v4
	v_pk_mul_f32 v[4:5], v[198:199], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[198:199], v[204:205], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[218:219], v[190:191], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[190:191], v[138:139], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[138:139], v[92:93], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[92:93], v[46:47], s[28:29] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v196, a10
	v_accvgpr_read_b32 v44, a198
	v_accvgpr_read_b32 v45, a199
	v_accvgpr_read_b32 v91, a145
	v_accvgpr_read_b32 v136, a102
	v_accvgpr_read_b32 v188, a250
	v_accvgpr_read_b32 v189, a251
	v_accvgpr_read_b32 v203, a17
	v_accvgpr_read_b32 v202, a16
	v_accvgpr_write_b32 a14, v198
	v_pk_mul_f32 v[242:243], v[246:247], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[246:247], v[244:245], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[244:245], v[144:145], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[144:145], v[98:99], s[28:29] op_sel_hi:[1,0]
	s_waitcnt vmcnt(0)
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshlrev_b64 v[46:47], 1, v[0:1]
	v_lshrrev_b32_e32 v0, 2, v206
	v_pk_mul_f32 v[98:99], v[52:53], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[6:7], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[248:249], v[196:197], s[28:29] op_sel_hi:[1,0]
	v_and_b32_e32 v6, 12, v0
	v_and_b32_e32 v196, 15, v206
	v_accvgpr_read_b32 v211, a23
	v_accvgpr_write_b32 a15, v199
	v_pk_mul_f32 v[198:199], v[202:203], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[204:205], v[188:189], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[188:189], v[136:137], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[136:137], v[90:91], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[90:91], v[44:45], s[28:29] op_sel_hi:[1,0]
	v_lshl_or_b32 v44, s27, 8, v207
	v_mad_u64_u32 v[206:207], s[0:1], v6, s26, v[196:197]
	v_accvgpr_read_b32 v38, a200
	v_accvgpr_read_b32 v82, a152
	v_accvgpr_read_b32 v133, a107
	v_accvgpr_read_b32 v210, a22
	v_accvgpr_write_b32 a18, v198
	v_add_u32_e32 v252, s26, v206
	v_accvgpr_read_b32 v39, a201
	v_accvgpr_read_b32 v84, a154
	v_accvgpr_read_b32 v85, a155
	v_accvgpr_read_b32 v131, a105
	v_accvgpr_read_b32 v130, a104
	v_accvgpr_write_b32 a19, v199
	v_pk_mul_f32 v[198:199], v[210:211], s[28:29] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v195, a9
	v_accvgpr_read_b32 v194, a8
	v_ashrrev_i32_e32 v253, 31, v252
	v_accvgpr_read_b32 v209, a21
	v_accvgpr_read_b32 v208, a20
	v_accvgpr_write_b32 a20, v198
	v_pk_mul_f32 v[176:177], v[130:131], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[130:131], v[84:85], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[84:85], v[38:39], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[250:251], v[194:195], s[28:29] op_sel_hi:[1,0]
	v_mad_i64_i32 v[194:195], s[0:1], s26, v44, 0
	v_lshlrev_b64 v[38:39], 1, v[252:253]
	v_add_u32_e32 v252, s26, v252
	v_accvgpr_read_b32 v86, a148
	v_accvgpr_write_b32 a21, v199
	v_pk_mul_f32 v[198:199], v[208:209], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], s[28:29] op_sel_hi:[1,0]
	v_lshl_add_u64 v[194:195], v[194:195], 1, s[24:25]
	v_ashrrev_i32_e32 v207, 31, v206
	v_add_u32_e32 v0, s26, v252
	v_accvgpr_read_b32 v43, a197
	v_accvgpr_read_b32 v54, a184
	v_accvgpr_read_b32 v88, a150
	v_accvgpr_read_b32 v89, a151
	v_accvgpr_read_b32 v135, a101
	v_accvgpr_read_b32 v134, a100
	v_accvgpr_read_b32 v187, a249
	v_accvgpr_write_b32 a22, v198
	v_accvgpr_write_b32 a17, v3
	v_lshl_add_u64 v[194:195], v[194:195], 0, v[46:47]
	v_lshlrev_b64 v[206:207], 1, v[206:207]
	v_ashrrev_i32_e32 v253, 31, v252
	v_ashrrev_i32_e32 v1, 31, v0
	v_accvgpr_read_b32 v8, a238
	v_accvgpr_read_b32 v9, a239
	v_accvgpr_read_b32 v55, a185
	v_accvgpr_read_b32 v100, a138
	v_accvgpr_read_b32 v101, a139
	v_accvgpr_read_b32 v147, a89
	v_accvgpr_read_b32 v146, a88
	v_accvgpr_write_b32 a23, v199
	v_accvgpr_write_b32 a16, v2
	v_pk_mul_f32 v[198:199], v[186:187], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[186:187], v[134:135], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[134:135], v[88:89], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[88:89], v[42:43], s[28:29] op_sel_hi:[1,0]
	v_lshl_add_u64 v[254:255], v[194:195], 0, v[206:207]
	v_lshl_add_u64 v[2:3], v[194:195], 0, v[38:39]
	v_lshlrev_b64 v[252:253], 1, v[252:253]
	v_lshlrev_b64 v[42:43], 1, v[0:1]
	v_accvgpr_read_b32 v0, a12
	v_pk_mul_f32 v[210:211], v[214:215], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[214:215], v[212:213], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[212:213], v[146:147], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[146:147], v[100:101], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[100:101], v[54:55], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[8:9], s[28:29] op_sel_hi:[1,0]
	global_store_short_d16_hi v[254:255], v4, off
	global_store_short_d16_hi v[2:3], v5, off
	v_lshl_add_u64 v[4:5], v[194:195], 0, v[252:253]
	v_accvgpr_read_b32 v1, a13
	v_lshl_add_u64 v[8:9], v[194:195], 0, v[42:43]
	global_store_short_d16_hi v[4:5], v0, off
	global_store_short_d16_hi v[8:9], v1, off
	v_accvgpr_read_b32 v0, a18
	v_accvgpr_read_b32 v1, a19
	global_store_short_d16_hi v[254:255], v0, off offset:32
	global_store_short_d16_hi v[2:3], v1, off offset:32
	v_accvgpr_read_b32 v0, a14
	v_accvgpr_read_b32 v1, a15
	global_store_short_d16_hi v[4:5], v0, off offset:32
	global_store_short_d16_hi v[8:9], v1, off offset:32
	v_accvgpr_read_b32 v0, a22
	v_accvgpr_read_b32 v1, a23
	global_store_short_d16_hi v[254:255], v0, off offset:64
	global_store_short_d16_hi v[2:3], v1, off offset:64
	v_accvgpr_read_b32 v0, a20
	v_accvgpr_read_b32 v1, a21
	global_store_short_d16_hi v[4:5], v0, off offset:64
	global_store_short_d16_hi v[8:9], v1, off offset:64
	global_store_short_d16_hi v[254:255], v214, off offset:96
	global_store_short_d16_hi v[2:3], v215, off offset:96
	global_store_short_d16_hi v[4:5], v210, off offset:96
	global_store_short_d16_hi v[8:9], v211, off offset:96
	v_or_b32_e32 v0, 16, v6
	v_mad_u64_u32 v[210:211], s[0:1], v0, s26, v[196:197]
	v_accvgpr_read_b32 v14, a228
	v_accvgpr_read_b32 v34, a204
	v_accvgpr_read_b32 v58, a180
	v_accvgpr_read_b32 v106, a128
	v_accvgpr_read_b32 v129, a111
	v_accvgpr_read_b32 v153, a87
	v_ashrrev_i32_e32 v211, 31, v210
	v_add_u32_e32 v214, s26, v210
	v_accvgpr_read_b32 v15, a229
	v_accvgpr_read_b32 v18, a224
	v_accvgpr_read_b32 v36, a206
	v_accvgpr_read_b32 v37, a207
	v_accvgpr_read_b32 v60, a182
	v_accvgpr_read_b32 v61, a183
	v_accvgpr_read_b32 v62, a176
	v_accvgpr_read_b32 v83, a153
	v_accvgpr_read_b32 v107, a129
	v_accvgpr_read_b32 v110, a124
	v_accvgpr_read_b32 v128, a110
	v_accvgpr_read_b32 v152, a86
	v_accvgpr_read_b32 v157, a83
	v_lshlrev_b64 v[210:211], 1, v[210:211]
	v_ashrrev_i32_e32 v215, 31, v214
	v_accvgpr_read_b32 v19, a225
	v_accvgpr_read_b32 v64, a178
	v_accvgpr_read_b32 v65, a179
	v_accvgpr_read_b32 v111, a125
	v_accvgpr_read_b32 v156, a82
	v_pk_mul_f32 v[220:221], v[152:153], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[174:175], v[128:129], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[152:153], v[106:107], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[128:129], v[82:83], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[106:107], v[60:61], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[82:83], v[36:37], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[14:15], s[28:29] op_sel_hi:[1,0]
	v_lshl_add_u64 v[14:15], v[194:195], 0, v[210:211]
	v_accvgpr_read_b32 v0, a24
	v_lshlrev_b64 v[36:37], 1, v[214:215]
	v_add_u32_e32 v214, s26, v214
	v_pk_mul_f32 v[208:209], v[156:157], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[156:157], v[110:111], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[110:111], v[64:65], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[64:65], v[18:19], s[28:29] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v1, a25
	global_store_short_d16_hi v[14:15], v0, off
	v_lshl_add_u64 v[18:19], v[194:195], 0, v[36:37]
	v_add_u32_e32 v0, s26, v214
	v_accvgpr_read_b32 v10, a232
	v_accvgpr_read_b32 v40, a202
	v_accvgpr_read_b32 v41, a203
	v_accvgpr_read_b32 v87, a149
	v_accvgpr_read_b32 v102, a132
	v_accvgpr_read_b32 v132, a106
	global_store_short_d16_hi v[18:19], v1, off
	v_ashrrev_i32_e32 v215, 31, v214
	v_ashrrev_i32_e32 v1, 31, v0
	v_accvgpr_read_b32 v11, a233
	v_accvgpr_read_b32 v12, a234
	v_accvgpr_read_b32 v13, a235
	v_accvgpr_read_b32 v56, a186
	v_accvgpr_read_b32 v57, a187
	v_accvgpr_read_b32 v59, a181
	v_accvgpr_read_b32 v103, a133
	v_accvgpr_read_b32 v104, a134
	v_accvgpr_read_b32 v105, a135
	v_accvgpr_read_b32 v151, a85
	v_accvgpr_read_b32 v150, a84
	v_pk_mul_f32 v[180:181], v[132:133], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[132:133], v[86:87], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[86:87], v[40:41], s[28:29] op_sel_hi:[1,0]
	v_lshlrev_b64 v[214:215], 1, v[214:215]
	v_lshlrev_b64 v[40:41], 1, v[0:1]
	v_accvgpr_read_b32 v0, a16
	v_pk_mul_f32 v[200:201], v[150:151], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[150:151], v[104:105], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[148:149], v[102:103], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[104:105], v[58:59], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[102:103], v[56:57], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[12:13], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[10:11], s[28:29] op_sel_hi:[1,0]
	v_lshl_add_u64 v[10:11], v[194:195], 0, v[214:215]
	v_accvgpr_read_b32 v1, a17
	v_lshl_add_u64 v[12:13], v[194:195], 0, v[40:41]
	global_store_short_d16_hi v[10:11], v0, off
	global_store_short_d16_hi v[12:13], v1, off
	v_accvgpr_read_b32 v0, a30
	v_accvgpr_read_b32 v1, a31
	global_store_short_d16_hi v[14:15], v0, off offset:32
	global_store_short_d16_hi v[18:19], v1, off offset:32
	v_accvgpr_read_b32 v0, a26
	v_accvgpr_read_b32 v1, a27
	global_store_short_d16_hi v[10:11], v0, off offset:32
	global_store_short_d16_hi v[12:13], v1, off offset:32
	v_accvgpr_read_b32 v0, a34
	v_accvgpr_read_b32 v1, a35
	global_store_short_d16_hi v[14:15], v0, off offset:64
	global_store_short_d16_hi v[18:19], v1, off offset:64
	v_accvgpr_read_b32 v0, a32
	v_accvgpr_read_b32 v1, a33
	global_store_short_d16_hi v[10:11], v0, off offset:64
	global_store_short_d16_hi v[12:13], v1, off offset:64
	global_store_short_d16_hi v[14:15], v230, off offset:96
	global_store_short_d16_hi v[18:19], v231, off offset:96
	global_store_short_d16_hi v[10:11], v226, off offset:96
	global_store_short_d16_hi v[12:13], v227, off offset:96
	v_or_b32_e32 v0, 32, v6
	v_mad_u64_u32 v[226:227], s[0:1], v0, s26, v[196:197]
	v_accvgpr_read_b32 v26, a212
	v_accvgpr_read_b32 v30, a208
	v_accvgpr_read_b32 v70, a168
	v_accvgpr_read_b32 v78, a156
	v_accvgpr_read_b32 v121, a119
	v_accvgpr_read_b32 v125, a115
	v_accvgpr_read_b32 v173, a63
	v_ashrrev_i32_e32 v227, 31, v226
	v_add_u32_e32 v230, s26, v226
	v_accvgpr_read_b32 v22, a220
	v_accvgpr_read_b32 v27, a213
	v_accvgpr_read_b32 v32, a210
	v_accvgpr_read_b32 v33, a211
	v_accvgpr_read_b32 v72, a170
	v_accvgpr_read_b32 v73, a171
	v_accvgpr_read_b32 v79, a157
	v_accvgpr_read_b32 v114, a120
	v_accvgpr_read_b32 v119, a117
	v_accvgpr_read_b32 v118, a116
	v_accvgpr_read_b32 v124, a114
	v_accvgpr_read_b32 v171, a61
	v_accvgpr_read_b32 v170, a60
	v_lshlrev_b64 v[226:227], 1, v[226:227]
	v_ashrrev_i32_e32 v231, 31, v230
	v_accvgpr_read_b32 v24, a222
	v_accvgpr_read_b32 v25, a223
	v_accvgpr_read_b32 v71, a169
	v_accvgpr_read_b32 v116, a122
	v_accvgpr_read_b32 v117, a123
	v_accvgpr_read_b32 v163, a69
	v_accvgpr_read_b32 v162, a68
	v_pk_mul_f32 v[222:223], v[170:171], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[170:171], v[124:125], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[164:165], v[118:119], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[124:125], v[78:79], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[118:119], v[72:73], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[78:79], v[32:33], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[72:73], v[26:27], s[28:29] op_sel_hi:[1,0]
	v_lshl_add_u64 v[26:27], v[194:195], 0, v[226:227]
	v_accvgpr_read_b32 v0, a36
	v_lshlrev_b64 v[32:33], 1, v[230:231]
	v_add_u32_e32 v230, s26, v230
	v_pk_mul_f32 v[232:233], v[162:163], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[162:163], v[116:117], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[116:117], v[70:71], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[70:71], v[24:25], s[28:29] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v1, a37
	global_store_short_d16_hi v[26:27], v0, off
	v_lshl_add_u64 v[24:25], v[194:195], 0, v[32:33]
	v_add_u32_e32 v0, s26, v230
	v_accvgpr_read_b32 v35, a205
	v_accvgpr_read_b32 v66, a172
	v_accvgpr_read_b32 v80, a158
	v_accvgpr_read_b32 v81, a159
	v_accvgpr_read_b32 v127, a109
	v_accvgpr_read_b32 v126, a108
	v_accvgpr_read_b32 v161, a75
	v_accvgpr_read_b32 v172, a62
	global_store_short_d16_hi v[24:25], v1, off
	v_ashrrev_i32_e32 v231, 31, v230
	v_ashrrev_i32_e32 v1, 31, v0
	v_accvgpr_read_b32 v20, a226
	v_accvgpr_read_b32 v21, a227
	v_accvgpr_read_b32 v23, a221
	v_accvgpr_read_b32 v67, a173
	v_accvgpr_read_b32 v68, a174
	v_accvgpr_read_b32 v69, a175
	v_accvgpr_read_b32 v112, a126
	v_accvgpr_read_b32 v113, a127
	v_accvgpr_read_b32 v115, a121
	v_accvgpr_read_b32 v160, a74
	v_accvgpr_read_b32 v159, a73
	v_accvgpr_read_b32 v158, a72
	v_pk_mul_f32 v[238:239], v[172:173], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[172:173], v[126:127], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[126:127], v[80:81], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[80:81], v[34:35], s[28:29] op_sel_hi:[1,0]
	v_lshlrev_b64 v[230:231], 1, v[230:231]
	v_lshlrev_b64 v[34:35], 1, v[0:1]
	v_accvgpr_read_b32 v0, a28
	v_pk_mul_f32 v[240:241], v[160:161], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[224:225], v[158:159], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[160:161], v[114:115], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[158:159], v[112:113], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[114:115], v[68:69], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[112:113], v[66:67], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[68:69], v[22:23], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[66:67], v[20:21], s[28:29] op_sel_hi:[1,0]
	v_lshl_add_u64 v[22:23], v[194:195], 0, v[230:231]
	v_accvgpr_read_b32 v1, a29
	v_lshl_add_u64 v[20:21], v[194:195], 0, v[34:35]
	global_store_short_d16_hi v[22:23], v0, off
	global_store_short_d16_hi v[20:21], v1, off
	v_accvgpr_read_b32 v0, a42
	v_accvgpr_read_b32 v1, a43
	global_store_short_d16_hi v[26:27], v0, off offset:32
	global_store_short_d16_hi v[24:25], v1, off offset:32
	v_accvgpr_read_b32 v0, a38
	v_accvgpr_read_b32 v1, a39
	global_store_short_d16_hi v[22:23], v0, off offset:32
	global_store_short_d16_hi v[20:21], v1, off offset:32
	v_accvgpr_read_b32 v0, a46
	v_accvgpr_read_b32 v1, a47
	global_store_short_d16_hi v[26:27], v0, off offset:64
	global_store_short_d16_hi v[24:25], v1, off offset:64
	v_accvgpr_read_b32 v0, a44
	v_accvgpr_read_b32 v1, a45
	global_store_short_d16_hi v[22:23], v0, off offset:64
	global_store_short_d16_hi v[20:21], v1, off offset:64
	global_store_short_d16_hi v[26:27], v246, off offset:96
	global_store_short_d16_hi v[24:25], v247, off offset:96
	global_store_short_d16_hi v[22:23], v242, off offset:96
	global_store_short_d16_hi v[20:21], v243, off offset:96
	v_or_b32_e32 v0, 48, v6
	v_mad_u64_u32 v[196:197], s[0:1], v0, s26, v[196:197]
	v_accvgpr_read_b32 v74, a160
	v_accvgpr_read_b32 v169, a67
	v_ashrrev_i32_e32 v197, 31, v196
	v_add_u32_e32 v242, s26, v196
	v_accvgpr_read_b32 v28, a214
	v_accvgpr_read_b32 v29, a215
	v_accvgpr_read_b32 v75, a161
	v_accvgpr_read_b32 v120, a118
	v_accvgpr_read_b32 v167, a65
	v_accvgpr_read_b32 v166, a64
	v_lshlrev_b64 v[196:197], 1, v[196:197]
	v_ashrrev_i32_e32 v243, 31, v242
	v_accvgpr_read_b32 v16, a230
	v_accvgpr_read_b32 v17, a231
	v_accvgpr_read_b32 v63, a177
	v_accvgpr_read_b32 v108, a130
	v_accvgpr_read_b32 v109, a131
	v_accvgpr_read_b32 v155, a81
	v_accvgpr_read_b32 v154, a80
	v_accvgpr_read_b32 v192, a246
	v_accvgpr_read_b32 v193, a247
	v_pk_mul_f32 v[202:203], v[166:167], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[166:167], v[120:121], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[120:121], v[74:75], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[74:75], v[28:29], s[28:29] op_sel_hi:[1,0]
	v_lshl_add_u64 v[246:247], v[194:195], 0, v[196:197]
	v_accvgpr_read_b32 v0, a48
	v_lshlrev_b64 v[28:29], 1, v[242:243]
	v_add_u32_e32 v242, s26, v242
	v_pk_mul_f32 v[234:235], v[192:193], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[192:193], v[154:155], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[154:155], v[108:109], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[108:109], v[62:63], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[62:63], v[16:17], s[28:29] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v1, a49
	global_store_short_d16_hi v[246:247], v0, off
	v_lshl_add_u64 v[16:17], v[194:195], 0, v[28:29]
	v_add_u32_e32 v0, s26, v242
	v_accvgpr_read_b32 v31, a209
	v_accvgpr_read_b32 v76, a162
	v_accvgpr_read_b32 v77, a163
	v_accvgpr_read_b32 v123, a113
	v_accvgpr_read_b32 v122, a112
	v_accvgpr_read_b32 v168, a66
	global_store_short_d16_hi v[16:17], v1, off
	v_ashrrev_i32_e32 v243, 31, v242
	v_ashrrev_i32_e32 v1, 31, v0
	v_accvgpr_read_b32 v182, a252
	v_pk_mul_f32 v[236:237], v[168:169], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[168:169], v[122:123], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[122:123], v[76:77], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[76:77], v[30:31], s[28:29] op_sel_hi:[1,0]
	v_lshlrev_b64 v[242:243], 1, v[242:243]
	v_lshlrev_b64 v[30:31], 1, v[0:1]
	v_accvgpr_mov_b32 a8, a40
	v_accvgpr_read_b32 v183, a253
	v_accvgpr_read_b32 v184, a254
	v_accvgpr_read_b32 v185, a255
	v_lshl_add_u64 v[6:7], v[194:195], 0, v[242:243]
	v_accvgpr_mov_b32 a9, a41
	v_lshl_add_u64 v[0:1], v[194:195], 0, v[30:31]
	v_or_b32_e32 v44, 0x80, v44
	v_pk_mul_f32 v[184:185], v[184:185], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[182:183], v[182:183], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[178:179], v[178:179], s[28:29] op_sel_hi:[1,0]
	global_store_short_d16_hi v[6:7], a8, off
	global_store_short_d16_hi v[0:1], a9, off
	global_store_short_d16_hi v[246:247], v218, off offset:32
	global_store_short_d16_hi v[16:17], v219, off offset:32
	global_store_short_d16_hi v[6:7], v234, off offset:32
	global_store_short_d16_hi v[0:1], v235, off offset:32
	global_store_short_d16_hi v[246:247], v198, off offset:64
	global_store_short_d16_hi v[16:17], v199, off offset:64
	global_store_short_d16_hi v[6:7], v204, off offset:64
	global_store_short_d16_hi v[0:1], v205, off offset:64
	global_store_short_d16_hi v[246:247], v182, off offset:96
	global_store_short_d16_hi v[16:17], v183, off offset:96
	global_store_short_d16_hi v[6:7], v184, off offset:96
	global_store_short_d16_hi v[0:1], v185, off offset:96
	global_store_short_d16_hi v[254:255], v178, off offset:256
	global_store_short_d16_hi v[2:3], v179, off offset:256
	v_mad_i64_i32 v[198:199], s[0:1], s26, v44, 0
	v_lshl_add_u64 v[198:199], v[198:199], 1, s[24:25]
	v_lshl_add_u64 v[44:45], v[198:199], 0, v[46:47]
	v_accvgpr_read_b32 v46, a50
	v_accvgpr_read_b32 v47, a51
	s_mov_b64 s[0:1], 0x100
	global_store_short_d16_hi v[4:5], v46, off offset:256
	v_lshl_add_u64 v[4:5], v[194:195], 0, s[0:1]
	global_store_short_d16_hi v[8:9], v47, off offset:256
	v_accvgpr_read_b32 v46, a54
	v_lshl_add_u64 v[8:9], v[4:5], 0, v[206:207]
	v_accvgpr_read_b32 v47, a55
	v_lshl_add_u64 v[194:195], v[4:5], 0, v[38:39]
	global_store_short_d16_hi v[8:9], v46, off offset:32
	global_store_short_d16_hi v[194:195], v47, off offset:32
	v_accvgpr_read_b32 v46, a52
	v_accvgpr_read_b32 v185, a7
	v_lshl_add_u64 v[198:199], v[4:5], 0, v[252:253]
	v_accvgpr_read_b32 v47, a53
	v_lshl_add_u64 v[204:205], v[4:5], 0, v[42:43]
	v_accvgpr_read_b32 v184, a6
	v_accvgpr_read_b32 v183, a5
	v_accvgpr_read_b32 v182, a4
	global_store_short_d16_hi v[198:199], v46, off offset:32
	global_store_short_d16_hi v[204:205], v47, off offset:32
	global_store_short_d16_hi v[8:9], v222, off offset:64
	global_store_short_d16_hi v[194:195], v223, off offset:64
	v_accvgpr_read_b32 v46, a58
	v_pk_mul_f32 v[2:3], v[184:185], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[178:179], v[182:183], s[28:29] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v185, a3
	global_store_short_d16_hi v[198:199], v238, off offset:64
	global_store_short_d16_hi v[204:205], v239, off offset:64
	global_store_short_d16_hi v[8:9], v202, off offset:96
	v_lshl_add_u64 v[8:9], v[4:5], 0, v[210:211]
	v_accvgpr_read_b32 v47, a59
	v_accvgpr_read_b32 v184, a2
	v_accvgpr_read_b32 v183, a1
	v_accvgpr_read_b32 v182, a0
	v_lshl_add_u64 v[222:223], v[44:45], 0, s[0:1]
	global_store_short_d16_hi v[194:195], v203, off offset:96
	global_store_short_d16_hi v[198:199], v236, off offset:96
	global_store_short_d16_hi v[204:205], v237, off offset:96
	global_store_short_d16_hi v[14:15], v232, off offset:256
	global_store_short_d16_hi v[18:19], v233, off offset:256
	v_lshl_add_u64 v[18:19], v[4:5], 0, v[36:37]
	global_store_short_d16_hi v[10:11], v46, off offset:256
	global_store_short_d16_hi v[12:13], v47, off offset:256
	v_lshl_add_u64 v[236:237], v[4:5], 0, v[214:215]
	global_store_short_d16_hi v[8:9], v224, off offset:32
	global_store_short_d16_hi v[18:19], v225, off offset:32
	v_lshl_add_u64 v[224:225], v[4:5], 0, v[40:41]
	v_accvgpr_read_b32 v46, a56
	v_pk_mul_f32 v[184:185], v[184:185], s[28:29] op_sel_hi:[1,0]
	v_pk_mul_f32 v[182:183], v[182:183], s[28:29] op_sel_hi:[1,0]
	v_lshl_add_u64 v[218:219], v[44:45], 0, v[206:207]
	v_lshl_add_u64 v[194:195], v[4:5], 0, v[226:227]
	v_lshl_add_u64 v[202:203], v[4:5], 0, v[196:197]
	v_lshl_add_u64 v[198:199], v[44:45], 0, v[38:39]
	v_lshl_add_u64 v[204:205], v[44:45], 0, v[210:211]
	v_lshl_add_u64 v[234:235], v[44:45], 0, v[226:227]
	v_lshl_add_u64 v[14:15], v[44:45], 0, v[196:197]
	v_lshl_add_u64 v[206:207], v[222:223], 0, v[206:207]
	v_lshl_add_u64 v[38:39], v[222:223], 0, v[38:39]
	v_lshl_add_u64 v[210:211], v[222:223], 0, v[210:211]
	v_lshl_add_u64 v[226:227], v[222:223], 0, v[226:227]
	v_lshl_add_u64 v[196:197], v[222:223], 0, v[196:197]
	v_lshl_add_u64 v[232:233], v[4:5], 0, v[32:33]
	v_lshl_add_u64 v[10:11], v[4:5], 0, v[28:29]
	v_lshl_add_u64 v[12:13], v[44:45], 0, v[252:253]
	global_store_short_d16_hi v[236:237], v240, off offset:32
	global_store_short_d16_hi v[224:225], v241, off offset:32
	v_lshl_add_u64 v[238:239], v[44:45], 0, v[36:37]
	global_store_short_d16_hi v[8:9], v192, off offset:64
	global_store_short_d16_hi v[18:19], v193, off offset:64
	v_lshl_add_u64 v[192:193], v[44:45], 0, v[32:33]
	global_store_short_d16_hi v[236:237], v208, off offset:64
	global_store_short_d16_hi v[224:225], v209, off offset:64
	v_lshl_add_u64 v[208:209], v[44:45], 0, v[28:29]
	v_lshl_add_u64 v[240:241], v[222:223], 0, v[252:253]
	v_lshl_add_u64 v[36:37], v[222:223], 0, v[36:37]
	v_lshl_add_u64 v[32:33], v[222:223], 0, v[32:33]
	v_lshl_add_u64 v[28:29], v[222:223], 0, v[28:29]
	global_store_short_d16_hi v[8:9], v200, off offset:96
	v_lshl_add_u64 v[8:9], v[4:5], 0, v[230:231]
	global_store_short_d16_hi v[18:19], v201, off offset:96
	v_lshl_add_u64 v[18:19], v[4:5], 0, v[242:243]
	v_lshl_add_u64 v[200:201], v[44:45], 0, v[42:43]
	global_store_short_d16_hi v[236:237], v220, off offset:96
	v_lshl_add_u64 v[236:237], v[44:45], 0, v[214:215]
	global_store_short_d16_hi v[224:225], v221, off offset:96
	v_lshl_add_u64 v[220:221], v[44:45], 0, v[230:231]
	v_lshl_add_u64 v[224:225], v[44:45], 0, v[242:243]
	v_lshl_add_u64 v[42:43], v[222:223], 0, v[42:43]
	v_lshl_add_u64 v[214:215], v[222:223], 0, v[214:215]
	v_lshl_add_u64 v[230:231], v[222:223], 0, v[230:231]
	v_lshl_add_u64 v[242:243], v[222:223], 0, v[242:243]
	global_store_short_d16_hi v[26:27], v212, off offset:256
	v_lshl_add_u64 v[26:27], v[4:5], 0, v[34:35]
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[30:31]
	global_store_short_d16_hi v[24:25], v213, off offset:256
	v_lshl_add_u64 v[24:25], v[44:45], 0, v[40:41]
	v_lshl_add_u64 v[212:213], v[44:45], 0, v[34:35]
	v_lshl_add_u64 v[44:45], v[44:45], 0, v[30:31]
	v_lshl_add_u64 v[40:41], v[222:223], 0, v[40:41]
	v_lshl_add_u64 v[34:35], v[222:223], 0, v[34:35]
	v_lshl_add_u64 v[30:31], v[222:223], 0, v[30:31]
	v_accvgpr_read_b32 v47, a57
	global_store_short_d16_hi v[22:23], v46, off offset:256
	global_store_short_d16_hi v[20:21], v47, off offset:256
	global_store_short_d16_hi v[194:195], v228, off offset:32
	global_store_short_d16_hi v[232:233], v229, off offset:32
	global_store_short_d16_hi v[8:9], v244, off offset:32
	global_store_short_d16_hi v[26:27], v245, off offset:32
	global_store_short_d16_hi v[194:195], v190, off offset:64
	global_store_short_d16_hi v[232:233], v191, off offset:64
	global_store_short_d16_hi v[8:9], v216, off offset:64
	global_store_short_d16_hi v[26:27], v217, off offset:64
	global_store_short_d16_hi v[194:195], v186, off offset:96
	global_store_short_d16_hi v[232:233], v187, off offset:96
	global_store_short_d16_hi v[8:9], v188, off offset:96
	global_store_short_d16_hi v[26:27], v189, off offset:96
	global_store_short_d16_hi v[246:247], v176, off offset:256
	global_store_short_d16_hi v[16:17], v177, off offset:256
	global_store_short_d16_hi v[6:7], v180, off offset:256
	global_store_short_d16_hi v[0:1], v181, off offset:256
	global_store_short_d16_hi v[202:203], v172, off offset:32
	global_store_short_d16_hi v[10:11], v173, off offset:32
	global_store_short_d16_hi v[18:19], v174, off offset:32
	global_store_short_d16_hi v[4:5], v175, off offset:32
	global_store_short_d16_hi v[202:203], v168, off offset:64
	global_store_short_d16_hi v[10:11], v169, off offset:64
	global_store_short_d16_hi v[18:19], v170, off offset:64
	global_store_short_d16_hi v[4:5], v171, off offset:64
	global_store_short_d16_hi v[202:203], v164, off offset:96
	global_store_short_d16_hi v[10:11], v165, off offset:96
	global_store_short_d16_hi v[18:19], v166, off offset:96
	global_store_short_d16_hi v[4:5], v167, off offset:96
	global_store_short_d16_hi v[218:219], v160, off
	global_store_short_d16_hi v[198:199], v161, off
	global_store_short_d16_hi v[12:13], v162, off
	global_store_short_d16_hi v[200:201], v163, off
	global_store_short_d16_hi v[218:219], v156, off offset:32
	global_store_short_d16_hi v[198:199], v157, off offset:32
	global_store_short_d16_hi v[12:13], v158, off offset:32
	global_store_short_d16_hi v[200:201], v159, off offset:32
	global_store_short_d16_hi v[218:219], v152, off offset:64
	global_store_short_d16_hi v[198:199], v153, off offset:64
	global_store_short_d16_hi v[12:13], v154, off offset:64
	global_store_short_d16_hi v[200:201], v155, off offset:64
	global_store_short_d16_hi v[218:219], v148, off offset:96
	global_store_short_d16_hi v[198:199], v149, off offset:96
	global_store_short_d16_hi v[12:13], v150, off offset:96
	global_store_short_d16_hi v[200:201], v151, off offset:96
	global_store_short_d16_hi v[204:205], v144, off
	global_store_short_d16_hi v[238:239], v145, off
	global_store_short_d16_hi v[236:237], v146, off
	global_store_short_d16_hi v[24:25], v147, off
	global_store_short_d16_hi v[204:205], v140, off offset:32
	global_store_short_d16_hi v[238:239], v141, off offset:32
	global_store_short_d16_hi v[236:237], v142, off offset:32
	global_store_short_d16_hi v[24:25], v143, off offset:32
	global_store_short_d16_hi v[204:205], v136, off offset:64
	global_store_short_d16_hi v[238:239], v137, off offset:64
	global_store_short_d16_hi v[236:237], v138, off offset:64
	global_store_short_d16_hi v[24:25], v139, off offset:64
	global_store_short_d16_hi v[204:205], v132, off offset:96
	global_store_short_d16_hi v[238:239], v133, off offset:96
	global_store_short_d16_hi v[236:237], v134, off offset:96
	global_store_short_d16_hi v[24:25], v135, off offset:96
	global_store_short_d16_hi v[234:235], v128, off
	global_store_short_d16_hi v[192:193], v129, off
	global_store_short_d16_hi v[220:221], v130, off
	global_store_short_d16_hi v[212:213], v131, off
	global_store_short_d16_hi v[234:235], v124, off offset:32
	global_store_short_d16_hi v[192:193], v125, off offset:32
	global_store_short_d16_hi v[220:221], v126, off offset:32
	global_store_short_d16_hi v[212:213], v127, off offset:32
	global_store_short_d16_hi v[234:235], v120, off offset:64
	global_store_short_d16_hi v[192:193], v121, off offset:64
	global_store_short_d16_hi v[220:221], v122, off offset:64
	global_store_short_d16_hi v[212:213], v123, off offset:64
	global_store_short_d16_hi v[234:235], v116, off offset:96
	global_store_short_d16_hi v[192:193], v117, off offset:96
	global_store_short_d16_hi v[220:221], v118, off offset:96
	global_store_short_d16_hi v[212:213], v119, off offset:96
	global_store_short_d16_hi v[14:15], v112, off
	global_store_short_d16_hi v[208:209], v113, off
	global_store_short_d16_hi v[224:225], v114, off
	global_store_short_d16_hi v[44:45], v115, off
	global_store_short_d16_hi v[14:15], v108, off offset:32
	global_store_short_d16_hi v[208:209], v109, off offset:32
	global_store_short_d16_hi v[224:225], v110, off offset:32
	global_store_short_d16_hi v[44:45], v111, off offset:32
	global_store_short_d16_hi v[14:15], v104, off offset:64
	global_store_short_d16_hi v[208:209], v105, off offset:64
	global_store_short_d16_hi v[224:225], v106, off offset:64
	global_store_short_d16_hi v[44:45], v107, off offset:64
	global_store_short_d16_hi v[14:15], v100, off offset:96
	global_store_short_d16_hi v[208:209], v101, off offset:96
	global_store_short_d16_hi v[224:225], v102, off offset:96
	global_store_short_d16_hi v[44:45], v103, off offset:96
	global_store_short_d16_hi v[218:219], v96, off offset:256
	global_store_short_d16_hi v[198:199], v97, off offset:256
	global_store_short_d16_hi v[12:13], v98, off offset:256
	global_store_short_d16_hi v[200:201], v99, off offset:256
	global_store_short_d16_hi v[206:207], v92, off offset:32
	global_store_short_d16_hi v[38:39], v93, off offset:32
	global_store_short_d16_hi v[240:241], v94, off offset:32
	global_store_short_d16_hi v[42:43], v95, off offset:32
	global_store_short_d16_hi v[206:207], v88, off offset:64
	global_store_short_d16_hi v[38:39], v89, off offset:64
	global_store_short_d16_hi v[240:241], v90, off offset:64
	global_store_short_d16_hi v[42:43], v91, off offset:64
	global_store_short_d16_hi v[206:207], v84, off offset:96
	global_store_short_d16_hi v[38:39], v85, off offset:96
	global_store_short_d16_hi v[240:241], v86, off offset:96
	global_store_short_d16_hi v[42:43], v87, off offset:96
	global_store_short_d16_hi v[204:205], v80, off offset:256
	global_store_short_d16_hi v[238:239], v81, off offset:256
	global_store_short_d16_hi v[236:237], v82, off offset:256
	global_store_short_d16_hi v[24:25], v83, off offset:256
	global_store_short_d16_hi v[210:211], v76, off offset:32
	global_store_short_d16_hi v[36:37], v77, off offset:32
	global_store_short_d16_hi v[214:215], v78, off offset:32
	global_store_short_d16_hi v[40:41], v79, off offset:32
	global_store_short_d16_hi v[210:211], v72, off offset:64
	global_store_short_d16_hi v[36:37], v73, off offset:64
	global_store_short_d16_hi v[214:215], v74, off offset:64
	global_store_short_d16_hi v[40:41], v75, off offset:64
	global_store_short_d16_hi v[210:211], v68, off offset:96
	global_store_short_d16_hi v[36:37], v69, off offset:96
	global_store_short_d16_hi v[214:215], v70, off offset:96
	global_store_short_d16_hi v[40:41], v71, off offset:96
	global_store_short_d16_hi v[234:235], v64, off offset:256
	global_store_short_d16_hi v[192:193], v65, off offset:256
	global_store_short_d16_hi v[220:221], v66, off offset:256
	global_store_short_d16_hi v[212:213], v67, off offset:256
	global_store_short_d16_hi v[226:227], v60, off offset:32
	global_store_short_d16_hi v[32:33], v61, off offset:32
	global_store_short_d16_hi v[230:231], v62, off offset:32
	global_store_short_d16_hi v[34:35], v63, off offset:32
	global_store_short_d16_hi v[226:227], v56, off offset:64
	global_store_short_d16_hi v[32:33], v57, off offset:64
	global_store_short_d16_hi v[230:231], v58, off offset:64
	global_store_short_d16_hi v[34:35], v59, off offset:64
	global_store_short_d16_hi v[226:227], v52, off offset:96
	global_store_short_d16_hi v[32:33], v53, off offset:96
	global_store_short_d16_hi v[230:231], v54, off offset:96
	global_store_short_d16_hi v[34:35], v55, off offset:96
	global_store_short_d16_hi v[14:15], v48, off offset:256
	global_store_short_d16_hi v[208:209], v49, off offset:256
	global_store_short_d16_hi v[224:225], v50, off offset:256
	global_store_short_d16_hi v[44:45], v51, off offset:256
	global_store_short_d16_hi v[196:197], v250, off offset:32
	global_store_short_d16_hi v[28:29], v251, off offset:32
	global_store_short_d16_hi v[242:243], v248, off offset:32
	global_store_short_d16_hi v[30:31], v249, off offset:32
	global_store_short_d16_hi v[196:197], v178, off offset:64
	global_store_short_d16_hi v[28:29], v179, off offset:64
	global_store_short_d16_hi v[242:243], v2, off offset:64
	global_store_short_d16_hi v[30:31], v3, off offset:64
	global_store_short_d16_hi v[196:197], v182, off offset:96
	global_store_short_d16_hi v[28:29], v183, off offset:96
	global_store_short_d16_hi v[242:243], v184, off offset:96
	global_store_short_d16_hi v[30:31], v185, off offset:96
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z22mxfp4_gluon_cpp_kernel13gluon_globals
		.amdhsa_group_segment_fixed_size 131072
		.amdhsa_private_segment_fixed_size 12
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
		.amdhsa_enable_private_segment 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 512
		.amdhsa_next_free_sgpr 96
		.amdhsa_accum_offset 256
		.amdhsa_reserve_vcc 0
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
	.size	_Z22mxfp4_gluon_cpp_kernel13gluon_globals, .Lfunc_end0-_Z22mxfp4_gluon_cpp_kernel13gluon_globals
                                        ; -- End function
	.set _Z22mxfp4_gluon_cpp_kernel13gluon_globals.num_vgpr, 256
	.set _Z22mxfp4_gluon_cpp_kernel13gluon_globals.num_agpr, 256
	.set _Z22mxfp4_gluon_cpp_kernel13gluon_globals.numbered_sgpr, 85
	.set _Z22mxfp4_gluon_cpp_kernel13gluon_globals.private_seg_size, 12
	.set _Z22mxfp4_gluon_cpp_kernel13gluon_globals.uses_vcc, 0
	.set _Z22mxfp4_gluon_cpp_kernel13gluon_globals.uses_flat_scratch, 0
	.set _Z22mxfp4_gluon_cpp_kernel13gluon_globals.has_dyn_sized_stack, 0
	.set _Z22mxfp4_gluon_cpp_kernel13gluon_globals.has_recursion, 0
	.set _Z22mxfp4_gluon_cpp_kernel13gluon_globals.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 35124
; TotalNumSgprs: 91
; NumVgprs: 256
; NumAgprs: 256
; TotalNumVgprs: 512
; ScratchSize: 12
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 131072 bytes/workgroup (compile time only)
; SGPRBlocks: 12
; VGPRBlocks: 63
; NumSGPRsForWavesPerEU: 102
; NumVGPRsForWavesPerEU: 512
; AccumOffset: 256
; Occupancy: 1
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 1
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
	.type	__hip_cuid_49d55448659c59f7,@object ; @__hip_cuid_49d55448659c59f7
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_49d55448659c59f7
__hip_cuid_49d55448659c59f7:
	.byte	0                               ; 0x0
	.size	__hip_cuid_49d55448659c59f7, 1

	.ident	"clang version 20.0.0git (https://github.com/ROCm/llvm-project.git 27682a16360e33e37c4f3cc6adf9a620733f8fe1)"
	.ident	"AMD clang version 20.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.1.0 25425 1b0eada6b0ee93e2e694c8c146d23fca90bc11c5)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_49d55448659c59f7
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
    .group_segment_fixed_size: 131072
    .kernarg_segment_align: 8
    .kernarg_segment_size: 504
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _Z22mxfp4_gluon_cpp_kernel13gluon_globals
    .private_segment_fixed_size: 12
    .sgpr_count:     91
    .sgpr_spill_count: 0
    .symbol:         _Z22mxfp4_gluon_cpp_kernel13gluon_globals.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     512
    .vgpr_spill_count: 2
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
