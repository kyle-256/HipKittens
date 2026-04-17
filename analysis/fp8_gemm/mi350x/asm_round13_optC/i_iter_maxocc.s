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
	s_waitcnt lgkmcnt(0)
	s_add_i32 s3, s6, 7
	s_ashr_i32 s4, s3, 31
	s_ashr_i32 s10, s6, 31
	s_lshr_b32 s4, s4, 29
	s_lshr_b32 s5, s10, 29
	s_add_i32 s3, s3, s4
	s_ashr_i32 s4, s3, 3
	s_add_i32 s3, s6, s5
	s_and_b32 s3, s3, -8
	s_sub_i32 s3, s6, s3
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
	s_mul_i32 s2, s3, s4
	s_sub_i32 s3, s5, s3
	s_add_i32 s8, s4, -1
	s_mul_i32 s3, s3, s8
	s_add_i32 s8, s3, s2
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
	s_load_dwordx2 s[38:39], s[0:1], 0x20
	s_load_dwordx2 s[34:35], s[0:1], 0x30
	s_load_dwordx2 s[36:37], s[0:1], 0x50
	s_load_dwordx2 s[2:3], s[0:1], 0x60
	s_load_dwordx2 s[4:5], s[0:1], 0x80
	s_load_dwordx2 s[8:9], s[0:1], 0x90
	s_load_dwordx2 s[12:13], s[0:1], 0xb0
	s_load_dwordx2 s[26:27], s[0:1], 0xc0
	s_load_dwordx2 s[28:29], s[0:1], 0xe0
	s_load_dword s24, s[0:1], 0xf0
	v_lshrrev_b32_e32 v1, 6, v0
	v_lshlrev_b32_e32 v1, 10, v1
	s_abs_i32 s11, s7
	v_readfirstlane_b32 s33, v1
	s_add_i32 s44, s33, 0x18000
	s_add_i32 s45, s44, 0x4000
	s_ashr_i32 s1, s7, 31
	s_waitcnt lgkmcnt(0)
	s_lshr_b32 s5, s1, 23
	s_add_i32 s5, s7, s5
	s_add_i32 s39, s33, 0x4000
	s_add_i32 s40, s33, 0x8000
	s_add_i32 s41, s40, 0x4000
	s_add_i32 s42, s33, 0x10000
	s_add_i32 s43, s42, 0x4000
	s_lshr_b32 s0, s10, 25
	s_add_i32 s6, s6, s0
	s_ashr_i32 s0, s6, 7
	s_ashr_i32 s6, s5, 9
	s_lshl_b32 s6, s6, 2
	s_sub_i32 s0, s0, s6
	s_min_i32 s0, s0, 4
	s_abs_i32 s10, s0
	s_sub_i32 s13, 0, s10
	s_movk_i32 s37, 0x70
	v_cvt_f32_u32_e32 v1, s10
	v_lshrrev_b32_e32 v6, 7, v0
	v_bfe_u32 v7, v0, 6, 1
	v_and_b32_e32 v84, 48, v0
	v_rcp_iflag_f32_e32 v1, v1
	v_lshlrev_b32_e32 v8, 6, v7
	v_lshlrev_b32_e32 v83, 13, v6
	v_lshlrev_b32_e32 v82, 3, v0
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	v_lshlrev_b32_e32 v86, 13, v7
	v_or_b32_e32 v87, 0x10000, v86
	s_mov_b32 s73, 0
	v_readfirstlane_b32 s14, v1
	s_mul_i32 s13, s13, s14
	s_mul_hi_u32 s13, s14, s13
	s_add_i32 s14, s14, s13
	s_mul_hi_u32 s13, s11, s14
	s_mul_i32 s13, s13, s10
	s_sub_i32 s11, s11, s13
	s_sub_i32 s13, s11, s10
	s_cmp_ge_u32 s11, s10
	s_cselect_b32 s11, s13, s11
	s_sub_i32 s13, s11, s10
	s_cmp_ge_u32 s11, s10
	s_cselect_b32 s11, s13, s11
	s_xor_b32 s11, s11, s1
	s_sub_i32 s25, s11, s1
	s_add_i32 s25, s25, s6
	s_mov_b32 s29, 7
	s_and_b32 s1, s5, 0xfffffe00
	s_sub_i32 s1, s7, s1
	s_xor_b32 s0, s1, s0
	s_ashr_i32 s0, s0, 31
	v_lshlrev_b32_e32 v1, 4, v0
	v_bitop3_b32 v2, v1, s37, v0 bitop3:0x48
	v_lshrrev_b32_e32 v1, 3, v0
	s_abs_i32 s1, s1
	s_mul_hi_u32 s5, s1, s14
	s_mul_i32 s6, s5, s10
	s_sub_i32 s1, s1, s6
	s_sub_i32 s7, s1, s10
	v_accvgpr_write_b32 a195, 0
	s_add_i32 s6, s5, 1
	s_cmp_ge_u32 s1, s10
	s_cselect_b32 s5, s6, s5
	s_cselect_b32 s1, s7, s1
	s_add_i32 s6, s5, 1
	s_cmp_ge_u32 s1, s10
	s_cselect_b32 s1, s6, s5
	s_lshl_b32 s46, s25, 8
	s_or_b32 s49, s46, 0x80
	s_xor_b32 s1, s1, s0
	s_sub_i32 s10, s1, s0
	s_lshl_b32 s52, s10, 8
	s_or_b32 s56, s52, 0x80
	v_mad_u64_u32 v[66:67], s[0:1], v1, s38, v[2:3]
	s_lshl_b32 s0, s38, 5
	s_nop 0
	v_add_u32_e32 v67, s0, v66
	v_add_u32_e32 v94, s0, v67
	v_or_b32_e32 v182, s52, v8
	s_mul_i32 s52, s36, s52
	s_or_b32 s69, s52, 0x80
	v_or_b32_e32 v3, 0x60, v1
	v_mad_u64_u32 v[70:71], s[0:1], v1, s36, v[2:3]
	v_lshlrev_b32_e32 v1, 6, v6
	s_add_u32 s75, s34, s52
	v_mad_u64_u32 v[68:69], s[0:1], v3, s38, v[2:3]
	s_lshl_b32 s0, s36, 5
	s_nop 0
	v_add_u32_e32 v69, s0, v70
	v_add_u32_e32 v71, s0, v69
	v_mad_u64_u32 v[72:73], s[0:1], v3, s36, v[2:3]
	v_and_b32_e32 v73, 0x1f8, v82
	s_mov_b32 s57, s52
	v_or_b32_e32 v2, s46, v1
	v_ashrrev_i32_e32 v4, 6, v2
	v_mov_b64_e32 v[2:3], s[2:3]
	v_mad_i64_i32 v[4:5], s[0:1], s4, v4, v[2:3]
	s_nop 0
	v_readfirstlane_b32 s0, v4
	v_or_b32_e32 v4, s49, v1
	v_ashrrev_i32_e32 v4, 6, v4
	v_mad_i64_i32 v[2:3], s[4:5], s4, v4, v[2:3]
	s_nop 0
	v_readfirstlane_b32 s11, v2
	s_mov_b32 s2, -1
	v_ashrrev_i32_e32 v4, 6, v182
	v_readfirstlane_b32 s13, v3
	v_mov_b64_e32 v[2:3], s[8:9]
	v_readfirstlane_b32 s1, v5
	v_mad_i64_i32 v[4:5], s[8:9], s12, v4, v[2:3]
	s_nop 0
	v_readfirstlane_b32 s14, v5
	s_mov_b32 s3, 0x110000
	;;#ASMSTART
	buffer_load_dwordx2 v[80:81], v73, s[0:3], s73 offen
	;;#ASMEND
	s_mov_b64 s[22:23], s[2:3]
	s_mov_b64 s[20:21], s[0:1]
	s_mov_b32 s20, s34
	s_mov_b64 s[6:7], s[2:3]
	s_mov_b64 s[4:5], s[0:1]
	s_mov_b32 s4, s11
	s_mov_b64 s[10:11], s[2:3]
	s_mov_b64 s[8:9], s[0:1]
	s_mov_b32 s9, s14
	s_mov_b32 s21, s35
	s_mov_b32 s35, s33
	s_mov_b32 s5, s13
	;;#ASMSTART
	buffer_load_dwordx2 v[74:75], v73, s[4:7], s73 offen
	;;#ASMEND
	v_readfirstlane_b32 s13, v4
	s_mov_b32 s8, s13
	;;#ASMSTART
	buffer_load_dwordx2 v[78:79], v73, s[8:11], s73 offen
	;;#ASMEND
	v_or_b32_e32 v4, s56, v8
	v_ashrrev_i32_e32 v4, 6, v4
	v_mad_i64_i32 v[2:3], s[12:13], s12, v4, v[2:3]
	s_nop 0
	v_readfirstlane_b32 s16, v2
	v_lshlrev_b32_e32 v2, 7, v0
	v_and_b32_e32 v85, 0x780, v2
	v_or_b32_e32 v34, v85, v84
	v_or_b32_e32 v50, v34, v87
	v_or_b32_e32 v87, v85, v87
	s_mul_i32 s56, s36, s56
	s_add_i32 s76, s56, 0x80
	v_or_b32_e32 v18, v34, v83
	v_bitop3_b32 v14, v82, v18, s37 bitop3:0x6c
	v_or_b32_e32 v18, 64, v18
	v_bitop3_b32 v30, v82, v18, s37 bitop3:0x6c
	s_add_u32 s36, s34, s56
	v_or_b32_e32 v83, v85, v83
	v_or_b32_e32 v88, v83, v84
	v_bitop3_b32 v95, v82, v88, s37 bitop3:0x6c
	v_or_b32_e32 v88, 64, v84
	v_or_b32_e32 v89, v83, v88
	v_bitop3_b32 v96, v82, v89, s37 bitop3:0x6c
	v_or_b32_e32 v89, 0x4000, v83
	v_or_b32_e32 v90, v89, v84
	v_bitop3_b32 v97, v82, v90, s37 bitop3:0x6c
	v_lshrrev_b32_e32 v34, 4, v50
	v_bitop3_b32 v46, v34, v50, s37 bitop3:0x6c
	v_add_u32_e32 v50, 64, v50
	v_lshrrev_b32_e32 v51, 4, v50
	v_bitop3_b32 v62, v51, v50, s37 bitop3:0x6c
	v_or_b32_e32 v89, v89, v88
	v_bitop3_b32 v98, v82, v89, s37 bitop3:0x6c
	v_add_u32_e32 v89, v87, v84
	v_lshrrev_b32_e32 v90, 4, v89
	v_bitop3_b32 v99, v90, v89, s37 bitop3:0x6c
	v_or_b32_e32 v85, v86, v85
	v_or_b32_e32 v86, 0x14000, v85
	v_add_u32_e32 v87, v87, v88
	v_lshrrev_b32_e32 v89, 4, v87
	v_bitop3_b32 v100, v89, v87, s37 bitop3:0x6c
	v_or_b32_e32 v87, v86, v84
	v_bitop3_b32 v101, v82, v87, s37 bitop3:0x6c
	s_mov_b32 s60, s56
	v_or_b32_e32 v86, v86, v88
	v_bitop3_b32 v102, v82, v86, s37 bitop3:0x6c
	v_or_b32_e32 v86, 0x18000, v85
	v_or_b32_e32 v87, v86, v84
	v_bitop3_b32 v103, v82, v87, s37 bitop3:0x6c
	s_mov_b64 s[14:15], s[2:3]
	s_mov_b64 s[12:13], s[0:1]
	s_mov_b32 s12, s16
	v_or_b32_e32 v86, v86, v88
	v_bitop3_b32 v104, v82, v86, s37 bitop3:0x6c
	v_or_b32_e32 v85, 0x1c000, v85
	v_or_b32_e32 v86, v85, v84
	v_bitop3_b32 v105, v82, v86, s37 bitop3:0x6c
	v_readfirstlane_b32 s17, v3
	s_mov_b32 s13, s17
	;;#ASMSTART
	buffer_load_dwordx2 v[76:77], v73, s[12:15], s73 offen
	;;#ASMEND
	v_or_b32_e32 v85, v85, v88
	v_bitop3_b32 v106, v82, v85, s37 bitop3:0x6c
	v_or_b32_e32 v85, 0x8000, v83
	v_or_b32_e32 v86, v85, v84
	v_bitop3_b32 v107, v82, v86, s37 bitop3:0x6c
	s_mov_b64 s[18:19], s[2:3]
	s_mov_b64 s[16:17], s[0:1]
	s_mov_b32 s16, s30
	v_or_b32_e32 v85, v85, v88
	v_bitop3_b32 v108, v82, v85, s37 bitop3:0x6c
	s_mov_b32 s17, s31
	s_mul_i32 s31, s38, s46
	s_or_b32 s63, s31, 0x80
	v_or_b32_e32 v83, 0xc000, v83
	v_or_b32_e32 v84, v83, v84
	v_bitop3_b32 v109, v82, v84, s37 bitop3:0x6c
	s_add_i32 s46, s33, 0x1000
	s_mov_b32 s47, s46
	v_or_b32_e32 v83, v83, v88
	v_bitop3_b32 v110, v82, v83, s37 bitop3:0x6c
	s_mul_i32 s38, s38, s49
	s_add_i32 s66, s38, 0x80
	s_add_u32 s72, s30, s31
	s_add_u32 s72, s72, 0x100
	s_mov_b32 s49, s40
	s_add_u32 s74, s30, s38
	s_add_u32 s73, s74, 0x100
	s_add_u32 s74, s75, 0x100
	s_add_u32 s75, s36, 0x100
	s_mov_b64 s[36:37], 0
	s_mov_b32 s53, s38
	s_mov_b32 s50, s31
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s35
	s_movk_i32 s35, 0x1000
	buffer_load_dwordx4 v66, s[16:19], s50 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s47
	s_add_i32 s47, s33, 0x2000
	s_mov_b32 s48, s47
	buffer_load_dwordx4 v67, s[16:19], s50 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s48
	s_add_i32 s48, s33, 0x3000
	s_mov_b32 s51, s48
	buffer_load_dwordx4 v94, s[16:19], s50 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s51
	v_accvgpr_write_b32 a194, 0
	buffer_load_dwordx4 v68, s[16:19], s50 offen lds
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s49
	s_add_i32 s49, s40, 0x1000
	s_mov_b32 s50, s49
	buffer_load_dwordx4 v66, s[16:19], s53 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s50
	s_add_i32 s50, s40, 0x2000
	s_mov_b32 s51, s50
	buffer_load_dwordx4 v67, s[16:19], s53 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s51
	s_add_i32 s51, s40, 0x3000
	s_mov_b32 s54, s51
	buffer_load_dwordx4 v94, s[16:19], s53 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s54
	v_accvgpr_write_b32 a193, 0
	buffer_load_dwordx4 v68, s[16:19], s53 offen lds
	s_mov_b32 s53, s42
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s53
	s_add_i32 s53, s42, 0x1000
	s_mov_b32 s54, s53
	buffer_load_dwordx4 v70, s[20:23], s57 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s54
	s_add_i32 s54, s42, 0x2000
	s_mov_b32 s55, s54
	buffer_load_dwordx4 v69, s[20:23], s57 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s55
	s_add_i32 s55, s42, 0x3000
	s_mov_b32 s58, s55
	buffer_load_dwordx4 v71, s[20:23], s57 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s58
	v_accvgpr_write_b32 a192, 0
	buffer_load_dwordx4 v72, s[20:23], s57 offen lds
	s_mov_b32 s57, s44
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s57
	s_add_i32 s57, s44, 0x1000
	s_mov_b32 s58, s57
	buffer_load_dwordx4 v70, s[20:23], s60 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s58
	s_add_i32 s58, s44, 0x2000
	s_mov_b32 s59, s58
	buffer_load_dwordx4 v69, s[20:23], s60 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s59
	s_add_i32 s59, s44, 0x3000
	s_mov_b32 s61, s59
	buffer_load_dwordx4 v71, s[20:23], s60 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s61
	v_accvgpr_write_b32 a199, 0
	buffer_load_dwordx4 v72, s[20:23], s60 offen lds
	s_mov_b32 s60, s39
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s60
	s_add_i32 s60, s33, 0x5000
	s_mov_b32 s61, s60
	buffer_load_dwordx4 v66, s[16:19], s63 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s61
	s_add_i32 s61, s33, 0x6000
	s_mov_b32 s62, s61
	buffer_load_dwordx4 v67, s[16:19], s63 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s62
	s_add_i32 s62, s33, 0x7000
	s_mov_b32 s64, s62
	buffer_load_dwordx4 v94, s[16:19], s63 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s64
	v_accvgpr_write_b32 a198, 0
	buffer_load_dwordx4 v68, s[16:19], s63 offen lds
	s_mov_b32 s63, s41
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s63
	s_add_i32 s63, s40, 0x5000
	s_mov_b32 s64, s63
	buffer_load_dwordx4 v66, s[16:19], s66 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s64
	s_add_i32 s64, s40, 0x6000
	s_mov_b32 s65, s64
	buffer_load_dwordx4 v67, s[16:19], s66 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s65
	s_add_i32 s65, s40, 0x7000
	s_mov_b32 s67, s65
	buffer_load_dwordx4 v94, s[16:19], s66 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s67
	v_accvgpr_write_b32 a197, 0
	buffer_load_dwordx4 v68, s[16:19], s66 offen lds
	s_mov_b32 s66, s43
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s66
	s_add_i32 s66, s42, 0x5000
	s_mov_b32 s67, s66
	buffer_load_dwordx4 v70, s[20:23], s69 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s67
	s_add_i32 s67, s42, 0x6000
	s_mov_b32 s68, s67
	buffer_load_dwordx4 v69, s[20:23], s69 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s68
	s_add_i32 s68, s42, 0x7000
	s_mov_b32 s70, s68
	buffer_load_dwordx4 v71, s[20:23], s69 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s70
	v_accvgpr_write_b32 a196, 0
	buffer_load_dwordx4 v72, s[20:23], s69 offen lds
	s_mov_b32 s69, s45
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s69
	s_add_i32 s69, s44, 0x5000
	s_mov_b32 s70, s69
	buffer_load_dwordx4 v70, s[20:23], s76 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s70
	s_add_i32 s70, s44, 0x6000
	s_mov_b32 s71, s70
	buffer_load_dwordx4 v69, s[20:23], s76 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s71
	s_add_i32 s71, s44, 0x7000
	s_mov_b32 s77, s71
	buffer_load_dwordx4 v71, s[20:23], s76 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s77
	v_accvgpr_write_b32 a203, 0
	buffer_load_dwordx4 v72, s[20:23], s76 offen lds
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
	v_accvgpr_write_b32 a219, 0
	v_accvgpr_write_b32 a218, 0
	v_accvgpr_write_b32 a217, 0
	v_accvgpr_write_b32 a216, 0
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
	v_accvgpr_write_b32 a247, 0
	v_accvgpr_write_b32 a246, 0
	v_accvgpr_write_b32 a245, 0
	v_accvgpr_write_b32 a244, 0
	v_accvgpr_write_b32 a251, 0
	v_accvgpr_write_b32 a250, 0
	v_accvgpr_write_b32 a249, 0
	v_accvgpr_write_b32 a248, 0
	v_accvgpr_write_b32 a255, 0
	v_accvgpr_write_b32 a254, 0
	v_accvgpr_write_b32 a253, 0
	v_accvgpr_write_b32 a252, 0
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
	v_accvgpr_write_b32 a167, 0
	v_accvgpr_write_b32 a166, 0
	v_accvgpr_write_b32 a165, 0
	v_accvgpr_write_b32 a164, 0
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
	v_accvgpr_write_b32 a191, 0
	v_accvgpr_write_b32 a190, 0
	v_accvgpr_write_b32 a189, 0
	v_accvgpr_write_b32 a188, 0
	v_accvgpr_write_b32 a15, 0
	v_accvgpr_write_b32 a14, 0
	v_accvgpr_write_b32 a13, 0
	v_accvgpr_write_b32 a12, 0
	v_accvgpr_write_b32 a7, 0
	v_accvgpr_write_b32 a6, 0
	v_accvgpr_write_b32 a5, 0
	v_accvgpr_write_b32 a4, 0
	v_accvgpr_write_b32 a11, 0
	v_accvgpr_write_b32 a10, 0
	v_accvgpr_write_b32 a9, 0
	v_accvgpr_write_b32 a8, 0
	v_accvgpr_write_b32 a3, 0
	v_accvgpr_write_b32 a2, 0
	v_accvgpr_write_b32 a1, 0
	v_accvgpr_write_b32 a0, 0
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
	v_accvgpr_write_b32 a123, 0
	v_accvgpr_write_b32 a122, 0
	v_accvgpr_write_b32 a121, 0
	v_accvgpr_write_b32 a120, 0
	v_accvgpr_write_b32 a127, 0
	v_accvgpr_write_b32 a126, 0
	v_accvgpr_write_b32 a125, 0
	v_accvgpr_write_b32 a124, 0
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
	v_accvgpr_write_b32 a27, 0
	v_accvgpr_write_b32 a26, 0
	v_accvgpr_write_b32 a25, 0
	v_accvgpr_write_b32 a24, 0
	v_accvgpr_write_b32 a71, 0
	v_accvgpr_write_b32 a70, 0
	v_accvgpr_write_b32 a69, 0
	v_accvgpr_write_b32 a68, 0
	v_accvgpr_write_b32 a75, 0
	v_accvgpr_write_b32 a74, 0
	v_accvgpr_write_b32 a73, 0
	v_accvgpr_write_b32 a72, 0
	v_accvgpr_write_b32 a79, 0
	v_accvgpr_write_b32 a78, 0
	v_accvgpr_write_b32 a77, 0
	v_accvgpr_write_b32 a76, 0
	v_accvgpr_write_b32 a23, 0
	v_accvgpr_write_b32 a22, 0
	v_accvgpr_write_b32 a21, 0
	v_accvgpr_write_b32 a20, 0
	v_accvgpr_write_b32 a19, 0
	v_accvgpr_write_b32 a18, 0
	v_accvgpr_write_b32 a17, 0
	v_accvgpr_write_b32 a16, 0
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
.LBB0_7:                                ; =>This Inner Loop Header: Depth=1
	s_add_u32 s76, s31, s36
	s_add_u32 s80, s76, 0x100
	s_mov_b32 m0, s33
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[2:5], v[34:37], a[28:31],  v80, v78 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[140:143], v103 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[2:5], v[38:41], a[32:35],  v80, v78 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[144:147], v103 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[2:5], v[42:45], a[36:39],  v80, v79 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[148:151], v103 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[2:5], v[46:49], a[40:43],  v80, v79 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[152:155], v103 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[18:21], v[50:53], a[28:31],  v80, v78 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[156:159], v104 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[18:21], v[54:57], a[32:35],  v80, v78 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[160:163], v104 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[18:21], v[58:61], a[36:39],  v80, v79 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[164:167], v104 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[18:21], v[62:65], a[40:43],  v80, v79 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[168:171], v104 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[6:9], v[34:37], a[44:47],  v80, v78 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[6:9], v[38:41], a[48:51],  v80, v78 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[6:9], v[42:45], a[52:55],  v80, v79 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[6:9], v[46:49], a[56:59],  v80, v79 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[22:25], v[50:53], a[44:47],  v80, v78 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[22:25], v[54:57], a[48:51],  v80, v78 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[22:25], v[58:61], a[52:55],  v80, v79 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[22:25], v[62:65], a[56:59],  v80, v79 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[10:13], v[34:37], a[60:63],  v81, v78 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[10:13], v[38:41], a[64:67],  v81, v78 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[10:13], v[42:45], a[24:27], v81, v79 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[10:13], v[46:49], a[68:71], v81, v79 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[26:29], v[50:53], a[60:63],  v81, v78 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[26:29], v[54:57], a[64:67],  v81, v78 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[26:29], v[58:61], a[24:27], v81, v79 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[26:29], v[62:65], a[68:71], v81, v79 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[14:17], v[34:37], a[72:75], v81, v78 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[14:17], v[38:41], a[76:79], v81, v78 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[14:17], v[42:45], a[20:23], v81, v79 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[14:17], v[46:49], a[16:19], v81, v79 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[30:33], v[50:53], a[72:75], v81, v78 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[30:33], v[54:57], a[76:79], v81, v78 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[30:33], v[58:61], a[20:23], v81, v79 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[30:33], v[62:65], a[16:19], v81, v79 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[2:5], v[140:143], a[12:15], v80, v76 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[90:93], v107 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[2:5], v[144:147], a[4:7], v80, v76 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[112:115], v107 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[2:5], v[148:151], a[8:11], v80, v77 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[116:119], v107 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[2:5], v[152:155], a[0:3], v80, v77 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[120:123], v107 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[18:21], v[156:159], a[12:15], v80, v76 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[124:127], v108 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[18:21], v[160:163], a[4:7], v80, v76 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[128:131], v108 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[18:21], v[164:167], a[8:11], v80, v77 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[132:135], v108 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[18:21], v[168:171], a[0:3], v80, v77 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[136:139], v108 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[6:9], v[140:143], a[80:83], v80, v76 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[6:9], v[144:147], a[84:87], v80, v76 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[6:9], v[148:151], a[88:91], v80, v77 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[6:9], v[152:155], a[92:95], v80, v77 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[22:25], v[156:159], a[80:83], v80, v76 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[22:25], v[160:163], a[84:87], v80, v76 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[22:25], v[164:167], a[88:91], v80, v77 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[22:25], v[168:171], a[92:95], v80, v77 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[10:13], v[140:143], a[96:99], v81, v76 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[10:13], v[144:147], a[100:103], v81, v76 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[10:13], v[148:151], a[104:107], v81, v77 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[10:13], v[152:155], a[108:111], v81, v77 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[26:29], v[156:159], a[96:99], v81, v76 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[26:29], v[160:163], a[100:103], v81, v76 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[26:29], v[164:167], a[104:107], v81, v77 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[26:29], v[168:171], a[108:111], v81, v77 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[14:17], v[140:143], a[112:115], v81, v76 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[14:17], v[144:147], a[116:119], v81, v76 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[14:17], v[148:151], a[120:123], v81, v77 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[14:17], v[152:155], a[124:127], v81, v77 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[30:33], v[156:159], a[112:115], v81, v76 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[30:33], v[160:163], a[116:119], v81, v76 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[30:33], v[164:167], a[120:123], v81, v77 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[30:33], v[168:171], a[124:127], v81, v77 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(20)
s_barrier

	;;#ASMEND
	s_add_u32 s77, s38, s36
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[90:93], v[34:37], a[128:131],  v74, v78 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[2:5], v97 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[90:93], v[38:41], a[132:135],  v74, v78 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[6:9], v97 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[90:93], v[42:45], a[136:139],  v74, v79 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[10:13], v97 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[90:93], v[46:49], a[140:143],  v74, v79 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[14:17], v97 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[124:127], v[50:53], a[128:131],  v74, v78 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[18:21], v98 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[124:127], v[54:57], a[132:135],  v74, v78 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[22:25], v98 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[124:127], v[58:61], a[136:139],  v74, v79 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[26:29], v98 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[124:127], v[62:65], a[140:143],  v74, v79 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v98 offset:6144

	;;#ASMEND
	buffer_load_dwordx4 v66, s[16:19], s80 offen lds
	s_mov_b32 m0, s46
	s_add_u32 s81, s77, 0x100
	buffer_load_dwordx4 v67, s[16:19], s80 offen lds
	s_mov_b32 m0, s47
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[112:115], v[34:37], a[144:147],  v74, v78 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[112:115], v[38:41], a[148:151],  v74, v78 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[112:115], v[42:45], a[152:155],  v74, v79 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[112:115], v[46:49], a[156:159],  v74, v79 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[128:131], v[50:53], a[144:147],  v74, v78 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[128:131], v[54:57], a[148:151],  v74, v78 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[128:131], v[58:61], a[152:155],  v74, v79 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[128:131], v[62:65], a[156:159],  v74, v79 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_u32 s78, s52, s36
	buffer_load_dwordx4 v94, s[16:19], s80 offen lds
	s_mov_b32 m0, s48
	s_add_u32 s82, s78, 0x100
	buffer_load_dwordx4 v68, s[16:19], s80 offen lds
	s_mov_b32 m0, s40
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[116:119], v[34:37], a[160:163],  v75, v78 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[116:119], v[38:41], a[164:167],  v75, v78 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[116:119], v[42:45], a[168:171], v75, v79 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[116:119], v[46:49], a[172:175], v75, v79 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[132:135], v[50:53], a[160:163],  v75, v78 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[132:135], v[54:57], a[164:167],  v75, v78 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[132:135], v[58:61], a[168:171], v75, v79 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[132:135], v[62:65], a[172:175], v75, v79 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_u32 s79, s56, s36
	buffer_load_dwordx4 v66, s[16:19], s81 offen lds
	s_mov_b32 m0, s49
	s_add_u32 s83, s79, 0x100
	buffer_load_dwordx4 v67, s[16:19], s81 offen lds
	s_mov_b32 m0, s50
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[120:123], v[34:37], a[176:179], v75, v78 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[120:123], v[38:41], a[180:183], v75, v78 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[120:123], v[42:45], a[184:187], v75, v79 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[120:123], v[46:49], a[188:191], v75, v79 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[136:139], v[50:53], a[176:179], v75, v78 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[136:139], v[54:57], a[180:183], v75, v78 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[136:139], v[58:61], a[184:187], v75, v79 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[136:139], v[62:65], a[188:191], v75, v79 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_i32 s84, s35, 0xfffff200
	buffer_load_dwordx4 v94, s[16:19], s81 offen lds
	s_mov_b32 m0, s51
	;;#ASMSTART
	buffer_load_dwordx2 v[88:89], v73, s[0:3], s84 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[82:83], v73, s[4:7], s84 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[86:87], v73, s[8:11], s84 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[84:85], v73, s[12:15], s84 offen
	;;#ASMEND
	s_add_u32 s80, s79, 0x180
	buffer_load_dwordx4 v68, s[16:19], s81 offen lds
	s_mov_b32 m0, s42
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[90:93], v[140:143], a[192:195],  v74, v76 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v101 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[90:93], v[144:147], a[196:199],  v74, v76 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v101 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[90:93], v[148:151], a[200:203],  v74, v77 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v101 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[90:93], v[152:155], a[204:207],  v74, v77 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v101 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[124:127], v[156:159], a[192:195],  v74, v76 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v102 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[124:127], v[160:163], a[196:199],  v74, v76 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v102 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[124:127], v[164:167], a[200:203],  v74, v77 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v102 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[124:127], v[168:171], a[204:207],  v74, v77 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[62:65], v102 offset:6144

	;;#ASMEND
	s_add_u32 s81, s78, 0x180
	buffer_load_dwordx4 v70, s[20:23], s82 offen lds
	s_mov_b32 m0, s53
	s_add_i32 s84, s35, 0xfffff400
	buffer_load_dwordx4 v69, s[20:23], s82 offen lds
	s_mov_b32 m0, s54
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[112:115], v[140:143], a[208:211],  v74, v76 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[112:115], v[144:147], a[212:215],  v74, v76 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[112:115], v[148:151], a[216:219],  v74, v77 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[112:115], v[152:155], a[220:223],  v74, v77 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[128:131], v[156:159], a[208:211],  v74, v76 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[128:131], v[160:163], a[212:215],  v74, v76 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[128:131], v[164:167], a[216:219],  v74, v77 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[128:131], v[168:171], a[220:223],  v74, v77 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[78:79], v73, s[12:15], s84 offen
	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v71, s[20:23], s82 offen lds
	s_mov_b32 m0, s55
	s_nop 0
	buffer_load_dwordx4 v72, s[20:23], s82 offen lds
	s_mov_b32 m0, s44
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
	s_add_u32 s82, s77, 0x180
	buffer_load_dwordx4 v70, s[20:23], s83 offen lds
	s_mov_b32 m0, s57
	s_nop 0
	buffer_load_dwordx4 v69, s[20:23], s83 offen lds
	s_mov_b32 m0, s58
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[120:123], v[140:143], a[240:243], v75, v76 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[120:123], v[144:147], a[244:247], v75, v76 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[120:123], v[148:151], a[248:251], v75, v77 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[120:123], v[152:155], a[252:255], v75, v77 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[136:139], v[156:159], a[240:243], v75, v76 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[136:139], v[160:163], a[244:247], v75, v76 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[136:139], v[164:167], a[248:251], v75, v77 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[136:139], v[168:171], a[252:255], v75, v77 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[92:93], v73, s[0:3], s84 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[74:75], v73, s[4:7], s84 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[90:91], v73, s[8:11], s84 offen
	;;#ASMEND
	s_add_i32 s84, s35, 0xfffff600
	buffer_load_dwordx4 v71, s[20:23], s83 offen lds
	s_mov_b32 m0, s59
	;;#ASMSTART
	buffer_load_dwordx2 v[76:77], v73, s[4:7], s84 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[80:81], v73, s[12:15], s84 offen
	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v72, s[20:23], s83 offen lds
	s_add_u32 s83, s76, 0x180
	s_mov_b32 m0, s39
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[2:5], v[34:37], a[28:31],  v88, v86 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[144:147], v105 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[2:5], v[38:41], a[32:35],  v88, v86 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[148:151], v105 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[2:5], v[42:45], a[36:39],  v88, v87 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[152:155], v105 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[2:5], v[46:49], a[40:43],  v88, v87 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[156:159], v105 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[18:21], v[50:53], a[28:31],  v88, v86 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[160:163], v106 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[18:21], v[54:57], a[32:35],  v88, v86 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[164:167], v106 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[18:21], v[58:61], a[36:39],  v88, v87 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[168:171], v106 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[18:21], v[62:65], a[40:43],  v88, v87 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[172:175], v106 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[6:9], v[34:37], a[44:47],  v88, v86 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[6:9], v[38:41], a[48:51],  v88, v86 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[6:9], v[42:45], a[52:55],  v88, v87 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[6:9], v[46:49], a[56:59],  v88, v87 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[22:25], v[50:53], a[44:47],  v88, v86 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[22:25], v[54:57], a[48:51],  v88, v86 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[22:25], v[58:61], a[52:55],  v88, v87 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[22:25], v[62:65], a[56:59],  v88, v87 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[10:13], v[34:37], a[60:63],  v89, v86 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[10:13], v[38:41], a[64:67],  v89, v86 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[10:13], v[42:45], a[24:27], v89, v87 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[10:13], v[46:49], a[68:71], v89, v87 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[26:29], v[50:53], a[60:63],  v89, v86 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[26:29], v[54:57], a[64:67],  v89, v86 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[26:29], v[58:61], a[24:27], v89, v87 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[26:29], v[62:65], a[68:71], v89, v87 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[14:17], v[34:37], a[72:75], v89, v86 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[14:17], v[38:41], a[76:79], v89, v86 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[14:17], v[42:45], a[20:23], v89, v87 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[14:17], v[46:49], a[16:19], v89, v87 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[30:33], v[50:53], a[72:75], v89, v86 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[30:33], v[54:57], a[76:79], v89, v86 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[30:33], v[58:61], a[20:23], v89, v87 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[30:33], v[62:65], a[16:19], v89, v87 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[2:5], v[144:147], a[12:15], v88, v84 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[112:115], v109 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[2:5], v[148:151], a[4:7], v88, v84 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[116:119], v109 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[2:5], v[152:155], a[8:11], v88, v85 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[120:123], v109 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[2:5], v[156:159], a[0:3], v88, v85 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[124:127], v109 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[18:21], v[160:163], a[12:15], v88, v84 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[128:131], v110 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[18:21], v[164:167], a[4:7], v88, v84 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[132:135], v110 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[18:21], v[168:171], a[8:11], v88, v85 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[136:139], v110 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[18:21], v[172:175], a[0:3], v88, v85 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[140:143], v110 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[6:9], v[144:147], a[80:83], v88, v84 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[6:9], v[148:151], a[84:87], v88, v84 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[6:9], v[152:155], a[88:91], v88, v85 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[6:9], v[156:159], a[92:95], v88, v85 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[22:25], v[160:163], a[80:83], v88, v84 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[22:25], v[164:167], a[84:87], v88, v84 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[22:25], v[168:171], a[88:91], v88, v85 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[22:25], v[172:175], a[92:95], v88, v85 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[10:13], v[144:147], a[96:99], v89, v84 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[10:13], v[148:151], a[100:103], v89, v84 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[10:13], v[152:155], a[104:107], v89, v85 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[10:13], v[156:159], a[108:111], v89, v85 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[26:29], v[160:163], a[96:99], v89, v84 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[26:29], v[164:167], a[100:103], v89, v84 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[26:29], v[168:171], a[104:107], v89, v85 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[26:29], v[172:175], a[108:111], v89, v85 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[14:17], v[144:147], a[112:115], v89, v84 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[14:17], v[148:151], a[116:119], v89, v84 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[14:17], v[152:155], a[120:123], v89, v85 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[14:17], v[156:159], a[124:127], v89, v85 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[30:33], v[160:163], a[112:115], v89, v84 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[30:33], v[164:167], a[116:119], v89, v84 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[30:33], v[168:171], a[120:123], v89, v85 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[30:33], v[172:175], a[124:127], v89, v85 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(20)
s_barrier

	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[88:89], v73, s[0:3], s84 offen
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[112:115], v[34:37], a[128:131],  v82, v86 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[2:5], v95 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[112:115], v[38:41], a[132:135],  v82, v86 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[6:9], v95 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[112:115], v[42:45], a[136:139],  v82, v87 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[10:13], v95 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[112:115], v[46:49], a[140:143],  v82, v87 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[14:17], v95 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[128:131], v[50:53], a[128:131],  v82, v86 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[18:21], v96 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[128:131], v[54:57], a[132:135],  v82, v86 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[22:25], v96 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[128:131], v[58:61], a[136:139],  v82, v87 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[26:29], v96 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[128:131], v[62:65], a[140:143],  v82, v87 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v96 offset:6144

	;;#ASMEND
	buffer_load_dwordx4 v66, s[16:19], s83 offen lds
	s_mov_b32 m0, s60
	s_nop 0
	buffer_load_dwordx4 v67, s[16:19], s83 offen lds
	s_mov_b32 m0, s61
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[116:119], v[34:37], a[144:147],  v82, v86 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[116:119], v[38:41], a[148:151],  v82, v86 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[116:119], v[42:45], a[152:155],  v82, v87 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[116:119], v[46:49], a[156:159],  v82, v87 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[132:135], v[50:53], a[144:147],  v82, v86 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[132:135], v[54:57], a[148:151],  v82, v86 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[132:135], v[58:61], a[152:155],  v82, v87 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[132:135], v[62:65], a[156:159],  v82, v87 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v94, s[16:19], s83 offen lds
	s_mov_b32 m0, s62
	s_nop 0
	buffer_load_dwordx4 v68, s[16:19], s83 offen lds
	s_mov_b32 m0, s41
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[120:123], v[34:37], a[160:163],  v83, v86 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[120:123], v[38:41], a[164:167],  v83, v86 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[120:123], v[42:45], a[168:171], v83, v87 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[120:123], v[46:49], a[172:175], v83, v87 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[136:139], v[50:53], a[160:163],  v83, v86 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[136:139], v[54:57], a[164:167],  v83, v86 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[136:139], v[58:61], a[168:171], v83, v87 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[136:139], v[62:65], a[172:175], v83, v87 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_u32 s83, s76, 0x200
	buffer_load_dwordx4 v66, s[16:19], s82 offen lds
	s_mov_b32 m0, s63
	s_nop 0
	buffer_load_dwordx4 v67, s[16:19], s82 offen lds
	s_mov_b32 m0, s64
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[124:127], v[34:37], a[176:179], v83, v86 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[124:127], v[38:41], a[180:183], v83, v86 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[124:127], v[42:45], a[184:187], v83, v87 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[124:127], v[46:49], a[188:191], v83, v87 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[140:143], v[50:53], a[176:179], v83, v86 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[140:143], v[54:57], a[180:183], v83, v86 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[140:143], v[58:61], a[184:187], v83, v87 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[140:143], v[62:65], a[188:191], v83, v87 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v94, s[16:19], s82 offen lds
	s_mov_b32 m0, s65
	s_nop 0
	buffer_load_dwordx4 v68, s[16:19], s82 offen lds
	s_mov_b32 m0, s43
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[112:115], v[144:147], a[192:195],  v82, v84 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v99 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[112:115], v[148:151], a[196:199],  v82, v84 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v99 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[112:115], v[152:155], a[200:203],  v82, v85 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v99 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[112:115], v[156:159], a[204:207],  v82, v85 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v99 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[128:131], v[160:163], a[192:195],  v82, v84 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v100 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[128:131], v[164:167], a[196:199],  v82, v84 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v100 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[128:131], v[168:171], a[200:203],  v82, v85 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v100 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[128:131], v[172:175], a[204:207],  v82, v85 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[62:65], v100 offset:6144

	;;#ASMEND
	s_add_u32 s82, s77, 0x200
	buffer_load_dwordx4 v70, s[20:23], s81 offen lds
	s_mov_b32 m0, s66
	s_nop 0
	buffer_load_dwordx4 v69, s[20:23], s81 offen lds
	s_mov_b32 m0, s67
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[116:119], v[144:147], a[208:211],  v82, v84 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[116:119], v[148:151], a[212:215],  v82, v84 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[116:119], v[152:155], a[216:219],  v82, v85 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[116:119], v[156:159], a[220:223],  v82, v85 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[132:135], v[160:163], a[208:211],  v82, v84 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[132:135], v[164:167], a[212:215],  v82, v84 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[132:135], v[168:171], a[216:219],  v82, v85 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[132:135], v[172:175], a[220:223],  v82, v85 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v71, s[20:23], s81 offen lds
	s_mov_b32 m0, s68
	s_nop 0
	buffer_load_dwordx4 v72, s[20:23], s81 offen lds
	s_mov_b32 m0, s45
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
	buffer_load_dwordx4 v70, s[20:23], s80 offen lds
	s_mov_b32 m0, s69
	s_nop 0
	buffer_load_dwordx4 v69, s[20:23], s80 offen lds
	s_mov_b32 m0, s70
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[124:127], v[144:147], a[240:243], v83, v84 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[124:127], v[148:151], a[244:247], v83, v84 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[124:127], v[152:155], a[248:251], v83, v85 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[124:127], v[156:159], a[252:255], v83, v85 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[140:143], v[160:163], a[240:243], v83, v84 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[140:143], v[164:167], a[244:247], v83, v84 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[140:143], v[168:171], a[248:251], v83, v85 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[140:143], v[172:175], a[252:255], v83, v85 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[84:85], v73, s[8:11], s84 offen
	;;#ASMEND
	s_add_i32 s84, s35, 0xfffff800
	buffer_load_dwordx4 v71, s[20:23], s80 offen lds
	s_mov_b32 m0, s71
	;;#ASMSTART
	buffer_load_dwordx2 v[86:87], v73, s[8:11], s84 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[82:83], v73, s[12:15], s84 offen
	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v72, s[20:23], s80 offen lds
	s_mov_b32 m0, s33
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[2:5], v[34:37], a[28:31],  v92, v90 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[144:147], v103 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[2:5], v[38:41], a[32:35],  v92, v90 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[148:151], v103 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[2:5], v[42:45], a[36:39],  v92, v91 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[152:155], v103 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[2:5], v[46:49], a[40:43],  v92, v91 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[156:159], v103 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[18:21], v[50:53], a[28:31],  v92, v90 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[160:163], v104 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[18:21], v[54:57], a[32:35],  v92, v90 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[164:167], v104 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[18:21], v[58:61], a[36:39],  v92, v91 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[168:171], v104 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[18:21], v[62:65], a[40:43],  v92, v91 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[172:175], v104 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[6:9], v[34:37], a[44:47],  v92, v90 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[6:9], v[38:41], a[48:51],  v92, v90 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[6:9], v[42:45], a[52:55],  v92, v91 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[6:9], v[46:49], a[56:59],  v92, v91 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[22:25], v[50:53], a[44:47],  v92, v90 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[22:25], v[54:57], a[48:51],  v92, v90 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[22:25], v[58:61], a[52:55],  v92, v91 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[22:25], v[62:65], a[56:59],  v92, v91 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[10:13], v[34:37], a[60:63],  v93, v90 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[10:13], v[38:41], a[64:67],  v93, v90 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[10:13], v[42:45], a[24:27], v93, v91 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[10:13], v[46:49], a[68:71], v93, v91 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[26:29], v[50:53], a[60:63],  v93, v90 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[26:29], v[54:57], a[64:67],  v93, v90 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[26:29], v[58:61], a[24:27], v93, v91 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[26:29], v[62:65], a[68:71], v93, v91 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[14:17], v[34:37], a[72:75], v93, v90 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[14:17], v[38:41], a[76:79], v93, v90 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[14:17], v[42:45], a[20:23], v93, v91 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[14:17], v[46:49], a[16:19], v93, v91 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[30:33], v[50:53], a[72:75], v93, v90 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[30:33], v[54:57], a[76:79], v93, v90 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[30:33], v[58:61], a[20:23], v93, v91 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[30:33], v[62:65], a[16:19], v93, v91 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[2:5], v[144:147], a[12:15], v92, v78 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[112:115], v107 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[2:5], v[148:151], a[4:7], v92, v78 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[116:119], v107 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[2:5], v[152:155], a[8:11], v92, v79 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[120:123], v107 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[2:5], v[156:159], a[0:3], v92, v79 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[124:127], v107 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[18:21], v[160:163], a[12:15], v92, v78 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[128:131], v108 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[18:21], v[164:167], a[4:7], v92, v78 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[132:135], v108 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[18:21], v[168:171], a[8:11], v92, v79 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[136:139], v108 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[18:21], v[172:175], a[0:3], v92, v79 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[140:143], v108 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[6:9], v[144:147], a[80:83], v92, v78 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[6:9], v[148:151], a[84:87], v92, v78 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[6:9], v[152:155], a[88:91], v92, v79 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[6:9], v[156:159], a[92:95], v92, v79 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[22:25], v[160:163], a[80:83], v92, v78 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[22:25], v[164:167], a[84:87], v92, v78 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[22:25], v[168:171], a[88:91], v92, v79 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[22:25], v[172:175], a[92:95], v92, v79 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[10:13], v[144:147], a[96:99], v93, v78 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[10:13], v[148:151], a[100:103], v93, v78 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[10:13], v[152:155], a[104:107], v93, v79 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[10:13], v[156:159], a[108:111], v93, v79 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[26:29], v[160:163], a[96:99], v93, v78 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[26:29], v[164:167], a[100:103], v93, v78 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[26:29], v[168:171], a[104:107], v93, v79 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[26:29], v[172:175], a[108:111], v93, v79 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[14:17], v[144:147], a[112:115], v93, v78 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[14:17], v[148:151], a[116:119], v93, v78 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[14:17], v[152:155], a[120:123], v93, v79 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[14:17], v[156:159], a[124:127], v93, v79 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[30:33], v[160:163], a[112:115], v93, v78 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[30:33], v[164:167], a[116:119], v93, v78 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[30:33], v[168:171], a[120:123], v93, v79 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[30:33], v[172:175], a[124:127], v93, v79 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(20)
s_barrier

	;;#ASMEND
	s_add_u32 s80, s79, 0x200
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[112:115], v[34:37], a[128:131],  v74, v90 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[2:5], v97 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[112:115], v[38:41], a[132:135],  v74, v90 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[6:9], v97 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[112:115], v[42:45], a[136:139],  v74, v91 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[10:13], v97 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[112:115], v[46:49], a[140:143],  v74, v91 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[14:17], v97 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[128:131], v[50:53], a[128:131],  v74, v90 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[18:21], v98 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[128:131], v[54:57], a[132:135],  v74, v90 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[22:25], v98 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[128:131], v[58:61], a[136:139],  v74, v91 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[26:29], v98 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[128:131], v[62:65], a[140:143],  v74, v91 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v98 offset:6144

	;;#ASMEND
	buffer_load_dwordx4 v66, s[16:19], s83 offen lds
	s_mov_b32 m0, s46
	s_nop 0
	buffer_load_dwordx4 v67, s[16:19], s83 offen lds
	s_mov_b32 m0, s47
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[116:119], v[34:37], a[144:147],  v74, v90 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[116:119], v[38:41], a[148:151],  v74, v90 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[116:119], v[42:45], a[152:155],  v74, v91 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[116:119], v[46:49], a[156:159],  v74, v91 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[132:135], v[50:53], a[144:147],  v74, v90 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[132:135], v[54:57], a[148:151],  v74, v90 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[132:135], v[58:61], a[152:155],  v74, v91 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[132:135], v[62:65], a[156:159],  v74, v91 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v94, s[16:19], s83 offen lds
	s_mov_b32 m0, s48
	s_nop 0
	buffer_load_dwordx4 v68, s[16:19], s83 offen lds
	s_mov_b32 m0, s40
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[120:123], v[34:37], a[160:163],  v75, v90 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[120:123], v[38:41], a[164:167],  v75, v90 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[120:123], v[42:45], a[168:171], v75, v91 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[120:123], v[46:49], a[172:175], v75, v91 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[136:139], v[50:53], a[160:163],  v75, v90 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[136:139], v[54:57], a[164:167],  v75, v90 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[136:139], v[58:61], a[168:171], v75, v91 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[136:139], v[62:65], a[172:175], v75, v91 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_u32 s83, s76, 0x280
	buffer_load_dwordx4 v66, s[16:19], s82 offen lds
	s_mov_b32 m0, s49
	s_nop 0
	buffer_load_dwordx4 v67, s[16:19], s82 offen lds
	s_mov_b32 m0, s50
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[124:127], v[34:37], a[176:179], v75, v90 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[124:127], v[38:41], a[180:183], v75, v90 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[124:127], v[42:45], a[184:187], v75, v91 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[124:127], v[46:49], a[188:191], v75, v91 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[140:143], v[50:53], a[176:179], v75, v90 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[140:143], v[54:57], a[180:183], v75, v90 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[140:143], v[58:61], a[184:187], v75, v91 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[140:143], v[62:65], a[188:191], v75, v91 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[90:91], v73, s[0:3], s84 offen
	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v94, s[16:19], s82 offen lds
	s_mov_b32 m0, s51
	s_nop 0
	buffer_load_dwordx4 v68, s[16:19], s82 offen lds
	s_mov_b32 m0, s42
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[112:115], v[144:147], a[192:195],  v74, v78 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v101 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[112:115], v[148:151], a[196:199],  v74, v78 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v101 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[112:115], v[152:155], a[200:203],  v74, v79 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v101 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[112:115], v[156:159], a[204:207],  v74, v79 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v101 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[128:131], v[160:163], a[192:195],  v74, v78 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v102 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[128:131], v[164:167], a[196:199],  v74, v78 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v102 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[128:131], v[168:171], a[200:203],  v74, v79 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v102 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[128:131], v[172:175], a[204:207],  v74, v79 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[62:65], v102 offset:6144

	;;#ASMEND
	s_add_u32 s82, s77, 0x280
	buffer_load_dwordx4 v70, s[20:23], s81 offen lds
	s_mov_b32 m0, s53
	s_nop 0
	buffer_load_dwordx4 v69, s[20:23], s81 offen lds
	s_mov_b32 m0, s54
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[116:119], v[144:147], a[208:211],  v74, v78 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[116:119], v[148:151], a[212:215],  v74, v78 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[116:119], v[152:155], a[216:219],  v74, v79 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[116:119], v[156:159], a[220:223],  v74, v79 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[132:135], v[160:163], a[208:211],  v74, v78 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[132:135], v[164:167], a[212:215],  v74, v78 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[132:135], v[168:171], a[216:219],  v74, v79 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[132:135], v[172:175], a[220:223],  v74, v79 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v71, s[20:23], s81 offen lds
	s_mov_b32 m0, s55
	s_nop 0
	buffer_load_dwordx4 v72, s[20:23], s81 offen lds
	s_mov_b32 m0, s44
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
	s_add_u32 s81, s78, 0x280
	buffer_load_dwordx4 v70, s[20:23], s80 offen lds
	s_mov_b32 m0, s57
	s_nop 0
	buffer_load_dwordx4 v69, s[20:23], s80 offen lds
	s_mov_b32 m0, s58
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[124:127], v[144:147], a[240:243], v75, v78 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[124:127], v[148:151], a[244:247], v75, v78 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[124:127], v[152:155], a[248:251], v75, v79 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[124:127], v[156:159], a[252:255], v75, v79 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[140:143], v[160:163], a[240:243], v75, v78 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[140:143], v[164:167], a[244:247], v75, v78 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[140:143], v[168:171], a[248:251], v75, v79 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[140:143], v[172:175], a[252:255], v75, v79 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[74:75], v73, s[4:7], s84 offen
	;;#ASMEND
	s_add_i32 s84, s35, 0xfffffa00
	buffer_load_dwordx4 v71, s[20:23], s80 offen lds
	s_mov_b32 m0, s59
	;;#ASMSTART
	buffer_load_dwordx2 v[78:79], v73, s[4:7], s84 offen
	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v72, s[20:23], s80 offen lds
	s_mov_b32 m0, s39
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[2:5], v[34:37], a[28:31],  v88, v84 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[144:147], v105 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[2:5], v[38:41], a[32:35],  v88, v84 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[148:151], v105 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[2:5], v[42:45], a[36:39],  v88, v85 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[152:155], v105 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[2:5], v[46:49], a[40:43],  v88, v85 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[156:159], v105 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[18:21], v[50:53], a[28:31],  v88, v84 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[160:163], v106 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[18:21], v[54:57], a[32:35],  v88, v84 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[164:167], v106 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[18:21], v[58:61], a[36:39],  v88, v85 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[168:171], v106 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[18:21], v[62:65], a[40:43],  v88, v85 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[172:175], v106 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[6:9], v[34:37], a[44:47],  v88, v84 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[6:9], v[38:41], a[48:51],  v88, v84 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[6:9], v[42:45], a[52:55],  v88, v85 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[6:9], v[46:49], a[56:59],  v88, v85 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[22:25], v[50:53], a[44:47],  v88, v84 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[22:25], v[54:57], a[48:51],  v88, v84 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[22:25], v[58:61], a[52:55],  v88, v85 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[22:25], v[62:65], a[56:59],  v88, v85 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[10:13], v[34:37], a[60:63],  v89, v84 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[10:13], v[38:41], a[64:67],  v89, v84 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[10:13], v[42:45], a[24:27], v89, v85 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[10:13], v[46:49], a[68:71], v89, v85 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[26:29], v[50:53], a[60:63],  v89, v84 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[26:29], v[54:57], a[64:67],  v89, v84 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[26:29], v[58:61], a[24:27], v89, v85 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[26:29], v[62:65], a[68:71], v89, v85 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[14:17], v[34:37], a[72:75], v89, v84 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[14:17], v[38:41], a[76:79], v89, v84 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[14:17], v[42:45], a[20:23], v89, v85 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[14:17], v[46:49], a[16:19], v89, v85 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[30:33], v[50:53], a[72:75], v89, v84 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[30:33], v[54:57], a[76:79], v89, v84 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[30:33], v[58:61], a[20:23], v89, v85 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[30:33], v[62:65], a[16:19], v89, v85 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[2:5], v[144:147], a[12:15], v88, v80 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[112:115], v109 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[2:5], v[148:151], a[4:7], v88, v80 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[116:119], v109 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[2:5], v[152:155], a[8:11], v88, v81 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[120:123], v109 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[2:5], v[156:159], a[0:3], v88, v81 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[124:127], v109 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[18:21], v[160:163], a[12:15], v88, v80 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[128:131], v110 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[18:21], v[164:167], a[4:7], v88, v80 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[132:135], v110 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[18:21], v[168:171], a[8:11], v88, v81 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[136:139], v110 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[18:21], v[172:175], a[0:3], v88, v81 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[140:143], v110 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[6:9], v[144:147], a[80:83], v88, v80 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[6:9], v[148:151], a[84:87], v88, v80 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[6:9], v[152:155], a[88:91], v88, v81 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[6:9], v[156:159], a[92:95], v88, v81 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[22:25], v[160:163], a[80:83], v88, v80 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[22:25], v[164:167], a[84:87], v88, v80 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[22:25], v[168:171], a[88:91], v88, v81 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[22:25], v[172:175], a[92:95], v88, v81 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[10:13], v[144:147], a[96:99], v89, v80 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[10:13], v[148:151], a[100:103], v89, v80 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[10:13], v[152:155], a[104:107], v89, v81 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[10:13], v[156:159], a[108:111], v89, v81 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[26:29], v[160:163], a[96:99], v89, v80 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[26:29], v[164:167], a[100:103], v89, v80 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[26:29], v[168:171], a[104:107], v89, v81 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[26:29], v[172:175], a[108:111], v89, v81 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[14:17], v[144:147], a[112:115], v89, v80 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[14:17], v[148:151], a[116:119], v89, v80 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[14:17], v[152:155], a[120:123], v89, v81 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[14:17], v[156:159], a[124:127], v89, v81 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[30:33], v[160:163], a[112:115], v89, v80 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[30:33], v[164:167], a[116:119], v89, v80 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[30:33], v[168:171], a[120:123], v89, v81 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[30:33], v[172:175], a[124:127], v89, v81 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(20)
s_barrier

	;;#ASMEND
	s_add_u32 s80, s79, 0x280
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[112:115], v[34:37], a[128:131],  v76, v84 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[2:5], v95 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[112:115], v[38:41], a[132:135],  v76, v84 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[6:9], v95 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[112:115], v[42:45], a[136:139],  v76, v85 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[10:13], v95 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[112:115], v[46:49], a[140:143],  v76, v85 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[14:17], v95 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[128:131], v[50:53], a[128:131],  v76, v84 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[18:21], v96 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[128:131], v[54:57], a[132:135],  v76, v84 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[22:25], v96 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[128:131], v[58:61], a[136:139],  v76, v85 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[26:29], v96 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[128:131], v[62:65], a[140:143],  v76, v85 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v96 offset:6144

	;;#ASMEND
	buffer_load_dwordx4 v66, s[16:19], s83 offen lds
	s_mov_b32 m0, s60
	;;#ASMSTART
	buffer_load_dwordx2 v[88:89], v73, s[0:3], s84 offen
	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v67, s[16:19], s83 offen lds
	s_mov_b32 m0, s61
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[116:119], v[34:37], a[144:147],  v76, v84 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[116:119], v[38:41], a[148:151],  v76, v84 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[116:119], v[42:45], a[152:155],  v76, v85 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[116:119], v[46:49], a[156:159],  v76, v85 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[132:135], v[50:53], a[144:147],  v76, v84 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[132:135], v[54:57], a[148:151],  v76, v84 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[132:135], v[58:61], a[152:155],  v76, v85 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[132:135], v[62:65], a[156:159],  v76, v85 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v94, s[16:19], s83 offen lds
	s_mov_b32 m0, s62
	s_nop 0
	buffer_load_dwordx4 v68, s[16:19], s83 offen lds
	s_mov_b32 m0, s41
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[120:123], v[34:37], a[160:163],  v77, v84 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[120:123], v[38:41], a[164:167],  v77, v84 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[120:123], v[42:45], a[168:171], v77, v85 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[120:123], v[46:49], a[172:175], v77, v85 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[136:139], v[50:53], a[160:163],  v77, v84 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[136:139], v[54:57], a[164:167],  v77, v84 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[136:139], v[58:61], a[168:171], v77, v85 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[136:139], v[62:65], a[172:175], v77, v85 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_u32 s83, s76, 0x300
	buffer_load_dwordx4 v66, s[16:19], s82 offen lds
	s_mov_b32 m0, s63
	s_nop 0
	buffer_load_dwordx4 v67, s[16:19], s82 offen lds
	s_mov_b32 m0, s64
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[124:127], v[34:37], a[176:179], v77, v84 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[124:127], v[38:41], a[180:183], v77, v84 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[124:127], v[42:45], a[184:187], v77, v85 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[124:127], v[46:49], a[188:191], v77, v85 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[140:143], v[50:53], a[176:179], v77, v84 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[140:143], v[54:57], a[180:183], v77, v84 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[140:143], v[58:61], a[184:187], v77, v85 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[140:143], v[62:65], a[188:191], v77, v85 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[84:85], v73, s[8:11], s84 offen
	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v94, s[16:19], s82 offen lds
	s_mov_b32 m0, s65
	s_nop 0
	buffer_load_dwordx4 v68, s[16:19], s82 offen lds
	s_mov_b32 m0, s43
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[112:115], v[144:147], a[192:195],  v76, v80 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v99 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[112:115], v[148:151], a[196:199],  v76, v80 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v99 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[112:115], v[152:155], a[200:203],  v76, v81 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v99 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[112:115], v[156:159], a[204:207],  v76, v81 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v99 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[128:131], v[160:163], a[192:195],  v76, v80 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v100 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[128:131], v[164:167], a[196:199],  v76, v80 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v100 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[128:131], v[168:171], a[200:203],  v76, v81 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v100 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[128:131], v[172:175], a[204:207],  v76, v81 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[62:65], v100 offset:6144

	;;#ASMEND
	s_add_u32 s82, s77, 0x300
	buffer_load_dwordx4 v70, s[20:23], s81 offen lds
	s_mov_b32 m0, s66
	s_nop 0
	buffer_load_dwordx4 v69, s[20:23], s81 offen lds
	s_mov_b32 m0, s67
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[116:119], v[144:147], a[208:211],  v76, v80 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[116:119], v[148:151], a[212:215],  v76, v80 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[116:119], v[152:155], a[216:219],  v76, v81 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[116:119], v[156:159], a[220:223],  v76, v81 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[132:135], v[160:163], a[208:211],  v76, v80 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[132:135], v[164:167], a[212:215],  v76, v80 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[132:135], v[168:171], a[216:219],  v76, v81 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[132:135], v[172:175], a[220:223],  v76, v81 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v71, s[20:23], s81 offen lds
	s_mov_b32 m0, s68
	s_nop 0
	buffer_load_dwordx4 v72, s[20:23], s81 offen lds
	s_mov_b32 m0, s45
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
	buffer_load_dwordx4 v70, s[20:23], s80 offen lds
	s_mov_b32 m0, s69
	s_nop 0
	buffer_load_dwordx4 v69, s[20:23], s80 offen lds
	s_mov_b32 m0, s70
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[124:127], v[144:147], a[240:243], v77, v80 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[124:127], v[148:151], a[244:247], v77, v80 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[124:127], v[152:155], a[248:251], v77, v81 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[124:127], v[156:159], a[252:255], v77, v81 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[140:143], v[160:163], a[240:243], v77, v80 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[140:143], v[164:167], a[244:247], v77, v80 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[140:143], v[168:171], a[248:251], v77, v81 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[140:143], v[172:175], a[252:255], v77, v81 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[80:81], v73, s[12:15], s84 offen
	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v71, s[20:23], s80 offen lds
	s_mov_b32 m0, s71
	s_nop 0
	buffer_load_dwordx4 v72, s[20:23], s80 offen lds
	s_mov_b32 m0, s33
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[2:5], v[34:37], a[28:31],  v90, v86 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[144:147], v103 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[2:5], v[38:41], a[32:35],  v90, v86 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[148:151], v103 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[2:5], v[42:45], a[36:39],  v90, v87 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[152:155], v103 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[2:5], v[46:49], a[40:43],  v90, v87 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[156:159], v103 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[18:21], v[50:53], a[28:31],  v90, v86 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[160:163], v104 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[18:21], v[54:57], a[32:35],  v90, v86 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[164:167], v104 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[18:21], v[58:61], a[36:39],  v90, v87 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[168:171], v104 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[18:21], v[62:65], a[40:43],  v90, v87 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[172:175], v104 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[6:9], v[34:37], a[44:47],  v90, v86 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[6:9], v[38:41], a[48:51],  v90, v86 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[6:9], v[42:45], a[52:55],  v90, v87 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[6:9], v[46:49], a[56:59],  v90, v87 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[22:25], v[50:53], a[44:47],  v90, v86 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[22:25], v[54:57], a[48:51],  v90, v86 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[22:25], v[58:61], a[52:55],  v90, v87 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[22:25], v[62:65], a[56:59],  v90, v87 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[10:13], v[34:37], a[60:63],  v91, v86 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[10:13], v[38:41], a[64:67],  v91, v86 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[10:13], v[42:45], a[24:27], v91, v87 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[10:13], v[46:49], a[68:71], v91, v87 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[26:29], v[50:53], a[60:63],  v91, v86 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[26:29], v[54:57], a[64:67],  v91, v86 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[26:29], v[58:61], a[24:27], v91, v87 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[26:29], v[62:65], a[68:71], v91, v87 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[14:17], v[34:37], a[72:75], v91, v86 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[14:17], v[38:41], a[76:79], v91, v86 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[14:17], v[42:45], a[20:23], v91, v87 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[14:17], v[46:49], a[16:19], v91, v87 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[30:33], v[50:53], a[72:75], v91, v86 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[30:33], v[54:57], a[76:79], v91, v86 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[30:33], v[58:61], a[20:23], v91, v87 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[30:33], v[62:65], a[16:19], v91, v87 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[2:5], v[144:147], a[12:15], v90, v82 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[112:115], v107 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[2:5], v[148:151], a[4:7], v90, v82 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[116:119], v107 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[2:5], v[152:155], a[8:11], v90, v83 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[120:123], v107 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[2:5], v[156:159], a[0:3], v90, v83 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[124:127], v107 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[18:21], v[160:163], a[12:15], v90, v82 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[128:131], v108 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[18:21], v[164:167], a[4:7], v90, v82 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[132:135], v108 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[18:21], v[168:171], a[8:11], v90, v83 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[136:139], v108 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[18:21], v[172:175], a[0:3], v90, v83 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[140:143], v108 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[6:9], v[144:147], a[80:83], v90, v82 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[6:9], v[148:151], a[84:87], v90, v82 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[6:9], v[152:155], a[88:91], v90, v83 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[6:9], v[156:159], a[92:95], v90, v83 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[22:25], v[160:163], a[80:83], v90, v82 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[22:25], v[164:167], a[84:87], v90, v82 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[22:25], v[168:171], a[88:91], v90, v83 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[22:25], v[172:175], a[92:95], v90, v83 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[10:13], v[144:147], a[96:99], v91, v82 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[10:13], v[148:151], a[100:103], v91, v82 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[10:13], v[152:155], a[104:107], v91, v83 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[10:13], v[156:159], a[108:111], v91, v83 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[26:29], v[160:163], a[96:99], v91, v82 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[26:29], v[164:167], a[100:103], v91, v82 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[26:29], v[168:171], a[104:107], v91, v83 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[26:29], v[172:175], a[108:111], v91, v83 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[14:17], v[144:147], a[112:115], v91, v82 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[14:17], v[148:151], a[116:119], v91, v82 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[14:17], v[152:155], a[120:123], v91, v83 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[14:17], v[156:159], a[124:127], v91, v83 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[30:33], v[160:163], a[112:115], v91, v82 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[30:33], v[164:167], a[116:119], v91, v82 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[30:33], v[168:171], a[120:123], v91, v83 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[30:33], v[172:175], a[124:127], v91, v83 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(20)
s_barrier

	;;#ASMEND
	s_add_u32 s80, s79, 0x300
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[112:115], v[34:37], a[128:131],  v74, v86 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[2:5], v97 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[112:115], v[38:41], a[132:135],  v74, v86 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[6:9], v97 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[112:115], v[42:45], a[136:139],  v74, v87 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[10:13], v97 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[112:115], v[46:49], a[140:143],  v74, v87 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[14:17], v97 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[128:131], v[50:53], a[128:131],  v74, v86 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[18:21], v98 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[128:131], v[54:57], a[132:135],  v74, v86 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[22:25], v98 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[128:131], v[58:61], a[136:139],  v74, v87 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[26:29], v98 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[128:131], v[62:65], a[140:143],  v74, v87 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v98 offset:6144

	;;#ASMEND
	buffer_load_dwordx4 v66, s[16:19], s83 offen lds
	s_mov_b32 m0, s46
	s_nop 0
	buffer_load_dwordx4 v67, s[16:19], s83 offen lds
	s_mov_b32 m0, s47
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[116:119], v[34:37], a[144:147],  v74, v86 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[116:119], v[38:41], a[148:151],  v74, v86 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[116:119], v[42:45], a[152:155],  v74, v87 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[116:119], v[46:49], a[156:159],  v74, v87 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[132:135], v[50:53], a[144:147],  v74, v86 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[132:135], v[54:57], a[148:151],  v74, v86 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[132:135], v[58:61], a[152:155],  v74, v87 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[132:135], v[62:65], a[156:159],  v74, v87 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v94, s[16:19], s83 offen lds
	s_mov_b32 m0, s48
	s_nop 0
	buffer_load_dwordx4 v68, s[16:19], s83 offen lds
	s_mov_b32 m0, s40
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[120:123], v[34:37], a[160:163],  v75, v86 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[120:123], v[38:41], a[164:167],  v75, v86 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[120:123], v[42:45], a[168:171], v75, v87 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[120:123], v[46:49], a[172:175], v75, v87 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[136:139], v[50:53], a[160:163],  v75, v86 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[136:139], v[54:57], a[164:167],  v75, v86 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[136:139], v[58:61], a[168:171], v75, v87 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[136:139], v[62:65], a[172:175], v75, v87 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v66, s[16:19], s82 offen lds
	s_mov_b32 m0, s49
	s_nop 0
	buffer_load_dwordx4 v67, s[16:19], s82 offen lds
	s_mov_b32 m0, s50
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[124:127], v[34:37], a[176:179], v75, v86 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[124:127], v[38:41], a[180:183], v75, v86 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[124:127], v[42:45], a[184:187], v75, v87 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[124:127], v[46:49], a[188:191], v75, v87 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[140:143], v[50:53], a[176:179], v75, v86 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[140:143], v[54:57], a[180:183], v75, v86 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[140:143], v[58:61], a[184:187], v75, v87 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[140:143], v[62:65], a[188:191], v75, v87 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v94, s[16:19], s82 offen lds
	s_mov_b32 m0, s51
	s_nop 0
	buffer_load_dwordx4 v68, s[16:19], s82 offen lds
	s_mov_b32 m0, s42
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[112:115], v[144:147], a[192:195],  v74, v82 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v101 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[112:115], v[148:151], a[196:199],  v74, v82 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v101 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[112:115], v[152:155], a[200:203],  v74, v83 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v101 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[112:115], v[156:159], a[204:207],  v74, v83 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v101 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[128:131], v[160:163], a[192:195],  v74, v82 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v102 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[128:131], v[164:167], a[196:199],  v74, v82 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v102 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[128:131], v[168:171], a[200:203],  v74, v83 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v102 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[128:131], v[172:175], a[204:207],  v74, v83 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[62:65], v102 offset:6144

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v70, s[20:23], s81 offen lds
	s_mov_b32 m0, s53
	s_nop 0
	buffer_load_dwordx4 v69, s[20:23], s81 offen lds
	s_mov_b32 m0, s54
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[116:119], v[144:147], a[208:211],  v74, v82 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[116:119], v[148:151], a[212:215],  v74, v82 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[116:119], v[152:155], a[216:219],  v74, v83 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[116:119], v[156:159], a[220:223],  v74, v83 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[132:135], v[160:163], a[208:211],  v74, v82 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[132:135], v[164:167], a[212:215],  v74, v82 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[132:135], v[168:171], a[216:219],  v74, v83 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[132:135], v[172:175], a[220:223],  v74, v83 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v71, s[20:23], s81 offen lds
	s_mov_b32 m0, s55
	s_nop 0
	buffer_load_dwordx4 v72, s[20:23], s81 offen lds
	s_mov_b32 m0, s44
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[120:123], v[144:147], a[224:227],  v75, v82 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[120:123], v[148:151], a[228:231],  v75, v82 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[120:123], v[152:155], a[232:235], v75, v83 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[120:123], v[156:159], a[236:239], v75, v83 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[136:139], v[160:163], a[224:227],  v75, v82 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[136:139], v[164:167], a[228:231],  v75, v82 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[136:139], v[168:171], a[232:235], v75, v83 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[136:139], v[172:175], a[236:239], v75, v83 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_u32 s81, s76, 0x380
	buffer_load_dwordx4 v70, s[20:23], s80 offen lds
	s_mov_b32 m0, s57
	s_add_u32 s76, s79, 0x380
	buffer_load_dwordx4 v69, s[20:23], s80 offen lds
	s_mov_b32 m0, s58
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[124:127], v[144:147], a[240:243], v75, v82 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[124:127], v[148:151], a[244:247], v75, v82 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[124:127], v[152:155], a[248:251], v75, v83 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[124:127], v[156:159], a[252:255], v75, v83 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[140:143], v[160:163], a[240:243], v75, v82 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[140:143], v[164:167], a[244:247], v75, v82 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[140:143], v[168:171], a[248:251], v75, v83 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[140:143], v[172:175], a[252:255], v75, v83 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_i32 s79, s35, 0xfffffc00
	buffer_load_dwordx4 v71, s[20:23], s80 offen lds
	s_mov_b32 m0, s59
	;;#ASMSTART
	buffer_load_dwordx2 v[92:93], v73, s[0:3], s79 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[74:75], v73, s[4:7], s79 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[90:91], v73, s[8:11], s79 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[76:77], v73, s[12:15], s79 offen
	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v72, s[20:23], s80 offen lds
	s_mov_b32 m0, s39
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[2:5], v[34:37], a[28:31],  v88, v84 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[144:147], v105 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[2:5], v[38:41], a[32:35],  v88, v84 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[148:151], v105 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[2:5], v[42:45], a[36:39],  v88, v85 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[152:155], v105 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[2:5], v[46:49], a[40:43],  v88, v85 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[156:159], v105 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[18:21], v[50:53], a[28:31],  v88, v84 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[160:163], v106 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[18:21], v[54:57], a[32:35],  v88, v84 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[164:167], v106 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[18:21], v[58:61], a[36:39],  v88, v85 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[168:171], v106 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[18:21], v[62:65], a[40:43],  v88, v85 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[172:175], v106 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[6:9], v[34:37], a[44:47],  v88, v84 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[6:9], v[38:41], a[48:51],  v88, v84 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[6:9], v[42:45], a[52:55],  v88, v85 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[6:9], v[46:49], a[56:59],  v88, v85 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[22:25], v[50:53], a[44:47],  v88, v84 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[22:25], v[54:57], a[48:51],  v88, v84 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[22:25], v[58:61], a[52:55],  v88, v85 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[22:25], v[62:65], a[56:59],  v88, v85 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[10:13], v[34:37], a[60:63],  v89, v84 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[10:13], v[38:41], a[64:67],  v89, v84 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[10:13], v[42:45], a[24:27], v89, v85 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[10:13], v[46:49], a[68:71], v89, v85 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[26:29], v[50:53], a[60:63],  v89, v84 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[26:29], v[54:57], a[64:67],  v89, v84 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[26:29], v[58:61], a[24:27], v89, v85 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[26:29], v[62:65], a[68:71], v89, v85 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[14:17], v[34:37], a[72:75], v89, v84 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[14:17], v[38:41], a[76:79], v89, v84 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[14:17], v[42:45], a[20:23], v89, v85 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[14:17], v[46:49], a[16:19], v89, v85 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[30:33], v[50:53], a[72:75], v89, v84 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[30:33], v[54:57], a[76:79], v89, v84 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[30:33], v[58:61], a[20:23], v89, v85 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[30:33], v[62:65], a[16:19], v89, v85 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[2:5], v[144:147], a[12:15], v88, v80 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[112:115], v109 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[2:5], v[148:151], a[4:7], v88, v80 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[116:119], v109 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[2:5], v[152:155], a[8:11], v88, v81 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[120:123], v109 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[2:5], v[156:159], a[0:3], v88, v81 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[124:127], v109 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[18:21], v[160:163], a[12:15], v88, v80 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[128:131], v110 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[18:21], v[164:167], a[4:7], v88, v80 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[132:135], v110 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[18:21], v[168:171], a[8:11], v88, v81 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[136:139], v110 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[18:21], v[172:175], a[0:3], v88, v81 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[140:143], v110 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[6:9], v[144:147], a[80:83], v88, v80 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[6:9], v[148:151], a[84:87], v88, v80 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[6:9], v[152:155], a[88:91], v88, v81 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[6:9], v[156:159], a[92:95], v88, v81 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[22:25], v[160:163], a[80:83], v88, v80 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[22:25], v[164:167], a[84:87], v88, v80 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[22:25], v[168:171], a[88:91], v88, v81 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[22:25], v[172:175], a[92:95], v88, v81 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[10:13], v[144:147], a[96:99], v89, v80 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[10:13], v[148:151], a[100:103], v89, v80 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[10:13], v[152:155], a[104:107], v89, v81 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[10:13], v[156:159], a[108:111], v89, v81 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[26:29], v[160:163], a[96:99], v89, v80 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[26:29], v[164:167], a[100:103], v89, v80 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[26:29], v[168:171], a[104:107], v89, v81 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[26:29], v[172:175], a[108:111], v89, v81 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[14:17], v[144:147], a[112:115], v89, v80 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[14:17], v[148:151], a[116:119], v89, v80 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[14:17], v[152:155], a[120:123], v89, v81 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[14:17], v[156:159], a[124:127], v89, v81 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[30:33], v[160:163], a[112:115], v89, v80 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[30:33], v[164:167], a[116:119], v89, v80 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[30:33], v[168:171], a[120:123], v89, v81 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[30:33], v[172:175], a[124:127], v89, v81 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(20)
s_barrier

	;;#ASMEND
	s_add_u32 s80, s77, 0x380
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[112:115], v[34:37], a[128:131],  v78, v84 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[2:5], v95 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[112:115], v[38:41], a[132:135],  v78, v84 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[6:9], v95 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[112:115], v[42:45], a[136:139],  v78, v85 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[10:13], v95 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[112:115], v[46:49], a[140:143],  v78, v85 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[14:17], v95 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[128:131], v[50:53], a[128:131],  v78, v84 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[18:21], v96 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[128:131], v[54:57], a[132:135],  v78, v84 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[22:25], v96 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[128:131], v[58:61], a[136:139],  v78, v85 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[26:29], v96 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[128:131], v[62:65], a[140:143],  v78, v85 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v96 offset:6144

	;;#ASMEND
	buffer_load_dwordx4 v66, s[16:19], s81 offen lds
	s_mov_b32 m0, s60
	s_add_u32 s77, s78, 0x380
	buffer_load_dwordx4 v67, s[16:19], s81 offen lds
	s_mov_b32 m0, s61
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[116:119], v[34:37], a[144:147],  v78, v84 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[116:119], v[38:41], a[148:151],  v78, v84 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[116:119], v[42:45], a[152:155],  v78, v85 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[116:119], v[46:49], a[156:159],  v78, v85 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[132:135], v[50:53], a[144:147],  v78, v84 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[132:135], v[54:57], a[148:151],  v78, v84 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[132:135], v[58:61], a[152:155],  v78, v85 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[132:135], v[62:65], a[156:159],  v78, v85 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_i32 s78, s29, -1
	buffer_load_dwordx4 v94, s[16:19], s81 offen lds
	s_mov_b32 m0, s62
	s_nop 0
	buffer_load_dwordx4 v68, s[16:19], s81 offen lds
	s_mov_b32 m0, s41
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[120:123], v[34:37], a[160:163],  v79, v84 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[120:123], v[38:41], a[164:167],  v79, v84 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[120:123], v[42:45], a[168:171], v79, v85 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[120:123], v[46:49], a[172:175], v79, v85 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[136:139], v[50:53], a[160:163],  v79, v84 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[136:139], v[54:57], a[164:167],  v79, v84 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[136:139], v[58:61], a[168:171], v79, v85 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[136:139], v[62:65], a[172:175], v79, v85 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v66, s[16:19], s80 offen lds
	s_mov_b32 m0, s63
	s_nop 0
	buffer_load_dwordx4 v67, s[16:19], s80 offen lds
	s_mov_b32 m0, s64
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[124:127], v[34:37], a[176:179], v79, v84 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[124:127], v[38:41], a[180:183], v79, v84 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[124:127], v[42:45], a[184:187], v79, v85 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[124:127], v[46:49], a[188:191], v79, v85 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[140:143], v[50:53], a[176:179], v79, v84 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[140:143], v[54:57], a[180:183], v79, v84 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[140:143], v[58:61], a[184:187], v79, v85 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[140:143], v[62:65], a[188:191], v79, v85 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v94, s[16:19], s80 offen lds
	s_mov_b32 m0, s65
	s_nop 0
	buffer_load_dwordx4 v68, s[16:19], s80 offen lds
	s_mov_b32 m0, s43
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[112:115], v[144:147], a[192:195],  v78, v80 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v99 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[112:115], v[148:151], a[196:199],  v78, v80 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v99 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[112:115], v[152:155], a[200:203],  v78, v81 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v99 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[112:115], v[156:159], a[204:207],  v78, v81 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v99 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[128:131], v[160:163], a[192:195],  v78, v80 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v100 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[128:131], v[164:167], a[196:199],  v78, v80 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v100 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[128:131], v[168:171], a[200:203],  v78, v81 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v100 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[128:131], v[172:175], a[204:207],  v78, v81 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[62:65], v100 offset:6144

	;;#ASMEND
	s_add_i32 s80, s35, 0xfffffe00
	buffer_load_dwordx4 v70, s[20:23], s77 offen lds
	s_mov_b32 m0, s66
	;;#ASMSTART
	buffer_load_dwordx2 v[88:89], v73, s[0:3], s80 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[82:83], v73, s[4:7], s80 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[86:87], v73, s[8:11], s80 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[84:85], v73, s[12:15], s80 offen
	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v69, s[20:23], s77 offen lds
	s_mov_b32 m0, s67
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[116:119], v[144:147], a[208:211],  v78, v80 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[116:119], v[148:151], a[212:215],  v78, v80 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[116:119], v[152:155], a[216:219],  v78, v81 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[116:119], v[156:159], a[220:223],  v78, v81 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[132:135], v[160:163], a[208:211],  v78, v80 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[132:135], v[164:167], a[212:215],  v78, v80 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[132:135], v[168:171], a[216:219],  v78, v81 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[132:135], v[172:175], a[220:223],  v78, v81 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v71, s[20:23], s77 offen lds
	s_mov_b32 m0, s68
	s_nop 0
	buffer_load_dwordx4 v72, s[20:23], s77 offen lds
	s_mov_b32 m0, s45
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[120:123], v[144:147], a[224:227],  v79, v80 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[120:123], v[148:151], a[228:231],  v79, v80 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[120:123], v[152:155], a[232:235], v79, v81 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[120:123], v[156:159], a[236:239], v79, v81 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[136:139], v[160:163], a[224:227],  v79, v80 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[136:139], v[164:167], a[228:231],  v79, v80 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[136:139], v[168:171], a[232:235], v79, v81 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[136:139], v[172:175], a[236:239], v79, v81 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v70, s[20:23], s76 offen lds
	s_mov_b32 m0, s69
	s_nop 0
	buffer_load_dwordx4 v69, s[20:23], s76 offen lds
	s_mov_b32 m0, s70
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[124:127], v[144:147], a[240:243], v79, v80 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[124:127], v[148:151], a[244:247], v79, v80 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[124:127], v[152:155], a[248:251], v79, v81 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[124:127], v[156:159], a[252:255], v79, v81 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[140:143], v[160:163], a[240:243], v79, v80 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[140:143], v[164:167], a[244:247], v79, v80 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[140:143], v[168:171], a[248:251], v79, v81 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[140:143], v[172:175], a[252:255], v79, v81 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v71, s[20:23], s76 offen lds
	s_mov_b32 m0, s71
	s_nop 0
	buffer_load_dwordx4 v72, s[20:23], s76 offen lds
	s_min_u32 s76, s78, 0x6d
	s_lshl_b32 s76, s76, 7
	s_add_u32 s77, s72, s76
	s_sub_u32 s79, s77, s30
	s_mov_b32 m0, s33
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[2:5], v[34:37], a[28:31],  v92, v90 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[140:143], v103 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[2:5], v[38:41], a[32:35],  v92, v90 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[144:147], v103 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[2:5], v[42:45], a[36:39],  v92, v91 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[148:151], v103 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[2:5], v[46:49], a[40:43],  v92, v91 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[152:155], v103 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[18:21], v[50:53], a[28:31],  v92, v90 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[156:159], v104 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[18:21], v[54:57], a[32:35],  v92, v90 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[160:163], v104 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[18:21], v[58:61], a[36:39],  v92, v91 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[164:167], v104 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[18:21], v[62:65], a[40:43],  v92, v91 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[168:171], v104 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[6:9], v[34:37], a[44:47],  v92, v90 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[6:9], v[38:41], a[48:51],  v92, v90 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[6:9], v[42:45], a[52:55],  v92, v91 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[6:9], v[46:49], a[56:59],  v92, v91 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[22:25], v[50:53], a[44:47],  v92, v90 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[22:25], v[54:57], a[48:51],  v92, v90 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[22:25], v[58:61], a[52:55],  v92, v91 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[22:25], v[62:65], a[56:59],  v92, v91 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[10:13], v[34:37], a[60:63],  v93, v90 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[10:13], v[38:41], a[64:67],  v93, v90 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[10:13], v[42:45], a[24:27], v93, v91 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[10:13], v[46:49], a[68:71], v93, v91 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[26:29], v[50:53], a[60:63],  v93, v90 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[26:29], v[54:57], a[64:67],  v93, v90 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[26:29], v[58:61], a[24:27], v93, v91 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[26:29], v[62:65], a[68:71], v93, v91 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[14:17], v[34:37], a[72:75], v93, v90 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[14:17], v[38:41], a[76:79], v93, v90 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[14:17], v[42:45], a[20:23], v93, v91 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[14:17], v[46:49], a[16:19], v93, v91 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[30:33], v[50:53], a[72:75], v93, v90 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[30:33], v[54:57], a[76:79], v93, v90 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[30:33], v[58:61], a[20:23], v93, v91 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[30:33], v[62:65], a[16:19], v93, v91 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[2:5], v[140:143], a[12:15], v92, v76 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[78:81], v107 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[2:5], v[144:147], a[4:7], v92, v76 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[112:115], v107 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[2:5], v[148:151], a[8:11], v92, v77 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[116:119], v107 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[2:5], v[152:155], a[0:3], v92, v77 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[120:123], v107 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[18:21], v[156:159], a[12:15], v92, v76 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[124:127], v108 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[18:21], v[160:163], a[4:7], v92, v76 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[128:131], v108 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[18:21], v[164:167], a[8:11], v92, v77 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[132:135], v108 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[18:21], v[168:171], a[0:3], v92, v77 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[136:139], v108 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[6:9], v[140:143], a[80:83], v92, v76 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[6:9], v[144:147], a[84:87], v92, v76 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[6:9], v[148:151], a[88:91], v92, v77 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[6:9], v[152:155], a[92:95], v92, v77 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[22:25], v[156:159], a[80:83], v92, v76 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[22:25], v[160:163], a[84:87], v92, v76 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[22:25], v[164:167], a[88:91], v92, v77 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[22:25], v[168:171], a[92:95], v92, v77 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[10:13], v[140:143], a[96:99], v93, v76 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[10:13], v[144:147], a[100:103], v93, v76 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[10:13], v[148:151], a[104:107], v93, v77 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[10:13], v[152:155], a[108:111], v93, v77 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[26:29], v[156:159], a[96:99], v93, v76 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[26:29], v[160:163], a[100:103], v93, v76 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[26:29], v[164:167], a[104:107], v93, v77 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[26:29], v[168:171], a[108:111], v93, v77 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[14:17], v[140:143], a[112:115], v93, v76 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[14:17], v[144:147], a[116:119], v93, v76 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[14:17], v[148:151], a[120:123], v93, v77 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[14:17], v[152:155], a[124:127], v93, v77 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[30:33], v[156:159], a[112:115], v93, v76 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[30:33], v[160:163], a[116:119], v93, v76 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[30:33], v[164:167], a[120:123], v93, v77 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[30:33], v[168:171], a[124:127], v93, v77 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(20)
s_barrier

	;;#ASMEND
	s_add_u32 s77, s73, s76
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[78:81], v[34:37], a[128:131],  v74, v90 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[2:5], v97 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[78:81], v[38:41], a[132:135],  v74, v90 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[6:9], v97 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[78:81], v[42:45], a[136:139],  v74, v91 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[10:13], v97 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[78:81], v[46:49], a[140:143],  v74, v91 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[14:17], v97 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[124:127], v[50:53], a[128:131],  v74, v90 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[18:21], v98 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[124:127], v[54:57], a[132:135],  v74, v90 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[22:25], v98 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[124:127], v[58:61], a[136:139],  v74, v91 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[26:29], v98 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[124:127], v[62:65], a[140:143],  v74, v91 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v98 offset:6144

	;;#ASMEND
	buffer_load_dwordx4 v66, s[16:19], s79 offen lds
	s_mov_b32 m0, s46
	s_sub_u32 s78, s77, s30
	buffer_load_dwordx4 v67, s[16:19], s79 offen lds
	s_mov_b32 m0, s47
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[112:115], v[34:37], a[144:147],  v74, v90 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[112:115], v[38:41], a[148:151],  v74, v90 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[112:115], v[42:45], a[152:155],  v74, v91 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[112:115], v[46:49], a[156:159],  v74, v91 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[128:131], v[50:53], a[144:147],  v74, v90 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[128:131], v[54:57], a[148:151],  v74, v90 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[128:131], v[58:61], a[152:155],  v74, v91 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[128:131], v[62:65], a[156:159],  v74, v91 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_u32 s77, s74, s76
	buffer_load_dwordx4 v94, s[16:19], s79 offen lds
	s_mov_b32 m0, s48
	s_sub_u32 s77, s77, s34
	buffer_load_dwordx4 v68, s[16:19], s79 offen lds
	s_mov_b32 m0, s40
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[116:119], v[34:37], a[160:163],  v75, v90 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[116:119], v[38:41], a[164:167],  v75, v90 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[116:119], v[42:45], a[168:171], v75, v91 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[116:119], v[46:49], a[172:175], v75, v91 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[132:135], v[50:53], a[160:163],  v75, v90 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[132:135], v[54:57], a[164:167],  v75, v90 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[132:135], v[58:61], a[168:171], v75, v91 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[132:135], v[62:65], a[172:175], v75, v91 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_u32 s76, s75, s76
	buffer_load_dwordx4 v66, s[16:19], s78 offen lds
	s_mov_b32 m0, s49
	s_sub_u32 s76, s76, s34
	buffer_load_dwordx4 v67, s[16:19], s78 offen lds
	s_mov_b32 m0, s50
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[120:123], v[34:37], a[176:179], v75, v90 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[120:123], v[38:41], a[180:183], v75, v90 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[120:123], v[42:45], a[184:187], v75, v91 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[120:123], v[46:49], a[188:191], v75, v91 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[136:139], v[50:53], a[176:179], v75, v90 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[136:139], v[54:57], a[180:183], v75, v90 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[136:139], v[58:61], a[184:187], v75, v91 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[136:139], v[62:65], a[188:191], v75, v91 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v94, s[16:19], s78 offen lds
	s_mov_b32 m0, s51
	s_nop 0
	buffer_load_dwordx4 v68, s[16:19], s78 offen lds
	s_mov_b32 m0, s42
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[78:81], v[140:143], a[192:195],  v74, v76 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v101 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[78:81], v[144:147], a[196:199],  v74, v76 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v101 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[78:81], v[148:151], a[200:203],  v74, v77 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v101 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[78:81], v[152:155], a[204:207],  v74, v77 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v101 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[124:127], v[156:159], a[192:195],  v74, v76 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v102 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[124:127], v[160:163], a[196:199],  v74, v76 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v102 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[124:127], v[164:167], a[200:203],  v74, v77 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v102 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[124:127], v[168:171], a[204:207],  v74, v77 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[62:65], v102 offset:6144

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v70, s[20:23], s77 offen lds
	s_mov_b32 m0, s53
	s_nop 0
	buffer_load_dwordx4 v69, s[20:23], s77 offen lds
	s_mov_b32 m0, s54
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[112:115], v[140:143], a[208:211],  v74, v76 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[112:115], v[144:147], a[212:215],  v74, v76 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[112:115], v[148:151], a[216:219],  v74, v77 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[112:115], v[152:155], a[220:223],  v74, v77 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[128:131], v[156:159], a[208:211],  v74, v76 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[128:131], v[160:163], a[212:215],  v74, v76 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[128:131], v[164:167], a[216:219],  v74, v77 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[128:131], v[168:171], a[220:223],  v74, v77 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v71, s[20:23], s77 offen lds
	s_mov_b32 m0, s55
	s_nop 0
	buffer_load_dwordx4 v72, s[20:23], s77 offen lds
	s_mov_b32 m0, s44
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
	buffer_load_dwordx4 v70, s[20:23], s76 offen lds
	s_mov_b32 m0, s57
	s_nop 0
	buffer_load_dwordx4 v69, s[20:23], s76 offen lds
	s_mov_b32 m0, s58
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[120:123], v[140:143], a[240:243], v75, v76 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[120:123], v[144:147], a[244:247], v75, v76 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[120:123], v[148:151], a[248:251], v75, v77 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[120:123], v[152:155], a[252:255], v75, v77 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[136:139], v[156:159], a[240:243], v75, v76 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[136:139], v[160:163], a[244:247], v75, v76 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[136:139], v[164:167], a[248:251], v75, v77 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[136:139], v[168:171], a[252:255], v75, v77 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v71, s[20:23], s76 offen lds
	s_mov_b32 m0, s59
	s_nop 0
	buffer_load_dwordx4 v72, s[20:23], s76 offen lds
	s_min_u32 s76, s29, 0x6d
	s_lshl_b32 s79, s76, 7
	s_add_u32 s76, s72, s79
	s_sub_u32 s78, s76, s30
	s_mov_b32 m0, s39
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[2:5], v[34:37], a[28:31],  v88, v86 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[140:143], v105 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[2:5], v[38:41], a[32:35],  v88, v86 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[144:147], v105 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[2:5], v[42:45], a[36:39],  v88, v87 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[148:151], v105 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[2:5], v[46:49], a[40:43],  v88, v87 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[152:155], v105 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[18:21], v[50:53], a[28:31],  v88, v86 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[156:159], v106 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[18:21], v[54:57], a[32:35],  v88, v86 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[160:163], v106 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[18:21], v[58:61], a[36:39],  v88, v87 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[164:167], v106 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[18:21], v[62:65], a[40:43],  v88, v87 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[168:171], v106 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[6:9], v[34:37], a[44:47],  v88, v86 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[6:9], v[38:41], a[48:51],  v88, v86 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[6:9], v[42:45], a[52:55],  v88, v87 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[6:9], v[46:49], a[56:59],  v88, v87 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[22:25], v[50:53], a[44:47],  v88, v86 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[22:25], v[54:57], a[48:51],  v88, v86 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[22:25], v[58:61], a[52:55],  v88, v87 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[22:25], v[62:65], a[56:59],  v88, v87 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[10:13], v[34:37], a[60:63],  v89, v86 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[10:13], v[38:41], a[64:67],  v89, v86 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[10:13], v[42:45], a[24:27], v89, v87 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[10:13], v[46:49], a[68:71], v89, v87 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[26:29], v[50:53], a[60:63],  v89, v86 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[26:29], v[54:57], a[64:67],  v89, v86 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[26:29], v[58:61], a[24:27], v89, v87 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[26:29], v[62:65], a[68:71], v89, v87 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[14:17], v[34:37], a[72:75], v89, v86 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[14:17], v[38:41], a[76:79], v89, v86 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[14:17], v[42:45], a[20:23], v89, v87 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[14:17], v[46:49], a[16:19], v89, v87 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[30:33], v[50:53], a[72:75], v89, v86 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[30:33], v[54:57], a[76:79], v89, v86 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[30:33], v[58:61], a[20:23], v89, v87 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[30:33], v[62:65], a[16:19], v89, v87 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[2:5], v[140:143], a[12:15], v88, v84 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[90:93], v109 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[2:5], v[144:147], a[4:7], v88, v84 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[112:115], v109 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[2:5], v[148:151], a[8:11], v88, v85 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[116:119], v109 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[2:5], v[152:155], a[0:3], v88, v85 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[120:123], v109 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[18:21], v[156:159], a[12:15], v88, v84 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[124:127], v110 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[18:21], v[160:163], a[4:7], v88, v84 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[128:131], v110 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[18:21], v[164:167], a[8:11], v88, v85 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[132:135], v110 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[18:21], v[168:171], a[0:3], v88, v85 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[136:139], v110 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[6:9], v[140:143], a[80:83], v88, v84 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[6:9], v[144:147], a[84:87], v88, v84 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[6:9], v[148:151], a[88:91], v88, v85 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[6:9], v[152:155], a[92:95], v88, v85 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[22:25], v[156:159], a[80:83], v88, v84 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[22:25], v[160:163], a[84:87], v88, v84 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[22:25], v[164:167], a[88:91], v88, v85 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[22:25], v[168:171], a[92:95], v88, v85 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[10:13], v[140:143], a[96:99], v89, v84 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[10:13], v[144:147], a[100:103], v89, v84 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[10:13], v[148:151], a[104:107], v89, v85 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[10:13], v[152:155], a[108:111], v89, v85 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[26:29], v[156:159], a[96:99], v89, v84 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[26:29], v[160:163], a[100:103], v89, v84 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[26:29], v[164:167], a[104:107], v89, v85 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[26:29], v[168:171], a[108:111], v89, v85 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[14:17], v[140:143], a[112:115], v89, v84 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[14:17], v[144:147], a[116:119], v89, v84 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[14:17], v[148:151], a[120:123], v89, v85 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[14:17], v[152:155], a[124:127], v89, v85 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[30:33], v[156:159], a[112:115], v89, v84 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[30:33], v[160:163], a[116:119], v89, v84 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[30:33], v[164:167], a[120:123], v89, v85 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[30:33], v[168:171], a[124:127], v89, v85 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(20)
s_barrier

	;;#ASMEND
	s_add_u32 s76, s73, s79
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[90:93], v[34:37], a[128:131],  v82, v86 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[2:5], v95 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[90:93], v[38:41], a[132:135],  v82, v86 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[6:9], v95 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[90:93], v[42:45], a[136:139],  v82, v87 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[10:13], v95 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[90:93], v[46:49], a[140:143],  v82, v87 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[14:17], v95 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[124:127], v[50:53], a[128:131],  v82, v86 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[18:21], v96 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[124:127], v[54:57], a[132:135],  v82, v86 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[22:25], v96 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[124:127], v[58:61], a[136:139],  v82, v87 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[26:29], v96 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[124:127], v[62:65], a[140:143],  v82, v87 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v96 offset:6144

	;;#ASMEND
	buffer_load_dwordx4 v66, s[16:19], s78 offen lds
	s_mov_b32 m0, s60
	s_sub_u32 s77, s76, s30
	buffer_load_dwordx4 v67, s[16:19], s78 offen lds
	s_mov_b32 m0, s61
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[112:115], v[34:37], a[144:147],  v82, v86 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[112:115], v[38:41], a[148:151],  v82, v86 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[112:115], v[42:45], a[152:155],  v82, v87 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[112:115], v[46:49], a[156:159],  v82, v87 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[128:131], v[50:53], a[144:147],  v82, v86 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[128:131], v[54:57], a[148:151],  v82, v86 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[128:131], v[58:61], a[152:155],  v82, v87 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[128:131], v[62:65], a[156:159],  v82, v87 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_u32 s76, s74, s79
	buffer_load_dwordx4 v94, s[16:19], s78 offen lds
	s_mov_b32 m0, s62
	s_sub_u32 s76, s76, s34
	buffer_load_dwordx4 v68, s[16:19], s78 offen lds
	s_mov_b32 m0, s41
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[116:119], v[34:37], a[160:163],  v83, v86 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[116:119], v[38:41], a[164:167],  v83, v86 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[116:119], v[42:45], a[168:171], v83, v87 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[116:119], v[46:49], a[172:175], v83, v87 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[132:135], v[50:53], a[160:163],  v83, v86 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[132:135], v[54:57], a[164:167],  v83, v86 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[132:135], v[58:61], a[168:171], v83, v87 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[132:135], v[62:65], a[172:175], v83, v87 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_u32 s79, s75, s79
	buffer_load_dwordx4 v66, s[16:19], s77 offen lds
	s_mov_b32 m0, s63
	s_sub_u32 s79, s79, s34
	buffer_load_dwordx4 v67, s[16:19], s77 offen lds
	s_mov_b32 m0, s64
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[120:123], v[34:37], a[176:179], v83, v86 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[120:123], v[38:41], a[180:183], v83, v86 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[120:123], v[42:45], a[184:187], v83, v87 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[120:123], v[46:49], a[188:191], v83, v87 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[136:139], v[50:53], a[176:179], v83, v86 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[136:139], v[54:57], a[180:183], v83, v86 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[136:139], v[58:61], a[184:187], v83, v87 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[136:139], v[62:65], a[188:191], v83, v87 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_cmpk_lg_i32 s36, 0x3400
	buffer_load_dwordx4 v94, s[16:19], s77 offen lds
	s_mov_b32 m0, s65
	s_cselect_b32 s80, s35, 0xde00
	buffer_load_dwordx4 v68, s[16:19], s77 offen lds
	s_mov_b32 m0, s43
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[90:93], v[140:143], a[192:195],  v82, v84 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v99 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[90:93], v[144:147], a[196:199],  v82, v84 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v99 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[90:93], v[148:151], a[200:203],  v82, v85 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v99 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[90:93], v[152:155], a[204:207],  v82, v85 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v99 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[124:127], v[156:159], a[192:195],  v82, v84 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v100 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[124:127], v[160:163], a[196:199],  v82, v84 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v100 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[124:127], v[164:167], a[200:203],  v82, v85 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v100 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[124:127], v[168:171], a[204:207],  v82, v85 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[62:65], v100 offset:6144

	;;#ASMEND
	s_add_u32 s36, s36, 0x400
	buffer_load_dwordx4 v70, s[20:23], s76 offen lds
	s_mov_b32 m0, s66
	s_addc_u32 s37, s37, 0
	buffer_load_dwordx4 v69, s[20:23], s76 offen lds
	s_mov_b32 m0, s67
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[112:115], v[140:143], a[208:211],  v82, v84 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[112:115], v[144:147], a[212:215],  v82, v84 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[112:115], v[148:151], a[216:219],  v82, v85 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[112:115], v[152:155], a[220:223],  v82, v85 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[128:131], v[156:159], a[208:211],  v82, v84 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[128:131], v[160:163], a[212:215],  v82, v84 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[128:131], v[164:167], a[216:219],  v82, v85 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[128:131], v[168:171], a[220:223],  v82, v85 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_addk_i32 s35, 0x1000
	buffer_load_dwordx4 v71, s[20:23], s76 offen lds
	s_mov_b32 m0, s68
	s_add_i32 s29, s29, 8
	buffer_load_dwordx4 v72, s[20:23], s76 offen lds
	s_mov_b32 m0, s45
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
	s_cmpk_eq_i32 s36, 0x3800
	buffer_load_dwordx4 v70, s[20:23], s79 offen lds
	s_mov_b32 m0, s69
	;;#ASMSTART
	buffer_load_dwordx2 v[80:81], v73, s[0:3], s80 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[74:75], v73, s[4:7], s80 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[78:79], v73, s[8:11], s80 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[76:77], v73, s[12:15], s80 offen
	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v69, s[20:23], s79 offen lds
	s_mov_b32 m0, s70
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[120:123], v[140:143], a[240:243], v83, v84 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[120:123], v[144:147], a[244:247], v83, v84 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[120:123], v[148:151], a[248:251], v83, v85 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[120:123], v[152:155], a[252:255], v83, v85 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[136:139], v[156:159], a[240:243], v83, v84 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[136:139], v[160:163], a[244:247], v83, v84 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[136:139], v[164:167], a[248:251], v83, v85 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[136:139], v[168:171], a[252:255], v83, v85 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v71, s[20:23], s79 offen lds
	s_mov_b32 m0, s71
	s_nop 0
	buffer_load_dwordx4 v72, s[20:23], s79 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_cbranch_scc0 .LBB0_7
; %bb.8:
	v_lshl_or_b32 v252, s25, 8, v1
	v_lshrrev_b32_e32 v1, 2, v0
	v_accvgpr_read_b32 v187, a31
	v_mad_i64_i32 v[212:213], s[0:1], s28, v252, 0
	v_ashrrev_i32_e32 v183, 31, v182
	v_and_b32_e32 v239, 12, v1
	v_and_b32_e32 v238, 15, v0
	v_accvgpr_read_b32 v186, a30
	v_lshl_add_u64 v[214:215], v[212:213], 1, s[26:27]
	v_lshlrev_b64 v[212:213], 1, v[182:183]
	v_mad_u64_u32 v[182:183], s[0:1], v239, s28, v[238:239]
	v_pk_mul_f32 v[240:241], v[186:187], s[24:25] op_sel_hi:[1,0]
	v_add_u32_e32 v186, s28, v182
	v_ashrrev_i32_e32 v183, 31, v182
	v_ashrrev_i32_e32 v187, 31, v186
	v_accvgpr_read_b32 v185, a29
	v_accvgpr_read_b32 v184, a28
	v_lshl_add_u64 v[214:215], v[214:215], 0, v[212:213]
	v_lshlrev_b64 v[0:1], 1, v[182:183]
	v_lshlrev_b64 v[182:183], 1, v[186:187]
	v_add_u32_e32 v186, s28, v186
	v_pk_mul_f32 v[184:185], v[184:185], s[24:25] op_sel_hi:[1,0]
	v_lshl_add_u64 v[216:217], v[214:215], 0, v[0:1]
	v_lshl_add_u64 v[242:243], v[214:215], 0, v[182:183]
	v_ashrrev_i32_e32 v187, 31, v186
	global_store_short_d16_hi v[216:217], v184, off
	global_store_short_d16_hi v[242:243], v185, off
	v_lshlrev_b64 v[184:185], 1, v[186:187]
	v_add_u32_e32 v186, s28, v186
	v_accvgpr_read_b32 v195, a43
	v_accvgpr_read_b32 v233, a39
	v_accvgpr_read_b32 v237, a35
	v_ashrrev_i32_e32 v187, 31, v186
	v_accvgpr_read_b32 v194, a42
	v_accvgpr_read_b32 v193, a41
	v_accvgpr_read_b32 v192, a40
	v_accvgpr_read_b32 v231, a37
	v_accvgpr_read_b32 v230, a36
	v_accvgpr_read_b32 v235, a33
	v_accvgpr_read_b32 v234, a32
	v_lshlrev_b64 v[186:187], 1, v[186:187]
	v_accvgpr_read_b32 v191, a47
	v_accvgpr_read_b32 v232, a38
	v_accvgpr_read_b32 v236, a34
	v_lshl_add_u64 v[244:245], v[214:215], 0, v[184:185]
	v_lshl_add_u64 v[246:247], v[214:215], 0, v[186:187]
	v_pk_mul_f32 v[234:235], v[234:235], s[24:25] op_sel_hi:[1,0]
	v_pk_mul_f32 v[230:231], v[230:231], s[24:25] op_sel_hi:[1,0]
	v_pk_mul_f32 v[194:195], v[194:195], s[24:25] op_sel_hi:[1,0]
	v_pk_mul_f32 v[192:193], v[192:193], s[24:25] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v190, a46
	global_store_short_d16_hi v[244:245], v240, off
	global_store_short_d16_hi v[246:247], v241, off
	v_pk_mul_f32 v[236:237], v[236:237], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[216:217], v234, off offset:32
	global_store_short_d16_hi v[242:243], v235, off offset:32
	global_store_short_d16_hi v[244:245], v236, off offset:32
	global_store_short_d16_hi v[246:247], v237, off offset:32
	v_pk_mul_f32 v[232:233], v[232:233], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[216:217], v230, off offset:64
	global_store_short_d16_hi v[242:243], v231, off offset:64
	global_store_short_d16_hi v[244:245], v232, off offset:64
	global_store_short_d16_hi v[246:247], v233, off offset:64
	global_store_short_d16_hi v[216:217], v192, off offset:96
	global_store_short_d16_hi v[242:243], v193, off offset:96
	global_store_short_d16_hi v[244:245], v194, off offset:96
	global_store_short_d16_hi v[246:247], v195, off offset:96
	v_or_b32_e32 v194, 16, v239
	v_pk_mul_f32 v[230:231], v[190:191], s[24:25] op_sel_hi:[1,0]
	v_mad_u64_u32 v[190:191], s[0:1], v194, s28, v[238:239]
	v_add_u32_e32 v194, s28, v190
	v_accvgpr_read_b32 v189, a45
	v_accvgpr_read_b32 v188, a44
	v_ashrrev_i32_e32 v191, 31, v190
	v_ashrrev_i32_e32 v195, 31, v194
	v_pk_mul_f32 v[192:193], v[188:189], s[24:25] op_sel_hi:[1,0]
	v_lshlrev_b64 v[188:189], 1, v[190:191]
	v_lshlrev_b64 v[190:191], 1, v[194:195]
	v_add_u32_e32 v194, s28, v194
	v_lshl_add_u64 v[232:233], v[214:215], 0, v[188:189]
	v_lshl_add_u64 v[234:235], v[214:215], 0, v[190:191]
	v_ashrrev_i32_e32 v195, 31, v194
	global_store_short_d16_hi v[232:233], v192, off
	global_store_short_d16_hi v[234:235], v193, off
	v_lshlrev_b64 v[192:193], 1, v[194:195]
	v_add_u32_e32 v194, s28, v194
	v_accvgpr_read_b32 v203, a59
	v_accvgpr_read_b32 v225, a55
	v_accvgpr_read_b32 v229, a51
	v_ashrrev_i32_e32 v195, 31, v194
	v_accvgpr_read_b32 v202, a58
	v_accvgpr_read_b32 v201, a57
	v_accvgpr_read_b32 v200, a56
	v_accvgpr_read_b32 v223, a53
	v_accvgpr_read_b32 v222, a52
	v_accvgpr_read_b32 v227, a49
	v_accvgpr_read_b32 v226, a48
	v_lshlrev_b64 v[194:195], 1, v[194:195]
	v_accvgpr_read_b32 v199, a63
	v_accvgpr_read_b32 v224, a54
	v_accvgpr_read_b32 v228, a50
	v_lshl_add_u64 v[236:237], v[214:215], 0, v[192:193]
	v_lshl_add_u64 v[240:241], v[214:215], 0, v[194:195]
	v_pk_mul_f32 v[226:227], v[226:227], s[24:25] op_sel_hi:[1,0]
	v_pk_mul_f32 v[222:223], v[222:223], s[24:25] op_sel_hi:[1,0]
	v_pk_mul_f32 v[202:203], v[202:203], s[24:25] op_sel_hi:[1,0]
	v_pk_mul_f32 v[200:201], v[200:201], s[24:25] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v198, a62
	global_store_short_d16_hi v[236:237], v230, off
	global_store_short_d16_hi v[240:241], v231, off
	v_pk_mul_f32 v[228:229], v[228:229], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[232:233], v226, off offset:32
	global_store_short_d16_hi v[234:235], v227, off offset:32
	global_store_short_d16_hi v[236:237], v228, off offset:32
	global_store_short_d16_hi v[240:241], v229, off offset:32
	v_pk_mul_f32 v[224:225], v[224:225], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[232:233], v222, off offset:64
	global_store_short_d16_hi v[234:235], v223, off offset:64
	global_store_short_d16_hi v[236:237], v224, off offset:64
	global_store_short_d16_hi v[240:241], v225, off offset:64
	global_store_short_d16_hi v[232:233], v200, off offset:96
	global_store_short_d16_hi v[234:235], v201, off offset:96
	global_store_short_d16_hi v[236:237], v202, off offset:96
	global_store_short_d16_hi v[240:241], v203, off offset:96
	v_or_b32_e32 v202, 32, v239
	v_pk_mul_f32 v[222:223], v[198:199], s[24:25] op_sel_hi:[1,0]
	v_mad_u64_u32 v[198:199], s[0:1], v202, s28, v[238:239]
	v_add_u32_e32 v202, s28, v198
	v_accvgpr_read_b32 v197, a61
	v_accvgpr_read_b32 v196, a60
	v_ashrrev_i32_e32 v199, 31, v198
	v_ashrrev_i32_e32 v203, 31, v202
	v_pk_mul_f32 v[200:201], v[196:197], s[24:25] op_sel_hi:[1,0]
	v_lshlrev_b64 v[196:197], 1, v[198:199]
	v_lshlrev_b64 v[198:199], 1, v[202:203]
	v_add_u32_e32 v202, s28, v202
	v_lshl_add_u64 v[224:225], v[214:215], 0, v[196:197]
	v_lshl_add_u64 v[226:227], v[214:215], 0, v[198:199]
	v_ashrrev_i32_e32 v203, 31, v202
	global_store_short_d16_hi v[224:225], v200, off
	global_store_short_d16_hi v[226:227], v201, off
	v_lshlrev_b64 v[200:201], 1, v[202:203]
	v_add_u32_e32 v202, s28, v202
	v_ashrrev_i32_e32 v203, 31, v202
	v_accvgpr_read_b32 v221, a67
	v_lshlrev_b64 v[202:203], 1, v[202:203]
	v_accvgpr_read_b32 v220, a66
	v_accvgpr_read_b32 v219, a65
	v_accvgpr_read_b32 v218, a64
	v_lshl_add_u64 v[228:229], v[214:215], 0, v[200:201]
	v_lshl_add_u64 v[230:231], v[214:215], 0, v[202:203]
	v_accvgpr_read_b32 v211, a71
	global_store_short_d16_hi v[228:229], v222, off
	global_store_short_d16_hi v[230:231], v223, off
	v_pk_mul_f32 v[222:223], v[220:221], s[24:25] op_sel_hi:[1,0]
	v_pk_mul_f32 v[248:249], v[218:219], s[24:25] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v221, a27
	v_accvgpr_read_b32 v210, a70
	v_accvgpr_read_b32 v209, a69
	v_accvgpr_read_b32 v208, a68
	v_accvgpr_read_b32 v219, a25
	v_accvgpr_read_b32 v218, a24
	v_accvgpr_read_b32 v207, a75
	v_accvgpr_read_b32 v220, a26
	v_pk_mul_f32 v[218:219], v[218:219], s[24:25] op_sel_hi:[1,0]
	v_pk_mul_f32 v[210:211], v[210:211], s[24:25] op_sel_hi:[1,0]
	v_pk_mul_f32 v[208:209], v[208:209], s[24:25] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v206, a74
	global_store_short_d16_hi v[224:225], v248, off offset:32
	global_store_short_d16_hi v[226:227], v249, off offset:32
	global_store_short_d16_hi v[228:229], v222, off offset:32
	global_store_short_d16_hi v[230:231], v223, off offset:32
	v_pk_mul_f32 v[220:221], v[220:221], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[224:225], v218, off offset:64
	global_store_short_d16_hi v[226:227], v219, off offset:64
	global_store_short_d16_hi v[228:229], v220, off offset:64
	global_store_short_d16_hi v[230:231], v221, off offset:64
	global_store_short_d16_hi v[224:225], v208, off offset:96
	global_store_short_d16_hi v[226:227], v209, off offset:96
	global_store_short_d16_hi v[228:229], v210, off offset:96
	global_store_short_d16_hi v[230:231], v211, off offset:96
	v_or_b32_e32 v210, 48, v239
	v_pk_mul_f32 v[218:219], v[206:207], s[24:25] op_sel_hi:[1,0]
	v_mad_u64_u32 v[206:207], s[0:1], v210, s28, v[238:239]
	v_add_u32_e32 v210, s28, v206
	v_accvgpr_read_b32 v205, a73
	v_accvgpr_read_b32 v204, a72
	v_ashrrev_i32_e32 v207, 31, v206
	v_ashrrev_i32_e32 v211, 31, v210
	v_pk_mul_f32 v[208:209], v[204:205], s[24:25] op_sel_hi:[1,0]
	v_lshlrev_b64 v[204:205], 1, v[206:207]
	v_lshlrev_b64 v[206:207], 1, v[210:211]
	v_add_u32_e32 v210, s28, v210
	v_lshl_add_u64 v[220:221], v[214:215], 0, v[204:205]
	v_lshl_add_u64 v[222:223], v[214:215], 0, v[206:207]
	v_ashrrev_i32_e32 v211, 31, v210
	global_store_short_d16_hi v[220:221], v208, off
	global_store_short_d16_hi v[222:223], v209, off
	v_lshlrev_b64 v[208:209], 1, v[210:211]
	v_add_u32_e32 v210, s28, v210
	v_accvgpr_read_b32 v181, a79
	v_ashrrev_i32_e32 v211, 31, v210
	v_accvgpr_read_b32 v179, a77
	v_accvgpr_read_b32 v178, a76
	v_lshlrev_b64 v[210:211], 1, v[210:211]
	v_accvgpr_read_b32 v180, a78
	v_lshl_add_u64 v[238:239], v[214:215], 0, v[208:209]
	v_lshl_add_u64 v[248:249], v[214:215], 0, v[210:211]
	v_pk_mul_f32 v[178:179], v[178:179], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[238:239], v218, off
	global_store_short_d16_hi v[248:249], v219, off
	v_pk_mul_f32 v[218:219], v[180:181], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[220:221], v178, off offset:32
	global_store_short_d16_hi v[222:223], v179, off offset:32
	v_accvgpr_read_b32 v181, a23
	v_accvgpr_read_b32 v179, a21
	v_accvgpr_read_b32 v178, a20
	v_accvgpr_read_b32 v180, a22
	v_pk_mul_f32 v[178:179], v[178:179], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[238:239], v218, off offset:32
	global_store_short_d16_hi v[248:249], v219, off offset:32
	v_pk_mul_f32 v[218:219], v[180:181], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[220:221], v178, off offset:64
	global_store_short_d16_hi v[222:223], v179, off offset:64
	v_accvgpr_read_b32 v181, a19
	v_accvgpr_read_b32 v179, a17
	v_accvgpr_read_b32 v178, a16
	v_accvgpr_read_b32 v180, a18
	v_pk_mul_f32 v[178:179], v[178:179], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[238:239], v218, off offset:64
	global_store_short_d16_hi v[248:249], v219, off offset:64
	v_pk_mul_f32 v[218:219], v[180:181], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[220:221], v178, off offset:96
	global_store_short_d16_hi v[222:223], v179, off offset:96
	v_accvgpr_read_b32 v181, a15
	v_accvgpr_read_b32 v179, a13
	v_accvgpr_read_b32 v178, a12
	global_store_short_d16_hi v[238:239], v218, off offset:96
	global_store_short_d16_hi v[248:249], v219, off offset:96
	v_pk_mul_f32 v[218:219], v[178:179], s[24:25] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v180, a14
	global_store_short_d16_hi v[216:217], v218, off offset:256
	global_store_short_d16_hi v[242:243], v219, off offset:256
	v_accvgpr_read_b32 v219, a7
	s_mov_b64 s[0:1], 0x100
	v_pk_mul_f32 v[250:251], v[180:181], s[24:25] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v181, a11
	v_accvgpr_read_b32 v217, a5
	v_accvgpr_read_b32 v216, a4
	v_lshl_add_u64 v[214:215], v[214:215], 0, s[0:1]
	v_accvgpr_read_b32 v179, a9
	v_accvgpr_read_b32 v178, a8
	v_accvgpr_read_b32 v218, a6
	global_store_short_d16_hi v[244:245], v250, off offset:256
	global_store_short_d16_hi v[246:247], v251, off offset:256
	v_pk_mul_f32 v[216:217], v[216:217], s[24:25] op_sel_hi:[1,0]
	v_lshl_add_u64 v[242:243], v[214:215], 0, v[0:1]
	v_lshl_add_u64 v[244:245], v[214:215], 0, v[182:183]
	v_accvgpr_read_b32 v180, a10
	v_pk_mul_f32 v[218:219], v[218:219], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[242:243], v216, off offset:32
	global_store_short_d16_hi v[244:245], v217, off offset:32
	v_lshl_add_u64 v[216:217], v[214:215], 0, v[184:185]
	v_lshl_add_u64 v[246:247], v[214:215], 0, v[186:187]
	v_pk_mul_f32 v[178:179], v[178:179], s[24:25] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v177, a83
	global_store_short_d16_hi v[216:217], v218, off offset:32
	global_store_short_d16_hi v[246:247], v219, off offset:32
	v_pk_mul_f32 v[218:219], v[180:181], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[242:243], v178, off offset:64
	global_store_short_d16_hi v[244:245], v179, off offset:64
	v_accvgpr_read_b32 v181, a3
	v_accvgpr_read_b32 v173, a87
	v_accvgpr_read_b32 v176, a82
	v_accvgpr_read_b32 v175, a81
	v_accvgpr_read_b32 v174, a80
	v_accvgpr_read_b32 v179, a1
	v_accvgpr_read_b32 v178, a0
	v_accvgpr_read_b32 v161, a99
	v_accvgpr_read_b32 v165, a95
	v_accvgpr_read_b32 v169, a91
	v_accvgpr_read_b32 v171, a85
	v_accvgpr_read_b32 v170, a84
	v_accvgpr_read_b32 v180, a2
	v_pk_mul_f32 v[178:179], v[178:179], s[24:25] op_sel_hi:[1,0]
	v_pk_mul_f32 v[176:177], v[176:177], s[24:25] op_sel_hi:[1,0]
	v_pk_mul_f32 v[174:175], v[174:175], s[24:25] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v157, a103
	v_accvgpr_read_b32 v160, a98
	v_accvgpr_read_b32 v159, a97
	v_accvgpr_read_b32 v158, a96
	v_accvgpr_read_b32 v163, a93
	v_accvgpr_read_b32 v162, a92
	v_accvgpr_read_b32 v167, a89
	v_accvgpr_read_b32 v166, a88
	v_accvgpr_read_b32 v172, a86
	global_store_short_d16_hi v[216:217], v218, off offset:64
	global_store_short_d16_hi v[246:247], v219, off offset:64
	v_pk_mul_f32 v[180:181], v[180:181], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[242:243], v178, off offset:96
	global_store_short_d16_hi v[244:245], v179, off offset:96
	global_store_short_d16_hi v[216:217], v180, off offset:96
	global_store_short_d16_hi v[246:247], v181, off offset:96
	global_store_short_d16_hi v[232:233], v174, off offset:256
	global_store_short_d16_hi v[234:235], v175, off offset:256
	global_store_short_d16_hi v[236:237], v176, off offset:256
	global_store_short_d16_hi v[240:241], v177, off offset:256
	v_pk_mul_f32 v[170:171], v[170:171], s[24:25] op_sel_hi:[1,0]
	v_lshl_add_u64 v[174:175], v[214:215], 0, v[188:189]
	v_lshl_add_u64 v[176:177], v[214:215], 0, v[190:191]
	v_accvgpr_read_b32 v145, a115
	v_accvgpr_read_b32 v149, a111
	v_accvgpr_read_b32 v153, a107
	v_accvgpr_read_b32 v155, a101
	v_accvgpr_read_b32 v154, a100
	v_accvgpr_read_b32 v164, a94
	v_accvgpr_read_b32 v168, a90
	v_pk_mul_f32 v[172:173], v[172:173], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[174:175], v170, off offset:32
	global_store_short_d16_hi v[176:177], v171, off offset:32
	v_lshl_add_u64 v[170:171], v[214:215], 0, v[192:193]
	v_lshl_add_u64 v[178:179], v[214:215], 0, v[194:195]
	v_pk_mul_f32 v[166:167], v[166:167], s[24:25] op_sel_hi:[1,0]
	v_pk_mul_f32 v[162:163], v[162:163], s[24:25] op_sel_hi:[1,0]
	v_pk_mul_f32 v[160:161], v[160:161], s[24:25] op_sel_hi:[1,0]
	v_pk_mul_f32 v[158:159], v[158:159], s[24:25] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v141, a119
	v_accvgpr_read_b32 v144, a114
	v_accvgpr_read_b32 v143, a113
	v_accvgpr_read_b32 v142, a112
	v_accvgpr_read_b32 v147, a109
	v_accvgpr_read_b32 v146, a108
	v_accvgpr_read_b32 v151, a105
	v_accvgpr_read_b32 v150, a104
	v_accvgpr_read_b32 v156, a102
	global_store_short_d16_hi v[170:171], v172, off offset:32
	global_store_short_d16_hi v[178:179], v173, off offset:32
	v_pk_mul_f32 v[168:169], v[168:169], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[174:175], v166, off offset:64
	global_store_short_d16_hi v[176:177], v167, off offset:64
	global_store_short_d16_hi v[170:171], v168, off offset:64
	global_store_short_d16_hi v[178:179], v169, off offset:64
	v_pk_mul_f32 v[164:165], v[164:165], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[174:175], v162, off offset:96
	global_store_short_d16_hi v[176:177], v163, off offset:96
	global_store_short_d16_hi v[170:171], v164, off offset:96
	global_store_short_d16_hi v[178:179], v165, off offset:96
	global_store_short_d16_hi v[224:225], v158, off offset:256
	global_store_short_d16_hi v[226:227], v159, off offset:256
	global_store_short_d16_hi v[228:229], v160, off offset:256
	global_store_short_d16_hi v[230:231], v161, off offset:256
	v_pk_mul_f32 v[154:155], v[154:155], s[24:25] op_sel_hi:[1,0]
	v_lshl_add_u64 v[158:159], v[214:215], 0, v[196:197]
	v_lshl_add_u64 v[160:161], v[214:215], 0, v[198:199]
	v_accvgpr_read_b32 v133, a127
	v_accvgpr_read_b32 v137, a123
	v_accvgpr_read_b32 v139, a117
	v_accvgpr_read_b32 v138, a116
	v_accvgpr_read_b32 v148, a110
	v_accvgpr_read_b32 v152, a106
	v_pk_mul_f32 v[156:157], v[156:157], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[158:159], v154, off offset:32
	global_store_short_d16_hi v[160:161], v155, off offset:32
	v_lshl_add_u64 v[154:155], v[214:215], 0, v[200:201]
	v_lshl_add_u64 v[162:163], v[214:215], 0, v[202:203]
	v_pk_mul_f32 v[150:151], v[150:151], s[24:25] op_sel_hi:[1,0]
	v_pk_mul_f32 v[146:147], v[146:147], s[24:25] op_sel_hi:[1,0]
	v_pk_mul_f32 v[144:145], v[144:145], s[24:25] op_sel_hi:[1,0]
	v_pk_mul_f32 v[142:143], v[142:143], s[24:25] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v131, a125
	v_accvgpr_read_b32 v130, a124
	v_accvgpr_read_b32 v135, a121
	v_accvgpr_read_b32 v134, a120
	v_accvgpr_read_b32 v140, a118
	global_store_short_d16_hi v[154:155], v156, off offset:32
	global_store_short_d16_hi v[162:163], v157, off offset:32
	v_pk_mul_f32 v[152:153], v[152:153], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[158:159], v150, off offset:64
	global_store_short_d16_hi v[160:161], v151, off offset:64
	global_store_short_d16_hi v[154:155], v152, off offset:64
	global_store_short_d16_hi v[162:163], v153, off offset:64
	v_pk_mul_f32 v[148:149], v[148:149], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[158:159], v146, off offset:96
	global_store_short_d16_hi v[160:161], v147, off offset:96
	global_store_short_d16_hi v[154:155], v148, off offset:96
	global_store_short_d16_hi v[162:163], v149, off offset:96
	global_store_short_d16_hi v[220:221], v142, off offset:256
	global_store_short_d16_hi v[222:223], v143, off offset:256
	global_store_short_d16_hi v[238:239], v144, off offset:256
	global_store_short_d16_hi v[248:249], v145, off offset:256
	v_pk_mul_f32 v[138:139], v[138:139], s[24:25] op_sel_hi:[1,0]
	v_lshl_add_u64 v[142:143], v[214:215], 0, v[204:205]
	v_lshl_add_u64 v[144:145], v[214:215], 0, v[206:207]
	v_accvgpr_read_b32 v132, a126
	v_accvgpr_read_b32 v136, a122
	v_pk_mul_f32 v[140:141], v[140:141], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[142:143], v138, off offset:32
	global_store_short_d16_hi v[144:145], v139, off offset:32
	v_lshl_add_u64 v[138:139], v[214:215], 0, v[208:209]
	v_lshl_add_u64 v[146:147], v[214:215], 0, v[210:211]
	v_pk_mul_f32 v[134:135], v[134:135], s[24:25] op_sel_hi:[1,0]
	v_pk_mul_f32 v[130:131], v[130:131], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[138:139], v140, off offset:32
	global_store_short_d16_hi v[146:147], v141, off offset:32
	v_pk_mul_f32 v[136:137], v[136:137], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[142:143], v134, off offset:64
	global_store_short_d16_hi v[144:145], v135, off offset:64
	global_store_short_d16_hi v[138:139], v136, off offset:64
	global_store_short_d16_hi v[146:147], v137, off offset:64
	v_pk_mul_f32 v[132:133], v[132:133], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[142:143], v130, off offset:96
	global_store_short_d16_hi v[144:145], v131, off offset:96
	global_store_short_d16_hi v[138:139], v132, off offset:96
	global_store_short_d16_hi v[146:147], v133, off offset:96
	v_or_b32_e32 v130, 0x80, v252
	v_mad_i64_i32 v[130:131], s[2:3], s28, v130, 0
	v_accvgpr_read_b32 v126, a128
	v_lshl_add_u64 v[130:131], v[130:131], 1, s[26:27]
	v_accvgpr_read_b32 v114, a140
	v_accvgpr_read_b32 v118, a136
	v_accvgpr_read_b32 v122, a132
	v_accvgpr_read_b32 v127, a129
	v_lshl_add_u64 v[130:131], v[130:131], 0, v[212:213]
	v_accvgpr_read_b32 v110, a144
	v_accvgpr_read_b32 v115, a141
	v_accvgpr_read_b32 v116, a142
	v_accvgpr_read_b32 v117, a143
	v_accvgpr_read_b32 v119, a137
	v_accvgpr_read_b32 v123, a133
	v_accvgpr_read_b32 v128, a130
	v_accvgpr_read_b32 v129, a131
	v_pk_mul_f32 v[126:127], v[126:127], s[24:25] op_sel_hi:[1,0]
	v_lshl_add_u64 v[132:133], v[130:131], 0, v[0:1]
	v_lshl_add_u64 v[134:135], v[130:131], 0, v[182:183]
	v_accvgpr_read_b32 v98, a156
	v_accvgpr_read_b32 v102, a152
	v_accvgpr_read_b32 v106, a148
	v_accvgpr_read_b32 v111, a145
	v_accvgpr_read_b32 v120, a138
	v_accvgpr_read_b32 v121, a139
	v_accvgpr_read_b32 v124, a134
	v_accvgpr_read_b32 v125, a135
	v_pk_mul_f32 v[128:129], v[128:129], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[132:133], v126, off
	global_store_short_d16_hi v[134:135], v127, off
	v_lshl_add_u64 v[126:127], v[130:131], 0, v[184:185]
	v_lshl_add_u64 v[136:137], v[130:131], 0, v[186:187]
	v_pk_mul_f32 v[122:123], v[122:123], s[24:25] op_sel_hi:[1,0]
	v_pk_mul_f32 v[118:119], v[118:119], s[24:25] op_sel_hi:[1,0]
	v_pk_mul_f32 v[116:117], v[116:117], s[24:25] op_sel_hi:[1,0]
	v_pk_mul_f32 v[114:115], v[114:115], s[24:25] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v94, a160
	v_accvgpr_read_b32 v99, a157
	v_accvgpr_read_b32 v100, a158
	v_accvgpr_read_b32 v101, a159
	v_accvgpr_read_b32 v103, a153
	v_accvgpr_read_b32 v107, a149
	v_accvgpr_read_b32 v112, a146
	v_accvgpr_read_b32 v113, a147
	global_store_short_d16_hi v[126:127], v128, off
	global_store_short_d16_hi v[136:137], v129, off
	v_pk_mul_f32 v[124:125], v[124:125], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[132:133], v122, off offset:32
	global_store_short_d16_hi v[134:135], v123, off offset:32
	global_store_short_d16_hi v[126:127], v124, off offset:32
	global_store_short_d16_hi v[136:137], v125, off offset:32
	v_pk_mul_f32 v[120:121], v[120:121], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[132:133], v118, off offset:64
	global_store_short_d16_hi v[134:135], v119, off offset:64
	global_store_short_d16_hi v[126:127], v120, off offset:64
	global_store_short_d16_hi v[136:137], v121, off offset:64
	global_store_short_d16_hi v[132:133], v114, off offset:96
	global_store_short_d16_hi v[134:135], v115, off offset:96
	global_store_short_d16_hi v[126:127], v116, off offset:96
	global_store_short_d16_hi v[136:137], v117, off offset:96
	v_pk_mul_f32 v[110:111], v[110:111], s[24:25] op_sel_hi:[1,0]
	v_lshl_add_u64 v[114:115], v[130:131], 0, v[188:189]
	v_lshl_add_u64 v[116:117], v[130:131], 0, v[190:191]
	v_accvgpr_read_b32 v82, a172
	v_accvgpr_read_b32 v86, a168
	v_accvgpr_read_b32 v90, a164
	v_accvgpr_read_b32 v95, a161
	v_accvgpr_read_b32 v104, a154
	v_accvgpr_read_b32 v105, a155
	v_accvgpr_read_b32 v108, a150
	v_accvgpr_read_b32 v109, a151
	v_pk_mul_f32 v[112:113], v[112:113], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[114:115], v110, off
	global_store_short_d16_hi v[116:117], v111, off
	v_lshl_add_u64 v[110:111], v[130:131], 0, v[192:193]
	v_lshl_add_u64 v[118:119], v[130:131], 0, v[194:195]
	v_pk_mul_f32 v[106:107], v[106:107], s[24:25] op_sel_hi:[1,0]
	v_pk_mul_f32 v[102:103], v[102:103], s[24:25] op_sel_hi:[1,0]
	v_pk_mul_f32 v[100:101], v[100:101], s[24:25] op_sel_hi:[1,0]
	v_pk_mul_f32 v[98:99], v[98:99], s[24:25] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v78, a176
	v_accvgpr_read_b32 v83, a173
	v_accvgpr_read_b32 v84, a174
	v_accvgpr_read_b32 v85, a175
	v_accvgpr_read_b32 v87, a169
	v_accvgpr_read_b32 v91, a165
	v_accvgpr_read_b32 v96, a162
	v_accvgpr_read_b32 v97, a163
	global_store_short_d16_hi v[110:111], v112, off
	global_store_short_d16_hi v[118:119], v113, off
	v_pk_mul_f32 v[108:109], v[108:109], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[114:115], v106, off offset:32
	global_store_short_d16_hi v[116:117], v107, off offset:32
	global_store_short_d16_hi v[110:111], v108, off offset:32
	global_store_short_d16_hi v[118:119], v109, off offset:32
	v_pk_mul_f32 v[104:105], v[104:105], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[114:115], v102, off offset:64
	global_store_short_d16_hi v[116:117], v103, off offset:64
	global_store_short_d16_hi v[110:111], v104, off offset:64
	global_store_short_d16_hi v[118:119], v105, off offset:64
	global_store_short_d16_hi v[114:115], v98, off offset:96
	global_store_short_d16_hi v[116:117], v99, off offset:96
	global_store_short_d16_hi v[110:111], v100, off offset:96
	global_store_short_d16_hi v[118:119], v101, off offset:96
	v_pk_mul_f32 v[94:95], v[94:95], s[24:25] op_sel_hi:[1,0]
	v_lshl_add_u64 v[98:99], v[130:131], 0, v[196:197]
	v_lshl_add_u64 v[100:101], v[130:131], 0, v[198:199]
	v_accvgpr_read_b32 v66, a188
	v_accvgpr_read_b32 v70, a184
	v_accvgpr_read_b32 v74, a180
	v_accvgpr_read_b32 v79, a177
	v_accvgpr_read_b32 v88, a170
	v_accvgpr_read_b32 v89, a171
	v_accvgpr_read_b32 v92, a166
	v_accvgpr_read_b32 v93, a167
	v_pk_mul_f32 v[96:97], v[96:97], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[98:99], v94, off
	global_store_short_d16_hi v[100:101], v95, off
	v_lshl_add_u64 v[94:95], v[130:131], 0, v[200:201]
	v_lshl_add_u64 v[102:103], v[130:131], 0, v[202:203]
	v_pk_mul_f32 v[90:91], v[90:91], s[24:25] op_sel_hi:[1,0]
	v_pk_mul_f32 v[86:87], v[86:87], s[24:25] op_sel_hi:[1,0]
	v_pk_mul_f32 v[84:85], v[84:85], s[24:25] op_sel_hi:[1,0]
	v_pk_mul_f32 v[82:83], v[82:83], s[24:25] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v62, a192
	v_accvgpr_read_b32 v67, a189
	v_accvgpr_read_b32 v71, a185
	v_accvgpr_read_b32 v75, a181
	v_accvgpr_read_b32 v80, a178
	v_accvgpr_read_b32 v81, a179
	global_store_short_d16_hi v[94:95], v96, off
	global_store_short_d16_hi v[102:103], v97, off
	v_pk_mul_f32 v[92:93], v[92:93], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[98:99], v90, off offset:32
	global_store_short_d16_hi v[100:101], v91, off offset:32
	global_store_short_d16_hi v[94:95], v92, off offset:32
	global_store_short_d16_hi v[102:103], v93, off offset:32
	v_pk_mul_f32 v[88:89], v[88:89], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[98:99], v86, off offset:64
	global_store_short_d16_hi v[100:101], v87, off offset:64
	global_store_short_d16_hi v[94:95], v88, off offset:64
	global_store_short_d16_hi v[102:103], v89, off offset:64
	global_store_short_d16_hi v[98:99], v82, off offset:96
	global_store_short_d16_hi v[100:101], v83, off offset:96
	global_store_short_d16_hi v[94:95], v84, off offset:96
	global_store_short_d16_hi v[102:103], v85, off offset:96
	v_pk_mul_f32 v[78:79], v[78:79], s[24:25] op_sel_hi:[1,0]
	v_lshl_add_u64 v[82:83], v[130:131], 0, v[204:205]
	v_lshl_add_u64 v[84:85], v[130:131], 0, v[206:207]
	v_accvgpr_read_b32 v58, a196
	v_accvgpr_read_b32 v63, a193
	v_accvgpr_read_b32 v68, a190
	v_accvgpr_read_b32 v69, a191
	v_accvgpr_read_b32 v72, a186
	v_accvgpr_read_b32 v73, a187
	v_accvgpr_read_b32 v76, a182
	v_accvgpr_read_b32 v77, a183
	v_pk_mul_f32 v[80:81], v[80:81], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[82:83], v78, off
	global_store_short_d16_hi v[84:85], v79, off
	v_lshl_add_u64 v[78:79], v[130:131], 0, v[208:209]
	v_lshl_add_u64 v[86:87], v[130:131], 0, v[210:211]
	v_pk_mul_f32 v[74:75], v[74:75], s[24:25] op_sel_hi:[1,0]
	v_pk_mul_f32 v[70:71], v[70:71], s[24:25] op_sel_hi:[1,0]
	v_pk_mul_f32 v[66:67], v[66:67], s[24:25] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v50, a204
	v_accvgpr_read_b32 v54, a200
	v_accvgpr_read_b32 v59, a197
	v_accvgpr_read_b32 v64, a194
	v_accvgpr_read_b32 v65, a195
	global_store_short_d16_hi v[78:79], v80, off
	global_store_short_d16_hi v[86:87], v81, off
	v_pk_mul_f32 v[76:77], v[76:77], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[82:83], v74, off offset:32
	global_store_short_d16_hi v[84:85], v75, off offset:32
	global_store_short_d16_hi v[78:79], v76, off offset:32
	global_store_short_d16_hi v[86:87], v77, off offset:32
	v_pk_mul_f32 v[72:73], v[72:73], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[82:83], v70, off offset:64
	global_store_short_d16_hi v[84:85], v71, off offset:64
	global_store_short_d16_hi v[78:79], v72, off offset:64
	global_store_short_d16_hi v[86:87], v73, off offset:64
	v_pk_mul_f32 v[68:69], v[68:69], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[82:83], v66, off offset:96
	global_store_short_d16_hi v[84:85], v67, off offset:96
	global_store_short_d16_hi v[78:79], v68, off offset:96
	global_store_short_d16_hi v[86:87], v69, off offset:96
	v_lshl_add_u64 v[66:67], v[130:131], 0, s[0:1]
	v_pk_mul_f32 v[62:63], v[62:63], s[24:25] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v46, a208
	v_accvgpr_read_b32 v51, a205
	v_accvgpr_read_b32 v55, a201
	v_accvgpr_read_b32 v60, a198
	v_accvgpr_read_b32 v61, a199
	v_pk_mul_f32 v[64:65], v[64:65], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[132:133], v62, off offset:256
	global_store_short_d16_hi v[134:135], v63, off offset:256
	global_store_short_d16_hi v[126:127], v64, off offset:256
	global_store_short_d16_hi v[136:137], v65, off offset:256
	v_pk_mul_f32 v[58:59], v[58:59], s[24:25] op_sel_hi:[1,0]
	v_lshl_add_u64 v[0:1], v[66:67], 0, v[0:1]
	v_lshl_add_u64 v[62:63], v[66:67], 0, v[182:183]
	v_accvgpr_read_b32 v42, a212
	v_accvgpr_read_b32 v47, a209
	v_accvgpr_read_b32 v48, a210
	v_accvgpr_read_b32 v49, a211
	v_accvgpr_read_b32 v52, a206
	v_accvgpr_read_b32 v53, a207
	v_accvgpr_read_b32 v56, a202
	v_accvgpr_read_b32 v57, a203
	v_pk_mul_f32 v[60:61], v[60:61], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[0:1], v58, off offset:32
	global_store_short_d16_hi v[62:63], v59, off offset:32
	v_lshl_add_u64 v[58:59], v[66:67], 0, v[184:185]
	v_lshl_add_u64 v[64:65], v[66:67], 0, v[186:187]
	v_pk_mul_f32 v[54:55], v[54:55], s[24:25] op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], s[24:25] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v43, a213
	v_accvgpr_read_b32 v44, a214
	v_accvgpr_read_b32 v45, a215
	global_store_short_d16_hi v[58:59], v60, off offset:32
	global_store_short_d16_hi v[64:65], v61, off offset:32
	v_pk_mul_f32 v[56:57], v[56:57], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[0:1], v54, off offset:64
	global_store_short_d16_hi v[62:63], v55, off offset:64
	global_store_short_d16_hi v[58:59], v56, off offset:64
	global_store_short_d16_hi v[64:65], v57, off offset:64
	v_pk_mul_f32 v[52:53], v[52:53], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[0:1], v50, off offset:96
	global_store_short_d16_hi v[62:63], v51, off offset:96
	global_store_short_d16_hi v[58:59], v52, off offset:96
	global_store_short_d16_hi v[64:65], v53, off offset:96
	v_pk_mul_f32 v[0:1], v[48:49], s[24:25] op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], s[24:25] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v38, a216
	global_store_short_d16_hi v[114:115], v46, off offset:256
	global_store_short_d16_hi v[116:117], v47, off offset:256
	global_store_short_d16_hi v[110:111], v0, off offset:256
	global_store_short_d16_hi v[118:119], v1, off offset:256
	v_pk_mul_f32 v[0:1], v[44:45], s[24:25] op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], s[24:25] op_sel_hi:[1,0]
	v_lshl_add_u64 v[44:45], v[66:67], 0, v[188:189]
	v_lshl_add_u64 v[46:47], v[66:67], 0, v[190:191]
	v_accvgpr_read_b32 v34, a220
	v_accvgpr_read_b32 v39, a217
	v_accvgpr_read_b32 v40, a218
	v_accvgpr_read_b32 v41, a219
	global_store_short_d16_hi v[44:45], v42, off offset:32
	global_store_short_d16_hi v[46:47], v43, off offset:32
	v_lshl_add_u64 v[42:43], v[66:67], 0, v[192:193]
	v_lshl_add_u64 v[48:49], v[66:67], 0, v[194:195]
	v_accvgpr_read_b32 v30, a224
	v_accvgpr_read_b32 v35, a221
	v_accvgpr_read_b32 v36, a222
	v_accvgpr_read_b32 v37, a223
	global_store_short_d16_hi v[42:43], v0, off offset:32
	global_store_short_d16_hi v[48:49], v1, off offset:32
	v_pk_mul_f32 v[0:1], v[40:41], s[24:25] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], s[24:25] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v26, a228
	v_accvgpr_read_b32 v31, a225
	v_accvgpr_read_b32 v32, a226
	v_accvgpr_read_b32 v33, a227
	global_store_short_d16_hi v[44:45], v38, off offset:64
	global_store_short_d16_hi v[46:47], v39, off offset:64
	global_store_short_d16_hi v[42:43], v0, off offset:64
	global_store_short_d16_hi v[48:49], v1, off offset:64
	v_pk_mul_f32 v[0:1], v[36:37], s[24:25] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], s[24:25] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v27, a229
	v_accvgpr_read_b32 v28, a230
	v_accvgpr_read_b32 v29, a231
	global_store_short_d16_hi v[44:45], v34, off offset:96
	global_store_short_d16_hi v[46:47], v35, off offset:96
	global_store_short_d16_hi v[42:43], v0, off offset:96
	global_store_short_d16_hi v[48:49], v1, off offset:96
	v_pk_mul_f32 v[0:1], v[32:33], s[24:25] op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], s[24:25] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v22, a232
	global_store_short_d16_hi v[98:99], v30, off offset:256
	global_store_short_d16_hi v[100:101], v31, off offset:256
	global_store_short_d16_hi v[94:95], v0, off offset:256
	global_store_short_d16_hi v[102:103], v1, off offset:256
	v_pk_mul_f32 v[0:1], v[28:29], s[24:25] op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], s[24:25] op_sel_hi:[1,0]
	v_lshl_add_u64 v[28:29], v[66:67], 0, v[196:197]
	v_lshl_add_u64 v[30:31], v[66:67], 0, v[198:199]
	v_accvgpr_read_b32 v18, a236
	v_accvgpr_read_b32 v23, a233
	v_accvgpr_read_b32 v24, a234
	v_accvgpr_read_b32 v25, a235
	global_store_short_d16_hi v[28:29], v26, off offset:32
	global_store_short_d16_hi v[30:31], v27, off offset:32
	v_lshl_add_u64 v[26:27], v[66:67], 0, v[200:201]
	v_lshl_add_u64 v[32:33], v[66:67], 0, v[202:203]
	v_accvgpr_read_b32 v14, a240
	v_accvgpr_read_b32 v19, a237
	v_accvgpr_read_b32 v20, a238
	v_accvgpr_read_b32 v21, a239
	global_store_short_d16_hi v[26:27], v0, off offset:32
	global_store_short_d16_hi v[32:33], v1, off offset:32
	v_pk_mul_f32 v[0:1], v[24:25], s[24:25] op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], s[24:25] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v10, a244
	v_accvgpr_read_b32 v15, a241
	v_accvgpr_read_b32 v16, a242
	v_accvgpr_read_b32 v17, a243
	global_store_short_d16_hi v[28:29], v22, off offset:64
	global_store_short_d16_hi v[30:31], v23, off offset:64
	global_store_short_d16_hi v[26:27], v0, off offset:64
	global_store_short_d16_hi v[32:33], v1, off offset:64
	v_pk_mul_f32 v[0:1], v[20:21], s[24:25] op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], s[24:25] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v11, a245
	v_accvgpr_read_b32 v12, a246
	v_accvgpr_read_b32 v13, a247
	global_store_short_d16_hi v[28:29], v18, off offset:96
	global_store_short_d16_hi v[30:31], v19, off offset:96
	global_store_short_d16_hi v[26:27], v0, off offset:96
	global_store_short_d16_hi v[32:33], v1, off offset:96
	v_pk_mul_f32 v[0:1], v[16:17], s[24:25] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], s[24:25] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v2, a252
	v_accvgpr_read_b32 v6, a248
	global_store_short_d16_hi v[82:83], v14, off offset:256
	global_store_short_d16_hi v[84:85], v15, off offset:256
	global_store_short_d16_hi v[78:79], v0, off offset:256
	global_store_short_d16_hi v[86:87], v1, off offset:256
	v_pk_mul_f32 v[0:1], v[12:13], s[24:25] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], s[24:25] op_sel_hi:[1,0]
	v_lshl_add_u64 v[12:13], v[66:67], 0, v[204:205]
	v_lshl_add_u64 v[14:15], v[66:67], 0, v[206:207]
	v_accvgpr_read_b32 v3, a253
	v_accvgpr_read_b32 v7, a249
	v_accvgpr_read_b32 v8, a250
	v_accvgpr_read_b32 v9, a251
	global_store_short_d16_hi v[12:13], v10, off offset:32
	global_store_short_d16_hi v[14:15], v11, off offset:32
	v_lshl_add_u64 v[10:11], v[66:67], 0, v[208:209]
	v_lshl_add_u64 v[16:17], v[66:67], 0, v[210:211]
	v_accvgpr_read_b32 v4, a254
	v_accvgpr_read_b32 v5, a255
	global_store_short_d16_hi v[10:11], v0, off offset:32
	global_store_short_d16_hi v[16:17], v1, off offset:32
	v_pk_mul_f32 v[0:1], v[8:9], s[24:25] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], s[24:25] op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[12:13], v6, off offset:64
	global_store_short_d16_hi v[14:15], v7, off offset:64
	global_store_short_d16_hi v[10:11], v0, off offset:64
	global_store_short_d16_hi v[16:17], v1, off offset:64
	v_pk_mul_f32 v[0:1], v[4:5], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[12:13], v2, off offset:96
	global_store_short_d16_hi v[14:15], v3, off offset:96
	global_store_short_d16_hi v[10:11], v0, off offset:96
	global_store_short_d16_hi v[16:17], v1, off offset:96
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z22mxfp4_gluon_cpp_kernel13gluon_globals
		.amdhsa_group_segment_fixed_size 131072
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
	.set _Z22mxfp4_gluon_cpp_kernel13gluon_globals.num_vgpr, 253
	.set _Z22mxfp4_gluon_cpp_kernel13gluon_globals.num_agpr, 256
	.set _Z22mxfp4_gluon_cpp_kernel13gluon_globals.numbered_sgpr, 85
	.set _Z22mxfp4_gluon_cpp_kernel13gluon_globals.private_seg_size, 0
	.set _Z22mxfp4_gluon_cpp_kernel13gluon_globals.uses_vcc, 0
	.set _Z22mxfp4_gluon_cpp_kernel13gluon_globals.uses_flat_scratch, 0
	.set _Z22mxfp4_gluon_cpp_kernel13gluon_globals.has_dyn_sized_stack, 0
	.set _Z22mxfp4_gluon_cpp_kernel13gluon_globals.has_recursion, 0
	.set _Z22mxfp4_gluon_cpp_kernel13gluon_globals.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 34368
; TotalNumSgprs: 91
; NumVgprs: 253
; NumAgprs: 256
; TotalNumVgprs: 512
; ScratchSize: 0
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
	.type	__hip_cuid_aabaeefbe12fded9,@object ; @__hip_cuid_aabaeefbe12fded9
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_aabaeefbe12fded9
__hip_cuid_aabaeefbe12fded9:
	.byte	0                               ; 0x0
	.size	__hip_cuid_aabaeefbe12fded9, 1

	.ident	"clang version 20.0.0git (https://github.com/ROCm/llvm-project.git 27682a16360e33e37c4f3cc6adf9a620733f8fe1)"
	.ident	"AMD clang version 20.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.1.0 25425 1b0eada6b0ee93e2e694c8c146d23fca90bc11c5)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_aabaeefbe12fded9
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
    .private_segment_fixed_size: 0
    .sgpr_count:     91
    .sgpr_spill_count: 0
    .symbol:         _Z22mxfp4_gluon_cpp_kernel13gluon_globals.kd
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
