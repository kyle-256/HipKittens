	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.text
	.protected	_Z22mxfp4_gluon_cpp_kernel13gluon_globals ; -- Begin function _Z22mxfp4_gluon_cpp_kernel13gluon_globals
	.globl	_Z22mxfp4_gluon_cpp_kernel13gluon_globals
	.p2align	8
	.type	_Z22mxfp4_gluon_cpp_kernel13gluon_globals,@function
_Z22mxfp4_gluon_cpp_kernel13gluon_globals: ; @_Z22mxfp4_gluon_cpp_kernel13gluon_globals
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
.LBB0_6:                                ; %.preheader1192
	s_load_dwordx2 s[20:21], s[0:1], 0x0
	s_load_dwordx2 s[30:31], s[0:1], 0x20
	s_load_dwordx2 s[36:37], s[0:1], 0x30
	s_load_dwordx2 s[34:35], s[0:1], 0x50
	s_load_dwordx2 s[12:13], s[0:1], 0xb0
	s_load_dwordx2 s[14:15], s[0:1], 0x90
	s_load_dwordx2 s[6:7], s[0:1], 0x80
	s_load_dwordx2 s[4:5], s[0:1], 0x60
	s_load_dwordx2 s[24:25], s[0:1], 0xc0
	s_load_dwordx2 s[28:29], s[0:1], 0xe0
	s_load_dword s26, s[0:1], 0xf0
	s_lshr_b32 s0, s9, 27
	s_add_i32 s8, s8, s0
	s_ashr_i32 s0, s8, 5
	v_lshrrev_b32_e32 v1, 6, v0
	v_lshlrev_b32_e32 v1, 10, v1
	s_abs_i32 s9, s2
	s_waitcnt lgkmcnt(0)
	v_readfirstlane_b32 s31, v1
	s_add_i32 s41, s31, 0x18000
	s_add_i32 s42, s41, 0x4000
	s_ashr_i32 s1, s2, 31
	s_lshr_b32 s3, s1, 25
	s_add_i32 s3, s2, s3
	s_ashr_i32 s7, s3, 7
	s_lshl_b32 s7, s7, 2
	s_sub_i32 s0, s0, s7
	s_min_i32 s0, s0, 4
	s_abs_i32 s8, s0
	s_sub_i32 s10, 0, s8
	s_add_i32 s33, s31, 0x4000
	v_cvt_f32_u32_e32 v1, s8
	v_lshlrev_b32_e32 v2, 4, v0
	s_add_i32 s35, s31, 0x8000
	s_add_i32 s38, s35, 0x4000
	v_rcp_iflag_f32_e32 v1, v1
	s_add_i32 s39, s31, 0x10000
	s_add_i32 s40, s39, 0x4000
	s_movk_i32 s69, 0x70
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	v_bitop3_b32 v2, v2, s69, v0 bitop3:0x48
	v_lshrrev_b32_e32 v3, 3, v0
	v_or_b32_e32 v4, 0x60, v3
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
	s_sub_i32 s27, s9, s1
	s_add_i32 s27, s27, s7
	v_bfe_u32 v134, v0, 6, 1
	v_and_b32_e32 v76, 48, v0
	s_and_b32 s1, s3, 0xffffff80
	s_sub_i32 s1, s2, s1
	s_xor_b32 s0, s1, s0
	s_ashr_i32 s0, s0, 31
	v_lshrrev_b32_e32 v1, 7, v0
	v_lshlrev_b32_e32 v75, 13, v1
	s_abs_i32 s1, s1
	s_mul_hi_u32 s2, s1, s11
	s_mul_i32 s3, s2, s8
	s_sub_i32 s1, s1, s3
	s_sub_i32 s7, s1, s8
	v_lshlrev_b32_e32 v74, 3, v0
	s_add_i32 s3, s2, 1
	s_cmp_ge_u32 s1, s8
	s_cselect_b32 s2, s3, s2
	v_lshlrev_b32_e32 v78, 13, v134
	v_or_b32_e32 v79, 0x10000, v78
	s_cselect_b32 s1, s7, s1
	s_add_i32 s3, s2, 1
	s_cmp_ge_u32 s1, s8
	s_cselect_b32 s1, s3, s2
	s_mov_b32 s3, 0x110000
	s_mov_b32 s43, 0
	s_xor_b32 s1, s1, s0
	s_sub_i32 s29, s1, s0
	s_lshl_b32 s50, s29, 8
	s_mul_i32 s63, s50, s34
	s_or_b32 s66, s63, 0x80
	s_lshl_b32 s44, s27, 8
	s_mul_i32 s57, s44, s30
	s_or_b32 s60, s57, 0x80
	s_mov_b32 s53, s63
	s_mov_b32 s46, s57
	;;#ASMSTART
	;;#ASMEND
	v_mad_u64_u32 v[98:99], s[0:1], v3, s30, v[2:3]
	s_lshl_b32 s0, s30, 5
	s_nop 0
	v_add_u32_e32 v99, s0, v98
	v_add_u32_e32 v122, s0, v99
	v_mad_u64_u32 v[102:103], s[0:1], v3, s34, v[2:3]
	s_mov_b32 s2, -1
	v_mad_u64_u32 v[100:101], s[0:1], v4, s30, v[2:3]
	s_lshl_b32 s0, s34, 5
	s_nop 0
	v_add_u32_e32 v101, s0, v102
	v_add_u32_e32 v103, s0, v101
	v_mad_u64_u32 v[104:105], s[0:1], v4, s34, v[2:3]
	v_and_b32_e32 v105, 0x1f8, v74
	v_accvgpr_write_b32 a195, 0
	v_lshl_or_b32 v4, v1, 6, s44
	v_ashrrev_i32_e32 v2, 6, v4
	v_mul_lo_u32 v2, v2, s6
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], s[4:5], 0, v[2:3]
	s_add_i32 s44, s31, 0x2000
	s_mov_b32 s45, s44
	v_readfirstlane_b32 s1, v3
	v_readfirstlane_b32 s0, v2
	v_or_b32_e32 v2, 0x80, v4
	v_ashrrev_i32_e32 v2, 6, v2
	v_mul_lo_u32 v2, v2, s6
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], s[4:5], 0, v[2:3]
	s_mov_b64 s[6:7], s[2:3]
	s_mov_b64 s[4:5], s[0:1]
	v_lshl_or_b32 v4, v134, 6, s50
	s_mov_b32 s50, s39
	v_readfirstlane_b32 s9, v3
	s_mov_b32 s5, s9
	v_readfirstlane_b32 s8, v2
	s_mov_b32 s4, s8
	s_mov_b64 s[10:11], s[2:3]
	s_mov_b64 s[8:9], s[0:1]
	v_ashrrev_i32_e32 v2, 6, v4
	v_mul_lo_u32 v2, v2, s12
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], s[14:15], 0, v[2:3]
	v_accvgpr_write_b32 a194, 0
	v_readfirstlane_b32 s13, v2
	v_readfirstlane_b32 s16, v3
	s_mov_b32 s8, s13
	v_or_b32_e32 v2, 0x80, v4
	v_ashrrev_i32_e32 v2, 6, v2
	v_mul_lo_u32 v2, v2, s12
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], s[14:15], 0, v[2:3]
	s_mov_b64 s[14:15], s[2:3]
	s_mov_b32 s9, s16
	v_readfirstlane_b32 s16, v2
	s_mov_b64 s[12:13], s[0:1]
	s_mov_b32 s12, s16
	v_readfirstlane_b32 s17, v3
	s_mov_b32 s13, s17
	s_mov_b64 s[18:19], s[2:3]
	s_mov_b64 s[16:17], s[0:1]
	s_mov_b32 s16, s20
	s_mov_b32 s17, s21
	s_mov_b64 s[22:23], s[2:3]
	s_mov_b64 s[20:21], s[0:1]
	s_mov_b32 s20, s36
	s_mov_b32 s36, s31
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s36
	s_add_i32 s36, s31, 0x1000
	s_mov_b32 s21, s37
	s_mov_b32 s37, s36
	buffer_load_dwordx4 v98, s[16:19], s46 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s37
	v_lshlrev_b32_e32 v2, 7, v0
	v_and_b32_e32 v77, 0x780, v2
	v_or_b32_e32 v2, v77, v76
	v_or_b32_e32 v18, v2, v79
	v_or_b32_e32 v79, v77, v79
	buffer_load_dwordx4 v99, s[16:19], s46 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s45
	s_add_i32 s45, s31, 0x3000
	s_mov_b32 s47, s45
	v_or_b32_e32 v3, v2, v75
	v_bitop3_b32 v4, v74, v3, s69 bitop3:0x6c
	v_or_b32_e32 v3, 64, v3
	v_bitop3_b32 v3, v74, v3, s69 bitop3:0x6c
	buffer_load_dwordx4 v122, s[16:19], s46 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s47
	s_mov_b32 s47, s35
	v_or_b32_e32 v75, v77, v75
	v_or_b32_e32 v80, v75, v76
	v_bitop3_b32 v123, v74, v80, s69 bitop3:0x6c
	v_or_b32_e32 v80, 64, v76
	v_or_b32_e32 v81, v75, v80
	v_bitop3_b32 v124, v74, v81, s69 bitop3:0x6c
	v_or_b32_e32 v81, 0x4000, v75
	v_or_b32_e32 v82, v81, v76
	v_bitop3_b32 v125, v74, v82, s69 bitop3:0x6c
	buffer_load_dwordx4 v100, s[16:19], s46 offen lds
	s_lshl_b32 s46, s30, 7
	s_add_i32 s46, s57, s46
	s_add_i32 s64, s46, 0x80
	v_or_b32_e32 v81, v81, v80
	v_bitop3_b32 v126, v74, v81, s69 bitop3:0x6c
	v_add_u32_e32 v81, v79, v76
	v_lshrrev_b32_e32 v82, 4, v81
	v_bitop3_b32 v127, v82, v81, s69 bitop3:0x6c
	s_mov_b32 s57, s33
	v_add_u32_e32 v79, v79, v80
	v_lshrrev_b32_e32 v81, 4, v79
	v_bitop3_b32 v128, v81, v79, s69 bitop3:0x6c
	s_mov_b32 s51, s46
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s47
	s_add_i32 s47, s35, 0x1000
	s_mov_b32 s48, s47
	s_mul_i32 s30, s30, s27
	s_movk_i32 s37, 0x2000
	buffer_load_dwordx4 v98, s[16:19], s51 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s48
	s_add_i32 s48, s35, 0x2000
	s_mov_b32 s49, s48
	v_or_b32_e32 v77, v78, v77
	v_or_b32_e32 v78, 0x14000, v77
	v_or_b32_e32 v79, v78, v76
	v_bitop3_b32 v129, v74, v79, s69 bitop3:0x6c
	buffer_load_dwordx4 v99, s[16:19], s51 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s49
	s_add_i32 s49, s35, 0x3000
	s_mov_b32 s52, s49
	v_or_b32_e32 v78, v78, v80
	v_bitop3_b32 v130, v74, v78, s69 bitop3:0x6c
	v_or_b32_e32 v78, 0x18000, v77
	v_or_b32_e32 v79, v78, v76
	v_bitop3_b32 v131, v74, v79, s69 bitop3:0x6c
	buffer_load_dwordx4 v122, s[16:19], s51 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s52
	v_or_b32_e32 v78, v78, v80
	v_bitop3_b32 v132, v74, v78, s69 bitop3:0x6c
	buffer_load_dwordx4 v100, s[16:19], s51 offen lds
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s50
	s_add_i32 s50, s39, 0x1000
	s_mov_b32 s51, s50
	v_or_b32_e32 v77, 0x1c000, v77
	v_or_b32_e32 v78, v77, v76
	v_bitop3_b32 v133, v74, v78, s69 bitop3:0x6c
	buffer_load_dwordx4 v102, s[20:23], s53 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s51
	s_add_i32 s51, s39, 0x2000
	s_mov_b32 s52, s51
	v_or_b32_e32 v77, v77, v80
	v_bitop3_b32 v135, v74, v77, s69 bitop3:0x6c
	v_or_b32_e32 v77, 0x8000, v75
	v_or_b32_e32 v78, v77, v76
	v_bitop3_b32 v136, v74, v78, s69 bitop3:0x6c
	buffer_load_dwordx4 v101, s[20:23], s53 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s52
	s_add_i32 s52, s39, 0x3000
	s_mov_b32 s54, s52
	v_or_b32_e32 v77, v77, v80
	v_bitop3_b32 v137, v74, v77, s69 bitop3:0x6c
	buffer_load_dwordx4 v103, s[20:23], s53 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s54
	s_mov_b32 s54, s41
	v_or_b32_e32 v75, 0xc000, v75
	v_or_b32_e32 v76, v75, v76
	v_bitop3_b32 v138, v74, v76, s69 bitop3:0x6c
	buffer_load_dwordx4 v104, s[20:23], s53 offen lds
	s_lshl_b32 s53, s34, 7
	s_add_i32 s53, s63, s53
	s_add_i32 s70, s53, 0x80
	v_or_b32_e32 v75, v75, v80
	v_bitop3_b32 v139, v74, v75, s69 bitop3:0x6c
	s_mov_b32 s63, s40
	s_mov_b32 s58, s53
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s54
	s_add_i32 s54, s41, 0x1000
	s_mov_b32 s55, s54
	s_mul_i32 s34, s34, s29
	s_lshl_b32 s30, s30, 8
	buffer_load_dwordx4 v102, s[20:23], s58 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s55
	s_add_i32 s55, s41, 0x2000
	s_mov_b32 s56, s55
	v_lshrrev_b32_e32 v2, 4, v18
	v_bitop3_b32 v14, v2, v18, s69 bitop3:0x6c
	v_add_u32_e32 v18, 64, v18
	v_lshrrev_b32_e32 v19, 4, v18
	v_bitop3_b32 v30, v19, v18, s69 bitop3:0x6c
	s_lshl_b32 s34, s34, 8
	buffer_load_dwordx4 v101, s[20:23], s58 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s56
	s_add_i32 s56, s41, 0x3000
	s_mov_b32 s59, s56
	buffer_load_dwordx4 v103, s[20:23], s58 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s59
	s_mov_b32 s69, 15
	buffer_load_dwordx4 v104, s[20:23], s58 offen lds
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s57
	s_add_i32 s57, s31, 0x5000
	s_mov_b32 s58, s57
	buffer_load_dwordx4 v98, s[16:19], s60 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s58
	s_add_i32 s58, s31, 0x6000
	s_mov_b32 s59, s58
	buffer_load_dwordx4 v99, s[16:19], s60 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s59
	s_add_i32 s59, s31, 0x7000
	s_mov_b32 s61, s59
	buffer_load_dwordx4 v122, s[16:19], s60 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s61
	v_accvgpr_write_b32 a193, 0
	buffer_load_dwordx4 v100, s[16:19], s60 offen lds
	s_mov_b32 s60, s38
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s60
	s_add_i32 s60, s35, 0x5000
	s_mov_b32 s61, s60
	buffer_load_dwordx4 v98, s[16:19], s64 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s61
	s_add_i32 s61, s35, 0x6000
	s_mov_b32 s62, s61
	buffer_load_dwordx4 v99, s[16:19], s64 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s62
	s_add_i32 s62, s35, 0x7000
	s_mov_b32 s65, s62
	buffer_load_dwordx4 v122, s[16:19], s64 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s65
	v_accvgpr_write_b32 a192, 0
	buffer_load_dwordx4 v100, s[16:19], s64 offen lds
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s63
	s_add_i32 s63, s39, 0x5000
	s_mov_b32 s64, s63
	buffer_load_dwordx4 v102, s[20:23], s66 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s64
	s_add_i32 s64, s39, 0x6000
	s_mov_b32 s65, s64
	buffer_load_dwordx4 v101, s[20:23], s66 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s65
	s_add_i32 s65, s39, 0x7000
	s_mov_b32 s67, s65
	buffer_load_dwordx4 v103, s[20:23], s66 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s67
	v_accvgpr_write_b32 a199, 0
	buffer_load_dwordx4 v104, s[20:23], s66 offen lds
	s_mov_b32 s66, s42
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s66
	s_add_i32 s66, s41, 0x5000
	s_mov_b32 s67, s66
	buffer_load_dwordx4 v102, s[20:23], s70 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s67
	s_add_i32 s67, s41, 0x6000
	s_mov_b32 s68, s67
	buffer_load_dwordx4 v101, s[20:23], s70 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s68
	s_add_i32 s68, s41, 0x7000
	s_mov_b32 s71, s68
	buffer_load_dwordx4 v103, s[20:23], s70 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s71
	v_accvgpr_write_b32 a198, 0
	buffer_load_dwordx4 v104, s[20:23], s70 offen lds
	s_mov_b32 s70, 0
	;;#ASMSTART
	buffer_load_dwordx2 v[72:73], v105, s[0:3], s70 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[66:67], v105, s[4:7], s70 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[70:71], v105, s[8:11], s70 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[68:69], v105, s[12:15], s70 offen
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[34:37], v4 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[38:41], v4 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[42:45], v4 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[46:49], v4 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[50:53], v3 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[54:57], v3 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[58:61], v3 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[62:65], v3 offset:0x1800

	;;#ASMEND
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
	v_accvgpr_write_b32 a3, 0
	v_accvgpr_write_b32 a2, 0
	v_accvgpr_write_b32 a1, 0
	v_accvgpr_write_b32 a0, 0
	v_accvgpr_write_b32 a7, 0
	v_accvgpr_write_b32 a6, 0
	v_accvgpr_write_b32 a5, 0
	v_accvgpr_write_b32 a4, 0
	v_accvgpr_write_b32 a11, 0
	v_accvgpr_write_b32 a10, 0
	v_accvgpr_write_b32 a9, 0
	v_accvgpr_write_b32 a8, 0
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
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
.LBB0_7:                                ; %.preheader
                                        ; =>This Inner Loop Header: Depth=1
	s_add_i32 s70, s30, s43
	s_add_i32 s74, s70, 0x100
	s_mov_b32 m0, s31
	s_add_i32 s78, s37, 0xffffe200
	;;#ASMSTART
	buffer_load_dwordx2 v[76:77], v105, s[0:3], s78 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[106:107], v105, s[4:7], s78 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[74:75], v105, s[8:11], s78 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[108:109], v105, s[12:15], s78 offen
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[34:37], v[2:5], a[28:31],  v72, v70 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[140:143], v131 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[34:37], v[6:9], a[32:35],  v72, v70 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[144:147], v131 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[34:37], v[10:13], a[36:39],  v72, v71 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[148:151], v131 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[34:37], v[14:17], a[40:43],  v72, v71 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[152:155], v131 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[50:53], v[18:21], a[28:31],  v72, v70 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[156:159], v132 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[50:53], v[22:25], a[32:35],  v72, v70 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[160:163], v132 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[50:53], v[26:29], a[36:39],  v72, v71 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[164:167], v132 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[50:53], v[30:33], a[40:43],  v72, v71 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[168:171], v132 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[38:41], v[2:5], a[44:47],  v72, v70 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[38:41], v[6:9], a[48:51],  v72, v70 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[38:41], v[10:13], a[52:55],  v72, v71 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[38:41], v[14:17], a[56:59],  v72, v71 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[54:57], v[18:21], a[44:47],  v72, v70 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[54:57], v[22:25], a[48:51],  v72, v70 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[54:57], v[26:29], a[52:55],  v72, v71 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[54:57], v[30:33], a[56:59],  v72, v71 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[42:45], v[2:5], a[60:63],  v73, v70 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[42:45], v[6:9], a[64:67],  v73, v70 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[42:45], v[10:13], a[68:71], v73, v71 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[42:45], v[14:17], a[72:75], v73, v71 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[58:61], v[18:21], a[60:63],  v73, v70 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[58:61], v[22:25], a[64:67],  v73, v70 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[58:61], v[26:29], a[68:71], v73, v71 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[58:61], v[30:33], a[72:75], v73, v71 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[46:49], v[2:5], a[76:79], v73, v70 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[46:49], v[6:9], a[80:83], v73, v70 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[46:49], v[10:13], a[84:87], v73, v71 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[46:49], v[14:17], a[88:91], v73, v71 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[62:65], v[18:21], a[76:79], v73, v70 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[62:65], v[22:25], a[80:83], v73, v70 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[62:65], v[26:29], a[84:87], v73, v71 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[62:65], v[30:33], a[88:91], v73, v71 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[34:37], v[140:143], a[0:3], v72, v68 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[78:81], v136 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[34:37], v[144:147], a[4:7], v72, v68 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[82:85], v136 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[34:37], v[148:151], a[8:11], v72, v69 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[86:89], v136 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[34:37], v[152:155], a[12:15], v72, v69 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[90:93], v136 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[50:53], v[156:159], a[0:3], v72, v68 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[94:97], v137 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[50:53], v[160:163], a[4:7], v72, v68 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[110:113], v137 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[50:53], v[164:167], a[8:11], v72, v69 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[114:117], v137 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[50:53], v[168:171], a[12:15], v72, v69 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[118:121], v137 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[38:41], v[140:143], a[16:19], v72, v68 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[38:41], v[144:147], a[20:23], v72, v68 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[38:41], v[148:151], a[24:27], v72, v69 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[38:41], v[152:155], a[92:95], v72, v69 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[54:57], v[156:159], a[16:19], v72, v68 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[54:57], v[160:163], a[20:23], v72, v68 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[54:57], v[164:167], a[24:27], v72, v69 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[54:57], v[168:171], a[92:95], v72, v69 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[42:45], v[140:143], a[96:99], v73, v68 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[42:45], v[144:147], a[100:103], v73, v68 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[42:45], v[148:151], a[104:107], v73, v69 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[42:45], v[152:155], a[108:111], v73, v69 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[58:61], v[156:159], a[96:99], v73, v68 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[58:61], v[160:163], a[100:103], v73, v68 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[58:61], v[164:167], a[104:107], v73, v69 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[58:61], v[168:171], a[108:111], v73, v69 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[46:49], v[140:143], a[112:115], v73, v68 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[46:49], v[144:147], a[116:119], v73, v68 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[46:49], v[148:151], a[120:123], v73, v69 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[46:49], v[152:155], a[124:127], v73, v69 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[62:65], v[156:159], a[112:115], v73, v68 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[62:65], v[160:163], a[116:119], v73, v68 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[62:65], v[164:167], a[120:123], v73, v69 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[62:65], v[168:171], a[124:127], v73, v69 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[78:81], v[2:5], a[128:131],  v66, v70 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v125 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[78:81], v[6:9], a[132:135],  v66, v70 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v125 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[78:81], v[10:13], a[136:139],  v66, v71 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v125 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[78:81], v[14:17], a[140:143],  v66, v71 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v125 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[94:97], v[18:21], a[128:131],  v66, v70 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v126 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[94:97], v[22:25], a[132:135],  v66, v70 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v126 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[94:97], v[26:29], a[136:139],  v66, v71 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v126 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[94:97], v[30:33], a[140:143],  v66, v71 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[62:65], v126 offset:6144

	;;#ASMEND
	buffer_load_dwordx4 v98, s[16:19], s74 offen lds
	s_mov_b32 m0, s36
	s_add_i32 s71, s46, s43
	buffer_load_dwordx4 v99, s[16:19], s74 offen lds
	s_mov_b32 m0, s44
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[82:85], v[2:5], a[144:147],  v66, v70 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[82:85], v[6:9], a[148:151],  v66, v70 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[82:85], v[10:13], a[152:155],  v66, v71 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[82:85], v[14:17], a[156:159],  v66, v71 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[110:113], v[18:21], a[144:147],  v66, v70 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[110:113], v[22:25], a[148:151],  v66, v70 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[110:113], v[26:29], a[152:155],  v66, v71 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[110:113], v[30:33], a[156:159],  v66, v71 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_i32 s75, s71, 0x100
	buffer_load_dwordx4 v122, s[16:19], s74 offen lds
	s_mov_b32 m0, s45
	s_add_i32 s72, s34, s43
	buffer_load_dwordx4 v100, s[16:19], s74 offen lds
	s_mov_b32 m0, s35
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[86:89], v[2:5], a[160:163],  v67, v70 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[86:89], v[6:9], a[164:167],  v67, v70 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[86:89], v[10:13], a[168:171], v67, v71 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[86:89], v[14:17], a[172:175], v67, v71 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[114:117], v[18:21], a[160:163],  v67, v70 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[114:117], v[22:25], a[164:167],  v67, v70 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[114:117], v[26:29], a[168:171], v67, v71 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[114:117], v[30:33], a[172:175], v67, v71 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_i32 s76, s72, 0x100
	buffer_load_dwordx4 v98, s[16:19], s75 offen lds
	s_mov_b32 m0, s47
	s_add_i32 s73, s53, s43
	buffer_load_dwordx4 v99, s[16:19], s75 offen lds
	s_mov_b32 m0, s48
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[90:93], v[2:5], a[176:179], v67, v70 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[90:93], v[6:9], a[180:183], v67, v70 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[90:93], v[10:13], a[184:187], v67, v71 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[90:93], v[14:17], a[188:191], v67, v71 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[118:121], v[18:21], a[176:179], v67, v70 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[118:121], v[22:25], a[180:183], v67, v70 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[118:121], v[26:29], a[184:187], v67, v71 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[118:121], v[30:33], a[188:191], v67, v71 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_i32 s77, s73, 0x100
	buffer_load_dwordx4 v122, s[16:19], s75 offen lds
	s_mov_b32 m0, s49
	s_add_i32 s78, s37, 0xffffe400
	buffer_load_dwordx4 v100, s[16:19], s75 offen lds
	s_mov_b32 m0, s39
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[78:81], v[140:143], a[192:195],  v66, v68 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[2:5], v129 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[78:81], v[144:147], a[196:199],  v66, v68 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[6:9], v129 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[78:81], v[148:151], a[200:203],  v66, v69 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[10:13], v129 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[78:81], v[152:155], a[204:207],  v66, v69 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[14:17], v129 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[94:97], v[156:159], a[192:195],  v66, v68 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[18:21], v130 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[94:97], v[160:163], a[196:199],  v66, v68 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[22:25], v130 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[94:97], v[164:167], a[200:203],  v66, v69 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[26:29], v130 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[94:97], v[168:171], a[204:207],  v66, v69 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v130 offset:6144

	;;#ASMEND
	s_add_i32 s75, s72, 0x180
	buffer_load_dwordx4 v102, s[20:23], s76 offen lds
	s_mov_b32 m0, s50
	s_add_i32 s74, s73, 0x180
	buffer_load_dwordx4 v101, s[20:23], s76 offen lds
	s_mov_b32 m0, s51
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[82:85], v[140:143], a[208:211],  v66, v68 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[82:85], v[144:147], a[212:215],  v66, v68 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[82:85], v[148:151], a[216:219],  v66, v69 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[82:85], v[152:155], a[220:223],  v66, v69 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[110:113], v[156:159], a[208:211],  v66, v68 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[110:113], v[160:163], a[212:215],  v66, v68 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[110:113], v[164:167], a[216:219],  v66, v69 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[110:113], v[168:171], a[220:223],  v66, v69 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v103, s[20:23], s76 offen lds
	s_mov_b32 m0, s52
	s_nop 0
	buffer_load_dwordx4 v104, s[20:23], s76 offen lds
	s_mov_b32 m0, s41
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[86:89], v[140:143], a[224:227],  v67, v68 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[86:89], v[144:147], a[228:231],  v67, v68 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[86:89], v[148:151], a[232:235], v67, v69 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[86:89], v[152:155], a[236:239], v67, v69 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[114:117], v[156:159], a[224:227],  v67, v68 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[114:117], v[160:163], a[228:231],  v67, v68 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[114:117], v[164:167], a[232:235], v67, v69 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[114:117], v[168:171], a[236:239], v67, v69 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_i32 s76, s71, 0x180
	buffer_load_dwordx4 v102, s[20:23], s77 offen lds
	s_mov_b32 m0, s54
	s_nop 0
	buffer_load_dwordx4 v101, s[20:23], s77 offen lds
	s_mov_b32 m0, s55
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[90:93], v[140:143], a[240:243], v67, v68 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[90:93], v[144:147], a[244:247], v67, v68 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[90:93], v[148:151], a[248:251], v67, v69 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[90:93], v[152:155], a[252:255], v67, v69 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[118:121], v[156:159], a[240:243], v67, v68 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[118:121], v[160:163], a[244:247], v67, v68 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[118:121], v[164:167], a[248:251], v67, v69 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[118:121], v[168:171], a[252:255], v67, v69 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v103, s[20:23], s77 offen lds
	s_mov_b32 m0, s56
	s_nop 0
	buffer_load_dwordx4 v104, s[20:23], s77 offen lds
	s_add_i32 s77, s70, 0x180
	s_mov_b32 m0, s33
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[118:119], v105, s[0:3], s78 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[110:111], v105, s[4:7], s78 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[114:115], v105, s[8:11], s78 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[112:113], v105, s[12:15], s78 offen
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[34:37], v[2:5], a[28:31],  v76, v74 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[172:175], v133 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[34:37], v[6:9], a[32:35],  v76, v74 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[176:179], v133 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[34:37], v[10:13], a[36:39],  v76, v75 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[180:183], v133 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[34:37], v[14:17], a[40:43],  v76, v75 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[184:187], v133 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[50:53], v[18:21], a[28:31],  v76, v74 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[188:191], v135 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[50:53], v[22:25], a[32:35],  v76, v74 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[192:195], v135 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[50:53], v[26:29], a[36:39],  v76, v75 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[196:199], v135 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[50:53], v[30:33], a[40:43],  v76, v75 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[200:203], v135 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[38:41], v[2:5], a[44:47],  v76, v74 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[38:41], v[6:9], a[48:51],  v76, v74 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[38:41], v[10:13], a[52:55],  v76, v75 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[38:41], v[14:17], a[56:59],  v76, v75 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[54:57], v[18:21], a[44:47],  v76, v74 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[54:57], v[22:25], a[48:51],  v76, v74 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[54:57], v[26:29], a[52:55],  v76, v75 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[54:57], v[30:33], a[56:59],  v76, v75 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[42:45], v[2:5], a[60:63],  v77, v74 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[42:45], v[6:9], a[64:67],  v77, v74 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[42:45], v[10:13], a[68:71], v77, v75 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[42:45], v[14:17], a[72:75], v77, v75 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[58:61], v[18:21], a[60:63],  v77, v74 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[58:61], v[22:25], a[64:67],  v77, v74 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[58:61], v[26:29], a[68:71], v77, v75 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[58:61], v[30:33], a[72:75], v77, v75 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[46:49], v[2:5], a[76:79], v77, v74 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[46:49], v[6:9], a[80:83], v77, v74 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[46:49], v[10:13], a[84:87], v77, v75 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[46:49], v[14:17], a[88:91], v77, v75 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[62:65], v[18:21], a[76:79], v77, v74 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[62:65], v[22:25], a[80:83], v77, v74 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[62:65], v[26:29], a[84:87], v77, v75 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[62:65], v[30:33], a[88:91], v77, v75 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[34:37], v[172:175], a[0:3], v76, v108 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[140:143], v138 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[34:37], v[176:179], a[4:7], v76, v108 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[144:147], v138 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[34:37], v[180:183], a[8:11], v76, v109 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[148:151], v138 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[34:37], v[184:187], a[12:15], v76, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[152:155], v138 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[50:53], v[188:191], a[0:3], v76, v108 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[156:159], v139 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[50:53], v[192:195], a[4:7], v76, v108 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[160:163], v139 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[50:53], v[196:199], a[8:11], v76, v109 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[164:167], v139 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[50:53], v[200:203], a[12:15], v76, v109 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[168:171], v139 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[38:41], v[172:175], a[16:19], v76, v108 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[38:41], v[176:179], a[20:23], v76, v108 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[38:41], v[180:183], a[24:27], v76, v109 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[38:41], v[184:187], a[92:95], v76, v109 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[54:57], v[188:191], a[16:19], v76, v108 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[54:57], v[192:195], a[20:23], v76, v108 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[54:57], v[196:199], a[24:27], v76, v109 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[54:57], v[200:203], a[92:95], v76, v109 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[42:45], v[172:175], a[96:99], v77, v108 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[42:45], v[176:179], a[100:103], v77, v108 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[42:45], v[180:183], a[104:107], v77, v109 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[42:45], v[184:187], a[108:111], v77, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[58:61], v[188:191], a[96:99], v77, v108 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[58:61], v[192:195], a[100:103], v77, v108 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[58:61], v[196:199], a[104:107], v77, v109 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[58:61], v[200:203], a[108:111], v77, v109 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[46:49], v[172:175], a[112:115], v77, v108 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[46:49], v[176:179], a[116:119], v77, v108 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[46:49], v[180:183], a[120:123], v77, v109 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[46:49], v[184:187], a[124:127], v77, v109 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[62:65], v[188:191], a[112:115], v77, v108 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[62:65], v[192:195], a[116:119], v77, v108 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[62:65], v[196:199], a[120:123], v77, v109 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[62:65], v[200:203], a[124:127], v77, v109 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[140:143], v[2:5], a[128:131],  v106, v74 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v123 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[140:143], v[6:9], a[132:135],  v106, v74 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v123 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[140:143], v[10:13], a[136:139],  v106, v75 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v123 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[140:143], v[14:17], a[140:143],  v106, v75 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v123 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[156:159], v[18:21], a[128:131],  v106, v74 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v124 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[156:159], v[22:25], a[132:135],  v106, v74 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v124 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[156:159], v[26:29], a[136:139],  v106, v75 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v124 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[156:159], v[30:33], a[140:143],  v106, v75 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[62:65], v124 offset:6144

	;;#ASMEND
	buffer_load_dwordx4 v98, s[16:19], s77 offen lds
	s_mov_b32 m0, s57
	s_add_i32 s78, s37, 0xffffe600
	buffer_load_dwordx4 v99, s[16:19], s77 offen lds
	s_mov_b32 m0, s58
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[144:147], v[2:5], a[144:147],  v106, v74 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[144:147], v[6:9], a[148:151],  v106, v74 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[144:147], v[10:13], a[152:155],  v106, v75 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[144:147], v[14:17], a[156:159],  v106, v75 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[160:163], v[18:21], a[144:147],  v106, v74 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[160:163], v[22:25], a[148:151],  v106, v74 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[160:163], v[26:29], a[152:155],  v106, v75 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[160:163], v[30:33], a[156:159],  v106, v75 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v122, s[16:19], s77 offen lds
	s_mov_b32 m0, s59
	s_nop 0
	buffer_load_dwordx4 v100, s[16:19], s77 offen lds
	s_mov_b32 m0, s38
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[148:151], v[2:5], a[160:163],  v107, v74 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[148:151], v[6:9], a[164:167],  v107, v74 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[148:151], v[10:13], a[168:171], v107, v75 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[148:151], v[14:17], a[172:175], v107, v75 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[164:167], v[18:21], a[160:163],  v107, v74 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[164:167], v[22:25], a[164:167],  v107, v74 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[164:167], v[26:29], a[168:171], v107, v75 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[164:167], v[30:33], a[172:175], v107, v75 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_i32 s77, s70, 0x200
	buffer_load_dwordx4 v98, s[16:19], s76 offen lds
	s_mov_b32 m0, s60
	s_nop 0
	buffer_load_dwordx4 v99, s[16:19], s76 offen lds
	s_mov_b32 m0, s61
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[152:155], v[2:5], a[176:179], v107, v74 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[152:155], v[6:9], a[180:183], v107, v74 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[152:155], v[10:13], a[184:187], v107, v75 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[152:155], v[14:17], a[188:191], v107, v75 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[168:171], v[18:21], a[176:179], v107, v74 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[168:171], v[22:25], a[180:183], v107, v74 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[168:171], v[26:29], a[184:187], v107, v75 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[168:171], v[30:33], a[188:191], v107, v75 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v122, s[16:19], s76 offen lds
	s_mov_b32 m0, s62
	s_nop 0
	buffer_load_dwordx4 v100, s[16:19], s76 offen lds
	s_mov_b32 m0, s40
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[140:143], v[172:175], a[192:195],  v106, v108 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[66:69], v127 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[140:143], v[176:179], a[196:199],  v106, v108 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[70:73], v127 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[140:143], v[180:183], a[200:203],  v106, v109 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[74:77], v127 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[140:143], v[184:187], a[204:207],  v106, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[78:81], v127 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[156:159], v[188:191], a[192:195],  v106, v108 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[82:85], v128 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[156:159], v[192:195], a[196:199],  v106, v108 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[86:89], v128 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[156:159], v[196:199], a[200:203],  v106, v109 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[90:93], v128 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[156:159], v[200:203], a[204:207],  v106, v109 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[94:97], v128 offset:6144

	;;#ASMEND
	s_add_i32 s76, s71, 0x200
	buffer_load_dwordx4 v102, s[20:23], s75 offen lds
	s_mov_b32 m0, s63
	s_nop 0
	buffer_load_dwordx4 v101, s[20:23], s75 offen lds
	s_mov_b32 m0, s64
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[144:147], v[172:175], a[208:211],  v106, v108 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[144:147], v[176:179], a[212:215],  v106, v108 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[144:147], v[180:183], a[216:219],  v106, v109 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[144:147], v[184:187], a[220:223],  v106, v109 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[160:163], v[188:191], a[208:211],  v106, v108 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[160:163], v[192:195], a[212:215],  v106, v108 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[160:163], v[196:199], a[216:219],  v106, v109 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[160:163], v[200:203], a[220:223],  v106, v109 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v103, s[20:23], s75 offen lds
	s_mov_b32 m0, s65
	s_nop 0
	buffer_load_dwordx4 v104, s[20:23], s75 offen lds
	s_mov_b32 m0, s42
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[148:151], v[172:175], a[224:227],  v107, v108 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[148:151], v[176:179], a[228:231],  v107, v108 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[148:151], v[180:183], a[232:235], v107, v109 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[148:151], v[184:187], a[236:239], v107, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[164:167], v[188:191], a[224:227],  v107, v108 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[164:167], v[192:195], a[228:231],  v107, v108 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[164:167], v[196:199], a[232:235], v107, v109 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[164:167], v[200:203], a[236:239], v107, v109 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_i32 s75, s72, 0x200
	buffer_load_dwordx4 v102, s[20:23], s74 offen lds
	s_mov_b32 m0, s66
	s_nop 0
	buffer_load_dwordx4 v101, s[20:23], s74 offen lds
	s_mov_b32 m0, s67
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[152:155], v[172:175], a[240:243], v107, v108 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[152:155], v[176:179], a[244:247], v107, v108 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[152:155], v[180:183], a[248:251], v107, v109 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[152:155], v[184:187], a[252:255], v107, v109 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[168:171], v[188:191], a[240:243], v107, v108 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[168:171], v[192:195], a[244:247], v107, v108 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[168:171], v[196:199], a[248:251], v107, v109 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[168:171], v[200:203], a[252:255], v107, v109 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v103, s[20:23], s74 offen lds
	s_mov_b32 m0, s68
	s_nop 0
	buffer_load_dwordx4 v104, s[20:23], s74 offen lds
	s_mov_b32 m0, s31
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[120:121], v105, s[0:3], s78 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[106:107], v105, s[4:7], s78 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[116:117], v105, s[8:11], s78 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[108:109], v105, s[12:15], s78 offen
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[34:37], v[66:69], a[28:31],  v118, v114 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[172:175], v131 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[34:37], v[70:73], a[32:35],  v118, v114 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[176:179], v131 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[34:37], v[74:77], a[36:39],  v118, v115 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[180:183], v131 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[34:37], v[78:81], a[40:43],  v118, v115 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[184:187], v131 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[50:53], v[82:85], a[28:31],  v118, v114 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[188:191], v132 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[50:53], v[86:89], a[32:35],  v118, v114 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[192:195], v132 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[50:53], v[90:93], a[36:39],  v118, v115 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[196:199], v132 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[50:53], v[94:97], a[40:43],  v118, v115 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[200:203], v132 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[38:41], v[66:69], a[44:47],  v118, v114 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[38:41], v[70:73], a[48:51],  v118, v114 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[38:41], v[74:77], a[52:55],  v118, v115 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[38:41], v[78:81], a[56:59],  v118, v115 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[54:57], v[82:85], a[44:47],  v118, v114 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[54:57], v[86:89], a[48:51],  v118, v114 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[54:57], v[90:93], a[52:55],  v118, v115 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[54:57], v[94:97], a[56:59],  v118, v115 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[42:45], v[66:69], a[60:63],  v119, v114 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[42:45], v[70:73], a[64:67],  v119, v114 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[42:45], v[74:77], a[68:71], v119, v115 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[42:45], v[78:81], a[72:75], v119, v115 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[58:61], v[82:85], a[60:63],  v119, v114 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[58:61], v[86:89], a[64:67],  v119, v114 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[58:61], v[90:93], a[68:71], v119, v115 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[58:61], v[94:97], a[72:75], v119, v115 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[46:49], v[66:69], a[76:79], v119, v114 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[46:49], v[70:73], a[80:83], v119, v114 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[46:49], v[74:77], a[84:87], v119, v115 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[46:49], v[78:81], a[88:91], v119, v115 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[62:65], v[82:85], a[76:79], v119, v114 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[62:65], v[86:89], a[80:83], v119, v114 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[62:65], v[90:93], a[84:87], v119, v115 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[62:65], v[94:97], a[88:91], v119, v115 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[34:37], v[172:175], a[0:3], v118, v112 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[140:143], v136 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[34:37], v[176:179], a[4:7], v118, v112 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[144:147], v136 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[34:37], v[180:183], a[8:11], v118, v113 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[148:151], v136 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[34:37], v[184:187], a[12:15], v118, v113 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[152:155], v136 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[50:53], v[188:191], a[0:3], v118, v112 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[156:159], v137 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[50:53], v[192:195], a[4:7], v118, v112 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[160:163], v137 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[50:53], v[196:199], a[8:11], v118, v113 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[164:167], v137 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[50:53], v[200:203], a[12:15], v118, v113 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[168:171], v137 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[38:41], v[172:175], a[16:19], v118, v112 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[38:41], v[176:179], a[20:23], v118, v112 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[38:41], v[180:183], a[24:27], v118, v113 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[38:41], v[184:187], a[92:95], v118, v113 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[54:57], v[188:191], a[16:19], v118, v112 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[54:57], v[192:195], a[20:23], v118, v112 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[54:57], v[196:199], a[24:27], v118, v113 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[54:57], v[200:203], a[92:95], v118, v113 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[42:45], v[172:175], a[96:99], v119, v112 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[42:45], v[176:179], a[100:103], v119, v112 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[42:45], v[180:183], a[104:107], v119, v113 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[42:45], v[184:187], a[108:111], v119, v113 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[58:61], v[188:191], a[96:99], v119, v112 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[58:61], v[192:195], a[100:103], v119, v112 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[58:61], v[196:199], a[104:107], v119, v113 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[58:61], v[200:203], a[108:111], v119, v113 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[46:49], v[172:175], a[112:115], v119, v112 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[46:49], v[176:179], a[116:119], v119, v112 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[46:49], v[180:183], a[120:123], v119, v113 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[46:49], v[184:187], a[124:127], v119, v113 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[62:65], v[188:191], a[112:115], v119, v112 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[62:65], v[192:195], a[116:119], v119, v112 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[62:65], v[196:199], a[120:123], v119, v113 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[62:65], v[200:203], a[124:127], v119, v113 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[140:143], v[66:69], a[128:131],  v110, v114 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[2:5], v125 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[140:143], v[70:73], a[132:135],  v110, v114 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[6:9], v125 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[140:143], v[74:77], a[136:139],  v110, v115 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[10:13], v125 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[140:143], v[78:81], a[140:143],  v110, v115 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[14:17], v125 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[156:159], v[82:85], a[128:131],  v110, v114 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[18:21], v126 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[156:159], v[86:89], a[132:135],  v110, v114 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[22:25], v126 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[156:159], v[90:93], a[136:139],  v110, v115 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[26:29], v126 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[156:159], v[94:97], a[140:143],  v110, v115 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v126 offset:6144

	;;#ASMEND
	buffer_load_dwordx4 v98, s[16:19], s77 offen lds
	s_mov_b32 m0, s36
	s_add_i32 s74, s73, 0x200
	buffer_load_dwordx4 v99, s[16:19], s77 offen lds
	s_mov_b32 m0, s44
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[144:147], v[66:69], a[144:147],  v110, v114 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[144:147], v[70:73], a[148:151],  v110, v114 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[144:147], v[74:77], a[152:155],  v110, v115 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[144:147], v[78:81], a[156:159],  v110, v115 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[160:163], v[82:85], a[144:147],  v110, v114 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[160:163], v[86:89], a[148:151],  v110, v114 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[160:163], v[90:93], a[152:155],  v110, v115 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[160:163], v[94:97], a[156:159],  v110, v115 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_i32 s78, s37, 0xffffe800
	buffer_load_dwordx4 v122, s[16:19], s77 offen lds
	s_mov_b32 m0, s45
	s_nop 0
	buffer_load_dwordx4 v100, s[16:19], s77 offen lds
	s_mov_b32 m0, s35
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[148:151], v[66:69], a[160:163],  v111, v114 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[148:151], v[70:73], a[164:167],  v111, v114 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[148:151], v[74:77], a[168:171], v111, v115 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[148:151], v[78:81], a[172:175], v111, v115 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[164:167], v[82:85], a[160:163],  v111, v114 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[164:167], v[86:89], a[164:167],  v111, v114 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[164:167], v[90:93], a[168:171], v111, v115 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[164:167], v[94:97], a[172:175], v111, v115 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_i32 s77, s70, 0x280
	buffer_load_dwordx4 v98, s[16:19], s76 offen lds
	s_mov_b32 m0, s47
	s_nop 0
	buffer_load_dwordx4 v99, s[16:19], s76 offen lds
	s_mov_b32 m0, s48
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[152:155], v[66:69], a[176:179], v111, v114 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[152:155], v[70:73], a[180:183], v111, v114 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[152:155], v[74:77], a[184:187], v111, v115 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[152:155], v[78:81], a[188:191], v111, v115 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[168:171], v[82:85], a[176:179], v111, v114 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[168:171], v[86:89], a[180:183], v111, v114 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[168:171], v[90:93], a[184:187], v111, v115 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[168:171], v[94:97], a[188:191], v111, v115 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v122, s[16:19], s76 offen lds
	s_mov_b32 m0, s49
	s_nop 0
	buffer_load_dwordx4 v100, s[16:19], s76 offen lds
	s_mov_b32 m0, s39
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[140:143], v[172:175], a[192:195],  v110, v112 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v129 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[140:143], v[176:179], a[196:199],  v110, v112 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v129 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[140:143], v[180:183], a[200:203],  v110, v113 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v129 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[140:143], v[184:187], a[204:207],  v110, v113 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v129 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[156:159], v[188:191], a[192:195],  v110, v112 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v130 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[156:159], v[192:195], a[196:199],  v110, v112 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v130 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[156:159], v[196:199], a[200:203],  v110, v113 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v130 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[156:159], v[200:203], a[204:207],  v110, v113 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[62:65], v130 offset:6144

	;;#ASMEND
	s_add_i32 s76, s71, 0x280
	buffer_load_dwordx4 v102, s[20:23], s75 offen lds
	s_mov_b32 m0, s50
	s_nop 0
	buffer_load_dwordx4 v101, s[20:23], s75 offen lds
	s_mov_b32 m0, s51
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[144:147], v[172:175], a[208:211],  v110, v112 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[144:147], v[176:179], a[212:215],  v110, v112 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[144:147], v[180:183], a[216:219],  v110, v113 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[144:147], v[184:187], a[220:223],  v110, v113 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[160:163], v[188:191], a[208:211],  v110, v112 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[160:163], v[192:195], a[212:215],  v110, v112 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[160:163], v[196:199], a[216:219],  v110, v113 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[160:163], v[200:203], a[220:223],  v110, v113 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v103, s[20:23], s75 offen lds
	s_mov_b32 m0, s52
	s_nop 0
	buffer_load_dwordx4 v104, s[20:23], s75 offen lds
	s_mov_b32 m0, s41
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[148:151], v[172:175], a[224:227],  v111, v112 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[148:151], v[176:179], a[228:231],  v111, v112 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[148:151], v[180:183], a[232:235], v111, v113 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[148:151], v[184:187], a[236:239], v111, v113 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[164:167], v[188:191], a[224:227],  v111, v112 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[164:167], v[192:195], a[228:231],  v111, v112 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[164:167], v[196:199], a[232:235], v111, v113 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[164:167], v[200:203], a[236:239], v111, v113 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_i32 s75, s72, 0x280
	buffer_load_dwordx4 v102, s[20:23], s74 offen lds
	s_mov_b32 m0, s54
	s_nop 0
	buffer_load_dwordx4 v101, s[20:23], s74 offen lds
	s_mov_b32 m0, s55
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[152:155], v[172:175], a[240:243], v111, v112 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[152:155], v[176:179], a[244:247], v111, v112 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[152:155], v[180:183], a[248:251], v111, v113 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[152:155], v[184:187], a[252:255], v111, v113 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[168:171], v[188:191], a[240:243], v111, v112 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[168:171], v[192:195], a[244:247], v111, v112 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[168:171], v[196:199], a[248:251], v111, v113 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[168:171], v[200:203], a[252:255], v111, v113 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v103, s[20:23], s74 offen lds
	s_mov_b32 m0, s56
	s_nop 0
	buffer_load_dwordx4 v104, s[20:23], s74 offen lds
	s_mov_b32 m0, s33
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[78:79], v105, s[0:3], s78 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[66:67], v105, s[4:7], s78 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[74:75], v105, s[8:11], s78 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[70:71], v105, s[12:15], s78 offen
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[2:5], v[34:37], a[28:31],  v120, v116 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[152:155], v133 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[2:5], v[38:41], a[32:35],  v120, v116 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[156:159], v133 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[2:5], v[42:45], a[36:39],  v120, v117 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[160:163], v133 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[2:5], v[46:49], a[40:43],  v120, v117 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[164:167], v133 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[18:21], v[50:53], a[28:31],  v120, v116 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[168:171], v135 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[18:21], v[54:57], a[32:35],  v120, v116 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[172:175], v135 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[18:21], v[58:61], a[36:39],  v120, v117 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[176:179], v135 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[18:21], v[62:65], a[40:43],  v120, v117 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[180:183], v135 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[6:9], v[34:37], a[44:47],  v120, v116 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[6:9], v[38:41], a[48:51],  v120, v116 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[6:9], v[42:45], a[52:55],  v120, v117 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[6:9], v[46:49], a[56:59],  v120, v117 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[22:25], v[50:53], a[44:47],  v120, v116 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[22:25], v[54:57], a[48:51],  v120, v116 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[22:25], v[58:61], a[52:55],  v120, v117 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[22:25], v[62:65], a[56:59],  v120, v117 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[10:13], v[34:37], a[60:63],  v121, v116 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[10:13], v[38:41], a[64:67],  v121, v116 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[10:13], v[42:45], a[68:71], v121, v117 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[10:13], v[46:49], a[72:75], v121, v117 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[26:29], v[50:53], a[60:63],  v121, v116 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[26:29], v[54:57], a[64:67],  v121, v116 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[26:29], v[58:61], a[68:71], v121, v117 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[26:29], v[62:65], a[72:75], v121, v117 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[14:17], v[34:37], a[76:79], v121, v116 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[14:17], v[38:41], a[80:83], v121, v116 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[14:17], v[42:45], a[84:87], v121, v117 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[14:17], v[46:49], a[88:91], v121, v117 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[30:33], v[50:53], a[76:79], v121, v116 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[30:33], v[54:57], a[80:83], v121, v116 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[30:33], v[58:61], a[84:87], v121, v117 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[30:33], v[62:65], a[88:91], v121, v117 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[2:5], v[152:155], a[0:3], v120, v108 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[80:83], v138 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[2:5], v[156:159], a[4:7], v120, v108 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[84:87], v138 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[2:5], v[160:163], a[8:11], v120, v109 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[88:91], v138 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[2:5], v[164:167], a[12:15], v120, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[92:95], v138 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[18:21], v[168:171], a[0:3], v120, v108 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[110:113], v139 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[18:21], v[172:175], a[4:7], v120, v108 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[140:143], v139 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[18:21], v[176:179], a[8:11], v120, v109 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[144:147], v139 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[18:21], v[180:183], a[12:15], v120, v109 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[148:151], v139 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[6:9], v[152:155], a[16:19], v120, v108 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[6:9], v[156:159], a[20:23], v120, v108 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[6:9], v[160:163], a[24:27], v120, v109 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[6:9], v[164:167], a[92:95], v120, v109 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[22:25], v[168:171], a[16:19], v120, v108 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[22:25], v[172:175], a[20:23], v120, v108 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[22:25], v[176:179], a[24:27], v120, v109 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[22:25], v[180:183], a[92:95], v120, v109 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[10:13], v[152:155], a[96:99], v121, v108 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[10:13], v[156:159], a[100:103], v121, v108 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[10:13], v[160:163], a[104:107], v121, v109 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[10:13], v[164:167], a[108:111], v121, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[26:29], v[168:171], a[96:99], v121, v108 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[26:29], v[172:175], a[100:103], v121, v108 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[26:29], v[176:179], a[104:107], v121, v109 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[26:29], v[180:183], a[108:111], v121, v109 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[14:17], v[152:155], a[112:115], v121, v108 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[14:17], v[156:159], a[116:119], v121, v108 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[14:17], v[160:163], a[120:123], v121, v109 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[14:17], v[164:167], a[124:127], v121, v109 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[30:33], v[168:171], a[112:115], v121, v108 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[30:33], v[172:175], a[116:119], v121, v108 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[30:33], v[176:179], a[120:123], v121, v109 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[30:33], v[180:183], a[124:127], v121, v109 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[80:83], v[34:37], a[128:131],  v106, v116 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[2:5], v123 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[80:83], v[38:41], a[132:135],  v106, v116 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[6:9], v123 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[80:83], v[42:45], a[136:139],  v106, v117 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[10:13], v123 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[80:83], v[46:49], a[140:143],  v106, v117 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[14:17], v123 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[110:113], v[50:53], a[128:131],  v106, v116 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[18:21], v124 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[110:113], v[54:57], a[132:135],  v106, v116 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[22:25], v124 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[110:113], v[58:61], a[136:139],  v106, v117 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[26:29], v124 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[110:113], v[62:65], a[140:143],  v106, v117 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v124 offset:6144

	;;#ASMEND
	buffer_load_dwordx4 v98, s[16:19], s77 offen lds
	s_mov_b32 m0, s57
	s_add_i32 s74, s73, 0x280
	buffer_load_dwordx4 v99, s[16:19], s77 offen lds
	s_mov_b32 m0, s58
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[84:87], v[34:37], a[144:147],  v106, v116 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[84:87], v[38:41], a[148:151],  v106, v116 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[84:87], v[42:45], a[152:155],  v106, v117 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[84:87], v[46:49], a[156:159],  v106, v117 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[140:143], v[50:53], a[144:147],  v106, v116 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[140:143], v[54:57], a[148:151],  v106, v116 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[140:143], v[58:61], a[152:155],  v106, v117 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[140:143], v[62:65], a[156:159],  v106, v117 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_i32 s78, s37, 0xffffea00
	buffer_load_dwordx4 v122, s[16:19], s77 offen lds
	s_mov_b32 m0, s59
	s_nop 0
	buffer_load_dwordx4 v100, s[16:19], s77 offen lds
	s_mov_b32 m0, s38
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[88:91], v[34:37], a[160:163],  v107, v116 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[88:91], v[38:41], a[164:167],  v107, v116 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[88:91], v[42:45], a[168:171], v107, v117 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[88:91], v[46:49], a[172:175], v107, v117 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[144:147], v[50:53], a[160:163],  v107, v116 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[144:147], v[54:57], a[164:167],  v107, v116 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[144:147], v[58:61], a[168:171], v107, v117 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[144:147], v[62:65], a[172:175], v107, v117 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_i32 s77, s70, 0x300
	buffer_load_dwordx4 v98, s[16:19], s76 offen lds
	s_mov_b32 m0, s60
	s_nop 0
	buffer_load_dwordx4 v99, s[16:19], s76 offen lds
	s_mov_b32 m0, s61
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[92:95], v[34:37], a[176:179], v107, v116 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[92:95], v[38:41], a[180:183], v107, v116 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[92:95], v[42:45], a[184:187], v107, v117 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[92:95], v[46:49], a[188:191], v107, v117 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[148:151], v[50:53], a[176:179], v107, v116 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[148:151], v[54:57], a[180:183], v107, v116 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[148:151], v[58:61], a[184:187], v107, v117 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[148:151], v[62:65], a[188:191], v107, v117 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v122, s[16:19], s76 offen lds
	s_mov_b32 m0, s62
	s_nop 0
	buffer_load_dwordx4 v100, s[16:19], s76 offen lds
	s_mov_b32 m0, s40
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[80:83], v[152:155], a[192:195],  v106, v108 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v127 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[80:83], v[156:159], a[196:199],  v106, v108 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v127 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[80:83], v[160:163], a[200:203],  v106, v109 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v127 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[80:83], v[164:167], a[204:207],  v106, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v127 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[110:113], v[168:171], a[192:195],  v106, v108 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v128 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[110:113], v[172:175], a[196:199],  v106, v108 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v128 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[110:113], v[176:179], a[200:203],  v106, v109 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v128 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[110:113], v[180:183], a[204:207],  v106, v109 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[62:65], v128 offset:6144

	;;#ASMEND
	s_add_i32 s76, s71, 0x300
	buffer_load_dwordx4 v102, s[20:23], s75 offen lds
	s_mov_b32 m0, s63
	s_nop 0
	buffer_load_dwordx4 v101, s[20:23], s75 offen lds
	s_mov_b32 m0, s64
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[84:87], v[152:155], a[208:211],  v106, v108 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[84:87], v[156:159], a[212:215],  v106, v108 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[84:87], v[160:163], a[216:219],  v106, v109 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[84:87], v[164:167], a[220:223],  v106, v109 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[140:143], v[168:171], a[208:211],  v106, v108 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[140:143], v[172:175], a[212:215],  v106, v108 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[140:143], v[176:179], a[216:219],  v106, v109 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[140:143], v[180:183], a[220:223],  v106, v109 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v103, s[20:23], s75 offen lds
	s_mov_b32 m0, s65
	s_nop 0
	buffer_load_dwordx4 v104, s[20:23], s75 offen lds
	s_mov_b32 m0, s42
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[88:91], v[152:155], a[224:227],  v107, v108 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[88:91], v[156:159], a[228:231],  v107, v108 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[88:91], v[160:163], a[232:235], v107, v109 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[88:91], v[164:167], a[236:239], v107, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[144:147], v[168:171], a[224:227],  v107, v108 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[144:147], v[172:175], a[228:231],  v107, v108 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[144:147], v[176:179], a[232:235], v107, v109 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[144:147], v[180:183], a[236:239], v107, v109 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_i32 s75, s72, 0x300
	buffer_load_dwordx4 v102, s[20:23], s74 offen lds
	s_mov_b32 m0, s66
	s_nop 0
	buffer_load_dwordx4 v101, s[20:23], s74 offen lds
	s_mov_b32 m0, s67
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[92:95], v[152:155], a[240:243], v107, v108 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[92:95], v[156:159], a[244:247], v107, v108 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[92:95], v[160:163], a[248:251], v107, v109 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[92:95], v[164:167], a[252:255], v107, v109 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[148:151], v[168:171], a[240:243], v107, v108 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[148:151], v[172:175], a[244:247], v107, v108 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[148:151], v[176:179], a[248:251], v107, v109 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[148:151], v[180:183], a[252:255], v107, v109 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v103, s[20:23], s74 offen lds
	s_mov_b32 m0, s68
	s_nop 0
	buffer_load_dwordx4 v104, s[20:23], s74 offen lds
	s_mov_b32 m0, s31
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[80:81], v105, s[0:3], s78 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[68:69], v105, s[4:7], s78 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[76:77], v105, s[8:11], s78 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[72:73], v105, s[12:15], s78 offen
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[2:5], v[34:37], a[28:31],  v78, v74 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[140:143], v131 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[2:5], v[38:41], a[32:35],  v78, v74 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[144:147], v131 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[2:5], v[42:45], a[36:39],  v78, v75 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[148:151], v131 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[2:5], v[46:49], a[40:43],  v78, v75 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[152:155], v131 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[18:21], v[50:53], a[28:31],  v78, v74 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[156:159], v132 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[18:21], v[54:57], a[32:35],  v78, v74 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[160:163], v132 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[18:21], v[58:61], a[36:39],  v78, v75 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[164:167], v132 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[18:21], v[62:65], a[40:43],  v78, v75 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[168:171], v132 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[6:9], v[34:37], a[44:47],  v78, v74 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[6:9], v[38:41], a[48:51],  v78, v74 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[6:9], v[42:45], a[52:55],  v78, v75 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[6:9], v[46:49], a[56:59],  v78, v75 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[22:25], v[50:53], a[44:47],  v78, v74 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[22:25], v[54:57], a[48:51],  v78, v74 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[22:25], v[58:61], a[52:55],  v78, v75 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[22:25], v[62:65], a[56:59],  v78, v75 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[10:13], v[34:37], a[60:63],  v79, v74 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[10:13], v[38:41], a[64:67],  v79, v74 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[10:13], v[42:45], a[68:71], v79, v75 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[10:13], v[46:49], a[72:75], v79, v75 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[26:29], v[50:53], a[60:63],  v79, v74 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[26:29], v[54:57], a[64:67],  v79, v74 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[26:29], v[58:61], a[68:71], v79, v75 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[26:29], v[62:65], a[72:75], v79, v75 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[14:17], v[34:37], a[76:79], v79, v74 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[14:17], v[38:41], a[80:83], v79, v74 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[14:17], v[42:45], a[84:87], v79, v75 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[14:17], v[46:49], a[88:91], v79, v75 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[30:33], v[50:53], a[76:79], v79, v74 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[30:33], v[54:57], a[80:83], v79, v74 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[30:33], v[58:61], a[84:87], v79, v75 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[30:33], v[62:65], a[88:91], v79, v75 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[2:5], v[140:143], a[0:3], v78, v70 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[82:85], v136 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[2:5], v[144:147], a[4:7], v78, v70 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[86:89], v136 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[2:5], v[148:151], a[8:11], v78, v71 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[90:93], v136 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[2:5], v[152:155], a[12:15], v78, v71 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[94:97], v136 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[18:21], v[156:159], a[0:3], v78, v70 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[106:109], v137 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[18:21], v[160:163], a[4:7], v78, v70 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[110:113], v137 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[18:21], v[164:167], a[8:11], v78, v71 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[114:117], v137 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[18:21], v[168:171], a[12:15], v78, v71 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[118:121], v137 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[6:9], v[140:143], a[16:19], v78, v70 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[6:9], v[144:147], a[20:23], v78, v70 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[6:9], v[148:151], a[24:27], v78, v71 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[6:9], v[152:155], a[92:95], v78, v71 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[22:25], v[156:159], a[16:19], v78, v70 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[22:25], v[160:163], a[20:23], v78, v70 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[22:25], v[164:167], a[24:27], v78, v71 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[22:25], v[168:171], a[92:95], v78, v71 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[10:13], v[140:143], a[96:99], v79, v70 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[10:13], v[144:147], a[100:103], v79, v70 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[10:13], v[148:151], a[104:107], v79, v71 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[10:13], v[152:155], a[108:111], v79, v71 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[26:29], v[156:159], a[96:99], v79, v70 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[26:29], v[160:163], a[100:103], v79, v70 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[26:29], v[164:167], a[104:107], v79, v71 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[26:29], v[168:171], a[108:111], v79, v71 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[14:17], v[140:143], a[112:115], v79, v70 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[14:17], v[144:147], a[116:119], v79, v70 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[14:17], v[148:151], a[120:123], v79, v71 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[14:17], v[152:155], a[124:127], v79, v71 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[30:33], v[156:159], a[112:115], v79, v70 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[30:33], v[160:163], a[116:119], v79, v70 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[30:33], v[164:167], a[120:123], v79, v71 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[30:33], v[168:171], a[124:127], v79, v71 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[82:85], v[34:37], a[128:131],  v66, v74 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[2:5], v125 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[82:85], v[38:41], a[132:135],  v66, v74 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[6:9], v125 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[82:85], v[42:45], a[136:139],  v66, v75 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[10:13], v125 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[82:85], v[46:49], a[140:143],  v66, v75 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[14:17], v125 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[106:109], v[50:53], a[128:131],  v66, v74 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[18:21], v126 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[106:109], v[54:57], a[132:135],  v66, v74 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[22:25], v126 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[106:109], v[58:61], a[136:139],  v66, v75 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[26:29], v126 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[106:109], v[62:65], a[140:143],  v66, v75 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v126 offset:6144

	;;#ASMEND
	buffer_load_dwordx4 v98, s[16:19], s77 offen lds
	s_mov_b32 m0, s36
	s_add_i32 s74, s73, 0x300
	buffer_load_dwordx4 v99, s[16:19], s77 offen lds
	s_mov_b32 m0, s44
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[86:89], v[34:37], a[144:147],  v66, v74 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[86:89], v[38:41], a[148:151],  v66, v74 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[86:89], v[42:45], a[152:155],  v66, v75 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[86:89], v[46:49], a[156:159],  v66, v75 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[110:113], v[50:53], a[144:147],  v66, v74 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[110:113], v[54:57], a[148:151],  v66, v74 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[110:113], v[58:61], a[152:155],  v66, v75 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[110:113], v[62:65], a[156:159],  v66, v75 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_i32 s78, s37, 0xffffec00
	buffer_load_dwordx4 v122, s[16:19], s77 offen lds
	s_mov_b32 m0, s45
	s_nop 0
	buffer_load_dwordx4 v100, s[16:19], s77 offen lds
	s_mov_b32 m0, s35
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[90:93], v[34:37], a[160:163],  v67, v74 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[90:93], v[38:41], a[164:167],  v67, v74 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[90:93], v[42:45], a[168:171], v67, v75 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[90:93], v[46:49], a[172:175], v67, v75 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[114:117], v[50:53], a[160:163],  v67, v74 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[114:117], v[54:57], a[164:167],  v67, v74 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[114:117], v[58:61], a[168:171], v67, v75 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[114:117], v[62:65], a[172:175], v67, v75 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_i32 s77, s70, 0x380
	buffer_load_dwordx4 v98, s[16:19], s76 offen lds
	s_mov_b32 m0, s47
	s_nop 0
	buffer_load_dwordx4 v99, s[16:19], s76 offen lds
	s_mov_b32 m0, s48
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[94:97], v[34:37], a[176:179], v67, v74 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[94:97], v[38:41], a[180:183], v67, v74 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[94:97], v[42:45], a[184:187], v67, v75 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[94:97], v[46:49], a[188:191], v67, v75 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[118:121], v[50:53], a[176:179], v67, v74 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[118:121], v[54:57], a[180:183], v67, v74 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[118:121], v[58:61], a[184:187], v67, v75 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[118:121], v[62:65], a[188:191], v67, v75 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v122, s[16:19], s76 offen lds
	s_mov_b32 m0, s49
	s_nop 0
	buffer_load_dwordx4 v100, s[16:19], s76 offen lds
	s_mov_b32 m0, s39
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[82:85], v[140:143], a[192:195],  v66, v70 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v129 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[82:85], v[144:147], a[196:199],  v66, v70 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v129 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[82:85], v[148:151], a[200:203],  v66, v71 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v129 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[82:85], v[152:155], a[204:207],  v66, v71 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v129 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[106:109], v[156:159], a[192:195],  v66, v70 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v130 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[106:109], v[160:163], a[196:199],  v66, v70 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v130 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[106:109], v[164:167], a[200:203],  v66, v71 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v130 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[106:109], v[168:171], a[204:207],  v66, v71 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[62:65], v130 offset:6144

	;;#ASMEND
	s_add_i32 s76, s71, 0x380
	buffer_load_dwordx4 v102, s[20:23], s75 offen lds
	s_mov_b32 m0, s50
	s_nop 0
	buffer_load_dwordx4 v101, s[20:23], s75 offen lds
	s_mov_b32 m0, s51
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[86:89], v[140:143], a[208:211],  v66, v70 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[86:89], v[144:147], a[212:215],  v66, v70 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[86:89], v[148:151], a[216:219],  v66, v71 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[86:89], v[152:155], a[220:223],  v66, v71 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[110:113], v[156:159], a[208:211],  v66, v70 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[110:113], v[160:163], a[212:215],  v66, v70 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[110:113], v[164:167], a[216:219],  v66, v71 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[110:113], v[168:171], a[220:223],  v66, v71 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v103, s[20:23], s75 offen lds
	s_mov_b32 m0, s52
	s_nop 0
	buffer_load_dwordx4 v104, s[20:23], s75 offen lds
	s_mov_b32 m0, s41
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[90:93], v[140:143], a[224:227],  v67, v70 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[90:93], v[144:147], a[228:231],  v67, v70 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[90:93], v[148:151], a[232:235], v67, v71 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[90:93], v[152:155], a[236:239], v67, v71 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[114:117], v[156:159], a[224:227],  v67, v70 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[114:117], v[160:163], a[228:231],  v67, v70 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[114:117], v[164:167], a[232:235], v67, v71 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[114:117], v[168:171], a[236:239], v67, v71 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_i32 s75, s72, 0x380
	buffer_load_dwordx4 v102, s[20:23], s74 offen lds
	s_mov_b32 m0, s54
	s_nop 0
	buffer_load_dwordx4 v101, s[20:23], s74 offen lds
	s_mov_b32 m0, s55
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[94:97], v[140:143], a[240:243], v67, v70 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[94:97], v[144:147], a[244:247], v67, v70 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[94:97], v[148:151], a[248:251], v67, v71 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[94:97], v[152:155], a[252:255], v67, v71 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[118:121], v[156:159], a[240:243], v67, v70 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[118:121], v[160:163], a[244:247], v67, v70 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[118:121], v[164:167], a[248:251], v67, v71 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[118:121], v[168:171], a[252:255], v67, v71 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v103, s[20:23], s74 offen lds
	s_mov_b32 m0, s56
	s_nop 0
	buffer_load_dwordx4 v104, s[20:23], s74 offen lds
	s_mov_b32 m0, s33
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[78:79], v105, s[0:3], s78 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[66:67], v105, s[4:7], s78 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[74:75], v105, s[8:11], s78 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[70:71], v105, s[12:15], s78 offen
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[2:5], v[34:37], a[28:31],  v80, v76 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[140:143], v133 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[2:5], v[38:41], a[32:35],  v80, v76 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[144:147], v133 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[2:5], v[42:45], a[36:39],  v80, v77 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[148:151], v133 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[2:5], v[46:49], a[40:43],  v80, v77 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[152:155], v133 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[18:21], v[50:53], a[28:31],  v80, v76 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[156:159], v135 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[18:21], v[54:57], a[32:35],  v80, v76 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[160:163], v135 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[18:21], v[58:61], a[36:39],  v80, v77 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[164:167], v135 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[18:21], v[62:65], a[40:43],  v80, v77 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[168:171], v135 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[6:9], v[34:37], a[44:47],  v80, v76 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[6:9], v[38:41], a[48:51],  v80, v76 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[6:9], v[42:45], a[52:55],  v80, v77 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[6:9], v[46:49], a[56:59],  v80, v77 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[22:25], v[50:53], a[44:47],  v80, v76 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[22:25], v[54:57], a[48:51],  v80, v76 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[22:25], v[58:61], a[52:55],  v80, v77 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[22:25], v[62:65], a[56:59],  v80, v77 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[10:13], v[34:37], a[60:63],  v81, v76 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[10:13], v[38:41], a[64:67],  v81, v76 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[10:13], v[42:45], a[68:71], v81, v77 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[10:13], v[46:49], a[72:75], v81, v77 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[26:29], v[50:53], a[60:63],  v81, v76 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[26:29], v[54:57], a[64:67],  v81, v76 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[26:29], v[58:61], a[68:71], v81, v77 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[26:29], v[62:65], a[72:75], v81, v77 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[14:17], v[34:37], a[76:79], v81, v76 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[14:17], v[38:41], a[80:83], v81, v76 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[14:17], v[42:45], a[84:87], v81, v77 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[14:17], v[46:49], a[88:91], v81, v77 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[30:33], v[50:53], a[76:79], v81, v76 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[30:33], v[54:57], a[80:83], v81, v76 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[30:33], v[58:61], a[84:87], v81, v77 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[30:33], v[62:65], a[88:91], v81, v77 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[2:5], v[140:143], a[0:3], v80, v72 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[82:85], v138 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[2:5], v[144:147], a[4:7], v80, v72 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[86:89], v138 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[2:5], v[148:151], a[8:11], v80, v73 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[90:93], v138 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[2:5], v[152:155], a[12:15], v80, v73 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[94:97], v138 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[18:21], v[156:159], a[0:3], v80, v72 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[106:109], v139 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[18:21], v[160:163], a[4:7], v80, v72 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[110:113], v139 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[18:21], v[164:167], a[8:11], v80, v73 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[114:117], v139 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[18:21], v[168:171], a[12:15], v80, v73 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[118:121], v139 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[6:9], v[140:143], a[16:19], v80, v72 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[6:9], v[144:147], a[20:23], v80, v72 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[6:9], v[148:151], a[24:27], v80, v73 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[6:9], v[152:155], a[92:95], v80, v73 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[22:25], v[156:159], a[16:19], v80, v72 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[22:25], v[160:163], a[20:23], v80, v72 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[22:25], v[164:167], a[24:27], v80, v73 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[22:25], v[168:171], a[92:95], v80, v73 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[10:13], v[140:143], a[96:99], v81, v72 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[10:13], v[144:147], a[100:103], v81, v72 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[10:13], v[148:151], a[104:107], v81, v73 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[10:13], v[152:155], a[108:111], v81, v73 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[26:29], v[156:159], a[96:99], v81, v72 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[26:29], v[160:163], a[100:103], v81, v72 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[26:29], v[164:167], a[104:107], v81, v73 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[26:29], v[168:171], a[108:111], v81, v73 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[14:17], v[140:143], a[112:115], v81, v72 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[14:17], v[144:147], a[116:119], v81, v72 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[14:17], v[148:151], a[120:123], v81, v73 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[14:17], v[152:155], a[124:127], v81, v73 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[30:33], v[156:159], a[112:115], v81, v72 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[30:33], v[160:163], a[116:119], v81, v72 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[30:33], v[164:167], a[120:123], v81, v73 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[30:33], v[168:171], a[124:127], v81, v73 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[82:85], v[34:37], a[128:131],  v68, v76 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[2:5], v123 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[82:85], v[38:41], a[132:135],  v68, v76 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[6:9], v123 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[82:85], v[42:45], a[136:139],  v68, v77 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[10:13], v123 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[82:85], v[46:49], a[140:143],  v68, v77 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[14:17], v123 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[106:109], v[50:53], a[128:131],  v68, v76 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[18:21], v124 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[106:109], v[54:57], a[132:135],  v68, v76 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[22:25], v124 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[106:109], v[58:61], a[136:139],  v68, v77 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[26:29], v124 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[106:109], v[62:65], a[140:143],  v68, v77 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v124 offset:6144

	;;#ASMEND
	buffer_load_dwordx4 v98, s[16:19], s77 offen lds
	s_mov_b32 m0, s57
	s_add_i32 s74, s73, 0x380
	buffer_load_dwordx4 v99, s[16:19], s77 offen lds
	s_mov_b32 m0, s58
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[86:89], v[34:37], a[144:147],  v68, v76 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[86:89], v[38:41], a[148:151],  v68, v76 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[86:89], v[42:45], a[152:155],  v68, v77 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[86:89], v[46:49], a[156:159],  v68, v77 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[110:113], v[50:53], a[144:147],  v68, v76 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[110:113], v[54:57], a[148:151],  v68, v76 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[110:113], v[58:61], a[152:155],  v68, v77 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[110:113], v[62:65], a[156:159],  v68, v77 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_i32 s78, s37, 0xffffee00
	buffer_load_dwordx4 v122, s[16:19], s77 offen lds
	s_mov_b32 m0, s59
	s_nop 0
	buffer_load_dwordx4 v100, s[16:19], s77 offen lds
	s_mov_b32 m0, s38
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[90:93], v[34:37], a[160:163],  v69, v76 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[90:93], v[38:41], a[164:167],  v69, v76 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[90:93], v[42:45], a[168:171], v69, v77 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[90:93], v[46:49], a[172:175], v69, v77 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[114:117], v[50:53], a[160:163],  v69, v76 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[114:117], v[54:57], a[164:167],  v69, v76 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[114:117], v[58:61], a[168:171], v69, v77 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[114:117], v[62:65], a[172:175], v69, v77 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_i32 s77, s70, 0x400
	buffer_load_dwordx4 v98, s[16:19], s76 offen lds
	s_mov_b32 m0, s60
	s_nop 0
	buffer_load_dwordx4 v99, s[16:19], s76 offen lds
	s_mov_b32 m0, s61
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[94:97], v[34:37], a[176:179], v69, v76 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[94:97], v[38:41], a[180:183], v69, v76 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[94:97], v[42:45], a[184:187], v69, v77 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[94:97], v[46:49], a[188:191], v69, v77 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[118:121], v[50:53], a[176:179], v69, v76 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[118:121], v[54:57], a[180:183], v69, v76 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[118:121], v[58:61], a[184:187], v69, v77 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[118:121], v[62:65], a[188:191], v69, v77 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v122, s[16:19], s76 offen lds
	s_mov_b32 m0, s62
	s_nop 0
	buffer_load_dwordx4 v100, s[16:19], s76 offen lds
	s_mov_b32 m0, s40
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[82:85], v[140:143], a[192:195],  v68, v72 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v127 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[82:85], v[144:147], a[196:199],  v68, v72 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v127 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[82:85], v[148:151], a[200:203],  v68, v73 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v127 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[82:85], v[152:155], a[204:207],  v68, v73 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v127 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[106:109], v[156:159], a[192:195],  v68, v72 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v128 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[106:109], v[160:163], a[196:199],  v68, v72 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v128 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[106:109], v[164:167], a[200:203],  v68, v73 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v128 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[106:109], v[168:171], a[204:207],  v68, v73 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[62:65], v128 offset:6144

	;;#ASMEND
	s_add_i32 s76, s71, 0x400
	buffer_load_dwordx4 v102, s[20:23], s75 offen lds
	s_mov_b32 m0, s63
	s_nop 0
	buffer_load_dwordx4 v101, s[20:23], s75 offen lds
	s_mov_b32 m0, s64
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[86:89], v[140:143], a[208:211],  v68, v72 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[86:89], v[144:147], a[212:215],  v68, v72 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[86:89], v[148:151], a[216:219],  v68, v73 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[86:89], v[152:155], a[220:223],  v68, v73 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[110:113], v[156:159], a[208:211],  v68, v72 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[110:113], v[160:163], a[212:215],  v68, v72 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[110:113], v[164:167], a[216:219],  v68, v73 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[110:113], v[168:171], a[220:223],  v68, v73 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v103, s[20:23], s75 offen lds
	s_mov_b32 m0, s65
	s_nop 0
	buffer_load_dwordx4 v104, s[20:23], s75 offen lds
	s_mov_b32 m0, s42
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[90:93], v[140:143], a[224:227],  v69, v72 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[90:93], v[144:147], a[228:231],  v69, v72 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[90:93], v[148:151], a[232:235], v69, v73 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[90:93], v[152:155], a[236:239], v69, v73 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[114:117], v[156:159], a[224:227],  v69, v72 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[114:117], v[160:163], a[228:231],  v69, v72 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[114:117], v[164:167], a[232:235], v69, v73 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[114:117], v[168:171], a[236:239], v69, v73 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_i32 s75, s72, 0x400
	buffer_load_dwordx4 v102, s[20:23], s74 offen lds
	s_mov_b32 m0, s66
	s_nop 0
	buffer_load_dwordx4 v101, s[20:23], s74 offen lds
	s_mov_b32 m0, s67
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[94:97], v[140:143], a[240:243], v69, v72 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[94:97], v[144:147], a[244:247], v69, v72 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[94:97], v[148:151], a[248:251], v69, v73 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[94:97], v[152:155], a[252:255], v69, v73 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[118:121], v[156:159], a[240:243], v69, v72 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[118:121], v[160:163], a[244:247], v69, v72 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[118:121], v[164:167], a[248:251], v69, v73 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[118:121], v[168:171], a[252:255], v69, v73 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v103, s[20:23], s74 offen lds
	s_mov_b32 m0, s68
	s_nop 0
	buffer_load_dwordx4 v104, s[20:23], s74 offen lds
	s_mov_b32 m0, s31
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[80:81], v105, s[0:3], s78 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[68:69], v105, s[4:7], s78 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[76:77], v105, s[8:11], s78 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[72:73], v105, s[12:15], s78 offen
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[2:5], v[34:37], a[28:31],  v78, v74 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[140:143], v131 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[2:5], v[38:41], a[32:35],  v78, v74 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[144:147], v131 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[2:5], v[42:45], a[36:39],  v78, v75 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[148:151], v131 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[2:5], v[46:49], a[40:43],  v78, v75 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[152:155], v131 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[18:21], v[50:53], a[28:31],  v78, v74 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[156:159], v132 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[18:21], v[54:57], a[32:35],  v78, v74 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[160:163], v132 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[18:21], v[58:61], a[36:39],  v78, v75 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[164:167], v132 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[18:21], v[62:65], a[40:43],  v78, v75 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[168:171], v132 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[6:9], v[34:37], a[44:47],  v78, v74 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[6:9], v[38:41], a[48:51],  v78, v74 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[6:9], v[42:45], a[52:55],  v78, v75 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[6:9], v[46:49], a[56:59],  v78, v75 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[22:25], v[50:53], a[44:47],  v78, v74 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[22:25], v[54:57], a[48:51],  v78, v74 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[22:25], v[58:61], a[52:55],  v78, v75 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[22:25], v[62:65], a[56:59],  v78, v75 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[10:13], v[34:37], a[60:63],  v79, v74 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[10:13], v[38:41], a[64:67],  v79, v74 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[10:13], v[42:45], a[68:71], v79, v75 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[10:13], v[46:49], a[72:75], v79, v75 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[26:29], v[50:53], a[60:63],  v79, v74 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[26:29], v[54:57], a[64:67],  v79, v74 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[26:29], v[58:61], a[68:71], v79, v75 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[26:29], v[62:65], a[72:75], v79, v75 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[14:17], v[34:37], a[76:79], v79, v74 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[14:17], v[38:41], a[80:83], v79, v74 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[14:17], v[42:45], a[84:87], v79, v75 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[14:17], v[46:49], a[88:91], v79, v75 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[30:33], v[50:53], a[76:79], v79, v74 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[30:33], v[54:57], a[80:83], v79, v74 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[30:33], v[58:61], a[84:87], v79, v75 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[30:33], v[62:65], a[88:91], v79, v75 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[2:5], v[140:143], a[0:3], v78, v70 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[82:85], v136 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[2:5], v[144:147], a[4:7], v78, v70 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[86:89], v136 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[2:5], v[148:151], a[8:11], v78, v71 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[90:93], v136 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[2:5], v[152:155], a[12:15], v78, v71 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[94:97], v136 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[18:21], v[156:159], a[0:3], v78, v70 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[106:109], v137 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[18:21], v[160:163], a[4:7], v78, v70 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[110:113], v137 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[18:21], v[164:167], a[8:11], v78, v71 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[114:117], v137 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[18:21], v[168:171], a[12:15], v78, v71 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[118:121], v137 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[6:9], v[140:143], a[16:19], v78, v70 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[6:9], v[144:147], a[20:23], v78, v70 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[6:9], v[148:151], a[24:27], v78, v71 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[6:9], v[152:155], a[92:95], v78, v71 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[22:25], v[156:159], a[16:19], v78, v70 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[22:25], v[160:163], a[20:23], v78, v70 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[22:25], v[164:167], a[24:27], v78, v71 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[22:25], v[168:171], a[92:95], v78, v71 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[10:13], v[140:143], a[96:99], v79, v70 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[10:13], v[144:147], a[100:103], v79, v70 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[10:13], v[148:151], a[104:107], v79, v71 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[10:13], v[152:155], a[108:111], v79, v71 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[26:29], v[156:159], a[96:99], v79, v70 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[26:29], v[160:163], a[100:103], v79, v70 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[26:29], v[164:167], a[104:107], v79, v71 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[26:29], v[168:171], a[108:111], v79, v71 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[14:17], v[140:143], a[112:115], v79, v70 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[14:17], v[144:147], a[116:119], v79, v70 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[14:17], v[148:151], a[120:123], v79, v71 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[14:17], v[152:155], a[124:127], v79, v71 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[30:33], v[156:159], a[112:115], v79, v70 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[30:33], v[160:163], a[116:119], v79, v70 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[30:33], v[164:167], a[120:123], v79, v71 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[30:33], v[168:171], a[124:127], v79, v71 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[82:85], v[34:37], a[128:131],  v66, v74 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[2:5], v125 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[82:85], v[38:41], a[132:135],  v66, v74 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[6:9], v125 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[82:85], v[42:45], a[136:139],  v66, v75 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[10:13], v125 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[82:85], v[46:49], a[140:143],  v66, v75 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[14:17], v125 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[106:109], v[50:53], a[128:131],  v66, v74 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[18:21], v126 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[106:109], v[54:57], a[132:135],  v66, v74 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[22:25], v126 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[106:109], v[58:61], a[136:139],  v66, v75 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[26:29], v126 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[106:109], v[62:65], a[140:143],  v66, v75 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v126 offset:6144

	;;#ASMEND
	buffer_load_dwordx4 v98, s[16:19], s77 offen lds
	s_mov_b32 m0, s36
	s_add_i32 s74, s73, 0x400
	buffer_load_dwordx4 v99, s[16:19], s77 offen lds
	s_mov_b32 m0, s44
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[86:89], v[34:37], a[144:147],  v66, v74 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[86:89], v[38:41], a[148:151],  v66, v74 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[86:89], v[42:45], a[152:155],  v66, v75 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[86:89], v[46:49], a[156:159],  v66, v75 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[110:113], v[50:53], a[144:147],  v66, v74 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[110:113], v[54:57], a[148:151],  v66, v74 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[110:113], v[58:61], a[152:155],  v66, v75 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[110:113], v[62:65], a[156:159],  v66, v75 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_i32 s78, s37, 0xfffff000
	buffer_load_dwordx4 v122, s[16:19], s77 offen lds
	s_mov_b32 m0, s45
	s_nop 0
	buffer_load_dwordx4 v100, s[16:19], s77 offen lds
	s_mov_b32 m0, s35
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[90:93], v[34:37], a[160:163],  v67, v74 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[90:93], v[38:41], a[164:167],  v67, v74 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[90:93], v[42:45], a[168:171], v67, v75 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[90:93], v[46:49], a[172:175], v67, v75 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[114:117], v[50:53], a[160:163],  v67, v74 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[114:117], v[54:57], a[164:167],  v67, v74 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[114:117], v[58:61], a[168:171], v67, v75 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[114:117], v[62:65], a[172:175], v67, v75 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_i32 s77, s70, 0x480
	buffer_load_dwordx4 v98, s[16:19], s76 offen lds
	s_mov_b32 m0, s47
	s_nop 0
	buffer_load_dwordx4 v99, s[16:19], s76 offen lds
	s_mov_b32 m0, s48
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[94:97], v[34:37], a[176:179], v67, v74 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[94:97], v[38:41], a[180:183], v67, v74 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[94:97], v[42:45], a[184:187], v67, v75 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[94:97], v[46:49], a[188:191], v67, v75 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[118:121], v[50:53], a[176:179], v67, v74 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[118:121], v[54:57], a[180:183], v67, v74 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[118:121], v[58:61], a[184:187], v67, v75 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[118:121], v[62:65], a[188:191], v67, v75 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v122, s[16:19], s76 offen lds
	s_mov_b32 m0, s49
	s_nop 0
	buffer_load_dwordx4 v100, s[16:19], s76 offen lds
	s_mov_b32 m0, s39
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[82:85], v[140:143], a[192:195],  v66, v70 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v129 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[82:85], v[144:147], a[196:199],  v66, v70 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v129 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[82:85], v[148:151], a[200:203],  v66, v71 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v129 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[82:85], v[152:155], a[204:207],  v66, v71 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v129 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[106:109], v[156:159], a[192:195],  v66, v70 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v130 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[106:109], v[160:163], a[196:199],  v66, v70 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v130 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[106:109], v[164:167], a[200:203],  v66, v71 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v130 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[106:109], v[168:171], a[204:207],  v66, v71 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[62:65], v130 offset:6144

	;;#ASMEND
	s_add_i32 s76, s71, 0x480
	buffer_load_dwordx4 v102, s[20:23], s75 offen lds
	s_mov_b32 m0, s50
	s_nop 0
	buffer_load_dwordx4 v101, s[20:23], s75 offen lds
	s_mov_b32 m0, s51
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[86:89], v[140:143], a[208:211],  v66, v70 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[86:89], v[144:147], a[212:215],  v66, v70 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[86:89], v[148:151], a[216:219],  v66, v71 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[86:89], v[152:155], a[220:223],  v66, v71 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[110:113], v[156:159], a[208:211],  v66, v70 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[110:113], v[160:163], a[212:215],  v66, v70 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[110:113], v[164:167], a[216:219],  v66, v71 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[110:113], v[168:171], a[220:223],  v66, v71 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v103, s[20:23], s75 offen lds
	s_mov_b32 m0, s52
	s_nop 0
	buffer_load_dwordx4 v104, s[20:23], s75 offen lds
	s_mov_b32 m0, s41
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[90:93], v[140:143], a[224:227],  v67, v70 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[90:93], v[144:147], a[228:231],  v67, v70 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[90:93], v[148:151], a[232:235], v67, v71 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[90:93], v[152:155], a[236:239], v67, v71 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[114:117], v[156:159], a[224:227],  v67, v70 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[114:117], v[160:163], a[228:231],  v67, v70 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[114:117], v[164:167], a[232:235], v67, v71 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[114:117], v[168:171], a[236:239], v67, v71 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_i32 s75, s72, 0x480
	buffer_load_dwordx4 v102, s[20:23], s74 offen lds
	s_mov_b32 m0, s54
	s_nop 0
	buffer_load_dwordx4 v101, s[20:23], s74 offen lds
	s_mov_b32 m0, s55
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[94:97], v[140:143], a[240:243], v67, v70 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[94:97], v[144:147], a[244:247], v67, v70 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[94:97], v[148:151], a[248:251], v67, v71 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[94:97], v[152:155], a[252:255], v67, v71 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[118:121], v[156:159], a[240:243], v67, v70 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[118:121], v[160:163], a[244:247], v67, v70 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[118:121], v[164:167], a[248:251], v67, v71 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[118:121], v[168:171], a[252:255], v67, v71 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v103, s[20:23], s74 offen lds
	s_mov_b32 m0, s56
	s_nop 0
	buffer_load_dwordx4 v104, s[20:23], s74 offen lds
	s_mov_b32 m0, s33
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[78:79], v105, s[0:3], s78 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[66:67], v105, s[4:7], s78 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[74:75], v105, s[8:11], s78 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[70:71], v105, s[12:15], s78 offen
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[2:5], v[34:37], a[28:31],  v80, v76 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[140:143], v133 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[2:5], v[38:41], a[32:35],  v80, v76 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[144:147], v133 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[2:5], v[42:45], a[36:39],  v80, v77 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[148:151], v133 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[2:5], v[46:49], a[40:43],  v80, v77 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[152:155], v133 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[18:21], v[50:53], a[28:31],  v80, v76 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[156:159], v135 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[18:21], v[54:57], a[32:35],  v80, v76 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[160:163], v135 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[18:21], v[58:61], a[36:39],  v80, v77 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[164:167], v135 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[18:21], v[62:65], a[40:43],  v80, v77 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[168:171], v135 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[6:9], v[34:37], a[44:47],  v80, v76 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[6:9], v[38:41], a[48:51],  v80, v76 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[6:9], v[42:45], a[52:55],  v80, v77 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[6:9], v[46:49], a[56:59],  v80, v77 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[22:25], v[50:53], a[44:47],  v80, v76 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[22:25], v[54:57], a[48:51],  v80, v76 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[22:25], v[58:61], a[52:55],  v80, v77 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[22:25], v[62:65], a[56:59],  v80, v77 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[10:13], v[34:37], a[60:63],  v81, v76 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[10:13], v[38:41], a[64:67],  v81, v76 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[10:13], v[42:45], a[68:71], v81, v77 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[10:13], v[46:49], a[72:75], v81, v77 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[26:29], v[50:53], a[60:63],  v81, v76 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[26:29], v[54:57], a[64:67],  v81, v76 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[26:29], v[58:61], a[68:71], v81, v77 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[26:29], v[62:65], a[72:75], v81, v77 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[14:17], v[34:37], a[76:79], v81, v76 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[14:17], v[38:41], a[80:83], v81, v76 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[14:17], v[42:45], a[84:87], v81, v77 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[14:17], v[46:49], a[88:91], v81, v77 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[30:33], v[50:53], a[76:79], v81, v76 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[30:33], v[54:57], a[80:83], v81, v76 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[30:33], v[58:61], a[84:87], v81, v77 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[30:33], v[62:65], a[88:91], v81, v77 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[2:5], v[140:143], a[0:3], v80, v72 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[82:85], v138 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[2:5], v[144:147], a[4:7], v80, v72 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[86:89], v138 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[2:5], v[148:151], a[8:11], v80, v73 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[90:93], v138 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[2:5], v[152:155], a[12:15], v80, v73 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[94:97], v138 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[18:21], v[156:159], a[0:3], v80, v72 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[106:109], v139 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[18:21], v[160:163], a[4:7], v80, v72 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[110:113], v139 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[18:21], v[164:167], a[8:11], v80, v73 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[114:117], v139 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[18:21], v[168:171], a[12:15], v80, v73 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[118:121], v139 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[6:9], v[140:143], a[16:19], v80, v72 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[6:9], v[144:147], a[20:23], v80, v72 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[6:9], v[148:151], a[24:27], v80, v73 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[6:9], v[152:155], a[92:95], v80, v73 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[22:25], v[156:159], a[16:19], v80, v72 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[22:25], v[160:163], a[20:23], v80, v72 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[22:25], v[164:167], a[24:27], v80, v73 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[22:25], v[168:171], a[92:95], v80, v73 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[10:13], v[140:143], a[96:99], v81, v72 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[10:13], v[144:147], a[100:103], v81, v72 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[10:13], v[148:151], a[104:107], v81, v73 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[10:13], v[152:155], a[108:111], v81, v73 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[26:29], v[156:159], a[96:99], v81, v72 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[26:29], v[160:163], a[100:103], v81, v72 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[26:29], v[164:167], a[104:107], v81, v73 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[26:29], v[168:171], a[108:111], v81, v73 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[14:17], v[140:143], a[112:115], v81, v72 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[14:17], v[144:147], a[116:119], v81, v72 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[14:17], v[148:151], a[120:123], v81, v73 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[14:17], v[152:155], a[124:127], v81, v73 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[30:33], v[156:159], a[112:115], v81, v72 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[30:33], v[160:163], a[116:119], v81, v72 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[30:33], v[164:167], a[120:123], v81, v73 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[30:33], v[168:171], a[124:127], v81, v73 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[82:85], v[34:37], a[128:131],  v68, v76 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[2:5], v123 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[82:85], v[38:41], a[132:135],  v68, v76 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[6:9], v123 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[82:85], v[42:45], a[136:139],  v68, v77 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[10:13], v123 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[82:85], v[46:49], a[140:143],  v68, v77 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[14:17], v123 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[106:109], v[50:53], a[128:131],  v68, v76 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[18:21], v124 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[106:109], v[54:57], a[132:135],  v68, v76 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[22:25], v124 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[106:109], v[58:61], a[136:139],  v68, v77 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[26:29], v124 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[106:109], v[62:65], a[140:143],  v68, v77 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v124 offset:6144

	;;#ASMEND
	buffer_load_dwordx4 v98, s[16:19], s77 offen lds
	s_mov_b32 m0, s57
	s_add_i32 s74, s73, 0x480
	buffer_load_dwordx4 v99, s[16:19], s77 offen lds
	s_mov_b32 m0, s58
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[86:89], v[34:37], a[144:147],  v68, v76 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[86:89], v[38:41], a[148:151],  v68, v76 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[86:89], v[42:45], a[152:155],  v68, v77 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[86:89], v[46:49], a[156:159],  v68, v77 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[110:113], v[50:53], a[144:147],  v68, v76 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[110:113], v[54:57], a[148:151],  v68, v76 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[110:113], v[58:61], a[152:155],  v68, v77 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[110:113], v[62:65], a[156:159],  v68, v77 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_i32 s78, s37, 0xfffff200
	buffer_load_dwordx4 v122, s[16:19], s77 offen lds
	s_mov_b32 m0, s59
	s_nop 0
	buffer_load_dwordx4 v100, s[16:19], s77 offen lds
	s_mov_b32 m0, s38
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[90:93], v[34:37], a[160:163],  v69, v76 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[90:93], v[38:41], a[164:167],  v69, v76 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[90:93], v[42:45], a[168:171], v69, v77 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[90:93], v[46:49], a[172:175], v69, v77 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[114:117], v[50:53], a[160:163],  v69, v76 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[114:117], v[54:57], a[164:167],  v69, v76 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[114:117], v[58:61], a[168:171], v69, v77 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[114:117], v[62:65], a[172:175], v69, v77 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_i32 s77, s70, 0x500
	buffer_load_dwordx4 v98, s[16:19], s76 offen lds
	s_mov_b32 m0, s60
	s_nop 0
	buffer_load_dwordx4 v99, s[16:19], s76 offen lds
	s_mov_b32 m0, s61
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[94:97], v[34:37], a[176:179], v69, v76 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[94:97], v[38:41], a[180:183], v69, v76 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[94:97], v[42:45], a[184:187], v69, v77 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[94:97], v[46:49], a[188:191], v69, v77 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[118:121], v[50:53], a[176:179], v69, v76 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[118:121], v[54:57], a[180:183], v69, v76 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[118:121], v[58:61], a[184:187], v69, v77 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[118:121], v[62:65], a[188:191], v69, v77 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v122, s[16:19], s76 offen lds
	s_mov_b32 m0, s62
	s_nop 0
	buffer_load_dwordx4 v100, s[16:19], s76 offen lds
	s_mov_b32 m0, s40
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[82:85], v[140:143], a[192:195],  v68, v72 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v127 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[82:85], v[144:147], a[196:199],  v68, v72 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v127 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[82:85], v[148:151], a[200:203],  v68, v73 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v127 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[82:85], v[152:155], a[204:207],  v68, v73 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v127 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[106:109], v[156:159], a[192:195],  v68, v72 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v128 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[106:109], v[160:163], a[196:199],  v68, v72 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v128 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[106:109], v[164:167], a[200:203],  v68, v73 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v128 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[106:109], v[168:171], a[204:207],  v68, v73 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[62:65], v128 offset:6144

	;;#ASMEND
	s_add_i32 s76, s71, 0x500
	buffer_load_dwordx4 v102, s[20:23], s75 offen lds
	s_mov_b32 m0, s63
	s_nop 0
	buffer_load_dwordx4 v101, s[20:23], s75 offen lds
	s_mov_b32 m0, s64
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[86:89], v[140:143], a[208:211],  v68, v72 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[86:89], v[144:147], a[212:215],  v68, v72 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[86:89], v[148:151], a[216:219],  v68, v73 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[86:89], v[152:155], a[220:223],  v68, v73 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[110:113], v[156:159], a[208:211],  v68, v72 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[110:113], v[160:163], a[212:215],  v68, v72 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[110:113], v[164:167], a[216:219],  v68, v73 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[110:113], v[168:171], a[220:223],  v68, v73 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v103, s[20:23], s75 offen lds
	s_mov_b32 m0, s65
	s_nop 0
	buffer_load_dwordx4 v104, s[20:23], s75 offen lds
	s_mov_b32 m0, s42
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[90:93], v[140:143], a[224:227],  v69, v72 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[90:93], v[144:147], a[228:231],  v69, v72 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[90:93], v[148:151], a[232:235], v69, v73 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[90:93], v[152:155], a[236:239], v69, v73 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[114:117], v[156:159], a[224:227],  v69, v72 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[114:117], v[160:163], a[228:231],  v69, v72 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[114:117], v[164:167], a[232:235], v69, v73 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[114:117], v[168:171], a[236:239], v69, v73 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_i32 s75, s72, 0x500
	buffer_load_dwordx4 v102, s[20:23], s74 offen lds
	s_mov_b32 m0, s66
	s_nop 0
	buffer_load_dwordx4 v101, s[20:23], s74 offen lds
	s_mov_b32 m0, s67
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[94:97], v[140:143], a[240:243], v69, v72 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[94:97], v[144:147], a[244:247], v69, v72 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[94:97], v[148:151], a[248:251], v69, v73 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[94:97], v[152:155], a[252:255], v69, v73 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[118:121], v[156:159], a[240:243], v69, v72 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[118:121], v[160:163], a[244:247], v69, v72 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[118:121], v[164:167], a[248:251], v69, v73 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[118:121], v[168:171], a[252:255], v69, v73 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v103, s[20:23], s74 offen lds
	s_mov_b32 m0, s68
	s_nop 0
	buffer_load_dwordx4 v104, s[20:23], s74 offen lds
	s_mov_b32 m0, s31
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[80:81], v105, s[0:3], s78 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[68:69], v105, s[4:7], s78 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[76:77], v105, s[8:11], s78 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[72:73], v105, s[12:15], s78 offen
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[2:5], v[34:37], a[28:31],  v78, v74 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[140:143], v131 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[2:5], v[38:41], a[32:35],  v78, v74 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[144:147], v131 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[2:5], v[42:45], a[36:39],  v78, v75 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[148:151], v131 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[2:5], v[46:49], a[40:43],  v78, v75 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[152:155], v131 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[18:21], v[50:53], a[28:31],  v78, v74 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[156:159], v132 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[18:21], v[54:57], a[32:35],  v78, v74 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[160:163], v132 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[18:21], v[58:61], a[36:39],  v78, v75 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[164:167], v132 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[18:21], v[62:65], a[40:43],  v78, v75 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[168:171], v132 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[6:9], v[34:37], a[44:47],  v78, v74 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[6:9], v[38:41], a[48:51],  v78, v74 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[6:9], v[42:45], a[52:55],  v78, v75 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[6:9], v[46:49], a[56:59],  v78, v75 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[22:25], v[50:53], a[44:47],  v78, v74 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[22:25], v[54:57], a[48:51],  v78, v74 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[22:25], v[58:61], a[52:55],  v78, v75 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[22:25], v[62:65], a[56:59],  v78, v75 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[10:13], v[34:37], a[60:63],  v79, v74 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[10:13], v[38:41], a[64:67],  v79, v74 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[10:13], v[42:45], a[68:71], v79, v75 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[10:13], v[46:49], a[72:75], v79, v75 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[26:29], v[50:53], a[60:63],  v79, v74 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[26:29], v[54:57], a[64:67],  v79, v74 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[26:29], v[58:61], a[68:71], v79, v75 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[26:29], v[62:65], a[72:75], v79, v75 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[14:17], v[34:37], a[76:79], v79, v74 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[14:17], v[38:41], a[80:83], v79, v74 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[14:17], v[42:45], a[84:87], v79, v75 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[14:17], v[46:49], a[88:91], v79, v75 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[30:33], v[50:53], a[76:79], v79, v74 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[30:33], v[54:57], a[80:83], v79, v74 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[30:33], v[58:61], a[84:87], v79, v75 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[30:33], v[62:65], a[88:91], v79, v75 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[2:5], v[140:143], a[0:3], v78, v70 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[82:85], v136 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[2:5], v[144:147], a[4:7], v78, v70 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[86:89], v136 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[2:5], v[148:151], a[8:11], v78, v71 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[90:93], v136 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[2:5], v[152:155], a[12:15], v78, v71 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[94:97], v136 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[18:21], v[156:159], a[0:3], v78, v70 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[106:109], v137 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[18:21], v[160:163], a[4:7], v78, v70 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[110:113], v137 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[18:21], v[164:167], a[8:11], v78, v71 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[114:117], v137 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[18:21], v[168:171], a[12:15], v78, v71 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[118:121], v137 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[6:9], v[140:143], a[16:19], v78, v70 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[6:9], v[144:147], a[20:23], v78, v70 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[6:9], v[148:151], a[24:27], v78, v71 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[6:9], v[152:155], a[92:95], v78, v71 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[22:25], v[156:159], a[16:19], v78, v70 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[22:25], v[160:163], a[20:23], v78, v70 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[22:25], v[164:167], a[24:27], v78, v71 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[22:25], v[168:171], a[92:95], v78, v71 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[10:13], v[140:143], a[96:99], v79, v70 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[10:13], v[144:147], a[100:103], v79, v70 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[10:13], v[148:151], a[104:107], v79, v71 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[10:13], v[152:155], a[108:111], v79, v71 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[26:29], v[156:159], a[96:99], v79, v70 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[26:29], v[160:163], a[100:103], v79, v70 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[26:29], v[164:167], a[104:107], v79, v71 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[26:29], v[168:171], a[108:111], v79, v71 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[14:17], v[140:143], a[112:115], v79, v70 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[14:17], v[144:147], a[116:119], v79, v70 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[14:17], v[148:151], a[120:123], v79, v71 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[14:17], v[152:155], a[124:127], v79, v71 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[30:33], v[156:159], a[112:115], v79, v70 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[30:33], v[160:163], a[116:119], v79, v70 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[30:33], v[164:167], a[120:123], v79, v71 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[30:33], v[168:171], a[124:127], v79, v71 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[82:85], v[34:37], a[128:131],  v66, v74 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[2:5], v125 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[82:85], v[38:41], a[132:135],  v66, v74 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[6:9], v125 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[82:85], v[42:45], a[136:139],  v66, v75 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[10:13], v125 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[82:85], v[46:49], a[140:143],  v66, v75 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[14:17], v125 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[106:109], v[50:53], a[128:131],  v66, v74 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[18:21], v126 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[106:109], v[54:57], a[132:135],  v66, v74 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[22:25], v126 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[106:109], v[58:61], a[136:139],  v66, v75 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[26:29], v126 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[106:109], v[62:65], a[140:143],  v66, v75 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v126 offset:6144

	;;#ASMEND
	buffer_load_dwordx4 v98, s[16:19], s77 offen lds
	s_mov_b32 m0, s36
	s_add_i32 s74, s73, 0x500
	buffer_load_dwordx4 v99, s[16:19], s77 offen lds
	s_mov_b32 m0, s44
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[86:89], v[34:37], a[144:147],  v66, v74 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[86:89], v[38:41], a[148:151],  v66, v74 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[86:89], v[42:45], a[152:155],  v66, v75 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[86:89], v[46:49], a[156:159],  v66, v75 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[110:113], v[50:53], a[144:147],  v66, v74 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[110:113], v[54:57], a[148:151],  v66, v74 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[110:113], v[58:61], a[152:155],  v66, v75 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[110:113], v[62:65], a[156:159],  v66, v75 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_i32 s78, s37, 0xfffff400
	buffer_load_dwordx4 v122, s[16:19], s77 offen lds
	s_mov_b32 m0, s45
	s_nop 0
	buffer_load_dwordx4 v100, s[16:19], s77 offen lds
	s_mov_b32 m0, s35
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[90:93], v[34:37], a[160:163],  v67, v74 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[90:93], v[38:41], a[164:167],  v67, v74 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[90:93], v[42:45], a[168:171], v67, v75 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[90:93], v[46:49], a[172:175], v67, v75 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[114:117], v[50:53], a[160:163],  v67, v74 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[114:117], v[54:57], a[164:167],  v67, v74 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[114:117], v[58:61], a[168:171], v67, v75 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[114:117], v[62:65], a[172:175], v67, v75 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_i32 s77, s70, 0x580
	buffer_load_dwordx4 v98, s[16:19], s76 offen lds
	s_mov_b32 m0, s47
	s_nop 0
	buffer_load_dwordx4 v99, s[16:19], s76 offen lds
	s_mov_b32 m0, s48
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[94:97], v[34:37], a[176:179], v67, v74 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[94:97], v[38:41], a[180:183], v67, v74 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[94:97], v[42:45], a[184:187], v67, v75 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[94:97], v[46:49], a[188:191], v67, v75 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[118:121], v[50:53], a[176:179], v67, v74 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[118:121], v[54:57], a[180:183], v67, v74 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[118:121], v[58:61], a[184:187], v67, v75 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[118:121], v[62:65], a[188:191], v67, v75 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v122, s[16:19], s76 offen lds
	s_mov_b32 m0, s49
	s_nop 0
	buffer_load_dwordx4 v100, s[16:19], s76 offen lds
	s_mov_b32 m0, s39
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[82:85], v[140:143], a[192:195],  v66, v70 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v129 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[82:85], v[144:147], a[196:199],  v66, v70 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v129 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[82:85], v[148:151], a[200:203],  v66, v71 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v129 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[82:85], v[152:155], a[204:207],  v66, v71 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v129 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[106:109], v[156:159], a[192:195],  v66, v70 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v130 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[106:109], v[160:163], a[196:199],  v66, v70 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v130 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[106:109], v[164:167], a[200:203],  v66, v71 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v130 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[106:109], v[168:171], a[204:207],  v66, v71 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[62:65], v130 offset:6144

	;;#ASMEND
	s_add_i32 s76, s71, 0x580
	buffer_load_dwordx4 v102, s[20:23], s75 offen lds
	s_mov_b32 m0, s50
	s_nop 0
	buffer_load_dwordx4 v101, s[20:23], s75 offen lds
	s_mov_b32 m0, s51
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[86:89], v[140:143], a[208:211],  v66, v70 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[86:89], v[144:147], a[212:215],  v66, v70 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[86:89], v[148:151], a[216:219],  v66, v71 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[86:89], v[152:155], a[220:223],  v66, v71 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[110:113], v[156:159], a[208:211],  v66, v70 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[110:113], v[160:163], a[212:215],  v66, v70 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[110:113], v[164:167], a[216:219],  v66, v71 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[110:113], v[168:171], a[220:223],  v66, v71 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v103, s[20:23], s75 offen lds
	s_mov_b32 m0, s52
	s_nop 0
	buffer_load_dwordx4 v104, s[20:23], s75 offen lds
	s_mov_b32 m0, s41
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[90:93], v[140:143], a[224:227],  v67, v70 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[90:93], v[144:147], a[228:231],  v67, v70 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[90:93], v[148:151], a[232:235], v67, v71 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[90:93], v[152:155], a[236:239], v67, v71 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[114:117], v[156:159], a[224:227],  v67, v70 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[114:117], v[160:163], a[228:231],  v67, v70 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[114:117], v[164:167], a[232:235], v67, v71 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[114:117], v[168:171], a[236:239], v67, v71 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_i32 s75, s72, 0x580
	buffer_load_dwordx4 v102, s[20:23], s74 offen lds
	s_mov_b32 m0, s54
	s_nop 0
	buffer_load_dwordx4 v101, s[20:23], s74 offen lds
	s_mov_b32 m0, s55
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[94:97], v[140:143], a[240:243], v67, v70 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[94:97], v[144:147], a[244:247], v67, v70 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[94:97], v[148:151], a[248:251], v67, v71 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[94:97], v[152:155], a[252:255], v67, v71 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[118:121], v[156:159], a[240:243], v67, v70 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[118:121], v[160:163], a[244:247], v67, v70 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[118:121], v[164:167], a[248:251], v67, v71 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[118:121], v[168:171], a[252:255], v67, v71 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v103, s[20:23], s74 offen lds
	s_mov_b32 m0, s56
	s_nop 0
	buffer_load_dwordx4 v104, s[20:23], s74 offen lds
	s_mov_b32 m0, s33
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[78:79], v105, s[0:3], s78 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[66:67], v105, s[4:7], s78 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[74:75], v105, s[8:11], s78 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[70:71], v105, s[12:15], s78 offen
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[2:5], v[34:37], a[28:31],  v80, v76 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[140:143], v133 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[2:5], v[38:41], a[32:35],  v80, v76 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[144:147], v133 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[2:5], v[42:45], a[36:39],  v80, v77 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[148:151], v133 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[2:5], v[46:49], a[40:43],  v80, v77 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[152:155], v133 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[18:21], v[50:53], a[28:31],  v80, v76 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[156:159], v135 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[18:21], v[54:57], a[32:35],  v80, v76 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[160:163], v135 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[18:21], v[58:61], a[36:39],  v80, v77 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[164:167], v135 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[18:21], v[62:65], a[40:43],  v80, v77 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[168:171], v135 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[6:9], v[34:37], a[44:47],  v80, v76 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[6:9], v[38:41], a[48:51],  v80, v76 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[6:9], v[42:45], a[52:55],  v80, v77 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[6:9], v[46:49], a[56:59],  v80, v77 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[22:25], v[50:53], a[44:47],  v80, v76 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[22:25], v[54:57], a[48:51],  v80, v76 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[22:25], v[58:61], a[52:55],  v80, v77 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[22:25], v[62:65], a[56:59],  v80, v77 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[10:13], v[34:37], a[60:63],  v81, v76 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[10:13], v[38:41], a[64:67],  v81, v76 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[10:13], v[42:45], a[68:71], v81, v77 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[10:13], v[46:49], a[72:75], v81, v77 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[26:29], v[50:53], a[60:63],  v81, v76 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[26:29], v[54:57], a[64:67],  v81, v76 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[26:29], v[58:61], a[68:71], v81, v77 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[26:29], v[62:65], a[72:75], v81, v77 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[14:17], v[34:37], a[76:79], v81, v76 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[14:17], v[38:41], a[80:83], v81, v76 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[14:17], v[42:45], a[84:87], v81, v77 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[14:17], v[46:49], a[88:91], v81, v77 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[30:33], v[50:53], a[76:79], v81, v76 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[30:33], v[54:57], a[80:83], v81, v76 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[30:33], v[58:61], a[84:87], v81, v77 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[30:33], v[62:65], a[88:91], v81, v77 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[2:5], v[140:143], a[0:3], v80, v72 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[82:85], v138 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[2:5], v[144:147], a[4:7], v80, v72 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[86:89], v138 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[2:5], v[148:151], a[8:11], v80, v73 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[90:93], v138 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[2:5], v[152:155], a[12:15], v80, v73 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[94:97], v138 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[18:21], v[156:159], a[0:3], v80, v72 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[106:109], v139 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[18:21], v[160:163], a[4:7], v80, v72 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[110:113], v139 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[18:21], v[164:167], a[8:11], v80, v73 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[114:117], v139 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[18:21], v[168:171], a[12:15], v80, v73 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[118:121], v139 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[6:9], v[140:143], a[16:19], v80, v72 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[6:9], v[144:147], a[20:23], v80, v72 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[6:9], v[148:151], a[24:27], v80, v73 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[6:9], v[152:155], a[92:95], v80, v73 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[22:25], v[156:159], a[16:19], v80, v72 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[22:25], v[160:163], a[20:23], v80, v72 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[22:25], v[164:167], a[24:27], v80, v73 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[22:25], v[168:171], a[92:95], v80, v73 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[10:13], v[140:143], a[96:99], v81, v72 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[10:13], v[144:147], a[100:103], v81, v72 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[10:13], v[148:151], a[104:107], v81, v73 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[10:13], v[152:155], a[108:111], v81, v73 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[26:29], v[156:159], a[96:99], v81, v72 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[26:29], v[160:163], a[100:103], v81, v72 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[26:29], v[164:167], a[104:107], v81, v73 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[26:29], v[168:171], a[108:111], v81, v73 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[14:17], v[140:143], a[112:115], v81, v72 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[14:17], v[144:147], a[116:119], v81, v72 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[14:17], v[148:151], a[120:123], v81, v73 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[14:17], v[152:155], a[124:127], v81, v73 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[30:33], v[156:159], a[112:115], v81, v72 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[30:33], v[160:163], a[116:119], v81, v72 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[30:33], v[164:167], a[120:123], v81, v73 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[30:33], v[168:171], a[124:127], v81, v73 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[82:85], v[34:37], a[128:131],  v68, v76 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[2:5], v123 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[82:85], v[38:41], a[132:135],  v68, v76 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[6:9], v123 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[82:85], v[42:45], a[136:139],  v68, v77 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[10:13], v123 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[82:85], v[46:49], a[140:143],  v68, v77 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[14:17], v123 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[106:109], v[50:53], a[128:131],  v68, v76 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[18:21], v124 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[106:109], v[54:57], a[132:135],  v68, v76 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[22:25], v124 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[106:109], v[58:61], a[136:139],  v68, v77 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[26:29], v124 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[106:109], v[62:65], a[140:143],  v68, v77 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v124 offset:6144

	;;#ASMEND
	buffer_load_dwordx4 v98, s[16:19], s77 offen lds
	s_mov_b32 m0, s57
	s_add_i32 s74, s73, 0x580
	buffer_load_dwordx4 v99, s[16:19], s77 offen lds
	s_mov_b32 m0, s58
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[86:89], v[34:37], a[144:147],  v68, v76 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[86:89], v[38:41], a[148:151],  v68, v76 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[86:89], v[42:45], a[152:155],  v68, v77 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[86:89], v[46:49], a[156:159],  v68, v77 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[110:113], v[50:53], a[144:147],  v68, v76 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[110:113], v[54:57], a[148:151],  v68, v76 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[110:113], v[58:61], a[152:155],  v68, v77 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[110:113], v[62:65], a[156:159],  v68, v77 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_i32 s78, s37, 0xfffff600
	buffer_load_dwordx4 v122, s[16:19], s77 offen lds
	s_mov_b32 m0, s59
	s_nop 0
	buffer_load_dwordx4 v100, s[16:19], s77 offen lds
	s_mov_b32 m0, s38
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[90:93], v[34:37], a[160:163],  v69, v76 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[90:93], v[38:41], a[164:167],  v69, v76 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[90:93], v[42:45], a[168:171], v69, v77 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[90:93], v[46:49], a[172:175], v69, v77 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[114:117], v[50:53], a[160:163],  v69, v76 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[114:117], v[54:57], a[164:167],  v69, v76 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[114:117], v[58:61], a[168:171], v69, v77 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[114:117], v[62:65], a[172:175], v69, v77 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_i32 s77, s70, 0x600
	buffer_load_dwordx4 v98, s[16:19], s76 offen lds
	s_mov_b32 m0, s60
	s_nop 0
	buffer_load_dwordx4 v99, s[16:19], s76 offen lds
	s_mov_b32 m0, s61
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[94:97], v[34:37], a[176:179], v69, v76 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[94:97], v[38:41], a[180:183], v69, v76 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[94:97], v[42:45], a[184:187], v69, v77 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[94:97], v[46:49], a[188:191], v69, v77 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[118:121], v[50:53], a[176:179], v69, v76 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[118:121], v[54:57], a[180:183], v69, v76 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[118:121], v[58:61], a[184:187], v69, v77 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[118:121], v[62:65], a[188:191], v69, v77 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v122, s[16:19], s76 offen lds
	s_mov_b32 m0, s62
	s_nop 0
	buffer_load_dwordx4 v100, s[16:19], s76 offen lds
	s_mov_b32 m0, s40
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[82:85], v[140:143], a[192:195],  v68, v72 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v127 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[82:85], v[144:147], a[196:199],  v68, v72 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v127 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[82:85], v[148:151], a[200:203],  v68, v73 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v127 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[82:85], v[152:155], a[204:207],  v68, v73 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v127 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[106:109], v[156:159], a[192:195],  v68, v72 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v128 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[106:109], v[160:163], a[196:199],  v68, v72 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v128 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[106:109], v[164:167], a[200:203],  v68, v73 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v128 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[106:109], v[168:171], a[204:207],  v68, v73 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[62:65], v128 offset:6144

	;;#ASMEND
	s_add_i32 s76, s71, 0x600
	buffer_load_dwordx4 v102, s[20:23], s75 offen lds
	s_mov_b32 m0, s63
	s_nop 0
	buffer_load_dwordx4 v101, s[20:23], s75 offen lds
	s_mov_b32 m0, s64
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[86:89], v[140:143], a[208:211],  v68, v72 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[86:89], v[144:147], a[212:215],  v68, v72 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[86:89], v[148:151], a[216:219],  v68, v73 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[86:89], v[152:155], a[220:223],  v68, v73 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[110:113], v[156:159], a[208:211],  v68, v72 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[110:113], v[160:163], a[212:215],  v68, v72 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[110:113], v[164:167], a[216:219],  v68, v73 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[110:113], v[168:171], a[220:223],  v68, v73 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v103, s[20:23], s75 offen lds
	s_mov_b32 m0, s65
	s_nop 0
	buffer_load_dwordx4 v104, s[20:23], s75 offen lds
	s_mov_b32 m0, s42
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[90:93], v[140:143], a[224:227],  v69, v72 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[90:93], v[144:147], a[228:231],  v69, v72 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[90:93], v[148:151], a[232:235], v69, v73 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[90:93], v[152:155], a[236:239], v69, v73 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[114:117], v[156:159], a[224:227],  v69, v72 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[114:117], v[160:163], a[228:231],  v69, v72 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[114:117], v[164:167], a[232:235], v69, v73 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[114:117], v[168:171], a[236:239], v69, v73 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_i32 s75, s72, 0x600
	buffer_load_dwordx4 v102, s[20:23], s74 offen lds
	s_mov_b32 m0, s66
	s_nop 0
	buffer_load_dwordx4 v101, s[20:23], s74 offen lds
	s_mov_b32 m0, s67
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[94:97], v[140:143], a[240:243], v69, v72 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[94:97], v[144:147], a[244:247], v69, v72 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[94:97], v[148:151], a[248:251], v69, v73 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[94:97], v[152:155], a[252:255], v69, v73 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[118:121], v[156:159], a[240:243], v69, v72 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[118:121], v[160:163], a[244:247], v69, v72 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[118:121], v[164:167], a[248:251], v69, v73 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[118:121], v[168:171], a[252:255], v69, v73 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v103, s[20:23], s74 offen lds
	s_mov_b32 m0, s68
	s_nop 0
	buffer_load_dwordx4 v104, s[20:23], s74 offen lds
	s_mov_b32 m0, s31
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[80:81], v105, s[0:3], s78 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[68:69], v105, s[4:7], s78 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[76:77], v105, s[8:11], s78 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[72:73], v105, s[12:15], s78 offen
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[2:5], v[34:37], a[28:31],  v78, v74 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[140:143], v131 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[2:5], v[38:41], a[32:35],  v78, v74 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[144:147], v131 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[2:5], v[42:45], a[36:39],  v78, v75 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[148:151], v131 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[2:5], v[46:49], a[40:43],  v78, v75 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[152:155], v131 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[18:21], v[50:53], a[28:31],  v78, v74 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[156:159], v132 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[18:21], v[54:57], a[32:35],  v78, v74 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[160:163], v132 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[18:21], v[58:61], a[36:39],  v78, v75 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[164:167], v132 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[18:21], v[62:65], a[40:43],  v78, v75 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[168:171], v132 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[6:9], v[34:37], a[44:47],  v78, v74 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[6:9], v[38:41], a[48:51],  v78, v74 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[6:9], v[42:45], a[52:55],  v78, v75 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[6:9], v[46:49], a[56:59],  v78, v75 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[22:25], v[50:53], a[44:47],  v78, v74 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[22:25], v[54:57], a[48:51],  v78, v74 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[22:25], v[58:61], a[52:55],  v78, v75 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[22:25], v[62:65], a[56:59],  v78, v75 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[10:13], v[34:37], a[60:63],  v79, v74 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[10:13], v[38:41], a[64:67],  v79, v74 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[10:13], v[42:45], a[68:71], v79, v75 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[10:13], v[46:49], a[72:75], v79, v75 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[26:29], v[50:53], a[60:63],  v79, v74 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[26:29], v[54:57], a[64:67],  v79, v74 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[26:29], v[58:61], a[68:71], v79, v75 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[26:29], v[62:65], a[72:75], v79, v75 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[14:17], v[34:37], a[76:79], v79, v74 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[14:17], v[38:41], a[80:83], v79, v74 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[14:17], v[42:45], a[84:87], v79, v75 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[14:17], v[46:49], a[88:91], v79, v75 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[30:33], v[50:53], a[76:79], v79, v74 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[30:33], v[54:57], a[80:83], v79, v74 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[30:33], v[58:61], a[84:87], v79, v75 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[30:33], v[62:65], a[88:91], v79, v75 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[2:5], v[140:143], a[0:3], v78, v70 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[82:85], v136 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[2:5], v[144:147], a[4:7], v78, v70 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[86:89], v136 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[2:5], v[148:151], a[8:11], v78, v71 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[90:93], v136 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[2:5], v[152:155], a[12:15], v78, v71 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[94:97], v136 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[18:21], v[156:159], a[0:3], v78, v70 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[106:109], v137 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[18:21], v[160:163], a[4:7], v78, v70 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[110:113], v137 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[18:21], v[164:167], a[8:11], v78, v71 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[114:117], v137 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[18:21], v[168:171], a[12:15], v78, v71 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[118:121], v137 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[6:9], v[140:143], a[16:19], v78, v70 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[6:9], v[144:147], a[20:23], v78, v70 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[6:9], v[148:151], a[24:27], v78, v71 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[6:9], v[152:155], a[92:95], v78, v71 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[22:25], v[156:159], a[16:19], v78, v70 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[22:25], v[160:163], a[20:23], v78, v70 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[22:25], v[164:167], a[24:27], v78, v71 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[22:25], v[168:171], a[92:95], v78, v71 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[10:13], v[140:143], a[96:99], v79, v70 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[10:13], v[144:147], a[100:103], v79, v70 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[10:13], v[148:151], a[104:107], v79, v71 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[10:13], v[152:155], a[108:111], v79, v71 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[26:29], v[156:159], a[96:99], v79, v70 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[26:29], v[160:163], a[100:103], v79, v70 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[26:29], v[164:167], a[104:107], v79, v71 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[26:29], v[168:171], a[108:111], v79, v71 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[14:17], v[140:143], a[112:115], v79, v70 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[14:17], v[144:147], a[116:119], v79, v70 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[14:17], v[148:151], a[120:123], v79, v71 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[14:17], v[152:155], a[124:127], v79, v71 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[30:33], v[156:159], a[112:115], v79, v70 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[30:33], v[160:163], a[116:119], v79, v70 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[30:33], v[164:167], a[120:123], v79, v71 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[30:33], v[168:171], a[124:127], v79, v71 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[82:85], v[34:37], a[128:131],  v66, v74 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[2:5], v125 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[82:85], v[38:41], a[132:135],  v66, v74 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[6:9], v125 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[82:85], v[42:45], a[136:139],  v66, v75 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[10:13], v125 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[82:85], v[46:49], a[140:143],  v66, v75 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[14:17], v125 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[106:109], v[50:53], a[128:131],  v66, v74 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[18:21], v126 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[106:109], v[54:57], a[132:135],  v66, v74 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[22:25], v126 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[106:109], v[58:61], a[136:139],  v66, v75 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[26:29], v126 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[106:109], v[62:65], a[140:143],  v66, v75 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v126 offset:6144

	;;#ASMEND
	buffer_load_dwordx4 v98, s[16:19], s77 offen lds
	s_mov_b32 m0, s36
	s_add_i32 s74, s73, 0x600
	buffer_load_dwordx4 v99, s[16:19], s77 offen lds
	s_mov_b32 m0, s44
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[86:89], v[34:37], a[144:147],  v66, v74 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[86:89], v[38:41], a[148:151],  v66, v74 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[86:89], v[42:45], a[152:155],  v66, v75 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[86:89], v[46:49], a[156:159],  v66, v75 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[110:113], v[50:53], a[144:147],  v66, v74 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[110:113], v[54:57], a[148:151],  v66, v74 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[110:113], v[58:61], a[152:155],  v66, v75 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[110:113], v[62:65], a[156:159],  v66, v75 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_i32 s78, s37, 0xfffff800
	buffer_load_dwordx4 v122, s[16:19], s77 offen lds
	s_mov_b32 m0, s45
	s_nop 0
	buffer_load_dwordx4 v100, s[16:19], s77 offen lds
	s_mov_b32 m0, s35
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[90:93], v[34:37], a[160:163],  v67, v74 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[90:93], v[38:41], a[164:167],  v67, v74 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[90:93], v[42:45], a[168:171], v67, v75 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[90:93], v[46:49], a[172:175], v67, v75 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[114:117], v[50:53], a[160:163],  v67, v74 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[114:117], v[54:57], a[164:167],  v67, v74 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[114:117], v[58:61], a[168:171], v67, v75 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[114:117], v[62:65], a[172:175], v67, v75 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_i32 s77, s70, 0x680
	buffer_load_dwordx4 v98, s[16:19], s76 offen lds
	s_mov_b32 m0, s47
	s_nop 0
	buffer_load_dwordx4 v99, s[16:19], s76 offen lds
	s_mov_b32 m0, s48
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[94:97], v[34:37], a[176:179], v67, v74 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[94:97], v[38:41], a[180:183], v67, v74 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[94:97], v[42:45], a[184:187], v67, v75 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[94:97], v[46:49], a[188:191], v67, v75 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[118:121], v[50:53], a[176:179], v67, v74 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[118:121], v[54:57], a[180:183], v67, v74 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[118:121], v[58:61], a[184:187], v67, v75 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[118:121], v[62:65], a[188:191], v67, v75 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v122, s[16:19], s76 offen lds
	s_mov_b32 m0, s49
	s_nop 0
	buffer_load_dwordx4 v100, s[16:19], s76 offen lds
	s_mov_b32 m0, s39
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[82:85], v[140:143], a[192:195],  v66, v70 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v129 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[82:85], v[144:147], a[196:199],  v66, v70 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v129 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[82:85], v[148:151], a[200:203],  v66, v71 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v129 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[82:85], v[152:155], a[204:207],  v66, v71 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v129 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[106:109], v[156:159], a[192:195],  v66, v70 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v130 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[106:109], v[160:163], a[196:199],  v66, v70 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v130 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[106:109], v[164:167], a[200:203],  v66, v71 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v130 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[106:109], v[168:171], a[204:207],  v66, v71 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[62:65], v130 offset:6144

	;;#ASMEND
	s_add_i32 s76, s71, 0x680
	buffer_load_dwordx4 v102, s[20:23], s75 offen lds
	s_mov_b32 m0, s50
	s_nop 0
	buffer_load_dwordx4 v101, s[20:23], s75 offen lds
	s_mov_b32 m0, s51
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[86:89], v[140:143], a[208:211],  v66, v70 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[86:89], v[144:147], a[212:215],  v66, v70 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[86:89], v[148:151], a[216:219],  v66, v71 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[86:89], v[152:155], a[220:223],  v66, v71 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[110:113], v[156:159], a[208:211],  v66, v70 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[110:113], v[160:163], a[212:215],  v66, v70 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[110:113], v[164:167], a[216:219],  v66, v71 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[110:113], v[168:171], a[220:223],  v66, v71 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v103, s[20:23], s75 offen lds
	s_mov_b32 m0, s52
	s_nop 0
	buffer_load_dwordx4 v104, s[20:23], s75 offen lds
	s_mov_b32 m0, s41
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[90:93], v[140:143], a[224:227],  v67, v70 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[90:93], v[144:147], a[228:231],  v67, v70 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[90:93], v[148:151], a[232:235], v67, v71 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[90:93], v[152:155], a[236:239], v67, v71 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[114:117], v[156:159], a[224:227],  v67, v70 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[114:117], v[160:163], a[228:231],  v67, v70 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[114:117], v[164:167], a[232:235], v67, v71 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[114:117], v[168:171], a[236:239], v67, v71 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_i32 s75, s72, 0x680
	buffer_load_dwordx4 v102, s[20:23], s74 offen lds
	s_mov_b32 m0, s54
	s_nop 0
	buffer_load_dwordx4 v101, s[20:23], s74 offen lds
	s_mov_b32 m0, s55
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[94:97], v[140:143], a[240:243], v67, v70 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[94:97], v[144:147], a[244:247], v67, v70 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[94:97], v[148:151], a[248:251], v67, v71 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[94:97], v[152:155], a[252:255], v67, v71 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[118:121], v[156:159], a[240:243], v67, v70 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[118:121], v[160:163], a[244:247], v67, v70 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[118:121], v[164:167], a[248:251], v67, v71 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[118:121], v[168:171], a[252:255], v67, v71 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v103, s[20:23], s74 offen lds
	s_mov_b32 m0, s56
	s_nop 0
	buffer_load_dwordx4 v104, s[20:23], s74 offen lds
	s_mov_b32 m0, s33
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[78:79], v105, s[0:3], s78 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[66:67], v105, s[4:7], s78 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[74:75], v105, s[8:11], s78 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[70:71], v105, s[12:15], s78 offen
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[2:5], v[34:37], a[28:31],  v80, v76 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[140:143], v133 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[2:5], v[38:41], a[32:35],  v80, v76 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[144:147], v133 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[2:5], v[42:45], a[36:39],  v80, v77 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[148:151], v133 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[2:5], v[46:49], a[40:43],  v80, v77 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[152:155], v133 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[18:21], v[50:53], a[28:31],  v80, v76 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[156:159], v135 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[18:21], v[54:57], a[32:35],  v80, v76 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[160:163], v135 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[18:21], v[58:61], a[36:39],  v80, v77 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[164:167], v135 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[18:21], v[62:65], a[40:43],  v80, v77 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[168:171], v135 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[6:9], v[34:37], a[44:47],  v80, v76 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[6:9], v[38:41], a[48:51],  v80, v76 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[6:9], v[42:45], a[52:55],  v80, v77 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[6:9], v[46:49], a[56:59],  v80, v77 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[22:25], v[50:53], a[44:47],  v80, v76 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[22:25], v[54:57], a[48:51],  v80, v76 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[22:25], v[58:61], a[52:55],  v80, v77 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[22:25], v[62:65], a[56:59],  v80, v77 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[10:13], v[34:37], a[60:63],  v81, v76 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[10:13], v[38:41], a[64:67],  v81, v76 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[10:13], v[42:45], a[68:71], v81, v77 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[10:13], v[46:49], a[72:75], v81, v77 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[26:29], v[50:53], a[60:63],  v81, v76 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[26:29], v[54:57], a[64:67],  v81, v76 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[26:29], v[58:61], a[68:71], v81, v77 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[26:29], v[62:65], a[72:75], v81, v77 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[14:17], v[34:37], a[76:79], v81, v76 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[14:17], v[38:41], a[80:83], v81, v76 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[14:17], v[42:45], a[84:87], v81, v77 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[14:17], v[46:49], a[88:91], v81, v77 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[30:33], v[50:53], a[76:79], v81, v76 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[30:33], v[54:57], a[80:83], v81, v76 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[30:33], v[58:61], a[84:87], v81, v77 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[30:33], v[62:65], a[88:91], v81, v77 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[2:5], v[140:143], a[0:3], v80, v72 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[82:85], v138 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[2:5], v[144:147], a[4:7], v80, v72 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[86:89], v138 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[2:5], v[148:151], a[8:11], v80, v73 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[90:93], v138 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[2:5], v[152:155], a[12:15], v80, v73 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[94:97], v138 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[18:21], v[156:159], a[0:3], v80, v72 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[106:109], v139 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[18:21], v[160:163], a[4:7], v80, v72 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[110:113], v139 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[18:21], v[164:167], a[8:11], v80, v73 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[114:117], v139 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[18:21], v[168:171], a[12:15], v80, v73 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[118:121], v139 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[6:9], v[140:143], a[16:19], v80, v72 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[6:9], v[144:147], a[20:23], v80, v72 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[6:9], v[148:151], a[24:27], v80, v73 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[6:9], v[152:155], a[92:95], v80, v73 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[22:25], v[156:159], a[16:19], v80, v72 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[22:25], v[160:163], a[20:23], v80, v72 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[22:25], v[164:167], a[24:27], v80, v73 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[22:25], v[168:171], a[92:95], v80, v73 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[10:13], v[140:143], a[96:99], v81, v72 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[10:13], v[144:147], a[100:103], v81, v72 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[10:13], v[148:151], a[104:107], v81, v73 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[10:13], v[152:155], a[108:111], v81, v73 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[26:29], v[156:159], a[96:99], v81, v72 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[26:29], v[160:163], a[100:103], v81, v72 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[26:29], v[164:167], a[104:107], v81, v73 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[26:29], v[168:171], a[108:111], v81, v73 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[14:17], v[140:143], a[112:115], v81, v72 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[14:17], v[144:147], a[116:119], v81, v72 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[14:17], v[148:151], a[120:123], v81, v73 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[14:17], v[152:155], a[124:127], v81, v73 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[30:33], v[156:159], a[112:115], v81, v72 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[30:33], v[160:163], a[116:119], v81, v72 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[30:33], v[164:167], a[120:123], v81, v73 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[30:33], v[168:171], a[124:127], v81, v73 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[82:85], v[34:37], a[128:131],  v68, v76 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[2:5], v123 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[82:85], v[38:41], a[132:135],  v68, v76 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[6:9], v123 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[82:85], v[42:45], a[136:139],  v68, v77 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[10:13], v123 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[82:85], v[46:49], a[140:143],  v68, v77 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[14:17], v123 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[106:109], v[50:53], a[128:131],  v68, v76 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[18:21], v124 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[106:109], v[54:57], a[132:135],  v68, v76 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[22:25], v124 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[106:109], v[58:61], a[136:139],  v68, v77 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[26:29], v124 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[106:109], v[62:65], a[140:143],  v68, v77 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v124 offset:6144

	;;#ASMEND
	buffer_load_dwordx4 v98, s[16:19], s77 offen lds
	s_mov_b32 m0, s57
	s_add_i32 s74, s73, 0x680
	buffer_load_dwordx4 v99, s[16:19], s77 offen lds
	s_mov_b32 m0, s58
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[86:89], v[34:37], a[144:147],  v68, v76 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[86:89], v[38:41], a[148:151],  v68, v76 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[86:89], v[42:45], a[152:155],  v68, v77 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[86:89], v[46:49], a[156:159],  v68, v77 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[110:113], v[50:53], a[144:147],  v68, v76 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[110:113], v[54:57], a[148:151],  v68, v76 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[110:113], v[58:61], a[152:155],  v68, v77 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[110:113], v[62:65], a[156:159],  v68, v77 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_i32 s78, s37, 0xfffffa00
	buffer_load_dwordx4 v122, s[16:19], s77 offen lds
	s_mov_b32 m0, s59
	s_nop 0
	buffer_load_dwordx4 v100, s[16:19], s77 offen lds
	s_mov_b32 m0, s38
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[90:93], v[34:37], a[160:163],  v69, v76 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[90:93], v[38:41], a[164:167],  v69, v76 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[90:93], v[42:45], a[168:171], v69, v77 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[90:93], v[46:49], a[172:175], v69, v77 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[114:117], v[50:53], a[160:163],  v69, v76 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[114:117], v[54:57], a[164:167],  v69, v76 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[114:117], v[58:61], a[168:171], v69, v77 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[114:117], v[62:65], a[172:175], v69, v77 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_i32 s77, s70, 0x700
	buffer_load_dwordx4 v98, s[16:19], s76 offen lds
	s_mov_b32 m0, s60
	s_nop 0
	buffer_load_dwordx4 v99, s[16:19], s76 offen lds
	s_mov_b32 m0, s61
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[94:97], v[34:37], a[176:179], v69, v76 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[94:97], v[38:41], a[180:183], v69, v76 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[94:97], v[42:45], a[184:187], v69, v77 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[94:97], v[46:49], a[188:191], v69, v77 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[118:121], v[50:53], a[176:179], v69, v76 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[118:121], v[54:57], a[180:183], v69, v76 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[118:121], v[58:61], a[184:187], v69, v77 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[118:121], v[62:65], a[188:191], v69, v77 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v122, s[16:19], s76 offen lds
	s_mov_b32 m0, s62
	s_nop 0
	buffer_load_dwordx4 v100, s[16:19], s76 offen lds
	s_mov_b32 m0, s40
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[82:85], v[140:143], a[192:195],  v68, v72 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v127 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[82:85], v[144:147], a[196:199],  v68, v72 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v127 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[82:85], v[148:151], a[200:203],  v68, v73 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v127 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[82:85], v[152:155], a[204:207],  v68, v73 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v127 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[106:109], v[156:159], a[192:195],  v68, v72 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v128 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[106:109], v[160:163], a[196:199],  v68, v72 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v128 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[106:109], v[164:167], a[200:203],  v68, v73 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v128 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[106:109], v[168:171], a[204:207],  v68, v73 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[62:65], v128 offset:6144

	;;#ASMEND
	s_add_i32 s76, s71, 0x700
	buffer_load_dwordx4 v102, s[20:23], s75 offen lds
	s_mov_b32 m0, s63
	s_nop 0
	buffer_load_dwordx4 v101, s[20:23], s75 offen lds
	s_mov_b32 m0, s64
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[86:89], v[140:143], a[208:211],  v68, v72 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[86:89], v[144:147], a[212:215],  v68, v72 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[86:89], v[148:151], a[216:219],  v68, v73 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[86:89], v[152:155], a[220:223],  v68, v73 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[110:113], v[156:159], a[208:211],  v68, v72 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[110:113], v[160:163], a[212:215],  v68, v72 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[110:113], v[164:167], a[216:219],  v68, v73 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[110:113], v[168:171], a[220:223],  v68, v73 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v103, s[20:23], s75 offen lds
	s_mov_b32 m0, s65
	s_nop 0
	buffer_load_dwordx4 v104, s[20:23], s75 offen lds
	s_mov_b32 m0, s42
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[90:93], v[140:143], a[224:227],  v69, v72 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[90:93], v[144:147], a[228:231],  v69, v72 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[90:93], v[148:151], a[232:235], v69, v73 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[90:93], v[152:155], a[236:239], v69, v73 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[114:117], v[156:159], a[224:227],  v69, v72 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[114:117], v[160:163], a[228:231],  v69, v72 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[114:117], v[164:167], a[232:235], v69, v73 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[114:117], v[168:171], a[236:239], v69, v73 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_i32 s75, s72, 0x700
	buffer_load_dwordx4 v102, s[20:23], s74 offen lds
	s_mov_b32 m0, s66
	s_nop 0
	buffer_load_dwordx4 v101, s[20:23], s74 offen lds
	s_mov_b32 m0, s67
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[94:97], v[140:143], a[240:243], v69, v72 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[94:97], v[144:147], a[244:247], v69, v72 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[94:97], v[148:151], a[248:251], v69, v73 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[94:97], v[152:155], a[252:255], v69, v73 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[118:121], v[156:159], a[240:243], v69, v72 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[118:121], v[160:163], a[244:247], v69, v72 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[118:121], v[164:167], a[248:251], v69, v73 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[118:121], v[168:171], a[252:255], v69, v73 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v103, s[20:23], s74 offen lds
	s_mov_b32 m0, s68
	s_nop 0
	buffer_load_dwordx4 v104, s[20:23], s74 offen lds
	s_mov_b32 m0, s31
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[72:73], v105, s[0:3], s78 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[106:107], v105, s[4:7], s78 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[68:69], v105, s[8:11], s78 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[108:109], v105, s[12:15], s78 offen
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[2:5], v[34:37], a[28:31],  v78, v74 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[144:147], v131 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[2:5], v[38:41], a[32:35],  v78, v74 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[148:151], v131 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[2:5], v[42:45], a[36:39],  v78, v75 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[152:155], v131 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[2:5], v[46:49], a[40:43],  v78, v75 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[156:159], v131 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[18:21], v[50:53], a[28:31],  v78, v74 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[160:163], v132 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[18:21], v[54:57], a[32:35],  v78, v74 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[164:167], v132 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[18:21], v[58:61], a[36:39],  v78, v75 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[168:171], v132 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[18:21], v[62:65], a[40:43],  v78, v75 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[172:175], v132 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[6:9], v[34:37], a[44:47],  v78, v74 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[6:9], v[38:41], a[48:51],  v78, v74 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[6:9], v[42:45], a[52:55],  v78, v75 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[6:9], v[46:49], a[56:59],  v78, v75 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[22:25], v[50:53], a[44:47],  v78, v74 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[22:25], v[54:57], a[48:51],  v78, v74 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[22:25], v[58:61], a[52:55],  v78, v75 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[22:25], v[62:65], a[56:59],  v78, v75 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[10:13], v[34:37], a[60:63],  v79, v74 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[10:13], v[38:41], a[64:67],  v79, v74 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[10:13], v[42:45], a[68:71], v79, v75 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[10:13], v[46:49], a[72:75], v79, v75 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[26:29], v[50:53], a[60:63],  v79, v74 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[26:29], v[54:57], a[64:67],  v79, v74 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[26:29], v[58:61], a[68:71], v79, v75 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[26:29], v[62:65], a[72:75], v79, v75 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[14:17], v[34:37], a[76:79], v79, v74 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[14:17], v[38:41], a[80:83], v79, v74 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[14:17], v[42:45], a[84:87], v79, v75 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[14:17], v[46:49], a[88:91], v79, v75 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[30:33], v[50:53], a[76:79], v79, v74 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[30:33], v[54:57], a[80:83], v79, v74 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[30:33], v[58:61], a[84:87], v79, v75 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[30:33], v[62:65], a[88:91], v79, v75 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[2:5], v[144:147], a[0:3], v78, v70 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[80:83], v136 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[2:5], v[148:151], a[4:7], v78, v70 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[84:87], v136 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[2:5], v[152:155], a[8:11], v78, v71 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[88:91], v136 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[2:5], v[156:159], a[12:15], v78, v71 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[92:95], v136 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[18:21], v[160:163], a[0:3], v78, v70 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[110:113], v137 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[18:21], v[164:167], a[4:7], v78, v70 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[114:117], v137 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[18:21], v[168:171], a[8:11], v78, v71 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[118:121], v137 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[18:21], v[172:175], a[12:15], v78, v71 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[140:143], v137 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[6:9], v[144:147], a[16:19], v78, v70 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[6:9], v[148:151], a[20:23], v78, v70 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[6:9], v[152:155], a[24:27], v78, v71 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[6:9], v[156:159], a[92:95], v78, v71 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[22:25], v[160:163], a[16:19], v78, v70 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[22:25], v[164:167], a[20:23], v78, v70 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[22:25], v[168:171], a[24:27], v78, v71 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[22:25], v[172:175], a[92:95], v78, v71 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[10:13], v[144:147], a[96:99], v79, v70 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[10:13], v[148:151], a[100:103], v79, v70 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[10:13], v[152:155], a[104:107], v79, v71 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[10:13], v[156:159], a[108:111], v79, v71 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[26:29], v[160:163], a[96:99], v79, v70 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[26:29], v[164:167], a[100:103], v79, v70 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[26:29], v[168:171], a[104:107], v79, v71 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[26:29], v[172:175], a[108:111], v79, v71 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[14:17], v[144:147], a[112:115], v79, v70 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[14:17], v[148:151], a[116:119], v79, v70 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[14:17], v[152:155], a[120:123], v79, v71 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[14:17], v[156:159], a[124:127], v79, v71 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[30:33], v[160:163], a[112:115], v79, v70 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[30:33], v[164:167], a[116:119], v79, v70 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[30:33], v[168:171], a[120:123], v79, v71 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[30:33], v[172:175], a[124:127], v79, v71 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[80:83], v[34:37], a[128:131],  v66, v74 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[2:5], v125 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[80:83], v[38:41], a[132:135],  v66, v74 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[6:9], v125 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[80:83], v[42:45], a[136:139],  v66, v75 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[10:13], v125 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[80:83], v[46:49], a[140:143],  v66, v75 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[14:17], v125 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[110:113], v[50:53], a[128:131],  v66, v74 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[18:21], v126 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[110:113], v[54:57], a[132:135],  v66, v74 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[22:25], v126 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[110:113], v[58:61], a[136:139],  v66, v75 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[26:29], v126 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[110:113], v[62:65], a[140:143],  v66, v75 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v126 offset:6144

	;;#ASMEND
	buffer_load_dwordx4 v98, s[16:19], s77 offen lds
	s_mov_b32 m0, s36
	s_add_i32 s74, s73, 0x700
	buffer_load_dwordx4 v99, s[16:19], s77 offen lds
	s_mov_b32 m0, s44
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[84:87], v[34:37], a[144:147],  v66, v74 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[84:87], v[38:41], a[148:151],  v66, v74 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[84:87], v[42:45], a[152:155],  v66, v75 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[84:87], v[46:49], a[156:159],  v66, v75 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[114:117], v[50:53], a[144:147],  v66, v74 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[114:117], v[54:57], a[148:151],  v66, v74 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[114:117], v[58:61], a[152:155],  v66, v75 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[114:117], v[62:65], a[156:159],  v66, v75 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v122, s[16:19], s77 offen lds
	s_mov_b32 m0, s45
	s_nop 0
	buffer_load_dwordx4 v100, s[16:19], s77 offen lds
	s_mov_b32 m0, s35
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[88:91], v[34:37], a[160:163],  v67, v74 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[88:91], v[38:41], a[164:167],  v67, v74 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[88:91], v[42:45], a[168:171], v67, v75 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[88:91], v[46:49], a[172:175], v67, v75 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[118:121], v[50:53], a[160:163],  v67, v74 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[118:121], v[54:57], a[164:167],  v67, v74 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[118:121], v[58:61], a[168:171], v67, v75 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[118:121], v[62:65], a[172:175], v67, v75 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v98, s[16:19], s76 offen lds
	s_mov_b32 m0, s47
	s_nop 0
	buffer_load_dwordx4 v99, s[16:19], s76 offen lds
	s_mov_b32 m0, s48
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[92:95], v[34:37], a[176:179], v67, v74 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[92:95], v[38:41], a[180:183], v67, v74 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[92:95], v[42:45], a[184:187], v67, v75 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[92:95], v[46:49], a[188:191], v67, v75 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[140:143], v[50:53], a[176:179], v67, v74 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[140:143], v[54:57], a[180:183], v67, v74 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[140:143], v[58:61], a[184:187], v67, v75 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[140:143], v[62:65], a[188:191], v67, v75 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v122, s[16:19], s76 offen lds
	s_mov_b32 m0, s49
	s_nop 0
	buffer_load_dwordx4 v100, s[16:19], s76 offen lds
	s_mov_b32 m0, s39
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[80:83], v[144:147], a[192:195],  v66, v70 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v129 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[80:83], v[148:151], a[196:199],  v66, v70 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v129 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[80:83], v[152:155], a[200:203],  v66, v71 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v129 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[80:83], v[156:159], a[204:207],  v66, v71 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v129 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[110:113], v[160:163], a[192:195],  v66, v70 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v130 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[110:113], v[164:167], a[196:199],  v66, v70 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v130 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[110:113], v[168:171], a[200:203],  v66, v71 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v130 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[110:113], v[172:175], a[204:207],  v66, v71 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[62:65], v130 offset:6144

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v102, s[20:23], s75 offen lds
	s_mov_b32 m0, s50
	s_nop 0
	buffer_load_dwordx4 v101, s[20:23], s75 offen lds
	s_mov_b32 m0, s51
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[84:87], v[144:147], a[208:211],  v66, v70 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[84:87], v[148:151], a[212:215],  v66, v70 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[84:87], v[152:155], a[216:219],  v66, v71 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[84:87], v[156:159], a[220:223],  v66, v71 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[114:117], v[160:163], a[208:211],  v66, v70 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[114:117], v[164:167], a[212:215],  v66, v70 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[114:117], v[168:171], a[216:219],  v66, v71 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[114:117], v[172:175], a[220:223],  v66, v71 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v103, s[20:23], s75 offen lds
	s_mov_b32 m0, s52
	s_nop 0
	buffer_load_dwordx4 v104, s[20:23], s75 offen lds
	s_mov_b32 m0, s41
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[88:91], v[144:147], a[224:227],  v67, v70 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[88:91], v[148:151], a[228:231],  v67, v70 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[88:91], v[152:155], a[232:235], v67, v71 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[88:91], v[156:159], a[236:239], v67, v71 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[118:121], v[160:163], a[224:227],  v67, v70 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[118:121], v[164:167], a[228:231],  v67, v70 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[118:121], v[168:171], a[232:235], v67, v71 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[118:121], v[172:175], a[236:239], v67, v71 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_i32 s75, s70, 0x780
	buffer_load_dwordx4 v102, s[20:23], s74 offen lds
	s_mov_b32 m0, s54
	s_add_i32 s70, s73, 0x780
	buffer_load_dwordx4 v101, s[20:23], s74 offen lds
	s_mov_b32 m0, s55
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[92:95], v[144:147], a[240:243], v67, v70 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[92:95], v[148:151], a[244:247], v67, v70 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[92:95], v[152:155], a[248:251], v67, v71 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[92:95], v[156:159], a[252:255], v67, v71 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[140:143], v[160:163], a[240:243], v67, v70 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[140:143], v[164:167], a[244:247], v67, v70 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[140:143], v[168:171], a[248:251], v67, v71 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[140:143], v[172:175], a[252:255], v67, v71 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_i32 s73, s37, 0xfffffc00
	buffer_load_dwordx4 v103, s[20:23], s74 offen lds
	s_mov_b32 m0, s56
	s_nop 0
	buffer_load_dwordx4 v104, s[20:23], s74 offen lds
	s_mov_b32 m0, s33
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[120:121], v105, s[0:3], s73 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[114:115], v105, s[4:7], s73 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[118:119], v105, s[8:11], s73 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[116:117], v105, s[12:15], s73 offen
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[2:5], v[34:37], a[28:31],  v72, v68 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[168:171], v133 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[2:5], v[38:41], a[32:35],  v72, v68 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[172:175], v133 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[2:5], v[42:45], a[36:39],  v72, v69 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[176:179], v133 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[2:5], v[46:49], a[40:43],  v72, v69 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[180:183], v133 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[18:21], v[50:53], a[28:31],  v72, v68 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[184:187], v135 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[18:21], v[54:57], a[32:35],  v72, v68 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[188:191], v135 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[18:21], v[58:61], a[36:39],  v72, v69 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[192:195], v135 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[18:21], v[62:65], a[40:43],  v72, v69 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[196:199], v135 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[6:9], v[34:37], a[44:47],  v72, v68 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[6:9], v[38:41], a[48:51],  v72, v68 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[6:9], v[42:45], a[52:55],  v72, v69 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[6:9], v[46:49], a[56:59],  v72, v69 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[22:25], v[50:53], a[44:47],  v72, v68 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[22:25], v[54:57], a[48:51],  v72, v68 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[22:25], v[58:61], a[52:55],  v72, v69 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[22:25], v[62:65], a[56:59],  v72, v69 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[10:13], v[34:37], a[60:63],  v73, v68 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[10:13], v[38:41], a[64:67],  v73, v68 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[10:13], v[42:45], a[68:71], v73, v69 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[10:13], v[46:49], a[72:75], v73, v69 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[26:29], v[50:53], a[60:63],  v73, v68 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[26:29], v[54:57], a[64:67],  v73, v68 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[26:29], v[58:61], a[68:71], v73, v69 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[26:29], v[62:65], a[72:75], v73, v69 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[14:17], v[34:37], a[76:79], v73, v68 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[14:17], v[38:41], a[80:83], v73, v68 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[14:17], v[42:45], a[84:87], v73, v69 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[14:17], v[46:49], a[88:91], v73, v69 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[30:33], v[50:53], a[76:79], v73, v68 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[30:33], v[54:57], a[80:83], v73, v68 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[30:33], v[58:61], a[84:87], v73, v69 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[30:33], v[62:65], a[88:91], v73, v69 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[2:5], v[168:171], a[0:3], v72, v108 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[110:113], v138 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[2:5], v[172:175], a[4:7], v72, v108 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[140:143], v138 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[2:5], v[176:179], a[8:11], v72, v109 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[144:147], v138 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[2:5], v[180:183], a[12:15], v72, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[148:151], v138 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[18:21], v[184:187], a[0:3], v72, v108 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[152:155], v139 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[18:21], v[188:191], a[4:7], v72, v108 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[156:159], v139 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[18:21], v[192:195], a[8:11], v72, v109 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[160:163], v139 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[18:21], v[196:199], a[12:15], v72, v109 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[164:167], v139 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[6:9], v[168:171], a[16:19], v72, v108 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[6:9], v[172:175], a[20:23], v72, v108 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[6:9], v[176:179], a[24:27], v72, v109 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[6:9], v[180:183], a[92:95], v72, v109 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[22:25], v[184:187], a[16:19], v72, v108 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[22:25], v[188:191], a[20:23], v72, v108 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[22:25], v[192:195], a[24:27], v72, v109 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[22:25], v[196:199], a[92:95], v72, v109 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[10:13], v[168:171], a[96:99], v73, v108 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[10:13], v[172:175], a[100:103], v73, v108 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[10:13], v[176:179], a[104:107], v73, v109 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[10:13], v[180:183], a[108:111], v73, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[26:29], v[184:187], a[96:99], v73, v108 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[26:29], v[188:191], a[100:103], v73, v108 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[26:29], v[192:195], a[104:107], v73, v109 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[26:29], v[196:199], a[108:111], v73, v109 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[14:17], v[168:171], a[112:115], v73, v108 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[14:17], v[172:175], a[116:119], v73, v108 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[14:17], v[176:179], a[120:123], v73, v109 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[14:17], v[180:183], a[124:127], v73, v109 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[30:33], v[184:187], a[112:115], v73, v108 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[30:33], v[188:191], a[116:119], v73, v108 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[30:33], v[192:195], a[120:123], v73, v109 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[30:33], v[196:199], a[124:127], v73, v109 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[110:113], v[34:37], a[128:131],  v106, v68 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[2:5], v123 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[110:113], v[38:41], a[132:135],  v106, v68 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[6:9], v123 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[110:113], v[42:45], a[136:139],  v106, v69 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[10:13], v123 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[110:113], v[46:49], a[140:143],  v106, v69 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[14:17], v123 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[152:155], v[50:53], a[128:131],  v106, v68 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[18:21], v124 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[152:155], v[54:57], a[132:135],  v106, v68 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[22:25], v124 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[152:155], v[58:61], a[136:139],  v106, v69 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[26:29], v124 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[152:155], v[62:65], a[140:143],  v106, v69 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v124 offset:6144

	;;#ASMEND
	buffer_load_dwordx4 v98, s[16:19], s75 offen lds
	s_mov_b32 m0, s57
	s_add_i32 s74, s71, 0x780
	buffer_load_dwordx4 v99, s[16:19], s75 offen lds
	s_mov_b32 m0, s58
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[140:143], v[34:37], a[144:147],  v106, v68 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[140:143], v[38:41], a[148:151],  v106, v68 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[140:143], v[42:45], a[152:155],  v106, v69 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[140:143], v[46:49], a[156:159],  v106, v69 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[156:159], v[50:53], a[144:147],  v106, v68 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[156:159], v[54:57], a[148:151],  v106, v68 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[156:159], v[58:61], a[152:155],  v106, v69 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[156:159], v[62:65], a[156:159],  v106, v69 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_i32 s71, s72, 0x780
	buffer_load_dwordx4 v122, s[16:19], s75 offen lds
	s_mov_b32 m0, s59
	s_add_i32 s72, s69, -1
	buffer_load_dwordx4 v100, s[16:19], s75 offen lds
	s_mov_b32 m0, s38
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[144:147], v[34:37], a[160:163],  v107, v68 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[144:147], v[38:41], a[164:167],  v107, v68 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[144:147], v[42:45], a[168:171], v107, v69 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[144:147], v[46:49], a[172:175], v107, v69 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[160:163], v[50:53], a[160:163],  v107, v68 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[160:163], v[54:57], a[164:167],  v107, v68 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[160:163], v[58:61], a[168:171], v107, v69 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[160:163], v[62:65], a[172:175], v107, v69 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v98, s[16:19], s74 offen lds
	s_mov_b32 m0, s60
	s_nop 0
	buffer_load_dwordx4 v99, s[16:19], s74 offen lds
	s_mov_b32 m0, s61
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[148:151], v[34:37], a[176:179], v107, v68 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[148:151], v[38:41], a[180:183], v107, v68 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[148:151], v[42:45], a[184:187], v107, v69 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[148:151], v[46:49], a[188:191], v107, v69 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[164:167], v[50:53], a[176:179], v107, v68 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[164:167], v[54:57], a[180:183], v107, v68 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[164:167], v[58:61], a[184:187], v107, v69 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[164:167], v[62:65], a[188:191], v107, v69 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v122, s[16:19], s74 offen lds
	s_mov_b32 m0, s62
	s_nop 0
	buffer_load_dwordx4 v100, s[16:19], s74 offen lds
	s_mov_b32 m0, s40
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[110:113], v[168:171], a[192:195],  v106, v108 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[66:69], v127 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[110:113], v[172:175], a[196:199],  v106, v108 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[70:73], v127 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[110:113], v[176:179], a[200:203],  v106, v109 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[74:77], v127 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[110:113], v[180:183], a[204:207],  v106, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[78:81], v127 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[152:155], v[184:187], a[192:195],  v106, v108 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[82:85], v128 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[152:155], v[188:191], a[196:199],  v106, v108 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[86:89], v128 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[152:155], v[192:195], a[200:203],  v106, v109 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[90:93], v128 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[152:155], v[196:199], a[204:207],  v106, v109 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[94:97], v128 offset:6144

	;;#ASMEND
	s_add_i32 s74, s37, 0xfffffe00
	buffer_load_dwordx4 v102, s[20:23], s71 offen lds
	s_mov_b32 m0, s63
	s_nop 0
	buffer_load_dwordx4 v101, s[20:23], s71 offen lds
	s_mov_b32 m0, s64
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[140:143], v[168:171], a[208:211],  v106, v108 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[140:143], v[172:175], a[212:215],  v106, v108 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[140:143], v[176:179], a[216:219],  v106, v109 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[140:143], v[180:183], a[220:223],  v106, v109 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[156:159], v[184:187], a[208:211],  v106, v108 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[156:159], v[188:191], a[212:215],  v106, v108 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[156:159], v[192:195], a[216:219],  v106, v109 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[156:159], v[196:199], a[220:223],  v106, v109 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v103, s[20:23], s71 offen lds
	s_mov_b32 m0, s65
	s_nop 0
	buffer_load_dwordx4 v104, s[20:23], s71 offen lds
	s_mov_b32 m0, s42
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[144:147], v[168:171], a[224:227],  v107, v108 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[144:147], v[172:175], a[228:231],  v107, v108 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[144:147], v[176:179], a[232:235], v107, v109 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[144:147], v[180:183], a[236:239], v107, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[160:163], v[184:187], a[224:227],  v107, v108 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[160:163], v[188:191], a[228:231],  v107, v108 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[160:163], v[192:195], a[232:235], v107, v109 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[160:163], v[196:199], a[236:239], v107, v109 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v102, s[20:23], s70 offen lds
	s_mov_b32 m0, s66
	s_nop 0
	buffer_load_dwordx4 v101, s[20:23], s70 offen lds
	s_mov_b32 m0, s67
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[148:151], v[168:171], a[240:243], v107, v108 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[148:151], v[172:175], a[244:247], v107, v108 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[148:151], v[176:179], a[248:251], v107, v109 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[148:151], v[180:183], a[252:255], v107, v109 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[164:167], v[184:187], a[240:243], v107, v108 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[164:167], v[188:191], a[244:247], v107, v108 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[164:167], v[192:195], a[248:251], v107, v109 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[164:167], v[196:199], a[252:255], v107, v109 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v103, s[20:23], s70 offen lds
	s_mov_b32 m0, s68
	s_nop 0
	buffer_load_dwordx4 v104, s[20:23], s70 offen lds
	s_min_u32 s70, s72, 29
	s_lshl_b32 s70, s70, 7
	s_addk_i32 s70, 0x100
	s_add_i32 s73, s70, s30
	s_mov_b32 m0, s31
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[112:113], v105, s[0:3], s74 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[106:107], v105, s[4:7], s74 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[110:111], v105, s[8:11], s74 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[108:109], v105, s[12:15], s74 offen
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[2:5], v[66:69], a[28:31],  v120, v118 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[172:175], v131 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[2:5], v[70:73], a[32:35],  v120, v118 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[176:179], v131 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[2:5], v[74:77], a[36:39],  v120, v119 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[180:183], v131 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[2:5], v[78:81], a[40:43],  v120, v119 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[184:187], v131 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[18:21], v[82:85], a[28:31],  v120, v118 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[188:191], v132 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[18:21], v[86:89], a[32:35],  v120, v118 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[192:195], v132 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[18:21], v[90:93], a[36:39],  v120, v119 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[196:199], v132 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[18:21], v[94:97], a[40:43],  v120, v119 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[200:203], v132 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[6:9], v[66:69], a[44:47],  v120, v118 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[6:9], v[70:73], a[48:51],  v120, v118 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[6:9], v[74:77], a[52:55],  v120, v119 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[6:9], v[78:81], a[56:59],  v120, v119 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[22:25], v[82:85], a[44:47],  v120, v118 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[22:25], v[86:89], a[48:51],  v120, v118 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[22:25], v[90:93], a[52:55],  v120, v119 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[22:25], v[94:97], a[56:59],  v120, v119 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[10:13], v[66:69], a[60:63],  v121, v118 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[10:13], v[70:73], a[64:67],  v121, v118 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[10:13], v[74:77], a[68:71], v121, v119 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[10:13], v[78:81], a[72:75], v121, v119 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[26:29], v[82:85], a[60:63],  v121, v118 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[26:29], v[86:89], a[64:67],  v121, v118 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[26:29], v[90:93], a[68:71], v121, v119 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[26:29], v[94:97], a[72:75], v121, v119 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[14:17], v[66:69], a[76:79], v121, v118 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[14:17], v[70:73], a[80:83], v121, v118 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[14:17], v[74:77], a[84:87], v121, v119 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[14:17], v[78:81], a[88:91], v121, v119 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[30:33], v[82:85], a[76:79], v121, v118 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[30:33], v[86:89], a[80:83], v121, v118 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[30:33], v[90:93], a[84:87], v121, v119 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[30:33], v[94:97], a[88:91], v121, v119 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[2:5], v[172:175], a[0:3], v120, v116 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[140:143], v136 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[2:5], v[176:179], a[4:7], v120, v116 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[144:147], v136 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[2:5], v[180:183], a[8:11], v120, v117 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[148:151], v136 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[2:5], v[184:187], a[12:15], v120, v117 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[152:155], v136 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[18:21], v[188:191], a[0:3], v120, v116 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[156:159], v137 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[18:21], v[192:195], a[4:7], v120, v116 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[160:163], v137 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[18:21], v[196:199], a[8:11], v120, v117 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[164:167], v137 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[18:21], v[200:203], a[12:15], v120, v117 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[168:171], v137 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[6:9], v[172:175], a[16:19], v120, v116 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[6:9], v[176:179], a[20:23], v120, v116 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[6:9], v[180:183], a[24:27], v120, v117 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[6:9], v[184:187], a[92:95], v120, v117 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[22:25], v[188:191], a[16:19], v120, v116 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[22:25], v[192:195], a[20:23], v120, v116 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[22:25], v[196:199], a[24:27], v120, v117 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[22:25], v[200:203], a[92:95], v120, v117 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[10:13], v[172:175], a[96:99], v121, v116 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[10:13], v[176:179], a[100:103], v121, v116 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[10:13], v[180:183], a[104:107], v121, v117 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[10:13], v[184:187], a[108:111], v121, v117 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[26:29], v[188:191], a[96:99], v121, v116 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[26:29], v[192:195], a[100:103], v121, v116 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[26:29], v[196:199], a[104:107], v121, v117 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[26:29], v[200:203], a[108:111], v121, v117 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[14:17], v[172:175], a[112:115], v121, v116 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[14:17], v[176:179], a[116:119], v121, v116 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[14:17], v[180:183], a[120:123], v121, v117 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[14:17], v[184:187], a[124:127], v121, v117 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[30:33], v[188:191], a[112:115], v121, v116 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[30:33], v[192:195], a[116:119], v121, v116 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[30:33], v[196:199], a[120:123], v121, v117 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[30:33], v[200:203], a[124:127], v121, v117 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[140:143], v[66:69], a[128:131],  v114, v118 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v125 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[140:143], v[70:73], a[132:135],  v114, v118 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v125 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[140:143], v[74:77], a[136:139],  v114, v119 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v125 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[140:143], v[78:81], a[140:143],  v114, v119 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v125 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[156:159], v[82:85], a[128:131],  v114, v118 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v126 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[156:159], v[86:89], a[132:135],  v114, v118 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v126 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[156:159], v[90:93], a[136:139],  v114, v119 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v126 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[156:159], v[94:97], a[140:143],  v114, v119 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[62:65], v126 offset:6144

	;;#ASMEND
	buffer_load_dwordx4 v98, s[16:19], s73 offen lds
	s_mov_b32 m0, s36
	s_add_i32 s72, s70, s46
	buffer_load_dwordx4 v99, s[16:19], s73 offen lds
	s_mov_b32 m0, s44
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[144:147], v[66:69], a[144:147],  v114, v118 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[144:147], v[70:73], a[148:151],  v114, v118 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[144:147], v[74:77], a[152:155],  v114, v119 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[144:147], v[78:81], a[156:159],  v114, v119 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[160:163], v[82:85], a[144:147],  v114, v118 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[160:163], v[86:89], a[148:151],  v114, v118 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[160:163], v[90:93], a[152:155],  v114, v119 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[160:163], v[94:97], a[156:159],  v114, v119 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_i32 s71, s70, s34
	buffer_load_dwordx4 v122, s[16:19], s73 offen lds
	s_mov_b32 m0, s45
	s_add_i32 s70, s70, s53
	buffer_load_dwordx4 v100, s[16:19], s73 offen lds
	s_mov_b32 m0, s35
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[148:151], v[66:69], a[160:163],  v115, v118 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[148:151], v[70:73], a[164:167],  v115, v118 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[148:151], v[74:77], a[168:171], v115, v119 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[148:151], v[78:81], a[172:175], v115, v119 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[164:167], v[82:85], a[160:163],  v115, v118 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[164:167], v[86:89], a[164:167],  v115, v118 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[164:167], v[90:93], a[168:171], v115, v119 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[164:167], v[94:97], a[172:175], v115, v119 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v98, s[16:19], s72 offen lds
	s_mov_b32 m0, s47
	s_nop 0
	buffer_load_dwordx4 v99, s[16:19], s72 offen lds
	s_mov_b32 m0, s48
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[152:155], v[66:69], a[176:179], v115, v118 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[152:155], v[70:73], a[180:183], v115, v118 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[152:155], v[74:77], a[184:187], v115, v119 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[152:155], v[78:81], a[188:191], v115, v119 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[168:171], v[82:85], a[176:179], v115, v118 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[168:171], v[86:89], a[180:183], v115, v118 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[168:171], v[90:93], a[184:187], v115, v119 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[168:171], v[94:97], a[188:191], v115, v119 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v122, s[16:19], s72 offen lds
	s_mov_b32 m0, s49
	s_nop 0
	buffer_load_dwordx4 v100, s[16:19], s72 offen lds
	s_mov_b32 m0, s39
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[140:143], v[172:175], a[192:195],  v114, v116 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[2:5], v129 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[140:143], v[176:179], a[196:199],  v114, v116 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[6:9], v129 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[140:143], v[180:183], a[200:203],  v114, v117 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[10:13], v129 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[140:143], v[184:187], a[204:207],  v114, v117 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[14:17], v129 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[156:159], v[188:191], a[192:195],  v114, v116 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[18:21], v130 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[156:159], v[192:195], a[196:199],  v114, v116 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[22:25], v130 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[156:159], v[196:199], a[200:203],  v114, v117 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[26:29], v130 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[156:159], v[200:203], a[204:207],  v114, v117 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v130 offset:6144

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v102, s[20:23], s71 offen lds
	s_mov_b32 m0, s50
	s_nop 0
	buffer_load_dwordx4 v101, s[20:23], s71 offen lds
	s_mov_b32 m0, s51
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[144:147], v[172:175], a[208:211],  v114, v116 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[144:147], v[176:179], a[212:215],  v114, v116 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[144:147], v[180:183], a[216:219],  v114, v117 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[144:147], v[184:187], a[220:223],  v114, v117 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[160:163], v[188:191], a[208:211],  v114, v116 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[160:163], v[192:195], a[212:215],  v114, v116 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[160:163], v[196:199], a[216:219],  v114, v117 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[160:163], v[200:203], a[220:223],  v114, v117 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v103, s[20:23], s71 offen lds
	s_mov_b32 m0, s52
	s_nop 0
	buffer_load_dwordx4 v104, s[20:23], s71 offen lds
	s_mov_b32 m0, s41
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[148:151], v[172:175], a[224:227],  v115, v116 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[148:151], v[176:179], a[228:231],  v115, v116 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[148:151], v[180:183], a[232:235], v115, v117 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[148:151], v[184:187], a[236:239], v115, v117 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[164:167], v[188:191], a[224:227],  v115, v116 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[164:167], v[192:195], a[228:231],  v115, v116 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[164:167], v[196:199], a[232:235], v115, v117 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[164:167], v[200:203], a[236:239], v115, v117 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v102, s[20:23], s70 offen lds
	s_mov_b32 m0, s54
	s_nop 0
	buffer_load_dwordx4 v101, s[20:23], s70 offen lds
	s_mov_b32 m0, s55
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[152:155], v[172:175], a[240:243], v115, v116 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[152:155], v[176:179], a[244:247], v115, v116 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[152:155], v[180:183], a[248:251], v115, v117 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[152:155], v[184:187], a[252:255], v115, v117 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[168:171], v[188:191], a[240:243], v115, v116 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[168:171], v[192:195], a[244:247], v115, v116 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[168:171], v[196:199], a[248:251], v115, v117 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[168:171], v[200:203], a[252:255], v115, v117 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v103, s[20:23], s70 offen lds
	s_mov_b32 m0, s56
	s_nop 0
	buffer_load_dwordx4 v104, s[20:23], s70 offen lds
	s_min_u32 s70, s69, 29
	s_lshl_b32 s70, s70, 7
	s_addk_i32 s70, 0x100
	s_add_i32 s73, s70, s30
	s_add_i32 s72, s70, s46
	s_add_i32 s71, s70, s34
	s_add_i32 s70, s70, s53
	s_cmpk_lg_i32 s43, 0x800
	s_mov_b32 m0, s33
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_cselect_b32 s74, s37, 0x3e00
	;;#ASMSTART
	buffer_load_dwordx2 v[72:73], v105, s[0:3], s74 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[66:67], v105, s[4:7], s74 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[70:71], v105, s[8:11], s74 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[68:69], v105, s[12:15], s74 offen
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[34:37], v[2:5], a[28:31],  v112, v110 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[140:143], v133 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[34:37], v[6:9], a[32:35],  v112, v110 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[144:147], v133 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[34:37], v[10:13], a[36:39],  v112, v111 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[148:151], v133 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[34:37], v[14:17], a[40:43],  v112, v111 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[152:155], v133 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[50:53], v[18:21], a[28:31],  v112, v110 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[156:159], v135 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[50:53], v[22:25], a[32:35],  v112, v110 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[160:163], v135 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[50:53], v[26:29], a[36:39],  v112, v111 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[164:167], v135 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[50:53], v[30:33], a[40:43],  v112, v111 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[168:171], v135 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[38:41], v[2:5], a[44:47],  v112, v110 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[38:41], v[6:9], a[48:51],  v112, v110 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[38:41], v[10:13], a[52:55],  v112, v111 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[38:41], v[14:17], a[56:59],  v112, v111 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[54:57], v[18:21], a[44:47],  v112, v110 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[54:57], v[22:25], a[48:51],  v112, v110 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[54:57], v[26:29], a[52:55],  v112, v111 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[54:57], v[30:33], a[56:59],  v112, v111 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[42:45], v[2:5], a[60:63],  v113, v110 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[42:45], v[6:9], a[64:67],  v113, v110 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[42:45], v[10:13], a[68:71], v113, v111 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[42:45], v[14:17], a[72:75], v113, v111 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[58:61], v[18:21], a[60:63],  v113, v110 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[58:61], v[22:25], a[64:67],  v113, v110 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[58:61], v[26:29], a[68:71], v113, v111 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[58:61], v[30:33], a[72:75], v113, v111 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[46:49], v[2:5], a[76:79], v113, v110 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[46:49], v[6:9], a[80:83], v113, v110 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[46:49], v[10:13], a[84:87], v113, v111 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[46:49], v[14:17], a[88:91], v113, v111 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[62:65], v[18:21], a[76:79], v113, v110 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[62:65], v[22:25], a[80:83], v113, v110 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[62:65], v[26:29], a[84:87], v113, v111 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[62:65], v[30:33], a[88:91], v113, v111 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[34:37], v[140:143], a[0:3], v112, v108 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[74:77], v138 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[34:37], v[144:147], a[4:7], v112, v108 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[78:81], v138 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[34:37], v[148:151], a[8:11], v112, v109 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[82:85], v138 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[34:37], v[152:155], a[12:15], v112, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[86:89], v138 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[50:53], v[156:159], a[0:3], v112, v108 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[90:93], v139 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[50:53], v[160:163], a[4:7], v112, v108 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[94:97], v139 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[50:53], v[164:167], a[8:11], v112, v109 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[114:117], v139 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[50:53], v[168:171], a[12:15], v112, v109 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[118:121], v139 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[38:41], v[140:143], a[16:19], v112, v108 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[38:41], v[144:147], a[20:23], v112, v108 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[38:41], v[148:151], a[24:27], v112, v109 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[38:41], v[152:155], a[92:95], v112, v109 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[54:57], v[156:159], a[16:19], v112, v108 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[54:57], v[160:163], a[20:23], v112, v108 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[54:57], v[164:167], a[24:27], v112, v109 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[54:57], v[168:171], a[92:95], v112, v109 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[42:45], v[140:143], a[96:99], v113, v108 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[42:45], v[144:147], a[100:103], v113, v108 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[42:45], v[148:151], a[104:107], v113, v109 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[42:45], v[152:155], a[108:111], v113, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[58:61], v[156:159], a[96:99], v113, v108 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[58:61], v[160:163], a[100:103], v113, v108 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[58:61], v[164:167], a[104:107], v113, v109 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[58:61], v[168:171], a[108:111], v113, v109 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[46:49], v[140:143], a[112:115], v113, v108 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[46:49], v[144:147], a[116:119], v113, v108 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[46:49], v[148:151], a[120:123], v113, v109 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[46:49], v[152:155], a[124:127], v113, v109 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[62:65], v[156:159], a[112:115], v113, v108 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[62:65], v[160:163], a[116:119], v113, v108 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[62:65], v[164:167], a[120:123], v113, v109 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[62:65], v[168:171], a[124:127], v113, v109 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[74:77], v[2:5], a[128:131],  v106, v110 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v123 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[74:77], v[6:9], a[132:135],  v106, v110 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v123 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[74:77], v[10:13], a[136:139],  v106, v111 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v123 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[74:77], v[14:17], a[140:143],  v106, v111 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v123 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[90:93], v[18:21], a[128:131],  v106, v110 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v124 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[90:93], v[22:25], a[132:135],  v106, v110 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v124 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[90:93], v[26:29], a[136:139],  v106, v111 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v124 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[90:93], v[30:33], a[140:143],  v106, v111 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[62:65], v124 offset:6144

	;;#ASMEND
	buffer_load_dwordx4 v98, s[16:19], s73 offen lds
	s_mov_b32 m0, s57
	s_addk_i32 s43, 0x800
	buffer_load_dwordx4 v99, s[16:19], s73 offen lds
	s_mov_b32 m0, s58
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[78:81], v[2:5], a[144:147],  v106, v110 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[78:81], v[6:9], a[148:151],  v106, v110 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[78:81], v[10:13], a[152:155],  v106, v111 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[78:81], v[14:17], a[156:159],  v106, v111 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[94:97], v[18:21], a[144:147],  v106, v110 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[94:97], v[22:25], a[148:151],  v106, v110 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[94:97], v[26:29], a[152:155],  v106, v111 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[94:97], v[30:33], a[156:159],  v106, v111 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_addk_i32 s37, 0x2000
	buffer_load_dwordx4 v122, s[16:19], s73 offen lds
	s_mov_b32 m0, s59
	s_add_i32 s69, s69, 16
	buffer_load_dwordx4 v100, s[16:19], s73 offen lds
	s_mov_b32 m0, s38
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[82:85], v[2:5], a[160:163],  v107, v110 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[82:85], v[6:9], a[164:167],  v107, v110 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[82:85], v[10:13], a[168:171], v107, v111 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[82:85], v[14:17], a[172:175], v107, v111 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[114:117], v[18:21], a[160:163],  v107, v110 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[114:117], v[22:25], a[164:167],  v107, v110 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[114:117], v[26:29], a[168:171], v107, v111 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[114:117], v[30:33], a[172:175], v107, v111 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_cmpk_eq_i32 s43, 0x1000
	buffer_load_dwordx4 v98, s[16:19], s72 offen lds
	s_mov_b32 m0, s60
	s_nop 0
	buffer_load_dwordx4 v99, s[16:19], s72 offen lds
	s_mov_b32 m0, s61
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[86:89], v[2:5], a[176:179], v107, v110 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[86:89], v[6:9], a[180:183], v107, v110 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[86:89], v[10:13], a[184:187], v107, v111 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[86:89], v[14:17], a[188:191], v107, v111 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[118:121], v[18:21], a[176:179], v107, v110 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[118:121], v[22:25], a[180:183], v107, v110 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[118:121], v[26:29], a[184:187], v107, v111 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[118:121], v[30:33], a[188:191], v107, v111 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v122, s[16:19], s72 offen lds
	s_mov_b32 m0, s62
	s_nop 0
	buffer_load_dwordx4 v100, s[16:19], s72 offen lds
	s_mov_b32 m0, s40
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[74:77], v[140:143], a[192:195],  v106, v108 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[2:5], v127 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[74:77], v[144:147], a[196:199],  v106, v108 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[6:9], v127 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[74:77], v[148:151], a[200:203],  v106, v109 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[10:13], v127 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[74:77], v[152:155], a[204:207],  v106, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[14:17], v127 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[90:93], v[156:159], a[192:195],  v106, v108 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[18:21], v128 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[90:93], v[160:163], a[196:199],  v106, v108 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[22:25], v128 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[90:93], v[164:167], a[200:203],  v106, v109 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[26:29], v128 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[90:93], v[168:171], a[204:207],  v106, v109 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v128 offset:6144

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v102, s[20:23], s71 offen lds
	s_mov_b32 m0, s63
	s_nop 0
	buffer_load_dwordx4 v101, s[20:23], s71 offen lds
	s_mov_b32 m0, s64
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[78:81], v[140:143], a[208:211],  v106, v108 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[78:81], v[144:147], a[212:215],  v106, v108 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[78:81], v[148:151], a[216:219],  v106, v109 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[78:81], v[152:155], a[220:223],  v106, v109 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[94:97], v[156:159], a[208:211],  v106, v108 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[94:97], v[160:163], a[212:215],  v106, v108 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[94:97], v[164:167], a[216:219],  v106, v109 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[94:97], v[168:171], a[220:223],  v106, v109 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v103, s[20:23], s71 offen lds
	s_mov_b32 m0, s65
	s_nop 0
	buffer_load_dwordx4 v104, s[20:23], s71 offen lds
	s_mov_b32 m0, s42
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[82:85], v[140:143], a[224:227],  v107, v108 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[82:85], v[144:147], a[228:231],  v107, v108 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[82:85], v[148:151], a[232:235], v107, v109 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[82:85], v[152:155], a[236:239], v107, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[114:117], v[156:159], a[224:227],  v107, v108 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[114:117], v[160:163], a[228:231],  v107, v108 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[114:117], v[164:167], a[232:235], v107, v109 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[114:117], v[168:171], a[236:239], v107, v109 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v102, s[20:23], s70 offen lds
	s_mov_b32 m0, s66
	s_nop 0
	buffer_load_dwordx4 v101, s[20:23], s70 offen lds
	s_mov_b32 m0, s67
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[86:89], v[140:143], a[240:243], v107, v108 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[86:89], v[144:147], a[244:247], v107, v108 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[86:89], v[148:151], a[248:251], v107, v109 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[86:89], v[152:155], a[252:255], v107, v109 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[118:121], v[156:159], a[240:243], v107, v108 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[118:121], v[160:163], a[244:247], v107, v108 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[118:121], v[164:167], a[248:251], v107, v109 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[118:121], v[168:171], a[252:255], v107, v109 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v103, s[20:23], s70 offen lds
	s_mov_b32 m0, s68
	s_nop 0
	buffer_load_dwordx4 v104, s[20:23], s70 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_cbranch_scc0 .LBB0_7
; %bb.8:
	v_lshl_or_b32 v1, s27, 2, v1
	v_lshl_or_b32 v2, s29, 2, v134
	v_mul_lo_u32 v3, v1, s28
	v_accvgpr_read_b32 v185, a79
	v_add_lshl_u32 v134, v3, v2, 6
	v_lshrrev_b32_e32 v4, 2, v0
	v_accvgpr_read_b32 v183, a77
	v_accvgpr_read_b32 v182, a76
	v_ashrrev_i32_e32 v135, 31, v134
	v_and_b32_e32 v4, 12, v4
	v_and_b32_e32 v0, 15, v0
	v_pk_mul_f32 v[250:251], v[182:183], s[26:27] op_sel_hi:[1,0]
	v_lshl_add_u64 v[182:183], v[134:135], 1, s[24:25]
	v_mad_u64_u32 v[134:135], s[0:1], v4, s28, v[0:1]
	v_ashrrev_i32_e32 v135, 31, v134
	v_accvgpr_read_b32 v189, a75
	v_lshlrev_b64 v[4:5], 1, v[134:135]
	v_add_u32_e32 v134, s28, v134
	v_accvgpr_read_b32 v181, a83
	v_accvgpr_read_b32 v187, a73
	v_accvgpr_read_b32 v186, a72
	v_ashrrev_i32_e32 v135, 31, v134
	v_accvgpr_read_b32 v177, a87
	v_accvgpr_read_b32 v180, a82
	v_accvgpr_read_b32 v193, a71
	v_accvgpr_read_b32 v233, a31
	v_pk_mul_f32 v[246:247], v[186:187], s[26:27] op_sel_hi:[1,0]
	v_lshlrev_b64 v[186:187], 1, v[134:135]
	v_add_u32_e32 v134, s28, v134
	v_accvgpr_read_b32 v176, a86
	v_accvgpr_read_b32 v175, a85
	v_accvgpr_read_b32 v174, a84
	v_accvgpr_read_b32 v188, a74
	v_accvgpr_read_b32 v192, a70
	v_accvgpr_read_b32 v231, a29
	v_accvgpr_read_b32 v230, a28
	v_pk_mul_f32 v[252:253], v[180:181], s[26:27] op_sel_hi:[1,0]
	v_add_u32_e32 v180, s28, v134
	v_accvgpr_read_b32 v191, a69
	v_accvgpr_read_b32 v190, a68
	v_accvgpr_read_b32 v225, a39
	v_accvgpr_read_b32 v229, a35
	v_pk_mul_f32 v[230:231], v[230:231], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[240:241], v[192:193], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[244:245], v[188:189], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[188:189], v[176:177], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[192:193], v[174:175], s[26:27] op_sel_hi:[1,0]
	v_lshl_add_u64 v[174:175], v[182:183], 0, v[4:5]
	v_lshl_add_u64 v[176:177], v[182:183], 0, v[186:187]
	v_ashrrev_i32_e32 v135, 31, v134
	v_ashrrev_i32_e32 v181, 31, v180
	v_accvgpr_read_b32 v179, a81
	v_accvgpr_read_b32 v178, a80
	v_accvgpr_read_b32 v184, a78
	v_accvgpr_read_b32 v221, a43
	v_accvgpr_read_b32 v223, a37
	v_accvgpr_read_b32 v222, a36
	v_accvgpr_read_b32 v227, a33
	v_accvgpr_read_b32 v226, a32
	v_accvgpr_read_b32 v232, a30
	v_pk_mul_f32 v[242:243], v[190:191], s[26:27] op_sel_hi:[1,0]
	global_store_short_d16_hi v[174:175], v230, off
	global_store_short_d16_hi v[176:177], v231, off
	v_lshlrev_b64 v[230:231], 1, v[134:135]
	v_lshlrev_b64 v[190:191], 1, v[180:181]
	v_accvgpr_read_b32 v220, a42
	v_accvgpr_read_b32 v219, a41
	v_accvgpr_read_b32 v218, a40
	v_accvgpr_read_b32 v224, a38
	v_accvgpr_read_b32 v228, a34
	v_pk_mul_f32 v[136:137], v[232:233], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[226:227], v[226:227], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[222:223], v[222:223], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[248:249], v[184:185], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[254:255], v[178:179], s[26:27] op_sel_hi:[1,0]
	v_lshl_add_u64 v[178:179], v[182:183], 0, v[230:231]
	v_lshl_add_u64 v[184:185], v[182:183], 0, v[190:191]
	s_mul_i32 s0, s28, 13
	v_pk_mul_f32 v[228:229], v[228:229], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[224:225], v[224:225], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[220:221], v[220:221], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[218:219], v[218:219], s[26:27] op_sel_hi:[1,0]
	global_store_short_d16_hi v[178:179], v136, off
	global_store_short_d16_hi v[184:185], v137, off
	global_store_short_d16_hi v[174:175], v226, off offset:32
	global_store_short_d16_hi v[176:177], v227, off offset:32
	global_store_short_d16_hi v[178:179], v228, off offset:32
	global_store_short_d16_hi v[184:185], v229, off offset:32
	global_store_short_d16_hi v[174:175], v222, off offset:64
	global_store_short_d16_hi v[176:177], v223, off offset:64
	global_store_short_d16_hi v[178:179], v224, off offset:64
	global_store_short_d16_hi v[184:185], v225, off offset:64
	global_store_short_d16_hi v[174:175], v218, off offset:96
	global_store_short_d16_hi v[176:177], v219, off offset:96
	global_store_short_d16_hi v[178:179], v220, off offset:96
	global_store_short_d16_hi v[184:185], v221, off offset:96
	v_add_u32_e32 v174, s0, v180
	v_ashrrev_i32_e32 v175, 31, v174
	v_lshlrev_b64 v[224:225], 1, v[174:175]
	v_add_u32_e32 v174, s28, v174
	v_ashrrev_i32_e32 v175, 31, v174
	v_lshlrev_b64 v[226:227], 1, v[174:175]
	v_add_u32_e32 v174, s28, v174
	v_accvgpr_read_b32 v201, a63
	v_accvgpr_read_b32 v217, a47
	v_add_u32_e32 v218, s28, v174
	v_accvgpr_read_b32 v200, a62
	v_accvgpr_read_b32 v199, a61
	v_accvgpr_read_b32 v198, a60
	v_accvgpr_read_b32 v209, a55
	v_accvgpr_read_b32 v213, a51
	v_accvgpr_read_b32 v215, a45
	v_accvgpr_read_b32 v214, a44
	v_ashrrev_i32_e32 v175, 31, v174
	v_ashrrev_i32_e32 v219, 31, v218
	v_accvgpr_read_b32 v205, a59
	v_accvgpr_read_b32 v207, a53
	v_accvgpr_read_b32 v206, a52
	v_accvgpr_read_b32 v211, a49
	v_accvgpr_read_b32 v210, a48
	v_accvgpr_read_b32 v216, a46
	v_pk_mul_f32 v[214:215], v[214:215], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[232:233], v[200:201], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[234:235], v[198:199], s[26:27] op_sel_hi:[1,0]
	v_lshl_add_u64 v[184:185], v[182:183], 0, v[224:225]
	v_lshl_add_u64 v[200:201], v[182:183], 0, v[226:227]
	v_lshlrev_b64 v[228:229], 1, v[174:175]
	v_lshlrev_b64 v[198:199], 1, v[218:219]
	v_accvgpr_read_b32 v204, a58
	v_accvgpr_read_b32 v203, a57
	v_accvgpr_read_b32 v202, a56
	v_accvgpr_read_b32 v208, a54
	v_accvgpr_read_b32 v212, a50
	v_pk_mul_f32 v[216:217], v[216:217], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[210:211], v[210:211], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[206:207], v[206:207], s[26:27] op_sel_hi:[1,0]
	global_store_short_d16_hi v[184:185], v214, off
	global_store_short_d16_hi v[200:201], v215, off
	v_lshl_add_u64 v[214:215], v[182:183], 0, v[228:229]
	v_lshl_add_u64 v[220:221], v[182:183], 0, v[198:199]
	v_pk_mul_f32 v[212:213], v[212:213], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[208:209], v[208:209], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[204:205], v[204:205], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[202:203], v[202:203], s[26:27] op_sel_hi:[1,0]
	global_store_short_d16_hi v[214:215], v216, off
	global_store_short_d16_hi v[220:221], v217, off
	global_store_short_d16_hi v[184:185], v210, off offset:32
	global_store_short_d16_hi v[200:201], v211, off offset:32
	global_store_short_d16_hi v[214:215], v212, off offset:32
	global_store_short_d16_hi v[220:221], v213, off offset:32
	global_store_short_d16_hi v[184:185], v206, off offset:64
	global_store_short_d16_hi v[200:201], v207, off offset:64
	global_store_short_d16_hi v[214:215], v208, off offset:64
	global_store_short_d16_hi v[220:221], v209, off offset:64
	global_store_short_d16_hi v[184:185], v202, off offset:96
	global_store_short_d16_hi v[200:201], v203, off offset:96
	global_store_short_d16_hi v[214:215], v204, off offset:96
	global_store_short_d16_hi v[220:221], v205, off offset:96
	v_add_u32_e32 v184, s0, v218
	v_ashrrev_i32_e32 v185, 31, v184
	v_lshlrev_b64 v[200:201], 1, v[184:185]
	v_add_u32_e32 v184, s28, v184
	v_ashrrev_i32_e32 v185, 31, v184
	v_lshlrev_b64 v[202:203], 1, v[184:185]
	v_add_u32_e32 v184, s28, v184
	v_ashrrev_i32_e32 v185, 31, v184
	v_lshlrev_b64 v[204:205], 1, v[184:185]
	v_add_u32_e32 v184, s28, v184
	v_accvgpr_read_b32 v197, a67
	v_ashrrev_i32_e32 v185, 31, v184
	v_accvgpr_read_b32 v195, a65
	v_accvgpr_read_b32 v194, a64
	v_lshlrev_b64 v[206:207], 1, v[184:185]
	v_add_u32_e32 v184, s0, v184
	v_accvgpr_read_b32 v196, a66
	v_pk_mul_f32 v[238:239], v[194:195], s[26:27] op_sel_hi:[1,0]
	v_lshl_add_u64 v[208:209], v[182:183], 0, v[200:201]
	v_lshl_add_u64 v[210:211], v[182:183], 0, v[202:203]
	v_lshl_add_u64 v[212:213], v[182:183], 0, v[204:205]
	v_lshl_add_u64 v[214:215], v[182:183], 0, v[206:207]
	v_ashrrev_i32_e32 v185, 31, v184
	v_pk_mul_f32 v[236:237], v[196:197], s[26:27] op_sel_hi:[1,0]
	global_store_short_d16_hi v[208:209], v234, off
	global_store_short_d16_hi v[210:211], v235, off
	global_store_short_d16_hi v[212:213], v232, off
	global_store_short_d16_hi v[214:215], v233, off
	global_store_short_d16_hi v[208:209], v238, off offset:32
	global_store_short_d16_hi v[210:211], v239, off offset:32
	global_store_short_d16_hi v[212:213], v236, off offset:32
	global_store_short_d16_hi v[214:215], v237, off offset:32
	global_store_short_d16_hi v[208:209], v242, off offset:64
	global_store_short_d16_hi v[210:211], v243, off offset:64
	global_store_short_d16_hi v[212:213], v240, off offset:64
	global_store_short_d16_hi v[214:215], v241, off offset:64
	global_store_short_d16_hi v[208:209], v246, off offset:96
	global_store_short_d16_hi v[210:211], v247, off offset:96
	global_store_short_d16_hi v[212:213], v244, off offset:96
	global_store_short_d16_hi v[214:215], v245, off offset:96
	v_lshlrev_b64 v[208:209], 1, v[184:185]
	v_add_u32_e32 v184, s28, v184
	v_ashrrev_i32_e32 v185, 31, v184
	v_lshlrev_b64 v[210:211], 1, v[184:185]
	v_add_u32_e32 v184, s28, v184
	v_ashrrev_i32_e32 v185, 31, v184
	v_lshlrev_b64 v[212:213], 1, v[184:185]
	v_add_u32_e32 v184, s28, v184
	v_ashrrev_i32_e32 v185, 31, v184
	v_accvgpr_read_b32 v149, a91
	v_lshlrev_b64 v[214:215], 1, v[184:185]
	v_or_b32_e32 v0, 2, v1
	v_accvgpr_read_b32 v148, a90
	v_accvgpr_read_b32 v147, a89
	v_accvgpr_read_b32 v146, a88
	v_lshl_add_u64 v[236:237], v[182:183], 0, v[208:209]
	v_lshl_add_u64 v[238:239], v[182:183], 0, v[210:211]
	v_lshl_add_u64 v[240:241], v[182:183], 0, v[212:213]
	v_lshl_add_u64 v[182:183], v[182:183], 0, v[214:215]
	v_mul_lo_u32 v0, v0, s28
	v_pk_mul_f32 v[194:195], v[148:149], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[196:197], v[146:147], s[26:27] op_sel_hi:[1,0]
	global_store_short_d16_hi v[236:237], v250, off
	global_store_short_d16_hi v[238:239], v251, off
	global_store_short_d16_hi v[240:241], v248, off
	global_store_short_d16_hi v[182:183], v249, off
	global_store_short_d16_hi v[236:237], v254, off offset:32
	global_store_short_d16_hi v[238:239], v255, off offset:32
	global_store_short_d16_hi v[240:241], v252, off offset:32
	global_store_short_d16_hi v[182:183], v253, off offset:32
	global_store_short_d16_hi v[236:237], v192, off offset:64
	global_store_short_d16_hi v[238:239], v193, off offset:64
	global_store_short_d16_hi v[240:241], v188, off offset:64
	global_store_short_d16_hi v[182:183], v189, off offset:64
	global_store_short_d16_hi v[236:237], v196, off offset:96
	global_store_short_d16_hi v[238:239], v197, off offset:96
	global_store_short_d16_hi v[240:241], v194, off offset:96
	global_store_short_d16_hi v[182:183], v195, off offset:96
	v_add_lshl_u32 v182, v0, v2, 6
	v_accvgpr_read_b32 v173, a131
	v_ashrrev_i32_e32 v183, 31, v182
	v_accvgpr_read_b32 v161, a143
	v_accvgpr_read_b32 v171, a129
	v_accvgpr_read_b32 v170, a128
	v_lshl_add_u64 v[182:183], v[182:183], 1, s[24:25]
	v_accvgpr_read_b32 v157, a147
	v_accvgpr_read_b32 v160, a142
	v_accvgpr_read_b32 v159, a141
	v_accvgpr_read_b32 v158, a140
	v_accvgpr_read_b32 v165, a139
	v_accvgpr_read_b32 v169, a135
	v_accvgpr_read_b32 v172, a130
	v_pk_mul_f32 v[170:171], v[170:171], s[26:27] op_sel_hi:[1,0]
	v_lshl_add_u64 v[184:185], v[182:183], 0, v[4:5]
	v_lshl_add_u64 v[188:189], v[182:183], 0, v[186:187]
	v_accvgpr_read_b32 v138, a156
	v_accvgpr_read_b32 v155, a145
	v_accvgpr_read_b32 v154, a144
	v_accvgpr_read_b32 v164, a138
	v_accvgpr_read_b32 v163, a137
	v_accvgpr_read_b32 v162, a136
	v_accvgpr_read_b32 v168, a134
	v_accvgpr_read_b32 v167, a133
	v_accvgpr_read_b32 v166, a132
	v_pk_mul_f32 v[172:173], v[172:173], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[160:161], v[160:161], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[158:159], v[158:159], s[26:27] op_sel_hi:[1,0]
	global_store_short_d16_hi v[184:185], v170, off
	global_store_short_d16_hi v[188:189], v171, off
	v_lshl_add_u64 v[170:171], v[182:183], 0, v[230:231]
	v_lshl_add_u64 v[192:193], v[182:183], 0, v[190:191]
	v_accvgpr_read_b32 v130, a160
	v_accvgpr_read_b32 v139, a157
	v_accvgpr_read_b32 v140, a158
	v_accvgpr_read_b32 v141, a159
	v_accvgpr_read_b32 v142, a152
	v_accvgpr_read_b32 v153, a151
	v_accvgpr_read_b32 v156, a146
	v_pk_mul_f32 v[168:169], v[168:169], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[166:167], v[166:167], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[164:165], v[164:165], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[162:163], v[162:163], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[154:155], v[154:155], s[26:27] op_sel_hi:[1,0]
	global_store_short_d16_hi v[170:171], v172, off
	global_store_short_d16_hi v[192:193], v173, off
	global_store_short_d16_hi v[184:185], v166, off offset:32
	global_store_short_d16_hi v[188:189], v167, off offset:32
	global_store_short_d16_hi v[170:171], v168, off offset:32
	global_store_short_d16_hi v[192:193], v169, off offset:32
	global_store_short_d16_hi v[184:185], v162, off offset:64
	global_store_short_d16_hi v[188:189], v163, off offset:64
	global_store_short_d16_hi v[170:171], v164, off offset:64
	global_store_short_d16_hi v[192:193], v165, off offset:64
	global_store_short_d16_hi v[184:185], v158, off offset:96
	global_store_short_d16_hi v[188:189], v159, off offset:96
	global_store_short_d16_hi v[170:171], v160, off offset:96
	global_store_short_d16_hi v[192:193], v161, off offset:96
	v_lshl_add_u64 v[158:159], v[182:183], 0, v[224:225]
	v_lshl_add_u64 v[160:161], v[182:183], 0, v[226:227]
	v_accvgpr_read_b32 v131, a161
	v_accvgpr_read_b32 v143, a153
	v_accvgpr_read_b32 v144, a154
	v_accvgpr_read_b32 v145, a155
	v_accvgpr_read_b32 v152, a150
	v_accvgpr_read_b32 v151, a149
	v_accvgpr_read_b32 v150, a148
	v_pk_mul_f32 v[156:157], v[156:157], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[140:141], v[140:141], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[138:139], v[138:139], s[26:27] op_sel_hi:[1,0]
	global_store_short_d16_hi v[158:159], v154, off
	global_store_short_d16_hi v[160:161], v155, off
	v_lshl_add_u64 v[154:155], v[182:183], 0, v[228:229]
	v_lshl_add_u64 v[162:163], v[182:183], 0, v[198:199]
	v_accvgpr_read_b32 v118, a172
	v_accvgpr_read_b32 v122, a168
	v_accvgpr_read_b32 v126, a164
	v_accvgpr_read_b32 v132, a162
	v_accvgpr_read_b32 v133, a163
	v_pk_mul_f32 v[152:153], v[152:153], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[150:151], v[150:151], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[144:145], v[144:145], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[142:143], v[142:143], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[130:131], v[130:131], s[26:27] op_sel_hi:[1,0]
	global_store_short_d16_hi v[154:155], v156, off
	global_store_short_d16_hi v[162:163], v157, off
	global_store_short_d16_hi v[158:159], v150, off offset:32
	global_store_short_d16_hi v[160:161], v151, off offset:32
	global_store_short_d16_hi v[154:155], v152, off offset:32
	global_store_short_d16_hi v[162:163], v153, off offset:32
	global_store_short_d16_hi v[158:159], v142, off offset:64
	global_store_short_d16_hi v[160:161], v143, off offset:64
	global_store_short_d16_hi v[154:155], v144, off offset:64
	global_store_short_d16_hi v[162:163], v145, off offset:64
	global_store_short_d16_hi v[158:159], v138, off offset:96
	global_store_short_d16_hi v[160:161], v139, off offset:96
	global_store_short_d16_hi v[154:155], v140, off offset:96
	global_store_short_d16_hi v[162:163], v141, off offset:96
	v_lshl_add_u64 v[138:139], v[182:183], 0, v[200:201]
	v_lshl_add_u64 v[140:141], v[182:183], 0, v[202:203]
	v_accvgpr_read_b32 v114, a176
	v_accvgpr_read_b32 v119, a173
	v_accvgpr_read_b32 v120, a174
	v_accvgpr_read_b32 v121, a175
	v_accvgpr_read_b32 v123, a169
	v_accvgpr_read_b32 v124, a170
	v_accvgpr_read_b32 v125, a171
	v_accvgpr_read_b32 v127, a165
	v_accvgpr_read_b32 v128, a166
	v_accvgpr_read_b32 v129, a167
	v_pk_mul_f32 v[132:133], v[132:133], s[26:27] op_sel_hi:[1,0]
	global_store_short_d16_hi v[138:139], v130, off
	global_store_short_d16_hi v[140:141], v131, off
	v_lshl_add_u64 v[130:131], v[182:183], 0, v[204:205]
	v_lshl_add_u64 v[142:143], v[182:183], 0, v[206:207]
	v_or_b32_e32 v1, 2, v2
	v_accvgpr_read_b32 v102, a188
	v_accvgpr_read_b32 v115, a177
	v_pk_mul_f32 v[128:129], v[128:129], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[126:127], v[126:127], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[124:125], v[124:125], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[122:123], v[122:123], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[120:121], v[120:121], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[118:119], v[118:119], s[26:27] op_sel_hi:[1,0]
	global_store_short_d16_hi v[130:131], v132, off
	global_store_short_d16_hi v[142:143], v133, off
	global_store_short_d16_hi v[138:139], v126, off offset:32
	global_store_short_d16_hi v[140:141], v127, off offset:32
	global_store_short_d16_hi v[130:131], v128, off offset:32
	global_store_short_d16_hi v[142:143], v129, off offset:32
	global_store_short_d16_hi v[138:139], v122, off offset:64
	global_store_short_d16_hi v[140:141], v123, off offset:64
	global_store_short_d16_hi v[130:131], v124, off offset:64
	global_store_short_d16_hi v[142:143], v125, off offset:64
	global_store_short_d16_hi v[138:139], v118, off offset:96
	global_store_short_d16_hi v[140:141], v119, off offset:96
	global_store_short_d16_hi v[130:131], v120, off offset:96
	global_store_short_d16_hi v[142:143], v121, off offset:96
	v_add_lshl_u32 v130, v3, v1, 6
	v_accvgpr_read_b32 v104, a190
	v_accvgpr_read_b32 v105, a191
	v_accvgpr_read_b32 v106, a184
	v_accvgpr_read_b32 v110, a180
	v_accvgpr_read_b32 v116, a178
	v_accvgpr_read_b32 v117, a179
	v_accvgpr_read_b32 v137, a27
	v_accvgpr_read_b32 v223, a3
	v_pk_mul_f32 v[114:115], v[114:115], s[26:27] op_sel_hi:[1,0]
	v_lshl_add_u64 v[118:119], v[182:183], 0, v[208:209]
	v_lshl_add_u64 v[120:121], v[182:183], 0, v[210:211]
	v_ashrrev_i32_e32 v131, 31, v130
	v_accvgpr_read_b32 v103, a189
	v_accvgpr_read_b32 v107, a185
	v_accvgpr_read_b32 v108, a186
	v_accvgpr_read_b32 v109, a187
	v_accvgpr_read_b32 v111, a181
	v_accvgpr_read_b32 v112, a182
	v_accvgpr_read_b32 v113, a183
	v_accvgpr_read_b32 v135, a25
	v_accvgpr_read_b32 v134, a24
	v_accvgpr_read_b32 v219, a7
	v_accvgpr_read_b32 v221, a1
	v_accvgpr_read_b32 v220, a0
	v_pk_mul_f32 v[116:117], v[116:117], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[104:105], v[104:105], s[26:27] op_sel_hi:[1,0]
	global_store_short_d16_hi v[118:119], v114, off
	global_store_short_d16_hi v[120:121], v115, off
	v_lshl_add_u64 v[114:115], v[182:183], 0, v[212:213]
	v_lshl_add_u64 v[122:123], v[182:183], 0, v[214:215]
	v_lshl_add_u64 v[130:131], v[130:131], 1, s[24:25]
	v_accvgpr_read_b32 v101, a95
	v_accvgpr_read_b32 v136, a26
	v_accvgpr_read_b32 v177, a19
	v_accvgpr_read_b32 v181, a15
	v_accvgpr_read_b32 v235, a11
	v_accvgpr_read_b32 v218, a6
	v_accvgpr_read_b32 v217, a5
	v_accvgpr_read_b32 v216, a4
	v_accvgpr_read_b32 v222, a2
	v_pk_mul_f32 v[112:113], v[112:113], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[110:111], v[110:111], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[108:109], v[108:109], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[106:107], v[106:107], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[102:103], v[102:103], s[26:27] op_sel_hi:[1,0]
	global_store_short_d16_hi v[114:115], v116, off
	global_store_short_d16_hi v[122:123], v117, off
	global_store_short_d16_hi v[118:119], v110, off offset:32
	global_store_short_d16_hi v[120:121], v111, off offset:32
	global_store_short_d16_hi v[114:115], v112, off offset:32
	global_store_short_d16_hi v[122:123], v113, off offset:32
	global_store_short_d16_hi v[118:119], v106, off offset:64
	global_store_short_d16_hi v[120:121], v107, off offset:64
	global_store_short_d16_hi v[114:115], v108, off offset:64
	global_store_short_d16_hi v[122:123], v109, off offset:64
	global_store_short_d16_hi v[118:119], v102, off offset:96
	global_store_short_d16_hi v[120:121], v103, off offset:96
	global_store_short_d16_hi v[114:115], v104, off offset:96
	global_store_short_d16_hi v[122:123], v105, off offset:96
	v_pk_mul_f32 v[104:105], v[220:221], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[128:129], v[134:135], s[26:27] op_sel_hi:[1,0]
	v_lshl_add_u64 v[132:133], v[130:131], 0, v[4:5]
	v_lshl_add_u64 v[134:135], v[130:131], 0, v[186:187]
	v_accvgpr_read_b32 v94, a96
	v_accvgpr_read_b32 v100, a94
	v_accvgpr_read_b32 v99, a93
	v_accvgpr_read_b32 v98, a92
	v_accvgpr_read_b32 v149, a23
	v_accvgpr_read_b32 v176, a18
	v_accvgpr_read_b32 v175, a17
	v_accvgpr_read_b32 v174, a16
	v_accvgpr_read_b32 v180, a14
	v_accvgpr_read_b32 v179, a13
	v_accvgpr_read_b32 v178, a12
	v_accvgpr_read_b32 v234, a10
	v_accvgpr_read_b32 v233, a9
	v_accvgpr_read_b32 v232, a8
	v_pk_mul_f32 v[102:103], v[222:223], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[106:107], v[218:219], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[108:109], v[216:217], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[126:127], v[136:137], s[26:27] op_sel_hi:[1,0]
	global_store_short_d16_hi v[132:133], v104, off
	global_store_short_d16_hi v[134:135], v105, off
	v_lshl_add_u64 v[104:105], v[130:131], 0, v[230:231]
	v_lshl_add_u64 v[136:137], v[130:131], 0, v[190:191]
	v_accvgpr_read_b32 v82, a108
	v_accvgpr_read_b32 v95, a97
	v_accvgpr_read_b32 v148, a22
	v_accvgpr_read_b32 v147, a21
	v_accvgpr_read_b32 v146, a20
	v_pk_mul_f32 v[110:111], v[234:235], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[112:113], v[232:233], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[114:115], v[180:181], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[116:117], v[178:179], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[118:119], v[176:177], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[120:121], v[174:175], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[100:101], v[100:101], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[98:99], v[98:99], s[26:27] op_sel_hi:[1,0]
	global_store_short_d16_hi v[104:105], v102, off
	global_store_short_d16_hi v[136:137], v103, off
	global_store_short_d16_hi v[132:133], v108, off offset:32
	global_store_short_d16_hi v[134:135], v109, off offset:32
	global_store_short_d16_hi v[104:105], v106, off offset:32
	global_store_short_d16_hi v[136:137], v107, off offset:32
	global_store_short_d16_hi v[132:133], v112, off offset:64
	global_store_short_d16_hi v[134:135], v113, off offset:64
	global_store_short_d16_hi v[104:105], v110, off offset:64
	global_store_short_d16_hi v[136:137], v111, off offset:64
	global_store_short_d16_hi v[132:133], v116, off offset:96
	global_store_short_d16_hi v[134:135], v117, off offset:96
	global_store_short_d16_hi v[104:105], v114, off offset:96
	global_store_short_d16_hi v[136:137], v115, off offset:96
	v_lshl_add_u64 v[102:103], v[130:131], 0, v[224:225]
	v_lshl_add_u64 v[104:105], v[130:131], 0, v[226:227]
	v_lshl_add_u64 v[106:107], v[130:131], 0, v[228:229]
	v_lshl_add_u64 v[108:109], v[130:131], 0, v[198:199]
	v_accvgpr_read_b32 v78, a112
	v_accvgpr_read_b32 v83, a109
	v_accvgpr_read_b32 v84, a110
	v_accvgpr_read_b32 v85, a111
	v_accvgpr_read_b32 v86, a104
	v_accvgpr_read_b32 v90, a100
	v_accvgpr_read_b32 v96, a98
	v_accvgpr_read_b32 v97, a99
	v_pk_mul_f32 v[122:123], v[148:149], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[124:125], v[146:147], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[94:95], v[94:95], s[26:27] op_sel_hi:[1,0]
	global_store_short_d16_hi v[102:103], v120, off
	global_store_short_d16_hi v[104:105], v121, off
	global_store_short_d16_hi v[106:107], v118, off
	global_store_short_d16_hi v[108:109], v119, off
	global_store_short_d16_hi v[102:103], v124, off offset:32
	global_store_short_d16_hi v[104:105], v125, off offset:32
	global_store_short_d16_hi v[106:107], v122, off offset:32
	global_store_short_d16_hi v[108:109], v123, off offset:32
	global_store_short_d16_hi v[102:103], v128, off offset:64
	global_store_short_d16_hi v[104:105], v129, off offset:64
	global_store_short_d16_hi v[106:107], v126, off offset:64
	global_store_short_d16_hi v[108:109], v127, off offset:64
	global_store_short_d16_hi v[102:103], v98, off offset:96
	global_store_short_d16_hi v[104:105], v99, off offset:96
	global_store_short_d16_hi v[106:107], v100, off offset:96
	global_store_short_d16_hi v[108:109], v101, off offset:96
	v_lshl_add_u64 v[98:99], v[130:131], 0, v[200:201]
	v_lshl_add_u64 v[100:101], v[130:131], 0, v[202:203]
	v_accvgpr_read_b32 v66, a124
	v_accvgpr_read_b32 v79, a113
	v_accvgpr_read_b32 v87, a105
	v_accvgpr_read_b32 v88, a106
	v_accvgpr_read_b32 v89, a107
	v_accvgpr_read_b32 v91, a101
	v_accvgpr_read_b32 v92, a102
	v_accvgpr_read_b32 v93, a103
	v_pk_mul_f32 v[96:97], v[96:97], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[84:85], v[84:85], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[82:83], v[82:83], s[26:27] op_sel_hi:[1,0]
	global_store_short_d16_hi v[98:99], v94, off
	global_store_short_d16_hi v[100:101], v95, off
	v_lshl_add_u64 v[94:95], v[130:131], 0, v[204:205]
	v_lshl_add_u64 v[102:103], v[130:131], 0, v[206:207]
	v_accvgpr_read_b32 v67, a125
	v_accvgpr_read_b32 v70, a120
	v_accvgpr_read_b32 v74, a116
	v_accvgpr_read_b32 v80, a114
	v_accvgpr_read_b32 v81, a115
	v_pk_mul_f32 v[92:93], v[92:93], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[90:91], v[90:91], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[88:89], v[88:89], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[86:87], v[86:87], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[78:79], v[78:79], s[26:27] op_sel_hi:[1,0]
	global_store_short_d16_hi v[94:95], v96, off
	global_store_short_d16_hi v[102:103], v97, off
	global_store_short_d16_hi v[98:99], v90, off offset:32
	global_store_short_d16_hi v[100:101], v91, off offset:32
	global_store_short_d16_hi v[94:95], v92, off offset:32
	global_store_short_d16_hi v[102:103], v93, off offset:32
	global_store_short_d16_hi v[98:99], v86, off offset:64
	global_store_short_d16_hi v[100:101], v87, off offset:64
	global_store_short_d16_hi v[94:95], v88, off offset:64
	global_store_short_d16_hi v[102:103], v89, off offset:64
	global_store_short_d16_hi v[98:99], v82, off offset:96
	global_store_short_d16_hi v[100:101], v83, off offset:96
	global_store_short_d16_hi v[94:95], v84, off offset:96
	global_store_short_d16_hi v[102:103], v85, off offset:96
	v_lshl_add_u64 v[82:83], v[130:131], 0, v[208:209]
	v_lshl_add_u64 v[84:85], v[130:131], 0, v[210:211]
	v_accvgpr_read_b32 v68, a126
	v_accvgpr_read_b32 v69, a127
	v_accvgpr_read_b32 v71, a121
	v_accvgpr_read_b32 v72, a122
	v_accvgpr_read_b32 v73, a123
	v_accvgpr_read_b32 v75, a117
	v_accvgpr_read_b32 v76, a118
	v_accvgpr_read_b32 v77, a119
	v_pk_mul_f32 v[80:81], v[80:81], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[66:67], v[66:67], s[26:27] op_sel_hi:[1,0]
	global_store_short_d16_hi v[82:83], v78, off
	global_store_short_d16_hi v[84:85], v79, off
	v_lshl_add_u64 v[78:79], v[130:131], 0, v[212:213]
	v_lshl_add_u64 v[86:87], v[130:131], 0, v[214:215]
	v_pk_mul_f32 v[76:77], v[76:77], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[74:75], v[74:75], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[72:73], v[72:73], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[70:71], v[70:71], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[68:69], v[68:69], s[26:27] op_sel_hi:[1,0]
	global_store_short_d16_hi v[78:79], v80, off
	global_store_short_d16_hi v[86:87], v81, off
	global_store_short_d16_hi v[82:83], v74, off offset:32
	global_store_short_d16_hi v[84:85], v75, off offset:32
	global_store_short_d16_hi v[78:79], v76, off offset:32
	global_store_short_d16_hi v[86:87], v77, off offset:32
	global_store_short_d16_hi v[82:83], v70, off offset:64
	global_store_short_d16_hi v[84:85], v71, off offset:64
	global_store_short_d16_hi v[78:79], v72, off offset:64
	global_store_short_d16_hi v[86:87], v73, off offset:64
	global_store_short_d16_hi v[82:83], v66, off offset:96
	global_store_short_d16_hi v[84:85], v67, off offset:96
	global_store_short_d16_hi v[78:79], v68, off offset:96
	global_store_short_d16_hi v[86:87], v69, off offset:96
	v_add_lshl_u32 v66, v0, v1, 6
	v_accvgpr_read_b32 v62, a192
	v_ashrrev_i32_e32 v67, 31, v66
	v_accvgpr_read_b32 v50, a204
	v_accvgpr_read_b32 v63, a193
	v_mov_b64_e32 v[138:139], v[4:5]
	v_lshl_add_u64 v[66:67], v[66:67], 1, s[24:25]
	v_accvgpr_read_b32 v46, a208
	v_accvgpr_read_b32 v51, a205
	v_accvgpr_read_b32 v54, a200
	v_accvgpr_read_b32 v58, a196
	v_accvgpr_read_b32 v64, a194
	v_accvgpr_read_b32 v65, a195
	v_pk_mul_f32 v[62:63], v[62:63], s[26:27] op_sel_hi:[1,0]
	v_lshl_add_u64 v[0:1], v[66:67], 0, v[138:139]
	v_lshl_add_u64 v[68:69], v[66:67], 0, v[186:187]
	v_accvgpr_read_b32 v34, a220
	v_accvgpr_read_b32 v47, a209
	v_accvgpr_read_b32 v52, a206
	v_accvgpr_read_b32 v53, a207
	v_accvgpr_read_b32 v55, a201
	v_accvgpr_read_b32 v56, a202
	v_accvgpr_read_b32 v57, a203
	v_accvgpr_read_b32 v59, a197
	v_accvgpr_read_b32 v60, a198
	v_accvgpr_read_b32 v61, a199
	v_pk_mul_f32 v[64:65], v[64:65], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], s[26:27] op_sel_hi:[1,0]
	global_store_short_d16_hi v[0:1], v62, off
	global_store_short_d16_hi v[68:69], v63, off
	v_lshl_add_u64 v[62:63], v[66:67], 0, v[230:231]
	v_lshl_add_u64 v[70:71], v[66:67], 0, v[190:191]
	v_accvgpr_read_b32 v30, a224
	v_accvgpr_read_b32 v35, a221
	v_accvgpr_read_b32 v38, a216
	v_accvgpr_read_b32 v42, a212
	v_accvgpr_read_b32 v48, a210
	v_accvgpr_read_b32 v49, a211
	v_pk_mul_f32 v[60:61], v[60:61], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], s[26:27] op_sel_hi:[1,0]
	global_store_short_d16_hi v[62:63], v64, off
	global_store_short_d16_hi v[70:71], v65, off
	global_store_short_d16_hi v[0:1], v58, off offset:32
	global_store_short_d16_hi v[68:69], v59, off offset:32
	global_store_short_d16_hi v[62:63], v60, off offset:32
	global_store_short_d16_hi v[70:71], v61, off offset:32
	global_store_short_d16_hi v[0:1], v54, off offset:64
	global_store_short_d16_hi v[68:69], v55, off offset:64
	global_store_short_d16_hi v[62:63], v56, off offset:64
	global_store_short_d16_hi v[70:71], v57, off offset:64
	global_store_short_d16_hi v[0:1], v50, off offset:96
	global_store_short_d16_hi v[68:69], v51, off offset:96
	global_store_short_d16_hi v[62:63], v52, off offset:96
	global_store_short_d16_hi v[70:71], v53, off offset:96
	v_lshl_add_u64 v[0:1], v[66:67], 0, v[224:225]
	v_lshl_add_u64 v[50:51], v[66:67], 0, v[226:227]
	v_accvgpr_read_b32 v18, a236
	v_accvgpr_read_b32 v31, a225
	v_accvgpr_read_b32 v36, a222
	v_accvgpr_read_b32 v37, a223
	v_accvgpr_read_b32 v39, a217
	v_accvgpr_read_b32 v40, a218
	v_accvgpr_read_b32 v41, a219
	v_accvgpr_read_b32 v43, a213
	v_accvgpr_read_b32 v44, a214
	v_accvgpr_read_b32 v45, a215
	v_pk_mul_f32 v[48:49], v[48:49], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], s[26:27] op_sel_hi:[1,0]
	global_store_short_d16_hi v[0:1], v46, off
	global_store_short_d16_hi v[50:51], v47, off
	v_lshl_add_u64 v[46:47], v[66:67], 0, v[228:229]
	v_lshl_add_u64 v[52:53], v[66:67], 0, v[198:199]
	v_accvgpr_read_b32 v14, a240
	v_accvgpr_read_b32 v19, a237
	v_accvgpr_read_b32 v22, a232
	v_accvgpr_read_b32 v26, a228
	v_accvgpr_read_b32 v32, a226
	v_accvgpr_read_b32 v33, a227
	v_pk_mul_f32 v[44:45], v[44:45], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], s[26:27] op_sel_hi:[1,0]
	global_store_short_d16_hi v[46:47], v48, off
	global_store_short_d16_hi v[52:53], v49, off
	global_store_short_d16_hi v[0:1], v42, off offset:32
	global_store_short_d16_hi v[50:51], v43, off offset:32
	global_store_short_d16_hi v[46:47], v44, off offset:32
	global_store_short_d16_hi v[52:53], v45, off offset:32
	global_store_short_d16_hi v[0:1], v38, off offset:64
	global_store_short_d16_hi v[50:51], v39, off offset:64
	global_store_short_d16_hi v[46:47], v40, off offset:64
	global_store_short_d16_hi v[52:53], v41, off offset:64
	global_store_short_d16_hi v[0:1], v34, off offset:96
	global_store_short_d16_hi v[50:51], v35, off offset:96
	global_store_short_d16_hi v[46:47], v36, off offset:96
	global_store_short_d16_hi v[52:53], v37, off offset:96
	v_lshl_add_u64 v[0:1], v[66:67], 0, v[200:201]
	v_lshl_add_u64 v[34:35], v[66:67], 0, v[202:203]
	v_accvgpr_read_b32 v15, a241
	v_accvgpr_read_b32 v20, a238
	v_accvgpr_read_b32 v21, a239
	v_accvgpr_read_b32 v23, a233
	v_accvgpr_read_b32 v24, a234
	v_accvgpr_read_b32 v25, a235
	v_accvgpr_read_b32 v27, a229
	v_accvgpr_read_b32 v28, a230
	v_accvgpr_read_b32 v29, a231
	v_pk_mul_f32 v[32:33], v[32:33], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], s[26:27] op_sel_hi:[1,0]
	global_store_short_d16_hi v[0:1], v30, off
	global_store_short_d16_hi v[34:35], v31, off
	v_lshl_add_u64 v[30:31], v[66:67], 0, v[204:205]
	v_lshl_add_u64 v[36:37], v[66:67], 0, v[206:207]
	v_accvgpr_read_b32 v6, a248
	v_accvgpr_read_b32 v10, a244
	v_accvgpr_read_b32 v16, a242
	v_accvgpr_read_b32 v17, a243
	v_pk_mul_f32 v[28:29], v[28:29], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], s[26:27] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v2, a252
	global_store_short_d16_hi v[30:31], v32, off
	global_store_short_d16_hi v[36:37], v33, off
	global_store_short_d16_hi v[0:1], v26, off offset:32
	global_store_short_d16_hi v[34:35], v27, off offset:32
	global_store_short_d16_hi v[30:31], v28, off offset:32
	global_store_short_d16_hi v[36:37], v29, off offset:32
	global_store_short_d16_hi v[0:1], v22, off offset:64
	global_store_short_d16_hi v[34:35], v23, off offset:64
	global_store_short_d16_hi v[30:31], v24, off offset:64
	global_store_short_d16_hi v[36:37], v25, off offset:64
	global_store_short_d16_hi v[0:1], v18, off offset:96
	global_store_short_d16_hi v[34:35], v19, off offset:96
	global_store_short_d16_hi v[30:31], v20, off offset:96
	global_store_short_d16_hi v[36:37], v21, off offset:96
	v_lshl_add_u64 v[0:1], v[66:67], 0, v[208:209]
	v_lshl_add_u64 v[18:19], v[66:67], 0, v[210:211]
	v_accvgpr_read_b32 v7, a249
	v_accvgpr_read_b32 v8, a250
	v_accvgpr_read_b32 v9, a251
	v_accvgpr_read_b32 v11, a245
	v_accvgpr_read_b32 v12, a246
	v_accvgpr_read_b32 v13, a247
	v_pk_mul_f32 v[16:17], v[16:17], s[26:27] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v3, a253
	v_accvgpr_read_b32 v4, a254
	v_accvgpr_read_b32 v5, a255
	global_store_short_d16_hi v[0:1], v14, off
	global_store_short_d16_hi v[18:19], v15, off
	v_lshl_add_u64 v[14:15], v[66:67], 0, v[212:213]
	v_lshl_add_u64 v[20:21], v[66:67], 0, v[214:215]
	v_pk_mul_f32 v[12:13], v[12:13], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], s[26:27] op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], s[26:27] op_sel_hi:[1,0]
	global_store_short_d16_hi v[14:15], v16, off
	global_store_short_d16_hi v[20:21], v17, off
	global_store_short_d16_hi v[0:1], v10, off offset:32
	global_store_short_d16_hi v[18:19], v11, off offset:32
	global_store_short_d16_hi v[14:15], v12, off offset:32
	global_store_short_d16_hi v[20:21], v13, off offset:32
	global_store_short_d16_hi v[0:1], v6, off offset:64
	global_store_short_d16_hi v[18:19], v7, off offset:64
	global_store_short_d16_hi v[14:15], v8, off offset:64
	global_store_short_d16_hi v[20:21], v9, off offset:64
	global_store_short_d16_hi v[0:1], v2, off offset:96
	global_store_short_d16_hi v[18:19], v3, off offset:96
	global_store_short_d16_hi v[14:15], v4, off offset:96
	global_store_short_d16_hi v[20:21], v5, off offset:96
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
	.set _Z22mxfp4_gluon_cpp_kernel13gluon_globals.num_vgpr, 256
	.set _Z22mxfp4_gluon_cpp_kernel13gluon_globals.num_agpr, 256
	.set _Z22mxfp4_gluon_cpp_kernel13gluon_globals.numbered_sgpr, 79
	.set _Z22mxfp4_gluon_cpp_kernel13gluon_globals.private_seg_size, 0
	.set _Z22mxfp4_gluon_cpp_kernel13gluon_globals.uses_vcc, 0
	.set _Z22mxfp4_gluon_cpp_kernel13gluon_globals.uses_flat_scratch, 0
	.set _Z22mxfp4_gluon_cpp_kernel13gluon_globals.has_dyn_sized_stack, 0
	.set _Z22mxfp4_gluon_cpp_kernel13gluon_globals.has_recursion, 0
	.set _Z22mxfp4_gluon_cpp_kernel13gluon_globals.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 57960
; TotalNumSgprs: 85
; NumVgprs: 256
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
	.type	__hip_cuid_327ab34ed3c1cf8b,@object ; @__hip_cuid_327ab34ed3c1cf8b
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_327ab34ed3c1cf8b
__hip_cuid_327ab34ed3c1cf8b:
	.byte	0                               ; 0x0
	.size	__hip_cuid_327ab34ed3c1cf8b, 1

	.ident	"clang version 20.0.0git (https://github.com/ROCm/llvm-project.git 27682a16360e33e37c4f3cc6adf9a620733f8fe1)"
	.ident	"AMD clang version 20.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.1.0 25425 1b0eada6b0ee93e2e694c8c146d23fca90bc11c5)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_327ab34ed3c1cf8b
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
    .sgpr_count:     85
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
