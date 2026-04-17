	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.text
	.protected	_Z22mxfp4_gluon_cpp_kernel13gluon_globals ; -- Begin function _Z22mxfp4_gluon_cpp_kernel13gluon_globals
	.globl	_Z22mxfp4_gluon_cpp_kernel13gluon_globals
	.p2align	8
	.type	_Z22mxfp4_gluon_cpp_kernel13gluon_globals,@function
_Z22mxfp4_gluon_cpp_kernel13gluon_globals: ; @_Z22mxfp4_gluon_cpp_kernel13gluon_globals
; %bb.0:
	s_load_dword s4, s[0:1], 0xf8
	s_waitcnt lgkmcnt(0)
	s_ashr_i32 s5, s4, 31
	s_add_i32 s3, s4, 7
	s_ashr_i32 s6, s3, 31
	s_lshr_b32 s7, s5, 29
	s_add_i32 s7, s4, s7
	s_lshr_b32 s6, s6, 29
	s_add_i32 s3, s3, s6
	s_and_b32 s7, s7, -8
	s_ashr_i32 s6, s3, 3
	s_sub_i32 s3, s4, s7
	s_cmp_lg_u32 s3, 0
	s_cselect_b32 s3, s3, 8
	s_ashr_i32 s7, s2, 31
	s_lshr_b32 s7, s7, 29
	s_add_i32 s8, s2, s7
	s_and_b32 s7, s8, -8
	s_sub_i32 s7, s2, s7
	s_cmp_ge_i32 s7, s3
	s_cbranch_scc0 .LBB0_2
; %bb.1:
	s_add_i32 s2, s6, -1
	s_sub_i32 s9, s7, s3
	s_mul_i32 s2, s9, s2
	s_mul_i32 s3, s3, s6
	s_add_i32 s9, s2, s3
	s_ashr_i32 s2, s8, 3
	s_cbranch_execz .LBB0_3
	s_branch .LBB0_4
.LBB0_2:
                                        ; implicit-def: $sgpr9
	s_ashr_i32 s2, s8, 3
.LBB0_3:
	s_mul_i32 s9, s6, s7
.LBB0_4:
	s_add_i32 s2, s9, s2
	s_cmp_ge_i32 s2, s4
	s_cbranch_scc0 .LBB0_6
; %bb.5:
	s_endpgm
.LBB0_6:
	s_ashr_i32 s3, s2, 31
	s_lshr_b32 s6, s3, 23
	s_add_i32 s6, s2, s6
	s_lshr_b32 s5, s5, 25
	s_add_i32 s4, s4, s5
	s_ashr_i32 s4, s4, 7
	s_abs_i32 s10, s2
	s_ashr_i32 s5, s6, 9
	s_lshl_b32 s5, s5, 2
	s_sub_i32 s4, s4, s5
	s_min_i32 s4, s4, 4
	s_abs_i32 s7, s4
	s_sub_i32 s8, 0, s7
	v_lshrrev_b32_e32 v4, 7, v0
	s_load_dwordx2 s[34:35], s[0:1], 0x0
	v_cvt_f32_u32_e32 v1, s7
	s_movk_i32 s36, 0x70
	v_lshlrev_b32_e32 v5, 4, v0
	v_and_b32_e32 v72, 48, v0
	v_rcp_iflag_f32_e32 v1, v1
	v_lshlrev_b32_e32 v67, 13, v4
	v_lshlrev_b32_e32 v73, 3, v0
	v_bfe_u32 v18, v0, 6, 1
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	v_lshlrev_b32_e32 v22, 6, v18
	v_or_b32_e32 v79, 64, v72
	v_lshlrev_b32_e32 v68, 13, v18
	v_readfirstlane_b32 s9, v1
	s_mul_i32 s8, s8, s9
	s_mul_hi_u32 s8, s9, s8
	s_add_i32 s9, s9, s8
	s_mul_hi_u32 s8, s10, s9
	s_mul_i32 s8, s8, s7
	s_sub_i32 s8, s10, s8
	s_sub_i32 s10, s8, s7
	v_or_b32_e32 v69, 0x10000, v68
	v_lshrrev_b32_e32 v1, 6, v0
	v_lshlrev_b32_e32 v1, 10, v1
	s_mov_b32 s73, 0
	v_readfirstlane_b32 s25, v1
	s_add_i32 s43, s25, 0x4000
	v_accvgpr_write_b32 a144, 0
	v_lshlrev_b32_e32 v1, 6, v4
	s_add_i32 s33, s25, 0x8000
	s_add_i32 s42, s33, 0x4000
	s_add_i32 s38, s25, 0x10000
	s_add_i32 s41, s38, 0x4000
	s_add_i32 s39, s25, 0x18000
	s_add_i32 s40, s39, 0x4000
	s_cmp_ge_u32 s8, s7
	s_cselect_b32 s8, s10, s8
	s_sub_i32 s10, s8, s7
	s_cmp_ge_u32 s8, s7
	s_cselect_b32 s8, s10, s8
	s_xor_b32 s8, s8, s3
	s_sub_i32 s44, s8, s3
	s_add_i32 s44, s44, s5
	v_accvgpr_write_b32 a145, 0
	s_and_b32 s6, s6, 0xfffffe00
	s_sub_i32 s2, s2, s6
	s_abs_i32 s6, s2
	s_mul_hi_u32 s9, s6, s9
	s_add_i32 s5, s9, 1
	s_xor_b32 s2, s2, s4
	s_ashr_i32 s22, s2, 31
	s_mul_i32 s10, s9, s7
	s_sub_i32 s3, s6, s10
	s_sub_i32 s4, s3, s7
	s_cmp_ge_u32 s3, s7
	s_cselect_b32 s2, s4, s3
	s_cselect_b32 s3, s5, s9
	s_add_i32 s4, s3, 1
	s_cmp_ge_u32 s2, s7
	s_cselect_b32 s23, s4, s3
	s_load_dwordx2 s[2:3], s[0:1], 0xb0
	s_load_dwordx2 s[18:19], s[0:1], 0x90
	s_load_dwordx2 s[30:31], s[0:1], 0x30
	s_load_dwordx2 s[20:21], s[0:1], 0x50
	s_load_dwordx2 s[12:13], s[0:1], 0x20
	s_load_dwordx2 s[16:17], s[0:1], 0x80
	s_load_dwordx2 s[4:5], s[0:1], 0x60
	s_waitcnt lgkmcnt(0)
	v_mov_b64_e32 v[32:33], s[18:19]
	s_mov_b32 s6, -1
	v_accvgpr_write_b32 a146, 0
	s_lshl_b32 s15, s12, 5
	v_mov_b64_e32 v[30:31], s[4:5]
	s_lshl_b32 s14, s44, 8
	s_mov_b32 s13, s25
	s_add_i32 s46, s25, 0x2000
	v_or_b32_e32 v2, s14, v1
	v_ashrrev_i32_e32 v2, 6, v2
	v_mad_i64_i32 v[2:3], s[4:5], s16, v2, v[30:31]
	s_nop 0
	v_readfirstlane_b32 s4, v2
	v_bitop3_b32 v2, v5, s36, v0 bitop3:0x48
	s_mul_i32 s45, s12, s14
	s_or_b32 s17, s14, 0x80
	v_or_b32_e32 v26, s17, v1
	v_ashrrev_i32_e32 v70, 6, v26
	v_readfirstlane_b32 s5, v3
	v_lshrrev_b32_e32 v3, 3, v0
	v_or_b32_e32 v5, 0x60, v3
	s_mul_i32 s48, s12, s17
	s_add_u32 s72, s34, s45
	s_add_u32 s74, s34, s48
	s_mov_b32 s3, s45
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s13
	s_add_i32 s47, s25, 0x3000
	s_add_i32 s49, s33, 0x1000
	s_add_i32 s50, s33, 0x2000
	s_add_i32 s51, s33, 0x3000
	s_add_i32 s53, s38, 0x2000
	s_add_i32 s54, s38, 0x3000
	s_mov_b32 s24, s54
	s_add_i32 s56, s39, 0x1000
	s_add_i32 s57, s39, 0x2000
	s_add_i32 s58, s39, 0x3000
	s_add_i32 s59, s25, 0x5000
	s_add_i32 s60, s25, 0x6000
	s_add_i32 s61, s25, 0x7000
	s_add_i32 s62, s33, 0x5000
	s_add_i32 s63, s33, 0x6000
	s_add_i32 s64, s33, 0x7000
	s_add_i32 s65, s38, 0x5000
	s_add_i32 s66, s38, 0x6000
	s_add_i32 s67, s38, 0x7000
	s_add_i32 s68, s39, 0x5000
	s_add_i32 s69, s39, 0x6000
	s_add_i32 s70, s39, 0x7000
	s_mov_b32 s7, 0x110000
	s_mov_b64 s[10:11], s[6:7]
	s_mov_b64 s[8:9], s[4:5]
	v_mad_u64_u32 v[130:131], s[8:9], v3, s12, v[2:3]
	v_add_u32_e32 v131, s15, v130
	v_add_u32_e32 v139, s15, v131
	v_mad_u64_u32 v[132:133], s[14:15], v5, s12, v[2:3]
	s_mov_b32 s12, s48
	s_mov_b32 s8, s34
	s_mov_b32 s9, s35
	buffer_load_dwordx4 v130, s[8:11], s3 offen lds
	s_add_i32 s35, s25, 0x1000
	s_mov_b32 s13, s35
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s13
	s_mov_b32 s13, s46
	buffer_load_dwordx4 v131, s[8:11], s3 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s13
	s_mov_b32 s13, s47
	buffer_load_dwordx4 v139, s[8:11], s3 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s13
	s_xor_b32 s13, s23, s22
	s_lshl_b32 s23, s20, 5
	buffer_load_dwordx4 v132, s[8:11], s3 offen lds
	s_mov_b32 s3, s33
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s3
	s_mov_b32 s3, s49
	buffer_load_dwordx4 v130, s[8:11], s12 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s3
	s_mov_b32 s3, s50
	buffer_load_dwordx4 v131, s[8:11], s12 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s3
	s_sub_i32 s3, s13, s22
	s_lshl_b32 s3, s3, 8
	v_or_b32_e32 v138, s3, v22
	v_ashrrev_i32_e32 v71, 6, v138
	buffer_load_dwordx4 v139, s[8:11], s12 offen lds
	s_mul_i32 s52, s20, s3
	s_add_u32 s75, s30, s52
	s_or_b32 s26, s3, 0x80
	v_or_b32_e32 v27, s26, v22
	v_ashrrev_i32_e32 v75, 6, v27
	s_mov_b32 s21, s52
	s_mul_i32 s55, s20, s26
	s_add_u32 s76, s30, s55
	s_mov_b32 s22, s38
	s_mov_b32 s13, s51
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s13
	v_accvgpr_write_b32 a147, 0
	buffer_load_dwordx4 v132, s[8:11], s12 offen lds
	s_mov_b64 s[14:15], s[6:7]
	s_mov_b64 s[12:13], s[4:5]
	v_mad_u64_u32 v[134:135], s[12:13], v3, s20, v[2:3]
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s22
	s_mov_b32 s12, s30
	s_mov_b32 s13, s31
	s_add_i32 s31, s38, 0x1000
	s_mov_b32 s22, s31
	buffer_load_dwordx4 v134, s[12:15], s21 offen lds
	;;#ASMSTART
	;;#ASMEND
	v_add_u32_e32 v133, s23, v134
	s_mov_b32 m0, s22
	s_mov_b32 s22, s53
	buffer_load_dwordx4 v133, s[12:15], s21 offen lds
	;;#ASMSTART
	;;#ASMEND
	v_add_u32_e32 v135, s23, v133
	s_mov_b32 m0, s22
	v_mad_u64_u32 v[136:137], s[22:23], v5, s20, v[2:3]
	s_mov_b32 s20, s55
	buffer_load_dwordx4 v135, s[12:15], s21 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s24
	v_lshlrev_b32_e32 v2, 7, v0
	v_and_b32_e32 v66, 0x780, v2
	v_or_b32_e32 v78, v66, v69
	buffer_load_dwordx4 v136, s[12:15], s21 offen lds
	s_mov_b32 s21, s39
	v_or_b32_e32 v2, v66, v72
	v_or_b32_e32 v6, v2, v69
	v_add_u32_e32 v15, 64, v6
	v_lshrrev_b32_e32 v16, 4, v15
	v_bitop3_b32 v74, v16, v15, s36 bitop3:0x6c
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s21
	s_mov_b32 s21, s56
	v_or_b32_e32 v3, v2, v67
	v_bitop3_b32 v4, v73, v3, s36 bitop3:0x6c
	v_or_b32_e32 v3, 64, v3
	v_bitop3_b32 v3, v73, v3, s36 bitop3:0x6c
	buffer_load_dwordx4 v134, s[12:15], s20 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s21
	s_mov_b32 s21, s57
	v_lshrrev_b32_e32 v2, 4, v6
	v_bitop3_b32 v14, v2, v6, s36 bitop3:0x6c
	buffer_load_dwordx4 v133, s[12:15], s20 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s21
	s_mov_b32 s21, s58
	v_or_b32_e32 v76, v68, v66
	v_or_b32_e32 v82, 0x18000, v76
	buffer_load_dwordx4 v135, s[12:15], s20 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s21
	s_or_b32 s21, s45, 0x80
	v_or_b32_e32 v81, 0x1c000, v76
	v_or_b32_e32 v76, 0x14000, v76
	buffer_load_dwordx4 v136, s[12:15], s20 offen lds
	s_mov_b32 s20, s43
	v_or_b32_e32 v77, v66, v67
	v_or_b32_e32 v83, 0x4000, v77
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s20
	s_mov_b32 s20, s59
	v_or_b32_e32 v80, 0x8000, v77
	buffer_load_dwordx4 v130, s[8:11], s21 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s20
	s_mov_b32 s20, s60
	buffer_load_dwordx4 v131, s[8:11], s21 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s20
	s_mov_b32 s20, s61
	buffer_load_dwordx4 v139, s[8:11], s21 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s20
	s_add_i32 s20, s48, 0x80
	buffer_load_dwordx4 v132, s[8:11], s21 offen lds
	s_mov_b32 s21, s42
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s21
	s_mov_b32 s21, s62
	buffer_load_dwordx4 v130, s[8:11], s20 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s21
	s_mov_b32 s21, s63
	buffer_load_dwordx4 v131, s[8:11], s20 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s21
	s_mov_b32 s21, s64
	buffer_load_dwordx4 v139, s[8:11], s20 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s21
	s_or_b32 s21, s52, 0x80
	buffer_load_dwordx4 v132, s[8:11], s20 offen lds
	s_mov_b32 s20, s41
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s20
	s_mov_b32 s20, s65
	buffer_load_dwordx4 v134, s[12:15], s21 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s20
	s_mov_b32 s20, s66
	buffer_load_dwordx4 v133, s[12:15], s21 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s20
	s_mov_b32 s20, s67
	buffer_load_dwordx4 v135, s[12:15], s21 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s20
	s_add_i32 s20, s55, 0x80
	buffer_load_dwordx4 v136, s[12:15], s21 offen lds
	s_mov_b32 s21, s40
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s21
	s_mov_b32 s21, s68
	buffer_load_dwordx4 v134, s[12:15], s20 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s21
	s_mov_b32 s21, s69
	buffer_load_dwordx4 v133, s[12:15], s20 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s21
	s_mov_b32 s21, s70
	buffer_load_dwordx4 v135, s[12:15], s20 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s21
	v_and_b32_e32 v137, 0x1f8, v73
	buffer_load_dwordx4 v136, s[12:15], s20 offen lds
	s_mov_b64 s[22:23], s[6:7]
	s_mov_b64 s[20:21], s[4:5]
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
	ds_read_b128 v[18:21], v74 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[22:25], v74 offset:0x800

	;;#ASMEND
	s_load_dword s24, s[0:1], 0xf0
	s_load_dwordx2 s[26:27], s[0:1], 0xe0
	s_load_dwordx2 s[28:29], s[0:1], 0xc0
	v_accvgpr_write_b32 a160, 0
	v_mad_i64_i32 v[68:69], s[0:1], s2, v71, v[32:33]
	;;#ASMSTART
	ds_read_b128 v[26:29], v74 offset:0x1000

	;;#ASMEND
	v_mad_i64_i32 v[66:67], s[0:1], s16, v70, v[30:31]
	s_nop 0
	v_readfirstlane_b32 s20, v66
	v_or_b32_e32 v66, v77, v72
	v_bitop3_b32 v149, v73, v66, s36 bitop3:0x6c
	v_readfirstlane_b32 s21, v67
	v_accvgpr_write_b32 a161, 0
	v_mad_i64_i32 v[70:71], s[0:1], s2, v75, v[32:33]
	s_mov_b64 s[0:1], s[4:5]
	v_readfirstlane_b32 s0, v70
	v_or_b32_e32 v70, v76, v72
	v_bitop3_b32 v159, v73, v70, s36 bitop3:0x6c
	v_or_b32_e32 v67, v77, v79
	v_bitop3_b32 v148, v73, v67, s36 bitop3:0x6c
	s_mov_b64 s[2:3], s[6:7]
	v_readfirstlane_b32 s1, v71
	v_accvgpr_write_b32 a162, 0
	;;#ASMSTART
	ds_read_b128 v[30:33], v74 offset:0x1800

	;;#ASMEND
	v_add_u32_e32 v74, v78, v79
	v_lshrrev_b32_e32 v84, 4, v74
	v_bitop3_b32 v162, v84, v74, s36 bitop3:0x6c
	v_or_b32_e32 v71, v76, v79
	v_bitop3_b32 v158, v73, v71, s36 bitop3:0x6c
	v_accvgpr_write_b32 a163, 0
	v_or_b32_e32 v76, v82, v72
	v_bitop3_b32 v157, v73, v76, s36 bitop3:0x6c
	v_add_u32_e32 v75, v78, v72
	v_or_b32_e32 v78, 0xc000, v77
	v_or_b32_e32 v77, v82, v79
	v_bitop3_b32 v156, v73, v77, s36 bitop3:0x6c
	s_mov_b64 s[18:19], s[6:7]
	s_mov_b64 s[16:17], s[4:5]
	v_readfirstlane_b32 s16, v68
	v_or_b32_e32 v68, v83, v72
	v_bitop3_b32 v161, v73, v68, s36 bitop3:0x6c
	v_or_b32_e32 v82, v81, v72
	v_bitop3_b32 v155, v73, v82, s36 bitop3:0x6c
	v_readfirstlane_b32 s17, v69
	v_accvgpr_write_b32 a172, 0
	v_or_b32_e32 v81, v81, v79
	v_bitop3_b32 v154, v73, v81, s36 bitop3:0x6c
	v_or_b32_e32 v69, v83, v79
	v_bitop3_b32 v160, v73, v69, s36 bitop3:0x6c
	v_or_b32_e32 v83, v80, v72
	v_bitop3_b32 v153, v73, v83, s36 bitop3:0x6c
	v_or_b32_e32 v80, v80, v79
	v_bitop3_b32 v152, v73, v80, s36 bitop3:0x6c
	v_or_b32_e32 v72, v78, v72
	v_bitop3_b32 v151, v73, v72, s36 bitop3:0x6c
	v_or_b32_e32 v78, v78, v79
	v_bitop3_b32 v150, v73, v78, s36 bitop3:0x6c
	v_lshrrev_b32_e32 v79, 4, v75
	v_bitop3_b32 v163, v79, v75, s36 bitop3:0x6c
	v_accvgpr_write_b32 a173, 0
	v_accvgpr_write_b32 a174, 0
	v_accvgpr_write_b32 a175, 0
	v_accvgpr_write_b32 a188, 0
	v_accvgpr_write_b32 a189, 0
	v_accvgpr_write_b32 a190, 0
	v_accvgpr_write_b32 a191, 0
	v_accvgpr_write_b32 a196, 0
	v_accvgpr_write_b32 a197, 0
	v_accvgpr_write_b32 a198, 0
	v_accvgpr_write_b32 a199, 0
	v_accvgpr_write_b32 a204, 0
	v_accvgpr_write_b32 a205, 0
	v_accvgpr_write_b32 a206, 0
	v_accvgpr_write_b32 a207, 0
	v_accvgpr_write_b32 a212, 0
	v_accvgpr_write_b32 a213, 0
	v_accvgpr_write_b32 a214, 0
	v_accvgpr_write_b32 a215, 0
	v_accvgpr_write_b32 a216, 0
	v_accvgpr_write_b32 a217, 0
	v_accvgpr_write_b32 a218, 0
	v_accvgpr_write_b32 a219, 0
	v_accvgpr_write_b32 a220, 0
	v_accvgpr_write_b32 a221, 0
	v_accvgpr_write_b32 a222, 0
	v_accvgpr_write_b32 a223, 0
	v_accvgpr_write_b32 a228, 0
	v_accvgpr_write_b32 a229, 0
	v_accvgpr_write_b32 a230, 0
	v_accvgpr_write_b32 a231, 0
	v_accvgpr_write_b32 a232, 0
	v_accvgpr_write_b32 a233, 0
	v_accvgpr_write_b32 a234, 0
	v_accvgpr_write_b32 a235, 0
	v_accvgpr_write_b32 a236, 0
	v_accvgpr_write_b32 a237, 0
	v_accvgpr_write_b32 a238, 0
	v_accvgpr_write_b32 a239, 0
	v_accvgpr_write_b32 a240, 0
	v_accvgpr_write_b32 a241, 0
	v_accvgpr_write_b32 a242, 0
	v_accvgpr_write_b32 a243, 0
	v_accvgpr_write_b32 a244, 0
	v_accvgpr_write_b32 a245, 0
	v_accvgpr_write_b32 a246, 0
	v_accvgpr_write_b32 a247, 0
	v_accvgpr_write_b32 a248, 0
	v_accvgpr_write_b32 a249, 0
	v_accvgpr_write_b32 a250, 0
	v_accvgpr_write_b32 a251, 0
	v_accvgpr_write_b32 a252, 0
	v_accvgpr_write_b32 a253, 0
	v_accvgpr_write_b32 a254, 0
	v_accvgpr_write_b32 a255, 0
	v_accvgpr_write_b32 a40, 0
	v_accvgpr_write_b32 a41, 0
	v_accvgpr_write_b32 a42, 0
	v_accvgpr_write_b32 a43, 0
	v_accvgpr_write_b32 a56, 0
	v_accvgpr_write_b32 a57, 0
	v_accvgpr_write_b32 a58, 0
	v_accvgpr_write_b32 a59, 0
	v_accvgpr_write_b32 a68, 0
	v_accvgpr_write_b32 a69, 0
	v_accvgpr_write_b32 a70, 0
	v_accvgpr_write_b32 a71, 0
	v_accvgpr_write_b32 a80, 0
	v_accvgpr_write_b32 a81, 0
	v_accvgpr_write_b32 a82, 0
	v_accvgpr_write_b32 a83, 0
	v_accvgpr_write_b32 a92, 0
	v_accvgpr_write_b32 a93, 0
	v_accvgpr_write_b32 a94, 0
	v_accvgpr_write_b32 a95, 0
	v_accvgpr_write_b32 a104, 0
	v_accvgpr_write_b32 a105, 0
	v_accvgpr_write_b32 a106, 0
	v_accvgpr_write_b32 a107, 0
	v_accvgpr_write_b32 a116, 0
	v_accvgpr_write_b32 a117, 0
	v_accvgpr_write_b32 a118, 0
	v_accvgpr_write_b32 a119, 0
	v_accvgpr_write_b32 a128, 0
	v_accvgpr_write_b32 a129, 0
	v_accvgpr_write_b32 a130, 0
	v_accvgpr_write_b32 a131, 0
	v_accvgpr_write_b32 a136, 0
	v_accvgpr_write_b32 a137, 0
	v_accvgpr_write_b32 a138, 0
	v_accvgpr_write_b32 a139, 0
	v_accvgpr_write_b32 a152, 0
	v_accvgpr_write_b32 a153, 0
	v_accvgpr_write_b32 a154, 0
	v_accvgpr_write_b32 a155, 0
	v_accvgpr_write_b32 a168, 0
	v_accvgpr_write_b32 a169, 0
	v_accvgpr_write_b32 a170, 0
	v_accvgpr_write_b32 a171, 0
	v_accvgpr_write_b32 a180, 0
	v_accvgpr_write_b32 a181, 0
	v_accvgpr_write_b32 a182, 0
	v_accvgpr_write_b32 a183, 0
	v_accvgpr_write_b32 a184, 0
	v_accvgpr_write_b32 a185, 0
	v_accvgpr_write_b32 a186, 0
	v_accvgpr_write_b32 a187, 0
	v_accvgpr_write_b32 a192, 0
	v_accvgpr_write_b32 a193, 0
	v_accvgpr_write_b32 a194, 0
	v_accvgpr_write_b32 a195, 0
	v_accvgpr_write_b32 a200, 0
	v_accvgpr_write_b32 a201, 0
	v_accvgpr_write_b32 a202, 0
	v_accvgpr_write_b32 a203, 0
	v_accvgpr_write_b32 a208, 0
	v_accvgpr_write_b32 a209, 0
	v_accvgpr_write_b32 a210, 0
	v_accvgpr_write_b32 a211, 0
	v_accvgpr_write_b32 a48, 0
	v_accvgpr_write_b32 a49, 0
	v_accvgpr_write_b32 a50, 0
	v_accvgpr_write_b32 a51, 0
	v_accvgpr_write_b32 a16, 0
	v_accvgpr_write_b32 a17, 0
	v_accvgpr_write_b32 a18, 0
	v_accvgpr_write_b32 a19, 0
	v_accvgpr_write_b32 a20, 0
	v_accvgpr_write_b32 a21, 0
	v_accvgpr_write_b32 a22, 0
	v_accvgpr_write_b32 a23, 0
	v_accvgpr_write_b32 a28, 0
	v_accvgpr_write_b32 a29, 0
	v_accvgpr_write_b32 a30, 0
	v_accvgpr_write_b32 a31, 0
	v_accvgpr_write_b32 a36, 0
	v_accvgpr_write_b32 a37, 0
	v_accvgpr_write_b32 a38, 0
	v_accvgpr_write_b32 a39, 0
	v_accvgpr_write_b32 a52, 0
	v_accvgpr_write_b32 a53, 0
	v_accvgpr_write_b32 a54, 0
	v_accvgpr_write_b32 a55, 0
	v_accvgpr_write_b32 a64, 0
	v_accvgpr_write_b32 a65, 0
	v_accvgpr_write_b32 a66, 0
	v_accvgpr_write_b32 a67, 0
	v_accvgpr_write_b32 a76, 0
	v_accvgpr_write_b32 a77, 0
	v_accvgpr_write_b32 a78, 0
	v_accvgpr_write_b32 a79, 0
	v_accvgpr_write_b32 a88, 0
	v_accvgpr_write_b32 a89, 0
	v_accvgpr_write_b32 a90, 0
	v_accvgpr_write_b32 a91, 0
	v_accvgpr_write_b32 a96, 0
	v_accvgpr_write_b32 a97, 0
	v_accvgpr_write_b32 a98, 0
	v_accvgpr_write_b32 a99, 0
	v_accvgpr_write_b32 a108, 0
	v_accvgpr_write_b32 a109, 0
	v_accvgpr_write_b32 a110, 0
	v_accvgpr_write_b32 a111, 0
	v_accvgpr_write_b32 a120, 0
	v_accvgpr_write_b32 a121, 0
	v_accvgpr_write_b32 a122, 0
	v_accvgpr_write_b32 a123, 0
	v_accvgpr_write_b32 a132, 0
	v_accvgpr_write_b32 a133, 0
	v_accvgpr_write_b32 a134, 0
	v_accvgpr_write_b32 a135, 0
	v_accvgpr_write_b32 a148, 0
	v_accvgpr_write_b32 a149, 0
	v_accvgpr_write_b32 a150, 0
	v_accvgpr_write_b32 a151, 0
	v_accvgpr_write_b32 a164, 0
	v_accvgpr_write_b32 a165, 0
	v_accvgpr_write_b32 a166, 0
	v_accvgpr_write_b32 a167, 0
	v_accvgpr_write_b32 a176, 0
	v_accvgpr_write_b32 a177, 0
	v_accvgpr_write_b32 a178, 0
	v_accvgpr_write_b32 a179, 0
	v_accvgpr_write_b32 a224, 0
	v_accvgpr_write_b32 a225, 0
	v_accvgpr_write_b32 a226, 0
	v_accvgpr_write_b32 a227, 0
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
	v_accvgpr_write_b32 a24, 0
	v_accvgpr_write_b32 a25, 0
	v_accvgpr_write_b32 a26, 0
	v_accvgpr_write_b32 a27, 0
	v_accvgpr_write_b32 a32, 0
	v_accvgpr_write_b32 a33, 0
	v_accvgpr_write_b32 a34, 0
	v_accvgpr_write_b32 a35, 0
	v_accvgpr_write_b32 a44, 0
	v_accvgpr_write_b32 a45, 0
	v_accvgpr_write_b32 a46, 0
	v_accvgpr_write_b32 a47, 0
	v_accvgpr_write_b32 a60, 0
	v_accvgpr_write_b32 a61, 0
	v_accvgpr_write_b32 a62, 0
	v_accvgpr_write_b32 a63, 0
	v_accvgpr_write_b32 a72, 0
	v_accvgpr_write_b32 a73, 0
	v_accvgpr_write_b32 a74, 0
	v_accvgpr_write_b32 a75, 0
	v_accvgpr_write_b32 a84, 0
	v_accvgpr_write_b32 a85, 0
	v_accvgpr_write_b32 a86, 0
	v_accvgpr_write_b32 a87, 0
	v_accvgpr_write_b32 a100, 0
	v_accvgpr_write_b32 a101, 0
	v_accvgpr_write_b32 a102, 0
	v_accvgpr_write_b32 a103, 0
	v_accvgpr_write_b32 a112, 0
	v_accvgpr_write_b32 a113, 0
	v_accvgpr_write_b32 a114, 0
	v_accvgpr_write_b32 a115, 0
	v_accvgpr_write_b32 a124, 0
	v_accvgpr_write_b32 a125, 0
	v_accvgpr_write_b32 a126, 0
	v_accvgpr_write_b32 a127, 0
	v_accvgpr_write_b32 a140, 0
	v_accvgpr_write_b32 a141, 0
	v_accvgpr_write_b32 a142, 0
	v_accvgpr_write_b32 a143, 0
	v_accvgpr_write_b32 a156, 0
	v_accvgpr_write_b32 a157, 0
	v_accvgpr_write_b32 a158, 0
	v_accvgpr_write_b32 a159, 0
	s_mov_b64 s[36:37], 0
	s_waitcnt lgkmcnt(0)
	s_movk_i32 s27, 0x1000
	s_mov_b32 s71, 7
	;;#ASMSTART
	buffer_load_dwordx2 v[146:147], v137, s[4:7], s73 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[140:141], v137, s[20:23], s73 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[144:145], v137, s[16:19], s73 offen
	;;#ASMEND
	s_add_u32 s72, s72, 0x100
	;;#ASMSTART
	buffer_load_dwordx2 v[142:143], v137, s[0:3], s73 offen
	;;#ASMEND
	s_add_u32 s73, s74, 0x100
	s_add_u32 s74, s75, 0x100
	s_add_u32 s75, s76, 0x100
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
.LBB0_7:                                ; =>This Inner Loop Header: Depth=1
	s_add_u32 s76, s45, s36
	s_add_u32 s77, s76, 0x100
	s_mov_b32 m0, s25
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255],  v[34:37], v[2:5], a[252:255],  v146, v144 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[98:101], v157 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251],  v[34:37], v[6:9], a[248:251],  v146, v144 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[102:105], v157 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247],  v[34:37], v[10:13], a[244:247],  v146, v145 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[106:109], v157 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243],  v[34:37], v[14:17], a[240:243],  v146, v145 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[110:113], v157 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255],  v[50:53], v[18:21], a[252:255],  v146, v144 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[114:117], v156 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251],  v[50:53], v[22:25], a[248:251],  v146, v144 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[118:121], v156 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247],  v[50:53], v[26:29], a[244:247],  v146, v145 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[122:125], v156 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243],  v[50:53], v[30:33], a[240:243],  v146, v145 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[126:129], v156 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239],  v[38:41], v[2:5], a[236:239],  v146, v144 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235],  v[38:41], v[6:9], a[232:235],  v146, v144 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[38:41], v[10:13], a[228:231],  v146, v145 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[38:41], v[14:17], a[220:223],  v146, v145 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239],  v[54:57], v[18:21], a[236:239],  v146, v144 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235],  v[54:57], v[22:25], a[232:235],  v146, v144 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[54:57], v[26:29], a[228:231],  v146, v145 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[54:57], v[30:33], a[220:223],  v146, v145 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[42:45], v[2:5], a[216:219],  v147, v144 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[42:45], v[6:9], a[212:215],  v147, v144 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[42:45], v[10:13], a[204:207], v147, v145 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[42:45], v[14:17], a[196:199], v147, v145 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[58:61], v[18:21], a[216:219],  v147, v144 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[58:61], v[22:25], a[212:215],  v147, v144 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[58:61], v[26:29], a[204:207], v147, v145 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[58:61], v[30:33], a[196:199], v147, v145 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[46:49], v[2:5], a[188:191], v147, v144 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[46:49], v[6:9], a[172:175], v147, v144 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[46:49], v[10:13], a[160:163], v147, v145 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[46:49], v[14:17], a[144:147], v147, v145 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[62:65], v[18:21], a[188:191], v147, v144 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[62:65], v[22:25], a[172:175], v147, v144 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[62:65], v[26:29], a[160:163], v147, v145 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[62:65], v[30:33], a[144:147], v147, v145 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[34:37], v[98:101], a[208:211], v146, v142 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[66:69], v153 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[34:37], v[102:105], a[200:203], v146, v142 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[70:73], v153 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[34:37], v[106:109], a[192:195], v146, v143 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[74:77], v153 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[34:37], v[110:113], a[184:187], v146, v143 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[78:81], v153 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[50:53], v[114:117], a[208:211], v146, v142 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[82:85], v152 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[50:53], v[118:121], a[200:203], v146, v142 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[86:89], v152 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[50:53], v[122:125], a[192:195], v146, v143 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[90:93], v152 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[50:53], v[126:129], a[184:187], v146, v143 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[94:97], v152 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[38:41], v[98:101], a[180:183], v146, v142 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[38:41], v[102:105], a[168:171], v146, v142 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[38:41], v[106:109], a[152:155], v146, v143 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[38:41], v[110:113], a[136:139], v146, v143 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[54:57], v[114:117], a[180:183], v146, v142 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[54:57], v[118:121], a[168:171], v146, v142 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[54:57], v[122:125], a[152:155], v146, v143 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[54:57], v[126:129], a[136:139], v146, v143 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[42:45], v[98:101], a[128:131], v147, v142 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[42:45], v[102:105], a[116:119], v147, v142 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[42:45], v[106:109], a[104:107], v147, v143 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[42:45], v[110:113], a[92:95], v147, v143 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[58:61], v[114:117], a[128:131], v147, v142 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[58:61], v[118:121], a[116:119], v147, v142 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[58:61], v[122:125], a[104:107], v147, v143 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[58:61], v[126:129], a[92:95], v147, v143 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[46:49], v[98:101], a[80:83], v147, v142 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[46:49], v[102:105], a[68:71], v147, v142 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[46:49], v[106:109], a[56:59], v147, v143 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[46:49], v[110:113], a[40:43], v147, v143 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[62:65], v[114:117], a[80:83], v147, v142 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[62:65], v[118:121], a[68:71], v147, v142 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[62:65], v[122:125], a[56:59], v147, v143 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[62:65], v[126:129], a[40:43], v147, v143 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(20)
s_barrier

	;;#ASMEND
	s_add_u32 s78, s52, s36
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179],  v[66:69], v[2:5], a[176:179],  v140, v144 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v161 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[66:69], v[6:9], a[164:167],  v140, v144 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v161 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[66:69], v[10:13], a[148:151],  v140, v145 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v161 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[66:69], v[14:17], a[132:135],  v140, v145 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v161 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179],  v[82:85], v[18:21], a[176:179],  v140, v144 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v160 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[82:85], v[22:25], a[164:167],  v140, v144 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v160 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[82:85], v[26:29], a[148:151],  v140, v145 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v160 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[82:85], v[30:33], a[132:135],  v140, v145 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[62:65], v160 offset:6144

	;;#ASMEND
	buffer_load_dwordx4 v130, s[8:11], s77 offen lds
	s_mov_b32 m0, s35
	s_add_u32 s80, s78, 0x100
	buffer_load_dwordx4 v131, s[8:11], s77 offen lds
	s_mov_b32 m0, s46
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123],  v[70:73], v[2:5], a[120:123],  v140, v144 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111],  v[70:73], v[6:9], a[108:111],  v140, v144 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99],  v[70:73], v[10:13], a[96:99],  v140, v145 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91],  v[70:73], v[14:17], a[88:91],  v140, v145 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123],  v[86:89], v[18:21], a[120:123],  v140, v144 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111],  v[86:89], v[22:25], a[108:111],  v140, v144 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99],  v[86:89], v[26:29], a[96:99],  v140, v145 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91],  v[86:89], v[30:33], a[88:91],  v140, v145 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_i32 s81, s27, 0xfffff200
	buffer_load_dwordx4 v139, s[8:11], s77 offen lds
	s_mov_b32 m0, s47
	s_nop 0
	buffer_load_dwordx4 v132, s[8:11], s77 offen lds
	s_add_u32 s77, s48, s36
	s_add_u32 s79, s77, 0x100
	s_mov_b32 m0, s33
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79],  v[74:77], v[2:5], a[76:79],  v141, v144 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[74:77], v[6:9], a[64:67],  v141, v144 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[74:77], v[10:13], a[52:55], v141, v145 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[74:77], v[14:17], a[36:39], v141, v145 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79],  v[90:93], v[18:21], a[76:79],  v141, v144 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[90:93], v[22:25], a[64:67],  v141, v144 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[90:93], v[26:29], a[52:55], v141, v145 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[90:93], v[30:33], a[36:39], v141, v145 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v130, s[8:11], s79 offen lds
	s_mov_b32 m0, s49
	s_nop 0
	buffer_load_dwordx4 v131, s[8:11], s79 offen lds
	s_mov_b32 m0, s50
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[78:81], v[2:5], a[28:31], v141, v144 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[78:81], v[6:9], a[20:23], v141, v144 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[78:81], v[10:13], a[16:19], v141, v145 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[78:81], v[14:17], a[48:51], v141, v145 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[94:97], v[18:21], a[28:31], v141, v144 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[94:97], v[22:25], a[20:23], v141, v144 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[94:97], v[26:29], a[16:19], v141, v145 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[94:97], v[30:33], a[48:51], v141, v145 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[144:145], v137, s[4:7], s81 offen
	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v139, s[8:11], s79 offen lds
	s_mov_b32 m0, s51
	s_nop 0
	buffer_load_dwordx4 v132, s[8:11], s79 offen lds
	s_mov_b32 m0, s38
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[66:69], v[98:101], a[156:159],  v140, v142 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[2:5], v159 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[66:69], v[102:105], a[140:143],  v140, v142 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[6:9], v159 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127],  v[66:69], v[106:109], a[124:127],  v140, v143 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[10:13], v159 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115],  v[66:69], v[110:113], a[112:115],  v140, v143 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[14:17], v159 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[82:85], v[114:117], a[156:159],  v140, v142 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[18:21], v158 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[82:85], v[118:121], a[140:143],  v140, v142 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[22:25], v158 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127],  v[82:85], v[122:125], a[124:127],  v140, v143 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[26:29], v158 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115],  v[82:85], v[126:129], a[112:115],  v140, v143 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v158 offset:6144

	;;#ASMEND
	s_add_u32 s79, s55, s36
	buffer_load_dwordx4 v134, s[12:15], s80 offen lds
	s_mov_b32 m0, s31
	s_nop 0
	buffer_load_dwordx4 v133, s[12:15], s80 offen lds
	s_mov_b32 m0, s53
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103],  v[70:73], v[98:101], a[100:103],  v140, v142 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87],  v[70:73], v[102:105], a[84:87],  v140, v142 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75],  v[70:73], v[106:109], a[72:75],  v140, v143 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[70:73], v[110:113], a[60:63],  v140, v143 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103],  v[86:89], v[114:117], a[100:103],  v140, v142 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87],  v[86:89], v[118:121], a[84:87],  v140, v142 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75],  v[86:89], v[122:125], a[72:75],  v140, v143 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[86:89], v[126:129], a[60:63],  v140, v143 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v135, s[12:15], s80 offen lds
	s_mov_b32 m0, s54
	s_nop 0
	buffer_load_dwordx4 v136, s[12:15], s80 offen lds
	s_add_u32 s80, s79, 0x100
	s_mov_b32 m0, s39
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[74:77], v[98:101], a[44:47],  v141, v142 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[74:77], v[102:105], a[32:35],  v141, v142 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[74:77], v[106:109], a[24:27], v141, v143 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[74:77], v[110:113], a[12:15], v141, v143 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[90:93], v[114:117], a[44:47],  v141, v142 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[90:93], v[118:121], a[32:35],  v141, v142 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[90:93], v[122:125], a[24:27], v141, v143 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[90:93], v[126:129], a[12:15], v141, v143 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v134, s[12:15], s80 offen lds
	s_mov_b32 m0, s56
	s_nop 0
	buffer_load_dwordx4 v133, s[12:15], s80 offen lds
	s_mov_b32 m0, s57
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[78:81], v[98:101], a[8:11], v141, v142 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[78:81], v[102:105], a[4:7], v141, v142 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[78:81], v[106:109], a[0:3], v141, v143 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[78:81], v[110:113], a[224:227], v141, v143 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[94:97], v[114:117], a[8:11], v141, v142 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[94:97], v[118:121], a[4:7], v141, v142 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[94:97], v[122:125], a[0:3], v141, v143 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[94:97], v[126:129], a[224:227], v141, v143 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[140:141], v137, s[0:3], s81 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[142:143], v137, s[16:19], s81 offen
	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v135, s[12:15], s80 offen lds
	s_mov_b32 m0, s58
	s_nop 0
	buffer_load_dwordx4 v136, s[12:15], s80 offen lds
	s_add_u32 s80, s76, 0x180
	s_mov_b32 m0, s43
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255],  v[34:37], v[2:5], a[252:255],  v144, v142 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[98:101], v155 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251],  v[34:37], v[6:9], a[248:251],  v144, v142 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[102:105], v155 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247],  v[34:37], v[10:13], a[244:247],  v144, v143 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[106:109], v155 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243],  v[34:37], v[14:17], a[240:243],  v144, v143 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[110:113], v155 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255],  v[50:53], v[18:21], a[252:255],  v144, v142 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[114:117], v154 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251],  v[50:53], v[22:25], a[248:251],  v144, v142 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[118:121], v154 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247],  v[50:53], v[26:29], a[244:247],  v144, v143 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[122:125], v154 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243],  v[50:53], v[30:33], a[240:243],  v144, v143 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[126:129], v154 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239],  v[38:41], v[2:5], a[236:239],  v144, v142 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235],  v[38:41], v[6:9], a[232:235],  v144, v142 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[38:41], v[10:13], a[228:231],  v144, v143 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[38:41], v[14:17], a[220:223],  v144, v143 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239],  v[54:57], v[18:21], a[236:239],  v144, v142 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235],  v[54:57], v[22:25], a[232:235],  v144, v142 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[54:57], v[26:29], a[228:231],  v144, v143 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[54:57], v[30:33], a[220:223],  v144, v143 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[42:45], v[2:5], a[216:219],  v145, v142 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[42:45], v[6:9], a[212:215],  v145, v142 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[42:45], v[10:13], a[204:207], v145, v143 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[42:45], v[14:17], a[196:199], v145, v143 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[58:61], v[18:21], a[216:219],  v145, v142 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[58:61], v[22:25], a[212:215],  v145, v142 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[58:61], v[26:29], a[204:207], v145, v143 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[58:61], v[30:33], a[196:199], v145, v143 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[46:49], v[2:5], a[188:191], v145, v142 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[46:49], v[6:9], a[172:175], v145, v142 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[46:49], v[10:13], a[160:163], v145, v143 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[46:49], v[14:17], a[144:147], v145, v143 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[62:65], v[18:21], a[188:191], v145, v142 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[62:65], v[22:25], a[172:175], v145, v142 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[62:65], v[26:29], a[160:163], v145, v143 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[62:65], v[30:33], a[144:147], v145, v143 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[34:37], v[98:101], a[208:211], v144, v140 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[66:69], v151 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[34:37], v[102:105], a[200:203], v144, v140 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[70:73], v151 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[34:37], v[106:109], a[192:195], v144, v141 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[74:77], v151 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[34:37], v[110:113], a[184:187], v144, v141 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[78:81], v151 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[50:53], v[114:117], a[208:211], v144, v140 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[82:85], v150 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[50:53], v[118:121], a[200:203], v144, v140 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[86:89], v150 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[50:53], v[122:125], a[192:195], v144, v141 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[90:93], v150 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[50:53], v[126:129], a[184:187], v144, v141 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[94:97], v150 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[38:41], v[98:101], a[180:183], v144, v140 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[38:41], v[102:105], a[168:171], v144, v140 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[38:41], v[106:109], a[152:155], v144, v141 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[38:41], v[110:113], a[136:139], v144, v141 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[54:57], v[114:117], a[180:183], v144, v140 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[54:57], v[118:121], a[168:171], v144, v140 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[54:57], v[122:125], a[152:155], v144, v141 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[54:57], v[126:129], a[136:139], v144, v141 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[42:45], v[98:101], a[128:131], v145, v140 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[42:45], v[102:105], a[116:119], v145, v140 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[42:45], v[106:109], a[104:107], v145, v141 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[42:45], v[110:113], a[92:95], v145, v141 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[58:61], v[114:117], a[128:131], v145, v140 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[58:61], v[118:121], a[116:119], v145, v140 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[58:61], v[122:125], a[104:107], v145, v141 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[58:61], v[126:129], a[92:95], v145, v141 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[46:49], v[98:101], a[80:83], v145, v140 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[46:49], v[102:105], a[68:71], v145, v140 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[46:49], v[106:109], a[56:59], v145, v141 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[46:49], v[110:113], a[40:43], v145, v141 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[62:65], v[114:117], a[80:83], v145, v140 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[62:65], v[118:121], a[68:71], v145, v140 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[62:65], v[122:125], a[56:59], v145, v141 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[62:65], v[126:129], a[40:43], v145, v141 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[34:35], v137, s[20:23], s81 offen
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(20)
s_barrier

	;;#ASMEND
	s_add_i32 s81, s27, 0xfffff400
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179],  v[66:69], v[2:5], a[176:179],  v34, v142 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[144:147], v149 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[66:69], v[6:9], a[164:167],  v34, v142 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[164:167], v149 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[66:69], v[10:13], a[148:151],  v34, v143 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[168:171], v149 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[66:69], v[14:17], a[132:135],  v34, v143 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[172:175], v149 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179],  v[82:85], v[18:21], a[176:179],  v34, v142 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[176:179], v148 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[82:85], v[22:25], a[164:167],  v34, v142 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[180:183], v148 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[82:85], v[26:29], a[148:151],  v34, v143 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[184:187], v148 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[82:85], v[30:33], a[132:135],  v34, v143 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[188:191], v148 offset:6144

	;;#ASMEND
	buffer_load_dwordx4 v130, s[8:11], s80 offen lds
	s_mov_b32 m0, s59
	s_nop 0
	buffer_load_dwordx4 v131, s[8:11], s80 offen lds
	s_mov_b32 m0, s60
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123],  v[70:73], v[2:5], a[120:123],  v34, v142 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111],  v[70:73], v[6:9], a[108:111],  v34, v142 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99],  v[70:73], v[10:13], a[96:99],  v34, v143 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91],  v[70:73], v[14:17], a[88:91],  v34, v143 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123],  v[86:89], v[18:21], a[120:123],  v34, v142 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111],  v[86:89], v[22:25], a[108:111],  v34, v142 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99],  v[86:89], v[26:29], a[96:99],  v34, v143 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91],  v[86:89], v[30:33], a[88:91],  v34, v143 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v139, s[8:11], s80 offen lds
	s_mov_b32 m0, s61
	s_nop 0
	buffer_load_dwordx4 v132, s[8:11], s80 offen lds
	s_add_u32 s80, s77, 0x180
	s_mov_b32 m0, s42
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79],  v[74:77], v[2:5], a[76:79],  v35, v142 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[74:77], v[6:9], a[64:67],  v35, v142 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[74:77], v[10:13], a[52:55], v35, v143 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[74:77], v[14:17], a[36:39], v35, v143 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79],  v[90:93], v[18:21], a[76:79],  v35, v142 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[90:93], v[22:25], a[64:67],  v35, v142 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[90:93], v[26:29], a[52:55], v35, v143 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[90:93], v[30:33], a[36:39], v35, v143 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v130, s[8:11], s80 offen lds
	s_mov_b32 m0, s62
	s_nop 0
	buffer_load_dwordx4 v131, s[8:11], s80 offen lds
	s_mov_b32 m0, s63
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[78:81], v[2:5], a[28:31], v35, v142 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[78:81], v[6:9], a[20:23], v35, v142 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[78:81], v[10:13], a[16:19], v35, v143 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[78:81], v[14:17], a[48:51], v35, v143 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[94:97], v[18:21], a[28:31], v35, v142 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[94:97], v[22:25], a[20:23], v35, v142 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[94:97], v[26:29], a[16:19], v35, v143 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[94:97], v[30:33], a[48:51], v35, v143 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v139, s[8:11], s80 offen lds
	s_mov_b32 m0, s64
	s_nop 0
	buffer_load_dwordx4 v132, s[8:11], s80 offen lds
	s_add_u32 s80, s78, 0x180
	s_mov_b32 m0, s41
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[66:69], v[98:101], a[156:159],  v34, v140 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[2:5], v163 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[66:69], v[102:105], a[140:143],  v34, v140 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[6:9], v163 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127],  v[66:69], v[106:109], a[124:127],  v34, v141 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[10:13], v163 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115],  v[66:69], v[110:113], a[112:115],  v34, v141 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[14:17], v163 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[82:85], v[114:117], a[156:159],  v34, v140 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[18:21], v162 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[82:85], v[118:121], a[140:143],  v34, v140 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[22:25], v162 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127],  v[82:85], v[122:125], a[124:127],  v34, v141 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[26:29], v162 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115],  v[82:85], v[126:129], a[112:115],  v34, v141 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v162 offset:6144

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v134, s[12:15], s80 offen lds
	s_mov_b32 m0, s65
	s_nop 0
	buffer_load_dwordx4 v133, s[12:15], s80 offen lds
	s_mov_b32 m0, s66
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103],  v[70:73], v[98:101], a[100:103],  v34, v140 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87],  v[70:73], v[102:105], a[84:87],  v34, v140 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75],  v[70:73], v[106:109], a[72:75],  v34, v141 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[70:73], v[110:113], a[60:63],  v34, v141 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103],  v[86:89], v[114:117], a[100:103],  v34, v140 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87],  v[86:89], v[118:121], a[84:87],  v34, v140 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75],  v[86:89], v[122:125], a[72:75],  v34, v141 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[86:89], v[126:129], a[60:63],  v34, v141 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v135, s[12:15], s80 offen lds
	s_mov_b32 m0, s67
	s_nop 0
	buffer_load_dwordx4 v136, s[12:15], s80 offen lds
	s_add_u32 s80, s79, 0x180
	s_mov_b32 m0, s40
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[74:77], v[98:101], a[44:47],  v35, v140 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[74:77], v[102:105], a[32:35],  v35, v140 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[74:77], v[106:109], a[24:27], v35, v141 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[74:77], v[110:113], a[12:15], v35, v141 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[90:93], v[114:117], a[44:47],  v35, v140 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[90:93], v[118:121], a[32:35],  v35, v140 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[90:93], v[122:125], a[24:27], v35, v141 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[90:93], v[126:129], a[12:15], v35, v141 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v134, s[12:15], s80 offen lds
	s_mov_b32 m0, s68
	s_nop 0
	buffer_load_dwordx4 v133, s[12:15], s80 offen lds
	s_mov_b32 m0, s69
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[78:81], v[98:101], a[8:11], v35, v140 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[78:81], v[102:105], a[4:7], v35, v140 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[78:81], v[106:109], a[0:3], v35, v141 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[78:81], v[110:113], a[224:227], v35, v141 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[94:97], v[114:117], a[8:11], v35, v140 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[94:97], v[118:121], a[4:7], v35, v140 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[94:97], v[122:125], a[0:3], v35, v141 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[94:97], v[126:129], a[224:227], v35, v141 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[100:101], v137, s[4:7], s81 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[98:99], v137, s[0:3], s81 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[102:103], v137, s[16:19], s81 offen
	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v135, s[12:15], s80 offen lds
	s_mov_b32 m0, s70
	s_nop 0
	buffer_load_dwordx4 v136, s[12:15], s80 offen lds
	s_add_u32 s80, s76, 0x200
	s_mov_b32 m0, s25
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255],  v[144:147], v[2:5], a[252:255],  v100, v102 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[66:69], v157 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251],  v[144:147], v[6:9], a[248:251],  v100, v102 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[70:73], v157 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247],  v[144:147], v[10:13], a[244:247],  v100, v103 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[74:77], v157 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243],  v[144:147], v[14:17], a[240:243],  v100, v103 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[78:81], v157 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255],  v[176:179], v[18:21], a[252:255],  v100, v102 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[82:85], v156 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251],  v[176:179], v[22:25], a[248:251],  v100, v102 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[86:89], v156 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247],  v[176:179], v[26:29], a[244:247],  v100, v103 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[90:93], v156 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243],  v[176:179], v[30:33], a[240:243],  v100, v103 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[94:97], v156 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239],  v[164:167], v[2:5], a[236:239],  v100, v102 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235],  v[164:167], v[6:9], a[232:235],  v100, v102 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[164:167], v[10:13], a[228:231],  v100, v103 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[164:167], v[14:17], a[220:223],  v100, v103 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239],  v[180:183], v[18:21], a[236:239],  v100, v102 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235],  v[180:183], v[22:25], a[232:235],  v100, v102 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[180:183], v[26:29], a[228:231],  v100, v103 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[180:183], v[30:33], a[220:223],  v100, v103 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[168:171], v[2:5], a[216:219],  v101, v102 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[168:171], v[6:9], a[212:215],  v101, v102 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[168:171], v[10:13], a[204:207], v101, v103 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[168:171], v[14:17], a[196:199], v101, v103 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[184:187], v[18:21], a[216:219],  v101, v102 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[184:187], v[22:25], a[212:215],  v101, v102 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[184:187], v[26:29], a[204:207], v101, v103 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[184:187], v[30:33], a[196:199], v101, v103 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[172:175], v[2:5], a[188:191], v101, v102 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[172:175], v[6:9], a[172:175], v101, v102 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[172:175], v[10:13], a[160:163], v101, v103 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[172:175], v[14:17], a[144:147], v101, v103 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[188:191], v[18:21], a[188:191], v101, v102 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[188:191], v[22:25], a[172:175], v101, v102 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[188:191], v[26:29], a[160:163], v101, v103 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[188:191], v[30:33], a[144:147], v101, v103 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[144:147], v[66:69], a[208:211], v100, v98 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v153 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[144:147], v[70:73], a[200:203], v100, v98 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v153 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[144:147], v[74:77], a[192:195], v100, v99 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v153 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[144:147], v[78:81], a[184:187], v100, v99 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v153 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[176:179], v[82:85], a[208:211], v100, v98 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v152 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[176:179], v[86:89], a[200:203], v100, v98 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v152 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[176:179], v[90:93], a[192:195], v100, v99 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v152 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[176:179], v[94:97], a[184:187], v100, v99 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[62:65], v152 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[164:167], v[66:69], a[180:183], v100, v98 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[164:167], v[70:73], a[168:171], v100, v98 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[164:167], v[74:77], a[152:155], v100, v99 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[164:167], v[78:81], a[136:139], v100, v99 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[180:183], v[82:85], a[180:183], v100, v98 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[180:183], v[86:89], a[168:171], v100, v98 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[180:183], v[90:93], a[152:155], v100, v99 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[180:183], v[94:97], a[136:139], v100, v99 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[168:171], v[66:69], a[128:131], v101, v98 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[168:171], v[70:73], a[116:119], v101, v98 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[168:171], v[74:77], a[104:107], v101, v99 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[168:171], v[78:81], a[92:95], v101, v99 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[184:187], v[82:85], a[128:131], v101, v98 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[184:187], v[86:89], a[116:119], v101, v98 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[184:187], v[90:93], a[104:107], v101, v99 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[184:187], v[94:97], a[92:95], v101, v99 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[172:175], v[66:69], a[80:83], v101, v98 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[172:175], v[70:73], a[68:71], v101, v98 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[172:175], v[74:77], a[56:59], v101, v99 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[172:175], v[78:81], a[40:43], v101, v99 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[188:191], v[82:85], a[80:83], v101, v98 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[188:191], v[86:89], a[68:71], v101, v98 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[188:191], v[90:93], a[56:59], v101, v99 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[188:191], v[94:97], a[40:43], v101, v99 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[100:101], v137, s[20:23], s81 offen
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(20)
s_barrier

	;;#ASMEND
	s_add_i32 s81, s27, 0xfffff600
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179],  v[34:37], v[2:5], a[176:179],  v100, v102 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[104:107], v161 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[34:37], v[6:9], a[164:167],  v100, v102 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[108:111], v161 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[34:37], v[10:13], a[148:151],  v100, v103 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[112:115], v161 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[34:37], v[14:17], a[132:135],  v100, v103 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[116:119], v161 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179],  v[50:53], v[18:21], a[176:179],  v100, v102 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[120:123], v160 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[50:53], v[22:25], a[164:167],  v100, v102 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[124:127], v160 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[50:53], v[26:29], a[148:151],  v100, v103 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[140:143], v160 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[50:53], v[30:33], a[132:135],  v100, v103 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[144:147], v160 offset:6144

	;;#ASMEND
	buffer_load_dwordx4 v130, s[8:11], s80 offen lds
	s_mov_b32 m0, s35
	s_nop 0
	buffer_load_dwordx4 v131, s[8:11], s80 offen lds
	s_mov_b32 m0, s46
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123],  v[38:41], v[2:5], a[120:123],  v100, v102 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111],  v[38:41], v[6:9], a[108:111],  v100, v102 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99],  v[38:41], v[10:13], a[96:99],  v100, v103 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91],  v[38:41], v[14:17], a[88:91],  v100, v103 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123],  v[54:57], v[18:21], a[120:123],  v100, v102 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111],  v[54:57], v[22:25], a[108:111],  v100, v102 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99],  v[54:57], v[26:29], a[96:99],  v100, v103 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91],  v[54:57], v[30:33], a[88:91],  v100, v103 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v139, s[8:11], s80 offen lds
	s_mov_b32 m0, s47
	s_nop 0
	buffer_load_dwordx4 v132, s[8:11], s80 offen lds
	s_add_u32 s80, s77, 0x200
	s_mov_b32 m0, s33
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79],  v[42:45], v[2:5], a[76:79],  v101, v102 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[42:45], v[6:9], a[64:67],  v101, v102 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[42:45], v[10:13], a[52:55], v101, v103 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[42:45], v[14:17], a[36:39], v101, v103 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79],  v[58:61], v[18:21], a[76:79],  v101, v102 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[58:61], v[22:25], a[64:67],  v101, v102 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[58:61], v[26:29], a[52:55], v101, v103 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[58:61], v[30:33], a[36:39], v101, v103 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v130, s[8:11], s80 offen lds
	s_mov_b32 m0, s49
	s_nop 0
	buffer_load_dwordx4 v131, s[8:11], s80 offen lds
	s_mov_b32 m0, s50
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[46:49], v[2:5], a[28:31], v101, v102 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[46:49], v[6:9], a[20:23], v101, v102 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[46:49], v[10:13], a[16:19], v101, v103 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[46:49], v[14:17], a[48:51], v101, v103 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[62:65], v[18:21], a[28:31], v101, v102 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[62:65], v[22:25], a[20:23], v101, v102 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[62:65], v[26:29], a[16:19], v101, v103 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[62:65], v[30:33], a[48:51], v101, v103 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[102:103], v137, s[16:19], s81 offen
	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v139, s[8:11], s80 offen lds
	s_mov_b32 m0, s51
	s_nop 0
	buffer_load_dwordx4 v132, s[8:11], s80 offen lds
	s_add_u32 s80, s78, 0x200
	s_mov_b32 m0, s38
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[34:37], v[66:69], a[156:159],  v100, v98 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[2:5], v159 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[34:37], v[70:73], a[140:143],  v100, v98 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[6:9], v159 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127],  v[34:37], v[74:77], a[124:127],  v100, v99 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[10:13], v159 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115],  v[34:37], v[78:81], a[112:115],  v100, v99 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[14:17], v159 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[50:53], v[82:85], a[156:159],  v100, v98 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[18:21], v158 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[50:53], v[86:89], a[140:143],  v100, v98 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[22:25], v158 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127],  v[50:53], v[90:93], a[124:127],  v100, v99 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[26:29], v158 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115],  v[50:53], v[94:97], a[112:115],  v100, v99 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v158 offset:6144

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v134, s[12:15], s80 offen lds
	s_mov_b32 m0, s31
	s_nop 0
	buffer_load_dwordx4 v133, s[12:15], s80 offen lds
	s_mov_b32 m0, s53
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103],  v[38:41], v[66:69], a[100:103],  v100, v98 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87],  v[38:41], v[70:73], a[84:87],  v100, v98 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75],  v[38:41], v[74:77], a[72:75],  v100, v99 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[38:41], v[78:81], a[60:63],  v100, v99 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103],  v[54:57], v[82:85], a[100:103],  v100, v98 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87],  v[54:57], v[86:89], a[84:87],  v100, v98 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75],  v[54:57], v[90:93], a[72:75],  v100, v99 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[54:57], v[94:97], a[60:63],  v100, v99 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v135, s[12:15], s80 offen lds
	s_mov_b32 m0, s54
	s_nop 0
	buffer_load_dwordx4 v136, s[12:15], s80 offen lds
	s_add_u32 s80, s79, 0x200
	s_mov_b32 m0, s39
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[42:45], v[66:69], a[44:47],  v101, v98 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[42:45], v[70:73], a[32:35],  v101, v98 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[42:45], v[74:77], a[24:27], v101, v99 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[42:45], v[78:81], a[12:15], v101, v99 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[58:61], v[82:85], a[44:47],  v101, v98 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[58:61], v[86:89], a[32:35],  v101, v98 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[58:61], v[90:93], a[24:27], v101, v99 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[58:61], v[94:97], a[12:15], v101, v99 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v134, s[12:15], s80 offen lds
	s_mov_b32 m0, s56
	s_nop 0
	buffer_load_dwordx4 v133, s[12:15], s80 offen lds
	s_mov_b32 m0, s57
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[46:49], v[66:69], a[8:11], v101, v98 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[46:49], v[70:73], a[4:7], v101, v98 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[46:49], v[74:77], a[0:3], v101, v99 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[46:49], v[78:81], a[224:227], v101, v99 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[62:65], v[82:85], a[8:11], v101, v98 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[62:65], v[86:89], a[4:7], v101, v98 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[62:65], v[90:93], a[0:3], v101, v99 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[62:65], v[94:97], a[224:227], v101, v99 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[100:101], v137, s[4:7], s81 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[98:99], v137, s[0:3], s81 offen
	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v135, s[12:15], s80 offen lds
	s_mov_b32 m0, s58
	s_nop 0
	buffer_load_dwordx4 v136, s[12:15], s80 offen lds
	s_add_u32 s80, s76, 0x280
	s_mov_b32 m0, s43
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255],  v[104:107], v[2:5], a[252:255],  v100, v102 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[66:69], v155 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251],  v[104:107], v[6:9], a[248:251],  v100, v102 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[70:73], v155 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247],  v[104:107], v[10:13], a[244:247],  v100, v103 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[74:77], v155 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243],  v[104:107], v[14:17], a[240:243],  v100, v103 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[78:81], v155 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255],  v[120:123], v[18:21], a[252:255],  v100, v102 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[82:85], v154 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251],  v[120:123], v[22:25], a[248:251],  v100, v102 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[86:89], v154 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247],  v[120:123], v[26:29], a[244:247],  v100, v103 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[90:93], v154 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243],  v[120:123], v[30:33], a[240:243],  v100, v103 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[94:97], v154 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239],  v[108:111], v[2:5], a[236:239],  v100, v102 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235],  v[108:111], v[6:9], a[232:235],  v100, v102 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[108:111], v[10:13], a[228:231],  v100, v103 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[108:111], v[14:17], a[220:223],  v100, v103 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239],  v[124:127], v[18:21], a[236:239],  v100, v102 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235],  v[124:127], v[22:25], a[232:235],  v100, v102 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[124:127], v[26:29], a[228:231],  v100, v103 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[124:127], v[30:33], a[220:223],  v100, v103 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[112:115], v[2:5], a[216:219],  v101, v102 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[112:115], v[6:9], a[212:215],  v101, v102 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[112:115], v[10:13], a[204:207], v101, v103 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[112:115], v[14:17], a[196:199], v101, v103 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[140:143], v[18:21], a[216:219],  v101, v102 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[140:143], v[22:25], a[212:215],  v101, v102 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[140:143], v[26:29], a[204:207], v101, v103 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[140:143], v[30:33], a[196:199], v101, v103 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[116:119], v[2:5], a[188:191], v101, v102 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[116:119], v[6:9], a[172:175], v101, v102 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[116:119], v[10:13], a[160:163], v101, v103 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[116:119], v[14:17], a[144:147], v101, v103 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[144:147], v[18:21], a[188:191], v101, v102 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[144:147], v[22:25], a[172:175], v101, v102 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[144:147], v[26:29], a[160:163], v101, v103 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[144:147], v[30:33], a[144:147], v101, v103 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[104:107], v[66:69], a[208:211], v100, v98 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v151 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[104:107], v[70:73], a[200:203], v100, v98 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v151 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[104:107], v[74:77], a[192:195], v100, v99 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v151 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[104:107], v[78:81], a[184:187], v100, v99 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v151 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[120:123], v[82:85], a[208:211], v100, v98 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v150 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[120:123], v[86:89], a[200:203], v100, v98 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v150 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[120:123], v[90:93], a[192:195], v100, v99 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v150 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[120:123], v[94:97], a[184:187], v100, v99 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[62:65], v150 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[108:111], v[66:69], a[180:183], v100, v98 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[108:111], v[70:73], a[168:171], v100, v98 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[108:111], v[74:77], a[152:155], v100, v99 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[108:111], v[78:81], a[136:139], v100, v99 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[124:127], v[82:85], a[180:183], v100, v98 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[124:127], v[86:89], a[168:171], v100, v98 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[124:127], v[90:93], a[152:155], v100, v99 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[124:127], v[94:97], a[136:139], v100, v99 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[112:115], v[66:69], a[128:131], v101, v98 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[112:115], v[70:73], a[116:119], v101, v98 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[112:115], v[74:77], a[104:107], v101, v99 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[112:115], v[78:81], a[92:95], v101, v99 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[140:143], v[82:85], a[128:131], v101, v98 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[140:143], v[86:89], a[116:119], v101, v98 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[140:143], v[90:93], a[104:107], v101, v99 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[140:143], v[94:97], a[92:95], v101, v99 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[116:119], v[66:69], a[80:83], v101, v98 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[116:119], v[70:73], a[68:71], v101, v98 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[116:119], v[74:77], a[56:59], v101, v99 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[116:119], v[78:81], a[40:43], v101, v99 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[144:147], v[82:85], a[80:83], v101, v98 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[144:147], v[86:89], a[68:71], v101, v98 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[144:147], v[90:93], a[56:59], v101, v99 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[144:147], v[94:97], a[40:43], v101, v99 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[100:101], v137, s[20:23], s81 offen
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(20)
s_barrier

	;;#ASMEND
	s_add_i32 s81, s27, 0xfffff800
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179],  v[34:37], v[2:5], a[176:179],  v100, v102 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[104:107], v149 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[34:37], v[6:9], a[164:167],  v100, v102 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[108:111], v149 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[34:37], v[10:13], a[148:151],  v100, v103 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[112:115], v149 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[34:37], v[14:17], a[132:135],  v100, v103 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[116:119], v149 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179],  v[50:53], v[18:21], a[176:179],  v100, v102 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[120:123], v148 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[50:53], v[22:25], a[164:167],  v100, v102 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[124:127], v148 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[50:53], v[26:29], a[148:151],  v100, v103 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[140:143], v148 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[50:53], v[30:33], a[132:135],  v100, v103 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[144:147], v148 offset:6144

	;;#ASMEND
	buffer_load_dwordx4 v130, s[8:11], s80 offen lds
	s_mov_b32 m0, s59
	s_nop 0
	buffer_load_dwordx4 v131, s[8:11], s80 offen lds
	s_mov_b32 m0, s60
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123],  v[38:41], v[2:5], a[120:123],  v100, v102 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111],  v[38:41], v[6:9], a[108:111],  v100, v102 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99],  v[38:41], v[10:13], a[96:99],  v100, v103 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91],  v[38:41], v[14:17], a[88:91],  v100, v103 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123],  v[54:57], v[18:21], a[120:123],  v100, v102 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111],  v[54:57], v[22:25], a[108:111],  v100, v102 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99],  v[54:57], v[26:29], a[96:99],  v100, v103 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91],  v[54:57], v[30:33], a[88:91],  v100, v103 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v139, s[8:11], s80 offen lds
	s_mov_b32 m0, s61
	s_nop 0
	buffer_load_dwordx4 v132, s[8:11], s80 offen lds
	s_add_u32 s80, s77, 0x280
	s_mov_b32 m0, s42
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79],  v[42:45], v[2:5], a[76:79],  v101, v102 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[42:45], v[6:9], a[64:67],  v101, v102 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[42:45], v[10:13], a[52:55], v101, v103 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[42:45], v[14:17], a[36:39], v101, v103 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79],  v[58:61], v[18:21], a[76:79],  v101, v102 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[58:61], v[22:25], a[64:67],  v101, v102 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[58:61], v[26:29], a[52:55], v101, v103 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[58:61], v[30:33], a[36:39], v101, v103 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v130, s[8:11], s80 offen lds
	s_mov_b32 m0, s62
	s_nop 0
	buffer_load_dwordx4 v131, s[8:11], s80 offen lds
	s_mov_b32 m0, s63
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[46:49], v[2:5], a[28:31], v101, v102 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[46:49], v[6:9], a[20:23], v101, v102 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[46:49], v[10:13], a[16:19], v101, v103 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[46:49], v[14:17], a[48:51], v101, v103 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[62:65], v[18:21], a[28:31], v101, v102 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[62:65], v[22:25], a[20:23], v101, v102 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[62:65], v[26:29], a[16:19], v101, v103 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[62:65], v[30:33], a[48:51], v101, v103 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[102:103], v137, s[16:19], s81 offen
	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v139, s[8:11], s80 offen lds
	s_mov_b32 m0, s64
	s_nop 0
	buffer_load_dwordx4 v132, s[8:11], s80 offen lds
	s_add_u32 s80, s78, 0x280
	s_mov_b32 m0, s41
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[34:37], v[66:69], a[156:159],  v100, v98 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[2:5], v163 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[34:37], v[70:73], a[140:143],  v100, v98 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[6:9], v163 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127],  v[34:37], v[74:77], a[124:127],  v100, v99 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[10:13], v163 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115],  v[34:37], v[78:81], a[112:115],  v100, v99 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[14:17], v163 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[50:53], v[82:85], a[156:159],  v100, v98 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[18:21], v162 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[50:53], v[86:89], a[140:143],  v100, v98 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[22:25], v162 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127],  v[50:53], v[90:93], a[124:127],  v100, v99 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[26:29], v162 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115],  v[50:53], v[94:97], a[112:115],  v100, v99 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v162 offset:6144

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v134, s[12:15], s80 offen lds
	s_mov_b32 m0, s65
	s_nop 0
	buffer_load_dwordx4 v133, s[12:15], s80 offen lds
	s_mov_b32 m0, s66
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103],  v[38:41], v[66:69], a[100:103],  v100, v98 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87],  v[38:41], v[70:73], a[84:87],  v100, v98 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75],  v[38:41], v[74:77], a[72:75],  v100, v99 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[38:41], v[78:81], a[60:63],  v100, v99 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103],  v[54:57], v[82:85], a[100:103],  v100, v98 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87],  v[54:57], v[86:89], a[84:87],  v100, v98 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75],  v[54:57], v[90:93], a[72:75],  v100, v99 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[54:57], v[94:97], a[60:63],  v100, v99 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v135, s[12:15], s80 offen lds
	s_mov_b32 m0, s67
	s_nop 0
	buffer_load_dwordx4 v136, s[12:15], s80 offen lds
	s_add_u32 s80, s79, 0x280
	s_mov_b32 m0, s40
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[42:45], v[66:69], a[44:47],  v101, v98 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[42:45], v[70:73], a[32:35],  v101, v98 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[42:45], v[74:77], a[24:27], v101, v99 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[42:45], v[78:81], a[12:15], v101, v99 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[58:61], v[82:85], a[44:47],  v101, v98 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[58:61], v[86:89], a[32:35],  v101, v98 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[58:61], v[90:93], a[24:27], v101, v99 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[58:61], v[94:97], a[12:15], v101, v99 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v134, s[12:15], s80 offen lds
	s_mov_b32 m0, s68
	s_nop 0
	buffer_load_dwordx4 v133, s[12:15], s80 offen lds
	s_mov_b32 m0, s69
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[46:49], v[66:69], a[8:11], v101, v98 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[46:49], v[70:73], a[4:7], v101, v98 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[46:49], v[74:77], a[0:3], v101, v99 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[46:49], v[78:81], a[224:227], v101, v99 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[62:65], v[82:85], a[8:11], v101, v98 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[62:65], v[86:89], a[4:7], v101, v98 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[62:65], v[90:93], a[0:3], v101, v99 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[62:65], v[94:97], a[224:227], v101, v99 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[100:101], v137, s[4:7], s81 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[98:99], v137, s[0:3], s81 offen
	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v135, s[12:15], s80 offen lds
	s_mov_b32 m0, s70
	s_nop 0
	buffer_load_dwordx4 v136, s[12:15], s80 offen lds
	s_add_u32 s80, s76, 0x300
	s_mov_b32 m0, s25
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255],  v[104:107], v[2:5], a[252:255],  v100, v102 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[66:69], v157 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251],  v[104:107], v[6:9], a[248:251],  v100, v102 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[70:73], v157 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247],  v[104:107], v[10:13], a[244:247],  v100, v103 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[74:77], v157 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243],  v[104:107], v[14:17], a[240:243],  v100, v103 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[78:81], v157 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255],  v[120:123], v[18:21], a[252:255],  v100, v102 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[82:85], v156 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251],  v[120:123], v[22:25], a[248:251],  v100, v102 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[86:89], v156 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247],  v[120:123], v[26:29], a[244:247],  v100, v103 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[90:93], v156 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243],  v[120:123], v[30:33], a[240:243],  v100, v103 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[94:97], v156 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239],  v[108:111], v[2:5], a[236:239],  v100, v102 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235],  v[108:111], v[6:9], a[232:235],  v100, v102 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[108:111], v[10:13], a[228:231],  v100, v103 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[108:111], v[14:17], a[220:223],  v100, v103 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239],  v[124:127], v[18:21], a[236:239],  v100, v102 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235],  v[124:127], v[22:25], a[232:235],  v100, v102 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[124:127], v[26:29], a[228:231],  v100, v103 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[124:127], v[30:33], a[220:223],  v100, v103 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[112:115], v[2:5], a[216:219],  v101, v102 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[112:115], v[6:9], a[212:215],  v101, v102 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[112:115], v[10:13], a[204:207], v101, v103 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[112:115], v[14:17], a[196:199], v101, v103 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[140:143], v[18:21], a[216:219],  v101, v102 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[140:143], v[22:25], a[212:215],  v101, v102 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[140:143], v[26:29], a[204:207], v101, v103 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[140:143], v[30:33], a[196:199], v101, v103 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[116:119], v[2:5], a[188:191], v101, v102 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[116:119], v[6:9], a[172:175], v101, v102 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[116:119], v[10:13], a[160:163], v101, v103 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[116:119], v[14:17], a[144:147], v101, v103 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[144:147], v[18:21], a[188:191], v101, v102 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[144:147], v[22:25], a[172:175], v101, v102 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[144:147], v[26:29], a[160:163], v101, v103 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[144:147], v[30:33], a[144:147], v101, v103 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[104:107], v[66:69], a[208:211], v100, v98 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v153 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[104:107], v[70:73], a[200:203], v100, v98 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v153 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[104:107], v[74:77], a[192:195], v100, v99 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v153 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[104:107], v[78:81], a[184:187], v100, v99 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v153 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[120:123], v[82:85], a[208:211], v100, v98 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v152 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[120:123], v[86:89], a[200:203], v100, v98 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v152 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[120:123], v[90:93], a[192:195], v100, v99 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v152 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[120:123], v[94:97], a[184:187], v100, v99 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[62:65], v152 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[108:111], v[66:69], a[180:183], v100, v98 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[108:111], v[70:73], a[168:171], v100, v98 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[108:111], v[74:77], a[152:155], v100, v99 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[108:111], v[78:81], a[136:139], v100, v99 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[124:127], v[82:85], a[180:183], v100, v98 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[124:127], v[86:89], a[168:171], v100, v98 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[124:127], v[90:93], a[152:155], v100, v99 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[124:127], v[94:97], a[136:139], v100, v99 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[112:115], v[66:69], a[128:131], v101, v98 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[112:115], v[70:73], a[116:119], v101, v98 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[112:115], v[74:77], a[104:107], v101, v99 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[112:115], v[78:81], a[92:95], v101, v99 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[140:143], v[82:85], a[128:131], v101, v98 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[140:143], v[86:89], a[116:119], v101, v98 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[140:143], v[90:93], a[104:107], v101, v99 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[140:143], v[94:97], a[92:95], v101, v99 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[116:119], v[66:69], a[80:83], v101, v98 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[116:119], v[70:73], a[68:71], v101, v98 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[116:119], v[74:77], a[56:59], v101, v99 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[116:119], v[78:81], a[40:43], v101, v99 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[144:147], v[82:85], a[80:83], v101, v98 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[144:147], v[86:89], a[68:71], v101, v98 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[144:147], v[90:93], a[56:59], v101, v99 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[144:147], v[94:97], a[40:43], v101, v99 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[100:101], v137, s[20:23], s81 offen
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(20)
s_barrier

	;;#ASMEND
	s_add_i32 s81, s27, 0xfffffa00
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179],  v[34:37], v[2:5], a[176:179],  v100, v102 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[104:107], v161 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[34:37], v[6:9], a[164:167],  v100, v102 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[108:111], v161 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[34:37], v[10:13], a[148:151],  v100, v103 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[112:115], v161 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[34:37], v[14:17], a[132:135],  v100, v103 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[116:119], v161 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179],  v[50:53], v[18:21], a[176:179],  v100, v102 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[120:123], v160 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[50:53], v[22:25], a[164:167],  v100, v102 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[124:127], v160 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[50:53], v[26:29], a[148:151],  v100, v103 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[140:143], v160 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[50:53], v[30:33], a[132:135],  v100, v103 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[144:147], v160 offset:6144

	;;#ASMEND
	buffer_load_dwordx4 v130, s[8:11], s80 offen lds
	s_mov_b32 m0, s35
	s_add_u32 s76, s76, 0x380
	buffer_load_dwordx4 v131, s[8:11], s80 offen lds
	s_mov_b32 m0, s46
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123],  v[38:41], v[2:5], a[120:123],  v100, v102 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111],  v[38:41], v[6:9], a[108:111],  v100, v102 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99],  v[38:41], v[10:13], a[96:99],  v100, v103 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91],  v[38:41], v[14:17], a[88:91],  v100, v103 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123],  v[54:57], v[18:21], a[120:123],  v100, v102 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111],  v[54:57], v[22:25], a[108:111],  v100, v102 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99],  v[54:57], v[26:29], a[96:99],  v100, v103 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91],  v[54:57], v[30:33], a[88:91],  v100, v103 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v139, s[8:11], s80 offen lds
	s_mov_b32 m0, s47
	s_nop 0
	buffer_load_dwordx4 v132, s[8:11], s80 offen lds
	s_add_u32 s80, s77, 0x300
	s_mov_b32 m0, s33
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79],  v[42:45], v[2:5], a[76:79],  v101, v102 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[42:45], v[6:9], a[64:67],  v101, v102 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[42:45], v[10:13], a[52:55], v101, v103 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[42:45], v[14:17], a[36:39], v101, v103 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79],  v[58:61], v[18:21], a[76:79],  v101, v102 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[58:61], v[22:25], a[64:67],  v101, v102 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[58:61], v[26:29], a[52:55], v101, v103 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[58:61], v[30:33], a[36:39], v101, v103 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v130, s[8:11], s80 offen lds
	s_mov_b32 m0, s49
	s_nop 0
	buffer_load_dwordx4 v131, s[8:11], s80 offen lds
	s_mov_b32 m0, s50
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[46:49], v[2:5], a[28:31], v101, v102 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[46:49], v[6:9], a[20:23], v101, v102 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[46:49], v[10:13], a[16:19], v101, v103 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[46:49], v[14:17], a[48:51], v101, v103 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[62:65], v[18:21], a[28:31], v101, v102 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[62:65], v[22:25], a[20:23], v101, v102 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[62:65], v[26:29], a[16:19], v101, v103 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[62:65], v[30:33], a[48:51], v101, v103 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[102:103], v137, s[16:19], s81 offen
	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v139, s[8:11], s80 offen lds
	s_mov_b32 m0, s51
	s_nop 0
	buffer_load_dwordx4 v132, s[8:11], s80 offen lds
	s_add_u32 s80, s78, 0x300
	s_mov_b32 m0, s38
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[34:37], v[66:69], a[156:159],  v100, v98 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[2:5], v159 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[34:37], v[70:73], a[140:143],  v100, v98 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[6:9], v159 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127],  v[34:37], v[74:77], a[124:127],  v100, v99 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[10:13], v159 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115],  v[34:37], v[78:81], a[112:115],  v100, v99 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[14:17], v159 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[50:53], v[82:85], a[156:159],  v100, v98 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[18:21], v158 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[50:53], v[86:89], a[140:143],  v100, v98 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[22:25], v158 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127],  v[50:53], v[90:93], a[124:127],  v100, v99 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[26:29], v158 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115],  v[50:53], v[94:97], a[112:115],  v100, v99 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v158 offset:6144

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v134, s[12:15], s80 offen lds
	s_mov_b32 m0, s31
	s_nop 0
	buffer_load_dwordx4 v133, s[12:15], s80 offen lds
	s_mov_b32 m0, s53
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103],  v[38:41], v[66:69], a[100:103],  v100, v98 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87],  v[38:41], v[70:73], a[84:87],  v100, v98 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75],  v[38:41], v[74:77], a[72:75],  v100, v99 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[38:41], v[78:81], a[60:63],  v100, v99 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103],  v[54:57], v[82:85], a[100:103],  v100, v98 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87],  v[54:57], v[86:89], a[84:87],  v100, v98 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75],  v[54:57], v[90:93], a[72:75],  v100, v99 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[54:57], v[94:97], a[60:63],  v100, v99 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v135, s[12:15], s80 offen lds
	s_mov_b32 m0, s54
	s_nop 0
	buffer_load_dwordx4 v136, s[12:15], s80 offen lds
	s_add_u32 s80, s79, 0x300
	s_mov_b32 m0, s39
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[42:45], v[66:69], a[44:47],  v101, v98 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[42:45], v[70:73], a[32:35],  v101, v98 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[42:45], v[74:77], a[24:27], v101, v99 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[42:45], v[78:81], a[12:15], v101, v99 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[58:61], v[82:85], a[44:47],  v101, v98 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[58:61], v[86:89], a[32:35],  v101, v98 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[58:61], v[90:93], a[24:27], v101, v99 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[58:61], v[94:97], a[12:15], v101, v99 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v134, s[12:15], s80 offen lds
	s_mov_b32 m0, s56
	s_nop 0
	buffer_load_dwordx4 v133, s[12:15], s80 offen lds
	s_mov_b32 m0, s57
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[46:49], v[66:69], a[8:11], v101, v98 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[46:49], v[70:73], a[4:7], v101, v98 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[46:49], v[74:77], a[0:3], v101, v99 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[46:49], v[78:81], a[224:227], v101, v99 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[62:65], v[82:85], a[8:11], v101, v98 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[62:65], v[86:89], a[4:7], v101, v98 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[62:65], v[90:93], a[0:3], v101, v99 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[62:65], v[94:97], a[224:227], v101, v99 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[100:101], v137, s[4:7], s81 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[98:99], v137, s[0:3], s81 offen
	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v135, s[12:15], s80 offen lds
	s_mov_b32 m0, s58
	s_nop 0
	buffer_load_dwordx4 v136, s[12:15], s80 offen lds
	s_mov_b32 m0, s43
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255],  v[104:107], v[2:5], a[252:255],  v100, v102 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[66:69], v155 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251],  v[104:107], v[6:9], a[248:251],  v100, v102 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[70:73], v155 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247],  v[104:107], v[10:13], a[244:247],  v100, v103 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[74:77], v155 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243],  v[104:107], v[14:17], a[240:243],  v100, v103 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[78:81], v155 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255],  v[120:123], v[18:21], a[252:255],  v100, v102 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[82:85], v154 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251],  v[120:123], v[22:25], a[248:251],  v100, v102 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[86:89], v154 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247],  v[120:123], v[26:29], a[244:247],  v100, v103 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[90:93], v154 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243],  v[120:123], v[30:33], a[240:243],  v100, v103 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[94:97], v154 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239],  v[108:111], v[2:5], a[236:239],  v100, v102 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235],  v[108:111], v[6:9], a[232:235],  v100, v102 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[108:111], v[10:13], a[228:231],  v100, v103 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[108:111], v[14:17], a[220:223],  v100, v103 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239],  v[124:127], v[18:21], a[236:239],  v100, v102 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235],  v[124:127], v[22:25], a[232:235],  v100, v102 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[124:127], v[26:29], a[228:231],  v100, v103 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[124:127], v[30:33], a[220:223],  v100, v103 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[112:115], v[2:5], a[216:219],  v101, v102 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[112:115], v[6:9], a[212:215],  v101, v102 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[112:115], v[10:13], a[204:207], v101, v103 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[112:115], v[14:17], a[196:199], v101, v103 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[140:143], v[18:21], a[216:219],  v101, v102 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[140:143], v[22:25], a[212:215],  v101, v102 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[140:143], v[26:29], a[204:207], v101, v103 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[140:143], v[30:33], a[196:199], v101, v103 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[116:119], v[2:5], a[188:191], v101, v102 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[116:119], v[6:9], a[172:175], v101, v102 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[116:119], v[10:13], a[160:163], v101, v103 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[116:119], v[14:17], a[144:147], v101, v103 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[144:147], v[18:21], a[188:191], v101, v102 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[144:147], v[22:25], a[172:175], v101, v102 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[144:147], v[26:29], a[160:163], v101, v103 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[144:147], v[30:33], a[144:147], v101, v103 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[104:107], v[66:69], a[208:211], v100, v98 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v151 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[104:107], v[70:73], a[200:203], v100, v98 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v151 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[104:107], v[74:77], a[192:195], v100, v99 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v151 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[104:107], v[78:81], a[184:187], v100, v99 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v151 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[120:123], v[82:85], a[208:211], v100, v98 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v150 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[120:123], v[86:89], a[200:203], v100, v98 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v150 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[120:123], v[90:93], a[192:195], v100, v99 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v150 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[120:123], v[94:97], a[184:187], v100, v99 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[62:65], v150 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[108:111], v[66:69], a[180:183], v100, v98 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[108:111], v[70:73], a[168:171], v100, v98 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[108:111], v[74:77], a[152:155], v100, v99 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[108:111], v[78:81], a[136:139], v100, v99 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[124:127], v[82:85], a[180:183], v100, v98 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[124:127], v[86:89], a[168:171], v100, v98 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[124:127], v[90:93], a[152:155], v100, v99 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[124:127], v[94:97], a[136:139], v100, v99 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[112:115], v[66:69], a[128:131], v101, v98 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[112:115], v[70:73], a[116:119], v101, v98 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[112:115], v[74:77], a[104:107], v101, v99 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[112:115], v[78:81], a[92:95], v101, v99 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[140:143], v[82:85], a[128:131], v101, v98 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[140:143], v[86:89], a[116:119], v101, v98 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[140:143], v[90:93], a[104:107], v101, v99 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[140:143], v[94:97], a[92:95], v101, v99 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[116:119], v[66:69], a[80:83], v101, v98 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[116:119], v[70:73], a[68:71], v101, v98 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[116:119], v[74:77], a[56:59], v101, v99 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[116:119], v[78:81], a[40:43], v101, v99 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[144:147], v[82:85], a[80:83], v101, v98 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[144:147], v[86:89], a[68:71], v101, v98 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[144:147], v[90:93], a[56:59], v101, v99 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[144:147], v[94:97], a[40:43], v101, v99 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[100:101], v137, s[20:23], s81 offen
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(20)
s_barrier

	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179],  v[34:37], v[2:5], a[176:179],  v100, v102 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[104:107], v149 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[34:37], v[6:9], a[164:167],  v100, v102 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[108:111], v149 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[34:37], v[10:13], a[148:151],  v100, v103 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[112:115], v149 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[34:37], v[14:17], a[132:135],  v100, v103 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[116:119], v149 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179],  v[50:53], v[18:21], a[176:179],  v100, v102 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[120:123], v148 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[50:53], v[22:25], a[164:167],  v100, v102 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[124:127], v148 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[50:53], v[26:29], a[148:151],  v100, v103 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[140:143], v148 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[50:53], v[30:33], a[132:135],  v100, v103 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[144:147], v148 offset:6144

	;;#ASMEND
	buffer_load_dwordx4 v130, s[8:11], s76 offen lds
	s_mov_b32 m0, s59
	s_nop 0
	buffer_load_dwordx4 v131, s[8:11], s76 offen lds
	s_mov_b32 m0, s60
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123],  v[38:41], v[2:5], a[120:123],  v100, v102 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111],  v[38:41], v[6:9], a[108:111],  v100, v102 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99],  v[38:41], v[10:13], a[96:99],  v100, v103 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91],  v[38:41], v[14:17], a[88:91],  v100, v103 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123],  v[54:57], v[18:21], a[120:123],  v100, v102 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111],  v[54:57], v[22:25], a[108:111],  v100, v102 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99],  v[54:57], v[26:29], a[96:99],  v100, v103 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91],  v[54:57], v[30:33], a[88:91],  v100, v103 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v139, s[8:11], s76 offen lds
	s_mov_b32 m0, s61
	s_nop 0
	buffer_load_dwordx4 v132, s[8:11], s76 offen lds
	s_add_u32 s76, s77, 0x380
	s_mov_b32 m0, s42
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79],  v[42:45], v[2:5], a[76:79],  v101, v102 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[42:45], v[6:9], a[64:67],  v101, v102 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[42:45], v[10:13], a[52:55], v101, v103 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[42:45], v[14:17], a[36:39], v101, v103 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79],  v[58:61], v[18:21], a[76:79],  v101, v102 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[58:61], v[22:25], a[64:67],  v101, v102 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[58:61], v[26:29], a[52:55], v101, v103 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[58:61], v[30:33], a[36:39], v101, v103 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_i32 s77, s27, 0xfffffc00
	buffer_load_dwordx4 v130, s[8:11], s76 offen lds
	s_mov_b32 m0, s62
	s_nop 0
	buffer_load_dwordx4 v131, s[8:11], s76 offen lds
	s_mov_b32 m0, s63
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[46:49], v[2:5], a[28:31], v101, v102 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[46:49], v[6:9], a[20:23], v101, v102 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[46:49], v[10:13], a[16:19], v101, v103 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[46:49], v[14:17], a[48:51], v101, v103 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[62:65], v[18:21], a[28:31], v101, v102 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[62:65], v[22:25], a[20:23], v101, v102 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[62:65], v[26:29], a[16:19], v101, v103 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[62:65], v[30:33], a[48:51], v101, v103 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[102:103], v137, s[16:19], s77 offen
	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v139, s[8:11], s76 offen lds
	s_mov_b32 m0, s64
	s_nop 0
	buffer_load_dwordx4 v132, s[8:11], s76 offen lds
	s_add_u32 s76, s78, 0x380
	s_mov_b32 m0, s41
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[34:37], v[66:69], a[156:159],  v100, v98 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[2:5], v163 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[34:37], v[70:73], a[140:143],  v100, v98 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[6:9], v163 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127],  v[34:37], v[74:77], a[124:127],  v100, v99 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[10:13], v163 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115],  v[34:37], v[78:81], a[112:115],  v100, v99 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[14:17], v163 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[50:53], v[82:85], a[156:159],  v100, v98 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[18:21], v162 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[50:53], v[86:89], a[140:143],  v100, v98 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[22:25], v162 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127],  v[50:53], v[90:93], a[124:127],  v100, v99 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[26:29], v162 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115],  v[50:53], v[94:97], a[112:115],  v100, v99 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v162 offset:6144

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v134, s[12:15], s76 offen lds
	s_mov_b32 m0, s65
	s_nop 0
	buffer_load_dwordx4 v133, s[12:15], s76 offen lds
	s_mov_b32 m0, s66
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103],  v[38:41], v[66:69], a[100:103],  v100, v98 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87],  v[38:41], v[70:73], a[84:87],  v100, v98 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75],  v[38:41], v[74:77], a[72:75],  v100, v99 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[38:41], v[78:81], a[60:63],  v100, v99 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103],  v[54:57], v[82:85], a[100:103],  v100, v98 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87],  v[54:57], v[86:89], a[84:87],  v100, v98 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75],  v[54:57], v[90:93], a[72:75],  v100, v99 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[54:57], v[94:97], a[60:63],  v100, v99 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v135, s[12:15], s76 offen lds
	s_mov_b32 m0, s67
	s_nop 0
	buffer_load_dwordx4 v136, s[12:15], s76 offen lds
	s_add_u32 s76, s79, 0x380
	s_mov_b32 m0, s40
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[42:45], v[66:69], a[44:47],  v101, v98 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[42:45], v[70:73], a[32:35],  v101, v98 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[42:45], v[74:77], a[24:27], v101, v99 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[42:45], v[78:81], a[12:15], v101, v99 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[58:61], v[82:85], a[44:47],  v101, v98 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[58:61], v[86:89], a[32:35],  v101, v98 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[58:61], v[90:93], a[24:27], v101, v99 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[58:61], v[94:97], a[12:15], v101, v99 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v134, s[12:15], s76 offen lds
	s_mov_b32 m0, s68
	s_nop 0
	buffer_load_dwordx4 v133, s[12:15], s76 offen lds
	s_mov_b32 m0, s69
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[46:49], v[66:69], a[8:11], v101, v98 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[46:49], v[70:73], a[4:7], v101, v98 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[46:49], v[74:77], a[0:3], v101, v99 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[46:49], v[78:81], a[224:227], v101, v99 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[62:65], v[82:85], a[8:11], v101, v98 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[62:65], v[86:89], a[4:7], v101, v98 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[62:65], v[90:93], a[0:3], v101, v99 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[62:65], v[94:97], a[224:227], v101, v99 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[100:101], v137, s[4:7], s77 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[98:99], v137, s[0:3], s77 offen
	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v135, s[12:15], s76 offen lds
	s_mov_b32 m0, s70
	s_nop 0
	buffer_load_dwordx4 v136, s[12:15], s76 offen lds
	s_add_i32 s76, s71, -1
	s_min_u32 s76, s76, 0x6d
	s_lshl_b32 s76, s76, 7
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255],  v[104:107], v[2:5], a[252:255],  v100, v102 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[66:69], v157 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251],  v[104:107], v[6:9], a[248:251],  v100, v102 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[70:73], v157 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247],  v[104:107], v[10:13], a[244:247],  v100, v103 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[74:77], v157 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243],  v[104:107], v[14:17], a[240:243],  v100, v103 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[78:81], v157 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255],  v[120:123], v[18:21], a[252:255],  v100, v102 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[82:85], v156 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251],  v[120:123], v[22:25], a[248:251],  v100, v102 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[86:89], v156 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247],  v[120:123], v[26:29], a[244:247],  v100, v103 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[90:93], v156 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243],  v[120:123], v[30:33], a[240:243],  v100, v103 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[94:97], v156 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239],  v[108:111], v[2:5], a[236:239],  v100, v102 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235],  v[108:111], v[6:9], a[232:235],  v100, v102 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[108:111], v[10:13], a[228:231],  v100, v103 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[108:111], v[14:17], a[220:223],  v100, v103 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239],  v[124:127], v[18:21], a[236:239],  v100, v102 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235],  v[124:127], v[22:25], a[232:235],  v100, v102 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[124:127], v[26:29], a[228:231],  v100, v103 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[124:127], v[30:33], a[220:223],  v100, v103 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[112:115], v[2:5], a[216:219],  v101, v102 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[112:115], v[6:9], a[212:215],  v101, v102 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[112:115], v[10:13], a[204:207], v101, v103 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[112:115], v[14:17], a[196:199], v101, v103 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[140:143], v[18:21], a[216:219],  v101, v102 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[140:143], v[22:25], a[212:215],  v101, v102 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[140:143], v[26:29], a[204:207], v101, v103 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[140:143], v[30:33], a[196:199], v101, v103 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[116:119], v[2:5], a[188:191], v101, v102 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[116:119], v[6:9], a[172:175], v101, v102 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[116:119], v[10:13], a[160:163], v101, v103 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[116:119], v[14:17], a[144:147], v101, v103 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[144:147], v[18:21], a[188:191], v101, v102 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[144:147], v[22:25], a[172:175], v101, v102 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[144:147], v[26:29], a[160:163], v101, v103 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[144:147], v[30:33], a[144:147], v101, v103 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[104:107], v[66:69], a[208:211], v100, v98 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v153 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[104:107], v[70:73], a[200:203], v100, v98 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v153 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[104:107], v[74:77], a[192:195], v100, v99 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v153 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[104:107], v[78:81], a[184:187], v100, v99 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v153 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[120:123], v[82:85], a[208:211], v100, v98 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v152 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[120:123], v[86:89], a[200:203], v100, v98 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v152 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[120:123], v[90:93], a[192:195], v100, v99 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v152 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[120:123], v[94:97], a[184:187], v100, v99 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[62:65], v152 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[108:111], v[66:69], a[180:183], v100, v98 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[108:111], v[70:73], a[168:171], v100, v98 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[108:111], v[74:77], a[152:155], v100, v99 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[108:111], v[78:81], a[136:139], v100, v99 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[124:127], v[82:85], a[180:183], v100, v98 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[124:127], v[86:89], a[168:171], v100, v98 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[124:127], v[90:93], a[152:155], v100, v99 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[124:127], v[94:97], a[136:139], v100, v99 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[112:115], v[66:69], a[128:131], v101, v98 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[112:115], v[70:73], a[116:119], v101, v98 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[112:115], v[74:77], a[104:107], v101, v99 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[112:115], v[78:81], a[92:95], v101, v99 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[140:143], v[82:85], a[128:131], v101, v98 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[140:143], v[86:89], a[116:119], v101, v98 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[140:143], v[90:93], a[104:107], v101, v99 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[140:143], v[94:97], a[92:95], v101, v99 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[116:119], v[66:69], a[80:83], v101, v98 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[116:119], v[70:73], a[68:71], v101, v98 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[116:119], v[74:77], a[56:59], v101, v99 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[116:119], v[78:81], a[40:43], v101, v99 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[144:147], v[82:85], a[80:83], v101, v98 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[144:147], v[86:89], a[68:71], v101, v98 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[144:147], v[90:93], a[56:59], v101, v99 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[144:147], v[94:97], a[40:43], v101, v99 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[100:101], v137, s[20:23], s77 offen
	;;#ASMEND
	s_add_u32 s77, s72, s76
	s_sub_u32 s77, s77, s34
	s_mov_b32 m0, s25
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(20)
s_barrier

	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179],  v[34:37], v[2:5], a[176:179],  v100, v102 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[164:167], v161 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[34:37], v[6:9], a[164:167],  v100, v102 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[168:171], v161 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[34:37], v[10:13], a[148:151],  v100, v103 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[172:175], v161 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[34:37], v[14:17], a[132:135],  v100, v103 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[176:179], v161 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179],  v[50:53], v[18:21], a[176:179],  v100, v102 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[180:183], v160 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[50:53], v[22:25], a[164:167],  v100, v102 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[184:187], v160 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[50:53], v[26:29], a[148:151],  v100, v103 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[188:191], v160 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[50:53], v[30:33], a[132:135],  v100, v103 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[192:195], v160 offset:6144

	;;#ASMEND
	s_add_u32 s78, s73, s76
	buffer_load_dwordx4 v130, s[8:11], s77 offen lds
	s_mov_b32 m0, s35
	s_nop 0
	buffer_load_dwordx4 v131, s[8:11], s77 offen lds
	s_mov_b32 m0, s46
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123],  v[38:41], v[2:5], a[120:123],  v100, v102 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111],  v[38:41], v[6:9], a[108:111],  v100, v102 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99],  v[38:41], v[10:13], a[96:99],  v100, v103 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91],  v[38:41], v[14:17], a[88:91],  v100, v103 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123],  v[54:57], v[18:21], a[120:123],  v100, v102 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111],  v[54:57], v[22:25], a[108:111],  v100, v102 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99],  v[54:57], v[26:29], a[96:99],  v100, v103 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91],  v[54:57], v[30:33], a[88:91],  v100, v103 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v139, s[8:11], s77 offen lds
	s_mov_b32 m0, s47
	s_nop 0
	buffer_load_dwordx4 v132, s[8:11], s77 offen lds
	s_sub_u32 s77, s78, s34
	s_mov_b32 m0, s33
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79],  v[42:45], v[2:5], a[76:79],  v101, v102 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[42:45], v[6:9], a[64:67],  v101, v102 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[42:45], v[10:13], a[52:55], v101, v103 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[42:45], v[14:17], a[36:39], v101, v103 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79],  v[58:61], v[18:21], a[76:79],  v101, v102 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[58:61], v[22:25], a[64:67],  v101, v102 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[58:61], v[26:29], a[52:55], v101, v103 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[58:61], v[30:33], a[36:39], v101, v103 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_u32 s78, s74, s76
	buffer_load_dwordx4 v130, s[8:11], s77 offen lds
	s_mov_b32 m0, s49
	s_add_u32 s76, s75, s76
	buffer_load_dwordx4 v131, s[8:11], s77 offen lds
	s_mov_b32 m0, s50
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[46:49], v[2:5], a[28:31], v101, v102 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[46:49], v[6:9], a[20:23], v101, v102 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[46:49], v[10:13], a[16:19], v101, v103 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[46:49], v[14:17], a[48:51], v101, v103 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[62:65], v[18:21], a[28:31], v101, v102 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[62:65], v[22:25], a[20:23], v101, v102 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[62:65], v[26:29], a[16:19], v101, v103 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[62:65], v[30:33], a[48:51], v101, v103 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_sub_u32 s76, s76, s30
	buffer_load_dwordx4 v139, s[8:11], s77 offen lds
	s_mov_b32 m0, s51
	s_nop 0
	buffer_load_dwordx4 v132, s[8:11], s77 offen lds
	s_sub_u32 s77, s78, s30
	s_mov_b32 m0, s38
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[34:37], v[66:69], a[156:159],  v100, v98 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[2:5], v159 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[34:37], v[70:73], a[140:143],  v100, v98 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[6:9], v159 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127],  v[34:37], v[74:77], a[124:127],  v100, v99 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[10:13], v159 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115],  v[34:37], v[78:81], a[112:115],  v100, v99 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[14:17], v159 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[50:53], v[82:85], a[156:159],  v100, v98 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[18:21], v158 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[50:53], v[86:89], a[140:143],  v100, v98 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[22:25], v158 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127],  v[50:53], v[90:93], a[124:127],  v100, v99 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[26:29], v158 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115],  v[50:53], v[94:97], a[112:115],  v100, v99 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v158 offset:6144

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v134, s[12:15], s77 offen lds
	s_mov_b32 m0, s31
	s_nop 0
	buffer_load_dwordx4 v133, s[12:15], s77 offen lds
	s_mov_b32 m0, s53
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103],  v[38:41], v[66:69], a[100:103],  v100, v98 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87],  v[38:41], v[70:73], a[84:87],  v100, v98 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75],  v[38:41], v[74:77], a[72:75],  v100, v99 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[38:41], v[78:81], a[60:63],  v100, v99 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103],  v[54:57], v[82:85], a[100:103],  v100, v98 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87],  v[54:57], v[86:89], a[84:87],  v100, v98 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75],  v[54:57], v[90:93], a[72:75],  v100, v99 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[54:57], v[94:97], a[60:63],  v100, v99 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v135, s[12:15], s77 offen lds
	s_mov_b32 m0, s54
	s_nop 0
	buffer_load_dwordx4 v136, s[12:15], s77 offen lds
	s_mov_b32 m0, s39
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[42:45], v[66:69], a[44:47],  v101, v98 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[42:45], v[70:73], a[32:35],  v101, v98 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[42:45], v[74:77], a[24:27], v101, v99 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[42:45], v[78:81], a[12:15], v101, v99 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[58:61], v[82:85], a[44:47],  v101, v98 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[58:61], v[86:89], a[32:35],  v101, v98 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[58:61], v[90:93], a[24:27], v101, v99 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[58:61], v[94:97], a[12:15], v101, v99 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_i32 s77, s27, 0xfffffe00
	buffer_load_dwordx4 v134, s[12:15], s76 offen lds
	s_mov_b32 m0, s56
	;;#ASMSTART
	buffer_load_dwordx2 v[140:141], v137, s[0:3], s77 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[144:145], v137, s[16:19], s77 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[142:143], v137, s[20:23], s77 offen
	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v133, s[12:15], s76 offen lds
	s_mov_b32 m0, s57
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[46:49], v[66:69], a[8:11], v101, v98 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[46:49], v[70:73], a[4:7], v101, v98 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[46:49], v[74:77], a[0:3], v101, v99 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[46:49], v[78:81], a[224:227], v101, v99 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[62:65], v[82:85], a[8:11], v101, v98 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[62:65], v[86:89], a[4:7], v101, v98 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[62:65], v[90:93], a[0:3], v101, v99 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[62:65], v[94:97], a[224:227], v101, v99 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[34:35], v137, s[4:7], s77 offen
	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v135, s[12:15], s76 offen lds
	s_mov_b32 m0, s58
	s_nop 0
	buffer_load_dwordx4 v136, s[12:15], s76 offen lds
	s_min_u32 s76, s71, 0x6d
	s_lshl_b32 s76, s76, 7
	s_add_u32 s77, s72, s76
	s_sub_u32 s77, s77, s34
	s_mov_b32 m0, s43
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255],  v[164:167], v[2:5], a[252:255],  v34, v144 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[98:101], v155 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251],  v[164:167], v[6:9], a[248:251],  v34, v144 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[102:105], v155 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247],  v[164:167], v[10:13], a[244:247],  v34, v145 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[106:109], v155 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243],  v[164:167], v[14:17], a[240:243],  v34, v145 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[110:113], v155 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255],  v[180:183], v[18:21], a[252:255],  v34, v144 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[114:117], v154 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251],  v[180:183], v[22:25], a[248:251],  v34, v144 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[118:121], v154 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247],  v[180:183], v[26:29], a[244:247],  v34, v145 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[122:125], v154 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243],  v[180:183], v[30:33], a[240:243],  v34, v145 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[126:129], v154 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239],  v[168:171], v[2:5], a[236:239],  v34, v144 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235],  v[168:171], v[6:9], a[232:235],  v34, v144 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[168:171], v[10:13], a[228:231],  v34, v145 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[168:171], v[14:17], a[220:223],  v34, v145 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239],  v[184:187], v[18:21], a[236:239],  v34, v144 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235],  v[184:187], v[22:25], a[232:235],  v34, v144 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[184:187], v[26:29], a[228:231],  v34, v145 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[184:187], v[30:33], a[220:223],  v34, v145 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[172:175], v[2:5], a[216:219],  v35, v144 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[172:175], v[6:9], a[212:215],  v35, v144 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[172:175], v[10:13], a[204:207], v35, v145 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[172:175], v[14:17], a[196:199], v35, v145 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[188:191], v[18:21], a[216:219],  v35, v144 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[188:191], v[22:25], a[212:215],  v35, v144 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[188:191], v[26:29], a[204:207], v35, v145 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[188:191], v[30:33], a[196:199], v35, v145 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[176:179], v[2:5], a[188:191], v35, v144 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[176:179], v[6:9], a[172:175], v35, v144 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[176:179], v[10:13], a[160:163], v35, v145 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[176:179], v[14:17], a[144:147], v35, v145 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[192:195], v[18:21], a[188:191], v35, v144 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[192:195], v[22:25], a[172:175], v35, v144 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[192:195], v[26:29], a[160:163], v35, v145 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[192:195], v[30:33], a[144:147], v35, v145 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[164:167], v[98:101], a[208:211], v34, v140 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[66:69], v151 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[164:167], v[102:105], a[200:203], v34, v140 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[70:73], v151 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[164:167], v[106:109], a[192:195], v34, v141 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[74:77], v151 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[164:167], v[110:113], a[184:187], v34, v141 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[78:81], v151 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[180:183], v[114:117], a[208:211], v34, v140 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[82:85], v150 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[180:183], v[118:121], a[200:203], v34, v140 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[86:89], v150 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[180:183], v[122:125], a[192:195], v34, v141 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[90:93], v150 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[180:183], v[126:129], a[184:187], v34, v141 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[94:97], v150 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[168:171], v[98:101], a[180:183], v34, v140 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[168:171], v[102:105], a[168:171], v34, v140 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[168:171], v[106:109], a[152:155], v34, v141 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[168:171], v[110:113], a[136:139], v34, v141 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[184:187], v[114:117], a[180:183], v34, v140 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[184:187], v[118:121], a[168:171], v34, v140 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[184:187], v[122:125], a[152:155], v34, v141 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[184:187], v[126:129], a[136:139], v34, v141 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[172:175], v[98:101], a[128:131], v35, v140 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[172:175], v[102:105], a[116:119], v35, v140 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[172:175], v[106:109], a[104:107], v35, v141 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[172:175], v[110:113], a[92:95], v35, v141 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[188:191], v[114:117], a[128:131], v35, v140 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[188:191], v[118:121], a[116:119], v35, v140 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[188:191], v[122:125], a[104:107], v35, v141 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[188:191], v[126:129], a[92:95], v35, v141 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[176:179], v[98:101], a[80:83], v35, v140 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[176:179], v[102:105], a[68:71], v35, v140 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[176:179], v[106:109], a[56:59], v35, v141 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[176:179], v[110:113], a[40:43], v35, v141 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[192:195], v[114:117], a[80:83], v35, v140 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[192:195], v[118:121], a[68:71], v35, v140 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[192:195], v[122:125], a[56:59], v35, v141 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[192:195], v[126:129], a[40:43], v35, v141 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(20)
s_barrier

	;;#ASMEND
	s_add_u32 s78, s73, s76
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179],  v[66:69], v[2:5], a[176:179],  v142, v144 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v149 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[66:69], v[6:9], a[164:167],  v142, v144 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v149 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[66:69], v[10:13], a[148:151],  v142, v145 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v149 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[66:69], v[14:17], a[132:135],  v142, v145 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v149 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179],  v[82:85], v[18:21], a[176:179],  v142, v144 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v148 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[82:85], v[22:25], a[164:167],  v142, v144 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v148 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[82:85], v[26:29], a[148:151],  v142, v145 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v148 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[82:85], v[30:33], a[132:135],  v142, v145 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[62:65], v148 offset:6144

	;;#ASMEND
	buffer_load_dwordx4 v130, s[8:11], s77 offen lds
	s_mov_b32 m0, s59
	s_nop 0
	buffer_load_dwordx4 v131, s[8:11], s77 offen lds
	s_mov_b32 m0, s60
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123],  v[70:73], v[2:5], a[120:123],  v142, v144 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111],  v[70:73], v[6:9], a[108:111],  v142, v144 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99],  v[70:73], v[10:13], a[96:99],  v142, v145 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91],  v[70:73], v[14:17], a[88:91],  v142, v145 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123],  v[86:89], v[18:21], a[120:123],  v142, v144 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111],  v[86:89], v[22:25], a[108:111],  v142, v144 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99],  v[86:89], v[26:29], a[96:99],  v142, v145 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91],  v[86:89], v[30:33], a[88:91],  v142, v145 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v139, s[8:11], s77 offen lds
	s_mov_b32 m0, s61
	s_nop 0
	buffer_load_dwordx4 v132, s[8:11], s77 offen lds
	s_sub_u32 s77, s78, s34
	s_mov_b32 m0, s42
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79],  v[74:77], v[2:5], a[76:79],  v143, v144 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[74:77], v[6:9], a[64:67],  v143, v144 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[74:77], v[10:13], a[52:55], v143, v145 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[74:77], v[14:17], a[36:39], v143, v145 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79],  v[90:93], v[18:21], a[76:79],  v143, v144 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[90:93], v[22:25], a[64:67],  v143, v144 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[90:93], v[26:29], a[52:55], v143, v145 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[90:93], v[30:33], a[36:39], v143, v145 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_u32 s78, s74, s76
	buffer_load_dwordx4 v130, s[8:11], s77 offen lds
	s_mov_b32 m0, s62
	s_add_u32 s76, s75, s76
	buffer_load_dwordx4 v131, s[8:11], s77 offen lds
	s_mov_b32 m0, s63
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[78:81], v[2:5], a[28:31], v143, v144 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[78:81], v[6:9], a[20:23], v143, v144 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[78:81], v[10:13], a[16:19], v143, v145 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[78:81], v[14:17], a[48:51], v143, v145 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[94:97], v[18:21], a[28:31], v143, v144 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[94:97], v[22:25], a[20:23], v143, v144 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[94:97], v[26:29], a[16:19], v143, v145 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[94:97], v[30:33], a[48:51], v143, v145 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_sub_u32 s76, s76, s30
	buffer_load_dwordx4 v139, s[8:11], s77 offen lds
	s_mov_b32 m0, s64
	s_nop 0
	buffer_load_dwordx4 v132, s[8:11], s77 offen lds
	s_sub_u32 s77, s78, s30
	s_mov_b32 m0, s41
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[66:69], v[98:101], a[156:159],  v142, v140 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[2:5], v163 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[66:69], v[102:105], a[140:143],  v142, v140 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[6:9], v163 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127],  v[66:69], v[106:109], a[124:127],  v142, v141 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[10:13], v163 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115],  v[66:69], v[110:113], a[112:115],  v142, v141 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[14:17], v163 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[82:85], v[114:117], a[156:159],  v142, v140 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[18:21], v162 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[82:85], v[118:121], a[140:143],  v142, v140 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[22:25], v162 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127],  v[82:85], v[122:125], a[124:127],  v142, v141 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[26:29], v162 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115],  v[82:85], v[126:129], a[112:115],  v142, v141 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v162 offset:6144

	;;#ASMEND
	s_cmpk_lg_i32 s36, 0x3400
	buffer_load_dwordx4 v134, s[12:15], s77 offen lds
	s_mov_b32 m0, s65
	s_nop 0
	buffer_load_dwordx4 v133, s[12:15], s77 offen lds
	s_mov_b32 m0, s66
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103],  v[70:73], v[98:101], a[100:103],  v142, v140 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87],  v[70:73], v[102:105], a[84:87],  v142, v140 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75],  v[70:73], v[106:109], a[72:75],  v142, v141 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[70:73], v[110:113], a[60:63],  v142, v141 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103],  v[86:89], v[114:117], a[100:103],  v142, v140 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87],  v[86:89], v[118:121], a[84:87],  v142, v140 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75],  v[86:89], v[122:125], a[72:75],  v142, v141 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[86:89], v[126:129], a[60:63],  v142, v141 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v135, s[12:15], s77 offen lds
	s_mov_b32 m0, s67
	s_nop 0
	buffer_load_dwordx4 v136, s[12:15], s77 offen lds
	s_mov_b32 m0, s40
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[74:77], v[98:101], a[44:47],  v143, v140 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[74:77], v[102:105], a[32:35],  v143, v140 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[74:77], v[106:109], a[24:27], v143, v141 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[74:77], v[110:113], a[12:15], v143, v141 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[90:93], v[114:117], a[44:47],  v143, v140 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[90:93], v[118:121], a[32:35],  v143, v140 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[90:93], v[122:125], a[24:27], v143, v141 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[90:93], v[126:129], a[12:15], v143, v141 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v134, s[12:15], s76 offen lds
	s_mov_b32 m0, s68
	s_nop 0
	buffer_load_dwordx4 v133, s[12:15], s76 offen lds
	s_mov_b32 m0, s69
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[78:81], v[98:101], a[8:11], v143, v140 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[78:81], v[102:105], a[4:7], v143, v140 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[78:81], v[106:109], a[0:3], v143, v141 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[78:81], v[110:113], a[224:227], v143, v141 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[94:97], v[114:117], a[8:11], v143, v140 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[94:97], v[118:121], a[4:7], v143, v140 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[94:97], v[122:125], a[0:3], v143, v141 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[94:97], v[126:129], a[224:227], v143, v141 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v135, s[12:15], s76 offen lds
	s_mov_b32 m0, s70
	s_nop 0
	buffer_load_dwordx4 v136, s[12:15], s76 offen lds
	s_cselect_b32 s76, s27, 0xde00
	s_add_u32 s36, s36, 0x400
	s_addc_u32 s37, s37, 0
	s_add_i32 s71, s71, 8
	s_addk_i32 s27, 0x1000
	s_cmpk_eq_i32 s36, 0x3800
	;;#ASMSTART
	buffer_load_dwordx2 v[146:147], v137, s[4:7], s76 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[144:145], v137, s[16:19], s76 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[142:143], v137, s[0:3], s76 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[140:141], v137, s[20:23], s76 offen
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_cbranch_scc0 .LBB0_7
; %bb.8:
	v_lshrrev_b32_e32 v2, 2, v0
	v_and_b32_e32 v29, 12, v2
	v_and_b32_e32 v28, 15, v0
	v_lshl_or_b32 v74, s44, 8, v1
	v_mad_u64_u32 v[0:1], s[0:1], v29, s26, v[28:29]
	v_mad_i64_i32 v[2:3], s[0:1], s26, v74, 0
	v_ashrrev_i32_e32 v139, 31, v138
	v_add_u32_e32 v4, s26, v0
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshlrev_b64 v[32:33], 1, v[138:139]
	v_lshl_add_u64 v[2:3], v[2:3], 1, s[28:29]
	v_ashrrev_i32_e32 v5, 31, v4
	v_add_u32_e32 v10, s26, v4
	v_accvgpr_read_b32 v6, a252
	v_lshlrev_b64 v[0:1], 1, v[0:1]
	v_lshl_add_u64 v[38:39], v[2:3], 0, v[32:33]
	v_accvgpr_read_b32 v7, a253
	v_lshlrev_b64 v[2:3], 1, v[4:5]
	v_add_u32_e32 v12, s26, v10
	v_accvgpr_read_b32 v8, a254
	v_accvgpr_read_b32 v9, a255
	v_lshl_add_u64 v[40:41], v[38:39], 0, v[0:1]
	v_ashrrev_i32_e32 v11, 31, v10
	v_pk_mul_f32 v[6:7], v[6:7], s[24:25] op_sel_hi:[1,0]
	v_lshl_add_u64 v[42:43], v[38:39], 0, v[2:3]
	v_ashrrev_i32_e32 v13, 31, v12
	global_store_short_d16_hi v[40:41], v6, off
	v_lshlrev_b64 v[4:5], 1, v[10:11]
	global_store_short_d16_hi v[42:43], v7, off
	v_lshlrev_b64 v[6:7], 1, v[12:13]
	v_pk_mul_f32 v[12:13], v[8:9], s[24:25] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v8, a248
	v_accvgpr_read_b32 v9, a249
	v_lshl_add_u64 v[44:45], v[38:39], 0, v[4:5]
	v_accvgpr_read_b32 v10, a250
	v_accvgpr_read_b32 v11, a251
	v_lshl_add_u64 v[46:47], v[38:39], 0, v[6:7]
	v_pk_mul_f32 v[8:9], v[8:9], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[44:45], v12, off
	global_store_short_d16_hi v[46:47], v13, off
	global_store_short_d16_hi v[40:41], v8, off offset:32
	v_pk_mul_f32 v[12:13], v[10:11], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[42:43], v9, off offset:32
	v_accvgpr_read_b32 v8, a244
	v_accvgpr_read_b32 v9, a245
	v_accvgpr_read_b32 v10, a246
	v_accvgpr_read_b32 v11, a247
	global_store_short_d16_hi v[44:45], v12, off offset:32
	v_pk_mul_f32 v[8:9], v[8:9], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[46:47], v13, off offset:32
	global_store_short_d16_hi v[40:41], v8, off offset:64
	v_pk_mul_f32 v[12:13], v[10:11], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[42:43], v9, off offset:64
	v_accvgpr_read_b32 v8, a240
	global_store_short_d16_hi v[44:45], v12, off offset:64
	v_or_b32_e32 v12, 16, v29
	v_accvgpr_read_b32 v9, a241
	v_mad_u64_u32 v[16:17], s[0:1], v12, s26, v[28:29]
	v_accvgpr_read_b32 v10, a242
	v_accvgpr_read_b32 v11, a243
	v_pk_mul_f32 v[8:9], v[8:9], s[24:25] op_sel_hi:[1,0]
	v_ashrrev_i32_e32 v17, 31, v16
	v_add_u32_e32 v18, s26, v16
	global_store_short_d16_hi v[46:47], v13, off offset:64
	global_store_short_d16_hi v[40:41], v8, off offset:96
	v_pk_mul_f32 v[10:11], v[10:11], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[42:43], v9, off offset:96
	v_lshlrev_b64 v[8:9], 1, v[16:17]
	v_ashrrev_i32_e32 v19, 31, v18
	v_add_u32_e32 v16, s26, v18
	v_accvgpr_read_b32 v12, a236
	global_store_short_d16_hi v[44:45], v10, off offset:96
	global_store_short_d16_hi v[46:47], v11, off offset:96
	v_lshlrev_b64 v[10:11], 1, v[18:19]
	v_add_u32_e32 v18, s26, v16
	v_accvgpr_read_b32 v13, a237
	v_accvgpr_read_b32 v14, a238
	v_accvgpr_read_b32 v15, a239
	v_ashrrev_i32_e32 v17, 31, v16
	v_ashrrev_i32_e32 v19, 31, v18
	v_pk_mul_f32 v[20:21], v[12:13], s[24:25] op_sel_hi:[1,0]
	v_lshlrev_b64 v[12:13], 1, v[16:17]
	v_pk_mul_f32 v[22:23], v[14:15], s[24:25] op_sel_hi:[1,0]
	v_lshlrev_b64 v[14:15], 1, v[18:19]
	v_accvgpr_read_b32 v16, a232
	v_accvgpr_read_b32 v17, a233
	v_lshl_add_u64 v[48:49], v[38:39], 0, v[8:9]
	v_lshl_add_u64 v[50:51], v[38:39], 0, v[10:11]
	v_lshl_add_u64 v[52:53], v[38:39], 0, v[12:13]
	v_accvgpr_read_b32 v18, a234
	v_accvgpr_read_b32 v19, a235
	v_lshl_add_u64 v[54:55], v[38:39], 0, v[14:15]
	v_pk_mul_f32 v[16:17], v[16:17], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[48:49], v20, off
	global_store_short_d16_hi v[50:51], v21, off
	global_store_short_d16_hi v[52:53], v22, off
	global_store_short_d16_hi v[54:55], v23, off
	global_store_short_d16_hi v[48:49], v16, off offset:32
	v_pk_mul_f32 v[20:21], v[18:19], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[50:51], v17, off offset:32
	v_accvgpr_read_b32 v16, a228
	v_accvgpr_read_b32 v17, a229
	v_accvgpr_read_b32 v18, a230
	v_accvgpr_read_b32 v19, a231
	global_store_short_d16_hi v[52:53], v20, off offset:32
	v_pk_mul_f32 v[16:17], v[16:17], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[54:55], v21, off offset:32
	global_store_short_d16_hi v[48:49], v16, off offset:64
	v_pk_mul_f32 v[20:21], v[18:19], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[50:51], v17, off offset:64
	v_accvgpr_read_b32 v16, a220
	global_store_short_d16_hi v[52:53], v20, off offset:64
	v_or_b32_e32 v20, 32, v29
	v_accvgpr_read_b32 v17, a221
	v_mad_u64_u32 v[24:25], s[0:1], v20, s26, v[28:29]
	v_accvgpr_read_b32 v18, a222
	v_accvgpr_read_b32 v19, a223
	v_pk_mul_f32 v[16:17], v[16:17], s[24:25] op_sel_hi:[1,0]
	v_ashrrev_i32_e32 v25, 31, v24
	v_add_u32_e32 v26, s26, v24
	global_store_short_d16_hi v[54:55], v21, off offset:64
	global_store_short_d16_hi v[48:49], v16, off offset:96
	v_pk_mul_f32 v[18:19], v[18:19], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[50:51], v17, off offset:96
	v_lshlrev_b64 v[16:17], 1, v[24:25]
	v_ashrrev_i32_e32 v27, 31, v26
	v_add_u32_e32 v24, s26, v26
	v_accvgpr_read_b32 v20, a216
	global_store_short_d16_hi v[52:53], v18, off offset:96
	global_store_short_d16_hi v[54:55], v19, off offset:96
	v_lshlrev_b64 v[18:19], 1, v[26:27]
	v_add_u32_e32 v26, s26, v24
	v_accvgpr_read_b32 v21, a217
	v_accvgpr_read_b32 v22, a218
	v_accvgpr_read_b32 v23, a219
	v_ashrrev_i32_e32 v25, 31, v24
	v_ashrrev_i32_e32 v27, 31, v26
	v_pk_mul_f32 v[30:31], v[20:21], s[24:25] op_sel_hi:[1,0]
	v_lshlrev_b64 v[20:21], 1, v[24:25]
	v_pk_mul_f32 v[34:35], v[22:23], s[24:25] op_sel_hi:[1,0]
	v_lshlrev_b64 v[22:23], 1, v[26:27]
	v_accvgpr_read_b32 v24, a212
	v_accvgpr_read_b32 v25, a213
	v_lshl_add_u64 v[56:57], v[38:39], 0, v[16:17]
	v_lshl_add_u64 v[58:59], v[38:39], 0, v[18:19]
	v_lshl_add_u64 v[60:61], v[38:39], 0, v[20:21]
	v_accvgpr_read_b32 v26, a214
	v_accvgpr_read_b32 v27, a215
	v_lshl_add_u64 v[62:63], v[38:39], 0, v[22:23]
	v_pk_mul_f32 v[24:25], v[24:25], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[56:57], v30, off
	global_store_short_d16_hi v[58:59], v31, off
	global_store_short_d16_hi v[60:61], v34, off
	global_store_short_d16_hi v[62:63], v35, off
	global_store_short_d16_hi v[56:57], v24, off offset:32
	v_pk_mul_f32 v[30:31], v[26:27], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[58:59], v25, off offset:32
	v_accvgpr_read_b32 v24, a204
	v_accvgpr_read_b32 v25, a205
	v_accvgpr_read_b32 v26, a206
	v_accvgpr_read_b32 v27, a207
	v_pk_mul_f32 v[24:25], v[24:25], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[60:61], v30, off offset:32
	global_store_short_d16_hi v[62:63], v31, off offset:32
	global_store_short_d16_hi v[56:57], v24, off offset:64
	v_pk_mul_f32 v[30:31], v[26:27], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[58:59], v25, off offset:64
	v_accvgpr_read_b32 v24, a196
	v_or_b32_e32 v29, 48, v29
	v_accvgpr_read_b32 v25, a197
	v_mad_u64_u32 v[34:35], s[0:1], v29, s26, v[28:29]
	v_accvgpr_read_b32 v26, a198
	v_accvgpr_read_b32 v27, a199
	v_pk_mul_f32 v[24:25], v[24:25], s[24:25] op_sel_hi:[1,0]
	v_ashrrev_i32_e32 v35, 31, v34
	v_add_u32_e32 v36, s26, v34
	global_store_short_d16_hi v[60:61], v30, off offset:64
	global_store_short_d16_hi v[62:63], v31, off offset:64
	global_store_short_d16_hi v[56:57], v24, off offset:96
	v_pk_mul_f32 v[26:27], v[26:27], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[58:59], v25, off offset:96
	v_lshlrev_b64 v[24:25], 1, v[34:35]
	v_ashrrev_i32_e32 v37, 31, v36
	v_add_u32_e32 v34, s26, v36
	v_accvgpr_read_b32 v28, a188
	global_store_short_d16_hi v[60:61], v26, off offset:96
	global_store_short_d16_hi v[62:63], v27, off offset:96
	v_lshlrev_b64 v[26:27], 1, v[36:37]
	v_add_u32_e32 v36, s26, v34
	v_accvgpr_read_b32 v29, a189
	v_accvgpr_read_b32 v30, a190
	v_accvgpr_read_b32 v31, a191
	v_ashrrev_i32_e32 v35, 31, v34
	v_ashrrev_i32_e32 v37, 31, v36
	v_pk_mul_f32 v[64:65], v[28:29], s[24:25] op_sel_hi:[1,0]
	v_lshlrev_b64 v[28:29], 1, v[34:35]
	v_pk_mul_f32 v[70:71], v[30:31], s[24:25] op_sel_hi:[1,0]
	v_lshlrev_b64 v[30:31], 1, v[36:37]
	v_accvgpr_read_b32 v34, a172
	v_lshl_add_u64 v[66:67], v[38:39], 0, v[24:25]
	v_lshl_add_u64 v[68:69], v[38:39], 0, v[26:27]
	v_accvgpr_read_b32 v35, a173
	global_store_short_d16_hi v[66:67], v64, off
	global_store_short_d16_hi v[68:69], v65, off
	v_lshl_add_u64 v[64:65], v[38:39], 0, v[28:29]
	v_accvgpr_read_b32 v36, a174
	v_accvgpr_read_b32 v37, a175
	v_lshl_add_u64 v[72:73], v[38:39], 0, v[30:31]
	v_pk_mul_f32 v[34:35], v[34:35], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[64:65], v70, off
	global_store_short_d16_hi v[72:73], v71, off
	global_store_short_d16_hi v[66:67], v34, off offset:32
	v_pk_mul_f32 v[70:71], v[36:37], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[68:69], v35, off offset:32
	v_accvgpr_read_b32 v34, a160
	v_accvgpr_read_b32 v35, a161
	v_accvgpr_read_b32 v36, a162
	v_accvgpr_read_b32 v37, a163
	v_pk_mul_f32 v[34:35], v[34:35], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[64:65], v70, off offset:32
	global_store_short_d16_hi v[72:73], v71, off offset:32
	global_store_short_d16_hi v[66:67], v34, off offset:64
	v_pk_mul_f32 v[70:71], v[36:37], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[68:69], v35, off offset:64
	v_accvgpr_read_b32 v34, a144
	v_accvgpr_read_b32 v35, a145
	v_accvgpr_read_b32 v36, a146
	v_accvgpr_read_b32 v37, a147
	v_pk_mul_f32 v[34:35], v[34:35], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[64:65], v70, off offset:64
	global_store_short_d16_hi v[72:73], v71, off offset:64
	global_store_short_d16_hi v[66:67], v34, off offset:96
	v_pk_mul_f32 v[70:71], v[36:37], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[68:69], v35, off offset:96
	v_accvgpr_read_b32 v34, a208
	v_accvgpr_read_b32 v35, a209
	v_accvgpr_read_b32 v36, a210
	v_accvgpr_read_b32 v37, a211
	v_pk_mul_f32 v[34:35], v[34:35], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[64:65], v70, off offset:96
	global_store_short_d16_hi v[72:73], v71, off offset:96
	global_store_short_d16_hi v[40:41], v34, off offset:256
	v_pk_mul_f32 v[40:41], v[36:37], s[24:25] op_sel_hi:[1,0]
	s_mov_b64 s[0:1], 0x100
	global_store_short_d16_hi v[42:43], v35, off offset:256
	v_accvgpr_read_b32 v34, a200
	v_accvgpr_read_b32 v35, a201
	v_lshl_add_u64 v[38:39], v[38:39], 0, s[0:1]
	v_accvgpr_read_b32 v36, a202
	v_accvgpr_read_b32 v37, a203
	global_store_short_d16_hi v[44:45], v40, off offset:256
	v_pk_mul_f32 v[34:35], v[34:35], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[46:47], v41, off offset:256
	v_lshl_add_u64 v[40:41], v[38:39], 0, v[0:1]
	v_lshl_add_u64 v[42:43], v[38:39], 0, v[2:3]
	global_store_short_d16_hi v[40:41], v34, off offset:32
	v_pk_mul_f32 v[44:45], v[36:37], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[42:43], v35, off offset:32
	v_accvgpr_read_b32 v34, a192
	v_accvgpr_read_b32 v35, a193
	v_lshl_add_u64 v[46:47], v[38:39], 0, v[4:5]
	v_accvgpr_read_b32 v36, a194
	v_accvgpr_read_b32 v37, a195
	v_lshl_add_u64 v[70:71], v[38:39], 0, v[6:7]
	v_pk_mul_f32 v[34:35], v[34:35], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[46:47], v44, off offset:32
	global_store_short_d16_hi v[70:71], v45, off offset:32
	global_store_short_d16_hi v[40:41], v34, off offset:64
	v_pk_mul_f32 v[44:45], v[36:37], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[42:43], v35, off offset:64
	v_accvgpr_read_b32 v34, a184
	v_accvgpr_read_b32 v35, a185
	v_accvgpr_read_b32 v36, a186
	v_accvgpr_read_b32 v37, a187
	v_pk_mul_f32 v[34:35], v[34:35], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[46:47], v44, off offset:64
	global_store_short_d16_hi v[70:71], v45, off offset:64
	global_store_short_d16_hi v[40:41], v34, off offset:96
	v_pk_mul_f32 v[40:41], v[36:37], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[42:43], v35, off offset:96
	v_accvgpr_read_b32 v34, a180
	v_accvgpr_read_b32 v35, a181
	v_accvgpr_read_b32 v36, a182
	v_accvgpr_read_b32 v37, a183
	v_pk_mul_f32 v[34:35], v[34:35], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[46:47], v40, off offset:96
	global_store_short_d16_hi v[70:71], v41, off offset:96
	global_store_short_d16_hi v[48:49], v34, off offset:256
	v_pk_mul_f32 v[40:41], v[36:37], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[50:51], v35, off offset:256
	v_accvgpr_read_b32 v34, a168
	v_accvgpr_read_b32 v35, a169
	v_accvgpr_read_b32 v36, a170
	v_accvgpr_read_b32 v37, a171
	global_store_short_d16_hi v[52:53], v40, off offset:256
	v_pk_mul_f32 v[34:35], v[34:35], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[54:55], v41, off offset:256
	v_lshl_add_u64 v[40:41], v[38:39], 0, v[8:9]
	v_lshl_add_u64 v[42:43], v[38:39], 0, v[10:11]
	global_store_short_d16_hi v[40:41], v34, off offset:32
	v_pk_mul_f32 v[44:45], v[36:37], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[42:43], v35, off offset:32
	v_accvgpr_read_b32 v34, a152
	v_accvgpr_read_b32 v35, a153
	v_lshl_add_u64 v[46:47], v[38:39], 0, v[12:13]
	v_accvgpr_read_b32 v36, a154
	v_accvgpr_read_b32 v37, a155
	v_lshl_add_u64 v[48:49], v[38:39], 0, v[14:15]
	v_pk_mul_f32 v[34:35], v[34:35], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[46:47], v44, off offset:32
	global_store_short_d16_hi v[48:49], v45, off offset:32
	global_store_short_d16_hi v[40:41], v34, off offset:64
	v_pk_mul_f32 v[44:45], v[36:37], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[42:43], v35, off offset:64
	v_accvgpr_read_b32 v34, a136
	v_accvgpr_read_b32 v35, a137
	v_accvgpr_read_b32 v36, a138
	v_accvgpr_read_b32 v37, a139
	v_pk_mul_f32 v[34:35], v[34:35], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[46:47], v44, off offset:64
	global_store_short_d16_hi v[48:49], v45, off offset:64
	global_store_short_d16_hi v[40:41], v34, off offset:96
	v_pk_mul_f32 v[40:41], v[36:37], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[42:43], v35, off offset:96
	v_accvgpr_read_b32 v34, a128
	v_accvgpr_read_b32 v35, a129
	v_accvgpr_read_b32 v36, a130
	v_accvgpr_read_b32 v37, a131
	v_pk_mul_f32 v[34:35], v[34:35], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[46:47], v40, off offset:96
	global_store_short_d16_hi v[48:49], v41, off offset:96
	global_store_short_d16_hi v[56:57], v34, off offset:256
	v_pk_mul_f32 v[40:41], v[36:37], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[58:59], v35, off offset:256
	v_accvgpr_read_b32 v34, a116
	v_accvgpr_read_b32 v35, a117
	v_accvgpr_read_b32 v36, a118
	v_accvgpr_read_b32 v37, a119
	global_store_short_d16_hi v[60:61], v40, off offset:256
	v_pk_mul_f32 v[34:35], v[34:35], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[62:63], v41, off offset:256
	v_lshl_add_u64 v[40:41], v[38:39], 0, v[16:17]
	v_lshl_add_u64 v[42:43], v[38:39], 0, v[18:19]
	global_store_short_d16_hi v[40:41], v34, off offset:32
	v_pk_mul_f32 v[44:45], v[36:37], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[42:43], v35, off offset:32
	v_accvgpr_read_b32 v34, a104
	v_accvgpr_read_b32 v35, a105
	v_lshl_add_u64 v[46:47], v[38:39], 0, v[20:21]
	v_accvgpr_read_b32 v36, a106
	v_accvgpr_read_b32 v37, a107
	v_lshl_add_u64 v[48:49], v[38:39], 0, v[22:23]
	v_pk_mul_f32 v[34:35], v[34:35], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[46:47], v44, off offset:32
	global_store_short_d16_hi v[48:49], v45, off offset:32
	global_store_short_d16_hi v[40:41], v34, off offset:64
	v_pk_mul_f32 v[44:45], v[36:37], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[42:43], v35, off offset:64
	v_accvgpr_read_b32 v34, a92
	v_accvgpr_read_b32 v35, a93
	v_accvgpr_read_b32 v36, a94
	v_accvgpr_read_b32 v37, a95
	v_pk_mul_f32 v[34:35], v[34:35], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[46:47], v44, off offset:64
	global_store_short_d16_hi v[48:49], v45, off offset:64
	global_store_short_d16_hi v[40:41], v34, off offset:96
	v_pk_mul_f32 v[40:41], v[36:37], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[42:43], v35, off offset:96
	v_accvgpr_read_b32 v34, a80
	v_accvgpr_read_b32 v35, a81
	v_accvgpr_read_b32 v36, a82
	v_accvgpr_read_b32 v37, a83
	v_pk_mul_f32 v[34:35], v[34:35], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[46:47], v40, off offset:96
	global_store_short_d16_hi v[48:49], v41, off offset:96
	global_store_short_d16_hi v[66:67], v34, off offset:256
	v_pk_mul_f32 v[40:41], v[36:37], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[68:69], v35, off offset:256
	v_accvgpr_read_b32 v34, a68
	v_accvgpr_read_b32 v35, a69
	v_accvgpr_read_b32 v36, a70
	v_accvgpr_read_b32 v37, a71
	global_store_short_d16_hi v[64:65], v40, off offset:256
	v_pk_mul_f32 v[34:35], v[34:35], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[72:73], v41, off offset:256
	v_lshl_add_u64 v[40:41], v[38:39], 0, v[24:25]
	v_lshl_add_u64 v[42:43], v[38:39], 0, v[26:27]
	global_store_short_d16_hi v[40:41], v34, off offset:32
	v_pk_mul_f32 v[44:45], v[36:37], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[42:43], v35, off offset:32
	v_accvgpr_read_b32 v34, a56
	v_accvgpr_read_b32 v35, a57
	v_lshl_add_u64 v[46:47], v[38:39], 0, v[28:29]
	v_accvgpr_read_b32 v36, a58
	v_accvgpr_read_b32 v37, a59
	v_lshl_add_u64 v[38:39], v[38:39], 0, v[30:31]
	v_pk_mul_f32 v[34:35], v[34:35], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[46:47], v44, off offset:32
	global_store_short_d16_hi v[38:39], v45, off offset:32
	global_store_short_d16_hi v[40:41], v34, off offset:64
	v_pk_mul_f32 v[44:45], v[36:37], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[42:43], v35, off offset:64
	v_accvgpr_read_b32 v34, a40
	v_accvgpr_read_b32 v35, a41
	global_store_short_d16_hi v[46:47], v44, off offset:64
	v_or_b32_e32 v44, 0x80, v74
	v_accvgpr_read_b32 v36, a42
	v_accvgpr_read_b32 v37, a43
	v_pk_mul_f32 v[34:35], v[34:35], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[38:39], v45, off offset:64
	v_mad_i64_i32 v[44:45], s[2:3], s26, v44, 0
	global_store_short_d16_hi v[40:41], v34, off offset:96
	v_pk_mul_f32 v[40:41], v[36:37], s[24:25] op_sel_hi:[1,0]
	v_lshl_add_u64 v[44:45], v[44:45], 1, s[28:29]
	global_store_short_d16_hi v[42:43], v35, off offset:96
	v_accvgpr_read_b32 v34, a176
	v_accvgpr_read_b32 v35, a177
	v_lshl_add_u64 v[42:43], v[44:45], 0, v[32:33]
	global_store_short_d16_hi v[46:47], v40, off offset:96
	v_pk_mul_f32 v[32:33], v[34:35], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[38:39], v41, off offset:96
	v_lshl_add_u64 v[38:39], v[42:43], 0, v[0:1]
	v_lshl_add_u64 v[40:41], v[42:43], 0, v[2:3]
	global_store_short_d16_hi v[38:39], v32, off
	global_store_short_d16_hi v[40:41], v33, off
	v_accvgpr_read_b32 v32, a164
	v_accvgpr_read_b32 v36, a178
	v_accvgpr_read_b32 v37, a179
	v_accvgpr_read_b32 v33, a165
	v_pk_mul_f32 v[36:37], v[36:37], s[24:25] op_sel_hi:[1,0]
	v_lshl_add_u64 v[44:45], v[42:43], 0, v[4:5]
	v_accvgpr_read_b32 v34, a166
	v_accvgpr_read_b32 v35, a167
	v_lshl_add_u64 v[46:47], v[42:43], 0, v[6:7]
	v_pk_mul_f32 v[32:33], v[32:33], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[44:45], v36, off
	global_store_short_d16_hi v[46:47], v37, off
	global_store_short_d16_hi v[38:39], v32, off offset:32
	v_pk_mul_f32 v[36:37], v[34:35], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[40:41], v33, off offset:32
	v_accvgpr_read_b32 v32, a148
	v_accvgpr_read_b32 v33, a149
	v_accvgpr_read_b32 v34, a150
	v_accvgpr_read_b32 v35, a151
	v_pk_mul_f32 v[32:33], v[32:33], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[44:45], v36, off offset:32
	global_store_short_d16_hi v[46:47], v37, off offset:32
	global_store_short_d16_hi v[38:39], v32, off offset:64
	v_pk_mul_f32 v[36:37], v[34:35], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[40:41], v33, off offset:64
	v_accvgpr_read_b32 v32, a132
	v_accvgpr_read_b32 v33, a133
	v_accvgpr_read_b32 v34, a134
	v_accvgpr_read_b32 v35, a135
	v_pk_mul_f32 v[32:33], v[32:33], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[44:45], v36, off offset:64
	global_store_short_d16_hi v[46:47], v37, off offset:64
	global_store_short_d16_hi v[38:39], v32, off offset:96
	v_pk_mul_f32 v[36:37], v[34:35], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[40:41], v33, off offset:96
	v_accvgpr_read_b32 v32, a120
	v_accvgpr_read_b32 v33, a121
	v_accvgpr_read_b32 v34, a122
	v_accvgpr_read_b32 v35, a123
	global_store_short_d16_hi v[44:45], v36, off offset:96
	v_pk_mul_f32 v[32:33], v[32:33], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[46:47], v37, off offset:96
	v_lshl_add_u64 v[36:37], v[42:43], 0, v[8:9]
	v_lshl_add_u64 v[48:49], v[42:43], 0, v[10:11]
	global_store_short_d16_hi v[36:37], v32, off
	v_pk_mul_f32 v[50:51], v[34:35], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[48:49], v33, off
	v_accvgpr_read_b32 v32, a108
	v_accvgpr_read_b32 v33, a109
	v_lshl_add_u64 v[52:53], v[42:43], 0, v[12:13]
	v_accvgpr_read_b32 v34, a110
	v_accvgpr_read_b32 v35, a111
	v_lshl_add_u64 v[54:55], v[42:43], 0, v[14:15]
	v_pk_mul_f32 v[32:33], v[32:33], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[52:53], v50, off
	global_store_short_d16_hi v[54:55], v51, off
	global_store_short_d16_hi v[36:37], v32, off offset:32
	v_pk_mul_f32 v[50:51], v[34:35], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[48:49], v33, off offset:32
	v_accvgpr_read_b32 v32, a96
	v_accvgpr_read_b32 v33, a97
	v_accvgpr_read_b32 v34, a98
	v_accvgpr_read_b32 v35, a99
	v_pk_mul_f32 v[32:33], v[32:33], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[52:53], v50, off offset:32
	global_store_short_d16_hi v[54:55], v51, off offset:32
	global_store_short_d16_hi v[36:37], v32, off offset:64
	v_pk_mul_f32 v[50:51], v[34:35], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[48:49], v33, off offset:64
	v_accvgpr_read_b32 v32, a88
	v_accvgpr_read_b32 v33, a89
	v_accvgpr_read_b32 v34, a90
	v_accvgpr_read_b32 v35, a91
	v_pk_mul_f32 v[32:33], v[32:33], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[52:53], v50, off offset:64
	global_store_short_d16_hi v[54:55], v51, off offset:64
	global_store_short_d16_hi v[36:37], v32, off offset:96
	v_pk_mul_f32 v[50:51], v[34:35], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[48:49], v33, off offset:96
	v_accvgpr_read_b32 v32, a76
	v_accvgpr_read_b32 v33, a77
	v_accvgpr_read_b32 v34, a78
	v_accvgpr_read_b32 v35, a79
	global_store_short_d16_hi v[52:53], v50, off offset:96
	v_pk_mul_f32 v[32:33], v[32:33], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[54:55], v51, off offset:96
	v_lshl_add_u64 v[50:51], v[42:43], 0, v[16:17]
	v_lshl_add_u64 v[56:57], v[42:43], 0, v[18:19]
	global_store_short_d16_hi v[50:51], v32, off
	v_pk_mul_f32 v[58:59], v[34:35], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[56:57], v33, off
	v_accvgpr_read_b32 v32, a64
	v_accvgpr_read_b32 v33, a65
	v_lshl_add_u64 v[60:61], v[42:43], 0, v[20:21]
	v_accvgpr_read_b32 v34, a66
	v_accvgpr_read_b32 v35, a67
	v_lshl_add_u64 v[62:63], v[42:43], 0, v[22:23]
	v_pk_mul_f32 v[32:33], v[32:33], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[60:61], v58, off
	global_store_short_d16_hi v[62:63], v59, off
	global_store_short_d16_hi v[50:51], v32, off offset:32
	v_pk_mul_f32 v[58:59], v[34:35], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[56:57], v33, off offset:32
	v_accvgpr_read_b32 v32, a52
	v_accvgpr_read_b32 v33, a53
	v_accvgpr_read_b32 v34, a54
	v_accvgpr_read_b32 v35, a55
	v_pk_mul_f32 v[32:33], v[32:33], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[60:61], v58, off offset:32
	global_store_short_d16_hi v[62:63], v59, off offset:32
	global_store_short_d16_hi v[50:51], v32, off offset:64
	v_pk_mul_f32 v[58:59], v[34:35], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[56:57], v33, off offset:64
	v_accvgpr_read_b32 v32, a36
	v_accvgpr_read_b32 v33, a37
	v_accvgpr_read_b32 v34, a38
	v_accvgpr_read_b32 v35, a39
	v_pk_mul_f32 v[32:33], v[32:33], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[60:61], v58, off offset:64
	global_store_short_d16_hi v[62:63], v59, off offset:64
	global_store_short_d16_hi v[50:51], v32, off offset:96
	v_pk_mul_f32 v[58:59], v[34:35], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[56:57], v33, off offset:96
	v_accvgpr_read_b32 v35, a31
	v_accvgpr_read_b32 v33, a29
	v_accvgpr_read_b32 v32, a28
	v_accvgpr_read_b32 v34, a30
	global_store_short_d16_hi v[60:61], v58, off offset:96
	v_pk_mul_f32 v[32:33], v[32:33], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[62:63], v59, off offset:96
	v_lshl_add_u64 v[58:59], v[42:43], 0, v[24:25]
	v_lshl_add_u64 v[64:65], v[42:43], 0, v[26:27]
	global_store_short_d16_hi v[58:59], v32, off
	v_pk_mul_f32 v[66:67], v[34:35], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[64:65], v33, off
	v_accvgpr_read_b32 v35, a23
	v_accvgpr_read_b32 v33, a21
	v_accvgpr_read_b32 v32, a20
	v_lshl_add_u64 v[68:69], v[42:43], 0, v[28:29]
	v_accvgpr_read_b32 v34, a22
	v_lshl_add_u64 v[70:71], v[42:43], 0, v[30:31]
	v_pk_mul_f32 v[32:33], v[32:33], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[68:69], v66, off
	global_store_short_d16_hi v[70:71], v67, off
	global_store_short_d16_hi v[58:59], v32, off offset:32
	v_pk_mul_f32 v[66:67], v[34:35], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[64:65], v33, off offset:32
	v_accvgpr_read_b32 v35, a19
	v_accvgpr_read_b32 v33, a17
	v_accvgpr_read_b32 v32, a16
	v_accvgpr_read_b32 v34, a18
	v_pk_mul_f32 v[32:33], v[32:33], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[68:69], v66, off offset:32
	global_store_short_d16_hi v[70:71], v67, off offset:32
	global_store_short_d16_hi v[58:59], v32, off offset:64
	v_pk_mul_f32 v[66:67], v[34:35], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[64:65], v33, off offset:64
	v_accvgpr_read_b32 v32, a48
	v_accvgpr_read_b32 v33, a49
	v_accvgpr_read_b32 v34, a50
	v_accvgpr_read_b32 v35, a51
	v_pk_mul_f32 v[32:33], v[32:33], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[68:69], v66, off offset:64
	global_store_short_d16_hi v[70:71], v67, off offset:64
	global_store_short_d16_hi v[58:59], v32, off offset:96
	v_pk_mul_f32 v[66:67], v[34:35], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[64:65], v33, off offset:96
	v_accvgpr_read_b32 v32, a156
	v_accvgpr_read_b32 v33, a157
	v_accvgpr_read_b32 v34, a158
	v_accvgpr_read_b32 v35, a159
	v_pk_mul_f32 v[32:33], v[32:33], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[68:69], v66, off offset:96
	global_store_short_d16_hi v[70:71], v67, off offset:96
	global_store_short_d16_hi v[38:39], v32, off offset:256
	v_pk_mul_f32 v[38:39], v[34:35], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[40:41], v33, off offset:256
	v_lshl_add_u64 v[40:41], v[42:43], 0, s[0:1]
	v_accvgpr_read_b32 v32, a140
	global_store_short_d16_hi v[44:45], v38, off offset:256
	global_store_short_d16_hi v[46:47], v39, off offset:256
	v_lshl_add_u64 v[38:39], v[40:41], 0, v[0:1]
	v_lshl_add_u64 v[42:43], v[40:41], 0, v[2:3]
	v_accvgpr_read_b32 v0, a124
	v_accvgpr_read_b32 v33, a141
	v_accvgpr_read_b32 v34, a142
	v_accvgpr_read_b32 v35, a143
	v_accvgpr_read_b32 v1, a125
	v_pk_mul_f32 v[32:33], v[32:33], s[24:25] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], s[24:25] op_sel_hi:[1,0]
	v_lshl_add_u64 v[4:5], v[40:41], 0, v[4:5]
	v_accvgpr_read_b32 v2, a126
	v_accvgpr_read_b32 v3, a127
	v_lshl_add_u64 v[6:7], v[40:41], 0, v[6:7]
	v_pk_mul_f32 v[0:1], v[0:1], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[38:39], v32, off offset:32
	global_store_short_d16_hi v[42:43], v33, off offset:32
	global_store_short_d16_hi v[4:5], v34, off offset:32
	global_store_short_d16_hi v[6:7], v35, off offset:32
	global_store_short_d16_hi v[38:39], v0, off offset:64
	v_pk_mul_f32 v[32:33], v[2:3], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[42:43], v1, off offset:64
	v_accvgpr_read_b32 v0, a112
	v_accvgpr_read_b32 v1, a113
	v_accvgpr_read_b32 v2, a114
	v_accvgpr_read_b32 v3, a115
	v_pk_mul_f32 v[0:1], v[0:1], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[4:5], v32, off offset:64
	global_store_short_d16_hi v[6:7], v33, off offset:64
	global_store_short_d16_hi v[38:39], v0, off offset:96
	v_pk_mul_f32 v[32:33], v[2:3], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[42:43], v1, off offset:96
	v_accvgpr_read_b32 v0, a100
	v_accvgpr_read_b32 v1, a101
	v_accvgpr_read_b32 v2, a102
	v_accvgpr_read_b32 v3, a103
	v_pk_mul_f32 v[0:1], v[0:1], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[4:5], v32, off offset:96
	global_store_short_d16_hi v[6:7], v33, off offset:96
	global_store_short_d16_hi v[36:37], v0, off offset:256
	v_pk_mul_f32 v[4:5], v[2:3], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[48:49], v1, off offset:256
	v_accvgpr_read_b32 v0, a84
	v_accvgpr_read_b32 v1, a85
	v_accvgpr_read_b32 v2, a86
	v_accvgpr_read_b32 v3, a87
	global_store_short_d16_hi v[52:53], v4, off offset:256
	v_pk_mul_f32 v[0:1], v[0:1], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[54:55], v5, off offset:256
	v_lshl_add_u64 v[4:5], v[40:41], 0, v[8:9]
	v_lshl_add_u64 v[6:7], v[40:41], 0, v[10:11]
	global_store_short_d16_hi v[4:5], v0, off offset:32
	v_pk_mul_f32 v[8:9], v[2:3], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[6:7], v1, off offset:32
	v_accvgpr_read_b32 v0, a72
	v_accvgpr_read_b32 v1, a73
	v_lshl_add_u64 v[10:11], v[40:41], 0, v[12:13]
	v_accvgpr_read_b32 v2, a74
	v_accvgpr_read_b32 v3, a75
	v_lshl_add_u64 v[12:13], v[40:41], 0, v[14:15]
	v_pk_mul_f32 v[0:1], v[0:1], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[10:11], v8, off offset:32
	global_store_short_d16_hi v[12:13], v9, off offset:32
	global_store_short_d16_hi v[4:5], v0, off offset:64
	v_pk_mul_f32 v[8:9], v[2:3], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[6:7], v1, off offset:64
	v_accvgpr_read_b32 v0, a60
	v_accvgpr_read_b32 v1, a61
	v_accvgpr_read_b32 v2, a62
	v_accvgpr_read_b32 v3, a63
	v_pk_mul_f32 v[0:1], v[0:1], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[10:11], v8, off offset:64
	global_store_short_d16_hi v[12:13], v9, off offset:64
	global_store_short_d16_hi v[4:5], v0, off offset:96
	v_pk_mul_f32 v[4:5], v[2:3], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[6:7], v1, off offset:96
	v_accvgpr_read_b32 v0, a44
	v_accvgpr_read_b32 v1, a45
	v_accvgpr_read_b32 v2, a46
	v_accvgpr_read_b32 v3, a47
	v_pk_mul_f32 v[0:1], v[0:1], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[10:11], v4, off offset:96
	global_store_short_d16_hi v[12:13], v5, off offset:96
	global_store_short_d16_hi v[50:51], v0, off offset:256
	v_pk_mul_f32 v[4:5], v[2:3], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[56:57], v1, off offset:256
	v_accvgpr_read_b32 v0, a32
	v_accvgpr_read_b32 v1, a33
	v_accvgpr_read_b32 v2, a34
	v_accvgpr_read_b32 v3, a35
	global_store_short_d16_hi v[60:61], v4, off offset:256
	v_pk_mul_f32 v[0:1], v[0:1], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[62:63], v5, off offset:256
	v_lshl_add_u64 v[4:5], v[40:41], 0, v[16:17]
	v_lshl_add_u64 v[6:7], v[40:41], 0, v[18:19]
	global_store_short_d16_hi v[4:5], v0, off offset:32
	v_pk_mul_f32 v[8:9], v[2:3], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[6:7], v1, off offset:32
	v_accvgpr_read_b32 v0, a24
	v_accvgpr_read_b32 v1, a25
	v_lshl_add_u64 v[10:11], v[40:41], 0, v[20:21]
	v_accvgpr_read_b32 v2, a26
	v_accvgpr_read_b32 v3, a27
	v_lshl_add_u64 v[12:13], v[40:41], 0, v[22:23]
	v_pk_mul_f32 v[0:1], v[0:1], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[10:11], v8, off offset:32
	global_store_short_d16_hi v[12:13], v9, off offset:32
	global_store_short_d16_hi v[4:5], v0, off offset:64
	v_pk_mul_f32 v[8:9], v[2:3], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[6:7], v1, off offset:64
	v_accvgpr_read_b32 v0, a12
	v_accvgpr_read_b32 v1, a13
	v_accvgpr_read_b32 v2, a14
	v_accvgpr_read_b32 v3, a15
	v_pk_mul_f32 v[0:1], v[0:1], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[10:11], v8, off offset:64
	global_store_short_d16_hi v[12:13], v9, off offset:64
	global_store_short_d16_hi v[4:5], v0, off offset:96
	v_pk_mul_f32 v[4:5], v[2:3], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[6:7], v1, off offset:96
	v_accvgpr_read_b32 v0, a8
	v_accvgpr_read_b32 v1, a9
	v_accvgpr_read_b32 v2, a10
	v_accvgpr_read_b32 v3, a11
	v_pk_mul_f32 v[0:1], v[0:1], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[10:11], v4, off offset:96
	global_store_short_d16_hi v[12:13], v5, off offset:96
	global_store_short_d16_hi v[58:59], v0, off offset:256
	v_pk_mul_f32 v[4:5], v[2:3], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[64:65], v1, off offset:256
	v_accvgpr_read_b32 v0, a4
	v_accvgpr_read_b32 v1, a5
	v_accvgpr_read_b32 v2, a6
	v_accvgpr_read_b32 v3, a7
	global_store_short_d16_hi v[68:69], v4, off offset:256
	v_pk_mul_f32 v[0:1], v[0:1], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[70:71], v5, off offset:256
	v_lshl_add_u64 v[10:11], v[40:41], 0, v[24:25]
	v_lshl_add_u64 v[4:5], v[40:41], 0, v[26:27]
	global_store_short_d16_hi v[10:11], v0, off offset:32
	v_pk_mul_f32 v[12:13], v[2:3], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[4:5], v1, off offset:32
	v_accvgpr_read_b32 v0, a0
	v_accvgpr_read_b32 v1, a1
	v_lshl_add_u64 v[6:7], v[40:41], 0, v[28:29]
	v_accvgpr_read_b32 v2, a2
	v_accvgpr_read_b32 v3, a3
	v_lshl_add_u64 v[8:9], v[40:41], 0, v[30:31]
	v_pk_mul_f32 v[0:1], v[0:1], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[6:7], v12, off offset:32
	global_store_short_d16_hi v[8:9], v13, off offset:32
	global_store_short_d16_hi v[10:11], v0, off offset:64
	v_pk_mul_f32 v[12:13], v[2:3], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[4:5], v1, off offset:64
	v_accvgpr_read_b32 v0, a224
	v_accvgpr_read_b32 v1, a225
	v_accvgpr_read_b32 v2, a226
	v_accvgpr_read_b32 v3, a227
	v_pk_mul_f32 v[0:1], v[0:1], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[6:7], v12, off offset:64
	global_store_short_d16_hi v[8:9], v13, off offset:64
	global_store_short_d16_hi v[10:11], v0, off offset:96
	v_pk_mul_f32 v[2:3], v[2:3], s[24:25] op_sel_hi:[1,0]
	global_store_short_d16_hi v[4:5], v1, off offset:96
	global_store_short_d16_hi v[6:7], v2, off offset:96
	global_store_short_d16_hi v[8:9], v3, off offset:96
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
		.amdhsa_next_free_vgpr 452
		.amdhsa_next_free_sgpr 96
		.amdhsa_accum_offset 196
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
	.set _Z22mxfp4_gluon_cpp_kernel13gluon_globals.num_vgpr, 196
	.set _Z22mxfp4_gluon_cpp_kernel13gluon_globals.num_agpr, 256
	.set _Z22mxfp4_gluon_cpp_kernel13gluon_globals.numbered_sgpr, 82
	.set _Z22mxfp4_gluon_cpp_kernel13gluon_globals.private_seg_size, 0
	.set _Z22mxfp4_gluon_cpp_kernel13gluon_globals.uses_vcc, 0
	.set _Z22mxfp4_gluon_cpp_kernel13gluon_globals.uses_flat_scratch, 0
	.set _Z22mxfp4_gluon_cpp_kernel13gluon_globals.has_dyn_sized_stack, 0
	.set _Z22mxfp4_gluon_cpp_kernel13gluon_globals.has_recursion, 0
	.set _Z22mxfp4_gluon_cpp_kernel13gluon_globals.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 34460
; TotalNumSgprs: 88
; NumVgprs: 196
; NumAgprs: 256
; TotalNumVgprs: 452
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 131072 bytes/workgroup (compile time only)
; SGPRBlocks: 12
; VGPRBlocks: 56
; NumSGPRsForWavesPerEU: 102
; NumVGPRsForWavesPerEU: 452
; AccumOffset: 196
; Occupancy: 1
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 48
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.type	__hip_cuid_ab5295047a34030d,@object ; @__hip_cuid_ab5295047a34030d
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_ab5295047a34030d
__hip_cuid_ab5295047a34030d:
	.byte	0                               ; 0x0
	.size	__hip_cuid_ab5295047a34030d, 1

	.ident	"clang version 20.0.0git (https://github.com/ROCm/llvm-project.git 27682a16360e33e37c4f3cc6adf9a620733f8fe1)"
	.ident	"AMD clang version 20.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.1.0 25425 1b0eada6b0ee93e2e694c8c146d23fca90bc11c5)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_ab5295047a34030d
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
    .sgpr_count:     88
    .sgpr_spill_count: 0
    .symbol:         _Z22mxfp4_gluon_cpp_kernel13gluon_globals.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     452
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
