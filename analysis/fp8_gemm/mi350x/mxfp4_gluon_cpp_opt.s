	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.text
	.protected	_Z22mxfp4_gluon_cpp_kernel13gluon_globals ; -- Begin function _Z22mxfp4_gluon_cpp_kernel13gluon_globals
	.globl	_Z22mxfp4_gluon_cpp_kernel13gluon_globals
	.p2align	8
	.type	_Z22mxfp4_gluon_cpp_kernel13gluon_globals,@function
_Z22mxfp4_gluon_cpp_kernel13gluon_globals: ; @_Z22mxfp4_gluon_cpp_kernel13gluon_globals
; %bb.0:
	s_load_dword s33, s[0:1], 0xe0
	s_load_dword s34, s[0:1], 0xf0
	s_abs_i32 s5, s2
	v_bfe_u32 v174, v0, 6, 1
	v_lshrrev_b32_e32 v6, 6, v0
	s_waitcnt lgkmcnt(0)
	s_ashr_i32 s3, s33, 31
	s_lshr_b32 s3, s3, 24
	s_add_i32 s3, s33, s3
	s_ashr_i32 s3, s3, 8
	s_xor_b32 s6, s2, s3
	s_ashr_i32 s6, s6, 31
	s_load_dwordx2 s[16:17], s[0:1], 0x60
	s_load_dword s18, s[0:1], 0x80
	s_load_dwordx2 s[36:37], s[0:1], 0x90
	s_load_dword s28, s[0:1], 0xb0
	s_load_dwordx2 s[40:41], s[0:1], 0x0
	s_load_dword s85, s[0:1], 0x20
	s_movk_i32 s83, 0x70
	s_abs_i32 s4, s3
	s_sub_i32 s7, 0, s4
	s_load_dword s84, s[0:1], 0x50
	s_load_dwordx2 s[56:57], s[0:1], 0x30
	v_cvt_f32_u32_e32 v1, s4
	s_load_dwordx2 s[44:45], s[0:1], 0xc0
	v_and_b32_e32 v67, 48, v0
	v_lshlrev_b32_e32 v71, 3, v0
	v_rcp_iflag_f32_e32 v1, v1
	v_lshlrev_b32_e32 v69, 13, v174
	v_or_b32_e32 v70, 0x10000, v69
	s_mov_b32 s47, 0
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	v_accvgpr_write_b32 a195, 0
	v_accvgpr_write_b32 a194, 0
	v_accvgpr_write_b32 a193, 0
	v_readfirstlane_b32 s8, v1
	s_mul_i32 s7, s7, s8
	s_mul_hi_u32 s7, s8, s7
	s_add_i32 s8, s8, s7
	s_mul_hi_u32 s7, s5, s8
	s_mul_i32 s8, s7, s4
	s_sub_i32 s5, s5, s8
	s_add_i32 s8, s7, 1
	s_sub_i32 s9, s5, s4
	s_cmp_ge_u32 s5, s4
	s_cselect_b32 s7, s8, s7
	s_cselect_b32 s5, s9, s5
	s_add_i32 s8, s7, 1
	s_cmp_ge_u32 s5, s4
	s_cselect_b32 s4, s8, s7
	s_xor_b32 s4, s4, s6
	s_sub_i32 s35, s4, s6
	v_lshrrev_b32_e32 v1, 7, v0
	s_mov_b32 s7, 0x110000
	s_lshl_b32 s55, s35, 8
	s_mul_i32 s3, s35, s3
	s_sub_i32 s46, s2, s3
	s_lshl_b32 s64, s46, 8
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s71, s55, s85
	s_mul_i32 s77, s64, s84
	s_or_b32 s74, s71, 0x80
	s_mov_b32 s6, -1
	s_mov_b32 s67, s77
	s_mov_b32 s59, s71
	;;#ASMSTART
	;;#ASMEND
	v_lshl_or_b32 v2, v1, 6, s55
	v_ashrrev_i32_e32 v5, 5, v2
	v_or_b32_e32 v4, 0x80, v2
	v_ashrrev_i32_e32 v4, 5, v4
	v_mul_lo_u32 v2, v5, s18
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], s[16:17], 0, v[2:3]
	s_or_b32 s80, s77, 0x80
	v_readfirstlane_b32 s4, v2
	v_mul_lo_u32 v2, v4, s18
	v_readfirstlane_b32 s5, v3
	s_mov_b64 s[10:11], s[6:7]
	s_mov_b64 s[8:9], s[4:5]
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], s[16:17], 0, v[2:3]
	s_mov_b64 s[26:27], s[6:7]
	v_readfirstlane_b32 s12, v2
	s_mov_b32 s8, s12
	v_readfirstlane_b32 s13, v3
	s_mov_b32 s9, s13
	s_mov_b64 s[14:15], s[6:7]
	s_mov_b64 s[12:13], s[4:5]
	v_or_b32_e32 v2, 1, v5
	v_mul_lo_u32 v2, v2, s18
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], s[16:17], 0, v[2:3]
	s_mov_b64 s[24:25], s[4:5]
	v_readfirstlane_b32 s19, v2
	v_readfirstlane_b32 s20, v3
	s_mov_b32 s13, s20
	v_or_b32_e32 v2, 1, v4
	v_mul_lo_u32 v2, v2, s18
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], s[16:17], 0, v[2:3]
	s_mov_b32 s12, s19
	v_readfirstlane_b32 s20, v2
	v_lshl_or_b32 v2, v174, 6, s64
	v_ashrrev_i32_e32 v7, 5, v2
	s_mov_b64 s[18:19], s[6:7]
	v_mul_lo_u32 v2, v7, s28
	v_readfirstlane_b32 s21, v3
	s_mov_b64 s[16:17], s[4:5]
	v_ashrrev_i32_e32 v3, 31, v2
	s_mov_b32 s16, s20
	s_mov_b32 s17, s21
	v_lshl_add_u64 v[4:5], s[36:37], 0, v[2:3]
	s_mov_b64 s[22:23], s[6:7]
	v_readfirstlane_b32 s2, v4
	s_mov_b64 s[20:21], s[4:5]
	s_mov_b32 s20, s2
	s_lshl_b32 s2, s28, 2
	v_add_u32_e32 v2, s2, v2
	v_ashrrev_i32_e32 v3, 31, v2
	v_readfirstlane_b32 s3, v5
	v_lshl_add_u64 v[2:3], s[36:37], 0, v[2:3]
	s_mov_b32 s21, s3
	v_readfirstlane_b32 s3, v2
	v_readfirstlane_b32 s29, v3
	s_mov_b32 s25, s29
	v_or_b32_e32 v2, 1, v7
	v_mul_lo_u32 v2, v2, s28
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[4:5], s[36:37], 0, v[2:3]
	v_add_u32_e32 v2, s2, v2
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], s[36:37], 0, v[2:3]
	s_mov_b64 s[30:31], s[6:7]
	s_mov_b32 s24, s3
	v_readfirstlane_b32 s3, v4
	v_readfirstlane_b32 s2, v2
	v_lshlrev_b32_e32 v2, 10, v6
	s_mov_b64 s[28:29], s[4:5]
	s_mov_b32 s28, s3
	v_readfirstlane_b32 s3, v3
	v_lshrrev_b32_e32 v3, 3, v0
	v_or_b32_e32 v4, 0x60, v3
	v_readfirstlane_b32 s48, v2
	v_readfirstlane_b32 s38, v5
	s_mov_b32 s29, s38
	s_mov_b64 s[38:39], s[6:7]
	s_mov_b64 s[36:37], s[4:5]
	s_mov_b32 s36, s2
	s_mov_b32 s55, s48
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s55
	s_add_i32 s53, s48, 0x18000
	s_add_i32 s54, s53, 0x4000
	s_mov_b32 s37, s3
	v_lshlrev_b32_e32 v2, 4, v0
	v_bitop3_b32 v2, v2, s83, v0 bitop3:0x48
	v_mad_u64_u32 v[146:147], s[0:1], v3, s85, v[2:3]
	s_lshl_b32 s0, s85, 5
	s_nop 0
	v_add_u32_e32 v147, s0, v146
	v_add_u32_e32 v154, s0, v147
	v_mad_u64_u32 v[150:151], s[0:1], v3, s84, v[2:3]
	s_add_i32 s49, s48, 0x4000
	v_mad_u64_u32 v[148:149], s[0:1], v4, s85, v[2:3]
	s_lshl_b32 s0, s84, 5
	s_nop 0
	v_add_u32_e32 v149, s0, v150
	v_add_u32_e32 v151, s0, v149
	v_mad_u64_u32 v[152:153], s[0:1], v4, s84, v[2:3]
	s_mov_b64 s[0:1], s[4:5]
	s_mov_b32 s0, s40
	s_add_i32 s50, s48, 0xc000
	s_mov_b64 s[2:3], s[6:7]
	s_mov_b32 s1, s41
	buffer_load_dwordx4 v146, s[0:3], s59 offen lds
	v_lshlrev_b32_e32 v2, 2, v0
	v_and_b32_e32 v153, 0xfc, v2
	v_lshlrev_b32_e32 v2, 7, v0
	v_and_b32_e32 v68, 0x780, v2
	v_lshlrev_b32_e32 v66, 13, v1
	s_mov_b64 s[42:43], s[6:7]
	s_mov_b64 s[40:41], s[4:5]
	s_mov_b32 s40, s56
	s_add_i32 s56, s48, 0x1000
	s_add_i32 s51, s48, 0x10000
	s_mov_b32 s64, s51
	s_mov_b32 s41, s57
	s_mov_b32 s57, s56
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s57
	s_add_i32 s57, s48, 0x2000
	s_mov_b32 s58, s57
	s_add_i32 s52, s51, 0x4000
	buffer_load_dwordx4 v147, s[0:3], s59 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s58
	s_add_i32 s58, s48, 0x3000
	s_mov_b32 s60, s58
	buffer_load_dwordx4 v154, s[0:3], s59 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s60
	s_add_i32 s60, s48, 0x8000
	s_mov_b32 s61, s60
	buffer_load_dwordx4 v148, s[0:3], s59 offen lds
	s_lshl_b32 s59, s85, 7
	s_add_i32 s59, s71, s59
	s_add_i32 s78, s59, 0x80
	v_or_b32_e32 v2, v68, v67
	v_or_b32_e32 v18, v2, v70
	v_or_b32_e32 v70, v68, v70
	s_mov_b32 s71, s49
	v_or_b32_e32 v3, v2, v66
	v_bitop3_b32 v4, v71, v3, s83 bitop3:0x6c
	v_or_b32_e32 v3, 64, v3
	v_bitop3_b32 v3, v71, v3, s83 bitop3:0x6c
	s_mov_b32 s65, s59
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s61
	s_add_i32 s61, s48, 0x9000
	s_mov_b32 s62, s61
	v_or_b32_e32 v66, v68, v66
	v_or_b32_e32 v68, v69, v68
	v_or_b32_e32 v69, 0x14000, v68
	buffer_load_dwordx4 v146, s[0:3], s65 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s62
	s_add_i32 s62, s48, 0xa000
	s_mov_b32 s63, s62
	v_lshrrev_b32_e32 v2, 4, v18
	v_bitop3_b32 v19, v2, v18, s83 bitop3:0x6c
	v_add_u32_e32 v18, 64, v18
	buffer_load_dwordx4 v147, s[0:3], s65 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s63
	s_add_i32 s63, s48, 0xb000
	s_mov_b32 s66, s63
	buffer_load_dwordx4 v154, s[0:3], s65 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s66
	s_mul_i32 s85, s85, s35
	buffer_load_dwordx4 v148, s[0:3], s65 offen lds
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s64
	s_add_i32 s64, s51, 0x1000
	s_mov_b32 s65, s64
	buffer_load_dwordx4 v150, s[40:43], s67 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s65
	s_add_i32 s65, s51, 0x2000
	s_mov_b32 s66, s65
	buffer_load_dwordx4 v149, s[40:43], s67 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s66
	s_add_i32 s66, s51, 0x3000
	s_mov_b32 s68, s66
	buffer_load_dwordx4 v151, s[40:43], s67 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s68
	s_mov_b32 s68, s53
	buffer_load_dwordx4 v152, s[40:43], s67 offen lds
	s_lshl_b32 s67, s84, 7
	s_add_i32 s67, s77, s67
	s_add_i32 s86, s67, 0x80
	s_mov_b32 s72, s67
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s68
	s_add_i32 s68, s53, 0x1000
	s_mov_b32 s69, s68
	s_mov_b32 s77, s52
	buffer_load_dwordx4 v150, s[40:43], s72 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s69
	s_add_i32 s69, s53, 0x2000
	s_mov_b32 s70, s69
	buffer_load_dwordx4 v149, s[40:43], s72 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s70
	s_add_i32 s70, s53, 0x3000
	s_mov_b32 s73, s70
	buffer_load_dwordx4 v151, s[40:43], s72 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s73
	s_mul_i32 s84, s84, s46
	buffer_load_dwordx4 v152, s[40:43], s72 offen lds
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s71
	s_add_i32 s71, s48, 0x5000
	s_mov_b32 s72, s71
	buffer_load_dwordx4 v146, s[0:3], s74 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s72
	s_add_i32 s72, s48, 0x6000
	s_mov_b32 s73, s72
	buffer_load_dwordx4 v147, s[0:3], s74 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s73
	s_add_i32 s73, s48, 0x7000
	s_mov_b32 s75, s73
	buffer_load_dwordx4 v154, s[0:3], s74 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s75
	s_movk_i32 s55, 0x1000
	buffer_load_dwordx4 v148, s[0:3], s74 offen lds
	s_mov_b32 s74, s50
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s74
	s_add_i32 s74, s48, 0xd000
	s_mov_b32 s75, s74
	buffer_load_dwordx4 v146, s[0:3], s78 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s75
	s_add_i32 s75, s48, 0xe000
	s_mov_b32 s76, s75
	buffer_load_dwordx4 v147, s[0:3], s78 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s76
	s_add_i32 s76, s48, 0xf000
	s_mov_b32 s79, s76
	buffer_load_dwordx4 v154, s[0:3], s78 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s79
	s_lshl_b32 s84, s84, 8
	buffer_load_dwordx4 v148, s[0:3], s78 offen lds
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s77
	s_add_i32 s77, s51, 0x5000
	s_mov_b32 s78, s77
	buffer_load_dwordx4 v150, s[40:43], s80 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s78
	s_add_i32 s78, s51, 0x6000
	s_mov_b32 s79, s78
	buffer_load_dwordx4 v149, s[40:43], s80 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s79
	s_add_i32 s79, s51, 0x7000
	s_mov_b32 s81, s79
	buffer_load_dwordx4 v151, s[40:43], s80 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s81
	v_accvgpr_write_b32 a192, 0
	buffer_load_dwordx4 v152, s[40:43], s80 offen lds
	s_mov_b32 s80, s54
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s80
	s_add_i32 s80, s53, 0x5000
	s_mov_b32 s81, s80
	buffer_load_dwordx4 v150, s[40:43], s86 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s81
	s_add_i32 s81, s53, 0x6000
	s_mov_b32 s82, s81
	buffer_load_dwordx4 v149, s[40:43], s86 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s82
	s_add_i32 s82, s53, 0x7000
	s_mov_b32 s87, s82
	buffer_load_dwordx4 v151, s[40:43], s86 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s87
	v_accvgpr_write_b32 a199, 0
	buffer_load_dwordx4 v152, s[40:43], s86 offen lds
	buffer_load_dword v178, v153, s[4:7], 0 offen
	buffer_load_dword v171, v153, s[8:11], 0 offen
	buffer_load_dword v179, v153, s[12:15], 0 offen
	buffer_load_dword v172, v153, s[16:19], 0 offen
	buffer_load_dword v176, v153, s[20:23], 0 offen
	buffer_load_dword v173, v153, s[24:27], 0 offen
	buffer_load_dword v177, v153, s[28:31], 0 offen
	buffer_load_dword v175, v153, s[36:39], 0 offen
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
	ds_read_b128 v[2:5], v19 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[6:9], v19 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[10:13], v19 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[14:17], v19 offset:0x1800

	;;#ASMEND
	v_lshrrev_b32_e32 v19, 4, v18
	v_bitop3_b32 v72, v19, v18, s83 bitop3:0x6c
	;;#ASMSTART
	ds_read_b128 v[18:21], v72 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[22:25], v72 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[26:29], v72 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[30:33], v72 offset:0x1800

	;;#ASMEND
	v_or_b32_e32 v72, v66, v67
	v_bitop3_b32 v155, v71, v72, s83 bitop3:0x6c
	v_or_b32_e32 v72, 64, v67
	v_or_b32_e32 v73, v66, v72
	v_bitop3_b32 v156, v71, v73, s83 bitop3:0x6c
	v_or_b32_e32 v73, 0x4000, v66
	v_or_b32_e32 v74, v73, v67
	v_or_b32_e32 v73, v73, v72
	v_bitop3_b32 v158, v71, v73, s83 bitop3:0x6c
	v_add_u32_e32 v73, v70, v67
	v_bitop3_b32 v157, v71, v74, s83 bitop3:0x6c
	v_lshrrev_b32_e32 v74, 4, v73
	v_add_u32_e32 v70, v70, v72
	v_bitop3_b32 v159, v74, v73, s83 bitop3:0x6c
	v_lshrrev_b32_e32 v73, 4, v70
	v_bitop3_b32 v160, v73, v70, s83 bitop3:0x6c
	v_or_b32_e32 v70, v69, v67
	v_or_b32_e32 v69, v69, v72
	v_bitop3_b32 v162, v71, v69, s83 bitop3:0x6c
	v_or_b32_e32 v69, 0x18000, v68
	v_bitop3_b32 v161, v71, v70, s83 bitop3:0x6c
	v_or_b32_e32 v70, v69, v67
	v_or_b32_e32 v69, v69, v72
	v_or_b32_e32 v68, 0x1c000, v68
	v_bitop3_b32 v164, v71, v69, s83 bitop3:0x6c
	v_or_b32_e32 v69, v68, v67
	v_or_b32_e32 v68, v68, v72
	v_bitop3_b32 v166, v71, v68, s83 bitop3:0x6c
	v_or_b32_e32 v68, 0x8000, v66
	v_or_b32_e32 v66, 0xc000, v66
	v_bitop3_b32 v165, v71, v69, s83 bitop3:0x6c
	v_or_b32_e32 v69, v68, v67
	v_or_b32_e32 v68, v68, v72
	v_or_b32_e32 v67, v66, v67
	v_bitop3_b32 v163, v71, v70, s83 bitop3:0x6c
	v_or_b32_e32 v66, v66, v72
	v_bitop3_b32 v167, v71, v69, s83 bitop3:0x6c
	v_bitop3_b32 v168, v71, v68, s83 bitop3:0x6c
	v_bitop3_b32 v169, v71, v67, s83 bitop3:0x6c
	v_bitop3_b32 v170, v71, v66, s83 bitop3:0x6c
	s_lshl_b32 s83, s85, 8
	s_mov_b32 s85, 15
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
.LBB0_1:                                ; =>This Inner Loop Header: Depth=1
	s_add_i32 s86, s55, 0xfffff100
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[34:37], v[2:5], a[28:31],  v178, v176 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[114:117], v163 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[34:37], v[6:9], a[32:35],  v178, v176 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[118:121], v163 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[34:37], v[10:13], a[36:39],  v178, v177 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[122:125], v163 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[34:37], v[14:17], a[40:43],  v178, v177 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[126:129], v163 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[50:53], v[18:21], a[28:31],  v178, v176 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[130:133], v164 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[50:53], v[22:25], a[32:35],  v178, v176 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[134:137], v164 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[50:53], v[26:29], a[36:39],  v178, v177 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[138:141], v164 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[50:53], v[30:33], a[40:43],  v178, v177 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[142:145], v164 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[38:41], v[2:5], a[44:47],  v178, v176 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[38:41], v[6:9], a[48:51],  v178, v176 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[38:41], v[10:13], a[52:55],  v178, v177 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[38:41], v[14:17], a[56:59],  v178, v177 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[54:57], v[18:21], a[44:47],  v178, v176 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[54:57], v[22:25], a[48:51],  v178, v176 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[54:57], v[26:29], a[52:55],  v178, v177 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[54:57], v[30:33], a[56:59],  v178, v177 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[42:45], v[2:5], a[60:63],  v179, v176 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[42:45], v[6:9], a[64:67],  v179, v176 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[42:45], v[10:13], a[68:71], v179, v177 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[42:45], v[14:17], a[72:75], v179, v177 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[58:61], v[18:21], a[60:63],  v179, v176 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[58:61], v[22:25], a[64:67],  v179, v176 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[58:61], v[26:29], a[68:71], v179, v177 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[58:61], v[30:33], a[72:75], v179, v177 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[46:49], v[2:5], a[76:79], v179, v176 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[46:49], v[6:9], a[80:83], v179, v176 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[46:49], v[10:13], a[84:87], v179, v177 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[46:49], v[14:17], a[88:91], v179, v177 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[62:65], v[18:21], a[76:79], v179, v176 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[62:65], v[22:25], a[80:83], v179, v176 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[62:65], v[26:29], a[84:87], v179, v177 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[62:65], v[30:33], a[88:91], v179, v177 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[34:37], v[114:117], a[0:3], v178, v173 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[66:69], v167 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[34:37], v[118:121], a[4:7], v178, v173 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[70:73], v167 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[34:37], v[122:125], a[8:11], v178, v175 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[74:77], v167 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[34:37], v[126:129], a[12:15], v178, v175 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[78:81], v167 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[50:53], v[130:133], a[0:3], v178, v173 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[98:101], v168 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[50:53], v[134:137], a[4:7], v178, v173 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[102:105], v168 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[50:53], v[138:141], a[8:11], v178, v175 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[106:109], v168 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[50:53], v[142:145], a[12:15], v178, v175 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[110:113], v168 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[38:41], v[114:117], a[16:19], v178, v173 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[38:41], v[118:121], a[20:23], v178, v173 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[38:41], v[122:125], a[24:27], v178, v175 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[38:41], v[126:129], a[92:95], v178, v175 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[54:57], v[130:133], a[16:19], v178, v173 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[54:57], v[134:137], a[20:23], v178, v173 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[54:57], v[138:141], a[24:27], v178, v175 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[54:57], v[142:145], a[92:95], v178, v175 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[42:45], v[114:117], a[96:99], v179, v173 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[42:45], v[118:121], a[100:103], v179, v173 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[42:45], v[122:125], a[104:107], v179, v175 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[42:45], v[126:129], a[108:111], v179, v175 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[58:61], v[130:133], a[96:99], v179, v173 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[58:61], v[134:137], a[100:103], v179, v173 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[58:61], v[138:141], a[104:107], v179, v175 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[58:61], v[142:145], a[108:111], v179, v175 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[46:49], v[114:117], a[112:115], v179, v173 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[46:49], v[118:121], a[116:119], v179, v173 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[46:49], v[122:125], a[120:123], v179, v175 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[46:49], v[126:129], a[124:127], v179, v175 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[62:65], v[130:133], a[112:115], v179, v173 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[62:65], v[134:137], a[116:119], v179, v173 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[62:65], v[138:141], a[120:123], v179, v175 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[62:65], v[142:145], a[124:127], v179, v175 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_barrier
	buffer_load_dword v216, v153, s[4:7], s86 offen
	buffer_load_dword v178, v153, s[8:11], s86 offen
	buffer_load_dword v217, v153, s[12:15], s86 offen
	buffer_load_dword v179, v153, s[16:19], s86 offen
	buffer_load_dword v182, v153, s[20:23], s86 offen
	buffer_load_dword v180, v153, s[24:27], s86 offen
	buffer_load_dword v183, v153, s[28:31], s86 offen
	buffer_load_dword v181, v153, s[36:39], s86 offen
	s_add_i32 s86, s83, s47
	s_add_i32 s90, s86, 0x100
	s_mov_b32 m0, s48
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[66:69], v[2:5], a[128:131],  v171, v176 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[184:187], v157 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[66:69], v[6:9], a[132:135],  v171, v176 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[188:191], v157 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[66:69], v[10:13], a[136:139],  v171, v177 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[192:195], v157 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[66:69], v[14:17], a[140:143],  v171, v177 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[196:199], v157 offset:6144

	s_add_i32 s87, s59, s47
	buffer_load_dwordx4 v146, s[0:3], s90 offen lds
	s_mov_b32 m0, s56
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[98:101], v[18:21], a[128:131],  v171, v176 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[200:203], v158 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[98:101], v[22:25], a[132:135],  v171, v176 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[204:207], v158 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[98:101], v[26:29], a[136:139],  v171, v177 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[208:211], v158 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[98:101], v[30:33], a[140:143],  v171, v177 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[212:215], v158 offset:6144

	s_add_i32 s91, s87, 0x100
	buffer_load_dwordx4 v147, s[0:3], s90 offen lds
	s_mov_b32 m0, s57
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[70:73], v[2:5], a[144:147],  v171, v176 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[70:73], v[6:9], a[148:151],  v171, v176 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[70:73], v[10:13], a[152:155],  v171, v177 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[70:73], v[14:17], a[156:159],  v171, v177 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_add_i32 s88, s84, s47
	buffer_load_dwordx4 v154, s[0:3], s90 offen lds
	s_mov_b32 m0, s58
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[102:105], v[18:21], a[144:147],  v171, v176 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[102:105], v[22:25], a[148:151],  v171, v176 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[102:105], v[26:29], a[152:155],  v171, v177 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[102:105], v[30:33], a[156:159],  v171, v177 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_add_i32 s92, s88, 0x100
	buffer_load_dwordx4 v148, s[0:3], s90 offen lds
	s_mov_b32 m0, s60
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[74:77], v[2:5], a[160:163],  v172, v176 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[74:77], v[6:9], a[164:167],  v172, v176 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[74:77], v[10:13], a[168:171], v172, v177 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[74:77], v[14:17], a[172:175], v172, v177 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_add_i32 s89, s67, s47
	buffer_load_dwordx4 v146, s[0:3], s91 offen lds
	s_mov_b32 m0, s61
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[106:109], v[18:21], a[160:163],  v172, v176 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[106:109], v[22:25], a[164:167],  v172, v176 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[106:109], v[26:29], a[168:171], v172, v177 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[106:109], v[30:33], a[172:175], v172, v177 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_add_i32 s93, s89, 0x100
	buffer_load_dwordx4 v147, s[0:3], s91 offen lds
	s_mov_b32 m0, s62
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[78:81], v[2:5], a[176:179], v172, v176 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[78:81], v[6:9], a[180:183], v172, v176 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[78:81], v[10:13], a[184:187], v172, v177 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[78:81], v[14:17], a[188:191], v172, v177 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_add_i32 s90, s55, 0xfffff200
	buffer_load_dwordx4 v154, s[0:3], s91 offen lds
	s_mov_b32 m0, s63
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[110:113], v[18:21], a[176:179], v172, v176 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[110:113], v[22:25], a[180:183], v172, v176 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[110:113], v[26:29], a[184:187], v172, v177 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[110:113], v[30:33], a[188:191], v172, v177 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v148, s[0:3], s91 offen lds
	s_mov_b32 m0, s51
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[66:69], v[114:117], a[192:195],  v171, v173 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[2:5], v161 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[66:69], v[118:121], a[196:199],  v171, v173 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[6:9], v161 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[66:69], v[122:125], a[200:203],  v171, v175 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[10:13], v161 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[66:69], v[126:129], a[204:207],  v171, v175 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[14:17], v161 offset:6144

	s_add_i32 s91, s87, 0x180
	buffer_load_dwordx4 v150, s[40:43], s92 offen lds
	s_mov_b32 m0, s64
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[98:101], v[130:133], a[192:195],  v171, v173 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[82:85], v162 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[98:101], v[134:137], a[196:199],  v171, v173 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[86:89], v162 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[98:101], v[138:141], a[200:203],  v171, v175 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[90:93], v162 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[98:101], v[142:145], a[204:207],  v171, v175 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[94:97], v162 offset:6144

	s_nop 0
	buffer_load_dwordx4 v149, s[40:43], s92 offen lds
	s_mov_b32 m0, s65
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[70:73], v[114:117], a[208:211],  v171, v173 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[70:73], v[118:121], a[212:215],  v171, v173 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[70:73], v[122:125], a[216:219],  v171, v175 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[70:73], v[126:129], a[220:223],  v171, v175 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v151, s[40:43], s92 offen lds
	s_mov_b32 m0, s66
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[102:105], v[130:133], a[208:211],  v171, v173 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[102:105], v[134:137], a[212:215],  v171, v173 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[102:105], v[138:141], a[216:219],  v171, v175 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[102:105], v[142:145], a[220:223],  v171, v175 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v152, s[40:43], s92 offen lds
	s_mov_b32 m0, s53
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[74:77], v[114:117], a[224:227],  v172, v173 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[74:77], v[118:121], a[228:231],  v172, v173 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[74:77], v[122:125], a[232:235], v172, v175 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[74:77], v[126:129], a[236:239], v172, v175 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_add_i32 s92, s88, 0x180
	buffer_load_dwordx4 v150, s[40:43], s93 offen lds
	s_mov_b32 m0, s68
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[106:109], v[130:133], a[224:227],  v172, v173 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[106:109], v[134:137], a[228:231],  v172, v173 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[106:109], v[138:141], a[232:235], v172, v175 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[106:109], v[142:145], a[236:239], v172, v175 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v149, s[40:43], s93 offen lds
	s_mov_b32 m0, s69
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[78:81], v[114:117], a[240:243], v172, v173 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[78:81], v[118:121], a[244:247], v172, v173 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[78:81], v[122:125], a[248:251], v172, v175 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[78:81], v[126:129], a[252:255], v172, v175 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v151, s[40:43], s93 offen lds
	s_mov_b32 m0, s70
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[110:113], v[130:133], a[240:243], v172, v173 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[110:113], v[134:137], a[244:247], v172, v173 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[110:113], v[138:141], a[248:251], v172, v175 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[110:113], v[142:145], a[252:255], v172, v175 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v152, s[40:43], s93 offen lds
	s_waitcnt lgkmcnt(0)
	s_mov_b32 m0, s49
	s_add_i32 s93, s89, 0x180
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[184:187], v[2:5], a[28:31],  v216, v182 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v165 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[184:187], v[6:9], a[32:35],  v216, v182 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v165 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[184:187], v[10:13], a[36:39],  v216, v183 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v165 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[184:187], v[14:17], a[40:43],  v216, v183 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[62:65], v165 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[200:203], v[82:85], a[28:31],  v216, v182 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[66:69], v166 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[200:203], v[86:89], a[32:35],  v216, v182 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[70:73], v166 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[200:203], v[90:93], a[36:39],  v216, v183 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[74:77], v166 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[200:203], v[94:97], a[40:43],  v216, v183 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[78:81], v166 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[188:191], v[2:5], a[44:47],  v216, v182 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[188:191], v[6:9], a[48:51],  v216, v182 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[188:191], v[10:13], a[52:55],  v216, v183 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[188:191], v[14:17], a[56:59],  v216, v183 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[204:207], v[82:85], a[44:47],  v216, v182 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[204:207], v[86:89], a[48:51],  v216, v182 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[204:207], v[90:93], a[52:55],  v216, v183 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[204:207], v[94:97], a[56:59],  v216, v183 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[192:195], v[2:5], a[60:63],  v217, v182 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[192:195], v[6:9], a[64:67],  v217, v182 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[192:195], v[10:13], a[68:71], v217, v183 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[192:195], v[14:17], a[72:75], v217, v183 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[208:211], v[82:85], a[60:63],  v217, v182 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[208:211], v[86:89], a[64:67],  v217, v182 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[208:211], v[90:93], a[68:71], v217, v183 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[208:211], v[94:97], a[72:75], v217, v183 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[196:199], v[2:5], a[76:79], v217, v182 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[196:199], v[6:9], a[80:83], v217, v182 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[196:199], v[10:13], a[84:87], v217, v183 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[196:199], v[14:17], a[88:91], v217, v183 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[212:215], v[82:85], a[76:79], v217, v182 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[212:215], v[86:89], a[80:83], v217, v182 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[212:215], v[90:93], a[84:87], v217, v183 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[212:215], v[94:97], a[88:91], v217, v183 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[184:187], v[50:53], a[0:3], v216, v180 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[18:21], v169 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[184:187], v[54:57], a[4:7], v216, v180 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[22:25], v169 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[184:187], v[58:61], a[8:11], v216, v181 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[26:29], v169 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[184:187], v[62:65], a[12:15], v216, v181 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v169 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[200:203], v[66:69], a[0:3], v216, v180 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v170 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[200:203], v[70:73], a[4:7], v216, v180 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v170 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[200:203], v[74:77], a[8:11], v216, v181 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v170 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[200:203], v[78:81], a[12:15], v216, v181 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v170 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[188:191], v[50:53], a[16:19], v216, v180 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[188:191], v[54:57], a[20:23], v216, v180 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[188:191], v[58:61], a[24:27], v216, v181 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[188:191], v[62:65], a[92:95], v216, v181 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[204:207], v[66:69], a[16:19], v216, v180 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[204:207], v[70:73], a[20:23], v216, v180 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[204:207], v[74:77], a[24:27], v216, v181 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[204:207], v[78:81], a[92:95], v216, v181 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[192:195], v[50:53], a[96:99], v217, v180 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[192:195], v[54:57], a[100:103], v217, v180 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[192:195], v[58:61], a[104:107], v217, v181 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[192:195], v[62:65], a[108:111], v217, v181 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[208:211], v[66:69], a[96:99], v217, v180 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[208:211], v[70:73], a[100:103], v217, v180 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[208:211], v[74:77], a[104:107], v217, v181 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[208:211], v[78:81], a[108:111], v217, v181 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[196:199], v[50:53], a[112:115], v217, v180 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[196:199], v[54:57], a[116:119], v217, v180 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[196:199], v[58:61], a[120:123], v217, v181 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[196:199], v[62:65], a[124:127], v217, v181 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[212:215], v[66:69], a[112:115], v217, v180 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[212:215], v[70:73], a[116:119], v217, v180 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[212:215], v[74:77], a[120:123], v217, v181 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[212:215], v[78:81], a[124:127], v217, v181 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_waitcnt lgkmcnt(0)
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_barrier
	buffer_load_dword v99, v153, s[4:7], s90 offen
	buffer_load_dword v98, v153, s[8:11], s90 offen
	buffer_load_dword v101, v153, s[12:15], s90 offen
	buffer_load_dword v100, v153, s[16:19], s90 offen
	buffer_load_dword v105, v153, s[20:23], s90 offen
	buffer_load_dword v102, v153, s[24:27], s90 offen
	buffer_load_dword v106, v153, s[28:31], s90 offen
	buffer_load_dword v104, v153, s[36:39], s90 offen
	s_add_i32 s90, s86, 0x180
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[18:21], v[2:5], a[128:131],  v178, v182 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[108:111], v155 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[18:21], v[6:9], a[132:135],  v178, v182 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[112:115], v155 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[18:21], v[10:13], a[136:139],  v178, v183 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[116:119], v155 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[18:21], v[14:17], a[140:143],  v178, v183 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[120:123], v155 offset:6144

	buffer_load_dwordx4 v146, s[0:3], s90 offen lds
	s_mov_b32 m0, s71
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[34:37], v[82:85], a[128:131],  v178, v182 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[124:127], v156 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[34:37], v[86:89], a[132:135],  v178, v182 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[128:131], v156 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[34:37], v[90:93], a[136:139],  v178, v183 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[132:135], v156 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[34:37], v[94:97], a[140:143],  v178, v183 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[136:139], v156 offset:6144

	s_nop 0
	buffer_load_dwordx4 v147, s[0:3], s90 offen lds
	s_mov_b32 m0, s72
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[22:25], v[2:5], a[144:147],  v178, v182 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[22:25], v[6:9], a[148:151],  v178, v182 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[22:25], v[10:13], a[152:155],  v178, v183 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[22:25], v[14:17], a[156:159],  v178, v183 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v154, s[0:3], s90 offen lds
	s_mov_b32 m0, s73
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[38:41], v[82:85], a[144:147],  v178, v182 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[38:41], v[86:89], a[148:151],  v178, v182 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[38:41], v[90:93], a[152:155],  v178, v183 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[38:41], v[94:97], a[156:159],  v178, v183 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v148, s[0:3], s90 offen lds
	s_mov_b32 m0, s50
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[26:29], v[2:5], a[160:163],  v179, v182 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[26:29], v[6:9], a[164:167],  v179, v182 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[26:29], v[10:13], a[168:171], v179, v183 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[26:29], v[14:17], a[172:175], v179, v183 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_add_i32 s90, s55, 0xfffff300
	buffer_load_dwordx4 v146, s[0:3], s91 offen lds
	s_mov_b32 m0, s74
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[42:45], v[82:85], a[160:163],  v179, v182 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[42:45], v[86:89], a[164:167],  v179, v182 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[42:45], v[90:93], a[168:171], v179, v183 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[42:45], v[94:97], a[172:175], v179, v183 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v147, s[0:3], s91 offen lds
	s_mov_b32 m0, s75
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[30:33], v[2:5], a[176:179], v179, v182 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[30:33], v[6:9], a[180:183], v179, v182 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[30:33], v[10:13], a[184:187], v179, v183 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[30:33], v[14:17], a[188:191], v179, v183 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v154, s[0:3], s91 offen lds
	s_mov_b32 m0, s76
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[46:49], v[82:85], a[176:179], v179, v182 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[46:49], v[86:89], a[180:183], v179, v182 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[46:49], v[90:93], a[184:187], v179, v183 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[46:49], v[94:97], a[188:191], v179, v183 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v148, s[0:3], s91 offen lds
	s_mov_b32 m0, s52
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[18:21], v[50:53], a[192:195],  v178, v180 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[2:5], v159 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[18:21], v[54:57], a[196:199],  v178, v180 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[6:9], v159 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[18:21], v[58:61], a[200:203],  v178, v181 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[10:13], v159 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[18:21], v[62:65], a[204:207],  v178, v181 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[14:17], v159 offset:6144

	s_add_i32 s91, s87, 0x200
	buffer_load_dwordx4 v150, s[40:43], s92 offen lds
	s_mov_b32 m0, s77
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[34:37], v[66:69], a[192:195],  v178, v180 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[82:85], v160 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[34:37], v[70:73], a[196:199],  v178, v180 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[86:89], v160 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[34:37], v[74:77], a[200:203],  v178, v181 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[90:93], v160 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[34:37], v[78:81], a[204:207],  v178, v181 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[94:97], v160 offset:6144

	s_nop 0
	buffer_load_dwordx4 v149, s[40:43], s92 offen lds
	s_mov_b32 m0, s78
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[22:25], v[50:53], a[208:211],  v178, v180 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[22:25], v[54:57], a[212:215],  v178, v180 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[22:25], v[58:61], a[216:219],  v178, v181 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[22:25], v[62:65], a[220:223],  v178, v181 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v151, s[40:43], s92 offen lds
	s_mov_b32 m0, s79
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[38:41], v[66:69], a[208:211],  v178, v180 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[38:41], v[70:73], a[212:215],  v178, v180 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[38:41], v[74:77], a[216:219],  v178, v181 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[38:41], v[78:81], a[220:223],  v178, v181 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v152, s[40:43], s92 offen lds
	s_mov_b32 m0, s54
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[26:29], v[50:53], a[224:227],  v179, v180 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[26:29], v[54:57], a[228:231],  v179, v180 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[26:29], v[58:61], a[232:235], v179, v181 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[26:29], v[62:65], a[236:239], v179, v181 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_add_i32 s92, s88, 0x200
	buffer_load_dwordx4 v150, s[40:43], s93 offen lds
	s_mov_b32 m0, s80
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[42:45], v[66:69], a[224:227],  v179, v180 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[42:45], v[70:73], a[228:231],  v179, v180 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[42:45], v[74:77], a[232:235], v179, v181 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[42:45], v[78:81], a[236:239], v179, v181 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v149, s[40:43], s93 offen lds
	s_mov_b32 m0, s81
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[30:33], v[50:53], a[240:243], v179, v180 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[30:33], v[54:57], a[244:247], v179, v180 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[30:33], v[58:61], a[248:251], v179, v181 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[30:33], v[62:65], a[252:255], v179, v181 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v151, s[40:43], s93 offen lds
	s_mov_b32 m0, s82
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[46:49], v[66:69], a[240:243], v179, v180 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[46:49], v[70:73], a[244:247], v179, v180 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[46:49], v[74:77], a[248:251], v179, v181 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[46:49], v[78:81], a[252:255], v179, v181 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v152, s[40:43], s93 offen lds
	s_waitcnt lgkmcnt(0)
	s_mov_b32 m0, s48
	s_add_i32 s93, s89, 0x200
	s_waitcnt vmcnt(16)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[108:111], v[2:5], a[28:31],  v99, v105 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v163 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[108:111], v[6:9], a[32:35],  v99, v105 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v163 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[108:111], v[10:13], a[36:39],  v99, v106 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v163 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[108:111], v[14:17], a[40:43],  v99, v106 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[62:65], v163 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[124:127], v[82:85], a[28:31],  v99, v105 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[66:69], v164 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[124:127], v[86:89], a[32:35],  v99, v105 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[70:73], v164 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[124:127], v[90:93], a[36:39],  v99, v106 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[74:77], v164 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[124:127], v[94:97], a[40:43],  v99, v106 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[78:81], v164 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[112:115], v[2:5], a[44:47],  v99, v105 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[112:115], v[6:9], a[48:51],  v99, v105 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[112:115], v[10:13], a[52:55],  v99, v106 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[112:115], v[14:17], a[56:59],  v99, v106 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[128:131], v[82:85], a[44:47],  v99, v105 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[128:131], v[86:89], a[48:51],  v99, v105 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[128:131], v[90:93], a[52:55],  v99, v106 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[128:131], v[94:97], a[56:59],  v99, v106 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[116:119], v[2:5], a[60:63],  v101, v105 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[116:119], v[6:9], a[64:67],  v101, v105 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[116:119], v[10:13], a[68:71], v101, v106 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[116:119], v[14:17], a[72:75], v101, v106 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[132:135], v[82:85], a[60:63],  v101, v105 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[132:135], v[86:89], a[64:67],  v101, v105 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[132:135], v[90:93], a[68:71], v101, v106 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[132:135], v[94:97], a[72:75], v101, v106 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[120:123], v[2:5], a[76:79], v101, v105 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[120:123], v[6:9], a[80:83], v101, v105 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[120:123], v[10:13], a[84:87], v101, v106 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[120:123], v[14:17], a[88:91], v101, v106 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[136:139], v[82:85], a[76:79], v101, v105 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[136:139], v[86:89], a[80:83], v101, v105 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[136:139], v[90:93], a[84:87], v101, v106 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[136:139], v[94:97], a[88:91], v101, v106 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[108:111], v[50:53], a[0:3], v99, v102 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[18:21], v167 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[108:111], v[54:57], a[4:7], v99, v102 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[22:25], v167 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[108:111], v[58:61], a[8:11], v99, v104 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[26:29], v167 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[108:111], v[62:65], a[12:15], v99, v104 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v167 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[124:127], v[66:69], a[0:3], v99, v102 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v168 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[124:127], v[70:73], a[4:7], v99, v102 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v168 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[124:127], v[74:77], a[8:11], v99, v104 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v168 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[124:127], v[78:81], a[12:15], v99, v104 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v168 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[112:115], v[50:53], a[16:19], v99, v102 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[112:115], v[54:57], a[20:23], v99, v102 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[112:115], v[58:61], a[24:27], v99, v104 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[112:115], v[62:65], a[92:95], v99, v104 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[128:131], v[66:69], a[16:19], v99, v102 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[128:131], v[70:73], a[20:23], v99, v102 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[128:131], v[74:77], a[24:27], v99, v104 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[128:131], v[78:81], a[92:95], v99, v104 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[116:119], v[50:53], a[96:99], v101, v102 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[116:119], v[54:57], a[100:103], v101, v102 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[116:119], v[58:61], a[104:107], v101, v104 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[116:119], v[62:65], a[108:111], v101, v104 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[132:135], v[66:69], a[96:99], v101, v102 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[132:135], v[70:73], a[100:103], v101, v102 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[132:135], v[74:77], a[104:107], v101, v104 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[132:135], v[78:81], a[108:111], v101, v104 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[120:123], v[50:53], a[112:115], v101, v102 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[120:123], v[54:57], a[116:119], v101, v102 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[120:123], v[58:61], a[120:123], v101, v104 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[120:123], v[62:65], a[124:127], v101, v104 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[136:139], v[66:69], a[112:115], v101, v102 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[136:139], v[70:73], a[116:119], v101, v102 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[136:139], v[74:77], a[120:123], v101, v104 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[136:139], v[78:81], a[124:127], v101, v104 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_waitcnt lgkmcnt(0)
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_barrier
	buffer_load_dword v108, v153, s[4:7], s90 offen
	buffer_load_dword v99, v153, s[8:11], s90 offen
	buffer_load_dword v110, v153, s[12:15], s90 offen
	buffer_load_dword v101, v153, s[16:19], s90 offen
	buffer_load_dword v109, v153, s[20:23], s90 offen
	buffer_load_dword v103, v153, s[24:27], s90 offen
	buffer_load_dword v111, v153, s[28:31], s90 offen
	buffer_load_dword v107, v153, s[36:39], s90 offen
	s_add_i32 s90, s86, 0x200
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[18:21], v[2:5], a[128:131],  v98, v105 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[112:115], v157 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[18:21], v[6:9], a[132:135],  v98, v105 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[116:119], v157 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[18:21], v[10:13], a[136:139],  v98, v106 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[120:123], v157 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[18:21], v[14:17], a[140:143],  v98, v106 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[124:127], v157 offset:6144

	buffer_load_dwordx4 v146, s[0:3], s90 offen lds
	s_mov_b32 m0, s56
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[34:37], v[82:85], a[128:131],  v98, v105 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[128:131], v158 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[34:37], v[86:89], a[132:135],  v98, v105 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[132:135], v158 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[34:37], v[90:93], a[136:139],  v98, v106 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[136:139], v158 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[34:37], v[94:97], a[140:143],  v98, v106 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[140:143], v158 offset:6144

	s_nop 0
	buffer_load_dwordx4 v147, s[0:3], s90 offen lds
	s_mov_b32 m0, s57
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[22:25], v[2:5], a[144:147],  v98, v105 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[22:25], v[6:9], a[148:151],  v98, v105 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[22:25], v[10:13], a[152:155],  v98, v106 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[22:25], v[14:17], a[156:159],  v98, v106 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v154, s[0:3], s90 offen lds
	s_mov_b32 m0, s58
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[38:41], v[82:85], a[144:147],  v98, v105 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[38:41], v[86:89], a[148:151],  v98, v105 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[38:41], v[90:93], a[152:155],  v98, v106 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[38:41], v[94:97], a[156:159],  v98, v106 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v148, s[0:3], s90 offen lds
	s_mov_b32 m0, s60
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[26:29], v[2:5], a[160:163],  v100, v105 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[26:29], v[6:9], a[164:167],  v100, v105 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[26:29], v[10:13], a[168:171], v100, v106 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[26:29], v[14:17], a[172:175], v100, v106 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_add_i32 s90, s55, 0xfffff400
	buffer_load_dwordx4 v146, s[0:3], s91 offen lds
	s_mov_b32 m0, s61
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[42:45], v[82:85], a[160:163],  v100, v105 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[42:45], v[86:89], a[164:167],  v100, v105 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[42:45], v[90:93], a[168:171], v100, v106 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[42:45], v[94:97], a[172:175], v100, v106 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v147, s[0:3], s91 offen lds
	s_mov_b32 m0, s62
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[30:33], v[2:5], a[176:179], v100, v105 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[30:33], v[6:9], a[180:183], v100, v105 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[30:33], v[10:13], a[184:187], v100, v106 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[30:33], v[14:17], a[188:191], v100, v106 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v154, s[0:3], s91 offen lds
	s_mov_b32 m0, s63
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[46:49], v[82:85], a[176:179], v100, v105 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[46:49], v[86:89], a[180:183], v100, v105 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[46:49], v[90:93], a[184:187], v100, v106 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[46:49], v[94:97], a[188:191], v100, v106 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v148, s[0:3], s91 offen lds
	s_mov_b32 m0, s51
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[18:21], v[50:53], a[192:195],  v98, v102 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[2:5], v161 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[18:21], v[54:57], a[196:199],  v98, v102 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[6:9], v161 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[18:21], v[58:61], a[200:203],  v98, v104 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[10:13], v161 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[18:21], v[62:65], a[204:207],  v98, v104 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[14:17], v161 offset:6144

	s_add_i32 s91, s87, 0x280
	buffer_load_dwordx4 v150, s[40:43], s92 offen lds
	s_mov_b32 m0, s64
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[34:37], v[66:69], a[192:195],  v98, v102 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[82:85], v162 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[34:37], v[70:73], a[196:199],  v98, v102 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[86:89], v162 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[34:37], v[74:77], a[200:203],  v98, v104 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[90:93], v162 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[34:37], v[78:81], a[204:207],  v98, v104 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[94:97], v162 offset:6144

	s_nop 0
	buffer_load_dwordx4 v149, s[40:43], s92 offen lds
	s_mov_b32 m0, s65
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[22:25], v[50:53], a[208:211],  v98, v102 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[22:25], v[54:57], a[212:215],  v98, v102 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[22:25], v[58:61], a[216:219],  v98, v104 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[22:25], v[62:65], a[220:223],  v98, v104 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v151, s[40:43], s92 offen lds
	s_mov_b32 m0, s66
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[38:41], v[66:69], a[208:211],  v98, v102 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[38:41], v[70:73], a[212:215],  v98, v102 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[38:41], v[74:77], a[216:219],  v98, v104 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[38:41], v[78:81], a[220:223],  v98, v104 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v152, s[40:43], s92 offen lds
	s_mov_b32 m0, s53
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[26:29], v[50:53], a[224:227],  v100, v102 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[26:29], v[54:57], a[228:231],  v100, v102 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[26:29], v[58:61], a[232:235], v100, v104 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[26:29], v[62:65], a[236:239], v100, v104 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_add_i32 s92, s88, 0x280
	buffer_load_dwordx4 v150, s[40:43], s93 offen lds
	s_mov_b32 m0, s68
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[42:45], v[66:69], a[224:227],  v100, v102 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[42:45], v[70:73], a[228:231],  v100, v102 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[42:45], v[74:77], a[232:235], v100, v104 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[42:45], v[78:81], a[236:239], v100, v104 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v149, s[40:43], s93 offen lds
	s_mov_b32 m0, s69
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[30:33], v[50:53], a[240:243], v100, v102 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[30:33], v[54:57], a[244:247], v100, v102 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[30:33], v[58:61], a[248:251], v100, v104 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[30:33], v[62:65], a[252:255], v100, v104 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v151, s[40:43], s93 offen lds
	s_mov_b32 m0, s70
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[46:49], v[66:69], a[240:243], v100, v102 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[46:49], v[70:73], a[244:247], v100, v102 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[46:49], v[74:77], a[248:251], v100, v104 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[46:49], v[78:81], a[252:255], v100, v104 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v152, s[40:43], s93 offen lds
	s_waitcnt lgkmcnt(0)
	s_mov_b32 m0, s49
	s_add_i32 s93, s89, 0x280
	s_waitcnt vmcnt(16)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[112:115], v[2:5], a[28:31],  v108, v109 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v165 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[112:115], v[6:9], a[32:35],  v108, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v165 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[112:115], v[10:13], a[36:39],  v108, v111 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v165 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[112:115], v[14:17], a[40:43],  v108, v111 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[62:65], v165 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[128:131], v[82:85], a[28:31],  v108, v109 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[66:69], v166 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[128:131], v[86:89], a[32:35],  v108, v109 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[70:73], v166 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[128:131], v[90:93], a[36:39],  v108, v111 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[74:77], v166 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[128:131], v[94:97], a[40:43],  v108, v111 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[78:81], v166 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[116:119], v[2:5], a[44:47],  v108, v109 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[116:119], v[6:9], a[48:51],  v108, v109 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[116:119], v[10:13], a[52:55],  v108, v111 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[116:119], v[14:17], a[56:59],  v108, v111 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[132:135], v[82:85], a[44:47],  v108, v109 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[132:135], v[86:89], a[48:51],  v108, v109 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[132:135], v[90:93], a[52:55],  v108, v111 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[132:135], v[94:97], a[56:59],  v108, v111 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[120:123], v[2:5], a[60:63],  v110, v109 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[120:123], v[6:9], a[64:67],  v110, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[120:123], v[10:13], a[68:71], v110, v111 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[120:123], v[14:17], a[72:75], v110, v111 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[136:139], v[82:85], a[60:63],  v110, v109 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[136:139], v[86:89], a[64:67],  v110, v109 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[136:139], v[90:93], a[68:71], v110, v111 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[136:139], v[94:97], a[72:75], v110, v111 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[124:127], v[2:5], a[76:79], v110, v109 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[124:127], v[6:9], a[80:83], v110, v109 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[124:127], v[10:13], a[84:87], v110, v111 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[124:127], v[14:17], a[88:91], v110, v111 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[140:143], v[82:85], a[76:79], v110, v109 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[140:143], v[86:89], a[80:83], v110, v109 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[140:143], v[90:93], a[84:87], v110, v111 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[140:143], v[94:97], a[88:91], v110, v111 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[112:115], v[50:53], a[0:3], v108, v103 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[18:21], v169 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[112:115], v[54:57], a[4:7], v108, v103 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[22:25], v169 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[112:115], v[58:61], a[8:11], v108, v107 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[26:29], v169 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[112:115], v[62:65], a[12:15], v108, v107 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v169 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[128:131], v[66:69], a[0:3], v108, v103 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v170 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[128:131], v[70:73], a[4:7], v108, v103 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v170 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[128:131], v[74:77], a[8:11], v108, v107 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v170 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[128:131], v[78:81], a[12:15], v108, v107 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v170 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[116:119], v[50:53], a[16:19], v108, v103 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[116:119], v[54:57], a[20:23], v108, v103 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[116:119], v[58:61], a[24:27], v108, v107 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[116:119], v[62:65], a[92:95], v108, v107 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[132:135], v[66:69], a[16:19], v108, v103 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[132:135], v[70:73], a[20:23], v108, v103 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[132:135], v[74:77], a[24:27], v108, v107 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[132:135], v[78:81], a[92:95], v108, v107 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[120:123], v[50:53], a[96:99], v110, v103 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[120:123], v[54:57], a[100:103], v110, v103 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[120:123], v[58:61], a[104:107], v110, v107 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[120:123], v[62:65], a[108:111], v110, v107 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[136:139], v[66:69], a[96:99], v110, v103 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[136:139], v[70:73], a[100:103], v110, v103 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[136:139], v[74:77], a[104:107], v110, v107 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[136:139], v[78:81], a[108:111], v110, v107 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[124:127], v[50:53], a[112:115], v110, v103 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[124:127], v[54:57], a[116:119], v110, v103 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[124:127], v[58:61], a[120:123], v110, v107 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[124:127], v[62:65], a[124:127], v110, v107 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[140:143], v[66:69], a[112:115], v110, v103 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[140:143], v[70:73], a[116:119], v110, v103 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[140:143], v[74:77], a[120:123], v110, v107 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[140:143], v[78:81], a[124:127], v110, v107 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_waitcnt lgkmcnt(0)
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_barrier
	buffer_load_dword v98, v153, s[4:7], s90 offen
	buffer_load_dword v104, v153, s[8:11], s90 offen
	buffer_load_dword v100, v153, s[12:15], s90 offen
	buffer_load_dword v105, v153, s[16:19], s90 offen
	buffer_load_dword v110, v153, s[20:23], s90 offen
	buffer_load_dword v106, v153, s[24:27], s90 offen
	buffer_load_dword v112, v153, s[28:31], s90 offen
	buffer_load_dword v108, v153, s[36:39], s90 offen
	s_add_i32 s90, s86, 0x280
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[18:21], v[2:5], a[128:131],  v99, v109 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[114:117], v155 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[18:21], v[6:9], a[132:135],  v99, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[118:121], v155 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[18:21], v[10:13], a[136:139],  v99, v111 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[122:125], v155 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[18:21], v[14:17], a[140:143],  v99, v111 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[126:129], v155 offset:6144

	buffer_load_dwordx4 v146, s[0:3], s90 offen lds
	s_mov_b32 m0, s71
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[34:37], v[82:85], a[128:131],  v99, v109 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[130:133], v156 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[34:37], v[86:89], a[132:135],  v99, v109 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[134:137], v156 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[34:37], v[90:93], a[136:139],  v99, v111 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[138:141], v156 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[34:37], v[94:97], a[140:143],  v99, v111 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[142:145], v156 offset:6144

	s_nop 0
	buffer_load_dwordx4 v147, s[0:3], s90 offen lds
	s_mov_b32 m0, s72
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[22:25], v[2:5], a[144:147],  v99, v109 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[22:25], v[6:9], a[148:151],  v99, v109 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[22:25], v[10:13], a[152:155],  v99, v111 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[22:25], v[14:17], a[156:159],  v99, v111 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v154, s[0:3], s90 offen lds
	s_mov_b32 m0, s73
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[38:41], v[82:85], a[144:147],  v99, v109 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[38:41], v[86:89], a[148:151],  v99, v109 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[38:41], v[90:93], a[152:155],  v99, v111 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[38:41], v[94:97], a[156:159],  v99, v111 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v148, s[0:3], s90 offen lds
	s_mov_b32 m0, s50
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[26:29], v[2:5], a[160:163],  v101, v109 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[26:29], v[6:9], a[164:167],  v101, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[26:29], v[10:13], a[168:171], v101, v111 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[26:29], v[14:17], a[172:175], v101, v111 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_add_i32 s90, s55, 0xfffff500
	buffer_load_dwordx4 v146, s[0:3], s91 offen lds
	s_mov_b32 m0, s74
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[42:45], v[82:85], a[160:163],  v101, v109 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[42:45], v[86:89], a[164:167],  v101, v109 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[42:45], v[90:93], a[168:171], v101, v111 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[42:45], v[94:97], a[172:175], v101, v111 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v147, s[0:3], s91 offen lds
	s_mov_b32 m0, s75
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[30:33], v[2:5], a[176:179], v101, v109 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[30:33], v[6:9], a[180:183], v101, v109 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[30:33], v[10:13], a[184:187], v101, v111 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[30:33], v[14:17], a[188:191], v101, v111 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v154, s[0:3], s91 offen lds
	s_mov_b32 m0, s76
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[46:49], v[82:85], a[176:179], v101, v109 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[46:49], v[86:89], a[180:183], v101, v109 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[46:49], v[90:93], a[184:187], v101, v111 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[46:49], v[94:97], a[188:191], v101, v111 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v148, s[0:3], s91 offen lds
	s_mov_b32 m0, s52
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[18:21], v[50:53], a[192:195],  v99, v103 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[2:5], v159 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[18:21], v[54:57], a[196:199],  v99, v103 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[6:9], v159 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[18:21], v[58:61], a[200:203],  v99, v107 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[10:13], v159 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[18:21], v[62:65], a[204:207],  v99, v107 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[14:17], v159 offset:6144

	s_add_i32 s91, s87, 0x300
	buffer_load_dwordx4 v150, s[40:43], s92 offen lds
	s_mov_b32 m0, s77
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[34:37], v[66:69], a[192:195],  v99, v103 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[82:85], v160 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[34:37], v[70:73], a[196:199],  v99, v103 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[86:89], v160 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[34:37], v[74:77], a[200:203],  v99, v107 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[90:93], v160 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[34:37], v[78:81], a[204:207],  v99, v107 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[94:97], v160 offset:6144

	s_nop 0
	buffer_load_dwordx4 v149, s[40:43], s92 offen lds
	s_mov_b32 m0, s78
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[22:25], v[50:53], a[208:211],  v99, v103 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[22:25], v[54:57], a[212:215],  v99, v103 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[22:25], v[58:61], a[216:219],  v99, v107 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[22:25], v[62:65], a[220:223],  v99, v107 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v151, s[40:43], s92 offen lds
	s_mov_b32 m0, s79
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[38:41], v[66:69], a[208:211],  v99, v103 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[38:41], v[70:73], a[212:215],  v99, v103 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[38:41], v[74:77], a[216:219],  v99, v107 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[38:41], v[78:81], a[220:223],  v99, v107 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v152, s[40:43], s92 offen lds
	s_mov_b32 m0, s54
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[26:29], v[50:53], a[224:227],  v101, v103 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[26:29], v[54:57], a[228:231],  v101, v103 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[26:29], v[58:61], a[232:235], v101, v107 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[26:29], v[62:65], a[236:239], v101, v107 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_add_i32 s92, s88, 0x300
	buffer_load_dwordx4 v150, s[40:43], s93 offen lds
	s_mov_b32 m0, s80
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[42:45], v[66:69], a[224:227],  v101, v103 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[42:45], v[70:73], a[228:231],  v101, v103 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[42:45], v[74:77], a[232:235], v101, v107 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[42:45], v[78:81], a[236:239], v101, v107 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v149, s[40:43], s93 offen lds
	s_mov_b32 m0, s81
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[30:33], v[50:53], a[240:243], v101, v103 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[30:33], v[54:57], a[244:247], v101, v103 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[30:33], v[58:61], a[248:251], v101, v107 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[30:33], v[62:65], a[252:255], v101, v107 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v151, s[40:43], s93 offen lds
	s_mov_b32 m0, s82
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[46:49], v[66:69], a[240:243], v101, v103 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[46:49], v[70:73], a[244:247], v101, v103 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[46:49], v[74:77], a[248:251], v101, v107 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[46:49], v[78:81], a[252:255], v101, v107 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v152, s[40:43], s93 offen lds
	s_waitcnt lgkmcnt(0)
	s_waitcnt vmcnt(16)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[114:117], v[2:5], a[28:31],  v98, v110 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v163 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[114:117], v[6:9], a[32:35],  v98, v110 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v163 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[114:117], v[10:13], a[36:39],  v98, v112 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v163 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[114:117], v[14:17], a[40:43],  v98, v112 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[62:65], v163 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[130:133], v[82:85], a[28:31],  v98, v110 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[66:69], v164 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[130:133], v[86:89], a[32:35],  v98, v110 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[70:73], v164 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[130:133], v[90:93], a[36:39],  v98, v112 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[74:77], v164 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[130:133], v[94:97], a[40:43],  v98, v112 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[78:81], v164 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[118:121], v[2:5], a[44:47],  v98, v110 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[118:121], v[6:9], a[48:51],  v98, v110 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[118:121], v[10:13], a[52:55],  v98, v112 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[118:121], v[14:17], a[56:59],  v98, v112 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[134:137], v[82:85], a[44:47],  v98, v110 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[134:137], v[86:89], a[48:51],  v98, v110 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[134:137], v[90:93], a[52:55],  v98, v112 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[134:137], v[94:97], a[56:59],  v98, v112 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[122:125], v[2:5], a[60:63],  v100, v110 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[122:125], v[6:9], a[64:67],  v100, v110 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[122:125], v[10:13], a[68:71], v100, v112 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[122:125], v[14:17], a[72:75], v100, v112 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[138:141], v[82:85], a[60:63],  v100, v110 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[138:141], v[86:89], a[64:67],  v100, v110 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[138:141], v[90:93], a[68:71], v100, v112 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[138:141], v[94:97], a[72:75], v100, v112 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[126:129], v[2:5], a[76:79], v100, v110 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[126:129], v[6:9], a[80:83], v100, v110 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[126:129], v[10:13], a[84:87], v100, v112 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[126:129], v[14:17], a[88:91], v100, v112 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[142:145], v[82:85], a[76:79], v100, v110 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[142:145], v[86:89], a[80:83], v100, v110 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[142:145], v[90:93], a[84:87], v100, v112 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[142:145], v[94:97], a[88:91], v100, v112 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[114:117], v[50:53], a[0:3], v98, v106 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[18:21], v167 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[114:117], v[54:57], a[4:7], v98, v106 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[22:25], v167 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[114:117], v[58:61], a[8:11], v98, v108 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[26:29], v167 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[114:117], v[62:65], a[12:15], v98, v108 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v167 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[130:133], v[66:69], a[0:3], v98, v106 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v168 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[130:133], v[70:73], a[4:7], v98, v106 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v168 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[130:133], v[74:77], a[8:11], v98, v108 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v168 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[130:133], v[78:81], a[12:15], v98, v108 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v168 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[118:121], v[50:53], a[16:19], v98, v106 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[118:121], v[54:57], a[20:23], v98, v106 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[118:121], v[58:61], a[24:27], v98, v108 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[118:121], v[62:65], a[92:95], v98, v108 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[134:137], v[66:69], a[16:19], v98, v106 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[134:137], v[70:73], a[20:23], v98, v106 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[134:137], v[74:77], a[24:27], v98, v108 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[134:137], v[78:81], a[92:95], v98, v108 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[122:125], v[50:53], a[96:99], v100, v106 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[122:125], v[54:57], a[100:103], v100, v106 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[122:125], v[58:61], a[104:107], v100, v108 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[122:125], v[62:65], a[108:111], v100, v108 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[138:141], v[66:69], a[96:99], v100, v106 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[138:141], v[70:73], a[100:103], v100, v106 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[138:141], v[74:77], a[104:107], v100, v108 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[138:141], v[78:81], a[108:111], v100, v108 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[126:129], v[50:53], a[112:115], v100, v106 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[126:129], v[54:57], a[116:119], v100, v106 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[126:129], v[58:61], a[120:123], v100, v108 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[126:129], v[62:65], a[124:127], v100, v108 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[142:145], v[66:69], a[112:115], v100, v106 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[142:145], v[70:73], a[116:119], v100, v106 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[142:145], v[74:77], a[120:123], v100, v108 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[142:145], v[78:81], a[124:127], v100, v108 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_waitcnt lgkmcnt(0)
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_barrier
	buffer_load_dword v107, v153, s[4:7], s90 offen
	buffer_load_dword v98, v153, s[8:11], s90 offen
	buffer_load_dword v109, v153, s[12:15], s90 offen
	buffer_load_dword v99, v153, s[16:19], s90 offen
	buffer_load_dword v102, v153, s[20:23], s90 offen
	buffer_load_dword v100, v153, s[24:27], s90 offen
	buffer_load_dword v103, v153, s[28:31], s90 offen
	buffer_load_dword v101, v153, s[36:39], s90 offen
	s_add_i32 s90, s86, 0x300
	s_mov_b32 m0, s48
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[18:21], v[2:5], a[128:131],  v104, v110 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[114:117], v157 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[18:21], v[6:9], a[132:135],  v104, v110 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[118:121], v157 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[18:21], v[10:13], a[136:139],  v104, v112 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[122:125], v157 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[18:21], v[14:17], a[140:143],  v104, v112 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[126:129], v157 offset:6144

	s_add_i32 s93, s89, 0x300
	buffer_load_dwordx4 v146, s[0:3], s90 offen lds
	s_mov_b32 m0, s56
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[34:37], v[82:85], a[128:131],  v104, v110 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[130:133], v158 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[34:37], v[86:89], a[132:135],  v104, v110 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[134:137], v158 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[34:37], v[90:93], a[136:139],  v104, v112 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[138:141], v158 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[34:37], v[94:97], a[140:143],  v104, v112 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[142:145], v158 offset:6144

	s_nop 0
	buffer_load_dwordx4 v147, s[0:3], s90 offen lds
	s_mov_b32 m0, s57
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[22:25], v[2:5], a[144:147],  v104, v110 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[22:25], v[6:9], a[148:151],  v104, v110 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[22:25], v[10:13], a[152:155],  v104, v112 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[22:25], v[14:17], a[156:159],  v104, v112 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v154, s[0:3], s90 offen lds
	s_mov_b32 m0, s58
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[38:41], v[82:85], a[144:147],  v104, v110 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[38:41], v[86:89], a[148:151],  v104, v110 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[38:41], v[90:93], a[152:155],  v104, v112 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[38:41], v[94:97], a[156:159],  v104, v112 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v148, s[0:3], s90 offen lds
	s_mov_b32 m0, s60
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[26:29], v[2:5], a[160:163],  v105, v110 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[26:29], v[6:9], a[164:167],  v105, v110 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[26:29], v[10:13], a[168:171], v105, v112 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[26:29], v[14:17], a[172:175], v105, v112 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_add_i32 s90, s55, 0xfffff600
	buffer_load_dwordx4 v146, s[0:3], s91 offen lds
	s_mov_b32 m0, s61
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[42:45], v[82:85], a[160:163],  v105, v110 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[42:45], v[86:89], a[164:167],  v105, v110 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[42:45], v[90:93], a[168:171], v105, v112 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[42:45], v[94:97], a[172:175], v105, v112 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v147, s[0:3], s91 offen lds
	s_mov_b32 m0, s62
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[30:33], v[2:5], a[176:179], v105, v110 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[30:33], v[6:9], a[180:183], v105, v110 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[30:33], v[10:13], a[184:187], v105, v112 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[30:33], v[14:17], a[188:191], v105, v112 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v154, s[0:3], s91 offen lds
	s_mov_b32 m0, s63
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[46:49], v[82:85], a[176:179], v105, v110 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[46:49], v[86:89], a[180:183], v105, v110 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[46:49], v[90:93], a[184:187], v105, v112 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[46:49], v[94:97], a[188:191], v105, v112 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v148, s[0:3], s91 offen lds
	s_mov_b32 m0, s51
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[18:21], v[50:53], a[192:195],  v104, v106 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[2:5], v161 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[18:21], v[54:57], a[196:199],  v104, v106 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[6:9], v161 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[18:21], v[58:61], a[200:203],  v104, v108 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[10:13], v161 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[18:21], v[62:65], a[204:207],  v104, v108 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[14:17], v161 offset:6144

	s_add_i32 s91, s87, 0x380
	buffer_load_dwordx4 v150, s[40:43], s92 offen lds
	s_mov_b32 m0, s64
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[34:37], v[66:69], a[192:195],  v104, v106 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[82:85], v162 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[34:37], v[70:73], a[196:199],  v104, v106 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[86:89], v162 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[34:37], v[74:77], a[200:203],  v104, v108 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[90:93], v162 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[34:37], v[78:81], a[204:207],  v104, v108 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[94:97], v162 offset:6144

	s_nop 0
	buffer_load_dwordx4 v149, s[40:43], s92 offen lds
	s_mov_b32 m0, s65
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[22:25], v[50:53], a[208:211],  v104, v106 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[22:25], v[54:57], a[212:215],  v104, v106 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[22:25], v[58:61], a[216:219],  v104, v108 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[22:25], v[62:65], a[220:223],  v104, v108 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v151, s[40:43], s92 offen lds
	s_mov_b32 m0, s66
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[38:41], v[66:69], a[208:211],  v104, v106 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[38:41], v[70:73], a[212:215],  v104, v106 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[38:41], v[74:77], a[216:219],  v104, v108 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[38:41], v[78:81], a[220:223],  v104, v108 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v152, s[40:43], s92 offen lds
	s_mov_b32 m0, s53
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[26:29], v[50:53], a[224:227],  v105, v106 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[26:29], v[54:57], a[228:231],  v105, v106 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[26:29], v[58:61], a[232:235], v105, v108 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[26:29], v[62:65], a[236:239], v105, v108 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_add_i32 s92, s88, 0x380
	buffer_load_dwordx4 v150, s[40:43], s93 offen lds
	s_mov_b32 m0, s68
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[42:45], v[66:69], a[224:227],  v105, v106 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[42:45], v[70:73], a[228:231],  v105, v106 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[42:45], v[74:77], a[232:235], v105, v108 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[42:45], v[78:81], a[236:239], v105, v108 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v149, s[40:43], s93 offen lds
	s_mov_b32 m0, s69
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[30:33], v[50:53], a[240:243], v105, v106 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[30:33], v[54:57], a[244:247], v105, v106 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[30:33], v[58:61], a[248:251], v105, v108 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[30:33], v[62:65], a[252:255], v105, v108 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v151, s[40:43], s93 offen lds
	s_mov_b32 m0, s70
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[46:49], v[66:69], a[240:243], v105, v106 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[46:49], v[70:73], a[244:247], v105, v106 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[46:49], v[74:77], a[248:251], v105, v108 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[46:49], v[78:81], a[252:255], v105, v108 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v152, s[40:43], s93 offen lds
	s_waitcnt lgkmcnt(0)
	s_waitcnt vmcnt(16)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[114:117], v[2:5], a[28:31],  v107, v102 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v165 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[114:117], v[6:9], a[32:35],  v107, v102 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v165 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[114:117], v[10:13], a[36:39],  v107, v103 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v165 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[114:117], v[14:17], a[40:43],  v107, v103 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[62:65], v165 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[130:133], v[82:85], a[28:31],  v107, v102 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[66:69], v166 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[130:133], v[86:89], a[32:35],  v107, v102 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[70:73], v166 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[130:133], v[90:93], a[36:39],  v107, v103 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[74:77], v166 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[130:133], v[94:97], a[40:43],  v107, v103 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[78:81], v166 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[118:121], v[2:5], a[44:47],  v107, v102 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[118:121], v[6:9], a[48:51],  v107, v102 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[118:121], v[10:13], a[52:55],  v107, v103 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[118:121], v[14:17], a[56:59],  v107, v103 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[134:137], v[82:85], a[44:47],  v107, v102 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[134:137], v[86:89], a[48:51],  v107, v102 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[134:137], v[90:93], a[52:55],  v107, v103 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[134:137], v[94:97], a[56:59],  v107, v103 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[122:125], v[2:5], a[60:63],  v109, v102 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[122:125], v[6:9], a[64:67],  v109, v102 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[122:125], v[10:13], a[68:71], v109, v103 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[122:125], v[14:17], a[72:75], v109, v103 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[138:141], v[82:85], a[60:63],  v109, v102 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[138:141], v[86:89], a[64:67],  v109, v102 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[138:141], v[90:93], a[68:71], v109, v103 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[138:141], v[94:97], a[72:75], v109, v103 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[126:129], v[2:5], a[76:79], v109, v102 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[126:129], v[6:9], a[80:83], v109, v102 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[126:129], v[10:13], a[84:87], v109, v103 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[126:129], v[14:17], a[88:91], v109, v103 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[142:145], v[82:85], a[76:79], v109, v102 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[142:145], v[86:89], a[80:83], v109, v102 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[142:145], v[90:93], a[84:87], v109, v103 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[142:145], v[94:97], a[88:91], v109, v103 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[114:117], v[50:53], a[0:3], v107, v100 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[18:21], v169 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[114:117], v[54:57], a[4:7], v107, v100 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[22:25], v169 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[114:117], v[58:61], a[8:11], v107, v101 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[26:29], v169 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[114:117], v[62:65], a[12:15], v107, v101 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v169 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[130:133], v[66:69], a[0:3], v107, v100 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v170 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[130:133], v[70:73], a[4:7], v107, v100 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v170 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[130:133], v[74:77], a[8:11], v107, v101 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v170 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[130:133], v[78:81], a[12:15], v107, v101 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v170 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[118:121], v[50:53], a[16:19], v107, v100 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[118:121], v[54:57], a[20:23], v107, v100 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[118:121], v[58:61], a[24:27], v107, v101 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[118:121], v[62:65], a[92:95], v107, v101 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[134:137], v[66:69], a[16:19], v107, v100 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[134:137], v[70:73], a[20:23], v107, v100 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[134:137], v[74:77], a[24:27], v107, v101 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[134:137], v[78:81], a[92:95], v107, v101 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[122:125], v[50:53], a[96:99], v109, v100 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[122:125], v[54:57], a[100:103], v109, v100 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[122:125], v[58:61], a[104:107], v109, v101 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[122:125], v[62:65], a[108:111], v109, v101 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[138:141], v[66:69], a[96:99], v109, v100 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[138:141], v[70:73], a[100:103], v109, v100 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[138:141], v[74:77], a[104:107], v109, v101 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[138:141], v[78:81], a[108:111], v109, v101 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[126:129], v[50:53], a[112:115], v109, v100 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[126:129], v[54:57], a[116:119], v109, v100 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[126:129], v[58:61], a[120:123], v109, v101 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[126:129], v[62:65], a[124:127], v109, v101 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[142:145], v[66:69], a[112:115], v109, v100 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[142:145], v[70:73], a[116:119], v109, v100 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[142:145], v[74:77], a[120:123], v109, v101 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[142:145], v[78:81], a[124:127], v109, v101 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_waitcnt lgkmcnt(0)
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_barrier
	buffer_load_dword v108, v153, s[4:7], s90 offen
	buffer_load_dword v104, v153, s[8:11], s90 offen
	buffer_load_dword v110, v153, s[12:15], s90 offen
	buffer_load_dword v105, v153, s[16:19], s90 offen
	buffer_load_dword v109, v153, s[20:23], s90 offen
	buffer_load_dword v106, v153, s[24:27], s90 offen
	buffer_load_dword v111, v153, s[28:31], s90 offen
	buffer_load_dword v107, v153, s[36:39], s90 offen
	s_add_i32 s90, s86, 0x380
	s_mov_b32 m0, s49
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[18:21], v[2:5], a[128:131],  v98, v102 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[112:115], v155 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[18:21], v[6:9], a[132:135],  v98, v102 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[116:119], v155 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[18:21], v[10:13], a[136:139],  v98, v103 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[120:123], v155 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[18:21], v[14:17], a[140:143],  v98, v103 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[124:127], v155 offset:6144

	s_add_i32 s93, s89, 0x380
	buffer_load_dwordx4 v146, s[0:3], s90 offen lds
	s_mov_b32 m0, s71
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[34:37], v[82:85], a[128:131],  v98, v102 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[128:131], v156 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[34:37], v[86:89], a[132:135],  v98, v102 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[132:135], v156 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[34:37], v[90:93], a[136:139],  v98, v103 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[136:139], v156 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[34:37], v[94:97], a[140:143],  v98, v103 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[140:143], v156 offset:6144

	s_nop 0
	buffer_load_dwordx4 v147, s[0:3], s90 offen lds
	s_mov_b32 m0, s72
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[22:25], v[2:5], a[144:147],  v98, v102 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[22:25], v[6:9], a[148:151],  v98, v102 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[22:25], v[10:13], a[152:155],  v98, v103 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[22:25], v[14:17], a[156:159],  v98, v103 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v154, s[0:3], s90 offen lds
	s_mov_b32 m0, s73
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[38:41], v[82:85], a[144:147],  v98, v102 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[38:41], v[86:89], a[148:151],  v98, v102 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[38:41], v[90:93], a[152:155],  v98, v103 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[38:41], v[94:97], a[156:159],  v98, v103 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v148, s[0:3], s90 offen lds
	s_mov_b32 m0, s50
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[26:29], v[2:5], a[160:163],  v99, v102 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[26:29], v[6:9], a[164:167],  v99, v102 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[26:29], v[10:13], a[168:171], v99, v103 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[26:29], v[14:17], a[172:175], v99, v103 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_add_i32 s90, s55, 0xfffff700
	buffer_load_dwordx4 v146, s[0:3], s91 offen lds
	s_mov_b32 m0, s74
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[42:45], v[82:85], a[160:163],  v99, v102 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[42:45], v[86:89], a[164:167],  v99, v102 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[42:45], v[90:93], a[168:171], v99, v103 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[42:45], v[94:97], a[172:175], v99, v103 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v147, s[0:3], s91 offen lds
	s_mov_b32 m0, s75
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[30:33], v[2:5], a[176:179], v99, v102 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[30:33], v[6:9], a[180:183], v99, v102 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[30:33], v[10:13], a[184:187], v99, v103 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[30:33], v[14:17], a[188:191], v99, v103 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v154, s[0:3], s91 offen lds
	s_mov_b32 m0, s76
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[46:49], v[82:85], a[176:179], v99, v102 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[46:49], v[86:89], a[180:183], v99, v102 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[46:49], v[90:93], a[184:187], v99, v103 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[46:49], v[94:97], a[188:191], v99, v103 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v148, s[0:3], s91 offen lds
	s_mov_b32 m0, s52
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[18:21], v[50:53], a[192:195],  v98, v100 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[2:5], v159 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[18:21], v[54:57], a[196:199],  v98, v100 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[6:9], v159 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[18:21], v[58:61], a[200:203],  v98, v101 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[10:13], v159 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[18:21], v[62:65], a[204:207],  v98, v101 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[14:17], v159 offset:6144

	s_add_i32 s91, s87, 0x400
	buffer_load_dwordx4 v150, s[40:43], s92 offen lds
	s_mov_b32 m0, s77
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[34:37], v[66:69], a[192:195],  v98, v100 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[82:85], v160 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[34:37], v[70:73], a[196:199],  v98, v100 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[86:89], v160 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[34:37], v[74:77], a[200:203],  v98, v101 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[90:93], v160 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[34:37], v[78:81], a[204:207],  v98, v101 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[94:97], v160 offset:6144

	s_nop 0
	buffer_load_dwordx4 v149, s[40:43], s92 offen lds
	s_mov_b32 m0, s78
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[22:25], v[50:53], a[208:211],  v98, v100 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[22:25], v[54:57], a[212:215],  v98, v100 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[22:25], v[58:61], a[216:219],  v98, v101 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[22:25], v[62:65], a[220:223],  v98, v101 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v151, s[40:43], s92 offen lds
	s_mov_b32 m0, s79
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[38:41], v[66:69], a[208:211],  v98, v100 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[38:41], v[70:73], a[212:215],  v98, v100 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[38:41], v[74:77], a[216:219],  v98, v101 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[38:41], v[78:81], a[220:223],  v98, v101 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v152, s[40:43], s92 offen lds
	s_mov_b32 m0, s54
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[26:29], v[50:53], a[224:227],  v99, v100 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[26:29], v[54:57], a[228:231],  v99, v100 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[26:29], v[58:61], a[232:235], v99, v101 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[26:29], v[62:65], a[236:239], v99, v101 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_add_i32 s92, s88, 0x400
	buffer_load_dwordx4 v150, s[40:43], s93 offen lds
	s_mov_b32 m0, s80
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[42:45], v[66:69], a[224:227],  v99, v100 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[42:45], v[70:73], a[228:231],  v99, v100 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[42:45], v[74:77], a[232:235], v99, v101 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[42:45], v[78:81], a[236:239], v99, v101 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v149, s[40:43], s93 offen lds
	s_mov_b32 m0, s81
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[30:33], v[50:53], a[240:243], v99, v100 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[30:33], v[54:57], a[244:247], v99, v100 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[30:33], v[58:61], a[248:251], v99, v101 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[30:33], v[62:65], a[252:255], v99, v101 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v151, s[40:43], s93 offen lds
	s_mov_b32 m0, s82
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[46:49], v[66:69], a[240:243], v99, v100 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[46:49], v[70:73], a[244:247], v99, v100 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[46:49], v[74:77], a[248:251], v99, v101 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[46:49], v[78:81], a[252:255], v99, v101 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v152, s[40:43], s93 offen lds
	s_waitcnt lgkmcnt(0)
	s_waitcnt vmcnt(16)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[112:115], v[2:5], a[28:31],  v108, v109 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v163 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[112:115], v[6:9], a[32:35],  v108, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v163 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[112:115], v[10:13], a[36:39],  v108, v111 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v163 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[112:115], v[14:17], a[40:43],  v108, v111 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[62:65], v163 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[128:131], v[82:85], a[28:31],  v108, v109 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[66:69], v164 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[128:131], v[86:89], a[32:35],  v108, v109 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[70:73], v164 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[128:131], v[90:93], a[36:39],  v108, v111 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[74:77], v164 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[128:131], v[94:97], a[40:43],  v108, v111 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[78:81], v164 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[116:119], v[2:5], a[44:47],  v108, v109 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[116:119], v[6:9], a[48:51],  v108, v109 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[116:119], v[10:13], a[52:55],  v108, v111 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[116:119], v[14:17], a[56:59],  v108, v111 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[132:135], v[82:85], a[44:47],  v108, v109 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[132:135], v[86:89], a[48:51],  v108, v109 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[132:135], v[90:93], a[52:55],  v108, v111 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[132:135], v[94:97], a[56:59],  v108, v111 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[120:123], v[2:5], a[60:63],  v110, v109 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[120:123], v[6:9], a[64:67],  v110, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[120:123], v[10:13], a[68:71], v110, v111 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[120:123], v[14:17], a[72:75], v110, v111 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[136:139], v[82:85], a[60:63],  v110, v109 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[136:139], v[86:89], a[64:67],  v110, v109 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[136:139], v[90:93], a[68:71], v110, v111 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[136:139], v[94:97], a[72:75], v110, v111 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[124:127], v[2:5], a[76:79], v110, v109 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[124:127], v[6:9], a[80:83], v110, v109 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[124:127], v[10:13], a[84:87], v110, v111 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[124:127], v[14:17], a[88:91], v110, v111 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[140:143], v[82:85], a[76:79], v110, v109 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[140:143], v[86:89], a[80:83], v110, v109 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[140:143], v[90:93], a[84:87], v110, v111 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[140:143], v[94:97], a[88:91], v110, v111 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[112:115], v[50:53], a[0:3], v108, v106 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[18:21], v167 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[112:115], v[54:57], a[4:7], v108, v106 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[22:25], v167 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[112:115], v[58:61], a[8:11], v108, v107 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[26:29], v167 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[112:115], v[62:65], a[12:15], v108, v107 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v167 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[128:131], v[66:69], a[0:3], v108, v106 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v168 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[128:131], v[70:73], a[4:7], v108, v106 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v168 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[128:131], v[74:77], a[8:11], v108, v107 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v168 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[128:131], v[78:81], a[12:15], v108, v107 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v168 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[116:119], v[50:53], a[16:19], v108, v106 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[116:119], v[54:57], a[20:23], v108, v106 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[116:119], v[58:61], a[24:27], v108, v107 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[116:119], v[62:65], a[92:95], v108, v107 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[132:135], v[66:69], a[16:19], v108, v106 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[132:135], v[70:73], a[20:23], v108, v106 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[132:135], v[74:77], a[24:27], v108, v107 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[132:135], v[78:81], a[92:95], v108, v107 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[120:123], v[50:53], a[96:99], v110, v106 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[120:123], v[54:57], a[100:103], v110, v106 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[120:123], v[58:61], a[104:107], v110, v107 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[120:123], v[62:65], a[108:111], v110, v107 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[136:139], v[66:69], a[96:99], v110, v106 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[136:139], v[70:73], a[100:103], v110, v106 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[136:139], v[74:77], a[104:107], v110, v107 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[136:139], v[78:81], a[108:111], v110, v107 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[124:127], v[50:53], a[112:115], v110, v106 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[124:127], v[54:57], a[116:119], v110, v106 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[124:127], v[58:61], a[120:123], v110, v107 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[124:127], v[62:65], a[124:127], v110, v107 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[140:143], v[66:69], a[112:115], v110, v106 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[140:143], v[70:73], a[116:119], v110, v106 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[140:143], v[74:77], a[120:123], v110, v107 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[140:143], v[78:81], a[124:127], v110, v107 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_waitcnt lgkmcnt(0)
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_barrier
	buffer_load_dword v99, v153, s[4:7], s90 offen
	buffer_load_dword v98, v153, s[8:11], s90 offen
	buffer_load_dword v101, v153, s[12:15], s90 offen
	buffer_load_dword v100, v153, s[16:19], s90 offen
	buffer_load_dword v110, v153, s[20:23], s90 offen
	buffer_load_dword v102, v153, s[24:27], s90 offen
	buffer_load_dword v112, v153, s[28:31], s90 offen
	buffer_load_dword v108, v153, s[36:39], s90 offen
	s_add_i32 s90, s86, 0x400
	s_mov_b32 m0, s48
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[18:21], v[2:5], a[128:131],  v104, v109 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[114:117], v157 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[18:21], v[6:9], a[132:135],  v104, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[118:121], v157 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[18:21], v[10:13], a[136:139],  v104, v111 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[122:125], v157 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[18:21], v[14:17], a[140:143],  v104, v111 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[126:129], v157 offset:6144

	s_add_i32 s93, s89, 0x400
	buffer_load_dwordx4 v146, s[0:3], s90 offen lds
	s_mov_b32 m0, s56
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[34:37], v[82:85], a[128:131],  v104, v109 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[130:133], v158 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[34:37], v[86:89], a[132:135],  v104, v109 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[134:137], v158 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[34:37], v[90:93], a[136:139],  v104, v111 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[138:141], v158 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[34:37], v[94:97], a[140:143],  v104, v111 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[142:145], v158 offset:6144

	s_nop 0
	buffer_load_dwordx4 v147, s[0:3], s90 offen lds
	s_mov_b32 m0, s57
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[22:25], v[2:5], a[144:147],  v104, v109 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[22:25], v[6:9], a[148:151],  v104, v109 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[22:25], v[10:13], a[152:155],  v104, v111 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[22:25], v[14:17], a[156:159],  v104, v111 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v154, s[0:3], s90 offen lds
	s_mov_b32 m0, s58
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[38:41], v[82:85], a[144:147],  v104, v109 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[38:41], v[86:89], a[148:151],  v104, v109 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[38:41], v[90:93], a[152:155],  v104, v111 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[38:41], v[94:97], a[156:159],  v104, v111 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v148, s[0:3], s90 offen lds
	s_mov_b32 m0, s60
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[26:29], v[2:5], a[160:163],  v105, v109 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[26:29], v[6:9], a[164:167],  v105, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[26:29], v[10:13], a[168:171], v105, v111 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[26:29], v[14:17], a[172:175], v105, v111 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_add_i32 s90, s55, 0xfffff800
	buffer_load_dwordx4 v146, s[0:3], s91 offen lds
	s_mov_b32 m0, s61
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[42:45], v[82:85], a[160:163],  v105, v109 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[42:45], v[86:89], a[164:167],  v105, v109 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[42:45], v[90:93], a[168:171], v105, v111 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[42:45], v[94:97], a[172:175], v105, v111 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v147, s[0:3], s91 offen lds
	s_mov_b32 m0, s62
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[30:33], v[2:5], a[176:179], v105, v109 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[30:33], v[6:9], a[180:183], v105, v109 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[30:33], v[10:13], a[184:187], v105, v111 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[30:33], v[14:17], a[188:191], v105, v111 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v154, s[0:3], s91 offen lds
	s_mov_b32 m0, s63
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[46:49], v[82:85], a[176:179], v105, v109 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[46:49], v[86:89], a[180:183], v105, v109 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[46:49], v[90:93], a[184:187], v105, v111 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[46:49], v[94:97], a[188:191], v105, v111 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v148, s[0:3], s91 offen lds
	s_mov_b32 m0, s51
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[18:21], v[50:53], a[192:195],  v104, v106 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[2:5], v161 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[18:21], v[54:57], a[196:199],  v104, v106 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[6:9], v161 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[18:21], v[58:61], a[200:203],  v104, v107 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[10:13], v161 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[18:21], v[62:65], a[204:207],  v104, v107 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[14:17], v161 offset:6144

	s_add_i32 s91, s87, 0x480
	buffer_load_dwordx4 v150, s[40:43], s92 offen lds
	s_mov_b32 m0, s64
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[34:37], v[66:69], a[192:195],  v104, v106 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[82:85], v162 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[34:37], v[70:73], a[196:199],  v104, v106 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[86:89], v162 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[34:37], v[74:77], a[200:203],  v104, v107 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[90:93], v162 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[34:37], v[78:81], a[204:207],  v104, v107 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[94:97], v162 offset:6144

	s_nop 0
	buffer_load_dwordx4 v149, s[40:43], s92 offen lds
	s_mov_b32 m0, s65
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[22:25], v[50:53], a[208:211],  v104, v106 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[22:25], v[54:57], a[212:215],  v104, v106 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[22:25], v[58:61], a[216:219],  v104, v107 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[22:25], v[62:65], a[220:223],  v104, v107 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v151, s[40:43], s92 offen lds
	s_mov_b32 m0, s66
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[38:41], v[66:69], a[208:211],  v104, v106 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[38:41], v[70:73], a[212:215],  v104, v106 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[38:41], v[74:77], a[216:219],  v104, v107 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[38:41], v[78:81], a[220:223],  v104, v107 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v152, s[40:43], s92 offen lds
	s_mov_b32 m0, s53
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[26:29], v[50:53], a[224:227],  v105, v106 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[26:29], v[54:57], a[228:231],  v105, v106 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[26:29], v[58:61], a[232:235], v105, v107 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[26:29], v[62:65], a[236:239], v105, v107 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_add_i32 s92, s88, 0x480
	buffer_load_dwordx4 v150, s[40:43], s93 offen lds
	s_mov_b32 m0, s68
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[42:45], v[66:69], a[224:227],  v105, v106 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[42:45], v[70:73], a[228:231],  v105, v106 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[42:45], v[74:77], a[232:235], v105, v107 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[42:45], v[78:81], a[236:239], v105, v107 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v149, s[40:43], s93 offen lds
	s_mov_b32 m0, s69
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[30:33], v[50:53], a[240:243], v105, v106 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[30:33], v[54:57], a[244:247], v105, v106 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[30:33], v[58:61], a[248:251], v105, v107 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[30:33], v[62:65], a[252:255], v105, v107 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v151, s[40:43], s93 offen lds
	s_mov_b32 m0, s70
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[46:49], v[66:69], a[240:243], v105, v106 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[46:49], v[70:73], a[244:247], v105, v106 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[46:49], v[74:77], a[248:251], v105, v107 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[46:49], v[78:81], a[252:255], v105, v107 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v152, s[40:43], s93 offen lds
	s_waitcnt lgkmcnt(0)
	s_waitcnt vmcnt(16)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[114:117], v[2:5], a[28:31],  v99, v110 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v165 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[114:117], v[6:9], a[32:35],  v99, v110 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v165 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[114:117], v[10:13], a[36:39],  v99, v112 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v165 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[114:117], v[14:17], a[40:43],  v99, v112 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[62:65], v165 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[130:133], v[82:85], a[28:31],  v99, v110 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[66:69], v166 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[130:133], v[86:89], a[32:35],  v99, v110 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[70:73], v166 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[130:133], v[90:93], a[36:39],  v99, v112 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[74:77], v166 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[130:133], v[94:97], a[40:43],  v99, v112 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[78:81], v166 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[118:121], v[2:5], a[44:47],  v99, v110 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[118:121], v[6:9], a[48:51],  v99, v110 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[118:121], v[10:13], a[52:55],  v99, v112 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[118:121], v[14:17], a[56:59],  v99, v112 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[134:137], v[82:85], a[44:47],  v99, v110 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[134:137], v[86:89], a[48:51],  v99, v110 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[134:137], v[90:93], a[52:55],  v99, v112 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[134:137], v[94:97], a[56:59],  v99, v112 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[122:125], v[2:5], a[60:63],  v101, v110 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[122:125], v[6:9], a[64:67],  v101, v110 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[122:125], v[10:13], a[68:71], v101, v112 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[122:125], v[14:17], a[72:75], v101, v112 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[138:141], v[82:85], a[60:63],  v101, v110 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[138:141], v[86:89], a[64:67],  v101, v110 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[138:141], v[90:93], a[68:71], v101, v112 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[138:141], v[94:97], a[72:75], v101, v112 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[126:129], v[2:5], a[76:79], v101, v110 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[126:129], v[6:9], a[80:83], v101, v110 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[126:129], v[10:13], a[84:87], v101, v112 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[126:129], v[14:17], a[88:91], v101, v112 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[142:145], v[82:85], a[76:79], v101, v110 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[142:145], v[86:89], a[80:83], v101, v110 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[142:145], v[90:93], a[84:87], v101, v112 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[142:145], v[94:97], a[88:91], v101, v112 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[114:117], v[50:53], a[0:3], v99, v102 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[18:21], v169 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[114:117], v[54:57], a[4:7], v99, v102 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[22:25], v169 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[114:117], v[58:61], a[8:11], v99, v108 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[26:29], v169 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[114:117], v[62:65], a[12:15], v99, v108 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v169 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[130:133], v[66:69], a[0:3], v99, v102 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v170 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[130:133], v[70:73], a[4:7], v99, v102 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v170 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[130:133], v[74:77], a[8:11], v99, v108 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v170 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[130:133], v[78:81], a[12:15], v99, v108 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v170 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[118:121], v[50:53], a[16:19], v99, v102 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[118:121], v[54:57], a[20:23], v99, v102 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[118:121], v[58:61], a[24:27], v99, v108 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[118:121], v[62:65], a[92:95], v99, v108 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[134:137], v[66:69], a[16:19], v99, v102 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[134:137], v[70:73], a[20:23], v99, v102 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[134:137], v[74:77], a[24:27], v99, v108 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[134:137], v[78:81], a[92:95], v99, v108 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[122:125], v[50:53], a[96:99], v101, v102 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[122:125], v[54:57], a[100:103], v101, v102 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[122:125], v[58:61], a[104:107], v101, v108 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[122:125], v[62:65], a[108:111], v101, v108 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[138:141], v[66:69], a[96:99], v101, v102 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[138:141], v[70:73], a[100:103], v101, v102 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[138:141], v[74:77], a[104:107], v101, v108 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[138:141], v[78:81], a[108:111], v101, v108 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[126:129], v[50:53], a[112:115], v101, v102 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[126:129], v[54:57], a[116:119], v101, v102 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[126:129], v[58:61], a[120:123], v101, v108 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[126:129], v[62:65], a[124:127], v101, v108 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[142:145], v[66:69], a[112:115], v101, v102 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[142:145], v[70:73], a[116:119], v101, v102 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[142:145], v[74:77], a[120:123], v101, v108 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[142:145], v[78:81], a[124:127], v101, v108 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_waitcnt lgkmcnt(0)
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_barrier
	buffer_load_dword v105, v153, s[4:7], s90 offen
	buffer_load_dword v99, v153, s[8:11], s90 offen
	buffer_load_dword v109, v153, s[12:15], s90 offen
	buffer_load_dword v101, v153, s[16:19], s90 offen
	buffer_load_dword v106, v153, s[20:23], s90 offen
	buffer_load_dword v103, v153, s[24:27], s90 offen
	buffer_load_dword v107, v153, s[28:31], s90 offen
	buffer_load_dword v104, v153, s[36:39], s90 offen
	s_add_i32 s90, s86, 0x480
	s_mov_b32 m0, s49
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[18:21], v[2:5], a[128:131],  v98, v110 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[114:117], v155 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[18:21], v[6:9], a[132:135],  v98, v110 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[118:121], v155 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[18:21], v[10:13], a[136:139],  v98, v112 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[122:125], v155 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[18:21], v[14:17], a[140:143],  v98, v112 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[126:129], v155 offset:6144

	s_add_i32 s93, s89, 0x480
	buffer_load_dwordx4 v146, s[0:3], s90 offen lds
	s_mov_b32 m0, s71
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[34:37], v[82:85], a[128:131],  v98, v110 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[130:133], v156 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[34:37], v[86:89], a[132:135],  v98, v110 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[134:137], v156 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[34:37], v[90:93], a[136:139],  v98, v112 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[138:141], v156 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[34:37], v[94:97], a[140:143],  v98, v112 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[142:145], v156 offset:6144

	s_nop 0
	buffer_load_dwordx4 v147, s[0:3], s90 offen lds
	s_mov_b32 m0, s72
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[22:25], v[2:5], a[144:147],  v98, v110 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[22:25], v[6:9], a[148:151],  v98, v110 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[22:25], v[10:13], a[152:155],  v98, v112 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[22:25], v[14:17], a[156:159],  v98, v112 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v154, s[0:3], s90 offen lds
	s_mov_b32 m0, s73
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[38:41], v[82:85], a[144:147],  v98, v110 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[38:41], v[86:89], a[148:151],  v98, v110 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[38:41], v[90:93], a[152:155],  v98, v112 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[38:41], v[94:97], a[156:159],  v98, v112 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v148, s[0:3], s90 offen lds
	s_mov_b32 m0, s50
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[26:29], v[2:5], a[160:163],  v100, v110 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[26:29], v[6:9], a[164:167],  v100, v110 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[26:29], v[10:13], a[168:171], v100, v112 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[26:29], v[14:17], a[172:175], v100, v112 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_add_i32 s90, s55, 0xfffff900
	buffer_load_dwordx4 v146, s[0:3], s91 offen lds
	s_mov_b32 m0, s74
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[42:45], v[82:85], a[160:163],  v100, v110 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[42:45], v[86:89], a[164:167],  v100, v110 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[42:45], v[90:93], a[168:171], v100, v112 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[42:45], v[94:97], a[172:175], v100, v112 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v147, s[0:3], s91 offen lds
	s_mov_b32 m0, s75
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[30:33], v[2:5], a[176:179], v100, v110 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[30:33], v[6:9], a[180:183], v100, v110 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[30:33], v[10:13], a[184:187], v100, v112 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[30:33], v[14:17], a[188:191], v100, v112 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v154, s[0:3], s91 offen lds
	s_mov_b32 m0, s76
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[46:49], v[82:85], a[176:179], v100, v110 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[46:49], v[86:89], a[180:183], v100, v110 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[46:49], v[90:93], a[184:187], v100, v112 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[46:49], v[94:97], a[188:191], v100, v112 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v148, s[0:3], s91 offen lds
	s_mov_b32 m0, s52
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[18:21], v[50:53], a[192:195],  v98, v102 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[2:5], v159 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[18:21], v[54:57], a[196:199],  v98, v102 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[6:9], v159 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[18:21], v[58:61], a[200:203],  v98, v108 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[10:13], v159 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[18:21], v[62:65], a[204:207],  v98, v108 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[14:17], v159 offset:6144

	s_add_i32 s91, s87, 0x500
	buffer_load_dwordx4 v150, s[40:43], s92 offen lds
	s_mov_b32 m0, s77
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[34:37], v[66:69], a[192:195],  v98, v102 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[82:85], v160 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[34:37], v[70:73], a[196:199],  v98, v102 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[86:89], v160 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[34:37], v[74:77], a[200:203],  v98, v108 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[90:93], v160 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[34:37], v[78:81], a[204:207],  v98, v108 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[94:97], v160 offset:6144

	s_nop 0
	buffer_load_dwordx4 v149, s[40:43], s92 offen lds
	s_mov_b32 m0, s78
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[22:25], v[50:53], a[208:211],  v98, v102 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[22:25], v[54:57], a[212:215],  v98, v102 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[22:25], v[58:61], a[216:219],  v98, v108 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[22:25], v[62:65], a[220:223],  v98, v108 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v151, s[40:43], s92 offen lds
	s_mov_b32 m0, s79
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[38:41], v[66:69], a[208:211],  v98, v102 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[38:41], v[70:73], a[212:215],  v98, v102 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[38:41], v[74:77], a[216:219],  v98, v108 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[38:41], v[78:81], a[220:223],  v98, v108 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v152, s[40:43], s92 offen lds
	s_mov_b32 m0, s54
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[26:29], v[50:53], a[224:227],  v100, v102 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[26:29], v[54:57], a[228:231],  v100, v102 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[26:29], v[58:61], a[232:235], v100, v108 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[26:29], v[62:65], a[236:239], v100, v108 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_add_i32 s92, s88, 0x500
	buffer_load_dwordx4 v150, s[40:43], s93 offen lds
	s_mov_b32 m0, s80
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[42:45], v[66:69], a[224:227],  v100, v102 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[42:45], v[70:73], a[228:231],  v100, v102 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[42:45], v[74:77], a[232:235], v100, v108 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[42:45], v[78:81], a[236:239], v100, v108 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v149, s[40:43], s93 offen lds
	s_mov_b32 m0, s81
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[30:33], v[50:53], a[240:243], v100, v102 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[30:33], v[54:57], a[244:247], v100, v102 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[30:33], v[58:61], a[248:251], v100, v108 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[30:33], v[62:65], a[252:255], v100, v108 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v151, s[40:43], s93 offen lds
	s_mov_b32 m0, s82
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[46:49], v[66:69], a[240:243], v100, v102 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[46:49], v[70:73], a[244:247], v100, v102 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[46:49], v[74:77], a[248:251], v100, v108 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[46:49], v[78:81], a[252:255], v100, v108 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v152, s[40:43], s93 offen lds
	s_waitcnt lgkmcnt(0)
	s_waitcnt vmcnt(16)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[114:117], v[2:5], a[28:31],  v105, v106 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v163 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[114:117], v[6:9], a[32:35],  v105, v106 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v163 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[114:117], v[10:13], a[36:39],  v105, v107 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v163 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[114:117], v[14:17], a[40:43],  v105, v107 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[62:65], v163 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[130:133], v[82:85], a[28:31],  v105, v106 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[66:69], v164 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[130:133], v[86:89], a[32:35],  v105, v106 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[70:73], v164 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[130:133], v[90:93], a[36:39],  v105, v107 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[74:77], v164 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[130:133], v[94:97], a[40:43],  v105, v107 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[78:81], v164 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[118:121], v[2:5], a[44:47],  v105, v106 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[118:121], v[6:9], a[48:51],  v105, v106 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[118:121], v[10:13], a[52:55],  v105, v107 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[118:121], v[14:17], a[56:59],  v105, v107 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[134:137], v[82:85], a[44:47],  v105, v106 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[134:137], v[86:89], a[48:51],  v105, v106 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[134:137], v[90:93], a[52:55],  v105, v107 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[134:137], v[94:97], a[56:59],  v105, v107 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[122:125], v[2:5], a[60:63],  v109, v106 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[122:125], v[6:9], a[64:67],  v109, v106 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[122:125], v[10:13], a[68:71], v109, v107 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[122:125], v[14:17], a[72:75], v109, v107 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[138:141], v[82:85], a[60:63],  v109, v106 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[138:141], v[86:89], a[64:67],  v109, v106 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[138:141], v[90:93], a[68:71], v109, v107 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[138:141], v[94:97], a[72:75], v109, v107 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[126:129], v[2:5], a[76:79], v109, v106 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[126:129], v[6:9], a[80:83], v109, v106 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[126:129], v[10:13], a[84:87], v109, v107 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[126:129], v[14:17], a[88:91], v109, v107 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[142:145], v[82:85], a[76:79], v109, v106 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[142:145], v[86:89], a[80:83], v109, v106 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[142:145], v[90:93], a[84:87], v109, v107 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[142:145], v[94:97], a[88:91], v109, v107 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[114:117], v[50:53], a[0:3], v105, v103 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[18:21], v167 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[114:117], v[54:57], a[4:7], v105, v103 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[22:25], v167 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[114:117], v[58:61], a[8:11], v105, v104 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[26:29], v167 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[114:117], v[62:65], a[12:15], v105, v104 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v167 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[130:133], v[66:69], a[0:3], v105, v103 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v168 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[130:133], v[70:73], a[4:7], v105, v103 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v168 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[130:133], v[74:77], a[8:11], v105, v104 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v168 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[130:133], v[78:81], a[12:15], v105, v104 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v168 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[118:121], v[50:53], a[16:19], v105, v103 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[118:121], v[54:57], a[20:23], v105, v103 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[118:121], v[58:61], a[24:27], v105, v104 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[118:121], v[62:65], a[92:95], v105, v104 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[134:137], v[66:69], a[16:19], v105, v103 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[134:137], v[70:73], a[20:23], v105, v103 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[134:137], v[74:77], a[24:27], v105, v104 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[134:137], v[78:81], a[92:95], v105, v104 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[122:125], v[50:53], a[96:99], v109, v103 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[122:125], v[54:57], a[100:103], v109, v103 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[122:125], v[58:61], a[104:107], v109, v104 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[122:125], v[62:65], a[108:111], v109, v104 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[138:141], v[66:69], a[96:99], v109, v103 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[138:141], v[70:73], a[100:103], v109, v103 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[138:141], v[74:77], a[104:107], v109, v104 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[138:141], v[78:81], a[108:111], v109, v104 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[126:129], v[50:53], a[112:115], v109, v103 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[126:129], v[54:57], a[116:119], v109, v103 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[126:129], v[58:61], a[120:123], v109, v104 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[126:129], v[62:65], a[124:127], v109, v104 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[142:145], v[66:69], a[112:115], v109, v103 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[142:145], v[70:73], a[116:119], v109, v103 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[142:145], v[74:77], a[120:123], v109, v104 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[142:145], v[78:81], a[124:127], v109, v104 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_waitcnt lgkmcnt(0)
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_barrier
	buffer_load_dword v109, v153, s[4:7], s90 offen
	buffer_load_dword v98, v153, s[8:11], s90 offen
	buffer_load_dword v111, v153, s[12:15], s90 offen
	buffer_load_dword v100, v153, s[16:19], s90 offen
	buffer_load_dword v108, v153, s[20:23], s90 offen
	buffer_load_dword v102, v153, s[24:27], s90 offen
	buffer_load_dword v110, v153, s[28:31], s90 offen
	buffer_load_dword v105, v153, s[36:39], s90 offen
	s_add_i32 s90, s86, 0x500
	s_mov_b32 m0, s48
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[18:21], v[2:5], a[128:131],  v99, v106 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[112:115], v157 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[18:21], v[6:9], a[132:135],  v99, v106 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[116:119], v157 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[18:21], v[10:13], a[136:139],  v99, v107 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[120:123], v157 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[18:21], v[14:17], a[140:143],  v99, v107 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[124:127], v157 offset:6144

	s_add_i32 s93, s89, 0x500
	buffer_load_dwordx4 v146, s[0:3], s90 offen lds
	s_mov_b32 m0, s56
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[34:37], v[82:85], a[128:131],  v99, v106 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[128:131], v158 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[34:37], v[86:89], a[132:135],  v99, v106 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[132:135], v158 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[34:37], v[90:93], a[136:139],  v99, v107 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[136:139], v158 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[34:37], v[94:97], a[140:143],  v99, v107 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[140:143], v158 offset:6144

	s_nop 0
	buffer_load_dwordx4 v147, s[0:3], s90 offen lds
	s_mov_b32 m0, s57
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[22:25], v[2:5], a[144:147],  v99, v106 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[22:25], v[6:9], a[148:151],  v99, v106 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[22:25], v[10:13], a[152:155],  v99, v107 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[22:25], v[14:17], a[156:159],  v99, v107 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v154, s[0:3], s90 offen lds
	s_mov_b32 m0, s58
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[38:41], v[82:85], a[144:147],  v99, v106 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[38:41], v[86:89], a[148:151],  v99, v106 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[38:41], v[90:93], a[152:155],  v99, v107 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[38:41], v[94:97], a[156:159],  v99, v107 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v148, s[0:3], s90 offen lds
	s_mov_b32 m0, s60
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[26:29], v[2:5], a[160:163],  v101, v106 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[26:29], v[6:9], a[164:167],  v101, v106 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[26:29], v[10:13], a[168:171], v101, v107 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[26:29], v[14:17], a[172:175], v101, v107 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_add_i32 s90, s55, 0xfffffa00
	buffer_load_dwordx4 v146, s[0:3], s91 offen lds
	s_mov_b32 m0, s61
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[42:45], v[82:85], a[160:163],  v101, v106 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[42:45], v[86:89], a[164:167],  v101, v106 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[42:45], v[90:93], a[168:171], v101, v107 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[42:45], v[94:97], a[172:175], v101, v107 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v147, s[0:3], s91 offen lds
	s_mov_b32 m0, s62
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[30:33], v[2:5], a[176:179], v101, v106 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[30:33], v[6:9], a[180:183], v101, v106 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[30:33], v[10:13], a[184:187], v101, v107 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[30:33], v[14:17], a[188:191], v101, v107 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v154, s[0:3], s91 offen lds
	s_mov_b32 m0, s63
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[46:49], v[82:85], a[176:179], v101, v106 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[46:49], v[86:89], a[180:183], v101, v106 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[46:49], v[90:93], a[184:187], v101, v107 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[46:49], v[94:97], a[188:191], v101, v107 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v148, s[0:3], s91 offen lds
	s_mov_b32 m0, s51
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[18:21], v[50:53], a[192:195],  v99, v103 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[2:5], v161 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[18:21], v[54:57], a[196:199],  v99, v103 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[6:9], v161 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[18:21], v[58:61], a[200:203],  v99, v104 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[10:13], v161 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[18:21], v[62:65], a[204:207],  v99, v104 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[14:17], v161 offset:6144

	s_add_i32 s91, s87, 0x580
	buffer_load_dwordx4 v150, s[40:43], s92 offen lds
	s_mov_b32 m0, s64
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[34:37], v[66:69], a[192:195],  v99, v103 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[82:85], v162 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[34:37], v[70:73], a[196:199],  v99, v103 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[86:89], v162 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[34:37], v[74:77], a[200:203],  v99, v104 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[90:93], v162 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[34:37], v[78:81], a[204:207],  v99, v104 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[94:97], v162 offset:6144

	s_nop 0
	buffer_load_dwordx4 v149, s[40:43], s92 offen lds
	s_mov_b32 m0, s65
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[22:25], v[50:53], a[208:211],  v99, v103 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[22:25], v[54:57], a[212:215],  v99, v103 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[22:25], v[58:61], a[216:219],  v99, v104 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[22:25], v[62:65], a[220:223],  v99, v104 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v151, s[40:43], s92 offen lds
	s_mov_b32 m0, s66
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[38:41], v[66:69], a[208:211],  v99, v103 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[38:41], v[70:73], a[212:215],  v99, v103 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[38:41], v[74:77], a[216:219],  v99, v104 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[38:41], v[78:81], a[220:223],  v99, v104 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v152, s[40:43], s92 offen lds
	s_mov_b32 m0, s53
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[26:29], v[50:53], a[224:227],  v101, v103 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[26:29], v[54:57], a[228:231],  v101, v103 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[26:29], v[58:61], a[232:235], v101, v104 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[26:29], v[62:65], a[236:239], v101, v104 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_add_i32 s92, s88, 0x580
	buffer_load_dwordx4 v150, s[40:43], s93 offen lds
	s_mov_b32 m0, s68
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[42:45], v[66:69], a[224:227],  v101, v103 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[42:45], v[70:73], a[228:231],  v101, v103 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[42:45], v[74:77], a[232:235], v101, v104 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[42:45], v[78:81], a[236:239], v101, v104 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v149, s[40:43], s93 offen lds
	s_mov_b32 m0, s69
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[30:33], v[50:53], a[240:243], v101, v103 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[30:33], v[54:57], a[244:247], v101, v103 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[30:33], v[58:61], a[248:251], v101, v104 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[30:33], v[62:65], a[252:255], v101, v104 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v151, s[40:43], s93 offen lds
	s_mov_b32 m0, s70
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[46:49], v[66:69], a[240:243], v101, v103 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[46:49], v[70:73], a[244:247], v101, v103 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[46:49], v[74:77], a[248:251], v101, v104 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[46:49], v[78:81], a[252:255], v101, v104 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v152, s[40:43], s93 offen lds
	s_waitcnt lgkmcnt(0)
	s_waitcnt vmcnt(16)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[112:115], v[2:5], a[28:31],  v109, v108 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v165 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[112:115], v[6:9], a[32:35],  v109, v108 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v165 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[112:115], v[10:13], a[36:39],  v109, v110 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v165 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[112:115], v[14:17], a[40:43],  v109, v110 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[62:65], v165 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[128:131], v[82:85], a[28:31],  v109, v108 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[66:69], v166 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[128:131], v[86:89], a[32:35],  v109, v108 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[70:73], v166 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[128:131], v[90:93], a[36:39],  v109, v110 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[74:77], v166 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[128:131], v[94:97], a[40:43],  v109, v110 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[78:81], v166 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[116:119], v[2:5], a[44:47],  v109, v108 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[116:119], v[6:9], a[48:51],  v109, v108 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[116:119], v[10:13], a[52:55],  v109, v110 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[116:119], v[14:17], a[56:59],  v109, v110 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[132:135], v[82:85], a[44:47],  v109, v108 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[132:135], v[86:89], a[48:51],  v109, v108 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[132:135], v[90:93], a[52:55],  v109, v110 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[132:135], v[94:97], a[56:59],  v109, v110 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[120:123], v[2:5], a[60:63],  v111, v108 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[120:123], v[6:9], a[64:67],  v111, v108 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[120:123], v[10:13], a[68:71], v111, v110 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[120:123], v[14:17], a[72:75], v111, v110 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[136:139], v[82:85], a[60:63],  v111, v108 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[136:139], v[86:89], a[64:67],  v111, v108 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[136:139], v[90:93], a[68:71], v111, v110 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[136:139], v[94:97], a[72:75], v111, v110 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[124:127], v[2:5], a[76:79], v111, v108 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[124:127], v[6:9], a[80:83], v111, v108 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[124:127], v[10:13], a[84:87], v111, v110 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[124:127], v[14:17], a[88:91], v111, v110 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[140:143], v[82:85], a[76:79], v111, v108 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[140:143], v[86:89], a[80:83], v111, v108 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[140:143], v[90:93], a[84:87], v111, v110 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[140:143], v[94:97], a[88:91], v111, v110 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[112:115], v[50:53], a[0:3], v109, v102 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[18:21], v169 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[112:115], v[54:57], a[4:7], v109, v102 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[22:25], v169 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[112:115], v[58:61], a[8:11], v109, v105 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[26:29], v169 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[112:115], v[62:65], a[12:15], v109, v105 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v169 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[128:131], v[66:69], a[0:3], v109, v102 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v170 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[128:131], v[70:73], a[4:7], v109, v102 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v170 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[128:131], v[74:77], a[8:11], v109, v105 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v170 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[128:131], v[78:81], a[12:15], v109, v105 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v170 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[116:119], v[50:53], a[16:19], v109, v102 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[116:119], v[54:57], a[20:23], v109, v102 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[116:119], v[58:61], a[24:27], v109, v105 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[116:119], v[62:65], a[92:95], v109, v105 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[132:135], v[66:69], a[16:19], v109, v102 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[132:135], v[70:73], a[20:23], v109, v102 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[132:135], v[74:77], a[24:27], v109, v105 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[132:135], v[78:81], a[92:95], v109, v105 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[120:123], v[50:53], a[96:99], v111, v102 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[120:123], v[54:57], a[100:103], v111, v102 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[120:123], v[58:61], a[104:107], v111, v105 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[120:123], v[62:65], a[108:111], v111, v105 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[136:139], v[66:69], a[96:99], v111, v102 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[136:139], v[70:73], a[100:103], v111, v102 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[136:139], v[74:77], a[104:107], v111, v105 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[136:139], v[78:81], a[108:111], v111, v105 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[124:127], v[50:53], a[112:115], v111, v102 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[124:127], v[54:57], a[116:119], v111, v102 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[124:127], v[58:61], a[120:123], v111, v105 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[124:127], v[62:65], a[124:127], v111, v105 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[140:143], v[66:69], a[112:115], v111, v102 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[140:143], v[70:73], a[116:119], v111, v102 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[140:143], v[74:77], a[120:123], v111, v105 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[140:143], v[78:81], a[124:127], v111, v105 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_waitcnt lgkmcnt(0)
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_barrier
	buffer_load_dword v104, v153, s[4:7], s90 offen
	buffer_load_dword v99, v153, s[8:11], s90 offen
	buffer_load_dword v106, v153, s[12:15], s90 offen
	buffer_load_dword v101, v153, s[16:19], s90 offen
	buffer_load_dword v109, v153, s[20:23], s90 offen
	buffer_load_dword v103, v153, s[24:27], s90 offen
	buffer_load_dword v111, v153, s[28:31], s90 offen
	buffer_load_dword v107, v153, s[36:39], s90 offen
	s_add_i32 s90, s86, 0x580
	s_mov_b32 m0, s49
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[18:21], v[2:5], a[128:131],  v98, v108 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[112:115], v155 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[18:21], v[6:9], a[132:135],  v98, v108 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[116:119], v155 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[18:21], v[10:13], a[136:139],  v98, v110 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[120:123], v155 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[18:21], v[14:17], a[140:143],  v98, v110 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[124:127], v155 offset:6144

	s_add_i32 s93, s89, 0x580
	buffer_load_dwordx4 v146, s[0:3], s90 offen lds
	s_mov_b32 m0, s71
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[34:37], v[82:85], a[128:131],  v98, v108 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[128:131], v156 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[34:37], v[86:89], a[132:135],  v98, v108 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[132:135], v156 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[34:37], v[90:93], a[136:139],  v98, v110 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[136:139], v156 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[34:37], v[94:97], a[140:143],  v98, v110 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[140:143], v156 offset:6144

	s_nop 0
	buffer_load_dwordx4 v147, s[0:3], s90 offen lds
	s_mov_b32 m0, s72
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[22:25], v[2:5], a[144:147],  v98, v108 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[22:25], v[6:9], a[148:151],  v98, v108 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[22:25], v[10:13], a[152:155],  v98, v110 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[22:25], v[14:17], a[156:159],  v98, v110 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v154, s[0:3], s90 offen lds
	s_mov_b32 m0, s73
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[38:41], v[82:85], a[144:147],  v98, v108 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[38:41], v[86:89], a[148:151],  v98, v108 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[38:41], v[90:93], a[152:155],  v98, v110 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[38:41], v[94:97], a[156:159],  v98, v110 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v148, s[0:3], s90 offen lds
	s_mov_b32 m0, s50
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[26:29], v[2:5], a[160:163],  v100, v108 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[26:29], v[6:9], a[164:167],  v100, v108 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[26:29], v[10:13], a[168:171], v100, v110 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[26:29], v[14:17], a[172:175], v100, v110 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_add_i32 s90, s55, 0xfffffb00
	buffer_load_dwordx4 v146, s[0:3], s91 offen lds
	s_mov_b32 m0, s74
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[42:45], v[82:85], a[160:163],  v100, v108 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[42:45], v[86:89], a[164:167],  v100, v108 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[42:45], v[90:93], a[168:171], v100, v110 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[42:45], v[94:97], a[172:175], v100, v110 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v147, s[0:3], s91 offen lds
	s_mov_b32 m0, s75
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[30:33], v[2:5], a[176:179], v100, v108 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[30:33], v[6:9], a[180:183], v100, v108 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[30:33], v[10:13], a[184:187], v100, v110 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[30:33], v[14:17], a[188:191], v100, v110 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v154, s[0:3], s91 offen lds
	s_mov_b32 m0, s76
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[46:49], v[82:85], a[176:179], v100, v108 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[46:49], v[86:89], a[180:183], v100, v108 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[46:49], v[90:93], a[184:187], v100, v110 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[46:49], v[94:97], a[188:191], v100, v110 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v148, s[0:3], s91 offen lds
	s_mov_b32 m0, s52
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[18:21], v[50:53], a[192:195],  v98, v102 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[2:5], v159 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[18:21], v[54:57], a[196:199],  v98, v102 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[6:9], v159 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[18:21], v[58:61], a[200:203],  v98, v105 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[10:13], v159 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[18:21], v[62:65], a[204:207],  v98, v105 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[14:17], v159 offset:6144

	s_add_i32 s91, s87, 0x600
	buffer_load_dwordx4 v150, s[40:43], s92 offen lds
	s_mov_b32 m0, s77
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[34:37], v[66:69], a[192:195],  v98, v102 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[82:85], v160 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[34:37], v[70:73], a[196:199],  v98, v102 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[86:89], v160 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[34:37], v[74:77], a[200:203],  v98, v105 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[90:93], v160 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[34:37], v[78:81], a[204:207],  v98, v105 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[94:97], v160 offset:6144

	s_nop 0
	buffer_load_dwordx4 v149, s[40:43], s92 offen lds
	s_mov_b32 m0, s78
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[22:25], v[50:53], a[208:211],  v98, v102 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[22:25], v[54:57], a[212:215],  v98, v102 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[22:25], v[58:61], a[216:219],  v98, v105 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[22:25], v[62:65], a[220:223],  v98, v105 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v151, s[40:43], s92 offen lds
	s_mov_b32 m0, s79
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[38:41], v[66:69], a[208:211],  v98, v102 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[38:41], v[70:73], a[212:215],  v98, v102 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[38:41], v[74:77], a[216:219],  v98, v105 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[38:41], v[78:81], a[220:223],  v98, v105 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v152, s[40:43], s92 offen lds
	s_mov_b32 m0, s54
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[26:29], v[50:53], a[224:227],  v100, v102 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[26:29], v[54:57], a[228:231],  v100, v102 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[26:29], v[58:61], a[232:235], v100, v105 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[26:29], v[62:65], a[236:239], v100, v105 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_add_i32 s92, s88, 0x600
	buffer_load_dwordx4 v150, s[40:43], s93 offen lds
	s_mov_b32 m0, s80
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[42:45], v[66:69], a[224:227],  v100, v102 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[42:45], v[70:73], a[228:231],  v100, v102 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[42:45], v[74:77], a[232:235], v100, v105 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[42:45], v[78:81], a[236:239], v100, v105 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v149, s[40:43], s93 offen lds
	s_mov_b32 m0, s81
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[30:33], v[50:53], a[240:243], v100, v102 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[30:33], v[54:57], a[244:247], v100, v102 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[30:33], v[58:61], a[248:251], v100, v105 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[30:33], v[62:65], a[252:255], v100, v105 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v151, s[40:43], s93 offen lds
	s_mov_b32 m0, s82
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[46:49], v[66:69], a[240:243], v100, v102 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[46:49], v[70:73], a[244:247], v100, v102 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[46:49], v[74:77], a[248:251], v100, v105 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[46:49], v[78:81], a[252:255], v100, v105 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v152, s[40:43], s93 offen lds
	s_waitcnt lgkmcnt(0)
	s_waitcnt vmcnt(16)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[112:115], v[2:5], a[28:31],  v104, v109 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v163 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[112:115], v[6:9], a[32:35],  v104, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v163 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[112:115], v[10:13], a[36:39],  v104, v111 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v163 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[112:115], v[14:17], a[40:43],  v104, v111 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[62:65], v163 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[128:131], v[82:85], a[28:31],  v104, v109 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[66:69], v164 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[128:131], v[86:89], a[32:35],  v104, v109 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[70:73], v164 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[128:131], v[90:93], a[36:39],  v104, v111 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[74:77], v164 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[128:131], v[94:97], a[40:43],  v104, v111 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[78:81], v164 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[116:119], v[2:5], a[44:47],  v104, v109 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[116:119], v[6:9], a[48:51],  v104, v109 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[116:119], v[10:13], a[52:55],  v104, v111 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[116:119], v[14:17], a[56:59],  v104, v111 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[132:135], v[82:85], a[44:47],  v104, v109 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[132:135], v[86:89], a[48:51],  v104, v109 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[132:135], v[90:93], a[52:55],  v104, v111 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[132:135], v[94:97], a[56:59],  v104, v111 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[120:123], v[2:5], a[60:63],  v106, v109 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[120:123], v[6:9], a[64:67],  v106, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[120:123], v[10:13], a[68:71], v106, v111 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[120:123], v[14:17], a[72:75], v106, v111 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[136:139], v[82:85], a[60:63],  v106, v109 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[136:139], v[86:89], a[64:67],  v106, v109 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[136:139], v[90:93], a[68:71], v106, v111 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[136:139], v[94:97], a[72:75], v106, v111 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[124:127], v[2:5], a[76:79], v106, v109 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[124:127], v[6:9], a[80:83], v106, v109 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[124:127], v[10:13], a[84:87], v106, v111 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[124:127], v[14:17], a[88:91], v106, v111 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[140:143], v[82:85], a[76:79], v106, v109 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[140:143], v[86:89], a[80:83], v106, v109 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[140:143], v[90:93], a[84:87], v106, v111 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[140:143], v[94:97], a[88:91], v106, v111 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[112:115], v[50:53], a[0:3], v104, v103 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[18:21], v167 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[112:115], v[54:57], a[4:7], v104, v103 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[22:25], v167 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[112:115], v[58:61], a[8:11], v104, v107 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[26:29], v167 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[112:115], v[62:65], a[12:15], v104, v107 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v167 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[128:131], v[66:69], a[0:3], v104, v103 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v168 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[128:131], v[70:73], a[4:7], v104, v103 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v168 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[128:131], v[74:77], a[8:11], v104, v107 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v168 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[128:131], v[78:81], a[12:15], v104, v107 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v168 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[116:119], v[50:53], a[16:19], v104, v103 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[116:119], v[54:57], a[20:23], v104, v103 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[116:119], v[58:61], a[24:27], v104, v107 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[116:119], v[62:65], a[92:95], v104, v107 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[132:135], v[66:69], a[16:19], v104, v103 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[132:135], v[70:73], a[20:23], v104, v103 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[132:135], v[74:77], a[24:27], v104, v107 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[132:135], v[78:81], a[92:95], v104, v107 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[120:123], v[50:53], a[96:99], v106, v103 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[120:123], v[54:57], a[100:103], v106, v103 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[120:123], v[58:61], a[104:107], v106, v107 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[120:123], v[62:65], a[108:111], v106, v107 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[136:139], v[66:69], a[96:99], v106, v103 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[136:139], v[70:73], a[100:103], v106, v103 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[136:139], v[74:77], a[104:107], v106, v107 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[136:139], v[78:81], a[108:111], v106, v107 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[124:127], v[50:53], a[112:115], v106, v103 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[124:127], v[54:57], a[116:119], v106, v103 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[124:127], v[58:61], a[120:123], v106, v107 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[124:127], v[62:65], a[124:127], v106, v107 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[140:143], v[66:69], a[112:115], v106, v103 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[140:143], v[70:73], a[116:119], v106, v103 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[140:143], v[74:77], a[120:123], v106, v107 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[140:143], v[78:81], a[124:127], v106, v107 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_waitcnt lgkmcnt(0)
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_barrier
	buffer_load_dword v98, v153, s[4:7], s90 offen
	buffer_load_dword v104, v153, s[8:11], s90 offen
	buffer_load_dword v100, v153, s[12:15], s90 offen
	buffer_load_dword v105, v153, s[16:19], s90 offen
	buffer_load_dword v110, v153, s[20:23], s90 offen
	buffer_load_dword v106, v153, s[24:27], s90 offen
	buffer_load_dword v112, v153, s[28:31], s90 offen
	buffer_load_dword v108, v153, s[36:39], s90 offen
	s_add_i32 s90, s86, 0x600
	s_mov_b32 m0, s48
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[18:21], v[2:5], a[128:131],  v99, v109 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[114:117], v157 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[18:21], v[6:9], a[132:135],  v99, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[118:121], v157 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[18:21], v[10:13], a[136:139],  v99, v111 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[122:125], v157 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[18:21], v[14:17], a[140:143],  v99, v111 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[126:129], v157 offset:6144

	s_add_i32 s93, s89, 0x600
	buffer_load_dwordx4 v146, s[0:3], s90 offen lds
	s_mov_b32 m0, s56
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[34:37], v[82:85], a[128:131],  v99, v109 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[130:133], v158 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[34:37], v[86:89], a[132:135],  v99, v109 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[134:137], v158 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[34:37], v[90:93], a[136:139],  v99, v111 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[138:141], v158 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[34:37], v[94:97], a[140:143],  v99, v111 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[142:145], v158 offset:6144

	s_nop 0
	buffer_load_dwordx4 v147, s[0:3], s90 offen lds
	s_mov_b32 m0, s57
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[22:25], v[2:5], a[144:147],  v99, v109 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[22:25], v[6:9], a[148:151],  v99, v109 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[22:25], v[10:13], a[152:155],  v99, v111 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[22:25], v[14:17], a[156:159],  v99, v111 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v154, s[0:3], s90 offen lds
	s_mov_b32 m0, s58
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[38:41], v[82:85], a[144:147],  v99, v109 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[38:41], v[86:89], a[148:151],  v99, v109 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[38:41], v[90:93], a[152:155],  v99, v111 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[38:41], v[94:97], a[156:159],  v99, v111 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v148, s[0:3], s90 offen lds
	s_mov_b32 m0, s60
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[26:29], v[2:5], a[160:163],  v101, v109 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[26:29], v[6:9], a[164:167],  v101, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[26:29], v[10:13], a[168:171], v101, v111 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[26:29], v[14:17], a[172:175], v101, v111 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_add_i32 s90, s55, 0xfffffc00
	buffer_load_dwordx4 v146, s[0:3], s91 offen lds
	s_mov_b32 m0, s61
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[42:45], v[82:85], a[160:163],  v101, v109 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[42:45], v[86:89], a[164:167],  v101, v109 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[42:45], v[90:93], a[168:171], v101, v111 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[42:45], v[94:97], a[172:175], v101, v111 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v147, s[0:3], s91 offen lds
	s_mov_b32 m0, s62
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[30:33], v[2:5], a[176:179], v101, v109 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[30:33], v[6:9], a[180:183], v101, v109 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[30:33], v[10:13], a[184:187], v101, v111 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[30:33], v[14:17], a[188:191], v101, v111 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v154, s[0:3], s91 offen lds
	s_mov_b32 m0, s63
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[46:49], v[82:85], a[176:179], v101, v109 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[46:49], v[86:89], a[180:183], v101, v109 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[46:49], v[90:93], a[184:187], v101, v111 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[46:49], v[94:97], a[188:191], v101, v111 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v148, s[0:3], s91 offen lds
	s_mov_b32 m0, s51
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[18:21], v[50:53], a[192:195],  v99, v103 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[2:5], v161 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[18:21], v[54:57], a[196:199],  v99, v103 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[6:9], v161 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[18:21], v[58:61], a[200:203],  v99, v107 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[10:13], v161 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[18:21], v[62:65], a[204:207],  v99, v107 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[14:17], v161 offset:6144

	s_add_i32 s91, s87, 0x680
	buffer_load_dwordx4 v150, s[40:43], s92 offen lds
	s_mov_b32 m0, s64
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[34:37], v[66:69], a[192:195],  v99, v103 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[82:85], v162 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[34:37], v[70:73], a[196:199],  v99, v103 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[86:89], v162 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[34:37], v[74:77], a[200:203],  v99, v107 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[90:93], v162 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[34:37], v[78:81], a[204:207],  v99, v107 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[94:97], v162 offset:6144

	s_nop 0
	buffer_load_dwordx4 v149, s[40:43], s92 offen lds
	s_mov_b32 m0, s65
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[22:25], v[50:53], a[208:211],  v99, v103 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[22:25], v[54:57], a[212:215],  v99, v103 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[22:25], v[58:61], a[216:219],  v99, v107 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[22:25], v[62:65], a[220:223],  v99, v107 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v151, s[40:43], s92 offen lds
	s_mov_b32 m0, s66
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[38:41], v[66:69], a[208:211],  v99, v103 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[38:41], v[70:73], a[212:215],  v99, v103 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[38:41], v[74:77], a[216:219],  v99, v107 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[38:41], v[78:81], a[220:223],  v99, v107 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v152, s[40:43], s92 offen lds
	s_mov_b32 m0, s53
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[26:29], v[50:53], a[224:227],  v101, v103 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[26:29], v[54:57], a[228:231],  v101, v103 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[26:29], v[58:61], a[232:235], v101, v107 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[26:29], v[62:65], a[236:239], v101, v107 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_add_i32 s92, s88, 0x680
	buffer_load_dwordx4 v150, s[40:43], s93 offen lds
	s_mov_b32 m0, s68
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[42:45], v[66:69], a[224:227],  v101, v103 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[42:45], v[70:73], a[228:231],  v101, v103 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[42:45], v[74:77], a[232:235], v101, v107 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[42:45], v[78:81], a[236:239], v101, v107 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v149, s[40:43], s93 offen lds
	s_mov_b32 m0, s69
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[30:33], v[50:53], a[240:243], v101, v103 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[30:33], v[54:57], a[244:247], v101, v103 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[30:33], v[58:61], a[248:251], v101, v107 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[30:33], v[62:65], a[252:255], v101, v107 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v151, s[40:43], s93 offen lds
	s_mov_b32 m0, s70
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[46:49], v[66:69], a[240:243], v101, v103 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[46:49], v[70:73], a[244:247], v101, v103 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[46:49], v[74:77], a[248:251], v101, v107 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[46:49], v[78:81], a[252:255], v101, v107 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v152, s[40:43], s93 offen lds
	s_waitcnt lgkmcnt(0)
	s_waitcnt vmcnt(16)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[114:117], v[2:5], a[28:31],  v98, v110 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v165 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[114:117], v[6:9], a[32:35],  v98, v110 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v165 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[114:117], v[10:13], a[36:39],  v98, v112 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v165 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[114:117], v[14:17], a[40:43],  v98, v112 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[62:65], v165 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[130:133], v[82:85], a[28:31],  v98, v110 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[66:69], v166 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[130:133], v[86:89], a[32:35],  v98, v110 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[70:73], v166 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[130:133], v[90:93], a[36:39],  v98, v112 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[74:77], v166 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[130:133], v[94:97], a[40:43],  v98, v112 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[78:81], v166 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[118:121], v[2:5], a[44:47],  v98, v110 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[118:121], v[6:9], a[48:51],  v98, v110 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[118:121], v[10:13], a[52:55],  v98, v112 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[118:121], v[14:17], a[56:59],  v98, v112 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[134:137], v[82:85], a[44:47],  v98, v110 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[134:137], v[86:89], a[48:51],  v98, v110 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[134:137], v[90:93], a[52:55],  v98, v112 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[134:137], v[94:97], a[56:59],  v98, v112 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[122:125], v[2:5], a[60:63],  v100, v110 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[122:125], v[6:9], a[64:67],  v100, v110 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[122:125], v[10:13], a[68:71], v100, v112 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[122:125], v[14:17], a[72:75], v100, v112 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[138:141], v[82:85], a[60:63],  v100, v110 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[138:141], v[86:89], a[64:67],  v100, v110 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[138:141], v[90:93], a[68:71], v100, v112 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[138:141], v[94:97], a[72:75], v100, v112 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[126:129], v[2:5], a[76:79], v100, v110 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[126:129], v[6:9], a[80:83], v100, v110 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[126:129], v[10:13], a[84:87], v100, v112 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[126:129], v[14:17], a[88:91], v100, v112 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[142:145], v[82:85], a[76:79], v100, v110 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[142:145], v[86:89], a[80:83], v100, v110 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[142:145], v[90:93], a[84:87], v100, v112 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[142:145], v[94:97], a[88:91], v100, v112 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[114:117], v[50:53], a[0:3], v98, v106 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[18:21], v169 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[114:117], v[54:57], a[4:7], v98, v106 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[22:25], v169 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[114:117], v[58:61], a[8:11], v98, v108 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[26:29], v169 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[114:117], v[62:65], a[12:15], v98, v108 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v169 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[130:133], v[66:69], a[0:3], v98, v106 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v170 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[130:133], v[70:73], a[4:7], v98, v106 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v170 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[130:133], v[74:77], a[8:11], v98, v108 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v170 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[130:133], v[78:81], a[12:15], v98, v108 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v170 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[118:121], v[50:53], a[16:19], v98, v106 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[118:121], v[54:57], a[20:23], v98, v106 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[118:121], v[58:61], a[24:27], v98, v108 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[118:121], v[62:65], a[92:95], v98, v108 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[134:137], v[66:69], a[16:19], v98, v106 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[134:137], v[70:73], a[20:23], v98, v106 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[134:137], v[74:77], a[24:27], v98, v108 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[134:137], v[78:81], a[92:95], v98, v108 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[122:125], v[50:53], a[96:99], v100, v106 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[122:125], v[54:57], a[100:103], v100, v106 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[122:125], v[58:61], a[104:107], v100, v108 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[122:125], v[62:65], a[108:111], v100, v108 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[138:141], v[66:69], a[96:99], v100, v106 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[138:141], v[70:73], a[100:103], v100, v106 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[138:141], v[74:77], a[104:107], v100, v108 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[138:141], v[78:81], a[108:111], v100, v108 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[126:129], v[50:53], a[112:115], v100, v106 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[126:129], v[54:57], a[116:119], v100, v106 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[126:129], v[58:61], a[120:123], v100, v108 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[126:129], v[62:65], a[124:127], v100, v108 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[142:145], v[66:69], a[112:115], v100, v106 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[142:145], v[70:73], a[116:119], v100, v106 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[142:145], v[74:77], a[120:123], v100, v108 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[142:145], v[78:81], a[124:127], v100, v108 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_waitcnt lgkmcnt(0)
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_barrier
	buffer_load_dword v107, v153, s[4:7], s90 offen
	buffer_load_dword v98, v153, s[8:11], s90 offen
	buffer_load_dword v109, v153, s[12:15], s90 offen
	buffer_load_dword v99, v153, s[16:19], s90 offen
	buffer_load_dword v102, v153, s[20:23], s90 offen
	buffer_load_dword v100, v153, s[24:27], s90 offen
	buffer_load_dword v103, v153, s[28:31], s90 offen
	buffer_load_dword v101, v153, s[36:39], s90 offen
	s_add_i32 s90, s86, 0x680
	s_mov_b32 m0, s49
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[18:21], v[2:5], a[128:131],  v104, v110 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[114:117], v155 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[18:21], v[6:9], a[132:135],  v104, v110 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[118:121], v155 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[18:21], v[10:13], a[136:139],  v104, v112 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[122:125], v155 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[18:21], v[14:17], a[140:143],  v104, v112 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[126:129], v155 offset:6144

	s_add_i32 s93, s89, 0x680
	buffer_load_dwordx4 v146, s[0:3], s90 offen lds
	s_mov_b32 m0, s71
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[34:37], v[82:85], a[128:131],  v104, v110 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[130:133], v156 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[34:37], v[86:89], a[132:135],  v104, v110 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[134:137], v156 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[34:37], v[90:93], a[136:139],  v104, v112 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[138:141], v156 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[34:37], v[94:97], a[140:143],  v104, v112 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[142:145], v156 offset:6144

	s_nop 0
	buffer_load_dwordx4 v147, s[0:3], s90 offen lds
	s_mov_b32 m0, s72
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[22:25], v[2:5], a[144:147],  v104, v110 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[22:25], v[6:9], a[148:151],  v104, v110 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[22:25], v[10:13], a[152:155],  v104, v112 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[22:25], v[14:17], a[156:159],  v104, v112 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v154, s[0:3], s90 offen lds
	s_mov_b32 m0, s73
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[38:41], v[82:85], a[144:147],  v104, v110 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[38:41], v[86:89], a[148:151],  v104, v110 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[38:41], v[90:93], a[152:155],  v104, v112 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[38:41], v[94:97], a[156:159],  v104, v112 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v148, s[0:3], s90 offen lds
	s_mov_b32 m0, s50
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[26:29], v[2:5], a[160:163],  v105, v110 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[26:29], v[6:9], a[164:167],  v105, v110 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[26:29], v[10:13], a[168:171], v105, v112 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[26:29], v[14:17], a[172:175], v105, v112 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_add_i32 s90, s55, 0xfffffd00
	buffer_load_dwordx4 v146, s[0:3], s91 offen lds
	s_mov_b32 m0, s74
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[42:45], v[82:85], a[160:163],  v105, v110 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[42:45], v[86:89], a[164:167],  v105, v110 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[42:45], v[90:93], a[168:171], v105, v112 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[42:45], v[94:97], a[172:175], v105, v112 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v147, s[0:3], s91 offen lds
	s_mov_b32 m0, s75
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[30:33], v[2:5], a[176:179], v105, v110 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[30:33], v[6:9], a[180:183], v105, v110 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[30:33], v[10:13], a[184:187], v105, v112 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[30:33], v[14:17], a[188:191], v105, v112 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v154, s[0:3], s91 offen lds
	s_mov_b32 m0, s76
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[46:49], v[82:85], a[176:179], v105, v110 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[46:49], v[86:89], a[180:183], v105, v110 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[46:49], v[90:93], a[184:187], v105, v112 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[46:49], v[94:97], a[188:191], v105, v112 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v148, s[0:3], s91 offen lds
	s_mov_b32 m0, s52
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[18:21], v[50:53], a[192:195],  v104, v106 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[2:5], v159 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[18:21], v[54:57], a[196:199],  v104, v106 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[6:9], v159 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[18:21], v[58:61], a[200:203],  v104, v108 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[10:13], v159 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[18:21], v[62:65], a[204:207],  v104, v108 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[14:17], v159 offset:6144

	s_add_i32 s91, s87, 0x700
	buffer_load_dwordx4 v150, s[40:43], s92 offen lds
	s_mov_b32 m0, s77
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[34:37], v[66:69], a[192:195],  v104, v106 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[82:85], v160 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[34:37], v[70:73], a[196:199],  v104, v106 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[86:89], v160 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[34:37], v[74:77], a[200:203],  v104, v108 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[90:93], v160 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[34:37], v[78:81], a[204:207],  v104, v108 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[94:97], v160 offset:6144

	s_addk_i32 s87, 0x780
	buffer_load_dwordx4 v149, s[40:43], s92 offen lds
	s_mov_b32 m0, s78
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[22:25], v[50:53], a[208:211],  v104, v106 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[22:25], v[54:57], a[212:215],  v104, v106 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[22:25], v[58:61], a[216:219],  v104, v108 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[22:25], v[62:65], a[220:223],  v104, v108 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v151, s[40:43], s92 offen lds
	s_mov_b32 m0, s79
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[38:41], v[66:69], a[208:211],  v104, v106 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[38:41], v[70:73], a[212:215],  v104, v106 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[38:41], v[74:77], a[216:219],  v104, v108 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[38:41], v[78:81], a[220:223],  v104, v108 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v152, s[40:43], s92 offen lds
	s_mov_b32 m0, s54
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[26:29], v[50:53], a[224:227],  v105, v106 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[26:29], v[54:57], a[228:231],  v105, v106 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[26:29], v[58:61], a[232:235], v105, v108 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[26:29], v[62:65], a[236:239], v105, v108 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_add_i32 s92, s88, 0x700
	buffer_load_dwordx4 v150, s[40:43], s93 offen lds
	s_mov_b32 m0, s80
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[42:45], v[66:69], a[224:227],  v105, v106 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[42:45], v[70:73], a[228:231],  v105, v106 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[42:45], v[74:77], a[232:235], v105, v108 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[42:45], v[78:81], a[236:239], v105, v108 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_addk_i32 s88, 0x780
	buffer_load_dwordx4 v149, s[40:43], s93 offen lds
	s_mov_b32 m0, s81
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[30:33], v[50:53], a[240:243], v105, v106 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[30:33], v[54:57], a[244:247], v105, v106 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[30:33], v[58:61], a[248:251], v105, v108 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[30:33], v[62:65], a[252:255], v105, v108 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v151, s[40:43], s93 offen lds
	s_mov_b32 m0, s82
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[46:49], v[66:69], a[240:243], v105, v106 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[46:49], v[70:73], a[244:247], v105, v106 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[46:49], v[74:77], a[248:251], v105, v108 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[46:49], v[78:81], a[252:255], v105, v108 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v152, s[40:43], s93 offen lds
	s_waitcnt lgkmcnt(0)
	s_waitcnt vmcnt(16)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[114:117], v[2:5], a[28:31],  v107, v102 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v163 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[114:117], v[6:9], a[32:35],  v107, v102 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v163 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[114:117], v[10:13], a[36:39],  v107, v103 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v163 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[114:117], v[14:17], a[40:43],  v107, v103 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[62:65], v163 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[130:133], v[82:85], a[28:31],  v107, v102 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[66:69], v164 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[130:133], v[86:89], a[32:35],  v107, v102 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[70:73], v164 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[130:133], v[90:93], a[36:39],  v107, v103 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[74:77], v164 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[130:133], v[94:97], a[40:43],  v107, v103 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[78:81], v164 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[118:121], v[2:5], a[44:47],  v107, v102 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[118:121], v[6:9], a[48:51],  v107, v102 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[118:121], v[10:13], a[52:55],  v107, v103 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[118:121], v[14:17], a[56:59],  v107, v103 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[134:137], v[82:85], a[44:47],  v107, v102 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[134:137], v[86:89], a[48:51],  v107, v102 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[134:137], v[90:93], a[52:55],  v107, v103 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[134:137], v[94:97], a[56:59],  v107, v103 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[122:125], v[2:5], a[60:63],  v109, v102 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[122:125], v[6:9], a[64:67],  v109, v102 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[122:125], v[10:13], a[68:71], v109, v103 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[122:125], v[14:17], a[72:75], v109, v103 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[138:141], v[82:85], a[60:63],  v109, v102 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[138:141], v[86:89], a[64:67],  v109, v102 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[138:141], v[90:93], a[68:71], v109, v103 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[138:141], v[94:97], a[72:75], v109, v103 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[126:129], v[2:5], a[76:79], v109, v102 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[126:129], v[6:9], a[80:83], v109, v102 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[126:129], v[10:13], a[84:87], v109, v103 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[126:129], v[14:17], a[88:91], v109, v103 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[142:145], v[82:85], a[76:79], v109, v102 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[142:145], v[86:89], a[80:83], v109, v102 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[142:145], v[90:93], a[84:87], v109, v103 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[142:145], v[94:97], a[88:91], v109, v103 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[114:117], v[50:53], a[0:3], v107, v100 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[18:21], v167 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[114:117], v[54:57], a[4:7], v107, v100 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[22:25], v167 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[114:117], v[58:61], a[8:11], v107, v101 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[26:29], v167 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[114:117], v[62:65], a[12:15], v107, v101 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v167 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[130:133], v[66:69], a[0:3], v107, v100 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v168 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[130:133], v[70:73], a[4:7], v107, v100 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v168 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[130:133], v[74:77], a[8:11], v107, v101 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v168 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[130:133], v[78:81], a[12:15], v107, v101 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v168 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[118:121], v[50:53], a[16:19], v107, v100 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[118:121], v[54:57], a[20:23], v107, v100 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[118:121], v[58:61], a[24:27], v107, v101 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[118:121], v[62:65], a[92:95], v107, v101 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[134:137], v[66:69], a[16:19], v107, v100 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[134:137], v[70:73], a[20:23], v107, v100 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[134:137], v[74:77], a[24:27], v107, v101 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[134:137], v[78:81], a[92:95], v107, v101 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[122:125], v[50:53], a[96:99], v109, v100 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[122:125], v[54:57], a[100:103], v109, v100 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[122:125], v[58:61], a[104:107], v109, v101 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[122:125], v[62:65], a[108:111], v109, v101 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[138:141], v[66:69], a[96:99], v109, v100 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[138:141], v[70:73], a[100:103], v109, v100 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[138:141], v[74:77], a[104:107], v109, v101 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[138:141], v[78:81], a[108:111], v109, v101 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[126:129], v[50:53], a[112:115], v109, v100 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[126:129], v[54:57], a[116:119], v109, v100 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[126:129], v[58:61], a[120:123], v109, v101 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[126:129], v[62:65], a[124:127], v109, v101 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[142:145], v[66:69], a[112:115], v109, v100 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[142:145], v[70:73], a[116:119], v109, v100 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[142:145], v[74:77], a[120:123], v109, v101 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[142:145], v[78:81], a[124:127], v109, v101 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_waitcnt lgkmcnt(0)
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_barrier
	buffer_load_dword v142, v153, s[4:7], s90 offen
	buffer_load_dword v104, v153, s[8:11], s90 offen
	buffer_load_dword v143, v153, s[12:15], s90 offen
	buffer_load_dword v105, v153, s[16:19], s90 offen
	buffer_load_dword v108, v153, s[20:23], s90 offen
	buffer_load_dword v106, v153, s[24:27], s90 offen
	buffer_load_dword v109, v153, s[28:31], s90 offen
	buffer_load_dword v107, v153, s[36:39], s90 offen
	s_add_i32 s90, s86, 0x700
	s_mov_b32 m0, s48
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[18:21], v[2:5], a[128:131],  v98, v102 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[110:113], v157 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[18:21], v[6:9], a[132:135],  v98, v102 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[114:117], v157 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[18:21], v[10:13], a[136:139],  v98, v103 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[118:121], v157 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[18:21], v[14:17], a[140:143],  v98, v103 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[122:125], v157 offset:6144

	s_add_i32 s93, s89, 0x700
	buffer_load_dwordx4 v146, s[0:3], s90 offen lds
	s_mov_b32 m0, s56
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[34:37], v[82:85], a[128:131],  v98, v102 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[126:129], v158 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[34:37], v[86:89], a[132:135],  v98, v102 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[130:133], v158 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[34:37], v[90:93], a[136:139],  v98, v103 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[134:137], v158 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[34:37], v[94:97], a[140:143],  v98, v103 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[138:141], v158 offset:6144

	s_addk_i32 s86, 0x780
	buffer_load_dwordx4 v147, s[0:3], s90 offen lds
	s_mov_b32 m0, s57
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[22:25], v[2:5], a[144:147],  v98, v102 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[22:25], v[6:9], a[148:151],  v98, v102 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[22:25], v[10:13], a[152:155],  v98, v103 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[22:25], v[14:17], a[156:159],  v98, v103 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_addk_i32 s89, 0x780
	buffer_load_dwordx4 v154, s[0:3], s90 offen lds
	s_mov_b32 m0, s58
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[38:41], v[82:85], a[144:147],  v98, v102 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[38:41], v[86:89], a[148:151],  v98, v102 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[38:41], v[90:93], a[152:155],  v98, v103 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[38:41], v[94:97], a[156:159],  v98, v103 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v148, s[0:3], s90 offen lds
	s_mov_b32 m0, s60
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[26:29], v[2:5], a[160:163],  v99, v102 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[26:29], v[6:9], a[164:167],  v99, v102 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[26:29], v[10:13], a[168:171], v99, v103 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[26:29], v[14:17], a[172:175], v99, v103 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_add_i32 s90, s85, -1
	buffer_load_dwordx4 v146, s[0:3], s91 offen lds
	s_mov_b32 m0, s61
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[42:45], v[82:85], a[160:163],  v99, v102 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[42:45], v[86:89], a[164:167],  v99, v102 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[42:45], v[90:93], a[168:171], v99, v103 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[42:45], v[94:97], a[172:175], v99, v103 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v147, s[0:3], s91 offen lds
	s_mov_b32 m0, s62
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[30:33], v[2:5], a[176:179], v99, v102 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[30:33], v[6:9], a[180:183], v99, v102 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[30:33], v[10:13], a[184:187], v99, v103 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[30:33], v[14:17], a[188:191], v99, v103 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v154, s[0:3], s91 offen lds
	s_mov_b32 m0, s63
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[46:49], v[82:85], a[176:179], v99, v102 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[46:49], v[86:89], a[180:183], v99, v102 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[46:49], v[90:93], a[184:187], v99, v103 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[46:49], v[94:97], a[188:191], v99, v103 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v148, s[0:3], s91 offen lds
	s_mov_b32 m0, s51
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[18:21], v[50:53], a[192:195],  v98, v100 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[2:5], v161 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[18:21], v[54:57], a[196:199],  v98, v100 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[6:9], v161 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[18:21], v[58:61], a[200:203],  v98, v101 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[10:13], v161 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[18:21], v[62:65], a[204:207],  v98, v101 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[14:17], v161 offset:6144

	s_add_i32 s91, s55, 0xfffffe00
	buffer_load_dwordx4 v150, s[40:43], s92 offen lds
	s_mov_b32 m0, s64
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[34:37], v[66:69], a[192:195],  v98, v100 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[82:85], v162 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[34:37], v[70:73], a[196:199],  v98, v100 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[86:89], v162 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[34:37], v[74:77], a[200:203],  v98, v101 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[90:93], v162 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[34:37], v[78:81], a[204:207],  v98, v101 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[94:97], v162 offset:6144

	s_nop 0
	buffer_load_dwordx4 v149, s[40:43], s92 offen lds
	s_mov_b32 m0, s65
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[22:25], v[50:53], a[208:211],  v98, v100 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[22:25], v[54:57], a[212:215],  v98, v100 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[22:25], v[58:61], a[216:219],  v98, v101 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[22:25], v[62:65], a[220:223],  v98, v101 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v151, s[40:43], s92 offen lds
	s_mov_b32 m0, s66
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[38:41], v[66:69], a[208:211],  v98, v100 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[38:41], v[70:73], a[212:215],  v98, v100 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[38:41], v[74:77], a[216:219],  v98, v101 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[38:41], v[78:81], a[220:223],  v98, v101 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v152, s[40:43], s92 offen lds
	s_mov_b32 m0, s53
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[26:29], v[50:53], a[224:227],  v99, v100 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[26:29], v[54:57], a[228:231],  v99, v100 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[26:29], v[58:61], a[232:235], v99, v101 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[26:29], v[62:65], a[236:239], v99, v101 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v150, s[40:43], s93 offen lds
	s_mov_b32 m0, s68
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[42:45], v[66:69], a[224:227],  v99, v100 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[42:45], v[70:73], a[228:231],  v99, v100 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[42:45], v[74:77], a[232:235], v99, v101 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[42:45], v[78:81], a[236:239], v99, v101 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v149, s[40:43], s93 offen lds
	s_mov_b32 m0, s69
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[30:33], v[50:53], a[240:243], v99, v100 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[30:33], v[54:57], a[244:247], v99, v100 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[30:33], v[58:61], a[248:251], v99, v101 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[30:33], v[62:65], a[252:255], v99, v101 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v151, s[40:43], s93 offen lds
	s_mov_b32 m0, s70
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[46:49], v[66:69], a[240:243], v99, v100 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[46:49], v[70:73], a[244:247], v99, v100 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[46:49], v[74:77], a[248:251], v99, v101 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[46:49], v[78:81], a[252:255], v99, v101 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v152, s[40:43], s93 offen lds
	s_waitcnt lgkmcnt(0)
	s_waitcnt vmcnt(16)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[110:113], v[2:5], a[28:31],  v142, v108 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v165 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[110:113], v[6:9], a[32:35],  v142, v108 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v165 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[110:113], v[10:13], a[36:39],  v142, v109 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v165 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[110:113], v[14:17], a[40:43],  v142, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[62:65], v165 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[126:129], v[82:85], a[28:31],  v142, v108 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[66:69], v166 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[126:129], v[86:89], a[32:35],  v142, v108 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[70:73], v166 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[126:129], v[90:93], a[36:39],  v142, v109 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[74:77], v166 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[126:129], v[94:97], a[40:43],  v142, v109 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[78:81], v166 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[114:117], v[2:5], a[44:47],  v142, v108 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[114:117], v[6:9], a[48:51],  v142, v108 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[114:117], v[10:13], a[52:55],  v142, v109 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[114:117], v[14:17], a[56:59],  v142, v109 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[130:133], v[82:85], a[44:47],  v142, v108 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[130:133], v[86:89], a[48:51],  v142, v108 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[130:133], v[90:93], a[52:55],  v142, v109 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[130:133], v[94:97], a[56:59],  v142, v109 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[118:121], v[2:5], a[60:63],  v143, v108 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[118:121], v[6:9], a[64:67],  v143, v108 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[118:121], v[10:13], a[68:71], v143, v109 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[118:121], v[14:17], a[72:75], v143, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[134:137], v[82:85], a[60:63],  v143, v108 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[134:137], v[86:89], a[64:67],  v143, v108 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[134:137], v[90:93], a[68:71], v143, v109 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[134:137], v[94:97], a[72:75], v143, v109 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[122:125], v[2:5], a[76:79], v143, v108 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[122:125], v[6:9], a[80:83], v143, v108 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[122:125], v[10:13], a[84:87], v143, v109 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[122:125], v[14:17], a[88:91], v143, v109 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[138:141], v[82:85], a[76:79], v143, v108 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[138:141], v[86:89], a[80:83], v143, v108 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[138:141], v[90:93], a[84:87], v143, v109 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[138:141], v[94:97], a[88:91], v143, v109 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[110:113], v[50:53], a[0:3], v142, v106 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[18:21], v169 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[110:113], v[54:57], a[4:7], v142, v106 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[22:25], v169 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[110:113], v[58:61], a[8:11], v142, v107 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[26:29], v169 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[110:113], v[62:65], a[12:15], v142, v107 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v169 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[126:129], v[66:69], a[0:3], v142, v106 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v170 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[126:129], v[70:73], a[4:7], v142, v106 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v170 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[126:129], v[74:77], a[8:11], v142, v107 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v170 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[126:129], v[78:81], a[12:15], v142, v107 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v170 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[114:117], v[50:53], a[16:19], v142, v106 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[114:117], v[54:57], a[20:23], v142, v106 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[114:117], v[58:61], a[24:27], v142, v107 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[114:117], v[62:65], a[92:95], v142, v107 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[130:133], v[66:69], a[16:19], v142, v106 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[130:133], v[70:73], a[20:23], v142, v106 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[130:133], v[74:77], a[24:27], v142, v107 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[130:133], v[78:81], a[92:95], v142, v107 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[118:121], v[50:53], a[96:99], v143, v106 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[118:121], v[54:57], a[100:103], v143, v106 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[118:121], v[58:61], a[104:107], v143, v107 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[118:121], v[62:65], a[108:111], v143, v107 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[134:137], v[66:69], a[96:99], v143, v106 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[134:137], v[70:73], a[100:103], v143, v106 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[134:137], v[74:77], a[104:107], v143, v107 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[134:137], v[78:81], a[108:111], v143, v107 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[122:125], v[50:53], a[112:115], v143, v106 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[122:125], v[54:57], a[116:119], v143, v106 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[122:125], v[58:61], a[120:123], v143, v107 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[122:125], v[62:65], a[124:127], v143, v107 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[138:141], v[66:69], a[112:115], v143, v106 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[138:141], v[70:73], a[116:119], v143, v106 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[138:141], v[74:77], a[120:123], v143, v107 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[138:141], v[78:81], a[124:127], v143, v107 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_waitcnt lgkmcnt(0)
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_barrier
	buffer_load_dword v142, v153, s[4:7], s91 offen
	buffer_load_dword v98, v153, s[8:11], s91 offen
	buffer_load_dword v143, v153, s[12:15], s91 offen
	buffer_load_dword v99, v153, s[16:19], s91 offen
	buffer_load_dword v102, v153, s[20:23], s91 offen
	buffer_load_dword v100, v153, s[24:27], s91 offen
	buffer_load_dword v103, v153, s[28:31], s91 offen
	buffer_load_dword v101, v153, s[36:39], s91 offen
	s_mov_b32 m0, s49
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[18:21], v[2:5], a[128:131],  v104, v108 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[110:113], v155 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[18:21], v[6:9], a[132:135],  v104, v108 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[114:117], v155 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[18:21], v[10:13], a[136:139],  v104, v109 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[118:121], v155 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[18:21], v[14:17], a[140:143],  v104, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[122:125], v155 offset:6144

	s_nop 0
	buffer_load_dwordx4 v146, s[0:3], s86 offen lds
	s_mov_b32 m0, s71
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[34:37], v[82:85], a[128:131],  v104, v108 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[126:129], v156 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[34:37], v[86:89], a[132:135],  v104, v108 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[130:133], v156 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[34:37], v[90:93], a[136:139],  v104, v109 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[134:137], v156 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[34:37], v[94:97], a[140:143],  v104, v109 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[138:141], v156 offset:6144

	s_nop 0
	buffer_load_dwordx4 v147, s[0:3], s86 offen lds
	s_mov_b32 m0, s72
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[22:25], v[2:5], a[144:147],  v104, v108 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[22:25], v[6:9], a[148:151],  v104, v108 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[22:25], v[10:13], a[152:155],  v104, v109 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[22:25], v[14:17], a[156:159],  v104, v109 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v154, s[0:3], s86 offen lds
	s_mov_b32 m0, s73
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[38:41], v[82:85], a[144:147],  v104, v108 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[38:41], v[86:89], a[148:151],  v104, v108 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[38:41], v[90:93], a[152:155],  v104, v109 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[38:41], v[94:97], a[156:159],  v104, v109 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v148, s[0:3], s86 offen lds
	s_mov_b32 m0, s50
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[26:29], v[2:5], a[160:163],  v105, v108 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[26:29], v[6:9], a[164:167],  v105, v108 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[26:29], v[10:13], a[168:171], v105, v109 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[26:29], v[14:17], a[172:175], v105, v109 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_add_i32 s86, s55, 0xffffff00
	buffer_load_dwordx4 v146, s[0:3], s87 offen lds
	s_mov_b32 m0, s74
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[42:45], v[82:85], a[160:163],  v105, v108 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[42:45], v[86:89], a[164:167],  v105, v108 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[42:45], v[90:93], a[168:171], v105, v109 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[42:45], v[94:97], a[172:175], v105, v109 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v147, s[0:3], s87 offen lds
	s_mov_b32 m0, s75
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[30:33], v[2:5], a[176:179], v105, v108 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[30:33], v[6:9], a[180:183], v105, v108 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[30:33], v[10:13], a[184:187], v105, v109 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[30:33], v[14:17], a[188:191], v105, v109 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v154, s[0:3], s87 offen lds
	s_mov_b32 m0, s76
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[46:49], v[82:85], a[176:179], v105, v108 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[46:49], v[86:89], a[180:183], v105, v108 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[46:49], v[90:93], a[184:187], v105, v109 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[46:49], v[94:97], a[188:191], v105, v109 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v148, s[0:3], s87 offen lds
	s_mov_b32 m0, s52
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[18:21], v[50:53], a[192:195],  v104, v106 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[2:5], v159 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[18:21], v[54:57], a[196:199],  v104, v106 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[6:9], v159 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[18:21], v[58:61], a[200:203],  v104, v107 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[10:13], v159 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[18:21], v[62:65], a[204:207],  v104, v107 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[14:17], v159 offset:6144

	s_nop 0
	buffer_load_dwordx4 v150, s[40:43], s88 offen lds
	s_mov_b32 m0, s77
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[34:37], v[66:69], a[192:195],  v104, v106 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[82:85], v160 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[34:37], v[70:73], a[196:199],  v104, v106 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[86:89], v160 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[34:37], v[74:77], a[200:203],  v104, v107 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[90:93], v160 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[34:37], v[78:81], a[204:207],  v104, v107 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[94:97], v160 offset:6144

	s_nop 0
	buffer_load_dwordx4 v149, s[40:43], s88 offen lds
	s_mov_b32 m0, s78
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[22:25], v[50:53], a[208:211],  v104, v106 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[22:25], v[54:57], a[212:215],  v104, v106 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[22:25], v[58:61], a[216:219],  v104, v107 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[22:25], v[62:65], a[220:223],  v104, v107 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v151, s[40:43], s88 offen lds
	s_mov_b32 m0, s79
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[38:41], v[66:69], a[208:211],  v104, v106 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[38:41], v[70:73], a[212:215],  v104, v106 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[38:41], v[74:77], a[216:219],  v104, v107 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[38:41], v[78:81], a[220:223],  v104, v107 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v152, s[40:43], s88 offen lds
	s_mov_b32 m0, s54
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[26:29], v[50:53], a[224:227],  v105, v106 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[26:29], v[54:57], a[228:231],  v105, v106 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[26:29], v[58:61], a[232:235], v105, v107 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[26:29], v[62:65], a[236:239], v105, v107 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v150, s[40:43], s89 offen lds
	s_mov_b32 m0, s80
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[42:45], v[66:69], a[224:227],  v105, v106 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[42:45], v[70:73], a[228:231],  v105, v106 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[42:45], v[74:77], a[232:235], v105, v107 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[42:45], v[78:81], a[236:239], v105, v107 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v149, s[40:43], s89 offen lds
	s_mov_b32 m0, s81
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[30:33], v[50:53], a[240:243], v105, v106 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[30:33], v[54:57], a[244:247], v105, v106 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[30:33], v[58:61], a[248:251], v105, v107 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[30:33], v[62:65], a[252:255], v105, v107 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v151, s[40:43], s89 offen lds
	s_mov_b32 m0, s82
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[46:49], v[66:69], a[240:243], v105, v106 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[46:49], v[70:73], a[244:247], v105, v106 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[46:49], v[74:77], a[248:251], v105, v107 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[46:49], v[78:81], a[252:255], v105, v107 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v152, s[40:43], s89 offen lds
	s_waitcnt lgkmcnt(0)
	s_waitcnt vmcnt(16)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[110:113], v[2:5], a[28:31],  v142, v102 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v163 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[110:113], v[6:9], a[32:35],  v142, v102 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v163 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[110:113], v[10:13], a[36:39],  v142, v103 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v163 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[110:113], v[14:17], a[40:43],  v142, v103 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[62:65], v163 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[126:129], v[82:85], a[28:31],  v142, v102 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[66:69], v164 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[126:129], v[86:89], a[32:35],  v142, v102 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[70:73], v164 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[126:129], v[90:93], a[36:39],  v142, v103 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[74:77], v164 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[126:129], v[94:97], a[40:43],  v142, v103 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[78:81], v164 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[114:117], v[2:5], a[44:47],  v142, v102 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[114:117], v[6:9], a[48:51],  v142, v102 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[114:117], v[10:13], a[52:55],  v142, v103 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[114:117], v[14:17], a[56:59],  v142, v103 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[130:133], v[82:85], a[44:47],  v142, v102 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[130:133], v[86:89], a[48:51],  v142, v102 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[130:133], v[90:93], a[52:55],  v142, v103 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[130:133], v[94:97], a[56:59],  v142, v103 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[118:121], v[2:5], a[60:63],  v143, v102 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[118:121], v[6:9], a[64:67],  v143, v102 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[118:121], v[10:13], a[68:71], v143, v103 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[118:121], v[14:17], a[72:75], v143, v103 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[134:137], v[82:85], a[60:63],  v143, v102 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[134:137], v[86:89], a[64:67],  v143, v102 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[134:137], v[90:93], a[68:71], v143, v103 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[134:137], v[94:97], a[72:75], v143, v103 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[122:125], v[2:5], a[76:79], v143, v102 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[122:125], v[6:9], a[80:83], v143, v102 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[122:125], v[10:13], a[84:87], v143, v103 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[122:125], v[14:17], a[88:91], v143, v103 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[138:141], v[82:85], a[76:79], v143, v102 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[138:141], v[86:89], a[80:83], v143, v102 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[138:141], v[90:93], a[84:87], v143, v103 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[138:141], v[94:97], a[88:91], v143, v103 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[110:113], v[50:53], a[0:3], v142, v100 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[18:21], v167 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[110:113], v[54:57], a[4:7], v142, v100 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[22:25], v167 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[110:113], v[58:61], a[8:11], v142, v101 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[26:29], v167 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[110:113], v[62:65], a[12:15], v142, v101 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v167 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[126:129], v[66:69], a[0:3], v142, v100 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v168 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[126:129], v[70:73], a[4:7], v142, v100 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v168 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[126:129], v[74:77], a[8:11], v142, v101 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v168 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[126:129], v[78:81], a[12:15], v142, v101 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v168 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[114:117], v[50:53], a[16:19], v142, v100 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[114:117], v[54:57], a[20:23], v142, v100 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[114:117], v[58:61], a[24:27], v142, v101 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[114:117], v[62:65], a[92:95], v142, v101 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[130:133], v[66:69], a[16:19], v142, v100 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[130:133], v[70:73], a[20:23], v142, v100 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[130:133], v[74:77], a[24:27], v142, v101 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[130:133], v[78:81], a[92:95], v142, v101 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[118:121], v[50:53], a[96:99], v143, v100 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[118:121], v[54:57], a[100:103], v143, v100 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[118:121], v[58:61], a[104:107], v143, v101 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[118:121], v[62:65], a[108:111], v143, v101 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[134:137], v[66:69], a[96:99], v143, v100 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[134:137], v[70:73], a[100:103], v143, v100 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[134:137], v[74:77], a[104:107], v143, v101 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[134:137], v[78:81], a[108:111], v143, v101 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[122:125], v[50:53], a[112:115], v143, v100 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[122:125], v[54:57], a[116:119], v143, v100 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[122:125], v[58:61], a[120:123], v143, v101 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[122:125], v[62:65], a[124:127], v143, v101 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[138:141], v[66:69], a[112:115], v143, v100 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[138:141], v[70:73], a[116:119], v143, v100 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[138:141], v[74:77], a[120:123], v143, v101 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[138:141], v[78:81], a[124:127], v143, v101 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_waitcnt lgkmcnt(0)
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_barrier
	buffer_load_dword v171, v153, s[4:7], s86 offen
	buffer_load_dword v180, v153, s[8:11], s86 offen
	buffer_load_dword v172, v153, s[12:15], s86 offen
	buffer_load_dword v181, v153, s[16:19], s86 offen
	buffer_load_dword v184, v153, s[20:23], s86 offen
	buffer_load_dword v182, v153, s[24:27], s86 offen
	buffer_load_dword v185, v153, s[28:31], s86 offen
	buffer_load_dword v183, v153, s[36:39], s86 offen
	s_min_u32 s86, s90, 29
	s_lshl_b32 s86, s86, 7
	s_addk_i32 s86, 0x100
	s_add_i32 s87, s86, s83
	s_mov_b32 m0, s48
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[18:21], v[2:5], a[128:131],  v98, v102 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[176:179], v157 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[18:21], v[6:9], a[132:135],  v98, v102 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[186:189], v157 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[18:21], v[10:13], a[136:139],  v98, v103 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[190:193], v157 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[18:21], v[14:17], a[140:143],  v98, v103 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[194:197], v157 offset:6144

	s_add_i32 s88, s86, s59
	buffer_load_dwordx4 v146, s[0:3], s87 offen lds
	s_mov_b32 m0, s56
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[34:37], v[82:85], a[128:131],  v98, v102 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[198:201], v158 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[34:37], v[86:89], a[132:135],  v98, v102 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[202:205], v158 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[34:37], v[90:93], a[136:139],  v98, v103 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[206:209], v158 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[34:37], v[94:97], a[140:143],  v98, v103 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[210:213], v158 offset:6144

	s_add_i32 s89, s86, s84
	buffer_load_dwordx4 v147, s[0:3], s87 offen lds
	s_mov_b32 m0, s57
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[22:25], v[2:5], a[144:147],  v98, v102 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[22:25], v[6:9], a[148:151],  v98, v102 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[22:25], v[10:13], a[152:155],  v98, v103 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[22:25], v[14:17], a[156:159],  v98, v103 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_add_i32 s86, s86, s67
	buffer_load_dwordx4 v154, s[0:3], s87 offen lds
	s_mov_b32 m0, s58
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[38:41], v[82:85], a[144:147],  v98, v102 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[38:41], v[86:89], a[148:151],  v98, v102 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[38:41], v[90:93], a[152:155],  v98, v103 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[38:41], v[94:97], a[156:159],  v98, v103 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_cmpk_lg_i32 s47, 0x800
	buffer_load_dwordx4 v148, s[0:3], s87 offen lds
	s_mov_b32 m0, s60
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[26:29], v[2:5], a[160:163],  v99, v102 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[26:29], v[6:9], a[164:167],  v99, v102 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[26:29], v[10:13], a[168:171], v99, v103 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[26:29], v[14:17], a[172:175], v99, v103 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v146, s[0:3], s88 offen lds
	s_mov_b32 m0, s61
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[42:45], v[82:85], a[160:163],  v99, v102 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[42:45], v[86:89], a[164:167],  v99, v102 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[42:45], v[90:93], a[168:171], v99, v103 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[42:45], v[94:97], a[172:175], v99, v103 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v147, s[0:3], s88 offen lds
	s_mov_b32 m0, s62
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[30:33], v[2:5], a[176:179], v99, v102 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[30:33], v[6:9], a[180:183], v99, v102 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[30:33], v[10:13], a[184:187], v99, v103 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[30:33], v[14:17], a[188:191], v99, v103 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v154, s[0:3], s88 offen lds
	s_mov_b32 m0, s63
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[46:49], v[82:85], a[176:179], v99, v102 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[46:49], v[86:89], a[180:183], v99, v102 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[46:49], v[90:93], a[184:187], v99, v103 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[46:49], v[94:97], a[188:191], v99, v103 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v148, s[0:3], s88 offen lds
	s_mov_b32 m0, s51
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[18:21], v[50:53], a[192:195],  v98, v100 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[2:5], v161 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[18:21], v[54:57], a[196:199],  v98, v100 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[6:9], v161 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[18:21], v[58:61], a[200:203],  v98, v101 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[10:13], v161 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[18:21], v[62:65], a[204:207],  v98, v101 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[14:17], v161 offset:6144

	s_nop 0
	buffer_load_dwordx4 v150, s[40:43], s89 offen lds
	s_mov_b32 m0, s64
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[34:37], v[66:69], a[192:195],  v98, v100 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[130:133], v162 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[34:37], v[70:73], a[196:199],  v98, v100 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[134:137], v162 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[34:37], v[74:77], a[200:203],  v98, v101 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[138:141], v162 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[34:37], v[78:81], a[204:207],  v98, v101 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[142:145], v162 offset:6144

	s_nop 0
	buffer_load_dwordx4 v149, s[40:43], s89 offen lds
	s_mov_b32 m0, s65
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[22:25], v[50:53], a[208:211],  v98, v100 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[22:25], v[54:57], a[212:215],  v98, v100 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[22:25], v[58:61], a[216:219],  v98, v101 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[22:25], v[62:65], a[220:223],  v98, v101 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v151, s[40:43], s89 offen lds
	s_mov_b32 m0, s66
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[38:41], v[66:69], a[208:211],  v98, v100 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[38:41], v[70:73], a[212:215],  v98, v100 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[38:41], v[74:77], a[216:219],  v98, v101 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[38:41], v[78:81], a[220:223],  v98, v101 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v152, s[40:43], s89 offen lds
	s_mov_b32 m0, s53
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[26:29], v[50:53], a[224:227],  v99, v100 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[26:29], v[54:57], a[228:231],  v99, v100 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[26:29], v[58:61], a[232:235], v99, v101 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[26:29], v[62:65], a[236:239], v99, v101 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v150, s[40:43], s86 offen lds
	s_mov_b32 m0, s68
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[42:45], v[66:69], a[224:227],  v99, v100 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[42:45], v[70:73], a[228:231],  v99, v100 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[42:45], v[74:77], a[232:235], v99, v101 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[42:45], v[78:81], a[236:239], v99, v101 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v149, s[40:43], s86 offen lds
	s_mov_b32 m0, s69
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[30:33], v[50:53], a[240:243], v99, v100 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[30:33], v[54:57], a[244:247], v99, v100 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[30:33], v[58:61], a[248:251], v99, v101 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[30:33], v[62:65], a[252:255], v99, v101 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v151, s[40:43], s86 offen lds
	s_mov_b32 m0, s70
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[46:49], v[66:69], a[240:243], v99, v100 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[46:49], v[70:73], a[244:247], v99, v100 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[46:49], v[74:77], a[248:251], v99, v101 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[46:49], v[78:81], a[252:255], v99, v101 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v152, s[40:43], s86 offen lds
	s_cselect_b32 s86, s55, 0x1f00
	s_waitcnt lgkmcnt(0)
	s_waitcnt vmcnt(16)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[176:179], v[2:5], a[28:31],  v171, v184 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[98:101], v165 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[176:179], v[6:9], a[32:35],  v171, v184 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[102:105], v165 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[176:179], v[10:13], a[36:39],  v171, v185 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[106:109], v165 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[176:179], v[14:17], a[40:43],  v171, v185 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[110:113], v165 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[198:201], v[130:133], a[28:31],  v171, v184 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[114:117], v166 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[198:201], v[134:137], a[32:35],  v171, v184 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[118:121], v166 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[198:201], v[138:141], a[36:39],  v171, v185 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[122:125], v166 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[198:201], v[142:145], a[40:43],  v171, v185 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[126:129], v166 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[186:189], v[2:5], a[44:47],  v171, v184 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[186:189], v[6:9], a[48:51],  v171, v184 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[186:189], v[10:13], a[52:55],  v171, v185 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[186:189], v[14:17], a[56:59],  v171, v185 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[202:205], v[130:133], a[44:47],  v171, v184 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[202:205], v[134:137], a[48:51],  v171, v184 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[202:205], v[138:141], a[52:55],  v171, v185 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[202:205], v[142:145], a[56:59],  v171, v185 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[190:193], v[2:5], a[60:63],  v172, v184 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[190:193], v[6:9], a[64:67],  v172, v184 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[190:193], v[10:13], a[68:71], v172, v185 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[190:193], v[14:17], a[72:75], v172, v185 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[206:209], v[130:133], a[60:63],  v172, v184 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[206:209], v[134:137], a[64:67],  v172, v184 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[206:209], v[138:141], a[68:71], v172, v185 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[206:209], v[142:145], a[72:75], v172, v185 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[194:197], v[2:5], a[76:79], v172, v184 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[194:197], v[6:9], a[80:83], v172, v184 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[194:197], v[10:13], a[84:87], v172, v185 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[194:197], v[14:17], a[88:91], v172, v185 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[210:213], v[130:133], a[76:79], v172, v184 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[210:213], v[134:137], a[80:83], v172, v184 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[210:213], v[138:141], a[84:87], v172, v185 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[210:213], v[142:145], a[88:91], v172, v185 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[176:179], v[98:101], a[0:3], v171, v182 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[66:69], v169 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[176:179], v[102:105], a[4:7], v171, v182 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[70:73], v169 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[176:179], v[106:109], a[8:11], v171, v183 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[74:77], v169 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[176:179], v[110:113], a[12:15], v171, v183 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[78:81], v169 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[198:201], v[114:117], a[0:3], v171, v182 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[82:85], v170 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[198:201], v[118:121], a[4:7], v171, v182 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[86:89], v170 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[198:201], v[122:125], a[8:11], v171, v183 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[90:93], v170 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[198:201], v[126:129], a[12:15], v171, v183 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[94:97], v170 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[186:189], v[98:101], a[16:19], v171, v182 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[186:189], v[102:105], a[20:23], v171, v182 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[186:189], v[106:109], a[24:27], v171, v183 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[186:189], v[110:113], a[92:95], v171, v183 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[202:205], v[114:117], a[16:19], v171, v182 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[202:205], v[118:121], a[20:23], v171, v182 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[202:205], v[122:125], a[24:27], v171, v183 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[202:205], v[126:129], a[92:95], v171, v183 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[190:193], v[98:101], a[96:99], v172, v182 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[190:193], v[102:105], a[100:103], v172, v182 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[190:193], v[106:109], a[104:107], v172, v183 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[190:193], v[110:113], a[108:111], v172, v183 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[206:209], v[114:117], a[96:99], v172, v182 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[206:209], v[118:121], a[100:103], v172, v182 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[206:209], v[122:125], a[104:107], v172, v183 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[206:209], v[126:129], a[108:111], v172, v183 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[194:197], v[98:101], a[112:115], v172, v182 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[194:197], v[102:105], a[116:119], v172, v182 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[194:197], v[106:109], a[120:123], v172, v183 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[194:197], v[110:113], a[124:127], v172, v183 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[210:213], v[114:117], a[112:115], v172, v182 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[210:213], v[118:121], a[116:119], v172, v182 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[210:213], v[122:125], a[120:123], v172, v183 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[210:213], v[126:129], a[124:127], v172, v183 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_waitcnt lgkmcnt(0)
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_barrier
	buffer_load_dword v178, v153, s[4:7], s86 offen
	buffer_load_dword v171, v153, s[8:11], s86 offen
	buffer_load_dword v179, v153, s[12:15], s86 offen
	buffer_load_dword v172, v153, s[16:19], s86 offen
	buffer_load_dword v176, v153, s[20:23], s86 offen
	buffer_load_dword v173, v153, s[24:27], s86 offen
	buffer_load_dword v177, v153, s[28:31], s86 offen
	buffer_load_dword v175, v153, s[36:39], s86 offen
	s_min_u32 s86, s85, 29
	s_lshl_b32 s86, s86, 7
	s_addk_i32 s86, 0x100
	s_add_i32 s89, s86, s83
	s_mov_b32 m0, s49
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[66:69], v[2:5], a[128:131],  v180, v184 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v155 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[66:69], v[6:9], a[132:135],  v180, v184 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v155 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[66:69], v[10:13], a[136:139],  v180, v185 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v155 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[66:69], v[14:17], a[140:143],  v180, v185 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v155 offset:6144

	s_add_i32 s88, s86, s59
	buffer_load_dwordx4 v146, s[0:3], s89 offen lds
	s_mov_b32 m0, s71
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[82:85], v[130:133], a[128:131],  v180, v184 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v156 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[82:85], v[134:137], a[132:135],  v180, v184 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v156 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[82:85], v[138:141], a[136:139],  v180, v185 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v156 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[82:85], v[142:145], a[140:143],  v180, v185 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[62:65], v156 offset:6144

	s_add_i32 s87, s86, s84
	buffer_load_dwordx4 v147, s[0:3], s89 offen lds
	s_mov_b32 m0, s72
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[70:73], v[2:5], a[144:147],  v180, v184 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[70:73], v[6:9], a[148:151],  v180, v184 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[70:73], v[10:13], a[152:155],  v180, v185 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[70:73], v[14:17], a[156:159],  v180, v185 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_add_i32 s86, s86, s67
	buffer_load_dwordx4 v154, s[0:3], s89 offen lds
	s_mov_b32 m0, s73
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[86:89], v[130:133], a[144:147],  v180, v184 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[86:89], v[134:137], a[148:151],  v180, v184 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[86:89], v[138:141], a[152:155],  v180, v185 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[86:89], v[142:145], a[156:159],  v180, v185 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_addk_i32 s47, 0x800
	buffer_load_dwordx4 v148, s[0:3], s89 offen lds
	s_mov_b32 m0, s50
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[74:77], v[2:5], a[160:163],  v181, v184 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[74:77], v[6:9], a[164:167],  v181, v184 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[74:77], v[10:13], a[168:171], v181, v185 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[74:77], v[14:17], a[172:175], v181, v185 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_addk_i32 s55, 0x1000
	buffer_load_dwordx4 v146, s[0:3], s88 offen lds
	s_mov_b32 m0, s74
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[90:93], v[130:133], a[160:163],  v181, v184 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[90:93], v[134:137], a[164:167],  v181, v184 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[90:93], v[138:141], a[168:171], v181, v185 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[90:93], v[142:145], a[172:175], v181, v185 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_add_i32 s85, s85, 16
	buffer_load_dwordx4 v147, s[0:3], s88 offen lds
	s_mov_b32 m0, s75
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[78:81], v[2:5], a[176:179], v181, v184 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[78:81], v[6:9], a[180:183], v181, v184 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[78:81], v[10:13], a[184:187], v181, v185 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[78:81], v[14:17], a[188:191], v181, v185 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_cmpk_eq_i32 s47, 0x1000
	buffer_load_dwordx4 v154, s[0:3], s88 offen lds
	s_mov_b32 m0, s76
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[94:97], v[130:133], a[176:179], v181, v184 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[94:97], v[134:137], a[180:183], v181, v184 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[94:97], v[138:141], a[184:187], v181, v185 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[94:97], v[142:145], a[188:191], v181, v185 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v148, s[0:3], s88 offen lds
	s_mov_b32 m0, s52
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[66:69], v[98:101], a[192:195],  v180, v182 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[2:5], v159 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[66:69], v[102:105], a[196:199],  v180, v182 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[6:9], v159 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[66:69], v[106:109], a[200:203],  v180, v183 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[10:13], v159 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[66:69], v[110:113], a[204:207],  v180, v183 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[14:17], v159 offset:6144

	s_nop 0
	buffer_load_dwordx4 v150, s[40:43], s87 offen lds
	s_mov_b32 m0, s77
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[82:85], v[114:117], a[192:195],  v180, v182 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[18:21], v160 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[82:85], v[118:121], a[196:199],  v180, v182 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[22:25], v160 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[82:85], v[122:125], a[200:203],  v180, v183 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[26:29], v160 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[82:85], v[126:129], a[204:207],  v180, v183 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v160 offset:6144

	s_nop 0
	buffer_load_dwordx4 v149, s[40:43], s87 offen lds
	s_mov_b32 m0, s78
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[70:73], v[98:101], a[208:211],  v180, v182 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[70:73], v[102:105], a[212:215],  v180, v182 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[70:73], v[106:109], a[216:219],  v180, v183 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[70:73], v[110:113], a[220:223],  v180, v183 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v151, s[40:43], s87 offen lds
	s_mov_b32 m0, s79
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[86:89], v[114:117], a[208:211],  v180, v182 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[86:89], v[118:121], a[212:215],  v180, v182 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[86:89], v[122:125], a[216:219],  v180, v183 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[86:89], v[126:129], a[220:223],  v180, v183 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v152, s[40:43], s87 offen lds
	s_mov_b32 m0, s54
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[74:77], v[98:101], a[224:227],  v181, v182 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[74:77], v[102:105], a[228:231],  v181, v182 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[74:77], v[106:109], a[232:235], v181, v183 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[74:77], v[110:113], a[236:239], v181, v183 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v150, s[40:43], s86 offen lds
	s_mov_b32 m0, s80
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[90:93], v[114:117], a[224:227],  v181, v182 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[90:93], v[118:121], a[228:231],  v181, v182 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[90:93], v[122:125], a[232:235], v181, v183 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[90:93], v[126:129], a[236:239], v181, v183 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v149, s[40:43], s86 offen lds
	s_mov_b32 m0, s81
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[78:81], v[98:101], a[240:243], v181, v182 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[78:81], v[102:105], a[244:247], v181, v182 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[78:81], v[106:109], a[248:251], v181, v183 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[78:81], v[110:113], a[252:255], v181, v183 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v151, s[40:43], s86 offen lds
	s_mov_b32 m0, s82
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[94:97], v[114:117], a[240:243], v181, v182 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[94:97], v[118:121], a[244:247], v181, v182 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[94:97], v[122:125], a[248:251], v181, v183 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[94:97], v[126:129], a[252:255], v181, v183 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	s_nop 0
	buffer_load_dwordx4 v152, s[40:43], s86 offen lds
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_cbranch_scc0 .LBB0_1
; %bb.2:
	v_accvgpr_read_b32 v137, a91
	v_accvgpr_read_b32 v199, a63
	v_lshl_or_b32 v1, s35, 2, v1
	v_accvgpr_read_b32 v135, a89
	v_accvgpr_read_b32 v134, a88
	v_accvgpr_read_b32 v197, a61
	v_accvgpr_read_b32 v196, a60
	v_lshl_or_b32 v2, s46, 2, v174
	v_mul_lo_u32 v3, v1, s33
	v_accvgpr_read_b32 v183, a79
	v_pk_mul_f32 v[234:235], v[196:197], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[196:197], v[134:135], s[34:35] op_sel_hi:[1,0]
	v_add_lshl_u32 v134, v2, v3, 6
	v_lshrrev_b32_e32 v4, 2, v0
	v_accvgpr_read_b32 v182, a78
	v_ashrrev_i32_e32 v135, 31, v134
	v_and_b32_e32 v4, 12, v4
	v_and_b32_e32 v0, 15, v0
	v_pk_mul_f32 v[248:249], v[182:183], s[34:35] op_sel_hi:[1,0]
	v_lshl_add_u64 v[182:183], v[134:135], 1, s[44:45]
	v_mad_u64_u32 v[134:135], s[0:1], v4, s33, v[0:1]
	v_ashrrev_i32_e32 v135, 31, v134
	v_accvgpr_read_b32 v187, a75
	v_lshlrev_b64 v[4:5], 1, v[134:135]
	v_add_u32_e32 v134, s33, v134
	v_accvgpr_read_b32 v186, a74
	v_ashrrev_i32_e32 v135, 31, v134
	s_waitcnt vmcnt(17)
	v_accvgpr_read_b32 v179, a83
	v_accvgpr_read_b32 v181, a77
	v_accvgpr_read_b32 v180, a76
	v_accvgpr_read_b32 v231, a31
	v_pk_mul_f32 v[244:245], v[186:187], s[34:35] op_sel_hi:[1,0]
	v_lshlrev_b64 v[186:187], 1, v[134:135]
	v_add_u32_e32 v134, s33, v134
	v_accvgpr_read_b32 v177, a81
	v_accvgpr_read_b32 v176, a80
	v_accvgpr_read_b32 v191, a71
	v_accvgpr_read_b32 v229, a29
	v_accvgpr_read_b32 v228, a28
	v_pk_mul_f32 v[250:251], v[180:181], s[34:35] op_sel_hi:[1,0]
	v_add_u32_e32 v180, s33, v134
	v_accvgpr_read_b32 v190, a70
	v_accvgpr_read_b32 v223, a39
	v_accvgpr_read_b32 v227, a35
	v_pk_mul_f32 v[228:229], v[228:229], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[254:255], v[176:177], s[34:35] op_sel_hi:[1,0]
	s_waitcnt vmcnt(16)
	v_lshl_add_u64 v[174:175], v[182:183], 0, v[4:5]
	v_lshl_add_u64 v[176:177], v[182:183], 0, v[186:187]
	v_ashrrev_i32_e32 v135, 31, v134
	v_ashrrev_i32_e32 v181, 31, v180
	v_accvgpr_read_b32 v178, a82
	v_accvgpr_read_b32 v185, a73
	v_accvgpr_read_b32 v184, a72
	v_accvgpr_read_b32 v219, a43
	v_accvgpr_read_b32 v221, a37
	v_accvgpr_read_b32 v220, a36
	v_accvgpr_read_b32 v225, a33
	v_accvgpr_read_b32 v224, a32
	v_accvgpr_read_b32 v230, a30
	v_pk_mul_f32 v[240:241], v[190:191], s[34:35] op_sel_hi:[1,0]
	global_store_short_d16_hi v[174:175], v228, off
	global_store_short_d16_hi v[176:177], v229, off
	v_lshlrev_b64 v[228:229], 1, v[134:135]
	v_lshlrev_b64 v[190:191], 1, v[180:181]
	v_accvgpr_read_b32 v218, a42
	v_accvgpr_read_b32 v217, a41
	v_accvgpr_read_b32 v216, a40
	v_accvgpr_read_b32 v222, a38
	v_accvgpr_read_b32 v226, a34
	v_pk_mul_f32 v[230:231], v[230:231], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[224:225], v[224:225], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[220:221], v[220:221], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[246:247], v[184:185], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[252:253], v[178:179], s[34:35] op_sel_hi:[1,0]
	v_lshl_add_u64 v[178:179], v[182:183], 0, v[228:229]
	v_lshl_add_u64 v[184:185], v[182:183], 0, v[190:191]
	s_mul_i32 s0, s33, 13
	v_pk_mul_f32 v[226:227], v[226:227], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[222:223], v[222:223], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[218:219], v[218:219], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[216:217], v[216:217], s[34:35] op_sel_hi:[1,0]
	global_store_short_d16_hi v[178:179], v230, off
	global_store_short_d16_hi v[184:185], v231, off
	global_store_short_d16_hi v[174:175], v224, off offset:32
	global_store_short_d16_hi v[176:177], v225, off offset:32
	global_store_short_d16_hi v[178:179], v226, off offset:32
	global_store_short_d16_hi v[184:185], v227, off offset:32
	global_store_short_d16_hi v[174:175], v220, off offset:64
	global_store_short_d16_hi v[176:177], v221, off offset:64
	global_store_short_d16_hi v[178:179], v222, off offset:64
	global_store_short_d16_hi v[184:185], v223, off offset:64
	global_store_short_d16_hi v[174:175], v216, off offset:96
	global_store_short_d16_hi v[176:177], v217, off offset:96
	global_store_short_d16_hi v[178:179], v218, off offset:96
	global_store_short_d16_hi v[184:185], v219, off offset:96
	v_add_u32_e32 v174, s0, v180
	v_ashrrev_i32_e32 v175, 31, v174
	v_lshlrev_b64 v[224:225], 1, v[174:175]
	v_add_u32_e32 v174, s33, v174
	v_ashrrev_i32_e32 v175, 31, v174
	v_lshlrev_b64 v[226:227], 1, v[174:175]
	v_add_u32_e32 v174, s33, v174
	v_accvgpr_read_b32 v215, a47
	v_add_u32_e32 v218, s33, v174
	v_accvgpr_read_b32 v198, a62
	v_accvgpr_read_b32 v207, a55
	v_accvgpr_read_b32 v211, a51
	v_accvgpr_read_b32 v213, a45
	v_accvgpr_read_b32 v212, a44
	v_ashrrev_i32_e32 v175, 31, v174
	v_ashrrev_i32_e32 v219, 31, v218
	v_accvgpr_read_b32 v203, a59
	v_accvgpr_read_b32 v205, a53
	v_accvgpr_read_b32 v204, a52
	v_accvgpr_read_b32 v209, a49
	v_accvgpr_read_b32 v208, a48
	v_accvgpr_read_b32 v214, a46
	v_pk_mul_f32 v[212:213], v[212:213], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[232:233], v[198:199], s[34:35] op_sel_hi:[1,0]
	v_lshl_add_u64 v[184:185], v[182:183], 0, v[224:225]
	v_lshl_add_u64 v[216:217], v[182:183], 0, v[226:227]
	v_lshlrev_b64 v[230:231], 1, v[174:175]
	v_lshlrev_b64 v[198:199], 1, v[218:219]
	v_accvgpr_read_b32 v202, a58
	v_accvgpr_read_b32 v201, a57
	v_accvgpr_read_b32 v200, a56
	v_accvgpr_read_b32 v206, a54
	v_accvgpr_read_b32 v210, a50
	v_pk_mul_f32 v[214:215], v[214:215], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[208:209], v[208:209], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[204:205], v[204:205], s[34:35] op_sel_hi:[1,0]
	global_store_short_d16_hi v[184:185], v212, off
	global_store_short_d16_hi v[216:217], v213, off
	v_lshl_add_u64 v[212:213], v[182:183], 0, v[230:231]
	v_lshl_add_u64 v[220:221], v[182:183], 0, v[198:199]
	v_pk_mul_f32 v[210:211], v[210:211], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[206:207], v[206:207], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[202:203], v[202:203], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[200:201], v[200:201], s[34:35] op_sel_hi:[1,0]
	global_store_short_d16_hi v[212:213], v214, off
	global_store_short_d16_hi v[220:221], v215, off
	global_store_short_d16_hi v[184:185], v208, off offset:32
	global_store_short_d16_hi v[216:217], v209, off offset:32
	global_store_short_d16_hi v[212:213], v210, off offset:32
	global_store_short_d16_hi v[220:221], v211, off offset:32
	global_store_short_d16_hi v[184:185], v204, off offset:64
	global_store_short_d16_hi v[216:217], v205, off offset:64
	global_store_short_d16_hi v[212:213], v206, off offset:64
	global_store_short_d16_hi v[220:221], v207, off offset:64
	global_store_short_d16_hi v[184:185], v200, off offset:96
	global_store_short_d16_hi v[216:217], v201, off offset:96
	global_store_short_d16_hi v[212:213], v202, off offset:96
	global_store_short_d16_hi v[220:221], v203, off offset:96
	v_add_u32_e32 v184, s0, v218
	v_ashrrev_i32_e32 v185, 31, v184
	v_lshlrev_b64 v[200:201], 1, v[184:185]
	v_add_u32_e32 v184, s33, v184
	v_ashrrev_i32_e32 v185, 31, v184
	v_lshlrev_b64 v[202:203], 1, v[184:185]
	v_add_u32_e32 v184, s33, v184
	v_ashrrev_i32_e32 v185, 31, v184
	v_lshlrev_b64 v[204:205], 1, v[184:185]
	v_add_u32_e32 v184, s33, v184
	v_accvgpr_read_b32 v195, a67
	v_ashrrev_i32_e32 v185, 31, v184
	v_accvgpr_read_b32 v189, a69
	v_accvgpr_read_b32 v188, a68
	v_accvgpr_read_b32 v193, a65
	v_accvgpr_read_b32 v192, a64
	v_lshlrev_b64 v[206:207], 1, v[184:185]
	v_add_u32_e32 v184, s0, v184
	v_accvgpr_read_b32 v194, a66
	v_pk_mul_f32 v[238:239], v[192:193], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[242:243], v[188:189], s[34:35] op_sel_hi:[1,0]
	v_lshl_add_u64 v[208:209], v[182:183], 0, v[200:201]
	v_lshl_add_u64 v[210:211], v[182:183], 0, v[202:203]
	v_lshl_add_u64 v[212:213], v[182:183], 0, v[204:205]
	v_lshl_add_u64 v[214:215], v[182:183], 0, v[206:207]
	v_ashrrev_i32_e32 v185, 31, v184
	v_pk_mul_f32 v[236:237], v[194:195], s[34:35] op_sel_hi:[1,0]
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
	v_add_u32_e32 v184, s33, v184
	v_ashrrev_i32_e32 v185, 31, v184
	v_lshlrev_b64 v[210:211], 1, v[184:185]
	v_add_u32_e32 v184, s33, v184
	v_ashrrev_i32_e32 v185, 31, v184
	v_lshlrev_b64 v[212:213], 1, v[184:185]
	v_add_u32_e32 v184, s33, v184
	v_ashrrev_i32_e32 v185, 31, v184
	v_accvgpr_read_b32 v149, a87
	v_lshlrev_b64 v[214:215], 1, v[184:185]
	v_or_b32_e32 v0, 2, v1
	v_accvgpr_read_b32 v136, a90
	v_accvgpr_read_b32 v148, a86
	v_accvgpr_read_b32 v147, a85
	v_accvgpr_read_b32 v146, a84
	v_lshl_add_u64 v[236:237], v[182:183], 0, v[208:209]
	v_lshl_add_u64 v[238:239], v[182:183], 0, v[210:211]
	v_lshl_add_u64 v[240:241], v[182:183], 0, v[212:213]
	v_lshl_add_u64 v[182:183], v[182:183], 0, v[214:215]
	v_mul_lo_u32 v0, v0, s33
	v_pk_mul_f32 v[188:189], v[148:149], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[192:193], v[146:147], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[194:195], v[136:137], s[34:35] op_sel_hi:[1,0]
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
	v_lshl_add_u64 v[182:183], v[182:183], 1, s[44:45]
	v_accvgpr_read_b32 v157, a147
	v_accvgpr_read_b32 v160, a142
	v_accvgpr_read_b32 v159, a141
	v_accvgpr_read_b32 v158, a140
	v_accvgpr_read_b32 v165, a139
	v_accvgpr_read_b32 v169, a135
	v_accvgpr_read_b32 v172, a130
	v_pk_mul_f32 v[170:171], v[170:171], s[34:35] op_sel_hi:[1,0]
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
	v_pk_mul_f32 v[172:173], v[172:173], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[160:161], v[160:161], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[158:159], v[158:159], s[34:35] op_sel_hi:[1,0]
	global_store_short_d16_hi v[184:185], v170, off
	global_store_short_d16_hi v[188:189], v171, off
	v_lshl_add_u64 v[170:171], v[182:183], 0, v[228:229]
	v_lshl_add_u64 v[192:193], v[182:183], 0, v[190:191]
	v_accvgpr_read_b32 v130, a160
	v_accvgpr_read_b32 v139, a157
	v_accvgpr_read_b32 v140, a158
	v_accvgpr_read_b32 v141, a159
	v_accvgpr_read_b32 v142, a152
	v_accvgpr_read_b32 v153, a151
	v_accvgpr_read_b32 v156, a146
	v_pk_mul_f32 v[168:169], v[168:169], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[166:167], v[166:167], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[164:165], v[164:165], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[162:163], v[162:163], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[154:155], v[154:155], s[34:35] op_sel_hi:[1,0]
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
	v_pk_mul_f32 v[156:157], v[156:157], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[140:141], v[140:141], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[138:139], v[138:139], s[34:35] op_sel_hi:[1,0]
	global_store_short_d16_hi v[158:159], v154, off
	global_store_short_d16_hi v[160:161], v155, off
	v_lshl_add_u64 v[154:155], v[182:183], 0, v[230:231]
	v_lshl_add_u64 v[162:163], v[182:183], 0, v[198:199]
	v_accvgpr_read_b32 v118, a172
	v_accvgpr_read_b32 v122, a168
	v_accvgpr_read_b32 v126, a164
	v_accvgpr_read_b32 v132, a162
	v_accvgpr_read_b32 v133, a163
	v_pk_mul_f32 v[152:153], v[152:153], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[150:151], v[150:151], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[144:145], v[144:145], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[142:143], v[142:143], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[130:131], v[130:131], s[34:35] op_sel_hi:[1,0]
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
	v_pk_mul_f32 v[132:133], v[132:133], s[34:35] op_sel_hi:[1,0]
	global_store_short_d16_hi v[138:139], v130, off
	global_store_short_d16_hi v[140:141], v131, off
	v_lshl_add_u64 v[130:131], v[182:183], 0, v[204:205]
	v_lshl_add_u64 v[142:143], v[182:183], 0, v[206:207]
	v_or_b32_e32 v1, 2, v2
	v_accvgpr_read_b32 v102, a188
	v_accvgpr_read_b32 v115, a177
	v_pk_mul_f32 v[128:129], v[128:129], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[126:127], v[126:127], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[124:125], v[124:125], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[122:123], v[122:123], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[120:121], v[120:121], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[118:119], v[118:119], s[34:35] op_sel_hi:[1,0]
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
	v_add_lshl_u32 v130, v1, v3, 6
	v_accvgpr_read_b32 v104, a190
	v_accvgpr_read_b32 v105, a191
	v_accvgpr_read_b32 v106, a184
	v_accvgpr_read_b32 v110, a180
	v_accvgpr_read_b32 v116, a178
	v_accvgpr_read_b32 v117, a179
	v_accvgpr_read_b32 v137, a27
	v_accvgpr_read_b32 v223, a3
	v_pk_mul_f32 v[114:115], v[114:115], s[34:35] op_sel_hi:[1,0]
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
	v_pk_mul_f32 v[116:117], v[116:117], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[104:105], v[104:105], s[34:35] op_sel_hi:[1,0]
	global_store_short_d16_hi v[118:119], v114, off
	global_store_short_d16_hi v[120:121], v115, off
	v_lshl_add_u64 v[114:115], v[182:183], 0, v[212:213]
	v_lshl_add_u64 v[122:123], v[182:183], 0, v[214:215]
	v_lshl_add_u64 v[130:131], v[130:131], 1, s[44:45]
	v_accvgpr_read_b32 v101, a95
	v_accvgpr_read_b32 v136, a26
	v_accvgpr_read_b32 v177, a19
	v_accvgpr_read_b32 v181, a15
	v_accvgpr_read_b32 v235, a11
	v_accvgpr_read_b32 v218, a6
	v_accvgpr_read_b32 v217, a5
	v_accvgpr_read_b32 v216, a4
	v_accvgpr_read_b32 v222, a2
	v_pk_mul_f32 v[112:113], v[112:113], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[110:111], v[110:111], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[108:109], v[108:109], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[106:107], v[106:107], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[102:103], v[102:103], s[34:35] op_sel_hi:[1,0]
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
	v_pk_mul_f32 v[104:105], v[220:221], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[128:129], v[134:135], s[34:35] op_sel_hi:[1,0]
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
	v_pk_mul_f32 v[102:103], v[222:223], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[106:107], v[218:219], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[108:109], v[216:217], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[126:127], v[136:137], s[34:35] op_sel_hi:[1,0]
	global_store_short_d16_hi v[132:133], v104, off
	global_store_short_d16_hi v[134:135], v105, off
	v_lshl_add_u64 v[104:105], v[130:131], 0, v[228:229]
	v_lshl_add_u64 v[136:137], v[130:131], 0, v[190:191]
	v_accvgpr_read_b32 v82, a108
	v_accvgpr_read_b32 v95, a97
	v_accvgpr_read_b32 v148, a22
	v_accvgpr_read_b32 v147, a21
	v_accvgpr_read_b32 v146, a20
	v_pk_mul_f32 v[110:111], v[234:235], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[112:113], v[232:233], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[114:115], v[180:181], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[116:117], v[178:179], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[118:119], v[176:177], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[120:121], v[174:175], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[100:101], v[100:101], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[98:99], v[98:99], s[34:35] op_sel_hi:[1,0]
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
	v_lshl_add_u64 v[106:107], v[130:131], 0, v[230:231]
	v_lshl_add_u64 v[108:109], v[130:131], 0, v[198:199]
	v_accvgpr_read_b32 v78, a112
	v_accvgpr_read_b32 v83, a109
	v_accvgpr_read_b32 v84, a110
	v_accvgpr_read_b32 v85, a111
	v_accvgpr_read_b32 v86, a104
	v_accvgpr_read_b32 v90, a100
	v_accvgpr_read_b32 v96, a98
	v_accvgpr_read_b32 v97, a99
	v_pk_mul_f32 v[122:123], v[148:149], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[124:125], v[146:147], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[94:95], v[94:95], s[34:35] op_sel_hi:[1,0]
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
	v_pk_mul_f32 v[96:97], v[96:97], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[84:85], v[84:85], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[82:83], v[82:83], s[34:35] op_sel_hi:[1,0]
	global_store_short_d16_hi v[98:99], v94, off
	global_store_short_d16_hi v[100:101], v95, off
	v_lshl_add_u64 v[94:95], v[130:131], 0, v[204:205]
	v_lshl_add_u64 v[102:103], v[130:131], 0, v[206:207]
	v_accvgpr_read_b32 v67, a125
	v_accvgpr_read_b32 v70, a120
	v_accvgpr_read_b32 v74, a116
	v_accvgpr_read_b32 v80, a114
	v_accvgpr_read_b32 v81, a115
	v_pk_mul_f32 v[92:93], v[92:93], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[90:91], v[90:91], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[88:89], v[88:89], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[86:87], v[86:87], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[78:79], v[78:79], s[34:35] op_sel_hi:[1,0]
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
	v_pk_mul_f32 v[80:81], v[80:81], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[66:67], v[66:67], s[34:35] op_sel_hi:[1,0]
	global_store_short_d16_hi v[82:83], v78, off
	global_store_short_d16_hi v[84:85], v79, off
	v_lshl_add_u64 v[78:79], v[130:131], 0, v[212:213]
	v_lshl_add_u64 v[86:87], v[130:131], 0, v[214:215]
	v_pk_mul_f32 v[76:77], v[76:77], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[74:75], v[74:75], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[72:73], v[72:73], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[70:71], v[70:71], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[68:69], v[68:69], s[34:35] op_sel_hi:[1,0]
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
	v_lshl_add_u64 v[66:67], v[66:67], 1, s[44:45]
	v_accvgpr_read_b32 v46, a208
	v_accvgpr_read_b32 v51, a205
	v_accvgpr_read_b32 v54, a200
	v_accvgpr_read_b32 v58, a196
	v_accvgpr_read_b32 v64, a194
	v_accvgpr_read_b32 v65, a195
	v_pk_mul_f32 v[62:63], v[62:63], s[34:35] op_sel_hi:[1,0]
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
	v_pk_mul_f32 v[64:65], v[64:65], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], s[34:35] op_sel_hi:[1,0]
	global_store_short_d16_hi v[0:1], v62, off
	global_store_short_d16_hi v[68:69], v63, off
	v_lshl_add_u64 v[62:63], v[66:67], 0, v[228:229]
	v_lshl_add_u64 v[70:71], v[66:67], 0, v[190:191]
	v_accvgpr_read_b32 v30, a224
	v_accvgpr_read_b32 v35, a221
	v_accvgpr_read_b32 v38, a216
	v_accvgpr_read_b32 v42, a212
	v_accvgpr_read_b32 v48, a210
	v_accvgpr_read_b32 v49, a211
	v_pk_mul_f32 v[60:61], v[60:61], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], s[34:35] op_sel_hi:[1,0]
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
	v_pk_mul_f32 v[48:49], v[48:49], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], s[34:35] op_sel_hi:[1,0]
	global_store_short_d16_hi v[0:1], v46, off
	global_store_short_d16_hi v[50:51], v47, off
	v_lshl_add_u64 v[46:47], v[66:67], 0, v[230:231]
	v_lshl_add_u64 v[52:53], v[66:67], 0, v[198:199]
	v_accvgpr_read_b32 v14, a240
	v_accvgpr_read_b32 v19, a237
	v_accvgpr_read_b32 v22, a232
	v_accvgpr_read_b32 v26, a228
	v_accvgpr_read_b32 v32, a226
	v_accvgpr_read_b32 v33, a227
	v_pk_mul_f32 v[44:45], v[44:45], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], s[34:35] op_sel_hi:[1,0]
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
	v_pk_mul_f32 v[32:33], v[32:33], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], s[34:35] op_sel_hi:[1,0]
	global_store_short_d16_hi v[0:1], v30, off
	global_store_short_d16_hi v[34:35], v31, off
	v_lshl_add_u64 v[30:31], v[66:67], 0, v[204:205]
	v_lshl_add_u64 v[36:37], v[66:67], 0, v[206:207]
	v_accvgpr_read_b32 v6, a248
	v_accvgpr_read_b32 v10, a244
	v_accvgpr_read_b32 v16, a242
	v_accvgpr_read_b32 v17, a243
	v_pk_mul_f32 v[28:29], v[28:29], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], s[34:35] op_sel_hi:[1,0]
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
	v_pk_mul_f32 v[16:17], v[16:17], s[34:35] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v3, a253
	v_accvgpr_read_b32 v4, a254
	v_accvgpr_read_b32 v5, a255
	global_store_short_d16_hi v[0:1], v14, off
	global_store_short_d16_hi v[18:19], v15, off
	v_lshl_add_u64 v[14:15], v[66:67], 0, v[212:213]
	v_lshl_add_u64 v[20:21], v[66:67], 0, v[214:215]
	v_pk_mul_f32 v[12:13], v[12:13], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], s[34:35] op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], s[34:35] op_sel_hi:[1,0]
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
		.amdhsa_kernarg_size 248
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
	.set _Z22mxfp4_gluon_cpp_kernel13gluon_globals.numbered_sgpr, 94
	.set _Z22mxfp4_gluon_cpp_kernel13gluon_globals.private_seg_size, 0
	.set _Z22mxfp4_gluon_cpp_kernel13gluon_globals.uses_vcc, 0
	.set _Z22mxfp4_gluon_cpp_kernel13gluon_globals.uses_flat_scratch, 0
	.set _Z22mxfp4_gluon_cpp_kernel13gluon_globals.has_dyn_sized_stack, 0
	.set _Z22mxfp4_gluon_cpp_kernel13gluon_globals.has_recursion, 0
	.set _Z22mxfp4_gluon_cpp_kernel13gluon_globals.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 58060
; TotalNumSgprs: 100
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
	.type	__hip_cuid_c23cf5e16dfd5ed9,@object ; @__hip_cuid_c23cf5e16dfd5ed9
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_c23cf5e16dfd5ed9
__hip_cuid_c23cf5e16dfd5ed9:
	.byte	0                               ; 0x0
	.size	__hip_cuid_c23cf5e16dfd5ed9, 1

	.ident	"clang version 20.0.0git (https://github.com/ROCm/llvm-project.git 27682a16360e33e37c4f3cc6adf9a620733f8fe1)"
	.ident	"AMD clang version 20.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.1.0 25425 1b0eada6b0ee93e2e694c8c146d23fca90bc11c5)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_c23cf5e16dfd5ed9
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     256
    .args:
      - .offset:         0
        .size:           248
        .value_kind:     by_value
    .group_segment_fixed_size: 131072
    .kernarg_segment_align: 8
    .kernarg_segment_size: 248
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _Z22mxfp4_gluon_cpp_kernel13gluon_globals
    .private_segment_fixed_size: 0
    .sgpr_count:     100
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
