	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.text
	.protected	_Z21mxfp4_colfirst_kernel16colfirst_globals ; -- Begin function _Z21mxfp4_colfirst_kernel16colfirst_globals
	.globl	_Z21mxfp4_colfirst_kernel16colfirst_globals
	.p2align	8
	.type	_Z21mxfp4_colfirst_kernel16colfirst_globals,@function
_Z21mxfp4_colfirst_kernel16colfirst_globals: ; @_Z21mxfp4_colfirst_kernel16colfirst_globals
; %bb.0:
	s_ashr_i32 s3, s2, 31
	s_lshr_b32 s3, s3, 27
	s_load_dwordx2 s[20:21], s[0:1], 0x60
	s_load_dword s22, s[0:1], 0x80
	s_load_dwordx2 s[40:41], s[0:1], 0x90
	s_load_dword s33, s[0:1], 0xb0
	s_load_dwordx2 s[34:35], s[0:1], 0xc0
	s_load_dwordx2 s[48:49], s[0:1], 0xe0
	s_load_dwordx4 s[4:7], s[0:1], 0xf0
	s_add_i32 s3, s2, s3
	s_waitcnt lgkmcnt(0)
	s_ashr_i32 s6, s3, 5
	v_lshrrev_b32_e32 v1, 7, v0
	s_lshl_b32 s58, s6, 8
	v_lshl_or_b32 v2, v1, 6, s58
	v_ashrrev_i32_e32 v5, 5, v2
	s_mov_b32 s11, 0x110000
	v_or_b32_e32 v4, 0x80, v2
	v_ashrrev_i32_e32 v4, 5, v4
	s_mov_b32 s10, -1
	v_mul_lo_u32 v2, v5, s22
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], s[20:21], 0, v[2:3]
	s_andn2_b32 s3, s3, 31
	v_readfirstlane_b32 s9, v3
	v_readfirstlane_b32 s8, v2
	v_bfe_u32 v166, v0, 6, 1
	s_mov_b64 s[30:31], s[10:11]
	s_mov_b64 s[28:29], s[8:9]
	v_mul_lo_u32 v2, v4, s22
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], s[20:21], 0, v[2:3]
	s_mov_b64 s[14:15], s[10:11]
	s_mov_b64 s[12:13], s[8:9]
	v_readfirstlane_b32 s16, v3
	s_mov_b32 s13, s16
	s_mov_b64 s[18:19], s[10:11]
	s_mov_b64 s[16:17], s[8:9]
	v_readfirstlane_b32 s7, v2
	s_mov_b32 s12, s7
	s_load_dwordx2 s[44:45], s[0:1], 0x0
	s_load_dword s82, s[0:1], 0x20
	v_lshrrev_b32_e32 v6, 6, v0
	v_or_b32_e32 v2, 1, v5
	v_mul_lo_u32 v2, v2, s22
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], s[20:21], 0, v[2:3]
	s_movk_i32 s84, 0x70
	v_readfirstlane_b32 s23, v3
	s_mov_b32 s17, s23
	v_readfirstlane_b32 s7, v2
	s_mov_b32 s16, s7
	s_load_dword s83, s[0:1], 0x50
	s_load_dwordx2 s[56:57], s[0:1], 0x30
	v_or_b32_e32 v2, 1, v4
	v_mul_lo_u32 v2, v2, s22
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], s[20:21], 0, v[2:3]
	s_mov_b64 s[22:23], s[10:11]
	s_mov_b64 s[20:21], s[8:9]
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s71, s58, s82
	s_mov_b32 s59, s71
	;;#ASMSTART
	;;#ASMEND
	v_readfirstlane_b32 s24, v3
	s_mov_b32 s21, s24
	s_mov_b64 s[26:27], s[10:11]
	s_mov_b64 s[24:25], s[8:9]
	v_readfirstlane_b32 s7, v2
	s_mov_b32 s20, s7
	s_sub_i32 s7, s2, s3
	s_lshl_b32 s63, s7, 8
	s_mov_b32 s60, 0x8000
	s_mul_i32 s77, s63, s83
	v_lshl_or_b32 v2, v166, 6, s63
	v_ashrrev_i32_e32 v7, 5, v2
	v_mul_lo_u32 v2, v7, s33
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[4:5], s[40:41], 0, v[2:3]
	s_mov_b32 s64, 0x10000
	v_readfirstlane_b32 s3, v5
	s_mov_b32 s25, s3
	v_readfirstlane_b32 s2, v4
	s_mov_b32 s24, s2
	s_lshl_b32 s2, s33, 2
	v_add_u32_e32 v2, s2, v2
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], s[40:41], 0, v[2:3]
	s_mov_b32 s67, 0x18000
	v_readfirstlane_b32 s36, v3
	s_mov_b32 s29, s36
	s_mov_b64 s[38:39], s[10:11]
	s_mov_b64 s[36:37], s[8:9]
	v_readfirstlane_b32 s3, v2
	s_mov_b32 s28, s3
	s_movk_i32 s70, 0x4000
	v_or_b32_e32 v2, 1, v7
	v_mul_lo_u32 v2, v2, s33
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[4:5], s[40:41], 0, v[2:3]
	v_add_u32_e32 v2, s2, v2
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], s[40:41], 0, v[2:3]
	s_mov_b64 s[42:43], s[10:11]
	s_mov_b64 s[40:41], s[8:9]
	s_mov_b32 s73, 0xc000
	v_readfirstlane_b32 s2, v2
	s_mov_b32 s40, s2
	v_readfirstlane_b32 s33, v5
	s_mov_b32 s37, s33
	v_lshlrev_b32_e32 v2, 10, v6
	v_readfirstlane_b32 s3, v4
	s_mov_b32 s36, s3
	v_readfirstlane_b32 s3, v3
	s_mov_b32 s41, s3
	v_readfirstlane_b32 s33, v2
	s_add_i32 s55, s33, 0x18000
	v_lshrrev_b32_e32 v3, 3, v0
	v_or_b32_e32 v4, 0x60, v3
	v_lshlrev_b32_e32 v2, 4, v0
	v_bitop3_b32 v2, v2, s84, v0 bitop3:0x48
	v_mad_u64_u32 v[130:131], s[0:1], v3, s82, v[2:3]
	s_lshl_b32 s0, s82, 5
	s_nop 0
	v_add_u32_e32 v131, s0, v130
	v_add_u32_e32 v138, s0, v131
	v_mad_u64_u32 v[134:135], s[0:1], v3, s83, v[2:3]
	s_add_i32 s49, s33, 0x4000
	v_mad_u64_u32 v[132:133], s[0:1], v4, s82, v[2:3]
	s_lshl_b32 s0, s83, 5
	s_nop 0
	v_add_u32_e32 v133, s0, v134
	v_add_u32_e32 v135, s0, v133
	v_mad_u64_u32 v[136:137], s[0:1], v4, s83, v[2:3]
	s_mov_b64 s[0:1], s[8:9]
	s_mov_b32 s0, s44
	s_add_i32 s50, s33, 0xc000
	s_mov_b64 s[2:3], s[10:11]
	s_mov_b32 s1, s45
	s_mov_b64 s[46:47], s[10:11]
	s_mov_b64 s[44:45], s[8:9]
	s_mov_b32 s44, s56
	s_mov_b32 s56, 0
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 s56, s33
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s56
	s_add_i32 s56, s33, 0x1000
	s_mov_b32 s76, 0x14000
	buffer_load_dwordx4 v130, s[0:3], s59 offen lds
	s_mov_b32 s45, s57
	s_mov_b32 s57, s56
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s57
	s_add_i32 s57, s33, 0x2000
	s_mov_b32 s58, s57
	s_mov_b32 s79, 0x1c000
	buffer_load_dwordx4 v131, s[0:3], s59 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s58
	s_add_i32 s58, s33, 0x3000
	s_mov_b32 s61, s58
	v_lshlrev_b32_e32 v2, 2, v0
	v_and_b32_e32 v137, 0xfc, v2
	v_and_b32_e32 v3, 48, v0
	buffer_load_dwordx4 v138, s[0:3], s59 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s61
	s_add_i32 s51, s33, 0x8000
	buffer_load_dwordx4 v132, s[0:3], s59 offen lds
	s_lshl_b32 s59, s82, 7
	s_add_i32 s59, s71, s59
	s_add_i32 s78, s59, 0x80
	s_add_i32 s52, s33, 0x14000
	s_mov_b32 s65, s59
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 s60, s51
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s60
	s_add_i32 s60, s33, 0x9000
	s_mov_b32 s61, s60
	v_lshlrev_b32_e32 v4, 7, v0
	buffer_load_dwordx4 v130, s[0:3], s65 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s61
	s_add_i32 s61, s33, 0xa000
	s_mov_b32 s62, s61
	s_add_i32 s53, s33, 0x10000
	s_mov_b32 s63, s53
	buffer_load_dwordx4 v131, s[0:3], s65 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s62
	s_add_i32 s62, s33, 0xb000
	s_mov_b32 s66, s62
	s_add_i32 s54, s33, 0x1c000
	buffer_load_dwordx4 v138, s[0:3], s65 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s66
	s_mov_b32 s66, s77
	v_lshlrev_b32_e32 v2, 13, v1
	buffer_load_dwordx4 v132, s[0:3], s65 offen lds
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s63
	s_add_i32 s63, s53, 0x1000
	s_mov_b32 s64, s63
	v_lshlrev_b32_e32 v7, 3, v0
	buffer_load_dwordx4 v134, s[44:47], s66 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s64
	s_add_i32 s64, s53, 0x2000
	s_mov_b32 s65, s64
	s_mul_i32 s82, s6, s82
	s_mov_b32 s85, 0
	buffer_load_dwordx4 v133, s[44:47], s66 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s65
	s_add_i32 s65, s53, 0x3000
	s_mov_b32 s68, s65
	s_lshl_b32 s82, s82, 8
	buffer_load_dwordx4 v135, s[44:47], s66 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s68
	v_accvgpr_write_b32 a195, 0
	buffer_load_dwordx4 v136, s[44:47], s66 offen lds
	s_lshl_b32 s66, s83, 7
	s_add_i32 s66, s77, s66
	s_add_i32 s86, s66, 0x80
	s_mul_i32 s83, s7, s83
	s_lshl_b32 s83, s83, 8
	s_mov_b32 s72, s66
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 s67, s55
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s67
	s_add_i32 s67, s55, 0x1000
	s_mov_b32 s68, s67
	buffer_load_dwordx4 v134, s[44:47], s72 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s68
	s_add_i32 s68, s55, 0x2000
	s_mov_b32 s69, s68
	buffer_load_dwordx4 v133, s[44:47], s72 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s69
	s_add_i32 s69, s55, 0x3000
	s_mov_b32 s74, s69
	buffer_load_dwordx4 v135, s[44:47], s72 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s74
	s_or_b32 s74, s71, 0x80
	buffer_load_dwordx4 v136, s[44:47], s72 offen lds
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 s70, s49
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s70
	s_add_i32 s70, s33, 0x5000
	s_mov_b32 s71, s70
	buffer_load_dwordx4 v130, s[0:3], s74 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s71
	s_add_i32 s71, s33, 0x6000
	s_mov_b32 s72, s71
	buffer_load_dwordx4 v131, s[0:3], s74 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s72
	s_add_i32 s72, s33, 0x7000
	s_mov_b32 s75, s72
	buffer_load_dwordx4 v138, s[0:3], s74 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s75
	v_accvgpr_write_b32 a194, 0
	buffer_load_dwordx4 v132, s[0:3], s74 offen lds
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 s73, s50
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s73
	s_add_i32 s73, s50, 0x1000
	s_mov_b32 s74, s73
	buffer_load_dwordx4 v130, s[0:3], s78 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s74
	s_add_i32 s74, s50, 0x2000
	s_mov_b32 s75, s74
	buffer_load_dwordx4 v131, s[0:3], s78 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s75
	s_add_i32 s75, s50, 0x3000
	s_mov_b32 s80, s75
	buffer_load_dwordx4 v138, s[0:3], s78 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s80
	s_or_b32 s80, s77, 0x80
	buffer_load_dwordx4 v132, s[0:3], s78 offen lds
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 s76, s52
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s76
	s_add_i32 s76, s52, 0x1000
	s_mov_b32 s77, s76
	buffer_load_dwordx4 v134, s[44:47], s80 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s77
	s_add_i32 s77, s52, 0x2000
	s_mov_b32 s78, s77
	buffer_load_dwordx4 v133, s[44:47], s80 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s78
	s_add_i32 s78, s52, 0x3000
	s_mov_b32 s81, s78
	buffer_load_dwordx4 v135, s[44:47], s80 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s81
	v_accvgpr_write_b32 a193, 0
	buffer_load_dwordx4 v136, s[44:47], s80 offen lds
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 s79, s54
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s79
	s_add_i32 s79, s54, 0x1000
	s_mov_b32 s80, s79
	buffer_load_dwordx4 v134, s[44:47], s86 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s80
	s_add_i32 s80, s54, 0x2000
	s_mov_b32 s81, s80
	buffer_load_dwordx4 v133, s[44:47], s86 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s81
	s_add_i32 s81, s54, 0x3000
	s_mov_b32 s87, s81
	buffer_load_dwordx4 v135, s[44:47], s86 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s87
	v_accvgpr_write_b32 a192, 0
	buffer_load_dwordx4 v136, s[44:47], s86 offen lds
	buffer_load_dword v161, v137, s[8:11], 0 offen
	buffer_load_dword v155, v137, s[12:15], 0 offen
	buffer_load_dword v162, v137, s[16:19], 0 offen
	buffer_load_dword v156, v137, s[20:23], 0 offen
	buffer_load_dword v160, v137, s[24:27], 0 offen
	buffer_load_dword v158, v137, s[28:31], 0 offen
	buffer_load_dword v159, v137, s[36:39], 0 offen
	buffer_load_dword v157, v137, s[40:43], 0 offen
	s_movk_i32 s86, 0x780
	v_and_or_b32 v3, v4, s86, v3
	v_or_b32_e32 v6, v3, v2
	v_lshlrev_b32_e32 v4, 13, v166
	v_or_b32_e32 v8, 64, v6
	v_bitop3_b32 v140, v7, v8, s84 bitop3:0x6c
	v_or_b32_e32 v8, v4, v3
	v_or_b32_e32 v9, 0x10000, v8
	v_bitop3_b32 v141, v7, v9, s84 bitop3:0x6c
	v_or_b32_e32 v9, 0x10040, v8
	v_bitop3_b32 v142, v7, v9, s84 bitop3:0x6c
	v_or_b32_e32 v9, 0x18000, v4
	v_or_b32_e32 v5, 64, v3
	v_or_b32_e32 v10, v3, v9
	v_lshrrev_b32_e32 v11, 4, v10
	v_or_b32_e32 v9, v5, v9
	v_bitop3_b32 v143, v11, v10, s84 bitop3:0x6c
	v_lshrrev_b32_e32 v10, 4, v9
	v_bitop3_b32 v144, v10, v9, s84 bitop3:0x6c
	v_or_b32_e32 v9, 0x8000, v2
	v_or_b32_e32 v10, v3, v9
	v_lshrrev_b32_e32 v11, 4, v10
	v_bitop3_b32 v145, v11, v10, s84 bitop3:0x6c
	v_or_b32_e32 v9, v5, v9
	v_lshrrev_b32_e32 v10, 4, v9
	v_bitop3_b32 v139, v7, v6, s84 bitop3:0x6c
	v_bitop3_b32 v146, v10, v9, s84 bitop3:0x6c
	v_or_b32_e32 v9, 0x4000, v6
	v_or_b32_e32 v6, 0x4040, v6
	v_bitop3_b32 v148, v7, v6, s84 bitop3:0x6c
	v_or_b32_e32 v6, 0x14000, v8
	v_bitop3_b32 v149, v7, v6, s84 bitop3:0x6c
	v_or_b32_e32 v6, 0x14040, v8
	v_or_b32_e32 v4, 0x1c000, v4
	v_bitop3_b32 v150, v7, v6, s84 bitop3:0x6c
	v_or_b32_e32 v6, v3, v4
	v_bitop3_b32 v147, v7, v9, s84 bitop3:0x6c
	v_lshrrev_b32_e32 v7, 4, v6
	v_or_b32_e32 v4, v5, v4
	v_or_b32_e32 v2, 0xc000, v2
	v_bitop3_b32 v151, v7, v6, s84 bitop3:0x6c
	v_lshrrev_b32_e32 v6, 4, v4
	v_or_b32_e32 v3, v3, v2
	v_bitop3_b32 v152, v6, v4, s84 bitop3:0x6c
	v_lshrrev_b32_e32 v4, 4, v3
	v_or_b32_e32 v2, v5, v2
	v_bitop3_b32 v153, v4, v3, s84 bitop3:0x6c
	v_lshrrev_b32_e32 v3, 4, v2
	v_bitop3_b32 v154, v3, v2, s84 bitop3:0x6c
	s_movk_i32 s84, 0x200
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
.LBB0_1:                                ; =>This Inner Loop Header: Depth=1
	;;#ASMSTART
	s_waitcnt vmcnt(16)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[98:101], v139 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[102:105], v139 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[106:109], v139 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[110:113], v139 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[114:117], v140 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[118:121], v140 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[122:125], v140 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[126:129], v140 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[90:93], v141 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[82:85], v141 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[10:13], v141 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[2:5], v141 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[94:97], v142 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[86:89], v142 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[22:25], v142 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[6:9], v142 offset:0x1800

	;;#ASMEND
	s_add_i32 s87, s84, 0xffffff00
	buffer_load_dword v170, v137, s[8:11], s87 offen
	buffer_load_dword v163, v137, s[12:15], s87 offen
	buffer_load_dword v171, v137, s[16:19], s87 offen
	buffer_load_dword v164, v137, s[20:23], s87 offen
	buffer_load_dword v169, v137, s[24:27], s87 offen
	buffer_load_dword v167, v137, s[28:31], s87 offen
	buffer_load_dword v168, v137, s[36:39], s87 offen
	buffer_load_dword v165, v137, s[40:43], s87 offen
	s_min_u32 s87, s85, 29
	s_lshl_b32 s87, s87, 7
	s_addk_i32 s87, 0x100
	s_add_i32 s90, s87, s82
	s_add_i32 s89, s87, s59
	s_add_i32 s88, s87, s83
	s_add_i32 s87, s87, s66
	s_mov_b32 s91, s33
	;;#ASMSTART
	s_waitcnt lgkmcnt(0) vmcnt(8)
	;;#ASMEND
	s_waitcnt vmcnt(9)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[98:101], v[90:93], a[28:31],  v161, v160 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[114:117], v[94:97], a[28:31],  v161, v160 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[102:105], v[90:93], a[44:47],  v161, v160 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[118:121], v[94:97], a[44:47],  v161, v160 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[106:109], v[90:93], a[60:63],  v162, v160 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[122:125], v[94:97], a[60:63],  v162, v160 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[110:113], v[90:93], a[76:79], v162, v160 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[126:129], v[94:97], a[76:79], v162, v160 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[74:77], v143 offset:0
ds_read_b128 v[34:37], v143 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[98:101], v[82:85], a[32:35],  v161, v160 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[114:117], v[86:89], a[32:35],  v161, v160 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[102:105], v[82:85], a[48:51],  v161, v160 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[118:121], v[86:89], a[48:51],  v161, v160 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[106:109], v[82:85], a[64:67],  v162, v160 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[122:125], v[86:89], a[64:67],  v162, v160 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[110:113], v[82:85], a[80:83], v162, v160 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[126:129], v[86:89], a[80:83], v162, v160 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[26:29], v143 offset:4096
ds_read_b128 v[14:17], v143 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[98:101], v[10:13], a[36:39],  v161, v159 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[114:117], v[22:25], a[36:39],  v161, v159 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[102:105], v[10:13], a[52:55],  v161, v159 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[118:121], v[22:25], a[52:55],  v161, v159 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[106:109], v[10:13], a[68:71], v162, v159 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[122:125], v[22:25], a[68:71], v162, v159 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[110:113], v[10:13], a[84:87], v162, v159 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[126:129], v[22:25], a[84:87], v162, v159 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[78:81], v144 offset:0
ds_read_b128 v[38:41], v144 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[98:101], v[2:5], a[40:43],  v161, v159 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[114:117], v[6:9], a[40:43],  v161, v159 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[102:105], v[2:5], a[56:59],  v161, v159 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[118:121], v[6:9], a[56:59],  v161, v159 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[106:109], v[2:5], a[72:75], v162, v159 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[122:125], v[6:9], a[72:75], v162, v159 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[110:113], v[2:5], a[88:91], v162, v159 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[126:129], v[6:9], a[88:91], v162, v159 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v144 offset:4096
ds_read_b128 v[18:21], v144 offset:6144

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_add_i32 s86, s85, 1
	s_waitcnt vmcnt(8)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3],  v[98:101], v[74:77], a[0:3],  v161, v158 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3],  v[114:117], v[78:81], a[0:3],  v161, v158 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19],  v[102:105], v[74:77], a[16:19],  v161, v158 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19],  v[118:121], v[78:81], a[16:19],  v161, v158 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99],  v[106:109], v[74:77], a[96:99],  v162, v158 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99],  v[122:125], v[78:81], a[96:99],  v162, v158 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[110:113], v[74:77], a[112:115], v162, v158 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[126:129], v[78:81], a[112:115], v162, v158 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v145 offset:0
ds_read_b128 v[46:49], v145 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7],  v[98:101], v[34:37], a[4:7],  v161, v158 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7],  v[114:117], v[38:41], a[4:7],  v161, v158 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23],  v[102:105], v[34:37], a[20:23],  v161, v158 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23],  v[118:121], v[38:41], a[20:23],  v161, v158 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103],  v[106:109], v[34:37], a[100:103],  v162, v158 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103],  v[122:125], v[38:41], a[100:103],  v162, v158 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[110:113], v[34:37], a[116:119], v162, v158 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[126:129], v[38:41], a[116:119], v162, v158 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v145 offset:4096
ds_read_b128 v[54:57], v145 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11],  v[98:101], v[26:29], a[8:11],  v161, v157 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11],  v[114:117], v[30:33], a[8:11],  v161, v157 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27],  v[102:105], v[26:29], a[24:27],  v161, v157 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27],  v[118:121], v[30:33], a[24:27],  v161, v157 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[106:109], v[26:29], a[104:107], v162, v157 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[122:125], v[30:33], a[104:107], v162, v157 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[110:113], v[26:29], a[120:123], v162, v157 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[126:129], v[30:33], a[120:123], v162, v157 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v146 offset:0
ds_read_b128 v[62:65], v146 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15],  v[98:101], v[14:17], a[12:15],  v161, v157 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15],  v[114:117], v[18:21], a[12:15],  v161, v157 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95],  v[102:105], v[14:17], a[92:95],  v161, v157 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95],  v[118:121], v[18:21], a[92:95],  v161, v157 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[106:109], v[14:17], a[108:111], v162, v157 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[122:125], v[18:21], a[108:111], v162, v157 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[110:113], v[14:17], a[124:127], v162, v157 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[126:129], v[18:21], a[124:127], v162, v157 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[66:69], v146 offset:4096
ds_read_b128 v[70:73], v146 offset:6144

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[42:45], v[90:93], a[128:131],  v155, v160 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[58:61], v[94:97], a[128:131],  v155, v160 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[46:49], v[90:93], a[144:147],  v155, v160 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[62:65], v[94:97], a[144:147],  v155, v160 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[50:53], v[90:93], a[160:163],  v156, v160 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[66:69], v[94:97], a[160:163],  v156, v160 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[54:57], v[90:93], a[176:179], v156, v160 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[70:73], v[94:97], a[176:179], v156, v160 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s91
	s_mov_b32 s91, s56
	buffer_load_dwordx4 v130, s[0:3], s90 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s91
	s_mov_b32 s91, s57
	buffer_load_dwordx4 v131, s[0:3], s90 offen lds
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[42:45], v[82:85], a[132:135],  v155, v160 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[58:61], v[86:89], a[132:135],  v155, v160 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[46:49], v[82:85], a[148:151],  v155, v160 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[62:65], v[86:89], a[148:151],  v155, v160 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[50:53], v[82:85], a[164:167],  v156, v160 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[66:69], v[86:89], a[164:167],  v156, v160 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[54:57], v[82:85], a[180:183], v156, v160 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[70:73], v[86:89], a[180:183], v156, v160 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s91
	s_mov_b32 s91, s58
	buffer_load_dwordx4 v138, s[0:3], s90 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s91
	s_nop 0
	buffer_load_dwordx4 v132, s[0:3], s90 offen lds
	s_mov_b32 s90, s51
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[42:45], v[10:13], a[136:139],  v155, v159 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[58:61], v[22:25], a[136:139],  v155, v159 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[46:49], v[10:13], a[152:155],  v155, v159 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[62:65], v[22:25], a[152:155],  v155, v159 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[50:53], v[10:13], a[168:171], v156, v159 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[66:69], v[22:25], a[168:171], v156, v159 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[54:57], v[10:13], a[184:187], v156, v159 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[70:73], v[22:25], a[184:187], v156, v159 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s90
	s_mov_b32 s90, s60
	buffer_load_dwordx4 v130, s[0:3], s89 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s90
	s_mov_b32 s90, s61
	buffer_load_dwordx4 v131, s[0:3], s89 offen lds
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[42:45], v[2:5], a[140:143],  v155, v159 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[58:61], v[6:9], a[140:143],  v155, v159 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[46:49], v[2:5], a[156:159],  v155, v159 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[62:65], v[6:9], a[156:159],  v155, v159 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[50:53], v[2:5], a[172:175], v156, v159 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[66:69], v[6:9], a[172:175], v156, v159 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[54:57], v[2:5], a[188:191], v156, v159 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[70:73], v[6:9], a[188:191], v156, v159 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s90
	s_mov_b32 s90, s62
	buffer_load_dwordx4 v138, s[0:3], s89 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s90
	s_mov_b32 s90, s49
	buffer_load_dwordx4 v132, s[0:3], s89 offen lds
	s_mov_b32 s89, s53
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[42:45], v[74:77], a[192:195],  v155, v158 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[58:61], v[78:81], a[192:195],  v155, v158 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[46:49], v[74:77], a[208:211],  v155, v158 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[62:65], v[78:81], a[208:211],  v155, v158 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[50:53], v[74:77], a[224:227],  v156, v158 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[66:69], v[78:81], a[224:227],  v156, v158 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[54:57], v[74:77], a[240:243], v156, v158 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[70:73], v[78:81], a[240:243], v156, v158 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s89
	s_mov_b32 s89, s63
	buffer_load_dwordx4 v134, s[44:47], s88 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s89
	s_mov_b32 s89, s64
	buffer_load_dwordx4 v133, s[44:47], s88 offen lds
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[42:45], v[34:37], a[196:199],  v155, v158 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[58:61], v[38:41], a[196:199],  v155, v158 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[46:49], v[34:37], a[212:215],  v155, v158 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[62:65], v[38:41], a[212:215],  v155, v158 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[50:53], v[34:37], a[228:231],  v156, v158 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[66:69], v[38:41], a[228:231],  v156, v158 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[54:57], v[34:37], a[244:247], v156, v158 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[70:73], v[38:41], a[244:247], v156, v158 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s89
	s_mov_b32 s89, s65
	buffer_load_dwordx4 v135, s[44:47], s88 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s89
	s_nop 0
	buffer_load_dwordx4 v136, s[44:47], s88 offen lds
	s_mov_b32 s88, s55
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[42:45], v[26:29], a[200:203],  v155, v157 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[58:61], v[30:33], a[200:203],  v155, v157 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[46:49], v[26:29], a[216:219],  v155, v157 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[62:65], v[30:33], a[216:219],  v155, v157 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[50:53], v[26:29], a[232:235], v156, v157 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[66:69], v[30:33], a[232:235], v156, v157 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[54:57], v[26:29], a[248:251], v156, v157 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[70:73], v[30:33], a[248:251], v156, v157 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s88
	s_mov_b32 s88, s67
	buffer_load_dwordx4 v134, s[44:47], s87 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s88
	s_mov_b32 s88, s68
	buffer_load_dwordx4 v133, s[44:47], s87 offen lds
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[42:45], v[14:17], a[204:207],  v155, v157 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[58:61], v[18:21], a[204:207],  v155, v157 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[46:49], v[14:17], a[220:223],  v155, v157 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[62:65], v[18:21], a[220:223],  v155, v157 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[50:53], v[14:17], a[236:239], v156, v157 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[66:69], v[18:21], a[236:239], v156, v157 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[54:57], v[14:17], a[252:255], v156, v157 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[70:73], v[18:21], a[252:255], v156, v157 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s88
	s_mov_b32 s88, s69
	buffer_load_dwordx4 v135, s[44:47], s87 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s88
	s_nop 0
	buffer_load_dwordx4 v136, s[44:47], s87 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(16)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[98:101], v147 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[102:105], v147 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[106:109], v147 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[110:113], v147 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[114:117], v148 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[118:121], v148 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[122:125], v148 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[126:129], v148 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[90:93], v149 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[58:61], v149 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[18:21], v149 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[2:5], v149 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[94:97], v150 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[62:65], v150 offset:0x800

	;;#ASMEND
	s_add_i32 s87, s85, 2
	;;#ASMSTART
	ds_read_b128 v[54:57], v150 offset:0x1000

	;;#ASMEND
	s_cmp_lg_u32 s85, 30
	;;#ASMSTART
	ds_read_b128 v[6:9], v150 offset:0x1800

	;;#ASMEND
	s_cselect_b32 s85, s84, 0x1f00
	buffer_load_dword v161, v137, s[8:11], s85 offen
	buffer_load_dword v155, v137, s[12:15], s85 offen
	buffer_load_dword v162, v137, s[16:19], s85 offen
	buffer_load_dword v156, v137, s[20:23], s85 offen
	buffer_load_dword v160, v137, s[24:27], s85 offen
	buffer_load_dword v158, v137, s[28:31], s85 offen
	buffer_load_dword v159, v137, s[36:39], s85 offen
	buffer_load_dword v157, v137, s[40:43], s85 offen
	s_min_u32 s85, s86, 29
	s_lshl_b32 s85, s85, 7
	s_addk_i32 s85, 0x100
	s_add_i32 s89, s85, s82
	s_add_i32 s88, s85, s59
	s_add_i32 s86, s85, s83
	s_add_i32 s85, s85, s66
	;;#ASMSTART
	s_waitcnt lgkmcnt(0) vmcnt(8)
	;;#ASMEND
	s_waitcnt vmcnt(25)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[98:101], v[90:93], a[28:31],  v170, v169 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[114:117], v[94:97], a[28:31],  v170, v169 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[102:105], v[90:93], a[44:47],  v170, v169 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[118:121], v[94:97], a[44:47],  v170, v169 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[106:109], v[90:93], a[60:63],  v171, v169 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[122:125], v[94:97], a[60:63],  v171, v169 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[110:113], v[90:93], a[76:79], v171, v169 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[126:129], v[94:97], a[76:79], v171, v169 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[82:85], v151 offset:0
ds_read_b128 v[74:77], v151 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[98:101], v[58:61], a[32:35],  v170, v169 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[114:117], v[62:65], a[32:35],  v170, v169 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[102:105], v[58:61], a[48:51],  v170, v169 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[118:121], v[62:65], a[48:51],  v170, v169 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[106:109], v[58:61], a[64:67],  v171, v169 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[122:125], v[62:65], a[64:67],  v171, v169 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[110:113], v[58:61], a[80:83], v171, v169 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[126:129], v[62:65], a[80:83], v171, v169 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[66:69], v151 offset:4096
ds_read_b128 v[10:13], v151 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[98:101], v[18:21], a[36:39],  v170, v168 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[114:117], v[54:57], a[36:39],  v170, v168 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[102:105], v[18:21], a[52:55],  v170, v168 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[118:121], v[54:57], a[52:55],  v170, v168 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[106:109], v[18:21], a[68:71], v171, v168 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[122:125], v[54:57], a[68:71], v171, v168 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[110:113], v[18:21], a[84:87], v171, v168 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[126:129], v[54:57], a[84:87], v171, v168 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[86:89], v152 offset:0
ds_read_b128 v[78:81], v152 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[98:101], v[2:5], a[40:43],  v170, v168 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[114:117], v[6:9], a[40:43],  v170, v168 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[102:105], v[2:5], a[56:59],  v170, v168 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[118:121], v[6:9], a[56:59],  v170, v168 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[106:109], v[2:5], a[72:75], v171, v168 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[122:125], v[6:9], a[72:75], v171, v168 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[110:113], v[2:5], a[88:91], v171, v168 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[126:129], v[6:9], a[88:91], v171, v168 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[70:73], v152 offset:4096
ds_read_b128 v[14:17], v152 offset:6144

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_addk_i32 s84, 0x200
	s_waitcnt vmcnt(24)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3],  v[98:101], v[82:85], a[0:3],  v170, v167 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3],  v[114:117], v[86:89], a[0:3],  v170, v167 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19],  v[102:105], v[82:85], a[16:19],  v170, v167 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19],  v[118:121], v[86:89], a[16:19],  v170, v167 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99],  v[106:109], v[82:85], a[96:99],  v171, v167 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99],  v[122:125], v[86:89], a[96:99],  v171, v167 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[110:113], v[82:85], a[112:115], v171, v167 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[126:129], v[86:89], a[112:115], v171, v167 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[22:25], v153 offset:0
ds_read_b128 v[26:29], v153 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7],  v[98:101], v[74:77], a[4:7],  v170, v167 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7],  v[114:117], v[78:81], a[4:7],  v170, v167 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23],  v[102:105], v[74:77], a[20:23],  v170, v167 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23],  v[118:121], v[78:81], a[20:23],  v170, v167 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103],  v[106:109], v[74:77], a[100:103],  v171, v167 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103],  v[122:125], v[78:81], a[100:103],  v171, v167 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[110:113], v[74:77], a[116:119], v171, v167 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[126:129], v[78:81], a[116:119], v171, v167 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v153 offset:4096
ds_read_b128 v[34:37], v153 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11],  v[98:101], v[66:69], a[8:11],  v170, v165 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11],  v[114:117], v[70:73], a[8:11],  v170, v165 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27],  v[102:105], v[66:69], a[24:27],  v170, v165 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27],  v[118:121], v[70:73], a[24:27],  v170, v165 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[106:109], v[66:69], a[104:107], v171, v165 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[122:125], v[70:73], a[104:107], v171, v165 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[110:113], v[66:69], a[120:123], v171, v165 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[126:129], v[70:73], a[120:123], v171, v165 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v154 offset:0
ds_read_b128 v[42:45], v154 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15],  v[98:101], v[10:13], a[12:15],  v170, v165 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15],  v[114:117], v[14:17], a[12:15],  v170, v165 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95],  v[102:105], v[10:13], a[92:95],  v170, v165 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95],  v[118:121], v[14:17], a[92:95],  v170, v165 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[106:109], v[10:13], a[108:111], v171, v165 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[122:125], v[14:17], a[108:111], v171, v165 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[110:113], v[10:13], a[124:127], v171, v165 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[126:129], v[14:17], a[124:127], v171, v165 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v154 offset:4096
ds_read_b128 v[50:53], v154 offset:6144

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[22:25], v[90:93], a[128:131],  v163, v169 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[38:41], v[94:97], a[128:131],  v163, v169 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[26:29], v[90:93], a[144:147],  v163, v169 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[42:45], v[94:97], a[144:147],  v163, v169 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[30:33], v[90:93], a[160:163],  v164, v169 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[46:49], v[94:97], a[160:163],  v164, v169 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[34:37], v[90:93], a[176:179], v164, v169 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[50:53], v[94:97], a[176:179], v164, v169 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s90
	s_mov_b32 s90, s70
	buffer_load_dwordx4 v130, s[0:3], s89 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s90
	s_mov_b32 s90, s71
	buffer_load_dwordx4 v131, s[0:3], s89 offen lds
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[22:25], v[58:61], a[132:135],  v163, v169 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[38:41], v[62:65], a[132:135],  v163, v169 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[26:29], v[58:61], a[148:151],  v163, v169 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[42:45], v[62:65], a[148:151],  v163, v169 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[30:33], v[58:61], a[164:167],  v164, v169 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[46:49], v[62:65], a[164:167],  v164, v169 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[34:37], v[58:61], a[180:183], v164, v169 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[50:53], v[62:65], a[180:183], v164, v169 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s90
	s_mov_b32 s90, s72
	buffer_load_dwordx4 v138, s[0:3], s89 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s90
	s_cmp_eq_u32 s87, 32
	buffer_load_dwordx4 v132, s[0:3], s89 offen lds
	s_mov_b32 s89, s50
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[22:25], v[18:21], a[136:139],  v163, v168 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[38:41], v[54:57], a[136:139],  v163, v168 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[26:29], v[18:21], a[152:155],  v163, v168 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[42:45], v[54:57], a[152:155],  v163, v168 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[30:33], v[18:21], a[168:171], v164, v168 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[46:49], v[54:57], a[168:171], v164, v168 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[34:37], v[18:21], a[184:187], v164, v168 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[50:53], v[54:57], a[184:187], v164, v168 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s89
	s_mov_b32 s89, s73
	buffer_load_dwordx4 v130, s[0:3], s88 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s89
	s_mov_b32 s89, s74
	buffer_load_dwordx4 v131, s[0:3], s88 offen lds
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[22:25], v[2:5], a[140:143],  v163, v168 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[38:41], v[6:9], a[140:143],  v163, v168 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[26:29], v[2:5], a[156:159],  v163, v168 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[42:45], v[6:9], a[156:159],  v163, v168 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[30:33], v[2:5], a[172:175], v164, v168 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[46:49], v[6:9], a[172:175], v164, v168 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[34:37], v[2:5], a[188:191], v164, v168 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[50:53], v[6:9], a[188:191], v164, v168 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s89
	s_mov_b32 s89, s75
	buffer_load_dwordx4 v138, s[0:3], s88 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s89
	s_nop 0
	buffer_load_dwordx4 v132, s[0:3], s88 offen lds
	s_mov_b32 s88, s52
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[22:25], v[82:85], a[192:195],  v163, v167 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[38:41], v[86:89], a[192:195],  v163, v167 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[26:29], v[82:85], a[208:211],  v163, v167 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[42:45], v[86:89], a[208:211],  v163, v167 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[30:33], v[82:85], a[224:227],  v164, v167 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[46:49], v[86:89], a[224:227],  v164, v167 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[34:37], v[82:85], a[240:243], v164, v167 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[50:53], v[86:89], a[240:243], v164, v167 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s88
	s_mov_b32 s88, s76
	buffer_load_dwordx4 v134, s[44:47], s86 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s88
	s_mov_b32 s88, s77
	buffer_load_dwordx4 v133, s[44:47], s86 offen lds
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[22:25], v[74:77], a[196:199],  v163, v167 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[38:41], v[78:81], a[196:199],  v163, v167 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[26:29], v[74:77], a[212:215],  v163, v167 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[42:45], v[78:81], a[212:215],  v163, v167 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[30:33], v[74:77], a[228:231],  v164, v167 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[46:49], v[78:81], a[228:231],  v164, v167 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[34:37], v[74:77], a[244:247], v164, v167 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[50:53], v[78:81], a[244:247], v164, v167 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s88
	s_mov_b32 s88, s78
	buffer_load_dwordx4 v135, s[44:47], s86 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s88
	s_nop 0
	buffer_load_dwordx4 v136, s[44:47], s86 offen lds
	s_mov_b32 s86, s54
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[22:25], v[66:69], a[200:203],  v163, v165 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[38:41], v[70:73], a[200:203],  v163, v165 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[26:29], v[66:69], a[216:219],  v163, v165 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[42:45], v[70:73], a[216:219],  v163, v165 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[30:33], v[66:69], a[232:235], v164, v165 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[46:49], v[70:73], a[232:235], v164, v165 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[34:37], v[66:69], a[248:251], v164, v165 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[50:53], v[70:73], a[248:251], v164, v165 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s86
	s_mov_b32 s86, s79
	buffer_load_dwordx4 v134, s[44:47], s85 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s86
	s_mov_b32 s86, s80
	buffer_load_dwordx4 v133, s[44:47], s85 offen lds
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[22:25], v[10:13], a[204:207],  v163, v165 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[38:41], v[14:17], a[204:207],  v163, v165 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[26:29], v[10:13], a[220:223],  v163, v165 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[42:45], v[14:17], a[220:223],  v163, v165 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[30:33], v[10:13], a[236:239], v164, v165 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[46:49], v[14:17], a[236:239], v164, v165 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[34:37], v[10:13], a[252:255], v164, v165 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[50:53], v[14:17], a[252:255], v164, v165 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s86
	s_mov_b32 s86, s81
	buffer_load_dwordx4 v135, s[44:47], s85 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s86
	s_nop 0
	buffer_load_dwordx4 v136, s[44:47], s85 offen lds
	s_mov_b32 s85, s87
	s_cbranch_scc0 .LBB0_1
; %bb.2:
	v_lshl_or_b32 v1, s6, 2, v1
	v_accvgpr_read_b32 v231, a31
	v_lshl_or_b32 v2, s7, 2, v166
	v_mul_lo_u32 v3, v1, s48
	v_accvgpr_read_b32 v183, a79
	v_accvgpr_read_b32 v230, a30
	v_accvgpr_read_b32 v229, a29
	v_accvgpr_read_b32 v228, a28
	v_accvgpr_write_b32 a30, v1
	v_add_lshl_u32 v166, v3, v2, 6
	v_lshrrev_b32_e32 v1, 2, v0
	v_accvgpr_read_b32 v171, a91
	v_accvgpr_read_b32 v179, a83
	v_accvgpr_read_b32 v181, a77
	v_accvgpr_read_b32 v180, a76
	v_ashrrev_i32_e32 v167, 31, v166
	v_and_b32_e32 v1, 12, v1
	v_and_b32_e32 v0, 15, v0
	v_accvgpr_read_b32 v169, a89
	v_accvgpr_read_b32 v168, a88
	v_accvgpr_read_b32 v177, a81
	v_accvgpr_read_b32 v176, a80
	v_accvgpr_read_b32 v187, a75
	v_pk_mul_f32 v[250:251], v[180:181], s[4:5] op_sel_hi:[1,0]
	v_lshl_add_u64 v[180:181], v[166:167], 1, s[34:35]
	v_mad_u64_u32 v[166:167], s[0:1], v1, s48, v[0:1]
	v_accvgpr_read_b32 v185, a73
	v_accvgpr_read_b32 v184, a72
	v_pk_mul_f32 v[254:255], v[176:177], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[176:177], v[168:169], s[4:5] op_sel_hi:[1,0]
	v_add_u32_e32 v168, s48, v166
	v_pk_mul_f32 v[246:247], v[184:185], s[4:5] op_sel_hi:[1,0]
	v_add_u32_e32 v184, s48, v168
	v_ashrrev_i32_e32 v167, 31, v166
	v_ashrrev_i32_e32 v169, 31, v168
	v_ashrrev_i32_e32 v185, 31, v184
	v_accvgpr_read_b32 v178, a82
	v_accvgpr_read_b32 v182, a78
	v_lshlrev_b64 v[0:1], 1, v[166:167]
	v_lshlrev_b64 v[166:167], 1, v[168:169]
	v_lshlrev_b64 v[168:169], 1, v[184:185]
	v_add_u32_e32 v184, s48, v184
	v_accvgpr_read_b32 v207, a55
	v_accvgpr_read_b32 v223, a39
	v_accvgpr_read_b32 v227, a35
	v_pk_mul_f32 v[228:229], v[228:229], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[248:249], v[182:183], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[252:253], v[178:179], s[4:5] op_sel_hi:[1,0]
	v_lshl_add_u64 v[178:179], v[180:181], 0, v[0:1]
	v_lshl_add_u64 v[182:183], v[180:181], 0, v[166:167]
	v_ashrrev_i32_e32 v185, 31, v184
	v_accvgpr_read_b32 v186, a74
	v_accvgpr_read_b32 v205, a53
	v_accvgpr_read_b32 v204, a52
	v_accvgpr_read_b32 v219, a43
	v_accvgpr_read_b32 v221, a37
	v_accvgpr_read_b32 v220, a36
	v_accvgpr_read_b32 v225, a33
	v_accvgpr_read_b32 v224, a32
	global_store_short_d16_hi v[178:179], v228, off
	global_store_short_d16_hi v[182:183], v229, off
	v_lshlrev_b64 v[228:229], 1, v[184:185]
	v_accvgpr_read_b32 v218, a42
	v_accvgpr_read_b32 v217, a41
	v_accvgpr_read_b32 v216, a40
	v_accvgpr_read_b32 v222, a38
	v_accvgpr_read_b32 v226, a34
	v_pk_mul_f32 v[230:231], v[230:231], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[224:225], v[224:225], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[220:221], v[220:221], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[234:235], v[204:205], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[244:245], v[186:187], s[4:5] op_sel_hi:[1,0]
	v_lshl_add_u64 v[186:187], v[180:181], 0, v[168:169]
	v_lshl_add_u64 v[204:205], v[180:181], 0, v[228:229]
	s_mul_i32 s0, s48, 13
	v_pk_mul_f32 v[226:227], v[226:227], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[222:223], v[222:223], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[218:219], v[218:219], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[216:217], v[216:217], s[4:5] op_sel_hi:[1,0]
	global_store_short_d16_hi v[186:187], v230, off
	global_store_short_d16_hi v[204:205], v231, off
	global_store_short_d16_hi v[178:179], v224, off offset:32
	global_store_short_d16_hi v[182:183], v225, off offset:32
	global_store_short_d16_hi v[186:187], v226, off offset:32
	global_store_short_d16_hi v[204:205], v227, off offset:32
	global_store_short_d16_hi v[178:179], v220, off offset:64
	global_store_short_d16_hi v[182:183], v221, off offset:64
	global_store_short_d16_hi v[186:187], v222, off offset:64
	global_store_short_d16_hi v[204:205], v223, off offset:64
	global_store_short_d16_hi v[178:179], v216, off offset:96
	global_store_short_d16_hi v[182:183], v217, off offset:96
	global_store_short_d16_hi v[186:187], v218, off offset:96
	global_store_short_d16_hi v[204:205], v219, off offset:96
	v_add_u32_e32 v178, s0, v184
	v_ashrrev_i32_e32 v179, 31, v178
	v_accvgpr_read_b32 v215, a47
	v_lshlrev_b64 v[226:227], 1, v[178:179]
	v_add_u32_e32 v178, s48, v178
	v_accvgpr_read_b32 v213, a45
	v_accvgpr_read_b32 v212, a44
	v_ashrrev_i32_e32 v179, 31, v178
	v_pk_mul_f32 v[212:213], v[212:213], s[4:5] op_sel_hi:[1,0]
	v_lshl_add_u64 v[182:183], v[180:181], 0, v[226:227]
	v_lshlrev_b64 v[224:225], 1, v[178:179]
	v_add_u32_e32 v178, s48, v178
	global_store_short_d16_hi v[182:183], v212, off
	v_lshl_add_u64 v[184:185], v[180:181], 0, v[224:225]
	v_add_u32_e32 v212, s48, v178
	v_accvgpr_read_b32 v211, a51
	global_store_short_d16_hi v[184:185], v213, off
	v_ashrrev_i32_e32 v179, 31, v178
	v_ashrrev_i32_e32 v213, 31, v212
	v_accvgpr_read_b32 v203, a59
	v_accvgpr_read_b32 v209, a49
	v_accvgpr_read_b32 v208, a48
	v_accvgpr_read_b32 v214, a46
	v_lshlrev_b64 v[230:231], 1, v[178:179]
	v_lshlrev_b64 v[178:179], 1, v[212:213]
	v_accvgpr_read_b32 v202, a58
	v_accvgpr_read_b32 v201, a57
	v_accvgpr_read_b32 v200, a56
	v_accvgpr_read_b32 v206, a54
	v_accvgpr_read_b32 v210, a50
	v_pk_mul_f32 v[214:215], v[214:215], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[208:209], v[208:209], s[4:5] op_sel_hi:[1,0]
	v_lshl_add_u64 v[186:187], v[180:181], 0, v[230:231]
	v_lshl_add_u64 v[216:217], v[180:181], 0, v[178:179]
	v_pk_mul_f32 v[210:211], v[210:211], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[232:233], v[206:207], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[236:237], v[202:203], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[238:239], v[200:201], s[4:5] op_sel_hi:[1,0]
	global_store_short_d16_hi v[186:187], v214, off
	global_store_short_d16_hi v[216:217], v215, off
	global_store_short_d16_hi v[182:183], v208, off offset:32
	global_store_short_d16_hi v[184:185], v209, off offset:32
	global_store_short_d16_hi v[186:187], v210, off offset:32
	global_store_short_d16_hi v[216:217], v211, off offset:32
	global_store_short_d16_hi v[182:183], v234, off offset:64
	global_store_short_d16_hi v[184:185], v235, off offset:64
	global_store_short_d16_hi v[186:187], v232, off offset:64
	global_store_short_d16_hi v[216:217], v233, off offset:64
	global_store_short_d16_hi v[182:183], v238, off offset:96
	global_store_short_d16_hi v[184:185], v239, off offset:96
	global_store_short_d16_hi v[186:187], v236, off offset:96
	global_store_short_d16_hi v[216:217], v237, off offset:96
	v_add_u32_e32 v182, s0, v212
	v_accvgpr_read_b32 v199, a63
	v_ashrrev_i32_e32 v183, 31, v182
	v_add_u32_e32 v184, s48, v182
	v_accvgpr_read_b32 v197, a61
	v_accvgpr_read_b32 v196, a60
	v_lshlrev_b64 v[232:233], 1, v[182:183]
	v_ashrrev_i32_e32 v185, 31, v184
	v_pk_mul_f32 v[242:243], v[196:197], s[4:5] op_sel_hi:[1,0]
	v_lshl_add_u64 v[234:235], v[180:181], 0, v[232:233]
	v_lshlrev_b64 v[182:183], 1, v[184:185]
	v_add_u32_e32 v186, s48, v184
	global_store_short_d16_hi v[234:235], v242, off
	v_lshl_add_u64 v[236:237], v[180:181], 0, v[182:183]
	v_add_u32_e32 v242, s48, v186
	v_accvgpr_read_b32 v191, a71
	v_accvgpr_read_b32 v195, a67
	global_store_short_d16_hi v[236:237], v243, off
	v_ashrrev_i32_e32 v187, 31, v186
	v_ashrrev_i32_e32 v243, 31, v242
	v_accvgpr_read_b32 v189, a69
	v_accvgpr_read_b32 v188, a68
	v_accvgpr_read_b32 v193, a65
	v_accvgpr_read_b32 v192, a64
	v_accvgpr_read_b32 v198, a62
	v_accvgpr_write_b32 a29, v1
	v_lshlrev_b64 v[184:185], 1, v[186:187]
	v_lshlrev_b64 v[186:187], 1, v[242:243]
	v_accvgpr_read_b32 v190, a70
	v_accvgpr_read_b32 v194, a66
	v_pk_mul_f32 v[240:241], v[198:199], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[192:193], v[192:193], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[188:189], v[188:189], s[4:5] op_sel_hi:[1,0]
	v_accvgpr_write_b32 a28, v0
	v_lshl_add_u64 v[238:239], v[180:181], 0, v[184:185]
	v_lshl_add_u64 v[0:1], v[180:181], 0, v[186:187]
	v_pk_mul_f32 v[194:195], v[194:195], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[190:191], v[190:191], s[4:5] op_sel_hi:[1,0]
	global_store_short_d16_hi v[238:239], v240, off
	global_store_short_d16_hi v[0:1], v241, off
	global_store_short_d16_hi v[234:235], v192, off offset:32
	global_store_short_d16_hi v[236:237], v193, off offset:32
	global_store_short_d16_hi v[238:239], v194, off offset:32
	global_store_short_d16_hi v[0:1], v195, off offset:32
	global_store_short_d16_hi v[234:235], v188, off offset:64
	global_store_short_d16_hi v[236:237], v189, off offset:64
	global_store_short_d16_hi v[238:239], v190, off offset:64
	global_store_short_d16_hi v[0:1], v191, off offset:64
	global_store_short_d16_hi v[234:235], v246, off offset:96
	global_store_short_d16_hi v[236:237], v247, off offset:96
	global_store_short_d16_hi v[238:239], v244, off offset:96
	global_store_short_d16_hi v[0:1], v245, off offset:96
	v_add_u32_e32 v0, s0, v242
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshlrev_b64 v[188:189], 1, v[0:1]
	v_add_u32_e32 v0, s48, v0
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshlrev_b64 v[190:191], 1, v[0:1]
	v_add_u32_e32 v0, s48, v0
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshlrev_b64 v[192:193], 1, v[0:1]
	v_add_u32_e32 v0, s48, v0
	v_ashrrev_i32_e32 v1, 31, v0
	v_accvgpr_read_b32 v170, a90
	v_accvgpr_read_b32 v175, a87
	v_accvgpr_read_b32 v219, a7
	v_lshlrev_b64 v[194:195], 1, v[0:1]
	v_accvgpr_read_b32 v174, a86
	v_accvgpr_read_b32 v173, a85
	v_accvgpr_read_b32 v172, a84
	v_pk_mul_f32 v[170:171], v[170:171], s[4:5] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v211, a15
	v_accvgpr_read_b32 v218, a6
	v_lshl_add_u64 v[4:5], v[180:181], 0, v[188:189]
	v_lshl_add_u64 v[234:235], v[180:181], 0, v[190:191]
	v_lshl_add_u64 v[236:237], v[180:181], 0, v[192:193]
	v_lshl_add_u64 v[0:1], v[180:181], 0, v[194:195]
	v_pk_mul_f32 v[174:175], v[174:175], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[172:173], v[172:173], s[4:5] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v210, a14
	v_accvgpr_read_b32 v223, a3
	global_store_short_d16_hi v[4:5], v250, off
	global_store_short_d16_hi v[234:235], v251, off
	global_store_short_d16_hi v[236:237], v248, off
	global_store_short_d16_hi v[0:1], v249, off
	global_store_short_d16_hi v[4:5], v254, off offset:32
	global_store_short_d16_hi v[234:235], v255, off offset:32
	global_store_short_d16_hi v[236:237], v252, off offset:32
	global_store_short_d16_hi v[0:1], v253, off offset:32
	global_store_short_d16_hi v[4:5], v172, off offset:64
	global_store_short_d16_hi v[234:235], v173, off offset:64
	global_store_short_d16_hi v[236:237], v174, off offset:64
	global_store_short_d16_hi v[0:1], v175, off offset:64
	global_store_short_d16_hi v[4:5], v176, off offset:96
	global_store_short_d16_hi v[234:235], v177, off offset:96
	global_store_short_d16_hi v[236:237], v170, off offset:96
	global_store_short_d16_hi v[0:1], v171, off offset:96
	v_pk_mul_f32 v[170:171], v[218:219], s[4:5] op_sel_hi:[1,0]
	v_or_b32_e32 v218, 2, v2
	v_accvgpr_read_b32 v221, a1
	v_accvgpr_read_b32 v220, a0
	v_pk_mul_f32 v[180:181], v[210:211], s[4:5] op_sel_hi:[1,0]
	v_add_lshl_u32 v210, v3, v218, 6
	v_accvgpr_read_b32 v215, a11
	v_pk_mul_f32 v[4:5], v[220:221], s[4:5] op_sel_hi:[1,0]
	v_ashrrev_i32_e32 v211, 31, v210
	v_accvgpr_read_b32 v221, a29
	v_accvgpr_read_b32 v214, a10
	v_accvgpr_read_b32 v213, a9
	v_accvgpr_read_b32 v212, a8
	v_lshl_add_u64 v[210:211], v[210:211], 1, s[34:35]
	v_accvgpr_read_b32 v220, a28
	v_accvgpr_read_b32 v207, a19
	v_accvgpr_read_b32 v217, a5
	v_accvgpr_read_b32 v216, a4
	v_accvgpr_read_b32 v222, a2
	v_pk_mul_f32 v[174:175], v[214:215], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[176:177], v[212:213], s[4:5] op_sel_hi:[1,0]
	v_lshl_add_u64 v[212:213], v[210:211], 0, v[220:221]
	v_lshl_add_u64 v[214:215], v[210:211], 0, v[166:167]
	s_waitcnt vmcnt(62)
	v_accvgpr_read_b32 v161, a99
	v_accvgpr_read_b32 v165, a95
	v_accvgpr_read_b32 v199, a27
	v_accvgpr_read_b32 v203, a23
	v_accvgpr_read_b32 v206, a18
	v_accvgpr_read_b32 v205, a17
	v_accvgpr_read_b32 v204, a16
	v_accvgpr_read_b32 v209, a13
	v_accvgpr_read_b32 v208, a12
	v_pk_mul_f32 v[0:1], v[222:223], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[172:173], v[216:217], s[4:5] op_sel_hi:[1,0]
	global_store_short_d16_hi v[212:213], v4, off
	global_store_short_d16_hi v[214:215], v5, off
	v_lshl_add_u64 v[4:5], v[210:211], 0, v[168:169]
	v_lshl_add_u64 v[216:217], v[210:211], 0, v[228:229]
	v_accvgpr_read_b32 v159, a97
	v_accvgpr_read_b32 v158, a96
	v_accvgpr_read_b32 v164, a94
	v_accvgpr_read_b32 v163, a93
	v_accvgpr_read_b32 v162, a92
	v_accvgpr_read_b32 v198, a26
	v_accvgpr_read_b32 v197, a25
	v_accvgpr_read_b32 v196, a24
	v_accvgpr_read_b32 v202, a22
	v_accvgpr_read_b32 v201, a21
	v_accvgpr_read_b32 v200, a20
	v_pk_mul_f32 v[208:209], v[208:209], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[206:207], v[206:207], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[204:205], v[204:205], s[4:5] op_sel_hi:[1,0]
	global_store_short_d16_hi v[4:5], v0, off
	global_store_short_d16_hi v[216:217], v1, off
	global_store_short_d16_hi v[212:213], v172, off offset:32
	global_store_short_d16_hi v[214:215], v173, off offset:32
	global_store_short_d16_hi v[4:5], v170, off offset:32
	global_store_short_d16_hi v[216:217], v171, off offset:32
	global_store_short_d16_hi v[212:213], v176, off offset:64
	global_store_short_d16_hi v[214:215], v177, off offset:64
	global_store_short_d16_hi v[4:5], v174, off offset:64
	global_store_short_d16_hi v[216:217], v175, off offset:64
	global_store_short_d16_hi v[212:213], v208, off offset:96
	global_store_short_d16_hi v[214:215], v209, off offset:96
	global_store_short_d16_hi v[4:5], v180, off offset:96
	global_store_short_d16_hi v[216:217], v181, off offset:96
	v_lshl_add_u64 v[0:1], v[210:211], 0, v[226:227]
	v_lshl_add_u64 v[4:5], v[210:211], 0, v[224:225]
	v_lshl_add_u64 v[170:171], v[210:211], 0, v[230:231]
	v_lshl_add_u64 v[172:173], v[210:211], 0, v[178:179]
	v_accvgpr_read_b32 v145, a115
	v_accvgpr_read_b32 v149, a111
	v_accvgpr_read_b32 v153, a107
	v_accvgpr_read_b32 v157, a103
	v_accvgpr_read_b32 v160, a98
	v_pk_mul_f32 v[202:203], v[202:203], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[200:201], v[200:201], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[198:199], v[198:199], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[196:197], v[196:197], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[164:165], v[164:165], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[162:163], v[162:163], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[158:159], v[158:159], s[4:5] op_sel_hi:[1,0]
	global_store_short_d16_hi v[0:1], v204, off
	global_store_short_d16_hi v[4:5], v205, off
	global_store_short_d16_hi v[170:171], v206, off
	global_store_short_d16_hi v[172:173], v207, off
	global_store_short_d16_hi v[0:1], v200, off offset:32
	global_store_short_d16_hi v[4:5], v201, off offset:32
	global_store_short_d16_hi v[170:171], v202, off offset:32
	global_store_short_d16_hi v[172:173], v203, off offset:32
	global_store_short_d16_hi v[0:1], v196, off offset:64
	global_store_short_d16_hi v[4:5], v197, off offset:64
	global_store_short_d16_hi v[170:171], v198, off offset:64
	global_store_short_d16_hi v[172:173], v199, off offset:64
	global_store_short_d16_hi v[0:1], v162, off offset:96
	global_store_short_d16_hi v[4:5], v163, off offset:96
	global_store_short_d16_hi v[170:171], v164, off offset:96
	global_store_short_d16_hi v[172:173], v165, off offset:96
	v_lshl_add_u64 v[0:1], v[210:211], 0, v[232:233]
	v_lshl_add_u64 v[4:5], v[210:211], 0, v[182:183]
	v_accvgpr_read_b32 v143, a113
	v_accvgpr_read_b32 v142, a112
	v_accvgpr_read_b32 v148, a110
	v_accvgpr_read_b32 v147, a109
	v_accvgpr_read_b32 v146, a108
	v_accvgpr_read_b32 v152, a106
	v_accvgpr_read_b32 v151, a105
	v_accvgpr_read_b32 v150, a104
	v_accvgpr_read_b32 v156, a102
	v_accvgpr_read_b32 v155, a101
	v_accvgpr_read_b32 v154, a100
	v_pk_mul_f32 v[160:161], v[160:161], s[4:5] op_sel_hi:[1,0]
	global_store_short_d16_hi v[0:1], v158, off
	global_store_short_d16_hi v[4:5], v159, off
	v_lshl_add_u64 v[158:159], v[210:211], 0, v[184:185]
	v_lshl_add_u64 v[162:163], v[210:211], 0, v[186:187]
	v_accvgpr_read_b32 v3, a30
	v_accvgpr_read_b32 v126, a128
	v_accvgpr_read_b32 v133, a127
	v_accvgpr_read_b32 v137, a123
	v_accvgpr_read_b32 v141, a119
	v_accvgpr_read_b32 v144, a114
	v_pk_mul_f32 v[156:157], v[156:157], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[154:155], v[154:155], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[152:153], v[152:153], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[150:151], v[150:151], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[148:149], v[148:149], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[146:147], v[146:147], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[142:143], v[142:143], s[4:5] op_sel_hi:[1,0]
	global_store_short_d16_hi v[158:159], v160, off
	global_store_short_d16_hi v[162:163], v161, off
	global_store_short_d16_hi v[0:1], v154, off offset:32
	global_store_short_d16_hi v[4:5], v155, off offset:32
	global_store_short_d16_hi v[158:159], v156, off offset:32
	global_store_short_d16_hi v[162:163], v157, off offset:32
	global_store_short_d16_hi v[0:1], v150, off offset:64
	global_store_short_d16_hi v[4:5], v151, off offset:64
	global_store_short_d16_hi v[158:159], v152, off offset:64
	global_store_short_d16_hi v[162:163], v153, off offset:64
	global_store_short_d16_hi v[0:1], v146, off offset:96
	global_store_short_d16_hi v[4:5], v147, off offset:96
	global_store_short_d16_hi v[158:159], v148, off offset:96
	global_store_short_d16_hi v[162:163], v149, off offset:96
	v_lshl_add_u64 v[0:1], v[210:211], 0, v[188:189]
	v_lshl_add_u64 v[4:5], v[210:211], 0, v[190:191]
	v_or_b32_e32 v3, 2, v3
	v_accvgpr_read_b32 v127, a129
	v_accvgpr_read_b32 v132, a126
	v_accvgpr_read_b32 v131, a125
	v_accvgpr_read_b32 v130, a124
	v_accvgpr_read_b32 v136, a122
	v_accvgpr_read_b32 v135, a121
	v_accvgpr_read_b32 v134, a120
	v_accvgpr_read_b32 v140, a118
	v_accvgpr_read_b32 v139, a117
	v_accvgpr_read_b32 v138, a116
	v_pk_mul_f32 v[144:145], v[144:145], s[4:5] op_sel_hi:[1,0]
	global_store_short_d16_hi v[0:1], v142, off
	global_store_short_d16_hi v[4:5], v143, off
	v_lshl_add_u64 v[142:143], v[210:211], 0, v[192:193]
	v_lshl_add_u64 v[146:147], v[210:211], 0, v[194:195]
	v_mul_lo_u32 v3, v3, s48
	v_pk_mul_f32 v[140:141], v[140:141], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[138:139], v[138:139], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[136:137], v[136:137], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[134:135], v[134:135], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[132:133], v[132:133], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[130:131], v[130:131], s[4:5] op_sel_hi:[1,0]
	global_store_short_d16_hi v[142:143], v144, off
	global_store_short_d16_hi v[146:147], v145, off
	global_store_short_d16_hi v[0:1], v138, off offset:32
	global_store_short_d16_hi v[4:5], v139, off offset:32
	global_store_short_d16_hi v[142:143], v140, off offset:32
	global_store_short_d16_hi v[146:147], v141, off offset:32
	global_store_short_d16_hi v[0:1], v134, off offset:64
	global_store_short_d16_hi v[4:5], v135, off offset:64
	global_store_short_d16_hi v[142:143], v136, off offset:64
	global_store_short_d16_hi v[146:147], v137, off offset:64
	global_store_short_d16_hi v[0:1], v130, off offset:96
	global_store_short_d16_hi v[4:5], v131, off offset:96
	global_store_short_d16_hi v[142:143], v132, off offset:96
	global_store_short_d16_hi v[146:147], v133, off offset:96
	v_pk_mul_f32 v[4:5], v[126:127], s[4:5] op_sel_hi:[1,0]
	v_add_lshl_u32 v126, v3, v2, 6
	v_ashrrev_i32_e32 v127, 31, v126
	v_accvgpr_read_b32 v128, a130
	v_accvgpr_read_b32 v129, a131
	v_lshl_add_u64 v[126:127], v[126:127], 1, s[34:35]
	v_accvgpr_read_b32 v110, a144
	v_accvgpr_read_b32 v114, a140
	v_accvgpr_read_b32 v118, a136
	v_accvgpr_read_b32 v122, a132
	v_pk_mul_f32 v[0:1], v[128:129], s[4:5] op_sel_hi:[1,0]
	v_lshl_add_u64 v[128:129], v[126:127], 0, v[220:221]
	v_lshl_add_u64 v[130:131], v[126:127], 0, v[166:167]
	v_accvgpr_read_b32 v111, a145
	v_accvgpr_read_b32 v115, a141
	v_accvgpr_read_b32 v116, a142
	v_accvgpr_read_b32 v117, a143
	v_accvgpr_read_b32 v119, a137
	v_accvgpr_read_b32 v120, a138
	v_accvgpr_read_b32 v121, a139
	v_accvgpr_read_b32 v123, a133
	v_accvgpr_read_b32 v124, a134
	v_accvgpr_read_b32 v125, a135
	global_store_short_d16_hi v[128:129], v4, off
	global_store_short_d16_hi v[130:131], v5, off
	v_lshl_add_u64 v[4:5], v[126:127], 0, v[168:169]
	v_lshl_add_u64 v[132:133], v[126:127], 0, v[228:229]
	v_accvgpr_read_b32 v94, a160
	v_accvgpr_read_b32 v98, a156
	v_accvgpr_read_b32 v102, a152
	v_accvgpr_read_b32 v106, a148
	v_accvgpr_read_b32 v112, a146
	v_accvgpr_read_b32 v113, a147
	v_pk_mul_f32 v[124:125], v[124:125], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[122:123], v[122:123], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[120:121], v[120:121], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[118:119], v[118:119], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[116:117], v[116:117], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[114:115], v[114:115], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[110:111], v[110:111], s[4:5] op_sel_hi:[1,0]
	global_store_short_d16_hi v[4:5], v0, off
	global_store_short_d16_hi v[132:133], v1, off
	global_store_short_d16_hi v[128:129], v122, off offset:32
	global_store_short_d16_hi v[130:131], v123, off offset:32
	global_store_short_d16_hi v[4:5], v124, off offset:32
	global_store_short_d16_hi v[132:133], v125, off offset:32
	global_store_short_d16_hi v[128:129], v118, off offset:64
	global_store_short_d16_hi v[130:131], v119, off offset:64
	global_store_short_d16_hi v[4:5], v120, off offset:64
	global_store_short_d16_hi v[132:133], v121, off offset:64
	global_store_short_d16_hi v[128:129], v114, off offset:96
	global_store_short_d16_hi v[130:131], v115, off offset:96
	global_store_short_d16_hi v[4:5], v116, off offset:96
	global_store_short_d16_hi v[132:133], v117, off offset:96
	v_lshl_add_u64 v[0:1], v[126:127], 0, v[226:227]
	v_lshl_add_u64 v[4:5], v[126:127], 0, v[224:225]
	v_accvgpr_read_b32 v95, a161
	v_accvgpr_read_b32 v99, a157
	v_accvgpr_read_b32 v100, a158
	v_accvgpr_read_b32 v101, a159
	v_accvgpr_read_b32 v103, a153
	v_accvgpr_read_b32 v104, a154
	v_accvgpr_read_b32 v105, a155
	v_accvgpr_read_b32 v107, a149
	v_accvgpr_read_b32 v108, a150
	v_accvgpr_read_b32 v109, a151
	v_pk_mul_f32 v[112:113], v[112:113], s[4:5] op_sel_hi:[1,0]
	global_store_short_d16_hi v[0:1], v110, off
	global_store_short_d16_hi v[4:5], v111, off
	v_lshl_add_u64 v[110:111], v[126:127], 0, v[230:231]
	v_lshl_add_u64 v[114:115], v[126:127], 0, v[178:179]
	v_accvgpr_read_b32 v78, a176
	v_accvgpr_read_b32 v82, a172
	v_accvgpr_read_b32 v86, a168
	v_accvgpr_read_b32 v90, a164
	v_accvgpr_read_b32 v96, a162
	v_accvgpr_read_b32 v97, a163
	v_pk_mul_f32 v[108:109], v[108:109], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[106:107], v[106:107], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[104:105], v[104:105], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[102:103], v[102:103], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[100:101], v[100:101], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[98:99], v[98:99], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[94:95], v[94:95], s[4:5] op_sel_hi:[1,0]
	global_store_short_d16_hi v[110:111], v112, off
	global_store_short_d16_hi v[114:115], v113, off
	global_store_short_d16_hi v[0:1], v106, off offset:32
	global_store_short_d16_hi v[4:5], v107, off offset:32
	global_store_short_d16_hi v[110:111], v108, off offset:32
	global_store_short_d16_hi v[114:115], v109, off offset:32
	global_store_short_d16_hi v[0:1], v102, off offset:64
	global_store_short_d16_hi v[4:5], v103, off offset:64
	global_store_short_d16_hi v[110:111], v104, off offset:64
	global_store_short_d16_hi v[114:115], v105, off offset:64
	global_store_short_d16_hi v[0:1], v98, off offset:96
	global_store_short_d16_hi v[4:5], v99, off offset:96
	global_store_short_d16_hi v[110:111], v100, off offset:96
	global_store_short_d16_hi v[114:115], v101, off offset:96
	v_lshl_add_u64 v[0:1], v[126:127], 0, v[232:233]
	v_lshl_add_u64 v[4:5], v[126:127], 0, v[182:183]
	v_accvgpr_read_b32 v66, a188
	v_accvgpr_read_b32 v79, a177
	v_accvgpr_read_b32 v83, a173
	v_accvgpr_read_b32 v84, a174
	v_accvgpr_read_b32 v85, a175
	v_accvgpr_read_b32 v87, a169
	v_accvgpr_read_b32 v88, a170
	v_accvgpr_read_b32 v89, a171
	v_accvgpr_read_b32 v91, a165
	v_accvgpr_read_b32 v92, a166
	v_accvgpr_read_b32 v93, a167
	v_pk_mul_f32 v[96:97], v[96:97], s[4:5] op_sel_hi:[1,0]
	global_store_short_d16_hi v[0:1], v94, off
	global_store_short_d16_hi v[4:5], v95, off
	v_lshl_add_u64 v[94:95], v[126:127], 0, v[184:185]
	v_lshl_add_u64 v[98:99], v[126:127], 0, v[186:187]
	v_accvgpr_read_b32 v62, a192
	v_accvgpr_read_b32 v67, a189
	v_accvgpr_read_b32 v70, a184
	v_accvgpr_read_b32 v74, a180
	v_accvgpr_read_b32 v80, a178
	v_accvgpr_read_b32 v81, a179
	v_pk_mul_f32 v[92:93], v[92:93], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[90:91], v[90:91], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[88:89], v[88:89], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[86:87], v[86:87], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[84:85], v[84:85], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[82:83], v[82:83], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[78:79], v[78:79], s[4:5] op_sel_hi:[1,0]
	global_store_short_d16_hi v[94:95], v96, off
	global_store_short_d16_hi v[98:99], v97, off
	global_store_short_d16_hi v[0:1], v90, off offset:32
	global_store_short_d16_hi v[4:5], v91, off offset:32
	global_store_short_d16_hi v[94:95], v92, off offset:32
	global_store_short_d16_hi v[98:99], v93, off offset:32
	global_store_short_d16_hi v[0:1], v86, off offset:64
	global_store_short_d16_hi v[4:5], v87, off offset:64
	global_store_short_d16_hi v[94:95], v88, off offset:64
	global_store_short_d16_hi v[98:99], v89, off offset:64
	global_store_short_d16_hi v[0:1], v82, off offset:96
	global_store_short_d16_hi v[4:5], v83, off offset:96
	global_store_short_d16_hi v[94:95], v84, off offset:96
	global_store_short_d16_hi v[98:99], v85, off offset:96
	v_lshl_add_u64 v[0:1], v[126:127], 0, v[188:189]
	v_lshl_add_u64 v[4:5], v[126:127], 0, v[190:191]
	v_accvgpr_read_b32 v64, a194
	v_accvgpr_read_b32 v65, a195
	v_accvgpr_read_b32 v68, a190
	v_accvgpr_read_b32 v69, a191
	v_accvgpr_read_b32 v71, a185
	v_accvgpr_read_b32 v72, a186
	v_accvgpr_read_b32 v73, a187
	v_accvgpr_read_b32 v75, a181
	v_accvgpr_read_b32 v76, a182
	v_accvgpr_read_b32 v77, a183
	v_pk_mul_f32 v[80:81], v[80:81], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[66:67], v[66:67], s[4:5] op_sel_hi:[1,0]
	global_store_short_d16_hi v[0:1], v78, off
	global_store_short_d16_hi v[4:5], v79, off
	v_lshl_add_u64 v[78:79], v[126:127], 0, v[192:193]
	v_lshl_add_u64 v[82:83], v[126:127], 0, v[194:195]
	v_pk_mul_f32 v[76:77], v[76:77], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[74:75], v[74:75], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[72:73], v[72:73], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[70:71], v[70:71], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[68:69], v[68:69], s[4:5] op_sel_hi:[1,0]
	global_store_short_d16_hi v[78:79], v80, off
	global_store_short_d16_hi v[82:83], v81, off
	global_store_short_d16_hi v[0:1], v74, off offset:32
	global_store_short_d16_hi v[4:5], v75, off offset:32
	global_store_short_d16_hi v[78:79], v76, off offset:32
	global_store_short_d16_hi v[82:83], v77, off offset:32
	global_store_short_d16_hi v[0:1], v70, off offset:64
	global_store_short_d16_hi v[4:5], v71, off offset:64
	global_store_short_d16_hi v[78:79], v72, off offset:64
	global_store_short_d16_hi v[82:83], v73, off offset:64
	global_store_short_d16_hi v[0:1], v66, off offset:96
	global_store_short_d16_hi v[4:5], v67, off offset:96
	global_store_short_d16_hi v[78:79], v68, off offset:96
	global_store_short_d16_hi v[82:83], v69, off offset:96
	v_pk_mul_f32 v[0:1], v[64:65], s[4:5] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v64, a252
	v_accvgpr_read_b32 v63, a193
	v_accvgpr_read_b32 v66, a254
	v_accvgpr_read_b32 v67, a255
	v_pk_mul_f32 v[4:5], v[62:63], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[62:63], v[66:67], s[4:5] op_sel_hi:[1,0]
	v_add_lshl_u32 v66, v3, v218, 6
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 1, s[34:35]
	v_accvgpr_read_b32 v46, a208
	v_accvgpr_read_b32 v50, a204
	v_accvgpr_read_b32 v54, a200
	v_accvgpr_read_b32 v58, a196
	v_lshl_add_u64 v[2:3], v[66:67], 0, v[220:221]
	v_lshl_add_u64 v[68:69], v[66:67], 0, v[166:167]
	v_accvgpr_read_b32 v47, a209
	v_accvgpr_read_b32 v51, a205
	v_accvgpr_read_b32 v52, a206
	v_accvgpr_read_b32 v53, a207
	v_accvgpr_read_b32 v55, a201
	v_accvgpr_read_b32 v56, a202
	v_accvgpr_read_b32 v57, a203
	v_accvgpr_read_b32 v59, a197
	v_accvgpr_read_b32 v60, a198
	v_accvgpr_read_b32 v61, a199
	global_store_short_d16_hi v[2:3], v4, off
	global_store_short_d16_hi v[68:69], v5, off
	v_lshl_add_u64 v[4:5], v[66:67], 0, v[168:169]
	v_lshl_add_u64 v[70:71], v[66:67], 0, v[228:229]
	v_accvgpr_read_b32 v30, a224
	v_accvgpr_read_b32 v34, a220
	v_accvgpr_read_b32 v38, a216
	v_accvgpr_read_b32 v42, a212
	v_accvgpr_read_b32 v48, a210
	v_accvgpr_read_b32 v49, a211
	v_pk_mul_f32 v[60:61], v[60:61], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], s[4:5] op_sel_hi:[1,0]
	global_store_short_d16_hi v[4:5], v0, off
	global_store_short_d16_hi v[70:71], v1, off
	global_store_short_d16_hi v[2:3], v58, off offset:32
	global_store_short_d16_hi v[68:69], v59, off offset:32
	global_store_short_d16_hi v[4:5], v60, off offset:32
	global_store_short_d16_hi v[70:71], v61, off offset:32
	global_store_short_d16_hi v[2:3], v54, off offset:64
	global_store_short_d16_hi v[68:69], v55, off offset:64
	global_store_short_d16_hi v[4:5], v56, off offset:64
	global_store_short_d16_hi v[70:71], v57, off offset:64
	global_store_short_d16_hi v[2:3], v50, off offset:96
	global_store_short_d16_hi v[68:69], v51, off offset:96
	global_store_short_d16_hi v[4:5], v52, off offset:96
	global_store_short_d16_hi v[70:71], v53, off offset:96
	v_lshl_add_u64 v[0:1], v[66:67], 0, v[226:227]
	v_lshl_add_u64 v[2:3], v[66:67], 0, v[224:225]
	v_accvgpr_read_b32 v31, a225
	v_accvgpr_read_b32 v35, a221
	v_accvgpr_read_b32 v36, a222
	v_accvgpr_read_b32 v37, a223
	v_accvgpr_read_b32 v39, a217
	v_accvgpr_read_b32 v40, a218
	v_accvgpr_read_b32 v41, a219
	v_accvgpr_read_b32 v43, a213
	v_accvgpr_read_b32 v44, a214
	v_accvgpr_read_b32 v45, a215
	v_pk_mul_f32 v[48:49], v[48:49], s[4:5] op_sel_hi:[1,0]
	global_store_short_d16_hi v[0:1], v46, off
	global_store_short_d16_hi v[2:3], v47, off
	v_lshl_add_u64 v[4:5], v[66:67], 0, v[230:231]
	v_lshl_add_u64 v[46:47], v[66:67], 0, v[178:179]
	v_accvgpr_read_b32 v14, a240
	v_accvgpr_read_b32 v18, a236
	v_accvgpr_read_b32 v22, a232
	v_accvgpr_read_b32 v26, a228
	v_accvgpr_read_b32 v32, a226
	v_accvgpr_read_b32 v33, a227
	v_pk_mul_f32 v[44:45], v[44:45], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], s[4:5] op_sel_hi:[1,0]
	global_store_short_d16_hi v[4:5], v48, off
	global_store_short_d16_hi v[46:47], v49, off
	global_store_short_d16_hi v[0:1], v42, off offset:32
	global_store_short_d16_hi v[2:3], v43, off offset:32
	global_store_short_d16_hi v[4:5], v44, off offset:32
	global_store_short_d16_hi v[46:47], v45, off offset:32
	global_store_short_d16_hi v[0:1], v38, off offset:64
	global_store_short_d16_hi v[2:3], v39, off offset:64
	global_store_short_d16_hi v[4:5], v40, off offset:64
	global_store_short_d16_hi v[46:47], v41, off offset:64
	global_store_short_d16_hi v[0:1], v34, off offset:96
	global_store_short_d16_hi v[2:3], v35, off offset:96
	global_store_short_d16_hi v[4:5], v36, off offset:96
	global_store_short_d16_hi v[46:47], v37, off offset:96
	v_lshl_add_u64 v[0:1], v[66:67], 0, v[232:233]
	v_lshl_add_u64 v[2:3], v[66:67], 0, v[182:183]
	v_accvgpr_read_b32 v15, a241
	v_accvgpr_read_b32 v19, a237
	v_accvgpr_read_b32 v20, a238
	v_accvgpr_read_b32 v21, a239
	v_accvgpr_read_b32 v23, a233
	v_accvgpr_read_b32 v24, a234
	v_accvgpr_read_b32 v25, a235
	v_accvgpr_read_b32 v27, a229
	v_accvgpr_read_b32 v28, a230
	v_accvgpr_read_b32 v29, a231
	v_pk_mul_f32 v[32:33], v[32:33], s[4:5] op_sel_hi:[1,0]
	global_store_short_d16_hi v[0:1], v30, off
	global_store_short_d16_hi v[2:3], v31, off
	v_lshl_add_u64 v[4:5], v[66:67], 0, v[184:185]
	v_lshl_add_u64 v[30:31], v[66:67], 0, v[186:187]
	v_accvgpr_read_b32 v6, a248
	v_accvgpr_read_b32 v10, a244
	v_accvgpr_read_b32 v16, a242
	v_accvgpr_read_b32 v17, a243
	v_pk_mul_f32 v[28:29], v[28:29], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], s[4:5] op_sel_hi:[1,0]
	global_store_short_d16_hi v[4:5], v32, off
	global_store_short_d16_hi v[30:31], v33, off
	global_store_short_d16_hi v[0:1], v26, off offset:32
	global_store_short_d16_hi v[2:3], v27, off offset:32
	global_store_short_d16_hi v[4:5], v28, off offset:32
	global_store_short_d16_hi v[30:31], v29, off offset:32
	global_store_short_d16_hi v[0:1], v22, off offset:64
	global_store_short_d16_hi v[2:3], v23, off offset:64
	global_store_short_d16_hi v[4:5], v24, off offset:64
	global_store_short_d16_hi v[30:31], v25, off offset:64
	global_store_short_d16_hi v[0:1], v18, off offset:96
	global_store_short_d16_hi v[2:3], v19, off offset:96
	global_store_short_d16_hi v[4:5], v20, off offset:96
	global_store_short_d16_hi v[30:31], v21, off offset:96
	v_lshl_add_u64 v[0:1], v[66:67], 0, v[188:189]
	v_lshl_add_u64 v[2:3], v[66:67], 0, v[190:191]
	v_accvgpr_read_b32 v7, a249
	v_accvgpr_read_b32 v8, a250
	v_accvgpr_read_b32 v9, a251
	v_accvgpr_read_b32 v11, a245
	v_accvgpr_read_b32 v12, a246
	v_accvgpr_read_b32 v13, a247
	v_pk_mul_f32 v[16:17], v[16:17], s[4:5] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v65, a253
	global_store_short_d16_hi v[0:1], v14, off
	global_store_short_d16_hi v[2:3], v15, off
	v_lshl_add_u64 v[4:5], v[66:67], 0, v[192:193]
	v_lshl_add_u64 v[14:15], v[66:67], 0, v[194:195]
	v_pk_mul_f32 v[12:13], v[12:13], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[64:65], v[64:65], s[4:5] op_sel_hi:[1,0]
	global_store_short_d16_hi v[4:5], v16, off
	global_store_short_d16_hi v[14:15], v17, off
	global_store_short_d16_hi v[0:1], v10, off offset:32
	global_store_short_d16_hi v[2:3], v11, off offset:32
	global_store_short_d16_hi v[4:5], v12, off offset:32
	global_store_short_d16_hi v[14:15], v13, off offset:32
	global_store_short_d16_hi v[0:1], v6, off offset:64
	global_store_short_d16_hi v[2:3], v7, off offset:64
	global_store_short_d16_hi v[4:5], v8, off offset:64
	global_store_short_d16_hi v[14:15], v9, off offset:64
	global_store_short_d16_hi v[0:1], v64, off offset:96
	global_store_short_d16_hi v[2:3], v65, off offset:96
	global_store_short_d16_hi v[4:5], v62, off offset:96
	global_store_short_d16_hi v[14:15], v63, off offset:96
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z21mxfp4_colfirst_kernel16colfirst_globals
		.amdhsa_group_segment_fixed_size 131072
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 272
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
	.size	_Z21mxfp4_colfirst_kernel16colfirst_globals, .Lfunc_end0-_Z21mxfp4_colfirst_kernel16colfirst_globals
                                        ; -- End function
	.set _Z21mxfp4_colfirst_kernel16colfirst_globals.num_vgpr, 256
	.set _Z21mxfp4_colfirst_kernel16colfirst_globals.num_agpr, 256
	.set _Z21mxfp4_colfirst_kernel16colfirst_globals.numbered_sgpr, 92
	.set _Z21mxfp4_colfirst_kernel16colfirst_globals.private_seg_size, 0
	.set _Z21mxfp4_colfirst_kernel16colfirst_globals.uses_vcc, 0
	.set _Z21mxfp4_colfirst_kernel16colfirst_globals.uses_flat_scratch, 0
	.set _Z21mxfp4_colfirst_kernel16colfirst_globals.has_dyn_sized_stack, 0
	.set _Z21mxfp4_colfirst_kernel16colfirst_globals.has_recursion, 0
	.set _Z21mxfp4_colfirst_kernel16colfirst_globals.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 16152
; TotalNumSgprs: 98
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
	.type	__hip_cuid_387e5843377cfc19,@object ; @__hip_cuid_387e5843377cfc19
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_387e5843377cfc19
__hip_cuid_387e5843377cfc19:
	.byte	0                               ; 0x0
	.size	__hip_cuid_387e5843377cfc19, 1

	.ident	"clang version 20.0.0git (https://github.com/ROCm/llvm-project.git 27682a16360e33e37c4f3cc6adf9a620733f8fe1)"
	.ident	"AMD clang version 20.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.1.0 25425 1b0eada6b0ee93e2e694c8c146d23fca90bc11c5)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_387e5843377cfc19
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     256
    .args:
      - .offset:         0
        .size:           272
        .value_kind:     by_value
    .group_segment_fixed_size: 131072
    .kernarg_segment_align: 8
    .kernarg_segment_size: 272
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _Z21mxfp4_colfirst_kernel16colfirst_globals
    .private_segment_fixed_size: 0
    .sgpr_count:     98
    .sgpr_spill_count: 0
    .symbol:         _Z21mxfp4_colfirst_kernel16colfirst_globals.kd
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
