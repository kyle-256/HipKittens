	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.text
	.protected	_Z22mxfp4_gluon_cpp_kernel13gluon_globals ; -- Begin function _Z22mxfp4_gluon_cpp_kernel13gluon_globals
	.globl	_Z22mxfp4_gluon_cpp_kernel13gluon_globals
	.p2align	8
	.type	_Z22mxfp4_gluon_cpp_kernel13gluon_globals,@function
_Z22mxfp4_gluon_cpp_kernel13gluon_globals: ; @_Z22mxfp4_gluon_cpp_kernel13gluon_globals
; %bb.0:
	s_ashr_i32 s3, s2, 31
	s_lshr_b32 s3, s3, 27
	s_add_i32 s3, s2, s3
	s_ashr_i32 s33, s3, 5
	v_lshrrev_b32_e32 v1, 7, v0
	s_lshl_b32 s55, s33, 8
	v_lshl_or_b32 v2, v1, 6, s55
	v_ashrrev_i32_e32 v5, 5, v2
	s_load_dwordx2 s[16:17], s[0:1], 0x60
	s_load_dword s18, s[0:1], 0x80
	s_load_dwordx2 s[36:37], s[0:1], 0x90
	s_load_dword s28, s[0:1], 0xb0
	s_load_dwordx2 s[34:35], s[0:1], 0xc0
	s_load_dwordx2 s[46:47], s[0:1], 0xe0
	s_load_dword s44, s[0:1], 0xf0
	s_mov_b32 s7, 0x110000
	v_or_b32_e32 v4, 0x80, v2
	v_ashrrev_i32_e32 v4, 5, v4
	s_mov_b32 s6, -1
	s_waitcnt lgkmcnt(0)
	v_mul_lo_u32 v2, v5, s18
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], s[16:17], 0, v[2:3]
	s_andn2_b32 s3, s3, 31
	s_sub_i32 s45, s2, s3
	v_bfe_u32 v146, v0, 6, 1
	s_lshl_b32 s58, s45, 8
	v_readfirstlane_b32 s5, v3
	v_readfirstlane_b32 s4, v2
	s_mov_b64 s[26:27], s[6:7]
	s_load_dwordx2 s[40:41], s[0:1], 0x0
	s_load_dword s60, s[0:1], 0x20
	v_lshrrev_b32_e32 v6, 6, v0
	s_mov_b64 s[24:25], s[4:5]
	s_movk_i32 s59, 0x70
	v_mul_lo_u32 v2, v4, s18
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], s[16:17], 0, v[2:3]
	s_mov_b64 s[10:11], s[6:7]
	s_mov_b64 s[8:9], s[4:5]
	v_readfirstlane_b32 s13, v3
	s_mov_b32 s9, s13
	v_readfirstlane_b32 s12, v2
	s_mov_b32 s8, s12
	s_mov_b64 s[14:15], s[6:7]
	s_mov_b64 s[12:13], s[4:5]
	s_load_dword s61, s[0:1], 0x50
	s_load_dwordx2 s[56:57], s[0:1], 0x30
	v_or_b32_e32 v2, 1, v5
	v_mul_lo_u32 v2, v2, s18
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], s[16:17], 0, v[2:3]
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s55, s55, s60
	v_readfirstlane_b32 s20, v3
	s_mov_b32 s13, s20
	v_readfirstlane_b32 s19, v2
	s_mov_b32 s12, s19
	v_and_b32_e32 v67, 48, v0
	v_or_b32_e32 v2, 1, v4
	v_mul_lo_u32 v2, v2, s18
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], s[16:17], 0, v[2:3]
	s_mov_b64 s[18:19], s[6:7]
	s_mov_b64 s[16:17], s[4:5]
	v_lshlrev_b32_e32 v66, 13, v1
	v_readfirstlane_b32 s21, v3
	s_mov_b32 s17, s21
	v_readfirstlane_b32 s20, v2
	s_mov_b32 s16, s20
	s_mov_b64 s[22:23], s[6:7]
	s_mov_b64 s[20:21], s[4:5]
	v_lshlrev_b32_e32 v71, 3, v0
	v_lshl_or_b32 v2, v146, 6, s58
	v_ashrrev_i32_e32 v7, 5, v2
	v_mul_lo_u32 v2, v7, s28
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[4:5], s[36:37], 0, v[2:3]
	v_lshlrev_b32_e32 v69, 13, v146
	v_or_b32_e32 v70, 0x10000, v69
	v_readfirstlane_b32 s3, v5
	s_mov_b32 s21, s3
	v_readfirstlane_b32 s2, v4
	s_mov_b32 s20, s2
	s_lshl_b32 s2, s28, 2
	v_add_u32_e32 v2, s2, v2
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], s[36:37], 0, v[2:3]
	v_accvgpr_write_b32 a195, 0
	v_readfirstlane_b32 s29, v3
	s_mov_b32 s25, s29
	v_readfirstlane_b32 s3, v2
	s_mov_b32 s24, s3
	v_or_b32_e32 v2, 1, v7
	v_mul_lo_u32 v2, v2, s28
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[4:5], s[36:37], 0, v[2:3]
	v_add_u32_e32 v2, s2, v2
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], s[36:37], 0, v[2:3]
	s_mov_b64 s[30:31], s[6:7]
	s_mov_b64 s[28:29], s[4:5]
	v_readfirstlane_b32 s2, v2
	v_lshlrev_b32_e32 v2, 10, v6
	v_readfirstlane_b32 s38, v5
	s_mov_b32 s29, s38
	s_mov_b64 s[38:39], s[6:7]
	s_mov_b64 s[36:37], s[4:5]
	s_mov_b32 s36, s2
	v_readfirstlane_b32 s47, v2
	s_add_i32 s53, s47, 0x18000
	s_add_i32 s54, s53, 0x4000
	v_readfirstlane_b32 s3, v4
	s_mov_b32 s28, s3
	v_readfirstlane_b32 s3, v3
	s_mov_b32 s37, s3
	v_lshlrev_b32_e32 v2, 4, v0
	v_bitop3_b32 v2, v2, s59, v0 bitop3:0x48
	v_lshrrev_b32_e32 v3, 3, v0
	v_or_b32_e32 v4, 0x60, v3
	s_add_i32 s48, s47, 0x4000
	v_mad_u64_u32 v[130:131], s[0:1], v3, s60, v[2:3]
	s_lshl_b32 s0, s60, 5
	s_nop 0
	v_add_u32_e32 v131, s0, v130
	v_add_u32_e32 v138, s0, v131
	v_mad_u64_u32 v[134:135], s[0:1], v3, s61, v[2:3]
	s_add_i32 s49, s47, 0xc000
	v_mad_u64_u32 v[132:133], s[0:1], v4, s60, v[2:3]
	s_lshl_b32 s0, s61, 5
	s_nop 0
	v_add_u32_e32 v133, s0, v134
	v_add_u32_e32 v135, s0, v133
	v_mad_u64_u32 v[136:137], s[0:1], v4, s61, v[2:3]
	s_mov_b64 s[0:1], s[4:5]
	s_mov_b32 s0, s40
	s_add_i32 s50, s47, 0x8000
	s_mov_b64 s[2:3], s[6:7]
	s_mov_b32 s1, s41
	s_mov_b64 s[42:43], s[6:7]
	s_mov_b64 s[40:41], s[4:5]
	s_mov_b32 s40, s56
	s_mov_b32 s56, s55
	;;#ASMSTART
	;;#ASMEND
	v_lshlrev_b32_e32 v2, 2, v0
	v_and_b32_e32 v137, 0xfc, v2
	v_lshlrev_b32_e32 v2, 7, v0
	v_and_b32_e32 v68, 0x780, v2
	v_or_b32_e32 v2, v68, v67
	v_or_b32_e32 v18, v2, v70
	v_or_b32_e32 v70, v68, v70
	s_mov_b32 s41, s57
	s_mov_b32 s57, s47
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s57
	s_add_i32 s57, s47, 0x1000
	v_or_b32_e32 v3, v2, v66
	v_bitop3_b32 v4, v71, v3, s59 bitop3:0x6c
	v_or_b32_e32 v3, 64, v3
	v_bitop3_b32 v3, v71, v3, s59 bitop3:0x6c
	buffer_load_dwordx4 v130, s[0:3], s56 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s57
	s_add_i32 s57, s47, 0x2000
	v_or_b32_e32 v66, v68, v66
	v_or_b32_e32 v72, v66, v67
	v_bitop3_b32 v145, v71, v72, s59 bitop3:0x6c
	v_or_b32_e32 v72, 64, v67
	v_or_b32_e32 v73, v66, v72
	v_bitop3_b32 v147, v71, v73, s59 bitop3:0x6c
	v_or_b32_e32 v73, 0x4000, v66
	v_or_b32_e32 v74, v73, v67
	v_bitop3_b32 v148, v71, v74, s59 bitop3:0x6c
	buffer_load_dwordx4 v131, s[0:3], s56 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s57
	s_add_i32 s57, s47, 0x3000
	v_or_b32_e32 v73, v73, v72
	v_bitop3_b32 v149, v71, v73, s59 bitop3:0x6c
	v_add_u32_e32 v73, v70, v67
	v_lshrrev_b32_e32 v74, 4, v73
	v_bitop3_b32 v150, v74, v73, s59 bitop3:0x6c
	buffer_load_dwordx4 v138, s[0:3], s56 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s57
	v_add_u32_e32 v70, v70, v72
	v_lshrrev_b32_e32 v73, 4, v70
	v_bitop3_b32 v151, v73, v70, s59 bitop3:0x6c
	buffer_load_dwordx4 v132, s[0:3], s56 offen lds
	s_lshl_b32 s56, s60, 7
	s_add_i32 s56, s55, s56
	s_mov_b32 s57, s56
	;;#ASMSTART
	;;#ASMEND
	v_or_b32_e32 v68, v69, v68
	v_or_b32_e32 v69, 0x14000, v68
	v_or_b32_e32 v70, v69, v67
	v_bitop3_b32 v152, v71, v70, s59 bitop3:0x6c
	s_mov_b32 s60, s50
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s60
	s_add_i32 s60, s47, 0x9000
	v_or_b32_e32 v69, v69, v72
	v_bitop3_b32 v153, v71, v69, s59 bitop3:0x6c
	v_or_b32_e32 v69, 0x18000, v68
	v_or_b32_e32 v70, v69, v67
	v_bitop3_b32 v154, v71, v70, s59 bitop3:0x6c
	buffer_load_dwordx4 v130, s[0:3], s57 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s60
	s_add_i32 s60, s47, 0xa000
	v_or_b32_e32 v69, v69, v72
	v_bitop3_b32 v155, v71, v69, s59 bitop3:0x6c
	buffer_load_dwordx4 v131, s[0:3], s57 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s60
	s_add_i32 s60, s47, 0xb000
	v_or_b32_e32 v68, 0x1c000, v68
	v_or_b32_e32 v69, v68, v67
	v_bitop3_b32 v156, v71, v69, s59 bitop3:0x6c
	buffer_load_dwordx4 v138, s[0:3], s57 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s60
	v_or_b32_e32 v68, v68, v72
	v_bitop3_b32 v157, v71, v68, s59 bitop3:0x6c
	v_or_b32_e32 v68, 0x8000, v66
	v_or_b32_e32 v69, v68, v67
	v_bitop3_b32 v158, v71, v69, s59 bitop3:0x6c
	buffer_load_dwordx4 v132, s[0:3], s57 offen lds
	s_mul_i32 s57, s58, s61
	s_mov_b32 s58, s57
	;;#ASMSTART
	;;#ASMEND
	v_or_b32_e32 v68, v68, v72
	v_bitop3_b32 v159, v71, v68, s59 bitop3:0x6c
	v_or_b32_e32 v66, 0xc000, v66
	v_or_b32_e32 v67, v66, v67
	v_bitop3_b32 v160, v71, v67, s59 bitop3:0x6c
	v_lshrrev_b32_e32 v2, 4, v18
	v_bitop3_b32 v14, v2, v18, s59 bitop3:0x6c
	v_add_u32_e32 v18, 64, v18
	v_lshrrev_b32_e32 v19, 4, v18
	v_bitop3_b32 v30, v19, v18, s59 bitop3:0x6c
	v_or_b32_e32 v66, v66, v72
	v_bitop3_b32 v161, v71, v66, s59 bitop3:0x6c
	s_movk_i32 s59, 0x100
	s_add_i32 s51, s47, 0x10000
	s_mov_b32 s60, s51
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s60
	s_add_i32 s60, s51, 0x1000
	s_add_i32 s52, s51, 0x4000
	buffer_load_dwordx4 v134, s[40:43], s58 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s60
	s_add_i32 s60, s51, 0x2000
	buffer_load_dwordx4 v133, s[40:43], s58 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s60
	s_add_i32 s60, s51, 0x3000
	buffer_load_dwordx4 v135, s[40:43], s58 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s60
	v_accvgpr_write_b32 a194, 0
	buffer_load_dwordx4 v136, s[40:43], s58 offen lds
	s_lshl_b32 s58, s61, 7
	s_add_i32 s58, s57, s58
	s_mov_b32 s60, s58
	s_mov_b32 s61, s53
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s61
	s_add_i32 s61, s53, 0x1000
	buffer_load_dwordx4 v134, s[40:43], s60 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s61
	s_add_i32 s61, s53, 0x2000
	buffer_load_dwordx4 v133, s[40:43], s60 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s61
	s_add_i32 s61, s53, 0x3000
	buffer_load_dwordx4 v135, s[40:43], s60 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s61
	s_mov_b32 s61, s48
	buffer_load_dwordx4 v136, s[40:43], s60 offen lds
	s_or_b32 s60, s55, 0x80
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s61
	s_add_i32 s61, s47, 0x5000
	buffer_load_dwordx4 v130, s[0:3], s60 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s61
	s_add_i32 s61, s47, 0x6000
	buffer_load_dwordx4 v131, s[0:3], s60 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s61
	s_add_i32 s61, s47, 0x7000
	buffer_load_dwordx4 v138, s[0:3], s60 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s61
	s_mov_b32 s61, s49
	buffer_load_dwordx4 v132, s[0:3], s60 offen lds
	s_add_i32 s60, s56, 0x80
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s61
	s_add_i32 s61, s47, 0xd000
	buffer_load_dwordx4 v130, s[0:3], s60 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s61
	s_add_i32 s61, s47, 0xe000
	buffer_load_dwordx4 v131, s[0:3], s60 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s61
	s_add_i32 s61, s47, 0xf000
	buffer_load_dwordx4 v138, s[0:3], s60 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s61
	s_mov_b32 s61, s52
	buffer_load_dwordx4 v132, s[0:3], s60 offen lds
	s_or_b32 s60, s57, 0x80
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s61
	s_add_i32 s61, s51, 0x5000
	buffer_load_dwordx4 v134, s[40:43], s60 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s61
	s_add_i32 s61, s51, 0x6000
	buffer_load_dwordx4 v133, s[40:43], s60 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s61
	s_add_i32 s61, s51, 0x7000
	buffer_load_dwordx4 v135, s[40:43], s60 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s61
	s_mov_b32 s61, s54
	buffer_load_dwordx4 v136, s[40:43], s60 offen lds
	s_add_i32 s60, s58, 0x80
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s61
	s_add_i32 s61, s53, 0x5000
	buffer_load_dwordx4 v134, s[40:43], s60 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s61
	s_add_i32 s61, s53, 0x6000
	buffer_load_dwordx4 v133, s[40:43], s60 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s61
	s_add_i32 s61, s53, 0x7000
	buffer_load_dwordx4 v135, s[40:43], s60 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s61
	v_accvgpr_write_b32 a193, 0
	buffer_load_dwordx4 v136, s[40:43], s60 offen lds
	buffer_load_dword v98, v137, s[4:7], 0 offen
	buffer_load_dword v139, v137, s[8:11], 0 offen
	buffer_load_dword v99, v137, s[12:15], 0 offen
	buffer_load_dword v140, v137, s[16:19], 0 offen
	buffer_load_dword v141, v137, s[20:23], 0 offen
	buffer_load_dword v142, v137, s[24:27], 0 offen
	buffer_load_dword v143, v137, s[28:31], 0 offen
	buffer_load_dword v144, v137, s[36:39], 0 offen
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
	s_mov_b32 s60, 0
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
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
.LBB0_1:                                ; =>This Inner Loop Header: Depth=1
	s_min_u32 s61, s60, 29
	s_lshl_b32 s61, s61, 7
	s_addk_i32 s61, 0x100
	s_and_b32 s62, s60, 1
	s_add_i32 s63, s61, s55
	s_bitcmp1_b32 s60, 0
	s_cselect_b32 s64, s48, s47
	s_cselect_b32 s65, s49, s50
	s_cselect_b32 s66, s52, s51
	s_cselect_b32 s67, s54, s53
	s_add_i32 s68, s61, s56
	s_add_i32 s69, s61, s57
	s_add_i32 s70, s61, s58
	s_add_i32 s61, s60, 1
	s_add_i32 s71, s64, 0x1000
	s_add_i32 s72, s64, 0x2000
	s_add_i32 s73, s64, 0x3000
	s_add_i32 s74, s65, 0x1000
	s_add_i32 s75, s65, 0x2000
	s_add_i32 s76, s65, 0x3000
	s_add_i32 s77, s66, 0x1000
	s_add_i32 s78, s66, 0x2000
	s_add_i32 s79, s66, 0x3000
	s_add_i32 s80, s67, 0x1000
	s_add_i32 s81, s67, 0x2000
	s_add_i32 s82, s67, 0x3000
	s_cmp_lg_u32 s60, 31
	s_cselect_b32 s60, s59, 0x1f00
	s_cmp_eq_u32 s62, 0
	s_cselect_b64 vcc, -1, 0
	v_cndmask_b32_e32 v66, v156, v154, vcc
	v_cndmask_b32_e32 v67, v157, v155, vcc
	v_cndmask_b32_e32 v100, v160, v158, vcc
	v_cndmask_b32_e32 v101, v161, v159, vcc
	s_mov_b32 m0, s64
	buffer_load_dword v162, v137, s[4:7], s60 offen
	buffer_load_dword v163, v137, s[8:11], s60 offen
	buffer_load_dword v164, v137, s[12:15], s60 offen
	buffer_load_dword v165, v137, s[16:19], s60 offen
	buffer_load_dword v166, v137, s[20:23], s60 offen
	buffer_load_dword v167, v137, s[24:27], s60 offen
	buffer_load_dword v168, v137, s[28:31], s60 offen
	buffer_load_dword v169, v137, s[36:39], s60 offen
	s_waitcnt vmcnt(9)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[34:37], v[2:5], a[60:63],  v98, v141 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[66:69], v66 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[34:37], v[6:9], a[64:67],  v98, v141 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[70:73], v66 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71],  v[34:37], v[10:13], a[68:71],  v98, v143 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[74:77], v66 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75],  v[34:37], v[14:17], a[72:75],  v98, v143 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[78:81], v66 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63],  v[50:53], v[18:21], a[60:63],  v98, v141 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[82:85], v67 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[50:53], v[22:25], a[64:67],  v98, v141 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[86:89], v67 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71],  v[50:53], v[26:29], a[68:71],  v98, v143 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[90:93], v67 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75],  v[50:53], v[30:33], a[72:75],  v98, v143 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[94:97], v67 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79],  v[38:41], v[2:5], a[76:79],  v98, v141 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83],  v[38:41], v[6:9], a[80:83],  v98, v141 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87],  v[38:41], v[10:13], a[84:87],  v98, v143 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91],  v[38:41], v[14:17], a[88:91],  v98, v143 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79],  v[54:57], v[18:21], a[76:79],  v98, v141 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83],  v[54:57], v[22:25], a[80:83],  v98, v141 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87],  v[54:57], v[26:29], a[84:87],  v98, v143 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91],  v[54:57], v[30:33], a[88:91],  v98, v143 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95],  v[42:45], v[2:5], a[92:95],  v99, v141 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99],  v[42:45], v[6:9], a[96:99],  v99, v141 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[42:45], v[10:13], a[100:103], v99, v143 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[42:45], v[14:17], a[104:107], v99, v143 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95],  v[58:61], v[18:21], a[92:95],  v99, v141 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99],  v[58:61], v[22:25], a[96:99],  v99, v141 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[58:61], v[26:29], a[100:103], v99, v143 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[58:61], v[30:33], a[104:107], v99, v143 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[46:49], v[2:5], a[108:111], v99, v141 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[46:49], v[6:9], a[112:115], v99, v141 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[46:49], v[10:13], a[116:119], v99, v143 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[46:49], v[14:17], a[120:123], v99, v143 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[62:65], v[18:21], a[108:111], v99, v141 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[62:65], v[22:25], a[112:115], v99, v141 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[62:65], v[26:29], a[116:119], v99, v143 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[62:65], v[30:33], a[120:123], v99, v143 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_addk_i32 s59, 0x100
	s_waitcnt vmcnt(8)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[34:37], v[66:69], a[32:35],  v98, v142 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[98:101], v100 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[34:37], v[70:73], a[36:39],  v98, v142 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[102:105], v100 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[34:37], v[74:77], a[40:43],  v98, v144 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[106:109], v100 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[34:37], v[78:81], a[44:47],  v98, v144 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[110:113], v100 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[50:53], v[82:85], a[32:35],  v98, v142 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[114:117], v101 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[50:53], v[86:89], a[36:39],  v98, v142 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[118:121], v101 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[50:53], v[90:93], a[40:43],  v98, v144 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[122:125], v101 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[50:53], v[94:97], a[44:47],  v98, v144 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[126:129], v101 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[38:41], v[66:69], a[48:51],  v98, v142 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[38:41], v[70:73], a[52:55],  v98, v142 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[38:41], v[74:77], a[56:59],  v98, v144 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127],  v[38:41], v[78:81], a[124:127],  v98, v144 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[54:57], v[82:85], a[48:51],  v98, v142 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[54:57], v[86:89], a[52:55],  v98, v142 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[54:57], v[90:93], a[56:59],  v98, v144 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127],  v[54:57], v[94:97], a[124:127],  v98, v144 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[42:45], v[66:69], a[128:131],  v99, v142 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[42:45], v[70:73], a[132:135],  v99, v142 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[42:45], v[74:77], a[136:139], v99, v144 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[42:45], v[78:81], a[140:143], v99, v144 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[58:61], v[82:85], a[128:131],  v99, v142 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[58:61], v[86:89], a[132:135],  v99, v142 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[58:61], v[90:93], a[136:139], v99, v144 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[58:61], v[94:97], a[140:143], v99, v144 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[46:49], v[66:69], a[144:147], v99, v142 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[46:49], v[70:73], a[148:151], v99, v142 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[46:49], v[74:77], a[152:155], v99, v144 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[46:49], v[78:81], a[156:159], v99, v144 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[62:65], v[82:85], a[144:147], v99, v142 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[62:65], v[86:89], a[148:151], v99, v142 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[62:65], v[90:93], a[152:155], v99, v144 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[62:65], v[94:97], a[156:159], v99, v144 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(8)
	;;#ASMEND
	s_barrier
	buffer_load_dwordx4 v130, s[0:3], s63 offen lds
	s_mov_b32 m0, s71
	v_cndmask_b32_e32 v34, v145, v148, vcc
	buffer_load_dwordx4 v131, s[0:3], s63 offen lds
	s_mov_b32 m0, s72
	v_cndmask_b32_e32 v35, v147, v149, vcc
	buffer_load_dwordx4 v138, s[0:3], s63 offen lds
	s_mov_b32 m0, s73
	s_mov_b32 s60, s61
	buffer_load_dwordx4 v132, s[0:3], s63 offen lds
	s_mov_b32 m0, s65
	v_cndmask_b32_e32 v170, v150, v152, vcc
	buffer_load_dwordx4 v130, s[0:3], s68 offen lds
	s_mov_b32 m0, s74
	v_cndmask_b32_e32 v171, v151, v153, vcc
	buffer_load_dwordx4 v131, s[0:3], s68 offen lds
	s_mov_b32 m0, s75
	s_cmp_eq_u32 s61, 32
	buffer_load_dwordx4 v138, s[0:3], s68 offen lds
	s_mov_b32 m0, s76
	s_nop 0
	buffer_load_dwordx4 v132, s[0:3], s68 offen lds
	s_mov_b32 m0, s66
	s_nop 0
	buffer_load_dwordx4 v134, s[40:43], s69 offen lds
	s_mov_b32 m0, s77
	s_nop 0
	buffer_load_dwordx4 v133, s[40:43], s69 offen lds
	s_mov_b32 m0, s78
	s_nop 0
	buffer_load_dwordx4 v135, s[40:43], s69 offen lds
	s_mov_b32 m0, s79
	s_nop 0
	buffer_load_dwordx4 v136, s[40:43], s69 offen lds
	s_mov_b32 m0, s67
	s_nop 0
	buffer_load_dwordx4 v134, s[40:43], s70 offen lds
	s_mov_b32 m0, s80
	s_nop 0
	buffer_load_dwordx4 v133, s[40:43], s70 offen lds
	s_mov_b32 m0, s81
	s_nop 0
	buffer_load_dwordx4 v135, s[40:43], s70 offen lds
	s_mov_b32 m0, s82
	s_nop 0
	buffer_load_dwordx4 v136, s[40:43], s70 offen lds
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[98:101], v[2:5], a[160:163],  v139, v141 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v34 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[98:101], v[6:9], a[164:167],  v139, v141 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v34 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171],  v[98:101], v[10:13], a[168:171],  v139, v143 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v34 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175],  v[98:101], v[14:17], a[172:175],  v139, v143 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v34 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[114:117], v[18:21], a[160:163],  v139, v141 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v35 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[114:117], v[22:25], a[164:167],  v139, v141 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v35 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171],  v[114:117], v[26:29], a[168:171],  v139, v143 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v35 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175],  v[114:117], v[30:33], a[172:175],  v139, v143 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[62:65], v35 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179],  v[102:105], v[2:5], a[176:179],  v139, v141 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183],  v[102:105], v[6:9], a[180:183],  v139, v141 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187],  v[102:105], v[10:13], a[184:187],  v139, v143 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191],  v[102:105], v[14:17], a[188:191],  v139, v143 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179],  v[118:121], v[18:21], a[176:179],  v139, v141 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183],  v[118:121], v[22:25], a[180:183],  v139, v141 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187],  v[118:121], v[26:29], a[184:187],  v139, v143 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191],  v[118:121], v[30:33], a[188:191],  v139, v143 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3],  v[106:109], v[2:5], a[0:3],  v140, v141 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7],  v[106:109], v[6:9], a[4:7],  v140, v141 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[106:109], v[10:13], a[8:11], v140, v143 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[106:109], v[14:17], a[12:15], v140, v143 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3],  v[122:125], v[18:21], a[0:3],  v140, v141 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7],  v[122:125], v[22:25], a[4:7],  v140, v141 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[122:125], v[26:29], a[8:11], v140, v143 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[122:125], v[30:33], a[12:15], v140, v143 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[110:113], v[2:5], a[16:19], v140, v141 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[110:113], v[6:9], a[20:23], v140, v141 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[110:113], v[10:13], a[24:27], v140, v143 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[110:113], v[14:17], a[28:31], v140, v143 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[126:129], v[18:21], a[16:19], v140, v141 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[126:129], v[22:25], a[20:23], v140, v141 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[126:129], v[26:29], a[24:27], v140, v143 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[126:129], v[30:33], a[28:31], v140, v143 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[98:101], v[66:69], a[192:195],  v139, v142 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[2:5], v170 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[98:101], v[70:73], a[196:199],  v139, v142 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[6:9], v170 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[98:101], v[74:77], a[200:203],  v139, v144 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[10:13], v170 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[98:101], v[78:81], a[204:207],  v139, v144 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[14:17], v170 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[114:117], v[82:85], a[192:195],  v139, v142 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[18:21], v171 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[114:117], v[86:89], a[196:199],  v139, v142 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[22:25], v171 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[114:117], v[90:93], a[200:203],  v139, v144 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[26:29], v171 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[114:117], v[94:97], a[204:207],  v139, v144 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v171 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[102:105], v[66:69], a[208:211],  v139, v142 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[102:105], v[70:73], a[212:215],  v139, v142 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[102:105], v[74:77], a[216:219],  v139, v144 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[102:105], v[78:81], a[220:223],  v139, v144 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[118:121], v[82:85], a[208:211],  v139, v142 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[118:121], v[86:89], a[212:215],  v139, v142 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[118:121], v[90:93], a[216:219],  v139, v144 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[118:121], v[94:97], a[220:223],  v139, v144 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[106:109], v[66:69], a[224:227],  v140, v142 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[106:109], v[70:73], a[228:231],  v140, v142 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[106:109], v[74:77], a[232:235], v140, v144 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[106:109], v[78:81], a[236:239], v140, v144 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[122:125], v[82:85], a[224:227],  v140, v142 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[122:125], v[86:89], a[228:231],  v140, v142 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[122:125], v[90:93], a[232:235], v140, v144 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[122:125], v[94:97], a[236:239], v140, v144 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[110:113], v[66:69], a[240:243], v140, v142 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[110:113], v[70:73], a[244:247], v140, v142 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[110:113], v[74:77], a[248:251], v140, v144 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[110:113], v[78:81], a[252:255], v140, v144 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[126:129], v[82:85], a[240:243], v140, v142 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[126:129], v[86:89], a[244:247], v140, v142 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[126:129], v[90:93], a[248:251], v140, v144 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[126:129], v[94:97], a[252:255], v140, v144 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_waitcnt vmcnt(23)
	v_mov_b32_e32 v98, v162
	s_waitcnt vmcnt(22)
	v_mov_b32_e32 v139, v163
	s_waitcnt vmcnt(21)
	v_mov_b32_e32 v99, v164
	s_waitcnt vmcnt(20)
	v_mov_b32_e32 v140, v165
	s_waitcnt vmcnt(19)
	v_mov_b32_e32 v141, v166
	s_waitcnt vmcnt(18)
	v_mov_b32_e32 v142, v167
	s_waitcnt vmcnt(17)
	v_mov_b32_e32 v143, v168
	s_waitcnt vmcnt(16)
	v_mov_b32_e32 v144, v169
	s_cbranch_scc0 .LBB0_1
; %bb.2:
	v_accvgpr_read_b32 v137, a123
	v_accvgpr_read_b32 v197, a99
	v_lshl_or_b32 v1, s33, 2, v1
	v_accvgpr_read_b32 v135, a121
	v_accvgpr_read_b32 v134, a120
	v_accvgpr_read_b32 v196, a98
	v_lshl_or_b32 v2, s45, 2, v146
	v_mul_lo_u32 v3, v1, s46
	v_accvgpr_read_b32 v185, a111
	v_pk_mul_f32 v[236:237], v[196:197], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[196:197], v[134:135], s[44:45] op_sel_hi:[1,0]
	v_add_lshl_u32 v134, v3, v2, 6
	v_lshrrev_b32_e32 v4, 2, v0
	v_accvgpr_read_b32 v183, a109
	v_accvgpr_read_b32 v182, a108
	v_ashrrev_i32_e32 v135, 31, v134
	v_and_b32_e32 v4, 12, v4
	v_and_b32_e32 v0, 15, v0
	v_pk_mul_f32 v[250:251], v[182:183], s[44:45] op_sel_hi:[1,0]
	v_lshl_add_u64 v[182:183], v[134:135], 1, s[34:35]
	v_mad_u64_u32 v[134:135], s[0:1], v4, s46, v[0:1]
	v_ashrrev_i32_e32 v135, 31, v134
	v_accvgpr_read_b32 v189, a107
	v_lshlrev_b64 v[4:5], 1, v[134:135]
	v_add_u32_e32 v134, s46, v134
	v_accvgpr_read_b32 v181, a115
	v_accvgpr_read_b32 v187, a105
	v_accvgpr_read_b32 v186, a104
	v_ashrrev_i32_e32 v135, 31, v134
	v_accvgpr_read_b32 v177, a119
	v_accvgpr_read_b32 v180, a114
	v_accvgpr_read_b32 v193, a103
	v_accvgpr_read_b32 v233, a63
	v_pk_mul_f32 v[246:247], v[186:187], s[44:45] op_sel_hi:[1,0]
	v_lshlrev_b64 v[186:187], 1, v[134:135]
	v_add_u32_e32 v134, s46, v134
	v_accvgpr_read_b32 v176, a118
	v_accvgpr_read_b32 v175, a117
	v_accvgpr_read_b32 v174, a116
	v_accvgpr_read_b32 v188, a106
	v_accvgpr_read_b32 v192, a102
	v_accvgpr_read_b32 v231, a61
	v_accvgpr_read_b32 v230, a60
	v_pk_mul_f32 v[252:253], v[180:181], s[44:45] op_sel_hi:[1,0]
	v_add_u32_e32 v180, s46, v134
	v_accvgpr_read_b32 v191, a101
	v_accvgpr_read_b32 v190, a100
	v_accvgpr_read_b32 v225, a71
	v_accvgpr_read_b32 v229, a67
	v_pk_mul_f32 v[230:231], v[230:231], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[240:241], v[192:193], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[244:245], v[188:189], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[188:189], v[176:177], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[192:193], v[174:175], s[44:45] op_sel_hi:[1,0]
	v_lshl_add_u64 v[174:175], v[182:183], 0, v[4:5]
	v_lshl_add_u64 v[176:177], v[182:183], 0, v[186:187]
	v_ashrrev_i32_e32 v135, 31, v134
	v_ashrrev_i32_e32 v181, 31, v180
	v_accvgpr_read_b32 v179, a113
	v_accvgpr_read_b32 v178, a112
	v_accvgpr_read_b32 v184, a110
	v_accvgpr_read_b32 v221, a75
	v_accvgpr_read_b32 v223, a69
	v_accvgpr_read_b32 v222, a68
	v_accvgpr_read_b32 v227, a65
	v_accvgpr_read_b32 v226, a64
	v_accvgpr_read_b32 v232, a62
	v_pk_mul_f32 v[242:243], v[190:191], s[44:45] op_sel_hi:[1,0]
	global_store_short_d16_hi v[174:175], v230, off
	global_store_short_d16_hi v[176:177], v231, off
	v_lshlrev_b64 v[230:231], 1, v[134:135]
	v_lshlrev_b64 v[190:191], 1, v[180:181]
	v_accvgpr_read_b32 v220, a74
	v_accvgpr_read_b32 v219, a73
	v_accvgpr_read_b32 v218, a72
	v_accvgpr_read_b32 v224, a70
	v_accvgpr_read_b32 v228, a66
	v_pk_mul_f32 v[148:149], v[232:233], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[226:227], v[226:227], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[222:223], v[222:223], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[248:249], v[184:185], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[254:255], v[178:179], s[44:45] op_sel_hi:[1,0]
	v_lshl_add_u64 v[178:179], v[182:183], 0, v[230:231]
	v_lshl_add_u64 v[184:185], v[182:183], 0, v[190:191]
	s_mul_i32 s0, s46, 13
	v_pk_mul_f32 v[228:229], v[228:229], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[224:225], v[224:225], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[220:221], v[220:221], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[218:219], v[218:219], s[44:45] op_sel_hi:[1,0]
	global_store_short_d16_hi v[178:179], v148, off
	global_store_short_d16_hi v[184:185], v149, off
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
	v_add_u32_e32 v174, s46, v174
	v_ashrrev_i32_e32 v175, 31, v174
	v_lshlrev_b64 v[226:227], 1, v[174:175]
	v_add_u32_e32 v174, s46, v174
	v_accvgpr_read_b32 v201, a95
	v_accvgpr_read_b32 v217, a79
	v_add_u32_e32 v218, s46, v174
	v_accvgpr_read_b32 v200, a94
	v_accvgpr_read_b32 v199, a93
	v_accvgpr_read_b32 v198, a92
	v_accvgpr_read_b32 v209, a87
	v_accvgpr_read_b32 v213, a83
	v_accvgpr_read_b32 v215, a77
	v_accvgpr_read_b32 v214, a76
	v_ashrrev_i32_e32 v175, 31, v174
	v_ashrrev_i32_e32 v219, 31, v218
	v_accvgpr_read_b32 v205, a91
	v_accvgpr_read_b32 v207, a85
	v_accvgpr_read_b32 v206, a84
	v_accvgpr_read_b32 v211, a81
	v_accvgpr_read_b32 v210, a80
	v_accvgpr_read_b32 v216, a78
	v_pk_mul_f32 v[214:215], v[214:215], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[232:233], v[200:201], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[234:235], v[198:199], s[44:45] op_sel_hi:[1,0]
	v_lshl_add_u64 v[184:185], v[182:183], 0, v[224:225]
	v_lshl_add_u64 v[200:201], v[182:183], 0, v[226:227]
	v_lshlrev_b64 v[228:229], 1, v[174:175]
	v_lshlrev_b64 v[198:199], 1, v[218:219]
	v_accvgpr_read_b32 v204, a90
	v_accvgpr_read_b32 v203, a89
	v_accvgpr_read_b32 v202, a88
	v_accvgpr_read_b32 v208, a86
	v_accvgpr_read_b32 v212, a82
	v_pk_mul_f32 v[216:217], v[216:217], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[210:211], v[210:211], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[206:207], v[206:207], s[44:45] op_sel_hi:[1,0]
	global_store_short_d16_hi v[184:185], v214, off
	global_store_short_d16_hi v[200:201], v215, off
	v_lshl_add_u64 v[214:215], v[182:183], 0, v[228:229]
	v_lshl_add_u64 v[220:221], v[182:183], 0, v[198:199]
	v_pk_mul_f32 v[212:213], v[212:213], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[208:209], v[208:209], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[204:205], v[204:205], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[202:203], v[202:203], s[44:45] op_sel_hi:[1,0]
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
	v_add_u32_e32 v184, s46, v184
	v_ashrrev_i32_e32 v185, 31, v184
	v_lshlrev_b64 v[202:203], 1, v[184:185]
	v_add_u32_e32 v184, s46, v184
	v_ashrrev_i32_e32 v185, 31, v184
	v_lshlrev_b64 v[204:205], 1, v[184:185]
	v_add_u32_e32 v184, s46, v184
	v_ashrrev_i32_e32 v185, 31, v184
	v_accvgpr_read_b32 v195, a97
	v_accvgpr_read_b32 v194, a96
	v_lshlrev_b64 v[206:207], 1, v[184:185]
	v_add_u32_e32 v184, s0, v184
	v_pk_mul_f32 v[238:239], v[194:195], s[44:45] op_sel_hi:[1,0]
	v_lshl_add_u64 v[208:209], v[182:183], 0, v[200:201]
	v_lshl_add_u64 v[210:211], v[182:183], 0, v[202:203]
	v_lshl_add_u64 v[212:213], v[182:183], 0, v[204:205]
	v_lshl_add_u64 v[214:215], v[182:183], 0, v[206:207]
	v_ashrrev_i32_e32 v185, 31, v184
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
	v_add_u32_e32 v184, s46, v184
	v_ashrrev_i32_e32 v185, 31, v184
	v_lshlrev_b64 v[210:211], 1, v[184:185]
	v_add_u32_e32 v184, s46, v184
	v_ashrrev_i32_e32 v185, 31, v184
	v_lshlrev_b64 v[212:213], 1, v[184:185]
	v_add_u32_e32 v184, s46, v184
	v_ashrrev_i32_e32 v185, 31, v184
	v_lshlrev_b64 v[214:215], 1, v[184:185]
	v_or_b32_e32 v0, 2, v1
	v_accvgpr_read_b32 v136, a122
	v_lshl_add_u64 v[236:237], v[182:183], 0, v[208:209]
	v_lshl_add_u64 v[238:239], v[182:183], 0, v[210:211]
	v_lshl_add_u64 v[240:241], v[182:183], 0, v[212:213]
	v_lshl_add_u64 v[182:183], v[182:183], 0, v[214:215]
	v_mul_lo_u32 v0, v0, s46
	v_pk_mul_f32 v[194:195], v[136:137], s[44:45] op_sel_hi:[1,0]
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
	v_accvgpr_read_b32 v173, a163
	v_ashrrev_i32_e32 v183, 31, v182
	v_accvgpr_read_b32 v158, a172
	v_accvgpr_read_b32 v171, a161
	v_accvgpr_read_b32 v170, a160
	v_lshl_add_u64 v[182:183], v[182:183], 1, s[34:35]
	v_accvgpr_read_b32 v154, a176
	v_accvgpr_read_b32 v159, a173
	v_accvgpr_read_b32 v160, a174
	v_accvgpr_read_b32 v161, a175
	v_accvgpr_read_b32 v162, a168
	v_accvgpr_read_b32 v169, a167
	v_accvgpr_read_b32 v172, a162
	v_pk_mul_f32 v[170:171], v[170:171], s[44:45] op_sel_hi:[1,0]
	v_lshl_add_u64 v[184:185], v[182:183], 0, v[4:5]
	v_lshl_add_u64 v[188:189], v[182:183], 0, v[186:187]
	v_accvgpr_read_b32 v138, a188
	v_accvgpr_read_b32 v155, a177
	v_accvgpr_read_b32 v163, a169
	v_accvgpr_read_b32 v164, a170
	v_accvgpr_read_b32 v165, a171
	v_accvgpr_read_b32 v168, a166
	v_accvgpr_read_b32 v167, a165
	v_accvgpr_read_b32 v166, a164
	v_pk_mul_f32 v[172:173], v[172:173], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[160:161], v[160:161], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[158:159], v[158:159], s[44:45] op_sel_hi:[1,0]
	global_store_short_d16_hi v[184:185], v170, off
	global_store_short_d16_hi v[188:189], v171, off
	v_lshl_add_u64 v[170:171], v[182:183], 0, v[230:231]
	v_lshl_add_u64 v[192:193], v[182:183], 0, v[190:191]
	v_accvgpr_read_b32 v133, a3
	v_accvgpr_read_b32 v139, a189
	v_accvgpr_read_b32 v140, a190
	v_accvgpr_read_b32 v141, a191
	v_accvgpr_read_b32 v142, a184
	v_accvgpr_read_b32 v150, a180
	v_accvgpr_read_b32 v156, a178
	v_accvgpr_read_b32 v157, a179
	v_pk_mul_f32 v[168:169], v[168:169], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[166:167], v[166:167], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[164:165], v[164:165], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[162:163], v[162:163], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[154:155], v[154:155], s[44:45] op_sel_hi:[1,0]
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
	v_accvgpr_read_b32 v131, a1
	v_accvgpr_read_b32 v130, a0
	v_accvgpr_read_b32 v143, a185
	v_accvgpr_read_b32 v144, a186
	v_accvgpr_read_b32 v145, a187
	v_accvgpr_read_b32 v151, a181
	v_accvgpr_read_b32 v152, a182
	v_accvgpr_read_b32 v153, a183
	v_pk_mul_f32 v[156:157], v[156:157], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[140:141], v[140:141], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[138:139], v[138:139], s[44:45] op_sel_hi:[1,0]
	global_store_short_d16_hi v[158:159], v154, off
	global_store_short_d16_hi v[160:161], v155, off
	v_lshl_add_u64 v[154:155], v[182:183], 0, v[228:229]
	v_lshl_add_u64 v[162:163], v[182:183], 0, v[198:199]
	v_accvgpr_read_b32 v121, a15
	v_accvgpr_read_b32 v125, a11
	v_accvgpr_read_b32 v129, a7
	v_accvgpr_read_b32 v132, a2
	v_pk_mul_f32 v[152:153], v[152:153], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[150:151], v[150:151], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[144:145], v[144:145], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[142:143], v[142:143], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[130:131], v[130:131], s[44:45] op_sel_hi:[1,0]
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
	v_accvgpr_read_b32 v117, a19
	v_accvgpr_read_b32 v120, a14
	v_accvgpr_read_b32 v119, a13
	v_accvgpr_read_b32 v118, a12
	v_accvgpr_read_b32 v124, a10
	v_accvgpr_read_b32 v123, a9
	v_accvgpr_read_b32 v122, a8
	v_accvgpr_read_b32 v128, a6
	v_accvgpr_read_b32 v127, a5
	v_accvgpr_read_b32 v126, a4
	v_pk_mul_f32 v[132:133], v[132:133], s[44:45] op_sel_hi:[1,0]
	global_store_short_d16_hi v[138:139], v130, off
	global_store_short_d16_hi v[140:141], v131, off
	v_lshl_add_u64 v[130:131], v[182:183], 0, v[204:205]
	v_lshl_add_u64 v[142:143], v[182:183], 0, v[206:207]
	v_or_b32_e32 v1, 2, v2
	v_accvgpr_read_b32 v105, a31
	v_accvgpr_read_b32 v115, a17
	v_accvgpr_read_b32 v114, a16
	v_pk_mul_f32 v[128:129], v[128:129], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[126:127], v[126:127], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[124:125], v[124:125], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[122:123], v[122:123], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[120:121], v[120:121], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[118:119], v[118:119], s[44:45] op_sel_hi:[1,0]
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
	v_accvgpr_read_b32 v104, a30
	v_accvgpr_read_b32 v109, a27
	v_accvgpr_read_b32 v113, a23
	v_accvgpr_read_b32 v116, a18
	v_accvgpr_read_b32 v137, a59
	v_accvgpr_read_b32 v223, a35
	v_pk_mul_f32 v[114:115], v[114:115], s[44:45] op_sel_hi:[1,0]
	v_lshl_add_u64 v[118:119], v[182:183], 0, v[208:209]
	v_lshl_add_u64 v[120:121], v[182:183], 0, v[210:211]
	v_ashrrev_i32_e32 v131, 31, v130
	v_accvgpr_read_b32 v103, a29
	v_accvgpr_read_b32 v102, a28
	v_accvgpr_read_b32 v108, a26
	v_accvgpr_read_b32 v107, a25
	v_accvgpr_read_b32 v106, a24
	v_accvgpr_read_b32 v112, a22
	v_accvgpr_read_b32 v111, a21
	v_accvgpr_read_b32 v110, a20
	v_accvgpr_read_b32 v135, a57
	v_accvgpr_read_b32 v134, a56
	v_accvgpr_read_b32 v219, a39
	v_accvgpr_read_b32 v221, a33
	v_accvgpr_read_b32 v220, a32
	v_pk_mul_f32 v[116:117], v[116:117], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[104:105], v[104:105], s[44:45] op_sel_hi:[1,0]
	global_store_short_d16_hi v[118:119], v114, off
	global_store_short_d16_hi v[120:121], v115, off
	v_lshl_add_u64 v[114:115], v[182:183], 0, v[212:213]
	v_lshl_add_u64 v[122:123], v[182:183], 0, v[214:215]
	v_lshl_add_u64 v[130:131], v[130:131], 1, s[34:35]
	v_accvgpr_read_b32 v98, a124
	v_accvgpr_read_b32 v136, a58
	v_accvgpr_read_b32 v177, a51
	v_accvgpr_read_b32 v181, a47
	v_accvgpr_read_b32 v235, a43
	v_accvgpr_read_b32 v218, a38
	v_accvgpr_read_b32 v217, a37
	v_accvgpr_read_b32 v216, a36
	v_accvgpr_read_b32 v222, a34
	v_pk_mul_f32 v[112:113], v[112:113], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[110:111], v[110:111], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[108:109], v[108:109], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[106:107], v[106:107], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[102:103], v[102:103], s[44:45] op_sel_hi:[1,0]
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
	v_pk_mul_f32 v[104:105], v[220:221], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[128:129], v[134:135], s[44:45] op_sel_hi:[1,0]
	v_lshl_add_u64 v[132:133], v[130:131], 0, v[4:5]
	v_lshl_add_u64 v[134:135], v[130:131], 0, v[186:187]
	v_accvgpr_read_b32 v94, a128
	v_accvgpr_read_b32 v99, a125
	v_accvgpr_read_b32 v100, a126
	v_accvgpr_read_b32 v101, a127
	v_accvgpr_read_b32 v149, a55
	v_accvgpr_read_b32 v176, a50
	v_accvgpr_read_b32 v175, a49
	v_accvgpr_read_b32 v174, a48
	v_accvgpr_read_b32 v180, a46
	v_accvgpr_read_b32 v179, a45
	v_accvgpr_read_b32 v178, a44
	v_accvgpr_read_b32 v234, a42
	v_accvgpr_read_b32 v233, a41
	v_accvgpr_read_b32 v232, a40
	v_pk_mul_f32 v[102:103], v[222:223], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[106:107], v[218:219], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[108:109], v[216:217], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[126:127], v[136:137], s[44:45] op_sel_hi:[1,0]
	global_store_short_d16_hi v[132:133], v104, off
	global_store_short_d16_hi v[134:135], v105, off
	v_lshl_add_u64 v[104:105], v[130:131], 0, v[230:231]
	v_lshl_add_u64 v[136:137], v[130:131], 0, v[190:191]
	v_accvgpr_read_b32 v82, a140
	v_accvgpr_read_b32 v95, a129
	v_accvgpr_read_b32 v148, a54
	v_accvgpr_read_b32 v147, a53
	v_accvgpr_read_b32 v146, a52
	v_pk_mul_f32 v[110:111], v[234:235], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[112:113], v[232:233], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[114:115], v[180:181], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[116:117], v[178:179], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[118:119], v[176:177], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[120:121], v[174:175], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[100:101], v[100:101], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[98:99], v[98:99], s[44:45] op_sel_hi:[1,0]
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
	v_accvgpr_read_b32 v78, a144
	v_accvgpr_read_b32 v83, a141
	v_accvgpr_read_b32 v84, a142
	v_accvgpr_read_b32 v85, a143
	v_accvgpr_read_b32 v86, a136
	v_accvgpr_read_b32 v90, a132
	v_accvgpr_read_b32 v96, a130
	v_accvgpr_read_b32 v97, a131
	v_pk_mul_f32 v[122:123], v[148:149], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[124:125], v[146:147], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[94:95], v[94:95], s[44:45] op_sel_hi:[1,0]
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
	v_accvgpr_read_b32 v66, a156
	v_accvgpr_read_b32 v79, a145
	v_accvgpr_read_b32 v87, a137
	v_accvgpr_read_b32 v88, a138
	v_accvgpr_read_b32 v89, a139
	v_accvgpr_read_b32 v91, a133
	v_accvgpr_read_b32 v92, a134
	v_accvgpr_read_b32 v93, a135
	v_pk_mul_f32 v[96:97], v[96:97], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[84:85], v[84:85], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[82:83], v[82:83], s[44:45] op_sel_hi:[1,0]
	global_store_short_d16_hi v[98:99], v94, off
	global_store_short_d16_hi v[100:101], v95, off
	v_lshl_add_u64 v[94:95], v[130:131], 0, v[204:205]
	v_lshl_add_u64 v[102:103], v[130:131], 0, v[206:207]
	v_accvgpr_read_b32 v67, a157
	v_accvgpr_read_b32 v70, a152
	v_accvgpr_read_b32 v74, a148
	v_accvgpr_read_b32 v80, a146
	v_accvgpr_read_b32 v81, a147
	v_pk_mul_f32 v[92:93], v[92:93], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[90:91], v[90:91], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[88:89], v[88:89], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[86:87], v[86:87], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[78:79], v[78:79], s[44:45] op_sel_hi:[1,0]
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
	v_accvgpr_read_b32 v68, a158
	v_accvgpr_read_b32 v69, a159
	v_accvgpr_read_b32 v71, a153
	v_accvgpr_read_b32 v72, a154
	v_accvgpr_read_b32 v73, a155
	v_accvgpr_read_b32 v75, a149
	v_accvgpr_read_b32 v76, a150
	v_accvgpr_read_b32 v77, a151
	v_pk_mul_f32 v[80:81], v[80:81], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[66:67], v[66:67], s[44:45] op_sel_hi:[1,0]
	global_store_short_d16_hi v[82:83], v78, off
	global_store_short_d16_hi v[84:85], v79, off
	v_lshl_add_u64 v[78:79], v[130:131], 0, v[212:213]
	v_lshl_add_u64 v[86:87], v[130:131], 0, v[214:215]
	v_pk_mul_f32 v[76:77], v[76:77], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[74:75], v[74:75], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[72:73], v[72:73], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[70:71], v[70:71], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[68:69], v[68:69], s[44:45] op_sel_hi:[1,0]
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
	v_lshl_add_u64 v[66:67], v[66:67], 1, s[34:35]
	v_accvgpr_read_b32 v46, a208
	v_accvgpr_read_b32 v51, a205
	v_accvgpr_read_b32 v54, a200
	v_accvgpr_read_b32 v58, a196
	v_accvgpr_read_b32 v64, a194
	v_accvgpr_read_b32 v65, a195
	v_pk_mul_f32 v[62:63], v[62:63], s[44:45] op_sel_hi:[1,0]
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
	v_pk_mul_f32 v[64:65], v[64:65], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], s[44:45] op_sel_hi:[1,0]
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
	v_pk_mul_f32 v[60:61], v[60:61], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[56:57], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[54:55], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[52:53], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], s[44:45] op_sel_hi:[1,0]
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
	v_pk_mul_f32 v[48:49], v[48:49], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], s[44:45] op_sel_hi:[1,0]
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
	v_pk_mul_f32 v[44:45], v[44:45], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[36:37], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[30:31], v[30:31], s[44:45] op_sel_hi:[1,0]
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
	v_pk_mul_f32 v[32:33], v[32:33], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], s[44:45] op_sel_hi:[1,0]
	global_store_short_d16_hi v[0:1], v30, off
	global_store_short_d16_hi v[34:35], v31, off
	v_lshl_add_u64 v[30:31], v[66:67], 0, v[204:205]
	v_lshl_add_u64 v[36:37], v[66:67], 0, v[206:207]
	v_accvgpr_read_b32 v6, a248
	v_accvgpr_read_b32 v10, a244
	v_accvgpr_read_b32 v16, a242
	v_accvgpr_read_b32 v17, a243
	v_pk_mul_f32 v[28:29], v[28:29], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], s[44:45] op_sel_hi:[1,0]
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
	v_pk_mul_f32 v[16:17], v[16:17], s[44:45] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v3, a253
	v_accvgpr_read_b32 v4, a254
	v_accvgpr_read_b32 v5, a255
	global_store_short_d16_hi v[0:1], v14, off
	global_store_short_d16_hi v[18:19], v15, off
	v_lshl_add_u64 v[14:15], v[66:67], 0, v[212:213]
	v_lshl_add_u64 v[20:21], v[66:67], 0, v[214:215]
	v_pk_mul_f32 v[12:13], v[12:13], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], s[44:45] op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], s[44:45] op_sel_hi:[1,0]
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
	.size	_Z22mxfp4_gluon_cpp_kernel13gluon_globals, .Lfunc_end0-_Z22mxfp4_gluon_cpp_kernel13gluon_globals
                                        ; -- End function
	.set _Z22mxfp4_gluon_cpp_kernel13gluon_globals.num_vgpr, 256
	.set _Z22mxfp4_gluon_cpp_kernel13gluon_globals.num_agpr, 256
	.set _Z22mxfp4_gluon_cpp_kernel13gluon_globals.numbered_sgpr, 83
	.set _Z22mxfp4_gluon_cpp_kernel13gluon_globals.private_seg_size, 0
	.set _Z22mxfp4_gluon_cpp_kernel13gluon_globals.uses_vcc, 1
	.set _Z22mxfp4_gluon_cpp_kernel13gluon_globals.uses_flat_scratch, 0
	.set _Z22mxfp4_gluon_cpp_kernel13gluon_globals.has_dyn_sized_stack, 0
	.set _Z22mxfp4_gluon_cpp_kernel13gluon_globals.has_recursion, 0
	.set _Z22mxfp4_gluon_cpp_kernel13gluon_globals.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 13436
; TotalNumSgprs: 89
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
    .sgpr_count:     89
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
