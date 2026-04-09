	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.text
	.protected	_Z20mxfp8_rewrite_kernel15rewrite_globals ; -- Begin function _Z20mxfp8_rewrite_kernel15rewrite_globals
	.globl	_Z20mxfp8_rewrite_kernel15rewrite_globals
	.p2align	8
	.type	_Z20mxfp8_rewrite_kernel15rewrite_globals,@function
_Z20mxfp8_rewrite_kernel15rewrite_globals: ; @_Z20mxfp8_rewrite_kernel15rewrite_globals
; %bb.0:
	s_load_dwordx2 s[54:55], s[0:1], 0x0
	s_load_dwordx2 s[60:61], s[0:1], 0x20
	s_load_dwordx2 s[56:57], s[0:1], 0x30
	s_load_dwordx2 s[58:59], s[0:1], 0x50
	s_load_dwordx2 s[34:35], s[0:1], 0xc0
	s_load_dwordx2 s[52:53], s[0:1], 0xe0
	s_load_dword s3, s[0:1], 0x110
	s_load_dwordx2 s[4:5], s[0:1], 0x60
	s_load_dwordx2 s[6:7], s[0:1], 0x80
	s_load_dwordx2 s[16:17], s[0:1], 0x90
	s_load_dwordx2 s[18:19], s[0:1], 0xb0
	s_waitcnt lgkmcnt(0)
	s_cmp_lt_i32 s3, 8
	s_cselect_b64 s[8:9], -1, 0
	s_and_b32 s7, s3, 7
	s_cmp_lg_u32 s7, 0
	s_cselect_b64 s[10:11], -1, 0
	s_or_b64 s[8:9], s[8:9], s[10:11]
	s_mov_b32 s59, 0
	s_and_b64 vcc, exec, s[8:9]
	s_cbranch_vccnz .LBB0_2
; %bb.1:
	s_ashr_i32 s7, s2, 31
	s_lshr_b32 s7, s7, 29
	s_add_i32 s7, s2, s7
	s_ashr_i32 s8, s7, 3
	s_and_b32 s7, s7, -8
	s_lshr_b32 s3, s3, 3
	s_sub_i32 s2, s2, s7
	s_mul_i32 s2, s3, s2
	s_add_i32 s2, s2, s8
.LBB0_2:
	s_ashr_i32 s3, s2, 31
	s_lshr_b32 s3, s3, 26
	s_add_i32 s3, s2, s3
	s_ashr_i32 s7, s3, 6
	s_lshl_b32 s53, s7, 1
	s_sub_i32 s7, 32, s53
	s_cmpk_gt_i32 s2, 0x3ff
	s_cselect_b32 s7, s7, 2
	s_abs_i32 s8, s7
	v_cvt_f32_u32_e32 v1, s8
	s_andn2_b32 s3, s3, 63
	s_load_dword s33, s[0:1], 0xf0
	s_sub_i32 s0, s2, s3
	v_rcp_iflag_f32_e32 v1, v1
	s_xor_b32 s1, s0, s7
	s_sub_i32 s3, 0, s8
	s_ashr_i32 s63, s1, 31
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	s_abs_i32 s2, s0
	s_mov_b32 s39, 0x110000
	s_mov_b32 s38, -1
	v_readfirstlane_b32 s1, v1
	s_mul_i32 s3, s3, s1
	s_mul_hi_u32 s3, s1, s3
	s_add_i32 s1, s1, s3
	s_mul_hi_u32 s1, s2, s1
	s_mul_i32 s3, s1, s8
	s_sub_i32 s2, s2, s3
	s_add_i32 s9, s1, 1
	s_sub_i32 s3, s2, s8
	s_cmp_ge_u32 s2, s8
	s_cselect_b32 s1, s9, s1
	s_cselect_b32 s2, s3, s2
	s_add_i32 s3, s1, 1
	s_cmp_ge_u32 s2, s8
	s_cselect_b32 s1, s3, s1
	s_xor_b32 s64, s1, s63
	s_sub_i32 s12, s64, s63
	s_mul_i32 s1, s12, s7
	s_sub_i32 s0, s0, s1
	s_add_i32 s53, s53, s0
	v_lshrrev_b32_e32 v1, 7, v0
	s_lshl_b32 s48, s53, 8
	v_lshl_or_b32 v4, v1, 6, s48
	v_ashrrev_i32_e32 v6, 5, v4
	v_mul_lo_u32 v2, v6, s6
	v_ashrrev_i32_e32 v3, 31, v2
	v_or_b32_e32 v5, 0x80, v4
	v_lshl_add_u64 v[2:3], s[4:5], 0, v[2:3]
	s_lshl_b32 s62, s12, 8
	v_readfirstlane_b32 s36, v2
	v_ashrrev_i32_e32 v2, 5, v5
	v_mul_lo_u32 v2, v2, s6
	v_readfirstlane_b32 s0, v3
	v_ashrrev_i32_e32 v3, 31, v2
	s_mov_b64 s[24:25], s[36:37]
	v_lshl_add_u64 v[2:3], s[4:5], 0, v[2:3]
	s_mov_b64 s[26:27], s[38:39]
	v_readfirstlane_b32 s36, v2
	v_or_b32_e32 v2, 1, v6
	v_mul_lo_u32 v2, v2, s6
	v_readfirstlane_b32 s7, v3
	v_ashrrev_i32_e32 v3, 31, v2
	s_mov_b32 s25, s0
	s_mov_b64 s[0:1], s[36:37]
	v_lshl_add_u64 v[2:3], s[4:5], 0, v[2:3]
	s_mov_b64 s[2:3], s[38:39]
	v_readfirstlane_b32 s36, v2
	v_or_b32_e32 v2, 0xa0, v4
	v_ashrrev_i32_e32 v2, 5, v2
	v_mul_lo_u32 v2, v2, s6
	s_mov_b32 s1, s7
	v_readfirstlane_b32 s7, v3
	v_ashrrev_i32_e32 v3, 31, v2
	s_mov_b64 s[8:9], s[36:37]
	v_lshl_add_u64 v[2:3], s[4:5], 0, v[2:3]
	v_bfe_u32 v6, v0, 6, 1
	s_mov_b64 s[10:11], s[38:39]
	v_readfirstlane_b32 s36, v2
	v_lshl_or_b32 v2, v6, 6, s62
	v_ashrrev_i32_e32 v7, 5, v2
	scratch_store_dword off, v2, off        ; 4-byte Folded Spill
	v_mul_lo_u32 v2, v7, s18
	v_readfirstlane_b32 s13, v3
	v_ashrrev_i32_e32 v3, 31, v2
	s_lshl_b32 s19, s18, 2
	s_mov_b32 s9, s7
	s_mov_b64 s[4:5], s[36:37]
	v_lshl_add_u64 v[4:5], s[16:17], 0, v[2:3]
	v_add_u32_e32 v2, s19, v2
	s_mov_b64 s[6:7], s[38:39]
	v_readfirstlane_b32 s36, v4
	v_ashrrev_i32_e32 v3, 31, v2
	s_mov_b64 s[28:29], s[36:37]
	v_lshl_add_u64 v[2:3], s[16:17], 0, v[2:3]
	s_mov_b64 s[30:31], s[38:39]
	v_readfirstlane_b32 s36, v2
	v_or_b32_e32 v2, 1, v7
	v_mul_lo_u32 v2, v2, s18
	v_readfirstlane_b32 s12, v5
	v_readfirstlane_b32 s20, v3
	v_ashrrev_i32_e32 v3, 31, v2
	s_mov_b32 s5, s13
	s_mov_b32 s29, s12
	s_mov_b64 s[12:13], s[36:37]
	v_lshl_add_u64 v[4:5], s[16:17], 0, v[2:3]
	v_add_u32_e32 v2, s19, v2
	s_mov_b64 s[14:15], s[38:39]
	v_readfirstlane_b32 s36, v4
	v_ashrrev_i32_e32 v3, 31, v2
	s_mov_b32 s13, s20
	s_mov_b64 s[20:21], s[36:37]
	v_lshl_add_u64 v[2:3], s[16:17], 0, v[2:3]
	s_mov_b64 s[22:23], s[38:39]
	v_readfirstlane_b32 s36, v2
	v_lshlrev_b32_e32 v2, 2, v0
	v_readfirstlane_b32 s18, v5
	v_readfirstlane_b32 s37, v3
	v_and_b32_e32 v123, 0xfc, v2
	v_lshlrev_b32_e32 v2, 4, v0
	s_movk_i32 s65, 0x70
	s_mov_b32 s21, s18
	s_mov_b64 s[16:17], s[36:37]
	s_mul_i32 s61, s48, s60
	v_bitop3_b32 v66, v2, s65, v0 bitop3:0x48
	v_lshrrev_b32_e32 v67, 3, v0
	s_mov_b64 s[18:19], s[38:39]
	s_ashr_i32 s37, s61, 31
	v_and_b32_e32 v105, 0xc00, v2
	v_mad_u64_u32 v[106:107], s[40:41], v67, s60, v[66:67]
	s_add_u32 s36, s54, s61
	s_addc_u32 s37, s55, s37
	v_or_b32_e32 v107, 0x1000, v105
	s_lshl_b32 s38, s60, 7
	v_or_b32_e32 v4, 0x60, v67
	v_readfirstlane_b32 s41, v107
	v_readfirstlane_b32 s40, v105
	s_mov_b32 m0, s40
	s_lshl_b32 s40, s60, 5
	v_add_u32_e32 v2, s40, v106
	v_add_u32_e32 v3, s40, v2
	v_or_b32_e32 v124, 0x2000, v105
	buffer_load_dwordx4 v106, s[36:39], 0 offen lds
	s_mov_b32 m0, s41
	v_readfirstlane_b32 s40, v124
	buffer_load_dwordx4 v2, s[36:39], 0 offen lds
	s_mov_b32 m0, s40
	v_mad_u64_u32 v[108:109], s[40:41], v4, s60, v[66:67]
	v_or_b32_e32 v109, 0x3000, v105
	s_mul_i32 s62, s62, s58
	s_ashr_i32 s41, s62, 31
	v_readfirstlane_b32 s40, v109
	v_or_b32_e32 v125, 0x10000, v105
	v_mad_u64_u32 v[110:111], s[44:45], v67, s58, v[66:67]
	buffer_load_dwordx4 v3, s[36:39], 0 offen lds
	s_mov_b32 m0, s40
	s_add_u32 s40, s56, s62
	s_addc_u32 s41, s57, s41
	v_or_b32_e32 v111, 0x11000, v105
	s_lshl_b32 s42, s58, 7
	s_mov_b32 s43, s39
	v_readfirstlane_b32 s45, v111
	buffer_load_dwordx4 v108, s[36:39], 0 offen lds
	v_readfirstlane_b32 s44, v125
	s_mov_b32 m0, s44
	s_lshl_b32 s44, s58, 5
	v_add_u32_e32 v5, s44, v110
	v_or_b32_e32 v126, 0x12000, v105
	v_add_u32_e32 v7, s44, v5
	v_readfirstlane_b32 s44, v126
	buffer_load_dwordx4 v110, s[40:43], 0 offen lds
	s_mov_b32 m0, s45
	v_or_b32_e32 v127, 0x14000, v105
	s_mov_b32 s46, s42
	buffer_load_dwordx4 v5, s[40:43], 0 offen lds
	s_mov_b32 m0, s44
	v_mad_u64_u32 v[112:113], s[44:45], v4, s58, v[66:67]
	v_or_b32_e32 v113, 0x13000, v105
	s_mov_b32 s47, s39
	v_readfirstlane_b32 s44, v113
	v_or_b32_e32 v4, 0x15000, v105
	buffer_load_dwordx4 v7, s[40:43], 0 offen lds
	s_mov_b32 m0, s44
	v_or_b32_e32 v128, 0x4000, v105
	buffer_load_dwordx4 v112, s[40:43], 0 offen lds
	s_add_i32 s43, s62, s42
	s_ashr_i32 s45, s43, 31
	s_add_u32 s44, s56, s43
	s_addc_u32 s45, s57, s45
	v_or_b32_e32 v129, 0x5000, v105
	v_readfirstlane_b32 s43, v127
	s_mov_b32 m0, s43
	v_readfirstlane_b32 s43, v4
	v_or_b32_e32 v4, 0x16000, v105
	s_mov_b32 s50, s38
	buffer_load_dwordx4 v110, s[44:47], 0 offen lds
	s_mov_b32 m0, s43
	v_readfirstlane_b32 s43, v4
	v_or_b32_e32 v4, 0x17000, v105
	s_mov_b32 s51, s39
	buffer_load_dwordx4 v5, s[44:47], 0 offen lds
	s_mov_b32 m0, s43
	v_readfirstlane_b32 s43, v4
	v_or_b32_e32 v130, 0x6000, v105
	buffer_load_dwordx4 v7, s[44:47], 0 offen lds
	s_mov_b32 m0, s43
	v_or_b32_e32 v131, 0x7000, v105
	buffer_load_dwordx4 v112, s[44:47], 0 offen lds
	s_or_b32 s46, s48, 0x80
	s_mul_i32 s46, s46, s60
	s_ashr_i32 s43, s46, 31
	s_add_u32 s48, s54, s46
	s_addc_u32 s49, s55, s43
	v_readfirstlane_b32 s43, v128
	s_mov_b32 m0, s43
	v_readfirstlane_b32 s43, v129
	buffer_load_dwordx4 v106, s[48:51], 0 offen lds
	s_mov_b32 m0, s43
	v_readfirstlane_b32 s43, v130
	buffer_load_dwordx4 v2, s[48:51], 0 offen lds
	s_mov_b32 m0, s43
	v_readfirstlane_b32 s43, v131
	v_or_b32_e32 v132, 0x8000, v105
	buffer_load_dwordx4 v3, s[48:51], 0 offen lds
	s_mov_b32 m0, s43
	s_add_u32 s36, s36, 0x80
	v_readfirstlane_b32 s43, v132
	v_or_b32_e32 v133, 0x9000, v105
	buffer_load_dwordx4 v108, s[48:51], 0 offen lds
	s_addc_u32 s37, s37, 0
	s_mov_b32 m0, s43
	v_readfirstlane_b32 s43, v133
	v_or_b32_e32 v134, 0xa000, v105
	buffer_load_dwordx4 v106, s[36:39], 0 offen lds
	s_mov_b32 m0, s43
	v_readfirstlane_b32 s43, v134
	v_or_b32_e32 v135, 0xb000, v105
	buffer_load_dwordx4 v2, s[36:39], 0 offen lds
	s_mov_b32 m0, s43
	v_readfirstlane_b32 s43, v135
	buffer_load_dwordx4 v3, s[36:39], 0 offen lds
	s_mov_b32 m0, s43
	v_or_b32_e32 v136, 0x18000, v105
	buffer_load_dwordx4 v108, s[36:39], 0 offen lds
	s_add_u32 s40, s40, 0x80
	v_readfirstlane_b32 s36, v136
	v_or_b32_e32 v4, 0x19000, v105
	s_addc_u32 s41, s41, 0
	s_mov_b32 s43, s39
	s_mov_b32 m0, s36
	v_readfirstlane_b32 s36, v4
	v_or_b32_e32 v4, 0x1a000, v105
	buffer_load_dwordx4 v110, s[40:43], 0 offen lds
	s_mov_b32 m0, s36
	v_readfirstlane_b32 s36, v4
	v_or_b32_e32 v4, 0x1b000, v105
	buffer_load_dwordx4 v5, s[40:43], 0 offen lds
	s_mov_b32 m0, s36
	v_readfirstlane_b32 s36, v4
	buffer_load_dwordx4 v7, s[40:43], 0 offen lds
	s_mov_b32 m0, s36
	v_or_b32_e32 v137, 0x1c000, v105
	buffer_load_dwordx4 v112, s[40:43], 0 offen lds
	s_add_u32 s40, s44, 0x80
	v_readfirstlane_b32 s36, v137
	v_or_b32_e32 v4, 0x1d000, v105
	s_addc_u32 s41, s45, 0
	s_mov_b32 m0, s36
	v_readfirstlane_b32 s36, v4
	v_or_b32_e32 v4, 0x1e000, v105
	buffer_load_dwordx4 v110, s[40:43], 0 offen lds
	s_mov_b32 m0, s36
	v_readfirstlane_b32 s36, v4
	v_or_b32_e32 v4, 0x1f000, v105
	buffer_load_dwordx4 v5, s[40:43], 0 offen lds
	s_mov_b32 m0, s36
	v_readfirstlane_b32 s36, v4
	buffer_load_dwordx4 v7, s[40:43], 0 offen lds
	s_mov_b32 m0, s36
	v_or_b32_e32 v138, 0xc000, v105
	buffer_load_dwordx4 v112, s[40:43], 0 offen lds
	s_add_u32 s36, s48, 0x80
	v_readfirstlane_b32 s40, v138
	v_or_b32_e32 v139, 0xd000, v105
	s_addc_u32 s37, s49, 0
	s_mov_b32 m0, s40
	v_readfirstlane_b32 s40, v139
	v_or_b32_e32 v140, 0xe000, v105
	buffer_load_dwordx4 v106, s[36:39], 0 offen lds
	s_mov_b32 m0, s40
	v_readfirstlane_b32 s40, v140
	v_or_b32_e32 v141, 0xf000, v105
	buffer_load_dwordx4 v2, s[36:39], 0 offen lds
	s_mov_b32 m0, s40
	v_readfirstlane_b32 s40, v141
	buffer_load_dwordx4 v3, s[36:39], 0 offen lds
	s_mov_b32 m0, s40
	v_and_b32_e32 v2, 48, v0
	buffer_load_dwordx4 v108, s[36:39], 0 offen lds
	v_lshlrev_b32_e32 v3, 7, v0
	s_movk_i32 s36, 0x780
	v_lshlrev_b32_e32 v102, 13, v1
	v_and_or_b32 v103, v3, s36, v2
	v_lshlrev_b32_e32 v68, 3, v0
	v_or_b32_e32 v69, v102, v103
	;;#ASMSTART
	s_waitcnt vmcnt(28)
	;;#ASMEND
	s_barrier
	v_bitop3_b32 v142, v68, v69, s65 bitop3:0x6c
	;;#ASMSTART
	ds_read_b128 v[42:45], v142 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[58:61], v142 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[50:53], v142 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[34:37], v142 offset:0x1800

	;;#ASMEND
	v_or_b32_e32 v2, 64, v69
	v_bitop3_b32 v143, v68, v2, s65 bitop3:0x6c
	;;#ASMSTART
	ds_read_b128 v[46:49], v143 offset:0

	;;#ASMEND
	v_lshlrev_b32_e32 v104, 13, v6
	;;#ASMSTART
	ds_read_b128 v[62:65], v143 offset:0x800

	;;#ASMEND
	v_or_b32_e32 v70, v104, v103
	;;#ASMSTART
	ds_read_b128 v[54:57], v143 offset:0x1000

	;;#ASMEND
	v_or_b32_e32 v2, 0x10000, v70
	;;#ASMSTART
	ds_read_b128 v[38:41], v143 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(24)
	;;#ASMEND
	s_barrier
	v_bitop3_b32 v144, v68, v2, s65 bitop3:0x6c
	;;#ASMSTART
	ds_read_b128 v[2:5], v144 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[10:13], v144 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[18:21], v144 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[26:29], v144 offset:0x1800

	;;#ASMEND
	v_or_b32_e32 v6, 0x10040, v70
	v_bitop3_b32 v145, v68, v6, s65 bitop3:0x6c
	;;#ASMSTART
	ds_read_b128 v[6:9], v145 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[14:17], v145 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[22:25], v145 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[30:33], v145 offset:0x1800

	;;#ASMEND
	buffer_load_dword v170, v123, s[24:27], 0 offen
	buffer_load_dword v168, v123, s[0:3], 0 offen
	buffer_load_dword v169, v123, s[8:11], 0 offen
	buffer_load_dword v153, v123, s[4:7], 0 offen
	buffer_load_dword v166, v123, s[28:31], 0 offen
	buffer_load_dword v165, v123, s[12:15], 0 offen
	buffer_load_dword v167, v123, s[20:23], 0 offen
	buffer_load_dword v158, v123, s[16:19], 0 offen
	v_accvgpr_write_b32 a3, 0
	v_accvgpr_write_b32 a2, 0
	v_accvgpr_write_b32 a1, 0
	v_accvgpr_write_b32 a0, 0
	v_accvgpr_read_b32 v191, a3
	v_or_b32_e32 v71, 32, v67
	v_accvgpr_read_b32 v190, a2
	v_accvgpr_read_b32 v189, a1
	v_accvgpr_read_b32 v188, a0
	v_accvgpr_write_b32 a3, 0
	v_mad_u64_u32 v[114:115], s[36:37], v71, s60, v[66:67]
	v_or_b32_e32 v67, 64, v67
	v_accvgpr_write_b32 a2, 0
	v_accvgpr_write_b32 a1, 0
	v_accvgpr_write_b32 a0, 0
	v_accvgpr_read_b32 v199, a3
	v_mad_u64_u32 v[116:117], s[36:37], v67, s60, v[66:67]
	v_mad_u64_u32 v[118:119], s[36:37], v71, s58, v[66:67]
	v_mad_u64_u32 v[120:121], s[36:37], v67, s58, v[66:67]
	v_or_b32_e32 v66, 0x14000, v70
	v_or_b32_e32 v67, 0x14040, v70
	v_accvgpr_read_b32 v198, a2
	v_accvgpr_read_b32 v197, a1
	v_accvgpr_read_b32 v196, a0
	v_accvgpr_write_b32 a3, 0
	v_bitop3_b32 v115, v68, v67, s65 bitop3:0x6c
	v_bitop3_b32 v117, v68, v66, s65 bitop3:0x6c
	v_or_b32_e32 v66, 0x4000, v69
	v_or_b32_e32 v67, 0x4040, v69
	v_accvgpr_write_b32 a2, 0
	v_accvgpr_write_b32 a1, 0
	v_accvgpr_write_b32 a0, 0
	v_accvgpr_read_b32 v195, a3
	v_bitop3_b32 v119, v68, v67, s65 bitop3:0x6c
	v_bitop3_b32 v121, v68, v66, s65 bitop3:0x6c
	v_or_b32_e32 v66, 0x8000, v69
	v_or_b32_e32 v67, 0x8040, v69
	v_accvgpr_read_b32 v194, a2
	v_accvgpr_read_b32 v193, a1
	v_accvgpr_read_b32 v192, a0
	v_accvgpr_write_b32 a3, 0
	v_bitop3_b32 v148, v68, v67, s65 bitop3:0x6c
	v_bitop3_b32 v149, v68, v66, s65 bitop3:0x6c
	v_or_b32_e32 v66, 0x18000, v70
	v_or_b32_e32 v67, 0x18040, v70
	s_lshl_b32 s36, s64, 8
	s_lshl_b32 s37, s63, 8
	v_accvgpr_write_b32 a2, 0
	v_accvgpr_write_b32 a1, 0
	v_accvgpr_write_b32 a0, 0
	v_accvgpr_read_b32 v203, a3
	v_bitop3_b32 v151, v68, v67, s65 bitop3:0x6c
	v_bitop3_b32 v152, v68, v66, s65 bitop3:0x6c
	v_or_b32_e32 v66, 0x1c000, v70
	v_or_b32_e32 v67, 0x1c040, v70
	s_sub_i32 s36, s36, s37
	v_accvgpr_read_b32 v202, a2
	v_accvgpr_read_b32 v201, a1
	v_accvgpr_read_b32 v200, a0
	v_accvgpr_write_b32 a3, 0
	v_bitop3_b32 v154, v68, v67, s65 bitop3:0x6c
	v_bitop3_b32 v155, v68, v66, s65 bitop3:0x6c
	v_or_b32_e32 v66, 0xc000, v69
	v_or_b32_e32 v67, 0xc040, v69
	s_or_b32 s44, s36, 0x80
	v_accvgpr_write_b32 a2, 0
	v_accvgpr_write_b32 a1, 0
	v_accvgpr_write_b32 a0, 0
	v_accvgpr_read_b32 v187, a3
	v_or_b32_e32 v146, 0x2000, v127
	v_or_b32_e32 v147, 0x1000, v127
	v_or_b32_e32 v150, 0x3000, v127
	v_or_b32_e32 v156, 0x2000, v136
	v_or_b32_e32 v157, 0x1000, v136
	v_bitop3_b32 v159, v68, v67, s65 bitop3:0x6c
	v_bitop3_b32 v160, v68, v66, s65 bitop3:0x6c
	v_or_b32_e32 v161, 0x3000, v136
	v_or_b32_e32 v162, 0x2000, v137
	v_or_b32_e32 v163, 0x1000, v137
	v_or_b32_e32 v164, 0x3000, v137
	s_mul_i32 s44, s44, s58
	v_accvgpr_write_b32 a255, 0
	v_accvgpr_write_b32 a254, 0
	v_accvgpr_write_b32 a253, 0
	v_accvgpr_write_b32 a252, 0
	v_accvgpr_write_b32 a251, 0
	v_accvgpr_write_b32 a250, 0
	v_accvgpr_write_b32 a249, 0
	v_accvgpr_write_b32 a248, 0
	v_accvgpr_write_b32 a247, 0
	v_accvgpr_write_b32 a246, 0
	v_accvgpr_write_b32 a245, 0
	v_accvgpr_write_b32 a244, 0
	v_accvgpr_write_b32 a243, 0
	v_accvgpr_write_b32 a242, 0
	v_accvgpr_write_b32 a241, 0
	v_accvgpr_write_b32 a240, 0
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
	v_accvgpr_write_b32 a207, 0
	v_accvgpr_write_b32 a206, 0
	v_accvgpr_write_b32 a205, 0
	v_accvgpr_write_b32 a204, 0
	v_accvgpr_write_b32 a219, 0
	v_accvgpr_write_b32 a218, 0
	v_accvgpr_write_b32 a217, 0
	v_accvgpr_write_b32 a216, 0
	v_accvgpr_write_b32 a215, 0
	v_accvgpr_write_b32 a214, 0
	v_accvgpr_write_b32 a213, 0
	v_accvgpr_write_b32 a212, 0
	v_accvgpr_write_b32 a223, 0
	v_accvgpr_write_b32 a222, 0
	v_accvgpr_write_b32 a221, 0
	v_accvgpr_write_b32 a220, 0
	v_accvgpr_write_b32 a191, 0
	v_accvgpr_write_b32 a190, 0
	v_accvgpr_write_b32 a189, 0
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
	v_accvgpr_write_b32 a187, 0
	v_accvgpr_write_b32 a186, 0
	v_accvgpr_write_b32 a185, 0
	v_accvgpr_write_b32 a184, 0
	v_accvgpr_write_b32 a211, 0
	v_accvgpr_write_b32 a210, 0
	v_accvgpr_write_b32 a209, 0
	v_accvgpr_write_b32 a208, 0
	v_accvgpr_write_b32 a183, 0
	v_accvgpr_write_b32 a182, 0
	v_accvgpr_write_b32 a181, 0
	v_accvgpr_write_b32 a180, 0
	v_accvgpr_write_b32 a107, 0
	v_accvgpr_write_b32 a106, 0
	v_accvgpr_write_b32 a105, 0
	v_accvgpr_write_b32 a104, 0
	v_accvgpr_write_b32 a179, 0
	v_accvgpr_write_b32 a178, 0
	v_accvgpr_write_b32 a177, 0
	v_accvgpr_write_b32 a176, 0
	v_accvgpr_write_b32 a155, 0
	v_accvgpr_write_b32 a154, 0
	v_accvgpr_write_b32 a153, 0
	v_accvgpr_write_b32 a152, 0
	v_accvgpr_write_b32 a139, 0
	v_accvgpr_write_b32 a138, 0
	v_accvgpr_write_b32 a137, 0
	v_accvgpr_write_b32 a136, 0
	v_accvgpr_write_b32 a147, 0
	v_accvgpr_write_b32 a146, 0
	v_accvgpr_write_b32 a145, 0
	v_accvgpr_write_b32 a144, 0
	v_accvgpr_write_b32 a39, 0
	v_accvgpr_write_b32 a38, 0
	v_accvgpr_write_b32 a37, 0
	v_accvgpr_write_b32 a36, 0
	v_accvgpr_write_b32 a127, 0
	v_accvgpr_write_b32 a126, 0
	v_accvgpr_write_b32 a125, 0
	v_accvgpr_write_b32 a124, 0
	v_accvgpr_write_b32 a119, 0
	v_accvgpr_write_b32 a118, 0
	v_accvgpr_write_b32 a117, 0
	v_accvgpr_write_b32 a116, 0
	v_accvgpr_write_b32 a99, 0
	v_accvgpr_write_b32 a98, 0
	v_accvgpr_write_b32 a97, 0
	v_accvgpr_write_b32 a96, 0
	v_accvgpr_write_b32 a175, 0
	v_accvgpr_write_b32 a174, 0
	v_accvgpr_write_b32 a173, 0
	v_accvgpr_write_b32 a172, 0
	v_accvgpr_write_b32 a163, 0
	v_accvgpr_write_b32 a162, 0
	v_accvgpr_write_b32 a161, 0
	v_accvgpr_write_b32 a160, 0
	v_accvgpr_write_b32 a143, 0
	v_accvgpr_write_b32 a142, 0
	v_accvgpr_write_b32 a141, 0
	v_accvgpr_write_b32 a140, 0
	v_accvgpr_write_b32 a131, 0
	v_accvgpr_write_b32 a130, 0
	v_accvgpr_write_b32 a129, 0
	v_accvgpr_write_b32 a128, 0
	v_accvgpr_write_b32 a135, 0
	v_accvgpr_write_b32 a134, 0
	v_accvgpr_write_b32 a133, 0
	v_accvgpr_write_b32 a132, 0
	v_accvgpr_write_b32 a27, 0
	v_accvgpr_write_b32 a26, 0
	v_accvgpr_write_b32 a25, 0
	v_accvgpr_write_b32 a24, 0
	v_accvgpr_write_b32 a151, 0
	v_accvgpr_write_b32 a150, 0
	v_accvgpr_write_b32 a149, 0
	v_accvgpr_write_b32 a148, 0
	v_accvgpr_write_b32 a31, 0
	v_accvgpr_write_b32 a30, 0
	v_accvgpr_write_b32 a29, 0
	v_accvgpr_write_b32 a28, 0
	v_accvgpr_write_b32 a87, 0
	v_accvgpr_write_b32 a86, 0
	v_accvgpr_write_b32 a85, 0
	v_accvgpr_write_b32 a84, 0
	v_accvgpr_write_b32 a111, 0
	v_accvgpr_write_b32 a110, 0
	v_accvgpr_write_b32 a109, 0
	v_accvgpr_write_b32 a108, 0
	v_accvgpr_write_b32 a15, 0
	v_accvgpr_write_b32 a14, 0
	v_accvgpr_write_b32 a13, 0
	v_accvgpr_write_b32 a12, 0
	v_accvgpr_write_b32 a51, 0
	v_accvgpr_write_b32 a50, 0
	v_accvgpr_write_b32 a49, 0
	v_accvgpr_write_b32 a48, 0
	v_accvgpr_write_b32 a103, 0
	v_accvgpr_write_b32 a102, 0
	v_accvgpr_write_b32 a101, 0
	v_accvgpr_write_b32 a100, 0
	v_accvgpr_write_b32 a77, 0
	v_accvgpr_write_b32 a76, 0
	v_accvgpr_write_b32 a75, 0
	v_accvgpr_write_b32 a74, 0
	v_accvgpr_write_b32 a59, 0
	v_accvgpr_write_b32 a58, 0
	v_accvgpr_write_b32 a57, 0
	v_accvgpr_write_b32 a56, 0
	v_accvgpr_write_b32 a69, 0
	v_accvgpr_write_b32 a68, 0
	v_accvgpr_write_b32 a67, 0
	v_accvgpr_write_b32 a66, 0
	v_accvgpr_write_b32 a47, 0
	v_accvgpr_write_b32 a46, 0
	v_accvgpr_write_b32 a45, 0
	v_accvgpr_write_b32 a44, 0
	v_accvgpr_write_b32 a123, 0
	v_accvgpr_write_b32 a122, 0
	v_accvgpr_write_b32 a121, 0
	v_accvgpr_write_b32 a120, 0
	v_accvgpr_write_b32 a115, 0
	v_accvgpr_write_b32 a114, 0
	v_accvgpr_write_b32 a113, 0
	v_accvgpr_write_b32 a112, 0
	v_accvgpr_write_b32 a73, 0
	v_accvgpr_write_b32 a72, 0
	v_accvgpr_write_b32 a71, 0
	v_accvgpr_write_b32 a70, 0
	v_accvgpr_write_b32 a63, 0
	v_accvgpr_write_b32 a62, 0
	v_accvgpr_write_b32 a61, 0
	v_accvgpr_write_b32 a60, 0
	v_accvgpr_write_b32 a55, 0
	v_accvgpr_write_b32 a54, 0
	v_accvgpr_write_b32 a53, 0
	v_accvgpr_write_b32 a52, 0
	v_accvgpr_write_b32 a23, 0
	v_accvgpr_write_b32 a22, 0
	v_accvgpr_write_b32 a21, 0
	v_accvgpr_write_b32 a20, 0
	v_accvgpr_write_b32 a43, 0
	v_accvgpr_write_b32 a42, 0
	v_accvgpr_write_b32 a41, 0
	v_accvgpr_write_b32 a40, 0
	v_accvgpr_read_b32 v186, a2
	v_accvgpr_read_b32 v185, a1
	v_accvgpr_read_b32 v184, a0
	v_accvgpr_write_b32 a91, 0
	v_accvgpr_write_b32 a90, 0
	v_accvgpr_write_b32 a89, 0
	v_accvgpr_write_b32 a88, 0
	v_accvgpr_write_b32 a35, 0
	v_accvgpr_write_b32 a34, 0
	v_accvgpr_write_b32 a33, 0
	v_accvgpr_write_b32 a32, 0
	v_accvgpr_write_b32 a81, 0
	v_accvgpr_write_b32 a80, 0
	v_accvgpr_write_b32 a79, 0
	v_accvgpr_write_b32 a78, 0
	v_accvgpr_write_b32 a167, 0
	v_accvgpr_write_b32 a166, 0
	v_accvgpr_write_b32 a165, 0
	v_accvgpr_write_b32 a164, 0
	v_accvgpr_write_b32 a3, 0
	v_accvgpr_write_b32 a2, 0
	v_accvgpr_write_b32 a1, 0
	v_accvgpr_write_b32 a0, 0
	v_accvgpr_write_b32 a95, 0
	v_accvgpr_write_b32 a94, 0
	v_accvgpr_write_b32 a93, 0
	v_accvgpr_write_b32 a92, 0
	v_accvgpr_write_b32 a19, 0
	v_accvgpr_write_b32 a18, 0
	v_accvgpr_write_b32 a17, 0
	v_accvgpr_write_b32 a16, 0
.LBB0_3:                                ; =>This Inner Loop Header: Depth=1
	s_add_i32 s45, s61, s59
	s_waitcnt vmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[58:65], v[26:33], a[236:239], v170, v167 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_add_i32 s36, s45, 0x100
	s_ashr_i32 s37, s36, 31
	v_readfirstlane_b32 s40, v105
	s_add_u32 s36, s54, s36
	s_addc_u32 s37, s55, s37
	s_mov_b32 m0, s40
	;;#ASMSTART
	s_waitcnt vmcnt(16)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_barrier
	buffer_load_dwordx4 v106, s[36:39], 0 offen lds
	;;#ASMSTART
	ds_read_b128 v[66:69], v117 offset:0

	;;#ASMEND
	v_readfirstlane_b32 s40, v107
	;;#ASMSTART
	ds_read_b128 v[70:73], v115 offset:0

	;;#ASMEND
	s_mov_b32 m0, s40
	v_readfirstlane_b32 s40, v124
	buffer_load_dwordx4 v114, s[36:39], 0 offen lds
	v_accvgpr_read_b32 v207, a7
	;;#ASMSTART
	ds_read_b128 v[74:77], v117 offset:0x800

	;;#ASMEND
	v_accvgpr_read_b32 v206, a6
	v_accvgpr_read_b32 v205, a5
	v_accvgpr_read_b32 v204, a4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[50:57], v[26:33], a[220:223], v169, v167 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	;;#ASMSTART
	ds_read_b128 v[78:81], v115 offset:0x800

	;;#ASMEND
	s_mov_b32 m0, s40
	v_accvgpr_read_b32 v94, a136
	v_accvgpr_read_b32 v183, a3
	v_accvgpr_read_b32 v97, a139
	v_accvgpr_read_b32 v96, a138
	v_accvgpr_read_b32 v95, a137
	buffer_load_dwordx4 v116, s[36:39], 0 offen lds
	v_accvgpr_read_b32 v182, a2
	v_accvgpr_read_b32 v181, a1
	v_accvgpr_read_b32 v180, a0
	v_accvgpr_write_b32 a0, v94
	;;#ASMSTART
	ds_read_b128 v[82:85], v117 offset:0x1000

	;;#ASMEND
	v_accvgpr_write_b32 a1, v95
	v_accvgpr_write_b32 a2, v96
	v_accvgpr_write_b32 a3, v97
	;;#ASMSTART
	ds_read_b128 v[86:89], v115 offset:0x1000

	;;#ASMEND
	v_accvgpr_read_b32 v211, a7
	v_accvgpr_read_b32 v210, a6
	s_waitcnt vmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[58:65], v[82:89], a[0:3], v170, v158 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_accvgpr_read_b32 v209, a5
	v_accvgpr_read_b32 v208, a4
	v_accvgpr_read_b32 v172, a116
	v_accvgpr_read_b32 v175, a119
	v_accvgpr_read_b32 v174, a118
	v_accvgpr_read_b32 v173, a117
	v_accvgpr_mov_b32 a159, a127
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[58:65], v[74:81], a[152:155], v170, v165 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_accvgpr_mov_b32 a158, a126
	v_accvgpr_mov_b32 a157, a125
	v_accvgpr_mov_b32 a156, a124
	v_readfirstlane_b32 s40, v109
	v_accvgpr_read_b32 v223, a3
	v_accvgpr_read_b32 v222, a2
	v_accvgpr_read_b32 v221, a1
	v_accvgpr_read_b32 v220, a0
	v_accvgpr_write_b32 a0, v172
	v_accvgpr_write_b32 a1, v173
	v_accvgpr_write_b32 a2, v174
	v_accvgpr_write_b32 a3, v175
	v_accvgpr_read_b32 v215, a7
	v_accvgpr_read_b32 v214, a6
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[50:57], v[82:89], a[0:3], v169, v158 op_sel_hi:[0,0,0]
	v_accvgpr_read_b32 v213, a5
	v_accvgpr_read_b32 v212, a4
	v_accvgpr_read_b32 v90, a104
	s_mov_b32 m0, s40
	v_accvgpr_read_b32 v93, a107
	v_accvgpr_read_b32 v92, a106
	v_accvgpr_read_b32 v91, a105
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[50:57], v[74:81], a[156:159], v169, v165 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	buffer_load_dwordx4 v108, s[36:39], 0 offen lds
	v_accvgpr_read_b32 v101, a147
	v_accvgpr_read_b32 v100, a146
	v_accvgpr_read_b32 v99, a145
	v_accvgpr_read_b32 v175, a3
	v_accvgpr_read_b32 v174, a2
	v_accvgpr_read_b32 v173, a1
	v_accvgpr_read_b32 v172, a0
	v_accvgpr_write_b32 a0, v90
	v_accvgpr_read_b32 v98, a144
	v_accvgpr_mov_b32 a147, a143
	v_accvgpr_mov_b32 a146, a142
	v_accvgpr_mov_b32 a145, a141
	v_accvgpr_mov_b32 a144, a140
	v_accvgpr_write_b32 a1, v91
	v_accvgpr_write_b32 a2, v92
	v_accvgpr_write_b32 a3, v93
	;;#ASMSTART
	ds_read_b128 v[90:93], v117 offset:0x1800

	;;#ASMEND
	s_add_i32 s47, s62, s59
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[34:41], v[82:89], a[144:147], v169, v158 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	;;#ASMSTART
	ds_read_b128 v[94:97], v115 offset:0x1800

	;;#ASMEND
	s_add_i32 s37, s47, 0x100
	v_accvgpr_read_b32 v219, a7
	v_accvgpr_read_b32 v176, a96
	s_ashr_i32 s41, s37, 31
	v_accvgpr_read_b32 v218, a6
	v_accvgpr_read_b32 v217, a5
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[42:49], v[90:97], a[0:3], v170, v158 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_accvgpr_read_b32 v216, a4
	v_accvgpr_read_b32 v179, a99
	v_accvgpr_read_b32 v178, a98
	v_accvgpr_read_b32 v177, a97
	v_readfirstlane_b32 s36, v125
	s_add_u32 s40, s56, s37
	s_mov_b32 s43, s39
	v_accvgpr_write_b32 a0, v98
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[42:49], v[2:9], a[252:255], v170, v166 op_sel_hi:[0,0,0]
	v_accvgpr_write_b32 a1, v99
	v_accvgpr_write_b32 a2, v100
	v_accvgpr_write_b32 a3, v101
	s_addc_u32 s41, s57, s41
	s_mov_b32 m0, s36
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	v_readfirstlane_b32 s36, v111
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[42:49], v[10:17], a[248:251], v170, v166 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	buffer_load_dwordx4 v110, s[40:43], 0 offen lds
	v_accvgpr_mov_b32 a171, a39
	v_accvgpr_mov_b32 a170, a38
	v_accvgpr_mov_b32 a169, a37
	v_accvgpr_mov_b32 a168, a36
	s_mov_b32 m0, s36
	v_readfirstlane_b32 s36, v126
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[42:49], v[18:25], a[244:247], v170, v167 op_sel_hi:[0,0,0]
	v_accvgpr_mov_b32 a99, a87
	v_accvgpr_mov_b32 a98, a86
	v_accvgpr_mov_b32 a97, a85
	v_accvgpr_mov_b32 a96, a84
	v_accvgpr_write_b32 a84, v192
	v_accvgpr_write_b32 a85, v193
	v_accvgpr_write_b32 a86, v194
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[42:49], v[26:33], a[240:243], v170, v167 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_accvgpr_write_b32 a87, v195
	s_add_i32 s48, s44, s59
	v_readfirstlane_b32 s37, v147
	v_readfirstlane_b32 s50, v146
	v_readfirstlane_b32 s51, v150
	v_readfirstlane_b32 s58, v128
	v_readfirstlane_b32 s63, v141
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[42:49], v[66:73], a[184:187], v170, v165 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[42:49], v[74:81], a[208:211], v170, v165 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[42:49], v[82:89], a[180:183], v170, v158 op_sel_hi:[0,0,0]
	v_accvgpr_read_b32 v42, a128
	v_accvgpr_read_b32 v45, a131
	v_accvgpr_read_b32 v44, a130
	v_accvgpr_read_b32 v43, a129
	v_accvgpr_write_b32 a4, v42
	v_accvgpr_write_b32 a5, v43
	v_accvgpr_write_b32 a6, v44
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[58:65], v[90:97], a[0:3], v170, v158 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_accvgpr_write_b32 a7, v45
	v_accvgpr_read_b32 v42, a24
	v_accvgpr_read_b32 v45, a27
	v_accvgpr_read_b32 v44, a26
	v_accvgpr_read_b32 v43, a25
	v_mov_b64_e32 v[46:47], v[200:201]
	v_mov_b64_e32 v[48:49], v[202:203]
	v_accvgpr_write_b32 a0, v176
	v_accvgpr_write_b32 a1, v177
	v_accvgpr_write_b32 a2, v178
	v_accvgpr_write_b32 a3, v179
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[50:57], v[18:25], a[212:215], v169, v167 op_sel_hi:[0,0,0]
	s_nop 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[50:57], v[90:97], a[0:3], v169, v158 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	s_nop 6
	v_accvgpr_write_b32 a0, v42
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[34:41], v[2:9], a[188:191], v169, v166 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_accvgpr_write_b32 a1, v43
	v_accvgpr_write_b32 a2, v44
	v_accvgpr_write_b32 a3, v45
	v_accvgpr_read_b32 v42, a148
	v_accvgpr_read_b32 v45, a151
	v_accvgpr_read_b32 v44, a150
	v_accvgpr_read_b32 v43, a149
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[34:41], v[10:17], a[192:195], v169, v166 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[34:41], v[18:25], a[196:199], v169, v167 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[34:41], v[26:33], a[200:203], v169, v167 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[34:41], v[66:73], a[172:175], v169, v165 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[34:41], v[74:81], a[160:163], v169, v165 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[34:41], v[90:97], a[4:7], v169, v158 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	;;#ASMSTART
	ds_read_b128 v[34:37], v121 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[38:41], v119 offset:0

	;;#ASMEND
	buffer_load_dwordx4 v118, s[40:43], 0 offen lds
	s_mov_b32 m0, s36
	v_readfirstlane_b32 s36, v113
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[34:41], v[10:17], a[0:3], v168, v166 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	s_nop 6
	v_accvgpr_write_b32 a0, v188
	v_accvgpr_write_b32 a1, v189
	v_accvgpr_write_b32 a2, v190
	v_accvgpr_write_b32 a3, v191
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[50:57], v[2:9], a[204:207], v169, v166 op_sel_hi:[0,0,0]
	s_nop 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[34:41], v[18:25], a[0:3], v168, v167 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[50:57], v[10:17], a[216:219], v169, v166 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	s_nop 7
	s_nop 2
	v_accvgpr_read_b32 v179, a3
	v_accvgpr_read_b32 v178, a2
	v_accvgpr_read_b32 v177, a1
	v_accvgpr_read_b32 v176, a0
	v_accvgpr_write_b32 a0, v42
	v_accvgpr_write_b32 a1, v43
	v_accvgpr_write_b32 a2, v44
	v_accvgpr_write_b32 a3, v45
	v_accvgpr_read_b32 v42, a112
	v_accvgpr_read_b32 v45, a115
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[34:41], v[26:33], a[0:3], v168, v167 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_accvgpr_read_b32 v44, a114
	v_accvgpr_read_b32 v43, a113
	s_nop 4
	v_accvgpr_write_b32 a0, v42
	v_accvgpr_write_b32 a1, v43
	v_accvgpr_write_b32 a2, v44
	v_accvgpr_write_b32 a3, v45
	v_accvgpr_read_b32 v42, a70
	v_accvgpr_read_b32 v45, a73
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[34:41], v[74:81], a[0:3], v168, v165 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_accvgpr_read_b32 v44, a72
	v_accvgpr_read_b32 v43, a71
	s_nop 4
	v_accvgpr_write_b32 a0, v42
	v_accvgpr_write_b32 a1, v43
	v_accvgpr_write_b32 a2, v44
	v_accvgpr_write_b32 a3, v45
	v_mov_b32_e32 v42, v46
	v_mov_b32_e32 v45, v49
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[34:41], v[82:89], a[0:3], v168, v158 op_sel_hi:[0,0,0]
	v_mov_b32_e32 v44, v48
	v_mov_b32_e32 v43, v47
	v_accvgpr_read_b32 v46, a60
	v_accvgpr_read_b32 v49, a63
	v_accvgpr_read_b32 v48, a62
	v_accvgpr_read_b32 v47, a61
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[50:57], v[66:73], a[168:171], v169, v165 op_sel_hi:[0,0,0]
	v_accvgpr_read_b32 v50, a108
	v_accvgpr_read_b32 v51, a109
	v_accvgpr_read_b32 v52, a110
	v_accvgpr_read_b32 v53, a111
	s_nop 0
	v_accvgpr_read_b32 v203, a3
	v_accvgpr_read_b32 v202, a2
	v_accvgpr_read_b32 v201, a1
	v_accvgpr_read_b32 v200, a0
	v_accvgpr_write_b32 a0, v42
	v_accvgpr_write_b32 a1, v43
	v_accvgpr_write_b32 a2, v44
	v_accvgpr_write_b32 a3, v45
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[34:41], v[2:9], a[132:135], v168, v166 op_sel_hi:[0,0,0]
	v_accvgpr_mov_b32 a111, a31
	v_accvgpr_mov_b32 a110, a30
	v_accvgpr_mov_b32 a109, a29
	v_accvgpr_mov_b32 a108, a28
	v_mov_b64_e32 v[42:43], v[196:197]
	v_mov_b64_e32 v[44:45], v[198:199]
	v_accvgpr_read_b32 v55, a77
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[34:41], v[66:73], a[120:123], v168, v165 op_sel_hi:[0,0,0]
	v_accvgpr_read_b32 v54, a76
	v_accvgpr_read_b32 v56, a88
	v_accvgpr_read_b32 v57, a89
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[34:41], v[90:97], a[0:3], v168, v158 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	;;#ASMSTART
	ds_read_b128 v[34:37], v121 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[38:41], v119 offset:0x800

	;;#ASMEND
	buffer_load_dwordx4 v120, s[40:43], 0 offen lds
	s_mov_b32 m0, s36
	v_readfirstlane_b32 s36, v127
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[34:41], v[2:9], a[108:111], v168, v166 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[58:65], v[2:9], a[224:227], v170, v166 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	s_nop 7
	s_nop 2
	v_accvgpr_read_b32 v227, a3
	v_accvgpr_read_b32 v226, a2
	v_accvgpr_read_b32 v225, a1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[58:65], v[10:17], a[228:231], v170, v166 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_accvgpr_read_b32 v224, a0
	v_accvgpr_write_b32 a0, v46
	v_accvgpr_write_b32 a1, v47
	v_accvgpr_write_b32 a2, v48
	v_accvgpr_write_b32 a3, v49
	v_accvgpr_read_b32 v46, a12
	v_accvgpr_read_b32 v49, a15
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[58:65], v[18:25], a[232:235], v170, v167 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_accvgpr_read_b32 v48, a14
	v_accvgpr_read_b32 v47, a13
	v_accvgpr_mov_b32 a15, a69
	v_accvgpr_mov_b32 a14, a68
	v_accvgpr_mov_b32 a13, a67
	v_accvgpr_mov_b32 a12, a66
	v_accvgpr_write_b32 a66, v176
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[58:65], v[66:73], a[176:179], v170, v165 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_accvgpr_read_b32 v60, a92
	v_accvgpr_read_b32 v61, a93
	v_accvgpr_read_b32 v62, a94
	v_accvgpr_read_b32 v63, a95
	v_accvgpr_write_b32 a92, v42
	v_mov_b32_e32 v42, v50
	v_accvgpr_read_b32 v50, a52
	v_accvgpr_write_b32 a95, v45
	v_accvgpr_write_b32 a94, v44
	v_accvgpr_write_b32 a93, v43
	v_mov_b32_e32 v45, v53
	v_mov_b32_e32 v44, v52
	v_mov_b32_e32 v43, v51
	v_mfma_scale_f32_16x16x128_f8f6f4 a[70:73], v[34:41], v[66:73], a[0:3], v168, v165 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_accvgpr_read_b32 v53, a55
	v_accvgpr_read_b32 v52, a54
	v_accvgpr_read_b32 v51, a53
	v_accvgpr_read_b32 v59, a91
	v_accvgpr_read_b32 v58, a90
	v_accvgpr_write_b32 a67, v177
	v_accvgpr_write_b32 a68, v178
	v_accvgpr_write_b32 a0, v50
	v_accvgpr_write_b32 a1, v51
	v_accvgpr_write_b32 a2, v52
	v_accvgpr_write_b32 a3, v53
	v_accvgpr_read_b32 v50, a20
	v_accvgpr_read_b32 v53, a23
	v_mfma_scale_f32_16x16x128_f8f6f4 a[62:65], v[34:41], v[74:81], a[0:3], v168, v165 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_accvgpr_read_b32 v52, a22
	v_accvgpr_read_b32 v51, a21
	v_accvgpr_write_b32 a69, v179
	s_nop 3
	v_accvgpr_write_b32 a0, v50
	v_accvgpr_write_b32 a1, v51
	v_accvgpr_write_b32 a2, v52
	v_accvgpr_write_b32 a3, v53
	v_accvgpr_read_b32 v50, a40
	v_accvgpr_read_b32 v53, a43
	v_accvgpr_read_b32 v52, a42
	v_accvgpr_read_b32 v51, a41
	v_accvgpr_write_b32 a4, v50
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[34:41], v[82:89], a[0:3], v168, v158 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_accvgpr_write_b32 a5, v51
	v_accvgpr_write_b32 a6, v52
	v_accvgpr_write_b32 a7, v53
	v_accvgpr_read_b32 v51, a103
	v_accvgpr_read_b32 v50, a102
	v_accvgpr_read_b32 v53, a75
	v_accvgpr_read_b32 v52, a74
	v_accvgpr_write_b32 a0, v42
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[34:41], v[10:17], a[96:99], v168, v166 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_accvgpr_write_b32 a1, v43
	v_accvgpr_write_b32 a2, v44
	v_accvgpr_write_b32 a3, v45
	v_accvgpr_read_b32 v42, a48
	v_accvgpr_read_b32 v45, a51
	v_accvgpr_read_b32 v44, a50
	v_accvgpr_read_b32 v43, a49
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[34:41], v[18:25], a[92:95], v168, v167 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[34:41], v[26:33], a[84:87], v168, v167 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[34:41], v[90:97], a[4:7], v168, v158 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	;;#ASMSTART
	ds_read_b128 v[34:37], v121 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[38:41], v119 offset:0x1000

	;;#ASMEND
	buffer_load_dwordx4 v112, s[40:43], 0 offen lds
	s_add_i32 s40, s48, 0x100
	s_ashr_i32 s41, s40, 31
	s_add_u32 s40, s56, s40
	s_addc_u32 s41, s57, s41
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[34:41], v[2:9], a[0:3], v153, v166 op_sel_hi:[0,0,0]
	s_mov_b32 m0, s36
	s_add_i32 s49, s46, s59
	s_add_i32 s36, s49, 0x100
	s_ashr_i32 s60, s36, 31
	s_add_u32 s36, s54, s36
	s_nop 1
	v_accvgpr_write_b32 a0, v46
	v_accvgpr_write_b32 a1, v47
	v_accvgpr_write_b32 a2, v48
	v_accvgpr_write_b32 a3, v49
	v_accvgpr_read_b32 v48, a100
	v_accvgpr_read_b32 v49, a101
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[34:41], v[10:17], a[0:3], v153, v166 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_accvgpr_read_b32 v46, a44
	v_accvgpr_read_b32 v47, a45
	s_nop 4
	v_accvgpr_write_b32 a0, v42
	v_accvgpr_write_b32 a1, v43
	v_accvgpr_write_b32 a2, v44
	v_accvgpr_write_b32 a3, v45
	v_accvgpr_read_b32 v42, a56
	v_accvgpr_read_b32 v45, a59
	v_mfma_scale_f32_16x16x128_f8f6f4 a[82:85], v[34:41], v[18:25], a[0:3], v153, v167 op_sel_hi:[0,0,0]
	v_accvgpr_read_b32 v44, a58
	v_accvgpr_read_b32 v43, a57
	s_nop 4
	v_accvgpr_write_b32 a0, v48
	v_accvgpr_write_b32 a1, v49
	v_accvgpr_write_b32 a2, v50
	v_accvgpr_write_b32 a3, v51
	v_accvgpr_read_b32 v49, a47
	v_accvgpr_read_b32 v48, a46
	v_mfma_scale_f32_16x16x128_f8f6f4 a[74:77], v[34:41], v[26:33], a[0:3], v153, v167 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	s_nop 6
	v_accvgpr_write_b32 a0, v184
	v_accvgpr_write_b32 a1, v185
	v_accvgpr_write_b32 a2, v186
	v_accvgpr_write_b32 a3, v187
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[34:41], v[66:73], a[0:3], v153, v165 op_sel_hi:[0,0,0]
	s_nop 6
	v_accvgpr_write_b32 a0, v56
	v_accvgpr_write_b32 a1, v57
	v_accvgpr_write_b32 a2, v58
	v_accvgpr_write_b32 a3, v59
	v_accvgpr_read_b32 v56, a32
	v_accvgpr_read_b32 v59, a35
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[34:41], v[74:81], a[0:3], v153, v165 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_accvgpr_read_b32 v58, a34
	v_accvgpr_read_b32 v57, a33
	s_nop 4
	v_accvgpr_write_b32 a0, v56
	v_accvgpr_write_b32 a1, v57
	v_accvgpr_write_b32 a2, v58
	v_accvgpr_write_b32 a3, v59
	v_accvgpr_read_b32 v56, a78
	v_accvgpr_read_b32 v59, a81
	v_accvgpr_read_b32 v58, a80
	v_accvgpr_read_b32 v57, a79
	v_accvgpr_write_b32 a4, v56
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[34:41], v[82:89], a[0:3], v153, v158 op_sel_hi:[0,0,0]
	v_accvgpr_write_b32 a5, v57
	v_accvgpr_write_b32 a6, v58
	v_accvgpr_write_b32 a7, v59
	s_nop 3
	v_accvgpr_write_b32 a0, v52
	v_accvgpr_write_b32 a1, v53
	v_accvgpr_write_b32 a2, v54
	v_accvgpr_write_b32 a3, v55
	v_mfma_scale_f32_16x16x128_f8f6f4 a[78:81], v[34:41], v[90:97], a[4:7], v153, v158 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	;;#ASMSTART
	ds_read_b128 v[34:37], v121 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[38:41], v119 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(16)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	buffer_load_dwordx4 v110, s[40:43], 0 offen lds
	s_mov_b32 m0, s37
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[34:41], v[2:9], a[0:3], v153, v166 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_accvgpr_read_b32 v2, a164
	v_accvgpr_read_b32 v5, a167
	v_accvgpr_read_b32 v4, a166
	v_accvgpr_read_b32 v3, a165
	v_mov_b32_e32 v6, v180
	v_mov_b32_e32 v9, v183
	v_mov_b32_e32 v8, v182
	v_accvgpr_write_b32 a0, v42
	v_accvgpr_write_b32 a1, v43
	v_accvgpr_write_b32 a2, v44
	v_accvgpr_write_b32 a3, v45
	v_mov_b32_e32 v7, v181
	v_accvgpr_write_b32 a4, v6
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[34:41], v[10:17], a[0:3], v153, v166 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mov_b32_e32 v13, v63
	v_accvgpr_write_b32 a5, v7
	v_accvgpr_write_b32 a6, v8
	v_accvgpr_write_b32 a7, v9
	v_mov_b32_e32 v12, v62
	v_mov_b32_e32 v11, v61
	v_mov_b32_e32 v10, v60
	v_accvgpr_write_b32 a0, v46
	v_accvgpr_write_b32 a1, v47
	v_accvgpr_write_b32 a2, v48
	v_accvgpr_write_b32 a3, v49
	v_accvgpr_read_b32 v17, a19
	v_accvgpr_read_b32 v16, a18
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[34:41], v[26:33], a[0:3], v153, v167 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_accvgpr_read_b32 v15, a17
	v_accvgpr_read_b32 v14, a16
	v_accvgpr_write_b32 a19, v13
	v_accvgpr_write_b32 a18, v12
	v_accvgpr_write_b32 a17, v11
	v_accvgpr_write_b32 a16, v10
	v_accvgpr_write_b32 a103, v17
	v_accvgpr_write_b32 a0, v2
	v_accvgpr_write_b32 a1, v3
	v_accvgpr_write_b32 a2, v4
	v_accvgpr_write_b32 a3, v5
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[34:41], v[74:81], a[4:7], v153, v165 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_accvgpr_write_b32 a102, v16
	v_accvgpr_write_b32 a101, v15
	v_accvgpr_write_b32 a100, v14
	s_addc_u32 s37, s55, s60
	s_addk_i32 s45, 0x180
	v_readfirstlane_b32 s60, v140
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[34:41], v[66:73], a[0:3], v153, v165 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	;;#ASMSTART
	ds_read_b128 v[70:73], v149 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[74:77], v148 offset:0

	;;#ASMEND
	buffer_load_dwordx4 v118, s[40:43], 0 offen lds
	s_mov_b32 m0, s50
	v_accvgpr_write_b32 a4, v220
	v_accvgpr_write_b32 a5, v221
	v_accvgpr_write_b32 a6, v222
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[34:41], v[82:89], a[16:19], v153, v158 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	;;#ASMSTART
	ds_read_b128 v[86:89], v149 offset:0x800

	;;#ASMEND
	v_accvgpr_write_b32 a7, v223
	v_accvgpr_write_b32 a0, v204
	v_accvgpr_write_b32 a1, v205
	v_accvgpr_write_b32 a2, v206
	v_accvgpr_write_b32 a3, v207
	v_readfirstlane_b32 s50, v164
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[34:41], v[90:97], a[100:103], v153, v158 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	;;#ASMSTART
	ds_read_b128 v[90:93], v148 offset:0x800

	;;#ASMEND
	buffer_load_dwordx4 v120, s[40:43], 0 offen lds
	;;#ASMSTART
	ds_read_b128 v[94:97], v149 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[98:101], v148 offset:0x1000

	;;#ASMEND
	s_mov_b32 m0, s51
	v_readfirstlane_b32 s51, v138
	buffer_load_dwordx4 v112, s[40:43], 0 offen lds
	;;#ASMSTART
	ds_read_b128 v[78:81], v149 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[82:85], v148 offset:0x1800

	;;#ASMEND
	s_mov_b32 m0, s58
	v_readfirstlane_b32 s40, v129
	buffer_load_dwordx4 v106, s[36:39], 0 offen lds
	;;#ASMSTART
	ds_read_b128 v[2:5], v152 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[6:9], v151 offset:0

	;;#ASMEND
	s_mov_b32 m0, s40
	v_readfirstlane_b32 s40, v130
	buffer_load_dwordx4 v114, s[36:39], 0 offen lds
	;;#ASMSTART
	ds_read_b128 v[10:13], v152 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[14:17], v151 offset:0x800

	;;#ASMEND
	s_mov_b32 m0, s40
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[34:41], v[18:25], a[12:15], v153, v167 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	buffer_load_dwordx4 v116, s[36:39], 0 offen lds
	;;#ASMSTART
	ds_read_b128 v[18:21], v152 offset:0x1000

	;;#ASMEND
	v_readfirstlane_b32 s40, v131
	;;#ASMSTART
	ds_read_b128 v[22:25], v151 offset:0x1000

	;;#ASMEND
	s_mov_b32 m0, s40
	v_readfirstlane_b32 s40, v132
	buffer_load_dwordx4 v108, s[36:39], 0 offen lds
	s_ashr_i32 s37, s45, 31
	s_add_u32 s36, s54, s45
	;;#ASMSTART
	ds_read_b128 v[34:37], v152 offset:0x1800

	;;#ASMEND
	s_addc_u32 s37, s55, s37
	s_mov_b32 m0, s40
	;;#ASMSTART
	ds_read_b128 v[38:41], v151 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(16)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_barrier
	buffer_load_dwordx4 v106, s[36:39], 0 offen lds
	;;#ASMSTART
	ds_read_b128 v[26:29], v155 offset:0

	;;#ASMEND
	v_readfirstlane_b32 s40, v133
	;;#ASMSTART
	ds_read_b128 v[30:33], v154 offset:0

	;;#ASMEND
	s_mov_b32 m0, s40
	v_readfirstlane_b32 s40, v134
	buffer_load_dwordx4 v114, s[36:39], 0 offen lds
	;;#ASMSTART
	ds_read_b128 v[54:57], v155 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[58:61], v154 offset:0x800

	;;#ASMEND
	s_mov_b32 m0, s40
	v_readfirstlane_b32 s40, v135
	buffer_load_dwordx4 v116, s[36:39], 0 offen lds
	;;#ASMSTART
	ds_read_b128 v[46:49], v155 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[50:53], v154 offset:0x1000

	;;#ASMEND
	s_mov_b32 m0, s40
	s_addk_i32 s47, 0x180
	buffer_load_dwordx4 v108, s[36:39], 0 offen lds
	s_ashr_i32 s37, s47, 31
	v_readfirstlane_b32 s36, v136
	s_add_u32 s40, s56, s47
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[94:101], v[26:33], a[8:11], v169, v165 op_sel_hi:[1,1,0]
	;;#ASMSTART
	ds_read_b128 v[62:65], v155 offset:0x1800

	;;#ASMEND
	s_addc_u32 s41, s57, s37
	s_mov_b32 m0, s36
	;;#ASMSTART
	ds_read_b128 v[66:69], v154 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	v_readfirstlane_b32 s36, v157
	buffer_load_dwordx4 v110, s[40:43], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[78:85], v[54:61], a[156:159], v169, v165 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_mov_b32 m0, s36
	v_readfirstlane_b32 s36, v156
	s_addk_i32 s48, 0x180
	s_ashr_i32 s37, s48, 31
	v_readfirstlane_b32 s45, v163
	v_readfirstlane_b32 s47, v162
	v_readfirstlane_b32 s58, v139
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[86:93], v[46:53], a[4:7], v170, v158 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	s_nop 6
	v_accvgpr_write_b32 a4, v172
	v_accvgpr_write_b32 a5, v173
	v_accvgpr_write_b32 a6, v174
	v_accvgpr_write_b32 a7, v175
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[70:77], v[2:9], a[252:255], v170, v166 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[70:77], v[10:17], a[248:251], v170, v166 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[70:77], v[18:25], a[244:247], v170, v167 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[70:77], v[34:41], a[240:243], v170, v167 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[70:77], v[26:33], a[184:187], v170, v165 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[70:77], v[54:61], a[208:211], v170, v165 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[70:77], v[46:53], a[180:183], v170, v158 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[94:101], v[46:53], a[4:7], v169, v158 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[70:77], v[62:69], a[144:147], v170, v158 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	;;#ASMSTART
	ds_read_b128 v[70:73], v160 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[74:77], v159 offset:0

	;;#ASMEND
	buffer_load_dwordx4 v118, s[40:43], 0 offen lds
	s_mov_b32 m0, s36
	v_readfirstlane_b32 s36, v161
	v_mfma_scale_f32_16x16x128_f8f6f4 a[66:69], v[70:77], v[18:25], a[66:69], v168, v167 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[86:93], v[62:69], a[128:131], v170, v158 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_nop 7
	s_nop 2
	v_accvgpr_read_b32 v191, a69
	v_accvgpr_read_b32 v190, a68
	v_accvgpr_read_b32 v189, a67
	v_accvgpr_read_b32 v188, a66
	v_accvgpr_write_b32 a66, v200
	v_accvgpr_write_b32 a67, v201
	v_accvgpr_write_b32 a68, v202
	v_accvgpr_write_b32 a69, v203
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[78:85], v[62:69], a[116:119], v169, v158 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_nop 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[70:77], v[46:53], a[66:69], v168, v158 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[66:69], v[70:77], v[62:69], a[120:123], v168, v158 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[70:77], v[2:9], a[132:135], v168, v166 op_sel_hi:[1,1,0]
	s_nop 7
	s_nop 2
	v_accvgpr_read_b32 v203, a69
	v_accvgpr_read_b32 v202, a68
	v_accvgpr_read_b32 v201, a67
	v_accvgpr_read_b32 v200, a66
	v_accvgpr_write_b32 a66, v224
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[70:77], v[10:17], a[136:139], v168, v166 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_accvgpr_write_b32 a67, v225
	v_accvgpr_write_b32 a68, v226
	v_accvgpr_write_b32 a69, v227
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[70:77], v[34:41], a[148:151], v168, v167 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[70:77], v[26:33], a[124:127], v168, v165 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[70:77], v[54:61], a[112:115], v168, v165 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	;;#ASMSTART
	ds_read_b128 v[70:73], v160 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[74:77], v159 offset:0x800

	;;#ASMEND
	buffer_load_dwordx4 v120, s[40:43], 0 offen lds
	s_mov_b32 m0, s36
	v_readfirstlane_b32 s36, v137
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[70:77], v[2:9], a[66:69], v168, v166 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[66:69], v[70:77], v[18:25], a[168:171], v168, v167 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[78:85], v[2:9], a[212:215], v169, v166 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	s_nop 7
	s_nop 2
	v_accvgpr_read_b32 v199, a69
	v_accvgpr_read_b32 v198, a68
	v_accvgpr_read_b32 v197, a67
	v_accvgpr_read_b32 v196, a66
	v_mfma_scale_f32_16x16x128_f8f6f4 a[66:69], v[70:77], v[34:41], a[108:111], v168, v167 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[94:101], v[18:25], a[236:239], v169, v167 op_sel_hi:[1,1,0]
	s_nop 7
	s_nop 2
	v_accvgpr_read_b32 v195, a69
	v_accvgpr_read_b32 v194, a68
	v_accvgpr_read_b32 v193, a67
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[86:93], v[34:41], a[0:3], v170, v167 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_accvgpr_read_b32 v192, a66
	s_nop 5
	v_accvgpr_write_b32 a0, v208
	v_accvgpr_write_b32 a1, v209
	v_accvgpr_write_b32 a2, v210
	v_accvgpr_write_b32 a3, v211
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[78:85], v[10:17], a[220:223], v169, v166 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_nop 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[94:101], v[34:41], a[0:3], v169, v167 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	s_nop 6
	v_accvgpr_write_b32 a0, v212
	v_accvgpr_write_b32 a1, v213
	v_accvgpr_write_b32 a2, v214
	v_accvgpr_write_b32 a3, v215
	v_mfma_scale_f32_16x16x128_f8f6f4 a[66:69], v[70:77], v[46:53], a[104:107], v168, v158 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	s_nop 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[86:93], v[54:61], a[0:3], v170, v165 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_nop 6
	v_accvgpr_write_b32 a0, v216
	v_accvgpr_write_b32 a1, v217
	v_accvgpr_write_b32 a2, v218
	v_accvgpr_write_b32 a3, v219
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[70:77], v[10:17], a[160:163], v168, v166 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_nop 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[94:101], v[54:61], a[0:3], v169, v165 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[70:73], v[70:77], v[26:33], a[70:73], v168, v165 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[62:65], v[70:77], v[54:61], a[62:65], v168, v165 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[70:77], v[62:69], a[52:55], v168, v158 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	;;#ASMSTART
	ds_read_b128 v[70:73], v160 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[74:77], v159 offset:0x1000

	;;#ASMEND
	buffer_load_dwordx4 v112, s[40:43], 0 offen lds
	s_add_u32 s40, s56, s48
	s_addc_u32 s41, s57, s37
	s_mov_b32 m0, s36
	s_addk_i32 s49, 0x180
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[70:77], v[2:9], a[96:99], v153, v166 op_sel_hi:[1,1,0]
	s_ashr_i32 s37, s49, 31
	s_add_u32 s36, s54, s49
	s_addc_u32 s37, s55, s37
	s_addk_i32 s59, 0x100
	s_cmpk_eq_i32 s59, 0x1f00
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[70:77], v[10:17], a[92:95], v153, v166 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[70:77], v[18:25], a[82:85], v153, v167 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[70:77], v[34:41], a[74:77], v153, v167 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[70:77], v[26:33], a[40:43], v153, v165 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[70:77], v[54:61], a[88:91], v153, v165 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	s_nop 7
	s_nop 2
	v_accvgpr_read_b32 v187, a43
	v_accvgpr_read_b32 v186, a42
	v_accvgpr_read_b32 v185, a41
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[70:77], v[46:53], a[32:35], v153, v158 op_sel_hi:[1,1,0]
	v_accvgpr_read_b32 v184, a40
	v_accvgpr_mov_b32 a40, a52
	v_accvgpr_mov_b32 a41, a53
	v_accvgpr_mov_b32 a42, a54
	v_accvgpr_mov_b32 a43, a55
	v_accvgpr_mov_b32 a52, a62
	v_accvgpr_mov_b32 a53, a63
	v_mfma_scale_f32_16x16x128_f8f6f4 a[78:81], v[70:77], v[62:69], a[78:81], v153, v158 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	;;#ASMSTART
	ds_read_b128 v[70:73], v160 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[74:77], v159 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(16)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	buffer_load_dwordx4 v110, s[40:43], 0 offen lds
	;;#ASMSTART
	ds_read_b128 v[42:45], v142 offset:0

	;;#ASMEND
	v_mfma_scale_f32_16x16x128_f8f6f4 a[74:77], v[70:77], v[2:9], a[20:23], v153, v166 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	s_mov_b32 m0, s45
	v_accvgpr_mov_b32 a54, a64
	v_accvgpr_mov_b32 a55, a65
	v_accvgpr_mov_b32 a60, a70
	v_accvgpr_mov_b32 a61, a71
	v_accvgpr_mov_b32 a62, a72
	v_accvgpr_mov_b32 a63, a73
	v_accvgpr_mov_b32 a20, a66
	v_accvgpr_mov_b32 a21, a67
	v_accvgpr_mov_b32 a22, a68
	v_accvgpr_mov_b32 a23, a69
	v_mfma_scale_f32_16x16x128_f8f6f4 a[66:69], v[70:77], v[18:25], a[28:31], v153, v167 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_accvgpr_mov_b32 a70, a116
	v_accvgpr_mov_b32 a71, a117
	v_accvgpr_mov_b32 a72, a118
	v_accvgpr_mov_b32 a73, a119
	v_accvgpr_mov_b32 a116, a164
	v_accvgpr_mov_b32 a117, a165
	v_accvgpr_mov_b32 a118, a166
	v_accvgpr_mov_b32 a28, a120
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[78:85], v[46:53], a[140:143], v169, v158 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_accvgpr_mov_b32 a29, a121
	v_accvgpr_mov_b32 a30, a122
	v_accvgpr_mov_b32 a31, a123
	v_accvgpr_mov_b32 a120, a124
	v_accvgpr_mov_b32 a121, a125
	v_accvgpr_mov_b32 a122, a126
	v_accvgpr_mov_b32 a123, a127
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[70:77], v[46:53], a[16:19], v153, v158 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	;;#ASMSTART
	ds_read_b128 v[46:49], v143 offset:0

	;;#ASMEND
	v_accvgpr_mov_b32 a127, a3
	buffer_load_dwordx4 v118, s[40:43], 0 offen lds
	v_accvgpr_mov_b32 a126, a2
	v_accvgpr_mov_b32 a125, a1
	v_accvgpr_mov_b32 a124, a0
	s_mov_b32 m0, s47
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[70:77], v[54:61], a[44:47], v153, v165 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	;;#ASMSTART
	ds_read_b128 v[58:61], v142 offset:0x800

	;;#ASMEND
	v_accvgpr_mov_b32 a119, a167
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[94:101], v[62:69], a[152:155], v169, v158 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[70:77], v[62:69], a[100:103], v153, v158 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	;;#ASMSTART
	ds_read_b128 v[62:65], v143 offset:0x800

	;;#ASMEND
	buffer_load_dwordx4 v120, s[40:43], 0 offen lds
	;;#ASMSTART
	ds_read_b128 v[50:53], v142 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[54:57], v143 offset:0x1000

	;;#ASMEND
	s_mov_b32 m0, s50
	s_nop 0
	buffer_load_dwordx4 v112, s[40:43], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[78:85], v[34:41], a[200:203], v169, v167 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_mov_b32 m0, s51
	s_nop 4
	v_accvgpr_mov_b32 a96, a152
	v_accvgpr_mov_b32 a97, a153
	v_accvgpr_mov_b32 a98, a154
	v_accvgpr_mov_b32 a99, a155
	v_accvgpr_mov_b32 a100, a104
	v_accvgpr_mov_b32 a155, a15
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[70:77], v[34:41], a[48:51], v153, v167 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	;;#ASMSTART
	ds_read_b128 v[34:37], v142 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[38:41], v143 offset:0x1800

	;;#ASMEND
	buffer_load_dwordx4 v106, s[36:39], 0 offen lds
	s_mov_b32 m0, s58
	v_accvgpr_mov_b32 a101, a105
	v_accvgpr_mov_b32 a102, a106
	v_accvgpr_mov_b32 a103, a107
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[86:93], v[2:9], a[224:227], v170, v166 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_accvgpr_mov_b32 a107, a7
	v_accvgpr_mov_b32 a154, a14
	v_accvgpr_mov_b32 a153, a13
	v_accvgpr_mov_b32 a152, a12
	v_accvgpr_mov_b32 a12, a168
	v_accvgpr_mov_b32 a106, a6
	v_accvgpr_mov_b32 a44, a48
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[94:101], v[2:9], a[204:207], v169, v166 op_sel_hi:[1,1,0]
	;;#ASMSTART
	ds_read_b128 v[2:5], v144 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[6:9], v145 offset:0

	;;#ASMEND
	buffer_load_dwordx4 v114, s[36:39], 0 offen lds
	s_mov_b32 m0, s60
	v_accvgpr_mov_b32 a45, a49
	v_accvgpr_mov_b32 a46, a50
	v_accvgpr_mov_b32 a47, a51
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[86:93], v[10:17], a[228:231], v170, v166 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_accvgpr_mov_b32 a48, a84
	v_accvgpr_mov_b32 a49, a85
	v_accvgpr_mov_b32 a50, a86
	v_accvgpr_mov_b32 a51, a87
	v_accvgpr_mov_b32 a84, a160
	v_accvgpr_mov_b32 a85, a161
	v_accvgpr_mov_b32 a86, a162
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[94:101], v[10:17], a[216:219], v169, v166 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_accvgpr_mov_b32 a87, a163
	v_accvgpr_mov_b32 a163, a11
	v_accvgpr_mov_b32 a105, a5
	v_accvgpr_mov_b32 a104, a4
	v_accvgpr_mov_b32 a13, a169
	v_accvgpr_mov_b32 a14, a170
	v_accvgpr_mov_b32 a15, a171
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[70:77], v[10:17], a[56:59], v153, v166 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	;;#ASMSTART
	ds_read_b128 v[10:13], v144 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[14:17], v145 offset:0x800

	;;#ASMEND
	buffer_load_dwordx4 v116, s[36:39], 0 offen lds
	s_mov_b32 m0, s63
	v_accvgpr_mov_b32 a162, a10
	v_accvgpr_mov_b32 a161, a9
	v_accvgpr_mov_b32 a160, a8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[86:93], v[18:25], a[232:235], v170, v167 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[78:85], v[18:25], a[196:199], v169, v167 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	;;#ASMSTART
	ds_read_b128 v[18:21], v144 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[22:25], v145 offset:0x1000

	;;#ASMEND
	buffer_load_dwordx4 v108, s[36:39], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[86:93], v[26:33], a[176:179], v170, v165 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[78:85], v[26:33], a[172:175], v169, v165 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[70:77], v[26:33], a[24:27], v153, v165 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	;;#ASMSTART
	ds_read_b128 v[26:29], v144 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[30:33], v145 offset:0x1800

	;;#ASMEND
	buffer_load_dword v170, v123, s[24:27], s59 offen
	buffer_load_dword v168, v123, s[0:3], s59 offen
	buffer_load_dword v169, v123, s[8:11], s59 offen
	buffer_load_dword v153, v123, s[4:7], s59 offen
	buffer_load_dword v166, v123, s[28:31], s59 offen
	buffer_load_dword v165, v123, s[12:15], s59 offen
	buffer_load_dword v167, v123, s[20:23], s59 offen
	buffer_load_dword v158, v123, s[16:19], s59 offen
	v_accvgpr_mov_b32 a24, a136
	v_accvgpr_mov_b32 a25, a137
	v_accvgpr_mov_b32 a26, a138
	v_accvgpr_mov_b32 a27, a139
	v_accvgpr_mov_b32 a136, a156
	v_accvgpr_mov_b32 a137, a157
	v_accvgpr_mov_b32 a138, a158
	v_accvgpr_mov_b32 a139, a159
	s_cbranch_scc0 .LBB0_3
; %bb.4:
	s_movk_i32 s37, 0x1f00
	buffer_load_dword v98, v123, s[24:27], s37 offen
	buffer_load_dword v99, v123, s[28:31], s37 offen
	buffer_load_dword v100, v123, s[20:23], s37 offen
	buffer_load_dword v101, v123, s[8:11], s37 offen
	s_waitcnt vmcnt(8)
	v_accvgpr_read_b32 v155, a23
	v_accvgpr_read_b32 v154, a22
	v_accvgpr_read_b32 v153, a21
	v_accvgpr_read_b32 v152, a20
	v_accvgpr_read_b32 v183, a31
	v_accvgpr_read_b32 v182, a30
	v_accvgpr_read_b32 v181, a29
	v_accvgpr_read_b32 v180, a28
	buffer_load_dword v105, v123, s[12:15], s37 offen
	v_accvgpr_read_b32 v179, a69
	s_waitcnt vmcnt(6)
	v_accvgpr_read_b32 v167, a73
	v_accvgpr_read_b32 v178, a68
	v_accvgpr_read_b32 v177, a67
	v_accvgpr_read_b32 v176, a66
	v_accvgpr_read_b32 v166, a72
	v_accvgpr_read_b32 v165, a71
	v_accvgpr_read_b32 v164, a70
	buffer_load_dword v106, v123, s[16:19], s37 offen
	v_add_u32_e32 v110, v103, v104
	v_add_u32_e32 v66, 0x14000, v110
	v_lshrrev_b32_e32 v67, 4, v66
	s_movk_i32 s36, 0x70
	buffer_load_dword v107, v123, s[0:3], s37 offen
	buffer_load_dword v108, v123, s[4:7], s37 offen
	;;#ASMSTART
	s_waitcnt vmcnt(16)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	v_bitop3_b32 v70, v67, v66, s36 bitop3:0x6c
	;;#ASMSTART
	ds_read_b128 v[90:93], v70 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[74:77], v70 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[66:69], v70 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[82:85], v70 offset:0x1800

	;;#ASMEND
	v_add_u32_e32 v70, 0x14040, v110
	v_accvgpr_read_b32 v147, a3
	v_lshrrev_b32_e32 v71, 4, v70
	v_accvgpr_read_b32 v146, a2
	v_accvgpr_read_b32 v145, a1
	v_accvgpr_read_b32 v144, a0
	v_bitop3_b32 v86, v71, v70, s36 bitop3:0x6c
	;;#ASMSTART
	ds_read_b128 v[94:97], v86 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[78:81], v86 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[70:73], v86 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[86:89], v86 offset:0x1800

	;;#ASMEND
	v_accvgpr_read_b32 v136, a164
	v_accvgpr_read_b32 v137, a165
	v_accvgpr_read_b32 v138, a166
	v_accvgpr_read_b32 v139, a167
	v_accvgpr_read_b32 v143, a19
	v_accvgpr_read_b32 v142, a18
	v_accvgpr_read_b32 v141, a17
	v_accvgpr_read_b32 v140, a16
	v_accvgpr_read_b32 v135, a35
	v_accvgpr_read_b32 v134, a34
	v_accvgpr_read_b32 v133, a33
	v_accvgpr_read_b32 v132, a32
	v_add_u32_e32 v109, v103, v102
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	v_accvgpr_read_b32 v175, a151
	v_accvgpr_read_b32 v174, a150
	v_accvgpr_read_b32 v173, a149
	v_accvgpr_read_b32 v172, a148
	v_accvgpr_read_b32 v163, a115
	v_accvgpr_read_b32 v162, a114
	v_accvgpr_read_b32 v161, a113
	v_accvgpr_read_b32 v160, a112
	v_accvgpr_read_b32 v119, a103
	v_accvgpr_read_b32 v116, a100
	v_accvgpr_read_b32 v118, a102
	v_accvgpr_read_b32 v117, a101
	v_accvgpr_read_b32 v127, a81
	v_accvgpr_read_b32 v126, a80
	v_accvgpr_read_b32 v125, a79
	v_accvgpr_read_b32 v124, a78
	s_waitcnt vmcnt(8)
	v_accvgpr_read_b32 v159, a63
	v_accvgpr_read_b32 v156, a60
	v_accvgpr_read_b32 v158, a62
	v_accvgpr_read_b32 v157, a61
	v_accvgpr_read_b32 v151, a95
	s_waitcnt vmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[58:65], v[2:9], a[224:227], v98, v99 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_accvgpr_read_b32 v150, a94
	v_accvgpr_read_b32 v149, a93
	v_accvgpr_read_b32 v148, a92
	v_accvgpr_read_b32 v131, a91
	v_accvgpr_read_b32 v128, a88
	v_accvgpr_read_b32 v130, a90
	v_accvgpr_read_b32 v129, a89
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[58:65], v[10:17], a[228:231], v98, v99 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[58:65], v[2:9], a[20:23], v98, v99 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	s_waitcnt vmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[58:65], v[18:25], a[232:235], v98, v100 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[58:65], v[18:25], a[20:23], v98, v100 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[58:65], v[26:33], a[236:239], v98, v100 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[58:65], v[26:33], a[20:23], v98, v100 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_waitcnt vmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[50:57], v[2:9], a[204:207], v101, v99 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[50:57], v[2:9], a[20:23], v101, v99 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[50:57], v[10:17], a[216:219], v101, v99 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[50:57], v[10:17], a[20:23], v101, v99 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[50:57], v[18:25], a[212:215], v101, v100 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[50:57], v[18:25], a[20:23], v101, v100 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[50:57], v[26:33], a[220:223], v101, v100 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[50:57], v[26:33], a[20:23], v101, v100 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[34:41], v[2:9], a[188:191], v101, v99 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[34:41], v[2:9], a[20:23], v101, v99 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[34:41], v[10:17], a[192:195], v101, v99 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[34:41], v[10:17], a[20:23], v101, v99 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[34:41], v[18:25], a[196:199], v101, v100 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[34:41], v[18:25], a[20:23], v101, v100 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[34:41], v[26:33], a[200:203], v101, v100 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[42:49], v[2:9], a[252:255], v98, v99 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[34:41], v[26:33], a[20:23], v101, v100 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_waitcnt vmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[42:49], v[90:97], a[184:187], v98, v105 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[42:49], v[90:97], a[20:23], v98, v105 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[42:49], v[74:81], a[208:211], v98, v105 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[42:49], v[74:81], a[20:23], v98, v105 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	s_waitcnt vmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[42:49], v[66:73], a[180:183], v98, v106 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[42:49], v[66:73], a[20:23], v98, v106 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[42:49], v[82:89], a[104:107], v98, v106 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[42:49], v[82:89], a[20:23], v98, v106 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[58:65], v[90:97], a[176:179], v98, v105 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[58:65], v[90:97], a[20:23], v98, v105 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[58:65], v[74:81], a[152:155], v98, v105 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[58:65], v[74:81], a[20:23], v98, v105 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[58:65], v[66:73], a[136:139], v98, v106 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[58:65], v[66:73], a[20:23], v98, v106 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[58:65], v[82:89], a[144:147], v98, v106 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[58:65], v[82:89], a[20:23], v98, v106 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[50:57], v[90:97], a[36:39], v101, v105 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[50:57], v[90:97], a[20:23], v101, v105 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[50:57], v[74:81], a[124:127], v101, v105 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[50:57], v[74:81], a[20:23], v101, v105 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[50:57], v[66:73], a[116:119], v101, v106 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[42:49], v[26:33], a[240:243], v98, v100 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[50:57], v[66:73], a[20:23], v101, v106 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[50:57], v[82:89], a[96:99], v101, v106 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[42:49], v[18:25], a[244:247], v98, v100 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[50:57], v[82:89], a[20:23], v101, v106 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[34:41], v[90:97], a[172:175], v101, v105 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[34:41], v[90:97], a[20:23], v101, v105 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[34:41], v[74:81], a[160:163], v101, v105 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[42:49], v[10:17], a[248:251], v98, v99 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[58:65], v[10:17], a[28:31], v98, v99 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[34:41], v[74:81], a[20:23], v101, v105 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[34:41], v[66:73], a[140:143], v101, v106 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[34:41], v[82:89], a[128:131], v101, v106 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[42:49], v[2:9], a[0:3], v98, v99 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[42:49], v[10:17], a[4:7], v98, v99 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[42:49], v[18:25], a[8:11], v98, v100 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[42:49], v[26:33], a[16:19], v98, v100 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_add_u32_e32 v42, 0x4000, v109
	v_lshrrev_b32_e32 v43, 4, v42
	v_bitop3_b32 v46, v43, v42, s36 bitop3:0x6c
	;;#ASMSTART
	ds_read_b128 v[42:45], v46 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[50:53], v46 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[58:61], v46 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[112:115], v46 offset:0x1800

	;;#ASMEND
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[34:41], v[66:73], a[20:23], v101, v106 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[34:41], v[82:89], a[28:31], v101, v106 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_add_u32_e32 v34, 0x4040, v109
	v_lshrrev_b32_e32 v35, 4, v34
	v_bitop3_b32 v34, v35, v34, s36 bitop3:0x6c
	;;#ASMSTART
	ds_read_b128 v[46:49], v34 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[54:57], v34 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[62:65], v34 offset:0x1000

	;;#ASMEND
	s_waitcnt vmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[42:49], v[2:9], a[132:135], v107, v99 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[42:49], v[2:9], a[28:31], v107, v99 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[42:49], v[10:17], a[24:27], v107, v99 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[42:49], v[10:17], a[28:31], v107, v99 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	s_nop 5
	v_accvgpr_write_b32 a24, v180
	v_accvgpr_write_b32 a25, v181
	v_accvgpr_write_b32 a26, v182
	v_accvgpr_write_b32 a27, v183
	s_nop 0
	v_accvgpr_write_b32 a28, v188
	v_accvgpr_write_b32 a29, v189
	v_accvgpr_write_b32 a30, v190
	v_accvgpr_write_b32 a31, v191
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[42:49], v[18:25], a[28:31], v107, v100 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[42:49], v[18:25], a[28:31], v107, v100 op_sel_hi:[1,1,0]
	s_nop 7
	s_nop 2
	v_accvgpr_write_b32 a28, v172
	v_accvgpr_write_b32 a29, v173
	v_accvgpr_write_b32 a30, v174
	v_accvgpr_write_b32 a31, v175
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[42:49], v[26:33], a[28:31], v107, v100 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[42:49], v[26:33], a[28:31], v107, v100 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[50:57], v[2:9], a[24:27], v107, v99 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[50:57], v[2:9], a[28:31], v107, v99 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	s_nop 5
	v_accvgpr_write_b32 a24, v196
	v_accvgpr_write_b32 a25, v197
	v_accvgpr_write_b32 a26, v198
	v_accvgpr_write_b32 a27, v199
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[50:57], v[10:17], a[84:87], v107, v99 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[50:57], v[10:17], a[28:31], v107, v99 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[50:57], v[18:25], a[24:27], v107, v100 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	s_nop 6
	v_accvgpr_write_b32 a24, v192
	v_accvgpr_write_b32 a25, v193
	v_accvgpr_write_b32 a26, v194
	v_accvgpr_write_b32 a27, v195
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[50:57], v[18:25], a[28:31], v107, v100 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	s_nop 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[50:57], v[26:33], a[24:27], v107, v100 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[50:57], v[26:33], a[28:31], v107, v100 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_waitcnt vmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[58:65], v[2:9], a[108:111], v108, v99 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[58:65], v[2:9], a[28:31], v108, v99 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[58:65], v[10:17], a[12:15], v108, v99 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[58:65], v[10:17], a[28:31], v108, v99 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	s_nop 5
	v_accvgpr_write_b32 a12, v176
	v_accvgpr_write_b32 a13, v177
	v_accvgpr_write_b32 a14, v178
	v_accvgpr_write_b32 a15, v179
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[58:65], v[18:25], a[48:51], v108, v100 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[58:65], v[18:25], a[28:31], v108, v100 op_sel_hi:[1,1,0]
	s_nop 7
	s_nop 2
	v_accvgpr_write_b32 a28, v116
	v_accvgpr_write_b32 a29, v117
	v_accvgpr_write_b32 a30, v118
	v_accvgpr_write_b32 a31, v119
	;;#ASMSTART
	ds_read_b128 v[116:119], v34 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(8)
	;;#ASMEND
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[112:119], v[18:25], a[12:15], v108, v100 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[58:65], v[26:33], a[28:31], v108, v100 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[112:119], v[18:25], a[12:15], v108, v100 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[112:119], v[26:33], a[44:47], v108, v100 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[58:65], v[26:33], a[28:31], v108, v100 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[112:119], v[2:9], a[74:77], v108, v99 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[112:119], v[26:33], a[12:15], v108, v100 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[42:49], v[90:97], a[120:123], v107, v105 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[42:49], v[90:97], a[12:15], v107, v105 op_sel_hi:[1,1,0]
	s_nop 7
	s_nop 2
	v_accvgpr_write_b32 a12, v160
	v_accvgpr_write_b32 a13, v161
	v_accvgpr_write_b32 a14, v162
	v_accvgpr_write_b32 a15, v163
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[112:119], v[2:9], a[28:31], v108, v99 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_or_b32_e32 v2, 0x8000, v109
	v_lshrrev_b32_e32 v3, 4, v109
	v_bitop3_b32 v2, v3, v2, s36 bitop3:0x6c
	v_add_u32_e32 v6, 0x1c040, v110
	v_lshrrev_b32_e32 v7, 4, v6
	v_bitop3_b32 v6, v7, v6, s36 bitop3:0x6c
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[42:49], v[74:81], a[12:15], v107, v105 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[42:49], v[74:81], a[12:15], v107, v105 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	s_nop 7
	s_nop 2
	v_accvgpr_write_b32 a12, v164
	v_accvgpr_write_b32 a13, v165
	v_accvgpr_write_b32 a14, v166
	v_accvgpr_write_b32 a15, v167
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[112:119], v[10:17], a[56:59], v108, v99 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_nop 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[42:49], v[66:73], a[12:15], v107, v106 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[42:49], v[66:73], a[12:15], v107, v106 op_sel_hi:[1,1,0]
	s_nop 7
	s_nop 2
	v_accvgpr_write_b32 a12, v200
	v_accvgpr_write_b32 a13, v201
	v_accvgpr_write_b32 a14, v202
	v_accvgpr_write_b32 a15, v203
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[112:119], v[10:17], a[28:31], v108, v99 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_nop 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[42:49], v[82:89], a[12:15], v107, v106 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[42:49], v[82:89], a[12:15], v107, v106 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	s_nop 7
	s_nop 2
	v_accvgpr_write_b32 a12, v156
	v_accvgpr_write_b32 a13, v157
	v_accvgpr_write_b32 a14, v158
	v_accvgpr_write_b32 a15, v159
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[50:57], v[90:97], a[12:15], v107, v105 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[50:57], v[90:97], a[12:15], v107, v105 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[50:57], v[74:81], a[52:55], v107, v105 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[50:57], v[74:81], a[12:15], v107, v105 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_nop 7
	s_nop 2
	v_accvgpr_write_b32 a12, v152
	v_accvgpr_write_b32 a13, v153
	v_accvgpr_write_b32 a14, v154
	v_accvgpr_write_b32 a15, v155
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[50:57], v[66:73], a[12:15], v107, v106 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[50:57], v[66:73], a[12:15], v107, v106 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[50:57], v[82:89], a[40:43], v107, v106 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[50:57], v[82:89], a[12:15], v107, v106 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_nop 7
	s_nop 2
	v_accvgpr_write_b32 a12, v184
	v_accvgpr_write_b32 a13, v185
	v_accvgpr_write_b32 a14, v186
	v_accvgpr_write_b32 a15, v187
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[58:65], v[90:97], a[12:15], v108, v105 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[58:65], v[90:97], a[12:15], v108, v105 op_sel_hi:[1,1,0]
	s_nop 7
	s_nop 2
	v_accvgpr_write_b32 a12, v128
	v_accvgpr_write_b32 a13, v129
	v_accvgpr_write_b32 a14, v130
	v_accvgpr_write_b32 a15, v131
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[58:65], v[74:81], a[12:15], v108, v105 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[58:65], v[74:81], a[12:15], v108, v105 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	s_nop 7
	s_nop 2
	v_accvgpr_write_b32 a12, v132
	v_accvgpr_write_b32 a13, v133
	v_accvgpr_write_b32 a14, v134
	v_accvgpr_write_b32 a15, v135
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[58:65], v[66:73], a[12:15], v108, v106 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[58:65], v[66:73], a[12:15], v108, v106 op_sel_hi:[1,1,0]
	s_nop 7
	s_nop 3
	v_accvgpr_read_b32 v247, a15
	v_accvgpr_read_b32 v246, a14
	v_accvgpr_read_b32 v245, a13
	v_accvgpr_read_b32 v244, a12
	v_accvgpr_write_b32 a12, v124
	v_accvgpr_write_b32 a13, v125
	v_accvgpr_write_b32 a14, v126
	v_accvgpr_write_b32 a15, v127
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[58:65], v[82:89], a[12:15], v108, v106 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[58:65], v[82:89], a[12:15], v108, v106 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	s_nop 7
	s_nop 3
	v_accvgpr_read_b32 v127, a15
	v_accvgpr_read_b32 v126, a14
	v_accvgpr_read_b32 v125, a13
	v_accvgpr_read_b32 v124, a12
	v_accvgpr_write_b32 a12, v136
	v_accvgpr_write_b32 a13, v137
	v_accvgpr_write_b32 a14, v138
	v_accvgpr_write_b32 a15, v139
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[112:119], v[90:97], a[12:15], v108, v105 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[112:119], v[90:97], a[12:15], v108, v105 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	;;#ASMSTART
	ds_read_b128 v[90:93], v2 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[50:53], v2 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[34:37], v2 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[98:101], v2 offset:0x1800

	;;#ASMEND
	v_add_u32_e32 v2, 0x8040, v109
	v_lshrrev_b32_e32 v3, 4, v2
	v_bitop3_b32 v2, v3, v2, s36 bitop3:0x6c
	;;#ASMSTART
	ds_read_b128 v[94:97], v2 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[54:57], v2 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[38:41], v2 offset:0x1000

	;;#ASMEND
	v_lshrrev_b32_e32 v3, 4, v110
	s_nop 7
	v_accvgpr_read_b32 v243, a15
	v_accvgpr_read_b32 v242, a14
	v_accvgpr_read_b32 v241, a13
	v_accvgpr_read_b32 v240, a12
	v_accvgpr_write_b32 a12, v144
	v_accvgpr_write_b32 a13, v145
	v_accvgpr_write_b32 a14, v146
	v_accvgpr_write_b32 a15, v147
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[112:119], v[74:81], a[12:15], v108, v105 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[112:119], v[74:81], a[12:15], v108, v105 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	;;#ASMSTART
	ds_read_b128 v[102:105], v2 offset:0x1800

	;;#ASMEND
	v_or_b32_e32 v2, 0x18000, v110
	v_bitop3_b32 v2, v3, v2, s36 bitop3:0x6c
	;;#ASMSTART
	ds_read_b128 v[74:77], v2 offset:0

	;;#ASMEND
	s_nop 7
	s_nop 1
	v_accvgpr_read_b32 v251, a15
	v_accvgpr_read_b32 v250, a14
	v_accvgpr_read_b32 v249, a13
	v_accvgpr_read_b32 v248, a12
	v_accvgpr_write_b32 a12, v148
	v_accvgpr_write_b32 a13, v149
	v_accvgpr_write_b32 a14, v150
	v_accvgpr_write_b32 a15, v151
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[112:119], v[66:73], a[12:15], v108, v106 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[112:119], v[66:73], a[12:15], v108, v106 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	;;#ASMSTART
	ds_read_b128 v[66:69], v2 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[58:61], v2 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[42:45], v2 offset:0x1800

	;;#ASMEND
	v_add_u32_e32 v2, 0x18040, v110
	v_lshrrev_b32_e32 v3, 4, v2
	v_bitop3_b32 v2, v3, v2, s36 bitop3:0x6c
	;;#ASMSTART
	ds_read_b128 v[78:81], v2 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[70:73], v2 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[62:65], v2 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[46:49], v2 offset:0x1800

	;;#ASMEND
	buffer_load_dword v111, v123, s[24:27], s37 offen
	buffer_load_dword v191, v123, s[28:31], s37 offen
	buffer_load_dword v192, v123, s[20:23], s37 offen
	s_waitcnt vmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[90:97], v[66:73], a[4:7], v111, v191 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	s_nop 3
	v_accvgpr_read_b32 v131, a15
	v_accvgpr_read_b32 v130, a14
	v_accvgpr_read_b32 v129, a13
	v_accvgpr_read_b32 v128, a12
	v_accvgpr_write_b32 a12, v140
	v_accvgpr_write_b32 a13, v141
	v_accvgpr_write_b32 a14, v142
	v_accvgpr_write_b32 a15, v143
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[90:97], v[66:73], a[4:7], v111, v191 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_add_u32_e32 v2, 0x1c000, v110
	v_lshrrev_b32_e32 v3, 4, v2
	v_bitop3_b32 v2, v3, v2, s36 bitop3:0x6c
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[112:119], v[82:89], a[12:15], v108, v106 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[112:119], v[82:89], a[12:15], v108, v106 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	buffer_load_dword v86, v123, s[8:11], s37 offen
	buffer_load_dword v108, v123, s[12:15], s37 offen
	buffer_load_dword v106, v123, s[16:19], s37 offen
	buffer_load_dword v133, v123, s[0:3], s37 offen
	buffer_load_dword v107, v123, s[4:7], s37 offen
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[26:29], v2 offset:0

	;;#ASMEND
	s_waitcnt vmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[90:97], v[58:65], a[8:11], v111, v192 op_sel_hi:[0,0,0]
	;;#ASMSTART
	ds_read_b128 v[18:21], v2 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[10:13], v2 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[2:5], v2 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[30:33], v6 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[22:25], v6 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[14:17], v6 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[6:9], v6 offset:0x1800

	;;#ASMEND
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[90:97], v[42:49], a[16:19], v111, v192 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_nop 2
	v_accvgpr_read_b32 v115, a15
	v_accvgpr_read_b32 v114, a14
	v_accvgpr_read_b32 v113, a13
	v_accvgpr_read_b32 v112, a12
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[90:97], v[42:49], a[8:11], v111, v192 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[50:57], v[74:81], a[228:231], v111, v191 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[50:57], v[74:81], a[8:11], v111, v191 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[50:57], v[66:73], a[32:35], v111, v191 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[50:57], v[66:73], a[8:11], v111, v191 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[50:57], v[58:65], a[68:71], v111, v192 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[50:57], v[58:65], a[8:11], v111, v192 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[50:57], v[42:49], a[224:227], v111, v192 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[50:57], v[42:49], a[8:11], v111, v192 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_waitcnt vmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[34:41], v[74:81], a[204:207], v86, v191 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[34:41], v[74:81], a[8:11], v86, v191 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[34:41], v[66:73], a[216:219], v86, v191 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[34:41], v[66:73], a[8:11], v86, v191 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[34:41], v[58:65], a[212:215], v86, v192 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[34:41], v[58:65], a[8:11], v86, v192 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[34:41], v[42:49], a[220:223], v86, v192 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[34:41], v[42:49], a[8:11], v86, v192 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[98:105], v[74:81], a[188:191], v86, v191 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[98:105], v[74:81], a[8:11], v86, v191 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[98:105], v[66:73], a[192:195], v86, v191 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[98:105], v[66:73], a[8:11], v86, v191 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[98:105], v[58:65], a[196:199], v86, v192 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[98:105], v[58:65], a[8:11], v86, v192 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[98:105], v[42:49], a[252:255], v86, v192 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[98:105], v[42:49], a[8:11], v86, v192 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_waitcnt vmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[90:97], v[26:33], a[184:187], v111, v108 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[90:97], v[26:33], a[8:11], v111, v108 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[90:97], v[18:25], a[200:203], v111, v108 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[90:97], v[18:25], a[8:11], v111, v108 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	s_waitcnt vmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[90:97], v[10:17], a[180:183], v111, v106 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[50:57], v[18:25], a[152:155], v111, v108 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[90:97], v[10:17], a[8:11], v111, v106 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[90:97], v[2:9], a[164:167], v111, v106 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[50:57], v[26:33], a[176:179], v111, v108 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[50:57], v[18:25], a[152:155], v111, v108 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[50:57], v[10:17], a[208:211], v111, v106 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[50:57], v[26:33], a[164:167], v111, v108 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[50:57], v[10:17], a[152:155], v111, v106 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[50:57], v[2:9], a[232:235], v111, v106 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[50:57], v[2:9], a[152:155], v111, v106 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[34:41], v[26:33], a[168:171], v86, v108 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[34:41], v[10:17], a[240:243], v86, v106 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[34:41], v[10:17], a[168:171], v86, v106 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[34:41], v[2:9], a[244:247], v86, v106 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[90:97], v[74:81], a[0:3], v111, v191 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[34:41], v[2:9], a[168:171], v86, v106 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[98:105], v[26:33], a[172:175], v86, v108 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[90:97], v[74:81], a[0:3], v111, v191 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[98:105], v[26:33], a[168:171], v86, v108 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[98:105], v[18:25], a[248:251], v86, v108 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[98:105], v[10:17], a[20:23], v86, v106 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[98:105], v[2:9], a[36:39], v86, v106 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[34:41], v[26:33], a[152:155], v86, v108 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[34:41], v[18:25], a[236:239], v86, v108 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[98:105], v[18:25], a[168:171], v86, v108 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[98:105], v[10:17], a[20:23], v86, v106 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[98:105], v[2:9], a[36:39], v86, v106 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_nop 2
	v_accvgpr_read_b32 v98, a0
	v_accvgpr_read_b32 v99, a1
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v98, s33, v98
	v_accvgpr_write_b32 a34, v98
	v_mul_f32_e32 v98, s33, v99
	v_accvgpr_write_b32 a35, v98
	v_accvgpr_read_b32 v98, a2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[90:97], v[58:65], a[4:7], v111, v192 op_sel_hi:[1,1,0]
	v_accvgpr_read_b32 v99, a3
	v_mul_f32_e32 v98, s33, v98
	v_accvgpr_write_b32 a33, v98
	v_mul_f32_e32 v98, s33, v99
	v_accvgpr_write_b32 a32, v98
	v_accvgpr_read_b32 v98, a160
	v_accvgpr_read_b32 v99, a161
	v_mul_f32_e32 v98, s33, v98
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[34:41], v[18:25], a[152:155], v86, v108 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_add_u32_e32 v34, 0xc000, v109
	v_accvgpr_write_b32 a160, v98
	v_mul_f32_e32 v98, s33, v99
	v_lshrrev_b32_e32 v35, 4, v34
	v_accvgpr_write_b32 a161, v98
	v_accvgpr_read_b32 v98, a162
	v_bitop3_b32 v34, v35, v34, s36 bitop3:0x6c
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[90:97], v[2:9], a[8:11], v111, v106 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	;;#ASMSTART
	ds_read_b128 v[90:93], v34 offset:0

	;;#ASMEND
	v_accvgpr_read_b32 v99, a163
	v_mul_f32_e32 v98, s33, v98
	;;#ASMSTART
	ds_read_b128 v[82:85], v34 offset:0x800

	;;#ASMEND
	v_accvgpr_write_b32 a162, v98
	v_mul_f32_e32 v98, s33, v99
	;;#ASMSTART
	ds_read_b128 v[50:53], v34 offset:0x1000

	;;#ASMEND
	v_add_u32_e32 v38, 0xc040, v109
	v_accvgpr_write_b32 a163, v98
	v_accvgpr_read_b32 v98, a4
	;;#ASMSTART
	ds_read_b128 v[34:37], v34 offset:0x1800

	;;#ASMEND
	v_lshrrev_b32_e32 v39, 4, v38
	v_accvgpr_read_b32 v99, a5
	v_mul_f32_e32 v98, s33, v98
	v_bitop3_b32 v38, v39, v38, s36 bitop3:0x6c
	;;#ASMSTART
	ds_read_b128 v[94:97], v38 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[86:89], v38 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[54:57], v38 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[38:41], v38 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_waitcnt vmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[90:97], v[66:73], a[140:143], v133, v191 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	s_nop 6
	v_accvgpr_write_b32 a140, v98
	v_mul_f32_e32 v98, s33, v99
	v_accvgpr_write_b32 a141, v98
	v_accvgpr_read_b32 v98, a6
	v_accvgpr_read_b32 v99, a7
	v_mul_f32_e32 v98, s33, v98
	v_accvgpr_write_b32 a142, v98
	v_mul_f32_e32 v98, s33, v99
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[90:97], v[58:65], a[136:139], v133, v192 op_sel_hi:[0,0,0]
	v_accvgpr_write_b32 a143, v98
	v_accvgpr_read_b32 v98, a156
	v_accvgpr_read_b32 v99, a157
	v_mul_f32_e32 v98, s33, v98
	v_accvgpr_write_b32 a156, v98
	v_mul_f32_e32 v118, s33, v99
	v_accvgpr_read_b32 v98, a158
	v_accvgpr_read_b32 v99, a159
	v_mul_f32_e32 v119, s33, v98
	v_mul_f32_e32 v117, s33, v99
	v_accvgpr_read_b32 v98, a144
	v_accvgpr_read_b32 v99, a145
	v_mul_f32_e32 v121, s33, v98
	v_mul_f32_e32 v98, s33, v99
	v_accvgpr_write_b32 a145, v98
	v_accvgpr_read_b32 v98, a146
	v_accvgpr_read_b32 v99, a147
	v_mul_f32_e32 v120, s33, v98
	v_mul_f32_e32 v98, s33, v99
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[90:97], v[58:65], a[4:7], v133, v192 op_sel_hi:[1,1,0]
	v_accvgpr_write_b32 a144, v98
	v_accvgpr_read_b32 v98, a16
	v_accvgpr_read_b32 v99, a17
	v_mul_f32_e32 v98, s33, v98
	v_accvgpr_write_b32 a146, v98
	v_mul_f32_e32 v98, s33, v99
	v_accvgpr_write_b32 a147, v98
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[90:97], v[42:49], a[132:135], v133, v192 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_accvgpr_read_b32 v98, a18
	v_accvgpr_read_b32 v99, a19
	v_mul_f32_e32 v98, s33, v98
	s_nop 3
	v_accvgpr_write_b32 a132, v98
	v_mul_f32_e32 v98, s33, v99
	v_accvgpr_write_b32 a133, v98
	v_accvgpr_read_b32 v98, a120
	v_accvgpr_read_b32 v99, a121
	v_mul_f32_e32 v123, s33, v98
	v_mul_f32_e32 v116, s33, v99
	v_accvgpr_read_b32 v98, a122
	v_accvgpr_read_b32 v99, a123
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[90:97], v[42:49], a[4:7], v133, v192 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mul_f32_e32 v136, s33, v98
	v_mul_f32_e32 v137, s33, v99
	v_accvgpr_read_b32 v98, a68
	v_accvgpr_read_b32 v99, a69
	v_mul_f32_e32 v139, s33, v98
	v_mul_f32_e32 v153, s33, v99
	v_accvgpr_read_b32 v98, a70
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[82:89], v[74:81], a[128:131], v133, v191 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_accvgpr_read_b32 v99, a71
	v_mul_f32_e32 v157, s33, v98
	v_mul_f32_e32 v158, s33, v99
	v_accvgpr_read_b32 v98, a24
	v_accvgpr_read_b32 v99, a25
	v_mul_f32_e32 v162, s33, v98
	v_mul_f32_e32 v154, s33, v99
	v_accvgpr_read_b32 v98, a26
	v_accvgpr_read_b32 v99, a27
	v_mul_f32_e32 v147, s33, v98
	v_mul_f32_e32 v138, s33, v99
	v_accvgpr_read_b32 v98, a228
	v_accvgpr_read_b32 v99, a229
	v_mul_f32_e32 v148, s33, v98
	v_mul_f32_e32 v140, s33, v99
	v_accvgpr_read_b32 v98, a230
	v_accvgpr_read_b32 v99, a231
	v_mul_f32_e32 v142, s33, v98
	v_mul_f32_e32 v143, s33, v99
	v_accvgpr_read_b32 v98, a224
	v_accvgpr_read_b32 v99, a225
	v_mul_f32_e32 v160, s33, v98
	v_mul_f32_e32 v161, s33, v99
	v_accvgpr_read_b32 v98, a226
	v_accvgpr_read_b32 v99, a227
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[82:89], v[74:81], a[4:7], v133, v191 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mul_f32_e32 v163, s33, v98
	v_mul_f32_e32 v178, s33, v99
	v_accvgpr_read_b32 v98, a212
	v_accvgpr_read_b32 v99, a213
	v_mul_f32_e32 v181, s33, v98
	v_mul_f32_e32 v182, s33, v99
	v_accvgpr_read_b32 v98, a214
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[82:89], v[66:73], a[124:127], v133, v191 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_accvgpr_read_b32 v99, a215
	v_mul_f32_e32 v189, s33, v98
	v_mul_f32_e32 v190, s33, v99
	v_accvgpr_read_b32 v98, a204
	v_accvgpr_read_b32 v99, a205
	v_mul_f32_e32 v188, s33, v98
	v_mul_f32_e32 v183, s33, v99
	v_accvgpr_read_b32 v98, a206
	v_accvgpr_read_b32 v99, a207
	v_mul_f32_e32 v186, s33, v98
	v_mul_f32_e32 v179, s33, v99
	v_accvgpr_read_b32 v98, a12
	v_accvgpr_read_b32 v99, a13
	v_mul_f32_e32 v174, s33, v98
	v_mul_f32_e32 v167, s33, v99
	v_accvgpr_read_b32 v98, a14
	v_accvgpr_read_b32 v99, a15
	s_waitcnt vmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[50:57], v[66:73], a[104:107], v107, v191 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mul_f32_e32 v180, s33, v98
	v_accvgpr_read_b32 v98, a192
	v_mul_f32_e32 v170, s33, v99
	v_accvgpr_read_b32 v99, a193
	v_mul_f32_e32 v168, s33, v98
	v_accvgpr_read_b32 v98, a194
	v_mul_f32_e32 v175, s33, v99
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[82:89], v[66:73], a[4:7], v133, v191 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_accvgpr_read_b32 v99, a195
	v_mul_f32_e32 v171, s33, v98
	v_accvgpr_read_b32 v98, a188
	v_mul_f32_e32 v164, s33, v99
	v_accvgpr_read_b32 v99, a189
	v_mul_f32_e32 v176, s33, v98
	v_accvgpr_read_b32 v98, a190
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[82:89], v[58:65], a[116:119], v133, v192 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mul_f32_e32 v169, s33, v99
	v_accvgpr_read_b32 v99, a191
	v_mul_f32_e32 v165, s33, v98
	v_accvgpr_read_b32 v98, a216
	v_mul_f32_e32 v159, s33, v99
	v_accvgpr_read_b32 v99, a217
	v_mul_f32_e32 v166, s33, v98
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[50:57], v[66:73], a[12:15], v107, v191 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_accvgpr_read_b32 v98, a218
	v_mul_f32_e32 v144, s33, v99
	v_accvgpr_read_b32 v99, a219
	v_mul_f32_e32 v98, s33, v98
	v_accvgpr_write_b32 a105, v98
	v_mul_f32_e32 v98, s33, v99
	v_accvgpr_write_b32 a104, v98
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[50:57], v[58:65], a[100:103], v107, v192 op_sel_hi:[0,0,0]
	v_accvgpr_read_b32 v98, a196
	v_accvgpr_read_b32 v99, a197
	v_mul_f32_e32 v98, s33, v98
	v_accvgpr_write_b32 a107, v98
	v_mul_f32_e32 v98, s33, v99
	v_accvgpr_write_b32 a106, v98
	v_accvgpr_read_b32 v98, a198
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[82:89], v[58:65], a[4:7], v133, v192 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_accvgpr_read_b32 v99, a199
	v_mul_f32_e32 v98, s33, v98
	v_mul_f32_e32 v145, s33, v99
	v_accvgpr_read_b32 v99, a185
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[82:89], v[42:49], a[112:115], v133, v192 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[50:57], v[58:65], a[12:15], v107, v192 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[50:57], v[42:49], a[96:99], v107, v192 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[82:89], v[42:49], a[4:7], v133, v192 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[50:57], v[74:81], a[108:111], v107, v191 op_sel_hi:[0,0,0]
	s_nop 6
	v_accvgpr_write_b32 a108, v98
	v_accvgpr_read_b32 v98, a184
	v_mul_f32_e32 v134, s33, v98
	v_mul_f32_e32 v98, s33, v99
	v_accvgpr_write_b32 a109, v98
	v_accvgpr_read_b32 v98, a186
	v_accvgpr_read_b32 v99, a187
	v_mul_f32_e32 v132, s33, v98
	v_mul_f32_e32 v146, s33, v99
	v_accvgpr_read_b32 v98, a8
	v_accvgpr_read_b32 v99, a9
	v_mul_f32_e32 v141, s33, v98
	v_mul_f32_e32 v98, s33, v99
	v_accvgpr_write_b32 a110, v98
	v_accvgpr_read_b32 v98, a10
	v_accvgpr_read_b32 v99, a11
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[50:57], v[42:49], a[12:15], v107, v192 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mul_f32_e32 v98, s33, v98
	v_accvgpr_write_b32 a96, v98
	v_mul_f32_e32 v98, s33, v99
	v_accvgpr_read_b32 v99, a181
	v_mul_f32_e32 v156, s33, v99
	v_accvgpr_read_b32 v99, a183
	v_mul_f32_e32 v173, s33, v99
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[34:41], v[74:81], a[92:95], v107, v191 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_accvgpr_read_b32 v99, a177
	v_mul_f32_e32 v184, s33, v99
	v_accvgpr_read_b32 v99, a179
	v_mul_f32_e32 v185, s33, v99
	v_accvgpr_read_b32 v99, a165
	v_mul_f32_e32 v150, s33, v99
	v_accvgpr_write_b32 a97, v98
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[34:41], v[74:81], a[12:15], v107, v191 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_accvgpr_read_b32 v98, a180
	v_mul_f32_e32 v155, s33, v98
	v_accvgpr_read_b32 v98, a182
	v_mul_f32_e32 v172, s33, v98
	v_accvgpr_read_b32 v98, a176
	v_mul_f32_e32 v177, s33, v98
	v_accvgpr_read_b32 v98, a178
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[34:41], v[66:73], a[88:91], v107, v191 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mul_f32_e32 v187, s33, v98
	v_accvgpr_read_b32 v98, a164
	v_mul_f32_e32 v149, s33, v98
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[34:41], v[42:49], a[76:79], v107, v192 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[90:97], v[74:81], a[148:151], v133, v191 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[34:41], v[58:65], a[84:87], v107, v192 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[90:97], v[66:73], a[0:3], v133, v191 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[34:41], v[66:73], a[12:15], v107, v191 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_accvgpr_read_b32 v67, a221
	v_accvgpr_read_b32 v66, a220
	v_mul_f32_e32 v105, s33, v67
	v_accvgpr_read_b32 v67, a223
	v_mul_f32_e32 v104, s33, v66
	v_accvgpr_read_b32 v66, a222
	v_mul_f32_e32 v101, s33, v67
	v_accvgpr_read_b32 v67, a153
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[34:41], v[42:49], a[76:79], v107, v192 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_accvgpr_read_b32 v42, a202
	v_accvgpr_read_b32 v43, a203
	v_mul_f32_e32 v103, s33, v66
	v_accvgpr_read_b32 v66, a152
	v_mul_f32_e32 v102, s33, v67
	v_accvgpr_read_b32 v67, a155
	v_mul_f32_e32 v70, s33, v42
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[90:97], v[74:81], a[148:151], v133, v191 op_sel_hi:[1,1,0]
	v_mul_f32_e32 v71, s33, v43
	v_accvgpr_read_b32 v42, a172
	v_accvgpr_read_b32 v43, a173
	v_mul_f32_e32 v100, s33, v66
	v_accvgpr_read_b32 v66, a154
	v_mul_f32_e32 v99, s33, v67
	v_mul_f32_e32 v68, s33, v42
	v_mul_f32_e32 v67, s33, v43
	v_accvgpr_read_b32 v42, a174
	v_accvgpr_read_b32 v43, a175
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[50:57], v[74:81], a[4:7], v107, v191 op_sel_hi:[1,1,0]
	v_mul_f32_e32 v81, s33, v66
	v_mul_f32_e32 v66, s33, v42
	v_mul_f32_e32 v69, s33, v43
	v_accvgpr_read_b32 v42, a168
	v_accvgpr_read_b32 v43, a169
	v_mul_f32_e32 v72, s33, v42
	v_accvgpr_read_b32 v42, a170
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[34:41], v[58:65], a[84:87], v107, v192 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mul_f32_e32 v64, s33, v43
	v_accvgpr_read_b32 v43, a171
	v_mul_f32_e32 v73, s33, v42
	v_mul_f32_e32 v65, s33, v43
	v_accvgpr_read_b32 v42, a20
	v_accvgpr_read_b32 v43, a21
	v_mul_f32_e32 v191, s33, v42
	v_mul_f32_e32 v192, s33, v43
	v_accvgpr_read_b32 v42, a22
	v_accvgpr_read_b32 v43, a23
	v_mul_f32_e32 v193, s33, v42
	v_mul_f32_e32 v194, s33, v43
	v_accvgpr_read_b32 v42, a36
	v_accvgpr_read_b32 v43, a37
	v_mul_f32_e32 v195, s33, v42
	v_mul_f32_e32 v196, s33, v43
	v_accvgpr_read_b32 v42, a38
	v_accvgpr_read_b32 v43, a39
	v_mul_f32_e32 v197, s33, v42
	v_mul_f32_e32 v200, s33, v43
	v_accvgpr_read_b32 v42, a148
	v_accvgpr_read_b32 v43, a149
	v_mul_f32_e32 v199, s33, v42
	v_mul_f32_e32 v198, s33, v43
	v_accvgpr_read_b32 v42, a150
	v_accvgpr_read_b32 v43, a151
	v_mul_f32_e32 v202, s33, v42
	v_mul_f32_e32 v201, s33, v43
	v_accvgpr_read_b32 v42, a0
	v_accvgpr_read_b32 v43, a1
	v_mul_f32_e32 v203, s33, v42
	v_mul_f32_e32 v204, s33, v43
	v_accvgpr_read_b32 v42, a2
	v_accvgpr_read_b32 v43, a3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[82:89], v[26:33], a[56:59], v133, v108 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mul_f32_e32 v205, s33, v42
	v_mul_f32_e32 v206, s33, v43
	v_accvgpr_read_b32 v42, a136
	v_accvgpr_read_b32 v43, a137
	v_mul_f32_e32 v207, s33, v42
	v_mul_f32_e32 v208, s33, v43
	v_accvgpr_read_b32 v42, a138
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[90:97], v[26:33], a[80:83], v133, v108 op_sel_hi:[0,0,0]
	v_accvgpr_read_b32 v43, a139
	v_accvgpr_read_b32 v74, a166
	v_mul_f32_e32 v135, s33, v74
	v_accvgpr_read_b32 v74, a232
	v_accvgpr_read_b32 v75, a167
	v_mul_f32_e32 v111, s33, v74
	v_accvgpr_read_b32 v74, a234
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[90:97], v[18:25], a[72:75], v133, v108 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mul_f32_e32 v151, s33, v75
	v_accvgpr_read_b32 v75, a233
	v_mul_f32_e32 v109, s33, v74
	v_mul_f32_e32 v152, s33, v75
	v_accvgpr_read_b32 v75, a235
	v_mul_f32_e32 v110, s33, v75
	v_accvgpr_read_b32 v58, a208
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[90:97], v[10:17], a[64:67], v133, v106 op_sel_hi:[0,0,0]
	v_accvgpr_read_b32 v59, a209
	v_mul_f32_e32 v80, s33, v58
	v_accvgpr_read_b32 v58, a210
	v_mul_f32_e32 v98, s33, v59
	v_accvgpr_read_b32 v59, a211
	v_mul_f32_e32 v78, s33, v58
	v_accvgpr_read_b32 v58, a200
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[90:97], v[2:9], a[60:63], v133, v106 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mul_f32_e32 v79, s33, v59
	v_accvgpr_read_b32 v59, a201
	v_mul_f32_e32 v76, s33, v58
	v_mul_f32_e32 v77, s33, v59
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[82:89], v[26:33], a[0:3], v133, v108 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[82:89], v[18:25], a[52:55], v133, v108 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[90:97], v[26:33], a[80:83], v133, v108 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[90:97], v[18:25], a[72:75], v133, v108 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[90:97], v[10:17], a[36:39], v133, v106 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[90:97], v[2:9], a[60:63], v133, v106 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mul_f32_e32 v90, s33, v42
	v_mul_f32_e32 v91, s33, v43
	v_accvgpr_read_b32 v42, a16
	v_accvgpr_read_b32 v43, a17
	v_mul_f32_e32 v92, s33, v42
	v_mul_f32_e32 v93, s33, v43
	v_accvgpr_read_b32 v42, a18
	v_accvgpr_read_b32 v43, a19
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[82:89], v[18:25], a[0:3], v133, v108 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mul_f32_e32 v94, s33, v42
	v_mul_f32_e32 v95, s33, v43
	v_accvgpr_read_b32 v42, a128
	v_accvgpr_read_b32 v43, a129
	v_mul_f32_e32 v209, s33, v42
	v_mul_f32_e32 v96, s33, v43
	v_accvgpr_read_b32 v42, a130
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[82:89], v[10:17], a[48:51], v133, v106 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_accvgpr_read_b32 v43, a131
	v_mul_f32_e32 v210, s33, v42
	v_mul_f32_e32 v97, s33, v43
	v_accvgpr_read_b32 v42, a124
	v_accvgpr_read_b32 v43, a125
	v_mul_f32_e32 v211, s33, v42
	v_mul_f32_e32 v212, s33, v43
	v_accvgpr_read_b32 v42, a126
	v_accvgpr_read_b32 v43, a127
	v_mul_f32_e32 v213, s33, v42
	v_mul_f32_e32 v214, s33, v43
	v_accvgpr_read_b32 v42, a116
	v_accvgpr_read_b32 v43, a117
	v_mul_f32_e32 v215, s33, v42
	v_mul_f32_e32 v216, s33, v43
	v_accvgpr_read_b32 v42, a118
	v_accvgpr_read_b32 v43, a119
	v_mul_f32_e32 v217, s33, v42
	v_mul_f32_e32 v218, s33, v43
	v_accvgpr_read_b32 v42, a24
	v_accvgpr_read_b32 v43, a25
	v_mul_f32_e32 v219, s33, v42
	v_mul_f32_e32 v220, s33, v43
	v_accvgpr_read_b32 v42, a26
	v_accvgpr_read_b32 v43, a27
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[82:89], v[10:17], a[0:3], v133, v106 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mul_f32_e32 v221, s33, v42
	v_accvgpr_read_b32 v42, a4
	v_mul_f32_e32 v224, s33, v42
	v_accvgpr_read_b32 v42, a6
	v_mul_f32_e32 v222, s33, v43
	v_accvgpr_read_b32 v43, a5
	v_mul_f32_e32 v226, s33, v42
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[82:89], v[2:9], a[44:47], v133, v106 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_accvgpr_read_b32 v42, a68
	v_mul_f32_e32 v223, s33, v43
	v_accvgpr_read_b32 v43, a7
	v_mul_f32_e32 v227, s33, v42
	v_accvgpr_read_b32 v42, a70
	v_mul_f32_e32 v225, s33, v43
	v_accvgpr_read_b32 v43, a69
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[82:89], v[2:9], a[0:3], v133, v106 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mul_f32_e32 v74, s33, v42
	v_accvgpr_read_b32 v42, a100
	v_mul_f32_e32 v228, s33, v43
	v_accvgpr_read_b32 v43, a71
	v_mul_f32_e32 v84, s33, v42
	v_accvgpr_read_b32 v42, a102
	v_mul_f32_e32 v75, s33, v43
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[50:57], v[26:33], a[40:43], v107, v108 op_sel_hi:[0,0,0]
	v_accvgpr_read_b32 v43, a101
	v_mul_f32_e32 v86, s33, v42
	v_accvgpr_read_b32 v42, a8
	v_mul_f32_e32 v85, s33, v43
	v_accvgpr_read_b32 v43, a103
	v_mul_f32_e32 v88, s33, v42
	v_accvgpr_read_b32 v42, a10
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[50:57], v[26:33], a[0:3], v107, v108 op_sel_hi:[1,1,0]
	v_mul_f32_e32 v87, s33, v43
	v_accvgpr_read_b32 v43, a9
	v_mul_f32_e32 v133, s33, v42
	v_accvgpr_read_b32 v42, a92
	v_mul_f32_e32 v89, s33, v43
	v_accvgpr_read_b32 v43, a11
	v_mul_f32_e32 v232, s33, v42
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[50:57], v[18:25], a[28:31], v107, v108 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_accvgpr_read_b32 v42, a94
	v_mul_f32_e32 v229, s33, v43
	v_accvgpr_read_b32 v43, a93
	v_mul_f32_e32 v233, s33, v42
	v_accvgpr_read_b32 v42, a12
	v_mul_f32_e32 v230, s33, v43
	v_accvgpr_read_b32 v43, a95
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[50:57], v[18:25], a[0:3], v107, v108 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mul_f32_e32 v234, s33, v42
	v_accvgpr_read_b32 v42, a14
	v_mul_f32_e32 v231, s33, v43
	v_accvgpr_read_b32 v43, a13
	v_mul_f32_e32 v236, s33, v42
	v_accvgpr_read_b32 v42, a84
	v_mul_f32_e32 v235, s33, v43
	v_accvgpr_write_b32 a0, v244
	v_accvgpr_write_b32 a1, v245
	v_accvgpr_write_b32 a2, v246
	v_accvgpr_write_b32 a3, v247
	v_accvgpr_read_b32 v43, a15
	v_mul_f32_e32 v238, s33, v42
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[50:57], v[10:17], a[0:3], v107, v106 op_sel_hi:[0,0,0]
	v_accvgpr_read_b32 v42, a86
	v_accvgpr_write_b32 a4, v248
	v_accvgpr_write_b32 a5, v249
	v_accvgpr_write_b32 a6, v250
	v_accvgpr_write_b32 a7, v251
	v_mul_f32_e32 v237, s33, v43
	v_accvgpr_read_b32 v43, a85
	v_accvgpr_write_b32 a0, v240
	v_accvgpr_write_b32 a1, v241
	v_accvgpr_write_b32 a2, v242
	v_accvgpr_write_b32 a3, v243
	v_mul_f32_e32 v240, s33, v42
	v_accvgpr_read_b32 v42, a76
	v_mul_f32_e32 v242, s33, v42
	v_accvgpr_read_b32 v42, a78
	v_mul_f32_e32 v244, s33, v42
	scratch_load_dword v42, off, off        ; 4-byte Folded Reload
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[34:41], v[26:33], a[0:3], v107, v108 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mul_f32_e32 v239, s33, v43
	v_accvgpr_read_b32 v43, a87
	v_mul_f32_e32 v241, s33, v43
	v_accvgpr_read_b32 v43, a77
	v_mul_f32_e32 v243, s33, v43
	v_lshl_or_b32 v43, s53, 2, v1
	v_mul_lo_u32 v1, v43, s52
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[34:41], v[18:25], a[4:7], v107, v108 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[34:41], v[26:33], a[0:3], v107, v108 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_accvgpr_read_b32 v26, a79
	v_mul_f32_e32 v245, s33, v26
	v_accvgpr_read_b32 v26, a80
	v_accvgpr_read_b32 v27, a81
	v_mul_f32_e32 v246, s33, v26
	v_mul_f32_e32 v247, s33, v27
	v_accvgpr_read_b32 v26, a82
	v_accvgpr_read_b32 v27, a83
	v_mul_f32_e32 v248, s33, v26
	v_mul_f32_e32 v249, s33, v27
	v_accvgpr_read_b32 v26, a20
	v_accvgpr_read_b32 v27, a21
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[34:41], v[18:25], a[4:7], v107, v108 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_accvgpr_read_b32 v18, a22
	v_accvgpr_read_b32 v19, a23
	v_mul_f32_e32 v252, s33, v18
	v_accvgpr_read_b32 v18, a36
	v_mul_f32_e32 v251, s33, v19
	v_accvgpr_read_b32 v19, a37
	v_mul_f32_e32 v253, s33, v18
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[50:57], v[10:17], a[12:15], v107, v106 op_sel_hi:[1,1,0]
	v_accvgpr_read_b32 v18, a38
	v_mul_f32_e32 v254, s33, v19
	v_accvgpr_read_b32 v19, a39
	v_mul_f32_e32 v255, s33, v18
	v_accvgpr_read_b32 v18, a60
	v_mul_f32_e32 v122, s33, v19
	v_accvgpr_read_b32 v19, a61
	v_accvgpr_write_b32 a12, v124
	v_accvgpr_write_b32 a13, v125
	v_accvgpr_write_b32 a14, v126
	v_accvgpr_write_b32 a15, v127
	v_mul_f32_e32 v82, s33, v18
	v_accvgpr_read_b32 v18, a62
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[50:57], v[2:9], a[12:15], v107, v106 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mul_f32_e32 v83, s33, v19
	v_accvgpr_read_b32 v19, a63
	v_mul_f32_e32 v62, s33, v18
	v_accvgpr_read_b32 v18, a56
	v_mul_f32_e32 v63, s33, v19
	v_accvgpr_read_b32 v19, a57
	v_mul_f32_e32 v60, s33, v18
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[50:57], v[2:9], a[12:15], v107, v106 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_accvgpr_read_b32 v18, a58
	v_mul_f32_e32 v61, s33, v19
	v_accvgpr_read_b32 v19, a59
	v_mul_f32_e32 v58, s33, v18
	v_accvgpr_read_b32 v18, a52
	v_mul_f32_e32 v59, s33, v19
	v_accvgpr_read_b32 v19, a53
	v_accvgpr_write_b32 a12, v128
	v_accvgpr_write_b32 a13, v129
	v_accvgpr_write_b32 a14, v130
	v_accvgpr_write_b32 a15, v131
	v_mul_f32_e32 v56, s33, v18
	v_accvgpr_read_b32 v18, a54
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[34:41], v[10:17], a[12:15], v107, v106 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mul_f32_e32 v54, s33, v19
	v_accvgpr_read_b32 v19, a55
	v_mul_f32_e32 v57, s33, v18
	v_mul_f32_e32 v55, s33, v19
	v_mul_f32_e32 v250, s33, v26
	v_mul_f32_e32 v108, s33, v27
	v_accvgpr_read_b32 v30, a19
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[34:41], v[10:17], a[12:15], v107, v106 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_accvgpr_read_b32 v10, a24
	v_accvgpr_read_b32 v11, a25
	v_mul_f32_e32 v52, s33, v10
	v_mul_f32_e32 v53, s33, v11
	v_accvgpr_read_b32 v10, a26
	v_accvgpr_read_b32 v11, a27
	v_accvgpr_write_b32 a24, v112
	v_accvgpr_write_b32 a25, v113
	v_accvgpr_write_b32 a26, v114
	v_accvgpr_write_b32 a27, v115
	v_mul_f32_e32 v50, s33, v10
	v_accvgpr_read_b32 v10, a44
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[34:41], v[2:9], a[24:27], v107, v106 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mul_f32_e32 v48, s33, v10
	v_accvgpr_read_b32 v10, a46
	v_mul_f32_e32 v46, s33, v10
	v_mul_f32_e32 v51, s33, v11
	v_accvgpr_read_b32 v11, a45
	v_mul_f32_e32 v49, s33, v11
	v_accvgpr_read_b32 v11, a47
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[34:41], v[2:9], a[24:27], v107, v106 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_accvgpr_read_b32 v2, a16
	v_accvgpr_read_b32 v3, a17
	v_mul_f32_e32 v44, s33, v2
	s_waitcnt vmcnt(0)
	v_lshl_add_u32 v2, v1, 6, v42
	v_lshrrev_b32_e32 v1, 2, v0
	v_mul_f32_e32 v45, s33, v3
	v_ashrrev_i32_e32 v3, 31, v2
	v_and_b32_e32 v1, 12, v1
	v_and_b32_e32 v0, 15, v0
	v_lshl_add_u64 v[32:33], v[2:3], 1, s[34:35]
	v_mad_u64_u32 v[2:3], s[0:1], v1, s52, v[0:1]
	v_add_u32_e32 v4, s52, v2
	v_add_u32_e32 v6, s52, v4
	v_add_u32_e32 v8, s52, v6
	s_mul_i32 s0, s52, 13
	v_add_u32_e32 v10, s0, v8
	v_add_u32_e32 v12, s52, v10
	v_add_u32_e32 v14, s52, v12
	v_add_u32_e32 v18, s52, v14
	v_mul_f32_e32 v47, s33, v11
	v_ashrrev_i32_e32 v3, 31, v2
	v_ashrrev_i32_e32 v5, 31, v4
	v_ashrrev_i32_e32 v7, 31, v6
	v_ashrrev_i32_e32 v9, 31, v8
	v_ashrrev_i32_e32 v11, 31, v10
	v_ashrrev_i32_e32 v13, 31, v12
	v_ashrrev_i32_e32 v15, 31, v14
	v_ashrrev_i32_e32 v19, 31, v18
	v_lshlrev_b64 v[34:35], 1, v[2:3]
	v_lshlrev_b64 v[2:3], 1, v[4:5]
	v_lshlrev_b64 v[4:5], 1, v[6:7]
	v_lshlrev_b64 v[6:7], 1, v[8:9]
	v_lshlrev_b64 v[8:9], 1, v[10:11]
	v_lshlrev_b64 v[10:11], 1, v[12:13]
	v_lshlrev_b64 v[12:13], 1, v[14:15]
	v_lshlrev_b64 v[14:15], 1, v[18:19]
	v_add_u32_e32 v18, s0, v18
	v_add_u32_e32 v20, s52, v18
	v_lshl_add_u64 v[36:37], v[32:33], 0, v[34:35]
	v_lshl_add_u64 v[38:39], v[32:33], 0, v[2:3]
	v_lshl_add_u64 v[40:41], v[32:33], 0, v[4:5]
	v_lshl_add_u64 v[106:107], v[32:33], 0, v[6:7]
	v_add_u32_e32 v22, s52, v20
	global_store_short_d16_hi v[36:37], a34, off
	global_store_short_d16_hi v[38:39], a35, off
	global_store_short_d16_hi v[40:41], a33, off
	global_store_short_d16_hi v[106:107], a32, off
	global_store_short_d16_hi v[36:37], a160, off offset:32
	global_store_short_d16_hi v[38:39], a161, off offset:32
	global_store_short_d16_hi v[40:41], a162, off offset:32
	global_store_short_d16_hi v[106:107], a163, off offset:32
	global_store_short_d16_hi v[36:37], a140, off offset:64
	global_store_short_d16_hi v[38:39], a141, off offset:64
	global_store_short_d16_hi v[40:41], a142, off offset:64
	global_store_short_d16_hi v[106:107], a143, off offset:64
	global_store_short_d16_hi v[36:37], a156, off offset:96
	global_store_short_d16_hi v[38:39], v118, off offset:96
	global_store_short_d16_hi v[40:41], v119, off offset:96
	global_store_short_d16_hi v[106:107], v117, off offset:96
	v_lshl_add_u64 v[118:119], v[32:33], 0, v[8:9]
	v_lshl_add_u64 v[0:1], v[32:33], 0, v[10:11]
	v_lshl_add_u64 v[16:17], v[32:33], 0, v[12:13]
	v_add_u32_e32 v26, s52, v22
	global_store_short_d16_hi v[118:119], v121, off
	global_store_short_d16_hi v[0:1], a145, off
	global_store_short_d16_hi v[16:17], v120, off
	v_lshl_add_u64 v[120:121], v[32:33], 0, v[14:15]
	v_ashrrev_i32_e32 v19, 31, v18
	v_ashrrev_i32_e32 v21, 31, v20
	v_ashrrev_i32_e32 v23, 31, v22
	v_ashrrev_i32_e32 v27, 31, v26
	global_store_short_d16_hi v[120:121], a144, off
	global_store_short_d16_hi v[118:119], a146, off offset:32
	global_store_short_d16_hi v[0:1], a147, off offset:32
	global_store_short_d16_hi v[16:17], a132, off offset:32
	global_store_short_d16_hi v[120:121], a133, off offset:32
	global_store_short_d16_hi v[118:119], v123, off offset:64
	global_store_short_d16_hi v[0:1], v116, off offset:64
	global_store_short_d16_hi v[16:17], v136, off offset:64
	global_store_short_d16_hi v[120:121], v137, off offset:64
	global_store_short_d16_hi v[118:119], v139, off offset:96
	global_store_short_d16_hi v[0:1], v153, off offset:96
	global_store_short_d16_hi v[16:17], v157, off offset:96
	global_store_short_d16_hi v[120:121], v158, off offset:96
	v_lshlrev_b64 v[116:117], 1, v[18:19]
	v_lshlrev_b64 v[18:19], 1, v[20:21]
	v_lshlrev_b64 v[20:21], 1, v[22:23]
	v_lshlrev_b64 v[22:23], 1, v[26:27]
	v_add_u32_e32 v26, s0, v26
	v_accvgpr_read_b32 v27, a18
	v_mul_f32_e32 v124, s33, v30
	v_add_u32_e32 v30, s52, v26
	v_lshl_add_u64 v[126:127], v[32:33], 0, v[116:117]
	v_lshl_add_u64 v[136:137], v[32:33], 0, v[18:19]
	v_lshl_add_u64 v[24:25], v[32:33], 0, v[20:21]
	v_lshl_add_u64 v[28:29], v[32:33], 0, v[22:23]
	v_mul_f32_e32 v123, s33, v27
	v_ashrrev_i32_e32 v27, 31, v26
	v_ashrrev_i32_e32 v31, 31, v30
	global_store_short_d16_hi v[126:127], v162, off
	global_store_short_d16_hi v[136:137], v154, off
	global_store_short_d16_hi v[24:25], v147, off
	global_store_short_d16_hi v[28:29], v138, off
	v_lshlrev_b64 v[138:139], 1, v[26:27]
	v_lshlrev_b64 v[26:27], 1, v[30:31]
	v_add_u32_e32 v30, s52, v30
	v_ashrrev_i32_e32 v31, 31, v30
	global_store_short_d16_hi v[126:127], v148, off offset:32
	global_store_short_d16_hi v[136:137], v140, off offset:32
	global_store_short_d16_hi v[24:25], v142, off offset:32
	global_store_short_d16_hi v[28:29], v143, off offset:32
	global_store_short_d16_hi v[126:127], v160, off offset:64
	global_store_short_d16_hi v[136:137], v161, off offset:64
	global_store_short_d16_hi v[24:25], v163, off offset:64
	global_store_short_d16_hi v[28:29], v178, off offset:64
	global_store_short_d16_hi v[126:127], v181, off offset:96
	global_store_short_d16_hi v[136:137], v182, off offset:96
	global_store_short_d16_hi v[24:25], v189, off offset:96
	global_store_short_d16_hi v[28:29], v190, off offset:96
	v_lshlrev_b64 v[142:143], 1, v[30:31]
	v_add_u32_e32 v30, s52, v30
	v_ashrrev_i32_e32 v31, 31, v30
	v_lshl_add_u64 v[160:161], v[32:33], 0, v[138:139]
	v_lshl_add_u64 v[162:163], v[32:33], 0, v[26:27]
	v_lshlrev_b64 v[30:31], 1, v[30:31]
	global_store_short_d16_hi v[160:161], v188, off
	global_store_short_d16_hi v[162:163], v183, off
	v_lshl_add_u64 v[182:183], v[32:33], 0, v[142:143]
	v_lshl_add_u64 v[188:189], v[32:33], 0, v[30:31]
	global_store_short_d16_hi v[182:183], v186, off
	global_store_short_d16_hi v[188:189], v179, off
	global_store_short_d16_hi v[160:161], v174, off offset:32
	global_store_short_d16_hi v[162:163], v167, off offset:32
	global_store_short_d16_hi v[182:183], v180, off offset:32
	global_store_short_d16_hi v[188:189], v170, off offset:32
	global_store_short_d16_hi v[160:161], v168, off offset:64
	global_store_short_d16_hi v[162:163], v175, off offset:64
	global_store_short_d16_hi v[182:183], v171, off offset:64
	global_store_short_d16_hi v[188:189], v164, off offset:64
	global_store_short_d16_hi v[160:161], v176, off offset:96
	global_store_short_d16_hi v[162:163], v169, off offset:96
	global_store_short_d16_hi v[182:183], v165, off offset:96
	global_store_short_d16_hi v[188:189], v159, off offset:96
	global_store_short_d16_hi v[36:37], v166, off offset:256
	v_accvgpr_read_b32 v36, a1
	v_mul_f32_e32 v166, s33, v36
	v_accvgpr_read_b32 v36, a2
	v_accvgpr_read_b32 v37, a3
	v_mul_f32_e32 v169, s33, v36
	v_mul_f32_e32 v170, s33, v37
	v_accvgpr_read_b32 v36, a4
	v_accvgpr_read_b32 v37, a5
	s_mov_b64 s[0:1], 0x100
	global_store_short_d16_hi v[38:39], v144, off offset:256
	v_mul_f32_e32 v144, s33, v36
	v_mul_f32_e32 v171, s33, v37
	v_accvgpr_read_b32 v36, a6
	v_accvgpr_read_b32 v37, a7
	v_lshl_add_u64 v[32:33], v[32:33], 0, s[0:1]
	global_store_short_d16_hi v[40:41], a105, off offset:256
	v_mul_f32_e32 v129, s33, v36
	v_mul_f32_e32 v174, s33, v37
	global_store_short_d16_hi v[106:107], a104, off offset:256
	v_lshl_add_u64 v[36:37], v[32:33], 0, v[34:35]
	v_lshl_add_u64 v[38:39], v[32:33], 0, v[2:3]
	v_lshl_add_u64 v[40:41], v[32:33], 0, v[4:5]
	v_lshl_add_u64 v[106:107], v[32:33], 0, v[6:7]
	global_store_short_d16_hi v[36:37], a107, off offset:32
	global_store_short_d16_hi v[38:39], a106, off offset:32
	global_store_short_d16_hi v[40:41], a108, off offset:32
	global_store_short_d16_hi v[106:107], v145, off offset:32
	global_store_short_d16_hi v[36:37], v134, off offset:64
	global_store_short_d16_hi v[38:39], a109, off offset:64
	global_store_short_d16_hi v[40:41], v132, off offset:64
	global_store_short_d16_hi v[106:107], v146, off offset:64
	global_store_short_d16_hi v[36:37], v141, off offset:96
	v_accvgpr_read_b32 v36, a25
	v_mul_f32_e32 v141, s33, v36
	v_accvgpr_read_b32 v36, a26
	v_accvgpr_read_b32 v37, a27
	v_mul_f32_e32 v145, s33, v36
	v_mul_f32_e32 v146, s33, v37
	global_store_short_d16_hi v[38:39], a110, off offset:96
	global_store_short_d16_hi v[40:41], a96, off offset:96
	global_store_short_d16_hi v[106:107], a97, off offset:96
	global_store_short_d16_hi v[118:119], v155, off offset:256
	global_store_short_d16_hi v[0:1], v156, off offset:256
	global_store_short_d16_hi v[16:17], v172, off offset:256
	global_store_short_d16_hi v[120:121], v173, off offset:256
	v_lshl_add_u64 v[0:1], v[32:33], 0, v[8:9]
	v_lshl_add_u64 v[16:17], v[32:33], 0, v[10:11]
	v_lshl_add_u64 v[36:37], v[32:33], 0, v[12:13]
	v_lshl_add_u64 v[38:39], v[32:33], 0, v[14:15]
	global_store_short_d16_hi v[0:1], v177, off offset:32
	global_store_short_d16_hi v[16:17], v184, off offset:32
	global_store_short_d16_hi v[36:37], v187, off offset:32
	global_store_short_d16_hi v[38:39], v185, off offset:32
	global_store_short_d16_hi v[0:1], v149, off offset:64
	global_store_short_d16_hi v[16:17], v150, off offset:64
	global_store_short_d16_hi v[36:37], v135, off offset:64
	global_store_short_d16_hi v[38:39], v151, off offset:64
	global_store_short_d16_hi v[0:1], v111, off offset:96
	global_store_short_d16_hi v[16:17], v152, off offset:96
	global_store_short_d16_hi v[36:37], v109, off offset:96
	global_store_short_d16_hi v[38:39], v110, off offset:96
	global_store_short_d16_hi v[126:127], v104, off offset:256
	global_store_short_d16_hi v[136:137], v105, off offset:256
	global_store_short_d16_hi v[24:25], v103, off offset:256
	global_store_short_d16_hi v[28:29], v101, off offset:256
	v_lshl_add_u64 v[0:1], v[32:33], 0, v[116:117]
	v_lshl_add_u64 v[16:17], v[32:33], 0, v[18:19]
	v_lshl_add_u64 v[24:25], v[32:33], 0, v[20:21]
	v_lshl_add_u64 v[28:29], v[32:33], 0, v[22:23]
	global_store_short_d16_hi v[0:1], v100, off offset:32
	global_store_short_d16_hi v[16:17], v102, off offset:32
	global_store_short_d16_hi v[24:25], v81, off offset:32
	global_store_short_d16_hi v[28:29], v99, off offset:32
	global_store_short_d16_hi v[0:1], v80, off offset:64
	global_store_short_d16_hi v[16:17], v98, off offset:64
	global_store_short_d16_hi v[24:25], v78, off offset:64
	global_store_short_d16_hi v[28:29], v79, off offset:64
	global_store_short_d16_hi v[0:1], v76, off offset:96
	global_store_short_d16_hi v[16:17], v77, off offset:96
	global_store_short_d16_hi v[24:25], v70, off offset:96
	global_store_short_d16_hi v[28:29], v71, off offset:96
	global_store_short_d16_hi v[160:161], v68, off offset:256
	global_store_short_d16_hi v[162:163], v67, off offset:256
	global_store_short_d16_hi v[182:183], v66, off offset:256
	global_store_short_d16_hi v[188:189], v69, off offset:256
	v_lshl_add_u64 v[0:1], v[32:33], 0, v[138:139]
	v_lshl_add_u64 v[16:17], v[32:33], 0, v[26:27]
	v_lshl_add_u64 v[24:25], v[32:33], 0, v[142:143]
	v_lshl_add_u64 v[28:29], v[32:33], 0, v[30:31]
	global_store_short_d16_hi v[0:1], v72, off offset:32
	global_store_short_d16_hi v[16:17], v64, off offset:32
	global_store_short_d16_hi v[24:25], v73, off offset:32
	global_store_short_d16_hi v[28:29], v65, off offset:32
	global_store_short_d16_hi v[0:1], v191, off offset:64
	global_store_short_d16_hi v[16:17], v192, off offset:64
	global_store_short_d16_hi v[24:25], v193, off offset:64
	global_store_short_d16_hi v[28:29], v194, off offset:64
	global_store_short_d16_hi v[0:1], v195, off offset:96
	global_store_short_d16_hi v[16:17], v196, off offset:96
	global_store_short_d16_hi v[24:25], v197, off offset:96
	global_store_short_d16_hi v[28:29], v200, off offset:96
	v_or_b32_e32 v0, 2, v43
	v_mul_lo_u32 v0, v0, s52
	v_lshl_add_u32 v0, v0, 6, v42
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshl_add_u64 v[0:1], v[0:1], 1, s[34:35]
	v_lshl_add_u64 v[16:17], v[0:1], 0, v[34:35]
	v_lshl_add_u64 v[24:25], v[0:1], 0, v[2:3]
	v_lshl_add_u64 v[28:29], v[0:1], 0, v[4:5]
	v_lshl_add_u64 v[32:33], v[0:1], 0, v[6:7]
	v_lshl_add_u64 v[36:37], v[0:1], 0, v[8:9]
	v_lshl_add_u64 v[38:39], v[0:1], 0, v[10:11]
	v_lshl_add_u64 v[40:41], v[0:1], 0, v[12:13]
	v_lshl_add_u64 v[64:65], v[0:1], 0, v[14:15]
	v_lshl_add_u64 v[66:67], v[0:1], 0, v[116:117]
	v_lshl_add_u64 v[68:69], v[0:1], 0, v[18:19]
	v_lshl_add_u64 v[70:71], v[0:1], 0, v[20:21]
	v_lshl_add_u64 v[72:73], v[0:1], 0, v[22:23]
	global_store_short_d16_hi v[16:17], v199, off
	global_store_short_d16_hi v[24:25], v198, off
	global_store_short_d16_hi v[28:29], v202, off
	global_store_short_d16_hi v[32:33], v201, off
	global_store_short_d16_hi v[16:17], v203, off offset:32
	global_store_short_d16_hi v[24:25], v204, off offset:32
	global_store_short_d16_hi v[28:29], v205, off offset:32
	global_store_short_d16_hi v[32:33], v206, off offset:32
	global_store_short_d16_hi v[16:17], v207, off offset:64
	global_store_short_d16_hi v[24:25], v208, off offset:64
	global_store_short_d16_hi v[28:29], v90, off offset:64
	global_store_short_d16_hi v[32:33], v91, off offset:64
	global_store_short_d16_hi v[16:17], v92, off offset:96
	global_store_short_d16_hi v[24:25], v93, off offset:96
	global_store_short_d16_hi v[28:29], v94, off offset:96
	global_store_short_d16_hi v[32:33], v95, off offset:96
	global_store_short_d16_hi v[36:37], v209, off
	global_store_short_d16_hi v[38:39], v96, off
	global_store_short_d16_hi v[40:41], v210, off
	global_store_short_d16_hi v[64:65], v97, off
	global_store_short_d16_hi v[36:37], v211, off offset:32
	global_store_short_d16_hi v[38:39], v212, off offset:32
	global_store_short_d16_hi v[40:41], v213, off offset:32
	global_store_short_d16_hi v[64:65], v214, off offset:32
	global_store_short_d16_hi v[36:37], v215, off offset:64
	global_store_short_d16_hi v[38:39], v216, off offset:64
	global_store_short_d16_hi v[40:41], v217, off offset:64
	global_store_short_d16_hi v[64:65], v218, off offset:64
	global_store_short_d16_hi v[36:37], v219, off offset:96
	global_store_short_d16_hi v[38:39], v220, off offset:96
	global_store_short_d16_hi v[40:41], v221, off offset:96
	global_store_short_d16_hi v[64:65], v222, off offset:96
	global_store_short_d16_hi v[66:67], v224, off
	global_store_short_d16_hi v[68:69], v223, off
	global_store_short_d16_hi v[70:71], v226, off
	global_store_short_d16_hi v[72:73], v225, off
	global_store_short_d16_hi v[66:67], v227, off offset:32
	global_store_short_d16_hi v[68:69], v228, off offset:32
	global_store_short_d16_hi v[70:71], v74, off offset:32
	global_store_short_d16_hi v[72:73], v75, off offset:32
	global_store_short_d16_hi v[66:67], v84, off offset:64
	global_store_short_d16_hi v[68:69], v85, off offset:64
	global_store_short_d16_hi v[70:71], v86, off offset:64
	global_store_short_d16_hi v[72:73], v87, off offset:64
	global_store_short_d16_hi v[66:67], v88, off offset:96
	global_store_short_d16_hi v[68:69], v89, off offset:96
	global_store_short_d16_hi v[70:71], v133, off offset:96
	global_store_short_d16_hi v[72:73], v229, off offset:96
	v_lshl_add_u64 v[74:75], v[0:1], 0, v[138:139]
	v_lshl_add_u64 v[76:77], v[0:1], 0, v[26:27]
	v_lshl_add_u64 v[78:79], v[0:1], 0, v[142:143]
	v_lshl_add_u64 v[80:81], v[0:1], 0, v[30:31]
	v_lshl_add_u64 v[0:1], v[0:1], 0, s[0:1]
	global_store_short_d16_hi v[74:75], v232, off
	global_store_short_d16_hi v[76:77], v230, off
	global_store_short_d16_hi v[78:79], v233, off
	global_store_short_d16_hi v[80:81], v231, off
	global_store_short_d16_hi v[74:75], v234, off offset:32
	global_store_short_d16_hi v[76:77], v235, off offset:32
	global_store_short_d16_hi v[78:79], v236, off offset:32
	global_store_short_d16_hi v[80:81], v237, off offset:32
	global_store_short_d16_hi v[74:75], v238, off offset:64
	global_store_short_d16_hi v[76:77], v239, off offset:64
	global_store_short_d16_hi v[78:79], v240, off offset:64
	global_store_short_d16_hi v[80:81], v241, off offset:64
	global_store_short_d16_hi v[74:75], v242, off offset:96
	global_store_short_d16_hi v[76:77], v243, off offset:96
	global_store_short_d16_hi v[78:79], v244, off offset:96
	global_store_short_d16_hi v[80:81], v245, off offset:96
	global_store_short_d16_hi v[16:17], v246, off offset:256
	global_store_short_d16_hi v[24:25], v247, off offset:256
	global_store_short_d16_hi v[28:29], v248, off offset:256
	global_store_short_d16_hi v[32:33], v249, off offset:256
	v_lshl_add_u64 v[16:17], v[0:1], 0, v[34:35]
	v_lshl_add_u64 v[2:3], v[0:1], 0, v[2:3]
	v_lshl_add_u64 v[4:5], v[0:1], 0, v[4:5]
	v_lshl_add_u64 v[6:7], v[0:1], 0, v[6:7]
	v_accvgpr_read_b32 v128, a8
	v_accvgpr_read_b32 v140, a9
	v_accvgpr_read_b32 v147, a10
	v_accvgpr_read_b32 v148, a11
	global_store_short_d16_hi v[16:17], v250, off offset:32
	global_store_short_d16_hi v[2:3], v108, off offset:32
	global_store_short_d16_hi v[4:5], v252, off offset:32
	global_store_short_d16_hi v[6:7], v251, off offset:32
	global_store_short_d16_hi v[16:17], v253, off offset:64
	global_store_short_d16_hi v[2:3], v254, off offset:64
	global_store_short_d16_hi v[4:5], v255, off offset:64
	global_store_short_d16_hi v[6:7], v122, off offset:64
	global_store_short_d16_hi v[16:17], v82, off offset:96
	global_store_short_d16_hi v[2:3], v83, off offset:96
	global_store_short_d16_hi v[4:5], v62, off offset:96
	global_store_short_d16_hi v[6:7], v63, off offset:96
	global_store_short_d16_hi v[36:37], v60, off offset:256
	global_store_short_d16_hi v[38:39], v61, off offset:256
	global_store_short_d16_hi v[40:41], v58, off offset:256
	global_store_short_d16_hi v[64:65], v59, off offset:256
	v_lshl_add_u64 v[2:3], v[0:1], 0, v[8:9]
	v_lshl_add_u64 v[4:5], v[0:1], 0, v[10:11]
	v_lshl_add_u64 v[6:7], v[0:1], 0, v[12:13]
	v_lshl_add_u64 v[8:9], v[0:1], 0, v[14:15]
	v_mul_f32_e32 v128, s33, v128
	v_mul_f32_e32 v140, s33, v140
	v_mul_f32_e32 v147, s33, v147
	v_mul_f32_e32 v148, s33, v148
	v_accvgpr_read_b32 v153, a20
	v_accvgpr_read_b32 v154, a21
	v_accvgpr_read_b32 v157, a22
	v_accvgpr_read_b32 v158, a23
	v_accvgpr_read_b32 v164, a28
	v_accvgpr_read_b32 v167, a29
	v_accvgpr_read_b32 v168, a30
	v_accvgpr_read_b32 v165, a31
	v_accvgpr_read_b32 v159, a0
	global_store_short_d16_hi v[2:3], v56, off offset:32
	global_store_short_d16_hi v[4:5], v54, off offset:32
	global_store_short_d16_hi v[6:7], v57, off offset:32
	global_store_short_d16_hi v[8:9], v55, off offset:32
	global_store_short_d16_hi v[2:3], v52, off offset:64
	global_store_short_d16_hi v[4:5], v53, off offset:64
	global_store_short_d16_hi v[6:7], v50, off offset:64
	global_store_short_d16_hi v[8:9], v51, off offset:64
	global_store_short_d16_hi v[2:3], v48, off offset:96
	global_store_short_d16_hi v[4:5], v49, off offset:96
	global_store_short_d16_hi v[6:7], v46, off offset:96
	global_store_short_d16_hi v[8:9], v47, off offset:96
	global_store_short_d16_hi v[66:67], v44, off offset:256
	global_store_short_d16_hi v[68:69], v45, off offset:256
	global_store_short_d16_hi v[70:71], v123, off offset:256
	global_store_short_d16_hi v[72:73], v124, off offset:256
	v_lshl_add_u64 v[2:3], v[0:1], 0, v[116:117]
	v_lshl_add_u64 v[4:5], v[0:1], 0, v[18:19]
	v_lshl_add_u64 v[6:7], v[0:1], 0, v[20:21]
	v_lshl_add_u64 v[8:9], v[0:1], 0, v[22:23]
	v_mul_f32_e32 v153, s33, v153
	v_mul_f32_e32 v154, s33, v154
	v_mul_f32_e32 v157, s33, v157
	v_mul_f32_e32 v158, s33, v158
	v_mul_f32_e32 v164, s33, v164
	v_mul_f32_e32 v167, s33, v167
	v_mul_f32_e32 v168, s33, v168
	v_mul_f32_e32 v165, s33, v165
	v_mul_f32_e32 v159, s33, v159
	v_accvgpr_read_b32 v125, a12
	v_accvgpr_read_b32 v130, a13
	v_accvgpr_read_b32 v131, a14
	v_accvgpr_read_b32 v132, a15
	v_accvgpr_read_b32 v134, a24
	global_store_short_d16_hi v[2:3], v128, off offset:32
	global_store_short_d16_hi v[4:5], v140, off offset:32
	global_store_short_d16_hi v[6:7], v147, off offset:32
	global_store_short_d16_hi v[8:9], v148, off offset:32
	global_store_short_d16_hi v[2:3], v153, off offset:64
	global_store_short_d16_hi v[4:5], v154, off offset:64
	global_store_short_d16_hi v[6:7], v157, off offset:64
	global_store_short_d16_hi v[8:9], v158, off offset:64
	global_store_short_d16_hi v[2:3], v164, off offset:96
	global_store_short_d16_hi v[4:5], v167, off offset:96
	global_store_short_d16_hi v[6:7], v168, off offset:96
	global_store_short_d16_hi v[8:9], v165, off offset:96
	global_store_short_d16_hi v[74:75], v159, off offset:256
	global_store_short_d16_hi v[76:77], v166, off offset:256
	global_store_short_d16_hi v[78:79], v169, off offset:256
	global_store_short_d16_hi v[80:81], v170, off offset:256
	v_lshl_add_u64 v[2:3], v[0:1], 0, v[138:139]
	v_lshl_add_u64 v[4:5], v[0:1], 0, v[26:27]
	v_lshl_add_u64 v[6:7], v[0:1], 0, v[142:143]
	v_lshl_add_u64 v[0:1], v[0:1], 0, v[30:31]
	v_mul_f32_e32 v125, s33, v125
	v_mul_f32_e32 v130, s33, v130
	v_mul_f32_e32 v131, s33, v131
	v_mul_f32_e32 v132, s33, v132
	v_mul_f32_e32 v134, s33, v134
	global_store_short_d16_hi v[2:3], v144, off offset:32
	global_store_short_d16_hi v[4:5], v171, off offset:32
	global_store_short_d16_hi v[6:7], v129, off offset:32
	global_store_short_d16_hi v[0:1], v174, off offset:32
	global_store_short_d16_hi v[2:3], v125, off offset:64
	global_store_short_d16_hi v[4:5], v130, off offset:64
	global_store_short_d16_hi v[6:7], v131, off offset:64
	global_store_short_d16_hi v[0:1], v132, off offset:64
	global_store_short_d16_hi v[2:3], v134, off offset:96
	global_store_short_d16_hi v[4:5], v141, off offset:96
	global_store_short_d16_hi v[6:7], v145, off offset:96
	global_store_short_d16_hi v[0:1], v146, off offset:96
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z20mxfp8_rewrite_kernel15rewrite_globals
		.amdhsa_group_segment_fixed_size 131072
		.amdhsa_private_segment_fixed_size 8
		.amdhsa_kernarg_size 528
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
	.size	_Z20mxfp8_rewrite_kernel15rewrite_globals, .Lfunc_end0-_Z20mxfp8_rewrite_kernel15rewrite_globals
                                        ; -- End function
	.set _Z20mxfp8_rewrite_kernel15rewrite_globals.num_vgpr, 256
	.set _Z20mxfp8_rewrite_kernel15rewrite_globals.num_agpr, 256
	.set _Z20mxfp8_rewrite_kernel15rewrite_globals.numbered_sgpr, 66
	.set _Z20mxfp8_rewrite_kernel15rewrite_globals.private_seg_size, 8
	.set _Z20mxfp8_rewrite_kernel15rewrite_globals.uses_vcc, 1
	.set _Z20mxfp8_rewrite_kernel15rewrite_globals.uses_flat_scratch, 0
	.set _Z20mxfp8_rewrite_kernel15rewrite_globals.has_dyn_sized_stack, 0
	.set _Z20mxfp8_rewrite_kernel15rewrite_globals.has_recursion, 0
	.set _Z20mxfp8_rewrite_kernel15rewrite_globals.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 25524
; TotalNumSgprs: 72
; NumVgprs: 256
; NumAgprs: 256
; TotalNumVgprs: 512
; ScratchSize: 8
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
	.type	__hip_cuid_28d3e56c8328096f,@object ; @__hip_cuid_28d3e56c8328096f
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_28d3e56c8328096f
__hip_cuid_28d3e56c8328096f:
	.byte	0                               ; 0x0
	.size	__hip_cuid_28d3e56c8328096f, 1

	.ident	"clang version 20.0.0git (https://github.com/ROCm/llvm-project.git 27682a16360e33e37c4f3cc6adf9a620733f8fe1)"
	.ident	"AMD clang version 20.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.1.0 25425 1b0eada6b0ee93e2e694c8c146d23fca90bc11c5)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_28d3e56c8328096f
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     256
    .args:
      - .offset:         0
        .size:           272
        .value_kind:     by_value
      - .offset:         272
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         276
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         280
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         284
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         286
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         288
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         290
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         292
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         294
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         312
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         320
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         328
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         336
        .size:           2
        .value_kind:     hidden_grid_dims
    .group_segment_fixed_size: 131072
    .kernarg_segment_align: 8
    .kernarg_segment_size: 528
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _Z20mxfp8_rewrite_kernel15rewrite_globals
    .private_segment_fixed_size: 8
    .sgpr_count:     72
    .sgpr_spill_count: 0
    .symbol:         _Z20mxfp8_rewrite_kernel15rewrite_globals.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     512
    .vgpr_spill_count: 1
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
