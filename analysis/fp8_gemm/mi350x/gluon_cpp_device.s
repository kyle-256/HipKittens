	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.text
	.protected	_Z22mxfp4_gluon_cpp_kernel13gluon_globals ; -- Begin function _Z22mxfp4_gluon_cpp_kernel13gluon_globals
	.globl	_Z22mxfp4_gluon_cpp_kernel13gluon_globals
	.p2align	8
	.type	_Z22mxfp4_gluon_cpp_kernel13gluon_globals,@function
_Z22mxfp4_gluon_cpp_kernel13gluon_globals: ; @_Z22mxfp4_gluon_cpp_kernel13gluon_globals
; %bb.0:
	s_load_dword s70, s[4:5], 0x50
	s_load_dwordx2 s[80:81], s[4:5], 0x60
	s_load_dword s71, s[4:5], 0x80
	s_load_dwordx2 s[68:69], s[4:5], 0x90
	s_load_dword s82, s[4:5], 0xb0
	s_load_dwordx2 s[54:55], s[4:5], 0xc0
	s_load_dwordx2 s[66:67], s[4:5], 0xe0
	s_load_dword s64, s[4:5], 0xf0
	s_mov_b64 s[48:49], s[4:5]
	s_add_u32 s52, s48, 0xf8
	s_mov_b32 s33, s10
	s_mov_b32 s50, s9
	s_mov_b32 s51, s8
	s_mov_b64 s[34:35], s[6:7]
	s_mov_b64 s[38:39], s[0:1]
	s_addc_u32 s53, s49, 0
	v_mov_b32_e32 v42, v0
	s_mov_b64 s[4:5], s[38:39]
	s_mov_b64 s[6:7], s[2:3]
	s_mov_b64 s[8:9], s[52:53]
	s_mov_b64 s[10:11], s[34:35]
	s_mov_b32 s12, s51
	s_mov_b32 s13, s50
	s_mov_b32 s14, s33
	v_mov_b32_e32 v31, v0
	v_mov_b32_e32 v0, 0
	s_mov_b32 s32, 16
	s_mov_b64 s[36:37], s[2:3]
	s_getpc_b64 s[0:1]
	s_add_u32 s0, s0, __ockl_get_group_id@rel32@lo+4
	s_addc_u32 s1, s1, __ockl_get_group_id@rel32@hi+12
	s_mov_b32 s65, 0
	s_swappc_b64 s[30:31], s[0:1]
	v_mov_b32_e32 v41, v0
	v_ashrrev_i32_e32 v0, 31, v41
	v_lshrrev_b32_e32 v0, 27, v0
	v_add_u32_e32 v40, v41, v0
	s_mov_b64 s[4:5], s[38:39]
	s_mov_b64 s[6:7], s[36:37]
	s_mov_b64 s[8:9], s[52:53]
	s_mov_b64 s[10:11], s[34:35]
	s_mov_b32 s12, s51
	s_mov_b32 s13, s50
	s_mov_b32 s14, s33
	v_mov_b32_e32 v31, v42
	v_mov_b32_e32 v0, 0
	v_ashrrev_i32_e32 v124, 5, v40
	s_getpc_b64 s[0:1]
	s_add_u32 s0, s0, __ockl_get_local_id@rel32@lo+4
	s_addc_u32 s1, s1, __ockl_get_local_id@rel32@hi+12
	scratch_store_dword off, v42, off       ; 4-byte Folded Spill
	s_swappc_b64 s[30:31], s[0:1]
	v_lshrrev_b32_e32 v80, 7, v0
	v_lshlrev_b32_e32 v1, 8, v124
	v_lshl_add_u32 v2, v80, 6, v1
	v_ashrrev_i32_e32 v5, 5, v2
	v_bfe_u32 v82, v0, 6, 1
	s_mov_b32 s3, 0x110000
	v_add_u32_e32 v4, 0x80, v2
	v_ashrrev_i32_e32 v4, 5, v4
	s_mov_b32 s2, -1
	v_mul_lo_u32 v2, v5, s71
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], s[80:81], 0, v[2:3]
	s_lshl_b32 s28, s82, 2
	v_readfirstlane_b32 s1, v3
	v_readfirstlane_b32 s0, v2
	s_mov_b64 s[6:7], s[2:3]
	s_mov_b64 s[4:5], s[0:1]
	s_load_dwordx2 s[44:45], s[48:49], 0x0
	s_load_dword s63, s[48:49], 0x20
	s_movk_i32 s62, 0x70
	v_mul_lo_u32 v2, v4, s71
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], s[80:81], 0, v[2:3]
	s_waitcnt lgkmcnt(0)
	v_mul_lo_u32 v1, v1, s63
	v_lshl_add_u32 v68, s63, 7, v1
	v_ashrrev_i32_e32 v69, 31, v68
	v_readfirstlane_b32 s9, v3
	s_mov_b32 s5, s9
	v_readfirstlane_b32 s67, v1
	v_readfirstlane_b32 s8, v2
	s_mov_b32 s4, s8
	s_mov_b64 s[10:11], s[2:3]
	s_mov_b64 s[8:9], s[0:1]
	v_and_b32_e32 v92, 48, v0
	v_lshlrev_b32_e32 v89, 3, v0
	v_or_b32_e32 v2, 1, v5
	v_mul_lo_u32 v2, v2, s71
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], s[80:81], 0, v[2:3]
	v_lshlrev_b32_e32 v84, 13, v82
	v_or_b32_e32 v79, 0x10000, v84
	v_readfirstlane_b32 s13, v3
	s_mov_b32 s9, s13
	v_readfirstlane_b32 s12, v2
	s_mov_b32 s8, s12
	s_mov_b64 s[14:15], s[2:3]
	s_mov_b64 s[12:13], s[0:1]
	v_or_b32_e32 v94, 64, v92
	v_or_b32_e32 v2, 1, v4
	v_mul_lo_u32 v2, v2, s71
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], s[80:81], 0, v[2:3]
	v_accvgpr_write_b32 a195, 0
	v_readfirstlane_b32 s17, v3
	s_mov_b32 s13, s17
	v_readfirstlane_b32 s16, v2
	s_mov_b32 s12, s16
	s_mov_b64 s[18:19], s[2:3]
	s_mov_b64 s[16:17], s[0:1]
	v_and_b32_e32 v2, 0xffffffe0, v40
	v_sub_u32_e32 v81, v41, v2
	v_lshlrev_b32_e32 v6, 8, v81
	v_mul_lo_u32 v1, v6, s70
	v_lshl_add_u32 v70, s70, 7, v1
	v_ashrrev_i32_e32 v71, 31, v70
	v_lshl_or_b32 v2, v82, 6, v6
	v_ashrrev_i32_e32 v7, 5, v2
	v_mul_lo_u32 v2, v7, s82
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[4:5], s[68:69], 0, v[2:3]
	v_add_u32_e32 v2, s28, v2
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], s[68:69], 0, v[2:3]
	v_readfirstlane_b32 s71, v1
	v_lshlrev_b32_e32 v1, 7, v0
	v_and_b32_e32 v85, 0x780, v1
	v_or_b32_e32 v88, v84, v85
	v_or_b32_e32 v90, 0x1c000, v88
	v_readfirstlane_b32 s25, v3
	v_or_b32_e32 v87, 0x18000, v88
	v_or_b32_e32 v32, v85, v92
	v_or_b32_e32 v50, v32, v79
	v_readfirstlane_b32 s24, v2
	v_or_b32_e32 v2, 1, v7
	v_mul_lo_u32 v2, v2, s82
	v_ashrrev_i32_e32 v3, 31, v2
	v_readfirstlane_b32 s21, v5
	s_mov_b32 s17, s21
	v_readfirstlane_b32 s20, v4
	s_mov_b32 s16, s20
	s_mov_b64 s[22:23], s[2:3]
	s_mov_b64 s[20:21], s[0:1]
	s_mov_b32 s20, s24
	v_lshl_add_u64 v[4:5], s[68:69], 0, v[2:3]
	v_add_u32_e32 v2, s28, v2
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], s[68:69], 0, v[2:3]
	s_load_dwordx2 s[68:69], s[48:49], 0x30
	s_mov_b32 s21, s25
	s_mov_b64 s[26:27], s[2:3]
	s_mov_b64 s[24:25], s[0:1]
	v_readfirstlane_b32 s41, v3
	v_readfirstlane_b32 s40, v2
	v_lshlrev_b32_e32 v2, 4, v0
	v_and_b32_e32 v3, 0xc00, v2
	v_bitop3_b32 v2, v2, s62, v0 bitop3:0x48
	v_readfirstlane_b32 s30, v5
	s_mov_b32 s25, s30
	v_readfirstlane_b32 s52, v3
	s_add_i32 s60, s52, 0x18000
	s_add_i32 s61, s60, 0x4000
	v_readfirstlane_b32 s29, v4
	s_mov_b32 s24, s29
	s_mov_b64 s[30:31], s[2:3]
	s_mov_b64 s[28:29], s[0:1]
	s_mov_b32 s28, s40
	v_bfe_u32 v3, v0, 3, 5
	s_mov_b32 s29, s41
	v_mad_u64_u32 v[40:41], s[40:41], v3, s63, v[2:3]
	s_lshl_b32 s40, s63, 5
	s_nop 0
	v_add_u32_e32 v41, s40, v40
	v_add_u32_e32 v72, s40, v41
	v_add_u32_e32 v73, s40, v72
	v_mad_u64_u32 v[66:67], s[40:41], v3, s70, v[2:3]
	s_lshl_b32 s40, s70, 5
	s_nop 0
	v_add_u32_e32 v67, s40, v66
	v_add_u32_e32 v74, s40, v67
	v_add_u32_e32 v75, s40, v74
	s_mov_b64 s[42:43], s[2:3]
	s_mov_b64 s[40:41], s[0:1]
	s_mov_b32 s40, s44
	s_add_i32 s53, s52, 0x4000
	s_mov_b32 s41, s45
	s_mov_b64 s[46:47], s[2:3]
	s_mov_b64 s[44:45], s[0:1]
	s_waitcnt lgkmcnt(0)
	s_mov_b32 s44, s68
	s_mov_b32 s68, s67
	;;#ASMSTART
	;;#ASMEND
	v_lshlrev_b32_e32 v2, 2, v0
	v_and_b32_e32 v76, 0xfc, v2
	s_bitset1_b32 s67, 7
	s_mov_b32 s45, s69
	s_mov_b32 s69, s52
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s69
	s_add_i32 s69, s52, 0x1000
	s_add_i32 s56, s52, 0xc000
	s_mov_b32 s72, s56
	buffer_load_dwordx4 v40, s[40:43], s68 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s69
	s_add_i32 s69, s52, 0x2000
	s_add_i32 s57, s52, 0x8000
	buffer_load_dwordx4 v41, s[40:43], s68 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s69
	s_add_i32 s69, s52, 0x3000
	s_add_i32 s58, s52, 0x10000
	s_add_i32 s59, s58, 0x4000
	buffer_load_dwordx4 v72, s[40:43], s68 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s69
	s_mov_b32 s69, s57
	buffer_load_dwordx4 v73, s[40:43], s68 offen lds
	v_readfirstlane_b32 s68, v68
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s69
	s_add_i32 s69, s52, 0x9000
	v_accvgpr_write_b32 a194, 0
	v_accvgpr_write_b32 a193, 0
	v_accvgpr_write_b32 a192, 0
	buffer_load_dwordx4 v40, s[40:43], s68 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s69
	s_add_i32 s69, s52, 0xa000
	buffer_load_dwordx4 v41, s[40:43], s68 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s69
	s_add_i32 s69, s52, 0xb000
	buffer_load_dwordx4 v72, s[40:43], s68 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s69
	s_mov_b32 s69, s58
	buffer_load_dwordx4 v73, s[40:43], s68 offen lds
	s_mov_b32 s68, s71
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s69
	s_add_i32 s69, s58, 0x1000
	buffer_load_dwordx4 v66, s[44:47], s68 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s69
	s_add_i32 s69, s58, 0x2000
	buffer_load_dwordx4 v67, s[44:47], s68 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s69
	s_add_i32 s69, s58, 0x3000
	buffer_load_dwordx4 v74, s[44:47], s68 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s69
	s_mov_b32 s69, s60
	buffer_load_dwordx4 v75, s[44:47], s68 offen lds
	v_readfirstlane_b32 s68, v70
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s69
	s_add_i32 s69, s60, 0x1000
	v_accvgpr_write_b32 a199, 0
	v_accvgpr_write_b32 a198, 0
	v_accvgpr_write_b32 a197, 0
	buffer_load_dwordx4 v66, s[44:47], s68 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s69
	s_add_i32 s69, s60, 0x2000
	buffer_load_dwordx4 v67, s[44:47], s68 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s69
	s_add_i32 s69, s60, 0x3000
	buffer_load_dwordx4 v74, s[44:47], s68 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s69
	v_accvgpr_write_b32 a196, 0
	buffer_load_dwordx4 v75, s[44:47], s68 offen lds
	s_mov_b32 s68, s53
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s68
	s_add_i32 s68, s52, 0x5000
	buffer_load_dwordx4 v40, s[40:43], s67 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s68
	s_add_i32 s68, s52, 0x6000
	buffer_load_dwordx4 v41, s[40:43], s67 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s68
	s_add_i32 s68, s52, 0x7000
	buffer_load_dwordx4 v72, s[40:43], s67 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s68
	s_mov_b64 s[68:69], 0x80
	v_lshl_add_u64 v[2:3], v[68:69], 0, s[68:69]
	buffer_load_dwordx4 v73, s[40:43], s67 offen lds
	v_readfirstlane_b32 s67, v2
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s72
	s_add_i32 s72, s52, 0xd000
	v_lshl_add_u64 v[2:3], v[70:71], 0, s[68:69]
	s_mov_b32 s68, s61
	v_lshlrev_b32_e32 v69, 13, v80
	buffer_load_dwordx4 v40, s[40:43], s67 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s72
	s_add_i32 s72, s52, 0xe000
	buffer_load_dwordx4 v41, s[40:43], s67 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s72
	s_add_i32 s72, s52, 0xf000
	buffer_load_dwordx4 v72, s[40:43], s67 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s72
	v_or_b32_e32 v16, v32, v69
	buffer_load_dwordx4 v73, s[40:43], s67 offen lds
	s_or_b32 s67, s71, 0x80
	s_mov_b32 s71, s59
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s71
	s_add_i32 s71, s58, 0x5000
	buffer_load_dwordx4 v66, s[44:47], s67 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s71
	s_add_i32 s71, s58, 0x6000
	buffer_load_dwordx4 v67, s[44:47], s67 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s71
	s_add_i32 s71, s58, 0x7000
	buffer_load_dwordx4 v74, s[44:47], s67 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s71
	v_bitop3_b32 v12, v89, v16, s62 bitop3:0x6c
	buffer_load_dwordx4 v75, s[44:47], s67 offen lds
	v_readfirstlane_b32 s67, v2
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s68
	s_add_i32 s68, s60, 0x5000
	v_or_b32_e32 v16, 64, v16
	v_bitop3_b32 v28, v89, v16, s62 bitop3:0x6c
	v_or_b32_e32 v93, v85, v69
	buffer_load_dwordx4 v66, s[44:47], s67 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s68
	s_add_i32 s68, s60, 0x6000
	buffer_load_dwordx4 v67, s[44:47], s67 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s68
	s_add_i32 s68, s60, 0x7000
	buffer_load_dwordx4 v74, s[44:47], s67 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s68
	v_lshrrev_b32_e32 v32, 4, v50
	buffer_load_dwordx4 v75, s[44:47], s67 offen lds
	buffer_load_dword v102, v76, s[0:3], 0 offen
	buffer_load_dword v96, v76, s[4:7], 0 offen
	buffer_load_dword v103, v76, s[8:11], 0 offen
	buffer_load_dword v97, v76, s[12:15], 0 offen
	buffer_load_dword v100, v76, s[16:19], 0 offen
	buffer_load_dword v98, v76, s[20:23], 0 offen
	buffer_load_dword v101, v76, s[24:27], 0 offen
	buffer_load_dword v99, v76, s[28:31], 0 offen
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[0:3], v12 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[4:7], v12 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[8:11], v12 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[12:15], v12 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[16:19], v28 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[20:23], v28 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[24:27], v28 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[28:31], v28 offset:0x1800

	;;#ASMEND
	v_add_u32_e32 v78, 0x4000, v93
	v_bitop3_b32 v46, v32, v50, s62 bitop3:0x6c
	;;#ASMSTART
	ds_read_b128 v[32:35], v46 offset:0

	;;#ASMEND
	v_or_b32_e32 v77, v78, v92
	v_lshrrev_b32_e32 v83, 4, v78
	v_or_b32_e32 v78, v78, v94
	;;#ASMSTART
	ds_read_b128 v[36:39], v46 offset:0x800

	;;#ASMEND
	v_bitop3_b32 v77, v83, v77, s62 bitop3:0x6c
	v_bitop3_b32 v78, v83, v78, s62 bitop3:0x6c
	v_or_b32_e32 v83, v85, v79
	;;#ASMSTART
	ds_read_b128 v[42:45], v46 offset:0x1000

	;;#ASMEND
	v_add_u32_e32 v50, 64, v50
	v_add_u32_e32 v79, v83, v92
	;;#ASMSTART
	ds_read_b128 v[46:49], v46 offset:0x1800

	;;#ASMEND
	v_lshrrev_b32_e32 v51, 4, v50
	v_lshrrev_b32_e32 v86, 4, v79
	v_add_u32_e32 v83, v83, v94
	v_bitop3_b32 v62, v51, v50, s62 bitop3:0x6c
	;;#ASMSTART
	ds_read_b128 v[50:53], v62 offset:0

	;;#ASMEND
	v_bitop3_b32 v79, v86, v79, s62 bitop3:0x6c
	v_lshrrev_b32_e32 v86, 4, v83
	v_or_b32_e32 v85, 0x14000, v88
	;;#ASMSTART
	ds_read_b128 v[54:57], v62 offset:0x800

	;;#ASMEND
	v_or_b32_e32 v69, v93, v92
	v_or_b32_e32 v71, v93, v94
	v_bitop3_b32 v83, v86, v83, s62 bitop3:0x6c
	v_or_b32_e32 v84, v85, v92
	v_or_b32_e32 v85, v85, v94
	v_or_b32_e32 v86, v87, v92
	v_or_b32_e32 v88, v90, v92
	v_add_u32_e32 v91, 0x8000, v93
	;;#ASMSTART
	ds_read_b128 v[58:61], v62 offset:0x1000

	;;#ASMEND
	v_bitop3_b32 v69, v89, v69, s62 bitop3:0x6c
	v_or_b32_e32 v87, v87, v94
	v_bitop3_b32 v71, v89, v71, s62 bitop3:0x6c
	v_bitop3_b32 v84, v89, v84, s62 bitop3:0x6c
	v_or_b32_e32 v90, v90, v94
	v_bitop3_b32 v85, v89, v85, s62 bitop3:0x6c
	v_bitop3_b32 v86, v89, v86, s62 bitop3:0x6c
	v_bitop3_b32 v87, v89, v87, s62 bitop3:0x6c
	v_bitop3_b32 v88, v89, v88, s62 bitop3:0x6c
	v_bitop3_b32 v89, v89, v90, s62 bitop3:0x6c
	v_or_b32_e32 v90, v91, v92
	v_lshrrev_b32_e32 v95, 4, v91
	v_or_b32_e32 v91, v91, v94
	v_add_u32_e32 v93, 0xc000, v93
	;;#ASMSTART
	ds_read_b128 v[62:65], v62 offset:0x1800

	;;#ASMEND
	v_bitop3_b32 v90, v95, v90, s62 bitop3:0x6c
	v_bitop3_b32 v91, v95, v91, s62 bitop3:0x6c
	v_or_b32_e32 v92, v93, v92
	v_lshrrev_b32_e32 v95, 4, v93
	v_or_b32_e32 v93, v93, v94
	v_bitop3_b32 v92, v95, v92, s62 bitop3:0x6c
	v_bitop3_b32 v93, v95, v93, s62 bitop3:0x6c
	v_mul_lo_u32 v94, s63, v124
	v_mul_lo_u32 v95, v81, s70
	v_lshlrev_b32_e32 v94, 8, v94
	v_lshlrev_b32_e32 v95, 8, v95
	s_movk_i32 s62, 0x100
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
	v_accvgpr_write_b32 a127, 0
	v_accvgpr_write_b32 a126, 0
	v_accvgpr_write_b32 a125, 0
	v_accvgpr_write_b32 a124, 0
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
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
.LBB0_1:                                ; =>This Inner Loop Header: Depth=1
	s_min_u32 s67, s65, 29
	s_lshl_b32 s67, s67, 7
	s_and_b32 s63, s65, 1
	s_addk_i32 s67, 0x100
	s_bitcmp1_b32 s65, 0
	s_cselect_b32 s68, s53, s52
	s_cselect_b32 s70, s56, s57
	s_cselect_b32 s71, s59, s58
	s_cselect_b32 s72, s61, s60
	v_add_u32_e32 v104, s67, v94
	v_add_u32_e32 v105, s67, v68
	v_add_u32_e32 v106, s67, v95
	v_add_u32_e32 v107, s67, v70
	s_add_i32 s67, s65, 1
	s_mov_b32 s76, s68
	s_add_i32 s77, s68, 0x1000
	s_add_i32 s78, s68, 0x2000
	s_addk_i32 s68, 0x3000
	s_mov_b32 s79, s70
	s_add_i32 s80, s70, 0x1000
	s_add_i32 s81, s70, 0x2000
	s_addk_i32 s70, 0x3000
	s_mov_b32 s82, s71
	s_add_i32 s83, s71, 0x1000
	s_add_i32 s84, s71, 0x2000
	s_addk_i32 s71, 0x3000
	s_mov_b32 s85, s72
	s_add_i32 s86, s72, 0x1000
	s_add_i32 s87, s72, 0x2000
	s_addk_i32 s72, 0x3000
	s_cmp_lg_u32 s65, 31
	s_cselect_b32 s65, s62, 0x1f00
	s_cmp_eq_u32 s63, 0
	v_readfirstlane_b32 s69, v104
	s_cselect_b64 vcc, -1, 0
	v_readfirstlane_b32 s73, v105
	v_readfirstlane_b32 s74, v106
	v_readfirstlane_b32 s75, v107
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	v_cndmask_b32_e32 v104, v88, v86, vcc
	v_cndmask_b32_e32 v105, v89, v87, vcc
	v_cndmask_b32_e32 v138, v92, v90, vcc
	v_cndmask_b32_e32 v139, v93, v91, vcc
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
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	buffer_load_dword v125, v76, s[0:3], s65 offen
	buffer_load_dword v170, v76, s[4:7], s65 offen
	buffer_load_dword v171, v76, s[8:11], s65 offen
	buffer_load_dword v172, v76, s[12:15], s65 offen
	buffer_load_dword v173, v76, s[16:19], s65 offen
	buffer_load_dword v174, v76, s[20:23], s65 offen
	buffer_load_dword v175, v76, s[24:27], s65 offen
	buffer_load_dword v176, v76, s[28:31], s65 offen
	v_cndmask_b32_e32 v177, v69, v77, vcc
	v_cndmask_b32_e32 v178, v71, v78, vcc
	s_waitcnt vmcnt(9)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15],  v[0:3], v[32:35], a[12:15],  v102, v100 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[104:107], v104 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19],  v[0:3], v[36:39], a[16:19],  v102, v100 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[108:111], v104 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23],  v[0:3], v[42:45], a[20:23],  v102, v101 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[112:115], v104 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27],  v[0:3], v[46:49], a[24:27],  v102, v101 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[116:119], v104 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15],  v[16:19], v[50:53], a[12:15],  v102, v100 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[120:123], v105 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19],  v[16:19], v[54:57], a[16:19],  v102, v100 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[126:129], v105 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23],  v[16:19], v[58:61], a[20:23],  v102, v101 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[130:133], v105 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27],  v[16:19], v[62:65], a[24:27],  v102, v101 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[134:137], v105 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[4:7], v[32:35], a[28:31],  v102, v100 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[4:7], v[36:39], a[32:35],  v102, v100 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[4:7], v[42:45], a[36:39],  v102, v101 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[4:7], v[46:49], a[40:43],  v102, v101 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[20:23], v[50:53], a[28:31],  v102, v100 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[20:23], v[54:57], a[32:35],  v102, v100 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[20:23], v[58:61], a[36:39],  v102, v101 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[20:23], v[62:65], a[40:43],  v102, v101 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[8:11], v[32:35], a[44:47],  v103, v100 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[8:11], v[36:39], a[48:51],  v103, v100 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[8:11], v[42:45], a[52:55], v103, v101 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[8:11], v[46:49], a[56:59], v103, v101 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[24:27], v[50:53], a[44:47],  v103, v100 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[24:27], v[54:57], a[48:51],  v103, v100 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[24:27], v[58:61], a[52:55], v103, v101 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[24:27], v[62:65], a[56:59], v103, v101 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[12:15], v[32:35], a[60:63], v103, v100 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[12:15], v[36:39], a[64:67], v103, v100 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[12:15], v[42:45], a[68:71], v103, v101 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[12:15], v[46:49], a[72:75], v103, v101 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[28:31], v[50:53], a[60:63], v103, v100 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[28:31], v[54:57], a[64:67], v103, v100 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[28:31], v[58:61], a[68:71], v103, v101 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[28:31], v[62:65], a[72:75], v103, v101 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	v_cndmask_b32_e32 v179, v79, v84, vcc
	s_waitcnt vmcnt(8)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3],  v[0:3], v[104:107], a[0:3],  v102, v98 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[138:141], v138 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7],  v[0:3], v[108:111], a[4:7],  v102, v98 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[142:145], v138 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11],  v[0:3], v[112:115], a[8:11],  v102, v99 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[146:149], v138 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79],  v[0:3], v[116:119], a[76:79],  v102, v99 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[150:153], v138 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3],  v[16:19], v[120:123], a[0:3],  v102, v98 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[154:157], v139 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7],  v[16:19], v[126:129], a[4:7],  v102, v98 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[158:161], v139 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11],  v[16:19], v[130:133], a[8:11],  v102, v99 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[162:165], v139 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79],  v[16:19], v[134:137], a[76:79],  v102, v99 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[166:169], v139 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83],  v[4:7], v[104:107], a[80:83],  v102, v98 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87],  v[4:7], v[108:111], a[84:87],  v102, v98 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91],  v[4:7], v[112:115], a[88:91],  v102, v99 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95],  v[4:7], v[116:119], a[92:95],  v102, v99 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83],  v[20:23], v[120:123], a[80:83],  v102, v98 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87],  v[20:23], v[126:129], a[84:87],  v102, v98 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91],  v[20:23], v[130:133], a[88:91],  v102, v99 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95],  v[20:23], v[134:137], a[92:95],  v102, v99 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99],  v[8:11], v[104:107], a[96:99],  v103, v98 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103],  v[8:11], v[108:111], a[100:103],  v103, v98 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[8:11], v[112:115], a[104:107], v103, v99 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[8:11], v[116:119], a[108:111], v103, v99 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99],  v[24:27], v[120:123], a[96:99],  v103, v98 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103],  v[24:27], v[126:129], a[100:103],  v103, v98 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[24:27], v[130:133], a[104:107], v103, v99 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[24:27], v[134:137], a[108:111], v103, v99 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[12:15], v[104:107], a[112:115], v103, v98 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[12:15], v[108:111], a[116:119], v103, v98 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[12:15], v[112:115], a[120:123], v103, v99 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[12:15], v[116:119], a[124:127], v103, v99 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[28:31], v[120:123], a[112:115], v103, v98 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[28:31], v[126:129], a[116:119], v103, v98 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[28:31], v[130:133], a[120:123], v103, v99 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[28:31], v[134:137], a[124:127], v103, v99 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[138:141], v[32:35], a[128:131],  v96, v100 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[0:3], v177 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[138:141], v[36:39], a[132:135],  v96, v100 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[4:7], v177 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[138:141], v[42:45], a[136:139],  v96, v101 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[8:11], v177 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[138:141], v[46:49], a[140:143],  v96, v101 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[12:15], v177 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131],  v[154:157], v[50:53], a[128:131],  v96, v100 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[16:19], v178 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[154:157], v[54:57], a[132:135],  v96, v100 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[20:23], v178 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139],  v[154:157], v[58:61], a[136:139],  v96, v101 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[24:27], v178 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143],  v[154:157], v[62:65], a[140:143],  v96, v101 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[28:31], v178 offset:6144

	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s76
	v_cndmask_b32_e32 v180, v83, v85, vcc
	buffer_load_dwordx4 v40, s[40:43], s69 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s77
	s_addk_i32 s62, 0x100
	buffer_load_dwordx4 v41, s[40:43], s69 offen lds
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[142:145], v[32:35], a[144:147],  v96, v100 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[142:145], v[36:39], a[148:151],  v96, v100 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[142:145], v[42:45], a[152:155],  v96, v101 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[142:145], v[46:49], a[156:159],  v96, v101 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[158:161], v[50:53], a[144:147],  v96, v100 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151],  v[158:161], v[54:57], a[148:151],  v96, v100 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155],  v[158:161], v[58:61], a[152:155],  v96, v101 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[158:161], v[62:65], a[156:159],  v96, v101 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s78
	s_mov_b32 s65, s67
	buffer_load_dwordx4 v72, s[40:43], s69 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s68
	s_cmp_eq_u32 s67, 32
	buffer_load_dwordx4 v73, s[40:43], s69 offen lds
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[146:149], v[32:35], a[160:163],  v97, v100 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[146:149], v[36:39], a[164:167],  v97, v100 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[146:149], v[42:45], a[168:171], v97, v101 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[146:149], v[46:49], a[172:175], v97, v101 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163],  v[162:165], v[50:53], a[160:163],  v97, v100 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167],  v[162:165], v[54:57], a[164:167],  v97, v100 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[162:165], v[58:61], a[168:171], v97, v101 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[162:165], v[62:65], a[172:175], v97, v101 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s79
	s_waitcnt vmcnt(11)
	v_mov_b32_e32 v102, v125
	buffer_load_dwordx4 v40, s[40:43], s73 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s80
	s_waitcnt vmcnt(10)
	v_mov_b32_e32 v103, v171
	buffer_load_dwordx4 v41, s[40:43], s73 offen lds
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[150:153], v[32:35], a[176:179], v97, v100 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[150:153], v[36:39], a[180:183], v97, v100 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[150:153], v[42:45], a[184:187], v97, v101 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[150:153], v[46:49], a[188:191], v97, v101 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[166:169], v[50:53], a[176:179], v97, v100 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[166:169], v[54:57], a[180:183], v97, v100 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[166:169], v[58:61], a[184:187], v97, v101 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[166:169], v[62:65], a[188:191], v97, v101 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s81
	s_waitcnt vmcnt(9)
	v_mov_b32_e32 v100, v173
	buffer_load_dwordx4 v72, s[40:43], s73 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s70
	s_waitcnt vmcnt(8)
	v_mov_b32_e32 v101, v175
	buffer_load_dwordx4 v73, s[40:43], s73 offen lds
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[138:141], v[104:107], a[192:195],  v96, v98 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[32:35], v179 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[138:141], v[108:111], a[196:199],  v96, v98 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[36:39], v179 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[138:141], v[112:115], a[200:203],  v96, v99 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v179 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[138:141], v[116:119], a[204:207],  v96, v99 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v179 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[154:157], v[120:123], a[192:195],  v96, v98 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v180 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199],  v[154:157], v[126:129], a[196:199],  v96, v98 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v180 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[154:157], v[130:133], a[200:203],  v96, v99 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v180 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207],  v[154:157], v[134:137], a[204:207],  v96, v99 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[62:65], v180 offset:6144

	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s82
	s_nop 0
	buffer_load_dwordx4 v66, s[44:47], s74 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s83
	s_nop 0
	buffer_load_dwordx4 v67, s[44:47], s74 offen lds
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[142:145], v[104:107], a[208:211],  v96, v98 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[142:145], v[108:111], a[212:215],  v96, v98 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[142:145], v[112:115], a[216:219],  v96, v99 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[142:145], v[116:119], a[220:223],  v96, v99 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[158:161], v[120:123], a[208:211],  v96, v98 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215],  v[158:161], v[126:129], a[212:215],  v96, v98 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[158:161], v[130:133], a[216:219],  v96, v99 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223],  v[158:161], v[134:137], a[220:223],  v96, v99 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s84
	s_nop 0
	buffer_load_dwordx4 v74, s[44:47], s74 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s71
	s_nop 0
	buffer_load_dwordx4 v75, s[44:47], s74 offen lds
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[146:149], v[104:107], a[224:227],  v97, v98 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[146:149], v[108:111], a[228:231],  v97, v98 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[146:149], v[112:115], a[232:235], v97, v99 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[146:149], v[116:119], a[236:239], v97, v99 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[162:165], v[120:123], a[224:227],  v97, v98 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231],  v[162:165], v[126:129], a[228:231],  v97, v98 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[162:165], v[130:133], a[232:235], v97, v99 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[162:165], v[134:137], a[236:239], v97, v99 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s85
	s_nop 0
	buffer_load_dwordx4 v66, s[44:47], s75 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s86
	s_nop 0
	buffer_load_dwordx4 v67, s[44:47], s75 offen lds
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[150:153], v[104:107], a[240:243], v97, v98 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[150:153], v[108:111], a[244:247], v97, v98 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[150:153], v[112:115], a[248:251], v97, v99 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[150:153], v[116:119], a[252:255], v97, v99 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[166:169], v[120:123], a[240:243], v97, v98 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[166:169], v[126:129], a[244:247], v97, v98 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[166:169], v[130:133], a[248:251], v97, v99 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[166:169], v[134:137], a[252:255], v97, v99 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s87
	v_mov_b32_e32 v96, v170
	buffer_load_dwordx4 v74, s[44:47], s75 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s72
	v_mov_b32_e32 v97, v172
	buffer_load_dwordx4 v75, s[44:47], s75 offen lds
	v_mov_b32_e32 v98, v174
	s_waitcnt vmcnt(16)
	v_mov_b32_e32 v99, v176
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_cbranch_scc0 .LBB0_1
; %bb.2:
	v_accvgpr_read_b32 v28, a44
	v_accvgpr_read_b32 v30, a46
	v_accvgpr_read_b32 v31, a47
	v_pk_mul_f32 v[44:45], v[30:31], s[64:65] op_sel_hi:[1,0]
	scratch_load_dword v31, off, off        ; 4-byte Folded Reload
	v_accvgpr_read_b32 v0, a72
	v_accvgpr_read_b32 v1, a73
	v_accvgpr_read_b32 v2, a74
	v_accvgpr_read_b32 v3, a75
	v_accvgpr_read_b32 v24, a48
	v_pk_mul_f32 v[0:1], v[0:1], s[64:65] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v25, a49
	v_accvgpr_read_b32 v26, a50
	v_accvgpr_read_b32 v27, a51
	v_pk_mul_f32 v[2:3], v[2:3], s[64:65] op_sel_hi:[1,0]
	v_accvgpr_write_b32 a49, v1
	v_accvgpr_read_b32 v29, a45
	v_accvgpr_write_b32 a45, v3
	v_accvgpr_write_b32 a48, v0
	v_lshl_add_u32 v0, v124, 2, v80
	v_accvgpr_read_b32 v4, a68
	v_accvgpr_read_b32 v8, a64
	v_accvgpr_read_b32 v12, a60
	v_accvgpr_read_b32 v51, a35
	v_accvgpr_write_b32 a44, v2
	v_lshl_or_b32 v1, v81, 2, v82
	v_mul_lo_u32 v2, v0, s66
	v_accvgpr_read_b32 v5, a69
	v_accvgpr_read_b32 v6, a70
	v_accvgpr_read_b32 v7, a71
	v_accvgpr_read_b32 v9, a65
	v_accvgpr_read_b32 v10, a66
	v_accvgpr_read_b32 v11, a67
	v_accvgpr_read_b32 v14, a62
	v_accvgpr_read_b32 v15, a63
	v_accvgpr_read_b32 v50, a34
	v_accvgpr_read_b32 v49, a33
	v_accvgpr_read_b32 v48, a32
	v_accvgpr_write_b32 a34, v0
	v_add_lshl_u32 v0, v1, v2, 6
	v_accvgpr_read_b32 v16, a56
	v_accvgpr_read_b32 v20, a52
	v_accvgpr_read_b32 v32, a40
	v_accvgpr_read_b32 v36, a36
	v_accvgpr_read_b32 v55, a31
	v_accvgpr_read_b32 v67, a27
	v_accvgpr_read_b32 v71, a23
	v_accvgpr_read_b32 v87, a19
	v_accvgpr_read_b32 v99, a15
	v_pk_mul_f32 v[14:15], v[14:15], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], s[64:65] op_sel_hi:[1,0]
	v_accvgpr_write_b32 a33, v1
	v_ashrrev_i32_e32 v1, 31, v0
	s_add_u32 s8, s48, 0xf8
	v_accvgpr_read_b32 v88, a172
	v_accvgpr_read_b32 v92, a168
	v_accvgpr_read_b32 v104, a164
	v_accvgpr_read_b32 v108, a160
	v_accvgpr_read_b32 v120, a156
	v_accvgpr_read_b32 v136, a152
	v_accvgpr_read_b32 v155, a151
	v_accvgpr_read_b32 v171, a147
	v_accvgpr_read_b32 v175, a143
	v_accvgpr_read_b32 v187, a139
	v_accvgpr_read_b32 v191, a135
	v_accvgpr_read_b32 v203, a131
	v_accvgpr_read_b32 v13, a61
	v_accvgpr_read_b32 v17, a57
	v_accvgpr_read_b32 v18, a58
	v_accvgpr_read_b32 v19, a59
	v_accvgpr_read_b32 v21, a53
	v_accvgpr_read_b32 v22, a54
	v_accvgpr_read_b32 v23, a55
	v_accvgpr_read_b32 v33, a41
	v_accvgpr_read_b32 v34, a42
	v_accvgpr_read_b32 v35, a43
	v_accvgpr_read_b32 v37, a37
	v_accvgpr_read_b32 v38, a38
	v_accvgpr_read_b32 v39, a39
	v_accvgpr_read_b32 v54, a30
	v_accvgpr_read_b32 v53, a29
	v_accvgpr_read_b32 v52, a28
	v_accvgpr_read_b32 v66, a26
	v_accvgpr_read_b32 v65, a25
	v_accvgpr_read_b32 v64, a24
	v_accvgpr_read_b32 v70, a22
	v_accvgpr_read_b32 v69, a21
	v_accvgpr_read_b32 v68, a20
	v_accvgpr_read_b32 v86, a18
	v_accvgpr_read_b32 v85, a17
	v_accvgpr_read_b32 v84, a16
	v_accvgpr_read_b32 v98, a14
	v_accvgpr_read_b32 v97, a13
	v_accvgpr_read_b32 v96, a12
	v_accvgpr_write_b32 a43, v15
	v_accvgpr_write_b32 a37, v11
	v_accvgpr_write_b32 a41, v9
	v_accvgpr_write_b32 a39, v7
	v_accvgpr_write_b32 a47, v5
	v_lshl_add_u64 v[78:79], v[0:1], 1, s[54:55]
	s_addc_u32 s9, s49, 0
	v_accvgpr_read_b32 v127, a11
	v_accvgpr_read_b32 v143, a7
	v_accvgpr_read_b32 v159, a3
	s_mov_b64 s[4:5], s[38:39]
	s_mov_b64 s[6:7], s[36:37]
	s_mov_b64 s[10:11], s[34:35]
	s_mov_b32 s12, s51
	s_mov_b32 s13, s50
	s_mov_b32 s14, s33
	v_mov_b32_e32 v0, 0
	v_accvgpr_read_b32 v89, a173
	v_accvgpr_read_b32 v90, a174
	v_accvgpr_read_b32 v91, a175
	v_accvgpr_read_b32 v93, a169
	v_accvgpr_read_b32 v94, a170
	v_accvgpr_read_b32 v95, a171
	v_accvgpr_read_b32 v105, a165
	v_accvgpr_read_b32 v106, a166
	v_accvgpr_read_b32 v107, a167
	v_accvgpr_read_b32 v109, a161
	v_accvgpr_read_b32 v110, a162
	v_accvgpr_read_b32 v111, a163
	v_accvgpr_read_b32 v121, a157
	v_accvgpr_read_b32 v122, a158
	v_accvgpr_read_b32 v123, a159
	v_accvgpr_read_b32 v137, a153
	v_accvgpr_read_b32 v138, a154
	v_accvgpr_read_b32 v139, a155
	v_accvgpr_read_b32 v154, a150
	v_accvgpr_read_b32 v153, a149
	v_accvgpr_read_b32 v152, a148
	v_accvgpr_read_b32 v170, a146
	v_accvgpr_read_b32 v169, a145
	v_accvgpr_read_b32 v168, a144
	v_accvgpr_read_b32 v174, a142
	v_accvgpr_read_b32 v173, a141
	v_accvgpr_read_b32 v172, a140
	v_accvgpr_read_b32 v186, a138
	v_accvgpr_read_b32 v185, a137
	v_accvgpr_read_b32 v184, a136
	v_accvgpr_read_b32 v190, a134
	v_accvgpr_read_b32 v189, a133
	v_accvgpr_read_b32 v188, a132
	v_accvgpr_read_b32 v202, a130
	v_accvgpr_read_b32 v201, a129
	v_accvgpr_read_b32 v200, a128
	v_pk_mul_f32 v[204:205], v[98:99], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[206:207], v[96:97], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[216:217], v[86:87], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[218:219], v[84:85], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[220:221], v[70:71], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[222:223], v[68:69], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[232:233], v[66:67], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[234:235], v[64:65], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[236:237], v[54:55], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[238:239], v[52:53], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[248:249], v[50:51], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[250:251], v[48:49], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[252:253], v[38:39], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[254:255], v[36:37], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[34:35], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[32:33], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[28:29], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[26:27], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[24:25], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[22:23], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[62:63], v[20:21], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[72:73], v[18:19], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[74:75], v[16:17], s[64:65] op_sel_hi:[1,0]
	v_accvgpr_write_b32 a42, v14
	v_pk_mul_f32 v[76:77], v[12:13], s[64:65] op_sel_hi:[1,0]
	v_accvgpr_write_b32 a36, v10
	v_accvgpr_write_b32 a40, v8
	v_accvgpr_write_b32 a38, v6
	v_accvgpr_write_b32 a46, v4
	v_accvgpr_write_b32 a32, v2
	s_getpc_b64 s[0:1]
	s_add_u32 s0, s0, __ockl_get_local_id@rel32@lo+4
	s_addc_u32 s1, s1, __ockl_get_local_id@rel32@hi+12
	v_accvgpr_read_b32 v126, a10
	v_accvgpr_read_b32 v125, a9
	v_accvgpr_read_b32 v124, a8
	v_accvgpr_read_b32 v142, a6
	v_accvgpr_read_b32 v141, a5
	v_accvgpr_read_b32 v140, a4
	v_accvgpr_read_b32 v158, a2
	v_accvgpr_read_b32 v157, a1
	v_accvgpr_read_b32 v156, a0
	s_swappc_b64 s[30:31], s[0:1]
	v_lshrrev_b32_e32 v1, 2, v0
	v_and_b32_e32 v1, 12, v1
	v_and_b32_e32 v0, 15, v0
	v_mad_u64_u32 v[2:3], s[0:1], v1, s66, v[0:1]
	v_add_u32_e32 v4, s66, v2
	v_add_u32_e32 v6, s66, v4
	v_add_u32_e32 v14, s66, v6
	v_ashrrev_i32_e32 v3, 31, v2
	v_ashrrev_i32_e32 v5, 31, v4
	v_ashrrev_i32_e32 v7, 31, v6
	v_ashrrev_i32_e32 v15, 31, v14
	v_lshlrev_b64 v[0:1], 1, v[2:3]
	v_lshlrev_b64 v[2:3], 1, v[4:5]
	v_lshlrev_b64 v[4:5], 1, v[6:7]
	v_lshlrev_b64 v[6:7], 1, v[14:15]
	v_lshl_add_u64 v[8:9], v[78:79], 0, v[0:1]
	v_lshl_add_u64 v[10:11], v[78:79], 0, v[2:3]
	v_lshl_add_u64 v[12:13], v[78:79], 0, v[4:5]
	v_lshl_add_u64 v[16:17], v[78:79], 0, v[6:7]
	s_mul_i32 s0, s66, 13
	global_store_short_d16_hi v[8:9], v206, off
	global_store_short_d16_hi v[10:11], v207, off
	global_store_short_d16_hi v[12:13], v204, off
	global_store_short_d16_hi v[16:17], v205, off
	global_store_short_d16_hi v[8:9], v218, off offset:32
	global_store_short_d16_hi v[10:11], v219, off offset:32
	global_store_short_d16_hi v[12:13], v216, off offset:32
	global_store_short_d16_hi v[16:17], v217, off offset:32
	global_store_short_d16_hi v[8:9], v222, off offset:64
	global_store_short_d16_hi v[10:11], v223, off offset:64
	global_store_short_d16_hi v[12:13], v220, off offset:64
	global_store_short_d16_hi v[16:17], v221, off offset:64
	global_store_short_d16_hi v[8:9], v234, off offset:96
	global_store_short_d16_hi v[10:11], v235, off offset:96
	global_store_short_d16_hi v[12:13], v232, off offset:96
	global_store_short_d16_hi v[16:17], v233, off offset:96
	v_add_u32_e32 v10, s0, v14
	v_add_u32_e32 v12, s66, v10
	v_add_u32_e32 v14, s66, v12
	v_add_u32_e32 v22, s66, v14
	v_ashrrev_i32_e32 v11, 31, v10
	v_ashrrev_i32_e32 v13, 31, v12
	v_ashrrev_i32_e32 v15, 31, v14
	v_ashrrev_i32_e32 v23, 31, v22
	v_lshlrev_b64 v[8:9], 1, v[10:11]
	v_lshlrev_b64 v[10:11], 1, v[12:13]
	v_lshlrev_b64 v[12:13], 1, v[14:15]
	v_lshlrev_b64 v[14:15], 1, v[22:23]
	v_lshl_add_u64 v[16:17], v[78:79], 0, v[8:9]
	v_lshl_add_u64 v[18:19], v[78:79], 0, v[10:11]
	v_lshl_add_u64 v[20:21], v[78:79], 0, v[12:13]
	v_lshl_add_u64 v[24:25], v[78:79], 0, v[14:15]
	global_store_short_d16_hi v[16:17], v238, off
	global_store_short_d16_hi v[18:19], v239, off
	global_store_short_d16_hi v[20:21], v236, off
	global_store_short_d16_hi v[24:25], v237, off
	global_store_short_d16_hi v[16:17], v250, off offset:32
	global_store_short_d16_hi v[18:19], v251, off offset:32
	global_store_short_d16_hi v[20:21], v248, off offset:32
	global_store_short_d16_hi v[24:25], v249, off offset:32
	global_store_short_d16_hi v[16:17], v254, off offset:64
	global_store_short_d16_hi v[18:19], v255, off offset:64
	global_store_short_d16_hi v[20:21], v252, off offset:64
	global_store_short_d16_hi v[24:25], v253, off offset:64
	global_store_short_d16_hi v[16:17], v42, off offset:96
	global_store_short_d16_hi v[18:19], v43, off offset:96
	global_store_short_d16_hi v[20:21], v40, off offset:96
	global_store_short_d16_hi v[24:25], v41, off offset:96
	v_add_u32_e32 v18, s0, v22
	v_add_u32_e32 v20, s66, v18
	v_add_u32_e32 v22, s66, v20
	v_add_u32_e32 v30, s66, v22
	v_ashrrev_i32_e32 v19, 31, v18
	v_ashrrev_i32_e32 v21, 31, v20
	v_ashrrev_i32_e32 v23, 31, v22
	v_ashrrev_i32_e32 v31, 31, v30
	v_lshlrev_b64 v[16:17], 1, v[18:19]
	v_lshlrev_b64 v[18:19], 1, v[20:21]
	v_lshlrev_b64 v[20:21], 1, v[22:23]
	v_lshlrev_b64 v[22:23], 1, v[30:31]
	v_lshl_add_u64 v[24:25], v[78:79], 0, v[16:17]
	v_lshl_add_u64 v[26:27], v[78:79], 0, v[18:19]
	v_lshl_add_u64 v[28:29], v[78:79], 0, v[20:21]
	v_lshl_add_u64 v[32:33], v[78:79], 0, v[22:23]
	global_store_short_d16_hi v[24:25], v46, off
	global_store_short_d16_hi v[26:27], v47, off
	global_store_short_d16_hi v[28:29], v44, off
	global_store_short_d16_hi v[32:33], v45, off
	global_store_short_d16_hi v[24:25], v58, off offset:32
	global_store_short_d16_hi v[26:27], v59, off offset:32
	global_store_short_d16_hi v[28:29], v56, off offset:32
	global_store_short_d16_hi v[32:33], v57, off offset:32
	global_store_short_d16_hi v[24:25], v62, off offset:64
	global_store_short_d16_hi v[26:27], v63, off offset:64
	global_store_short_d16_hi v[28:29], v60, off offset:64
	global_store_short_d16_hi v[32:33], v61, off offset:64
	global_store_short_d16_hi v[24:25], v74, off offset:96
	global_store_short_d16_hi v[26:27], v75, off offset:96
	global_store_short_d16_hi v[28:29], v72, off offset:96
	global_store_short_d16_hi v[32:33], v73, off offset:96
	v_add_u32_e32 v26, s0, v30
	v_add_u32_e32 v28, s66, v26
	v_add_u32_e32 v30, s66, v28
	v_ashrrev_i32_e32 v27, 31, v26
	v_ashrrev_i32_e32 v29, 31, v28
	v_ashrrev_i32_e32 v31, 31, v30
	v_lshlrev_b64 v[24:25], 1, v[26:27]
	v_lshlrev_b64 v[26:27], 1, v[28:29]
	v_lshlrev_b64 v[28:29], 1, v[30:31]
	v_add_u32_e32 v30, s66, v30
	v_ashrrev_i32_e32 v31, 31, v30
	v_accvgpr_read_b32 v40, a42
	v_lshlrev_b64 v[30:31], 1, v[30:31]
	v_lshl_add_u64 v[32:33], v[78:79], 0, v[24:25]
	v_lshl_add_u64 v[34:35], v[78:79], 0, v[26:27]
	v_lshl_add_u64 v[36:37], v[78:79], 0, v[28:29]
	v_accvgpr_read_b32 v41, a43
	v_lshl_add_u64 v[38:39], v[78:79], 0, v[30:31]
	global_store_short_d16_hi v[32:33], v76, off
	global_store_short_d16_hi v[34:35], v77, off
	global_store_short_d16_hi v[36:37], v40, off
	global_store_short_d16_hi v[38:39], v41, off
	v_accvgpr_read_b32 v40, a40
	v_accvgpr_read_b32 v41, a41
	global_store_short_d16_hi v[32:33], v40, off offset:32
	global_store_short_d16_hi v[34:35], v41, off offset:32
	v_accvgpr_read_b32 v41, a37
	v_pk_mul_f32 v[72:73], v[94:95], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[74:75], v[92:93], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[76:77], v[90:91], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[78:79], v[88:89], s[64:65] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v82, a176
	v_accvgpr_read_b32 v86, a180
	v_accvgpr_read_b32 v90, a184
	v_accvgpr_read_b32 v94, a188
	v_accvgpr_read_b32 v40, a36
	v_accvgpr_read_b32 v84, a178
	v_accvgpr_read_b32 v85, a179
	v_accvgpr_read_b32 v88, a182
	v_accvgpr_read_b32 v89, a183
	v_accvgpr_read_b32 v92, a186
	v_accvgpr_read_b32 v93, a187
	v_accvgpr_read_b32 v96, a190
	v_accvgpr_read_b32 v97, a191
	global_store_short_d16_hi v[36:37], v40, off offset:32
	global_store_short_d16_hi v[38:39], v41, off offset:32
	v_accvgpr_read_b32 v40, a46
	v_pk_mul_f32 v[80:81], v[84:85], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[84:85], v[88:89], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[88:89], v[92:93], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[92:93], v[96:97], s[64:65] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v96, a34
	v_accvgpr_read_b32 v41, a47
	v_add_u32_e32 v96, 2, v96
	global_store_short_d16_hi v[32:33], v40, off offset:64
	global_store_short_d16_hi v[34:35], v41, off offset:64
	v_accvgpr_read_b32 v41, a39
	v_pk_mul_f32 v[70:71], v[104:105], s[64:65] op_sel_hi:[1,0]
	v_mul_lo_u32 v104, v96, s66
	v_accvgpr_read_b32 v105, a33
	v_accvgpr_read_b32 v40, a38
	v_add_lshl_u32 v96, v104, v105, 6
	global_store_short_d16_hi v[36:37], v40, off offset:64
	global_store_short_d16_hi v[38:39], v41, off offset:64
	v_accvgpr_read_b32 v40, a48
	v_ashrrev_i32_e32 v97, 31, v96
	v_accvgpr_read_b32 v41, a49
	global_store_short_d16_hi v[32:33], v40, off offset:96
	global_store_short_d16_hi v[34:35], v41, off offset:96
	v_accvgpr_read_b32 v32, a44
	v_lshl_add_u64 v[96:97], v[96:97], 1, s[54:55]
	v_accvgpr_read_b32 v33, a45
	v_pk_mul_f32 v[34:35], v[200:201], s[64:65] op_sel_hi:[1,0]
	v_lshl_add_u64 v[98:99], v[96:97], 0, v[0:1]
	v_lshl_add_u64 v[100:101], v[96:97], 0, v[2:3]
	global_store_short_d16_hi v[36:37], v32, off offset:96
	global_store_short_d16_hi v[38:39], v33, off offset:96
	v_pk_mul_f32 v[32:33], v[202:203], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[190:191], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[188:189], s[64:65] op_sel_hi:[1,0]
	global_store_short_d16_hi v[98:99], v34, off
	global_store_short_d16_hi v[100:101], v35, off
	v_lshl_add_u64 v[34:35], v[96:97], 0, v[4:5]
	v_lshl_add_u64 v[102:103], v[96:97], 0, v[6:7]
	v_pk_mul_f32 v[40:41], v[186:187], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[184:185], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[174:175], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[172:173], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[170:171], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[168:169], s[64:65] op_sel_hi:[1,0]
	global_store_short_d16_hi v[34:35], v32, off
	global_store_short_d16_hi v[102:103], v33, off
	global_store_short_d16_hi v[98:99], v38, off offset:32
	global_store_short_d16_hi v[100:101], v39, off offset:32
	global_store_short_d16_hi v[34:35], v36, off offset:32
	global_store_short_d16_hi v[102:103], v37, off offset:32
	global_store_short_d16_hi v[98:99], v42, off offset:64
	global_store_short_d16_hi v[100:101], v43, off offset:64
	global_store_short_d16_hi v[34:35], v40, off offset:64
	global_store_short_d16_hi v[102:103], v41, off offset:64
	global_store_short_d16_hi v[98:99], v46, off offset:96
	global_store_short_d16_hi v[100:101], v47, off offset:96
	global_store_short_d16_hi v[34:35], v44, off offset:96
	global_store_short_d16_hi v[102:103], v45, off offset:96
	v_lshl_add_u64 v[32:33], v[96:97], 0, v[8:9]
	v_lshl_add_u64 v[34:35], v[96:97], 0, v[10:11]
	v_lshl_add_u64 v[36:37], v[96:97], 0, v[12:13]
	v_lshl_add_u64 v[38:39], v[96:97], 0, v[14:15]
	v_pk_mul_f32 v[52:53], v[154:155], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[152:153], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[138:139], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[136:137], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[122:123], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[62:63], v[120:121], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[64:65], v[110:111], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[66:67], v[108:109], s[64:65] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v83, a177
	v_accvgpr_read_b32 v87, a181
	v_accvgpr_read_b32 v91, a185
	v_accvgpr_read_b32 v95, a189
	global_store_short_d16_hi v[32:33], v50, off
	global_store_short_d16_hi v[34:35], v51, off
	global_store_short_d16_hi v[36:37], v48, off
	global_store_short_d16_hi v[38:39], v49, off
	global_store_short_d16_hi v[32:33], v54, off offset:32
	global_store_short_d16_hi v[34:35], v55, off offset:32
	global_store_short_d16_hi v[36:37], v52, off offset:32
	global_store_short_d16_hi v[38:39], v53, off offset:32
	global_store_short_d16_hi v[32:33], v58, off offset:64
	global_store_short_d16_hi v[34:35], v59, off offset:64
	global_store_short_d16_hi v[36:37], v56, off offset:64
	global_store_short_d16_hi v[38:39], v57, off offset:64
	global_store_short_d16_hi v[32:33], v62, off offset:96
	global_store_short_d16_hi v[34:35], v63, off offset:96
	global_store_short_d16_hi v[36:37], v60, off offset:96
	global_store_short_d16_hi v[38:39], v61, off offset:96
	v_lshl_add_u64 v[32:33], v[96:97], 0, v[16:17]
	v_lshl_add_u64 v[34:35], v[96:97], 0, v[18:19]
	v_lshl_add_u64 v[36:37], v[96:97], 0, v[20:21]
	v_lshl_add_u64 v[38:39], v[96:97], 0, v[22:23]
	v_pk_mul_f32 v[68:69], v[106:107], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[82:83], v[82:83], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[86:87], v[86:87], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[90:91], v[90:91], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[94:95], v[94:95], s[64:65] op_sel_hi:[1,0]
	global_store_short_d16_hi v[32:33], v66, off
	global_store_short_d16_hi v[34:35], v67, off
	global_store_short_d16_hi v[36:37], v64, off
	global_store_short_d16_hi v[38:39], v65, off
	global_store_short_d16_hi v[32:33], v70, off offset:32
	global_store_short_d16_hi v[34:35], v71, off offset:32
	global_store_short_d16_hi v[36:37], v68, off offset:32
	global_store_short_d16_hi v[38:39], v69, off offset:32
	global_store_short_d16_hi v[32:33], v74, off offset:64
	global_store_short_d16_hi v[34:35], v75, off offset:64
	global_store_short_d16_hi v[36:37], v72, off offset:64
	global_store_short_d16_hi v[38:39], v73, off offset:64
	global_store_short_d16_hi v[32:33], v78, off offset:96
	global_store_short_d16_hi v[34:35], v79, off offset:96
	global_store_short_d16_hi v[36:37], v76, off offset:96
	global_store_short_d16_hi v[38:39], v77, off offset:96
	v_lshl_add_u64 v[32:33], v[96:97], 0, v[24:25]
	v_lshl_add_u64 v[34:35], v[96:97], 0, v[26:27]
	v_lshl_add_u64 v[36:37], v[96:97], 0, v[28:29]
	v_lshl_add_u64 v[38:39], v[96:97], 0, v[30:31]
	global_store_short_d16_hi v[32:33], v82, off
	global_store_short_d16_hi v[34:35], v83, off
	global_store_short_d16_hi v[36:37], v80, off
	global_store_short_d16_hi v[38:39], v81, off
	global_store_short_d16_hi v[32:33], v86, off offset:32
	global_store_short_d16_hi v[34:35], v87, off offset:32
	global_store_short_d16_hi v[36:37], v84, off offset:32
	global_store_short_d16_hi v[38:39], v85, off offset:32
	global_store_short_d16_hi v[32:33], v90, off offset:64
	global_store_short_d16_hi v[34:35], v91, off offset:64
	global_store_short_d16_hi v[36:37], v88, off offset:64
	global_store_short_d16_hi v[38:39], v89, off offset:64
	global_store_short_d16_hi v[32:33], v94, off offset:96
	global_store_short_d16_hi v[34:35], v95, off offset:96
	global_store_short_d16_hi v[36:37], v92, off offset:96
	global_store_short_d16_hi v[38:39], v93, off offset:96
	v_accvgpr_read_b32 v46, a76
	v_accvgpr_read_b32 v50, a80
	v_accvgpr_read_b32 v54, a84
	v_accvgpr_read_b32 v58, a88
	v_accvgpr_read_b32 v62, a92
	v_accvgpr_read_b32 v66, a96
	v_accvgpr_read_b32 v70, a100
	v_accvgpr_read_b32 v74, a104
	v_accvgpr_read_b32 v78, a108
	v_accvgpr_read_b32 v82, a112
	v_accvgpr_read_b32 v86, a116
	v_accvgpr_read_b32 v90, a120
	v_accvgpr_read_b32 v94, a124
	v_accvgpr_read_b32 v48, a78
	v_accvgpr_read_b32 v49, a79
	v_accvgpr_read_b32 v52, a82
	v_accvgpr_read_b32 v53, a83
	v_accvgpr_read_b32 v56, a86
	v_accvgpr_read_b32 v57, a87
	v_accvgpr_read_b32 v60, a90
	v_accvgpr_read_b32 v61, a91
	v_accvgpr_read_b32 v64, a94
	v_accvgpr_read_b32 v65, a95
	v_accvgpr_read_b32 v68, a98
	v_accvgpr_read_b32 v69, a99
	v_accvgpr_read_b32 v72, a102
	v_accvgpr_read_b32 v73, a103
	v_accvgpr_read_b32 v76, a106
	v_accvgpr_read_b32 v77, a107
	v_accvgpr_read_b32 v80, a110
	v_accvgpr_read_b32 v81, a111
	v_accvgpr_read_b32 v84, a114
	v_accvgpr_read_b32 v85, a115
	v_accvgpr_read_b32 v88, a118
	v_accvgpr_read_b32 v89, a119
	v_accvgpr_read_b32 v92, a122
	v_accvgpr_read_b32 v93, a123
	v_accvgpr_read_b32 v96, a126
	v_accvgpr_read_b32 v97, a127
	v_pk_mul_f32 v[44:45], v[48:49], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[52:53], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[56:57], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[60:61], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[64:65], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[64:65], v[68:69], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[68:69], v[72:73], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[72:73], v[76:77], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[76:77], v[80:81], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[80:81], v[84:85], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[84:85], v[88:89], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[88:89], v[92:93], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[92:93], v[96:97], s[64:65] op_sel_hi:[1,0]
	v_or_b32_e32 v105, 2, v105
	v_accvgpr_read_b32 v96, a32
	v_add_lshl_u32 v96, v105, v96, 6
	v_ashrrev_i32_e32 v97, 31, v96
	v_lshl_add_u64 v[96:97], v[96:97], 1, s[54:55]
	v_pk_mul_f32 v[34:35], v[156:157], s[64:65] op_sel_hi:[1,0]
	v_lshl_add_u64 v[98:99], v[96:97], 0, v[0:1]
	v_lshl_add_u64 v[100:101], v[96:97], 0, v[2:3]
	v_pk_mul_f32 v[32:33], v[158:159], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[142:143], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[140:141], s[64:65] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v47, a77
	v_accvgpr_read_b32 v51, a81
	global_store_short_d16_hi v[98:99], v34, off
	global_store_short_d16_hi v[100:101], v35, off
	v_lshl_add_u64 v[34:35], v[96:97], 0, v[4:5]
	v_lshl_add_u64 v[102:103], v[96:97], 0, v[6:7]
	v_pk_mul_f32 v[40:41], v[126:127], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[124:125], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], s[64:65] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v55, a85
	v_accvgpr_read_b32 v59, a89
	v_accvgpr_read_b32 v63, a93
	v_accvgpr_read_b32 v67, a97
	global_store_short_d16_hi v[34:35], v32, off
	global_store_short_d16_hi v[102:103], v33, off
	global_store_short_d16_hi v[98:99], v38, off offset:32
	global_store_short_d16_hi v[100:101], v39, off offset:32
	global_store_short_d16_hi v[34:35], v36, off offset:32
	global_store_short_d16_hi v[102:103], v37, off offset:32
	global_store_short_d16_hi v[98:99], v42, off offset:64
	global_store_short_d16_hi v[100:101], v43, off offset:64
	global_store_short_d16_hi v[34:35], v40, off offset:64
	global_store_short_d16_hi v[102:103], v41, off offset:64
	global_store_short_d16_hi v[98:99], v46, off offset:96
	global_store_short_d16_hi v[100:101], v47, off offset:96
	global_store_short_d16_hi v[34:35], v44, off offset:96
	global_store_short_d16_hi v[102:103], v45, off offset:96
	v_lshl_add_u64 v[32:33], v[96:97], 0, v[8:9]
	v_lshl_add_u64 v[34:35], v[96:97], 0, v[10:11]
	v_lshl_add_u64 v[36:37], v[96:97], 0, v[12:13]
	v_lshl_add_u64 v[38:39], v[96:97], 0, v[14:15]
	v_pk_mul_f32 v[54:55], v[54:55], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[62:63], v[62:63], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[66:67], v[66:67], s[64:65] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v71, a101
	v_accvgpr_read_b32 v75, a105
	v_accvgpr_read_b32 v79, a109
	v_accvgpr_read_b32 v83, a113
	v_accvgpr_read_b32 v87, a117
	v_accvgpr_read_b32 v91, a121
	v_accvgpr_read_b32 v95, a125
	global_store_short_d16_hi v[32:33], v50, off
	global_store_short_d16_hi v[34:35], v51, off
	global_store_short_d16_hi v[36:37], v48, off
	global_store_short_d16_hi v[38:39], v49, off
	global_store_short_d16_hi v[32:33], v54, off offset:32
	global_store_short_d16_hi v[34:35], v55, off offset:32
	global_store_short_d16_hi v[36:37], v52, off offset:32
	global_store_short_d16_hi v[38:39], v53, off offset:32
	global_store_short_d16_hi v[32:33], v58, off offset:64
	global_store_short_d16_hi v[34:35], v59, off offset:64
	global_store_short_d16_hi v[36:37], v56, off offset:64
	global_store_short_d16_hi v[38:39], v57, off offset:64
	global_store_short_d16_hi v[32:33], v62, off offset:96
	global_store_short_d16_hi v[34:35], v63, off offset:96
	global_store_short_d16_hi v[36:37], v60, off offset:96
	global_store_short_d16_hi v[38:39], v61, off offset:96
	v_lshl_add_u64 v[32:33], v[96:97], 0, v[16:17]
	v_lshl_add_u64 v[34:35], v[96:97], 0, v[18:19]
	v_lshl_add_u64 v[36:37], v[96:97], 0, v[20:21]
	v_lshl_add_u64 v[38:39], v[96:97], 0, v[22:23]
	v_pk_mul_f32 v[70:71], v[70:71], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[74:75], v[74:75], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[78:79], v[78:79], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[82:83], v[82:83], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[86:87], v[86:87], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[90:91], v[90:91], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[94:95], v[94:95], s[64:65] op_sel_hi:[1,0]
	global_store_short_d16_hi v[32:33], v66, off
	global_store_short_d16_hi v[34:35], v67, off
	global_store_short_d16_hi v[36:37], v64, off
	global_store_short_d16_hi v[38:39], v65, off
	global_store_short_d16_hi v[32:33], v70, off offset:32
	global_store_short_d16_hi v[34:35], v71, off offset:32
	global_store_short_d16_hi v[36:37], v68, off offset:32
	global_store_short_d16_hi v[38:39], v69, off offset:32
	global_store_short_d16_hi v[32:33], v74, off offset:64
	global_store_short_d16_hi v[34:35], v75, off offset:64
	global_store_short_d16_hi v[36:37], v72, off offset:64
	global_store_short_d16_hi v[38:39], v73, off offset:64
	global_store_short_d16_hi v[32:33], v78, off offset:96
	global_store_short_d16_hi v[34:35], v79, off offset:96
	global_store_short_d16_hi v[36:37], v76, off offset:96
	global_store_short_d16_hi v[38:39], v77, off offset:96
	v_lshl_add_u64 v[32:33], v[96:97], 0, v[24:25]
	v_lshl_add_u64 v[34:35], v[96:97], 0, v[26:27]
	v_lshl_add_u64 v[36:37], v[96:97], 0, v[28:29]
	v_lshl_add_u64 v[38:39], v[96:97], 0, v[30:31]
	global_store_short_d16_hi v[32:33], v82, off
	global_store_short_d16_hi v[34:35], v83, off
	global_store_short_d16_hi v[36:37], v80, off
	global_store_short_d16_hi v[38:39], v81, off
	global_store_short_d16_hi v[32:33], v86, off offset:32
	global_store_short_d16_hi v[34:35], v87, off offset:32
	global_store_short_d16_hi v[36:37], v84, off offset:32
	global_store_short_d16_hi v[38:39], v85, off offset:32
	global_store_short_d16_hi v[32:33], v90, off offset:64
	global_store_short_d16_hi v[34:35], v91, off offset:64
	global_store_short_d16_hi v[36:37], v88, off offset:64
	global_store_short_d16_hi v[38:39], v89, off offset:64
	global_store_short_d16_hi v[32:33], v94, off offset:96
	global_store_short_d16_hi v[34:35], v95, off offset:96
	global_store_short_d16_hi v[36:37], v92, off offset:96
	global_store_short_d16_hi v[38:39], v93, off offset:96
	v_accvgpr_read_b32 v34, a192
	v_accvgpr_read_b32 v38, a196
	v_accvgpr_read_b32 v42, a200
	v_accvgpr_read_b32 v46, a204
	v_accvgpr_read_b32 v50, a208
	v_accvgpr_read_b32 v54, a212
	v_accvgpr_read_b32 v58, a216
	v_accvgpr_read_b32 v62, a220
	v_accvgpr_read_b32 v66, a224
	v_accvgpr_read_b32 v70, a228
	v_accvgpr_read_b32 v74, a232
	v_accvgpr_read_b32 v78, a236
	v_accvgpr_read_b32 v82, a240
	v_accvgpr_read_b32 v86, a244
	v_accvgpr_read_b32 v90, a248
	v_accvgpr_read_b32 v94, a252
	v_accvgpr_read_b32 v36, a194
	v_accvgpr_read_b32 v37, a195
	v_accvgpr_read_b32 v40, a198
	v_accvgpr_read_b32 v41, a199
	v_accvgpr_read_b32 v44, a202
	v_accvgpr_read_b32 v45, a203
	v_accvgpr_read_b32 v48, a206
	v_accvgpr_read_b32 v49, a207
	v_accvgpr_read_b32 v52, a210
	v_accvgpr_read_b32 v53, a211
	v_accvgpr_read_b32 v56, a214
	v_accvgpr_read_b32 v57, a215
	v_accvgpr_read_b32 v60, a218
	v_accvgpr_read_b32 v61, a219
	v_accvgpr_read_b32 v64, a222
	v_accvgpr_read_b32 v65, a223
	v_accvgpr_read_b32 v68, a226
	v_accvgpr_read_b32 v69, a227
	v_accvgpr_read_b32 v72, a230
	v_accvgpr_read_b32 v73, a231
	v_accvgpr_read_b32 v76, a234
	v_accvgpr_read_b32 v77, a235
	v_accvgpr_read_b32 v80, a238
	v_accvgpr_read_b32 v81, a239
	v_accvgpr_read_b32 v84, a242
	v_accvgpr_read_b32 v85, a243
	v_accvgpr_read_b32 v88, a246
	v_accvgpr_read_b32 v89, a247
	v_accvgpr_read_b32 v92, a250
	v_accvgpr_read_b32 v93, a251
	v_accvgpr_read_b32 v96, a254
	v_accvgpr_read_b32 v97, a255
	v_pk_mul_f32 v[32:33], v[36:37], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[40:41], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[44:45], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[48:49], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[52:53], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[56:57], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[60:61], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[64:65], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[64:65], v[68:69], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[68:69], v[72:73], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[72:73], v[76:77], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[76:77], v[80:81], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[80:81], v[84:85], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[84:85], v[88:89], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[88:89], v[92:93], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[92:93], v[96:97], s[64:65] op_sel_hi:[1,0]
	v_add_lshl_u32 v96, v104, v105, 6
	v_ashrrev_i32_e32 v97, 31, v96
	v_accvgpr_read_b32 v35, a193
	v_lshl_add_u64 v[96:97], v[96:97], 1, s[54:55]
	v_pk_mul_f32 v[34:35], v[34:35], s[64:65] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v39, a197
	v_accvgpr_read_b32 v43, a201
	v_accvgpr_read_b32 v47, a205
	v_accvgpr_read_b32 v51, a209
	v_lshl_add_u64 v[0:1], v[96:97], 0, v[0:1]
	v_lshl_add_u64 v[2:3], v[96:97], 0, v[2:3]
	v_lshl_add_u64 v[4:5], v[96:97], 0, v[4:5]
	v_lshl_add_u64 v[6:7], v[96:97], 0, v[6:7]
	v_pk_mul_f32 v[38:39], v[38:39], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], s[64:65] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v55, a213
	v_accvgpr_read_b32 v59, a217
	v_accvgpr_read_b32 v63, a221
	v_accvgpr_read_b32 v67, a225
	global_store_short_d16_hi v[0:1], v34, off
	global_store_short_d16_hi v[2:3], v35, off
	global_store_short_d16_hi v[4:5], v32, off
	global_store_short_d16_hi v[6:7], v33, off
	global_store_short_d16_hi v[0:1], v38, off offset:32
	global_store_short_d16_hi v[2:3], v39, off offset:32
	global_store_short_d16_hi v[4:5], v36, off offset:32
	global_store_short_d16_hi v[6:7], v37, off offset:32
	global_store_short_d16_hi v[0:1], v42, off offset:64
	global_store_short_d16_hi v[2:3], v43, off offset:64
	global_store_short_d16_hi v[4:5], v40, off offset:64
	global_store_short_d16_hi v[6:7], v41, off offset:64
	global_store_short_d16_hi v[0:1], v46, off offset:96
	global_store_short_d16_hi v[2:3], v47, off offset:96
	global_store_short_d16_hi v[4:5], v44, off offset:96
	global_store_short_d16_hi v[6:7], v45, off offset:96
	v_lshl_add_u64 v[0:1], v[96:97], 0, v[8:9]
	v_lshl_add_u64 v[2:3], v[96:97], 0, v[10:11]
	v_lshl_add_u64 v[4:5], v[96:97], 0, v[12:13]
	v_lshl_add_u64 v[6:7], v[96:97], 0, v[14:15]
	v_pk_mul_f32 v[54:55], v[54:55], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[62:63], v[62:63], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[66:67], v[66:67], s[64:65] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v71, a229
	v_accvgpr_read_b32 v75, a233
	v_accvgpr_read_b32 v79, a237
	v_accvgpr_read_b32 v83, a241
	global_store_short_d16_hi v[0:1], v50, off
	global_store_short_d16_hi v[2:3], v51, off
	global_store_short_d16_hi v[4:5], v48, off
	global_store_short_d16_hi v[6:7], v49, off
	global_store_short_d16_hi v[0:1], v54, off offset:32
	global_store_short_d16_hi v[2:3], v55, off offset:32
	global_store_short_d16_hi v[4:5], v52, off offset:32
	global_store_short_d16_hi v[6:7], v53, off offset:32
	global_store_short_d16_hi v[0:1], v58, off offset:64
	global_store_short_d16_hi v[2:3], v59, off offset:64
	global_store_short_d16_hi v[4:5], v56, off offset:64
	global_store_short_d16_hi v[6:7], v57, off offset:64
	global_store_short_d16_hi v[0:1], v62, off offset:96
	global_store_short_d16_hi v[2:3], v63, off offset:96
	global_store_short_d16_hi v[4:5], v60, off offset:96
	global_store_short_d16_hi v[6:7], v61, off offset:96
	v_lshl_add_u64 v[0:1], v[96:97], 0, v[16:17]
	v_lshl_add_u64 v[2:3], v[96:97], 0, v[18:19]
	v_lshl_add_u64 v[4:5], v[96:97], 0, v[20:21]
	v_lshl_add_u64 v[6:7], v[96:97], 0, v[22:23]
	v_pk_mul_f32 v[70:71], v[70:71], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[74:75], v[74:75], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[78:79], v[78:79], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[82:83], v[82:83], s[64:65] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v87, a245
	v_accvgpr_read_b32 v91, a249
	v_accvgpr_read_b32 v95, a253
	global_store_short_d16_hi v[0:1], v66, off
	global_store_short_d16_hi v[2:3], v67, off
	global_store_short_d16_hi v[4:5], v64, off
	global_store_short_d16_hi v[6:7], v65, off
	global_store_short_d16_hi v[0:1], v70, off offset:32
	global_store_short_d16_hi v[2:3], v71, off offset:32
	global_store_short_d16_hi v[4:5], v68, off offset:32
	global_store_short_d16_hi v[6:7], v69, off offset:32
	global_store_short_d16_hi v[0:1], v74, off offset:64
	global_store_short_d16_hi v[2:3], v75, off offset:64
	global_store_short_d16_hi v[4:5], v72, off offset:64
	global_store_short_d16_hi v[6:7], v73, off offset:64
	global_store_short_d16_hi v[0:1], v78, off offset:96
	global_store_short_d16_hi v[2:3], v79, off offset:96
	global_store_short_d16_hi v[4:5], v76, off offset:96
	global_store_short_d16_hi v[6:7], v77, off offset:96
	v_lshl_add_u64 v[0:1], v[96:97], 0, v[24:25]
	v_lshl_add_u64 v[2:3], v[96:97], 0, v[26:27]
	v_lshl_add_u64 v[4:5], v[96:97], 0, v[28:29]
	v_lshl_add_u64 v[6:7], v[96:97], 0, v[30:31]
	v_pk_mul_f32 v[86:87], v[86:87], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[90:91], v[90:91], s[64:65] op_sel_hi:[1,0]
	v_pk_mul_f32 v[94:95], v[94:95], s[64:65] op_sel_hi:[1,0]
	global_store_short_d16_hi v[0:1], v82, off
	global_store_short_d16_hi v[2:3], v83, off
	global_store_short_d16_hi v[4:5], v80, off
	global_store_short_d16_hi v[6:7], v81, off
	global_store_short_d16_hi v[0:1], v86, off offset:32
	global_store_short_d16_hi v[2:3], v87, off offset:32
	global_store_short_d16_hi v[4:5], v84, off offset:32
	global_store_short_d16_hi v[6:7], v85, off offset:32
	global_store_short_d16_hi v[0:1], v90, off offset:64
	global_store_short_d16_hi v[2:3], v91, off offset:64
	global_store_short_d16_hi v[4:5], v88, off offset:64
	global_store_short_d16_hi v[6:7], v89, off offset:64
	global_store_short_d16_hi v[0:1], v94, off offset:96
	global_store_short_d16_hi v[2:3], v95, off offset:96
	global_store_short_d16_hi v[4:5], v92, off offset:96
	global_store_short_d16_hi v[6:7], v93, off offset:96
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z22mxfp4_gluon_cpp_kernel13gluon_globals
		.amdhsa_group_segment_fixed_size 131072
		.amdhsa_private_segment_fixed_size 16
		.amdhsa_kernarg_size 504
		.amdhsa_user_sgpr_count 8
		.amdhsa_user_sgpr_dispatch_ptr 1
		.amdhsa_user_sgpr_queue_ptr 1
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 1
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 1
		.amdhsa_enable_private_segment 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 1
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 2
		.amdhsa_next_free_vgpr max(totalnumvgprs(_Z22mxfp4_gluon_cpp_kernel13gluon_globals.num_agpr, _Z22mxfp4_gluon_cpp_kernel13gluon_globals.num_vgpr), 1, 257)
		.amdhsa_next_free_sgpr (max(_Z22mxfp4_gluon_cpp_kernel13gluon_globals.numbered_sgpr+6, 1, 102))-6
		.amdhsa_accum_offset ((((((alignto(max(1, _Z22mxfp4_gluon_cpp_kernel13gluon_globals.num_vgpr), 4))/4)-1)&(~65536))&63)+1)*4
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
	.set _Z22mxfp4_gluon_cpp_kernel13gluon_globals.num_vgpr, max(256, amdgpu.max_num_vgpr)
	.set _Z22mxfp4_gluon_cpp_kernel13gluon_globals.num_agpr, max(256, amdgpu.max_num_agpr)
	.set _Z22mxfp4_gluon_cpp_kernel13gluon_globals.numbered_sgpr, max(88, amdgpu.max_num_sgpr)
	.set _Z22mxfp4_gluon_cpp_kernel13gluon_globals.private_seg_size, 16
	.set _Z22mxfp4_gluon_cpp_kernel13gluon_globals.uses_vcc, 1
	.set _Z22mxfp4_gluon_cpp_kernel13gluon_globals.uses_flat_scratch, 1
	.set _Z22mxfp4_gluon_cpp_kernel13gluon_globals.has_dyn_sized_stack, 1
	.set _Z22mxfp4_gluon_cpp_kernel13gluon_globals.has_recursion, 1
	.set _Z22mxfp4_gluon_cpp_kernel13gluon_globals.has_indirect_call, 1
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 14012
; TotalNumSgprs: _Z22mxfp4_gluon_cpp_kernel13gluon_globals.numbered_sgpr+6
; NumVgprs: _Z22mxfp4_gluon_cpp_kernel13gluon_globals.num_vgpr
; NumAgprs: _Z22mxfp4_gluon_cpp_kernel13gluon_globals.num_agpr
; TotalNumVgprs: totalnumvgprs(_Z22mxfp4_gluon_cpp_kernel13gluon_globals.num_agpr, _Z22mxfp4_gluon_cpp_kernel13gluon_globals.num_vgpr)
; ScratchSize: 16
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 131072 bytes/workgroup (compile time only)
; SGPRBlocks: ((alignto(max(max(_Z22mxfp4_gluon_cpp_kernel13gluon_globals.numbered_sgpr+(extrasgprs(_Z22mxfp4_gluon_cpp_kernel13gluon_globals.uses_vcc, _Z22mxfp4_gluon_cpp_kernel13gluon_globals.uses_flat_scratch, 1)), 1, 102), 1), 8))/8)-1
; VGPRBlocks: ((alignto(max(max(totalnumvgprs(_Z22mxfp4_gluon_cpp_kernel13gluon_globals.num_agpr, _Z22mxfp4_gluon_cpp_kernel13gluon_globals.num_vgpr), 1, 257), 1), 8))/8)-1
; NumSGPRsForWavesPerEU: max(_Z22mxfp4_gluon_cpp_kernel13gluon_globals.numbered_sgpr+6, 1, 102)
; NumVGPRsForWavesPerEU: max(totalnumvgprs(_Z22mxfp4_gluon_cpp_kernel13gluon_globals.num_agpr, _Z22mxfp4_gluon_cpp_kernel13gluon_globals.num_vgpr), 1, 257)
; AccumOffset: ((((alignto(max(1, _Z22mxfp4_gluon_cpp_kernel13gluon_globals.num_vgpr), 4))/4)-1)+1)*4
; Occupancy: occupancy(8, 8, 512, 8, 1, max(_Z22mxfp4_gluon_cpp_kernel13gluon_globals.numbered_sgpr+(extrasgprs(_Z22mxfp4_gluon_cpp_kernel13gluon_globals.uses_vcc, _Z22mxfp4_gluon_cpp_kernel13gluon_globals.uses_flat_scratch, 1)), 1, 102), max(totalnumvgprs(_Z22mxfp4_gluon_cpp_kernel13gluon_globals.num_agpr, _Z22mxfp4_gluon_cpp_kernel13gluon_globals.num_vgpr), 1, 257))
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 1
; COMPUTE_PGM_RSRC2:USER_SGPR: 8
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 2
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: ((((alignto(max(1, _Z22mxfp4_gluon_cpp_kernel13gluon_globals.num_vgpr), 4))/4)-1)&(~65536))&63
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.type	__hip_cuid_2cf8002b32632256,@object ; @__hip_cuid_2cf8002b32632256
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_2cf8002b32632256
__hip_cuid_2cf8002b32632256:
	.byte	0                               ; 0x0
	.size	__hip_cuid_2cf8002b32632256, 1

	.hidden	__ockl_get_group_id
	.hidden	__ockl_get_local_id
	.ident	"clang version 20.0.0git (https://github.com/ROCm/llvm-project.git 27682a16360e33e37c4f3cc6adf9a620733f8fe1)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_2cf8002b32632256
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
      - .offset:         328
        .size:           8
        .value_kind:     hidden_hostcall_buffer
      - .offset:         336
        .size:           8
        .value_kind:     hidden_multigrid_sync_arg
      - .offset:         344
        .size:           8
        .value_kind:     hidden_heap_v1
      - .offset:         352
        .size:           8
        .value_kind:     hidden_default_queue
      - .offset:         360
        .size:           8
        .value_kind:     hidden_completion_action
      - .offset:         448
        .size:           8
        .value_kind:     hidden_queue_ptr
    .group_segment_fixed_size: 131072
    .kernarg_segment_align: 8
    .kernarg_segment_size: 504
    .max_flat_workgroup_size: 256
    .name:           _Z22mxfp4_gluon_cpp_kernel13gluon_globals
    .private_segment_fixed_size: 16
    .sgpr_count:     94
    .sgpr_spill_count: 0
    .symbol:         _Z22mxfp4_gluon_cpp_kernel13gluon_globals.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: true
    .vgpr_count:     512
    .vgpr_spill_count: 1
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
