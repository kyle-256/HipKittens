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
	v_mov_b32_e32 v212, v0
	s_waitcnt lgkmcnt(0)
	s_ashr_i32 s5, s4, 31
	s_add_i32 s6, s4, 7
	s_lshr_b32 s3, s5, 29
	s_ashr_i32 s7, s6, 31
	s_add_i32 s3, s4, s3
	s_lshr_b32 s7, s7, 29
	s_and_b32 s3, s3, -8
	s_add_i32 s6, s6, s7
	s_sub_i32 s3, s4, s3
	s_ashr_i32 s6, s6, 3
	s_cmp_lg_u32 s3, 0
	s_cselect_b32 s8, s3, 8
	s_ashr_i32 s3, s2, 31
	s_lshr_b32 s3, s3, 29
	s_add_i32 s9, s2, s3
	s_and_b32 s3, s9, -8
	s_sub_i32 s7, s2, s3
	s_cmp_ge_i32 s7, s8
	s_cbranch_scc0 .LBB0_2
; %bb.1:
	s_mul_i32 s10, s8, s6
	s_sub_i32 s8, s7, s8
	s_add_i32 s11, s6, -1
	s_mul_i32 s8, s8, s11
	s_add_i32 s8, s8, s10
	s_ashr_i32 s2, s9, 3
	s_cbranch_execz .LBB0_3
	s_branch .LBB0_4
.LBB0_2:
                                        ; implicit-def: $sgpr8
	s_ashr_i32 s2, s9, 3
.LBB0_3:
	s_mul_i32 s8, s6, s7
.LBB0_4:
	s_add_i32 s2, s8, s2
	s_cmp_ge_i32 s2, s4
	s_cbranch_scc0 .LBB0_6
; %bb.5:
	s_endpgm
.LBB0_6:
	v_bfe_u32 v8, v212, 6, 1
	v_lshlrev_b32_e32 v1, 7, v212
	v_lshlrev_b32_e32 v0, 13, v8
	v_and_b32_e32 v67, 0x780, v1
	v_and_b32_e32 v70, 48, v212
	v_lshrrev_b32_e32 v3, 7, v212
	v_or_b32_e32 v66, 0x10000, v0
	v_or_b32_e32 v9, v67, v70
	s_load_dwordx2 s[6:7], s[0:1], 0x60
	v_or_b32_e32 v10, v9, v66
	v_lshlrev_b32_e32 v11, 13, v3
	v_or_b32_e32 v68, v0, v67
	v_or_b32_e32 v71, v67, v11
	s_movk_i32 s28, 0x70
	v_or_b32_e32 v72, 0x4000, v71
	v_add_u32_e32 v0, 64, v10
	v_lshrrev_b32_e32 v1, 4, v0
	v_bitop3_b32 v30, v1, v0, s28 bitop3:0x6c
	v_or_b32_e32 v0, v72, v70
	v_lshlrev_b32_e32 v73, 3, v212
	v_lshrrev_b32_e32 v2, 6, v212
	v_lshlrev_b32_e32 v2, 10, v2
	v_lshrrev_b32_e32 v12, 3, v212
	s_lshr_b32 s3, s5, 25
	v_or_b32_e32 v13, 0x60, v12
	v_readfirstlane_b32 s38, v2
	s_add_i32 s4, s4, s3
	s_load_dwordx2 s[30:31], s[0:1], 0x0
	v_bitop3_b32 v176, v73, v0, s28 bitop3:0x6c
	s_waitcnt lgkmcnt(0)
	v_mov_b64_e32 v[0:1], s[6:7]
	s_load_dwordx2 s[6:7], s[0:1], 0x20
	s_waitcnt lgkmcnt(0)
	s_ashr_i32 s7, s2, 31
	v_lshlrev_b32_e32 v2, 4, v212
	v_bitop3_b32 v2, v2, s28, v212 bitop3:0x48
	s_ashr_i32 s3, s4, 7
	v_mad_u64_u32 v[164:165], s[8:9], v13, s6, v[2:3]
	s_lshr_b32 s4, s7, 23
	s_add_i32 s8, s2, s4
	s_ashr_i32 s4, s8, 9
	s_lshl_b32 s9, s4, 2
	s_sub_i32 s3, s3, s9
	s_min_i32 s3, s3, 4
	s_abs_i32 s10, s3
	s_sub_i32 s5, 0, s10
	s_add_i32 s40, s38, 0x8000
	s_add_i32 s44, s38, 0x18000
	s_add_i32 s39, s38, 0x4000
	s_add_i32 s45, s44, 0x4000
	v_cvt_f32_u32_e32 v4, s10
	v_lshlrev_b32_e32 v197, 6, v3
	s_add_i32 s41, s40, 0x4000
	s_mov_b32 s20, 0
	v_rcp_iflag_f32_e32 v4, v4
	s_add_i32 s42, s38, 0x10000
	s_add_i32 s43, s42, 0x4000
	v_and_b32_e32 v177, 0x1f8, v73
	v_mul_f32_e32 v4, 0x4f7ffffe, v4
	v_cvt_u32_f32_e32 v4, v4
	s_mov_b32 s55, s40
	s_mov_b32 s36, s39
	s_mov_b32 s37, s41
	v_readfirstlane_b32 s4, v4
	s_mul_i32 s5, s5, s4
	s_mul_hi_u32 s5, s4, s5
	s_add_i32 s11, s4, s5
	s_abs_i32 s4, s2
	s_mul_hi_u32 s5, s4, s11
	s_mul_i32 s5, s5, s10
	s_sub_i32 s12, s4, s5
	s_sub_i32 s13, s12, s10
	s_cmp_ge_u32 s12, s10
	s_cselect_b32 s12, s13, s12
	s_sub_i32 s13, s12, s10
	s_cmp_ge_u32 s12, s10
	s_cselect_b32 s12, s13, s12
	s_xor_b32 s12, s12, s7
	s_sub_i32 s33, s12, s7
	s_add_i32 s33, s33, s9
	s_mov_b32 s29, s43
	s_and_b32 s7, s8, 0xfffffe00
	s_sub_i32 s2, s2, s7
	s_xor_b32 s3, s2, s3
	s_ashr_i32 s7, s3, 31
	s_load_dwordx2 s[4:5], s[0:1], 0x80
	s_waitcnt lgkmcnt(0)
	s_mov_b32 s5, s38
	s_abs_i32 s2, s2
	s_mul_hi_u32 s3, s2, s11
	s_add_i32 s9, s3, 1
	v_or_b32_e32 v69, 0x14000, v68
	v_or_b32_e32 v74, v69, v70
	v_bitop3_b32 v182, v73, v74, s28 bitop3:0x6c
	v_or_b32_e32 v74, 64, v70
	v_or_b32_e32 v69, v69, v74
	v_bitop3_b32 v185, v73, v69, s28 bitop3:0x6c
	s_mul_i32 s8, s3, s10
	s_sub_i32 s2, s2, s8
	s_sub_i32 s8, s2, s10
	s_cmp_ge_u32 s2, s10
	s_cselect_b32 s3, s9, s3
	s_mov_b32 s76, 7
	s_cselect_b32 s2, s8, s2
	s_add_i32 s8, s3, 1
	s_cmp_ge_u32 s2, s10
	s_cselect_b32 s8, s8, s3
	s_add_i32 s75, s44, 0x7000
	s_movk_i32 s77, 0x1000
	s_lshl_b32 s9, s33, 8
	s_mul_i32 s48, s6, s9
	s_mov_b32 s24, s48
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s5
	s_add_u32 s5, s30, s48
	s_add_u32 s49, s5, 0x100
	s_add_i32 s47, s38, 0x1000
	s_or_b32 s10, s9, 0x80
	s_mul_i32 s50, s6, s10
	s_mov_b32 s25, s50
	s_add_i32 s52, s38, 0x2000
	s_mov_b32 s27, s52
	s_add_u32 s5, s30, s50
	s_add_u32 s51, s5, 0x100
	s_lshl_b32 s5, s6, 5
	v_or_b32_e32 v3, s10, v197
	v_ashrrev_i32_e32 v3, 6, v3
	v_mad_i64_i32 v[4:5], s[2:3], s4, v3, v[0:1]
	s_nop 0
	v_readfirstlane_b32 s12, v4
	s_add_i32 s53, s38, 0x3000
	s_mov_b32 s34, s53
	v_or_b32_e32 v3, s9, v197
	v_ashrrev_i32_e32 v3, 6, v3
	v_mad_i64_i32 v[0:1], s[2:3], s4, v3, v[0:1]
	s_load_dwordx2 s[2:3], s[0:1], 0x90
	v_accvgpr_write_b32 a3, 0
	s_add_i32 s54, s40, 0x1000
	s_mov_b32 s35, s54
	v_readfirstlane_b32 s4, v0
	s_waitcnt lgkmcnt(0)
	v_mov_b64_e32 v[6:7], s[2:3]
	s_xor_b32 s2, s8, s7
	s_sub_i32 s2, s2, s7
	s_lshl_b32 s16, s2, 8
	s_or_b32 s26, s16, 0x80
	s_add_i32 s59, s42, 0x1000
	s_load_dwordx2 s[2:3], s[0:1], 0x50
	s_waitcnt lgkmcnt(0)
	s_mov_b32 s3, s47
	s_mov_b32 s7, 0x110000
	s_add_i32 s60, s42, 0x2000
	s_mov_b32 s63, s60
	s_mul_i32 s46, s2, s16
	s_mov_b32 s62, s46
	v_mad_u64_u32 v[166:167], s[8:9], v12, s6, v[2:3]
	v_add_u32_e32 v165, s5, v166
	v_add_u32_e32 v167, s5, v165
	v_readfirstlane_b32 s5, v1
	s_mul_i32 s57, s2, s26
	s_add_i32 s61, s42, 0x3000
	s_mov_b32 s65, s61
	v_lshlrev_b32_e32 v3, 6, v8
	v_or_b32_e32 v196, s16, v3
	v_ashrrev_i32_e32 v0, 6, v196
	s_mov_b32 s6, -1
	s_mov_b64 s[10:11], s[6:7]
	s_mov_b64 s[8:9], s[4:5]
	s_mov_b32 s8, s12
	s_mov_b64 s[14:15], s[6:7]
	s_mov_b64 s[12:13], s[4:5]
	s_load_dwordx2 s[16:17], s[0:1], 0xb0
	s_mov_b32 s64, s57
	v_readfirstlane_b32 s9, v5
	;;#ASMSTART
	buffer_load_dwordx2 v[130:131], v177, s[8:11], s20 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[136:137], v177, s[4:7], s20 offen
	;;#ASMEND
	s_waitcnt lgkmcnt(0)
	v_mad_i64_i32 v[0:1], s[12:13], s16, v0, v[6:7]
	s_nop 0
	v_readfirstlane_b32 s12, v0
	v_or_b32_e32 v0, s26, v3
	v_ashrrev_i32_e32 v0, 6, v0
	s_add_i32 s66, s38, 0x7000
	v_readfirstlane_b32 s13, v1
	;;#ASMSTART
	buffer_load_dwordx2 v[134:135], v177, s[12:15], s20 offen
	;;#ASMEND
	s_add_i32 s67, s40, 0x5000
	v_mad_i64_i32 v[0:1], s[16:17], s16, v0, v[6:7]
	s_nop 0
	v_readfirstlane_b32 s21, v0
	s_add_i32 s68, s40, 0x6000
	s_add_i32 s69, s40, 0x7000
	s_mov_b64 s[18:19], s[6:7]
	s_mov_b64 s[16:17], s[4:5]
	s_mov_b32 s16, s21
	s_add_i32 s70, s42, 0x5000
	v_readfirstlane_b32 s17, v1
	;;#ASMSTART
	buffer_load_dwordx2 v[132:133], v177, s[16:19], s20 offen
	;;#ASMEND
	s_mov_b64 s[22:23], s[6:7]
	s_mov_b64 s[20:21], s[4:5]
	s_mov_b32 s20, s30
	s_add_i32 s71, s42, 0x6000
	s_mov_b32 s21, s31
	buffer_load_dwordx4 v166, s[20:23], s24 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s3
	s_add_i32 s72, s42, 0x7000
	buffer_load_dwordx4 v165, s[20:23], s24 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s27
	s_add_i32 s31, s40, 0x2000
	s_mov_b32 s3, s31
	buffer_load_dwordx4 v167, s[20:23], s24 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s34
	s_add_i32 s73, s44, 0x5000
	buffer_load_dwordx4 v164, s[20:23], s24 offen lds
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s55
	s_add_i32 s55, s40, 0x3000
	s_add_i32 s74, s44, 0x6000
	buffer_load_dwordx4 v166, s[20:23], s25 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s35
	s_load_dwordx2 s[34:35], s[0:1], 0x30
	v_or_b32_e32 v0, v71, v70
	buffer_load_dwordx4 v165, s[20:23], s25 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s3
	s_mov_b32 s3, s55
	buffer_load_dwordx4 v167, s[20:23], s25 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s3
	s_mov_b32 s3, s42
	buffer_load_dwordx4 v164, s[20:23], s25 offen lds
	v_mad_u64_u32 v[168:169], s[24:25], v12, s2, v[2:3]
	s_mov_b64 s[26:27], s[6:7]
	s_mov_b64 s[24:25], s[4:5]
	s_waitcnt lgkmcnt(0)
	s_mov_b32 s24, s34
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s3
	s_add_u32 s3, s34, s46
	s_add_u32 s56, s3, 0x100
	s_add_u32 s3, s34, s57
	s_add_u32 s58, s3, 0x100
	s_mov_b32 s3, s59
	s_lshl_b32 s25, s2, 5
	v_add_u32_e32 v169, s25, v168
	v_add_u32_e32 v179, s25, v169
	s_mov_b32 s25, s35
	buffer_load_dwordx4 v168, s[24:27], s62 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s3
	v_mad_u64_u32 v[170:171], s[2:3], v13, s2, v[2:3]
	v_bitop3_b32 v171, v73, v0, s28 bitop3:0x6c
	v_or_b32_e32 v0, 0x8000, v71
	v_or_b32_e32 v1, v0, v70
	v_bitop3_b32 v180, v73, v1, s28 bitop3:0x6c
	v_or_b32_e32 v1, 0xc000, v71
	v_or_b32_e32 v2, v1, v70
	v_bitop3_b32 v181, v73, v2, s28 bitop3:0x6c
	v_or_b32_e32 v2, v9, v11
	v_or_b32_e32 v3, 64, v2
	v_bitop3_b32 v3, v73, v3, s28 bitop3:0x6c
	s_add_i32 s35, s44, 0x1000
	v_bitop3_b32 v2, v73, v2, s28 bitop3:0x6c
	v_or_b32_e32 v71, v71, v74
	v_bitop3_b32 v183, v73, v71, s28 bitop3:0x6c
	v_or_b32_e32 v71, v72, v74
	v_bitop3_b32 v184, v73, v71, s28 bitop3:0x6c
	v_or_b32_e32 v0, v0, v74
	v_bitop3_b32 v186, v73, v0, s28 bitop3:0x6c
	v_or_b32_e32 v0, v1, v74
	v_bitop3_b32 v187, v73, v0, s28 bitop3:0x6c
	v_or_b32_e32 v0, 0x18000, v68
	v_or_b32_e32 v1, v0, v70
	v_bitop3_b32 v188, v73, v1, s28 bitop3:0x6c
	s_mov_b32 s2, s44
	v_or_b32_e32 v0, v0, v74
	v_bitop3_b32 v189, v73, v0, s28 bitop3:0x6c
	v_or_b32_e32 v0, 0x1c000, v68
	v_or_b32_e32 v1, v0, v70
	v_bitop3_b32 v190, v73, v1, s28 bitop3:0x6c
	buffer_load_dwordx4 v169, s[24:27], s62 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s63
	s_add_i32 s63, s44, 0x3000
	v_or_b32_e32 v0, v0, v74
	v_bitop3_b32 v191, v73, v0, s28 bitop3:0x6c
	v_or_b32_e32 v0, v67, v66
	v_add_u32_e32 v1, v0, v70
	v_lshrrev_b32_e32 v66, 4, v1
	v_bitop3_b32 v192, v66, v1, s28 bitop3:0x6c
	buffer_load_dwordx4 v179, s[24:27], s62 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s65
	s_add_i32 s65, s38, 0x6000
	v_add_u32_e32 v0, v0, v74
	v_lshrrev_b32_e32 v1, 4, v0
	v_bitop3_b32 v193, v1, v0, s28 bitop3:0x6c
	buffer_load_dwordx4 v170, s[24:27], s62 offen lds
	s_add_i32 s62, s44, 0x2000
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s2
	s_mov_b32 s2, s35
	buffer_load_dwordx4 v168, s[24:27], s64 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s2
	s_mov_b32 s2, s62
	buffer_load_dwordx4 v169, s[24:27], s64 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s2
	s_mov_b32 s2, s63
	buffer_load_dwordx4 v179, s[24:27], s64 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s2
	s_or_b32 s2, s48, 0x80
	buffer_load_dwordx4 v170, s[24:27], s64 offen lds
	s_add_i32 s64, s38, 0x5000
	s_mov_b32 s3, s64
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s36
	v_accvgpr_write_b32 a2, 0
	buffer_load_dwordx4 v166, s[20:23], s2 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s3
	s_mov_b32 s3, s65
	buffer_load_dwordx4 v165, s[20:23], s2 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s3
	s_mov_b32 s3, s66
	buffer_load_dwordx4 v167, s[20:23], s2 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s3
	s_mov_b32 s3, s67
	buffer_load_dwordx4 v164, s[20:23], s2 offen lds
	s_add_i32 s2, s50, 0x80
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s37
	s_mov_b64 s[36:37], 0
	buffer_load_dwordx4 v166, s[20:23], s2 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s3
	s_mov_b32 s3, s68
	buffer_load_dwordx4 v165, s[20:23], s2 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s3
	s_mov_b32 s3, s69
	buffer_load_dwordx4 v167, s[20:23], s2 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s3
	s_mov_b32 s3, s70
	buffer_load_dwordx4 v164, s[20:23], s2 offen lds
	s_or_b32 s2, s46, 0x80
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s29
	v_accvgpr_write_b32 a1, 0
	buffer_load_dwordx4 v168, s[24:27], s2 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s3
	s_mov_b32 s3, s71
	buffer_load_dwordx4 v169, s[24:27], s2 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s3
	s_mov_b32 s3, s72
	buffer_load_dwordx4 v179, s[24:27], s2 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s3
	s_mov_b32 s3, s45
	buffer_load_dwordx4 v170, s[24:27], s2 offen lds
	s_add_i32 s2, s57, 0x80
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s3
	s_mov_b32 s3, s73
	buffer_load_dwordx4 v168, s[24:27], s2 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s3
	s_mov_b32 s3, s74
	buffer_load_dwordx4 v169, s[24:27], s2 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s3
	s_mov_b32 s3, s75
	buffer_load_dwordx4 v179, s[24:27], s2 offen lds
	;;#ASMSTART
	;;#ASMEND
	s_mov_b32 m0, s3
	v_accvgpr_write_b32 a0, 0
	buffer_load_dwordx4 v170, s[24:27], s2 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[34:37], v2 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[38:41], v2 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[42:45], v2 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[46:49], v2 offset:0x1800

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
	v_lshrrev_b32_e32 v2, 4, v10
	v_bitop3_b32 v14, v2, v10, s28 bitop3:0x6c
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
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_load_dwordx2 s[2:3], s[0:1], 0xc0
	s_load_dwordx2 s[28:29], s[0:1], 0xe0
	s_nop 0
	s_load_dword s0, s[0:1], 0xf0
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
	v_accvgpr_write_b32 a31, 0
	v_accvgpr_write_b32 a30, 0
	v_accvgpr_write_b32 a29, 0
	v_accvgpr_write_b32 a28, 0
	v_accvgpr_write_b32 a39, 0
	v_accvgpr_write_b32 a38, 0
	v_accvgpr_write_b32 a37, 0
	v_accvgpr_write_b32 a36, 0
	v_accvgpr_write_b32 a47, 0
	v_accvgpr_write_b32 a46, 0
	v_accvgpr_write_b32 a45, 0
	v_accvgpr_write_b32 a44, 0
	v_accvgpr_write_b32 a55, 0
	v_accvgpr_write_b32 a54, 0
	v_accvgpr_write_b32 a53, 0
	v_accvgpr_write_b32 a52, 0
	v_accvgpr_write_b32 a63, 0
	v_accvgpr_write_b32 a62, 0
	v_accvgpr_write_b32 a61, 0
	v_accvgpr_write_b32 a60, 0
	v_accvgpr_write_b32 a75, 0
	v_accvgpr_write_b32 a74, 0
	v_accvgpr_write_b32 a73, 0
	v_accvgpr_write_b32 a72, 0
	v_accvgpr_write_b32 a87, 0
	v_accvgpr_write_b32 a86, 0
	v_accvgpr_write_b32 a85, 0
	v_accvgpr_write_b32 a84, 0
	v_accvgpr_write_b32 a99, 0
	v_accvgpr_write_b32 a98, 0
	v_accvgpr_write_b32 a97, 0
	v_accvgpr_write_b32 a96, 0
	v_accvgpr_write_b32 a111, 0
	v_accvgpr_write_b32 a110, 0
	v_accvgpr_write_b32 a109, 0
	v_accvgpr_write_b32 a108, 0
	v_accvgpr_write_b32 a123, 0
	v_accvgpr_write_b32 a122, 0
	v_accvgpr_write_b32 a121, 0
	v_accvgpr_write_b32 a120, 0
	v_accvgpr_write_b32 a27, 0
	v_accvgpr_write_b32 a26, 0
	v_accvgpr_write_b32 a25, 0
	v_accvgpr_write_b32 a24, 0
	v_accvgpr_write_b32 a35, 0
	v_accvgpr_write_b32 a34, 0
	v_accvgpr_write_b32 a33, 0
	v_accvgpr_write_b32 a32, 0
	v_accvgpr_write_b32 a43, 0
	v_accvgpr_write_b32 a42, 0
	v_accvgpr_write_b32 a41, 0
	v_accvgpr_write_b32 a40, 0
	v_accvgpr_write_b32 a51, 0
	v_accvgpr_write_b32 a50, 0
	v_accvgpr_write_b32 a49, 0
	v_accvgpr_write_b32 a48, 0
	v_accvgpr_write_b32 a59, 0
	v_accvgpr_write_b32 a58, 0
	v_accvgpr_write_b32 a57, 0
	v_accvgpr_write_b32 a56, 0
	v_accvgpr_write_b32 a67, 0
	v_accvgpr_write_b32 a66, 0
	v_accvgpr_write_b32 a65, 0
	v_accvgpr_write_b32 a64, 0
	v_accvgpr_write_b32 a79, 0
	v_accvgpr_write_b32 a78, 0
	v_accvgpr_write_b32 a77, 0
	v_accvgpr_write_b32 a76, 0
	v_accvgpr_write_b32 a91, 0
	v_accvgpr_write_b32 a90, 0
	v_accvgpr_write_b32 a89, 0
	v_accvgpr_write_b32 a88, 0
	v_accvgpr_write_b32 a103, 0
	v_accvgpr_write_b32 a102, 0
	v_accvgpr_write_b32 a101, 0
	v_accvgpr_write_b32 a100, 0
	v_accvgpr_write_b32 a115, 0
	v_accvgpr_write_b32 a114, 0
	v_accvgpr_write_b32 a113, 0
	v_accvgpr_write_b32 a112, 0
	v_accvgpr_write_b32 a127, 0
	v_accvgpr_write_b32 a126, 0
	v_accvgpr_write_b32 a125, 0
	v_accvgpr_write_b32 a124, 0
	v_accvgpr_write_b32 a139, 0
	v_accvgpr_write_b32 a138, 0
	v_accvgpr_write_b32 a137, 0
	v_accvgpr_write_b32 a136, 0
	v_accvgpr_write_b32 a151, 0
	v_accvgpr_write_b32 a150, 0
	v_accvgpr_write_b32 a149, 0
	v_accvgpr_write_b32 a148, 0
	v_accvgpr_write_b32 a163, 0
	v_accvgpr_write_b32 a162, 0
	v_accvgpr_write_b32 a161, 0
	v_accvgpr_write_b32 a160, 0
	v_accvgpr_write_b32 a175, 0
	v_accvgpr_write_b32 a174, 0
	v_accvgpr_write_b32 a173, 0
	v_accvgpr_write_b32 a172, 0
	v_accvgpr_write_b32 a187, 0
	v_accvgpr_write_b32 a186, 0
	v_accvgpr_write_b32 a185, 0
	v_accvgpr_write_b32 a184, 0
	v_accvgpr_write_b32 a71, 0
	v_accvgpr_write_b32 a70, 0
	v_accvgpr_write_b32 a69, 0
	v_accvgpr_write_b32 a68, 0
	v_accvgpr_write_b32 a83, 0
	v_accvgpr_write_b32 a82, 0
	v_accvgpr_write_b32 a81, 0
	v_accvgpr_write_b32 a80, 0
	v_accvgpr_write_b32 a95, 0
	v_accvgpr_write_b32 a94, 0
	v_accvgpr_write_b32 a93, 0
	v_accvgpr_write_b32 a92, 0
	v_accvgpr_write_b32 a107, 0
	v_accvgpr_write_b32 a106, 0
	v_accvgpr_write_b32 a105, 0
	v_accvgpr_write_b32 a104, 0
	v_accvgpr_write_b32 a119, 0
	v_accvgpr_write_b32 a118, 0
	v_accvgpr_write_b32 a117, 0
	v_accvgpr_write_b32 a116, 0
	v_accvgpr_write_b32 a131, 0
	v_accvgpr_write_b32 a130, 0
	v_accvgpr_write_b32 a129, 0
	v_accvgpr_write_b32 a128, 0
	v_accvgpr_write_b32 a143, 0
	v_accvgpr_write_b32 a142, 0
	v_accvgpr_write_b32 a141, 0
	v_accvgpr_write_b32 a140, 0
	v_accvgpr_write_b32 a155, 0
	v_accvgpr_write_b32 a154, 0
	v_accvgpr_write_b32 a153, 0
	v_accvgpr_write_b32 a152, 0
	v_accvgpr_write_b32 a167, 0
	v_accvgpr_write_b32 a166, 0
	v_accvgpr_write_b32 a165, 0
	v_accvgpr_write_b32 a164, 0
	v_accvgpr_write_b32 a179, 0
	v_accvgpr_write_b32 a178, 0
	v_accvgpr_write_b32 a177, 0
	v_accvgpr_write_b32 a176, 0
	v_accvgpr_write_b32 a191, 0
	v_accvgpr_write_b32 a190, 0
	v_accvgpr_write_b32 a189, 0
	v_accvgpr_write_b32 a188, 0
	v_accvgpr_write_b32 a199, 0
	v_accvgpr_write_b32 a198, 0
	v_accvgpr_write_b32 a197, 0
	v_accvgpr_write_b32 a196, 0
	v_accvgpr_write_b32 a207, 0
	v_accvgpr_write_b32 a206, 0
	v_accvgpr_write_b32 a205, 0
	v_accvgpr_write_b32 a204, 0
	v_accvgpr_write_b32 a215, 0
	v_accvgpr_write_b32 a214, 0
	v_accvgpr_write_b32 a213, 0
	v_accvgpr_write_b32 a212, 0
	v_accvgpr_write_b32 a223, 0
	v_accvgpr_write_b32 a222, 0
	v_accvgpr_write_b32 a221, 0
	v_accvgpr_write_b32 a220, 0
	v_accvgpr_write_b32 a231, 0
	v_accvgpr_write_b32 a230, 0
	v_accvgpr_write_b32 a229, 0
	v_accvgpr_write_b32 a228, 0
	v_accvgpr_write_b32 a135, 0
	v_accvgpr_write_b32 a134, 0
	v_accvgpr_write_b32 a133, 0
	v_accvgpr_write_b32 a132, 0
	v_accvgpr_write_b32 a147, 0
	v_accvgpr_write_b32 a146, 0
	v_accvgpr_write_b32 a145, 0
	v_accvgpr_write_b32 a144, 0
	v_accvgpr_write_b32 a159, 0
	v_accvgpr_write_b32 a158, 0
	v_accvgpr_write_b32 a157, 0
	v_accvgpr_write_b32 a156, 0
	v_accvgpr_write_b32 a171, 0
	v_accvgpr_write_b32 a170, 0
	v_accvgpr_write_b32 a169, 0
	v_accvgpr_write_b32 a168, 0
	v_accvgpr_write_b32 a183, 0
	v_accvgpr_write_b32 a182, 0
	v_accvgpr_write_b32 a181, 0
	v_accvgpr_write_b32 a180, 0
	v_accvgpr_write_b32 a195, 0
	v_accvgpr_write_b32 a194, 0
	v_accvgpr_write_b32 a193, 0
	v_accvgpr_write_b32 a192, 0
	v_accvgpr_write_b32 a203, 0
	v_accvgpr_write_b32 a202, 0
	v_accvgpr_write_b32 a201, 0
	v_accvgpr_write_b32 a200, 0
	v_accvgpr_write_b32 a211, 0
	v_accvgpr_write_b32 a210, 0
	v_accvgpr_write_b32 a209, 0
	v_accvgpr_write_b32 a208, 0
	v_accvgpr_write_b32 a219, 0
	v_accvgpr_write_b32 a218, 0
	v_accvgpr_write_b32 a217, 0
	v_accvgpr_write_b32 a216, 0
	v_accvgpr_write_b32 a227, 0
	v_accvgpr_write_b32 a226, 0
	v_accvgpr_write_b32 a225, 0
	v_accvgpr_write_b32 a224, 0
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
.LBB0_7:                                ; =>This Inner Loop Header: Depth=1
	s_add_u32 s78, s48, s36
	s_mov_b32 m0, s38
	s_add_u32 s81, s78, 0x100
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[34:37], v[2:5], a[132:135],  v136, v134 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[98:101], v188 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[34:37], v[6:9], a[144:147],  v136, v134 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[102:105], v188 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[34:37], v[10:13], a[156:159],  v136, v135 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[106:109], v188 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171],  v[34:37], v[14:17], a[168:171],  v136, v135 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[110:113], v188 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[50:53], v[18:21], a[132:135],  v136, v134 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[114:117], v189 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[50:53], v[22:25], a[144:147],  v136, v134 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[118:121], v189 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[50:53], v[26:29], a[156:159],  v136, v135 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[122:125], v189 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171],  v[50:53], v[30:33], a[168:171],  v136, v135 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[126:129], v189 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183],  v[38:41], v[2:5], a[180:183],  v136, v134 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[38:41], v[6:9], a[192:195],  v136, v134 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[38:41], v[10:13], a[200:203],  v136, v135 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[38:41], v[14:17], a[208:211],  v136, v135 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183],  v[54:57], v[18:21], a[180:183],  v136, v134 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[54:57], v[22:25], a[192:195],  v136, v134 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[54:57], v[26:29], a[200:203],  v136, v135 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[54:57], v[30:33], a[208:211],  v136, v135 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[42:45], v[2:5], a[216:219],  v137, v134 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[42:45], v[6:9], a[224:227],  v137, v134 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[42:45], v[10:13], a[232:235], v137, v135 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[42:45], v[14:17], a[236:239], v137, v135 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[58:61], v[18:21], a[216:219],  v137, v134 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[58:61], v[22:25], a[224:227],  v137, v134 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[58:61], v[26:29], a[232:235], v137, v135 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[58:61], v[30:33], a[236:239], v137, v135 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[46:49], v[2:5], a[240:243], v137, v134 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[46:49], v[6:9], a[244:247], v137, v134 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[46:49], v[10:13], a[248:251], v137, v135 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[46:49], v[14:17], a[252:255], v137, v135 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[62:65], v[18:21], a[240:243], v137, v134 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[62:65], v[22:25], a[244:247], v137, v134 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[62:65], v[26:29], a[248:251], v137, v135 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[62:65], v[30:33], a[252:255], v137, v135 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[34:37], v[98:101], a[68:71], v136, v132 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[66:69], v180 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[34:37], v[102:105], a[80:83], v136, v132 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[70:73], v180 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[34:37], v[106:109], a[92:95], v136, v133 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[74:77], v180 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[34:37], v[110:113], a[104:107], v136, v133 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[78:81], v180 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[50:53], v[114:117], a[68:71], v136, v132 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[82:85], v186 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[50:53], v[118:121], a[80:83], v136, v132 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[86:89], v186 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[50:53], v[122:125], a[92:95], v136, v133 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[90:93], v186 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[50:53], v[126:129], a[104:107], v136, v133 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[94:97], v186 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[38:41], v[98:101], a[116:119], v136, v132 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[38:41], v[102:105], a[128:131], v136, v132 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[38:41], v[106:109], a[140:143], v136, v133 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[38:41], v[110:113], a[152:155], v136, v133 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[54:57], v[114:117], a[116:119], v136, v132 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[54:57], v[118:121], a[128:131], v136, v132 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[54:57], v[122:125], a[140:143], v136, v133 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[54:57], v[126:129], a[152:155], v136, v133 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[42:45], v[98:101], a[164:167], v137, v132 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[42:45], v[102:105], a[176:179], v137, v132 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[42:45], v[106:109], a[188:191], v137, v133 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[42:45], v[110:113], a[196:199], v137, v133 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[58:61], v[114:117], a[164:167], v137, v132 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[58:61], v[118:121], a[176:179], v137, v132 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[58:61], v[122:125], a[188:191], v137, v133 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[58:61], v[126:129], a[196:199], v137, v133 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[46:49], v[98:101], a[204:207], v137, v132 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[46:49], v[102:105], a[212:215], v137, v132 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[46:49], v[106:109], a[220:223], v137, v133 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[46:49], v[110:113], a[228:231], v137, v133 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[62:65], v[114:117], a[204:207], v137, v132 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[62:65], v[118:121], a[212:215], v137, v132 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[62:65], v[122:125], a[220:223], v137, v133 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[62:65], v[126:129], a[228:231], v137, v133 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(20)
s_barrier

	;;#ASMEND
	s_add_u32 s1, s50, s36
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27],  v[66:69], v[2:5], a[24:27],  v130, v134 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v176 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[66:69], v[6:9], a[32:35],  v130, v134 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v176 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[66:69], v[10:13], a[40:43],  v130, v135 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v176 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[66:69], v[14:17], a[48:51],  v130, v135 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v176 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27],  v[82:85], v[18:21], a[24:27],  v130, v134 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v184 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[82:85], v[22:25], a[32:35],  v130, v134 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v184 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[82:85], v[26:29], a[40:43],  v130, v135 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v184 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[82:85], v[30:33], a[48:51],  v130, v135 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[62:65], v184 offset:6144

	;;#ASMEND
	buffer_load_dwordx4 v166, s[20:23], s81 offen lds
	s_mov_b32 m0, s47
	s_add_u32 s79, s1, 0x100
	buffer_load_dwordx4 v165, s[20:23], s81 offen lds
	s_mov_b32 m0, s52
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[70:73], v[2:5], a[56:59],  v130, v134 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[70:73], v[6:9], a[64:67],  v130, v134 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79],  v[70:73], v[10:13], a[76:79],  v130, v135 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91],  v[70:73], v[14:17], a[88:91],  v130, v135 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[86:89], v[18:21], a[56:59],  v130, v134 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[86:89], v[22:25], a[64:67],  v130, v134 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79],  v[86:89], v[26:29], a[76:79],  v130, v135 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91],  v[86:89], v[30:33], a[88:91],  v130, v135 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_waitcnt lgkmcnt(0)
	s_add_u32 s29, s46, s36
	buffer_load_dwordx4 v167, s[20:23], s81 offen lds
	s_mov_b32 m0, s53
	s_add_u32 s80, s29, 0x100
	buffer_load_dwordx4 v164, s[20:23], s81 offen lds
	s_mov_b32 m0, s40
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103],  v[74:77], v[2:5], a[100:103],  v131, v134 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115],  v[74:77], v[6:9], a[112:115],  v131, v134 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[74:77], v[10:13], a[124:127], v131, v135 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[74:77], v[14:17], a[136:139], v131, v135 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103],  v[90:93], v[18:21], a[100:103],  v131, v134 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115],  v[90:93], v[22:25], a[112:115],  v131, v134 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[90:93], v[26:29], a[124:127], v131, v135 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[90:93], v[30:33], a[136:139], v131, v135 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_u32 s81, s78, 0x180
	buffer_load_dwordx4 v166, s[20:23], s79 offen lds
	s_mov_b32 m0, s54
	s_nop 0
	buffer_load_dwordx4 v165, s[20:23], s79 offen lds
	s_mov_b32 m0, s31
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[78:81], v[2:5], a[148:151], v131, v134 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[78:81], v[6:9], a[160:163], v131, v134 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[78:81], v[10:13], a[172:175], v131, v135 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[78:81], v[14:17], a[184:187], v131, v135 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[94:97], v[18:21], a[148:151], v131, v134 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[94:97], v[22:25], a[160:163], v131, v134 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[94:97], v[26:29], a[172:175], v131, v135 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[94:97], v[30:33], a[184:187], v131, v135 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v167, s[20:23], s79 offen lds
	s_mov_b32 m0, s55
	s_nop 0
	buffer_load_dwordx4 v164, s[20:23], s79 offen lds
	s_mov_b32 m0, s42
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3],  v[66:69], v[98:101], a[0:3],  v130, v132 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[2:5], v182 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7],  v[66:69], v[102:105], a[4:7],  v130, v132 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[6:9], v182 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11],  v[66:69], v[106:109], a[8:11],  v130, v133 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[10:13], v182 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15],  v[66:69], v[110:113], a[12:15],  v130, v133 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[14:17], v182 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3],  v[82:85], v[114:117], a[0:3],  v130, v132 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[18:21], v185 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7],  v[82:85], v[118:121], a[4:7],  v130, v132 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[22:25], v185 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11],  v[82:85], v[122:125], a[8:11],  v130, v133 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[26:29], v185 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15],  v[82:85], v[126:129], a[12:15],  v130, v133 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v185 offset:6144

	;;#ASMEND
	s_add_u32 s79, s57, s36
	buffer_load_dwordx4 v168, s[24:27], s80 offen lds
	s_mov_b32 m0, s59
	s_nop 0
	buffer_load_dwordx4 v169, s[24:27], s80 offen lds
	s_mov_b32 m0, s60
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19],  v[70:73], v[98:101], a[16:19],  v130, v132 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23],  v[70:73], v[102:105], a[20:23],  v130, v132 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[70:73], v[106:109], a[28:31],  v130, v133 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[70:73], v[110:113], a[36:39],  v130, v133 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19],  v[86:89], v[114:117], a[16:19],  v130, v132 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23],  v[86:89], v[118:121], a[20:23],  v130, v132 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[86:89], v[122:125], a[28:31],  v130, v133 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[86:89], v[126:129], a[36:39],  v130, v133 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v179, s[24:27], s80 offen lds
	s_mov_b32 m0, s61
	s_nop 0
	buffer_load_dwordx4 v170, s[24:27], s80 offen lds
	s_mov_b32 m0, s44
	s_add_u32 s80, s79, 0x100
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[74:77], v[98:101], a[44:47],  v131, v132 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[74:77], v[102:105], a[52:55],  v131, v132 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[74:77], v[106:109], a[60:63], v131, v133 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[74:77], v[110:113], a[72:75], v131, v133 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[90:93], v[114:117], a[44:47],  v131, v132 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[90:93], v[118:121], a[52:55],  v131, v132 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[90:93], v[122:125], a[60:63], v131, v133 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[90:93], v[126:129], a[72:75], v131, v133 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	buffer_load_dwordx4 v168, s[24:27], s80 offen lds
	s_mov_b32 m0, s35
	s_nop 0
	buffer_load_dwordx4 v169, s[24:27], s80 offen lds
	s_mov_b32 m0, s62
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[78:81], v[98:101], a[84:87], v131, v132 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[78:81], v[102:105], a[96:99], v131, v132 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[78:81], v[106:109], a[108:111], v131, v133 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[78:81], v[110:113], a[120:123], v131, v133 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[94:97], v[114:117], a[84:87], v131, v132 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[94:97], v[118:121], a[96:99], v131, v132 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[94:97], v[122:125], a[108:111], v131, v133 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[94:97], v[126:129], a[120:123], v131, v133 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v179, s[24:27], s80 offen lds
	s_mov_b32 m0, s63
	s_nop 0
	buffer_load_dwordx4 v170, s[24:27], s80 offen lds
	s_mov_b32 m0, s39
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_add_i32 s80, s77, 0xfffff200
	;;#ASMSTART
	buffer_load_dwordx2 v[0:1], v177, s[4:7], s80 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[130:131], v177, s[12:15], s80 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[172:173], v177, s[16:19], s80 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[174:175], v177, s[8:11], s80 offen
	;;#ASMEND
	s_add_u32 s80, s1, 0x180
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[34:37], v[2:5], a[132:135],  v0, v130 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[98:101], v190 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[34:37], v[6:9], a[144:147],  v0, v130 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[102:105], v190 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[34:37], v[10:13], a[156:159],  v0, v131 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[106:109], v190 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171],  v[34:37], v[14:17], a[168:171],  v0, v131 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[110:113], v190 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[50:53], v[18:21], a[132:135],  v0, v130 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[114:117], v191 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[50:53], v[22:25], a[144:147],  v0, v130 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[118:121], v191 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[50:53], v[26:29], a[156:159],  v0, v131 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[122:125], v191 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171],  v[50:53], v[30:33], a[168:171],  v0, v131 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[126:129], v191 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183],  v[38:41], v[2:5], a[180:183],  v0, v130 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[38:41], v[6:9], a[192:195],  v0, v130 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[38:41], v[10:13], a[200:203],  v0, v131 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[38:41], v[14:17], a[208:211],  v0, v131 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183],  v[54:57], v[18:21], a[180:183],  v0, v130 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[54:57], v[22:25], a[192:195],  v0, v130 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[54:57], v[26:29], a[200:203],  v0, v131 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[54:57], v[30:33], a[208:211],  v0, v131 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[42:45], v[2:5], a[216:219],  v1, v130 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[42:45], v[6:9], a[224:227],  v1, v130 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[42:45], v[10:13], a[232:235], v1, v131 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[42:45], v[14:17], a[236:239], v1, v131 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[58:61], v[18:21], a[216:219],  v1, v130 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[58:61], v[22:25], a[224:227],  v1, v130 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[58:61], v[26:29], a[232:235], v1, v131 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[58:61], v[30:33], a[236:239], v1, v131 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[46:49], v[2:5], a[240:243], v1, v130 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[46:49], v[6:9], a[244:247], v1, v130 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[46:49], v[10:13], a[248:251], v1, v131 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[46:49], v[14:17], a[252:255], v1, v131 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[62:65], v[18:21], a[240:243], v1, v130 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[62:65], v[22:25], a[244:247], v1, v130 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[62:65], v[26:29], a[248:251], v1, v131 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[62:65], v[30:33], a[252:255], v1, v131 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[34:37], v[98:101], a[68:71], v0, v172 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[66:69], v181 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[34:37], v[102:105], a[80:83], v0, v172 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[70:73], v181 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[34:37], v[106:109], a[92:95], v0, v173 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[74:77], v181 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[34:37], v[110:113], a[104:107], v0, v173 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[78:81], v181 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[50:53], v[114:117], a[68:71], v0, v172 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[82:85], v187 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[50:53], v[118:121], a[80:83], v0, v172 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[86:89], v187 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[50:53], v[122:125], a[92:95], v0, v173 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[90:93], v187 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[50:53], v[126:129], a[104:107], v0, v173 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[94:97], v187 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[38:41], v[98:101], a[116:119], v0, v172 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[38:41], v[102:105], a[128:131], v0, v172 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[38:41], v[106:109], a[140:143], v0, v173 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[38:41], v[110:113], a[152:155], v0, v173 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[54:57], v[114:117], a[116:119], v0, v172 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[54:57], v[118:121], a[128:131], v0, v172 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[54:57], v[122:125], a[140:143], v0, v173 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[54:57], v[126:129], a[152:155], v0, v173 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[42:45], v[98:101], a[164:167], v1, v172 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[42:45], v[102:105], a[176:179], v1, v172 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[42:45], v[106:109], a[188:191], v1, v173 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[42:45], v[110:113], a[196:199], v1, v173 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[58:61], v[114:117], a[164:167], v1, v172 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[58:61], v[118:121], a[176:179], v1, v172 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[58:61], v[122:125], a[188:191], v1, v173 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[58:61], v[126:129], a[196:199], v1, v173 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[46:49], v[98:101], a[204:207], v1, v172 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[46:49], v[102:105], a[212:215], v1, v172 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[46:49], v[106:109], a[220:223], v1, v173 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[46:49], v[110:113], a[228:231], v1, v173 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[62:65], v[114:117], a[204:207], v1, v172 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[62:65], v[118:121], a[212:215], v1, v172 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[62:65], v[122:125], a[220:223], v1, v173 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[62:65], v[126:129], a[228:231], v1, v173 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(20)
s_barrier

	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27],  v[66:69], v[2:5], a[24:27],  v174, v130 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v171 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[66:69], v[6:9], a[32:35],  v174, v130 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v171 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[66:69], v[10:13], a[40:43],  v174, v131 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v171 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[66:69], v[14:17], a[48:51],  v174, v131 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v171 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27],  v[82:85], v[18:21], a[24:27],  v174, v130 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v183 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[82:85], v[22:25], a[32:35],  v174, v130 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v183 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[82:85], v[26:29], a[40:43],  v174, v131 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v183 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[82:85], v[30:33], a[48:51],  v174, v131 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[62:65], v183 offset:6144

	;;#ASMEND
	buffer_load_dwordx4 v166, s[20:23], s81 offen lds
	s_mov_b32 m0, s64
	s_nop 0
	buffer_load_dwordx4 v165, s[20:23], s81 offen lds
	s_mov_b32 m0, s65
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[70:73], v[2:5], a[56:59],  v174, v130 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[70:73], v[6:9], a[64:67],  v174, v130 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79],  v[70:73], v[10:13], a[76:79],  v174, v131 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91],  v[70:73], v[14:17], a[88:91],  v174, v131 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[86:89], v[18:21], a[56:59],  v174, v130 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[86:89], v[22:25], a[64:67],  v174, v130 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79],  v[86:89], v[26:29], a[76:79],  v174, v131 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91],  v[86:89], v[30:33], a[88:91],  v174, v131 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v167, s[20:23], s81 offen lds
	s_mov_b32 m0, s66
	s_nop 0
	buffer_load_dwordx4 v164, s[20:23], s81 offen lds
	s_mov_b32 m0, s41
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103],  v[74:77], v[2:5], a[100:103],  v175, v130 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115],  v[74:77], v[6:9], a[112:115],  v175, v130 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[74:77], v[10:13], a[124:127], v175, v131 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[74:77], v[14:17], a[136:139], v175, v131 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103],  v[90:93], v[18:21], a[100:103],  v175, v130 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115],  v[90:93], v[22:25], a[112:115],  v175, v130 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[90:93], v[26:29], a[124:127], v175, v131 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[90:93], v[30:33], a[136:139], v175, v131 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_u32 s81, s78, 0x200
	buffer_load_dwordx4 v166, s[20:23], s80 offen lds
	s_mov_b32 m0, s67
	s_nop 0
	buffer_load_dwordx4 v165, s[20:23], s80 offen lds
	s_mov_b32 m0, s68
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[78:81], v[2:5], a[148:151], v175, v130 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[78:81], v[6:9], a[160:163], v175, v130 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[78:81], v[10:13], a[172:175], v175, v131 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[78:81], v[14:17], a[184:187], v175, v131 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[94:97], v[18:21], a[148:151], v175, v130 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[94:97], v[22:25], a[160:163], v175, v130 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[94:97], v[26:29], a[172:175], v175, v131 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[94:97], v[30:33], a[184:187], v175, v131 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v167, s[20:23], s80 offen lds
	s_mov_b32 m0, s69
	s_nop 0
	buffer_load_dwordx4 v164, s[20:23], s80 offen lds
	s_mov_b32 m0, s43
	s_add_u32 s80, s29, 0x180
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3],  v[66:69], v[98:101], a[0:3],  v174, v172 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[130:133], v192 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7],  v[66:69], v[102:105], a[4:7],  v174, v172 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[134:137], v192 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11],  v[66:69], v[106:109], a[8:11],  v174, v173 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[138:141], v192 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15],  v[66:69], v[110:113], a[12:15],  v174, v173 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[142:145], v192 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3],  v[82:85], v[114:117], a[0:3],  v174, v172 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[146:149], v193 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7],  v[82:85], v[118:121], a[4:7],  v174, v172 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[150:153], v193 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11],  v[82:85], v[122:125], a[8:11],  v174, v173 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[154:157], v193 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15],  v[82:85], v[126:129], a[12:15],  v174, v173 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[158:161], v193 offset:6144

	;;#ASMEND
	buffer_load_dwordx4 v168, s[24:27], s80 offen lds
	s_mov_b32 m0, s70
	s_nop 0
	buffer_load_dwordx4 v169, s[24:27], s80 offen lds
	s_mov_b32 m0, s71
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19],  v[70:73], v[98:101], a[16:19],  v174, v172 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23],  v[70:73], v[102:105], a[20:23],  v174, v172 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[70:73], v[106:109], a[28:31],  v174, v173 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[70:73], v[110:113], a[36:39],  v174, v173 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19],  v[86:89], v[114:117], a[16:19],  v174, v172 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23],  v[86:89], v[118:121], a[20:23],  v174, v172 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[86:89], v[122:125], a[28:31],  v174, v173 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[86:89], v[126:129], a[36:39],  v174, v173 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v179, s[24:27], s80 offen lds
	s_mov_b32 m0, s72
	s_nop 0
	buffer_load_dwordx4 v170, s[24:27], s80 offen lds
	s_mov_b32 m0, s45
	s_add_u32 s80, s79, 0x180
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[74:77], v[98:101], a[44:47],  v175, v172 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[74:77], v[102:105], a[52:55],  v175, v172 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[74:77], v[106:109], a[60:63], v175, v173 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[74:77], v[110:113], a[72:75], v175, v173 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[90:93], v[114:117], a[44:47],  v175, v172 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[90:93], v[118:121], a[52:55],  v175, v172 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[90:93], v[122:125], a[60:63], v175, v173 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[90:93], v[126:129], a[72:75], v175, v173 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	buffer_load_dwordx4 v168, s[24:27], s80 offen lds
	s_mov_b32 m0, s73
	s_nop 0
	buffer_load_dwordx4 v169, s[24:27], s80 offen lds
	s_mov_b32 m0, s74
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[78:81], v[98:101], a[84:87], v175, v172 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[78:81], v[102:105], a[96:99], v175, v172 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[78:81], v[106:109], a[108:111], v175, v173 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[78:81], v[110:113], a[120:123], v175, v173 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[94:97], v[114:117], a[84:87], v175, v172 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[94:97], v[118:121], a[96:99], v175, v172 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[94:97], v[122:125], a[108:111], v175, v173 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[94:97], v[126:129], a[120:123], v175, v173 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v179, s[24:27], s80 offen lds
	s_mov_b32 m0, s75
	s_nop 0
	buffer_load_dwordx4 v170, s[24:27], s80 offen lds
	s_mov_b32 m0, s38
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_add_i32 s80, s77, 0xfffff400
	;;#ASMSTART
	buffer_load_dwordx2 v[0:1], v177, s[4:7], s80 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[174:175], v177, s[12:15], s80 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[162:163], v177, s[16:19], s80 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[172:173], v177, s[8:11], s80 offen
	;;#ASMEND
	s_add_u32 s80, s1, 0x200
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[34:37], v[130:133], a[132:135],  v0, v174 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[98:101], v188 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[34:37], v[134:137], a[144:147],  v0, v174 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[102:105], v188 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[34:37], v[138:141], a[156:159],  v0, v175 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[106:109], v188 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171],  v[34:37], v[142:145], a[168:171],  v0, v175 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[110:113], v188 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[50:53], v[146:149], a[132:135],  v0, v174 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[114:117], v189 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[50:53], v[150:153], a[144:147],  v0, v174 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[118:121], v189 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[50:53], v[154:157], a[156:159],  v0, v175 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[122:125], v189 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171],  v[50:53], v[158:161], a[168:171],  v0, v175 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[126:129], v189 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183],  v[38:41], v[130:133], a[180:183],  v0, v174 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[38:41], v[134:137], a[192:195],  v0, v174 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[38:41], v[138:141], a[200:203],  v0, v175 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[38:41], v[142:145], a[208:211],  v0, v175 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183],  v[54:57], v[146:149], a[180:183],  v0, v174 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[54:57], v[150:153], a[192:195],  v0, v174 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[54:57], v[154:157], a[200:203],  v0, v175 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[54:57], v[158:161], a[208:211],  v0, v175 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[42:45], v[130:133], a[216:219],  v1, v174 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[42:45], v[134:137], a[224:227],  v1, v174 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[42:45], v[138:141], a[232:235], v1, v175 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[42:45], v[142:145], a[236:239], v1, v175 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[58:61], v[146:149], a[216:219],  v1, v174 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[58:61], v[150:153], a[224:227],  v1, v174 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[58:61], v[154:157], a[232:235], v1, v175 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[58:61], v[158:161], a[236:239], v1, v175 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[46:49], v[130:133], a[240:243], v1, v174 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[46:49], v[134:137], a[244:247], v1, v174 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[46:49], v[138:141], a[248:251], v1, v175 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[46:49], v[142:145], a[252:255], v1, v175 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[62:65], v[146:149], a[240:243], v1, v174 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[62:65], v[150:153], a[244:247], v1, v174 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[62:65], v[154:157], a[248:251], v1, v175 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[62:65], v[158:161], a[252:255], v1, v175 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[34:37], v[98:101], a[68:71], v0, v162 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[66:69], v180 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[34:37], v[102:105], a[80:83], v0, v162 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[70:73], v180 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[34:37], v[106:109], a[92:95], v0, v163 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[74:77], v180 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[34:37], v[110:113], a[104:107], v0, v163 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[78:81], v180 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[50:53], v[114:117], a[68:71], v0, v162 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[82:85], v186 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[50:53], v[118:121], a[80:83], v0, v162 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[86:89], v186 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[50:53], v[122:125], a[92:95], v0, v163 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[90:93], v186 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[50:53], v[126:129], a[104:107], v0, v163 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[94:97], v186 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[38:41], v[98:101], a[116:119], v0, v162 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[38:41], v[102:105], a[128:131], v0, v162 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[38:41], v[106:109], a[140:143], v0, v163 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[38:41], v[110:113], a[152:155], v0, v163 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[54:57], v[114:117], a[116:119], v0, v162 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[54:57], v[118:121], a[128:131], v0, v162 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[54:57], v[122:125], a[140:143], v0, v163 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[54:57], v[126:129], a[152:155], v0, v163 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[42:45], v[98:101], a[164:167], v1, v162 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[42:45], v[102:105], a[176:179], v1, v162 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[42:45], v[106:109], a[188:191], v1, v163 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[42:45], v[110:113], a[196:199], v1, v163 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[58:61], v[114:117], a[164:167], v1, v162 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[58:61], v[118:121], a[176:179], v1, v162 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[58:61], v[122:125], a[188:191], v1, v163 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[58:61], v[126:129], a[196:199], v1, v163 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[46:49], v[98:101], a[204:207], v1, v162 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[46:49], v[102:105], a[212:215], v1, v162 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[46:49], v[106:109], a[220:223], v1, v163 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[46:49], v[110:113], a[228:231], v1, v163 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[62:65], v[114:117], a[204:207], v1, v162 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[62:65], v[118:121], a[212:215], v1, v162 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[62:65], v[122:125], a[220:223], v1, v163 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[62:65], v[126:129], a[228:231], v1, v163 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(20)
s_barrier

	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27],  v[66:69], v[130:133], a[24:27],  v172, v174 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[2:5], v176 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[66:69], v[134:137], a[32:35],  v172, v174 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[6:9], v176 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[66:69], v[138:141], a[40:43],  v172, v175 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[10:13], v176 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[66:69], v[142:145], a[48:51],  v172, v175 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[14:17], v176 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27],  v[82:85], v[146:149], a[24:27],  v172, v174 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[18:21], v184 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[82:85], v[150:153], a[32:35],  v172, v174 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[22:25], v184 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[82:85], v[154:157], a[40:43],  v172, v175 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[26:29], v184 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[82:85], v[158:161], a[48:51],  v172, v175 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v184 offset:6144

	;;#ASMEND
	buffer_load_dwordx4 v166, s[20:23], s81 offen lds
	s_mov_b32 m0, s47
	s_nop 0
	buffer_load_dwordx4 v165, s[20:23], s81 offen lds
	s_mov_b32 m0, s52
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[70:73], v[130:133], a[56:59],  v172, v174 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[70:73], v[134:137], a[64:67],  v172, v174 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79],  v[70:73], v[138:141], a[76:79],  v172, v175 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91],  v[70:73], v[142:145], a[88:91],  v172, v175 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[86:89], v[146:149], a[56:59],  v172, v174 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[86:89], v[150:153], a[64:67],  v172, v174 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79],  v[86:89], v[154:157], a[76:79],  v172, v175 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91],  v[86:89], v[158:161], a[88:91],  v172, v175 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v167, s[20:23], s81 offen lds
	s_mov_b32 m0, s53
	s_nop 0
	buffer_load_dwordx4 v164, s[20:23], s81 offen lds
	s_mov_b32 m0, s40
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103],  v[74:77], v[130:133], a[100:103],  v173, v174 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115],  v[74:77], v[134:137], a[112:115],  v173, v174 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[74:77], v[138:141], a[124:127], v173, v175 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[74:77], v[142:145], a[136:139], v173, v175 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103],  v[90:93], v[146:149], a[100:103],  v173, v174 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115],  v[90:93], v[150:153], a[112:115],  v173, v174 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[90:93], v[154:157], a[124:127], v173, v175 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[90:93], v[158:161], a[136:139], v173, v175 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_u32 s81, s78, 0x280
	buffer_load_dwordx4 v166, s[20:23], s80 offen lds
	s_mov_b32 m0, s54
	s_nop 0
	buffer_load_dwordx4 v165, s[20:23], s80 offen lds
	s_mov_b32 m0, s31
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[78:81], v[130:133], a[148:151], v173, v174 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[78:81], v[134:137], a[160:163], v173, v174 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[78:81], v[138:141], a[172:175], v173, v175 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[78:81], v[142:145], a[184:187], v173, v175 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[94:97], v[146:149], a[148:151], v173, v174 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[94:97], v[150:153], a[160:163], v173, v174 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[94:97], v[154:157], a[172:175], v173, v175 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[94:97], v[158:161], a[184:187], v173, v175 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v167, s[20:23], s80 offen lds
	s_mov_b32 m0, s55
	s_nop 0
	buffer_load_dwordx4 v164, s[20:23], s80 offen lds
	s_mov_b32 m0, s42
	s_add_u32 s80, s29, 0x200
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3],  v[66:69], v[98:101], a[0:3],  v172, v162 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v182 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7],  v[66:69], v[102:105], a[4:7],  v172, v162 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v182 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11],  v[66:69], v[106:109], a[8:11],  v172, v163 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v182 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15],  v[66:69], v[110:113], a[12:15],  v172, v163 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v182 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3],  v[82:85], v[114:117], a[0:3],  v172, v162 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v185 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7],  v[82:85], v[118:121], a[4:7],  v172, v162 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v185 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11],  v[82:85], v[122:125], a[8:11],  v172, v163 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v185 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15],  v[82:85], v[126:129], a[12:15],  v172, v163 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[62:65], v185 offset:6144

	;;#ASMEND
	buffer_load_dwordx4 v168, s[24:27], s80 offen lds
	s_mov_b32 m0, s59
	s_nop 0
	buffer_load_dwordx4 v169, s[24:27], s80 offen lds
	s_mov_b32 m0, s60
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19],  v[70:73], v[98:101], a[16:19],  v172, v162 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23],  v[70:73], v[102:105], a[20:23],  v172, v162 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[70:73], v[106:109], a[28:31],  v172, v163 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[70:73], v[110:113], a[36:39],  v172, v163 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19],  v[86:89], v[114:117], a[16:19],  v172, v162 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23],  v[86:89], v[118:121], a[20:23],  v172, v162 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[86:89], v[122:125], a[28:31],  v172, v163 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[86:89], v[126:129], a[36:39],  v172, v163 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v179, s[24:27], s80 offen lds
	s_mov_b32 m0, s61
	s_nop 0
	buffer_load_dwordx4 v170, s[24:27], s80 offen lds
	s_mov_b32 m0, s44
	s_add_u32 s80, s79, 0x200
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[74:77], v[98:101], a[44:47],  v173, v162 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[74:77], v[102:105], a[52:55],  v173, v162 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[74:77], v[106:109], a[60:63], v173, v163 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[74:77], v[110:113], a[72:75], v173, v163 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[90:93], v[114:117], a[44:47],  v173, v162 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[90:93], v[118:121], a[52:55],  v173, v162 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[90:93], v[122:125], a[60:63], v173, v163 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[90:93], v[126:129], a[72:75], v173, v163 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	buffer_load_dwordx4 v168, s[24:27], s80 offen lds
	s_mov_b32 m0, s35
	s_nop 0
	buffer_load_dwordx4 v169, s[24:27], s80 offen lds
	s_mov_b32 m0, s62
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[78:81], v[98:101], a[84:87], v173, v162 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[78:81], v[102:105], a[96:99], v173, v162 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[78:81], v[106:109], a[108:111], v173, v163 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[78:81], v[110:113], a[120:123], v173, v163 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[94:97], v[114:117], a[84:87], v173, v162 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[94:97], v[118:121], a[96:99], v173, v162 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[94:97], v[122:125], a[108:111], v173, v163 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[94:97], v[126:129], a[120:123], v173, v163 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v179, s[24:27], s80 offen lds
	s_mov_b32 m0, s63
	s_nop 0
	buffer_load_dwordx4 v170, s[24:27], s80 offen lds
	s_mov_b32 m0, s39
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_add_i32 s80, s77, 0xfffff600
	;;#ASMSTART
	buffer_load_dwordx2 v[0:1], v177, s[4:7], s80 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[132:133], v177, s[12:15], s80 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[134:135], v177, s[16:19], s80 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[130:131], v177, s[8:11], s80 offen
	;;#ASMEND
	s_add_u32 s80, s1, 0x280
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[2:5], v[34:37], a[132:135],  v0, v132 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[98:101], v190 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[2:5], v[38:41], a[144:147],  v0, v132 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[102:105], v190 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[2:5], v[42:45], a[156:159],  v0, v133 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[106:109], v190 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171],  v[2:5], v[46:49], a[168:171],  v0, v133 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[110:113], v190 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[18:21], v[50:53], a[132:135],  v0, v132 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[114:117], v191 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[18:21], v[54:57], a[144:147],  v0, v132 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[118:121], v191 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[18:21], v[58:61], a[156:159],  v0, v133 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[122:125], v191 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171],  v[18:21], v[62:65], a[168:171],  v0, v133 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[126:129], v191 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183],  v[6:9], v[34:37], a[180:183],  v0, v132 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[6:9], v[38:41], a[192:195],  v0, v132 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[6:9], v[42:45], a[200:203],  v0, v133 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[6:9], v[46:49], a[208:211],  v0, v133 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183],  v[22:25], v[50:53], a[180:183],  v0, v132 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[22:25], v[54:57], a[192:195],  v0, v132 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[22:25], v[58:61], a[200:203],  v0, v133 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[22:25], v[62:65], a[208:211],  v0, v133 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[10:13], v[34:37], a[216:219],  v1, v132 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[10:13], v[38:41], a[224:227],  v1, v132 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[10:13], v[42:45], a[232:235], v1, v133 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[10:13], v[46:49], a[236:239], v1, v133 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[26:29], v[50:53], a[216:219],  v1, v132 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[26:29], v[54:57], a[224:227],  v1, v132 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[26:29], v[58:61], a[232:235], v1, v133 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[26:29], v[62:65], a[236:239], v1, v133 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[14:17], v[34:37], a[240:243], v1, v132 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[14:17], v[38:41], a[244:247], v1, v132 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[14:17], v[42:45], a[248:251], v1, v133 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[14:17], v[46:49], a[252:255], v1, v133 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[30:33], v[50:53], a[240:243], v1, v132 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[30:33], v[54:57], a[244:247], v1, v132 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[30:33], v[58:61], a[248:251], v1, v133 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[30:33], v[62:65], a[252:255], v1, v133 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[2:5], v[98:101], a[68:71], v0, v134 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[66:69], v181 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[2:5], v[102:105], a[80:83], v0, v134 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[70:73], v181 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[2:5], v[106:109], a[92:95], v0, v135 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[74:77], v181 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[2:5], v[110:113], a[104:107], v0, v135 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[78:81], v181 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[18:21], v[114:117], a[68:71], v0, v134 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[82:85], v187 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[18:21], v[118:121], a[80:83], v0, v134 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[86:89], v187 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[18:21], v[122:125], a[92:95], v0, v135 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[90:93], v187 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[18:21], v[126:129], a[104:107], v0, v135 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[94:97], v187 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[6:9], v[98:101], a[116:119], v0, v134 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[6:9], v[102:105], a[128:131], v0, v134 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[6:9], v[106:109], a[140:143], v0, v135 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[6:9], v[110:113], a[152:155], v0, v135 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[22:25], v[114:117], a[116:119], v0, v134 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[22:25], v[118:121], a[128:131], v0, v134 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[22:25], v[122:125], a[140:143], v0, v135 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[22:25], v[126:129], a[152:155], v0, v135 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[10:13], v[98:101], a[164:167], v1, v134 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[10:13], v[102:105], a[176:179], v1, v134 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[10:13], v[106:109], a[188:191], v1, v135 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[10:13], v[110:113], a[196:199], v1, v135 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[26:29], v[114:117], a[164:167], v1, v134 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[26:29], v[118:121], a[176:179], v1, v134 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[26:29], v[122:125], a[188:191], v1, v135 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[26:29], v[126:129], a[196:199], v1, v135 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[14:17], v[98:101], a[204:207], v1, v134 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[14:17], v[102:105], a[212:215], v1, v134 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[14:17], v[106:109], a[220:223], v1, v135 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[14:17], v[110:113], a[228:231], v1, v135 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[30:33], v[114:117], a[204:207], v1, v134 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[30:33], v[118:121], a[212:215], v1, v134 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[30:33], v[122:125], a[220:223], v1, v135 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[30:33], v[126:129], a[228:231], v1, v135 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(20)
s_barrier

	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27],  v[66:69], v[34:37], a[24:27],  v130, v132 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[2:5], v171 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[66:69], v[38:41], a[32:35],  v130, v132 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[6:9], v171 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[66:69], v[42:45], a[40:43],  v130, v133 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[10:13], v171 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[66:69], v[46:49], a[48:51],  v130, v133 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[14:17], v171 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27],  v[82:85], v[50:53], a[24:27],  v130, v132 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[18:21], v183 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[82:85], v[54:57], a[32:35],  v130, v132 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[22:25], v183 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[82:85], v[58:61], a[40:43],  v130, v133 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[26:29], v183 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[82:85], v[62:65], a[48:51],  v130, v133 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v183 offset:6144

	;;#ASMEND
	buffer_load_dwordx4 v166, s[20:23], s81 offen lds
	s_mov_b32 m0, s64
	s_nop 0
	buffer_load_dwordx4 v165, s[20:23], s81 offen lds
	s_mov_b32 m0, s65
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[70:73], v[34:37], a[56:59],  v130, v132 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[70:73], v[38:41], a[64:67],  v130, v132 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79],  v[70:73], v[42:45], a[76:79],  v130, v133 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91],  v[70:73], v[46:49], a[88:91],  v130, v133 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[86:89], v[50:53], a[56:59],  v130, v132 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[86:89], v[54:57], a[64:67],  v130, v132 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79],  v[86:89], v[58:61], a[76:79],  v130, v133 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91],  v[86:89], v[62:65], a[88:91],  v130, v133 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v167, s[20:23], s81 offen lds
	s_mov_b32 m0, s66
	s_nop 0
	buffer_load_dwordx4 v164, s[20:23], s81 offen lds
	s_mov_b32 m0, s41
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103],  v[74:77], v[34:37], a[100:103],  v131, v132 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115],  v[74:77], v[38:41], a[112:115],  v131, v132 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[74:77], v[42:45], a[124:127], v131, v133 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[74:77], v[46:49], a[136:139], v131, v133 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103],  v[90:93], v[50:53], a[100:103],  v131, v132 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115],  v[90:93], v[54:57], a[112:115],  v131, v132 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[90:93], v[58:61], a[124:127], v131, v133 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[90:93], v[62:65], a[136:139], v131, v133 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_u32 s81, s78, 0x300
	buffer_load_dwordx4 v166, s[20:23], s80 offen lds
	s_mov_b32 m0, s67
	s_add_u32 s78, s78, 0x380
	buffer_load_dwordx4 v165, s[20:23], s80 offen lds
	s_mov_b32 m0, s68
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[78:81], v[34:37], a[148:151], v131, v132 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[78:81], v[38:41], a[160:163], v131, v132 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[78:81], v[42:45], a[172:175], v131, v133 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[78:81], v[46:49], a[184:187], v131, v133 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[94:97], v[50:53], a[148:151], v131, v132 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[94:97], v[54:57], a[160:163], v131, v132 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[94:97], v[58:61], a[172:175], v131, v133 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[94:97], v[62:65], a[184:187], v131, v133 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v167, s[20:23], s80 offen lds
	s_mov_b32 m0, s69
	s_nop 0
	buffer_load_dwordx4 v164, s[20:23], s80 offen lds
	s_mov_b32 m0, s43
	s_add_u32 s80, s29, 0x280
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3],  v[66:69], v[98:101], a[0:3],  v130, v134 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v192 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7],  v[66:69], v[102:105], a[4:7],  v130, v134 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v192 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11],  v[66:69], v[106:109], a[8:11],  v130, v135 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v192 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15],  v[66:69], v[110:113], a[12:15],  v130, v135 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v192 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3],  v[82:85], v[114:117], a[0:3],  v130, v134 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v193 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7],  v[82:85], v[118:121], a[4:7],  v130, v134 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v193 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11],  v[82:85], v[122:125], a[8:11],  v130, v135 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v193 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15],  v[82:85], v[126:129], a[12:15],  v130, v135 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[62:65], v193 offset:6144

	;;#ASMEND
	buffer_load_dwordx4 v168, s[24:27], s80 offen lds
	s_mov_b32 m0, s70
	s_nop 0
	buffer_load_dwordx4 v169, s[24:27], s80 offen lds
	s_mov_b32 m0, s71
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19],  v[70:73], v[98:101], a[16:19],  v130, v134 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23],  v[70:73], v[102:105], a[20:23],  v130, v134 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[70:73], v[106:109], a[28:31],  v130, v135 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[70:73], v[110:113], a[36:39],  v130, v135 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19],  v[86:89], v[114:117], a[16:19],  v130, v134 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23],  v[86:89], v[118:121], a[20:23],  v130, v134 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[86:89], v[122:125], a[28:31],  v130, v135 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[86:89], v[126:129], a[36:39],  v130, v135 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v179, s[24:27], s80 offen lds
	s_mov_b32 m0, s72
	s_nop 0
	buffer_load_dwordx4 v170, s[24:27], s80 offen lds
	s_mov_b32 m0, s45
	s_add_u32 s80, s79, 0x280
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[74:77], v[98:101], a[44:47],  v131, v134 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[74:77], v[102:105], a[52:55],  v131, v134 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[74:77], v[106:109], a[60:63], v131, v135 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[74:77], v[110:113], a[72:75], v131, v135 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[90:93], v[114:117], a[44:47],  v131, v134 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[90:93], v[118:121], a[52:55],  v131, v134 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[90:93], v[122:125], a[60:63], v131, v135 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[90:93], v[126:129], a[72:75], v131, v135 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	buffer_load_dwordx4 v168, s[24:27], s80 offen lds
	s_mov_b32 m0, s73
	s_nop 0
	buffer_load_dwordx4 v169, s[24:27], s80 offen lds
	s_mov_b32 m0, s74
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[78:81], v[98:101], a[84:87], v131, v134 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[78:81], v[102:105], a[96:99], v131, v134 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[78:81], v[106:109], a[108:111], v131, v135 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[78:81], v[110:113], a[120:123], v131, v135 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[94:97], v[114:117], a[84:87], v131, v134 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[94:97], v[118:121], a[96:99], v131, v134 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[94:97], v[122:125], a[108:111], v131, v135 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[94:97], v[126:129], a[120:123], v131, v135 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v179, s[24:27], s80 offen lds
	s_mov_b32 m0, s75
	s_nop 0
	buffer_load_dwordx4 v170, s[24:27], s80 offen lds
	s_mov_b32 m0, s38
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_add_i32 s80, s77, 0xfffff800
	;;#ASMSTART
	buffer_load_dwordx2 v[0:1], v177, s[4:7], s80 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[132:133], v177, s[12:15], s80 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[134:135], v177, s[16:19], s80 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[130:131], v177, s[8:11], s80 offen
	;;#ASMEND
	s_add_u32 s80, s1, 0x300
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[2:5], v[34:37], a[132:135],  v0, v132 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[98:101], v188 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[2:5], v[38:41], a[144:147],  v0, v132 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[102:105], v188 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[2:5], v[42:45], a[156:159],  v0, v133 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[106:109], v188 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171],  v[2:5], v[46:49], a[168:171],  v0, v133 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[110:113], v188 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[18:21], v[50:53], a[132:135],  v0, v132 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[114:117], v189 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[18:21], v[54:57], a[144:147],  v0, v132 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[118:121], v189 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[18:21], v[58:61], a[156:159],  v0, v133 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[122:125], v189 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171],  v[18:21], v[62:65], a[168:171],  v0, v133 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[126:129], v189 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183],  v[6:9], v[34:37], a[180:183],  v0, v132 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[6:9], v[38:41], a[192:195],  v0, v132 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[6:9], v[42:45], a[200:203],  v0, v133 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[6:9], v[46:49], a[208:211],  v0, v133 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183],  v[22:25], v[50:53], a[180:183],  v0, v132 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[22:25], v[54:57], a[192:195],  v0, v132 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[22:25], v[58:61], a[200:203],  v0, v133 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[22:25], v[62:65], a[208:211],  v0, v133 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[10:13], v[34:37], a[216:219],  v1, v132 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[10:13], v[38:41], a[224:227],  v1, v132 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[10:13], v[42:45], a[232:235], v1, v133 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[10:13], v[46:49], a[236:239], v1, v133 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[26:29], v[50:53], a[216:219],  v1, v132 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[26:29], v[54:57], a[224:227],  v1, v132 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[26:29], v[58:61], a[232:235], v1, v133 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[26:29], v[62:65], a[236:239], v1, v133 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[14:17], v[34:37], a[240:243], v1, v132 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[14:17], v[38:41], a[244:247], v1, v132 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[14:17], v[42:45], a[248:251], v1, v133 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[14:17], v[46:49], a[252:255], v1, v133 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[30:33], v[50:53], a[240:243], v1, v132 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[30:33], v[54:57], a[244:247], v1, v132 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[30:33], v[58:61], a[248:251], v1, v133 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[30:33], v[62:65], a[252:255], v1, v133 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[2:5], v[98:101], a[68:71], v0, v134 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[66:69], v180 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[2:5], v[102:105], a[80:83], v0, v134 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[70:73], v180 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[2:5], v[106:109], a[92:95], v0, v135 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[74:77], v180 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[2:5], v[110:113], a[104:107], v0, v135 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[78:81], v180 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[18:21], v[114:117], a[68:71], v0, v134 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[82:85], v186 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[18:21], v[118:121], a[80:83], v0, v134 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[86:89], v186 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[18:21], v[122:125], a[92:95], v0, v135 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[90:93], v186 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[18:21], v[126:129], a[104:107], v0, v135 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[94:97], v186 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[6:9], v[98:101], a[116:119], v0, v134 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[6:9], v[102:105], a[128:131], v0, v134 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[6:9], v[106:109], a[140:143], v0, v135 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[6:9], v[110:113], a[152:155], v0, v135 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[22:25], v[114:117], a[116:119], v0, v134 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[22:25], v[118:121], a[128:131], v0, v134 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[22:25], v[122:125], a[140:143], v0, v135 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[22:25], v[126:129], a[152:155], v0, v135 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[10:13], v[98:101], a[164:167], v1, v134 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[10:13], v[102:105], a[176:179], v1, v134 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[10:13], v[106:109], a[188:191], v1, v135 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[10:13], v[110:113], a[196:199], v1, v135 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[26:29], v[114:117], a[164:167], v1, v134 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[26:29], v[118:121], a[176:179], v1, v134 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[26:29], v[122:125], a[188:191], v1, v135 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[26:29], v[126:129], a[196:199], v1, v135 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[14:17], v[98:101], a[204:207], v1, v134 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[14:17], v[102:105], a[212:215], v1, v134 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[14:17], v[106:109], a[220:223], v1, v135 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[14:17], v[110:113], a[228:231], v1, v135 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[30:33], v[114:117], a[204:207], v1, v134 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[30:33], v[118:121], a[212:215], v1, v134 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[30:33], v[122:125], a[220:223], v1, v135 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[30:33], v[126:129], a[228:231], v1, v135 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(20)
s_barrier

	;;#ASMEND
	s_add_u32 s1, s1, 0x380
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27],  v[66:69], v[34:37], a[24:27],  v130, v132 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[2:5], v176 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[66:69], v[38:41], a[32:35],  v130, v132 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[6:9], v176 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[66:69], v[42:45], a[40:43],  v130, v133 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[10:13], v176 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[66:69], v[46:49], a[48:51],  v130, v133 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[14:17], v176 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27],  v[82:85], v[50:53], a[24:27],  v130, v132 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[18:21], v184 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[82:85], v[54:57], a[32:35],  v130, v132 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[22:25], v184 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[82:85], v[58:61], a[40:43],  v130, v133 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[26:29], v184 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[82:85], v[62:65], a[48:51],  v130, v133 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v184 offset:6144

	;;#ASMEND
	buffer_load_dwordx4 v166, s[20:23], s81 offen lds
	s_mov_b32 m0, s47
	s_nop 0
	buffer_load_dwordx4 v165, s[20:23], s81 offen lds
	s_mov_b32 m0, s52
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[70:73], v[34:37], a[56:59],  v130, v132 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[70:73], v[38:41], a[64:67],  v130, v132 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79],  v[70:73], v[42:45], a[76:79],  v130, v133 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91],  v[70:73], v[46:49], a[88:91],  v130, v133 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[86:89], v[50:53], a[56:59],  v130, v132 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[86:89], v[54:57], a[64:67],  v130, v132 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79],  v[86:89], v[58:61], a[76:79],  v130, v133 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91],  v[86:89], v[62:65], a[88:91],  v130, v133 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v167, s[20:23], s81 offen lds
	s_mov_b32 m0, s53
	s_nop 0
	buffer_load_dwordx4 v164, s[20:23], s81 offen lds
	s_mov_b32 m0, s40
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103],  v[74:77], v[34:37], a[100:103],  v131, v132 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115],  v[74:77], v[38:41], a[112:115],  v131, v132 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[74:77], v[42:45], a[124:127], v131, v133 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[74:77], v[46:49], a[136:139], v131, v133 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103],  v[90:93], v[50:53], a[100:103],  v131, v132 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115],  v[90:93], v[54:57], a[112:115],  v131, v132 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[90:93], v[58:61], a[124:127], v131, v133 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[90:93], v[62:65], a[136:139], v131, v133 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v166, s[20:23], s80 offen lds
	s_mov_b32 m0, s54
	s_nop 0
	buffer_load_dwordx4 v165, s[20:23], s80 offen lds
	s_mov_b32 m0, s31
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[78:81], v[34:37], a[148:151], v131, v132 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[78:81], v[38:41], a[160:163], v131, v132 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[78:81], v[42:45], a[172:175], v131, v133 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[78:81], v[46:49], a[184:187], v131, v133 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[94:97], v[50:53], a[148:151], v131, v132 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[94:97], v[54:57], a[160:163], v131, v132 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[94:97], v[58:61], a[172:175], v131, v133 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[94:97], v[62:65], a[184:187], v131, v133 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v167, s[20:23], s80 offen lds
	s_mov_b32 m0, s55
	s_nop 0
	buffer_load_dwordx4 v164, s[20:23], s80 offen lds
	s_mov_b32 m0, s42
	s_add_u32 s80, s29, 0x300
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3],  v[66:69], v[98:101], a[0:3],  v130, v134 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v182 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7],  v[66:69], v[102:105], a[4:7],  v130, v134 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v182 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11],  v[66:69], v[106:109], a[8:11],  v130, v135 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v182 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15],  v[66:69], v[110:113], a[12:15],  v130, v135 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v182 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3],  v[82:85], v[114:117], a[0:3],  v130, v134 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v185 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7],  v[82:85], v[118:121], a[4:7],  v130, v134 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v185 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11],  v[82:85], v[122:125], a[8:11],  v130, v135 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v185 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15],  v[82:85], v[126:129], a[12:15],  v130, v135 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[62:65], v185 offset:6144

	;;#ASMEND
	buffer_load_dwordx4 v168, s[24:27], s80 offen lds
	s_mov_b32 m0, s59
	s_nop 0
	buffer_load_dwordx4 v169, s[24:27], s80 offen lds
	s_mov_b32 m0, s60
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19],  v[70:73], v[98:101], a[16:19],  v130, v134 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23],  v[70:73], v[102:105], a[20:23],  v130, v134 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[70:73], v[106:109], a[28:31],  v130, v135 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[70:73], v[110:113], a[36:39],  v130, v135 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19],  v[86:89], v[114:117], a[16:19],  v130, v134 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23],  v[86:89], v[118:121], a[20:23],  v130, v134 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[86:89], v[122:125], a[28:31],  v130, v135 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[86:89], v[126:129], a[36:39],  v130, v135 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v179, s[24:27], s80 offen lds
	s_mov_b32 m0, s61
	s_nop 0
	buffer_load_dwordx4 v170, s[24:27], s80 offen lds
	s_mov_b32 m0, s44
	s_add_u32 s80, s79, 0x300
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[74:77], v[98:101], a[44:47],  v131, v134 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[74:77], v[102:105], a[52:55],  v131, v134 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[74:77], v[106:109], a[60:63], v131, v135 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[74:77], v[110:113], a[72:75], v131, v135 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[90:93], v[114:117], a[44:47],  v131, v134 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[90:93], v[118:121], a[52:55],  v131, v134 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[90:93], v[122:125], a[60:63], v131, v135 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[90:93], v[126:129], a[72:75], v131, v135 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	buffer_load_dwordx4 v168, s[24:27], s80 offen lds
	s_mov_b32 m0, s35
	s_nop 0
	buffer_load_dwordx4 v169, s[24:27], s80 offen lds
	s_mov_b32 m0, s62
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[78:81], v[98:101], a[84:87], v131, v134 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[78:81], v[102:105], a[96:99], v131, v134 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[78:81], v[106:109], a[108:111], v131, v135 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[78:81], v[110:113], a[120:123], v131, v135 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[94:97], v[114:117], a[84:87], v131, v134 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[94:97], v[118:121], a[96:99], v131, v134 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[94:97], v[122:125], a[108:111], v131, v135 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[94:97], v[126:129], a[120:123], v131, v135 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v179, s[24:27], s80 offen lds
	s_mov_b32 m0, s63
	s_nop 0
	buffer_load_dwordx4 v170, s[24:27], s80 offen lds
	s_mov_b32 m0, s39
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_add_i32 s80, s77, 0xfffffa00
	;;#ASMSTART
	buffer_load_dwordx2 v[0:1], v177, s[4:7], s80 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[132:133], v177, s[12:15], s80 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[134:135], v177, s[16:19], s80 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[130:131], v177, s[8:11], s80 offen
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[2:5], v[34:37], a[132:135],  v0, v132 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[98:101], v190 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[2:5], v[38:41], a[144:147],  v0, v132 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[102:105], v190 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[2:5], v[42:45], a[156:159],  v0, v133 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[106:109], v190 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171],  v[2:5], v[46:49], a[168:171],  v0, v133 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[110:113], v190 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[18:21], v[50:53], a[132:135],  v0, v132 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[114:117], v191 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[18:21], v[54:57], a[144:147],  v0, v132 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[118:121], v191 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[18:21], v[58:61], a[156:159],  v0, v133 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[122:125], v191 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171],  v[18:21], v[62:65], a[168:171],  v0, v133 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[126:129], v191 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183],  v[6:9], v[34:37], a[180:183],  v0, v132 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[6:9], v[38:41], a[192:195],  v0, v132 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[6:9], v[42:45], a[200:203],  v0, v133 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[6:9], v[46:49], a[208:211],  v0, v133 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183],  v[22:25], v[50:53], a[180:183],  v0, v132 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[22:25], v[54:57], a[192:195],  v0, v132 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[22:25], v[58:61], a[200:203],  v0, v133 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[22:25], v[62:65], a[208:211],  v0, v133 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[10:13], v[34:37], a[216:219],  v1, v132 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[10:13], v[38:41], a[224:227],  v1, v132 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[10:13], v[42:45], a[232:235], v1, v133 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[10:13], v[46:49], a[236:239], v1, v133 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[26:29], v[50:53], a[216:219],  v1, v132 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[26:29], v[54:57], a[224:227],  v1, v132 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[26:29], v[58:61], a[232:235], v1, v133 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[26:29], v[62:65], a[236:239], v1, v133 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[14:17], v[34:37], a[240:243], v1, v132 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[14:17], v[38:41], a[244:247], v1, v132 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[14:17], v[42:45], a[248:251], v1, v133 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[14:17], v[46:49], a[252:255], v1, v133 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[30:33], v[50:53], a[240:243], v1, v132 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[30:33], v[54:57], a[244:247], v1, v132 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[30:33], v[58:61], a[248:251], v1, v133 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[30:33], v[62:65], a[252:255], v1, v133 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[2:5], v[98:101], a[68:71], v0, v134 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[66:69], v181 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[2:5], v[102:105], a[80:83], v0, v134 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[70:73], v181 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[2:5], v[106:109], a[92:95], v0, v135 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[74:77], v181 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[2:5], v[110:113], a[104:107], v0, v135 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[78:81], v181 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[18:21], v[114:117], a[68:71], v0, v134 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[82:85], v187 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[18:21], v[118:121], a[80:83], v0, v134 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[86:89], v187 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[18:21], v[122:125], a[92:95], v0, v135 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[90:93], v187 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[18:21], v[126:129], a[104:107], v0, v135 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[94:97], v187 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[6:9], v[98:101], a[116:119], v0, v134 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[6:9], v[102:105], a[128:131], v0, v134 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[6:9], v[106:109], a[140:143], v0, v135 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[6:9], v[110:113], a[152:155], v0, v135 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[22:25], v[114:117], a[116:119], v0, v134 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[22:25], v[118:121], a[128:131], v0, v134 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[22:25], v[122:125], a[140:143], v0, v135 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[22:25], v[126:129], a[152:155], v0, v135 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[10:13], v[98:101], a[164:167], v1, v134 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[10:13], v[102:105], a[176:179], v1, v134 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[10:13], v[106:109], a[188:191], v1, v135 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[10:13], v[110:113], a[196:199], v1, v135 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[26:29], v[114:117], a[164:167], v1, v134 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[26:29], v[118:121], a[176:179], v1, v134 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[26:29], v[122:125], a[188:191], v1, v135 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[26:29], v[126:129], a[196:199], v1, v135 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[14:17], v[98:101], a[204:207], v1, v134 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[14:17], v[102:105], a[212:215], v1, v134 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[14:17], v[106:109], a[220:223], v1, v135 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[14:17], v[110:113], a[228:231], v1, v135 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[30:33], v[114:117], a[204:207], v1, v134 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[30:33], v[118:121], a[212:215], v1, v134 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[30:33], v[122:125], a[220:223], v1, v135 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[30:33], v[126:129], a[228:231], v1, v135 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(20)
s_barrier

	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27],  v[66:69], v[34:37], a[24:27],  v130, v132 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[2:5], v171 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[66:69], v[38:41], a[32:35],  v130, v132 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[6:9], v171 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[66:69], v[42:45], a[40:43],  v130, v133 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[10:13], v171 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[66:69], v[46:49], a[48:51],  v130, v133 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[14:17], v171 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27],  v[82:85], v[50:53], a[24:27],  v130, v132 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[18:21], v183 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[82:85], v[54:57], a[32:35],  v130, v132 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[22:25], v183 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[82:85], v[58:61], a[40:43],  v130, v133 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[26:29], v183 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[82:85], v[62:65], a[48:51],  v130, v133 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v183 offset:6144

	;;#ASMEND
	buffer_load_dwordx4 v166, s[20:23], s78 offen lds
	s_mov_b32 m0, s64
	s_nop 0
	buffer_load_dwordx4 v165, s[20:23], s78 offen lds
	s_mov_b32 m0, s65
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[70:73], v[34:37], a[56:59],  v130, v132 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[70:73], v[38:41], a[64:67],  v130, v132 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79],  v[70:73], v[42:45], a[76:79],  v130, v133 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91],  v[70:73], v[46:49], a[88:91],  v130, v133 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[86:89], v[50:53], a[56:59],  v130, v132 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[86:89], v[54:57], a[64:67],  v130, v132 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79],  v[86:89], v[58:61], a[76:79],  v130, v133 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91],  v[86:89], v[62:65], a[88:91],  v130, v133 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v167, s[20:23], s78 offen lds
	s_mov_b32 m0, s66
	s_nop 0
	buffer_load_dwordx4 v164, s[20:23], s78 offen lds
	s_mov_b32 m0, s41
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103],  v[74:77], v[34:37], a[100:103],  v131, v132 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115],  v[74:77], v[38:41], a[112:115],  v131, v132 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[74:77], v[42:45], a[124:127], v131, v133 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[74:77], v[46:49], a[136:139], v131, v133 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103],  v[90:93], v[50:53], a[100:103],  v131, v132 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115],  v[90:93], v[54:57], a[112:115],  v131, v132 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[90:93], v[58:61], a[124:127], v131, v133 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[90:93], v[62:65], a[136:139], v131, v133 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v166, s[20:23], s1 offen lds
	s_mov_b32 m0, s67
	s_nop 0
	buffer_load_dwordx4 v165, s[20:23], s1 offen lds
	s_mov_b32 m0, s68
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[78:81], v[34:37], a[148:151], v131, v132 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[78:81], v[38:41], a[160:163], v131, v132 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[78:81], v[42:45], a[172:175], v131, v133 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[78:81], v[46:49], a[184:187], v131, v133 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[94:97], v[50:53], a[148:151], v131, v132 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[94:97], v[54:57], a[160:163], v131, v132 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[94:97], v[58:61], a[172:175], v131, v133 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[94:97], v[62:65], a[184:187], v131, v133 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v167, s[20:23], s1 offen lds
	s_mov_b32 m0, s69
	s_nop 0
	buffer_load_dwordx4 v164, s[20:23], s1 offen lds
	s_mov_b32 m0, s43
	s_add_u32 s1, s29, 0x380
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3],  v[66:69], v[98:101], a[0:3],  v130, v134 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v192 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7],  v[66:69], v[102:105], a[4:7],  v130, v134 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v192 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11],  v[66:69], v[106:109], a[8:11],  v130, v135 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v192 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15],  v[66:69], v[110:113], a[12:15],  v130, v135 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v192 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3],  v[82:85], v[114:117], a[0:3],  v130, v134 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v193 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7],  v[82:85], v[118:121], a[4:7],  v130, v134 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v193 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11],  v[82:85], v[122:125], a[8:11],  v130, v135 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v193 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15],  v[82:85], v[126:129], a[12:15],  v130, v135 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[62:65], v193 offset:6144

	;;#ASMEND
	buffer_load_dwordx4 v168, s[24:27], s1 offen lds
	s_mov_b32 m0, s70
	s_add_i32 s29, s77, 0xfffffc00
	buffer_load_dwordx4 v169, s[24:27], s1 offen lds
	s_mov_b32 m0, s71
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19],  v[70:73], v[98:101], a[16:19],  v130, v134 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23],  v[70:73], v[102:105], a[20:23],  v130, v134 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[70:73], v[106:109], a[28:31],  v130, v135 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[70:73], v[110:113], a[36:39],  v130, v135 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19],  v[86:89], v[114:117], a[16:19],  v130, v134 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23],  v[86:89], v[118:121], a[20:23],  v130, v134 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[86:89], v[122:125], a[28:31],  v130, v135 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[86:89], v[126:129], a[36:39],  v130, v135 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[0:1], v177, s[4:7], s29 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[162:163], v177, s[16:19], s29 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[172:173], v177, s[8:11], s29 offen
	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v179, s[24:27], s1 offen lds
	s_mov_b32 m0, s72
	s_nop 0
	buffer_load_dwordx4 v170, s[24:27], s1 offen lds
	s_mov_b32 m0, s45
	s_add_u32 s1, s79, 0x380
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[74:77], v[98:101], a[44:47],  v131, v134 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[74:77], v[102:105], a[52:55],  v131, v134 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[74:77], v[106:109], a[60:63], v131, v135 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[74:77], v[110:113], a[72:75], v131, v135 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[90:93], v[114:117], a[44:47],  v131, v134 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[90:93], v[118:121], a[52:55],  v131, v134 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[90:93], v[122:125], a[60:63], v131, v135 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[90:93], v[126:129], a[72:75], v131, v135 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	buffer_load_dwordx4 v168, s[24:27], s1 offen lds
	s_mov_b32 m0, s73
	s_nop 0
	buffer_load_dwordx4 v169, s[24:27], s1 offen lds
	s_mov_b32 m0, s74
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[78:81], v[98:101], a[84:87], v131, v134 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[78:81], v[102:105], a[96:99], v131, v134 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[78:81], v[106:109], a[108:111], v131, v135 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[78:81], v[110:113], a[120:123], v131, v135 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[94:97], v[114:117], a[84:87], v131, v134 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[94:97], v[118:121], a[96:99], v131, v134 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[94:97], v[122:125], a[108:111], v131, v135 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[94:97], v[126:129], a[120:123], v131, v135 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[130:131], v177, s[12:15], s29 offen
	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v179, s[24:27], s1 offen lds
	s_mov_b32 m0, s75
	s_nop 0
	buffer_load_dwordx4 v170, s[24:27], s1 offen lds
	s_add_i32 s1, s76, -1
	s_min_u32 s1, s1, 0x6d
	s_lshl_b32 s1, s1, 7
	s_add_u32 s78, s49, s1
	s_mov_b32 m0, s38
	s_sub_u32 s78, s78, s30
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[2:5], v[34:37], a[132:135],  v0, v130 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[98:101], v188 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[2:5], v[38:41], a[144:147],  v0, v130 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[102:105], v188 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[2:5], v[42:45], a[156:159],  v0, v131 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[106:109], v188 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171],  v[2:5], v[46:49], a[168:171],  v0, v131 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[110:113], v188 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[18:21], v[50:53], a[132:135],  v0, v130 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[114:117], v189 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[18:21], v[54:57], a[144:147],  v0, v130 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[118:121], v189 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[18:21], v[58:61], a[156:159],  v0, v131 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[122:125], v189 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171],  v[18:21], v[62:65], a[168:171],  v0, v131 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[126:129], v189 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183],  v[6:9], v[34:37], a[180:183],  v0, v130 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[6:9], v[38:41], a[192:195],  v0, v130 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[6:9], v[42:45], a[200:203],  v0, v131 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[6:9], v[46:49], a[208:211],  v0, v131 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183],  v[22:25], v[50:53], a[180:183],  v0, v130 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[22:25], v[54:57], a[192:195],  v0, v130 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[22:25], v[58:61], a[200:203],  v0, v131 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[22:25], v[62:65], a[208:211],  v0, v131 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[10:13], v[34:37], a[216:219],  v1, v130 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[10:13], v[38:41], a[224:227],  v1, v130 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[10:13], v[42:45], a[232:235], v1, v131 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[10:13], v[46:49], a[236:239], v1, v131 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[26:29], v[50:53], a[216:219],  v1, v130 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[26:29], v[54:57], a[224:227],  v1, v130 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[26:29], v[58:61], a[232:235], v1, v131 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[26:29], v[62:65], a[236:239], v1, v131 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[14:17], v[34:37], a[240:243], v1, v130 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[14:17], v[38:41], a[244:247], v1, v130 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[14:17], v[42:45], a[248:251], v1, v131 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[14:17], v[46:49], a[252:255], v1, v131 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[30:33], v[50:53], a[240:243], v1, v130 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[30:33], v[54:57], a[244:247], v1, v130 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[30:33], v[58:61], a[248:251], v1, v131 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[30:33], v[62:65], a[252:255], v1, v131 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[2:5], v[98:101], a[68:71], v0, v162 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[66:69], v180 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[2:5], v[102:105], a[80:83], v0, v162 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[70:73], v180 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[2:5], v[106:109], a[92:95], v0, v163 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[74:77], v180 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[2:5], v[110:113], a[104:107], v0, v163 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[78:81], v180 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[18:21], v[114:117], a[68:71], v0, v162 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[82:85], v186 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[18:21], v[118:121], a[80:83], v0, v162 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[86:89], v186 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[18:21], v[122:125], a[92:95], v0, v163 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[90:93], v186 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[18:21], v[126:129], a[104:107], v0, v163 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[94:97], v186 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[6:9], v[98:101], a[116:119], v0, v162 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[6:9], v[102:105], a[128:131], v0, v162 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[6:9], v[106:109], a[140:143], v0, v163 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[6:9], v[110:113], a[152:155], v0, v163 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[22:25], v[114:117], a[116:119], v0, v162 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[22:25], v[118:121], a[128:131], v0, v162 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[22:25], v[122:125], a[140:143], v0, v163 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[22:25], v[126:129], a[152:155], v0, v163 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[10:13], v[98:101], a[164:167], v1, v162 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[10:13], v[102:105], a[176:179], v1, v162 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[10:13], v[106:109], a[188:191], v1, v163 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[10:13], v[110:113], a[196:199], v1, v163 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[26:29], v[114:117], a[164:167], v1, v162 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[26:29], v[118:121], a[176:179], v1, v162 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[26:29], v[122:125], a[188:191], v1, v163 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[26:29], v[126:129], a[196:199], v1, v163 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[14:17], v[98:101], a[204:207], v1, v162 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[14:17], v[102:105], a[212:215], v1, v162 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[14:17], v[106:109], a[220:223], v1, v163 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[14:17], v[110:113], a[228:231], v1, v163 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[30:33], v[114:117], a[204:207], v1, v162 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[30:33], v[118:121], a[212:215], v1, v162 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[30:33], v[122:125], a[220:223], v1, v163 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[30:33], v[126:129], a[228:231], v1, v163 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(20)
s_barrier

	;;#ASMEND
	s_add_u32 s29, s51, s1
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27],  v[66:69], v[34:37], a[24:27],  v172, v130 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[2:5], v176 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[66:69], v[38:41], a[32:35],  v172, v130 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[6:9], v176 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[66:69], v[42:45], a[40:43],  v172, v131 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[10:13], v176 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[66:69], v[46:49], a[48:51],  v172, v131 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[14:17], v176 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27],  v[82:85], v[50:53], a[24:27],  v172, v130 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[18:21], v184 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[82:85], v[54:57], a[32:35],  v172, v130 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[22:25], v184 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[82:85], v[58:61], a[40:43],  v172, v131 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[26:29], v184 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[82:85], v[62:65], a[48:51],  v172, v131 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v184 offset:6144

	;;#ASMEND
	buffer_load_dwordx4 v166, s[20:23], s78 offen lds
	s_mov_b32 m0, s47
	s_sub_u32 s29, s29, s30
	buffer_load_dwordx4 v165, s[20:23], s78 offen lds
	s_mov_b32 m0, s52
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[70:73], v[34:37], a[56:59],  v172, v130 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[70:73], v[38:41], a[64:67],  v172, v130 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79],  v[70:73], v[42:45], a[76:79],  v172, v131 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91],  v[70:73], v[46:49], a[88:91],  v172, v131 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[86:89], v[50:53], a[56:59],  v172, v130 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[86:89], v[54:57], a[64:67],  v172, v130 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79],  v[86:89], v[58:61], a[76:79],  v172, v131 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91],  v[86:89], v[62:65], a[88:91],  v172, v131 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v167, s[20:23], s78 offen lds
	s_mov_b32 m0, s53
	s_nop 0
	buffer_load_dwordx4 v164, s[20:23], s78 offen lds
	s_mov_b32 m0, s40
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103],  v[74:77], v[34:37], a[100:103],  v173, v130 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115],  v[74:77], v[38:41], a[112:115],  v173, v130 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[74:77], v[42:45], a[124:127], v173, v131 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[74:77], v[46:49], a[136:139], v173, v131 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103],  v[90:93], v[50:53], a[100:103],  v173, v130 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115],  v[90:93], v[54:57], a[112:115],  v173, v130 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[90:93], v[58:61], a[124:127], v173, v131 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[90:93], v[62:65], a[136:139], v173, v131 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v166, s[20:23], s29 offen lds
	s_mov_b32 m0, s54
	s_nop 0
	buffer_load_dwordx4 v165, s[20:23], s29 offen lds
	s_mov_b32 m0, s31
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[78:81], v[34:37], a[148:151], v173, v130 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[78:81], v[38:41], a[160:163], v173, v130 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[78:81], v[42:45], a[172:175], v173, v131 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[78:81], v[46:49], a[184:187], v173, v131 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[94:97], v[50:53], a[148:151], v173, v130 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[94:97], v[54:57], a[160:163], v173, v130 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[94:97], v[58:61], a[172:175], v173, v131 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[94:97], v[62:65], a[184:187], v173, v131 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v167, s[20:23], s29 offen lds
	s_mov_b32 m0, s55
	s_nop 0
	buffer_load_dwordx4 v164, s[20:23], s29 offen lds
	s_add_u32 s29, s56, s1
	s_mov_b32 m0, s42
	s_sub_u32 s29, s29, s34
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3],  v[66:69], v[98:101], a[0:3],  v172, v162 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[130:133], v182 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7],  v[66:69], v[102:105], a[4:7],  v172, v162 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[134:137], v182 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11],  v[66:69], v[106:109], a[8:11],  v172, v163 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[138:141], v182 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15],  v[66:69], v[110:113], a[12:15],  v172, v163 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[142:145], v182 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3],  v[82:85], v[114:117], a[0:3],  v172, v162 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[146:149], v185 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7],  v[82:85], v[118:121], a[4:7],  v172, v162 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[150:153], v185 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11],  v[82:85], v[122:125], a[8:11],  v172, v163 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[154:157], v185 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15],  v[82:85], v[126:129], a[12:15],  v172, v163 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[158:161], v185 offset:6144

	;;#ASMEND
	buffer_load_dwordx4 v168, s[24:27], s29 offen lds
	s_mov_b32 m0, s59
	s_add_u32 s1, s58, s1
	buffer_load_dwordx4 v169, s[24:27], s29 offen lds
	s_mov_b32 m0, s60
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19],  v[70:73], v[98:101], a[16:19],  v172, v162 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23],  v[70:73], v[102:105], a[20:23],  v172, v162 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[70:73], v[106:109], a[28:31],  v172, v163 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[70:73], v[110:113], a[36:39],  v172, v163 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19],  v[86:89], v[114:117], a[16:19],  v172, v162 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23],  v[86:89], v[118:121], a[20:23],  v172, v162 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[86:89], v[122:125], a[28:31],  v172, v163 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[86:89], v[126:129], a[36:39],  v172, v163 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_sub_u32 s1, s1, s34
	buffer_load_dwordx4 v179, s[24:27], s29 offen lds
	s_mov_b32 m0, s61
	s_nop 0
	buffer_load_dwordx4 v170, s[24:27], s29 offen lds
	s_mov_b32 m0, s44
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[74:77], v[98:101], a[44:47],  v173, v162 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[74:77], v[102:105], a[52:55],  v173, v162 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[74:77], v[106:109], a[60:63], v173, v163 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[74:77], v[110:113], a[72:75], v173, v163 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[90:93], v[114:117], a[44:47],  v173, v162 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[90:93], v[118:121], a[52:55],  v173, v162 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[90:93], v[122:125], a[60:63], v173, v163 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[90:93], v[126:129], a[72:75], v173, v163 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_add_i32 s29, s77, 0xfffffe00
	buffer_load_dwordx4 v168, s[24:27], s1 offen lds
	s_mov_b32 m0, s35
	;;#ASMSTART
	buffer_load_dwordx2 v[0:1], v177, s[4:7], s29 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[174:175], v177, s[8:11], s29 offen
	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v169, s[24:27], s1 offen lds
	s_mov_b32 m0, s62
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[78:81], v[98:101], a[84:87], v173, v162 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[78:81], v[102:105], a[96:99], v173, v162 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[78:81], v[106:109], a[108:111], v173, v163 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[78:81], v[110:113], a[120:123], v173, v163 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[94:97], v[114:117], a[84:87], v173, v162 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[94:97], v[118:121], a[96:99], v173, v162 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[94:97], v[122:125], a[108:111], v173, v163 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[94:97], v[126:129], a[120:123], v173, v163 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[162:163], v177, s[12:15], s29 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[172:173], v177, s[16:19], s29 offen
	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v179, s[24:27], s1 offen lds
	s_mov_b32 m0, s63
	s_nop 0
	buffer_load_dwordx4 v170, s[24:27], s1 offen lds
	s_min_u32 s1, s76, 0x6d
	s_lshl_b32 s1, s1, 7
	s_add_u32 s78, s49, s1
	s_mov_b32 m0, s39
	s_sub_u32 s78, s78, s30
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[2:5], v[130:133], a[132:135],  v0, v162 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[98:101], v190 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[2:5], v[134:137], a[144:147],  v0, v162 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[102:105], v190 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[2:5], v[138:141], a[156:159],  v0, v163 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[106:109], v190 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171],  v[2:5], v[142:145], a[168:171],  v0, v163 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[110:113], v190 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135],  v[18:21], v[146:149], a[132:135],  v0, v162 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[114:117], v191 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147],  v[18:21], v[150:153], a[144:147],  v0, v162 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[118:121], v191 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159],  v[18:21], v[154:157], a[156:159],  v0, v163 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[122:125], v191 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171],  v[18:21], v[158:161], a[168:171],  v0, v163 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[126:129], v191 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183],  v[6:9], v[130:133], a[180:183],  v0, v162 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[6:9], v[134:137], a[192:195],  v0, v162 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[6:9], v[138:141], a[200:203],  v0, v163 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[6:9], v[142:145], a[208:211],  v0, v163 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183],  v[22:25], v[146:149], a[180:183],  v0, v162 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195],  v[22:25], v[150:153], a[192:195],  v0, v162 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203],  v[22:25], v[154:157], a[200:203],  v0, v163 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211],  v[22:25], v[158:161], a[208:211],  v0, v163 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[10:13], v[130:133], a[216:219],  v1, v162 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[10:13], v[134:137], a[224:227],  v1, v162 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[10:13], v[138:141], a[232:235], v1, v163 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[10:13], v[142:145], a[236:239], v1, v163 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219],  v[26:29], v[146:149], a[216:219],  v1, v162 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227],  v[26:29], v[150:153], a[224:227],  v1, v162 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[26:29], v[154:157], a[232:235], v1, v163 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[26:29], v[158:161], a[236:239], v1, v163 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[14:17], v[130:133], a[240:243], v1, v162 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[14:17], v[134:137], a[244:247], v1, v162 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[14:17], v[138:141], a[248:251], v1, v163 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[14:17], v[142:145], a[252:255], v1, v163 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[30:33], v[146:149], a[240:243], v1, v162 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[30:33], v[150:153], a[244:247], v1, v162 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[30:33], v[154:157], a[248:251], v1, v163 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[30:33], v[158:161], a[252:255], v1, v163 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[2:5], v[98:101], a[68:71], v0, v172 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[66:69], v181 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[2:5], v[102:105], a[80:83], v0, v172 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[70:73], v181 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[2:5], v[106:109], a[92:95], v0, v173 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[74:77], v181 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[2:5], v[110:113], a[104:107], v0, v173 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[78:81], v181 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[18:21], v[114:117], a[68:71], v0, v172 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[82:85], v187 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[18:21], v[118:121], a[80:83], v0, v172 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[86:89], v187 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[18:21], v[122:125], a[92:95], v0, v173 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[90:93], v187 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[18:21], v[126:129], a[104:107], v0, v173 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[94:97], v187 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[6:9], v[98:101], a[116:119], v0, v172 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[6:9], v[102:105], a[128:131], v0, v172 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[6:9], v[106:109], a[140:143], v0, v173 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[6:9], v[110:113], a[152:155], v0, v173 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[22:25], v[114:117], a[116:119], v0, v172 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[22:25], v[118:121], a[128:131], v0, v172 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[22:25], v[122:125], a[140:143], v0, v173 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[22:25], v[126:129], a[152:155], v0, v173 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[10:13], v[98:101], a[164:167], v1, v172 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[10:13], v[102:105], a[176:179], v1, v172 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[10:13], v[106:109], a[188:191], v1, v173 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[10:13], v[110:113], a[196:199], v1, v173 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[26:29], v[114:117], a[164:167], v1, v172 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[26:29], v[118:121], a[176:179], v1, v172 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[26:29], v[122:125], a[188:191], v1, v173 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[26:29], v[126:129], a[196:199], v1, v173 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[14:17], v[98:101], a[204:207], v1, v172 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[14:17], v[102:105], a[212:215], v1, v172 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[14:17], v[106:109], a[220:223], v1, v173 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[14:17], v[110:113], a[228:231], v1, v173 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[30:33], v[114:117], a[204:207], v1, v172 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[30:33], v[118:121], a[212:215], v1, v172 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[30:33], v[122:125], a[220:223], v1, v173 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[30:33], v[126:129], a[228:231], v1, v173 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(20)
s_barrier

	;;#ASMEND
	s_add_u32 s29, s51, s1
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27],  v[66:69], v[130:133], a[24:27],  v174, v162 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[34:37], v171 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[66:69], v[134:137], a[32:35],  v174, v162 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[38:41], v171 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[66:69], v[138:141], a[40:43],  v174, v163 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[42:45], v171 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[66:69], v[142:145], a[48:51],  v174, v163 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[46:49], v171 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27],  v[82:85], v[146:149], a[24:27],  v174, v162 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[50:53], v183 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35],  v[82:85], v[150:153], a[32:35],  v174, v162 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[54:57], v183 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43],  v[82:85], v[154:157], a[40:43],  v174, v163 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[58:61], v183 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51],  v[82:85], v[158:161], a[48:51],  v174, v163 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[62:65], v183 offset:6144

	;;#ASMEND
	buffer_load_dwordx4 v166, s[20:23], s78 offen lds
	s_mov_b32 m0, s64
	s_sub_u32 s29, s29, s30
	buffer_load_dwordx4 v165, s[20:23], s78 offen lds
	s_mov_b32 m0, s65
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[70:73], v[130:133], a[56:59],  v174, v162 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[70:73], v[134:137], a[64:67],  v174, v162 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79],  v[70:73], v[138:141], a[76:79],  v174, v163 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91],  v[70:73], v[142:145], a[88:91],  v174, v163 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59],  v[86:89], v[146:149], a[56:59],  v174, v162 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67],  v[86:89], v[150:153], a[64:67],  v174, v162 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79],  v[86:89], v[154:157], a[76:79],  v174, v163 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91],  v[86:89], v[158:161], a[88:91],  v174, v163 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v167, s[20:23], s78 offen lds
	s_mov_b32 m0, s66
	s_nop 0
	buffer_load_dwordx4 v164, s[20:23], s78 offen lds
	s_mov_b32 m0, s41
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103],  v[74:77], v[130:133], a[100:103],  v175, v162 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115],  v[74:77], v[134:137], a[112:115],  v175, v162 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[74:77], v[138:141], a[124:127], v175, v163 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[74:77], v[142:145], a[136:139], v175, v163 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103],  v[90:93], v[146:149], a[100:103],  v175, v162 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115],  v[90:93], v[150:153], a[112:115],  v175, v162 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[90:93], v[154:157], a[124:127], v175, v163 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[90:93], v[158:161], a[136:139], v175, v163 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v166, s[20:23], s29 offen lds
	s_mov_b32 m0, s67
	s_nop 0
	buffer_load_dwordx4 v165, s[20:23], s29 offen lds
	s_mov_b32 m0, s68
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[78:81], v[130:133], a[148:151], v175, v162 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[78:81], v[134:137], a[160:163], v175, v162 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[78:81], v[138:141], a[172:175], v175, v163 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[78:81], v[142:145], a[184:187], v175, v163 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[94:97], v[146:149], a[148:151], v175, v162 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[94:97], v[150:153], a[160:163], v175, v162 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[94:97], v[154:157], a[172:175], v175, v163 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[94:97], v[158:161], a[184:187], v175, v163 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_nop 0
	buffer_load_dwordx4 v167, s[20:23], s29 offen lds
	s_mov_b32 m0, s69
	s_nop 0
	buffer_load_dwordx4 v164, s[20:23], s29 offen lds
	s_add_u32 s29, s56, s1
	s_mov_b32 m0, s43
	s_sub_u32 s29, s29, s34
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3],  v[66:69], v[98:101], a[0:3],  v174, v172 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[2:5], v192 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7],  v[66:69], v[102:105], a[4:7],  v174, v172 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[6:9], v192 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11],  v[66:69], v[106:109], a[8:11],  v174, v173 op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[10:13], v192 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15],  v[66:69], v[110:113], a[12:15],  v174, v173 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
ds_read_b128 v[14:17], v192 offset:6144
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3],  v[82:85], v[114:117], a[0:3],  v174, v172 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[18:21], v193 offset:0
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7],  v[82:85], v[118:121], a[4:7],  v174, v172 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[22:25], v193 offset:2048
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11],  v[82:85], v[122:125], a[8:11],  v174, v173 op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[26:29], v193 offset:4096
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15],  v[82:85], v[126:129], a[12:15],  v174, v173 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
ds_read_b128 v[30:33], v193 offset:6144

	;;#ASMEND
	buffer_load_dwordx4 v168, s[24:27], s29 offen lds
	s_mov_b32 m0, s70
	s_add_u32 s1, s58, s1
	buffer_load_dwordx4 v169, s[24:27], s29 offen lds
	s_mov_b32 m0, s71
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19],  v[70:73], v[98:101], a[16:19],  v174, v172 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23],  v[70:73], v[102:105], a[20:23],  v174, v172 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[70:73], v[106:109], a[28:31],  v174, v173 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[70:73], v[110:113], a[36:39],  v174, v173 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19],  v[86:89], v[114:117], a[16:19],  v174, v172 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23],  v[86:89], v[118:121], a[20:23],  v174, v172 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31],  v[86:89], v[122:125], a[28:31],  v174, v173 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39],  v[86:89], v[126:129], a[36:39],  v174, v173 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_sub_u32 s1, s1, s34
	buffer_load_dwordx4 v179, s[24:27], s29 offen lds
	s_mov_b32 m0, s72
	s_cmpk_lg_i32 s36, 0x3400
	buffer_load_dwordx4 v170, s[24:27], s29 offen lds
	s_mov_b32 m0, s45
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[74:77], v[98:101], a[44:47],  v175, v172 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[74:77], v[102:105], a[52:55],  v175, v172 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[74:77], v[106:109], a[60:63], v175, v173 op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[74:77], v[110:113], a[72:75], v175, v173 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47],  v[90:93], v[114:117], a[44:47],  v175, v172 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55],  v[90:93], v[118:121], a[52:55],  v175, v172 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[90:93], v[122:125], a[60:63], v175, v173 op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[90:93], v[126:129], a[72:75], v175, v173 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_cselect_b32 s29, s77, 0xde00
	buffer_load_dwordx4 v168, s[24:27], s1 offen lds
	s_mov_b32 m0, s73
	s_add_u32 s36, s36, 0x400
	buffer_load_dwordx4 v169, s[24:27], s1 offen lds
	s_mov_b32 m0, s74
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[78:81], v[98:101], a[84:87], v175, v172 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[78:81], v[102:105], a[96:99], v175, v172 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[78:81], v[106:109], a[108:111], v175, v173 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[78:81], v[110:113], a[120:123], v175, v173 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[94:97], v[114:117], a[84:87], v175, v172 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[94:97], v[118:121], a[96:99], v175, v172 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[94:97], v[122:125], a[108:111], v175, v173 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[94:97], v[126:129], a[120:123], v175, v173 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4

	;;#ASMEND
	s_addc_u32 s37, s37, 0
	buffer_load_dwordx4 v179, s[24:27], s1 offen lds
	s_mov_b32 m0, s75
	s_addk_i32 s77, 0x1000
	buffer_load_dwordx4 v170, s[24:27], s1 offen lds
	s_add_i32 s76, s76, 8
	s_cmpk_eq_i32 s36, 0x3800
	;;#ASMSTART
	buffer_load_dwordx2 v[136:137], v177, s[4:7], s29 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[130:131], v177, s[8:11], s29 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[134:135], v177, s[12:15], s29 offen
	;;#ASMEND
	;;#ASMSTART
	buffer_load_dwordx2 v[132:133], v177, s[16:19], s29 offen
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_cbranch_scc0 .LBB0_7
; %bb.8:
	v_accvgpr_read_b32 v151, a111
	v_accvgpr_read_b32 v150, a110
	v_pk_mul_f32 v[0:1], v[150:151], s[0:1] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v153, a99
	v_accvgpr_read_b32 v149, a109
	v_accvgpr_read_b32 v148, a108
	v_accvgpr_write_b32 a109, v1
	v_accvgpr_read_b32 v152, a98
	v_accvgpr_write_b32 a108, v0
	v_pk_mul_f32 v[0:1], v[152:153], s[0:1] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v155, a87
	v_accvgpr_read_b32 v151, a97
	v_accvgpr_read_b32 v150, a96
	v_accvgpr_write_b32 a97, v1
	v_accvgpr_read_b32 v154, a86
	v_accvgpr_write_b32 a96, v0
	v_pk_mul_f32 v[0:1], v[154:155], s[0:1] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v189, a75
	v_accvgpr_read_b32 v153, a85
	v_accvgpr_read_b32 v152, a84
	v_accvgpr_write_b32 a85, v1
	v_accvgpr_read_b32 v188, a74
	v_accvgpr_write_b32 a84, v0
	v_pk_mul_f32 v[0:1], v[188:189], s[0:1] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v191, a63
	v_accvgpr_read_b32 v187, a73
	v_accvgpr_read_b32 v186, a72
	v_accvgpr_write_b32 a73, v1
	v_accvgpr_read_b32 v190, a62
	v_accvgpr_write_b32 a72, v0
	v_pk_mul_f32 v[0:1], v[190:191], s[0:1] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v139, a55
	v_accvgpr_read_b32 v189, a61
	v_accvgpr_read_b32 v188, a60
	v_accvgpr_write_b32 a61, v1
	v_accvgpr_read_b32 v138, a54
	v_accvgpr_write_b32 a60, v0
	v_pk_mul_f32 v[0:1], v[138:139], s[0:1] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v143, a47
	v_accvgpr_read_b32 v137, a53
	v_accvgpr_read_b32 v136, a52
	v_accvgpr_write_b32 a53, v1
	v_accvgpr_read_b32 v142, a46
	v_accvgpr_write_b32 a52, v0
	v_pk_mul_f32 v[0:1], v[142:143], s[0:1] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v219, a39
	v_accvgpr_read_b32 v141, a45
	v_accvgpr_read_b32 v140, a44
	v_accvgpr_write_b32 a45, v1
	v_accvgpr_read_b32 v218, a38
	v_accvgpr_write_b32 a44, v0
	v_pk_mul_f32 v[0:1], v[218:219], s[0:1] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v145, a31
	v_accvgpr_read_b32 v217, a37
	v_accvgpr_read_b32 v216, a36
	v_accvgpr_write_b32 a37, v1
	v_accvgpr_read_b32 v144, a30
	v_accvgpr_write_b32 a36, v0
	v_pk_mul_f32 v[0:1], v[144:145], s[0:1] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v125, a23
	v_accvgpr_read_b32 v143, a29
	v_accvgpr_read_b32 v142, a28
	v_accvgpr_write_b32 a29, v1
	v_accvgpr_read_b32 v124, a22
	v_accvgpr_write_b32 a28, v0
	v_pk_mul_f32 v[0:1], v[124:125], s[0:1] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v177, a19
	v_accvgpr_read_b32 v123, a21
	v_accvgpr_read_b32 v122, a20
	v_accvgpr_write_b32 a21, v1
	v_accvgpr_read_b32 v176, a18
	v_accvgpr_write_b32 a20, v0
	v_pk_mul_f32 v[0:1], v[176:177], s[0:1] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v185, a15
	v_accvgpr_read_b32 v175, a17
	v_accvgpr_read_b32 v174, a16
	v_accvgpr_write_b32 a17, v1
	v_accvgpr_read_b32 v184, a14
	v_accvgpr_write_b32 a16, v0
	v_pk_mul_f32 v[0:1], v[184:185], s[0:1] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v195, a11
	v_accvgpr_read_b32 v183, a13
	v_accvgpr_read_b32 v182, a12
	v_accvgpr_write_b32 a13, v1
	v_accvgpr_read_b32 v194, a10
	v_accvgpr_write_b32 a12, v0
	v_pk_mul_f32 v[0:1], v[194:195], s[0:1] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v205, a7
	v_accvgpr_read_b32 v193, a9
	v_accvgpr_read_b32 v192, a8
	v_accvgpr_write_b32 a9, v1
	v_accvgpr_read_b32 v204, a6
	v_accvgpr_write_b32 a8, v0
	v_pk_mul_f32 v[0:1], v[204:205], s[0:1] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v207, a3
	v_accvgpr_read_b32 v203, a5
	v_accvgpr_read_b32 v202, a4
	v_accvgpr_write_b32 a5, v1
	v_accvgpr_read_b32 v206, a2
	v_accvgpr_write_b32 a4, v0
	v_pk_mul_f32 v[0:1], v[206:207], s[0:1] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v209, a187
	v_accvgpr_read_b32 v205, a1
	v_accvgpr_read_b32 v204, a0
	v_accvgpr_write_b32 a0, v0
	v_accvgpr_read_b32 v208, a186
	v_accvgpr_write_b32 a1, v1
	v_pk_mul_f32 v[0:1], v[208:209], s[0:1] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v128, a172
	v_accvgpr_write_b32 a3, v1
	v_accvgpr_read_b32 v130, a174
	v_accvgpr_read_b32 v131, a175
	v_accvgpr_write_b32 a2, v0
	v_pk_mul_f32 v[0:1], v[130:131], s[0:1] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v211, a163
	v_accvgpr_write_b32 a7, v1
	v_accvgpr_read_b32 v210, a162
	v_accvgpr_write_b32 a6, v0
	v_pk_mul_f32 v[0:1], v[210:211], s[0:1] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v221, a151
	v_accvgpr_write_b32 a11, v1
	v_accvgpr_read_b32 v220, a150
	v_accvgpr_write_b32 a10, v0
	v_pk_mul_f32 v[0:1], v[220:221], s[0:1] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v132, a136
	v_accvgpr_write_b32 a15, v1
	v_accvgpr_read_b32 v134, a138
	v_accvgpr_read_b32 v135, a139
	v_accvgpr_write_b32 a14, v0
	v_pk_mul_f32 v[0:1], v[134:135], s[0:1] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v46, a124
	v_accvgpr_write_b32 a19, v1
	v_accvgpr_read_b32 v48, a126
	v_accvgpr_read_b32 v49, a127
	v_accvgpr_write_b32 a18, v0
	v_pk_mul_f32 v[0:1], v[48:49], s[0:1] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v48, a112
	v_accvgpr_write_b32 a23, v1
	v_accvgpr_read_b32 v50, a114
	v_accvgpr_read_b32 v51, a115
	v_accvgpr_write_b32 a22, v0
	v_pk_mul_f32 v[0:1], v[50:51], s[0:1] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v50, a100
	v_accvgpr_write_b32 a31, v1
	v_accvgpr_read_b32 v52, a102
	v_accvgpr_read_b32 v53, a103
	v_accvgpr_write_b32 a30, v0
	v_pk_mul_f32 v[0:1], v[52:53], s[0:1] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v52, a88
	v_accvgpr_write_b32 a39, v1
	v_accvgpr_read_b32 v54, a90
	v_accvgpr_read_b32 v55, a91
	v_accvgpr_write_b32 a38, v0
	v_pk_mul_f32 v[0:1], v[54:55], s[0:1] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v54, a76
	v_accvgpr_write_b32 a47, v1
	v_accvgpr_read_b32 v56, a78
	v_accvgpr_read_b32 v57, a79
	v_accvgpr_write_b32 a46, v0
	v_pk_mul_f32 v[0:1], v[56:57], s[0:1] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v56, a64
	v_accvgpr_write_b32 a55, v1
	v_accvgpr_read_b32 v58, a66
	v_accvgpr_read_b32 v59, a67
	v_accvgpr_write_b32 a54, v0
	v_pk_mul_f32 v[0:1], v[58:59], s[0:1] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v61, a59
	v_accvgpr_write_b32 a63, v1
	v_accvgpr_read_b32 v60, a58
	v_accvgpr_write_b32 a62, v0
	v_pk_mul_f32 v[0:1], v[60:61], s[0:1] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v63, a51
	v_accvgpr_read_b32 v59, a57
	v_accvgpr_read_b32 v58, a56
	v_accvgpr_write_b32 a57, v1
	v_accvgpr_read_b32 v62, a50
	v_accvgpr_write_b32 a56, v0
	v_pk_mul_f32 v[0:1], v[62:63], s[0:1] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v65, a43
	v_accvgpr_read_b32 v61, a49
	v_accvgpr_read_b32 v60, a48
	v_accvgpr_write_b32 a49, v1
	v_accvgpr_read_b32 v64, a42
	v_accvgpr_write_b32 a48, v0
	v_pk_mul_f32 v[0:1], v[64:65], s[0:1] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v67, a35
	v_accvgpr_read_b32 v63, a41
	v_accvgpr_read_b32 v62, a40
	v_accvgpr_write_b32 a41, v1
	v_accvgpr_read_b32 v66, a34
	v_accvgpr_write_b32 a40, v0
	v_pk_mul_f32 v[0:1], v[66:67], s[0:1] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v69, a27
	v_accvgpr_read_b32 v65, a33
	v_accvgpr_read_b32 v64, a32
	v_accvgpr_write_b32 a33, v1
	v_accvgpr_read_b32 v68, a26
	v_accvgpr_write_b32 a32, v0
	v_pk_mul_f32 v[0:1], v[68:69], s[0:1] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v68, a228
	v_accvgpr_read_b32 v70, a230
	v_accvgpr_read_b32 v71, a231
	v_pk_mul_f32 v[210:211], v[70:71], s[0:1] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v70, a220
	v_accvgpr_read_b32 v72, a222
	v_accvgpr_read_b32 v73, a223
	v_pk_mul_f32 v[198:199], v[72:73], s[0:1] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v72, a212
	v_accvgpr_read_b32 v74, a214
	v_accvgpr_read_b32 v75, a215
	v_pk_mul_f32 v[194:195], v[74:75], s[0:1] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v74, a204
	v_accvgpr_read_b32 v76, a206
	v_accvgpr_read_b32 v77, a207
	v_pk_mul_f32 v[184:185], v[76:77], s[0:1] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v76, a196
	v_accvgpr_read_b32 v78, a198
	v_accvgpr_read_b32 v79, a199
	v_pk_mul_f32 v[176:177], v[78:79], s[0:1] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v78, a188
	v_accvgpr_read_b32 v80, a190
	v_accvgpr_read_b32 v81, a191
	v_pk_mul_f32 v[220:221], v[80:81], s[0:1] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v80, a176
	v_accvgpr_read_b32 v82, a178
	v_accvgpr_read_b32 v83, a179
	v_pk_mul_f32 v[144:145], v[82:83], s[0:1] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v82, a164
	v_accvgpr_read_b32 v84, a166
	v_accvgpr_read_b32 v85, a167
	v_pk_mul_f32 v[138:139], v[84:85], s[0:1] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v84, a152
	v_accvgpr_read_b32 v86, a154
	v_accvgpr_read_b32 v87, a155
	v_pk_mul_f32 v[222:223], v[86:87], s[0:1] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v86, a140
	v_accvgpr_read_b32 v88, a142
	v_accvgpr_read_b32 v89, a143
	v_pk_mul_f32 v[224:225], v[88:89], s[0:1] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v88, a128
	v_accvgpr_read_b32 v90, a130
	v_accvgpr_read_b32 v91, a131
	v_pk_mul_f32 v[228:229], v[90:91], s[0:1] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v90, a116
	v_accvgpr_read_b32 v92, a118
	v_accvgpr_read_b32 v93, a119
	v_pk_mul_f32 v[230:231], v[92:93], s[0:1] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v92, a104
	v_accvgpr_read_b32 v94, a106
	v_accvgpr_read_b32 v95, a107
	v_pk_mul_f32 v[234:235], v[94:95], s[0:1] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v97, a95
	v_accvgpr_read_b32 v96, a94
	v_pk_mul_f32 v[240:241], v[96:97], s[0:1] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v99, a83
	v_accvgpr_read_b32 v98, a82
	v_pk_mul_f32 v[242:243], v[98:99], s[0:1] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v101, a71
	v_accvgpr_read_b32 v100, a70
	v_pk_mul_f32 v[248:249], v[100:101], s[0:1] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v100, a252
	v_accvgpr_read_b32 v102, a254
	v_accvgpr_read_b32 v103, a255
	v_pk_mul_f32 v[250:251], v[102:103], s[0:1] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v102, a248
	v_accvgpr_read_b32 v104, a250
	v_accvgpr_read_b32 v105, a251
	v_pk_mul_f32 v[254:255], v[104:105], s[0:1] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v104, a244
	v_accvgpr_read_b32 v67, a25
	v_accvgpr_read_b32 v66, a24
	v_accvgpr_write_b32 a25, v1
	v_accvgpr_read_b32 v106, a246
	v_accvgpr_read_b32 v107, a247
	v_accvgpr_read_b32 v4, a156
	v_accvgpr_write_b32 a24, v0
	v_pk_mul_f32 v[44:45], v[106:107], s[0:1] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v106, a240
	v_accvgpr_read_b32 v6, a158
	v_accvgpr_read_b32 v7, a159
	v_accvgpr_read_b32 v0, a132
	v_lshl_or_b32 v170, s33, 8, v197
	v_accvgpr_read_b32 v108, a242
	v_accvgpr_read_b32 v109, a243
	v_accvgpr_read_b32 v181, a171
	v_pk_mul_f32 v[24:25], v[6:7], s[0:1] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v2, a134
	v_accvgpr_read_b32 v3, a135
	v_mad_i64_i32 v[6:7], s[4:5], s28, v170, 0
	v_ashrrev_i32_e32 v197, 31, v196
	v_pk_mul_f32 v[42:43], v[108:109], s[0:1] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v108, a236
	v_accvgpr_read_b32 v180, a170
	v_pk_mul_f32 v[20:21], v[2:3], s[0:1] op_sel_hi:[1,0]
	v_lshl_add_u64 v[6:7], v[6:7], 1, s[2:3]
	v_lshlrev_b64 v[158:159], 1, v[196:197]
	v_lshrrev_b32_e32 v2, 2, v212
	v_accvgpr_read_b32 v110, a238
	v_accvgpr_read_b32 v111, a239
	v_accvgpr_read_b32 v10, a180
	v_pk_mul_f32 v[28:29], v[180:181], s[0:1] op_sel_hi:[1,0]
	v_lshl_add_u64 v[180:181], v[6:7], 0, v[158:159]
	v_and_b32_e32 v2, 12, v2
	v_and_b32_e32 v6, 15, v212
	v_pk_mul_f32 v[40:41], v[110:111], s[0:1] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v110, a232
	v_accvgpr_read_b32 v12, a182
	v_accvgpr_read_b32 v13, a183
	v_or_b32_e32 v3, 16, v2
	v_mad_u64_u32 v[8:9], s[4:5], v2, s28, v[6:7]
	v_or_b32_e32 v7, 32, v2
	v_or_b32_e32 v2, 48, v2
	v_accvgpr_read_b32 v112, a234
	v_accvgpr_read_b32 v113, a235
	v_pk_mul_f32 v[30:31], v[12:13], s[0:1] op_sel_hi:[1,0]
	v_mad_u64_u32 v[130:131], s[4:5], v7, s28, v[6:7]
	v_mad_u64_u32 v[146:147], s[4:5], v2, s28, v[6:7]
	v_mad_u64_u32 v[12:13], s[4:5], v3, s28, v[6:7]
	v_add_u32_e32 v6, s28, v8
	v_pk_mul_f32 v[214:215], v[112:113], s[0:1] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v112, a224
	v_ashrrev_i32_e32 v9, 31, v8
	v_add_u32_e32 v190, s28, v6
	v_accvgpr_read_b32 v114, a226
	v_accvgpr_read_b32 v115, a227
	v_accvgpr_read_b32 v1, a133
	v_lshlrev_b64 v[212:213], 1, v[8:9]
	v_ashrrev_i32_e32 v7, 31, v6
	v_ashrrev_i32_e32 v191, 31, v190
	v_pk_mul_f32 v[200:201], v[114:115], s[0:1] op_sel_hi:[1,0]
	v_lshl_add_u64 v[196:197], v[180:181], 0, v[212:213]
	v_lshlrev_b64 v[116:117], 1, v[6:7]
	v_lshlrev_b64 v[114:115], 1, v[190:191]
	v_pk_mul_f32 v[0:1], v[0:1], s[0:1] op_sel_hi:[1,0]
	v_lshl_add_u64 v[8:9], v[180:181], 0, v[116:117]
	v_lshl_add_u64 v[6:7], v[180:181], 0, v[114:115]
	global_store_short_d16_hi v[196:197], v0, off
	global_store_short_d16_hi v[8:9], v1, off
	global_store_short_d16_hi v[6:7], v20, off
	v_add_u32_e32 v0, s28, v190
	v_add_u32_e32 v14, s28, v12
	v_ashrrev_i32_e32 v1, 31, v0
	v_accvgpr_read_b32 v124, a144
	v_add_u32_e32 v2, s28, v14
	v_lshlrev_b64 v[190:191], 1, v[0:1]
	v_accvgpr_read_b32 v5, a157
	v_accvgpr_read_b32 v125, a145
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[0:1], v[180:181], 0, v[190:191]
	v_accvgpr_read_b32 v118, a208
	v_accvgpr_read_b32 v179, a169
	v_accvgpr_read_b32 v178, a168
	v_accvgpr_read_b32 v126, a146
	v_accvgpr_read_b32 v127, a147
	v_ashrrev_i32_e32 v13, 31, v12
	v_lshlrev_b64 v[168:169], 1, v[2:3]
	global_store_short_d16_hi v[0:1], v21, off
	v_pk_mul_f32 v[20:21], v[124:125], s[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], s[0:1] op_sel_hi:[1,0]
	v_add_u32_e32 v2, s28, v2
	v_accvgpr_read_b32 v120, a210
	v_accvgpr_read_b32 v121, a211
	v_accvgpr_read_b32 v16, a192
	v_accvgpr_read_b32 v11, a181
	v_pk_mul_f32 v[22:23], v[126:127], s[0:1] op_sel_hi:[1,0]
	v_lshlrev_b64 v[172:173], 1, v[12:13]
	v_ashrrev_i32_e32 v15, 31, v14
	global_store_short_d16_hi v[196:197], v20, off offset:32
	global_store_short_d16_hi v[8:9], v21, off offset:32
	global_store_short_d16_hi v[6:7], v22, off offset:32
	global_store_short_d16_hi v[0:1], v23, off offset:32
	global_store_short_d16_hi v[196:197], v4, off offset:64
	global_store_short_d16_hi v[8:9], v5, off offset:64
	global_store_short_d16_hi v[6:7], v24, off offset:64
	global_store_short_d16_hi v[0:1], v25, off offset:64
	v_pk_mul_f32 v[4:5], v[178:179], s[0:1] op_sel_hi:[1,0]
	v_ashrrev_i32_e32 v3, 31, v2
	v_pk_mul_f32 v[36:37], v[120:121], s[0:1] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v160, a200
	v_accvgpr_read_b32 v17, a193
	v_ashrrev_i32_e32 v131, 31, v130
	v_lshl_add_u64 v[252:253], v[180:181], 0, v[172:173]
	v_lshlrev_b64 v[120:121], 1, v[14:15]
	global_store_short_d16_hi v[196:197], v4, off offset:96
	global_store_short_d16_hi v[8:9], v5, off offset:96
	global_store_short_d16_hi v[6:7], v28, off offset:96
	global_store_short_d16_hi v[0:1], v29, off offset:96
	v_pk_mul_f32 v[4:5], v[10:11], s[0:1] op_sel_hi:[1,0]
	v_lshlrev_b64 v[2:3], 1, v[2:3]
	v_accvgpr_read_b32 v161, a201
	v_accvgpr_read_b32 v18, a194
	v_accvgpr_read_b32 v19, a195
	v_lshlrev_b64 v[126:127], 1, v[130:131]
	v_lshl_add_u64 v[12:13], v[180:181], 0, v[120:121]
	v_lshl_add_u64 v[14:15], v[180:181], 0, v[168:169]
	v_add_u32_e32 v130, s28, v130
	global_store_short_d16_hi v[252:253], v4, off
	global_store_short_d16_hi v[12:13], v5, off
	global_store_short_d16_hi v[14:15], v30, off
	v_lshl_add_u64 v[4:5], v[180:181], 0, v[2:3]
	v_pk_mul_f32 v[10:11], v[16:17], s[0:1] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v154, a216
	v_accvgpr_read_b32 v119, a209
	v_accvgpr_read_b32 v162, a202
	v_accvgpr_read_b32 v163, a203
	v_pk_mul_f32 v[32:33], v[18:19], s[0:1] op_sel_hi:[1,0]
	v_add_u32_e32 v18, s28, v130
	global_store_short_d16_hi v[4:5], v31, off
	global_store_short_d16_hi v[252:253], v10, off offset:32
	global_store_short_d16_hi v[12:13], v11, off offset:32
	global_store_short_d16_hi v[14:15], v32, off offset:32
	global_store_short_d16_hi v[4:5], v33, off offset:32
	v_pk_mul_f32 v[10:11], v[160:161], s[0:1] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v155, a217
	v_accvgpr_read_b32 v156, a218
	v_accvgpr_read_b32 v157, a219
	v_pk_mul_f32 v[34:35], v[162:163], s[0:1] op_sel_hi:[1,0]
	v_ashrrev_i32_e32 v131, 31, v130
	v_ashrrev_i32_e32 v19, 31, v18
	global_store_short_d16_hi v[252:253], v10, off offset:64
	global_store_short_d16_hi v[12:13], v11, off offset:64
	global_store_short_d16_hi v[14:15], v34, off offset:64
	global_store_short_d16_hi v[4:5], v35, off offset:64
	v_pk_mul_f32 v[10:11], v[118:119], s[0:1] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[156:157], s[0:1] op_sel_hi:[1,0]
	v_lshl_add_u64 v[238:239], v[180:181], 0, v[126:127]
	v_lshlrev_b64 v[156:157], 1, v[130:131]
	v_lshlrev_b64 v[162:163], 1, v[18:19]
	global_store_short_d16_hi v[252:253], v10, off offset:96
	global_store_short_d16_hi v[12:13], v11, off offset:96
	global_store_short_d16_hi v[14:15], v36, off offset:96
	global_store_short_d16_hi v[4:5], v37, off offset:96
	v_pk_mul_f32 v[10:11], v[154:155], s[0:1] op_sel_hi:[1,0]
	v_lshl_add_u64 v[244:245], v[180:181], 0, v[156:157]
	v_lshl_add_u64 v[246:247], v[180:181], 0, v[162:163]
	global_store_short_d16_hi v[238:239], v10, off
	global_store_short_d16_hi v[244:245], v11, off
	global_store_short_d16_hi v[246:247], v38, off
	v_add_u32_e32 v10, s28, v18
	v_ashrrev_i32_e32 v11, 31, v10
	v_accvgpr_read_b32 v113, a225
	v_ashrrev_i32_e32 v147, 31, v146
	v_lshlrev_b64 v[10:11], 1, v[10:11]
	v_accvgpr_read_b32 v111, a233
	v_lshlrev_b64 v[134:135], 1, v[146:147]
	v_add_u32_e32 v146, s28, v146
	v_lshl_add_u64 v[16:17], v[180:181], 0, v[10:11]
	v_pk_mul_f32 v[18:19], v[112:113], s[0:1] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v109, a237
	v_add_u32_e32 v26, s28, v146
	global_store_short_d16_hi v[16:17], v39, off
	global_store_short_d16_hi v[238:239], v18, off offset:32
	global_store_short_d16_hi v[244:245], v19, off offset:32
	global_store_short_d16_hi v[246:247], v200, off offset:32
	global_store_short_d16_hi v[16:17], v201, off offset:32
	v_pk_mul_f32 v[18:19], v[110:111], s[0:1] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v107, a241
	v_ashrrev_i32_e32 v147, 31, v146
	v_ashrrev_i32_e32 v27, 31, v26
	global_store_short_d16_hi v[238:239], v18, off offset:64
	global_store_short_d16_hi v[244:245], v19, off offset:64
	global_store_short_d16_hi v[246:247], v214, off offset:64
	global_store_short_d16_hi v[16:17], v215, off offset:64
	v_pk_mul_f32 v[18:19], v[108:109], s[0:1] op_sel_hi:[1,0]
	v_lshl_add_u64 v[226:227], v[180:181], 0, v[134:135]
	v_lshlrev_b64 v[130:131], 1, v[146:147]
	v_lshlrev_b64 v[146:147], 1, v[26:27]
	global_store_short_d16_hi v[238:239], v18, off offset:96
	global_store_short_d16_hi v[244:245], v19, off offset:96
	global_store_short_d16_hi v[246:247], v40, off offset:96
	global_store_short_d16_hi v[16:17], v41, off offset:96
	v_pk_mul_f32 v[18:19], v[106:107], s[0:1] op_sel_hi:[1,0]
	v_lshl_add_u64 v[232:233], v[180:181], 0, v[130:131]
	v_lshl_add_u64 v[236:237], v[180:181], 0, v[146:147]
	global_store_short_d16_hi v[226:227], v18, off
	global_store_short_d16_hi v[232:233], v19, off
	global_store_short_d16_hi v[236:237], v42, off
	v_add_u32_e32 v18, s28, v26
	v_ashrrev_i32_e32 v19, 31, v18
	v_accvgpr_read_b32 v105, a245
	v_lshlrev_b64 v[106:107], 1, v[18:19]
	v_accvgpr_read_b32 v103, a249
	v_lshl_add_u64 v[18:19], v[180:181], 0, v[106:107]
	v_pk_mul_f32 v[20:21], v[104:105], s[0:1] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v101, a253
	global_store_short_d16_hi v[18:19], v43, off
	global_store_short_d16_hi v[226:227], v20, off offset:32
	global_store_short_d16_hi v[232:233], v21, off offset:32
	global_store_short_d16_hi v[236:237], v44, off offset:32
	global_store_short_d16_hi v[18:19], v45, off offset:32
	v_pk_mul_f32 v[20:21], v[102:103], s[0:1] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v99, a69
	v_accvgpr_read_b32 v98, a68
	global_store_short_d16_hi v[226:227], v20, off offset:64
	global_store_short_d16_hi v[232:233], v21, off offset:64
	global_store_short_d16_hi v[236:237], v254, off offset:64
	global_store_short_d16_hi v[18:19], v255, off offset:64
	v_pk_mul_f32 v[20:21], v[100:101], s[0:1] op_sel_hi:[1,0]
	s_mov_b64 s[4:5], 0x100
	v_accvgpr_read_b32 v97, a81
	v_accvgpr_read_b32 v96, a80
	global_store_short_d16_hi v[226:227], v20, off offset:96
	global_store_short_d16_hi v[232:233], v21, off offset:96
	global_store_short_d16_hi v[236:237], v250, off offset:96
	global_store_short_d16_hi v[18:19], v251, off offset:96
	v_lshl_add_u64 v[20:21], v[180:181], 0, s[4:5]
	v_pk_mul_f32 v[24:25], v[98:99], s[0:1] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v95, a93
	v_accvgpr_read_b32 v94, a92
	v_lshl_add_u64 v[22:23], v[20:21], 0, v[212:213]
	global_store_short_d16_hi v[196:197], v24, off offset:256
	global_store_short_d16_hi v[8:9], v25, off offset:256
	global_store_short_d16_hi v[6:7], v248, off offset:256
	global_store_short_d16_hi v[0:1], v249, off offset:256
	v_pk_mul_f32 v[0:1], v[96:97], s[0:1] op_sel_hi:[1,0]
	v_lshl_add_u64 v[6:7], v[20:21], 0, v[116:117]
	v_accvgpr_read_b32 v93, a105
	global_store_short_d16_hi v[22:23], v0, off offset:32
	global_store_short_d16_hi v[6:7], v1, off offset:32
	v_lshl_add_u64 v[0:1], v[20:21], 0, v[114:115]
	v_lshl_add_u64 v[8:9], v[20:21], 0, v[190:191]
	v_pk_mul_f32 v[24:25], v[94:95], s[0:1] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v91, a117
	global_store_short_d16_hi v[0:1], v242, off offset:32
	global_store_short_d16_hi v[8:9], v243, off offset:32
	global_store_short_d16_hi v[22:23], v24, off offset:64
	global_store_short_d16_hi v[6:7], v25, off offset:64
	global_store_short_d16_hi v[0:1], v240, off offset:64
	global_store_short_d16_hi v[8:9], v241, off offset:64
	v_pk_mul_f32 v[24:25], v[92:93], s[0:1] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v89, a129
	global_store_short_d16_hi v[22:23], v24, off offset:96
	global_store_short_d16_hi v[6:7], v25, off offset:96
	global_store_short_d16_hi v[0:1], v234, off offset:96
	global_store_short_d16_hi v[8:9], v235, off offset:96
	v_pk_mul_f32 v[6:7], v[90:91], s[0:1] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v87, a141
	v_lshl_add_u64 v[0:1], v[20:21], 0, v[172:173]
	global_store_short_d16_hi v[252:253], v6, off offset:256
	global_store_short_d16_hi v[12:13], v7, off offset:256
	global_store_short_d16_hi v[14:15], v230, off offset:256
	global_store_short_d16_hi v[4:5], v231, off offset:256
	v_pk_mul_f32 v[4:5], v[88:89], s[0:1] op_sel_hi:[1,0]
	v_lshl_add_u64 v[6:7], v[20:21], 0, v[120:121]
	v_accvgpr_read_b32 v85, a153
	global_store_short_d16_hi v[0:1], v4, off offset:32
	global_store_short_d16_hi v[6:7], v5, off offset:32
	v_lshl_add_u64 v[4:5], v[20:21], 0, v[168:169]
	v_lshl_add_u64 v[8:9], v[20:21], 0, v[2:3]
	v_pk_mul_f32 v[12:13], v[86:87], s[0:1] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v83, a165
	global_store_short_d16_hi v[4:5], v228, off offset:32
	global_store_short_d16_hi v[8:9], v229, off offset:32
	global_store_short_d16_hi v[0:1], v12, off offset:64
	global_store_short_d16_hi v[6:7], v13, off offset:64
	global_store_short_d16_hi v[4:5], v224, off offset:64
	global_store_short_d16_hi v[8:9], v225, off offset:64
	v_pk_mul_f32 v[12:13], v[84:85], s[0:1] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v81, a177
	global_store_short_d16_hi v[0:1], v12, off offset:96
	global_store_short_d16_hi v[6:7], v13, off offset:96
	global_store_short_d16_hi v[4:5], v222, off offset:96
	global_store_short_d16_hi v[8:9], v223, off offset:96
	v_pk_mul_f32 v[4:5], v[82:83], s[0:1] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v79, a189
	v_lshl_add_u64 v[0:1], v[20:21], 0, v[126:127]
	global_store_short_d16_hi v[238:239], v4, off offset:256
	global_store_short_d16_hi v[244:245], v5, off offset:256
	global_store_short_d16_hi v[246:247], v138, off offset:256
	global_store_short_d16_hi v[16:17], v139, off offset:256
	v_pk_mul_f32 v[4:5], v[80:81], s[0:1] op_sel_hi:[1,0]
	v_lshl_add_u64 v[6:7], v[20:21], 0, v[156:157]
	v_accvgpr_read_b32 v77, a197
	global_store_short_d16_hi v[0:1], v4, off offset:32
	global_store_short_d16_hi v[6:7], v5, off offset:32
	v_lshl_add_u64 v[4:5], v[20:21], 0, v[162:163]
	v_lshl_add_u64 v[8:9], v[20:21], 0, v[10:11]
	v_pk_mul_f32 v[12:13], v[78:79], s[0:1] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v75, a205
	global_store_short_d16_hi v[4:5], v144, off offset:32
	global_store_short_d16_hi v[8:9], v145, off offset:32
	global_store_short_d16_hi v[0:1], v12, off offset:64
	global_store_short_d16_hi v[6:7], v13, off offset:64
	global_store_short_d16_hi v[4:5], v220, off offset:64
	global_store_short_d16_hi v[8:9], v221, off offset:64
	v_pk_mul_f32 v[12:13], v[76:77], s[0:1] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v73, a213
	global_store_short_d16_hi v[0:1], v12, off offset:96
	global_store_short_d16_hi v[6:7], v13, off offset:96
	global_store_short_d16_hi v[4:5], v176, off offset:96
	global_store_short_d16_hi v[8:9], v177, off offset:96
	v_pk_mul_f32 v[4:5], v[74:75], s[0:1] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v71, a221
	v_lshl_add_u64 v[0:1], v[20:21], 0, v[134:135]
	global_store_short_d16_hi v[226:227], v4, off offset:256
	global_store_short_d16_hi v[232:233], v5, off offset:256
	global_store_short_d16_hi v[236:237], v184, off offset:256
	global_store_short_d16_hi v[18:19], v185, off offset:256
	v_pk_mul_f32 v[4:5], v[72:73], s[0:1] op_sel_hi:[1,0]
	v_lshl_add_u64 v[6:7], v[20:21], 0, v[130:131]
	v_accvgpr_read_b32 v69, a229
	global_store_short_d16_hi v[0:1], v4, off offset:32
	global_store_short_d16_hi v[6:7], v5, off offset:32
	v_lshl_add_u64 v[4:5], v[20:21], 0, v[146:147]
	v_lshl_add_u64 v[8:9], v[20:21], 0, v[106:107]
	v_pk_mul_f32 v[12:13], v[70:71], s[0:1] op_sel_hi:[1,0]
	global_store_short_d16_hi v[4:5], v194, off offset:32
	global_store_short_d16_hi v[8:9], v195, off offset:32
	global_store_short_d16_hi v[0:1], v12, off offset:64
	global_store_short_d16_hi v[6:7], v13, off offset:64
	global_store_short_d16_hi v[4:5], v198, off offset:64
	global_store_short_d16_hi v[8:9], v199, off offset:64
	v_pk_mul_f32 v[12:13], v[68:69], s[0:1] op_sel_hi:[1,0]
	global_store_short_d16_hi v[0:1], v12, off offset:96
	global_store_short_d16_hi v[6:7], v13, off offset:96
	global_store_short_d16_hi v[4:5], v210, off offset:96
	global_store_short_d16_hi v[8:9], v211, off offset:96
	v_or_b32_e32 v0, 0x80, v170
	v_mad_i64_i32 v[0:1], s[6:7], s28, v0, 0
	v_lshl_add_u64 v[0:1], v[0:1], 1, s[2:3]
	v_lshl_add_u64 v[0:1], v[0:1], 0, v[158:159]
	v_pk_mul_f32 v[6:7], v[66:67], s[0:1] op_sel_hi:[1,0]
	v_lshl_add_u64 v[8:9], v[0:1], 0, v[212:213]
	v_lshl_add_u64 v[12:13], v[0:1], 0, v[116:117]
	v_accvgpr_read_b32 v18, a24
	global_store_short_d16_hi v[8:9], v6, off
	global_store_short_d16_hi v[12:13], v7, off
	v_pk_mul_f32 v[6:7], v[64:65], s[0:1] op_sel_hi:[1,0]
	v_lshl_add_u64 v[14:15], v[0:1], 0, v[114:115]
	v_accvgpr_read_b32 v19, a25
	v_lshl_add_u64 v[16:17], v[0:1], 0, v[190:191]
	global_store_short_d16_hi v[14:15], v18, off
	global_store_short_d16_hi v[16:17], v19, off
	global_store_short_d16_hi v[8:9], v6, off offset:32
	global_store_short_d16_hi v[12:13], v7, off offset:32
	v_accvgpr_read_b32 v6, a32
	v_accvgpr_read_b32 v7, a33
	global_store_short_d16_hi v[14:15], v6, off offset:32
	global_store_short_d16_hi v[16:17], v7, off offset:32
	v_pk_mul_f32 v[6:7], v[62:63], s[0:1] op_sel_hi:[1,0]
	global_store_short_d16_hi v[8:9], v6, off offset:64
	global_store_short_d16_hi v[12:13], v7, off offset:64
	v_accvgpr_read_b32 v6, a40
	v_accvgpr_read_b32 v7, a41
	global_store_short_d16_hi v[14:15], v6, off offset:64
	global_store_short_d16_hi v[16:17], v7, off offset:64
	v_pk_mul_f32 v[6:7], v[60:61], s[0:1] op_sel_hi:[1,0]
	global_store_short_d16_hi v[8:9], v6, off offset:96
	global_store_short_d16_hi v[12:13], v7, off offset:96
	v_accvgpr_read_b32 v6, a48
	v_accvgpr_read_b32 v7, a49
	v_accvgpr_read_b32 v57, a65
	global_store_short_d16_hi v[14:15], v6, off offset:96
	global_store_short_d16_hi v[16:17], v7, off offset:96
	v_pk_mul_f32 v[6:7], v[58:59], s[0:1] op_sel_hi:[1,0]
	v_lshl_add_u64 v[18:19], v[0:1], 0, v[172:173]
	v_lshl_add_u64 v[20:21], v[0:1], 0, v[120:121]
	v_accvgpr_read_b32 v26, a56
	global_store_short_d16_hi v[18:19], v6, off
	global_store_short_d16_hi v[20:21], v7, off
	v_pk_mul_f32 v[6:7], v[56:57], s[0:1] op_sel_hi:[1,0]
	v_lshl_add_u64 v[22:23], v[0:1], 0, v[168:169]
	v_accvgpr_read_b32 v27, a57
	v_lshl_add_u64 v[24:25], v[0:1], 0, v[2:3]
	global_store_short_d16_hi v[22:23], v26, off
	global_store_short_d16_hi v[24:25], v27, off
	global_store_short_d16_hi v[18:19], v6, off offset:32
	global_store_short_d16_hi v[20:21], v7, off offset:32
	v_accvgpr_read_b32 v6, a62
	v_accvgpr_read_b32 v55, a77
	v_accvgpr_read_b32 v7, a63
	global_store_short_d16_hi v[22:23], v6, off offset:32
	global_store_short_d16_hi v[24:25], v7, off offset:32
	v_pk_mul_f32 v[6:7], v[54:55], s[0:1] op_sel_hi:[1,0]
	global_store_short_d16_hi v[18:19], v6, off offset:64
	global_store_short_d16_hi v[20:21], v7, off offset:64
	v_accvgpr_read_b32 v6, a54
	v_accvgpr_read_b32 v53, a89
	v_accvgpr_read_b32 v7, a55
	global_store_short_d16_hi v[22:23], v6, off offset:64
	global_store_short_d16_hi v[24:25], v7, off offset:64
	v_pk_mul_f32 v[6:7], v[52:53], s[0:1] op_sel_hi:[1,0]
	global_store_short_d16_hi v[18:19], v6, off offset:96
	global_store_short_d16_hi v[20:21], v7, off offset:96
	v_accvgpr_read_b32 v6, a46
	v_accvgpr_read_b32 v51, a101
	v_accvgpr_read_b32 v7, a47
	v_accvgpr_read_b32 v49, a113
	global_store_short_d16_hi v[22:23], v6, off offset:96
	global_store_short_d16_hi v[24:25], v7, off offset:96
	v_pk_mul_f32 v[6:7], v[50:51], s[0:1] op_sel_hi:[1,0]
	v_lshl_add_u64 v[26:27], v[0:1], 0, v[126:127]
	v_lshl_add_u64 v[28:29], v[0:1], 0, v[156:157]
	v_accvgpr_read_b32 v34, a38
	global_store_short_d16_hi v[26:27], v6, off
	global_store_short_d16_hi v[28:29], v7, off
	v_pk_mul_f32 v[6:7], v[48:49], s[0:1] op_sel_hi:[1,0]
	v_lshl_add_u64 v[30:31], v[0:1], 0, v[162:163]
	v_accvgpr_read_b32 v35, a39
	v_lshl_add_u64 v[32:33], v[0:1], 0, v[10:11]
	global_store_short_d16_hi v[30:31], v34, off
	global_store_short_d16_hi v[32:33], v35, off
	global_store_short_d16_hi v[26:27], v6, off offset:32
	global_store_short_d16_hi v[28:29], v7, off offset:32
	v_accvgpr_read_b32 v6, a30
	v_accvgpr_read_b32 v47, a125
	v_accvgpr_read_b32 v7, a31
	global_store_short_d16_hi v[30:31], v6, off offset:32
	global_store_short_d16_hi v[32:33], v7, off offset:32
	v_pk_mul_f32 v[6:7], v[46:47], s[0:1] op_sel_hi:[1,0]
	global_store_short_d16_hi v[26:27], v6, off offset:64
	global_store_short_d16_hi v[28:29], v7, off offset:64
	v_accvgpr_read_b32 v6, a22
	v_accvgpr_read_b32 v133, a137
	v_accvgpr_read_b32 v7, a23
	global_store_short_d16_hi v[30:31], v6, off offset:64
	global_store_short_d16_hi v[32:33], v7, off offset:64
	v_pk_mul_f32 v[6:7], v[132:133], s[0:1] op_sel_hi:[1,0]
	global_store_short_d16_hi v[26:27], v6, off offset:96
	global_store_short_d16_hi v[28:29], v7, off offset:96
	v_accvgpr_read_b32 v6, a18
	v_accvgpr_read_b32 v219, a149
	v_accvgpr_read_b32 v218, a148
	v_accvgpr_read_b32 v7, a19
	v_accvgpr_read_b32 v209, a161
	v_accvgpr_read_b32 v208, a160
	global_store_short_d16_hi v[30:31], v6, off offset:96
	global_store_short_d16_hi v[32:33], v7, off offset:96
	v_pk_mul_f32 v[6:7], v[218:219], s[0:1] op_sel_hi:[1,0]
	v_lshl_add_u64 v[34:35], v[0:1], 0, v[134:135]
	v_lshl_add_u64 v[36:37], v[0:1], 0, v[130:131]
	v_accvgpr_read_b32 v41, a15
	v_lshl_add_u64 v[68:69], v[0:1], 0, s[4:5]
	global_store_short_d16_hi v[34:35], v6, off
	global_store_short_d16_hi v[36:37], v7, off
	v_pk_mul_f32 v[6:7], v[208:209], s[0:1] op_sel_hi:[1,0]
	v_lshl_add_u64 v[38:39], v[0:1], 0, v[146:147]
	v_accvgpr_read_b32 v40, a14
	v_lshl_add_u64 v[0:1], v[0:1], 0, v[106:107]
	global_store_short_d16_hi v[38:39], v40, off
	global_store_short_d16_hi v[0:1], v41, off
	global_store_short_d16_hi v[34:35], v6, off offset:32
	global_store_short_d16_hi v[36:37], v7, off offset:32
	v_accvgpr_read_b32 v6, a10
	v_accvgpr_read_b32 v129, a173
	v_accvgpr_read_b32 v7, a11
	global_store_short_d16_hi v[38:39], v6, off offset:32
	global_store_short_d16_hi v[0:1], v7, off offset:32
	v_pk_mul_f32 v[6:7], v[128:129], s[0:1] op_sel_hi:[1,0]
	global_store_short_d16_hi v[34:35], v6, off offset:64
	global_store_short_d16_hi v[36:37], v7, off offset:64
	v_accvgpr_read_b32 v6, a6
	v_accvgpr_read_b32 v207, a185
	v_accvgpr_read_b32 v206, a184
	v_accvgpr_read_b32 v7, a7
	global_store_short_d16_hi v[38:39], v6, off offset:64
	global_store_short_d16_hi v[0:1], v7, off offset:64
	v_pk_mul_f32 v[6:7], v[206:207], s[0:1] op_sel_hi:[1,0]
	global_store_short_d16_hi v[34:35], v6, off offset:96
	global_store_short_d16_hi v[36:37], v7, off offset:96
	v_accvgpr_read_b32 v7, a3
	v_accvgpr_read_b32 v6, a2
	global_store_short_d16_hi v[38:39], v6, off offset:96
	global_store_short_d16_hi v[0:1], v7, off offset:96
	v_pk_mul_f32 v[6:7], v[204:205], s[0:1] op_sel_hi:[1,0]
	global_store_short_d16_hi v[8:9], v6, off offset:256
	global_store_short_d16_hi v[12:13], v7, off offset:256
	v_accvgpr_read_b32 v7, a1
	v_accvgpr_read_b32 v6, a0
	v_lshl_add_u64 v[4:5], v[68:69], 0, v[212:213]
	global_store_short_d16_hi v[14:15], v6, off offset:256
	global_store_short_d16_hi v[16:17], v7, off offset:256
	v_pk_mul_f32 v[6:7], v[202:203], s[0:1] op_sel_hi:[1,0]
	v_lshl_add_u64 v[8:9], v[68:69], 0, v[116:117]
	v_accvgpr_read_b32 v15, a5
	global_store_short_d16_hi v[4:5], v6, off offset:32
	global_store_short_d16_hi v[8:9], v7, off offset:32
	v_lshl_add_u64 v[6:7], v[68:69], 0, v[114:115]
	v_accvgpr_read_b32 v14, a4
	v_lshl_add_u64 v[12:13], v[68:69], 0, v[190:191]
	global_store_short_d16_hi v[6:7], v14, off offset:32
	global_store_short_d16_hi v[12:13], v15, off offset:32
	v_pk_mul_f32 v[14:15], v[192:193], s[0:1] op_sel_hi:[1,0]
	global_store_short_d16_hi v[4:5], v14, off offset:64
	global_store_short_d16_hi v[8:9], v15, off offset:64
	v_accvgpr_read_b32 v15, a9
	v_accvgpr_read_b32 v14, a8
	global_store_short_d16_hi v[6:7], v14, off offset:64
	global_store_short_d16_hi v[12:13], v15, off offset:64
	v_pk_mul_f32 v[14:15], v[182:183], s[0:1] op_sel_hi:[1,0]
	global_store_short_d16_hi v[4:5], v14, off offset:96
	global_store_short_d16_hi v[8:9], v15, off offset:96
	v_accvgpr_read_b32 v4, a12
	v_accvgpr_read_b32 v5, a13
	global_store_short_d16_hi v[6:7], v4, off offset:96
	global_store_short_d16_hi v[12:13], v5, off offset:96
	v_pk_mul_f32 v[6:7], v[174:175], s[0:1] op_sel_hi:[1,0]
	global_store_short_d16_hi v[18:19], v6, off offset:256
	global_store_short_d16_hi v[20:21], v7, off offset:256
	v_accvgpr_read_b32 v6, a16
	v_accvgpr_read_b32 v7, a17
	v_lshl_add_u64 v[4:5], v[68:69], 0, v[172:173]
	global_store_short_d16_hi v[22:23], v6, off offset:256
	global_store_short_d16_hi v[24:25], v7, off offset:256
	v_pk_mul_f32 v[6:7], v[122:123], s[0:1] op_sel_hi:[1,0]
	v_lshl_add_u64 v[8:9], v[68:69], 0, v[120:121]
	v_accvgpr_read_b32 v12, a20
	global_store_short_d16_hi v[4:5], v6, off offset:32
	global_store_short_d16_hi v[8:9], v7, off offset:32
	v_lshl_add_u64 v[6:7], v[68:69], 0, v[168:169]
	v_accvgpr_read_b32 v13, a21
	v_lshl_add_u64 v[2:3], v[68:69], 0, v[2:3]
	global_store_short_d16_hi v[6:7], v12, off offset:32
	global_store_short_d16_hi v[2:3], v13, off offset:32
	v_pk_mul_f32 v[12:13], v[142:143], s[0:1] op_sel_hi:[1,0]
	global_store_short_d16_hi v[4:5], v12, off offset:64
	global_store_short_d16_hi v[8:9], v13, off offset:64
	v_accvgpr_read_b32 v12, a28
	v_accvgpr_read_b32 v13, a29
	global_store_short_d16_hi v[6:7], v12, off offset:64
	global_store_short_d16_hi v[2:3], v13, off offset:64
	v_pk_mul_f32 v[12:13], v[216:217], s[0:1] op_sel_hi:[1,0]
	global_store_short_d16_hi v[4:5], v12, off offset:96
	global_store_short_d16_hi v[8:9], v13, off offset:96
	v_accvgpr_read_b32 v4, a36
	v_accvgpr_read_b32 v5, a37
	global_store_short_d16_hi v[6:7], v4, off offset:96
	global_store_short_d16_hi v[2:3], v5, off offset:96
	v_pk_mul_f32 v[4:5], v[140:141], s[0:1] op_sel_hi:[1,0]
	global_store_short_d16_hi v[26:27], v4, off offset:256
	global_store_short_d16_hi v[28:29], v5, off offset:256
	v_accvgpr_read_b32 v4, a44
	v_accvgpr_read_b32 v5, a45
	v_lshl_add_u64 v[2:3], v[68:69], 0, v[126:127]
	global_store_short_d16_hi v[30:31], v4, off offset:256
	global_store_short_d16_hi v[32:33], v5, off offset:256
	v_pk_mul_f32 v[4:5], v[136:137], s[0:1] op_sel_hi:[1,0]
	v_lshl_add_u64 v[6:7], v[68:69], 0, v[156:157]
	v_accvgpr_read_b32 v12, a52
	global_store_short_d16_hi v[2:3], v4, off offset:32
	global_store_short_d16_hi v[6:7], v5, off offset:32
	v_lshl_add_u64 v[4:5], v[68:69], 0, v[162:163]
	v_accvgpr_read_b32 v13, a53
	v_lshl_add_u64 v[8:9], v[68:69], 0, v[10:11]
	v_pk_mul_f32 v[10:11], v[188:189], s[0:1] op_sel_hi:[1,0]
	global_store_short_d16_hi v[4:5], v12, off offset:32
	global_store_short_d16_hi v[8:9], v13, off offset:32
	global_store_short_d16_hi v[2:3], v10, off offset:64
	global_store_short_d16_hi v[6:7], v11, off offset:64
	v_accvgpr_read_b32 v10, a60
	v_accvgpr_read_b32 v11, a61
	global_store_short_d16_hi v[4:5], v10, off offset:64
	global_store_short_d16_hi v[8:9], v11, off offset:64
	v_pk_mul_f32 v[10:11], v[186:187], s[0:1] op_sel_hi:[1,0]
	global_store_short_d16_hi v[2:3], v10, off offset:96
	global_store_short_d16_hi v[6:7], v11, off offset:96
	v_accvgpr_read_b32 v2, a72
	v_accvgpr_read_b32 v3, a73
	global_store_short_d16_hi v[4:5], v2, off offset:96
	global_store_short_d16_hi v[8:9], v3, off offset:96
	v_pk_mul_f32 v[4:5], v[152:153], s[0:1] op_sel_hi:[1,0]
	global_store_short_d16_hi v[34:35], v4, off offset:256
	global_store_short_d16_hi v[36:37], v5, off offset:256
	v_accvgpr_read_b32 v4, a84
	v_accvgpr_read_b32 v5, a85
	v_lshl_add_u64 v[2:3], v[68:69], 0, v[134:135]
	global_store_short_d16_hi v[38:39], v4, off offset:256
	global_store_short_d16_hi v[0:1], v5, off offset:256
	v_pk_mul_f32 v[0:1], v[150:151], s[0:1] op_sel_hi:[1,0]
	v_lshl_add_u64 v[4:5], v[68:69], 0, v[130:131]
	v_accvgpr_read_b32 v8, a96
	global_store_short_d16_hi v[2:3], v0, off offset:32
	global_store_short_d16_hi v[4:5], v1, off offset:32
	v_lshl_add_u64 v[0:1], v[68:69], 0, v[146:147]
	v_accvgpr_read_b32 v9, a97
	v_lshl_add_u64 v[6:7], v[68:69], 0, v[106:107]
	global_store_short_d16_hi v[0:1], v8, off offset:32
	global_store_short_d16_hi v[6:7], v9, off offset:32
	v_pk_mul_f32 v[8:9], v[148:149], s[0:1] op_sel_hi:[1,0]
	v_accvgpr_read_b32 v167, a123
	global_store_short_d16_hi v[2:3], v8, off offset:64
	global_store_short_d16_hi v[4:5], v9, off offset:64
	v_accvgpr_read_b32 v8, a108
	v_accvgpr_read_b32 v165, a121
	v_accvgpr_read_b32 v164, a120
	v_accvgpr_read_b32 v9, a109
	v_accvgpr_read_b32 v166, a122
	global_store_short_d16_hi v[0:1], v8, off offset:64
	global_store_short_d16_hi v[6:7], v9, off offset:64
	v_pk_mul_f32 v[8:9], v[164:165], s[0:1] op_sel_hi:[1,0]
	global_store_short_d16_hi v[2:3], v8, off offset:96
	global_store_short_d16_hi v[4:5], v9, off offset:96
	v_pk_mul_f32 v[2:3], v[166:167], s[0:1] op_sel_hi:[1,0]
	global_store_short_d16_hi v[0:1], v2, off offset:96
	global_store_short_d16_hi v[6:7], v3, off offset:96
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
	.set _Z22mxfp4_gluon_cpp_kernel13gluon_globals.numbered_sgpr, 82
	.set _Z22mxfp4_gluon_cpp_kernel13gluon_globals.private_seg_size, 0
	.set _Z22mxfp4_gluon_cpp_kernel13gluon_globals.uses_vcc, 0
	.set _Z22mxfp4_gluon_cpp_kernel13gluon_globals.uses_flat_scratch, 0
	.set _Z22mxfp4_gluon_cpp_kernel13gluon_globals.has_dyn_sized_stack, 0
	.set _Z22mxfp4_gluon_cpp_kernel13gluon_globals.has_recursion, 0
	.set _Z22mxfp4_gluon_cpp_kernel13gluon_globals.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 35424
; TotalNumSgprs: 88
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
	.type	__hip_cuid_d4b7908c1ced4a24,@object ; @__hip_cuid_d4b7908c1ced4a24
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_d4b7908c1ced4a24
__hip_cuid_d4b7908c1ced4a24:
	.byte	0                               ; 0x0
	.size	__hip_cuid_d4b7908c1ced4a24, 1

	.ident	"clang version 20.0.0git (https://github.com/ROCm/llvm-project.git 27682a16360e33e37c4f3cc6adf9a620733f8fe1)"
	.ident	"AMD clang version 20.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.1.0 25425 1b0eada6b0ee93e2e694c8c146d23fca90bc11c5)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_d4b7908c1ced4a24
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
    .vgpr_count:     512
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
