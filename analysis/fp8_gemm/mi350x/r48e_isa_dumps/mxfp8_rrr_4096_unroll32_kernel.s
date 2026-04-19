	.protected	_Z29rrr_exact_8wave_scaled_kernelILb1ELi2EEv14layout_globals ; -- Begin function _Z29rrr_exact_8wave_scaled_kernelILb1ELi2EEv14layout_globals
	.globl	_Z29rrr_exact_8wave_scaled_kernelILb1ELi2EEv14layout_globals
	.p2align	8
	.type	_Z29rrr_exact_8wave_scaled_kernelILb1ELi2EEv14layout_globals,@function
_Z29rrr_exact_8wave_scaled_kernelILb1ELi2EEv14layout_globals: ; @_Z29rrr_exact_8wave_scaled_kernelILb1ELi2EEv14layout_globals
; %bb.0:
	s_load_dwordx2 s[18:19], s[0:1], 0x60
	s_load_dwordx2 s[16:17], s[0:1], 0x90
	s_load_dword s3, s[0:1], 0x128
	s_load_dwordx2 s[30:31], s[0:1], 0x0
	s_load_dwordx2 s[10:11], s[0:1], 0x20
	s_load_dwordx2 s[28:29], s[0:1], 0x30
	s_load_dwordx2 s[26:27], s[0:1], 0x50
	s_waitcnt lgkmcnt(0)
	s_cmp_lt_i32 s3, 8
	s_cselect_b64 s[4:5], -1, 0
	s_and_b32 s6, s3, 7
	s_cmp_lg_u32 s6, 0
	s_cselect_b64 s[6:7], -1, 0
	s_or_b64 s[4:5], s[4:5], s[6:7]
	v_mov_b32_e32 v178, v0
	s_and_b64 vcc, exec, s[4:5]
	s_cbranch_vccnz .LBB3_2
; %bb.1:
	s_ashr_i32 s4, s2, 31
	s_lshr_b32 s4, s4, 29
	s_add_i32 s4, s2, s4
	s_ashr_i32 s5, s4, 3
	s_and_b32 s4, s4, -8
	s_lshr_b32 s3, s3, 3
	s_sub_i32 s2, s2, s4
	s_mul_i32 s2, s3, s2
	s_add_i32 s2, s2, s5
.LBB3_2:
	s_ashr_i32 s4, s2, 31
	s_lshr_b32 s4, s4, 26
	s_add_i32 s4, s2, s4
	s_ashr_i32 s7, s4, 6
	s_load_dword s3, s[0:1], 0x108
	s_lshl_b32 s5, s7, 2
	s_sub_i32 s4, 16, s5
	s_cmpk_gt_i32 s2, 0xff
	s_cselect_b32 s6, s4, 4
	s_mov_b32 s27, 16
	s_cmp_lt_i32 s6, 1
	s_mov_b32 s4, 16
	s_cbranch_scc1 .LBB3_4
; %bb.3:
	s_abs_i32 s4, s6
	v_cvt_f32_u32_e32 v0, s4
	s_lshl_b32 s7, s7, 6
	s_sub_i32 s2, s2, s7
	s_sub_i32 s7, 0, s4
	v_rcp_iflag_f32_e32 v0, v0
	s_abs_i32 s9, s2
	s_xor_b32 s8, s2, s6
	s_ashr_i32 s8, s8, 31
	v_mul_f32_e32 v0, 0x4f7ffffe, v0
	v_cvt_u32_f32_e32 v0, v0
	s_nop 0
	v_readfirstlane_b32 s11, v0
	s_mul_i32 s7, s7, s11
	s_mul_hi_u32 s7, s11, s7
	s_add_i32 s11, s11, s7
	s_mul_hi_u32 s7, s9, s11
	s_mul_i32 s11, s7, s4
	s_sub_i32 s9, s9, s11
	s_add_i32 s12, s7, 1
	s_sub_i32 s11, s9, s4
	s_cmp_ge_u32 s9, s4
	s_cselect_b32 s7, s12, s7
	s_cselect_b32 s9, s11, s9
	s_add_i32 s11, s7, 1
	s_cmp_ge_u32 s9, s4
	s_cselect_b32 s4, s11, s7
	s_xor_b32 s4, s4, s8
	s_sub_i32 s4, s4, s8
	s_mul_i32 s6, s4, s6
	s_sub_i32 s2, s2, s6
	s_add_i32 s27, s2, s5
.LBB3_4:
	s_waitcnt lgkmcnt(0)
	s_ashr_i32 s2, s3, 31
	s_lshr_b32 s2, s2, 27
	s_add_i32 s3, s3, s2
	s_ashr_i32 s2, s3, 5
	s_add_i32 s2, s2, 7
	v_lshlrev_b32_e32 v1, 4, v178
	s_movk_i32 s6, 0x70
	v_lshrrev_b32_e32 v2, 3, v178
	s_and_b32 s2, s2, -8
	s_ashr_i32 s5, s4, 31
	v_bitop3_b32 v0, v1, s6, v178 bitop3:0x48
	v_or_b32_e32 v3, 64, v2
	s_lshl_b32 s33, s4, 8
	s_lshl_b32 s39, s2, 7
	s_lshl_b32 s36, s2, 6
	s_lshl_b64 s[2:3], s[4:5], 2
	v_mad_u64_u32 v[248:249], s[4:5], v2, s10, v[0:1]
	v_mad_u64_u32 v[250:251], s[4:5], v3, s10, v[0:1]
	v_lshlrev_b32_e32 v0, 1, v178
	v_bitop3_b32 v0, v0, s6, v1 bitop3:0x48
	v_mad_u64_u32 v[254:255], s[4:5], v2, s26, v[0:1]
	v_mad_u64_u32 v[252:253], s[4:5], v3, s26, v[0:1]
	v_and_b32_e32 v0, 0x1c00, v1
	v_and_b32_e32 v1, 0x180, v178
	s_ashr_i32 s20, s27, 31
	s_ashr_i32 s5, s33, 31
	v_or_b32_e32 v103, v0, v1
	v_or_b32_e32 v2, 0x2200, v1
	s_add_u32 s4, s28, s33
	v_readfirstlane_b32 s8, v103
	v_or_b32_e32 v104, v0, v2
	s_addc_u32 s5, s29, s5
	s_lshl_b32 s6, s26, 7
	s_mov_b32 s7, 0x110000
	s_mov_b32 m0, s8
	v_readfirstlane_b32 s8, v104
	buffer_load_dwordx4 v254, s[4:7], 0 offen lds
	s_mov_b32 m0, s8
	s_mul_i32 s8, s27, s10
	s_lshl_b32 s21, s8, 8
	s_ashr_i32 s9, s21, 31
	v_add_u32_e32 v10, 0x11000, v0
	s_add_u32 s8, s30, s21
	v_readfirstlane_b32 s12, v10
	v_add_u32_e32 v150, 0x2000, v10
	buffer_load_dwordx4 v252, s[4:7], 0 offen lds
	s_addc_u32 s9, s31, s9
	s_lshl_b32 s10, s10, 7
	s_mov_b32 s11, s7
	s_mov_b32 m0, s12
	v_readfirstlane_b32 s12, v150
	v_add_u32_e32 v3, 0x4400, v0
	buffer_load_dwordx4 v248, s[8:11], 0 offen lds
	s_mov_b32 m0, s12
	v_or_b32_e32 v151, v3, v1
	buffer_load_dwordx4 v250, s[8:11], 0 offen lds
	s_add_u32 s12, s4, 0x80
	v_readfirstlane_b32 s11, v151
	v_add_u32_e32 v152, v3, v2
	s_addc_u32 s13, s5, 0
	s_mov_b32 s14, s6
	s_mov_b32 s15, s7
	s_mov_b32 m0, s11
	v_readfirstlane_b32 s11, v152
	buffer_load_dwordx4 v254, s[12:15], 0 offen lds
	s_mov_b32 m0, s11
	s_add_i32 s11, s21, s10
	s_ashr_i32 s40, s11, 31
	v_or_b32_e32 v218, 0x4000, v10
	buffer_load_dwordx4 v252, s[12:15], 0 offen lds
	s_add_u32 s12, s30, s11
	v_readfirstlane_b32 s21, v218
	v_add_u32_e32 v219, 0x6000, v10
	s_addc_u32 s13, s31, s40
	s_mov_b32 s14, s10
	s_mov_b32 m0, s21
	v_readfirstlane_b32 s21, v219
	buffer_load_dwordx4 v248, s[12:15], 0 offen lds
	s_mov_b32 m0, s21
	v_lshrrev_b32_e32 v9, 8, v178
	buffer_load_dwordx4 v250, s[12:15], 0 offen lds
	s_load_dwordx2 s[14:15], s[0:1], 0xc0
	s_load_dwordx2 s[24:25], s[0:1], 0xe0
	v_bfe_u32 v3, v178, 6, 2
	v_lshl_or_b32 v6, s27, 1, v9
	v_mov_b64_e32 v[4:5], s[18:19]
	v_or_b32_e32 v8, s2, v3
	v_mad_u64_u32 v[4:5], s[18:19], v6, s39, v[4:5]
	v_mov_b64_e32 v[6:7], s[16:17]
	s_mul_i32 s20, s20, s39
	s_mul_i32 s18, s3, s36
	v_mad_u64_u32 v[6:7], s[2:3], v8, s36, v[6:7]
	v_add_u32_e32 v5, s20, v5
	v_add_u32_e32 v7, s18, v7
	v_readfirstlane_b32 s21, v4
	v_readfirstlane_b32 s20, v5
	v_readfirstlane_b32 s37, v6
	s_waitcnt lgkmcnt(0)
	v_readfirstlane_b32 s25, v7
	s_movk_i32 s34, 0x1c00
	s_movk_i32 s35, 0x2000
	v_cmp_eq_u32_e32 vcc, 1, v9
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB3_6
; %bb.5:
	s_barrier
.LBB3_6:
	s_or_b64 exec, exec, s[2:3]
	v_lshlrev_b32_e32 v7, 5, v3
	v_add_u32_e32 v3, 0x8800, v0
	s_ashr_i32 s2, s6, 31
	v_add_u32_e32 v5, v3, v1
	s_mov_b64 s[18:19], s[6:7]
	s_add_u32 s48, s4, s6
	v_readfirstlane_b32 s46, v5
	v_add_u32_e32 v3, v3, v2
	s_mov_b64 s[16:17], s[4:5]
	s_addc_u32 s49, s5, s2
	s_mov_b32 s50, s6
	s_mov_b32 s51, s7
	s_mov_b32 m0, s46
	v_readfirstlane_b32 s47, v3
	v_or_b32_e32 v236, 0x8000, v10
	s_mov_b32 s16, s21
	s_mov_b32 s17, s20
	s_mov_b64 s[22:23], s[6:7]
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	buffer_load_dwordx4 v254, s[48:51], 0 offen lds
	s_mov_b32 m0, s47
	s_add_u32 s52, s8, 0x80
	v_readfirstlane_b32 s2, v236
	v_add_u32_e32 v237, 0xa000, v10
	v_add_u32_e32 v0, 0xcc00, v0
	s_mov_b64 s[20:21], s[4:5]
	buffer_load_dwordx4 v252, s[48:51], 0 offen lds
	s_addc_u32 s53, s9, 0
	s_mov_b32 s54, s10
	s_mov_b32 s55, s7
	s_mov_b32 m0, s2
	v_readfirstlane_b32 s2, v237
	v_add_u32_e32 v1, v0, v1
	v_and_b32_e32 v6, 15, v178
	s_mov_b32 s20, s37
	buffer_load_dwordx4 v248, s[52:55], 0 offen lds
	s_mov_b32 m0, s2
	s_add_u32 s48, s48, 0x80
	v_readfirstlane_b32 s37, v1
	v_add_u32_e32 v0, v0, v2
	v_bfe_u32 v4, v178, 4, 2
	buffer_load_dwordx4 v250, s[52:55], 0 offen lds
	s_addc_u32 s49, s49, 0
	s_mov_b32 m0, s37
	scratch_store_dword off, v0, off offset:88 ; 4-byte Folded Spill
	v_readfirstlane_b32 s38, v0
	v_lshlrev_b32_e32 v0, 4, v6
	buffer_load_dwordx4 v254, s[48:51], 0 offen lds
	s_mov_b32 m0, s38
	v_lshl_or_b32 v244, v4, 8, v0
	v_mov_b32_e32 v0, v6
	scratch_store_dword off, v5, off offset:104 ; 4-byte Folded Spill
	scratch_store_dword off, v3, off offset:108 ; 4-byte Folded Spill
	scratch_store_dword off, v236, off offset:96 ; 4-byte Folded Spill
	scratch_store_dword off, v237, off offset:136 ; 4-byte Folded Spill
	scratch_store_dword off, v1, off offset:84 ; 4-byte Folded Spill
	buffer_load_dwordx4 v252, s[48:51], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(6)
	;;#ASMEND
	s_barrier
	scratch_store_dwordx2 off, v[0:1], off offset:176 ; 8-byte Folded Spill
	v_lshlrev_b32_e32 v0, 3, v6
	v_lshl_or_b32 v243, v4, 7, v0
	v_bfe_u32 v0, v178, 1, 3
	v_lshlrev_b32_e32 v3, 11, v4
	v_add_u32_e32 v5, v4, v0
	v_lshlrev_b32_e32 v1, 3, v178
	v_lshl_or_b32 v22, v5, 7, v3
	v_or_b32_e32 v3, 4, v4
	v_and_b32_e32 v2, 8, v1
	v_lshlrev_b32_e32 v184, 4, v0
	v_lshlrev_b32_e32 v4, 11, v3
	v_add_u32_e32 v0, v3, v0
	v_or_b32_e32 v185, v7, v2
	v_lshl_or_b32 v0, v0, 7, v4
	v_lshlrev_b32_e32 v4, 7, v178
	v_mov_b32_e32 v12, v10
	v_bitop3_b32 v6, v7, v184, v2 bitop3:0x36
	v_bitop3_b32 v2, v185, v184, 16 bitop3:0x36
	v_and_b32_e32 v3, 48, v178
	v_and_b32_e32 v4, 0x780, v4
	scratch_store_dword off, v7, off offset:184 ; 4-byte Folded Spill
	v_or_b32_e32 v7, v2, v22
	v_lshlrev_b32_e32 v23, 13, v9
	v_or_b32_e32 v5, v4, v3
	v_and_b32_e32 v1, 0x70, v1
	v_or_b32_e32 v27, 0xc000, v12
	v_add_u32_e32 v26, 0xe000, v12
	v_mov_b32_e32 v153, v10
	s_mov_b32 s21, s25
	v_or_b32_e32 v235, v22, v6
	scratch_store_dword off, v7, off offset:56 ; 4-byte Folded Spill
	v_or_b32_e32 v238, v0, v2
	scratch_store_dword off, v9, off offset:172 ; 4-byte Folded Spill
	v_or_b32_e32 v2, 0x11000, v23
	v_bitop3_b32 v24, v4, v1, v3 bitop3:0x36
	v_bitop3_b32 v1, v5, v1, 64 bitop3:0x36
	scratch_store_dword off, v27, off offset:80 ; 4-byte Folded Spill
	scratch_store_dword off, v26, off offset:76 ; 4-byte Folded Spill
	scratch_store_dword off, v153, off offset:92 ; 4-byte Folded Spill
	s_mov_b32 s18, s39
	s_mov_b32 s22, s36
	v_or_b32_e32 v255, v24, v2
	v_or_b32_e32 v251, v1, v2
	buffer_load_dwordx4 v[18:21], v244, s[16:19], 0 offen
	buffer_load_dwordx2 v[182:183], v243, s[20:23], 0 offen
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v235 offset:0
ds_read_b64_tr_b8 v[4:5], v235 offset:1024

	;;#ASMEND
	scratch_store_dword off, v235, off offset:60 ; 4-byte Folded Spill
	v_or_b32_e32 v14, v0, v6
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v7 offset:0
ds_read_b64_tr_b8 v[12:13], v7 offset:1024

	;;#ASMEND
	scratch_store_dword off, v14, off offset:24 ; 4-byte Folded Spill
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v14 offset:0
ds_read_b64_tr_b8 v[8:9], v14 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v238 offset:0
ds_read_b64_tr_b8 v[16:17], v238 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[54:57], v255 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[62:65], v255 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[70:73], v255 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[78:81], v255 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[58:61], v251 offset:0

	;;#ASMEND
	s_add_u32 s11, s30, s11
	;;#ASMSTART
	ds_read_b128 v[66:69], v251 offset:0x800

	;;#ASMEND
	s_addc_u32 s30, s31, s40
	;;#ASMSTART
	ds_read_b128 v[74:77], v251 offset:0x1000

	;;#ASMEND
	s_add_u32 s40, s11, 0x80
	v_readfirstlane_b32 s31, v27
	;;#ASMSTART
	ds_read_b128 v[82:85], v251 offset:0x1800

	;;#ASMEND
	s_addc_u32 s41, s30, 0
	s_mov_b32 s42, s10
	s_mov_b32 s43, s7
	s_mov_b32 m0, s31
	v_readfirstlane_b32 s36, v26
	buffer_load_dwordx4 v248, s[40:43], 0 offen lds
	s_mov_b32 m0, s36
	s_load_dword s25, s[0:1], 0xf4
	buffer_load_dwordx4 v250, s[40:43], 0 offen lds
	v_or_b32_e32 v25, 16, v185
	v_add_u32_e32 v26, 0x4400, v22
	v_bitop3_b32 v239, v26, v185, v184 bitop3:0xf6
	v_bitop3_b32 v91, v25, v26, v184 bitop3:0xde
	v_or_b32_e32 v26, 0x15000, v23
	v_add_u32_e32 v90, 0x4400, v0
	v_or_b32_e32 v253, v24, v26
	v_or_b32_e32 v249, v1, v26
	v_add_u32_e32 v26, 0x8800, v22
	v_add_u32_e32 v194, 0x8800, v0
	v_add_u32_e32 v22, 0xcc00, v22
	v_bitop3_b32 v102, v90, v25, v184 bitop3:0xf6
	v_bitop3_b32 v242, v26, v185, v184 bitop3:0xf6
	v_bitop3_b32 v196, v25, v26, v184 bitop3:0xde
	v_bitop3_b32 v195, v194, v25, v184 bitop3:0xf6
	v_or_b32_e32 v26, 0x19000, v23
	v_bitop3_b32 v223, v22, v185, v184 bitop3:0xf6
	v_bitop3_b32 v222, v25, v22, v184 bitop3:0xde
	v_add_u32_e32 v0, 0xcc00, v0
	v_or_b32_e32 v22, 0x1d000, v23
	scratch_store_dword off, v91, off offset:20 ; 4-byte Folded Spill
	scratch_store_dword off, v102, off offset:4 ; 4-byte Folded Spill
	scratch_store_dword off, v196, off offset:52 ; 4-byte Folded Spill
	scratch_store_dword off, v195, off offset:48 ; 4-byte Folded Spill
	v_or_b32_e32 v240, v24, v26
	v_or_b32_e32 v241, v1, v26
	scratch_store_dword off, v223, off offset:28 ; 4-byte Folded Spill
	scratch_store_dword off, v222, off offset:140 ; 4-byte Folded Spill
	v_bitop3_b32 v234, v0, v25, v184 bitop3:0xf6
	v_or_b32_e32 v245, v24, v22
	v_or_b32_e32 v247, v1, v22
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	s_waitcnt vmcnt(10)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[54:61], v[2:9], 0, v18, v182 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[54:61], v[10:17], 0, v18, v182 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[62:69], v[2:9], 0, v18, v182 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[62:69], v[10:17], 0, v18, v182 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[70:77], v[2:9], 0, v20, v182 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[70:77], v[10:17], 0, v20, v182 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], v[78:85], v[2:9], 0, v20, v182 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[78:85], v[10:17], 0, v20, v182 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[86:87], v239 offset:0
ds_read_b64_tr_b8 v[88:89], v239 offset:1024

	;;#ASMEND
	s_lshl_b32 s0, s26, 8
	;;#ASMSTART
	ds_read_b64_tr_b8 v[94:95], v91 offset:0
ds_read_b64_tr_b8 v[96:97], v91 offset:1024

	;;#ASMEND
	s_ashr_i32 s1, s0, 31
	v_bitop3_b32 v1, v90, v185, v184 bitop3:0xf6
	;;#ASMSTART
	ds_read_b64_tr_b8 v[90:91], v1 offset:0
ds_read_b64_tr_b8 v[92:93], v1 offset:1024

	;;#ASMEND
	s_add_u32 s0, s4, s0
	v_readfirstlane_b32 s39, v103
	;;#ASMSTART
	ds_read_b64_tr_b8 v[98:99], v102 offset:0
ds_read_b64_tr_b8 v[100:101], v102 offset:1024

	;;#ASMEND
	s_addc_u32 s1, s5, s1
	s_mov_b32 s2, s6
	s_mov_b32 s3, s7
	s_mov_b32 m0, s39
	v_readfirstlane_b32 s40, v104
	buffer_load_dwordx4 v254, s[0:3], 0 offen lds
	s_mov_b32 m0, s40
	s_nop 0
	buffer_load_dwordx4 v252, s[0:3], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[118:121], v[54:61], v[86:93], 0, v18, v183 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[122:125], v[54:61], v[94:101], 0, v18, v183 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[126:129], v[62:69], v[86:93], 0, v18, v183 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[130:133], v[62:69], v[94:101], 0, v18, v183 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[134:137], v[70:77], v[86:93], 0, v20, v183 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[138:141], v[70:77], v[94:101], 0, v20, v183 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[142:145], v[78:85], v[86:93], 0, v20, v183 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[146:149], v[78:85], v[94:101], 0, v20, v183 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	scratch_store_dword off, v103, off offset:116 ; 4-byte Folded Spill
	scratch_store_dword off, v104, off offset:112 ; 4-byte Folded Spill
	;;#ASMSTART
	ds_read_b128 v[102:105], v253 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[110:113], v253 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[170:173], v253 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[186:189], v253 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[106:109], v249 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[114:117], v249 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[174:177], v249 offset:0x1000

	;;#ASMEND
	s_add_u32 s48, s0, 0x80
	v_readfirstlane_b32 s2, v151
	;;#ASMSTART
	ds_read_b128 v[190:193], v249 offset:0x1800

	;;#ASMEND
	s_addc_u32 s49, s1, 0
	s_mov_b32 m0, s2
	v_readfirstlane_b32 s41, v152
	buffer_load_dwordx4 v254, s[48:51], 0 offen lds
	s_mov_b32 m0, s41
	s_nop 0
	buffer_load_dwordx4 v252, s[48:51], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[102:109], v[2:9], 0, v19, v182 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[102:109], v[10:17], 0, v19, v182 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[110:117], v[2:9], 0, v19, v182 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[110:117], v[10:17], 0, v19, v182 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[170:177], v[2:9], 0, v21, v182 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[170:177], v[10:17], 0, v21, v182 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[186:193], v[2:9], 0, v21, v182 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[186:193], v[10:17], 0, v21, v182 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_u32 s48, s8, 0x100
	v_readfirstlane_b32 s42, v153
	s_addc_u32 s49, s9, 0
	s_mov_b32 s50, s10
	s_mov_b32 m0, s42
	v_readfirstlane_b32 s43, v150
	buffer_load_dwordx4 v248, s[48:51], 0 offen lds
	s_mov_b32 m0, s43
	s_mov_b64 s[0:1], 0x100
	buffer_load_dwordx4 v250, s[48:51], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	scratch_store_dword off, v150, off offset:100 ; 4-byte Folded Spill
	scratch_store_dword off, v151, off offset:132 ; 4-byte Folded Spill
	scratch_store_dword off, v152, off offset:128 ; 4-byte Folded Spill
	scratch_store_dword off, v178, off offset:168 ; 4-byte Folded Spill
	v_mfma_scale_f32_16x16x128_f8f6f4 v[178:181], v[186:193], v[94:101], 0, v21, v183 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[150:153], v[102:109], v[86:93], 0, v19, v183 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[154:157], v[102:109], v[94:101], 0, v19, v183 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[158:161], v[110:117], v[86:93], 0, v19, v183 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[162:165], v[110:117], v[94:101], 0, v19, v183 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[166:169], v[170:177], v[86:93], 0, v21, v183 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[170:173], v[170:177], v[94:101], 0, v21, v183 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[174:177], v[186:193], v[86:93], 0, v21, v183 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v242 offset:0
ds_read_b64_tr_b8 v[4:5], v242 offset:1024

	;;#ASMEND
	scratch_store_dword off, v242, off offset:44 ; 4-byte Folded Spill
	v_bitop3_b32 v14, v194, v185, v184 bitop3:0xf6
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v196 offset:0
ds_read_b64_tr_b8 v[12:13], v196 offset:1024

	;;#ASMEND
	scratch_store_dword off, v14, off       ; 4-byte Folded Spill
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v14 offset:0
ds_read_b64_tr_b8 v[8:9], v14 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v195 offset:0
ds_read_b64_tr_b8 v[16:17], v195 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[186:189], v240 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[194:197], v240 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[202:205], v240 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[210:213], v240 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[190:193], v241 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[198:201], v241 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[206:209], v241 offset:0x1000

	;;#ASMEND
	s_add_u32 s48, s11, 0x100
	v_readfirstlane_b32 s44, v218
	;;#ASMSTART
	ds_read_b128 v[214:217], v241 offset:0x1800

	;;#ASMEND
	s_addc_u32 s49, s30, 0
	s_mov_b32 m0, s44
	v_readfirstlane_b32 s45, v219
	buffer_load_dwordx4 v248, s[48:51], 0 offen lds
	s_mov_b32 m0, s45
	s_nop 0
	buffer_load_dwordx4 v250, s[48:51], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[186:193], v[2:9], v[22:25], v18, v182 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[186:193], v[10:17], v[26:29], v18, v182 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[194:201], v[2:9], v[30:33], v18, v182 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[98:101], v[194:201], v[10:17], v[34:37], v18, v182 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[102:105], v[202:209], v[2:9], v[38:41], v20, v182 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[106:109], v[202:209], v[10:17], v[42:45], v20, v182 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[110:113], v[210:217], v[2:9], v[46:49], v20, v182 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[114:117], v[210:217], v[10:17], v[50:53], v20, v182 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	scratch_store_dword off, v218, off offset:124 ; 4-byte Folded Spill
	scratch_store_dword off, v219, off offset:120 ; 4-byte Folded Spill
	s_mul_i32 s3, s26, 0x180
	;;#ASMSTART
	ds_read_b64_tr_b8 v[218:219], v223 offset:0
ds_read_b64_tr_b8 v[220:221], v223 offset:1024

	;;#ASMEND
	s_add_i32 s48, s3, s33
	;;#ASMSTART
	ds_read_b64_tr_b8 v[226:227], v222 offset:0
ds_read_b64_tr_b8 v[228:229], v222 offset:1024

	;;#ASMEND
	s_ashr_i32 s49, s48, 31
	v_bitop3_b32 v0, v0, v185, v184 bitop3:0xf6
	;;#ASMSTART
	ds_read_b64_tr_b8 v[222:223], v0 offset:0
ds_read_b64_tr_b8 v[224:225], v0 offset:1024

	;;#ASMEND
	s_add_u32 s52, s28, s48
	;;#ASMSTART
	ds_read_b64_tr_b8 v[230:231], v234 offset:0
ds_read_b64_tr_b8 v[232:233], v234 offset:1024

	;;#ASMEND
	s_addc_u32 s53, s29, s49
	s_mov_b32 s54, s6
	s_mov_b32 m0, s46
	s_nop 0
	buffer_load_dwordx4 v254, s[52:55], 0 offen lds
	s_mov_b32 m0, s47
	s_nop 0
	buffer_load_dwordx4 v252, s[52:55], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[186:193], v[218:225], v[118:121], v18, v183 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[186:193], v[226:233], v[122:125], v18, v183 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[194:201], v[218:225], v[126:129], v18, v183 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[194:201], v[226:233], v[130:133], v18, v183 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[202:209], v[218:225], v[134:137], v20, v183 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[202:209], v[226:233], v[138:141], v20, v183 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], v[210:217], v[218:225], v[142:145], v20, v183 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[210:217], v[226:233], v[146:149], v20, v183 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[122:125], v245 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[130:133], v245 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[138:141], v245 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[184:187], v245 offset:0x1800

	;;#ASMEND
	s_ashr_i32 s46, s3, 31
	;;#ASMSTART
	ds_read_b128 v[126:129], v247 offset:0

	;;#ASMEND
	s_add_u32 s3, s4, s3
	;;#ASMSTART
	ds_read_b128 v[134:137], v247 offset:0x800

	;;#ASMEND
	s_addc_u32 s46, s5, s46
	;;#ASMSTART
	ds_read_b128 v[142:145], v247 offset:0x1000

	;;#ASMEND
	s_add_u32 s52, s3, 0x80
	;;#ASMSTART
	ds_read_b128 v[188:191], v247 offset:0x1800

	;;#ASMEND
	s_addc_u32 s53, s46, 0
	s_mov_b32 m0, s37
	s_nop 0
	buffer_load_dwordx4 v254, s[52:55], 0 offen lds
	s_mov_b32 m0, s38
	s_nop 0
	buffer_load_dwordx4 v252, s[52:55], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[122:129], v[2:9], v[54:57], v19, v182 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[122:129], v[10:17], v[58:61], v19, v182 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[130:137], v[2:9], v[62:65], v19, v182 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[130:137], v[10:17], v[66:69], v19, v182 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[138:145], v[2:9], v[70:73], v21, v182 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[138:145], v[10:17], v[74:77], v21, v182 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[184:191], v[2:9], v[78:81], v21, v182 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[184:191], v[10:17], v[82:85], v21, v182 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_u32 s52, s8, 0x180
	v_readfirstlane_b32 s37, v236
	s_addc_u32 s53, s9, 0
	s_mov_b32 s54, s10
	s_mov_b32 m0, s37
	v_readfirstlane_b32 s38, v237
	buffer_load_dwordx4 v248, s[52:55], 0 offen lds
	s_mov_b32 m0, s38
	s_nop 0
	buffer_load_dwordx4 v250, s[52:55], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[118:121], v[122:129], v[218:225], v[150:153], v19, v183 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[122:125], v[122:129], v[226:233], v[154:157], v19, v183 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[126:129], v[130:137], v[218:225], v[158:161], v19, v183 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[130:133], v[130:137], v[226:233], v[162:165], v19, v183 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[134:137], v[138:145], v[218:225], v[166:169], v21, v183 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[138:141], v[138:145], v[226:233], v[170:173], v21, v183 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[142:145], v[184:191], v[218:225], v[174:177], v21, v183 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[146:149], v[184:191], v[226:233], v[178:181], v21, v183 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	v_mov_b32_e32 v236, v243
	s_barrier
	s_movk_i32 s3, 0x400
	s_movk_i32 s46, 0x200
	scratch_store_dword off, v236, off offset:64 ; 4-byte Folded Spill
	buffer_load_dwordx4 v[18:21], v244, s[16:19], s3 offen
	buffer_load_dwordx2 v[214:215], v243, s[20:23], s46 offen
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v235 offset:0
ds_read_b64_tr_b8 v[4:5], v235 offset:1024

	;;#ASMEND
	scratch_load_dword v243, off, off offset:56 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v243 offset:0
ds_read_b64_tr_b8 v[12:13], v243 offset:1024

	;;#ASMEND
	scratch_load_dword v246, off, off offset:24 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v246 offset:0
ds_read_b64_tr_b8 v[8:9], v246 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v238 offset:0
ds_read_b64_tr_b8 v[16:17], v238 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[154:157], v255 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[162:165], v255 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[170:173], v255 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[216:219], v255 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[158:161], v251 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[166:169], v251 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[174:177], v251 offset:0x1000

	;;#ASMEND
	s_add_u32 s52, s11, 0x180
	;;#ASMSTART
	ds_read_b128 v[220:223], v251 offset:0x1800

	;;#ASMEND
	s_addc_u32 s53, s30, 0
	s_mov_b32 m0, s31
	v_mov_b32_e32 v235, v238
	buffer_load_dwordx4 v248, s[52:55], 0 offen lds
	s_mov_b32 m0, s36
	s_nop 0
	buffer_load_dwordx4 v250, s[52:55], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[182:185], v[154:161], v[2:9], v[86:89], v18, v214 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[186:189], v[154:161], v[10:17], v[90:93], v18, v214 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[190:193], v[162:169], v[2:9], v[94:97], v18, v214 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[194:197], v[162:169], v[10:17], v[98:101], v18, v214 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[198:201], v[170:177], v[2:9], v[102:105], v20, v214 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[202:205], v[170:177], v[10:17], v[106:109], v20, v214 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[206:209], v[216:223], v[2:9], v[110:113], v20, v214 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[210:213], v[216:223], v[10:17], v[114:117], v20, v214 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[110:111], v239 offset:0
ds_read_b64_tr_b8 v[112:113], v239 offset:1024

	;;#ASMEND
	s_add_i32 s31, s48, s6
	scratch_load_dword v237, off, off offset:20 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[224:225], v237 offset:0
ds_read_b64_tr_b8 v[226:227], v237 offset:1024

	;;#ASMEND
	s_lshl_b32 s36, s26, 9
	s_ashr_i32 s46, s31, 31
	;;#ASMSTART
	ds_read_b64_tr_b8 v[114:115], v1 offset:0
ds_read_b64_tr_b8 v[116:117], v1 offset:1024

	;;#ASMEND
	s_add_u32 s48, s28, s31
	v_mov_b32_e32 v238, v239
	v_mov_b32_e32 v239, v1
	scratch_load_dword v1, off, off offset:4 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[228:229], v1 offset:0
ds_read_b64_tr_b8 v[230:231], v1 offset:1024

	;;#ASMEND
	s_addc_u32 s49, s29, s46
	s_mov_b32 s50, s6
	s_mov_b32 m0, s39
	s_nop 0
	buffer_load_dwordx4 v254, s[48:51], 0 offen lds
	s_mov_b32 m0, s40
	s_nop 0
	buffer_load_dwordx4 v252, s[48:51], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[178:181], v[216:223], v[224:231], v[50:53], v20, v215 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[150:153], v[154:161], v[110:117], v[22:25], v18, v215 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[154:157], v[154:161], v[224:231], v[26:29], v18, v215 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[158:161], v[162:169], v[110:117], v[30:33], v18, v215 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[162:165], v[162:169], v[224:231], v[34:37], v18, v215 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[166:169], v[170:177], v[110:117], v[38:41], v20, v215 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[170:173], v[170:177], v[224:231], v[42:45], v20, v215 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[174:177], v[216:223], v[110:117], v[46:49], v20, v215 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[22:25], v253 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[30:33], v253 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[38:41], v253 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[46:49], v253 offset:0x1800

	;;#ASMEND
	s_ashr_i32 s39, s36, 31
	;;#ASMSTART
	ds_read_b128 v[26:29], v249 offset:0

	;;#ASMEND
	s_add_u32 s36, s4, s36
	;;#ASMSTART
	ds_read_b128 v[34:37], v249 offset:0x800

	;;#ASMEND
	s_addc_u32 s39, s5, s39
	;;#ASMSTART
	ds_read_b128 v[42:45], v249 offset:0x1000

	;;#ASMEND
	s_add_u32 s48, s36, 0x80
	;;#ASMSTART
	ds_read_b128 v[50:53], v249 offset:0x1800

	;;#ASMEND
	s_addc_u32 s49, s39, 0
	s_mov_b32 m0, s2
	s_nop 0
	buffer_load_dwordx4 v254, s[48:51], 0 offen lds
	s_mov_b32 m0, s41
	s_nop 0
	buffer_load_dwordx4 v252, s[48:51], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[22:29], v[2:9], v[54:57], v19, v214 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[22:29], v[10:17], v[58:61], v19, v214 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[30:37], v[2:9], v[62:65], v19, v214 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[30:37], v[10:17], v[66:69], v19, v214 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[38:45], v[2:9], v[70:73], v21, v214 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[38:45], v[10:17], v[74:77], v21, v214 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[46:53], v[2:9], v[78:81], v21, v214 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[46:53], v[10:17], v[82:85], v21, v214 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_u32 s48, s8, 0x200
	s_addc_u32 s49, s9, 0
	s_mov_b32 s50, s10
	s_mov_b32 m0, s42
	s_nop 0
	buffer_load_dwordx4 v248, s[48:51], 0 offen lds
	s_mov_b32 m0, s43
	s_nop 0
	buffer_load_dwordx4 v250, s[48:51], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[22:29], v[110:117], v[118:121], v19, v215 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[22:29], v[224:231], v[122:125], v19, v215 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[30:37], v[110:117], v[126:129], v19, v215 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[98:101], v[30:37], v[224:231], v[130:133], v19, v215 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[102:105], v[38:45], v[110:117], v[134:137], v21, v215 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[106:109], v[38:45], v[224:231], v[138:141], v21, v215 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[110:113], v[46:53], v[110:117], v[142:145], v21, v215 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[114:117], v[46:53], v[224:231], v[146:149], v21, v215 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v242 offset:0
ds_read_b64_tr_b8 v[4:5], v242 offset:1024

	;;#ASMEND
	scratch_load_dword v6, off, off offset:52 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v6 offset:0
ds_read_b64_tr_b8 v[12:13], v6 offset:1024

	;;#ASMEND
	scratch_load_dword v242, off, off       ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v242 offset:0
ds_read_b64_tr_b8 v[8:9], v242 offset:1024

	;;#ASMEND
	scratch_load_dword v22, off, off offset:48 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v22 offset:0
ds_read_b64_tr_b8 v[16:17], v22 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[122:125], v240 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[130:133], v240 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[138:141], v240 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[216:219], v240 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[126:129], v241 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[134:137], v241 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[142:145], v241 offset:0x1000

	;;#ASMEND
	s_add_u32 s40, s11, 0x200
	;;#ASMSTART
	ds_read_b128 v[220:223], v241 offset:0x1800

	;;#ASMEND
	s_addc_u32 s41, s30, 0
	s_mov_b32 s42, s10
	s_mov_b32 s43, s7
	s_mov_b32 m0, s44
	s_nop 0
	buffer_load_dwordx4 v248, s[40:43], 0 offen lds
	s_mov_b32 m0, s45
	s_nop 0
	buffer_load_dwordx4 v250, s[40:43], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[2:9], v[182:185], v18, v214 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[122:129], v[10:17], v[186:189], v18, v214 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[130:137], v[2:9], v[190:193], v18, v214 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[130:137], v[10:17], v[194:197], v18, v214 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[138:145], v[2:9], v[198:201], v20, v214 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[138:145], v[10:17], v[202:205], v20, v214 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], v[216:223], v[2:9], v[206:209], v20, v214 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[216:223], v[10:17], v[210:213], v20, v214 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	scratch_load_dword v233, off, off offset:28 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[182:183], v233 offset:0
ds_read_b64_tr_b8 v[184:185], v233 offset:1024

	;;#ASMEND
	scratch_load_dword v232, off, off offset:140 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[190:191], v232 offset:0
ds_read_b64_tr_b8 v[192:193], v232 offset:1024

	;;#ASMEND
	scratch_store_dword off, v0, off offset:32 ; 4-byte Folded Spill
	;;#ASMSTART
	ds_read_b64_tr_b8 v[186:187], v0 offset:0
ds_read_b64_tr_b8 v[188:189], v0 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[194:195], v234 offset:0
ds_read_b64_tr_b8 v[196:197], v234 offset:1024

	;;#ASMEND
	scratch_load_dword v118, off, off offset:104 ; 4-byte Folded Reload
	s_add_i32 s40, s31, s6
	s_ashr_i32 s2, s40, 31
	s_add_u32 s48, s28, s40
	s_addc_u32 s49, s29, s2
	s_mov_b32 s50, s6
	s_mul_i32 s2, s26, 0x280
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s47, v118
	scratch_load_dword v118, off, off offset:108 ; 4-byte Folded Reload
	s_mov_b32 m0, s47
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s31, v118
	buffer_load_dwordx4 v254, s[48:51], 0 offen lds
	s_mov_b32 m0, s31
	s_nop 0
	buffer_load_dwordx4 v252, s[48:51], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[118:121], v[122:129], v[182:189], v[150:153], v18, v215 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[122:125], v[122:129], v[190:197], v[154:157], v18, v215 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[126:129], v[130:137], v[182:189], v[158:161], v18, v215 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[130:133], v[130:137], v[190:197], v[162:165], v18, v215 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[134:137], v[138:145], v[182:189], v[166:169], v20, v215 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[138:141], v[138:145], v[190:197], v[170:173], v20, v215 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[142:145], v[216:223], v[182:189], v[174:177], v20, v215 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[146:149], v[216:223], v[190:197], v[178:181], v20, v215 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[198:201], v245 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[206:209], v245 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[216:219], v245 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[224:227], v245 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[202:205], v247 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[210:213], v247 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[220:223], v247 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[228:231], v247 offset:0x1800

	;;#ASMEND
	scratch_load_dword v18, off, off offset:84 ; 4-byte Folded Reload
	s_ashr_i32 s36, s2, 31
	s_add_u32 s2, s4, s2
	s_addc_u32 s36, s5, s36
	s_add_u32 s48, s2, 0x80
	s_addc_u32 s49, s36, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s36, v18
	scratch_load_dword v18, off, off offset:88 ; 4-byte Folded Reload
	s_mov_b32 m0, s36
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s39, v18
	buffer_load_dwordx4 v254, s[48:51], 0 offen lds
	s_mov_b32 m0, s39
	s_nop 0
	buffer_load_dwordx4 v252, s[48:51], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[178:181], v[224:231], v[10:17], v[82:85], v21, v214 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[150:153], v[198:205], v[2:9], v[54:57], v19, v214 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[154:157], v[198:205], v[10:17], v[58:61], v19, v214 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[158:161], v[206:213], v[2:9], v[62:65], v19, v214 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[162:165], v[206:213], v[10:17], v[66:69], v19, v214 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[166:169], v[216:223], v[2:9], v[70:73], v21, v214 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[170:173], v[216:223], v[10:17], v[74:77], v21, v214 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[174:177], v[224:231], v[2:9], v[78:81], v21, v214 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_u32 s48, s8, 0x280
	s_addc_u32 s49, s9, 0
	s_mov_b32 s50, s10
	s_mov_b32 m0, s37
	s_nop 0
	buffer_load_dwordx4 v248, s[48:51], 0 offen lds
	s_mov_b32 m0, s38
	s_nop 0
	buffer_load_dwordx4 v250, s[48:51], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[198:205], v[182:189], v[86:89], v19, v215 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[198:205], v[190:197], v[90:93], v19, v215 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[206:213], v[182:189], v[94:97], v19, v215 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[206:213], v[190:197], v[98:101], v19, v215 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[216:223], v[182:189], v[102:105], v21, v215 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[216:223], v[190:197], v[106:109], v21, v215 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[224:231], v[182:189], v[110:113], v21, v215 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[224:231], v[190:197], v[114:117], v21, v215 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	s_movk_i32 s2, 0x800
	scratch_store_dword off, v244, off offset:148 ; 4-byte Folded Spill
	buffer_load_dwordx4 v[18:21], v244, s[16:19], s2 offen
	buffer_load_dwordx2 v[182:183], v236, s[20:23], s3 offen
	s_add_u32 s48, s11, 0x280
	scratch_load_dword v236, off, off offset:60 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v236 offset:0
ds_read_b64_tr_b8 v[4:5], v236 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v243 offset:0
ds_read_b64_tr_b8 v[12:13], v243 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v246 offset:0
ds_read_b64_tr_b8 v[8:9], v246 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v235 offset:0
ds_read_b64_tr_b8 v[16:17], v235 offset:1024

	;;#ASMEND
	scratch_store_dword off, v235, off offset:12 ; 4-byte Folded Spill
	;;#ASMSTART
	ds_read_b128 v[90:93], v255 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[98:101], v255 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[106:109], v255 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[184:187], v255 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[94:97], v251 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[102:105], v251 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[110:113], v251 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[188:191], v251 offset:0x1800

	;;#ASMEND
	scratch_load_dword v86, off, off offset:80 ; 4-byte Folded Reload
	s_addc_u32 s49, s30, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s3, v86
	scratch_load_dword v86, off, off offset:76 ; 4-byte Folded Reload
	s_mov_b32 m0, s3
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s37, v86
	buffer_load_dwordx4 v248, s[48:51], 0 offen lds
	s_mov_b32 m0, s37
	s_nop 0
	buffer_load_dwordx4 v250, s[48:51], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[2:9], v[22:25], v18, v182 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[90:97], v[10:17], v[26:29], v18, v182 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[98:105], v[2:9], v[30:33], v18, v182 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[98:105], v[10:17], v[34:37], v18, v182 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[106:113], v[2:9], v[38:41], v20, v182 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[106:113], v[10:17], v[42:45], v20, v182 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], v[184:191], v[2:9], v[46:49], v20, v182 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[184:191], v[10:17], v[50:53], v20, v182 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[192:193], v238 offset:0
ds_read_b64_tr_b8 v[194:195], v238 offset:1024

	;;#ASMEND
	scratch_store_dword off, v238, off offset:68 ; 4-byte Folded Spill
	;;#ASMSTART
	ds_read_b64_tr_b8 v[200:201], v237 offset:0
ds_read_b64_tr_b8 v[202:203], v237 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[196:197], v239 offset:0
ds_read_b64_tr_b8 v[198:199], v239 offset:1024

	;;#ASMEND
	scratch_store_dword off, v239, off offset:8 ; 4-byte Folded Spill
	;;#ASMSTART
	ds_read_b64_tr_b8 v[204:205], v1 offset:0
ds_read_b64_tr_b8 v[206:207], v1 offset:1024

	;;#ASMEND
	scratch_load_dword v86, off, off offset:116 ; 4-byte Folded Reload
	s_add_i32 s48, s40, s6
	s_ashr_i32 s38, s48, 31
	s_add_u32 s52, s28, s48
	s_addc_u32 s53, s29, s38
	s_mov_b32 s54, s6
	s_mul_i32 s41, s26, 0x300
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s38, v86
	scratch_load_dword v86, off, off offset:112 ; 4-byte Folded Reload
	s_mov_b32 m0, s38
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s40, v86
	buffer_load_dwordx4 v254, s[52:55], 0 offen lds
	s_mov_b32 m0, s40
	s_nop 0
	buffer_load_dwordx4 v252, s[52:55], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[90:97], v[192:199], v[118:121], v18, v183 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[90:97], v[200:207], v[122:125], v18, v183 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[98:105], v[192:199], v[126:129], v18, v183 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[98:101], v[98:105], v[200:207], v[130:133], v18, v183 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[102:105], v[106:113], v[192:199], v[134:137], v20, v183 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[106:109], v[106:113], v[200:207], v[138:141], v20, v183 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[110:113], v[184:191], v[192:199], v[142:145], v20, v183 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[114:117], v[184:191], v[200:207], v[146:149], v20, v183 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[184:187], v253 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[208:211], v253 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[216:219], v253 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[224:227], v253 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[188:191], v249 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[212:215], v249 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[220:223], v249 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[228:231], v249 offset:0x1800

	;;#ASMEND
	scratch_load_dword v118, off, off offset:132 ; 4-byte Folded Reload
	s_ashr_i32 s42, s41, 31
	s_add_u32 s41, s4, s41
	s_addc_u32 s42, s5, s42
	s_add_u32 s52, s41, 0x80
	s_addc_u32 s53, s42, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s41, v118
	scratch_load_dword v118, off, off offset:128 ; 4-byte Folded Reload
	s_mov_b32 m0, s41
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s42, v118
	buffer_load_dwordx4 v254, s[52:55], 0 offen lds
	s_mov_b32 m0, s42
	s_nop 0
	buffer_load_dwordx4 v252, s[52:55], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[118:121], v[184:191], v[2:9], v[150:153], v19, v182 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[122:125], v[184:191], v[10:17], v[154:157], v19, v182 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[126:129], v[208:215], v[2:9], v[158:161], v19, v182 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[130:133], v[208:215], v[10:17], v[162:165], v19, v182 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[134:137], v[216:223], v[2:9], v[166:169], v21, v182 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[138:141], v[216:223], v[10:17], v[170:173], v21, v182 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[142:145], v[224:231], v[2:9], v[174:177], v21, v182 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[146:149], v[224:231], v[10:17], v[178:181], v21, v182 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	scratch_load_dword v2, off, off offset:92 ; 4-byte Folded Reload
	s_add_u32 s52, s8, 0x300
	s_addc_u32 s53, s9, 0
	s_mov_b32 s54, s10
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s43, v2
	scratch_load_dword v2, off, off offset:100 ; 4-byte Folded Reload
	s_mov_b32 m0, s43
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s44, v2
	buffer_load_dwordx4 v248, s[52:55], 0 offen lds
	s_mov_b32 m0, s44
	s_nop 0
	buffer_load_dwordx4 v250, s[52:55], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[184:191], v[192:199], v[54:57], v19, v183 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[184:191], v[200:207], v[58:61], v19, v183 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[208:215], v[192:199], v[62:65], v19, v183 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[208:215], v[200:207], v[66:69], v19, v183 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[216:223], v[192:199], v[70:73], v21, v183 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[216:223], v[200:207], v[74:77], v21, v183 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[224:231], v[192:199], v[78:81], v21, v183 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[224:231], v[200:207], v[82:85], v21, v183 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	scratch_load_dword v6, off, off offset:44 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v6 offset:0
ds_read_b64_tr_b8 v[4:5], v6 offset:1024

	;;#ASMEND
	scratch_load_dword v6, off, off offset:52 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v6 offset:0
ds_read_b64_tr_b8 v[12:13], v6 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v242 offset:0
ds_read_b64_tr_b8 v[8:9], v242 offset:1024

	;;#ASMEND
	scratch_load_dword v150, off, off offset:48 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v150 offset:0
ds_read_b64_tr_b8 v[16:17], v150 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[150:153], v240 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[158:161], v240 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[166:169], v240 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[174:177], v240 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[154:157], v241 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[162:165], v241 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[170:173], v241 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[178:181], v241 offset:0x1800

	;;#ASMEND
	scratch_load_dword v184, off, off offset:124 ; 4-byte Folded Reload
	s_add_u32 s52, s11, 0x300
	s_addc_u32 s53, s30, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s45, v184
	scratch_load_dword v184, off, off offset:120 ; 4-byte Folded Reload
	s_mov_b32 m0, s45
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s46, v184
	buffer_load_dwordx4 v248, s[52:55], 0 offen lds
	s_mov_b32 m0, s46
	s_nop 0
	buffer_load_dwordx4 v250, s[52:55], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[150:157], v[2:9], v[22:25], v18, v182 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[150:157], v[10:17], v[26:29], v18, v182 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[158:165], v[2:9], v[30:33], v18, v182 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[158:165], v[10:17], v[34:37], v18, v182 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[166:173], v[2:9], v[38:41], v20, v182 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[166:173], v[10:17], v[42:45], v20, v182 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], v[174:181], v[2:9], v[46:49], v20, v182 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[174:181], v[10:17], v[50:53], v20, v182 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_mov_b32 m0, s47
	;;#ASMSTART
	ds_read_b64_tr_b8 v[184:185], v233 offset:0
ds_read_b64_tr_b8 v[186:187], v233 offset:1024

	;;#ASMEND
	s_add_i32 s47, s48, s6
	;;#ASMSTART
	ds_read_b64_tr_b8 v[192:193], v232 offset:0
ds_read_b64_tr_b8 v[194:195], v232 offset:1024

	;;#ASMEND
	s_ashr_i32 s49, s47, 31
	;;#ASMSTART
	ds_read_b64_tr_b8 v[188:189], v0 offset:0
ds_read_b64_tr_b8 v[190:191], v0 offset:1024

	;;#ASMEND
	s_add_u32 s48, s28, s47
	;;#ASMSTART
	ds_read_b64_tr_b8 v[196:197], v234 offset:0
ds_read_b64_tr_b8 v[198:199], v234 offset:1024

	;;#ASMEND
	s_addc_u32 s49, s29, s49
	s_mov_b32 s50, s6
	buffer_load_dwordx4 v254, s[48:51], 0 offen lds
	s_mov_b32 m0, s31
	v_mov_b32_e32 v0, v234
	buffer_load_dwordx4 v252, s[48:51], 0 offen lds
	scratch_store_dword off, v0, off offset:144 ; 4-byte Folded Spill
	s_mul_i32 s31, s26, 0x380
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[150:157], v[184:191], v[86:89], v18, v183 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[150:157], v[192:199], v[90:93], v18, v183 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[158:165], v[184:191], v[94:97], v18, v183 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[98:101], v[158:165], v[192:199], v[98:101], v18, v183 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[102:105], v[166:173], v[184:191], v[102:105], v20, v183 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[106:109], v[166:173], v[192:199], v[106:109], v20, v183 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[110:113], v[174:181], v[184:191], v[110:113], v20, v183 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[114:117], v[174:181], v[192:199], v[114:117], v20, v183 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[150:153], v245 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[158:161], v245 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[166:169], v245 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[174:177], v245 offset:0x1800

	;;#ASMEND
	s_ashr_i32 s48, s31, 31
	;;#ASMSTART
	ds_read_b128 v[154:157], v247 offset:0

	;;#ASMEND
	s_add_u32 s31, s4, s31
	;;#ASMSTART
	ds_read_b128 v[162:165], v247 offset:0x800

	;;#ASMEND
	s_addc_u32 s49, s5, s48
	;;#ASMSTART
	ds_read_b128 v[170:173], v247 offset:0x1000

	;;#ASMEND
	s_add_u32 s48, s31, 0x80
	;;#ASMSTART
	ds_read_b128 v[178:181], v247 offset:0x1800

	;;#ASMEND
	s_addc_u32 s49, s49, 0
	s_mov_b32 m0, s36
	s_nop 0
	buffer_load_dwordx4 v254, s[48:51], 0 offen lds
	s_mov_b32 m0, s39
	s_nop 0
	buffer_load_dwordx4 v252, s[48:51], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[118:121], v[150:157], v[2:9], v[118:121], v19, v182 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[122:125], v[150:157], v[10:17], v[122:125], v19, v182 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[126:129], v[158:165], v[2:9], v[126:129], v19, v182 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[130:133], v[158:165], v[10:17], v[130:133], v19, v182 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[134:137], v[166:173], v[2:9], v[134:137], v21, v182 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[138:141], v[166:173], v[10:17], v[138:141], v21, v182 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[142:145], v[174:181], v[2:9], v[142:145], v21, v182 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[146:149], v[174:181], v[10:17], v[146:149], v21, v182 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	scratch_load_dword v2, off, off offset:96 ; 4-byte Folded Reload
	s_add_u32 s48, s8, 0x380
	s_addc_u32 s49, s9, 0
	s_mov_b32 s50, s10
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s31, v2
	scratch_load_dword v2, off, off offset:136 ; 4-byte Folded Reload
	s_mov_b32 m0, s31
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s39, v2
	buffer_load_dwordx4 v248, s[48:51], 0 offen lds
	s_mov_b32 m0, s39
	s_nop 0
	buffer_load_dwordx4 v250, s[48:51], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[150:157], v[184:191], v[54:57], v19, v183 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[150:157], v[192:199], v[58:61], v19, v183 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[158:165], v[184:191], v[62:65], v19, v183 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[158:165], v[192:199], v[66:69], v19, v183 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[166:173], v[184:191], v[70:73], v21, v183 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[166:173], v[192:199], v[74:77], v21, v183 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[174:181], v[184:191], v[78:81], v21, v183 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[174:181], v[192:199], v[82:85], v21, v183 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	scratch_load_dword v2, off, off offset:64 ; 4-byte Folded Reload
	s_movk_i32 s36, 0xc00
	s_movk_i32 s48, 0x600
	buffer_load_dwordx4 v[18:21], v244, s[16:19], s36 offen
	s_mov_b32 m0, s3
	s_waitcnt vmcnt(1)
	buffer_load_dwordx2 v[214:215], v2, s[20:23], s48 offen
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v236 offset:0
ds_read_b64_tr_b8 v[4:5], v236 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v243 offset:0
ds_read_b64_tr_b8 v[12:13], v243 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v246 offset:0
ds_read_b64_tr_b8 v[8:9], v246 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v235 offset:0
ds_read_b64_tr_b8 v[16:17], v235 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[150:153], v255 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[158:161], v255 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[166:169], v255 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[174:177], v255 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[154:157], v251 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[162:165], v251 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[170:173], v251 offset:0x1000

	;;#ASMEND
	s_add_u32 s48, s11, 0x380
	;;#ASMSTART
	ds_read_b128 v[178:181], v251 offset:0x1800

	;;#ASMEND
	s_addc_u32 s49, s30, 0
	buffer_load_dwordx4 v248, s[48:51], 0 offen lds
	s_mov_b32 m0, s37
	s_nop 0
	buffer_load_dwordx4 v250, s[48:51], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	s_waitcnt vmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[182:185], v[150:157], v[2:9], v[22:25], v18, v214 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[186:189], v[150:157], v[10:17], v[26:29], v18, v214 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[190:193], v[158:165], v[2:9], v[30:33], v18, v214 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[194:197], v[158:165], v[10:17], v[34:37], v18, v214 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[198:201], v[166:173], v[2:9], v[38:41], v20, v214 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[202:205], v[166:173], v[10:17], v[42:45], v20, v214 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[206:209], v[174:181], v[2:9], v[46:49], v20, v214 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[210:213], v[174:181], v[10:17], v[50:53], v20, v214 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[216:217], v238 offset:0
ds_read_b64_tr_b8 v[218:219], v238 offset:1024

	;;#ASMEND
	s_add_i32 s3, s47, s6
	;;#ASMSTART
	ds_read_b64_tr_b8 v[224:225], v237 offset:0
ds_read_b64_tr_b8 v[226:227], v237 offset:1024

	;;#ASMEND
	s_lshl_b32 s37, s26, 10
	s_ashr_i32 s47, s3, 31
	;;#ASMSTART
	ds_read_b64_tr_b8 v[220:221], v239 offset:0
ds_read_b64_tr_b8 v[222:223], v239 offset:1024

	;;#ASMEND
	s_add_u32 s48, s28, s3
	;;#ASMSTART
	ds_read_b64_tr_b8 v[228:229], v1 offset:0
ds_read_b64_tr_b8 v[230:231], v1 offset:1024

	;;#ASMEND
	s_addc_u32 s49, s29, s47
	s_mov_b32 s50, s6
	s_mov_b32 m0, s38
	v_mov_b32_e32 v242, v237
	buffer_load_dwordx4 v254, s[48:51], 0 offen lds
	s_mov_b32 m0, s40
	v_mov_b32_e32 v243, v1
	buffer_load_dwordx4 v252, s[48:51], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[150:157], v[216:223], v[86:89], v18, v215 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[150:157], v[224:231], v[90:93], v18, v215 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[158:165], v[216:223], v[94:97], v18, v215 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[98:101], v[158:165], v[224:231], v[98:101], v18, v215 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[102:105], v[166:173], v[216:223], v[102:105], v20, v215 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[106:109], v[166:173], v[224:231], v[106:109], v20, v215 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[110:113], v[174:181], v[216:223], v[110:113], v20, v215 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[114:117], v[174:181], v[224:231], v[114:117], v20, v215 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[154:157], v253 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[162:165], v253 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[170:173], v253 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[232:235], v253 offset:0x1800

	;;#ASMEND
	s_ashr_i32 s38, s37, 31
	;;#ASMSTART
	ds_read_b128 v[158:161], v249 offset:0

	;;#ASMEND
	s_add_u32 s37, s4, s37
	;;#ASMSTART
	ds_read_b128 v[166:169], v249 offset:0x800

	;;#ASMEND
	s_addc_u32 s38, s5, s38
	;;#ASMSTART
	ds_read_b128 v[174:177], v249 offset:0x1000

	;;#ASMEND
	s_add_u32 s48, s37, 0x80
	;;#ASMSTART
	ds_read_b128 v[236:239], v249 offset:0x1800

	;;#ASMEND
	s_addc_u32 s49, s38, 0
	s_mov_b32 m0, s41
	s_nop 0
	buffer_load_dwordx4 v254, s[48:51], 0 offen lds
	s_mov_b32 m0, s42
	s_nop 0
	buffer_load_dwordx4 v252, s[48:51], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[154:161], v[2:9], v[118:121], v19, v214 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[154:161], v[10:17], v[122:125], v19, v214 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[162:169], v[2:9], v[126:129], v19, v214 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[162:169], v[10:17], v[130:133], v19, v214 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[170:177], v[2:9], v[134:137], v21, v214 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[170:177], v[10:17], v[138:141], v21, v214 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], v[232:239], v[2:9], v[142:145], v21, v214 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[232:239], v[10:17], v[146:149], v21, v214 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_u32 s48, s8, 0x400
	s_addc_u32 s49, s9, 0
	s_mov_b32 s50, s10
	s_mov_b32 m0, s43
	s_nop 0
	buffer_load_dwordx4 v248, s[48:51], 0 offen lds
	s_mov_b32 m0, s44
	s_nop 0
	buffer_load_dwordx4 v250, s[48:51], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[178:181], v[232:239], v[224:231], v[82:85], v21, v215 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[150:153], v[154:161], v[216:223], v[54:57], v19, v215 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[154:157], v[154:161], v[224:231], v[58:61], v19, v215 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[158:161], v[162:169], v[216:223], v[62:65], v19, v215 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[162:165], v[162:169], v[224:231], v[66:69], v19, v215 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[166:169], v[170:177], v[216:223], v[70:73], v21, v215 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[170:173], v[170:177], v[224:231], v[74:77], v21, v215 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[174:177], v[232:239], v[216:223], v[78:81], v21, v215 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	scratch_load_dword v233, off, off offset:44 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v233 offset:0
ds_read_b64_tr_b8 v[4:5], v233 offset:1024

	;;#ASMEND
	scratch_load_dword v232, off, off offset:52 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v232 offset:0
ds_read_b64_tr_b8 v[12:13], v232 offset:1024

	;;#ASMEND
	scratch_load_dword v235, off, off       ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v235 offset:0
ds_read_b64_tr_b8 v[8:9], v235 offset:1024

	;;#ASMEND
	scratch_load_dword v234, off, off offset:48 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v234 offset:0
ds_read_b64_tr_b8 v[16:17], v234 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[118:121], v240 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[126:129], v240 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[134:137], v240 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[142:145], v240 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[122:125], v241 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[130:133], v241 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[138:141], v241 offset:0x1000

	;;#ASMEND
	s_add_u32 s40, s11, 0x400
	;;#ASMSTART
	ds_read_b128 v[146:149], v241 offset:0x1800

	;;#ASMEND
	s_addc_u32 s41, s30, 0
	s_mov_b32 s42, s10
	s_mov_b32 s43, s7
	s_mov_b32 m0, s45
	s_nop 0
	buffer_load_dwordx4 v248, s[40:43], 0 offen lds
	s_mov_b32 m0, s46
	s_nop 0
	buffer_load_dwordx4 v250, s[40:43], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[118:125], v[2:9], v[182:185], v18, v214 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[118:125], v[10:17], v[186:189], v18, v214 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[126:133], v[2:9], v[190:193], v18, v214 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[126:133], v[10:17], v[194:197], v18, v214 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[134:141], v[2:9], v[198:201], v20, v214 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[134:141], v[10:17], v[202:205], v20, v214 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[142:149], v[2:9], v[206:209], v20, v214 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[142:149], v[10:17], v[210:213], v20, v214 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	scratch_load_dword v236, off, off offset:28 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[182:183], v236 offset:0
ds_read_b64_tr_b8 v[184:185], v236 offset:1024

	;;#ASMEND
	scratch_load_dword v246, off, off offset:140 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[190:191], v246 offset:0
ds_read_b64_tr_b8 v[192:193], v246 offset:1024

	;;#ASMEND
	scratch_load_dword v1, off, off offset:32 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[186:187], v1 offset:0
ds_read_b64_tr_b8 v[188:189], v1 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[194:195], v0 offset:0
ds_read_b64_tr_b8 v[196:197], v0 offset:1024

	;;#ASMEND
	scratch_load_dword v0, off, off offset:104 ; 4-byte Folded Reload
	s_add_i32 s41, s3, s6
	s_ashr_i32 s3, s41, 31
	s_add_u32 s48, s28, s41
	s_addc_u32 s49, s29, s3
	s_mov_b32 s50, s6
	s_mul_i32 s37, s26, 0x480
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s44, v0
	scratch_load_dword v0, off, off offset:108 ; 4-byte Folded Reload
	s_mov_b32 m0, s44
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s3, v0
	buffer_load_dwordx4 v254, s[48:51], 0 offen lds
	s_mov_b32 m0, s3
	s_nop 0
	buffer_load_dwordx4 v252, s[48:51], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[118:125], v[182:189], v[86:89], v18, v215 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[118:125], v[190:197], v[90:93], v18, v215 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[126:133], v[182:189], v[94:97], v18, v215 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[98:101], v[126:133], v[190:197], v[98:101], v18, v215 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[102:105], v[134:141], v[182:189], v[102:105], v20, v215 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[106:109], v[134:141], v[190:197], v[106:109], v20, v215 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[110:113], v[142:149], v[182:189], v[110:113], v20, v215 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[114:117], v[142:149], v[190:197], v[114:117], v20, v215 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	scratch_store_dword off, v245, off offset:16 ; 4-byte Folded Spill
	;;#ASMSTART
	ds_read_b128 v[198:201], v245 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[206:209], v245 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[216:219], v245 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[224:227], v245 offset:0x1800

	;;#ASMEND
	scratch_store_dword off, v247, off offset:36 ; 4-byte Folded Spill
	;;#ASMSTART
	ds_read_b128 v[202:205], v247 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[210:213], v247 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[220:223], v247 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[228:231], v247 offset:0x1800

	;;#ASMEND
	scratch_load_dword v0, off, off offset:84 ; 4-byte Folded Reload
	s_ashr_i32 s38, s37, 31
	s_add_u32 s37, s4, s37
	s_addc_u32 s38, s5, s38
	s_add_u32 s48, s37, 0x80
	s_addc_u32 s49, s38, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s38, v0
	scratch_load_dword v0, off, off offset:88 ; 4-byte Folded Reload
	s_mov_b32 m0, s38
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s40, v0
	buffer_load_dwordx4 v254, s[48:51], 0 offen lds
	s_mov_b32 m0, s40
	s_nop 0
	buffer_load_dwordx4 v252, s[48:51], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[118:121], v[198:205], v[2:9], v[22:25], v19, v214 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[122:125], v[198:205], v[10:17], v[26:29], v19, v214 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[126:129], v[206:213], v[2:9], v[30:33], v19, v214 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[130:133], v[206:213], v[10:17], v[34:37], v19, v214 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[134:137], v[216:223], v[2:9], v[38:41], v21, v214 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[138:141], v[216:223], v[10:17], v[42:45], v21, v214 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[142:145], v[224:231], v[2:9], v[46:49], v21, v214 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[146:149], v[224:231], v[10:17], v[50:53], v21, v214 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_u32 s48, s8, 0x480
	s_addc_u32 s49, s9, 0
	s_mov_b32 s50, s10
	s_mov_b32 m0, s31
	s_nop 0
	buffer_load_dwordx4 v248, s[48:51], 0 offen lds
	s_mov_b32 m0, s39
	s_nop 0
	buffer_load_dwordx4 v250, s[48:51], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[198:205], v[182:189], v[150:153], v19, v215 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[198:205], v[190:197], v[154:157], v19, v215 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[206:213], v[182:189], v[158:161], v19, v215 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[206:213], v[190:197], v[162:165], v19, v215 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[216:223], v[182:189], v[166:169], v21, v215 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[216:223], v[190:197], v[170:173], v21, v215 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], v[224:231], v[182:189], v[174:177], v21, v215 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[224:231], v[190:197], v[178:181], v21, v215 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	scratch_load_dword v247, off, off offset:148 ; 4-byte Folded Reload
	scratch_load_dword v0, off, off offset:64 ; 4-byte Folded Reload
	scratch_load_dword v1, off, off offset:60 ; 4-byte Folded Reload
	s_movk_i32 s31, 0x1000
	s_add_u32 s48, s11, 0x480
	s_addc_u32 s49, s30, 0
	s_waitcnt vmcnt(2)
	buffer_load_dwordx4 v[18:21], v247, s[16:19], s31 offen
	s_waitcnt vmcnt(2)
	buffer_load_dwordx2 v[182:183], v0, s[20:23], s2 offen
	s_waitcnt vmcnt(2)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v1 offset:0
ds_read_b64_tr_b8 v[4:5], v1 offset:1024

	;;#ASMEND
	scratch_load_dword v237, off, off offset:56 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v237 offset:0
ds_read_b64_tr_b8 v[12:13], v237 offset:1024

	;;#ASMEND
	scratch_load_dword v0, off, off offset:24 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v0 offset:0
ds_read_b64_tr_b8 v[8:9], v0 offset:1024

	;;#ASMEND
	scratch_load_dword v0, off, off offset:12 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v0 offset:0
ds_read_b64_tr_b8 v[16:17], v0 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[150:153], v255 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[158:161], v255 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[166:169], v255 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[174:177], v255 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[154:157], v251 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[162:165], v251 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[170:173], v251 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[178:181], v251 offset:0x1800

	;;#ASMEND
	scratch_load_dword v0, off, off offset:80 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s2, v0
	scratch_load_dword v0, off, off offset:76 ; 4-byte Folded Reload
	s_mov_b32 m0, s2
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s37, v0
	buffer_load_dwordx4 v248, s[48:51], 0 offen lds
	s_mov_b32 m0, s37
	s_nop 0
	buffer_load_dwordx4 v250, s[48:51], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[150:157], v[2:9], v[54:57], v18, v182 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[150:157], v[10:17], v[58:61], v18, v182 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[158:165], v[2:9], v[62:65], v18, v182 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[158:165], v[10:17], v[66:69], v18, v182 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[166:173], v[2:9], v[70:73], v20, v182 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[166:173], v[10:17], v[74:77], v20, v182 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[174:181], v[2:9], v[78:81], v20, v182 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[174:181], v[10:17], v[82:85], v20, v182 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	scratch_load_dword v0, off, off offset:68 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[184:185], v0 offset:0
ds_read_b64_tr_b8 v[186:187], v0 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[192:193], v242 offset:0
ds_read_b64_tr_b8 v[194:195], v242 offset:1024

	;;#ASMEND
	scratch_load_dword v196, off, off offset:8 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[188:189], v196 offset:0
ds_read_b64_tr_b8 v[190:191], v196 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[196:197], v243 offset:0
ds_read_b64_tr_b8 v[198:199], v243 offset:1024

	;;#ASMEND
	scratch_load_dword v200, off, off offset:116 ; 4-byte Folded Reload
	s_add_i32 s46, s41, s6
	s_ashr_i32 s39, s46, 31
	s_add_u32 s52, s28, s46
	s_addc_u32 s53, s29, s39
	s_mov_b32 s54, s6
	s_mul_i32 s41, s26, 0x500
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s49, v200
	scratch_load_dword v200, off, off offset:112 ; 4-byte Folded Reload
	s_mov_b32 m0, s49
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s39, v200
	buffer_load_dwordx4 v254, s[52:55], 0 offen lds
	s_mov_b32 m0, s39
	s_nop 0
	buffer_load_dwordx4 v252, s[52:55], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[150:157], v[184:191], v[86:89], v18, v183 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[150:157], v[192:199], v[90:93], v18, v183 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[158:165], v[184:191], v[94:97], v18, v183 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[98:101], v[158:165], v[192:199], v[98:101], v18, v183 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[102:105], v[166:173], v[184:191], v[102:105], v20, v183 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[106:109], v[166:173], v[192:199], v[106:109], v20, v183 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[110:113], v[174:181], v[184:191], v[110:113], v20, v183 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[114:117], v[174:181], v[192:199], v[114:117], v20, v183 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[154:157], v253 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[162:165], v253 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[170:173], v253 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[200:203], v253 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[158:161], v249 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[166:169], v249 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[174:177], v249 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[204:207], v249 offset:0x1800

	;;#ASMEND
	scratch_load_dword v150, off, off offset:132 ; 4-byte Folded Reload
	s_ashr_i32 s42, s41, 31
	s_add_u32 s41, s4, s41
	s_addc_u32 s42, s5, s42
	s_add_u32 s52, s41, 0x80
	s_addc_u32 s53, s42, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s41, v150
	scratch_load_dword v150, off, off offset:128 ; 4-byte Folded Reload
	s_mov_b32 m0, s41
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s42, v150
	buffer_load_dwordx4 v254, s[52:55], 0 offen lds
	s_mov_b32 m0, s42
	s_nop 0
	buffer_load_dwordx4 v252, s[52:55], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[118:121], v[154:161], v[2:9], v[118:121], v19, v182 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[122:125], v[154:161], v[10:17], v[122:125], v19, v182 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[126:129], v[162:169], v[2:9], v[126:129], v19, v182 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[130:133], v[162:169], v[10:17], v[130:133], v19, v182 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[134:137], v[170:177], v[2:9], v[134:137], v21, v182 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[138:141], v[170:177], v[10:17], v[138:141], v21, v182 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[142:145], v[200:207], v[2:9], v[142:145], v21, v182 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[146:149], v[200:207], v[10:17], v[146:149], v21, v182 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	scratch_load_dword v2, off, off offset:92 ; 4-byte Folded Reload
	s_add_u32 s52, s8, 0x500
	s_addc_u32 s53, s9, 0
	s_mov_b32 s54, s10
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s43, v2
	scratch_load_dword v2, off, off offset:100 ; 4-byte Folded Reload
	s_mov_b32 m0, s43
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s45, v2
	buffer_load_dwordx4 v248, s[52:55], 0 offen lds
	s_mov_b32 m0, s45
	s_nop 0
	buffer_load_dwordx4 v250, s[52:55], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[178:181], v[200:207], v[192:199], v[50:53], v21, v183 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[150:153], v[154:161], v[184:191], v[22:25], v19, v183 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[154:157], v[154:161], v[192:199], v[26:29], v19, v183 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[158:161], v[162:169], v[184:191], v[30:33], v19, v183 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[162:165], v[162:169], v[192:199], v[34:37], v19, v183 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[166:169], v[170:177], v[184:191], v[38:41], v21, v183 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[170:173], v[170:177], v[192:199], v[42:45], v21, v183 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[174:177], v[200:207], v[184:191], v[46:49], v21, v183 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v233 offset:0
ds_read_b64_tr_b8 v[4:5], v233 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v232 offset:0
ds_read_b64_tr_b8 v[12:13], v232 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v235 offset:0
ds_read_b64_tr_b8 v[8:9], v235 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v234 offset:0
ds_read_b64_tr_b8 v[16:17], v234 offset:1024

	;;#ASMEND
	scratch_store_dword off, v240, off offset:152 ; 4-byte Folded Spill
	;;#ASMSTART
	ds_read_b128 v[184:187], v240 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[192:195], v240 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[200:203], v240 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[208:211], v240 offset:0x1800

	;;#ASMEND
	scratch_store_dword off, v241, off offset:40 ; 4-byte Folded Spill
	;;#ASMSTART
	ds_read_b128 v[188:191], v241 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[196:199], v241 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[204:207], v241 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[212:215], v241 offset:0x1800

	;;#ASMEND
	scratch_load_dword v22, off, off offset:124 ; 4-byte Folded Reload
	s_add_u32 s52, s11, 0x500
	s_addc_u32 s53, s30, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s47, v22
	scratch_load_dword v22, off, off offset:120 ; 4-byte Folded Reload
	s_mov_b32 m0, s47
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s48, v22
	buffer_load_dwordx4 v248, s[52:55], 0 offen lds
	s_mov_b32 m0, s48
	s_nop 0
	buffer_load_dwordx4 v250, s[52:55], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[184:191], v[10:17], v[58:61], v18, v182 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[192:199], v[2:9], v[62:65], v18, v182 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	s_nop 7
	s_nop 2
	scratch_store_dwordx4 off, v[22:25], off offset:188 ; 16-byte Folded Spill
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[192:199], v[10:17], v[66:69], v18, v182 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[200:207], v[2:9], v[70:73], v20, v182 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[200:207], v[10:17], v[74:77], v20, v182 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[232:235], v[184:191], v[2:9], v[54:57], v18, v182 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[238:241], v[208:215], v[2:9], v[78:81], v20, v182 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[242:245], v[208:215], v[10:17], v[82:85], v20, v182 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[216:217], v236 offset:0
ds_read_b64_tr_b8 v[218:219], v236 offset:1024

	;;#ASMEND
	s_add_i32 s50, s46, s6
	s_mov_b32 m0, s44
	;;#ASMSTART
	ds_read_b64_tr_b8 v[224:225], v246 offset:0
ds_read_b64_tr_b8 v[226:227], v246 offset:1024

	;;#ASMEND
	s_ashr_i32 s44, s50, 31
	scratch_load_dword v22, off, off offset:32 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[220:221], v22 offset:0
ds_read_b64_tr_b8 v[222:223], v22 offset:1024

	;;#ASMEND
	s_add_u32 s52, s28, s50
	scratch_load_dword v22, off, off offset:144 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[228:229], v22 offset:0
ds_read_b64_tr_b8 v[230:231], v22 offset:1024

	;;#ASMEND
	s_addc_u32 s53, s29, s44
	s_mov_b32 s54, s6
	buffer_load_dwordx4 v254, s[52:55], 0 offen lds
	s_mov_b32 m0, s3
	s_mul_i32 s3, s26, 0x580
	buffer_load_dwordx4 v252, s[52:55], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[184:191], v[216:223], v[86:89], v18, v183 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[184:191], v[224:231], v[90:93], v18, v183 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[192:199], v[216:223], v[94:97], v18, v183 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[192:199], v[224:231], v[98:101], v18, v183 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[200:207], v[216:223], v[102:105], v20, v183 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[200:207], v[224:231], v[106:109], v20, v183 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[208:215], v[216:223], v[110:113], v20, v183 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[208:215], v[224:231], v[114:117], v20, v183 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	scratch_load_dword v18, off, off offset:16 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b128 v[184:187], v18 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[192:195], v18 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[200:203], v18 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[208:211], v18 offset:0x1800

	;;#ASMEND
	s_ashr_i32 s44, s3, 31
	scratch_load_dword v18, off, off offset:36 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b128 v[188:191], v18 offset:0

	;;#ASMEND
	s_add_u32 s3, s4, s3
	;;#ASMSTART
	ds_read_b128 v[196:199], v18 offset:0x800

	;;#ASMEND
	s_addc_u32 s44, s5, s44
	;;#ASMSTART
	ds_read_b128 v[204:207], v18 offset:0x1000

	;;#ASMEND
	s_add_u32 s52, s3, 0x80
	;;#ASMSTART
	ds_read_b128 v[212:215], v18 offset:0x1800

	;;#ASMEND
	s_addc_u32 s53, s44, 0
	s_mov_b32 m0, s38
	s_nop 0
	buffer_load_dwordx4 v254, s[52:55], 0 offen lds
	s_mov_b32 m0, s40
	s_nop 0
	buffer_load_dwordx4 v252, s[52:55], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[184:191], v[2:9], v[118:121], v19, v182 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[184:191], v[10:17], v[122:125], v19, v182 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[192:199], v[2:9], v[126:129], v19, v182 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[98:101], v[192:199], v[10:17], v[130:133], v19, v182 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[102:105], v[200:207], v[2:9], v[134:137], v21, v182 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[106:109], v[200:207], v[10:17], v[138:141], v21, v182 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[110:113], v[208:215], v[2:9], v[142:145], v21, v182 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[114:117], v[208:215], v[10:17], v[146:149], v21, v182 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	scratch_load_dword v2, off, off offset:96 ; 4-byte Folded Reload
	s_add_u32 s52, s8, 0x580
	s_addc_u32 s53, s9, 0
	s_mov_b32 s54, s10
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s44, v2
	scratch_load_dword v2, off, off offset:136 ; 4-byte Folded Reload
	s_mov_b32 m0, s44
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s46, v2
	buffer_load_dwordx4 v248, s[52:55], 0 offen lds
	s_mov_b32 m0, s46
	s_nop 0
	buffer_load_dwordx4 v250, s[52:55], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[118:121], v[184:191], v[216:223], v[150:153], v19, v183 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[122:125], v[184:191], v[224:231], v[154:157], v19, v183 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[126:129], v[192:199], v[216:223], v[158:161], v19, v183 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[130:133], v[192:199], v[224:231], v[162:165], v19, v183 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[134:137], v[200:207], v[216:223], v[166:169], v21, v183 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[138:141], v[200:207], v[224:231], v[170:173], v21, v183 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[142:145], v[208:215], v[216:223], v[174:177], v21, v183 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[146:149], v[208:215], v[224:231], v[178:181], v21, v183 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	scratch_load_dword v2, off, off offset:64 ; 4-byte Folded Reload
	s_movk_i32 s3, 0x1400
	s_movk_i32 s38, 0xa00
	buffer_load_dwordx4 v[18:21], v247, s[16:19], s3 offen
	s_add_u32 s52, s11, 0x580
	s_addc_u32 s53, s30, 0
	s_mov_b32 m0, s2
	s_waitcnt vmcnt(1)
	buffer_load_dwordx2 v[246:247], v2, s[20:23], s38 offen
	;;#ASMSTART
	ds_read_b64_tr_b8 v[46:47], v1 offset:0
ds_read_b64_tr_b8 v[48:49], v1 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[22:23], v237 offset:0
ds_read_b64_tr_b8 v[24:25], v237 offset:1024

	;;#ASMEND
	scratch_load_dword v1, off, off offset:24 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[50:51], v1 offset:0
ds_read_b64_tr_b8 v[52:53], v1 offset:1024

	;;#ASMEND
	scratch_load_dword v1, off, off offset:12 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[26:27], v1 offset:0
ds_read_b64_tr_b8 v[28:29], v1 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[150:153], v255 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[158:161], v255 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[166:169], v255 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[174:177], v255 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[154:157], v251 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[162:165], v251 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[170:173], v251 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[178:181], v251 offset:0x1800

	;;#ASMEND
	buffer_load_dwordx4 v248, s[52:55], 0 offen lds
	s_mov_b32 m0, s37
	s_nop 0
	buffer_load_dwordx4 v250, s[52:55], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	scratch_load_dwordx4 v[2:5], off, off offset:188 ; 16-byte Folded Reload
	v_mfma_scale_f32_16x16x128_f8f6f4 v[214:217], v[150:157], v[46:53], v[232:235], v18, v246 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[222:225], v[158:165], v[46:53], v[30:33], v18, v246 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[226:229], v[158:165], v[22:29], v[34:37], v18, v246 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[230:233], v[166:173], v[46:53], v[38:41], v20, v246 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[234:237], v[166:173], v[22:29], v[42:45], v20, v246 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[238:241], v[174:181], v[46:53], v[238:241], v20, v246 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[242:245], v[174:181], v[22:29], v[242:245], v20, v246 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_waitcnt vmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[218:221], v[150:157], v[22:29], v[2:5], v18, v246 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[8:9], v0 offset:0
ds_read_b64_tr_b8 v[10:11], v0 offset:1024

	;;#ASMEND
	s_add_i32 s2, s50, s6
	s_nop 3
	scratch_load_dword v4, off, off offset:20 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[0:1], v4 offset:0
ds_read_b64_tr_b8 v[2:3], v4 offset:1024

	;;#ASMEND
	s_ashr_i32 s37, s2, 31
	scratch_load_dword v4, off, off offset:8 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[12:13], v4 offset:0
ds_read_b64_tr_b8 v[14:15], v4 offset:1024

	;;#ASMEND
	s_add_u32 s52, s28, s2
	s_mov_b32 m0, s49
	scratch_load_dword v16, off, off offset:4 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[4:5], v16 offset:0
ds_read_b64_tr_b8 v[6:7], v16 offset:1024

	;;#ASMEND
	s_addc_u32 s53, s29, s37
	s_mov_b32 s54, s6
	buffer_load_dwordx4 v254, s[52:55], 0 offen lds
	s_mov_b32 m0, s39
	s_mul_i32 s37, s26, 0x600
	buffer_load_dwordx4 v252, s[52:55], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[182:185], v[150:157], v[8:15], v[54:57], v18, v247 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[186:189], v[150:157], v[0:7], v[58:61], v18, v247 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[190:193], v[158:165], v[8:15], v[62:65], v18, v247 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[194:197], v[158:165], v[0:7], v[66:69], v18, v247 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[198:201], v[166:173], v[8:15], v[70:73], v20, v247 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[202:205], v[166:173], v[0:7], v[74:77], v20, v247 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[206:209], v[174:181], v[8:15], v[78:81], v20, v247 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[210:213], v[174:181], v[0:7], v[82:85], v20, v247 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[62:65], v253 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[70:73], v253 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[78:81], v253 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[150:153], v253 offset:0x1800

	;;#ASMEND
	s_ashr_i32 s38, s37, 31
	;;#ASMSTART
	ds_read_b128 v[66:69], v249 offset:0

	;;#ASMEND
	s_add_u32 s37, s4, s37
	;;#ASMSTART
	ds_read_b128 v[74:77], v249 offset:0x800

	;;#ASMEND
	s_addc_u32 s38, s5, s38
	;;#ASMSTART
	ds_read_b128 v[82:85], v249 offset:0x1000

	;;#ASMEND
	s_add_u32 s52, s37, 0x80
	;;#ASMSTART
	ds_read_b128 v[154:157], v249 offset:0x1800

	;;#ASMEND
	s_addc_u32 s53, s38, 0
	s_mov_b32 m0, s41
	s_nop 0
	buffer_load_dwordx4 v254, s[52:55], 0 offen lds
	s_mov_b32 m0, s42
	s_nop 0
	buffer_load_dwordx4 v252, s[52:55], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[62:69], v[46:53], v[86:89], v19, v246 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[62:69], v[22:29], v[90:93], v19, v246 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[70:77], v[46:53], v[94:97], v19, v246 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[70:77], v[22:29], v[98:101], v19, v246 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[78:85], v[46:53], v[102:105], v21, v246 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[78:85], v[22:29], v[106:109], v21, v246 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], v[150:157], v[46:53], v[110:113], v21, v246 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[150:157], v[22:29], v[114:117], v21, v246 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_u32 s52, s8, 0x600
	s_addc_u32 s53, s9, 0
	s_mov_b32 s54, s10
	s_mov_b32 m0, s43
	s_nop 0
	buffer_load_dwordx4 v248, s[52:55], 0 offen lds
	s_mov_b32 m0, s45
	s_nop 0
	buffer_load_dwordx4 v250, s[52:55], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[62:69], v[8:15], v[118:121], v19, v247 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[62:69], v[0:7], v[122:125], v19, v247 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[178:181], v[150:157], v[0:7], v[146:149], v21, v247 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[158:161], v[70:77], v[8:15], v[126:129], v19, v247 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[162:165], v[70:77], v[0:7], v[130:133], v19, v247 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[166:169], v[78:85], v[8:15], v[134:137], v21, v247 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[170:173], v[78:85], v[0:7], v[138:141], v21, v247 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[174:177], v[150:157], v[8:15], v[142:145], v21, v247 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	scratch_load_dword v0, off, off offset:44 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v0 offset:0
ds_read_b64_tr_b8 v[4:5], v0 offset:1024

	;;#ASMEND
	scratch_load_dword v1, off, off offset:52 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v1 offset:0
ds_read_b64_tr_b8 v[12:13], v1 offset:1024

	;;#ASMEND
	scratch_load_dword v0, off, off         ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v0 offset:0
ds_read_b64_tr_b8 v[8:9], v0 offset:1024

	;;#ASMEND
	scratch_load_dword v0, off, off offset:48 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v0 offset:0
ds_read_b64_tr_b8 v[16:17], v0 offset:1024

	;;#ASMEND
	scratch_load_dword v0, off, off offset:152 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b128 v[90:93], v0 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[98:101], v0 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[106:109], v0 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[114:117], v0 offset:0x1800

	;;#ASMEND
	scratch_load_dword v0, off, off offset:40 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b128 v[94:97], v0 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[102:105], v0 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[110:113], v0 offset:0x1000

	;;#ASMEND
	s_add_u32 s40, s11, 0x600
	;;#ASMSTART
	ds_read_b128 v[118:121], v0 offset:0x1800

	;;#ASMEND
	s_addc_u32 s41, s30, 0
	s_mov_b32 s42, s10
	s_mov_b32 s43, s7
	s_mov_b32 m0, s47
	s_nop 0
	buffer_load_dwordx4 v248, s[40:43], 0 offen lds
	s_mov_b32 m0, s48
	s_nop 0
	buffer_load_dwordx4 v250, s[40:43], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[98:105], v[2:9], v[222:225], v18, v246 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[98:105], v[10:17], v[226:229], v18, v246 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[106:113], v[2:9], v[230:233], v20, v246 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[106:113], v[10:17], v[234:237], v20, v246 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[114:121], v[2:9], v[238:241], v20, v246 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[114:121], v[10:17], v[242:245], v20, v246 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[214:217], v[90:97], v[2:9], v[214:217], v18, v246 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[218:221], v[90:97], v[10:17], v[218:221], v18, v246 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	scratch_load_dword v0, off, off offset:28 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[150:151], v0 offset:0
ds_read_b64_tr_b8 v[152:153], v0 offset:1024

	;;#ASMEND
	scratch_load_dword v231, off, off offset:140 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[222:223], v231 offset:0
ds_read_b64_tr_b8 v[224:225], v231 offset:1024

	;;#ASMEND
	scratch_load_dword v0, off, off offset:32 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[154:155], v0 offset:0
ds_read_b64_tr_b8 v[156:157], v0 offset:1024

	;;#ASMEND
	scratch_load_dword v230, off, off offset:144 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[226:227], v230 offset:0
ds_read_b64_tr_b8 v[228:229], v230 offset:1024

	;;#ASMEND
	scratch_load_dword v0, off, off offset:104 ; 4-byte Folded Reload
	s_add_i32 s41, s2, s6
	s_ashr_i32 s2, s41, 31
	s_add_u32 s48, s28, s41
	s_addc_u32 s49, s29, s2
	s_mov_b32 s50, s6
	s_mul_i32 s2, s26, 0x680
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s47, v0
	scratch_load_dword v0, off, off offset:108 ; 4-byte Folded Reload
	s_mov_b32 m0, s47
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s37, v0
	buffer_load_dwordx4 v254, s[48:51], 0 offen lds
	s_mov_b32 m0, s37
	s_nop 0
	buffer_load_dwordx4 v252, s[48:51], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[90:97], v[150:157], v[182:185], v18, v247 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[90:97], v[222:229], v[186:189], v18, v247 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[98:105], v[150:157], v[190:193], v18, v247 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[98:101], v[98:105], v[222:229], v[194:197], v18, v247 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[102:105], v[106:113], v[150:157], v[198:201], v20, v247 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[106:109], v[106:113], v[222:229], v[202:205], v20, v247 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[110:113], v[114:121], v[150:157], v[206:209], v20, v247 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[114:117], v[114:121], v[222:229], v[210:213], v20, v247 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	scratch_load_dword v0, off, off offset:16 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b128 v[182:185], v0 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[190:193], v0 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[198:201], v0 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[206:209], v0 offset:0x1800

	;;#ASMEND
	scratch_load_dword v0, off, off offset:36 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b128 v[186:189], v0 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[194:197], v0 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[202:205], v0 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[210:213], v0 offset:0x1800

	;;#ASMEND
	scratch_load_dword v0, off, off offset:84 ; 4-byte Folded Reload
	s_ashr_i32 s38, s2, 31
	s_add_u32 s2, s4, s2
	s_addc_u32 s38, s5, s38
	s_add_u32 s48, s2, 0x80
	s_addc_u32 s49, s38, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s38, v0
	scratch_load_dword v0, off, off offset:88 ; 4-byte Folded Reload
	s_mov_b32 m0, s38
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s40, v0
	buffer_load_dwordx4 v254, s[48:51], 0 offen lds
	s_mov_b32 m0, s40
	s_nop 0
	buffer_load_dwordx4 v252, s[48:51], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[118:121], v[182:189], v[2:9], v[54:57], v19, v246 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[122:125], v[182:189], v[10:17], v[58:61], v19, v246 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[126:129], v[190:197], v[2:9], v[30:33], v19, v246 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[130:133], v[190:197], v[10:17], v[34:37], v19, v246 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[134:137], v[198:205], v[2:9], v[38:41], v21, v246 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[138:141], v[198:205], v[10:17], v[42:45], v21, v246 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[142:145], v[206:213], v[2:9], v[46:49], v21, v246 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[146:149], v[206:213], v[10:17], v[50:53], v21, v246 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_u32 s48, s8, 0x680
	s_addc_u32 s49, s9, 0
	s_mov_b32 s50, s10
	s_mov_b32 m0, s44
	s_nop 0
	buffer_load_dwordx4 v248, s[48:51], 0 offen lds
	s_mov_b32 m0, s46
	s_nop 0
	buffer_load_dwordx4 v250, s[48:51], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[182:189], v[150:157], v[22:25], v19, v247 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[182:189], v[222:229], v[26:29], v19, v247 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[190:197], v[150:157], v[158:161], v19, v247 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[190:197], v[222:229], v[162:165], v19, v247 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[198:205], v[150:157], v[166:169], v21, v247 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[198:205], v[222:229], v[170:173], v21, v247 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], v[206:213], v[150:157], v[174:177], v21, v247 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[206:213], v[222:229], v[178:181], v21, v247 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	scratch_load_dword v246, off, off offset:148 ; 4-byte Folded Reload
	scratch_load_dword v247, off, off offset:64 ; 4-byte Folded Reload
	scratch_load_dword v0, off, off offset:60 ; 4-byte Folded Reload
	s_movk_i32 s2, 0x1800
	s_add_u32 s48, s11, 0x680
	s_addc_u32 s49, s30, 0
	s_waitcnt vmcnt(2)
	buffer_load_dwordx4 v[18:21], v246, s[16:19], s2 offen
	s_waitcnt vmcnt(2)
	buffer_load_dwordx2 v[182:183], v247, s[20:23], s36 offen
	s_waitcnt vmcnt(2)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v0 offset:0
ds_read_b64_tr_b8 v[4:5], v0 offset:1024

	;;#ASMEND
	scratch_load_dword v236, off, off offset:56 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v236 offset:0
ds_read_b64_tr_b8 v[12:13], v236 offset:1024

	;;#ASMEND
	scratch_load_dword v237, off, off offset:24 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v237 offset:0
ds_read_b64_tr_b8 v[8:9], v237 offset:1024

	;;#ASMEND
	scratch_load_dword v54, off, off offset:12 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v54 offset:0
ds_read_b64_tr_b8 v[16:17], v54 offset:1024

	;;#ASMEND
	scratch_store_dword off, v255, off offset:72 ; 4-byte Folded Spill
	;;#ASMSTART
	ds_read_b128 v[150:153], v255 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[158:161], v255 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[166:169], v255 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[174:177], v255 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[154:157], v251 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[162:165], v251 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[170:173], v251 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[178:181], v251 offset:0x1800

	;;#ASMEND
	scratch_load_dword v54, off, off offset:80 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s36, v54
	scratch_load_dword v54, off, off offset:76 ; 4-byte Folded Reload
	s_mov_b32 m0, s36
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s39, v54
	buffer_load_dwordx4 v248, s[48:51], 0 offen lds
	s_mov_b32 m0, s39
	s_nop 0
	buffer_load_dwordx4 v250, s[48:51], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[150:157], v[2:9], v[214:217], v18, v182 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[150:157], v[10:17], v[218:221], v18, v182 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[158:165], v[2:9], v[62:65], v18, v182 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[158:165], v[10:17], v[66:69], v18, v182 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[166:173], v[2:9], v[70:73], v20, v182 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[166:173], v[10:17], v[74:77], v20, v182 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[174:181], v[2:9], v[78:81], v20, v182 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[174:181], v[10:17], v[82:85], v20, v182 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	scratch_load_dword v188, off, off offset:68 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[184:185], v188 offset:0
ds_read_b64_tr_b8 v[186:187], v188 offset:1024

	;;#ASMEND
	scratch_load_dword v188, off, off offset:20 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[192:193], v188 offset:0
ds_read_b64_tr_b8 v[194:195], v188 offset:1024

	;;#ASMEND
	scratch_load_dword v196, off, off offset:8 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[188:189], v196 offset:0
ds_read_b64_tr_b8 v[190:191], v196 offset:1024

	;;#ASMEND
	scratch_load_dword v200, off, off offset:4 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[196:197], v200 offset:0
ds_read_b64_tr_b8 v[198:199], v200 offset:1024

	;;#ASMEND
	scratch_load_dword v200, off, off offset:116 ; 4-byte Folded Reload
	s_add_i32 s50, s41, s6
	s_ashr_i32 s41, s50, 31
	s_add_u32 s52, s28, s50
	s_addc_u32 s53, s29, s41
	s_mov_b32 s54, s6
	s_mul_i32 s43, s26, 0x700
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s41, v200
	scratch_load_dword v200, off, off offset:112 ; 4-byte Folded Reload
	s_mov_b32 m0, s41
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s42, v200
	buffer_load_dwordx4 v254, s[52:55], 0 offen lds
	s_mov_b32 m0, s42
	s_nop 0
	buffer_load_dwordx4 v252, s[52:55], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[150:157], v[184:191], v[86:89], v18, v183 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[150:157], v[192:199], v[90:93], v18, v183 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[158:165], v[184:191], v[94:97], v18, v183 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[98:101], v[158:165], v[192:199], v[98:101], v18, v183 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[102:105], v[166:173], v[184:191], v[102:105], v20, v183 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[106:109], v[166:173], v[192:199], v[106:109], v20, v183 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[110:113], v[174:181], v[184:191], v[110:113], v20, v183 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[114:117], v[174:181], v[192:199], v[114:117], v20, v183 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[154:157], v253 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[162:165], v253 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[170:173], v253 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[200:203], v253 offset:0x1800

	;;#ASMEND
	scratch_store_dword off, v249, off offset:156 ; 4-byte Folded Spill
	;;#ASMSTART
	ds_read_b128 v[158:161], v249 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[166:169], v249 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[174:177], v249 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[204:207], v249 offset:0x1800

	;;#ASMEND
	scratch_load_dword v150, off, off offset:132 ; 4-byte Folded Reload
	s_ashr_i32 s44, s43, 31
	s_add_u32 s43, s4, s43
	s_addc_u32 s44, s5, s44
	s_add_u32 s52, s43, 0x80
	s_addc_u32 s53, s44, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s43, v150
	scratch_load_dword v150, off, off offset:128 ; 4-byte Folded Reload
	s_mov_b32 m0, s43
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s44, v150
	buffer_load_dwordx4 v254, s[52:55], 0 offen lds
	s_mov_b32 m0, s44
	s_nop 0
	buffer_load_dwordx4 v252, s[52:55], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[118:121], v[154:161], v[2:9], v[118:121], v19, v182 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[122:125], v[154:161], v[10:17], v[122:125], v19, v182 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[126:129], v[162:169], v[2:9], v[126:129], v19, v182 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[130:133], v[162:169], v[10:17], v[130:133], v19, v182 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[134:137], v[170:177], v[2:9], v[134:137], v21, v182 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[138:141], v[170:177], v[10:17], v[138:141], v21, v182 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[142:145], v[200:207], v[2:9], v[142:145], v21, v182 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[146:149], v[200:207], v[10:17], v[146:149], v21, v182 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	scratch_load_dword v2, off, off offset:92 ; 4-byte Folded Reload
	s_add_u32 s52, s8, 0x700
	s_addc_u32 s53, s9, 0
	s_mov_b32 s54, s10
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s45, v2
	scratch_load_dword v2, off, off offset:100 ; 4-byte Folded Reload
	s_mov_b32 m0, s45
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s46, v2
	buffer_load_dwordx4 v248, s[52:55], 0 offen lds
	s_mov_b32 m0, s46
	s_nop 0
	buffer_load_dwordx4 v250, s[52:55], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[178:181], v[200:207], v[192:199], v[50:53], v21, v183 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[150:153], v[154:161], v[184:191], v[22:25], v19, v183 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[154:157], v[154:161], v[192:199], v[26:29], v19, v183 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[158:161], v[162:169], v[184:191], v[30:33], v19, v183 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[162:165], v[162:169], v[192:199], v[34:37], v19, v183 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[166:169], v[170:177], v[184:191], v[38:41], v21, v183 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[170:173], v[170:177], v[192:199], v[42:45], v21, v183 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[174:177], v[200:207], v[184:191], v[46:49], v21, v183 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	scratch_load_dword v6, off, off offset:44 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v6 offset:0
ds_read_b64_tr_b8 v[4:5], v6 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v1 offset:0
ds_read_b64_tr_b8 v[12:13], v1 offset:1024

	;;#ASMEND
	scratch_load_dword v1, off, off         ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v1 offset:0
ds_read_b64_tr_b8 v[8:9], v1 offset:1024

	;;#ASMEND
	scratch_load_dword v1, off, off offset:48 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v1 offset:0
ds_read_b64_tr_b8 v[16:17], v1 offset:1024

	;;#ASMEND
	scratch_load_dword v255, off, off offset:152 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b128 v[184:187], v255 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[192:195], v255 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[200:203], v255 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[208:211], v255 offset:0x1800

	;;#ASMEND
	scratch_load_dword v1, off, off offset:40 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b128 v[188:191], v1 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[196:199], v1 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[204:207], v1 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[212:215], v1 offset:0x1800

	;;#ASMEND
	scratch_load_dword v1, off, off offset:124 ; 4-byte Folded Reload
	s_add_u32 s52, s11, 0x700
	s_addc_u32 s53, s30, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s48, v1
	scratch_load_dword v1, off, off offset:120 ; 4-byte Folded Reload
	s_mov_b32 m0, s48
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s49, v1
	buffer_load_dwordx4 v248, s[52:55], 0 offen lds
	s_mov_b32 m0, s49
	s_nop 0
	buffer_load_dwordx4 v250, s[52:55], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[184:191], v[10:17], v[58:61], v18, v182 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[192:199], v[2:9], v[62:65], v18, v182 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	s_nop 7
	s_nop 2
	scratch_store_dwordx4 off, v[22:25], off offset:188 ; 16-byte Folded Spill
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[192:199], v[10:17], v[66:69], v18, v182 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[200:207], v[2:9], v[70:73], v20, v182 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[200:207], v[10:17], v[74:77], v20, v182 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[232:235], v[184:191], v[2:9], v[54:57], v18, v182 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[238:241], v[208:215], v[2:9], v[78:81], v20, v182 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[242:245], v[208:215], v[10:17], v[82:85], v20, v182 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	scratch_load_dword v1, off, off offset:28 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[216:217], v1 offset:0
ds_read_b64_tr_b8 v[218:219], v1 offset:1024

	;;#ASMEND
	s_add_i32 s50, s50, s6
	s_mov_b32 m0, s47
	;;#ASMSTART
	ds_read_b64_tr_b8 v[224:225], v231 offset:0
ds_read_b64_tr_b8 v[226:227], v231 offset:1024

	;;#ASMEND
	s_ashr_i32 s47, s50, 31
	scratch_load_dword v1, off, off offset:32 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[220:221], v1 offset:0
ds_read_b64_tr_b8 v[222:223], v1 offset:1024

	;;#ASMEND
	s_add_u32 s52, s28, s50
	v_mov_b32_e32 v1, v230
	;;#ASMSTART
	ds_read_b64_tr_b8 v[228:229], v1 offset:0
ds_read_b64_tr_b8 v[230:231], v1 offset:1024

	;;#ASMEND
	s_addc_u32 s53, s29, s47
	s_mov_b32 s54, s6
	buffer_load_dwordx4 v254, s[52:55], 0 offen lds
	s_mov_b32 m0, s37
	s_mul_i32 s37, s26, 0x780
	buffer_load_dwordx4 v252, s[52:55], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[184:191], v[216:223], v[86:89], v18, v183 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[184:191], v[224:231], v[90:93], v18, v183 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[192:199], v[216:223], v[94:97], v18, v183 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[192:199], v[224:231], v[98:101], v18, v183 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[200:207], v[216:223], v[102:105], v20, v183 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[200:207], v[224:231], v[106:109], v20, v183 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[208:215], v[216:223], v[110:113], v20, v183 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[208:215], v[224:231], v[114:117], v20, v183 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	scratch_load_dword v1, off, off offset:16 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b128 v[184:187], v1 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[192:195], v1 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[200:203], v1 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[208:211], v1 offset:0x1800

	;;#ASMEND
	s_ashr_i32 s47, s37, 31
	scratch_load_dword v1, off, off offset:36 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b128 v[188:191], v1 offset:0

	;;#ASMEND
	s_add_u32 s37, s4, s37
	;;#ASMSTART
	ds_read_b128 v[196:199], v1 offset:0x800

	;;#ASMEND
	s_addc_u32 s47, s5, s47
	;;#ASMSTART
	ds_read_b128 v[204:207], v1 offset:0x1000

	;;#ASMEND
	s_add_u32 s52, s37, 0x80
	;;#ASMSTART
	ds_read_b128 v[212:215], v1 offset:0x1800

	;;#ASMEND
	s_addc_u32 s53, s47, 0
	s_mov_b32 m0, s38
	s_nop 0
	buffer_load_dwordx4 v254, s[52:55], 0 offen lds
	s_mov_b32 m0, s40
	s_nop 0
	buffer_load_dwordx4 v252, s[52:55], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[184:191], v[2:9], v[118:121], v19, v182 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[184:191], v[10:17], v[122:125], v19, v182 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[192:199], v[2:9], v[126:129], v19, v182 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[98:101], v[192:199], v[10:17], v[130:133], v19, v182 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[102:105], v[200:207], v[2:9], v[134:137], v21, v182 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[106:109], v[200:207], v[10:17], v[138:141], v21, v182 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[110:113], v[208:215], v[2:9], v[142:145], v21, v182 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[114:117], v[208:215], v[10:17], v[146:149], v21, v182 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	scratch_load_dword v1, off, off offset:96 ; 4-byte Folded Reload
	s_add_u32 s52, s8, 0x780
	s_addc_u32 s53, s9, 0
	s_mov_b32 s54, s10
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s37, v1
	scratch_load_dword v1, off, off offset:136 ; 4-byte Folded Reload
	s_mov_b32 m0, s37
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s47, v1
	buffer_load_dwordx4 v248, s[52:55], 0 offen lds
	s_mov_b32 m0, s47
	s_nop 0
	buffer_load_dwordx4 v250, s[52:55], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[118:121], v[184:191], v[216:223], v[150:153], v19, v183 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[122:125], v[184:191], v[224:231], v[154:157], v19, v183 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[126:129], v[192:199], v[216:223], v[158:161], v19, v183 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[130:133], v[192:199], v[224:231], v[162:165], v19, v183 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[134:137], v[200:207], v[216:223], v[166:169], v21, v183 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[138:141], v[200:207], v[224:231], v[170:173], v21, v183 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[142:145], v[208:215], v[216:223], v[174:177], v21, v183 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[146:149], v[208:215], v[224:231], v[178:181], v21, v183 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_movk_i32 s38, 0xe00
	s_barrier
	buffer_load_dwordx4 v[18:21], v246, s[16:19], s34 offen
	s_add_u32 s52, s11, 0x780
	buffer_load_dwordx2 v[246:247], v247, s[20:23], s38 offen
	;;#ASMSTART
	ds_read_b64_tr_b8 v[46:47], v0 offset:0
ds_read_b64_tr_b8 v[48:49], v0 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[22:23], v236 offset:0
ds_read_b64_tr_b8 v[24:25], v236 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[50:51], v237 offset:0
ds_read_b64_tr_b8 v[52:53], v237 offset:1024

	;;#ASMEND
	scratch_load_dword v0, off, off offset:12 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[26:27], v0 offset:0
ds_read_b64_tr_b8 v[28:29], v0 offset:1024

	;;#ASMEND
	scratch_load_dword v0, off, off offset:72 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b128 v[150:153], v0 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[158:161], v0 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[166:169], v0 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[174:177], v0 offset:0x1800

	;;#ASMEND
	scratch_store_dword off, v251, off offset:160 ; 4-byte Folded Spill
	;;#ASMSTART
	ds_read_b128 v[154:157], v251 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[162:165], v251 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[170:173], v251 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[178:181], v251 offset:0x1800

	;;#ASMEND
	s_addc_u32 s53, s30, 0
	s_mov_b32 m0, s36
	s_nop 0
	buffer_load_dwordx4 v248, s[52:55], 0 offen lds
	s_mov_b32 m0, s39
	s_nop 0
	buffer_load_dwordx4 v250, s[52:55], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	scratch_load_dwordx4 v[0:3], off, off offset:188 ; 16-byte Folded Reload
	v_mfma_scale_f32_16x16x128_f8f6f4 v[214:217], v[150:157], v[46:53], v[232:235], v18, v246 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[222:225], v[158:165], v[46:53], v[30:33], v18, v246 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[226:229], v[158:165], v[22:29], v[34:37], v18, v246 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[230:233], v[166:173], v[46:53], v[38:41], v20, v246 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[234:237], v[166:173], v[22:29], v[42:45], v20, v246 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[238:241], v[174:181], v[46:53], v[238:241], v20, v246 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[242:245], v[174:181], v[22:29], v[242:245], v20, v246 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_waitcnt vmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[218:221], v[150:157], v[22:29], v[0:3], v18, v246 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_nop 4
	scratch_load_dword v0, off, off offset:68 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[8:9], v0 offset:0
ds_read_b64_tr_b8 v[10:11], v0 offset:1024

	;;#ASMEND
	s_add_i32 s36, s50, s6
	scratch_load_dword v249, off, off offset:20 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[0:1], v249 offset:0
ds_read_b64_tr_b8 v[2:3], v249 offset:1024

	;;#ASMEND
	s_lshl_b32 s38, s26, 11
	s_ashr_i32 s39, s36, 31
	scratch_load_dword v4, off, off offset:8 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[12:13], v4 offset:0
ds_read_b64_tr_b8 v[14:15], v4 offset:1024

	;;#ASMEND
	s_add_u32 s52, s28, s36
	scratch_load_dword v16, off, off offset:4 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[4:5], v16 offset:0
ds_read_b64_tr_b8 v[6:7], v16 offset:1024

	;;#ASMEND
	s_addc_u32 s53, s29, s39
	s_mov_b32 s54, s6
	s_mov_b32 m0, s41
	s_nop 0
	buffer_load_dwordx4 v254, s[52:55], 0 offen lds
	s_mov_b32 m0, s42
	s_nop 0
	buffer_load_dwordx4 v252, s[52:55], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[182:185], v[150:157], v[8:15], v[54:57], v18, v247 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[186:189], v[150:157], v[0:7], v[58:61], v18, v247 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[190:193], v[158:165], v[8:15], v[62:65], v18, v247 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[194:197], v[158:165], v[0:7], v[66:69], v18, v247 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[198:201], v[166:173], v[8:15], v[70:73], v20, v247 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[202:205], v[166:173], v[0:7], v[74:77], v20, v247 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[206:209], v[174:181], v[8:15], v[78:81], v20, v247 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[210:213], v[174:181], v[0:7], v[82:85], v20, v247 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[62:65], v253 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[70:73], v253 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[78:81], v253 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[150:153], v253 offset:0x1800

	;;#ASMEND
	s_ashr_i32 s39, s38, 31
	scratch_load_dword v16, off, off offset:156 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b128 v[66:69], v16 offset:0

	;;#ASMEND
	s_add_u32 s38, s4, s38
	;;#ASMSTART
	ds_read_b128 v[74:77], v16 offset:0x800

	;;#ASMEND
	s_addc_u32 s39, s5, s39
	;;#ASMSTART
	ds_read_b128 v[82:85], v16 offset:0x1000

	;;#ASMEND
	s_add_u32 s52, s38, 0x80
	;;#ASMSTART
	ds_read_b128 v[154:157], v16 offset:0x1800

	;;#ASMEND
	s_addc_u32 s53, s39, 0
	s_mov_b32 m0, s43
	s_nop 0
	buffer_load_dwordx4 v254, s[52:55], 0 offen lds
	s_mov_b32 m0, s44
	s_nop 0
	buffer_load_dwordx4 v252, s[52:55], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[62:69], v[46:53], v[86:89], v19, v246 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[62:69], v[22:29], v[90:93], v19, v246 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[70:77], v[46:53], v[94:97], v19, v246 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[70:77], v[22:29], v[98:101], v19, v246 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[78:85], v[46:53], v[102:105], v21, v246 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[78:85], v[22:29], v[106:109], v21, v246 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], v[150:157], v[46:53], v[110:113], v21, v246 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[150:157], v[22:29], v[114:117], v21, v246 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_u32 s40, s8, 0x800
	s_addc_u32 s41, s9, 0
	s_mov_b32 s42, s10
	s_mov_b32 s43, s7
	s_mov_b32 m0, s45
	s_nop 0
	buffer_load_dwordx4 v248, s[40:43], 0 offen lds
	s_mov_b32 m0, s46
	s_nop 0
	buffer_load_dwordx4 v250, s[40:43], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[62:69], v[8:15], v[118:121], v19, v247 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[62:69], v[0:7], v[122:125], v19, v247 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[178:181], v[150:157], v[0:7], v[146:149], v21, v247 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[158:161], v[70:77], v[8:15], v[126:129], v19, v247 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[162:165], v[70:77], v[0:7], v[130:133], v19, v247 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[166:169], v[78:85], v[8:15], v[134:137], v21, v247 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[170:173], v[78:85], v[0:7], v[138:141], v21, v247 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[174:177], v[150:157], v[8:15], v[142:145], v21, v247 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	scratch_load_dword v0, off, off offset:44 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v0 offset:0
ds_read_b64_tr_b8 v[4:5], v0 offset:1024

	;;#ASMEND
	scratch_load_dword v1, off, off offset:52 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v1 offset:0
ds_read_b64_tr_b8 v[12:13], v1 offset:1024

	;;#ASMEND
	scratch_load_dword v0, off, off         ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v0 offset:0
ds_read_b64_tr_b8 v[8:9], v0 offset:1024

	;;#ASMEND
	scratch_load_dword v251, off, off offset:48 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v251 offset:0
ds_read_b64_tr_b8 v[16:17], v251 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[90:93], v255 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[98:101], v255 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[106:109], v255 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[114:117], v255 offset:0x1800

	;;#ASMEND
	scratch_load_dword v0, off, off offset:40 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b128 v[94:97], v0 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[102:105], v0 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[110:113], v0 offset:0x1000

	;;#ASMEND
	s_add_u32 s40, s11, 0x800
	;;#ASMSTART
	ds_read_b128 v[118:121], v0 offset:0x1800

	;;#ASMEND
	s_addc_u32 s41, s30, 0
	s_mov_b32 m0, s48
	s_nop 0
	buffer_load_dwordx4 v248, s[40:43], 0 offen lds
	s_mov_b32 m0, s49
	s_nop 0
	buffer_load_dwordx4 v250, s[40:43], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[98:105], v[2:9], v[222:225], v18, v246 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[98:105], v[10:17], v[226:229], v18, v246 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[106:113], v[2:9], v[230:233], v20, v246 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[106:113], v[10:17], v[234:237], v20, v246 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[114:121], v[2:9], v[238:241], v20, v246 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[114:121], v[10:17], v[242:245], v20, v246 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[214:217], v[90:97], v[2:9], v[214:217], v18, v246 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[218:221], v[90:97], v[10:17], v[218:221], v18, v246 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	scratch_load_dword v0, off, off offset:28 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[150:151], v0 offset:0
ds_read_b64_tr_b8 v[152:153], v0 offset:1024

	;;#ASMEND
	scratch_load_dword v230, off, off offset:140 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[222:223], v230 offset:0
ds_read_b64_tr_b8 v[224:225], v230 offset:1024

	;;#ASMEND
	scratch_load_dword v231, off, off offset:32 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[154:155], v231 offset:0
ds_read_b64_tr_b8 v[156:157], v231 offset:1024

	;;#ASMEND
	scratch_load_dword v236, off, off offset:144 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[226:227], v236 offset:0
ds_read_b64_tr_b8 v[228:229], v236 offset:1024

	;;#ASMEND
	scratch_load_dword v0, off, off offset:104 ; 4-byte Folded Reload
	s_add_i32 s48, s36, s6
	s_ashr_i32 s36, s48, 31
	s_add_u32 s40, s28, s48
	s_addc_u32 s41, s29, s36
	s_mov_b32 s42, s6
	s_mul_i32 s38, s26, 0x880
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s44, v0
	scratch_load_dword v0, off, off offset:108 ; 4-byte Folded Reload
	s_mov_b32 m0, s44
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s36, v0
	buffer_load_dwordx4 v254, s[40:43], 0 offen lds
	s_mov_b32 m0, s36
	s_nop 0
	buffer_load_dwordx4 v252, s[40:43], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[90:97], v[150:157], v[182:185], v18, v247 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[90:97], v[222:229], v[186:189], v18, v247 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[98:105], v[150:157], v[190:193], v18, v247 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[98:101], v[98:105], v[222:229], v[194:197], v18, v247 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[102:105], v[106:113], v[150:157], v[198:201], v20, v247 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[106:109], v[106:113], v[222:229], v[202:205], v20, v247 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[110:113], v[114:121], v[150:157], v[206:209], v20, v247 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[114:117], v[114:121], v[222:229], v[210:213], v20, v247 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	scratch_load_dword v0, off, off offset:16 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b128 v[182:185], v0 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[190:193], v0 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[198:201], v0 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[206:209], v0 offset:0x1800

	;;#ASMEND
	scratch_load_dword v0, off, off offset:36 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b128 v[186:189], v0 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[194:197], v0 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[202:205], v0 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[210:213], v0 offset:0x1800

	;;#ASMEND
	scratch_load_dword v0, off, off offset:84 ; 4-byte Folded Reload
	s_ashr_i32 s39, s38, 31
	s_add_u32 s38, s4, s38
	s_addc_u32 s39, s5, s39
	s_add_u32 s52, s38, 0x80
	s_addc_u32 s53, s39, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s38, v0
	scratch_load_dword v0, off, off offset:88 ; 4-byte Folded Reload
	s_mov_b32 m0, s38
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s40, v0
	buffer_load_dwordx4 v254, s[52:55], 0 offen lds
	s_mov_b32 m0, s40
	s_nop 0
	buffer_load_dwordx4 v252, s[52:55], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[118:121], v[182:189], v[2:9], v[54:57], v19, v246 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[122:125], v[182:189], v[10:17], v[58:61], v19, v246 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[126:129], v[190:197], v[2:9], v[30:33], v19, v246 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[130:133], v[190:197], v[10:17], v[34:37], v19, v246 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[134:137], v[198:205], v[2:9], v[38:41], v21, v246 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[138:141], v[198:205], v[10:17], v[42:45], v21, v246 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[142:145], v[206:213], v[2:9], v[46:49], v21, v246 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[146:149], v[206:213], v[10:17], v[50:53], v21, v246 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_u32 s52, s8, 0x880
	s_addc_u32 s53, s9, 0
	s_mov_b32 s54, s10
	s_mov_b32 m0, s37
	s_nop 0
	buffer_load_dwordx4 v248, s[52:55], 0 offen lds
	s_mov_b32 m0, s47
	s_nop 0
	buffer_load_dwordx4 v250, s[52:55], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[182:189], v[150:157], v[22:25], v19, v247 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[182:189], v[222:229], v[26:29], v19, v247 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[190:197], v[150:157], v[158:161], v19, v247 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[190:197], v[222:229], v[162:165], v19, v247 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[198:205], v[150:157], v[166:169], v21, v247 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[198:205], v[222:229], v[170:173], v21, v247 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], v[206:213], v[150:157], v[174:177], v21, v247 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[206:213], v[222:229], v[178:181], v21, v247 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	scratch_load_dword v246, off, off offset:148 ; 4-byte Folded Reload
	scratch_load_dword v247, off, off offset:64 ; 4-byte Folded Reload
	scratch_load_dword v0, off, off offset:60 ; 4-byte Folded Reload
	s_add_u32 s52, s11, 0x880
	s_addc_u32 s53, s30, 0
	s_waitcnt vmcnt(2)
	buffer_load_dwordx4 v[18:21], v246, s[16:19], s35 offen
	s_waitcnt vmcnt(2)
	buffer_load_dwordx2 v[182:183], v247, s[20:23], s31 offen
	s_waitcnt vmcnt(2)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v0 offset:0
ds_read_b64_tr_b8 v[4:5], v0 offset:1024

	;;#ASMEND
	scratch_load_dword v237, off, off offset:56 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v237 offset:0
ds_read_b64_tr_b8 v[12:13], v237 offset:1024

	;;#ASMEND
	scratch_load_dword v0, off, off offset:24 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v0 offset:0
ds_read_b64_tr_b8 v[8:9], v0 offset:1024

	;;#ASMEND
	scratch_load_dword v0, off, off offset:12 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v0 offset:0
ds_read_b64_tr_b8 v[16:17], v0 offset:1024

	;;#ASMEND
	scratch_load_dword v0, off, off offset:72 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b128 v[150:153], v0 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[158:161], v0 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[166:169], v0 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[174:177], v0 offset:0x1800

	;;#ASMEND
	scratch_load_dword v0, off, off offset:160 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b128 v[154:157], v0 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[162:165], v0 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[170:173], v0 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[178:181], v0 offset:0x1800

	;;#ASMEND
	scratch_load_dword v0, off, off offset:80 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s31, v0
	scratch_load_dword v0, off, off offset:76 ; 4-byte Folded Reload
	s_mov_b32 m0, s31
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s35, v0
	buffer_load_dwordx4 v248, s[52:55], 0 offen lds
	s_mov_b32 m0, s35
	s_nop 0
	buffer_load_dwordx4 v250, s[52:55], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[150:157], v[2:9], v[214:217], v18, v182 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[150:157], v[10:17], v[218:221], v18, v182 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[158:165], v[2:9], v[62:65], v18, v182 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[158:165], v[10:17], v[66:69], v18, v182 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[166:173], v[2:9], v[70:73], v20, v182 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[166:173], v[10:17], v[74:77], v20, v182 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[174:181], v[2:9], v[78:81], v20, v182 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[174:181], v[10:17], v[82:85], v20, v182 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	scratch_load_dword v0, off, off offset:68 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[184:185], v0 offset:0
ds_read_b64_tr_b8 v[186:187], v0 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[192:193], v249 offset:0
ds_read_b64_tr_b8 v[194:195], v249 offset:1024

	;;#ASMEND
	scratch_load_dword v196, off, off offset:8 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[188:189], v196 offset:0
ds_read_b64_tr_b8 v[190:191], v196 offset:1024

	;;#ASMEND
	scratch_load_dword v200, off, off offset:4 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[196:197], v200 offset:0
ds_read_b64_tr_b8 v[198:199], v200 offset:1024

	;;#ASMEND
	scratch_load_dword v200, off, off offset:116 ; 4-byte Folded Reload
	s_add_i32 s48, s48, s6
	s_ashr_i32 s37, s48, 31
	s_add_u32 s52, s28, s48
	s_addc_u32 s53, s29, s37
	s_mov_b32 s54, s6
	s_mul_i32 s39, s26, 0x900
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s47, v200
	scratch_load_dword v200, off, off offset:112 ; 4-byte Folded Reload
	s_mov_b32 m0, s47
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s37, v200
	buffer_load_dwordx4 v254, s[52:55], 0 offen lds
	s_mov_b32 m0, s37
	s_nop 0
	buffer_load_dwordx4 v252, s[52:55], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[150:157], v[184:191], v[86:89], v18, v183 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[150:157], v[192:199], v[90:93], v18, v183 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[158:165], v[184:191], v[94:97], v18, v183 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[98:101], v[158:165], v[192:199], v[98:101], v18, v183 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[102:105], v[166:173], v[184:191], v[102:105], v20, v183 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[106:109], v[166:173], v[192:199], v[106:109], v20, v183 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[110:113], v[174:181], v[184:191], v[110:113], v20, v183 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[114:117], v[174:181], v[192:199], v[114:117], v20, v183 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[154:157], v253 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[162:165], v253 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[170:173], v253 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[200:203], v253 offset:0x1800

	;;#ASMEND
	scratch_load_dword v249, off, off offset:156 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b128 v[158:161], v249 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[166:169], v249 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[174:177], v249 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[204:207], v249 offset:0x1800

	;;#ASMEND
	scratch_load_dword v150, off, off offset:132 ; 4-byte Folded Reload
	s_ashr_i32 s41, s39, 31
	s_add_u32 s39, s4, s39
	s_addc_u32 s41, s5, s41
	s_add_u32 s52, s39, 0x80
	s_addc_u32 s53, s41, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s39, v150
	scratch_load_dword v150, off, off offset:128 ; 4-byte Folded Reload
	s_mov_b32 m0, s39
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s41, v150
	buffer_load_dwordx4 v254, s[52:55], 0 offen lds
	s_mov_b32 m0, s41
	s_nop 0
	buffer_load_dwordx4 v252, s[52:55], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[118:121], v[154:161], v[2:9], v[118:121], v19, v182 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[122:125], v[154:161], v[10:17], v[122:125], v19, v182 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[126:129], v[162:169], v[2:9], v[126:129], v19, v182 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[130:133], v[162:169], v[10:17], v[130:133], v19, v182 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[134:137], v[170:177], v[2:9], v[134:137], v21, v182 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[138:141], v[170:177], v[10:17], v[138:141], v21, v182 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[142:145], v[200:207], v[2:9], v[142:145], v21, v182 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[146:149], v[200:207], v[10:17], v[146:149], v21, v182 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	scratch_load_dword v2, off, off offset:92 ; 4-byte Folded Reload
	s_add_u32 s52, s8, 0x900
	s_addc_u32 s53, s9, 0
	s_mov_b32 s54, s10
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s42, v2
	scratch_load_dword v2, off, off offset:100 ; 4-byte Folded Reload
	s_mov_b32 m0, s42
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s43, v2
	buffer_load_dwordx4 v248, s[52:55], 0 offen lds
	s_mov_b32 m0, s43
	s_nop 0
	buffer_load_dwordx4 v250, s[52:55], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[178:181], v[200:207], v[192:199], v[50:53], v21, v183 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[150:153], v[154:161], v[184:191], v[22:25], v19, v183 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[154:157], v[154:161], v[192:199], v[26:29], v19, v183 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[158:161], v[162:169], v[184:191], v[30:33], v19, v183 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[162:165], v[162:169], v[192:199], v[34:37], v19, v183 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[166:169], v[170:177], v[184:191], v[38:41], v21, v183 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[170:173], v[170:177], v[192:199], v[42:45], v21, v183 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[174:177], v[200:207], v[184:191], v[46:49], v21, v183 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	scratch_load_dword v6, off, off offset:44 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v6 offset:0
ds_read_b64_tr_b8 v[4:5], v6 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v1 offset:0
ds_read_b64_tr_b8 v[12:13], v1 offset:1024

	;;#ASMEND
	scratch_load_dword v1, off, off         ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v1 offset:0
ds_read_b64_tr_b8 v[8:9], v1 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v251 offset:0
ds_read_b64_tr_b8 v[16:17], v251 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[184:187], v255 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[192:195], v255 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[200:203], v255 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[208:211], v255 offset:0x1800

	;;#ASMEND
	scratch_load_dword v1, off, off offset:40 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b128 v[188:191], v1 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[196:199], v1 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[204:207], v1 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[212:215], v1 offset:0x1800

	;;#ASMEND
	scratch_load_dword v22, off, off offset:124 ; 4-byte Folded Reload
	s_add_u32 s52, s11, 0x900
	s_addc_u32 s53, s30, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s45, v22
	scratch_load_dword v22, off, off offset:120 ; 4-byte Folded Reload
	s_mov_b32 m0, s45
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s46, v22
	buffer_load_dwordx4 v248, s[52:55], 0 offen lds
	s_mov_b32 m0, s46
	s_nop 0
	buffer_load_dwordx4 v250, s[52:55], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[184:191], v[10:17], v[58:61], v18, v182 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[192:199], v[2:9], v[62:65], v18, v182 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	s_nop 7
	s_nop 2
	scratch_store_dwordx4 off, v[22:25], off offset:188 ; 16-byte Folded Spill
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[192:199], v[10:17], v[66:69], v18, v182 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[200:207], v[2:9], v[70:73], v20, v182 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[200:207], v[10:17], v[74:77], v20, v182 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[232:235], v[184:191], v[2:9], v[54:57], v18, v182 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[238:241], v[208:215], v[2:9], v[78:81], v20, v182 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[242:245], v[208:215], v[10:17], v[82:85], v20, v182 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	scratch_load_dword v1, off, off offset:28 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[216:217], v1 offset:0
ds_read_b64_tr_b8 v[218:219], v1 offset:1024

	;;#ASMEND
	s_add_i32 s48, s48, s6
	s_mov_b32 m0, s44
	;;#ASMSTART
	ds_read_b64_tr_b8 v[224:225], v230 offset:0
ds_read_b64_tr_b8 v[226:227], v230 offset:1024

	;;#ASMEND
	s_ashr_i32 s44, s48, 31
	;;#ASMSTART
	ds_read_b64_tr_b8 v[220:221], v231 offset:0
ds_read_b64_tr_b8 v[222:223], v231 offset:1024

	;;#ASMEND
	s_add_u32 s52, s28, s48
	;;#ASMSTART
	ds_read_b64_tr_b8 v[228:229], v236 offset:0
ds_read_b64_tr_b8 v[230:231], v236 offset:1024

	;;#ASMEND
	s_addc_u32 s53, s29, s44
	s_mov_b32 s54, s6
	buffer_load_dwordx4 v254, s[52:55], 0 offen lds
	s_mov_b32 m0, s36
	s_mul_i32 s36, s26, 0x980
	buffer_load_dwordx4 v252, s[52:55], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[184:191], v[216:223], v[86:89], v18, v183 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[184:191], v[224:231], v[90:93], v18, v183 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[192:199], v[216:223], v[94:97], v18, v183 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[192:199], v[224:231], v[98:101], v18, v183 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[200:207], v[216:223], v[102:105], v20, v183 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[200:207], v[224:231], v[106:109], v20, v183 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[208:215], v[216:223], v[110:113], v20, v183 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[208:215], v[224:231], v[114:117], v20, v183 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	scratch_load_dword v1, off, off offset:16 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b128 v[184:187], v1 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[192:195], v1 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[200:203], v1 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[208:211], v1 offset:0x1800

	;;#ASMEND
	s_ashr_i32 s44, s36, 31
	scratch_load_dword v1, off, off offset:36 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b128 v[188:191], v1 offset:0

	;;#ASMEND
	s_add_u32 s36, s4, s36
	;;#ASMSTART
	ds_read_b128 v[196:199], v1 offset:0x800

	;;#ASMEND
	s_addc_u32 s44, s5, s44
	;;#ASMSTART
	ds_read_b128 v[204:207], v1 offset:0x1000

	;;#ASMEND
	s_add_u32 s52, s36, 0x80
	;;#ASMSTART
	ds_read_b128 v[212:215], v1 offset:0x1800

	;;#ASMEND
	s_addc_u32 s53, s44, 0
	s_mov_b32 m0, s38
	s_nop 0
	buffer_load_dwordx4 v254, s[52:55], 0 offen lds
	s_mov_b32 m0, s40
	s_nop 0
	buffer_load_dwordx4 v252, s[52:55], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[184:191], v[2:9], v[118:121], v19, v182 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[184:191], v[10:17], v[122:125], v19, v182 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[192:199], v[2:9], v[126:129], v19, v182 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[98:101], v[192:199], v[10:17], v[130:133], v19, v182 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[102:105], v[200:207], v[2:9], v[134:137], v21, v182 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[106:109], v[200:207], v[10:17], v[138:141], v21, v182 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[110:113], v[208:215], v[2:9], v[142:145], v21, v182 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[114:117], v[208:215], v[10:17], v[146:149], v21, v182 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	scratch_load_dword v2, off, off offset:96 ; 4-byte Folded Reload
	s_add_u32 s52, s8, 0x980
	s_addc_u32 s53, s9, 0
	s_mov_b32 s54, s10
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s40, v2
	scratch_load_dword v2, off, off offset:136 ; 4-byte Folded Reload
	s_mov_b32 m0, s40
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s44, v2
	buffer_load_dwordx4 v248, s[52:55], 0 offen lds
	s_mov_b32 m0, s44
	s_nop 0
	buffer_load_dwordx4 v250, s[52:55], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[118:121], v[184:191], v[216:223], v[150:153], v19, v183 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[122:125], v[184:191], v[224:231], v[154:157], v19, v183 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[126:129], v[192:199], v[216:223], v[158:161], v19, v183 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[130:133], v[192:199], v[224:231], v[162:165], v19, v183 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[134:137], v[200:207], v[216:223], v[166:169], v21, v183 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[138:141], v[200:207], v[224:231], v[170:173], v21, v183 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[142:145], v[208:215], v[216:223], v[174:177], v21, v183 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[146:149], v[208:215], v[224:231], v[178:181], v21, v183 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_movk_i32 s36, 0x2400
	s_barrier
	buffer_load_dwordx4 v[18:21], v246, s[16:19], s36 offen
	s_movk_i32 s36, 0x1200
	buffer_load_dwordx2 v[246:247], v247, s[20:23], s36 offen
	s_add_u32 s52, s11, 0x980
	scratch_load_dword v1, off, off offset:60 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[46:47], v1 offset:0
ds_read_b64_tr_b8 v[48:49], v1 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[22:23], v237 offset:0
ds_read_b64_tr_b8 v[24:25], v237 offset:1024

	;;#ASMEND
	scratch_load_dword v1, off, off offset:24 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[50:51], v1 offset:0
ds_read_b64_tr_b8 v[52:53], v1 offset:1024

	;;#ASMEND
	scratch_load_dword v1, off, off offset:12 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[26:27], v1 offset:0
ds_read_b64_tr_b8 v[28:29], v1 offset:1024

	;;#ASMEND
	scratch_load_dword v1, off, off offset:72 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b128 v[150:153], v1 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[158:161], v1 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[166:169], v1 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[174:177], v1 offset:0x1800

	;;#ASMEND
	scratch_load_dword v251, off, off offset:160 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b128 v[154:157], v251 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[162:165], v251 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[170:173], v251 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[178:181], v251 offset:0x1800

	;;#ASMEND
	s_addc_u32 s53, s30, 0
	s_mov_b32 m0, s31
	s_nop 0
	buffer_load_dwordx4 v248, s[52:55], 0 offen lds
	s_mov_b32 m0, s35
	s_nop 0
	buffer_load_dwordx4 v250, s[52:55], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	scratch_load_dwordx4 v[2:5], off, off offset:188 ; 16-byte Folded Reload
	v_mfma_scale_f32_16x16x128_f8f6f4 v[214:217], v[150:157], v[46:53], v[232:235], v18, v246 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[222:225], v[158:165], v[46:53], v[30:33], v18, v246 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[226:229], v[158:165], v[22:29], v[34:37], v18, v246 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[230:233], v[166:173], v[46:53], v[38:41], v20, v246 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[234:237], v[166:173], v[22:29], v[42:45], v20, v246 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[238:241], v[174:181], v[46:53], v[238:241], v20, v246 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[242:245], v[174:181], v[22:29], v[242:245], v20, v246 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_waitcnt vmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[218:221], v[150:157], v[22:29], v[2:5], v18, v246 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[8:9], v0 offset:0
ds_read_b64_tr_b8 v[10:11], v0 offset:1024

	;;#ASMEND
	s_add_i32 s31, s48, s6
	s_nop 3
	scratch_load_dword v4, off, off offset:20 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[0:1], v4 offset:0
ds_read_b64_tr_b8 v[2:3], v4 offset:1024

	;;#ASMEND
	s_ashr_i32 s35, s31, 31
	scratch_load_dword v4, off, off offset:8 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[12:13], v4 offset:0
ds_read_b64_tr_b8 v[14:15], v4 offset:1024

	;;#ASMEND
	s_add_u32 s48, s28, s31
	s_mov_b32 m0, s47
	scratch_load_dword v16, off, off offset:4 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[4:5], v16 offset:0
ds_read_b64_tr_b8 v[6:7], v16 offset:1024

	;;#ASMEND
	s_addc_u32 s49, s29, s35
	s_mov_b32 s50, s6
	buffer_load_dwordx4 v254, s[48:51], 0 offen lds
	s_mov_b32 m0, s37
	s_mul_i32 s35, s26, 0xa00
	buffer_load_dwordx4 v252, s[48:51], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[182:185], v[150:157], v[8:15], v[54:57], v18, v247 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[186:189], v[150:157], v[0:7], v[58:61], v18, v247 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[190:193], v[158:165], v[8:15], v[62:65], v18, v247 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[194:197], v[158:165], v[0:7], v[66:69], v18, v247 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[198:201], v[166:173], v[8:15], v[70:73], v20, v247 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[202:205], v[166:173], v[0:7], v[74:77], v20, v247 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[206:209], v[174:181], v[8:15], v[78:81], v20, v247 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[210:213], v[174:181], v[0:7], v[82:85], v20, v247 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	scratch_store_dword off, v253, off offset:164 ; 4-byte Folded Spill
	;;#ASMSTART
	ds_read_b128 v[62:65], v253 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[70:73], v253 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[78:81], v253 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[150:153], v253 offset:0x1800

	;;#ASMEND
	s_ashr_i32 s36, s35, 31
	;;#ASMSTART
	ds_read_b128 v[66:69], v249 offset:0

	;;#ASMEND
	s_add_u32 s35, s4, s35
	;;#ASMSTART
	ds_read_b128 v[74:77], v249 offset:0x800

	;;#ASMEND
	s_addc_u32 s36, s5, s36
	;;#ASMSTART
	ds_read_b128 v[82:85], v249 offset:0x1000

	;;#ASMEND
	s_add_u32 s48, s35, 0x80
	;;#ASMSTART
	ds_read_b128 v[154:157], v249 offset:0x1800

	;;#ASMEND
	s_addc_u32 s49, s36, 0
	s_mov_b32 m0, s39
	s_nop 0
	buffer_load_dwordx4 v254, s[48:51], 0 offen lds
	s_mov_b32 m0, s41
	s_nop 0
	buffer_load_dwordx4 v252, s[48:51], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[62:69], v[46:53], v[86:89], v19, v246 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[62:69], v[22:29], v[90:93], v19, v246 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[70:77], v[46:53], v[94:97], v19, v246 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[70:77], v[22:29], v[98:101], v19, v246 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[78:85], v[46:53], v[102:105], v21, v246 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[78:85], v[22:29], v[106:109], v21, v246 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], v[150:157], v[46:53], v[110:113], v21, v246 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[150:157], v[22:29], v[114:117], v21, v246 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_u32 s36, s8, 0xa00
	s_addc_u32 s37, s9, 0
	s_mov_b32 s38, s10
	s_mov_b32 s39, s7
	s_mov_b32 m0, s42
	s_nop 0
	buffer_load_dwordx4 v248, s[36:39], 0 offen lds
	s_mov_b32 m0, s43
	s_nop 0
	buffer_load_dwordx4 v250, s[36:39], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[62:69], v[8:15], v[118:121], v19, v247 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[62:69], v[0:7], v[122:125], v19, v247 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[178:181], v[150:157], v[0:7], v[146:149], v21, v247 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[158:161], v[70:77], v[8:15], v[126:129], v19, v247 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[162:165], v[70:77], v[0:7], v[130:133], v19, v247 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[166:169], v[78:85], v[8:15], v[134:137], v21, v247 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[170:173], v[78:85], v[0:7], v[138:141], v21, v247 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[174:177], v[150:157], v[8:15], v[142:145], v21, v247 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	scratch_load_dword v255, off, off offset:44 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v255 offset:0
ds_read_b64_tr_b8 v[4:5], v255 offset:1024

	;;#ASMEND
	scratch_load_dword v253, off, off offset:52 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v253 offset:0
ds_read_b64_tr_b8 v[12:13], v253 offset:1024

	;;#ASMEND
	scratch_load_dword v249, off, off       ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v249 offset:0
ds_read_b64_tr_b8 v[8:9], v249 offset:1024

	;;#ASMEND
	scratch_load_dword v1, off, off offset:48 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v1 offset:0
ds_read_b64_tr_b8 v[16:17], v1 offset:1024

	;;#ASMEND
	scratch_load_dword v0, off, off offset:152 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b128 v[90:93], v0 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[98:101], v0 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[106:109], v0 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[114:117], v0 offset:0x1800

	;;#ASMEND
	scratch_load_dword v0, off, off offset:40 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b128 v[94:97], v0 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[102:105], v0 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[110:113], v0 offset:0x1000

	;;#ASMEND
	s_add_u32 s36, s11, 0xa00
	;;#ASMSTART
	ds_read_b128 v[118:121], v0 offset:0x1800

	;;#ASMEND
	s_addc_u32 s37, s30, 0
	s_mov_b32 m0, s45
	s_nop 0
	buffer_load_dwordx4 v248, s[36:39], 0 offen lds
	s_mov_b32 m0, s46
	s_nop 0
	buffer_load_dwordx4 v250, s[36:39], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[98:105], v[2:9], v[222:225], v18, v246 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[98:105], v[10:17], v[226:229], v18, v246 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[106:113], v[2:9], v[230:233], v20, v246 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[106:113], v[10:17], v[234:237], v20, v246 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[114:121], v[2:9], v[238:241], v20, v246 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[114:121], v[10:17], v[242:245], v20, v246 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[214:217], v[90:97], v[2:9], v[214:217], v18, v246 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[218:221], v[90:97], v[10:17], v[218:221], v18, v246 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	scratch_load_dword v0, off, off offset:28 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[150:151], v0 offset:0
ds_read_b64_tr_b8 v[152:153], v0 offset:1024

	;;#ASMEND
	scratch_load_dword v231, off, off offset:140 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[222:223], v231 offset:0
ds_read_b64_tr_b8 v[224:225], v231 offset:1024

	;;#ASMEND
	scratch_load_dword v0, off, off offset:32 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[154:155], v0 offset:0
ds_read_b64_tr_b8 v[156:157], v0 offset:1024

	;;#ASMEND
	scratch_load_dword v230, off, off offset:144 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[226:227], v230 offset:0
ds_read_b64_tr_b8 v[228:229], v230 offset:1024

	;;#ASMEND
	scratch_load_dword v0, off, off offset:104 ; 4-byte Folded Reload
	s_add_i32 s37, s31, s6
	s_ashr_i32 s31, s37, 31
	s_add_u32 s48, s28, s37
	s_addc_u32 s49, s29, s31
	s_mul_i32 s35, s26, 0xa80
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s45, v0
	scratch_load_dword v0, off, off offset:108 ; 4-byte Folded Reload
	s_mov_b32 m0, s45
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s31, v0
	buffer_load_dwordx4 v254, s[48:51], 0 offen lds
	s_mov_b32 m0, s31
	s_nop 0
	buffer_load_dwordx4 v252, s[48:51], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[90:97], v[150:157], v[182:185], v18, v247 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[90:97], v[222:229], v[186:189], v18, v247 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[98:105], v[150:157], v[190:193], v18, v247 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[98:101], v[98:105], v[222:229], v[194:197], v18, v247 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[102:105], v[106:113], v[150:157], v[198:201], v20, v247 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[106:109], v[106:113], v[222:229], v[202:205], v20, v247 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[110:113], v[114:121], v[150:157], v[206:209], v20, v247 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[114:117], v[114:121], v[222:229], v[210:213], v20, v247 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	scratch_load_dword v0, off, off offset:16 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b128 v[182:185], v0 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[190:193], v0 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[198:201], v0 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[206:209], v0 offset:0x1800

	;;#ASMEND
	scratch_load_dword v0, off, off offset:36 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b128 v[186:189], v0 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[194:197], v0 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[202:205], v0 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[210:213], v0 offset:0x1800

	;;#ASMEND
	scratch_load_dword v0, off, off offset:84 ; 4-byte Folded Reload
	s_ashr_i32 s36, s35, 31
	s_add_u32 s35, s4, s35
	s_addc_u32 s36, s5, s36
	s_add_u32 s48, s35, 0x80
	s_addc_u32 s49, s36, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s36, v0
	scratch_load_dword v0, off, off offset:88 ; 4-byte Folded Reload
	s_mov_b32 m0, s36
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s38, v0
	buffer_load_dwordx4 v254, s[48:51], 0 offen lds
	s_mov_b32 m0, s38
	s_nop 0
	buffer_load_dwordx4 v252, s[48:51], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[118:121], v[182:189], v[2:9], v[54:57], v19, v246 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[122:125], v[182:189], v[10:17], v[58:61], v19, v246 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[126:129], v[190:197], v[2:9], v[30:33], v19, v246 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[130:133], v[190:197], v[10:17], v[34:37], v19, v246 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[134:137], v[198:205], v[2:9], v[38:41], v21, v246 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[138:141], v[198:205], v[10:17], v[42:45], v21, v246 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[142:145], v[206:213], v[2:9], v[46:49], v21, v246 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[146:149], v[206:213], v[10:17], v[50:53], v21, v246 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_u32 s48, s8, 0xa80
	s_addc_u32 s49, s9, 0
	s_mov_b32 s50, s10
	s_mov_b32 m0, s40
	s_nop 0
	buffer_load_dwordx4 v248, s[48:51], 0 offen lds
	s_mov_b32 m0, s44
	s_nop 0
	buffer_load_dwordx4 v250, s[48:51], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[182:189], v[150:157], v[22:25], v19, v247 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[182:189], v[222:229], v[26:29], v19, v247 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[190:197], v[150:157], v[158:161], v19, v247 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[190:197], v[222:229], v[162:165], v19, v247 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[198:205], v[150:157], v[166:169], v21, v247 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[198:205], v[222:229], v[170:173], v21, v247 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], v[206:213], v[150:157], v[174:177], v21, v247 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[206:213], v[222:229], v[178:181], v21, v247 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	scratch_load_dword v246, off, off offset:148 ; 4-byte Folded Reload
	scratch_load_dword v247, off, off offset:64 ; 4-byte Folded Reload
	scratch_load_dword v236, off, off offset:60 ; 4-byte Folded Reload
	s_movk_i32 s35, 0x2800
	s_add_u32 s40, s11, 0xa80
	s_addc_u32 s41, s30, 0
	s_mov_b32 s42, s10
	s_mov_b32 s43, s7
	s_waitcnt vmcnt(2)
	buffer_load_dwordx4 v[18:21], v246, s[16:19], s35 offen
	s_waitcnt vmcnt(2)
	buffer_load_dwordx2 v[182:183], v247, s[20:23], s3 offen
	s_waitcnt vmcnt(2)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v236 offset:0
ds_read_b64_tr_b8 v[4:5], v236 offset:1024

	;;#ASMEND
	scratch_load_dword v0, off, off offset:56 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v0 offset:0
ds_read_b64_tr_b8 v[12:13], v0 offset:1024

	;;#ASMEND
	scratch_load_dword v237, off, off offset:24 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v237 offset:0
ds_read_b64_tr_b8 v[8:9], v237 offset:1024

	;;#ASMEND
	scratch_load_dword v0, off, off offset:12 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v0 offset:0
ds_read_b64_tr_b8 v[16:17], v0 offset:1024

	;;#ASMEND
	scratch_load_dword v0, off, off offset:72 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b128 v[150:153], v0 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[158:161], v0 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[166:169], v0 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[174:177], v0 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[154:157], v251 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[162:165], v251 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[170:173], v251 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[178:181], v251 offset:0x1800

	;;#ASMEND
	scratch_load_dword v54, off, off offset:80 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s3, v54
	scratch_load_dword v54, off, off offset:76 ; 4-byte Folded Reload
	s_mov_b32 m0, s3
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s35, v54
	buffer_load_dwordx4 v248, s[40:43], 0 offen lds
	s_mov_b32 m0, s35
	s_nop 0
	buffer_load_dwordx4 v250, s[40:43], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[150:157], v[2:9], v[214:217], v18, v182 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[150:157], v[10:17], v[218:221], v18, v182 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[158:165], v[2:9], v[62:65], v18, v182 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[158:165], v[10:17], v[66:69], v18, v182 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[166:173], v[2:9], v[70:73], v20, v182 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[166:173], v[10:17], v[74:77], v20, v182 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[174:181], v[2:9], v[78:81], v20, v182 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[174:181], v[10:17], v[82:85], v20, v182 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	scratch_load_dword v0, off, off offset:68 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[184:185], v0 offset:0
ds_read_b64_tr_b8 v[186:187], v0 offset:1024

	;;#ASMEND
	scratch_load_dword v188, off, off offset:20 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[192:193], v188 offset:0
ds_read_b64_tr_b8 v[194:195], v188 offset:1024

	;;#ASMEND
	scratch_load_dword v196, off, off offset:8 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[188:189], v196 offset:0
ds_read_b64_tr_b8 v[190:191], v196 offset:1024

	;;#ASMEND
	scratch_load_dword v200, off, off offset:4 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[196:197], v200 offset:0
ds_read_b64_tr_b8 v[198:199], v200 offset:1024

	;;#ASMEND
	scratch_load_dword v200, off, off offset:116 ; 4-byte Folded Reload
	s_add_i32 s47, s37, s6
	s_ashr_i32 s37, s47, 31
	s_add_u32 s40, s28, s47
	s_addc_u32 s41, s29, s37
	s_mov_b32 s42, s6
	s_mul_i32 s39, s26, 0xb00
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s46, v200
	scratch_load_dword v200, off, off offset:112 ; 4-byte Folded Reload
	s_mov_b32 m0, s46
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s37, v200
	buffer_load_dwordx4 v254, s[40:43], 0 offen lds
	s_mov_b32 m0, s37
	s_nop 0
	buffer_load_dwordx4 v252, s[40:43], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[150:157], v[184:191], v[86:89], v18, v183 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[150:157], v[192:199], v[90:93], v18, v183 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[158:165], v[184:191], v[94:97], v18, v183 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[98:101], v[158:165], v[192:199], v[98:101], v18, v183 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[102:105], v[166:173], v[184:191], v[102:105], v20, v183 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[106:109], v[166:173], v[192:199], v[106:109], v20, v183 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[110:113], v[174:181], v[184:191], v[110:113], v20, v183 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[114:117], v[174:181], v[192:199], v[114:117], v20, v183 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	scratch_load_dword v150, off, off offset:164 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b128 v[154:157], v150 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[162:165], v150 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[170:173], v150 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[200:203], v150 offset:0x1800

	;;#ASMEND
	scratch_load_dword v150, off, off offset:156 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b128 v[158:161], v150 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[166:169], v150 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[174:177], v150 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[204:207], v150 offset:0x1800

	;;#ASMEND
	scratch_load_dword v150, off, off offset:132 ; 4-byte Folded Reload
	s_ashr_i32 s40, s39, 31
	s_add_u32 s39, s4, s39
	s_addc_u32 s40, s5, s40
	s_add_u32 s48, s39, 0x80
	s_addc_u32 s49, s40, 0
	s_mov_b32 s50, s6
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s39, v150
	scratch_load_dword v150, off, off offset:128 ; 4-byte Folded Reload
	s_mov_b32 m0, s39
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s40, v150
	buffer_load_dwordx4 v254, s[48:51], 0 offen lds
	s_mov_b32 m0, s40
	s_nop 0
	buffer_load_dwordx4 v252, s[48:51], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[118:121], v[154:161], v[2:9], v[118:121], v19, v182 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[122:125], v[154:161], v[10:17], v[122:125], v19, v182 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[126:129], v[162:169], v[2:9], v[126:129], v19, v182 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[130:133], v[162:169], v[10:17], v[130:133], v19, v182 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[134:137], v[170:177], v[2:9], v[134:137], v21, v182 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[138:141], v[170:177], v[10:17], v[138:141], v21, v182 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[142:145], v[200:207], v[2:9], v[142:145], v21, v182 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[146:149], v[200:207], v[10:17], v[146:149], v21, v182 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	scratch_load_dword v2, off, off offset:92 ; 4-byte Folded Reload
	s_add_u32 s48, s8, 0xb00
	s_addc_u32 s49, s9, 0
	s_mov_b32 s50, s10
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s41, v2
	scratch_load_dword v2, off, off offset:100 ; 4-byte Folded Reload
	s_mov_b32 m0, s41
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s42, v2
	buffer_load_dwordx4 v248, s[48:51], 0 offen lds
	s_mov_b32 m0, s42
	s_nop 0
	buffer_load_dwordx4 v250, s[48:51], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[178:181], v[200:207], v[192:199], v[50:53], v21, v183 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[150:153], v[154:161], v[184:191], v[22:25], v19, v183 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[154:157], v[154:161], v[192:199], v[26:29], v19, v183 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[158:161], v[162:169], v[184:191], v[30:33], v19, v183 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[162:165], v[162:169], v[192:199], v[34:37], v19, v183 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[166:169], v[170:177], v[184:191], v[38:41], v21, v183 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[170:173], v[170:177], v[192:199], v[42:45], v21, v183 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[174:177], v[200:207], v[184:191], v[46:49], v21, v183 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v255 offset:0
ds_read_b64_tr_b8 v[4:5], v255 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v253 offset:0
ds_read_b64_tr_b8 v[12:13], v253 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v249 offset:0
ds_read_b64_tr_b8 v[8:9], v249 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v1 offset:0
ds_read_b64_tr_b8 v[16:17], v1 offset:1024

	;;#ASMEND
	scratch_load_dword v255, off, off offset:152 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b128 v[184:187], v255 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[192:195], v255 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[200:203], v255 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[208:211], v255 offset:0x1800

	;;#ASMEND
	scratch_load_dword v1, off, off offset:40 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b128 v[188:191], v1 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[196:199], v1 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[204:207], v1 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[212:215], v1 offset:0x1800

	;;#ASMEND
	scratch_load_dword v22, off, off offset:124 ; 4-byte Folded Reload
	s_add_u32 s48, s11, 0xb00
	s_addc_u32 s49, s30, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s43, v22
	scratch_load_dword v22, off, off offset:120 ; 4-byte Folded Reload
	s_mov_b32 m0, s43
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s44, v22
	buffer_load_dwordx4 v248, s[48:51], 0 offen lds
	s_mov_b32 m0, s44
	s_nop 0
	buffer_load_dwordx4 v250, s[48:51], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[184:191], v[10:17], v[58:61], v18, v182 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[192:199], v[2:9], v[62:65], v18, v182 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	s_nop 7
	s_nop 2
	scratch_store_dwordx4 off, v[22:25], off offset:188 ; 16-byte Folded Spill
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[192:199], v[10:17], v[66:69], v18, v182 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[200:207], v[2:9], v[70:73], v20, v182 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[200:207], v[10:17], v[74:77], v20, v182 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[232:235], v[184:191], v[2:9], v[54:57], v18, v182 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[238:241], v[208:215], v[2:9], v[78:81], v20, v182 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[242:245], v[208:215], v[10:17], v[82:85], v20, v182 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_mov_b32 m0, s45
	scratch_load_dword v1, off, off offset:28 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[216:217], v1 offset:0
ds_read_b64_tr_b8 v[218:219], v1 offset:1024

	;;#ASMEND
	s_add_i32 s45, s47, s6
	;;#ASMSTART
	ds_read_b64_tr_b8 v[224:225], v231 offset:0
ds_read_b64_tr_b8 v[226:227], v231 offset:1024

	;;#ASMEND
	s_ashr_i32 s47, s45, 31
	scratch_load_dword v1, off, off offset:32 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[220:221], v1 offset:0
ds_read_b64_tr_b8 v[222:223], v1 offset:1024

	;;#ASMEND
	s_add_u32 s48, s28, s45
	v_mov_b32_e32 v1, v230
	;;#ASMSTART
	ds_read_b64_tr_b8 v[228:229], v1 offset:0
ds_read_b64_tr_b8 v[230:231], v1 offset:1024

	;;#ASMEND
	s_addc_u32 s49, s29, s47
	s_mov_b32 s50, s6
	buffer_load_dwordx4 v254, s[48:51], 0 offen lds
	s_mov_b32 m0, s31
	s_mul_i32 s31, s26, 0xb80
	buffer_load_dwordx4 v252, s[48:51], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[184:191], v[216:223], v[86:89], v18, v183 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[184:191], v[224:231], v[90:93], v18, v183 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[192:199], v[216:223], v[94:97], v18, v183 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[192:199], v[224:231], v[98:101], v18, v183 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[200:207], v[216:223], v[102:105], v20, v183 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[200:207], v[224:231], v[106:109], v20, v183 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[208:215], v[216:223], v[110:113], v20, v183 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[208:215], v[224:231], v[114:117], v20, v183 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	scratch_load_dword v1, off, off offset:16 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b128 v[184:187], v1 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[192:195], v1 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[200:203], v1 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[208:211], v1 offset:0x1800

	;;#ASMEND
	s_ashr_i32 s47, s31, 31
	scratch_load_dword v1, off, off offset:36 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b128 v[188:191], v1 offset:0

	;;#ASMEND
	s_add_u32 s31, s4, s31
	;;#ASMSTART
	ds_read_b128 v[196:199], v1 offset:0x800

	;;#ASMEND
	s_addc_u32 s47, s5, s47
	;;#ASMSTART
	ds_read_b128 v[204:207], v1 offset:0x1000

	;;#ASMEND
	s_add_u32 s48, s31, 0x80
	;;#ASMSTART
	ds_read_b128 v[212:215], v1 offset:0x1800

	;;#ASMEND
	s_addc_u32 s49, s47, 0
	s_mov_b32 m0, s36
	s_nop 0
	buffer_load_dwordx4 v254, s[48:51], 0 offen lds
	s_mov_b32 m0, s38
	s_nop 0
	buffer_load_dwordx4 v252, s[48:51], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[184:191], v[2:9], v[118:121], v19, v182 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[184:191], v[10:17], v[122:125], v19, v182 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[192:199], v[2:9], v[126:129], v19, v182 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[98:101], v[192:199], v[10:17], v[130:133], v19, v182 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[102:105], v[200:207], v[2:9], v[134:137], v21, v182 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[106:109], v[200:207], v[10:17], v[138:141], v21, v182 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[110:113], v[208:215], v[2:9], v[142:145], v21, v182 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[114:117], v[208:215], v[10:17], v[146:149], v21, v182 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	scratch_load_dword v2, off, off offset:96 ; 4-byte Folded Reload
	s_add_u32 s48, s8, 0xb80
	s_addc_u32 s49, s9, 0
	s_mov_b32 s50, s10
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s31, v2
	scratch_load_dword v2, off, off offset:136 ; 4-byte Folded Reload
	s_mov_b32 m0, s31
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s36, v2
	buffer_load_dwordx4 v248, s[48:51], 0 offen lds
	s_mov_b32 m0, s36
	s_nop 0
	buffer_load_dwordx4 v250, s[48:51], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[118:121], v[184:191], v[216:223], v[150:153], v19, v183 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[122:125], v[184:191], v[224:231], v[154:157], v19, v183 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[126:129], v[192:199], v[216:223], v[158:161], v19, v183 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[130:133], v[192:199], v[224:231], v[162:165], v19, v183 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[134:137], v[200:207], v[216:223], v[166:169], v21, v183 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[138:141], v[200:207], v[224:231], v[170:173], v21, v183 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[142:145], v[208:215], v[216:223], v[174:177], v21, v183 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[146:149], v[208:215], v[224:231], v[178:181], v21, v183 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_movk_i32 s38, 0x2c00
	s_barrier
	buffer_load_dwordx4 v[18:21], v246, s[16:19], s38 offen
	s_movk_i32 s38, 0x1600
	buffer_load_dwordx2 v[246:247], v247, s[20:23], s38 offen
	;;#ASMSTART
	ds_read_b64_tr_b8 v[46:47], v236 offset:0
ds_read_b64_tr_b8 v[48:49], v236 offset:1024

	;;#ASMEND
	scratch_load_dword v1, off, off offset:56 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[22:23], v1 offset:0
ds_read_b64_tr_b8 v[24:25], v1 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[50:51], v237 offset:0
ds_read_b64_tr_b8 v[52:53], v237 offset:1024

	;;#ASMEND
	scratch_load_dword v1, off, off offset:12 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[26:27], v1 offset:0
ds_read_b64_tr_b8 v[28:29], v1 offset:1024

	;;#ASMEND
	scratch_load_dword v1, off, off offset:72 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b128 v[150:153], v1 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[158:161], v1 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[166:169], v1 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[174:177], v1 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[154:157], v251 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[162:165], v251 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[170:173], v251 offset:0x1000

	;;#ASMEND
	s_add_u32 s48, s11, 0xb80
	;;#ASMSTART
	ds_read_b128 v[178:181], v251 offset:0x1800

	;;#ASMEND
	s_addc_u32 s49, s30, 0
	s_mov_b32 m0, s3
	s_nop 0
	buffer_load_dwordx4 v248, s[48:51], 0 offen lds
	s_mov_b32 m0, s35
	s_nop 0
	buffer_load_dwordx4 v250, s[48:51], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	scratch_load_dwordx4 v[2:5], off, off offset:188 ; 16-byte Folded Reload
	v_mfma_scale_f32_16x16x128_f8f6f4 v[214:217], v[150:157], v[46:53], v[232:235], v18, v246 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[222:225], v[158:165], v[46:53], v[30:33], v18, v246 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[226:229], v[158:165], v[22:29], v[34:37], v18, v246 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[230:233], v[166:173], v[46:53], v[38:41], v20, v246 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[234:237], v[166:173], v[22:29], v[42:45], v20, v246 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[238:241], v[174:181], v[46:53], v[238:241], v20, v246 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[242:245], v[174:181], v[22:29], v[242:245], v20, v246 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_waitcnt vmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[218:221], v[150:157], v[22:29], v[2:5], v18, v246 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[8:9], v0 offset:0
ds_read_b64_tr_b8 v[10:11], v0 offset:1024

	;;#ASMEND
	s_add_i32 s38, s45, s6
	s_nop 3
	scratch_load_dword v4, off, off offset:20 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[0:1], v4 offset:0
ds_read_b64_tr_b8 v[2:3], v4 offset:1024

	;;#ASMEND
	s_ashr_i32 s3, s38, 31
	scratch_load_dword v4, off, off offset:8 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[12:13], v4 offset:0
ds_read_b64_tr_b8 v[14:15], v4 offset:1024

	;;#ASMEND
	s_add_u32 s48, s28, s38
	s_mov_b32 m0, s46
	scratch_load_dword v16, off, off offset:4 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[4:5], v16 offset:0
ds_read_b64_tr_b8 v[6:7], v16 offset:1024

	;;#ASMEND
	s_addc_u32 s49, s29, s3
	s_mov_b32 s50, s6
	buffer_load_dwordx4 v254, s[48:51], 0 offen lds
	s_mov_b32 m0, s37
	s_mul_i32 s3, s26, 0xc00
	buffer_load_dwordx4 v252, s[48:51], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[182:185], v[150:157], v[8:15], v[54:57], v18, v247 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[186:189], v[150:157], v[0:7], v[58:61], v18, v247 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[190:193], v[158:165], v[8:15], v[62:65], v18, v247 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[194:197], v[158:165], v[0:7], v[66:69], v18, v247 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[198:201], v[166:173], v[8:15], v[70:73], v20, v247 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[202:205], v[166:173], v[0:7], v[74:77], v20, v247 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[206:209], v[174:181], v[8:15], v[78:81], v20, v247 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[210:213], v[174:181], v[0:7], v[82:85], v20, v247 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	scratch_load_dword v16, off, off offset:164 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b128 v[62:65], v16 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[70:73], v16 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[78:81], v16 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[150:153], v16 offset:0x1800

	;;#ASMEND
	s_ashr_i32 s35, s3, 31
	scratch_load_dword v249, off, off offset:156 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b128 v[66:69], v249 offset:0

	;;#ASMEND
	s_add_u32 s3, s4, s3
	;;#ASMSTART
	ds_read_b128 v[74:77], v249 offset:0x800

	;;#ASMEND
	s_addc_u32 s35, s5, s35
	;;#ASMSTART
	ds_read_b128 v[82:85], v249 offset:0x1000

	;;#ASMEND
	s_add_u32 s48, s3, 0x80
	;;#ASMSTART
	ds_read_b128 v[154:157], v249 offset:0x1800

	;;#ASMEND
	s_addc_u32 s49, s35, 0
	s_mov_b32 m0, s39
	s_nop 0
	buffer_load_dwordx4 v254, s[48:51], 0 offen lds
	s_mov_b32 m0, s40
	s_nop 0
	buffer_load_dwordx4 v252, s[48:51], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[62:69], v[46:53], v[86:89], v19, v246 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[62:69], v[22:29], v[90:93], v19, v246 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[70:77], v[46:53], v[94:97], v19, v246 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[70:77], v[22:29], v[98:101], v19, v246 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[78:85], v[46:53], v[102:105], v21, v246 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[78:85], v[22:29], v[106:109], v21, v246 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], v[150:157], v[46:53], v[110:113], v21, v246 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[150:157], v[22:29], v[114:117], v21, v246 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_u32 s48, s8, 0xc00
	s_addc_u32 s49, s9, 0
	s_mov_b32 s50, s10
	s_mov_b32 m0, s41
	s_nop 0
	buffer_load_dwordx4 v248, s[48:51], 0 offen lds
	s_mov_b32 m0, s42
	s_nop 0
	buffer_load_dwordx4 v250, s[48:51], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[62:69], v[8:15], v[118:121], v19, v247 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[62:69], v[0:7], v[122:125], v19, v247 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[178:181], v[150:157], v[0:7], v[146:149], v21, v247 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[158:161], v[70:77], v[8:15], v[126:129], v19, v247 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[162:165], v[70:77], v[0:7], v[130:133], v19, v247 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[166:169], v[78:85], v[8:15], v[134:137], v21, v247 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[170:173], v[78:85], v[0:7], v[138:141], v21, v247 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[174:177], v[150:157], v[8:15], v[142:145], v21, v247 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	scratch_load_dword v1, off, off offset:44 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v1 offset:0
ds_read_b64_tr_b8 v[4:5], v1 offset:1024

	;;#ASMEND
	scratch_load_dword v253, off, off offset:52 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v253 offset:0
ds_read_b64_tr_b8 v[12:13], v253 offset:1024

	;;#ASMEND
	scratch_load_dword v0, off, off         ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v0 offset:0
ds_read_b64_tr_b8 v[8:9], v0 offset:1024

	;;#ASMEND
	scratch_load_dword v251, off, off offset:48 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v251 offset:0
ds_read_b64_tr_b8 v[16:17], v251 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[90:93], v255 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[98:101], v255 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[106:109], v255 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[114:117], v255 offset:0x1800

	;;#ASMEND
	scratch_load_dword v0, off, off offset:40 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b128 v[94:97], v0 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[102:105], v0 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[110:113], v0 offset:0x1000

	;;#ASMEND
	s_add_u32 s48, s11, 0xc00
	;;#ASMSTART
	ds_read_b128 v[118:121], v0 offset:0x1800

	;;#ASMEND
	s_addc_u32 s49, s30, 0
	s_mov_b32 m0, s43
	s_nop 0
	buffer_load_dwordx4 v248, s[48:51], 0 offen lds
	s_mov_b32 m0, s44
	s_nop 0
	buffer_load_dwordx4 v250, s[48:51], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[98:105], v[2:9], v[222:225], v18, v246 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[98:105], v[10:17], v[226:229], v18, v246 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[106:113], v[2:9], v[230:233], v20, v246 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[106:113], v[10:17], v[234:237], v20, v246 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[114:121], v[2:9], v[238:241], v20, v246 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[114:121], v[10:17], v[242:245], v20, v246 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[214:217], v[90:97], v[2:9], v[214:217], v18, v246 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[218:221], v[90:97], v[10:17], v[218:221], v18, v246 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	scratch_load_dword v255, off, off offset:28 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[150:151], v255 offset:0
ds_read_b64_tr_b8 v[152:153], v255 offset:1024

	;;#ASMEND
	scratch_load_dword v231, off, off offset:140 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[222:223], v231 offset:0
ds_read_b64_tr_b8 v[224:225], v231 offset:1024

	;;#ASMEND
	scratch_load_dword v0, off, off offset:32 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[154:155], v0 offset:0
ds_read_b64_tr_b8 v[156:157], v0 offset:1024

	;;#ASMEND
	scratch_load_dword v230, off, off offset:144 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[226:227], v230 offset:0
ds_read_b64_tr_b8 v[228:229], v230 offset:1024

	;;#ASMEND
	scratch_load_dword v0, off, off offset:104 ; 4-byte Folded Reload
	s_add_i32 s38, s38, s6
	s_ashr_i32 s3, s38, 31
	s_add_u32 s44, s28, s38
	s_addc_u32 s45, s29, s3
	s_mov_b32 s46, s6
	s_mov_b32 s47, s7
	s_mul_i32 s35, s26, 0xc80
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s42, v0
	scratch_load_dword v0, off, off offset:108 ; 4-byte Folded Reload
	s_mov_b32 m0, s42
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s3, v0
	buffer_load_dwordx4 v254, s[44:47], 0 offen lds
	s_mov_b32 m0, s3
	s_nop 0
	buffer_load_dwordx4 v252, s[44:47], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[90:97], v[150:157], v[182:185], v18, v247 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[90:97], v[222:229], v[186:189], v18, v247 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[98:105], v[150:157], v[190:193], v18, v247 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[98:101], v[98:105], v[222:229], v[194:197], v18, v247 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[102:105], v[106:113], v[150:157], v[198:201], v20, v247 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[106:109], v[106:113], v[222:229], v[202:205], v20, v247 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[110:113], v[114:121], v[150:157], v[206:209], v20, v247 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[114:117], v[114:121], v[222:229], v[210:213], v20, v247 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	scratch_load_dword v0, off, off offset:16 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b128 v[182:185], v0 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[190:193], v0 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[198:201], v0 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[206:209], v0 offset:0x1800

	;;#ASMEND
	scratch_load_dword v0, off, off offset:36 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b128 v[186:189], v0 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[194:197], v0 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[202:205], v0 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[210:213], v0 offset:0x1800

	;;#ASMEND
	scratch_load_dword v0, off, off offset:84 ; 4-byte Folded Reload
	s_ashr_i32 s37, s35, 31
	s_add_u32 s35, s4, s35
	s_addc_u32 s37, s5, s37
	s_add_u32 s44, s35, 0x80
	s_addc_u32 s45, s37, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s35, v0
	scratch_load_dword v0, off, off offset:88 ; 4-byte Folded Reload
	s_mov_b32 m0, s35
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s37, v0
	buffer_load_dwordx4 v254, s[44:47], 0 offen lds
	s_mov_b32 m0, s37
	s_nop 0
	buffer_load_dwordx4 v252, s[44:47], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[118:121], v[182:189], v[2:9], v[54:57], v19, v246 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[122:125], v[182:189], v[10:17], v[58:61], v19, v246 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[126:129], v[190:197], v[2:9], v[30:33], v19, v246 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[130:133], v[190:197], v[10:17], v[34:37], v19, v246 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[134:137], v[198:205], v[2:9], v[38:41], v21, v246 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[138:141], v[198:205], v[10:17], v[42:45], v21, v246 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[142:145], v[206:213], v[2:9], v[46:49], v21, v246 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[146:149], v[206:213], v[10:17], v[50:53], v21, v246 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_u32 s44, s8, 0xc80
	s_addc_u32 s45, s9, 0
	s_mov_b32 s46, s10
	s_mov_b32 m0, s31
	s_nop 0
	buffer_load_dwordx4 v248, s[44:47], 0 offen lds
	s_mov_b32 m0, s36
	s_nop 0
	buffer_load_dwordx4 v250, s[44:47], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[182:189], v[150:157], v[22:25], v19, v247 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[182:189], v[222:229], v[26:29], v19, v247 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[190:197], v[150:157], v[158:161], v19, v247 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[190:197], v[222:229], v[162:165], v19, v247 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[198:205], v[150:157], v[166:169], v21, v247 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[198:205], v[222:229], v[170:173], v21, v247 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], v[206:213], v[150:157], v[174:177], v21, v247 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[206:213], v[222:229], v[178:181], v21, v247 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	scratch_load_dword v0, off, off offset:148 ; 4-byte Folded Reload
	scratch_load_dword v246, off, off offset:64 ; 4-byte Folded Reload
	scratch_load_dword v237, off, off offset:60 ; 4-byte Folded Reload
	s_movk_i32 s31, 0x3000
	s_add_u32 s44, s11, 0xc80
	s_addc_u32 s45, s30, 0
	s_waitcnt vmcnt(2)
	buffer_load_dwordx4 v[18:21], v0, s[16:19], s31 offen
	s_waitcnt vmcnt(2)
	buffer_load_dwordx2 v[182:183], v246, s[20:23], s2 offen
	s_waitcnt vmcnt(2)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v237 offset:0
ds_read_b64_tr_b8 v[4:5], v237 offset:1024

	;;#ASMEND
	scratch_load_dword v236, off, off offset:56 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v236 offset:0
ds_read_b64_tr_b8 v[12:13], v236 offset:1024

	;;#ASMEND
	scratch_load_dword v14, off, off offset:24 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v14 offset:0
ds_read_b64_tr_b8 v[8:9], v14 offset:1024

	;;#ASMEND
	scratch_load_dword v54, off, off offset:12 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v54 offset:0
ds_read_b64_tr_b8 v[16:17], v54 offset:1024

	;;#ASMEND
	scratch_load_dword v54, off, off offset:72 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b128 v[150:153], v54 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[158:161], v54 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[166:169], v54 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[174:177], v54 offset:0x1800

	;;#ASMEND
	scratch_load_dword v54, off, off offset:160 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b128 v[154:157], v54 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[162:165], v54 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[170:173], v54 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[178:181], v54 offset:0x1800

	;;#ASMEND
	scratch_load_dword v54, off, off offset:80 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s2, v54
	scratch_load_dword v54, off, off offset:76 ; 4-byte Folded Reload
	s_mov_b32 m0, s2
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s31, v54
	buffer_load_dwordx4 v248, s[44:47], 0 offen lds
	s_mov_b32 m0, s31
	s_nop 0
	buffer_load_dwordx4 v250, s[44:47], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[150:157], v[2:9], v[214:217], v18, v182 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[150:157], v[10:17], v[218:221], v18, v182 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[158:165], v[2:9], v[62:65], v18, v182 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[158:165], v[10:17], v[66:69], v18, v182 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[166:173], v[2:9], v[70:73], v20, v182 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[166:173], v[10:17], v[74:77], v20, v182 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[174:181], v[2:9], v[78:81], v20, v182 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[174:181], v[10:17], v[82:85], v20, v182 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	scratch_load_dword v188, off, off offset:68 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[184:185], v188 offset:0
ds_read_b64_tr_b8 v[186:187], v188 offset:1024

	;;#ASMEND
	scratch_load_dword v188, off, off offset:20 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[192:193], v188 offset:0
ds_read_b64_tr_b8 v[194:195], v188 offset:1024

	;;#ASMEND
	scratch_load_dword v196, off, off offset:8 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[188:189], v196 offset:0
ds_read_b64_tr_b8 v[190:191], v196 offset:1024

	;;#ASMEND
	scratch_load_dword v200, off, off offset:4 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[196:197], v200 offset:0
ds_read_b64_tr_b8 v[198:199], v200 offset:1024

	;;#ASMEND
	scratch_load_dword v200, off, off offset:116 ; 4-byte Folded Reload
	s_add_i32 s46, s38, s6
	s_ashr_i32 s36, s46, 31
	s_add_u32 s48, s28, s46
	s_addc_u32 s49, s29, s36
	s_mov_b32 s50, s6
	s_mul_i32 s38, s26, 0xd00
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s45, v200
	scratch_load_dword v200, off, off offset:112 ; 4-byte Folded Reload
	s_mov_b32 m0, s45
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s36, v200
	buffer_load_dwordx4 v254, s[48:51], 0 offen lds
	s_mov_b32 m0, s36
	s_nop 0
	buffer_load_dwordx4 v252, s[48:51], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[150:157], v[184:191], v[86:89], v18, v183 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[150:157], v[192:199], v[90:93], v18, v183 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[158:165], v[184:191], v[94:97], v18, v183 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[98:101], v[158:165], v[192:199], v[98:101], v18, v183 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[102:105], v[166:173], v[184:191], v[102:105], v20, v183 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[106:109], v[166:173], v[192:199], v[106:109], v20, v183 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[110:113], v[174:181], v[184:191], v[110:113], v20, v183 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[114:117], v[174:181], v[192:199], v[114:117], v20, v183 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	scratch_load_dword v150, off, off offset:164 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b128 v[154:157], v150 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[162:165], v150 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[170:173], v150 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[200:203], v150 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[158:161], v249 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[166:169], v249 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[174:177], v249 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[204:207], v249 offset:0x1800

	;;#ASMEND
	scratch_load_dword v150, off, off offset:132 ; 4-byte Folded Reload
	s_ashr_i32 s39, s38, 31
	s_add_u32 s38, s4, s38
	s_addc_u32 s39, s5, s39
	s_add_u32 s48, s38, 0x80
	s_addc_u32 s49, s39, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s38, v150
	scratch_load_dword v150, off, off offset:128 ; 4-byte Folded Reload
	s_mov_b32 m0, s38
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s39, v150
	buffer_load_dwordx4 v254, s[48:51], 0 offen lds
	s_mov_b32 m0, s39
	s_nop 0
	buffer_load_dwordx4 v252, s[48:51], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[118:121], v[154:161], v[2:9], v[118:121], v19, v182 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[122:125], v[154:161], v[10:17], v[122:125], v19, v182 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[126:129], v[162:169], v[2:9], v[126:129], v19, v182 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[130:133], v[162:169], v[10:17], v[130:133], v19, v182 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[134:137], v[170:177], v[2:9], v[134:137], v21, v182 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[138:141], v[170:177], v[10:17], v[138:141], v21, v182 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[142:145], v[200:207], v[2:9], v[142:145], v21, v182 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[146:149], v[200:207], v[10:17], v[146:149], v21, v182 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	scratch_load_dword v2, off, off offset:92 ; 4-byte Folded Reload
	s_add_u32 s48, s8, 0xd00
	s_addc_u32 s49, s9, 0
	s_mov_b32 s50, s10
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s40, v2
	scratch_load_dword v2, off, off offset:100 ; 4-byte Folded Reload
	s_mov_b32 m0, s40
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s41, v2
	buffer_load_dwordx4 v248, s[48:51], 0 offen lds
	s_mov_b32 m0, s41
	s_nop 0
	buffer_load_dwordx4 v250, s[48:51], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[178:181], v[200:207], v[192:199], v[50:53], v21, v183 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[150:153], v[154:161], v[184:191], v[22:25], v19, v183 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[154:157], v[154:161], v[192:199], v[26:29], v19, v183 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[158:161], v[162:169], v[184:191], v[30:33], v19, v183 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[162:165], v[162:169], v[192:199], v[34:37], v19, v183 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[166:169], v[170:177], v[184:191], v[38:41], v21, v183 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[170:173], v[170:177], v[192:199], v[42:45], v21, v183 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[174:177], v[200:207], v[184:191], v[46:49], v21, v183 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v1 offset:0
ds_read_b64_tr_b8 v[4:5], v1 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v253 offset:0
ds_read_b64_tr_b8 v[12:13], v253 offset:1024

	;;#ASMEND
	scratch_load_dword v1, off, off         ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v1 offset:0
ds_read_b64_tr_b8 v[8:9], v1 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v251 offset:0
ds_read_b64_tr_b8 v[16:17], v251 offset:1024

	;;#ASMEND
	scratch_load_dword v1, off, off offset:152 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b128 v[184:187], v1 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[192:195], v1 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[200:203], v1 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[208:211], v1 offset:0x1800

	;;#ASMEND
	scratch_load_dword v1, off, off offset:40 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b128 v[188:191], v1 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[196:199], v1 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[204:207], v1 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[212:215], v1 offset:0x1800

	;;#ASMEND
	scratch_load_dword v22, off, off offset:124 ; 4-byte Folded Reload
	s_add_u32 s48, s11, 0xd00
	s_addc_u32 s49, s30, 0
	v_mov_b32_e32 v249, v253
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s43, v22
	scratch_load_dword v22, off, off offset:120 ; 4-byte Folded Reload
	s_mov_b32 m0, s43
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s44, v22
	buffer_load_dwordx4 v248, s[48:51], 0 offen lds
	s_mov_b32 m0, s44
	s_nop 0
	buffer_load_dwordx4 v250, s[48:51], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[184:191], v[10:17], v[58:61], v18, v182 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[192:199], v[2:9], v[62:65], v18, v182 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	s_nop 7
	s_nop 2
	scratch_store_dwordx4 off, v[22:25], off offset:188 ; 16-byte Folded Spill
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[192:199], v[10:17], v[66:69], v18, v182 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[200:207], v[2:9], v[70:73], v20, v182 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[200:207], v[10:17], v[74:77], v20, v182 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[232:235], v[184:191], v[2:9], v[54:57], v18, v182 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[238:241], v[208:215], v[2:9], v[78:81], v20, v182 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[242:245], v[208:215], v[10:17], v[82:85], v20, v182 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[216:217], v255 offset:0
ds_read_b64_tr_b8 v[218:219], v255 offset:1024

	;;#ASMEND
	s_add_i32 s46, s46, s6
	s_mov_b32 m0, s42
	;;#ASMSTART
	ds_read_b64_tr_b8 v[224:225], v231 offset:0
ds_read_b64_tr_b8 v[226:227], v231 offset:1024

	;;#ASMEND
	s_ashr_i32 s42, s46, 31
	scratch_load_dword v1, off, off offset:32 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[220:221], v1 offset:0
ds_read_b64_tr_b8 v[222:223], v1 offset:1024

	;;#ASMEND
	s_add_u32 s48, s28, s46
	v_mov_b32_e32 v1, v230
	;;#ASMSTART
	ds_read_b64_tr_b8 v[228:229], v1 offset:0
ds_read_b64_tr_b8 v[230:231], v1 offset:1024

	;;#ASMEND
	s_addc_u32 s49, s29, s42
	s_mov_b32 s50, s6
	buffer_load_dwordx4 v254, s[48:51], 0 offen lds
	s_mov_b32 m0, s3
	s_mul_i32 s3, s26, 0xd80
	buffer_load_dwordx4 v252, s[48:51], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[184:191], v[216:223], v[86:89], v18, v183 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[184:191], v[224:231], v[90:93], v18, v183 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[192:199], v[216:223], v[94:97], v18, v183 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[192:199], v[224:231], v[98:101], v18, v183 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[200:207], v[216:223], v[102:105], v20, v183 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[200:207], v[224:231], v[106:109], v20, v183 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[208:215], v[216:223], v[110:113], v20, v183 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[208:215], v[224:231], v[114:117], v20, v183 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	scratch_load_dword v1, off, off offset:16 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b128 v[184:187], v1 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[192:195], v1 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[200:203], v1 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[208:211], v1 offset:0x1800

	;;#ASMEND
	s_ashr_i32 s42, s3, 31
	scratch_load_dword v1, off, off offset:36 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b128 v[188:191], v1 offset:0

	;;#ASMEND
	s_add_u32 s3, s4, s3
	;;#ASMSTART
	ds_read_b128 v[196:199], v1 offset:0x800

	;;#ASMEND
	s_addc_u32 s42, s5, s42
	;;#ASMSTART
	ds_read_b128 v[204:207], v1 offset:0x1000

	;;#ASMEND
	s_add_u32 s48, s3, 0x80
	;;#ASMSTART
	ds_read_b128 v[212:215], v1 offset:0x1800

	;;#ASMEND
	s_addc_u32 s49, s42, 0
	s_mov_b32 m0, s35
	s_nop 0
	buffer_load_dwordx4 v254, s[48:51], 0 offen lds
	s_mov_b32 m0, s37
	s_nop 0
	buffer_load_dwordx4 v252, s[48:51], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[184:191], v[2:9], v[118:121], v19, v182 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[184:191], v[10:17], v[122:125], v19, v182 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[192:199], v[2:9], v[126:129], v19, v182 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[98:101], v[192:199], v[10:17], v[130:133], v19, v182 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[102:105], v[200:207], v[2:9], v[134:137], v21, v182 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[106:109], v[200:207], v[10:17], v[138:141], v21, v182 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[110:113], v[208:215], v[2:9], v[142:145], v21, v182 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[114:117], v[208:215], v[10:17], v[146:149], v21, v182 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	scratch_load_dword v2, off, off offset:96 ; 4-byte Folded Reload
	s_add_u32 s48, s8, 0xd80
	s_addc_u32 s49, s9, 0
	s_mov_b32 s50, s10
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s37, v2
	scratch_load_dword v2, off, off offset:136 ; 4-byte Folded Reload
	s_mov_b32 m0, s37
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s42, v2
	buffer_load_dwordx4 v248, s[48:51], 0 offen lds
	s_mov_b32 m0, s42
	s_nop 0
	buffer_load_dwordx4 v250, s[48:51], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[118:121], v[184:191], v[216:223], v[150:153], v19, v183 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[122:125], v[184:191], v[224:231], v[154:157], v19, v183 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[126:129], v[192:199], v[216:223], v[158:161], v19, v183 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[130:133], v[192:199], v[224:231], v[162:165], v19, v183 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[134:137], v[200:207], v[216:223], v[166:169], v21, v183 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[138:141], v[200:207], v[224:231], v[170:173], v21, v183 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[142:145], v[208:215], v[216:223], v[174:177], v21, v183 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[146:149], v[208:215], v[224:231], v[178:181], v21, v183 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_movk_i32 s3, 0x3400
	s_barrier
	buffer_load_dwordx4 v[18:21], v0, s[16:19], s3 offen
	s_movk_i32 s3, 0x1a00
	buffer_load_dwordx2 v[246:247], v246, s[20:23], s3 offen
	;;#ASMSTART
	ds_read_b64_tr_b8 v[46:47], v237 offset:0
ds_read_b64_tr_b8 v[48:49], v237 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[22:23], v236 offset:0
ds_read_b64_tr_b8 v[24:25], v236 offset:1024

	;;#ASMEND
	scratch_load_dword v0, off, off offset:24 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[50:51], v0 offset:0
ds_read_b64_tr_b8 v[52:53], v0 offset:1024

	;;#ASMEND
	scratch_load_dword v0, off, off offset:12 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[26:27], v0 offset:0
ds_read_b64_tr_b8 v[28:29], v0 offset:1024

	;;#ASMEND
	scratch_load_dword v0, off, off offset:72 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b128 v[150:153], v0 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[158:161], v0 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[166:169], v0 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[174:177], v0 offset:0x1800

	;;#ASMEND
	scratch_load_dword v251, off, off offset:160 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b128 v[154:157], v251 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[162:165], v251 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[170:173], v251 offset:0x1000

	;;#ASMEND
	s_add_u32 s48, s11, 0xd80
	;;#ASMSTART
	ds_read_b128 v[178:181], v251 offset:0x1800

	;;#ASMEND
	s_addc_u32 s49, s30, 0
	s_mov_b32 m0, s2
	s_nop 0
	buffer_load_dwordx4 v248, s[48:51], 0 offen lds
	s_mov_b32 m0, s31
	s_nop 0
	buffer_load_dwordx4 v250, s[48:51], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	scratch_load_dwordx4 v[0:3], off, off offset:188 ; 16-byte Folded Reload
	v_mfma_scale_f32_16x16x128_f8f6f4 v[214:217], v[150:157], v[46:53], v[232:235], v18, v246 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[222:225], v[158:165], v[46:53], v[30:33], v18, v246 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[226:229], v[158:165], v[22:29], v[34:37], v18, v246 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[230:233], v[166:173], v[46:53], v[38:41], v20, v246 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[234:237], v[166:173], v[22:29], v[42:45], v20, v246 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[238:241], v[174:181], v[46:53], v[238:241], v20, v246 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[242:245], v[174:181], v[22:29], v[242:245], v20, v246 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_waitcnt vmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[218:221], v[150:157], v[22:29], v[0:3], v18, v246 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_nop 4
	scratch_load_dword v0, off, off offset:68 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[8:9], v0 offset:0
ds_read_b64_tr_b8 v[10:11], v0 offset:1024

	;;#ASMEND
	s_add_i32 s2, s46, s6
	scratch_load_dword v4, off, off offset:20 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[0:1], v4 offset:0
ds_read_b64_tr_b8 v[2:3], v4 offset:1024

	;;#ASMEND
	s_ashr_i32 s3, s2, 31
	scratch_load_dword v4, off, off offset:8 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[12:13], v4 offset:0
ds_read_b64_tr_b8 v[14:15], v4 offset:1024

	;;#ASMEND
	s_add_u32 s48, s28, s2
	s_mov_b32 m0, s45
	scratch_load_dword v16, off, off offset:4 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[4:5], v16 offset:0
ds_read_b64_tr_b8 v[6:7], v16 offset:1024

	;;#ASMEND
	s_addc_u32 s49, s29, s3
	s_mov_b32 s50, s6
	buffer_load_dwordx4 v254, s[48:51], 0 offen lds
	s_mov_b32 m0, s36
	s_mul_i32 s3, s26, 0xe00
	buffer_load_dwordx4 v252, s[48:51], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[182:185], v[150:157], v[8:15], v[54:57], v18, v247 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[186:189], v[150:157], v[0:7], v[58:61], v18, v247 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[190:193], v[158:165], v[8:15], v[62:65], v18, v247 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[194:197], v[158:165], v[0:7], v[66:69], v18, v247 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[198:201], v[166:173], v[8:15], v[70:73], v20, v247 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[202:205], v[166:173], v[0:7], v[74:77], v20, v247 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[206:209], v[174:181], v[8:15], v[78:81], v20, v247 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[210:213], v[174:181], v[0:7], v[82:85], v20, v247 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	scratch_load_dword v253, off, off offset:164 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b128 v[62:65], v253 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[70:73], v253 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[78:81], v253 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[150:153], v253 offset:0x1800

	;;#ASMEND
	s_ashr_i32 s31, s3, 31
	scratch_load_dword v16, off, off offset:156 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b128 v[66:69], v16 offset:0

	;;#ASMEND
	s_add_u32 s3, s4, s3
	;;#ASMSTART
	ds_read_b128 v[74:77], v16 offset:0x800

	;;#ASMEND
	s_addc_u32 s31, s5, s31
	;;#ASMSTART
	ds_read_b128 v[82:85], v16 offset:0x1000

	;;#ASMEND
	s_add_u32 s48, s3, 0x80
	;;#ASMSTART
	ds_read_b128 v[154:157], v16 offset:0x1800

	;;#ASMEND
	s_addc_u32 s49, s31, 0
	s_mov_b32 m0, s38
	s_nop 0
	buffer_load_dwordx4 v254, s[48:51], 0 offen lds
	s_mov_b32 m0, s39
	s_nop 0
	buffer_load_dwordx4 v252, s[48:51], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[62:69], v[46:53], v[86:89], v19, v246 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[62:69], v[22:29], v[90:93], v19, v246 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[70:77], v[46:53], v[94:97], v19, v246 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[70:77], v[22:29], v[98:101], v19, v246 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[78:85], v[46:53], v[102:105], v21, v246 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[78:85], v[22:29], v[106:109], v21, v246 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], v[150:157], v[46:53], v[110:113], v21, v246 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[150:157], v[22:29], v[114:117], v21, v246 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_u32 s48, s8, 0xe00
	s_addc_u32 s49, s9, 0
	s_mov_b32 s50, s10
	s_mov_b32 m0, s40
	s_nop 0
	buffer_load_dwordx4 v248, s[48:51], 0 offen lds
	s_mov_b32 m0, s41
	s_nop 0
	buffer_load_dwordx4 v250, s[48:51], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[62:69], v[8:15], v[118:121], v19, v247 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[62:69], v[0:7], v[122:125], v19, v247 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[178:181], v[150:157], v[0:7], v[146:149], v21, v247 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[158:161], v[70:77], v[8:15], v[126:129], v19, v247 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[162:165], v[70:77], v[0:7], v[130:133], v19, v247 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[166:169], v[78:85], v[8:15], v[134:137], v21, v247 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[170:173], v[78:85], v[0:7], v[138:141], v21, v247 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[174:177], v[150:157], v[8:15], v[142:145], v21, v247 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	scratch_load_dword v0, off, off offset:44 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v0 offset:0
ds_read_b64_tr_b8 v[4:5], v0 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v249 offset:0
ds_read_b64_tr_b8 v[12:13], v249 offset:1024

	;;#ASMEND
	scratch_load_dword v1, off, off         ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v1 offset:0
ds_read_b64_tr_b8 v[8:9], v1 offset:1024

	;;#ASMEND
	scratch_load_dword v1, off, off offset:48 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v1 offset:0
ds_read_b64_tr_b8 v[16:17], v1 offset:1024

	;;#ASMEND
	scratch_load_dword v255, off, off offset:152 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b128 v[90:93], v255 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[98:101], v255 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[106:109], v255 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[114:117], v255 offset:0x1800

	;;#ASMEND
	scratch_load_dword v62, off, off offset:40 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b128 v[94:97], v62 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[102:105], v62 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[110:113], v62 offset:0x1000

	;;#ASMEND
	s_add_u32 s48, s11, 0xe00
	;;#ASMSTART
	ds_read_b128 v[118:121], v62 offset:0x1800

	;;#ASMEND
	s_addc_u32 s49, s30, 0
	s_mov_b32 m0, s43
	s_nop 0
	buffer_load_dwordx4 v248, s[48:51], 0 offen lds
	s_mov_b32 m0, s44
	s_nop 0
	buffer_load_dwordx4 v250, s[48:51], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[98:105], v[2:9], v[222:225], v18, v246 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[98:105], v[10:17], v[226:229], v18, v246 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[106:113], v[2:9], v[230:233], v20, v246 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[106:113], v[10:17], v[234:237], v20, v246 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[114:121], v[2:9], v[238:241], v20, v246 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[114:121], v[10:17], v[242:245], v20, v246 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[214:217], v[90:97], v[2:9], v[214:217], v18, v246 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[218:221], v[90:97], v[10:17], v[218:221], v18, v246 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	scratch_load_dword v230, off, off offset:28 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[150:151], v230 offset:0
ds_read_b64_tr_b8 v[152:153], v230 offset:1024

	;;#ASMEND
	s_nop 0
	scratch_load_dword v244, off, off offset:140 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[222:223], v244 offset:0
ds_read_b64_tr_b8 v[224:225], v244 offset:1024

	;;#ASMEND
	scratch_load_dword v232, off, off offset:32 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[154:155], v232 offset:0
ds_read_b64_tr_b8 v[156:157], v232 offset:1024

	;;#ASMEND
	scratch_load_dword v231, off, off offset:144 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[226:227], v231 offset:0
ds_read_b64_tr_b8 v[228:229], v231 offset:1024

	;;#ASMEND
	scratch_load_dword v86, off, off offset:104 ; 4-byte Folded Reload
	s_add_i32 s39, s2, s6
	s_ashr_i32 s2, s39, 31
	s_add_u32 s44, s28, s39
	s_addc_u32 s45, s29, s2
	s_mov_b32 s46, s6
	s_mul_i32 s2, s26, 0xe80
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s38, v86
	scratch_load_dword v86, off, off offset:108 ; 4-byte Folded Reload
	s_mov_b32 m0, s38
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s3, v86
	buffer_load_dwordx4 v254, s[44:47], 0 offen lds
	s_mov_b32 m0, s3
	s_nop 0
	buffer_load_dwordx4 v252, s[44:47], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[90:97], v[150:157], v[182:185], v18, v247 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[90:97], v[222:229], v[186:189], v18, v247 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[98:105], v[150:157], v[190:193], v18, v247 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[98:101], v[98:105], v[222:229], v[194:197], v18, v247 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[102:105], v[106:113], v[150:157], v[198:201], v20, v247 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[106:109], v[106:113], v[222:229], v[202:205], v20, v247 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[110:113], v[114:121], v[150:157], v[206:209], v20, v247 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[114:117], v[114:121], v[222:229], v[210:213], v20, v247 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	scratch_load_dword v18, off, off offset:16 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b128 v[182:185], v18 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[190:193], v18 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[198:201], v18 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[206:209], v18 offset:0x1800

	;;#ASMEND
	scratch_load_dword v249, off, off offset:36 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b128 v[186:189], v249 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[194:197], v249 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[202:205], v249 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[210:213], v249 offset:0x1800

	;;#ASMEND
	scratch_load_dword v18, off, off offset:84 ; 4-byte Folded Reload
	s_ashr_i32 s31, s2, 31
	s_add_u32 s2, s4, s2
	s_addc_u32 s31, s5, s31
	s_add_u32 s44, s2, 0x80
	s_addc_u32 s45, s31, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s35, v18
	scratch_load_dword v18, off, off offset:88 ; 4-byte Folded Reload
	s_mov_b32 m0, s35
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s36, v18
	buffer_load_dwordx4 v254, s[44:47], 0 offen lds
	s_mov_b32 m0, s36
	s_nop 0
	buffer_load_dwordx4 v252, s[44:47], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[118:121], v[182:189], v[2:9], v[54:57], v19, v246 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[122:125], v[182:189], v[10:17], v[58:61], v19, v246 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[126:129], v[190:197], v[2:9], v[30:33], v19, v246 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[130:133], v[190:197], v[10:17], v[34:37], v19, v246 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[134:137], v[198:205], v[2:9], v[38:41], v21, v246 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[138:141], v[198:205], v[10:17], v[42:45], v21, v246 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[142:145], v[206:213], v[2:9], v[46:49], v21, v246 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[146:149], v[206:213], v[10:17], v[50:53], v21, v246 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_u32 s44, s8, 0xe80
	s_addc_u32 s45, s9, 0
	s_mov_b32 s46, s10
	s_mov_b32 m0, s37
	s_nop 0
	buffer_load_dwordx4 v248, s[44:47], 0 offen lds
	s_mov_b32 m0, s42
	s_nop 0
	buffer_load_dwordx4 v250, s[44:47], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[182:189], v[150:157], v[22:25], v19, v247 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[182:189], v[222:229], v[26:29], v19, v247 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[190:197], v[150:157], v[158:161], v19, v247 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[190:197], v[222:229], v[162:165], v19, v247 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[198:205], v[150:157], v[166:169], v21, v247 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[198:205], v[222:229], v[170:173], v21, v247 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], v[206:213], v[150:157], v[174:177], v21, v247 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[206:213], v[222:229], v[178:181], v21, v247 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	scratch_load_dword v240, off, off offset:148 ; 4-byte Folded Reload
	scratch_load_dword v241, off, off offset:64 ; 4-byte Folded Reload
	scratch_load_dword v243, off, off offset:60 ; 4-byte Folded Reload
	s_movk_i32 s2, 0x3800
	s_add_u32 s40, s11, 0xe80
	s_addc_u32 s41, s30, 0
	s_mov_b32 s42, s10
	s_mov_b32 s43, s7
	s_waitcnt vmcnt(2)
	buffer_load_dwordx4 v[18:21], v240, s[16:19], s2 offen
	s_waitcnt vmcnt(2)
	buffer_load_dwordx2 v[182:183], v241, s[20:23], s34 offen
	s_waitcnt vmcnt(2)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v243 offset:0
ds_read_b64_tr_b8 v[4:5], v243 offset:1024

	;;#ASMEND
	scratch_load_dword v242, off, off offset:56 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v242 offset:0
ds_read_b64_tr_b8 v[12:13], v242 offset:1024

	;;#ASMEND
	scratch_load_dword v239, off, off offset:24 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v239 offset:0
ds_read_b64_tr_b8 v[8:9], v239 offset:1024

	;;#ASMEND
	scratch_load_dword v238, off, off offset:12 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v238 offset:0
ds_read_b64_tr_b8 v[16:17], v238 offset:1024

	;;#ASMEND
	scratch_load_dword v224, off, off offset:72 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b128 v[150:153], v224 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[158:161], v224 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[166:169], v224 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[174:177], v224 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[154:157], v251 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[162:165], v251 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[170:173], v251 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[178:181], v251 offset:0x1800

	;;#ASMEND
	scratch_load_dword v54, off, off offset:80 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s2, v54
	scratch_load_dword v54, off, off offset:76 ; 4-byte Folded Reload
	s_mov_b32 m0, s2
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s31, v54
	buffer_load_dwordx4 v248, s[40:43], 0 offen lds
	s_mov_b32 m0, s31
	s_nop 0
	buffer_load_dwordx4 v250, s[40:43], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[150:157], v[2:9], v[214:217], v18, v182 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[150:157], v[10:17], v[218:221], v18, v182 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[158:165], v[2:9], v[62:65], v18, v182 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[158:165], v[10:17], v[66:69], v18, v182 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[166:173], v[2:9], v[70:73], v20, v182 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[166:173], v[10:17], v[74:77], v20, v182 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[174:181], v[2:9], v[78:81], v20, v182 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[174:181], v[10:17], v[82:85], v20, v182 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	scratch_load_dword v237, off, off offset:68 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[184:185], v237 offset:0
ds_read_b64_tr_b8 v[186:187], v237 offset:1024

	;;#ASMEND
	scratch_load_dword v225, off, off offset:20 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[192:193], v225 offset:0
ds_read_b64_tr_b8 v[194:195], v225 offset:1024

	;;#ASMEND
	scratch_load_dword v227, off, off offset:8 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[188:189], v227 offset:0
ds_read_b64_tr_b8 v[190:191], v227 offset:1024

	;;#ASMEND
	scratch_load_dword v226, off, off offset:4 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[196:197], v226 offset:0
ds_read_b64_tr_b8 v[198:199], v226 offset:1024

	;;#ASMEND
	scratch_load_dword v200, off, off offset:116 ; 4-byte Folded Reload
	s_add_i32 s34, s39, s6
	s_ashr_i32 s37, s34, 31
	s_add_u32 s40, s28, s34
	s_addc_u32 s41, s29, s37
	s_mov_b32 s42, s6
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s37, v200
	scratch_load_dword v200, off, off offset:112 ; 4-byte Folded Reload
	s_mov_b32 m0, s37
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s37, v200
	buffer_load_dwordx4 v254, s[40:43], 0 offen lds
	s_mov_b32 m0, s37
	s_mul_i32 s37, s26, 0xf00
	buffer_load_dwordx4 v252, s[40:43], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[150:157], v[184:191], v[86:89], v18, v183 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[150:157], v[192:199], v[90:93], v18, v183 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[158:165], v[184:191], v[94:97], v18, v183 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[98:101], v[158:165], v[192:199], v[98:101], v18, v183 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[102:105], v[166:173], v[184:191], v[102:105], v20, v183 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[106:109], v[166:173], v[192:199], v[106:109], v20, v183 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[110:113], v[174:181], v[184:191], v[110:113], v20, v183 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[114:117], v[174:181], v[192:199], v[114:117], v20, v183 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[154:157], v253 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[162:165], v253 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[170:173], v253 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[200:203], v253 offset:0x1800

	;;#ASMEND
	scratch_load_dword v245, off, off offset:156 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b128 v[158:161], v245 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[166:169], v245 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[174:177], v245 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[204:207], v245 offset:0x1800

	;;#ASMEND
	scratch_load_dword v150, off, off offset:132 ; 4-byte Folded Reload
	s_ashr_i32 s39, s37, 31
	s_add_u32 s37, s4, s37
	s_addc_u32 s39, s5, s39
	s_add_u32 s40, s37, 0x80
	s_addc_u32 s41, s39, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s37, v150
	scratch_load_dword v150, off, off offset:128 ; 4-byte Folded Reload
	s_mov_b32 m0, s37
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s37, v150
	buffer_load_dwordx4 v254, s[40:43], 0 offen lds
	s_mov_b32 m0, s37
	s_nop 0
	buffer_load_dwordx4 v252, s[40:43], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[118:121], v[154:161], v[2:9], v[118:121], v19, v182 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[122:125], v[154:161], v[10:17], v[122:125], v19, v182 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[126:129], v[162:169], v[2:9], v[126:129], v19, v182 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[130:133], v[162:169], v[10:17], v[130:133], v19, v182 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[134:137], v[170:177], v[2:9], v[134:137], v21, v182 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[138:141], v[170:177], v[10:17], v[138:141], v21, v182 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[142:145], v[200:207], v[2:9], v[142:145], v21, v182 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[146:149], v[200:207], v[10:17], v[146:149], v21, v182 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	scratch_load_dword v2, off, off offset:92 ; 4-byte Folded Reload
	s_add_u32 s40, s8, 0xf00
	s_addc_u32 s41, s9, 0
	s_mov_b32 s42, s10
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s37, v2
	scratch_load_dword v2, off, off offset:100 ; 4-byte Folded Reload
	s_mov_b32 m0, s37
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s37, v2
	buffer_load_dwordx4 v248, s[40:43], 0 offen lds
	s_mov_b32 m0, s37
	s_nop 0
	buffer_load_dwordx4 v250, s[40:43], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[178:181], v[200:207], v[192:199], v[50:53], v21, v183 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[150:153], v[154:161], v[184:191], v[22:25], v19, v183 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[154:157], v[154:161], v[192:199], v[26:29], v19, v183 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[158:161], v[162:169], v[184:191], v[30:33], v19, v183 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[162:165], v[162:169], v[192:199], v[34:37], v19, v183 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[166:169], v[170:177], v[184:191], v[38:41], v21, v183 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[170:173], v[170:177], v[192:199], v[42:45], v21, v183 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[174:177], v[200:207], v[184:191], v[46:49], v21, v183 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v0 offset:0
ds_read_b64_tr_b8 v[4:5], v0 offset:1024

	;;#ASMEND
	scratch_load_dword v236, off, off offset:52 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v236 offset:0
ds_read_b64_tr_b8 v[12:13], v236 offset:1024

	;;#ASMEND
	scratch_load_dword v235, off, off       ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v235 offset:0
ds_read_b64_tr_b8 v[8:9], v235 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v1 offset:0
ds_read_b64_tr_b8 v[16:17], v1 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[184:187], v255 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[192:195], v255 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[200:203], v255 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[208:211], v255 offset:0x1800

	;;#ASMEND
	scratch_load_dword v228, off, off offset:40 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b128 v[188:191], v228 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[196:199], v228 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[204:207], v228 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[212:215], v228 offset:0x1800

	;;#ASMEND
	scratch_load_dword v22, off, off offset:124 ; 4-byte Folded Reload
	s_add_u32 s40, s11, 0xf00
	s_addc_u32 s41, s30, 0
	v_mov_b32_e32 v234, v1
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s11, v22
	scratch_load_dword v22, off, off offset:120 ; 4-byte Folded Reload
	s_mov_b32 m0, s11
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s11, v22
	buffer_load_dwordx4 v248, s[40:43], 0 offen lds
	s_mov_b32 m0, s11
	s_nop 0
	buffer_load_dwordx4 v250, s[40:43], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[184:191], v[2:9], v[54:57], v18, v182 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[184:191], v[10:17], v[58:61], v18, v182 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[192:199], v[2:9], v[62:65], v18, v182 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[192:199], v[10:17], v[66:69], v18, v182 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[200:207], v[2:9], v[70:73], v20, v182 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[200:207], v[10:17], v[74:77], v20, v182 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], v[208:215], v[2:9], v[78:81], v20, v182 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[208:215], v[10:17], v[82:85], v20, v182 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[78:79], v230 offset:0
ds_read_b64_tr_b8 v[80:81], v230 offset:1024

	;;#ASMEND
	s_add_i32 s34, s34, s6
	;;#ASMSTART
	ds_read_b64_tr_b8 v[216:217], v244 offset:0
ds_read_b64_tr_b8 v[218:219], v244 offset:1024

	;;#ASMEND
	s_ashr_i32 s11, s34, 31
	;;#ASMSTART
	ds_read_b64_tr_b8 v[82:83], v232 offset:0
ds_read_b64_tr_b8 v[84:85], v232 offset:1024

	;;#ASMEND
	s_add_u32 s40, s28, s34
	s_mov_b32 m0, s38
	;;#ASMSTART
	ds_read_b64_tr_b8 v[220:221], v231 offset:0
ds_read_b64_tr_b8 v[222:223], v231 offset:1024

	;;#ASMEND
	s_addc_u32 s41, s29, s11
	s_mov_b32 s42, s6
	buffer_load_dwordx4 v254, s[40:43], 0 offen lds
	s_mov_b32 m0, s3
	s_mul_i32 s3, s26, 0xf80
	buffer_load_dwordx4 v252, s[40:43], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[184:191], v[78:85], v[86:89], v18, v183 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[184:191], v[216:223], v[90:93], v18, v183 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[192:199], v[78:85], v[94:97], v18, v183 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[98:101], v[192:199], v[216:223], v[98:101], v18, v183 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[102:105], v[200:207], v[78:85], v[102:105], v20, v183 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[106:109], v[200:207], v[216:223], v[106:109], v20, v183 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[110:113], v[208:215], v[78:85], v[110:113], v20, v183 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[114:117], v[208:215], v[216:223], v[114:117], v20, v183 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	scratch_load_dword v1, off, off offset:16 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b128 v[58:61], v1 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[66:69], v1 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[184:187], v1 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[192:195], v1 offset:0x1800

	;;#ASMEND
	s_ashr_i32 s11, s3, 31
	;;#ASMSTART
	ds_read_b128 v[62:65], v249 offset:0

	;;#ASMEND
	s_add_u32 s3, s4, s3
	;;#ASMSTART
	ds_read_b128 v[70:73], v249 offset:0x800

	;;#ASMEND
	s_addc_u32 s5, s5, s11
	;;#ASMSTART
	ds_read_b128 v[188:191], v249 offset:0x1000

	;;#ASMEND
	s_add_u32 s4, s3, 0x80
	;;#ASMSTART
	ds_read_b128 v[196:199], v249 offset:0x1800

	;;#ASMEND
	s_addc_u32 s5, s5, 0
	s_mov_b32 m0, s35
	s_nop 0
	buffer_load_dwordx4 v254, s[4:7], 0 offen lds
	s_mov_b32 m0, s36
	s_nop 0
	buffer_load_dwordx4 v252, s[4:7], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[118:121], v[58:65], v[2:9], v[118:121], v19, v182 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[122:125], v[58:65], v[10:17], v[122:125], v19, v182 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[126:129], v[66:73], v[2:9], v[126:129], v19, v182 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[130:133], v[66:73], v[10:17], v[130:133], v19, v182 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[134:137], v[184:191], v[2:9], v[134:137], v21, v182 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[138:141], v[184:191], v[10:17], v[138:141], v21, v182 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[142:145], v[192:199], v[2:9], v[142:145], v21, v182 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[146:149], v[192:199], v[10:17], v[146:149], v21, v182 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	scratch_load_dword v2, off, off offset:96 ; 4-byte Folded Reload
	s_add_u32 s8, s8, 0xf80
	s_addc_u32 s9, s9, 0
	s_mov_b32 s11, s7
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s3, v2
	scratch_load_dword v2, off, off offset:136 ; 4-byte Folded Reload
	s_mov_b32 m0, s3
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s3, v2
	buffer_load_dwordx4 v248, s[8:11], 0 offen lds
	s_mov_b32 m0, s3
	s_nop 0
	buffer_load_dwordx4 v250, s[8:11], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[58:65], v[78:85], v[150:153], v19, v183 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[58:65], v[216:223], v[154:157], v19, v183 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[66:73], v[78:85], v[158:161], v19, v183 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[66:73], v[216:223], v[162:165], v19, v183 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[184:191], v[78:85], v[166:169], v21, v183 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[184:191], v[216:223], v[170:173], v21, v183 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[192:199], v[78:85], v[174:177], v21, v183 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[192:199], v[216:223], v[178:181], v21, v183 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_movk_i32 s3, 0x3c00
	s_barrier
	buffer_load_dwordx4 v[18:21], v240, s[16:19], s3 offen
	s_movk_i32 s3, 0x1e00
	buffer_load_dwordx2 v[150:151], v241, s[20:23], s3 offen
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v243 offset:0
ds_read_b64_tr_b8 v[4:5], v243 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v242 offset:0
ds_read_b64_tr_b8 v[12:13], v242 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v239 offset:0
ds_read_b64_tr_b8 v[8:9], v239 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v238 offset:0
ds_read_b64_tr_b8 v[16:17], v238 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[152:155], v224 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[160:163], v224 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[168:171], v224 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[176:179], v224 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[156:159], v251 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[164:167], v251 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[172:175], v251 offset:0x1000

	;;#ASMEND
	s_add_u32 s8, s12, 0xf80
	;;#ASMSTART
	ds_read_b128 v[180:183], v251 offset:0x1800

	;;#ASMEND
	s_addc_u32 s9, s13, 0
	s_mov_b32 m0, s2
	s_nop 0
	buffer_load_dwordx4 v248, s[8:11], 0 offen lds
	s_mov_b32 m0, s31
	s_nop 0
	buffer_load_dwordx4 v250, s[8:11], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	s_waitcnt vmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[152:159], v[2:9], v[22:25], v18, v150 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[152:159], v[10:17], v[26:29], v18, v150 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[160:167], v[2:9], v[30:33], v18, v150 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[160:167], v[10:17], v[34:37], v18, v150 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[168:175], v[2:9], v[38:41], v20, v150 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[168:175], v[10:17], v[42:45], v20, v150 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], v[176:183], v[2:9], v[46:49], v20, v150 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[176:183], v[10:17], v[50:53], v20, v150 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[184:185], v237 offset:0
ds_read_b64_tr_b8 v[186:187], v237 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[192:193], v225 offset:0
ds_read_b64_tr_b8 v[194:195], v225 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[188:189], v227 offset:0
ds_read_b64_tr_b8 v[190:191], v227 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[196:197], v226 offset:0
ds_read_b64_tr_b8 v[198:199], v226 offset:1024

	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[152:159], v[184:191], v[86:89], v18, v151 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[152:159], v[192:199], v[90:93], v18, v151 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[160:167], v[184:191], v[94:97], v18, v151 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[98:101], v[160:167], v[192:199], v[98:101], v18, v151 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[102:105], v[168:175], v[184:191], v[102:105], v20, v151 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[106:109], v[168:175], v[192:199], v[106:109], v20, v151 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[110:113], v[176:183], v[184:191], v[110:113], v20, v151 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[114:117], v[176:183], v[192:199], v[114:117], v20, v151 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[152:155], v253 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[160:163], v253 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[168:171], v253 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[176:179], v253 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[156:159], v245 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[164:167], v245 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[172:175], v245 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[180:183], v245 offset:0x1800

	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[118:121], v[152:159], v[2:9], v[118:121], v19, v150 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[122:125], v[152:159], v[10:17], v[122:125], v19, v150 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[126:129], v[160:167], v[2:9], v[126:129], v19, v150 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[130:133], v[160:167], v[10:17], v[130:133], v19, v150 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[134:137], v[168:175], v[2:9], v[134:137], v21, v150 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[138:141], v[168:175], v[10:17], v[138:141], v21, v150 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[142:145], v[176:183], v[2:9], v[142:145], v21, v150 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[146:149], v[176:183], v[10:17], v[146:149], v21, v150 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v0 offset:0
ds_read_b64_tr_b8 v[4:5], v0 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v236 offset:0
ds_read_b64_tr_b8 v[12:13], v236 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v235 offset:0
ds_read_b64_tr_b8 v[8:9], v235 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v234 offset:0
ds_read_b64_tr_b8 v[16:17], v234 offset:1024

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(2)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[200:203], v[152:159], v[184:191], v[54:57], v19, v151 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[204:207], v[152:159], v[192:199], v[58:61], v19, v151 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[208:211], v[160:167], v[184:191], v[62:65], v19, v151 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[212:215], v[160:167], v[192:199], v[66:69], v19, v151 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[216:219], v[168:175], v[184:191], v[70:73], v21, v151 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[220:223], v[168:175], v[192:199], v[74:77], v21, v151 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[224:227], v[176:183], v[184:191], v[78:81], v21, v151 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[192:195], v[176:183], v[192:199], v[82:85], v21, v151 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b128 v[62:65], v255 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[152:155], v255 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[160:163], v255 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[168:171], v255 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[66:69], v228 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[156:159], v228 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[164:167], v228 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[172:175], v228 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[62:69], v[2:9], v[22:25], v18, v150 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[62:69], v[10:17], v[26:29], v18, v150 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[152:159], v[2:9], v[30:33], v18, v150 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[152:159], v[10:17], v[34:37], v18, v150 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[160:167], v[2:9], v[38:41], v20, v150 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[160:167], v[10:17], v[42:45], v20, v150 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[168:175], v[2:9], v[46:49], v20, v150 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[168:175], v[10:17], v[50:53], v20, v150 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b64_tr_b8 v[176:177], v230 offset:0
ds_read_b64_tr_b8 v[178:179], v230 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[184:185], v244 offset:0
ds_read_b64_tr_b8 v[186:187], v244 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[180:181], v232 offset:0
ds_read_b64_tr_b8 v[182:183], v232 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[188:189], v231 offset:0
ds_read_b64_tr_b8 v[190:191], v231 offset:1024

	;;#ASMEND
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[62:69], v[176:183], v[86:89], v18, v151 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[62:69], v[184:191], v[90:93], v18, v151 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[152:159], v[176:183], v[94:97], v18, v151 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[152:159], v[184:191], v[98:101], v18, v151 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[160:167], v[176:183], v[102:105], v20, v151 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[160:167], v[184:191], v[106:109], v20, v151 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[168:175], v[176:183], v[110:113], v20, v151 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], v[168:175], v[184:191], v[114:117], v20, v151 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[86:89], v1 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[152:155], v1 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[160:163], v1 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[168:171], v1 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[90:93], v249 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[156:159], v249 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[164:167], v249 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[172:175], v249 offset:0x1800

	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[106:109], v[86:93], v[2:9], v[118:121], v19, v150 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[86:93], v[10:17], v[122:125], v19, v150 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[118:121], v[152:159], v[2:9], v[126:129], v19, v150 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[102:105], v[152:159], v[10:17], v[130:133], v19, v150 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[126:129], v[160:167], v[2:9], v[134:137], v21, v150 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[114:117], v[160:167], v[10:17], v[138:141], v21, v150 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[130:133], v[168:175], v[2:9], v[142:145], v21, v150 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[122:125], v[168:175], v[10:17], v[146:149], v21, v150 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[110:113], v[86:93], v[176:183], v[200:203], v19, v151 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[98:101], v[86:93], v[184:191], v[204:207], v19, v151 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[152:159], v[176:183], v[208:211], v19, v151 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[160:167], v[176:183], v[216:219], v21, v151 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[6:9], v[168:175], v[176:183], v[224:227], v21, v151 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	scratch_load_dword v180, off, off offset:168 ; 4-byte Folded Reload
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[152:159], v[184:191], v[212:215], v19, v151 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[160:167], v[184:191], v[220:223], v21, v151 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[2:5], v[168:175], v[184:191], v[192:195], v21, v151 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_movk_i32 s2, 0x100
	s_waitcnt vmcnt(0)
	v_cmp_gt_u32_e32 vcc, s2, v180
	s_barrier
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB3_8
; %bb.7:
	s_barrier
.LBB3_8:
	s_or_b64 exec, exec, s[2:3]
	scratch_load_dword v0, off, off offset:172 ; 4-byte Folded Reload
	s_waitcnt lgkmcnt(0)
	s_nop 1
	v_mul_f32_e32 v176, s25, v4
	v_mul_f32_e32 v177, s25, v5
	scratch_load_dwordx2 v[4:5], off, off offset:176 ; 8-byte Folded Reload
	v_mul_f32_e32 v174, s25, v2
	v_lshrrev_b32_e32 v2, 2, v180
	v_and_b32_e32 v2, 12, v2
	v_mul_f32_e32 v175, s25, v3
	v_mul_f32_e32 v172, s25, v8
	v_mul_f32_e32 v173, s25, v9
	v_mul_f32_e32 v168, s25, v12
	v_mul_f32_e32 v169, s25, v13
	v_mul_f32_e32 v18, s25, v54
	v_mul_f32_e32 v19, s25, v55
	v_mul_f32_e32 v164, s25, v16
	v_mul_f32_e32 v165, s25, v17
	v_mul_f32_e32 v166, s25, v10
	v_mul_f32_e32 v167, s25, v11
	v_mul_f32_e32 v170, s25, v6
	v_mul_f32_e32 v171, s25, v7
	v_mul_f32_e32 v20, s25, v56
	v_mul_f32_e32 v21, s25, v57
	v_mul_f32_e32 v162, s25, v14
	v_mul_f32_e32 v163, s25, v15
	v_mul_f32_e32 v22, s25, v22
	v_mul_f32_e32 v23, s25, v23
	v_mul_f32_e32 v24, s25, v24
	v_mul_f32_e32 v25, s25, v25
	v_mul_f32_e32 v56, s25, v60
	v_mul_f32_e32 v57, s25, v61
	v_mul_f32_e32 v60, s25, v28
	v_mul_f32_e32 v61, s25, v29
	v_mul_f32_e32 v140, s25, v32
	v_mul_f32_e32 v141, s25, v33
	v_mul_f32_e32 v148, s25, v36
	v_mul_f32_e32 v149, s25, v37
	v_mul_f32_e32 v136, s25, v40
	v_mul_f32_e32 v137, s25, v41
	v_mul_f32_e32 v144, s25, v44
	v_mul_f32_e32 v145, s25, v45
	v_mul_f32_e32 v160, s25, v48
	v_mul_f32_e32 v161, s25, v49
	v_mul_f32_e32 v156, s25, v52
	v_mul_f32_e32 v157, s25, v53
	v_mul_f32_e32 v54, s25, v58
	v_mul_f32_e32 v55, s25, v59
	v_mul_f32_e32 v58, s25, v26
	v_mul_f32_e32 v59, s25, v27
	v_mul_f32_e32 v138, s25, v30
	v_mul_f32_e32 v139, s25, v31
	v_mul_f32_e32 v146, s25, v34
	v_mul_f32_e32 v147, s25, v35
	v_mul_f32_e32 v134, s25, v38
	v_mul_f32_e32 v135, s25, v39
	v_mul_f32_e32 v142, s25, v42
	v_mul_f32_e32 v143, s25, v43
	v_mul_f32_e32 v82, s25, v82
	v_mul_f32_e32 v150, s25, v62
	v_mul_f32_e32 v151, s25, v63
	v_mul_f32_e32 v152, s25, v64
	v_mul_f32_e32 v153, s25, v65
	v_mul_f32_e32 v154, s25, v50
	v_mul_f32_e32 v155, s25, v51
	v_mul_f32_e32 v158, s25, v46
	v_mul_f32_e32 v159, s25, v47
	v_mul_f32_e32 v83, s25, v83
	v_mul_f32_e32 v84, s25, v84
	v_mul_f32_e32 v85, s25, v85
	v_mul_f32_e32 v78, s25, v78
	v_mul_f32_e32 v79, s25, v79
	v_mul_f32_e32 v80, s25, v80
	v_mul_f32_e32 v81, s25, v81
	v_mul_f32_e32 v74, s25, v74
	v_mul_f32_e32 v75, s25, v75
	v_mul_f32_e32 v76, s25, v76
	v_mul_f32_e32 v77, s25, v77
	v_mul_f32_e32 v70, s25, v70
	v_mul_f32_e32 v71, s25, v71
	v_mul_f32_e32 v72, s25, v72
	v_mul_f32_e32 v73, s25, v73
	s_waitcnt vmcnt(1)
	v_lshl_or_b32 v178, s27, 2, v0
	scratch_load_dword v0, off, off offset:184 ; 4-byte Folded Reload
	v_mul_f32_e32 v66, s25, v66
	s_waitcnt vmcnt(1)
	v_mad_u64_u32 v[2:3], s[2:3], v2, s24, v[4:5]
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshlrev_b64 v[4:5], 1, v[2:3]
	v_add_u32_e32 v2, s24, v2
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshlrev_b64 v[8:9], 1, v[2:3]
	v_add_u32_e32 v2, s24, v2
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshlrev_b64 v[12:13], 1, v[2:3]
	v_add_u32_e32 v2, s24, v2
	v_ashrrev_i32_e32 v3, 31, v2
	s_mul_i32 s2, s24, 13
	v_lshlrev_b64 v[16:17], 1, v[2:3]
	v_add_u32_e32 v2, s2, v2
	v_ashrrev_i32_e32 v3, 31, v2
	v_mul_f32_e32 v67, s25, v67
	v_mul_f32_e32 v68, s25, v68
	v_mul_f32_e32 v69, s25, v69
	v_mul_f32_e32 v106, s25, v106
	v_mul_f32_e32 v107, s25, v107
	v_mul_f32_e32 v108, s25, v108
	v_mul_f32_e32 v109, s25, v109
	v_mul_f32_e32 v118, s25, v118
	v_mul_f32_e32 v119, s25, v119
	v_mul_f32_e32 v120, s25, v120
	v_mul_f32_e32 v121, s25, v121
	v_mul_f32_e32 v126, s25, v126
	v_mul_f32_e32 v127, s25, v127
	v_mul_f32_e32 v128, s25, v128
	v_mul_f32_e32 v129, s25, v129
	v_mul_f32_e32 v130, s25, v130
	v_mul_f32_e32 v131, s25, v131
	v_mul_f32_e32 v132, s25, v132
	v_mul_f32_e32 v133, s25, v133
	v_mul_f32_e32 v110, s25, v110
	v_mul_f32_e32 v98, s25, v98
	v_mul_f32_e32 v94, s25, v94
	v_mul_f32_e32 v95, s25, v95
	v_mul_f32_e32 v96, s25, v96
	v_mul_f32_e32 v97, s25, v97
	v_mul_f32_e32 v102, s25, v102
	v_mul_f32_e32 v103, s25, v103
	v_mul_f32_e32 v104, s25, v104
	v_mul_f32_e32 v105, s25, v105
	v_mul_f32_e32 v114, s25, v114
	v_mul_f32_e32 v115, s25, v115
	v_mul_f32_e32 v116, s25, v116
	v_mul_f32_e32 v117, s25, v117
	v_mul_f32_e32 v122, s25, v122
	v_mul_f32_e32 v123, s25, v123
	v_mul_f32_e32 v124, s25, v124
	v_mul_f32_e32 v125, s25, v125
	v_mul_f32_e32 v111, s25, v111
	v_mul_f32_e32 v112, s25, v112
	v_mul_f32_e32 v113, s25, v113
	v_mul_f32_e32 v99, s25, v99
	v_mul_f32_e32 v100, s25, v100
	v_mul_f32_e32 v101, s25, v101
	v_mul_f32_e32 v90, s25, v90
	v_mul_f32_e32 v91, s25, v91
	v_mul_f32_e32 v92, s25, v92
	v_mul_f32_e32 v93, s25, v93
	v_mul_f32_e32 v86, s25, v86
	v_mul_f32_e32 v87, s25, v87
	v_mul_f32_e32 v88, s25, v88
	v_mul_f32_e32 v89, s25, v89
	s_waitcnt vmcnt(0)
	v_or_b32_e32 v179, s33, v0
	v_mul_lo_u32 v0, v178, s24
	v_lshl_add_u32 v0, v0, 6, v179
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshl_add_u64 v[0:1], v[0:1], 1, s[14:15]
	v_lshl_add_u64 v[6:7], v[0:1], 0, v[4:5]
	v_lshl_add_u64 v[10:11], v[0:1], 0, v[8:9]
	global_store_short_d16_hi v[6:7], v18, off
	global_store_short_d16_hi v[10:11], v19, off
	v_lshl_add_u64 v[14:15], v[0:1], 0, v[12:13]
	v_lshl_add_u64 v[18:19], v[0:1], 0, v[16:17]
	global_store_short_d16_hi v[14:15], v20, off
	global_store_short_d16_hi v[18:19], v21, off
	global_store_short_d16_hi v[6:7], v22, off offset:32
	global_store_short_d16_hi v[10:11], v23, off offset:32
	global_store_short_d16_hi v[14:15], v24, off offset:32
	global_store_short_d16_hi v[18:19], v25, off offset:32
	v_lshlrev_b64 v[20:21], 1, v[2:3]
	v_add_u32_e32 v2, s24, v2
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshlrev_b64 v[24:25], 1, v[2:3]
	v_add_u32_e32 v2, s24, v2
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshlrev_b64 v[28:29], 1, v[2:3]
	v_add_u32_e32 v2, s24, v2
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshlrev_b64 v[32:33], 1, v[2:3]
	v_add_u32_e32 v2, s2, v2
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshlrev_b64 v[36:37], 1, v[2:3]
	v_add_u32_e32 v2, s24, v2
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshlrev_b64 v[40:41], 1, v[2:3]
	v_add_u32_e32 v2, s24, v2
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshlrev_b64 v[44:45], 1, v[2:3]
	v_add_u32_e32 v2, s24, v2
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshlrev_b64 v[48:49], 1, v[2:3]
	v_add_u32_e32 v2, s2, v2
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshlrev_b64 v[52:53], 1, v[2:3]
	v_add_u32_e32 v2, s24, v2
	v_lshl_add_u64 v[22:23], v[0:1], 0, v[20:21]
	v_lshl_add_u64 v[26:27], v[0:1], 0, v[24:25]
	v_lshl_add_u64 v[30:31], v[0:1], 0, v[28:29]
	v_lshl_add_u64 v[34:35], v[0:1], 0, v[32:33]
	v_ashrrev_i32_e32 v3, 31, v2
	global_store_short_d16_hi v[22:23], v54, off
	global_store_short_d16_hi v[26:27], v55, off
	global_store_short_d16_hi v[30:31], v56, off
	global_store_short_d16_hi v[34:35], v57, off
	global_store_short_d16_hi v[22:23], v58, off offset:32
	global_store_short_d16_hi v[26:27], v59, off offset:32
	global_store_short_d16_hi v[30:31], v60, off offset:32
	global_store_short_d16_hi v[34:35], v61, off offset:32
	v_lshlrev_b64 v[56:57], 1, v[2:3]
	v_add_u32_e32 v2, s24, v2
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshlrev_b64 v[60:61], 1, v[2:3]
	v_add_u32_e32 v2, s24, v2
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshlrev_b64 v[2:3], 1, v[2:3]
	v_lshl_add_u64 v[38:39], v[0:1], 0, v[36:37]
	v_lshl_add_u64 v[42:43], v[0:1], 0, v[40:41]
	v_lshl_add_u64 v[46:47], v[0:1], 0, v[44:45]
	v_lshl_add_u64 v[50:51], v[0:1], 0, v[48:49]
	v_lshl_add_u64 v[54:55], v[0:1], 0, v[52:53]
	v_lshl_add_u64 v[58:59], v[0:1], 0, v[56:57]
	v_lshl_add_u64 v[62:63], v[0:1], 0, v[60:61]
	v_lshl_add_u64 v[64:65], v[0:1], 0, v[2:3]
	v_lshl_add_u64 v[0:1], v[0:1], 0, s[0:1]
	global_store_short_d16_hi v[38:39], v134, off
	global_store_short_d16_hi v[42:43], v135, off
	global_store_short_d16_hi v[46:47], v136, off
	global_store_short_d16_hi v[50:51], v137, off
	global_store_short_d16_hi v[38:39], v138, off offset:32
	global_store_short_d16_hi v[42:43], v139, off offset:32
	global_store_short_d16_hi v[46:47], v140, off offset:32
	global_store_short_d16_hi v[50:51], v141, off offset:32
	global_store_short_d16_hi v[54:55], v142, off
	global_store_short_d16_hi v[58:59], v143, off
	global_store_short_d16_hi v[62:63], v144, off
	global_store_short_d16_hi v[64:65], v145, off
	global_store_short_d16_hi v[54:55], v146, off offset:32
	global_store_short_d16_hi v[58:59], v147, off offset:32
	global_store_short_d16_hi v[62:63], v148, off offset:32
	global_store_short_d16_hi v[64:65], v149, off offset:32
	global_store_short_d16_hi v[6:7], v82, off offset:256
	global_store_short_d16_hi v[10:11], v83, off offset:256
	global_store_short_d16_hi v[14:15], v84, off offset:256
	global_store_short_d16_hi v[18:19], v85, off offset:256
	v_lshl_add_u64 v[6:7], v[0:1], 0, v[4:5]
	global_store_short_d16_hi v[6:7], v78, off offset:32
	v_lshl_add_u64 v[6:7], v[0:1], 0, v[8:9]
	global_store_short_d16_hi v[6:7], v79, off offset:32
	v_lshl_add_u64 v[6:7], v[0:1], 0, v[12:13]
	global_store_short_d16_hi v[6:7], v80, off offset:32
	v_lshl_add_u64 v[6:7], v[0:1], 0, v[16:17]
	global_store_short_d16_hi v[6:7], v81, off offset:32
	global_store_short_d16_hi v[22:23], v74, off offset:256
	global_store_short_d16_hi v[26:27], v75, off offset:256
	global_store_short_d16_hi v[30:31], v76, off offset:256
	global_store_short_d16_hi v[34:35], v77, off offset:256
	v_lshl_add_u64 v[6:7], v[0:1], 0, v[20:21]
	global_store_short_d16_hi v[6:7], v70, off offset:32
	v_lshl_add_u64 v[6:7], v[0:1], 0, v[24:25]
	global_store_short_d16_hi v[6:7], v71, off offset:32
	v_lshl_add_u64 v[6:7], v[0:1], 0, v[28:29]
	global_store_short_d16_hi v[6:7], v72, off offset:32
	v_lshl_add_u64 v[6:7], v[0:1], 0, v[32:33]
	global_store_short_d16_hi v[6:7], v73, off offset:32
	global_store_short_d16_hi v[38:39], v66, off offset:256
	global_store_short_d16_hi v[42:43], v67, off offset:256
	global_store_short_d16_hi v[46:47], v68, off offset:256
	global_store_short_d16_hi v[50:51], v69, off offset:256
	v_lshl_add_u64 v[6:7], v[0:1], 0, v[36:37]
	global_store_short_d16_hi v[6:7], v150, off offset:32
	v_lshl_add_u64 v[6:7], v[0:1], 0, v[40:41]
	global_store_short_d16_hi v[6:7], v151, off offset:32
	v_lshl_add_u64 v[6:7], v[0:1], 0, v[44:45]
	global_store_short_d16_hi v[6:7], v152, off offset:32
	v_lshl_add_u64 v[6:7], v[0:1], 0, v[48:49]
	global_store_short_d16_hi v[6:7], v153, off offset:32
	global_store_short_d16_hi v[54:55], v154, off offset:256
	global_store_short_d16_hi v[58:59], v155, off offset:256
	global_store_short_d16_hi v[62:63], v156, off offset:256
	global_store_short_d16_hi v[64:65], v157, off offset:256
	v_lshl_add_u64 v[6:7], v[0:1], 0, v[52:53]
	global_store_short_d16_hi v[6:7], v158, off offset:32
	v_lshl_add_u64 v[6:7], v[0:1], 0, v[56:57]
	global_store_short_d16_hi v[6:7], v159, off offset:32
	v_lshl_add_u64 v[6:7], v[0:1], 0, v[60:61]
	v_lshl_add_u64 v[0:1], v[0:1], 0, v[2:3]
	global_store_short_d16_hi v[6:7], v160, off offset:32
	global_store_short_d16_hi v[0:1], v161, off offset:32
	v_or_b32_e32 v0, 2, v178
	v_mul_lo_u32 v0, v0, s24
	v_lshl_add_u32 v0, v0, 6, v179
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshl_add_u64 v[0:1], v[0:1], 1, s[14:15]
	v_lshl_add_u64 v[6:7], v[0:1], 0, v[4:5]
	v_lshl_add_u64 v[10:11], v[0:1], 0, v[8:9]
	v_lshl_add_u64 v[14:15], v[0:1], 0, v[12:13]
	v_lshl_add_u64 v[18:19], v[0:1], 0, v[16:17]
	v_lshl_add_u64 v[22:23], v[0:1], 0, v[20:21]
	v_lshl_add_u64 v[26:27], v[0:1], 0, v[24:25]
	v_lshl_add_u64 v[30:31], v[0:1], 0, v[28:29]
	v_lshl_add_u64 v[34:35], v[0:1], 0, v[32:33]
	v_lshl_add_u64 v[38:39], v[0:1], 0, v[36:37]
	v_lshl_add_u64 v[42:43], v[0:1], 0, v[40:41]
	v_lshl_add_u64 v[46:47], v[0:1], 0, v[44:45]
	v_lshl_add_u64 v[50:51], v[0:1], 0, v[48:49]
	v_lshl_add_u64 v[54:55], v[0:1], 0, v[52:53]
	v_lshl_add_u64 v[58:59], v[0:1], 0, v[56:57]
	v_lshl_add_u64 v[62:63], v[0:1], 0, v[60:61]
	v_lshl_add_u64 v[64:65], v[0:1], 0, v[2:3]
	v_lshl_add_u64 v[0:1], v[0:1], 0, s[0:1]
	v_lshl_add_u64 v[4:5], v[0:1], 0, v[4:5]
	global_store_short_d16_hi v[6:7], v106, off
	global_store_short_d16_hi v[10:11], v107, off
	global_store_short_d16_hi v[14:15], v108, off
	global_store_short_d16_hi v[18:19], v109, off
	global_store_short_d16_hi v[6:7], v94, off offset:32
	global_store_short_d16_hi v[10:11], v95, off offset:32
	global_store_short_d16_hi v[14:15], v96, off offset:32
	global_store_short_d16_hi v[18:19], v97, off offset:32
	global_store_short_d16_hi v[22:23], v118, off
	global_store_short_d16_hi v[26:27], v119, off
	global_store_short_d16_hi v[30:31], v120, off
	global_store_short_d16_hi v[34:35], v121, off
	global_store_short_d16_hi v[22:23], v102, off offset:32
	global_store_short_d16_hi v[26:27], v103, off offset:32
	global_store_short_d16_hi v[30:31], v104, off offset:32
	global_store_short_d16_hi v[34:35], v105, off offset:32
	global_store_short_d16_hi v[38:39], v126, off
	global_store_short_d16_hi v[42:43], v127, off
	global_store_short_d16_hi v[46:47], v128, off
	global_store_short_d16_hi v[50:51], v129, off
	global_store_short_d16_hi v[38:39], v114, off offset:32
	global_store_short_d16_hi v[42:43], v115, off offset:32
	global_store_short_d16_hi v[46:47], v116, off offset:32
	global_store_short_d16_hi v[50:51], v117, off offset:32
	global_store_short_d16_hi v[54:55], v130, off
	global_store_short_d16_hi v[58:59], v131, off
	global_store_short_d16_hi v[62:63], v132, off
	global_store_short_d16_hi v[64:65], v133, off
	global_store_short_d16_hi v[54:55], v122, off offset:32
	global_store_short_d16_hi v[58:59], v123, off offset:32
	global_store_short_d16_hi v[62:63], v124, off offset:32
	global_store_short_d16_hi v[64:65], v125, off offset:32
	global_store_short_d16_hi v[6:7], v110, off offset:256
	global_store_short_d16_hi v[10:11], v111, off offset:256
	global_store_short_d16_hi v[14:15], v112, off offset:256
	global_store_short_d16_hi v[18:19], v113, off offset:256
	global_store_short_d16_hi v[4:5], v98, off offset:32
	v_lshl_add_u64 v[4:5], v[0:1], 0, v[8:9]
	global_store_short_d16_hi v[4:5], v99, off offset:32
	v_lshl_add_u64 v[4:5], v[0:1], 0, v[12:13]
	global_store_short_d16_hi v[4:5], v100, off offset:32
	v_lshl_add_u64 v[4:5], v[0:1], 0, v[16:17]
	global_store_short_d16_hi v[4:5], v101, off offset:32
	global_store_short_d16_hi v[22:23], v90, off offset:256
	global_store_short_d16_hi v[26:27], v91, off offset:256
	global_store_short_d16_hi v[30:31], v92, off offset:256
	global_store_short_d16_hi v[34:35], v93, off offset:256
	v_lshl_add_u64 v[4:5], v[0:1], 0, v[20:21]
	global_store_short_d16_hi v[4:5], v86, off offset:32
	v_lshl_add_u64 v[4:5], v[0:1], 0, v[24:25]
	global_store_short_d16_hi v[4:5], v87, off offset:32
	v_lshl_add_u64 v[4:5], v[0:1], 0, v[28:29]
	global_store_short_d16_hi v[4:5], v88, off offset:32
	v_lshl_add_u64 v[4:5], v[0:1], 0, v[32:33]
	global_store_short_d16_hi v[4:5], v89, off offset:32
	global_store_short_d16_hi v[38:39], v162, off offset:256
	global_store_short_d16_hi v[42:43], v163, off offset:256
	global_store_short_d16_hi v[46:47], v164, off offset:256
	global_store_short_d16_hi v[50:51], v165, off offset:256
	v_lshl_add_u64 v[4:5], v[0:1], 0, v[36:37]
	global_store_short_d16_hi v[4:5], v166, off offset:32
	v_lshl_add_u64 v[4:5], v[0:1], 0, v[40:41]
	global_store_short_d16_hi v[4:5], v167, off offset:32
	v_lshl_add_u64 v[4:5], v[0:1], 0, v[44:45]
	global_store_short_d16_hi v[4:5], v168, off offset:32
	v_lshl_add_u64 v[4:5], v[0:1], 0, v[48:49]
	global_store_short_d16_hi v[4:5], v169, off offset:32
	global_store_short_d16_hi v[54:55], v170, off offset:256
	global_store_short_d16_hi v[58:59], v171, off offset:256
	global_store_short_d16_hi v[62:63], v172, off offset:256
	global_store_short_d16_hi v[64:65], v173, off offset:256
	v_lshl_add_u64 v[4:5], v[0:1], 0, v[52:53]
	global_store_short_d16_hi v[4:5], v174, off offset:32
	v_lshl_add_u64 v[4:5], v[0:1], 0, v[56:57]
	global_store_short_d16_hi v[4:5], v175, off offset:32
	v_lshl_add_u64 v[4:5], v[0:1], 0, v[60:61]
	v_lshl_add_u64 v[0:1], v[0:1], 0, v[2:3]
	global_store_short_d16_hi v[4:5], v176, off offset:32
	global_store_short_d16_hi v[0:1], v177, off offset:32
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z29rrr_exact_8wave_scaled_kernelILb1ELi2EEv14layout_globals
		.amdhsa_group_segment_fixed_size 135168
		.amdhsa_private_segment_fixed_size 208
		.amdhsa_kernarg_size 552
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
		.amdhsa_next_free_vgpr 256
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
	.section	.text._Z29rrr_exact_8wave_scaled_kernelILb1ELi2EEv14layout_globals,"axG",@progbits,_Z29rrr_exact_8wave_scaled_kernelILb1ELi2EEv14layout_globals,comdat
.Lfunc_end3:
	.size	_Z29rrr_exact_8wave_scaled_kernelILb1ELi2EEv14layout_globals, .Lfunc_end3-_Z29rrr_exact_8wave_scaled_kernelILb1ELi2EEv14layout_globals
                                        ; -- End function
	.set _Z29rrr_exact_8wave_scaled_kernelILb1ELi2EEv14layout_globals.num_vgpr, 256
	.set _Z29rrr_exact_8wave_scaled_kernelILb1ELi2EEv14layout_globals.num_agpr, 0
	.set _Z29rrr_exact_8wave_scaled_kernelILb1ELi2EEv14layout_globals.numbered_sgpr, 56
	.set _Z29rrr_exact_8wave_scaled_kernelILb1ELi2EEv14layout_globals.private_seg_size, 208
	.set _Z29rrr_exact_8wave_scaled_kernelILb1ELi2EEv14layout_globals.uses_vcc, 1
	.set _Z29rrr_exact_8wave_scaled_kernelILb1ELi2EEv14layout_globals.uses_flat_scratch, 0
	.set _Z29rrr_exact_8wave_scaled_kernelILb1ELi2EEv14layout_globals.has_dyn_sized_stack, 0
	.set _Z29rrr_exact_8wave_scaled_kernelILb1ELi2EEv14layout_globals.has_recursion, 0
	.set _Z29rrr_exact_8wave_scaled_kernelILb1ELi2EEv14layout_globals.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 53264
; TotalNumSgprs: 62
; NumVgprs: 256
; NumAgprs: 0
; TotalNumVgprs: 256
; ScratchSize: 208
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 135168 bytes/workgroup (compile time only)
; SGPRBlocks: 12
; VGPRBlocks: 31
; NumSGPRsForWavesPerEU: 102
; NumVGPRsForWavesPerEU: 256
; AccumOffset: 256
; Occupancy: 2
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
	.section	.text._Z29crr_exact_8wave_scaled_kernelILb1ELi2EEv14layout_globals,"axG",@progbits,_Z29crr_exact_8wave_scaled_kernelILb1ELi2EEv14layout_globals,comdat
