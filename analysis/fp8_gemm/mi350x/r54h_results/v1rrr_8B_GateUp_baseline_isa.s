_Z29rrr_exact_8wave_scaled_kernelILb1ELi1EEv14layout_globals: ; @_Z29rrr_exact_8wave_scaled_kernelILb1ELi1EEv14layout_globals
; %bb.0:
	s_ashr_i32 s3, s2, 31
	s_lshr_b32 s3, s3, 29
	s_load_dwordx2 s[22:23], s[0:1], 0x0
	s_load_dwordx2 s[6:7], s[0:1], 0x20
	s_load_dwordx2 s[4:5], s[0:1], 0x30
	s_load_dwordx2 s[26:27], s[0:1], 0x50
	s_load_dwordx2 s[20:21], s[0:1], 0x80
	s_load_dwordx2 s[28:29], s[0:1], 0xb0
	s_add_i32 s3, s2, s3
	s_waitcnt lgkmcnt(0)
	s_ashr_i32 s7, s3, 3
	s_mulk_i32 s2, 0x70
	s_mul_i32 s8, s7, 0xfffffc81
	s_add_i32 s8, s8, s2
	s_mul_hi_i32 s2, s8, 0x92492493
	s_add_i32 s2, s2, s8
	s_lshr_b32 s7, s2, 31
	s_ashr_i32 s9, s2, 7
	s_add_i32 s9, s9, s7
	s_lshl_b32 s2, s9, 2
	s_sub_i32 s7, 16, s2
	s_cmpk_gt_i32 s8, 0x37f
	s_cselect_b32 s7, s7, 4
	s_movk_i32 s3, 0x70
	s_mov_b32 s21, 16
	s_cmp_lt_i32 s7, 1
	s_mov_b32 s10, 56
	s_cbranch_scc1 .LBB20_2
; %bb.1:
	s_abs_i32 s10, s7
	v_cvt_f32_u32_e32 v1, s10
	s_mulk_i32 s9, 0xe0
	s_sub_i32 s8, s8, s9
	s_sub_i32 s9, 0, s10
	v_rcp_iflag_f32_e32 v1, v1
	s_abs_i32 s12, s8
	s_xor_b32 s11, s8, s7
	s_ashr_i32 s11, s11, 31
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	s_nop 0
	v_readfirstlane_b32 s13, v1
	s_mul_i32 s9, s9, s13
	s_mul_hi_u32 s9, s13, s9
	s_add_i32 s13, s13, s9
	s_mul_hi_u32 s9, s12, s13
	s_mul_i32 s13, s9, s10
	s_sub_i32 s12, s12, s13
	s_add_i32 s14, s9, 1
	s_sub_i32 s13, s12, s10
	s_cmp_ge_u32 s12, s10
	s_cselect_b32 s9, s14, s9
	s_cselect_b32 s12, s13, s12
	s_add_i32 s13, s9, 1
	s_cmp_ge_u32 s12, s10
	s_cselect_b32 s9, s13, s9
	s_xor_b32 s9, s9, s11
	s_sub_i32 s10, s9, s11
	s_mul_i32 s7, s10, s7
	s_sub_i32 s7, s8, s7
	s_add_i32 s21, s7, s2
.LBB20_2:
	v_lshlrev_b32_e32 v1, 4, v0
	v_lshrrev_b32_e32 v3, 3, v0
	v_bitop3_b32 v2, v1, s3, v0 bitop3:0x48
	v_or_b32_e32 v4, 64, v3
	v_mad_u64_u32 v[148:149], s[8:9], v3, s6, v[2:3]
	v_mad_u64_u32 v[146:147], s[8:9], v4, s6, v[2:3]
	v_lshlrev_b32_e32 v2, 1, v0
	v_bitop3_b32 v2, v2, s3, v1 bitop3:0x48
	s_lshl_b32 s31, s10, 8
	v_mad_u64_u32 v[150:151], s[2:3], v3, s26, v[2:3]
	v_mad_u64_u32 v[152:153], s[2:3], v4, s26, v[2:3]
	s_lshl_b32 s33, s21, 8
	s_ashr_i32 s2, s31, 31
	v_and_b32_e32 v2, 0x1c00, v1
	v_and_b32_e32 v3, 0x180, v0
	s_add_u32 s12, s4, s31
	v_or_b32_e32 v254, v2, v3
	v_or_b32_e32 v4, 0x2200, v3
	s_addc_u32 s13, s5, s2
	s_lshl_b32 s18, s26, 7
	s_mov_b32 s7, 0x110000
	v_readfirstlane_b32 s2, v254
	v_or_b32_e32 v255, v2, v4
	s_mov_b32 s14, s18
	s_mov_b32 s15, s7
	s_mov_b32 m0, s2
	v_readfirstlane_b32 s2, v255
	s_mul_i32 s27, s33, s6
	buffer_load_dwordx4 v150, s[12:15], 0 offen lds
	s_mov_b32 m0, s2
	s_ashr_i32 s2, s27, 31
	s_add_u32 s4, s22, s27
	v_add_u32_e32 v251, 0x11000, v2
	s_addc_u32 s5, s23, s2
	v_readfirstlane_b32 s2, v251
	v_add_u32_e32 v162, 0x2000, v251
	v_add_u32_e32 v1, 0x4400, v2
	buffer_load_dwordx4 v152, s[12:15], 0 offen lds
	s_lshl_b32 s6, s6, 7
	s_mov_b32 m0, s2
	v_readfirstlane_b32 s2, v162
	v_or_b32_e32 v196, v1, v3
	buffer_load_dwordx4 v148, s[4:7], 0 offen lds
	s_mov_b32 m0, s2
	s_add_u32 s16, s12, 0x80
	v_readfirstlane_b32 s2, v196
	v_add_u32_e32 v197, v1, v4
	buffer_load_dwordx4 v146, s[4:7], 0 offen lds
	s_addc_u32 s17, s13, 0
	s_mov_b32 s19, s7
	s_mov_b32 m0, s2
	v_readfirstlane_b32 s2, v197
	s_add_i32 s29, s27, s6
	buffer_load_dwordx4 v150, s[16:19], 0 offen lds
	s_mov_b32 m0, s2
	s_ashr_i32 s2, s29, 31
	s_add_u32 s8, s22, s29
	v_or_b32_e32 v198, 0x4000, v251
	s_addc_u32 s9, s23, s2
	v_readfirstlane_b32 s2, v198
	v_add_u32_e32 v199, 0x6000, v251
	buffer_load_dwordx4 v152, s[16:19], 0 offen lds
	s_mov_b32 s10, s6
	s_mov_b32 s11, s7
	s_mov_b32 m0, s2
	v_readfirstlane_b32 s2, v199
	buffer_load_dwordx4 v148, s[8:11], 0 offen lds
	s_mov_b32 m0, s2
	v_lshrrev_b32_e32 v11, 8, v0
	buffer_load_dwordx4 v146, s[8:11], 0 offen lds
	s_load_dwordx2 s[10:11], s[0:1], 0xe0
	s_load_dwordx2 s[14:15], s[0:1], 0x60
	s_load_dwordx2 s[2:3], s[0:1], 0xc0
	s_load_dwordx2 s[24:25], s[0:1], 0x90
	s_mov_b32 s30, 0
	v_cmp_eq_u32_e32 vcc, 1, v11
	s_and_saveexec_b64 s[16:17], vcc
	s_cbranch_execz .LBB20_4
; %bb.3:
	s_barrier
.LBB20_4:
	s_or_b64 exec, exec, s[16:17]
	v_lshl_or_b32 v1, v11, 6, s33
	v_add_u32_e32 v5, 0x80, v1
	v_ashrrev_i32_e32 v6, 5, v1
	v_add_u32_e32 v1, 0xa0, v1
	v_ashrrev_i32_e32 v1, 5, v1
	v_ashrrev_i32_e32 v5, 5, v5
	v_mul_lo_u32 v14, v1, s20
	v_lshrrev_b32_e32 v1, 1, v0
	v_mul_lo_u32 v16, v5, s20
	v_and_b32_e32 v5, 0x60, v1
	v_or_b32_e32 v1, s31, v5
	scratch_store_dword off, v1, off        ; 4-byte Folded Spill
	v_ashrrev_i32_e32 v1, 5, v1
	s_waitcnt lgkmcnt(0)
	s_load_dword s11, s[0:1], 0xf4
	v_mul_lo_u32 v20, v1, s28
	s_ashr_i32 s0, s18, 31
	v_add_u32_e32 v1, 0x8800, v2
	s_add_u32 s16, s12, s18
	v_add_u32_e32 v225, v1, v3
	s_addc_u32 s17, s13, s0
	v_readfirstlane_b32 s0, v225
	v_add_u32_e32 v1, v1, v4
	s_mov_b32 s19, s7
	s_mov_b32 m0, s0
	v_readfirstlane_b32 s0, v1
	v_or_b32_e32 v177, 0x8000, v251
	scratch_store_dword off, v6, off offset:20 ; 4-byte Folded Spill
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	buffer_load_dwordx4 v150, s[16:19], 0 offen lds
	s_mov_b32 m0, s0
	s_add_u32 s4, s4, 0x80
	v_readfirstlane_b32 s0, v177
	v_add_u32_e32 v163, 0xa000, v251
	v_add_u32_e32 v2, 0xcc00, v2
	buffer_load_dwordx4 v152, s[16:19], 0 offen lds
	s_addc_u32 s5, s5, 0
	s_mov_b32 m0, s0
	v_readfirstlane_b32 s0, v163
	v_add_u32_e32 v252, v2, v3
	buffer_load_dwordx4 v148, s[4:7], 0 offen lds
	s_mov_b32 m0, s0
	s_add_u32 s16, s16, 0x80
	v_readfirstlane_b32 s0, v252
	v_add_u32_e32 v214, v2, v4
	buffer_load_dwordx4 v146, s[4:7], 0 offen lds
	s_addc_u32 s17, s17, 0
	s_mov_b32 m0, s0
	v_readfirstlane_b32 s0, v214
	buffer_load_dwordx4 v150, s[16:19], 0 offen lds
	s_mov_b32 m0, s0
	v_mul_lo_u32 v12, v6, s20
	buffer_load_dwordx4 v152, s[16:19], 0 offen lds
	v_bfe_u32 v2, v0, 1, 3
	v_lshlrev_b32_e32 v3, 3, v0
	v_bfe_u32 v6, v0, 4, 2
	v_and_b32_e32 v4, 8, v3
	v_lshlrev_b32_e32 v7, 11, v6
	v_add_u32_e32 v8, v6, v2
	v_lshlrev_b32_e32 v9, 4, v2
	v_or_b32_e32 v10, v5, v4
	v_lshl_or_b32 v7, v8, 7, v7
	v_bitop3_b32 v8, v5, v9, v4 bitop3:0x36
	v_or_b32_e32 v4, 4, v6
	v_lshlrev_b32_e32 v5, 11, v4
	v_add_u32_e32 v2, v4, v2
	v_lshl_or_b32 v202, v2, 7, v5
	;;#ASMSTART
	s_waitcnt vmcnt(6)
	;;#ASMEND
	s_barrier
	v_or_b32_e32 v147, v7, v8
	scratch_store_dword off, v8, off offset:4 ; 4-byte Folded Spill
	v_or_b32_e32 v215, v8, v202
	v_lshlrev_b32_e32 v8, 7, v0
	v_bitop3_b32 v4, v10, v9, 16 bitop3:0x36
	v_and_b32_e32 v6, 48, v0
	v_and_b32_e32 v8, 0x780, v8
	v_or_b32_e32 v183, v4, v7
	v_or_b32_e32 v205, v202, v4
	v_lshlrev_b32_e32 v4, 13, v11
	v_or_b32_e32 v11, v8, v6
	v_and_b32_e32 v3, 0x70, v3
	v_or_b32_e32 v5, 0x11000, v4
	v_bitop3_b32 v6, v8, v3, v6 bitop3:0x36
	v_bitop3_b32 v3, v11, v3, 64 bitop3:0x36
	v_or_b32_e32 v2, 16, v10
	v_or_b32_e32 v204, v6, v5
	v_or_b32_e32 v203, v3, v5
	v_add_u32_e32 v5, 0x4400, v7
	v_bitop3_b32 v250, v5, v10, v9 bitop3:0xf6
	v_bitop3_b32 v153, v2, v5, v9 bitop3:0xde
	v_add_u32_e32 v5, 0x4400, v202
	v_bitop3_b32 v216, v5, v10, v9 bitop3:0xf6
	scratch_store_dword off, v5, off offset:16 ; 4-byte Folded Spill
	v_bitop3_b32 v253, v5, v2, v9 bitop3:0xf6
	v_or_b32_e32 v5, 0x15000, v4
	v_or_b32_e32 v187, v6, v5
	v_or_b32_e32 v186, v3, v5
	v_add_u32_e32 v5, 0x8800, v7
	v_bitop3_b32 v185, v5, v10, v9 bitop3:0xf6
	v_bitop3_b32 v184, v2, v5, v9 bitop3:0xde
	v_add_u32_e32 v5, 0x8800, v202
	v_bitop3_b32 v217, v5, v10, v9 bitop3:0xf6
	scratch_store_dword off, v5, off offset:12 ; 4-byte Folded Spill
	v_bitop3_b32 v182, v5, v2, v9 bitop3:0xf6
	v_or_b32_e32 v5, 0x19000, v4
	v_or_b32_e32 v181, v6, v5
	v_or_b32_e32 v180, v3, v5
	v_add_u32_e32 v5, 0xcc00, v7
	v_bitop3_b32 v179, v5, v10, v9 bitop3:0xf6
	v_bitop3_b32 v178, v2, v5, v9 bitop3:0xde
	v_add_u32_e32 v5, 0xcc00, v202
	v_bitop3_b32 v176, v5, v2, v9 bitop3:0xf6
	v_or_b32_e32 v2, 0x1d000, v4
	v_or_b32_e32 v151, v6, v2
	v_or_b32_e32 v149, v3, v2
	v_and_b32_e32 v2, 63, v0
	v_ashrrev_i32_e32 v13, 31, v12
	v_lshlrev_b32_e32 v18, 2, v2
	v_mov_b32_e32 v19, 0
	v_ashrrev_i32_e32 v17, 31, v16
	v_lshl_add_u64 v[2:3], v[18:19], 0, v[12:13]
	v_lshl_add_u64 v[164:165], s[14:15], 0, v[2:3]
	v_lshl_add_u64 v[2:3], v[18:19], 0, v[16:17]
	v_lshl_add_u64 v[166:167], s[14:15], 0, v[2:3]
	v_add_u32_e32 v2, s20, v12
	v_ashrrev_i32_e32 v3, 31, v2
	v_ashrrev_i32_e32 v15, 31, v14
	v_lshl_add_u64 v[2:3], v[18:19], 0, v[2:3]
	v_ashrrev_i32_e32 v21, 31, v20
	v_lshl_add_u32 v22, s28, 2, v20
	v_lshl_add_u64 v[168:169], s[14:15], 0, v[2:3]
	v_lshl_add_u64 v[2:3], v[18:19], 0, v[14:15]
	v_ashrrev_i32_e32 v23, 31, v22
	v_lshl_add_u64 v[170:171], s[14:15], 0, v[2:3]
	v_lshl_add_u64 v[2:3], v[18:19], 0, v[20:21]
	v_lshl_add_u64 v[172:173], s[24:25], 0, v[2:3]
	v_lshl_add_u64 v[2:3], v[18:19], 0, v[22:23]
	v_or_b32_e32 v201, 0xc000, v251
	v_add_u32_e32 v200, 0xe000, v251
	v_bitop3_b32 v218, v5, v10, v9 bitop3:0xf6
	s_lshl_b32 s28, s26, 8
	s_mulk_i32 s26, 0x180
	scratch_store_dwordx2 off, v[20:21], off offset:48 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[22:23], off offset:56 ; 8-byte Folded Spill
	v_lshl_add_u64 v[174:175], s[24:25], 0, v[2:3]
	s_mov_b64 s[0:1], 0
	v_mov_b32_e32 v18, v19
	v_mov_b32_e32 v20, v19
	v_mov_b32_e32 v21, v19
	v_mov_b32_e32 v22, v19
	v_mov_b32_e32 v23, v19
	v_mov_b32_e32 v24, v19
	v_mov_b32_e32 v25, v19
	v_mov_b32_e32 v26, v19
	v_mov_b32_e32 v27, v19
	v_mov_b32_e32 v28, v19
	v_mov_b32_e32 v29, v19
	v_mov_b32_e32 v30, v19
	v_mov_b32_e32 v31, v19
	v_mov_b32_e32 v32, v19
	v_mov_b32_e32 v33, v19
	v_mov_b32_e32 v34, v19
	v_mov_b32_e32 v35, v19
	v_mov_b32_e32 v36, v19
	v_mov_b32_e32 v37, v19
	v_mov_b32_e32 v38, v19
	v_mov_b32_e32 v39, v19
	v_mov_b32_e32 v40, v19
	v_mov_b32_e32 v41, v19
	v_mov_b32_e32 v42, v19
	v_mov_b32_e32 v43, v19
	v_mov_b32_e32 v44, v19
	v_mov_b32_e32 v45, v19
	v_mov_b32_e32 v46, v19
	v_mov_b32_e32 v47, v19
	v_mov_b32_e32 v48, v19
	v_mov_b32_e32 v49, v19
	v_mov_b32_e32 v70, v19
	v_mov_b32_e32 v71, v19
	v_mov_b32_e32 v72, v19
	v_mov_b32_e32 v73, v19
	v_mov_b32_e32 v78, v19
	v_mov_b32_e32 v79, v19
	v_mov_b32_e32 v80, v19
	v_mov_b32_e32 v81, v19
	v_mov_b32_e32 v62, v19
	v_mov_b32_e32 v63, v19
	v_mov_b32_e32 v64, v19
	v_mov_b32_e32 v65, v19
	v_mov_b32_e32 v74, v19
	v_mov_b32_e32 v75, v19
	v_mov_b32_e32 v76, v19
	v_mov_b32_e32 v77, v19
	v_mov_b32_e32 v54, v19
	v_mov_b32_e32 v55, v19
	v_mov_b32_e32 v56, v19
	v_mov_b32_e32 v57, v19
	v_mov_b32_e32 v66, v19
	v_mov_b32_e32 v67, v19
	v_mov_b32_e32 v68, v19
	v_mov_b32_e32 v69, v19
	v_mov_b32_e32 v50, v19
	v_mov_b32_e32 v51, v19
	v_mov_b32_e32 v52, v19
	v_mov_b32_e32 v53, v19
	v_mov_b32_e32 v58, v19
	v_mov_b32_e32 v59, v19
	v_mov_b32_e32 v60, v19
	v_mov_b32_e32 v61, v19
	v_mov_b32_e32 v82, v19
	v_mov_b32_e32 v83, v19
	v_mov_b32_e32 v84, v19
	v_mov_b32_e32 v85, v19
	v_mov_b32_e32 v86, v19
	v_mov_b32_e32 v87, v19
	v_mov_b32_e32 v88, v19
	v_mov_b32_e32 v89, v19
	v_mov_b32_e32 v90, v19
	v_mov_b32_e32 v91, v19
	v_mov_b32_e32 v92, v19
	v_mov_b32_e32 v93, v19
	v_mov_b32_e32 v94, v19
	v_mov_b32_e32 v95, v19
	v_mov_b32_e32 v96, v19
	v_mov_b32_e32 v97, v19
	v_mov_b32_e32 v98, v19
	v_mov_b32_e32 v99, v19
	v_mov_b32_e32 v100, v19
	v_mov_b32_e32 v101, v19
	v_mov_b32_e32 v102, v19
	v_mov_b32_e32 v103, v19
	v_mov_b32_e32 v104, v19
	v_mov_b32_e32 v105, v19
	v_mov_b32_e32 v106, v19
	v_mov_b32_e32 v107, v19
	v_mov_b32_e32 v108, v19
	v_mov_b32_e32 v109, v19
	v_mov_b32_e32 v110, v19
	v_mov_b32_e32 v111, v19
	v_mov_b32_e32 v112, v19
	v_mov_b32_e32 v113, v19
	v_mov_b32_e32 v134, v19
	v_mov_b32_e32 v135, v19
	v_mov_b32_e32 v136, v19
	v_mov_b32_e32 v137, v19
	v_mov_b32_e32 v142, v19
	v_mov_b32_e32 v143, v19
	v_mov_b32_e32 v144, v19
	v_mov_b32_e32 v145, v19
	v_mov_b32_e32 v126, v19
	v_mov_b32_e32 v127, v19
	v_mov_b32_e32 v128, v19
	v_mov_b32_e32 v129, v19
	v_mov_b32_e32 v138, v19
	v_mov_b32_e32 v139, v19
	v_mov_b32_e32 v140, v19
	v_mov_b32_e32 v141, v19
	v_mov_b32_e32 v118, v19
	v_mov_b32_e32 v119, v19
	v_mov_b32_e32 v120, v19
	v_mov_b32_e32 v121, v19
	v_mov_b32_e32 v130, v19
	v_mov_b32_e32 v131, v19
	v_mov_b32_e32 v132, v19
	v_mov_b32_e32 v133, v19
	v_mov_b32_e32 v114, v19
	v_mov_b32_e32 v115, v19
	v_mov_b32_e32 v116, v19
	v_mov_b32_e32 v117, v19
	v_mov_b32_e32 v122, v19
	v_mov_b32_e32 v123, v19
	v_mov_b32_e32 v124, v19
	v_mov_b32_e32 v125, v19
	scratch_store_dword off, v5, off offset:8 ; 4-byte Folded Spill
	scratch_store_dwordx2 off, v[16:17], off offset:40 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[12:13], off offset:24 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[14:15], off offset:32 ; 8-byte Folded Spill
.LBB20_5:                               ; =>This Inner Loop Header: Depth=1
	v_lshl_add_u64 v[2:3], v[164:165], 0, s[0:1]
	v_lshl_add_u64 v[4:5], v[166:167], 0, s[0:1]
	v_lshl_add_u64 v[6:7], v[168:169], 0, s[0:1]
	v_lshl_add_u64 v[8:9], v[170:171], 0, s[0:1]
	global_load_dword v224, v[2:3], off
	global_load_dword v220, v[4:5], off
	global_load_dword v223, v[6:7], off
	global_load_dword v219, v[8:9], off
	v_lshl_add_u64 v[2:3], v[172:173], 0, s[0:1]
	v_lshl_add_u64 v[4:5], v[174:175], 0, s[0:1]
	global_load_dword v222, v[2:3], off
	global_load_dword v221, v[4:5], off
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v147 offset:0
ds_read_b64_tr_b8 v[4:5], v147 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v183 offset:0
ds_read_b64_tr_b8 v[12:13], v183 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v215 offset:0
ds_read_b64_tr_b8 v[8:9], v215 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v205 offset:0
ds_read_b64_tr_b8 v[16:17], v205 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[226:229], v204 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[234:237], v204 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[242:245], v204 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[154:157], v204 offset:0x1800

	;;#ASMEND
	s_add_i32 s33, s29, s0
	;;#ASMSTART
	ds_read_b128 v[230:233], v203 offset:0

	;;#ASMEND
	s_add_i32 s4, s33, 0x80
	;;#ASMSTART
	ds_read_b128 v[238:241], v203 offset:0x800

	;;#ASMEND
	s_ashr_i32 s5, s4, 31
	;;#ASMSTART
	ds_read_b128 v[246:249], v203 offset:0x1000

	;;#ASMEND
	s_add_u32 s4, s22, s4
	v_readfirstlane_b32 s16, v201
	;;#ASMSTART
	ds_read_b128 v[158:161], v203 offset:0x1800

	;;#ASMEND
	s_addc_u32 s5, s23, s5
	s_mov_b32 m0, s16
	v_readfirstlane_b32 s16, v200
	buffer_load_dwordx4 v148, s[4:7], 0 offen lds
	s_mov_b32 m0, s16
	s_nop 0
	buffer_load_dwordx4 v146, s[4:7], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	s_waitcnt vmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[134:137], v[226:233], v[2:9], v[134:137], v224, v222 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[142:145], v[226:233], v[10:17], v[142:145], v224, v222 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[126:129], v[234:241], v[2:9], v[126:129], v224, v222 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[138:141], v[234:241], v[10:17], v[138:141], v224, v222 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[118:121], v[242:249], v[2:9], v[118:121], v223, v222 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[130:133], v[242:249], v[10:17], v[130:133], v223, v222 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[114:117], v[154:161], v[2:9], v[114:117], v223, v222 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[122:125], v[154:161], v[10:17], v[122:125], v223, v222 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_i32 s31, s28, s30
	;;#ASMSTART
	ds_read_b64_tr_b8 v[206:207], v250 offset:0
ds_read_b64_tr_b8 v[208:209], v250 offset:1024

	;;#ASMEND
	s_ashr_i32 s4, s31, 31
	;;#ASMSTART
	ds_read_b64_tr_b8 v[188:189], v153 offset:0
ds_read_b64_tr_b8 v[190:191], v153 offset:1024

	;;#ASMEND
	s_add_u32 s16, s12, s31
	;;#ASMSTART
	ds_read_b64_tr_b8 v[210:211], v216 offset:0
ds_read_b64_tr_b8 v[212:213], v216 offset:1024

	;;#ASMEND
	s_addc_u32 s17, s13, s4
	v_readfirstlane_b32 s4, v254
	;;#ASMSTART
	ds_read_b64_tr_b8 v[192:193], v253 offset:0
ds_read_b64_tr_b8 v[194:195], v253 offset:1024

	;;#ASMEND
	s_mov_b32 s19, s7
	s_mov_b32 m0, s4
	v_readfirstlane_b32 s4, v255
	buffer_load_dwordx4 v150, s[16:19], 0 offen lds
	s_mov_b32 m0, s4
	s_nop 0
	buffer_load_dwordx4 v152, s[16:19], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	s_waitcnt vmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[226:233], v[206:213], v[82:85], v224, v221 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[226:233], v[188:195], v[86:89], v224, v221 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[234:241], v[206:213], v[90:93], v224, v221 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[234:241], v[188:195], v[94:97], v224, v221 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[98:101], v[242:249], v[206:213], v[98:101], v223, v221 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[102:105], v[242:249], v[188:195], v[102:105], v223, v221 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[106:109], v[154:161], v[206:213], v[106:109], v223, v221 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[110:113], v[154:161], v[188:195], v[110:113], v223, v221 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[154:157], v187 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[226:229], v187 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[234:237], v187 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[242:245], v187 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[158:161], v186 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[230:233], v186 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[238:241], v186 offset:0x1000

	;;#ASMEND
	s_add_u32 s16, s16, 0x80
	v_readfirstlane_b32 s4, v196
	;;#ASMSTART
	ds_read_b128 v[246:249], v186 offset:0x1800

	;;#ASMEND
	s_addc_u32 s17, s17, 0
	s_mov_b32 m0, s4
	v_readfirstlane_b32 s4, v197
	buffer_load_dwordx4 v150, s[16:19], 0 offen lds
	s_mov_b32 m0, s4
	s_nop 0
	buffer_load_dwordx4 v152, s[16:19], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[154:161], v[2:9], v[70:73], v220, v222 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[154:161], v[10:17], v[78:81], v220, v222 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[226:233], v[2:9], v[62:65], v220, v222 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[226:233], v[10:17], v[74:77], v220, v222 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[234:241], v[2:9], v[54:57], v219, v222 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[234:241], v[10:17], v[66:69], v219, v222 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[242:249], v[2:9], v[50:53], v219, v222 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[242:249], v[10:17], v[58:61], v219, v222 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_i32 s34, s27, s0
	s_add_i32 s4, s34, 0x100
	s_ashr_i32 s5, s4, 31
	s_add_u32 s4, s22, s4
	v_readfirstlane_b32 s16, v251
	s_addc_u32 s5, s23, s5
	s_mov_b32 m0, s16
	v_readfirstlane_b32 s16, v162
	buffer_load_dwordx4 v148, s[4:7], 0 offen lds
	s_mov_b32 m0, s16
	s_nop 0
	buffer_load_dwordx4 v146, s[4:7], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[154:161], v[206:213], v[18:21], v220, v221 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[154:161], v[188:195], v[22:25], v220, v221 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[226:233], v[206:213], v[26:29], v220, v221 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[226:233], v[188:195], v[30:33], v220, v221 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[234:241], v[206:213], v[34:37], v219, v221 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[234:241], v[188:195], v[38:41], v219, v221 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[242:249], v[206:213], v[42:45], v219, v221 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], v[242:249], v[188:195], v[46:49], v219, v221 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v185 offset:0
ds_read_b64_tr_b8 v[4:5], v185 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v184 offset:0
ds_read_b64_tr_b8 v[12:13], v184 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v217 offset:0
ds_read_b64_tr_b8 v[8:9], v217 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v182 offset:0
ds_read_b64_tr_b8 v[16:17], v182 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[154:157], v181 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[188:191], v181 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[206:209], v181 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[226:229], v181 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[158:161], v180 offset:0

	;;#ASMEND
	s_addk_i32 s33, 0x100
	;;#ASMSTART
	ds_read_b128 v[192:195], v180 offset:0x800

	;;#ASMEND
	s_ashr_i32 s5, s33, 31
	;;#ASMSTART
	ds_read_b128 v[210:213], v180 offset:0x1000

	;;#ASMEND
	s_add_u32 s4, s22, s33
	v_readfirstlane_b32 s16, v198
	;;#ASMSTART
	ds_read_b128 v[230:233], v180 offset:0x1800

	;;#ASMEND
	s_addc_u32 s5, s23, s5
	s_mov_b32 m0, s16
	v_readfirstlane_b32 s16, v199
	buffer_load_dwordx4 v148, s[4:7], 0 offen lds
	s_mov_b32 m0, s16
	s_nop 0
	buffer_load_dwordx4 v146, s[4:7], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[134:137], v[154:161], v[2:9], v[134:137], v224, v222 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[142:145], v[154:161], v[10:17], v[142:145], v224, v222 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[126:129], v[188:195], v[2:9], v[126:129], v224, v222 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[138:141], v[188:195], v[10:17], v[138:141], v224, v222 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[118:121], v[206:213], v[2:9], v[118:121], v223, v222 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[130:133], v[206:213], v[10:17], v[130:133], v223, v222 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[114:117], v[226:233], v[2:9], v[114:117], v223, v222 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[122:125], v[226:233], v[10:17], v[122:125], v223, v222 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[234:235], v179 offset:0
ds_read_b64_tr_b8 v[236:237], v179 offset:1024

	;;#ASMEND
	s_add_i32 s4, s26, s30
	;;#ASMSTART
	ds_read_b64_tr_b8 v[242:243], v178 offset:0
ds_read_b64_tr_b8 v[244:245], v178 offset:1024

	;;#ASMEND
	s_ashr_i32 s5, s4, 31
	;;#ASMSTART
	ds_read_b64_tr_b8 v[238:239], v218 offset:0
ds_read_b64_tr_b8 v[240:241], v218 offset:1024

	;;#ASMEND
	s_add_u32 s16, s12, s4
	v_readfirstlane_b32 s4, v225
	;;#ASMSTART
	ds_read_b64_tr_b8 v[246:247], v176 offset:0
ds_read_b64_tr_b8 v[248:249], v176 offset:1024

	;;#ASMEND
	s_addc_u32 s17, s13, s5
	s_mov_b32 m0, s4
	v_readfirstlane_b32 s4, v1
	buffer_load_dwordx4 v150, s[16:19], 0 offen lds
	s_mov_b32 m0, s4
	s_nop 0
	buffer_load_dwordx4 v152, s[16:19], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[154:161], v[234:241], v[82:85], v224, v221 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[154:161], v[242:249], v[86:89], v224, v221 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[188:195], v[234:241], v[90:93], v224, v221 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[188:195], v[242:249], v[94:97], v224, v221 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[98:101], v[206:213], v[234:241], v[98:101], v223, v221 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[102:105], v[206:213], v[242:249], v[102:105], v223, v221 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[106:109], v[226:233], v[234:241], v[106:109], v223, v221 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[110:113], v[226:233], v[242:249], v[110:113], v223, v221 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[154:157], v151 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[188:191], v151 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[206:209], v151 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[226:229], v151 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[158:161], v149 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[192:195], v149 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[210:213], v149 offset:0x1000

	;;#ASMEND
	s_add_u32 s16, s16, 0x80
	v_readfirstlane_b32 s4, v252
	;;#ASMSTART
	ds_read_b128 v[230:233], v149 offset:0x1800

	;;#ASMEND
	s_addc_u32 s17, s17, 0
	s_mov_b32 m0, s4
	v_readfirstlane_b32 s4, v214
	buffer_load_dwordx4 v150, s[16:19], 0 offen lds
	s_mov_b32 m0, s4
	s_nop 0
	buffer_load_dwordx4 v152, s[16:19], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[154:161], v[2:9], v[70:73], v220, v222 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[154:161], v[10:17], v[78:81], v220, v222 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[188:195], v[2:9], v[62:65], v220, v222 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[188:195], v[10:17], v[74:77], v220, v222 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[206:213], v[2:9], v[54:57], v219, v222 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[206:213], v[10:17], v[66:69], v219, v222 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[226:233], v[2:9], v[50:53], v219, v222 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[226:233], v[10:17], v[58:61], v219, v222 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_addk_i32 s34, 0x180
	s_ashr_i32 s5, s34, 31
	s_add_u32 s4, s22, s34
	v_readfirstlane_b32 s16, v177
	s_addc_u32 s5, s23, s5
	s_mov_b32 m0, s16
	v_readfirstlane_b32 s16, v163
	buffer_load_dwordx4 v148, s[4:7], 0 offen lds
	s_mov_b32 m0, s16
	s_nop 0
	buffer_load_dwordx4 v146, s[4:7], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[154:161], v[234:241], v[18:21], v220, v221 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[154:161], v[242:249], v[22:25], v220, v221 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[188:195], v[234:241], v[26:29], v220, v221 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[188:195], v[242:249], v[30:33], v220, v221 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[206:213], v[234:241], v[34:37], v219, v221 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[206:213], v[242:249], v[38:41], v219, v221 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[226:233], v[234:241], v[42:45], v219, v221 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], v[226:233], v[242:249], v[46:49], v219, v221 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_add_u32 s0, s0, 0x100
	s_addc_u32 s1, s1, 0
	s_cmpk_eq_i32 s0, 0xf00
	s_mov_b32 s30, s31
	s_barrier
	s_cbranch_scc0 .LBB20_5
; %bb.6:
	scratch_load_dwordx2 v[2:3], off, off offset:24 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[8:9], off, off offset:32 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[4:5], off, off offset:40 ; 8-byte Folded Reload
	scratch_load_dword v1, off, off offset:20 ; 4-byte Folded Reload
	scratch_load_dwordx2 v[10:11], off, off offset:48 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[12:13], off, off offset:56 ; 8-byte Folded Reload
	v_mov_b32_e32 v15, 0
	s_add_u32 s4, s8, 0xf80
	v_readfirstlane_b32 s0, v201
	s_addc_u32 s5, s9, 0
	s_mov_b32 s7, 0x110000
	s_mov_b32 m0, s0
	v_readfirstlane_b32 s0, v200
	s_waitcnt vmcnt(5)
	v_lshl_add_u64 v[2:3], s[14:15], 0, v[2:3]
	s_waitcnt vmcnt(4)
	v_lshl_add_u64 v[8:9], s[14:15], 0, v[8:9]
	s_waitcnt vmcnt(3)
	v_lshl_add_u64 v[4:5], s[14:15], 0, v[4:5]
	s_waitcnt vmcnt(2)
	v_or_b32_e32 v1, 1, v1
	v_mul_lo_u32 v6, v1, s20
	v_mov_b32_e32 v1, 0xf00
	v_ashrrev_i32_e32 v7, 31, v6
	v_lshl_or_b32 v14, v0, 2, v1
	v_lshl_add_u64 v[6:7], s[14:15], 0, v[6:7]
	s_waitcnt vmcnt(1)
	v_lshl_add_u64 v[10:11], s[24:25], 0, v[10:11]
	s_waitcnt vmcnt(0)
	v_lshl_add_u64 v[12:13], s[24:25], 0, v[12:13]
	v_lshl_add_u64 v[2:3], v[2:3], 0, v[14:15]
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[14:15]
	v_lshl_add_u64 v[6:7], v[6:7], 0, v[14:15]
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[14:15]
	global_load_dword v1, v[2:3], off
	global_load_dword v152, v[4:5], off
	global_load_dword v172, v[6:7], off
	global_load_dword v150, v[8:9], off
	v_lshl_add_u64 v[2:3], v[10:11], 0, v[14:15]
	v_lshl_add_u64 v[4:5], v[12:13], 0, v[14:15]
	global_load_dword v155, v[2:3], off
	global_load_dword v154, v[4:5], off
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v147 offset:0
ds_read_b64_tr_b8 v[4:5], v147 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v183 offset:0
ds_read_b64_tr_b8 v[12:13], v183 offset:1024

	;;#ASMEND
	scratch_load_dword v147, off, off offset:4 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v14, v202, v147
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v14 offset:0
ds_read_b64_tr_b8 v[8:9], v14 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v205 offset:0
ds_read_b64_tr_b8 v[16:17], v205 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[156:159], v204 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[164:167], v204 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[206:209], v204 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[214:217], v204 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[160:163], v203 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[168:171], v203 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[210:213], v203 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[218:221], v203 offset:0x1800

	;;#ASMEND
	buffer_load_dwordx4 v148, s[4:7], 0 offen lds
	s_mov_b32 m0, s0
	s_nop 0
	buffer_load_dwordx4 v146, s[4:7], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[126:129], v[164:171], v[2:9], v[126:129], v1, v155 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[118:121], v[206:213], v[2:9], v[118:121], v172, v155 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[114:117], v[214:221], v[2:9], v[114:117], v172, v155 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[122:125], v[214:221], v[10:17], v[122:125], v172, v155 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[134:137], v[156:163], v[2:9], v[134:137], v1, v155 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[142:145], v[156:163], v[10:17], v[142:145], v1, v155 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[138:141], v[164:171], v[10:17], v[138:141], v1, v155 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[130:133], v[206:213], v[10:17], v[130:133], v172, v155 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[194:195], v250 offset:0
ds_read_b64_tr_b8 v[196:197], v250 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[222:223], v153 offset:0
ds_read_b64_tr_b8 v[224:225], v153 offset:1024

	;;#ASMEND
	scratch_load_dword v146, off, off offset:16 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v146, v146, v147
	;;#ASMSTART
	ds_read_b64_tr_b8 v[198:199], v146 offset:0
ds_read_b64_tr_b8 v[200:201], v146 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[226:227], v253 offset:0
ds_read_b64_tr_b8 v[228:229], v253 offset:1024

	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[156:163], v[194:201], v[82:85], v1, v154 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[156:163], v[222:229], v[86:89], v1, v154 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[164:171], v[194:201], v[90:93], v1, v154 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[164:171], v[222:229], v[94:97], v1, v154 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[98:101], v[206:213], v[194:201], v[98:101], v172, v154 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[102:105], v[206:213], v[222:229], v[102:105], v172, v154 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[106:109], v[214:221], v[194:201], v[106:109], v172, v154 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[110:113], v[214:221], v[222:229], v[110:113], v172, v154 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[156:159], v187 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[164:167], v187 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[202:205], v187 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[210:213], v187 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[160:163], v186 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[168:171], v186 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[206:209], v186 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[214:217], v186 offset:0x1800

	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[156:163], v[2:9], v[70:73], v152, v155 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[156:163], v[10:17], v[78:81], v152, v155 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[164:171], v[2:9], v[62:65], v152, v155 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[164:171], v[10:17], v[74:77], v152, v155 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[202:209], v[2:9], v[54:57], v150, v155 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[202:209], v[10:17], v[66:69], v150, v155 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[210:217], v[2:9], v[50:53], v150, v155 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[210:217], v[10:17], v[58:61], v150, v155 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v185 offset:0
ds_read_b64_tr_b8 v[4:5], v185 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v184 offset:0
ds_read_b64_tr_b8 v[12:13], v184 offset:1024

	;;#ASMEND
	scratch_load_dword v6, off, off offset:12 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v14, v6, v147
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v14 offset:0
ds_read_b64_tr_b8 v[8:9], v14 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v182 offset:0
ds_read_b64_tr_b8 v[16:17], v182 offset:1024

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(2)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[218:221], v[156:163], v[194:201], v[18:21], v152, v154 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[230:233], v[156:163], v[222:229], v[22:25], v152, v154 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[234:237], v[164:171], v[194:201], v[26:29], v152, v154 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[238:241], v[164:171], v[222:229], v[30:33], v152, v154 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[242:245], v[202:209], v[194:201], v[34:37], v150, v154 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[202:205], v[202:209], v[222:229], v[38:41], v150, v154 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[198:201], v[210:217], v[194:201], v[42:45], v150, v154 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[206:209], v[210:217], v[222:229], v[46:49], v150, v154 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b128 v[156:159], v181 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[164:167], v181 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[182:185], v181 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[190:193], v181 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[160:163], v180 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[168:171], v180 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[186:189], v180 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[194:197], v180 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[156:163], v[2:9], v[134:137], v1, v155 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[156:163], v[10:17], v[142:145], v1, v155 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[164:171], v[2:9], v[126:129], v1, v155 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[164:171], v[10:17], v[138:141], v1, v155 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[182:189], v[2:9], v[118:121], v172, v155 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[182:189], v[10:17], v[130:133], v172, v155 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], v[190:197], v[2:9], v[114:117], v172, v155 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[190:197], v[10:17], v[122:125], v172, v155 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b64_tr_b8 v[130:131], v179 offset:0
ds_read_b64_tr_b8 v[132:133], v179 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[138:139], v178 offset:0
ds_read_b64_tr_b8 v[140:141], v178 offset:1024

	;;#ASMEND
	s_nop 3
	scratch_load_dword v114, off, off offset:8 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v114, v114, v147
	;;#ASMSTART
	ds_read_b64_tr_b8 v[134:135], v114 offset:0
ds_read_b64_tr_b8 v[136:137], v114 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[142:143], v176 offset:0
ds_read_b64_tr_b8 v[144:145], v176 offset:1024

	;;#ASMEND
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[126:129], v[156:163], v[130:137], v[82:85], v1, v154 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[122:125], v[156:163], v[138:145], v[86:89], v1, v154 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[118:121], v[164:171], v[130:137], v[90:93], v1, v154 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[114:117], v[164:171], v[138:145], v[94:97], v1, v154 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[182:189], v[130:137], v[98:101], v172, v154 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[182:189], v[138:145], v[102:105], v172, v154 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[190:197], v[130:137], v[106:109], v172, v154 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[190:197], v[138:145], v[110:113], v172, v154 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[156:159], v151 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[164:167], v151 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[172:175], v151 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[180:183], v151 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[160:163], v149 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[168:171], v149 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[176:179], v149 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[184:187], v149 offset:0x1800

	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[98:101], v[156:163], v[2:9], v[70:73], v152, v155 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[156:163], v[10:17], v[78:81], v152, v155 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[164:171], v[2:9], v[62:65], v152, v155 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[164:171], v[10:17], v[74:77], v152, v155 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[106:109], v[172:179], v[2:9], v[54:57], v150, v155 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[172:179], v[10:17], v[66:69], v150, v155 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[110:113], v[180:187], v[2:9], v[50:53], v150, v155 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[102:105], v[180:187], v[10:17], v[58:61], v150, v155 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[156:163], v[130:137], v[218:221], v152, v154 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[156:163], v[138:145], v[230:233], v152, v154 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[164:171], v[130:137], v[234:237], v152, v154 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[164:171], v[138:145], v[238:241], v152, v154 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[172:179], v[130:137], v[242:245], v150, v154 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[172:179], v[138:145], v[202:205], v150, v154 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[6:9], v[180:187], v[130:137], v[198:201], v150, v154 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[2:5], v[180:187], v[138:145], v[206:209], v150, v154 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_movk_i32 s0, 0x100
	v_cmp_gt_u32_e32 vcc, s0, v0
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB20_8
; %bb.7:
	s_barrier
.LBB20_8:
	s_or_b64 exec, exec, s[0:1]
	scratch_load_dword v184, off, off       ; 4-byte Folded Reload
	v_lshrrev_b32_e32 v1, 8, v0
	v_lshl_or_b32 v183, s21, 2, v1
	v_mul_lo_u32 v1, v183, s10
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v179, s11, v2
	v_mul_f32_e32 v181, s11, v4
	v_mul_f32_e32 v182, s11, v5
	v_mul_f32_e32 v177, s11, v8
	v_mul_f32_e32 v178, s11, v9
	v_mul_f32_e32 v173, s11, v12
	v_mul_f32_e32 v174, s11, v13
	v_mul_f32_e32 v180, s11, v3
	v_mul_f32_e32 v169, s11, v16
	v_mul_f32_e32 v170, s11, v17
	v_mul_f32_e32 v26, s11, v26
	v_mul_f32_e32 v27, s11, v27
	v_mul_f32_e32 v28, s11, v28
	v_mul_f32_e32 v29, s11, v29
	v_mul_f32_e32 v130, s11, v18
	v_mul_f32_e32 v131, s11, v19
	v_mul_f32_e32 v20, s11, v20
	v_mul_f32_e32 v21, s11, v21
	v_mul_f32_e32 v167, s11, v14
	v_mul_f32_e32 v168, s11, v15
	v_mul_f32_e32 v171, s11, v10
	v_mul_f32_e32 v172, s11, v11
	v_mul_f32_e32 v175, s11, v6
	v_mul_f32_e32 v176, s11, v7
	v_mul_f32_e32 v134, s11, v24
	v_mul_f32_e32 v135, s11, v25
	v_mul_f32_e32 v34, s11, v34
	v_mul_f32_e32 v35, s11, v35
	v_mul_f32_e32 v132, s11, v22
	v_mul_f32_e32 v133, s11, v23
	v_mul_f32_e32 v141, s11, v32
	v_mul_f32_e32 v142, s11, v33
	v_mul_f32_e32 v36, s11, v36
	v_mul_f32_e32 v37, s11, v37
	v_mul_f32_e32 v139, s11, v30
	v_mul_f32_e32 v140, s11, v31
	v_mul_f32_e32 v149, s11, v40
	v_mul_f32_e32 v150, s11, v41
	v_mul_f32_e32 v137, s11, v44
	v_mul_f32_e32 v138, s11, v45
	v_mul_f32_e32 v145, s11, v48
	v_mul_f32_e32 v146, s11, v49
	v_mul_f32_e32 v165, s11, v52
	v_mul_f32_e32 v166, s11, v53
	v_mul_f32_e32 v161, s11, v56
	v_mul_f32_e32 v162, s11, v57
	v_mul_f32_e32 v157, s11, v60
	v_mul_f32_e32 v158, s11, v61
	v_mul_f32_e32 v42, s11, v42
	v_mul_f32_e32 v147, s11, v38
	v_mul_f32_e32 v148, s11, v39
	v_mul_f32_e32 v136, s11, v43
	v_mul_f32_e32 v143, s11, v46
	v_mul_f32_e32 v144, s11, v47
	v_mul_f32_e32 v126, s11, v126
	v_mul_f32_e32 v151, s11, v62
	v_mul_f32_e32 v152, s11, v63
	v_mul_f32_e32 v153, s11, v64
	v_mul_f32_e32 v154, s11, v65
	v_mul_f32_e32 v155, s11, v58
	v_mul_f32_e32 v156, s11, v59
	v_mul_f32_e32 v159, s11, v54
	v_mul_f32_e32 v160, s11, v55
	v_mul_f32_e32 v163, s11, v50
	v_mul_f32_e32 v164, s11, v51
	v_mul_f32_e32 v127, s11, v127
	v_mul_f32_e32 v128, s11, v128
	v_mul_f32_e32 v129, s11, v129
	v_mul_f32_e32 v122, s11, v122
	v_mul_f32_e32 v123, s11, v123
	v_mul_f32_e32 v124, s11, v124
	v_mul_f32_e32 v125, s11, v125
	v_mul_f32_e32 v118, s11, v118
	v_mul_f32_e32 v119, s11, v119
	v_mul_f32_e32 v120, s11, v120
	v_mul_f32_e32 v121, s11, v121
	s_waitcnt vmcnt(0)
	v_lshl_add_u32 v2, v1, 6, v184
	v_lshrrev_b32_e32 v1, 2, v0
	v_and_b32_e32 v1, 12, v1
	v_and_b32_e32 v0, 15, v0
	v_mad_u64_u32 v[0:1], s[0:1], v1, s10, v[0:1]
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshlrev_b64 v[4:5], 1, v[0:1]
	v_add_u32_e32 v0, s10, v0
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshlrev_b64 v[8:9], 1, v[0:1]
	v_add_u32_e32 v0, s10, v0
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshlrev_b64 v[12:13], 1, v[0:1]
	v_add_u32_e32 v0, s10, v0
	v_ashrrev_i32_e32 v3, 31, v2
	v_ashrrev_i32_e32 v1, 31, v0
	s_mul_i32 s0, s10, 13
	v_lshl_add_u64 v[2:3], v[2:3], 1, s[2:3]
	v_lshlrev_b64 v[16:17], 1, v[0:1]
	v_add_u32_e32 v0, s0, v0
	v_lshl_add_u64 v[6:7], v[2:3], 0, v[4:5]
	v_lshl_add_u64 v[10:11], v[2:3], 0, v[8:9]
	v_lshl_add_u64 v[14:15], v[2:3], 0, v[12:13]
	v_lshl_add_u64 v[18:19], v[2:3], 0, v[16:17]
	v_ashrrev_i32_e32 v1, 31, v0
	global_store_short_d16_hi v[6:7], v26, off
	global_store_short_d16_hi v[10:11], v27, off
	global_store_short_d16_hi v[14:15], v28, off
	global_store_short_d16_hi v[18:19], v29, off
	global_store_short_d16_hi v[6:7], v130, off offset:32
	global_store_short_d16_hi v[10:11], v131, off offset:32
	global_store_short_d16_hi v[14:15], v20, off offset:32
	global_store_short_d16_hi v[18:19], v21, off offset:32
	v_lshlrev_b64 v[20:21], 1, v[0:1]
	v_add_u32_e32 v0, s10, v0
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshlrev_b64 v[24:25], 1, v[0:1]
	v_add_u32_e32 v0, s10, v0
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshlrev_b64 v[28:29], 1, v[0:1]
	v_add_u32_e32 v0, s10, v0
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshl_add_u64 v[22:23], v[2:3], 0, v[20:21]
	v_lshl_add_u64 v[26:27], v[2:3], 0, v[24:25]
	v_lshlrev_b64 v[32:33], 1, v[0:1]
	v_add_u32_e32 v0, s0, v0
	global_store_short_d16_hi v[22:23], v34, off
	global_store_short_d16_hi v[26:27], v35, off
	v_lshl_add_u64 v[30:31], v[2:3], 0, v[28:29]
	v_lshl_add_u64 v[34:35], v[2:3], 0, v[32:33]
	v_ashrrev_i32_e32 v1, 31, v0
	global_store_short_d16_hi v[30:31], v36, off
	global_store_short_d16_hi v[34:35], v37, off
	global_store_short_d16_hi v[22:23], v132, off offset:32
	global_store_short_d16_hi v[26:27], v133, off offset:32
	global_store_short_d16_hi v[30:31], v134, off offset:32
	global_store_short_d16_hi v[34:35], v135, off offset:32
	v_lshlrev_b64 v[36:37], 1, v[0:1]
	v_add_u32_e32 v0, s10, v0
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshlrev_b64 v[40:41], 1, v[0:1]
	v_add_u32_e32 v0, s10, v0
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshlrev_b64 v[44:45], 1, v[0:1]
	v_add_u32_e32 v0, s10, v0
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshlrev_b64 v[48:49], 1, v[0:1]
	v_add_u32_e32 v0, s0, v0
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshlrev_b64 v[52:53], 1, v[0:1]
	v_add_u32_e32 v0, s10, v0
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshlrev_b64 v[56:57], 1, v[0:1]
	v_add_u32_e32 v0, s10, v0
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshlrev_b64 v[60:61], 1, v[0:1]
	v_add_u32_e32 v0, s10, v0
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshl_add_u64 v[38:39], v[2:3], 0, v[36:37]
	v_lshlrev_b64 v[0:1], 1, v[0:1]
	s_mov_b64 s[0:1], 0x100
	global_store_short_d16_hi v[38:39], v42, off
	v_lshl_add_u64 v[42:43], v[2:3], 0, v[40:41]
	v_lshl_add_u64 v[46:47], v[2:3], 0, v[44:45]
	v_lshl_add_u64 v[50:51], v[2:3], 0, v[48:49]
	v_lshl_add_u64 v[54:55], v[2:3], 0, v[52:53]
	v_lshl_add_u64 v[58:59], v[2:3], 0, v[56:57]
	v_lshl_add_u64 v[62:63], v[2:3], 0, v[60:61]
	v_lshl_add_u64 v[64:65], v[2:3], 0, v[0:1]
	v_lshl_add_u64 v[2:3], v[2:3], 0, s[0:1]
	global_store_short_d16_hi v[42:43], v136, off
	global_store_short_d16_hi v[46:47], v137, off
	global_store_short_d16_hi v[50:51], v138, off
	global_store_short_d16_hi v[38:39], v139, off offset:32
	global_store_short_d16_hi v[42:43], v140, off offset:32
	global_store_short_d16_hi v[46:47], v141, off offset:32
	global_store_short_d16_hi v[50:51], v142, off offset:32
	global_store_short_d16_hi v[54:55], v143, off
	global_store_short_d16_hi v[58:59], v144, off
	global_store_short_d16_hi v[62:63], v145, off
	global_store_short_d16_hi v[64:65], v146, off
	global_store_short_d16_hi v[54:55], v147, off offset:32
	global_store_short_d16_hi v[58:59], v148, off offset:32
	global_store_short_d16_hi v[62:63], v149, off offset:32
	global_store_short_d16_hi v[64:65], v150, off offset:32
	global_store_short_d16_hi v[6:7], v126, off offset:256
	global_store_short_d16_hi v[10:11], v127, off offset:256
	global_store_short_d16_hi v[14:15], v128, off offset:256
	global_store_short_d16_hi v[18:19], v129, off offset:256
	v_lshl_add_u64 v[6:7], v[2:3], 0, v[4:5]
	global_store_short_d16_hi v[6:7], v122, off offset:32
	v_lshl_add_u64 v[6:7], v[2:3], 0, v[8:9]
	global_store_short_d16_hi v[6:7], v123, off offset:32
	v_lshl_add_u64 v[6:7], v[2:3], 0, v[12:13]
	global_store_short_d16_hi v[6:7], v124, off offset:32
	v_lshl_add_u64 v[6:7], v[2:3], 0, v[16:17]
	v_mul_f32_e32 v114, s11, v114
	global_store_short_d16_hi v[6:7], v125, off offset:32
	global_store_short_d16_hi v[22:23], v118, off offset:256
	global_store_short_d16_hi v[26:27], v119, off offset:256
	global_store_short_d16_hi v[30:31], v120, off offset:256
	global_store_short_d16_hi v[34:35], v121, off offset:256
	v_lshl_add_u64 v[6:7], v[2:3], 0, v[20:21]
	v_mul_f32_e32 v115, s11, v115
	global_store_short_d16_hi v[6:7], v114, off offset:32
	v_lshl_add_u64 v[6:7], v[2:3], 0, v[24:25]
	v_mul_f32_e32 v116, s11, v116
	global_store_short_d16_hi v[6:7], v115, off offset:32
	v_lshl_add_u64 v[6:7], v[2:3], 0, v[28:29]
	v_mul_f32_e32 v117, s11, v117
	global_store_short_d16_hi v[6:7], v116, off offset:32
	v_lshl_add_u64 v[6:7], v[2:3], 0, v[32:33]
	v_mul_f32_e32 v94, s11, v94
	v_mul_f32_e32 v95, s11, v95
	v_mul_f32_e32 v96, s11, v96
	v_mul_f32_e32 v97, s11, v97
	v_mul_f32_e32 v90, s11, v90
	global_store_short_d16_hi v[6:7], v117, off offset:32
	global_store_short_d16_hi v[38:39], v94, off offset:256
	global_store_short_d16_hi v[42:43], v95, off offset:256
	global_store_short_d16_hi v[46:47], v96, off offset:256
	global_store_short_d16_hi v[50:51], v97, off offset:256
	v_lshl_add_u64 v[6:7], v[2:3], 0, v[36:37]
	v_mul_f32_e32 v91, s11, v91
	global_store_short_d16_hi v[6:7], v90, off offset:32
	v_lshl_add_u64 v[6:7], v[2:3], 0, v[40:41]
	v_mul_f32_e32 v92, s11, v92
	global_store_short_d16_hi v[6:7], v91, off offset:32
	v_lshl_add_u64 v[6:7], v[2:3], 0, v[44:45]
	v_mul_f32_e32 v93, s11, v93
	global_store_short_d16_hi v[6:7], v92, off offset:32
	v_lshl_add_u64 v[6:7], v[2:3], 0, v[48:49]
	v_mul_f32_e32 v86, s11, v86
	v_mul_f32_e32 v87, s11, v87
	v_mul_f32_e32 v88, s11, v88
	v_mul_f32_e32 v89, s11, v89
	v_mul_f32_e32 v82, s11, v82
	global_store_short_d16_hi v[6:7], v93, off offset:32
	global_store_short_d16_hi v[54:55], v86, off offset:256
	global_store_short_d16_hi v[58:59], v87, off offset:256
	global_store_short_d16_hi v[62:63], v88, off offset:256
	global_store_short_d16_hi v[64:65], v89, off offset:256
	v_lshl_add_u64 v[6:7], v[2:3], 0, v[52:53]
	v_mul_f32_e32 v83, s11, v83
	global_store_short_d16_hi v[6:7], v82, off offset:32
	v_lshl_add_u64 v[6:7], v[2:3], 0, v[56:57]
	v_mul_f32_e32 v84, s11, v84
	v_mul_f32_e32 v85, s11, v85
	global_store_short_d16_hi v[6:7], v83, off offset:32
	v_lshl_add_u64 v[6:7], v[2:3], 0, v[60:61]
	v_lshl_add_u64 v[2:3], v[2:3], 0, v[0:1]
	global_store_short_d16_hi v[6:7], v84, off offset:32
	global_store_short_d16_hi v[2:3], v85, off offset:32
	v_or_b32_e32 v2, 2, v183
	v_mul_lo_u32 v2, v2, s10
	v_lshl_add_u32 v2, v2, 6, v184
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], v[2:3], 1, s[2:3]
	v_lshl_add_u64 v[6:7], v[2:3], 0, v[4:5]
	v_lshl_add_u64 v[10:11], v[2:3], 0, v[8:9]
	v_lshl_add_u64 v[14:15], v[2:3], 0, v[12:13]
	v_lshl_add_u64 v[18:19], v[2:3], 0, v[16:17]
	v_lshl_add_u64 v[22:23], v[2:3], 0, v[20:21]
	v_lshl_add_u64 v[26:27], v[2:3], 0, v[24:25]
	v_lshl_add_u64 v[30:31], v[2:3], 0, v[28:29]
	v_lshl_add_u64 v[34:35], v[2:3], 0, v[32:33]
	v_lshl_add_u64 v[38:39], v[2:3], 0, v[36:37]
	v_lshl_add_u64 v[42:43], v[2:3], 0, v[40:41]
	v_lshl_add_u64 v[46:47], v[2:3], 0, v[44:45]
	v_lshl_add_u64 v[50:51], v[2:3], 0, v[48:49]
	v_lshl_add_u64 v[54:55], v[2:3], 0, v[52:53]
	v_lshl_add_u64 v[58:59], v[2:3], 0, v[56:57]
	v_lshl_add_u64 v[62:63], v[2:3], 0, v[60:61]
	v_lshl_add_u64 v[64:65], v[2:3], 0, v[0:1]
	v_lshl_add_u64 v[2:3], v[2:3], 0, s[0:1]
	v_mul_f32_e32 v98, s11, v98
	v_mul_f32_e32 v99, s11, v99
	v_mul_f32_e32 v100, s11, v100
	v_mul_f32_e32 v101, s11, v101
	v_mul_f32_e32 v78, s11, v78
	v_mul_f32_e32 v79, s11, v79
	v_mul_f32_e32 v80, s11, v80
	v_mul_f32_e32 v81, s11, v81
	v_mul_f32_e32 v106, s11, v106
	v_mul_f32_e32 v107, s11, v107
	v_mul_f32_e32 v108, s11, v108
	v_mul_f32_e32 v109, s11, v109
	v_mul_f32_e32 v110, s11, v110
	v_mul_f32_e32 v111, s11, v111
	v_mul_f32_e32 v112, s11, v112
	v_mul_f32_e32 v113, s11, v113
	v_mul_f32_e32 v66, s11, v66
	v_lshl_add_u64 v[4:5], v[2:3], 0, v[4:5]
	v_mul_f32_e32 v70, s11, v70
	v_mul_f32_e32 v71, s11, v71
	v_mul_f32_e32 v72, s11, v72
	v_mul_f32_e32 v73, s11, v73
	v_mul_f32_e32 v74, s11, v74
	v_mul_f32_e32 v75, s11, v75
	v_mul_f32_e32 v76, s11, v76
	v_mul_f32_e32 v77, s11, v77
	v_mul_f32_e32 v102, s11, v102
	v_mul_f32_e32 v103, s11, v103
	v_mul_f32_e32 v104, s11, v104
	v_mul_f32_e32 v105, s11, v105
	v_mul_f32_e32 v67, s11, v67
	v_mul_f32_e32 v68, s11, v68
	v_mul_f32_e32 v69, s11, v69
	global_store_short_d16_hi v[6:7], v98, off
	global_store_short_d16_hi v[10:11], v99, off
	global_store_short_d16_hi v[14:15], v100, off
	global_store_short_d16_hi v[18:19], v101, off
	global_store_short_d16_hi v[6:7], v70, off offset:32
	global_store_short_d16_hi v[10:11], v71, off offset:32
	global_store_short_d16_hi v[14:15], v72, off offset:32
	global_store_short_d16_hi v[18:19], v73, off offset:32
	global_store_short_d16_hi v[22:23], v78, off
	global_store_short_d16_hi v[26:27], v79, off
	global_store_short_d16_hi v[30:31], v80, off
	global_store_short_d16_hi v[34:35], v81, off
	global_store_short_d16_hi v[22:23], v151, off offset:32
	global_store_short_d16_hi v[26:27], v152, off offset:32
	global_store_short_d16_hi v[30:31], v153, off offset:32
	global_store_short_d16_hi v[34:35], v154, off offset:32
	global_store_short_d16_hi v[38:39], v106, off
	global_store_short_d16_hi v[42:43], v107, off
	global_store_short_d16_hi v[46:47], v108, off
	global_store_short_d16_hi v[50:51], v109, off
	global_store_short_d16_hi v[38:39], v74, off offset:32
	global_store_short_d16_hi v[42:43], v75, off offset:32
	global_store_short_d16_hi v[46:47], v76, off offset:32
	global_store_short_d16_hi v[50:51], v77, off offset:32
	global_store_short_d16_hi v[54:55], v110, off
	global_store_short_d16_hi v[58:59], v111, off
	global_store_short_d16_hi v[62:63], v112, off
	global_store_short_d16_hi v[64:65], v113, off
	global_store_short_d16_hi v[54:55], v102, off offset:32
	global_store_short_d16_hi v[58:59], v103, off offset:32
	global_store_short_d16_hi v[62:63], v104, off offset:32
	global_store_short_d16_hi v[64:65], v105, off offset:32
	global_store_short_d16_hi v[6:7], v66, off offset:256
	global_store_short_d16_hi v[10:11], v67, off offset:256
	global_store_short_d16_hi v[14:15], v68, off offset:256
	global_store_short_d16_hi v[18:19], v69, off offset:256
	global_store_short_d16_hi v[4:5], v155, off offset:32
	v_lshl_add_u64 v[4:5], v[2:3], 0, v[8:9]
	global_store_short_d16_hi v[4:5], v156, off offset:32
	v_lshl_add_u64 v[4:5], v[2:3], 0, v[12:13]
	global_store_short_d16_hi v[4:5], v157, off offset:32
	v_lshl_add_u64 v[4:5], v[2:3], 0, v[16:17]
	global_store_short_d16_hi v[4:5], v158, off offset:32
	global_store_short_d16_hi v[22:23], v159, off offset:256
	global_store_short_d16_hi v[26:27], v160, off offset:256
	global_store_short_d16_hi v[30:31], v161, off offset:256
	global_store_short_d16_hi v[34:35], v162, off offset:256
	v_lshl_add_u64 v[4:5], v[2:3], 0, v[20:21]
	global_store_short_d16_hi v[4:5], v163, off offset:32
	v_lshl_add_u64 v[4:5], v[2:3], 0, v[24:25]
	global_store_short_d16_hi v[4:5], v164, off offset:32
	v_lshl_add_u64 v[4:5], v[2:3], 0, v[28:29]
	global_store_short_d16_hi v[4:5], v165, off offset:32
	v_lshl_add_u64 v[4:5], v[2:3], 0, v[32:33]
	global_store_short_d16_hi v[4:5], v166, off offset:32
	global_store_short_d16_hi v[38:39], v167, off offset:256
	global_store_short_d16_hi v[42:43], v168, off offset:256
	global_store_short_d16_hi v[46:47], v169, off offset:256
	global_store_short_d16_hi v[50:51], v170, off offset:256
	v_lshl_add_u64 v[4:5], v[2:3], 0, v[36:37]
	global_store_short_d16_hi v[4:5], v171, off offset:32
	v_lshl_add_u64 v[4:5], v[2:3], 0, v[40:41]
	global_store_short_d16_hi v[4:5], v172, off offset:32
	v_lshl_add_u64 v[4:5], v[2:3], 0, v[44:45]
	global_store_short_d16_hi v[4:5], v173, off offset:32
	v_lshl_add_u64 v[4:5], v[2:3], 0, v[48:49]
	global_store_short_d16_hi v[4:5], v174, off offset:32
	global_store_short_d16_hi v[54:55], v175, off offset:256
	global_store_short_d16_hi v[58:59], v176, off offset:256
	global_store_short_d16_hi v[62:63], v177, off offset:256
	global_store_short_d16_hi v[64:65], v178, off offset:256
	v_lshl_add_u64 v[4:5], v[2:3], 0, v[52:53]
	global_store_short_d16_hi v[4:5], v179, off offset:32
	v_lshl_add_u64 v[4:5], v[2:3], 0, v[56:57]
	global_store_short_d16_hi v[4:5], v180, off offset:32
	v_lshl_add_u64 v[4:5], v[2:3], 0, v[60:61]
	v_lshl_add_u64 v[0:1], v[2:3], 0, v[0:1]
	global_store_short_d16_hi v[4:5], v181, off offset:32
	global_store_short_d16_hi v[0:1], v182, off offset:32
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z29rrr_exact_8wave_scaled_kernelILb1ELi1EEv14layout_globals
