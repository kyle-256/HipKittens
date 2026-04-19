_Z29rrr_exact_8wave_scaled_kernelILb1ELi2EEv14layout_globals: ; @_Z29rrr_exact_8wave_scaled_kernelILb1ELi2EEv14layout_globals
; %bb.0:
	s_load_dwordx2 s[22:23], s[0:1], 0x60
	s_load_dwordx2 s[20:21], s[0:1], 0x90
	s_load_dword s3, s[0:1], 0x128
	s_load_dwordx2 s[24:25], s[0:1], 0x0
	s_load_dwordx2 s[6:7], s[0:1], 0x20
	s_load_dwordx2 s[4:5], s[0:1], 0x30
	s_load_dwordx2 s[26:27], s[0:1], 0x50
	s_waitcnt lgkmcnt(0)
	s_cmp_lt_i32 s3, 8
	s_cselect_b64 s[8:9], -1, 0
	s_and_b32 s7, s3, 7
	s_cmp_lg_u32 s7, 0
	s_cselect_b64 s[10:11], -1, 0
	s_or_b64 s[8:9], s[8:9], s[10:11]
	s_and_b64 vcc, exec, s[8:9]
	s_cbranch_vccnz .LBB3_2
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
.LBB3_2:
	s_ashr_i32 s7, s2, 31
	s_lshr_b32 s7, s7, 26
	s_add_i32 s7, s2, s7
	s_ashr_i32 s10, s7, 6
	s_load_dword s3, s[0:1], 0x108
	s_lshl_b32 s7, s10, 2
	s_sub_i32 s8, 16, s7
	s_cmpk_gt_i32 s2, 0xff
	s_cselect_b32 s9, s8, 4
	s_mov_b32 s27, 16
	s_cmp_lt_i32 s9, 1
	s_mov_b32 s8, 16
	s_cbranch_scc1 .LBB3_4
; %bb.3:
	s_abs_i32 s8, s9
	v_cvt_f32_u32_e32 v1, s8
	s_lshl_b32 s10, s10, 6
	s_sub_i32 s2, s2, s10
	s_sub_i32 s10, 0, s8
	v_rcp_iflag_f32_e32 v1, v1
	s_abs_i32 s12, s2
	s_xor_b32 s11, s2, s9
	s_ashr_i32 s11, s11, 31
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	s_nop 0
	v_readfirstlane_b32 s13, v1
	s_mul_i32 s10, s10, s13
	s_mul_hi_u32 s10, s13, s10
	s_add_i32 s13, s13, s10
	s_mul_hi_u32 s10, s12, s13
	s_mul_i32 s13, s10, s8
	s_sub_i32 s12, s12, s13
	s_add_i32 s14, s10, 1
	s_sub_i32 s13, s12, s8
	s_cmp_ge_u32 s12, s8
	s_cselect_b32 s10, s14, s10
	s_cselect_b32 s12, s13, s12
	s_add_i32 s13, s10, 1
	s_cmp_ge_u32 s12, s8
	s_cselect_b32 s8, s13, s10
	s_xor_b32 s8, s8, s11
	s_sub_i32 s8, s8, s11
	s_mul_i32 s9, s8, s9
	s_sub_i32 s2, s2, s9
	s_add_i32 s27, s2, s7
.LBB3_4:
	s_waitcnt lgkmcnt(0)
	s_ashr_i32 s2, s3, 31
	s_lshr_b32 s2, s2, 27
	s_add_i32 s3, s3, s2
	s_ashr_i32 s2, s3, 5
	s_add_i32 s2, s2, 7
	v_lshlrev_b32_e32 v1, 4, v0
	s_movk_i32 s7, 0x70
	v_lshrrev_b32_e32 v3, 3, v0
	s_and_b32 s2, s2, -8
	s_ashr_i32 s9, s8, 31
	v_bitop3_b32 v2, v1, s7, v0 bitop3:0x48
	v_or_b32_e32 v4, 64, v3
	s_lshl_b32 s28, s8, 8
	s_lshl_b32 s35, s2, 7
	s_lshl_b32 s34, s2, 6
	s_lshl_b64 s[2:3], s[8:9], 2
	v_mad_u64_u32 v[154:155], s[8:9], v3, s6, v[2:3]
	v_mad_u64_u32 v[152:153], s[8:9], v4, s6, v[2:3]
	v_lshlrev_b32_e32 v2, 1, v0
	v_bitop3_b32 v2, v2, s7, v1 bitop3:0x48
	s_ashr_i32 s31, s27, 31
	v_mad_u64_u32 v[156:157], s[8:9], v3, s26, v[2:3]
	v_mad_u64_u32 v[158:159], s[8:9], v4, s26, v[2:3]
	s_ashr_i32 s7, s28, 31
	v_and_b32_e32 v2, 0x1c00, v1
	v_and_b32_e32 v3, 0x180, v0
	s_add_u32 s12, s4, s28
	v_or_b32_e32 v157, v2, v3
	v_or_b32_e32 v4, 0x2200, v3
	s_addc_u32 s13, s5, s7
	s_lshl_b32 s18, s26, 7
	s_mov_b32 s7, 0x110000
	v_readfirstlane_b32 s4, v157
	v_or_b32_e32 v174, v2, v4
	s_mov_b32 s14, s18
	s_mov_b32 s15, s7
	s_mov_b32 m0, s4
	v_readfirstlane_b32 s4, v174
	buffer_load_dwordx4 v156, s[12:15], 0 offen lds
	s_mov_b32 m0, s4
	s_mul_i32 s4, s27, s6
	s_lshl_b32 s29, s4, 8
	s_ashr_i32 s5, s29, 31
	s_add_u32 s4, s24, s29
	v_add_u32_e32 v176, 0x11000, v2
	s_addc_u32 s5, s25, s5
	s_lshl_b32 s6, s6, 7
	v_readfirstlane_b32 s8, v176
	v_add_u32_e32 v177, 0x2000, v176
	v_add_u32_e32 v1, 0x4400, v2
	buffer_load_dwordx4 v158, s[12:15], 0 offen lds
	s_mov_b32 m0, s8
	v_readfirstlane_b32 s8, v177
	s_add_u32 s16, s12, 0x80
	v_or_b32_e32 v180, v1, v3
	buffer_load_dwordx4 v154, s[4:7], 0 offen lds
	s_mov_b32 m0, s8
	s_addc_u32 s17, s13, 0
	v_readfirstlane_b32 s8, v180
	v_add_u32_e32 v181, v1, v4
	s_add_i32 s30, s29, s6
	buffer_load_dwordx4 v152, s[4:7], 0 offen lds
	s_mov_b32 s19, s7
	s_mov_b32 m0, s8
	v_readfirstlane_b32 s8, v181
	s_ashr_i32 s9, s30, 31
	v_or_b32_e32 v183, 0x4000, v176
	buffer_load_dwordx4 v156, s[16:19], 0 offen lds
	s_mov_b32 m0, s8
	s_add_u32 s8, s24, s30
	v_readfirstlane_b32 s14, v183
	v_add_u32_e32 v184, 0x6000, v176
	buffer_load_dwordx4 v158, s[16:19], 0 offen lds
	s_addc_u32 s9, s25, s9
	s_mov_b32 s10, s6
	s_mov_b32 s11, s7
	s_mov_b32 m0, s14
	v_readfirstlane_b32 s14, v184
	buffer_load_dwordx4 v154, s[8:11], 0 offen lds
	s_mov_b32 m0, s14
	v_lshrrev_b32_e32 v1, 8, v0
	buffer_load_dwordx4 v152, s[8:11], 0 offen lds
	s_load_dwordx2 s[10:11], s[0:1], 0xc0
	s_load_dwordx2 s[14:15], s[0:1], 0xe0
	v_bfe_u32 v5, v0, 6, 2
	v_lshl_or_b32 v8, s27, 1, v1
	v_mov_b64_e32 v[6:7], s[22:23]
	v_or_b32_e32 v10, s2, v5
	v_mad_u64_u32 v[6:7], s[16:17], v8, s35, v[6:7]
	v_mov_b64_e32 v[8:9], s[20:21]
	s_mul_i32 s31, s31, s35
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s15, s3, s34
	v_mad_u64_u32 v[8:9], s[2:3], v10, s34, v[8:9]
	v_add_u32_e32 v7, s31, v7
	v_add_u32_e32 v9, s15, v9
	v_readfirstlane_b32 s20, v6
	v_readfirstlane_b32 s19, v7
	v_readfirstlane_b32 s17, v8
	v_readfirstlane_b32 s16, v9
	s_mov_b32 s31, 0
	v_cmp_eq_u32_e32 vcc, 1, v1
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB3_6
; %bb.5:
	s_barrier
.LBB3_6:
	s_or_b64 exec, exec, s[2:3]
	s_load_dword s15, s[0:1], 0xf4
	s_mov_b64 s[0:1], s[4:5]
	s_mov_b64 s[2:3], s[6:7]
	s_mov_b32 s0, s20
	s_mov_b64 s[22:23], s[6:7]
	v_lshlrev_b32_e32 v151, 5, v5
	s_mov_b64 s[20:21], s[4:5]
	s_ashr_i32 s2, s18, 31
	v_add_u32_e32 v5, 0x8800, v2
	s_mov_b32 s21, s16
	s_add_u32 s16, s12, s18
	v_add_u32_e32 v188, v5, v3
	s_mov_b32 s20, s17
	s_addc_u32 s17, s13, s2
	v_readfirstlane_b32 s2, v188
	v_add_u32_e32 v189, v5, v4
	s_mov_b32 s1, s19
	s_mov_b32 s19, s7
	s_mov_b32 m0, s2
	v_readfirstlane_b32 s2, v189
	v_or_b32_e32 v191, 0x8000, v176
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	buffer_load_dwordx4 v156, s[16:19], 0 offen lds
	s_mov_b32 m0, s2
	s_add_u32 s4, s4, 0x80
	v_readfirstlane_b32 s2, v191
	v_add_u32_e32 v192, 0xa000, v176
	v_add_u32_e32 v2, 0xcc00, v2
	buffer_load_dwordx4 v158, s[16:19], 0 offen lds
	s_addc_u32 s5, s5, 0
	s_mov_b32 m0, s2
	v_readfirstlane_b32 s2, v192
	v_add_u32_e32 v194, v2, v3
	buffer_load_dwordx4 v154, s[4:7], 0 offen lds
	s_mov_b32 m0, s2
	s_add_u32 s16, s16, 0x80
	v_readfirstlane_b32 s2, v194
	v_add_u32_e32 v195, v2, v4
	buffer_load_dwordx4 v152, s[4:7], 0 offen lds
	s_addc_u32 s17, s17, 0
	s_mov_b32 m0, s2
	v_readfirstlane_b32 s2, v195
	buffer_load_dwordx4 v156, s[16:19], 0 offen lds
	s_mov_b32 m0, s2
	v_and_b32_e32 v150, 15, v0
	buffer_load_dwordx4 v158, s[16:19], 0 offen lds
	v_bfe_u32 v6, v0, 4, 2
	v_lshlrev_b32_e32 v2, 4, v150
	v_lshl_or_b32 v196, v6, 8, v2
	v_lshlrev_b32_e32 v2, 3, v150
	v_lshl_or_b32 v197, v6, 7, v2
	v_bfe_u32 v2, v0, 1, 3
	v_lshlrev_b32_e32 v3, 3, v0
	v_and_b32_e32 v4, 8, v3
	v_lshlrev_b32_e32 v8, 4, v2
	v_or_b32_e32 v9, v151, v4
	v_bitop3_b32 v159, v151, v8, v4 bitop3:0x36
	v_or_b32_e32 v4, 4, v6
	v_lshlrev_b32_e32 v5, 11, v6
	v_add_u32_e32 v7, v6, v2
	v_lshlrev_b32_e32 v6, 11, v4
	v_add_u32_e32 v2, v4, v2
	v_lshlrev_b32_e32 v10, 7, v0
	v_lshl_or_b32 v5, v7, 7, v5
	v_lshl_or_b32 v198, v2, 7, v6
	v_bitop3_b32 v4, v9, v8, 16 bitop3:0x36
	v_and_b32_e32 v7, 48, v0
	v_and_b32_e32 v10, 0x780, v10
	v_or_b32_e32 v200, v4, v5
	v_or_b32_e32 v193, v198, v4
	v_lshlrev_b32_e32 v4, 13, v1
	v_or_b32_e32 v11, v10, v7
	v_and_b32_e32 v3, 0x70, v3
	v_or_b32_e32 v6, 0x11000, v4
	v_bitop3_b32 v7, v10, v3, v7 bitop3:0x36
	v_bitop3_b32 v3, v11, v3, 64 bitop3:0x36
	v_or_b32_e32 v2, 16, v9
	v_or_b32_e32 v190, v7, v6
	v_or_b32_e32 v187, v3, v6
	v_add_u32_e32 v6, 0x4400, v5
	v_bitop3_b32 v182, v6, v9, v8 bitop3:0xf6
	v_bitop3_b32 v179, v2, v6, v8 bitop3:0xde
	v_or_b32_e32 v6, 0x15000, v4
	v_or_b32_e32 v199, v5, v159
	v_add_u32_e32 v178, 0x4400, v198
	v_or_b32_e32 v173, v7, v6
	v_or_b32_e32 v172, v3, v6
	v_add_u32_e32 v6, 0x8800, v5
	v_add_u32_e32 v169, 0x8800, v198
	v_add_u32_e32 v5, 0xcc00, v5
	v_add_u32_e32 v163, 0xcc00, v198
	v_bitop3_b32 v175, v178, v2, v8 bitop3:0xf6
	v_bitop3_b32 v171, v6, v9, v8 bitop3:0xf6
	v_bitop3_b32 v170, v2, v6, v8 bitop3:0xde
	v_bitop3_b32 v168, v169, v2, v8 bitop3:0xf6
	v_or_b32_e32 v6, 0x19000, v4
	v_bitop3_b32 v164, v2, v5, v8 bitop3:0xde
	v_bitop3_b32 v162, v163, v2, v8 bitop3:0xf6
	v_or_b32_e32 v2, 0x1d000, v4
	v_mov_b32_e32 v18, 0
	v_or_b32_e32 v201, v159, v198
	v_or_b32_e32 v186, 0xc000, v176
	v_add_u32_e32 v185, 0xe000, v176
	v_bitop3_b32 v202, v178, v9, v8 bitop3:0xf6
	v_bitop3_b32 v203, v169, v9, v8 bitop3:0xf6
	v_or_b32_e32 v167, v7, v6
	v_or_b32_e32 v166, v3, v6
	v_bitop3_b32 v165, v5, v9, v8 bitop3:0xf6
	v_bitop3_b32 v204, v163, v9, v8 bitop3:0xf6
	v_or_b32_e32 v155, v7, v2
	v_or_b32_e32 v153, v3, v2
	s_lshl_b32 s33, s26, 8
	s_mulk_i32 s26, 0x180
	s_mov_b32 s2, s35
	s_mov_b32 s22, s34
	s_mov_b32 s34, 0
	s_mov_b32 s36, 0
	s_mov_b32 s35, 0
	v_mov_b32_e32 v19, v18
	v_mov_b32_e32 v20, v18
	v_mov_b32_e32 v21, v18
	v_mov_b32_e32 v22, v18
	v_mov_b32_e32 v23, v18
	v_mov_b32_e32 v24, v18
	v_mov_b32_e32 v25, v18
	v_mov_b32_e32 v26, v18
	v_mov_b32_e32 v27, v18
	v_mov_b32_e32 v28, v18
	v_mov_b32_e32 v29, v18
	v_mov_b32_e32 v30, v18
	v_mov_b32_e32 v31, v18
	v_mov_b32_e32 v32, v18
	v_mov_b32_e32 v33, v18
	v_mov_b32_e32 v34, v18
	v_mov_b32_e32 v35, v18
	v_mov_b32_e32 v36, v18
	v_mov_b32_e32 v37, v18
	v_mov_b32_e32 v38, v18
	v_mov_b32_e32 v39, v18
	v_mov_b32_e32 v40, v18
	v_mov_b32_e32 v41, v18
	v_mov_b32_e32 v42, v18
	v_mov_b32_e32 v43, v18
	v_mov_b32_e32 v44, v18
	v_mov_b32_e32 v45, v18
	v_mov_b32_e32 v46, v18
	v_mov_b32_e32 v47, v18
	v_mov_b32_e32 v48, v18
	v_mov_b32_e32 v49, v18
	v_mov_b32_e32 v70, v18
	v_mov_b32_e32 v71, v18
	v_mov_b32_e32 v72, v18
	v_mov_b32_e32 v73, v18
	v_mov_b32_e32 v78, v18
	v_mov_b32_e32 v79, v18
	v_mov_b32_e32 v80, v18
	v_mov_b32_e32 v81, v18
	v_mov_b32_e32 v62, v18
	v_mov_b32_e32 v63, v18
	v_mov_b32_e32 v64, v18
	v_mov_b32_e32 v65, v18
	v_mov_b32_e32 v74, v18
	v_mov_b32_e32 v75, v18
	v_mov_b32_e32 v76, v18
	v_mov_b32_e32 v77, v18
	v_mov_b32_e32 v54, v18
	v_mov_b32_e32 v55, v18
	v_mov_b32_e32 v56, v18
	v_mov_b32_e32 v57, v18
	v_mov_b32_e32 v66, v18
	v_mov_b32_e32 v67, v18
	v_mov_b32_e32 v68, v18
	v_mov_b32_e32 v69, v18
	v_mov_b32_e32 v50, v18
	v_mov_b32_e32 v51, v18
	v_mov_b32_e32 v52, v18
	v_mov_b32_e32 v53, v18
	v_mov_b32_e32 v58, v18
	v_mov_b32_e32 v59, v18
	v_mov_b32_e32 v60, v18
	v_mov_b32_e32 v61, v18
	v_mov_b32_e32 v82, v18
	v_mov_b32_e32 v83, v18
	v_mov_b32_e32 v84, v18
	v_mov_b32_e32 v85, v18
	v_mov_b32_e32 v86, v18
	v_mov_b32_e32 v87, v18
	v_mov_b32_e32 v88, v18
	v_mov_b32_e32 v89, v18
	v_mov_b32_e32 v90, v18
	v_mov_b32_e32 v91, v18
	v_mov_b32_e32 v92, v18
	v_mov_b32_e32 v93, v18
	v_mov_b32_e32 v94, v18
	v_mov_b32_e32 v95, v18
	v_mov_b32_e32 v96, v18
	v_mov_b32_e32 v97, v18
	v_mov_b32_e32 v98, v18
	v_mov_b32_e32 v99, v18
	v_mov_b32_e32 v100, v18
	v_mov_b32_e32 v101, v18
	v_mov_b32_e32 v102, v18
	v_mov_b32_e32 v103, v18
	v_mov_b32_e32 v104, v18
	v_mov_b32_e32 v105, v18
	v_mov_b32_e32 v106, v18
	v_mov_b32_e32 v107, v18
	v_mov_b32_e32 v108, v18
	v_mov_b32_e32 v109, v18
	v_mov_b32_e32 v110, v18
	v_mov_b32_e32 v111, v18
	v_mov_b32_e32 v112, v18
	v_mov_b32_e32 v113, v18
	v_mov_b32_e32 v134, v18
	v_mov_b32_e32 v135, v18
	v_mov_b32_e32 v136, v18
	v_mov_b32_e32 v137, v18
	v_mov_b32_e32 v142, v18
	v_mov_b32_e32 v143, v18
	v_mov_b32_e32 v144, v18
	v_mov_b32_e32 v145, v18
	v_mov_b32_e32 v126, v18
	v_mov_b32_e32 v127, v18
	v_mov_b32_e32 v128, v18
	v_mov_b32_e32 v129, v18
	v_mov_b32_e32 v138, v18
	v_mov_b32_e32 v139, v18
	v_mov_b32_e32 v140, v18
	v_mov_b32_e32 v141, v18
	v_mov_b32_e32 v118, v18
	v_mov_b32_e32 v119, v18
	v_mov_b32_e32 v120, v18
	v_mov_b32_e32 v121, v18
	v_mov_b32_e32 v130, v18
	v_mov_b32_e32 v131, v18
	v_mov_b32_e32 v132, v18
	v_mov_b32_e32 v133, v18
	v_mov_b32_e32 v114, v18
	v_mov_b32_e32 v115, v18
	v_mov_b32_e32 v116, v18
	v_mov_b32_e32 v117, v18
	v_mov_b32_e32 v122, v18
	v_mov_b32_e32 v123, v18
	v_mov_b32_e32 v124, v18
	v_mov_b32_e32 v125, v18
	;;#ASMSTART
	s_waitcnt vmcnt(6)
	;;#ASMEND
	s_barrier
.LBB3_7:                                ; =>This Inner Loop Header: Depth=1
	buffer_load_dwordx4 v[146:149], v196, s[0:3], s34 offen
	buffer_load_dwordx2 v[160:161], v197, s[20:23], s31 offen
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v199 offset:0
ds_read_b64_tr_b8 v[4:5], v199 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v200 offset:0
ds_read_b64_tr_b8 v[12:13], v200 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v201 offset:0
ds_read_b64_tr_b8 v[8:9], v201 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v193 offset:0
ds_read_b64_tr_b8 v[16:17], v193 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[206:209], v190 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[214:217], v190 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[222:225], v190 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[230:233], v190 offset:0x1800

	;;#ASMEND
	s_add_i32 s38, s30, s35
	;;#ASMSTART
	ds_read_b128 v[210:213], v187 offset:0

	;;#ASMEND
	s_add_i32 s4, s38, 0x80
	;;#ASMSTART
	ds_read_b128 v[218:221], v187 offset:0x800

	;;#ASMEND
	s_ashr_i32 s5, s4, 31
	;;#ASMSTART
	ds_read_b128 v[226:229], v187 offset:0x1000

	;;#ASMEND
	s_add_u32 s4, s24, s4
	v_readfirstlane_b32 s16, v186
	;;#ASMSTART
	ds_read_b128 v[234:237], v187 offset:0x1800

	;;#ASMEND
	s_addc_u32 s5, s25, s5
	s_mov_b32 m0, s16
	v_readfirstlane_b32 s16, v185
	buffer_load_dwordx4 v154, s[4:7], 0 offen lds
	s_mov_b32 m0, s16
	s_nop 0
	buffer_load_dwordx4 v152, s[4:7], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	s_waitcnt vmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[134:137], v[206:213], v[2:9], v[134:137], v146, v160 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[142:145], v[206:213], v[10:17], v[142:145], v146, v160 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[126:129], v[214:221], v[2:9], v[126:129], v146, v160 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[138:141], v[214:221], v[10:17], v[138:141], v146, v160 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[118:121], v[222:229], v[2:9], v[118:121], v148, v160 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[130:133], v[222:229], v[10:17], v[130:133], v148, v160 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[114:117], v[230:237], v[2:9], v[114:117], v148, v160 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[122:125], v[230:237], v[10:17], v[122:125], v148, v160 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_i32 s37, s33, s36
	;;#ASMSTART
	ds_read_b64_tr_b8 v[238:239], v182 offset:0
ds_read_b64_tr_b8 v[240:241], v182 offset:1024

	;;#ASMEND
	s_ashr_i32 s4, s37, 31
	;;#ASMSTART
	ds_read_b64_tr_b8 v[246:247], v179 offset:0
ds_read_b64_tr_b8 v[248:249], v179 offset:1024

	;;#ASMEND
	s_add_u32 s16, s12, s37
	;;#ASMSTART
	ds_read_b64_tr_b8 v[242:243], v202 offset:0
ds_read_b64_tr_b8 v[244:245], v202 offset:1024

	;;#ASMEND
	s_addc_u32 s17, s13, s4
	v_readfirstlane_b32 s4, v157
	;;#ASMSTART
	ds_read_b64_tr_b8 v[250:251], v175 offset:0
ds_read_b64_tr_b8 v[252:253], v175 offset:1024

	;;#ASMEND
	s_mov_b32 s19, s7
	s_mov_b32 m0, s4
	v_readfirstlane_b32 s4, v174
	buffer_load_dwordx4 v156, s[16:19], 0 offen lds
	s_mov_b32 m0, s4
	s_nop 0
	buffer_load_dwordx4 v158, s[16:19], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[206:213], v[238:245], v[82:85], v146, v161 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[206:213], v[246:253], v[86:89], v146, v161 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[214:221], v[238:245], v[90:93], v146, v161 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[214:221], v[246:253], v[94:97], v146, v161 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[98:101], v[222:229], v[238:245], v[98:101], v148, v161 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[102:105], v[222:229], v[246:253], v[102:105], v148, v161 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[106:109], v[230:237], v[238:245], v[106:109], v148, v161 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[110:113], v[230:237], v[246:253], v[110:113], v148, v161 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[206:209], v173 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[214:217], v173 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[222:225], v173 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[230:233], v173 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[210:213], v172 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[218:221], v172 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[226:229], v172 offset:0x1000

	;;#ASMEND
	s_add_u32 s16, s16, 0x80
	v_readfirstlane_b32 s4, v180
	;;#ASMSTART
	ds_read_b128 v[234:237], v172 offset:0x1800

	;;#ASMEND
	s_addc_u32 s17, s17, 0
	s_mov_b32 m0, s4
	v_readfirstlane_b32 s4, v181
	buffer_load_dwordx4 v156, s[16:19], 0 offen lds
	s_mov_b32 m0, s4
	s_nop 0
	buffer_load_dwordx4 v158, s[16:19], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[206:213], v[2:9], v[70:73], v147, v160 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[206:213], v[10:17], v[78:81], v147, v160 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[214:221], v[2:9], v[62:65], v147, v160 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[214:221], v[10:17], v[74:77], v147, v160 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[222:229], v[2:9], v[54:57], v149, v160 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[222:229], v[10:17], v[66:69], v149, v160 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[230:237], v[2:9], v[50:53], v149, v160 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[230:237], v[10:17], v[58:61], v149, v160 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_i32 s39, s29, s35
	s_add_i32 s4, s39, 0x100
	s_ashr_i32 s5, s4, 31
	s_add_u32 s4, s24, s4
	v_readfirstlane_b32 s16, v176
	s_addc_u32 s5, s25, s5
	s_mov_b32 m0, s16
	v_readfirstlane_b32 s16, v177
	buffer_load_dwordx4 v154, s[4:7], 0 offen lds
	s_mov_b32 m0, s16
	s_nop 0
	buffer_load_dwordx4 v152, s[4:7], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[206:213], v[238:245], v[18:21], v147, v161 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[206:213], v[246:253], v[22:25], v147, v161 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[214:221], v[238:245], v[26:29], v147, v161 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[214:221], v[246:253], v[30:33], v147, v161 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[222:229], v[238:245], v[34:37], v149, v161 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[222:229], v[246:253], v[38:41], v149, v161 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[230:237], v[238:245], v[42:45], v149, v161 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], v[230:237], v[246:253], v[46:49], v149, v161 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v171 offset:0
ds_read_b64_tr_b8 v[4:5], v171 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v170 offset:0
ds_read_b64_tr_b8 v[12:13], v170 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v203 offset:0
ds_read_b64_tr_b8 v[8:9], v203 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v168 offset:0
ds_read_b64_tr_b8 v[16:17], v168 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[206:209], v167 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[214:217], v167 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[222:225], v167 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[230:233], v167 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[210:213], v166 offset:0

	;;#ASMEND
	s_addk_i32 s38, 0x100
	;;#ASMSTART
	ds_read_b128 v[218:221], v166 offset:0x800

	;;#ASMEND
	s_ashr_i32 s5, s38, 31
	;;#ASMSTART
	ds_read_b128 v[226:229], v166 offset:0x1000

	;;#ASMEND
	s_add_u32 s4, s24, s38
	v_readfirstlane_b32 s16, v183
	;;#ASMSTART
	ds_read_b128 v[234:237], v166 offset:0x1800

	;;#ASMEND
	s_addc_u32 s5, s25, s5
	s_mov_b32 m0, s16
	v_readfirstlane_b32 s16, v184
	buffer_load_dwordx4 v154, s[4:7], 0 offen lds
	s_mov_b32 m0, s16
	s_nop 0
	buffer_load_dwordx4 v152, s[4:7], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[134:137], v[206:213], v[2:9], v[134:137], v146, v160 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[142:145], v[206:213], v[10:17], v[142:145], v146, v160 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[126:129], v[214:221], v[2:9], v[126:129], v146, v160 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[138:141], v[214:221], v[10:17], v[138:141], v146, v160 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[118:121], v[222:229], v[2:9], v[118:121], v148, v160 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[130:133], v[222:229], v[10:17], v[130:133], v148, v160 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[114:117], v[230:237], v[2:9], v[114:117], v148, v160 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[122:125], v[230:237], v[10:17], v[122:125], v148, v160 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[238:239], v165 offset:0
ds_read_b64_tr_b8 v[240:241], v165 offset:1024

	;;#ASMEND
	s_add_i32 s4, s26, s36
	;;#ASMSTART
	ds_read_b64_tr_b8 v[246:247], v164 offset:0
ds_read_b64_tr_b8 v[248:249], v164 offset:1024

	;;#ASMEND
	s_ashr_i32 s5, s4, 31
	;;#ASMSTART
	ds_read_b64_tr_b8 v[242:243], v204 offset:0
ds_read_b64_tr_b8 v[244:245], v204 offset:1024

	;;#ASMEND
	s_add_u32 s16, s12, s4
	v_readfirstlane_b32 s4, v188
	;;#ASMSTART
	ds_read_b64_tr_b8 v[250:251], v162 offset:0
ds_read_b64_tr_b8 v[252:253], v162 offset:1024

	;;#ASMEND
	s_addc_u32 s17, s13, s5
	s_mov_b32 m0, s4
	v_readfirstlane_b32 s4, v189
	buffer_load_dwordx4 v156, s[16:19], 0 offen lds
	s_mov_b32 m0, s4
	s_nop 0
	buffer_load_dwordx4 v158, s[16:19], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[206:213], v[238:245], v[82:85], v146, v161 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[206:213], v[246:253], v[86:89], v146, v161 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[214:221], v[238:245], v[90:93], v146, v161 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[214:221], v[246:253], v[94:97], v146, v161 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[98:101], v[222:229], v[238:245], v[98:101], v148, v161 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[102:105], v[222:229], v[246:253], v[102:105], v148, v161 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[106:109], v[230:237], v[238:245], v[106:109], v148, v161 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[110:113], v[230:237], v[246:253], v[110:113], v148, v161 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[206:209], v155 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[214:217], v155 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[222:225], v155 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[230:233], v155 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[210:213], v153 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[218:221], v153 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[226:229], v153 offset:0x1000

	;;#ASMEND
	s_add_u32 s16, s16, 0x80
	v_readfirstlane_b32 s4, v194
	;;#ASMSTART
	ds_read_b128 v[234:237], v153 offset:0x1800

	;;#ASMEND
	s_addc_u32 s17, s17, 0
	s_mov_b32 m0, s4
	v_readfirstlane_b32 s4, v195
	buffer_load_dwordx4 v156, s[16:19], 0 offen lds
	s_mov_b32 m0, s4
	s_nop 0
	buffer_load_dwordx4 v158, s[16:19], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[206:213], v[2:9], v[70:73], v147, v160 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[206:213], v[10:17], v[78:81], v147, v160 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[214:221], v[2:9], v[62:65], v147, v160 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[214:221], v[10:17], v[74:77], v147, v160 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[222:229], v[2:9], v[54:57], v149, v160 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[222:229], v[10:17], v[66:69], v149, v160 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[230:237], v[2:9], v[50:53], v149, v160 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[230:237], v[10:17], v[58:61], v149, v160 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_addk_i32 s39, 0x180
	s_ashr_i32 s5, s39, 31
	s_add_u32 s4, s24, s39
	v_readfirstlane_b32 s16, v191
	s_addc_u32 s5, s25, s5
	s_mov_b32 m0, s16
	v_readfirstlane_b32 s16, v192
	buffer_load_dwordx4 v154, s[4:7], 0 offen lds
	s_mov_b32 m0, s16
	s_nop 0
	buffer_load_dwordx4 v152, s[4:7], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[206:213], v[238:245], v[18:21], v147, v161 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[206:213], v[246:253], v[22:25], v147, v161 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[214:221], v[238:245], v[26:29], v147, v161 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[214:221], v[246:253], v[30:33], v147, v161 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[222:229], v[238:245], v[34:37], v149, v161 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[222:229], v[246:253], v[38:41], v149, v161 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[230:237], v[238:245], v[42:45], v149, v161 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], v[230:237], v[246:253], v[46:49], v149, v161 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_addk_i32 s35, 0x100
	s_addk_i32 s34, 0x400
	s_addk_i32 s31, 0x200
	s_cmpk_eq_i32 s35, 0xf00
	s_mov_b32 s36, s37
	s_barrier
	s_cbranch_scc0 .LBB3_7
; %bb.8:
	s_movk_i32 s4, 0x3c00
	buffer_load_dwordx4 v[146:149], v196, s[0:3], s4 offen
	s_movk_i32 s0, 0x1e00
	buffer_load_dwordx2 v[156:157], v197, s[20:23], s0 offen
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v199 offset:0
ds_read_b64_tr_b8 v[4:5], v199 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v200 offset:0
ds_read_b64_tr_b8 v[12:13], v200 offset:1024

	;;#ASMEND
	v_add_u32_e32 v14, v198, v159
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v14 offset:0
ds_read_b64_tr_b8 v[8:9], v14 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v193 offset:0
ds_read_b64_tr_b8 v[16:17], v193 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[192:195], v190 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[200:203], v190 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[208:211], v190 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[216:219], v190 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[196:199], v187 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[204:207], v187 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[212:215], v187 offset:0x1000

	;;#ASMEND
	s_add_u32 s4, s8, 0xf80
	v_readfirstlane_b32 s0, v186
	;;#ASMSTART
	ds_read_b128 v[220:223], v187 offset:0x1800

	;;#ASMEND
	s_addc_u32 s5, s9, 0
	s_mov_b32 s7, 0x110000
	s_mov_b32 m0, s0
	v_readfirstlane_b32 s0, v185
	buffer_load_dwordx4 v154, s[4:7], 0 offen lds
	s_mov_b32 m0, s0
	s_nop 0
	buffer_load_dwordx4 v152, s[4:7], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	s_waitcnt vmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[126:129], v[200:207], v[2:9], v[126:129], v146, v156 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[118:121], v[208:215], v[2:9], v[118:121], v148, v156 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[114:117], v[216:223], v[2:9], v[114:117], v148, v156 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[122:125], v[216:223], v[10:17], v[122:125], v148, v156 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[134:137], v[192:199], v[2:9], v[134:137], v146, v156 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[142:145], v[192:199], v[10:17], v[142:145], v146, v156 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[138:141], v[200:207], v[10:17], v[138:141], v146, v156 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[130:133], v[208:215], v[10:17], v[130:133], v148, v156 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[184:185], v182 offset:0
ds_read_b64_tr_b8 v[186:187], v182 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[224:225], v179 offset:0
ds_read_b64_tr_b8 v[226:227], v179 offset:1024

	;;#ASMEND
	v_add_u32_e32 v152, v178, v159
	;;#ASMSTART
	ds_read_b64_tr_b8 v[188:189], v152 offset:0
ds_read_b64_tr_b8 v[190:191], v152 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[228:229], v175 offset:0
ds_read_b64_tr_b8 v[230:231], v175 offset:1024

	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[192:199], v[184:191], v[82:85], v146, v157 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[192:199], v[224:231], v[86:89], v146, v157 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[200:207], v[184:191], v[90:93], v146, v157 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[200:207], v[224:231], v[94:97], v146, v157 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[98:101], v[208:215], v[184:191], v[98:101], v148, v157 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[102:105], v[208:215], v[224:231], v[102:105], v148, v157 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[106:109], v[216:223], v[184:191], v[106:109], v148, v157 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[110:113], v[216:223], v[224:231], v[110:113], v148, v157 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[174:177], v173 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[192:195], v173 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[200:203], v173 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[208:211], v173 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[178:181], v172 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[196:199], v172 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[204:207], v172 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[212:215], v172 offset:0x1800

	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[174:181], v[2:9], v[70:73], v147, v156 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[174:181], v[10:17], v[78:81], v147, v156 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[192:199], v[2:9], v[62:65], v147, v156 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[192:199], v[10:17], v[74:77], v147, v156 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[200:207], v[2:9], v[54:57], v149, v156 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[200:207], v[10:17], v[66:69], v149, v156 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[208:215], v[2:9], v[50:53], v149, v156 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[208:215], v[10:17], v[58:61], v149, v156 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v171 offset:0
ds_read_b64_tr_b8 v[4:5], v171 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v170 offset:0
ds_read_b64_tr_b8 v[12:13], v170 offset:1024

	;;#ASMEND
	v_add_u32_e32 v14, v169, v159
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v14 offset:0
ds_read_b64_tr_b8 v[8:9], v14 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v168 offset:0
ds_read_b64_tr_b8 v[16:17], v168 offset:1024

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(2)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[216:219], v[174:181], v[184:191], v[18:21], v147, v157 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[220:223], v[174:181], v[224:231], v[22:25], v147, v157 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[232:235], v[192:199], v[184:191], v[26:29], v147, v157 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[236:239], v[192:199], v[224:231], v[30:33], v147, v157 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[240:243], v[200:207], v[184:191], v[34:37], v149, v157 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[200:203], v[200:207], v[224:231], v[38:41], v149, v157 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[204:207], v[208:215], v[184:191], v[42:45], v149, v157 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[208:211], v[208:215], v[224:231], v[46:49], v149, v157 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b128 v[168:171], v167 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[176:179], v167 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[184:187], v167 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[192:195], v167 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[172:175], v166 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[180:183], v166 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[188:191], v166 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[196:199], v166 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[168:175], v[2:9], v[134:137], v146, v156 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[168:175], v[10:17], v[142:145], v146, v156 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[176:183], v[2:9], v[126:129], v146, v156 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[176:183], v[10:17], v[138:141], v146, v156 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[184:191], v[2:9], v[118:121], v148, v156 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[184:191], v[10:17], v[130:133], v148, v156 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], v[192:199], v[2:9], v[114:117], v148, v156 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[192:199], v[10:17], v[122:125], v148, v156 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b64_tr_b8 v[130:131], v165 offset:0
ds_read_b64_tr_b8 v[132:133], v165 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[138:139], v164 offset:0
ds_read_b64_tr_b8 v[140:141], v164 offset:1024

	;;#ASMEND
	s_nop 3
	v_add_u32_e32 v114, v163, v159
	;;#ASMSTART
	ds_read_b64_tr_b8 v[134:135], v114 offset:0
ds_read_b64_tr_b8 v[136:137], v114 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[142:143], v162 offset:0
ds_read_b64_tr_b8 v[144:145], v162 offset:1024

	;;#ASMEND
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[126:129], v[168:175], v[130:137], v[82:85], v146, v157 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[122:125], v[168:175], v[138:145], v[86:89], v146, v157 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[118:121], v[176:183], v[130:137], v[90:93], v146, v157 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[114:117], v[176:183], v[138:145], v[94:97], v146, v157 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[184:191], v[130:137], v[98:101], v148, v157 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[184:191], v[138:145], v[102:105], v148, v157 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[192:199], v[130:137], v[106:109], v148, v157 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[192:199], v[138:145], v[110:113], v148, v157 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[158:161], v155 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[166:169], v155 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[174:177], v155 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[182:185], v155 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[162:165], v153 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[170:173], v153 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[178:181], v153 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[186:189], v153 offset:0x1800

	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[98:101], v[158:165], v[2:9], v[70:73], v147, v156 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[158:165], v[10:17], v[78:81], v147, v156 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[166:173], v[2:9], v[62:65], v147, v156 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[166:173], v[10:17], v[74:77], v147, v156 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[106:109], v[174:181], v[2:9], v[54:57], v149, v156 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[174:181], v[10:17], v[66:69], v149, v156 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[110:113], v[182:189], v[2:9], v[50:53], v149, v156 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[102:105], v[182:189], v[10:17], v[58:61], v149, v156 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[158:165], v[130:137], v[216:219], v147, v157 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[158:165], v[138:145], v[220:223], v147, v157 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[166:173], v[130:137], v[232:235], v147, v157 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[166:173], v[138:145], v[236:239], v147, v157 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[174:181], v[130:137], v[240:243], v149, v157 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[174:181], v[138:145], v[200:203], v149, v157 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[6:9], v[182:189], v[130:137], v[204:207], v149, v157 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[2:5], v[182:189], v[138:145], v[208:211], v149, v157 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_movk_i32 s0, 0x100
	v_cmp_gt_u32_e32 vcc, s0, v0
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB3_10
; %bb.9:
	s_barrier
.LBB3_10:
	s_or_b64 exec, exec, s[0:1]
	v_lshl_or_b32 v185, s27, 2, v1
	v_lshrrev_b32_e32 v0, 2, v0
	v_or_b32_e32 v151, s28, v151
	v_mul_lo_u32 v1, v185, s14
	v_and_b32_e32 v0, 12, v0
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v181, s15, v2
	v_lshl_add_u32 v2, v1, 6, v151
	v_mad_u64_u32 v[0:1], s[0:1], v0, s14, v[150:151]
	v_ashrrev_i32_e32 v1, 31, v0
	v_mul_f32_e32 v183, s15, v4
	v_mul_f32_e32 v184, s15, v5
	v_lshlrev_b64 v[4:5], 1, v[0:1]
	v_add_u32_e32 v0, s14, v0
	v_ashrrev_i32_e32 v1, 31, v0
	v_mul_f32_e32 v179, s15, v8
	v_mul_f32_e32 v180, s15, v9
	v_lshlrev_b64 v[8:9], 1, v[0:1]
	v_add_u32_e32 v0, s14, v0
	v_ashrrev_i32_e32 v1, 31, v0
	v_mul_f32_e32 v175, s15, v12
	v_mul_f32_e32 v176, s15, v13
	v_lshlrev_b64 v[12:13], 1, v[0:1]
	v_add_u32_e32 v0, s14, v0
	v_mul_f32_e32 v182, s15, v3
	v_ashrrev_i32_e32 v3, 31, v2
	v_ashrrev_i32_e32 v1, 31, v0
	s_mul_i32 s0, s14, 13
	v_mul_f32_e32 v171, s15, v16
	v_mul_f32_e32 v172, s15, v17
	v_lshl_add_u64 v[2:3], v[2:3], 1, s[10:11]
	v_lshlrev_b64 v[16:17], 1, v[0:1]
	v_add_u32_e32 v0, s0, v0
	v_mul_f32_e32 v26, s15, v26
	v_mul_f32_e32 v27, s15, v27
	v_mul_f32_e32 v28, s15, v28
	v_mul_f32_e32 v29, s15, v29
	v_mul_f32_e32 v130, s15, v18
	v_mul_f32_e32 v131, s15, v19
	v_mul_f32_e32 v20, s15, v20
	v_mul_f32_e32 v21, s15, v21
	v_mul_f32_e32 v169, s15, v14
	v_mul_f32_e32 v170, s15, v15
	v_mul_f32_e32 v173, s15, v10
	v_mul_f32_e32 v174, s15, v11
	v_mul_f32_e32 v177, s15, v6
	v_mul_f32_e32 v178, s15, v7
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
	v_add_u32_e32 v0, s14, v0
	v_ashrrev_i32_e32 v1, 31, v0
	v_mul_f32_e32 v134, s15, v24
	v_mul_f32_e32 v135, s15, v25
	v_lshlrev_b64 v[24:25], 1, v[0:1]
	v_add_u32_e32 v0, s14, v0
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshlrev_b64 v[28:29], 1, v[0:1]
	v_add_u32_e32 v0, s14, v0
	v_ashrrev_i32_e32 v1, 31, v0
	v_mul_f32_e32 v34, s15, v34
	v_mul_f32_e32 v35, s15, v35
	v_mul_f32_e32 v132, s15, v22
	v_mul_f32_e32 v133, s15, v23
	v_mul_f32_e32 v141, s15, v32
	v_mul_f32_e32 v142, s15, v33
	v_lshl_add_u64 v[22:23], v[2:3], 0, v[20:21]
	v_lshl_add_u64 v[26:27], v[2:3], 0, v[24:25]
	v_lshlrev_b64 v[32:33], 1, v[0:1]
	v_add_u32_e32 v0, s0, v0
	v_mul_f32_e32 v36, s15, v36
	v_mul_f32_e32 v37, s15, v37
	v_mul_f32_e32 v139, s15, v30
	v_mul_f32_e32 v140, s15, v31
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
	v_add_u32_e32 v0, s14, v0
	v_ashrrev_i32_e32 v1, 31, v0
	v_mul_f32_e32 v149, s15, v40
	v_mul_f32_e32 v152, s15, v41
	v_lshlrev_b64 v[40:41], 1, v[0:1]
	v_add_u32_e32 v0, s14, v0
	v_ashrrev_i32_e32 v1, 31, v0
	v_mul_f32_e32 v137, s15, v44
	v_mul_f32_e32 v138, s15, v45
	v_lshlrev_b64 v[44:45], 1, v[0:1]
	v_add_u32_e32 v0, s14, v0
	v_ashrrev_i32_e32 v1, 31, v0
	v_mul_f32_e32 v145, s15, v48
	v_mul_f32_e32 v146, s15, v49
	v_lshlrev_b64 v[48:49], 1, v[0:1]
	v_add_u32_e32 v0, s0, v0
	v_ashrrev_i32_e32 v1, 31, v0
	v_mul_f32_e32 v167, s15, v52
	v_mul_f32_e32 v168, s15, v53
	v_lshlrev_b64 v[52:53], 1, v[0:1]
	v_add_u32_e32 v0, s14, v0
	v_ashrrev_i32_e32 v1, 31, v0
	v_mul_f32_e32 v163, s15, v56
	v_mul_f32_e32 v164, s15, v57
	v_lshlrev_b64 v[56:57], 1, v[0:1]
	v_add_u32_e32 v0, s14, v0
	v_ashrrev_i32_e32 v1, 31, v0
	v_mul_f32_e32 v159, s15, v60
	v_mul_f32_e32 v160, s15, v61
	v_lshlrev_b64 v[60:61], 1, v[0:1]
	v_add_u32_e32 v0, s14, v0
	v_ashrrev_i32_e32 v1, 31, v0
	v_mul_f32_e32 v42, s15, v42
	v_mul_f32_e32 v147, s15, v38
	v_mul_f32_e32 v148, s15, v39
	v_lshl_add_u64 v[38:39], v[2:3], 0, v[36:37]
	v_lshlrev_b64 v[0:1], 1, v[0:1]
	s_mov_b64 s[0:1], 0x100
	v_mul_f32_e32 v136, s15, v43
	v_mul_f32_e32 v143, s15, v46
	v_mul_f32_e32 v144, s15, v47
	v_mul_f32_e32 v126, s15, v126
	v_mul_f32_e32 v153, s15, v62
	v_mul_f32_e32 v154, s15, v63
	v_mul_f32_e32 v155, s15, v64
	v_mul_f32_e32 v156, s15, v65
	v_mul_f32_e32 v157, s15, v58
	v_mul_f32_e32 v158, s15, v59
	v_mul_f32_e32 v161, s15, v54
	v_mul_f32_e32 v162, s15, v55
	v_mul_f32_e32 v165, s15, v50
	v_mul_f32_e32 v166, s15, v51
	global_store_short_d16_hi v[38:39], v42, off
	v_lshl_add_u64 v[42:43], v[2:3], 0, v[40:41]
	v_lshl_add_u64 v[46:47], v[2:3], 0, v[44:45]
	v_lshl_add_u64 v[50:51], v[2:3], 0, v[48:49]
	v_lshl_add_u64 v[54:55], v[2:3], 0, v[52:53]
	v_lshl_add_u64 v[58:59], v[2:3], 0, v[56:57]
	v_lshl_add_u64 v[62:63], v[2:3], 0, v[60:61]
	v_lshl_add_u64 v[64:65], v[2:3], 0, v[0:1]
	v_lshl_add_u64 v[2:3], v[2:3], 0, s[0:1]
	v_mul_f32_e32 v127, s15, v127
	v_mul_f32_e32 v128, s15, v128
	v_mul_f32_e32 v129, s15, v129
	v_mul_f32_e32 v122, s15, v122
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
	global_store_short_d16_hi v[64:65], v152, off offset:32
	global_store_short_d16_hi v[6:7], v126, off offset:256
	global_store_short_d16_hi v[10:11], v127, off offset:256
	global_store_short_d16_hi v[14:15], v128, off offset:256
	global_store_short_d16_hi v[18:19], v129, off offset:256
	v_lshl_add_u64 v[6:7], v[2:3], 0, v[4:5]
	v_mul_f32_e32 v123, s15, v123
	global_store_short_d16_hi v[6:7], v122, off offset:32
	v_lshl_add_u64 v[6:7], v[2:3], 0, v[8:9]
	v_mul_f32_e32 v124, s15, v124
	global_store_short_d16_hi v[6:7], v123, off offset:32
	v_lshl_add_u64 v[6:7], v[2:3], 0, v[12:13]
	v_mul_f32_e32 v125, s15, v125
	global_store_short_d16_hi v[6:7], v124, off offset:32
	v_lshl_add_u64 v[6:7], v[2:3], 0, v[16:17]
	v_mul_f32_e32 v118, s15, v118
	v_mul_f32_e32 v119, s15, v119
	v_mul_f32_e32 v120, s15, v120
	v_mul_f32_e32 v121, s15, v121
	v_mul_f32_e32 v114, s15, v114
	global_store_short_d16_hi v[6:7], v125, off offset:32
	global_store_short_d16_hi v[22:23], v118, off offset:256
	global_store_short_d16_hi v[26:27], v119, off offset:256
	global_store_short_d16_hi v[30:31], v120, off offset:256
	global_store_short_d16_hi v[34:35], v121, off offset:256
	v_lshl_add_u64 v[6:7], v[2:3], 0, v[20:21]
	v_mul_f32_e32 v115, s15, v115
	global_store_short_d16_hi v[6:7], v114, off offset:32
	v_lshl_add_u64 v[6:7], v[2:3], 0, v[24:25]
	v_mul_f32_e32 v116, s15, v116
	global_store_short_d16_hi v[6:7], v115, off offset:32
	v_lshl_add_u64 v[6:7], v[2:3], 0, v[28:29]
	v_mul_f32_e32 v117, s15, v117
	global_store_short_d16_hi v[6:7], v116, off offset:32
	v_lshl_add_u64 v[6:7], v[2:3], 0, v[32:33]
	v_mul_f32_e32 v94, s15, v94
	v_mul_f32_e32 v95, s15, v95
	v_mul_f32_e32 v96, s15, v96
	v_mul_f32_e32 v97, s15, v97
	v_mul_f32_e32 v90, s15, v90
	global_store_short_d16_hi v[6:7], v117, off offset:32
	global_store_short_d16_hi v[38:39], v94, off offset:256
	global_store_short_d16_hi v[42:43], v95, off offset:256
	global_store_short_d16_hi v[46:47], v96, off offset:256
	global_store_short_d16_hi v[50:51], v97, off offset:256
	v_lshl_add_u64 v[6:7], v[2:3], 0, v[36:37]
	v_mul_f32_e32 v91, s15, v91
	global_store_short_d16_hi v[6:7], v90, off offset:32
	v_lshl_add_u64 v[6:7], v[2:3], 0, v[40:41]
	v_mul_f32_e32 v92, s15, v92
	global_store_short_d16_hi v[6:7], v91, off offset:32
	v_lshl_add_u64 v[6:7], v[2:3], 0, v[44:45]
	v_mul_f32_e32 v93, s15, v93
	global_store_short_d16_hi v[6:7], v92, off offset:32
	v_lshl_add_u64 v[6:7], v[2:3], 0, v[48:49]
	v_mul_f32_e32 v86, s15, v86
	v_mul_f32_e32 v87, s15, v87
	v_mul_f32_e32 v88, s15, v88
	v_mul_f32_e32 v89, s15, v89
	v_mul_f32_e32 v82, s15, v82
	global_store_short_d16_hi v[6:7], v93, off offset:32
	global_store_short_d16_hi v[54:55], v86, off offset:256
	global_store_short_d16_hi v[58:59], v87, off offset:256
	global_store_short_d16_hi v[62:63], v88, off offset:256
	global_store_short_d16_hi v[64:65], v89, off offset:256
	v_lshl_add_u64 v[6:7], v[2:3], 0, v[52:53]
	v_mul_f32_e32 v83, s15, v83
	global_store_short_d16_hi v[6:7], v82, off offset:32
	v_lshl_add_u64 v[6:7], v[2:3], 0, v[56:57]
	v_mul_f32_e32 v84, s15, v84
	v_mul_f32_e32 v85, s15, v85
	global_store_short_d16_hi v[6:7], v83, off offset:32
	v_lshl_add_u64 v[6:7], v[2:3], 0, v[60:61]
	v_lshl_add_u64 v[2:3], v[2:3], 0, v[0:1]
	global_store_short_d16_hi v[6:7], v84, off offset:32
	global_store_short_d16_hi v[2:3], v85, off offset:32
	v_or_b32_e32 v2, 2, v185
	v_mul_lo_u32 v2, v2, s14
	v_lshl_add_u32 v2, v2, 6, v151
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], v[2:3], 1, s[10:11]
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
	v_mul_f32_e32 v98, s15, v98
	v_mul_f32_e32 v99, s15, v99
	v_mul_f32_e32 v100, s15, v100
	v_mul_f32_e32 v101, s15, v101
	v_mul_f32_e32 v78, s15, v78
	v_mul_f32_e32 v79, s15, v79
	v_mul_f32_e32 v80, s15, v80
	v_mul_f32_e32 v81, s15, v81
	v_mul_f32_e32 v106, s15, v106
	v_mul_f32_e32 v107, s15, v107
	v_mul_f32_e32 v108, s15, v108
	v_mul_f32_e32 v109, s15, v109
	v_mul_f32_e32 v110, s15, v110
	v_mul_f32_e32 v111, s15, v111
	v_mul_f32_e32 v112, s15, v112
	v_mul_f32_e32 v113, s15, v113
	v_mul_f32_e32 v66, s15, v66
	v_lshl_add_u64 v[4:5], v[2:3], 0, v[4:5]
	v_mul_f32_e32 v70, s15, v70
	v_mul_f32_e32 v71, s15, v71
	v_mul_f32_e32 v72, s15, v72
	v_mul_f32_e32 v73, s15, v73
	v_mul_f32_e32 v74, s15, v74
	v_mul_f32_e32 v75, s15, v75
	v_mul_f32_e32 v76, s15, v76
	v_mul_f32_e32 v77, s15, v77
	v_mul_f32_e32 v102, s15, v102
	v_mul_f32_e32 v103, s15, v103
	v_mul_f32_e32 v104, s15, v104
	v_mul_f32_e32 v105, s15, v105
	v_mul_f32_e32 v67, s15, v67
	v_mul_f32_e32 v68, s15, v68
	v_mul_f32_e32 v69, s15, v69
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
	global_store_short_d16_hi v[22:23], v153, off offset:32
	global_store_short_d16_hi v[26:27], v154, off offset:32
	global_store_short_d16_hi v[30:31], v155, off offset:32
	global_store_short_d16_hi v[34:35], v156, off offset:32
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
	global_store_short_d16_hi v[4:5], v157, off offset:32
	v_lshl_add_u64 v[4:5], v[2:3], 0, v[8:9]
	global_store_short_d16_hi v[4:5], v158, off offset:32
	v_lshl_add_u64 v[4:5], v[2:3], 0, v[12:13]
	global_store_short_d16_hi v[4:5], v159, off offset:32
	v_lshl_add_u64 v[4:5], v[2:3], 0, v[16:17]
	global_store_short_d16_hi v[4:5], v160, off offset:32
	global_store_short_d16_hi v[22:23], v161, off offset:256
	global_store_short_d16_hi v[26:27], v162, off offset:256
	global_store_short_d16_hi v[30:31], v163, off offset:256
	global_store_short_d16_hi v[34:35], v164, off offset:256
	v_lshl_add_u64 v[4:5], v[2:3], 0, v[20:21]
	global_store_short_d16_hi v[4:5], v165, off offset:32
	v_lshl_add_u64 v[4:5], v[2:3], 0, v[24:25]
	global_store_short_d16_hi v[4:5], v166, off offset:32
	v_lshl_add_u64 v[4:5], v[2:3], 0, v[28:29]
	global_store_short_d16_hi v[4:5], v167, off offset:32
	v_lshl_add_u64 v[4:5], v[2:3], 0, v[32:33]
	global_store_short_d16_hi v[4:5], v168, off offset:32
	global_store_short_d16_hi v[38:39], v169, off offset:256
	global_store_short_d16_hi v[42:43], v170, off offset:256
	global_store_short_d16_hi v[46:47], v171, off offset:256
	global_store_short_d16_hi v[50:51], v172, off offset:256
	v_lshl_add_u64 v[4:5], v[2:3], 0, v[36:37]
	global_store_short_d16_hi v[4:5], v173, off offset:32
	v_lshl_add_u64 v[4:5], v[2:3], 0, v[40:41]
	global_store_short_d16_hi v[4:5], v174, off offset:32
	v_lshl_add_u64 v[4:5], v[2:3], 0, v[44:45]
	global_store_short_d16_hi v[4:5], v175, off offset:32
	v_lshl_add_u64 v[4:5], v[2:3], 0, v[48:49]
	global_store_short_d16_hi v[4:5], v176, off offset:32
	global_store_short_d16_hi v[54:55], v177, off offset:256
	global_store_short_d16_hi v[58:59], v178, off offset:256
	global_store_short_d16_hi v[62:63], v179, off offset:256
	global_store_short_d16_hi v[64:65], v180, off offset:256
	v_lshl_add_u64 v[4:5], v[2:3], 0, v[52:53]
	global_store_short_d16_hi v[4:5], v181, off offset:32
	v_lshl_add_u64 v[4:5], v[2:3], 0, v[56:57]
	global_store_short_d16_hi v[4:5], v182, off offset:32
	v_lshl_add_u64 v[4:5], v[2:3], 0, v[60:61]
	v_lshl_add_u64 v[0:1], v[2:3], 0, v[0:1]
	global_store_short_d16_hi v[4:5], v183, off offset:32
	global_store_short_d16_hi v[0:1], v184, off offset:32
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z29rrr_exact_8wave_scaled_kernelILb1ELi2EEv14layout_globals
		.amdhsa_group_segment_fixed_size 135168
		.amdhsa_private_segment_fixed_size 0
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
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 254
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
	.set _Z29rrr_exact_8wave_scaled_kernelILb1ELi2EEv14layout_globals.num_vgpr, 254
	.set _Z29rrr_exact_8wave_scaled_kernelILb1ELi2EEv14layout_globals.num_agpr, 0
	.set _Z29rrr_exact_8wave_scaled_kernelILb1ELi2EEv14layout_globals.numbered_sgpr, 40
	.set _Z29rrr_exact_8wave_scaled_kernelILb1ELi2EEv14layout_globals.private_seg_size, 0
	.set _Z29rrr_exact_8wave_scaled_kernelILb1ELi2EEv14layout_globals.uses_vcc, 1
	.set _Z29rrr_exact_8wave_scaled_kernelILb1ELi2EEv14layout_globals.uses_flat_scratch, 0
	.set _Z29rrr_exact_8wave_scaled_kernelILb1ELi2EEv14layout_globals.has_dyn_sized_stack, 0
	.set _Z29rrr_exact_8wave_scaled_kernelILb1ELi2EEv14layout_globals.has_recursion, 0
	.set _Z29rrr_exact_8wave_scaled_kernelILb1ELi2EEv14layout_globals.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 9772
; TotalNumSgprs: 46
; NumVgprs: 254
; NumAgprs: 0
; TotalNumVgprs: 254
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 135168 bytes/workgroup (compile time only)
; SGPRBlocks: 12
; VGPRBlocks: 31
; NumSGPRsForWavesPerEU: 102
; NumVGPRsForWavesPerEU: 254
; AccumOffset: 256
; Occupancy: 2
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
	.section	.text._Z29crr_exact_8wave_scaled_kernelILb1ELi2EEv14layout_globals,"axG",@progbits,_Z29crr_exact_8wave_scaled_kernelILb1ELi2EEv14layout_globals,comdat
	.protected	_Z29crr_exact_8wave_scaled_kernelILb1ELi2EEv14layout_globals ; -- Begin function _Z29crr_exact_8wave_scaled_kernelILb1ELi2EEv14layout_globals
	.globl	_Z29crr_exact_8wave_scaled_kernelILb1ELi2EEv14layout_globals
	.p2align	8
	.type	_Z29crr_exact_8wave_scaled_kernelILb1ELi2EEv14layout_globals,@function
