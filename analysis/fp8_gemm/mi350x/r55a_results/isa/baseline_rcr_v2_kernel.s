_Z29rcr_exact_8wave_scaled_kernelILb1ELi2EEv14layout_globals: ; @_Z29rcr_exact_8wave_scaled_kernelILb1ELi2EEv14layout_globals
; %bb.0:
	s_load_dwordx2 s[6:7], s[0:1], 0x60
	s_load_dwordx2 s[4:5], s[0:1], 0x90
	s_load_dword s3, s[0:1], 0x128
	s_load_dwordx2 s[28:29], s[0:1], 0x0
	s_load_dwordx2 s[30:31], s[0:1], 0x20
	s_load_dwordx2 s[34:35], s[0:1], 0x30
	s_load_dwordx2 s[8:9], s[0:1], 0x50
	s_waitcnt lgkmcnt(0)
	s_cmp_lt_i32 s3, 8
	s_cselect_b64 s[10:11], -1, 0
	s_and_b32 s9, s3, 7
	s_cmp_lg_u32 s9, 0
	s_cselect_b64 s[12:13], -1, 0
	s_or_b64 s[10:11], s[10:11], s[12:13]
	s_and_b64 vcc, exec, s[10:11]
	s_cbranch_vccnz .LBB2_2
; %bb.1:
	s_ashr_i32 s9, s2, 31
	s_lshr_b32 s9, s9, 29
	s_add_i32 s9, s2, s9
	s_ashr_i32 s10, s9, 3
	s_and_b32 s9, s9, -8
	s_lshr_b32 s3, s3, 3
	s_sub_i32 s2, s2, s9
	s_mul_i32 s2, s3, s2
	s_add_i32 s2, s2, s10
.LBB2_2:
	s_ashr_i32 s9, s2, 31
	s_lshr_b32 s9, s9, 25
	s_add_i32 s9, s2, s9
	s_ashr_i32 s12, s9, 7
	s_load_dword s3, s[0:1], 0x108
	s_lshl_b32 s9, s12, 2
	s_sub_i32 s10, 16, s9
	s_cmpk_gt_i32 s2, 0x1ff
	s_cselect_b32 s11, s10, 4
	s_mov_b32 s31, 16
	s_cmp_lt_i32 s11, 1
	s_mov_b32 s10, 32
	s_cbranch_scc1 .LBB2_4
; %bb.3:
	s_abs_i32 s10, s11
	v_cvt_f32_u32_e32 v1, s10
	s_lshl_b32 s12, s12, 7
	s_sub_i32 s2, s2, s12
	s_sub_i32 s12, 0, s10
	v_rcp_iflag_f32_e32 v1, v1
	s_abs_i32 s14, s2
	s_xor_b32 s13, s2, s11
	s_ashr_i32 s13, s13, 31
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	s_nop 0
	v_readfirstlane_b32 s15, v1
	s_mul_i32 s12, s12, s15
	s_mul_hi_u32 s12, s15, s12
	s_add_i32 s15, s15, s12
	s_mul_hi_u32 s12, s14, s15
	s_mul_i32 s15, s12, s10
	s_sub_i32 s14, s14, s15
	s_add_i32 s16, s12, 1
	s_sub_i32 s15, s14, s10
	s_cmp_ge_u32 s14, s10
	s_cselect_b32 s12, s16, s12
	s_cselect_b32 s14, s15, s14
	s_add_i32 s15, s12, 1
	s_cmp_ge_u32 s14, s10
	s_cselect_b32 s10, s15, s12
	s_xor_b32 s10, s10, s13
	s_sub_i32 s10, s10, s13
	s_mul_i32 s11, s10, s11
	s_sub_i32 s2, s2, s11
	s_add_i32 s31, s2, s9
.LBB2_4:                                ; %.critedge
	s_waitcnt lgkmcnt(0)
	s_ashr_i32 s2, s3, 31
	s_lshr_b32 s2, s2, 27
	s_add_i32 s3, s3, s2
	s_ashr_i32 s2, s3, 5
	v_lshlrev_b32_e32 v1, 4, v0
	s_movk_i32 s36, 0x70
	v_lshrrev_b32_e32 v3, 3, v0
	s_lshl_b32 s33, s10, 8
	s_add_i32 s2, s2, 7
	v_bitop3_b32 v2, v1, s36, v0 bitop3:0x48
	v_or_b32_e32 v4, 64, v3
	s_and_b32 s2, s2, -8
	s_ashr_i32 s11, s10, 31
	s_mul_i32 s37, s33, s8
	v_mad_u64_u32 v[150:151], s[12:13], v4, s30, v[2:3]
	s_lshl_b32 s39, s2, 7
	s_lshl_b32 s38, s2, 6
	s_ashr_i32 s20, s31, 31
	s_lshl_b64 s[2:3], s[10:11], 2
	s_ashr_i32 s40, s37, 31
	v_mad_u64_u32 v[152:153], s[12:13], v3, s30, v[2:3]
	s_add_u32 s16, s34, s37
	v_and_b32_e32 v151, 0x1c00, v1
	v_mad_u64_u32 v[154:155], s[12:13], v3, s8, v[2:3]
	s_addc_u32 s17, s35, s40
	v_or_b32_e32 v153, 0x10000, v151
	s_lshl_b32 s41, s31, 8
	v_mad_u64_u32 v[156:157], s[12:13], v4, s8, v[2:3]
	s_lshl_b32 s18, s8, 7
	s_mov_b32 s11, 0x110000
	v_readfirstlane_b32 s8, v153
	v_or_b32_e32 v155, 0x2000, v153
	s_mul_i32 s42, s41, s30
	s_mov_b32 s19, s11
	s_mov_b32 m0, s8
	v_readfirstlane_b32 s8, v155
	s_ashr_i32 s43, s42, 31
	buffer_load_dwordx4 v154, s[16:19], 0 offen lds
	s_mov_b32 m0, s8
	s_add_u32 s8, s28, s42
	s_addc_u32 s9, s29, s43
	v_readfirstlane_b32 s12, v151
	v_or_b32_e32 v157, 0x2000, v151
	s_add_i32 s44, s37, s18
	buffer_load_dwordx4 v156, s[16:19], 0 offen lds
	s_lshl_b32 s10, s30, 7
	s_mov_b32 m0, s12
	v_readfirstlane_b32 s12, v157
	s_ashr_i32 s45, s44, 31
	v_or_b32_e32 v160, 0x4000, v153
	buffer_load_dwordx4 v152, s[8:11], 0 offen lds
	s_mov_b32 m0, s12
	s_add_u32 s24, s34, s44
	v_readfirstlane_b32 s12, v160
	v_or_b32_e32 v161, 0x6000, v153
	buffer_load_dwordx4 v150, s[8:11], 0 offen lds
	s_addc_u32 s25, s35, s45
	s_mov_b32 s26, s18
	s_mov_b32 s27, s11
	s_mov_b32 m0, s12
	v_readfirstlane_b32 s12, v161
	buffer_load_dwordx4 v154, s[24:27], 0 offen lds
	s_mov_b32 m0, s12
	s_add_i32 s12, s42, s10
	s_ashr_i32 s13, s12, 31
	v_or_b32_e32 v162, 0x4000, v151
	s_add_u32 s12, s28, s12
	v_readfirstlane_b32 s19, v162
	v_or_b32_e32 v163, 0x6000, v151
	buffer_load_dwordx4 v156, s[24:27], 0 offen lds
	s_addc_u32 s13, s29, s13
	s_mov_b32 s14, s10
	s_mov_b32 s15, s11
	s_mov_b32 m0, s19
	v_readfirstlane_b32 s19, v163
	buffer_load_dwordx4 v152, s[12:15], 0 offen lds
	s_mov_b32 m0, s19
	v_lshrrev_b32_e32 v1, 8, v0
	buffer_load_dwordx4 v150, s[12:15], 0 offen lds
	s_load_dwordx2 s[14:15], s[0:1], 0xc0
	s_load_dwordx2 s[26:27], s[0:1], 0xe0
	v_bfe_u32 v207, v0, 6, 2
	v_lshl_or_b32 v4, s31, 1, v1
	v_mov_b64_e32 v[2:3], s[6:7]
	v_or_b32_e32 v6, s2, v207
	v_mad_u64_u32 v[2:3], s[6:7], v4, s39, v[2:3]
	v_mov_b64_e32 v[4:5], s[4:5]
	s_mul_i32 s20, s20, s39
	s_mul_i32 s6, s3, s38
	v_mad_u64_u32 v[4:5], s[2:3], v6, s38, v[4:5]
	v_add_u32_e32 v3, s20, v3
	v_add_u32_e32 v5, s6, v5
	v_readfirstlane_b32 s21, v2
	v_readfirstlane_b32 s20, v3
	v_readfirstlane_b32 s46, v4
	v_readfirstlane_b32 s19, v5
	s_waitcnt lgkmcnt(0)
	s_mov_b32 s27, 0
	v_cmp_eq_u32_e32 vcc, 1, v1
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB2_6
; %bb.5:
	s_barrier
.LBB2_6:
	s_or_b64 exec, exec, s[2:3]
	s_load_dwordx4 s[4:7], s[0:1], 0xf4
	s_mov_b64 s[0:1], s[8:9]
	s_mov_b64 s[2:3], s[10:11]
	s_mov_b32 s0, s21
	s_mov_b32 s1, s20
	s_mov_b64 s[22:23], s[10:11]
	v_or_b32_e32 v167, 0x8000, v153
	s_mov_b64 s[20:21], s[8:9]
	s_add_u32 s16, s16, 0x80
	v_readfirstlane_b32 s2, v167
	v_or_b32_e32 v168, 0xa000, v153
	s_mov_b32 s21, s19
	s_addc_u32 s17, s17, 0
	s_mov_b32 s19, s11
	s_mov_b32 m0, s2
	v_readfirstlane_b32 s2, v168
	v_or_b32_e32 v170, 0x8000, v151
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	buffer_load_dwordx4 v154, s[16:19], 0 offen lds
	s_mov_b32 m0, s2
	s_add_u32 s8, s8, 0x80
	v_readfirstlane_b32 s2, v170
	v_or_b32_e32 v171, 0xa000, v151
	buffer_load_dwordx4 v156, s[16:19], 0 offen lds
	s_addc_u32 s9, s9, 0
	s_mov_b32 m0, s2
	v_readfirstlane_b32 s2, v171
	v_or_b32_e32 v172, 0xc000, v153
	buffer_load_dwordx4 v152, s[8:11], 0 offen lds
	s_mov_b32 m0, s2
	s_add_u32 s16, s24, 0x80
	v_readfirstlane_b32 s2, v172
	v_or_b32_e32 v180, 0xe000, v153
	buffer_load_dwordx4 v150, s[8:11], 0 offen lds
	s_addc_u32 s17, s25, 0
	s_mov_b32 m0, s2
	v_readfirstlane_b32 s2, v180
	buffer_load_dwordx4 v154, s[16:19], 0 offen lds
	s_mov_b32 m0, s2
	v_and_b32_e32 v206, 15, v0
	buffer_load_dwordx4 v156, s[16:19], 0 offen lds
	v_bfe_u32 v2, v0, 4, 2
	v_lshlrev_b32_e32 v3, 4, v206
	v_lshl_or_b32 v181, v2, 8, v3
	v_lshlrev_b32_e32 v3, 3, v206
	v_lshl_or_b32 v182, v2, 7, v3
	v_and_b32_e32 v2, 48, v0
	v_lshlrev_b32_e32 v3, 7, v0
	s_movk_i32 s2, 0x780
	v_and_or_b32 v2, v3, s2, v2
	v_lshl_or_b32 v3, v207, 12, v2
	v_or_b32_e32 v4, 0x10000, v3
	v_lshlrev_b32_e32 v5, 3, v0
	v_bitop3_b32 v183, v5, v4, s36 bitop3:0x6c
	v_or_b32_e32 v4, 0x10040, v3
	v_lshl_or_b32 v2, v1, 13, v2
	v_bitop3_b32 v173, v5, v4, s36 bitop3:0x6c
	v_or_b32_e32 v4, 64, v2
	v_bitop3_b32 v166, v5, v4, s36 bitop3:0x6c
	v_or_b32_e32 v4, 0x14000, v3
	v_bitop3_b32 v179, v5, v4, s36 bitop3:0x6c
	v_or_b32_e32 v4, 0x14040, v3
	v_bitop3_b32 v178, v5, v4, s36 bitop3:0x6c
	v_or_b32_e32 v4, 0x4000, v2
	v_bitop3_b32 v177, v5, v4, s36 bitop3:0x6c
	v_or_b32_e32 v4, 0x4040, v2
	v_bitop3_b32 v176, v5, v4, s36 bitop3:0x6c
	v_or_b32_e32 v4, 0x18000, v3
	v_bitop3_b32 v175, v5, v4, s36 bitop3:0x6c
	v_or_b32_e32 v4, 0x18040, v3
	v_bitop3_b32 v174, v5, v4, s36 bitop3:0x6c
	v_or_b32_e32 v4, 0x8000, v2
	v_bitop3_b32 v199, v5, v4, s36 bitop3:0x6c
	v_or_b32_e32 v4, 0x8040, v2
	v_bitop3_b32 v198, v5, v4, s36 bitop3:0x6c
	v_or_b32_e32 v4, 0x1c000, v3
	v_or_b32_e32 v3, 0x1c040, v3
	s_add_u32 s24, s28, s42
	v_bitop3_b32 v169, v5, v2, s36 bitop3:0x6c
	v_bitop3_b32 v212, v5, v3, s36 bitop3:0x6c
	v_or_b32_e32 v3, 0xc000, v2
	v_or_b32_e32 v2, 0xc040, v2
	s_addc_u32 s25, s29, s43
	v_bitop3_b32 v213, v5, v4, s36 bitop3:0x6c
	v_bitop3_b32 v211, v5, v3, s36 bitop3:0x6c
	v_bitop3_b32 v210, v5, v2, s36 bitop3:0x6c
	s_add_u32 s36, s34, s37
	s_addc_u32 s37, s35, s40
	s_add_u32 s34, s34, s44
	s_addc_u32 s35, s35, s45
	s_add_i32 s2, s41, 0x80
	v_mov_b32_e32 v18, 0
	s_mov_b32 s20, s46
	v_or_b32_e32 v165, 0xc000, v151
	v_or_b32_e32 v164, 0xe000, v151
	s_mul_i32 s30, s2, s30
	s_waitcnt lgkmcnt(0)
	s_mov_b64 s[6:7], 0
	s_mov_b32 s2, s39
	s_mov_b32 s22, s38
	s_mov_b32 s38, 0
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
	v_mov_b32_e32 v90, v18
	v_mov_b32_e32 v91, v18
	v_mov_b32_e32 v92, v18
	v_mov_b32_e32 v93, v18
	v_mov_b32_e32 v82, v18
	v_mov_b32_e32 v83, v18
	v_mov_b32_e32 v84, v18
	v_mov_b32_e32 v85, v18
	v_mov_b32_e32 v86, v18
	v_mov_b32_e32 v87, v18
	v_mov_b32_e32 v88, v18
	v_mov_b32_e32 v89, v18
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
	v_mov_b32_e32 v114, v18
	v_mov_b32_e32 v115, v18
	v_mov_b32_e32 v116, v18
	v_mov_b32_e32 v117, v18
	v_mov_b32_e32 v118, v18
	v_mov_b32_e32 v119, v18
	v_mov_b32_e32 v120, v18
	v_mov_b32_e32 v121, v18
	v_mov_b32_e32 v142, v18
	v_mov_b32_e32 v143, v18
	v_mov_b32_e32 v144, v18
	v_mov_b32_e32 v145, v18
	v_mov_b32_e32 v138, v18
	v_mov_b32_e32 v139, v18
	v_mov_b32_e32 v140, v18
	v_mov_b32_e32 v141, v18
	v_mov_b32_e32 v126, v18
	v_mov_b32_e32 v127, v18
	v_mov_b32_e32 v128, v18
	v_mov_b32_e32 v129, v18
	v_mov_b32_e32 v134, v18
	v_mov_b32_e32 v135, v18
	v_mov_b32_e32 v136, v18
	v_mov_b32_e32 v137, v18
	v_mov_b32_e32 v122, v18
	v_mov_b32_e32 v123, v18
	v_mov_b32_e32 v124, v18
	v_mov_b32_e32 v125, v18
	v_mov_b32_e32 v130, v18
	v_mov_b32_e32 v131, v18
	v_mov_b32_e32 v132, v18
	v_mov_b32_e32 v133, v18
	;;#ASMSTART
	s_waitcnt vmcnt(6)
	;;#ASMEND
	s_barrier
.LBB2_7:                                ; =>This Inner Loop Header: Depth=1
	buffer_load_dwordx4 v[146:149], v181, s[0:3], s38 offen
	buffer_load_dwordx2 v[158:159], v182, s[20:23], s27 offen
	;;#ASMSTART
	ds_read_b128 v[2:5], v183 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[10:13], v183 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[6:9], v173 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[14:17], v173 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[184:187], v169 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[214:217], v169 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[222:225], v169 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[230:233], v169 offset:0x1800

	;;#ASMEND
	s_add_i32 s39, s30, s6
	;;#ASMSTART
	ds_read_b128 v[188:191], v166 offset:0

	;;#ASMEND
	s_add_i32 s8, s39, 0x80
	;;#ASMSTART
	ds_read_b128 v[218:221], v166 offset:0x800

	;;#ASMEND
	s_ashr_i32 s9, s8, 31
	;;#ASMSTART
	ds_read_b128 v[226:229], v166 offset:0x1000

	;;#ASMEND
	s_add_u32 s8, s28, s8
	v_readfirstlane_b32 s16, v165
	;;#ASMSTART
	ds_read_b128 v[234:237], v166 offset:0x1800

	;;#ASMEND
	s_addc_u32 s9, s29, s9
	s_mov_b32 m0, s16
	v_readfirstlane_b32 s16, v164
	buffer_load_dwordx4 v152, s[8:11], 0 offen lds
	s_mov_b32 m0, s16
	s_nop 0
	buffer_load_dwordx4 v150, s[8:11], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	s_waitcnt vmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[142:145], v[184:191], v[2:9], v[142:145], v146, v158 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[138:141], v[184:191], v[10:17], v[138:141], v146, v158 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[126:129], v[214:221], v[2:9], v[126:129], v146, v158 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[134:137], v[214:221], v[10:17], v[134:137], v146, v158 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[122:125], v[222:229], v[2:9], v[122:125], v148, v158 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[130:133], v[222:229], v[10:17], v[130:133], v148, v158 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[118:121], v[230:237], v[2:9], v[118:121], v148, v158 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[114:117], v[230:237], v[10:17], v[114:117], v148, v158 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b128 v[238:241], v179 offset:0

	;;#ASMEND
	s_add_u32 s40, s36, s6
	;;#ASMSTART
	ds_read_b128 v[246:249], v179 offset:0x800

	;;#ASMEND
	s_addc_u32 s41, s37, s7
	;;#ASMSTART
	ds_read_b128 v[242:245], v178 offset:0

	;;#ASMEND
	s_add_u32 s16, s40, 0x100
	v_readfirstlane_b32 s8, v153
	;;#ASMSTART
	ds_read_b128 v[250:253], v178 offset:0x800

	;;#ASMEND
	s_addc_u32 s17, s41, 0
	s_mov_b32 s19, s11
	s_mov_b32 m0, s8
	v_readfirstlane_b32 s8, v155
	buffer_load_dwordx4 v154, s[16:19], 0 offen lds
	s_mov_b32 m0, s8
	s_nop 0
	buffer_load_dwordx4 v156, s[16:19], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[184:191], v[238:245], v[90:93], v146, v159 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[184:191], v[246:253], v[82:85], v146, v159 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[214:221], v[238:245], v[86:89], v146, v159 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[214:221], v[246:253], v[94:97], v146, v159 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[98:101], v[222:229], v[238:245], v[98:101], v148, v159 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[102:105], v[222:229], v[246:253], v[102:105], v148, v159 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[106:109], v[230:237], v[238:245], v[106:109], v148, v159 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[110:113], v[230:237], v[246:253], v[110:113], v148, v159 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[184:187], v177 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[214:217], v177 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[222:225], v177 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[230:233], v177 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[188:191], v176 offset:0

	;;#ASMEND
	s_add_u32 s42, s24, s6
	;;#ASMSTART
	ds_read_b128 v[218:221], v176 offset:0x800

	;;#ASMEND
	s_addc_u32 s43, s25, s7
	;;#ASMSTART
	ds_read_b128 v[226:229], v176 offset:0x1000

	;;#ASMEND
	s_add_u32 s8, s42, 0x100
	v_readfirstlane_b32 s16, v151
	;;#ASMSTART
	ds_read_b128 v[234:237], v176 offset:0x1800

	;;#ASMEND
	s_addc_u32 s9, s43, 0
	s_mov_b32 m0, s16
	v_readfirstlane_b32 s16, v157
	buffer_load_dwordx4 v152, s[8:11], 0 offen lds
	s_mov_b32 m0, s16
	s_nop 0
	buffer_load_dwordx4 v150, s[8:11], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[184:191], v[2:9], v[70:73], v147, v158 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[184:191], v[10:17], v[78:81], v147, v158 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[214:221], v[2:9], v[62:65], v147, v158 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[214:221], v[10:17], v[74:77], v147, v158 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[222:229], v[2:9], v[54:57], v149, v158 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[222:229], v[10:17], v[66:69], v149, v158 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[230:237], v[2:9], v[50:53], v149, v158 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[230:237], v[10:17], v[58:61], v149, v158 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_u32 s44, s34, s6
	s_addc_u32 s45, s35, s7
	s_add_u32 s16, s44, 0x100
	v_readfirstlane_b32 s8, v160
	s_addc_u32 s17, s45, 0
	s_mov_b32 m0, s8
	v_readfirstlane_b32 s8, v161
	buffer_load_dwordx4 v154, s[16:19], 0 offen lds
	s_mov_b32 m0, s8
	s_nop 0
	buffer_load_dwordx4 v156, s[16:19], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(6)
	;;#ASMEND
	s_barrier
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[184:191], v[238:245], v[18:21], v147, v159 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[184:191], v[246:253], v[22:25], v147, v159 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[214:221], v[238:245], v[26:29], v147, v159 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[214:221], v[246:253], v[30:33], v147, v159 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[222:229], v[238:245], v[34:37], v149, v159 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[222:229], v[246:253], v[38:41], v149, v159 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[230:237], v[238:245], v[42:45], v149, v159 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], v[230:237], v[246:253], v[46:49], v149, v159 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[2:5], v175 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[10:13], v175 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[6:9], v174 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[14:17], v174 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[184:187], v199 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[214:217], v199 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[222:225], v199 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[230:233], v199 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[188:191], v198 offset:0

	;;#ASMEND
	s_addk_i32 s39, 0x100
	;;#ASMSTART
	ds_read_b128 v[218:221], v198 offset:0x800

	;;#ASMEND
	s_ashr_i32 s9, s39, 31
	;;#ASMSTART
	ds_read_b128 v[226:229], v198 offset:0x1000

	;;#ASMEND
	s_add_u32 s8, s28, s39
	v_readfirstlane_b32 s16, v162
	;;#ASMSTART
	ds_read_b128 v[234:237], v198 offset:0x1800

	;;#ASMEND
	s_addc_u32 s9, s29, s9
	s_mov_b32 m0, s16
	v_readfirstlane_b32 s16, v163
	buffer_load_dwordx4 v152, s[8:11], 0 offen lds
	s_mov_b32 m0, s16
	s_nop 0
	buffer_load_dwordx4 v150, s[8:11], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[142:145], v[184:191], v[2:9], v[142:145], v146, v158 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[138:141], v[184:191], v[10:17], v[138:141], v146, v158 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[126:129], v[214:221], v[2:9], v[126:129], v146, v158 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[134:137], v[214:221], v[10:17], v[134:137], v146, v158 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[122:125], v[222:229], v[2:9], v[122:125], v148, v158 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[130:133], v[222:229], v[10:17], v[130:133], v148, v158 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[118:121], v[230:237], v[2:9], v[118:121], v148, v158 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[114:117], v[230:237], v[10:17], v[114:117], v148, v158 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b128 v[238:241], v213 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[246:249], v213 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[242:245], v212 offset:0

	;;#ASMEND
	s_add_u32 s16, s40, 0x180
	v_readfirstlane_b32 s8, v167
	;;#ASMSTART
	ds_read_b128 v[250:253], v212 offset:0x800

	;;#ASMEND
	s_addc_u32 s17, s41, 0
	s_mov_b32 m0, s8
	v_readfirstlane_b32 s8, v168
	buffer_load_dwordx4 v154, s[16:19], 0 offen lds
	s_mov_b32 m0, s8
	s_nop 0
	buffer_load_dwordx4 v156, s[16:19], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[184:191], v[238:245], v[90:93], v146, v159 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[184:191], v[246:253], v[82:85], v146, v159 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[214:221], v[238:245], v[86:89], v146, v159 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[214:221], v[246:253], v[94:97], v146, v159 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[98:101], v[222:229], v[238:245], v[98:101], v148, v159 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[102:105], v[222:229], v[246:253], v[102:105], v148, v159 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[106:109], v[230:237], v[238:245], v[106:109], v148, v159 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[110:113], v[230:237], v[246:253], v[110:113], v148, v159 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[184:187], v211 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[214:217], v211 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[222:225], v211 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[230:233], v211 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[188:191], v210 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[218:221], v210 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[226:229], v210 offset:0x1000

	;;#ASMEND
	s_add_u32 s8, s42, 0x180
	v_readfirstlane_b32 s16, v170
	;;#ASMSTART
	ds_read_b128 v[234:237], v210 offset:0x1800

	;;#ASMEND
	s_addc_u32 s9, s43, 0
	s_mov_b32 m0, s16
	v_readfirstlane_b32 s16, v171
	buffer_load_dwordx4 v152, s[8:11], 0 offen lds
	s_mov_b32 m0, s16
	s_nop 0
	buffer_load_dwordx4 v150, s[8:11], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[184:191], v[2:9], v[70:73], v147, v158 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[184:191], v[10:17], v[78:81], v147, v158 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[214:221], v[2:9], v[62:65], v147, v158 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[214:221], v[10:17], v[74:77], v147, v158 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[222:229], v[2:9], v[54:57], v149, v158 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[222:229], v[10:17], v[66:69], v149, v158 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[230:237], v[2:9], v[50:53], v149, v158 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[230:237], v[10:17], v[58:61], v149, v158 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_u32 s16, s44, 0x180
	v_readfirstlane_b32 s8, v172
	s_addc_u32 s17, s45, 0
	s_mov_b32 m0, s8
	v_readfirstlane_b32 s8, v180
	buffer_load_dwordx4 v154, s[16:19], 0 offen lds
	s_mov_b32 m0, s8
	s_nop 0
	buffer_load_dwordx4 v156, s[16:19], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(6)
	;;#ASMEND
	s_barrier
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[184:191], v[238:245], v[18:21], v147, v159 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[184:191], v[246:253], v[22:25], v147, v159 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[214:221], v[238:245], v[26:29], v147, v159 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[214:221], v[246:253], v[30:33], v147, v159 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[222:229], v[238:245], v[34:37], v149, v159 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[222:229], v[246:253], v[38:41], v149, v159 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[230:237], v[238:245], v[42:45], v149, v159 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], v[230:237], v[246:253], v[46:49], v149, v159 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_add_u32 s6, s6, 0x100
	s_addc_u32 s7, s7, 0
	s_addk_i32 s38, 0x400
	s_addk_i32 s27, 0x200
	s_cmpk_eq_i32 s6, 0x6f00
	s_barrier
	s_cbranch_scc0 .LBB2_7
; %bb.8:                                ; %_ZZ29rcr_exact_8wave_scaled_kernelILb1ELi2EEv14layout_globalsENKUliE2_clEi.exit
	s_mov_b32 s6, 0x1bc00
	buffer_load_dwordx4 v[146:149], v181, s[0:3], s6 offen
	s_mov_b32 s0, 0xde00
	buffer_load_dwordx2 v[208:209], v182, s[20:23], s0 offen
	;;#ASMSTART
	ds_read_b128 v[2:5], v183 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[10:13], v183 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[6:9], v173 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[14:17], v173 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[180:183], v169 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[188:191], v169 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[214:217], v169 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[222:225], v169 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[184:187], v166 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[192:195], v166 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[218:221], v166 offset:0x1000

	;;#ASMEND
	s_add_u32 s8, s12, 0x6f80
	v_readfirstlane_b32 s0, v165
	;;#ASMSTART
	ds_read_b128 v[226:229], v166 offset:0x1800

	;;#ASMEND
	s_addc_u32 s9, s13, 0
	s_mov_b32 s11, 0x110000
	s_mov_b32 m0, s0
	v_readfirstlane_b32 s0, v164
	buffer_load_dwordx4 v152, s[8:11], 0 offen lds
	s_mov_b32 m0, s0
	s_nop 0
	buffer_load_dwordx4 v150, s[8:11], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	s_waitcnt vmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[158:161], v[180:187], v[2:9], v[142:145], v146, v208 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[162:165], v[180:187], v[10:17], v[138:141], v146, v208 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[166:169], v[188:195], v[2:9], v[126:129], v146, v208 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[138:141], v[214:221], v[2:9], v[122:125], v148, v208 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[142:145], v[214:221], v[10:17], v[130:133], v148, v208 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[150:153], v[222:229], v[2:9], v[118:121], v148, v208 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[170:173], v[188:195], v[10:17], v[134:137], v146, v208 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[154:157], v[222:229], v[10:17], v[114:117], v148, v208 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b128 v[230:233], v179 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[238:241], v179 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[234:237], v178 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[242:245], v178 offset:0x800

	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[180:187], v[230:237], v[90:93], v146, v209 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[180:187], v[238:245], v[82:85], v146, v209 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[188:195], v[230:237], v[86:89], v146, v209 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[118:121], v[188:195], v[238:245], v[94:97], v146, v209 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[122:125], v[214:221], v[230:237], v[98:101], v148, v209 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[126:129], v[214:221], v[238:245], v[102:105], v148, v209 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[130:133], v[222:229], v[230:237], v[106:109], v148, v209 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[134:137], v[222:229], v[238:245], v[110:113], v148, v209 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[178:181], v177 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[186:189], v177 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[214:217], v177 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[222:225], v177 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[182:185], v176 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[190:193], v176 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[218:221], v176 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[226:229], v176 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[186:193], v[2:9], v[62:65], v147, v208 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[214:221], v[2:9], v[54:57], v149, v208 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[222:229], v[2:9], v[50:53], v149, v208 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[106:109], v[178:185], v[2:9], v[70:73], v147, v208 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[110:113], v[178:185], v[10:17], v[78:81], v147, v208 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[114:117], v[186:193], v[10:17], v[74:77], v147, v208 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[98:101], v[214:221], v[10:17], v[66:69], v149, v208 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[102:105], v[222:229], v[10:17], v[58:61], v149, v208 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[2:5], v175 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[10:13], v175 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[6:9], v174 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[14:17], v174 offset:0x800

	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[178:185], v[230:237], v[18:21], v147, v209 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[178:185], v[238:245], v[22:25], v147, v209 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[174:177], v[186:193], v[230:237], v[26:29], v147, v209 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[178:181], v[186:193], v[238:245], v[30:33], v147, v209 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[182:185], v[214:221], v[230:237], v[34:37], v149, v209 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[186:189], v[214:221], v[238:245], v[38:41], v149, v209 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[190:193], v[222:229], v[230:237], v[42:45], v149, v209 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[194:197], v[222:229], v[238:245], v[46:49], v149, v209 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b128 v[26:29], v199 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[34:37], v199 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[42:45], v199 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[214:217], v199 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[30:33], v198 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[38:41], v198 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[46:49], v198 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[218:221], v198 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_lshrrev_b32_e32 v58, 16, v146
	v_lshrrev_b32_e32 v146, 16, v208
	v_lshrrev_b32_e32 v59, 16, v148
	s_nop 0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[158:161], v[26:33], v[2:9], v[158:161], v58, v146 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[198:201], v[26:33], v[10:17], v[162:165], v58, v146 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[202:205], v[34:41], v[2:9], v[166:169], v58, v146 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[162:165], v[34:41], v[10:17], v[170:173], v58, v146 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[166:169], v[42:49], v[2:9], v[138:141], v59, v146 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[142:145], v[42:49], v[10:17], v[142:145], v59, v146 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[150:153], v[214:221], v[2:9], v[150:153], v59, v146 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[138:141], v[214:221], v[10:17], v[154:157], v59, v146 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[222:225], v213 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[230:233], v213 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[226:229], v212 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[234:237], v212 offset:0x800

	;;#ASMEND
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_lshrrev_b32_e32 v148, 16, v209
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[26:33], v[222:229], v[90:93], v58, v148 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[26:33], v[230:237], v[82:85], v58, v148 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[34:41], v[222:229], v[86:89], v58, v148 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[34:41], v[230:237], v[118:121], v58, v148 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[42:49], v[222:229], v[122:125], v59, v148 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[42:49], v[230:237], v[126:129], v59, v148 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[214:221], v[222:229], v[130:133], v59, v148 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[214:221], v[230:237], v[134:137], v59, v148 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[118:121], v211 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[126:129], v211 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[212:215], v211 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[238:241], v211 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[122:125], v210 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[130:133], v210 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[216:219], v210 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[242:245], v210 offset:0x1800

	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	s_nop 2
	v_lshrrev_b32_e32 v134, 16, v147
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[118:125], v[2:9], v[106:109], v134, v146 op_sel_hi:[0,0,0]
	s_nop 6
	v_lshrrev_b32_e32 v106, 16, v149
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], v[118:125], v[10:17], v[110:113], v134, v146 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[126:133], v[2:9], v[62:65], v134, v146 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[126:133], v[10:17], v[114:117], v134, v146 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[212:219], v[2:9], v[54:57], v106, v146 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[212:219], v[10:17], v[98:101], v106, v146 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[238:245], v[2:9], v[50:53], v106, v146 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[238:245], v[10:17], v[102:105], v106, v146 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[118:125], v[222:229], v[18:21], v134, v148 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[118:125], v[230:237], v[22:25], v134, v148 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[126:133], v[222:229], v[174:177], v134, v148 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[126:133], v[230:237], v[178:181], v134, v148 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[212:219], v[222:229], v[182:185], v106, v148 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[212:219], v[230:237], v[186:189], v106, v148 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[6:9], v[238:245], v[222:229], v[190:193], v106, v148 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[2:5], v[238:245], v[230:237], v[194:197], v106, v148 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_movk_i32 s0, 0x100
	v_cmp_gt_u32_e32 vcc, s0, v0
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB2_10
; %bb.9:
	s_barrier
.LBB2_10:
	s_or_b64 exec, exec, s[0:1]
	v_pk_mul_f32 v[120:121], v[168:169], s[4:5] op_sel_hi:[1,0]
	v_lshl_or_b32 v168, s31, 2, v1
	v_lshl_or_b32 v169, v207, 5, s33
	v_mul_lo_u32 v1, v168, s26
	v_lshl_add_u32 v98, v1, 6, v169
	v_lshrrev_b32_e32 v0, 2, v0
	v_ashrrev_i32_e32 v99, 31, v98
	v_and_b32_e32 v0, 12, v0
	v_lshl_add_u64 v[134:135], v[98:99], 1, s[14:15]
	v_mad_u64_u32 v[98:99], s[0:1], v0, s26, v[206:207]
	v_add_u32_e32 v102, s26, v98
	v_ashrrev_i32_e32 v99, 31, v98
	v_ashrrev_i32_e32 v103, 31, v102
	v_lshlrev_b64 v[0:1], 1, v[98:99]
	v_lshlrev_b64 v[98:99], 1, v[102:103]
	v_add_u32_e32 v102, s26, v102
	v_pk_mul_f32 v[124:125], v[142:143], s[4:5] op_sel_hi:[1,0]
	v_add_u32_e32 v142, s26, v102
	v_pk_mul_f32 v[100:101], v[158:159], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[132:133], v[138:139], s[4:5] op_sel_hi:[1,0]
	v_lshl_add_u64 v[136:137], v[134:135], 0, v[0:1]
	v_lshl_add_u64 v[138:139], v[134:135], 0, v[98:99]
	v_ashrrev_i32_e32 v103, 31, v102
	v_ashrrev_i32_e32 v143, 31, v142
	global_store_short_d16_hi v[136:137], v100, off
	global_store_short_d16_hi v[138:139], v101, off
	v_lshlrev_b64 v[100:101], 1, v[102:103]
	v_lshlrev_b64 v[102:103], 1, v[142:143]
	v_pk_mul_f32 v[104:105], v[160:161], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[106:107], v[200:201], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[122:123], v[144:145], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[130:131], v[140:141], s[4:5] op_sel_hi:[1,0]
	v_lshl_add_u64 v[140:141], v[134:135], 0, v[100:101]
	v_lshl_add_u64 v[144:145], v[134:135], 0, v[102:103]
	s_mul_i32 s0, s26, 13
	v_pk_mul_f32 v[108:109], v[198:199], s[4:5] op_sel_hi:[1,0]
	global_store_short_d16_hi v[140:141], v104, off
	global_store_short_d16_hi v[144:145], v105, off
	global_store_short_d16_hi v[136:137], v108, off offset:32
	global_store_short_d16_hi v[138:139], v109, off offset:32
	global_store_short_d16_hi v[140:141], v106, off offset:32
	global_store_short_d16_hi v[144:145], v107, off offset:32
	v_add_u32_e32 v106, s0, v142
	v_ashrrev_i32_e32 v107, 31, v106
	v_lshlrev_b64 v[104:105], 1, v[106:107]
	v_add_u32_e32 v108, s26, v106
	v_pk_mul_f32 v[110:111], v[202:203], s[4:5] op_sel_hi:[1,0]
	v_lshl_add_u64 v[142:143], v[134:135], 0, v[104:105]
	v_ashrrev_i32_e32 v109, 31, v108
	global_store_short_d16_hi v[142:143], v110, off
	v_lshlrev_b64 v[106:107], 1, v[108:109]
	v_add_u32_e32 v110, s26, v108
	v_pk_mul_f32 v[126:127], v[150:151], s[4:5] op_sel_hi:[1,0]
	v_lshl_add_u64 v[146:147], v[134:135], 0, v[106:107]
	v_add_u32_e32 v150, s26, v110
	global_store_short_d16_hi v[146:147], v111, off
	v_ashrrev_i32_e32 v111, 31, v110
	v_ashrrev_i32_e32 v151, 31, v150
	v_lshlrev_b64 v[108:109], 1, v[110:111]
	v_lshlrev_b64 v[110:111], 1, v[150:151]
	v_pk_mul_f32 v[112:113], v[204:205], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[114:115], v[164:165], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[128:129], v[152:153], s[4:5] op_sel_hi:[1,0]
	v_lshl_add_u64 v[148:149], v[134:135], 0, v[108:109]
	v_lshl_add_u64 v[152:153], v[134:135], 0, v[110:111]
	v_pk_mul_f32 v[116:117], v[162:163], s[4:5] op_sel_hi:[1,0]
	global_store_short_d16_hi v[148:149], v112, off
	global_store_short_d16_hi v[152:153], v113, off
	global_store_short_d16_hi v[142:143], v116, off offset:32
	global_store_short_d16_hi v[146:147], v117, off offset:32
	global_store_short_d16_hi v[148:149], v114, off offset:32
	global_store_short_d16_hi v[152:153], v115, off offset:32
	v_add_u32_e32 v114, s0, v150
	v_ashrrev_i32_e32 v115, 31, v114
	v_lshlrev_b64 v[112:113], 1, v[114:115]
	v_add_u32_e32 v116, s26, v114
	v_pk_mul_f32 v[118:119], v[166:167], s[4:5] op_sel_hi:[1,0]
	v_lshl_add_u64 v[150:151], v[134:135], 0, v[112:113]
	v_ashrrev_i32_e32 v117, 31, v116
	global_store_short_d16_hi v[150:151], v118, off
	v_lshlrev_b64 v[114:115], 1, v[116:117]
	v_add_u32_e32 v118, s26, v116
	v_lshl_add_u64 v[154:155], v[134:135], 0, v[114:115]
	v_add_u32_e32 v158, s26, v118
	global_store_short_d16_hi v[154:155], v119, off
	v_ashrrev_i32_e32 v119, 31, v118
	v_ashrrev_i32_e32 v159, 31, v158
	v_lshlrev_b64 v[116:117], 1, v[118:119]
	v_lshlrev_b64 v[118:119], 1, v[158:159]
	v_lshl_add_u64 v[156:157], v[134:135], 0, v[116:117]
	v_lshl_add_u64 v[160:161], v[134:135], 0, v[118:119]
	global_store_short_d16_hi v[156:157], v120, off
	global_store_short_d16_hi v[160:161], v121, off
	global_store_short_d16_hi v[150:151], v124, off offset:32
	global_store_short_d16_hi v[154:155], v125, off offset:32
	global_store_short_d16_hi v[156:157], v122, off offset:32
	global_store_short_d16_hi v[160:161], v123, off offset:32
	v_add_u32_e32 v122, s0, v158
	v_ashrrev_i32_e32 v123, 31, v122
	v_add_u32_e32 v124, s26, v122
	v_lshlrev_b64 v[120:121], 1, v[122:123]
	v_ashrrev_i32_e32 v125, 31, v124
	v_lshl_add_u64 v[158:159], v[134:135], 0, v[120:121]
	v_lshlrev_b64 v[122:123], 1, v[124:125]
	global_store_short_d16_hi v[158:159], v126, off
	v_lshl_add_u64 v[162:163], v[134:135], 0, v[122:123]
	v_add_u32_e32 v126, s26, v124
	global_store_short_d16_hi v[162:163], v127, off
	v_ashrrev_i32_e32 v127, 31, v126
	v_lshlrev_b64 v[124:125], 1, v[126:127]
	v_add_u32_e32 v126, s26, v126
	v_ashrrev_i32_e32 v127, 31, v126
	v_lshlrev_b64 v[126:127], 1, v[126:127]
	v_lshl_add_u64 v[164:165], v[134:135], 0, v[124:125]
	v_lshl_add_u64 v[166:167], v[134:135], 0, v[126:127]
	s_mov_b64 s[0:1], 0x100
	global_store_short_d16_hi v[164:165], v128, off
	global_store_short_d16_hi v[166:167], v129, off
	global_store_short_d16_hi v[158:159], v132, off offset:32
	global_store_short_d16_hi v[162:163], v133, off offset:32
	global_store_short_d16_hi v[164:165], v130, off offset:32
	global_store_short_d16_hi v[166:167], v131, off offset:32
	v_pk_mul_f32 v[90:91], v[90:91], s[4:5] op_sel_hi:[1,0]
	v_lshl_add_u64 v[128:129], v[134:135], 0, s[0:1]
	v_pk_mul_f32 v[92:93], v[92:93], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[94:95], v[94:95], s[4:5] op_sel_hi:[1,0]
	global_store_short_d16_hi v[136:137], v90, off offset:256
	global_store_short_d16_hi v[138:139], v91, off offset:256
	global_store_short_d16_hi v[140:141], v92, off offset:256
	global_store_short_d16_hi v[144:145], v93, off offset:256
	v_lshl_add_u64 v[90:91], v[128:129], 0, v[0:1]
	global_store_short_d16_hi v[90:91], v94, off offset:32
	v_lshl_add_u64 v[90:91], v[128:129], 0, v[98:99]
	v_pk_mul_f32 v[96:97], v[96:97], s[4:5] op_sel_hi:[1,0]
	global_store_short_d16_hi v[90:91], v95, off offset:32
	v_lshl_add_u64 v[90:91], v[128:129], 0, v[100:101]
	v_pk_mul_f32 v[86:87], v[86:87], s[4:5] op_sel_hi:[1,0]
	global_store_short_d16_hi v[90:91], v96, off offset:32
	v_lshl_add_u64 v[90:91], v[128:129], 0, v[102:103]
	v_pk_mul_f32 v[88:89], v[88:89], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[82:83], v[82:83], s[4:5] op_sel_hi:[1,0]
	global_store_short_d16_hi v[90:91], v97, off offset:32
	global_store_short_d16_hi v[142:143], v86, off offset:256
	global_store_short_d16_hi v[146:147], v87, off offset:256
	global_store_short_d16_hi v[148:149], v88, off offset:256
	global_store_short_d16_hi v[152:153], v89, off offset:256
	v_lshl_add_u64 v[86:87], v[128:129], 0, v[104:105]
	global_store_short_d16_hi v[86:87], v82, off offset:32
	v_lshl_add_u64 v[86:87], v[128:129], 0, v[106:107]
	v_pk_mul_f32 v[84:85], v[84:85], s[4:5] op_sel_hi:[1,0]
	global_store_short_d16_hi v[86:87], v83, off offset:32
	v_lshl_add_u64 v[82:83], v[128:129], 0, v[108:109]
	v_pk_mul_f32 v[78:79], v[78:79], s[4:5] op_sel_hi:[1,0]
	global_store_short_d16_hi v[82:83], v84, off offset:32
	v_lshl_add_u64 v[82:83], v[128:129], 0, v[110:111]
	v_pk_mul_f32 v[80:81], v[80:81], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[74:75], v[74:75], s[4:5] op_sel_hi:[1,0]
	global_store_short_d16_hi v[82:83], v85, off offset:32
	global_store_short_d16_hi v[150:151], v78, off offset:256
	global_store_short_d16_hi v[154:155], v79, off offset:256
	global_store_short_d16_hi v[156:157], v80, off offset:256
	global_store_short_d16_hi v[160:161], v81, off offset:256
	v_lshl_add_u64 v[78:79], v[128:129], 0, v[112:113]
	global_store_short_d16_hi v[78:79], v74, off offset:32
	v_lshl_add_u64 v[78:79], v[128:129], 0, v[114:115]
	v_pk_mul_f32 v[76:77], v[76:77], s[4:5] op_sel_hi:[1,0]
	global_store_short_d16_hi v[78:79], v75, off offset:32
	v_lshl_add_u64 v[74:75], v[128:129], 0, v[116:117]
	v_pk_mul_f32 v[70:71], v[70:71], s[4:5] op_sel_hi:[1,0]
	global_store_short_d16_hi v[74:75], v76, off offset:32
	v_lshl_add_u64 v[74:75], v[128:129], 0, v[118:119]
	v_pk_mul_f32 v[72:73], v[72:73], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[58:59], s[4:5] op_sel_hi:[1,0]
	global_store_short_d16_hi v[74:75], v77, off offset:32
	global_store_short_d16_hi v[158:159], v70, off offset:256
	global_store_short_d16_hi v[162:163], v71, off offset:256
	global_store_short_d16_hi v[164:165], v72, off offset:256
	global_store_short_d16_hi v[166:167], v73, off offset:256
	v_lshl_add_u64 v[70:71], v[128:129], 0, v[120:121]
	global_store_short_d16_hi v[70:71], v58, off offset:32
	v_lshl_add_u64 v[70:71], v[128:129], 0, v[122:123]
	v_pk_mul_f32 v[60:61], v[60:61], s[4:5] op_sel_hi:[1,0]
	global_store_short_d16_hi v[70:71], v59, off offset:32
	v_lshl_add_u64 v[58:59], v[128:129], 0, v[124:125]
	global_store_short_d16_hi v[58:59], v60, off offset:32
	v_lshl_add_u64 v[58:59], v[128:129], 0, v[126:127]
	global_store_short_d16_hi v[58:59], v61, off offset:32
	v_pk_mul_f32 v[60:61], v[66:67], s[4:5] op_sel_hi:[1,0]
	v_or_b32_e32 v66, 2, v168
	v_mul_lo_u32 v66, v66, s26
	v_lshl_add_u32 v66, v66, 6, v169
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 1, s[14:15]
	v_pk_mul_f32 v[58:59], v[68:69], s[4:5] op_sel_hi:[1,0]
	v_lshl_add_u64 v[68:69], v[66:67], 0, v[0:1]
	v_lshl_add_u64 v[70:71], v[66:67], 0, v[98:99]
	v_pk_mul_f32 v[48:49], v[48:49], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], s[4:5] op_sel_hi:[1,0]
	global_store_short_d16_hi v[68:69], v60, off
	global_store_short_d16_hi v[70:71], v61, off
	v_lshl_add_u64 v[60:61], v[66:67], 0, v[100:101]
	v_lshl_add_u64 v[72:73], v[66:67], 0, v[102:103]
	v_pk_mul_f32 v[62:63], v[62:63], s[4:5] op_sel_hi:[1,0]
	global_store_short_d16_hi v[60:61], v58, off
	global_store_short_d16_hi v[72:73], v59, off
	global_store_short_d16_hi v[68:69], v46, off offset:32
	global_store_short_d16_hi v[70:71], v47, off offset:32
	global_store_short_d16_hi v[60:61], v48, off offset:32
	global_store_short_d16_hi v[72:73], v49, off offset:32
	v_lshl_add_u64 v[46:47], v[66:67], 0, v[104:105]
	v_lshl_add_u64 v[48:49], v[66:67], 0, v[106:107]
	v_pk_mul_f32 v[64:65], v[64:65], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[44:45], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], s[4:5] op_sel_hi:[1,0]
	global_store_short_d16_hi v[46:47], v62, off
	global_store_short_d16_hi v[48:49], v63, off
	v_lshl_add_u64 v[58:59], v[66:67], 0, v[108:109]
	v_lshl_add_u64 v[62:63], v[66:67], 0, v[110:111]
	v_pk_mul_f32 v[54:55], v[54:55], s[4:5] op_sel_hi:[1,0]
	global_store_short_d16_hi v[58:59], v64, off
	global_store_short_d16_hi v[62:63], v65, off
	global_store_short_d16_hi v[46:47], v42, off offset:32
	global_store_short_d16_hi v[48:49], v43, off offset:32
	global_store_short_d16_hi v[58:59], v44, off offset:32
	global_store_short_d16_hi v[62:63], v45, off offset:32
	v_lshl_add_u64 v[42:43], v[66:67], 0, v[112:113]
	v_lshl_add_u64 v[44:45], v[66:67], 0, v[114:115]
	v_pk_mul_f32 v[56:57], v[56:57], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[40:41], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], s[4:5] op_sel_hi:[1,0]
	global_store_short_d16_hi v[42:43], v54, off
	global_store_short_d16_hi v[44:45], v55, off
	v_lshl_add_u64 v[54:55], v[66:67], 0, v[116:117]
	v_lshl_add_u64 v[64:65], v[66:67], 0, v[118:119]
	v_pk_mul_f32 v[50:51], v[50:51], s[4:5] op_sel_hi:[1,0]
	global_store_short_d16_hi v[54:55], v56, off
	global_store_short_d16_hi v[64:65], v57, off
	global_store_short_d16_hi v[42:43], v38, off offset:32
	global_store_short_d16_hi v[44:45], v39, off offset:32
	global_store_short_d16_hi v[54:55], v40, off offset:32
	global_store_short_d16_hi v[64:65], v41, off offset:32
	v_lshl_add_u64 v[38:39], v[66:67], 0, v[120:121]
	v_lshl_add_u64 v[40:41], v[66:67], 0, v[122:123]
	v_pk_mul_f32 v[52:53], v[52:53], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], s[4:5] op_sel_hi:[1,0]
	global_store_short_d16_hi v[38:39], v50, off
	global_store_short_d16_hi v[40:41], v51, off
	v_lshl_add_u64 v[50:51], v[66:67], 0, v[124:125]
	v_lshl_add_u64 v[56:57], v[66:67], 0, v[126:127]
	v_pk_mul_f32 v[36:37], v[36:37], s[4:5] op_sel_hi:[1,0]
	global_store_short_d16_hi v[50:51], v52, off
	global_store_short_d16_hi v[56:57], v53, off
	global_store_short_d16_hi v[38:39], v34, off offset:32
	global_store_short_d16_hi v[40:41], v35, off offset:32
	global_store_short_d16_hi v[50:51], v36, off offset:32
	global_store_short_d16_hi v[56:57], v37, off offset:32
	v_lshl_add_u64 v[34:35], v[66:67], 0, s[0:1]
	v_pk_mul_f32 v[30:31], v[30:31], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], s[4:5] op_sel_hi:[1,0]
	v_lshl_add_u64 v[0:1], v[34:35], 0, v[0:1]
	v_pk_mul_f32 v[32:33], v[32:33], s[4:5] op_sel_hi:[1,0]
	global_store_short_d16_hi v[68:69], v30, off offset:256
	global_store_short_d16_hi v[70:71], v31, off offset:256
	global_store_short_d16_hi v[60:61], v32, off offset:256
	global_store_short_d16_hi v[72:73], v33, off offset:256
	global_store_short_d16_hi v[0:1], v26, off offset:32
	v_lshl_add_u64 v[0:1], v[34:35], 0, v[98:99]
	v_pk_mul_f32 v[28:29], v[28:29], s[4:5] op_sel_hi:[1,0]
	global_store_short_d16_hi v[0:1], v27, off offset:32
	v_lshl_add_u64 v[0:1], v[34:35], 0, v[100:101]
	global_store_short_d16_hi v[0:1], v28, off offset:32
	v_lshl_add_u64 v[0:1], v[34:35], 0, v[102:103]
	v_pk_mul_f32 v[24:25], v[24:25], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], s[4:5] op_sel_hi:[1,0]
	global_store_short_d16_hi v[0:1], v29, off offset:32
	global_store_short_d16_hi v[46:47], v22, off offset:256
	global_store_short_d16_hi v[48:49], v23, off offset:256
	global_store_short_d16_hi v[58:59], v24, off offset:256
	global_store_short_d16_hi v[62:63], v25, off offset:256
	v_lshl_add_u64 v[0:1], v[34:35], 0, v[104:105]
	global_store_short_d16_hi v[0:1], v18, off offset:32
	v_lshl_add_u64 v[0:1], v[34:35], 0, v[106:107]
	v_pk_mul_f32 v[20:21], v[20:21], s[4:5] op_sel_hi:[1,0]
	global_store_short_d16_hi v[0:1], v19, off offset:32
	v_lshl_add_u64 v[0:1], v[34:35], 0, v[108:109]
	global_store_short_d16_hi v[0:1], v20, off offset:32
	v_lshl_add_u64 v[0:1], v[34:35], 0, v[110:111]
	v_pk_mul_f32 v[16:17], v[16:17], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], s[4:5] op_sel_hi:[1,0]
	global_store_short_d16_hi v[0:1], v21, off offset:32
	global_store_short_d16_hi v[42:43], v14, off offset:256
	global_store_short_d16_hi v[44:45], v15, off offset:256
	global_store_short_d16_hi v[54:55], v16, off offset:256
	global_store_short_d16_hi v[64:65], v17, off offset:256
	v_lshl_add_u64 v[0:1], v[34:35], 0, v[112:113]
	global_store_short_d16_hi v[0:1], v10, off offset:32
	v_lshl_add_u64 v[0:1], v[34:35], 0, v[114:115]
	v_pk_mul_f32 v[12:13], v[12:13], s[4:5] op_sel_hi:[1,0]
	global_store_short_d16_hi v[0:1], v11, off offset:32
	v_lshl_add_u64 v[0:1], v[34:35], 0, v[116:117]
	global_store_short_d16_hi v[0:1], v12, off offset:32
	v_lshl_add_u64 v[0:1], v[34:35], 0, v[118:119]
	v_pk_mul_f32 v[8:9], v[8:9], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], s[4:5] op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], s[4:5] op_sel_hi:[1,0]
	global_store_short_d16_hi v[0:1], v13, off offset:32
	global_store_short_d16_hi v[38:39], v6, off offset:256
	global_store_short_d16_hi v[40:41], v7, off offset:256
	global_store_short_d16_hi v[50:51], v8, off offset:256
	global_store_short_d16_hi v[56:57], v9, off offset:256
	v_lshl_add_u64 v[0:1], v[34:35], 0, v[120:121]
	global_store_short_d16_hi v[0:1], v2, off offset:32
	v_lshl_add_u64 v[0:1], v[34:35], 0, v[122:123]
	v_pk_mul_f32 v[4:5], v[4:5], s[4:5] op_sel_hi:[1,0]
	global_store_short_d16_hi v[0:1], v3, off offset:32
	v_lshl_add_u64 v[0:1], v[34:35], 0, v[124:125]
	global_store_short_d16_hi v[0:1], v4, off offset:32
	v_lshl_add_u64 v[0:1], v[34:35], 0, v[126:127]
	global_store_short_d16_hi v[0:1], v5, off offset:32
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z29rcr_exact_8wave_scaled_kernelILb1ELi2EEv14layout_globals
		.amdhsa_group_segment_fixed_size 131072
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
	.section	.text._Z29rcr_exact_8wave_scaled_kernelILb1ELi2EEv14layout_globals,"axG",@progbits,_Z29rcr_exact_8wave_scaled_kernelILb1ELi2EEv14layout_globals,comdat
.Lfunc_end2:
