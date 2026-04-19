_Z22rcr_exact_8wave_kernel23rcr_exact_8wave_globals: ; @_Z22rcr_exact_8wave_kernel23rcr_exact_8wave_globals
; %bb.0:
	s_load_dwordx2 s[6:7], s[0:1], 0x0
	s_load_dwordx2 s[8:9], s[0:1], 0x10
	s_load_dwordx2 s[4:5], s[0:1], 0x20
	s_ashr_i32 s0, s2, 31
	s_lshr_b32 s0, s0, 28
	s_add_i32 s0, s2, s0
	s_ashr_i32 s12, s0, 4
	s_and_b32 s0, s0, -16
	s_sub_i32 s13, s2, s0
	s_lshl_b32 s15, s13, 20
	v_lshlrev_b32_e32 v1, 4, v0
	s_ashr_i32 s16, s15, 31
	s_waitcnt lgkmcnt(0)
	s_add_u32 s0, s8, s15
	v_and_b32_e32 v148, 0x1c00, v1
	v_lshlrev_b32_e32 v3, 9, v0
	s_addc_u32 s1, s9, s16
	v_or_b32_e32 v149, 0x10000, v148
	s_lshl_b32 s14, s12, 20
	v_xor_b32_e32 v2, v1, v0
	v_and_b32_e32 v3, 0x3f000, v3
	s_movk_i32 s17, 0x70
	s_mov_b32 s3, 0x110000
	s_mov_b32 s2, 0x80000
	v_readfirstlane_b32 s10, v149
	v_or_b32_e32 v1, 0x2000, v149
	s_ashr_i32 s18, s14, 31
	v_and_or_b32 v147, v2, s17, v3
	s_mov_b32 m0, s10
	v_readfirstlane_b32 s10, v1
	s_add_u32 s19, s6, s14
	s_mov_b64 s[26:27], s[2:3]
	v_or_b32_e32 v146, 0x40000, v147
	buffer_load_dwordx4 v147, s[0:3], 0 offen lds
	s_mov_b32 m0, s10
	s_addc_u32 s20, s7, s18
	s_mov_b64 s[24:25], s[0:1]
	v_readfirstlane_b32 s10, v148
	v_or_b32_e32 v152, 0x2000, v148
	buffer_load_dwordx4 v146, s[0:3], 0 offen lds
	s_mov_b32 s24, s19
	s_mov_b32 s25, s20
	s_mov_b32 m0, s10
	v_readfirstlane_b32 s10, v152
	buffer_load_dwordx4 v147, s[24:27], 0 offen lds
	s_mov_b32 m0, s10
	s_or_b32 s10, s15, 0x80000
	buffer_load_dwordx4 v146, s[24:27], 0 offen lds
	s_ashr_i32 s11, s10, 31
	s_mov_b64 s[26:27], s[2:3]
	s_add_u32 s10, s8, s10
	s_mov_b64 s[24:25], s[0:1]
	v_or_b32_e32 v1, 0x4000, v149
	s_addc_u32 s11, s9, s11
	s_mov_b32 s24, s10
	v_readfirstlane_b32 s10, v1
	v_or_b32_e32 v1, 0x6000, v149
	s_mov_b32 s25, s11
	s_mov_b32 m0, s10
	v_readfirstlane_b32 s10, v1
	buffer_load_dwordx4 v147, s[24:27], 0 offen lds
	s_mov_b32 m0, s10
	s_or_b32 s10, s14, 0x80000
	buffer_load_dwordx4 v146, s[24:27], 0 offen lds
	s_ashr_i32 s11, s10, 31
	s_mov_b64 s[26:27], s[2:3]
	s_add_u32 s10, s6, s10
	s_mov_b64 s[24:25], s[0:1]
	v_or_b32_e32 v153, 0x4000, v148
	s_addc_u32 s11, s7, s11
	s_mov_b32 s24, s10
	v_readfirstlane_b32 s10, v153
	v_or_b32_e32 v154, 0x6000, v148
	s_mov_b32 s25, s11
	s_mov_b32 m0, s10
	v_readfirstlane_b32 s10, v154
	buffer_load_dwordx4 v147, s[24:27], 0 offen lds
	s_mov_b32 m0, s10
	v_lshrrev_b32_e32 v1, 8, v0
	buffer_load_dwordx4 v146, s[24:27], 0 offen lds
	v_cmp_eq_u32_e32 vcc, 1, v1
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_2
; %bb.1:
	s_barrier
.LBB0_2:
	s_or_b64 exec, exec, s[10:11]
	v_or_b32_e32 v2, 0x8000, v149
	s_add_u32 s0, s0, 0x80
	v_readfirstlane_b32 s10, v2
	v_or_b32_e32 v2, 0xa000, v149
	s_addc_u32 s1, s1, 0
	s_mov_b32 m0, s10
	v_readfirstlane_b32 s10, v2
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	buffer_load_dwordx4 v147, s[0:3], 0 offen lds
	s_mov_b32 m0, s10
	v_or_b32_e32 v158, 0x8000, v148
	buffer_load_dwordx4 v146, s[0:3], 0 offen lds
	s_add_u32 s0, s19, 0x80
	v_readfirstlane_b32 s10, v158
	v_or_b32_e32 v159, 0xa000, v148
	s_addc_u32 s1, s20, 0
	s_mov_b32 m0, s10
	v_readfirstlane_b32 s10, v159
	buffer_load_dwordx4 v147, s[0:3], 0 offen lds
	s_mov_b32 m0, s10
	v_or_b32_e32 v2, 0xc000, v149
	buffer_load_dwordx4 v146, s[0:3], 0 offen lds
	s_or_b32 s0, s15, 0x80080
	s_ashr_i32 s1, s0, 31
	s_add_u32 s0, s8, s0
	v_readfirstlane_b32 s10, v2
	v_or_b32_e32 v2, 0xe000, v149
	s_addc_u32 s1, s9, s1
	s_mov_b32 m0, s10
	v_readfirstlane_b32 s10, v2
	buffer_load_dwordx4 v147, s[0:3], 0 offen lds
	s_mov_b32 m0, s10
	v_and_b32_e32 v2, 48, v0
	buffer_load_dwordx4 v146, s[0:3], 0 offen lds
	v_lshlrev_b32_e32 v3, 7, v0
	s_movk_i32 s0, 0x780
	v_bfe_u32 v178, v0, 6, 2
	v_and_or_b32 v2, v3, s0, v2
	v_lshl_or_b32 v3, v178, 12, v2
	v_or_b32_e32 v4, 0x10000, v3
	v_lshlrev_b32_e32 v5, 3, v0
	v_bitop3_b32 v161, v5, v4, s17 bitop3:0x6c
	v_or_b32_e32 v4, 0x10040, v3
	v_lshl_or_b32 v2, v1, 13, v2
	v_bitop3_b32 v160, v5, v4, s17 bitop3:0x6c
	v_or_b32_e32 v4, 64, v2
	v_bitop3_b32 v156, v5, v4, s17 bitop3:0x6c
	v_or_b32_e32 v4, 0x14000, v3
	v_bitop3_b32 v151, v5, v4, s17 bitop3:0x6c
	v_or_b32_e32 v4, 0x14040, v3
	v_bitop3_b32 v150, v5, v4, s17 bitop3:0x6c
	v_or_b32_e32 v4, 0x4000, v2
	v_bitop3_b32 v163, v5, v4, s17 bitop3:0x6c
	v_or_b32_e32 v4, 0x4040, v2
	v_bitop3_b32 v162, v5, v4, s17 bitop3:0x6c
	v_or_b32_e32 v4, 0x18000, v3
	v_bitop3_b32 v186, v5, v4, s17 bitop3:0x6c
	v_or_b32_e32 v4, 0x18040, v3
	v_bitop3_b32 v185, v5, v4, s17 bitop3:0x6c
	v_or_b32_e32 v4, 0x8000, v2
	v_bitop3_b32 v184, v5, v4, s17 bitop3:0x6c
	v_or_b32_e32 v4, 0x8040, v2
	s_add_u32 s10, s6, s14
	v_bitop3_b32 v183, v5, v4, s17 bitop3:0x6c
	v_or_b32_e32 v4, 0x1c000, v3
	v_or_b32_e32 v3, 0x1c040, v3
	s_addc_u32 s11, s7, s18
	v_bitop3_b32 v157, v5, v2, s17 bitop3:0x6c
	v_bitop3_b32 v181, v5, v3, s17 bitop3:0x6c
	v_or_b32_e32 v3, 0xc000, v2
	v_or_b32_e32 v2, 0xc040, v2
	s_add_u32 s15, s8, s15
	v_mov_b32_e32 v82, 0
	v_or_b32_e32 v155, 0xc000, v148
	v_or_b32_e32 v164, 0xe000, v148
	v_or_b32_e32 v165, 0x12000, v148
	v_or_b32_e32 v166, 0x14000, v148
	v_or_b32_e32 v167, 0x16000, v148
	v_bitop3_b32 v182, v5, v4, s17 bitop3:0x6c
	v_or_b32_e32 v168, 0x18000, v148
	v_or_b32_e32 v169, 0x1a000, v148
	v_bitop3_b32 v180, v5, v3, s17 bitop3:0x6c
	v_bitop3_b32 v179, v5, v2, s17 bitop3:0x6c
	v_or_b32_e32 v170, 0x1c000, v148
	v_or_b32_e32 v171, 0x1e000, v148
	s_addc_u32 s16, s9, s16
	s_mov_b64 s[8:9], 0
	v_mov_b32_e32 v83, v82
	v_mov_b32_e32 v84, v82
	v_mov_b32_e32 v85, v82
	v_mov_b32_e32 v110, v82
	v_mov_b32_e32 v111, v82
	v_mov_b32_e32 v112, v82
	v_mov_b32_e32 v113, v82
	v_mov_b32_e32 v118, v82
	v_mov_b32_e32 v119, v82
	v_mov_b32_e32 v120, v82
	v_mov_b32_e32 v121, v82
	v_mov_b32_e32 v126, v82
	v_mov_b32_e32 v127, v82
	v_mov_b32_e32 v128, v82
	v_mov_b32_e32 v129, v82
	v_mov_b32_e32 v130, v82
	v_mov_b32_e32 v131, v82
	v_mov_b32_e32 v132, v82
	v_mov_b32_e32 v133, v82
	v_mov_b32_e32 v134, v82
	v_mov_b32_e32 v135, v82
	v_mov_b32_e32 v136, v82
	v_mov_b32_e32 v137, v82
	v_mov_b32_e32 v138, v82
	v_mov_b32_e32 v139, v82
	v_mov_b32_e32 v140, v82
	v_mov_b32_e32 v141, v82
	v_mov_b32_e32 v142, v82
	v_mov_b32_e32 v143, v82
	v_mov_b32_e32 v144, v82
	v_mov_b32_e32 v145, v82
	v_mov_b32_e32 v78, v82
	v_mov_b32_e32 v79, v82
	v_mov_b32_e32 v80, v82
	v_mov_b32_e32 v81, v82
	v_mov_b32_e32 v90, v82
	v_mov_b32_e32 v91, v82
	v_mov_b32_e32 v92, v82
	v_mov_b32_e32 v93, v82
	v_mov_b32_e32 v94, v82
	v_mov_b32_e32 v95, v82
	v_mov_b32_e32 v96, v82
	v_mov_b32_e32 v97, v82
	v_mov_b32_e32 v98, v82
	v_mov_b32_e32 v99, v82
	v_mov_b32_e32 v100, v82
	v_mov_b32_e32 v101, v82
	v_mov_b32_e32 v102, v82
	v_mov_b32_e32 v103, v82
	v_mov_b32_e32 v104, v82
	v_mov_b32_e32 v105, v82
	v_mov_b32_e32 v106, v82
	v_mov_b32_e32 v107, v82
	v_mov_b32_e32 v108, v82
	v_mov_b32_e32 v109, v82
	v_mov_b32_e32 v114, v82
	v_mov_b32_e32 v115, v82
	v_mov_b32_e32 v116, v82
	v_mov_b32_e32 v117, v82
	v_mov_b32_e32 v122, v82
	v_mov_b32_e32 v123, v82
	v_mov_b32_e32 v124, v82
	v_mov_b32_e32 v125, v82
	v_mov_b32_e32 v22, v82
	v_mov_b32_e32 v23, v82
	v_mov_b32_e32 v24, v82
	v_mov_b32_e32 v25, v82
	v_mov_b32_e32 v30, v82
	v_mov_b32_e32 v31, v82
	v_mov_b32_e32 v32, v82
	v_mov_b32_e32 v33, v82
	v_mov_b32_e32 v46, v82
	v_mov_b32_e32 v47, v82
	v_mov_b32_e32 v48, v82
	v_mov_b32_e32 v49, v82
	v_mov_b32_e32 v54, v82
	v_mov_b32_e32 v55, v82
	v_mov_b32_e32 v56, v82
	v_mov_b32_e32 v57, v82
	v_mov_b32_e32 v66, v82
	v_mov_b32_e32 v67, v82
	v_mov_b32_e32 v68, v82
	v_mov_b32_e32 v69, v82
	v_mov_b32_e32 v70, v82
	v_mov_b32_e32 v71, v82
	v_mov_b32_e32 v72, v82
	v_mov_b32_e32 v73, v82
	v_mov_b32_e32 v74, v82
	v_mov_b32_e32 v75, v82
	v_mov_b32_e32 v76, v82
	v_mov_b32_e32 v77, v82
	v_mov_b32_e32 v86, v82
	v_mov_b32_e32 v87, v82
	v_mov_b32_e32 v88, v82
	v_mov_b32_e32 v89, v82
	v_mov_b32_e32 v18, v82
	v_mov_b32_e32 v19, v82
	v_mov_b32_e32 v20, v82
	v_mov_b32_e32 v21, v82
	v_mov_b32_e32 v26, v82
	v_mov_b32_e32 v27, v82
	v_mov_b32_e32 v28, v82
	v_mov_b32_e32 v29, v82
	v_mov_b32_e32 v34, v82
	v_mov_b32_e32 v35, v82
	v_mov_b32_e32 v36, v82
	v_mov_b32_e32 v37, v82
	v_mov_b32_e32 v38, v82
	v_mov_b32_e32 v39, v82
	v_mov_b32_e32 v40, v82
	v_mov_b32_e32 v41, v82
	v_mov_b32_e32 v42, v82
	v_mov_b32_e32 v43, v82
	v_mov_b32_e32 v44, v82
	v_mov_b32_e32 v45, v82
	v_mov_b32_e32 v50, v82
	v_mov_b32_e32 v51, v82
	v_mov_b32_e32 v52, v82
	v_mov_b32_e32 v53, v82
	v_mov_b32_e32 v58, v82
	v_mov_b32_e32 v59, v82
	v_mov_b32_e32 v60, v82
	v_mov_b32_e32 v61, v82
	v_mov_b32_e32 v62, v82
	v_mov_b32_e32 v63, v82
	v_mov_b32_e32 v64, v82
	v_mov_b32_e32 v65, v82
	;;#ASMSTART
	s_waitcnt vmcnt(6)
	;;#ASMEND
	s_barrier
.LBB0_3:                                ; =>This Inner Loop Header: Depth=1
	;;#ASMSTART
	ds_read_b128 v[2:5], v161 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[10:13], v161 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[6:9], v160 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[14:17], v160 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[188:191], v157 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[196:199], v157 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[204:207], v157 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[212:215], v157 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[192:195], v156 offset:0

	;;#ASMEND
	s_add_u32 s17, s10, s8
	;;#ASMSTART
	ds_read_b128 v[200:203], v156 offset:0x800

	;;#ASMEND
	s_addc_u32 s18, s11, s9
	;;#ASMSTART
	ds_read_b128 v[208:211], v156 offset:0x1000

	;;#ASMEND
	s_add_u32 s0, s17, 0x80080
	v_readfirstlane_b32 s19, v155
	;;#ASMSTART
	ds_read_b128 v[216:219], v156 offset:0x1800

	;;#ASMEND
	s_addc_u32 s1, s18, 0
	s_mov_b32 m0, s19
	v_readfirstlane_b32 s19, v164
	buffer_load_dwordx4 v147, s[0:3], 0 offen lds
	s_mov_b32 m0, s19
	s_nop 0
	buffer_load_dwordx4 v146, s[0:3], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[142:145], v[188:195], v[2:9], v[142:145]
	v_mfma_f32_16x16x128_f8f6f4 v[138:141], v[188:195], v[10:17], v[138:141]
	v_mfma_f32_16x16x128_f8f6f4 v[134:137], v[196:203], v[2:9], v[134:137]
	v_mfma_f32_16x16x128_f8f6f4 v[130:133], v[196:203], v[10:17], v[130:133]
	v_mfma_f32_16x16x128_f8f6f4 v[126:129], v[204:211], v[2:9], v[126:129]
	v_mfma_f32_16x16x128_f8f6f4 v[118:121], v[204:211], v[10:17], v[118:121]
	v_mfma_f32_16x16x128_f8f6f4 v[110:113], v[212:219], v[2:9], v[110:113]
	v_mfma_f32_16x16x128_f8f6f4 v[82:85], v[212:219], v[10:17], v[82:85]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b128 v[220:223], v151 offset:0

	;;#ASMEND
	s_add_u32 s19, s15, s8
	;;#ASMSTART
	ds_read_b128 v[228:231], v151 offset:0x800

	;;#ASMEND
	s_addc_u32 s20, s16, s9
	;;#ASMSTART
	ds_read_b128 v[224:227], v150 offset:0

	;;#ASMEND
	s_add_u32 s0, s19, 0x100
	v_readfirstlane_b32 s21, v149
	;;#ASMSTART
	ds_read_b128 v[232:235], v150 offset:0x800

	;;#ASMEND
	s_addc_u32 s1, s20, 0
	s_mov_b32 m0, s21
	v_readfirstlane_b32 s21, v165
	buffer_load_dwordx4 v147, s[0:3], 0 offen lds
	s_mov_b32 m0, s21
	s_nop 0
	buffer_load_dwordx4 v146, s[0:3], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[122:125], v[188:195], v[220:227], v[122:125]
	v_mfma_f32_16x16x128_f8f6f4 v[114:117], v[188:195], v[228:235], v[114:117]
	v_mfma_f32_16x16x128_f8f6f4 v[106:109], v[196:203], v[220:227], v[106:109]
	v_mfma_f32_16x16x128_f8f6f4 v[102:105], v[196:203], v[228:235], v[102:105]
	v_mfma_f32_16x16x128_f8f6f4 v[98:101], v[204:211], v[220:227], v[98:101]
	v_mfma_f32_16x16x128_f8f6f4 v[94:97], v[204:211], v[228:235], v[94:97]
	v_mfma_f32_16x16x128_f8f6f4 v[90:93], v[212:219], v[220:227], v[90:93]
	v_mfma_f32_16x16x128_f8f6f4 v[78:81], v[212:219], v[228:235], v[78:81]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[188:191], v163 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[196:199], v163 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[204:207], v163 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[212:215], v163 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[192:195], v162 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[200:203], v162 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[208:211], v162 offset:0x1000

	;;#ASMEND
	s_add_u32 s0, s17, 0x100
	v_readfirstlane_b32 s21, v148
	;;#ASMSTART
	ds_read_b128 v[216:219], v162 offset:0x1800

	;;#ASMEND
	s_addc_u32 s1, s18, 0
	s_mov_b32 m0, s21
	v_readfirstlane_b32 s21, v152
	buffer_load_dwordx4 v147, s[0:3], 0 offen lds
	s_mov_b32 m0, s21
	s_nop 0
	buffer_load_dwordx4 v146, s[0:3], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[86:89], v[188:195], v[2:9], v[86:89]
	v_mfma_f32_16x16x128_f8f6f4 v[74:77], v[188:195], v[10:17], v[74:77]
	v_mfma_f32_16x16x128_f8f6f4 v[70:73], v[196:203], v[2:9], v[70:73]
	v_mfma_f32_16x16x128_f8f6f4 v[66:69], v[196:203], v[10:17], v[66:69]
	v_mfma_f32_16x16x128_f8f6f4 v[54:57], v[204:211], v[2:9], v[54:57]
	v_mfma_f32_16x16x128_f8f6f4 v[46:49], v[204:211], v[10:17], v[46:49]
	v_mfma_f32_16x16x128_f8f6f4 v[30:33], v[212:219], v[2:9], v[30:33]
	v_mfma_f32_16x16x128_f8f6f4 v[22:25], v[212:219], v[10:17], v[22:25]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_u32 s0, s19, 0x80100
	v_readfirstlane_b32 s21, v166
	s_addc_u32 s1, s20, 0
	s_mov_b32 m0, s21
	v_readfirstlane_b32 s21, v167
	buffer_load_dwordx4 v147, s[0:3], 0 offen lds
	s_mov_b32 m0, s21
	s_nop 0
	buffer_load_dwordx4 v146, s[0:3], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(6)
	;;#ASMEND
	s_barrier
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[62:65], v[188:195], v[220:227], v[62:65]
	v_mfma_f32_16x16x128_f8f6f4 v[58:61], v[188:195], v[228:235], v[58:61]
	v_mfma_f32_16x16x128_f8f6f4 v[50:53], v[196:203], v[220:227], v[50:53]
	v_mfma_f32_16x16x128_f8f6f4 v[42:45], v[196:203], v[228:235], v[42:45]
	v_mfma_f32_16x16x128_f8f6f4 v[38:41], v[204:211], v[220:227], v[38:41]
	v_mfma_f32_16x16x128_f8f6f4 v[34:37], v[204:211], v[228:235], v[34:37]
	v_mfma_f32_16x16x128_f8f6f4 v[26:29], v[212:219], v[220:227], v[26:29]
	v_mfma_f32_16x16x128_f8f6f4 v[18:21], v[212:219], v[228:235], v[18:21]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[2:5], v186 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[10:13], v186 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[6:9], v185 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[14:17], v185 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[188:191], v184 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[196:199], v184 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[204:207], v184 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[212:215], v184 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[192:195], v183 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[200:203], v183 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[208:211], v183 offset:0x1000

	;;#ASMEND
	s_add_u32 s0, s17, 0x80100
	v_readfirstlane_b32 s21, v153
	;;#ASMSTART
	ds_read_b128 v[216:219], v183 offset:0x1800

	;;#ASMEND
	s_addc_u32 s1, s18, 0
	s_mov_b32 m0, s21
	v_readfirstlane_b32 s21, v154
	buffer_load_dwordx4 v147, s[0:3], 0 offen lds
	s_mov_b32 m0, s21
	s_nop 0
	buffer_load_dwordx4 v146, s[0:3], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[142:145], v[188:195], v[2:9], v[142:145]
	v_mfma_f32_16x16x128_f8f6f4 v[138:141], v[188:195], v[10:17], v[138:141]
	v_mfma_f32_16x16x128_f8f6f4 v[134:137], v[196:203], v[2:9], v[134:137]
	v_mfma_f32_16x16x128_f8f6f4 v[130:133], v[196:203], v[10:17], v[130:133]
	v_mfma_f32_16x16x128_f8f6f4 v[126:129], v[204:211], v[2:9], v[126:129]
	v_mfma_f32_16x16x128_f8f6f4 v[118:121], v[204:211], v[10:17], v[118:121]
	v_mfma_f32_16x16x128_f8f6f4 v[110:113], v[212:219], v[2:9], v[110:113]
	v_mfma_f32_16x16x128_f8f6f4 v[82:85], v[212:219], v[10:17], v[82:85]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b128 v[220:223], v182 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[228:231], v182 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[224:227], v181 offset:0

	;;#ASMEND
	s_add_u32 s0, s19, 0x180
	v_readfirstlane_b32 s21, v168
	;;#ASMSTART
	ds_read_b128 v[232:235], v181 offset:0x800

	;;#ASMEND
	s_addc_u32 s1, s20, 0
	s_mov_b32 m0, s21
	v_readfirstlane_b32 s21, v169
	buffer_load_dwordx4 v147, s[0:3], 0 offen lds
	s_mov_b32 m0, s21
	s_nop 0
	buffer_load_dwordx4 v146, s[0:3], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[122:125], v[188:195], v[220:227], v[122:125]
	v_mfma_f32_16x16x128_f8f6f4 v[114:117], v[188:195], v[228:235], v[114:117]
	v_mfma_f32_16x16x128_f8f6f4 v[106:109], v[196:203], v[220:227], v[106:109]
	v_mfma_f32_16x16x128_f8f6f4 v[102:105], v[196:203], v[228:235], v[102:105]
	v_mfma_f32_16x16x128_f8f6f4 v[98:101], v[204:211], v[220:227], v[98:101]
	v_mfma_f32_16x16x128_f8f6f4 v[94:97], v[204:211], v[228:235], v[94:97]
	v_mfma_f32_16x16x128_f8f6f4 v[90:93], v[212:219], v[220:227], v[90:93]
	v_mfma_f32_16x16x128_f8f6f4 v[78:81], v[212:219], v[228:235], v[78:81]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[188:191], v180 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[196:199], v180 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[204:207], v180 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[212:215], v180 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[192:195], v179 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[200:203], v179 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[208:211], v179 offset:0x1000

	;;#ASMEND
	s_add_u32 s0, s17, 0x180
	v_readfirstlane_b32 s17, v158
	;;#ASMSTART
	ds_read_b128 v[216:219], v179 offset:0x1800

	;;#ASMEND
	s_addc_u32 s1, s18, 0
	s_mov_b32 m0, s17
	v_readfirstlane_b32 s17, v159
	buffer_load_dwordx4 v147, s[0:3], 0 offen lds
	s_mov_b32 m0, s17
	s_nop 0
	buffer_load_dwordx4 v146, s[0:3], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[86:89], v[188:195], v[2:9], v[86:89]
	v_mfma_f32_16x16x128_f8f6f4 v[74:77], v[188:195], v[10:17], v[74:77]
	v_mfma_f32_16x16x128_f8f6f4 v[70:73], v[196:203], v[2:9], v[70:73]
	v_mfma_f32_16x16x128_f8f6f4 v[66:69], v[196:203], v[10:17], v[66:69]
	v_mfma_f32_16x16x128_f8f6f4 v[54:57], v[204:211], v[2:9], v[54:57]
	v_mfma_f32_16x16x128_f8f6f4 v[46:49], v[204:211], v[10:17], v[46:49]
	v_mfma_f32_16x16x128_f8f6f4 v[30:33], v[212:219], v[2:9], v[30:33]
	v_mfma_f32_16x16x128_f8f6f4 v[22:25], v[212:219], v[10:17], v[22:25]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_u32 s0, s19, 0x80180
	v_readfirstlane_b32 s17, v170
	s_addc_u32 s1, s20, 0
	s_mov_b32 m0, s17
	v_readfirstlane_b32 s17, v171
	buffer_load_dwordx4 v147, s[0:3], 0 offen lds
	s_mov_b32 m0, s17
	s_nop 0
	buffer_load_dwordx4 v146, s[0:3], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(6)
	;;#ASMEND
	s_barrier
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[62:65], v[188:195], v[220:227], v[62:65]
	v_mfma_f32_16x16x128_f8f6f4 v[58:61], v[188:195], v[228:235], v[58:61]
	v_mfma_f32_16x16x128_f8f6f4 v[50:53], v[196:203], v[220:227], v[50:53]
	v_mfma_f32_16x16x128_f8f6f4 v[42:45], v[196:203], v[228:235], v[42:45]
	v_mfma_f32_16x16x128_f8f6f4 v[38:41], v[204:211], v[220:227], v[38:41]
	v_mfma_f32_16x16x128_f8f6f4 v[34:37], v[204:211], v[228:235], v[34:37]
	v_mfma_f32_16x16x128_f8f6f4 v[26:29], v[212:219], v[220:227], v[26:29]
	v_mfma_f32_16x16x128_f8f6f4 v[18:21], v[212:219], v[228:235], v[18:21]
	s_setprio 0
	s_add_u32 s8, s8, 0x100
	s_addc_u32 s9, s9, 0
	s_cmpk_eq_i32 s8, 0xf00
	s_barrier
	s_cbranch_scc0 .LBB0_3
; %bb.4:
	;;#ASMSTART
	ds_read_b128 v[2:5], v161 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[10:13], v161 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[6:9], v160 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[14:17], v160 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[164:167], v157 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[188:191], v157 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[196:199], v157 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[204:207], v157 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[168:171], v156 offset:0

	;;#ASMEND
	s_or_b32 s0, s14, 0x80f80
	;;#ASMSTART
	ds_read_b128 v[192:195], v156 offset:0x800

	;;#ASMEND
	s_ashr_i32 s1, s0, 31
	;;#ASMSTART
	ds_read_b128 v[200:203], v156 offset:0x1000

	;;#ASMEND
	s_add_u32 s0, s6, s0
	v_readfirstlane_b32 s6, v155
	;;#ASMSTART
	ds_read_b128 v[208:211], v156 offset:0x1800

	;;#ASMEND
	s_addc_u32 s1, s7, s1
	s_mov_b32 s3, 0x110000
	s_mov_b32 s2, 0x80000
	s_mov_b32 m0, s6
	s_nop 0
	buffer_load_dwordx4 v147, s[0:3], 0 offen lds
	v_or_b32_e32 v147, 0xe000, v148
	s_nop 0
	v_readfirstlane_b32 s6, v147
	s_mov_b32 m0, s6
	s_nop 0
	buffer_load_dwordx4 v146, s[0:3], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[142:145], v[164:171], v[2:9], v[142:145]
	v_mfma_f32_16x16x128_f8f6f4 v[138:141], v[164:171], v[10:17], v[138:141]
	v_mfma_f32_16x16x128_f8f6f4 v[134:137], v[188:195], v[2:9], v[134:137]
	v_mfma_f32_16x16x128_f8f6f4 v[130:133], v[188:195], v[10:17], v[130:133]
	v_mfma_f32_16x16x128_f8f6f4 v[82:85], v[204:211], v[10:17], v[82:85]
	v_mfma_f32_16x16x128_f8f6f4 v[126:129], v[196:203], v[2:9], v[126:129]
	v_mfma_f32_16x16x128_f8f6f4 v[118:121], v[196:203], v[10:17], v[118:121]
	v_mfma_f32_16x16x128_f8f6f4 v[146:149], v[204:211], v[2:9], v[110:113]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b128 v[212:215], v151 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[220:223], v151 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[216:219], v150 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[224:227], v150 offset:0x800

	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[106:109], v[188:195], v[212:219], v[106:109]
	v_mfma_f32_16x16x128_f8f6f4 v[90:93], v[204:211], v[212:219], v[90:93]
	v_mfma_f32_16x16x128_f8f6f4 v[78:81], v[204:211], v[220:227], v[78:81]
	v_mfma_f32_16x16x128_f8f6f4 v[122:125], v[164:171], v[212:219], v[122:125]
	v_mfma_f32_16x16x128_f8f6f4 v[114:117], v[164:171], v[220:227], v[114:117]
	v_mfma_f32_16x16x128_f8f6f4 v[150:153], v[188:195], v[220:227], v[102:105]
	v_mfma_f32_16x16x128_f8f6f4 v[154:157], v[196:203], v[212:219], v[98:101]
	v_mfma_f32_16x16x128_f8f6f4 v[158:161], v[196:203], v[220:227], v[94:97]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[94:97], v163 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[188:191], v163 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[196:199], v163 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[204:207], v163 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[98:101], v162 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[192:195], v162 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[200:203], v162 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[208:211], v162 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[86:89], v[94:101], v[2:9], v[86:89]
	v_mfma_f32_16x16x128_f8f6f4 v[70:73], v[188:195], v[2:9], v[70:73]
	v_mfma_f32_16x16x128_f8f6f4 v[30:33], v[204:211], v[2:9], v[30:33]
	v_mfma_f32_16x16x128_f8f6f4 v[22:25], v[204:211], v[10:17], v[22:25]
	v_mfma_f32_16x16x128_f8f6f4 v[162:165], v[94:101], v[10:17], v[74:77]
	v_mfma_f32_16x16x128_f8f6f4 v[166:169], v[188:195], v[10:17], v[66:69]
	v_mfma_f32_16x16x128_f8f6f4 v[170:173], v[196:203], v[2:9], v[54:57]
	v_mfma_f32_16x16x128_f8f6f4 v[174:177], v[196:203], v[10:17], v[46:49]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[2:5], v186 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[10:13], v186 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[6:9], v185 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[14:17], v185 offset:0x800

	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[228:231], v[94:101], v[212:219], v[62:65]
	v_mfma_f32_16x16x128_f8f6f4 v[232:235], v[94:101], v[220:227], v[58:61]
	v_mfma_f32_16x16x128_f8f6f4 v[236:239], v[188:195], v[212:219], v[50:53]
	v_mfma_f32_16x16x128_f8f6f4 v[240:243], v[188:195], v[220:227], v[42:45]
	v_mfma_f32_16x16x128_f8f6f4 v[244:247], v[196:203], v[212:219], v[38:41]
	v_mfma_f32_16x16x128_f8f6f4 v[248:251], v[196:203], v[220:227], v[34:37]
	v_mfma_f32_16x16x128_f8f6f4 v[212:215], v[204:211], v[212:219], v[26:29]
	v_mfma_f32_16x16x128_f8f6f4 v[208:211], v[204:211], v[220:227], v[18:21]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b128 v[34:37], v184 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[46:49], v184 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[58:61], v184 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[184:187], v184 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[38:41], v183 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[50:53], v183 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[62:65], v183 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[188:191], v183 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[142:145], v[34:41], v[2:9], v[142:145]
	v_mfma_f32_16x16x128_f8f6f4 v[138:141], v[34:41], v[10:17], v[138:141]
	v_mfma_f32_16x16x128_f8f6f4 v[134:137], v[46:53], v[2:9], v[134:137]
	v_mfma_f32_16x16x128_f8f6f4 v[130:133], v[46:53], v[10:17], v[130:133]
	v_mfma_f32_16x16x128_f8f6f4 v[110:113], v[58:65], v[2:9], v[126:129]
	v_mfma_f32_16x16x128_f8f6f4 v[102:105], v[58:65], v[10:17], v[118:121]
	v_mfma_f32_16x16x128_f8f6f4 v[74:77], v[184:191], v[2:9], v[146:149]
	v_mfma_f32_16x16x128_f8f6f4 v[42:45], v[184:191], v[10:17], v[82:85]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[192:195], v182 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[200:203], v182 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[196:199], v181 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[204:207], v181 offset:0x800

	;;#ASMEND
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[94:97], v[34:41], v[192:199], v[122:125]
	v_mfma_f32_16x16x128_f8f6f4 v[98:101], v[34:41], v[200:207], v[114:117]
	v_mfma_f32_16x16x128_f8f6f4 v[82:85], v[46:53], v[192:199], v[106:109]
	v_mfma_f32_16x16x128_f8f6f4 v[66:69], v[46:53], v[200:207], v[150:153]
	v_mfma_f32_16x16x128_f8f6f4 v[54:57], v[58:65], v[192:199], v[154:157]
	v_mfma_f32_16x16x128_f8f6f4 v[46:49], v[58:65], v[200:207], v[158:161]
	v_mfma_f32_16x16x128_f8f6f4 v[34:37], v[184:191], v[192:199], v[90:93]
	v_mfma_f32_16x16x128_f8f6f4 v[26:29], v[184:191], v[200:207], v[78:81]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[114:117], v180 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[122:125], v180 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[146:149], v180 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[154:157], v180 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[118:121], v179 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[126:129], v179 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[150:153], v179 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[158:161], v179 offset:0x1800

	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[106:109], v[114:121], v[2:9], v[86:89]
	v_mfma_f32_16x16x128_f8f6f4 v[86:89], v[114:121], v[10:17], v[162:165]
	v_mfma_f32_16x16x128_f8f6f4 v[90:93], v[122:129], v[2:9], v[70:73]
	v_mfma_f32_16x16x128_f8f6f4 v[70:73], v[122:129], v[10:17], v[166:169]
	v_mfma_f32_16x16x128_f8f6f4 v[78:81], v[146:153], v[2:9], v[170:173]
	v_mfma_f32_16x16x128_f8f6f4 v[58:61], v[146:153], v[10:17], v[174:177]
	v_mfma_f32_16x16x128_f8f6f4 v[62:65], v[154:161], v[2:9], v[30:33]
	v_mfma_f32_16x16x128_f8f6f4 v[50:53], v[154:161], v[10:17], v[22:25]
	v_mfma_f32_16x16x128_f8f6f4 v[38:41], v[114:121], v[192:199], v[228:231]
	v_mfma_f32_16x16x128_f8f6f4 v[30:33], v[114:121], v[200:207], v[232:235]
	v_mfma_f32_16x16x128_f8f6f4 v[22:25], v[122:129], v[192:199], v[236:239]
	v_mfma_f32_16x16x128_f8f6f4 v[18:21], v[122:129], v[200:207], v[240:243]
	v_mfma_f32_16x16x128_f8f6f4 v[14:17], v[146:153], v[192:199], v[244:247]
	v_mfma_f32_16x16x128_f8f6f4 v[10:13], v[146:153], v[200:207], v[248:251]
	v_mfma_f32_16x16x128_f8f6f4 v[6:9], v[154:161], v[192:199], v[212:215]
	v_mfma_f32_16x16x128_f8f6f4 v[2:5], v[154:161], v[200:207], v[208:211]
	s_setprio 0
	s_movk_i32 s0, 0x100
	v_cmp_gt_u32_e32 vcc, s0, v0
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_6
; %bb.5:
	s_barrier
.LBB0_6:
	s_or_b64 exec, exec, s[0:1]
	v_lshlrev_b32_e32 v1, 18, v1
	v_lshlrev_b32_e32 v114, 5, v178
	v_lshl_or_b32 v1, s12, 20, v1
	v_lshl_or_b32 v114, s13, 8, v114
	v_add_u32_e32 v146, v1, v114
	v_lshlrev_b32_e32 v1, 10, v0
	s_mov_b32 s0, 0xc00f
	v_bitop3_b32 v0, v1, s0, v0 bitop3:0xc8
	v_ashrrev_i32_e32 v147, 31, v146
	v_lshlrev_b32_e32 v0, 1, v0
	v_mov_b32_e32 v1, 0
	v_lshl_add_u64 v[148:149], v[146:147], 1, s[4:5]
	v_or_b32_e32 v114, 0x2000, v0
	v_mov_b32_e32 v115, v1
	v_lshl_add_u64 v[116:117], v[148:149], 0, v[114:115]
	global_store_short_d16_hi v[116:117], v143, off
	v_or_b32_e32 v116, 0x4000, v0
	v_mov_b32_e32 v117, v1
	v_lshl_add_u64 v[118:119], v[148:149], 0, v[116:117]
	global_store_short_d16_hi v[118:119], v144, off
	v_or_b32_e32 v118, 0x6000, v0
	v_mov_b32_e32 v119, v1
	v_lshl_add_u64 v[150:151], v[148:149], 0, v[0:1]
	v_lshl_add_u64 v[120:121], v[148:149], 0, v[118:119]
	global_store_short_d16_hi v[150:151], v142, off
	global_store_short_d16_hi v[120:121], v145, off
	global_store_short_d16_hi v[150:151], v138, off offset:32
	v_or_b32_e32 v120, 0x2020, v0
	v_mov_b32_e32 v121, v1
	v_lshl_add_u64 v[122:123], v[148:149], 0, v[120:121]
	global_store_short_d16_hi v[122:123], v139, off
	v_or_b32_e32 v122, 0x4020, v0
	v_mov_b32_e32 v123, v1
	v_lshl_add_u64 v[124:125], v[148:149], 0, v[122:123]
	global_store_short_d16_hi v[124:125], v140, off
	v_or_b32_e32 v124, 0x6020, v0
	v_mov_b32_e32 v125, v1
	v_lshl_add_u64 v[126:127], v[148:149], 0, v[124:125]
	global_store_short_d16_hi v[126:127], v141, off
	v_or_b32_e32 v126, 0x20000, v0
	v_mov_b32_e32 v127, v1
	v_lshl_add_u64 v[128:129], v[148:149], 0, v[126:127]
	global_store_short_d16_hi v[128:129], v134, off
	v_or_b32_e32 v128, 0x22000, v0
	v_mov_b32_e32 v129, v1
	v_lshl_add_u64 v[138:139], v[148:149], 0, v[128:129]
	global_store_short_d16_hi v[138:139], v135, off
	v_or_b32_e32 v134, 0x24000, v0
	v_mov_b32_e32 v135, v1
	v_lshl_add_u64 v[138:139], v[148:149], 0, v[134:135]
	global_store_short_d16_hi v[138:139], v136, off
	v_or_b32_e32 v138, 0x26000, v0
	v_mov_b32_e32 v139, v1
	v_lshl_add_u64 v[140:141], v[148:149], 0, v[138:139]
	global_store_short_d16_hi v[140:141], v137, off
	v_or_b32_e32 v136, 0x20020, v0
	v_mov_b32_e32 v137, v1
	v_lshl_add_u64 v[140:141], v[148:149], 0, v[136:137]
	global_store_short_d16_hi v[140:141], v130, off
	v_or_b32_e32 v140, 0x22020, v0
	v_mov_b32_e32 v141, v1
	v_lshl_add_u64 v[142:143], v[148:149], 0, v[140:141]
	global_store_short_d16_hi v[142:143], v131, off
	v_or_b32_e32 v130, 0x24020, v0
	v_mov_b32_e32 v131, v1
	v_lshl_add_u64 v[142:143], v[148:149], 0, v[130:131]
	global_store_short_d16_hi v[142:143], v132, off
	v_or_b32_e32 v142, 0x26020, v0
	v_mov_b32_e32 v143, v1
	v_lshl_add_u64 v[144:145], v[148:149], 0, v[142:143]
	global_store_short_d16_hi v[144:145], v133, off
	v_or_b32_e32 v132, 0x40000, v0
	v_mov_b32_e32 v133, v1
	v_lshl_add_u64 v[144:145], v[148:149], 0, v[132:133]
	global_store_short_d16_hi v[144:145], v110, off
	v_or_b32_e32 v144, 0x42000, v0
	v_mov_b32_e32 v145, v1
	v_lshl_add_u64 v[152:153], v[148:149], 0, v[144:145]
	global_store_short_d16_hi v[152:153], v111, off
	v_or_b32_e32 v110, 0x44000, v0
	v_mov_b32_e32 v111, v1
	v_lshl_add_u64 v[152:153], v[148:149], 0, v[110:111]
	global_store_short_d16_hi v[152:153], v112, off
	v_or_b32_e32 v152, 0x46000, v0
	v_mov_b32_e32 v153, v1
	v_lshl_add_u64 v[154:155], v[148:149], 0, v[152:153]
	global_store_short_d16_hi v[154:155], v113, off
	v_or_b32_e32 v112, 0x40020, v0
	v_mov_b32_e32 v113, v1
	v_lshl_add_u64 v[154:155], v[148:149], 0, v[112:113]
	global_store_short_d16_hi v[154:155], v102, off
	v_or_b32_e32 v154, 0x42020, v0
	v_mov_b32_e32 v155, v1
	v_lshl_add_u64 v[156:157], v[148:149], 0, v[154:155]
	global_store_short_d16_hi v[156:157], v103, off
	v_or_b32_e32 v102, 0x44020, v0
	v_mov_b32_e32 v103, v1
	v_lshl_add_u64 v[156:157], v[148:149], 0, v[102:103]
	global_store_short_d16_hi v[156:157], v104, off
	v_or_b32_e32 v156, 0x46020, v0
	v_mov_b32_e32 v157, v1
	v_lshl_add_u64 v[158:159], v[148:149], 0, v[156:157]
	global_store_short_d16_hi v[158:159], v105, off
	v_or_b32_e32 v104, 0x60000, v0
	v_mov_b32_e32 v105, v1
	v_lshl_add_u64 v[158:159], v[148:149], 0, v[104:105]
	global_store_short_d16_hi v[158:159], v74, off
	v_or_b32_e32 v158, 0x62000, v0
	v_mov_b32_e32 v159, v1
	v_lshl_add_u64 v[160:161], v[148:149], 0, v[158:159]
	global_store_short_d16_hi v[160:161], v75, off
	v_or_b32_e32 v74, 0x64000, v0
	v_mov_b32_e32 v75, v1
	v_lshl_add_u64 v[160:161], v[148:149], 0, v[74:75]
	global_store_short_d16_hi v[160:161], v76, off
	v_or_b32_e32 v160, 0x66000, v0
	v_mov_b32_e32 v161, v1
	v_lshl_add_u64 v[162:163], v[148:149], 0, v[160:161]
	global_store_short_d16_hi v[162:163], v77, off
	v_or_b32_e32 v76, 0x60020, v0
	v_mov_b32_e32 v77, v1
	v_lshl_add_u64 v[162:163], v[148:149], 0, v[76:77]
	global_store_short_d16_hi v[162:163], v42, off
	v_or_b32_e32 v162, 0x62020, v0
	v_mov_b32_e32 v163, v1
	v_lshl_add_u64 v[164:165], v[148:149], 0, v[162:163]
	global_store_short_d16_hi v[164:165], v43, off
	v_or_b32_e32 v42, 0x64020, v0
	v_mov_b32_e32 v43, v1
	v_lshl_add_u64 v[164:165], v[148:149], 0, v[42:43]
	global_store_short_d16_hi v[164:165], v44, off
	v_or_b32_e32 v164, 0x66020, v0
	v_mov_b32_e32 v165, v1
	v_lshl_add_u64 v[166:167], v[148:149], 0, v[164:165]
	s_mov_b64 s[0:1], 0x100
	global_store_short_d16_hi v[166:167], v45, off
	v_lshl_add_u64 v[44:45], v[148:149], 0, s[0:1]
	v_lshl_add_u64 v[148:149], v[44:45], 0, v[114:115]
	global_store_short_d16_hi v[150:151], v94, off offset:256
	global_store_short_d16_hi v[148:149], v95, off
	v_lshl_add_u64 v[94:95], v[44:45], 0, v[116:117]
	global_store_short_d16_hi v[94:95], v96, off
	v_lshl_add_u64 v[94:95], v[44:45], 0, v[118:119]
	global_store_short_d16_hi v[94:95], v97, off
	v_lshl_add_u64 v[94:95], v[44:45], 0, v[0:1]
	global_store_short_d16_hi v[94:95], v98, off offset:32
	v_lshl_add_u64 v[94:95], v[44:45], 0, v[120:121]
	global_store_short_d16_hi v[94:95], v99, off
	v_lshl_add_u64 v[94:95], v[44:45], 0, v[122:123]
	global_store_short_d16_hi v[94:95], v100, off
	v_lshl_add_u64 v[94:95], v[44:45], 0, v[124:125]
	global_store_short_d16_hi v[94:95], v101, off
	v_lshl_add_u64 v[94:95], v[44:45], 0, v[126:127]
	global_store_short_d16_hi v[94:95], v82, off
	v_lshl_add_u64 v[94:95], v[44:45], 0, v[128:129]
	global_store_short_d16_hi v[94:95], v83, off
	v_lshl_add_u64 v[82:83], v[44:45], 0, v[134:135]
	global_store_short_d16_hi v[82:83], v84, off
	v_lshl_add_u64 v[82:83], v[44:45], 0, v[138:139]
	global_store_short_d16_hi v[82:83], v85, off
	v_lshl_add_u64 v[82:83], v[44:45], 0, v[136:137]
	global_store_short_d16_hi v[82:83], v66, off
	v_lshl_add_u64 v[82:83], v[44:45], 0, v[140:141]
	global_store_short_d16_hi v[82:83], v67, off
	v_lshl_add_u64 v[66:67], v[44:45], 0, v[130:131]
	global_store_short_d16_hi v[66:67], v68, off
	v_lshl_add_u64 v[66:67], v[44:45], 0, v[142:143]
	global_store_short_d16_hi v[66:67], v69, off
	v_lshl_add_u64 v[66:67], v[44:45], 0, v[132:133]
	global_store_short_d16_hi v[66:67], v54, off
	v_lshl_add_u64 v[66:67], v[44:45], 0, v[144:145]
	global_store_short_d16_hi v[66:67], v55, off
	v_lshl_add_u64 v[54:55], v[44:45], 0, v[110:111]
	global_store_short_d16_hi v[54:55], v56, off
	v_lshl_add_u64 v[54:55], v[44:45], 0, v[152:153]
	global_store_short_d16_hi v[54:55], v57, off
	v_lshl_add_u64 v[54:55], v[44:45], 0, v[112:113]
	global_store_short_d16_hi v[54:55], v46, off
	v_lshl_add_u64 v[54:55], v[44:45], 0, v[154:155]
	global_store_short_d16_hi v[54:55], v47, off
	v_lshl_add_u64 v[46:47], v[44:45], 0, v[102:103]
	global_store_short_d16_hi v[46:47], v48, off
	v_lshl_add_u64 v[46:47], v[44:45], 0, v[156:157]
	global_store_short_d16_hi v[46:47], v49, off
	v_lshl_add_u64 v[46:47], v[44:45], 0, v[104:105]
	global_store_short_d16_hi v[46:47], v34, off
	v_lshl_add_u64 v[46:47], v[44:45], 0, v[158:159]
	global_store_short_d16_hi v[46:47], v35, off
	v_lshl_add_u64 v[34:35], v[44:45], 0, v[74:75]
	global_store_short_d16_hi v[34:35], v36, off
	v_lshl_add_u64 v[34:35], v[44:45], 0, v[160:161]
	global_store_short_d16_hi v[34:35], v37, off
	v_lshl_add_u64 v[34:35], v[44:45], 0, v[76:77]
	global_store_short_d16_hi v[34:35], v26, off
	v_lshl_add_u64 v[34:35], v[44:45], 0, v[162:163]
	global_store_short_d16_hi v[34:35], v27, off
	v_lshl_add_u64 v[26:27], v[44:45], 0, v[42:43]
	global_store_short_d16_hi v[26:27], v28, off
	v_lshl_add_u64 v[26:27], v[44:45], 0, v[164:165]
	global_store_short_d16_hi v[26:27], v29, off
	v_add_u32_e32 v26, 0x80000, v146
	v_ashrrev_i32_e32 v27, 31, v26
	v_lshl_add_u64 v[26:27], v[26:27], 1, s[4:5]
	v_lshl_add_u64 v[34:35], v[26:27], 0, v[114:115]
	global_store_short_d16_hi v[34:35], v107, off
	v_lshl_add_u64 v[34:35], v[26:27], 0, v[116:117]
	v_lshl_add_u64 v[28:29], v[26:27], 0, v[0:1]
	global_store_short_d16_hi v[34:35], v108, off
	v_lshl_add_u64 v[34:35], v[26:27], 0, v[118:119]
	global_store_short_d16_hi v[28:29], v106, off
	global_store_short_d16_hi v[34:35], v109, off
	global_store_short_d16_hi v[28:29], v86, off offset:32
	v_lshl_add_u64 v[34:35], v[26:27], 0, v[120:121]
	global_store_short_d16_hi v[34:35], v87, off
	v_lshl_add_u64 v[34:35], v[26:27], 0, v[122:123]
	global_store_short_d16_hi v[34:35], v88, off
	v_lshl_add_u64 v[34:35], v[26:27], 0, v[124:125]
	global_store_short_d16_hi v[34:35], v89, off
	v_lshl_add_u64 v[34:35], v[26:27], 0, v[126:127]
	global_store_short_d16_hi v[34:35], v90, off
	v_lshl_add_u64 v[34:35], v[26:27], 0, v[128:129]
	global_store_short_d16_hi v[34:35], v91, off
	v_lshl_add_u64 v[34:35], v[26:27], 0, v[134:135]
	global_store_short_d16_hi v[34:35], v92, off
	v_lshl_add_u64 v[34:35], v[26:27], 0, v[138:139]
	global_store_short_d16_hi v[34:35], v93, off
	v_lshl_add_u64 v[34:35], v[26:27], 0, v[136:137]
	global_store_short_d16_hi v[34:35], v70, off
	v_lshl_add_u64 v[34:35], v[26:27], 0, v[140:141]
	global_store_short_d16_hi v[34:35], v71, off
	v_lshl_add_u64 v[34:35], v[26:27], 0, v[130:131]
	global_store_short_d16_hi v[34:35], v72, off
	v_lshl_add_u64 v[34:35], v[26:27], 0, v[142:143]
	global_store_short_d16_hi v[34:35], v73, off
	v_lshl_add_u64 v[34:35], v[26:27], 0, v[132:133]
	global_store_short_d16_hi v[34:35], v78, off
	v_lshl_add_u64 v[34:35], v[26:27], 0, v[144:145]
	global_store_short_d16_hi v[34:35], v79, off
	v_lshl_add_u64 v[34:35], v[26:27], 0, v[110:111]
	global_store_short_d16_hi v[34:35], v80, off
	v_lshl_add_u64 v[34:35], v[26:27], 0, v[152:153]
	global_store_short_d16_hi v[34:35], v81, off
	v_lshl_add_u64 v[34:35], v[26:27], 0, v[112:113]
	global_store_short_d16_hi v[34:35], v58, off
	v_lshl_add_u64 v[34:35], v[26:27], 0, v[154:155]
	global_store_short_d16_hi v[34:35], v59, off
	v_lshl_add_u64 v[34:35], v[26:27], 0, v[102:103]
	global_store_short_d16_hi v[34:35], v60, off
	v_lshl_add_u64 v[34:35], v[26:27], 0, v[156:157]
	global_store_short_d16_hi v[34:35], v61, off
	v_lshl_add_u64 v[34:35], v[26:27], 0, v[104:105]
	global_store_short_d16_hi v[34:35], v62, off
	v_lshl_add_u64 v[34:35], v[26:27], 0, v[158:159]
	global_store_short_d16_hi v[34:35], v63, off
	v_lshl_add_u64 v[34:35], v[26:27], 0, v[74:75]
	global_store_short_d16_hi v[34:35], v64, off
	v_lshl_add_u64 v[34:35], v[26:27], 0, v[160:161]
	global_store_short_d16_hi v[34:35], v65, off
	v_lshl_add_u64 v[34:35], v[26:27], 0, v[76:77]
	global_store_short_d16_hi v[34:35], v50, off
	v_lshl_add_u64 v[34:35], v[26:27], 0, v[162:163]
	global_store_short_d16_hi v[34:35], v51, off
	v_lshl_add_u64 v[34:35], v[26:27], 0, v[42:43]
	global_store_short_d16_hi v[34:35], v52, off
	v_lshl_add_u64 v[34:35], v[26:27], 0, v[164:165]
	v_lshl_add_u64 v[26:27], v[26:27], 0, s[0:1]
	v_lshl_add_u64 v[0:1], v[26:27], 0, v[0:1]
	global_store_short_d16_hi v[0:1], v30, off offset:32
	v_lshl_add_u64 v[0:1], v[26:27], 0, v[120:121]
	global_store_short_d16_hi v[0:1], v31, off
	v_lshl_add_u64 v[0:1], v[26:27], 0, v[122:123]
	global_store_short_d16_hi v[0:1], v32, off
	v_lshl_add_u64 v[0:1], v[26:27], 0, v[124:125]
	global_store_short_d16_hi v[0:1], v33, off
	v_lshl_add_u64 v[0:1], v[26:27], 0, v[126:127]
	global_store_short_d16_hi v[0:1], v22, off
	v_lshl_add_u64 v[0:1], v[26:27], 0, v[128:129]
	global_store_short_d16_hi v[0:1], v23, off
	v_lshl_add_u64 v[0:1], v[26:27], 0, v[134:135]
	global_store_short_d16_hi v[0:1], v24, off
	v_lshl_add_u64 v[0:1], v[26:27], 0, v[138:139]
	global_store_short_d16_hi v[0:1], v25, off
	v_lshl_add_u64 v[0:1], v[26:27], 0, v[136:137]
	global_store_short_d16_hi v[0:1], v18, off
	v_lshl_add_u64 v[0:1], v[26:27], 0, v[140:141]
	global_store_short_d16_hi v[0:1], v19, off
	v_lshl_add_u64 v[0:1], v[26:27], 0, v[130:131]
	global_store_short_d16_hi v[0:1], v20, off
	v_lshl_add_u64 v[0:1], v[26:27], 0, v[142:143]
	global_store_short_d16_hi v[0:1], v21, off
	v_lshl_add_u64 v[0:1], v[26:27], 0, v[132:133]
	global_store_short_d16_hi v[0:1], v14, off
	v_lshl_add_u64 v[0:1], v[26:27], 0, v[144:145]
	global_store_short_d16_hi v[0:1], v15, off
	v_lshl_add_u64 v[0:1], v[26:27], 0, v[110:111]
	global_store_short_d16_hi v[0:1], v16, off
	v_lshl_add_u64 v[0:1], v[26:27], 0, v[152:153]
	global_store_short_d16_hi v[0:1], v17, off
	v_lshl_add_u64 v[0:1], v[26:27], 0, v[112:113]
	global_store_short_d16_hi v[0:1], v10, off
	v_lshl_add_u64 v[0:1], v[26:27], 0, v[154:155]
	global_store_short_d16_hi v[0:1], v11, off
	v_lshl_add_u64 v[0:1], v[26:27], 0, v[102:103]
	global_store_short_d16_hi v[0:1], v12, off
	v_lshl_add_u64 v[0:1], v[26:27], 0, v[156:157]
	global_store_short_d16_hi v[0:1], v13, off
	v_lshl_add_u64 v[0:1], v[26:27], 0, v[104:105]
	global_store_short_d16_hi v[0:1], v6, off
	v_lshl_add_u64 v[0:1], v[26:27], 0, v[158:159]
	global_store_short_d16_hi v[0:1], v7, off
	v_lshl_add_u64 v[0:1], v[26:27], 0, v[74:75]
	global_store_short_d16_hi v[0:1], v8, off
	v_lshl_add_u64 v[0:1], v[26:27], 0, v[160:161]
	global_store_short_d16_hi v[0:1], v9, off
	v_lshl_add_u64 v[0:1], v[26:27], 0, v[76:77]
	global_store_short_d16_hi v[28:29], v38, off offset:256
	v_lshl_add_u64 v[28:29], v[26:27], 0, v[114:115]
	global_store_short_d16_hi v[0:1], v2, off
	v_lshl_add_u64 v[0:1], v[26:27], 0, v[162:163]
	global_store_short_d16_hi v[28:29], v39, off
	v_lshl_add_u64 v[28:29], v[26:27], 0, v[116:117]
	global_store_short_d16_hi v[0:1], v3, off
	v_lshl_add_u64 v[0:1], v[26:27], 0, v[42:43]
	global_store_short_d16_hi v[28:29], v40, off
	v_lshl_add_u64 v[28:29], v[26:27], 0, v[118:119]
	global_store_short_d16_hi v[0:1], v4, off
	v_lshl_add_u64 v[0:1], v[26:27], 0, v[164:165]
	global_store_short_d16_hi v[34:35], v53, off
	global_store_short_d16_hi v[28:29], v41, off
	global_store_short_d16_hi v[0:1], v5, off
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z22rcr_exact_8wave_kernel23rcr_exact_8wave_globals
		.amdhsa_group_segment_fixed_size 131072
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 56
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
		.amdhsa_next_free_vgpr 252
		.amdhsa_next_free_sgpr 96
		.amdhsa_accum_offset 252
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
	.size	_Z22rcr_exact_8wave_kernel23rcr_exact_8wave_globals, .Lfunc_end0-_Z22rcr_exact_8wave_kernel23rcr_exact_8wave_globals
                                        ; -- End function
	.set _Z22rcr_exact_8wave_kernel23rcr_exact_8wave_globals.num_vgpr, 252
	.set _Z22rcr_exact_8wave_kernel23rcr_exact_8wave_globals.num_agpr, 0
	.set _Z22rcr_exact_8wave_kernel23rcr_exact_8wave_globals.numbered_sgpr, 28
	.set _Z22rcr_exact_8wave_kernel23rcr_exact_8wave_globals.private_seg_size, 0
	.set _Z22rcr_exact_8wave_kernel23rcr_exact_8wave_globals.uses_vcc, 1
	.set _Z22rcr_exact_8wave_kernel23rcr_exact_8wave_globals.uses_flat_scratch, 0
	.set _Z22rcr_exact_8wave_kernel23rcr_exact_8wave_globals.has_dyn_sized_stack, 0
	.set _Z22rcr_exact_8wave_kernel23rcr_exact_8wave_globals.has_recursion, 0
	.set _Z22rcr_exact_8wave_kernel23rcr_exact_8wave_globals.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 7696
; TotalNumSgprs: 34
; NumVgprs: 252
; NumAgprs: 0
; TotalNumVgprs: 252
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 131072 bytes/workgroup (compile time only)
; SGPRBlocks: 12
; VGPRBlocks: 31
; NumSGPRsForWavesPerEU: 102
; NumVGPRsForWavesPerEU: 252
; AccumOffset: 252
; Occupancy: 2
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 62
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.protected	_Z22rrr_exact_8wave_kernel23rrr_exact_8wave_globals ; -- Begin function _Z22rrr_exact_8wave_kernel23rrr_exact_8wave_globals
	.globl	_Z22rrr_exact_8wave_kernel23rrr_exact_8wave_globals
	.p2align	8
	.type	_Z22rrr_exact_8wave_kernel23rrr_exact_8wave_globals,@function
