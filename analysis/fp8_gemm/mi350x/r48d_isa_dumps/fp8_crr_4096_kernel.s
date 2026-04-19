_Z22crr_exact_8wave_kernel23crr_exact_8wave_globals: ; @_Z22crr_exact_8wave_kernel23crr_exact_8wave_globals
; %bb.0:
	s_load_dwordx2 s[8:9], s[0:1], 0x0
	s_load_dwordx2 s[6:7], s[0:1], 0x10
	s_load_dwordx2 s[4:5], s[0:1], 0x20
	s_ashr_i32 s0, s2, 31
	s_lshr_b32 s0, s0, 28
	s_add_i32 s0, s2, s0
	s_ashr_i32 s10, s0, 4
	s_and_b32 s0, s0, 0xfffff0
	s_sub_i32 s0, s2, s0
	v_lshlrev_b32_e32 v1, 4, v0
	v_lshlrev_b32_e32 v2, 1, v0
	v_lshlrev_b32_e32 v3, 9, v0
	v_xor_b32_e32 v2, v2, v1
	v_and_b32_e32 v3, 0x3f000, v3
	s_movk_i32 s1, 0x70
	s_lshl_b32 s11, s0, 8
	v_and_b32_e32 v148, 0x1c00, v1
	v_and_or_b32 v147, v2, s1, v3
	s_ashr_i32 s17, s11, 31
	v_add_u32_e32 v2, 0x11000, v148
	v_and_b32_e32 v149, 0x180, v0
	s_waitcnt lgkmcnt(0)
	s_add_u32 s19, s6, s11
	v_or_b32_e32 v1, v2, v149
	v_or_b32_e32 v150, 0x2200, v149
	s_addc_u32 s20, s7, s17
	v_readfirstlane_b32 s12, v1
	v_add_u32_e32 v1, v2, v150
	s_lshl_b32 s16, s10, 8
	s_mov_b32 s3, 0x110000
	s_mov_b32 s2, 0x80000
	s_mov_b32 s0, s19
	s_mov_b32 s1, s20
	s_mov_b32 m0, s12
	v_readfirstlane_b32 s12, v1
	s_ashr_i32 s18, s16, 31
	buffer_load_dwordx4 v147, s[0:3], 0 offen lds
	s_mov_b32 m0, s12
	s_add_u32 s12, s8, s16
	v_or_b32_e32 v151, v148, v149
	v_or_b32_e32 v146, 0x40000, v147
	s_addc_u32 s13, s9, s18
	v_readfirstlane_b32 s14, v151
	v_or_b32_e32 v152, v148, v150
	buffer_load_dwordx4 v146, s[0:3], 0 offen lds
	s_mov_b32 s0, s12
	s_mov_b32 s1, s13
	s_mov_b32 m0, s14
	v_readfirstlane_b32 s14, v152
	v_add_u32_e32 v1, 0x4400, v2
	buffer_load_dwordx4 v147, s[0:3], 0 offen lds
	s_mov_b32 m0, s14
	v_add_u32_e32 v3, v1, v149
	buffer_load_dwordx4 v146, s[0:3], 0 offen lds
	s_add_u32 s0, s19, 0x80
	v_readfirstlane_b32 s14, v3
	v_add_u32_e32 v1, v1, v150
	s_addc_u32 s1, s20, 0
	s_mov_b32 m0, s14
	v_readfirstlane_b32 s14, v1
	v_add_u32_e32 v1, 0x4400, v148
	buffer_load_dwordx4 v147, s[0:3], 0 offen lds
	s_mov_b32 m0, s14
	v_or_b32_e32 v3, v1, v149
	buffer_load_dwordx4 v146, s[0:3], 0 offen lds
	s_add_u32 s0, s12, 0x80
	v_readfirstlane_b32 s14, v3
	v_add_u32_e32 v1, v1, v150
	s_addc_u32 s1, s13, 0
	s_mov_b32 m0, s14
	v_readfirstlane_b32 s14, v1
	buffer_load_dwordx4 v147, s[0:3], 0 offen lds
	s_mov_b32 m0, s14
	v_lshrrev_b32_e32 v1, 8, v0
	buffer_load_dwordx4 v146, s[0:3], 0 offen lds
	s_mov_b32 s14, 1
	s_mov_b32 s15, 0
	v_cmp_eq_u32_e32 vcc, 1, v1
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB2_2
; %bb.1:
	s_barrier
.LBB2_2:
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v3, 0x8800, v2
	v_add_u32_e32 v4, v3, v149
	s_add_u32 s0, s19, 0x80000
	v_readfirstlane_b32 s21, v4
	v_add_u32_e32 v3, v3, v150
	s_addc_u32 s1, s20, 0
	s_mov_b32 m0, s21
	v_readfirstlane_b32 s21, v3
	v_add_u32_e32 v3, 0x8800, v148
	;;#ASMSTART
	s_waitcnt vmcnt(2)
	;;#ASMEND
	s_barrier
	buffer_load_dwordx4 v147, s[0:3], 0 offen lds
	s_mov_b32 m0, s21
	v_add_u32_e32 v4, v3, v149
	buffer_load_dwordx4 v146, s[0:3], 0 offen lds
	s_add_u32 s0, s12, 0x80000
	v_readfirstlane_b32 s21, v4
	v_add_u32_e32 v3, v3, v150
	s_addc_u32 s1, s13, 0
	s_mov_b32 m0, s21
	v_readfirstlane_b32 s21, v3
	v_add_u32_e32 v2, 0xcc00, v2
	buffer_load_dwordx4 v147, s[0:3], 0 offen lds
	s_mov_b32 m0, s21
	v_add_u32_e32 v3, v2, v149
	buffer_load_dwordx4 v146, s[0:3], 0 offen lds
	s_add_u32 s0, s19, 0x80080
	v_readfirstlane_b32 s19, v3
	v_add_u32_e32 v2, v2, v150
	s_addc_u32 s1, s20, 0
	s_mov_b32 m0, s19
	v_readfirstlane_b32 s19, v2
	buffer_load_dwordx4 v147, s[0:3], 0 offen lds
	s_mov_b32 m0, s19
	v_lshrrev_b32_e32 v2, 1, v0
	buffer_load_dwordx4 v146, s[0:3], 0 offen lds
	v_and_b32_e32 v170, 0x60, v2
	v_bfe_u32 v2, v0, 1, 3
	v_bfe_u32 v4, v0, 4, 2
	v_lshlrev_b32_e32 v3, 3, v0
	v_lshlrev_b32_e32 v5, 11, v4
	v_add_u32_e32 v6, v4, v2
	v_or_b32_e32 v4, 4, v4
	v_and_b32_e32 v3, 8, v3
	v_lshlrev_b32_e32 v7, 4, v2
	v_lshl_or_b32 v176, v6, 7, v5
	v_lshlrev_b32_e32 v5, 11, v4
	v_add_u32_e32 v2, v4, v2
	s_add_u32 s8, s8, s16
	v_or_b32_e32 v8, v170, v3
	v_lshl_or_b32 v174, v2, 7, v5
	v_lshlrev_b32_e32 v2, 6, v1
	s_addc_u32 s9, s9, s18
	v_bitop3_b32 v178, v170, v7, v3 bitop3:0x36
	v_bitop3_b32 v177, v8, v7, 16 bitop3:0x36
	v_or_b32_e32 v4, v2, v3
	s_add_u32 s16, s6, s11
	v_mov_b32_e32 v98, 0
	v_or_b32_e32 v153, v176, v178
	v_or_b32_e32 v154, v176, v177
	v_or_b32_e32 v155, v174, v178
	v_or_b32_e32 v156, v174, v177
	v_bitop3_b32 v175, v2, v7, v3 bitop3:0x36
	v_bitop3_b32 v173, v4, v7, 16 bitop3:0x36
	v_bitop3_b32 v172, v4, v7, 32 bitop3:0x36
	v_bitop3_b32 v171, v4, v7, 48 bitop3:0x36
	s_addc_u32 s17, s7, s17
	s_mov_b64 s[6:7], 0
	v_mov_b32_e32 v99, v98
	v_mov_b32_e32 v100, v98
	v_mov_b32_e32 v101, v98
	v_mov_b32_e32 v110, v98
	v_mov_b32_e32 v111, v98
	v_mov_b32_e32 v112, v98
	v_mov_b32_e32 v113, v98
	v_mov_b32_e32 v118, v98
	v_mov_b32_e32 v119, v98
	v_mov_b32_e32 v120, v98
	v_mov_b32_e32 v121, v98
	v_mov_b32_e32 v126, v98
	v_mov_b32_e32 v127, v98
	v_mov_b32_e32 v128, v98
	v_mov_b32_e32 v129, v98
	v_mov_b32_e32 v130, v98
	v_mov_b32_e32 v131, v98
	v_mov_b32_e32 v132, v98
	v_mov_b32_e32 v133, v98
	v_mov_b32_e32 v134, v98
	v_mov_b32_e32 v135, v98
	v_mov_b32_e32 v136, v98
	v_mov_b32_e32 v137, v98
	v_mov_b32_e32 v138, v98
	v_mov_b32_e32 v139, v98
	v_mov_b32_e32 v140, v98
	v_mov_b32_e32 v141, v98
	v_mov_b32_e32 v142, v98
	v_mov_b32_e32 v143, v98
	v_mov_b32_e32 v144, v98
	v_mov_b32_e32 v145, v98
	v_mov_b32_e32 v82, v98
	v_mov_b32_e32 v83, v98
	v_mov_b32_e32 v84, v98
	v_mov_b32_e32 v85, v98
	v_mov_b32_e32 v86, v98
	v_mov_b32_e32 v87, v98
	v_mov_b32_e32 v88, v98
	v_mov_b32_e32 v89, v98
	v_mov_b32_e32 v90, v98
	v_mov_b32_e32 v91, v98
	v_mov_b32_e32 v92, v98
	v_mov_b32_e32 v93, v98
	v_mov_b32_e32 v94, v98
	v_mov_b32_e32 v95, v98
	v_mov_b32_e32 v96, v98
	v_mov_b32_e32 v97, v98
	v_mov_b32_e32 v102, v98
	v_mov_b32_e32 v103, v98
	v_mov_b32_e32 v104, v98
	v_mov_b32_e32 v105, v98
	v_mov_b32_e32 v106, v98
	v_mov_b32_e32 v107, v98
	v_mov_b32_e32 v108, v98
	v_mov_b32_e32 v109, v98
	v_mov_b32_e32 v114, v98
	v_mov_b32_e32 v115, v98
	v_mov_b32_e32 v116, v98
	v_mov_b32_e32 v117, v98
	v_mov_b32_e32 v122, v98
	v_mov_b32_e32 v123, v98
	v_mov_b32_e32 v124, v98
	v_mov_b32_e32 v125, v98
	v_mov_b32_e32 v42, v98
	v_mov_b32_e32 v43, v98
	v_mov_b32_e32 v44, v98
	v_mov_b32_e32 v45, v98
	v_mov_b32_e32 v50, v98
	v_mov_b32_e32 v51, v98
	v_mov_b32_e32 v52, v98
	v_mov_b32_e32 v53, v98
	v_mov_b32_e32 v58, v98
	v_mov_b32_e32 v59, v98
	v_mov_b32_e32 v60, v98
	v_mov_b32_e32 v61, v98
	v_mov_b32_e32 v62, v98
	v_mov_b32_e32 v63, v98
	v_mov_b32_e32 v64, v98
	v_mov_b32_e32 v65, v98
	v_mov_b32_e32 v66, v98
	v_mov_b32_e32 v67, v98
	v_mov_b32_e32 v68, v98
	v_mov_b32_e32 v69, v98
	v_mov_b32_e32 v70, v98
	v_mov_b32_e32 v71, v98
	v_mov_b32_e32 v72, v98
	v_mov_b32_e32 v73, v98
	v_mov_b32_e32 v74, v98
	v_mov_b32_e32 v75, v98
	v_mov_b32_e32 v76, v98
	v_mov_b32_e32 v77, v98
	v_mov_b32_e32 v78, v98
	v_mov_b32_e32 v79, v98
	v_mov_b32_e32 v80, v98
	v_mov_b32_e32 v81, v98
	v_mov_b32_e32 v18, v98
	v_mov_b32_e32 v19, v98
	v_mov_b32_e32 v20, v98
	v_mov_b32_e32 v21, v98
	v_mov_b32_e32 v22, v98
	v_mov_b32_e32 v23, v98
	v_mov_b32_e32 v24, v98
	v_mov_b32_e32 v25, v98
	v_mov_b32_e32 v26, v98
	v_mov_b32_e32 v27, v98
	v_mov_b32_e32 v28, v98
	v_mov_b32_e32 v29, v98
	v_mov_b32_e32 v30, v98
	v_mov_b32_e32 v31, v98
	v_mov_b32_e32 v32, v98
	v_mov_b32_e32 v33, v98
	v_mov_b32_e32 v34, v98
	v_mov_b32_e32 v35, v98
	v_mov_b32_e32 v36, v98
	v_mov_b32_e32 v37, v98
	v_mov_b32_e32 v38, v98
	v_mov_b32_e32 v39, v98
	v_mov_b32_e32 v40, v98
	v_mov_b32_e32 v41, v98
	v_mov_b32_e32 v46, v98
	v_mov_b32_e32 v47, v98
	v_mov_b32_e32 v48, v98
	v_mov_b32_e32 v49, v98
	v_mov_b32_e32 v54, v98
	v_mov_b32_e32 v55, v98
	v_mov_b32_e32 v56, v98
	v_mov_b32_e32 v57, v98
	;;#ASMSTART
	s_waitcnt vmcnt(6)
	;;#ASMEND
	s_barrier
.LBB2_3:                                ; =>This Inner Loop Header: Depth=1
	s_mul_i32 s19, s15, 0x8800
	s_add_i32 s18, s19, 0x11000
	v_add_u32_e32 v6, s18, v153
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v6 offset:0
ds_read_b64_tr_b8 v[4:5], v6 offset:1024

	;;#ASMEND
	v_add_u32_e32 v6, s18, v154
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v6 offset:0
ds_read_b64_tr_b8 v[12:13], v6 offset:1024

	;;#ASMEND
	v_add_u32_e32 v14, s18, v155
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v14 offset:0
ds_read_b64_tr_b8 v[8:9], v14 offset:1024

	;;#ASMEND
	v_add_u32_e32 v157, s18, v156
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v157 offset:0
ds_read_b64_tr_b8 v[16:17], v157 offset:1024

	;;#ASMEND
	v_add_u32_e32 v157, s19, v176
	v_add_u32_e32 v162, v157, v175
	;;#ASMSTART
	ds_read_b64_tr_b8 v[158:159], v162 offset:0
ds_read_b64_tr_b8 v[160:161], v162 offset:1024

	;;#ASMEND
	v_add_u32_e32 v162, v157, v173
	;;#ASMSTART
	ds_read_b64_tr_b8 v[180:181], v162 offset:0
ds_read_b64_tr_b8 v[182:183], v162 offset:1024

	;;#ASMEND
	v_add_u32_e32 v162, v157, v172
	;;#ASMSTART
	ds_read_b64_tr_b8 v[188:189], v162 offset:0
ds_read_b64_tr_b8 v[190:191], v162 offset:1024

	;;#ASMEND
	v_add_u32_e32 v157, v157, v171
	;;#ASMSTART
	ds_read_b64_tr_b8 v[196:197], v157 offset:0
ds_read_b64_tr_b8 v[198:199], v157 offset:1024

	;;#ASMEND
	v_add_u32_e32 v157, s19, v174
	v_add_u32_e32 v166, v157, v175
	;;#ASMSTART
	ds_read_b64_tr_b8 v[162:163], v166 offset:0
ds_read_b64_tr_b8 v[164:165], v166 offset:1024

	;;#ASMEND
	s_mul_i32 s0, s14, 0x8800
	v_add_u32_e32 v166, v157, v173
	;;#ASMSTART
	ds_read_b64_tr_b8 v[184:185], v166 offset:0
ds_read_b64_tr_b8 v[186:187], v166 offset:1024

	;;#ASMEND
	s_add_i32 s20, s0, 0x4400
	v_add_u32_e32 v166, v157, v172
	;;#ASMSTART
	ds_read_b64_tr_b8 v[192:193], v166 offset:0
ds_read_b64_tr_b8 v[194:195], v166 offset:1024

	;;#ASMEND
	v_add_u32_e32 v157, v157, v171
	s_add_u32 s21, s8, s6
	;;#ASMSTART
	ds_read_b64_tr_b8 v[200:201], v157 offset:0
ds_read_b64_tr_b8 v[202:203], v157 offset:1024

	;;#ASMEND
	s_addc_u32 s22, s9, s7
	v_add_u32_e32 v157, s20, v151
	s_add_u32 s0, s21, 0x80080
	v_readfirstlane_b32 s23, v157
	v_add_u32_e32 v157, s20, v152
	s_addc_u32 s1, s22, 0
	s_mov_b32 m0, s23
	v_readfirstlane_b32 s20, v157
	buffer_load_dwordx4 v147, s[0:3], 0 offen lds
	s_mov_b32 m0, s20
	s_add_i32 s20, s19, 0x15400
	buffer_load_dwordx4 v146, s[0:3], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(3)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[142:145], v[158:165], v[2:9], v[142:145]
	v_add_u32_e32 v157, s20, v153
	;;#ASMSTART
	ds_read_b64_tr_b8 v[204:205], v157 offset:0
ds_read_b64_tr_b8 v[206:207], v157 offset:1024

	;;#ASMEND
	v_add_u32_e32 v157, s20, v154
	;;#ASMSTART
	ds_read_b64_tr_b8 v[212:213], v157 offset:0
ds_read_b64_tr_b8 v[214:215], v157 offset:1024

	;;#ASMEND
	v_add_u32_e32 v157, s20, v155
	;;#ASMSTART
	ds_read_b64_tr_b8 v[208:209], v157 offset:0
ds_read_b64_tr_b8 v[210:211], v157 offset:1024

	;;#ASMEND
	v_add_u32_e32 v157, s20, v156
	v_mfma_f32_16x16x128_f8f6f4 v[138:141], v[158:165], v[10:17], v[138:141]
	;;#ASMSTART
	ds_read_b64_tr_b8 v[216:217], v157 offset:0
ds_read_b64_tr_b8 v[218:219], v157 offset:1024

	;;#ASMEND
	v_mfma_f32_16x16x128_f8f6f4 v[134:137], v[180:187], v[2:9], v[134:137]
	v_mfma_f32_16x16x128_f8f6f4 v[130:133], v[180:187], v[10:17], v[130:133]
	v_mfma_f32_16x16x128_f8f6f4 v[126:129], v[188:195], v[2:9], v[126:129]
	v_mfma_f32_16x16x128_f8f6f4 v[118:121], v[188:195], v[10:17], v[118:121]
	v_mfma_f32_16x16x128_f8f6f4 v[110:113], v[196:203], v[2:9], v[110:113]
	v_mfma_f32_16x16x128_f8f6f4 v[98:101], v[196:203], v[10:17], v[98:101]
	s_setprio 0
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[122:125], v[158:165], v[204:211], v[122:125]
	v_mfma_f32_16x16x128_f8f6f4 v[114:117], v[158:165], v[212:219], v[114:117]
	v_mfma_f32_16x16x128_f8f6f4 v[106:109], v[180:187], v[204:211], v[106:109]
	v_mfma_f32_16x16x128_f8f6f4 v[102:105], v[180:187], v[212:219], v[102:105]
	v_mfma_f32_16x16x128_f8f6f4 v[94:97], v[188:195], v[204:211], v[94:97]
	v_mfma_f32_16x16x128_f8f6f4 v[90:93], v[188:195], v[212:219], v[90:93]
	v_mfma_f32_16x16x128_f8f6f4 v[86:89], v[196:203], v[204:211], v[86:89]
	v_mfma_f32_16x16x128_f8f6f4 v[82:85], v[196:203], v[212:219], v[82:85]
	s_setprio 0
	s_add_i32 s0, s19, 0x4400
	v_add_u32_e32 v157, s0, v176
	s_barrier
	v_add_u32_e32 v162, v157, v175
	;;#ASMSTART
	ds_read_b64_tr_b8 v[158:159], v162 offset:0
ds_read_b64_tr_b8 v[160:161], v162 offset:1024

	;;#ASMEND
	v_add_u32_e32 v162, v157, v173
	;;#ASMSTART
	ds_read_b64_tr_b8 v[180:181], v162 offset:0
ds_read_b64_tr_b8 v[182:183], v162 offset:1024

	;;#ASMEND
	v_add_u32_e32 v162, v157, v172
	;;#ASMSTART
	ds_read_b64_tr_b8 v[188:189], v162 offset:0
ds_read_b64_tr_b8 v[190:191], v162 offset:1024

	;;#ASMEND
	v_add_u32_e32 v157, v157, v171
	;;#ASMSTART
	ds_read_b64_tr_b8 v[196:197], v157 offset:0
ds_read_b64_tr_b8 v[198:199], v157 offset:1024

	;;#ASMEND
	v_add_u32_e32 v157, s0, v174
	v_add_u32_e32 v166, v157, v175
	;;#ASMSTART
	ds_read_b64_tr_b8 v[162:163], v166 offset:0
ds_read_b64_tr_b8 v[164:165], v166 offset:1024

	;;#ASMEND
	v_add_u32_e32 v166, v157, v173
	;;#ASMSTART
	ds_read_b64_tr_b8 v[184:185], v166 offset:0
ds_read_b64_tr_b8 v[186:187], v166 offset:1024

	;;#ASMEND
	v_add_u32_e32 v166, v157, v172
	;;#ASMSTART
	ds_read_b64_tr_b8 v[192:193], v166 offset:0
ds_read_b64_tr_b8 v[194:195], v166 offset:1024

	;;#ASMEND
	v_add_u32_e32 v157, v157, v171
	;;#ASMSTART
	ds_read_b64_tr_b8 v[200:201], v157 offset:0
ds_read_b64_tr_b8 v[202:203], v157 offset:1024

	;;#ASMEND
	v_add_u32_e32 v157, s19, v151
	s_add_u32 s0, s21, 0x100000
	v_readfirstlane_b32 s21, v157
	v_add_u32_e32 v157, s19, v152
	s_addc_u32 s1, s22, 0
	s_mov_b32 m0, s21
	v_readfirstlane_b32 s19, v157
	buffer_load_dwordx4 v147, s[0:3], 0 offen lds
	s_mov_b32 m0, s19
	s_add_u32 s19, s16, s6
	s_addc_u32 s21, s17, s7
	v_add_u32_e32 v157, s20, v151
	buffer_load_dwordx4 v146, s[0:3], 0 offen lds
	s_add_u32 s0, s19, 0x100080
	v_readfirstlane_b32 s22, v157
	v_add_u32_e32 v157, s20, v152
	s_addc_u32 s1, s21, 0
	s_mov_b32 m0, s22
	v_readfirstlane_b32 s20, v157
	buffer_load_dwordx4 v147, s[0:3], 0 offen lds
	s_mov_b32 m0, s20
	s_nop 0
	buffer_load_dwordx4 v146, s[0:3], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[78:81], v[158:165], v[2:9], v[78:81]
	v_mfma_f32_16x16x128_f8f6f4 v[74:77], v[158:165], v[10:17], v[74:77]
	v_mfma_f32_16x16x128_f8f6f4 v[70:73], v[180:187], v[2:9], v[70:73]
	v_mfma_f32_16x16x128_f8f6f4 v[66:69], v[180:187], v[10:17], v[66:69]
	v_mfma_f32_16x16x128_f8f6f4 v[62:65], v[188:195], v[2:9], v[62:65]
	v_mfma_f32_16x16x128_f8f6f4 v[58:61], v[188:195], v[10:17], v[58:61]
	v_mfma_f32_16x16x128_f8f6f4 v[50:53], v[196:203], v[2:9], v[50:53]
	v_mfma_f32_16x16x128_f8f6f4 v[42:45], v[196:203], v[10:17], v[42:45]
	v_mfma_f32_16x16x128_f8f6f4 v[18:21], v[158:165], v[204:211], v[18:21]
	v_mfma_f32_16x16x128_f8f6f4 v[22:25], v[158:165], v[212:219], v[22:25]
	v_mfma_f32_16x16x128_f8f6f4 v[26:29], v[180:187], v[204:211], v[26:29]
	v_mfma_f32_16x16x128_f8f6f4 v[30:33], v[180:187], v[212:219], v[30:33]
	v_mfma_f32_16x16x128_f8f6f4 v[34:37], v[188:195], v[204:211], v[34:37]
	v_mfma_f32_16x16x128_f8f6f4 v[38:41], v[188:195], v[212:219], v[38:41]
	v_mfma_f32_16x16x128_f8f6f4 v[46:49], v[196:203], v[204:211], v[46:49]
	v_mfma_f32_16x16x128_f8f6f4 v[54:57], v[196:203], v[212:219], v[54:57]
	s_setprio 0
	v_add_u32_e32 v2, s18, v151
	s_add_u32 s0, s19, 0x100000
	v_readfirstlane_b32 s19, v2
	v_add_u32_e32 v2, s18, v152
	s_addc_u32 s1, s21, 0
	s_mov_b32 m0, s19
	v_readfirstlane_b32 s18, v2
	s_barrier
	buffer_load_dwordx4 v147, s[0:3], 0 offen lds
	s_mov_b32 m0, s18
	s_xor_b32 s15, s15, 1
	buffer_load_dwordx4 v146, s[0:3], 0 offen lds
	s_xor_b32 s14, s14, 1
	s_add_u32 s6, s6, 0x80000
	s_addc_u32 s7, s7, 0
	s_cmp_eq_u32 s6, 0xf00000
	s_cbranch_scc0 .LBB2_3
; %bb.4:
	v_add_u32_e32 v6, 0x11000, v176
	v_add_u32_e32 v7, v6, v178
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v7 offset:0
ds_read_b64_tr_b8 v[4:5], v7 offset:1024

	;;#ASMEND
	v_add_u32_e32 v6, v6, v177
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v6 offset:0
ds_read_b64_tr_b8 v[12:13], v6 offset:1024

	;;#ASMEND
	v_add_u32_e32 v14, 0x11000, v174
	v_add_u32_e32 v15, v14, v178
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v15 offset:0
ds_read_b64_tr_b8 v[8:9], v15 offset:1024

	;;#ASMEND
	v_add_u32_e32 v151, v14, v177
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v151 offset:0
ds_read_b64_tr_b8 v[16:17], v151 offset:1024

	;;#ASMEND
	v_add_u32_e32 v151, v176, v175
	;;#ASMSTART
	ds_read_b64_tr_b8 v[152:153], v151 offset:0
ds_read_b64_tr_b8 v[154:155], v151 offset:1024

	;;#ASMEND
	v_add_u32_e32 v151, v176, v173
	;;#ASMSTART
	ds_read_b64_tr_b8 v[160:161], v151 offset:0
ds_read_b64_tr_b8 v[162:163], v151 offset:1024

	;;#ASMEND
	v_add_u32_e32 v151, v176, v172
	;;#ASMSTART
	ds_read_b64_tr_b8 v[180:181], v151 offset:0
ds_read_b64_tr_b8 v[182:183], v151 offset:1024

	;;#ASMEND
	v_add_u32_e32 v151, v176, v171
	;;#ASMSTART
	ds_read_b64_tr_b8 v[188:189], v151 offset:0
ds_read_b64_tr_b8 v[190:191], v151 offset:1024

	;;#ASMEND
	v_add_u32_e32 v151, v174, v175
	;;#ASMSTART
	ds_read_b64_tr_b8 v[156:157], v151 offset:0
ds_read_b64_tr_b8 v[158:159], v151 offset:1024

	;;#ASMEND
	v_add_u32_e32 v148, 0xcc00, v148
	v_add_u32_e32 v151, v174, v173
	;;#ASMSTART
	ds_read_b64_tr_b8 v[164:165], v151 offset:0
ds_read_b64_tr_b8 v[166:167], v151 offset:1024

	;;#ASMEND
	v_add_u32_e32 v149, v148, v149
	v_add_u32_e32 v151, v174, v172
	;;#ASMSTART
	ds_read_b64_tr_b8 v[184:185], v151 offset:0
ds_read_b64_tr_b8 v[186:187], v151 offset:1024

	;;#ASMEND
	s_add_u32 s0, s12, 0xf80080
	v_readfirstlane_b32 s6, v149
	v_add_u32_e32 v151, v174, v171
	;;#ASMSTART
	ds_read_b64_tr_b8 v[192:193], v151 offset:0
ds_read_b64_tr_b8 v[194:195], v151 offset:1024

	;;#ASMEND
	s_addc_u32 s1, s13, 0
	s_mov_b32 s3, 0x110000
	s_mov_b32 s2, 0x80000
	s_mov_b32 m0, s6
	s_nop 0
	buffer_load_dwordx4 v147, s[0:3], 0 offen lds
	v_add_u32_e32 v147, v148, v150
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
	v_mfma_f32_16x16x128_f8f6f4 v[142:145], v[152:159], v[2:9], v[142:145]
	v_mfma_f32_16x16x128_f8f6f4 v[138:141], v[152:159], v[10:17], v[138:141]
	v_mfma_f32_16x16x128_f8f6f4 v[134:137], v[160:167], v[2:9], v[134:137]
	v_mfma_f32_16x16x128_f8f6f4 v[130:133], v[160:167], v[10:17], v[130:133]
	v_mfma_f32_16x16x128_f8f6f4 v[126:129], v[180:187], v[2:9], v[126:129]
	v_mfma_f32_16x16x128_f8f6f4 v[118:121], v[180:187], v[10:17], v[118:121]
	v_mfma_f32_16x16x128_f8f6f4 v[98:101], v[188:195], v[10:17], v[98:101]
	v_mfma_f32_16x16x128_f8f6f4 v[110:113], v[188:195], v[2:9], v[110:113]
	s_setprio 0
	v_add_u32_e32 v146, 0x15400, v176
	s_barrier
	v_add_u32_e32 v147, v146, v178
	;;#ASMSTART
	ds_read_b64_tr_b8 v[196:197], v147 offset:0
ds_read_b64_tr_b8 v[198:199], v147 offset:1024

	;;#ASMEND
	v_add_u32_e32 v146, v146, v177
	;;#ASMSTART
	ds_read_b64_tr_b8 v[204:205], v146 offset:0
ds_read_b64_tr_b8 v[206:207], v146 offset:1024

	;;#ASMEND
	v_add_u32_e32 v146, 0x15400, v174
	v_add_u32_e32 v147, v146, v178
	;;#ASMSTART
	ds_read_b64_tr_b8 v[200:201], v147 offset:0
ds_read_b64_tr_b8 v[202:203], v147 offset:1024

	;;#ASMEND
	v_add_u32_e32 v146, v146, v177
	;;#ASMSTART
	ds_read_b64_tr_b8 v[208:209], v146 offset:0
ds_read_b64_tr_b8 v[210:211], v146 offset:1024

	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[102:105], v[160:167], v[204:211], v[102:105]
	v_mfma_f32_16x16x128_f8f6f4 v[94:97], v[180:187], v[196:203], v[94:97]
	v_mfma_f32_16x16x128_f8f6f4 v[90:93], v[180:187], v[204:211], v[90:93]
	v_mfma_f32_16x16x128_f8f6f4 v[86:89], v[188:195], v[196:203], v[86:89]
	v_mfma_f32_16x16x128_f8f6f4 v[82:85], v[188:195], v[204:211], v[82:85]
	v_mfma_f32_16x16x128_f8f6f4 v[122:125], v[152:159], v[196:203], v[122:125]
	v_mfma_f32_16x16x128_f8f6f4 v[114:117], v[152:159], v[204:211], v[114:117]
	v_mfma_f32_16x16x128_f8f6f4 v[106:109], v[160:167], v[196:203], v[106:109]
	s_setprio 0
	v_add_u32_e32 v146, 0x4400, v176
	s_barrier
	v_add_u32_e32 v147, v146, v175
	;;#ASMSTART
	ds_read_b64_tr_b8 v[180:181], v147 offset:0
ds_read_b64_tr_b8 v[182:183], v147 offset:1024

	;;#ASMEND
	v_add_u32_e32 v147, v146, v173
	;;#ASMSTART
	ds_read_b64_tr_b8 v[188:189], v147 offset:0
ds_read_b64_tr_b8 v[190:191], v147 offset:1024

	;;#ASMEND
	v_add_u32_e32 v147, v146, v172
	;;#ASMSTART
	ds_read_b64_tr_b8 v[212:213], v147 offset:0
ds_read_b64_tr_b8 v[214:215], v147 offset:1024

	;;#ASMEND
	v_add_u32_e32 v146, v146, v171
	;;#ASMSTART
	ds_read_b64_tr_b8 v[220:221], v146 offset:0
ds_read_b64_tr_b8 v[222:223], v146 offset:1024

	;;#ASMEND
	v_add_u32_e32 v146, 0x4400, v174
	v_add_u32_e32 v147, v146, v175
	;;#ASMSTART
	ds_read_b64_tr_b8 v[184:185], v147 offset:0
ds_read_b64_tr_b8 v[186:187], v147 offset:1024

	;;#ASMEND
	v_add_u32_e32 v147, v146, v173
	;;#ASMSTART
	ds_read_b64_tr_b8 v[192:193], v147 offset:0
ds_read_b64_tr_b8 v[194:195], v147 offset:1024

	;;#ASMEND
	v_add_u32_e32 v147, v146, v172
	;;#ASMSTART
	ds_read_b64_tr_b8 v[216:217], v147 offset:0
ds_read_b64_tr_b8 v[218:219], v147 offset:1024

	;;#ASMEND
	v_add_u32_e32 v146, v146, v171
	;;#ASMSTART
	ds_read_b64_tr_b8 v[224:225], v146 offset:0
ds_read_b64_tr_b8 v[226:227], v146 offset:1024

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(2)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[74:77], v[180:187], v[10:17], v[74:77]
	v_mfma_f32_16x16x128_f8f6f4 v[66:69], v[188:195], v[10:17], v[66:69]
	v_mfma_f32_16x16x128_f8f6f4 v[146:149], v[180:187], v[2:9], v[78:81]
	v_mfma_f32_16x16x128_f8f6f4 v[150:153], v[188:195], v[2:9], v[70:73]
	v_mfma_f32_16x16x128_f8f6f4 v[154:157], v[212:219], v[2:9], v[62:65]
	v_mfma_f32_16x16x128_f8f6f4 v[158:161], v[212:219], v[10:17], v[58:61]
	v_mfma_f32_16x16x128_f8f6f4 v[162:165], v[220:227], v[2:9], v[50:53]
	v_mfma_f32_16x16x128_f8f6f4 v[166:169], v[220:227], v[10:17], v[42:45]
	s_setprio 0
	v_add_u32_e32 v6, 0x19800, v176
	s_barrier
	v_add_u32_e32 v7, v6, v178
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v7 offset:0
ds_read_b64_tr_b8 v[4:5], v7 offset:1024

	;;#ASMEND
	v_add_u32_e32 v6, v6, v177
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v6 offset:0
ds_read_b64_tr_b8 v[12:13], v6 offset:1024

	;;#ASMEND
	v_add_u32_e32 v14, 0x19800, v174
	v_add_u32_e32 v15, v14, v178
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v15 offset:0
ds_read_b64_tr_b8 v[8:9], v15 offset:1024

	;;#ASMEND
	v_add_u32_e32 v42, v14, v177
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v42 offset:0
ds_read_b64_tr_b8 v[16:17], v42 offset:1024

	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[228:231], v[180:187], v[196:203], v[18:21]
	v_mfma_f32_16x16x128_f8f6f4 v[232:235], v[180:187], v[204:211], v[22:25]
	v_mfma_f32_16x16x128_f8f6f4 v[236:239], v[188:195], v[196:203], v[26:29]
	v_mfma_f32_16x16x128_f8f6f4 v[240:243], v[188:195], v[204:211], v[30:33]
	v_mfma_f32_16x16x128_f8f6f4 v[244:247], v[212:219], v[196:203], v[34:37]
	v_mfma_f32_16x16x128_f8f6f4 v[248:251], v[212:219], v[204:211], v[38:41]
	v_mfma_f32_16x16x128_f8f6f4 v[252:255], v[220:227], v[196:203], v[46:49]
	v_mfma_f32_16x16x128_f8f6f4 v[220:223], v[220:227], v[204:211], v[54:57]
	s_setprio 0
	v_add_u32_e32 v22, 0x8800, v176
	s_barrier
	v_add_u32_e32 v23, v22, v175
	;;#ASMSTART
	ds_read_b64_tr_b8 v[18:19], v23 offset:0
ds_read_b64_tr_b8 v[20:21], v23 offset:1024

	;;#ASMEND
	v_add_u32_e32 v23, v22, v173
	;;#ASMSTART
	ds_read_b64_tr_b8 v[26:27], v23 offset:0
ds_read_b64_tr_b8 v[28:29], v23 offset:1024

	;;#ASMEND
	v_add_u32_e32 v23, v22, v172
	;;#ASMSTART
	ds_read_b64_tr_b8 v[38:39], v23 offset:0
ds_read_b64_tr_b8 v[40:41], v23 offset:1024

	;;#ASMEND
	v_add_u32_e32 v22, v22, v171
	;;#ASMSTART
	ds_read_b64_tr_b8 v[180:181], v22 offset:0
ds_read_b64_tr_b8 v[182:183], v22 offset:1024

	;;#ASMEND
	v_add_u32_e32 v34, 0x8800, v174
	v_add_u32_e32 v30, v34, v175
	;;#ASMSTART
	ds_read_b64_tr_b8 v[22:23], v30 offset:0
ds_read_b64_tr_b8 v[24:25], v30 offset:1024

	;;#ASMEND
	v_add_u32_e32 v35, v34, v173
	;;#ASMSTART
	ds_read_b64_tr_b8 v[30:31], v35 offset:0
ds_read_b64_tr_b8 v[32:33], v35 offset:1024

	;;#ASMEND
	v_add_u32_e32 v35, v34, v172
	;;#ASMSTART
	ds_read_b64_tr_b8 v[42:43], v35 offset:0
ds_read_b64_tr_b8 v[44:45], v35 offset:1024

	;;#ASMEND
	v_add_u32_e32 v34, v34, v171
	;;#ASMSTART
	ds_read_b64_tr_b8 v[184:185], v34 offset:0
ds_read_b64_tr_b8 v[186:187], v34 offset:1024

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[142:145], v[18:25], v[2:9], v[142:145]
	v_mfma_f32_16x16x128_f8f6f4 v[138:141], v[18:25], v[10:17], v[138:141]
	v_mfma_f32_16x16x128_f8f6f4 v[134:137], v[26:33], v[2:9], v[134:137]
	v_mfma_f32_16x16x128_f8f6f4 v[130:133], v[26:33], v[10:17], v[130:133]
	v_mfma_f32_16x16x128_f8f6f4 v[126:129], v[38:45], v[2:9], v[126:129]
	v_mfma_f32_16x16x128_f8f6f4 v[118:121], v[38:45], v[10:17], v[118:121]
	v_mfma_f32_16x16x128_f8f6f4 v[62:65], v[180:187], v[2:9], v[110:113]
	v_mfma_f32_16x16x128_f8f6f4 v[34:37], v[180:187], v[10:17], v[98:101]
	s_setprio 0
	v_add_u32_e32 v46, 0x1dc00, v176
	s_barrier
	v_add_u32_e32 v47, v46, v178
	;;#ASMSTART
	ds_read_b64_tr_b8 v[188:189], v47 offset:0
ds_read_b64_tr_b8 v[190:191], v47 offset:1024

	;;#ASMEND
	v_add_u32_e32 v46, v46, v177
	;;#ASMSTART
	ds_read_b64_tr_b8 v[196:197], v46 offset:0
ds_read_b64_tr_b8 v[198:199], v46 offset:1024

	;;#ASMEND
	v_add_u32_e32 v46, 0x1dc00, v174
	v_add_u32_e32 v47, v46, v178
	;;#ASMSTART
	ds_read_b64_tr_b8 v[192:193], v47 offset:0
ds_read_b64_tr_b8 v[194:195], v47 offset:1024

	;;#ASMEND
	v_add_u32_e32 v46, v46, v177
	;;#ASMSTART
	ds_read_b64_tr_b8 v[200:201], v46 offset:0
ds_read_b64_tr_b8 v[202:203], v46 offset:1024

	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[98:101], v[18:25], v[188:195], v[122:125]
	v_mfma_f32_16x16x128_f8f6f4 v[78:81], v[18:25], v[196:203], v[114:117]
	v_mfma_f32_16x16x128_f8f6f4 v[70:73], v[26:33], v[188:195], v[106:109]
	v_mfma_f32_16x16x128_f8f6f4 v[58:61], v[26:33], v[196:203], v[102:105]
	v_mfma_f32_16x16x128_f8f6f4 v[50:53], v[38:45], v[188:195], v[94:97]
	v_mfma_f32_16x16x128_f8f6f4 v[42:45], v[38:45], v[196:203], v[90:93]
	v_mfma_f32_16x16x128_f8f6f4 v[30:33], v[180:187], v[188:195], v[86:89]
	v_mfma_f32_16x16x128_f8f6f4 v[22:25], v[180:187], v[196:203], v[82:85]
	s_setprio 0
	v_add_u32_e32 v18, 0xcc00, v176
	s_barrier
	v_add_u32_e32 v19, v18, v175
	;;#ASMSTART
	ds_read_b64_tr_b8 v[106:107], v19 offset:0
ds_read_b64_tr_b8 v[108:109], v19 offset:1024

	;;#ASMEND
	v_add_u32_e32 v19, v18, v173
	;;#ASMSTART
	ds_read_b64_tr_b8 v[176:177], v19 offset:0
ds_read_b64_tr_b8 v[178:179], v19 offset:1024

	;;#ASMEND
	v_add_u32_e32 v19, v18, v172
	;;#ASMSTART
	ds_read_b64_tr_b8 v[204:205], v19 offset:0
ds_read_b64_tr_b8 v[206:207], v19 offset:1024

	;;#ASMEND
	v_add_u32_e32 v18, v18, v171
	;;#ASMSTART
	ds_read_b64_tr_b8 v[212:213], v18 offset:0
ds_read_b64_tr_b8 v[214:215], v18 offset:1024

	;;#ASMEND
	v_add_u32_e32 v18, 0xcc00, v174
	v_add_u32_e32 v19, v18, v175
	;;#ASMSTART
	ds_read_b64_tr_b8 v[110:111], v19 offset:0
ds_read_b64_tr_b8 v[112:113], v19 offset:1024

	;;#ASMEND
	v_add_u32_e32 v19, v18, v173
	;;#ASMSTART
	ds_read_b64_tr_b8 v[180:181], v19 offset:0
ds_read_b64_tr_b8 v[182:183], v19 offset:1024

	;;#ASMEND
	v_add_u32_e32 v19, v18, v172
	;;#ASMSTART
	ds_read_b64_tr_b8 v[208:209], v19 offset:0
ds_read_b64_tr_b8 v[210:211], v19 offset:1024

	;;#ASMEND
	v_add_u32_e32 v18, v18, v171
	;;#ASMSTART
	ds_read_b64_tr_b8 v[216:217], v18 offset:0
ds_read_b64_tr_b8 v[218:219], v18 offset:1024

	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[102:105], v[106:113], v[2:9], v[146:149]
	v_mfma_f32_16x16x128_f8f6f4 v[94:97], v[106:113], v[10:17], v[74:77]
	v_mfma_f32_16x16x128_f8f6f4 v[90:93], v[176:183], v[2:9], v[150:153]
	v_mfma_f32_16x16x128_f8f6f4 v[86:89], v[176:183], v[10:17], v[66:69]
	v_mfma_f32_16x16x128_f8f6f4 v[82:85], v[204:211], v[2:9], v[154:157]
	v_mfma_f32_16x16x128_f8f6f4 v[74:77], v[204:211], v[10:17], v[158:161]
	v_mfma_f32_16x16x128_f8f6f4 v[66:69], v[212:219], v[2:9], v[162:165]
	v_mfma_f32_16x16x128_f8f6f4 v[54:57], v[212:219], v[10:17], v[166:169]
	v_mfma_f32_16x16x128_f8f6f4 v[46:49], v[106:113], v[188:195], v[228:231]
	v_mfma_f32_16x16x128_f8f6f4 v[38:41], v[106:113], v[196:203], v[232:235]
	v_mfma_f32_16x16x128_f8f6f4 v[26:29], v[176:183], v[188:195], v[236:239]
	v_mfma_f32_16x16x128_f8f6f4 v[18:21], v[176:183], v[196:203], v[240:243]
	v_mfma_f32_16x16x128_f8f6f4 v[14:17], v[204:211], v[188:195], v[244:247]
	v_mfma_f32_16x16x128_f8f6f4 v[10:13], v[204:211], v[196:203], v[248:251]
	v_mfma_f32_16x16x128_f8f6f4 v[6:9], v[212:219], v[188:195], v[252:255]
	v_mfma_f32_16x16x128_f8f6f4 v[2:5], v[212:219], v[196:203], v[220:223]
	s_setprio 0
	s_movk_i32 s0, 0x100
	v_cmp_gt_u32_e32 vcc, s0, v0
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB2_6
; %bb.5:
	s_barrier
.LBB2_6:
	s_or_b64 exec, exec, s[0:1]
	v_lshlrev_b32_e32 v1, 18, v1
	v_lshl_or_b32 v1, s10, 20, v1
	v_or_b32_e32 v106, s11, v170
	v_add_u32_e32 v146, v1, v106
	v_lshlrev_b32_e32 v1, 10, v0
	s_mov_b32 s0, 0xc00f
	v_bitop3_b32 v0, v1, s0, v0 bitop3:0xc8
	v_ashrrev_i32_e32 v147, 31, v146
	v_lshlrev_b32_e32 v0, 1, v0
	v_mov_b32_e32 v1, 0
	v_lshl_add_u64 v[148:149], v[146:147], 1, s[4:5]
	v_or_b32_e32 v106, 0x2000, v0
	v_mov_b32_e32 v107, v1
	v_lshl_add_u64 v[108:109], v[148:149], 0, v[106:107]
	global_store_short_d16_hi v[108:109], v143, off
	v_or_b32_e32 v108, 0x4000, v0
	v_mov_b32_e32 v109, v1
	v_lshl_add_u64 v[110:111], v[148:149], 0, v[108:109]
	global_store_short_d16_hi v[110:111], v144, off
	v_or_b32_e32 v110, 0x6000, v0
	v_mov_b32_e32 v111, v1
	v_lshl_add_u64 v[150:151], v[148:149], 0, v[0:1]
	v_lshl_add_u64 v[112:113], v[148:149], 0, v[110:111]
	global_store_short_d16_hi v[150:151], v142, off
	global_store_short_d16_hi v[112:113], v145, off
	global_store_short_d16_hi v[150:151], v138, off offset:32
	v_or_b32_e32 v112, 0x2020, v0
	v_mov_b32_e32 v113, v1
	v_lshl_add_u64 v[114:115], v[148:149], 0, v[112:113]
	global_store_short_d16_hi v[114:115], v139, off
	v_or_b32_e32 v114, 0x4020, v0
	v_mov_b32_e32 v115, v1
	v_lshl_add_u64 v[116:117], v[148:149], 0, v[114:115]
	global_store_short_d16_hi v[116:117], v140, off
	v_or_b32_e32 v116, 0x6020, v0
	v_mov_b32_e32 v117, v1
	v_lshl_add_u64 v[122:123], v[148:149], 0, v[116:117]
	global_store_short_d16_hi v[122:123], v141, off
	v_or_b32_e32 v122, 0x20000, v0
	v_mov_b32_e32 v123, v1
	v_lshl_add_u64 v[124:125], v[148:149], 0, v[122:123]
	global_store_short_d16_hi v[124:125], v134, off
	v_or_b32_e32 v124, 0x22000, v0
	v_mov_b32_e32 v125, v1
	v_lshl_add_u64 v[138:139], v[148:149], 0, v[124:125]
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
	global_store_short_d16_hi v[144:145], v126, off
	v_or_b32_e32 v144, 0x42000, v0
	v_mov_b32_e32 v145, v1
	v_lshl_add_u64 v[152:153], v[148:149], 0, v[144:145]
	global_store_short_d16_hi v[152:153], v127, off
	v_or_b32_e32 v126, 0x44000, v0
	v_mov_b32_e32 v127, v1
	v_lshl_add_u64 v[152:153], v[148:149], 0, v[126:127]
	global_store_short_d16_hi v[152:153], v128, off
	v_or_b32_e32 v152, 0x46000, v0
	v_mov_b32_e32 v153, v1
	v_lshl_add_u64 v[154:155], v[148:149], 0, v[152:153]
	global_store_short_d16_hi v[154:155], v129, off
	v_or_b32_e32 v128, 0x40020, v0
	v_mov_b32_e32 v129, v1
	v_lshl_add_u64 v[154:155], v[148:149], 0, v[128:129]
	global_store_short_d16_hi v[154:155], v118, off
	v_or_b32_e32 v154, 0x42020, v0
	v_mov_b32_e32 v155, v1
	v_lshl_add_u64 v[156:157], v[148:149], 0, v[154:155]
	global_store_short_d16_hi v[156:157], v119, off
	v_or_b32_e32 v118, 0x44020, v0
	v_mov_b32_e32 v119, v1
	v_lshl_add_u64 v[156:157], v[148:149], 0, v[118:119]
	global_store_short_d16_hi v[156:157], v120, off
	v_or_b32_e32 v156, 0x46020, v0
	v_mov_b32_e32 v157, v1
	v_lshl_add_u64 v[158:159], v[148:149], 0, v[156:157]
	global_store_short_d16_hi v[158:159], v121, off
	v_or_b32_e32 v120, 0x60000, v0
	v_mov_b32_e32 v121, v1
	v_lshl_add_u64 v[158:159], v[148:149], 0, v[120:121]
	global_store_short_d16_hi v[158:159], v62, off
	v_or_b32_e32 v158, 0x62000, v0
	v_mov_b32_e32 v159, v1
	v_lshl_add_u64 v[160:161], v[148:149], 0, v[158:159]
	global_store_short_d16_hi v[160:161], v63, off
	v_or_b32_e32 v62, 0x64000, v0
	v_mov_b32_e32 v63, v1
	v_lshl_add_u64 v[160:161], v[148:149], 0, v[62:63]
	global_store_short_d16_hi v[160:161], v64, off
	v_or_b32_e32 v160, 0x66000, v0
	v_mov_b32_e32 v161, v1
	v_lshl_add_u64 v[162:163], v[148:149], 0, v[160:161]
	global_store_short_d16_hi v[162:163], v65, off
	v_or_b32_e32 v64, 0x60020, v0
	v_mov_b32_e32 v65, v1
	v_lshl_add_u64 v[162:163], v[148:149], 0, v[64:65]
	global_store_short_d16_hi v[162:163], v34, off
	v_or_b32_e32 v162, 0x62020, v0
	v_mov_b32_e32 v163, v1
	v_lshl_add_u64 v[164:165], v[148:149], 0, v[162:163]
	global_store_short_d16_hi v[164:165], v35, off
	v_or_b32_e32 v34, 0x64020, v0
	v_mov_b32_e32 v35, v1
	v_lshl_add_u64 v[164:165], v[148:149], 0, v[34:35]
	global_store_short_d16_hi v[164:165], v36, off
	v_or_b32_e32 v164, 0x66020, v0
	v_mov_b32_e32 v165, v1
	v_lshl_add_u64 v[166:167], v[148:149], 0, v[164:165]
	s_mov_b64 s[0:1], 0x100
	global_store_short_d16_hi v[166:167], v37, off
	v_lshl_add_u64 v[36:37], v[148:149], 0, s[0:1]
	v_lshl_add_u64 v[148:149], v[36:37], 0, v[106:107]
	global_store_short_d16_hi v[150:151], v98, off offset:256
	global_store_short_d16_hi v[148:149], v99, off
	v_lshl_add_u64 v[98:99], v[36:37], 0, v[108:109]
	global_store_short_d16_hi v[98:99], v100, off
	v_lshl_add_u64 v[98:99], v[36:37], 0, v[110:111]
	global_store_short_d16_hi v[98:99], v101, off
	v_lshl_add_u64 v[98:99], v[36:37], 0, v[0:1]
	global_store_short_d16_hi v[98:99], v78, off offset:32
	v_lshl_add_u64 v[98:99], v[36:37], 0, v[112:113]
	global_store_short_d16_hi v[98:99], v79, off
	v_lshl_add_u64 v[78:79], v[36:37], 0, v[114:115]
	global_store_short_d16_hi v[78:79], v80, off
	v_lshl_add_u64 v[78:79], v[36:37], 0, v[116:117]
	global_store_short_d16_hi v[78:79], v81, off
	v_lshl_add_u64 v[78:79], v[36:37], 0, v[122:123]
	global_store_short_d16_hi v[78:79], v70, off
	v_lshl_add_u64 v[78:79], v[36:37], 0, v[124:125]
	global_store_short_d16_hi v[78:79], v71, off
	v_lshl_add_u64 v[70:71], v[36:37], 0, v[134:135]
	global_store_short_d16_hi v[70:71], v72, off
	v_lshl_add_u64 v[70:71], v[36:37], 0, v[138:139]
	global_store_short_d16_hi v[70:71], v73, off
	v_lshl_add_u64 v[70:71], v[36:37], 0, v[136:137]
	global_store_short_d16_hi v[70:71], v58, off
	v_lshl_add_u64 v[70:71], v[36:37], 0, v[140:141]
	global_store_short_d16_hi v[70:71], v59, off
	v_lshl_add_u64 v[58:59], v[36:37], 0, v[130:131]
	global_store_short_d16_hi v[58:59], v60, off
	v_lshl_add_u64 v[58:59], v[36:37], 0, v[142:143]
	global_store_short_d16_hi v[58:59], v61, off
	v_lshl_add_u64 v[58:59], v[36:37], 0, v[132:133]
	global_store_short_d16_hi v[58:59], v50, off
	v_lshl_add_u64 v[58:59], v[36:37], 0, v[144:145]
	global_store_short_d16_hi v[58:59], v51, off
	v_lshl_add_u64 v[50:51], v[36:37], 0, v[126:127]
	global_store_short_d16_hi v[50:51], v52, off
	v_lshl_add_u64 v[50:51], v[36:37], 0, v[152:153]
	global_store_short_d16_hi v[50:51], v53, off
	v_lshl_add_u64 v[50:51], v[36:37], 0, v[128:129]
	global_store_short_d16_hi v[50:51], v42, off
	v_lshl_add_u64 v[50:51], v[36:37], 0, v[154:155]
	global_store_short_d16_hi v[50:51], v43, off
	v_lshl_add_u64 v[42:43], v[36:37], 0, v[118:119]
	global_store_short_d16_hi v[42:43], v44, off
	v_lshl_add_u64 v[42:43], v[36:37], 0, v[156:157]
	global_store_short_d16_hi v[42:43], v45, off
	v_lshl_add_u64 v[42:43], v[36:37], 0, v[120:121]
	global_store_short_d16_hi v[42:43], v30, off
	v_lshl_add_u64 v[42:43], v[36:37], 0, v[158:159]
	global_store_short_d16_hi v[42:43], v31, off
	v_lshl_add_u64 v[30:31], v[36:37], 0, v[62:63]
	global_store_short_d16_hi v[30:31], v32, off
	v_lshl_add_u64 v[30:31], v[36:37], 0, v[160:161]
	global_store_short_d16_hi v[30:31], v33, off
	v_lshl_add_u64 v[30:31], v[36:37], 0, v[64:65]
	global_store_short_d16_hi v[30:31], v22, off
	v_lshl_add_u64 v[30:31], v[36:37], 0, v[162:163]
	global_store_short_d16_hi v[30:31], v23, off
	v_lshl_add_u64 v[22:23], v[36:37], 0, v[34:35]
	global_store_short_d16_hi v[22:23], v24, off
	v_lshl_add_u64 v[22:23], v[36:37], 0, v[164:165]
	global_store_short_d16_hi v[22:23], v25, off
	v_add_u32_e32 v22, 0x80000, v146
	v_ashrrev_i32_e32 v23, 31, v22
	v_lshl_add_u64 v[22:23], v[22:23], 1, s[4:5]
	v_lshl_add_u64 v[30:31], v[22:23], 0, v[106:107]
	global_store_short_d16_hi v[30:31], v103, off
	v_lshl_add_u64 v[30:31], v[22:23], 0, v[108:109]
	v_lshl_add_u64 v[24:25], v[22:23], 0, v[0:1]
	global_store_short_d16_hi v[30:31], v104, off
	v_lshl_add_u64 v[30:31], v[22:23], 0, v[110:111]
	global_store_short_d16_hi v[24:25], v102, off
	global_store_short_d16_hi v[30:31], v105, off
	global_store_short_d16_hi v[24:25], v94, off offset:32
	v_lshl_add_u64 v[30:31], v[22:23], 0, v[112:113]
	global_store_short_d16_hi v[30:31], v95, off
	v_lshl_add_u64 v[30:31], v[22:23], 0, v[114:115]
	global_store_short_d16_hi v[30:31], v96, off
	v_lshl_add_u64 v[30:31], v[22:23], 0, v[116:117]
	global_store_short_d16_hi v[30:31], v97, off
	v_lshl_add_u64 v[30:31], v[22:23], 0, v[122:123]
	global_store_short_d16_hi v[30:31], v90, off
	v_lshl_add_u64 v[30:31], v[22:23], 0, v[124:125]
	global_store_short_d16_hi v[30:31], v91, off
	v_lshl_add_u64 v[30:31], v[22:23], 0, v[134:135]
	global_store_short_d16_hi v[30:31], v92, off
	v_lshl_add_u64 v[30:31], v[22:23], 0, v[138:139]
	global_store_short_d16_hi v[30:31], v93, off
	v_lshl_add_u64 v[30:31], v[22:23], 0, v[136:137]
	global_store_short_d16_hi v[30:31], v86, off
	v_lshl_add_u64 v[30:31], v[22:23], 0, v[140:141]
	global_store_short_d16_hi v[30:31], v87, off
	v_lshl_add_u64 v[30:31], v[22:23], 0, v[130:131]
	global_store_short_d16_hi v[30:31], v88, off
	v_lshl_add_u64 v[30:31], v[22:23], 0, v[142:143]
	global_store_short_d16_hi v[30:31], v89, off
	v_lshl_add_u64 v[30:31], v[22:23], 0, v[132:133]
	global_store_short_d16_hi v[30:31], v82, off
	v_lshl_add_u64 v[30:31], v[22:23], 0, v[144:145]
	global_store_short_d16_hi v[30:31], v83, off
	v_lshl_add_u64 v[30:31], v[22:23], 0, v[126:127]
	global_store_short_d16_hi v[30:31], v84, off
	v_lshl_add_u64 v[30:31], v[22:23], 0, v[152:153]
	global_store_short_d16_hi v[30:31], v85, off
	v_lshl_add_u64 v[30:31], v[22:23], 0, v[128:129]
	global_store_short_d16_hi v[30:31], v74, off
	v_lshl_add_u64 v[30:31], v[22:23], 0, v[154:155]
	global_store_short_d16_hi v[30:31], v75, off
	v_lshl_add_u64 v[30:31], v[22:23], 0, v[118:119]
	global_store_short_d16_hi v[30:31], v76, off
	v_lshl_add_u64 v[30:31], v[22:23], 0, v[156:157]
	global_store_short_d16_hi v[30:31], v77, off
	v_lshl_add_u64 v[30:31], v[22:23], 0, v[120:121]
	global_store_short_d16_hi v[30:31], v66, off
	v_lshl_add_u64 v[30:31], v[22:23], 0, v[158:159]
	global_store_short_d16_hi v[30:31], v67, off
	v_lshl_add_u64 v[30:31], v[22:23], 0, v[62:63]
	global_store_short_d16_hi v[30:31], v68, off
	v_lshl_add_u64 v[30:31], v[22:23], 0, v[160:161]
	global_store_short_d16_hi v[30:31], v69, off
	v_lshl_add_u64 v[30:31], v[22:23], 0, v[64:65]
	global_store_short_d16_hi v[30:31], v54, off
	v_lshl_add_u64 v[30:31], v[22:23], 0, v[162:163]
	global_store_short_d16_hi v[30:31], v55, off
	v_lshl_add_u64 v[30:31], v[22:23], 0, v[34:35]
	global_store_short_d16_hi v[30:31], v56, off
	v_lshl_add_u64 v[30:31], v[22:23], 0, v[164:165]
	v_lshl_add_u64 v[22:23], v[22:23], 0, s[0:1]
	v_lshl_add_u64 v[0:1], v[22:23], 0, v[0:1]
	global_store_short_d16_hi v[0:1], v38, off offset:32
	v_lshl_add_u64 v[0:1], v[22:23], 0, v[112:113]
	global_store_short_d16_hi v[0:1], v39, off
	v_lshl_add_u64 v[0:1], v[22:23], 0, v[114:115]
	global_store_short_d16_hi v[0:1], v40, off
	v_lshl_add_u64 v[0:1], v[22:23], 0, v[116:117]
	global_store_short_d16_hi v[0:1], v41, off
	v_lshl_add_u64 v[0:1], v[22:23], 0, v[122:123]
	global_store_short_d16_hi v[0:1], v26, off
	v_lshl_add_u64 v[0:1], v[22:23], 0, v[124:125]
	global_store_short_d16_hi v[0:1], v27, off
	v_lshl_add_u64 v[0:1], v[22:23], 0, v[134:135]
	global_store_short_d16_hi v[0:1], v28, off
	v_lshl_add_u64 v[0:1], v[22:23], 0, v[138:139]
	global_store_short_d16_hi v[0:1], v29, off
	v_lshl_add_u64 v[0:1], v[22:23], 0, v[136:137]
	global_store_short_d16_hi v[0:1], v18, off
	v_lshl_add_u64 v[0:1], v[22:23], 0, v[140:141]
	global_store_short_d16_hi v[0:1], v19, off
	v_lshl_add_u64 v[0:1], v[22:23], 0, v[130:131]
	global_store_short_d16_hi v[0:1], v20, off
	v_lshl_add_u64 v[0:1], v[22:23], 0, v[142:143]
	global_store_short_d16_hi v[0:1], v21, off
	v_lshl_add_u64 v[0:1], v[22:23], 0, v[132:133]
	global_store_short_d16_hi v[0:1], v14, off
	v_lshl_add_u64 v[0:1], v[22:23], 0, v[144:145]
	global_store_short_d16_hi v[0:1], v15, off
	v_lshl_add_u64 v[0:1], v[22:23], 0, v[126:127]
	global_store_short_d16_hi v[0:1], v16, off
	v_lshl_add_u64 v[0:1], v[22:23], 0, v[152:153]
	global_store_short_d16_hi v[0:1], v17, off
	v_lshl_add_u64 v[0:1], v[22:23], 0, v[128:129]
	global_store_short_d16_hi v[0:1], v10, off
	v_lshl_add_u64 v[0:1], v[22:23], 0, v[154:155]
	global_store_short_d16_hi v[0:1], v11, off
	v_lshl_add_u64 v[0:1], v[22:23], 0, v[118:119]
	global_store_short_d16_hi v[0:1], v12, off
	v_lshl_add_u64 v[0:1], v[22:23], 0, v[156:157]
	global_store_short_d16_hi v[0:1], v13, off
	v_lshl_add_u64 v[0:1], v[22:23], 0, v[120:121]
	global_store_short_d16_hi v[0:1], v6, off
	v_lshl_add_u64 v[0:1], v[22:23], 0, v[158:159]
	global_store_short_d16_hi v[0:1], v7, off
	v_lshl_add_u64 v[0:1], v[22:23], 0, v[62:63]
	global_store_short_d16_hi v[0:1], v8, off
	v_lshl_add_u64 v[0:1], v[22:23], 0, v[160:161]
	global_store_short_d16_hi v[0:1], v9, off
	v_lshl_add_u64 v[0:1], v[22:23], 0, v[64:65]
	global_store_short_d16_hi v[24:25], v46, off offset:256
	v_lshl_add_u64 v[24:25], v[22:23], 0, v[106:107]
	global_store_short_d16_hi v[0:1], v2, off
	v_lshl_add_u64 v[0:1], v[22:23], 0, v[162:163]
	global_store_short_d16_hi v[24:25], v47, off
	v_lshl_add_u64 v[24:25], v[22:23], 0, v[108:109]
	global_store_short_d16_hi v[0:1], v3, off
	v_lshl_add_u64 v[0:1], v[22:23], 0, v[34:35]
	global_store_short_d16_hi v[24:25], v48, off
	v_lshl_add_u64 v[24:25], v[22:23], 0, v[110:111]
	global_store_short_d16_hi v[0:1], v4, off
	v_lshl_add_u64 v[0:1], v[22:23], 0, v[164:165]
	global_store_short_d16_hi v[30:31], v57, off
	global_store_short_d16_hi v[24:25], v49, off
	global_store_short_d16_hi v[0:1], v5, off
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z22crr_exact_8wave_kernel23crr_exact_8wave_globals
		.amdhsa_group_segment_fixed_size 139264
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
	.text
.Lfunc_end2:
	.size	_Z22crr_exact_8wave_kernel23crr_exact_8wave_globals, .Lfunc_end2-_Z22crr_exact_8wave_kernel23crr_exact_8wave_globals
                                        ; -- End function
	.set _Z22crr_exact_8wave_kernel23crr_exact_8wave_globals.num_vgpr, 256
	.set _Z22crr_exact_8wave_kernel23crr_exact_8wave_globals.num_agpr, 0
	.set _Z22crr_exact_8wave_kernel23crr_exact_8wave_globals.numbered_sgpr, 24
	.set _Z22crr_exact_8wave_kernel23crr_exact_8wave_globals.private_seg_size, 0
	.set _Z22crr_exact_8wave_kernel23crr_exact_8wave_globals.uses_vcc, 1
	.set _Z22crr_exact_8wave_kernel23crr_exact_8wave_globals.uses_flat_scratch, 0
	.set _Z22crr_exact_8wave_kernel23crr_exact_8wave_globals.has_dyn_sized_stack, 0
	.set _Z22crr_exact_8wave_kernel23crr_exact_8wave_globals.has_recursion, 0
	.set _Z22crr_exact_8wave_kernel23crr_exact_8wave_globals.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 8100
; TotalNumSgprs: 30
; NumVgprs: 256
; NumAgprs: 0
; TotalNumVgprs: 256
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 139264 bytes/workgroup (compile time only)
; SGPRBlocks: 12
; VGPRBlocks: 31
; NumSGPRsForWavesPerEU: 102
; NumVGPRsForWavesPerEU: 256
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
	.section	.text._Z11gemm_kernelIL6Layout0EEv14layout_globals,"axG",@progbits,_Z11gemm_kernelIL6Layout0EEv14layout_globals,comdat
	.protected	_Z11gemm_kernelIL6Layout0EEv14layout_globals ; -- Begin function _Z11gemm_kernelIL6Layout0EEv14layout_globals
	.globl	_Z11gemm_kernelIL6Layout0EEv14layout_globals
	.p2align	8
	.type	_Z11gemm_kernelIL6Layout0EEv14layout_globals,@function
