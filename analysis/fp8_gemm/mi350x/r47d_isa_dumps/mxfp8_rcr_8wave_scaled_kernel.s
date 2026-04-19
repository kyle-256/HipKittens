_Z29rcr_exact_8wave_scaled_kernelILb1ELi2EEv14layout_globals: ; @_Z29rcr_exact_8wave_scaled_kernelILb1ELi2EEv14layout_globals
; %bb.0:
	s_load_dwordx2 s[40:41], s[0:1], 0x0
	s_load_dword s8, s[0:1], 0x20
	s_load_dwordx2 s[34:35], s[0:1], 0x30
	s_load_dword s7, s[0:1], 0x50
	s_load_dword s43, s[0:1], 0x108
	s_ashr_i32 s3, s2, 31
	s_lshr_b32 s3, s3, 28
	s_add_i32 s3, s2, s3
	s_ashr_i32 s33, s3, 4
	s_and_b32 s3, s3, -16
	s_sub_i32 s2, s2, s3
	v_lshlrev_b32_e32 v1, 4, v0
	s_movk_i32 s3, 0x70
	s_waitcnt lgkmcnt(0)
	s_ashr_i32 s12, s43, 31
	v_bitop3_b32 v2, v1, s3, v0 bitop3:0x48
	s_lshr_b32 s3, s12, 27
	s_add_i32 s3, s43, s3
	v_lshrrev_b32_e32 v3, 3, v0
	s_ashr_i32 s3, s3, 5
	v_or_b32_e32 v4, 64, v3
	s_add_i32 s3, s3, 7
	v_mad_u64_u32 v[152:153], s[4:5], v3, s8, v[2:3]
	v_mad_u64_u32 v[150:151], s[4:5], v4, s8, v[2:3]
	v_mad_u64_u32 v[156:157], s[4:5], v3, s7, v[2:3]
	v_mad_u64_u32 v[154:155], s[4:5], v4, s7, v[2:3]
	s_lshl_b32 s42, s2, 8
	s_and_b32 s3, s3, -8
	s_lshl_b32 s6, s3, 7
	s_lshl_b32 s44, s3, 6
	s_ashr_i32 s3, s2, 31
	s_mul_i32 s4, s42, s7
	s_ashr_i32 s13, s33, 31
	s_lshl_b64 s[2:3], s[2:3], 2
	s_ashr_i32 s5, s4, 31
	v_and_b32_e32 v153, 0x1c00, v1
	s_add_u32 s28, s34, s4
	v_or_b32_e32 v161, 0x10000, v153
	s_addc_u32 s29, s35, s5
	s_lshl_b32 s22, s7, 7
	s_mov_b32 s7, 0x110000
	v_readfirstlane_b32 s5, v161
	v_or_b32_e32 v162, 0x2000, v161
	s_mov_b32 s30, s22
	s_mov_b32 s31, s7
	s_mov_b32 m0, s5
	v_readfirstlane_b32 s5, v162
	buffer_load_dwordx4 v156, s[28:31], 0 offen lds
	s_mov_b32 m0, s5
	s_mul_i32 s5, s33, s8
	buffer_load_dwordx4 v154, s[28:31], 0 offen lds
	s_lshl_b32 s30, s5, 8
	s_ashr_i32 s5, s30, 31
	s_add_u32 s24, s40, s30
	s_addc_u32 s25, s41, s5
	s_lshl_b32 s14, s8, 7
	v_readfirstlane_b32 s5, v153
	v_or_b32_e32 v160, 0x2000, v153
	s_mov_b32 s26, s14
	s_mov_b32 s27, s7
	s_mov_b32 m0, s5
	v_readfirstlane_b32 s5, v160
	buffer_load_dwordx4 v152, s[24:27], 0 offen lds
	s_mov_b32 m0, s5
	v_or_b32_e32 v158, 0x4000, v161
	buffer_load_dwordx4 v150, s[24:27], 0 offen lds
	s_add_i32 s27, s4, s22
	s_ashr_i32 s4, s27, 31
	s_add_u32 s8, s34, s27
	s_addc_u32 s9, s35, s4
	v_readfirstlane_b32 s4, v158
	v_or_b32_e32 v159, 0x6000, v161
	s_mov_b32 s10, s22
	s_mov_b32 s11, s7
	s_mov_b32 m0, s4
	v_readfirstlane_b32 s4, v159
	s_add_i32 s30, s30, s14
	buffer_load_dwordx4 v156, s[8:11], 0 offen lds
	s_mov_b32 m0, s4
	s_ashr_i32 s4, s30, 31
	s_add_u32 s16, s40, s30
	v_or_b32_e32 v163, 0x4000, v153
	s_addc_u32 s17, s41, s4
	v_readfirstlane_b32 s4, v163
	v_or_b32_e32 v164, 0x6000, v153
	buffer_load_dwordx4 v154, s[8:11], 0 offen lds
	s_mov_b32 s18, s14
	s_mov_b32 s19, s7
	s_mov_b32 m0, s4
	v_readfirstlane_b32 s4, v164
	buffer_load_dwordx4 v152, s[16:19], 0 offen lds
	s_mov_b32 m0, s4
	v_lshrrev_b32_e32 v1, 8, v0
	buffer_load_dwordx4 v150, s[16:19], 0 offen lds
	s_load_dwordx2 s[4:5], s[0:1], 0x60
	s_load_dwordx2 s[10:11], s[0:1], 0x90
	v_bfe_u32 v191, v0, 6, 2
	v_lshl_or_b32 v4, s33, 1, v1
	v_or_b32_e32 v6, s2, v191
	s_waitcnt lgkmcnt(0)
	v_mov_b64_e32 v[2:3], s[4:5]
	v_mad_u64_u32 v[2:3], s[4:5], v4, s6, v[2:3]
	v_mov_b64_e32 v[4:5], s[10:11]
	s_mul_i32 s13, s13, s6
	s_mul_i32 s4, s3, s44
	v_mad_u64_u32 v[4:5], s[2:3], v6, s44, v[4:5]
	v_add_u32_e32 v3, s13, v3
	v_add_u32_e32 v5, s4, v5
	v_readfirstlane_b32 s4, v2
	v_readfirstlane_b32 s5, v3
	v_readfirstlane_b32 s31, v4
	v_readfirstlane_b32 s37, v5
	s_mov_b32 s36, 0
	v_cmp_eq_u32_e32 vcc, 1, v1
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB2_2
; %bb.1:
	s_barrier
.LBB2_2:
	s_or_b64 exec, exec, s[2:3]
	s_lshr_b32 s2, s12, 25
	s_add_i32 s2, s43, s2
	s_ashr_i32 s26, s2, 7
	v_or_b32_e32 v165, 0x8000, v161
	s_add_u32 s20, s28, 0x80
	v_readfirstlane_b32 s2, v165
	v_or_b32_e32 v166, 0xa000, v161
	s_addc_u32 s21, s29, 0
	s_mov_b32 s23, s7
	s_mov_b32 m0, s2
	v_readfirstlane_b32 s2, v166
	v_or_b32_e32 v167, 0x8000, v153
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	buffer_load_dwordx4 v156, s[20:23], 0 offen lds
	s_mov_b32 m0, s2
	s_add_u32 s12, s24, 0x80
	v_readfirstlane_b32 s2, v167
	v_or_b32_e32 v168, 0xa000, v153
	buffer_load_dwordx4 v154, s[20:23], 0 offen lds
	s_addc_u32 s13, s25, 0
	s_mov_b32 s15, s7
	s_mov_b32 m0, s2
	v_readfirstlane_b32 s2, v168
	v_or_b32_e32 v169, 0xc000, v161
	buffer_load_dwordx4 v152, s[12:15], 0 offen lds
	s_mov_b32 m0, s2
	s_add_u32 s20, s8, 0x80
	v_readfirstlane_b32 s2, v169
	v_or_b32_e32 v170, 0xe000, v161
	buffer_load_dwordx4 v150, s[12:15], 0 offen lds
	s_addc_u32 s21, s9, 0
	s_mov_b32 m0, s2
	v_readfirstlane_b32 s2, v170
	buffer_load_dwordx4 v156, s[20:23], 0 offen lds
	s_mov_b32 m0, s2
	s_mov_b64 s[10:11], s[6:7]
	buffer_load_dwordx4 v154, s[20:23], 0 offen lds
	s_load_dwordx2 s[2:3], s[0:1], 0xc0
	s_load_dwordx2 s[18:19], s[0:1], 0xe0
	s_waitcnt lgkmcnt(0)
	s_add_i32 s19, s26, -2
	s_lshr_b32 s10, s19, 31
	s_mov_b64 s[8:9], s[4:5]
	s_add_i32 s10, s19, s10
	v_and_b32_e32 v190, 15, v0
	s_mov_b32 s8, s31
	s_ashr_i32 s31, s10, 1
	v_bfe_u32 v174, v0, 4, 2
	s_mov_b32 s9, s37
	s_cmpk_lt_i32 s43, 0x200
	s_mov_b32 s10, s44
	v_lshlrev_b32_e32 v176, 4, v190
	v_lshlrev_b32_e32 v175, 3, v190
	v_and_b32_e32 v155, 48, v0
	v_lshlrev_b32_e32 v157, 7, v0
	v_lshlrev_b32_e32 v151, 3, v0
	;;#ASMSTART
	s_waitcnt vmcnt(6)
	;;#ASMEND
	s_barrier
	s_cbranch_scc1 .LBB2_5
; %bb.3:
	s_movk_i32 s12, 0x780
	v_and_or_b32 v2, v157, s12, v155
	v_lshl_or_b32 v3, v191, 12, v2
	v_or_b32_e32 v4, 0x10000, v3
	s_movk_i32 s12, 0x70
	v_bitop3_b32 v173, v151, v4, s12 bitop3:0x6c
	v_or_b32_e32 v4, 0x10040, v3
	v_lshl_or_b32 v2, v1, 13, v2
	v_bitop3_b32 v177, v151, v4, s12 bitop3:0x6c
	v_or_b32_e32 v4, 64, v2
	v_bitop3_b32 v179, v151, v4, s12 bitop3:0x6c
	v_or_b32_e32 v4, 0x14000, v3
	v_bitop3_b32 v182, v151, v4, s12 bitop3:0x6c
	v_or_b32_e32 v4, 0x14040, v3
	v_bitop3_b32 v183, v151, v4, s12 bitop3:0x6c
	v_or_b32_e32 v4, 0x4000, v2
	v_bitop3_b32 v184, v151, v4, s12 bitop3:0x6c
	v_or_b32_e32 v4, 0x4040, v2
	v_bitop3_b32 v185, v151, v4, s12 bitop3:0x6c
	v_or_b32_e32 v4, 0x18000, v3
	v_bitop3_b32 v186, v151, v4, s12 bitop3:0x6c
	v_or_b32_e32 v4, 0x18040, v3
	v_bitop3_b32 v187, v151, v4, s12 bitop3:0x6c
	v_or_b32_e32 v4, 0x8000, v2
	v_bitop3_b32 v188, v151, v4, s12 bitop3:0x6c
	v_or_b32_e32 v4, 0x8040, v2
	v_bitop3_b32 v189, v151, v4, s12 bitop3:0x6c
	v_or_b32_e32 v4, 0x1c000, v3
	v_or_b32_e32 v3, 0x1c040, v3
	v_bitop3_b32 v178, v151, v2, s12 bitop3:0x6c
	v_bitop3_b32 v195, v151, v3, s12 bitop3:0x6c
	v_or_b32_e32 v3, 0xc000, v2
	v_or_b32_e32 v2, 0xc040, v2
	v_mov_b32_e32 v22, 0
	v_lshl_or_b32 v171, v174, 8, v176
	v_lshl_or_b32 v172, v174, 7, v175
	v_or_b32_e32 v180, 0xc000, v153
	v_or_b32_e32 v181, 0xe000, v153
	v_bitop3_b32 v194, v151, v4, s12 bitop3:0x6c
	v_bitop3_b32 v196, v151, v3, s12 bitop3:0x6c
	v_bitop3_b32 v197, v151, v2, s12 bitop3:0x6c
	s_max_i32 s36, s31, 1
	s_mov_b32 s37, 0
	s_mov_b32 s15, 0x110000
	s_mov_b32 s38, 0
	s_mov_b32 s39, 0
	v_mov_b32_e32 v23, v22
	v_mov_b32_e32 v24, v22
	v_mov_b32_e32 v25, v22
	v_mov_b32_e32 v26, v22
	v_mov_b32_e32 v27, v22
	v_mov_b32_e32 v28, v22
	v_mov_b32_e32 v29, v22
	v_mov_b32_e32 v30, v22
	v_mov_b32_e32 v31, v22
	v_mov_b32_e32 v32, v22
	v_mov_b32_e32 v33, v22
	v_mov_b32_e32 v34, v22
	v_mov_b32_e32 v35, v22
	v_mov_b32_e32 v36, v22
	v_mov_b32_e32 v37, v22
	v_mov_b32_e32 v38, v22
	v_mov_b32_e32 v39, v22
	v_mov_b32_e32 v40, v22
	v_mov_b32_e32 v41, v22
	v_mov_b32_e32 v42, v22
	v_mov_b32_e32 v43, v22
	v_mov_b32_e32 v44, v22
	v_mov_b32_e32 v45, v22
	v_mov_b32_e32 v46, v22
	v_mov_b32_e32 v47, v22
	v_mov_b32_e32 v48, v22
	v_mov_b32_e32 v49, v22
	v_mov_b32_e32 v50, v22
	v_mov_b32_e32 v51, v22
	v_mov_b32_e32 v52, v22
	v_mov_b32_e32 v53, v22
	v_mov_b32_e32 v62, v22
	v_mov_b32_e32 v63, v22
	v_mov_b32_e32 v64, v22
	v_mov_b32_e32 v65, v22
	v_mov_b32_e32 v78, v22
	v_mov_b32_e32 v79, v22
	v_mov_b32_e32 v80, v22
	v_mov_b32_e32 v81, v22
	v_mov_b32_e32 v66, v22
	v_mov_b32_e32 v67, v22
	v_mov_b32_e32 v68, v22
	v_mov_b32_e32 v69, v22
	v_mov_b32_e32 v82, v22
	v_mov_b32_e32 v83, v22
	v_mov_b32_e32 v84, v22
	v_mov_b32_e32 v85, v22
	v_mov_b32_e32 v54, v22
	v_mov_b32_e32 v55, v22
	v_mov_b32_e32 v56, v22
	v_mov_b32_e32 v57, v22
	v_mov_b32_e32 v70, v22
	v_mov_b32_e32 v71, v22
	v_mov_b32_e32 v72, v22
	v_mov_b32_e32 v73, v22
	v_mov_b32_e32 v58, v22
	v_mov_b32_e32 v59, v22
	v_mov_b32_e32 v60, v22
	v_mov_b32_e32 v61, v22
	v_mov_b32_e32 v74, v22
	v_mov_b32_e32 v75, v22
	v_mov_b32_e32 v76, v22
	v_mov_b32_e32 v77, v22
	v_mov_b32_e32 v90, v22
	v_mov_b32_e32 v91, v22
	v_mov_b32_e32 v92, v22
	v_mov_b32_e32 v93, v22
	v_mov_b32_e32 v86, v22
	v_mov_b32_e32 v87, v22
	v_mov_b32_e32 v88, v22
	v_mov_b32_e32 v89, v22
	v_mov_b32_e32 v94, v22
	v_mov_b32_e32 v95, v22
	v_mov_b32_e32 v96, v22
	v_mov_b32_e32 v97, v22
	v_mov_b32_e32 v98, v22
	v_mov_b32_e32 v99, v22
	v_mov_b32_e32 v100, v22
	v_mov_b32_e32 v101, v22
	v_mov_b32_e32 v102, v22
	v_mov_b32_e32 v103, v22
	v_mov_b32_e32 v104, v22
	v_mov_b32_e32 v105, v22
	v_mov_b32_e32 v106, v22
	v_mov_b32_e32 v107, v22
	v_mov_b32_e32 v108, v22
	v_mov_b32_e32 v109, v22
	v_mov_b32_e32 v110, v22
	v_mov_b32_e32 v111, v22
	v_mov_b32_e32 v112, v22
	v_mov_b32_e32 v113, v22
	v_mov_b32_e32 v114, v22
	v_mov_b32_e32 v115, v22
	v_mov_b32_e32 v116, v22
	v_mov_b32_e32 v117, v22
	v_mov_b32_e32 v118, v22
	v_mov_b32_e32 v119, v22
	v_mov_b32_e32 v120, v22
	v_mov_b32_e32 v121, v22
	v_mov_b32_e32 v122, v22
	v_mov_b32_e32 v123, v22
	v_mov_b32_e32 v124, v22
	v_mov_b32_e32 v125, v22
	v_mov_b32_e32 v126, v22
	v_mov_b32_e32 v127, v22
	v_mov_b32_e32 v128, v22
	v_mov_b32_e32 v129, v22
	v_mov_b32_e32 v146, v22
	v_mov_b32_e32 v147, v22
	v_mov_b32_e32 v148, v22
	v_mov_b32_e32 v149, v22
	v_mov_b32_e32 v138, v22
	v_mov_b32_e32 v139, v22
	v_mov_b32_e32 v140, v22
	v_mov_b32_e32 v141, v22
	v_mov_b32_e32 v134, v22
	v_mov_b32_e32 v135, v22
	v_mov_b32_e32 v136, v22
	v_mov_b32_e32 v137, v22
	v_mov_b32_e32 v142, v22
	v_mov_b32_e32 v143, v22
	v_mov_b32_e32 v144, v22
	v_mov_b32_e32 v145, v22
	v_mov_b32_e32 v130, v22
	v_mov_b32_e32 v131, v22
	v_mov_b32_e32 v132, v22
	v_mov_b32_e32 v133, v22
.LBB2_4:                                ; =>This Inner Loop Header: Depth=1
	buffer_load_dwordx4 v[18:21], v171, s[4:7], s38 offen
	buffer_load_dwordx2 v[192:193], v172, s[8:11], s37 offen
	;;#ASMSTART
	ds_read_b128 v[2:5], v173 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[10:13], v173 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[6:9], v177 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[14:17], v177 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[198:201], v178 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[206:209], v178 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[214:217], v178 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[222:225], v178 offset:0x1800

	;;#ASMEND
	s_add_i32 s44, s30, s39
	;;#ASMSTART
	ds_read_b128 v[202:205], v179 offset:0

	;;#ASMEND
	s_add_i32 s12, s44, 0x80
	;;#ASMSTART
	ds_read_b128 v[210:213], v179 offset:0x800

	;;#ASMEND
	s_ashr_i32 s13, s12, 31
	;;#ASMSTART
	ds_read_b128 v[218:221], v179 offset:0x1000

	;;#ASMEND
	s_add_u32 s12, s40, s12
	v_readfirstlane_b32 s20, v180
	;;#ASMSTART
	ds_read_b128 v[226:229], v179 offset:0x1800

	;;#ASMEND
	s_addc_u32 s13, s41, s13
	s_mov_b32 m0, s20
	v_readfirstlane_b32 s20, v181
	buffer_load_dwordx4 v152, s[12:15], 0 offen lds
	s_mov_b32 m0, s20
	s_nop 0
	buffer_load_dwordx4 v150, s[12:15], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	s_waitcnt vmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[146:149], v[198:205], v[2:9], v[146:149], v18, v192 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[138:141], v[198:205], v[10:17], v[138:141], v18, v192 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[134:137], v[206:213], v[2:9], v[134:137], v18, v192 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[142:145], v[206:213], v[10:17], v[142:145], v18, v192 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[130:133], v[214:221], v[2:9], v[130:133], v20, v192 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[126:129], v[214:221], v[10:17], v[126:129], v20, v192 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[122:125], v[222:229], v[2:9], v[122:125], v20, v192 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[118:121], v[222:229], v[10:17], v[118:121], v20, v192 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b128 v[230:233], v182 offset:0

	;;#ASMEND
	s_add_i32 s43, s39, 0x100
	;;#ASMSTART
	ds_read_b128 v[238:241], v182 offset:0x800

	;;#ASMEND
	s_ashr_i32 s13, s43, 31
	;;#ASMSTART
	ds_read_b128 v[234:237], v183 offset:0

	;;#ASMEND
	s_add_u32 s20, s28, s43
	v_readfirstlane_b32 s12, v161
	;;#ASMSTART
	ds_read_b128 v[242:245], v183 offset:0x800

	;;#ASMEND
	s_addc_u32 s21, s29, s13
	s_mov_b32 s23, s15
	s_mov_b32 m0, s12
	v_readfirstlane_b32 s12, v162
	buffer_load_dwordx4 v156, s[20:23], 0 offen lds
	s_mov_b32 m0, s12
	s_nop 0
	buffer_load_dwordx4 v154, s[20:23], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[198:205], v[230:237], v[90:93], v18, v193 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[198:205], v[238:245], v[86:89], v18, v193 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[206:213], v[230:237], v[94:97], v18, v193 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[98:101], v[206:213], v[238:245], v[98:101], v18, v193 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[102:105], v[214:221], v[230:237], v[102:105], v20, v193 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[106:109], v[214:221], v[238:245], v[106:109], v20, v193 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[110:113], v[222:229], v[230:237], v[110:113], v20, v193 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[114:117], v[222:229], v[238:245], v[114:117], v20, v193 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[198:201], v184 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[206:209], v184 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[214:217], v184 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[222:225], v184 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[202:205], v185 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[210:213], v185 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[218:221], v185 offset:0x1000

	;;#ASMEND
	s_add_u32 s12, s24, s43
	v_readfirstlane_b32 s20, v153
	;;#ASMSTART
	ds_read_b128 v[226:229], v185 offset:0x1800

	;;#ASMEND
	s_addc_u32 s13, s25, s13
	s_mov_b32 m0, s20
	v_readfirstlane_b32 s20, v160
	buffer_load_dwordx4 v152, s[12:15], 0 offen lds
	s_mov_b32 m0, s20
	s_nop 0
	buffer_load_dwordx4 v150, s[12:15], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[198:205], v[2:9], v[62:65], v19, v192 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[198:205], v[10:17], v[78:81], v19, v192 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[206:213], v[2:9], v[66:69], v19, v192 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[206:213], v[10:17], v[82:85], v19, v192 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[214:221], v[2:9], v[54:57], v21, v192 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[214:221], v[10:17], v[70:73], v21, v192 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[222:229], v[2:9], v[58:61], v21, v192 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[222:229], v[10:17], v[74:77], v21, v192 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_i32 s45, s27, s39
	s_add_i32 s12, s45, 0x100
	s_ashr_i32 s13, s12, 31
	s_add_u32 s20, s34, s12
	v_readfirstlane_b32 s12, v158
	s_addc_u32 s21, s35, s13
	s_mov_b32 m0, s12
	v_readfirstlane_b32 s12, v159
	buffer_load_dwordx4 v156, s[20:23], 0 offen lds
	s_mov_b32 m0, s12
	s_nop 0
	buffer_load_dwordx4 v154, s[20:23], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(6)
	;;#ASMEND
	s_barrier
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[198:205], v[230:237], v[22:25], v19, v193 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[198:205], v[238:245], v[26:29], v19, v193 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[206:213], v[230:237], v[30:33], v19, v193 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[206:213], v[238:245], v[34:37], v19, v193 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[214:221], v[230:237], v[38:41], v21, v193 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[214:221], v[238:245], v[42:45], v21, v193 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], v[222:229], v[230:237], v[46:49], v21, v193 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[222:229], v[238:245], v[50:53], v21, v193 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[2:5], v186 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[10:13], v186 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[6:9], v187 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[14:17], v187 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[198:201], v188 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[206:209], v188 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[214:217], v188 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[222:225], v188 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[202:205], v189 offset:0

	;;#ASMEND
	s_addk_i32 s44, 0x100
	;;#ASMSTART
	ds_read_b128 v[210:213], v189 offset:0x800

	;;#ASMEND
	s_ashr_i32 s13, s44, 31
	;;#ASMSTART
	ds_read_b128 v[218:221], v189 offset:0x1000

	;;#ASMEND
	s_add_u32 s12, s40, s44
	v_readfirstlane_b32 s20, v163
	;;#ASMSTART
	ds_read_b128 v[226:229], v189 offset:0x1800

	;;#ASMEND
	s_addc_u32 s13, s41, s13
	s_mov_b32 m0, s20
	v_readfirstlane_b32 s20, v164
	buffer_load_dwordx4 v152, s[12:15], 0 offen lds
	s_mov_b32 m0, s20
	s_nop 0
	buffer_load_dwordx4 v150, s[12:15], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[146:149], v[198:205], v[2:9], v[146:149], v18, v192 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[138:141], v[198:205], v[10:17], v[138:141], v18, v192 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[134:137], v[206:213], v[2:9], v[134:137], v18, v192 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[142:145], v[206:213], v[10:17], v[142:145], v18, v192 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[130:133], v[214:221], v[2:9], v[130:133], v20, v192 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[126:129], v[214:221], v[10:17], v[126:129], v20, v192 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[122:125], v[222:229], v[2:9], v[122:125], v20, v192 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[118:121], v[222:229], v[10:17], v[118:121], v20, v192 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b128 v[230:233], v194 offset:0

	;;#ASMEND
	s_add_i32 s12, s39, 0x180
	;;#ASMSTART
	ds_read_b128 v[238:241], v194 offset:0x800

	;;#ASMEND
	s_ashr_i32 s13, s12, 31
	;;#ASMSTART
	ds_read_b128 v[234:237], v195 offset:0

	;;#ASMEND
	s_add_u32 s20, s28, s12
	v_readfirstlane_b32 s39, v165
	;;#ASMSTART
	ds_read_b128 v[242:245], v195 offset:0x800

	;;#ASMEND
	s_addc_u32 s21, s29, s13
	s_mov_b32 m0, s39
	v_readfirstlane_b32 s39, v166
	buffer_load_dwordx4 v156, s[20:23], 0 offen lds
	s_mov_b32 m0, s39
	s_nop 0
	buffer_load_dwordx4 v154, s[20:23], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[198:205], v[230:237], v[90:93], v18, v193 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[198:205], v[238:245], v[86:89], v18, v193 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[206:213], v[230:237], v[94:97], v18, v193 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[98:101], v[206:213], v[238:245], v[98:101], v18, v193 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[102:105], v[214:221], v[230:237], v[102:105], v20, v193 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[106:109], v[214:221], v[238:245], v[106:109], v20, v193 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[110:113], v[222:229], v[230:237], v[110:113], v20, v193 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[114:117], v[222:229], v[238:245], v[114:117], v20, v193 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[198:201], v196 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[206:209], v196 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[214:217], v196 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[222:225], v196 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[202:205], v197 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[210:213], v197 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[218:221], v197 offset:0x1000

	;;#ASMEND
	s_add_u32 s12, s24, s12
	v_readfirstlane_b32 s20, v167
	;;#ASMSTART
	ds_read_b128 v[226:229], v197 offset:0x1800

	;;#ASMEND
	s_addc_u32 s13, s25, s13
	s_mov_b32 m0, s20
	v_readfirstlane_b32 s20, v168
	buffer_load_dwordx4 v152, s[12:15], 0 offen lds
	s_mov_b32 m0, s20
	s_nop 0
	buffer_load_dwordx4 v150, s[12:15], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[198:205], v[2:9], v[62:65], v19, v192 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[198:205], v[10:17], v[78:81], v19, v192 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[206:213], v[2:9], v[66:69], v19, v192 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[206:213], v[10:17], v[82:85], v19, v192 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[214:221], v[2:9], v[54:57], v21, v192 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[214:221], v[10:17], v[70:73], v21, v192 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[222:229], v[2:9], v[58:61], v21, v192 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[222:229], v[10:17], v[74:77], v21, v192 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_addk_i32 s45, 0x180
	s_ashr_i32 s12, s45, 31
	s_add_u32 s20, s34, s45
	s_addc_u32 s21, s35, s12
	v_readfirstlane_b32 s12, v169
	s_mov_b32 m0, s12
	v_readfirstlane_b32 s12, v170
	buffer_load_dwordx4 v156, s[20:23], 0 offen lds
	s_mov_b32 m0, s12
	s_nop 0
	buffer_load_dwordx4 v154, s[20:23], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(6)
	;;#ASMEND
	s_barrier
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[198:205], v[230:237], v[22:25], v19, v193 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[198:205], v[238:245], v[26:29], v19, v193 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[206:213], v[230:237], v[30:33], v19, v193 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[206:213], v[238:245], v[34:37], v19, v193 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[214:221], v[230:237], v[38:41], v21, v193 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[214:221], v[238:245], v[42:45], v21, v193 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], v[222:229], v[230:237], v[46:49], v21, v193 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[222:229], v[238:245], v[50:53], v21, v193 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_add_i32 s36, s36, -1
	s_addk_i32 s38, 0x400
	s_addk_i32 s37, 0x200
	s_cmp_eq_u32 s36, 0
	s_mov_b32 s39, s43
	s_barrier
	s_cbranch_scc0 .LBB2_4
	s_branch .LBB2_6
.LBB2_5:
	s_mov_b32 s37, s36
	s_mov_b32 s38, s36
	s_mov_b32 s39, s36
	v_mov_b64_e32 v[22:23], s[36:37]
	v_mov_b64_e32 v[24:25], s[38:39]
	v_mov_b64_e32 v[28:29], v[24:25]
	v_mov_b64_e32 v[32:33], v[24:25]
	v_mov_b64_e32 v[36:37], v[24:25]
	v_mov_b64_e32 v[40:41], v[24:25]
	v_mov_b64_e32 v[44:45], v[24:25]
	v_mov_b64_e32 v[48:49], v[24:25]
	v_mov_b64_e32 v[52:53], v[24:25]
	v_mov_b64_e32 v[64:65], v[24:25]
	v_mov_b64_e32 v[80:81], v[24:25]
	v_mov_b64_e32 v[68:69], v[24:25]
	v_mov_b64_e32 v[84:85], v[24:25]
	v_mov_b64_e32 v[56:57], v[24:25]
	v_mov_b64_e32 v[72:73], v[24:25]
	v_mov_b64_e32 v[60:61], v[24:25]
	v_mov_b64_e32 v[76:77], v[24:25]
	v_mov_b64_e32 v[92:93], v[24:25]
	v_mov_b64_e32 v[88:89], v[24:25]
	v_mov_b64_e32 v[96:97], v[24:25]
	v_mov_b64_e32 v[100:101], v[24:25]
	v_mov_b64_e32 v[104:105], v[24:25]
	v_mov_b64_e32 v[108:109], v[24:25]
	v_mov_b64_e32 v[112:113], v[24:25]
	v_mov_b64_e32 v[116:117], v[24:25]
	v_mov_b64_e32 v[120:121], v[24:25]
	v_mov_b64_e32 v[124:125], v[24:25]
	v_mov_b64_e32 v[128:129], v[24:25]
	v_mov_b64_e32 v[148:149], v[24:25]
	v_mov_b64_e32 v[140:141], v[24:25]
	v_mov_b64_e32 v[136:137], v[24:25]
	v_mov_b64_e32 v[144:145], v[24:25]
	v_mov_b64_e32 v[132:133], v[24:25]
	v_mov_b64_e32 v[26:27], v[22:23]
	v_mov_b64_e32 v[30:31], v[22:23]
	v_mov_b64_e32 v[34:35], v[22:23]
	v_mov_b64_e32 v[38:39], v[22:23]
	v_mov_b64_e32 v[42:43], v[22:23]
	v_mov_b64_e32 v[46:47], v[22:23]
	v_mov_b64_e32 v[50:51], v[22:23]
	v_mov_b64_e32 v[62:63], v[22:23]
	v_mov_b64_e32 v[78:79], v[22:23]
	v_mov_b64_e32 v[66:67], v[22:23]
	v_mov_b64_e32 v[82:83], v[22:23]
	v_mov_b64_e32 v[54:55], v[22:23]
	v_mov_b64_e32 v[70:71], v[22:23]
	v_mov_b64_e32 v[58:59], v[22:23]
	v_mov_b64_e32 v[74:75], v[22:23]
	v_mov_b64_e32 v[90:91], v[22:23]
	v_mov_b64_e32 v[86:87], v[22:23]
	v_mov_b64_e32 v[94:95], v[22:23]
	v_mov_b64_e32 v[98:99], v[22:23]
	v_mov_b64_e32 v[102:103], v[22:23]
	v_mov_b64_e32 v[106:107], v[22:23]
	v_mov_b64_e32 v[110:111], v[22:23]
	v_mov_b64_e32 v[114:115], v[22:23]
	v_mov_b64_e32 v[118:119], v[22:23]
	v_mov_b64_e32 v[122:123], v[22:23]
	v_mov_b64_e32 v[126:127], v[22:23]
	v_mov_b64_e32 v[146:147], v[22:23]
	v_mov_b64_e32 v[138:139], v[22:23]
	v_mov_b64_e32 v[134:135], v[22:23]
	v_mov_b64_e32 v[142:143], v[22:23]
	v_mov_b64_e32 v[130:131], v[22:23]
                                        ; implicit-def: $vgpr193
                                        ; implicit-def: $vgpr19
.LBB2_6:
	s_load_dwordx4 s[36:39], s[0:1], 0xf4
	s_lshl_b32 s0, s31, 1
	s_cmp_lg_u32 s19, s0
	s_waitcnt lgkmcnt(0)
	s_mov_b32 s38, 1
	s_cbranch_scc0 .LBB2_8
; %bb.7:
	v_lshl_or_b32 v2, v174, 8, v176
	s_lshl_b32 s0, s31, 10
	buffer_load_dwordx4 v[18:21], v2, s[4:7], s0 offen
	v_lshl_or_b32 v2, v174, 7, v175
	s_lshl_b32 s0, s31, 9
	buffer_load_dwordx2 v[192:193], v2, s[8:11], s0 offen
	s_movk_i32 s0, 0x780
	v_and_or_b32 v163, v157, s0, v155
	v_lshl_or_b32 v172, v191, 12, v163
	v_or_b32_e32 v2, 0x10000, v172
	s_movk_i32 s0, 0x70
	v_bitop3_b32 v6, v151, v2, s0 bitop3:0x6c
	;;#ASMSTART
	ds_read_b128 v[2:5], v6 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[10:13], v6 offset:0x800

	;;#ASMEND
	v_or_b32_e32 v6, 0x10040, v172
	v_bitop3_b32 v14, v151, v6, s0 bitop3:0x6c
	;;#ASMSTART
	ds_read_b128 v[6:9], v14 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[14:17], v14 offset:0x800

	;;#ASMEND
	v_lshl_or_b32 v173, v1, 13, v163
	v_bitop3_b32 v163, v151, v173, s0 bitop3:0x6c
	;;#ASMSTART
	ds_read_b128 v[164:167], v163 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[178:181], v163 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[194:197], v163 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[202:205], v163 offset:0x1800

	;;#ASMEND
	v_or_b32_e32 v163, 64, v173
	v_bitop3_b32 v163, v151, v163, s0 bitop3:0x6c
	;;#ASMSTART
	ds_read_b128 v[168:171], v163 offset:0

	;;#ASMEND
	s_lshl_b32 s1, s31, 8
	;;#ASMSTART
	ds_read_b128 v[182:185], v163 offset:0x800

	;;#ASMEND
	s_add_i32 s12, s30, s1
	;;#ASMSTART
	ds_read_b128 v[198:201], v163 offset:0x1000

	;;#ASMEND
	s_addk_i32 s12, 0x80
	;;#ASMSTART
	ds_read_b128 v[206:209], v163 offset:0x1800

	;;#ASMEND
	s_ashr_i32 s13, s12, 31
	v_or_b32_e32 v163, 0xc000, v153
	s_add_u32 s12, s40, s12
	v_readfirstlane_b32 s20, v163
	v_or_b32_e32 v163, 0xe000, v153
	s_addc_u32 s13, s41, s13
	s_mov_b32 s15, 0x110000
	s_mov_b32 m0, s20
	v_readfirstlane_b32 s20, v163
	buffer_load_dwordx4 v152, s[12:15], 0 offen lds
	s_mov_b32 m0, s20
	s_nop 0
	buffer_load_dwordx4 v150, s[12:15], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	s_waitcnt vmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[146:149], v[164:171], v[2:9], v[146:149], v18, v192 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[138:141], v[164:171], v[10:17], v[138:141], v18, v192 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[134:137], v[178:185], v[2:9], v[134:137], v18, v192 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[142:145], v[178:185], v[10:17], v[142:145], v18, v192 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[130:133], v[194:201], v[2:9], v[130:133], v20, v192 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[126:129], v[194:201], v[10:17], v[126:129], v20, v192 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[122:125], v[202:209], v[2:9], v[122:125], v20, v192 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[118:121], v[202:209], v[10:17], v[118:121], v20, v192 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	v_or_b32_e32 v163, 0x14000, v172
	v_bitop3_b32 v163, v151, v163, s0 bitop3:0x6c
	;;#ASMSTART
	ds_read_b128 v[210:213], v163 offset:0

	;;#ASMEND
	s_addk_i32 s1, 0x100
	;;#ASMSTART
	ds_read_b128 v[218:221], v163 offset:0x800

	;;#ASMEND
	v_or_b32_e32 v163, 0x14040, v172
	s_ashr_i32 s13, s1, 31
	v_bitop3_b32 v163, v151, v163, s0 bitop3:0x6c
	;;#ASMSTART
	ds_read_b128 v[214:217], v163 offset:0

	;;#ASMEND
	s_add_u32 s20, s28, s1
	v_readfirstlane_b32 s12, v161
	;;#ASMSTART
	ds_read_b128 v[222:225], v163 offset:0x800

	;;#ASMEND
	s_addc_u32 s21, s29, s13
	s_mov_b32 s23, s15
	s_mov_b32 m0, s12
	v_readfirstlane_b32 s12, v162
	buffer_load_dwordx4 v156, s[20:23], 0 offen lds
	s_mov_b32 m0, s12
	s_nop 0
	buffer_load_dwordx4 v154, s[20:23], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[164:171], v[210:217], v[90:93], v18, v193 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[164:171], v[218:225], v[86:89], v18, v193 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[178:185], v[210:217], v[94:97], v18, v193 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[98:101], v[178:185], v[218:225], v[98:101], v18, v193 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[102:105], v[194:201], v[210:217], v[102:105], v20, v193 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[106:109], v[194:201], v[218:225], v[106:109], v20, v193 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[110:113], v[202:209], v[210:217], v[110:113], v20, v193 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[114:117], v[202:209], v[218:225], v[114:117], v20, v193 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	v_or_b32_e32 v161, 0x4000, v173
	s_barrier
	v_bitop3_b32 v161, v151, v161, s0 bitop3:0x6c
	;;#ASMSTART
	ds_read_b128 v[162:165], v161 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[178:181], v161 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[194:197], v161 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[202:205], v161 offset:0x1800

	;;#ASMEND
	v_or_b32_e32 v161, 0x4040, v173
	v_bitop3_b32 v161, v151, v161, s0 bitop3:0x6c
	;;#ASMSTART
	ds_read_b128 v[166:169], v161 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[182:185], v161 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[198:201], v161 offset:0x1000

	;;#ASMEND
	s_add_u32 s12, s24, s1
	v_readfirstlane_b32 s0, v153
	;;#ASMSTART
	ds_read_b128 v[206:209], v161 offset:0x1800

	;;#ASMEND
	s_addc_u32 s13, s25, s13
	s_mov_b32 m0, s0
	v_readfirstlane_b32 s0, v160
	buffer_load_dwordx4 v152, s[12:15], 0 offen lds
	s_mov_b32 m0, s0
	s_nop 0
	buffer_load_dwordx4 v150, s[12:15], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[162:169], v[2:9], v[62:65], v19, v192 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[162:169], v[10:17], v[78:81], v19, v192 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[178:185], v[2:9], v[66:69], v19, v192 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[178:185], v[10:17], v[82:85], v19, v192 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[194:201], v[2:9], v[54:57], v21, v192 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[194:201], v[10:17], v[70:73], v21, v192 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[202:209], v[2:9], v[58:61], v21, v192 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[202:209], v[10:17], v[74:77], v21, v192 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_i32 s1, s1, s27
	s_ashr_i32 s0, s1, 31
	s_add_u32 s20, s34, s1
	s_addc_u32 s21, s35, s0
	v_readfirstlane_b32 s0, v158
	s_mov_b32 m0, s0
	v_readfirstlane_b32 s0, v159
	buffer_load_dwordx4 v156, s[20:23], 0 offen lds
	s_mov_b32 m0, s0
	s_nop 0
	buffer_load_dwordx4 v154, s[20:23], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(6)
	;;#ASMEND
	s_barrier
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[162:169], v[210:217], v[22:25], v19, v193 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[162:169], v[218:225], v[26:29], v19, v193 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[178:185], v[210:217], v[30:33], v19, v193 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[178:185], v[218:225], v[34:37], v19, v193 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[194:201], v[210:217], v[38:41], v21, v193 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[194:201], v[218:225], v[42:45], v21, v193 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], v[202:209], v[210:217], v[46:49], v21, v193 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[202:209], v[218:225], v[50:53], v21, v193 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_mov_b64 s[0:1], 0
	s_barrier
	s_ashr_i32 s19, s19, 1
	s_cmp_eq_u32 s19, -1
	s_cbranch_scc0 .LBB2_9
	s_branch .LBB2_10
.LBB2_8:
	s_mov_b32 s38, 0
	s_mov_b64 s[0:1], 1
	s_ashr_i32 s19, s19, 1
	s_cmp_eq_u32 s19, -1
	s_cbranch_scc1 .LBB2_10
.LBB2_9:
	v_lshl_or_b32 v2, v174, 8, v176
	s_lshl_b32 s1, s19, 10
	buffer_load_dwordx4 v[18:21], v2, s[4:7], s1 offen
	v_lshl_or_b32 v2, v174, 7, v175
	s_lshl_b32 s1, s19, 9
	buffer_load_dwordx2 v[192:193], v2, s[8:11], s1 offen
.LBB2_10:
	s_movk_i32 s1, 0x780
	s_lshl_b32 s12, s38, 15
	v_lshlrev_b32_e32 v194, 12, v191
	v_and_or_b32 v195, v157, s1, v155
	v_or3_b32 v170, v194, s12, v195
	v_or_b32_e32 v2, 0x10000, v170
	s_movk_i32 s1, 0x70
	v_bitop3_b32 v6, v151, v2, s1 bitop3:0x6c
	;;#ASMSTART
	ds_read_b128 v[2:5], v6 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[10:13], v6 offset:0x800

	;;#ASMEND
	v_or_b32_e32 v6, 0x10040, v170
	v_bitop3_b32 v14, v151, v6, s1 bitop3:0x6c
	;;#ASMSTART
	ds_read_b128 v[6:9], v14 offset:0

	;;#ASMEND
	v_lshlrev_b32_e32 v177, 13, v1
	;;#ASMSTART
	ds_read_b128 v[14:17], v14 offset:0x800

	;;#ASMEND
	v_or3_b32 v171, s12, v177, v195
	v_bitop3_b32 v158, v151, v171, s1 bitop3:0x6c
	;;#ASMSTART
	ds_read_b128 v[154:157], v158 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[162:165], v158 offset:0x800

	;;#ASMEND
	s_add_i32 s20, s26, -1
	;;#ASMSTART
	ds_read_b128 v[178:181], v158 offset:0x1000

	;;#ASMEND
	s_lshl_b32 s12, s20, 7
	;;#ASMSTART
	ds_read_b128 v[196:199], v158 offset:0x1800

	;;#ASMEND
	v_or_b32_e32 v158, 64, v171
	s_ashr_i32 s13, s12, 31
	v_bitop3_b32 v172, v151, v158, s1 bitop3:0x6c
	;;#ASMSTART
	ds_read_b128 v[158:161], v172 offset:0

	;;#ASMEND
	s_add_u32 s12, s16, s12
	;;#ASMSTART
	ds_read_b128 v[166:169], v172 offset:0x800

	;;#ASMEND
	s_addc_u32 s13, s17, s13
	s_lshl_b32 s0, s0, 15
	;;#ASMSTART
	ds_read_b128 v[182:185], v172 offset:0x1000

	;;#ASMEND
	v_or_b32_e32 v153, s0, v153
	;;#ASMSTART
	ds_read_b128 v[200:203], v172 offset:0x1800

	;;#ASMEND
	v_or_b32_e32 v172, 0x4000, v153
	s_mov_b32 s15, 0x110000
	v_readfirstlane_b32 s16, v172
	s_mov_b32 m0, s16
	s_nop 0
	buffer_load_dwordx4 v152, s[12:15], 0 offen lds
	v_or_b32_e32 v152, 0x6000, v153
	s_nop 0
	v_readfirstlane_b32 s16, v152
	s_mov_b32 m0, s16
	s_nop 0
	buffer_load_dwordx4 v150, s[12:15], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	s_lshl_b32 s12, s26, 4
	s_waitcnt vmcnt(3)
	v_lshrrev_b32_e32 v150, s12, v18
	s_waitcnt vmcnt(2)
	v_lshrrev_b32_e32 v172, s12, v192
	v_lshrrev_b32_e32 v152, s12, v20
	s_nop 0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[146:149], v[154:161], v[2:9], v[146:149], v150, v172 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[138:141], v[154:161], v[10:17], v[138:141], v150, v172 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[134:137], v[162:169], v[2:9], v[134:137], v150, v172 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[142:145], v[162:169], v[10:17], v[142:145], v150, v172 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[130:133], v[178:185], v[2:9], v[130:133], v152, v172 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[126:129], v[178:185], v[10:17], v[126:129], v152, v172 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[122:125], v[196:203], v[2:9], v[122:125], v152, v172 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[118:121], v[196:203], v[10:17], v[118:121], v152, v172 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	v_or_b32_e32 v153, 0x14000, v170
	v_bitop3_b32 v153, v151, v153, s1 bitop3:0x6c
	;;#ASMSTART
	ds_read_b128 v[204:207], v153 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[212:215], v153 offset:0x800

	;;#ASMEND
	v_or_b32_e32 v153, 0x14040, v170
	v_bitop3_b32 v153, v151, v153, s1 bitop3:0x6c
	;;#ASMSTART
	ds_read_b128 v[208:211], v153 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[216:219], v153 offset:0x800

	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_lshrrev_b32_e32 v170, s12, v193
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[154:161], v[204:211], v[90:93], v150, v170 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[154:161], v[212:219], v[86:89], v150, v170 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[162:169], v[204:211], v[94:97], v150, v170 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[98:101], v[162:169], v[212:219], v[98:101], v150, v170 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[102:105], v[178:185], v[204:211], v[102:105], v152, v170 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[106:109], v[178:185], v[212:219], v[106:109], v152, v170 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[110:113], v[196:203], v[204:211], v[110:113], v152, v170 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[114:117], v[196:203], v[212:219], v[114:117], v152, v170 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	v_or_b32_e32 v150, 0x4000, v171
	s_barrier
	v_bitop3_b32 v150, v151, v150, s1 bitop3:0x6c
	;;#ASMSTART
	ds_read_b128 v[152:155], v150 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[160:163], v150 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[178:181], v150 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[196:199], v150 offset:0x1800

	;;#ASMEND
	v_or_b32_e32 v150, 0x4040, v171
	v_bitop3_b32 v150, v151, v150, s1 bitop3:0x6c
	;;#ASMSTART
	ds_read_b128 v[156:159], v150 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[164:167], v150 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[182:185], v150 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[200:203], v150 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_lshrrev_b32_e32 v168, s12, v19
	v_lshrrev_b32_e32 v171, s12, v21
	s_nop 0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[152:159], v[2:9], v[62:65], v168, v172 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[152:159], v[10:17], v[78:81], v168, v172 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[160:167], v[2:9], v[66:69], v168, v172 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[160:167], v[10:17], v[82:85], v168, v172 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[178:185], v[2:9], v[54:57], v171, v172 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[178:185], v[10:17], v[70:73], v171, v172 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[196:203], v[2:9], v[58:61], v171, v172 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[196:203], v[10:17], v[74:77], v171, v172 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	v_or3_b32 v6, v194, s0, v195
	v_or_b32_e32 v2, 0x10000, v6
	s_barrier
	v_bitop3_b32 v2, v151, v2, s1 bitop3:0x6c
	;;#ASMSTART
	ds_read_b128 v[10:13], v2 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[2:5], v2 offset:0x800

	;;#ASMEND
	v_or_b32_e32 v6, 0x10040, v6
	v_bitop3_b32 v6, v151, v6, s1 bitop3:0x6c
	;;#ASMSTART
	ds_read_b128 v[14:17], v6 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[6:9], v6 offset:0x800

	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[152:159], v[204:211], v[22:25], v168, v170 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[152:159], v[212:219], v[26:29], v168, v170 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[150:153], v[160:167], v[204:211], v[30:33], v168, v170 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[154:157], v[160:167], v[212:219], v[34:37], v168, v170 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[158:161], v[178:185], v[204:211], v[38:41], v171, v170 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[162:165], v[178:185], v[212:219], v[42:45], v171, v170 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[166:169], v[196:203], v[204:211], v[46:49], v171, v170 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[170:173], v[196:203], v[212:219], v[50:53], v171, v170 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_ashr_i32 s0, s20, 1
	s_cmp_eq_u32 s0, s19
	s_cbranch_scc1 .LBB2_12
; %bb.11:
	v_lshl_or_b32 v18, v174, 8, v176
	s_lshl_b32 s12, s0, 10
	v_lshl_or_b32 v30, v174, 7, v175
	s_lshl_b32 s0, s0, 9
	buffer_load_dwordx4 v[18:21], v18, s[4:7], s12 offen
	s_nop 0
	buffer_load_dwordx2 v[192:193], v30, s[8:11], s0 offen
.LBB2_12:
	s_xor_b32 s0, s38, 1
	s_lshl_b32 s0, s0, 15
	v_or_b32_e32 v30, s0, v177
	v_add_u32_e32 v224, v195, v30
	v_lshrrev_b32_e32 v30, 4, v224
	v_bitop3_b32 v34, v30, v224, s1 bitop3:0x6c
	;;#ASMSTART
	ds_read_b128 v[30:33], v34 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[38:41], v34 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[46:49], v34 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[196:199], v34 offset:0x1800

	;;#ASMEND
	v_add_u32_e32 v34, 64, v224
	v_lshrrev_b32_e32 v35, 4, v34
	v_bitop3_b32 v174, v35, v34, s1 bitop3:0x6c
	;;#ASMSTART
	ds_read_b128 v[34:37], v174 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[42:45], v174 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[50:53], v174 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[200:203], v174 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	s_lshl_b32 s4, s20, 4
	s_waitcnt vmcnt(1)
	v_lshrrev_b32_e32 v18, s4, v18
	s_waitcnt vmcnt(0)
	v_lshrrev_b32_e32 v228, s4, v192
	v_lshrrev_b32_e32 v20, s4, v20
	s_nop 0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[146:149], v[30:37], v[10:17], v[146:149], v18, v228 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[174:177], v[30:37], v[2:9], v[138:141], v18, v228 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[186:189], v[38:45], v[10:17], v[134:137], v18, v228 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[142:145], v[38:45], v[2:9], v[142:145], v18, v228 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[182:185], v[46:53], v[10:17], v[130:133], v20, v228 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[138:141], v[46:53], v[2:9], v[126:129], v20, v228 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[178:181], v[196:203], v[10:17], v[122:125], v20, v228 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[134:137], v[196:203], v[2:9], v[118:121], v20, v228 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_nop 5
	v_or_b32_e32 v118, s0, v194
	v_add_u32_e32 v118, v195, v118
	v_add_u32_e32 v119, 0x14000, v118
	v_lshrrev_b32_e32 v120, 4, v119
	s_barrier
	v_bitop3_b32 v119, v120, v119, s1 bitop3:0x6c
	;;#ASMSTART
	ds_read_b128 v[204:207], v119 offset:0

	;;#ASMEND
	v_add_u32_e32 v118, 0x14040, v118
	;;#ASMSTART
	ds_read_b128 v[212:215], v119 offset:0x800

	;;#ASMEND
	v_lshrrev_b32_e32 v119, 4, v118
	v_bitop3_b32 v118, v119, v118, s1 bitop3:0x6c
	;;#ASMSTART
	ds_read_b128 v[208:211], v118 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[216:219], v118 offset:0x800

	;;#ASMEND
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_lshrrev_b32_e32 v229, s4, v193
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[126:129], v[30:37], v[204:211], v[90:93], v18, v229 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[130:133], v[30:37], v[212:219], v[86:89], v18, v229 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[122:125], v[38:45], v[204:211], v[94:97], v18, v229 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[118:121], v[38:45], v[212:219], v[98:101], v18, v229 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[98:101], v[46:53], v[204:211], v[102:105], v20, v229 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[46:53], v[212:219], v[106:109], v20, v229 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[196:203], v[204:211], v[110:113], v20, v229 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[196:203], v[212:219], v[114:117], v20, v229 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	v_add_u32_e32 v18, 0x4000, v224
	v_lshrrev_b32_e32 v20, 4, v18
	s_barrier
	v_bitop3_b32 v18, v20, v18, s1 bitop3:0x6c
	;;#ASMSTART
	ds_read_b128 v[102:105], v18 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[110:113], v18 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[192:195], v18 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[220:223], v18 offset:0x1800

	;;#ASMEND
	v_add_u32_e32 v18, 0x4040, v224
	v_lshrrev_b32_e32 v20, 4, v18
	v_bitop3_b32 v18, v20, v18, s1 bitop3:0x6c
	;;#ASMSTART
	ds_read_b128 v[106:109], v18 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[114:117], v18 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[196:199], v18 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[224:227], v18 offset:0x1800

	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_lshrrev_b32_e32 v18, s4, v19
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[102:109], v[2:9], v[78:81], v18, v228 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	s_nop 6
	v_lshrrev_b32_e32 v78, s4, v21
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[102:109], v[10:17], v[62:65], v18, v228 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[110:117], v[10:17], v[66:69], v18, v228 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], v[110:117], v[2:9], v[82:85], v18, v228 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[192:199], v[10:17], v[54:57], v78, v228 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[192:199], v[2:9], v[70:73], v78, v228 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[220:227], v[10:17], v[58:61], v78, v228 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[220:227], v[2:9], v[74:77], v78, v228 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[102:109], v[204:211], v[22:25], v18, v229 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[102:109], v[212:219], v[26:29], v18, v229 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[110:117], v[204:211], v[150:153], v18, v229 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[110:117], v[212:219], v[154:157], v18, v229 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[192:199], v[204:211], v[158:161], v78, v229 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[192:199], v[212:219], v[162:165], v78, v229 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[6:9], v[220:227], v[204:211], v[166:169], v78, v229 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[2:5], v[220:227], v[212:219], v[170:173], v78, v229 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_movk_i32 s0, 0x100
	v_cmp_gt_u32_e32 vcc, s0, v0
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB2_14
; %bb.13:
	s_barrier
.LBB2_14:
	s_or_b64 exec, exec, s[0:1]
	v_lshl_or_b32 v168, s33, 2, v1
	v_lshl_or_b32 v169, v191, 5, s42
	v_mul_lo_u32 v1, v168, s18
	v_lshl_add_u32 v58, v1, 6, v169
	v_lshrrev_b32_e32 v0, 2, v0
	v_ashrrev_i32_e32 v59, 31, v58
	v_and_b32_e32 v0, 12, v0
	v_pk_mul_f32 v[116:117], v[134:135], s[36:37] op_sel_hi:[1,0]
	v_lshl_add_u64 v[134:135], v[58:59], 1, s[2:3]
	v_mad_u64_u32 v[58:59], s[0:1], v0, s18, v[190:191]
	v_add_u32_e32 v70, s18, v58
	v_ashrrev_i32_e32 v59, 31, v58
	v_ashrrev_i32_e32 v71, 31, v70
	v_lshlrev_b64 v[0:1], 1, v[58:59]
	v_lshlrev_b64 v[58:59], 1, v[70:71]
	v_add_u32_e32 v70, s18, v70
	v_pk_mul_f32 v[84:85], v[142:143], s[36:37] op_sel_hi:[1,0]
	v_add_u32_e32 v142, s18, v70
	v_pk_mul_f32 v[60:61], v[146:147], s[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[108:109], v[138:139], s[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[114:115], v[136:137], s[36:37] op_sel_hi:[1,0]
	v_lshl_add_u64 v[136:137], v[134:135], 0, v[0:1]
	v_lshl_add_u64 v[138:139], v[134:135], 0, v[58:59]
	v_ashrrev_i32_e32 v71, 31, v70
	v_ashrrev_i32_e32 v143, 31, v142
	global_store_short_d16_hi v[136:137], v60, off
	global_store_short_d16_hi v[138:139], v61, off
	v_lshlrev_b64 v[60:61], 1, v[70:71]
	v_lshlrev_b64 v[70:71], 1, v[142:143]
	v_pk_mul_f32 v[72:73], v[148:149], s[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[74:75], v[176:177], s[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[82:83], v[144:145], s[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[106:107], v[140:141], s[36:37] op_sel_hi:[1,0]
	v_lshl_add_u64 v[140:141], v[134:135], 0, v[60:61]
	v_lshl_add_u64 v[144:145], v[134:135], 0, v[70:71]
	s_mul_i32 s0, s18, 13
	v_pk_mul_f32 v[76:77], v[174:175], s[36:37] op_sel_hi:[1,0]
	global_store_short_d16_hi v[140:141], v72, off
	global_store_short_d16_hi v[144:145], v73, off
	global_store_short_d16_hi v[136:137], v76, off offset:32
	global_store_short_d16_hi v[138:139], v77, off offset:32
	global_store_short_d16_hi v[140:141], v74, off offset:32
	global_store_short_d16_hi v[144:145], v75, off offset:32
	v_add_u32_e32 v74, s0, v142
	v_ashrrev_i32_e32 v75, 31, v74
	v_lshlrev_b64 v[72:73], 1, v[74:75]
	v_add_u32_e32 v76, s18, v74
	v_pk_mul_f32 v[78:79], v[186:187], s[36:37] op_sel_hi:[1,0]
	v_lshl_add_u64 v[142:143], v[134:135], 0, v[72:73]
	v_ashrrev_i32_e32 v77, 31, v76
	global_store_short_d16_hi v[142:143], v78, off
	v_lshlrev_b64 v[74:75], 1, v[76:77]
	v_add_u32_e32 v78, s18, v76
	v_lshl_add_u64 v[146:147], v[134:135], 0, v[74:75]
	v_add_u32_e32 v150, s18, v78
	global_store_short_d16_hi v[146:147], v79, off
	v_ashrrev_i32_e32 v79, 31, v78
	v_ashrrev_i32_e32 v151, 31, v150
	v_lshlrev_b64 v[76:77], 1, v[78:79]
	v_lshlrev_b64 v[78:79], 1, v[150:151]
	v_pk_mul_f32 v[80:81], v[188:189], s[36:37] op_sel_hi:[1,0]
	v_lshl_add_u64 v[148:149], v[134:135], 0, v[76:77]
	v_lshl_add_u64 v[152:153], v[134:135], 0, v[78:79]
	global_store_short_d16_hi v[148:149], v80, off
	global_store_short_d16_hi v[152:153], v81, off
	global_store_short_d16_hi v[142:143], v84, off offset:32
	global_store_short_d16_hi v[146:147], v85, off offset:32
	global_store_short_d16_hi v[148:149], v82, off offset:32
	global_store_short_d16_hi v[152:153], v83, off offset:32
	v_add_u32_e32 v82, s0, v150
	v_ashrrev_i32_e32 v83, 31, v82
	v_lshlrev_b64 v[80:81], 1, v[82:83]
	v_add_u32_e32 v84, s18, v82
	v_pk_mul_f32 v[102:103], v[182:183], s[36:37] op_sel_hi:[1,0]
	v_lshl_add_u64 v[150:151], v[134:135], 0, v[80:81]
	v_ashrrev_i32_e32 v85, 31, v84
	global_store_short_d16_hi v[150:151], v102, off
	v_lshlrev_b64 v[82:83], 1, v[84:85]
	v_add_u32_e32 v102, s18, v84
	v_lshl_add_u64 v[154:155], v[134:135], 0, v[82:83]
	v_add_u32_e32 v158, s18, v102
	global_store_short_d16_hi v[154:155], v103, off
	v_ashrrev_i32_e32 v103, 31, v102
	v_ashrrev_i32_e32 v159, 31, v158
	v_lshlrev_b64 v[84:85], 1, v[102:103]
	v_lshlrev_b64 v[102:103], 1, v[158:159]
	v_pk_mul_f32 v[104:105], v[184:185], s[36:37] op_sel_hi:[1,0]
	v_lshl_add_u64 v[156:157], v[134:135], 0, v[84:85]
	v_lshl_add_u64 v[160:161], v[134:135], 0, v[102:103]
	global_store_short_d16_hi v[156:157], v104, off
	global_store_short_d16_hi v[160:161], v105, off
	global_store_short_d16_hi v[150:151], v108, off offset:32
	global_store_short_d16_hi v[154:155], v109, off offset:32
	global_store_short_d16_hi v[156:157], v106, off offset:32
	global_store_short_d16_hi v[160:161], v107, off offset:32
	v_add_u32_e32 v106, s0, v158
	v_ashrrev_i32_e32 v107, 31, v106
	v_add_u32_e32 v108, s18, v106
	v_lshlrev_b64 v[104:105], 1, v[106:107]
	v_ashrrev_i32_e32 v109, 31, v108
	v_pk_mul_f32 v[110:111], v[178:179], s[36:37] op_sel_hi:[1,0]
	v_lshl_add_u64 v[158:159], v[134:135], 0, v[104:105]
	v_lshlrev_b64 v[106:107], 1, v[108:109]
	global_store_short_d16_hi v[158:159], v110, off
	v_lshl_add_u64 v[162:163], v[134:135], 0, v[106:107]
	v_add_u32_e32 v110, s18, v108
	global_store_short_d16_hi v[162:163], v111, off
	v_ashrrev_i32_e32 v111, 31, v110
	v_lshlrev_b64 v[108:109], 1, v[110:111]
	v_add_u32_e32 v110, s18, v110
	v_ashrrev_i32_e32 v111, 31, v110
	v_lshlrev_b64 v[110:111], 1, v[110:111]
	v_pk_mul_f32 v[112:113], v[180:181], s[36:37] op_sel_hi:[1,0]
	v_lshl_add_u64 v[164:165], v[134:135], 0, v[108:109]
	v_lshl_add_u64 v[166:167], v[134:135], 0, v[110:111]
	s_mov_b64 s[0:1], 0x100
	global_store_short_d16_hi v[164:165], v112, off
	global_store_short_d16_hi v[166:167], v113, off
	global_store_short_d16_hi v[158:159], v116, off offset:32
	global_store_short_d16_hi v[162:163], v117, off offset:32
	global_store_short_d16_hi v[164:165], v114, off offset:32
	global_store_short_d16_hi v[166:167], v115, off offset:32
	v_pk_mul_f32 v[112:113], v[128:129], s[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[114:115], v[126:127], s[36:37] op_sel_hi:[1,0]
	v_lshl_add_u64 v[128:129], v[134:135], 0, s[0:1]
	v_pk_mul_f32 v[126:127], v[130:131], s[36:37] op_sel_hi:[1,0]
	global_store_short_d16_hi v[136:137], v114, off offset:256
	global_store_short_d16_hi v[138:139], v115, off offset:256
	global_store_short_d16_hi v[140:141], v112, off offset:256
	global_store_short_d16_hi v[144:145], v113, off offset:256
	v_lshl_add_u64 v[112:113], v[128:129], 0, v[0:1]
	global_store_short_d16_hi v[112:113], v126, off offset:32
	v_lshl_add_u64 v[112:113], v[128:129], 0, v[58:59]
	v_pk_mul_f32 v[116:117], v[132:133], s[36:37] op_sel_hi:[1,0]
	global_store_short_d16_hi v[112:113], v127, off offset:32
	v_lshl_add_u64 v[112:113], v[128:129], 0, v[60:61]
	global_store_short_d16_hi v[112:113], v116, off offset:32
	v_lshl_add_u64 v[112:113], v[128:129], 0, v[70:71]
	v_pk_mul_f32 v[124:125], v[124:125], s[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[122:123], v[122:123], s[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[118:119], v[118:119], s[36:37] op_sel_hi:[1,0]
	global_store_short_d16_hi v[112:113], v117, off offset:32
	global_store_short_d16_hi v[142:143], v122, off offset:256
	global_store_short_d16_hi v[146:147], v123, off offset:256
	global_store_short_d16_hi v[148:149], v124, off offset:256
	global_store_short_d16_hi v[152:153], v125, off offset:256
	v_lshl_add_u64 v[112:113], v[128:129], 0, v[72:73]
	global_store_short_d16_hi v[112:113], v118, off offset:32
	v_lshl_add_u64 v[112:113], v[128:129], 0, v[74:75]
	v_pk_mul_f32 v[120:121], v[120:121], s[36:37] op_sel_hi:[1,0]
	global_store_short_d16_hi v[112:113], v119, off offset:32
	v_lshl_add_u64 v[112:113], v[128:129], 0, v[76:77]
	v_pk_mul_f32 v[98:99], v[98:99], s[36:37] op_sel_hi:[1,0]
	global_store_short_d16_hi v[112:113], v120, off offset:32
	v_lshl_add_u64 v[112:113], v[128:129], 0, v[78:79]
	v_pk_mul_f32 v[100:101], v[100:101], s[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[94:95], v[94:95], s[36:37] op_sel_hi:[1,0]
	global_store_short_d16_hi v[112:113], v121, off offset:32
	global_store_short_d16_hi v[150:151], v98, off offset:256
	global_store_short_d16_hi v[154:155], v99, off offset:256
	global_store_short_d16_hi v[156:157], v100, off offset:256
	global_store_short_d16_hi v[160:161], v101, off offset:256
	v_lshl_add_u64 v[98:99], v[128:129], 0, v[80:81]
	global_store_short_d16_hi v[98:99], v94, off offset:32
	v_lshl_add_u64 v[98:99], v[128:129], 0, v[82:83]
	v_pk_mul_f32 v[96:97], v[96:97], s[36:37] op_sel_hi:[1,0]
	global_store_short_d16_hi v[98:99], v95, off offset:32
	v_lshl_add_u64 v[94:95], v[128:129], 0, v[84:85]
	v_pk_mul_f32 v[86:87], v[86:87], s[36:37] op_sel_hi:[1,0]
	global_store_short_d16_hi v[94:95], v96, off offset:32
	v_lshl_add_u64 v[94:95], v[128:129], 0, v[102:103]
	v_pk_mul_f32 v[88:89], v[88:89], s[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[38:39], s[36:37] op_sel_hi:[1,0]
	global_store_short_d16_hi v[94:95], v97, off offset:32
	global_store_short_d16_hi v[158:159], v86, off offset:256
	global_store_short_d16_hi v[162:163], v87, off offset:256
	global_store_short_d16_hi v[164:165], v88, off offset:256
	global_store_short_d16_hi v[166:167], v89, off offset:256
	v_lshl_add_u64 v[86:87], v[128:129], 0, v[104:105]
	global_store_short_d16_hi v[86:87], v38, off offset:32
	v_lshl_add_u64 v[86:87], v[128:129], 0, v[106:107]
	global_store_short_d16_hi v[86:87], v39, off offset:32
	v_or_b32_e32 v86, 2, v168
	v_mul_lo_u32 v86, v86, s18
	v_lshl_add_u32 v86, v86, 6, v169
	v_pk_mul_f32 v[40:41], v[40:41], s[36:37] op_sel_hi:[1,0]
	v_lshl_add_u64 v[38:39], v[128:129], 0, v[108:109]
	v_ashrrev_i32_e32 v87, 31, v86
	global_store_short_d16_hi v[38:39], v40, off offset:32
	v_lshl_add_u64 v[38:39], v[128:129], 0, v[110:111]
	v_lshl_add_u64 v[86:87], v[86:87], 1, s[2:3]
	global_store_short_d16_hi v[38:39], v41, off offset:32
	v_pk_mul_f32 v[40:41], v[90:91], s[36:37] op_sel_hi:[1,0]
	v_lshl_add_u64 v[88:89], v[86:87], 0, v[0:1]
	v_lshl_add_u64 v[90:91], v[86:87], 0, v[58:59]
	v_pk_mul_f32 v[38:39], v[92:93], s[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[50:51], v[50:51], s[36:37] op_sel_hi:[1,0]
	global_store_short_d16_hi v[88:89], v40, off
	global_store_short_d16_hi v[90:91], v41, off
	v_lshl_add_u64 v[40:41], v[86:87], 0, v[60:61]
	v_lshl_add_u64 v[92:93], v[86:87], 0, v[70:71]
	v_pk_mul_f32 v[52:53], v[52:53], s[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[66:67], v[66:67], s[36:37] op_sel_hi:[1,0]
	global_store_short_d16_hi v[40:41], v38, off
	global_store_short_d16_hi v[92:93], v39, off
	global_store_short_d16_hi v[88:89], v50, off offset:32
	global_store_short_d16_hi v[90:91], v51, off offset:32
	global_store_short_d16_hi v[40:41], v52, off offset:32
	global_store_short_d16_hi v[92:93], v53, off offset:32
	v_lshl_add_u64 v[38:39], v[86:87], 0, v[72:73]
	v_lshl_add_u64 v[50:51], v[86:87], 0, v[74:75]
	v_pk_mul_f32 v[68:69], v[68:69], s[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[48:49], s[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[46:47], s[36:37] op_sel_hi:[1,0]
	global_store_short_d16_hi v[38:39], v66, off
	global_store_short_d16_hi v[50:51], v67, off
	v_lshl_add_u64 v[52:53], v[86:87], 0, v[76:77]
	v_lshl_add_u64 v[66:67], v[86:87], 0, v[78:79]
	v_pk_mul_f32 v[62:63], v[62:63], s[36:37] op_sel_hi:[1,0]
	global_store_short_d16_hi v[52:53], v68, off
	global_store_short_d16_hi v[66:67], v69, off
	global_store_short_d16_hi v[38:39], v46, off offset:32
	global_store_short_d16_hi v[50:51], v47, off offset:32
	global_store_short_d16_hi v[52:53], v48, off offset:32
	global_store_short_d16_hi v[66:67], v49, off offset:32
	v_lshl_add_u64 v[46:47], v[86:87], 0, v[80:81]
	v_lshl_add_u64 v[48:49], v[86:87], 0, v[82:83]
	v_pk_mul_f32 v[64:65], v[64:65], s[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[44:45], s[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[42:43], s[36:37] op_sel_hi:[1,0]
	global_store_short_d16_hi v[46:47], v62, off
	global_store_short_d16_hi v[48:49], v63, off
	v_lshl_add_u64 v[62:63], v[86:87], 0, v[84:85]
	v_lshl_add_u64 v[68:69], v[86:87], 0, v[102:103]
	v_pk_mul_f32 v[54:55], v[54:55], s[36:37] op_sel_hi:[1,0]
	global_store_short_d16_hi v[62:63], v64, off
	global_store_short_d16_hi v[68:69], v65, off
	global_store_short_d16_hi v[46:47], v42, off offset:32
	global_store_short_d16_hi v[48:49], v43, off offset:32
	global_store_short_d16_hi v[62:63], v44, off offset:32
	global_store_short_d16_hi v[68:69], v45, off offset:32
	v_lshl_add_u64 v[42:43], v[86:87], 0, v[104:105]
	v_lshl_add_u64 v[44:45], v[86:87], 0, v[106:107]
	v_pk_mul_f32 v[56:57], v[56:57], s[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[34:35], v[34:35], s[36:37] op_sel_hi:[1,0]
	global_store_short_d16_hi v[42:43], v54, off
	global_store_short_d16_hi v[44:45], v55, off
	v_lshl_add_u64 v[54:55], v[86:87], 0, v[108:109]
	v_lshl_add_u64 v[64:65], v[86:87], 0, v[110:111]
	v_pk_mul_f32 v[36:37], v[36:37], s[36:37] op_sel_hi:[1,0]
	global_store_short_d16_hi v[54:55], v56, off
	global_store_short_d16_hi v[64:65], v57, off
	global_store_short_d16_hi v[42:43], v34, off offset:32
	global_store_short_d16_hi v[44:45], v35, off offset:32
	global_store_short_d16_hi v[54:55], v36, off offset:32
	global_store_short_d16_hi v[64:65], v37, off offset:32
	v_lshl_add_u64 v[34:35], v[86:87], 0, s[0:1]
	v_pk_mul_f32 v[30:31], v[30:31], s[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], s[36:37] op_sel_hi:[1,0]
	v_lshl_add_u64 v[0:1], v[34:35], 0, v[0:1]
	v_pk_mul_f32 v[32:33], v[32:33], s[36:37] op_sel_hi:[1,0]
	global_store_short_d16_hi v[88:89], v30, off offset:256
	global_store_short_d16_hi v[90:91], v31, off offset:256
	global_store_short_d16_hi v[40:41], v32, off offset:256
	global_store_short_d16_hi v[92:93], v33, off offset:256
	global_store_short_d16_hi v[0:1], v26, off offset:32
	v_lshl_add_u64 v[0:1], v[34:35], 0, v[58:59]
	v_pk_mul_f32 v[28:29], v[28:29], s[36:37] op_sel_hi:[1,0]
	global_store_short_d16_hi v[0:1], v27, off offset:32
	v_lshl_add_u64 v[0:1], v[34:35], 0, v[60:61]
	global_store_short_d16_hi v[0:1], v28, off offset:32
	v_lshl_add_u64 v[0:1], v[34:35], 0, v[70:71]
	v_pk_mul_f32 v[24:25], v[24:25], s[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], s[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], s[36:37] op_sel_hi:[1,0]
	global_store_short_d16_hi v[0:1], v29, off offset:32
	global_store_short_d16_hi v[38:39], v22, off offset:256
	global_store_short_d16_hi v[50:51], v23, off offset:256
	global_store_short_d16_hi v[52:53], v24, off offset:256
	global_store_short_d16_hi v[66:67], v25, off offset:256
	v_lshl_add_u64 v[0:1], v[34:35], 0, v[72:73]
	global_store_short_d16_hi v[0:1], v18, off offset:32
	v_lshl_add_u64 v[0:1], v[34:35], 0, v[74:75]
	v_pk_mul_f32 v[20:21], v[20:21], s[36:37] op_sel_hi:[1,0]
	global_store_short_d16_hi v[0:1], v19, off offset:32
	v_lshl_add_u64 v[0:1], v[34:35], 0, v[76:77]
	global_store_short_d16_hi v[0:1], v20, off offset:32
	v_lshl_add_u64 v[0:1], v[34:35], 0, v[78:79]
	v_pk_mul_f32 v[16:17], v[16:17], s[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[14:15], v[14:15], s[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], s[36:37] op_sel_hi:[1,0]
	global_store_short_d16_hi v[0:1], v21, off offset:32
	global_store_short_d16_hi v[46:47], v14, off offset:256
	global_store_short_d16_hi v[48:49], v15, off offset:256
	global_store_short_d16_hi v[62:63], v16, off offset:256
	global_store_short_d16_hi v[68:69], v17, off offset:256
	v_lshl_add_u64 v[0:1], v[34:35], 0, v[80:81]
	global_store_short_d16_hi v[0:1], v10, off offset:32
	v_lshl_add_u64 v[0:1], v[34:35], 0, v[82:83]
	v_pk_mul_f32 v[12:13], v[12:13], s[36:37] op_sel_hi:[1,0]
	global_store_short_d16_hi v[0:1], v11, off offset:32
	v_lshl_add_u64 v[0:1], v[34:35], 0, v[84:85]
	global_store_short_d16_hi v[0:1], v12, off offset:32
	v_lshl_add_u64 v[0:1], v[34:35], 0, v[102:103]
	v_pk_mul_f32 v[8:9], v[8:9], s[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], s[36:37] op_sel_hi:[1,0]
	v_pk_mul_f32 v[2:3], v[2:3], s[36:37] op_sel_hi:[1,0]
	global_store_short_d16_hi v[0:1], v13, off offset:32
	global_store_short_d16_hi v[42:43], v6, off offset:256
	global_store_short_d16_hi v[44:45], v7, off offset:256
	global_store_short_d16_hi v[54:55], v8, off offset:256
	global_store_short_d16_hi v[64:65], v9, off offset:256
	v_lshl_add_u64 v[0:1], v[34:35], 0, v[104:105]
	global_store_short_d16_hi v[0:1], v2, off offset:32
	v_lshl_add_u64 v[0:1], v[34:35], 0, v[106:107]
	v_pk_mul_f32 v[4:5], v[4:5], s[36:37] op_sel_hi:[1,0]
	global_store_short_d16_hi v[0:1], v3, off offset:32
	v_lshl_add_u64 v[0:1], v[34:35], 0, v[108:109]
	global_store_short_d16_hi v[0:1], v4, off offset:32
	v_lshl_add_u64 v[0:1], v[34:35], 0, v[110:111]
	global_store_short_d16_hi v[0:1], v5, off offset:32
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z29rcr_exact_8wave_scaled_kernelILb1ELi2EEv14layout_globals
		.amdhsa_group_segment_fixed_size 131072
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 296
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
		.amdhsa_next_free_vgpr 246
		.amdhsa_next_free_sgpr 96
		.amdhsa_accum_offset 248
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
