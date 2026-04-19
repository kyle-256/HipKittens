_Z29crr_exact_8wave_scaled_kernelILb1ELi2EEv14layout_globals: ; @_Z29crr_exact_8wave_scaled_kernelILb1ELi2EEv14layout_globals
; %bb.0:
	s_load_dwordx2 s[10:11], s[0:1], 0x60
	s_load_dwordx2 s[8:9], s[0:1], 0x90
	s_load_dword s3, s[0:1], 0x128
	s_load_dwordx2 s[22:23], s[0:1], 0x0
	s_load_dwordx2 s[24:25], s[0:1], 0x20
	s_load_dwordx2 s[28:29], s[0:1], 0x30
	s_load_dwordx2 s[30:31], s[0:1], 0x50
	s_waitcnt lgkmcnt(0)
	s_cmp_lt_i32 s3, 8
	s_cselect_b64 s[4:5], -1, 0
	s_and_b32 s6, s3, 7
	s_cmp_lg_u32 s6, 0
	s_cselect_b64 s[6:7], -1, 0
	s_or_b64 s[4:5], s[4:5], s[6:7]
	s_and_b64 vcc, exec, s[4:5]
	s_cbranch_vccnz .LBB4_2
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
.LBB4_2:
	s_ashr_i32 s3, s2, 31
	s_lshr_b32 s3, s3, 26
	s_add_i32 s3, s2, s3
	s_ashr_i32 s6, s3, 6
	s_load_dword s20, s[0:1], 0x108
	s_lshl_b32 s3, s6, 2
	s_sub_i32 s4, 16, s3
	s_cmpk_gt_i32 s2, 0xff
	s_cselect_b32 s5, s4, 4
	s_mov_b32 s25, 16
	s_cmp_lt_i32 s5, 1
	s_mov_b32 s4, 16
	s_cbranch_scc1 .LBB4_4
; %bb.3:
	s_abs_i32 s4, s5
	v_cvt_f32_u32_e32 v1, s4
	s_lshl_b32 s6, s6, 6
	s_sub_i32 s2, s2, s6
	s_sub_i32 s6, 0, s4
	v_rcp_iflag_f32_e32 v1, v1
	s_abs_i32 s12, s2
	s_xor_b32 s7, s2, s5
	s_ashr_i32 s7, s7, 31
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	s_nop 0
	v_readfirstlane_b32 s13, v1
	s_mul_i32 s6, s6, s13
	s_mul_hi_u32 s6, s13, s6
	s_add_i32 s13, s13, s6
	s_mul_hi_u32 s6, s12, s13
	s_mul_i32 s13, s6, s4
	s_sub_i32 s12, s12, s13
	s_add_i32 s14, s6, 1
	s_sub_i32 s13, s12, s4
	s_cmp_ge_u32 s12, s4
	s_cselect_b32 s6, s14, s6
	s_cselect_b32 s12, s13, s12
	s_add_i32 s13, s6, 1
	s_cmp_ge_u32 s12, s4
	s_cselect_b32 s4, s13, s6
	s_xor_b32 s4, s4, s7
	s_sub_i32 s4, s4, s7
	s_mul_i32 s5, s4, s5
	s_sub_i32 s2, s2, s5
	s_add_i32 s25, s2, s3
.LBB4_4:
	s_waitcnt lgkmcnt(0)
	s_ashr_i32 s26, s20, 31
	s_lshr_b32 s2, s26, 27
	s_add_i32 s2, s20, s2
	s_ashr_i32 s2, s2, 5
	s_add_i32 s2, s2, 7
	s_and_b32 s2, s2, -8
	s_ashr_i32 s5, s4, 31
	s_lshl_b32 s38, s2, 7
	s_lshl_b32 s37, s2, 6
	s_lshl_b64 s[2:3], s[4:5], 2
	v_lshlrev_b32_e32 v1, 4, v0
	v_lshlrev_b32_e32 v2, 1, v0
	s_movk_i32 s5, 0x70
	v_lshrrev_b32_e32 v3, 3, v0
	v_bitop3_b32 v2, v2, s5, v1 bitop3:0x48
	v_or_b32_e32 v4, 64, v3
	s_lshl_b32 s31, s4, 8
	v_and_b32_e32 v173, 0x1c00, v1
	s_ashr_i32 s12, s25, 31
	s_lshl_b32 s21, s25, 1
	v_mad_u64_u32 v[160:161], s[6:7], v3, s24, v[2:3]
	v_mad_u64_u32 v[158:159], s[6:7], v4, s24, v[2:3]
	v_mad_u64_u32 v[162:163], s[6:7], v3, s30, v[2:3]
	v_mad_u64_u32 v[164:165], s[6:7], v4, s30, v[2:3]
	s_lshl_b32 s35, s4, 1
	s_ashr_i32 s4, s31, 31
	v_add_u32_e32 v2, 0x11000, v173
	v_and_b32_e32 v174, 0x180, v0
	s_add_u32 s16, s28, s31
	v_or_b32_e32 v1, v2, v174
	v_or_b32_e32 v175, 0x2200, v174
	s_addc_u32 s17, s29, s4
	v_readfirstlane_b32 s4, v1
	v_add_u32_e32 v1, v2, v175
	s_lshl_b32 s27, s25, 8
	s_lshl_b32 s18, s30, 7
	s_mov_b32 s19, 0x110000
	s_mov_b32 m0, s4
	v_readfirstlane_b32 s4, v1
	s_ashr_i32 s5, s27, 31
	v_or_b32_e32 v179, v173, v174
	buffer_load_dwordx4 v162, s[16:19], 0 offen lds
	s_mov_b32 m0, s4
	s_add_u32 s4, s22, s27
	v_readfirstlane_b32 s13, v179
	v_or_b32_e32 v180, v173, v175
	buffer_load_dwordx4 v164, s[16:19], 0 offen lds
	s_addc_u32 s5, s23, s5
	s_lshl_b32 s6, s24, 7
	s_mov_b32 s7, s19
	s_mov_b32 m0, s13
	v_readfirstlane_b32 s13, v180
	buffer_load_dwordx4 v160, s[4:7], 0 offen lds
	s_mov_b32 m0, s13
	s_or_b32 s36, s35, 1
	buffer_load_dwordx4 v158, s[4:7], 0 offen lds
	s_lshl_b32 s4, s36, 7
	v_add_u32_e32 v1, 0x4400, v2
	s_ashr_i32 s5, s4, 31
	v_add_u32_e32 v3, v1, v174
	s_add_u32 s16, s28, s4
	v_readfirstlane_b32 s4, v3
	v_add_u32_e32 v1, v1, v175
	s_addc_u32 s17, s29, s5
	s_mov_b32 m0, s4
	v_readfirstlane_b32 s4, v1
	s_or_b32 s33, s21, 1
	buffer_load_dwordx4 v162, s[16:19], 0 offen lds
	s_mov_b32 m0, s4
	s_lshl_b32 s4, s33, 7
	v_add_u32_e32 v1, 0x4400, v173
	s_ashr_i32 s5, s4, 31
	v_or_b32_e32 v3, v1, v174
	s_add_u32 s4, s22, s4
	v_readfirstlane_b32 s13, v3
	v_add_u32_e32 v1, v1, v175
	buffer_load_dwordx4 v164, s[16:19], 0 offen lds
	s_addc_u32 s5, s23, s5
	s_mov_b32 m0, s13
	v_readfirstlane_b32 s13, v1
	buffer_load_dwordx4 v160, s[4:7], 0 offen lds
	s_mov_b32 m0, s13
	v_lshrrev_b32_e32 v1, 8, v0
	buffer_load_dwordx4 v158, s[4:7], 0 offen lds
	v_bfe_u32 v3, v0, 6, 2
	v_or_b32_e32 v6, s21, v1
	v_mov_b64_e32 v[4:5], s[10:11]
	v_or_b32_e32 v8, s2, v3
	v_mad_u64_u32 v[4:5], s[4:5], v6, s38, v[4:5]
	v_mov_b64_e32 v[6:7], s[8:9]
	s_mul_i32 s12, s12, s38
	s_mul_i32 s4, s3, s37
	v_mad_u64_u32 v[6:7], s[2:3], v8, s37, v[6:7]
	v_add_u32_e32 v5, s12, v5
	v_add_u32_e32 v7, s4, v7
	v_readfirstlane_b32 s5, v4
	v_readfirstlane_b32 s39, v5
	v_readfirstlane_b32 s4, v6
	v_readfirstlane_b32 s40, v7
	v_cmp_eq_u32_e32 vcc, 1, v1
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB4_6
; %bb.5:
	s_barrier
.LBB4_6:
	s_or_b64 exec, exec, s[2:3]
	s_mov_b32 s16, s5
	s_lshr_b32 s2, s26, 25
	s_mov_b64 s[8:9], s[16:17]
	s_add_i32 s2, s20, s2
	s_add_i32 s35, s35, s30
	s_mov_b64 s[10:11], s[18:19]
	s_mov_b32 s16, s4
	s_ashr_i32 s34, s2, 7
	s_lshl_b32 s2, s35, 7
	v_add_u32_e32 v4, 0x8800, v2
	s_mov_b64 s[12:13], s[16:17]
	s_ashr_i32 s3, s2, 31
	v_add_u32_e32 v5, v4, v174
	s_mov_b64 s[14:15], s[18:19]
	s_add_u32 s16, s28, s2
	v_readfirstlane_b32 s2, v5
	v_add_u32_e32 v4, v4, v175
	s_addc_u32 s17, s29, s3
	s_mov_b32 m0, s2
	v_readfirstlane_b32 s2, v4
	s_add_i32 s21, s21, s24
	;;#ASMSTART
	s_waitcnt vmcnt(2)
	;;#ASMEND
	s_barrier
	buffer_load_dwordx4 v162, s[16:19], 0 offen lds
	s_mov_b32 m0, s2
	s_lshl_b32 s2, s21, 7
	v_add_u32_e32 v4, 0x8800, v173
	s_ashr_i32 s3, s2, 31
	v_add_u32_e32 v5, v4, v174
	s_add_u32 s4, s22, s2
	v_readfirstlane_b32 s2, v5
	v_add_u32_e32 v4, v4, v175
	buffer_load_dwordx4 v164, s[16:19], 0 offen lds
	s_addc_u32 s5, s23, s3
	s_mov_b32 s7, s19
	s_mov_b32 m0, s2
	v_readfirstlane_b32 s2, v4
	s_add_i32 s36, s36, s30
	buffer_load_dwordx4 v160, s[4:7], 0 offen lds
	s_mov_b32 m0, s2
	s_lshl_b32 s2, s36, 7
	v_add_u32_e32 v2, 0xcc00, v2
	s_ashr_i32 s3, s2, 31
	v_add_u32_e32 v4, v2, v174
	s_add_u32 s16, s28, s2
	v_readfirstlane_b32 s2, v4
	v_add_u32_e32 v2, v2, v175
	buffer_load_dwordx4 v158, s[4:7], 0 offen lds
	s_addc_u32 s17, s29, s3
	s_mov_b32 m0, s2
	v_readfirstlane_b32 s2, v2
	buffer_load_dwordx4 v162, s[16:19], 0 offen lds
	s_mov_b32 m0, s2
	v_and_b32_e32 v166, 15, v0
	buffer_load_dwordx4 v164, s[16:19], 0 offen lds
	v_bfe_u32 v4, v0, 4, 2
	v_bfe_u32 v2, v0, 1, 3
	v_lshlrev_b32_e32 v167, 5, v3
	s_mov_b32 s9, s39
	s_mov_b32 s13, s40
	s_cmpk_gt_i32 s20, 0x17f
	v_lshlrev_b32_e32 v8, 4, v166
	v_lshlrev_b32_e32 v9, 3, v166
	v_lshlrev_b32_e32 v10, 3, v0
	v_lshlrev_b32_e32 v7, 11, v4
	v_add_u32_e32 v11, v4, v2
	v_lshlrev_b32_e32 v181, 4, v2
	v_or_b32_e32 v5, 4, v4
	v_lshlrev_b32_e32 v3, 6, v1
	;;#ASMSTART
	s_waitcnt vmcnt(6)
	;;#ASMEND
	s_barrier
	s_cbranch_scc1 .LBB4_8
; %bb.7:
	v_and_b32_e32 v12, 8, v10
	v_lshlrev_b32_e32 v13, 11, v5
	v_add_u32_e32 v14, v5, v2
	v_lshlrev_b32_e32 v6, 4, v2
	v_lshl_or_b32 v165, v14, 7, v13
	v_or_b32_e32 v13, v3, v12
	v_lshl_or_b32 v177, v4, 8, v8
	v_lshl_or_b32 v178, v4, 7, v9
	v_or_b32_e32 v176, v167, v12
	v_lshl_or_b32 v171, v11, 7, v7
	v_bitop3_b32 v172, v167, v6, v12 bitop3:0x36
	v_bitop3_b32 v170, v3, v6, v12 bitop3:0x36
	v_bitop3_b32 v163, v13, v6, 16 bitop3:0x36
	v_bitop3_b32 v161, v13, v6, 32 bitop3:0x36
	v_bitop3_b32 v159, v13, v6, 48 bitop3:0x36
	s_mov_b64 s[4:5], 0
	s_branch .LBB4_9
.LBB4_8:
	s_mov_b64 s[4:5], -1
                                        ; implicit-def: $vgpr177
                                        ; implicit-def: $vgpr178
                                        ; implicit-def: $vgpr6
                                        ; implicit-def: $vgpr176
                                        ; implicit-def: $vgpr171
                                        ; implicit-def: $vgpr172
                                        ; implicit-def: $vgpr165
                                        ; implicit-def: $vgpr170
                                        ; implicit-def: $vgpr163
                                        ; implicit-def: $vgpr161
                                        ; implicit-def: $vgpr159
.LBB4_9:
	s_load_dwordx2 s[2:3], s[0:1], 0xc0
	s_load_dwordx2 s[20:21], s[0:1], 0xe0
	s_add_i32 s35, s34, -2
	s_mov_b32 s36, 0
	s_mov_b32 s26, 1
	s_andn2_b64 vcc, exec, s[4:5]
	v_mov_b32_e32 v33, 0
	s_mov_b32 s10, s38
	s_mov_b32 s14, s37
	s_cbranch_vccnz .LBB4_16
; %bb.10:
	v_lshl_or_b32 v177, v4, 8, v8
	v_lshl_or_b32 v178, v4, 7, v9
	v_and_b32_e32 v4, 8, v10
	v_or_b32_e32 v176, v167, v4
	v_lshl_or_b32 v171, v11, 7, v7
	v_lshlrev_b32_e32 v7, 11, v5
	v_add_u32_e32 v2, v5, v2
	v_bitop3_b32 v172, v167, v181, v4 bitop3:0x36
	v_bitop3_b32 v6, v176, v181, 16 bitop3:0x36
	v_lshl_or_b32 v165, v2, 7, v7
	v_or_b32_e32 v2, v3, v4
	s_lshl_b32 s4, s30, 8
	v_mov_b32_e32 v30, 0
	v_or_b32_e32 v182, v171, v172
	v_or_b32_e32 v183, v171, v6
	v_or_b32_e32 v184, v165, v172
	v_or_b32_e32 v185, v165, v6
	v_bitop3_b32 v170, v3, v181, v4 bitop3:0x36
	v_bitop3_b32 v163, v2, v181, 16 bitop3:0x36
	v_bitop3_b32 v161, v2, v181, 32 bitop3:0x36
	v_bitop3_b32 v159, v2, v181, 48 bitop3:0x36
	s_max_i32 s37, s35, 1
	s_add_i32 s30, s31, s4
	s_lshl_b32 s38, s24, 8
	s_add_i32 s39, s6, 0x80
	s_mov_b32 s7, 0x110000
	s_mov_b32 s40, 0
	s_mov_b32 s41, 0
	s_waitcnt lgkmcnt(0)
	s_mov_b32 s21, 0
	v_mov_b32_e32 v31, v30
	v_mov_b32_e32 v32, v30
	v_mov_b32_e32 v33, v30
	v_mov_b32_e32 v62, v30
	v_mov_b32_e32 v63, v30
	v_mov_b32_e32 v64, v30
	v_mov_b32_e32 v65, v30
	v_mov_b32_e32 v78, v30
	v_mov_b32_e32 v79, v30
	v_mov_b32_e32 v80, v30
	v_mov_b32_e32 v81, v30
	v_mov_b32_e32 v94, v30
	v_mov_b32_e32 v95, v30
	v_mov_b32_e32 v96, v30
	v_mov_b32_e32 v97, v30
	v_mov_b32_e32 v110, v30
	v_mov_b32_e32 v111, v30
	v_mov_b32_e32 v112, v30
	v_mov_b32_e32 v113, v30
	v_mov_b32_e32 v126, v30
	v_mov_b32_e32 v127, v30
	v_mov_b32_e32 v128, v30
	v_mov_b32_e32 v129, v30
	v_mov_b32_e32 v138, v30
	v_mov_b32_e32 v139, v30
	v_mov_b32_e32 v140, v30
	v_mov_b32_e32 v141, v30
	v_mov_b32_e32 v146, v30
	v_mov_b32_e32 v147, v30
	v_mov_b32_e32 v148, v30
	v_mov_b32_e32 v149, v30
	v_mov_b32_e32 v58, v30
	v_mov_b32_e32 v59, v30
	v_mov_b32_e32 v60, v30
	v_mov_b32_e32 v61, v30
	v_mov_b32_e32 v74, v30
	v_mov_b32_e32 v75, v30
	v_mov_b32_e32 v76, v30
	v_mov_b32_e32 v77, v30
	v_mov_b32_e32 v90, v30
	v_mov_b32_e32 v91, v30
	v_mov_b32_e32 v92, v30
	v_mov_b32_e32 v93, v30
	v_mov_b32_e32 v106, v30
	v_mov_b32_e32 v107, v30
	v_mov_b32_e32 v108, v30
	v_mov_b32_e32 v109, v30
	v_mov_b32_e32 v122, v30
	v_mov_b32_e32 v123, v30
	v_mov_b32_e32 v124, v30
	v_mov_b32_e32 v125, v30
	v_mov_b32_e32 v134, v30
	v_mov_b32_e32 v135, v30
	v_mov_b32_e32 v136, v30
	v_mov_b32_e32 v137, v30
	v_mov_b32_e32 v142, v30
	v_mov_b32_e32 v143, v30
	v_mov_b32_e32 v144, v30
	v_mov_b32_e32 v145, v30
	v_mov_b32_e32 v150, v30
	v_mov_b32_e32 v151, v30
	v_mov_b32_e32 v152, v30
	v_mov_b32_e32 v153, v30
	v_mov_b32_e32 v26, v30
	v_mov_b32_e32 v27, v30
	v_mov_b32_e32 v28, v30
	v_mov_b32_e32 v29, v30
	v_mov_b32_e32 v38, v30
	v_mov_b32_e32 v39, v30
	v_mov_b32_e32 v40, v30
	v_mov_b32_e32 v41, v30
	v_mov_b32_e32 v46, v30
	v_mov_b32_e32 v47, v30
	v_mov_b32_e32 v48, v30
	v_mov_b32_e32 v49, v30
	v_mov_b32_e32 v54, v30
	v_mov_b32_e32 v55, v30
	v_mov_b32_e32 v56, v30
	v_mov_b32_e32 v57, v30
	v_mov_b32_e32 v70, v30
	v_mov_b32_e32 v71, v30
	v_mov_b32_e32 v72, v30
	v_mov_b32_e32 v73, v30
	v_mov_b32_e32 v86, v30
	v_mov_b32_e32 v87, v30
	v_mov_b32_e32 v88, v30
	v_mov_b32_e32 v89, v30
	v_mov_b32_e32 v102, v30
	v_mov_b32_e32 v103, v30
	v_mov_b32_e32 v104, v30
	v_mov_b32_e32 v105, v30
	v_mov_b32_e32 v118, v30
	v_mov_b32_e32 v119, v30
	v_mov_b32_e32 v120, v30
	v_mov_b32_e32 v121, v30
	v_mov_b32_e32 v34, v30
	v_mov_b32_e32 v35, v30
	v_mov_b32_e32 v36, v30
	v_mov_b32_e32 v37, v30
	v_mov_b32_e32 v42, v30
	v_mov_b32_e32 v43, v30
	v_mov_b32_e32 v44, v30
	v_mov_b32_e32 v45, v30
	v_mov_b32_e32 v50, v30
	v_mov_b32_e32 v51, v30
	v_mov_b32_e32 v52, v30
	v_mov_b32_e32 v53, v30
	v_mov_b32_e32 v66, v30
	v_mov_b32_e32 v67, v30
	v_mov_b32_e32 v68, v30
	v_mov_b32_e32 v69, v30
	v_mov_b32_e32 v82, v30
	v_mov_b32_e32 v83, v30
	v_mov_b32_e32 v84, v30
	v_mov_b32_e32 v85, v30
	v_mov_b32_e32 v98, v30
	v_mov_b32_e32 v99, v30
	v_mov_b32_e32 v100, v30
	v_mov_b32_e32 v101, v30
	v_mov_b32_e32 v114, v30
	v_mov_b32_e32 v115, v30
	v_mov_b32_e32 v116, v30
	v_mov_b32_e32 v117, v30
	v_mov_b32_e32 v130, v30
	v_mov_b32_e32 v131, v30
	v_mov_b32_e32 v132, v30
	v_mov_b32_e32 v133, v30
                                        ; implicit-def: $vgpr156
                                        ; implicit-def: $vgpr168
	s_branch .LBB4_12
.LBB4_11:                               ;   in Loop: Header=BB4_12 Depth=1
	s_mul_i32 s16, s21, 0x8800
	s_add_i32 s42, s16, 0x11000
	v_add_u32_e32 v6, s42, v182
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v6 offset:0
ds_read_b64_tr_b8 v[4:5], v6 offset:1024

	;;#ASMEND
	v_add_u32_e32 v6, s42, v183
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v6 offset:0
ds_read_b64_tr_b8 v[12:13], v6 offset:1024

	;;#ASMEND
	v_add_u32_e32 v14, s42, v184
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v14 offset:0
ds_read_b64_tr_b8 v[8:9], v14 offset:1024

	;;#ASMEND
	v_add_u32_e32 v18, s42, v185
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v18 offset:0
ds_read_b64_tr_b8 v[16:17], v18 offset:1024

	;;#ASMEND
	v_add_u32_e32 v18, s16, v171
	v_add_u32_e32 v19, v18, v170
	;;#ASMSTART
	ds_read_b64_tr_b8 v[186:187], v19 offset:0
ds_read_b64_tr_b8 v[188:189], v19 offset:1024

	;;#ASMEND
	v_add_u32_e32 v19, v18, v163
	;;#ASMSTART
	ds_read_b64_tr_b8 v[194:195], v19 offset:0
ds_read_b64_tr_b8 v[196:197], v19 offset:1024

	;;#ASMEND
	v_add_u32_e32 v19, v18, v161
	;;#ASMSTART
	ds_read_b64_tr_b8 v[202:203], v19 offset:0
ds_read_b64_tr_b8 v[204:205], v19 offset:1024

	;;#ASMEND
	v_add_u32_e32 v18, v18, v159
	;;#ASMSTART
	ds_read_b64_tr_b8 v[210:211], v18 offset:0
ds_read_b64_tr_b8 v[212:213], v18 offset:1024

	;;#ASMEND
	v_add_u32_e32 v18, s16, v165
	v_add_u32_e32 v19, v18, v170
	;;#ASMSTART
	ds_read_b64_tr_b8 v[190:191], v19 offset:0
ds_read_b64_tr_b8 v[192:193], v19 offset:1024

	;;#ASMEND
	v_add_u32_e32 v19, v18, v163
	;;#ASMSTART
	ds_read_b64_tr_b8 v[198:199], v19 offset:0
ds_read_b64_tr_b8 v[200:201], v19 offset:1024

	;;#ASMEND
	s_mul_i32 s4, s26, 0x8800
	v_add_u32_e32 v19, v18, v161
	;;#ASMSTART
	ds_read_b64_tr_b8 v[206:207], v19 offset:0
ds_read_b64_tr_b8 v[208:209], v19 offset:1024

	;;#ASMEND
	v_add_u32_e32 v18, v18, v159
	s_add_i32 s17, s4, 0x4400
	s_add_i32 s4, s39, s27
	;;#ASMSTART
	ds_read_b64_tr_b8 v[214:215], v18 offset:0
ds_read_b64_tr_b8 v[216:217], v18 offset:1024

	;;#ASMEND
	s_add_i32 s41, s41, 1
	s_ashr_i32 s5, s4, 31
	v_add_u32_e32 v18, s17, v179
	s_add_u32 s4, s22, s4
	v_readfirstlane_b32 s19, v18
	v_add_u32_e32 v18, s17, v180
	s_addc_u32 s5, s23, s5
	s_mov_b32 m0, s19
	v_readfirstlane_b32 s17, v18
	buffer_load_dwordx4 v160, s[4:7], 0 offen lds
	s_mov_b32 m0, s17
	s_add_i32 s43, s16, 0x15400
	buffer_load_dwordx4 v158, s[4:7], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(3)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	s_waitcnt vmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[146:149], v[186:193], v[2:9], v[146:149], v154, v168 op_sel_hi:[0,0,0]
	v_add_u32_e32 v18, s43, v182
	;;#ASMSTART
	ds_read_b64_tr_b8 v[218:219], v18 offset:0
ds_read_b64_tr_b8 v[220:221], v18 offset:1024

	;;#ASMEND
	v_add_u32_e32 v22, s43, v183
	;;#ASMSTART
	ds_read_b64_tr_b8 v[18:19], v22 offset:0
ds_read_b64_tr_b8 v[20:21], v22 offset:1024

	;;#ASMEND
	v_add_u32_e32 v22, s43, v184
	;;#ASMSTART
	ds_read_b64_tr_b8 v[222:223], v22 offset:0
ds_read_b64_tr_b8 v[224:225], v22 offset:1024

	;;#ASMEND
	v_add_u32_e32 v226, s43, v185
	v_mfma_scale_f32_16x16x128_f8f6f4 v[138:141], v[186:193], v[10:17], v[138:141], v154, v168 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	;;#ASMSTART
	ds_read_b64_tr_b8 v[22:23], v226 offset:0
ds_read_b64_tr_b8 v[24:25], v226 offset:1024

	;;#ASMEND
	v_mfma_scale_f32_16x16x128_f8f6f4 v[126:129], v[194:201], v[2:9], v[126:129], v154, v168 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[110:113], v[194:201], v[10:17], v[110:113], v154, v168 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[202:209], v[2:9], v[94:97], v156, v168 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[202:209], v[10:17], v[78:81], v156, v168 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[210:217], v[2:9], v[62:65], v156, v168 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[210:217], v[10:17], v[30:33], v156, v168 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[150:153], v[186:193], v[218:225], v[150:153], v154, v169 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[142:145], v[186:193], v[18:25], v[142:145], v154, v169 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[134:137], v[194:201], v[218:225], v[134:137], v154, v169 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[122:125], v[194:201], v[18:25], v[122:125], v154, v169 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[106:109], v[202:209], v[218:225], v[106:109], v156, v169 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[202:209], v[18:25], v[90:93], v156, v169 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[210:217], v[218:225], v[74:77], v156, v169 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[210:217], v[18:25], v[58:61], v156, v169 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_add_i32 s4, s16, 0x4400
	v_add_u32_e32 v190, s4, v171
	s_barrier
	v_add_u32_e32 v191, v190, v170
	;;#ASMSTART
	ds_read_b64_tr_b8 v[186:187], v191 offset:0
ds_read_b64_tr_b8 v[188:189], v191 offset:1024

	;;#ASMEND
	v_add_u32_e32 v191, v190, v163
	;;#ASMSTART
	ds_read_b64_tr_b8 v[194:195], v191 offset:0
ds_read_b64_tr_b8 v[196:197], v191 offset:1024

	;;#ASMEND
	v_add_u32_e32 v191, v190, v161
	;;#ASMSTART
	ds_read_b64_tr_b8 v[202:203], v191 offset:0
ds_read_b64_tr_b8 v[204:205], v191 offset:1024

	;;#ASMEND
	v_add_u32_e32 v190, v190, v159
	;;#ASMSTART
	ds_read_b64_tr_b8 v[210:211], v190 offset:0
ds_read_b64_tr_b8 v[212:213], v190 offset:1024

	;;#ASMEND
	v_add_u32_e32 v214, s4, v165
	v_add_u32_e32 v198, v214, v170
	;;#ASMSTART
	ds_read_b64_tr_b8 v[190:191], v198 offset:0
ds_read_b64_tr_b8 v[192:193], v198 offset:1024

	;;#ASMEND
	v_add_u32_e32 v206, v214, v163
	;;#ASMSTART
	ds_read_b64_tr_b8 v[198:199], v206 offset:0
ds_read_b64_tr_b8 v[200:201], v206 offset:1024

	;;#ASMEND
	v_add_u32_e32 v215, v214, v161
	;;#ASMSTART
	ds_read_b64_tr_b8 v[206:207], v215 offset:0
ds_read_b64_tr_b8 v[208:209], v215 offset:1024

	;;#ASMEND
	v_add_u32_e32 v226, v214, v159
	s_add_i32 s4, s38, s27
	;;#ASMSTART
	ds_read_b64_tr_b8 v[214:215], v226 offset:0
ds_read_b64_tr_b8 v[216:217], v226 offset:1024

	;;#ASMEND
	s_ashr_i32 s5, s4, 31
	v_add_u32_e32 v226, s16, v179
	s_add_u32 s4, s22, s4
	v_readfirstlane_b32 s17, v226
	v_add_u32_e32 v226, s16, v180
	s_addc_u32 s5, s23, s5
	s_mov_b32 m0, s17
	v_readfirstlane_b32 s16, v226
	buffer_load_dwordx4 v160, s[4:7], 0 offen lds
	s_mov_b32 m0, s16
	v_add_u32_e32 v226, s43, v179
	buffer_load_dwordx4 v158, s[4:7], 0 offen lds
	s_add_i32 s4, s30, 0x80
	s_ashr_i32 s5, s4, 31
	s_add_u32 s16, s28, s4
	v_readfirstlane_b32 s4, v226
	v_add_u32_e32 v226, s43, v180
	s_addc_u32 s17, s29, s5
	s_mov_b32 s19, s7
	s_mov_b32 m0, s4
	v_readfirstlane_b32 s4, v226
	buffer_load_dwordx4 v162, s[16:19], 0 offen lds
	s_mov_b32 m0, s4
	s_nop 0
	buffer_load_dwordx4 v164, s[16:19], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[118:121], v[186:193], v[2:9], v[118:121], v155, v168 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[102:105], v[186:193], v[10:17], v[102:105], v155, v168 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[194:201], v[2:9], v[86:89], v155, v168 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[194:201], v[10:17], v[70:73], v155, v168 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[202:209], v[2:9], v[54:57], v157, v168 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], v[202:209], v[10:17], v[46:49], v157, v168 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[210:217], v[2:9], v[38:41], v157, v168 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[210:217], v[10:17], v[26:29], v157, v168 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[130:133], v[186:193], v[218:225], v[130:133], v155, v169 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[114:117], v[186:193], v[18:25], v[114:117], v155, v169 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[98:101], v[194:201], v[218:225], v[98:101], v155, v169 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[194:201], v[18:25], v[82:85], v155, v169 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[202:209], v[218:225], v[66:69], v157, v169 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[202:209], v[18:25], v[50:53], v157, v169 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[210:217], v[218:225], v[42:45], v157, v169 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[210:217], v[18:25], v[34:37], v157, v169 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_ashr_i32 s4, s30, 31
	s_add_u32 s16, s28, s30
	v_add_u32_e32 v2, s42, v179
	s_addc_u32 s17, s29, s4
	v_readfirstlane_b32 s4, v2
	v_add_u32_e32 v2, s42, v180
	s_mov_b32 m0, s4
	v_readfirstlane_b32 s4, v2
	s_barrier
	buffer_load_dwordx4 v162, s[16:19], 0 offen lds
	s_mov_b32 m0, s4
	s_xor_b32 s21, s21, 1
	buffer_load_dwordx4 v164, s[16:19], 0 offen lds
	s_xor_b32 s26, s26, 1
	s_add_i32 s30, s30, s18
	s_add_i32 s27, s27, s6
	s_addk_i32 s40, 0x200
	s_addk_i32 s36, 0x100
	s_cmp_eq_u32 s37, s41
	s_cbranch_scc1 .LBB4_17
.LBB4_12:                               ; =>This Inner Loop Header: Depth=1
	s_bitcmp0_b32 s41, 0
	s_cbranch_scc1 .LBB4_14
; %bb.13:                               ;   in Loop: Header=BB4_12 Depth=1
	v_lshrrev_b32_e32 v154, 16, v154
	v_lshrrev_b32_e32 v155, 16, v155
	v_lshrrev_b32_e32 v156, 16, v156
	v_lshrrev_b32_e32 v157, 16, v157
	v_lshrrev_b32_e32 v168, 16, v168
	v_lshrrev_b32_e32 v169, 16, v169
	s_cbranch_execnz .LBB4_11
	s_branch .LBB4_15
.LBB4_14:                               ;   in Loop: Header=BB4_12 Depth=1
                                        ; implicit-def: $vgpr168
                                        ; implicit-def: $vgpr155
.LBB4_15:                               ;   in Loop: Header=BB4_12 Depth=1
	buffer_load_dwordx4 v[154:157], v177, s[8:11], s40 offen
	buffer_load_dwordx2 v[168:169], v178, s[12:15], s36 offen
	s_branch .LBB4_11
.LBB4_16:
	s_waitcnt lgkmcnt(0)
	s_mov_b32 s21, 0
	s_mov_b64 s[26:27], 1
	v_mov_b32_e32 v32, 0
	v_mov_b32_e32 v31, 0
	v_mov_b32_e32 v30, 0
	v_mov_b32_e32 v65, 0
	v_mov_b32_e32 v64, 0
	v_mov_b32_e32 v63, 0
	v_mov_b32_e32 v62, 0
	v_mov_b32_e32 v81, 0
	v_mov_b32_e32 v80, 0
	v_mov_b32_e32 v79, 0
	v_mov_b32_e32 v78, 0
	v_mov_b32_e32 v97, 0
	v_mov_b32_e32 v96, 0
	v_mov_b32_e32 v95, 0
	v_mov_b32_e32 v94, 0
	v_mov_b32_e32 v113, 0
	v_mov_b32_e32 v112, 0
	v_mov_b32_e32 v111, 0
	v_mov_b32_e32 v110, 0
	v_mov_b32_e32 v129, 0
	v_mov_b32_e32 v128, 0
	v_mov_b32_e32 v127, 0
	v_mov_b32_e32 v126, 0
	v_mov_b32_e32 v141, 0
	v_mov_b32_e32 v140, 0
	v_mov_b32_e32 v139, 0
	v_mov_b32_e32 v138, 0
	v_mov_b32_e32 v149, 0
	v_mov_b32_e32 v148, 0
	v_mov_b32_e32 v147, 0
	v_mov_b32_e32 v146, 0
	v_mov_b32_e32 v61, 0
	v_mov_b32_e32 v60, 0
	v_mov_b32_e32 v59, 0
	v_mov_b32_e32 v58, 0
	v_mov_b32_e32 v77, 0
	v_mov_b32_e32 v76, 0
	v_mov_b32_e32 v75, 0
	v_mov_b32_e32 v74, 0
	v_mov_b32_e32 v93, 0
	v_mov_b32_e32 v92, 0
	v_mov_b32_e32 v91, 0
	v_mov_b32_e32 v90, 0
	v_mov_b32_e32 v109, 0
	v_mov_b32_e32 v108, 0
	v_mov_b32_e32 v107, 0
	v_mov_b32_e32 v106, 0
	v_mov_b32_e32 v125, 0
	v_mov_b32_e32 v124, 0
	v_mov_b32_e32 v123, 0
	v_mov_b32_e32 v122, 0
	v_mov_b32_e32 v137, 0
	v_mov_b32_e32 v136, 0
	v_mov_b32_e32 v135, 0
	v_mov_b32_e32 v134, 0
	v_mov_b32_e32 v145, 0
	v_mov_b32_e32 v144, 0
	v_mov_b32_e32 v143, 0
	v_mov_b32_e32 v142, 0
	v_mov_b32_e32 v153, 0
	v_mov_b32_e32 v152, 0
	v_mov_b32_e32 v151, 0
	v_mov_b32_e32 v150, 0
	v_mov_b32_e32 v29, 0
	v_mov_b32_e32 v28, 0
	v_mov_b32_e32 v27, 0
	v_mov_b32_e32 v26, 0
	v_mov_b32_e32 v41, 0
	v_mov_b32_e32 v40, 0
	v_mov_b32_e32 v39, 0
	v_mov_b32_e32 v38, 0
	v_mov_b32_e32 v49, 0
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v47, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v57, 0
	v_mov_b32_e32 v56, 0
	v_mov_b32_e32 v55, 0
	v_mov_b32_e32 v54, 0
	v_mov_b32_e32 v73, 0
	v_mov_b32_e32 v72, 0
	v_mov_b32_e32 v71, 0
	v_mov_b32_e32 v70, 0
	v_mov_b32_e32 v89, 0
	v_mov_b32_e32 v88, 0
	v_mov_b32_e32 v87, 0
	v_mov_b32_e32 v86, 0
	v_mov_b32_e32 v105, 0
	v_mov_b32_e32 v104, 0
	v_mov_b32_e32 v103, 0
	v_mov_b32_e32 v102, 0
	v_mov_b32_e32 v121, 0
	v_mov_b32_e32 v120, 0
	v_mov_b32_e32 v119, 0
	v_mov_b32_e32 v118, 0
	v_mov_b32_e32 v37, 0
	v_mov_b32_e32 v36, 0
	v_mov_b32_e32 v35, 0
	v_mov_b32_e32 v34, 0
	v_mov_b32_e32 v45, 0
	v_mov_b32_e32 v44, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v42, 0
	v_mov_b32_e32 v53, 0
	v_mov_b32_e32 v52, 0
	v_mov_b32_e32 v51, 0
	v_mov_b32_e32 v50, 0
	v_mov_b32_e32 v69, 0
	v_mov_b32_e32 v68, 0
	v_mov_b32_e32 v67, 0
	v_mov_b32_e32 v66, 0
	v_mov_b32_e32 v85, 0
	v_mov_b32_e32 v84, 0
	v_mov_b32_e32 v83, 0
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v101, 0
	v_mov_b32_e32 v100, 0
	v_mov_b32_e32 v99, 0
	v_mov_b32_e32 v98, 0
	v_mov_b32_e32 v117, 0
	v_mov_b32_e32 v116, 0
	v_mov_b32_e32 v115, 0
	v_mov_b32_e32 v114, 0
	v_mov_b32_e32 v133, 0
	v_mov_b32_e32 v132, 0
	v_mov_b32_e32 v131, 0
	v_mov_b32_e32 v130, 0
	s_branch .LBB4_18
.LBB4_17:
	v_mov_b32_e32 v6, v181
.LBB4_18:
	s_load_dword s16, s[0:1], 0xf4
	s_ashr_i32 s0, s35, 1
	s_lshl_b32 s1, s0, 10
	s_lshl_b32 s0, s0, 9
	buffer_load_dwordx2 v[168:169], v178, s[12:15], s0 offen
	s_mul_i32 s0, s21, 0x8800
	buffer_load_dwordx4 v[18:21], v177, s[8:11], s1 offen
	s_add_i32 s1, s0, 0x11000
	v_add_u32_e32 v7, s1, v171
	v_add_u32_e32 v8, v7, v172
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v8 offset:0
ds_read_b64_tr_b8 v[4:5], v8 offset:1024

	;;#ASMEND
	v_bitop3_b32 v162, v176, v6, 16 bitop3:0x36
	v_add_u32_e32 v6, v7, v162
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v6 offset:0
ds_read_b64_tr_b8 v[12:13], v6 offset:1024

	;;#ASMEND
	v_add_u32_e32 v14, s1, v165
	v_add_u32_e32 v15, v14, v172
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v15 offset:0
ds_read_b64_tr_b8 v[8:9], v15 offset:1024

	;;#ASMEND
	v_add_u32_e32 v22, v14, v162
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v22 offset:0
ds_read_b64_tr_b8 v[16:17], v22 offset:1024

	;;#ASMEND
	v_add_u32_e32 v22, s0, v171
	v_add_u32_e32 v23, v22, v170
	;;#ASMSTART
	ds_read_b64_tr_b8 v[176:177], v23 offset:0
ds_read_b64_tr_b8 v[178:179], v23 offset:1024

	;;#ASMEND
	v_add_u32_e32 v23, v22, v163
	;;#ASMSTART
	ds_read_b64_tr_b8 v[184:185], v23 offset:0
ds_read_b64_tr_b8 v[186:187], v23 offset:1024

	;;#ASMEND
	v_add_u32_e32 v23, v22, v161
	;;#ASMSTART
	ds_read_b64_tr_b8 v[192:193], v23 offset:0
ds_read_b64_tr_b8 v[194:195], v23 offset:1024

	;;#ASMEND
	v_add_u32_e32 v22, v22, v159
	;;#ASMSTART
	ds_read_b64_tr_b8 v[200:201], v22 offset:0
ds_read_b64_tr_b8 v[202:203], v22 offset:1024

	;;#ASMEND
	v_add_u32_e32 v22, s0, v165
	v_add_u32_e32 v23, v22, v170
	;;#ASMSTART
	ds_read_b64_tr_b8 v[180:181], v23 offset:0
ds_read_b64_tr_b8 v[182:183], v23 offset:1024

	;;#ASMEND
	s_add_i32 s4, s34, -1
	v_add_u32_e32 v23, v22, v163
	;;#ASMSTART
	ds_read_b64_tr_b8 v[188:189], v23 offset:0
ds_read_b64_tr_b8 v[190:191], v23 offset:1024

	;;#ASMEND
	s_mul_i32 s1, s26, 0x8800
	s_mul_i32 s4, s4, s24
	v_add_u32_e32 v23, v22, v161
	;;#ASMSTART
	ds_read_b64_tr_b8 v[196:197], v23 offset:0
ds_read_b64_tr_b8 v[198:199], v23 offset:1024

	;;#ASMEND
	v_add_u32_e32 v22, v22, v159
	s_add_i32 s8, s1, 0x4400
	s_add_i32 s33, s33, s4
	;;#ASMSTART
	ds_read_b64_tr_b8 v[204:205], v22 offset:0
ds_read_b64_tr_b8 v[206:207], v22 offset:1024

	;;#ASMEND
	s_lshl_b32 s4, s33, 7
	v_add_u32_e32 v22, s8, v173
	s_ashr_i32 s5, s4, 31
	v_add_u32_e32 v23, v22, v174
	s_add_u32 s4, s22, s4
	v_readfirstlane_b32 s8, v23
	v_add_u32_e32 v22, v22, v175
	s_addc_u32 s5, s23, s5
	s_mov_b32 s7, 0x110000
	s_mov_b32 m0, s8
	v_readfirstlane_b32 s8, v22
	buffer_load_dwordx4 v160, s[4:7], 0 offen lds
	s_mov_b32 m0, s8
	s_nop 0
	buffer_load_dwordx4 v158, s[4:7], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	s_waitcnt vmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[176:183], v[2:9], v[146:149], v18, v168 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[138:141], v[176:183], v[10:17], v[138:141], v18, v168 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[126:129], v[184:191], v[2:9], v[126:129], v18, v168 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[110:113], v[184:191], v[10:17], v[110:113], v18, v168 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[192:199], v[2:9], v[94:97], v20, v168 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[192:199], v[10:17], v[78:81], v20, v168 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[200:207], v[2:9], v[62:65], v20, v168 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[200:207], v[10:17], v[30:33], v20, v168 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_add_i32 s4, s0, 0x15400
	v_add_u32_e32 v146, s4, v171
	s_barrier
	v_add_u32_e32 v147, v146, v172
	;;#ASMSTART
	ds_read_b64_tr_b8 v[208:209], v147 offset:0
ds_read_b64_tr_b8 v[210:211], v147 offset:1024

	;;#ASMEND
	v_add_u32_e32 v146, v146, v162
	;;#ASMSTART
	ds_read_b64_tr_b8 v[216:217], v146 offset:0
ds_read_b64_tr_b8 v[218:219], v146 offset:1024

	;;#ASMEND
	v_add_u32_e32 v146, s4, v165
	v_add_u32_e32 v147, v146, v172
	;;#ASMSTART
	ds_read_b64_tr_b8 v[212:213], v147 offset:0
ds_read_b64_tr_b8 v[214:215], v147 offset:1024

	;;#ASMEND
	v_add_u32_e32 v146, v146, v162
	;;#ASMSTART
	ds_read_b64_tr_b8 v[220:221], v146 offset:0
ds_read_b64_tr_b8 v[222:223], v146 offset:1024

	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[146:149], v[176:183], v[208:215], v[150:153], v18, v169 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[142:145], v[176:183], v[216:223], v[142:145], v18, v169 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[134:137], v[184:191], v[208:215], v[134:137], v18, v169 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[122:125], v[184:191], v[216:223], v[122:125], v18, v169 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[106:109], v[192:199], v[208:215], v[106:109], v20, v169 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[192:199], v[216:223], v[90:93], v20, v169 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[200:207], v[208:215], v[74:77], v20, v169 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[200:207], v[216:223], v[58:61], v20, v169 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_addk_i32 s0, 0x4400
	v_add_u32_e32 v154, s0, v171
	s_barrier
	v_add_u32_e32 v155, v154, v170
	;;#ASMSTART
	ds_read_b64_tr_b8 v[150:151], v155 offset:0
ds_read_b64_tr_b8 v[152:153], v155 offset:1024

	;;#ASMEND
	v_add_u32_e32 v155, v154, v163
	;;#ASMSTART
	ds_read_b64_tr_b8 v[174:175], v155 offset:0
ds_read_b64_tr_b8 v[176:177], v155 offset:1024

	;;#ASMEND
	v_add_u32_e32 v155, v154, v161
	;;#ASMSTART
	ds_read_b64_tr_b8 v[182:183], v155 offset:0
ds_read_b64_tr_b8 v[184:185], v155 offset:1024

	;;#ASMEND
	v_add_u32_e32 v154, v154, v159
	;;#ASMSTART
	ds_read_b64_tr_b8 v[190:191], v154 offset:0
ds_read_b64_tr_b8 v[192:193], v154 offset:1024

	;;#ASMEND
	v_add_u32_e32 v158, s0, v165
	v_add_u32_e32 v160, v158, v170
	;;#ASMSTART
	ds_read_b64_tr_b8 v[154:155], v160 offset:0
ds_read_b64_tr_b8 v[156:157], v160 offset:1024

	;;#ASMEND
	v_add_u32_e32 v160, v158, v163
	;;#ASMSTART
	ds_read_b64_tr_b8 v[178:179], v160 offset:0
ds_read_b64_tr_b8 v[180:181], v160 offset:1024

	;;#ASMEND
	v_add_u32_e32 v160, v158, v161
	;;#ASMSTART
	ds_read_b64_tr_b8 v[186:187], v160 offset:0
ds_read_b64_tr_b8 v[188:189], v160 offset:1024

	;;#ASMEND
	v_add_u32_e32 v158, v158, v159
	;;#ASMSTART
	ds_read_b64_tr_b8 v[194:195], v158 offset:0
ds_read_b64_tr_b8 v[196:197], v158 offset:1024

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(2)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[118:121], v[150:157], v[2:9], v[118:121], v19, v168 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[102:105], v[150:157], v[10:17], v[102:105], v19, v168 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[174:181], v[2:9], v[86:89], v19, v168 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[174:181], v[10:17], v[70:73], v19, v168 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[182:189], v[2:9], v[54:57], v21, v168 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], v[182:189], v[10:17], v[46:49], v21, v168 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[190:197], v[2:9], v[38:41], v21, v168 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[190:197], v[10:17], v[26:29], v21, v168 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_add_i32 s1, s1, 0x11000
	v_add_u32_e32 v6, s1, v171
	s_barrier
	v_add_u32_e32 v7, v6, v172
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v7 offset:0
ds_read_b64_tr_b8 v[4:5], v7 offset:1024

	;;#ASMEND
	v_add_u32_e32 v6, v6, v162
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v6 offset:0
ds_read_b64_tr_b8 v[12:13], v6 offset:1024

	;;#ASMEND
	v_add_u32_e32 v14, s1, v165
	v_add_u32_e32 v15, v14, v172
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v15 offset:0
ds_read_b64_tr_b8 v[8:9], v15 offset:1024

	;;#ASMEND
	v_add_u32_e32 v158, v14, v162
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v158 offset:0
ds_read_b64_tr_b8 v[16:17], v158 offset:1024

	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[130:133], v[150:157], v[208:215], v[130:133], v19, v169 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[114:117], v[150:157], v[216:223], v[114:117], v19, v169 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[98:101], v[174:181], v[208:215], v[98:101], v19, v169 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[174:181], v[216:223], v[82:85], v19, v169 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[182:189], v[208:215], v[66:69], v21, v169 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[182:189], v[216:223], v[50:53], v21, v169 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[190:197], v[208:215], v[42:45], v21, v169 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[190:197], v[216:223], v[34:37], v21, v169 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_xor_b32 s0, s21, 1
	s_mul_i32 s0, s0, 0x8800
	v_add_u32_e32 v150, s0, v171
	s_barrier
	v_add_u32_e32 v151, v150, v170
	;;#ASMSTART
	ds_read_b64_tr_b8 v[174:175], v151 offset:0
ds_read_b64_tr_b8 v[176:177], v151 offset:1024

	;;#ASMEND
	v_add_u32_e32 v151, v150, v163
	;;#ASMSTART
	ds_read_b64_tr_b8 v[182:183], v151 offset:0
ds_read_b64_tr_b8 v[184:185], v151 offset:1024

	;;#ASMEND
	v_add_u32_e32 v151, v150, v161
	;;#ASMSTART
	ds_read_b64_tr_b8 v[190:191], v151 offset:0
ds_read_b64_tr_b8 v[192:193], v151 offset:1024

	;;#ASMEND
	v_add_u32_e32 v150, v150, v159
	;;#ASMSTART
	ds_read_b64_tr_b8 v[198:199], v150 offset:0
ds_read_b64_tr_b8 v[200:201], v150 offset:1024

	;;#ASMEND
	v_add_u32_e32 v150, s0, v165
	v_add_u32_e32 v151, v150, v170
	;;#ASMSTART
	ds_read_b64_tr_b8 v[178:179], v151 offset:0
ds_read_b64_tr_b8 v[180:181], v151 offset:1024

	;;#ASMEND
	v_add_u32_e32 v151, v150, v163
	;;#ASMSTART
	ds_read_b64_tr_b8 v[186:187], v151 offset:0
ds_read_b64_tr_b8 v[188:189], v151 offset:1024

	;;#ASMEND
	v_add_u32_e32 v151, v150, v161
	;;#ASMSTART
	ds_read_b64_tr_b8 v[194:195], v151 offset:0
ds_read_b64_tr_b8 v[196:197], v151 offset:1024

	;;#ASMEND
	v_add_u32_e32 v150, v150, v159
	;;#ASMSTART
	ds_read_b64_tr_b8 v[202:203], v150 offset:0
ds_read_b64_tr_b8 v[204:205], v150 offset:1024

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[150:153], v[174:181], v[2:9], v[22:25], v18, v168 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[138:141], v[174:181], v[10:17], v[138:141], v18, v168 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[126:129], v[182:189], v[2:9], v[126:129], v18, v168 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[110:113], v[182:189], v[10:17], v[110:113], v18, v168 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[190:197], v[2:9], v[94:97], v20, v168 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[190:197], v[10:17], v[78:81], v20, v168 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[198:205], v[2:9], v[62:65], v20, v168 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[198:205], v[10:17], v[30:33], v20, v168 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_add_i32 s1, s0, 0x15400
	s_nop 4
	v_add_u32_e32 v30, s1, v171
	s_barrier
	v_add_u32_e32 v31, v30, v172
	;;#ASMSTART
	ds_read_b64_tr_b8 v[206:207], v31 offset:0
ds_read_b64_tr_b8 v[208:209], v31 offset:1024

	;;#ASMEND
	v_add_u32_e32 v30, v30, v162
	;;#ASMSTART
	ds_read_b64_tr_b8 v[214:215], v30 offset:0
ds_read_b64_tr_b8 v[216:217], v30 offset:1024

	;;#ASMEND
	v_add_u32_e32 v30, s1, v165
	v_add_u32_e32 v31, v30, v172
	;;#ASMSTART
	ds_read_b64_tr_b8 v[210:211], v31 offset:0
ds_read_b64_tr_b8 v[212:213], v31 offset:1024

	;;#ASMEND
	v_add_u32_e32 v30, v30, v162
	;;#ASMSTART
	ds_read_b64_tr_b8 v[218:219], v30 offset:0
ds_read_b64_tr_b8 v[220:221], v30 offset:1024

	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[146:149], v[174:181], v[206:213], v[146:149], v18, v169 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[142:145], v[174:181], v[214:221], v[142:145], v18, v169 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[134:137], v[182:189], v[206:213], v[134:137], v18, v169 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[122:125], v[182:189], v[214:221], v[122:125], v18, v169 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[106:109], v[190:197], v[206:213], v[106:109], v20, v169 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[190:197], v[214:221], v[90:93], v20, v169 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[198:205], v[206:213], v[74:77], v20, v169 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[198:205], v[214:221], v[58:61], v20, v169 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_addk_i32 s0, 0x4400
	v_add_u32_e32 v18, s0, v171
	s_barrier
	v_add_u32_e32 v20, v18, v170
	;;#ASMSTART
	ds_read_b64_tr_b8 v[172:173], v20 offset:0
ds_read_b64_tr_b8 v[174:175], v20 offset:1024

	;;#ASMEND
	v_add_u32_e32 v20, v18, v163
	;;#ASMSTART
	ds_read_b64_tr_b8 v[180:181], v20 offset:0
ds_read_b64_tr_b8 v[182:183], v20 offset:1024

	;;#ASMEND
	v_add_u32_e32 v20, v18, v161
	;;#ASMSTART
	ds_read_b64_tr_b8 v[188:189], v20 offset:0
ds_read_b64_tr_b8 v[190:191], v20 offset:1024

	;;#ASMEND
	v_add_u32_e32 v18, v18, v159
	;;#ASMSTART
	ds_read_b64_tr_b8 v[196:197], v18 offset:0
ds_read_b64_tr_b8 v[198:199], v18 offset:1024

	;;#ASMEND
	v_add_u32_e32 v18, s0, v165
	v_add_u32_e32 v20, v18, v170
	;;#ASMSTART
	ds_read_b64_tr_b8 v[176:177], v20 offset:0
ds_read_b64_tr_b8 v[178:179], v20 offset:1024

	;;#ASMEND
	v_add_u32_e32 v20, v18, v163
	;;#ASMSTART
	ds_read_b64_tr_b8 v[184:185], v20 offset:0
ds_read_b64_tr_b8 v[186:187], v20 offset:1024

	;;#ASMEND
	v_add_u32_e32 v20, v18, v161
	;;#ASMSTART
	ds_read_b64_tr_b8 v[192:193], v20 offset:0
ds_read_b64_tr_b8 v[194:195], v20 offset:1024

	;;#ASMEND
	v_add_u32_e32 v18, v18, v159
	;;#ASMSTART
	ds_read_b64_tr_b8 v[200:201], v18 offset:0
ds_read_b64_tr_b8 v[202:203], v18 offset:1024

	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[162:165], v[172:179], v[2:9], v[118:121], v19, v168 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[158:161], v[172:179], v[10:17], v[102:105], v19, v168 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[154:157], v[180:187], v[2:9], v[86:89], v19, v168 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[118:121], v[180:187], v[10:17], v[70:73], v19, v168 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[102:105], v[188:195], v[2:9], v[54:57], v21, v168 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[188:195], v[10:17], v[46:49], v21, v168 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[196:203], v[2:9], v[38:41], v21, v168 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[196:203], v[10:17], v[26:29], v21, v168 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[172:179], v[206:213], v[130:133], v19, v169 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], v[172:179], v[214:221], v[114:117], v19, v169 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[180:187], v[206:213], v[98:101], v19, v169 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[180:187], v[214:221], v[82:85], v19, v169 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[188:195], v[206:213], v[66:69], v21, v169 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[188:195], v[214:221], v[50:53], v21, v169 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[6:9], v[196:203], v[206:213], v[42:45], v21, v169 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[2:5], v[196:203], v[214:221], v[34:37], v21, v169 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_movk_i32 s0, 0x100
	v_cmp_gt_u32_e32 vcc, s0, v0
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB4_20
; %bb.19:
	s_barrier
.LBB4_20:
	s_or_b64 exec, exec, s[0:1]
	v_lshl_or_b32 v177, s25, 2, v1
	v_lshrrev_b32_e32 v0, 2, v0
	v_or_b32_e32 v167, s31, v167
	v_mul_lo_u32 v1, v177, s20
	v_and_b32_e32 v0, 12, v0
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v173, s16, v2
	v_lshl_add_u32 v2, v1, 6, v167
	v_mad_u64_u32 v[0:1], s[0:1], v0, s20, v[166:167]
	v_ashrrev_i32_e32 v1, 31, v0
	v_mul_f32_e32 v175, s16, v4
	v_mul_f32_e32 v176, s16, v5
	v_lshlrev_b64 v[4:5], 1, v[0:1]
	v_add_u32_e32 v0, s20, v0
	v_ashrrev_i32_e32 v1, 31, v0
	v_mul_f32_e32 v171, s16, v8
	v_mul_f32_e32 v172, s16, v9
	v_lshlrev_b64 v[8:9], 1, v[0:1]
	v_add_u32_e32 v0, s20, v0
	v_ashrrev_i32_e32 v1, 31, v0
	v_mul_f32_e32 v130, s16, v165
	v_mul_f32_e32 v165, s16, v12
	v_mul_f32_e32 v168, s16, v13
	v_mul_f32_e32 v174, s16, v3
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshlrev_b64 v[12:13], 1, v[0:1]
	v_add_u32_e32 v0, s20, v0
	v_lshl_add_u64 v[2:3], v[2:3], 1, s[2:3]
	v_ashrrev_i32_e32 v1, 31, v0
	s_mul_i32 s0, s20, 13
	v_mul_f32_e32 v18, s16, v150
	v_mul_f32_e32 v19, s16, v151
	v_mul_f32_e32 v43, s16, v127
	v_mul_f32_e32 v44, s16, v128
	v_mul_f32_e32 v45, s16, v129
	v_mul_f32_e32 v51, s16, v111
	v_mul_f32_e32 v111, s16, v134
	v_mul_f32_e32 v127, s16, v162
	v_mul_f32_e32 v128, s16, v163
	v_mul_f32_e32 v129, s16, v164
	v_mul_f32_e32 v134, s16, v161
	v_mul_f32_e32 v161, s16, v16
	v_mul_f32_e32 v162, s16, v17
	v_mul_f32_e32 v163, s16, v10
	v_mul_f32_e32 v164, s16, v11
	v_mul_f32_e32 v169, s16, v6
	v_mul_f32_e32 v170, s16, v7
	v_lshl_add_u64 v[6:7], v[2:3], 0, v[4:5]
	v_lshl_add_u64 v[10:11], v[2:3], 0, v[8:9]
	v_lshlrev_b64 v[16:17], 1, v[0:1]
	v_add_u32_e32 v0, s0, v0
	v_mul_f32_e32 v20, s16, v152
	v_mul_f32_e32 v21, s16, v153
	v_mul_f32_e32 v132, s16, v159
	v_mul_f32_e32 v133, s16, v160
	v_mul_f32_e32 v159, s16, v14
	v_mul_f32_e32 v160, s16, v15
	global_store_short_d16_hi v[6:7], v18, off
	global_store_short_d16_hi v[10:11], v19, off
	v_lshl_add_u64 v[14:15], v[2:3], 0, v[12:13]
	v_lshl_add_u64 v[18:19], v[2:3], 0, v[16:17]
	v_ashrrev_i32_e32 v1, 31, v0
	v_mul_f32_e32 v34, s16, v138
	v_mul_f32_e32 v35, s16, v139
	v_mul_f32_e32 v36, s16, v140
	v_mul_f32_e32 v37, s16, v141
	global_store_short_d16_hi v[14:15], v20, off
	global_store_short_d16_hi v[18:19], v21, off
	global_store_short_d16_hi v[6:7], v34, off offset:32
	global_store_short_d16_hi v[10:11], v35, off offset:32
	global_store_short_d16_hi v[14:15], v36, off offset:32
	global_store_short_d16_hi v[18:19], v37, off offset:32
	v_lshlrev_b64 v[20:21], 1, v[0:1]
	v_add_u32_e32 v0, s20, v0
	v_ashrrev_i32_e32 v1, 31, v0
	v_mul_f32_e32 v66, s16, v94
	v_mul_f32_e32 v85, s16, v24
	v_mul_f32_e32 v94, s16, v25
	v_lshlrev_b64 v[24:25], 1, v[0:1]
	v_add_u32_e32 v0, s20, v0
	v_ashrrev_i32_e32 v1, 31, v0
	v_mul_f32_e32 v131, s16, v158
	v_mul_f32_e32 v138, s16, v157
	v_mul_f32_e32 v157, s16, v28
	v_mul_f32_e32 v158, s16, v29
	v_lshlrev_b64 v[28:29], 1, v[0:1]
	v_add_u32_e32 v0, s20, v0
	v_ashrrev_i32_e32 v1, 31, v0
	v_mul_f32_e32 v42, s16, v126
	v_mul_f32_e32 v115, s16, v122
	v_mul_f32_e32 v122, s16, v125
	v_mul_f32_e32 v125, s16, v32
	v_mul_f32_e32 v126, s16, v33
	v_lshlrev_b64 v[32:33], 1, v[0:1]
	v_add_u32_e32 v0, s0, v0
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshlrev_b64 v[36:37], 1, v[0:1]
	v_add_u32_e32 v0, s20, v0
	v_ashrrev_i32_e32 v1, 31, v0
	v_mul_f32_e32 v52, s16, v112
	v_mul_f32_e32 v112, s16, v135
	v_mul_f32_e32 v135, s16, v154
	v_mul_f32_e32 v153, s16, v40
	v_mul_f32_e32 v154, s16, v41
	v_lshlrev_b64 v[40:41], 1, v[0:1]
	v_add_u32_e32 v0, s20, v0
	v_mul_f32_e32 v53, s16, v113
	v_mul_f32_e32 v83, s16, v22
	v_mul_f32_e32 v84, s16, v23
	v_mul_f32_e32 v113, s16, v136
	v_mul_f32_e32 v114, s16, v137
	v_mul_f32_e32 v116, s16, v123
	v_mul_f32_e32 v117, s16, v124
	v_mul_f32_e32 v123, s16, v30
	v_mul_f32_e32 v124, s16, v31
	v_mul_f32_e32 v136, s16, v155
	v_mul_f32_e32 v137, s16, v156
	v_mul_f32_e32 v155, s16, v26
	v_mul_f32_e32 v156, s16, v27
	v_lshl_add_u64 v[22:23], v[2:3], 0, v[20:21]
	v_lshl_add_u64 v[26:27], v[2:3], 0, v[24:25]
	v_lshl_add_u64 v[30:31], v[2:3], 0, v[28:29]
	v_lshl_add_u64 v[34:35], v[2:3], 0, v[32:33]
	v_ashrrev_i32_e32 v1, 31, v0
	v_mul_f32_e32 v50, s16, v110
	global_store_short_d16_hi v[22:23], v42, off
	global_store_short_d16_hi v[26:27], v43, off
	global_store_short_d16_hi v[30:31], v44, off
	global_store_short_d16_hi v[34:35], v45, off
	global_store_short_d16_hi v[22:23], v50, off offset:32
	global_store_short_d16_hi v[26:27], v51, off offset:32
	global_store_short_d16_hi v[30:31], v52, off offset:32
	global_store_short_d16_hi v[34:35], v53, off offset:32
	v_lshlrev_b64 v[44:45], 1, v[0:1]
	v_add_u32_e32 v0, s20, v0
	v_ashrrev_i32_e32 v1, 31, v0
	v_mul_f32_e32 v98, s16, v149
	v_mul_f32_e32 v149, s16, v48
	v_mul_f32_e32 v150, s16, v49
	v_lshlrev_b64 v[48:49], 1, v[0:1]
	v_add_u32_e32 v0, s0, v0
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshlrev_b64 v[52:53], 1, v[0:1]
	v_add_u32_e32 v0, s20, v0
	v_ashrrev_i32_e32 v1, 31, v0
	v_mul_f32_e32 v67, s16, v95
	v_mul_f32_e32 v95, s16, v146
	v_mul_f32_e32 v110, s16, v145
	v_mul_f32_e32 v145, s16, v56
	v_mul_f32_e32 v146, s16, v57
	v_lshlrev_b64 v[56:57], 1, v[0:1]
	v_add_u32_e32 v0, s20, v0
	v_ashrrev_i32_e32 v1, 31, v0
	v_mul_f32_e32 v99, s16, v142
	v_mul_f32_e32 v141, s16, v60
	v_mul_f32_e32 v142, s16, v61
	v_lshlrev_b64 v[60:61], 1, v[0:1]
	v_add_u32_e32 v0, s20, v0
	v_mul_f32_e32 v68, s16, v96
	v_mul_f32_e32 v69, s16, v97
	v_mul_f32_e32 v62, s16, v62
	v_mul_f32_e32 v63, s16, v63
	v_mul_f32_e32 v96, s16, v147
	v_mul_f32_e32 v97, s16, v148
	v_mul_f32_e32 v100, s16, v143
	v_mul_f32_e32 v101, s16, v144
	v_mul_f32_e32 v139, s16, v58
	v_mul_f32_e32 v140, s16, v59
	v_mul_f32_e32 v143, s16, v54
	v_mul_f32_e32 v144, s16, v55
	v_mul_f32_e32 v147, s16, v46
	v_mul_f32_e32 v148, s16, v47
	v_mul_f32_e32 v151, s16, v38
	v_mul_f32_e32 v152, s16, v39
	v_lshl_add_u64 v[38:39], v[2:3], 0, v[36:37]
	v_lshl_add_u64 v[42:43], v[2:3], 0, v[40:41]
	v_lshl_add_u64 v[46:47], v[2:3], 0, v[44:45]
	v_lshl_add_u64 v[50:51], v[2:3], 0, v[48:49]
	v_lshl_add_u64 v[54:55], v[2:3], 0, v[52:53]
	v_lshl_add_u64 v[58:59], v[2:3], 0, v[56:57]
	v_ashrrev_i32_e32 v1, 31, v0
	v_mul_f32_e32 v78, s16, v78
	v_mul_f32_e32 v79, s16, v79
	v_mul_f32_e32 v80, s16, v80
	v_mul_f32_e32 v81, s16, v81
	v_mul_f32_e32 v64, s16, v64
	global_store_short_d16_hi v[38:39], v66, off
	global_store_short_d16_hi v[42:43], v67, off
	global_store_short_d16_hi v[46:47], v68, off
	global_store_short_d16_hi v[50:51], v69, off
	global_store_short_d16_hi v[38:39], v78, off offset:32
	global_store_short_d16_hi v[42:43], v79, off offset:32
	global_store_short_d16_hi v[46:47], v80, off offset:32
	global_store_short_d16_hi v[50:51], v81, off offset:32
	global_store_short_d16_hi v[54:55], v62, off
	global_store_short_d16_hi v[58:59], v63, off
	v_lshl_add_u64 v[62:63], v[2:3], 0, v[60:61]
	v_lshlrev_b64 v[0:1], 1, v[0:1]
	s_mov_b64 s[0:1], 0x100
	v_mul_f32_e32 v82, s16, v65
	global_store_short_d16_hi v[62:63], v64, off
	v_lshl_add_u64 v[64:65], v[2:3], 0, v[0:1]
	v_lshl_add_u64 v[2:3], v[2:3], 0, s[0:1]
	global_store_short_d16_hi v[64:65], v82, off
	global_store_short_d16_hi v[54:55], v83, off offset:32
	global_store_short_d16_hi v[58:59], v84, off offset:32
	global_store_short_d16_hi v[62:63], v85, off offset:32
	global_store_short_d16_hi v[64:65], v94, off offset:32
	global_store_short_d16_hi v[6:7], v95, off offset:256
	global_store_short_d16_hi v[10:11], v96, off offset:256
	global_store_short_d16_hi v[14:15], v97, off offset:256
	global_store_short_d16_hi v[18:19], v98, off offset:256
	v_lshl_add_u64 v[6:7], v[2:3], 0, v[4:5]
	global_store_short_d16_hi v[6:7], v99, off offset:32
	v_lshl_add_u64 v[6:7], v[2:3], 0, v[8:9]
	global_store_short_d16_hi v[6:7], v100, off offset:32
	v_lshl_add_u64 v[6:7], v[2:3], 0, v[12:13]
	global_store_short_d16_hi v[6:7], v101, off offset:32
	v_lshl_add_u64 v[6:7], v[2:3], 0, v[16:17]
	global_store_short_d16_hi v[6:7], v110, off offset:32
	global_store_short_d16_hi v[22:23], v111, off offset:256
	global_store_short_d16_hi v[26:27], v112, off offset:256
	global_store_short_d16_hi v[30:31], v113, off offset:256
	global_store_short_d16_hi v[34:35], v114, off offset:256
	v_lshl_add_u64 v[6:7], v[2:3], 0, v[20:21]
	global_store_short_d16_hi v[6:7], v115, off offset:32
	v_lshl_add_u64 v[6:7], v[2:3], 0, v[24:25]
	global_store_short_d16_hi v[6:7], v116, off offset:32
	v_lshl_add_u64 v[6:7], v[2:3], 0, v[28:29]
	global_store_short_d16_hi v[6:7], v117, off offset:32
	v_lshl_add_u64 v[6:7], v[2:3], 0, v[32:33]
	v_mul_f32_e32 v106, s16, v106
	v_mul_f32_e32 v107, s16, v107
	v_mul_f32_e32 v108, s16, v108
	v_mul_f32_e32 v109, s16, v109
	v_mul_f32_e32 v90, s16, v90
	global_store_short_d16_hi v[6:7], v122, off offset:32
	global_store_short_d16_hi v[38:39], v106, off offset:256
	global_store_short_d16_hi v[42:43], v107, off offset:256
	global_store_short_d16_hi v[46:47], v108, off offset:256
	global_store_short_d16_hi v[50:51], v109, off offset:256
	v_lshl_add_u64 v[6:7], v[2:3], 0, v[36:37]
	v_mul_f32_e32 v91, s16, v91
	global_store_short_d16_hi v[6:7], v90, off offset:32
	v_lshl_add_u64 v[6:7], v[2:3], 0, v[40:41]
	v_mul_f32_e32 v92, s16, v92
	global_store_short_d16_hi v[6:7], v91, off offset:32
	v_lshl_add_u64 v[6:7], v[2:3], 0, v[44:45]
	v_mul_f32_e32 v93, s16, v93
	global_store_short_d16_hi v[6:7], v92, off offset:32
	v_lshl_add_u64 v[6:7], v[2:3], 0, v[48:49]
	v_mul_f32_e32 v74, s16, v74
	v_mul_f32_e32 v75, s16, v75
	v_mul_f32_e32 v76, s16, v76
	v_mul_f32_e32 v77, s16, v77
	global_store_short_d16_hi v[6:7], v93, off offset:32
	global_store_short_d16_hi v[54:55], v74, off offset:256
	global_store_short_d16_hi v[58:59], v75, off offset:256
	global_store_short_d16_hi v[62:63], v76, off offset:256
	global_store_short_d16_hi v[64:65], v77, off offset:256
	v_lshl_add_u64 v[6:7], v[2:3], 0, v[52:53]
	global_store_short_d16_hi v[6:7], v123, off offset:32
	v_lshl_add_u64 v[6:7], v[2:3], 0, v[56:57]
	global_store_short_d16_hi v[6:7], v124, off offset:32
	v_lshl_add_u64 v[6:7], v[2:3], 0, v[60:61]
	v_lshl_add_u64 v[2:3], v[2:3], 0, v[0:1]
	global_store_short_d16_hi v[6:7], v125, off offset:32
	global_store_short_d16_hi v[2:3], v126, off offset:32
	v_or_b32_e32 v2, 2, v177
	v_mul_lo_u32 v2, v2, s20
	v_lshl_add_u32 v2, v2, 6, v167
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
	v_mul_f32_e32 v102, s16, v102
	v_mul_f32_e32 v103, s16, v103
	v_mul_f32_e32 v104, s16, v104
	v_mul_f32_e32 v105, s16, v105
	v_mul_f32_e32 v70, s16, v70
	v_mul_f32_e32 v71, s16, v71
	v_mul_f32_e32 v72, s16, v72
	v_mul_f32_e32 v73, s16, v73
	v_lshl_add_u64 v[4:5], v[2:3], 0, v[4:5]
	v_mul_f32_e32 v118, s16, v118
	v_mul_f32_e32 v119, s16, v119
	v_mul_f32_e32 v120, s16, v120
	v_mul_f32_e32 v121, s16, v121
	v_mul_f32_e32 v86, s16, v86
	v_mul_f32_e32 v87, s16, v87
	v_mul_f32_e32 v88, s16, v88
	v_mul_f32_e32 v89, s16, v89
	global_store_short_d16_hi v[6:7], v127, off
	global_store_short_d16_hi v[10:11], v128, off
	global_store_short_d16_hi v[14:15], v129, off
	global_store_short_d16_hi v[18:19], v130, off
	global_store_short_d16_hi v[6:7], v131, off offset:32
	global_store_short_d16_hi v[10:11], v132, off offset:32
	global_store_short_d16_hi v[14:15], v133, off offset:32
	global_store_short_d16_hi v[18:19], v134, off offset:32
	global_store_short_d16_hi v[22:23], v135, off
	global_store_short_d16_hi v[26:27], v136, off
	global_store_short_d16_hi v[30:31], v137, off
	global_store_short_d16_hi v[34:35], v138, off
	global_store_short_d16_hi v[22:23], v118, off offset:32
	global_store_short_d16_hi v[26:27], v119, off offset:32
	global_store_short_d16_hi v[30:31], v120, off offset:32
	global_store_short_d16_hi v[34:35], v121, off offset:32
	global_store_short_d16_hi v[38:39], v102, off
	global_store_short_d16_hi v[42:43], v103, off
	global_store_short_d16_hi v[46:47], v104, off
	global_store_short_d16_hi v[50:51], v105, off
	global_store_short_d16_hi v[38:39], v86, off offset:32
	global_store_short_d16_hi v[42:43], v87, off offset:32
	global_store_short_d16_hi v[46:47], v88, off offset:32
	global_store_short_d16_hi v[50:51], v89, off offset:32
	global_store_short_d16_hi v[54:55], v70, off
	global_store_short_d16_hi v[58:59], v71, off
	global_store_short_d16_hi v[62:63], v72, off
	global_store_short_d16_hi v[64:65], v73, off
	global_store_short_d16_hi v[54:55], v139, off offset:32
	global_store_short_d16_hi v[58:59], v140, off offset:32
	global_store_short_d16_hi v[62:63], v141, off offset:32
	global_store_short_d16_hi v[64:65], v142, off offset:32
	global_store_short_d16_hi v[6:7], v143, off offset:256
	global_store_short_d16_hi v[10:11], v144, off offset:256
	global_store_short_d16_hi v[14:15], v145, off offset:256
	global_store_short_d16_hi v[18:19], v146, off offset:256
	global_store_short_d16_hi v[4:5], v147, off offset:32
	v_lshl_add_u64 v[4:5], v[2:3], 0, v[8:9]
	global_store_short_d16_hi v[4:5], v148, off offset:32
	v_lshl_add_u64 v[4:5], v[2:3], 0, v[12:13]
	global_store_short_d16_hi v[4:5], v149, off offset:32
	v_lshl_add_u64 v[4:5], v[2:3], 0, v[16:17]
	global_store_short_d16_hi v[4:5], v150, off offset:32
	global_store_short_d16_hi v[22:23], v151, off offset:256
	global_store_short_d16_hi v[26:27], v152, off offset:256
	global_store_short_d16_hi v[30:31], v153, off offset:256
	global_store_short_d16_hi v[34:35], v154, off offset:256
	v_lshl_add_u64 v[4:5], v[2:3], 0, v[20:21]
	global_store_short_d16_hi v[4:5], v155, off offset:32
	v_lshl_add_u64 v[4:5], v[2:3], 0, v[24:25]
	global_store_short_d16_hi v[4:5], v156, off offset:32
	v_lshl_add_u64 v[4:5], v[2:3], 0, v[28:29]
	global_store_short_d16_hi v[4:5], v157, off offset:32
	v_lshl_add_u64 v[4:5], v[2:3], 0, v[32:33]
	global_store_short_d16_hi v[4:5], v158, off offset:32
	global_store_short_d16_hi v[38:39], v159, off offset:256
	global_store_short_d16_hi v[42:43], v160, off offset:256
	global_store_short_d16_hi v[46:47], v161, off offset:256
	global_store_short_d16_hi v[50:51], v162, off offset:256
	v_lshl_add_u64 v[4:5], v[2:3], 0, v[36:37]
	global_store_short_d16_hi v[4:5], v163, off offset:32
	v_lshl_add_u64 v[4:5], v[2:3], 0, v[40:41]
	global_store_short_d16_hi v[4:5], v164, off offset:32
	v_lshl_add_u64 v[4:5], v[2:3], 0, v[44:45]
	global_store_short_d16_hi v[4:5], v165, off offset:32
	v_lshl_add_u64 v[4:5], v[2:3], 0, v[48:49]
	global_store_short_d16_hi v[4:5], v168, off offset:32
	global_store_short_d16_hi v[54:55], v169, off offset:256
	global_store_short_d16_hi v[58:59], v170, off offset:256
	global_store_short_d16_hi v[62:63], v171, off offset:256
	global_store_short_d16_hi v[64:65], v172, off offset:256
	v_lshl_add_u64 v[4:5], v[2:3], 0, v[52:53]
	global_store_short_d16_hi v[4:5], v173, off offset:32
	v_lshl_add_u64 v[4:5], v[2:3], 0, v[56:57]
	global_store_short_d16_hi v[4:5], v174, off offset:32
	v_lshl_add_u64 v[4:5], v[2:3], 0, v[60:61]
	v_lshl_add_u64 v[0:1], v[2:3], 0, v[0:1]
	global_store_short_d16_hi v[4:5], v175, off offset:32
	global_store_short_d16_hi v[0:1], v176, off offset:32
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z29crr_exact_8wave_scaled_kernelILb1ELi2EEv14layout_globals
		.amdhsa_group_segment_fixed_size 139264
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
		.amdhsa_next_free_vgpr 227
		.amdhsa_next_free_sgpr 96
		.amdhsa_accum_offset 228
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
	.section	.text._Z29crr_exact_8wave_scaled_kernelILb1ELi2EEv14layout_globals,"axG",@progbits,_Z29crr_exact_8wave_scaled_kernelILb1ELi2EEv14layout_globals,comdat
.Lfunc_end4:
	.size	_Z29crr_exact_8wave_scaled_kernelILb1ELi2EEv14layout_globals, .Lfunc_end4-_Z29crr_exact_8wave_scaled_kernelILb1ELi2EEv14layout_globals
                                        ; -- End function
	.set _Z29crr_exact_8wave_scaled_kernelILb1ELi2EEv14layout_globals.num_vgpr, 227
	.set _Z29crr_exact_8wave_scaled_kernelILb1ELi2EEv14layout_globals.num_agpr, 0
	.set _Z29crr_exact_8wave_scaled_kernelILb1ELi2EEv14layout_globals.numbered_sgpr, 44
	.set _Z29crr_exact_8wave_scaled_kernelILb1ELi2EEv14layout_globals.private_seg_size, 0
	.set _Z29crr_exact_8wave_scaled_kernelILb1ELi2EEv14layout_globals.uses_vcc, 1
	.set _Z29crr_exact_8wave_scaled_kernelILb1ELi2EEv14layout_globals.uses_flat_scratch, 0
	.set _Z29crr_exact_8wave_scaled_kernelILb1ELi2EEv14layout_globals.has_dyn_sized_stack, 0
	.set _Z29crr_exact_8wave_scaled_kernelILb1ELi2EEv14layout_globals.has_recursion, 0
	.set _Z29crr_exact_8wave_scaled_kernelILb1ELi2EEv14layout_globals.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 10172
; TotalNumSgprs: 50
; NumVgprs: 227
; NumAgprs: 0
; TotalNumVgprs: 227
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 139264 bytes/workgroup (compile time only)
; SGPRBlocks: 12
; VGPRBlocks: 28
; NumSGPRsForWavesPerEU: 102
; NumVGPRsForWavesPerEU: 227
; AccumOffset: 228
; Occupancy: 2
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 56
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.section	.text._Z11gemm_kernelIL6Layout1ELb0EEv14layout_globals,"axG",@progbits,_Z11gemm_kernelIL6Layout1ELb0EEv14layout_globals,comdat
	.protected	_Z11gemm_kernelIL6Layout1ELb0EEv14layout_globals ; -- Begin function _Z11gemm_kernelIL6Layout1ELb0EEv14layout_globals
	.globl	_Z11gemm_kernelIL6Layout1ELb0EEv14layout_globals
	.p2align	8
	.type	_Z11gemm_kernelIL6Layout1ELb0EEv14layout_globals,@function
