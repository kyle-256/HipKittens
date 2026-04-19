_Z22rrr_exact_8wave_kernel23rrr_exact_8wave_globals: ; @_Z22rrr_exact_8wave_kernel23rrr_exact_8wave_globals
; %bb.0:
	s_load_dwordx2 s[10:11], s[0:1], 0x0
	s_load_dwordx2 s[4:5], s[0:1], 0x10
	s_load_dwordx2 s[8:9], s[0:1], 0x20
	s_ashr_i32 s0, s2, 31
	s_lshr_b32 s0, s0, 28
	s_add_i32 s0, s2, s0
	v_lshlrev_b32_e32 v1, 4, v0
	v_lshlrev_b32_e32 v3, 9, v0
	s_ashr_i32 s12, s0, 4
	s_and_b32 s0, s0, 0xfffff0
	v_xor_b32_e32 v2, v1, v0
	v_and_b32_e32 v3, 0x3f000, v3
	s_movk_i32 s1, 0x70
	s_sub_i32 s0, s2, s0
	v_and_or_b32 v179, v2, s1, v3
	v_lshlrev_b32_e32 v2, 1, v0
	v_xor_b32_e32 v2, v2, v1
	s_lshl_b32 s13, s0, 8
	v_and_or_b32 v182, v2, s1, v3
	s_ashr_i32 s0, s13, 31
	v_and_b32_e32 v168, 0x1c00, v1
	v_and_b32_e32 v2, 0x180, v0
	s_waitcnt lgkmcnt(0)
	s_add_u32 s4, s4, s13
	v_or_b32_e32 v165, v168, v2
	v_or_b32_e32 v3, 0x2200, v2
	s_addc_u32 s5, s5, s0
	s_mov_b32 s3, 0x110000
	s_mov_b32 s2, 0x80000
	v_readfirstlane_b32 s0, v165
	v_or_b32_e32 v166, v168, v3
	s_mov_b32 s6, s2
	s_mov_b32 s7, s3
	s_mov_b32 m0, s0
	v_readfirstlane_b32 s0, v166
	s_lshl_b32 s14, s12, 20
	buffer_load_dwordx4 v182, s[4:7], 0 offen lds
	s_mov_b32 m0, s0
	s_ashr_i32 s0, s14, 31
	s_add_u32 s15, s10, s14
	s_mov_b64 s[22:23], s[6:7]
	v_add_u32_e32 v180, 0x11000, v168
	v_or_b32_e32 v181, 0x40000, v182
	s_addc_u32 s16, s11, s0
	s_mov_b64 s[20:21], s[4:5]
	v_readfirstlane_b32 s0, v180
	v_add_u32_e32 v1, 0x2000, v180
	buffer_load_dwordx4 v181, s[4:7], 0 offen lds
	s_mov_b32 s20, s15
	s_mov_b32 s21, s16
	s_mov_b32 m0, s0
	v_readfirstlane_b32 s0, v1
	v_or_b32_e32 v178, 0x40000, v179
	buffer_load_dwordx4 v179, s[20:23], 0 offen lds
	s_mov_b32 m0, s0
	v_add_u32_e32 v1, 0x4400, v168
	buffer_load_dwordx4 v178, s[20:23], 0 offen lds
	s_add_u32 s17, s4, 0x80
	s_mov_b64 s[22:23], s[6:7]
	v_or_b32_e32 v4, v1, v2
	s_addc_u32 s18, s5, 0
	s_mov_b64 s[20:21], s[4:5]
	v_readfirstlane_b32 s0, v4
	v_add_u32_e32 v1, v1, v3
	s_mov_b32 s20, s17
	s_mov_b32 s21, s18
	s_mov_b32 m0, s0
	v_readfirstlane_b32 s0, v1
	buffer_load_dwordx4 v182, s[20:23], 0 offen lds
	s_mov_b32 m0, s0
	s_or_b32 s0, s14, 0x80000
	buffer_load_dwordx4 v181, s[20:23], 0 offen lds
	s_ashr_i32 s1, s0, 31
	s_mov_b64 s[22:23], s[6:7]
	s_add_u32 s0, s10, s0
	s_mov_b64 s[20:21], s[4:5]
	v_or_b32_e32 v146, 0x4000, v180
	s_addc_u32 s1, s11, s1
	s_mov_b32 s20, s0
	v_readfirstlane_b32 s0, v146
	v_add_u32_e32 v1, 0x6000, v180
	s_mov_b32 s21, s1
	s_mov_b32 m0, s0
	v_readfirstlane_b32 s0, v1
	buffer_load_dwordx4 v179, s[20:23], 0 offen lds
	s_mov_b32 m0, s0
	v_lshrrev_b32_e32 v1, 8, v0
	buffer_load_dwordx4 v178, s[20:23], 0 offen lds
	v_cmp_eq_u32_e32 vcc, 1, v1
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB1_2
; %bb.1:
	s_barrier
.LBB1_2:
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v4, 0x8800, v168
	v_add_u32_e32 v5, v4, v2
	s_add_u32 s0, s4, 0x80000
	v_readfirstlane_b32 s6, v5
	v_add_u32_e32 v4, v4, v3
	s_addc_u32 s1, s5, 0
	s_mov_b32 m0, s6
	v_readfirstlane_b32 s6, v4
	v_or_b32_e32 v4, 0x8000, v180
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	buffer_load_dwordx4 v182, s[0:3], 0 offen lds
	s_mov_b32 m0, s6
	v_readfirstlane_b32 s6, v4
	v_add_u32_e32 v4, 0xa000, v180
	buffer_load_dwordx4 v181, s[0:3], 0 offen lds
	s_add_u32 s0, s15, 0x80
	s_mov_b32 m0, s6
	v_readfirstlane_b32 s6, v4
	v_add_u32_e32 v4, 0xcc00, v168
	s_addc_u32 s1, s16, 0
	v_add_u32_e32 v2, v4, v2
	buffer_load_dwordx4 v179, s[0:3], 0 offen lds
	s_mov_b32 m0, s6
	v_readfirstlane_b32 s6, v2
	v_add_u32_e32 v2, v4, v3
	buffer_load_dwordx4 v178, s[0:3], 0 offen lds
	s_mov_b32 m0, s6
	v_readfirstlane_b32 s6, v2
	v_lshrrev_b32_e32 v2, 1, v0
	v_and_b32_e32 v186, 0x60, v2
	v_bfe_u32 v2, v0, 1, 3
	v_bfe_u32 v5, v0, 4, 2
	v_lshlrev_b32_e32 v3, 3, v0
	v_lshlrev_b32_e32 v6, 11, v5
	v_add_u32_e32 v7, v5, v2
	v_or_b32_e32 v5, 4, v5
	v_and_b32_e32 v4, 8, v3
	v_lshlrev_b32_e32 v187, 4, v2
	v_lshl_or_b32 v190, v7, 7, v6
	v_lshlrev_b32_e32 v7, 11, v5
	v_add_u32_e32 v2, v5, v2
	v_or_b32_e32 v188, v186, v4
	v_bitop3_b32 v4, v186, v187, v4 bitop3:0x36
	v_lshl_or_b32 v189, v2, 7, v7
	s_add_u32 s0, s17, 0x80000
	v_or_b32_e32 v192, v190, v4
	v_or_b32_e32 v185, v189, v4
	v_lshlrev_b32_e32 v4, 7, v0
	s_addc_u32 s1, s18, 0
	v_and_b32_e32 v2, 48, v0
	v_and_b32_e32 v4, 0x780, v4
	buffer_load_dwordx4 v182, s[0:3], 0 offen lds
	s_mov_b32 m0, s6
	v_or_b32_e32 v5, v4, v2
	v_and_b32_e32 v3, 0x70, v3
	buffer_load_dwordx4 v181, s[0:3], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(6)
	;;#ASMEND
	s_barrier
	v_bitop3_b32 v6, v188, v187, 16 bitop3:0x36
	v_bitop3_b32 v171, v4, v3, v2 bitop3:0x36
	v_bitop3_b32 v169, v5, v3, 64 bitop3:0x36
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v192 offset:0
ds_read_b64_tr_b8 v[4:5], v192 offset:1024

	;;#ASMEND
	v_or_b32_e32 v184, v190, v6
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v184 offset:0
ds_read_b64_tr_b8 v[12:13], v184 offset:1024

	;;#ASMEND
	v_or_b32_e32 v183, v189, v6
	v_lshlrev_b32_e32 v170, 13, v1
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v185 offset:0
ds_read_b64_tr_b8 v[8:9], v185 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v183 offset:0
ds_read_b64_tr_b8 v[16:17], v183 offset:1024

	;;#ASMEND
	v_or_b32_e32 v18, 0x11000, v170
	v_or_b32_e32 v193, v171, v18
	;;#ASMSTART
	ds_read_b128 v[22:25], v193 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[30:33], v193 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[38:41], v193 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[46:49], v193 offset:0x1800

	;;#ASMEND
	v_or_b32_e32 v194, v169, v18
	;;#ASMSTART
	ds_read_b128 v[26:29], v194 offset:0

	;;#ASMEND
	s_or_b32 s0, s14, 0x80080
	;;#ASMSTART
	ds_read_b128 v[34:37], v194 offset:0x800

	;;#ASMEND
	s_ashr_i32 s1, s0, 31
	v_add_u32_e32 v195, 0x1d000, v168
	;;#ASMSTART
	ds_read_b128 v[42:45], v194 offset:0x1000

	;;#ASMEND
	s_add_u32 s0, s10, s0
	v_readfirstlane_b32 s6, v195
	v_add_u32_e32 v167, 0x1f000, v168
	;;#ASMSTART
	ds_read_b128 v[50:53], v194 offset:0x1800

	;;#ASMEND
	s_addc_u32 s1, s11, s1
	s_mov_b32 m0, s6
	v_readfirstlane_b32 s6, v167
	buffer_load_dwordx4 v179, s[0:3], 0 offen lds
	s_mov_b32 m0, s6
	v_or_b32_e32 v191, 16, v188
	buffer_load_dwordx4 v178, s[0:3], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[114:117], v[22:29], v[2:9], 0
	v_mfma_f32_16x16x128_f8f6f4 v[118:121], v[22:29], v[10:17], 0
	v_mfma_f32_16x16x128_f8f6f4 v[122:125], v[30:37], v[2:9], 0
	v_mfma_f32_16x16x128_f8f6f4 v[126:129], v[30:37], v[10:17], 0
	v_mfma_f32_16x16x128_f8f6f4 v[130:133], v[38:45], v[2:9], 0
	v_mfma_f32_16x16x128_f8f6f4 v[134:137], v[38:45], v[10:17], 0
	v_mfma_f32_16x16x128_f8f6f4 v[138:141], v[46:53], v[2:9], 0
	v_mfma_f32_16x16x128_f8f6f4 v[142:145], v[46:53], v[10:17], 0
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	v_add_u32_e32 v147, 0x4400, v192
	;;#ASMSTART
	ds_read_b64_tr_b8 v[106:107], v147 offset:0
ds_read_b64_tr_b8 v[108:109], v147 offset:1024

	;;#ASMEND
	v_add_u32_e32 v148, 0x4400, v184
	;;#ASMSTART
	ds_read_b64_tr_b8 v[154:155], v148 offset:0
ds_read_b64_tr_b8 v[156:157], v148 offset:1024

	;;#ASMEND
	v_add_u32_e32 v149, 0x4400, v185
	;;#ASMSTART
	ds_read_b64_tr_b8 v[110:111], v149 offset:0
ds_read_b64_tr_b8 v[112:113], v149 offset:1024

	;;#ASMEND
	s_add_u32 s0, s4, 0x100000
	v_readfirstlane_b32 s6, v165
	v_add_u32_e32 v150, 0x4400, v183
	;;#ASMSTART
	ds_read_b64_tr_b8 v[158:159], v150 offset:0
ds_read_b64_tr_b8 v[160:161], v150 offset:1024

	;;#ASMEND
	s_addc_u32 s1, s5, 0
	s_mov_b32 m0, s6
	v_readfirstlane_b32 s6, v166
	buffer_load_dwordx4 v182, s[0:3], 0 offen lds
	s_mov_b32 m0, s6
	s_nop 0
	buffer_load_dwordx4 v181, s[0:3], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[18:21], v[22:29], v[106:113], 0
	v_mfma_f32_16x16x128_f8f6f4 v[22:25], v[22:29], v[154:161], 0
	v_mfma_f32_16x16x128_f8f6f4 v[26:29], v[30:37], v[106:113], 0
	v_mfma_f32_16x16x128_f8f6f4 v[30:33], v[30:37], v[154:161], 0
	v_mfma_f32_16x16x128_f8f6f4 v[34:37], v[38:45], v[106:113], 0
	v_mfma_f32_16x16x128_f8f6f4 v[38:41], v[38:45], v[154:161], 0
	v_mfma_f32_16x16x128_f8f6f4 v[42:45], v[46:53], v[106:113], 0
	v_mfma_f32_16x16x128_f8f6f4 v[46:49], v[46:53], v[154:161], 0
	s_setprio 0
	v_or_b32_e32 v50, 0x15000, v170
	s_barrier
	v_or_b32_e32 v196, v171, v50
	;;#ASMSTART
	ds_read_b128 v[86:89], v196 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[94:97], v196 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[198:201], v196 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[206:209], v196 offset:0x1800

	;;#ASMEND
	v_or_b32_e32 v197, v169, v50
	;;#ASMSTART
	ds_read_b128 v[90:93], v197 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[98:101], v197 offset:0x800

	;;#ASMEND
	v_add_u32_e32 v151, 0x4400, v165
	;;#ASMSTART
	ds_read_b128 v[202:205], v197 offset:0x1000

	;;#ASMEND
	s_add_u32 s0, s4, 0x100080
	v_readfirstlane_b32 s17, v151
	v_add_u32_e32 v152, 0x4400, v166
	;;#ASMSTART
	ds_read_b128 v[210:213], v197 offset:0x1800

	;;#ASMEND
	s_addc_u32 s1, s5, 0
	s_mov_b32 m0, s17
	v_readfirstlane_b32 s18, v152
	buffer_load_dwordx4 v182, s[0:3], 0 offen lds
	s_mov_b32 m0, s18
	s_nop 0
	buffer_load_dwordx4 v181, s[0:3], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[50:53], v[86:93], v[2:9], 0
	v_mfma_f32_16x16x128_f8f6f4 v[54:57], v[86:93], v[10:17], 0
	v_mfma_f32_16x16x128_f8f6f4 v[58:61], v[94:101], v[2:9], 0
	v_mfma_f32_16x16x128_f8f6f4 v[62:65], v[94:101], v[10:17], 0
	v_mfma_f32_16x16x128_f8f6f4 v[66:69], v[198:205], v[2:9], 0
	v_mfma_f32_16x16x128_f8f6f4 v[70:73], v[198:205], v[10:17], 0
	v_mfma_f32_16x16x128_f8f6f4 v[74:77], v[206:213], v[2:9], 0
	v_mfma_f32_16x16x128_f8f6f4 v[78:81], v[206:213], v[10:17], 0
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_u32 s0, s15, 0x100
	v_readfirstlane_b32 s19, v180
	v_add_u32_e32 v153, 0x13000, v168
	s_addc_u32 s1, s16, 0
	s_mov_b32 m0, s19
	v_readfirstlane_b32 s20, v153
	buffer_load_dwordx4 v179, s[0:3], 0 offen lds
	s_mov_b32 m0, s20
	s_mov_b64 s[6:7], 0x100
	buffer_load_dwordx4 v178, s[0:3], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[82:85], v[86:93], v[106:113], 0
	v_mfma_f32_16x16x128_f8f6f4 v[86:89], v[86:93], v[154:161], 0
	v_mfma_f32_16x16x128_f8f6f4 v[90:93], v[94:101], v[106:113], 0
	v_mfma_f32_16x16x128_f8f6f4 v[94:97], v[94:101], v[154:161], 0
	v_mfma_f32_16x16x128_f8f6f4 v[98:101], v[198:205], v[106:113], 0
	v_mfma_f32_16x16x128_f8f6f4 v[102:105], v[198:205], v[154:161], 0
	v_mfma_f32_16x16x128_f8f6f4 v[106:109], v[206:213], v[106:113], 0
	v_mfma_f32_16x16x128_f8f6f4 v[110:113], v[206:213], v[154:161], 0
	s_setprio 0
	s_barrier
	v_add_u32_e32 v154, 0x8800, v192
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v154 offset:0
ds_read_b64_tr_b8 v[4:5], v154 offset:1024

	;;#ASMEND
	v_add_u32_e32 v155, 0x8800, v184
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v155 offset:0
ds_read_b64_tr_b8 v[12:13], v155 offset:1024

	;;#ASMEND
	v_add_u32_e32 v156, 0x8800, v185
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v156 offset:0
ds_read_b64_tr_b8 v[8:9], v156 offset:1024

	;;#ASMEND
	v_add_u32_e32 v157, 0x8800, v183
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v157 offset:0
ds_read_b64_tr_b8 v[16:17], v157 offset:1024

	;;#ASMEND
	v_or_b32_e32 v158, 0x19000, v170
	v_or_b32_e32 v198, v171, v158
	;;#ASMSTART
	ds_read_b128 v[200:203], v198 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[208:211], v198 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[216:219], v198 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[224:227], v198 offset:0x1800

	;;#ASMEND
	v_or_b32_e32 v199, v169, v158
	;;#ASMSTART
	ds_read_b128 v[204:207], v199 offset:0

	;;#ASMEND
	s_or_b32 s0, s14, 0x80100
	;;#ASMSTART
	ds_read_b128 v[212:215], v199 offset:0x800

	;;#ASMEND
	s_ashr_i32 s1, s0, 31
	;;#ASMSTART
	ds_read_b128 v[220:223], v199 offset:0x1000

	;;#ASMEND
	s_add_u32 s0, s10, s0
	v_readfirstlane_b32 s21, v146
	v_add_u32_e32 v158, 0x17000, v168
	;;#ASMSTART
	ds_read_b128 v[228:231], v199 offset:0x1800

	;;#ASMEND
	s_addc_u32 s1, s11, s1
	s_mov_b32 m0, s21
	v_readfirstlane_b32 s22, v158
	buffer_load_dwordx4 v179, s[0:3], 0 offen lds
	s_mov_b32 m0, s22
	s_nop 0
	buffer_load_dwordx4 v178, s[0:3], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[114:117], v[200:207], v[2:9], v[114:117]
	v_mfma_f32_16x16x128_f8f6f4 v[118:121], v[200:207], v[10:17], v[118:121]
	v_mfma_f32_16x16x128_f8f6f4 v[122:125], v[208:215], v[2:9], v[122:125]
	v_mfma_f32_16x16x128_f8f6f4 v[126:129], v[208:215], v[10:17], v[126:129]
	v_mfma_f32_16x16x128_f8f6f4 v[130:133], v[216:223], v[2:9], v[130:133]
	v_mfma_f32_16x16x128_f8f6f4 v[134:137], v[216:223], v[10:17], v[134:137]
	v_mfma_f32_16x16x128_f8f6f4 v[138:141], v[224:231], v[2:9], v[138:141]
	v_mfma_f32_16x16x128_f8f6f4 v[142:145], v[224:231], v[10:17], v[142:145]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	v_add_u32_e32 v159, 0xcc00, v192
	;;#ASMSTART
	ds_read_b64_tr_b8 v[232:233], v159 offset:0
ds_read_b64_tr_b8 v[234:235], v159 offset:1024

	;;#ASMEND
	v_add_u32_e32 v160, 0xcc00, v184
	;;#ASMSTART
	ds_read_b64_tr_b8 v[240:241], v160 offset:0
ds_read_b64_tr_b8 v[242:243], v160 offset:1024

	;;#ASMEND
	v_add_u32_e32 v163, 0x8800, v165
	v_add_u32_e32 v161, 0xcc00, v185
	;;#ASMSTART
	ds_read_b64_tr_b8 v[236:237], v161 offset:0
ds_read_b64_tr_b8 v[238:239], v161 offset:1024

	;;#ASMEND
	s_add_u32 s0, s4, 0x180000
	v_readfirstlane_b32 s23, v163
	v_add_u32_e32 v164, 0x8800, v166
	v_add_u32_e32 v162, 0xcc00, v183
	;;#ASMSTART
	ds_read_b64_tr_b8 v[244:245], v162 offset:0
ds_read_b64_tr_b8 v[246:247], v162 offset:1024

	;;#ASMEND
	s_addc_u32 s1, s5, 0
	s_mov_b32 m0, s23
	v_readfirstlane_b32 s24, v164
	buffer_load_dwordx4 v182, s[0:3], 0 offen lds
	s_mov_b32 m0, s24
	s_nop 0
	buffer_load_dwordx4 v181, s[0:3], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[18:21], v[200:207], v[232:239], v[18:21]
	v_mfma_f32_16x16x128_f8f6f4 v[22:25], v[200:207], v[240:247], v[22:25]
	v_mfma_f32_16x16x128_f8f6f4 v[26:29], v[208:215], v[232:239], v[26:29]
	v_mfma_f32_16x16x128_f8f6f4 v[30:33], v[208:215], v[240:247], v[30:33]
	v_mfma_f32_16x16x128_f8f6f4 v[34:37], v[216:223], v[232:239], v[34:37]
	v_mfma_f32_16x16x128_f8f6f4 v[38:41], v[216:223], v[240:247], v[38:41]
	v_mfma_f32_16x16x128_f8f6f4 v[42:45], v[224:231], v[232:239], v[42:45]
	v_mfma_f32_16x16x128_f8f6f4 v[46:49], v[224:231], v[240:247], v[46:49]
	s_setprio 0
	v_or_b32_e32 v174, 0x1d000, v170
	s_barrier
	v_or_b32_e32 v200, v171, v174
	;;#ASMSTART
	ds_read_b128 v[170:173], v200 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[206:209], v200 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[214:217], v200 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[222:225], v200 offset:0x1800

	;;#ASMEND
	v_or_b32_e32 v201, v169, v174
	;;#ASMSTART
	ds_read_b128 v[174:177], v201 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[210:213], v201 offset:0x800

	;;#ASMEND
	v_add_u32_e32 v202, 0xcc00, v165
	;;#ASMSTART
	ds_read_b128 v[218:221], v201 offset:0x1000

	;;#ASMEND
	s_add_u32 s0, s4, 0x180080
	v_readfirstlane_b32 s25, v202
	v_add_u32_e32 v203, 0xcc00, v166
	;;#ASMSTART
	ds_read_b128 v[226:229], v201 offset:0x1800

	;;#ASMEND
	s_addc_u32 s1, s5, 0
	s_mov_b32 m0, s25
	v_readfirstlane_b32 s26, v203
	buffer_load_dwordx4 v182, s[0:3], 0 offen lds
	s_mov_b32 m0, s26
	s_nop 0
	buffer_load_dwordx4 v181, s[0:3], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[50:53], v[170:177], v[2:9], v[50:53]
	v_mfma_f32_16x16x128_f8f6f4 v[54:57], v[170:177], v[10:17], v[54:57]
	v_mfma_f32_16x16x128_f8f6f4 v[58:61], v[206:213], v[2:9], v[58:61]
	v_mfma_f32_16x16x128_f8f6f4 v[62:65], v[206:213], v[10:17], v[62:65]
	v_mfma_f32_16x16x128_f8f6f4 v[66:69], v[214:221], v[2:9], v[66:69]
	v_mfma_f32_16x16x128_f8f6f4 v[70:73], v[214:221], v[10:17], v[70:73]
	v_mfma_f32_16x16x128_f8f6f4 v[74:77], v[222:229], v[2:9], v[74:77]
	v_mfma_f32_16x16x128_f8f6f4 v[78:81], v[222:229], v[10:17], v[78:81]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	v_add_u32_e32 v204, 0x19000, v168
	s_add_u32 s0, s15, 0x180
	v_readfirstlane_b32 s27, v204
	v_add_u32_e32 v205, 0x1b000, v168
	s_addc_u32 s1, s16, 0
	s_mov_b32 m0, s27
	v_readfirstlane_b32 s28, v205
	buffer_load_dwordx4 v179, s[0:3], 0 offen lds
	s_mov_b32 m0, s28
	s_nop 0
	buffer_load_dwordx4 v178, s[0:3], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[82:85], v[170:177], v[232:239], v[82:85]
	v_mfma_f32_16x16x128_f8f6f4 v[86:89], v[170:177], v[240:247], v[86:89]
	v_mfma_f32_16x16x128_f8f6f4 v[90:93], v[206:213], v[232:239], v[90:93]
	v_mfma_f32_16x16x128_f8f6f4 v[94:97], v[206:213], v[240:247], v[94:97]
	v_mfma_f32_16x16x128_f8f6f4 v[98:101], v[214:221], v[232:239], v[98:101]
	v_mfma_f32_16x16x128_f8f6f4 v[102:105], v[214:221], v[240:247], v[102:105]
	v_mfma_f32_16x16x128_f8f6f4 v[106:109], v[222:229], v[232:239], v[106:109]
	v_mfma_f32_16x16x128_f8f6f4 v[110:113], v[222:229], v[240:247], v[110:113]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v192 offset:0
ds_read_b64_tr_b8 v[4:5], v192 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v184 offset:0
ds_read_b64_tr_b8 v[12:13], v184 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v185 offset:0
ds_read_b64_tr_b8 v[8:9], v185 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v183 offset:0
ds_read_b64_tr_b8 v[16:17], v183 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[168:171], v193 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[206:209], v193 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[214:217], v193 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[222:225], v193 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[172:175], v194 offset:0

	;;#ASMEND
	s_or_b32 s0, s14, 0x80180
	;;#ASMSTART
	ds_read_b128 v[210:213], v194 offset:0x800

	;;#ASMEND
	s_ashr_i32 s1, s0, 31
	;;#ASMSTART
	ds_read_b128 v[218:221], v194 offset:0x1000

	;;#ASMEND
	s_add_u32 s0, s10, s0
	v_readfirstlane_b32 s29, v195
	;;#ASMSTART
	ds_read_b128 v[226:229], v194 offset:0x1800

	;;#ASMEND
	s_addc_u32 s1, s11, s1
	s_mov_b32 m0, s29
	v_readfirstlane_b32 s30, v167
	buffer_load_dwordx4 v179, s[0:3], 0 offen lds
	s_mov_b32 m0, s30
	s_nop 0
	buffer_load_dwordx4 v178, s[0:3], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[114:117], v[168:175], v[2:9], v[114:117]
	v_mfma_f32_16x16x128_f8f6f4 v[118:121], v[168:175], v[10:17], v[118:121]
	v_mfma_f32_16x16x128_f8f6f4 v[122:125], v[206:213], v[2:9], v[122:125]
	v_mfma_f32_16x16x128_f8f6f4 v[126:129], v[206:213], v[10:17], v[126:129]
	v_mfma_f32_16x16x128_f8f6f4 v[130:133], v[214:221], v[2:9], v[130:133]
	v_mfma_f32_16x16x128_f8f6f4 v[134:137], v[214:221], v[10:17], v[134:137]
	v_mfma_f32_16x16x128_f8f6f4 v[138:141], v[222:229], v[2:9], v[138:141]
	v_mfma_f32_16x16x128_f8f6f4 v[142:145], v[222:229], v[10:17], v[142:145]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[230:231], v147 offset:0
ds_read_b64_tr_b8 v[232:233], v147 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[238:239], v148 offset:0
ds_read_b64_tr_b8 v[240:241], v148 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[234:235], v149 offset:0
ds_read_b64_tr_b8 v[236:237], v149 offset:1024

	;;#ASMEND
	s_add_u32 s0, s4, 0x200000
	v_readfirstlane_b32 s31, v165
	;;#ASMSTART
	ds_read_b64_tr_b8 v[242:243], v150 offset:0
ds_read_b64_tr_b8 v[244:245], v150 offset:1024

	;;#ASMEND
	s_addc_u32 s1, s5, 0
	s_mov_b32 m0, s31
	v_readfirstlane_b32 s33, v166
	buffer_load_dwordx4 v182, s[0:3], 0 offen lds
	s_mov_b32 m0, s33
	s_nop 0
	buffer_load_dwordx4 v181, s[0:3], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[18:21], v[168:175], v[230:237], v[18:21]
	v_mfma_f32_16x16x128_f8f6f4 v[22:25], v[168:175], v[238:245], v[22:25]
	v_mfma_f32_16x16x128_f8f6f4 v[26:29], v[206:213], v[230:237], v[26:29]
	v_mfma_f32_16x16x128_f8f6f4 v[30:33], v[206:213], v[238:245], v[30:33]
	v_mfma_f32_16x16x128_f8f6f4 v[34:37], v[214:221], v[230:237], v[34:37]
	v_mfma_f32_16x16x128_f8f6f4 v[38:41], v[214:221], v[238:245], v[38:41]
	v_mfma_f32_16x16x128_f8f6f4 v[42:45], v[222:229], v[230:237], v[42:45]
	v_mfma_f32_16x16x128_f8f6f4 v[46:49], v[222:229], v[238:245], v[46:49]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[168:171], v196 offset:0

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
	ds_read_b128 v[172:175], v197 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[210:213], v197 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[218:221], v197 offset:0x1000

	;;#ASMEND
	s_add_u32 s0, s4, 0x200080
	;;#ASMSTART
	ds_read_b128 v[226:229], v197 offset:0x1800

	;;#ASMEND
	s_addc_u32 s1, s5, 0
	s_mov_b32 m0, s17
	s_nop 0
	buffer_load_dwordx4 v182, s[0:3], 0 offen lds
	s_mov_b32 m0, s18
	s_nop 0
	buffer_load_dwordx4 v181, s[0:3], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[50:53], v[168:175], v[2:9], v[50:53]
	v_mfma_f32_16x16x128_f8f6f4 v[54:57], v[168:175], v[10:17], v[54:57]
	v_mfma_f32_16x16x128_f8f6f4 v[58:61], v[206:213], v[2:9], v[58:61]
	v_mfma_f32_16x16x128_f8f6f4 v[62:65], v[206:213], v[10:17], v[62:65]
	v_mfma_f32_16x16x128_f8f6f4 v[66:69], v[214:221], v[2:9], v[66:69]
	v_mfma_f32_16x16x128_f8f6f4 v[70:73], v[214:221], v[10:17], v[70:73]
	v_mfma_f32_16x16x128_f8f6f4 v[74:77], v[222:229], v[2:9], v[74:77]
	v_mfma_f32_16x16x128_f8f6f4 v[78:81], v[222:229], v[10:17], v[78:81]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_u32 s0, s15, 0x200
	s_addc_u32 s1, s16, 0
	s_mov_b32 m0, s19
	s_nop 0
	buffer_load_dwordx4 v179, s[0:3], 0 offen lds
	s_mov_b32 m0, s20
	s_nop 0
	buffer_load_dwordx4 v178, s[0:3], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[82:85], v[168:175], v[230:237], v[82:85]
	v_mfma_f32_16x16x128_f8f6f4 v[86:89], v[168:175], v[238:245], v[86:89]
	v_mfma_f32_16x16x128_f8f6f4 v[90:93], v[206:213], v[230:237], v[90:93]
	v_mfma_f32_16x16x128_f8f6f4 v[94:97], v[206:213], v[238:245], v[94:97]
	v_mfma_f32_16x16x128_f8f6f4 v[98:101], v[214:221], v[230:237], v[98:101]
	v_mfma_f32_16x16x128_f8f6f4 v[102:105], v[214:221], v[238:245], v[102:105]
	v_mfma_f32_16x16x128_f8f6f4 v[106:109], v[222:229], v[230:237], v[106:109]
	v_mfma_f32_16x16x128_f8f6f4 v[110:113], v[222:229], v[238:245], v[110:113]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v154 offset:0
ds_read_b64_tr_b8 v[4:5], v154 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v155 offset:0
ds_read_b64_tr_b8 v[12:13], v155 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v156 offset:0
ds_read_b64_tr_b8 v[8:9], v156 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v157 offset:0
ds_read_b64_tr_b8 v[16:17], v157 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[168:171], v198 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[206:209], v198 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[214:217], v198 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[222:225], v198 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[172:175], v199 offset:0

	;;#ASMEND
	s_or_b32 s0, s14, 0x80200
	;;#ASMSTART
	ds_read_b128 v[210:213], v199 offset:0x800

	;;#ASMEND
	s_ashr_i32 s1, s0, 31
	;;#ASMSTART
	ds_read_b128 v[218:221], v199 offset:0x1000

	;;#ASMEND
	s_add_u32 s0, s10, s0
	;;#ASMSTART
	ds_read_b128 v[226:229], v199 offset:0x1800

	;;#ASMEND
	s_addc_u32 s1, s11, s1
	s_mov_b32 m0, s21
	s_nop 0
	buffer_load_dwordx4 v179, s[0:3], 0 offen lds
	s_mov_b32 m0, s22
	s_nop 0
	buffer_load_dwordx4 v178, s[0:3], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[114:117], v[168:175], v[2:9], v[114:117]
	v_mfma_f32_16x16x128_f8f6f4 v[118:121], v[168:175], v[10:17], v[118:121]
	v_mfma_f32_16x16x128_f8f6f4 v[122:125], v[206:213], v[2:9], v[122:125]
	v_mfma_f32_16x16x128_f8f6f4 v[126:129], v[206:213], v[10:17], v[126:129]
	v_mfma_f32_16x16x128_f8f6f4 v[130:133], v[214:221], v[2:9], v[130:133]
	v_mfma_f32_16x16x128_f8f6f4 v[134:137], v[214:221], v[10:17], v[134:137]
	v_mfma_f32_16x16x128_f8f6f4 v[138:141], v[222:229], v[2:9], v[138:141]
	v_mfma_f32_16x16x128_f8f6f4 v[142:145], v[222:229], v[10:17], v[142:145]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[230:231], v159 offset:0
ds_read_b64_tr_b8 v[232:233], v159 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[238:239], v160 offset:0
ds_read_b64_tr_b8 v[240:241], v160 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[234:235], v161 offset:0
ds_read_b64_tr_b8 v[236:237], v161 offset:1024

	;;#ASMEND
	s_add_u32 s0, s4, 0x280000
	;;#ASMSTART
	ds_read_b64_tr_b8 v[242:243], v162 offset:0
ds_read_b64_tr_b8 v[244:245], v162 offset:1024

	;;#ASMEND
	s_addc_u32 s1, s5, 0
	s_mov_b32 m0, s23
	s_nop 0
	buffer_load_dwordx4 v182, s[0:3], 0 offen lds
	s_mov_b32 m0, s24
	s_nop 0
	buffer_load_dwordx4 v181, s[0:3], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[18:21], v[168:175], v[230:237], v[18:21]
	v_mfma_f32_16x16x128_f8f6f4 v[22:25], v[168:175], v[238:245], v[22:25]
	v_mfma_f32_16x16x128_f8f6f4 v[26:29], v[206:213], v[230:237], v[26:29]
	v_mfma_f32_16x16x128_f8f6f4 v[30:33], v[206:213], v[238:245], v[30:33]
	v_mfma_f32_16x16x128_f8f6f4 v[34:37], v[214:221], v[230:237], v[34:37]
	v_mfma_f32_16x16x128_f8f6f4 v[38:41], v[214:221], v[238:245], v[38:41]
	v_mfma_f32_16x16x128_f8f6f4 v[42:45], v[222:229], v[230:237], v[42:45]
	v_mfma_f32_16x16x128_f8f6f4 v[46:49], v[222:229], v[238:245], v[46:49]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[168:171], v200 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[206:209], v200 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[214:217], v200 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[222:225], v200 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[172:175], v201 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[210:213], v201 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[218:221], v201 offset:0x1000

	;;#ASMEND
	s_add_u32 s0, s4, 0x280080
	;;#ASMSTART
	ds_read_b128 v[226:229], v201 offset:0x1800

	;;#ASMEND
	s_addc_u32 s1, s5, 0
	s_mov_b32 m0, s25
	s_nop 0
	buffer_load_dwordx4 v182, s[0:3], 0 offen lds
	s_mov_b32 m0, s26
	s_nop 0
	buffer_load_dwordx4 v181, s[0:3], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[50:53], v[168:175], v[2:9], v[50:53]
	v_mfma_f32_16x16x128_f8f6f4 v[54:57], v[168:175], v[10:17], v[54:57]
	v_mfma_f32_16x16x128_f8f6f4 v[58:61], v[206:213], v[2:9], v[58:61]
	v_mfma_f32_16x16x128_f8f6f4 v[62:65], v[206:213], v[10:17], v[62:65]
	v_mfma_f32_16x16x128_f8f6f4 v[66:69], v[214:221], v[2:9], v[66:69]
	v_mfma_f32_16x16x128_f8f6f4 v[70:73], v[214:221], v[10:17], v[70:73]
	v_mfma_f32_16x16x128_f8f6f4 v[74:77], v[222:229], v[2:9], v[74:77]
	v_mfma_f32_16x16x128_f8f6f4 v[78:81], v[222:229], v[10:17], v[78:81]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_u32 s0, s15, 0x280
	s_addc_u32 s1, s16, 0
	s_mov_b32 m0, s27
	s_nop 0
	buffer_load_dwordx4 v179, s[0:3], 0 offen lds
	s_mov_b32 m0, s28
	s_nop 0
	buffer_load_dwordx4 v178, s[0:3], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[82:85], v[168:175], v[230:237], v[82:85]
	v_mfma_f32_16x16x128_f8f6f4 v[86:89], v[168:175], v[238:245], v[86:89]
	v_mfma_f32_16x16x128_f8f6f4 v[90:93], v[206:213], v[230:237], v[90:93]
	v_mfma_f32_16x16x128_f8f6f4 v[94:97], v[206:213], v[238:245], v[94:97]
	v_mfma_f32_16x16x128_f8f6f4 v[98:101], v[214:221], v[230:237], v[98:101]
	v_mfma_f32_16x16x128_f8f6f4 v[102:105], v[214:221], v[238:245], v[102:105]
	v_mfma_f32_16x16x128_f8f6f4 v[106:109], v[222:229], v[230:237], v[106:109]
	v_mfma_f32_16x16x128_f8f6f4 v[110:113], v[222:229], v[238:245], v[110:113]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v192 offset:0
ds_read_b64_tr_b8 v[4:5], v192 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v184 offset:0
ds_read_b64_tr_b8 v[12:13], v184 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v185 offset:0
ds_read_b64_tr_b8 v[8:9], v185 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v183 offset:0
ds_read_b64_tr_b8 v[16:17], v183 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[168:171], v193 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[206:209], v193 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[214:217], v193 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[222:225], v193 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[172:175], v194 offset:0

	;;#ASMEND
	s_or_b32 s0, s14, 0x80280
	;;#ASMSTART
	ds_read_b128 v[210:213], v194 offset:0x800

	;;#ASMEND
	s_ashr_i32 s1, s0, 31
	;;#ASMSTART
	ds_read_b128 v[218:221], v194 offset:0x1000

	;;#ASMEND
	s_add_u32 s0, s10, s0
	;;#ASMSTART
	ds_read_b128 v[226:229], v194 offset:0x1800

	;;#ASMEND
	s_addc_u32 s1, s11, s1
	s_mov_b32 m0, s29
	s_nop 0
	buffer_load_dwordx4 v179, s[0:3], 0 offen lds
	s_mov_b32 m0, s30
	s_nop 0
	buffer_load_dwordx4 v178, s[0:3], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[114:117], v[168:175], v[2:9], v[114:117]
	v_mfma_f32_16x16x128_f8f6f4 v[118:121], v[168:175], v[10:17], v[118:121]
	v_mfma_f32_16x16x128_f8f6f4 v[122:125], v[206:213], v[2:9], v[122:125]
	v_mfma_f32_16x16x128_f8f6f4 v[126:129], v[206:213], v[10:17], v[126:129]
	v_mfma_f32_16x16x128_f8f6f4 v[130:133], v[214:221], v[2:9], v[130:133]
	v_mfma_f32_16x16x128_f8f6f4 v[134:137], v[214:221], v[10:17], v[134:137]
	v_mfma_f32_16x16x128_f8f6f4 v[138:141], v[222:229], v[2:9], v[138:141]
	v_mfma_f32_16x16x128_f8f6f4 v[142:145], v[222:229], v[10:17], v[142:145]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[230:231], v147 offset:0
ds_read_b64_tr_b8 v[232:233], v147 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[238:239], v148 offset:0
ds_read_b64_tr_b8 v[240:241], v148 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[234:235], v149 offset:0
ds_read_b64_tr_b8 v[236:237], v149 offset:1024

	;;#ASMEND
	s_add_u32 s0, s4, 0x300000
	;;#ASMSTART
	ds_read_b64_tr_b8 v[242:243], v150 offset:0
ds_read_b64_tr_b8 v[244:245], v150 offset:1024

	;;#ASMEND
	s_addc_u32 s1, s5, 0
	s_mov_b32 m0, s31
	s_nop 0
	buffer_load_dwordx4 v182, s[0:3], 0 offen lds
	s_mov_b32 m0, s33
	s_nop 0
	buffer_load_dwordx4 v181, s[0:3], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[18:21], v[168:175], v[230:237], v[18:21]
	v_mfma_f32_16x16x128_f8f6f4 v[22:25], v[168:175], v[238:245], v[22:25]
	v_mfma_f32_16x16x128_f8f6f4 v[26:29], v[206:213], v[230:237], v[26:29]
	v_mfma_f32_16x16x128_f8f6f4 v[30:33], v[206:213], v[238:245], v[30:33]
	v_mfma_f32_16x16x128_f8f6f4 v[34:37], v[214:221], v[230:237], v[34:37]
	v_mfma_f32_16x16x128_f8f6f4 v[38:41], v[214:221], v[238:245], v[38:41]
	v_mfma_f32_16x16x128_f8f6f4 v[42:45], v[222:229], v[230:237], v[42:45]
	v_mfma_f32_16x16x128_f8f6f4 v[46:49], v[222:229], v[238:245], v[46:49]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[168:171], v196 offset:0

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
	ds_read_b128 v[172:175], v197 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[210:213], v197 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[218:221], v197 offset:0x1000

	;;#ASMEND
	s_add_u32 s0, s4, 0x300080
	v_readfirstlane_b32 s17, v151
	;;#ASMSTART
	ds_read_b128 v[226:229], v197 offset:0x1800

	;;#ASMEND
	s_addc_u32 s1, s5, 0
	s_mov_b32 m0, s17
	v_readfirstlane_b32 s18, v152
	buffer_load_dwordx4 v182, s[0:3], 0 offen lds
	s_mov_b32 m0, s18
	s_nop 0
	buffer_load_dwordx4 v181, s[0:3], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[50:53], v[168:175], v[2:9], v[50:53]
	v_mfma_f32_16x16x128_f8f6f4 v[54:57], v[168:175], v[10:17], v[54:57]
	v_mfma_f32_16x16x128_f8f6f4 v[58:61], v[206:213], v[2:9], v[58:61]
	v_mfma_f32_16x16x128_f8f6f4 v[62:65], v[206:213], v[10:17], v[62:65]
	v_mfma_f32_16x16x128_f8f6f4 v[66:69], v[214:221], v[2:9], v[66:69]
	v_mfma_f32_16x16x128_f8f6f4 v[70:73], v[214:221], v[10:17], v[70:73]
	v_mfma_f32_16x16x128_f8f6f4 v[74:77], v[222:229], v[2:9], v[74:77]
	v_mfma_f32_16x16x128_f8f6f4 v[78:81], v[222:229], v[10:17], v[78:81]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_u32 s0, s15, 0x300
	v_readfirstlane_b32 s19, v180
	s_addc_u32 s1, s16, 0
	s_mov_b32 m0, s19
	v_readfirstlane_b32 s20, v153
	buffer_load_dwordx4 v179, s[0:3], 0 offen lds
	s_mov_b32 m0, s20
	s_nop 0
	buffer_load_dwordx4 v178, s[0:3], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[82:85], v[168:175], v[230:237], v[82:85]
	v_mfma_f32_16x16x128_f8f6f4 v[86:89], v[168:175], v[238:245], v[86:89]
	v_mfma_f32_16x16x128_f8f6f4 v[90:93], v[206:213], v[230:237], v[90:93]
	v_mfma_f32_16x16x128_f8f6f4 v[94:97], v[206:213], v[238:245], v[94:97]
	v_mfma_f32_16x16x128_f8f6f4 v[98:101], v[214:221], v[230:237], v[98:101]
	v_mfma_f32_16x16x128_f8f6f4 v[102:105], v[214:221], v[238:245], v[102:105]
	v_mfma_f32_16x16x128_f8f6f4 v[106:109], v[222:229], v[230:237], v[106:109]
	v_mfma_f32_16x16x128_f8f6f4 v[110:113], v[222:229], v[238:245], v[110:113]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v154 offset:0
ds_read_b64_tr_b8 v[4:5], v154 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v155 offset:0
ds_read_b64_tr_b8 v[12:13], v155 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v156 offset:0
ds_read_b64_tr_b8 v[8:9], v156 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v157 offset:0
ds_read_b64_tr_b8 v[16:17], v157 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[168:171], v198 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[206:209], v198 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[214:217], v198 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[222:225], v198 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[172:175], v199 offset:0

	;;#ASMEND
	s_or_b32 s0, s14, 0x80300
	;;#ASMSTART
	ds_read_b128 v[210:213], v199 offset:0x800

	;;#ASMEND
	s_ashr_i32 s1, s0, 31
	;;#ASMSTART
	ds_read_b128 v[218:221], v199 offset:0x1000

	;;#ASMEND
	s_add_u32 s0, s10, s0
	v_readfirstlane_b32 s21, v146
	;;#ASMSTART
	ds_read_b128 v[226:229], v199 offset:0x1800

	;;#ASMEND
	s_addc_u32 s1, s11, s1
	s_mov_b32 m0, s21
	v_readfirstlane_b32 s22, v158
	buffer_load_dwordx4 v179, s[0:3], 0 offen lds
	s_mov_b32 m0, s22
	s_nop 0
	buffer_load_dwordx4 v178, s[0:3], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[114:117], v[168:175], v[2:9], v[114:117]
	v_mfma_f32_16x16x128_f8f6f4 v[118:121], v[168:175], v[10:17], v[118:121]
	v_mfma_f32_16x16x128_f8f6f4 v[122:125], v[206:213], v[2:9], v[122:125]
	v_mfma_f32_16x16x128_f8f6f4 v[126:129], v[206:213], v[10:17], v[126:129]
	v_mfma_f32_16x16x128_f8f6f4 v[130:133], v[214:221], v[2:9], v[130:133]
	v_mfma_f32_16x16x128_f8f6f4 v[134:137], v[214:221], v[10:17], v[134:137]
	v_mfma_f32_16x16x128_f8f6f4 v[138:141], v[222:229], v[2:9], v[138:141]
	v_mfma_f32_16x16x128_f8f6f4 v[142:145], v[222:229], v[10:17], v[142:145]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[230:231], v159 offset:0
ds_read_b64_tr_b8 v[232:233], v159 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[238:239], v160 offset:0
ds_read_b64_tr_b8 v[240:241], v160 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[234:235], v161 offset:0
ds_read_b64_tr_b8 v[236:237], v161 offset:1024

	;;#ASMEND
	s_add_u32 s0, s4, 0x380000
	v_readfirstlane_b32 s23, v163
	;;#ASMSTART
	ds_read_b64_tr_b8 v[242:243], v162 offset:0
ds_read_b64_tr_b8 v[244:245], v162 offset:1024

	;;#ASMEND
	s_addc_u32 s1, s5, 0
	s_mov_b32 m0, s23
	v_readfirstlane_b32 s24, v164
	buffer_load_dwordx4 v182, s[0:3], 0 offen lds
	s_mov_b32 m0, s24
	s_nop 0
	buffer_load_dwordx4 v181, s[0:3], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[18:21], v[168:175], v[230:237], v[18:21]
	v_mfma_f32_16x16x128_f8f6f4 v[22:25], v[168:175], v[238:245], v[22:25]
	v_mfma_f32_16x16x128_f8f6f4 v[26:29], v[206:213], v[230:237], v[26:29]
	v_mfma_f32_16x16x128_f8f6f4 v[30:33], v[206:213], v[238:245], v[30:33]
	v_mfma_f32_16x16x128_f8f6f4 v[34:37], v[214:221], v[230:237], v[34:37]
	v_mfma_f32_16x16x128_f8f6f4 v[38:41], v[214:221], v[238:245], v[38:41]
	v_mfma_f32_16x16x128_f8f6f4 v[42:45], v[222:229], v[230:237], v[42:45]
	v_mfma_f32_16x16x128_f8f6f4 v[46:49], v[222:229], v[238:245], v[46:49]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[168:171], v200 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[206:209], v200 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[214:217], v200 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[222:225], v200 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[172:175], v201 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[210:213], v201 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[218:221], v201 offset:0x1000

	;;#ASMEND
	s_add_u32 s0, s4, 0x380080
	v_readfirstlane_b32 s25, v202
	;;#ASMSTART
	ds_read_b128 v[226:229], v201 offset:0x1800

	;;#ASMEND
	s_addc_u32 s1, s5, 0
	s_mov_b32 m0, s25
	v_readfirstlane_b32 s26, v203
	buffer_load_dwordx4 v182, s[0:3], 0 offen lds
	s_mov_b32 m0, s26
	s_nop 0
	buffer_load_dwordx4 v181, s[0:3], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[50:53], v[168:175], v[2:9], v[50:53]
	v_mfma_f32_16x16x128_f8f6f4 v[54:57], v[168:175], v[10:17], v[54:57]
	v_mfma_f32_16x16x128_f8f6f4 v[58:61], v[206:213], v[2:9], v[58:61]
	v_mfma_f32_16x16x128_f8f6f4 v[62:65], v[206:213], v[10:17], v[62:65]
	v_mfma_f32_16x16x128_f8f6f4 v[66:69], v[214:221], v[2:9], v[66:69]
	v_mfma_f32_16x16x128_f8f6f4 v[70:73], v[214:221], v[10:17], v[70:73]
	v_mfma_f32_16x16x128_f8f6f4 v[74:77], v[222:229], v[2:9], v[74:77]
	v_mfma_f32_16x16x128_f8f6f4 v[78:81], v[222:229], v[10:17], v[78:81]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_u32 s0, s15, 0x380
	v_readfirstlane_b32 s27, v204
	s_addc_u32 s1, s16, 0
	s_mov_b32 m0, s27
	v_readfirstlane_b32 s28, v205
	buffer_load_dwordx4 v179, s[0:3], 0 offen lds
	s_mov_b32 m0, s28
	s_nop 0
	buffer_load_dwordx4 v178, s[0:3], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[82:85], v[168:175], v[230:237], v[82:85]
	v_mfma_f32_16x16x128_f8f6f4 v[86:89], v[168:175], v[238:245], v[86:89]
	v_mfma_f32_16x16x128_f8f6f4 v[90:93], v[206:213], v[230:237], v[90:93]
	v_mfma_f32_16x16x128_f8f6f4 v[94:97], v[206:213], v[238:245], v[94:97]
	v_mfma_f32_16x16x128_f8f6f4 v[98:101], v[214:221], v[230:237], v[98:101]
	v_mfma_f32_16x16x128_f8f6f4 v[102:105], v[214:221], v[238:245], v[102:105]
	v_mfma_f32_16x16x128_f8f6f4 v[106:109], v[222:229], v[230:237], v[106:109]
	v_mfma_f32_16x16x128_f8f6f4 v[110:113], v[222:229], v[238:245], v[110:113]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v192 offset:0
ds_read_b64_tr_b8 v[4:5], v192 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v184 offset:0
ds_read_b64_tr_b8 v[12:13], v184 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v185 offset:0
ds_read_b64_tr_b8 v[8:9], v185 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v183 offset:0
ds_read_b64_tr_b8 v[16:17], v183 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[168:171], v193 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[206:209], v193 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[214:217], v193 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[222:225], v193 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[172:175], v194 offset:0

	;;#ASMEND
	s_or_b32 s0, s14, 0x80380
	;;#ASMSTART
	ds_read_b128 v[210:213], v194 offset:0x800

	;;#ASMEND
	s_ashr_i32 s1, s0, 31
	;;#ASMSTART
	ds_read_b128 v[218:221], v194 offset:0x1000

	;;#ASMEND
	s_add_u32 s0, s10, s0
	v_readfirstlane_b32 s29, v195
	;;#ASMSTART
	ds_read_b128 v[226:229], v194 offset:0x1800

	;;#ASMEND
	s_addc_u32 s1, s11, s1
	s_mov_b32 m0, s29
	v_readfirstlane_b32 s30, v167
	buffer_load_dwordx4 v179, s[0:3], 0 offen lds
	s_mov_b32 m0, s30
	s_nop 0
	buffer_load_dwordx4 v178, s[0:3], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[114:117], v[168:175], v[2:9], v[114:117]
	v_mfma_f32_16x16x128_f8f6f4 v[118:121], v[168:175], v[10:17], v[118:121]
	v_mfma_f32_16x16x128_f8f6f4 v[122:125], v[206:213], v[2:9], v[122:125]
	v_mfma_f32_16x16x128_f8f6f4 v[126:129], v[206:213], v[10:17], v[126:129]
	v_mfma_f32_16x16x128_f8f6f4 v[130:133], v[214:221], v[2:9], v[130:133]
	v_mfma_f32_16x16x128_f8f6f4 v[134:137], v[214:221], v[10:17], v[134:137]
	v_mfma_f32_16x16x128_f8f6f4 v[138:141], v[222:229], v[2:9], v[138:141]
	v_mfma_f32_16x16x128_f8f6f4 v[142:145], v[222:229], v[10:17], v[142:145]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[230:231], v147 offset:0
ds_read_b64_tr_b8 v[232:233], v147 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[238:239], v148 offset:0
ds_read_b64_tr_b8 v[240:241], v148 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[234:235], v149 offset:0
ds_read_b64_tr_b8 v[236:237], v149 offset:1024

	;;#ASMEND
	s_add_u32 s0, s4, 0x400000
	v_readfirstlane_b32 s31, v165
	;;#ASMSTART
	ds_read_b64_tr_b8 v[242:243], v150 offset:0
ds_read_b64_tr_b8 v[244:245], v150 offset:1024

	;;#ASMEND
	s_addc_u32 s1, s5, 0
	s_mov_b32 m0, s31
	v_readfirstlane_b32 s33, v166
	buffer_load_dwordx4 v182, s[0:3], 0 offen lds
	s_mov_b32 m0, s33
	s_nop 0
	buffer_load_dwordx4 v181, s[0:3], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[18:21], v[168:175], v[230:237], v[18:21]
	v_mfma_f32_16x16x128_f8f6f4 v[22:25], v[168:175], v[238:245], v[22:25]
	v_mfma_f32_16x16x128_f8f6f4 v[26:29], v[206:213], v[230:237], v[26:29]
	v_mfma_f32_16x16x128_f8f6f4 v[30:33], v[206:213], v[238:245], v[30:33]
	v_mfma_f32_16x16x128_f8f6f4 v[34:37], v[214:221], v[230:237], v[34:37]
	v_mfma_f32_16x16x128_f8f6f4 v[38:41], v[214:221], v[238:245], v[38:41]
	v_mfma_f32_16x16x128_f8f6f4 v[42:45], v[222:229], v[230:237], v[42:45]
	v_mfma_f32_16x16x128_f8f6f4 v[46:49], v[222:229], v[238:245], v[46:49]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[168:171], v196 offset:0

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
	ds_read_b128 v[172:175], v197 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[210:213], v197 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[218:221], v197 offset:0x1000

	;;#ASMEND
	s_add_u32 s0, s4, 0x400080
	;;#ASMSTART
	ds_read_b128 v[226:229], v197 offset:0x1800

	;;#ASMEND
	s_addc_u32 s1, s5, 0
	s_mov_b32 m0, s17
	s_nop 0
	buffer_load_dwordx4 v182, s[0:3], 0 offen lds
	s_mov_b32 m0, s18
	s_nop 0
	buffer_load_dwordx4 v181, s[0:3], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[50:53], v[168:175], v[2:9], v[50:53]
	v_mfma_f32_16x16x128_f8f6f4 v[54:57], v[168:175], v[10:17], v[54:57]
	v_mfma_f32_16x16x128_f8f6f4 v[58:61], v[206:213], v[2:9], v[58:61]
	v_mfma_f32_16x16x128_f8f6f4 v[62:65], v[206:213], v[10:17], v[62:65]
	v_mfma_f32_16x16x128_f8f6f4 v[66:69], v[214:221], v[2:9], v[66:69]
	v_mfma_f32_16x16x128_f8f6f4 v[70:73], v[214:221], v[10:17], v[70:73]
	v_mfma_f32_16x16x128_f8f6f4 v[74:77], v[222:229], v[2:9], v[74:77]
	v_mfma_f32_16x16x128_f8f6f4 v[78:81], v[222:229], v[10:17], v[78:81]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_u32 s0, s15, 0x400
	s_addc_u32 s1, s16, 0
	s_mov_b32 m0, s19
	s_nop 0
	buffer_load_dwordx4 v179, s[0:3], 0 offen lds
	s_mov_b32 m0, s20
	s_nop 0
	buffer_load_dwordx4 v178, s[0:3], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[82:85], v[168:175], v[230:237], v[82:85]
	v_mfma_f32_16x16x128_f8f6f4 v[86:89], v[168:175], v[238:245], v[86:89]
	v_mfma_f32_16x16x128_f8f6f4 v[90:93], v[206:213], v[230:237], v[90:93]
	v_mfma_f32_16x16x128_f8f6f4 v[94:97], v[206:213], v[238:245], v[94:97]
	v_mfma_f32_16x16x128_f8f6f4 v[98:101], v[214:221], v[230:237], v[98:101]
	v_mfma_f32_16x16x128_f8f6f4 v[102:105], v[214:221], v[238:245], v[102:105]
	v_mfma_f32_16x16x128_f8f6f4 v[106:109], v[222:229], v[230:237], v[106:109]
	v_mfma_f32_16x16x128_f8f6f4 v[110:113], v[222:229], v[238:245], v[110:113]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v154 offset:0
ds_read_b64_tr_b8 v[4:5], v154 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v155 offset:0
ds_read_b64_tr_b8 v[12:13], v155 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v156 offset:0
ds_read_b64_tr_b8 v[8:9], v156 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v157 offset:0
ds_read_b64_tr_b8 v[16:17], v157 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[168:171], v198 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[206:209], v198 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[214:217], v198 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[222:225], v198 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[172:175], v199 offset:0

	;;#ASMEND
	s_or_b32 s0, s14, 0x80400
	;;#ASMSTART
	ds_read_b128 v[210:213], v199 offset:0x800

	;;#ASMEND
	s_ashr_i32 s1, s0, 31
	;;#ASMSTART
	ds_read_b128 v[218:221], v199 offset:0x1000

	;;#ASMEND
	s_add_u32 s0, s10, s0
	;;#ASMSTART
	ds_read_b128 v[226:229], v199 offset:0x1800

	;;#ASMEND
	s_addc_u32 s1, s11, s1
	s_mov_b32 m0, s21
	s_nop 0
	buffer_load_dwordx4 v179, s[0:3], 0 offen lds
	s_mov_b32 m0, s22
	s_nop 0
	buffer_load_dwordx4 v178, s[0:3], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[114:117], v[168:175], v[2:9], v[114:117]
	v_mfma_f32_16x16x128_f8f6f4 v[118:121], v[168:175], v[10:17], v[118:121]
	v_mfma_f32_16x16x128_f8f6f4 v[122:125], v[206:213], v[2:9], v[122:125]
	v_mfma_f32_16x16x128_f8f6f4 v[126:129], v[206:213], v[10:17], v[126:129]
	v_mfma_f32_16x16x128_f8f6f4 v[130:133], v[214:221], v[2:9], v[130:133]
	v_mfma_f32_16x16x128_f8f6f4 v[134:137], v[214:221], v[10:17], v[134:137]
	v_mfma_f32_16x16x128_f8f6f4 v[138:141], v[222:229], v[2:9], v[138:141]
	v_mfma_f32_16x16x128_f8f6f4 v[142:145], v[222:229], v[10:17], v[142:145]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[230:231], v159 offset:0
ds_read_b64_tr_b8 v[232:233], v159 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[238:239], v160 offset:0
ds_read_b64_tr_b8 v[240:241], v160 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[234:235], v161 offset:0
ds_read_b64_tr_b8 v[236:237], v161 offset:1024

	;;#ASMEND
	s_add_u32 s0, s4, 0x480000
	;;#ASMSTART
	ds_read_b64_tr_b8 v[242:243], v162 offset:0
ds_read_b64_tr_b8 v[244:245], v162 offset:1024

	;;#ASMEND
	s_addc_u32 s1, s5, 0
	s_mov_b32 m0, s23
	s_nop 0
	buffer_load_dwordx4 v182, s[0:3], 0 offen lds
	s_mov_b32 m0, s24
	s_nop 0
	buffer_load_dwordx4 v181, s[0:3], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[18:21], v[168:175], v[230:237], v[18:21]
	v_mfma_f32_16x16x128_f8f6f4 v[22:25], v[168:175], v[238:245], v[22:25]
	v_mfma_f32_16x16x128_f8f6f4 v[26:29], v[206:213], v[230:237], v[26:29]
	v_mfma_f32_16x16x128_f8f6f4 v[30:33], v[206:213], v[238:245], v[30:33]
	v_mfma_f32_16x16x128_f8f6f4 v[34:37], v[214:221], v[230:237], v[34:37]
	v_mfma_f32_16x16x128_f8f6f4 v[38:41], v[214:221], v[238:245], v[38:41]
	v_mfma_f32_16x16x128_f8f6f4 v[42:45], v[222:229], v[230:237], v[42:45]
	v_mfma_f32_16x16x128_f8f6f4 v[46:49], v[222:229], v[238:245], v[46:49]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[168:171], v200 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[206:209], v200 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[214:217], v200 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[222:225], v200 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[172:175], v201 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[210:213], v201 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[218:221], v201 offset:0x1000

	;;#ASMEND
	s_add_u32 s0, s4, 0x480080
	;;#ASMSTART
	ds_read_b128 v[226:229], v201 offset:0x1800

	;;#ASMEND
	s_addc_u32 s1, s5, 0
	s_mov_b32 m0, s25
	s_nop 0
	buffer_load_dwordx4 v182, s[0:3], 0 offen lds
	s_mov_b32 m0, s26
	s_nop 0
	buffer_load_dwordx4 v181, s[0:3], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[50:53], v[168:175], v[2:9], v[50:53]
	v_mfma_f32_16x16x128_f8f6f4 v[54:57], v[168:175], v[10:17], v[54:57]
	v_mfma_f32_16x16x128_f8f6f4 v[58:61], v[206:213], v[2:9], v[58:61]
	v_mfma_f32_16x16x128_f8f6f4 v[62:65], v[206:213], v[10:17], v[62:65]
	v_mfma_f32_16x16x128_f8f6f4 v[66:69], v[214:221], v[2:9], v[66:69]
	v_mfma_f32_16x16x128_f8f6f4 v[70:73], v[214:221], v[10:17], v[70:73]
	v_mfma_f32_16x16x128_f8f6f4 v[74:77], v[222:229], v[2:9], v[74:77]
	v_mfma_f32_16x16x128_f8f6f4 v[78:81], v[222:229], v[10:17], v[78:81]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_u32 s0, s15, 0x480
	s_addc_u32 s1, s16, 0
	s_mov_b32 m0, s27
	s_nop 0
	buffer_load_dwordx4 v179, s[0:3], 0 offen lds
	s_mov_b32 m0, s28
	s_nop 0
	buffer_load_dwordx4 v178, s[0:3], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[82:85], v[168:175], v[230:237], v[82:85]
	v_mfma_f32_16x16x128_f8f6f4 v[86:89], v[168:175], v[238:245], v[86:89]
	v_mfma_f32_16x16x128_f8f6f4 v[90:93], v[206:213], v[230:237], v[90:93]
	v_mfma_f32_16x16x128_f8f6f4 v[94:97], v[206:213], v[238:245], v[94:97]
	v_mfma_f32_16x16x128_f8f6f4 v[98:101], v[214:221], v[230:237], v[98:101]
	v_mfma_f32_16x16x128_f8f6f4 v[102:105], v[214:221], v[238:245], v[102:105]
	v_mfma_f32_16x16x128_f8f6f4 v[106:109], v[222:229], v[230:237], v[106:109]
	v_mfma_f32_16x16x128_f8f6f4 v[110:113], v[222:229], v[238:245], v[110:113]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v192 offset:0
ds_read_b64_tr_b8 v[4:5], v192 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v184 offset:0
ds_read_b64_tr_b8 v[12:13], v184 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v185 offset:0
ds_read_b64_tr_b8 v[8:9], v185 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v183 offset:0
ds_read_b64_tr_b8 v[16:17], v183 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[168:171], v193 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[206:209], v193 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[214:217], v193 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[222:225], v193 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[172:175], v194 offset:0

	;;#ASMEND
	s_or_b32 s0, s14, 0x80480
	;;#ASMSTART
	ds_read_b128 v[210:213], v194 offset:0x800

	;;#ASMEND
	s_ashr_i32 s1, s0, 31
	;;#ASMSTART
	ds_read_b128 v[218:221], v194 offset:0x1000

	;;#ASMEND
	s_add_u32 s0, s10, s0
	;;#ASMSTART
	ds_read_b128 v[226:229], v194 offset:0x1800

	;;#ASMEND
	s_addc_u32 s1, s11, s1
	s_mov_b32 m0, s29
	s_nop 0
	buffer_load_dwordx4 v179, s[0:3], 0 offen lds
	s_mov_b32 m0, s30
	s_nop 0
	buffer_load_dwordx4 v178, s[0:3], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[114:117], v[168:175], v[2:9], v[114:117]
	v_mfma_f32_16x16x128_f8f6f4 v[118:121], v[168:175], v[10:17], v[118:121]
	v_mfma_f32_16x16x128_f8f6f4 v[122:125], v[206:213], v[2:9], v[122:125]
	v_mfma_f32_16x16x128_f8f6f4 v[126:129], v[206:213], v[10:17], v[126:129]
	v_mfma_f32_16x16x128_f8f6f4 v[130:133], v[214:221], v[2:9], v[130:133]
	v_mfma_f32_16x16x128_f8f6f4 v[134:137], v[214:221], v[10:17], v[134:137]
	v_mfma_f32_16x16x128_f8f6f4 v[138:141], v[222:229], v[2:9], v[138:141]
	v_mfma_f32_16x16x128_f8f6f4 v[142:145], v[222:229], v[10:17], v[142:145]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[230:231], v147 offset:0
ds_read_b64_tr_b8 v[232:233], v147 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[238:239], v148 offset:0
ds_read_b64_tr_b8 v[240:241], v148 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[234:235], v149 offset:0
ds_read_b64_tr_b8 v[236:237], v149 offset:1024

	;;#ASMEND
	s_add_u32 s0, s4, 0x500000
	;;#ASMSTART
	ds_read_b64_tr_b8 v[242:243], v150 offset:0
ds_read_b64_tr_b8 v[244:245], v150 offset:1024

	;;#ASMEND
	s_addc_u32 s1, s5, 0
	s_mov_b32 m0, s31
	s_nop 0
	buffer_load_dwordx4 v182, s[0:3], 0 offen lds
	s_mov_b32 m0, s33
	s_nop 0
	buffer_load_dwordx4 v181, s[0:3], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[18:21], v[168:175], v[230:237], v[18:21]
	v_mfma_f32_16x16x128_f8f6f4 v[22:25], v[168:175], v[238:245], v[22:25]
	v_mfma_f32_16x16x128_f8f6f4 v[26:29], v[206:213], v[230:237], v[26:29]
	v_mfma_f32_16x16x128_f8f6f4 v[30:33], v[206:213], v[238:245], v[30:33]
	v_mfma_f32_16x16x128_f8f6f4 v[34:37], v[214:221], v[230:237], v[34:37]
	v_mfma_f32_16x16x128_f8f6f4 v[38:41], v[214:221], v[238:245], v[38:41]
	v_mfma_f32_16x16x128_f8f6f4 v[42:45], v[222:229], v[230:237], v[42:45]
	v_mfma_f32_16x16x128_f8f6f4 v[46:49], v[222:229], v[238:245], v[46:49]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[168:171], v196 offset:0

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
	ds_read_b128 v[172:175], v197 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[210:213], v197 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[218:221], v197 offset:0x1000

	;;#ASMEND
	s_add_u32 s0, s4, 0x500080
	v_readfirstlane_b32 s17, v151
	;;#ASMSTART
	ds_read_b128 v[226:229], v197 offset:0x1800

	;;#ASMEND
	s_addc_u32 s1, s5, 0
	s_mov_b32 m0, s17
	v_readfirstlane_b32 s18, v152
	buffer_load_dwordx4 v182, s[0:3], 0 offen lds
	s_mov_b32 m0, s18
	s_nop 0
	buffer_load_dwordx4 v181, s[0:3], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[50:53], v[168:175], v[2:9], v[50:53]
	v_mfma_f32_16x16x128_f8f6f4 v[54:57], v[168:175], v[10:17], v[54:57]
	v_mfma_f32_16x16x128_f8f6f4 v[58:61], v[206:213], v[2:9], v[58:61]
	v_mfma_f32_16x16x128_f8f6f4 v[62:65], v[206:213], v[10:17], v[62:65]
	v_mfma_f32_16x16x128_f8f6f4 v[66:69], v[214:221], v[2:9], v[66:69]
	v_mfma_f32_16x16x128_f8f6f4 v[70:73], v[214:221], v[10:17], v[70:73]
	v_mfma_f32_16x16x128_f8f6f4 v[74:77], v[222:229], v[2:9], v[74:77]
	v_mfma_f32_16x16x128_f8f6f4 v[78:81], v[222:229], v[10:17], v[78:81]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_u32 s0, s15, 0x500
	v_readfirstlane_b32 s19, v180
	s_addc_u32 s1, s16, 0
	s_mov_b32 m0, s19
	v_readfirstlane_b32 s20, v153
	buffer_load_dwordx4 v179, s[0:3], 0 offen lds
	s_mov_b32 m0, s20
	s_nop 0
	buffer_load_dwordx4 v178, s[0:3], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[82:85], v[168:175], v[230:237], v[82:85]
	v_mfma_f32_16x16x128_f8f6f4 v[86:89], v[168:175], v[238:245], v[86:89]
	v_mfma_f32_16x16x128_f8f6f4 v[90:93], v[206:213], v[230:237], v[90:93]
	v_mfma_f32_16x16x128_f8f6f4 v[94:97], v[206:213], v[238:245], v[94:97]
	v_mfma_f32_16x16x128_f8f6f4 v[98:101], v[214:221], v[230:237], v[98:101]
	v_mfma_f32_16x16x128_f8f6f4 v[102:105], v[214:221], v[238:245], v[102:105]
	v_mfma_f32_16x16x128_f8f6f4 v[106:109], v[222:229], v[230:237], v[106:109]
	v_mfma_f32_16x16x128_f8f6f4 v[110:113], v[222:229], v[238:245], v[110:113]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v154 offset:0
ds_read_b64_tr_b8 v[4:5], v154 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v155 offset:0
ds_read_b64_tr_b8 v[12:13], v155 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v156 offset:0
ds_read_b64_tr_b8 v[8:9], v156 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v157 offset:0
ds_read_b64_tr_b8 v[16:17], v157 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[168:171], v198 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[206:209], v198 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[214:217], v198 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[222:225], v198 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[172:175], v199 offset:0

	;;#ASMEND
	s_or_b32 s0, s14, 0x80500
	;;#ASMSTART
	ds_read_b128 v[210:213], v199 offset:0x800

	;;#ASMEND
	s_ashr_i32 s1, s0, 31
	;;#ASMSTART
	ds_read_b128 v[218:221], v199 offset:0x1000

	;;#ASMEND
	s_add_u32 s0, s10, s0
	v_readfirstlane_b32 s21, v146
	;;#ASMSTART
	ds_read_b128 v[226:229], v199 offset:0x1800

	;;#ASMEND
	s_addc_u32 s1, s11, s1
	s_mov_b32 m0, s21
	v_readfirstlane_b32 s22, v158
	buffer_load_dwordx4 v179, s[0:3], 0 offen lds
	s_mov_b32 m0, s22
	s_nop 0
	buffer_load_dwordx4 v178, s[0:3], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[114:117], v[168:175], v[2:9], v[114:117]
	v_mfma_f32_16x16x128_f8f6f4 v[118:121], v[168:175], v[10:17], v[118:121]
	v_mfma_f32_16x16x128_f8f6f4 v[122:125], v[206:213], v[2:9], v[122:125]
	v_mfma_f32_16x16x128_f8f6f4 v[126:129], v[206:213], v[10:17], v[126:129]
	v_mfma_f32_16x16x128_f8f6f4 v[130:133], v[214:221], v[2:9], v[130:133]
	v_mfma_f32_16x16x128_f8f6f4 v[134:137], v[214:221], v[10:17], v[134:137]
	v_mfma_f32_16x16x128_f8f6f4 v[138:141], v[222:229], v[2:9], v[138:141]
	v_mfma_f32_16x16x128_f8f6f4 v[142:145], v[222:229], v[10:17], v[142:145]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[230:231], v159 offset:0
ds_read_b64_tr_b8 v[232:233], v159 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[238:239], v160 offset:0
ds_read_b64_tr_b8 v[240:241], v160 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[234:235], v161 offset:0
ds_read_b64_tr_b8 v[236:237], v161 offset:1024

	;;#ASMEND
	s_add_u32 s0, s4, 0x580000
	v_readfirstlane_b32 s23, v163
	;;#ASMSTART
	ds_read_b64_tr_b8 v[242:243], v162 offset:0
ds_read_b64_tr_b8 v[244:245], v162 offset:1024

	;;#ASMEND
	s_addc_u32 s1, s5, 0
	s_mov_b32 m0, s23
	v_readfirstlane_b32 s24, v164
	buffer_load_dwordx4 v182, s[0:3], 0 offen lds
	s_mov_b32 m0, s24
	s_nop 0
	buffer_load_dwordx4 v181, s[0:3], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[18:21], v[168:175], v[230:237], v[18:21]
	v_mfma_f32_16x16x128_f8f6f4 v[22:25], v[168:175], v[238:245], v[22:25]
	v_mfma_f32_16x16x128_f8f6f4 v[26:29], v[206:213], v[230:237], v[26:29]
	v_mfma_f32_16x16x128_f8f6f4 v[30:33], v[206:213], v[238:245], v[30:33]
	v_mfma_f32_16x16x128_f8f6f4 v[34:37], v[214:221], v[230:237], v[34:37]
	v_mfma_f32_16x16x128_f8f6f4 v[38:41], v[214:221], v[238:245], v[38:41]
	v_mfma_f32_16x16x128_f8f6f4 v[42:45], v[222:229], v[230:237], v[42:45]
	v_mfma_f32_16x16x128_f8f6f4 v[46:49], v[222:229], v[238:245], v[46:49]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[168:171], v200 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[206:209], v200 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[214:217], v200 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[222:225], v200 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[172:175], v201 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[210:213], v201 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[218:221], v201 offset:0x1000

	;;#ASMEND
	s_add_u32 s0, s4, 0x580080
	v_readfirstlane_b32 s25, v202
	;;#ASMSTART
	ds_read_b128 v[226:229], v201 offset:0x1800

	;;#ASMEND
	s_addc_u32 s1, s5, 0
	s_mov_b32 m0, s25
	v_readfirstlane_b32 s26, v203
	buffer_load_dwordx4 v182, s[0:3], 0 offen lds
	s_mov_b32 m0, s26
	s_nop 0
	buffer_load_dwordx4 v181, s[0:3], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[50:53], v[168:175], v[2:9], v[50:53]
	v_mfma_f32_16x16x128_f8f6f4 v[54:57], v[168:175], v[10:17], v[54:57]
	v_mfma_f32_16x16x128_f8f6f4 v[58:61], v[206:213], v[2:9], v[58:61]
	v_mfma_f32_16x16x128_f8f6f4 v[62:65], v[206:213], v[10:17], v[62:65]
	v_mfma_f32_16x16x128_f8f6f4 v[66:69], v[214:221], v[2:9], v[66:69]
	v_mfma_f32_16x16x128_f8f6f4 v[70:73], v[214:221], v[10:17], v[70:73]
	v_mfma_f32_16x16x128_f8f6f4 v[74:77], v[222:229], v[2:9], v[74:77]
	v_mfma_f32_16x16x128_f8f6f4 v[78:81], v[222:229], v[10:17], v[78:81]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_u32 s0, s15, 0x580
	v_readfirstlane_b32 s27, v204
	s_addc_u32 s1, s16, 0
	s_mov_b32 m0, s27
	v_readfirstlane_b32 s28, v205
	buffer_load_dwordx4 v179, s[0:3], 0 offen lds
	s_mov_b32 m0, s28
	s_nop 0
	buffer_load_dwordx4 v178, s[0:3], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[82:85], v[168:175], v[230:237], v[82:85]
	v_mfma_f32_16x16x128_f8f6f4 v[86:89], v[168:175], v[238:245], v[86:89]
	v_mfma_f32_16x16x128_f8f6f4 v[90:93], v[206:213], v[230:237], v[90:93]
	v_mfma_f32_16x16x128_f8f6f4 v[94:97], v[206:213], v[238:245], v[94:97]
	v_mfma_f32_16x16x128_f8f6f4 v[98:101], v[214:221], v[230:237], v[98:101]
	v_mfma_f32_16x16x128_f8f6f4 v[102:105], v[214:221], v[238:245], v[102:105]
	v_mfma_f32_16x16x128_f8f6f4 v[106:109], v[222:229], v[230:237], v[106:109]
	v_mfma_f32_16x16x128_f8f6f4 v[110:113], v[222:229], v[238:245], v[110:113]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v192 offset:0
ds_read_b64_tr_b8 v[4:5], v192 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v184 offset:0
ds_read_b64_tr_b8 v[12:13], v184 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v185 offset:0
ds_read_b64_tr_b8 v[8:9], v185 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v183 offset:0
ds_read_b64_tr_b8 v[16:17], v183 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[168:171], v193 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[206:209], v193 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[214:217], v193 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[222:225], v193 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[172:175], v194 offset:0

	;;#ASMEND
	s_or_b32 s0, s14, 0x80580
	;;#ASMSTART
	ds_read_b128 v[210:213], v194 offset:0x800

	;;#ASMEND
	s_ashr_i32 s1, s0, 31
	;;#ASMSTART
	ds_read_b128 v[218:221], v194 offset:0x1000

	;;#ASMEND
	s_add_u32 s0, s10, s0
	v_readfirstlane_b32 s29, v195
	;;#ASMSTART
	ds_read_b128 v[226:229], v194 offset:0x1800

	;;#ASMEND
	s_addc_u32 s1, s11, s1
	s_mov_b32 m0, s29
	v_readfirstlane_b32 s30, v167
	buffer_load_dwordx4 v179, s[0:3], 0 offen lds
	s_mov_b32 m0, s30
	s_nop 0
	buffer_load_dwordx4 v178, s[0:3], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[114:117], v[168:175], v[2:9], v[114:117]
	v_mfma_f32_16x16x128_f8f6f4 v[118:121], v[168:175], v[10:17], v[118:121]
	v_mfma_f32_16x16x128_f8f6f4 v[122:125], v[206:213], v[2:9], v[122:125]
	v_mfma_f32_16x16x128_f8f6f4 v[126:129], v[206:213], v[10:17], v[126:129]
	v_mfma_f32_16x16x128_f8f6f4 v[130:133], v[214:221], v[2:9], v[130:133]
	v_mfma_f32_16x16x128_f8f6f4 v[134:137], v[214:221], v[10:17], v[134:137]
	v_mfma_f32_16x16x128_f8f6f4 v[138:141], v[222:229], v[2:9], v[138:141]
	v_mfma_f32_16x16x128_f8f6f4 v[142:145], v[222:229], v[10:17], v[142:145]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[230:231], v147 offset:0
ds_read_b64_tr_b8 v[232:233], v147 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[238:239], v148 offset:0
ds_read_b64_tr_b8 v[240:241], v148 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[234:235], v149 offset:0
ds_read_b64_tr_b8 v[236:237], v149 offset:1024

	;;#ASMEND
	s_add_u32 s0, s4, 0x600000
	v_readfirstlane_b32 s31, v165
	;;#ASMSTART
	ds_read_b64_tr_b8 v[242:243], v150 offset:0
ds_read_b64_tr_b8 v[244:245], v150 offset:1024

	;;#ASMEND
	s_addc_u32 s1, s5, 0
	s_mov_b32 m0, s31
	v_readfirstlane_b32 s33, v166
	buffer_load_dwordx4 v182, s[0:3], 0 offen lds
	s_mov_b32 m0, s33
	s_nop 0
	buffer_load_dwordx4 v181, s[0:3], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[18:21], v[168:175], v[230:237], v[18:21]
	v_mfma_f32_16x16x128_f8f6f4 v[22:25], v[168:175], v[238:245], v[22:25]
	v_mfma_f32_16x16x128_f8f6f4 v[26:29], v[206:213], v[230:237], v[26:29]
	v_mfma_f32_16x16x128_f8f6f4 v[30:33], v[206:213], v[238:245], v[30:33]
	v_mfma_f32_16x16x128_f8f6f4 v[34:37], v[214:221], v[230:237], v[34:37]
	v_mfma_f32_16x16x128_f8f6f4 v[38:41], v[214:221], v[238:245], v[38:41]
	v_mfma_f32_16x16x128_f8f6f4 v[42:45], v[222:229], v[230:237], v[42:45]
	v_mfma_f32_16x16x128_f8f6f4 v[46:49], v[222:229], v[238:245], v[46:49]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[168:171], v196 offset:0

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
	ds_read_b128 v[172:175], v197 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[210:213], v197 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[218:221], v197 offset:0x1000

	;;#ASMEND
	s_add_u32 s0, s4, 0x600080
	;;#ASMSTART
	ds_read_b128 v[226:229], v197 offset:0x1800

	;;#ASMEND
	s_addc_u32 s1, s5, 0
	s_mov_b32 m0, s17
	s_nop 0
	buffer_load_dwordx4 v182, s[0:3], 0 offen lds
	s_mov_b32 m0, s18
	s_nop 0
	buffer_load_dwordx4 v181, s[0:3], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[50:53], v[168:175], v[2:9], v[50:53]
	v_mfma_f32_16x16x128_f8f6f4 v[54:57], v[168:175], v[10:17], v[54:57]
	v_mfma_f32_16x16x128_f8f6f4 v[58:61], v[206:213], v[2:9], v[58:61]
	v_mfma_f32_16x16x128_f8f6f4 v[62:65], v[206:213], v[10:17], v[62:65]
	v_mfma_f32_16x16x128_f8f6f4 v[66:69], v[214:221], v[2:9], v[66:69]
	v_mfma_f32_16x16x128_f8f6f4 v[70:73], v[214:221], v[10:17], v[70:73]
	v_mfma_f32_16x16x128_f8f6f4 v[74:77], v[222:229], v[2:9], v[74:77]
	v_mfma_f32_16x16x128_f8f6f4 v[78:81], v[222:229], v[10:17], v[78:81]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_u32 s0, s15, 0x600
	s_addc_u32 s1, s16, 0
	s_mov_b32 m0, s19
	s_nop 0
	buffer_load_dwordx4 v179, s[0:3], 0 offen lds
	s_mov_b32 m0, s20
	s_nop 0
	buffer_load_dwordx4 v178, s[0:3], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[82:85], v[168:175], v[230:237], v[82:85]
	v_mfma_f32_16x16x128_f8f6f4 v[86:89], v[168:175], v[238:245], v[86:89]
	v_mfma_f32_16x16x128_f8f6f4 v[90:93], v[206:213], v[230:237], v[90:93]
	v_mfma_f32_16x16x128_f8f6f4 v[94:97], v[206:213], v[238:245], v[94:97]
	v_mfma_f32_16x16x128_f8f6f4 v[98:101], v[214:221], v[230:237], v[98:101]
	v_mfma_f32_16x16x128_f8f6f4 v[102:105], v[214:221], v[238:245], v[102:105]
	v_mfma_f32_16x16x128_f8f6f4 v[106:109], v[222:229], v[230:237], v[106:109]
	v_mfma_f32_16x16x128_f8f6f4 v[110:113], v[222:229], v[238:245], v[110:113]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v154 offset:0
ds_read_b64_tr_b8 v[4:5], v154 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v155 offset:0
ds_read_b64_tr_b8 v[12:13], v155 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v156 offset:0
ds_read_b64_tr_b8 v[8:9], v156 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v157 offset:0
ds_read_b64_tr_b8 v[16:17], v157 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[168:171], v198 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[206:209], v198 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[214:217], v198 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[222:225], v198 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[172:175], v199 offset:0

	;;#ASMEND
	s_or_b32 s0, s14, 0x80600
	;;#ASMSTART
	ds_read_b128 v[210:213], v199 offset:0x800

	;;#ASMEND
	s_ashr_i32 s1, s0, 31
	;;#ASMSTART
	ds_read_b128 v[218:221], v199 offset:0x1000

	;;#ASMEND
	s_add_u32 s0, s10, s0
	;;#ASMSTART
	ds_read_b128 v[226:229], v199 offset:0x1800

	;;#ASMEND
	s_addc_u32 s1, s11, s1
	s_mov_b32 m0, s21
	s_nop 0
	buffer_load_dwordx4 v179, s[0:3], 0 offen lds
	s_mov_b32 m0, s22
	s_nop 0
	buffer_load_dwordx4 v178, s[0:3], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[114:117], v[168:175], v[2:9], v[114:117]
	v_mfma_f32_16x16x128_f8f6f4 v[118:121], v[168:175], v[10:17], v[118:121]
	v_mfma_f32_16x16x128_f8f6f4 v[122:125], v[206:213], v[2:9], v[122:125]
	v_mfma_f32_16x16x128_f8f6f4 v[126:129], v[206:213], v[10:17], v[126:129]
	v_mfma_f32_16x16x128_f8f6f4 v[130:133], v[214:221], v[2:9], v[130:133]
	v_mfma_f32_16x16x128_f8f6f4 v[134:137], v[214:221], v[10:17], v[134:137]
	v_mfma_f32_16x16x128_f8f6f4 v[138:141], v[222:229], v[2:9], v[138:141]
	v_mfma_f32_16x16x128_f8f6f4 v[142:145], v[222:229], v[10:17], v[142:145]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[230:231], v159 offset:0
ds_read_b64_tr_b8 v[232:233], v159 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[238:239], v160 offset:0
ds_read_b64_tr_b8 v[240:241], v160 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[234:235], v161 offset:0
ds_read_b64_tr_b8 v[236:237], v161 offset:1024

	;;#ASMEND
	s_add_u32 s0, s4, 0x680000
	;;#ASMSTART
	ds_read_b64_tr_b8 v[242:243], v162 offset:0
ds_read_b64_tr_b8 v[244:245], v162 offset:1024

	;;#ASMEND
	s_addc_u32 s1, s5, 0
	s_mov_b32 m0, s23
	s_nop 0
	buffer_load_dwordx4 v182, s[0:3], 0 offen lds
	s_mov_b32 m0, s24
	s_nop 0
	buffer_load_dwordx4 v181, s[0:3], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[18:21], v[168:175], v[230:237], v[18:21]
	v_mfma_f32_16x16x128_f8f6f4 v[22:25], v[168:175], v[238:245], v[22:25]
	v_mfma_f32_16x16x128_f8f6f4 v[26:29], v[206:213], v[230:237], v[26:29]
	v_mfma_f32_16x16x128_f8f6f4 v[30:33], v[206:213], v[238:245], v[30:33]
	v_mfma_f32_16x16x128_f8f6f4 v[34:37], v[214:221], v[230:237], v[34:37]
	v_mfma_f32_16x16x128_f8f6f4 v[38:41], v[214:221], v[238:245], v[38:41]
	v_mfma_f32_16x16x128_f8f6f4 v[42:45], v[222:229], v[230:237], v[42:45]
	v_mfma_f32_16x16x128_f8f6f4 v[46:49], v[222:229], v[238:245], v[46:49]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[168:171], v200 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[206:209], v200 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[214:217], v200 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[222:225], v200 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[172:175], v201 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[210:213], v201 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[218:221], v201 offset:0x1000

	;;#ASMEND
	s_add_u32 s0, s4, 0x680080
	;;#ASMSTART
	ds_read_b128 v[226:229], v201 offset:0x1800

	;;#ASMEND
	s_addc_u32 s1, s5, 0
	s_mov_b32 m0, s25
	s_nop 0
	buffer_load_dwordx4 v182, s[0:3], 0 offen lds
	s_mov_b32 m0, s26
	s_nop 0
	buffer_load_dwordx4 v181, s[0:3], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[50:53], v[168:175], v[2:9], v[50:53]
	v_mfma_f32_16x16x128_f8f6f4 v[54:57], v[168:175], v[10:17], v[54:57]
	v_mfma_f32_16x16x128_f8f6f4 v[58:61], v[206:213], v[2:9], v[58:61]
	v_mfma_f32_16x16x128_f8f6f4 v[62:65], v[206:213], v[10:17], v[62:65]
	v_mfma_f32_16x16x128_f8f6f4 v[66:69], v[214:221], v[2:9], v[66:69]
	v_mfma_f32_16x16x128_f8f6f4 v[70:73], v[214:221], v[10:17], v[70:73]
	v_mfma_f32_16x16x128_f8f6f4 v[74:77], v[222:229], v[2:9], v[74:77]
	v_mfma_f32_16x16x128_f8f6f4 v[78:81], v[222:229], v[10:17], v[78:81]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_u32 s0, s15, 0x680
	s_addc_u32 s1, s16, 0
	s_mov_b32 m0, s27
	s_nop 0
	buffer_load_dwordx4 v179, s[0:3], 0 offen lds
	s_mov_b32 m0, s28
	s_nop 0
	buffer_load_dwordx4 v178, s[0:3], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[82:85], v[168:175], v[230:237], v[82:85]
	v_mfma_f32_16x16x128_f8f6f4 v[86:89], v[168:175], v[238:245], v[86:89]
	v_mfma_f32_16x16x128_f8f6f4 v[90:93], v[206:213], v[230:237], v[90:93]
	v_mfma_f32_16x16x128_f8f6f4 v[94:97], v[206:213], v[238:245], v[94:97]
	v_mfma_f32_16x16x128_f8f6f4 v[98:101], v[214:221], v[230:237], v[98:101]
	v_mfma_f32_16x16x128_f8f6f4 v[102:105], v[214:221], v[238:245], v[102:105]
	v_mfma_f32_16x16x128_f8f6f4 v[106:109], v[222:229], v[230:237], v[106:109]
	v_mfma_f32_16x16x128_f8f6f4 v[110:113], v[222:229], v[238:245], v[110:113]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v192 offset:0
ds_read_b64_tr_b8 v[4:5], v192 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v184 offset:0
ds_read_b64_tr_b8 v[12:13], v184 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v185 offset:0
ds_read_b64_tr_b8 v[8:9], v185 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v183 offset:0
ds_read_b64_tr_b8 v[16:17], v183 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[168:171], v193 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[206:209], v193 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[214:217], v193 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[222:225], v193 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[172:175], v194 offset:0

	;;#ASMEND
	s_or_b32 s0, s14, 0x80680
	;;#ASMSTART
	ds_read_b128 v[210:213], v194 offset:0x800

	;;#ASMEND
	s_ashr_i32 s1, s0, 31
	;;#ASMSTART
	ds_read_b128 v[218:221], v194 offset:0x1000

	;;#ASMEND
	s_add_u32 s0, s10, s0
	;;#ASMSTART
	ds_read_b128 v[226:229], v194 offset:0x1800

	;;#ASMEND
	s_addc_u32 s1, s11, s1
	s_mov_b32 m0, s29
	s_nop 0
	buffer_load_dwordx4 v179, s[0:3], 0 offen lds
	s_mov_b32 m0, s30
	s_nop 0
	buffer_load_dwordx4 v178, s[0:3], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[114:117], v[168:175], v[2:9], v[114:117]
	v_mfma_f32_16x16x128_f8f6f4 v[118:121], v[168:175], v[10:17], v[118:121]
	v_mfma_f32_16x16x128_f8f6f4 v[122:125], v[206:213], v[2:9], v[122:125]
	v_mfma_f32_16x16x128_f8f6f4 v[126:129], v[206:213], v[10:17], v[126:129]
	v_mfma_f32_16x16x128_f8f6f4 v[130:133], v[214:221], v[2:9], v[130:133]
	v_mfma_f32_16x16x128_f8f6f4 v[134:137], v[214:221], v[10:17], v[134:137]
	v_mfma_f32_16x16x128_f8f6f4 v[138:141], v[222:229], v[2:9], v[138:141]
	v_mfma_f32_16x16x128_f8f6f4 v[142:145], v[222:229], v[10:17], v[142:145]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[230:231], v147 offset:0
ds_read_b64_tr_b8 v[232:233], v147 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[238:239], v148 offset:0
ds_read_b64_tr_b8 v[240:241], v148 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[234:235], v149 offset:0
ds_read_b64_tr_b8 v[236:237], v149 offset:1024

	;;#ASMEND
	s_add_u32 s0, s4, 0x700000
	;;#ASMSTART
	ds_read_b64_tr_b8 v[242:243], v150 offset:0
ds_read_b64_tr_b8 v[244:245], v150 offset:1024

	;;#ASMEND
	s_addc_u32 s1, s5, 0
	s_mov_b32 m0, s31
	s_nop 0
	buffer_load_dwordx4 v182, s[0:3], 0 offen lds
	s_mov_b32 m0, s33
	s_nop 0
	buffer_load_dwordx4 v181, s[0:3], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[18:21], v[168:175], v[230:237], v[18:21]
	v_mfma_f32_16x16x128_f8f6f4 v[22:25], v[168:175], v[238:245], v[22:25]
	v_mfma_f32_16x16x128_f8f6f4 v[26:29], v[206:213], v[230:237], v[26:29]
	v_mfma_f32_16x16x128_f8f6f4 v[30:33], v[206:213], v[238:245], v[30:33]
	v_mfma_f32_16x16x128_f8f6f4 v[34:37], v[214:221], v[230:237], v[34:37]
	v_mfma_f32_16x16x128_f8f6f4 v[38:41], v[214:221], v[238:245], v[38:41]
	v_mfma_f32_16x16x128_f8f6f4 v[42:45], v[222:229], v[230:237], v[42:45]
	v_mfma_f32_16x16x128_f8f6f4 v[46:49], v[222:229], v[238:245], v[46:49]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[168:171], v196 offset:0

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
	ds_read_b128 v[172:175], v197 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[210:213], v197 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[218:221], v197 offset:0x1000

	;;#ASMEND
	s_add_u32 s0, s4, 0x700080
	v_readfirstlane_b32 s17, v151
	;;#ASMSTART
	ds_read_b128 v[226:229], v197 offset:0x1800

	;;#ASMEND
	s_addc_u32 s1, s5, 0
	s_mov_b32 m0, s17
	v_readfirstlane_b32 s18, v152
	buffer_load_dwordx4 v182, s[0:3], 0 offen lds
	s_mov_b32 m0, s18
	s_nop 0
	buffer_load_dwordx4 v181, s[0:3], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[50:53], v[168:175], v[2:9], v[50:53]
	v_mfma_f32_16x16x128_f8f6f4 v[54:57], v[168:175], v[10:17], v[54:57]
	v_mfma_f32_16x16x128_f8f6f4 v[58:61], v[206:213], v[2:9], v[58:61]
	v_mfma_f32_16x16x128_f8f6f4 v[62:65], v[206:213], v[10:17], v[62:65]
	v_mfma_f32_16x16x128_f8f6f4 v[66:69], v[214:221], v[2:9], v[66:69]
	v_mfma_f32_16x16x128_f8f6f4 v[70:73], v[214:221], v[10:17], v[70:73]
	v_mfma_f32_16x16x128_f8f6f4 v[74:77], v[222:229], v[2:9], v[74:77]
	v_mfma_f32_16x16x128_f8f6f4 v[78:81], v[222:229], v[10:17], v[78:81]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_u32 s0, s15, 0x700
	v_readfirstlane_b32 s19, v180
	s_addc_u32 s1, s16, 0
	s_mov_b32 m0, s19
	v_readfirstlane_b32 s20, v153
	buffer_load_dwordx4 v179, s[0:3], 0 offen lds
	s_mov_b32 m0, s20
	s_nop 0
	buffer_load_dwordx4 v178, s[0:3], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[82:85], v[168:175], v[230:237], v[82:85]
	v_mfma_f32_16x16x128_f8f6f4 v[86:89], v[168:175], v[238:245], v[86:89]
	v_mfma_f32_16x16x128_f8f6f4 v[90:93], v[206:213], v[230:237], v[90:93]
	v_mfma_f32_16x16x128_f8f6f4 v[94:97], v[206:213], v[238:245], v[94:97]
	v_mfma_f32_16x16x128_f8f6f4 v[98:101], v[214:221], v[230:237], v[98:101]
	v_mfma_f32_16x16x128_f8f6f4 v[102:105], v[214:221], v[238:245], v[102:105]
	v_mfma_f32_16x16x128_f8f6f4 v[106:109], v[222:229], v[230:237], v[106:109]
	v_mfma_f32_16x16x128_f8f6f4 v[110:113], v[222:229], v[238:245], v[110:113]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v154 offset:0
ds_read_b64_tr_b8 v[4:5], v154 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v155 offset:0
ds_read_b64_tr_b8 v[12:13], v155 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v156 offset:0
ds_read_b64_tr_b8 v[8:9], v156 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v157 offset:0
ds_read_b64_tr_b8 v[16:17], v157 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[168:171], v198 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[206:209], v198 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[214:217], v198 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[222:225], v198 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[172:175], v199 offset:0

	;;#ASMEND
	s_or_b32 s0, s14, 0x80700
	;;#ASMSTART
	ds_read_b128 v[210:213], v199 offset:0x800

	;;#ASMEND
	s_ashr_i32 s1, s0, 31
	;;#ASMSTART
	ds_read_b128 v[218:221], v199 offset:0x1000

	;;#ASMEND
	s_add_u32 s0, s10, s0
	v_readfirstlane_b32 s21, v146
	;;#ASMSTART
	ds_read_b128 v[226:229], v199 offset:0x1800

	;;#ASMEND
	s_addc_u32 s1, s11, s1
	s_mov_b32 m0, s21
	v_readfirstlane_b32 s22, v158
	buffer_load_dwordx4 v179, s[0:3], 0 offen lds
	s_mov_b32 m0, s22
	s_nop 0
	buffer_load_dwordx4 v178, s[0:3], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[114:117], v[168:175], v[2:9], v[114:117]
	v_mfma_f32_16x16x128_f8f6f4 v[118:121], v[168:175], v[10:17], v[118:121]
	v_mfma_f32_16x16x128_f8f6f4 v[122:125], v[206:213], v[2:9], v[122:125]
	v_mfma_f32_16x16x128_f8f6f4 v[126:129], v[206:213], v[10:17], v[126:129]
	v_mfma_f32_16x16x128_f8f6f4 v[130:133], v[214:221], v[2:9], v[130:133]
	v_mfma_f32_16x16x128_f8f6f4 v[134:137], v[214:221], v[10:17], v[134:137]
	v_mfma_f32_16x16x128_f8f6f4 v[138:141], v[222:229], v[2:9], v[138:141]
	v_mfma_f32_16x16x128_f8f6f4 v[142:145], v[222:229], v[10:17], v[142:145]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[230:231], v159 offset:0
ds_read_b64_tr_b8 v[232:233], v159 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[238:239], v160 offset:0
ds_read_b64_tr_b8 v[240:241], v160 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[234:235], v161 offset:0
ds_read_b64_tr_b8 v[236:237], v161 offset:1024

	;;#ASMEND
	s_add_u32 s0, s4, 0x780000
	v_readfirstlane_b32 s23, v163
	;;#ASMSTART
	ds_read_b64_tr_b8 v[242:243], v162 offset:0
ds_read_b64_tr_b8 v[244:245], v162 offset:1024

	;;#ASMEND
	s_addc_u32 s1, s5, 0
	s_mov_b32 m0, s23
	v_readfirstlane_b32 s24, v164
	buffer_load_dwordx4 v182, s[0:3], 0 offen lds
	s_mov_b32 m0, s24
	s_nop 0
	buffer_load_dwordx4 v181, s[0:3], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[18:21], v[168:175], v[230:237], v[18:21]
	v_mfma_f32_16x16x128_f8f6f4 v[22:25], v[168:175], v[238:245], v[22:25]
	v_mfma_f32_16x16x128_f8f6f4 v[26:29], v[206:213], v[230:237], v[26:29]
	v_mfma_f32_16x16x128_f8f6f4 v[30:33], v[206:213], v[238:245], v[30:33]
	v_mfma_f32_16x16x128_f8f6f4 v[34:37], v[214:221], v[230:237], v[34:37]
	v_mfma_f32_16x16x128_f8f6f4 v[38:41], v[214:221], v[238:245], v[38:41]
	v_mfma_f32_16x16x128_f8f6f4 v[42:45], v[222:229], v[230:237], v[42:45]
	v_mfma_f32_16x16x128_f8f6f4 v[46:49], v[222:229], v[238:245], v[46:49]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[168:171], v200 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[206:209], v200 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[214:217], v200 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[222:225], v200 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[172:175], v201 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[210:213], v201 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[218:221], v201 offset:0x1000

	;;#ASMEND
	s_add_u32 s0, s4, 0x780080
	v_readfirstlane_b32 s25, v202
	;;#ASMSTART
	ds_read_b128 v[226:229], v201 offset:0x1800

	;;#ASMEND
	s_addc_u32 s1, s5, 0
	s_mov_b32 m0, s25
	v_readfirstlane_b32 s26, v203
	buffer_load_dwordx4 v182, s[0:3], 0 offen lds
	s_mov_b32 m0, s26
	s_nop 0
	buffer_load_dwordx4 v181, s[0:3], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[50:53], v[168:175], v[2:9], v[50:53]
	v_mfma_f32_16x16x128_f8f6f4 v[54:57], v[168:175], v[10:17], v[54:57]
	v_mfma_f32_16x16x128_f8f6f4 v[58:61], v[206:213], v[2:9], v[58:61]
	v_mfma_f32_16x16x128_f8f6f4 v[62:65], v[206:213], v[10:17], v[62:65]
	v_mfma_f32_16x16x128_f8f6f4 v[66:69], v[214:221], v[2:9], v[66:69]
	v_mfma_f32_16x16x128_f8f6f4 v[70:73], v[214:221], v[10:17], v[70:73]
	v_mfma_f32_16x16x128_f8f6f4 v[74:77], v[222:229], v[2:9], v[74:77]
	v_mfma_f32_16x16x128_f8f6f4 v[78:81], v[222:229], v[10:17], v[78:81]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_u32 s0, s15, 0x780
	v_readfirstlane_b32 s27, v204
	s_addc_u32 s1, s16, 0
	s_mov_b32 m0, s27
	v_readfirstlane_b32 s28, v205
	buffer_load_dwordx4 v179, s[0:3], 0 offen lds
	s_mov_b32 m0, s28
	s_nop 0
	buffer_load_dwordx4 v178, s[0:3], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[82:85], v[168:175], v[230:237], v[82:85]
	v_mfma_f32_16x16x128_f8f6f4 v[86:89], v[168:175], v[238:245], v[86:89]
	v_mfma_f32_16x16x128_f8f6f4 v[90:93], v[206:213], v[230:237], v[90:93]
	v_mfma_f32_16x16x128_f8f6f4 v[94:97], v[206:213], v[238:245], v[94:97]
	v_mfma_f32_16x16x128_f8f6f4 v[98:101], v[214:221], v[230:237], v[98:101]
	v_mfma_f32_16x16x128_f8f6f4 v[102:105], v[214:221], v[238:245], v[102:105]
	v_mfma_f32_16x16x128_f8f6f4 v[106:109], v[222:229], v[230:237], v[106:109]
	v_mfma_f32_16x16x128_f8f6f4 v[110:113], v[222:229], v[238:245], v[110:113]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v192 offset:0
ds_read_b64_tr_b8 v[4:5], v192 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v184 offset:0
ds_read_b64_tr_b8 v[12:13], v184 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v185 offset:0
ds_read_b64_tr_b8 v[8:9], v185 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v183 offset:0
ds_read_b64_tr_b8 v[16:17], v183 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[168:171], v193 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[206:209], v193 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[214:217], v193 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[222:225], v193 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[172:175], v194 offset:0

	;;#ASMEND
	s_or_b32 s0, s14, 0x80780
	;;#ASMSTART
	ds_read_b128 v[210:213], v194 offset:0x800

	;;#ASMEND
	s_ashr_i32 s1, s0, 31
	;;#ASMSTART
	ds_read_b128 v[218:221], v194 offset:0x1000

	;;#ASMEND
	s_add_u32 s0, s10, s0
	v_readfirstlane_b32 s29, v195
	;;#ASMSTART
	ds_read_b128 v[226:229], v194 offset:0x1800

	;;#ASMEND
	s_addc_u32 s1, s11, s1
	s_mov_b32 m0, s29
	v_readfirstlane_b32 s30, v167
	buffer_load_dwordx4 v179, s[0:3], 0 offen lds
	s_mov_b32 m0, s30
	s_nop 0
	buffer_load_dwordx4 v178, s[0:3], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[114:117], v[168:175], v[2:9], v[114:117]
	v_mfma_f32_16x16x128_f8f6f4 v[118:121], v[168:175], v[10:17], v[118:121]
	v_mfma_f32_16x16x128_f8f6f4 v[122:125], v[206:213], v[2:9], v[122:125]
	v_mfma_f32_16x16x128_f8f6f4 v[126:129], v[206:213], v[10:17], v[126:129]
	v_mfma_f32_16x16x128_f8f6f4 v[130:133], v[214:221], v[2:9], v[130:133]
	v_mfma_f32_16x16x128_f8f6f4 v[134:137], v[214:221], v[10:17], v[134:137]
	v_mfma_f32_16x16x128_f8f6f4 v[138:141], v[222:229], v[2:9], v[138:141]
	v_mfma_f32_16x16x128_f8f6f4 v[142:145], v[222:229], v[10:17], v[142:145]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[230:231], v147 offset:0
ds_read_b64_tr_b8 v[232:233], v147 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[238:239], v148 offset:0
ds_read_b64_tr_b8 v[240:241], v148 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[234:235], v149 offset:0
ds_read_b64_tr_b8 v[236:237], v149 offset:1024

	;;#ASMEND
	s_add_u32 s0, s4, 0x800000
	v_readfirstlane_b32 s31, v165
	;;#ASMSTART
	ds_read_b64_tr_b8 v[242:243], v150 offset:0
ds_read_b64_tr_b8 v[244:245], v150 offset:1024

	;;#ASMEND
	s_addc_u32 s1, s5, 0
	s_mov_b32 m0, s31
	v_readfirstlane_b32 s33, v166
	buffer_load_dwordx4 v182, s[0:3], 0 offen lds
	s_mov_b32 m0, s33
	s_nop 0
	buffer_load_dwordx4 v181, s[0:3], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[18:21], v[168:175], v[230:237], v[18:21]
	v_mfma_f32_16x16x128_f8f6f4 v[22:25], v[168:175], v[238:245], v[22:25]
	v_mfma_f32_16x16x128_f8f6f4 v[26:29], v[206:213], v[230:237], v[26:29]
	v_mfma_f32_16x16x128_f8f6f4 v[30:33], v[206:213], v[238:245], v[30:33]
	v_mfma_f32_16x16x128_f8f6f4 v[34:37], v[214:221], v[230:237], v[34:37]
	v_mfma_f32_16x16x128_f8f6f4 v[38:41], v[214:221], v[238:245], v[38:41]
	v_mfma_f32_16x16x128_f8f6f4 v[42:45], v[222:229], v[230:237], v[42:45]
	v_mfma_f32_16x16x128_f8f6f4 v[46:49], v[222:229], v[238:245], v[46:49]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[168:171], v196 offset:0

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
	ds_read_b128 v[172:175], v197 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[210:213], v197 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[218:221], v197 offset:0x1000

	;;#ASMEND
	s_add_u32 s0, s4, 0x800080
	;;#ASMSTART
	ds_read_b128 v[226:229], v197 offset:0x1800

	;;#ASMEND
	s_addc_u32 s1, s5, 0
	s_mov_b32 m0, s17
	s_nop 0
	buffer_load_dwordx4 v182, s[0:3], 0 offen lds
	s_mov_b32 m0, s18
	s_nop 0
	buffer_load_dwordx4 v181, s[0:3], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[50:53], v[168:175], v[2:9], v[50:53]
	v_mfma_f32_16x16x128_f8f6f4 v[54:57], v[168:175], v[10:17], v[54:57]
	v_mfma_f32_16x16x128_f8f6f4 v[58:61], v[206:213], v[2:9], v[58:61]
	v_mfma_f32_16x16x128_f8f6f4 v[62:65], v[206:213], v[10:17], v[62:65]
	v_mfma_f32_16x16x128_f8f6f4 v[66:69], v[214:221], v[2:9], v[66:69]
	v_mfma_f32_16x16x128_f8f6f4 v[70:73], v[214:221], v[10:17], v[70:73]
	v_mfma_f32_16x16x128_f8f6f4 v[74:77], v[222:229], v[2:9], v[74:77]
	v_mfma_f32_16x16x128_f8f6f4 v[78:81], v[222:229], v[10:17], v[78:81]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_u32 s0, s15, 0x800
	s_addc_u32 s1, s16, 0
	s_mov_b32 m0, s19
	s_nop 0
	buffer_load_dwordx4 v179, s[0:3], 0 offen lds
	s_mov_b32 m0, s20
	s_nop 0
	buffer_load_dwordx4 v178, s[0:3], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[82:85], v[168:175], v[230:237], v[82:85]
	v_mfma_f32_16x16x128_f8f6f4 v[86:89], v[168:175], v[238:245], v[86:89]
	v_mfma_f32_16x16x128_f8f6f4 v[90:93], v[206:213], v[230:237], v[90:93]
	v_mfma_f32_16x16x128_f8f6f4 v[94:97], v[206:213], v[238:245], v[94:97]
	v_mfma_f32_16x16x128_f8f6f4 v[98:101], v[214:221], v[230:237], v[98:101]
	v_mfma_f32_16x16x128_f8f6f4 v[102:105], v[214:221], v[238:245], v[102:105]
	v_mfma_f32_16x16x128_f8f6f4 v[106:109], v[222:229], v[230:237], v[106:109]
	v_mfma_f32_16x16x128_f8f6f4 v[110:113], v[222:229], v[238:245], v[110:113]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v154 offset:0
ds_read_b64_tr_b8 v[4:5], v154 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v155 offset:0
ds_read_b64_tr_b8 v[12:13], v155 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v156 offset:0
ds_read_b64_tr_b8 v[8:9], v156 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v157 offset:0
ds_read_b64_tr_b8 v[16:17], v157 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[168:171], v198 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[206:209], v198 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[214:217], v198 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[222:225], v198 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[172:175], v199 offset:0

	;;#ASMEND
	s_or_b32 s0, s14, 0x80800
	;;#ASMSTART
	ds_read_b128 v[210:213], v199 offset:0x800

	;;#ASMEND
	s_ashr_i32 s1, s0, 31
	;;#ASMSTART
	ds_read_b128 v[218:221], v199 offset:0x1000

	;;#ASMEND
	s_add_u32 s0, s10, s0
	;;#ASMSTART
	ds_read_b128 v[226:229], v199 offset:0x1800

	;;#ASMEND
	s_addc_u32 s1, s11, s1
	s_mov_b32 m0, s21
	s_nop 0
	buffer_load_dwordx4 v179, s[0:3], 0 offen lds
	s_mov_b32 m0, s22
	s_nop 0
	buffer_load_dwordx4 v178, s[0:3], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[114:117], v[168:175], v[2:9], v[114:117]
	v_mfma_f32_16x16x128_f8f6f4 v[118:121], v[168:175], v[10:17], v[118:121]
	v_mfma_f32_16x16x128_f8f6f4 v[122:125], v[206:213], v[2:9], v[122:125]
	v_mfma_f32_16x16x128_f8f6f4 v[126:129], v[206:213], v[10:17], v[126:129]
	v_mfma_f32_16x16x128_f8f6f4 v[130:133], v[214:221], v[2:9], v[130:133]
	v_mfma_f32_16x16x128_f8f6f4 v[134:137], v[214:221], v[10:17], v[134:137]
	v_mfma_f32_16x16x128_f8f6f4 v[138:141], v[222:229], v[2:9], v[138:141]
	v_mfma_f32_16x16x128_f8f6f4 v[142:145], v[222:229], v[10:17], v[142:145]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[230:231], v159 offset:0
ds_read_b64_tr_b8 v[232:233], v159 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[238:239], v160 offset:0
ds_read_b64_tr_b8 v[240:241], v160 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[234:235], v161 offset:0
ds_read_b64_tr_b8 v[236:237], v161 offset:1024

	;;#ASMEND
	s_add_u32 s0, s4, 0x880000
	;;#ASMSTART
	ds_read_b64_tr_b8 v[242:243], v162 offset:0
ds_read_b64_tr_b8 v[244:245], v162 offset:1024

	;;#ASMEND
	s_addc_u32 s1, s5, 0
	s_mov_b32 m0, s23
	s_nop 0
	buffer_load_dwordx4 v182, s[0:3], 0 offen lds
	s_mov_b32 m0, s24
	s_nop 0
	buffer_load_dwordx4 v181, s[0:3], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[18:21], v[168:175], v[230:237], v[18:21]
	v_mfma_f32_16x16x128_f8f6f4 v[22:25], v[168:175], v[238:245], v[22:25]
	v_mfma_f32_16x16x128_f8f6f4 v[26:29], v[206:213], v[230:237], v[26:29]
	v_mfma_f32_16x16x128_f8f6f4 v[30:33], v[206:213], v[238:245], v[30:33]
	v_mfma_f32_16x16x128_f8f6f4 v[34:37], v[214:221], v[230:237], v[34:37]
	v_mfma_f32_16x16x128_f8f6f4 v[38:41], v[214:221], v[238:245], v[38:41]
	v_mfma_f32_16x16x128_f8f6f4 v[42:45], v[222:229], v[230:237], v[42:45]
	v_mfma_f32_16x16x128_f8f6f4 v[46:49], v[222:229], v[238:245], v[46:49]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[168:171], v200 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[206:209], v200 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[214:217], v200 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[222:225], v200 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[172:175], v201 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[210:213], v201 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[218:221], v201 offset:0x1000

	;;#ASMEND
	s_add_u32 s0, s4, 0x880080
	;;#ASMSTART
	ds_read_b128 v[226:229], v201 offset:0x1800

	;;#ASMEND
	s_addc_u32 s1, s5, 0
	s_mov_b32 m0, s25
	s_nop 0
	buffer_load_dwordx4 v182, s[0:3], 0 offen lds
	s_mov_b32 m0, s26
	s_nop 0
	buffer_load_dwordx4 v181, s[0:3], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[50:53], v[168:175], v[2:9], v[50:53]
	v_mfma_f32_16x16x128_f8f6f4 v[54:57], v[168:175], v[10:17], v[54:57]
	v_mfma_f32_16x16x128_f8f6f4 v[58:61], v[206:213], v[2:9], v[58:61]
	v_mfma_f32_16x16x128_f8f6f4 v[62:65], v[206:213], v[10:17], v[62:65]
	v_mfma_f32_16x16x128_f8f6f4 v[66:69], v[214:221], v[2:9], v[66:69]
	v_mfma_f32_16x16x128_f8f6f4 v[70:73], v[214:221], v[10:17], v[70:73]
	v_mfma_f32_16x16x128_f8f6f4 v[74:77], v[222:229], v[2:9], v[74:77]
	v_mfma_f32_16x16x128_f8f6f4 v[78:81], v[222:229], v[10:17], v[78:81]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_u32 s0, s15, 0x880
	s_addc_u32 s1, s16, 0
	s_mov_b32 m0, s27
	s_nop 0
	buffer_load_dwordx4 v179, s[0:3], 0 offen lds
	s_mov_b32 m0, s28
	s_nop 0
	buffer_load_dwordx4 v178, s[0:3], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[82:85], v[168:175], v[230:237], v[82:85]
	v_mfma_f32_16x16x128_f8f6f4 v[86:89], v[168:175], v[238:245], v[86:89]
	v_mfma_f32_16x16x128_f8f6f4 v[90:93], v[206:213], v[230:237], v[90:93]
	v_mfma_f32_16x16x128_f8f6f4 v[94:97], v[206:213], v[238:245], v[94:97]
	v_mfma_f32_16x16x128_f8f6f4 v[98:101], v[214:221], v[230:237], v[98:101]
	v_mfma_f32_16x16x128_f8f6f4 v[102:105], v[214:221], v[238:245], v[102:105]
	v_mfma_f32_16x16x128_f8f6f4 v[106:109], v[222:229], v[230:237], v[106:109]
	v_mfma_f32_16x16x128_f8f6f4 v[110:113], v[222:229], v[238:245], v[110:113]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v192 offset:0
ds_read_b64_tr_b8 v[4:5], v192 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v184 offset:0
ds_read_b64_tr_b8 v[12:13], v184 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v185 offset:0
ds_read_b64_tr_b8 v[8:9], v185 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v183 offset:0
ds_read_b64_tr_b8 v[16:17], v183 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[168:171], v193 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[206:209], v193 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[214:217], v193 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[222:225], v193 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[172:175], v194 offset:0

	;;#ASMEND
	s_or_b32 s0, s14, 0x80880
	;;#ASMSTART
	ds_read_b128 v[210:213], v194 offset:0x800

	;;#ASMEND
	s_ashr_i32 s1, s0, 31
	;;#ASMSTART
	ds_read_b128 v[218:221], v194 offset:0x1000

	;;#ASMEND
	s_add_u32 s0, s10, s0
	;;#ASMSTART
	ds_read_b128 v[226:229], v194 offset:0x1800

	;;#ASMEND
	s_addc_u32 s1, s11, s1
	s_mov_b32 m0, s29
	s_nop 0
	buffer_load_dwordx4 v179, s[0:3], 0 offen lds
	s_mov_b32 m0, s30
	s_nop 0
	buffer_load_dwordx4 v178, s[0:3], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[114:117], v[168:175], v[2:9], v[114:117]
	v_mfma_f32_16x16x128_f8f6f4 v[118:121], v[168:175], v[10:17], v[118:121]
	v_mfma_f32_16x16x128_f8f6f4 v[122:125], v[206:213], v[2:9], v[122:125]
	v_mfma_f32_16x16x128_f8f6f4 v[126:129], v[206:213], v[10:17], v[126:129]
	v_mfma_f32_16x16x128_f8f6f4 v[130:133], v[214:221], v[2:9], v[130:133]
	v_mfma_f32_16x16x128_f8f6f4 v[134:137], v[214:221], v[10:17], v[134:137]
	v_mfma_f32_16x16x128_f8f6f4 v[138:141], v[222:229], v[2:9], v[138:141]
	v_mfma_f32_16x16x128_f8f6f4 v[142:145], v[222:229], v[10:17], v[142:145]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[230:231], v147 offset:0
ds_read_b64_tr_b8 v[232:233], v147 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[238:239], v148 offset:0
ds_read_b64_tr_b8 v[240:241], v148 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[234:235], v149 offset:0
ds_read_b64_tr_b8 v[236:237], v149 offset:1024

	;;#ASMEND
	s_add_u32 s0, s4, 0x900000
	;;#ASMSTART
	ds_read_b64_tr_b8 v[242:243], v150 offset:0
ds_read_b64_tr_b8 v[244:245], v150 offset:1024

	;;#ASMEND
	s_addc_u32 s1, s5, 0
	s_mov_b32 m0, s31
	s_nop 0
	buffer_load_dwordx4 v182, s[0:3], 0 offen lds
	s_mov_b32 m0, s33
	s_nop 0
	buffer_load_dwordx4 v181, s[0:3], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[18:21], v[168:175], v[230:237], v[18:21]
	v_mfma_f32_16x16x128_f8f6f4 v[22:25], v[168:175], v[238:245], v[22:25]
	v_mfma_f32_16x16x128_f8f6f4 v[26:29], v[206:213], v[230:237], v[26:29]
	v_mfma_f32_16x16x128_f8f6f4 v[30:33], v[206:213], v[238:245], v[30:33]
	v_mfma_f32_16x16x128_f8f6f4 v[34:37], v[214:221], v[230:237], v[34:37]
	v_mfma_f32_16x16x128_f8f6f4 v[38:41], v[214:221], v[238:245], v[38:41]
	v_mfma_f32_16x16x128_f8f6f4 v[42:45], v[222:229], v[230:237], v[42:45]
	v_mfma_f32_16x16x128_f8f6f4 v[46:49], v[222:229], v[238:245], v[46:49]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[168:171], v196 offset:0

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
	ds_read_b128 v[172:175], v197 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[210:213], v197 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[218:221], v197 offset:0x1000

	;;#ASMEND
	s_add_u32 s0, s4, 0x900080
	v_readfirstlane_b32 s17, v151
	;;#ASMSTART
	ds_read_b128 v[226:229], v197 offset:0x1800

	;;#ASMEND
	s_addc_u32 s1, s5, 0
	s_mov_b32 m0, s17
	v_readfirstlane_b32 s18, v152
	buffer_load_dwordx4 v182, s[0:3], 0 offen lds
	s_mov_b32 m0, s18
	s_nop 0
	buffer_load_dwordx4 v181, s[0:3], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[50:53], v[168:175], v[2:9], v[50:53]
	v_mfma_f32_16x16x128_f8f6f4 v[54:57], v[168:175], v[10:17], v[54:57]
	v_mfma_f32_16x16x128_f8f6f4 v[58:61], v[206:213], v[2:9], v[58:61]
	v_mfma_f32_16x16x128_f8f6f4 v[62:65], v[206:213], v[10:17], v[62:65]
	v_mfma_f32_16x16x128_f8f6f4 v[66:69], v[214:221], v[2:9], v[66:69]
	v_mfma_f32_16x16x128_f8f6f4 v[70:73], v[214:221], v[10:17], v[70:73]
	v_mfma_f32_16x16x128_f8f6f4 v[74:77], v[222:229], v[2:9], v[74:77]
	v_mfma_f32_16x16x128_f8f6f4 v[78:81], v[222:229], v[10:17], v[78:81]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_u32 s0, s15, 0x900
	v_readfirstlane_b32 s19, v180
	s_addc_u32 s1, s16, 0
	s_mov_b32 m0, s19
	v_readfirstlane_b32 s20, v153
	buffer_load_dwordx4 v179, s[0:3], 0 offen lds
	s_mov_b32 m0, s20
	s_nop 0
	buffer_load_dwordx4 v178, s[0:3], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[82:85], v[168:175], v[230:237], v[82:85]
	v_mfma_f32_16x16x128_f8f6f4 v[86:89], v[168:175], v[238:245], v[86:89]
	v_mfma_f32_16x16x128_f8f6f4 v[90:93], v[206:213], v[230:237], v[90:93]
	v_mfma_f32_16x16x128_f8f6f4 v[94:97], v[206:213], v[238:245], v[94:97]
	v_mfma_f32_16x16x128_f8f6f4 v[98:101], v[214:221], v[230:237], v[98:101]
	v_mfma_f32_16x16x128_f8f6f4 v[102:105], v[214:221], v[238:245], v[102:105]
	v_mfma_f32_16x16x128_f8f6f4 v[106:109], v[222:229], v[230:237], v[106:109]
	v_mfma_f32_16x16x128_f8f6f4 v[110:113], v[222:229], v[238:245], v[110:113]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v154 offset:0
ds_read_b64_tr_b8 v[4:5], v154 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v155 offset:0
ds_read_b64_tr_b8 v[12:13], v155 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v156 offset:0
ds_read_b64_tr_b8 v[8:9], v156 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v157 offset:0
ds_read_b64_tr_b8 v[16:17], v157 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[168:171], v198 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[206:209], v198 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[214:217], v198 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[222:225], v198 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[172:175], v199 offset:0

	;;#ASMEND
	s_or_b32 s0, s14, 0x80900
	;;#ASMSTART
	ds_read_b128 v[210:213], v199 offset:0x800

	;;#ASMEND
	s_ashr_i32 s1, s0, 31
	;;#ASMSTART
	ds_read_b128 v[218:221], v199 offset:0x1000

	;;#ASMEND
	s_add_u32 s0, s10, s0
	v_readfirstlane_b32 s21, v146
	;;#ASMSTART
	ds_read_b128 v[226:229], v199 offset:0x1800

	;;#ASMEND
	s_addc_u32 s1, s11, s1
	s_mov_b32 m0, s21
	v_readfirstlane_b32 s22, v158
	buffer_load_dwordx4 v179, s[0:3], 0 offen lds
	s_mov_b32 m0, s22
	s_nop 0
	buffer_load_dwordx4 v178, s[0:3], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[114:117], v[168:175], v[2:9], v[114:117]
	v_mfma_f32_16x16x128_f8f6f4 v[118:121], v[168:175], v[10:17], v[118:121]
	v_mfma_f32_16x16x128_f8f6f4 v[122:125], v[206:213], v[2:9], v[122:125]
	v_mfma_f32_16x16x128_f8f6f4 v[126:129], v[206:213], v[10:17], v[126:129]
	v_mfma_f32_16x16x128_f8f6f4 v[130:133], v[214:221], v[2:9], v[130:133]
	v_mfma_f32_16x16x128_f8f6f4 v[134:137], v[214:221], v[10:17], v[134:137]
	v_mfma_f32_16x16x128_f8f6f4 v[138:141], v[222:229], v[2:9], v[138:141]
	v_mfma_f32_16x16x128_f8f6f4 v[142:145], v[222:229], v[10:17], v[142:145]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[230:231], v159 offset:0
ds_read_b64_tr_b8 v[232:233], v159 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[238:239], v160 offset:0
ds_read_b64_tr_b8 v[240:241], v160 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[234:235], v161 offset:0
ds_read_b64_tr_b8 v[236:237], v161 offset:1024

	;;#ASMEND
	s_add_u32 s0, s4, 0x980000
	v_readfirstlane_b32 s23, v163
	;;#ASMSTART
	ds_read_b64_tr_b8 v[242:243], v162 offset:0
ds_read_b64_tr_b8 v[244:245], v162 offset:1024

	;;#ASMEND
	s_addc_u32 s1, s5, 0
	s_mov_b32 m0, s23
	v_readfirstlane_b32 s24, v164
	buffer_load_dwordx4 v182, s[0:3], 0 offen lds
	s_mov_b32 m0, s24
	s_nop 0
	buffer_load_dwordx4 v181, s[0:3], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[18:21], v[168:175], v[230:237], v[18:21]
	v_mfma_f32_16x16x128_f8f6f4 v[22:25], v[168:175], v[238:245], v[22:25]
	v_mfma_f32_16x16x128_f8f6f4 v[26:29], v[206:213], v[230:237], v[26:29]
	v_mfma_f32_16x16x128_f8f6f4 v[30:33], v[206:213], v[238:245], v[30:33]
	v_mfma_f32_16x16x128_f8f6f4 v[34:37], v[214:221], v[230:237], v[34:37]
	v_mfma_f32_16x16x128_f8f6f4 v[38:41], v[214:221], v[238:245], v[38:41]
	v_mfma_f32_16x16x128_f8f6f4 v[42:45], v[222:229], v[230:237], v[42:45]
	v_mfma_f32_16x16x128_f8f6f4 v[46:49], v[222:229], v[238:245], v[46:49]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[168:171], v200 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[206:209], v200 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[214:217], v200 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[222:225], v200 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[172:175], v201 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[210:213], v201 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[218:221], v201 offset:0x1000

	;;#ASMEND
	s_add_u32 s0, s4, 0x980080
	v_readfirstlane_b32 s25, v202
	;;#ASMSTART
	ds_read_b128 v[226:229], v201 offset:0x1800

	;;#ASMEND
	s_addc_u32 s1, s5, 0
	s_mov_b32 m0, s25
	v_readfirstlane_b32 s26, v203
	buffer_load_dwordx4 v182, s[0:3], 0 offen lds
	s_mov_b32 m0, s26
	s_nop 0
	buffer_load_dwordx4 v181, s[0:3], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[50:53], v[168:175], v[2:9], v[50:53]
	v_mfma_f32_16x16x128_f8f6f4 v[54:57], v[168:175], v[10:17], v[54:57]
	v_mfma_f32_16x16x128_f8f6f4 v[58:61], v[206:213], v[2:9], v[58:61]
	v_mfma_f32_16x16x128_f8f6f4 v[62:65], v[206:213], v[10:17], v[62:65]
	v_mfma_f32_16x16x128_f8f6f4 v[66:69], v[214:221], v[2:9], v[66:69]
	v_mfma_f32_16x16x128_f8f6f4 v[70:73], v[214:221], v[10:17], v[70:73]
	v_mfma_f32_16x16x128_f8f6f4 v[74:77], v[222:229], v[2:9], v[74:77]
	v_mfma_f32_16x16x128_f8f6f4 v[78:81], v[222:229], v[10:17], v[78:81]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_u32 s0, s15, 0x980
	v_readfirstlane_b32 s27, v204
	s_addc_u32 s1, s16, 0
	s_mov_b32 m0, s27
	v_readfirstlane_b32 s28, v205
	buffer_load_dwordx4 v179, s[0:3], 0 offen lds
	s_mov_b32 m0, s28
	s_nop 0
	buffer_load_dwordx4 v178, s[0:3], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[82:85], v[168:175], v[230:237], v[82:85]
	v_mfma_f32_16x16x128_f8f6f4 v[86:89], v[168:175], v[238:245], v[86:89]
	v_mfma_f32_16x16x128_f8f6f4 v[90:93], v[206:213], v[230:237], v[90:93]
	v_mfma_f32_16x16x128_f8f6f4 v[94:97], v[206:213], v[238:245], v[94:97]
	v_mfma_f32_16x16x128_f8f6f4 v[98:101], v[214:221], v[230:237], v[98:101]
	v_mfma_f32_16x16x128_f8f6f4 v[102:105], v[214:221], v[238:245], v[102:105]
	v_mfma_f32_16x16x128_f8f6f4 v[106:109], v[222:229], v[230:237], v[106:109]
	v_mfma_f32_16x16x128_f8f6f4 v[110:113], v[222:229], v[238:245], v[110:113]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v192 offset:0
ds_read_b64_tr_b8 v[4:5], v192 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v184 offset:0
ds_read_b64_tr_b8 v[12:13], v184 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v185 offset:0
ds_read_b64_tr_b8 v[8:9], v185 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v183 offset:0
ds_read_b64_tr_b8 v[16:17], v183 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[168:171], v193 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[206:209], v193 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[214:217], v193 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[222:225], v193 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[172:175], v194 offset:0

	;;#ASMEND
	s_or_b32 s0, s14, 0x80980
	;;#ASMSTART
	ds_read_b128 v[210:213], v194 offset:0x800

	;;#ASMEND
	s_ashr_i32 s1, s0, 31
	;;#ASMSTART
	ds_read_b128 v[218:221], v194 offset:0x1000

	;;#ASMEND
	s_add_u32 s0, s10, s0
	v_readfirstlane_b32 s29, v195
	;;#ASMSTART
	ds_read_b128 v[226:229], v194 offset:0x1800

	;;#ASMEND
	s_addc_u32 s1, s11, s1
	s_mov_b32 m0, s29
	v_readfirstlane_b32 s30, v167
	buffer_load_dwordx4 v179, s[0:3], 0 offen lds
	s_mov_b32 m0, s30
	s_nop 0
	buffer_load_dwordx4 v178, s[0:3], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[114:117], v[168:175], v[2:9], v[114:117]
	v_mfma_f32_16x16x128_f8f6f4 v[118:121], v[168:175], v[10:17], v[118:121]
	v_mfma_f32_16x16x128_f8f6f4 v[122:125], v[206:213], v[2:9], v[122:125]
	v_mfma_f32_16x16x128_f8f6f4 v[126:129], v[206:213], v[10:17], v[126:129]
	v_mfma_f32_16x16x128_f8f6f4 v[130:133], v[214:221], v[2:9], v[130:133]
	v_mfma_f32_16x16x128_f8f6f4 v[134:137], v[214:221], v[10:17], v[134:137]
	v_mfma_f32_16x16x128_f8f6f4 v[138:141], v[222:229], v[2:9], v[138:141]
	v_mfma_f32_16x16x128_f8f6f4 v[142:145], v[222:229], v[10:17], v[142:145]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[230:231], v147 offset:0
ds_read_b64_tr_b8 v[232:233], v147 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[238:239], v148 offset:0
ds_read_b64_tr_b8 v[240:241], v148 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[234:235], v149 offset:0
ds_read_b64_tr_b8 v[236:237], v149 offset:1024

	;;#ASMEND
	s_add_u32 s0, s4, 0xa00000
	v_readfirstlane_b32 s31, v165
	;;#ASMSTART
	ds_read_b64_tr_b8 v[242:243], v150 offset:0
ds_read_b64_tr_b8 v[244:245], v150 offset:1024

	;;#ASMEND
	s_addc_u32 s1, s5, 0
	s_mov_b32 m0, s31
	v_readfirstlane_b32 s33, v166
	buffer_load_dwordx4 v182, s[0:3], 0 offen lds
	s_mov_b32 m0, s33
	s_nop 0
	buffer_load_dwordx4 v181, s[0:3], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[18:21], v[168:175], v[230:237], v[18:21]
	v_mfma_f32_16x16x128_f8f6f4 v[22:25], v[168:175], v[238:245], v[22:25]
	v_mfma_f32_16x16x128_f8f6f4 v[26:29], v[206:213], v[230:237], v[26:29]
	v_mfma_f32_16x16x128_f8f6f4 v[30:33], v[206:213], v[238:245], v[30:33]
	v_mfma_f32_16x16x128_f8f6f4 v[34:37], v[214:221], v[230:237], v[34:37]
	v_mfma_f32_16x16x128_f8f6f4 v[38:41], v[214:221], v[238:245], v[38:41]
	v_mfma_f32_16x16x128_f8f6f4 v[42:45], v[222:229], v[230:237], v[42:45]
	v_mfma_f32_16x16x128_f8f6f4 v[46:49], v[222:229], v[238:245], v[46:49]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[168:171], v196 offset:0

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
	ds_read_b128 v[172:175], v197 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[210:213], v197 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[218:221], v197 offset:0x1000

	;;#ASMEND
	s_add_u32 s0, s4, 0xa00080
	;;#ASMSTART
	ds_read_b128 v[226:229], v197 offset:0x1800

	;;#ASMEND
	s_addc_u32 s1, s5, 0
	s_mov_b32 m0, s17
	s_nop 0
	buffer_load_dwordx4 v182, s[0:3], 0 offen lds
	s_mov_b32 m0, s18
	s_nop 0
	buffer_load_dwordx4 v181, s[0:3], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[50:53], v[168:175], v[2:9], v[50:53]
	v_mfma_f32_16x16x128_f8f6f4 v[54:57], v[168:175], v[10:17], v[54:57]
	v_mfma_f32_16x16x128_f8f6f4 v[58:61], v[206:213], v[2:9], v[58:61]
	v_mfma_f32_16x16x128_f8f6f4 v[62:65], v[206:213], v[10:17], v[62:65]
	v_mfma_f32_16x16x128_f8f6f4 v[66:69], v[214:221], v[2:9], v[66:69]
	v_mfma_f32_16x16x128_f8f6f4 v[70:73], v[214:221], v[10:17], v[70:73]
	v_mfma_f32_16x16x128_f8f6f4 v[74:77], v[222:229], v[2:9], v[74:77]
	v_mfma_f32_16x16x128_f8f6f4 v[78:81], v[222:229], v[10:17], v[78:81]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_u32 s0, s15, 0xa00
	s_addc_u32 s1, s16, 0
	s_mov_b32 m0, s19
	s_nop 0
	buffer_load_dwordx4 v179, s[0:3], 0 offen lds
	s_mov_b32 m0, s20
	s_nop 0
	buffer_load_dwordx4 v178, s[0:3], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[82:85], v[168:175], v[230:237], v[82:85]
	v_mfma_f32_16x16x128_f8f6f4 v[86:89], v[168:175], v[238:245], v[86:89]
	v_mfma_f32_16x16x128_f8f6f4 v[90:93], v[206:213], v[230:237], v[90:93]
	v_mfma_f32_16x16x128_f8f6f4 v[94:97], v[206:213], v[238:245], v[94:97]
	v_mfma_f32_16x16x128_f8f6f4 v[98:101], v[214:221], v[230:237], v[98:101]
	v_mfma_f32_16x16x128_f8f6f4 v[102:105], v[214:221], v[238:245], v[102:105]
	v_mfma_f32_16x16x128_f8f6f4 v[106:109], v[222:229], v[230:237], v[106:109]
	v_mfma_f32_16x16x128_f8f6f4 v[110:113], v[222:229], v[238:245], v[110:113]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v154 offset:0
ds_read_b64_tr_b8 v[4:5], v154 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v155 offset:0
ds_read_b64_tr_b8 v[12:13], v155 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v156 offset:0
ds_read_b64_tr_b8 v[8:9], v156 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v157 offset:0
ds_read_b64_tr_b8 v[16:17], v157 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[168:171], v198 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[206:209], v198 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[214:217], v198 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[222:225], v198 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[172:175], v199 offset:0

	;;#ASMEND
	s_or_b32 s0, s14, 0x80a00
	;;#ASMSTART
	ds_read_b128 v[210:213], v199 offset:0x800

	;;#ASMEND
	s_ashr_i32 s1, s0, 31
	;;#ASMSTART
	ds_read_b128 v[218:221], v199 offset:0x1000

	;;#ASMEND
	s_add_u32 s0, s10, s0
	;;#ASMSTART
	ds_read_b128 v[226:229], v199 offset:0x1800

	;;#ASMEND
	s_addc_u32 s1, s11, s1
	s_mov_b32 m0, s21
	s_nop 0
	buffer_load_dwordx4 v179, s[0:3], 0 offen lds
	s_mov_b32 m0, s22
	s_nop 0
	buffer_load_dwordx4 v178, s[0:3], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[114:117], v[168:175], v[2:9], v[114:117]
	v_mfma_f32_16x16x128_f8f6f4 v[118:121], v[168:175], v[10:17], v[118:121]
	v_mfma_f32_16x16x128_f8f6f4 v[122:125], v[206:213], v[2:9], v[122:125]
	v_mfma_f32_16x16x128_f8f6f4 v[126:129], v[206:213], v[10:17], v[126:129]
	v_mfma_f32_16x16x128_f8f6f4 v[130:133], v[214:221], v[2:9], v[130:133]
	v_mfma_f32_16x16x128_f8f6f4 v[134:137], v[214:221], v[10:17], v[134:137]
	v_mfma_f32_16x16x128_f8f6f4 v[138:141], v[222:229], v[2:9], v[138:141]
	v_mfma_f32_16x16x128_f8f6f4 v[142:145], v[222:229], v[10:17], v[142:145]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[230:231], v159 offset:0
ds_read_b64_tr_b8 v[232:233], v159 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[238:239], v160 offset:0
ds_read_b64_tr_b8 v[240:241], v160 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[234:235], v161 offset:0
ds_read_b64_tr_b8 v[236:237], v161 offset:1024

	;;#ASMEND
	s_add_u32 s0, s4, 0xa80000
	;;#ASMSTART
	ds_read_b64_tr_b8 v[242:243], v162 offset:0
ds_read_b64_tr_b8 v[244:245], v162 offset:1024

	;;#ASMEND
	s_addc_u32 s1, s5, 0
	s_mov_b32 m0, s23
	s_nop 0
	buffer_load_dwordx4 v182, s[0:3], 0 offen lds
	s_mov_b32 m0, s24
	s_nop 0
	buffer_load_dwordx4 v181, s[0:3], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[18:21], v[168:175], v[230:237], v[18:21]
	v_mfma_f32_16x16x128_f8f6f4 v[22:25], v[168:175], v[238:245], v[22:25]
	v_mfma_f32_16x16x128_f8f6f4 v[26:29], v[206:213], v[230:237], v[26:29]
	v_mfma_f32_16x16x128_f8f6f4 v[30:33], v[206:213], v[238:245], v[30:33]
	v_mfma_f32_16x16x128_f8f6f4 v[34:37], v[214:221], v[230:237], v[34:37]
	v_mfma_f32_16x16x128_f8f6f4 v[38:41], v[214:221], v[238:245], v[38:41]
	v_mfma_f32_16x16x128_f8f6f4 v[42:45], v[222:229], v[230:237], v[42:45]
	v_mfma_f32_16x16x128_f8f6f4 v[46:49], v[222:229], v[238:245], v[46:49]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[168:171], v200 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[206:209], v200 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[214:217], v200 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[222:225], v200 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[172:175], v201 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[210:213], v201 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[218:221], v201 offset:0x1000

	;;#ASMEND
	s_add_u32 s0, s4, 0xa80080
	;;#ASMSTART
	ds_read_b128 v[226:229], v201 offset:0x1800

	;;#ASMEND
	s_addc_u32 s1, s5, 0
	s_mov_b32 m0, s25
	s_nop 0
	buffer_load_dwordx4 v182, s[0:3], 0 offen lds
	s_mov_b32 m0, s26
	s_nop 0
	buffer_load_dwordx4 v181, s[0:3], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[50:53], v[168:175], v[2:9], v[50:53]
	v_mfma_f32_16x16x128_f8f6f4 v[54:57], v[168:175], v[10:17], v[54:57]
	v_mfma_f32_16x16x128_f8f6f4 v[58:61], v[206:213], v[2:9], v[58:61]
	v_mfma_f32_16x16x128_f8f6f4 v[62:65], v[206:213], v[10:17], v[62:65]
	v_mfma_f32_16x16x128_f8f6f4 v[66:69], v[214:221], v[2:9], v[66:69]
	v_mfma_f32_16x16x128_f8f6f4 v[70:73], v[214:221], v[10:17], v[70:73]
	v_mfma_f32_16x16x128_f8f6f4 v[74:77], v[222:229], v[2:9], v[74:77]
	v_mfma_f32_16x16x128_f8f6f4 v[78:81], v[222:229], v[10:17], v[78:81]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_u32 s0, s15, 0xa80
	s_addc_u32 s1, s16, 0
	s_mov_b32 m0, s27
	s_nop 0
	buffer_load_dwordx4 v179, s[0:3], 0 offen lds
	s_mov_b32 m0, s28
	s_nop 0
	buffer_load_dwordx4 v178, s[0:3], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[82:85], v[168:175], v[230:237], v[82:85]
	v_mfma_f32_16x16x128_f8f6f4 v[86:89], v[168:175], v[238:245], v[86:89]
	v_mfma_f32_16x16x128_f8f6f4 v[90:93], v[206:213], v[230:237], v[90:93]
	v_mfma_f32_16x16x128_f8f6f4 v[94:97], v[206:213], v[238:245], v[94:97]
	v_mfma_f32_16x16x128_f8f6f4 v[98:101], v[214:221], v[230:237], v[98:101]
	v_mfma_f32_16x16x128_f8f6f4 v[102:105], v[214:221], v[238:245], v[102:105]
	v_mfma_f32_16x16x128_f8f6f4 v[106:109], v[222:229], v[230:237], v[106:109]
	v_mfma_f32_16x16x128_f8f6f4 v[110:113], v[222:229], v[238:245], v[110:113]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v192 offset:0
ds_read_b64_tr_b8 v[4:5], v192 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v184 offset:0
ds_read_b64_tr_b8 v[12:13], v184 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v185 offset:0
ds_read_b64_tr_b8 v[8:9], v185 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v183 offset:0
ds_read_b64_tr_b8 v[16:17], v183 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[168:171], v193 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[206:209], v193 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[214:217], v193 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[222:225], v193 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[172:175], v194 offset:0

	;;#ASMEND
	s_or_b32 s0, s14, 0x80a80
	;;#ASMSTART
	ds_read_b128 v[210:213], v194 offset:0x800

	;;#ASMEND
	s_ashr_i32 s1, s0, 31
	;;#ASMSTART
	ds_read_b128 v[218:221], v194 offset:0x1000

	;;#ASMEND
	s_add_u32 s0, s10, s0
	;;#ASMSTART
	ds_read_b128 v[226:229], v194 offset:0x1800

	;;#ASMEND
	s_addc_u32 s1, s11, s1
	s_mov_b32 m0, s29
	s_nop 0
	buffer_load_dwordx4 v179, s[0:3], 0 offen lds
	s_mov_b32 m0, s30
	s_nop 0
	buffer_load_dwordx4 v178, s[0:3], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[114:117], v[168:175], v[2:9], v[114:117]
	v_mfma_f32_16x16x128_f8f6f4 v[118:121], v[168:175], v[10:17], v[118:121]
	v_mfma_f32_16x16x128_f8f6f4 v[122:125], v[206:213], v[2:9], v[122:125]
	v_mfma_f32_16x16x128_f8f6f4 v[126:129], v[206:213], v[10:17], v[126:129]
	v_mfma_f32_16x16x128_f8f6f4 v[130:133], v[214:221], v[2:9], v[130:133]
	v_mfma_f32_16x16x128_f8f6f4 v[134:137], v[214:221], v[10:17], v[134:137]
	v_mfma_f32_16x16x128_f8f6f4 v[138:141], v[222:229], v[2:9], v[138:141]
	v_mfma_f32_16x16x128_f8f6f4 v[142:145], v[222:229], v[10:17], v[142:145]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[230:231], v147 offset:0
ds_read_b64_tr_b8 v[232:233], v147 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[238:239], v148 offset:0
ds_read_b64_tr_b8 v[240:241], v148 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[234:235], v149 offset:0
ds_read_b64_tr_b8 v[236:237], v149 offset:1024

	;;#ASMEND
	s_add_u32 s0, s4, 0xb00000
	;;#ASMSTART
	ds_read_b64_tr_b8 v[242:243], v150 offset:0
ds_read_b64_tr_b8 v[244:245], v150 offset:1024

	;;#ASMEND
	s_addc_u32 s1, s5, 0
	s_mov_b32 m0, s31
	s_nop 0
	buffer_load_dwordx4 v182, s[0:3], 0 offen lds
	s_mov_b32 m0, s33
	s_nop 0
	buffer_load_dwordx4 v181, s[0:3], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[18:21], v[168:175], v[230:237], v[18:21]
	v_mfma_f32_16x16x128_f8f6f4 v[22:25], v[168:175], v[238:245], v[22:25]
	v_mfma_f32_16x16x128_f8f6f4 v[26:29], v[206:213], v[230:237], v[26:29]
	v_mfma_f32_16x16x128_f8f6f4 v[30:33], v[206:213], v[238:245], v[30:33]
	v_mfma_f32_16x16x128_f8f6f4 v[34:37], v[214:221], v[230:237], v[34:37]
	v_mfma_f32_16x16x128_f8f6f4 v[38:41], v[214:221], v[238:245], v[38:41]
	v_mfma_f32_16x16x128_f8f6f4 v[42:45], v[222:229], v[230:237], v[42:45]
	v_mfma_f32_16x16x128_f8f6f4 v[46:49], v[222:229], v[238:245], v[46:49]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[168:171], v196 offset:0

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
	ds_read_b128 v[172:175], v197 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[210:213], v197 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[218:221], v197 offset:0x1000

	;;#ASMEND
	s_add_u32 s0, s4, 0xb00080
	v_readfirstlane_b32 s17, v151
	;;#ASMSTART
	ds_read_b128 v[226:229], v197 offset:0x1800

	;;#ASMEND
	s_addc_u32 s1, s5, 0
	s_mov_b32 m0, s17
	v_readfirstlane_b32 s18, v152
	buffer_load_dwordx4 v182, s[0:3], 0 offen lds
	s_mov_b32 m0, s18
	s_nop 0
	buffer_load_dwordx4 v181, s[0:3], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[50:53], v[168:175], v[2:9], v[50:53]
	v_mfma_f32_16x16x128_f8f6f4 v[54:57], v[168:175], v[10:17], v[54:57]
	v_mfma_f32_16x16x128_f8f6f4 v[58:61], v[206:213], v[2:9], v[58:61]
	v_mfma_f32_16x16x128_f8f6f4 v[62:65], v[206:213], v[10:17], v[62:65]
	v_mfma_f32_16x16x128_f8f6f4 v[66:69], v[214:221], v[2:9], v[66:69]
	v_mfma_f32_16x16x128_f8f6f4 v[70:73], v[214:221], v[10:17], v[70:73]
	v_mfma_f32_16x16x128_f8f6f4 v[74:77], v[222:229], v[2:9], v[74:77]
	v_mfma_f32_16x16x128_f8f6f4 v[78:81], v[222:229], v[10:17], v[78:81]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_u32 s0, s15, 0xb00
	v_readfirstlane_b32 s19, v180
	s_addc_u32 s1, s16, 0
	s_mov_b32 m0, s19
	v_readfirstlane_b32 s20, v153
	buffer_load_dwordx4 v179, s[0:3], 0 offen lds
	s_mov_b32 m0, s20
	s_nop 0
	buffer_load_dwordx4 v178, s[0:3], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[82:85], v[168:175], v[230:237], v[82:85]
	v_mfma_f32_16x16x128_f8f6f4 v[86:89], v[168:175], v[238:245], v[86:89]
	v_mfma_f32_16x16x128_f8f6f4 v[90:93], v[206:213], v[230:237], v[90:93]
	v_mfma_f32_16x16x128_f8f6f4 v[94:97], v[206:213], v[238:245], v[94:97]
	v_mfma_f32_16x16x128_f8f6f4 v[98:101], v[214:221], v[230:237], v[98:101]
	v_mfma_f32_16x16x128_f8f6f4 v[102:105], v[214:221], v[238:245], v[102:105]
	v_mfma_f32_16x16x128_f8f6f4 v[106:109], v[222:229], v[230:237], v[106:109]
	v_mfma_f32_16x16x128_f8f6f4 v[110:113], v[222:229], v[238:245], v[110:113]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v154 offset:0
ds_read_b64_tr_b8 v[4:5], v154 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v155 offset:0
ds_read_b64_tr_b8 v[12:13], v155 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v156 offset:0
ds_read_b64_tr_b8 v[8:9], v156 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v157 offset:0
ds_read_b64_tr_b8 v[16:17], v157 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[168:171], v198 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[206:209], v198 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[214:217], v198 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[222:225], v198 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[172:175], v199 offset:0

	;;#ASMEND
	s_or_b32 s0, s14, 0x80b00
	;;#ASMSTART
	ds_read_b128 v[210:213], v199 offset:0x800

	;;#ASMEND
	s_ashr_i32 s1, s0, 31
	;;#ASMSTART
	ds_read_b128 v[218:221], v199 offset:0x1000

	;;#ASMEND
	s_add_u32 s0, s10, s0
	v_readfirstlane_b32 s21, v146
	;;#ASMSTART
	ds_read_b128 v[226:229], v199 offset:0x1800

	;;#ASMEND
	s_addc_u32 s1, s11, s1
	s_mov_b32 m0, s21
	v_readfirstlane_b32 s22, v158
	buffer_load_dwordx4 v179, s[0:3], 0 offen lds
	s_mov_b32 m0, s22
	s_nop 0
	buffer_load_dwordx4 v178, s[0:3], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[114:117], v[168:175], v[2:9], v[114:117]
	v_mfma_f32_16x16x128_f8f6f4 v[118:121], v[168:175], v[10:17], v[118:121]
	v_mfma_f32_16x16x128_f8f6f4 v[122:125], v[206:213], v[2:9], v[122:125]
	v_mfma_f32_16x16x128_f8f6f4 v[126:129], v[206:213], v[10:17], v[126:129]
	v_mfma_f32_16x16x128_f8f6f4 v[130:133], v[214:221], v[2:9], v[130:133]
	v_mfma_f32_16x16x128_f8f6f4 v[134:137], v[214:221], v[10:17], v[134:137]
	v_mfma_f32_16x16x128_f8f6f4 v[138:141], v[222:229], v[2:9], v[138:141]
	v_mfma_f32_16x16x128_f8f6f4 v[142:145], v[222:229], v[10:17], v[142:145]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[230:231], v159 offset:0
ds_read_b64_tr_b8 v[232:233], v159 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[238:239], v160 offset:0
ds_read_b64_tr_b8 v[240:241], v160 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[234:235], v161 offset:0
ds_read_b64_tr_b8 v[236:237], v161 offset:1024

	;;#ASMEND
	s_add_u32 s0, s4, 0xb80000
	v_readfirstlane_b32 s23, v163
	;;#ASMSTART
	ds_read_b64_tr_b8 v[242:243], v162 offset:0
ds_read_b64_tr_b8 v[244:245], v162 offset:1024

	;;#ASMEND
	s_addc_u32 s1, s5, 0
	s_mov_b32 m0, s23
	v_readfirstlane_b32 s24, v164
	buffer_load_dwordx4 v182, s[0:3], 0 offen lds
	s_mov_b32 m0, s24
	s_nop 0
	buffer_load_dwordx4 v181, s[0:3], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[18:21], v[168:175], v[230:237], v[18:21]
	v_mfma_f32_16x16x128_f8f6f4 v[22:25], v[168:175], v[238:245], v[22:25]
	v_mfma_f32_16x16x128_f8f6f4 v[26:29], v[206:213], v[230:237], v[26:29]
	v_mfma_f32_16x16x128_f8f6f4 v[30:33], v[206:213], v[238:245], v[30:33]
	v_mfma_f32_16x16x128_f8f6f4 v[34:37], v[214:221], v[230:237], v[34:37]
	v_mfma_f32_16x16x128_f8f6f4 v[38:41], v[214:221], v[238:245], v[38:41]
	v_mfma_f32_16x16x128_f8f6f4 v[42:45], v[222:229], v[230:237], v[42:45]
	v_mfma_f32_16x16x128_f8f6f4 v[46:49], v[222:229], v[238:245], v[46:49]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[168:171], v200 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[206:209], v200 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[214:217], v200 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[222:225], v200 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[172:175], v201 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[210:213], v201 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[218:221], v201 offset:0x1000

	;;#ASMEND
	s_add_u32 s0, s4, 0xb80080
	v_readfirstlane_b32 s25, v202
	;;#ASMSTART
	ds_read_b128 v[226:229], v201 offset:0x1800

	;;#ASMEND
	s_addc_u32 s1, s5, 0
	s_mov_b32 m0, s25
	v_readfirstlane_b32 s26, v203
	buffer_load_dwordx4 v182, s[0:3], 0 offen lds
	s_mov_b32 m0, s26
	s_nop 0
	buffer_load_dwordx4 v181, s[0:3], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[50:53], v[168:175], v[2:9], v[50:53]
	v_mfma_f32_16x16x128_f8f6f4 v[54:57], v[168:175], v[10:17], v[54:57]
	v_mfma_f32_16x16x128_f8f6f4 v[58:61], v[206:213], v[2:9], v[58:61]
	v_mfma_f32_16x16x128_f8f6f4 v[62:65], v[206:213], v[10:17], v[62:65]
	v_mfma_f32_16x16x128_f8f6f4 v[66:69], v[214:221], v[2:9], v[66:69]
	v_mfma_f32_16x16x128_f8f6f4 v[70:73], v[214:221], v[10:17], v[70:73]
	v_mfma_f32_16x16x128_f8f6f4 v[74:77], v[222:229], v[2:9], v[74:77]
	v_mfma_f32_16x16x128_f8f6f4 v[78:81], v[222:229], v[10:17], v[78:81]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_u32 s0, s15, 0xb80
	v_readfirstlane_b32 s27, v204
	s_addc_u32 s1, s16, 0
	s_mov_b32 m0, s27
	v_readfirstlane_b32 s28, v205
	buffer_load_dwordx4 v179, s[0:3], 0 offen lds
	s_mov_b32 m0, s28
	s_nop 0
	buffer_load_dwordx4 v178, s[0:3], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[82:85], v[168:175], v[230:237], v[82:85]
	v_mfma_f32_16x16x128_f8f6f4 v[86:89], v[168:175], v[238:245], v[86:89]
	v_mfma_f32_16x16x128_f8f6f4 v[90:93], v[206:213], v[230:237], v[90:93]
	v_mfma_f32_16x16x128_f8f6f4 v[94:97], v[206:213], v[238:245], v[94:97]
	v_mfma_f32_16x16x128_f8f6f4 v[98:101], v[214:221], v[230:237], v[98:101]
	v_mfma_f32_16x16x128_f8f6f4 v[102:105], v[214:221], v[238:245], v[102:105]
	v_mfma_f32_16x16x128_f8f6f4 v[106:109], v[222:229], v[230:237], v[106:109]
	v_mfma_f32_16x16x128_f8f6f4 v[110:113], v[222:229], v[238:245], v[110:113]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v192 offset:0
ds_read_b64_tr_b8 v[4:5], v192 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v184 offset:0
ds_read_b64_tr_b8 v[12:13], v184 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v185 offset:0
ds_read_b64_tr_b8 v[8:9], v185 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v183 offset:0
ds_read_b64_tr_b8 v[16:17], v183 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[168:171], v193 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[206:209], v193 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[214:217], v193 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[222:225], v193 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[172:175], v194 offset:0

	;;#ASMEND
	s_or_b32 s0, s14, 0x80b80
	;;#ASMSTART
	ds_read_b128 v[210:213], v194 offset:0x800

	;;#ASMEND
	s_ashr_i32 s1, s0, 31
	;;#ASMSTART
	ds_read_b128 v[218:221], v194 offset:0x1000

	;;#ASMEND
	s_add_u32 s0, s10, s0
	v_readfirstlane_b32 s29, v195
	;;#ASMSTART
	ds_read_b128 v[226:229], v194 offset:0x1800

	;;#ASMEND
	s_addc_u32 s1, s11, s1
	s_mov_b32 m0, s29
	v_readfirstlane_b32 s30, v167
	buffer_load_dwordx4 v179, s[0:3], 0 offen lds
	s_mov_b32 m0, s30
	s_nop 0
	buffer_load_dwordx4 v178, s[0:3], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[114:117], v[168:175], v[2:9], v[114:117]
	v_mfma_f32_16x16x128_f8f6f4 v[118:121], v[168:175], v[10:17], v[118:121]
	v_mfma_f32_16x16x128_f8f6f4 v[122:125], v[206:213], v[2:9], v[122:125]
	v_mfma_f32_16x16x128_f8f6f4 v[126:129], v[206:213], v[10:17], v[126:129]
	v_mfma_f32_16x16x128_f8f6f4 v[130:133], v[214:221], v[2:9], v[130:133]
	v_mfma_f32_16x16x128_f8f6f4 v[134:137], v[214:221], v[10:17], v[134:137]
	v_mfma_f32_16x16x128_f8f6f4 v[138:141], v[222:229], v[2:9], v[138:141]
	v_mfma_f32_16x16x128_f8f6f4 v[142:145], v[222:229], v[10:17], v[142:145]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[230:231], v147 offset:0
ds_read_b64_tr_b8 v[232:233], v147 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[238:239], v148 offset:0
ds_read_b64_tr_b8 v[240:241], v148 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[234:235], v149 offset:0
ds_read_b64_tr_b8 v[236:237], v149 offset:1024

	;;#ASMEND
	s_add_u32 s0, s4, 0xc00000
	v_readfirstlane_b32 s31, v165
	;;#ASMSTART
	ds_read_b64_tr_b8 v[242:243], v150 offset:0
ds_read_b64_tr_b8 v[244:245], v150 offset:1024

	;;#ASMEND
	s_addc_u32 s1, s5, 0
	s_mov_b32 m0, s31
	v_readfirstlane_b32 s33, v166
	buffer_load_dwordx4 v182, s[0:3], 0 offen lds
	s_mov_b32 m0, s33
	s_nop 0
	buffer_load_dwordx4 v181, s[0:3], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[18:21], v[168:175], v[230:237], v[18:21]
	v_mfma_f32_16x16x128_f8f6f4 v[22:25], v[168:175], v[238:245], v[22:25]
	v_mfma_f32_16x16x128_f8f6f4 v[26:29], v[206:213], v[230:237], v[26:29]
	v_mfma_f32_16x16x128_f8f6f4 v[30:33], v[206:213], v[238:245], v[30:33]
	v_mfma_f32_16x16x128_f8f6f4 v[34:37], v[214:221], v[230:237], v[34:37]
	v_mfma_f32_16x16x128_f8f6f4 v[38:41], v[214:221], v[238:245], v[38:41]
	v_mfma_f32_16x16x128_f8f6f4 v[42:45], v[222:229], v[230:237], v[42:45]
	v_mfma_f32_16x16x128_f8f6f4 v[46:49], v[222:229], v[238:245], v[46:49]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[168:171], v196 offset:0

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
	ds_read_b128 v[172:175], v197 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[210:213], v197 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[218:221], v197 offset:0x1000

	;;#ASMEND
	s_add_u32 s0, s4, 0xc00080
	;;#ASMSTART
	ds_read_b128 v[226:229], v197 offset:0x1800

	;;#ASMEND
	s_addc_u32 s1, s5, 0
	s_mov_b32 m0, s17
	s_nop 0
	buffer_load_dwordx4 v182, s[0:3], 0 offen lds
	s_mov_b32 m0, s18
	s_nop 0
	buffer_load_dwordx4 v181, s[0:3], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[50:53], v[168:175], v[2:9], v[50:53]
	v_mfma_f32_16x16x128_f8f6f4 v[54:57], v[168:175], v[10:17], v[54:57]
	v_mfma_f32_16x16x128_f8f6f4 v[58:61], v[206:213], v[2:9], v[58:61]
	v_mfma_f32_16x16x128_f8f6f4 v[62:65], v[206:213], v[10:17], v[62:65]
	v_mfma_f32_16x16x128_f8f6f4 v[66:69], v[214:221], v[2:9], v[66:69]
	v_mfma_f32_16x16x128_f8f6f4 v[70:73], v[214:221], v[10:17], v[70:73]
	v_mfma_f32_16x16x128_f8f6f4 v[74:77], v[222:229], v[2:9], v[74:77]
	v_mfma_f32_16x16x128_f8f6f4 v[78:81], v[222:229], v[10:17], v[78:81]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_u32 s0, s15, 0xc00
	s_addc_u32 s1, s16, 0
	s_mov_b32 m0, s19
	s_nop 0
	buffer_load_dwordx4 v179, s[0:3], 0 offen lds
	s_mov_b32 m0, s20
	s_nop 0
	buffer_load_dwordx4 v178, s[0:3], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[82:85], v[168:175], v[230:237], v[82:85]
	v_mfma_f32_16x16x128_f8f6f4 v[86:89], v[168:175], v[238:245], v[86:89]
	v_mfma_f32_16x16x128_f8f6f4 v[90:93], v[206:213], v[230:237], v[90:93]
	v_mfma_f32_16x16x128_f8f6f4 v[94:97], v[206:213], v[238:245], v[94:97]
	v_mfma_f32_16x16x128_f8f6f4 v[98:101], v[214:221], v[230:237], v[98:101]
	v_mfma_f32_16x16x128_f8f6f4 v[102:105], v[214:221], v[238:245], v[102:105]
	v_mfma_f32_16x16x128_f8f6f4 v[106:109], v[222:229], v[230:237], v[106:109]
	v_mfma_f32_16x16x128_f8f6f4 v[110:113], v[222:229], v[238:245], v[110:113]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v154 offset:0
ds_read_b64_tr_b8 v[4:5], v154 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v155 offset:0
ds_read_b64_tr_b8 v[12:13], v155 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v156 offset:0
ds_read_b64_tr_b8 v[8:9], v156 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v157 offset:0
ds_read_b64_tr_b8 v[16:17], v157 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[168:171], v198 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[206:209], v198 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[214:217], v198 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[222:225], v198 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[172:175], v199 offset:0

	;;#ASMEND
	s_or_b32 s0, s14, 0x80c00
	;;#ASMSTART
	ds_read_b128 v[210:213], v199 offset:0x800

	;;#ASMEND
	s_ashr_i32 s1, s0, 31
	;;#ASMSTART
	ds_read_b128 v[218:221], v199 offset:0x1000

	;;#ASMEND
	s_add_u32 s0, s10, s0
	;;#ASMSTART
	ds_read_b128 v[226:229], v199 offset:0x1800

	;;#ASMEND
	s_addc_u32 s1, s11, s1
	s_mov_b32 m0, s21
	s_nop 0
	buffer_load_dwordx4 v179, s[0:3], 0 offen lds
	s_mov_b32 m0, s22
	s_nop 0
	buffer_load_dwordx4 v178, s[0:3], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[114:117], v[168:175], v[2:9], v[114:117]
	v_mfma_f32_16x16x128_f8f6f4 v[118:121], v[168:175], v[10:17], v[118:121]
	v_mfma_f32_16x16x128_f8f6f4 v[122:125], v[206:213], v[2:9], v[122:125]
	v_mfma_f32_16x16x128_f8f6f4 v[126:129], v[206:213], v[10:17], v[126:129]
	v_mfma_f32_16x16x128_f8f6f4 v[130:133], v[214:221], v[2:9], v[130:133]
	v_mfma_f32_16x16x128_f8f6f4 v[134:137], v[214:221], v[10:17], v[134:137]
	v_mfma_f32_16x16x128_f8f6f4 v[138:141], v[222:229], v[2:9], v[138:141]
	v_mfma_f32_16x16x128_f8f6f4 v[142:145], v[222:229], v[10:17], v[142:145]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[230:231], v159 offset:0
ds_read_b64_tr_b8 v[232:233], v159 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[238:239], v160 offset:0
ds_read_b64_tr_b8 v[240:241], v160 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[234:235], v161 offset:0
ds_read_b64_tr_b8 v[236:237], v161 offset:1024

	;;#ASMEND
	s_add_u32 s0, s4, 0xc80000
	;;#ASMSTART
	ds_read_b64_tr_b8 v[242:243], v162 offset:0
ds_read_b64_tr_b8 v[244:245], v162 offset:1024

	;;#ASMEND
	s_addc_u32 s1, s5, 0
	s_mov_b32 m0, s23
	s_nop 0
	buffer_load_dwordx4 v182, s[0:3], 0 offen lds
	s_mov_b32 m0, s24
	s_nop 0
	buffer_load_dwordx4 v181, s[0:3], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[18:21], v[168:175], v[230:237], v[18:21]
	v_mfma_f32_16x16x128_f8f6f4 v[22:25], v[168:175], v[238:245], v[22:25]
	v_mfma_f32_16x16x128_f8f6f4 v[26:29], v[206:213], v[230:237], v[26:29]
	v_mfma_f32_16x16x128_f8f6f4 v[30:33], v[206:213], v[238:245], v[30:33]
	v_mfma_f32_16x16x128_f8f6f4 v[34:37], v[214:221], v[230:237], v[34:37]
	v_mfma_f32_16x16x128_f8f6f4 v[38:41], v[214:221], v[238:245], v[38:41]
	v_mfma_f32_16x16x128_f8f6f4 v[42:45], v[222:229], v[230:237], v[42:45]
	v_mfma_f32_16x16x128_f8f6f4 v[46:49], v[222:229], v[238:245], v[46:49]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[168:171], v200 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[206:209], v200 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[214:217], v200 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[222:225], v200 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[172:175], v201 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[210:213], v201 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[218:221], v201 offset:0x1000

	;;#ASMEND
	s_add_u32 s0, s4, 0xc80080
	;;#ASMSTART
	ds_read_b128 v[226:229], v201 offset:0x1800

	;;#ASMEND
	s_addc_u32 s1, s5, 0
	s_mov_b32 m0, s25
	s_nop 0
	buffer_load_dwordx4 v182, s[0:3], 0 offen lds
	s_mov_b32 m0, s26
	s_nop 0
	buffer_load_dwordx4 v181, s[0:3], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[50:53], v[168:175], v[2:9], v[50:53]
	v_mfma_f32_16x16x128_f8f6f4 v[54:57], v[168:175], v[10:17], v[54:57]
	v_mfma_f32_16x16x128_f8f6f4 v[58:61], v[206:213], v[2:9], v[58:61]
	v_mfma_f32_16x16x128_f8f6f4 v[62:65], v[206:213], v[10:17], v[62:65]
	v_mfma_f32_16x16x128_f8f6f4 v[66:69], v[214:221], v[2:9], v[66:69]
	v_mfma_f32_16x16x128_f8f6f4 v[70:73], v[214:221], v[10:17], v[70:73]
	v_mfma_f32_16x16x128_f8f6f4 v[74:77], v[222:229], v[2:9], v[74:77]
	v_mfma_f32_16x16x128_f8f6f4 v[78:81], v[222:229], v[10:17], v[78:81]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_u32 s0, s15, 0xc80
	s_addc_u32 s1, s16, 0
	s_mov_b32 m0, s27
	s_nop 0
	buffer_load_dwordx4 v179, s[0:3], 0 offen lds
	s_mov_b32 m0, s28
	s_nop 0
	buffer_load_dwordx4 v178, s[0:3], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[82:85], v[168:175], v[230:237], v[82:85]
	v_mfma_f32_16x16x128_f8f6f4 v[86:89], v[168:175], v[238:245], v[86:89]
	v_mfma_f32_16x16x128_f8f6f4 v[90:93], v[206:213], v[230:237], v[90:93]
	v_mfma_f32_16x16x128_f8f6f4 v[94:97], v[206:213], v[238:245], v[94:97]
	v_mfma_f32_16x16x128_f8f6f4 v[98:101], v[214:221], v[230:237], v[98:101]
	v_mfma_f32_16x16x128_f8f6f4 v[102:105], v[214:221], v[238:245], v[102:105]
	v_mfma_f32_16x16x128_f8f6f4 v[106:109], v[222:229], v[230:237], v[106:109]
	v_mfma_f32_16x16x128_f8f6f4 v[110:113], v[222:229], v[238:245], v[110:113]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v192 offset:0
ds_read_b64_tr_b8 v[4:5], v192 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v184 offset:0
ds_read_b64_tr_b8 v[12:13], v184 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v185 offset:0
ds_read_b64_tr_b8 v[8:9], v185 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v183 offset:0
ds_read_b64_tr_b8 v[16:17], v183 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[168:171], v193 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[206:209], v193 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[214:217], v193 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[222:225], v193 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[172:175], v194 offset:0

	;;#ASMEND
	s_or_b32 s0, s14, 0x80c80
	;;#ASMSTART
	ds_read_b128 v[210:213], v194 offset:0x800

	;;#ASMEND
	s_ashr_i32 s1, s0, 31
	;;#ASMSTART
	ds_read_b128 v[218:221], v194 offset:0x1000

	;;#ASMEND
	s_add_u32 s0, s10, s0
	;;#ASMSTART
	ds_read_b128 v[226:229], v194 offset:0x1800

	;;#ASMEND
	s_addc_u32 s1, s11, s1
	s_mov_b32 m0, s29
	s_nop 0
	buffer_load_dwordx4 v179, s[0:3], 0 offen lds
	s_mov_b32 m0, s30
	s_nop 0
	buffer_load_dwordx4 v178, s[0:3], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[114:117], v[168:175], v[2:9], v[114:117]
	v_mfma_f32_16x16x128_f8f6f4 v[118:121], v[168:175], v[10:17], v[118:121]
	v_mfma_f32_16x16x128_f8f6f4 v[122:125], v[206:213], v[2:9], v[122:125]
	v_mfma_f32_16x16x128_f8f6f4 v[126:129], v[206:213], v[10:17], v[126:129]
	v_mfma_f32_16x16x128_f8f6f4 v[130:133], v[214:221], v[2:9], v[130:133]
	v_mfma_f32_16x16x128_f8f6f4 v[134:137], v[214:221], v[10:17], v[134:137]
	v_mfma_f32_16x16x128_f8f6f4 v[138:141], v[222:229], v[2:9], v[138:141]
	v_mfma_f32_16x16x128_f8f6f4 v[142:145], v[222:229], v[10:17], v[142:145]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[230:231], v147 offset:0
ds_read_b64_tr_b8 v[232:233], v147 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[238:239], v148 offset:0
ds_read_b64_tr_b8 v[240:241], v148 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[234:235], v149 offset:0
ds_read_b64_tr_b8 v[236:237], v149 offset:1024

	;;#ASMEND
	s_add_u32 s0, s4, 0xd00000
	;;#ASMSTART
	ds_read_b64_tr_b8 v[242:243], v150 offset:0
ds_read_b64_tr_b8 v[244:245], v150 offset:1024

	;;#ASMEND
	s_addc_u32 s1, s5, 0
	s_mov_b32 m0, s31
	s_nop 0
	buffer_load_dwordx4 v182, s[0:3], 0 offen lds
	s_mov_b32 m0, s33
	s_nop 0
	buffer_load_dwordx4 v181, s[0:3], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[18:21], v[168:175], v[230:237], v[18:21]
	v_mfma_f32_16x16x128_f8f6f4 v[22:25], v[168:175], v[238:245], v[22:25]
	v_mfma_f32_16x16x128_f8f6f4 v[26:29], v[206:213], v[230:237], v[26:29]
	v_mfma_f32_16x16x128_f8f6f4 v[30:33], v[206:213], v[238:245], v[30:33]
	v_mfma_f32_16x16x128_f8f6f4 v[34:37], v[214:221], v[230:237], v[34:37]
	v_mfma_f32_16x16x128_f8f6f4 v[38:41], v[214:221], v[238:245], v[38:41]
	v_mfma_f32_16x16x128_f8f6f4 v[42:45], v[222:229], v[230:237], v[42:45]
	v_mfma_f32_16x16x128_f8f6f4 v[46:49], v[222:229], v[238:245], v[46:49]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[168:171], v196 offset:0

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
	ds_read_b128 v[172:175], v197 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[210:213], v197 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[218:221], v197 offset:0x1000

	;;#ASMEND
	s_add_u32 s0, s4, 0xd00080
	v_readfirstlane_b32 s17, v151
	;;#ASMSTART
	ds_read_b128 v[226:229], v197 offset:0x1800

	;;#ASMEND
	s_addc_u32 s1, s5, 0
	s_mov_b32 m0, s17
	v_readfirstlane_b32 s18, v152
	buffer_load_dwordx4 v182, s[0:3], 0 offen lds
	s_mov_b32 m0, s18
	s_nop 0
	buffer_load_dwordx4 v181, s[0:3], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[50:53], v[168:175], v[2:9], v[50:53]
	v_mfma_f32_16x16x128_f8f6f4 v[54:57], v[168:175], v[10:17], v[54:57]
	v_mfma_f32_16x16x128_f8f6f4 v[58:61], v[206:213], v[2:9], v[58:61]
	v_mfma_f32_16x16x128_f8f6f4 v[62:65], v[206:213], v[10:17], v[62:65]
	v_mfma_f32_16x16x128_f8f6f4 v[66:69], v[214:221], v[2:9], v[66:69]
	v_mfma_f32_16x16x128_f8f6f4 v[70:73], v[214:221], v[10:17], v[70:73]
	v_mfma_f32_16x16x128_f8f6f4 v[74:77], v[222:229], v[2:9], v[74:77]
	v_mfma_f32_16x16x128_f8f6f4 v[78:81], v[222:229], v[10:17], v[78:81]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_u32 s0, s15, 0xd00
	v_readfirstlane_b32 s19, v180
	s_addc_u32 s1, s16, 0
	s_mov_b32 m0, s19
	v_readfirstlane_b32 s20, v153
	buffer_load_dwordx4 v179, s[0:3], 0 offen lds
	s_mov_b32 m0, s20
	s_nop 0
	buffer_load_dwordx4 v178, s[0:3], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[82:85], v[168:175], v[230:237], v[82:85]
	v_mfma_f32_16x16x128_f8f6f4 v[86:89], v[168:175], v[238:245], v[86:89]
	v_mfma_f32_16x16x128_f8f6f4 v[90:93], v[206:213], v[230:237], v[90:93]
	v_mfma_f32_16x16x128_f8f6f4 v[94:97], v[206:213], v[238:245], v[94:97]
	v_mfma_f32_16x16x128_f8f6f4 v[98:101], v[214:221], v[230:237], v[98:101]
	v_mfma_f32_16x16x128_f8f6f4 v[102:105], v[214:221], v[238:245], v[102:105]
	v_mfma_f32_16x16x128_f8f6f4 v[106:109], v[222:229], v[230:237], v[106:109]
	v_mfma_f32_16x16x128_f8f6f4 v[110:113], v[222:229], v[238:245], v[110:113]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v154 offset:0
ds_read_b64_tr_b8 v[4:5], v154 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v155 offset:0
ds_read_b64_tr_b8 v[12:13], v155 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v156 offset:0
ds_read_b64_tr_b8 v[8:9], v156 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v157 offset:0
ds_read_b64_tr_b8 v[16:17], v157 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[168:171], v198 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[206:209], v198 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[214:217], v198 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[222:225], v198 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[172:175], v199 offset:0

	;;#ASMEND
	s_or_b32 s0, s14, 0x80d00
	;;#ASMSTART
	ds_read_b128 v[210:213], v199 offset:0x800

	;;#ASMEND
	s_ashr_i32 s1, s0, 31
	;;#ASMSTART
	ds_read_b128 v[218:221], v199 offset:0x1000

	;;#ASMEND
	s_add_u32 s0, s10, s0
	v_readfirstlane_b32 s21, v146
	;;#ASMSTART
	ds_read_b128 v[226:229], v199 offset:0x1800

	;;#ASMEND
	s_addc_u32 s1, s11, s1
	s_mov_b32 m0, s21
	v_readfirstlane_b32 s22, v158
	buffer_load_dwordx4 v179, s[0:3], 0 offen lds
	s_mov_b32 m0, s22
	s_nop 0
	buffer_load_dwordx4 v178, s[0:3], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[114:117], v[168:175], v[2:9], v[114:117]
	v_mfma_f32_16x16x128_f8f6f4 v[118:121], v[168:175], v[10:17], v[118:121]
	v_mfma_f32_16x16x128_f8f6f4 v[122:125], v[206:213], v[2:9], v[122:125]
	v_mfma_f32_16x16x128_f8f6f4 v[126:129], v[206:213], v[10:17], v[126:129]
	v_mfma_f32_16x16x128_f8f6f4 v[130:133], v[214:221], v[2:9], v[130:133]
	v_mfma_f32_16x16x128_f8f6f4 v[134:137], v[214:221], v[10:17], v[134:137]
	v_mfma_f32_16x16x128_f8f6f4 v[138:141], v[222:229], v[2:9], v[138:141]
	v_mfma_f32_16x16x128_f8f6f4 v[142:145], v[222:229], v[10:17], v[142:145]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[230:231], v159 offset:0
ds_read_b64_tr_b8 v[232:233], v159 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[238:239], v160 offset:0
ds_read_b64_tr_b8 v[240:241], v160 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[234:235], v161 offset:0
ds_read_b64_tr_b8 v[236:237], v161 offset:1024

	;;#ASMEND
	s_add_u32 s0, s4, 0xd80000
	v_readfirstlane_b32 s23, v163
	;;#ASMSTART
	ds_read_b64_tr_b8 v[242:243], v162 offset:0
ds_read_b64_tr_b8 v[244:245], v162 offset:1024

	;;#ASMEND
	s_addc_u32 s1, s5, 0
	s_mov_b32 m0, s23
	v_readfirstlane_b32 s24, v164
	buffer_load_dwordx4 v182, s[0:3], 0 offen lds
	s_mov_b32 m0, s24
	s_nop 0
	buffer_load_dwordx4 v181, s[0:3], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[18:21], v[168:175], v[230:237], v[18:21]
	v_mfma_f32_16x16x128_f8f6f4 v[22:25], v[168:175], v[238:245], v[22:25]
	v_mfma_f32_16x16x128_f8f6f4 v[26:29], v[206:213], v[230:237], v[26:29]
	v_mfma_f32_16x16x128_f8f6f4 v[30:33], v[206:213], v[238:245], v[30:33]
	v_mfma_f32_16x16x128_f8f6f4 v[34:37], v[214:221], v[230:237], v[34:37]
	v_mfma_f32_16x16x128_f8f6f4 v[38:41], v[214:221], v[238:245], v[38:41]
	v_mfma_f32_16x16x128_f8f6f4 v[42:45], v[222:229], v[230:237], v[42:45]
	v_mfma_f32_16x16x128_f8f6f4 v[46:49], v[222:229], v[238:245], v[46:49]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[168:171], v200 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[206:209], v200 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[214:217], v200 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[222:225], v200 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[172:175], v201 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[210:213], v201 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[218:221], v201 offset:0x1000

	;;#ASMEND
	s_add_u32 s0, s4, 0xd80080
	v_readfirstlane_b32 s25, v202
	;;#ASMSTART
	ds_read_b128 v[226:229], v201 offset:0x1800

	;;#ASMEND
	s_addc_u32 s1, s5, 0
	s_mov_b32 m0, s25
	v_readfirstlane_b32 s26, v203
	buffer_load_dwordx4 v182, s[0:3], 0 offen lds
	s_mov_b32 m0, s26
	s_nop 0
	buffer_load_dwordx4 v181, s[0:3], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[50:53], v[168:175], v[2:9], v[50:53]
	v_mfma_f32_16x16x128_f8f6f4 v[54:57], v[168:175], v[10:17], v[54:57]
	v_mfma_f32_16x16x128_f8f6f4 v[58:61], v[206:213], v[2:9], v[58:61]
	v_mfma_f32_16x16x128_f8f6f4 v[62:65], v[206:213], v[10:17], v[62:65]
	v_mfma_f32_16x16x128_f8f6f4 v[66:69], v[214:221], v[2:9], v[66:69]
	v_mfma_f32_16x16x128_f8f6f4 v[70:73], v[214:221], v[10:17], v[70:73]
	v_mfma_f32_16x16x128_f8f6f4 v[74:77], v[222:229], v[2:9], v[74:77]
	v_mfma_f32_16x16x128_f8f6f4 v[78:81], v[222:229], v[10:17], v[78:81]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_u32 s0, s15, 0xd80
	v_readfirstlane_b32 s27, v204
	s_addc_u32 s1, s16, 0
	s_mov_b32 m0, s27
	v_readfirstlane_b32 s28, v205
	buffer_load_dwordx4 v179, s[0:3], 0 offen lds
	s_mov_b32 m0, s28
	s_nop 0
	buffer_load_dwordx4 v178, s[0:3], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[82:85], v[168:175], v[230:237], v[82:85]
	v_mfma_f32_16x16x128_f8f6f4 v[86:89], v[168:175], v[238:245], v[86:89]
	v_mfma_f32_16x16x128_f8f6f4 v[90:93], v[206:213], v[230:237], v[90:93]
	v_mfma_f32_16x16x128_f8f6f4 v[94:97], v[206:213], v[238:245], v[94:97]
	v_mfma_f32_16x16x128_f8f6f4 v[98:101], v[214:221], v[230:237], v[98:101]
	v_mfma_f32_16x16x128_f8f6f4 v[102:105], v[214:221], v[238:245], v[102:105]
	v_mfma_f32_16x16x128_f8f6f4 v[106:109], v[222:229], v[230:237], v[106:109]
	v_mfma_f32_16x16x128_f8f6f4 v[110:113], v[222:229], v[238:245], v[110:113]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v192 offset:0
ds_read_b64_tr_b8 v[4:5], v192 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v184 offset:0
ds_read_b64_tr_b8 v[12:13], v184 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v185 offset:0
ds_read_b64_tr_b8 v[8:9], v185 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v183 offset:0
ds_read_b64_tr_b8 v[16:17], v183 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[168:171], v193 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[206:209], v193 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[214:217], v193 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[222:225], v193 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[172:175], v194 offset:0

	;;#ASMEND
	s_or_b32 s0, s14, 0x80d80
	;;#ASMSTART
	ds_read_b128 v[210:213], v194 offset:0x800

	;;#ASMEND
	s_ashr_i32 s1, s0, 31
	;;#ASMSTART
	ds_read_b128 v[218:221], v194 offset:0x1000

	;;#ASMEND
	s_add_u32 s0, s10, s0
	v_readfirstlane_b32 s29, v195
	;;#ASMSTART
	ds_read_b128 v[226:229], v194 offset:0x1800

	;;#ASMEND
	s_addc_u32 s1, s11, s1
	s_mov_b32 m0, s29
	v_readfirstlane_b32 s30, v167
	buffer_load_dwordx4 v179, s[0:3], 0 offen lds
	s_mov_b32 m0, s30
	s_nop 0
	buffer_load_dwordx4 v178, s[0:3], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[114:117], v[168:175], v[2:9], v[114:117]
	v_mfma_f32_16x16x128_f8f6f4 v[118:121], v[168:175], v[10:17], v[118:121]
	v_mfma_f32_16x16x128_f8f6f4 v[122:125], v[206:213], v[2:9], v[122:125]
	v_mfma_f32_16x16x128_f8f6f4 v[126:129], v[206:213], v[10:17], v[126:129]
	v_mfma_f32_16x16x128_f8f6f4 v[130:133], v[214:221], v[2:9], v[130:133]
	v_mfma_f32_16x16x128_f8f6f4 v[134:137], v[214:221], v[10:17], v[134:137]
	v_mfma_f32_16x16x128_f8f6f4 v[138:141], v[222:229], v[2:9], v[138:141]
	v_mfma_f32_16x16x128_f8f6f4 v[142:145], v[222:229], v[10:17], v[142:145]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[230:231], v147 offset:0
ds_read_b64_tr_b8 v[232:233], v147 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[238:239], v148 offset:0
ds_read_b64_tr_b8 v[240:241], v148 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[234:235], v149 offset:0
ds_read_b64_tr_b8 v[236:237], v149 offset:1024

	;;#ASMEND
	s_add_u32 s0, s4, 0xe00000
	v_readfirstlane_b32 s31, v165
	;;#ASMSTART
	ds_read_b64_tr_b8 v[242:243], v150 offset:0
ds_read_b64_tr_b8 v[244:245], v150 offset:1024

	;;#ASMEND
	s_addc_u32 s1, s5, 0
	s_mov_b32 m0, s31
	v_readfirstlane_b32 s33, v166
	buffer_load_dwordx4 v182, s[0:3], 0 offen lds
	s_mov_b32 m0, s33
	s_nop 0
	buffer_load_dwordx4 v181, s[0:3], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[18:21], v[168:175], v[230:237], v[18:21]
	v_mfma_f32_16x16x128_f8f6f4 v[22:25], v[168:175], v[238:245], v[22:25]
	v_mfma_f32_16x16x128_f8f6f4 v[26:29], v[206:213], v[230:237], v[26:29]
	v_mfma_f32_16x16x128_f8f6f4 v[30:33], v[206:213], v[238:245], v[30:33]
	v_mfma_f32_16x16x128_f8f6f4 v[34:37], v[214:221], v[230:237], v[34:37]
	v_mfma_f32_16x16x128_f8f6f4 v[38:41], v[214:221], v[238:245], v[38:41]
	v_mfma_f32_16x16x128_f8f6f4 v[42:45], v[222:229], v[230:237], v[42:45]
	v_mfma_f32_16x16x128_f8f6f4 v[46:49], v[222:229], v[238:245], v[46:49]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[166:169], v196 offset:0

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
	ds_read_b128 v[170:173], v197 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[210:213], v197 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[218:221], v197 offset:0x1000

	;;#ASMEND
	s_add_u32 s0, s4, 0xe00080
	;;#ASMSTART
	ds_read_b128 v[226:229], v197 offset:0x1800

	;;#ASMEND
	s_addc_u32 s1, s5, 0
	s_mov_b32 m0, s17
	s_nop 0
	buffer_load_dwordx4 v182, s[0:3], 0 offen lds
	s_mov_b32 m0, s18
	s_nop 0
	buffer_load_dwordx4 v181, s[0:3], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[50:53], v[166:173], v[2:9], v[50:53]
	v_mfma_f32_16x16x128_f8f6f4 v[54:57], v[166:173], v[10:17], v[54:57]
	v_mfma_f32_16x16x128_f8f6f4 v[58:61], v[206:213], v[2:9], v[58:61]
	v_mfma_f32_16x16x128_f8f6f4 v[62:65], v[206:213], v[10:17], v[62:65]
	v_mfma_f32_16x16x128_f8f6f4 v[66:69], v[214:221], v[2:9], v[66:69]
	v_mfma_f32_16x16x128_f8f6f4 v[70:73], v[214:221], v[10:17], v[70:73]
	v_mfma_f32_16x16x128_f8f6f4 v[74:77], v[222:229], v[2:9], v[74:77]
	v_mfma_f32_16x16x128_f8f6f4 v[78:81], v[222:229], v[10:17], v[78:81]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_u32 s0, s15, 0xe00
	s_addc_u32 s1, s16, 0
	s_mov_b32 m0, s19
	s_nop 0
	buffer_load_dwordx4 v179, s[0:3], 0 offen lds
	s_mov_b32 m0, s20
	s_nop 0
	buffer_load_dwordx4 v178, s[0:3], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[82:85], v[166:173], v[230:237], v[82:85]
	v_mfma_f32_16x16x128_f8f6f4 v[86:89], v[166:173], v[238:245], v[86:89]
	v_mfma_f32_16x16x128_f8f6f4 v[90:93], v[206:213], v[230:237], v[90:93]
	v_mfma_f32_16x16x128_f8f6f4 v[94:97], v[206:213], v[238:245], v[94:97]
	v_mfma_f32_16x16x128_f8f6f4 v[98:101], v[214:221], v[230:237], v[98:101]
	v_mfma_f32_16x16x128_f8f6f4 v[102:105], v[214:221], v[238:245], v[102:105]
	v_mfma_f32_16x16x128_f8f6f4 v[106:109], v[222:229], v[230:237], v[106:109]
	v_mfma_f32_16x16x128_f8f6f4 v[110:113], v[222:229], v[238:245], v[110:113]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v154 offset:0
ds_read_b64_tr_b8 v[4:5], v154 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v155 offset:0
ds_read_b64_tr_b8 v[12:13], v155 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v156 offset:0
ds_read_b64_tr_b8 v[8:9], v156 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v157 offset:0
ds_read_b64_tr_b8 v[16:17], v157 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[166:169], v198 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[206:209], v198 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[214:217], v198 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[222:225], v198 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[170:173], v199 offset:0

	;;#ASMEND
	s_or_b32 s0, s14, 0x80e00
	;;#ASMSTART
	ds_read_b128 v[210:213], v199 offset:0x800

	;;#ASMEND
	s_ashr_i32 s1, s0, 31
	;;#ASMSTART
	ds_read_b128 v[218:221], v199 offset:0x1000

	;;#ASMEND
	s_add_u32 s0, s10, s0
	;;#ASMSTART
	ds_read_b128 v[226:229], v199 offset:0x1800

	;;#ASMEND
	s_addc_u32 s1, s11, s1
	s_mov_b32 m0, s21
	s_nop 0
	buffer_load_dwordx4 v179, s[0:3], 0 offen lds
	s_mov_b32 m0, s22
	s_nop 0
	buffer_load_dwordx4 v178, s[0:3], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[114:117], v[166:173], v[2:9], v[114:117]
	v_mfma_f32_16x16x128_f8f6f4 v[118:121], v[166:173], v[10:17], v[118:121]
	v_mfma_f32_16x16x128_f8f6f4 v[122:125], v[206:213], v[2:9], v[122:125]
	v_mfma_f32_16x16x128_f8f6f4 v[126:129], v[206:213], v[10:17], v[126:129]
	v_mfma_f32_16x16x128_f8f6f4 v[130:133], v[214:221], v[2:9], v[130:133]
	v_mfma_f32_16x16x128_f8f6f4 v[134:137], v[214:221], v[10:17], v[134:137]
	v_mfma_f32_16x16x128_f8f6f4 v[138:141], v[222:229], v[2:9], v[138:141]
	v_mfma_f32_16x16x128_f8f6f4 v[142:145], v[222:229], v[10:17], v[142:145]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[230:231], v159 offset:0
ds_read_b64_tr_b8 v[232:233], v159 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[238:239], v160 offset:0
ds_read_b64_tr_b8 v[240:241], v160 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[234:235], v161 offset:0
ds_read_b64_tr_b8 v[236:237], v161 offset:1024

	;;#ASMEND
	s_add_u32 s0, s4, 0xe80000
	;;#ASMSTART
	ds_read_b64_tr_b8 v[242:243], v162 offset:0
ds_read_b64_tr_b8 v[244:245], v162 offset:1024

	;;#ASMEND
	s_addc_u32 s1, s5, 0
	s_mov_b32 m0, s23
	s_nop 0
	buffer_load_dwordx4 v182, s[0:3], 0 offen lds
	s_mov_b32 m0, s24
	s_nop 0
	buffer_load_dwordx4 v181, s[0:3], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[18:21], v[166:173], v[230:237], v[18:21]
	v_mfma_f32_16x16x128_f8f6f4 v[22:25], v[166:173], v[238:245], v[22:25]
	v_mfma_f32_16x16x128_f8f6f4 v[26:29], v[206:213], v[230:237], v[26:29]
	v_mfma_f32_16x16x128_f8f6f4 v[30:33], v[206:213], v[238:245], v[30:33]
	v_mfma_f32_16x16x128_f8f6f4 v[34:37], v[214:221], v[230:237], v[34:37]
	v_mfma_f32_16x16x128_f8f6f4 v[38:41], v[214:221], v[238:245], v[38:41]
	v_mfma_f32_16x16x128_f8f6f4 v[42:45], v[222:229], v[230:237], v[42:45]
	v_mfma_f32_16x16x128_f8f6f4 v[46:49], v[222:229], v[238:245], v[46:49]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[166:169], v200 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[206:209], v200 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[214:217], v200 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[222:225], v200 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[170:173], v201 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[210:213], v201 offset:0x800

	;;#ASMEND
	s_add_u32 s0, s4, 0xe80080
	;;#ASMSTART
	ds_read_b128 v[218:221], v201 offset:0x1000

	;;#ASMEND
	s_addc_u32 s1, s5, 0
	s_mov_b64 s[22:23], s[2:3]
	;;#ASMSTART
	ds_read_b128 v[226:229], v201 offset:0x1800

	;;#ASMEND
	s_mov_b64 s[20:21], s[0:1]
	s_mov_b32 m0, s25
	s_nop 0
	buffer_load_dwordx4 v182, s[20:23], 0 offen lds
	s_mov_b32 m0, s26
	s_nop 0
	buffer_load_dwordx4 v181, s[20:23], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[50:53], v[166:173], v[2:9], v[50:53]
	v_mfma_f32_16x16x128_f8f6f4 v[54:57], v[166:173], v[10:17], v[54:57]
	v_mfma_f32_16x16x128_f8f6f4 v[58:61], v[206:213], v[2:9], v[58:61]
	v_mfma_f32_16x16x128_f8f6f4 v[62:65], v[206:213], v[10:17], v[62:65]
	v_mfma_f32_16x16x128_f8f6f4 v[66:69], v[214:221], v[2:9], v[66:69]
	v_mfma_f32_16x16x128_f8f6f4 v[70:73], v[214:221], v[10:17], v[70:73]
	v_mfma_f32_16x16x128_f8f6f4 v[74:77], v[222:229], v[2:9], v[74:77]
	v_mfma_f32_16x16x128_f8f6f4 v[78:81], v[222:229], v[10:17], v[78:81]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_u32 s0, s15, 0xe80
	s_addc_u32 s1, s16, 0
	s_mov_b64 s[22:23], s[2:3]
	s_mov_b64 s[20:21], s[0:1]
	s_mov_b32 m0, s27
	s_nop 0
	buffer_load_dwordx4 v179, s[20:23], 0 offen lds
	s_mov_b32 m0, s28
	s_nop 0
	buffer_load_dwordx4 v178, s[20:23], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[82:85], v[166:173], v[230:237], v[82:85]
	v_mfma_f32_16x16x128_f8f6f4 v[86:89], v[166:173], v[238:245], v[86:89]
	v_mfma_f32_16x16x128_f8f6f4 v[90:93], v[206:213], v[230:237], v[90:93]
	v_mfma_f32_16x16x128_f8f6f4 v[94:97], v[206:213], v[238:245], v[94:97]
	v_mfma_f32_16x16x128_f8f6f4 v[98:101], v[214:221], v[230:237], v[98:101]
	v_mfma_f32_16x16x128_f8f6f4 v[102:105], v[214:221], v[238:245], v[102:105]
	v_mfma_f32_16x16x128_f8f6f4 v[106:109], v[222:229], v[230:237], v[106:109]
	v_mfma_f32_16x16x128_f8f6f4 v[110:113], v[222:229], v[238:245], v[110:113]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v192 offset:0
ds_read_b64_tr_b8 v[4:5], v192 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v184 offset:0
ds_read_b64_tr_b8 v[12:13], v184 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v185 offset:0
ds_read_b64_tr_b8 v[8:9], v185 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v183 offset:0
ds_read_b64_tr_b8 v[16:17], v183 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[166:169], v193 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[206:209], v193 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[214:217], v193 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[222:225], v193 offset:0x1800

	;;#ASMEND
	s_or_b32 s0, s14, 0x80e80
	;;#ASMSTART
	ds_read_b128 v[170:173], v194 offset:0

	;;#ASMEND
	s_ashr_i32 s1, s0, 31
	;;#ASMSTART
	ds_read_b128 v[210:213], v194 offset:0x800

	;;#ASMEND
	s_add_u32 s0, s10, s0
	;;#ASMSTART
	ds_read_b128 v[218:221], v194 offset:0x1000

	;;#ASMEND
	s_addc_u32 s1, s11, s1
	s_mov_b64 s[22:23], s[2:3]
	;;#ASMSTART
	ds_read_b128 v[226:229], v194 offset:0x1800

	;;#ASMEND
	s_mov_b64 s[20:21], s[0:1]
	s_mov_b32 m0, s29
	s_nop 0
	buffer_load_dwordx4 v179, s[20:23], 0 offen lds
	s_mov_b32 m0, s30
	s_nop 0
	buffer_load_dwordx4 v178, s[20:23], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[114:117], v[166:173], v[2:9], v[114:117]
	v_mfma_f32_16x16x128_f8f6f4 v[118:121], v[166:173], v[10:17], v[118:121]
	v_mfma_f32_16x16x128_f8f6f4 v[122:125], v[206:213], v[2:9], v[122:125]
	v_mfma_f32_16x16x128_f8f6f4 v[126:129], v[206:213], v[10:17], v[126:129]
	v_mfma_f32_16x16x128_f8f6f4 v[130:133], v[214:221], v[2:9], v[130:133]
	v_mfma_f32_16x16x128_f8f6f4 v[134:137], v[214:221], v[10:17], v[134:137]
	v_mfma_f32_16x16x128_f8f6f4 v[138:141], v[222:229], v[2:9], v[138:141]
	v_mfma_f32_16x16x128_f8f6f4 v[142:145], v[222:229], v[10:17], v[142:145]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[230:231], v147 offset:0
ds_read_b64_tr_b8 v[232:233], v147 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[238:239], v148 offset:0
ds_read_b64_tr_b8 v[240:241], v148 offset:1024

	;;#ASMEND
	s_add_u32 s0, s4, 0xf00000
	;;#ASMSTART
	ds_read_b64_tr_b8 v[234:235], v149 offset:0
ds_read_b64_tr_b8 v[236:237], v149 offset:1024

	;;#ASMEND
	s_addc_u32 s1, s5, 0
	s_mov_b64 s[22:23], s[2:3]
	;;#ASMSTART
	ds_read_b64_tr_b8 v[242:243], v150 offset:0
ds_read_b64_tr_b8 v[244:245], v150 offset:1024

	;;#ASMEND
	s_mov_b64 s[20:21], s[0:1]
	s_mov_b32 m0, s31
	s_nop 0
	buffer_load_dwordx4 v182, s[20:23], 0 offen lds
	s_mov_b32 m0, s33
	s_nop 0
	buffer_load_dwordx4 v181, s[20:23], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[18:21], v[166:173], v[230:237], v[18:21]
	v_mfma_f32_16x16x128_f8f6f4 v[22:25], v[166:173], v[238:245], v[22:25]
	v_mfma_f32_16x16x128_f8f6f4 v[26:29], v[206:213], v[230:237], v[26:29]
	v_mfma_f32_16x16x128_f8f6f4 v[30:33], v[206:213], v[238:245], v[30:33]
	v_mfma_f32_16x16x128_f8f6f4 v[34:37], v[214:221], v[230:237], v[34:37]
	v_mfma_f32_16x16x128_f8f6f4 v[38:41], v[214:221], v[238:245], v[38:41]
	v_mfma_f32_16x16x128_f8f6f4 v[42:45], v[222:229], v[230:237], v[42:45]
	v_mfma_f32_16x16x128_f8f6f4 v[46:49], v[222:229], v[238:245], v[46:49]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[166:169], v196 offset:0

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
	ds_read_b128 v[170:173], v197 offset:0

	;;#ASMEND
	s_add_u32 s0, s4, 0xf00080
	;;#ASMSTART
	ds_read_b128 v[210:213], v197 offset:0x800

	;;#ASMEND
	s_addc_u32 s1, s5, 0
	s_mov_b64 s[22:23], s[2:3]
	;;#ASMSTART
	ds_read_b128 v[218:221], v197 offset:0x1000

	;;#ASMEND
	s_mov_b64 s[20:21], s[0:1]
	v_readfirstlane_b32 s0, v151
	;;#ASMSTART
	ds_read_b128 v[226:229], v197 offset:0x1800

	;;#ASMEND
	s_mov_b32 m0, s0
	v_readfirstlane_b32 s0, v152
	buffer_load_dwordx4 v182, s[20:23], 0 offen lds
	s_mov_b32 m0, s0
	s_nop 0
	buffer_load_dwordx4 v181, s[20:23], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[50:53], v[166:173], v[2:9], v[50:53]
	v_mfma_f32_16x16x128_f8f6f4 v[54:57], v[166:173], v[10:17], v[54:57]
	v_mfma_f32_16x16x128_f8f6f4 v[58:61], v[206:213], v[2:9], v[58:61]
	v_mfma_f32_16x16x128_f8f6f4 v[62:65], v[206:213], v[10:17], v[62:65]
	v_mfma_f32_16x16x128_f8f6f4 v[66:69], v[214:221], v[2:9], v[66:69]
	v_mfma_f32_16x16x128_f8f6f4 v[70:73], v[214:221], v[10:17], v[70:73]
	v_mfma_f32_16x16x128_f8f6f4 v[74:77], v[222:229], v[2:9], v[74:77]
	v_mfma_f32_16x16x128_f8f6f4 v[78:81], v[222:229], v[10:17], v[78:81]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_u32 s0, s15, 0xf00
	s_addc_u32 s1, s16, 0
	s_mov_b64 s[22:23], s[2:3]
	s_mov_b64 s[20:21], s[0:1]
	v_readfirstlane_b32 s0, v180
	s_mov_b32 m0, s0
	v_readfirstlane_b32 s0, v153
	buffer_load_dwordx4 v179, s[20:23], 0 offen lds
	s_mov_b32 m0, s0
	s_nop 0
	buffer_load_dwordx4 v178, s[20:23], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[82:85], v[166:173], v[230:237], v[82:85]
	v_mfma_f32_16x16x128_f8f6f4 v[86:89], v[166:173], v[238:245], v[86:89]
	v_mfma_f32_16x16x128_f8f6f4 v[90:93], v[206:213], v[230:237], v[90:93]
	v_mfma_f32_16x16x128_f8f6f4 v[94:97], v[206:213], v[238:245], v[94:97]
	v_mfma_f32_16x16x128_f8f6f4 v[98:101], v[214:221], v[230:237], v[98:101]
	v_mfma_f32_16x16x128_f8f6f4 v[102:105], v[214:221], v[238:245], v[102:105]
	v_mfma_f32_16x16x128_f8f6f4 v[106:109], v[222:229], v[230:237], v[106:109]
	v_mfma_f32_16x16x128_f8f6f4 v[110:113], v[222:229], v[238:245], v[110:113]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v154 offset:0
ds_read_b64_tr_b8 v[4:5], v154 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v155 offset:0
ds_read_b64_tr_b8 v[12:13], v155 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v156 offset:0
ds_read_b64_tr_b8 v[8:9], v156 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v157 offset:0
ds_read_b64_tr_b8 v[16:17], v157 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[150:153], v198 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[166:169], v198 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[206:209], v198 offset:0x1000

	;;#ASMEND
	s_or_b32 s0, s14, 0x80f00
	;;#ASMSTART
	ds_read_b128 v[214:217], v198 offset:0x1800

	;;#ASMEND
	s_ashr_i32 s1, s0, 31
	;;#ASMSTART
	ds_read_b128 v[154:157], v199 offset:0

	;;#ASMEND
	s_add_u32 s0, s10, s0
	;;#ASMSTART
	ds_read_b128 v[170:173], v199 offset:0x800

	;;#ASMEND
	s_addc_u32 s1, s11, s1
	s_mov_b64 s[22:23], s[2:3]
	;;#ASMSTART
	ds_read_b128 v[210:213], v199 offset:0x1000

	;;#ASMEND
	s_mov_b64 s[20:21], s[0:1]
	v_readfirstlane_b32 s0, v146
	;;#ASMSTART
	ds_read_b128 v[218:221], v199 offset:0x1800

	;;#ASMEND
	s_mov_b32 m0, s0
	v_readfirstlane_b32 s0, v158
	buffer_load_dwordx4 v179, s[20:23], 0 offen lds
	s_mov_b32 m0, s0
	s_nop 0
	buffer_load_dwordx4 v178, s[20:23], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(8)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[114:117], v[150:157], v[2:9], v[114:117]
	v_mfma_f32_16x16x128_f8f6f4 v[118:121], v[150:157], v[10:17], v[118:121]
	v_mfma_f32_16x16x128_f8f6f4 v[122:125], v[166:173], v[2:9], v[122:125]
	v_mfma_f32_16x16x128_f8f6f4 v[126:129], v[166:173], v[10:17], v[126:129]
	v_mfma_f32_16x16x128_f8f6f4 v[130:133], v[206:213], v[2:9], v[130:133]
	v_mfma_f32_16x16x128_f8f6f4 v[134:137], v[206:213], v[10:17], v[134:137]
	v_mfma_f32_16x16x128_f8f6f4 v[138:141], v[214:221], v[2:9], v[138:141]
	v_mfma_f32_16x16x128_f8f6f4 v[142:145], v[214:221], v[10:17], v[142:145]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b64_tr_b8 v[222:223], v159 offset:0
ds_read_b64_tr_b8 v[224:225], v159 offset:1024

	;;#ASMEND
	s_add_u32 s0, s4, 0xf80000
	;;#ASMSTART
	ds_read_b64_tr_b8 v[230:231], v160 offset:0
ds_read_b64_tr_b8 v[232:233], v160 offset:1024

	;;#ASMEND
	s_addc_u32 s1, s5, 0
	s_mov_b64 s[22:23], s[2:3]
	;;#ASMSTART
	ds_read_b64_tr_b8 v[226:227], v161 offset:0
ds_read_b64_tr_b8 v[228:229], v161 offset:1024

	;;#ASMEND
	s_mov_b64 s[20:21], s[0:1]
	v_readfirstlane_b32 s0, v163
	;;#ASMSTART
	ds_read_b64_tr_b8 v[234:235], v162 offset:0
ds_read_b64_tr_b8 v[236:237], v162 offset:1024

	;;#ASMEND
	s_mov_b32 m0, s0
	v_readfirstlane_b32 s0, v164
	buffer_load_dwordx4 v182, s[20:23], 0 offen lds
	s_mov_b32 m0, s0
	s_nop 0
	buffer_load_dwordx4 v181, s[20:23], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[146:149], v[150:157], v[222:229], v[18:21]
	v_mfma_f32_16x16x128_f8f6f4 v[150:153], v[150:157], v[230:237], v[22:25]
	v_mfma_f32_16x16x128_f8f6f4 v[154:157], v[166:173], v[222:229], v[26:29]
	v_mfma_f32_16x16x128_f8f6f4 v[158:161], v[166:173], v[230:237], v[30:33]
	v_mfma_f32_16x16x128_f8f6f4 v[166:169], v[206:213], v[230:237], v[38:41]
	v_mfma_f32_16x16x128_f8f6f4 v[170:173], v[214:221], v[222:229], v[42:45]
	v_mfma_f32_16x16x128_f8f6f4 v[174:177], v[214:221], v[230:237], v[46:49]
	v_mfma_f32_16x16x128_f8f6f4 v[162:165], v[206:213], v[222:229], v[34:37]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[22:25], v200 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[30:33], v200 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[38:41], v200 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[206:209], v200 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[26:29], v201 offset:0

	;;#ASMEND
	s_add_u32 s0, s4, 0xf80080
	;;#ASMSTART
	ds_read_b128 v[34:37], v201 offset:0x800

	;;#ASMEND
	s_addc_u32 s1, s5, 0
	s_mov_b64 s[22:23], s[2:3]
	;;#ASMSTART
	ds_read_b128 v[42:45], v201 offset:0x1000

	;;#ASMEND
	s_mov_b64 s[20:21], s[0:1]
	v_readfirstlane_b32 s0, v202
	;;#ASMSTART
	ds_read_b128 v[210:213], v201 offset:0x1800

	;;#ASMEND
	s_mov_b32 m0, s0
	v_readfirstlane_b32 s0, v203
	buffer_load_dwordx4 v182, s[20:23], 0 offen lds
	s_mov_b32 m0, s0
	s_nop 0
	buffer_load_dwordx4 v181, s[20:23], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[50:53], v[22:29], v[2:9], v[50:53]
	v_mfma_f32_16x16x128_f8f6f4 v[54:57], v[22:29], v[10:17], v[54:57]
	v_mfma_f32_16x16x128_f8f6f4 v[58:61], v[30:37], v[2:9], v[58:61]
	v_mfma_f32_16x16x128_f8f6f4 v[62:65], v[30:37], v[10:17], v[62:65]
	v_mfma_f32_16x16x128_f8f6f4 v[66:69], v[38:45], v[2:9], v[66:69]
	v_mfma_f32_16x16x128_f8f6f4 v[70:73], v[38:45], v[10:17], v[70:73]
	v_mfma_f32_16x16x128_f8f6f4 v[74:77], v[206:213], v[2:9], v[74:77]
	v_mfma_f32_16x16x128_f8f6f4 v[78:81], v[206:213], v[10:17], v[78:81]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_u32 s0, s15, 0xf80
	s_addc_u32 s1, s16, 0
	s_mov_b64 s[18:19], s[2:3]
	s_mov_b64 s[16:17], s[0:1]
	v_readfirstlane_b32 s0, v204
	s_mov_b32 m0, s0
	v_readfirstlane_b32 s0, v205
	buffer_load_dwordx4 v179, s[16:19], 0 offen lds
	s_mov_b32 m0, s0
	s_nop 0
	buffer_load_dwordx4 v178, s[16:19], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[18:21], v[22:29], v[222:229], v[82:85]
	v_mfma_f32_16x16x128_f8f6f4 v[22:25], v[22:29], v[230:237], v[86:89]
	v_mfma_f32_16x16x128_f8f6f4 v[26:29], v[30:37], v[222:229], v[90:93]
	v_mfma_f32_16x16x128_f8f6f4 v[30:33], v[30:37], v[230:237], v[94:97]
	v_mfma_f32_16x16x128_f8f6f4 v[34:37], v[38:45], v[222:229], v[98:101]
	v_mfma_f32_16x16x128_f8f6f4 v[38:41], v[38:45], v[230:237], v[102:105]
	v_mfma_f32_16x16x128_f8f6f4 v[42:45], v[206:213], v[222:229], v[106:109]
	v_mfma_f32_16x16x128_f8f6f4 v[46:49], v[206:213], v[230:237], v[110:113]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v192 offset:0
ds_read_b64_tr_b8 v[4:5], v192 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v184 offset:0
ds_read_b64_tr_b8 v[12:13], v184 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v185 offset:0
ds_read_b64_tr_b8 v[8:9], v185 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v183 offset:0
ds_read_b64_tr_b8 v[16:17], v183 offset:1024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[202:205], v193 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[210:213], v193 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[218:221], v193 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[226:229], v193 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[206:209], v194 offset:0

	;;#ASMEND
	s_or_b32 s0, s14, 0x80f80
	;;#ASMSTART
	ds_read_b128 v[214:217], v194 offset:0x800

	;;#ASMEND
	s_ashr_i32 s1, s0, 31
	;;#ASMSTART
	ds_read_b128 v[222:225], v194 offset:0x1000

	;;#ASMEND
	s_add_u32 s0, s10, s0
	v_readfirstlane_b32 s4, v195
	v_add_u32_e32 v82, 0xe000, v180
	;;#ASMSTART
	ds_read_b128 v[230:233], v194 offset:0x1800

	;;#ASMEND
	s_addc_u32 s1, s11, s1
	s_mov_b32 m0, s4
	v_readfirstlane_b32 s4, v82
	buffer_load_dwordx4 v179, s[0:3], 0 offen lds
	s_mov_b32 m0, s4
	s_nop 0
	buffer_load_dwordx4 v178, s[0:3], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[82:85], v[202:209], v[2:9], v[114:117]
	v_mfma_f32_16x16x128_f8f6f4 v[86:89], v[202:209], v[10:17], v[118:121]
	v_mfma_f32_16x16x128_f8f6f4 v[90:93], v[210:217], v[2:9], v[122:125]
	v_mfma_f32_16x16x128_f8f6f4 v[94:97], v[210:217], v[10:17], v[126:129]
	v_mfma_f32_16x16x128_f8f6f4 v[98:101], v[218:225], v[2:9], v[130:133]
	v_mfma_f32_16x16x128_f8f6f4 v[102:105], v[218:225], v[10:17], v[134:137]
	v_mfma_f32_16x16x128_f8f6f4 v[106:109], v[226:233], v[2:9], v[138:141]
	v_mfma_f32_16x16x128_f8f6f4 v[110:113], v[226:233], v[10:17], v[142:145]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	v_add_u32_e32 v114, 0x4400, v190
	v_bitop3_b32 v115, v114, v188, v187 bitop3:0xf6
	;;#ASMSTART
	ds_read_b64_tr_b8 v[178:179], v115 offset:0
ds_read_b64_tr_b8 v[180:181], v115 offset:1024

	;;#ASMEND
	v_bitop3_b32 v114, v114, v191, v187 bitop3:0xf6
	;;#ASMSTART
	ds_read_b64_tr_b8 v[234:235], v114 offset:0
ds_read_b64_tr_b8 v[236:237], v114 offset:1024

	;;#ASMEND
	v_add_u32_e32 v114, 0x4400, v189
	v_bitop3_b32 v115, v114, v188, v187 bitop3:0xf6
	;;#ASMSTART
	ds_read_b64_tr_b8 v[182:183], v115 offset:0
ds_read_b64_tr_b8 v[184:185], v115 offset:1024

	;;#ASMEND
	v_bitop3_b32 v114, v114, v191, v187 bitop3:0xf6
	;;#ASMSTART
	ds_read_b64_tr_b8 v[238:239], v114 offset:0
ds_read_b64_tr_b8 v[240:241], v114 offset:1024

	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[114:117], v[202:209], v[178:185], v[146:149]
	v_mfma_f32_16x16x128_f8f6f4 v[118:121], v[202:209], v[234:241], v[150:153]
	v_mfma_f32_16x16x128_f8f6f4 v[122:125], v[210:217], v[178:185], v[154:157]
	v_mfma_f32_16x16x128_f8f6f4 v[126:129], v[210:217], v[234:241], v[158:161]
	v_mfma_f32_16x16x128_f8f6f4 v[130:133], v[218:225], v[178:185], v[162:165]
	v_mfma_f32_16x16x128_f8f6f4 v[134:137], v[218:225], v[234:241], v[166:169]
	v_mfma_f32_16x16x128_f8f6f4 v[138:141], v[226:233], v[178:185], v[170:173]
	v_mfma_f32_16x16x128_f8f6f4 v[142:145], v[226:233], v[234:241], v[174:177]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[166:169], v196 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[202:205], v196 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[210:213], v196 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[218:221], v196 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[170:173], v197 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[206:209], v197 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[214:217], v197 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[222:225], v197 offset:0x1800

	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[50:53], v[166:173], v[2:9], v[50:53]
	v_mfma_f32_16x16x128_f8f6f4 v[58:61], v[202:209], v[2:9], v[58:61]
	v_mfma_f32_16x16x128_f8f6f4 v[62:65], v[202:209], v[10:17], v[62:65]
	v_mfma_f32_16x16x128_f8f6f4 v[146:149], v[166:173], v[10:17], v[54:57]
	v_mfma_f32_16x16x128_f8f6f4 v[150:153], v[210:217], v[2:9], v[66:69]
	v_mfma_f32_16x16x128_f8f6f4 v[154:157], v[210:217], v[10:17], v[70:73]
	v_mfma_f32_16x16x128_f8f6f4 v[158:161], v[218:225], v[2:9], v[74:77]
	v_mfma_f32_16x16x128_f8f6f4 v[162:165], v[218:225], v[10:17], v[78:81]
	s_setprio 0
	v_add_u32_e32 v6, 0x8800, v190
	s_barrier
	v_bitop3_b32 v7, v6, v188, v187 bitop3:0xf6
	;;#ASMSTART
	ds_read_b64_tr_b8 v[2:3], v7 offset:0
ds_read_b64_tr_b8 v[4:5], v7 offset:1024

	;;#ASMEND
	v_bitop3_b32 v6, v6, v191, v187 bitop3:0xf6
	;;#ASMSTART
	ds_read_b64_tr_b8 v[10:11], v6 offset:0
ds_read_b64_tr_b8 v[12:13], v6 offset:1024

	;;#ASMEND
	v_add_u32_e32 v14, 0x8800, v189
	v_bitop3_b32 v15, v14, v188, v187 bitop3:0xf6
	;;#ASMSTART
	ds_read_b64_tr_b8 v[6:7], v15 offset:0
ds_read_b64_tr_b8 v[8:9], v15 offset:1024

	;;#ASMEND
	v_bitop3_b32 v54, v14, v191, v187 bitop3:0xf6
	;;#ASMSTART
	ds_read_b64_tr_b8 v[14:15], v54 offset:0
ds_read_b64_tr_b8 v[16:17], v54 offset:1024

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(2)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[226:229], v[166:173], v[178:185], v[18:21]
	v_mfma_f32_16x16x128_f8f6f4 v[230:233], v[166:173], v[234:241], v[22:25]
	v_mfma_f32_16x16x128_f8f6f4 v[242:245], v[202:209], v[178:185], v[26:29]
	v_mfma_f32_16x16x128_f8f6f4 v[246:249], v[202:209], v[234:241], v[30:33]
	v_mfma_f32_16x16x128_f8f6f4 v[250:253], v[210:217], v[178:185], v[34:37]
	v_mfma_f32_16x16x128_f8f6f4 v[210:213], v[210:217], v[234:241], v[38:41]
	v_mfma_f32_16x16x128_f8f6f4 v[214:217], v[218:225], v[178:185], v[42:45]
	v_mfma_f32_16x16x128_f8f6f4 v[218:221], v[218:225], v[234:241], v[46:49]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b128 v[18:21], v198 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[26:29], v198 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[42:45], v198 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[74:77], v198 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[22:25], v199 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[30:33], v199 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[46:49], v199 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[78:81], v199 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[182:185], v[18:25], v[2:9], v[82:85]
	v_mfma_f32_16x16x128_f8f6f4 v[178:181], v[18:25], v[10:17], v[86:89]
	v_mfma_f32_16x16x128_f8f6f4 v[174:177], v[26:33], v[2:9], v[90:93]
	v_mfma_f32_16x16x128_f8f6f4 v[170:173], v[26:33], v[10:17], v[94:97]
	v_mfma_f32_16x16x128_f8f6f4 v[166:169], v[42:49], v[2:9], v[98:101]
	v_mfma_f32_16x16x128_f8f6f4 v[94:97], v[42:49], v[10:17], v[102:105]
	v_mfma_f32_16x16x128_f8f6f4 v[66:69], v[74:81], v[2:9], v[106:109]
	v_mfma_f32_16x16x128_f8f6f4 v[38:41], v[74:81], v[10:17], v[110:113]
	s_setprio 0
	v_add_u32_e32 v34, 0xcc00, v190
	s_barrier
	v_bitop3_b32 v35, v34, v188, v187 bitop3:0xf6
	;;#ASMSTART
	ds_read_b64_tr_b8 v[192:193], v35 offset:0
ds_read_b64_tr_b8 v[194:195], v35 offset:1024

	;;#ASMEND
	v_bitop3_b32 v34, v34, v191, v187 bitop3:0xf6
	;;#ASMSTART
	ds_read_b64_tr_b8 v[202:203], v34 offset:0
ds_read_b64_tr_b8 v[204:205], v34 offset:1024

	;;#ASMEND
	v_add_u32_e32 v34, 0xcc00, v189
	v_bitop3_b32 v35, v34, v188, v187 bitop3:0xf6
	;;#ASMSTART
	ds_read_b64_tr_b8 v[196:197], v35 offset:0
ds_read_b64_tr_b8 v[198:199], v35 offset:1024

	;;#ASMEND
	v_bitop3_b32 v34, v34, v191, v187 bitop3:0xf6
	;;#ASMSTART
	ds_read_b64_tr_b8 v[206:207], v34 offset:0
ds_read_b64_tr_b8 v[208:209], v34 offset:1024

	;;#ASMEND
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[106:109], v[18:25], v[192:199], v[114:117]
	v_mfma_f32_16x16x128_f8f6f4 v[102:105], v[18:25], v[202:209], v[118:121]
	v_mfma_f32_16x16x128_f8f6f4 v[82:85], v[26:33], v[192:199], v[122:125]
	v_mfma_f32_16x16x128_f8f6f4 v[70:73], v[26:33], v[202:209], v[126:129]
	v_mfma_f32_16x16x128_f8f6f4 v[54:57], v[42:49], v[192:199], v[130:133]
	v_mfma_f32_16x16x128_f8f6f4 v[46:49], v[42:49], v[202:209], v[134:137]
	v_mfma_f32_16x16x128_f8f6f4 v[34:37], v[74:81], v[192:199], v[138:141]
	v_mfma_f32_16x16x128_f8f6f4 v[26:29], v[74:81], v[202:209], v[142:145]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[18:21], v200 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[110:113], v200 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[118:121], v200 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[126:129], v200 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[22:25], v201 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[114:117], v201 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[122:125], v201 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[130:133], v201 offset:0x1800

	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_f32_16x16x128_f8f6f4 v[98:101], v[18:25], v[2:9], v[50:53]
	v_mfma_f32_16x16x128_f8f6f4 v[86:89], v[18:25], v[10:17], v[146:149]
	v_mfma_f32_16x16x128_f8f6f4 v[90:93], v[110:117], v[2:9], v[58:61]
	v_mfma_f32_16x16x128_f8f6f4 v[74:77], v[110:117], v[10:17], v[62:65]
	v_mfma_f32_16x16x128_f8f6f4 v[78:81], v[118:125], v[2:9], v[150:153]
	v_mfma_f32_16x16x128_f8f6f4 v[58:61], v[118:125], v[10:17], v[154:157]
	v_mfma_f32_16x16x128_f8f6f4 v[62:65], v[126:133], v[2:9], v[158:161]
	v_mfma_f32_16x16x128_f8f6f4 v[50:53], v[126:133], v[10:17], v[162:165]
	v_mfma_f32_16x16x128_f8f6f4 v[42:45], v[18:25], v[192:199], v[226:229]
	v_mfma_f32_16x16x128_f8f6f4 v[30:33], v[18:25], v[202:209], v[230:233]
	v_mfma_f32_16x16x128_f8f6f4 v[22:25], v[110:117], v[192:199], v[242:245]
	v_mfma_f32_16x16x128_f8f6f4 v[18:21], v[110:117], v[202:209], v[246:249]
	v_mfma_f32_16x16x128_f8f6f4 v[14:17], v[118:125], v[192:199], v[250:253]
	v_mfma_f32_16x16x128_f8f6f4 v[10:13], v[118:125], v[202:209], v[210:213]
	v_mfma_f32_16x16x128_f8f6f4 v[6:9], v[126:133], v[192:199], v[214:217]
	v_mfma_f32_16x16x128_f8f6f4 v[2:5], v[126:133], v[202:209], v[218:221]
	s_setprio 0
	s_movk_i32 s0, 0x100
	v_cmp_gt_u32_e32 vcc, s0, v0
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB1_4
; %bb.3:
	s_barrier
.LBB1_4:
	s_or_b64 exec, exec, s[0:1]
	v_lshlrev_b32_e32 v1, 18, v1
	v_lshl_or_b32 v1, s12, 20, v1
	v_or_b32_e32 v110, s13, v186
	v_add_u32_e32 v130, v1, v110
	v_lshlrev_b32_e32 v1, 10, v0
	s_mov_b32 s0, 0xc00f
	v_bitop3_b32 v0, v1, s0, v0 bitop3:0xc8
	v_ashrrev_i32_e32 v131, 31, v130
	v_lshlrev_b32_e32 v0, 1, v0
	v_mov_b32_e32 v1, 0
	v_lshl_add_u64 v[132:133], v[130:131], 1, s[8:9]
	v_or_b32_e32 v110, 0x2000, v0
	v_mov_b32_e32 v111, v1
	v_lshl_add_u64 v[112:113], v[132:133], 0, v[110:111]
	global_store_short_d16_hi v[112:113], v183, off
	v_or_b32_e32 v112, 0x4000, v0
	v_mov_b32_e32 v113, v1
	v_lshl_add_u64 v[114:115], v[132:133], 0, v[112:113]
	global_store_short_d16_hi v[114:115], v184, off
	v_or_b32_e32 v114, 0x6000, v0
	v_mov_b32_e32 v115, v1
	v_lshl_add_u64 v[134:135], v[132:133], 0, v[0:1]
	v_lshl_add_u64 v[116:117], v[132:133], 0, v[114:115]
	global_store_short_d16_hi v[134:135], v182, off
	global_store_short_d16_hi v[116:117], v185, off
	global_store_short_d16_hi v[134:135], v178, off offset:32
	v_or_b32_e32 v116, 0x2020, v0
	v_mov_b32_e32 v117, v1
	v_lshl_add_u64 v[118:119], v[132:133], 0, v[116:117]
	global_store_short_d16_hi v[118:119], v179, off
	v_or_b32_e32 v118, 0x4020, v0
	v_mov_b32_e32 v119, v1
	v_lshl_add_u64 v[120:121], v[132:133], 0, v[118:119]
	global_store_short_d16_hi v[120:121], v180, off
	v_or_b32_e32 v120, 0x6020, v0
	v_mov_b32_e32 v121, v1
	v_lshl_add_u64 v[122:123], v[132:133], 0, v[120:121]
	global_store_short_d16_hi v[122:123], v181, off
	v_or_b32_e32 v122, 0x20000, v0
	v_mov_b32_e32 v123, v1
	v_lshl_add_u64 v[124:125], v[132:133], 0, v[122:123]
	global_store_short_d16_hi v[124:125], v174, off
	v_or_b32_e32 v124, 0x22000, v0
	v_mov_b32_e32 v125, v1
	v_lshl_add_u64 v[126:127], v[132:133], 0, v[124:125]
	global_store_short_d16_hi v[126:127], v175, off
	v_or_b32_e32 v126, 0x24000, v0
	v_mov_b32_e32 v127, v1
	v_lshl_add_u64 v[128:129], v[132:133], 0, v[126:127]
	global_store_short_d16_hi v[128:129], v176, off
	v_or_b32_e32 v128, 0x26000, v0
	v_mov_b32_e32 v129, v1
	v_lshl_add_u64 v[136:137], v[132:133], 0, v[128:129]
	global_store_short_d16_hi v[136:137], v177, off
	v_or_b32_e32 v136, 0x20020, v0
	v_mov_b32_e32 v137, v1
	v_lshl_add_u64 v[138:139], v[132:133], 0, v[136:137]
	global_store_short_d16_hi v[138:139], v170, off
	v_or_b32_e32 v138, 0x22020, v0
	v_mov_b32_e32 v139, v1
	v_lshl_add_u64 v[140:141], v[132:133], 0, v[138:139]
	global_store_short_d16_hi v[140:141], v171, off
	v_or_b32_e32 v140, 0x24020, v0
	v_mov_b32_e32 v141, v1
	v_lshl_add_u64 v[142:143], v[132:133], 0, v[140:141]
	global_store_short_d16_hi v[142:143], v172, off
	v_or_b32_e32 v142, 0x26020, v0
	v_mov_b32_e32 v143, v1
	v_lshl_add_u64 v[144:145], v[132:133], 0, v[142:143]
	global_store_short_d16_hi v[144:145], v173, off
	v_or_b32_e32 v144, 0x40000, v0
	v_mov_b32_e32 v145, v1
	v_lshl_add_u64 v[146:147], v[132:133], 0, v[144:145]
	global_store_short_d16_hi v[146:147], v166, off
	v_or_b32_e32 v146, 0x42000, v0
	v_mov_b32_e32 v147, v1
	v_lshl_add_u64 v[148:149], v[132:133], 0, v[146:147]
	global_store_short_d16_hi v[148:149], v167, off
	v_or_b32_e32 v148, 0x44000, v0
	v_mov_b32_e32 v149, v1
	v_lshl_add_u64 v[150:151], v[132:133], 0, v[148:149]
	global_store_short_d16_hi v[150:151], v168, off
	v_or_b32_e32 v150, 0x46000, v0
	v_mov_b32_e32 v151, v1
	v_lshl_add_u64 v[152:153], v[132:133], 0, v[150:151]
	global_store_short_d16_hi v[152:153], v169, off
	v_or_b32_e32 v152, 0x40020, v0
	v_mov_b32_e32 v153, v1
	v_lshl_add_u64 v[154:155], v[132:133], 0, v[152:153]
	global_store_short_d16_hi v[154:155], v94, off
	v_or_b32_e32 v154, 0x42020, v0
	v_mov_b32_e32 v155, v1
	v_lshl_add_u64 v[156:157], v[132:133], 0, v[154:155]
	global_store_short_d16_hi v[156:157], v95, off
	v_or_b32_e32 v94, 0x44020, v0
	v_mov_b32_e32 v95, v1
	v_lshl_add_u64 v[156:157], v[132:133], 0, v[94:95]
	global_store_short_d16_hi v[156:157], v96, off
	v_or_b32_e32 v156, 0x46020, v0
	v_mov_b32_e32 v157, v1
	v_lshl_add_u64 v[158:159], v[132:133], 0, v[156:157]
	global_store_short_d16_hi v[158:159], v97, off
	v_or_b32_e32 v96, 0x60000, v0
	v_mov_b32_e32 v97, v1
	v_lshl_add_u64 v[158:159], v[132:133], 0, v[96:97]
	global_store_short_d16_hi v[158:159], v66, off
	v_or_b32_e32 v158, 0x62000, v0
	v_mov_b32_e32 v159, v1
	v_lshl_add_u64 v[160:161], v[132:133], 0, v[158:159]
	global_store_short_d16_hi v[160:161], v67, off
	v_or_b32_e32 v66, 0x64000, v0
	v_mov_b32_e32 v67, v1
	v_lshl_add_u64 v[160:161], v[132:133], 0, v[66:67]
	global_store_short_d16_hi v[160:161], v68, off
	v_or_b32_e32 v160, 0x66000, v0
	v_mov_b32_e32 v161, v1
	v_lshl_add_u64 v[162:163], v[132:133], 0, v[160:161]
	global_store_short_d16_hi v[162:163], v69, off
	v_or_b32_e32 v68, 0x60020, v0
	v_mov_b32_e32 v69, v1
	v_lshl_add_u64 v[162:163], v[132:133], 0, v[68:69]
	global_store_short_d16_hi v[162:163], v38, off
	v_or_b32_e32 v162, 0x62020, v0
	v_mov_b32_e32 v163, v1
	v_lshl_add_u64 v[164:165], v[132:133], 0, v[162:163]
	global_store_short_d16_hi v[164:165], v39, off
	v_or_b32_e32 v38, 0x64020, v0
	v_mov_b32_e32 v39, v1
	v_lshl_add_u64 v[164:165], v[132:133], 0, v[38:39]
	global_store_short_d16_hi v[164:165], v40, off
	v_or_b32_e32 v164, 0x66020, v0
	v_mov_b32_e32 v165, v1
	v_lshl_add_u64 v[166:167], v[132:133], 0, v[164:165]
	global_store_short_d16_hi v[166:167], v41, off
	v_lshl_add_u64 v[40:41], v[132:133], 0, s[6:7]
	v_lshl_add_u64 v[132:133], v[40:41], 0, v[110:111]
	global_store_short_d16_hi v[134:135], v106, off offset:256
	global_store_short_d16_hi v[132:133], v107, off
	v_lshl_add_u64 v[106:107], v[40:41], 0, v[112:113]
	global_store_short_d16_hi v[106:107], v108, off
	v_lshl_add_u64 v[106:107], v[40:41], 0, v[114:115]
	global_store_short_d16_hi v[106:107], v109, off
	v_lshl_add_u64 v[106:107], v[40:41], 0, v[0:1]
	global_store_short_d16_hi v[106:107], v102, off offset:32
	v_lshl_add_u64 v[106:107], v[40:41], 0, v[116:117]
	global_store_short_d16_hi v[106:107], v103, off
	v_lshl_add_u64 v[102:103], v[40:41], 0, v[118:119]
	global_store_short_d16_hi v[102:103], v104, off
	v_lshl_add_u64 v[102:103], v[40:41], 0, v[120:121]
	global_store_short_d16_hi v[102:103], v105, off
	v_lshl_add_u64 v[102:103], v[40:41], 0, v[122:123]
	global_store_short_d16_hi v[102:103], v82, off
	v_lshl_add_u64 v[102:103], v[40:41], 0, v[124:125]
	global_store_short_d16_hi v[102:103], v83, off
	v_lshl_add_u64 v[82:83], v[40:41], 0, v[126:127]
	global_store_short_d16_hi v[82:83], v84, off
	v_lshl_add_u64 v[82:83], v[40:41], 0, v[128:129]
	global_store_short_d16_hi v[82:83], v85, off
	v_lshl_add_u64 v[82:83], v[40:41], 0, v[136:137]
	global_store_short_d16_hi v[82:83], v70, off
	v_lshl_add_u64 v[82:83], v[40:41], 0, v[138:139]
	global_store_short_d16_hi v[82:83], v71, off
	v_lshl_add_u64 v[70:71], v[40:41], 0, v[140:141]
	global_store_short_d16_hi v[70:71], v72, off
	v_lshl_add_u64 v[70:71], v[40:41], 0, v[142:143]
	global_store_short_d16_hi v[70:71], v73, off
	v_lshl_add_u64 v[70:71], v[40:41], 0, v[144:145]
	global_store_short_d16_hi v[70:71], v54, off
	v_lshl_add_u64 v[70:71], v[40:41], 0, v[146:147]
	global_store_short_d16_hi v[70:71], v55, off
	v_lshl_add_u64 v[54:55], v[40:41], 0, v[148:149]
	global_store_short_d16_hi v[54:55], v56, off
	v_lshl_add_u64 v[54:55], v[40:41], 0, v[150:151]
	global_store_short_d16_hi v[54:55], v57, off
	v_lshl_add_u64 v[54:55], v[40:41], 0, v[152:153]
	global_store_short_d16_hi v[54:55], v46, off
	v_lshl_add_u64 v[54:55], v[40:41], 0, v[154:155]
	global_store_short_d16_hi v[54:55], v47, off
	v_lshl_add_u64 v[46:47], v[40:41], 0, v[94:95]
	global_store_short_d16_hi v[46:47], v48, off
	v_lshl_add_u64 v[46:47], v[40:41], 0, v[156:157]
	global_store_short_d16_hi v[46:47], v49, off
	v_lshl_add_u64 v[46:47], v[40:41], 0, v[96:97]
	global_store_short_d16_hi v[46:47], v34, off
	v_lshl_add_u64 v[46:47], v[40:41], 0, v[158:159]
	global_store_short_d16_hi v[46:47], v35, off
	v_lshl_add_u64 v[34:35], v[40:41], 0, v[66:67]
	global_store_short_d16_hi v[34:35], v36, off
	v_lshl_add_u64 v[34:35], v[40:41], 0, v[160:161]
	global_store_short_d16_hi v[34:35], v37, off
	v_lshl_add_u64 v[34:35], v[40:41], 0, v[68:69]
	global_store_short_d16_hi v[34:35], v26, off
	v_lshl_add_u64 v[34:35], v[40:41], 0, v[162:163]
	global_store_short_d16_hi v[34:35], v27, off
	v_lshl_add_u64 v[26:27], v[40:41], 0, v[38:39]
	global_store_short_d16_hi v[26:27], v28, off
	v_lshl_add_u64 v[26:27], v[40:41], 0, v[164:165]
	global_store_short_d16_hi v[26:27], v29, off
	v_add_u32_e32 v26, 0x80000, v130
	v_ashrrev_i32_e32 v27, 31, v26
	v_lshl_add_u64 v[26:27], v[26:27], 1, s[8:9]
	v_lshl_add_u64 v[34:35], v[26:27], 0, v[110:111]
	global_store_short_d16_hi v[34:35], v99, off
	v_lshl_add_u64 v[34:35], v[26:27], 0, v[112:113]
	v_lshl_add_u64 v[28:29], v[26:27], 0, v[0:1]
	global_store_short_d16_hi v[34:35], v100, off
	v_lshl_add_u64 v[34:35], v[26:27], 0, v[114:115]
	global_store_short_d16_hi v[28:29], v98, off
	global_store_short_d16_hi v[34:35], v101, off
	global_store_short_d16_hi v[28:29], v86, off offset:32
	v_lshl_add_u64 v[34:35], v[26:27], 0, v[116:117]
	global_store_short_d16_hi v[34:35], v87, off
	v_lshl_add_u64 v[34:35], v[26:27], 0, v[118:119]
	global_store_short_d16_hi v[34:35], v88, off
	v_lshl_add_u64 v[34:35], v[26:27], 0, v[120:121]
	global_store_short_d16_hi v[34:35], v89, off
	v_lshl_add_u64 v[34:35], v[26:27], 0, v[122:123]
	global_store_short_d16_hi v[34:35], v90, off
	v_lshl_add_u64 v[34:35], v[26:27], 0, v[124:125]
	global_store_short_d16_hi v[34:35], v91, off
	v_lshl_add_u64 v[34:35], v[26:27], 0, v[126:127]
	global_store_short_d16_hi v[34:35], v92, off
	v_lshl_add_u64 v[34:35], v[26:27], 0, v[128:129]
	global_store_short_d16_hi v[34:35], v93, off
	v_lshl_add_u64 v[34:35], v[26:27], 0, v[136:137]
	global_store_short_d16_hi v[34:35], v74, off
	v_lshl_add_u64 v[34:35], v[26:27], 0, v[138:139]
	global_store_short_d16_hi v[34:35], v75, off
	v_lshl_add_u64 v[34:35], v[26:27], 0, v[140:141]
	global_store_short_d16_hi v[34:35], v76, off
	v_lshl_add_u64 v[34:35], v[26:27], 0, v[142:143]
	global_store_short_d16_hi v[34:35], v77, off
	v_lshl_add_u64 v[34:35], v[26:27], 0, v[144:145]
	global_store_short_d16_hi v[34:35], v78, off
	v_lshl_add_u64 v[34:35], v[26:27], 0, v[146:147]
	global_store_short_d16_hi v[34:35], v79, off
	v_lshl_add_u64 v[34:35], v[26:27], 0, v[148:149]
	global_store_short_d16_hi v[34:35], v80, off
	v_lshl_add_u64 v[34:35], v[26:27], 0, v[150:151]
	global_store_short_d16_hi v[34:35], v81, off
	v_lshl_add_u64 v[34:35], v[26:27], 0, v[152:153]
	global_store_short_d16_hi v[34:35], v58, off
	v_lshl_add_u64 v[34:35], v[26:27], 0, v[154:155]
	global_store_short_d16_hi v[34:35], v59, off
	v_lshl_add_u64 v[34:35], v[26:27], 0, v[94:95]
	global_store_short_d16_hi v[34:35], v60, off
	v_lshl_add_u64 v[34:35], v[26:27], 0, v[156:157]
	global_store_short_d16_hi v[34:35], v61, off
	v_lshl_add_u64 v[34:35], v[26:27], 0, v[96:97]
	global_store_short_d16_hi v[34:35], v62, off
	v_lshl_add_u64 v[34:35], v[26:27], 0, v[158:159]
	global_store_short_d16_hi v[34:35], v63, off
	v_lshl_add_u64 v[34:35], v[26:27], 0, v[66:67]
	global_store_short_d16_hi v[34:35], v64, off
	v_lshl_add_u64 v[34:35], v[26:27], 0, v[160:161]
	global_store_short_d16_hi v[34:35], v65, off
	v_lshl_add_u64 v[34:35], v[26:27], 0, v[68:69]
	global_store_short_d16_hi v[34:35], v50, off
	v_lshl_add_u64 v[34:35], v[26:27], 0, v[162:163]
	global_store_short_d16_hi v[34:35], v51, off
	v_lshl_add_u64 v[34:35], v[26:27], 0, v[38:39]
	global_store_short_d16_hi v[34:35], v52, off
	v_lshl_add_u64 v[34:35], v[26:27], 0, v[164:165]
	v_lshl_add_u64 v[26:27], v[26:27], 0, s[6:7]
	v_lshl_add_u64 v[0:1], v[26:27], 0, v[0:1]
	global_store_short_d16_hi v[0:1], v30, off offset:32
	v_lshl_add_u64 v[0:1], v[26:27], 0, v[116:117]
	global_store_short_d16_hi v[0:1], v31, off
	v_lshl_add_u64 v[0:1], v[26:27], 0, v[118:119]
	global_store_short_d16_hi v[0:1], v32, off
	v_lshl_add_u64 v[0:1], v[26:27], 0, v[120:121]
	global_store_short_d16_hi v[0:1], v33, off
	v_lshl_add_u64 v[0:1], v[26:27], 0, v[122:123]
	global_store_short_d16_hi v[0:1], v22, off
	v_lshl_add_u64 v[0:1], v[26:27], 0, v[124:125]
	global_store_short_d16_hi v[0:1], v23, off
	v_lshl_add_u64 v[0:1], v[26:27], 0, v[126:127]
	global_store_short_d16_hi v[0:1], v24, off
	v_lshl_add_u64 v[0:1], v[26:27], 0, v[128:129]
	global_store_short_d16_hi v[0:1], v25, off
	v_lshl_add_u64 v[0:1], v[26:27], 0, v[136:137]
	global_store_short_d16_hi v[0:1], v18, off
	v_lshl_add_u64 v[0:1], v[26:27], 0, v[138:139]
	global_store_short_d16_hi v[0:1], v19, off
	v_lshl_add_u64 v[0:1], v[26:27], 0, v[140:141]
	global_store_short_d16_hi v[0:1], v20, off
	v_lshl_add_u64 v[0:1], v[26:27], 0, v[142:143]
	global_store_short_d16_hi v[0:1], v21, off
	v_lshl_add_u64 v[0:1], v[26:27], 0, v[144:145]
	global_store_short_d16_hi v[0:1], v14, off
	v_lshl_add_u64 v[0:1], v[26:27], 0, v[146:147]
	global_store_short_d16_hi v[0:1], v15, off
	v_lshl_add_u64 v[0:1], v[26:27], 0, v[148:149]
	global_store_short_d16_hi v[0:1], v16, off
	v_lshl_add_u64 v[0:1], v[26:27], 0, v[150:151]
	global_store_short_d16_hi v[0:1], v17, off
	v_lshl_add_u64 v[0:1], v[26:27], 0, v[152:153]
	global_store_short_d16_hi v[0:1], v10, off
	v_lshl_add_u64 v[0:1], v[26:27], 0, v[154:155]
	global_store_short_d16_hi v[0:1], v11, off
	v_lshl_add_u64 v[0:1], v[26:27], 0, v[94:95]
	global_store_short_d16_hi v[0:1], v12, off
	v_lshl_add_u64 v[0:1], v[26:27], 0, v[156:157]
	global_store_short_d16_hi v[0:1], v13, off
	v_lshl_add_u64 v[0:1], v[26:27], 0, v[96:97]
	global_store_short_d16_hi v[0:1], v6, off
	v_lshl_add_u64 v[0:1], v[26:27], 0, v[158:159]
	global_store_short_d16_hi v[0:1], v7, off
	v_lshl_add_u64 v[0:1], v[26:27], 0, v[66:67]
	global_store_short_d16_hi v[0:1], v8, off
	v_lshl_add_u64 v[0:1], v[26:27], 0, v[160:161]
	global_store_short_d16_hi v[0:1], v9, off
	v_lshl_add_u64 v[0:1], v[26:27], 0, v[68:69]
	global_store_short_d16_hi v[28:29], v42, off offset:256
	v_lshl_add_u64 v[28:29], v[26:27], 0, v[110:111]
	global_store_short_d16_hi v[0:1], v2, off
	v_lshl_add_u64 v[0:1], v[26:27], 0, v[162:163]
	global_store_short_d16_hi v[28:29], v43, off
	v_lshl_add_u64 v[28:29], v[26:27], 0, v[112:113]
	global_store_short_d16_hi v[0:1], v3, off
	v_lshl_add_u64 v[0:1], v[26:27], 0, v[38:39]
	global_store_short_d16_hi v[28:29], v44, off
	v_lshl_add_u64 v[28:29], v[26:27], 0, v[114:115]
	global_store_short_d16_hi v[0:1], v4, off
	v_lshl_add_u64 v[0:1], v[26:27], 0, v[164:165]
	global_store_short_d16_hi v[34:35], v53, off
	global_store_short_d16_hi v[28:29], v45, off
	global_store_short_d16_hi v[0:1], v5, off
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z22rrr_exact_8wave_kernel23rrr_exact_8wave_globals
		.amdhsa_group_segment_fixed_size 135168
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
	.text
.Lfunc_end1:
	.size	_Z22rrr_exact_8wave_kernel23rrr_exact_8wave_globals, .Lfunc_end1-_Z22rrr_exact_8wave_kernel23rrr_exact_8wave_globals
                                        ; -- End function
	.set _Z22rrr_exact_8wave_kernel23rrr_exact_8wave_globals.num_vgpr, 254
	.set _Z22rrr_exact_8wave_kernel23rrr_exact_8wave_globals.num_agpr, 0
	.set _Z22rrr_exact_8wave_kernel23rrr_exact_8wave_globals.numbered_sgpr, 34
	.set _Z22rrr_exact_8wave_kernel23rrr_exact_8wave_globals.private_seg_size, 0
	.set _Z22rrr_exact_8wave_kernel23rrr_exact_8wave_globals.uses_vcc, 1
	.set _Z22rrr_exact_8wave_kernel23rrr_exact_8wave_globals.uses_flat_scratch, 0
	.set _Z22rrr_exact_8wave_kernel23rrr_exact_8wave_globals.has_dyn_sized_stack, 0
	.set _Z22rrr_exact_8wave_kernel23rrr_exact_8wave_globals.has_recursion, 0
	.set _Z22rrr_exact_8wave_kernel23rrr_exact_8wave_globals.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 39324
; TotalNumSgprs: 40
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
	.text
	.protected	_Z22crr_exact_8wave_kernel23crr_exact_8wave_globals ; -- Begin function _Z22crr_exact_8wave_kernel23crr_exact_8wave_globals
	.globl	_Z22crr_exact_8wave_kernel23crr_exact_8wave_globals
	.p2align	8
	.type	_Z22crr_exact_8wave_kernel23crr_exact_8wave_globals,@function
