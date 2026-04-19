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
