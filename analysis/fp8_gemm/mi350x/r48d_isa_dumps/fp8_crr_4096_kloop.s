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
