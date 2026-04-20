.LBB2_7:                                ; =>This Inner Loop Header: Depth=1
	buffer_load_dwordx4 v[146:149], v166, s[0:3], s38 offen
	buffer_load_dwordx2 v[158:159], v167, s[20:23], s27 offen
	;;#ASMSTART
	ds_read_b128 v[2:5], v177 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[10:13], v177 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[6:9], v174 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[14:17], v174 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[184:187], v171 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[192:195], v171 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[218:221], v171 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[226:229], v171 offset:0x1800

	;;#ASMEND
	s_add_i32 s39, s30, s6
	;;#ASMSTART
	ds_read_b128 v[188:191], v168 offset:0

	;;#ASMEND
	s_add_i32 s8, s39, 0x80
	;;#ASMSTART
	ds_read_b128 v[196:199], v168 offset:0x800

	;;#ASMEND
	s_ashr_i32 s9, s8, 31
	;;#ASMSTART
	ds_read_b128 v[222:225], v168 offset:0x1000

	;;#ASMEND
	s_add_u32 s8, s28, s8
	v_readfirstlane_b32 s16, v165
	;;#ASMSTART
	ds_read_b128 v[230:233], v168 offset:0x1800

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
	v_mfma_scale_f32_16x16x128_f8f6f4 v[134:137], v[192:199], v[2:9], v[134:137], v146, v158 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[130:133], v[192:199], v[10:17], v[130:133], v146, v158 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[126:129], v[218:225], v[2:9], v[126:129], v148, v158 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[122:125], v[218:225], v[10:17], v[122:125], v148, v158 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[226:233], v[2:9], v[86:89], v148, v158 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[226:233], v[10:17], v[82:85], v148, v158 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b128 v[234:237], v183 offset:0

	;;#ASMEND
	s_add_u32 s40, s36, s6
	;;#ASMSTART
	ds_read_b128 v[242:245], v183 offset:0x800

	;;#ASMEND
	s_addc_u32 s41, s37, s7
	;;#ASMSTART
	ds_read_b128 v[238:241], v182 offset:0

	;;#ASMEND
	s_add_u32 s16, s40, 0x100
	v_readfirstlane_b32 s8, v153
	;;#ASMSTART
	ds_read_b128 v[246:249], v182 offset:0x800

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
	v_mfma_scale_f32_16x16x128_f8f6f4 v[98:101], v[184:191], v[234:241], v[98:101], v146, v159 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[184:191], v[242:249], v[90:93], v146, v159 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[192:199], v[234:241], v[94:97], v146, v159 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[102:105], v[192:199], v[242:249], v[102:105], v146, v159 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[106:109], v[218:225], v[234:241], v[106:109], v148, v159 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[110:113], v[218:225], v[242:249], v[110:113], v148, v159 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[114:117], v[226:233], v[234:241], v[114:117], v148, v159 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[118:121], v[226:233], v[242:249], v[118:121], v148, v159 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[184:187], v181 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[192:195], v181 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[218:221], v181 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[226:229], v181 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[188:191], v180 offset:0

	;;#ASMEND
	s_add_u32 s42, s24, s6
	;;#ASMSTART
	ds_read_b128 v[196:199], v180 offset:0x800

	;;#ASMEND
	s_addc_u32 s43, s25, s7
	;;#ASMSTART
	ds_read_b128 v[222:225], v180 offset:0x1000

	;;#ASMEND
	s_add_u32 s8, s42, 0x100
	v_readfirstlane_b32 s16, v151
	;;#ASMSTART
	ds_read_b128 v[230:233], v180 offset:0x1800

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
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[192:199], v[2:9], v[62:65], v147, v158 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[192:199], v[10:17], v[74:77], v147, v158 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[218:225], v[2:9], v[54:57], v149, v158 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[218:225], v[10:17], v[66:69], v149, v158 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[226:233], v[2:9], v[50:53], v149, v158 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[226:233], v[10:17], v[58:61], v149, v158 op_sel:[1,1,0] op_sel_hi:[0,0,0]
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
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[184:191], v[234:241], v[18:21], v147, v159 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[184:191], v[242:249], v[22:25], v147, v159 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[192:199], v[234:241], v[26:29], v147, v159 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[192:199], v[242:249], v[30:33], v147, v159 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[218:225], v[234:241], v[34:37], v149, v159 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[218:225], v[242:249], v[38:41], v149, v159 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[226:233], v[234:241], v[42:45], v149, v159 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], v[226:233], v[242:249], v[46:49], v149, v159 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[2:5], v179 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[10:13], v179 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[6:9], v178 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[14:17], v178 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[184:187], v203 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[192:195], v203 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[218:221], v203 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[226:229], v203 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[188:191], v202 offset:0

	;;#ASMEND
	s_addk_i32 s39, 0x100
	;;#ASMSTART
	ds_read_b128 v[196:199], v202 offset:0x800

	;;#ASMEND
	s_ashr_i32 s9, s39, 31
	;;#ASMSTART
	ds_read_b128 v[222:225], v202 offset:0x1000

	;;#ASMEND
	s_add_u32 s8, s28, s39
	v_readfirstlane_b32 s16, v162
	;;#ASMSTART
	ds_read_b128 v[230:233], v202 offset:0x1800

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
	v_mfma_scale_f32_16x16x128_f8f6f4 v[134:137], v[192:199], v[2:9], v[134:137], v146, v158 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[130:133], v[192:199], v[10:17], v[130:133], v146, v158 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[126:129], v[218:225], v[2:9], v[126:129], v148, v158 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[122:125], v[218:225], v[10:17], v[122:125], v148, v158 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[226:233], v[2:9], v[86:89], v148, v158 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[226:233], v[10:17], v[82:85], v148, v158 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b128 v[234:237], v217 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[242:245], v217 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[238:241], v216 offset:0

	;;#ASMEND
	s_add_u32 s16, s40, 0x180
	v_readfirstlane_b32 s8, v169
	;;#ASMSTART
	ds_read_b128 v[246:249], v216 offset:0x800

	;;#ASMEND
	s_addc_u32 s17, s41, 0
	s_mov_b32 m0, s8
	v_readfirstlane_b32 s8, v170
	buffer_load_dwordx4 v154, s[16:19], 0 offen lds
	s_mov_b32 m0, s8
	s_nop 0
	buffer_load_dwordx4 v156, s[16:19], 0 offen lds
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[98:101], v[184:191], v[234:241], v[98:101], v146, v159 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[184:191], v[242:249], v[90:93], v146, v159 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[192:199], v[234:241], v[94:97], v146, v159 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[102:105], v[192:199], v[242:249], v[102:105], v146, v159 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[106:109], v[218:225], v[234:241], v[106:109], v148, v159 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[110:113], v[218:225], v[242:249], v[110:113], v148, v159 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[114:117], v[226:233], v[234:241], v[114:117], v148, v159 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[118:121], v[226:233], v[242:249], v[118:121], v148, v159 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[184:187], v215 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[192:195], v215 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[218:221], v215 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[226:229], v215 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[188:191], v214 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[196:199], v214 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[222:225], v214 offset:0x1000

	;;#ASMEND
	s_add_u32 s8, s42, 0x180
	v_readfirstlane_b32 s16, v172
	;;#ASMSTART
	ds_read_b128 v[230:233], v214 offset:0x1800

	;;#ASMEND
	s_addc_u32 s9, s43, 0
	s_mov_b32 m0, s16
	v_readfirstlane_b32 s16, v173
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
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[192:199], v[2:9], v[62:65], v147, v158 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[192:199], v[10:17], v[74:77], v147, v158 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[218:225], v[2:9], v[54:57], v149, v158 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[218:225], v[10:17], v[66:69], v149, v158 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[226:233], v[2:9], v[50:53], v149, v158 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[226:233], v[10:17], v[58:61], v149, v158 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_u32 s16, s44, 0x180
	v_readfirstlane_b32 s8, v175
	s_addc_u32 s17, s45, 0
	s_mov_b32 m0, s8
	v_readfirstlane_b32 s8, v176
	buffer_load_dwordx4 v154, s[16:19], 0 offen lds
	s_mov_b32 m0, s8
	s_nop 0
	buffer_load_dwordx4 v156, s[16:19], 0 offen lds
	;;#ASMSTART
	s_waitcnt vmcnt(6)
	;;#ASMEND
	s_barrier
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[184:191], v[234:241], v[18:21], v147, v159 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[184:191], v[242:249], v[22:25], v147, v159 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[192:199], v[234:241], v[26:29], v147, v159 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[192:199], v[242:249], v[30:33], v147, v159 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[218:225], v[234:241], v[34:37], v149, v159 op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[218:225], v[242:249], v[38:41], v149, v159 op_sel:[0,1,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[226:233], v[234:241], v[42:45], v149, v159 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], v[226:233], v[242:249], v[46:49], v149, v159 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_setprio 0
	s_add_u32 s6, s6, 0x100
	s_addc_u32 s7, s7, 0
	s_addk_i32 s38, 0x400
	s_addk_i32 s27, 0x200
	s_cmpk_eq_i32 s6, 0x6f00
	s_barrier
	s_cbranch_scc0 .LBB2_7
; %bb.8:
	s_mov_b32 s6, 0x1bc00
	buffer_load_dwordx4 v[146:149], v166, s[0:3], s6 offen
	s_mov_b32 s0, 0xde00
	buffer_load_dwordx2 v[212:213], v167, s[20:23], s0 offen
	;;#ASMSTART
	ds_read_b128 v[2:5], v177 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[10:13], v177 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[6:9], v174 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[14:17], v174 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[184:187], v171 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[192:195], v171 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[218:221], v171 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[226:229], v171 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[188:191], v168 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[196:199], v168 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[222:225], v168 offset:0x1000

	;;#ASMEND
	s_add_u32 s8, s12, 0x6f80
	v_readfirstlane_b32 s0, v165
	;;#ASMSTART
	ds_read_b128 v[230:233], v168 offset:0x1800

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
	v_mfma_scale_f32_16x16x128_f8f6f4 v[162:165], v[184:191], v[2:9], v[142:145], v146, v212 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[166:169], v[184:191], v[10:17], v[138:141], v146, v212 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[170:173], v[192:199], v[2:9], v[134:137], v146, v212 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[142:145], v[218:225], v[2:9], v[126:129], v148, v212 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[150:153], v[218:225], v[10:17], v[122:125], v148, v212 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[154:157], v[226:233], v[2:9], v[86:89], v148, v212 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[174:177], v[192:199], v[10:17], v[130:133], v146, v212 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[158:161], v[226:233], v[10:17], v[82:85], v148, v212 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b128 v[234:237], v183 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[242:245], v183 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[238:241], v182 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[246:249], v182 offset:0x800

	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[184:191], v[234:241], v[98:101], v146, v213 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[184:191], v[242:249], v[90:93], v146, v213 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[122:125], v[192:199], v[234:241], v[94:97], v146, v213 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[126:129], v[192:199], v[242:249], v[102:105], v146, v213 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[130:133], v[218:225], v[234:241], v[106:109], v148, v213 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[134:137], v[218:225], v[242:249], v[110:113], v148, v213 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[138:141], v[226:233], v[234:241], v[114:117], v148, v213 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[118:121], v[226:233], v[242:249], v[118:121], v148, v213 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[90:93], v181 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[182:185], v181 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[190:193], v181 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[218:221], v181 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[94:97], v180 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[186:189], v180 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[194:197], v180 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[222:225], v180 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[182:189], v[2:9], v[62:65], v147, v212 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[190:197], v[2:9], v[54:57], v149, v212 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[218:225], v[2:9], v[50:53], v149, v212 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[106:109], v[90:97], v[2:9], v[70:73], v147, v212 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[110:113], v[90:97], v[10:17], v[78:81], v147, v212 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[114:117], v[182:189], v[10:17], v[74:77], v147, v212 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[98:101], v[190:197], v[10:17], v[66:69], v149, v212 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[102:105], v[218:225], v[10:17], v[58:61], v149, v212 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[2:5], v179 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[10:13], v179 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[6:9], v178 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[14:17], v178 offset:0x800

	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[90:97], v[234:241], v[18:21], v147, v213 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[90:97], v[242:249], v[22:25], v147, v213 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[178:181], v[182:189], v[234:241], v[26:29], v147, v213 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[182:185], v[182:189], v[242:249], v[30:33], v147, v213 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[186:189], v[190:197], v[234:241], v[34:37], v149, v213 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[190:193], v[190:197], v[242:249], v[38:41], v149, v213 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[194:197], v[218:225], v[234:241], v[42:45], v149, v213 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[198:201], v[218:225], v[242:249], v[46:49], v149, v213 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	ds_read_b128 v[26:29], v203 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[34:37], v203 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[42:45], v203 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[218:221], v203 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[30:33], v202 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[38:41], v202 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[46:49], v202 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[222:225], v202 offset:0x1800

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
	v_lshrrev_b32_e32 v146, 16, v212
	v_lshrrev_b32_e32 v59, 16, v148
	s_nop 0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[162:165], v[26:33], v[2:9], v[162:165], v58, v146 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[202:205], v[26:33], v[10:17], v[166:169], v58, v146 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[206:209], v[34:41], v[2:9], v[170:173], v58, v146 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[166:169], v[34:41], v[10:17], v[174:177], v58, v146 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[170:173], v[42:49], v[2:9], v[142:145], v59, v146 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[150:153], v[42:49], v[10:17], v[150:153], v59, v146 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[154:157], v[218:225], v[2:9], v[154:157], v59, v146 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[142:145], v[218:225], v[10:17], v[158:161], v59, v146 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[226:229], v217 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[234:237], v217 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[230:233], v216 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[238:241], v216 offset:0x800

	;;#ASMEND
	s_barrier
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_lshrrev_b32_e32 v148, 16, v213
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[26:33], v[226:233], v[82:85], v58, v148 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[26:33], v[234:241], v[86:89], v58, v148 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[34:41], v[226:233], v[122:125], v58, v148 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[34:41], v[234:241], v[126:129], v58, v148 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[42:49], v[226:233], v[130:133], v59, v148 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[42:49], v[234:241], v[134:137], v59, v148 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[218:225], v[226:233], v[138:141], v59, v148 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[218:225], v[234:241], v[118:121], v59, v148 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	ds_read_b128 v[118:121], v215 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[126:129], v215 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[134:137], v215 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[216:219], v215 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[122:125], v214 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[130:133], v214 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[138:141], v214 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[220:223], v214 offset:0x1800

	;;#ASMEND
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_lshrrev_b32_e32 v147, 16, v147
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[118:125], v[2:9], v[106:109], v147, v146 op_sel_hi:[0,0,0]
	s_nop 6
	v_lshrrev_b32_e32 v106, 16, v149
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], v[118:125], v[10:17], v[110:113], v147, v146 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[126:133], v[2:9], v[62:65], v147, v146 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[126:133], v[10:17], v[114:117], v147, v146 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[134:141], v[2:9], v[54:57], v106, v146 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[134:141], v[10:17], v[98:101], v106, v146 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[216:223], v[2:9], v[50:53], v106, v146 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[216:223], v[10:17], v[102:105], v106, v146 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[118:125], v[226:233], v[18:21], v147, v148 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[118:125], v[234:241], v[22:25], v147, v148 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[126:133], v[226:233], v[178:181], v147, v148 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[126:133], v[234:241], v[182:185], v147, v148 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[134:141], v[226:233], v[186:189], v106, v148 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[134:141], v[234:241], v[190:193], v106, v148 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[6:9], v[216:223], v[226:233], v[194:197], v106, v148 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[2:5], v[216:223], v[234:241], v[198:201], v106, v148 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_movk_i32 s0, 0x100
	v_cmp_gt_u32_e32 vcc, s0, v0
	s_barrier
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB2_10
; %bb.9:
	s_barrier
