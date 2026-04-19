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
