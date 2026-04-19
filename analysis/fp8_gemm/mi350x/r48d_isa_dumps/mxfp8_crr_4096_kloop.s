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
