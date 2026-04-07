// Gluon a4w4 inner loop body (auto-extracted)
// 256 MFMAs, 72 ds_reads, 32 tile_loads
#define GLUON_LOOP_BODY_ASM \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[124:127], v[116:119], a[132:135], v32, v30, op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "ds_read_b128 v[164:167], v247\n" \
    "ds_read_b128 v[168:171], v247, offset:64\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[132:135], v[120:123], a[132:135], v32, v30, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "ds_read_b128 v[172:175], v247, offset:256\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[128:131], v[116:119], a[140:143], v32, v30, op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "ds_read_b128 v[176:179], v247, offset:320\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[136:139], v[120:123], a[140:143], v32, v30, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "ds_read_b128 v[180:183], v247, offset:512\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[140:143], v[116:119], a[144:147], v33, v30, op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "ds_read_b128 v[184:187], v247, offset:576\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[144:147], v[120:123], a[144:147], v33, v30, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "ds_read_b128 v[188:191], v247, offset:768\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[148:151], v[116:119], a[152:155], v33, v30, op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "ds_read_b128 v[192:195], v247, offset:832\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[152:155], v[120:123], a[152:155], v33, v30, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "s_waitcnt vmcnt(19)\n" \
    "ds_write_b32 v59, v162\n" \
    "s_and_b32 s1, s67, 0xffff\n" \
    "s_mov_b32 s0, s24\n" \
    "s_mov_b32 m0, s25\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[124:127], v[108:111], a[128:131], v32, v30, op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "buffer_load_dwordx4 v2, s[0:3], 0, offen, lds\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[132:135], v[112:115], a[128:131], v32, v30, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[128:131], v[108:111], a[136:139], v32, v30, op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[136:139], v[112:115], a[136:139], v32, v30, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "s_mov_b32 m0, s35\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[140:143], v[108:111], a[148:151], v33, v30, op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "buffer_load_dwordx4 v4, s[0:3], 0, offen, lds\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[144:147], v[112:115], a[148:151], v33, v30, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[148:151], v[108:111], a[156:159], v33, v30, op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[152:155], v[112:115], a[156:159], v33, v30, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "s_mov_b32 m0, s36\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[124:127], v[100:103], a[160:163], v32, v31, op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "buffer_load_dwordx4 v6, s[0:3], 0, offen, lds\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[132:135], v[104:107], a[160:163], v32, v31, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[128:131], v[100:103], a[164:167], v32, v31, op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[136:139], v[104:107], a[164:167], v32, v31, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "s_mov_b32 m0, s37\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[140:143], v[100:103], a[168:171], v33, v31, op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "buffer_load_dwordx4 v8, s[0:3], 0, offen, lds\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[144:147], v[104:107], a[168:171], v33, v31, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[148:151], v[100:103], a[172:175], v33, v31, op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[152:155], v[104:107], a[172:175], v33, v31, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "s_mov_b32 m0, s38\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[124:127], v[88:91], a[176:179], v32, v31, op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "buffer_load_dwordx4 v10, s[0:3], 0, offen, lds\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[132:135], v[96:99], a[176:179], v32, v31, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[128:131], v[88:91], a[180:183], v32, v31, op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[136:139], v[96:99], a[180:183], v32, v31, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "s_mov_b32 m0, s39\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[140:143], v[88:91], a[184:187], v33, v31, op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "buffer_load_dwordx4 v12, s[0:3], 0, offen, lds\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[144:147], v[96:99], a[184:187], v33, v31, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[148:151], v[88:91], a[188:191], v33, v31, op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[152:155], v[96:99], a[188:191], v33, v31, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "s_mov_b32 m0, s40\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[124:127], v[84:87], a[192:195], v32, v0, op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "buffer_load_dwordx4 v14, s[0:3], 0, offen, lds\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[132:135], v[92:95], a[192:195], v32, v0, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[128:131], v[84:87], a[196:199], v32, v0, op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[136:139], v[92:95], a[196:199], v32, v0, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "s_mov_b32 m0, s41\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[140:143], v[84:87], a[204:207], v33, v0, op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "buffer_load_dwordx4 v16, s[0:3], 0, offen, lds\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[144:147], v[92:95], a[204:207], v33, v0, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[148:151], v[84:87], a[212:215], v33, v0, op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[152:155], v[92:95], a[212:215], v33, v0, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "s_and_b32 s5, s66, 0xffff\n" \
    "s_mov_b32 s4, s65\n" \
    "s_mov_b32 m0, s42\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[124:127], v[76:79], a[200:203], v32, v0, op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "buffer_load_dwordx4 v18, s[4:7], 0, offen, lds\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[132:135], v[80:83], a[200:203], v32, v0, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[128:131], v[76:79], a[208:211], v32, v0, op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[136:139], v[80:83], a[208:211], v32, v0, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "s_mov_b32 m0, s43\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[140:143], v[76:79], a[216:219], v33, v0, op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "buffer_load_dwordx4 v20, s[4:7], 0, offen, lds\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[144:147], v[80:83], a[216:219], v33, v0, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[148:151], v[76:79], a[220:223], v33, v0, op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[152:155], v[80:83], a[220:223], v33, v0, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "s_mov_b32 m0, s44\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[124:127], v[68:71], a[224:227], v32, v1, op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "buffer_load_dwordx4 v22, s[4:7], 0, offen, lds\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[132:135], v[72:75], a[224:227], v32, v1, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[128:131], v[68:71], a[232:235], v32, v1, op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[136:139], v[72:75], a[232:235], v32, v1, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "s_mov_b32 m0, s45\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[140:143], v[68:71], a[240:243], v33, v1, op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "buffer_load_dwordx4 v24, s[4:7], 0, offen, lds\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[144:147], v[72:75], a[240:243], v33, v1, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[148:151], v[68:71], a[248:251], v33, v1, op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[152:155], v[72:75], a[248:251], v33, v1, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "s_and_b32 s17, s15, 0xffff\n" \
    "s_mov_b32 s16, s14\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[124:127], v[60:63], a[252:255], v32, v1, op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "buffer_load_dwordx2 v[158:159], v26, s[16:19], 0, offen\n" \
    "s_waitcnt lgkmcnt(0)\n" \
    "s_barrier\n" \
    "ds_read_b64_tr_b8 v[124:125], v50\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[132:135], v[64:67], a[252:255], v32, v1, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "s_and_b32 s9, s21, 0xffff\n" \
    "s_mov_b32 s8, s20\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[128:131], v[60:63], a[228:231], v32, v1, op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "buffer_load_dword v244, v28, s[8:11], 0, offen\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[136:139], v[64:67], a[228:231], v32, v1, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[140:143], v[60:63], a[236:239], v33, v1, op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[144:147], v[64:67], a[236:239], v33, v1, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[148:151], v[60:63], a[244:247], v33, v1, op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "s_waitcnt vmcnt(21), lgkmcnt(0)\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[152:155], v[64:67], a[244:247], v33, v1, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "s_barrier\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[164:167], v[116:119], a[124:127], v124, v30, op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "ds_read_b128 v[126:129], v46, offset:33792\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[168:171], v[120:123], a[124:127], v124, v30, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "ds_read_b128 v[130:133], v46, offset:33856\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[172:175], v[116:119], a[0:3], v124, v30, op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "ds_read_b128 v[134:137], v46, offset:34048\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[176:179], v[120:123], a[0:3], v124, v30, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "ds_read_b128 v[138:141], v46, offset:34112\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[180:183], v[116:119], a[4:7], v125, v30, op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "ds_read_b128 v[148:151], v46, offset:34304\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[184:187], v[120:123], a[4:7], v125, v30, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "ds_read_b128 v[152:155], v46, offset:34368\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[188:191], v[116:119], a[8:11], v125, v30, op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "ds_read_b128 v[196:199], v46, offset:34560\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[192:195], v[120:123], a[8:11], v125, v30, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "ds_read_b128 v[200:203], v46, offset:34624\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[164:167], v[108:111], a[12:15], v124, v30, op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "ds_read_b128 v[204:207], v46, offset:50688\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[168:171], v[112:115], a[12:15], v124, v30, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "ds_read_b128 v[208:211], v46, offset:50752\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[172:175], v[108:111], a[16:19], v124, v30, op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "ds_read_b128 v[212:215], v46, offset:50944\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[176:179], v[112:115], a[16:19], v124, v30, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "ds_read_b128 v[216:219], v46, offset:51008\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[180:183], v[108:111], a[20:23], v125, v30, op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "ds_read_b128 v[220:223], v46, offset:51200\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[184:187], v[112:115], a[20:23], v125, v30, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "ds_read_b128 v[224:227], v46, offset:51264\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[188:191], v[108:111], a[24:27], v125, v30, op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "ds_read_b128 v[228:231], v46, offset:51456\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[192:195], v[112:115], a[24:27], v125, v30, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "ds_read_b128 v[232:235], v46, offset:51520\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[164:167], v[100:103], a[28:31], v124, v31, op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "ds_read_b128 v[108:111], v245\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[168:171], v[104:107], a[28:31], v124, v31, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "ds_read_b128 v[112:115], v245, offset:64\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[172:175], v[100:103], a[32:35], v124, v31, op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "ds_read_b128 v[116:119], v245, offset:256\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[176:179], v[104:107], a[32:35], v124, v31, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "ds_read_b128 v[120:123], v245, offset:320\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[180:183], v[100:103], a[36:39], v125, v31, op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "ds_read_b128 v[142:145], v245, offset:512\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[184:187], v[104:107], a[36:39], v125, v31, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "ds_read_b128 v[236:239], v245, offset:576\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[188:191], v[100:103], a[44:47], v125, v31, op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "ds_read_b128 v[100:103], v245, offset:768\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[192:195], v[104:107], a[44:47], v125, v31, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "ds_read_b128 v[104:107], v245, offset:832\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[164:167], v[88:91], a[52:55], v124, v31, op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "s_waitcnt vmcnt(20)\n" \
    "ds_write_b64 v163, v[156:157]\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[168:171], v[96:99], a[52:55], v124, v31, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "s_waitcnt vmcnt(19)\n" \
    "ds_write_b32 v59, v160\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[172:175], v[88:91], a[40:43], v124, v31, op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[176:179], v[96:99], a[40:43], v124, v31, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[180:183], v[88:91], a[48:51], v125, v31, op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[184:187], v[96:99], a[48:51], v125, v31, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[188:191], v[88:91], a[56:59], v125, v31, op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[192:195], v[96:99], a[56:59], v125, v31, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[164:167], v[84:87], a[60:63], v124, v0, op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[168:171], v[92:95], a[60:63], v124, v0, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[172:175], v[84:87], a[64:67], v124, v0, op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[176:179], v[92:95], a[64:67], v124, v0, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[180:183], v[84:87], a[68:71], v125, v0, op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[184:187], v[92:95], a[68:71], v125, v0, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[188:191], v[84:87], a[72:75], v125, v0, op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[192:195], v[92:95], a[72:75], v125, v0, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[164:167], v[76:79], a[76:79], v124, v0, op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[168:171], v[80:83], a[76:79], v124, v0, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "s_mov_b32 m0, s46\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[172:175], v[76:79], a[80:83], v124, v0, op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "buffer_load_dwordx4 v19, s[4:7], 0, offen, lds\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[176:179], v[80:83], a[80:83], v124, v0, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[180:183], v[76:79], a[84:87], v125, v0, op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[184:187], v[80:83], a[84:87], v125, v0, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "s_mov_b32 m0, s47\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[188:191], v[76:79], a[88:91], v125, v0, op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "buffer_load_dwordx4 v21, s[4:7], 0, offen, lds\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[192:195], v[80:83], a[88:91], v125, v0, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[164:167], v[68:71], a[92:95], v124, v1, op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[168:171], v[72:75], a[92:95], v124, v1, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "s_mov_b32 m0, s48\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[172:175], v[68:71], a[96:99], v124, v1, op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "buffer_load_dwordx4 v23, s[4:7], 0, offen, lds\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[176:179], v[72:75], a[96:99], v124, v1, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[180:183], v[68:71], a[100:103], v125, v1, op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[184:187], v[72:75], a[100:103], v125, v1, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "s_mov_b32 m0, s49\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[188:191], v[68:71], a[104:107], v125, v1, op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "buffer_load_dwordx4 v25, s[4:7], 0, offen, lds\n" \
    "s_waitcnt lgkmcnt(0)\n" \
    "s_barrier\n" \
    "ds_read_b64_tr_b8 v[30:31], v51\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[192:195], v[72:75], a[104:107], v125, v1, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "ds_read_b64_tr_b8 v[34:35], v51, offset:128\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[164:167], v[60:63], a[108:111], v124, v1, op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "ds_read_b64_tr_b8 v[32:33], v50\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[168:171], v[64:67], a[108:111], v124, v1, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[172:175], v[60:63], a[112:115], v124, v1, op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "buffer_load_dword v162, v47, s[8:11], 0, offen\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[176:179], v[64:67], a[112:115], v124, v1, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[180:183], v[60:63], a[116:119], v125, v1, op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[184:187], v[64:67], a[116:119], v125, v1, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[188:191], v[60:63], a[120:123], v125, v1, op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "s_waitcnt vmcnt(20), lgkmcnt(0)\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[192:195], v[64:67], a[120:123], v125, v1, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "s_barrier\n" \
    "ds_read_b128 v[166:169], v246\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[108:111], v[126:129], a[132:135], v32, v30, op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "ds_read_b128 v[170:173], v246, offset:64\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[112:115], v[130:133], a[132:135], v32, v30, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "ds_read_b128 v[174:177], v246, offset:256\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[116:119], v[126:129], a[140:143], v32, v30, op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "ds_read_b128 v[178:181], v246, offset:320\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[120:123], v[130:133], a[140:143], v32, v30, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "ds_read_b128 v[182:185], v246, offset:512\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[142:145], v[126:129], a[144:147], v33, v30, op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "ds_read_b128 v[186:189], v246, offset:576\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[236:239], v[130:133], a[144:147], v33, v30, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "ds_read_b128 v[190:193], v246, offset:768\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[100:103], v[126:129], a[152:155], v33, v30, op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "ds_read_b128 v[240:243], v246, offset:832\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[104:107], v[130:133], a[152:155], v33, v30, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "s_waitcnt vmcnt(19)\n" \
    "ds_write_b32 v59, v161\n" \
    "s_mov_b32 m0, s50\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[108:111], v[134:137], a[128:131], v32, v30, op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "buffer_load_dwordx4 v3, s[0:3], 0, offen, lds\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[112:115], v[138:141], a[128:131], v32, v30, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[116:119], v[134:137], a[136:139], v32, v30, op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[120:123], v[138:141], a[136:139], v32, v30, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "s_mov_b32 m0, s51\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[142:145], v[134:137], a[148:151], v33, v30, op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "buffer_load_dwordx4 v5, s[0:3], 0, offen, lds\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[236:239], v[138:141], a[148:151], v33, v30, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[100:103], v[134:137], a[156:159], v33, v30, op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[104:107], v[138:141], a[156:159], v33, v30, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "s_mov_b32 m0, s52\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[108:111], v[148:151], a[160:163], v32, v31, op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "buffer_load_dwordx4 v7, s[0:3], 0, offen, lds\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[112:115], v[152:155], a[160:163], v32, v31, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[116:119], v[148:151], a[164:167], v32, v31, op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[120:123], v[152:155], a[164:167], v32, v31, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "s_mov_b32 m0, s53\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[142:145], v[148:151], a[168:171], v33, v31, op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "buffer_load_dwordx4 v9, s[0:3], 0, offen, lds\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[236:239], v[152:155], a[168:171], v33, v31, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[100:103], v[148:151], a[172:175], v33, v31, op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[104:107], v[152:155], a[172:175], v33, v31, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "s_mov_b32 m0, s54\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[108:111], v[196:199], a[176:179], v32, v31, op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "buffer_load_dwordx4 v11, s[0:3], 0, offen, lds\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[112:115], v[200:203], a[176:179], v32, v31, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[116:119], v[196:199], a[180:183], v32, v31, op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[120:123], v[200:203], a[180:183], v32, v31, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "s_mov_b32 m0, s55\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[142:145], v[196:199], a[184:187], v33, v31, op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "buffer_load_dwordx4 v13, s[0:3], 0, offen, lds\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[236:239], v[200:203], a[184:187], v33, v31, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[100:103], v[196:199], a[188:191], v33, v31, op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[104:107], v[200:203], a[188:191], v33, v31, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "s_mov_b32 m0, s56\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[108:111], v[204:207], a[192:195], v32, v34, op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "buffer_load_dwordx4 v15, s[0:3], 0, offen, lds\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[112:115], v[208:211], a[192:195], v32, v34, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[116:119], v[204:207], a[196:199], v32, v34, op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[120:123], v[208:211], a[196:199], v32, v34, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "s_mov_b32 m0, s57\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[142:145], v[204:207], a[204:207], v33, v34, op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "buffer_load_dwordx4 v17, s[0:3], 0, offen, lds\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[236:239], v[208:211], a[204:207], v33, v34, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[100:103], v[204:207], a[212:215], v33, v34, op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[104:107], v[208:211], a[212:215], v33, v34, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "s_mov_b32 m0, s58\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[108:111], v[212:215], a[200:203], v32, v34, op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "buffer_load_dwordx4 v38, s[4:7], 0, offen, lds\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[112:115], v[216:219], a[200:203], v32, v34, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[116:119], v[212:215], a[208:211], v32, v34, op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[120:123], v[216:219], a[208:211], v32, v34, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "s_mov_b32 m0, s59\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[142:145], v[212:215], a[216:219], v33, v34, op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "buffer_load_dwordx4 v39, s[4:7], 0, offen, lds\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[236:239], v[216:219], a[216:219], v33, v34, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[100:103], v[212:215], a[220:223], v33, v34, op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[104:107], v[216:219], a[220:223], v33, v34, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "s_mov_b32 m0, s60\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[108:111], v[220:223], a[224:227], v32, v35, op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "buffer_load_dwordx4 v40, s[4:7], 0, offen, lds\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[112:115], v[224:227], a[224:227], v32, v35, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[116:119], v[220:223], a[232:235], v32, v35, op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[120:123], v[224:227], a[232:235], v32, v35, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "s_mov_b32 m0, s61\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[142:145], v[220:223], a[240:243], v33, v35, op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "buffer_load_dwordx4 v41, s[4:7], 0, offen, lds\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[236:239], v[224:227], a[240:243], v33, v35, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[100:103], v[220:223], a[248:251], v33, v35, op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[104:107], v[224:227], a[248:251], v33, v35, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[108:111], v[228:231], a[252:255], v32, v35, op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "buffer_load_dwordx2 v[156:157], v27, s[16:19], 0, offen\n" \
    "s_waitcnt lgkmcnt(0)\n" \
    "s_barrier\n" \
    "ds_read_b64_tr_b8 v[164:165], v50\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[112:115], v[232:235], a[252:255], v32, v35, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[116:119], v[228:231], a[228:231], v32, v35, op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "buffer_load_dword v160, v48, s[8:11], 0, offen\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[120:123], v[232:235], a[228:231], v32, v35, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[142:145], v[228:231], a[236:239], v33, v35, op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[236:239], v[232:235], a[236:239], v33, v35, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[100:103], v[228:231], a[244:247], v33, v35, op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "s_waitcnt vmcnt(21), lgkmcnt(0)\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[104:107], v[232:235], a[244:247], v33, v35, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "s_barrier\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[166:169], v[126:129], a[124:127], v164, v30, op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "ds_read_b128 v[116:119], v46\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[170:173], v[130:133], a[124:127], v164, v30, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "ds_read_b128 v[120:123], v46, offset:64\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[174:177], v[126:129], a[0:3], v164, v30, op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "ds_read_b128 v[108:111], v46, offset:256\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[178:181], v[130:133], a[0:3], v164, v30, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "ds_read_b128 v[112:115], v46, offset:320\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[182:185], v[126:129], a[4:7], v165, v30, op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "ds_read_b128 v[100:103], v46, offset:512\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[186:189], v[130:133], a[4:7], v165, v30, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "ds_read_b128 v[104:107], v46, offset:576\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[190:193], v[126:129], a[8:11], v165, v30, op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "ds_read_b128 v[88:91], v46, offset:768\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[240:243], v[130:133], a[8:11], v165, v30, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "ds_read_b128 v[96:99], v46, offset:832\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[166:169], v[134:137], a[12:15], v164, v30, op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "ds_read_b128 v[84:87], v46, offset:16896\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[170:173], v[138:141], a[12:15], v164, v30, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "ds_read_b128 v[92:95], v46, offset:16960\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[174:177], v[134:137], a[16:19], v164, v30, op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "ds_read_b128 v[76:79], v46, offset:17152\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[178:181], v[138:141], a[16:19], v164, v30, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "ds_read_b128 v[80:83], v46, offset:17216\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[182:185], v[134:137], a[20:23], v165, v30, op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "ds_read_b128 v[68:71], v46, offset:17408\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[186:189], v[138:141], a[20:23], v165, v30, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "ds_read_b128 v[72:75], v46, offset:17472\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[190:193], v[134:137], a[24:27], v165, v30, op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "ds_read_b128 v[60:63], v46, offset:17664\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[240:243], v[138:141], a[24:27], v165, v30, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "ds_read_b128 v[64:67], v46, offset:17728\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[166:169], v[148:151], a[28:31], v164, v31, op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "ds_read_b128 v[124:127], v248\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[170:173], v[152:155], a[28:31], v164, v31, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "ds_read_b128 v[132:135], v248, offset:64\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[174:177], v[148:151], a[32:35], v164, v31, op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "ds_read_b128 v[128:131], v248, offset:256\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[178:181], v[152:155], a[32:35], v164, v31, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "ds_read_b128 v[136:139], v248, offset:320\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[182:185], v[148:151], a[36:39], v165, v31, op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "ds_read_b128 v[140:143], v248, offset:512\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[186:189], v[152:155], a[36:39], v165, v31, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "ds_read_b128 v[144:147], v248, offset:576\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[190:193], v[148:151], a[44:47], v165, v31, op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "ds_read_b128 v[148:151], v248, offset:768\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[240:243], v[152:155], a[44:47], v165, v31, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "ds_read_b128 v[152:155], v248, offset:832\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[166:169], v[196:199], a[52:55], v164, v31, op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "s_waitcnt vmcnt(20)\n" \
    "ds_write_b64 v163, v[158:159]\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[170:173], v[200:203], a[52:55], v164, v31, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "s_waitcnt vmcnt(19)\n" \
    "ds_write_b32 v59, v244\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[174:177], v[196:199], a[40:43], v164, v31, op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[178:181], v[200:203], a[40:43], v164, v31, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[182:185], v[196:199], a[48:51], v165, v31, op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[186:189], v[200:203], a[48:51], v165, v31, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[190:193], v[196:199], a[56:59], v165, v31, op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[240:243], v[200:203], a[56:59], v165, v31, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[166:169], v[204:207], a[60:63], v164, v34, op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[170:173], v[208:211], a[60:63], v164, v34, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[174:177], v[204:207], a[64:67], v164, v34, op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[178:181], v[208:211], a[64:67], v164, v34, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[182:185], v[204:207], a[68:71], v165, v34, op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[186:189], v[208:211], a[68:71], v165, v34, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[190:193], v[204:207], a[72:75], v165, v34, op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[240:243], v[208:211], a[72:75], v165, v34, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[166:169], v[212:215], a[76:79], v164, v34, op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[170:173], v[216:219], a[76:79], v164, v34, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "s_mov_b32 m0, s28\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[174:177], v[212:215], a[80:83], v164, v34, op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "buffer_load_dwordx4 v42, s[4:7], 0, offen, lds\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[178:181], v[216:219], a[80:83], v164, v34, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[182:185], v[212:215], a[84:87], v165, v34, op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[186:189], v[216:219], a[84:87], v165, v34, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "s_mov_b32 m0, s29\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[190:193], v[212:215], a[88:91], v165, v34, op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "buffer_load_dwordx4 v43, s[4:7], 0, offen, lds\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[240:243], v[216:219], a[88:91], v165, v34, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[166:169], v[220:223], a[92:95], v164, v35, op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[170:173], v[224:227], a[92:95], v164, v35, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "s_mov_b32 m0, s30\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[174:177], v[220:223], a[96:99], v164, v35, op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "buffer_load_dwordx4 v44, s[4:7], 0, offen, lds\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[178:181], v[224:227], a[96:99], v164, v35, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[182:185], v[220:223], a[100:103], v165, v35, op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[186:189], v[224:227], a[100:103], v165, v35, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "s_mov_b32 m0, s31\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[190:193], v[220:223], a[104:107], v165, v35, op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "buffer_load_dwordx4 v45, s[4:7], 0, offen, lds\n" \
    "s_waitcnt lgkmcnt(0)\n" \
    "s_barrier\n" \
    "ds_read_b64_tr_b8 v[30:31], v51\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[240:243], v[224:227], a[104:107], v165, v35, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "ds_read_b64_tr_b8 v[0:1], v51, offset:128\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[166:169], v[228:231], a[108:111], v164, v35, op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "ds_read_b64_tr_b8 v[32:33], v50\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[170:173], v[232:235], a[108:111], v164, v35, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[174:177], v[228:231], a[112:115], v164, v35, op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "buffer_load_dword v161, v49, s[8:11], 0, offen\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[178:181], v[232:235], a[112:115], v164, v35, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "s_add_u32 s65, s65, 0x100\n" \
    "s_addc_u32 s66, s66, 0\n" \
    "s_add_u32 s24, s24, 0x100\n" \
    "s_addc_u32 s67, s67, 0\n" \
    "s_add_u32 s20, s20, s63\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[182:185], v[228:231], a[116:119], v165, v35, op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "s_addc_u32 s21, s21, s64\n" \
    "s_add_u32 s14, s14, s27\n" \
    "s_addc_u32 s15, s15, s62\n" \
    "s_add_i32 s68, s68, 2\n" \
    "s_cmp_lt_u32 s68, 28\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[186:189], v[232:235], a[116:119], v165, v35, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "s_waitcnt vmcnt(20), lgkmcnt(0)\n" \
    "s_barrier\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[190:193], v[228:231], a[120:123], v165, v35, op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[240:243], v[232:235], a[120:123], v165, v35, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4\n"
