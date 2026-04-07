// Auto-generated interleaved ds_read + MFMA asm block
// 32 ds_reads + 128 MFMAs
#define INTERLEAVED_COMPUTE_ASM \
    /* -- Phase A: initial ds_reads -- */ \
    "ds_read_b128 v[100:103], v3 offset:0\n" \
    "ds_read_b128 v[104:107], v3 offset:0x800\n" \
    "ds_read_b128 v[108:111], v3 offset:0x1000\n" \
    "ds_read_b128 v[112:115], v3 offset:0x1800\n" \
    "ds_read_b128 v[30:33], v2 offset:0\n" \
    "ds_read_b128 v[26:29], v2 offset:0x800\n" \
    "ds_read_b128 v[22:25], v2 offset:0x1000\n" \
    "ds_read_b128 v[18:21], v2 offset:0x1800\n" \
    "ds_read_b128 v[116:119], v3 offset:0\n" \
    "ds_read_b128 v[74:77], v3 offset:0x800\n" \
    "ds_read_b128 v[70:73], v3 offset:0x1000\n" \
    "ds_read_b128 v[66:69], v3 offset:0x1800\n" \
    "ds_read_b128 v[14:17], v2 offset:0\n" \
    "ds_read_b128 v[10:13], v2 offset:0x800\n" \
    "ds_read_b128 v[6:9], v2 offset:0x1000\n" \
    "ds_read_b128 v[2:5], v2 offset:0x1800\n" \
    "s_waitcnt lgkmcnt(0) vmcnt(8)\n" \
    /* -- Phase B: interleaved ds_reads + MFMAs -- */ \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[100:103], v[120:123], a[0:3], v97, v98 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[100:103], v[124:127], a[4:7], v97, v98 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[100:103], v[128:131], a[8:11], v97, v96 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[100:103], v[132:135], a[12:15], v97, v96 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[104:107], v[120:123], a[16:19], v97, v98 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[104:107], v[124:127], a[20:23], v97, v98 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[104:107], v[128:131], a[24:27], v97, v96 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[104:107], v[132:135], a[28:31], v97, v96 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "ds_read_b128 v[120:123], v35 offset:0\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[108:111], v[120:123], a[32:35], v95, v98 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[108:111], v[124:127], a[36:39], v95, v98 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[108:111], v[128:131], a[40:43], v95, v96 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[108:111], v[132:135], a[44:47], v95, v96 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[112:115], v[120:123], a[48:51], v95, v98 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[112:115], v[124:127], a[52:55], v95, v98 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[112:115], v[128:131], a[56:59], v95, v96 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[112:115], v[132:135], a[60:63], v95, v96 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "ds_read_b128 v[124:127], v35 offset:0x800\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[100:103], v[136:139], a[80:83], v97, v93 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[100:103], v[140:143], a[84:87], v97, v93 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[100:103], v[144:147], a[88:91], v97, v94 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[100:103], v[148:151], a[92:95], v97, v94 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[104:107], v[136:139], a[96:99], v97, v93 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[104:107], v[140:143], a[100:103], v97, v93 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[104:107], v[144:147], a[104:107], v97, v94 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[104:107], v[148:151], a[108:111], v97, v94 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "ds_read_b128 v[128:131], v35 offset:0x1000\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[108:111], v[136:139], a[112:115], v95, v93 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[108:111], v[140:143], a[116:119], v95, v93 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[108:111], v[144:147], a[120:123], v95, v94 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[108:111], v[148:151], a[124:127], v95, v94 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[112:115], v[136:139], a[64:67], v95, v93 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[112:115], v[144:147], a[72:75], v95, v94 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[112:115], v[148:151], a[76:79], v95, v94 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[112:115], v[140:143], a[80:83], v95, v93 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "ds_read_b128 v[132:135], v35 offset:0x1800\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[116:119], v[120:123], a[128:131], v92, v98 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[116:119], v[124:127], a[132:135], v92, v98 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[116:119], v[128:131], a[136:139], v92, v96 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[116:119], v[132:135], a[140:143], v92, v96 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[74:77], v[120:123], a[144:147], v92, v98 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[74:77], v[124:127], a[148:151], v92, v98 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[74:77], v[128:131], a[152:155], v92, v96 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[74:77], v[132:135], a[156:159], v92, v96 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "ds_read_b128 v[50:53], v34 offset:0\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[70:73], v[120:123], a[160:163], v91, v98 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[70:73], v[124:127], a[164:167], v91, v98 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[70:73], v[128:131], a[168:171], v91, v96 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[70:73], v[132:135], a[172:175], v91, v96 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[66:69], v[120:123], a[176:179], v91, v98 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[66:69], v[124:127], a[180:183], v91, v98 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[66:69], v[128:131], a[184:187], v91, v96 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[66:69], v[132:135], a[188:191], v91, v96 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "ds_read_b128 v[54:57], v34 offset:0x800\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[116:119], v[136:139], a[192:195], v92, v93 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[116:119], v[140:143], a[196:199], v92, v93 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[116:119], v[144:147], a[200:203], v92, v94 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[116:119], v[148:151], a[204:207], v92, v94 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[74:77], v[136:139], a[208:211], v92, v93 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[74:77], v[140:143], a[212:215], v92, v93 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[74:77], v[144:147], a[216:219], v92, v94 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[74:77], v[148:151], a[220:223], v92, v94 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "ds_read_b128 v[58:61], v34 offset:0x1000\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[70:73], v[136:139], a[224:227], v91, v93 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[70:73], v[140:143], a[228:231], v91, v93 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[70:73], v[144:147], a[232:235], v91, v94 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[70:73], v[148:151], a[236:239], v91, v94 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[66:69], v[136:139], a[240:243], v91, v93 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[66:69], v[140:143], a[244:247], v91, v93 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[66:69], v[144:147], a[248:251], v91, v94 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[66:69], v[148:151], a[252:255], v91, v94 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "ds_read_b128 v[62:65], v34 offset:0x1800\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[30:33], v[50:53], a[0:3], v66, v67 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[30:33], v[54:57], a[4:7], v66, v67 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[30:33], v[58:61], a[8:11], v66, v68 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[30:33], v[62:65], a[12:15], v66, v68 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[26:29], v[50:53], a[16:19], v66, v67 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[26:29], v[54:57], a[20:23], v66, v67 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[26:29], v[58:61], a[24:27], v66, v68 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[26:29], v[62:65], a[28:31], v66, v68 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "ds_read_b128 v[136:139], v35 offset:0\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[22:25], v[50:53], a[32:35], v69, v67 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[22:25], v[54:57], a[36:39], v69, v67 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[22:25], v[58:61], a[40:43], v69, v68 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[22:25], v[62:65], a[44:47], v69, v68 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[18:21], v[50:53], a[48:51], v69, v67 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[18:21], v[54:57], a[52:55], v69, v67 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[18:21], v[58:61], a[56:59], v69, v68 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[18:21], v[62:65], a[60:63], v69, v68 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "ds_read_b128 v[140:143], v35 offset:0x800\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[30:33], v[34:37], a[68:71], v66, v70 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[30:33], v[38:41], a[84:87], v66, v70 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[30:33], v[42:45], a[88:91], v66, v71 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[30:33], v[46:49], a[92:95], v66, v71 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[26:29], v[34:37], a[96:99], v66, v70 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[26:29], v[38:41], a[100:103], v66, v70 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[26:29], v[42:45], a[104:107], v66, v71 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[26:29], v[46:49], a[108:111], v66, v71 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "ds_read_b128 v[144:147], v35 offset:0x1000\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[22:25], v[34:37], a[112:115], v69, v70 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[22:25], v[38:41], a[116:119], v69, v70 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[22:25], v[42:45], a[120:123], v69, v71 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[22:25], v[46:49], a[124:127], v69, v71 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[18:21], v[34:37], a[64:67], v69, v70 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[18:21], v[38:41], a[80:83], v69, v70 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[18:21], v[42:45], a[72:75], v69, v71 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[18:21], v[46:49], a[76:79], v69, v71 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "ds_read_b128 v[148:151], v35 offset:0x1800\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[14:17], v[50:53], a[128:131], v18, v67 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[14:17], v[54:57], a[132:135], v18, v67 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[14:17], v[58:61], a[136:139], v18, v68 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[14:17], v[62:65], a[140:143], v18, v68 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[10:13], v[50:53], a[144:147], v18, v67 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[10:13], v[54:57], a[148:151], v18, v67 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[10:13], v[58:61], a[152:155], v18, v68 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[10:13], v[62:65], a[156:159], v18, v68 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "ds_read_b128 v[34:37], v46 offset:0\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[6:9], v[50:53], a[160:163], v19, v67 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[6:9], v[54:57], a[164:167], v19, v67 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[6:9], v[58:61], a[168:171], v19, v68 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[6:9], v[62:65], a[172:175], v19, v68 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[2:5], v[50:53], a[176:179], v19, v67 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[2:5], v[54:57], a[180:183], v19, v67 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[2:5], v[58:61], a[184:187], v19, v68 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[2:5], v[62:65], a[188:191], v19, v68 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "ds_read_b128 v[38:41], v46 offset:0x800\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[14:17], v[34:37], a[192:195], v18, v70 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[14:17], v[38:41], a[196:199], v18, v70 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[14:17], v[42:45], a[200:203], v18, v71 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[14:17], v[46:49], a[204:207], v18, v71 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[10:13], v[34:37], a[208:211], v18, v70 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[10:13], v[38:41], a[212:215], v18, v70 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[10:13], v[42:45], a[216:219], v18, v71 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[10:13], v[46:49], a[220:223], v18, v71 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "ds_read_b128 v[42:45], v46 offset:0x1000\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[6:9], v[34:37], a[224:227], v19, v70 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[6:9], v[38:41], a[228:231], v19, v70 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[6:9], v[42:45], a[232:235], v19, v71 op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[6:9], v[46:49], a[236:239], v19, v71 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[2:5], v[34:37], a[240:243], v19, v70 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[2:5], v[38:41], a[244:247], v19, v70 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[2:5], v[42:45], a[248:251], v19, v71 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[2:5], v[46:49], a[252:255], v19, v71 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4\n" \
    "ds_read_b128 v[46:49], v46 offset:0x1800\n"
