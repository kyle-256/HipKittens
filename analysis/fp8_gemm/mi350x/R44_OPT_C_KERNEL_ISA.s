=== KERNEL ISA EXCERPT around back-edge (PC 0x1A884–0x1A9D0) ===
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[188:191], v[116:119], a[12:15], v107, v110 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 00000001A804: D3AC0800 0002DD6B D3AD8C0C 8432E9BC
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[188:191], v[120:123], a[8:11], v107, v110 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 00000001A814: D3AC1800 0002DD6B D3AD8C08 8422F1BC
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[188:191], v[152:155], a[4:7], v107, v111 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 00000001A824: D3AC0800 0002DF6B D3AD8C04 841331BC
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[188:191], v[156:159], a[0:3], v107, v111 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 00000001A834: D3AC1800 0002DF6B D3AD8C00 840339BC
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[204:207], v[160:163], a[12:15], v107, v110 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 00000001A844: D3AC0800 1802DD6B D3AD8C0C 843341CC
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[204:207], v[164:167], a[8:11], v107, v110 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 00000001A854: D3AC1800 1802DD6B D3AD8C08 842349CC
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[204:207], v[168:171], a[4:7], v107, v111 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 00000001A864: D3AC0800 1802DF6B D3AD8C04 841351CC
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[204:207], v[172:175], a[0:3], v107, v111 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 00000001A874: D3AC1800 1802DF6B D3AD8C00 840359CC
	buffer_load_dwordx4 v98, s[16:19], s54 offen lds           // 00000001A884: E05D1000 36040062
	s_mov_b32 m0, s55                                          // 00000001A88C: BEFC0037
	s_add_u32 s57, s35, s66                                    // 00000001A890: 80394223
	buffer_load_dwordx4 v99, s[16:19], s54 offen lds           // 00000001A894: E05D1000 36040063
	s_mov_b32 m0, s56                                          // 00000001A89C: BEFC0038
	s_sub_u32 s57, s57, s30                                    // 00000001A8A0: 80B91E39
	buffer_load_dwordx4 v126, s[16:19], s54 offen lds          // 00000001A8A4: E05D1000 3604007E
	s_mov_b32 m0, s58                                          // 00000001A8AC: BEFC003A
	s_add_i32 s59, s52, 0x1000                                 // 00000001A8B0: 813BFF34 00001000
	buffer_load_dwordx4 v100, s[16:19], s54 offen lds          // 00000001A8B8: E05D1000 36040064
	s_mov_b32 m0, s52                                          // 00000001A8C0: BEFC0034
	s_add_i32 s61, s52, 0x2000                                 // 00000001A8C4: 813DFF34 00002000
	buffer_load_dwordx4 v98, s[16:19], s57 offen lds           // 00000001A8CC: E05D1000 39040062
	s_mov_b32 m0, s59                                          // 00000001A8D4: BEFC003B
	s_add_i32 s62, s52, 0x3000                                 // 00000001A8D8: 813EFF34 00003000
	buffer_load_dwordx4 v99, s[16:19], s57 offen lds           // 00000001A8E0: E05D1000 39040063
	s_mov_b32 m0, s61                                          // 00000001A8E8: BEFC003D
	s_add_u32 s60, s38, s66                                    // 00000001A8EC: 803C4226
	buffer_load_dwordx4 v126, s[16:19], s57 offen lds          // 00000001A8F0: E05D1000 3904007E
	s_mov_b32 m0, s62                                          // 00000001A8F8: BEFC003E
	s_sub_u32 s60, s60, s34                                    // 00000001A8FC: 80BC223C
	s_add_i32 s63, s51, 0x1000                                 // 00000001A900: 813FFF33 00001000
	buffer_load_dwordx4 v100, s[16:19], s57 offen lds          // 00000001A908: E05D1000 39040064
	s_mov_b32 m0, s51                                          // 00000001A910: BEFC0033
	s_add_i32 s64, s51, 0x2000                                 // 00000001A914: 8140FF33 00002000
	buffer_load_dwordx4 v102, s[20:23], s60 offen lds          // 00000001A91C: E05D1000 3C050066
	s_mov_b32 m0, s63                                          // 00000001A924: BEFC003F
	s_add_i32 s65, s51, 0x3000                                 // 00000001A928: 8141FF33 00003000
	buffer_load_dwordx4 v101, s[20:23], s60 offen lds          // 00000001A930: E05D1000 3C050065
	s_mov_b32 m0, s64                                          // 00000001A938: BEFC0040
	s_add_u32 s66, s36, s66                                    // 00000001A93C: 80424224
	buffer_load_dwordx4 v103, s[20:23], s60 offen lds          // 00000001A940: E05D1000 3C050067
	s_mov_b32 m0, s65                                          // 00000001A948: BEFC0041
	s_sub_u32 s66, s66, s34                                    // 00000001A94C: 80C22242
	s_add_i32 s67, s50, 0x1000                                 // 00000001A950: 8143FF32 00001000
	buffer_load_dwordx4 v104, s[20:23], s60 offen lds          // 00000001A958: E05D1000 3C050068
	s_mov_b32 m0, s50                                          // 00000001A960: BEFC0032
	s_add_i32 s68, s50, 0x2000                                 // 00000001A964: 8144FF32 00002000
	buffer_load_dwordx4 v102, s[20:23], s66 offen lds          // 00000001A96C: E05D1000 42050066
	s_mov_b32 m0, s67                                          // 00000001A974: BEFC0043
	s_add_i32 s69, s50, 0x3000                                 // 00000001A978: 8145FF32 00003000
	buffer_load_dwordx4 v101, s[20:23], s66 offen lds          // 00000001A980: E05D1000 42050065
	s_mov_b32 m0, s68                                          // 00000001A988: BEFC0044
	s_lshl_b32 s70, s49, 9                                     // 00000001A98C: 8E468931
	buffer_load_dwordx4 v103, s[20:23], s66 offen lds          // 00000001A990: E05D1000 42050067
	s_mov_b32 m0, s69                                          // 00000001A998: BEFC0045
	s_add_i32 s44, s44, 37                                     // 00000001A99C: 812CA52C
	buffer_load_dwordx4 v104, s[20:23], s66 offen lds          // 00000001A9A0: E05D1000 42050068
	s_cmpk_eq_i32 s44, 0x70                                    // 00000001A9A8: B12C0070
	buffer_load_dwordx2 v[138:139], v105, s[0:3], s70 offen    // 00000001A9AC: E0541000 46008A69
	buffer_load_dwordx2 v[132:133], v105, s[4:7], s70 offen    // 00000001A9B4: E0541000 46018469
	buffer_load_dwordx2 v[136:137], v105, s[8:11], s70 offen   // 00000001A9BC: E0541000 46028869
	buffer_load_dwordx2 v[134:135], v105, s[12:15], s70 offen  // 00000001A9C4: E0541000 46038669
	s_waitcnt lgkmcnt(0)                                       // 00000001A9CC: BF8CC07F
	s_cbranch_scc0 40930                                       // 00000001A9D0: BF849FE2 <_Z22mxfp4_gluon_cpp_kernel13gluon_globals+0x105c>
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[66:69], v[34:37], a[252:255], v138, v136 op_sel_hi:[0,0,0] cbsz:4 blgp:4// 00000001A9D4: D3AC0000 0003118A D3AD8CFC 87F24542
	ds_read_b128 v[98:101], v131                               // 00000001A9E4: D9FE0000 62000083
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[66:69], v[38:41], a[248:251], v138, v136 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 00000001A9EC: D3AC1000 0003118A D3AD8CF8 87E24D42
	ds_read_b128 v[102:105], v131 offset:2048                  // 00000001A9FC: D9FE0800 66000083
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[66:69], v[42:45], a[244:247], v138, v137 op_sel_hi:[0,0,0] cbsz:4 blgp:4// 00000001AA04: D3AC0000 0003138A D3AD8CF4 87D25542
	ds_read_b128 v[106:109], v131 offset:4096                  // 00000001AA14: D9FE1000 6A000083
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[66:69], v[46:49], a[240:243], v138, v137 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 00000001AA1C: D3AC1000 0003138A D3AD8CF0 87C25D42
	ds_read_b128 v[110:113], v131 offset:6144                  // 00000001AA2C: D9FE1800 6E000083
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[82:85], v[50:53], a[252:255], v138, v136 op_sel_hi:[1,1,0] cbsz:4 blgp:4// 00000001AA34: D3AC0000 1803118A D3AD8CFC 87F26552
	ds_read_b128 v[114:117], v140                              // 00000001AA44: D9FE0000 7200008C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[82:85], v[54:57], a[248:251], v138, v136 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 00000001AA4C: D3AC1000 1803118A D3AD8CF8 87E26D52
	ds_read_b128 v[118:121], v140 offset:2048                  // 00000001AA5C: D9FE0800 7600008C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[82:85], v[58:61], a[244:247], v138, v137 op_sel_hi:[1,1,0] cbsz:4 blgp:4// 00000001AA64: D3AC0000 1803138A D3AD8CF4 87D27552
	ds_read_b128 v[122:125], v140 offset:4096                  // 00000001AA74: D9FE1000 7A00008C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[82:85], v[62:65], a[240:243], v138, v137 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 00000001AA7C: D3AC1000 1803138A D3AD8CF0 87C27D52
	ds_read_b128 v[126:129], v140 offset:6144                  // 00000001AA8C: D9FE1800 7E00008C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[70:73], v[34:37], a[236:239], v138, v136 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 00000001AA94: D3AC0800 0003118A D3AD8CEC 87B24546
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[70:73], v[38:41], a[232:235], v138, v136 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 00000001AAA4: D3AC1800 0003118A D3AD8CE8 87A24D46

=== KERNEL ISA EXCERPT for back-edge target (PC 0x295C onwards) — top of K-loop body ===
	v_accvgpr_write_b32 a193, 0                                // 000000002948: D3D940C1 18000080
	v_accvgpr_write_b32 a192, 0                                // 000000002950: D3D940C0 18000080
	s_waitcnt lgkmcnt(0)                                       // 000000002958: BF8CC07F
	s_and_b32 s50, s49, 1                                      // 00000000295C: 86328131
	s_bitcmp1_b32 s49, 0                                       // 000000002960: BF0D8031
	s_cselect_b32 s68, s33, s29                                // 000000002964: 85441D21
	s_cselect_b32 s67, s39, s37                                // 000000002968: 85432527
	s_cselect_b32 s66, s41, s40                                // 00000000296C: 85422829
	s_cselect_b32 s65, s43, s42                                // 000000002970: 85412A2B
	s_cmp_eq_u32 s50, 0                                        // 000000002974: BF068032
	s_cselect_b64 vcc, -1, 0                                   // 000000002978: 85EA80C1
	s_lshl_b32 s49, s49, 7                                     // 00000000297C: 8E318731
	s_add_u32 s50, s45, s49                                    // 000000002980: 8032312D
	v_cndmask_b32_e32 v2, v131, v148, vcc                      // 000000002984: 00052983
	v_cndmask_b32_e32 v3, v140, v149, vcc                      // 000000002988: 00072B8C
	v_cndmask_b32_e32 v4, v141, v150, vcc                      // 00000000298C: 00092D8D
	v_cndmask_b32_e32 v5, v142, v151, vcc                      // 000000002990: 000B2F8E
	s_sub_u32 s50, s50, s30                                    // 000000002994: 80B21E32
	s_add_i32 s71, s68, 0x1000                                 // 000000002998: 8147FF44 00001000
	s_mov_b32 m0, s68                                          // 0000000029A0: BEFC0044
	v_cndmask_b32_e32 v204, v127, v129, vcc                    // 0000000029A4: 0199037F
	v_cndmask_b32_e32 v205, v128, v143, vcc                    // 0000000029A8: 019B1F80
	v_cndmask_b32_e32 v206, v144, v146, vcc                    // 0000000029AC: 019D2590
	v_cndmask_b32_e32 v207, v145, v147, vcc                    // 0000000029B0: 019F2791
	s_add_i32 s70, s68, 0x2000                                 // 0000000029B4: 8146FF44 00002000
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[66:69], v[34:37], a[252:255], v138, v136 op_sel_hi:[0,0,0] cbsz:4 blgp:4// 0000000029BC: D3AC0000 0003118A D3AD8CFC 87F24542
	ds_read_b128 v[114:117], v2                                // 0000000029CC: D9FE0000 72000002
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[66:69], v[38:41], a[248:251], v138, v136 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 0000000029D4: D3AC1000 0003118A D3AD8CF8 87E24D42
	ds_read_b128 v[118:121], v2 offset:2048                    // 0000000029E4: D9FE0800 76000002
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[66:69], v[42:45], a[244:247], v138, v137 op_sel_hi:[0,0,0] cbsz:4 blgp:4// 0000000029EC: D3AC0000 0003138A D3AD8CF4 87D25542
	ds_read_b128 v[122:125], v2 offset:4096                    // 0000000029FC: D9FE1000 7A000002
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[66:69], v[46:49], a[240:243], v138, v137 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000002A04: D3AC1000 0003138A D3AD8CF0 87C25D42
	ds_read_b128 v[152:155], v2 offset:6144                    // 000000002A14: D9FE1800 98000002

=== STEADY-STATE BARRIER PATTERN (first 4 of 38) — note s_waitcnt vmcnt(8) before s_barrier ===
  --- s_barrier at line 327:
	buffer_load_dwordx4 v104, s[20:23], s46 offen lds          // 000000001F94: E05D1000 2E050068
	s_waitcnt vmcnt(0)                                         // 000000001F9C: BF8C0F70
	s_barrier                                                  // 000000001FA0: BF8A0000
	ds_read_b128 v[66:69], v10                                 // 000000001FA4: D9FE0000 4200000A
  --- s_barrier at line 756:
	s_waitcnt lgkmcnt(0)                                       // 000000002E40: BF8CC07F
	s_waitcnt vmcnt(8)                                         // 000000002E44: BF8C0F78
	s_barrier                                                  // 000000002E48: BF8A0000
	s_add_i32 s69, s68, 0x3000                                 // 000000002E4C: 8145FF44 00003000
  --- s_barrier at line 997:
	s_waitcnt lgkmcnt(0)                                       // 000000003904: BF8CC07F
	s_waitcnt vmcnt(8)                                         // 000000003908: BF8C0F78
	s_barrier                                                  // 00000000390C: BF8A0000
	s_add_i32 s57, s52, 0x3000                                 // 000000003910: 8139FF34 00003000
  --- s_barrier at line 1227:
	s_waitcnt lgkmcnt(0)                                       // 000000004394: BF8CC07F
	s_waitcnt vmcnt(8)                                         // 000000004398: BF8C0F78
	s_barrier                                                  // 00000000439C: BF8A0000
	s_add_u32 s84, s35, s81                                    // 0000000043A0: 80545123
