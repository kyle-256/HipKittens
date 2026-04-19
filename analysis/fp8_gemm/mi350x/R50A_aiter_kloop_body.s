;; ============================================================================
;; R50A: Aiter K-loop steady-state body excerpt (lines 660-860 of disasm)
;;
;; Source: f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co (aiter MI350X 256x256)
;;
;; OBSERVED PATTERN (steady-state, per K-pair):
;;   - 64 MFMAs total per K-pair (matches HK kpair_64mfma_step34)
;;   - ~16 ds_read_b128 + ~4 ds_read_b32 spread *evenly* across the MFMA chain
;;   - Pattern: 2-3 MFMAs followed by 1 ds_read, repeated ~16 times
;;   - Specifically (count per ~5-instr group): 2 MFMA + 1 ds_read + 2 MFMA + 1 ds_read
;;   - Aiter also interleaves buffer_load_dwordx4 (HBM prefetch for B) at same cadence
;;
;; HK current pattern (R44 baseline, kpair_64mfma_step34, FUSED_STEP34=0 path
;; uses kpair_32mfma_with_lds_and_pf and similar; FUSED_STEP34=1 uses
;; kpair_64mfma_step34 directly):
;;   - Step3 row 0: 8 MFMAs interleaved 1:1 with 8 ds_reads (good)
;;   - Step3 rows 1-3: 24 pure MFMAs (no ds_reads — BATCHED gap)
;;   - Step4 same shape
;;   => Total: 64 MFMAs, 16 ds_reads, but ds_reads concentrated in 2 of 8 rows
;;
;; AITER's mechanism: spreading the ds_reads keeps the MFMA pipeline at 1 ds_read
;; per 4 MFMAs (4-cycle latency hiding), avoiding the long pure-MFMA stretch
;; where ds_reads have to pile up at the next batch.
;;
;; R50A REWRITE PLAN:
;;   - Move 2 ds_reads from Row0 P0 (front-loaded) into Row1 (was pure MFMA)
;;   - Move 2 ds_reads from Row0 P1 into Row2
;;   - Net: each of the 4 rows in Step3 gets 2 ds_reads (vs current 4+4+0+0)
;;   - Total ds_read & MFMA count UNCHANGED
;;   - Only ISSUE ORDER changes
;; ============================================================================


0000000000003c68 <label_041A>:
	s_waitcnt vmcnt(10) lgkmcnt(0)                             // 000000003C68: BF8C007A
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[136:139], v[8:11], a[0:3], v208, v200 op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000003C6C: D3AC6000 000391D0 D3AD8C00 84021188
	s_barrier                                                  // 000000003C7C: BF8A0000
	s_nop 0                                                    // 000000003C80: BF800000
	s_nop 0                                                    // 000000003C84: BF800000
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[136:139], v[12:15], a[4:7], v208, v200 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000003C88: D3AC7000 000391D0 D3AD8C04 84121988
	buffer_load_dwordx4 v[168:171], v225, s[16:19], 0 offen    // 000000003C98: E05C1000 8004A8E1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[140:143], v[8:11], a[32:35], v208, v200 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000003CA0: D3AC6800 000391D0 D3AD8C20 8482118C
	ds_read_b128 v[72:75], v220 offset:16896                   // 000000003CB0: D9FE4200 480000DC
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[140:143], v[12:15], a[36:39], v208, v200 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000003CB8: D3AC7800 000391D0 D3AD8C24 8492198C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[136:139], v[16:19], a[8:11], v208, v201 op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000003CC8: D3AC6000 000393D0 D3AD8C08 84222188
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[136:139], v[20:23], a[12:15], v208, v201 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000003CD8: D3AC7000 000393D0 D3AD8C0C 84322988
	buffer_load_dwordx4 v[172:175], v226, s[16:19], 0 offen    // 000000003CE8: E05C1000 8004ACE2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[140:143], v[16:19], a[40:43], v208, v201 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000003CF0: D3AC6800 000393D0 D3AD8C28 84A2218C
	ds_read_b128 v[104:107], v220 offset:16960                 // 000000003D00: D9FE4240 680000DC
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[140:143], v[20:23], a[44:47], v208, v201 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000003D08: D3AC7800 000393D0 D3AD8C2C 84B2298C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[136:139], v[24:27], a[16:19], v208, v202 op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000003D18: D3AC6000 000395D0 D3AD8C10 84423188
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[136:139], v[28:31], a[20:23], v208, v202 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000003D28: D3AC7000 000395D0 D3AD8C14 84523988
	buffer_load_dwordx4 v[176:179], v227, s[16:19], 0 offen    // 000000003D38: E05C1000 8004B0E3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[140:143], v[24:27], a[48:51], v208, v202 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000003D40: D3AC6800 000395D0 D3AD8C30 84C2318C
	ds_read_b128 v[76:79], v220 offset:17408                   // 000000003D50: D9FE4400 4C0000DC
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[140:143], v[28:31], a[52:55], v208, v202 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000003D58: D3AC7800 000395D0 D3AD8C34 84D2398C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[136:139], v[32:35], a[24:27], v208, v203 op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000003D68: D3AC6000 000397D0 D3AD8C18 84624188
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[136:139], v[36:39], a[28:31], v208, v203 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000003D78: D3AC7000 000397D0 D3AD8C1C 84724988
	buffer_load_dwordx4 v[180:183], v228, s[16:19], 0 offen    // 000000003D88: E05C1000 8004B4E4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[140:143], v[32:35], a[56:59], v208, v203 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000003D90: D3AC6800 000397D0 D3AD8C38 84E2418C
	ds_read_b128 v[108:111], v220 offset:17472                 // 000000003DA0: D9FE4440 6C0000DC
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[140:143], v[36:39], a[60:63], v208, v203 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000003DA8: D3AC7800 000397D0 D3AD8C3C 84F2498C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[144:147], v[8:11], a[64:67], v209, v200 op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000003DB8: D3AC6000 000391D1 D3AD8C40 85021190
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[144:147], v[12:15], a[68:71], v209, v200 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000003DC8: D3AC7000 000391D1 D3AD8C44 85121990
	buffer_load_dwordx4 v[184:187], v229, s[16:19], 0 offen    // 000000003DD8: E05C1000 8004B8E5
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[148:151], v[8:11], a[96:99], v209, v200 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000003DE0: D3AC6800 000391D1 D3AD8C60 85821194
	ds_read_b128 v[80:83], v220 offset:21120                   // 000000003DF0: D9FE5280 500000DC
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[148:151], v[12:15], a[100:103], v209, v200 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000003DF8: D3AC7800 000391D1 D3AD8C64 85921994
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[144:147], v[16:19], a[72:75], v209, v201 op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000003E08: D3AC6000 000393D1 D3AD8C48 85222190
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[144:147], v[20:23], a[76:79], v209, v201 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000003E18: D3AC7000 000393D1 D3AD8C4C 85322990
	buffer_load_dwordx4 v[188:191], v230, s[16:19], 0 offen    // 000000003E28: E05C1000 8004BCE6
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[148:151], v[16:19], a[104:107], v209, v201 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000003E30: D3AC6800 000393D1 D3AD8C68 85A22194
	ds_read_b128 v[112:115], v220 offset:21184                 // 000000003E40: D9FE52C0 700000DC
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[148:151], v[20:23], a[108:111], v209, v201 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000003E48: D3AC7800 000393D1 D3AD8C6C 85B22994
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[144:147], v[24:27], a[80:83], v209, v202 op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000003E58: D3AC6000 000395D1 D3AD8C50 85423190
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[144:147], v[28:31], a[84:87], v209, v202 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000003E68: D3AC7000 000395D1 D3AD8C54 85523990
	buffer_load_dwordx4 v[192:195], v231, s[16:19], 0 offen    // 000000003E78: E05C1000 8004C0E7
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[148:151], v[24:27], a[112:115], v209, v202 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000003E80: D3AC6800 000395D1 D3AD8C70 85C23194
	ds_read_b128 v[84:87], v220 offset:21632                   // 000000003E90: D9FE5480 540000DC
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[148:151], v[28:31], a[116:119], v209, v202 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000003E98: D3AC7800 000395D1 D3AD8C74 85D23994
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[144:147], v[32:35], a[88:91], v209, v203 op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000003EA8: D3AC6000 000397D1 D3AD8C58 85624190
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[144:147], v[36:39], a[92:95], v209, v203 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000003EB8: D3AC7000 000397D1 D3AD8C5C 85724990
	buffer_load_dwordx4 v[196:199], v232, s[16:19], 0 offen    // 000000003EC8: E05C1000 8004C4E8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[148:151], v[32:35], a[120:123], v209, v203 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000003ED0: D3AC6800 000397D1 D3AD8C78 85E24194
	ds_read_b128 v[116:119], v220 offset:21696                 // 000000003EE0: D9FE54C0 740000DC
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[148:151], v[36:39], a[124:127], v209, v203 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000003EE8: D3AC7800 000397D1 D3AD8C7C 85F24994
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[152:155], v[40:43], a[0:3], v208, v200 op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000003EF8: D3AC6000 180391D0 D3AD8C00 84025198
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[152:155], v[44:47], a[4:7], v208, v200 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000003F08: D3AC7000 180391D0 D3AD8C04 84125998
	buffer_load_dword v210, v233, s[24:27], 0 offen            // 000000003F18: E0501000 8006D2E9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[156:159], v[40:43], a[32:35], v208, v200 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000003F20: D3AC6800 180391D0 D3AD8C20 8482519C
	ds_read_b128 v[88:91], v220 offset:25344                   // 000000003F30: D9FE6300 580000DC
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[156:159], v[44:47], a[36:39], v208, v200 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000003F38: D3AC7800 180391D0 D3AD8C24 8492599C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[152:155], v[48:51], a[8:11], v208, v201 op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000003F48: D3AC6000 180393D0 D3AD8C08 84226198
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[152:155], v[52:55], a[12:15], v208, v201 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000003F58: D3AC7000 180393D0 D3AD8C0C 84326998
	buffer_load_dword v211, v234, s[24:27], 0 offen            // 000000003F68: E0501000 8006D3EA
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[156:159], v[48:51], a[40:43], v208, v201 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000003F70: D3AC6800 180393D0 D3AD8C28 84A2619C
	ds_read_b128 v[120:123], v220 offset:25408                 // 000000003F80: D9FE6340 780000DC
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[156:159], v[52:55], a[44:47], v208, v201 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000003F88: D3AC7800 180393D0 D3AD8C2C 84B2699C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[152:155], v[56:59], a[16:19], v208, v202 op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000003F98: D3AC6000 180395D0 D3AD8C10 84427198
	s_add_u32 s53, 0x200, s50                                  // 000000003FA8: 803532FF 00000200
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[152:155], v[60:63], a[20:23], v208, v202 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000003FB0: D3AC7000 180395D0 D3AD8C14 84527998
	ds_read_b128 v[92:95], v220 offset:25856                   // 000000003FC0: D9FE6500 5C0000DC
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[156:159], v[56:59], a[48:51], v208, v202 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000003FC8: D3AC6800 180395D0 D3AD8C30 84C2719C
	s_cmp_lt_u32 s53, s51                                      // 000000003FD8: BF0A3335
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[156:159], v[60:63], a[52:55], v208, v202 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000003FDC: D3AC7800 180395D0 D3AD8C34 84D2799C
	ds_read_b128 v[124:127], v220 offset:25920                 // 000000003FEC: D9FE6540 7C0000DC
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[152:155], v[64:67], a[24:27], v208, v203 op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000003FF4: D3AC6000 180397D0 D3AD8C18 84628198
	s_cselect_b32 s62, s62, 0                                  // 000000004004: 853E803E
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[152:155], v[68:71], a[28:31], v208, v203 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004008: D3AC7000 180397D0 D3AD8C1C 84728998
	ds_read_b128 v[96:99], v220 offset:29568                   // 000000004018: D9FE7380 600000DC
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[156:159], v[64:67], a[56:59], v208, v203 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004020: D3AC6800 180397D0 D3AD8C38 84E2819C
	s_cselect_b32 s64, s64, 0                                  // 000000004030: 85408040
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[156:159], v[68:71], a[60:63], v208, v203 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004034: D3AC7800 180397D0 D3AD8C3C 84F2899C
	ds_read_b128 v[128:131], v220 offset:29632                 // 000000004044: D9FE73C0 800000DC
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[160:163], v[40:43], a[64:67], v209, v200 op_sel_hi:[1,1,0] cbsz:4 blgp:4// 00000000404C: D3AC6000 180391D1 D3AD8C40 850251A0
	s_add_u32 s16, s62, s16                                    // 00000000405C: 8010103E
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[160:163], v[44:47], a[68:71], v209, v200 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004060: D3AC7000 180391D1 D3AD8C44 851259A0
	ds_read_b128 v[100:103], v220 offset:30080                 // 000000004070: D9FE7580 640000DC
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[164:167], v[40:43], a[96:99], v209, v200 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004078: D3AC6800 180391D1 D3AD8C60 858251A4
	s_addc_u32 s17, 0, s17                                     // 000000004088: 82111180
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[164:167], v[44:47], a[100:103], v209, v200 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 00000000408C: D3AC7800 180391D1 D3AD8C64 859259A4
	ds_read_b128 v[132:135], v220 offset:30144                 // 00000000409C: D9FE75C0 840000DC
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[160:163], v[48:51], a[72:75], v209, v201 op_sel_hi:[1,1,0] cbsz:4 blgp:4// 0000000040A4: D3AC6000 180393D1 D3AD8C48 852261A0
	s_sub_u32 s18, s18, s62                                    // 0000000040B4: 80923E12
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[160:163], v[52:55], a[76:79], v209, v201 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 0000000040B8: D3AC7000 180393D1 D3AD8C4C 853269A0
	ds_read_b32 v204, v224 offset:1024                         // 0000000040C8: D86C0400 CC0000E0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[164:167], v[48:51], a[104:107], v209, v201 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 0000000040D0: D3AC6800 180393D1 D3AD8C68 85A261A4
	s_add_u32 s24, s64, s24                                    // 0000000040E0: 80181840
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[164:167], v[52:55], a[108:111], v209, v201 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 0000000040E4: D3AC7800 180393D1 D3AD8C6C 85B269A4
	ds_read_b32 v205, v224 offset:1280                         // 0000000040F4: D86C0500 CD0000E0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[160:163], v[56:59], a[80:83], v209, v202 op_sel_hi:[1,1,0] cbsz:4 blgp:4// 0000000040FC: D3AC6000 180395D1 D3AD8C50 854271A0
	s_addc_u32 s25, 0, s25                                     // 00000000410C: 82191980
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[160:163], v[60:63], a[84:87], v209, v202 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004110: D3AC7000 180395D1 D3AD8C54 855279A0
	ds_read_b32 v206, v224 offset:1536                         // 000000004120: D86C0600 CE0000E0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[164:167], v[56:59], a[112:115], v209, v202 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004128: D3AC4800 180395D1 D3AD8C70 85C271A4
	s_sub_u32 s26, s26, s64                                    // 000000004138: 809A401A
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[164:167], v[60:63], a[116:119], v209, v202 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 00000000413C: D3AC7800 180395D1 D3AD8C74 85D279A4
	ds_read_b32 v207, v224 offset:1792                         // 00000000414C: D86C0700 CF0000E0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[160:163], v[64:67], a[88:91], v209, v203 op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004154: D3AC6000 180397D1 D3AD8C58 856281A0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[160:163], v[68:71], a[92:95], v209, v203 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004164: D3AC7000 180397D1 D3AD8C5C 857289A0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[164:167], v[64:67], a[120:123], v209, v203 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004174: D3AC6800 180397D1 D3AD8C78 85E281A4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[164:167], v[68:71], a[124:127], v209, v203 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004184: D3AC7800 180397D1 D3AD8C7C 85F289A4
	s_waitcnt vmcnt(15) lgkmcnt(0)                             // 000000004194: BF8C007F
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[136:139], v[72:75], a[128:131], v208, v204 op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004198: D3AC6000 000399D0 D3AD8C80 86029188
	s_barrier                                                  // 0000000041A8: BF8A0000
	s_nop 0                                                    // 0000000041AC: BF800000
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[136:139], v[76:79], a[132:135], v208, v204 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 0000000041B0: D3AC7000 000399D0 D3AD8C84 86129988
	s_add_u32 m0, 0, s59                                       // 0000000041C0: 807C3B80
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[140:143], v[72:75], a[160:163], v208, v204 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 0000000041C4: D3AC6800 000399D0 D3AD8CA0 8682918C
	ds_read_b128 v[8:11], v221                                 // 0000000041D4: D9FE0000 080000DD
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[140:143], v[76:79], a[164:167], v208, v204 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 0000000041DC: D3AC7800 000399D0 D3AD8CA4 8692998C
	buffer_load_dwordx4 v212, s[12:15], 0 offen lds            // 0000000041EC: E05D1000 800300D4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[136:139], v[80:83], a[136:139], v208, v205 op_sel_hi:[0,0,0] cbsz:4 blgp:4// 0000000041F4: D3AC6000 00039BD0 D3AD8C88 8622A188
	ds_read_b128 v[40:43], v221 offset:64                      // 000000004204: D9FE0040 280000DD
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[136:139], v[84:87], a[140:143], v208, v205 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 00000000420C: D3AC7000 00039BD0 D3AD8C8C 8632A988
	s_add_u32 m0, 0x1080, s59                                  // 00000000421C: 807C3BFF 00001080
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[140:143], v[80:83], a[168:171], v208, v205 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004224: D3AC6800 00039BD0 D3AD8CA8 86A2A18C
	ds_read_b128 v[12:15], v221 offset:512                     // 000000004234: D9FE0200 0C0000DD
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[140:143], v[84:87], a[172:175], v208, v205 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 00000000423C: D3AC7800 00039BD0 D3AD8CAC 86B2A98C
	buffer_load_dwordx4 v213, s[12:15], 0 offen lds            // 00000000424C: E05D1000 800300D5
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[136:139], v[88:91], a[144:147], v208, v206 op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004254: D3AC4000 00039DD0 D3AD8C90 8642B188
	ds_read_b128 v[44:47], v221 offset:576                     // 000000004264: D9FE0240 2C0000DD
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[136:139], v[92:95], a[148:151], v208, v206 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 00000000426C: D3AC7000 00039DD0 D3AD8C94 8652B988
	s_add_u32 m0, 0x2100, s59                                  // 00000000427C: 807C3BFF 00002100
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[140:143], v[88:91], a[176:179], v208, v206 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004284: D3AC6800 00039DD0 D3AD8CB0 86C2B18C
	ds_read_b128 v[16:19], v221 offset:4224                    // 000000004294: D9FE1080 100000DD
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[140:143], v[92:95], a[180:183], v208, v206 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 00000000429C: D3AC7800 00039DD0 D3AD8CB4 86D2B98C
	buffer_load_dwordx4 v214, s[12:15], 0 offen lds            // 0000000042AC: E05D1000 800300D6
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[136:139], v[96:99], a[152:155], v208, v207 op_sel_hi:[0,0,0] cbsz:4 blgp:4// 0000000042B4: D3AC4000 00039FD0 D3AD8C98 8662C188
	ds_read_b128 v[48:51], v221 offset:4288                    // 0000000042C4: D9FE10C0 300000DD
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[136:139], v[100:103], a[156:159], v208, v207 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 0000000042CC: D3AC7000 00039FD0 D3AD8C9C 8672C988
	s_add_u32 m0, 0x3180, s59                                  // 0000000042DC: 807C3BFF 00003180
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[140:143], v[96:99], a[184:187], v208, v207 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 0000000042E4: D3AC6800 00039FD0 D3AD8CB8 86E2C18C
	ds_read_b128 v[20:23], v221 offset:4736                    // 0000000042F4: D9FE1280 140000DD
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[140:143], v[100:103], a[188:191], v208, v207 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 0000000042FC: D3AC7800 00039FD0 D3AD8CBC 86F2C98C
	buffer_load_dwordx4 v215, s[12:15], 0 offen lds            // 00000000430C: E05D1000 800300D7
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[144:147], v[72:75], a[192:195], v209, v204 op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004314: D3AC4000 000399D1 D3AD8CC0 87029190
	ds_read_b128 v[52:55], v221 offset:4800                    // 000000004324: D9FE12C0 340000DD
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[144:147], v[76:79], a[196:199], v209, v204 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 00000000432C: D3AC7000 000399D1 D3AD8CC4 87129990
	s_add_u32 m0, 0, s60                                       // 00000000433C: 807C3C80
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[148:151], v[72:75], a[224:227], v209, v204 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004340: D3AC4800 000399D1 D3AD8CE0 87829194
	ds_read_b128 v[24:27], v221 offset:8448                    // 000000004350: D9FE2100 180000DD
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[148:151], v[76:79], a[228:231], v209, v204 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004358: D3AC7800 000399D1 D3AD8CE4 87929994
	buffer_load_dword v222, s[20:23], 0 offen lds              // 000000004368: E0511000 800500DE
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[144:147], v[80:83], a[200:203], v209, v205 op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004370: D3AC4000 00039BD1 D3AD8CC8 8722A190
	ds_read_b128 v[56:59], v221 offset:8512                    // 000000004380: D9FE2140 380000DD
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[144:147], v[84:87], a[204:207], v209, v205 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004388: D3AC7000 00039BD1 D3AD8CCC 8732A990
	s_add_u32 m0, 0x4200, s59                                  // 000000004398: 807C3BFF 00004200
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[148:151], v[80:83], a[232:235], v209, v205 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 0000000043A0: D3AC6800 00039BD1 D3AD8CE8 87A2A194
	ds_read_b128 v[28:31], v221 offset:8960                    // 0000000043B0: D9FE2300 1C0000DD
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[148:151], v[84:87], a[236:239], v209, v205 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 0000000043B8: D3AC7800 00039BD1 D3AD8CEC 87B2A994
	buffer_load_dwordx4 v216, s[12:15], 0 offen lds            // 0000000043C8: E05D1000 800300D8
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[144:147], v[88:91], a[208:211], v209, v206 op_sel_hi:[0,0,0] cbsz:4 blgp:4// 0000000043D0: D3AC4000 00039DD1 D3AD8CD0 8742B190
	ds_read_b128 v[60:63], v221 offset:9024                    // 0000000043E0: D9FE2340 3C0000DD
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[144:147], v[92:95], a[212:215], v209, v206 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 0000000043E8: D3AC7000 00039DD1 D3AD8CD4 8752B990
	s_add_u32 m0, 0x5280, s59                                  // 0000000043F8: 807C3BFF 00005280
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[148:151], v[88:91], a[240:243], v209, v206 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004400: D3AC2800 00039DD1 D3AD8CF0 87C2B194
	ds_read_b128 v[32:35], v221 offset:12672                   // 000000004410: D9FE3180 200000DD
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[148:151], v[92:95], a[244:247], v209, v206 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004418: D3AC7800 00039DD1 D3AD8CF4 87D2B994
	buffer_load_dwordx4 v217, s[12:15], 0 offen lds            // 000000004428: E05D1000 800300D9
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[144:147], v[96:99], a[216:219], v209, v207 op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004430: D3AC6000 00039FD1 D3AD8CD8 8762C190
	ds_read_b128 v[64:67], v221 offset:12736                   // 000000004440: D9FE31C0 400000DD
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[144:147], v[100:103], a[220:223], v209, v207 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004448: D3AC3000 00039FD1 D3AD8CDC 8772C990
	s_add_u32 m0, 0x6300, s59                                  // 000000004458: 807C3BFF 00006300
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[148:151], v[96:99], a[248:251], v209, v207 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004460: D3AC6800 00039FD1 D3AD8CF8 87E2C194
	ds_read_b128 v[36:39], v221 offset:13184                   // 000000004470: D9FE3380 240000DD
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[148:151], v[100:103], a[252:255], v209, v207 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4// 000000004478: D3AC7800 00039FD1 D3AD8CFC 87F2C994
	buffer_load_dwordx4 v218, s[12:15], 0 offen lds            // 000000004488: E05D1000 800300DA
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[152:155], v[104:107], a[128:131], v208, v204 op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004490: D3AC6000 180399D0 D3AD8C80 8602D198
	ds_read_b128 v[68:71], v221 offset:13248                   // 0000000044A0: D9FE33C0 440000DD
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[152:155], v[108:111], a[132:135], v208, v204 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 0000000044A8: D3AC7000 180399D0 D3AD8C84 8612D998
	s_add_u32 m0, 0x7380, s59                                  // 0000000044B8: 807C3BFF 00007380
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[156:159], v[104:107], a[160:163], v208, v204 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 0000000044C0: D3AC6800 180399D0 D3AD8CA0 8682D19C
	ds_read_b32 v200, v224 offset:2048                         // 0000000044D0: D86C0800 C80000E0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[156:159], v[108:111], a[164:167], v208, v204 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 0000000044D8: D3AC7800 180399D0 D3AD8CA4 8692D99C
	buffer_load_dwordx4 v219, s[12:15], 0 offen lds            // 0000000044E8: E05D1000 800300DB
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[152:155], v[112:115], a[136:139], v208, v205 op_sel_hi:[1,1,0] cbsz:4 blgp:4// 0000000044F0: D3AC6000 18039BD0 D3AD8C88 8622E198
	ds_read_b32 v201, v224 offset:2304                         // 000000004500: D86C0900 C90000E0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[152:155], v[116:119], a[140:143], v208, v205 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004508: D3AC7000 18039BD0 D3AD8C8C 8632E998
	s_add_u32 m0, 0x400, s60                                   // 000000004518: 807C3CFF 00000400
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[156:159], v[112:115], a[168:171], v208, v205 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004520: D3AC6800 18039BD0 D3AD8CA8 86A2E19C
	ds_read_b32 v202, v224 offset:2560                         // 000000004530: D86C0A00 CA0000E0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[156:159], v[116:119], a[172:175], v208, v205 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004538: D3AC7800 18039BD0 D3AD8CAC 86B2E99C
	buffer_load_dword v223, s[20:23], 0 offen lds              // 000000004548: E0511000 800500DF
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[152:155], v[120:123], a[144:147], v208, v206 op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004550: D3AC6000 18039DD0 D3AD8C90 8642F198
	ds_read_b32 v203, v224 offset:2816                         // 000000004560: D86C0B00 CB0000E0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[152:155], v[124:127], a[148:151], v208, v206 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004568: D3AC7000 18039DD0 D3AD8C94 8652F998
	s_add_u32 s52, 0x300, s50                                  // 000000004578: 803432FF 00000300
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[156:159], v[120:123], a[176:179], v208, v206 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004580: D3AC6800 18039DD0 D3AD8CB0 86C2F19C
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[156:159], v[124:127], a[180:183], v208, v206 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 000000004590: D3AC7800 18039DD0 D3AD8CB4 86D2F99C
	s_cmp_lt_u32 s52, s51                                      // 0000000045A0: BF0A3334
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[152:155], v[128:131], a[152:155], v208, v207 op_sel_hi:[1,1,0] cbsz:4 blgp:4// 0000000045A4: D3AC6000 18039FD0 D3AD8C98 86630198
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[152:155], v[132:135], a[156:159], v208, v207 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4// 0000000045B4: D3AC7000 18039FD0 D3AD8C9C 86730998
