	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 5
	.text
	.globl	mxfp4_gluon_asm_kernel                     ; -- Begin function mxfp4_gluon_asm_kernel
	.p2align	8
	.type	mxfp4_gluon_asm_kernel,@function
mxfp4_gluon_asm_kernel:                            ; @mxfp4_gluon_asm_kernel
.Lfunc_begin0:
	.cfi_sections .debug_frame
	.cfi_startproc
; %bb.3:
s_load_dwordx8 s[8:15], s[4:5], 0x0
s_waitcnt lgkmcnt(0)
s_branch .LBB0_0
.p2align	8
; %bb.4:
.LBB0_0:
s_load_dwordx8 s[20:27], s[4:5], 0x20
s_load_dword s63, s[4:5], 0x40
v_and_b32_e32 v36, 0x3ff, v0
s_nop 0
v_readfirstlane_b32 s0, v36
s_bfe_u32 s5, s0, 0x20006
s_waitcnt lgkmcnt(0)
s_add_i32 s0, s22, 0xff
s_ashr_i32 s1, s0, 31
s_lshr_b32 s1, s1, 24
s_add_i32 s0, s0, s1
s_ashr_i32 s0, s0, 8
s_add_i32 s1, s23, 0xff
s_ashr_i32 s2, s1, 31
s_lshr_b32 s2, s2, 24
s_add_i32 s1, s1, s2
s_ashr_i32 s1, s1, 8
s_ashr_i32 s2, s16, 31
s_lshr_b32 s2, s2, 29
s_add_i32 s2, s16, s2
s_ashr_i32 s2, s2, 3
s_lshl_b32 s3, s16, 7
s_mulk_i32 s2, 0xfc01
s_add_i32 s2, s2, s3
s_lshl_b32 s1, s1, 2
s_xor_b32 s3, s2, s1
s_ashr_i32 s3, s3, 31
s_abs_i32 s4, s2
s_abs_i32 s6, s1
v_cvt_f32_u32_e32 v1, s6
v_rcp_iflag_f32_e32 v1, v1
s_nop 0
v_mul_f32_e32 v1, 0x4f7ffffe, v1
v_cvt_u32_f32_e32 v1, v1
s_sub_i32 s7, 0, s6
v_readfirstlane_b32 s16, v1
s_mul_i32 s7, s7, s16
s_mul_hi_u32 s7, s16, s7
s_add_i32 s16, s16, s7
s_mul_hi_u32 s7, s4, s16
s_mul_i32 s16, s7, s6
s_sub_i32 s4, s4, s16
s_add_i32 s16, s7, 1
s_sub_i32 s17, s4, s6
s_cmp_ge_u32 s4, s6
s_cselect_b32 s7, s16, s7
s_cselect_b32 s4, s17, s4
s_add_i32 s16, s7, 1
s_cmp_ge_u32 s4, s6
s_cselect_b32 s4, s16, s7
s_xor_b32 s4, s4, s3
s_sub_i32 s3, s4, s3
s_lshl_b32 s4, s3, 2
s_sub_i32 s0, s0, s4
s_min_i32 s0, s0, 4
s_mul_i32 s3, s3, s1
s_sub_i32 s1, s2, s3
s_xor_b32 s2, s1, s0
s_ashr_i32 s2, s2, 31
s_abs_i32 s3, s1
s_abs_i32 s6, s0
v_cvt_f32_u32_e32 v1, s6
v_rcp_iflag_f32_e32 v1, v1
s_nop 0
v_mul_f32_e32 v1, 0x4f7ffffe, v1
v_cvt_u32_f32_e32 v1, v1
s_sub_i32 s7, 0, s6
v_readfirstlane_b32 s16, v1
s_mul_i32 s7, s7, s16
s_mul_hi_u32 s7, s16, s7
s_add_i32 s16, s16, s7
s_mul_hi_u32 s7, s3, s16
s_mul_i32 s16, s7, s6
s_sub_i32 s3, s3, s16
s_add_i32 s16, s7, 1
s_sub_i32 s17, s3, s6
s_cmp_ge_u32 s3, s6
s_cselect_b32 s7, s16, s7
s_cselect_b32 s3, s17, s3
s_add_i32 s16, s7, 1
s_cmp_ge_u32 s3, s6
s_cselect_b32 s3, s16, s7
s_xor_b32 s3, s3, s2
s_sub_i32 s6, s3, s2
s_mul_i32 s0, s6, s0
s_sub_i32 s67, s1, s0
s_add_i32 s67, s67, s4
s_lshl_b32 s65, s5, 6
v_and_or_b32 v37, v0, 63, s65
v_lshlrev_b32_e32 v1, 1, v36
v_and_b32_e32 v1, 0x70, v1
v_or_b32_e32 v1, s5, v1
v_or_b32_e32 v20, 4, v1
v_or_b32_e32 v22, 8, v1
v_or_b32_e32 v25, 12, v1
v_or_b32_e32 v10, 0x80, v1
v_or_b32_e32 v12, 0x84, v1
v_or_b32_e32 v14, 0x88, v1
v_or_b32_e32 v16, 0x8c, v1
v_lshlrev_b32_e32 v2, 4, v36
v_and_b32_e32 v24, 0x70, v2
v_mad_u64_u32 v[2:3], s[0:1], v1, s24, v[24:25]
v_mad_u64_u32 v[4:5], s[0:1], v20, s24, v[24:25]
v_mad_u64_u32 v[6:7], s[0:1], v22, s24, v[24:25]
v_mad_u64_u32 v[8:9], s[0:1], v25, s24, v[24:25]
v_mad_u64_u32 v[10:11], s[0:1], v10, s24, v[24:25]
v_mad_u64_u32 v[12:13], s[0:1], v12, s24, v[24:25]
v_mad_u64_u32 v[14:15], s[0:1], v14, s24, v[24:25]
v_mad_u64_u32 v[16:17], s[0:1], v16, s24, v[24:25]
v_add_u32_e32 v3, 0x80, v2
v_add_u32_e32 v5, 0x80, v4
v_add_u32_e32 v7, 0x80, v6
v_add_u32_e32 v9, 0x80, v8
v_add_u32_e32 v11, 0x80, v10
v_add_u32_e32 v13, 0x80, v12
v_add_u32_e32 v15, 0x80, v14
v_add_u32_e32 v17, 0x80, v16
s_lshl_b32 s33, s67, 8
s_mul_i32 s0, s33, s24
s_ashr_i32 s1, s0, 31
s_add_u32 s0, s8, s0
s_addc_u32 s1, s9, s1
v_mad_u64_u32 v[18:19], s[2:3], v1, s25, v[24:25]
v_mad_u64_u32 v[20:21], s[2:3], v20, s25, v[24:25]
v_mad_u64_u32 v[22:23], s[2:3], v22, s25, v[24:25]
v_mad_u64_u32 v[24:25], s[2:3], v25, s25, v[24:25]
s_lshl_b32 s2, s25, 7
s_nop 0
v_add_u32_e32 v19, s2, v18
v_add_u32_e32 v21, s2, v20
v_add_u32_e32 v23, s2, v22
v_add_u32_e32 v25, s2, v24
v_add_u32_e32 v38, 0x80, v18
v_add_u32_e32 v39, 0x80, v20
v_add_u32_e32 v40, 0x80, v22
v_add_u32_e32 v41, 0x80, v24
v_add_u32_e32 v42, 0x80, v19
v_add_u32_e32 v43, 0x80, v21
v_add_u32_e32 v44, 0x80, v23
v_add_u32_e32 v45, 0x80, v25
s_lshl_b32 s34, s6, 8
s_mul_i32 s2, s34, s25
s_ashr_i32 s3, s2, 31
s_add_u32 s4, s10, s2
s_addc_u32 s10, s11, s3
v_lshrrev_b32_e32 v1, 5, v37
v_and_b32_e32 v28, 31, v0
v_lshl_or_b32 v26, v28, 3, s33
s_bfe_i32 s2, s67, 0x10017
v_add_u32_e32 v26, s2, v26
v_xor_b32_e32 v26, s2, v26
s_abs_i32 s3, s22
v_cvt_f32_u32_e32 v27, s3
v_rcp_iflag_f32_e32 v27, v27
s_nop 0
v_mul_f32_e32 v27, 0x4f7ffffe, v27
v_cvt_u32_f32_e32 v27, v27
s_sub_i32 s7, 0, s3
v_mul_lo_u32 v29, s7, v27
v_mul_hi_u32 v29, v27, v29
v_add_u32_e32 v27, v27, v29
v_mul_hi_u32 v27, v26, v27
v_mul_lo_u32 v27, v27, s3
v_sub_u32_e32 v26, v26, v27
v_cmp_le_u32_e32 vcc, s3, v26
v_subrev_u32_e32 v27, s3, v26
s_nop 0
v_cndmask_b32_e32 v26, v26, v27, vcc
v_cmp_le_u32_e32 vcc, s3, v26
v_subrev_u32_e32 v27, s3, v26
s_nop 0
v_cndmask_b32_e32 v26, v26, v27, vcc
v_xor_b32_e32 v26, s2, v26
v_subrev_u32_e32 v26, s2, v26
v_mad_u64_u32 v[26:27], s[2:3], v1, s27, v[26:27]
v_lshl_add_u32 v27, s27, 3, v26
v_lshl_or_b32 v28, v28, 2, s34
s_bfe_i32 s2, s6, 0x10017
v_add_u32_e32 v28, s2, v28
v_xor_b32_e32 v28, s2, v28
s_abs_i32 s3, s23
v_cvt_f32_u32_e32 v29, s3
v_rcp_iflag_f32_e32 v29, v29
s_nop 0
v_mul_f32_e32 v29, 0x4f7ffffe, v29
v_cvt_u32_f32_e32 v29, v29
s_sub_i32 s6, 0, s3
v_mul_lo_u32 v30, s6, v29
v_mul_hi_u32 v30, v29, v30
v_add_u32_e32 v29, v29, v30
v_mul_hi_u32 v29, v28, v29
v_mul_lo_u32 v29, v29, s3
v_sub_u32_e32 v28, v28, v29
v_cmp_le_u32_e32 vcc, s3, v28
v_subrev_u32_e32 v29, s3, v28
s_nop 0
v_cndmask_b32_e32 v28, v28, v29, vcc
v_cmp_le_u32_e32 vcc, s3, v28
v_subrev_u32_e32 v29, s3, v28
s_nop 0
v_cndmask_b32_e32 v28, v28, v29, vcc
v_xor_b32_e32 v28, s2, v28
v_subrev_u32_e32 v28, s2, v28
v_mad_u64_u32 v[28:29], s[2:3], v1, s63, v[28:29]
v_add_u32_e32 v47, 0x80, v28
s_lshl_b32 s2, s63, 3
v_add_u32_e32 v48, s2, v28
v_add_u32_e32 v49, s2, v47
s_and_b32 s1, s1, 0xffff
s_mov_b32 s3, 0x27000
s_mov_b32 s2, 0x7ffffffe
s_mul_i32 s11, s5, 0x420
s_add_i32 s25, s11, 0
s_mov_b32 m0, s25
s_nop 0
buffer_load_dwordx4 v2, s[0:3], 0, offen, lds
s_add_i32 s35, s25, 0x1080
s_mov_b32 m0, s35
s_nop 0
buffer_load_dwordx4 v4, s[0:3], 0, offen, lds
s_add_i32 s36, s25, 0x2100
s_mov_b32 m0, s36
s_nop 0
buffer_load_dwordx4 v6, s[0:3], 0, offen, lds
s_add_i32 s37, s25, 0x3180
s_mov_b32 m0, s37
s_nop 0
buffer_load_dwordx4 v8, s[0:3], 0, offen, lds
s_add_i32 s38, s25, 0x4200
s_mov_b32 m0, s38
s_nop 0
buffer_load_dwordx4 v10, s[0:3], 0, offen, lds
s_add_i32 s39, s25, 0x5280
s_mov_b32 m0, s39
s_nop 0
buffer_load_dwordx4 v12, s[0:3], 0, offen, lds
s_add_i32 s40, s25, 0x6300
s_mov_b32 m0, s40
s_nop 0
buffer_load_dwordx4 v14, s[0:3], 0, offen, lds
s_add_i32 s41, s25, 0x7380
s_mov_b32 m0, s41
s_nop 0
buffer_load_dwordx4 v16, s[0:3], 0, offen, lds
s_and_b32 s5, s10, 0xffff
s_mov_b32 s6, s2
s_mov_b32 s7, s3
s_add_i32 s22, 0, 0x107e0
s_add_i32 s42, s22, s11
s_mov_b32 m0, s42
s_nop 0
buffer_load_dwordx4 v18, s[4:7], 0, offen, lds
s_add_i32 s43, s42, 0x1080
s_mov_b32 m0, s43
s_nop 0
buffer_load_dwordx4 v20, s[4:7], 0, offen, lds
s_add_i32 s44, s42, 0x2100
s_mov_b32 m0, s44
s_nop 0
buffer_load_dwordx4 v22, s[4:7], 0, offen, lds
s_add_i32 s45, s42, 0x3180
s_mov_b32 m0, s45
s_nop 0
buffer_load_dwordx4 v24, s[4:7], 0, offen, lds
s_and_b32 s29, s15, 0xffff
s_mov_b32 s28, s14
s_mov_b32 s30, s2
s_mov_b32 s31, s3
buffer_load_dwordx2 v[32:33], v26, s[28:31], 0, offen
s_and_b32 s17, s21, 0xffff
s_mov_b32 s16, s20
s_mov_b32 s18, s2
s_mov_b32 s19, s3
buffer_load_dword v34, v28, s[16:19], 0, offen
s_add_i32 s46, s25, 0x18bc0
s_mov_b32 m0, s46
s_nop 0
buffer_load_dwordx4 v19, s[4:7], 0, offen, lds
s_add_i32 s47, s25, 0x19c40
s_mov_b32 m0, s47
s_nop 0
buffer_load_dwordx4 v21, s[4:7], 0, offen, lds
s_add_i32 s48, s25, 0x1acc0
s_mov_b32 m0, s48
s_nop 0
buffer_load_dwordx4 v23, s[4:7], 0, offen, lds
s_add_i32 s49, s25, 0x1bd40
s_mov_b32 m0, s49
s_nop 0
buffer_load_dwordx4 v25, s[4:7], 0, offen, lds
buffer_load_dword v162, v28, s[16:19], 0, offen, offset:128
s_waitcnt lgkmcnt(0)
s_barrier
s_add_i32 s50, s25, 0x8400
s_mov_b32 m0, s50
s_nop 0
buffer_load_dwordx4 v3, s[0:3], 0, offen, lds
s_add_i32 s51, s25, 0x9480
s_mov_b32 m0, s51
s_nop 0
buffer_load_dwordx4 v5, s[0:3], 0, offen, lds
s_add_i32 s52, s25, 0xa500
s_mov_b32 m0, s52
s_nop 0
buffer_load_dwordx4 v7, s[0:3], 0, offen, lds
s_add_i32 s53, s25, 0xb580
s_mov_b32 m0, s53
s_nop 0
buffer_load_dwordx4 v9, s[0:3], 0, offen, lds
s_add_i32 s54, s25, 0xc600
s_mov_b32 m0, s54
s_nop 0
buffer_load_dwordx4 v11, s[0:3], 0, offen, lds
s_add_i32 s55, s25, 0xd680
s_mov_b32 m0, s55
s_nop 0
buffer_load_dwordx4 v13, s[0:3], 0, offen, lds
s_add_i32 s56, s25, 0xe700
s_mov_b32 m0, s56
s_nop 0
buffer_load_dwordx4 v15, s[0:3], 0, offen, lds
s_add_i32 s57, s25, 0xf780
s_mov_b32 m0, s57
s_nop 0
buffer_load_dwordx4 v17, s[0:3], 0, offen, lds
s_add_i32 s58, s25, 0x149e0
s_mov_b32 m0, s58
s_nop 0
buffer_load_dwordx4 v38, s[4:7], 0, offen, lds
s_add_i32 s59, s25, 0x15a60
s_mov_b32 m0, s59
s_nop 0
buffer_load_dwordx4 v39, s[4:7], 0, offen, lds
s_add_i32 s60, s25, 0x16ae0
s_mov_b32 m0, s60
s_nop 0
buffer_load_dwordx4 v40, s[4:7], 0, offen, lds
s_add_i32 s61, s25, 0x17b60
s_mov_b32 m0, s61
s_nop 0
buffer_load_dwordx4 v41, s[4:7], 0, offen, lds
buffer_load_dwordx2 v[156:157], v27, s[28:31], 0, offen
buffer_load_dword v160, v48, s[16:19], 0, offen
s_add_i32 s28, s25, 0x1cdc0
s_mov_b32 m0, s28
s_nop 0
buffer_load_dwordx4 v42, s[4:7], 0, offen, lds
s_add_i32 s29, s25, 0x1de40
s_mov_b32 m0, s29
s_nop 0
buffer_load_dwordx4 v43, s[4:7], 0, offen, lds
s_add_i32 s30, s25, 0x1eec0
s_mov_b32 m0, s30
s_nop 0
buffer_load_dwordx4 v44, s[4:7], 0, offen, lds
s_add_i32 s31, s25, 0x1ff40
s_mov_b32 m0, s31
s_nop 0
buffer_load_dwordx4 v45, s[4:7], 0, offen, lds
buffer_load_dword v161, v49, s[16:19], 0, offen
s_lshl_b32 s27, s27, 4
s_ashr_i32 s62, s27, 31
s_lshl_b32 s63, s63, 4
s_ashr_i32 s64, s63, 31
s_waitcnt vmcnt(26), lgkmcnt(0)
s_barrier
v_and_b32_e32 v52, 15, v0
v_lshlrev_b32_e32 v1, 10, v52
s_movk_i32 s0, 0xb0
v_and_or_b32 v29, v37, s0, v1
v_lshlrev_b32_e32 v30, 5, v52
v_add3_u32 v46, v29, v30, 0
ds_read_b128 v[116:119], v46
ds_read_b128 v[120:123], v46, offset:64
ds_read_b128 v[108:111], v46, offset:256
ds_read_b128 v[112:115], v46, offset:320
ds_read_b128 v[100:103], v46, offset:512
ds_read_b128 v[104:107], v46, offset:576
ds_read_b128 v[88:91], v46, offset:768
ds_read_b128 v[96:99], v46, offset:832
ds_read_b128 v[84:87], v46, offset:16896
ds_read_b128 v[92:95], v46, offset:16960
ds_read_b128 v[76:79], v46, offset:17152
ds_read_b128 v[80:83], v46, offset:17216
ds_read_b128 v[68:71], v46, offset:17408
ds_read_b128 v[72:75], v46, offset:17472
ds_read_b128 v[60:63], v46, offset:17664
ds_read_b128 v[64:67], v46, offset:17728
v_and_b32_e32 v31, 48, v0
s_and_b32 s0, s65, 64
v_or_b32_e32 v1, v1, v31
v_add_u32_e32 v1, v1, v30
v_lshl_add_u32 v53, s0, 1, v1
v_add_u32_e32 v1, s22, v53
ds_read_b128 v[124:127], v1
ds_read_b128 v[132:135], v1, offset:64
ds_read_b128 v[128:131], v1, offset:256
ds_read_b128 v[136:139], v1, offset:320
ds_read_b128 v[140:143], v1, offset:512
ds_read_b128 v[144:147], v1, offset:576
ds_read_b128 v[148:151], v1, offset:768
ds_read_b128 v[152:155], v1, offset:832
v_lshlrev_b32_e32 v54, 3, v37
s_add_i32 s1, 0, 0x20fa0
v_add_u32_e32 v1, s1, v54
s_waitcnt vmcnt(25)
ds_write_b64 v1, v[32:33]
v_lshlrev_b32_e32 v55, 2, v37
s_add_i32 s5, 0, 0x217a0
v_add_u32_e32 v1, s5, v55
s_waitcnt vmcnt(24)
ds_write_b32 v1, v34
s_waitcnt lgkmcnt(0)
s_barrier
v_lshlrev_b32_e32 v29, 3, v36
v_and_b32_e32 v1, 0x68, v29
v_and_b32_e32 v32, 2, v0
s_and_b32 s22, s65, 0x80
s_lshr_b32 s6, s22, 3
s_add_i32 s6, s6, s1
v_lshl_add_u32 v0, v32, 9, s6
v_lshl_add_u32 v0, v31, 4, v0
v_add_u32_e32 v51, v0, v1
ds_read_b64_tr_b8 v[30:31], v51
ds_read_b64_tr_b8 v[0:1], v51, offset:128
v_and_b32_e32 v33, 0x1e8, v29
v_lshlrev_b32_e32 v32, 8, v32
s_lshr_b32 s23, s0, 2
v_add3_u32 v32, s5, v33, v32
v_add_u32_e32 v50, s23, v32
ds_read_b64_tr_b8 v[32:33], v50
s_add_u32 s65, s4, 0x100
s_addc_u32 s66, s10, 0
s_mul_i32 s0, s24, s67
s_lshl_b32 s0, s0, 8
s_ashr_i32 s1, s0, 31
s_add_u32 s0, s8, s0
s_addc_u32 s1, s9, s1
s_add_u32 s24, s0, 0x100
s_addc_u32 s67, s1, 0
s_add_u32 s20, s20, s63
s_addc_u32 s21, s21, s64
s_add_u32 s14, s14, s27
s_addc_u32 s15, s15, s62
v_mov_b32_e32 v34, 0
s_mov_b32 s68, -2
v_accvgpr_write_b32 a124, v34
v_accvgpr_write_b32 a125, v34
v_accvgpr_write_b32 a126, v34
v_accvgpr_write_b32 a127, v34
v_accvgpr_write_b32 a0, v34
v_accvgpr_write_b32 a1, v34
v_accvgpr_write_b32 a2, v34
v_accvgpr_write_b32 a3, v34
v_accvgpr_write_b32 a4, v34
v_accvgpr_write_b32 a5, v34
v_accvgpr_write_b32 a6, v34
v_accvgpr_write_b32 a7, v34
v_accvgpr_write_b32 a8, v34
v_accvgpr_write_b32 a9, v34
v_accvgpr_write_b32 a10, v34
v_accvgpr_write_b32 a11, v34
v_accvgpr_write_b32 a12, v34
v_accvgpr_write_b32 a13, v34
v_accvgpr_write_b32 a14, v34
v_accvgpr_write_b32 a15, v34
v_accvgpr_write_b32 a16, v34
v_accvgpr_write_b32 a17, v34
v_accvgpr_write_b32 a18, v34
v_accvgpr_write_b32 a19, v34
v_accvgpr_write_b32 a20, v34
v_accvgpr_write_b32 a21, v34
v_accvgpr_write_b32 a22, v34
v_accvgpr_write_b32 a23, v34
v_accvgpr_write_b32 a24, v34
v_accvgpr_write_b32 a25, v34
v_accvgpr_write_b32 a26, v34
v_accvgpr_write_b32 a27, v34
v_accvgpr_write_b32 a28, v34
v_accvgpr_write_b32 a29, v34
v_accvgpr_write_b32 a30, v34
v_accvgpr_write_b32 a31, v34
v_accvgpr_write_b32 a32, v34
v_accvgpr_write_b32 a33, v34
v_accvgpr_write_b32 a34, v34
v_accvgpr_write_b32 a35, v34
v_accvgpr_write_b32 a36, v34
v_accvgpr_write_b32 a37, v34
v_accvgpr_write_b32 a38, v34
v_accvgpr_write_b32 a39, v34
v_accvgpr_write_b32 a44, v34
v_accvgpr_write_b32 a45, v34
v_accvgpr_write_b32 a46, v34
v_accvgpr_write_b32 a47, v34
v_accvgpr_write_b32 a52, v34
v_accvgpr_write_b32 a53, v34
v_accvgpr_write_b32 a54, v34
v_accvgpr_write_b32 a55, v34
v_accvgpr_write_b32 a40, v34
v_accvgpr_write_b32 a41, v34
v_accvgpr_write_b32 a42, v34
v_accvgpr_write_b32 a43, v34
v_accvgpr_write_b32 a48, v34
v_accvgpr_write_b32 a49, v34
v_accvgpr_write_b32 a50, v34
v_accvgpr_write_b32 a51, v34
v_accvgpr_write_b32 a56, v34
v_accvgpr_write_b32 a57, v34
v_accvgpr_write_b32 a58, v34
v_accvgpr_write_b32 a59, v34
v_accvgpr_write_b32 a60, v34
v_accvgpr_write_b32 a61, v34
v_accvgpr_write_b32 a62, v34
v_accvgpr_write_b32 a63, v34
v_accvgpr_write_b32 a64, v34
v_accvgpr_write_b32 a65, v34
v_accvgpr_write_b32 a66, v34
v_accvgpr_write_b32 a67, v34
v_accvgpr_write_b32 a68, v34
v_accvgpr_write_b32 a69, v34
v_accvgpr_write_b32 a70, v34
v_accvgpr_write_b32 a71, v34
v_accvgpr_write_b32 a72, v34
v_accvgpr_write_b32 a73, v34
v_accvgpr_write_b32 a74, v34
v_accvgpr_write_b32 a75, v34
v_accvgpr_write_b32 a76, v34
v_accvgpr_write_b32 a77, v34
v_accvgpr_write_b32 a78, v34
v_accvgpr_write_b32 a79, v34
v_accvgpr_write_b32 a80, v34
v_accvgpr_write_b32 a81, v34
v_accvgpr_write_b32 a82, v34
v_accvgpr_write_b32 a83, v34
v_accvgpr_write_b32 a84, v34
v_accvgpr_write_b32 a85, v34
v_accvgpr_write_b32 a86, v34
v_accvgpr_write_b32 a87, v34
v_accvgpr_write_b32 a88, v34
v_accvgpr_write_b32 a89, v34
v_accvgpr_write_b32 a90, v34
v_accvgpr_write_b32 a91, v34
v_accvgpr_write_b32 a92, v34
v_accvgpr_write_b32 a93, v34
v_accvgpr_write_b32 a94, v34
v_accvgpr_write_b32 a95, v34
v_accvgpr_write_b32 a96, v34
v_accvgpr_write_b32 a97, v34
v_accvgpr_write_b32 a98, v34
v_accvgpr_write_b32 a99, v34
v_accvgpr_write_b32 a100, v34
v_accvgpr_write_b32 a101, v34
v_accvgpr_write_b32 a102, v34
v_accvgpr_write_b32 a103, v34
v_accvgpr_write_b32 a104, v34
v_accvgpr_write_b32 a105, v34
v_accvgpr_write_b32 a106, v34
v_accvgpr_write_b32 a107, v34
v_accvgpr_write_b32 a108, v34
v_accvgpr_write_b32 a109, v34
v_accvgpr_write_b32 a110, v34
v_accvgpr_write_b32 a111, v34
v_accvgpr_write_b32 a112, v34
v_accvgpr_write_b32 a113, v34
v_accvgpr_write_b32 a114, v34
v_accvgpr_write_b32 a115, v34
v_accvgpr_write_b32 a116, v34
v_accvgpr_write_b32 a117, v34
v_accvgpr_write_b32 a118, v34
v_accvgpr_write_b32 a119, v34
v_accvgpr_write_b32 a120, v34
v_accvgpr_write_b32 a121, v34
v_accvgpr_write_b32 a122, v34
v_accvgpr_write_b32 a123, v34
v_accvgpr_write_b32 a132, v34
v_accvgpr_write_b32 a133, v34
v_accvgpr_write_b32 a134, v34
v_accvgpr_write_b32 a135, v34
v_accvgpr_write_b32 a140, v34
v_accvgpr_write_b32 a141, v34
v_accvgpr_write_b32 a142, v34
v_accvgpr_write_b32 a143, v34
v_accvgpr_write_b32 a144, v34
v_accvgpr_write_b32 a145, v34
v_accvgpr_write_b32 a146, v34
v_accvgpr_write_b32 a147, v34
v_accvgpr_write_b32 a152, v34
v_accvgpr_write_b32 a153, v34
v_accvgpr_write_b32 a154, v34
v_accvgpr_write_b32 a155, v34
v_accvgpr_write_b32 a128, v34
v_accvgpr_write_b32 a129, v34
v_accvgpr_write_b32 a130, v34
v_accvgpr_write_b32 a131, v34
v_accvgpr_write_b32 a136, v34
v_accvgpr_write_b32 a137, v34
v_accvgpr_write_b32 a138, v34
v_accvgpr_write_b32 a139, v34
v_accvgpr_write_b32 a148, v34
v_accvgpr_write_b32 a149, v34
v_accvgpr_write_b32 a150, v34
v_accvgpr_write_b32 a151, v34
v_accvgpr_write_b32 a156, v34
v_accvgpr_write_b32 a157, v34
v_accvgpr_write_b32 a158, v34
v_accvgpr_write_b32 a159, v34
v_accvgpr_write_b32 a160, v34
v_accvgpr_write_b32 a161, v34
v_accvgpr_write_b32 a162, v34
v_accvgpr_write_b32 a163, v34
v_accvgpr_write_b32 a164, v34
v_accvgpr_write_b32 a165, v34
v_accvgpr_write_b32 a166, v34
v_accvgpr_write_b32 a167, v34
v_accvgpr_write_b32 a168, v34
v_accvgpr_write_b32 a169, v34
v_accvgpr_write_b32 a170, v34
v_accvgpr_write_b32 a171, v34
v_accvgpr_write_b32 a172, v34
v_accvgpr_write_b32 a173, v34
v_accvgpr_write_b32 a174, v34
v_accvgpr_write_b32 a175, v34
v_accvgpr_write_b32 a176, v34
v_accvgpr_write_b32 a177, v34
v_accvgpr_write_b32 a178, v34
v_accvgpr_write_b32 a179, v34
v_accvgpr_write_b32 a180, v34
v_accvgpr_write_b32 a181, v34
v_accvgpr_write_b32 a182, v34
v_accvgpr_write_b32 a183, v34
v_accvgpr_write_b32 a184, v34
v_accvgpr_write_b32 a185, v34
v_accvgpr_write_b32 a186, v34
v_accvgpr_write_b32 a187, v34
v_accvgpr_write_b32 a188, v34
v_accvgpr_write_b32 a189, v34
v_accvgpr_write_b32 a190, v34
v_accvgpr_write_b32 a191, v34
v_accvgpr_write_b32 a192, v34
v_accvgpr_write_b32 a193, v34
v_accvgpr_write_b32 a194, v34
v_accvgpr_write_b32 a195, v34
v_accvgpr_write_b32 a196, v34
v_accvgpr_write_b32 a197, v34
v_accvgpr_write_b32 a198, v34
v_accvgpr_write_b32 a199, v34
v_accvgpr_write_b32 a204, v34
v_accvgpr_write_b32 a205, v34
v_accvgpr_write_b32 a206, v34
v_accvgpr_write_b32 a207, v34
v_accvgpr_write_b32 a212, v34
v_accvgpr_write_b32 a213, v34
v_accvgpr_write_b32 a214, v34
v_accvgpr_write_b32 a215, v34
v_accvgpr_write_b32 a200, v34
v_accvgpr_write_b32 a201, v34
v_accvgpr_write_b32 a202, v34
v_accvgpr_write_b32 a203, v34
v_accvgpr_write_b32 a208, v34
v_accvgpr_write_b32 a209, v34
v_accvgpr_write_b32 a210, v34
v_accvgpr_write_b32 a211, v34
v_accvgpr_write_b32 a216, v34
v_accvgpr_write_b32 a217, v34
v_accvgpr_write_b32 a218, v34
v_accvgpr_write_b32 a219, v34
v_accvgpr_write_b32 a220, v34
v_accvgpr_write_b32 a221, v34
v_accvgpr_write_b32 a222, v34
v_accvgpr_write_b32 a223, v34
v_accvgpr_write_b32 a224, v34
v_accvgpr_write_b32 a225, v34
v_accvgpr_write_b32 a226, v34
v_accvgpr_write_b32 a227, v34
v_accvgpr_write_b32 a232, v34
v_accvgpr_write_b32 a233, v34
v_accvgpr_write_b32 a234, v34
v_accvgpr_write_b32 a235, v34
v_accvgpr_write_b32 a240, v34
v_accvgpr_write_b32 a241, v34
v_accvgpr_write_b32 a242, v34
v_accvgpr_write_b32 a243, v34
v_accvgpr_write_b32 a248, v34
v_accvgpr_write_b32 a249, v34
v_accvgpr_write_b32 a250, v34
v_accvgpr_write_b32 a251, v34
v_accvgpr_write_b32 a252, v34
v_accvgpr_write_b32 a253, v34
v_accvgpr_write_b32 a254, v34
v_accvgpr_write_b32 a255, v34
v_accvgpr_write_b32 a228, v34
v_accvgpr_write_b32 a229, v34
v_accvgpr_write_b32 a230, v34
v_accvgpr_write_b32 a231, v34
v_accvgpr_write_b32 a236, v34
v_accvgpr_write_b32 a237, v34
v_accvgpr_write_b32 a238, v34
v_accvgpr_write_b32 a239, v34
v_accvgpr_write_b32 a244, v34
v_accvgpr_write_b32 a245, v34
v_accvgpr_write_b32 a246, v34
v_accvgpr_write_b32 a247, v34
v_add_u32_e32 v56, 0, v53
v_add_u32_e32 v247, 0x18bc0, v56
v_add_u32_e32 v57, 0, v55
v_add_u32_e32 v59, 0x217a0, v57
s_mov_b32 s6, s2
s_mov_b32 s7, s3
s_mov_b32 s18, s2
s_mov_b32 s19, s3
s_mov_b32 s10, s2
s_mov_b32 s11, s3
v_add_u32_e32 v245, 0x149e0, v56
v_add_u32_e32 v58, 0, v54
v_add_u32_e32 v163, 0x20fa0, v58
v_add_u32_e32 v246, 0x1cdc0, v56
v_add_u32_e32 v248, 0x107e0, v56
s_waitcnt vmcnt(20), lgkmcnt(0)
s_barrier
.LBB0_1:
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[124:127], v[116:119], a[132:135], v32, v30, op_sel_hi:[0,0,0], cbsz:4, blgp:4
ds_read_b128 v[164:167], v247
ds_read_b128 v[168:171], v247, offset:64
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[132:135], v[120:123], a[132:135], v32, v30, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4
ds_read_b128 v[172:175], v247, offset:256
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[128:131], v[116:119], a[140:143], v32, v30, op_sel_hi:[1,0,0], cbsz:4, blgp:4
ds_read_b128 v[176:179], v247, offset:320
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[136:139], v[120:123], a[140:143], v32, v30, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4
ds_read_b128 v[180:183], v247, offset:512
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[140:143], v[116:119], a[144:147], v33, v30, op_sel_hi:[0,0,0], cbsz:4, blgp:4
ds_read_b128 v[184:187], v247, offset:576
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[144:147], v[120:123], a[144:147], v33, v30, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4
ds_read_b128 v[188:191], v247, offset:768
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[148:151], v[116:119], a[152:155], v33, v30, op_sel_hi:[1,0,0], cbsz:4, blgp:4
ds_read_b128 v[192:195], v247, offset:832
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[152:155], v[120:123], a[152:155], v33, v30, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4
s_waitcnt vmcnt(19)
ds_write_b32 v59, v162
s_and_b32 s1, s67, 0xffff
s_mov_b32 s0, s24
s_mov_b32 m0, s25
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[124:127], v[108:111], a[128:131], v32, v30, op_sel_hi:[0,1,0], cbsz:4, blgp:4
buffer_load_dwordx4 v2, s[0:3], 0, offen, lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[132:135], v[112:115], a[128:131], v32, v30, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[128:131], v[108:111], a[136:139], v32, v30, op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[136:139], v[112:115], a[136:139], v32, v30, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4
s_mov_b32 m0, s35
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[140:143], v[108:111], a[148:151], v33, v30, op_sel_hi:[0,1,0], cbsz:4, blgp:4
buffer_load_dwordx4 v4, s[0:3], 0, offen, lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[144:147], v[112:115], a[148:151], v33, v30, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[148:151], v[108:111], a[156:159], v33, v30, op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[152:155], v[112:115], a[156:159], v33, v30, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4
s_mov_b32 m0, s36
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[124:127], v[100:103], a[160:163], v32, v31, op_sel_hi:[0,0,0], cbsz:4, blgp:4
buffer_load_dwordx4 v6, s[0:3], 0, offen, lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[132:135], v[104:107], a[160:163], v32, v31, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[128:131], v[100:103], a[164:167], v32, v31, op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[136:139], v[104:107], a[164:167], v32, v31, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4
s_mov_b32 m0, s37
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[140:143], v[100:103], a[168:171], v33, v31, op_sel_hi:[0,0,0], cbsz:4, blgp:4
buffer_load_dwordx4 v8, s[0:3], 0, offen, lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[144:147], v[104:107], a[168:171], v33, v31, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[148:151], v[100:103], a[172:175], v33, v31, op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[152:155], v[104:107], a[172:175], v33, v31, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4
s_mov_b32 m0, s38
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[124:127], v[88:91], a[176:179], v32, v31, op_sel_hi:[0,1,0], cbsz:4, blgp:4
buffer_load_dwordx4 v10, s[0:3], 0, offen, lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[132:135], v[96:99], a[176:179], v32, v31, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[128:131], v[88:91], a[180:183], v32, v31, op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[136:139], v[96:99], a[180:183], v32, v31, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4
s_mov_b32 m0, s39
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[140:143], v[88:91], a[184:187], v33, v31, op_sel_hi:[0,1,0], cbsz:4, blgp:4
buffer_load_dwordx4 v12, s[0:3], 0, offen, lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[144:147], v[96:99], a[184:187], v33, v31, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[148:151], v[88:91], a[188:191], v33, v31, op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[152:155], v[96:99], a[188:191], v33, v31, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4
s_mov_b32 m0, s40
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[124:127], v[84:87], a[192:195], v32, v0, op_sel_hi:[0,0,0], cbsz:4, blgp:4
buffer_load_dwordx4 v14, s[0:3], 0, offen, lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[132:135], v[92:95], a[192:195], v32, v0, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[128:131], v[84:87], a[196:199], v32, v0, op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[136:139], v[92:95], a[196:199], v32, v0, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4
s_mov_b32 m0, s41
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[140:143], v[84:87], a[204:207], v33, v0, op_sel_hi:[0,0,0], cbsz:4, blgp:4
buffer_load_dwordx4 v16, s[0:3], 0, offen, lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[144:147], v[92:95], a[204:207], v33, v0, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[148:151], v[84:87], a[212:215], v33, v0, op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[152:155], v[92:95], a[212:215], v33, v0, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4
s_and_b32 s5, s66, 0xffff
s_mov_b32 s4, s65
s_mov_b32 m0, s42
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[124:127], v[76:79], a[200:203], v32, v0, op_sel_hi:[0,1,0], cbsz:4, blgp:4
buffer_load_dwordx4 v18, s[4:7], 0, offen, lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[132:135], v[80:83], a[200:203], v32, v0, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[128:131], v[76:79], a[208:211], v32, v0, op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[136:139], v[80:83], a[208:211], v32, v0, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4
s_mov_b32 m0, s43
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[140:143], v[76:79], a[216:219], v33, v0, op_sel_hi:[0,1,0], cbsz:4, blgp:4
buffer_load_dwordx4 v20, s[4:7], 0, offen, lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[144:147], v[80:83], a[216:219], v33, v0, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[148:151], v[76:79], a[220:223], v33, v0, op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[152:155], v[80:83], a[220:223], v33, v0, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4
s_mov_b32 m0, s44
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[124:127], v[68:71], a[224:227], v32, v1, op_sel_hi:[0,0,0], cbsz:4, blgp:4
buffer_load_dwordx4 v22, s[4:7], 0, offen, lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[132:135], v[72:75], a[224:227], v32, v1, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[128:131], v[68:71], a[232:235], v32, v1, op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[136:139], v[72:75], a[232:235], v32, v1, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4
s_mov_b32 m0, s45
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[140:143], v[68:71], a[240:243], v33, v1, op_sel_hi:[0,0,0], cbsz:4, blgp:4
buffer_load_dwordx4 v24, s[4:7], 0, offen, lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[144:147], v[72:75], a[240:243], v33, v1, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[148:151], v[68:71], a[248:251], v33, v1, op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[152:155], v[72:75], a[248:251], v33, v1, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4
s_and_b32 s17, s15, 0xffff
s_mov_b32 s16, s14
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[124:127], v[60:63], a[252:255], v32, v1, op_sel_hi:[0,1,0], cbsz:4, blgp:4
buffer_load_dwordx2 v[158:159], v26, s[16:19], 0, offen
s_waitcnt lgkmcnt(0)
s_barrier
ds_read_b64_tr_b8 v[124:125], v50
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[132:135], v[64:67], a[252:255], v32, v1, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4
s_and_b32 s9, s21, 0xffff
s_mov_b32 s8, s20
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[128:131], v[60:63], a[228:231], v32, v1, op_sel_hi:[1,1,0], cbsz:4, blgp:4
buffer_load_dword v244, v28, s[8:11], 0, offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[136:139], v[64:67], a[228:231], v32, v1, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[140:143], v[60:63], a[236:239], v33, v1, op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[144:147], v[64:67], a[236:239], v33, v1, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[148:151], v[60:63], a[244:247], v33, v1, op_sel_hi:[1,1,0], cbsz:4, blgp:4
s_waitcnt vmcnt(21), lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[152:155], v[64:67], a[244:247], v33, v1, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4
s_barrier
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[164:167], v[116:119], a[124:127], v124, v30, op_sel_hi:[0,0,0], cbsz:4, blgp:4
ds_read_b128 v[126:129], v46, offset:33792
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[168:171], v[120:123], a[124:127], v124, v30, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4
ds_read_b128 v[130:133], v46, offset:33856
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[172:175], v[116:119], a[0:3], v124, v30, op_sel_hi:[1,0,0], cbsz:4, blgp:4
ds_read_b128 v[134:137], v46, offset:34048
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[176:179], v[120:123], a[0:3], v124, v30, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4
ds_read_b128 v[138:141], v46, offset:34112
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[180:183], v[116:119], a[4:7], v125, v30, op_sel_hi:[0,0,0], cbsz:4, blgp:4
ds_read_b128 v[148:151], v46, offset:34304
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[184:187], v[120:123], a[4:7], v125, v30, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4
ds_read_b128 v[152:155], v46, offset:34368
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[188:191], v[116:119], a[8:11], v125, v30, op_sel_hi:[1,0,0], cbsz:4, blgp:4
ds_read_b128 v[196:199], v46, offset:34560
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[192:195], v[120:123], a[8:11], v125, v30, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4
ds_read_b128 v[200:203], v46, offset:34624
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[164:167], v[108:111], a[12:15], v124, v30, op_sel_hi:[0,1,0], cbsz:4, blgp:4
ds_read_b128 v[204:207], v46, offset:50688
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[168:171], v[112:115], a[12:15], v124, v30, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4
ds_read_b128 v[208:211], v46, offset:50752
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[172:175], v[108:111], a[16:19], v124, v30, op_sel_hi:[1,1,0], cbsz:4, blgp:4
ds_read_b128 v[212:215], v46, offset:50944
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[176:179], v[112:115], a[16:19], v124, v30, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4
ds_read_b128 v[216:219], v46, offset:51008
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[180:183], v[108:111], a[20:23], v125, v30, op_sel_hi:[0,1,0], cbsz:4, blgp:4
ds_read_b128 v[220:223], v46, offset:51200
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[184:187], v[112:115], a[20:23], v125, v30, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4
ds_read_b128 v[224:227], v46, offset:51264
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[188:191], v[108:111], a[24:27], v125, v30, op_sel_hi:[1,1,0], cbsz:4, blgp:4
ds_read_b128 v[228:231], v46, offset:51456
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[192:195], v[112:115], a[24:27], v125, v30, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4
ds_read_b128 v[232:235], v46, offset:51520
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[164:167], v[100:103], a[28:31], v124, v31, op_sel_hi:[0,0,0], cbsz:4, blgp:4
ds_read_b128 v[108:111], v245
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[168:171], v[104:107], a[28:31], v124, v31, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4
ds_read_b128 v[112:115], v245, offset:64
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[172:175], v[100:103], a[32:35], v124, v31, op_sel_hi:[1,0,0], cbsz:4, blgp:4
ds_read_b128 v[116:119], v245, offset:256
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[176:179], v[104:107], a[32:35], v124, v31, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4
ds_read_b128 v[120:123], v245, offset:320
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[180:183], v[100:103], a[36:39], v125, v31, op_sel_hi:[0,0,0], cbsz:4, blgp:4
ds_read_b128 v[142:145], v245, offset:512
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[184:187], v[104:107], a[36:39], v125, v31, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4
ds_read_b128 v[236:239], v245, offset:576
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[188:191], v[100:103], a[44:47], v125, v31, op_sel_hi:[1,0,0], cbsz:4, blgp:4
ds_read_b128 v[100:103], v245, offset:768
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[192:195], v[104:107], a[44:47], v125, v31, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4
ds_read_b128 v[104:107], v245, offset:832
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[164:167], v[88:91], a[52:55], v124, v31, op_sel_hi:[0,1,0], cbsz:4, blgp:4
s_waitcnt vmcnt(20)
ds_write_b64 v163, v[156:157]
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[168:171], v[96:99], a[52:55], v124, v31, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4
s_waitcnt vmcnt(19)
ds_write_b32 v59, v160
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[172:175], v[88:91], a[40:43], v124, v31, op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[176:179], v[96:99], a[40:43], v124, v31, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[180:183], v[88:91], a[48:51], v125, v31, op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[184:187], v[96:99], a[48:51], v125, v31, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[188:191], v[88:91], a[56:59], v125, v31, op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[192:195], v[96:99], a[56:59], v125, v31, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[164:167], v[84:87], a[60:63], v124, v0, op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[168:171], v[92:95], a[60:63], v124, v0, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[172:175], v[84:87], a[64:67], v124, v0, op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[176:179], v[92:95], a[64:67], v124, v0, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[180:183], v[84:87], a[68:71], v125, v0, op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[184:187], v[92:95], a[68:71], v125, v0, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[188:191], v[84:87], a[72:75], v125, v0, op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[192:195], v[92:95], a[72:75], v125, v0, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[164:167], v[76:79], a[76:79], v124, v0, op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[168:171], v[80:83], a[76:79], v124, v0, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4
s_mov_b32 m0, s46
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[172:175], v[76:79], a[80:83], v124, v0, op_sel_hi:[1,1,0], cbsz:4, blgp:4
buffer_load_dwordx4 v19, s[4:7], 0, offen, lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[176:179], v[80:83], a[80:83], v124, v0, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[180:183], v[76:79], a[84:87], v125, v0, op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[184:187], v[80:83], a[84:87], v125, v0, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4
s_mov_b32 m0, s47
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[188:191], v[76:79], a[88:91], v125, v0, op_sel_hi:[1,1,0], cbsz:4, blgp:4
buffer_load_dwordx4 v21, s[4:7], 0, offen, lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[192:195], v[80:83], a[88:91], v125, v0, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[164:167], v[68:71], a[92:95], v124, v1, op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[168:171], v[72:75], a[92:95], v124, v1, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4
s_mov_b32 m0, s48
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[172:175], v[68:71], a[96:99], v124, v1, op_sel_hi:[1,0,0], cbsz:4, blgp:4
buffer_load_dwordx4 v23, s[4:7], 0, offen, lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[176:179], v[72:75], a[96:99], v124, v1, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[180:183], v[68:71], a[100:103], v125, v1, op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[184:187], v[72:75], a[100:103], v125, v1, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4
s_mov_b32 m0, s49
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[188:191], v[68:71], a[104:107], v125, v1, op_sel_hi:[1,0,0], cbsz:4, blgp:4
buffer_load_dwordx4 v25, s[4:7], 0, offen, lds
s_waitcnt lgkmcnt(0)
s_barrier
ds_read_b64_tr_b8 v[30:31], v51
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[192:195], v[72:75], a[104:107], v125, v1, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4
ds_read_b64_tr_b8 v[34:35], v51, offset:128
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[164:167], v[60:63], a[108:111], v124, v1, op_sel_hi:[0,1,0], cbsz:4, blgp:4
ds_read_b64_tr_b8 v[32:33], v50
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[168:171], v[64:67], a[108:111], v124, v1, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[172:175], v[60:63], a[112:115], v124, v1, op_sel_hi:[1,1,0], cbsz:4, blgp:4
buffer_load_dword v162, v47, s[8:11], 0, offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[176:179], v[64:67], a[112:115], v124, v1, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[180:183], v[60:63], a[116:119], v125, v1, op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[184:187], v[64:67], a[116:119], v125, v1, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[188:191], v[60:63], a[120:123], v125, v1, op_sel_hi:[1,1,0], cbsz:4, blgp:4
s_waitcnt vmcnt(20), lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[192:195], v[64:67], a[120:123], v125, v1, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4
s_barrier
ds_read_b128 v[166:169], v246
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[108:111], v[126:129], a[132:135], v32, v30, op_sel_hi:[0,0,0], cbsz:4, blgp:4
ds_read_b128 v[170:173], v246, offset:64
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[112:115], v[130:133], a[132:135], v32, v30, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4
ds_read_b128 v[174:177], v246, offset:256
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[116:119], v[126:129], a[140:143], v32, v30, op_sel_hi:[1,0,0], cbsz:4, blgp:4
ds_read_b128 v[178:181], v246, offset:320
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[120:123], v[130:133], a[140:143], v32, v30, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4
ds_read_b128 v[182:185], v246, offset:512
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[142:145], v[126:129], a[144:147], v33, v30, op_sel_hi:[0,0,0], cbsz:4, blgp:4
ds_read_b128 v[186:189], v246, offset:576
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[236:239], v[130:133], a[144:147], v33, v30, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4
ds_read_b128 v[190:193], v246, offset:768
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[100:103], v[126:129], a[152:155], v33, v30, op_sel_hi:[1,0,0], cbsz:4, blgp:4
ds_read_b128 v[240:243], v246, offset:832
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[104:107], v[130:133], a[152:155], v33, v30, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4
s_waitcnt vmcnt(19)
ds_write_b32 v59, v161
s_mov_b32 m0, s50
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[108:111], v[134:137], a[128:131], v32, v30, op_sel_hi:[0,1,0], cbsz:4, blgp:4
buffer_load_dwordx4 v3, s[0:3], 0, offen, lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[112:115], v[138:141], a[128:131], v32, v30, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[116:119], v[134:137], a[136:139], v32, v30, op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[120:123], v[138:141], a[136:139], v32, v30, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4
s_mov_b32 m0, s51
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[142:145], v[134:137], a[148:151], v33, v30, op_sel_hi:[0,1,0], cbsz:4, blgp:4
buffer_load_dwordx4 v5, s[0:3], 0, offen, lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[236:239], v[138:141], a[148:151], v33, v30, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[100:103], v[134:137], a[156:159], v33, v30, op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[104:107], v[138:141], a[156:159], v33, v30, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4
s_mov_b32 m0, s52
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[108:111], v[148:151], a[160:163], v32, v31, op_sel_hi:[0,0,0], cbsz:4, blgp:4
buffer_load_dwordx4 v7, s[0:3], 0, offen, lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[112:115], v[152:155], a[160:163], v32, v31, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[116:119], v[148:151], a[164:167], v32, v31, op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[120:123], v[152:155], a[164:167], v32, v31, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4
s_mov_b32 m0, s53
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[142:145], v[148:151], a[168:171], v33, v31, op_sel_hi:[0,0,0], cbsz:4, blgp:4
buffer_load_dwordx4 v9, s[0:3], 0, offen, lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[236:239], v[152:155], a[168:171], v33, v31, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[100:103], v[148:151], a[172:175], v33, v31, op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[104:107], v[152:155], a[172:175], v33, v31, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4
s_mov_b32 m0, s54
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[108:111], v[196:199], a[176:179], v32, v31, op_sel_hi:[0,1,0], cbsz:4, blgp:4
buffer_load_dwordx4 v11, s[0:3], 0, offen, lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[112:115], v[200:203], a[176:179], v32, v31, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[116:119], v[196:199], a[180:183], v32, v31, op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[120:123], v[200:203], a[180:183], v32, v31, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4
s_mov_b32 m0, s55
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[142:145], v[196:199], a[184:187], v33, v31, op_sel_hi:[0,1,0], cbsz:4, blgp:4
buffer_load_dwordx4 v13, s[0:3], 0, offen, lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[236:239], v[200:203], a[184:187], v33, v31, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[100:103], v[196:199], a[188:191], v33, v31, op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[104:107], v[200:203], a[188:191], v33, v31, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4
s_mov_b32 m0, s56
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[108:111], v[204:207], a[192:195], v32, v34, op_sel_hi:[0,0,0], cbsz:4, blgp:4
buffer_load_dwordx4 v15, s[0:3], 0, offen, lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[112:115], v[208:211], a[192:195], v32, v34, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[116:119], v[204:207], a[196:199], v32, v34, op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[120:123], v[208:211], a[196:199], v32, v34, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4
s_mov_b32 m0, s57
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[142:145], v[204:207], a[204:207], v33, v34, op_sel_hi:[0,0,0], cbsz:4, blgp:4
buffer_load_dwordx4 v17, s[0:3], 0, offen, lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[236:239], v[208:211], a[204:207], v33, v34, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[100:103], v[204:207], a[212:215], v33, v34, op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[104:107], v[208:211], a[212:215], v33, v34, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4
s_mov_b32 m0, s58
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[108:111], v[212:215], a[200:203], v32, v34, op_sel_hi:[0,1,0], cbsz:4, blgp:4
buffer_load_dwordx4 v38, s[4:7], 0, offen, lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[112:115], v[216:219], a[200:203], v32, v34, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[116:119], v[212:215], a[208:211], v32, v34, op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[120:123], v[216:219], a[208:211], v32, v34, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4
s_mov_b32 m0, s59
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[142:145], v[212:215], a[216:219], v33, v34, op_sel_hi:[0,1,0], cbsz:4, blgp:4
buffer_load_dwordx4 v39, s[4:7], 0, offen, lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[236:239], v[216:219], a[216:219], v33, v34, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[100:103], v[212:215], a[220:223], v33, v34, op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[104:107], v[216:219], a[220:223], v33, v34, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4
s_mov_b32 m0, s60
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[108:111], v[220:223], a[224:227], v32, v35, op_sel_hi:[0,0,0], cbsz:4, blgp:4
buffer_load_dwordx4 v40, s[4:7], 0, offen, lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[112:115], v[224:227], a[224:227], v32, v35, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[116:119], v[220:223], a[232:235], v32, v35, op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[120:123], v[224:227], a[232:235], v32, v35, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4
s_mov_b32 m0, s61
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[142:145], v[220:223], a[240:243], v33, v35, op_sel_hi:[0,0,0], cbsz:4, blgp:4
buffer_load_dwordx4 v41, s[4:7], 0, offen, lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[236:239], v[224:227], a[240:243], v33, v35, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[100:103], v[220:223], a[248:251], v33, v35, op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[104:107], v[224:227], a[248:251], v33, v35, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[108:111], v[228:231], a[252:255], v32, v35, op_sel_hi:[0,1,0], cbsz:4, blgp:4
buffer_load_dwordx2 v[156:157], v27, s[16:19], 0, offen
s_waitcnt lgkmcnt(0)
s_barrier
ds_read_b64_tr_b8 v[164:165], v50
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[112:115], v[232:235], a[252:255], v32, v35, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[116:119], v[228:231], a[228:231], v32, v35, op_sel_hi:[1,1,0], cbsz:4, blgp:4
buffer_load_dword v160, v48, s[8:11], 0, offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[120:123], v[232:235], a[228:231], v32, v35, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[142:145], v[228:231], a[236:239], v33, v35, op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[236:239], v[232:235], a[236:239], v33, v35, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[100:103], v[228:231], a[244:247], v33, v35, op_sel_hi:[1,1,0], cbsz:4, blgp:4
s_waitcnt vmcnt(21), lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[104:107], v[232:235], a[244:247], v33, v35, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4
s_barrier
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[166:169], v[126:129], a[124:127], v164, v30, op_sel_hi:[0,0,0], cbsz:4, blgp:4
ds_read_b128 v[116:119], v46
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[170:173], v[130:133], a[124:127], v164, v30, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4
ds_read_b128 v[120:123], v46, offset:64
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[174:177], v[126:129], a[0:3], v164, v30, op_sel_hi:[1,0,0], cbsz:4, blgp:4
ds_read_b128 v[108:111], v46, offset:256
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[178:181], v[130:133], a[0:3], v164, v30, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4
ds_read_b128 v[112:115], v46, offset:320
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[182:185], v[126:129], a[4:7], v165, v30, op_sel_hi:[0,0,0], cbsz:4, blgp:4
ds_read_b128 v[100:103], v46, offset:512
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[186:189], v[130:133], a[4:7], v165, v30, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4
ds_read_b128 v[104:107], v46, offset:576
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[190:193], v[126:129], a[8:11], v165, v30, op_sel_hi:[1,0,0], cbsz:4, blgp:4
ds_read_b128 v[88:91], v46, offset:768
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[240:243], v[130:133], a[8:11], v165, v30, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4
ds_read_b128 v[96:99], v46, offset:832
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[166:169], v[134:137], a[12:15], v164, v30, op_sel_hi:[0,1,0], cbsz:4, blgp:4
ds_read_b128 v[84:87], v46, offset:16896
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[170:173], v[138:141], a[12:15], v164, v30, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4
ds_read_b128 v[92:95], v46, offset:16960
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[174:177], v[134:137], a[16:19], v164, v30, op_sel_hi:[1,1,0], cbsz:4, blgp:4
ds_read_b128 v[76:79], v46, offset:17152
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[178:181], v[138:141], a[16:19], v164, v30, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4
ds_read_b128 v[80:83], v46, offset:17216
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[182:185], v[134:137], a[20:23], v165, v30, op_sel_hi:[0,1,0], cbsz:4, blgp:4
ds_read_b128 v[68:71], v46, offset:17408
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[186:189], v[138:141], a[20:23], v165, v30, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4
ds_read_b128 v[72:75], v46, offset:17472
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[190:193], v[134:137], a[24:27], v165, v30, op_sel_hi:[1,1,0], cbsz:4, blgp:4
ds_read_b128 v[60:63], v46, offset:17664
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[240:243], v[138:141], a[24:27], v165, v30, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4
ds_read_b128 v[64:67], v46, offset:17728
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[166:169], v[148:151], a[28:31], v164, v31, op_sel_hi:[0,0,0], cbsz:4, blgp:4
ds_read_b128 v[124:127], v248
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[170:173], v[152:155], a[28:31], v164, v31, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4
ds_read_b128 v[132:135], v248, offset:64
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[174:177], v[148:151], a[32:35], v164, v31, op_sel_hi:[1,0,0], cbsz:4, blgp:4
ds_read_b128 v[128:131], v248, offset:256
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[178:181], v[152:155], a[32:35], v164, v31, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4
ds_read_b128 v[136:139], v248, offset:320
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[182:185], v[148:151], a[36:39], v165, v31, op_sel_hi:[0,0,0], cbsz:4, blgp:4
ds_read_b128 v[140:143], v248, offset:512
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[186:189], v[152:155], a[36:39], v165, v31, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4
ds_read_b128 v[144:147], v248, offset:576
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[190:193], v[148:151], a[44:47], v165, v31, op_sel_hi:[1,0,0], cbsz:4, blgp:4
ds_read_b128 v[148:151], v248, offset:768
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[240:243], v[152:155], a[44:47], v165, v31, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4
ds_read_b128 v[152:155], v248, offset:832
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[166:169], v[196:199], a[52:55], v164, v31, op_sel_hi:[0,1,0], cbsz:4, blgp:4
s_waitcnt vmcnt(20)
ds_write_b64 v163, v[158:159]
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[170:173], v[200:203], a[52:55], v164, v31, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4
s_waitcnt vmcnt(19)
ds_write_b32 v59, v244
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[174:177], v[196:199], a[40:43], v164, v31, op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[178:181], v[200:203], a[40:43], v164, v31, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[182:185], v[196:199], a[48:51], v165, v31, op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[186:189], v[200:203], a[48:51], v165, v31, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[190:193], v[196:199], a[56:59], v165, v31, op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[240:243], v[200:203], a[56:59], v165, v31, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[166:169], v[204:207], a[60:63], v164, v34, op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[170:173], v[208:211], a[60:63], v164, v34, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[174:177], v[204:207], a[64:67], v164, v34, op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[178:181], v[208:211], a[64:67], v164, v34, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[182:185], v[204:207], a[68:71], v165, v34, op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[186:189], v[208:211], a[68:71], v165, v34, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[190:193], v[204:207], a[72:75], v165, v34, op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[240:243], v[208:211], a[72:75], v165, v34, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[166:169], v[212:215], a[76:79], v164, v34, op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[170:173], v[216:219], a[76:79], v164, v34, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4
s_mov_b32 m0, s28
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[174:177], v[212:215], a[80:83], v164, v34, op_sel_hi:[1,1,0], cbsz:4, blgp:4
buffer_load_dwordx4 v42, s[4:7], 0, offen, lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[178:181], v[216:219], a[80:83], v164, v34, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[182:185], v[212:215], a[84:87], v165, v34, op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[186:189], v[216:219], a[84:87], v165, v34, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4
s_mov_b32 m0, s29
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[190:193], v[212:215], a[88:91], v165, v34, op_sel_hi:[1,1,0], cbsz:4, blgp:4
buffer_load_dwordx4 v43, s[4:7], 0, offen, lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[240:243], v[216:219], a[88:91], v165, v34, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[166:169], v[220:223], a[92:95], v164, v35, op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[170:173], v[224:227], a[92:95], v164, v35, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4
s_mov_b32 m0, s30
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[174:177], v[220:223], a[96:99], v164, v35, op_sel_hi:[1,0,0], cbsz:4, blgp:4
buffer_load_dwordx4 v44, s[4:7], 0, offen, lds
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[178:181], v[224:227], a[96:99], v164, v35, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[182:185], v[220:223], a[100:103], v165, v35, op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[186:189], v[224:227], a[100:103], v165, v35, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4
s_mov_b32 m0, s31
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[190:193], v[220:223], a[104:107], v165, v35, op_sel_hi:[1,0,0], cbsz:4, blgp:4
buffer_load_dwordx4 v45, s[4:7], 0, offen, lds
s_waitcnt lgkmcnt(0)
s_barrier
ds_read_b64_tr_b8 v[30:31], v51
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[240:243], v[224:227], a[104:107], v165, v35, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4
ds_read_b64_tr_b8 v[0:1], v51, offset:128
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[166:169], v[228:231], a[108:111], v164, v35, op_sel_hi:[0,1,0], cbsz:4, blgp:4
ds_read_b64_tr_b8 v[32:33], v50
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[170:173], v[232:235], a[108:111], v164, v35, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[174:177], v[228:231], a[112:115], v164, v35, op_sel_hi:[1,1,0], cbsz:4, blgp:4
buffer_load_dword v161, v49, s[8:11], 0, offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[178:181], v[232:235], a[112:115], v164, v35, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4
s_add_u32 s65, s65, 0x100
s_addc_u32 s66, s66, 0
s_add_u32 s24, s24, 0x100
s_addc_u32 s67, s67, 0
s_add_u32 s20, s20, s63
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[182:185], v[228:231], a[116:119], v165, v35, op_sel_hi:[0,1,0], cbsz:4, blgp:4
s_addc_u32 s21, s21, s64
s_add_u32 s14, s14, s27
s_addc_u32 s15, s15, s62
s_add_i32 s68, s68, 2
s_cmp_lt_u32 s68, 28
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[186:189], v[232:235], a[116:119], v165, v35, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4
s_waitcnt vmcnt(20), lgkmcnt(0)
s_barrier
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[190:193], v[228:231], a[120:123], v165, v35, op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[240:243], v[232:235], a[120:123], v165, v35, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4
s_cbranch_scc1 .LBB0_1
; %bb.2:
v_and_b32_e32 v26, 0xe0, v37
v_lshl_or_b32 v27, v52, 3, s34
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[124:127], v[116:119], a[132:135], v32, v30, op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[132:135], v[120:123], a[132:135], v32, v30, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[128:131], v[116:119], a[140:143], v32, v30, op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[136:139], v[120:123], a[140:143], v32, v30, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[140:143], v[116:119], a[144:147], v33, v30, op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[144:147], v[120:123], a[144:147], v33, v30, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[148:151], v[116:119], a[152:155], v33, v30, op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[152:155], v[120:123], a[152:155], v33, v30, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[124:127], v[108:111], a[128:131], v32, v30, op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[132:135], v[112:115], a[128:131], v32, v30, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[128:131], v[108:111], a[136:139], v32, v30, op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[136:139], v[112:115], a[136:139], v32, v30, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[140:143], v[108:111], a[148:151], v33, v30, op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[144:147], v[112:115], a[148:151], v33, v30, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[148:151], v[108:111], a[156:159], v33, v30, op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[152:155], v[112:115], a[156:159], v33, v30, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[124:127], v[100:103], a[160:163], v32, v31, op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[132:135], v[104:107], a[160:163], v32, v31, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[128:131], v[100:103], a[164:167], v32, v31, op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[136:139], v[104:107], a[164:167], v32, v31, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[140:143], v[100:103], a[168:171], v33, v31, op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[144:147], v[104:107], a[168:171], v33, v31, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[148:151], v[100:103], a[172:175], v33, v31, op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[152:155], v[104:107], a[172:175], v33, v31, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[124:127], v[88:91], a[176:179], v32, v31, op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[132:135], v[96:99], a[176:179], v32, v31, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[128:131], v[88:91], a[180:183], v32, v31, op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[136:139], v[96:99], a[180:183], v32, v31, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[140:143], v[88:91], a[184:187], v33, v31, op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[144:147], v[96:99], a[184:187], v33, v31, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[148:151], v[88:91], a[188:191], v33, v31, op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[152:155], v[96:99], a[188:191], v33, v31, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[124:127], v[84:87], a[192:195], v32, v0, op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[132:135], v[92:95], a[192:195], v32, v0, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4
s_nop 7
v_accvgpr_read_b32 v201, a195
v_accvgpr_read_b32 v200, a194
v_accvgpr_read_b32 v199, a193
v_accvgpr_read_b32 v198, a192
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[128:131], v[84:87], a[196:199], v32, v0, op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[136:139], v[92:95], a[192:195], v32, v0, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4
s_nop 7
v_accvgpr_read_b32 v205, a195
v_accvgpr_read_b32 v204, a194
v_accvgpr_read_b32 v203, a193
v_accvgpr_read_b32 v202, a192
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[140:143], v[84:87], a[204:207], v33, v0, op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[144:147], v[92:95], a[192:195], v33, v0, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[148:151], v[84:87], a[212:215], v33, v0, op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[152:155], v[92:95], a[192:195], v33, v0, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[124:127], v[76:79], a[200:203], v32, v0, op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[132:135], v[80:83], a[192:195], v32, v0, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[128:131], v[76:79], a[208:211], v32, v0, op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[136:139], v[80:83], a[196:199], v32, v0, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[140:143], v[76:79], a[216:219], v33, v0, op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[144:147], v[80:83], a[196:199], v33, v0, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[148:151], v[76:79], a[220:223], v33, v0, op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[152:155], v[80:83], a[208:211], v33, v0, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[124:127], v[68:71], a[224:227], v32, v1, op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[132:135], v[72:75], a[208:211], v32, v1, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[128:131], v[68:71], a[232:235], v32, v1, op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[136:139], v[72:75], a[208:211], v32, v1, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[140:143], v[68:71], a[240:243], v33, v1, op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[144:147], v[72:75], a[208:211], v33, v1, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[148:151], v[68:71], a[248:251], v33, v1, op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[152:155], v[72:75], a[208:211], v33, v1, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4
s_nop 7
v_accvgpr_read_b32 v194, a208
v_accvgpr_read_b32 v195, a209
v_accvgpr_read_b32 v196, a210
v_accvgpr_read_b32 v197, a211
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[124:127], v[60:63], a[252:255], v32, v1, op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[132:135], v[64:67], a[208:211], v32, v1, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[128:131], v[60:63], a[228:231], v32, v1, op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[136:139], v[64:67], a[208:211], v32, v1, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[140:143], v[60:63], a[236:239], v33, v1, op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[144:147], v[64:67], a[216:219], v33, v1, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[148:151], v[60:63], a[244:247], v33, v1, op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[152:155], v[64:67], a[216:219], v33, v1, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4
s_waitcnt vmcnt(20), lgkmcnt(0)
s_barrier
v_add_u32_e32 v2, 0x18bc0, v56
ds_read_b128 v[4:7], v2
ds_read_b128 v[10:13], v2, offset:64
ds_read_b128 v[14:17], v2, offset:256
ds_read_b128 v[18:21], v2, offset:320
ds_read_b128 v[22:25], v2, offset:512
ds_read_b128 v[32:35], v2, offset:576
ds_read_b128 v[38:41], v2, offset:768
ds_read_b128 v[42:45], v2, offset:832
v_add_u32_e32 v8, 0x217a0, v57
s_waitcnt vmcnt(19)
ds_write_b32 v8, v162
s_waitcnt lgkmcnt(0)
s_barrier
ds_read_b64_tr_b8 v[2:3], v50
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[4:7], v[116:119], a[124:127], v2, v30, op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[10:13], v[120:123], a[124:127], v2, v30, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[14:17], v[116:119], a[0:3], v2, v30, op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[18:21], v[120:123], a[0:3], v2, v30, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[22:25], v[116:119], a[4:7], v3, v30, op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[32:35], v[120:123], a[4:7], v3, v30, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[38:41], v[116:119], a[8:11], v3, v30, op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[42:45], v[120:123], a[8:11], v3, v30, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[4:7], v[108:111], a[12:15], v2, v30, op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[10:13], v[112:115], a[12:15], v2, v30, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[14:17], v[108:111], a[16:19], v2, v30, op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[18:21], v[112:115], a[16:19], v2, v30, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[22:25], v[108:111], a[20:23], v3, v30, op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[32:35], v[112:115], a[20:23], v3, v30, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[38:41], v[108:111], a[24:27], v3, v30, op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[42:45], v[112:115], a[24:27], v3, v30, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[4:7], v[100:103], a[28:31], v2, v31, op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[10:13], v[104:107], a[28:31], v2, v31, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[14:17], v[100:103], a[32:35], v2, v31, op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[18:21], v[104:107], a[28:31], v2, v31, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4
s_nop 7
v_accvgpr_read_b32 v193, a31
v_accvgpr_read_b32 v192, a30
v_accvgpr_read_b32 v191, a29
v_accvgpr_read_b32 v190, a28
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[22:25], v[100:103], a[36:39], v3, v31, op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[32:35], v[104:107], a[28:31], v3, v31, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[38:41], v[100:103], a[44:47], v3, v31, op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[42:45], v[104:107], a[32:35], v3, v31, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[4:7], v[88:91], a[52:55], v2, v31, op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[10:13], v[96:99], a[32:35], v2, v31, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[14:17], v[88:91], a[40:43], v2, v31, op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[18:21], v[96:99], a[36:39], v2, v31, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[22:25], v[88:91], a[48:51], v3, v31, op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[32:35], v[96:99], a[36:39], v3, v31, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[38:41], v[88:91], a[56:59], v3, v31, op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[42:45], v[96:99], a[40:43], v3, v31, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[4:7], v[84:87], a[60:63], v2, v0, op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[10:13], v[92:95], a[40:43], v2, v0, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4
s_nop 7
v_accvgpr_read_b32 v189, a43
v_accvgpr_read_b32 v188, a42
v_accvgpr_read_b32 v187, a41
v_accvgpr_read_b32 v186, a40
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[14:17], v[84:87], a[64:67], v2, v0, op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[18:21], v[92:95], a[40:43], v2, v0, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[22:25], v[84:87], a[68:71], v3, v0, op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[32:35], v[92:95], a[40:43], v3, v0, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[38:41], v[84:87], a[72:75], v3, v0, op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[42:45], v[92:95], a[48:51], v3, v0, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[4:7], v[76:79], a[76:79], v2, v0, op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[10:13], v[80:83], a[48:51], v2, v0, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[14:17], v[76:79], a[80:83], v2, v0, op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[18:21], v[80:83], a[52:55], v2, v0, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[22:25], v[76:79], a[84:87], v3, v0, op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[32:35], v[80:83], a[56:59], v3, v0, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[38:41], v[76:79], a[88:91], v3, v0, op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[42:45], v[80:83], a[60:63], v3, v0, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[4:7], v[68:71], a[92:95], v2, v1, op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[10:13], v[72:75], a[72:75], v2, v1, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4
s_nop 7
v_accvgpr_read_b32 v185, a75
v_accvgpr_read_b32 v184, a74
v_accvgpr_read_b32 v183, a73
v_accvgpr_read_b32 v182, a72
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[14:17], v[68:71], a[96:99], v2, v1, op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[18:21], v[72:75], a[72:75], v2, v1, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[22:25], v[68:71], a[100:103], v3, v1, op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[32:35], v[72:75], a[72:75], v3, v1, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[38:41], v[68:71], a[104:107], v3, v1, op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[42:45], v[72:75], a[76:79], v3, v1, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[4:7], v[60:63], a[108:111], v2, v1, op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[10:13], v[64:67], a[80:83], v2, v1, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[14:17], v[60:63], a[112:115], v2, v1, op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[18:21], v[64:67], a[84:87], v2, v1, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[22:25], v[60:63], a[116:119], v3, v1, op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[32:35], v[64:67], a[88:91], v3, v1, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[38:41], v[60:63], a[120:123], v3, v1, op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[42:45], v[64:67], a[92:95], v3, v1, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4
s_waitcnt vmcnt(7), lgkmcnt(0)
s_barrier
ds_read_b128 v[128:131], v46, offset:33792
ds_read_b128 v[132:135], v46, offset:33856
ds_read_b128 v[136:139], v46, offset:34048
ds_read_b128 v[140:143], v46, offset:34112
ds_read_b128 v[100:103], v46, offset:34304
ds_read_b128 v[104:107], v46, offset:34368
ds_read_b128 v[92:95], v46, offset:34560
ds_read_b128 v[96:99], v46, offset:34624
ds_read_b128 v[84:87], v46, offset:50688
ds_read_b128 v[88:91], v46, offset:50752
ds_read_b128 v[76:79], v46, offset:50944
ds_read_b128 v[80:83], v46, offset:51008
ds_read_b128 v[30:33], v46, offset:51200
ds_read_b128 v[38:41], v46, offset:51264
ds_read_b128 v[18:21], v46, offset:51456
ds_read_b128 v[22:25], v46, offset:51520
v_add_u32_e32 v0, 0x149e0, v56
ds_read_b128 v[144:147], v0
ds_read_b128 v[148:151], v0, offset:64
ds_read_b128 v[152:155], v0, offset:256
ds_read_b128 v[162:165], v0, offset:320
ds_read_b128 v[166:169], v0, offset:512
ds_read_b128 v[170:173], v0, offset:576
ds_read_b128 v[174:177], v0, offset:768
ds_read_b128 v[178:181], v0, offset:832
v_add_u32_e32 v0, 0x20fa0, v58
s_waitcnt vmcnt(6)
ds_write_b64 v0, v[156:157]
s_waitcnt vmcnt(5)
ds_write_b32 v8, v160
s_waitcnt lgkmcnt(0)
s_barrier
ds_read_b64_tr_b8 v[4:5], v51
ds_read_b64_tr_b8 v[0:1], v51, offset:128
ds_read_b64_tr_b8 v[6:7], v50
v_lshrrev_b32_e32 v2, 4, v37
v_or_b32_e32 v3, 16, v2
v_or_b32_e32 v9, 32, v2
v_or_b32_e32 v10, 48, v2
v_mul_lo_u32 v28, v2, s26
v_mul_lo_u32 v34, v3, s26
v_mul_lo_u32 v35, v9, s26
v_mul_lo_u32 v37, v10, s26
s_mul_i32 s0, s33, s26
s_ashr_i32 s1, s0, 31
s_lshl_b64 s[0:1], s[0:1], 1
s_add_u32 s12, s12, s0
s_addc_u32 s5, s13, s1
s_lshl_b32 s0, s26, 6
s_ashr_i32 s1, s0, 31
s_lshl_b64 s[0:1], s[0:1], 1
s_add_u32 s8, s12, s0
s_addc_u32 s3, s5, s1
s_add_u32 s4, s8, s0
s_addc_u32 s2, s3, s1
s_add_u32 s0, s4, s0
s_addc_u32 s1, s2, s1
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[144:147], v[128:131], a[132:135], v6, v4, op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[148:151], v[132:135], a[100:103], v6, v4, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[152:155], v[128:131], a[140:143], v6, v4, op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[162:165], v[132:135], a[104:107], v6, v4, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[166:169], v[128:131], a[144:147], v7, v4, op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[170:173], v[132:135], a[108:111], v7, v4, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[174:177], v[128:131], a[152:155], v7, v4, op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[178:181], v[132:135], a[112:115], v7, v4, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[144:147], v[136:139], a[128:131], v6, v4, op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[148:151], v[140:143], a[116:119], v6, v4, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[152:155], v[136:139], a[136:139], v6, v4, op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[162:165], v[140:143], a[120:123], v6, v4, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[166:169], v[136:139], a[148:151], v7, v4, op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[170:173], v[140:143], a[128:131], v7, v4, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[174:177], v[136:139], a[156:159], v7, v4, op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[178:181], v[140:143], a[132:135], v7, v4, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4
s_waitcnt vmcnt(1), lgkmcnt(0)
s_barrier
v_add_u32_e32 v2, 0x1cdc0, v56
ds_read_b128 v[42:45], v2
ds_read_b128 v[46:49], v2, offset:64
ds_read_b128 v[52:55], v2, offset:256
ds_read_b128 v[56:59], v2, offset:320
ds_read_b128 v[60:63], v2, offset:512
ds_read_b128 v[64:67], v2, offset:576
ds_read_b128 v[68:71], v2, offset:768
ds_read_b128 v[72:75], v2, offset:832
s_waitcnt vmcnt(0)
ds_write_b32 v8, v161
s_waitcnt lgkmcnt(0)
s_barrier
ds_read_b64_tr_b8 v[2:3], v50
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[144:147], v[100:103], a[160:163], v6, v5, op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[148:151], v[104:107], a[136:139], v6, v5, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[152:155], v[100:103], a[164:167], v6, v5, op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[162:165], v[104:107], a[140:143], v6, v5, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[166:169], v[100:103], a[168:171], v7, v5, op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[170:173], v[104:107], a[144:147], v7, v5, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[174:177], v[100:103], a[172:175], v7, v5, op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[178:181], v[104:107], a[148:151], v7, v5, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[144:147], v[92:95], a[176:179], v6, v5, op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[148:151], v[96:99], a[152:155], v6, v5, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[152:155], v[92:95], a[180:183], v6, v5, op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[162:165], v[96:99], a[156:159], v6, v5, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[166:169], v[92:95], a[184:187], v7, v5, op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[170:173], v[96:99], a[160:163], v7, v5, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[174:177], v[92:95], a[188:191], v7, v5, op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[178:181], v[96:99], a[164:167], v7, v5, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_accvgpr_read_b32 v8, a100
v_accvgpr_read_b32 v9, a101
v_cvt_pk_bf16_f32 v10, v8, v9
v_accvgpr_read_b32 v8, a102
v_accvgpr_read_b32 v9, a103
v_cvt_pk_bf16_f32 v11, v8, v9
v_accvgpr_read_b32 v8, a104
v_accvgpr_read_b32 v9, a105
v_cvt_pk_bf16_f32 v14, v8, v9
v_accvgpr_read_b32 v8, a106
v_accvgpr_read_b32 v9, a107
v_cvt_pk_bf16_f32 v15, v8, v9
v_accvgpr_read_b32 v8, a108
v_accvgpr_read_b32 v9, a109
v_cvt_pk_bf16_f32 v108, v8, v9
v_accvgpr_read_b32 v8, a110
v_accvgpr_read_b32 v9, a111
v_cvt_pk_bf16_f32 v109, v8, v9
v_accvgpr_read_b32 v8, a112
v_accvgpr_read_b32 v9, a113
v_cvt_pk_bf16_f32 v112, v8, v9
v_accvgpr_read_b32 v8, a114
v_accvgpr_read_b32 v9, a115
v_cvt_pk_bf16_f32 v113, v8, v9
v_accvgpr_read_b32 v8, a116
v_accvgpr_read_b32 v9, a117
v_cvt_pk_bf16_f32 v12, v8, v9
v_accvgpr_read_b32 v8, a118
v_accvgpr_read_b32 v9, a119
v_cvt_pk_bf16_f32 v13, v8, v9
v_accvgpr_read_b32 v8, a120
v_accvgpr_read_b32 v9, a121
v_cvt_pk_bf16_f32 v16, v8, v9
v_accvgpr_read_b32 v8, a122
v_accvgpr_read_b32 v9, a123
v_cvt_pk_bf16_f32 v17, v8, v9
v_accvgpr_read_b32 v8, a128
v_accvgpr_read_b32 v9, a129
v_cvt_pk_bf16_f32 v110, v8, v9
v_accvgpr_read_b32 v8, a130
v_accvgpr_read_b32 v9, a131
v_cvt_pk_bf16_f32 v111, v8, v9
v_accvgpr_read_b32 v8, a132
v_accvgpr_read_b32 v9, a133
v_cvt_pk_bf16_f32 v114, v8, v9
v_accvgpr_read_b32 v8, a134
v_accvgpr_read_b32 v9, a135
v_cvt_pk_bf16_f32 v115, v8, v9
v_lshlrev_b32_e32 v8, 8, v36
v_and_b32_e32 v50, 0x70, v29
v_and_b32_e32 v51, 1, v36
v_lshlrev_b32_e32 v9, 12, v51
v_and_b32_e32 v36, 16, v36
v_lshlrev_b32_e32 v116, 4, v36
s_movk_i32 s6, 0x2e00
v_and_or_b32 v8, v8, s6, v9
v_mov_b32_e32 v9, 0x70
v_bitop3_b32 v9, s23, v29, v9, bitop3:0x78
v_or3_b32 v29, v116, v8, v9
v_or_b32_e32 v116, s22, v29
v_add_u32_e32 v8, 0, v116
ds_write_b128 v8, v[10:13]
v_xad_u32 v9, v116, 32, 0
ds_write_b128 v9, v[14:17]
v_xad_u32 v10, v116, 64, 0
ds_write_b128 v10, v[108:111]
s_movk_i32 s6, 0x60
v_mov_b32_e32 v11, s22
v_bitop3_b32 v11, v29, s6, v11, bitop3:0x36
v_add_u32_e32 v11, 0, v11
ds_write_b128 v11, v[112:115]
s_waitcnt lgkmcnt(0)
s_barrier
v_lshlrev_b32_e32 v12, 4, v26
v_lshrrev_b32_e32 v13, 1, v26
v_lshlrev_b32_e32 v14, 8, v36
v_bitop3_b32 v12, v12, v13, v50, bitop3:0x36
v_lshl_add_u32 v13, v51, 13, 0
v_add3_u32 v12, v13, v14, v12
ds_read_b128 v[108:111], v12
ds_read_b128 v[114:117], v12, offset:256
ds_read_b128 v[118:121], v12, offset:128
ds_read_b128 v[124:127], v12, offset:384
s_and_b32 s13, s5, 0xffff
s_mov_b32 s15, 0x27000
s_mov_b32 s14, 0x7ffffffe
s_waitcnt lgkmcnt(3)
v_mov_b32_e32 v112, v108
v_mov_b32_e32 v113, v109
v_add_lshl_u32 v13, v28, v27, 1
s_waitcnt lgkmcnt(2)
buffer_store_dwordx4 v[112:115], v13, s[12:15], 0, offen
s_waitcnt lgkmcnt(1)
v_mov_b32_e32 v122, v118
v_mov_b32_e32 v123, v119
v_add_lshl_u32 v14, v34, v27, 1
s_waitcnt lgkmcnt(0)
buffer_store_dwordx4 v[122:125], v14, s[12:15], 0, offen
v_mov_b32_e32 v112, v116
v_mov_b32_e32 v113, v117
v_add_lshl_u32 v15, v35, v27, 1
buffer_store_dwordx4 v[110:113], v15, s[12:15], 0, offen
v_mov_b32_e32 v122, v126
v_mov_b32_e32 v123, v127
v_add_lshl_u32 v16, v37, v27, 1
buffer_store_dwordx4 v[120:123], v16, s[12:15], 0, offen
v_accvgpr_write_b32 a100, v198
v_accvgpr_write_b32 a101, v199
v_accvgpr_write_b32 a102, v200
v_accvgpr_write_b32 a103, v201
s_nop 1
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[144:147], v[84:87], a[100:103], v6, v0, op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[148:151], v[88:91], a[100:103], v6, v0, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_accvgpr_write_b32 a104, v202
v_accvgpr_write_b32 a105, v203
v_accvgpr_write_b32 a106, v204
v_accvgpr_write_b32 a107, v205
s_nop 1
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[152:155], v[84:87], a[104:107], v6, v0, op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[162:165], v[88:91], a[104:107], v6, v0, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[166:169], v[84:87], a[204:207], v7, v0, op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[170:173], v[88:91], a[108:111], v7, v0, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[174:177], v[84:87], a[212:215], v7, v0, op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[178:181], v[88:91], a[112:115], v7, v0, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[144:147], v[76:79], a[192:195], v6, v0, op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[148:151], v[80:83], a[116:119], v6, v0, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[152:155], v[76:79], a[200:203], v6, v0, op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[162:165], v[80:83], a[120:123], v6, v0, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[166:169], v[76:79], a[196:199], v7, v0, op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[170:173], v[80:83], a[128:131], v7, v0, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[174:177], v[76:79], a[220:223], v7, v0, op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[178:181], v[80:83], a[132:135], v7, v0, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_accvgpr_read_b32 v26, a136
v_accvgpr_read_b32 v17, a137
v_cvt_pk_bf16_f32 v26, v26, v17
v_accvgpr_read_b32 v28, a138
v_accvgpr_read_b32 v17, a139
v_cvt_pk_bf16_f32 v27, v28, v17
v_accvgpr_read_b32 v28, a140
v_accvgpr_read_b32 v17, a141
v_cvt_pk_bf16_f32 v34, v28, v17
v_accvgpr_read_b32 v28, a142
v_accvgpr_read_b32 v17, a143
v_cvt_pk_bf16_f32 v35, v28, v17
v_accvgpr_read_b32 v28, a144
v_accvgpr_read_b32 v17, a145
v_cvt_pk_bf16_f32 v108, v28, v17
v_accvgpr_read_b32 v28, a146
v_accvgpr_read_b32 v17, a147
v_cvt_pk_bf16_f32 v109, v28, v17
v_accvgpr_read_b32 v28, a148
v_accvgpr_read_b32 v17, a149
v_cvt_pk_bf16_f32 v112, v28, v17
v_accvgpr_read_b32 v28, a150
v_accvgpr_read_b32 v17, a151
v_cvt_pk_bf16_f32 v113, v28, v17
v_accvgpr_read_b32 v28, a152
v_accvgpr_read_b32 v17, a153
v_cvt_pk_bf16_f32 v28, v28, v17
v_accvgpr_read_b32 v36, a154
v_accvgpr_read_b32 v17, a155
v_cvt_pk_bf16_f32 v29, v36, v17
v_accvgpr_read_b32 v36, a156
v_accvgpr_read_b32 v17, a157
v_cvt_pk_bf16_f32 v36, v36, v17
v_accvgpr_read_b32 v50, a158
v_accvgpr_read_b32 v17, a159
v_cvt_pk_bf16_f32 v37, v50, v17
v_accvgpr_read_b32 v50, a160
v_accvgpr_read_b32 v17, a161
v_cvt_pk_bf16_f32 v110, v50, v17
v_accvgpr_read_b32 v50, a162
v_accvgpr_read_b32 v17, a163
v_cvt_pk_bf16_f32 v111, v50, v17
v_accvgpr_read_b32 v50, a164
v_accvgpr_read_b32 v17, a165
v_cvt_pk_bf16_f32 v114, v50, v17
v_accvgpr_read_b32 v50, a166
v_accvgpr_read_b32 v17, a167
v_cvt_pk_bf16_f32 v115, v50, v17
s_waitcnt lgkmcnt(0)
s_barrier
ds_write_b128 v8, v[26:29]
ds_write_b128 v9, v[34:37]
ds_write_b128 v10, v[108:111]
ds_write_b128 v11, v[112:115]
s_waitcnt lgkmcnt(0)
s_barrier
ds_read_b128 v[108:111], v12
ds_read_b128 v[114:117], v12, offset:256
ds_read_b128 v[118:121], v12, offset:128
ds_read_b128 v[124:127], v12, offset:384
s_and_b32 s9, s3, 0xffff
s_mov_b32 s10, s14
s_mov_b32 s11, s15
s_waitcnt lgkmcnt(3)
v_mov_b32_e32 v112, v108
v_mov_b32_e32 v113, v109
s_waitcnt lgkmcnt(2)
buffer_store_dwordx4 v[112:115], v13, s[8:11], 0, offen
s_waitcnt lgkmcnt(1)
v_mov_b32_e32 v122, v118
v_mov_b32_e32 v123, v119
s_waitcnt lgkmcnt(0)
buffer_store_dwordx4 v[122:125], v14, s[8:11], 0, offen
v_mov_b32_e32 v112, v116
v_mov_b32_e32 v113, v117
buffer_store_dwordx4 v[110:113], v15, s[8:11], 0, offen
v_mov_b32_e32 v122, v126
v_mov_b32_e32 v123, v127
buffer_store_dwordx4 v[120:123], v16, s[8:11], 0, offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[144:147], v[30:33], a[224:227], v6, v1, op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[148:151], v[38:41], a[136:139], v6, v1, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[152:155], v[30:33], a[232:235], v6, v1, op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[162:165], v[38:41], a[140:143], v6, v1, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[166:169], v[30:33], a[240:243], v7, v1, op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[170:173], v[38:41], a[144:147], v7, v1, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_accvgpr_write_b32 a148, v194
v_accvgpr_write_b32 a149, v195
v_accvgpr_write_b32 a150, v196
v_accvgpr_write_b32 a151, v197
s_nop 1
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[174:177], v[30:33], a[148:151], v7, v1, op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[178:181], v[38:41], a[148:151], v7, v1, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[144:147], v[18:21], a[248:251], v6, v1, op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[148:151], v[22:25], a[152:155], v6, v1, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[152:155], v[18:21], a[208:211], v6, v1, op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[162:165], v[22:25], a[156:159], v6, v1, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[166:169], v[18:21], a[228:231], v7, v1, op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[170:173], v[22:25], a[160:163], v7, v1, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[174:177], v[18:21], a[216:219], v7, v1, op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[178:181], v[22:25], a[164:167], v7, v1, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_accvgpr_read_b32 v6, a100
v_accvgpr_read_b32 v7, a101
v_cvt_pk_bf16_f32 v26, v6, v7
v_accvgpr_read_b32 v6, a102
v_accvgpr_read_b32 v7, a103
v_cvt_pk_bf16_f32 v27, v6, v7
v_accvgpr_read_b32 v6, a104
v_accvgpr_read_b32 v7, a105
v_cvt_pk_bf16_f32 v34, v6, v7
v_accvgpr_read_b32 v6, a106
v_accvgpr_read_b32 v7, a107
v_cvt_pk_bf16_f32 v35, v6, v7
v_accvgpr_read_b32 v6, a108
v_accvgpr_read_b32 v7, a109
v_cvt_pk_bf16_f32 v108, v6, v7
v_accvgpr_read_b32 v6, a110
v_accvgpr_read_b32 v7, a111
v_cvt_pk_bf16_f32 v109, v6, v7
v_accvgpr_read_b32 v6, a112
v_accvgpr_read_b32 v7, a113
v_cvt_pk_bf16_f32 v112, v6, v7
v_accvgpr_read_b32 v6, a114
v_accvgpr_read_b32 v7, a115
v_cvt_pk_bf16_f32 v113, v6, v7
v_accvgpr_read_b32 v6, a116
v_accvgpr_read_b32 v7, a117
v_cvt_pk_bf16_f32 v28, v6, v7
v_accvgpr_read_b32 v6, a118
v_accvgpr_read_b32 v7, a119
v_cvt_pk_bf16_f32 v29, v6, v7
v_accvgpr_read_b32 v6, a120
v_accvgpr_read_b32 v7, a121
v_cvt_pk_bf16_f32 v36, v6, v7
v_accvgpr_read_b32 v6, a122
v_accvgpr_read_b32 v7, a123
v_cvt_pk_bf16_f32 v37, v6, v7
v_accvgpr_read_b32 v6, a128
v_accvgpr_read_b32 v7, a129
v_cvt_pk_bf16_f32 v110, v6, v7
v_accvgpr_read_b32 v6, a130
v_accvgpr_read_b32 v7, a131
v_cvt_pk_bf16_f32 v111, v6, v7
v_accvgpr_read_b32 v6, a132
v_accvgpr_read_b32 v7, a133
v_cvt_pk_bf16_f32 v114, v6, v7
v_accvgpr_read_b32 v6, a134
v_accvgpr_read_b32 v7, a135
v_cvt_pk_bf16_f32 v115, v6, v7
s_waitcnt lgkmcnt(0)
s_barrier
ds_write_b128 v8, v[26:29]
ds_write_b128 v9, v[34:37]
ds_write_b128 v10, v[108:111]
ds_write_b128 v11, v[112:115]
s_waitcnt lgkmcnt(0)
s_barrier
ds_read_b128 v[108:111], v12
ds_read_b128 v[114:117], v12, offset:256
ds_read_b128 v[118:121], v12, offset:128
ds_read_b128 v[124:127], v12, offset:384
s_and_b32 s5, s2, 0xffff
s_mov_b32 s6, s14
s_mov_b32 s7, s15
s_waitcnt lgkmcnt(3)
v_mov_b32_e32 v112, v108
v_mov_b32_e32 v113, v109
s_waitcnt lgkmcnt(2)
buffer_store_dwordx4 v[112:115], v13, s[4:7], 0, offen
s_waitcnt lgkmcnt(1)
v_mov_b32_e32 v122, v118
v_mov_b32_e32 v123, v119
s_waitcnt lgkmcnt(0)
buffer_store_dwordx4 v[122:125], v14, s[4:7], 0, offen
v_mov_b32_e32 v112, v116
v_mov_b32_e32 v113, v117
buffer_store_dwordx4 v[110:113], v15, s[4:7], 0, offen
v_mov_b32_e32 v122, v126
v_mov_b32_e32 v123, v127
buffer_store_dwordx4 v[120:123], v16, s[4:7], 0, offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[42:45], v[128:131], a[252:255], v2, v4, op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[46:49], v[132:135], a[100:103], v2, v4, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[52:55], v[128:131], a[0:3], v2, v4, op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[56:59], v[132:135], a[0:3], v2, v4, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[60:63], v[128:131], a[4:7], v3, v4, op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[64:67], v[132:135], a[4:7], v3, v4, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[68:71], v[128:131], a[8:11], v3, v4, op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[72:75], v[132:135], a[8:11], v3, v4, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[42:45], v[136:139], a[12:15], v2, v4, op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[46:49], v[140:143], a[12:15], v2, v4, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[52:55], v[136:139], a[16:19], v2, v4, op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[56:59], v[140:143], a[16:19], v2, v4, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[60:63], v[136:139], a[20:23], v3, v4, op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[64:67], v[140:143], a[20:23], v3, v4, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[68:71], v[136:139], a[24:27], v3, v4, op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[72:75], v[140:143], a[24:27], v3, v4, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_accvgpr_read_b32 v4, a136
v_accvgpr_read_b32 v7, a137
v_cvt_pk_bf16_f32 v26, v4, v7
v_accvgpr_read_b32 v4, a138
v_accvgpr_read_b32 v7, a139
v_cvt_pk_bf16_f32 v27, v4, v7
v_accvgpr_read_b32 v4, a140
v_accvgpr_read_b32 v7, a141
v_cvt_pk_bf16_f32 v34, v4, v7
v_accvgpr_read_b32 v4, a142
v_accvgpr_read_b32 v7, a143
v_cvt_pk_bf16_f32 v35, v4, v7
v_accvgpr_read_b32 v4, a144
v_accvgpr_read_b32 v7, a145
v_cvt_pk_bf16_f32 v108, v4, v7
v_accvgpr_read_b32 v4, a146
v_accvgpr_read_b32 v7, a147
v_cvt_pk_bf16_f32 v109, v4, v7
v_accvgpr_read_b32 v4, a148
v_accvgpr_read_b32 v7, a149
v_cvt_pk_bf16_f32 v112, v4, v7
v_accvgpr_read_b32 v4, a150
v_accvgpr_read_b32 v7, a151
v_cvt_pk_bf16_f32 v113, v4, v7
v_accvgpr_read_b32 v4, a152
v_accvgpr_read_b32 v7, a153
v_cvt_pk_bf16_f32 v28, v4, v7
v_accvgpr_read_b32 v4, a154
v_accvgpr_read_b32 v7, a155
v_cvt_pk_bf16_f32 v29, v4, v7
v_accvgpr_read_b32 v4, a156
v_accvgpr_read_b32 v7, a157
v_cvt_pk_bf16_f32 v36, v4, v7
v_accvgpr_read_b32 v4, a158
v_accvgpr_read_b32 v7, a159
v_cvt_pk_bf16_f32 v37, v4, v7
v_accvgpr_read_b32 v4, a160
v_accvgpr_read_b32 v7, a161
v_cvt_pk_bf16_f32 v110, v4, v7
v_accvgpr_read_b32 v4, a162
v_accvgpr_read_b32 v7, a163
v_cvt_pk_bf16_f32 v111, v4, v7
v_accvgpr_read_b32 v4, a164
v_accvgpr_read_b32 v7, a165
v_cvt_pk_bf16_f32 v114, v4, v7
v_accvgpr_read_b32 v4, a166
v_accvgpr_read_b32 v7, a167
v_cvt_pk_bf16_f32 v115, v4, v7
s_waitcnt lgkmcnt(0)
s_barrier
ds_write_b128 v8, v[26:29]
ds_write_b128 v9, v[34:37]
ds_write_b128 v10, v[108:111]
ds_write_b128 v11, v[112:115]
s_waitcnt lgkmcnt(0)
s_barrier
ds_read_b128 v[108:111], v12
ds_read_b128 v[114:117], v12, offset:256
ds_read_b128 v[118:121], v12, offset:128
ds_read_b128 v[124:127], v12, offset:384
s_and_b32 s1, s1, 0xffff
s_mov_b32 s2, s14
s_mov_b32 s3, s15
s_waitcnt lgkmcnt(3)
v_mov_b32_e32 v112, v108
v_mov_b32_e32 v113, v109
s_waitcnt lgkmcnt(2)
buffer_store_dwordx4 v[112:115], v13, s[0:3], 0, offen
s_waitcnt lgkmcnt(1)
v_mov_b32_e32 v122, v118
v_mov_b32_e32 v123, v119
s_waitcnt lgkmcnt(0)
buffer_store_dwordx4 v[122:125], v14, s[0:3], 0, offen
v_mov_b32_e32 v112, v116
v_mov_b32_e32 v113, v117
buffer_store_dwordx4 v[110:113], v15, s[0:3], 0, offen
v_mov_b32_e32 v122, v126
v_mov_b32_e32 v123, v127
buffer_store_dwordx4 v[120:123], v16, s[0:3], 0, offen
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[42:45], v[100:103], a[124:127], v2, v5, op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[46:49], v[104:107], a[104:107], v2, v5, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_accvgpr_write_b32 a108, v190
v_accvgpr_write_b32 a109, v191
v_accvgpr_write_b32 a110, v192
v_accvgpr_write_b32 a111, v193
s_nop 1
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[52:55], v[100:103], a[108:111], v2, v5, op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[56:59], v[104:107], a[108:111], v2, v5, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[60:63], v[100:103], a[28:31], v3, v5, op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[64:67], v[104:107], a[28:31], v3, v5, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[68:71], v[100:103], a[44:47], v3, v5, op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[72:75], v[104:107], a[44:47], v3, v5, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[42:45], v[92:95], a[32:35], v2, v5, op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[46:49], v[96:99], a[32:35], v2, v5, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[52:55], v[92:95], a[236:239], v2, v5, op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[56:59], v[96:99], a[112:115], v2, v5, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[60:63], v[92:95], a[36:39], v3, v5, op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[64:67], v[96:99], a[36:39], v3, v5, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[68:71], v[92:95], a[244:247], v3, v5, op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[72:75], v[96:99], a[116:119], v3, v5, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_accvgpr_read_b32 v4, a100
v_accvgpr_read_b32 v5, a101
v_cvt_pk_bf16_f32 v4, v4, v5
v_accvgpr_read_b32 v6, a102
v_accvgpr_read_b32 v5, a103
v_cvt_pk_bf16_f32 v5, v6, v5
v_accvgpr_read_b32 v6, a0
v_accvgpr_read_b32 v7, a1
v_cvt_pk_bf16_f32 v26, v6, v7
v_accvgpr_read_b32 v6, a2
v_accvgpr_read_b32 v7, a3
v_cvt_pk_bf16_f32 v27, v6, v7
v_accvgpr_read_b32 v6, a4
v_accvgpr_read_b32 v7, a5
v_cvt_pk_bf16_f32 v34, v6, v7
v_accvgpr_read_b32 v6, a6
v_accvgpr_read_b32 v7, a7
v_cvt_pk_bf16_f32 v35, v6, v7
v_accvgpr_read_b32 v6, a8
v_accvgpr_read_b32 v7, a9
v_cvt_pk_bf16_f32 v92, v6, v7
v_accvgpr_read_b32 v6, a10
v_accvgpr_read_b32 v7, a11
v_cvt_pk_bf16_f32 v93, v6, v7
v_accvgpr_read_b32 v6, a12
v_accvgpr_read_b32 v7, a13
v_cvt_pk_bf16_f32 v6, v6, v7
v_accvgpr_read_b32 v28, a14
v_accvgpr_read_b32 v7, a15
v_cvt_pk_bf16_f32 v7, v28, v7
v_accvgpr_read_b32 v28, a16
v_accvgpr_read_b32 v17, a17
v_cvt_pk_bf16_f32 v28, v28, v17
v_accvgpr_read_b32 v36, a18
v_accvgpr_read_b32 v17, a19
v_cvt_pk_bf16_f32 v29, v36, v17
v_accvgpr_read_b32 v36, a20
v_accvgpr_read_b32 v17, a21
v_cvt_pk_bf16_f32 v36, v36, v17
v_accvgpr_read_b32 v50, a22
v_accvgpr_read_b32 v17, a23
v_cvt_pk_bf16_f32 v37, v50, v17
v_accvgpr_read_b32 v50, a24
v_accvgpr_read_b32 v17, a25
v_cvt_pk_bf16_f32 v94, v50, v17
v_accvgpr_read_b32 v50, a26
v_accvgpr_read_b32 v17, a27
v_cvt_pk_bf16_f32 v95, v50, v17
s_waitcnt lgkmcnt(0)
s_barrier
ds_write_b128 v8, v[4:7]
ds_write_b128 v9, v[26:29]
ds_write_b128 v10, v[34:37]
ds_write_b128 v11, v[92:95]
s_waitcnt lgkmcnt(0)
s_barrier
ds_read_b128 v[92:95], v12
ds_read_b128 v[98:101], v12, offset:256
ds_read_b128 v[102:105], v12, offset:128
ds_read_b128 v[108:111], v12, offset:384
s_waitcnt lgkmcnt(3)
v_mov_b32_e32 v96, v92
v_mov_b32_e32 v97, v93
s_waitcnt lgkmcnt(2)
buffer_store_dwordx4 v[96:99], v13, s[12:15], 0, offen, offset:256
s_waitcnt lgkmcnt(1)
v_mov_b32_e32 v106, v102
v_mov_b32_e32 v107, v103
s_waitcnt lgkmcnt(0)
buffer_store_dwordx4 v[106:109], v14, s[12:15], 0, offen, offset:256
v_mov_b32_e32 v96, v100
v_mov_b32_e32 v97, v101
buffer_store_dwordx4 v[94:97], v15, s[12:15], 0, offen, offset:256
v_mov_b32_e32 v106, v110
v_mov_b32_e32 v107, v111
buffer_store_dwordx4 v[104:107], v16, s[12:15], 0, offen, offset:256
v_accvgpr_write_b32 a0, v186
v_accvgpr_write_b32 a1, v187
v_accvgpr_write_b32 a2, v188
v_accvgpr_write_b32 a3, v189
s_nop 1
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[42:45], v[84:87], a[0:3], v2, v0, op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[46:49], v[88:91], a[0:3], v2, v0, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[52:55], v[84:87], a[64:67], v2, v0, op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[56:59], v[88:91], a[4:7], v2, v0, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[60:63], v[84:87], a[40:43], v3, v0, op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[64:67], v[88:91], a[8:11], v3, v0, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[68:71], v[84:87], a[68:71], v3, v0, op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[72:75], v[88:91], a[12:15], v3, v0, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[42:45], v[76:79], a[48:51], v2, v0, op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[46:49], v[80:83], a[16:19], v2, v0, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[52:55], v[76:79], a[52:55], v2, v0, op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[56:59], v[80:83], a[20:23], v2, v0, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[60:63], v[76:79], a[56:59], v3, v0, op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[64:67], v[80:83], a[24:27], v3, v0, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[68:71], v[76:79], a[60:63], v3, v0, op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[72:75], v[80:83], a[40:43], v3, v0, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_accvgpr_read_b32 v0, a104
v_accvgpr_read_b32 v5, a105
v_cvt_pk_bf16_f32 v4, v0, v5
v_accvgpr_read_b32 v0, a106
v_accvgpr_read_b32 v5, a107
v_cvt_pk_bf16_f32 v5, v0, v5
v_accvgpr_read_b32 v0, a108
v_accvgpr_read_b32 v7, a109
v_cvt_pk_bf16_f32 v26, v0, v7
v_accvgpr_read_b32 v0, a110
v_accvgpr_read_b32 v7, a111
v_cvt_pk_bf16_f32 v27, v0, v7
v_accvgpr_read_b32 v0, a28
v_accvgpr_read_b32 v7, a29
v_cvt_pk_bf16_f32 v34, v0, v7
v_accvgpr_read_b32 v0, a30
v_accvgpr_read_b32 v7, a31
v_cvt_pk_bf16_f32 v35, v0, v7
v_accvgpr_read_b32 v0, a44
v_accvgpr_read_b32 v7, a45
v_cvt_pk_bf16_f32 v76, v0, v7
v_accvgpr_read_b32 v0, a46
v_accvgpr_read_b32 v7, a47
v_cvt_pk_bf16_f32 v77, v0, v7
v_accvgpr_read_b32 v0, a32
v_accvgpr_read_b32 v7, a33
v_cvt_pk_bf16_f32 v6, v0, v7
v_accvgpr_read_b32 v0, a34
v_accvgpr_read_b32 v7, a35
v_cvt_pk_bf16_f32 v7, v0, v7
v_accvgpr_read_b32 v0, a112
v_accvgpr_read_b32 v17, a113
v_cvt_pk_bf16_f32 v28, v0, v17
v_accvgpr_read_b32 v0, a114
v_accvgpr_read_b32 v17, a115
v_cvt_pk_bf16_f32 v29, v0, v17
v_accvgpr_read_b32 v0, a36
v_accvgpr_read_b32 v17, a37
v_cvt_pk_bf16_f32 v36, v0, v17
v_accvgpr_read_b32 v0, a38
v_accvgpr_read_b32 v17, a39
v_cvt_pk_bf16_f32 v37, v0, v17
v_accvgpr_read_b32 v0, a116
v_accvgpr_read_b32 v17, a117
v_cvt_pk_bf16_f32 v78, v0, v17
v_accvgpr_read_b32 v0, a118
v_accvgpr_read_b32 v17, a119
v_cvt_pk_bf16_f32 v79, v0, v17
s_waitcnt lgkmcnt(0)
s_barrier
ds_write_b128 v8, v[4:7]
ds_write_b128 v9, v[26:29]
ds_write_b128 v10, v[34:37]
ds_write_b128 v11, v[76:79]
s_waitcnt lgkmcnt(0)
s_barrier
ds_read_b128 v[76:79], v12
ds_read_b128 v[82:85], v12, offset:256
ds_read_b128 v[86:89], v12, offset:128
ds_read_b128 v[92:95], v12, offset:384
s_waitcnt lgkmcnt(3)
v_mov_b32_e32 v80, v76
v_mov_b32_e32 v81, v77
s_waitcnt lgkmcnt(2)
buffer_store_dwordx4 v[80:83], v13, s[8:11], 0, offen, offset:256
s_waitcnt lgkmcnt(1)
v_mov_b32_e32 v90, v86
v_mov_b32_e32 v91, v87
s_waitcnt lgkmcnt(0)
buffer_store_dwordx4 v[90:93], v14, s[8:11], 0, offen, offset:256
v_mov_b32_e32 v80, v84
v_mov_b32_e32 v81, v85
buffer_store_dwordx4 v[78:81], v15, s[8:11], 0, offen, offset:256
v_mov_b32_e32 v90, v94
v_mov_b32_e32 v91, v95
buffer_store_dwordx4 v[88:91], v16, s[8:11], 0, offen, offset:256
v_accvgpr_write_b32 a28, v182
v_accvgpr_write_b32 a29, v183
v_accvgpr_write_b32 a30, v184
v_accvgpr_write_b32 a31, v185
s_nop 1
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[42:45], v[30:33], a[28:31], v2, v1, op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[46:49], v[38:41], a[28:31], v2, v1, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[52:55], v[30:33], a[96:99], v2, v1, op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[56:59], v[38:41], a[32:35], v2, v1, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[60:63], v[30:33], a[72:75], v3, v1, op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[64:67], v[38:41], a[36:39], v3, v1, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[68:71], v[30:33], a[76:79], v3, v1, op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[72:75], v[38:41], a[44:47], v3, v1, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[42:45], v[18:21], a[80:83], v2, v1, op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[46:49], v[22:25], a[48:51], v2, v1, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[52:55], v[18:21], a[84:87], v2, v1, op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[56:59], v[22:25], a[52:55], v2, v1, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[60:63], v[18:21], a[88:91], v3, v1, op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[64:67], v[22:25], a[56:59], v3, v1, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[68:71], v[18:21], a[92:95], v3, v1, op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[72:75], v[22:25], a[60:63], v3, v1, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4
v_accvgpr_read_b32 v0, a0
v_accvgpr_read_b32 v1, a1
v_cvt_pk_bf16_f32 v0, v0, v1
v_accvgpr_read_b32 v2, a2
v_accvgpr_read_b32 v1, a3
v_cvt_pk_bf16_f32 v1, v2, v1
v_accvgpr_read_b32 v2, a4
v_accvgpr_read_b32 v3, a5
v_cvt_pk_bf16_f32 v4, v2, v3
v_accvgpr_read_b32 v2, a6
v_accvgpr_read_b32 v3, a7
v_cvt_pk_bf16_f32 v5, v2, v3
v_accvgpr_read_b32 v2, a8
v_accvgpr_read_b32 v3, a9
v_cvt_pk_bf16_f32 v18, v2, v3
v_accvgpr_read_b32 v2, a10
v_accvgpr_read_b32 v3, a11
v_cvt_pk_bf16_f32 v19, v2, v3
v_accvgpr_read_b32 v2, a12
v_accvgpr_read_b32 v3, a13
v_cvt_pk_bf16_f32 v22, v2, v3
v_accvgpr_read_b32 v2, a14
v_accvgpr_read_b32 v3, a15
v_cvt_pk_bf16_f32 v23, v2, v3
v_accvgpr_read_b32 v2, a16
v_accvgpr_read_b32 v3, a17
v_cvt_pk_bf16_f32 v2, v2, v3
v_accvgpr_read_b32 v6, a18
v_accvgpr_read_b32 v3, a19
v_cvt_pk_bf16_f32 v3, v6, v3
v_accvgpr_read_b32 v6, a20
v_accvgpr_read_b32 v7, a21
v_cvt_pk_bf16_f32 v6, v6, v7
v_accvgpr_read_b32 v20, a22
v_accvgpr_read_b32 v7, a23
v_cvt_pk_bf16_f32 v7, v20, v7
v_accvgpr_read_b32 v20, a24
v_accvgpr_read_b32 v17, a25
v_cvt_pk_bf16_f32 v20, v20, v17
v_accvgpr_read_b32 v24, a26
v_accvgpr_read_b32 v17, a27
v_cvt_pk_bf16_f32 v21, v24, v17
v_accvgpr_read_b32 v24, a40
v_accvgpr_read_b32 v17, a41
v_cvt_pk_bf16_f32 v24, v24, v17
v_accvgpr_read_b32 v26, a42
v_accvgpr_read_b32 v17, a43
v_cvt_pk_bf16_f32 v25, v26, v17
s_waitcnt lgkmcnt(0)
s_barrier
ds_write_b128 v8, v[0:3]
ds_write_b128 v9, v[4:7]
ds_write_b128 v10, v[18:21]
ds_write_b128 v11, v[22:25]
s_waitcnt lgkmcnt(0)
s_barrier
ds_read_b128 v[0:3], v12
ds_read_b128 v[20:23], v12, offset:256
ds_read_b128 v[24:27], v12, offset:128
ds_read_b128 v[30:33], v12, offset:384
s_waitcnt lgkmcnt(3)
v_mov_b32_e32 v18, v0
v_mov_b32_e32 v19, v1
s_waitcnt lgkmcnt(2)
buffer_store_dwordx4 v[18:21], v13, s[4:7], 0, offen, offset:256
s_waitcnt lgkmcnt(1)
v_mov_b32_e32 v28, v24
v_mov_b32_e32 v29, v25
s_waitcnt lgkmcnt(0)
buffer_store_dwordx4 v[28:31], v14, s[4:7], 0, offen, offset:256
v_mov_b32_e32 v4, v22
v_mov_b32_e32 v5, v23
buffer_store_dwordx4 v[2:5], v15, s[4:7], 0, offen, offset:256
v_mov_b32_e32 v28, v32
v_mov_b32_e32 v29, v33
buffer_store_dwordx4 v[26:29], v16, s[4:7], 0, offen, offset:256
v_accvgpr_read_b32 v0, a28
v_accvgpr_read_b32 v1, a29
v_cvt_pk_bf16_f32 v0, v0, v1
v_accvgpr_read_b32 v2, a30
v_accvgpr_read_b32 v1, a31
v_cvt_pk_bf16_f32 v1, v2, v1
v_accvgpr_read_b32 v2, a32
v_accvgpr_read_b32 v3, a33
v_cvt_pk_bf16_f32 v4, v2, v3
v_accvgpr_read_b32 v2, a34
v_accvgpr_read_b32 v3, a35
v_cvt_pk_bf16_f32 v5, v2, v3
v_accvgpr_read_b32 v2, a36
v_accvgpr_read_b32 v3, a37
v_cvt_pk_bf16_f32 v18, v2, v3
v_accvgpr_read_b32 v2, a38
v_accvgpr_read_b32 v3, a39
v_cvt_pk_bf16_f32 v19, v2, v3
v_accvgpr_read_b32 v2, a44
v_accvgpr_read_b32 v3, a45
v_cvt_pk_bf16_f32 v22, v2, v3
v_accvgpr_read_b32 v2, a46
v_accvgpr_read_b32 v3, a47
v_cvt_pk_bf16_f32 v23, v2, v3
v_accvgpr_read_b32 v2, a48
v_accvgpr_read_b32 v3, a49
v_cvt_pk_bf16_f32 v2, v2, v3
v_accvgpr_read_b32 v6, a50
v_accvgpr_read_b32 v3, a51
v_cvt_pk_bf16_f32 v3, v6, v3
v_accvgpr_read_b32 v6, a52
v_accvgpr_read_b32 v7, a53
v_cvt_pk_bf16_f32 v6, v6, v7
v_accvgpr_read_b32 v20, a54
v_accvgpr_read_b32 v7, a55
v_cvt_pk_bf16_f32 v7, v20, v7
v_accvgpr_read_b32 v20, a56
v_accvgpr_read_b32 v17, a57
v_cvt_pk_bf16_f32 v20, v20, v17
v_accvgpr_read_b32 v24, a58
v_accvgpr_read_b32 v17, a59
v_cvt_pk_bf16_f32 v21, v24, v17
v_accvgpr_read_b32 v24, a60
v_accvgpr_read_b32 v17, a61
v_cvt_pk_bf16_f32 v24, v24, v17
v_accvgpr_read_b32 v26, a62
v_accvgpr_read_b32 v17, a63
v_cvt_pk_bf16_f32 v25, v26, v17
s_waitcnt lgkmcnt(0)
s_barrier
ds_write_b128 v8, v[0:3]
ds_write_b128 v9, v[4:7]
ds_write_b128 v10, v[18:21]
ds_write_b128 v11, v[22:25]
s_waitcnt lgkmcnt(0)
s_barrier
ds_read_b128 v[0:3], v12
ds_read_b128 v[6:9], v12, offset:256
ds_read_b128 v[18:21], v12, offset:128
ds_read_b128 v[24:27], v12, offset:384
s_waitcnt lgkmcnt(3)
v_mov_b32_e32 v4, v0
v_mov_b32_e32 v5, v1
s_waitcnt lgkmcnt(2)
buffer_store_dwordx4 v[4:7], v13, s[0:3], 0, offen, offset:256
s_waitcnt lgkmcnt(1)
v_mov_b32_e32 v22, v18
v_mov_b32_e32 v23, v19
s_waitcnt lgkmcnt(0)
buffer_store_dwordx4 v[22:25], v14, s[0:3], 0, offen, offset:256
v_mov_b32_e32 v4, v8
v_mov_b32_e32 v5, v9
buffer_store_dwordx4 v[2:5], v15, s[0:3], 0, offen, offset:256
v_mov_b32_e32 v22, v26
v_mov_b32_e32 v23, v27
buffer_store_dwordx4 v[20:23], v16, s[0:3], 0, offen, offset:256
s_endpgm
	s_endpgm
.Ltmp6:
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel mxfp4_gluon_asm_kernel
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 344
		.amdhsa_user_sgpr_count 16
		.amdhsa_user_sgpr_dispatch_ptr 1
		.amdhsa_user_sgpr_queue_ptr 1
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 1
		.amdhsa_user_sgpr_kernarg_preload_length 8
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 1
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 2
		.amdhsa_next_free_vgpr 512
		.amdhsa_next_free_sgpr 69
		.amdhsa_accum_offset 256
		.amdhsa_reserve_vcc 1
		.amdhsa_reserve_xnack_mask 1
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
.Lfunc_end0:
	.size	mxfp4_gluon_asm_kernel, .Lfunc_end0-mxfp4_gluon_asm_kernel
	.cfi_endproc
                                        ; -- End function
	.set mxfp4_gluon_asm_kernel.num_vgpr, 245
	.set mxfp4_gluon_asm_kernel.num_agpr, 256
	.set mxfp4_gluon_asm_kernel.numbered_sgpr, 69
	.set mxfp4_gluon_asm_kernel.num_named_barrier, 0
	.set mxfp4_gluon_asm_kernel.private_seg_size, 0
	.set mxfp4_gluon_asm_kernel.uses_vcc, 1
	.set mxfp4_gluon_asm_kernel.uses_flat_scratch, 0
	.set mxfp4_gluon_asm_kernel.has_dyn_sized_stack, 0
	.set mxfp4_gluon_asm_kernel.has_recursion, 0
	.set mxfp4_gluon_asm_kernel.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 20148
; TotalNumSgprs: 75
; NumVgprs: 245
; NumAgprs: 256
; TotalNumVgprs: 504
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 9
; VGPRBlocks: 62
; NumSGPRsForWavesPerEU: 75
; NumVGPRsForWavesPerEU: 504
; AccumOffset: 248
; Occupancy: 1
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 16
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 2
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 61
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.text
	.section	.debug_abbrev,"",@progbits
	.byte	1                               ; Abbreviation Code
	.byte	17                              ; DW_TAG_compile_unit
	.byte	1                               ; DW_CHILDREN_yes
	.byte	37                              ; DW_AT_producer
	.byte	14                              ; DW_FORM_strp
	.byte	19                              ; DW_AT_language
	.byte	5                               ; DW_FORM_data2
	.byte	3                               ; DW_AT_name
	.byte	14                              ; DW_FORM_strp
	.byte	16                              ; DW_AT_stmt_list
	.byte	23                              ; DW_FORM_sec_offset
	.byte	27                              ; DW_AT_comp_dir
	.byte	14                              ; DW_FORM_strp
	.byte	17                              ; DW_AT_low_pc
	.byte	1                               ; DW_FORM_addr
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	2                               ; Abbreviation Code
	.byte	46                              ; DW_TAG_subprogram
	.byte	0                               ; DW_CHILDREN_no
	.byte	3                               ; DW_AT_name
	.byte	14                              ; DW_FORM_strp
	.byte	32                              ; DW_AT_inline
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	3                               ; Abbreviation Code
	.byte	46                              ; DW_TAG_subprogram
	.byte	1                               ; DW_CHILDREN_yes
	.byte	17                              ; DW_AT_low_pc
	.byte	1                               ; DW_FORM_addr
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	4                               ; Abbreviation Code
	.byte	29                              ; DW_TAG_inlined_subroutine
	.byte	1                               ; DW_CHILDREN_yes
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	17                              ; DW_AT_low_pc
	.byte	1                               ; DW_FORM_addr
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	88                              ; DW_AT_call_file
	.byte	11                              ; DW_FORM_data1
	.byte	89                              ; DW_AT_call_line
	.byte	11                              ; DW_FORM_data1
	.byte	87                              ; DW_AT_call_column
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	5                               ; Abbreviation Code
	.byte	29                              ; DW_TAG_inlined_subroutine
	.byte	0                               ; DW_CHILDREN_no
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	17                              ; DW_AT_low_pc
	.byte	1                               ; DW_FORM_addr
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	88                              ; DW_AT_call_file
	.byte	11                              ; DW_FORM_data1
	.byte	89                              ; DW_AT_call_line
	.byte	11                              ; DW_FORM_data1
	.byte	87                              ; DW_AT_call_column
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	0                               ; EOM(3)
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"triton"                        ; string offset=0 ; triton
.Linfo_string1:
	.asciz	"matmul_kernel.py"              ; string offset=7 ; matmul_kernel.py
.Linfo_string2:
	.asciz	"/shared_nfs/kyle/gfx9-gluon-tutorials/kernels/gemm/a4w4" ; string offset=24 ; /shared_nfs/kyle/gfx9-gluon-tutorials/kernels/gemm/a4w4
.Linfo_string3:
	.asciz	"mxfp4_gluon_asm_kernel"                   ; string offset=80 ; mxfp4_gluon_asm_kernel
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     256
    .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
      - .offset:         40
        .size:           4
        .value_kind:     by_value
      - .offset:         44
        .size:           4
        .value_kind:     by_value
      - .offset:         48
        .size:           4
        .value_kind:     by_value
      - .offset:         52
        .size:           4
        .value_kind:     by_value
      - .offset:         56
        .size:           4
        .value_kind:     by_value
      - .offset:         60
        .size:           4
        .value_kind:     by_value
      - .offset:         64
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         72
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         80
        .size:           8
        .value_kind:     global_buffer
      - .offset:         88
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         92
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         96
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         100
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         102
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         104
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         106
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         108
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         110
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         128
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         136
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         144
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         152
        .size:           2
        .value_kind:     hidden_grid_dims
      - .offset:         168
        .size:           8
        .value_kind:     hidden_hostcall_buffer
      - .offset:         176
        .size:           8
        .value_kind:     hidden_multigrid_sync_arg
      - .offset:         184
        .size:           8
        .value_kind:     hidden_heap_v1
      - .offset:         192
        .size:           8
        .value_kind:     hidden_default_queue
      - .offset:         200
        .size:           8
        .value_kind:     hidden_completion_action
      - .offset:         208
        .size:           4
        .value_kind:     hidden_dynamic_lds_size
      - .offset:         288
        .size:           8
        .value_kind:     hidden_queue_ptr
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 344
    .max_flat_workgroup_size: 256
    .name:           mxfp4_gluon_asm_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     75
    .sgpr_spill_count: 0
    .symbol:         mxfp4_gluon_asm_kernel.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count: 512
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
	.section	.debug_line,"",@progbits
.Lline_table_start0: