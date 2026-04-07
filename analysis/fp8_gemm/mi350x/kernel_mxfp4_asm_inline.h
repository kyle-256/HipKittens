// 2227 lines of inline GCN assembly
#define MXFP4_KERNEL_ASM_BODY \
    "s_branch .Lmx_0\n" \
    ".Lmx_0:\n" \
    "s_load_dwordx8 s[20:27], s[4:5], 0x20\n" \
    "s_load_dword s63, s[4:5], 0x40\n" \
    "v_and_b32_e32 v36, 0x3ff, v0\n" \
    "s_nop 0\n" \
    "v_readfirstlane_b32 s0, v36\n" \
    "s_bfe_u32 s5, s0, 0x20006\n" \
    "s_waitcnt lgkmcnt(0)\n" \
    "s_add_i32 s0, s22, 0xff\n" \
    "s_ashr_i32 s1, s0, 31\n" \
    "s_lshr_b32 s1, s1, 24\n" \
    "s_add_i32 s0, s0, s1\n" \
    "s_ashr_i32 s0, s0, 8\n" \
    "s_add_i32 s1, s23, 0xff\n" \
    "s_ashr_i32 s2, s1, 31\n" \
    "s_lshr_b32 s2, s2, 24\n" \
    "s_add_i32 s1, s1, s2\n" \
    "s_ashr_i32 s1, s1, 8\n" \
    "s_ashr_i32 s2, s16, 31\n" \
    "s_lshr_b32 s2, s2, 29\n" \
    "s_add_i32 s2, s16, s2\n" \
    "s_ashr_i32 s2, s2, 3\n" \
    "s_lshl_b32 s3, s16, 7\n" \
    "s_mulk_i32 s2, 0xfc01\n" \
    "s_add_i32 s2, s2, s3\n" \
    "s_lshl_b32 s1, s1, 2\n" \
    "s_xor_b32 s3, s2, s1\n" \
    "s_ashr_i32 s3, s3, 31\n" \
    "s_abs_i32 s4, s2\n" \
    "s_abs_i32 s6, s1\n" \
    "v_cvt_f32_u32_e32 v1, s6\n" \
    "v_rcp_iflag_f32_e32 v1, v1\n" \
    "s_nop 0\n" \
    "v_mul_f32_e32 v1, 0x4f7ffffe, v1\n" \
    "v_cvt_u32_f32_e32 v1, v1\n" \
    "s_sub_i32 s7, 0, s6\n" \
    "v_readfirstlane_b32 s16, v1\n" \
    "s_mul_i32 s7, s7, s16\n" \
    "s_mul_hi_u32 s7, s16, s7\n" \
    "s_add_i32 s16, s16, s7\n" \
    "s_mul_hi_u32 s7, s4, s16\n" \
    "s_mul_i32 s16, s7, s6\n" \
    "s_sub_i32 s4, s4, s16\n" \
    "s_add_i32 s16, s7, 1\n" \
    "s_sub_i32 s17, s4, s6\n" \
    "s_cmp_ge_u32 s4, s6\n" \
    "s_cselect_b32 s7, s16, s7\n" \
    "s_cselect_b32 s4, s17, s4\n" \
    "s_add_i32 s16, s7, 1\n" \
    "s_cmp_ge_u32 s4, s6\n" \
    "s_cselect_b32 s4, s16, s7\n" \
    "s_xor_b32 s4, s4, s3\n" \
    "s_sub_i32 s3, s4, s3\n" \
    "s_lshl_b32 s4, s3, 2\n" \
    "s_sub_i32 s0, s0, s4\n" \
    "s_min_i32 s0, s0, 4\n" \
    "s_mul_i32 s3, s3, s1\n" \
    "s_sub_i32 s1, s2, s3\n" \
    "s_xor_b32 s2, s1, s0\n" \
    "s_ashr_i32 s2, s2, 31\n" \
    "s_abs_i32 s3, s1\n" \
    "s_abs_i32 s6, s0\n" \
    "v_cvt_f32_u32_e32 v1, s6\n" \
    "v_rcp_iflag_f32_e32 v1, v1\n" \
    "s_nop 0\n" \
    "v_mul_f32_e32 v1, 0x4f7ffffe, v1\n" \
    "v_cvt_u32_f32_e32 v1, v1\n" \
    "s_sub_i32 s7, 0, s6\n" \
    "v_readfirstlane_b32 s16, v1\n" \
    "s_mul_i32 s7, s7, s16\n" \
    "s_mul_hi_u32 s7, s16, s7\n" \
    "s_add_i32 s16, s16, s7\n" \
    "s_mul_hi_u32 s7, s3, s16\n" \
    "s_mul_i32 s16, s7, s6\n" \
    "s_sub_i32 s3, s3, s16\n" \
    "s_add_i32 s16, s7, 1\n" \
    "s_sub_i32 s17, s3, s6\n" \
    "s_cmp_ge_u32 s3, s6\n" \
    "s_cselect_b32 s7, s16, s7\n" \
    "s_cselect_b32 s3, s17, s3\n" \
    "s_add_i32 s16, s7, 1\n" \
    "s_cmp_ge_u32 s3, s6\n" \
    "s_cselect_b32 s3, s16, s7\n" \
    "s_xor_b32 s3, s3, s2\n" \
    "s_sub_i32 s6, s3, s2\n" \
    "s_mul_i32 s0, s6, s0\n" \
    "s_sub_i32 s67, s1, s0\n" \
    "s_add_i32 s67, s67, s4\n" \
    "s_lshl_b32 s65, s5, 6\n" \
    "v_and_or_b32 v37, v0, 63, s65\n" \
    "v_lshlrev_b32_e32 v1, 1, v36\n" \
    "v_and_b32_e32 v1, 0x70, v1\n" \
    "v_or_b32_e32 v1, s5, v1\n" \
    "v_or_b32_e32 v20, 4, v1\n" \
    "v_or_b32_e32 v22, 8, v1\n" \
    "v_or_b32_e32 v25, 12, v1\n" \
    "v_or_b32_e32 v10, 0x80, v1\n" \
    "v_or_b32_e32 v12, 0x84, v1\n" \
    "v_or_b32_e32 v14, 0x88, v1\n" \
    "v_or_b32_e32 v16, 0x8c, v1\n" \
    "v_lshlrev_b32_e32 v2, 4, v36\n" \
    "v_and_b32_e32 v24, 0x70, v2\n" \
    "v_mad_u64_u32 v[2:3], s[0:1], v1, s24, v[24:25]\n" \
    "v_mad_u64_u32 v[4:5], s[0:1], v20, s24, v[24:25]\n" \
    "v_mad_u64_u32 v[6:7], s[0:1], v22, s24, v[24:25]\n" \
    "v_mad_u64_u32 v[8:9], s[0:1], v25, s24, v[24:25]\n" \
    "v_mad_u64_u32 v[10:11], s[0:1], v10, s24, v[24:25]\n" \
    "v_mad_u64_u32 v[12:13], s[0:1], v12, s24, v[24:25]\n" \
    "v_mad_u64_u32 v[14:15], s[0:1], v14, s24, v[24:25]\n" \
    "v_mad_u64_u32 v[16:17], s[0:1], v16, s24, v[24:25]\n" \
    "v_add_u32_e32 v3, 0x80, v2\n" \
    "v_add_u32_e32 v5, 0x80, v4\n" \
    "v_add_u32_e32 v7, 0x80, v6\n" \
    "v_add_u32_e32 v9, 0x80, v8\n" \
    "v_add_u32_e32 v11, 0x80, v10\n" \
    "v_add_u32_e32 v13, 0x80, v12\n" \
    "v_add_u32_e32 v15, 0x80, v14\n" \
    "v_add_u32_e32 v17, 0x80, v16\n" \
    "s_lshl_b32 s33, s67, 8\n" \
    "s_mul_i32 s0, s33, s24\n" \
    "s_ashr_i32 s1, s0, 31\n" \
    "s_add_u32 s0, s8, s0\n" \
    "s_addc_u32 s1, s9, s1\n" \
    "v_mad_u64_u32 v[18:19], s[2:3], v1, s25, v[24:25]\n" \
    "v_mad_u64_u32 v[20:21], s[2:3], v20, s25, v[24:25]\n" \
    "v_mad_u64_u32 v[22:23], s[2:3], v22, s25, v[24:25]\n" \
    "v_mad_u64_u32 v[24:25], s[2:3], v25, s25, v[24:25]\n" \
    "s_lshl_b32 s2, s25, 7\n" \
    "s_nop 0\n" \
    "v_add_u32_e32 v19, s2, v18\n" \
    "v_add_u32_e32 v21, s2, v20\n" \
    "v_add_u32_e32 v23, s2, v22\n" \
    "v_add_u32_e32 v25, s2, v24\n" \
    "v_add_u32_e32 v38, 0x80, v18\n" \
    "v_add_u32_e32 v39, 0x80, v20\n" \
    "v_add_u32_e32 v40, 0x80, v22\n" \
    "v_add_u32_e32 v41, 0x80, v24\n" \
    "v_add_u32_e32 v42, 0x80, v19\n" \
    "v_add_u32_e32 v43, 0x80, v21\n" \
    "v_add_u32_e32 v44, 0x80, v23\n" \
    "v_add_u32_e32 v45, 0x80, v25\n" \
    "s_lshl_b32 s34, s6, 8\n" \
    "s_mul_i32 s2, s34, s25\n" \
    "s_ashr_i32 s3, s2, 31\n" \
    "s_add_u32 s4, s10, s2\n" \
    "s_addc_u32 s10, s11, s3\n" \
    "v_lshrrev_b32_e32 v1, 5, v37\n" \
    "v_and_b32_e32 v28, 31, v0\n" \
    "v_lshl_or_b32 v26, v28, 3, s33\n" \
    "s_bfe_i32 s2, s67, 0x10017\n" \
    "v_add_u32_e32 v26, s2, v26\n" \
    "v_xor_b32_e32 v26, s2, v26\n" \
    "s_abs_i32 s3, s22\n" \
    "v_cvt_f32_u32_e32 v27, s3\n" \
    "v_rcp_iflag_f32_e32 v27, v27\n" \
    "s_nop 0\n" \
    "v_mul_f32_e32 v27, 0x4f7ffffe, v27\n" \
    "v_cvt_u32_f32_e32 v27, v27\n" \
    "s_sub_i32 s7, 0, s3\n" \
    "v_mul_lo_u32 v29, s7, v27\n" \
    "v_mul_hi_u32 v29, v27, v29\n" \
    "v_add_u32_e32 v27, v27, v29\n" \
    "v_mul_hi_u32 v27, v26, v27\n" \
    "v_mul_lo_u32 v27, v27, s3\n" \
    "v_sub_u32_e32 v26, v26, v27\n" \
    "v_cmp_le_u32_e32 vcc, s3, v26\n" \
    "v_subrev_u32_e32 v27, s3, v26\n" \
    "s_nop 0\n" \
    "v_cndmask_b32_e32 v26, v26, v27, vcc\n" \
    "v_cmp_le_u32_e32 vcc, s3, v26\n" \
    "v_subrev_u32_e32 v27, s3, v26\n" \
    "s_nop 0\n" \
    "v_cndmask_b32_e32 v26, v26, v27, vcc\n" \
    "v_xor_b32_e32 v26, s2, v26\n" \
    "v_subrev_u32_e32 v26, s2, v26\n" \
    "v_mad_u64_u32 v[26:27], s[2:3], v1, s27, v[26:27]\n" \
    "v_lshl_add_u32 v27, s27, 3, v26\n" \
    "v_lshl_or_b32 v28, v28, 2, s34\n" \
    "s_bfe_i32 s2, s6, 0x10017\n" \
    "v_add_u32_e32 v28, s2, v28\n" \
    "v_xor_b32_e32 v28, s2, v28\n" \
    "s_abs_i32 s3, s23\n" \
    "v_cvt_f32_u32_e32 v29, s3\n" \
    "v_rcp_iflag_f32_e32 v29, v29\n" \
    "s_nop 0\n" \
    "v_mul_f32_e32 v29, 0x4f7ffffe, v29\n" \
    "v_cvt_u32_f32_e32 v29, v29\n" \
    "s_sub_i32 s6, 0, s3\n" \
    "v_mul_lo_u32 v30, s6, v29\n" \
    "v_mul_hi_u32 v30, v29, v30\n" \
    "v_add_u32_e32 v29, v29, v30\n" \
    "v_mul_hi_u32 v29, v28, v29\n" \
    "v_mul_lo_u32 v29, v29, s3\n" \
    "v_sub_u32_e32 v28, v28, v29\n" \
    "v_cmp_le_u32_e32 vcc, s3, v28\n" \
    "v_subrev_u32_e32 v29, s3, v28\n" \
    "s_nop 0\n" \
    "v_cndmask_b32_e32 v28, v28, v29, vcc\n" \
    "v_cmp_le_u32_e32 vcc, s3, v28\n" \
    "v_subrev_u32_e32 v29, s3, v28\n" \
    "s_nop 0\n" \
    "v_cndmask_b32_e32 v28, v28, v29, vcc\n" \
    "v_xor_b32_e32 v28, s2, v28\n" \
    "v_subrev_u32_e32 v28, s2, v28\n" \
    "v_mad_u64_u32 v[28:29], s[2:3], v1, s63, v[28:29]\n" \
    "v_add_u32_e32 v47, 0x80, v28\n" \
    "s_lshl_b32 s2, s63, 3\n" \
    "v_add_u32_e32 v48, s2, v28\n" \
    "v_add_u32_e32 v49, s2, v47\n" \
    "s_and_b32 s1, s1, 0xffff\n" \
    "s_mov_b32 s3, 0x27000\n" \
    "s_mov_b32 s2, 0x7ffffffe\n" \
    "s_mul_i32 s11, s5, 0x420\n" \
    "s_add_i32 s25, s11, 0\n" \
    "s_mov_b32 m0, s25\n" \
    "s_nop 0\n" \
    "buffer_load_dwordx4 v2, s[0:3], 0, offen, lds\n" \
    "s_add_i32 s35, s25, 0x1080\n" \
    "s_mov_b32 m0, s35\n" \
    "s_nop 0\n" \
    "buffer_load_dwordx4 v4, s[0:3], 0, offen, lds\n" \
    "s_add_i32 s36, s25, 0x2100\n" \
    "s_mov_b32 m0, s36\n" \
    "s_nop 0\n" \
    "buffer_load_dwordx4 v6, s[0:3], 0, offen, lds\n" \
    "s_add_i32 s37, s25, 0x3180\n" \
    "s_mov_b32 m0, s37\n" \
    "s_nop 0\n" \
    "buffer_load_dwordx4 v8, s[0:3], 0, offen, lds\n" \
    "s_add_i32 s38, s25, 0x4200\n" \
    "s_mov_b32 m0, s38\n" \
    "s_nop 0\n" \
    "buffer_load_dwordx4 v10, s[0:3], 0, offen, lds\n" \
    "s_add_i32 s39, s25, 0x5280\n" \
    "s_mov_b32 m0, s39\n" \
    "s_nop 0\n" \
    "buffer_load_dwordx4 v12, s[0:3], 0, offen, lds\n" \
    "s_add_i32 s40, s25, 0x6300\n" \
    "s_mov_b32 m0, s40\n" \
    "s_nop 0\n" \
    "buffer_load_dwordx4 v14, s[0:3], 0, offen, lds\n" \
    "s_add_i32 s41, s25, 0x7380\n" \
    "s_mov_b32 m0, s41\n" \
    "s_nop 0\n" \
    "buffer_load_dwordx4 v16, s[0:3], 0, offen, lds\n" \
    "s_and_b32 s5, s10, 0xffff\n" \
    "s_mov_b32 s6, s2\n" \
    "s_mov_b32 s7, s3\n" \
    "s_add_i32 s22, 0, 0x107e0\n" \
    "s_add_i32 s42, s22, s11\n" \
    "s_mov_b32 m0, s42\n" \
    "s_nop 0\n" \
    "buffer_load_dwordx4 v18, s[4:7], 0, offen, lds\n" \
    "s_add_i32 s43, s42, 0x1080\n" \
    "s_mov_b32 m0, s43\n" \
    "s_nop 0\n" \
    "buffer_load_dwordx4 v20, s[4:7], 0, offen, lds\n" \
    "s_add_i32 s44, s42, 0x2100\n" \
    "s_mov_b32 m0, s44\n" \
    "s_nop 0\n" \
    "buffer_load_dwordx4 v22, s[4:7], 0, offen, lds\n" \
    "s_add_i32 s45, s42, 0x3180\n" \
    "s_mov_b32 m0, s45\n" \
    "s_nop 0\n" \
    "buffer_load_dwordx4 v24, s[4:7], 0, offen, lds\n" \
    "s_and_b32 s29, s15, 0xffff\n" \
    "s_mov_b32 s28, s14\n" \
    "s_mov_b32 s30, s2\n" \
    "s_mov_b32 s31, s3\n" \
    "buffer_load_dwordx2 v[32:33], v26, s[28:31], 0, offen\n" \
    "s_and_b32 s17, s21, 0xffff\n" \
    "s_mov_b32 s16, s20\n" \
    "s_mov_b32 s18, s2\n" \
    "s_mov_b32 s19, s3\n" \
    "buffer_load_dword v34, v28, s[16:19], 0, offen\n" \
    "s_add_i32 s46, s25, 0x18bc0\n" \
    "s_mov_b32 m0, s46\n" \
    "s_nop 0\n" \
    "buffer_load_dwordx4 v19, s[4:7], 0, offen, lds\n" \
    "s_add_i32 s47, s25, 0x19c40\n" \
    "s_mov_b32 m0, s47\n" \
    "s_nop 0\n" \
    "buffer_load_dwordx4 v21, s[4:7], 0, offen, lds\n" \
    "s_add_i32 s48, s25, 0x1acc0\n" \
    "s_mov_b32 m0, s48\n" \
    "s_nop 0\n" \
    "buffer_load_dwordx4 v23, s[4:7], 0, offen, lds\n" \
    "s_add_i32 s49, s25, 0x1bd40\n" \
    "s_mov_b32 m0, s49\n" \
    "s_nop 0\n" \
    "buffer_load_dwordx4 v25, s[4:7], 0, offen, lds\n" \
    "buffer_load_dword v162, v28, s[16:19], 0, offen, offset:128\n" \
    "s_waitcnt lgkmcnt(0)\n" \
    "s_barrier\n" \
    "s_add_i32 s50, s25, 0x8400\n" \
    "s_mov_b32 m0, s50\n" \
    "s_nop 0\n" \
    "buffer_load_dwordx4 v3, s[0:3], 0, offen, lds\n" \
    "s_add_i32 s51, s25, 0x9480\n" \
    "s_mov_b32 m0, s51\n" \
    "s_nop 0\n" \
    "buffer_load_dwordx4 v5, s[0:3], 0, offen, lds\n" \
    "s_add_i32 s52, s25, 0xa500\n" \
    "s_mov_b32 m0, s52\n" \
    "s_nop 0\n" \
    "buffer_load_dwordx4 v7, s[0:3], 0, offen, lds\n" \
    "s_add_i32 s53, s25, 0xb580\n" \
    "s_mov_b32 m0, s53\n" \
    "s_nop 0\n" \
    "buffer_load_dwordx4 v9, s[0:3], 0, offen, lds\n" \
    "s_add_i32 s54, s25, 0xc600\n" \
    "s_mov_b32 m0, s54\n" \
    "s_nop 0\n" \
    "buffer_load_dwordx4 v11, s[0:3], 0, offen, lds\n" \
    "s_add_i32 s55, s25, 0xd680\n" \
    "s_mov_b32 m0, s55\n" \
    "s_nop 0\n" \
    "buffer_load_dwordx4 v13, s[0:3], 0, offen, lds\n" \
    "s_add_i32 s56, s25, 0xe700\n" \
    "s_mov_b32 m0, s56\n" \
    "s_nop 0\n" \
    "buffer_load_dwordx4 v15, s[0:3], 0, offen, lds\n" \
    "s_add_i32 s57, s25, 0xf780\n" \
    "s_mov_b32 m0, s57\n" \
    "s_nop 0\n" \
    "buffer_load_dwordx4 v17, s[0:3], 0, offen, lds\n" \
    "s_add_i32 s58, s25, 0x149e0\n" \
    "s_mov_b32 m0, s58\n" \
    "s_nop 0\n" \
    "buffer_load_dwordx4 v38, s[4:7], 0, offen, lds\n" \
    "s_add_i32 s59, s25, 0x15a60\n" \
    "s_mov_b32 m0, s59\n" \
    "s_nop 0\n" \
    "buffer_load_dwordx4 v39, s[4:7], 0, offen, lds\n" \
    "s_add_i32 s60, s25, 0x16ae0\n" \
    "s_mov_b32 m0, s60\n" \
    "s_nop 0\n" \
    "buffer_load_dwordx4 v40, s[4:7], 0, offen, lds\n" \
    "s_add_i32 s61, s25, 0x17b60\n" \
    "s_mov_b32 m0, s61\n" \
    "s_nop 0\n" \
    "buffer_load_dwordx4 v41, s[4:7], 0, offen, lds\n" \
    "buffer_load_dwordx2 v[156:157], v27, s[28:31], 0, offen\n" \
    "buffer_load_dword v160, v48, s[16:19], 0, offen\n" \
    "s_add_i32 s28, s25, 0x1cdc0\n" \
    "s_mov_b32 m0, s28\n" \
    "s_nop 0\n" \
    "buffer_load_dwordx4 v42, s[4:7], 0, offen, lds\n" \
    "s_add_i32 s29, s25, 0x1de40\n" \
    "s_mov_b32 m0, s29\n" \
    "s_nop 0\n" \
    "buffer_load_dwordx4 v43, s[4:7], 0, offen, lds\n" \
    "s_add_i32 s30, s25, 0x1eec0\n" \
    "s_mov_b32 m0, s30\n" \
    "s_nop 0\n" \
    "buffer_load_dwordx4 v44, s[4:7], 0, offen, lds\n" \
    "s_add_i32 s31, s25, 0x1ff40\n" \
    "s_mov_b32 m0, s31\n" \
    "s_nop 0\n" \
    "buffer_load_dwordx4 v45, s[4:7], 0, offen, lds\n" \
    "buffer_load_dword v161, v49, s[16:19], 0, offen\n" \
    "s_lshl_b32 s27, s27, 4\n" \
    "s_ashr_i32 s62, s27, 31\n" \
    "s_lshl_b32 s63, s63, 4\n" \
    "s_ashr_i32 s64, s63, 31\n" \
    "s_waitcnt vmcnt(26), lgkmcnt(0)\n" \
    "s_barrier\n" \
    "v_and_b32_e32 v52, 15, v0\n" \
    "v_lshlrev_b32_e32 v1, 10, v52\n" \
    "s_movk_i32 s0, 0xb0\n" \
    "v_and_or_b32 v29, v37, s0, v1\n" \
    "v_lshlrev_b32_e32 v30, 5, v52\n" \
    "v_add3_u32 v46, v29, v30, 0\n" \
    "ds_read_b128 v[116:119], v46\n" \
    "ds_read_b128 v[120:123], v46, offset:64\n" \
    "ds_read_b128 v[108:111], v46, offset:256\n" \
    "ds_read_b128 v[112:115], v46, offset:320\n" \
    "ds_read_b128 v[100:103], v46, offset:512\n" \
    "ds_read_b128 v[104:107], v46, offset:576\n" \
    "ds_read_b128 v[88:91], v46, offset:768\n" \
    "ds_read_b128 v[96:99], v46, offset:832\n" \
    "ds_read_b128 v[84:87], v46, offset:16896\n" \
    "ds_read_b128 v[92:95], v46, offset:16960\n" \
    "ds_read_b128 v[76:79], v46, offset:17152\n" \
    "ds_read_b128 v[80:83], v46, offset:17216\n" \
    "ds_read_b128 v[68:71], v46, offset:17408\n" \
    "ds_read_b128 v[72:75], v46, offset:17472\n" \
    "ds_read_b128 v[60:63], v46, offset:17664\n" \
    "ds_read_b128 v[64:67], v46, offset:17728\n" \
    "v_and_b32_e32 v31, 48, v0\n" \
    "s_and_b32 s0, s65, 64\n" \
    "v_or_b32_e32 v1, v1, v31\n" \
    "v_add_u32_e32 v1, v1, v30\n" \
    "v_lshl_add_u32 v53, s0, 1, v1\n" \
    "v_add_u32_e32 v1, s22, v53\n" \
    "ds_read_b128 v[124:127], v1\n" \
    "ds_read_b128 v[132:135], v1, offset:64\n" \
    "ds_read_b128 v[128:131], v1, offset:256\n" \
    "ds_read_b128 v[136:139], v1, offset:320\n" \
    "ds_read_b128 v[140:143], v1, offset:512\n" \
    "ds_read_b128 v[144:147], v1, offset:576\n" \
    "ds_read_b128 v[148:151], v1, offset:768\n" \
    "ds_read_b128 v[152:155], v1, offset:832\n" \
    "v_lshlrev_b32_e32 v54, 3, v37\n" \
    "s_add_i32 s1, 0, 0x20fa0\n" \
    "v_add_u32_e32 v1, s1, v54\n" \
    "s_waitcnt vmcnt(25)\n" \
    "ds_write_b64 v1, v[32:33]\n" \
    "v_lshlrev_b32_e32 v55, 2, v37\n" \
    "s_add_i32 s5, 0, 0x217a0\n" \
    "v_add_u32_e32 v1, s5, v55\n" \
    "s_waitcnt vmcnt(24)\n" \
    "ds_write_b32 v1, v34\n" \
    "s_waitcnt lgkmcnt(0)\n" \
    "s_barrier\n" \
    "v_lshlrev_b32_e32 v29, 3, v36\n" \
    "v_and_b32_e32 v1, 0x68, v29\n" \
    "v_and_b32_e32 v32, 2, v0\n" \
    "s_and_b32 s22, s65, 0x80\n" \
    "s_lshr_b32 s6, s22, 3\n" \
    "s_add_i32 s6, s6, s1\n" \
    "v_lshl_add_u32 v0, v32, 9, s6\n" \
    "v_lshl_add_u32 v0, v31, 4, v0\n" \
    "v_add_u32_e32 v51, v0, v1\n" \
    "ds_read_b64_tr_b8 v[30:31], v51\n" \
    "ds_read_b64_tr_b8 v[0:1], v51, offset:128\n" \
    "v_and_b32_e32 v33, 0x1e8, v29\n" \
    "v_lshlrev_b32_e32 v32, 8, v32\n" \
    "s_lshr_b32 s23, s0, 2\n" \
    "v_add3_u32 v32, s5, v33, v32\n" \
    "v_add_u32_e32 v50, s23, v32\n" \
    "ds_read_b64_tr_b8 v[32:33], v50\n" \
    "s_add_u32 s65, s4, 0x100\n" \
    "s_addc_u32 s66, s10, 0\n" \
    "s_mul_i32 s0, s24, s67\n" \
    "s_lshl_b32 s0, s0, 8\n" \
    "s_ashr_i32 s1, s0, 31\n" \
    "s_add_u32 s0, s8, s0\n" \
    "s_addc_u32 s1, s9, s1\n" \
    "s_add_u32 s24, s0, 0x100\n" \
    "s_addc_u32 s67, s1, 0\n" \
    "s_add_u32 s20, s20, s63\n" \
    "s_addc_u32 s21, s21, s64\n" \
    "s_add_u32 s14, s14, s27\n" \
    "s_addc_u32 s15, s15, s62\n" \
    "v_mov_b32_e32 v34, 0\n" \
    "s_mov_b32 s68, -2\n" \
    "v_accvgpr_write_b32 a124, v34\n" \
    "v_accvgpr_write_b32 a125, v34\n" \
    "v_accvgpr_write_b32 a126, v34\n" \
    "v_accvgpr_write_b32 a127, v34\n" \
    "v_accvgpr_write_b32 a0, v34\n" \
    "v_accvgpr_write_b32 a1, v34\n" \
    "v_accvgpr_write_b32 a2, v34\n" \
    "v_accvgpr_write_b32 a3, v34\n" \
    "v_accvgpr_write_b32 a4, v34\n" \
    "v_accvgpr_write_b32 a5, v34\n" \
    "v_accvgpr_write_b32 a6, v34\n" \
    "v_accvgpr_write_b32 a7, v34\n" \
    "v_accvgpr_write_b32 a8, v34\n" \
    "v_accvgpr_write_b32 a9, v34\n" \
    "v_accvgpr_write_b32 a10, v34\n" \
    "v_accvgpr_write_b32 a11, v34\n" \
    "v_accvgpr_write_b32 a12, v34\n" \
    "v_accvgpr_write_b32 a13, v34\n" \
    "v_accvgpr_write_b32 a14, v34\n" \
    "v_accvgpr_write_b32 a15, v34\n" \
    "v_accvgpr_write_b32 a16, v34\n" \
    "v_accvgpr_write_b32 a17, v34\n" \
    "v_accvgpr_write_b32 a18, v34\n" \
    "v_accvgpr_write_b32 a19, v34\n" \
    "v_accvgpr_write_b32 a20, v34\n" \
    "v_accvgpr_write_b32 a21, v34\n" \
    "v_accvgpr_write_b32 a22, v34\n" \
    "v_accvgpr_write_b32 a23, v34\n" \
    "v_accvgpr_write_b32 a24, v34\n" \
    "v_accvgpr_write_b32 a25, v34\n" \
    "v_accvgpr_write_b32 a26, v34\n" \
    "v_accvgpr_write_b32 a27, v34\n" \
    "v_accvgpr_write_b32 a28, v34\n" \
    "v_accvgpr_write_b32 a29, v34\n" \
    "v_accvgpr_write_b32 a30, v34\n" \
    "v_accvgpr_write_b32 a31, v34\n" \
    "v_accvgpr_write_b32 a32, v34\n" \
    "v_accvgpr_write_b32 a33, v34\n" \
    "v_accvgpr_write_b32 a34, v34\n" \
    "v_accvgpr_write_b32 a35, v34\n" \
    "v_accvgpr_write_b32 a36, v34\n" \
    "v_accvgpr_write_b32 a37, v34\n" \
    "v_accvgpr_write_b32 a38, v34\n" \
    "v_accvgpr_write_b32 a39, v34\n" \
    "v_accvgpr_write_b32 a44, v34\n" \
    "v_accvgpr_write_b32 a45, v34\n" \
    "v_accvgpr_write_b32 a46, v34\n" \
    "v_accvgpr_write_b32 a47, v34\n" \
    "v_accvgpr_write_b32 a52, v34\n" \
    "v_accvgpr_write_b32 a53, v34\n" \
    "v_accvgpr_write_b32 a54, v34\n" \
    "v_accvgpr_write_b32 a55, v34\n" \
    "v_accvgpr_write_b32 a40, v34\n" \
    "v_accvgpr_write_b32 a41, v34\n" \
    "v_accvgpr_write_b32 a42, v34\n" \
    "v_accvgpr_write_b32 a43, v34\n" \
    "v_accvgpr_write_b32 a48, v34\n" \
    "v_accvgpr_write_b32 a49, v34\n" \
    "v_accvgpr_write_b32 a50, v34\n" \
    "v_accvgpr_write_b32 a51, v34\n" \
    "v_accvgpr_write_b32 a56, v34\n" \
    "v_accvgpr_write_b32 a57, v34\n" \
    "v_accvgpr_write_b32 a58, v34\n" \
    "v_accvgpr_write_b32 a59, v34\n" \
    "v_accvgpr_write_b32 a60, v34\n" \
    "v_accvgpr_write_b32 a61, v34\n" \
    "v_accvgpr_write_b32 a62, v34\n" \
    "v_accvgpr_write_b32 a63, v34\n" \
    "v_accvgpr_write_b32 a64, v34\n" \
    "v_accvgpr_write_b32 a65, v34\n" \
    "v_accvgpr_write_b32 a66, v34\n" \
    "v_accvgpr_write_b32 a67, v34\n" \
    "v_accvgpr_write_b32 a68, v34\n" \
    "v_accvgpr_write_b32 a69, v34\n" \
    "v_accvgpr_write_b32 a70, v34\n" \
    "v_accvgpr_write_b32 a71, v34\n" \
    "v_accvgpr_write_b32 a72, v34\n" \
    "v_accvgpr_write_b32 a73, v34\n" \
    "v_accvgpr_write_b32 a74, v34\n" \
    "v_accvgpr_write_b32 a75, v34\n" \
    "v_accvgpr_write_b32 a76, v34\n" \
    "v_accvgpr_write_b32 a77, v34\n" \
    "v_accvgpr_write_b32 a78, v34\n" \
    "v_accvgpr_write_b32 a79, v34\n" \
    "v_accvgpr_write_b32 a80, v34\n" \
    "v_accvgpr_write_b32 a81, v34\n" \
    "v_accvgpr_write_b32 a82, v34\n" \
    "v_accvgpr_write_b32 a83, v34\n" \
    "v_accvgpr_write_b32 a84, v34\n" \
    "v_accvgpr_write_b32 a85, v34\n" \
    "v_accvgpr_write_b32 a86, v34\n" \
    "v_accvgpr_write_b32 a87, v34\n" \
    "v_accvgpr_write_b32 a88, v34\n" \
    "v_accvgpr_write_b32 a89, v34\n" \
    "v_accvgpr_write_b32 a90, v34\n" \
    "v_accvgpr_write_b32 a91, v34\n" \
    "v_accvgpr_write_b32 a92, v34\n" \
    "v_accvgpr_write_b32 a93, v34\n" \
    "v_accvgpr_write_b32 a94, v34\n" \
    "v_accvgpr_write_b32 a95, v34\n" \
    "v_accvgpr_write_b32 a96, v34\n" \
    "v_accvgpr_write_b32 a97, v34\n" \
    "v_accvgpr_write_b32 a98, v34\n" \
    "v_accvgpr_write_b32 a99, v34\n" \
    "v_accvgpr_write_b32 a100, v34\n" \
    "v_accvgpr_write_b32 a101, v34\n" \
    "v_accvgpr_write_b32 a102, v34\n" \
    "v_accvgpr_write_b32 a103, v34\n" \
    "v_accvgpr_write_b32 a104, v34\n" \
    "v_accvgpr_write_b32 a105, v34\n" \
    "v_accvgpr_write_b32 a106, v34\n" \
    "v_accvgpr_write_b32 a107, v34\n" \
    "v_accvgpr_write_b32 a108, v34\n" \
    "v_accvgpr_write_b32 a109, v34\n" \
    "v_accvgpr_write_b32 a110, v34\n" \
    "v_accvgpr_write_b32 a111, v34\n" \
    "v_accvgpr_write_b32 a112, v34\n" \
    "v_accvgpr_write_b32 a113, v34\n" \
    "v_accvgpr_write_b32 a114, v34\n" \
    "v_accvgpr_write_b32 a115, v34\n" \
    "v_accvgpr_write_b32 a116, v34\n" \
    "v_accvgpr_write_b32 a117, v34\n" \
    "v_accvgpr_write_b32 a118, v34\n" \
    "v_accvgpr_write_b32 a119, v34\n" \
    "v_accvgpr_write_b32 a120, v34\n" \
    "v_accvgpr_write_b32 a121, v34\n" \
    "v_accvgpr_write_b32 a122, v34\n" \
    "v_accvgpr_write_b32 a123, v34\n" \
    "v_accvgpr_write_b32 a132, v34\n" \
    "v_accvgpr_write_b32 a133, v34\n" \
    "v_accvgpr_write_b32 a134, v34\n" \
    "v_accvgpr_write_b32 a135, v34\n" \
    "v_accvgpr_write_b32 a140, v34\n" \
    "v_accvgpr_write_b32 a141, v34\n" \
    "v_accvgpr_write_b32 a142, v34\n" \
    "v_accvgpr_write_b32 a143, v34\n" \
    "v_accvgpr_write_b32 a144, v34\n" \
    "v_accvgpr_write_b32 a145, v34\n" \
    "v_accvgpr_write_b32 a146, v34\n" \
    "v_accvgpr_write_b32 a147, v34\n" \
    "v_accvgpr_write_b32 a152, v34\n" \
    "v_accvgpr_write_b32 a153, v34\n" \
    "v_accvgpr_write_b32 a154, v34\n" \
    "v_accvgpr_write_b32 a155, v34\n" \
    "v_accvgpr_write_b32 a128, v34\n" \
    "v_accvgpr_write_b32 a129, v34\n" \
    "v_accvgpr_write_b32 a130, v34\n" \
    "v_accvgpr_write_b32 a131, v34\n" \
    "v_accvgpr_write_b32 a136, v34\n" \
    "v_accvgpr_write_b32 a137, v34\n" \
    "v_accvgpr_write_b32 a138, v34\n" \
    "v_accvgpr_write_b32 a139, v34\n" \
    "v_accvgpr_write_b32 a148, v34\n" \
    "v_accvgpr_write_b32 a149, v34\n" \
    "v_accvgpr_write_b32 a150, v34\n" \
    "v_accvgpr_write_b32 a151, v34\n" \
    "v_accvgpr_write_b32 a156, v34\n" \
    "v_accvgpr_write_b32 a157, v34\n" \
    "v_accvgpr_write_b32 a158, v34\n" \
    "v_accvgpr_write_b32 a159, v34\n" \
    "v_accvgpr_write_b32 a160, v34\n" \
    "v_accvgpr_write_b32 a161, v34\n" \
    "v_accvgpr_write_b32 a162, v34\n" \
    "v_accvgpr_write_b32 a163, v34\n" \
    "v_accvgpr_write_b32 a164, v34\n" \
    "v_accvgpr_write_b32 a165, v34\n" \
    "v_accvgpr_write_b32 a166, v34\n" \
    "v_accvgpr_write_b32 a167, v34\n" \
    "v_accvgpr_write_b32 a168, v34\n" \
    "v_accvgpr_write_b32 a169, v34\n" \
    "v_accvgpr_write_b32 a170, v34\n" \
    "v_accvgpr_write_b32 a171, v34\n" \
    "v_accvgpr_write_b32 a172, v34\n" \
    "v_accvgpr_write_b32 a173, v34\n" \
    "v_accvgpr_write_b32 a174, v34\n" \
    "v_accvgpr_write_b32 a175, v34\n" \
    "v_accvgpr_write_b32 a176, v34\n" \
    "v_accvgpr_write_b32 a177, v34\n" \
    "v_accvgpr_write_b32 a178, v34\n" \
    "v_accvgpr_write_b32 a179, v34\n" \
    "v_accvgpr_write_b32 a180, v34\n" \
    "v_accvgpr_write_b32 a181, v34\n" \
    "v_accvgpr_write_b32 a182, v34\n" \
    "v_accvgpr_write_b32 a183, v34\n" \
    "v_accvgpr_write_b32 a184, v34\n" \
    "v_accvgpr_write_b32 a185, v34\n" \
    "v_accvgpr_write_b32 a186, v34\n" \
    "v_accvgpr_write_b32 a187, v34\n" \
    "v_accvgpr_write_b32 a188, v34\n" \
    "v_accvgpr_write_b32 a189, v34\n" \
    "v_accvgpr_write_b32 a190, v34\n" \
    "v_accvgpr_write_b32 a191, v34\n" \
    "v_accvgpr_write_b32 a192, v34\n" \
    "v_accvgpr_write_b32 a193, v34\n" \
    "v_accvgpr_write_b32 a194, v34\n" \
    "v_accvgpr_write_b32 a195, v34\n" \
    "v_accvgpr_write_b32 a196, v34\n" \
    "v_accvgpr_write_b32 a197, v34\n" \
    "v_accvgpr_write_b32 a198, v34\n" \
    "v_accvgpr_write_b32 a199, v34\n" \
    "v_accvgpr_write_b32 a204, v34\n" \
    "v_accvgpr_write_b32 a205, v34\n" \
    "v_accvgpr_write_b32 a206, v34\n" \
    "v_accvgpr_write_b32 a207, v34\n" \
    "v_accvgpr_write_b32 a212, v34\n" \
    "v_accvgpr_write_b32 a213, v34\n" \
    "v_accvgpr_write_b32 a214, v34\n" \
    "v_accvgpr_write_b32 a215, v34\n" \
    "v_accvgpr_write_b32 a200, v34\n" \
    "v_accvgpr_write_b32 a201, v34\n" \
    "v_accvgpr_write_b32 a202, v34\n" \
    "v_accvgpr_write_b32 a203, v34\n" \
    "v_accvgpr_write_b32 a208, v34\n" \
    "v_accvgpr_write_b32 a209, v34\n" \
    "v_accvgpr_write_b32 a210, v34\n" \
    "v_accvgpr_write_b32 a211, v34\n" \
    "v_accvgpr_write_b32 a216, v34\n" \
    "v_accvgpr_write_b32 a217, v34\n" \
    "v_accvgpr_write_b32 a218, v34\n" \
    "v_accvgpr_write_b32 a219, v34\n" \
    "v_accvgpr_write_b32 a220, v34\n" \
    "v_accvgpr_write_b32 a221, v34\n" \
    "v_accvgpr_write_b32 a222, v34\n" \
    "v_accvgpr_write_b32 a223, v34\n" \
    "v_accvgpr_write_b32 a224, v34\n" \
    "v_accvgpr_write_b32 a225, v34\n" \
    "v_accvgpr_write_b32 a226, v34\n" \
    "v_accvgpr_write_b32 a227, v34\n" \
    "v_accvgpr_write_b32 a232, v34\n" \
    "v_accvgpr_write_b32 a233, v34\n" \
    "v_accvgpr_write_b32 a234, v34\n" \
    "v_accvgpr_write_b32 a235, v34\n" \
    "v_accvgpr_write_b32 a240, v34\n" \
    "v_accvgpr_write_b32 a241, v34\n" \
    "v_accvgpr_write_b32 a242, v34\n" \
    "v_accvgpr_write_b32 a243, v34\n" \
    "v_accvgpr_write_b32 a248, v34\n" \
    "v_accvgpr_write_b32 a249, v34\n" \
    "v_accvgpr_write_b32 a250, v34\n" \
    "v_accvgpr_write_b32 a251, v34\n" \
    "v_accvgpr_write_b32 a252, v34\n" \
    "v_accvgpr_write_b32 a253, v34\n" \
    "v_accvgpr_write_b32 a254, v34\n" \
    "v_accvgpr_write_b32 a255, v34\n" \
    "v_accvgpr_write_b32 a228, v34\n" \
    "v_accvgpr_write_b32 a229, v34\n" \
    "v_accvgpr_write_b32 a230, v34\n" \
    "v_accvgpr_write_b32 a231, v34\n" \
    "v_accvgpr_write_b32 a236, v34\n" \
    "v_accvgpr_write_b32 a237, v34\n" \
    "v_accvgpr_write_b32 a238, v34\n" \
    "v_accvgpr_write_b32 a239, v34\n" \
    "v_accvgpr_write_b32 a244, v34\n" \
    "v_accvgpr_write_b32 a245, v34\n" \
    "v_accvgpr_write_b32 a246, v34\n" \
    "v_accvgpr_write_b32 a247, v34\n" \
    "v_add_u32_e32 v56, 0, v53\n" \
    "v_add_u32_e32 v247, 0x18bc0, v56\n" \
    "v_add_u32_e32 v57, 0, v55\n" \
    "v_add_u32_e32 v59, 0x217a0, v57\n" \
    "s_mov_b32 s6, s2\n" \
    "s_mov_b32 s7, s3\n" \
    "s_mov_b32 s18, s2\n" \
    "s_mov_b32 s19, s3\n" \
    "s_mov_b32 s10, s2\n" \
    "s_mov_b32 s11, s3\n" \
    "v_add_u32_e32 v245, 0x149e0, v56\n" \
    "v_add_u32_e32 v58, 0, v54\n" \
    "v_add_u32_e32 v163, 0x20fa0, v58\n" \
    "v_add_u32_e32 v246, 0x1cdc0, v56\n" \
    "v_add_u32_e32 v53, 0x107e0, v56\n" \
    "s_waitcnt vmcnt(20), lgkmcnt(0)\n" \
    "s_barrier\n" \
    ".Lmx_1:\n" \
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
    "ds_read_b128 v[124:127], v53\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[170:173], v[152:155], a[28:31], v164, v31, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "ds_read_b128 v[132:135], v53, offset:64\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[174:177], v[148:151], a[32:35], v164, v31, op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "ds_read_b128 v[128:131], v53, offset:256\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[178:181], v[152:155], a[32:35], v164, v31, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "ds_read_b128 v[136:139], v53, offset:320\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[182:185], v[148:151], a[36:39], v165, v31, op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "ds_read_b128 v[140:143], v53, offset:512\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[186:189], v[152:155], a[36:39], v165, v31, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "ds_read_b128 v[144:147], v53, offset:576\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[190:193], v[148:151], a[44:47], v165, v31, op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "ds_read_b128 v[148:151], v53, offset:768\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[240:243], v[152:155], a[44:47], v165, v31, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "ds_read_b128 v[152:155], v53, offset:832\n" \
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
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[240:243], v[232:235], a[120:123], v165, v35, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "s_cbranch_scc1 .Lmx_1\n" \
    "v_and_b32_e32 v26, 0xe0, v37\n" \
    "v_lshl_or_b32 v27, v52, 3, s34\n" \
    "s_waitcnt lgkmcnt(0)\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[124:127], v[116:119], a[132:135], v32, v30, op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[132:135], v[120:123], a[132:135], v32, v30, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[128:131], v[116:119], a[140:143], v32, v30, op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[136:139], v[120:123], a[140:143], v32, v30, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[140:143], v[116:119], a[144:147], v33, v30, op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[144:147], v[120:123], a[144:147], v33, v30, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[148:151], v[116:119], a[152:155], v33, v30, op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[152:155], v[120:123], a[152:155], v33, v30, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[124:127], v[108:111], a[128:131], v32, v30, op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[132:135], v[112:115], a[128:131], v32, v30, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[128:131], v[108:111], a[136:139], v32, v30, op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[136:139], v[112:115], a[136:139], v32, v30, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[140:143], v[108:111], a[148:151], v33, v30, op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[144:147], v[112:115], a[148:151], v33, v30, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[148:151], v[108:111], a[156:159], v33, v30, op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[152:155], v[112:115], a[156:159], v33, v30, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[124:127], v[100:103], a[160:163], v32, v31, op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[132:135], v[104:107], a[160:163], v32, v31, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[128:131], v[100:103], a[164:167], v32, v31, op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[136:139], v[104:107], a[164:167], v32, v31, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[140:143], v[100:103], a[168:171], v33, v31, op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], v[144:147], v[104:107], a[168:171], v33, v31, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[148:151], v[100:103], a[172:175], v33, v31, op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], v[152:155], v[104:107], a[172:175], v33, v31, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[124:127], v[88:91], a[176:179], v32, v31, op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], v[132:135], v[96:99], a[176:179], v32, v31, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[128:131], v[88:91], a[180:183], v32, v31, op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], v[136:139], v[96:99], a[180:183], v32, v31, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[140:143], v[88:91], a[184:187], v33, v31, op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], v[144:147], v[96:99], a[184:187], v33, v31, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[148:151], v[88:91], a[188:191], v33, v31, op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], v[152:155], v[96:99], a[188:191], v33, v31, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[124:127], v[84:87], a[192:195], v32, v0, op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[132:135], v[92:95], a[192:195], v32, v0, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "s_nop 7\n" \
    "v_accvgpr_read_b32 v201, a195\n" \
    "v_accvgpr_read_b32 v200, a194\n" \
    "v_accvgpr_read_b32 v199, a193\n" \
    "v_accvgpr_read_b32 v198, a192\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[128:131], v[84:87], a[196:199], v32, v0, op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[136:139], v[92:95], a[192:195], v32, v0, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "s_nop 7\n" \
    "v_accvgpr_read_b32 v205, a195\n" \
    "v_accvgpr_read_b32 v204, a194\n" \
    "v_accvgpr_read_b32 v203, a193\n" \
    "v_accvgpr_read_b32 v202, a192\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[140:143], v[84:87], a[204:207], v33, v0, op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], v[144:147], v[92:95], a[192:195], v33, v0, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[148:151], v[84:87], a[212:215], v33, v0, op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], v[152:155], v[92:95], a[192:195], v33, v0, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[124:127], v[76:79], a[200:203], v32, v0, op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], v[132:135], v[80:83], a[192:195], v32, v0, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[128:131], v[76:79], a[208:211], v32, v0, op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], v[136:139], v[80:83], a[196:199], v32, v0, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[140:143], v[76:79], a[216:219], v33, v0, op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], v[144:147], v[80:83], a[196:199], v33, v0, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[148:151], v[76:79], a[220:223], v33, v0, op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], v[152:155], v[80:83], a[208:211], v33, v0, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[124:127], v[68:71], a[224:227], v32, v1, op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], v[132:135], v[72:75], a[208:211], v32, v1, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[128:131], v[68:71], a[232:235], v32, v1, op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], v[136:139], v[72:75], a[208:211], v32, v1, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[140:143], v[68:71], a[240:243], v33, v1, op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], v[144:147], v[72:75], a[208:211], v33, v1, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[148:151], v[68:71], a[248:251], v33, v1, op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[152:155], v[72:75], a[208:211], v33, v1, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "s_nop 7\n" \
    "v_accvgpr_read_b32 v194, a208\n" \
    "v_accvgpr_read_b32 v195, a209\n" \
    "v_accvgpr_read_b32 v196, a210\n" \
    "v_accvgpr_read_b32 v197, a211\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[124:127], v[60:63], a[252:255], v32, v1, op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], v[132:135], v[64:67], a[208:211], v32, v1, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[128:131], v[60:63], a[228:231], v32, v1, op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], v[136:139], v[64:67], a[208:211], v32, v1, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[140:143], v[60:63], a[236:239], v33, v1, op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], v[144:147], v[64:67], a[216:219], v33, v1, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[148:151], v[60:63], a[244:247], v33, v1, op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], v[152:155], v[64:67], a[216:219], v33, v1, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "s_waitcnt vmcnt(20), lgkmcnt(0)\n" \
    "s_barrier\n" \
    "v_add_u32_e32 v2, 0x18bc0, v56\n" \
    "ds_read_b128 v[4:7], v2\n" \
    "ds_read_b128 v[10:13], v2, offset:64\n" \
    "ds_read_b128 v[14:17], v2, offset:256\n" \
    "ds_read_b128 v[18:21], v2, offset:320\n" \
    "ds_read_b128 v[22:25], v2, offset:512\n" \
    "ds_read_b128 v[32:35], v2, offset:576\n" \
    "ds_read_b128 v[38:41], v2, offset:768\n" \
    "ds_read_b128 v[42:45], v2, offset:832\n" \
    "v_add_u32_e32 v8, 0x217a0, v57\n" \
    "s_waitcnt vmcnt(19)\n" \
    "ds_write_b32 v8, v162\n" \
    "s_waitcnt lgkmcnt(0)\n" \
    "s_barrier\n" \
    "ds_read_b64_tr_b8 v[2:3], v50\n" \
    "s_waitcnt lgkmcnt(0)\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[4:7], v[116:119], a[124:127], v2, v30, op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], v[10:13], v[120:123], a[124:127], v2, v30, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[14:17], v[116:119], a[0:3], v2, v30, op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[18:21], v[120:123], a[0:3], v2, v30, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[22:25], v[116:119], a[4:7], v3, v30, op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[32:35], v[120:123], a[4:7], v3, v30, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[38:41], v[116:119], a[8:11], v3, v30, op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[42:45], v[120:123], a[8:11], v3, v30, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[4:7], v[108:111], a[12:15], v2, v30, op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[10:13], v[112:115], a[12:15], v2, v30, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[14:17], v[108:111], a[16:19], v2, v30, op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[18:21], v[112:115], a[16:19], v2, v30, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[22:25], v[108:111], a[20:23], v3, v30, op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[32:35], v[112:115], a[20:23], v3, v30, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[38:41], v[108:111], a[24:27], v3, v30, op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[42:45], v[112:115], a[24:27], v3, v30, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[4:7], v[100:103], a[28:31], v2, v31, op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[10:13], v[104:107], a[28:31], v2, v31, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[14:17], v[100:103], a[32:35], v2, v31, op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[18:21], v[104:107], a[28:31], v2, v31, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "s_nop 7\n" \
    "v_accvgpr_read_b32 v193, a31\n" \
    "v_accvgpr_read_b32 v192, a30\n" \
    "v_accvgpr_read_b32 v191, a29\n" \
    "v_accvgpr_read_b32 v190, a28\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[22:25], v[100:103], a[36:39], v3, v31, op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[32:35], v[104:107], a[28:31], v3, v31, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[38:41], v[100:103], a[44:47], v3, v31, op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[42:45], v[104:107], a[32:35], v3, v31, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[4:7], v[88:91], a[52:55], v2, v31, op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[10:13], v[96:99], a[32:35], v2, v31, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[14:17], v[88:91], a[40:43], v2, v31, op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], v[18:21], v[96:99], a[36:39], v2, v31, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[22:25], v[88:91], a[48:51], v3, v31, op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[32:35], v[96:99], a[36:39], v3, v31, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[38:41], v[88:91], a[56:59], v3, v31, op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], v[42:45], v[96:99], a[40:43], v3, v31, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[4:7], v[84:87], a[60:63], v2, v0, op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[10:13], v[92:95], a[40:43], v2, v0, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "s_nop 7\n" \
    "v_accvgpr_read_b32 v189, a43\n" \
    "v_accvgpr_read_b32 v188, a42\n" \
    "v_accvgpr_read_b32 v187, a41\n" \
    "v_accvgpr_read_b32 v186, a40\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[14:17], v[84:87], a[64:67], v2, v0, op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[18:21], v[92:95], a[40:43], v2, v0, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[22:25], v[84:87], a[68:71], v3, v0, op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[32:35], v[92:95], a[40:43], v3, v0, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[38:41], v[84:87], a[72:75], v3, v0, op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[42:45], v[92:95], a[48:51], v3, v0, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[4:7], v[76:79], a[76:79], v2, v0, op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[10:13], v[80:83], a[48:51], v2, v0, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[14:17], v[76:79], a[80:83], v2, v0, op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[18:21], v[80:83], a[52:55], v2, v0, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[22:25], v[76:79], a[84:87], v3, v0, op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[32:35], v[80:83], a[56:59], v3, v0, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[38:41], v[76:79], a[88:91], v3, v0, op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[42:45], v[80:83], a[60:63], v3, v0, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[4:7], v[68:71], a[92:95], v2, v1, op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[10:13], v[72:75], a[72:75], v2, v1, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "s_nop 7\n" \
    "v_accvgpr_read_b32 v185, a75\n" \
    "v_accvgpr_read_b32 v184, a74\n" \
    "v_accvgpr_read_b32 v183, a73\n" \
    "v_accvgpr_read_b32 v182, a72\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[14:17], v[68:71], a[96:99], v2, v1, op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[18:21], v[72:75], a[72:75], v2, v1, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[22:25], v[68:71], a[100:103], v3, v1, op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[32:35], v[72:75], a[72:75], v3, v1, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[38:41], v[68:71], a[104:107], v3, v1, op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[42:45], v[72:75], a[76:79], v3, v1, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[4:7], v[60:63], a[108:111], v2, v1, op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[10:13], v[64:67], a[80:83], v2, v1, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[14:17], v[60:63], a[112:115], v2, v1, op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[18:21], v[64:67], a[84:87], v2, v1, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[22:25], v[60:63], a[116:119], v3, v1, op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[32:35], v[64:67], a[88:91], v3, v1, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[38:41], v[60:63], a[120:123], v3, v1, op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[42:45], v[64:67], a[92:95], v3, v1, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "s_waitcnt vmcnt(7), lgkmcnt(0)\n" \
    "s_barrier\n" \
    "ds_read_b128 v[128:131], v46, offset:33792\n" \
    "ds_read_b128 v[132:135], v46, offset:33856\n" \
    "ds_read_b128 v[136:139], v46, offset:34048\n" \
    "ds_read_b128 v[140:143], v46, offset:34112\n" \
    "ds_read_b128 v[100:103], v46, offset:34304\n" \
    "ds_read_b128 v[104:107], v46, offset:34368\n" \
    "ds_read_b128 v[92:95], v46, offset:34560\n" \
    "ds_read_b128 v[96:99], v46, offset:34624\n" \
    "ds_read_b128 v[84:87], v46, offset:50688\n" \
    "ds_read_b128 v[88:91], v46, offset:50752\n" \
    "ds_read_b128 v[76:79], v46, offset:50944\n" \
    "ds_read_b128 v[80:83], v46, offset:51008\n" \
    "ds_read_b128 v[30:33], v46, offset:51200\n" \
    "ds_read_b128 v[38:41], v46, offset:51264\n" \
    "ds_read_b128 v[18:21], v46, offset:51456\n" \
    "ds_read_b128 v[22:25], v46, offset:51520\n" \
    "v_add_u32_e32 v0, 0x149e0, v56\n" \
    "ds_read_b128 v[144:147], v0\n" \
    "ds_read_b128 v[148:151], v0, offset:64\n" \
    "ds_read_b128 v[152:155], v0, offset:256\n" \
    "ds_read_b128 v[162:165], v0, offset:320\n" \
    "ds_read_b128 v[166:169], v0, offset:512\n" \
    "ds_read_b128 v[170:173], v0, offset:576\n" \
    "ds_read_b128 v[174:177], v0, offset:768\n" \
    "ds_read_b128 v[178:181], v0, offset:832\n" \
    "v_add_u32_e32 v0, 0x20fa0, v58\n" \
    "s_waitcnt vmcnt(6)\n" \
    "ds_write_b64 v0, v[156:157]\n" \
    "s_waitcnt vmcnt(5)\n" \
    "ds_write_b32 v8, v160\n" \
    "s_waitcnt lgkmcnt(0)\n" \
    "s_barrier\n" \
    "ds_read_b64_tr_b8 v[4:5], v51\n" \
    "ds_read_b64_tr_b8 v[0:1], v51, offset:128\n" \
    "ds_read_b64_tr_b8 v[6:7], v50\n" \
    "v_lshrrev_b32_e32 v2, 4, v37\n" \
    "v_or_b32_e32 v3, 16, v2\n" \
    "v_or_b32_e32 v9, 32, v2\n" \
    "v_or_b32_e32 v10, 48, v2\n" \
    "v_mul_lo_u32 v28, v2, s26\n" \
    "v_mul_lo_u32 v34, v3, s26\n" \
    "v_mul_lo_u32 v35, v9, s26\n" \
    "v_mul_lo_u32 v37, v10, s26\n" \
    "s_mul_i32 s0, s33, s26\n" \
    "s_ashr_i32 s1, s0, 31\n" \
    "s_lshl_b64 s[0:1], s[0:1], 1\n" \
    "s_add_u32 s12, s12, s0\n" \
    "s_addc_u32 s5, s13, s1\n" \
    "s_lshl_b32 s0, s26, 6\n" \
    "s_ashr_i32 s1, s0, 31\n" \
    "s_lshl_b64 s[0:1], s[0:1], 1\n" \
    "s_add_u32 s8, s12, s0\n" \
    "s_addc_u32 s3, s5, s1\n" \
    "s_add_u32 s4, s8, s0\n" \
    "s_addc_u32 s2, s3, s1\n" \
    "s_add_u32 s0, s4, s0\n" \
    "s_addc_u32 s1, s2, s1\n" \
    "s_waitcnt lgkmcnt(0)\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[144:147], v[128:131], a[132:135], v6, v4, op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[148:151], v[132:135], a[100:103], v6, v4, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[152:155], v[128:131], a[140:143], v6, v4, op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[162:165], v[132:135], a[104:107], v6, v4, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[166:169], v[128:131], a[144:147], v7, v4, op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[170:173], v[132:135], a[108:111], v7, v4, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[174:177], v[128:131], a[152:155], v7, v4, op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[178:181], v[132:135], a[112:115], v7, v4, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[144:147], v[136:139], a[128:131], v6, v4, op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[148:151], v[140:143], a[116:119], v6, v4, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[152:155], v[136:139], a[136:139], v6, v4, op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[162:165], v[140:143], a[120:123], v6, v4, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[166:169], v[136:139], a[148:151], v7, v4, op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[170:173], v[140:143], a[128:131], v7, v4, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[174:177], v[136:139], a[156:159], v7, v4, op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[178:181], v[140:143], a[132:135], v7, v4, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "s_waitcnt vmcnt(1), lgkmcnt(0)\n" \
    "s_barrier\n" \
    "v_add_u32_e32 v2, 0x1cdc0, v56\n" \
    "ds_read_b128 v[42:45], v2\n" \
    "ds_read_b128 v[46:49], v2, offset:64\n" \
    "ds_read_b128 v[52:55], v2, offset:256\n" \
    "ds_read_b128 v[56:59], v2, offset:320\n" \
    "ds_read_b128 v[60:63], v2, offset:512\n" \
    "ds_read_b128 v[64:67], v2, offset:576\n" \
    "ds_read_b128 v[68:71], v2, offset:768\n" \
    "ds_read_b128 v[72:75], v2, offset:832\n" \
    "s_waitcnt vmcnt(0)\n" \
    "ds_write_b32 v8, v161\n" \
    "s_waitcnt lgkmcnt(0)\n" \
    "s_barrier\n" \
    "ds_read_b64_tr_b8 v[2:3], v50\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[144:147], v[100:103], a[160:163], v6, v5, op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[148:151], v[104:107], a[136:139], v6, v5, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[152:155], v[100:103], a[164:167], v6, v5, op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[162:165], v[104:107], a[140:143], v6, v5, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[166:169], v[100:103], a[168:171], v7, v5, op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[170:173], v[104:107], a[144:147], v7, v5, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[174:177], v[100:103], a[172:175], v7, v5, op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[178:181], v[104:107], a[148:151], v7, v5, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[144:147], v[92:95], a[176:179], v6, v5, op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[148:151], v[96:99], a[152:155], v6, v5, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[152:155], v[92:95], a[180:183], v6, v5, op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[162:165], v[96:99], a[156:159], v6, v5, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[166:169], v[92:95], a[184:187], v7, v5, op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[170:173], v[96:99], a[160:163], v7, v5, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[174:177], v[92:95], a[188:191], v7, v5, op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[178:181], v[96:99], a[164:167], v7, v5, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_accvgpr_read_b32 v8, a100\n" \
    "v_accvgpr_read_b32 v9, a101\n" \
    "v_cvt_pk_bf16_f32 v10, v8, v9\n" \
    "v_accvgpr_read_b32 v8, a102\n" \
    "v_accvgpr_read_b32 v9, a103\n" \
    "v_cvt_pk_bf16_f32 v11, v8, v9\n" \
    "v_accvgpr_read_b32 v8, a104\n" \
    "v_accvgpr_read_b32 v9, a105\n" \
    "v_cvt_pk_bf16_f32 v14, v8, v9\n" \
    "v_accvgpr_read_b32 v8, a106\n" \
    "v_accvgpr_read_b32 v9, a107\n" \
    "v_cvt_pk_bf16_f32 v15, v8, v9\n" \
    "v_accvgpr_read_b32 v8, a108\n" \
    "v_accvgpr_read_b32 v9, a109\n" \
    "v_cvt_pk_bf16_f32 v108, v8, v9\n" \
    "v_accvgpr_read_b32 v8, a110\n" \
    "v_accvgpr_read_b32 v9, a111\n" \
    "v_cvt_pk_bf16_f32 v109, v8, v9\n" \
    "v_accvgpr_read_b32 v8, a112\n" \
    "v_accvgpr_read_b32 v9, a113\n" \
    "v_cvt_pk_bf16_f32 v112, v8, v9\n" \
    "v_accvgpr_read_b32 v8, a114\n" \
    "v_accvgpr_read_b32 v9, a115\n" \
    "v_cvt_pk_bf16_f32 v113, v8, v9\n" \
    "v_accvgpr_read_b32 v8, a116\n" \
    "v_accvgpr_read_b32 v9, a117\n" \
    "v_cvt_pk_bf16_f32 v12, v8, v9\n" \
    "v_accvgpr_read_b32 v8, a118\n" \
    "v_accvgpr_read_b32 v9, a119\n" \
    "v_cvt_pk_bf16_f32 v13, v8, v9\n" \
    "v_accvgpr_read_b32 v8, a120\n" \
    "v_accvgpr_read_b32 v9, a121\n" \
    "v_cvt_pk_bf16_f32 v16, v8, v9\n" \
    "v_accvgpr_read_b32 v8, a122\n" \
    "v_accvgpr_read_b32 v9, a123\n" \
    "v_cvt_pk_bf16_f32 v17, v8, v9\n" \
    "v_accvgpr_read_b32 v8, a128\n" \
    "v_accvgpr_read_b32 v9, a129\n" \
    "v_cvt_pk_bf16_f32 v110, v8, v9\n" \
    "v_accvgpr_read_b32 v8, a130\n" \
    "v_accvgpr_read_b32 v9, a131\n" \
    "v_cvt_pk_bf16_f32 v111, v8, v9\n" \
    "v_accvgpr_read_b32 v8, a132\n" \
    "v_accvgpr_read_b32 v9, a133\n" \
    "v_cvt_pk_bf16_f32 v114, v8, v9\n" \
    "v_accvgpr_read_b32 v8, a134\n" \
    "v_accvgpr_read_b32 v9, a135\n" \
    "v_cvt_pk_bf16_f32 v115, v8, v9\n" \
    "v_lshlrev_b32_e32 v8, 8, v36\n" \
    "v_and_b32_e32 v50, 0x70, v29\n" \
    "v_and_b32_e32 v51, 1, v36\n" \
    "v_lshlrev_b32_e32 v9, 12, v51\n" \
    "v_and_b32_e32 v36, 16, v36\n" \
    "v_lshlrev_b32_e32 v116, 4, v36\n" \
    "s_movk_i32 s6, 0x2e00\n" \
    "v_and_or_b32 v8, v8, s6, v9\n" \
    "v_mov_b32_e32 v9, 0x70\n" \
    "v_bitop3_b32 v9, s23, v29, v9, bitop3:0x78\n" \
    "v_or3_b32 v29, v116, v8, v9\n" \
    "v_or_b32_e32 v116, s22, v29\n" \
    "v_add_u32_e32 v8, 0, v116\n" \
    "ds_write_b128 v8, v[10:13]\n" \
    "v_xad_u32 v9, v116, 32, 0\n" \
    "ds_write_b128 v9, v[14:17]\n" \
    "v_xad_u32 v10, v116, 64, 0\n" \
    "ds_write_b128 v10, v[108:111]\n" \
    "s_movk_i32 s6, 0x60\n" \
    "v_mov_b32_e32 v11, s22\n" \
    "v_bitop3_b32 v11, v29, s6, v11, bitop3:0x36\n" \
    "v_add_u32_e32 v11, 0, v11\n" \
    "ds_write_b128 v11, v[112:115]\n" \
    "s_waitcnt lgkmcnt(0)\n" \
    "s_barrier\n" \
    "v_lshlrev_b32_e32 v12, 4, v26\n" \
    "v_lshrrev_b32_e32 v13, 1, v26\n" \
    "v_lshlrev_b32_e32 v14, 8, v36\n" \
    "v_bitop3_b32 v12, v12, v13, v50, bitop3:0x36\n" \
    "v_lshl_add_u32 v13, v51, 13, 0\n" \
    "v_add3_u32 v12, v13, v14, v12\n" \
    "ds_read_b128 v[108:111], v12\n" \
    "ds_read_b128 v[114:117], v12, offset:256\n" \
    "ds_read_b128 v[118:121], v12, offset:128\n" \
    "ds_read_b128 v[124:127], v12, offset:384\n" \
    "s_and_b32 s13, s5, 0xffff\n" \
    "s_mov_b32 s15, 0x27000\n" \
    "s_mov_b32 s14, 0x7ffffffe\n" \
    "s_waitcnt lgkmcnt(3)\n" \
    "v_mov_b32_e32 v112, v108\n" \
    "v_mov_b32_e32 v113, v109\n" \
    "v_add_lshl_u32 v13, v28, v27, 1\n" \
    "s_waitcnt lgkmcnt(2)\n" \
    "buffer_store_dwordx4 v[112:115], v13, s[12:15], 0, offen\n" \
    "s_waitcnt lgkmcnt(1)\n" \
    "v_mov_b32_e32 v122, v118\n" \
    "v_mov_b32_e32 v123, v119\n" \
    "v_add_lshl_u32 v14, v34, v27, 1\n" \
    "s_waitcnt lgkmcnt(0)\n" \
    "buffer_store_dwordx4 v[122:125], v14, s[12:15], 0, offen\n" \
    "v_mov_b32_e32 v112, v116\n" \
    "v_mov_b32_e32 v113, v117\n" \
    "v_add_lshl_u32 v15, v35, v27, 1\n" \
    "buffer_store_dwordx4 v[110:113], v15, s[12:15], 0, offen\n" \
    "v_mov_b32_e32 v122, v126\n" \
    "v_mov_b32_e32 v123, v127\n" \
    "v_add_lshl_u32 v16, v37, v27, 1\n" \
    "buffer_store_dwordx4 v[120:123], v16, s[12:15], 0, offen\n" \
    "v_accvgpr_write_b32 a100, v198\n" \
    "v_accvgpr_write_b32 a101, v199\n" \
    "v_accvgpr_write_b32 a102, v200\n" \
    "v_accvgpr_write_b32 a103, v201\n" \
    "s_nop 1\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[144:147], v[84:87], a[100:103], v6, v0, op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[148:151], v[88:91], a[100:103], v6, v0, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_accvgpr_write_b32 a104, v202\n" \
    "v_accvgpr_write_b32 a105, v203\n" \
    "v_accvgpr_write_b32 a106, v204\n" \
    "v_accvgpr_write_b32 a107, v205\n" \
    "s_nop 1\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[152:155], v[84:87], a[104:107], v6, v0, op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[162:165], v[88:91], a[104:107], v6, v0, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[166:169], v[84:87], a[204:207], v7, v0, op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[170:173], v[88:91], a[108:111], v7, v0, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[174:177], v[84:87], a[212:215], v7, v0, op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[178:181], v[88:91], a[112:115], v7, v0, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[144:147], v[76:79], a[192:195], v6, v0, op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[148:151], v[80:83], a[116:119], v6, v0, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[152:155], v[76:79], a[200:203], v6, v0, op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[162:165], v[80:83], a[120:123], v6, v0, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[166:169], v[76:79], a[196:199], v7, v0, op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[170:173], v[80:83], a[128:131], v7, v0, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[174:177], v[76:79], a[220:223], v7, v0, op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[178:181], v[80:83], a[132:135], v7, v0, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_accvgpr_read_b32 v26, a136\n" \
    "v_accvgpr_read_b32 v17, a137\n" \
    "v_cvt_pk_bf16_f32 v26, v26, v17\n" \
    "v_accvgpr_read_b32 v28, a138\n" \
    "v_accvgpr_read_b32 v17, a139\n" \
    "v_cvt_pk_bf16_f32 v27, v28, v17\n" \
    "v_accvgpr_read_b32 v28, a140\n" \
    "v_accvgpr_read_b32 v17, a141\n" \
    "v_cvt_pk_bf16_f32 v34, v28, v17\n" \
    "v_accvgpr_read_b32 v28, a142\n" \
    "v_accvgpr_read_b32 v17, a143\n" \
    "v_cvt_pk_bf16_f32 v35, v28, v17\n" \
    "v_accvgpr_read_b32 v28, a144\n" \
    "v_accvgpr_read_b32 v17, a145\n" \
    "v_cvt_pk_bf16_f32 v108, v28, v17\n" \
    "v_accvgpr_read_b32 v28, a146\n" \
    "v_accvgpr_read_b32 v17, a147\n" \
    "v_cvt_pk_bf16_f32 v109, v28, v17\n" \
    "v_accvgpr_read_b32 v28, a148\n" \
    "v_accvgpr_read_b32 v17, a149\n" \
    "v_cvt_pk_bf16_f32 v112, v28, v17\n" \
    "v_accvgpr_read_b32 v28, a150\n" \
    "v_accvgpr_read_b32 v17, a151\n" \
    "v_cvt_pk_bf16_f32 v113, v28, v17\n" \
    "v_accvgpr_read_b32 v28, a152\n" \
    "v_accvgpr_read_b32 v17, a153\n" \
    "v_cvt_pk_bf16_f32 v28, v28, v17\n" \
    "v_accvgpr_read_b32 v36, a154\n" \
    "v_accvgpr_read_b32 v17, a155\n" \
    "v_cvt_pk_bf16_f32 v29, v36, v17\n" \
    "v_accvgpr_read_b32 v36, a156\n" \
    "v_accvgpr_read_b32 v17, a157\n" \
    "v_cvt_pk_bf16_f32 v36, v36, v17\n" \
    "v_accvgpr_read_b32 v50, a158\n" \
    "v_accvgpr_read_b32 v17, a159\n" \
    "v_cvt_pk_bf16_f32 v37, v50, v17\n" \
    "v_accvgpr_read_b32 v50, a160\n" \
    "v_accvgpr_read_b32 v17, a161\n" \
    "v_cvt_pk_bf16_f32 v110, v50, v17\n" \
    "v_accvgpr_read_b32 v50, a162\n" \
    "v_accvgpr_read_b32 v17, a163\n" \
    "v_cvt_pk_bf16_f32 v111, v50, v17\n" \
    "v_accvgpr_read_b32 v50, a164\n" \
    "v_accvgpr_read_b32 v17, a165\n" \
    "v_cvt_pk_bf16_f32 v114, v50, v17\n" \
    "v_accvgpr_read_b32 v50, a166\n" \
    "v_accvgpr_read_b32 v17, a167\n" \
    "v_cvt_pk_bf16_f32 v115, v50, v17\n" \
    "s_waitcnt lgkmcnt(0)\n" \
    "s_barrier\n" \
    "ds_write_b128 v8, v[26:29]\n" \
    "ds_write_b128 v9, v[34:37]\n" \
    "ds_write_b128 v10, v[108:111]\n" \
    "ds_write_b128 v11, v[112:115]\n" \
    "s_waitcnt lgkmcnt(0)\n" \
    "s_barrier\n" \
    "ds_read_b128 v[108:111], v12\n" \
    "ds_read_b128 v[114:117], v12, offset:256\n" \
    "ds_read_b128 v[118:121], v12, offset:128\n" \
    "ds_read_b128 v[124:127], v12, offset:384\n" \
    "s_and_b32 s9, s3, 0xffff\n" \
    "s_mov_b32 s10, s14\n" \
    "s_mov_b32 s11, s15\n" \
    "s_waitcnt lgkmcnt(3)\n" \
    "v_mov_b32_e32 v112, v108\n" \
    "v_mov_b32_e32 v113, v109\n" \
    "s_waitcnt lgkmcnt(2)\n" \
    "buffer_store_dwordx4 v[112:115], v13, s[8:11], 0, offen\n" \
    "s_waitcnt lgkmcnt(1)\n" \
    "v_mov_b32_e32 v122, v118\n" \
    "v_mov_b32_e32 v123, v119\n" \
    "s_waitcnt lgkmcnt(0)\n" \
    "buffer_store_dwordx4 v[122:125], v14, s[8:11], 0, offen\n" \
    "v_mov_b32_e32 v112, v116\n" \
    "v_mov_b32_e32 v113, v117\n" \
    "buffer_store_dwordx4 v[110:113], v15, s[8:11], 0, offen\n" \
    "v_mov_b32_e32 v122, v126\n" \
    "v_mov_b32_e32 v123, v127\n" \
    "buffer_store_dwordx4 v[120:123], v16, s[8:11], 0, offen\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[144:147], v[30:33], a[224:227], v6, v1, op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[148:151], v[38:41], a[136:139], v6, v1, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[152:155], v[30:33], a[232:235], v6, v1, op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[162:165], v[38:41], a[140:143], v6, v1, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[166:169], v[30:33], a[240:243], v7, v1, op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[170:173], v[38:41], a[144:147], v7, v1, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_accvgpr_write_b32 a148, v194\n" \
    "v_accvgpr_write_b32 a149, v195\n" \
    "v_accvgpr_write_b32 a150, v196\n" \
    "v_accvgpr_write_b32 a151, v197\n" \
    "s_nop 1\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[174:177], v[30:33], a[148:151], v7, v1, op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[178:181], v[38:41], a[148:151], v7, v1, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[144:147], v[18:21], a[248:251], v6, v1, op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], v[148:151], v[22:25], a[152:155], v6, v1, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[152:155], v[18:21], a[208:211], v6, v1, op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], v[162:165], v[22:25], a[156:159], v6, v1, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[166:169], v[18:21], a[228:231], v7, v1, op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], v[170:173], v[22:25], a[160:163], v7, v1, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[174:177], v[18:21], a[216:219], v7, v1, op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], v[178:181], v[22:25], a[164:167], v7, v1, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_accvgpr_read_b32 v6, a100\n" \
    "v_accvgpr_read_b32 v7, a101\n" \
    "v_cvt_pk_bf16_f32 v26, v6, v7\n" \
    "v_accvgpr_read_b32 v6, a102\n" \
    "v_accvgpr_read_b32 v7, a103\n" \
    "v_cvt_pk_bf16_f32 v27, v6, v7\n" \
    "v_accvgpr_read_b32 v6, a104\n" \
    "v_accvgpr_read_b32 v7, a105\n" \
    "v_cvt_pk_bf16_f32 v34, v6, v7\n" \
    "v_accvgpr_read_b32 v6, a106\n" \
    "v_accvgpr_read_b32 v7, a107\n" \
    "v_cvt_pk_bf16_f32 v35, v6, v7\n" \
    "v_accvgpr_read_b32 v6, a108\n" \
    "v_accvgpr_read_b32 v7, a109\n" \
    "v_cvt_pk_bf16_f32 v108, v6, v7\n" \
    "v_accvgpr_read_b32 v6, a110\n" \
    "v_accvgpr_read_b32 v7, a111\n" \
    "v_cvt_pk_bf16_f32 v109, v6, v7\n" \
    "v_accvgpr_read_b32 v6, a112\n" \
    "v_accvgpr_read_b32 v7, a113\n" \
    "v_cvt_pk_bf16_f32 v112, v6, v7\n" \
    "v_accvgpr_read_b32 v6, a114\n" \
    "v_accvgpr_read_b32 v7, a115\n" \
    "v_cvt_pk_bf16_f32 v113, v6, v7\n" \
    "v_accvgpr_read_b32 v6, a116\n" \
    "v_accvgpr_read_b32 v7, a117\n" \
    "v_cvt_pk_bf16_f32 v28, v6, v7\n" \
    "v_accvgpr_read_b32 v6, a118\n" \
    "v_accvgpr_read_b32 v7, a119\n" \
    "v_cvt_pk_bf16_f32 v29, v6, v7\n" \
    "v_accvgpr_read_b32 v6, a120\n" \
    "v_accvgpr_read_b32 v7, a121\n" \
    "v_cvt_pk_bf16_f32 v36, v6, v7\n" \
    "v_accvgpr_read_b32 v6, a122\n" \
    "v_accvgpr_read_b32 v7, a123\n" \
    "v_cvt_pk_bf16_f32 v37, v6, v7\n" \
    "v_accvgpr_read_b32 v6, a128\n" \
    "v_accvgpr_read_b32 v7, a129\n" \
    "v_cvt_pk_bf16_f32 v110, v6, v7\n" \
    "v_accvgpr_read_b32 v6, a130\n" \
    "v_accvgpr_read_b32 v7, a131\n" \
    "v_cvt_pk_bf16_f32 v111, v6, v7\n" \
    "v_accvgpr_read_b32 v6, a132\n" \
    "v_accvgpr_read_b32 v7, a133\n" \
    "v_cvt_pk_bf16_f32 v114, v6, v7\n" \
    "v_accvgpr_read_b32 v6, a134\n" \
    "v_accvgpr_read_b32 v7, a135\n" \
    "v_cvt_pk_bf16_f32 v115, v6, v7\n" \
    "s_waitcnt lgkmcnt(0)\n" \
    "s_barrier\n" \
    "ds_write_b128 v8, v[26:29]\n" \
    "ds_write_b128 v9, v[34:37]\n" \
    "ds_write_b128 v10, v[108:111]\n" \
    "ds_write_b128 v11, v[112:115]\n" \
    "s_waitcnt lgkmcnt(0)\n" \
    "s_barrier\n" \
    "ds_read_b128 v[108:111], v12\n" \
    "ds_read_b128 v[114:117], v12, offset:256\n" \
    "ds_read_b128 v[118:121], v12, offset:128\n" \
    "ds_read_b128 v[124:127], v12, offset:384\n" \
    "s_and_b32 s5, s2, 0xffff\n" \
    "s_mov_b32 s6, s14\n" \
    "s_mov_b32 s7, s15\n" \
    "s_waitcnt lgkmcnt(3)\n" \
    "v_mov_b32_e32 v112, v108\n" \
    "v_mov_b32_e32 v113, v109\n" \
    "s_waitcnt lgkmcnt(2)\n" \
    "buffer_store_dwordx4 v[112:115], v13, s[4:7], 0, offen\n" \
    "s_waitcnt lgkmcnt(1)\n" \
    "v_mov_b32_e32 v122, v118\n" \
    "v_mov_b32_e32 v123, v119\n" \
    "s_waitcnt lgkmcnt(0)\n" \
    "buffer_store_dwordx4 v[122:125], v14, s[4:7], 0, offen\n" \
    "v_mov_b32_e32 v112, v116\n" \
    "v_mov_b32_e32 v113, v117\n" \
    "buffer_store_dwordx4 v[110:113], v15, s[4:7], 0, offen\n" \
    "v_mov_b32_e32 v122, v126\n" \
    "v_mov_b32_e32 v123, v127\n" \
    "buffer_store_dwordx4 v[120:123], v16, s[4:7], 0, offen\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[42:45], v[128:131], a[252:255], v2, v4, op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[46:49], v[132:135], a[100:103], v2, v4, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[52:55], v[128:131], a[0:3], v2, v4, op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[56:59], v[132:135], a[0:3], v2, v4, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[60:63], v[128:131], a[4:7], v3, v4, op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[64:67], v[132:135], a[4:7], v3, v4, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[68:71], v[128:131], a[8:11], v3, v4, op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[72:75], v[132:135], a[8:11], v3, v4, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[42:45], v[136:139], a[12:15], v2, v4, op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[46:49], v[140:143], a[12:15], v2, v4, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[52:55], v[136:139], a[16:19], v2, v4, op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[56:59], v[140:143], a[16:19], v2, v4, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[60:63], v[136:139], a[20:23], v3, v4, op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[64:67], v[140:143], a[20:23], v3, v4, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[68:71], v[136:139], a[24:27], v3, v4, op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[72:75], v[140:143], a[24:27], v3, v4, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_accvgpr_read_b32 v4, a136\n" \
    "v_accvgpr_read_b32 v7, a137\n" \
    "v_cvt_pk_bf16_f32 v26, v4, v7\n" \
    "v_accvgpr_read_b32 v4, a138\n" \
    "v_accvgpr_read_b32 v7, a139\n" \
    "v_cvt_pk_bf16_f32 v27, v4, v7\n" \
    "v_accvgpr_read_b32 v4, a140\n" \
    "v_accvgpr_read_b32 v7, a141\n" \
    "v_cvt_pk_bf16_f32 v34, v4, v7\n" \
    "v_accvgpr_read_b32 v4, a142\n" \
    "v_accvgpr_read_b32 v7, a143\n" \
    "v_cvt_pk_bf16_f32 v35, v4, v7\n" \
    "v_accvgpr_read_b32 v4, a144\n" \
    "v_accvgpr_read_b32 v7, a145\n" \
    "v_cvt_pk_bf16_f32 v108, v4, v7\n" \
    "v_accvgpr_read_b32 v4, a146\n" \
    "v_accvgpr_read_b32 v7, a147\n" \
    "v_cvt_pk_bf16_f32 v109, v4, v7\n" \
    "v_accvgpr_read_b32 v4, a148\n" \
    "v_accvgpr_read_b32 v7, a149\n" \
    "v_cvt_pk_bf16_f32 v112, v4, v7\n" \
    "v_accvgpr_read_b32 v4, a150\n" \
    "v_accvgpr_read_b32 v7, a151\n" \
    "v_cvt_pk_bf16_f32 v113, v4, v7\n" \
    "v_accvgpr_read_b32 v4, a152\n" \
    "v_accvgpr_read_b32 v7, a153\n" \
    "v_cvt_pk_bf16_f32 v28, v4, v7\n" \
    "v_accvgpr_read_b32 v4, a154\n" \
    "v_accvgpr_read_b32 v7, a155\n" \
    "v_cvt_pk_bf16_f32 v29, v4, v7\n" \
    "v_accvgpr_read_b32 v4, a156\n" \
    "v_accvgpr_read_b32 v7, a157\n" \
    "v_cvt_pk_bf16_f32 v36, v4, v7\n" \
    "v_accvgpr_read_b32 v4, a158\n" \
    "v_accvgpr_read_b32 v7, a159\n" \
    "v_cvt_pk_bf16_f32 v37, v4, v7\n" \
    "v_accvgpr_read_b32 v4, a160\n" \
    "v_accvgpr_read_b32 v7, a161\n" \
    "v_cvt_pk_bf16_f32 v110, v4, v7\n" \
    "v_accvgpr_read_b32 v4, a162\n" \
    "v_accvgpr_read_b32 v7, a163\n" \
    "v_cvt_pk_bf16_f32 v111, v4, v7\n" \
    "v_accvgpr_read_b32 v4, a164\n" \
    "v_accvgpr_read_b32 v7, a165\n" \
    "v_cvt_pk_bf16_f32 v114, v4, v7\n" \
    "v_accvgpr_read_b32 v4, a166\n" \
    "v_accvgpr_read_b32 v7, a167\n" \
    "v_cvt_pk_bf16_f32 v115, v4, v7\n" \
    "s_waitcnt lgkmcnt(0)\n" \
    "s_barrier\n" \
    "ds_write_b128 v8, v[26:29]\n" \
    "ds_write_b128 v9, v[34:37]\n" \
    "ds_write_b128 v10, v[108:111]\n" \
    "ds_write_b128 v11, v[112:115]\n" \
    "s_waitcnt lgkmcnt(0)\n" \
    "s_barrier\n" \
    "ds_read_b128 v[108:111], v12\n" \
    "ds_read_b128 v[114:117], v12, offset:256\n" \
    "ds_read_b128 v[118:121], v12, offset:128\n" \
    "ds_read_b128 v[124:127], v12, offset:384\n" \
    "s_and_b32 s1, s1, 0xffff\n" \
    "s_mov_b32 s2, s14\n" \
    "s_mov_b32 s3, s15\n" \
    "s_waitcnt lgkmcnt(3)\n" \
    "v_mov_b32_e32 v112, v108\n" \
    "v_mov_b32_e32 v113, v109\n" \
    "s_waitcnt lgkmcnt(2)\n" \
    "buffer_store_dwordx4 v[112:115], v13, s[0:3], 0, offen\n" \
    "s_waitcnt lgkmcnt(1)\n" \
    "v_mov_b32_e32 v122, v118\n" \
    "v_mov_b32_e32 v123, v119\n" \
    "s_waitcnt lgkmcnt(0)\n" \
    "buffer_store_dwordx4 v[122:125], v14, s[0:3], 0, offen\n" \
    "v_mov_b32_e32 v112, v116\n" \
    "v_mov_b32_e32 v113, v117\n" \
    "buffer_store_dwordx4 v[110:113], v15, s[0:3], 0, offen\n" \
    "v_mov_b32_e32 v122, v126\n" \
    "v_mov_b32_e32 v123, v127\n" \
    "buffer_store_dwordx4 v[120:123], v16, s[0:3], 0, offen\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[42:45], v[100:103], a[124:127], v2, v5, op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[46:49], v[104:107], a[104:107], v2, v5, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_accvgpr_write_b32 a108, v190\n" \
    "v_accvgpr_write_b32 a109, v191\n" \
    "v_accvgpr_write_b32 a110, v192\n" \
    "v_accvgpr_write_b32 a111, v193\n" \
    "s_nop 1\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[52:55], v[100:103], a[108:111], v2, v5, op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[56:59], v[104:107], a[108:111], v2, v5, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[60:63], v[100:103], a[28:31], v3, v5, op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[64:67], v[104:107], a[28:31], v3, v5, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[68:71], v[100:103], a[44:47], v3, v5, op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[72:75], v[104:107], a[44:47], v3, v5, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[42:45], v[92:95], a[32:35], v2, v5, op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[46:49], v[96:99], a[32:35], v2, v5, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[52:55], v[92:95], a[236:239], v2, v5, op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[56:59], v[96:99], a[112:115], v2, v5, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[60:63], v[92:95], a[36:39], v3, v5, op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[64:67], v[96:99], a[36:39], v3, v5, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[68:71], v[92:95], a[244:247], v3, v5, op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[72:75], v[96:99], a[116:119], v3, v5, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_accvgpr_read_b32 v4, a100\n" \
    "v_accvgpr_read_b32 v5, a101\n" \
    "v_cvt_pk_bf16_f32 v4, v4, v5\n" \
    "v_accvgpr_read_b32 v6, a102\n" \
    "v_accvgpr_read_b32 v5, a103\n" \
    "v_cvt_pk_bf16_f32 v5, v6, v5\n" \
    "v_accvgpr_read_b32 v6, a0\n" \
    "v_accvgpr_read_b32 v7, a1\n" \
    "v_cvt_pk_bf16_f32 v26, v6, v7\n" \
    "v_accvgpr_read_b32 v6, a2\n" \
    "v_accvgpr_read_b32 v7, a3\n" \
    "v_cvt_pk_bf16_f32 v27, v6, v7\n" \
    "v_accvgpr_read_b32 v6, a4\n" \
    "v_accvgpr_read_b32 v7, a5\n" \
    "v_cvt_pk_bf16_f32 v34, v6, v7\n" \
    "v_accvgpr_read_b32 v6, a6\n" \
    "v_accvgpr_read_b32 v7, a7\n" \
    "v_cvt_pk_bf16_f32 v35, v6, v7\n" \
    "v_accvgpr_read_b32 v6, a8\n" \
    "v_accvgpr_read_b32 v7, a9\n" \
    "v_cvt_pk_bf16_f32 v92, v6, v7\n" \
    "v_accvgpr_read_b32 v6, a10\n" \
    "v_accvgpr_read_b32 v7, a11\n" \
    "v_cvt_pk_bf16_f32 v93, v6, v7\n" \
    "v_accvgpr_read_b32 v6, a12\n" \
    "v_accvgpr_read_b32 v7, a13\n" \
    "v_cvt_pk_bf16_f32 v6, v6, v7\n" \
    "v_accvgpr_read_b32 v28, a14\n" \
    "v_accvgpr_read_b32 v7, a15\n" \
    "v_cvt_pk_bf16_f32 v7, v28, v7\n" \
    "v_accvgpr_read_b32 v28, a16\n" \
    "v_accvgpr_read_b32 v17, a17\n" \
    "v_cvt_pk_bf16_f32 v28, v28, v17\n" \
    "v_accvgpr_read_b32 v36, a18\n" \
    "v_accvgpr_read_b32 v17, a19\n" \
    "v_cvt_pk_bf16_f32 v29, v36, v17\n" \
    "v_accvgpr_read_b32 v36, a20\n" \
    "v_accvgpr_read_b32 v17, a21\n" \
    "v_cvt_pk_bf16_f32 v36, v36, v17\n" \
    "v_accvgpr_read_b32 v50, a22\n" \
    "v_accvgpr_read_b32 v17, a23\n" \
    "v_cvt_pk_bf16_f32 v37, v50, v17\n" \
    "v_accvgpr_read_b32 v50, a24\n" \
    "v_accvgpr_read_b32 v17, a25\n" \
    "v_cvt_pk_bf16_f32 v94, v50, v17\n" \
    "v_accvgpr_read_b32 v50, a26\n" \
    "v_accvgpr_read_b32 v17, a27\n" \
    "v_cvt_pk_bf16_f32 v95, v50, v17\n" \
    "s_waitcnt lgkmcnt(0)\n" \
    "s_barrier\n" \
    "ds_write_b128 v8, v[4:7]\n" \
    "ds_write_b128 v9, v[26:29]\n" \
    "ds_write_b128 v10, v[34:37]\n" \
    "ds_write_b128 v11, v[92:95]\n" \
    "s_waitcnt lgkmcnt(0)\n" \
    "s_barrier\n" \
    "ds_read_b128 v[92:95], v12\n" \
    "ds_read_b128 v[98:101], v12, offset:256\n" \
    "ds_read_b128 v[102:105], v12, offset:128\n" \
    "ds_read_b128 v[108:111], v12, offset:384\n" \
    "s_waitcnt lgkmcnt(3)\n" \
    "v_mov_b32_e32 v96, v92\n" \
    "v_mov_b32_e32 v97, v93\n" \
    "s_waitcnt lgkmcnt(2)\n" \
    "buffer_store_dwordx4 v[96:99], v13, s[12:15], 0, offen, offset:256\n" \
    "s_waitcnt lgkmcnt(1)\n" \
    "v_mov_b32_e32 v106, v102\n" \
    "v_mov_b32_e32 v107, v103\n" \
    "s_waitcnt lgkmcnt(0)\n" \
    "buffer_store_dwordx4 v[106:109], v14, s[12:15], 0, offen, offset:256\n" \
    "v_mov_b32_e32 v96, v100\n" \
    "v_mov_b32_e32 v97, v101\n" \
    "buffer_store_dwordx4 v[94:97], v15, s[12:15], 0, offen, offset:256\n" \
    "v_mov_b32_e32 v106, v110\n" \
    "v_mov_b32_e32 v107, v111\n" \
    "buffer_store_dwordx4 v[104:107], v16, s[12:15], 0, offen, offset:256\n" \
    "v_accvgpr_write_b32 a0, v186\n" \
    "v_accvgpr_write_b32 a1, v187\n" \
    "v_accvgpr_write_b32 a2, v188\n" \
    "v_accvgpr_write_b32 a3, v189\n" \
    "s_nop 1\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[42:45], v[84:87], a[0:3], v2, v0, op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[46:49], v[88:91], a[0:3], v2, v0, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[52:55], v[84:87], a[64:67], v2, v0, op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[56:59], v[88:91], a[4:7], v2, v0, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[60:63], v[84:87], a[40:43], v3, v0, op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[64:67], v[88:91], a[8:11], v3, v0, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[68:71], v[84:87], a[68:71], v3, v0, op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[72:75], v[88:91], a[12:15], v3, v0, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[42:45], v[76:79], a[48:51], v2, v0, op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[46:49], v[80:83], a[16:19], v2, v0, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[52:55], v[76:79], a[52:55], v2, v0, op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[56:59], v[80:83], a[20:23], v2, v0, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[60:63], v[76:79], a[56:59], v3, v0, op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[64:67], v[80:83], a[24:27], v3, v0, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[68:71], v[76:79], a[60:63], v3, v0, op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[72:75], v[80:83], a[40:43], v3, v0, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_accvgpr_read_b32 v0, a104\n" \
    "v_accvgpr_read_b32 v5, a105\n" \
    "v_cvt_pk_bf16_f32 v4, v0, v5\n" \
    "v_accvgpr_read_b32 v0, a106\n" \
    "v_accvgpr_read_b32 v5, a107\n" \
    "v_cvt_pk_bf16_f32 v5, v0, v5\n" \
    "v_accvgpr_read_b32 v0, a108\n" \
    "v_accvgpr_read_b32 v7, a109\n" \
    "v_cvt_pk_bf16_f32 v26, v0, v7\n" \
    "v_accvgpr_read_b32 v0, a110\n" \
    "v_accvgpr_read_b32 v7, a111\n" \
    "v_cvt_pk_bf16_f32 v27, v0, v7\n" \
    "v_accvgpr_read_b32 v0, a28\n" \
    "v_accvgpr_read_b32 v7, a29\n" \
    "v_cvt_pk_bf16_f32 v34, v0, v7\n" \
    "v_accvgpr_read_b32 v0, a30\n" \
    "v_accvgpr_read_b32 v7, a31\n" \
    "v_cvt_pk_bf16_f32 v35, v0, v7\n" \
    "v_accvgpr_read_b32 v0, a44\n" \
    "v_accvgpr_read_b32 v7, a45\n" \
    "v_cvt_pk_bf16_f32 v76, v0, v7\n" \
    "v_accvgpr_read_b32 v0, a46\n" \
    "v_accvgpr_read_b32 v7, a47\n" \
    "v_cvt_pk_bf16_f32 v77, v0, v7\n" \
    "v_accvgpr_read_b32 v0, a32\n" \
    "v_accvgpr_read_b32 v7, a33\n" \
    "v_cvt_pk_bf16_f32 v6, v0, v7\n" \
    "v_accvgpr_read_b32 v0, a34\n" \
    "v_accvgpr_read_b32 v7, a35\n" \
    "v_cvt_pk_bf16_f32 v7, v0, v7\n" \
    "v_accvgpr_read_b32 v0, a112\n" \
    "v_accvgpr_read_b32 v17, a113\n" \
    "v_cvt_pk_bf16_f32 v28, v0, v17\n" \
    "v_accvgpr_read_b32 v0, a114\n" \
    "v_accvgpr_read_b32 v17, a115\n" \
    "v_cvt_pk_bf16_f32 v29, v0, v17\n" \
    "v_accvgpr_read_b32 v0, a36\n" \
    "v_accvgpr_read_b32 v17, a37\n" \
    "v_cvt_pk_bf16_f32 v36, v0, v17\n" \
    "v_accvgpr_read_b32 v0, a38\n" \
    "v_accvgpr_read_b32 v17, a39\n" \
    "v_cvt_pk_bf16_f32 v37, v0, v17\n" \
    "v_accvgpr_read_b32 v0, a116\n" \
    "v_accvgpr_read_b32 v17, a117\n" \
    "v_cvt_pk_bf16_f32 v78, v0, v17\n" \
    "v_accvgpr_read_b32 v0, a118\n" \
    "v_accvgpr_read_b32 v17, a119\n" \
    "v_cvt_pk_bf16_f32 v79, v0, v17\n" \
    "s_waitcnt lgkmcnt(0)\n" \
    "s_barrier\n" \
    "ds_write_b128 v8, v[4:7]\n" \
    "ds_write_b128 v9, v[26:29]\n" \
    "ds_write_b128 v10, v[34:37]\n" \
    "ds_write_b128 v11, v[76:79]\n" \
    "s_waitcnt lgkmcnt(0)\n" \
    "s_barrier\n" \
    "ds_read_b128 v[76:79], v12\n" \
    "ds_read_b128 v[82:85], v12, offset:256\n" \
    "ds_read_b128 v[86:89], v12, offset:128\n" \
    "ds_read_b128 v[92:95], v12, offset:384\n" \
    "s_waitcnt lgkmcnt(3)\n" \
    "v_mov_b32_e32 v80, v76\n" \
    "v_mov_b32_e32 v81, v77\n" \
    "s_waitcnt lgkmcnt(2)\n" \
    "buffer_store_dwordx4 v[80:83], v13, s[8:11], 0, offen, offset:256\n" \
    "s_waitcnt lgkmcnt(1)\n" \
    "v_mov_b32_e32 v90, v86\n" \
    "v_mov_b32_e32 v91, v87\n" \
    "s_waitcnt lgkmcnt(0)\n" \
    "buffer_store_dwordx4 v[90:93], v14, s[8:11], 0, offen, offset:256\n" \
    "v_mov_b32_e32 v80, v84\n" \
    "v_mov_b32_e32 v81, v85\n" \
    "buffer_store_dwordx4 v[78:81], v15, s[8:11], 0, offen, offset:256\n" \
    "v_mov_b32_e32 v90, v94\n" \
    "v_mov_b32_e32 v91, v95\n" \
    "buffer_store_dwordx4 v[88:91], v16, s[8:11], 0, offen, offset:256\n" \
    "v_accvgpr_write_b32 a28, v182\n" \
    "v_accvgpr_write_b32 a29, v183\n" \
    "v_accvgpr_write_b32 a30, v184\n" \
    "v_accvgpr_write_b32 a31, v185\n" \
    "s_nop 1\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[42:45], v[30:33], a[28:31], v2, v1, op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[46:49], v[38:41], a[28:31], v2, v1, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[52:55], v[30:33], a[96:99], v2, v1, op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[56:59], v[38:41], a[32:35], v2, v1, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[60:63], v[30:33], a[72:75], v3, v1, op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[64:67], v[38:41], a[36:39], v3, v1, op_sel:[1,1,0], op_sel_hi:[0,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[68:71], v[30:33], a[76:79], v3, v1, op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[72:75], v[38:41], a[44:47], v3, v1, op_sel:[1,1,0], op_sel_hi:[1,0,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[42:45], v[18:21], a[80:83], v2, v1, op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[46:49], v[22:25], a[48:51], v2, v1, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[52:55], v[18:21], a[84:87], v2, v1, op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[56:59], v[22:25], a[52:55], v2, v1, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[60:63], v[18:21], a[88:91], v3, v1, op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[64:67], v[22:25], a[56:59], v3, v1, op_sel:[1,1,0], op_sel_hi:[0,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[68:71], v[18:21], a[92:95], v3, v1, op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[72:75], v[22:25], a[60:63], v3, v1, op_sel:[1,1,0], op_sel_hi:[1,1,0], cbsz:4, blgp:4\n" \
    "v_accvgpr_read_b32 v0, a0\n" \
    "v_accvgpr_read_b32 v1, a1\n" \
    "v_cvt_pk_bf16_f32 v0, v0, v1\n" \
    "v_accvgpr_read_b32 v2, a2\n" \
    "v_accvgpr_read_b32 v1, a3\n" \
    "v_cvt_pk_bf16_f32 v1, v2, v1\n" \
    "v_accvgpr_read_b32 v2, a4\n" \
    "v_accvgpr_read_b32 v3, a5\n" \
    "v_cvt_pk_bf16_f32 v4, v2, v3\n" \
    "v_accvgpr_read_b32 v2, a6\n" \
    "v_accvgpr_read_b32 v3, a7\n" \
    "v_cvt_pk_bf16_f32 v5, v2, v3\n" \
    "v_accvgpr_read_b32 v2, a8\n" \
    "v_accvgpr_read_b32 v3, a9\n" \
    "v_cvt_pk_bf16_f32 v18, v2, v3\n" \
    "v_accvgpr_read_b32 v2, a10\n" \
    "v_accvgpr_read_b32 v3, a11\n" \
    "v_cvt_pk_bf16_f32 v19, v2, v3\n" \
    "v_accvgpr_read_b32 v2, a12\n" \
    "v_accvgpr_read_b32 v3, a13\n" \
    "v_cvt_pk_bf16_f32 v22, v2, v3\n" \
    "v_accvgpr_read_b32 v2, a14\n" \
    "v_accvgpr_read_b32 v3, a15\n" \
    "v_cvt_pk_bf16_f32 v23, v2, v3\n" \
    "v_accvgpr_read_b32 v2, a16\n" \
    "v_accvgpr_read_b32 v3, a17\n" \
    "v_cvt_pk_bf16_f32 v2, v2, v3\n" \
    "v_accvgpr_read_b32 v6, a18\n" \
    "v_accvgpr_read_b32 v3, a19\n" \
    "v_cvt_pk_bf16_f32 v3, v6, v3\n" \
    "v_accvgpr_read_b32 v6, a20\n" \
    "v_accvgpr_read_b32 v7, a21\n" \
    "v_cvt_pk_bf16_f32 v6, v6, v7\n" \
    "v_accvgpr_read_b32 v20, a22\n" \
    "v_accvgpr_read_b32 v7, a23\n" \
    "v_cvt_pk_bf16_f32 v7, v20, v7\n" \
    "v_accvgpr_read_b32 v20, a24\n" \
    "v_accvgpr_read_b32 v17, a25\n" \
    "v_cvt_pk_bf16_f32 v20, v20, v17\n" \
    "v_accvgpr_read_b32 v24, a26\n" \
    "v_accvgpr_read_b32 v17, a27\n" \
    "v_cvt_pk_bf16_f32 v21, v24, v17\n" \
    "v_accvgpr_read_b32 v24, a40\n" \
    "v_accvgpr_read_b32 v17, a41\n" \
    "v_cvt_pk_bf16_f32 v24, v24, v17\n" \
    "v_accvgpr_read_b32 v26, a42\n" \
    "v_accvgpr_read_b32 v17, a43\n" \
    "v_cvt_pk_bf16_f32 v25, v26, v17\n" \
    "s_waitcnt lgkmcnt(0)\n" \
    "s_barrier\n" \
    "ds_write_b128 v8, v[0:3]\n" \
    "ds_write_b128 v9, v[4:7]\n" \
    "ds_write_b128 v10, v[18:21]\n" \
    "ds_write_b128 v11, v[22:25]\n" \
    "s_waitcnt lgkmcnt(0)\n" \
    "s_barrier\n" \
    "ds_read_b128 v[0:3], v12\n" \
    "ds_read_b128 v[20:23], v12, offset:256\n" \
    "ds_read_b128 v[24:27], v12, offset:128\n" \
    "ds_read_b128 v[30:33], v12, offset:384\n" \
    "s_waitcnt lgkmcnt(3)\n" \
    "v_mov_b32_e32 v18, v0\n" \
    "v_mov_b32_e32 v19, v1\n" \
    "s_waitcnt lgkmcnt(2)\n" \
    "buffer_store_dwordx4 v[18:21], v13, s[4:7], 0, offen, offset:256\n" \
    "s_waitcnt lgkmcnt(1)\n" \
    "v_mov_b32_e32 v28, v24\n" \
    "v_mov_b32_e32 v29, v25\n" \
    "s_waitcnt lgkmcnt(0)\n" \
    "buffer_store_dwordx4 v[28:31], v14, s[4:7], 0, offen, offset:256\n" \
    "v_mov_b32_e32 v4, v22\n" \
    "v_mov_b32_e32 v5, v23\n" \
    "buffer_store_dwordx4 v[2:5], v15, s[4:7], 0, offen, offset:256\n" \
    "v_mov_b32_e32 v28, v32\n" \
    "v_mov_b32_e32 v29, v33\n" \
    "buffer_store_dwordx4 v[26:29], v16, s[4:7], 0, offen, offset:256\n" \
    "v_accvgpr_read_b32 v0, a28\n" \
    "v_accvgpr_read_b32 v1, a29\n" \
    "v_cvt_pk_bf16_f32 v0, v0, v1\n" \
    "v_accvgpr_read_b32 v2, a30\n" \
    "v_accvgpr_read_b32 v1, a31\n" \
    "v_cvt_pk_bf16_f32 v1, v2, v1\n" \
    "v_accvgpr_read_b32 v2, a32\n" \
    "v_accvgpr_read_b32 v3, a33\n" \
    "v_cvt_pk_bf16_f32 v4, v2, v3\n" \
    "v_accvgpr_read_b32 v2, a34\n" \
    "v_accvgpr_read_b32 v3, a35\n" \
    "v_cvt_pk_bf16_f32 v5, v2, v3\n" \
    "v_accvgpr_read_b32 v2, a36\n" \
    "v_accvgpr_read_b32 v3, a37\n" \
    "v_cvt_pk_bf16_f32 v18, v2, v3\n" \
    "v_accvgpr_read_b32 v2, a38\n" \
    "v_accvgpr_read_b32 v3, a39\n" \
    "v_cvt_pk_bf16_f32 v19, v2, v3\n" \
    "v_accvgpr_read_b32 v2, a44\n" \
    "v_accvgpr_read_b32 v3, a45\n" \
    "v_cvt_pk_bf16_f32 v22, v2, v3\n" \
    "v_accvgpr_read_b32 v2, a46\n" \
    "v_accvgpr_read_b32 v3, a47\n" \
    "v_cvt_pk_bf16_f32 v23, v2, v3\n" \
    "v_accvgpr_read_b32 v2, a48\n" \
    "v_accvgpr_read_b32 v3, a49\n" \
    "v_cvt_pk_bf16_f32 v2, v2, v3\n" \
    "v_accvgpr_read_b32 v6, a50\n" \
    "v_accvgpr_read_b32 v3, a51\n" \
    "v_cvt_pk_bf16_f32 v3, v6, v3\n" \
    "v_accvgpr_read_b32 v6, a52\n" \
    "v_accvgpr_read_b32 v7, a53\n" \
    "v_cvt_pk_bf16_f32 v6, v6, v7\n" \
    "v_accvgpr_read_b32 v20, a54\n" \
    "v_accvgpr_read_b32 v7, a55\n" \
    "v_cvt_pk_bf16_f32 v7, v20, v7\n" \
    "v_accvgpr_read_b32 v20, a56\n" \
    "v_accvgpr_read_b32 v17, a57\n" \
    "v_cvt_pk_bf16_f32 v20, v20, v17\n" \
    "v_accvgpr_read_b32 v24, a58\n" \
    "v_accvgpr_read_b32 v17, a59\n" \
    "v_cvt_pk_bf16_f32 v21, v24, v17\n" \
    "v_accvgpr_read_b32 v24, a60\n" \
    "v_accvgpr_read_b32 v17, a61\n" \
    "v_cvt_pk_bf16_f32 v24, v24, v17\n" \
    "v_accvgpr_read_b32 v26, a62\n" \
    "v_accvgpr_read_b32 v17, a63\n" \
    "v_cvt_pk_bf16_f32 v25, v26, v17\n" \
    "s_waitcnt lgkmcnt(0)\n" \
    "s_barrier\n" \
    "ds_write_b128 v8, v[0:3]\n" \
    "ds_write_b128 v9, v[4:7]\n" \
    "ds_write_b128 v10, v[18:21]\n" \
    "ds_write_b128 v11, v[22:25]\n" \
    "s_waitcnt lgkmcnt(0)\n" \
    "s_barrier\n" \
    "ds_read_b128 v[0:3], v12\n" \
    "ds_read_b128 v[6:9], v12, offset:256\n" \
    "ds_read_b128 v[18:21], v12, offset:128\n" \
    "ds_read_b128 v[24:27], v12, offset:384\n" \
    "s_waitcnt lgkmcnt(3)\n" \
    "v_mov_b32_e32 v4, v0\n" \
    "v_mov_b32_e32 v5, v1\n" \
    "s_waitcnt lgkmcnt(2)\n" \
    "buffer_store_dwordx4 v[4:7], v13, s[0:3], 0, offen, offset:256\n" \
    "s_waitcnt lgkmcnt(1)\n" \
    "v_mov_b32_e32 v22, v18\n" \
    "v_mov_b32_e32 v23, v19\n" \
    "s_waitcnt lgkmcnt(0)\n" \
    "buffer_store_dwordx4 v[22:25], v14, s[0:3], 0, offen, offset:256\n" \
    "v_mov_b32_e32 v4, v8\n" \
    "v_mov_b32_e32 v5, v9\n" \
    "buffer_store_dwordx4 v[2:5], v15, s[0:3], 0, offen, offset:256\n" \
    "v_mov_b32_e32 v22, v26\n" \
    "v_mov_b32_e32 v23, v27\n" \
    "buffer_store_dwordx4 v[20:23], v16, s[0:3], 0, offen, offset:256\n"
