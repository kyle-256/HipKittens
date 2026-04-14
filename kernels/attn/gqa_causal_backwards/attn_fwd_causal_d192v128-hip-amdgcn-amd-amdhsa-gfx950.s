	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.text
	.protected	_Z10attend_ker12attn_globals ; -- Begin function _Z10attend_ker12attn_globals
	.globl	_Z10attend_ker12attn_globals
	.p2align	8
	.type	_Z10attend_ker12attn_globals,@function
_Z10attend_ker12attn_globals:           ; @_Z10attend_ker12attn_globals
; %bb.0:
	s_cmp_lg_u32 0, -1
	s_cselect_b32 s5, 0, 0
	s_and_b32 s8, s5, -16
	s_mov_b32 s7, 0
	s_and_b32 s6, s5, 15
	s_add_u32 s8, s8, 16
	s_load_dwordx2 s[42:43], s[0:1], 0x30
	s_load_dwordx4 s[28:31], s[0:1], 0x40
	s_load_dword s33, s[0:1], 0x50
	s_load_dwordx4 s[20:23], s[0:1], 0x70
	s_cmp_eq_u64 s[6:7], 0
	s_waitcnt lgkmcnt(0)
	s_cselect_b32 s21, s5, s8
	s_add_u32 s5, s21, 0xc000
	s_and_b32 s8, s5, -16
	s_and_b32 s6, s5, 15
	s_add_u32 s8, s8, 16
	v_lshrrev_b32_e32 v1, 2, v0
	s_cmp_eq_u64 s[6:7], 0
	v_lshlrev_b32_e32 v2, 4, v0
	v_and_b32_e32 v3, 32, v0
	v_and_b32_e32 v4, 16, v1
	s_cselect_b32 s45, s5, s8
	s_lshl_b32 s5, s2, 3
	v_bitop3_b32 v3, v4, v2, v3 bitop3:0x36
	s_and_b32 s5, s5, 56
	s_lshr_b32 s2, s2, 3
	v_lshrrev_b32_e32 v3, 1, v3
	s_add_i32 s5, s5, s2
	v_and_b32_e32 v3, 24, v3
	s_movk_i32 s2, 0x60
	v_and_or_b32 v5, v1, s2, v3
	v_or_b32_e32 v7, 0x2000, v2
	s_movk_i32 s2, 0x2fff
	v_cmp_lt_u32_e32 vcc, s2, v7
	s_mul_i32 s8, s30, s33
	v_bfe_u32 v4, v0, 2, 5
	v_cndmask_b32_e64 v8, 0, 32, vcc
	s_load_dword s23, s[0:1], 0x80
	v_mul_lo_u32 v6, v4, s8
	v_or_b32_e32 v4, v8, v4
	v_lshrrev_b32_e32 v8, 6, v7
	s_movk_i32 s6, 0x3000
	v_and_b32_e32 v8, 0xe0, v8
	s_mul_i32 s44, s4, s28
	s_lshr_b32 s31, s5, 3
	v_add_u32_e32 v9, 0x7fffff40, v8
	v_cmp_gt_u32_e32 vcc, s6, v7
	s_mul_i32 s6, s44, s30
	s_add_i32 s6, s6, s31
	v_cndmask_b32_e32 v7, v9, v8, vcc
	v_or_b32_e32 v3, v7, v3
	v_mul_lo_u32 v4, v4, s8
	s_lshl_b32 s2, s8, 5
	s_mul_i32 s6, s6, s33
	v_add_lshl_u32 v7, v3, v4, 1
	v_add3_u32 v3, v6, s2, v5
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s2, s22, s23
	s_ashr_i32 s7, s6, 31
	v_and_b32_e32 v35, 0x1c00, v2
	v_mov_b32_e32 v4, 0x80
	s_lshl_b32 s24, s2, 5
	s_lshl_b64 s[6:7], s[6:7], 1
	v_add_u32_e32 v2, s21, v35
	v_lshl_add_u32 v3, v3, 1, v4
	s_add_u32 s36, s42, s6
	v_readfirstlane_b32 s6, v2
	v_add_u32_e32 v4, 0x2000, v2
	v_add_lshl_u32 v10, v5, v6, 1
	s_addc_u32 s37, s43, s7
	s_lshl_b32 s38, s8, 7
	s_mov_b32 s39, 0x110000
	s_mov_b32 m0, s6
	v_readfirstlane_b32 s6, v4
	scratch_store_dword off, v2, off offset:236 ; 4-byte Folded Spill
	v_add_u32_e32 v2, 0x4000, v2
	buffer_load_dwordx4 v10, s[36:39], 0 offen lds
	s_mov_b32 m0, s6
	v_readfirstlane_b32 s6, v2
	buffer_load_dwordx4 v7, s[36:39], 0 offen lds
	s_mov_b32 m0, s6
	scratch_store_dword off, v10, off       ; 4-byte Folded Spill
	buffer_load_dwordx4 v3, s[36:39], 0 offen lds
	scratch_store_dword off, v4, off offset:240 ; 4-byte Folded Spill
	scratch_store_dword off, v7, off offset:4 ; 4-byte Folded Spill
	scratch_store_dword off, v2, off offset:244 ; 4-byte Folded Spill
	scratch_store_dword off, v3, off offset:8 ; 4-byte Folded Spill
	s_load_dwordx8 s[48:55], s[0:1], 0x0
	s_load_dword s25, s[0:1], 0x20
	s_load_dwordx2 s[40:41], s[0:1], 0x60
	s_load_dwordx8 s[8:15], s[0:1], 0x90
	s_load_dwordx2 s[34:35], s[0:1], 0xb0
	s_load_dwordx2 s[6:7], s[0:1], 0xc0
	s_load_dwordx2 s[28:29], s[0:1], 0xe0
	s_load_dwordx4 s[16:19], s[0:1], 0xd0
	v_lshrrev_b32_e32 v4, 4, v0
	v_bfe_u32 v6, v0, 2, 3
	v_lshlrev_b32_e32 v5, 3, v0
	v_and_or_b32 v6, v4, 24, v6
	v_and_b32_e32 v4, 0x60, v0
	v_lshrrev_b32_e32 v34, 6, v0
	v_and_or_b32 v4, v5, 24, v4
	v_lshl_or_b32 v36, s3, 3, v34
	v_mad_u64_u32 v[4:5], s[0:1], v6, s2, v[4:5]
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v2, s48
	v_mov_b32_e32 v3, s49
	v_lshlrev_b32_e32 v103, 5, v36
	v_lshlrev_b32_e32 v105, 1, v4
	v_add_lshl_u32 v110, v4, s24, 1
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	; sched_barrier mask(0x00000000)
	s_mul_i32 s0, s4, s52
	v_add_u32_e32 v4, s0, v103
	v_mul_lo_u32 v4, v4, s54
	v_add_u32_e32 v4, s5, v4
	s_mul_i32 s0, s54, s25
	s_mul_i32 s1, s50, s52
	v_mul_lo_u32 v4, v4, s25
	v_and_b32_e32 v104, 31, v0
	s_mul_i32 s1, s0, s1
	v_ashrrev_i32_e32 v5, 31, v4
	v_and_b32_e32 v6, 8, v1
	s_lshl_b32 s1, s1, 1
	v_mul_lo_u32 v7, v104, s0
	v_lshl_add_u64 v[2:3], v[4:5], 1, v[2:3]
	v_mov_b32_e32 v4, s1
	v_mov_b32_e32 v5, 0x20000
	v_add_lshl_u32 v14, v7, v6, 1
	s_mov_b64 s[36:37], exec
	s_barrier
.LBB0_1:                                ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s24, v2
	v_readfirstlane_b32 s25, v3
	v_readfirstlane_b32 s26, v4
	v_readfirstlane_b32 s27, v5
	v_cmp_eq_u64_e32 vcc, s[24:25], v[2:3]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[26:27], v[4:5]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_load_dwordx4 v[6:9], v14, s[24:27], 0 offen
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_1
; %bb.2:
	s_mov_b64 exec, s[36:37]
	s_mov_b64 s[36:37], exec
.LBB0_3:                                ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s24, v2
	v_readfirstlane_b32 s25, v3
	v_readfirstlane_b32 s26, v4
	v_readfirstlane_b32 s27, v5
	v_cmp_eq_u64_e32 vcc, s[24:25], v[2:3]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[26:27], v[4:5]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_load_dwordx4 v[10:13], v14, s[24:27], 0 offen offset:32
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_3
; %bb.4:
	s_mov_b64 exec, s[36:37]
	s_waitcnt vmcnt(1)
	v_lshlrev_b32_e32 v15, 16, v6
	v_and_b32_e32 v16, 0xffff0000, v6
	v_lshlrev_b32_e32 v17, 16, v7
	v_and_b32_e32 v18, 0xffff0000, v7
	v_lshlrev_b32_e32 v19, 16, v8
	v_and_b32_e32 v20, 0xffff0000, v8
	v_lshlrev_b32_e32 v21, 16, v9
	v_and_b32_e32 v22, 0xffff0000, v9
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v23, 16, v10
	v_and_b32_e32 v10, 0xffff0000, v10
	v_lshlrev_b32_e32 v24, 16, v11
	v_and_b32_e32 v11, 0xffff0000, v11
	v_lshlrev_b32_e32 v25, 16, v12
	v_and_b32_e32 v12, 0xffff0000, v12
	v_lshlrev_b32_e32 v26, 16, v13
	v_and_b32_e32 v13, 0xffff0000, v13
	s_mov_b64 s[36:37], exec
.LBB0_5:                                ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s24, v2
	v_readfirstlane_b32 s25, v3
	v_readfirstlane_b32 s26, v4
	v_readfirstlane_b32 s27, v5
	v_cmp_eq_u64_e32 vcc, s[24:25], v[2:3]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[26:27], v[4:5]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_load_dwordx4 v[6:9], v14, s[24:27], 0 offen offset:64
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_5
; %bb.6:
	s_mov_b64 exec, s[36:37]
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v27, 16, v6
	v_and_b32_e32 v28, 0xffff0000, v6
	v_lshlrev_b32_e32 v29, 16, v7
	v_and_b32_e32 v30, 0xffff0000, v7
	v_lshlrev_b32_e32 v31, 16, v8
	v_and_b32_e32 v32, 0xffff0000, v8
	v_lshlrev_b32_e32 v33, 16, v9
	v_and_b32_e32 v37, 0xffff0000, v9
	s_mov_b64 s[36:37], exec
.LBB0_7:                                ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s24, v2
	v_readfirstlane_b32 s25, v3
	v_readfirstlane_b32 s26, v4
	v_readfirstlane_b32 s27, v5
	v_cmp_eq_u64_e32 vcc, s[24:25], v[2:3]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[26:27], v[4:5]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_load_dwordx4 v[6:9], v14, s[24:27], 0 offen offset:96
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_7
; %bb.8:
	s_mov_b64 exec, s[36:37]
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v38, 16, v6
	v_and_b32_e32 v39, 0xffff0000, v6
	v_lshlrev_b32_e32 v40, 16, v7
	v_and_b32_e32 v41, 0xffff0000, v7
	v_lshlrev_b32_e32 v42, 16, v8
	v_and_b32_e32 v43, 0xffff0000, v8
	v_lshlrev_b32_e32 v44, 16, v9
	v_and_b32_e32 v45, 0xffff0000, v9
	s_mov_b64 s[36:37], exec
.LBB0_9:                                ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s24, v2
	v_readfirstlane_b32 s25, v3
	v_readfirstlane_b32 s26, v4
	v_readfirstlane_b32 s27, v5
	v_cmp_eq_u64_e32 vcc, s[24:25], v[2:3]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[26:27], v[4:5]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_load_dwordx4 v[6:9], v14, s[24:27], 0 offen offset:128
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_9
; %bb.10:
	s_mov_b64 exec, s[36:37]
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v46, 16, v6
	v_and_b32_e32 v47, 0xffff0000, v6
	v_lshlrev_b32_e32 v48, 16, v7
	v_and_b32_e32 v49, 0xffff0000, v7
	v_lshlrev_b32_e32 v50, 16, v8
	v_and_b32_e32 v51, 0xffff0000, v8
	v_lshlrev_b32_e32 v52, 16, v9
	v_and_b32_e32 v53, 0xffff0000, v9
	s_mov_b64 s[36:37], exec
.LBB0_11:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s24, v2
	v_readfirstlane_b32 s25, v3
	v_readfirstlane_b32 s26, v4
	v_readfirstlane_b32 s27, v5
	v_cmp_eq_u64_e32 vcc, s[24:25], v[2:3]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[26:27], v[4:5]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_load_dwordx4 v[6:9], v14, s[24:27], 0 offen offset:160
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_11
; %bb.12:
	s_mov_b64 exec, s[36:37]
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v54, 16, v6
	v_and_b32_e32 v55, 0xffff0000, v6
	v_lshlrev_b32_e32 v56, 16, v7
	v_and_b32_e32 v57, 0xffff0000, v7
	v_lshlrev_b32_e32 v58, 16, v8
	v_and_b32_e32 v59, 0xffff0000, v8
	v_lshlrev_b32_e32 v60, 16, v9
	v_and_b32_e32 v61, 0xffff0000, v9
	s_mov_b64 s[36:37], exec
.LBB0_13:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s24, v2
	v_readfirstlane_b32 s25, v3
	v_readfirstlane_b32 s26, v4
	v_readfirstlane_b32 s27, v5
	v_cmp_eq_u64_e32 vcc, s[24:25], v[2:3]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[26:27], v[4:5]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_load_dwordx4 v[6:9], v14, s[24:27], 0 offen offset:192
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_13
; %bb.14:
	s_mov_b64 exec, s[36:37]
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v62, 16, v6
	v_and_b32_e32 v63, 0xffff0000, v6
	v_lshlrev_b32_e32 v64, 16, v7
	v_and_b32_e32 v65, 0xffff0000, v7
	v_lshlrev_b32_e32 v66, 16, v8
	v_and_b32_e32 v67, 0xffff0000, v8
	v_lshlrev_b32_e32 v68, 16, v9
	v_and_b32_e32 v69, 0xffff0000, v9
	s_mov_b64 s[36:37], exec
.LBB0_15:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s24, v2
	v_readfirstlane_b32 s25, v3
	v_readfirstlane_b32 s26, v4
	v_readfirstlane_b32 s27, v5
	v_cmp_eq_u64_e32 vcc, s[24:25], v[2:3]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[26:27], v[4:5]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_load_dwordx4 v[6:9], v14, s[24:27], 0 offen offset:224
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_15
; %bb.16:
	s_mov_b64 exec, s[36:37]
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v70, 16, v6
	v_and_b32_e32 v71, 0xffff0000, v6
	v_lshlrev_b32_e32 v72, 16, v7
	v_and_b32_e32 v73, 0xffff0000, v7
	v_lshlrev_b32_e32 v74, 16, v8
	v_and_b32_e32 v75, 0xffff0000, v8
	v_lshlrev_b32_e32 v76, 16, v9
	v_and_b32_e32 v77, 0xffff0000, v9
	s_mov_b64 s[36:37], exec
.LBB0_17:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s24, v2
	v_readfirstlane_b32 s25, v3
	v_readfirstlane_b32 s26, v4
	v_readfirstlane_b32 s27, v5
	v_cmp_eq_u64_e32 vcc, s[24:25], v[2:3]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[26:27], v[4:5]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_load_dwordx4 v[6:9], v14, s[24:27], 0 offen offset:256
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_17
; %bb.18:
	s_mov_b64 exec, s[36:37]
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v78, 16, v6
	v_and_b32_e32 v79, 0xffff0000, v6
	v_lshlrev_b32_e32 v80, 16, v7
	v_and_b32_e32 v81, 0xffff0000, v7
	v_lshlrev_b32_e32 v82, 16, v8
	v_and_b32_e32 v83, 0xffff0000, v8
	v_lshlrev_b32_e32 v84, 16, v9
	v_and_b32_e32 v85, 0xffff0000, v9
	s_mov_b64 s[36:37], exec
.LBB0_19:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s24, v2
	v_readfirstlane_b32 s25, v3
	v_readfirstlane_b32 s26, v4
	v_readfirstlane_b32 s27, v5
	v_cmp_eq_u64_e32 vcc, s[24:25], v[2:3]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[26:27], v[4:5]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_load_dwordx4 v[6:9], v14, s[24:27], 0 offen offset:288
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_19
; %bb.20:
	s_mov_b64 exec, s[36:37]
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v86, 16, v6
	v_and_b32_e32 v87, 0xffff0000, v6
	v_lshlrev_b32_e32 v88, 16, v7
	v_and_b32_e32 v89, 0xffff0000, v7
	v_lshlrev_b32_e32 v90, 16, v8
	v_and_b32_e32 v91, 0xffff0000, v8
	v_lshlrev_b32_e32 v92, 16, v9
	v_and_b32_e32 v93, 0xffff0000, v9
	s_mov_b64 s[36:37], exec
.LBB0_21:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s24, v2
	v_readfirstlane_b32 s25, v3
	v_readfirstlane_b32 s26, v4
	v_readfirstlane_b32 s27, v5
	v_cmp_eq_u64_e32 vcc, s[24:25], v[2:3]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[26:27], v[4:5]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_load_dwordx4 v[6:9], v14, s[24:27], 0 offen offset:320
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_21
; %bb.22:
	s_mov_b64 exec, s[36:37]
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v94, 16, v6
	v_and_b32_e32 v95, 0xffff0000, v6
	v_lshlrev_b32_e32 v96, 16, v7
	v_and_b32_e32 v97, 0xffff0000, v7
	v_lshlrev_b32_e32 v98, 16, v8
	v_and_b32_e32 v99, 0xffff0000, v8
	v_lshlrev_b32_e32 v100, 16, v9
	v_and_b32_e32 v101, 0xffff0000, v9
	s_mov_b64 s[36:37], exec
.LBB0_23:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s24, v2
	v_readfirstlane_b32 s25, v3
	v_readfirstlane_b32 s26, v4
	v_readfirstlane_b32 s27, v5
	v_cmp_eq_u64_e32 vcc, s[24:25], v[2:3]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[26:27], v[4:5]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_load_dwordx4 v[6:9], v14, s[24:27], 0 offen offset:352
                                        ; implicit-def: $vgpr2_vgpr3_vgpr4_vgpr5
                                        ; implicit-def: $vgpr14
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_23
; %bb.24:
	s_mov_b64 exec, s[36:37]
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v2, 16, v6
	v_and_b32_e32 v3, 0xffff0000, v6
	v_lshlrev_b32_e32 v4, 16, v7
	v_and_b32_e32 v5, 0xffff0000, v7
	v_lshlrev_b32_e32 v6, 16, v8
	v_and_b32_e32 v7, 0xffff0000, v8
	v_lshlrev_b32_e32 v8, 16, v9
	v_and_b32_e32 v9, 0xffff0000, v9
	v_mul_f32_e32 v2, 0x3dd53b94, v2
	v_mul_f32_e32 v14, 0x3dd53b94, v15
	v_mul_f32_e32 v15, 0x3dd53b94, v16
	v_mul_f32_e32 v16, 0x3dd53b94, v17
	v_mul_f32_e32 v17, 0x3dd53b94, v18
	v_mul_f32_e32 v18, 0x3dd53b94, v19
	v_mul_f32_e32 v19, 0x3dd53b94, v20
	v_mul_f32_e32 v20, 0x3dd53b94, v21
	v_mul_f32_e32 v21, 0x3dd53b94, v22
	v_mul_f32_e32 v22, 0x3dd53b94, v23
	v_mul_f32_e32 v10, 0x3dd53b94, v10
	v_mul_f32_e32 v23, 0x3dd53b94, v24
	v_mul_f32_e32 v11, 0x3dd53b94, v11
	v_mul_f32_e32 v24, 0x3dd53b94, v25
	v_mul_f32_e32 v12, 0x3dd53b94, v12
	v_mul_f32_e32 v25, 0x3dd53b94, v26
	v_mul_f32_e32 v13, 0x3dd53b94, v13
	v_mul_f32_e32 v26, 0x3dd53b94, v27
	v_mul_f32_e32 v27, 0x3dd53b94, v28
	v_mul_f32_e32 v28, 0x3dd53b94, v29
	v_mul_f32_e32 v29, 0x3dd53b94, v30
	v_mul_f32_e32 v30, 0x3dd53b94, v31
	v_mul_f32_e32 v31, 0x3dd53b94, v32
	v_mul_f32_e32 v32, 0x3dd53b94, v33
	v_mul_f32_e32 v33, 0x3dd53b94, v37
	v_mul_f32_e32 v37, 0x3dd53b94, v38
	v_mul_f32_e32 v38, 0x3dd53b94, v39
	v_mul_f32_e32 v39, 0x3dd53b94, v40
	v_mul_f32_e32 v40, 0x3dd53b94, v41
	v_mul_f32_e32 v41, 0x3dd53b94, v42
	v_mul_f32_e32 v42, 0x3dd53b94, v43
	v_mul_f32_e32 v43, 0x3dd53b94, v44
	v_mul_f32_e32 v44, 0x3dd53b94, v45
	v_mul_f32_e32 v45, 0x3dd53b94, v46
	v_mul_f32_e32 v46, 0x3dd53b94, v47
	v_mul_f32_e32 v47, 0x3dd53b94, v48
	v_mul_f32_e32 v48, 0x3dd53b94, v49
	v_mul_f32_e32 v49, 0x3dd53b94, v50
	v_mul_f32_e32 v50, 0x3dd53b94, v51
	v_mul_f32_e32 v51, 0x3dd53b94, v52
	v_mul_f32_e32 v52, 0x3dd53b94, v53
	v_mul_f32_e32 v53, 0x3dd53b94, v54
	v_mul_f32_e32 v54, 0x3dd53b94, v55
	v_mul_f32_e32 v55, 0x3dd53b94, v56
	v_mul_f32_e32 v56, 0x3dd53b94, v57
	v_mul_f32_e32 v57, 0x3dd53b94, v58
	v_mul_f32_e32 v58, 0x3dd53b94, v59
	v_mul_f32_e32 v59, 0x3dd53b94, v60
	v_mul_f32_e32 v60, 0x3dd53b94, v61
	v_mul_f32_e32 v61, 0x3dd53b94, v62
	v_mul_f32_e32 v62, 0x3dd53b94, v63
	v_mul_f32_e32 v63, 0x3dd53b94, v64
	v_mul_f32_e32 v64, 0x3dd53b94, v65
	v_mul_f32_e32 v65, 0x3dd53b94, v66
	v_mul_f32_e32 v66, 0x3dd53b94, v67
	v_mul_f32_e32 v67, 0x3dd53b94, v68
	v_mul_f32_e32 v68, 0x3dd53b94, v69
	v_mul_f32_e32 v69, 0x3dd53b94, v70
	v_mul_f32_e32 v70, 0x3dd53b94, v71
	v_mul_f32_e32 v71, 0x3dd53b94, v72
	v_mul_f32_e32 v72, 0x3dd53b94, v73
	v_mul_f32_e32 v73, 0x3dd53b94, v74
	v_mul_f32_e32 v74, 0x3dd53b94, v75
	v_mul_f32_e32 v75, 0x3dd53b94, v76
	v_mul_f32_e32 v76, 0x3dd53b94, v77
	v_mul_f32_e32 v77, 0x3dd53b94, v78
	v_mul_f32_e32 v78, 0x3dd53b94, v79
	v_mul_f32_e32 v79, 0x3dd53b94, v80
	v_mul_f32_e32 v80, 0x3dd53b94, v81
	v_mul_f32_e32 v81, 0x3dd53b94, v82
	v_mul_f32_e32 v82, 0x3dd53b94, v83
	v_mul_f32_e32 v83, 0x3dd53b94, v84
	v_mul_f32_e32 v84, 0x3dd53b94, v85
	v_mul_f32_e32 v85, 0x3dd53b94, v86
	v_mul_f32_e32 v86, 0x3dd53b94, v87
	v_mul_f32_e32 v87, 0x3dd53b94, v88
	v_mul_f32_e32 v88, 0x3dd53b94, v89
	v_mul_f32_e32 v89, 0x3dd53b94, v90
	v_mul_f32_e32 v90, 0x3dd53b94, v91
	v_mul_f32_e32 v91, 0x3dd53b94, v92
	v_mul_f32_e32 v92, 0x3dd53b94, v93
	v_mul_f32_e32 v93, 0x3dd53b94, v94
	v_mul_f32_e32 v94, 0x3dd53b94, v95
	v_mul_f32_e32 v95, 0x3dd53b94, v96
	v_mul_f32_e32 v96, 0x3dd53b94, v97
	v_mul_f32_e32 v97, 0x3dd53b94, v98
	v_mul_f32_e32 v98, 0x3dd53b94, v99
	v_mul_f32_e32 v99, 0x3dd53b94, v100
	v_mul_f32_e32 v100, 0x3dd53b94, v101
	v_mul_f32_e32 v3, 0x3dd53b94, v3
	v_mul_f32_e32 v4, 0x3dd53b94, v4
	v_mul_f32_e32 v5, 0x3dd53b94, v5
	v_mul_f32_e32 v6, 0x3dd53b94, v6
	v_mul_f32_e32 v7, 0x3dd53b94, v7
	v_mul_f32_e32 v8, 0x3dd53b94, v8
	v_mul_f32_e32 v9, 0x3dd53b94, v9
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v116, v14, v15
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v117, v16, v17
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v118, v18, v19
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v119, v20, v21
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v112, v22, v10
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v113, v23, v11
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v114, v24, v12
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v115, v25, v13
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v120, v26, v27
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v121, v28, v29
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v122, v30, v31
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v123, v32, v33
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v128, v37, v38
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v129, v39, v40
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v130, v41, v42
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v131, v43, v44
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v124, v45, v46
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v125, v47, v48
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v126, v49, v50
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v127, v51, v52
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v132, v53, v54
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v133, v55, v56
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v134, v57, v58
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v135, v59, v60
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v136, v61, v62
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v137, v63, v64
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v138, v65, v66
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v139, v67, v68
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v140, v69, v70
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v141, v71, v72
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v142, v73, v74
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v143, v75, v76
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v168, v77, v78
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v169, v79, v80
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v170, v81, v82
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v171, v83, v84
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v172, v85, v86
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v173, v87, v88
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v174, v89, v90
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v175, v91, v92
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v164, v93, v94
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v165, v95, v96
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v166, v97, v98
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v167, v99, v100
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v106, v2, v3
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v107, v4, v5
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v108, v6, v7
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v109, v8, v9
	;;#ASMEND
	scratch_load_dword v2, off, off         ; 4-byte Folded Reload
	s_add_i32 s0, s44, 64
	s_mul_i32 s15, s0, s30
	s_add_i32 s15, s15, s31
	s_mul_i32 s0, s15, s33
	s_add_i32 s19, s21, 0x6000
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	v_add_u32_e32 v3, s19, v35
	s_add_u32 s36, s42, s0
	v_readfirstlane_b32 s0, v3
	s_addc_u32 s37, s43, s1
	s_mov_b32 m0, s0
	s_mul_i32 s11, s4, s20
	s_mul_i32 s17, s11, s22
	s_add_i32 s17, s17, s31
	s_mov_b32 s27, s39
	scratch_store_dword off, v3, off offset:224 ; 4-byte Folded Spill
	v_lshlrev_b32_e32 v37, 2, v0
	v_and_b32_e32 v38, 16, v0
	v_and_or_b32 v19, v37, 32, v38
	s_waitcnt vmcnt(1)
	buffer_load_dwordx4 v2, s[36:39], 0 offen lds
	v_add_u32_e32 v2, 0x2000, v3
	scratch_store_dword off, v2, off offset:228 ; 4-byte Folded Spill
	v_readfirstlane_b32 s0, v2
	scratch_load_dword v2, off, off offset:4 ; 4-byte Folded Reload
	s_mov_b32 m0, s0
	s_waitcnt vmcnt(0)
	buffer_load_dwordx4 v2, s[36:39], 0 offen lds
	v_add_u32_e32 v2, 0x4000, v3
	scratch_store_dword off, v2, off offset:232 ; 4-byte Folded Spill
	v_readfirstlane_b32 s0, v2
	scratch_load_dword v2, off, off offset:8 ; 4-byte Folded Reload
	s_mov_b32 m0, s0
	s_mul_i32 s0, s17, s23
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s24, s40, s0
	s_addc_u32 s25, s41, s1
	s_lshl_b32 s26, s2, 7
	v_lshrrev_b32_e32 v3, 1, v0
	v_and_b32_e32 v3, 16, v3
	s_waitcnt vmcnt(0)
	buffer_load_dwordx4 v2, s[36:39], 0 offen lds
	v_add_u32_e32 v2, s45, v35
	scratch_store_dword off, v2, off offset:212 ; 4-byte Folded Spill
	v_readfirstlane_b32 s0, v2
	v_add_u32_e32 v2, 0x2000, v2
	s_mov_b32 m0, s0
	v_readfirstlane_b32 s0, v2
	buffer_load_dwordx4 v105, s[24:27], 0 offen lds
	s_mov_b32 m0, s0
	scratch_store_dword off, v2, off offset:216 ; 4-byte Folded Spill
	buffer_load_dwordx4 v110, s[24:27], 0 offen lds
	v_lshlrev_b32_e32 v2, 6, v0
	v_and_b32_e32 v2, 0x7c0, v2
	v_bitop3_b32 v40, v3, v19, v2 bitop3:0x36
	v_or_b32_e32 v18, v3, v2
	v_add_u32_e32 v20, s21, v40
	;;#ASMSTART
	ds_read_b128 v[2:5], v20 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[6:9], v20 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[10:13], v20 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[14:17], v20 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[42:45], v20 offset:0x2000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[46:49], v20 offset:0x2800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[50:53], v20 offset:0x3000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[54:57], v20 offset:0x3800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[58:61], v20 offset:0x4000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[62:65], v20 offset:0x4800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[66:69], v20 offset:0x5000

	;;#ASMEND
	scratch_store_dword off, v20, off offset:208 ; 4-byte Folded Spill
	;;#ASMSTART
	ds_read_b128 v[70:73], v20 offset:0x5800

	;;#ASMEND
	v_bitop3_b32 v39, v18, v19, 32 bitop3:0x36
	v_add_u32_e32 v18, s21, v39
	;;#ASMSTART
	ds_read_b128 v[74:77], v18 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[78:81], v18 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[82:85], v18 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[86:89], v18 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[90:93], v18 offset:0x2000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[94:97], v18 offset:0x2800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[98:101], v18 offset:0x3000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[144:147], v18 offset:0x3800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[148:151], v18 offset:0x4000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[152:155], v18 offset:0x4800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[156:159], v18 offset:0x5000

	;;#ASMEND
	scratch_store_dword off, v18, off offset:220 ; 4-byte Folded Spill
	;;#ASMSTART
	ds_read_b128 v[160:163], v18 offset:0x5800

	;;#ASMEND
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(2)
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	v_mfma_f32_32x32x16_bf16 v[18:33], v[2:5], v[116:119], 0
	s_barrier
	v_mfma_f32_32x32x16_bf16 v[18:33], v[74:77], v[112:115], v[18:33]
	v_mfma_f32_32x32x16_bf16 v[18:33], v[6:9], v[120:123], v[18:33]
	v_mfma_f32_32x32x16_bf16 v[18:33], v[78:81], v[128:131], v[18:33]
	v_mfma_f32_32x32x16_bf16 v[18:33], v[10:13], v[124:127], v[18:33]
	v_mfma_f32_32x32x16_bf16 v[18:33], v[82:85], v[132:135], v[18:33]
	v_mfma_f32_32x32x16_bf16 v[18:33], v[14:17], v[136:139], v[18:33]
	v_mfma_f32_32x32x16_bf16 v[18:33], v[86:89], v[140:143], v[18:33]
	v_mov_b64_e32 v[84:85], v[164:165]
	v_mov_b64_e32 v[86:87], v[166:167]
	v_mfma_f32_32x32x16_bf16 v[18:33], v[42:45], v[168:171], v[18:33]
	v_mfma_f32_32x32x16_bf16 v[18:33], v[90:93], v[172:175], v[18:33]
	v_mov_b64_e32 v[88:89], v[168:169]
	v_mov_b64_e32 v[90:91], v[170:171]
	v_mfma_f32_32x32x16_bf16 v[2:17], v[50:53], v[116:119], 0
	v_mfma_f32_32x32x16_bf16 v[18:33], v[46:49], v[164:167], v[18:33]
	v_mov_b64_e32 v[48:49], v[112:113]
	v_mov_b64_e32 v[50:51], v[114:115]
	v_mov_b64_e32 v[44:45], v[106:107]
	v_mov_b64_e32 v[46:47], v[108:109]
	v_mfma_f32_32x32x16_bf16 v[2:17], v[98:101], v[48:51], v[2:17]
	v_mfma_f32_32x32x16_bf16 v[2:17], v[54:57], v[120:123], v[2:17]
	v_mov_b64_e32 v[52:53], v[120:121]
	v_mov_b64_e32 v[54:55], v[122:123]
	v_mfma_f32_32x32x16_bf16 v[2:17], v[144:147], v[128:131], v[2:17]
	v_mfma_f32_32x32x16_bf16 v[2:17], v[58:61], v[124:127], v[2:17]
	v_mov_b64_e32 v[56:57], v[124:125]
	v_mov_b64_e32 v[58:59], v[126:127]
	v_mfma_f32_32x32x16_bf16 v[2:17], v[148:151], v[132:135], v[2:17]
	v_mfma_f32_32x32x16_bf16 v[2:17], v[62:65], v[136:139], v[2:17]
	v_mov_b64_e32 v[60:61], v[136:137]
	v_mov_b64_e32 v[62:63], v[138:139]
	v_mfma_f32_32x32x16_bf16 v[2:17], v[152:155], v[140:143], v[2:17]
	v_mfma_f32_32x32x16_bf16 v[2:17], v[66:69], v[88:91], v[2:17]
	v_mfma_f32_32x32x16_bf16 v[2:17], v[156:159], v[172:175], v[2:17]
	v_mfma_f32_32x32x16_bf16 v[2:17], v[70:73], v[84:87], v[2:17]
	v_mfma_f32_32x32x16_bf16 v[18:33], v[94:97], v[44:47], v[18:33]
	v_mfma_f32_32x32x16_bf16 v[2:17], v[160:163], v[44:47], v[2:17]
	; sched_barrier mask(0x00000000)
	v_cmp_gt_i32_e32 vcc, 2, v36
	v_lshrrev_b32_e32 v36, 3, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execnz .LBB0_75
.LBB0_25:
	s_or_b64 exec, exec, s[0:1]
	v_max_f32_e32 v41, v19, v21
	v_max3_f32 v42, v18, v20, v22
	v_max3_f32 v41, v41, v23, v25
	v_max3_f32 v42, v42, v24, v26
	v_max3_f32 v41, v41, v27, v29
	v_max3_f32 v42, v42, v28, v30
	v_max3_f32 v41, v41, v31, v33
	v_max3_f32 v42, v42, v32, v2
	v_max3_f32 v41, v41, v3, v5
	v_max3_f32 v42, v42, v4, v6
	v_max3_f32 v41, v41, v7, v9
	v_max3_f32 v42, v42, v8, v10
	v_max3_f32 v41, v41, v11, v13
	v_max3_f32 v42, v42, v12, v14
	v_max3_f32 v41, v41, v15, v17
	v_max3_f32 v41, v42, v16, v41
	v_mov_b32_e32 v42, v41
	s_nop 1
	v_permlane32_swap_b32_e64 v41, v42 bound_ctrl:1
	v_max_f32_e32 v255, v41, v42
	v_sub_f32_e32 v18, v18, v255
	v_sub_f32_e32 v19, v19, v255
	v_sub_f32_e32 v20, v20, v255
	v_sub_f32_e32 v21, v21, v255
	v_sub_f32_e32 v22, v22, v255
	v_sub_f32_e32 v23, v23, v255
	v_sub_f32_e32 v24, v24, v255
	v_sub_f32_e32 v25, v25, v255
	v_sub_f32_e32 v26, v26, v255
	v_sub_f32_e32 v27, v27, v255
	v_sub_f32_e32 v28, v28, v255
	v_sub_f32_e32 v29, v29, v255
	v_sub_f32_e32 v30, v30, v255
	v_sub_f32_e32 v31, v31, v255
	v_sub_f32_e32 v32, v32, v255
	v_sub_f32_e32 v33, v33, v255
	; sched_barrier mask(0x00000000)
	s_movk_i32 s0, 0x100
	s_movk_i32 s2, 0xff
	v_cmp_gt_u32_e64 s[0:1], s0, v0
	v_cmp_lt_u32_e32 vcc, s2, v0
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB0_27
; %bb.26:
	; sched_barrier mask(0x00000000)
	s_barrier
.LBB0_27:
	s_or_b64 exec, exec, s[20:21]
	scratch_store_dwordx4 off, v[172:175], off offset:172 ; 16-byte Folded Spill
	scratch_store_dwordx4 off, v[60:63], off offset:120 ; 16-byte Folded Spill
	scratch_store_dwordx4 off, v[56:59], off offset:100 ; 16-byte Folded Spill
	scratch_store_dwordx4 off, v[52:55], off offset:84 ; 16-byte Folded Spill
	scratch_store_dwordx4 off, v[48:51], off offset:68 ; 16-byte Folded Spill
	scratch_store_dwordx4 off, v[44:47], off offset:52 ; 16-byte Folded Spill
	v_add_u32_e32 v0, s19, v40
	;;#ASMSTART
	ds_read_b128 v[80:83], v0 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[208:211], v0 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[184:187], v0 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[180:183], v0 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[176:179], v0 offset:0x2000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[172:175], v0 offset:0x2800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[168:171], v0 offset:0x3000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[152:155], v0 offset:0x3800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[156:159], v0 offset:0x4000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[160:163], v0 offset:0x4800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[164:167], v0 offset:0x5000

	;;#ASMEND
	scratch_store_dword off, v0, off offset:192 ; 4-byte Folded Spill
	;;#ASMSTART
	ds_read_b128 v[148:151], v0 offset:0x5800

	;;#ASMEND
	v_add_u32_e32 v0, s19, v39
	;;#ASMSTART
	ds_read_b128 v[232:235], v0 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[228:231], v0 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[124:127], v0 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[112:115], v0 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[216:219], v0 offset:0x2000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[212:215], v0 offset:0x2800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[188:191], v0 offset:0x3000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[196:199], v0 offset:0x3800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[200:203], v0 offset:0x4000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[204:207], v0 offset:0x4800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[192:195], v0 offset:0x5000

	;;#ASMEND
	scratch_store_dword off, v0, off offset:196 ; 4-byte Folded Spill
	;;#ASMSTART
	ds_read_b128 v[144:147], v0 offset:0x5800

	;;#ASMEND
	scratch_load_dword v0, off, off offset:236 ; 4-byte Folded Reload
	s_lshl_b32 s19, s30, 6
	s_add_i32 s15, s15, s19
	s_mul_i32 s20, s15, s33
	s_lshl_b32 s2, s3, 8
	s_add_i32 s3, s2, 0x13f
	s_ashr_i32 s13, s3, 31
	s_lshr_b32 s13, s13, 26
	s_add_i32 s13, s3, s13
	s_ashr_i32 s13, s13, 6
	s_ashr_i32 s21, s20, 31
	s_min_i32 s13, s13, 16
	s_lshl_b64 s[20:21], s[20:21], 1
	s_add_u32 s36, s42, s20
	s_addc_u32 s37, s43, s21
	s_mov_b32 s39, 0x110000
	s_add_i32 s29, s45, 0x4000
	s_mov_b32 s27, s39
	v_exp_f32_e32 v100, v18
	v_exp_f32_e32 v102, v19
	v_exp_f32_e32 v98, v20
	v_exp_f32_e32 v101, v21
	v_exp_f32_e32 v250, v22
	v_exp_f32_e32 v99, v23
	v_exp_f32_e32 v248, v24
	v_exp_f32_e32 v97, v25
	v_exp_f32_e32 v251, v26
	v_exp_f32_e32 v96, v27
	v_exp_f32_e32 v108, v28
	v_exp_f32_e32 v249, v29
	v_exp_f32_e32 v253, v30
	v_exp_f32_e32 v109, v31
	v_exp_f32_e32 v246, v32
	v_exp_f32_e32 v247, v33
	v_sub_f32_e32 v65, v17, v255
	v_sub_f32_e32 v64, v16, v255
	v_sub_f32_e32 v67, v15, v255
	v_sub_f32_e32 v66, v14, v255
	v_sub_f32_e32 v69, v13, v255
	v_sub_f32_e32 v68, v12, v255
	v_sub_f32_e32 v71, v11, v255
	v_sub_f32_e32 v70, v10, v255
	v_sub_f32_e32 v73, v9, v255
	v_sub_f32_e32 v72, v8, v255
	v_sub_f32_e32 v75, v7, v255
	v_sub_f32_e32 v74, v6, v255
	v_sub_f32_e32 v77, v5, v255
	v_sub_f32_e32 v76, v4, v255
	v_sub_f32_e32 v79, v3, v255
	v_sub_f32_e32 v78, v2, v255
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s15, v0
	scratch_load_dword v0, off, off         ; 4-byte Folded Reload
	s_mov_b32 m0, s15
	s_waitcnt vmcnt(0)
	buffer_load_dwordx4 v0, s[36:39], 0 offen lds
	scratch_load_dword v0, off, off offset:240 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s15, v0
	scratch_load_dword v0, off, off offset:4 ; 4-byte Folded Reload
	s_mov_b32 m0, s15
	s_waitcnt vmcnt(0)
	buffer_load_dwordx4 v0, s[36:39], 0 offen lds
	scratch_load_dword v0, off, off offset:244 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s15, v0
	scratch_load_dword v0, off, off offset:8 ; 4-byte Folded Reload
	s_mov_b32 m0, s15
	s_lshl_b32 s15, s22, 6
	s_add_i32 s15, s17, s15
	s_mul_i32 s20, s15, s23
	s_ashr_i32 s21, s20, 31
	s_lshl_b64 s[20:21], s[20:21], 1
	s_add_u32 s24, s40, s20
	s_addc_u32 s25, s41, s21
	s_waitcnt vmcnt(0)
	buffer_load_dwordx4 v0, s[36:39], 0 offen lds
	v_add_u32_e32 v0, s29, v35
	scratch_store_dword off, v0, off offset:200 ; 4-byte Folded Spill
	v_readfirstlane_b32 s15, v0
	v_add_u32_e32 v0, 0x2000, v0
	s_mov_b32 m0, s15
	v_readfirstlane_b32 s15, v0
	buffer_load_dwordx4 v105, s[24:27], 0 offen lds
	s_mov_b32 m0, s15
	scratch_store_dword off, v0, off offset:204 ; 4-byte Folded Spill
	buffer_load_dwordx4 v110, s[24:27], 0 offen lds
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	v_and_b32_e32 v2, 4, v36
	v_and_or_b32 v0, v1, 3, v2
	v_and_or_b32 v1, v37, 12, v38
	v_lshlrev_b32_e32 v1, 1, v1
	s_add_i32 s15, s13, -1
	v_lshl_or_b32 v0, v0, 6, v1
	s_cmpk_lt_i32 s3, 0x140
	s_mov_b32 s17, 3
	v_add_u32_e32 v1, s45, v0
	s_mov_b32 s19, 0
	v_mov_b32_e32 v15, 0
	v_add_u32_e32 v0, s29, v0
	s_barrier
	scratch_store_dword off, v1, off offset:188 ; 4-byte Folded Spill
	scratch_store_dword off, v0, off offset:152 ; 4-byte Folded Spill
	scratch_store_dword off, v103, off offset:48 ; 4-byte Folded Spill
	scratch_store_dword off, v110, off offset:116 ; 4-byte Folded Spill
	scratch_store_dword off, v105, off offset:12 ; 4-byte Folded Spill
	scratch_store_dwordx4 off, v[84:87], off offset:136 ; 16-byte Folded Spill
	scratch_store_dwordx4 off, v[88:91], off offset:156 ; 16-byte Folded Spill
	scratch_store_dword off, v104, off offset:312 ; 4-byte Folded Spill
	scratch_store_dword off, v2, off offset:316 ; 4-byte Folded Spill
	s_cbranch_scc1 .LBB0_33
; %bb.28:                               ; %.lr.ph
	v_lshlrev_b32_e32 v0, 5, v34
	v_add3_u32 v0, s2, v0, v104
	s_add_i32 s2, s11, 0x80
	s_mul_i32 s2, s2, s22
	s_add_i32 s21, s31, s2
	s_add_i32 s2, s11, 0xc0
	s_mul_i32 s2, s2, s22
	s_add_i32 s3, s44, 0x100
	s_add_i32 s35, s31, s2
	s_add_i32 s2, s44, 0xc0
	s_mul_i32 s3, s3, s30
	v_sub_u32_e32 v0, v0, v2
	s_mul_i32 s2, s2, s30
	s_add_i32 s20, s31, s3
	v_add_u32_e32 v0, 0xffffff60, v0
	s_add_i32 s45, s31, s2
	v_mov_b32_e32 v84, 0
	v_mov_b32_e32 v63, 0
	scratch_store_dwordx4 off, v[140:143], off offset:296 ; 16-byte Folded Spill
	scratch_store_dwordx4 off, v[132:135], off offset:264 ; 16-byte Folded Spill
	scratch_store_dwordx4 off, v[128:131], off offset:248 ; 16-byte Folded Spill
	s_mul_i32 s20, s20, s33
	scratch_store_dword off, v0, off offset:16 ; 4-byte Folded Spill
	s_mul_i32 s21, s21, s23
	s_movk_i32 s29, 0xc0
	s_mul_i32 s35, s35, s23
	s_mul_i32 s45, s45, s33
	v_mov_b32_e32 v236, 1.0
	s_mov_b32 s46, 0
	v_mov_b32_e32 v0, 0
	v_mov_b32_e32 v1, v84
	v_mov_b32_e32 v2, v84
	v_mov_b32_e32 v3, v84
	v_mov_b32_e32 v4, v84
	v_mov_b32_e32 v5, v84
	v_mov_b32_e32 v6, v84
	v_mov_b32_e32 v7, v84
	v_mov_b32_e32 v8, v84
	v_mov_b32_e32 v9, v84
	v_mov_b32_e32 v10, v84
	v_mov_b32_e32 v11, v84
	v_mov_b32_e32 v12, v84
	v_mov_b32_e32 v13, v84
	v_mov_b32_e32 v14, v84
	v_mov_b32_e32 v15, v84
	v_mov_b32_e32 v16, 0
	v_mov_b32_e32 v17, v84
	v_mov_b32_e32 v18, v84
	v_mov_b32_e32 v19, v84
	v_mov_b32_e32 v20, v84
	v_mov_b32_e32 v21, v84
	v_mov_b32_e32 v22, v84
	v_mov_b32_e32 v23, v84
	v_mov_b32_e32 v24, v84
	v_mov_b32_e32 v25, v84
	v_mov_b32_e32 v26, v84
	v_mov_b32_e32 v27, v84
	v_mov_b32_e32 v28, v84
	v_mov_b32_e32 v29, v84
	v_mov_b32_e32 v30, v84
	v_mov_b32_e32 v31, v84
	v_mov_b32_e32 v32, 0
	v_mov_b32_e32 v33, v84
	v_mov_b32_e32 v34, v84
	v_mov_b32_e32 v35, v84
	v_mov_b32_e32 v36, v84
	v_mov_b32_e32 v37, v84
	v_mov_b32_e32 v38, v84
	v_mov_b32_e32 v39, v84
	v_mov_b32_e32 v40, v84
	v_mov_b32_e32 v41, v84
	v_mov_b32_e32 v42, v84
	v_mov_b32_e32 v43, v84
	v_mov_b32_e32 v44, v84
	v_mov_b32_e32 v45, v84
	v_mov_b32_e32 v46, v84
	v_mov_b32_e32 v47, v84
	v_mov_b32_e32 v48, 0
	v_mov_b32_e32 v49, v84
	v_mov_b32_e32 v50, v84
	v_mov_b32_e32 v51, v84
	v_mov_b32_e32 v52, v84
	v_mov_b32_e32 v53, v84
	v_mov_b32_e32 v54, v84
	v_mov_b32_e32 v55, v84
	v_mov_b32_e32 v56, v84
	v_mov_b32_e32 v57, v84
	v_mov_b32_e32 v58, v84
	v_mov_b32_e32 v59, v84
	v_mov_b32_e32 v60, v84
	v_mov_b32_e32 v61, v84
	v_mov_b32_e32 v62, v84
	scratch_store_dword off, v63, off offset:20 ; 4-byte Folded Spill
	v_mov_b32_e32 v63, v84
	scratch_store_dwordx4 off, v[116:119], off offset:280 ; 16-byte Folded Spill
.LBB0_29:                               ; =>This Inner Loop Header: Depth=1
	scratch_load_dwordx4 v[238:241], off, off offset:68 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[224:227], off, off offset:84 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[128:131], off, off offset:248 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[136:139], off, off offset:100 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[132:135], off, off offset:264 ; 16-byte Folded Reload
	s_waitcnt vmcnt(7)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[80:83], v[116:119], 0
	v_exp_f32_e32 v104, v78
	v_exp_f32_e32 v105, v79
	v_exp_f32_e32 v106, v76
	scratch_load_dwordx4 v[140:143], off, off offset:296 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[120:123], off, off offset:172 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[242:245], off, off offset:52 ; 16-byte Folded Reload
	; sched_group_barrier mask(0x00000008) size(1) SyncID(1)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(1)
	s_waitcnt vmcnt(7)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[232:235], v[238:241], v[80:95]
	v_exp_f32_e32 v107, v77
	v_exp_f32_e32 v232, v74
	v_exp_f32_e32 v233, v75
	; sched_group_barrier mask(0x00000008) size(1) SyncID(1)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(1)
	s_waitcnt vmcnt(6)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[208:211], v[224:227], v[80:95]
	v_exp_f32_e32 v208, v72
	v_exp_f32_e32 v209, v73
	v_exp_f32_e32 v210, v70
	; sched_group_barrier mask(0x00000008) size(1) SyncID(1)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(1)
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[228:231], v[128:131], v[80:95]
	v_exp_f32_e32 v211, v71
	v_exp_f32_e32 v228, v68
	v_exp_f32_e32 v229, v69
	; sched_group_barrier mask(0x00000008) size(1) SyncID(1)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(1)
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[184:187], v[136:139], v[80:95]
	v_exp_f32_e32 v184, v66
	v_exp_f32_e32 v185, v67
	v_exp_f32_e32 v186, v64
	; sched_group_barrier mask(0x00000008) size(1) SyncID(1)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(1)
	s_waitcnt vmcnt(3)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[124:127], v[132:135], v[80:95]
	scratch_load_dwordx4 v[124:127], off, off offset:120 ; 16-byte Folded Reload
	v_exp_f32_e32 v187, v65
	; sched_group_barrier mask(0x00000008) size(1) SyncID(1)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(1)
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[180:183], v[124:127], v[80:95]
	v_mov_b64_e32 v[222:223], v[118:119]
	v_mov_b64_e32 v[220:221], v[116:117]
	scratch_load_dwordx4 v[116:119], off, off offset:156 ; 16-byte Folded Reload
	v_add_f32_e32 v64, v102, v100
	v_add_f32_e32 v64, v64, v98
	v_add_f32_e32 v64, v64, v101
	; sched_group_barrier mask(0x00000008) size(1) SyncID(1)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(1)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[112:115], v[140:143], v[80:95]
	scratch_load_dwordx4 v[112:115], off, off offset:136 ; 16-byte Folded Reload
	v_add_f32_e32 v64, v64, v250
	v_add_f32_e32 v64, v64, v99
	v_add_f32_e32 v64, v64, v248
	v_add_f32_e32 v64, v64, v97
	v_add_f32_e32 v64, v64, v251
	; sched_group_barrier mask(0x00000008) size(1) SyncID(1)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(1)
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[176:179], v[116:119], v[80:95]
	v_add_f32_e32 v64, v64, v96
	v_add_f32_e32 v64, v64, v108
	v_add_f32_e32 v64, v64, v249
	v_add_f32_e32 v64, v64, v253
	v_add_f32_e32 v64, v64, v109
	; sched_group_barrier mask(0x00000008) size(1) SyncID(1)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(1)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[216:219], v[120:123], v[80:95]
	v_add_f32_e32 v64, v64, v246
	v_add_f32_e32 v64, v64, v247
	v_add_f32_e32 v64, v64, v104
	v_add_f32_e32 v64, v64, v105
	v_add_f32_e32 v64, v64, v106
	; sched_group_barrier mask(0x00000008) size(1) SyncID(1)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(1)
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[172:175], v[112:115], v[80:95]
	v_add_f32_e32 v64, v64, v107
	v_add_f32_e32 v64, v64, v232
	v_add_f32_e32 v64, v64, v233
	v_add_f32_e32 v64, v64, v208
	v_add_f32_e32 v64, v64, v209
	; sched_group_barrier mask(0x00000008) size(1) SyncID(1)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(1)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[212:215], v[242:245], v[80:95]
	v_add_f32_e32 v64, v64, v210
	v_add_f32_e32 v64, v64, v211
	v_add_f32_e32 v64, v64, v228
	v_add_f32_e32 v64, v64, v229
	v_add_f32_e32 v64, v64, v184
	v_add_f32_e32 v64, v64, v185
	v_add_f32_e32 v103, v64, v186
	v_mfma_f32_32x32x16_bf16 v[64:79], v[168:171], v[220:223], 0
	; sched_group_barrier mask(0x00000008) size(1) SyncID(1)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(1)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[188:191], v[238:241], v[64:79]
	v_mfma_f32_32x32x16_bf16 v[64:79], v[152:155], v[224:227], v[64:79]
	v_mfma_f32_32x32x16_bf16 v[64:79], v[196:199], v[128:131], v[64:79]
	v_add_f32_e32 v103, v103, v187
	v_mov_b32_e32 v111, v103
	s_nop 1
	v_permlane32_swap_b32_e64 v103, v111 bound_ctrl:1
	scratch_store_dword off, v111, off offset:24 ; 4-byte Folded Spill
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v152, v100, v102
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v153, v98, v101
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[64:79], v[156:159], v[136:139], v[64:79]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v154, v250, v99
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v155, v248, v97
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v156, v251, v96
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v157, v108, v249
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v158, v253, v109
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v159, v246, v247
	;;#ASMEND
	; sched_group_barrier mask(0x00000008) size(1) SyncID(1)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(1)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(1)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(1)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(1)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(1)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(1)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(1)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[200:203], v[132:135], v[64:79]
	v_mfma_f32_32x32x16_bf16 v[64:79], v[160:163], v[124:127], v[64:79]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v160, v104, v105
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v161, v106, v107
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v162, v232, v233
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v163, v208, v209
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[64:79], v[204:207], v[140:143], v[64:79]
	v_mfma_f32_32x32x16_bf16 v[64:79], v[164:167], v[116:119], v[64:79]
	v_mfma_f32_32x32x16_bf16 v[64:79], v[192:195], v[120:123], v[64:79]
	v_mfma_f32_32x32x16_bf16 v[64:79], v[148:151], v[112:115], v[64:79]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v148, v210, v211
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v149, v228, v229
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v150, v184, v185
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v151, v186, v187
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[64:79], v[144:147], v[242:245], v[64:79]
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	scratch_load_dword v96, off, off offset:224 ; 4-byte Folded Reload
	s_add_i32 s2, s45, s46
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[2:3], s[2:3], 1
	s_add_u32 s36, s42, s2
	s_addc_u32 s37, s43, s3
	scratch_load_dword v100, off, off offset:188 ; 4-byte Folded Reload
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s2, v96
	scratch_load_dword v96, off, off        ; 4-byte Folded Reload
	s_mov_b32 m0, s2
	s_waitcnt vmcnt(0)
	buffer_load_dwordx4 v96, s[36:39], 0 offen lds
	scratch_load_dword v96, off, off offset:228 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s2, v96
	scratch_load_dword v96, off, off offset:4 ; 4-byte Folded Reload
	s_mov_b32 m0, s2
	s_waitcnt vmcnt(0)
	buffer_load_dwordx4 v96, s[36:39], 0 offen lds
	scratch_load_dword v96, off, off offset:232 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s2, v96
	scratch_load_dword v96, off, off offset:8 ; 4-byte Folded Reload
	s_mov_b32 m0, s2
	s_waitcnt vmcnt(0)
	buffer_load_dwordx4 v96, s[36:39], 0 offen lds
	;;#ASMSTART
	ds_read_b64_tr_b16 v[96:97], v100 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[164:165], v100 offset:0x200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[168:169], v100 offset:0x400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[144:145], v100 offset:0x600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[172:173], v100 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[176:177], v100 offset:0x1200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[180:181], v100 offset:0x1400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[184:185], v100 offset:0x1600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[188:189], v100 offset:0x2000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[192:193], v100 offset:0x2200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[196:197], v100 offset:0x2400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[200:201], v100 offset:0x2600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[204:205], v100 offset:0x3000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[208:209], v100 offset:0x3200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[212:213], v100 offset:0x3400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[216:217], v100 offset:0x3600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[98:99], v100 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[166:167], v100 offset:0xa00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[170:171], v100 offset:0xc00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[146:147], v100 offset:0xe00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[174:175], v100 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[178:179], v100 offset:0x1a00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[182:183], v100 offset:0x1c00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[186:187], v100 offset:0x1e00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[190:191], v100 offset:0x2800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[194:195], v100 offset:0x2a00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[198:199], v100 offset:0x2c00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[202:203], v100 offset:0x2e00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[206:207], v100 offset:0x3800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[210:211], v100 offset:0x3a00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[214:215], v100 offset:0x3c00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[218:219], v100 offset:0x3e00

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_f32_32x32x16_bf16 v[48:63], v[96:99], v[152:155], v[48:63]
	v_max_f32_e32 v96, v81, v83
	v_max3_f32 v97, v80, v82, v84
	v_max3_f32 v96, v96, v85, v87
	v_max3_f32 v97, v97, v86, v88
	v_max3_f32 v96, v96, v89, v91
	; sched_group_barrier mask(0x00000008) size(1) SyncID(2)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(2)
	v_mfma_f32_32x32x16_bf16 v[48:63], v[172:175], v[156:159], v[48:63]
	v_max3_f32 v97, v97, v90, v92
	v_max3_f32 v96, v96, v93, v95
	v_max3_f32 v97, v97, v94, v64
	v_max3_f32 v96, v96, v65, v67
	v_max3_f32 v97, v97, v66, v68
	; sched_group_barrier mask(0x00000008) size(1) SyncID(2)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(2)
	v_mfma_f32_32x32x16_bf16 v[48:63], v[188:191], v[160:163], v[48:63]
	v_max3_f32 v96, v96, v69, v71
	v_max3_f32 v97, v97, v70, v72
	v_max3_f32 v96, v96, v73, v75
	v_max3_f32 v97, v97, v74, v76
	v_max3_f32 v96, v96, v77, v79
	; sched_group_barrier mask(0x00000008) size(1) SyncID(2)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(2)
	v_mfma_f32_32x32x16_bf16 v[48:63], v[204:207], v[148:151], v[48:63]
	v_max3_f32 v96, v97, v78, v96
	v_mov_b32_e32 v97, v96
	s_nop 1
	v_permlane32_swap_b32_e64 v96, v97 bound_ctrl:1
	v_max3_f32 v96, v255, v96, v97
	v_sub_f32_e32 v80, v80, v96
	; sched_group_barrier mask(0x00000008) size(1) SyncID(2)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(2)
	v_mfma_f32_32x32x16_bf16 v[32:47], v[164:167], v[152:155], v[32:47]
	v_sub_f32_e32 v81, v81, v96
	v_sub_f32_e32 v82, v82, v96
	v_sub_f32_e32 v83, v83, v96
	v_sub_f32_e32 v84, v84, v96
	v_sub_f32_e32 v85, v85, v96
	; sched_group_barrier mask(0x00000008) size(1) SyncID(2)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(2)
	v_mfma_f32_32x32x16_bf16 v[32:47], v[176:179], v[156:159], v[32:47]
	v_sub_f32_e32 v86, v86, v96
	v_sub_f32_e32 v87, v87, v96
	v_sub_f32_e32 v88, v88, v96
	v_sub_f32_e32 v89, v89, v96
	v_sub_f32_e32 v90, v90, v96
	; sched_group_barrier mask(0x00000008) size(1) SyncID(2)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(2)
	v_mfma_f32_32x32x16_bf16 v[32:47], v[192:195], v[160:163], v[32:47]
	v_sub_f32_e32 v91, v91, v96
	v_sub_f32_e32 v92, v92, v96
	v_sub_f32_e32 v93, v93, v96
	v_sub_f32_e32 v94, v94, v96
	v_sub_f32_e32 v95, v95, v96
	; sched_group_barrier mask(0x00000008) size(1) SyncID(2)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(2)
	v_mfma_f32_32x32x16_bf16 v[32:47], v[208:211], v[148:151], v[32:47]
	v_sub_f32_e32 v97, v64, v96
	v_sub_f32_e32 v102, v65, v96
	v_sub_f32_e32 v104, v66, v96
	v_sub_f32_e32 v105, v67, v96
	v_sub_f32_e32 v106, v68, v96
	; sched_group_barrier mask(0x00000008) size(1) SyncID(2)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(2)
	v_mfma_f32_32x32x16_bf16 v[16:31], v[168:171], v[152:155], v[16:31]
	v_sub_f32_e32 v107, v69, v96
	v_sub_f32_e32 v111, v70, v96
	v_sub_f32_e32 v252, v71, v96
	v_sub_f32_e32 v251, v72, v96
	v_sub_f32_e32 v254, v73, v96
	; sched_group_barrier mask(0x00000008) size(1) SyncID(2)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(2)
	v_mfma_f32_32x32x16_bf16 v[16:31], v[180:183], v[156:159], v[16:31]
	v_sub_f32_e32 v64, v74, v96
	scratch_store_dword off, v64, off offset:32 ; 4-byte Folded Spill
	v_sub_f32_e32 v64, v76, v96
	scratch_store_dword off, v64, off offset:36 ; 4-byte Folded Spill
	v_sub_f32_e32 v64, v77, v96
	v_sub_f32_e32 v230, v75, v96
	scratch_store_dword off, v64, off offset:40 ; 4-byte Folded Spill
	v_sub_f32_e32 v64, v78, v96
	v_mfma_f32_32x32x16_bf16 v[16:31], v[196:199], v[160:163], v[16:31]
	v_exp_f32_e32 v237, v80
	v_exp_f32_e32 v232, v81
	v_exp_f32_e32 v233, v82
	scratch_store_dword off, v64, off offset:44 ; 4-byte Folded Spill
	v_sub_f32_e32 v64, v79, v96
	scratch_store_dword off, v64, off offset:28 ; 4-byte Folded Spill
	; sched_group_barrier mask(0x00000008) size(1) SyncID(2)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(2)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(2)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(2)
	v_mfma_f32_32x32x16_bf16 v[16:31], v[212:215], v[148:151], v[16:31]
	v_exp_f32_e32 v234, v83
	v_exp_f32_e32 v235, v84
	v_exp_f32_e32 v229, v85
	; sched_group_barrier mask(0x00000008) size(1) SyncID(2)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(2)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[144:147], v[152:155], v[0:15]
	v_exp_f32_e32 v231, v86
	v_exp_f32_e32 v228, v87
	v_exp_f32_e32 v246, v88
	; sched_group_barrier mask(0x00000008) size(1) SyncID(2)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(2)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[184:187], v[156:159], v[0:15]
	v_exp_f32_e32 v247, v89
	v_exp_f32_e32 v248, v90
	v_exp_f32_e32 v249, v91
	; sched_group_barrier mask(0x00000008) size(1) SyncID(2)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(2)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[200:203], v[160:163], v[0:15]
	v_exp_f32_e32 v250, v92
	v_exp_f32_e32 v253, v93
	v_exp_f32_e32 v108, v94
	; sched_group_barrier mask(0x00000008) size(1) SyncID(2)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(2)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[216:219], v[148:151], v[0:15]
	v_exp_f32_e32 v109, v95
	; sched_group_barrier mask(0x00000008) size(1) SyncID(2)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(2)
	; sched_barrier mask(0x00000000)
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	scratch_load_dword v64, off, off offset:212 ; 4-byte Folded Reload
	s_add_i32 s2, s21, s19
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[2:3], s[2:3], 1
	s_add_u32 s24, s40, s2
	s_addc_u32 s25, s41, s3
	s_mov_b32 s27, s39
	scratch_load_dword v68, off, off offset:208 ; 4-byte Folded Reload
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s2, v64
	scratch_load_dword v64, off, off offset:12 ; 4-byte Folded Reload
	s_mov_b32 m0, s2
	s_waitcnt vmcnt(0)
	buffer_load_dwordx4 v64, s[24:27], 0 offen lds
	scratch_load_dword v64, off, off offset:216 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s2, v64
	s_mov_b32 m0, s2
	s_nop 0
	buffer_load_dwordx4 v110, s[24:27], 0 offen lds
	;;#ASMSTART
	ds_read_b128 v[64:67], v68 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[80:83], v68 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[84:87], v68 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[88:91], v68 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[92:95], v68 offset:0x2000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[98:101], v68 offset:0x2800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[144:147], v68 offset:0x3000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[148:151], v68 offset:0x3800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[152:155], v68 offset:0x4000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[156:159], v68 offset:0x4800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[160:163], v68 offset:0x5000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[164:167], v68 offset:0x5800

	;;#ASMEND
	scratch_load_dword v68, off, off offset:220 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b128 v[168:171], v68 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[172:175], v68 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[176:179], v68 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[180:183], v68 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[184:187], v68 offset:0x2000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[188:191], v68 offset:0x2800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[192:195], v68 offset:0x3000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[196:199], v68 offset:0x3800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[200:203], v68 offset:0x4000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[204:207], v68 offset:0x4800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[208:211], v68 offset:0x5000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[212:215], v68 offset:0x5800

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_f32_32x32x16_bf16 v[64:79], v[64:67], v[220:223], 0
	v_exp_f32_e32 v110, v97
	v_exp_f32_e32 v102, v102
	v_exp_f32_e32 v104, v104
	; sched_group_barrier mask(0x00000008) size(1) SyncID(3)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(3)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[168:171], v[238:241], v[64:79]
	v_exp_f32_e32 v105, v105
	v_exp_f32_e32 v106, v106
	v_exp_f32_e32 v107, v107
	; sched_group_barrier mask(0x00000008) size(1) SyncID(3)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(3)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[80:83], v[224:227], v[64:79]
	scratch_load_dword v80, off, off offset:32 ; 4-byte Folded Reload
	v_exp_f32_e32 v111, v111
	v_exp_f32_e32 v168, v252
	v_exp_f32_e32 v169, v251
	; sched_group_barrier mask(0x00000008) size(1) SyncID(3)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(3)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[172:175], v[128:131], v[64:79]
	v_exp_f32_e32 v170, v254
	v_exp_f32_e32 v172, v230
	; sched_group_barrier mask(0x00000008) size(1) SyncID(3)
	s_waitcnt vmcnt(0)
	v_exp_f32_e32 v171, v80
	scratch_load_dword v80, off, off offset:36 ; 4-byte Folded Reload
	v_mfma_f32_32x32x16_bf16 v[64:79], v[84:87], v[136:139], v[64:79]
	; sched_group_barrier mask(0x00000400) size(3) SyncID(3)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(3)
	s_waitcnt vmcnt(0)
	v_exp_f32_e32 v173, v80
	scratch_load_dword v80, off, off offset:40 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_exp_f32_e32 v174, v80
	scratch_load_dword v80, off, off offset:44 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_exp_f32_e32 v175, v80
	scratch_load_dword v80, off, off offset:28 ; 4-byte Folded Reload
	v_mfma_f32_32x32x16_bf16 v[64:79], v[176:179], v[132:135], v[64:79]
	; sched_group_barrier mask(0x00000400) size(3) SyncID(3)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(3)
	s_waitcnt vmcnt(0)
	v_exp_f32_e32 v176, v80
	v_mfma_f32_32x32x16_bf16 v[64:79], v[88:91], v[124:127], v[64:79]
	v_add_f32_e32 v80, v232, v237
	v_add_f32_e32 v80, v80, v233
	v_add_f32_e32 v80, v80, v234
	v_add_f32_e32 v80, v80, v235
	v_add_f32_e32 v80, v80, v229
	; sched_group_barrier mask(0x00000400) size(3) SyncID(3)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(3)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(3)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[180:183], v[140:143], v[64:79]
	v_add_f32_e32 v80, v80, v231
	v_add_f32_e32 v80, v80, v228
	v_add_f32_e32 v80, v80, v246
	v_add_f32_e32 v80, v80, v247
	v_add_f32_e32 v80, v80, v248
	; sched_group_barrier mask(0x00000008) size(1) SyncID(3)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(3)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[92:95], v[116:119], v[64:79]
	v_add_f32_e32 v80, v80, v249
	v_add_f32_e32 v80, v80, v250
	v_add_f32_e32 v80, v80, v253
	v_add_f32_e32 v80, v80, v108
	v_add_f32_e32 v80, v80, v109
	; sched_group_barrier mask(0x00000008) size(1) SyncID(3)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(3)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[184:187], v[120:123], v[64:79]
	v_add_f32_e32 v80, v80, v110
	v_add_f32_e32 v80, v80, v102
	v_add_f32_e32 v80, v80, v104
	v_add_f32_e32 v80, v80, v105
	v_add_f32_e32 v80, v80, v106
	; sched_group_barrier mask(0x00000008) size(1) SyncID(3)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(3)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[98:101], v[112:115], v[64:79]
	v_add_f32_e32 v80, v80, v107
	v_add_f32_e32 v80, v80, v111
	v_add_f32_e32 v80, v80, v168
	v_add_f32_e32 v80, v80, v169
	v_add_f32_e32 v80, v80, v170
	; sched_group_barrier mask(0x00000008) size(1) SyncID(3)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(3)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[188:191], v[242:245], v[64:79]
	v_add_f32_e32 v80, v80, v171
	v_add_f32_e32 v80, v80, v172
	v_add_f32_e32 v80, v80, v173
	v_add_f32_e32 v80, v80, v174
	v_add_f32_e32 v97, v80, v175
	; sched_group_barrier mask(0x00000008) size(1) SyncID(3)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(3)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[144:147], v[220:223], 0
	v_add_f32_e32 v97, v97, v176
	v_mov_b32_e32 v98, v97
	s_nop 1
	v_permlane32_swap_b32_e64 v97, v98 bound_ctrl:1
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v144, v237, v232
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v145, v233, v234
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v146, v235, v229
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[80:95], v[192:195], v[238:241], v[80:95]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v147, v231, v228
	;;#ASMEND
	; sched_group_barrier mask(0x00000008) size(1) SyncID(3)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(3)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(3)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(3)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[148:151], v[224:227], v[80:95]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v148, v246, v247
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v149, v248, v249
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v150, v250, v253
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v151, v108, v109
	;;#ASMEND
	; sched_group_barrier mask(0x00000008) size(1) SyncID(3)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(3)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[196:199], v[128:131], v[80:95]
	; sched_group_barrier mask(0x00000008) size(1) SyncID(3)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(3)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[152:155], v[136:139], v[80:95]
	v_mfma_f32_32x32x16_bf16 v[80:95], v[200:203], v[132:135], v[80:95]
	v_mfma_f32_32x32x16_bf16 v[80:95], v[156:159], v[124:127], v[80:95]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v156, v110, v102
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v157, v104, v105
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v158, v106, v107
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v159, v111, v168
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v152, v169, v170
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v153, v171, v172
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v154, v173, v174
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[80:95], v[204:207], v[140:143], v[80:95]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v155, v175, v176
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[80:95], v[160:163], v[116:119], v[80:95]
	v_mfma_f32_32x32x16_bf16 v[80:95], v[208:211], v[120:123], v[80:95]
	v_mfma_f32_32x32x16_bf16 v[80:95], v[164:167], v[112:115], v[80:95]
	v_mfma_f32_32x32x16_bf16 v[80:95], v[212:215], v[242:245], v[80:95]
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	scratch_load_dword v99, off, off offset:236 ; 4-byte Folded Reload
	s_add_i32 s2, s20, s46
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[2:3], s[2:3], 1
	s_add_u32 s36, s42, s2
	s_addc_u32 s37, s43, s3
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s2, v99
	scratch_load_dword v99, off, off        ; 4-byte Folded Reload
	s_mov_b32 m0, s2
	s_waitcnt vmcnt(0)
	buffer_load_dwordx4 v99, s[36:39], 0 offen lds
	scratch_load_dword v99, off, off offset:240 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s2, v99
	scratch_load_dword v99, off, off offset:4 ; 4-byte Folded Reload
	s_mov_b32 m0, s2
	s_waitcnt vmcnt(0)
	buffer_load_dwordx4 v99, s[36:39], 0 offen lds
	scratch_load_dword v99, off, off offset:244 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s2, v99
	scratch_load_dword v99, off, off offset:8 ; 4-byte Folded Reload
	s_mov_b32 m0, s2
	s_waitcnt vmcnt(0)
	buffer_load_dwordx4 v99, s[36:39], 0 offen lds
	scratch_load_dword v99, off, off offset:152 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b16 v[220:221], v99 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[204:205], v99 offset:0x200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[188:189], v99 offset:0x400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[172:173], v99 offset:0x600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[216:217], v99 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[200:201], v99 offset:0x1200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[184:185], v99 offset:0x1400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[168:169], v99 offset:0x1600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[212:213], v99 offset:0x2000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[196:197], v99 offset:0x2200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[180:181], v99 offset:0x2400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[164:165], v99 offset:0x2600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[208:209], v99 offset:0x3000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[192:193], v99 offset:0x3200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[176:177], v99 offset:0x3400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[160:161], v99 offset:0x3600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[222:223], v99 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[206:207], v99 offset:0xa00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[190:191], v99 offset:0xc00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[174:175], v99 offset:0xe00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[218:219], v99 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[202:203], v99 offset:0x1a00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[186:187], v99 offset:0x1c00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[170:171], v99 offset:0x1e00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[214:215], v99 offset:0x2800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[198:199], v99 offset:0x2a00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[182:183], v99 offset:0x2c00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[166:167], v99 offset:0x2e00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[210:211], v99 offset:0x3800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[194:195], v99 offset:0x3a00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[178:179], v99 offset:0x3c00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[162:163], v99 offset:0x3e00

	;;#ASMEND
	scratch_load_dword v99, off, off offset:48 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_cmp_gt_i32_e32 vcc, s29, v99
	s_mov_b64 s[2:3], exec
	scratch_load_dword v120, off, off offset:16 ; 4-byte Folded Reload
	s_and_b64 s[24:25], s[2:3], vcc
	s_mov_b64 exec, s[24:25]
	s_cbranch_execz .LBB0_31
; %bb.30:                               ;   in Loop: Header=BB0_29 Depth=1
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v99, 32, v120
	v_mov_b32_e32 v100, 0xff800000
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[24:25], v99, 0
	v_cmp_lt_i32_e64 s[36:37], v99, 1
	v_cndmask_b32_e64 v64, v64, v100, s[24:25]
	v_cndmask_b32_e64 v65, v65, v100, s[36:37]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[24:25], v99, 2
	v_cmp_lt_i32_e64 s[36:37], v99, 3
	v_cndmask_b32_e64 v66, v66, v100, s[24:25]
	v_cndmask_b32_e64 v67, v67, v100, s[36:37]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[24:25], v99, 8
	v_cmp_lt_i32_e64 s[36:37], v99, 9
	v_cndmask_b32_e64 v68, v68, v100, s[24:25]
	v_cndmask_b32_e64 v69, v69, v100, s[36:37]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[24:25], v99, 10
	v_cmp_lt_i32_e64 s[36:37], v99, 11
	v_cndmask_b32_e64 v70, v70, v100, s[24:25]
	v_cndmask_b32_e64 v71, v71, v100, s[36:37]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[24:25], v99, 16
	v_cmp_lt_i32_e64 s[36:37], v99, 17
	v_cndmask_b32_e64 v72, v72, v100, s[24:25]
	v_cndmask_b32_e64 v73, v73, v100, s[36:37]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[24:25], v99, 18
	v_cmp_lt_i32_e64 s[36:37], v99, 19
	v_cndmask_b32_e64 v74, v74, v100, s[24:25]
	v_cndmask_b32_e64 v75, v75, v100, s[36:37]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[24:25], v99, 24
	v_cmp_lt_i32_e64 s[36:37], v99, 25
	v_cndmask_b32_e64 v76, v76, v100, s[24:25]
	v_cndmask_b32_e64 v77, v77, v100, s[36:37]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[24:25], v99, 26
	v_cmp_lt_i32_e64 s[36:37], v99, 27
	v_cndmask_b32_e64 v78, v78, v100, s[24:25]
	v_cndmask_b32_e64 v79, v79, v100, s[36:37]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[24:25], v120, 0
	v_cmp_lt_i32_e64 s[36:37], v120, 1
	v_cndmask_b32_e64 v80, v80, v100, s[24:25]
	v_cndmask_b32_e64 v81, v81, v100, s[36:37]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[24:25], v120, 2
	v_cmp_lt_i32_e64 s[36:37], v120, 3
	v_cndmask_b32_e64 v82, v82, v100, s[24:25]
	v_cndmask_b32_e64 v83, v83, v100, s[36:37]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[24:25], v120, 8
	v_cmp_lt_i32_e64 s[36:37], v120, 9
	v_cndmask_b32_e64 v84, v84, v100, s[24:25]
	v_cndmask_b32_e64 v85, v85, v100, s[36:37]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[24:25], v120, 10
	v_cmp_lt_i32_e64 s[36:37], v120, 11
	v_cndmask_b32_e64 v86, v86, v100, s[24:25]
	v_cndmask_b32_e64 v87, v87, v100, s[36:37]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[24:25], v120, 16
	v_cmp_lt_i32_e64 s[36:37], v120, 17
	v_cndmask_b32_e64 v88, v88, v100, s[24:25]
	v_cndmask_b32_e64 v89, v89, v100, s[36:37]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[24:25], v120, 18
	v_cmp_lt_i32_e64 s[36:37], v120, 19
	v_cndmask_b32_e64 v90, v90, v100, s[24:25]
	v_cndmask_b32_e64 v91, v91, v100, s[36:37]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[24:25], v120, 24
	v_cmp_lt_i32_e64 s[36:37], v120, 25
	v_cndmask_b32_e64 v92, v92, v100, s[24:25]
	v_cndmask_b32_e64 v93, v93, v100, s[36:37]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[24:25], v120, 26
	v_cmp_lt_i32_e64 s[36:37], v120, 27
	v_cndmask_b32_e64 v94, v94, v100, s[24:25]
	v_cndmask_b32_e64 v95, v95, v100, s[36:37]
	
	;;#ASMEND
.LBB0_31:                               ;   in Loop: Header=BB0_29 Depth=1
	s_or_b64 exec, exec, s[2:3]
	scratch_load_dword v99, off, off offset:20 ; 4-byte Folded Reload
	v_sub_f32_e32 v100, v255, v96
	v_exp_f32_e32 v224, v100
	s_waitcnt vmcnt(0)
	v_fmac_f32_e32 v103, v99, v236
	scratch_load_dword v99, off, off offset:24 ; 4-byte Folded Reload
	v_pk_mul_f32 v[62:63], v[224:225], v[62:63] op_sel_hi:[0,1]
	v_pk_mul_f32 v[60:61], v[224:225], v[60:61] op_sel_hi:[0,1]
	v_pk_mul_f32 v[58:59], v[224:225], v[58:59] op_sel_hi:[0,1]
	v_pk_mul_f32 v[56:57], v[224:225], v[56:57] op_sel_hi:[0,1]
	v_pk_mul_f32 v[54:55], v[224:225], v[54:55] op_sel_hi:[0,1]
	v_pk_mul_f32 v[52:53], v[224:225], v[52:53] op_sel_hi:[0,1]
	v_pk_mul_f32 v[50:51], v[224:225], v[50:51] op_sel_hi:[0,1]
	v_pk_mul_f32 v[48:49], v[224:225], v[48:49] op_sel_hi:[0,1]
	v_pk_mul_f32 v[46:47], v[224:225], v[46:47] op_sel_hi:[0,1]
	v_pk_mul_f32 v[44:45], v[224:225], v[44:45] op_sel_hi:[0,1]
	v_pk_mul_f32 v[42:43], v[224:225], v[42:43] op_sel_hi:[0,1]
	v_pk_mul_f32 v[40:41], v[224:225], v[40:41] op_sel_hi:[0,1]
	v_pk_mul_f32 v[38:39], v[224:225], v[38:39] op_sel_hi:[0,1]
	v_pk_mul_f32 v[36:37], v[224:225], v[36:37] op_sel_hi:[0,1]
	v_pk_mul_f32 v[34:35], v[224:225], v[34:35] op_sel_hi:[0,1]
	v_pk_mul_f32 v[32:33], v[224:225], v[32:33] op_sel_hi:[0,1]
	v_pk_mul_f32 v[30:31], v[224:225], v[30:31] op_sel_hi:[0,1]
	v_pk_mul_f32 v[28:29], v[224:225], v[28:29] op_sel_hi:[0,1]
	v_pk_mul_f32 v[26:27], v[224:225], v[26:27] op_sel_hi:[0,1]
	v_pk_mul_f32 v[24:25], v[224:225], v[24:25] op_sel_hi:[0,1]
	v_pk_mul_f32 v[22:23], v[224:225], v[22:23] op_sel_hi:[0,1]
	v_pk_mul_f32 v[20:21], v[224:225], v[20:21] op_sel_hi:[0,1]
	v_pk_mul_f32 v[18:19], v[224:225], v[18:19] op_sel_hi:[0,1]
	v_pk_mul_f32 v[16:17], v[224:225], v[16:17] op_sel_hi:[0,1]
	v_pk_mul_f32 v[14:15], v[224:225], v[14:15] op_sel_hi:[0,1]
	v_pk_mul_f32 v[12:13], v[224:225], v[12:13] op_sel_hi:[0,1]
	v_pk_mul_f32 v[10:11], v[224:225], v[10:11] op_sel_hi:[0,1]
	v_pk_mul_f32 v[8:9], v[224:225], v[8:9] op_sel_hi:[0,1]
	v_pk_mul_f32 v[6:7], v[224:225], v[6:7] op_sel_hi:[0,1]
	v_pk_mul_f32 v[4:5], v[224:225], v[4:5] op_sel_hi:[0,1]
	v_pk_mul_f32 v[2:3], v[224:225], v[2:3] op_sel_hi:[0,1]
	v_pk_mul_f32 v[0:1], v[224:225], v[0:1] op_sel_hi:[0,1]
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	s_waitcnt vmcnt(0)
	v_add_f32_e32 v99, v103, v99
	v_fmac_f32_e32 v97, v224, v99
	v_add_f32_e32 v121, v97, v98
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_f32_32x32x16_bf16 v[48:63], v[220:223], v[144:147], v[48:63]
	v_max_f32_e32 v97, v65, v67
	v_max3_f32 v98, v64, v66, v68
	v_max3_f32 v97, v97, v69, v71
	v_max3_f32 v98, v98, v70, v72
	v_max3_f32 v97, v97, v73, v75
	; sched_group_barrier mask(0x00000008) size(1) SyncID(4)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(4)
	v_mfma_f32_32x32x16_bf16 v[48:63], v[216:219], v[148:151], v[48:63]
	v_max3_f32 v98, v98, v74, v76
	v_max3_f32 v97, v97, v77, v79
	v_max3_f32 v98, v98, v78, v80
	v_max3_f32 v97, v97, v81, v83
	v_max3_f32 v98, v98, v82, v84
	; sched_group_barrier mask(0x00000008) size(1) SyncID(4)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(4)
	v_mfma_f32_32x32x16_bf16 v[48:63], v[212:215], v[156:159], v[48:63]
	v_max3_f32 v97, v97, v85, v87
	v_max3_f32 v98, v98, v86, v88
	v_max3_f32 v97, v97, v89, v91
	v_max3_f32 v98, v98, v90, v92
	v_max3_f32 v97, v97, v93, v95
	; sched_group_barrier mask(0x00000008) size(1) SyncID(4)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(4)
	v_mfma_f32_32x32x16_bf16 v[48:63], v[208:211], v[152:155], v[48:63]
	v_max3_f32 v97, v98, v94, v97
	v_mov_b32_e32 v98, v97
	s_nop 1
	v_permlane32_swap_b32_e64 v97, v98 bound_ctrl:1
	v_max3_f32 v255, v96, v97, v98
	v_sub_f32_e32 v96, v96, v255
	; sched_group_barrier mask(0x00000008) size(1) SyncID(4)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(4)
	v_mfma_f32_32x32x16_bf16 v[32:47], v[204:207], v[144:147], v[32:47]
	v_sub_f32_e32 v97, v64, v255
	v_sub_f32_e32 v98, v65, v255
	v_sub_f32_e32 v99, v66, v255
	v_sub_f32_e32 v101, v67, v255
	v_sub_f32_e32 v103, v68, v255
	; sched_group_barrier mask(0x00000008) size(1) SyncID(4)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(4)
	v_mfma_f32_32x32x16_bf16 v[32:47], v[200:203], v[148:151], v[32:47]
	v_sub_f32_e32 v104, v69, v255
	v_sub_f32_e32 v105, v70, v255
	v_sub_f32_e32 v106, v71, v255
	v_sub_f32_e32 v107, v72, v255
	v_sub_f32_e32 v108, v73, v255
	; sched_group_barrier mask(0x00000008) size(1) SyncID(4)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(4)
	v_mfma_f32_32x32x16_bf16 v[32:47], v[196:199], v[156:159], v[32:47]
	v_sub_f32_e32 v109, v74, v255
	v_sub_f32_e32 v110, v75, v255
	v_sub_f32_e32 v111, v76, v255
	v_sub_f32_e32 v196, v77, v255
	v_sub_f32_e32 v78, v78, v255
	; sched_group_barrier mask(0x00000008) size(1) SyncID(4)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(4)
	v_mfma_f32_32x32x16_bf16 v[32:47], v[192:195], v[152:155], v[32:47]
	v_sub_f32_e32 v197, v79, v255
	v_sub_f32_e32 v65, v95, v255
	v_sub_f32_e32 v64, v94, v255
	v_sub_f32_e32 v67, v93, v255
	v_sub_f32_e32 v66, v92, v255
	v_sub_f32_e32 v79, v81, v255
	; sched_group_barrier mask(0x00000008) size(1) SyncID(4)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(4)
	v_mfma_f32_32x32x16_bf16 v[16:31], v[188:191], v[144:147], v[16:31]
	v_sub_f32_e32 v69, v91, v255
	v_sub_f32_e32 v68, v90, v255
	v_sub_f32_e32 v71, v89, v255
	v_sub_f32_e32 v70, v88, v255
	v_sub_f32_e32 v73, v87, v255
	; sched_group_barrier mask(0x00000008) size(1) SyncID(4)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(4)
	v_mfma_f32_32x32x16_bf16 v[16:31], v[184:187], v[148:151], v[16:31]
	v_sub_f32_e32 v72, v86, v255
	v_sub_f32_e32 v75, v85, v255
	v_sub_f32_e32 v74, v84, v255
	v_sub_f32_e32 v77, v83, v255
	v_sub_f32_e32 v76, v82, v255
	; sched_group_barrier mask(0x00000008) size(1) SyncID(4)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(4)
	v_mfma_f32_32x32x16_bf16 v[16:31], v[180:183], v[156:159], v[16:31]
	v_exp_f32_e32 v100, v97
	v_exp_f32_e32 v102, v98
	v_exp_f32_e32 v236, v96
	; sched_group_barrier mask(0x00000008) size(1) SyncID(4)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(4)
	v_mfma_f32_32x32x16_bf16 v[16:31], v[176:179], v[152:155], v[16:31]
	v_exp_f32_e32 v98, v99
	v_exp_f32_e32 v101, v101
	v_exp_f32_e32 v250, v103
	; sched_group_barrier mask(0x00000008) size(1) SyncID(4)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(4)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[172:175], v[144:147], v[0:15]
	v_exp_f32_e32 v99, v104
	v_exp_f32_e32 v248, v105
	v_exp_f32_e32 v97, v106
	; sched_group_barrier mask(0x00000008) size(1) SyncID(4)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(4)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[168:171], v[148:151], v[0:15]
	v_exp_f32_e32 v251, v107
	v_exp_f32_e32 v96, v108
	v_exp_f32_e32 v108, v109
	; sched_group_barrier mask(0x00000008) size(1) SyncID(4)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(4)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[164:167], v[156:159], v[0:15]
	v_exp_f32_e32 v249, v110
	v_exp_f32_e32 v253, v111
	v_exp_f32_e32 v109, v196
	; sched_group_barrier mask(0x00000008) size(1) SyncID(4)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(4)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[160:163], v[152:155], v[0:15]
	v_exp_f32_e32 v246, v78
	v_exp_f32_e32 v247, v197
	v_sub_f32_e32 v78, v80, v255
	; sched_group_barrier mask(0x00000008) size(1) SyncID(4)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(4)
	; sched_barrier mask(0x00000000)
	v_pk_mul_f32 v[62:63], v[236:237], v[62:63] op_sel_hi:[0,1]
	v_pk_mul_f32 v[60:61], v[236:237], v[60:61] op_sel_hi:[0,1]
	v_pk_mul_f32 v[58:59], v[236:237], v[58:59] op_sel_hi:[0,1]
	v_pk_mul_f32 v[56:57], v[236:237], v[56:57] op_sel_hi:[0,1]
	v_pk_mul_f32 v[54:55], v[236:237], v[54:55] op_sel_hi:[0,1]
	v_pk_mul_f32 v[52:53], v[236:237], v[52:53] op_sel_hi:[0,1]
	v_pk_mul_f32 v[50:51], v[236:237], v[50:51] op_sel_hi:[0,1]
	v_pk_mul_f32 v[48:49], v[236:237], v[48:49] op_sel_hi:[0,1]
	v_pk_mul_f32 v[46:47], v[236:237], v[46:47] op_sel_hi:[0,1]
	v_pk_mul_f32 v[44:45], v[236:237], v[44:45] op_sel_hi:[0,1]
	v_pk_mul_f32 v[42:43], v[236:237], v[42:43] op_sel_hi:[0,1]
	v_pk_mul_f32 v[40:41], v[236:237], v[40:41] op_sel_hi:[0,1]
	v_pk_mul_f32 v[38:39], v[236:237], v[38:39] op_sel_hi:[0,1]
	v_pk_mul_f32 v[36:37], v[236:237], v[36:37] op_sel_hi:[0,1]
	v_pk_mul_f32 v[34:35], v[236:237], v[34:35] op_sel_hi:[0,1]
	v_pk_mul_f32 v[32:33], v[236:237], v[32:33] op_sel_hi:[0,1]
	v_pk_mul_f32 v[30:31], v[236:237], v[30:31] op_sel_hi:[0,1]
	v_pk_mul_f32 v[28:29], v[236:237], v[28:29] op_sel_hi:[0,1]
	v_pk_mul_f32 v[26:27], v[236:237], v[26:27] op_sel_hi:[0,1]
	v_pk_mul_f32 v[24:25], v[236:237], v[24:25] op_sel_hi:[0,1]
	v_pk_mul_f32 v[22:23], v[236:237], v[22:23] op_sel_hi:[0,1]
	v_pk_mul_f32 v[20:21], v[236:237], v[20:21] op_sel_hi:[0,1]
	v_pk_mul_f32 v[18:19], v[236:237], v[18:19] op_sel_hi:[0,1]
	v_pk_mul_f32 v[16:17], v[236:237], v[16:17] op_sel_hi:[0,1]
	v_pk_mul_f32 v[14:15], v[236:237], v[14:15] op_sel_hi:[0,1]
	v_pk_mul_f32 v[12:13], v[236:237], v[12:13] op_sel_hi:[0,1]
	v_pk_mul_f32 v[10:11], v[236:237], v[10:11] op_sel_hi:[0,1]
	v_pk_mul_f32 v[8:9], v[236:237], v[8:9] op_sel_hi:[0,1]
	v_pk_mul_f32 v[6:7], v[236:237], v[6:7] op_sel_hi:[0,1]
	v_pk_mul_f32 v[4:5], v[236:237], v[4:5] op_sel_hi:[0,1]
	v_pk_mul_f32 v[2:3], v[236:237], v[2:3] op_sel_hi:[0,1]
	v_pk_mul_f32 v[0:1], v[236:237], v[0:1] op_sel_hi:[0,1]
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	scratch_load_dword v80, off, off offset:200 ; 4-byte Folded Reload
	s_add_i32 s2, s35, s19
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[2:3], s[2:3], 1
	s_add_u32 s24, s40, s2
	scratch_load_dword v85, off, off offset:12 ; 4-byte Folded Reload
	scratch_load_dword v110, off, off offset:116 ; 4-byte Folded Reload
	s_addc_u32 s25, s41, s3
	s_mov_b32 s27, s39
	scratch_load_dword v84, off, off offset:192 ; 4-byte Folded Reload
	s_waitcnt vmcnt(3)
	v_readfirstlane_b32 s2, v80
	scratch_load_dword v80, off, off offset:204 ; 4-byte Folded Reload
	s_mov_b32 m0, s2
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s2, v80
	buffer_load_dwordx4 v85, s[24:27], 0 offen lds
	s_mov_b32 m0, s2
	s_nop 0
	buffer_load_dwordx4 v110, s[24:27], 0 offen lds
	;;#ASMSTART
	ds_read_b128 v[80:83], v84 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[208:211], v84 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[184:187], v84 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[180:183], v84 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[176:179], v84 offset:0x2000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[172:175], v84 offset:0x2800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[168:171], v84 offset:0x3000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[152:155], v84 offset:0x3800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[156:159], v84 offset:0x4000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[160:163], v84 offset:0x4800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[164:167], v84 offset:0x5000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[148:151], v84 offset:0x5800

	;;#ASMEND
	scratch_load_dword v84, off, off offset:196 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b128 v[232:235], v84 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[228:231], v84 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[124:127], v84 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[112:115], v84 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[216:219], v84 offset:0x2000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[212:215], v84 offset:0x2800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[188:191], v84 offset:0x3000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[196:199], v84 offset:0x3800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[200:203], v84 offset:0x4000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[204:207], v84 offset:0x4800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[192:195], v84 offset:0x5000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[144:147], v84 offset:0x5800

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	scratch_load_dwordx4 v[116:119], off, off offset:280 ; 16-byte Folded Reload
	s_add_i32 s17, s17, 2
	s_add_i32 s46, s46, s38
	s_add_i32 s19, s19, s26
	s_addk_i32 s29, 0x80
	s_cmp_ge_i32 s17, s15
	v_add_u32_e32 v120, 0xffffff80, v120
	s_cbranch_scc1 .LBB0_34
; %bb.32:                               ;   in Loop: Header=BB0_29 Depth=1
	scratch_store_dword off, v121, off offset:20 ; 4-byte Folded Spill
	scratch_store_dword off, v120, off offset:16 ; 4-byte Folded Spill
	s_branch .LBB0_29
.LBB0_33:
	v_mov_b32_e32 v14, v15
	v_mov_b32_e32 v13, v15
	v_mov_b32_e32 v12, v15
	v_mov_b32_e32 v11, v15
	v_mov_b32_e32 v10, v15
	v_mov_b32_e32 v9, v15
	v_mov_b32_e32 v8, v15
	v_mov_b32_e32 v7, v15
	v_mov_b32_e32 v6, v15
	v_mov_b32_e32 v5, v15
	v_mov_b32_e32 v4, v15
	v_mov_b32_e32 v3, v15
	v_mov_b32_e32 v2, v15
	v_mov_b32_e32 v1, v15
	v_mov_b32_e32 v0, v15
	v_mov_b32_e32 v31, v15
	v_mov_b32_e32 v30, v15
	v_mov_b32_e32 v29, v15
	v_mov_b32_e32 v28, v15
	v_mov_b32_e32 v27, v15
	v_mov_b32_e32 v26, v15
	v_mov_b32_e32 v25, v15
	v_mov_b32_e32 v24, v15
	v_mov_b32_e32 v23, v15
	v_mov_b32_e32 v22, v15
	v_mov_b32_e32 v21, v15
	v_mov_b32_e32 v20, v15
	v_mov_b32_e32 v19, v15
	v_mov_b32_e32 v18, v15
	v_mov_b32_e32 v17, v15
	v_mov_b32_e32 v16, v15
	v_mov_b32_e32 v47, v15
	v_mov_b32_e32 v46, v15
	v_mov_b32_e32 v45, v15
	v_mov_b32_e32 v44, v15
	v_mov_b32_e32 v43, v15
	v_mov_b32_e32 v42, v15
	v_mov_b32_e32 v41, v15
	v_mov_b32_e32 v40, v15
	v_mov_b32_e32 v39, v15
	v_mov_b32_e32 v38, v15
	v_mov_b32_e32 v37, v15
	v_mov_b32_e32 v36, v15
	v_mov_b32_e32 v35, v15
	v_mov_b32_e32 v34, v15
	v_mov_b32_e32 v33, v15
	v_mov_b32_e32 v32, v15
	v_mov_b32_e32 v63, v15
	v_mov_b32_e32 v62, v15
	v_mov_b32_e32 v61, v15
	v_mov_b32_e32 v60, v15
	v_mov_b32_e32 v59, v15
	v_mov_b32_e32 v58, v15
	v_mov_b32_e32 v57, v15
	v_mov_b32_e32 v56, v15
	v_mov_b32_e32 v55, v15
	v_mov_b32_e32 v54, v15
	v_mov_b32_e32 v53, v15
	v_mov_b32_e32 v52, v15
	v_mov_b32_e32 v51, v15
	v_mov_b32_e32 v50, v15
	v_mov_b32_e32 v49, v15
	v_mov_b32_e32 v48, v15
	v_mov_b32_e32 v84, v15
	s_branch .LBB0_35
.LBB0_34:                               ; %._crit_edge.loopexit
	v_mul_f32_e32 v84, v121, v236
.LBB0_35:                               ; %Flow
	scratch_load_dwordx4 v[220:223], off, off offset:68 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[240:243], off, off offset:84 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[224:227], off, off offset:100 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[136:139], off, off offset:120 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[120:123], off, off offset:156 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[104:107], off, off offset:52 ; 16-byte Folded Reload
	s_nop 0
	scratch_store_dword off, v84, off offset:16 ; 4-byte Folded Spill
	s_waitcnt vmcnt(7)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[80:83], v[116:119], 0
	v_exp_f32_e32 v103, v78
	v_exp_f32_e32 v237, v79
	v_exp_f32_e32 v239, v76
	; sched_group_barrier mask(0x00000008) size(1) SyncID(5)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(5)
	s_waitcnt vmcnt(6)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[232:235], v[220:223], v[80:95]
	v_exp_f32_e32 v232, v77
	v_exp_f32_e32 v233, v74
	v_exp_f32_e32 v234, v75
	; sched_group_barrier mask(0x00000008) size(1) SyncID(5)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(5)
	s_waitcnt vmcnt(5)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[208:211], v[240:243], v[80:95]
	v_exp_f32_e32 v208, v72
	v_exp_f32_e32 v209, v73
	v_exp_f32_e32 v210, v70
	; sched_group_barrier mask(0x00000008) size(1) SyncID(5)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(5)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[228:231], v[128:131], v[80:95]
	v_exp_f32_e32 v211, v71
	v_exp_f32_e32 v228, v68
	v_exp_f32_e32 v229, v69
	; sched_group_barrier mask(0x00000008) size(1) SyncID(5)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(5)
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[184:187], v[224:227], v[80:95]
	v_exp_f32_e32 v184, v66
	v_exp_f32_e32 v185, v67
	v_exp_f32_e32 v186, v64
	; sched_group_barrier mask(0x00000008) size(1) SyncID(5)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(5)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[124:127], v[132:135], v[80:95]
	v_exp_f32_e32 v187, v65
	scratch_load_dwordx4 v[124:127], off, off offset:172 ; 16-byte Folded Reload
	; sched_group_barrier mask(0x00000008) size(1) SyncID(5)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(5)
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[180:183], v[136:139], v[80:95]
	v_add_f32_e32 v64, v102, v100
	v_add_f32_e32 v64, v64, v98
	v_add_f32_e32 v64, v64, v101
	v_add_f32_e32 v64, v64, v250
	v_add_f32_e32 v64, v64, v99
	; sched_group_barrier mask(0x00000008) size(1) SyncID(5)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(5)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[112:115], v[140:143], v[80:95]
	scratch_load_dwordx4 v[112:115], off, off offset:136 ; 16-byte Folded Reload
	v_add_f32_e32 v64, v64, v248
	v_add_f32_e32 v64, v64, v97
	v_add_f32_e32 v64, v64, v251
	v_add_f32_e32 v64, v64, v96
	v_add_f32_e32 v64, v64, v108
	; sched_group_barrier mask(0x00000008) size(1) SyncID(5)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(5)
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[176:179], v[120:123], v[80:95]
	v_add_f32_e32 v64, v64, v249
	v_add_f32_e32 v64, v64, v253
	v_add_f32_e32 v64, v64, v109
	v_add_f32_e32 v64, v64, v246
	v_add_f32_e32 v64, v64, v247
	; sched_group_barrier mask(0x00000008) size(1) SyncID(5)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(5)
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[216:219], v[124:127], v[80:95]
	v_add_f32_e32 v64, v64, v103
	v_add_f32_e32 v64, v64, v237
	v_add_f32_e32 v64, v64, v239
	v_add_f32_e32 v64, v64, v232
	v_add_f32_e32 v64, v64, v233
	; sched_group_barrier mask(0x00000008) size(1) SyncID(5)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(5)
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[172:175], v[112:115], v[80:95]
	v_add_f32_e32 v64, v64, v234
	v_add_f32_e32 v64, v64, v208
	v_add_f32_e32 v64, v64, v209
	v_add_f32_e32 v64, v64, v210
	v_add_f32_e32 v64, v64, v211
	; sched_group_barrier mask(0x00000008) size(1) SyncID(5)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(5)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[212:215], v[104:107], v[80:95]
	v_add_f32_e32 v64, v64, v228
	v_add_f32_e32 v64, v64, v229
	v_add_f32_e32 v64, v64, v184
	v_add_f32_e32 v64, v64, v185
	v_add_f32_e32 v172, v64, v186
	; sched_group_barrier mask(0x00000008) size(1) SyncID(5)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(5)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[168:171], v[116:119], 0
	v_add_f32_e32 v236, v172, v187
	v_mov_b32_e32 v110, v236
	s_nop 1
	v_permlane32_swap_b32_e64 v236, v110 bound_ctrl:1
	scratch_store_dword off, v110, off offset:20 ; 4-byte Folded Spill
	; sched_group_barrier mask(0x00000008) size(1) SyncID(5)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(5)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[188:191], v[220:223], v[64:79]
	; sched_group_barrier mask(0x00000008) size(1) SyncID(5)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(5)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[152:155], v[240:243], v[64:79]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v152, v100, v102
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v153, v98, v101
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v154, v250, v99
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v155, v248, v97
	;;#ASMEND
	; sched_group_barrier mask(0x00000008) size(1) SyncID(5)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(5)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[196:199], v[128:131], v[64:79]
	; sched_group_barrier mask(0x00000008) size(1) SyncID(5)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(5)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[156:159], v[224:227], v[64:79]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v156, v251, v96
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v157, v108, v249
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v158, v253, v109
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v159, v246, v247
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[64:79], v[200:203], v[132:135], v[64:79]
	v_mfma_f32_32x32x16_bf16 v[64:79], v[160:163], v[136:139], v[64:79]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v160, v103, v237
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v161, v239, v232
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v162, v233, v234
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v163, v208, v209
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[64:79], v[204:207], v[140:143], v[64:79]
	v_mfma_f32_32x32x16_bf16 v[64:79], v[164:167], v[120:123], v[64:79]
	v_mfma_f32_32x32x16_bf16 v[64:79], v[192:195], v[124:127], v[64:79]
	v_mfma_f32_32x32x16_bf16 v[64:79], v[148:151], v[112:115], v[64:79]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v148, v210, v211
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v149, v228, v229
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v150, v184, v185
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v151, v186, v187
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[64:79], v[144:147], v[104:107], v[64:79]
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	scratch_load_dword v96, off, off offset:224 ; 4-byte Folded Reload
	s_lshl_b32 s15, s15, 6
	s_add_i32 s2, s44, s15
	s_mul_i32 s2, s2, s30
	s_add_i32 s2, s2, s31
	s_mul_i32 s2, s2, s33
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[2:3], s[2:3], 1
	s_add_u32 s36, s42, s2
	s_mov_b32 s27, 0x110000
	s_addc_u32 s37, s43, s3
	s_mov_b32 s39, s27
	s_lshl_b32 s13, s13, 6
	s_add_i32 s17, s13, 0xffffff80
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s2, v96
	scratch_load_dword v96, off, off        ; 4-byte Folded Reload
	s_mov_b32 m0, s2
	s_waitcnt vmcnt(0)
	buffer_load_dwordx4 v96, s[36:39], 0 offen lds
	scratch_load_dword v96, off, off offset:228 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s2, v96
	scratch_load_dword v96, off, off offset:4 ; 4-byte Folded Reload
	s_mov_b32 m0, s2
	s_waitcnt vmcnt(0)
	buffer_load_dwordx4 v96, s[36:39], 0 offen lds
	scratch_load_dword v96, off, off offset:232 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s2, v96
	scratch_load_dword v96, off, off offset:8 ; 4-byte Folded Reload
	s_mov_b32 m0, s2
	s_waitcnt vmcnt(0)
	buffer_load_dwordx4 v96, s[36:39], 0 offen lds
	scratch_load_dword v96, off, off offset:188 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b16 v[220:221], v96 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[204:205], v96 offset:0x200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[188:189], v96 offset:0x400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[172:173], v96 offset:0x600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[216:217], v96 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[200:201], v96 offset:0x1200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[184:185], v96 offset:0x1400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[168:169], v96 offset:0x1600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[212:213], v96 offset:0x2000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[196:197], v96 offset:0x2200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[180:181], v96 offset:0x2400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[164:165], v96 offset:0x2600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[208:209], v96 offset:0x3000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[192:193], v96 offset:0x3200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[176:177], v96 offset:0x3400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[144:145], v96 offset:0x3600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[222:223], v96 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[206:207], v96 offset:0xa00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[190:191], v96 offset:0xc00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[174:175], v96 offset:0xe00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[218:219], v96 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[202:203], v96 offset:0x1a00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[186:187], v96 offset:0x1c00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[170:171], v96 offset:0x1e00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[214:215], v96 offset:0x2800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[198:199], v96 offset:0x2a00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[182:183], v96 offset:0x2c00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[166:167], v96 offset:0x2e00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[210:211], v96 offset:0x3800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[194:195], v96 offset:0x3a00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[178:179], v96 offset:0x3c00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[146:147], v96 offset:0x3e00

	;;#ASMEND
	scratch_load_dword v96, off, off offset:48 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_cmp_gt_i32_e32 vcc, s17, v96
	s_mov_b64 s[2:3], exec
	scratch_load_dword v240, off, off offset:316 ; 4-byte Folded Reload
	s_and_b64 s[20:21], s[2:3], vcc
	s_mov_b64 exec, s[20:21]
	s_cbranch_execnz .LBB0_76
.LBB0_36:
	s_or_b64 exec, exec, s[2:3]
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_f32_32x32x16_bf16 v[48:63], v[220:223], v[152:155], v[48:63]
	v_max_f32_e32 v96, v81, v83
	v_max3_f32 v97, v80, v82, v84
	v_max3_f32 v96, v96, v85, v87
	v_max3_f32 v97, v97, v86, v88
	v_max3_f32 v96, v96, v89, v91
	; sched_group_barrier mask(0x00000008) size(1) SyncID(6)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(6)
	v_mfma_f32_32x32x16_bf16 v[48:63], v[216:219], v[156:159], v[48:63]
	v_max3_f32 v97, v97, v90, v92
	v_max3_f32 v96, v96, v93, v95
	v_max3_f32 v97, v97, v94, v64
	v_max3_f32 v96, v96, v65, v67
	v_max3_f32 v97, v97, v66, v68
	; sched_group_barrier mask(0x00000008) size(1) SyncID(6)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(6)
	v_mfma_f32_32x32x16_bf16 v[48:63], v[212:215], v[160:163], v[48:63]
	v_max3_f32 v96, v96, v69, v71
	v_max3_f32 v97, v97, v70, v72
	v_max3_f32 v96, v96, v73, v75
	v_max3_f32 v97, v97, v74, v76
	v_max3_f32 v96, v96, v77, v79
	; sched_group_barrier mask(0x00000008) size(1) SyncID(6)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(6)
	v_mfma_f32_32x32x16_bf16 v[48:63], v[208:211], v[148:151], v[48:63]
	v_max3_f32 v96, v97, v78, v96
	v_mov_b32_e32 v97, v96
	s_nop 1
	v_permlane32_swap_b32_e64 v96, v97 bound_ctrl:1
	v_max3_f32 v229, v255, v96, v97
	v_sub_f32_e32 v80, v80, v229
	v_sub_f32_e32 v219, v79, v229
	v_mfma_f32_32x32x16_bf16 v[32:47], v[204:207], v[152:155], v[32:47]
	v_sub_f32_e32 v81, v81, v229
	v_sub_f32_e32 v82, v82, v229
	v_sub_f32_e32 v83, v83, v229
	v_sub_f32_e32 v84, v84, v229
	v_sub_f32_e32 v85, v85, v229
	; sched_group_barrier mask(0x00000008) size(1) SyncID(6)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(6)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(6)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(6)
	v_mfma_f32_32x32x16_bf16 v[32:47], v[200:203], v[156:159], v[32:47]
	v_sub_f32_e32 v86, v86, v229
	v_sub_f32_e32 v87, v87, v229
	v_sub_f32_e32 v88, v88, v229
	v_sub_f32_e32 v89, v89, v229
	v_sub_f32_e32 v90, v90, v229
	; sched_group_barrier mask(0x00000008) size(1) SyncID(6)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(6)
	v_mfma_f32_32x32x16_bf16 v[32:47], v[196:199], v[160:163], v[32:47]
	v_sub_f32_e32 v91, v91, v229
	v_sub_f32_e32 v92, v92, v229
	v_sub_f32_e32 v93, v93, v229
	v_sub_f32_e32 v94, v94, v229
	v_sub_f32_e32 v95, v95, v229
	; sched_group_barrier mask(0x00000008) size(1) SyncID(6)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(6)
	v_mfma_f32_32x32x16_bf16 v[32:47], v[192:195], v[148:151], v[32:47]
	v_sub_f32_e32 v104, v64, v229
	v_sub_f32_e32 v105, v65, v229
	v_sub_f32_e32 v106, v66, v229
	v_sub_f32_e32 v107, v67, v229
	v_sub_f32_e32 v108, v68, v229
	; sched_group_barrier mask(0x00000008) size(1) SyncID(6)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(6)
	v_mfma_f32_32x32x16_bf16 v[16:31], v[188:191], v[152:155], v[16:31]
	v_sub_f32_e32 v109, v69, v229
	v_sub_f32_e32 v110, v70, v229
	v_sub_f32_e32 v111, v71, v229
	v_sub_f32_e32 v212, v72, v229
	v_sub_f32_e32 v213, v73, v229
	; sched_group_barrier mask(0x00000008) size(1) SyncID(6)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(6)
	v_mfma_f32_32x32x16_bf16 v[16:31], v[184:187], v[156:159], v[16:31]
	v_sub_f32_e32 v214, v74, v229
	v_sub_f32_e32 v215, v75, v229
	v_sub_f32_e32 v216, v76, v229
	v_sub_f32_e32 v217, v77, v229
	v_sub_f32_e32 v218, v78, v229
	; sched_group_barrier mask(0x00000008) size(1) SyncID(6)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(6)
	v_mfma_f32_32x32x16_bf16 v[16:31], v[180:183], v[160:163], v[16:31]
	v_exp_f32_e32 v220, v80
	v_exp_f32_e32 v221, v81
	v_exp_f32_e32 v222, v82
	; sched_group_barrier mask(0x00000008) size(1) SyncID(6)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(6)
	v_mfma_f32_32x32x16_bf16 v[16:31], v[176:179], v[148:151], v[16:31]
	v_exp_f32_e32 v223, v83
	v_exp_f32_e32 v224, v84
	v_exp_f32_e32 v230, v85
	; sched_group_barrier mask(0x00000008) size(1) SyncID(6)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(6)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[172:175], v[152:155], v[0:15]
	v_exp_f32_e32 v231, v86
	v_exp_f32_e32 v232, v87
	v_exp_f32_e32 v233, v88
	; sched_group_barrier mask(0x00000008) size(1) SyncID(6)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(6)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[168:171], v[156:159], v[0:15]
	v_exp_f32_e32 v234, v89
	v_exp_f32_e32 v235, v90
	v_exp_f32_e32 v237, v91
	; sched_group_barrier mask(0x00000008) size(1) SyncID(6)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(6)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[164:167], v[160:163], v[0:15]
	v_exp_f32_e32 v238, v92
	v_exp_f32_e32 v239, v93
	v_exp_f32_e32 v241, v94
	; sched_group_barrier mask(0x00000008) size(1) SyncID(6)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(6)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[144:147], v[148:151], v[0:15]
	v_exp_f32_e32 v242, v95
	; sched_group_barrier mask(0x00000008) size(1) SyncID(6)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(6)
	; sched_barrier mask(0x00000000)
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	scratch_load_dword v64, off, off offset:212 ; 4-byte Folded Reload
	s_add_i32 s2, s11, s17
	s_mul_i32 s2, s2, s22
	s_add_i32 s2, s2, s31
	s_mul_i32 s2, s2, s23
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[2:3], s[2:3], 1
	s_add_u32 s24, s40, s2
	s_addc_u32 s25, s41, s3
	scratch_load_dword v68, off, off offset:208 ; 4-byte Folded Reload
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s2, v64
	scratch_load_dword v64, off, off offset:12 ; 4-byte Folded Reload
	s_mov_b32 m0, s2
	s_waitcnt vmcnt(0)
	buffer_load_dwordx4 v64, s[24:27], 0 offen lds
	scratch_load_dword v64, off, off offset:216 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s2, v64
	scratch_load_dword v64, off, off offset:116 ; 4-byte Folded Reload
	s_mov_b32 m0, s2
	s_waitcnt vmcnt(0)
	buffer_load_dwordx4 v64, s[24:27], 0 offen lds
	;;#ASMSTART
	ds_read_b128 v[64:67], v68 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[80:83], v68 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[84:87], v68 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[88:91], v68 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[92:95], v68 offset:0x2000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[96:99], v68 offset:0x2800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[100:103], v68 offset:0x3000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[144:147], v68 offset:0x3800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[148:151], v68 offset:0x4000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[152:155], v68 offset:0x4800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[156:159], v68 offset:0x5000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[160:163], v68 offset:0x5800

	;;#ASMEND
	scratch_load_dword v68, off, off offset:220 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b128 v[164:167], v68 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[168:171], v68 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[172:175], v68 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[176:179], v68 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[180:183], v68 offset:0x2000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[184:187], v68 offset:0x2800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[188:191], v68 offset:0x3000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[192:195], v68 offset:0x3800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[196:199], v68 offset:0x4000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[200:203], v68 offset:0x4800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[204:207], v68 offset:0x5000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[208:211], v68 offset:0x5800

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(4)
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	scratch_load_dwordx4 v[248:251], off, off offset:68 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[244:247], off, off offset:84 ; 16-byte Folded Reload
	v_mfma_f32_32x32x16_bf16 v[64:79], v[64:67], v[116:119], 0
	v_exp_f32_e32 v104, v104
	v_exp_f32_e32 v105, v105
	v_exp_f32_e32 v106, v106
	; sched_group_barrier mask(0x00000008) size(1) SyncID(7)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(7)
	s_waitcnt vmcnt(1)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[164:167], v[248:251], v[64:79]
	v_exp_f32_e32 v107, v107
	v_exp_f32_e32 v108, v108
	v_exp_f32_e32 v109, v109
	; sched_group_barrier mask(0x00000008) size(1) SyncID(7)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(7)
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[80:83], v[244:247], v[64:79]
	v_exp_f32_e32 v110, v110
	v_exp_f32_e32 v111, v111
	v_exp_f32_e32 v164, v212
	; sched_group_barrier mask(0x00000008) size(1) SyncID(7)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(7)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[168:171], v[128:131], v[64:79]
	v_exp_f32_e32 v165, v213
	v_exp_f32_e32 v166, v214
	v_exp_f32_e32 v167, v215
	scratch_load_dwordx4 v[212:215], off, off offset:100 ; 16-byte Folded Reload
	; sched_group_barrier mask(0x00000008) size(1) SyncID(7)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(7)
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[84:87], v[212:215], v[64:79]
	v_exp_f32_e32 v168, v216
	v_exp_f32_e32 v169, v217
	v_exp_f32_e32 v170, v218
	; sched_group_barrier mask(0x00000008) size(1) SyncID(7)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(7)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[172:175], v[132:135], v[64:79]
	scratch_load_dwordx4 v[172:175], off, off offset:52 ; 16-byte Folded Reload
	v_exp_f32_e32 v171, v219
	; sched_group_barrier mask(0x00000008) size(1) SyncID(7)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(7)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[88:91], v[136:139], v[64:79]
	v_add_f32_e32 v80, v221, v220
	v_add_f32_e32 v80, v80, v222
	v_add_f32_e32 v80, v80, v223
	v_add_f32_e32 v80, v80, v224
	v_add_f32_e32 v80, v80, v230
	; sched_group_barrier mask(0x00000008) size(1) SyncID(7)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(7)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[176:179], v[140:143], v[64:79]
	v_add_f32_e32 v80, v80, v231
	v_add_f32_e32 v80, v80, v232
	v_add_f32_e32 v80, v80, v233
	v_add_f32_e32 v80, v80, v234
	v_add_f32_e32 v80, v80, v235
	; sched_group_barrier mask(0x00000008) size(1) SyncID(7)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(7)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[92:95], v[120:123], v[64:79]
	v_add_f32_e32 v80, v80, v237
	v_add_f32_e32 v80, v80, v238
	v_add_f32_e32 v80, v80, v239
	v_add_f32_e32 v80, v80, v241
	v_add_f32_e32 v80, v80, v242
	; sched_group_barrier mask(0x00000008) size(1) SyncID(7)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(7)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[180:183], v[124:127], v[64:79]
	v_add_f32_e32 v80, v80, v104
	v_add_f32_e32 v80, v80, v105
	v_add_f32_e32 v80, v80, v106
	v_add_f32_e32 v80, v80, v107
	v_add_f32_e32 v80, v80, v108
	; sched_group_barrier mask(0x00000008) size(1) SyncID(7)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(7)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[96:99], v[112:115], v[64:79]
	v_add_f32_e32 v80, v80, v109
	v_add_f32_e32 v80, v80, v110
	v_add_f32_e32 v80, v80, v111
	v_add_f32_e32 v80, v80, v164
	v_add_f32_e32 v80, v80, v165
	; sched_group_barrier mask(0x00000008) size(1) SyncID(7)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(7)
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[184:187], v[172:175], v[64:79]
	v_add_f32_e32 v80, v80, v166
	v_add_f32_e32 v80, v80, v167
	v_add_f32_e32 v80, v80, v168
	v_add_f32_e32 v80, v80, v169
	v_add_f32_e32 v96, v80, v170
	; sched_group_barrier mask(0x00000008) size(1) SyncID(7)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(7)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[100:103], v[116:119], 0
	v_add_f32_e32 v227, v96, v171
	v_mov_b32_e32 v228, v227
	s_nop 1
	v_permlane32_swap_b32_e64 v227, v228 bound_ctrl:1
	; sched_group_barrier mask(0x00000008) size(1) SyncID(7)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(7)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[188:191], v[248:251], v[80:95]
	; sched_group_barrier mask(0x00000008) size(1) SyncID(7)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(7)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[144:147], v[244:247], v[80:95]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v144, v220, v221
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v145, v222, v223
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v146, v224, v230
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v147, v231, v232
	;;#ASMEND
	; sched_group_barrier mask(0x00000008) size(1) SyncID(7)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(7)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[192:195], v[128:131], v[80:95]
	; sched_group_barrier mask(0x00000008) size(1) SyncID(7)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(7)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[148:151], v[212:215], v[80:95]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v148, v233, v234
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v149, v235, v237
	;;#ASMEND
	scratch_load_dword v96, off, off offset:48 ; 4-byte Folded Reload
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v150, v238, v239
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v151, v241, v242
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[80:95], v[196:199], v[132:135], v[80:95]
	v_mfma_f32_32x32x16_bf16 v[80:95], v[152:155], v[136:139], v[80:95]
	v_mfma_f32_32x32x16_bf16 v[80:95], v[200:203], v[140:143], v[80:95]
	v_mfma_f32_32x32x16_bf16 v[80:95], v[156:159], v[120:123], v[80:95]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v156, v104, v105
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v157, v106, v107
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v158, v108, v109
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v159, v110, v111
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v152, v164, v165
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v153, v166, v167
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v154, v168, v169
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[80:95], v[204:207], v[124:127], v[80:95]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v155, v170, v171
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[80:95], v[160:163], v[112:115], v[80:95]
	v_mfma_f32_32x32x16_bf16 v[80:95], v[208:211], v[172:175], v[80:95]
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	scratch_load_dword v97, off, off offset:152 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b16 v[220:221], v97 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[204:205], v97 offset:0x200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[188:189], v97 offset:0x400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[172:173], v97 offset:0x600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[216:217], v97 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[200:201], v97 offset:0x1200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[184:185], v97 offset:0x1400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[168:169], v97 offset:0x1600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[212:213], v97 offset:0x2000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[196:197], v97 offset:0x2200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[180:181], v97 offset:0x2400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[164:165], v97 offset:0x2600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[208:209], v97 offset:0x3000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[192:193], v97 offset:0x3200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[176:177], v97 offset:0x3400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[160:161], v97 offset:0x3600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[222:223], v97 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[206:207], v97 offset:0xa00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[190:191], v97 offset:0xc00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[174:175], v97 offset:0xe00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[218:219], v97 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[202:203], v97 offset:0x1a00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[186:187], v97 offset:0x1c00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[170:171], v97 offset:0x1e00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[214:215], v97 offset:0x2800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[198:199], v97 offset:0x2a00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[182:183], v97 offset:0x2c00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[166:167], v97 offset:0x2e00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[210:211], v97 offset:0x3800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[194:195], v97 offset:0x3a00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[178:179], v97 offset:0x3c00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[162:163], v97 offset:0x3e00

	;;#ASMEND
	scratch_load_dword v97, off, off offset:312 ; 4-byte Folded Reload
	v_cmp_gt_i32_e32 vcc, s15, v96
	s_waitcnt vmcnt(0)
	v_sub_u32_e32 v230, v97, v240
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_38
; %bb.37:
	v_subrev_u32_e32 v96, s17, v96
	v_add_u32_e32 v96, v230, v96
	v_mov_b32_e32 v97, 0xff800000
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v96, 0
	v_cmp_lt_i32_e64 s[24:25], v96, 1
	v_cndmask_b32_e64 v64, v64, v97, s[20:21]
	v_cndmask_b32_e64 v65, v65, v97, s[24:25]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v96, 2
	v_cmp_lt_i32_e64 s[24:25], v96, 3
	v_cndmask_b32_e64 v66, v66, v97, s[20:21]
	v_cndmask_b32_e64 v67, v67, v97, s[24:25]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v96, 8
	v_cmp_lt_i32_e64 s[24:25], v96, 9
	v_cndmask_b32_e64 v68, v68, v97, s[20:21]
	v_cndmask_b32_e64 v69, v69, v97, s[24:25]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v96, 10
	v_cmp_lt_i32_e64 s[24:25], v96, 11
	v_cndmask_b32_e64 v70, v70, v97, s[20:21]
	v_cndmask_b32_e64 v71, v71, v97, s[24:25]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v96, 16
	v_cmp_lt_i32_e64 s[24:25], v96, 17
	v_cndmask_b32_e64 v72, v72, v97, s[20:21]
	v_cndmask_b32_e64 v73, v73, v97, s[24:25]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v96, 18
	v_cmp_lt_i32_e64 s[24:25], v96, 19
	v_cndmask_b32_e64 v74, v74, v97, s[20:21]
	v_cndmask_b32_e64 v75, v75, v97, s[24:25]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v96, 24
	v_cmp_lt_i32_e64 s[24:25], v96, 25
	v_cndmask_b32_e64 v76, v76, v97, s[20:21]
	v_cndmask_b32_e64 v77, v77, v97, s[24:25]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v96, 26
	v_cmp_lt_i32_e64 s[24:25], v96, 27
	v_cndmask_b32_e64 v78, v78, v97, s[20:21]
	v_cndmask_b32_e64 v79, v79, v97, s[24:25]
	
	;;#ASMEND
	v_subrev_u32_e32 v96, 32, v96
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v96, 0
	v_cmp_lt_i32_e64 s[24:25], v96, 1
	v_cndmask_b32_e64 v80, v80, v97, s[20:21]
	v_cndmask_b32_e64 v81, v81, v97, s[24:25]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v96, 2
	v_cmp_lt_i32_e64 s[24:25], v96, 3
	v_cndmask_b32_e64 v82, v82, v97, s[20:21]
	v_cndmask_b32_e64 v83, v83, v97, s[24:25]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v96, 8
	v_cmp_lt_i32_e64 s[24:25], v96, 9
	v_cndmask_b32_e64 v84, v84, v97, s[20:21]
	v_cndmask_b32_e64 v85, v85, v97, s[24:25]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v96, 10
	v_cmp_lt_i32_e64 s[24:25], v96, 11
	v_cndmask_b32_e64 v86, v86, v97, s[20:21]
	v_cndmask_b32_e64 v87, v87, v97, s[24:25]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v96, 16
	v_cmp_lt_i32_e64 s[24:25], v96, 17
	v_cndmask_b32_e64 v88, v88, v97, s[20:21]
	v_cndmask_b32_e64 v89, v89, v97, s[24:25]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v96, 18
	v_cmp_lt_i32_e64 s[24:25], v96, 19
	v_cndmask_b32_e64 v90, v90, v97, s[20:21]
	v_cndmask_b32_e64 v91, v91, v97, s[24:25]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v96, 24
	v_cmp_lt_i32_e64 s[24:25], v96, 25
	v_cndmask_b32_e64 v92, v92, v97, s[20:21]
	v_cndmask_b32_e64 v93, v93, v97, s[24:25]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v96, 26
	v_cmp_lt_i32_e64 s[24:25], v96, 27
	v_cndmask_b32_e64 v94, v94, v97, s[20:21]
	v_cndmask_b32_e64 v95, v95, v97, s[24:25]
	
	;;#ASMEND
.LBB0_38:
	s_or_b64 exec, exec, s[2:3]
	v_sub_f32_e32 v96, v255, v229
	v_exp_f32_e32 v224, v96
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(2)
	;;#ASMEND
	s_nop 0
	v_pk_mul_f32 v[62:63], v[224:225], v[62:63] op_sel_hi:[0,1]
	v_pk_mul_f32 v[60:61], v[224:225], v[60:61] op_sel_hi:[0,1]
	v_pk_mul_f32 v[58:59], v[224:225], v[58:59] op_sel_hi:[0,1]
	v_pk_mul_f32 v[56:57], v[224:225], v[56:57] op_sel_hi:[0,1]
	v_pk_mul_f32 v[54:55], v[224:225], v[54:55] op_sel_hi:[0,1]
	v_pk_mul_f32 v[52:53], v[224:225], v[52:53] op_sel_hi:[0,1]
	v_pk_mul_f32 v[50:51], v[224:225], v[50:51] op_sel_hi:[0,1]
	v_pk_mul_f32 v[48:49], v[224:225], v[48:49] op_sel_hi:[0,1]
	v_pk_mul_f32 v[46:47], v[224:225], v[46:47] op_sel_hi:[0,1]
	v_pk_mul_f32 v[44:45], v[224:225], v[44:45] op_sel_hi:[0,1]
	v_pk_mul_f32 v[42:43], v[224:225], v[42:43] op_sel_hi:[0,1]
	v_pk_mul_f32 v[40:41], v[224:225], v[40:41] op_sel_hi:[0,1]
	v_pk_mul_f32 v[38:39], v[224:225], v[38:39] op_sel_hi:[0,1]
	v_pk_mul_f32 v[36:37], v[224:225], v[36:37] op_sel_hi:[0,1]
	v_pk_mul_f32 v[34:35], v[224:225], v[34:35] op_sel_hi:[0,1]
	v_pk_mul_f32 v[32:33], v[224:225], v[32:33] op_sel_hi:[0,1]
	v_pk_mul_f32 v[30:31], v[224:225], v[30:31] op_sel_hi:[0,1]
	v_pk_mul_f32 v[28:29], v[224:225], v[28:29] op_sel_hi:[0,1]
	v_pk_mul_f32 v[26:27], v[224:225], v[26:27] op_sel_hi:[0,1]
	v_pk_mul_f32 v[24:25], v[224:225], v[24:25] op_sel_hi:[0,1]
	v_pk_mul_f32 v[22:23], v[224:225], v[22:23] op_sel_hi:[0,1]
	v_pk_mul_f32 v[20:21], v[224:225], v[20:21] op_sel_hi:[0,1]
	v_pk_mul_f32 v[18:19], v[224:225], v[18:19] op_sel_hi:[0,1]
	v_pk_mul_f32 v[16:17], v[224:225], v[16:17] op_sel_hi:[0,1]
	v_pk_mul_f32 v[14:15], v[224:225], v[14:15] op_sel_hi:[0,1]
	v_pk_mul_f32 v[12:13], v[224:225], v[12:13] op_sel_hi:[0,1]
	v_pk_mul_f32 v[10:11], v[224:225], v[10:11] op_sel_hi:[0,1]
	v_pk_mul_f32 v[8:9], v[224:225], v[8:9] op_sel_hi:[0,1]
	v_pk_mul_f32 v[6:7], v[224:225], v[6:7] op_sel_hi:[0,1]
	v_pk_mul_f32 v[4:5], v[224:225], v[4:5] op_sel_hi:[0,1]
	v_pk_mul_f32 v[2:3], v[224:225], v[2:3] op_sel_hi:[0,1]
	v_pk_mul_f32 v[0:1], v[224:225], v[0:1] op_sel_hi:[0,1]
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_f32_32x32x16_bf16 v[48:63], v[220:223], v[144:147], v[48:63]
	v_max_f32_e32 v96, v65, v67
	v_max3_f32 v97, v64, v66, v68
	v_max3_f32 v96, v96, v69, v71
	v_max3_f32 v97, v97, v70, v72
	v_max3_f32 v96, v96, v73, v75
	; sched_group_barrier mask(0x00000008) size(1) SyncID(8)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(8)
	v_mfma_f32_32x32x16_bf16 v[48:63], v[216:219], v[148:151], v[48:63]
	v_max3_f32 v97, v97, v74, v76
	v_max3_f32 v96, v96, v77, v79
	v_max3_f32 v97, v97, v78, v80
	v_max3_f32 v96, v96, v81, v83
	v_max3_f32 v97, v97, v82, v84
	; sched_group_barrier mask(0x00000008) size(1) SyncID(8)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(8)
	v_mfma_f32_32x32x16_bf16 v[48:63], v[212:215], v[156:159], v[48:63]
	v_max3_f32 v96, v96, v85, v87
	v_max3_f32 v97, v97, v86, v88
	v_max3_f32 v96, v96, v89, v91
	v_max3_f32 v97, v97, v90, v92
	v_max3_f32 v96, v96, v93, v95
	; sched_group_barrier mask(0x00000008) size(1) SyncID(8)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(8)
	v_mfma_f32_32x32x16_bf16 v[48:63], v[208:211], v[152:155], v[48:63]
	v_max3_f32 v96, v97, v94, v96
	v_mov_b32_e32 v97, v96
	s_nop 1
	v_permlane32_swap_b32_e64 v96, v97 bound_ctrl:1
	v_max3_f32 v208, v229, v96, v97
	v_sub_f32_e32 v64, v64, v208
	v_sub_f32_e32 v220, v95, v208
	v_mfma_f32_32x32x16_bf16 v[32:47], v[204:207], v[144:147], v[32:47]
	v_sub_f32_e32 v65, v65, v208
	v_sub_f32_e32 v66, v66, v208
	v_sub_f32_e32 v67, v67, v208
	v_sub_f32_e32 v68, v68, v208
	v_sub_f32_e32 v69, v69, v208
	; sched_group_barrier mask(0x00000008) size(1) SyncID(8)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(8)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(8)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(8)
	v_mfma_f32_32x32x16_bf16 v[32:47], v[200:203], v[148:151], v[32:47]
	v_sub_f32_e32 v70, v70, v208
	v_sub_f32_e32 v71, v71, v208
	v_sub_f32_e32 v72, v72, v208
	v_sub_f32_e32 v73, v73, v208
	v_sub_f32_e32 v74, v74, v208
	; sched_group_barrier mask(0x00000008) size(1) SyncID(8)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(8)
	v_mfma_f32_32x32x16_bf16 v[32:47], v[196:199], v[156:159], v[32:47]
	v_sub_f32_e32 v75, v75, v208
	v_sub_f32_e32 v76, v76, v208
	v_sub_f32_e32 v77, v77, v208
	v_sub_f32_e32 v78, v78, v208
	v_sub_f32_e32 v79, v79, v208
	; sched_group_barrier mask(0x00000008) size(1) SyncID(8)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(8)
	v_mfma_f32_32x32x16_bf16 v[32:47], v[192:195], v[152:155], v[32:47]
	v_sub_f32_e32 v104, v80, v208
	v_sub_f32_e32 v105, v81, v208
	v_sub_f32_e32 v106, v82, v208
	v_sub_f32_e32 v107, v83, v208
	v_sub_f32_e32 v108, v84, v208
	; sched_group_barrier mask(0x00000008) size(1) SyncID(8)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(8)
	v_mfma_f32_32x32x16_bf16 v[16:31], v[188:191], v[144:147], v[16:31]
	v_sub_f32_e32 v109, v85, v208
	v_sub_f32_e32 v110, v86, v208
	v_sub_f32_e32 v111, v87, v208
	v_sub_f32_e32 v209, v88, v208
	v_sub_f32_e32 v214, v89, v208
	; sched_group_barrier mask(0x00000008) size(1) SyncID(8)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(8)
	v_mfma_f32_32x32x16_bf16 v[16:31], v[184:187], v[148:151], v[16:31]
	v_sub_f32_e32 v215, v90, v208
	v_sub_f32_e32 v216, v91, v208
	v_sub_f32_e32 v217, v92, v208
	v_sub_f32_e32 v218, v93, v208
	v_sub_f32_e32 v219, v94, v208
	; sched_group_barrier mask(0x00000008) size(1) SyncID(8)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(8)
	v_mfma_f32_32x32x16_bf16 v[16:31], v[180:183], v[156:159], v[16:31]
	v_exp_f32_e32 v221, v64
	v_exp_f32_e32 v222, v65
	v_exp_f32_e32 v223, v66
	; sched_group_barrier mask(0x00000008) size(1) SyncID(8)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(8)
	v_mfma_f32_32x32x16_bf16 v[16:31], v[176:179], v[152:155], v[16:31]
	v_exp_f32_e32 v231, v67
	v_exp_f32_e32 v232, v68
	v_exp_f32_e32 v233, v69
	; sched_group_barrier mask(0x00000008) size(1) SyncID(8)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(8)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[172:175], v[144:147], v[0:15]
	v_exp_f32_e32 v234, v70
	v_exp_f32_e32 v235, v71
	v_exp_f32_e32 v237, v72
	; sched_group_barrier mask(0x00000008) size(1) SyncID(8)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(8)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[168:171], v[148:151], v[0:15]
	v_exp_f32_e32 v238, v73
	v_exp_f32_e32 v239, v74
	v_exp_f32_e32 v241, v75
	; sched_group_barrier mask(0x00000008) size(1) SyncID(8)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(8)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[164:167], v[156:159], v[0:15]
	v_exp_f32_e32 v242, v76
	v_exp_f32_e32 v246, v77
	v_exp_f32_e32 v247, v78
	; sched_group_barrier mask(0x00000008) size(1) SyncID(8)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(8)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[160:163], v[152:155], v[0:15]
	v_exp_f32_e32 v248, v79
	; sched_group_barrier mask(0x00000008) size(1) SyncID(8)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(8)
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	scratch_load_dword v64, off, off offset:200 ; 4-byte Folded Reload
	s_add_i32 s2, s11, s15
	s_mul_i32 s2, s2, s22
	s_add_i32 s2, s2, s31
	s_mul_i32 s2, s2, s23
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[2:3], s[2:3], 1
	s_add_u32 s24, s40, s2
	s_addc_u32 s25, s41, s3
	scratch_load_dword v68, off, off offset:192 ; 4-byte Folded Reload
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s2, v64
	scratch_load_dword v64, off, off offset:12 ; 4-byte Folded Reload
	s_mov_b32 m0, s2
	s_waitcnt vmcnt(0)
	buffer_load_dwordx4 v64, s[24:27], 0 offen lds
	scratch_load_dword v64, off, off offset:204 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s2, v64
	scratch_load_dword v64, off, off offset:116 ; 4-byte Folded Reload
	s_mov_b32 m0, s2
	s_waitcnt vmcnt(0)
	buffer_load_dwordx4 v64, s[24:27], 0 offen lds
	;;#ASMSTART
	ds_read_b128 v[64:67], v68 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[80:83], v68 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[84:87], v68 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[88:91], v68 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[92:95], v68 offset:0x2000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[96:99], v68 offset:0x2800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[100:103], v68 offset:0x3000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[144:147], v68 offset:0x3800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[148:151], v68 offset:0x4000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[152:155], v68 offset:0x4800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[156:159], v68 offset:0x5000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[160:163], v68 offset:0x5800

	;;#ASMEND
	scratch_load_dword v68, off, off offset:196 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b128 v[164:167], v68 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[168:171], v68 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[172:175], v68 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[176:179], v68 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[180:183], v68 offset:0x2000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[184:187], v68 offset:0x2800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[188:191], v68 offset:0x3000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[192:195], v68 offset:0x3800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[196:199], v68 offset:0x4000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[200:203], v68 offset:0x4800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[204:207], v68 offset:0x5000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b128 v[210:213], v68 offset:0x5800

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(2)
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	scratch_load_dwordx4 v[112:115], off, off offset:68 ; 16-byte Folded Reload
	v_mfma_f32_32x32x16_bf16 v[64:79], v[64:67], v[116:119], 0
	v_exp_f32_e32 v244, v104
	v_exp_f32_e32 v245, v105
	v_exp_f32_e32 v249, v106
	scratch_load_dwordx4 v[120:123], off, off offset:100 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[124:127], off, off offset:120 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[252:255], off, off offset:156 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[136:139], off, off offset:172 ; 16-byte Folded Reload
	; sched_group_barrier mask(0x00000008) size(1) SyncID(9)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(9)
	s_waitcnt vmcnt(4)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[164:167], v[112:115], v[64:79]
	v_exp_f32_e32 v164, v107
	scratch_load_dwordx4 v[104:107], off, off offset:84 ; 16-byte Folded Reload
	v_exp_f32_e32 v108, v108
	v_exp_f32_e32 v109, v109
	; sched_group_barrier mask(0x00000008) size(1) SyncID(9)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(9)
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[80:83], v[104:107], v[64:79]
	v_exp_f32_e32 v110, v110
	v_exp_f32_e32 v111, v111
	v_exp_f32_e32 v165, v209
	; sched_group_barrier mask(0x00000008) size(1) SyncID(9)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(9)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[168:171], v[128:131], v[64:79]
	v_exp_f32_e32 v166, v214
	v_exp_f32_e32 v167, v215
	v_exp_f32_e32 v168, v216
	; sched_group_barrier mask(0x00000008) size(1) SyncID(9)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(9)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[84:87], v[120:123], v[64:79]
	v_exp_f32_e32 v169, v217
	v_exp_f32_e32 v170, v218
	v_exp_f32_e32 v171, v219
	scratch_load_dwordx4 v[214:217], off, off offset:52 ; 16-byte Folded Reload
	; sched_group_barrier mask(0x00000008) size(1) SyncID(9)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(9)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[172:175], v[132:135], v[64:79]
	v_exp_f32_e32 v172, v220
	; sched_group_barrier mask(0x00000008) size(1) SyncID(9)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(9)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[88:91], v[124:127], v[64:79]
	v_add_f32_e32 v80, v222, v221
	v_add_f32_e32 v80, v80, v223
	v_add_f32_e32 v80, v80, v231
	v_add_f32_e32 v80, v80, v232
	v_add_f32_e32 v80, v80, v233
	; sched_group_barrier mask(0x00000008) size(1) SyncID(9)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(9)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[176:179], v[140:143], v[64:79]
	v_add_f32_e32 v80, v80, v234
	v_add_f32_e32 v80, v80, v235
	v_add_f32_e32 v80, v80, v237
	v_add_f32_e32 v80, v80, v238
	v_add_f32_e32 v80, v80, v239
	; sched_group_barrier mask(0x00000008) size(1) SyncID(9)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(9)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[92:95], v[252:255], v[64:79]
	v_add_f32_e32 v80, v80, v241
	v_add_f32_e32 v80, v80, v242
	v_add_f32_e32 v80, v80, v246
	v_add_f32_e32 v80, v80, v247
	v_add_f32_e32 v80, v80, v248
	; sched_group_barrier mask(0x00000008) size(1) SyncID(9)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(9)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[180:183], v[136:139], v[64:79]
	scratch_load_dwordx4 v[180:183], off, off offset:136 ; 16-byte Folded Reload
	v_add_f32_e32 v80, v80, v244
	v_add_f32_e32 v80, v80, v245
	v_add_f32_e32 v80, v80, v249
	v_add_f32_e32 v80, v80, v164
	v_add_f32_e32 v80, v80, v108
	; sched_group_barrier mask(0x00000008) size(1) SyncID(9)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(9)
	s_waitcnt vmcnt(0)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[96:99], v[180:183], v[64:79]
	v_add_f32_e32 v80, v80, v109
	v_add_f32_e32 v80, v80, v110
	v_add_f32_e32 v80, v80, v111
	v_add_f32_e32 v80, v80, v165
	v_add_f32_e32 v80, v80, v166
	; sched_group_barrier mask(0x00000008) size(1) SyncID(9)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(9)
	v_mfma_f32_32x32x16_bf16 v[64:79], v[184:187], v[214:217], v[64:79]
	v_add_f32_e32 v80, v80, v167
	v_add_f32_e32 v80, v80, v168
	v_add_f32_e32 v80, v80, v169
	v_add_f32_e32 v80, v80, v170
	v_add_f32_e32 v96, v80, v171
	; sched_group_barrier mask(0x00000008) size(1) SyncID(9)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(9)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[100:103], v[116:119], 0
	v_add_f32_e32 v177, v96, v172
	v_mov_b32_e32 v178, v177
	s_nop 1
	v_permlane32_swap_b32_e64 v177, v178 bound_ctrl:1
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v100, v221, v222
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v101, v223, v231
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v102, v232, v233
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[80:95], v[188:191], v[112:115], v[80:95]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v103, v234, v235
	;;#ASMEND
	; sched_group_barrier mask(0x00000008) size(1) SyncID(9)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(9)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(9)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(9)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[144:147], v[104:107], v[80:95]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v104, v237, v238
	;;#ASMEND
	scratch_load_dword v184, off, off offset:48 ; 4-byte Folded Reload
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v105, v239, v241
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v106, v242, v246
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v107, v247, v248
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v112, v244, v245
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v113, v249, v164
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[80:95], v[192:195], v[128:131], v[80:95]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v114, v108, v109
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v115, v110, v111
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v108, v165, v166
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v109, v167, v168
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v110, v169, v170
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v111, v171, v172
	;;#ASMEND
	; sched_group_barrier mask(0x00000008) size(1) SyncID(9)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(9)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(9)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(9)
	v_mfma_f32_32x32x16_bf16 v[80:95], v[148:151], v[120:123], v[80:95]
	v_mfma_f32_32x32x16_bf16 v[80:95], v[196:199], v[132:135], v[80:95]
	v_mfma_f32_32x32x16_bf16 v[80:95], v[152:155], v[124:127], v[80:95]
	v_mfma_f32_32x32x16_bf16 v[80:95], v[200:203], v[140:143], v[80:95]
	v_mfma_f32_32x32x16_bf16 v[80:95], v[156:159], v[252:255], v[80:95]
	v_mfma_f32_32x32x16_bf16 v[80:95], v[204:207], v[136:139], v[80:95]
	v_mfma_f32_32x32x16_bf16 v[80:95], v[160:163], v[180:183], v[80:95]
	v_mfma_f32_32x32x16_bf16 v[80:95], v[210:213], v[214:217], v[80:95]
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	scratch_load_dword v98, off, off offset:188 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b16 v[172:173], v98 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[156:157], v98 offset:0x200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[140:141], v98 offset:0x400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[124:125], v98 offset:0x600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[168:169], v98 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[152:153], v98 offset:0x1200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[136:137], v98 offset:0x1400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[120:121], v98 offset:0x1600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[164:165], v98 offset:0x2000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[148:149], v98 offset:0x2200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[132:133], v98 offset:0x2400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[116:117], v98 offset:0x2600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[160:161], v98 offset:0x3000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[144:145], v98 offset:0x3200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[128:129], v98 offset:0x3400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[96:97], v98 offset:0x3600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[174:175], v98 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[158:159], v98 offset:0xa00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[142:143], v98 offset:0xc00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[126:127], v98 offset:0xe00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[170:171], v98 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[154:155], v98 offset:0x1a00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[138:139], v98 offset:0x1c00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[122:123], v98 offset:0x1e00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[166:167], v98 offset:0x2800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[150:151], v98 offset:0x2a00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[134:135], v98 offset:0x2c00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[118:119], v98 offset:0x2e00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[162:163], v98 offset:0x3800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[146:147], v98 offset:0x3a00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[130:131], v98 offset:0x3c00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[98:99], v98 offset:0x3e00

	;;#ASMEND
	v_cmp_gt_i32_e32 vcc, s13, v184
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_40
; %bb.39:
	v_subrev_u32_e32 v176, s15, v184
	v_add_u32_e32 v176, v230, v176
	v_mov_b32_e32 v179, 0xff800000
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v176, 0
	v_cmp_lt_i32_e64 s[22:23], v176, 1
	v_cndmask_b32_e64 v64, v64, v179, s[20:21]
	v_cndmask_b32_e64 v65, v65, v179, s[22:23]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v176, 2
	v_cmp_lt_i32_e64 s[22:23], v176, 3
	v_cndmask_b32_e64 v66, v66, v179, s[20:21]
	v_cndmask_b32_e64 v67, v67, v179, s[22:23]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v176, 8
	v_cmp_lt_i32_e64 s[22:23], v176, 9
	v_cndmask_b32_e64 v68, v68, v179, s[20:21]
	v_cndmask_b32_e64 v69, v69, v179, s[22:23]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v176, 10
	v_cmp_lt_i32_e64 s[22:23], v176, 11
	v_cndmask_b32_e64 v70, v70, v179, s[20:21]
	v_cndmask_b32_e64 v71, v71, v179, s[22:23]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v176, 16
	v_cmp_lt_i32_e64 s[22:23], v176, 17
	v_cndmask_b32_e64 v72, v72, v179, s[20:21]
	v_cndmask_b32_e64 v73, v73, v179, s[22:23]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v176, 18
	v_cmp_lt_i32_e64 s[22:23], v176, 19
	v_cndmask_b32_e64 v74, v74, v179, s[20:21]
	v_cndmask_b32_e64 v75, v75, v179, s[22:23]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v176, 24
	v_cmp_lt_i32_e64 s[22:23], v176, 25
	v_cndmask_b32_e64 v76, v76, v179, s[20:21]
	v_cndmask_b32_e64 v77, v77, v179, s[22:23]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v176, 26
	v_cmp_lt_i32_e64 s[22:23], v176, 27
	v_cndmask_b32_e64 v78, v78, v179, s[20:21]
	v_cndmask_b32_e64 v79, v79, v179, s[22:23]
	
	;;#ASMEND
	v_subrev_u32_e32 v176, 32, v176
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v176, 0
	v_cmp_lt_i32_e64 s[22:23], v176, 1
	v_cndmask_b32_e64 v80, v80, v179, s[20:21]
	v_cndmask_b32_e64 v81, v81, v179, s[22:23]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v176, 2
	v_cmp_lt_i32_e64 s[22:23], v176, 3
	v_cndmask_b32_e64 v82, v82, v179, s[20:21]
	v_cndmask_b32_e64 v83, v83, v179, s[22:23]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v176, 8
	v_cmp_lt_i32_e64 s[22:23], v176, 9
	v_cndmask_b32_e64 v84, v84, v179, s[20:21]
	v_cndmask_b32_e64 v85, v85, v179, s[22:23]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v176, 10
	v_cmp_lt_i32_e64 s[22:23], v176, 11
	v_cndmask_b32_e64 v86, v86, v179, s[20:21]
	v_cndmask_b32_e64 v87, v87, v179, s[22:23]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v176, 16
	v_cmp_lt_i32_e64 s[22:23], v176, 17
	v_cndmask_b32_e64 v88, v88, v179, s[20:21]
	v_cndmask_b32_e64 v89, v89, v179, s[22:23]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v176, 18
	v_cmp_lt_i32_e64 s[22:23], v176, 19
	v_cndmask_b32_e64 v90, v90, v179, s[20:21]
	v_cndmask_b32_e64 v91, v91, v179, s[22:23]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v176, 24
	v_cmp_lt_i32_e64 s[22:23], v176, 25
	v_cndmask_b32_e64 v92, v92, v179, s[20:21]
	v_cndmask_b32_e64 v93, v93, v179, s[22:23]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v176, 26
	v_cmp_lt_i32_e64 s[22:23], v176, 27
	v_cndmask_b32_e64 v94, v94, v179, s[20:21]
	v_cndmask_b32_e64 v95, v95, v179, s[22:23]
	
	;;#ASMEND
.LBB0_40:
	s_or_b64 exec, exec, s[2:3]
	v_sub_f32_e32 v176, v229, v208
	v_exp_f32_e32 v176, v176
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_nop 0
	v_pk_mul_f32 v[62:63], v[176:177], v[62:63] op_sel_hi:[0,1]
	v_pk_mul_f32 v[60:61], v[176:177], v[60:61] op_sel_hi:[0,1]
	v_pk_mul_f32 v[58:59], v[176:177], v[58:59] op_sel_hi:[0,1]
	v_pk_mul_f32 v[56:57], v[176:177], v[56:57] op_sel_hi:[0,1]
	v_pk_mul_f32 v[54:55], v[176:177], v[54:55] op_sel_hi:[0,1]
	v_pk_mul_f32 v[52:53], v[176:177], v[52:53] op_sel_hi:[0,1]
	v_pk_mul_f32 v[50:51], v[176:177], v[50:51] op_sel_hi:[0,1]
	v_pk_mul_f32 v[48:49], v[176:177], v[48:49] op_sel_hi:[0,1]
	v_pk_mul_f32 v[46:47], v[176:177], v[46:47] op_sel_hi:[0,1]
	v_pk_mul_f32 v[44:45], v[176:177], v[44:45] op_sel_hi:[0,1]
	v_pk_mul_f32 v[42:43], v[176:177], v[42:43] op_sel_hi:[0,1]
	v_pk_mul_f32 v[40:41], v[176:177], v[40:41] op_sel_hi:[0,1]
	v_pk_mul_f32 v[38:39], v[176:177], v[38:39] op_sel_hi:[0,1]
	v_pk_mul_f32 v[36:37], v[176:177], v[36:37] op_sel_hi:[0,1]
	v_pk_mul_f32 v[34:35], v[176:177], v[34:35] op_sel_hi:[0,1]
	v_pk_mul_f32 v[32:33], v[176:177], v[32:33] op_sel_hi:[0,1]
	v_pk_mul_f32 v[30:31], v[176:177], v[30:31] op_sel_hi:[0,1]
	v_pk_mul_f32 v[28:29], v[176:177], v[28:29] op_sel_hi:[0,1]
	v_pk_mul_f32 v[26:27], v[176:177], v[26:27] op_sel_hi:[0,1]
	v_pk_mul_f32 v[24:25], v[176:177], v[24:25] op_sel_hi:[0,1]
	v_pk_mul_f32 v[22:23], v[176:177], v[22:23] op_sel_hi:[0,1]
	v_pk_mul_f32 v[20:21], v[176:177], v[20:21] op_sel_hi:[0,1]
	v_pk_mul_f32 v[18:19], v[176:177], v[18:19] op_sel_hi:[0,1]
	v_pk_mul_f32 v[16:17], v[176:177], v[16:17] op_sel_hi:[0,1]
	v_pk_mul_f32 v[14:15], v[176:177], v[14:15] op_sel_hi:[0,1]
	v_pk_mul_f32 v[12:13], v[176:177], v[12:13] op_sel_hi:[0,1]
	v_pk_mul_f32 v[10:11], v[176:177], v[10:11] op_sel_hi:[0,1]
	v_pk_mul_f32 v[8:9], v[176:177], v[8:9] op_sel_hi:[0,1]
	v_pk_mul_f32 v[6:7], v[176:177], v[6:7] op_sel_hi:[0,1]
	v_pk_mul_f32 v[4:5], v[176:177], v[4:5] op_sel_hi:[0,1]
	v_pk_mul_f32 v[2:3], v[176:177], v[2:3] op_sel_hi:[0,1]
	v_pk_mul_f32 v[0:1], v[176:177], v[0:1] op_sel_hi:[0,1]
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	v_mfma_f32_32x32x16_bf16 v[48:63], v[172:175], v[100:103], v[48:63]
	v_max_f32_e32 v172, v65, v67
	v_max3_f32 v173, v64, v66, v68
	v_max3_f32 v172, v172, v69, v71
	v_max3_f32 v173, v173, v70, v72
	v_max3_f32 v172, v172, v73, v75
	; sched_group_barrier mask(0x00000008) size(1) SyncID(10)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(10)
	v_mfma_f32_32x32x16_bf16 v[48:63], v[168:171], v[104:107], v[48:63]
	v_max3_f32 v168, v173, v74, v76
	v_max3_f32 v169, v172, v77, v79
	v_max3_f32 v168, v168, v78, v80
	v_max3_f32 v169, v169, v81, v83
	v_max3_f32 v168, v168, v82, v84
	; sched_group_barrier mask(0x00000008) size(1) SyncID(10)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(10)
	v_mfma_f32_32x32x16_bf16 v[48:63], v[164:167], v[112:115], v[48:63]
	v_max3_f32 v164, v169, v85, v87
	v_max3_f32 v165, v168, v86, v88
	v_max3_f32 v164, v164, v89, v91
	v_max3_f32 v165, v165, v90, v92
	v_max3_f32 v164, v164, v93, v95
	; sched_group_barrier mask(0x00000008) size(1) SyncID(10)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(10)
	v_mfma_f32_32x32x16_bf16 v[48:63], v[160:163], v[108:111], v[48:63]
	v_max3_f32 v160, v165, v94, v164
	v_mov_b32_e32 v161, v160
	s_nop 1
	v_permlane32_swap_b32_e64 v160, v161 bound_ctrl:1
	v_max3_f32 v160, v208, v160, v161
	v_sub_f32_e32 v161, v208, v160
	v_sub_f32_e32 v94, v94, v160
	v_mfma_f32_32x32x16_bf16 v[32:47], v[156:159], v[100:103], v[32:47]
	v_sub_f32_e32 v64, v64, v160
	v_sub_f32_e32 v65, v65, v160
	v_sub_f32_e32 v66, v66, v160
	v_sub_f32_e32 v67, v67, v160
	v_sub_f32_e32 v68, v68, v160
	v_sub_f32_e32 v95, v95, v160
	; sched_group_barrier mask(0x00000008) size(1) SyncID(10)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(10)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(10)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(10)
	v_mfma_f32_32x32x16_bf16 v[32:47], v[152:155], v[104:107], v[32:47]
	v_sub_f32_e32 v69, v69, v160
	v_sub_f32_e32 v70, v70, v160
	v_sub_f32_e32 v71, v71, v160
	v_sub_f32_e32 v72, v72, v160
	v_sub_f32_e32 v73, v73, v160
	; sched_group_barrier mask(0x00000008) size(1) SyncID(10)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(10)
	v_mfma_f32_32x32x16_bf16 v[32:47], v[148:151], v[112:115], v[32:47]
	v_sub_f32_e32 v74, v74, v160
	v_sub_f32_e32 v75, v75, v160
	v_sub_f32_e32 v76, v76, v160
	v_sub_f32_e32 v77, v77, v160
	v_sub_f32_e32 v78, v78, v160
	; sched_group_barrier mask(0x00000008) size(1) SyncID(10)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(10)
	v_mfma_f32_32x32x16_bf16 v[32:47], v[144:147], v[108:111], v[32:47]
	v_sub_f32_e32 v79, v79, v160
	v_sub_f32_e32 v144, v80, v160
	v_sub_f32_e32 v81, v81, v160
	v_sub_f32_e32 v82, v82, v160
	v_sub_f32_e32 v83, v83, v160
	; sched_group_barrier mask(0x00000008) size(1) SyncID(10)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(10)
	v_mfma_f32_32x32x16_bf16 v[16:31], v[140:143], v[100:103], v[16:31]
	v_sub_f32_e32 v84, v84, v160
	v_sub_f32_e32 v85, v85, v160
	v_sub_f32_e32 v86, v86, v160
	v_sub_f32_e32 v87, v87, v160
	v_sub_f32_e32 v88, v88, v160
	; sched_group_barrier mask(0x00000008) size(1) SyncID(10)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(10)
	v_mfma_f32_32x32x16_bf16 v[16:31], v[136:139], v[104:107], v[16:31]
	v_sub_f32_e32 v89, v89, v160
	v_sub_f32_e32 v90, v90, v160
	v_sub_f32_e32 v91, v91, v160
	v_sub_f32_e32 v92, v92, v160
	v_sub_f32_e32 v93, v93, v160
	; sched_group_barrier mask(0x00000008) size(1) SyncID(10)
	; sched_group_barrier mask(0x00000002) size(5) SyncID(10)
	v_mfma_f32_32x32x16_bf16 v[16:31], v[132:135], v[112:115], v[16:31]
	v_exp_f32_e32 v80, v161
	v_exp_f32_e32 v64, v64
	v_exp_f32_e32 v65, v65
	; sched_group_barrier mask(0x00000008) size(1) SyncID(10)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(10)
	v_mfma_f32_32x32x16_bf16 v[16:31], v[128:131], v[108:111], v[16:31]
	v_exp_f32_e32 v66, v66
	v_exp_f32_e32 v67, v67
	v_exp_f32_e32 v68, v68
	; sched_group_barrier mask(0x00000008) size(1) SyncID(10)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(10)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[124:127], v[100:103], v[0:15]
	v_exp_f32_e32 v69, v69
	v_exp_f32_e32 v70, v70
	v_exp_f32_e32 v71, v71
	; sched_group_barrier mask(0x00000008) size(1) SyncID(10)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(10)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[120:123], v[104:107], v[0:15]
	v_exp_f32_e32 v72, v72
	v_exp_f32_e32 v73, v73
	v_exp_f32_e32 v74, v74
	; sched_group_barrier mask(0x00000008) size(1) SyncID(10)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(10)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[116:119], v[112:115], v[0:15]
	v_exp_f32_e32 v75, v75
	v_exp_f32_e32 v76, v76
	v_exp_f32_e32 v77, v77
	; sched_group_barrier mask(0x00000008) size(1) SyncID(10)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(10)
	v_mfma_f32_32x32x16_bf16 v[0:15], v[96:99], v[108:111], v[0:15]
	v_exp_f32_e32 v78, v78
	v_exp_f32_e32 v79, v79
	; sched_group_barrier mask(0x00000008) size(1) SyncID(10)
	; sched_group_barrier mask(0x00000400) size(3) SyncID(10)
	; sched_barrier mask(0x00000000)
	v_exp_f32_e32 v97, v81
	v_add_f32_e32 v81, v65, v64
	v_add_f32_e32 v81, v81, v66
	v_add_f32_e32 v81, v81, v67
	v_add_f32_e32 v81, v81, v68
	v_add_f32_e32 v81, v81, v69
	v_add_f32_e32 v81, v81, v70
	v_add_f32_e32 v81, v81, v71
	v_add_f32_e32 v81, v81, v72
	v_add_f32_e32 v81, v81, v73
	v_add_f32_e32 v81, v81, v74
	v_add_f32_e32 v81, v81, v75
	v_exp_f32_e32 v96, v144
	v_add_f32_e32 v81, v81, v76
	v_add_f32_e32 v81, v81, v77
	v_exp_f32_e32 v98, v82
	v_add_f32_e32 v81, v81, v78
	v_exp_f32_e32 v83, v83
	v_add_f32_e32 v81, v81, v79
	v_exp_f32_e32 v84, v84
	v_add_f32_e32 v81, v81, v96
	v_exp_f32_e32 v85, v85
	v_add_f32_e32 v81, v81, v97
	v_exp_f32_e32 v86, v86
	v_add_f32_e32 v81, v81, v98
	v_exp_f32_e32 v87, v87
	v_add_f32_e32 v81, v81, v83
	v_exp_f32_e32 v88, v88
	v_add_f32_e32 v81, v81, v84
	v_exp_f32_e32 v89, v89
	v_add_f32_e32 v81, v81, v85
	v_exp_f32_e32 v90, v90
	v_add_f32_e32 v81, v81, v86
	v_exp_f32_e32 v91, v91
	v_add_f32_e32 v81, v81, v87
	v_exp_f32_e32 v92, v92
	v_add_f32_e32 v81, v81, v88
	v_exp_f32_e32 v93, v93
	v_add_f32_e32 v81, v81, v89
	v_exp_f32_e32 v94, v94
	v_add_f32_e32 v81, v81, v90
	v_exp_f32_e32 v95, v95
	v_add_f32_e32 v81, v81, v91
	v_add_f32_e32 v81, v81, v92
	v_add_f32_e32 v81, v81, v93
	v_add_f32_e32 v81, v81, v94
	v_add_f32_e32 v81, v81, v95
	v_mov_b32_e32 v82, v81
	s_nop 1
	v_permlane32_swap_b32_e64 v81, v82 bound_ctrl:1
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v64, v64, v65
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v65, v66, v67
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v66, v68, v69
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v67, v70, v71
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v68, v72, v73
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v69, v74, v75
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v70, v76, v77
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v71, v78, v79
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v72, v96, v97
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v73, v98, v83
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v74, v84, v85
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v75, v86, v87
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v76, v88, v89
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v77, v90, v91
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v78, v92, v93
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v79, v94, v95
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	v_pk_mul_f32 v[62:63], v[80:81], v[62:63] op_sel_hi:[0,1]
	v_pk_mul_f32 v[60:61], v[80:81], v[60:61] op_sel_hi:[0,1]
	v_pk_mul_f32 v[58:59], v[80:81], v[58:59] op_sel_hi:[0,1]
	v_pk_mul_f32 v[56:57], v[80:81], v[56:57] op_sel_hi:[0,1]
	v_pk_mul_f32 v[54:55], v[80:81], v[54:55] op_sel_hi:[0,1]
	v_pk_mul_f32 v[52:53], v[80:81], v[52:53] op_sel_hi:[0,1]
	v_pk_mul_f32 v[50:51], v[80:81], v[50:51] op_sel_hi:[0,1]
	v_pk_mul_f32 v[48:49], v[80:81], v[48:49] op_sel_hi:[0,1]
	v_pk_mul_f32 v[46:47], v[80:81], v[46:47] op_sel_hi:[0,1]
	v_pk_mul_f32 v[44:45], v[80:81], v[44:45] op_sel_hi:[0,1]
	v_pk_mul_f32 v[42:43], v[80:81], v[42:43] op_sel_hi:[0,1]
	v_pk_mul_f32 v[40:41], v[80:81], v[40:41] op_sel_hi:[0,1]
	v_pk_mul_f32 v[38:39], v[80:81], v[38:39] op_sel_hi:[0,1]
	v_pk_mul_f32 v[36:37], v[80:81], v[36:37] op_sel_hi:[0,1]
	v_pk_mul_f32 v[34:35], v[80:81], v[34:35] op_sel_hi:[0,1]
	v_pk_mul_f32 v[32:33], v[80:81], v[32:33] op_sel_hi:[0,1]
	v_pk_mul_f32 v[30:31], v[80:81], v[30:31] op_sel_hi:[0,1]
	v_pk_mul_f32 v[28:29], v[80:81], v[28:29] op_sel_hi:[0,1]
	v_pk_mul_f32 v[26:27], v[80:81], v[26:27] op_sel_hi:[0,1]
	v_pk_mul_f32 v[24:25], v[80:81], v[24:25] op_sel_hi:[0,1]
	v_pk_mul_f32 v[22:23], v[80:81], v[22:23] op_sel_hi:[0,1]
	v_pk_mul_f32 v[20:21], v[80:81], v[20:21] op_sel_hi:[0,1]
	v_pk_mul_f32 v[18:19], v[80:81], v[18:19] op_sel_hi:[0,1]
	v_pk_mul_f32 v[16:17], v[80:81], v[16:17] op_sel_hi:[0,1]
	v_pk_mul_f32 v[14:15], v[80:81], v[14:15] op_sel_hi:[0,1]
	v_pk_mul_f32 v[12:13], v[80:81], v[12:13] op_sel_hi:[0,1]
	v_pk_mul_f32 v[10:11], v[80:81], v[10:11] op_sel_hi:[0,1]
	v_pk_mul_f32 v[8:9], v[80:81], v[8:9] op_sel_hi:[0,1]
	v_pk_mul_f32 v[6:7], v[80:81], v[6:7] op_sel_hi:[0,1]
	v_pk_mul_f32 v[4:5], v[80:81], v[4:5] op_sel_hi:[0,1]
	v_pk_mul_f32 v[2:3], v[80:81], v[2:3] op_sel_hi:[0,1]
	v_pk_mul_f32 v[0:1], v[80:81], v[0:1] op_sel_hi:[0,1]
	s_barrier
	; sched_barrier mask(0x00000000)
	scratch_load_dword v83, off, off offset:152 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_read_b64_tr_b16 v[84:85], v83 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[88:89], v83 offset:0x200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[92:93], v83 offset:0x400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[96:97], v83 offset:0x600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[100:101], v83 offset:0x1000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[104:105], v83 offset:0x1200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[108:109], v83 offset:0x1400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[112:113], v83 offset:0x1600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[116:117], v83 offset:0x2000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[120:121], v83 offset:0x2200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[124:125], v83 offset:0x2400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[128:129], v83 offset:0x2600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[132:133], v83 offset:0x3000

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[136:137], v83 offset:0x3200

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[140:141], v83 offset:0x3400

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[144:145], v83 offset:0x3600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[86:87], v83 offset:0x800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[90:91], v83 offset:0xa00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[94:95], v83 offset:0xc00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[98:99], v83 offset:0xe00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[102:103], v83 offset:0x1800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[106:107], v83 offset:0x1a00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[110:111], v83 offset:0x1c00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[114:115], v83 offset:0x1e00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[118:119], v83 offset:0x2800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[122:123], v83 offset:0x2a00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[126:127], v83 offset:0x2c00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[130:131], v83 offset:0x2e00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[134:135], v83 offset:0x3800

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[138:139], v83 offset:0x3a00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[142:143], v83 offset:0x3c00

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[146:147], v83 offset:0x3e00

	;;#ASMEND
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	v_mfma_f32_32x32x16_bf16 v[48:63], v[84:87], v[64:67], v[48:63]
	v_mfma_f32_32x32x16_bf16 v[32:47], v[88:91], v[64:67], v[32:47]
	v_mfma_f32_32x32x16_bf16 v[16:31], v[92:95], v[64:67], v[16:31]
	v_mfma_f32_32x32x16_bf16 v[0:15], v[96:99], v[64:67], v[0:15]
	v_mfma_f32_32x32x16_bf16 v[48:63], v[100:103], v[68:71], v[48:63]
	v_mfma_f32_32x32x16_bf16 v[32:47], v[104:107], v[68:71], v[32:47]
	v_mfma_f32_32x32x16_bf16 v[16:31], v[108:111], v[68:71], v[16:31]
	v_mfma_f32_32x32x16_bf16 v[0:15], v[112:115], v[68:71], v[0:15]
	v_mfma_f32_32x32x16_bf16 v[48:63], v[116:119], v[72:75], v[48:63]
	v_mfma_f32_32x32x16_bf16 v[32:47], v[120:123], v[72:75], v[32:47]
	v_mfma_f32_32x32x16_bf16 v[16:31], v[124:127], v[72:75], v[16:31]
	v_mfma_f32_32x32x16_bf16 v[0:15], v[128:131], v[72:75], v[0:15]
	v_mfma_f32_32x32x16_bf16 v[48:63], v[132:135], v[76:79], v[48:63]
	v_mfma_f32_32x32x16_bf16 v[32:47], v[136:139], v[76:79], v[32:47]
	v_mfma_f32_32x32x16_bf16 v[16:31], v[140:143], v[76:79], v[16:31]
	v_mfma_f32_32x32x16_bf16 v[0:15], v[144:147], v[76:79], v[0:15]
	; sched_barrier mask(0x00000000)
	s_barrier
	; sched_barrier mask(0x00000000)
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_cbranch_execz .LBB0_42
; %bb.41:
	s_barrier
.LBB0_42:
	s_or_b64 exec, exec, s[2:3]
	scratch_load_dword v64, off, off offset:16 ; 4-byte Folded Reload
	scratch_load_dword v65, off, off offset:20 ; 4-byte Folded Reload
	s_mul_i32 s0, s4, s12
	s_mul_i32 s1, s10, s12
	s_mov_b64 s[2:3], exec
	s_waitcnt vmcnt(1)
	v_add_f32_e32 v64, v64, v236
	s_waitcnt vmcnt(0)
	v_add_f32_e32 v64, v64, v65
	v_fmac_f32_e32 v227, v224, v64
	v_add_f32_e32 v64, v227, v228
	v_fmac_f32_e32 v177, v176, v64
	v_add_f32_e32 v64, v177, v178
	v_fmac_f32_e32 v81, v80, v64
	v_add_f32_e32 v64, v81, v82
	v_rcp_f32_e32 v82, v64
	v_mov_b32_e32 v80, s8
	v_mov_b32_e32 v81, s9
	v_mul_f32_e32 v66, v5, v82
	v_mul_f32_e32 v5, v49, v82
	v_mul_f32_e32 v49, v53, v82
	scratch_load_dword v53, off, off offset:312 ; 4-byte Folded Reload
	v_mul_f32_e32 v69, v0, v82
	v_add_u32_e32 v0, s0, v184
	v_mul_lo_u32 v0, v0, s14
	v_add_u32_e32 v0, s5, v0
	s_mul_i32 s0, s14, s34
	v_mul_lo_u32 v0, v0, s34
	s_mul_i32 s1, s0, s1
	v_mul_f32_e32 v70, v1, v82
	v_mul_f32_e32 v65, v4, v82
	v_mul_f32_e32 v67, v6, v82
	v_mul_f32_e32 v6, v12, v82
	v_mul_f32_e32 v4, v48, v82
	v_mul_f32_e32 v12, v50, v82
	v_mul_f32_e32 v48, v52, v82
	v_ashrrev_i32_e32 v1, 31, v0
	s_lshl_b32 s1, s1, 1
	v_mul_f32_e32 v15, v15, v82
	v_mul_f32_e32 v14, v14, v82
	v_mul_f32_e32 v71, v2, v82
	v_mul_f32_e32 v72, v3, v82
	v_mul_f32_e32 v68, v7, v82
	v_mul_f32_e32 v8, v8, v82
	v_mul_f32_e32 v9, v9, v82
	v_mul_f32_e32 v10, v10, v82
	v_mul_f32_e32 v11, v11, v82
	v_mul_f32_e32 v7, v13, v82
	v_mul_f32_e32 v13, v31, v82
	v_mul_f32_e32 v30, v30, v82
	v_mul_f32_e32 v74, v16, v82
	v_mul_f32_e32 v75, v17, v82
	v_mul_f32_e32 v76, v18, v82
	v_mul_f32_e32 v77, v19, v82
	v_mul_f32_e32 v31, v20, v82
	v_mul_f32_e32 v73, v21, v82
	v_mul_f32_e32 v22, v22, v82
	v_mul_f32_e32 v23, v23, v82
	v_mul_f32_e32 v18, v24, v82
	v_mul_f32_e32 v19, v25, v82
	v_mul_f32_e32 v20, v26, v82
	v_mul_f32_e32 v21, v27, v82
	v_mul_f32_e32 v16, v28, v82
	v_mul_f32_e32 v17, v29, v82
	v_mul_f32_e32 v24, v47, v82
	v_mul_f32_e32 v25, v46, v82
	v_mul_f32_e32 v46, v32, v82
	v_mul_f32_e32 v47, v33, v82
	v_mul_f32_e32 v78, v34, v82
	v_mul_f32_e32 v79, v35, v82
	v_mul_f32_e32 v34, v36, v82
	v_mul_f32_e32 v35, v37, v82
	v_mul_f32_e32 v36, v38, v82
	v_mul_f32_e32 v37, v39, v82
	v_mul_f32_e32 v28, v40, v82
	v_mul_f32_e32 v29, v41, v82
	v_mul_f32_e32 v32, v42, v82
	v_mul_f32_e32 v33, v43, v82
	v_mul_f32_e32 v26, v44, v82
	v_mul_f32_e32 v27, v45, v82
	v_mul_f32_e32 v38, v63, v82
	v_mul_f32_e32 v39, v62, v82
	v_mul_f32_e32 v62, v51, v82
	v_mul_f32_e32 v50, v54, v82
	v_mul_f32_e32 v51, v55, v82
	v_mul_f32_e32 v42, v56, v82
	v_mul_f32_e32 v43, v57, v82
	v_mul_f32_e32 v44, v58, v82
	v_mul_f32_e32 v45, v59, v82
	v_mul_f32_e32 v40, v60, v82
	v_mul_f32_e32 v41, v61, v82
	v_lshl_add_u64 v[0:1], v[0:1], 1, v[80:81]
	v_mov_b32_e32 v2, s1
	v_mov_b32_e32 v3, 0x20000
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v4, v4, v5
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v5, v12, v62
	;;#ASMEND
	s_waitcnt vmcnt(0)
	v_mul_lo_u32 v52, v53, s0
	v_add_lshl_u32 v12, v52, v240, 1
.LBB0_43:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s8, v0
	v_readfirstlane_b32 s9, v1
	v_readfirstlane_b32 s10, v2
	v_readfirstlane_b32 s11, v3
	v_cmp_eq_u64_e32 vcc, s[8:9], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[10:11], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[4:5], v12, s[8:11], 0 offen
                                        ; implicit-def: $vgpr4_vgpr5
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_43
; %bb.44:
	s_mov_b64 exec, s[2:3]
	s_mov_b64 s[2:3], exec
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v4, v48, v49
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v5, v50, v51
	;;#ASMEND
.LBB0_45:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s8, v0
	v_readfirstlane_b32 s9, v1
	v_readfirstlane_b32 s10, v2
	v_readfirstlane_b32 s11, v3
	v_cmp_eq_u64_e32 vcc, s[8:9], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[10:11], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[4:5], v12, s[8:11], 0 offen offset:16
                                        ; implicit-def: $vgpr4_vgpr5
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_45
; %bb.46:
	s_mov_b64 exec, s[2:3]
	s_mov_b64 s[2:3], exec
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v4, v42, v43
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v5, v44, v45
	;;#ASMEND
.LBB0_47:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s8, v0
	v_readfirstlane_b32 s9, v1
	v_readfirstlane_b32 s10, v2
	v_readfirstlane_b32 s11, v3
	v_cmp_eq_u64_e32 vcc, s[8:9], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[10:11], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[4:5], v12, s[8:11], 0 offen offset:32
                                        ; implicit-def: $vgpr4_vgpr5
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_47
; %bb.48:
	s_mov_b64 exec, s[2:3]
	s_mov_b64 s[2:3], exec
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v4, v40, v41
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v5, v39, v38
	;;#ASMEND
.LBB0_49:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s8, v0
	v_readfirstlane_b32 s9, v1
	v_readfirstlane_b32 s10, v2
	v_readfirstlane_b32 s11, v3
	v_cmp_eq_u64_e32 vcc, s[8:9], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[10:11], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[4:5], v12, s[8:11], 0 offen offset:48
                                        ; implicit-def: $vgpr4_vgpr5
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_49
; %bb.50:
	s_mov_b64 exec, s[2:3]
	s_mov_b64 s[2:3], exec
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v4, v46, v47
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v5, v78, v79
	;;#ASMEND
.LBB0_51:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s8, v0
	v_readfirstlane_b32 s9, v1
	v_readfirstlane_b32 s10, v2
	v_readfirstlane_b32 s11, v3
	v_cmp_eq_u64_e32 vcc, s[8:9], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[10:11], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[4:5], v12, s[8:11], 0 offen offset:64
                                        ; implicit-def: $vgpr4_vgpr5
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_51
; %bb.52:
	s_mov_b64 exec, s[2:3]
	s_mov_b64 s[2:3], exec
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v4, v34, v35
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v5, v36, v37
	;;#ASMEND
.LBB0_53:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s8, v0
	v_readfirstlane_b32 s9, v1
	v_readfirstlane_b32 s10, v2
	v_readfirstlane_b32 s11, v3
	v_cmp_eq_u64_e32 vcc, s[8:9], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[10:11], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[4:5], v12, s[8:11], 0 offen offset:80
                                        ; implicit-def: $vgpr4_vgpr5
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_53
; %bb.54:
	s_mov_b64 exec, s[2:3]
	s_mov_b64 s[2:3], exec
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v4, v28, v29
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v5, v32, v33
	;;#ASMEND
.LBB0_55:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s8, v0
	v_readfirstlane_b32 s9, v1
	v_readfirstlane_b32 s10, v2
	v_readfirstlane_b32 s11, v3
	v_cmp_eq_u64_e32 vcc, s[8:9], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[10:11], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[4:5], v12, s[8:11], 0 offen offset:96
                                        ; implicit-def: $vgpr4_vgpr5
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_55
; %bb.56:
	s_mov_b64 exec, s[2:3]
	s_mov_b64 s[2:3], exec
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v4, v26, v27
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v5, v25, v24
	;;#ASMEND
.LBB0_57:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s8, v0
	v_readfirstlane_b32 s9, v1
	v_readfirstlane_b32 s10, v2
	v_readfirstlane_b32 s11, v3
	v_cmp_eq_u64_e32 vcc, s[8:9], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[10:11], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[4:5], v12, s[8:11], 0 offen offset:112
                                        ; implicit-def: $vgpr4_vgpr5
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_57
; %bb.58:
	s_mov_b64 exec, s[2:3]
	s_mov_b64 s[2:3], exec
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v4, v74, v75
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v5, v76, v77
	;;#ASMEND
.LBB0_59:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s8, v0
	v_readfirstlane_b32 s9, v1
	v_readfirstlane_b32 s10, v2
	v_readfirstlane_b32 s11, v3
	v_cmp_eq_u64_e32 vcc, s[8:9], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[10:11], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[4:5], v12, s[8:11], 0 offen offset:128
                                        ; implicit-def: $vgpr4_vgpr5
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_59
; %bb.60:
	s_mov_b64 exec, s[2:3]
	s_mov_b64 s[2:3], exec
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v4, v31, v73
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v5, v22, v23
	;;#ASMEND
.LBB0_61:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s8, v0
	v_readfirstlane_b32 s9, v1
	v_readfirstlane_b32 s10, v2
	v_readfirstlane_b32 s11, v3
	v_cmp_eq_u64_e32 vcc, s[8:9], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[10:11], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[4:5], v12, s[8:11], 0 offen offset:144
                                        ; implicit-def: $vgpr4_vgpr5
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_61
; %bb.62:
	s_mov_b64 exec, s[2:3]
	s_mov_b64 s[2:3], exec
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v4, v18, v19
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v5, v20, v21
	;;#ASMEND
.LBB0_63:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s8, v0
	v_readfirstlane_b32 s9, v1
	v_readfirstlane_b32 s10, v2
	v_readfirstlane_b32 s11, v3
	v_cmp_eq_u64_e32 vcc, s[8:9], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[10:11], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[4:5], v12, s[8:11], 0 offen offset:160
                                        ; implicit-def: $vgpr4_vgpr5
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_63
; %bb.64:
	s_mov_b64 exec, s[2:3]
	s_mov_b64 s[2:3], exec
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v4, v16, v17
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v5, v30, v13
	;;#ASMEND
.LBB0_65:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s8, v0
	v_readfirstlane_b32 s9, v1
	v_readfirstlane_b32 s10, v2
	v_readfirstlane_b32 s11, v3
	v_cmp_eq_u64_e32 vcc, s[8:9], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[10:11], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[4:5], v12, s[8:11], 0 offen offset:176
                                        ; implicit-def: $vgpr4_vgpr5
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_65
; %bb.66:
	s_mov_b64 exec, s[2:3]
	s_mov_b64 s[2:3], exec
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v4, v69, v70
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v5, v71, v72
	;;#ASMEND
.LBB0_67:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s8, v0
	v_readfirstlane_b32 s9, v1
	v_readfirstlane_b32 s10, v2
	v_readfirstlane_b32 s11, v3
	v_cmp_eq_u64_e32 vcc, s[8:9], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[10:11], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[4:5], v12, s[8:11], 0 offen offset:192
                                        ; implicit-def: $vgpr4_vgpr5
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_67
; %bb.68:
	s_mov_b64 exec, s[2:3]
	s_mov_b64 s[2:3], exec
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v4, v65, v66
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v5, v67, v68
	;;#ASMEND
.LBB0_69:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s8, v0
	v_readfirstlane_b32 s9, v1
	v_readfirstlane_b32 s10, v2
	v_readfirstlane_b32 s11, v3
	v_cmp_eq_u64_e32 vcc, s[8:9], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[10:11], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[4:5], v12, s[8:11], 0 offen offset:208
                                        ; implicit-def: $vgpr4_vgpr5
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_69
; %bb.70:
	s_mov_b64 exec, s[2:3]
	s_mov_b64 s[2:3], exec
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v4, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v5, v10, v11
	;;#ASMEND
.LBB0_71:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s8, v0
	v_readfirstlane_b32 s9, v1
	v_readfirstlane_b32 s10, v2
	v_readfirstlane_b32 s11, v3
	v_cmp_eq_u64_e32 vcc, s[8:9], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[10:11], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[4:5], v12, s[8:11], 0 offen offset:224
                                        ; implicit-def: $vgpr4_vgpr5
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_71
; %bb.72:
	s_mov_b64 exec, s[2:3]
	s_mov_b64 s[2:3], exec
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v4, v6, v7
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v5, v14, v15
	;;#ASMEND
.LBB0_73:                               ; =>This Inner Loop Header: Depth=1
	v_readfirstlane_b32 s8, v0
	v_readfirstlane_b32 s9, v1
	v_readfirstlane_b32 s10, v2
	v_readfirstlane_b32 s11, v3
	v_cmp_eq_u64_e32 vcc, s[8:9], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[10:11], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_dwordx2 v[4:5], v12, s[8:11], 0 offen offset:240
                                        ; implicit-def: $vgpr0_vgpr1_vgpr2_vgpr3
                                        ; implicit-def: $vgpr4_vgpr5
                                        ; implicit-def: $vgpr12
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_73
; %bb.74:
	s_mov_b64 exec, s[2:3]
	s_mov_b32 s0, 0x800000
	v_cmp_gt_f32_e32 vcc, s0, v64
	s_mul_i32 s0, s4, s16
	s_add_i32 s0, s0, s5
	v_cndmask_b32_e64 v0, 0, 32, vcc
	v_ldexp_f32 v0, v64, v0
	v_log_f32_e32 v0, v0
	v_mov_b32_e32 v1, 0xc1b17218
	s_mul_i32 s0, s0, s18
	v_cndmask_b32_e32 v4, 0, v1, vcc
	s_mul_i32 s0, s0, s28
	v_fmac_f32_e32 v4, 0x3f317218, v0
	v_add_u32_e32 v0, s0, v184
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshl_add_u64 v[0:1], v[0:1], 2, s[6:7]
	v_lshlrev_b32_e32 v2, 2, v53
	v_mov_b32_e32 v3, 0
	v_fmac_f32_e32 v4, 0x3f317218, v160
	v_lshl_add_u64 v[0:1], v[0:1], 0, v[2:3]
	global_store_dword v[0:1], v4, off
	s_endpgm
.LBB0_75:
	v_and_b32_e32 v41, 4, v36
	v_or_b32_e32 v42, v103, v104
	v_sub_u32_e32 v41, v42, v41
	v_mov_b32_e32 v42, 0xff800000
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v41, 0
	v_cmp_lt_i32_e64 s[24:25], v41, 1
	v_cndmask_b32_e64 v18, v18, v42, s[20:21]
	v_cndmask_b32_e64 v19, v19, v42, s[24:25]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v41, 2
	v_cmp_lt_i32_e64 s[24:25], v41, 3
	v_cndmask_b32_e64 v20, v20, v42, s[20:21]
	v_cndmask_b32_e64 v21, v21, v42, s[24:25]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v41, 8
	v_cmp_lt_i32_e64 s[24:25], v41, 9
	v_cndmask_b32_e64 v22, v22, v42, s[20:21]
	v_cndmask_b32_e64 v23, v23, v42, s[24:25]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v41, 10
	v_cmp_lt_i32_e64 s[24:25], v41, 11
	v_cndmask_b32_e64 v24, v24, v42, s[20:21]
	v_cndmask_b32_e64 v25, v25, v42, s[24:25]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v41, 16
	v_cmp_lt_i32_e64 s[24:25], v41, 17
	v_cndmask_b32_e64 v26, v26, v42, s[20:21]
	v_cndmask_b32_e64 v27, v27, v42, s[24:25]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v41, 18
	v_cmp_lt_i32_e64 s[24:25], v41, 19
	v_cndmask_b32_e64 v28, v28, v42, s[20:21]
	v_cndmask_b32_e64 v29, v29, v42, s[24:25]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v41, 24
	v_cmp_lt_i32_e64 s[24:25], v41, 25
	v_cndmask_b32_e64 v30, v30, v42, s[20:21]
	v_cndmask_b32_e64 v31, v31, v42, s[24:25]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v41, 26
	v_cmp_lt_i32_e64 s[24:25], v41, 27
	v_cndmask_b32_e64 v32, v32, v42, s[20:21]
	v_cndmask_b32_e64 v33, v33, v42, s[24:25]
	
	;;#ASMEND
	v_subrev_u32_e32 v41, 32, v41
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v41, 0
	v_cmp_lt_i32_e64 s[24:25], v41, 1
	v_cndmask_b32_e64 v2, v2, v42, s[20:21]
	v_cndmask_b32_e64 v3, v3, v42, s[24:25]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v41, 2
	v_cmp_lt_i32_e64 s[24:25], v41, 3
	v_cndmask_b32_e64 v4, v4, v42, s[20:21]
	v_cndmask_b32_e64 v5, v5, v42, s[24:25]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v41, 8
	v_cmp_lt_i32_e64 s[24:25], v41, 9
	v_cndmask_b32_e64 v6, v6, v42, s[20:21]
	v_cndmask_b32_e64 v7, v7, v42, s[24:25]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v41, 10
	v_cmp_lt_i32_e64 s[24:25], v41, 11
	v_cndmask_b32_e64 v8, v8, v42, s[20:21]
	v_cndmask_b32_e64 v9, v9, v42, s[24:25]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v41, 16
	v_cmp_lt_i32_e64 s[24:25], v41, 17
	v_cndmask_b32_e64 v10, v10, v42, s[20:21]
	v_cndmask_b32_e64 v11, v11, v42, s[24:25]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v41, 18
	v_cmp_lt_i32_e64 s[24:25], v41, 19
	v_cndmask_b32_e64 v12, v12, v42, s[20:21]
	v_cndmask_b32_e64 v13, v13, v42, s[24:25]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v41, 24
	v_cmp_lt_i32_e64 s[24:25], v41, 25
	v_cndmask_b32_e64 v14, v14, v42, s[20:21]
	v_cndmask_b32_e64 v15, v15, v42, s[24:25]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v41, 26
	v_cmp_lt_i32_e64 s[24:25], v41, 27
	v_cndmask_b32_e64 v16, v16, v42, s[20:21]
	v_cndmask_b32_e64 v17, v17, v42, s[24:25]
	
	;;#ASMEND
	s_branch .LBB0_25
.LBB0_76:
	scratch_load_dword v97, off, off offset:312 ; 4-byte Folded Reload
	v_mov_b32_e32 v98, 0xff800000
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v96, v96, v97
	v_or_b32_e32 v97, s13, v240
	v_sub_u32_e32 v96, v96, v97
	v_add_u32_e32 v97, 0xc0, v96
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v97, 0
	v_cmp_lt_i32_e64 s[24:25], v97, 1
	v_cndmask_b32_e64 v80, v80, v98, s[20:21]
	v_cndmask_b32_e64 v81, v81, v98, s[24:25]
	
	;;#ASMEND
	v_add_u32_e32 v96, 0xa0, v96
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v97, 2
	v_cmp_lt_i32_e64 s[24:25], v97, 3
	v_cndmask_b32_e64 v82, v82, v98, s[20:21]
	v_cndmask_b32_e64 v83, v83, v98, s[24:25]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v97, 8
	v_cmp_lt_i32_e64 s[24:25], v97, 9
	v_cndmask_b32_e64 v84, v84, v98, s[20:21]
	v_cndmask_b32_e64 v85, v85, v98, s[24:25]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v97, 10
	v_cmp_lt_i32_e64 s[24:25], v97, 11
	v_cndmask_b32_e64 v86, v86, v98, s[20:21]
	v_cndmask_b32_e64 v87, v87, v98, s[24:25]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v97, 16
	v_cmp_lt_i32_e64 s[24:25], v97, 17
	v_cndmask_b32_e64 v88, v88, v98, s[20:21]
	v_cndmask_b32_e64 v89, v89, v98, s[24:25]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v97, 18
	v_cmp_lt_i32_e64 s[24:25], v97, 19
	v_cndmask_b32_e64 v90, v90, v98, s[20:21]
	v_cndmask_b32_e64 v91, v91, v98, s[24:25]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v97, 24
	v_cmp_lt_i32_e64 s[24:25], v97, 25
	v_cndmask_b32_e64 v92, v92, v98, s[20:21]
	v_cndmask_b32_e64 v93, v93, v98, s[24:25]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v97, 26
	v_cmp_lt_i32_e64 s[24:25], v97, 27
	v_cndmask_b32_e64 v94, v94, v98, s[20:21]
	v_cndmask_b32_e64 v95, v95, v98, s[24:25]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v96, 0
	v_cmp_lt_i32_e64 s[24:25], v96, 1
	v_cndmask_b32_e64 v64, v64, v98, s[20:21]
	v_cndmask_b32_e64 v65, v65, v98, s[24:25]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v96, 2
	v_cmp_lt_i32_e64 s[24:25], v96, 3
	v_cndmask_b32_e64 v66, v66, v98, s[20:21]
	v_cndmask_b32_e64 v67, v67, v98, s[24:25]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v96, 8
	v_cmp_lt_i32_e64 s[24:25], v96, 9
	v_cndmask_b32_e64 v68, v68, v98, s[20:21]
	v_cndmask_b32_e64 v69, v69, v98, s[24:25]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v96, 10
	v_cmp_lt_i32_e64 s[24:25], v96, 11
	v_cndmask_b32_e64 v70, v70, v98, s[20:21]
	v_cndmask_b32_e64 v71, v71, v98, s[24:25]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v96, 16
	v_cmp_lt_i32_e64 s[24:25], v96, 17
	v_cndmask_b32_e64 v72, v72, v98, s[20:21]
	v_cndmask_b32_e64 v73, v73, v98, s[24:25]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v96, 18
	v_cmp_lt_i32_e64 s[24:25], v96, 19
	v_cndmask_b32_e64 v74, v74, v98, s[20:21]
	v_cndmask_b32_e64 v75, v75, v98, s[24:25]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v96, 24
	v_cmp_lt_i32_e64 s[24:25], v96, 25
	v_cndmask_b32_e64 v76, v76, v98, s[20:21]
	v_cndmask_b32_e64 v77, v77, v98, s[24:25]
	
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v96, 26
	v_cmp_lt_i32_e64 s[24:25], v96, 27
	v_cndmask_b32_e64 v78, v78, v98, s[20:21]
	v_cndmask_b32_e64 v79, v79, v98, s[24:25]
	
	;;#ASMEND
	s_branch .LBB0_36
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z10attend_ker12attn_globals
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 324
		.amdhsa_kernarg_size 248
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 1
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 256
		.amdhsa_next_free_sgpr 56
		.amdhsa_accum_offset 256
		.amdhsa_reserve_vcc 1
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
	.size	_Z10attend_ker12attn_globals, .Lfunc_end0-_Z10attend_ker12attn_globals
                                        ; -- End function
	.set _Z10attend_ker12attn_globals.num_vgpr, 256
	.set _Z10attend_ker12attn_globals.num_agpr, 0
	.set _Z10attend_ker12attn_globals.numbered_sgpr, 56
	.set _Z10attend_ker12attn_globals.private_seg_size, 324
	.set _Z10attend_ker12attn_globals.uses_vcc, 1
	.set _Z10attend_ker12attn_globals.uses_flat_scratch, 0
	.set _Z10attend_ker12attn_globals.has_dyn_sized_stack, 0
	.set _Z10attend_ker12attn_globals.has_recursion, 0
	.set _Z10attend_ker12attn_globals.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 28620
; TotalNumSgprs: 62
; NumVgprs: 256
; NumAgprs: 0
; TotalNumVgprs: 256
; ScratchSize: 324
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 7
; VGPRBlocks: 31
; NumSGPRsForWavesPerEU: 62
; NumVGPRsForWavesPerEU: 256
; AccumOffset: 256
; Occupancy: 2
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 1
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 63
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.type	__hip_cuid_4cadff010e0e26a3,@object ; @__hip_cuid_4cadff010e0e26a3
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_4cadff010e0e26a3
__hip_cuid_4cadff010e0e26a3:
	.byte	0                               ; 0x0
	.size	__hip_cuid_4cadff010e0e26a3, 1

	.ident	"clang version 20.0.0git (https://github.com/ROCm/llvm-project.git 27682a16360e33e37c4f3cc6adf9a620733f8fe1)"
	.ident	"AMD clang version 20.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.1.0 25425 1b0eada6b0ee93e2e694c8c146d23fca90bc11c5)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __shm
	.addrsig_sym __hip_cuid_4cadff010e0e26a3
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .offset:         0
        .size:           248
        .value_kind:     by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 248
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 512
    .name:           _Z10attend_ker12attn_globals
    .private_segment_fixed_size: 324
    .sgpr_count:     62
    .sgpr_spill_count: 0
    .symbol:         _Z10attend_ker12attn_globals.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     256
    .vgpr_spill_count: 85
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
