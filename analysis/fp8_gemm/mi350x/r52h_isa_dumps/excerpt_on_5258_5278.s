	v_mfma_scale_f32_16x16x128_f8f6f4 v[98:101], v[222:229], v[238:245], v[98:101], v148, v161 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[102:105], v[222:229], v[246:253], v[102:105], v148, v161 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[106:109], v[230:237], v[238:245], v[106:109], v148, v161 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[110:113], v[230:237], v[246:253], v[110:113], v148, v161 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0
	s_barrier
	;;#ASMSTART
	;;#ASMEND
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
