===== MFMA #73 (context lines 68..74) =====
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	s_waitcnt vmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[142:145], v[184:191], v[2:9], v[142:145], v146, v158 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[138:141], v[184:191], v[10:17], v[138:141], v146, v158 op_sel:[0,1,0] op_sel_hi:[0,0,0]

===== MFMA #74 (context lines 69..75) =====
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	s_waitcnt vmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[142:145], v[184:191], v[2:9], v[142:145], v146, v158 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[138:141], v[184:191], v[10:17], v[138:141], v146, v158 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[126:129], v[214:221], v[2:9], v[126:129], v146, v158 op_sel:[1,0,0] op_sel_hi:[0,0,0]

===== MFMA #75 (context lines 70..76) =====
	;;#ASMEND
	s_setprio 1
	s_waitcnt vmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[142:145], v[184:191], v[2:9], v[142:145], v146, v158 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[138:141], v[184:191], v[10:17], v[138:141], v146, v158 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[126:129], v[214:221], v[2:9], v[126:129], v146, v158 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[134:137], v[214:221], v[10:17], v[134:137], v146, v158 op_sel:[1,1,0] op_sel_hi:[0,0,0]

===== MFMA #76 (context lines 71..77) =====
	s_setprio 1
	s_waitcnt vmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[142:145], v[184:191], v[2:9], v[142:145], v146, v158 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[138:141], v[184:191], v[10:17], v[138:141], v146, v158 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[126:129], v[214:221], v[2:9], v[126:129], v146, v158 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[134:137], v[214:221], v[10:17], v[134:137], v146, v158 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[122:125], v[222:229], v[2:9], v[122:125], v148, v158 op_sel_hi:[0,0,0]

===== MFMA #77 (context lines 72..78) =====
	s_waitcnt vmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[142:145], v[184:191], v[2:9], v[142:145], v146, v158 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[138:141], v[184:191], v[10:17], v[138:141], v146, v158 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[126:129], v[214:221], v[2:9], v[126:129], v146, v158 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[134:137], v[214:221], v[10:17], v[134:137], v146, v158 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[122:125], v[222:229], v[2:9], v[122:125], v148, v158 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[130:133], v[222:229], v[10:17], v[130:133], v148, v158 op_sel:[0,1,0] op_sel_hi:[0,0,0]

===== MFMA #78 (context lines 73..79) =====
	v_mfma_scale_f32_16x16x128_f8f6f4 v[142:145], v[184:191], v[2:9], v[142:145], v146, v158 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[138:141], v[184:191], v[10:17], v[138:141], v146, v158 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[126:129], v[214:221], v[2:9], v[126:129], v146, v158 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[134:137], v[214:221], v[10:17], v[134:137], v146, v158 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[122:125], v[222:229], v[2:9], v[122:125], v148, v158 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[130:133], v[222:229], v[10:17], v[130:133], v148, v158 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[118:121], v[230:237], v[2:9], v[118:121], v148, v158 op_sel:[1,0,0] op_sel_hi:[0,0,0]

===== MFMA #79 (context lines 74..80) =====
	v_mfma_scale_f32_16x16x128_f8f6f4 v[138:141], v[184:191], v[10:17], v[138:141], v146, v158 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[126:129], v[214:221], v[2:9], v[126:129], v146, v158 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[134:137], v[214:221], v[10:17], v[134:137], v146, v158 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[122:125], v[222:229], v[2:9], v[122:125], v148, v158 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[130:133], v[222:229], v[10:17], v[130:133], v148, v158 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[118:121], v[230:237], v[2:9], v[118:121], v148, v158 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[114:117], v[230:237], v[10:17], v[114:117], v148, v158 op_sel:[1,1,0] op_sel_hi:[0,0,0]

===== MFMA #80 (context lines 75..81) =====
	v_mfma_scale_f32_16x16x128_f8f6f4 v[126:129], v[214:221], v[2:9], v[126:129], v146, v158 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[134:137], v[214:221], v[10:17], v[134:137], v146, v158 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[122:125], v[222:229], v[2:9], v[122:125], v148, v158 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[130:133], v[222:229], v[10:17], v[130:133], v148, v158 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[118:121], v[230:237], v[2:9], v[118:121], v148, v158 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[114:117], v[230:237], v[10:17], v[114:117], v148, v158 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0

===== MFMA #117 (context lines 112..118) =====
	s_barrier
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[184:191], v[238:245], v[90:93], v146, v159 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[184:191], v[246:253], v[82:85], v146, v159 op_sel:[0,1,0] op_sel_hi:[0,0,0]

===== MFMA #118 (context lines 113..119) =====
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[184:191], v[238:245], v[90:93], v146, v159 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[184:191], v[246:253], v[82:85], v146, v159 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[214:221], v[238:245], v[86:89], v146, v159 op_sel:[1,0,0] op_sel_hi:[0,0,0]

===== MFMA #119 (context lines 114..120) =====
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[184:191], v[238:245], v[90:93], v146, v159 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[184:191], v[246:253], v[82:85], v146, v159 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[214:221], v[238:245], v[86:89], v146, v159 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[214:221], v[246:253], v[94:97], v146, v159 op_sel:[1,1,0] op_sel_hi:[0,0,0]

===== MFMA #120 (context lines 115..121) =====
	;;#ASMEND
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[184:191], v[238:245], v[90:93], v146, v159 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[184:191], v[246:253], v[82:85], v146, v159 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[214:221], v[238:245], v[86:89], v146, v159 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[214:221], v[246:253], v[94:97], v146, v159 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[98:101], v[222:229], v[238:245], v[98:101], v148, v159 op_sel_hi:[0,0,0]

===== MFMA #121 (context lines 116..122) =====
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[184:191], v[238:245], v[90:93], v146, v159 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[184:191], v[246:253], v[82:85], v146, v159 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[214:221], v[238:245], v[86:89], v146, v159 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[214:221], v[246:253], v[94:97], v146, v159 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[98:101], v[222:229], v[238:245], v[98:101], v148, v159 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[102:105], v[222:229], v[246:253], v[102:105], v148, v159 op_sel:[0,1,0] op_sel_hi:[0,0,0]

===== MFMA #122 (context lines 117..123) =====
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[184:191], v[238:245], v[90:93], v146, v159 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[184:191], v[246:253], v[82:85], v146, v159 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[214:221], v[238:245], v[86:89], v146, v159 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[214:221], v[246:253], v[94:97], v146, v159 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[98:101], v[222:229], v[238:245], v[98:101], v148, v159 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[102:105], v[222:229], v[246:253], v[102:105], v148, v159 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[106:109], v[230:237], v[238:245], v[106:109], v148, v159 op_sel:[1,0,0] op_sel_hi:[0,0,0]

===== MFMA #123 (context lines 118..124) =====
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[184:191], v[246:253], v[82:85], v146, v159 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[214:221], v[238:245], v[86:89], v146, v159 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[214:221], v[246:253], v[94:97], v146, v159 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[98:101], v[222:229], v[238:245], v[98:101], v148, v159 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[102:105], v[222:229], v[246:253], v[102:105], v148, v159 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[106:109], v[230:237], v[238:245], v[106:109], v148, v159 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[110:113], v[230:237], v[246:253], v[110:113], v148, v159 op_sel:[1,1,0] op_sel_hi:[0,0,0]

===== MFMA #124 (context lines 119..125) =====
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[214:221], v[238:245], v[86:89], v146, v159 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[214:221], v[246:253], v[94:97], v146, v159 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[98:101], v[222:229], v[238:245], v[98:101], v148, v159 op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[102:105], v[222:229], v[246:253], v[102:105], v148, v159 op_sel:[0,1,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[106:109], v[230:237], v[238:245], v[106:109], v148, v159 op_sel:[1,0,0] op_sel_hi:[0,0,0]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[110:113], v[230:237], v[246:253], v[110:113], v148, v159 op_sel:[1,1,0] op_sel_hi:[0,0,0]
	s_setprio 0

