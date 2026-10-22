; mapal-backend-llvm device module
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@mapal_smem = external addrspace(3) global [0 x i8], align 16
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
declare void @llvm.nvvm.barrier0()
declare void @llvm.nvvm.cp.async.cg.shared.global.16(ptr addrspace(3), ptr addrspace(1))
declare void @llvm.nvvm.cp.async.ca.shared.global.4(ptr addrspace(3), ptr addrspace(1))
declare void @llvm.nvvm.cp.async.ca.shared.global.8(ptr addrspace(3), ptr addrspace(1))
declare void @llvm.nvvm.cp.async.commit.group()
declare void @llvm.nvvm.cp.async.wait.group(i32 immarg)
declare float @llvm.fma.f32(float, float, float)
declare double @llvm.fma.f64(double, double, double)

define ptx_kernel void @mapal_tile_21c33b4fe77622e9(ptr addrspace(1) noalias readonly align 16 %a, ptr addrspace(1) noalias readonly align 16 %b, ptr addrspace(1) noalias align 16 %o, ptr addrspace(1) noalias align 16 %y0) "nvvm.maxntid"="32" "nvvm.minctasm"="16" {
entry:
  %v1 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %v2 = zext i32 %v1 to i64
  %v3 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %v4 = zext i32 %v3 to i64
  %v5 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  %v6 = zext i32 %v5 to i64
  %v7 = udiv i64 %v2, 4
  %v8 = urem i64 %v2, 4
  %v9 = mul i64 %v6, 32
  %v10 = mul i64 %v4, 16
  %v11 = alloca [16 x float], align 16
  %v12 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 0
  store float 0x0000000000000000, ptr %v12, align 4
  %v13 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 1
  store float 0x0000000000000000, ptr %v13, align 4
  %v14 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 2
  store float 0x0000000000000000, ptr %v14, align 4
  %v15 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 3
  store float 0x0000000000000000, ptr %v15, align 4
  %v16 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 4
  store float 0x0000000000000000, ptr %v16, align 4
  %v17 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 5
  store float 0x0000000000000000, ptr %v17, align 4
  %v18 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 6
  store float 0x0000000000000000, ptr %v18, align 4
  %v19 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 7
  store float 0x0000000000000000, ptr %v19, align 4
  %v20 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 8
  store float 0x0000000000000000, ptr %v20, align 4
  %v21 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 9
  store float 0x0000000000000000, ptr %v21, align 4
  %v22 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 10
  store float 0x0000000000000000, ptr %v22, align 4
  %v23 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 11
  store float 0x0000000000000000, ptr %v23, align 4
  %v24 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 12
  store float 0x0000000000000000, ptr %v24, align 4
  %v25 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 13
  store float 0x0000000000000000, ptr %v25, align 4
  %v26 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 14
  store float 0x0000000000000000, ptr %v26, align 4
  %v27 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 15
  store float 0x0000000000000000, ptr %v27, align 4
  %v28 = udiv i64 %v2, 2
  %v29 = urem i64 %v2, 2
  %v30 = mul i64 %v29, 4
  %v31 = add i64 %v9, %v28
  %v32 = mul i64 %v31, 64
  %v33 = mul i64 %v30, 1
  %v34 = add i64 %v32, %v33
  %v35 = add i64 %v34, 0
  %v36 = mul i64 %v30, 32
  %v37 = add i64 %v36, %v28
  %v38 = udiv i64 %v2, 4
  %v39 = urem i64 %v2, 4
  %v40 = mul i64 %v39, 4
  %v41 = mul i64 %v38, 64
  %v42 = add i64 %v10, %v40
  %v43 = mul i64 %v42, 1
  %v44 = add i64 %v41, %v43
  %v45 = add i64 %v44, 0
  %v46 = mul i64 %v38, 16
  %v47 = add i64 %v46, %v40
  %v48 = add i64 %v47, 256
  %v49 = mul i64 %v7, 4
  %v50 = mul i64 %v8, 4
  %v51 = mul i64 0, 1
  %v52 = mul i64 0, 64
  %v53 = add i64 %v35, %v51
  %v54 = getelementptr inbounds float, ptr addrspace(1) %a, i64 %v53
  %v55 = load <4 x float>, ptr addrspace(1) %v54, align 16
  %v56 = add i64 %v35, %v51
  %v57 = add i64 %v56, 1024
  %v58 = getelementptr inbounds float, ptr addrspace(1) %a, i64 %v57
  %v59 = load <4 x float>, ptr addrspace(1) %v58, align 16
  %v60 = add i64 %v45, %v52
  %v61 = getelementptr inbounds float, ptr addrspace(1) %b, i64 %v60
  %v62 = load <4 x float>, ptr addrspace(1) %v61, align 16
  br label %kt
kt:
  %v63 = phi i64 [0, %entry], [%v64, %ktl]
  %v65 = phi <4 x float> [%v55, %entry], [%v100, %ktl]
  %v66 = phi <4 x float> [%v59, %entry], [%v104, %ktl]
  %v67 = phi <4 x float> [%v62, %entry], [%v107, %ktl]
  %v68 = extractelement <4 x float> %v65, i64 0
  %v69 = add i64 %v37, 0
  %v70 = getelementptr inbounds float, ptr addrspace(3) @mapal_smem, i64 %v69
  store float %v68, ptr addrspace(3) %v70, align 4
  %v71 = extractelement <4 x float> %v65, i64 1
  %v72 = add i64 %v37, 32
  %v73 = getelementptr inbounds float, ptr addrspace(3) @mapal_smem, i64 %v72
  store float %v71, ptr addrspace(3) %v73, align 4
  %v74 = extractelement <4 x float> %v65, i64 2
  %v75 = add i64 %v37, 64
  %v76 = getelementptr inbounds float, ptr addrspace(3) @mapal_smem, i64 %v75
  store float %v74, ptr addrspace(3) %v76, align 4
  %v77 = extractelement <4 x float> %v65, i64 3
  %v78 = add i64 %v37, 96
  %v79 = getelementptr inbounds float, ptr addrspace(3) @mapal_smem, i64 %v78
  store float %v77, ptr addrspace(3) %v79, align 4
  %v80 = extractelement <4 x float> %v66, i64 0
  %v81 = add i64 %v37, 16
  %v82 = getelementptr inbounds float, ptr addrspace(3) @mapal_smem, i64 %v81
  store float %v80, ptr addrspace(3) %v82, align 4
  %v83 = extractelement <4 x float> %v66, i64 1
  %v84 = add i64 %v37, 48
  %v85 = getelementptr inbounds float, ptr addrspace(3) @mapal_smem, i64 %v84
  store float %v83, ptr addrspace(3) %v85, align 4
  %v86 = extractelement <4 x float> %v66, i64 2
  %v87 = add i64 %v37, 80
  %v88 = getelementptr inbounds float, ptr addrspace(3) @mapal_smem, i64 %v87
  store float %v86, ptr addrspace(3) %v88, align 4
  %v89 = extractelement <4 x float> %v66, i64 3
  %v90 = add i64 %v37, 112
  %v91 = getelementptr inbounds float, ptr addrspace(3) @mapal_smem, i64 %v90
  store float %v89, ptr addrspace(3) %v91, align 4
  %v92 = getelementptr inbounds float, ptr addrspace(3) @mapal_smem, i64 %v48
  store <4 x float> %v67, ptr addrspace(3) %v92, align 16
  call void @llvm.nvvm.barrier0()
  %v93 = add i64 %v63, 8
  %v94 = icmp ult i64 %v93, 64
  %v95 = select i1 %v94, i64 %v93, i64 %v63
  %v96 = mul i64 %v95, 1
  %v97 = mul i64 %v95, 64
  %v98 = add i64 %v35, %v96
  %v99 = getelementptr inbounds float, ptr addrspace(1) %a, i64 %v98
  %v100 = load <4 x float>, ptr addrspace(1) %v99, align 16
  %v101 = add i64 %v35, %v96
  %v102 = add i64 %v101, 1024
  %v103 = getelementptr inbounds float, ptr addrspace(1) %a, i64 %v102
  %v104 = load <4 x float>, ptr addrspace(1) %v103, align 16
  %v105 = add i64 %v45, %v97
  %v106 = getelementptr inbounds float, ptr addrspace(1) %b, i64 %v105
  %v107 = load <4 x float>, ptr addrspace(1) %v106, align 16
  br label %kk
kk:
  %v108 = phi i64 [0, %kt], [%v109, %kk]
  %v110 = mul i64 %v108, 32
  %v111 = add i64 %v110, %v49
  %v112 = add i64 %v111, 0
  %v113 = add i64 %v112, 0
  %v114 = add i64 %v113, 0
  %v115 = getelementptr inbounds float, ptr addrspace(3) @mapal_smem, i64 %v114
  %v116 = load <4 x float>, ptr addrspace(3) %v115, align 16
  %v117 = extractelement <4 x float> %v116, i64 0
  %v118 = extractelement <4 x float> %v116, i64 1
  %v119 = extractelement <4 x float> %v116, i64 2
  %v120 = extractelement <4 x float> %v116, i64 3
  %v121 = mul i64 %v108, 16
  %v122 = add i64 %v121, %v50
  %v123 = add i64 %v122, 256
  %v124 = add i64 %v123, 0
  %v125 = add i64 %v124, 0
  %v126 = getelementptr inbounds float, ptr addrspace(3) @mapal_smem, i64 %v125
  %v127 = load <4 x float>, ptr addrspace(3) %v126, align 16
  %v128 = extractelement <4 x float> %v127, i64 0
  %v129 = extractelement <4 x float> %v127, i64 1
  %v130 = extractelement <4 x float> %v127, i64 2
  %v131 = extractelement <4 x float> %v127, i64 3
  %v132 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 0
  %v133 = load float, ptr %v132, align 4
  %v135 = fmul float %v117, %v128
  %v134 = fadd float %v133, %v135
  store float %v134, ptr %v132, align 4
  %v136 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 1
  %v137 = load float, ptr %v136, align 4
  %v139 = fmul float %v117, %v129
  %v138 = fadd float %v137, %v139
  store float %v138, ptr %v136, align 4
  %v140 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 2
  %v141 = load float, ptr %v140, align 4
  %v143 = fmul float %v117, %v130
  %v142 = fadd float %v141, %v143
  store float %v142, ptr %v140, align 4
  %v144 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 3
  %v145 = load float, ptr %v144, align 4
  %v147 = fmul float %v117, %v131
  %v146 = fadd float %v145, %v147
  store float %v146, ptr %v144, align 4
  %v148 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 4
  %v149 = load float, ptr %v148, align 4
  %v151 = fmul float %v118, %v128
  %v150 = fadd float %v149, %v151
  store float %v150, ptr %v148, align 4
  %v152 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 5
  %v153 = load float, ptr %v152, align 4
  %v155 = fmul float %v118, %v129
  %v154 = fadd float %v153, %v155
  store float %v154, ptr %v152, align 4
  %v156 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 6
  %v157 = load float, ptr %v156, align 4
  %v159 = fmul float %v118, %v130
  %v158 = fadd float %v157, %v159
  store float %v158, ptr %v156, align 4
  %v160 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 7
  %v161 = load float, ptr %v160, align 4
  %v163 = fmul float %v118, %v131
  %v162 = fadd float %v161, %v163
  store float %v162, ptr %v160, align 4
  %v164 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 8
  %v165 = load float, ptr %v164, align 4
  %v167 = fmul float %v119, %v128
  %v166 = fadd float %v165, %v167
  store float %v166, ptr %v164, align 4
  %v168 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 9
  %v169 = load float, ptr %v168, align 4
  %v171 = fmul float %v119, %v129
  %v170 = fadd float %v169, %v171
  store float %v170, ptr %v168, align 4
  %v172 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 10
  %v173 = load float, ptr %v172, align 4
  %v175 = fmul float %v119, %v130
  %v174 = fadd float %v173, %v175
  store float %v174, ptr %v172, align 4
  %v176 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 11
  %v177 = load float, ptr %v176, align 4
  %v179 = fmul float %v119, %v131
  %v178 = fadd float %v177, %v179
  store float %v178, ptr %v176, align 4
  %v180 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 12
  %v181 = load float, ptr %v180, align 4
  %v183 = fmul float %v120, %v128
  %v182 = fadd float %v181, %v183
  store float %v182, ptr %v180, align 4
  %v184 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 13
  %v185 = load float, ptr %v184, align 4
  %v187 = fmul float %v120, %v129
  %v186 = fadd float %v185, %v187
  store float %v186, ptr %v184, align 4
  %v188 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 14
  %v189 = load float, ptr %v188, align 4
  %v191 = fmul float %v120, %v130
  %v190 = fadd float %v189, %v191
  store float %v190, ptr %v188, align 4
  %v192 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 15
  %v193 = load float, ptr %v192, align 4
  %v195 = fmul float %v120, %v131
  %v194 = fadd float %v193, %v195
  store float %v194, ptr %v192, align 4
  %v109 = add i64 %v108, 1
  %v196 = icmp eq i64 %v109, 8
  br i1 %v196, label %ktl, label %kk, !llvm.loop !0
ktl:
  call void @llvm.nvvm.barrier0()
  %v64 = add i64 %v63, 8
  %v197 = icmp eq i64 %v64, 64
  br i1 %v197, label %done, label %kt
done:
  %v198 = add i64 %v9, %v49
  %v199 = add i64 %v198, 0
  %v200 = mul i64 %v199, 64
  %v201 = add i64 %v200, %v10
  %v202 = add i64 %v201, %v50
  %v203 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 0
  %v204 = load float, ptr %v203, align 4
  %v205 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 1
  %v206 = load float, ptr %v205, align 4
  %v207 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 2
  %v208 = load float, ptr %v207, align 4
  %v209 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 3
  %v210 = load float, ptr %v209, align 4
  %v211 = add i64 %v202, 0
  %v212 = trunc i64 %v211 to i32
  %v213 = call float @fn7(float %v204)
  %v214 = add i64 %v211, 1
  %v215 = trunc i64 %v214 to i32
  %v216 = call float @fn7(float %v206)
  %v217 = add i64 %v211, 2
  %v218 = trunc i64 %v217 to i32
  %v219 = call float @fn7(float %v208)
  %v220 = add i64 %v211, 3
  %v221 = trunc i64 %v220 to i32
  %v222 = call float @fn7(float %v210)
  %v223 = insertelement <4 x float> poison, float %v213, i64 0
  %v224 = insertelement <4 x float> %v223, float %v216, i64 1
  %v225 = insertelement <4 x float> %v224, float %v219, i64 2
  %v226 = insertelement <4 x float> %v225, float %v222, i64 3
  %v227 = getelementptr inbounds float, ptr addrspace(1) %y0, i64 %v211
  store <4 x float> %v226, ptr addrspace(1) %v227, align 16
  %v228 = add i64 %v9, %v49
  %v229 = add i64 %v228, 1
  %v230 = mul i64 %v229, 64
  %v231 = add i64 %v230, %v10
  %v232 = add i64 %v231, %v50
  %v233 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 4
  %v234 = load float, ptr %v233, align 4
  %v235 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 5
  %v236 = load float, ptr %v235, align 4
  %v237 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 6
  %v238 = load float, ptr %v237, align 4
  %v239 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 7
  %v240 = load float, ptr %v239, align 4
  %v241 = add i64 %v232, 0
  %v242 = trunc i64 %v241 to i32
  %v243 = call float @fn7(float %v234)
  %v244 = add i64 %v241, 1
  %v245 = trunc i64 %v244 to i32
  %v246 = call float @fn7(float %v236)
  %v247 = add i64 %v241, 2
  %v248 = trunc i64 %v247 to i32
  %v249 = call float @fn7(float %v238)
  %v250 = add i64 %v241, 3
  %v251 = trunc i64 %v250 to i32
  %v252 = call float @fn7(float %v240)
  %v253 = insertelement <4 x float> poison, float %v243, i64 0
  %v254 = insertelement <4 x float> %v253, float %v246, i64 1
  %v255 = insertelement <4 x float> %v254, float %v249, i64 2
  %v256 = insertelement <4 x float> %v255, float %v252, i64 3
  %v257 = getelementptr inbounds float, ptr addrspace(1) %y0, i64 %v241
  store <4 x float> %v256, ptr addrspace(1) %v257, align 16
  %v258 = add i64 %v9, %v49
  %v259 = add i64 %v258, 2
  %v260 = mul i64 %v259, 64
  %v261 = add i64 %v260, %v10
  %v262 = add i64 %v261, %v50
  %v263 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 8
  %v264 = load float, ptr %v263, align 4
  %v265 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 9
  %v266 = load float, ptr %v265, align 4
  %v267 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 10
  %v268 = load float, ptr %v267, align 4
  %v269 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 11
  %v270 = load float, ptr %v269, align 4
  %v271 = add i64 %v262, 0
  %v272 = trunc i64 %v271 to i32
  %v273 = call float @fn7(float %v264)
  %v274 = add i64 %v271, 1
  %v275 = trunc i64 %v274 to i32
  %v276 = call float @fn7(float %v266)
  %v277 = add i64 %v271, 2
  %v278 = trunc i64 %v277 to i32
  %v279 = call float @fn7(float %v268)
  %v280 = add i64 %v271, 3
  %v281 = trunc i64 %v280 to i32
  %v282 = call float @fn7(float %v270)
  %v283 = insertelement <4 x float> poison, float %v273, i64 0
  %v284 = insertelement <4 x float> %v283, float %v276, i64 1
  %v285 = insertelement <4 x float> %v284, float %v279, i64 2
  %v286 = insertelement <4 x float> %v285, float %v282, i64 3
  %v287 = getelementptr inbounds float, ptr addrspace(1) %y0, i64 %v271
  store <4 x float> %v286, ptr addrspace(1) %v287, align 16
  %v288 = add i64 %v9, %v49
  %v289 = add i64 %v288, 3
  %v290 = mul i64 %v289, 64
  %v291 = add i64 %v290, %v10
  %v292 = add i64 %v291, %v50
  %v293 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 12
  %v294 = load float, ptr %v293, align 4
  %v295 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 13
  %v296 = load float, ptr %v295, align 4
  %v297 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 14
  %v298 = load float, ptr %v297, align 4
  %v299 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 15
  %v300 = load float, ptr %v299, align 4
  %v301 = add i64 %v292, 0
  %v302 = trunc i64 %v301 to i32
  %v303 = call float @fn7(float %v294)
  %v304 = add i64 %v301, 1
  %v305 = trunc i64 %v304 to i32
  %v306 = call float @fn7(float %v296)
  %v307 = add i64 %v301, 2
  %v308 = trunc i64 %v307 to i32
  %v309 = call float @fn7(float %v298)
  %v310 = add i64 %v301, 3
  %v311 = trunc i64 %v310 to i32
  %v312 = call float @fn7(float %v300)
  %v313 = insertelement <4 x float> poison, float %v303, i64 0
  %v314 = insertelement <4 x float> %v313, float %v306, i64 1
  %v315 = insertelement <4 x float> %v314, float %v309, i64 2
  %v316 = insertelement <4 x float> %v315, float %v312, i64 3
  %v317 = getelementptr inbounds float, ptr addrspace(1) %y0, i64 %v301
  store <4 x float> %v316, ptr addrspace(1) %v317, align 16
  ret void
}

define ptx_kernel void @mapal_tile_e14de9e486d3b9c5(ptr addrspace(1) noalias readonly align 16 %a, ptr addrspace(1) noalias readonly align 16 %b, ptr addrspace(1) noalias align 16 %o, ptr addrspace(1) noalias align 16 %y0) "nvvm.maxntid"="32" "nvvm.minctasm"="16" {
entry:
  %v1 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %v2 = zext i32 %v1 to i64
  %v3 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %v4 = zext i32 %v3 to i64
  %v5 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  %v6 = zext i32 %v5 to i64
  %v7 = udiv i64 %v2, 4
  %v8 = urem i64 %v2, 4
  %v9 = mul i64 %v6, 32
  %v10 = mul i64 %v4, 16
  %v11 = alloca [16 x float], align 16
  %v12 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 0
  store float 0x0000000000000000, ptr %v12, align 4
  %v13 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 1
  store float 0x0000000000000000, ptr %v13, align 4
  %v14 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 2
  store float 0x0000000000000000, ptr %v14, align 4
  %v15 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 3
  store float 0x0000000000000000, ptr %v15, align 4
  %v16 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 4
  store float 0x0000000000000000, ptr %v16, align 4
  %v17 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 5
  store float 0x0000000000000000, ptr %v17, align 4
  %v18 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 6
  store float 0x0000000000000000, ptr %v18, align 4
  %v19 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 7
  store float 0x0000000000000000, ptr %v19, align 4
  %v20 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 8
  store float 0x0000000000000000, ptr %v20, align 4
  %v21 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 9
  store float 0x0000000000000000, ptr %v21, align 4
  %v22 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 10
  store float 0x0000000000000000, ptr %v22, align 4
  %v23 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 11
  store float 0x0000000000000000, ptr %v23, align 4
  %v24 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 12
  store float 0x0000000000000000, ptr %v24, align 4
  %v25 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 13
  store float 0x0000000000000000, ptr %v25, align 4
  %v26 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 14
  store float 0x0000000000000000, ptr %v26, align 4
  %v27 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 15
  store float 0x0000000000000000, ptr %v27, align 4
  %v28 = udiv i64 %v2, 2
  %v29 = urem i64 %v2, 2
  %v30 = mul i64 %v29, 4
  %v31 = add i64 %v9, %v28
  %v32 = mul i64 %v31, 64
  %v33 = mul i64 %v30, 1
  %v34 = add i64 %v32, %v33
  %v35 = add i64 %v34, 0
  %v36 = mul i64 %v30, 32
  %v37 = add i64 %v36, %v28
  %v38 = udiv i64 %v2, 4
  %v39 = urem i64 %v2, 4
  %v40 = mul i64 %v39, 4
  %v41 = mul i64 %v38, 64
  %v42 = add i64 %v10, %v40
  %v43 = mul i64 %v42, 1
  %v44 = add i64 %v41, %v43
  %v45 = add i64 %v44, 0
  %v46 = mul i64 %v38, 16
  %v47 = add i64 %v46, %v40
  %v48 = add i64 %v47, 256
  %v49 = mul i64 %v7, 4
  %v50 = mul i64 %v8, 4
  %v51 = mul i64 0, 1
  %v52 = mul i64 0, 64
  %v53 = add i64 %v35, %v51
  %v54 = getelementptr inbounds float, ptr addrspace(1) %a, i64 %v53
  %v55 = load <4 x float>, ptr addrspace(1) %v54, align 16
  %v56 = add i64 %v35, %v51
  %v57 = add i64 %v56, 1024
  %v58 = getelementptr inbounds float, ptr addrspace(1) %a, i64 %v57
  %v59 = load <4 x float>, ptr addrspace(1) %v58, align 16
  %v60 = add i64 %v45, %v52
  %v61 = getelementptr inbounds float, ptr addrspace(1) %b, i64 %v60
  %v62 = load <4 x float>, ptr addrspace(1) %v61, align 16
  br label %kt
kt:
  %v63 = phi i64 [0, %entry], [%v64, %ktl]
  %v65 = phi <4 x float> [%v55, %entry], [%v100, %ktl]
  %v66 = phi <4 x float> [%v59, %entry], [%v104, %ktl]
  %v67 = phi <4 x float> [%v62, %entry], [%v107, %ktl]
  %v68 = extractelement <4 x float> %v65, i64 0
  %v69 = add i64 %v37, 0
  %v70 = getelementptr inbounds float, ptr addrspace(3) @mapal_smem, i64 %v69
  store float %v68, ptr addrspace(3) %v70, align 4
  %v71 = extractelement <4 x float> %v65, i64 1
  %v72 = add i64 %v37, 32
  %v73 = getelementptr inbounds float, ptr addrspace(3) @mapal_smem, i64 %v72
  store float %v71, ptr addrspace(3) %v73, align 4
  %v74 = extractelement <4 x float> %v65, i64 2
  %v75 = add i64 %v37, 64
  %v76 = getelementptr inbounds float, ptr addrspace(3) @mapal_smem, i64 %v75
  store float %v74, ptr addrspace(3) %v76, align 4
  %v77 = extractelement <4 x float> %v65, i64 3
  %v78 = add i64 %v37, 96
  %v79 = getelementptr inbounds float, ptr addrspace(3) @mapal_smem, i64 %v78
  store float %v77, ptr addrspace(3) %v79, align 4
  %v80 = extractelement <4 x float> %v66, i64 0
  %v81 = add i64 %v37, 16
  %v82 = getelementptr inbounds float, ptr addrspace(3) @mapal_smem, i64 %v81
  store float %v80, ptr addrspace(3) %v82, align 4
  %v83 = extractelement <4 x float> %v66, i64 1
  %v84 = add i64 %v37, 48
  %v85 = getelementptr inbounds float, ptr addrspace(3) @mapal_smem, i64 %v84
  store float %v83, ptr addrspace(3) %v85, align 4
  %v86 = extractelement <4 x float> %v66, i64 2
  %v87 = add i64 %v37, 80
  %v88 = getelementptr inbounds float, ptr addrspace(3) @mapal_smem, i64 %v87
  store float %v86, ptr addrspace(3) %v88, align 4
  %v89 = extractelement <4 x float> %v66, i64 3
  %v90 = add i64 %v37, 112
  %v91 = getelementptr inbounds float, ptr addrspace(3) @mapal_smem, i64 %v90
  store float %v89, ptr addrspace(3) %v91, align 4
  %v92 = getelementptr inbounds float, ptr addrspace(3) @mapal_smem, i64 %v48
  store <4 x float> %v67, ptr addrspace(3) %v92, align 16
  call void @llvm.nvvm.barrier0()
  %v93 = add i64 %v63, 8
  %v94 = icmp ult i64 %v93, 64
  %v95 = select i1 %v94, i64 %v93, i64 %v63
  %v96 = mul i64 %v95, 1
  %v97 = mul i64 %v95, 64
  %v98 = add i64 %v35, %v96
  %v99 = getelementptr inbounds float, ptr addrspace(1) %a, i64 %v98
  %v100 = load <4 x float>, ptr addrspace(1) %v99, align 16
  %v101 = add i64 %v35, %v96
  %v102 = add i64 %v101, 1024
  %v103 = getelementptr inbounds float, ptr addrspace(1) %a, i64 %v102
  %v104 = load <4 x float>, ptr addrspace(1) %v103, align 16
  %v105 = add i64 %v45, %v97
  %v106 = getelementptr inbounds float, ptr addrspace(1) %b, i64 %v105
  %v107 = load <4 x float>, ptr addrspace(1) %v106, align 16
  br label %kk
kk:
  %v108 = phi i64 [0, %kt], [%v109, %kk]
  %v110 = mul i64 %v108, 32
  %v111 = add i64 %v110, %v49
  %v112 = add i64 %v111, 0
  %v113 = add i64 %v112, 0
  %v114 = add i64 %v113, 0
  %v115 = getelementptr inbounds float, ptr addrspace(3) @mapal_smem, i64 %v114
  %v116 = load <4 x float>, ptr addrspace(3) %v115, align 16
  %v117 = extractelement <4 x float> %v116, i64 0
  %v118 = extractelement <4 x float> %v116, i64 1
  %v119 = extractelement <4 x float> %v116, i64 2
  %v120 = extractelement <4 x float> %v116, i64 3
  %v121 = mul i64 %v108, 16
  %v122 = add i64 %v121, %v50
  %v123 = add i64 %v122, 256
  %v124 = add i64 %v123, 0
  %v125 = add i64 %v124, 0
  %v126 = getelementptr inbounds float, ptr addrspace(3) @mapal_smem, i64 %v125
  %v127 = load <4 x float>, ptr addrspace(3) %v126, align 16
  %v128 = extractelement <4 x float> %v127, i64 0
  %v129 = extractelement <4 x float> %v127, i64 1
  %v130 = extractelement <4 x float> %v127, i64 2
  %v131 = extractelement <4 x float> %v127, i64 3
  %v132 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 0
  %v133 = load float, ptr %v132, align 4
  %v135 = fmul float %v117, %v128
  %v134 = fadd float %v133, %v135
  store float %v134, ptr %v132, align 4
  %v136 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 1
  %v137 = load float, ptr %v136, align 4
  %v139 = fmul float %v117, %v129
  %v138 = fadd float %v137, %v139
  store float %v138, ptr %v136, align 4
  %v140 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 2
  %v141 = load float, ptr %v140, align 4
  %v143 = fmul float %v117, %v130
  %v142 = fadd float %v141, %v143
  store float %v142, ptr %v140, align 4
  %v144 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 3
  %v145 = load float, ptr %v144, align 4
  %v147 = fmul float %v117, %v131
  %v146 = fadd float %v145, %v147
  store float %v146, ptr %v144, align 4
  %v148 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 4
  %v149 = load float, ptr %v148, align 4
  %v151 = fmul float %v118, %v128
  %v150 = fadd float %v149, %v151
  store float %v150, ptr %v148, align 4
  %v152 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 5
  %v153 = load float, ptr %v152, align 4
  %v155 = fmul float %v118, %v129
  %v154 = fadd float %v153, %v155
  store float %v154, ptr %v152, align 4
  %v156 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 6
  %v157 = load float, ptr %v156, align 4
  %v159 = fmul float %v118, %v130
  %v158 = fadd float %v157, %v159
  store float %v158, ptr %v156, align 4
  %v160 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 7
  %v161 = load float, ptr %v160, align 4
  %v163 = fmul float %v118, %v131
  %v162 = fadd float %v161, %v163
  store float %v162, ptr %v160, align 4
  %v164 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 8
  %v165 = load float, ptr %v164, align 4
  %v167 = fmul float %v119, %v128
  %v166 = fadd float %v165, %v167
  store float %v166, ptr %v164, align 4
  %v168 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 9
  %v169 = load float, ptr %v168, align 4
  %v171 = fmul float %v119, %v129
  %v170 = fadd float %v169, %v171
  store float %v170, ptr %v168, align 4
  %v172 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 10
  %v173 = load float, ptr %v172, align 4
  %v175 = fmul float %v119, %v130
  %v174 = fadd float %v173, %v175
  store float %v174, ptr %v172, align 4
  %v176 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 11
  %v177 = load float, ptr %v176, align 4
  %v179 = fmul float %v119, %v131
  %v178 = fadd float %v177, %v179
  store float %v178, ptr %v176, align 4
  %v180 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 12
  %v181 = load float, ptr %v180, align 4
  %v183 = fmul float %v120, %v128
  %v182 = fadd float %v181, %v183
  store float %v182, ptr %v180, align 4
  %v184 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 13
  %v185 = load float, ptr %v184, align 4
  %v187 = fmul float %v120, %v129
  %v186 = fadd float %v185, %v187
  store float %v186, ptr %v184, align 4
  %v188 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 14
  %v189 = load float, ptr %v188, align 4
  %v191 = fmul float %v120, %v130
  %v190 = fadd float %v189, %v191
  store float %v190, ptr %v188, align 4
  %v192 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 15
  %v193 = load float, ptr %v192, align 4
  %v195 = fmul float %v120, %v131
  %v194 = fadd float %v193, %v195
  store float %v194, ptr %v192, align 4
  %v109 = add i64 %v108, 1
  %v196 = icmp eq i64 %v109, 8
  br i1 %v196, label %ktl, label %kk, !llvm.loop !2
ktl:
  call void @llvm.nvvm.barrier0()
  %v64 = add i64 %v63, 8
  %v197 = icmp eq i64 %v64, 64
  br i1 %v197, label %done, label %kt
done:
  %v198 = add i64 %v9, %v49
  %v199 = add i64 %v198, 0
  %v200 = mul i64 %v199, 64
  %v201 = add i64 %v200, %v10
  %v202 = add i64 %v201, %v50
  %v203 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 0
  %v204 = load float, ptr %v203, align 4
  %v205 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 1
  %v206 = load float, ptr %v205, align 4
  %v207 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 2
  %v208 = load float, ptr %v207, align 4
  %v209 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 3
  %v210 = load float, ptr %v209, align 4
  %v211 = add i64 %v202, 0
  %v212 = trunc i64 %v211 to i32
  %v213 = call float @fn10(float %v204)
  %v214 = add i64 %v211, 1
  %v215 = trunc i64 %v214 to i32
  %v216 = call float @fn10(float %v206)
  %v217 = add i64 %v211, 2
  %v218 = trunc i64 %v217 to i32
  %v219 = call float @fn10(float %v208)
  %v220 = add i64 %v211, 3
  %v221 = trunc i64 %v220 to i32
  %v222 = call float @fn10(float %v210)
  %v223 = insertelement <4 x float> poison, float %v213, i64 0
  %v224 = insertelement <4 x float> %v223, float %v216, i64 1
  %v225 = insertelement <4 x float> %v224, float %v219, i64 2
  %v226 = insertelement <4 x float> %v225, float %v222, i64 3
  %v227 = getelementptr inbounds float, ptr addrspace(1) %y0, i64 %v211
  store <4 x float> %v226, ptr addrspace(1) %v227, align 16
  %v228 = add i64 %v9, %v49
  %v229 = add i64 %v228, 1
  %v230 = mul i64 %v229, 64
  %v231 = add i64 %v230, %v10
  %v232 = add i64 %v231, %v50
  %v233 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 4
  %v234 = load float, ptr %v233, align 4
  %v235 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 5
  %v236 = load float, ptr %v235, align 4
  %v237 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 6
  %v238 = load float, ptr %v237, align 4
  %v239 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 7
  %v240 = load float, ptr %v239, align 4
  %v241 = add i64 %v232, 0
  %v242 = trunc i64 %v241 to i32
  %v243 = call float @fn10(float %v234)
  %v244 = add i64 %v241, 1
  %v245 = trunc i64 %v244 to i32
  %v246 = call float @fn10(float %v236)
  %v247 = add i64 %v241, 2
  %v248 = trunc i64 %v247 to i32
  %v249 = call float @fn10(float %v238)
  %v250 = add i64 %v241, 3
  %v251 = trunc i64 %v250 to i32
  %v252 = call float @fn10(float %v240)
  %v253 = insertelement <4 x float> poison, float %v243, i64 0
  %v254 = insertelement <4 x float> %v253, float %v246, i64 1
  %v255 = insertelement <4 x float> %v254, float %v249, i64 2
  %v256 = insertelement <4 x float> %v255, float %v252, i64 3
  %v257 = getelementptr inbounds float, ptr addrspace(1) %y0, i64 %v241
  store <4 x float> %v256, ptr addrspace(1) %v257, align 16
  %v258 = add i64 %v9, %v49
  %v259 = add i64 %v258, 2
  %v260 = mul i64 %v259, 64
  %v261 = add i64 %v260, %v10
  %v262 = add i64 %v261, %v50
  %v263 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 8
  %v264 = load float, ptr %v263, align 4
  %v265 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 9
  %v266 = load float, ptr %v265, align 4
  %v267 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 10
  %v268 = load float, ptr %v267, align 4
  %v269 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 11
  %v270 = load float, ptr %v269, align 4
  %v271 = add i64 %v262, 0
  %v272 = trunc i64 %v271 to i32
  %v273 = call float @fn10(float %v264)
  %v274 = add i64 %v271, 1
  %v275 = trunc i64 %v274 to i32
  %v276 = call float @fn10(float %v266)
  %v277 = add i64 %v271, 2
  %v278 = trunc i64 %v277 to i32
  %v279 = call float @fn10(float %v268)
  %v280 = add i64 %v271, 3
  %v281 = trunc i64 %v280 to i32
  %v282 = call float @fn10(float %v270)
  %v283 = insertelement <4 x float> poison, float %v273, i64 0
  %v284 = insertelement <4 x float> %v283, float %v276, i64 1
  %v285 = insertelement <4 x float> %v284, float %v279, i64 2
  %v286 = insertelement <4 x float> %v285, float %v282, i64 3
  %v287 = getelementptr inbounds float, ptr addrspace(1) %y0, i64 %v271
  store <4 x float> %v286, ptr addrspace(1) %v287, align 16
  %v288 = add i64 %v9, %v49
  %v289 = add i64 %v288, 3
  %v290 = mul i64 %v289, 64
  %v291 = add i64 %v290, %v10
  %v292 = add i64 %v291, %v50
  %v293 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 12
  %v294 = load float, ptr %v293, align 4
  %v295 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 13
  %v296 = load float, ptr %v295, align 4
  %v297 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 14
  %v298 = load float, ptr %v297, align 4
  %v299 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 15
  %v300 = load float, ptr %v299, align 4
  %v301 = add i64 %v292, 0
  %v302 = trunc i64 %v301 to i32
  %v303 = call float @fn10(float %v294)
  %v304 = add i64 %v301, 1
  %v305 = trunc i64 %v304 to i32
  %v306 = call float @fn10(float %v296)
  %v307 = add i64 %v301, 2
  %v308 = trunc i64 %v307 to i32
  %v309 = call float @fn10(float %v298)
  %v310 = add i64 %v301, 3
  %v311 = trunc i64 %v310 to i32
  %v312 = call float @fn10(float %v300)
  %v313 = insertelement <4 x float> poison, float %v303, i64 0
  %v314 = insertelement <4 x float> %v313, float %v306, i64 1
  %v315 = insertelement <4 x float> %v314, float %v309, i64 2
  %v316 = insertelement <4 x float> %v315, float %v312, i64 3
  %v317 = getelementptr inbounds float, ptr addrspace(1) %y0, i64 %v301
  store <4 x float> %v316, ptr addrspace(1) %v317, align 16
  ret void
}

define internal float @fn1(i32 %arg) readonly nounwind willreturn {
entry:
  %o0 = alloca i32
  %o1 = alloca float
  %o2 = alloca { i32, i32 }
  %o3 = alloca i32
  %o4 = alloca { i32, i32 }
  %o5 = alloca i32
  store i32 %arg, ptr %o0
  %t0 = load i32, ptr %o0
  %t1 = getelementptr { i32, i32 }, ptr %o2, i32 0, i32 0
  store i32 %t0, ptr %t1
  %t2 = getelementptr { i32, i32 }, ptr %o2, i32 0, i32 1
  store i32 7, ptr %t2
  %t3 = getelementptr { i32, i32 }, ptr %o2, i32 0, i32 0
  %t4 = load i32, ptr %t3
  %t5 = getelementptr { i32, i32 }, ptr %o2, i32 0, i32 1
  %t6 = load i32, ptr %t5
  %t7 = srem i32 %t4, %t6
  store i32 %t7, ptr %o3
  %t8 = load i32, ptr %o3
  %t9 = getelementptr { i32, i32 }, ptr %o4, i32 0, i32 0
  store i32 %t8, ptr %t9
  %t10 = getelementptr { i32, i32 }, ptr %o4, i32 0, i32 1
  store i32 1, ptr %t10
  %t11 = getelementptr { i32, i32 }, ptr %o4, i32 0, i32 0
  %t12 = load i32, ptr %t11
  %t13 = getelementptr { i32, i32 }, ptr %o4, i32 0, i32 1
  %t14 = load i32, ptr %t13
  %t15 = add i32 %t12, %t14
  store i32 %t15, ptr %o5
  %t16 = load i32, ptr %o5
  %t17 = sitofp i32 %t16 to float
  store float %t17, ptr %o1
  %t18 = load float, ptr %o1
  ret float %t18
}

define ptx_kernel void @mapal_gen_1f19f0178349a042(ptr addrspace(1) %o, i64 %n) "nvvm.maxntid"="256" {
entry:
  %t = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %b = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %t64 = zext i32 %t to i64
  %b64 = zext i32 %b to i64
  %base = mul i64 %b64, 256
  %i = add i64 %base, %t64
  %in = icmp ult i64 %i, %n
  br i1 %in, label %body, label %done
body:
  %x = trunc i64 %i to i32
  %v = call float @fn1(i32 %x)
  %p = getelementptr inbounds float, ptr addrspace(1) %o, i64 %i
  store float %v, ptr addrspace(1) %p
  br label %done
done:
  ret void
}

define internal float @fn2(i32 %arg) readonly nounwind willreturn {
entry:
  %o0 = alloca i32
  %o1 = alloca float
  %o2 = alloca { i32, i32 }
  %o3 = alloca i32
  %o4 = alloca { i32, i32 }
  %o5 = alloca i32
  store i32 %arg, ptr %o0
  %t0 = load i32, ptr %o0
  %t1 = getelementptr { i32, i32 }, ptr %o2, i32 0, i32 0
  store i32 %t0, ptr %t1
  %t2 = getelementptr { i32, i32 }, ptr %o2, i32 0, i32 1
  store i32 5, ptr %t2
  %t3 = getelementptr { i32, i32 }, ptr %o2, i32 0, i32 0
  %t4 = load i32, ptr %t3
  %t5 = getelementptr { i32, i32 }, ptr %o2, i32 0, i32 1
  %t6 = load i32, ptr %t5
  %t7 = srem i32 %t4, %t6
  store i32 %t7, ptr %o3
  %t8 = load i32, ptr %o3
  %t9 = getelementptr { i32, i32 }, ptr %o4, i32 0, i32 0
  store i32 %t8, ptr %t9
  %t10 = getelementptr { i32, i32 }, ptr %o4, i32 0, i32 1
  store i32 2, ptr %t10
  %t11 = getelementptr { i32, i32 }, ptr %o4, i32 0, i32 0
  %t12 = load i32, ptr %t11
  %t13 = getelementptr { i32, i32 }, ptr %o4, i32 0, i32 1
  %t14 = load i32, ptr %t13
  %t15 = add i32 %t12, %t14
  store i32 %t15, ptr %o5
  %t16 = load i32, ptr %o5
  %t17 = sitofp i32 %t16 to float
  store float %t17, ptr %o1
  %t18 = load float, ptr %o1
  ret float %t18
}

define ptx_kernel void @mapal_gen_8c8c3bdb9c78e77f(ptr addrspace(1) %o, i64 %n) "nvvm.maxntid"="256" {
entry:
  %t = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %b = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %t64 = zext i32 %t to i64
  %b64 = zext i32 %b to i64
  %base = mul i64 %b64, 256
  %i = add i64 %base, %t64
  %in = icmp ult i64 %i, %n
  br i1 %in, label %body, label %done
body:
  %x = trunc i64 %i to i32
  %v = call float @fn2(i32 %x)
  %p = getelementptr inbounds float, ptr addrspace(1) %o, i64 %i
  store float %v, ptr addrspace(1) %p
  br label %done
done:
  ret void
}

define internal float @fn3(i32 %arg) readonly nounwind willreturn {
entry:
  %o0 = alloca i32
  %o1 = alloca float
  %o2 = alloca { i32, i32 }
  %o3 = alloca i32
  %o4 = alloca { i32, i32 }
  %o5 = alloca i32
  store i32 %arg, ptr %o0
  %t0 = load i32, ptr %o0
  %t1 = getelementptr { i32, i32 }, ptr %o2, i32 0, i32 0
  store i32 %t0, ptr %t1
  %t2 = getelementptr { i32, i32 }, ptr %o2, i32 0, i32 1
  store i32 11, ptr %t2
  %t3 = getelementptr { i32, i32 }, ptr %o2, i32 0, i32 0
  %t4 = load i32, ptr %t3
  %t5 = getelementptr { i32, i32 }, ptr %o2, i32 0, i32 1
  %t6 = load i32, ptr %t5
  %t7 = srem i32 %t4, %t6
  store i32 %t7, ptr %o3
  %t8 = load i32, ptr %o3
  %t9 = getelementptr { i32, i32 }, ptr %o4, i32 0, i32 0
  store i32 %t8, ptr %t9
  %t10 = getelementptr { i32, i32 }, ptr %o4, i32 0, i32 1
  store i32 1, ptr %t10
  %t11 = getelementptr { i32, i32 }, ptr %o4, i32 0, i32 0
  %t12 = load i32, ptr %t11
  %t13 = getelementptr { i32, i32 }, ptr %o4, i32 0, i32 1
  %t14 = load i32, ptr %t13
  %t15 = add i32 %t12, %t14
  store i32 %t15, ptr %o5
  %t16 = load i32, ptr %o5
  %t17 = sitofp i32 %t16 to float
  store float %t17, ptr %o1
  %t18 = load float, ptr %o1
  ret float %t18
}

define ptx_kernel void @mapal_gen_2f0666f212630a9c(ptr addrspace(1) %o, i64 %n) "nvvm.maxntid"="256" {
entry:
  %t = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %b = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %t64 = zext i32 %t to i64
  %b64 = zext i32 %b to i64
  %base = mul i64 %b64, 256
  %i = add i64 %base, %t64
  %in = icmp ult i64 %i, %n
  br i1 %in, label %body, label %done
body:
  %x = trunc i64 %i to i32
  %v = call float @fn3(i32 %x)
  %p = getelementptr inbounds float, ptr addrspace(1) %o, i64 %i
  store float %v, ptr addrspace(1) %p
  br label %done
done:
  ret void
}

define internal float @fn4(i32 %arg) readonly nounwind willreturn {
entry:
  %o0 = alloca i32
  %o1 = alloca float
  %o2 = alloca { i32, i32 }
  %o3 = alloca i32
  %o4 = alloca { i32, i32 }
  %o5 = alloca i32
  store i32 %arg, ptr %o0
  %t0 = load i32, ptr %o0
  %t1 = getelementptr { i32, i32 }, ptr %o2, i32 0, i32 0
  store i32 %t0, ptr %t1
  %t2 = getelementptr { i32, i32 }, ptr %o2, i32 0, i32 1
  store i32 13, ptr %t2
  %t3 = getelementptr { i32, i32 }, ptr %o2, i32 0, i32 0
  %t4 = load i32, ptr %t3
  %t5 = getelementptr { i32, i32 }, ptr %o2, i32 0, i32 1
  %t6 = load i32, ptr %t5
  %t7 = srem i32 %t4, %t6
  store i32 %t7, ptr %o3
  %t8 = load i32, ptr %o3
  %t9 = getelementptr { i32, i32 }, ptr %o4, i32 0, i32 0
  store i32 %t8, ptr %t9
  %t10 = getelementptr { i32, i32 }, ptr %o4, i32 0, i32 1
  store i32 1, ptr %t10
  %t11 = getelementptr { i32, i32 }, ptr %o4, i32 0, i32 0
  %t12 = load i32, ptr %t11
  %t13 = getelementptr { i32, i32 }, ptr %o4, i32 0, i32 1
  %t14 = load i32, ptr %t13
  %t15 = add i32 %t12, %t14
  store i32 %t15, ptr %o5
  %t16 = load i32, ptr %o5
  %t17 = sitofp i32 %t16 to float
  store float %t17, ptr %o1
  %t18 = load float, ptr %o1
  ret float %t18
}

define ptx_kernel void @mapal_gen_967dd4abc0e6e301(ptr addrspace(1) %o, i64 %n) "nvvm.maxntid"="256" {
entry:
  %t = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %b = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %t64 = zext i32 %t to i64
  %b64 = zext i32 %b to i64
  %base = mul i64 %b64, 256
  %i = add i64 %base, %t64
  %in = icmp ult i64 %i, %n
  br i1 %in, label %body, label %done
body:
  %x = trunc i64 %i to i32
  %v = call float @fn4(i32 %x)
  %p = getelementptr inbounds float, ptr addrspace(1) %o, i64 %i
  store float %v, ptr addrspace(1) %p
  br label %done
done:
  ret void
}

define internal float @fn7(float %arg) readonly nounwind willreturn {
entry:
  %o0 = alloca float
  %o1 = alloca float
  %o2 = alloca { float, float }
  store float %arg, ptr %o0
  %t0 = load float, ptr %o0
  %t1 = getelementptr { float, float }, ptr %o2, i32 0, i32 0
  store float %t0, ptr %t1
  %t2 = getelementptr { float, float }, ptr %o2, i32 0, i32 1
  store float 0x4000000000000000, ptr %t2
  %t3 = getelementptr { float, float }, ptr %o2, i32 0, i32 0
  %t4 = load float, ptr %t3
  %t5 = getelementptr { float, float }, ptr %o2, i32 0, i32 1
  %t6 = load float, ptr %t5
  %t7 = fmul float %t4, %t6
  store float %t7, ptr %o1
  %t8 = load float, ptr %o1
  ret float %t8
}

define internal float @fn10(float %arg) readonly nounwind willreturn {
entry:
  %o0 = alloca float
  %o1 = alloca float
  %o2 = alloca { float, float }
  store float %arg, ptr %o0
  %t0 = load float, ptr %o0
  %t1 = getelementptr { float, float }, ptr %o2, i32 0, i32 0
  store float %t0, ptr %t1
  %t2 = getelementptr { float, float }, ptr %o2, i32 0, i32 1
  store float 0x4000000000000000, ptr %t2
  %t3 = getelementptr { float, float }, ptr %o2, i32 0, i32 0
  %t4 = load float, ptr %t3
  %t5 = getelementptr { float, float }, ptr %o2, i32 0, i32 1
  %t6 = load float, ptr %t5
  %t7 = fmul float %t4, %t6
  store float %t7, ptr %o1
  %t8 = load float, ptr %o1
  ret float %t8
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.full"}
!2 = distinct !{!2, !3}
!3 = !{!"llvm.loop.unroll.full"}
