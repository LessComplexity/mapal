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

define ptx_kernel void @mapal_tile_52995b7688b5dc7c(ptr addrspace(1) noalias readonly align 16 %a, ptr addrspace(1) noalias readonly align 16 %b, ptr addrspace(1) noalias align 16 %o, ptr addrspace(1) noalias align 16 %y0) "nvvm.maxntid"="32" "nvvm.minctasm"="16" {
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
  %v28 = add i64 %v2, 0
  %v29 = udiv i64 %v28, 2
  %v30 = urem i64 %v28, 2
  %v31 = mul i64 %v30, 4
  %v32 = add i64 %v9, %v29
  %v33 = mul i64 %v32, 64
  %v34 = mul i64 %v31, 1
  %v35 = add i64 %v33, %v34
  %v36 = add i64 %v35, 0
  %v37 = mul i64 %v31, 36
  %v38 = add i64 %v37, %v29
  %v39 = add i64 %v2, 32
  %v40 = udiv i64 %v39, 2
  %v41 = urem i64 %v39, 2
  %v42 = mul i64 %v41, 4
  %v43 = add i64 %v9, %v40
  %v44 = mul i64 %v43, 64
  %v45 = mul i64 %v42, 1
  %v46 = add i64 %v44, %v45
  %v47 = add i64 %v46, 0
  %v48 = mul i64 %v42, 36
  %v49 = add i64 %v48, %v40
  %v50 = add i64 %v2, 0
  %v51 = udiv i64 %v50, 4
  %v52 = urem i64 %v50, 4
  %v53 = mul i64 %v52, 4
  %v54 = mul i64 %v51, 64
  %v55 = add i64 %v10, %v53
  %v56 = mul i64 %v55, 1
  %v57 = add i64 %v54, %v56
  %v58 = add i64 %v57, 0
  %v59 = mul i64 %v51, 16
  %v60 = add i64 %v59, %v53
  %v61 = add i64 %v60, 288
  %v62 = mul i64 %v7, 4
  %v63 = mul i64 %v8, 4
  %v64 = mul i64 0, 1
  %v65 = mul i64 0, 64
  %v66 = add i64 %v36, %v64
  %v67 = getelementptr inbounds float, ptr addrspace(1) %a, i64 %v66
  %v68 = load <4 x float>, ptr addrspace(1) %v67, align 16
  %v69 = add i64 %v47, %v64
  %v70 = getelementptr inbounds float, ptr addrspace(1) %a, i64 %v69
  %v71 = load <4 x float>, ptr addrspace(1) %v70, align 16
  %v72 = add i64 %v58, %v65
  %v73 = getelementptr inbounds float, ptr addrspace(1) %b, i64 %v72
  %v74 = load <4 x float>, ptr addrspace(1) %v73, align 16
  br label %kt
kt:
  %v75 = phi i64 [0, %entry], [%v76, %ktl]
  %v77 = phi <4 x float> [%v68, %entry], [%v112, %ktl]
  %v78 = phi <4 x float> [%v71, %entry], [%v115, %ktl]
  %v79 = phi <4 x float> [%v74, %entry], [%v118, %ktl]
  %v80 = extractelement <4 x float> %v77, i64 0
  %v81 = add i64 %v38, 0
  %v82 = getelementptr inbounds float, ptr addrspace(3) @mapal_smem, i64 %v81
  store float %v80, ptr addrspace(3) %v82, align 4
  %v83 = extractelement <4 x float> %v77, i64 1
  %v84 = add i64 %v38, 36
  %v85 = getelementptr inbounds float, ptr addrspace(3) @mapal_smem, i64 %v84
  store float %v83, ptr addrspace(3) %v85, align 4
  %v86 = extractelement <4 x float> %v77, i64 2
  %v87 = add i64 %v38, 72
  %v88 = getelementptr inbounds float, ptr addrspace(3) @mapal_smem, i64 %v87
  store float %v86, ptr addrspace(3) %v88, align 4
  %v89 = extractelement <4 x float> %v77, i64 3
  %v90 = add i64 %v38, 108
  %v91 = getelementptr inbounds float, ptr addrspace(3) @mapal_smem, i64 %v90
  store float %v89, ptr addrspace(3) %v91, align 4
  %v92 = extractelement <4 x float> %v78, i64 0
  %v93 = add i64 %v49, 0
  %v94 = getelementptr inbounds float, ptr addrspace(3) @mapal_smem, i64 %v93
  store float %v92, ptr addrspace(3) %v94, align 4
  %v95 = extractelement <4 x float> %v78, i64 1
  %v96 = add i64 %v49, 36
  %v97 = getelementptr inbounds float, ptr addrspace(3) @mapal_smem, i64 %v96
  store float %v95, ptr addrspace(3) %v97, align 4
  %v98 = extractelement <4 x float> %v78, i64 2
  %v99 = add i64 %v49, 72
  %v100 = getelementptr inbounds float, ptr addrspace(3) @mapal_smem, i64 %v99
  store float %v98, ptr addrspace(3) %v100, align 4
  %v101 = extractelement <4 x float> %v78, i64 3
  %v102 = add i64 %v49, 108
  %v103 = getelementptr inbounds float, ptr addrspace(3) @mapal_smem, i64 %v102
  store float %v101, ptr addrspace(3) %v103, align 4
  %v104 = getelementptr inbounds float, ptr addrspace(3) @mapal_smem, i64 %v61
  store <4 x float> %v79, ptr addrspace(3) %v104, align 16
  call void @llvm.nvvm.barrier0()
  %v105 = add i64 %v75, 8
  %v106 = icmp ult i64 %v105, 64
  %v107 = select i1 %v106, i64 %v105, i64 %v75
  %v108 = mul i64 %v107, 1
  %v109 = mul i64 %v107, 64
  %v110 = add i64 %v36, %v108
  %v111 = getelementptr inbounds float, ptr addrspace(1) %a, i64 %v110
  %v112 = load <4 x float>, ptr addrspace(1) %v111, align 16
  %v113 = add i64 %v47, %v108
  %v114 = getelementptr inbounds float, ptr addrspace(1) %a, i64 %v113
  %v115 = load <4 x float>, ptr addrspace(1) %v114, align 16
  %v116 = add i64 %v58, %v109
  %v117 = getelementptr inbounds float, ptr addrspace(1) %b, i64 %v116
  %v118 = load <4 x float>, ptr addrspace(1) %v117, align 16
  br label %kk
kk:
  %v119 = phi i64 [0, %kt], [%v120, %kk]
  %v121 = mul i64 %v119, 36
  %v122 = add i64 %v121, %v62
  %v123 = add i64 %v122, 0
  %v124 = add i64 %v123, 0
  %v125 = add i64 %v124, 0
  %v126 = getelementptr inbounds float, ptr addrspace(3) @mapal_smem, i64 %v125
  %v127 = load <4 x float>, ptr addrspace(3) %v126, align 16
  %v128 = extractelement <4 x float> %v127, i64 0
  %v129 = extractelement <4 x float> %v127, i64 1
  %v130 = extractelement <4 x float> %v127, i64 2
  %v131 = extractelement <4 x float> %v127, i64 3
  %v132 = mul i64 %v119, 16
  %v133 = add i64 %v132, %v63
  %v134 = add i64 %v133, 288
  %v135 = add i64 %v134, 0
  %v136 = add i64 %v135, 0
  %v137 = getelementptr inbounds float, ptr addrspace(3) @mapal_smem, i64 %v136
  %v138 = load <4 x float>, ptr addrspace(3) %v137, align 16
  %v139 = extractelement <4 x float> %v138, i64 0
  %v140 = extractelement <4 x float> %v138, i64 1
  %v141 = extractelement <4 x float> %v138, i64 2
  %v142 = extractelement <4 x float> %v138, i64 3
  %v143 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 0
  %v144 = load float, ptr %v143, align 4
  %v145 = call float @llvm.fma.f32(float %v128, float %v139, float %v144)
  store float %v145, ptr %v143, align 4
  %v146 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 1
  %v147 = load float, ptr %v146, align 4
  %v148 = call float @llvm.fma.f32(float %v128, float %v140, float %v147)
  store float %v148, ptr %v146, align 4
  %v149 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 2
  %v150 = load float, ptr %v149, align 4
  %v151 = call float @llvm.fma.f32(float %v128, float %v141, float %v150)
  store float %v151, ptr %v149, align 4
  %v152 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 3
  %v153 = load float, ptr %v152, align 4
  %v154 = call float @llvm.fma.f32(float %v128, float %v142, float %v153)
  store float %v154, ptr %v152, align 4
  %v155 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 4
  %v156 = load float, ptr %v155, align 4
  %v157 = call float @llvm.fma.f32(float %v129, float %v139, float %v156)
  store float %v157, ptr %v155, align 4
  %v158 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 5
  %v159 = load float, ptr %v158, align 4
  %v160 = call float @llvm.fma.f32(float %v129, float %v140, float %v159)
  store float %v160, ptr %v158, align 4
  %v161 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 6
  %v162 = load float, ptr %v161, align 4
  %v163 = call float @llvm.fma.f32(float %v129, float %v141, float %v162)
  store float %v163, ptr %v161, align 4
  %v164 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 7
  %v165 = load float, ptr %v164, align 4
  %v166 = call float @llvm.fma.f32(float %v129, float %v142, float %v165)
  store float %v166, ptr %v164, align 4
  %v167 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 8
  %v168 = load float, ptr %v167, align 4
  %v169 = call float @llvm.fma.f32(float %v130, float %v139, float %v168)
  store float %v169, ptr %v167, align 4
  %v170 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 9
  %v171 = load float, ptr %v170, align 4
  %v172 = call float @llvm.fma.f32(float %v130, float %v140, float %v171)
  store float %v172, ptr %v170, align 4
  %v173 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 10
  %v174 = load float, ptr %v173, align 4
  %v175 = call float @llvm.fma.f32(float %v130, float %v141, float %v174)
  store float %v175, ptr %v173, align 4
  %v176 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 11
  %v177 = load float, ptr %v176, align 4
  %v178 = call float @llvm.fma.f32(float %v130, float %v142, float %v177)
  store float %v178, ptr %v176, align 4
  %v179 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 12
  %v180 = load float, ptr %v179, align 4
  %v181 = call float @llvm.fma.f32(float %v131, float %v139, float %v180)
  store float %v181, ptr %v179, align 4
  %v182 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 13
  %v183 = load float, ptr %v182, align 4
  %v184 = call float @llvm.fma.f32(float %v131, float %v140, float %v183)
  store float %v184, ptr %v182, align 4
  %v185 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 14
  %v186 = load float, ptr %v185, align 4
  %v187 = call float @llvm.fma.f32(float %v131, float %v141, float %v186)
  store float %v187, ptr %v185, align 4
  %v188 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 15
  %v189 = load float, ptr %v188, align 4
  %v190 = call float @llvm.fma.f32(float %v131, float %v142, float %v189)
  store float %v190, ptr %v188, align 4
  %v120 = add i64 %v119, 1
  %v191 = icmp eq i64 %v120, 8
  br i1 %v191, label %ktl, label %kk, !llvm.loop !0
ktl:
  call void @llvm.nvvm.barrier0()
  %v76 = add i64 %v75, 8
  %v192 = icmp eq i64 %v76, 64
  br i1 %v192, label %done, label %kt
done:
  %v193 = add i64 %v9, %v62
  %v194 = add i64 %v193, 0
  %v195 = mul i64 %v194, 64
  %v196 = add i64 %v195, %v10
  %v197 = add i64 %v196, %v63
  %v198 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 0
  %v199 = load float, ptr %v198, align 4
  %v200 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 1
  %v201 = load float, ptr %v200, align 4
  %v202 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 2
  %v203 = load float, ptr %v202, align 4
  %v204 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 3
  %v205 = load float, ptr %v204, align 4
  %v206 = add i64 %v197, 0
  %v207 = trunc i64 %v206 to i32
  %v208 = call float @fn10(float %v199)
  %v209 = add i64 %v206, 1
  %v210 = trunc i64 %v209 to i32
  %v211 = call float @fn10(float %v201)
  %v212 = add i64 %v206, 2
  %v213 = trunc i64 %v212 to i32
  %v214 = call float @fn10(float %v203)
  %v215 = add i64 %v206, 3
  %v216 = trunc i64 %v215 to i32
  %v217 = call float @fn10(float %v205)
  %v218 = insertelement <4 x float> poison, float %v208, i64 0
  %v219 = insertelement <4 x float> %v218, float %v211, i64 1
  %v220 = insertelement <4 x float> %v219, float %v214, i64 2
  %v221 = insertelement <4 x float> %v220, float %v217, i64 3
  %v222 = getelementptr inbounds float, ptr addrspace(1) %y0, i64 %v206
  store <4 x float> %v221, ptr addrspace(1) %v222, align 16
  %v223 = add i64 %v9, %v62
  %v224 = add i64 %v223, 1
  %v225 = mul i64 %v224, 64
  %v226 = add i64 %v225, %v10
  %v227 = add i64 %v226, %v63
  %v228 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 4
  %v229 = load float, ptr %v228, align 4
  %v230 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 5
  %v231 = load float, ptr %v230, align 4
  %v232 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 6
  %v233 = load float, ptr %v232, align 4
  %v234 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 7
  %v235 = load float, ptr %v234, align 4
  %v236 = add i64 %v227, 0
  %v237 = trunc i64 %v236 to i32
  %v238 = call float @fn10(float %v229)
  %v239 = add i64 %v236, 1
  %v240 = trunc i64 %v239 to i32
  %v241 = call float @fn10(float %v231)
  %v242 = add i64 %v236, 2
  %v243 = trunc i64 %v242 to i32
  %v244 = call float @fn10(float %v233)
  %v245 = add i64 %v236, 3
  %v246 = trunc i64 %v245 to i32
  %v247 = call float @fn10(float %v235)
  %v248 = insertelement <4 x float> poison, float %v238, i64 0
  %v249 = insertelement <4 x float> %v248, float %v241, i64 1
  %v250 = insertelement <4 x float> %v249, float %v244, i64 2
  %v251 = insertelement <4 x float> %v250, float %v247, i64 3
  %v252 = getelementptr inbounds float, ptr addrspace(1) %y0, i64 %v236
  store <4 x float> %v251, ptr addrspace(1) %v252, align 16
  %v253 = add i64 %v9, %v62
  %v254 = add i64 %v253, 2
  %v255 = mul i64 %v254, 64
  %v256 = add i64 %v255, %v10
  %v257 = add i64 %v256, %v63
  %v258 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 8
  %v259 = load float, ptr %v258, align 4
  %v260 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 9
  %v261 = load float, ptr %v260, align 4
  %v262 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 10
  %v263 = load float, ptr %v262, align 4
  %v264 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 11
  %v265 = load float, ptr %v264, align 4
  %v266 = add i64 %v257, 0
  %v267 = trunc i64 %v266 to i32
  %v268 = call float @fn10(float %v259)
  %v269 = add i64 %v266, 1
  %v270 = trunc i64 %v269 to i32
  %v271 = call float @fn10(float %v261)
  %v272 = add i64 %v266, 2
  %v273 = trunc i64 %v272 to i32
  %v274 = call float @fn10(float %v263)
  %v275 = add i64 %v266, 3
  %v276 = trunc i64 %v275 to i32
  %v277 = call float @fn10(float %v265)
  %v278 = insertelement <4 x float> poison, float %v268, i64 0
  %v279 = insertelement <4 x float> %v278, float %v271, i64 1
  %v280 = insertelement <4 x float> %v279, float %v274, i64 2
  %v281 = insertelement <4 x float> %v280, float %v277, i64 3
  %v282 = getelementptr inbounds float, ptr addrspace(1) %y0, i64 %v266
  store <4 x float> %v281, ptr addrspace(1) %v282, align 16
  %v283 = add i64 %v9, %v62
  %v284 = add i64 %v283, 3
  %v285 = mul i64 %v284, 64
  %v286 = add i64 %v285, %v10
  %v287 = add i64 %v286, %v63
  %v288 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 12
  %v289 = load float, ptr %v288, align 4
  %v290 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 13
  %v291 = load float, ptr %v290, align 4
  %v292 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 14
  %v293 = load float, ptr %v292, align 4
  %v294 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 15
  %v295 = load float, ptr %v294, align 4
  %v296 = add i64 %v287, 0
  %v297 = trunc i64 %v296 to i32
  %v298 = call float @fn10(float %v289)
  %v299 = add i64 %v296, 1
  %v300 = trunc i64 %v299 to i32
  %v301 = call float @fn10(float %v291)
  %v302 = add i64 %v296, 2
  %v303 = trunc i64 %v302 to i32
  %v304 = call float @fn10(float %v293)
  %v305 = add i64 %v296, 3
  %v306 = trunc i64 %v305 to i32
  %v307 = call float @fn10(float %v295)
  %v308 = insertelement <4 x float> poison, float %v298, i64 0
  %v309 = insertelement <4 x float> %v308, float %v301, i64 1
  %v310 = insertelement <4 x float> %v309, float %v304, i64 2
  %v311 = insertelement <4 x float> %v310, float %v307, i64 3
  %v312 = getelementptr inbounds float, ptr addrspace(1) %y0, i64 %v296
  store <4 x float> %v311, ptr addrspace(1) %v312, align 16
  ret void
}

define ptx_kernel void @mapal_tile_614aadf92d26cb8a(ptr addrspace(1) noalias readonly align 16 %a, ptr addrspace(1) noalias readonly align 16 %b, ptr addrspace(1) noalias align 16 %o, ptr addrspace(1) noalias align 16 %y0) "nvvm.maxntid"="32" "nvvm.minctasm"="16" {
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
  %v28 = add i64 %v2, 0
  %v29 = udiv i64 %v28, 2
  %v30 = urem i64 %v28, 2
  %v31 = mul i64 %v30, 4
  %v32 = add i64 %v9, %v29
  %v33 = mul i64 %v32, 64
  %v34 = mul i64 %v31, 1
  %v35 = add i64 %v33, %v34
  %v36 = add i64 %v35, 0
  %v37 = mul i64 %v31, 36
  %v38 = add i64 %v37, %v29
  %v39 = add i64 %v2, 32
  %v40 = udiv i64 %v39, 2
  %v41 = urem i64 %v39, 2
  %v42 = mul i64 %v41, 4
  %v43 = add i64 %v9, %v40
  %v44 = mul i64 %v43, 64
  %v45 = mul i64 %v42, 1
  %v46 = add i64 %v44, %v45
  %v47 = add i64 %v46, 0
  %v48 = mul i64 %v42, 36
  %v49 = add i64 %v48, %v40
  %v50 = add i64 %v2, 0
  %v51 = udiv i64 %v50, 4
  %v52 = urem i64 %v50, 4
  %v53 = mul i64 %v52, 4
  %v54 = mul i64 %v51, 64
  %v55 = add i64 %v10, %v53
  %v56 = mul i64 %v55, 1
  %v57 = add i64 %v54, %v56
  %v58 = add i64 %v57, 0
  %v59 = mul i64 %v51, 16
  %v60 = add i64 %v59, %v53
  %v61 = add i64 %v60, 288
  %v62 = mul i64 %v7, 4
  %v63 = mul i64 %v8, 4
  %v64 = mul i64 0, 1
  %v65 = mul i64 0, 64
  %v66 = add i64 %v36, %v64
  %v67 = getelementptr inbounds float, ptr addrspace(1) %a, i64 %v66
  %v68 = load <4 x float>, ptr addrspace(1) %v67, align 16
  %v69 = add i64 %v47, %v64
  %v70 = getelementptr inbounds float, ptr addrspace(1) %a, i64 %v69
  %v71 = load <4 x float>, ptr addrspace(1) %v70, align 16
  %v72 = add i64 %v58, %v65
  %v73 = getelementptr inbounds float, ptr addrspace(1) %b, i64 %v72
  %v74 = load <4 x float>, ptr addrspace(1) %v73, align 16
  br label %kt
kt:
  %v75 = phi i64 [0, %entry], [%v76, %ktl]
  %v77 = phi <4 x float> [%v68, %entry], [%v112, %ktl]
  %v78 = phi <4 x float> [%v71, %entry], [%v115, %ktl]
  %v79 = phi <4 x float> [%v74, %entry], [%v118, %ktl]
  %v80 = extractelement <4 x float> %v77, i64 0
  %v81 = add i64 %v38, 0
  %v82 = getelementptr inbounds float, ptr addrspace(3) @mapal_smem, i64 %v81
  store float %v80, ptr addrspace(3) %v82, align 4
  %v83 = extractelement <4 x float> %v77, i64 1
  %v84 = add i64 %v38, 36
  %v85 = getelementptr inbounds float, ptr addrspace(3) @mapal_smem, i64 %v84
  store float %v83, ptr addrspace(3) %v85, align 4
  %v86 = extractelement <4 x float> %v77, i64 2
  %v87 = add i64 %v38, 72
  %v88 = getelementptr inbounds float, ptr addrspace(3) @mapal_smem, i64 %v87
  store float %v86, ptr addrspace(3) %v88, align 4
  %v89 = extractelement <4 x float> %v77, i64 3
  %v90 = add i64 %v38, 108
  %v91 = getelementptr inbounds float, ptr addrspace(3) @mapal_smem, i64 %v90
  store float %v89, ptr addrspace(3) %v91, align 4
  %v92 = extractelement <4 x float> %v78, i64 0
  %v93 = add i64 %v49, 0
  %v94 = getelementptr inbounds float, ptr addrspace(3) @mapal_smem, i64 %v93
  store float %v92, ptr addrspace(3) %v94, align 4
  %v95 = extractelement <4 x float> %v78, i64 1
  %v96 = add i64 %v49, 36
  %v97 = getelementptr inbounds float, ptr addrspace(3) @mapal_smem, i64 %v96
  store float %v95, ptr addrspace(3) %v97, align 4
  %v98 = extractelement <4 x float> %v78, i64 2
  %v99 = add i64 %v49, 72
  %v100 = getelementptr inbounds float, ptr addrspace(3) @mapal_smem, i64 %v99
  store float %v98, ptr addrspace(3) %v100, align 4
  %v101 = extractelement <4 x float> %v78, i64 3
  %v102 = add i64 %v49, 108
  %v103 = getelementptr inbounds float, ptr addrspace(3) @mapal_smem, i64 %v102
  store float %v101, ptr addrspace(3) %v103, align 4
  %v104 = getelementptr inbounds float, ptr addrspace(3) @mapal_smem, i64 %v61
  store <4 x float> %v79, ptr addrspace(3) %v104, align 16
  call void @llvm.nvvm.barrier0()
  %v105 = add i64 %v75, 8
  %v106 = icmp ult i64 %v105, 64
  %v107 = select i1 %v106, i64 %v105, i64 %v75
  %v108 = mul i64 %v107, 1
  %v109 = mul i64 %v107, 64
  %v110 = add i64 %v36, %v108
  %v111 = getelementptr inbounds float, ptr addrspace(1) %a, i64 %v110
  %v112 = load <4 x float>, ptr addrspace(1) %v111, align 16
  %v113 = add i64 %v47, %v108
  %v114 = getelementptr inbounds float, ptr addrspace(1) %a, i64 %v113
  %v115 = load <4 x float>, ptr addrspace(1) %v114, align 16
  %v116 = add i64 %v58, %v109
  %v117 = getelementptr inbounds float, ptr addrspace(1) %b, i64 %v116
  %v118 = load <4 x float>, ptr addrspace(1) %v117, align 16
  br label %kk
kk:
  %v119 = phi i64 [0, %kt], [%v120, %kk]
  %v121 = mul i64 %v119, 36
  %v122 = add i64 %v121, %v62
  %v123 = add i64 %v122, 0
  %v124 = add i64 %v123, 0
  %v125 = add i64 %v124, 0
  %v126 = getelementptr inbounds float, ptr addrspace(3) @mapal_smem, i64 %v125
  %v127 = load <4 x float>, ptr addrspace(3) %v126, align 16
  %v128 = extractelement <4 x float> %v127, i64 0
  %v129 = extractelement <4 x float> %v127, i64 1
  %v130 = extractelement <4 x float> %v127, i64 2
  %v131 = extractelement <4 x float> %v127, i64 3
  %v132 = mul i64 %v119, 16
  %v133 = add i64 %v132, %v63
  %v134 = add i64 %v133, 288
  %v135 = add i64 %v134, 0
  %v136 = add i64 %v135, 0
  %v137 = getelementptr inbounds float, ptr addrspace(3) @mapal_smem, i64 %v136
  %v138 = load <4 x float>, ptr addrspace(3) %v137, align 16
  %v139 = extractelement <4 x float> %v138, i64 0
  %v140 = extractelement <4 x float> %v138, i64 1
  %v141 = extractelement <4 x float> %v138, i64 2
  %v142 = extractelement <4 x float> %v138, i64 3
  %v143 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 0
  %v144 = load float, ptr %v143, align 4
  %v145 = call float @llvm.fma.f32(float %v128, float %v139, float %v144)
  store float %v145, ptr %v143, align 4
  %v146 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 1
  %v147 = load float, ptr %v146, align 4
  %v148 = call float @llvm.fma.f32(float %v128, float %v140, float %v147)
  store float %v148, ptr %v146, align 4
  %v149 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 2
  %v150 = load float, ptr %v149, align 4
  %v151 = call float @llvm.fma.f32(float %v128, float %v141, float %v150)
  store float %v151, ptr %v149, align 4
  %v152 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 3
  %v153 = load float, ptr %v152, align 4
  %v154 = call float @llvm.fma.f32(float %v128, float %v142, float %v153)
  store float %v154, ptr %v152, align 4
  %v155 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 4
  %v156 = load float, ptr %v155, align 4
  %v157 = call float @llvm.fma.f32(float %v129, float %v139, float %v156)
  store float %v157, ptr %v155, align 4
  %v158 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 5
  %v159 = load float, ptr %v158, align 4
  %v160 = call float @llvm.fma.f32(float %v129, float %v140, float %v159)
  store float %v160, ptr %v158, align 4
  %v161 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 6
  %v162 = load float, ptr %v161, align 4
  %v163 = call float @llvm.fma.f32(float %v129, float %v141, float %v162)
  store float %v163, ptr %v161, align 4
  %v164 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 7
  %v165 = load float, ptr %v164, align 4
  %v166 = call float @llvm.fma.f32(float %v129, float %v142, float %v165)
  store float %v166, ptr %v164, align 4
  %v167 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 8
  %v168 = load float, ptr %v167, align 4
  %v169 = call float @llvm.fma.f32(float %v130, float %v139, float %v168)
  store float %v169, ptr %v167, align 4
  %v170 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 9
  %v171 = load float, ptr %v170, align 4
  %v172 = call float @llvm.fma.f32(float %v130, float %v140, float %v171)
  store float %v172, ptr %v170, align 4
  %v173 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 10
  %v174 = load float, ptr %v173, align 4
  %v175 = call float @llvm.fma.f32(float %v130, float %v141, float %v174)
  store float %v175, ptr %v173, align 4
  %v176 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 11
  %v177 = load float, ptr %v176, align 4
  %v178 = call float @llvm.fma.f32(float %v130, float %v142, float %v177)
  store float %v178, ptr %v176, align 4
  %v179 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 12
  %v180 = load float, ptr %v179, align 4
  %v181 = call float @llvm.fma.f32(float %v131, float %v139, float %v180)
  store float %v181, ptr %v179, align 4
  %v182 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 13
  %v183 = load float, ptr %v182, align 4
  %v184 = call float @llvm.fma.f32(float %v131, float %v140, float %v183)
  store float %v184, ptr %v182, align 4
  %v185 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 14
  %v186 = load float, ptr %v185, align 4
  %v187 = call float @llvm.fma.f32(float %v131, float %v141, float %v186)
  store float %v187, ptr %v185, align 4
  %v188 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 15
  %v189 = load float, ptr %v188, align 4
  %v190 = call float @llvm.fma.f32(float %v131, float %v142, float %v189)
  store float %v190, ptr %v188, align 4
  %v120 = add i64 %v119, 1
  %v191 = icmp eq i64 %v120, 8
  br i1 %v191, label %ktl, label %kk, !llvm.loop !2
ktl:
  call void @llvm.nvvm.barrier0()
  %v76 = add i64 %v75, 8
  %v192 = icmp eq i64 %v76, 64
  br i1 %v192, label %done, label %kt
done:
  %v193 = add i64 %v9, %v62
  %v194 = add i64 %v193, 0
  %v195 = mul i64 %v194, 64
  %v196 = add i64 %v195, %v10
  %v197 = add i64 %v196, %v63
  %v198 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 0
  %v199 = load float, ptr %v198, align 4
  %v200 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 1
  %v201 = load float, ptr %v200, align 4
  %v202 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 2
  %v203 = load float, ptr %v202, align 4
  %v204 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 3
  %v205 = load float, ptr %v204, align 4
  %v206 = add i64 %v197, 0
  %v207 = trunc i64 %v206 to i32
  %v208 = call float @fn7(float %v199)
  %v209 = add i64 %v206, 1
  %v210 = trunc i64 %v209 to i32
  %v211 = call float @fn7(float %v201)
  %v212 = add i64 %v206, 2
  %v213 = trunc i64 %v212 to i32
  %v214 = call float @fn7(float %v203)
  %v215 = add i64 %v206, 3
  %v216 = trunc i64 %v215 to i32
  %v217 = call float @fn7(float %v205)
  %v218 = insertelement <4 x float> poison, float %v208, i64 0
  %v219 = insertelement <4 x float> %v218, float %v211, i64 1
  %v220 = insertelement <4 x float> %v219, float %v214, i64 2
  %v221 = insertelement <4 x float> %v220, float %v217, i64 3
  %v222 = getelementptr inbounds float, ptr addrspace(1) %y0, i64 %v206
  store <4 x float> %v221, ptr addrspace(1) %v222, align 16
  %v223 = add i64 %v9, %v62
  %v224 = add i64 %v223, 1
  %v225 = mul i64 %v224, 64
  %v226 = add i64 %v225, %v10
  %v227 = add i64 %v226, %v63
  %v228 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 4
  %v229 = load float, ptr %v228, align 4
  %v230 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 5
  %v231 = load float, ptr %v230, align 4
  %v232 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 6
  %v233 = load float, ptr %v232, align 4
  %v234 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 7
  %v235 = load float, ptr %v234, align 4
  %v236 = add i64 %v227, 0
  %v237 = trunc i64 %v236 to i32
  %v238 = call float @fn7(float %v229)
  %v239 = add i64 %v236, 1
  %v240 = trunc i64 %v239 to i32
  %v241 = call float @fn7(float %v231)
  %v242 = add i64 %v236, 2
  %v243 = trunc i64 %v242 to i32
  %v244 = call float @fn7(float %v233)
  %v245 = add i64 %v236, 3
  %v246 = trunc i64 %v245 to i32
  %v247 = call float @fn7(float %v235)
  %v248 = insertelement <4 x float> poison, float %v238, i64 0
  %v249 = insertelement <4 x float> %v248, float %v241, i64 1
  %v250 = insertelement <4 x float> %v249, float %v244, i64 2
  %v251 = insertelement <4 x float> %v250, float %v247, i64 3
  %v252 = getelementptr inbounds float, ptr addrspace(1) %y0, i64 %v236
  store <4 x float> %v251, ptr addrspace(1) %v252, align 16
  %v253 = add i64 %v9, %v62
  %v254 = add i64 %v253, 2
  %v255 = mul i64 %v254, 64
  %v256 = add i64 %v255, %v10
  %v257 = add i64 %v256, %v63
  %v258 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 8
  %v259 = load float, ptr %v258, align 4
  %v260 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 9
  %v261 = load float, ptr %v260, align 4
  %v262 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 10
  %v263 = load float, ptr %v262, align 4
  %v264 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 11
  %v265 = load float, ptr %v264, align 4
  %v266 = add i64 %v257, 0
  %v267 = trunc i64 %v266 to i32
  %v268 = call float @fn7(float %v259)
  %v269 = add i64 %v266, 1
  %v270 = trunc i64 %v269 to i32
  %v271 = call float @fn7(float %v261)
  %v272 = add i64 %v266, 2
  %v273 = trunc i64 %v272 to i32
  %v274 = call float @fn7(float %v263)
  %v275 = add i64 %v266, 3
  %v276 = trunc i64 %v275 to i32
  %v277 = call float @fn7(float %v265)
  %v278 = insertelement <4 x float> poison, float %v268, i64 0
  %v279 = insertelement <4 x float> %v278, float %v271, i64 1
  %v280 = insertelement <4 x float> %v279, float %v274, i64 2
  %v281 = insertelement <4 x float> %v280, float %v277, i64 3
  %v282 = getelementptr inbounds float, ptr addrspace(1) %y0, i64 %v266
  store <4 x float> %v281, ptr addrspace(1) %v282, align 16
  %v283 = add i64 %v9, %v62
  %v284 = add i64 %v283, 3
  %v285 = mul i64 %v284, 64
  %v286 = add i64 %v285, %v10
  %v287 = add i64 %v286, %v63
  %v288 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 12
  %v289 = load float, ptr %v288, align 4
  %v290 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 13
  %v291 = load float, ptr %v290, align 4
  %v292 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 14
  %v293 = load float, ptr %v292, align 4
  %v294 = getelementptr inbounds [16 x float], ptr %v11, i64 0, i64 15
  %v295 = load float, ptr %v294, align 4
  %v296 = add i64 %v287, 0
  %v297 = trunc i64 %v296 to i32
  %v298 = call float @fn7(float %v289)
  %v299 = add i64 %v296, 1
  %v300 = trunc i64 %v299 to i32
  %v301 = call float @fn7(float %v291)
  %v302 = add i64 %v296, 2
  %v303 = trunc i64 %v302 to i32
  %v304 = call float @fn7(float %v293)
  %v305 = add i64 %v296, 3
  %v306 = trunc i64 %v305 to i32
  %v307 = call float @fn7(float %v295)
  %v308 = insertelement <4 x float> poison, float %v298, i64 0
  %v309 = insertelement <4 x float> %v308, float %v301, i64 1
  %v310 = insertelement <4 x float> %v309, float %v304, i64 2
  %v311 = insertelement <4 x float> %v310, float %v307, i64 3
  %v312 = getelementptr inbounds float, ptr addrspace(1) %y0, i64 %v296
  store <4 x float> %v311, ptr addrspace(1) %v312, align 16
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
