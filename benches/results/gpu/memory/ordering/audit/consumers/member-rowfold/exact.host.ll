; mapal-backend-llvm emitted module
declare void @mapal_print_i32(i32, i1 zeroext)
declare void @mapal_print_i64(i64, i1 zeroext)
declare void @mapal_print_u8(i8 zeroext, i1 zeroext)
declare void @mapal_print_bool(i1 zeroext, i1 zeroext)
declare void @mapal_print_f32(float, i1 zeroext)
declare void @mapal_print_f64(double, i1 zeroext)
declare void @mapal_print_str(ptr, i64, i1 zeroext)
declare double @mapal_time_ms()
declare void @mapal_trap(i32) noreturn
declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)
declare void @mapal_gpu_init(i32)
declare i64 @mapal_gpu_alloc(ptr, i64)
declare void @mapal_gpu_h2d(i64, ptr, i64)
declare void @mapal_gpu_d2h(ptr, i64, i64, i64)
declare void @mapal_gpu_launch(i64, i64, i64, i64, i32, i32, i32, i32)
declare void @mapal_gpu_gen(i64, i64, i64)
declare void @mapal_gpu_free(ptr)
declare void @mapal_gpu_d2h_on(i32, ptr, i64, i64, i64)
declare void @mapal_gpu_launchv(i64, i32, i32, ptr, i32, i32, i32, i32)

define internal void @mapal_main() {
entry:
  %o2 = alloca [4096 x i32]
  %o3 = alloca [4096 x float]
  %o4 = alloca [4096 x float]
  %o5 = alloca [4096 x float]
  %o6 = alloca [4096 x float]
  %o7 = alloca [64 x i32]
  %o8 = alloca { ptr, ptr, ptr, ptr }
  %o9 = alloca [4096 x float]
  %o10 = alloca [4096 x float]
  %o11 = alloca { ptr, ptr, ptr, ptr }
  %o12 = alloca [4096 x float]
  %o13 = alloca [4096 x float]
  %o14 = alloca { ptr, ptr, ptr }
  %o15 = alloca [64 x float]
  %o16 = alloca { ptr, i32 }
  %o17 = alloca float
  %o18 = alloca float
  %o20 = alloca { ptr, i32 }
  %o21 = alloca float
  %o22 = alloca float
  %o24 = alloca { ptr, i32 }
  %o25 = alloca float
  %o26 = alloca float
  %o28 = alloca { ptr, i32 }
  %o29 = alloca float
  %o30 = alloca float
  %o32 = alloca { ptr, i32 }
  %o33 = alloca float
  %o34 = alloca float
  %o36 = alloca { ptr, i32 }
  %o37 = alloca float
  %o38 = alloca float
  %o40 = alloca { ptr, i32 }
  %o41 = alloca float
  %o42 = alloca float
  %o44 = alloca { ptr, i32 }
  %o45 = alloca float
  %o46 = alloca float
  %o48 = alloca { ptr, i32 }
  %o49 = alloca float
  %o50 = alloca float
  %o52 = alloca { ptr, i32 }
  %o53 = alloca float
  %o54 = alloca float
  %o56 = alloca { ptr, i32 }
  %o57 = alloca float
  %o58 = alloca float
  %o60 = alloca { ptr, i32 }
  %o61 = alloca float
  %o62 = alloca float
  %o64 = alloca { ptr, i32 }
  %o65 = alloca float
  %o66 = alloca float
  %o68 = alloca { ptr, i32 }
  %o69 = alloca float
  %o70 = alloca float
  %d0 = alloca i64
  store i64 0, ptr %d0
  %d2 = alloca i64
  store i64 0, ptr %d2
  %d4 = alloca i64
  store i64 0, ptr %d4
  %d6 = alloca i64
  store i64 0, ptr %d6
  %s10 = alloca i64
  %d33 = alloca i64
  store i64 0, ptr %d33
  %d35 = alloca i64
  store i64 0, ptr %d35
  %d38 = alloca i64
  store i64 0, ptr %d38
  %p40 = alloca [4 x i64]
  %d53 = alloca i64
  store i64 0, ptr %d53
  %p55 = alloca [4 x i64]
  %s64 = alloca i64
  %s71 = alloca { ptr, ptr, i32 }
  %t1 = call i64 @mapal_gpu_alloc(ptr %d0, i64 16384)
  call void @mapal_gpu_gen(i64 2241086273371086914, i64 %t1, i64 4096)
  %t3 = call i64 @mapal_gpu_alloc(ptr %d2, i64 16384)
  call void @mapal_gpu_gen(i64 -8319208597251102849, i64 %t3, i64 4096)
  %t5 = call i64 @mapal_gpu_alloc(ptr %d4, i64 16384)
  call void @mapal_gpu_gen(i64 3388508959519476380, i64 %t5, i64 4096)
  %t7 = call i64 @mapal_gpu_alloc(ptr %d6, i64 16384)
  call void @mapal_gpu_gen(i64 -7602686761790676223, i64 %t7, i64 4096)
  %t8 = getelementptr { ptr, ptr, ptr, ptr }, ptr %o8, i32 0, i32 3
  store ptr %o2, ptr %t8
  %t9 = getelementptr { ptr, ptr, ptr, ptr }, ptr %o11, i32 0, i32 3
  store ptr %o2, ptr %t9
  store i64 0, ptr %s10
  br label %bb11
bb11:
  %t14 = load i64, ptr %s10
  %t15 = icmp uge i64 %t14, 64
  br i1 %t15, label %bb13, label %bb12
bb12:
  %t16 = trunc i64 %t14 to i32
  %t17 = getelementptr [64 x i32], ptr %o7, i64 0, i64 %t14
  store i32 %t16, ptr %t17
  %t18 = add i64 %t14, 1
  store i64 %t18, ptr %s10
  br label %bb11
bb13:
  %t19 = getelementptr { ptr, ptr, ptr, ptr }, ptr %o8, i32 0, i32 0
  store ptr %o7, ptr %t19
  %t20 = getelementptr { ptr, ptr, ptr, ptr }, ptr %o11, i32 0, i32 0
  store ptr %o7, ptr %t20
  %t21 = getelementptr { ptr, ptr, ptr }, ptr %o14, i32 0, i32 0
  store ptr %o7, ptr %t21
  %t22 = getelementptr { ptr, ptr, ptr }, ptr %o14, i32 0, i32 2
  store ptr %o7, ptr %t22
  %t23 = getelementptr { ptr, ptr, ptr, ptr }, ptr %o8, i32 0, i32 1
  store ptr %o3, ptr %t23
  %t24 = getelementptr { ptr, i32 }, ptr %o40, i32 0, i32 0
  store ptr %o3, ptr %t24
  %t25 = getelementptr { ptr, ptr, ptr, ptr }, ptr %o8, i32 0, i32 2
  store ptr %o4, ptr %t25
  %t26 = getelementptr { ptr, i32 }, ptr %o44, i32 0, i32 0
  store ptr %o4, ptr %t26
  %t27 = getelementptr { ptr, ptr, ptr, ptr }, ptr %o11, i32 0, i32 1
  store ptr %o5, ptr %t27
  %t28 = getelementptr { ptr, i32 }, ptr %o48, i32 0, i32 0
  store ptr %o5, ptr %t28
  %t29 = getelementptr { ptr, ptr, ptr, ptr }, ptr %o11, i32 0, i32 2
  store ptr %o6, ptr %t29
  %t30 = getelementptr { ptr, i32 }, ptr %o52, i32 0, i32 0
  store ptr %o6, ptr %t30
  %t31 = load i64, ptr %d0
  %t32 = load i64, ptr %d2
  %t34 = call i64 @mapal_gpu_alloc(ptr %d33, i64 16384)
  %t36 = call i64 @mapal_gpu_alloc(ptr %d35, i64 16384)
  %t37 = call i64 @mapal_gpu_alloc(ptr %d33, i64 16384)
  %t39 = call i64 @mapal_gpu_alloc(ptr %d38, i64 16384)
  %t41 = getelementptr [4 x i64], ptr %p40, i64 0, i64 0
  store i64 %t31, ptr %t41
  %t42 = getelementptr [4 x i64], ptr %p40, i64 0, i64 1
  store i64 %t32, ptr %t42
  %t43 = getelementptr [4 x i64], ptr %p40, i64 0, i64 2
  store i64 %t37, ptr %t43
  %t44 = getelementptr [4 x i64], ptr %p40, i64 0, i64 3
  store i64 %t39, ptr %t44
  call void @mapal_gpu_launchv(i64 -5910159045450748452, i32 1, i32 4, ptr %p40, i32 4, i32 2, i32 32, i32 1536)
  call void @mapal_gpu_d2h_on(i32 1, ptr %o10, i64 %t39, i64 0, i64 16384)
  %t45 = getelementptr { ptr, i32 }, ptr %o56, i32 0, i32 0
  store ptr %o9, ptr %t45
  %t46 = getelementptr { ptr, i32 }, ptr %o60, i32 0, i32 0
  store ptr %o9, ptr %t46
  %t47 = getelementptr { ptr, ptr, ptr }, ptr %o14, i32 0, i32 1
  store ptr %o10, ptr %t47
  %t48 = getelementptr { ptr, i32 }, ptr %o32, i32 0, i32 0
  store ptr %o10, ptr %t48
  %t49 = getelementptr { ptr, i32 }, ptr %o36, i32 0, i32 0
  store ptr %o10, ptr %t49
  %t50 = load i64, ptr %d4
  %t51 = load i64, ptr %d6
  %t52 = call i64 @mapal_gpu_alloc(ptr %d35, i64 16384)
  %t54 = call i64 @mapal_gpu_alloc(ptr %d53, i64 16384)
  %t56 = getelementptr [4 x i64], ptr %p55, i64 0, i64 0
  store i64 %t50, ptr %t56
  %t57 = getelementptr [4 x i64], ptr %p55, i64 0, i64 1
  store i64 %t51, ptr %t57
  %t58 = getelementptr [4 x i64], ptr %p55, i64 0, i64 2
  store i64 %t52, ptr %t58
  %t59 = getelementptr [4 x i64], ptr %p55, i64 0, i64 3
  store i64 %t54, ptr %t59
  call void @mapal_gpu_launchv(i64 8546537189269670622, i32 2, i32 4, ptr %p55, i32 4, i32 2, i32 32, i32 1536)
  %t60 = getelementptr { ptr, i32 }, ptr %o64, i32 0, i32 0
  store ptr %o12, ptr %t60
  %t61 = getelementptr { ptr, i32 }, ptr %o68, i32 0, i32 0
  store ptr %o12, ptr %t61
  %t62 = getelementptr { ptr, i32 }, ptr %o24, i32 0, i32 0
  store ptr %o13, ptr %t62
  %t63 = getelementptr { ptr, i32 }, ptr %o28, i32 0, i32 0
  store ptr %o13, ptr %t63
  store i64 0, ptr %s64
  br label %bb65
bb65:
  %t68 = load i64, ptr %s64
  %t69 = icmp uge i64 %t68, 64
  br i1 %t69, label %bb67, label %bb66
bb66:
  %t70 = trunc i64 %t68 to i32
  %t72 = getelementptr { ptr, ptr, i32 }, ptr %s71, i32 0, i32 0
  store ptr %o7, ptr %t72
  %t73 = getelementptr { ptr, ptr, i32 }, ptr %s71, i32 0, i32 1
  store ptr %o10, ptr %t73
  %t74 = getelementptr { ptr, ptr, i32 }, ptr %s71, i32 0, i32 2
  store i32 %t70, ptr %t74
  %t75 = load { ptr, ptr, i32 }, ptr %s71
  %t76 = call float @fn12({ ptr, ptr, i32 } %t75)
  %t77 = getelementptr [64 x float], ptr %o15, i64 0, i64 %t68
  store float %t76, ptr %t77
  %t78 = add i64 %t68, 1
  store i64 %t78, ptr %s64
  br label %bb65
bb67:
  %t79 = getelementptr { ptr, i32 }, ptr %o16, i32 0, i32 0
  store ptr %o15, ptr %t79
  %t80 = getelementptr { ptr, i32 }, ptr %o20, i32 0, i32 0
  store ptr %o15, ptr %t80
  %t81 = getelementptr { ptr, i32 }, ptr %o16, i32 0, i32 1
  store i32 0, ptr %t81
  %t82 = getelementptr { ptr, i32 }, ptr %o16, i32 0, i32 1
  %t83 = load i32, ptr %t82
  %t84 = sext i32 %t83 to i64
  %t85 = getelementptr [64 x float], ptr %o15, i64 0, i64 %t84
  %t86 = load float, ptr %t85
  store float %t86, ptr %o17
  %t87 = load float, ptr %o17
  store float %t87, ptr %o18
  %t88 = load float, ptr %o18
  call void @mapal_print_f32(float %t88, i1 zeroext true)
  %t89 = getelementptr { ptr, i32 }, ptr %o20, i32 0, i32 1
  store i32 63, ptr %t89
  %t90 = getelementptr { ptr, i32 }, ptr %o20, i32 0, i32 1
  %t91 = load i32, ptr %t90
  %t92 = sext i32 %t91 to i64
  %t93 = getelementptr [64 x float], ptr %o15, i64 0, i64 %t92
  %t94 = load float, ptr %t93
  store float %t94, ptr %o21
  %t95 = load float, ptr %o21
  store float %t95, ptr %o22
  %t96 = load float, ptr %o22
  call void @mapal_print_f32(float %t96, i1 zeroext true)
  %t97 = getelementptr { ptr, i32 }, ptr %o24, i32 0, i32 1
  store i32 0, ptr %t97
  %t98 = getelementptr { ptr, i32 }, ptr %o24, i32 0, i32 1
  %t99 = load i32, ptr %t98
  %t100 = sext i32 %t99 to i64
  %t101 = getelementptr [4096 x float], ptr %o13, i64 0, i64 %t100
  %t102 = load i64, ptr %d53
  %t103 = mul i64 %t100, 4
  call void @mapal_gpu_d2h(ptr %t101, i64 %t102, i64 %t103, i64 4)
  %t104 = load float, ptr %t101
  store float %t104, ptr %o25
  %t105 = load float, ptr %o25
  store float %t105, ptr %o26
  %t106 = load float, ptr %o26
  call void @mapal_print_f32(float %t106, i1 zeroext true)
  %t107 = getelementptr { ptr, i32 }, ptr %o28, i32 0, i32 1
  store i32 4095, ptr %t107
  %t108 = getelementptr { ptr, i32 }, ptr %o28, i32 0, i32 1
  %t109 = load i32, ptr %t108
  %t110 = sext i32 %t109 to i64
  %t111 = getelementptr [4096 x float], ptr %o13, i64 0, i64 %t110
  %t112 = load i64, ptr %d53
  %t113 = mul i64 %t110, 4
  call void @mapal_gpu_d2h(ptr %t111, i64 %t112, i64 %t113, i64 4)
  %t114 = load float, ptr %t111
  store float %t114, ptr %o29
  call void @mapal_gpu_free(ptr %d53)
  %t115 = load float, ptr %o29
  store float %t115, ptr %o30
  %t116 = load float, ptr %o30
  call void @mapal_print_f32(float %t116, i1 zeroext true)
  %t117 = getelementptr { ptr, i32 }, ptr %o32, i32 0, i32 1
  store i32 0, ptr %t117
  %t118 = getelementptr { ptr, i32 }, ptr %o32, i32 0, i32 1
  %t119 = load i32, ptr %t118
  %t120 = sext i32 %t119 to i64
  %t121 = getelementptr [4096 x float], ptr %o10, i64 0, i64 %t120
  %t122 = load float, ptr %t121
  store float %t122, ptr %o33
  %t123 = load float, ptr %o33
  store float %t123, ptr %o34
  %t124 = load float, ptr %o34
  call void @mapal_print_f32(float %t124, i1 zeroext true)
  %t125 = getelementptr { ptr, i32 }, ptr %o36, i32 0, i32 1
  store i32 4095, ptr %t125
  %t126 = getelementptr { ptr, i32 }, ptr %o36, i32 0, i32 1
  %t127 = load i32, ptr %t126
  %t128 = sext i32 %t127 to i64
  %t129 = getelementptr [4096 x float], ptr %o10, i64 0, i64 %t128
  %t130 = load float, ptr %t129
  store float %t130, ptr %o37
  call void @mapal_gpu_free(ptr %d38)
  %t131 = load float, ptr %o37
  store float %t131, ptr %o38
  %t132 = load float, ptr %o38
  call void @mapal_print_f32(float %t132, i1 zeroext true)
  %t133 = getelementptr { ptr, i32 }, ptr %o40, i32 0, i32 1
  store i32 0, ptr %t133
  %t134 = getelementptr { ptr, i32 }, ptr %o40, i32 0, i32 1
  %t135 = load i32, ptr %t134
  %t136 = sext i32 %t135 to i64
  %t137 = getelementptr [4096 x float], ptr %o3, i64 0, i64 %t136
  %t138 = load i64, ptr %d0
  %t139 = mul i64 %t136, 4
  call void @mapal_gpu_d2h(ptr %t137, i64 %t138, i64 %t139, i64 4)
  %t140 = load float, ptr %t137
  store float %t140, ptr %o41
  call void @mapal_gpu_free(ptr %d0)
  %t141 = load float, ptr %o41
  store float %t141, ptr %o42
  %t142 = load float, ptr %o42
  call void @mapal_print_f32(float %t142, i1 zeroext true)
  %t143 = getelementptr { ptr, i32 }, ptr %o44, i32 0, i32 1
  store i32 0, ptr %t143
  %t144 = getelementptr { ptr, i32 }, ptr %o44, i32 0, i32 1
  %t145 = load i32, ptr %t144
  %t146 = sext i32 %t145 to i64
  %t147 = getelementptr [4096 x float], ptr %o4, i64 0, i64 %t146
  %t148 = load i64, ptr %d2
  %t149 = mul i64 %t146, 4
  call void @mapal_gpu_d2h(ptr %t147, i64 %t148, i64 %t149, i64 4)
  %t150 = load float, ptr %t147
  store float %t150, ptr %o45
  call void @mapal_gpu_free(ptr %d2)
  %t151 = load float, ptr %o45
  store float %t151, ptr %o46
  %t152 = load float, ptr %o46
  call void @mapal_print_f32(float %t152, i1 zeroext true)
  %t153 = getelementptr { ptr, i32 }, ptr %o48, i32 0, i32 1
  store i32 0, ptr %t153
  %t154 = getelementptr { ptr, i32 }, ptr %o48, i32 0, i32 1
  %t155 = load i32, ptr %t154
  %t156 = sext i32 %t155 to i64
  %t157 = getelementptr [4096 x float], ptr %o5, i64 0, i64 %t156
  %t158 = load i64, ptr %d4
  %t159 = mul i64 %t156, 4
  call void @mapal_gpu_d2h(ptr %t157, i64 %t158, i64 %t159, i64 4)
  %t160 = load float, ptr %t157
  store float %t160, ptr %o49
  call void @mapal_gpu_free(ptr %d4)
  %t161 = load float, ptr %o49
  store float %t161, ptr %o50
  %t162 = load float, ptr %o50
  call void @mapal_print_f32(float %t162, i1 zeroext true)
  %t163 = getelementptr { ptr, i32 }, ptr %o52, i32 0, i32 1
  store i32 0, ptr %t163
  %t164 = getelementptr { ptr, i32 }, ptr %o52, i32 0, i32 1
  %t165 = load i32, ptr %t164
  %t166 = sext i32 %t165 to i64
  %t167 = getelementptr [4096 x float], ptr %o6, i64 0, i64 %t166
  %t168 = load i64, ptr %d6
  %t169 = mul i64 %t166, 4
  call void @mapal_gpu_d2h(ptr %t167, i64 %t168, i64 %t169, i64 4)
  %t170 = load float, ptr %t167
  store float %t170, ptr %o53
  call void @mapal_gpu_free(ptr %d6)
  %t171 = load float, ptr %o53
  store float %t171, ptr %o54
  %t172 = load float, ptr %o54
  call void @mapal_print_f32(float %t172, i1 zeroext true)
  %t173 = getelementptr { ptr, i32 }, ptr %o56, i32 0, i32 1
  store i32 0, ptr %t173
  %t174 = getelementptr { ptr, i32 }, ptr %o56, i32 0, i32 1
  %t175 = load i32, ptr %t174
  %t176 = sext i32 %t175 to i64
  %t177 = getelementptr [4096 x float], ptr %o9, i64 0, i64 %t176
  %t178 = load i64, ptr %d33
  %t179 = mul i64 %t176, 4
  call void @mapal_gpu_d2h_on(i32 1, ptr %t177, i64 %t178, i64 %t179, i64 4)
  %t180 = load float, ptr %t177
  store float %t180, ptr %o57
  %t181 = load float, ptr %o57
  store float %t181, ptr %o58
  %t182 = load float, ptr %o58
  call void @mapal_print_f32(float %t182, i1 zeroext true)
  %t183 = getelementptr { ptr, i32 }, ptr %o60, i32 0, i32 1
  store i32 4095, ptr %t183
  %t184 = getelementptr { ptr, i32 }, ptr %o60, i32 0, i32 1
  %t185 = load i32, ptr %t184
  %t186 = sext i32 %t185 to i64
  %t187 = getelementptr [4096 x float], ptr %o9, i64 0, i64 %t186
  %t188 = load i64, ptr %d33
  %t189 = mul i64 %t186, 4
  call void @mapal_gpu_d2h_on(i32 1, ptr %t187, i64 %t188, i64 %t189, i64 4)
  %t190 = load float, ptr %t187
  store float %t190, ptr %o61
  call void @mapal_gpu_free(ptr %d33)
  %t191 = load float, ptr %o61
  store float %t191, ptr %o62
  %t192 = load float, ptr %o62
  call void @mapal_print_f32(float %t192, i1 zeroext true)
  %t193 = getelementptr { ptr, i32 }, ptr %o64, i32 0, i32 1
  store i32 0, ptr %t193
  %t194 = getelementptr { ptr, i32 }, ptr %o64, i32 0, i32 1
  %t195 = load i32, ptr %t194
  %t196 = sext i32 %t195 to i64
  %t197 = getelementptr [4096 x float], ptr %o12, i64 0, i64 %t196
  %t198 = load i64, ptr %d35
  %t199 = mul i64 %t196, 4
  call void @mapal_gpu_d2h_on(i32 2, ptr %t197, i64 %t198, i64 %t199, i64 4)
  %t200 = load float, ptr %t197
  store float %t200, ptr %o65
  %t201 = load float, ptr %o65
  store float %t201, ptr %o66
  %t202 = load float, ptr %o66
  call void @mapal_print_f32(float %t202, i1 zeroext true)
  %t203 = getelementptr { ptr, i32 }, ptr %o68, i32 0, i32 1
  store i32 4095, ptr %t203
  %t204 = getelementptr { ptr, i32 }, ptr %o68, i32 0, i32 1
  %t205 = load i32, ptr %t204
  %t206 = sext i32 %t205 to i64
  %t207 = getelementptr [4096 x float], ptr %o12, i64 0, i64 %t206
  %t208 = load i64, ptr %d35
  %t209 = mul i64 %t206, 4
  call void @mapal_gpu_d2h_on(i32 2, ptr %t207, i64 %t208, i64 %t209, i64 4)
  %t210 = load float, ptr %t207
  store float %t210, ptr %o69
  call void @mapal_gpu_free(ptr %d35)
  %t211 = load float, ptr %o69
  store float %t211, ptr %o70
  %t212 = load float, ptr %o70
  call void @mapal_print_f32(float %t212, i1 zeroext true)
  call void @mapal_gpu_free(ptr %d0)
  call void @mapal_gpu_free(ptr %d2)
  call void @mapal_gpu_free(ptr %d4)
  call void @mapal_gpu_free(ptr %d6)
  call void @mapal_gpu_free(ptr %d33)
  call void @mapal_gpu_free(ptr %d38)
  call void @mapal_gpu_free(ptr %d35)
  call void @mapal_gpu_free(ptr %d53)
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

define internal float @fn5({ ptr, i32, ptr, i32, float, i32 } %arg) readonly nounwind willreturn {
entry:
  %o0 = alloca { ptr, i32, ptr, i32, float, i32 }
  %o1 = alloca float
  %o2 = alloca ptr
  %o3 = alloca i32
  %o4 = alloca ptr
  %o5 = alloca i32
  %o6 = alloca float
  %o7 = alloca i32
  %o8 = alloca { i32, i32 }
  %o9 = alloca i32
  %o10 = alloca { i32, i32 }
  %o11 = alloca i32
  %o12 = alloca { ptr, i32 }
  %o13 = alloca float
  %o14 = alloca { i32, i32 }
  %o15 = alloca i32
  %o16 = alloca { i32, i32 }
  %o17 = alloca i32
  %o18 = alloca { ptr, i32 }
  %o19 = alloca float
  %o20 = alloca { float, float }
  %o21 = alloca float
  %o22 = alloca { float, float }
  store { ptr, i32, ptr, i32, float, i32 } %arg, ptr %o0
  %t0 = getelementptr { ptr, i32, ptr, i32, float, i32 }, ptr %o0, i32 0, i32 0
  %t1 = load ptr, ptr %t0
  store ptr %t1, ptr %o2
  %t2 = getelementptr { ptr, i32, ptr, i32, float, i32 }, ptr %o0, i32 0, i32 1
  %t3 = load i32, ptr %t2
  store i32 %t3, ptr %o3
  %t4 = getelementptr { ptr, i32, ptr, i32, float, i32 }, ptr %o0, i32 0, i32 2
  %t5 = load ptr, ptr %t4
  store ptr %t5, ptr %o4
  %t6 = getelementptr { ptr, i32, ptr, i32, float, i32 }, ptr %o0, i32 0, i32 3
  %t7 = load i32, ptr %t6
  store i32 %t7, ptr %o5
  %t8 = getelementptr { ptr, i32, ptr, i32, float, i32 }, ptr %o0, i32 0, i32 4
  %t9 = load float, ptr %t8
  store float %t9, ptr %o6
  %t10 = getelementptr { ptr, i32, ptr, i32, float, i32 }, ptr %o0, i32 0, i32 5
  %t11 = load i32, ptr %t10
  store i32 %t11, ptr %o7
  %t12 = load ptr, ptr %o2
  %t13 = getelementptr { ptr, i32 }, ptr %o12, i32 0, i32 0
  store ptr %t12, ptr %t13
  %t14 = load i32, ptr %o3
  %t15 = getelementptr { i32, i32 }, ptr %o8, i32 0, i32 0
  store i32 %t14, ptr %t15
  %t16 = load ptr, ptr %o4
  %t17 = getelementptr { ptr, i32 }, ptr %o18, i32 0, i32 0
  store ptr %t16, ptr %t17
  %t18 = load i32, ptr %o5
  %t19 = getelementptr { i32, i32 }, ptr %o16, i32 0, i32 1
  store i32 %t18, ptr %t19
  %t20 = load float, ptr %o6
  %t21 = getelementptr { float, float }, ptr %o22, i32 0, i32 0
  store float %t20, ptr %t21
  %t22 = load i32, ptr %o7
  %t23 = getelementptr { i32, i32 }, ptr %o10, i32 0, i32 1
  store i32 %t22, ptr %t23
  %t24 = load i32, ptr %o7
  %t25 = getelementptr { i32, i32 }, ptr %o14, i32 0, i32 0
  store i32 %t24, ptr %t25
  %t26 = getelementptr { i32, i32 }, ptr %o8, i32 0, i32 1
  store i32 64, ptr %t26
  %t27 = getelementptr { i32, i32 }, ptr %o8, i32 0, i32 0
  %t28 = load i32, ptr %t27
  %t29 = getelementptr { i32, i32 }, ptr %o8, i32 0, i32 1
  %t30 = load i32, ptr %t29
  %t31 = mul i32 %t28, %t30
  store i32 %t31, ptr %o9
  %t32 = load i32, ptr %o9
  %t33 = getelementptr { i32, i32 }, ptr %o10, i32 0, i32 0
  store i32 %t32, ptr %t33
  %t34 = getelementptr { i32, i32 }, ptr %o10, i32 0, i32 0
  %t35 = load i32, ptr %t34
  %t36 = getelementptr { i32, i32 }, ptr %o10, i32 0, i32 1
  %t37 = load i32, ptr %t36
  %t38 = add i32 %t35, %t37
  store i32 %t38, ptr %o11
  %t39 = load i32, ptr %o11
  %t40 = getelementptr { ptr, i32 }, ptr %o12, i32 0, i32 1
  store i32 %t39, ptr %t40
  %t41 = load ptr, ptr %o2
  %t42 = getelementptr { ptr, i32 }, ptr %o12, i32 0, i32 1
  %t43 = load i32, ptr %t42
  %t44 = sext i32 %t43 to i64
  %t45 = getelementptr [4096 x float], ptr %t41, i64 0, i64 %t44
  %t46 = load float, ptr %t45
  store float %t46, ptr %o13
  %t47 = load float, ptr %o13
  %t48 = getelementptr { float, float }, ptr %o20, i32 0, i32 0
  store float %t47, ptr %t48
  %t49 = getelementptr { i32, i32 }, ptr %o14, i32 0, i32 1
  store i32 64, ptr %t49
  %t50 = getelementptr { i32, i32 }, ptr %o14, i32 0, i32 0
  %t51 = load i32, ptr %t50
  %t52 = getelementptr { i32, i32 }, ptr %o14, i32 0, i32 1
  %t53 = load i32, ptr %t52
  %t54 = mul i32 %t51, %t53
  store i32 %t54, ptr %o15
  %t55 = load i32, ptr %o15
  %t56 = getelementptr { i32, i32 }, ptr %o16, i32 0, i32 0
  store i32 %t55, ptr %t56
  %t57 = getelementptr { i32, i32 }, ptr %o16, i32 0, i32 0
  %t58 = load i32, ptr %t57
  %t59 = getelementptr { i32, i32 }, ptr %o16, i32 0, i32 1
  %t60 = load i32, ptr %t59
  %t61 = add i32 %t58, %t60
  store i32 %t61, ptr %o17
  %t62 = load i32, ptr %o17
  %t63 = getelementptr { ptr, i32 }, ptr %o18, i32 0, i32 1
  store i32 %t62, ptr %t63
  %t64 = load ptr, ptr %o4
  %t65 = getelementptr { ptr, i32 }, ptr %o18, i32 0, i32 1
  %t66 = load i32, ptr %t65
  %t67 = sext i32 %t66 to i64
  %t68 = getelementptr [4096 x float], ptr %t64, i64 0, i64 %t67
  %t69 = load float, ptr %t68
  store float %t69, ptr %o19
  %t70 = load float, ptr %o19
  %t71 = getelementptr { float, float }, ptr %o20, i32 0, i32 1
  store float %t70, ptr %t71
  %t72 = getelementptr { float, float }, ptr %o20, i32 0, i32 0
  %t73 = load float, ptr %t72
  %t74 = getelementptr { float, float }, ptr %o20, i32 0, i32 1
  %t75 = load float, ptr %t74
  %t76 = fmul float %t73, %t75
  store float %t76, ptr %o21
  %t77 = load float, ptr %o21
  %t78 = getelementptr { float, float }, ptr %o22, i32 0, i32 1
  store float %t77, ptr %t78
  %t79 = getelementptr { float, float }, ptr %o22, i32 0, i32 0
  %t80 = load float, ptr %t79
  %t81 = getelementptr { float, float }, ptr %o22, i32 0, i32 1
  %t82 = load float, ptr %t81
  %t83 = fadd float %t80, %t82
  store float %t83, ptr %o1
  %t84 = load float, ptr %o1
  ret float %t84
}

define internal float @fn6({ ptr, ptr, ptr, i32 } %arg) readonly nounwind willreturn {
entry:
  %o0 = alloca { ptr, ptr, ptr, i32 }
  %o1 = alloca float
  %o2 = alloca ptr
  %o3 = alloca ptr
  %o4 = alloca ptr
  %o5 = alloca i32
  %o6 = alloca { i32, i32 }
  %o7 = alloca i32
  %o8 = alloca { i32, i32 }
  %o9 = alloca i32
  %o10 = alloca { ptr, i32, ptr, i32, float, ptr }
  %s38 = alloca float
  %s39 = alloca i64
  %s48 = alloca { ptr, i32, ptr, i32, float, i32 }
  store { ptr, ptr, ptr, i32 } %arg, ptr %o0
  %t0 = getelementptr { ptr, ptr, ptr, i32 }, ptr %o0, i32 0, i32 0
  %t1 = load ptr, ptr %t0
  store ptr %t1, ptr %o2
  %t2 = getelementptr { ptr, ptr, ptr, i32 }, ptr %o0, i32 0, i32 1
  %t3 = load ptr, ptr %t2
  store ptr %t3, ptr %o3
  %t4 = getelementptr { ptr, ptr, ptr, i32 }, ptr %o0, i32 0, i32 2
  %t5 = load ptr, ptr %t4
  store ptr %t5, ptr %o4
  %t6 = getelementptr { ptr, ptr, ptr, i32 }, ptr %o0, i32 0, i32 3
  %t7 = load i32, ptr %t6
  store i32 %t7, ptr %o5
  %t8 = load ptr, ptr %o2
  %t9 = getelementptr { ptr, i32, ptr, i32, float, ptr }, ptr %o10, i32 0, i32 5
  store ptr %t8, ptr %t9
  %t10 = load ptr, ptr %o3
  %t11 = getelementptr { ptr, i32, ptr, i32, float, ptr }, ptr %o10, i32 0, i32 0
  store ptr %t10, ptr %t11
  %t12 = load ptr, ptr %o4
  %t13 = getelementptr { ptr, i32, ptr, i32, float, ptr }, ptr %o10, i32 0, i32 2
  store ptr %t12, ptr %t13
  %t14 = load i32, ptr %o5
  %t15 = getelementptr { i32, i32 }, ptr %o6, i32 0, i32 0
  store i32 %t14, ptr %t15
  %t16 = load i32, ptr %o5
  %t17 = getelementptr { i32, i32 }, ptr %o8, i32 0, i32 0
  store i32 %t16, ptr %t17
  %t18 = getelementptr { i32, i32 }, ptr %o6, i32 0, i32 1
  store i32 64, ptr %t18
  %t19 = getelementptr { i32, i32 }, ptr %o6, i32 0, i32 0
  %t20 = load i32, ptr %t19
  %t21 = getelementptr { i32, i32 }, ptr %o6, i32 0, i32 1
  %t22 = load i32, ptr %t21
  %t23 = sdiv i32 %t20, %t22
  store i32 %t23, ptr %o7
  %t24 = load i32, ptr %o7
  %t25 = getelementptr { ptr, i32, ptr, i32, float, ptr }, ptr %o10, i32 0, i32 1
  store i32 %t24, ptr %t25
  %t26 = getelementptr { i32, i32 }, ptr %o8, i32 0, i32 1
  store i32 64, ptr %t26
  %t27 = getelementptr { i32, i32 }, ptr %o8, i32 0, i32 0
  %t28 = load i32, ptr %t27
  %t29 = getelementptr { i32, i32 }, ptr %o8, i32 0, i32 1
  %t30 = load i32, ptr %t29
  %t31 = srem i32 %t28, %t30
  store i32 %t31, ptr %o9
  %t32 = load i32, ptr %o9
  %t33 = getelementptr { ptr, i32, ptr, i32, float, ptr }, ptr %o10, i32 0, i32 3
  store i32 %t32, ptr %t33
  %t34 = getelementptr { ptr, i32, ptr, i32, float, ptr }, ptr %o10, i32 0, i32 4
  store float 0x0000000000000000, ptr %t34
  %t35 = getelementptr { ptr, i32, ptr, i32, float, ptr }, ptr %o10, i32 0, i32 4
  %t36 = load float, ptr %t35
  %t37 = load ptr, ptr %o2
  store float %t36, ptr %s38
  store i64 0, ptr %s39
  br label %bb40
bb40:
  %t43 = load i64, ptr %s39
  %t44 = icmp uge i64 %t43, 64
  br i1 %t44, label %bb42, label %bb41
bb41:
  %t45 = getelementptr [64 x i32], ptr %t37, i64 0, i64 %t43
  %t46 = load i32, ptr %t45
  %t47 = load float, ptr %s38
  %t49 = load ptr, ptr %o3
  %t50 = getelementptr { ptr, i32, ptr, i32, float, i32 }, ptr %s48, i32 0, i32 0
  store ptr %t49, ptr %t50
  %t51 = getelementptr { ptr, i32, ptr, i32, float, ptr }, ptr %o10, i32 0, i32 1
  %t52 = load i32, ptr %t51
  %t53 = getelementptr { ptr, i32, ptr, i32, float, i32 }, ptr %s48, i32 0, i32 1
  store i32 %t52, ptr %t53
  %t54 = load ptr, ptr %o4
  %t55 = getelementptr { ptr, i32, ptr, i32, float, i32 }, ptr %s48, i32 0, i32 2
  store ptr %t54, ptr %t55
  %t56 = getelementptr { ptr, i32, ptr, i32, float, ptr }, ptr %o10, i32 0, i32 3
  %t57 = load i32, ptr %t56
  %t58 = getelementptr { ptr, i32, ptr, i32, float, i32 }, ptr %s48, i32 0, i32 3
  store i32 %t57, ptr %t58
  %t59 = getelementptr { ptr, i32, ptr, i32, float, i32 }, ptr %s48, i32 0, i32 4
  store float %t47, ptr %t59
  %t60 = getelementptr { ptr, i32, ptr, i32, float, i32 }, ptr %s48, i32 0, i32 5
  store i32 %t46, ptr %t60
  %t61 = load { ptr, i32, ptr, i32, float, i32 }, ptr %s48
  %t62 = call float @fn5({ ptr, i32, ptr, i32, float, i32 } %t61)
  store float %t62, ptr %s38
  %t63 = add i64 %t43, 1
  store i64 %t63, ptr %s39
  br label %bb40
bb42:
  %t64 = load float, ptr %s38
  store float %t64, ptr %o1
  %t65 = load float, ptr %o1
  ret float %t65
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

define internal float @fn8({ ptr, i32, ptr, i32, float, i32 } %arg) readonly nounwind willreturn {
entry:
  %o0 = alloca { ptr, i32, ptr, i32, float, i32 }
  %o1 = alloca float
  %o2 = alloca ptr
  %o3 = alloca i32
  %o4 = alloca ptr
  %o5 = alloca i32
  %o6 = alloca float
  %o7 = alloca i32
  %o8 = alloca { i32, i32 }
  %o9 = alloca i32
  %o10 = alloca { i32, i32 }
  %o11 = alloca i32
  %o12 = alloca { ptr, i32 }
  %o13 = alloca float
  %o14 = alloca { i32, i32 }
  %o15 = alloca i32
  %o16 = alloca { i32, i32 }
  %o17 = alloca i32
  %o18 = alloca { ptr, i32 }
  %o19 = alloca float
  %o20 = alloca { float, float }
  %o21 = alloca float
  %o22 = alloca { float, float }
  store { ptr, i32, ptr, i32, float, i32 } %arg, ptr %o0
  %t0 = getelementptr { ptr, i32, ptr, i32, float, i32 }, ptr %o0, i32 0, i32 0
  %t1 = load ptr, ptr %t0
  store ptr %t1, ptr %o2
  %t2 = getelementptr { ptr, i32, ptr, i32, float, i32 }, ptr %o0, i32 0, i32 1
  %t3 = load i32, ptr %t2
  store i32 %t3, ptr %o3
  %t4 = getelementptr { ptr, i32, ptr, i32, float, i32 }, ptr %o0, i32 0, i32 2
  %t5 = load ptr, ptr %t4
  store ptr %t5, ptr %o4
  %t6 = getelementptr { ptr, i32, ptr, i32, float, i32 }, ptr %o0, i32 0, i32 3
  %t7 = load i32, ptr %t6
  store i32 %t7, ptr %o5
  %t8 = getelementptr { ptr, i32, ptr, i32, float, i32 }, ptr %o0, i32 0, i32 4
  %t9 = load float, ptr %t8
  store float %t9, ptr %o6
  %t10 = getelementptr { ptr, i32, ptr, i32, float, i32 }, ptr %o0, i32 0, i32 5
  %t11 = load i32, ptr %t10
  store i32 %t11, ptr %o7
  %t12 = load ptr, ptr %o2
  %t13 = getelementptr { ptr, i32 }, ptr %o12, i32 0, i32 0
  store ptr %t12, ptr %t13
  %t14 = load i32, ptr %o3
  %t15 = getelementptr { i32, i32 }, ptr %o8, i32 0, i32 0
  store i32 %t14, ptr %t15
  %t16 = load ptr, ptr %o4
  %t17 = getelementptr { ptr, i32 }, ptr %o18, i32 0, i32 0
  store ptr %t16, ptr %t17
  %t18 = load i32, ptr %o5
  %t19 = getelementptr { i32, i32 }, ptr %o16, i32 0, i32 1
  store i32 %t18, ptr %t19
  %t20 = load float, ptr %o6
  %t21 = getelementptr { float, float }, ptr %o22, i32 0, i32 0
  store float %t20, ptr %t21
  %t22 = load i32, ptr %o7
  %t23 = getelementptr { i32, i32 }, ptr %o10, i32 0, i32 1
  store i32 %t22, ptr %t23
  %t24 = load i32, ptr %o7
  %t25 = getelementptr { i32, i32 }, ptr %o14, i32 0, i32 0
  store i32 %t24, ptr %t25
  %t26 = getelementptr { i32, i32 }, ptr %o8, i32 0, i32 1
  store i32 64, ptr %t26
  %t27 = getelementptr { i32, i32 }, ptr %o8, i32 0, i32 0
  %t28 = load i32, ptr %t27
  %t29 = getelementptr { i32, i32 }, ptr %o8, i32 0, i32 1
  %t30 = load i32, ptr %t29
  %t31 = mul i32 %t28, %t30
  store i32 %t31, ptr %o9
  %t32 = load i32, ptr %o9
  %t33 = getelementptr { i32, i32 }, ptr %o10, i32 0, i32 0
  store i32 %t32, ptr %t33
  %t34 = getelementptr { i32, i32 }, ptr %o10, i32 0, i32 0
  %t35 = load i32, ptr %t34
  %t36 = getelementptr { i32, i32 }, ptr %o10, i32 0, i32 1
  %t37 = load i32, ptr %t36
  %t38 = add i32 %t35, %t37
  store i32 %t38, ptr %o11
  %t39 = load i32, ptr %o11
  %t40 = getelementptr { ptr, i32 }, ptr %o12, i32 0, i32 1
  store i32 %t39, ptr %t40
  %t41 = load ptr, ptr %o2
  %t42 = getelementptr { ptr, i32 }, ptr %o12, i32 0, i32 1
  %t43 = load i32, ptr %t42
  %t44 = sext i32 %t43 to i64
  %t45 = getelementptr [4096 x float], ptr %t41, i64 0, i64 %t44
  %t46 = load float, ptr %t45
  store float %t46, ptr %o13
  %t47 = load float, ptr %o13
  %t48 = getelementptr { float, float }, ptr %o20, i32 0, i32 0
  store float %t47, ptr %t48
  %t49 = getelementptr { i32, i32 }, ptr %o14, i32 0, i32 1
  store i32 64, ptr %t49
  %t50 = getelementptr { i32, i32 }, ptr %o14, i32 0, i32 0
  %t51 = load i32, ptr %t50
  %t52 = getelementptr { i32, i32 }, ptr %o14, i32 0, i32 1
  %t53 = load i32, ptr %t52
  %t54 = mul i32 %t51, %t53
  store i32 %t54, ptr %o15
  %t55 = load i32, ptr %o15
  %t56 = getelementptr { i32, i32 }, ptr %o16, i32 0, i32 0
  store i32 %t55, ptr %t56
  %t57 = getelementptr { i32, i32 }, ptr %o16, i32 0, i32 0
  %t58 = load i32, ptr %t57
  %t59 = getelementptr { i32, i32 }, ptr %o16, i32 0, i32 1
  %t60 = load i32, ptr %t59
  %t61 = add i32 %t58, %t60
  store i32 %t61, ptr %o17
  %t62 = load i32, ptr %o17
  %t63 = getelementptr { ptr, i32 }, ptr %o18, i32 0, i32 1
  store i32 %t62, ptr %t63
  %t64 = load ptr, ptr %o4
  %t65 = getelementptr { ptr, i32 }, ptr %o18, i32 0, i32 1
  %t66 = load i32, ptr %t65
  %t67 = sext i32 %t66 to i64
  %t68 = getelementptr [4096 x float], ptr %t64, i64 0, i64 %t67
  %t69 = load float, ptr %t68
  store float %t69, ptr %o19
  %t70 = load float, ptr %o19
  %t71 = getelementptr { float, float }, ptr %o20, i32 0, i32 1
  store float %t70, ptr %t71
  %t72 = getelementptr { float, float }, ptr %o20, i32 0, i32 0
  %t73 = load float, ptr %t72
  %t74 = getelementptr { float, float }, ptr %o20, i32 0, i32 1
  %t75 = load float, ptr %t74
  %t76 = fmul float %t73, %t75
  store float %t76, ptr %o21
  %t77 = load float, ptr %o21
  %t78 = getelementptr { float, float }, ptr %o22, i32 0, i32 1
  store float %t77, ptr %t78
  %t79 = getelementptr { float, float }, ptr %o22, i32 0, i32 0
  %t80 = load float, ptr %t79
  %t81 = getelementptr { float, float }, ptr %o22, i32 0, i32 1
  %t82 = load float, ptr %t81
  %t83 = fadd float %t80, %t82
  store float %t83, ptr %o1
  %t84 = load float, ptr %o1
  ret float %t84
}

define internal float @fn9({ ptr, ptr, ptr, i32 } %arg) readonly nounwind willreturn {
entry:
  %o0 = alloca { ptr, ptr, ptr, i32 }
  %o1 = alloca float
  %o2 = alloca ptr
  %o3 = alloca ptr
  %o4 = alloca ptr
  %o5 = alloca i32
  %o6 = alloca { i32, i32 }
  %o7 = alloca i32
  %o8 = alloca { i32, i32 }
  %o9 = alloca i32
  %o10 = alloca { ptr, i32, ptr, i32, float, ptr }
  %s38 = alloca float
  %s39 = alloca i64
  %s48 = alloca { ptr, i32, ptr, i32, float, i32 }
  store { ptr, ptr, ptr, i32 } %arg, ptr %o0
  %t0 = getelementptr { ptr, ptr, ptr, i32 }, ptr %o0, i32 0, i32 0
  %t1 = load ptr, ptr %t0
  store ptr %t1, ptr %o2
  %t2 = getelementptr { ptr, ptr, ptr, i32 }, ptr %o0, i32 0, i32 1
  %t3 = load ptr, ptr %t2
  store ptr %t3, ptr %o3
  %t4 = getelementptr { ptr, ptr, ptr, i32 }, ptr %o0, i32 0, i32 2
  %t5 = load ptr, ptr %t4
  store ptr %t5, ptr %o4
  %t6 = getelementptr { ptr, ptr, ptr, i32 }, ptr %o0, i32 0, i32 3
  %t7 = load i32, ptr %t6
  store i32 %t7, ptr %o5
  %t8 = load ptr, ptr %o2
  %t9 = getelementptr { ptr, i32, ptr, i32, float, ptr }, ptr %o10, i32 0, i32 5
  store ptr %t8, ptr %t9
  %t10 = load ptr, ptr %o3
  %t11 = getelementptr { ptr, i32, ptr, i32, float, ptr }, ptr %o10, i32 0, i32 0
  store ptr %t10, ptr %t11
  %t12 = load ptr, ptr %o4
  %t13 = getelementptr { ptr, i32, ptr, i32, float, ptr }, ptr %o10, i32 0, i32 2
  store ptr %t12, ptr %t13
  %t14 = load i32, ptr %o5
  %t15 = getelementptr { i32, i32 }, ptr %o6, i32 0, i32 0
  store i32 %t14, ptr %t15
  %t16 = load i32, ptr %o5
  %t17 = getelementptr { i32, i32 }, ptr %o8, i32 0, i32 0
  store i32 %t16, ptr %t17
  %t18 = getelementptr { i32, i32 }, ptr %o6, i32 0, i32 1
  store i32 64, ptr %t18
  %t19 = getelementptr { i32, i32 }, ptr %o6, i32 0, i32 0
  %t20 = load i32, ptr %t19
  %t21 = getelementptr { i32, i32 }, ptr %o6, i32 0, i32 1
  %t22 = load i32, ptr %t21
  %t23 = sdiv i32 %t20, %t22
  store i32 %t23, ptr %o7
  %t24 = load i32, ptr %o7
  %t25 = getelementptr { ptr, i32, ptr, i32, float, ptr }, ptr %o10, i32 0, i32 1
  store i32 %t24, ptr %t25
  %t26 = getelementptr { i32, i32 }, ptr %o8, i32 0, i32 1
  store i32 64, ptr %t26
  %t27 = getelementptr { i32, i32 }, ptr %o8, i32 0, i32 0
  %t28 = load i32, ptr %t27
  %t29 = getelementptr { i32, i32 }, ptr %o8, i32 0, i32 1
  %t30 = load i32, ptr %t29
  %t31 = srem i32 %t28, %t30
  store i32 %t31, ptr %o9
  %t32 = load i32, ptr %o9
  %t33 = getelementptr { ptr, i32, ptr, i32, float, ptr }, ptr %o10, i32 0, i32 3
  store i32 %t32, ptr %t33
  %t34 = getelementptr { ptr, i32, ptr, i32, float, ptr }, ptr %o10, i32 0, i32 4
  store float 0x0000000000000000, ptr %t34
  %t35 = getelementptr { ptr, i32, ptr, i32, float, ptr }, ptr %o10, i32 0, i32 4
  %t36 = load float, ptr %t35
  %t37 = load ptr, ptr %o2
  store float %t36, ptr %s38
  store i64 0, ptr %s39
  br label %bb40
bb40:
  %t43 = load i64, ptr %s39
  %t44 = icmp uge i64 %t43, 64
  br i1 %t44, label %bb42, label %bb41
bb41:
  %t45 = getelementptr [64 x i32], ptr %t37, i64 0, i64 %t43
  %t46 = load i32, ptr %t45
  %t47 = load float, ptr %s38
  %t49 = load ptr, ptr %o3
  %t50 = getelementptr { ptr, i32, ptr, i32, float, i32 }, ptr %s48, i32 0, i32 0
  store ptr %t49, ptr %t50
  %t51 = getelementptr { ptr, i32, ptr, i32, float, ptr }, ptr %o10, i32 0, i32 1
  %t52 = load i32, ptr %t51
  %t53 = getelementptr { ptr, i32, ptr, i32, float, i32 }, ptr %s48, i32 0, i32 1
  store i32 %t52, ptr %t53
  %t54 = load ptr, ptr %o4
  %t55 = getelementptr { ptr, i32, ptr, i32, float, i32 }, ptr %s48, i32 0, i32 2
  store ptr %t54, ptr %t55
  %t56 = getelementptr { ptr, i32, ptr, i32, float, ptr }, ptr %o10, i32 0, i32 3
  %t57 = load i32, ptr %t56
  %t58 = getelementptr { ptr, i32, ptr, i32, float, i32 }, ptr %s48, i32 0, i32 3
  store i32 %t57, ptr %t58
  %t59 = getelementptr { ptr, i32, ptr, i32, float, i32 }, ptr %s48, i32 0, i32 4
  store float %t47, ptr %t59
  %t60 = getelementptr { ptr, i32, ptr, i32, float, i32 }, ptr %s48, i32 0, i32 5
  store i32 %t46, ptr %t60
  %t61 = load { ptr, i32, ptr, i32, float, i32 }, ptr %s48
  %t62 = call float @fn8({ ptr, i32, ptr, i32, float, i32 } %t61)
  store float %t62, ptr %s38
  %t63 = add i64 %t43, 1
  store i64 %t63, ptr %s39
  br label %bb40
bb42:
  %t64 = load float, ptr %s38
  store float %t64, ptr %o1
  %t65 = load float, ptr %o1
  ret float %t65
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

define internal float @fn11({ ptr, i32, float, i32 } %arg) readonly nounwind willreturn {
entry:
  %o0 = alloca { ptr, i32, float, i32 }
  %o1 = alloca float
  %o2 = alloca ptr
  %o3 = alloca i32
  %o4 = alloca float
  %o5 = alloca i32
  %o6 = alloca { i32, i32 }
  %o7 = alloca i32
  %o8 = alloca { i32, i32 }
  %o9 = alloca i32
  %o10 = alloca { ptr, i32 }
  %o11 = alloca float
  %o12 = alloca { float, float }
  store { ptr, i32, float, i32 } %arg, ptr %o0
  %t0 = getelementptr { ptr, i32, float, i32 }, ptr %o0, i32 0, i32 0
  %t1 = load ptr, ptr %t0
  store ptr %t1, ptr %o2
  %t2 = getelementptr { ptr, i32, float, i32 }, ptr %o0, i32 0, i32 1
  %t3 = load i32, ptr %t2
  store i32 %t3, ptr %o3
  %t4 = getelementptr { ptr, i32, float, i32 }, ptr %o0, i32 0, i32 2
  %t5 = load float, ptr %t4
  store float %t5, ptr %o4
  %t6 = getelementptr { ptr, i32, float, i32 }, ptr %o0, i32 0, i32 3
  %t7 = load i32, ptr %t6
  store i32 %t7, ptr %o5
  %t8 = load ptr, ptr %o2
  %t9 = getelementptr { ptr, i32 }, ptr %o10, i32 0, i32 0
  store ptr %t8, ptr %t9
  %t10 = load i32, ptr %o3
  %t11 = getelementptr { i32, i32 }, ptr %o6, i32 0, i32 0
  store i32 %t10, ptr %t11
  %t12 = load float, ptr %o4
  %t13 = getelementptr { float, float }, ptr %o12, i32 0, i32 0
  store float %t12, ptr %t13
  %t14 = load i32, ptr %o5
  %t15 = getelementptr { i32, i32 }, ptr %o8, i32 0, i32 1
  store i32 %t14, ptr %t15
  %t16 = getelementptr { i32, i32 }, ptr %o6, i32 0, i32 1
  store i32 64, ptr %t16
  %t17 = getelementptr { i32, i32 }, ptr %o6, i32 0, i32 0
  %t18 = load i32, ptr %t17
  %t19 = getelementptr { i32, i32 }, ptr %o6, i32 0, i32 1
  %t20 = load i32, ptr %t19
  %t21 = mul i32 %t18, %t20
  store i32 %t21, ptr %o7
  %t22 = load i32, ptr %o7
  %t23 = getelementptr { i32, i32 }, ptr %o8, i32 0, i32 0
  store i32 %t22, ptr %t23
  %t24 = getelementptr { i32, i32 }, ptr %o8, i32 0, i32 0
  %t25 = load i32, ptr %t24
  %t26 = getelementptr { i32, i32 }, ptr %o8, i32 0, i32 1
  %t27 = load i32, ptr %t26
  %t28 = add i32 %t25, %t27
  store i32 %t28, ptr %o9
  %t29 = load i32, ptr %o9
  %t30 = getelementptr { ptr, i32 }, ptr %o10, i32 0, i32 1
  store i32 %t29, ptr %t30
  %t31 = load ptr, ptr %o2
  %t32 = getelementptr { ptr, i32 }, ptr %o10, i32 0, i32 1
  %t33 = load i32, ptr %t32
  %t34 = sext i32 %t33 to i64
  %t35 = getelementptr [4096 x float], ptr %t31, i64 0, i64 %t34
  %t36 = load float, ptr %t35
  store float %t36, ptr %o11
  %t37 = load float, ptr %o11
  %t38 = getelementptr { float, float }, ptr %o12, i32 0, i32 1
  store float %t37, ptr %t38
  %t39 = getelementptr { float, float }, ptr %o12, i32 0, i32 0
  %t40 = load float, ptr %t39
  %t41 = getelementptr { float, float }, ptr %o12, i32 0, i32 1
  %t42 = load float, ptr %t41
  %t43 = fadd float %t40, %t42
  store float %t43, ptr %o1
  %t44 = load float, ptr %o1
  ret float %t44
}

define internal float @fn12({ ptr, ptr, i32 } %arg) readonly nounwind willreturn {
entry:
  %o0 = alloca { ptr, ptr, i32 }
  %o1 = alloca float
  %o2 = alloca ptr
  %o3 = alloca ptr
  %o4 = alloca i32
  %o5 = alloca { ptr, i32, float, ptr }
  %s16 = alloca float
  %s17 = alloca i64
  %s26 = alloca { ptr, i32, float, i32 }
  store { ptr, ptr, i32 } %arg, ptr %o0
  %t0 = getelementptr { ptr, ptr, i32 }, ptr %o0, i32 0, i32 0
  %t1 = load ptr, ptr %t0
  store ptr %t1, ptr %o2
  %t2 = getelementptr { ptr, ptr, i32 }, ptr %o0, i32 0, i32 1
  %t3 = load ptr, ptr %t2
  store ptr %t3, ptr %o3
  %t4 = getelementptr { ptr, ptr, i32 }, ptr %o0, i32 0, i32 2
  %t5 = load i32, ptr %t4
  store i32 %t5, ptr %o4
  %t6 = load ptr, ptr %o2
  %t7 = getelementptr { ptr, i32, float, ptr }, ptr %o5, i32 0, i32 3
  store ptr %t6, ptr %t7
  %t8 = load ptr, ptr %o3
  %t9 = getelementptr { ptr, i32, float, ptr }, ptr %o5, i32 0, i32 0
  store ptr %t8, ptr %t9
  %t10 = load i32, ptr %o4
  %t11 = getelementptr { ptr, i32, float, ptr }, ptr %o5, i32 0, i32 1
  store i32 %t10, ptr %t11
  %t12 = getelementptr { ptr, i32, float, ptr }, ptr %o5, i32 0, i32 2
  store float 0x0000000000000000, ptr %t12
  %t13 = getelementptr { ptr, i32, float, ptr }, ptr %o5, i32 0, i32 2
  %t14 = load float, ptr %t13
  %t15 = load ptr, ptr %o2
  store float %t14, ptr %s16
  store i64 0, ptr %s17
  br label %bb18
bb18:
  %t21 = load i64, ptr %s17
  %t22 = icmp uge i64 %t21, 64
  br i1 %t22, label %bb20, label %bb19
bb19:
  %t23 = getelementptr [64 x i32], ptr %t15, i64 0, i64 %t21
  %t24 = load i32, ptr %t23
  %t25 = load float, ptr %s16
  %t27 = load ptr, ptr %o3
  %t28 = getelementptr { ptr, i32, float, i32 }, ptr %s26, i32 0, i32 0
  store ptr %t27, ptr %t28
  %t29 = getelementptr { ptr, i32, float, ptr }, ptr %o5, i32 0, i32 1
  %t30 = load i32, ptr %t29
  %t31 = getelementptr { ptr, i32, float, i32 }, ptr %s26, i32 0, i32 1
  store i32 %t30, ptr %t31
  %t32 = getelementptr { ptr, i32, float, i32 }, ptr %s26, i32 0, i32 2
  store float %t25, ptr %t32
  %t33 = getelementptr { ptr, i32, float, i32 }, ptr %s26, i32 0, i32 3
  store i32 %t24, ptr %t33
  %t34 = load { ptr, i32, float, i32 }, ptr %s26
  %t35 = call float @fn11({ ptr, i32, float, i32 } %t34)
  store float %t35, ptr %s16
  %t36 = add i64 %t21, 1
  store i64 %t36, ptr %s17
  br label %bb18
bb20:
  %t37 = load float, ptr %s16
  store float %t37, ptr %o1
  %t38 = load float, ptr %o1
  ret float %t38
}

define i32 @main() {
entry:
  call void @mapal_gpu_init(i32 3)
  call void @mapal_main()
  ret i32 0
}
