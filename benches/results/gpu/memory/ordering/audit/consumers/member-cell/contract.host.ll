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
  %o14 = alloca { ptr, ptr }
  %o15 = alloca [4096 x float]
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
  %d23 = alloca i64
  store i64 0, ptr %d23
  %d25 = alloca i64
  store i64 0, ptr %d25
  %d28 = alloca i64
  store i64 0, ptr %d28
  %p30 = alloca [4 x i64]
  %d43 = alloca i64
  store i64 0, ptr %d43
  %p45 = alloca [4 x i64]
  %d54 = alloca i64
  store i64 0, ptr %d54
  %dv57 = alloca [2 x i64]
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
  %t10 = getelementptr { ptr, ptr }, ptr %o14, i32 0, i32 1
  store ptr %o2, ptr %t10
  %t11 = getelementptr { ptr, ptr, ptr, ptr }, ptr %o8, i32 0, i32 0
  store ptr %o7, ptr %t11
  %t12 = getelementptr { ptr, ptr, ptr, ptr }, ptr %o11, i32 0, i32 0
  store ptr %o7, ptr %t12
  %t13 = getelementptr { ptr, ptr, ptr, ptr }, ptr %o8, i32 0, i32 1
  store ptr %o3, ptr %t13
  %t14 = getelementptr { ptr, i32 }, ptr %o40, i32 0, i32 0
  store ptr %o3, ptr %t14
  %t15 = getelementptr { ptr, ptr, ptr, ptr }, ptr %o8, i32 0, i32 2
  store ptr %o4, ptr %t15
  %t16 = getelementptr { ptr, i32 }, ptr %o44, i32 0, i32 0
  store ptr %o4, ptr %t16
  %t17 = getelementptr { ptr, ptr, ptr, ptr }, ptr %o11, i32 0, i32 1
  store ptr %o5, ptr %t17
  %t18 = getelementptr { ptr, i32 }, ptr %o48, i32 0, i32 0
  store ptr %o5, ptr %t18
  %t19 = getelementptr { ptr, ptr, ptr, ptr }, ptr %o11, i32 0, i32 2
  store ptr %o6, ptr %t19
  %t20 = getelementptr { ptr, i32 }, ptr %o52, i32 0, i32 0
  store ptr %o6, ptr %t20
  %t21 = load i64, ptr %d0
  %t22 = load i64, ptr %d2
  %t24 = call i64 @mapal_gpu_alloc(ptr %d23, i64 16384)
  %t26 = call i64 @mapal_gpu_alloc(ptr %d25, i64 16384)
  %t27 = call i64 @mapal_gpu_alloc(ptr %d23, i64 16384)
  %t29 = call i64 @mapal_gpu_alloc(ptr %d28, i64 16384)
  %t31 = getelementptr [4 x i64], ptr %p30, i64 0, i64 0
  store i64 %t21, ptr %t31
  %t32 = getelementptr [4 x i64], ptr %p30, i64 0, i64 1
  store i64 %t22, ptr %t32
  %t33 = getelementptr [4 x i64], ptr %p30, i64 0, i64 2
  store i64 %t27, ptr %t33
  %t34 = getelementptr [4 x i64], ptr %p30, i64 0, i64 3
  store i64 %t29, ptr %t34
  call void @mapal_gpu_launchv(i64 -5869913981016682307, i32 1, i32 4, ptr %p30, i32 4, i32 2, i32 32, i32 1664)
  %t35 = getelementptr { ptr, i32 }, ptr %o56, i32 0, i32 0
  store ptr %o9, ptr %t35
  %t36 = getelementptr { ptr, i32 }, ptr %o60, i32 0, i32 0
  store ptr %o9, ptr %t36
  %t37 = getelementptr { ptr, ptr }, ptr %o14, i32 0, i32 0
  store ptr %o10, ptr %t37
  %t38 = getelementptr { ptr, i32 }, ptr %o32, i32 0, i32 0
  store ptr %o10, ptr %t38
  %t39 = getelementptr { ptr, i32 }, ptr %o36, i32 0, i32 0
  store ptr %o10, ptr %t39
  %t40 = load i64, ptr %d4
  %t41 = load i64, ptr %d6
  %t42 = call i64 @mapal_gpu_alloc(ptr %d25, i64 16384)
  %t44 = call i64 @mapal_gpu_alloc(ptr %d43, i64 16384)
  %t46 = getelementptr [4 x i64], ptr %p45, i64 0, i64 0
  store i64 %t40, ptr %t46
  %t47 = getelementptr [4 x i64], ptr %p45, i64 0, i64 1
  store i64 %t41, ptr %t47
  %t48 = getelementptr [4 x i64], ptr %p45, i64 0, i64 2
  store i64 %t42, ptr %t48
  %t49 = getelementptr [4 x i64], ptr %p45, i64 0, i64 3
  store i64 %t44, ptr %t49
  call void @mapal_gpu_launchv(i64 -615245021611174823, i32 2, i32 4, ptr %p45, i32 4, i32 2, i32 32, i32 1664)
  %t50 = getelementptr { ptr, i32 }, ptr %o64, i32 0, i32 0
  store ptr %o12, ptr %t50
  %t51 = getelementptr { ptr, i32 }, ptr %o68, i32 0, i32 0
  store ptr %o12, ptr %t51
  %t52 = getelementptr { ptr, i32 }, ptr %o24, i32 0, i32 0
  store ptr %o13, ptr %t52
  %t53 = getelementptr { ptr, i32 }, ptr %o28, i32 0, i32 0
  store ptr %o13, ptr %t53
  %t55 = call i64 @mapal_gpu_alloc(ptr %d54, i64 16384)
  %t56 = load i64, ptr %d28
  %t58 = getelementptr [2 x i64], ptr %dv57, i64 0, i64 0
  store i64 %t55, ptr %t58
  %t59 = getelementptr [2 x i64], ptr %dv57, i64 0, i64 1
  store i64 %t56, ptr %t59
  call void @mapal_gpu_launchv(i64 4010187978629625089, i32 0, i32 2, ptr %dv57, i32 16, i32 1, i32 256, i32 0)
  %t60 = getelementptr { ptr, i32 }, ptr %o16, i32 0, i32 0
  store ptr %o15, ptr %t60
  %t61 = getelementptr { ptr, i32 }, ptr %o20, i32 0, i32 0
  store ptr %o15, ptr %t61
  %t62 = getelementptr { ptr, i32 }, ptr %o16, i32 0, i32 1
  store i32 0, ptr %t62
  %t63 = getelementptr { ptr, i32 }, ptr %o16, i32 0, i32 1
  %t64 = load i32, ptr %t63
  %t65 = sext i32 %t64 to i64
  %t66 = getelementptr [4096 x float], ptr %o15, i64 0, i64 %t65
  %t67 = load i64, ptr %d54
  %t68 = mul i64 %t65, 4
  call void @mapal_gpu_d2h(ptr %t66, i64 %t67, i64 %t68, i64 4)
  %t69 = load float, ptr %t66
  store float %t69, ptr %o17
  %t70 = load float, ptr %o17
  store float %t70, ptr %o18
  %t71 = load float, ptr %o18
  call void @mapal_print_f32(float %t71, i1 zeroext true)
  %t72 = getelementptr { ptr, i32 }, ptr %o20, i32 0, i32 1
  store i32 4095, ptr %t72
  %t73 = getelementptr { ptr, i32 }, ptr %o20, i32 0, i32 1
  %t74 = load i32, ptr %t73
  %t75 = sext i32 %t74 to i64
  %t76 = getelementptr [4096 x float], ptr %o15, i64 0, i64 %t75
  %t77 = load i64, ptr %d54
  %t78 = mul i64 %t75, 4
  call void @mapal_gpu_d2h(ptr %t76, i64 %t77, i64 %t78, i64 4)
  %t79 = load float, ptr %t76
  store float %t79, ptr %o21
  call void @mapal_gpu_free(ptr %d54)
  %t80 = load float, ptr %o21
  store float %t80, ptr %o22
  %t81 = load float, ptr %o22
  call void @mapal_print_f32(float %t81, i1 zeroext true)
  %t82 = getelementptr { ptr, i32 }, ptr %o24, i32 0, i32 1
  store i32 0, ptr %t82
  %t83 = getelementptr { ptr, i32 }, ptr %o24, i32 0, i32 1
  %t84 = load i32, ptr %t83
  %t85 = sext i32 %t84 to i64
  %t86 = getelementptr [4096 x float], ptr %o13, i64 0, i64 %t85
  %t87 = load i64, ptr %d43
  %t88 = mul i64 %t85, 4
  call void @mapal_gpu_d2h(ptr %t86, i64 %t87, i64 %t88, i64 4)
  %t89 = load float, ptr %t86
  store float %t89, ptr %o25
  %t90 = load float, ptr %o25
  store float %t90, ptr %o26
  %t91 = load float, ptr %o26
  call void @mapal_print_f32(float %t91, i1 zeroext true)
  %t92 = getelementptr { ptr, i32 }, ptr %o28, i32 0, i32 1
  store i32 4095, ptr %t92
  %t93 = getelementptr { ptr, i32 }, ptr %o28, i32 0, i32 1
  %t94 = load i32, ptr %t93
  %t95 = sext i32 %t94 to i64
  %t96 = getelementptr [4096 x float], ptr %o13, i64 0, i64 %t95
  %t97 = load i64, ptr %d43
  %t98 = mul i64 %t95, 4
  call void @mapal_gpu_d2h(ptr %t96, i64 %t97, i64 %t98, i64 4)
  %t99 = load float, ptr %t96
  store float %t99, ptr %o29
  call void @mapal_gpu_free(ptr %d43)
  %t100 = load float, ptr %o29
  store float %t100, ptr %o30
  %t101 = load float, ptr %o30
  call void @mapal_print_f32(float %t101, i1 zeroext true)
  %t102 = getelementptr { ptr, i32 }, ptr %o32, i32 0, i32 1
  store i32 0, ptr %t102
  %t103 = getelementptr { ptr, i32 }, ptr %o32, i32 0, i32 1
  %t104 = load i32, ptr %t103
  %t105 = sext i32 %t104 to i64
  %t106 = getelementptr [4096 x float], ptr %o10, i64 0, i64 %t105
  %t107 = load i64, ptr %d28
  %t108 = mul i64 %t105, 4
  call void @mapal_gpu_d2h(ptr %t106, i64 %t107, i64 %t108, i64 4)
  %t109 = load float, ptr %t106
  store float %t109, ptr %o33
  %t110 = load float, ptr %o33
  store float %t110, ptr %o34
  %t111 = load float, ptr %o34
  call void @mapal_print_f32(float %t111, i1 zeroext true)
  %t112 = getelementptr { ptr, i32 }, ptr %o36, i32 0, i32 1
  store i32 4095, ptr %t112
  %t113 = getelementptr { ptr, i32 }, ptr %o36, i32 0, i32 1
  %t114 = load i32, ptr %t113
  %t115 = sext i32 %t114 to i64
  %t116 = getelementptr [4096 x float], ptr %o10, i64 0, i64 %t115
  %t117 = load i64, ptr %d28
  %t118 = mul i64 %t115, 4
  call void @mapal_gpu_d2h(ptr %t116, i64 %t117, i64 %t118, i64 4)
  %t119 = load float, ptr %t116
  store float %t119, ptr %o37
  call void @mapal_gpu_free(ptr %d28)
  %t120 = load float, ptr %o37
  store float %t120, ptr %o38
  %t121 = load float, ptr %o38
  call void @mapal_print_f32(float %t121, i1 zeroext true)
  %t122 = getelementptr { ptr, i32 }, ptr %o40, i32 0, i32 1
  store i32 0, ptr %t122
  %t123 = getelementptr { ptr, i32 }, ptr %o40, i32 0, i32 1
  %t124 = load i32, ptr %t123
  %t125 = sext i32 %t124 to i64
  %t126 = getelementptr [4096 x float], ptr %o3, i64 0, i64 %t125
  %t127 = load i64, ptr %d0
  %t128 = mul i64 %t125, 4
  call void @mapal_gpu_d2h(ptr %t126, i64 %t127, i64 %t128, i64 4)
  %t129 = load float, ptr %t126
  store float %t129, ptr %o41
  call void @mapal_gpu_free(ptr %d0)
  %t130 = load float, ptr %o41
  store float %t130, ptr %o42
  %t131 = load float, ptr %o42
  call void @mapal_print_f32(float %t131, i1 zeroext true)
  %t132 = getelementptr { ptr, i32 }, ptr %o44, i32 0, i32 1
  store i32 0, ptr %t132
  %t133 = getelementptr { ptr, i32 }, ptr %o44, i32 0, i32 1
  %t134 = load i32, ptr %t133
  %t135 = sext i32 %t134 to i64
  %t136 = getelementptr [4096 x float], ptr %o4, i64 0, i64 %t135
  %t137 = load i64, ptr %d2
  %t138 = mul i64 %t135, 4
  call void @mapal_gpu_d2h(ptr %t136, i64 %t137, i64 %t138, i64 4)
  %t139 = load float, ptr %t136
  store float %t139, ptr %o45
  call void @mapal_gpu_free(ptr %d2)
  %t140 = load float, ptr %o45
  store float %t140, ptr %o46
  %t141 = load float, ptr %o46
  call void @mapal_print_f32(float %t141, i1 zeroext true)
  %t142 = getelementptr { ptr, i32 }, ptr %o48, i32 0, i32 1
  store i32 0, ptr %t142
  %t143 = getelementptr { ptr, i32 }, ptr %o48, i32 0, i32 1
  %t144 = load i32, ptr %t143
  %t145 = sext i32 %t144 to i64
  %t146 = getelementptr [4096 x float], ptr %o5, i64 0, i64 %t145
  %t147 = load i64, ptr %d4
  %t148 = mul i64 %t145, 4
  call void @mapal_gpu_d2h(ptr %t146, i64 %t147, i64 %t148, i64 4)
  %t149 = load float, ptr %t146
  store float %t149, ptr %o49
  call void @mapal_gpu_free(ptr %d4)
  %t150 = load float, ptr %o49
  store float %t150, ptr %o50
  %t151 = load float, ptr %o50
  call void @mapal_print_f32(float %t151, i1 zeroext true)
  %t152 = getelementptr { ptr, i32 }, ptr %o52, i32 0, i32 1
  store i32 0, ptr %t152
  %t153 = getelementptr { ptr, i32 }, ptr %o52, i32 0, i32 1
  %t154 = load i32, ptr %t153
  %t155 = sext i32 %t154 to i64
  %t156 = getelementptr [4096 x float], ptr %o6, i64 0, i64 %t155
  %t157 = load i64, ptr %d6
  %t158 = mul i64 %t155, 4
  call void @mapal_gpu_d2h(ptr %t156, i64 %t157, i64 %t158, i64 4)
  %t159 = load float, ptr %t156
  store float %t159, ptr %o53
  call void @mapal_gpu_free(ptr %d6)
  %t160 = load float, ptr %o53
  store float %t160, ptr %o54
  %t161 = load float, ptr %o54
  call void @mapal_print_f32(float %t161, i1 zeroext true)
  %t162 = getelementptr { ptr, i32 }, ptr %o56, i32 0, i32 1
  store i32 0, ptr %t162
  %t163 = getelementptr { ptr, i32 }, ptr %o56, i32 0, i32 1
  %t164 = load i32, ptr %t163
  %t165 = sext i32 %t164 to i64
  %t166 = getelementptr [4096 x float], ptr %o9, i64 0, i64 %t165
  %t167 = load i64, ptr %d23
  %t168 = mul i64 %t165, 4
  call void @mapal_gpu_d2h_on(i32 1, ptr %t166, i64 %t167, i64 %t168, i64 4)
  %t169 = load float, ptr %t166
  store float %t169, ptr %o57
  %t170 = load float, ptr %o57
  store float %t170, ptr %o58
  %t171 = load float, ptr %o58
  call void @mapal_print_f32(float %t171, i1 zeroext true)
  %t172 = getelementptr { ptr, i32 }, ptr %o60, i32 0, i32 1
  store i32 4095, ptr %t172
  %t173 = getelementptr { ptr, i32 }, ptr %o60, i32 0, i32 1
  %t174 = load i32, ptr %t173
  %t175 = sext i32 %t174 to i64
  %t176 = getelementptr [4096 x float], ptr %o9, i64 0, i64 %t175
  %t177 = load i64, ptr %d23
  %t178 = mul i64 %t175, 4
  call void @mapal_gpu_d2h_on(i32 1, ptr %t176, i64 %t177, i64 %t178, i64 4)
  %t179 = load float, ptr %t176
  store float %t179, ptr %o61
  call void @mapal_gpu_free(ptr %d23)
  %t180 = load float, ptr %o61
  store float %t180, ptr %o62
  %t181 = load float, ptr %o62
  call void @mapal_print_f32(float %t181, i1 zeroext true)
  %t182 = getelementptr { ptr, i32 }, ptr %o64, i32 0, i32 1
  store i32 0, ptr %t182
  %t183 = getelementptr { ptr, i32 }, ptr %o64, i32 0, i32 1
  %t184 = load i32, ptr %t183
  %t185 = sext i32 %t184 to i64
  %t186 = getelementptr [4096 x float], ptr %o12, i64 0, i64 %t185
  %t187 = load i64, ptr %d25
  %t188 = mul i64 %t185, 4
  call void @mapal_gpu_d2h_on(i32 2, ptr %t186, i64 %t187, i64 %t188, i64 4)
  %t189 = load float, ptr %t186
  store float %t189, ptr %o65
  %t190 = load float, ptr %o65
  store float %t190, ptr %o66
  %t191 = load float, ptr %o66
  call void @mapal_print_f32(float %t191, i1 zeroext true)
  %t192 = getelementptr { ptr, i32 }, ptr %o68, i32 0, i32 1
  store i32 4095, ptr %t192
  %t193 = getelementptr { ptr, i32 }, ptr %o68, i32 0, i32 1
  %t194 = load i32, ptr %t193
  %t195 = sext i32 %t194 to i64
  %t196 = getelementptr [4096 x float], ptr %o12, i64 0, i64 %t195
  %t197 = load i64, ptr %d25
  %t198 = mul i64 %t195, 4
  call void @mapal_gpu_d2h_on(i32 2, ptr %t196, i64 %t197, i64 %t198, i64 4)
  %t199 = load float, ptr %t196
  store float %t199, ptr %o69
  call void @mapal_gpu_free(ptr %d25)
  %t200 = load float, ptr %o69
  store float %t200, ptr %o70
  %t201 = load float, ptr %o70
  call void @mapal_print_f32(float %t201, i1 zeroext true)
  call void @mapal_gpu_free(ptr %d0)
  call void @mapal_gpu_free(ptr %d2)
  call void @mapal_gpu_free(ptr %d4)
  call void @mapal_gpu_free(ptr %d6)
  call void @mapal_gpu_free(ptr %d23)
  call void @mapal_gpu_free(ptr %d28)
  call void @mapal_gpu_free(ptr %d25)
  call void @mapal_gpu_free(ptr %d43)
  call void @mapal_gpu_free(ptr %d54)
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

define internal float @fn11({ ptr, i32 } %arg) readonly nounwind willreturn {
entry:
  %o0 = alloca { ptr, i32 }
  %o1 = alloca float
  %o2 = alloca ptr
  %o3 = alloca i32
  %o4 = alloca { i32, i32 }
  %o5 = alloca i32
  %o6 = alloca { ptr, i32 }
  store { ptr, i32 } %arg, ptr %o0
  %t0 = getelementptr { ptr, i32 }, ptr %o0, i32 0, i32 0
  %t1 = load ptr, ptr %t0
  store ptr %t1, ptr %o2
  %t2 = getelementptr { ptr, i32 }, ptr %o0, i32 0, i32 1
  %t3 = load i32, ptr %t2
  store i32 %t3, ptr %o3
  %t4 = load ptr, ptr %o2
  %t5 = getelementptr { ptr, i32 }, ptr %o6, i32 0, i32 0
  store ptr %t4, ptr %t5
  %t6 = load i32, ptr %o3
  %t7 = getelementptr { i32, i32 }, ptr %o4, i32 0, i32 1
  store i32 %t6, ptr %t7
  %t8 = getelementptr { i32, i32 }, ptr %o4, i32 0, i32 0
  store i32 4095, ptr %t8
  %t9 = getelementptr { i32, i32 }, ptr %o4, i32 0, i32 0
  %t10 = load i32, ptr %t9
  %t11 = getelementptr { i32, i32 }, ptr %o4, i32 0, i32 1
  %t12 = load i32, ptr %t11
  %t13 = sub i32 %t10, %t12
  store i32 %t13, ptr %o5
  %t14 = load i32, ptr %o5
  %t15 = getelementptr { ptr, i32 }, ptr %o6, i32 0, i32 1
  store i32 %t14, ptr %t15
  %t16 = load ptr, ptr %o2
  %t17 = getelementptr { ptr, i32 }, ptr %o6, i32 0, i32 1
  %t18 = load i32, ptr %t17
  %t19 = sext i32 %t18 to i64
  %t20 = getelementptr [4096 x float], ptr %t16, i64 0, i64 %t19
  %t21 = load float, ptr %t20
  store float %t21, ptr %o1
  %t22 = load float, ptr %o1
  ret float %t22
}

define i32 @main() {
entry:
  call void @mapal_gpu_init(i32 3)
  call void @mapal_main()
  ret i32 0
}
