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
declare i64 @mapal_gpu_arena(ptr, i64)
declare void @mapal_gpu_launchv(i64, i32, i32, ptr, i32, i32, i32, i32)

define internal void @mapal_main() {
entry:
  %darena = alloca i64
  store i64 0, ptr %darena
  %o8 = alloca { ptr, ptr, ptr, ptr }
  %o11 = alloca { ptr, ptr, ptr, ptr }
  %o14 = alloca { ptr, ptr }
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
  %d3 = alloca i64
  store i64 0, ptr %d3
  %d6 = alloca i64
  store i64 0, ptr %d6
  %d9 = alloca i64
  store i64 0, ptr %d9
  %d27 = alloca i64
  store i64 0, ptr %d27
  %d30 = alloca i64
  store i64 0, ptr %d30
  %p33 = alloca [4 x i64]
  %d45 = alloca i64
  store i64 0, ptr %d45
  %d48 = alloca i64
  store i64 0, ptr %d48
  %p51 = alloca [4 x i64]
  %d60 = alloca i64
  store i64 0, ptr %d60
  %dv64 = alloca [2 x i64]
  %s73 = alloca float
  %s83 = alloca float
  %s93 = alloca float
  %s103 = alloca float
  %s113 = alloca float
  %s123 = alloca float
  %s133 = alloca float
  %s143 = alloca float
  %s153 = alloca float
  %s163 = alloca float
  %s173 = alloca float
  %s183 = alloca float
  %s193 = alloca float
  %s203 = alloca float
  %t2 = call i64 @mapal_gpu_arena(ptr %darena, i64 147456)
  %t1 = add i64 %t2, 0
  store i64 %t1, ptr %d0
  call void @mapal_gpu_gen(i64 2241086273371086914, i64 %t1, i64 4096)
  %t5 = call i64 @mapal_gpu_arena(ptr %darena, i64 147456)
  %t4 = add i64 %t5, 16384
  store i64 %t4, ptr %d3
  call void @mapal_gpu_gen(i64 -8319208597251102849, i64 %t4, i64 4096)
  %t8 = call i64 @mapal_gpu_arena(ptr %darena, i64 147456)
  %t7 = add i64 %t8, 32768
  store i64 %t7, ptr %d6
  call void @mapal_gpu_gen(i64 3388508959519476380, i64 %t7, i64 4096)
  %t11 = call i64 @mapal_gpu_arena(ptr %darena, i64 147456)
  %t10 = add i64 %t11, 49152
  store i64 %t10, ptr %d9
  call void @mapal_gpu_gen(i64 -7602686761790676223, i64 %t10, i64 4096)
  %t12 = getelementptr { ptr, ptr, ptr, ptr }, ptr %o8, i32 0, i32 3
  store ptr null, ptr %t12
  %t13 = getelementptr { ptr, ptr, ptr, ptr }, ptr %o11, i32 0, i32 3
  store ptr null, ptr %t13
  %t14 = getelementptr { ptr, ptr }, ptr %o14, i32 0, i32 1
  store ptr null, ptr %t14
  %t15 = getelementptr { ptr, ptr, ptr, ptr }, ptr %o8, i32 0, i32 0
  store ptr null, ptr %t15
  %t16 = getelementptr { ptr, ptr, ptr, ptr }, ptr %o11, i32 0, i32 0
  store ptr null, ptr %t16
  %t17 = getelementptr { ptr, ptr, ptr, ptr }, ptr %o8, i32 0, i32 1
  store ptr null, ptr %t17
  %t18 = getelementptr { ptr, i32 }, ptr %o40, i32 0, i32 0
  store ptr null, ptr %t18
  %t19 = getelementptr { ptr, ptr, ptr, ptr }, ptr %o8, i32 0, i32 2
  store ptr null, ptr %t19
  %t20 = getelementptr { ptr, i32 }, ptr %o44, i32 0, i32 0
  store ptr null, ptr %t20
  %t21 = getelementptr { ptr, ptr, ptr, ptr }, ptr %o11, i32 0, i32 1
  store ptr null, ptr %t21
  %t22 = getelementptr { ptr, i32 }, ptr %o48, i32 0, i32 0
  store ptr null, ptr %t22
  %t23 = getelementptr { ptr, ptr, ptr, ptr }, ptr %o11, i32 0, i32 2
  store ptr null, ptr %t23
  %t24 = getelementptr { ptr, i32 }, ptr %o52, i32 0, i32 0
  store ptr null, ptr %t24
  %t25 = load i64, ptr %d0
  %t26 = load i64, ptr %d3
  %t29 = call i64 @mapal_gpu_arena(ptr %darena, i64 147456)
  %t28 = add i64 %t29, 81920
  store i64 %t28, ptr %d27
  %t32 = call i64 @mapal_gpu_arena(ptr %darena, i64 147456)
  %t31 = add i64 %t32, 65536
  store i64 %t31, ptr %d30
  %t34 = getelementptr [4 x i64], ptr %p33, i64 0, i64 0
  store i64 %t25, ptr %t34
  %t35 = getelementptr [4 x i64], ptr %p33, i64 0, i64 1
  store i64 %t26, ptr %t35
  %t36 = getelementptr [4 x i64], ptr %p33, i64 0, i64 2
  store i64 %t28, ptr %t36
  %t37 = getelementptr [4 x i64], ptr %p33, i64 0, i64 3
  store i64 %t31, ptr %t37
  call void @mapal_gpu_launchv(i64 -5910159045450748452, i32 0, i32 4, ptr %p33, i32 4, i32 2, i32 32, i32 1536)
  %t38 = getelementptr { ptr, ptr }, ptr %o14, i32 0, i32 0
  store ptr null, ptr %t38
  %t39 = getelementptr { ptr, i32 }, ptr %o56, i32 0, i32 0
  store ptr null, ptr %t39
  %t40 = getelementptr { ptr, i32 }, ptr %o60, i32 0, i32 0
  store ptr null, ptr %t40
  %t41 = getelementptr { ptr, i32 }, ptr %o32, i32 0, i32 0
  store ptr null, ptr %t41
  %t42 = getelementptr { ptr, i32 }, ptr %o36, i32 0, i32 0
  store ptr null, ptr %t42
  %t43 = load i64, ptr %d6
  %t44 = load i64, ptr %d9
  %t47 = call i64 @mapal_gpu_arena(ptr %darena, i64 147456)
  %t46 = add i64 %t47, 114688
  store i64 %t46, ptr %d45
  %t50 = call i64 @mapal_gpu_arena(ptr %darena, i64 147456)
  %t49 = add i64 %t50, 98304
  store i64 %t49, ptr %d48
  %t52 = getelementptr [4 x i64], ptr %p51, i64 0, i64 0
  store i64 %t43, ptr %t52
  %t53 = getelementptr [4 x i64], ptr %p51, i64 0, i64 1
  store i64 %t44, ptr %t53
  %t54 = getelementptr [4 x i64], ptr %p51, i64 0, i64 2
  store i64 %t46, ptr %t54
  %t55 = getelementptr [4 x i64], ptr %p51, i64 0, i64 3
  store i64 %t49, ptr %t55
  call void @mapal_gpu_launchv(i64 8546537189269670622, i32 0, i32 4, ptr %p51, i32 4, i32 2, i32 32, i32 1536)
  %t56 = getelementptr { ptr, i32 }, ptr %o64, i32 0, i32 0
  store ptr null, ptr %t56
  %t57 = getelementptr { ptr, i32 }, ptr %o68, i32 0, i32 0
  store ptr null, ptr %t57
  %t58 = getelementptr { ptr, i32 }, ptr %o24, i32 0, i32 0
  store ptr null, ptr %t58
  %t59 = getelementptr { ptr, i32 }, ptr %o28, i32 0, i32 0
  store ptr null, ptr %t59
  %t62 = call i64 @mapal_gpu_arena(ptr %darena, i64 147456)
  %t61 = add i64 %t62, 131072
  store i64 %t61, ptr %d60
  %t63 = load i64, ptr %d27
  %t65 = getelementptr [2 x i64], ptr %dv64, i64 0, i64 0
  store i64 %t61, ptr %t65
  %t66 = getelementptr [2 x i64], ptr %dv64, i64 0, i64 1
  store i64 %t63, ptr %t66
  call void @mapal_gpu_launchv(i64 4010187978629625089, i32 0, i32 2, ptr %dv64, i32 16, i32 1, i32 256, i32 0)
  %t67 = getelementptr { ptr, i32 }, ptr %o16, i32 0, i32 0
  store ptr null, ptr %t67
  %t68 = getelementptr { ptr, i32 }, ptr %o20, i32 0, i32 0
  store ptr null, ptr %t68
  %t69 = getelementptr { ptr, i32 }, ptr %o16, i32 0, i32 1
  store i32 0, ptr %t69
  %t70 = getelementptr { ptr, i32 }, ptr %o16, i32 0, i32 1
  %t71 = load i32, ptr %t70
  %t72 = sext i32 %t71 to i64
  %t74 = load i64, ptr %d60
  %t75 = mul i64 %t72, 4
  call void @mapal_gpu_d2h(ptr %s73, i64 %t74, i64 %t75, i64 4)
  %t76 = load float, ptr %s73
  store float %t76, ptr %o17
  %t77 = load float, ptr %o17
  store float %t77, ptr %o18
  %t78 = load float, ptr %o18
  call void @mapal_print_f32(float %t78, i1 zeroext true)
  %t79 = getelementptr { ptr, i32 }, ptr %o20, i32 0, i32 1
  store i32 4095, ptr %t79
  %t80 = getelementptr { ptr, i32 }, ptr %o20, i32 0, i32 1
  %t81 = load i32, ptr %t80
  %t82 = sext i32 %t81 to i64
  %t84 = load i64, ptr %d60
  %t85 = mul i64 %t82, 4
  call void @mapal_gpu_d2h(ptr %s83, i64 %t84, i64 %t85, i64 4)
  %t86 = load float, ptr %s83
  store float %t86, ptr %o21
  %t87 = load float, ptr %o21
  store float %t87, ptr %o22
  %t88 = load float, ptr %o22
  call void @mapal_print_f32(float %t88, i1 zeroext true)
  %t89 = getelementptr { ptr, i32 }, ptr %o24, i32 0, i32 1
  store i32 0, ptr %t89
  %t90 = getelementptr { ptr, i32 }, ptr %o24, i32 0, i32 1
  %t91 = load i32, ptr %t90
  %t92 = sext i32 %t91 to i64
  %t94 = load i64, ptr %d48
  %t95 = mul i64 %t92, 4
  call void @mapal_gpu_d2h(ptr %s93, i64 %t94, i64 %t95, i64 4)
  %t96 = load float, ptr %s93
  store float %t96, ptr %o25
  %t97 = load float, ptr %o25
  store float %t97, ptr %o26
  %t98 = load float, ptr %o26
  call void @mapal_print_f32(float %t98, i1 zeroext true)
  %t99 = getelementptr { ptr, i32 }, ptr %o28, i32 0, i32 1
  store i32 4095, ptr %t99
  %t100 = getelementptr { ptr, i32 }, ptr %o28, i32 0, i32 1
  %t101 = load i32, ptr %t100
  %t102 = sext i32 %t101 to i64
  %t104 = load i64, ptr %d48
  %t105 = mul i64 %t102, 4
  call void @mapal_gpu_d2h(ptr %s103, i64 %t104, i64 %t105, i64 4)
  %t106 = load float, ptr %s103
  store float %t106, ptr %o29
  %t107 = load float, ptr %o29
  store float %t107, ptr %o30
  %t108 = load float, ptr %o30
  call void @mapal_print_f32(float %t108, i1 zeroext true)
  %t109 = getelementptr { ptr, i32 }, ptr %o32, i32 0, i32 1
  store i32 0, ptr %t109
  %t110 = getelementptr { ptr, i32 }, ptr %o32, i32 0, i32 1
  %t111 = load i32, ptr %t110
  %t112 = sext i32 %t111 to i64
  %t114 = load i64, ptr %d30
  %t115 = mul i64 %t112, 4
  call void @mapal_gpu_d2h(ptr %s113, i64 %t114, i64 %t115, i64 4)
  %t116 = load float, ptr %s113
  store float %t116, ptr %o33
  %t117 = load float, ptr %o33
  store float %t117, ptr %o34
  %t118 = load float, ptr %o34
  call void @mapal_print_f32(float %t118, i1 zeroext true)
  %t119 = getelementptr { ptr, i32 }, ptr %o36, i32 0, i32 1
  store i32 4095, ptr %t119
  %t120 = getelementptr { ptr, i32 }, ptr %o36, i32 0, i32 1
  %t121 = load i32, ptr %t120
  %t122 = sext i32 %t121 to i64
  %t124 = load i64, ptr %d30
  %t125 = mul i64 %t122, 4
  call void @mapal_gpu_d2h(ptr %s123, i64 %t124, i64 %t125, i64 4)
  %t126 = load float, ptr %s123
  store float %t126, ptr %o37
  %t127 = load float, ptr %o37
  store float %t127, ptr %o38
  %t128 = load float, ptr %o38
  call void @mapal_print_f32(float %t128, i1 zeroext true)
  %t129 = getelementptr { ptr, i32 }, ptr %o40, i32 0, i32 1
  store i32 0, ptr %t129
  %t130 = getelementptr { ptr, i32 }, ptr %o40, i32 0, i32 1
  %t131 = load i32, ptr %t130
  %t132 = sext i32 %t131 to i64
  %t134 = load i64, ptr %d0
  %t135 = mul i64 %t132, 4
  call void @mapal_gpu_d2h(ptr %s133, i64 %t134, i64 %t135, i64 4)
  %t136 = load float, ptr %s133
  store float %t136, ptr %o41
  %t137 = load float, ptr %o41
  store float %t137, ptr %o42
  %t138 = load float, ptr %o42
  call void @mapal_print_f32(float %t138, i1 zeroext true)
  %t139 = getelementptr { ptr, i32 }, ptr %o44, i32 0, i32 1
  store i32 0, ptr %t139
  %t140 = getelementptr { ptr, i32 }, ptr %o44, i32 0, i32 1
  %t141 = load i32, ptr %t140
  %t142 = sext i32 %t141 to i64
  %t144 = load i64, ptr %d3
  %t145 = mul i64 %t142, 4
  call void @mapal_gpu_d2h(ptr %s143, i64 %t144, i64 %t145, i64 4)
  %t146 = load float, ptr %s143
  store float %t146, ptr %o45
  %t147 = load float, ptr %o45
  store float %t147, ptr %o46
  %t148 = load float, ptr %o46
  call void @mapal_print_f32(float %t148, i1 zeroext true)
  %t149 = getelementptr { ptr, i32 }, ptr %o48, i32 0, i32 1
  store i32 0, ptr %t149
  %t150 = getelementptr { ptr, i32 }, ptr %o48, i32 0, i32 1
  %t151 = load i32, ptr %t150
  %t152 = sext i32 %t151 to i64
  %t154 = load i64, ptr %d6
  %t155 = mul i64 %t152, 4
  call void @mapal_gpu_d2h(ptr %s153, i64 %t154, i64 %t155, i64 4)
  %t156 = load float, ptr %s153
  store float %t156, ptr %o49
  %t157 = load float, ptr %o49
  store float %t157, ptr %o50
  %t158 = load float, ptr %o50
  call void @mapal_print_f32(float %t158, i1 zeroext true)
  %t159 = getelementptr { ptr, i32 }, ptr %o52, i32 0, i32 1
  store i32 0, ptr %t159
  %t160 = getelementptr { ptr, i32 }, ptr %o52, i32 0, i32 1
  %t161 = load i32, ptr %t160
  %t162 = sext i32 %t161 to i64
  %t164 = load i64, ptr %d9
  %t165 = mul i64 %t162, 4
  call void @mapal_gpu_d2h(ptr %s163, i64 %t164, i64 %t165, i64 4)
  %t166 = load float, ptr %s163
  store float %t166, ptr %o53
  %t167 = load float, ptr %o53
  store float %t167, ptr %o54
  %t168 = load float, ptr %o54
  call void @mapal_print_f32(float %t168, i1 zeroext true)
  %t169 = getelementptr { ptr, i32 }, ptr %o56, i32 0, i32 1
  store i32 0, ptr %t169
  %t170 = getelementptr { ptr, i32 }, ptr %o56, i32 0, i32 1
  %t171 = load i32, ptr %t170
  %t172 = sext i32 %t171 to i64
  %t174 = load i64, ptr %d27
  %t175 = mul i64 %t172, 4
  call void @mapal_gpu_d2h(ptr %s173, i64 %t174, i64 %t175, i64 4)
  %t176 = load float, ptr %s173
  store float %t176, ptr %o57
  %t177 = load float, ptr %o57
  store float %t177, ptr %o58
  %t178 = load float, ptr %o58
  call void @mapal_print_f32(float %t178, i1 zeroext true)
  %t179 = getelementptr { ptr, i32 }, ptr %o60, i32 0, i32 1
  store i32 4095, ptr %t179
  %t180 = getelementptr { ptr, i32 }, ptr %o60, i32 0, i32 1
  %t181 = load i32, ptr %t180
  %t182 = sext i32 %t181 to i64
  %t184 = load i64, ptr %d27
  %t185 = mul i64 %t182, 4
  call void @mapal_gpu_d2h(ptr %s183, i64 %t184, i64 %t185, i64 4)
  %t186 = load float, ptr %s183
  store float %t186, ptr %o61
  %t187 = load float, ptr %o61
  store float %t187, ptr %o62
  %t188 = load float, ptr %o62
  call void @mapal_print_f32(float %t188, i1 zeroext true)
  %t189 = getelementptr { ptr, i32 }, ptr %o64, i32 0, i32 1
  store i32 0, ptr %t189
  %t190 = getelementptr { ptr, i32 }, ptr %o64, i32 0, i32 1
  %t191 = load i32, ptr %t190
  %t192 = sext i32 %t191 to i64
  %t194 = load i64, ptr %d45
  %t195 = mul i64 %t192, 4
  call void @mapal_gpu_d2h(ptr %s193, i64 %t194, i64 %t195, i64 4)
  %t196 = load float, ptr %s193
  store float %t196, ptr %o65
  %t197 = load float, ptr %o65
  store float %t197, ptr %o66
  %t198 = load float, ptr %o66
  call void @mapal_print_f32(float %t198, i1 zeroext true)
  %t199 = getelementptr { ptr, i32 }, ptr %o68, i32 0, i32 1
  store i32 4095, ptr %t199
  %t200 = getelementptr { ptr, i32 }, ptr %o68, i32 0, i32 1
  %t201 = load i32, ptr %t200
  %t202 = sext i32 %t201 to i64
  %t204 = load i64, ptr %d45
  %t205 = mul i64 %t202, 4
  call void @mapal_gpu_d2h(ptr %s203, i64 %t204, i64 %t205, i64 4)
  %t206 = load float, ptr %s203
  store float %t206, ptr %o69
  %t207 = load float, ptr %o69
  store float %t207, ptr %o70
  %t208 = load float, ptr %o70
  call void @mapal_print_f32(float %t208, i1 zeroext true)
  call void @mapal_gpu_free(ptr %darena)
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
  call void @mapal_gpu_init(i32 1)
  call void @mapal_main()
  ret i32 0
}
