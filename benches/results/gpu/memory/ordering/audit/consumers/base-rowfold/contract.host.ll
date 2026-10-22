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
  %o14 = alloca { ptr, ptr, ptr }
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
  %d28 = alloca i64
  store i64 0, ptr %d28
  %d31 = alloca i64
  store i64 0, ptr %d31
  %p34 = alloca [4 x i64]
  %d46 = alloca i64
  store i64 0, ptr %d46
  %d49 = alloca i64
  store i64 0, ptr %d49
  %p52 = alloca [4 x i64]
  %d62 = alloca i64
  store i64 0, ptr %d62
  %s71 = alloca float
  %s81 = alloca float
  %s91 = alloca float
  %s101 = alloca float
  %s111 = alloca float
  %s121 = alloca float
  %s131 = alloca float
  %s141 = alloca float
  %s151 = alloca float
  %s161 = alloca float
  %s171 = alloca float
  %s181 = alloca float
  %s191 = alloca float
  %s201 = alloca float
  %t2 = call i64 @mapal_gpu_arena(ptr %darena, i64 131328)
  %t1 = add i64 %t2, 0
  store i64 %t1, ptr %d0
  call void @mapal_gpu_gen(i64 2241086273371086914, i64 %t1, i64 4096)
  %t5 = call i64 @mapal_gpu_arena(ptr %darena, i64 131328)
  %t4 = add i64 %t5, 16384
  store i64 %t4, ptr %d3
  call void @mapal_gpu_gen(i64 -8319208597251102849, i64 %t4, i64 4096)
  %t8 = call i64 @mapal_gpu_arena(ptr %darena, i64 131328)
  %t7 = add i64 %t8, 32768
  store i64 %t7, ptr %d6
  call void @mapal_gpu_gen(i64 3388508959519476380, i64 %t7, i64 4096)
  %t11 = call i64 @mapal_gpu_arena(ptr %darena, i64 131328)
  %t10 = add i64 %t11, 49152
  store i64 %t10, ptr %d9
  call void @mapal_gpu_gen(i64 -7602686761790676223, i64 %t10, i64 4096)
  %t12 = getelementptr { ptr, ptr, ptr, ptr }, ptr %o8, i32 0, i32 3
  store ptr null, ptr %t12
  %t13 = getelementptr { ptr, ptr, ptr, ptr }, ptr %o11, i32 0, i32 3
  store ptr null, ptr %t13
  %t14 = getelementptr { ptr, ptr, ptr, ptr }, ptr %o8, i32 0, i32 0
  store ptr null, ptr %t14
  %t15 = getelementptr { ptr, ptr, ptr, ptr }, ptr %o11, i32 0, i32 0
  store ptr null, ptr %t15
  %t16 = getelementptr { ptr, ptr, ptr }, ptr %o14, i32 0, i32 0
  store ptr null, ptr %t16
  %t17 = getelementptr { ptr, ptr, ptr }, ptr %o14, i32 0, i32 2
  store ptr null, ptr %t17
  %t18 = getelementptr { ptr, ptr, ptr, ptr }, ptr %o8, i32 0, i32 1
  store ptr null, ptr %t18
  %t19 = getelementptr { ptr, i32 }, ptr %o40, i32 0, i32 0
  store ptr null, ptr %t19
  %t20 = getelementptr { ptr, ptr, ptr, ptr }, ptr %o8, i32 0, i32 2
  store ptr null, ptr %t20
  %t21 = getelementptr { ptr, i32 }, ptr %o44, i32 0, i32 0
  store ptr null, ptr %t21
  %t22 = getelementptr { ptr, ptr, ptr, ptr }, ptr %o11, i32 0, i32 1
  store ptr null, ptr %t22
  %t23 = getelementptr { ptr, i32 }, ptr %o48, i32 0, i32 0
  store ptr null, ptr %t23
  %t24 = getelementptr { ptr, ptr, ptr, ptr }, ptr %o11, i32 0, i32 2
  store ptr null, ptr %t24
  %t25 = getelementptr { ptr, i32 }, ptr %o52, i32 0, i32 0
  store ptr null, ptr %t25
  %t26 = load i64, ptr %d0
  %t27 = load i64, ptr %d3
  %t30 = call i64 @mapal_gpu_arena(ptr %darena, i64 131328)
  %t29 = add i64 %t30, 81920
  store i64 %t29, ptr %d28
  %t33 = call i64 @mapal_gpu_arena(ptr %darena, i64 131328)
  %t32 = add i64 %t33, 65536
  store i64 %t32, ptr %d31
  %t35 = getelementptr [4 x i64], ptr %p34, i64 0, i64 0
  store i64 %t26, ptr %t35
  %t36 = getelementptr [4 x i64], ptr %p34, i64 0, i64 1
  store i64 %t27, ptr %t36
  %t37 = getelementptr [4 x i64], ptr %p34, i64 0, i64 2
  store i64 %t29, ptr %t37
  %t38 = getelementptr [4 x i64], ptr %p34, i64 0, i64 3
  store i64 %t32, ptr %t38
  call void @mapal_gpu_launchv(i64 -5869913981016682307, i32 0, i32 4, ptr %p34, i32 4, i32 2, i32 32, i32 1664)
  %t39 = getelementptr { ptr, ptr, ptr }, ptr %o14, i32 0, i32 1
  store ptr null, ptr %t39
  %t40 = getelementptr { ptr, i32 }, ptr %o56, i32 0, i32 0
  store ptr null, ptr %t40
  %t41 = getelementptr { ptr, i32 }, ptr %o60, i32 0, i32 0
  store ptr null, ptr %t41
  %t42 = getelementptr { ptr, i32 }, ptr %o32, i32 0, i32 0
  store ptr null, ptr %t42
  %t43 = getelementptr { ptr, i32 }, ptr %o36, i32 0, i32 0
  store ptr null, ptr %t43
  %t44 = load i64, ptr %d6
  %t45 = load i64, ptr %d9
  %t48 = call i64 @mapal_gpu_arena(ptr %darena, i64 131328)
  %t47 = add i64 %t48, 114688
  store i64 %t47, ptr %d46
  %t51 = call i64 @mapal_gpu_arena(ptr %darena, i64 131328)
  %t50 = add i64 %t51, 98304
  store i64 %t50, ptr %d49
  %t53 = getelementptr [4 x i64], ptr %p52, i64 0, i64 0
  store i64 %t44, ptr %t53
  %t54 = getelementptr [4 x i64], ptr %p52, i64 0, i64 1
  store i64 %t45, ptr %t54
  %t55 = getelementptr [4 x i64], ptr %p52, i64 0, i64 2
  store i64 %t47, ptr %t55
  %t56 = getelementptr [4 x i64], ptr %p52, i64 0, i64 3
  store i64 %t50, ptr %t56
  call void @mapal_gpu_launchv(i64 -615245021611174823, i32 0, i32 4, ptr %p52, i32 4, i32 2, i32 32, i32 1664)
  %t57 = getelementptr { ptr, i32 }, ptr %o64, i32 0, i32 0
  store ptr null, ptr %t57
  %t58 = getelementptr { ptr, i32 }, ptr %o68, i32 0, i32 0
  store ptr null, ptr %t58
  %t59 = getelementptr { ptr, i32 }, ptr %o24, i32 0, i32 0
  store ptr null, ptr %t59
  %t60 = getelementptr { ptr, i32 }, ptr %o28, i32 0, i32 0
  store ptr null, ptr %t60
  %t61 = load i64, ptr %d28
  %t64 = call i64 @mapal_gpu_arena(ptr %darena, i64 131328)
  %t63 = add i64 %t64, 131072
  store i64 %t63, ptr %d62
  call void @mapal_gpu_launch(i64 4590226593432883463, i64 %t61, i64 %t61, i64 %t63, i32 2, i32 1, i32 32, i32 0)
  %t65 = getelementptr { ptr, i32 }, ptr %o16, i32 0, i32 0
  store ptr null, ptr %t65
  %t66 = getelementptr { ptr, i32 }, ptr %o20, i32 0, i32 0
  store ptr null, ptr %t66
  %t67 = getelementptr { ptr, i32 }, ptr %o16, i32 0, i32 1
  store i32 0, ptr %t67
  %t68 = getelementptr { ptr, i32 }, ptr %o16, i32 0, i32 1
  %t69 = load i32, ptr %t68
  %t70 = sext i32 %t69 to i64
  %t72 = load i64, ptr %d62
  %t73 = mul i64 %t70, 4
  call void @mapal_gpu_d2h(ptr %s71, i64 %t72, i64 %t73, i64 4)
  %t74 = load float, ptr %s71
  store float %t74, ptr %o17
  %t75 = load float, ptr %o17
  store float %t75, ptr %o18
  %t76 = load float, ptr %o18
  call void @mapal_print_f32(float %t76, i1 zeroext true)
  %t77 = getelementptr { ptr, i32 }, ptr %o20, i32 0, i32 1
  store i32 63, ptr %t77
  %t78 = getelementptr { ptr, i32 }, ptr %o20, i32 0, i32 1
  %t79 = load i32, ptr %t78
  %t80 = sext i32 %t79 to i64
  %t82 = load i64, ptr %d62
  %t83 = mul i64 %t80, 4
  call void @mapal_gpu_d2h(ptr %s81, i64 %t82, i64 %t83, i64 4)
  %t84 = load float, ptr %s81
  store float %t84, ptr %o21
  %t85 = load float, ptr %o21
  store float %t85, ptr %o22
  %t86 = load float, ptr %o22
  call void @mapal_print_f32(float %t86, i1 zeroext true)
  %t87 = getelementptr { ptr, i32 }, ptr %o24, i32 0, i32 1
  store i32 0, ptr %t87
  %t88 = getelementptr { ptr, i32 }, ptr %o24, i32 0, i32 1
  %t89 = load i32, ptr %t88
  %t90 = sext i32 %t89 to i64
  %t92 = load i64, ptr %d49
  %t93 = mul i64 %t90, 4
  call void @mapal_gpu_d2h(ptr %s91, i64 %t92, i64 %t93, i64 4)
  %t94 = load float, ptr %s91
  store float %t94, ptr %o25
  %t95 = load float, ptr %o25
  store float %t95, ptr %o26
  %t96 = load float, ptr %o26
  call void @mapal_print_f32(float %t96, i1 zeroext true)
  %t97 = getelementptr { ptr, i32 }, ptr %o28, i32 0, i32 1
  store i32 4095, ptr %t97
  %t98 = getelementptr { ptr, i32 }, ptr %o28, i32 0, i32 1
  %t99 = load i32, ptr %t98
  %t100 = sext i32 %t99 to i64
  %t102 = load i64, ptr %d49
  %t103 = mul i64 %t100, 4
  call void @mapal_gpu_d2h(ptr %s101, i64 %t102, i64 %t103, i64 4)
  %t104 = load float, ptr %s101
  store float %t104, ptr %o29
  %t105 = load float, ptr %o29
  store float %t105, ptr %o30
  %t106 = load float, ptr %o30
  call void @mapal_print_f32(float %t106, i1 zeroext true)
  %t107 = getelementptr { ptr, i32 }, ptr %o32, i32 0, i32 1
  store i32 0, ptr %t107
  %t108 = getelementptr { ptr, i32 }, ptr %o32, i32 0, i32 1
  %t109 = load i32, ptr %t108
  %t110 = sext i32 %t109 to i64
  %t112 = load i64, ptr %d31
  %t113 = mul i64 %t110, 4
  call void @mapal_gpu_d2h(ptr %s111, i64 %t112, i64 %t113, i64 4)
  %t114 = load float, ptr %s111
  store float %t114, ptr %o33
  %t115 = load float, ptr %o33
  store float %t115, ptr %o34
  %t116 = load float, ptr %o34
  call void @mapal_print_f32(float %t116, i1 zeroext true)
  %t117 = getelementptr { ptr, i32 }, ptr %o36, i32 0, i32 1
  store i32 4095, ptr %t117
  %t118 = getelementptr { ptr, i32 }, ptr %o36, i32 0, i32 1
  %t119 = load i32, ptr %t118
  %t120 = sext i32 %t119 to i64
  %t122 = load i64, ptr %d31
  %t123 = mul i64 %t120, 4
  call void @mapal_gpu_d2h(ptr %s121, i64 %t122, i64 %t123, i64 4)
  %t124 = load float, ptr %s121
  store float %t124, ptr %o37
  %t125 = load float, ptr %o37
  store float %t125, ptr %o38
  %t126 = load float, ptr %o38
  call void @mapal_print_f32(float %t126, i1 zeroext true)
  %t127 = getelementptr { ptr, i32 }, ptr %o40, i32 0, i32 1
  store i32 0, ptr %t127
  %t128 = getelementptr { ptr, i32 }, ptr %o40, i32 0, i32 1
  %t129 = load i32, ptr %t128
  %t130 = sext i32 %t129 to i64
  %t132 = load i64, ptr %d0
  %t133 = mul i64 %t130, 4
  call void @mapal_gpu_d2h(ptr %s131, i64 %t132, i64 %t133, i64 4)
  %t134 = load float, ptr %s131
  store float %t134, ptr %o41
  %t135 = load float, ptr %o41
  store float %t135, ptr %o42
  %t136 = load float, ptr %o42
  call void @mapal_print_f32(float %t136, i1 zeroext true)
  %t137 = getelementptr { ptr, i32 }, ptr %o44, i32 0, i32 1
  store i32 0, ptr %t137
  %t138 = getelementptr { ptr, i32 }, ptr %o44, i32 0, i32 1
  %t139 = load i32, ptr %t138
  %t140 = sext i32 %t139 to i64
  %t142 = load i64, ptr %d3
  %t143 = mul i64 %t140, 4
  call void @mapal_gpu_d2h(ptr %s141, i64 %t142, i64 %t143, i64 4)
  %t144 = load float, ptr %s141
  store float %t144, ptr %o45
  %t145 = load float, ptr %o45
  store float %t145, ptr %o46
  %t146 = load float, ptr %o46
  call void @mapal_print_f32(float %t146, i1 zeroext true)
  %t147 = getelementptr { ptr, i32 }, ptr %o48, i32 0, i32 1
  store i32 0, ptr %t147
  %t148 = getelementptr { ptr, i32 }, ptr %o48, i32 0, i32 1
  %t149 = load i32, ptr %t148
  %t150 = sext i32 %t149 to i64
  %t152 = load i64, ptr %d6
  %t153 = mul i64 %t150, 4
  call void @mapal_gpu_d2h(ptr %s151, i64 %t152, i64 %t153, i64 4)
  %t154 = load float, ptr %s151
  store float %t154, ptr %o49
  %t155 = load float, ptr %o49
  store float %t155, ptr %o50
  %t156 = load float, ptr %o50
  call void @mapal_print_f32(float %t156, i1 zeroext true)
  %t157 = getelementptr { ptr, i32 }, ptr %o52, i32 0, i32 1
  store i32 0, ptr %t157
  %t158 = getelementptr { ptr, i32 }, ptr %o52, i32 0, i32 1
  %t159 = load i32, ptr %t158
  %t160 = sext i32 %t159 to i64
  %t162 = load i64, ptr %d9
  %t163 = mul i64 %t160, 4
  call void @mapal_gpu_d2h(ptr %s161, i64 %t162, i64 %t163, i64 4)
  %t164 = load float, ptr %s161
  store float %t164, ptr %o53
  %t165 = load float, ptr %o53
  store float %t165, ptr %o54
  %t166 = load float, ptr %o54
  call void @mapal_print_f32(float %t166, i1 zeroext true)
  %t167 = getelementptr { ptr, i32 }, ptr %o56, i32 0, i32 1
  store i32 0, ptr %t167
  %t168 = getelementptr { ptr, i32 }, ptr %o56, i32 0, i32 1
  %t169 = load i32, ptr %t168
  %t170 = sext i32 %t169 to i64
  %t172 = load i64, ptr %d28
  %t173 = mul i64 %t170, 4
  call void @mapal_gpu_d2h(ptr %s171, i64 %t172, i64 %t173, i64 4)
  %t174 = load float, ptr %s171
  store float %t174, ptr %o57
  %t175 = load float, ptr %o57
  store float %t175, ptr %o58
  %t176 = load float, ptr %o58
  call void @mapal_print_f32(float %t176, i1 zeroext true)
  %t177 = getelementptr { ptr, i32 }, ptr %o60, i32 0, i32 1
  store i32 4095, ptr %t177
  %t178 = getelementptr { ptr, i32 }, ptr %o60, i32 0, i32 1
  %t179 = load i32, ptr %t178
  %t180 = sext i32 %t179 to i64
  %t182 = load i64, ptr %d28
  %t183 = mul i64 %t180, 4
  call void @mapal_gpu_d2h(ptr %s181, i64 %t182, i64 %t183, i64 4)
  %t184 = load float, ptr %s181
  store float %t184, ptr %o61
  %t185 = load float, ptr %o61
  store float %t185, ptr %o62
  %t186 = load float, ptr %o62
  call void @mapal_print_f32(float %t186, i1 zeroext true)
  %t187 = getelementptr { ptr, i32 }, ptr %o64, i32 0, i32 1
  store i32 0, ptr %t187
  %t188 = getelementptr { ptr, i32 }, ptr %o64, i32 0, i32 1
  %t189 = load i32, ptr %t188
  %t190 = sext i32 %t189 to i64
  %t192 = load i64, ptr %d46
  %t193 = mul i64 %t190, 4
  call void @mapal_gpu_d2h(ptr %s191, i64 %t192, i64 %t193, i64 4)
  %t194 = load float, ptr %s191
  store float %t194, ptr %o65
  %t195 = load float, ptr %o65
  store float %t195, ptr %o66
  %t196 = load float, ptr %o66
  call void @mapal_print_f32(float %t196, i1 zeroext true)
  %t197 = getelementptr { ptr, i32 }, ptr %o68, i32 0, i32 1
  store i32 4095, ptr %t197
  %t198 = getelementptr { ptr, i32 }, ptr %o68, i32 0, i32 1
  %t199 = load i32, ptr %t198
  %t200 = sext i32 %t199 to i64
  %t202 = load i64, ptr %d46
  %t203 = mul i64 %t200, 4
  call void @mapal_gpu_d2h(ptr %s201, i64 %t202, i64 %t203, i64 4)
  %t204 = load float, ptr %s201
  store float %t204, ptr %o69
  %t205 = load float, ptr %o69
  store float %t205, ptr %o70
  %t206 = load float, ptr %o70
  call void @mapal_print_f32(float %t206, i1 zeroext true)
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
  call void @mapal_gpu_init(i32 1)
  call void @mapal_main()
  ret i32 0
}
