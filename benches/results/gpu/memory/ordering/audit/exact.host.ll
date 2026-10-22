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
  %o14 = alloca { ptr, i32 }
  %o15 = alloca float
  %o16 = alloca float
  %o18 = alloca { ptr, i32 }
  %o19 = alloca float
  %o20 = alloca float
  %o22 = alloca { ptr, i32 }
  %o23 = alloca float
  %o24 = alloca float
  %o26 = alloca { ptr, i32 }
  %o27 = alloca float
  %o28 = alloca float
  %o30 = alloca { ptr, i32 }
  %o31 = alloca float
  %o32 = alloca float
  %o34 = alloca { ptr, i32 }
  %o35 = alloca float
  %o36 = alloca float
  %o38 = alloca { ptr, i32 }
  %o39 = alloca float
  %o40 = alloca float
  %o42 = alloca { ptr, i32 }
  %o43 = alloca float
  %o44 = alloca float
  %d0 = alloca i64
  store i64 0, ptr %d0
  %d2 = alloca i64
  store i64 0, ptr %d2
  %d4 = alloca i64
  store i64 0, ptr %d4
  %d6 = alloca i64
  store i64 0, ptr %d6
  %d22 = alloca i64
  store i64 0, ptr %d22
  %d24 = alloca i64
  store i64 0, ptr %d24
  %d26 = alloca i64
  store i64 0, ptr %d26
  %p28 = alloca [4 x i64]
  %d37 = alloca i64
  store i64 0, ptr %d37
  %p39 = alloca [4 x i64]
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
  %t10 = getelementptr { ptr, ptr, ptr, ptr }, ptr %o8, i32 0, i32 0
  store ptr %o7, ptr %t10
  %t11 = getelementptr { ptr, ptr, ptr, ptr }, ptr %o11, i32 0, i32 0
  store ptr %o7, ptr %t11
  %t12 = getelementptr { ptr, ptr, ptr, ptr }, ptr %o8, i32 0, i32 1
  store ptr %o3, ptr %t12
  %t13 = getelementptr { ptr, i32 }, ptr %o30, i32 0, i32 0
  store ptr %o3, ptr %t13
  %t14 = getelementptr { ptr, ptr, ptr, ptr }, ptr %o8, i32 0, i32 2
  store ptr %o4, ptr %t14
  %t15 = getelementptr { ptr, i32 }, ptr %o34, i32 0, i32 0
  store ptr %o4, ptr %t15
  %t16 = getelementptr { ptr, ptr, ptr, ptr }, ptr %o11, i32 0, i32 1
  store ptr %o5, ptr %t16
  %t17 = getelementptr { ptr, i32 }, ptr %o38, i32 0, i32 0
  store ptr %o5, ptr %t17
  %t18 = getelementptr { ptr, ptr, ptr, ptr }, ptr %o11, i32 0, i32 2
  store ptr %o6, ptr %t18
  %t19 = getelementptr { ptr, i32 }, ptr %o42, i32 0, i32 0
  store ptr %o6, ptr %t19
  %t20 = load i64, ptr %d0
  %t21 = load i64, ptr %d2
  %t23 = call i64 @mapal_gpu_alloc(ptr %d22, i64 16384)
  %t25 = call i64 @mapal_gpu_alloc(ptr %d24, i64 16384)
  %t27 = call i64 @mapal_gpu_alloc(ptr %d26, i64 16384)
  %t29 = getelementptr [4 x i64], ptr %p28, i64 0, i64 0
  store i64 %t20, ptr %t29
  %t30 = getelementptr [4 x i64], ptr %p28, i64 0, i64 1
  store i64 %t21, ptr %t30
  %t31 = getelementptr [4 x i64], ptr %p28, i64 0, i64 2
  store i64 0, ptr %t31
  %t32 = getelementptr [4 x i64], ptr %p28, i64 0, i64 3
  store i64 %t27, ptr %t32
  call void @mapal_gpu_launchv(i64 2432853438081934057, i32 1, i32 4, ptr %p28, i32 4, i32 2, i32 32, i32 1536)
  %t33 = getelementptr { ptr, i32 }, ptr %o22, i32 0, i32 0
  store ptr %o10, ptr %t33
  %t34 = getelementptr { ptr, i32 }, ptr %o26, i32 0, i32 0
  store ptr %o10, ptr %t34
  %t35 = load i64, ptr %d4
  %t36 = load i64, ptr %d6
  %t38 = call i64 @mapal_gpu_alloc(ptr %d37, i64 16384)
  %t40 = getelementptr [4 x i64], ptr %p39, i64 0, i64 0
  store i64 %t35, ptr %t40
  %t41 = getelementptr [4 x i64], ptr %p39, i64 0, i64 1
  store i64 %t36, ptr %t41
  %t42 = getelementptr [4 x i64], ptr %p39, i64 0, i64 2
  store i64 0, ptr %t42
  %t43 = getelementptr [4 x i64], ptr %p39, i64 0, i64 3
  store i64 %t38, ptr %t43
  call void @mapal_gpu_launchv(i64 -2211854674245207611, i32 2, i32 4, ptr %p39, i32 4, i32 2, i32 32, i32 1536)
  call void @mapal_gpu_free(ptr %d22)
  call void @mapal_gpu_free(ptr %d24)
  %t44 = getelementptr { ptr, i32 }, ptr %o14, i32 0, i32 0
  store ptr %o13, ptr %t44
  %t45 = getelementptr { ptr, i32 }, ptr %o18, i32 0, i32 0
  store ptr %o13, ptr %t45
  %t46 = getelementptr { ptr, i32 }, ptr %o14, i32 0, i32 1
  store i32 0, ptr %t46
  %t47 = getelementptr { ptr, i32 }, ptr %o14, i32 0, i32 1
  %t48 = load i32, ptr %t47
  %t49 = sext i32 %t48 to i64
  %t50 = getelementptr [4096 x float], ptr %o13, i64 0, i64 %t49
  %t51 = load i64, ptr %d37
  %t52 = mul i64 %t49, 4
  call void @mapal_gpu_d2h(ptr %t50, i64 %t51, i64 %t52, i64 4)
  %t53 = load float, ptr %t50
  store float %t53, ptr %o15
  %t54 = load float, ptr %o15
  store float %t54, ptr %o16
  %t55 = load float, ptr %o16
  call void @mapal_print_f32(float %t55, i1 zeroext true)
  %t56 = getelementptr { ptr, i32 }, ptr %o18, i32 0, i32 1
  store i32 4095, ptr %t56
  %t57 = getelementptr { ptr, i32 }, ptr %o18, i32 0, i32 1
  %t58 = load i32, ptr %t57
  %t59 = sext i32 %t58 to i64
  %t60 = getelementptr [4096 x float], ptr %o13, i64 0, i64 %t59
  %t61 = load i64, ptr %d37
  %t62 = mul i64 %t59, 4
  call void @mapal_gpu_d2h(ptr %t60, i64 %t61, i64 %t62, i64 4)
  %t63 = load float, ptr %t60
  store float %t63, ptr %o19
  call void @mapal_gpu_free(ptr %d37)
  %t64 = load float, ptr %o19
  store float %t64, ptr %o20
  %t65 = load float, ptr %o20
  call void @mapal_print_f32(float %t65, i1 zeroext true)
  %t66 = getelementptr { ptr, i32 }, ptr %o22, i32 0, i32 1
  store i32 0, ptr %t66
  %t67 = getelementptr { ptr, i32 }, ptr %o22, i32 0, i32 1
  %t68 = load i32, ptr %t67
  %t69 = sext i32 %t68 to i64
  %t70 = getelementptr [4096 x float], ptr %o10, i64 0, i64 %t69
  %t71 = load i64, ptr %d26
  %t72 = mul i64 %t69, 4
  call void @mapal_gpu_d2h(ptr %t70, i64 %t71, i64 %t72, i64 4)
  %t73 = load float, ptr %t70
  store float %t73, ptr %o23
  %t74 = load float, ptr %o23
  store float %t74, ptr %o24
  %t75 = load float, ptr %o24
  call void @mapal_print_f32(float %t75, i1 zeroext true)
  %t76 = getelementptr { ptr, i32 }, ptr %o26, i32 0, i32 1
  store i32 4095, ptr %t76
  %t77 = getelementptr { ptr, i32 }, ptr %o26, i32 0, i32 1
  %t78 = load i32, ptr %t77
  %t79 = sext i32 %t78 to i64
  %t80 = getelementptr [4096 x float], ptr %o10, i64 0, i64 %t79
  %t81 = load i64, ptr %d26
  %t82 = mul i64 %t79, 4
  call void @mapal_gpu_d2h(ptr %t80, i64 %t81, i64 %t82, i64 4)
  %t83 = load float, ptr %t80
  store float %t83, ptr %o27
  call void @mapal_gpu_free(ptr %d26)
  %t84 = load float, ptr %o27
  store float %t84, ptr %o28
  %t85 = load float, ptr %o28
  call void @mapal_print_f32(float %t85, i1 zeroext true)
  %t86 = getelementptr { ptr, i32 }, ptr %o30, i32 0, i32 1
  store i32 0, ptr %t86
  %t87 = getelementptr { ptr, i32 }, ptr %o30, i32 0, i32 1
  %t88 = load i32, ptr %t87
  %t89 = sext i32 %t88 to i64
  %t90 = getelementptr [4096 x float], ptr %o3, i64 0, i64 %t89
  %t91 = load i64, ptr %d0
  %t92 = mul i64 %t89, 4
  call void @mapal_gpu_d2h(ptr %t90, i64 %t91, i64 %t92, i64 4)
  %t93 = load float, ptr %t90
  store float %t93, ptr %o31
  call void @mapal_gpu_free(ptr %d0)
  %t94 = load float, ptr %o31
  store float %t94, ptr %o32
  %t95 = load float, ptr %o32
  call void @mapal_print_f32(float %t95, i1 zeroext true)
  %t96 = getelementptr { ptr, i32 }, ptr %o34, i32 0, i32 1
  store i32 0, ptr %t96
  %t97 = getelementptr { ptr, i32 }, ptr %o34, i32 0, i32 1
  %t98 = load i32, ptr %t97
  %t99 = sext i32 %t98 to i64
  %t100 = getelementptr [4096 x float], ptr %o4, i64 0, i64 %t99
  %t101 = load i64, ptr %d2
  %t102 = mul i64 %t99, 4
  call void @mapal_gpu_d2h(ptr %t100, i64 %t101, i64 %t102, i64 4)
  %t103 = load float, ptr %t100
  store float %t103, ptr %o35
  call void @mapal_gpu_free(ptr %d2)
  %t104 = load float, ptr %o35
  store float %t104, ptr %o36
  %t105 = load float, ptr %o36
  call void @mapal_print_f32(float %t105, i1 zeroext true)
  %t106 = getelementptr { ptr, i32 }, ptr %o38, i32 0, i32 1
  store i32 0, ptr %t106
  %t107 = getelementptr { ptr, i32 }, ptr %o38, i32 0, i32 1
  %t108 = load i32, ptr %t107
  %t109 = sext i32 %t108 to i64
  %t110 = getelementptr [4096 x float], ptr %o5, i64 0, i64 %t109
  %t111 = load i64, ptr %d4
  %t112 = mul i64 %t109, 4
  call void @mapal_gpu_d2h(ptr %t110, i64 %t111, i64 %t112, i64 4)
  %t113 = load float, ptr %t110
  store float %t113, ptr %o39
  call void @mapal_gpu_free(ptr %d4)
  %t114 = load float, ptr %o39
  store float %t114, ptr %o40
  %t115 = load float, ptr %o40
  call void @mapal_print_f32(float %t115, i1 zeroext true)
  %t116 = getelementptr { ptr, i32 }, ptr %o42, i32 0, i32 1
  store i32 0, ptr %t116
  %t117 = getelementptr { ptr, i32 }, ptr %o42, i32 0, i32 1
  %t118 = load i32, ptr %t117
  %t119 = sext i32 %t118 to i64
  %t120 = getelementptr [4096 x float], ptr %o6, i64 0, i64 %t119
  %t121 = load i64, ptr %d6
  %t122 = mul i64 %t119, 4
  call void @mapal_gpu_d2h(ptr %t120, i64 %t121, i64 %t122, i64 4)
  %t123 = load float, ptr %t120
  store float %t123, ptr %o43
  call void @mapal_gpu_free(ptr %d6)
  %t124 = load float, ptr %o43
  store float %t124, ptr %o44
  %t125 = load float, ptr %o44
  call void @mapal_print_f32(float %t125, i1 zeroext true)
  call void @mapal_gpu_free(ptr %d0)
  call void @mapal_gpu_free(ptr %d2)
  call void @mapal_gpu_free(ptr %d4)
  call void @mapal_gpu_free(ptr %d6)
  call void @mapal_gpu_free(ptr %d22)
  call void @mapal_gpu_free(ptr %d26)
  call void @mapal_gpu_free(ptr %d24)
  call void @mapal_gpu_free(ptr %d37)
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

define i32 @main() {
entry:
  call void @mapal_gpu_init(i32 3)
  call void @mapal_main()
  ret i32 0
}
