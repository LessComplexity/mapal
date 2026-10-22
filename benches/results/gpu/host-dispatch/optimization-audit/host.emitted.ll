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

@.str0 = private unnamed_addr constant [5 x i8] c"step="
@.str1 = private unnamed_addr constant [1 x i8] c"y"

define internal void @mapal_main() {
entry:
  %o16 = alloca { float, i32 }
  %o17 = alloca [128 x float]
  %o18 = alloca { float, i32 }
  %o19 = alloca [8 x float]
  %o20 = alloca { [128 x float], [8 x float], i32 }
  %o25 = alloca [8 x i32]
  %o40 = alloca i32
  %o44 = alloca { [128 x float], [8 x float], i32 }
  %o45 = alloca [128 x float]
  %o46 = alloca [8 x float]
  %o47 = alloca i32
  %o48 = alloca { i32, i32 }
  %o49 = alloca i1
  %o50 = alloca { i32, i32 }
  %o51 = alloca i32
  %o52 = alloca { i32, i32 }
  %o53 = alloca i32
  %o54 = alloca { i32, i32 }
  %o55 = alloca i32
  %o56 = alloca { i32, i32 }
  %o57 = alloca i32
  %o58 = alloca { i32, i32 }
  %o59 = alloca i32
  %o60 = alloca { i32, i32 }
  %o61 = alloca i32
  %o62 = alloca { i32, i32 }
  %o63 = alloca i32
  %o64 = alloca float
  %o65 = alloca { float, float }
  %o66 = alloca float
  %o67 = alloca { ptr, i32 }
  %o68 = alloca float
  %o69 = alloca { float, float }
  %o70 = alloca float
  %o71 = alloca { i32, i32 }
  %o72 = alloca i32
  %o73 = alloca { i32, i32 }
  %o74 = alloca i32
  %o75 = alloca { ptr, i32, float }
  %o77 = alloca { i32, i32 }
  %o78 = alloca i32
  %o79 = alloca { i32, i32 }
  %o80 = alloca i32
  %o81 = alloca { i32, i32 }
  %o82 = alloca i32
  %o83 = alloca { i32, i32 }
  %o84 = alloca i32
  %o85 = alloca { i32, i32 }
  %o86 = alloca i32
  %o87 = alloca { i32, i32 }
  %o88 = alloca i32
  %o89 = alloca { i32, i32 }
  %o90 = alloca i32
  %o91 = alloca float
  %o92 = alloca { float, float }
  %o93 = alloca float
  %o94 = alloca { ptr, i32 }
  %o95 = alloca float
  %o96 = alloca { float, float }
  %o97 = alloca float
  %o98 = alloca { i32, i32 }
  %o99 = alloca i32
  %o100 = alloca { i32, i32 }
  %o101 = alloca i32
  %o102 = alloca { ptr, i32, float }
  %o104 = alloca { i32, i32 }
  %o105 = alloca i32
  %o106 = alloca { i32, i32 }
  %o107 = alloca i32
  %o108 = alloca { i32, i32 }
  %o109 = alloca i32
  %o110 = alloca { i32, i32 }
  %o111 = alloca i32
  %o112 = alloca { i32, i32 }
  %o113 = alloca i32
  %o114 = alloca { i32, i32 }
  %o115 = alloca i32
  %o116 = alloca { i32, i32 }
  %o117 = alloca i32
  %o118 = alloca float
  %o119 = alloca { float, float }
  %o120 = alloca float
  %o121 = alloca { ptr, i32 }
  %o122 = alloca float
  %o123 = alloca { float, float }
  %o124 = alloca float
  %o125 = alloca { i32, i32 }
  %o126 = alloca i32
  %o127 = alloca { i32, i32 }
  %o128 = alloca i32
  %o129 = alloca { ptr, i32, float }
  %o131 = alloca { i32, i32 }
  %o132 = alloca i32
  %o133 = alloca { i32, i32 }
  %o134 = alloca i32
  %o135 = alloca { i32, i32 }
  %o136 = alloca i32
  %o137 = alloca { i32, i32 }
  %o138 = alloca i32
  %o139 = alloca { i32, i32 }
  %o140 = alloca i32
  %o141 = alloca { i32, i32 }
  %o142 = alloca i32
  %o143 = alloca { i32, i32 }
  %o144 = alloca i32
  %o145 = alloca float
  %o146 = alloca { float, float }
  %o147 = alloca float
  %o148 = alloca { ptr, i32 }
  %o149 = alloca float
  %o150 = alloca { float, float }
  %o151 = alloca float
  %o152 = alloca { i32, i32 }
  %o153 = alloca i32
  %o154 = alloca { i32, i32 }
  %o155 = alloca i32
  %o156 = alloca { ptr, i32, float }
  %o158 = alloca { i32, i32 }
  %o159 = alloca i32
  %o160 = alloca { i32, i32 }
  %o161 = alloca i32
  %o162 = alloca { i32, i32 }
  %o163 = alloca i32
  %o164 = alloca { i32, i32 }
  %o165 = alloca i32
  %o166 = alloca { i32, i32 }
  %o167 = alloca i32
  %o168 = alloca { i32, i32 }
  %o169 = alloca i32
  %o170 = alloca { i32, i32 }
  %o171 = alloca i32
  %o172 = alloca float
  %o173 = alloca { float, float }
  %o174 = alloca float
  %o175 = alloca { ptr, i32 }
  %o176 = alloca float
  %o177 = alloca { float, float }
  %o178 = alloca float
  %o179 = alloca { i32, i32 }
  %o180 = alloca i32
  %o181 = alloca { i32, i32 }
  %o182 = alloca i32
  %o183 = alloca { ptr, i32, float }
  %o185 = alloca { i32, i32 }
  %o186 = alloca i32
  %o187 = alloca { i32, i32 }
  %o188 = alloca i32
  %o189 = alloca { i32, i32 }
  %o190 = alloca i32
  %o191 = alloca { i32, i32 }
  %o192 = alloca i32
  %o193 = alloca { i32, i32 }
  %o194 = alloca i32
  %o195 = alloca { i32, i32 }
  %o196 = alloca i32
  %o197 = alloca { i32, i32 }
  %o198 = alloca i32
  %o199 = alloca float
  %o200 = alloca { float, float }
  %o201 = alloca float
  %o202 = alloca { ptr, i32 }
  %o203 = alloca float
  %o204 = alloca { float, float }
  %o205 = alloca float
  %o206 = alloca { i32, i32 }
  %o207 = alloca i32
  %o208 = alloca { i32, i32 }
  %o209 = alloca i32
  %o210 = alloca { ptr, i32, float }
  %o212 = alloca { i32, i32 }
  %o213 = alloca i32
  %o214 = alloca { i32, i32 }
  %o215 = alloca i32
  %o216 = alloca { i32, i32 }
  %o217 = alloca i32
  %o218 = alloca { i32, i32 }
  %o219 = alloca i32
  %o220 = alloca { i32, i32 }
  %o221 = alloca i32
  %o222 = alloca { i32, i32 }
  %o223 = alloca i32
  %o224 = alloca { i32, i32 }
  %o225 = alloca i32
  %o226 = alloca float
  %o227 = alloca { float, float }
  %o228 = alloca float
  %o229 = alloca { ptr, i32 }
  %o230 = alloca float
  %o231 = alloca { float, float }
  %o232 = alloca float
  %o233 = alloca { i32, i32 }
  %o234 = alloca i32
  %o235 = alloca { i32, i32 }
  %o236 = alloca i32
  %o237 = alloca { ptr, i32, float }
  %o239 = alloca { i32, i32 }
  %o240 = alloca i32
  %o241 = alloca { i32, i32 }
  %o242 = alloca i32
  %o243 = alloca { i32, i32 }
  %o244 = alloca i32
  %o245 = alloca { i32, i32 }
  %o246 = alloca i32
  %o247 = alloca { i32, i32 }
  %o248 = alloca i32
  %o249 = alloca { i32, i32 }
  %o250 = alloca i32
  %o251 = alloca { i32, i32 }
  %o252 = alloca i32
  %o253 = alloca float
  %o254 = alloca { float, float }
  %o255 = alloca float
  %o256 = alloca { ptr, i32 }
  %o257 = alloca float
  %o258 = alloca { float, float }
  %o259 = alloca float
  %o260 = alloca { i32, i32 }
  %o261 = alloca i32
  %o262 = alloca { i32, i32 }
  %o263 = alloca i32
  %o264 = alloca { ptr, i32, float }
  %o266 = alloca { ptr, ptr, ptr, ptr }
  %o268 = alloca { ptr, ptr, ptr, ptr }
  %o270 = alloca { i32, ptr }
  %o271 = alloca [8 x float]
  %o272 = alloca { ptr, ptr, ptr, ptr }
  %o274 = alloca { ptr, ptr }
  %o276 = alloca { ptr, ptr, ptr, ptr }
  %o278 = alloca { ptr, ptr, ptr, ptr }
  %o280 = alloca { ptr, ptr, ptr, ptr }
  %o282 = alloca { ptr, ptr, ptr }
  %o284 = alloca { ptr, ptr, ptr, ptr }
  %o286 = alloca { ptr, ptr, ptr }
  %o288 = alloca { ptr, ptr, ptr }
  %o289 = alloca [8 x float]
  %o290 = alloca { i32, i32 }
  %o291 = alloca i32
  %o292 = alloca { [128 x float], [8 x float], i32 }
  %o293 = alloca { { [128 x float], [8 x float], i32 }, i1 }
  %o294 = alloca { [8 x float], i1 }
  %o295 = alloca [8 x float]
  %o296 = alloca { ptr, i32 }
  %o297 = alloca float
  %o298 = alloca float
  %o300 = alloca { ptr, i32 }
  %o301 = alloca float
  %o302 = alloca float
  %o304 = alloca { ptr, i32 }
  %o305 = alloca float
  %o306 = alloca float
  %o308 = alloca { ptr, i32 }
  %o309 = alloca float
  %o310 = alloca float
  %o312 = alloca { ptr, i32 }
  %o313 = alloca float
  %o314 = alloca float
  %o316 = alloca { ptr, i32 }
  %o317 = alloca float
  %o318 = alloca float
  %o320 = alloca { ptr, i32 }
  %o321 = alloca float
  %o322 = alloca float
  %o324 = alloca { ptr, i32 }
  %o325 = alloca float
  %o326 = alloca float
  %d0 = alloca i64
  store i64 0, ptr %d0
  %d2 = alloca i64
  store i64 0, ptr %d2
  %d4 = alloca i64
  store i64 0, ptr %d4
  %d6 = alloca i64
  store i64 0, ptr %d6
  %d8 = alloca i64
  store i64 0, ptr %d8
  %d10 = alloca i64
  store i64 0, ptr %d10
  %d12 = alloca i64
  store i64 0, ptr %d12
  %s18 = alloca i64
  %s31 = alloca i64
  %s41 = alloca i64
  %s60 = alloca i1
  %d1027 = alloca i64
  store i64 0, ptr %d1027
  %d1045 = alloca i64
  store i64 0, ptr %d1045
  %d1050 = alloca i64
  store i64 0, ptr %d1050
  %s1053 = alloca i64
  %s1060 = alloca { i32, i32 }
  %d1069 = alloca i64
  store i64 0, ptr %d1069
  %d1074 = alloca i64
  store i64 0, ptr %d1074
  %p1076 = alloca [4 x i64]
  %d1085 = alloca i64
  store i64 0, ptr %d1085
  %d1090 = alloca i64
  store i64 0, ptr %d1090
  %d1097 = alloca i64
  store i64 0, ptr %d1097
  %p1099 = alloca [5 x i64]
  %d1111 = alloca i64
  store i64 0, ptr %d1111
  %p1113 = alloca [6 x i64]
  %t1 = call i64 @mapal_gpu_alloc(ptr %d0, i64 256)
  call void @mapal_gpu_gen(i64 570417511890179569, i64 %t1, i64 64)
  %t3 = call i64 @mapal_gpu_alloc(ptr %d2, i64 512)
  call void @mapal_gpu_gen(i64 -1791989716614967237, i64 %t3, i64 128)
  %t5 = call i64 @mapal_gpu_alloc(ptr %d4, i64 512)
  call void @mapal_gpu_gen(i64 -4223766215135587374, i64 %t5, i64 128)
  %t7 = call i64 @mapal_gpu_alloc(ptr %d6, i64 256)
  call void @mapal_gpu_gen(i64 -9196861651627557674, i64 %t7, i64 64)
  %t9 = call i64 @mapal_gpu_alloc(ptr %d8, i64 256)
  call void @mapal_gpu_gen(i64 5761537033769727445, i64 %t9, i64 64)
  %t11 = call i64 @mapal_gpu_alloc(ptr %d10, i64 64)
  call void @mapal_gpu_gen(i64 1396038385954328191, i64 %t11, i64 16)
  %t13 = call i64 @mapal_gpu_alloc(ptr %d12, i64 32)
  call void @mapal_gpu_gen(i64 -3200766962839905091, i64 %t13, i64 8)
  %t14 = getelementptr { float, i32 }, ptr %o16, i32 0, i32 0
  store float 0x0000000000000000, ptr %t14
  %t15 = getelementptr { float, i32 }, ptr %o16, i32 0, i32 1
  store i32 128, ptr %t15
  %t16 = getelementptr { float, i32 }, ptr %o16, i32 0, i32 0
  %t17 = load float, ptr %t16
  store i64 0, ptr %s18
  br label %bb19
bb19:
  %t22 = load i64, ptr %s18
  %t23 = icmp uge i64 %t22, 128
  br i1 %t23, label %bb21, label %bb20
bb20:
  %t24 = getelementptr [128 x float], ptr %o17, i64 0, i64 %t22
  store float %t17, ptr %t24
  %t25 = add i64 %t22, 1
  store i64 %t25, ptr %s18
  br label %bb19
bb21:
  %t26 = getelementptr { [128 x float], [8 x float], i32 }, ptr %o20, i32 0, i32 0
  call void @llvm.memcpy.p0.p0.i64(ptr %t26, ptr %o17, i64 ptrtoint (ptr getelementptr ([128 x float], ptr null, i64 1) to i64), i1 false)
  %t27 = getelementptr { float, i32 }, ptr %o18, i32 0, i32 0
  store float 0x0000000000000000, ptr %t27
  %t28 = getelementptr { float, i32 }, ptr %o18, i32 0, i32 1
  store i32 8, ptr %t28
  %t29 = getelementptr { float, i32 }, ptr %o18, i32 0, i32 0
  %t30 = load float, ptr %t29
  store i64 0, ptr %s31
  br label %bb32
bb32:
  %t35 = load i64, ptr %s31
  %t36 = icmp uge i64 %t35, 8
  br i1 %t36, label %bb34, label %bb33
bb33:
  %t37 = getelementptr [8 x float], ptr %o19, i64 0, i64 %t35
  store float %t30, ptr %t37
  %t38 = add i64 %t35, 1
  store i64 %t38, ptr %s31
  br label %bb32
bb34:
  %t39 = getelementptr { [128 x float], [8 x float], i32 }, ptr %o20, i32 0, i32 1
  call void @llvm.memcpy.p0.p0.i64(ptr %t39, ptr %o19, i64 ptrtoint (ptr getelementptr ([8 x float], ptr null, i64 1) to i64), i1 false)
  %t40 = getelementptr { [128 x float], [8 x float], i32 }, ptr %o20, i32 0, i32 2
  store i32 0, ptr %t40
  store i64 0, ptr %s41
  br label %bb42
bb42:
  %t45 = load i64, ptr %s41
  %t46 = icmp uge i64 %t45, 8
  br i1 %t46, label %bb44, label %bb43
bb43:
  %t47 = trunc i64 %t45 to i32
  %t48 = getelementptr [8 x i32], ptr %o25, i64 0, i64 %t45
  store i32 %t47, ptr %t48
  %t49 = add i64 %t45, 1
  store i64 %t49, ptr %s41
  br label %bb42
bb44:
  call void @mapal_print_str(ptr @.str0, i64 5, i1 zeroext false)
  store i32 -1, ptr %o40
  %t50 = load i32, ptr %o40
  call void @mapal_print_i32(i32 %t50, i1 zeroext true)
  call void @mapal_print_str(ptr @.str1, i64 1, i1 zeroext true)
  %t51 = getelementptr { ptr, i32 }, ptr %o296, i32 0, i32 1
  store i32 0, ptr %t51
  %t52 = getelementptr { ptr, i32 }, ptr %o300, i32 0, i32 1
  store i32 1, ptr %t52
  %t53 = getelementptr { ptr, i32 }, ptr %o304, i32 0, i32 1
  store i32 2, ptr %t53
  %t54 = getelementptr { ptr, i32 }, ptr %o308, i32 0, i32 1
  store i32 3, ptr %t54
  %t55 = getelementptr { ptr, i32 }, ptr %o312, i32 0, i32 1
  store i32 4, ptr %t55
  %t56 = getelementptr { ptr, i32 }, ptr %o316, i32 0, i32 1
  store i32 5, ptr %t56
  %t57 = getelementptr { ptr, i32 }, ptr %o320, i32 0, i32 1
  store i32 6, ptr %t57
  %t58 = getelementptr { ptr, i32 }, ptr %o324, i32 0, i32 1
  store i32 7, ptr %t58
  %t59 = load { [128 x float], [8 x float], i32 }, ptr %o20
  store { [128 x float], [8 x float], i32 } %t59, ptr %o44
  store i1 false, ptr %s60
  br label %bb61
bb61:
  %t65 = getelementptr { i32, i32 }, ptr %o48, i32 0, i32 1
  store i32 16, ptr %t65
  %t66 = getelementptr { [128 x float], [8 x float], i32 }, ptr %o44, i32 0, i32 1
  call void @llvm.memcpy.p0.p0.i64(ptr %o46, ptr %t66, i64 ptrtoint (ptr getelementptr ([8 x float], ptr null, i64 1) to i64), i1 false)
  %t67 = getelementptr { [128 x float], [8 x float], i32 }, ptr %o44, i32 0, i32 2
  %t68 = load i32, ptr %t67
  store i32 %t68, ptr %o47
  %t69 = getelementptr { [8 x float], i1 }, ptr %o294, i32 0, i32 0
  call void @llvm.memcpy.p0.p0.i64(ptr %t69, ptr %o46, i64 ptrtoint (ptr getelementptr ([8 x float], ptr null, i64 1) to i64), i1 false)
  %t70 = load i32, ptr %o47
  %t71 = getelementptr { i32, i32 }, ptr %o48, i32 0, i32 0
  store i32 %t70, ptr %t71
  %t72 = getelementptr { i32, i32 }, ptr %o48, i32 0, i32 0
  %t73 = load i32, ptr %t72
  %t74 = getelementptr { i32, i32 }, ptr %o48, i32 0, i32 1
  %t75 = load i32, ptr %t74
  %t76 = icmp slt i32 %t73, %t75
  store i1 %t76, ptr %o49
  %t77 = load i1, ptr %o49
  %t78 = getelementptr { [8 x float], i1 }, ptr %o294, i32 0, i32 1
  store i1 %t77, ptr %t78
  %t79 = getelementptr { [8 x float], i1 }, ptr %o294, i32 0, i32 1
  %t80 = load i1, ptr %t79
  br i1 %t80, label %bb62, label %bb63
bb62:
  %t81 = getelementptr { ptr, ptr, ptr, ptr }, ptr %o278, i32 0, i32 2
  store ptr null, ptr %t81
  %t82 = getelementptr { ptr, ptr, ptr, ptr }, ptr %o280, i32 0, i32 2
  store ptr null, ptr %t82
  %t83 = getelementptr { ptr, ptr, ptr, ptr }, ptr %o284, i32 0, i32 2
  store ptr null, ptr %t83
  %t84 = getelementptr { ptr, ptr, ptr, ptr }, ptr %o266, i32 0, i32 2
  store ptr null, ptr %t84
  %t85 = getelementptr { ptr, ptr, ptr, ptr }, ptr %o268, i32 0, i32 2
  store ptr null, ptr %t85
  %t86 = getelementptr { ptr, ptr, ptr }, ptr %o282, i32 0, i32 1
  store ptr null, ptr %t86
  %t87 = getelementptr { ptr, ptr, ptr }, ptr %o286, i32 0, i32 1
  store ptr null, ptr %t87
  %t88 = getelementptr { i32, i32 }, ptr %o50, i32 0, i32 1
  store i32 8, ptr %t88
  %t89 = getelementptr { i32, i32 }, ptr %o52, i32 0, i32 1
  store i32 0, ptr %t89
  %t90 = getelementptr { i32, i32 }, ptr %o54, i32 0, i32 1
  store i32 101, ptr %t90
  %t91 = getelementptr { i32, i32 }, ptr %o56, i32 0, i32 1
  store i32 7, ptr %t91
  %t92 = getelementptr { i32, i32 }, ptr %o58, i32 0, i32 1
  store i32 13, ptr %t92
  %t93 = getelementptr { i32, i32 }, ptr %o60, i32 0, i32 1
  store i32 101, ptr %t93
  %t94 = getelementptr { i32, i32 }, ptr %o62, i32 0, i32 1
  store i32 50, ptr %t94
  %t95 = getelementptr { float, float }, ptr %o65, i32 0, i32 1
  store float 0x4050000000000000, ptr %t95
  %t96 = getelementptr { ptr, i32 }, ptr %o67, i32 0, i32 1
  store i32 0, ptr %t96
  %t97 = getelementptr { i32, i32 }, ptr %o71, i32 0, i32 1
  store i32 8, ptr %t97
  %t98 = getelementptr { i32, i32 }, ptr %o73, i32 0, i32 1
  store i32 0, ptr %t98
  %t99 = getelementptr { i32, i32 }, ptr %o77, i32 0, i32 1
  store i32 8, ptr %t99
  %t100 = getelementptr { i32, i32 }, ptr %o79, i32 0, i32 1
  store i32 1, ptr %t100
  %t101 = getelementptr { i32, i32 }, ptr %o81, i32 0, i32 1
  store i32 101, ptr %t101
  %t102 = getelementptr { i32, i32 }, ptr %o83, i32 0, i32 1
  store i32 7, ptr %t102
  %t103 = getelementptr { i32, i32 }, ptr %o85, i32 0, i32 1
  store i32 13, ptr %t103
  %t104 = getelementptr { i32, i32 }, ptr %o87, i32 0, i32 1
  store i32 101, ptr %t104
  %t105 = getelementptr { i32, i32 }, ptr %o89, i32 0, i32 1
  store i32 50, ptr %t105
  %t106 = getelementptr { float, float }, ptr %o92, i32 0, i32 1
  store float 0x4050000000000000, ptr %t106
  %t107 = getelementptr { ptr, i32 }, ptr %o94, i32 0, i32 1
  store i32 1, ptr %t107
  %t108 = getelementptr { i32, i32 }, ptr %o98, i32 0, i32 1
  store i32 8, ptr %t108
  %t109 = getelementptr { i32, i32 }, ptr %o100, i32 0, i32 1
  store i32 1, ptr %t109
  %t110 = getelementptr { i32, i32 }, ptr %o104, i32 0, i32 1
  store i32 8, ptr %t110
  %t111 = getelementptr { i32, i32 }, ptr %o106, i32 0, i32 1
  store i32 2, ptr %t111
  %t112 = getelementptr { i32, i32 }, ptr %o108, i32 0, i32 1
  store i32 101, ptr %t112
  %t113 = getelementptr { i32, i32 }, ptr %o110, i32 0, i32 1
  store i32 7, ptr %t113
  %t114 = getelementptr { i32, i32 }, ptr %o112, i32 0, i32 1
  store i32 13, ptr %t114
  %t115 = getelementptr { i32, i32 }, ptr %o114, i32 0, i32 1
  store i32 101, ptr %t115
  %t116 = getelementptr { i32, i32 }, ptr %o116, i32 0, i32 1
  store i32 50, ptr %t116
  %t117 = getelementptr { float, float }, ptr %o119, i32 0, i32 1
  store float 0x4050000000000000, ptr %t117
  %t118 = getelementptr { ptr, i32 }, ptr %o121, i32 0, i32 1
  store i32 2, ptr %t118
  %t119 = getelementptr { i32, i32 }, ptr %o125, i32 0, i32 1
  store i32 8, ptr %t119
  %t120 = getelementptr { i32, i32 }, ptr %o127, i32 0, i32 1
  store i32 2, ptr %t120
  %t121 = getelementptr { i32, i32 }, ptr %o131, i32 0, i32 1
  store i32 8, ptr %t121
  %t122 = getelementptr { i32, i32 }, ptr %o133, i32 0, i32 1
  store i32 3, ptr %t122
  %t123 = getelementptr { i32, i32 }, ptr %o135, i32 0, i32 1
  store i32 101, ptr %t123
  %t124 = getelementptr { i32, i32 }, ptr %o137, i32 0, i32 1
  store i32 7, ptr %t124
  %t125 = getelementptr { i32, i32 }, ptr %o139, i32 0, i32 1
  store i32 13, ptr %t125
  %t126 = getelementptr { i32, i32 }, ptr %o141, i32 0, i32 1
  store i32 101, ptr %t126
  %t127 = getelementptr { i32, i32 }, ptr %o143, i32 0, i32 1
  store i32 50, ptr %t127
  %t128 = getelementptr { float, float }, ptr %o146, i32 0, i32 1
  store float 0x4050000000000000, ptr %t128
  %t129 = getelementptr { ptr, i32 }, ptr %o148, i32 0, i32 1
  store i32 3, ptr %t129
  %t130 = getelementptr { i32, i32 }, ptr %o152, i32 0, i32 1
  store i32 8, ptr %t130
  %t131 = getelementptr { i32, i32 }, ptr %o154, i32 0, i32 1
  store i32 3, ptr %t131
  %t132 = getelementptr { i32, i32 }, ptr %o158, i32 0, i32 1
  store i32 8, ptr %t132
  %t133 = getelementptr { i32, i32 }, ptr %o160, i32 0, i32 1
  store i32 4, ptr %t133
  %t134 = getelementptr { i32, i32 }, ptr %o162, i32 0, i32 1
  store i32 101, ptr %t134
  %t135 = getelementptr { i32, i32 }, ptr %o164, i32 0, i32 1
  store i32 7, ptr %t135
  %t136 = getelementptr { i32, i32 }, ptr %o166, i32 0, i32 1
  store i32 13, ptr %t136
  %t137 = getelementptr { i32, i32 }, ptr %o168, i32 0, i32 1
  store i32 101, ptr %t137
  %t138 = getelementptr { i32, i32 }, ptr %o170, i32 0, i32 1
  store i32 50, ptr %t138
  %t139 = getelementptr { float, float }, ptr %o173, i32 0, i32 1
  store float 0x4050000000000000, ptr %t139
  %t140 = getelementptr { ptr, i32 }, ptr %o175, i32 0, i32 1
  store i32 4, ptr %t140
  %t141 = getelementptr { i32, i32 }, ptr %o179, i32 0, i32 1
  store i32 8, ptr %t141
  %t142 = getelementptr { i32, i32 }, ptr %o181, i32 0, i32 1
  store i32 4, ptr %t142
  %t143 = getelementptr { i32, i32 }, ptr %o185, i32 0, i32 1
  store i32 8, ptr %t143
  %t144 = getelementptr { i32, i32 }, ptr %o187, i32 0, i32 1
  store i32 5, ptr %t144
  %t145 = getelementptr { i32, i32 }, ptr %o189, i32 0, i32 1
  store i32 101, ptr %t145
  %t146 = getelementptr { i32, i32 }, ptr %o191, i32 0, i32 1
  store i32 7, ptr %t146
  %t147 = getelementptr { i32, i32 }, ptr %o193, i32 0, i32 1
  store i32 13, ptr %t147
  %t148 = getelementptr { i32, i32 }, ptr %o195, i32 0, i32 1
  store i32 101, ptr %t148
  %t149 = getelementptr { i32, i32 }, ptr %o197, i32 0, i32 1
  store i32 50, ptr %t149
  %t150 = getelementptr { float, float }, ptr %o200, i32 0, i32 1
  store float 0x4050000000000000, ptr %t150
  %t151 = getelementptr { ptr, i32 }, ptr %o202, i32 0, i32 1
  store i32 5, ptr %t151
  %t152 = getelementptr { i32, i32 }, ptr %o206, i32 0, i32 1
  store i32 8, ptr %t152
  %t153 = getelementptr { i32, i32 }, ptr %o208, i32 0, i32 1
  store i32 5, ptr %t153
  %t154 = getelementptr { i32, i32 }, ptr %o212, i32 0, i32 1
  store i32 8, ptr %t154
  %t155 = getelementptr { i32, i32 }, ptr %o214, i32 0, i32 1
  store i32 6, ptr %t155
  %t156 = getelementptr { i32, i32 }, ptr %o216, i32 0, i32 1
  store i32 101, ptr %t156
  %t157 = getelementptr { i32, i32 }, ptr %o218, i32 0, i32 1
  store i32 7, ptr %t157
  %t158 = getelementptr { i32, i32 }, ptr %o220, i32 0, i32 1
  store i32 13, ptr %t158
  %t159 = getelementptr { i32, i32 }, ptr %o222, i32 0, i32 1
  store i32 101, ptr %t159
  %t160 = getelementptr { i32, i32 }, ptr %o224, i32 0, i32 1
  store i32 50, ptr %t160
  %t161 = getelementptr { float, float }, ptr %o227, i32 0, i32 1
  store float 0x4050000000000000, ptr %t161
  %t162 = getelementptr { ptr, i32 }, ptr %o229, i32 0, i32 1
  store i32 6, ptr %t162
  %t163 = getelementptr { i32, i32 }, ptr %o233, i32 0, i32 1
  store i32 8, ptr %t163
  %t164 = getelementptr { i32, i32 }, ptr %o235, i32 0, i32 1
  store i32 6, ptr %t164
  %t165 = getelementptr { i32, i32 }, ptr %o239, i32 0, i32 1
  store i32 8, ptr %t165
  %t166 = getelementptr { i32, i32 }, ptr %o241, i32 0, i32 1
  store i32 7, ptr %t166
  %t167 = getelementptr { i32, i32 }, ptr %o243, i32 0, i32 1
  store i32 101, ptr %t167
  %t168 = getelementptr { i32, i32 }, ptr %o245, i32 0, i32 1
  store i32 7, ptr %t168
  %t169 = getelementptr { i32, i32 }, ptr %o247, i32 0, i32 1
  store i32 13, ptr %t169
  %t170 = getelementptr { i32, i32 }, ptr %o249, i32 0, i32 1
  store i32 101, ptr %t170
  %t171 = getelementptr { i32, i32 }, ptr %o251, i32 0, i32 1
  store i32 50, ptr %t171
  %t172 = getelementptr { float, float }, ptr %o254, i32 0, i32 1
  store float 0x4050000000000000, ptr %t172
  %t173 = getelementptr { ptr, i32 }, ptr %o256, i32 0, i32 1
  store i32 7, ptr %t173
  %t174 = getelementptr { i32, i32 }, ptr %o260, i32 0, i32 1
  store i32 8, ptr %t174
  %t175 = getelementptr { i32, i32 }, ptr %o262, i32 0, i32 1
  store i32 7, ptr %t175
  %t176 = getelementptr { ptr, ptr, ptr, ptr }, ptr %o266, i32 0, i32 3
  store ptr null, ptr %t176
  %t177 = getelementptr { ptr, ptr, ptr, ptr }, ptr %o266, i32 0, i32 0
  store ptr null, ptr %t177
  %t178 = getelementptr { ptr, ptr, ptr, ptr }, ptr %o268, i32 0, i32 3
  store ptr null, ptr %t178
  %t179 = getelementptr { ptr, ptr, ptr, ptr }, ptr %o268, i32 0, i32 0
  store ptr null, ptr %t179
  %t180 = getelementptr { i32, ptr }, ptr %o270, i32 0, i32 1
  store ptr %o25, ptr %t180
  %t181 = getelementptr { ptr, ptr, ptr, ptr }, ptr %o272, i32 0, i32 3
  store ptr null, ptr %t181
  %t182 = getelementptr { ptr, ptr, ptr, ptr }, ptr %o276, i32 0, i32 0
  store ptr null, ptr %t182
  %t183 = getelementptr { ptr, ptr, ptr, ptr }, ptr %o272, i32 0, i32 0
  store ptr null, ptr %t183
  %t184 = getelementptr { ptr, ptr, ptr, ptr }, ptr %o276, i32 0, i32 3
  store ptr null, ptr %t184
  %t185 = getelementptr { ptr, ptr }, ptr %o274, i32 0, i32 1
  store ptr null, ptr %t185
  %t186 = getelementptr { ptr, ptr, ptr, ptr }, ptr %o278, i32 0, i32 3
  store ptr null, ptr %t186
  %t187 = getelementptr { ptr, ptr, ptr, ptr }, ptr %o278, i32 0, i32 0
  store ptr null, ptr %t187
  %t188 = getelementptr { ptr, ptr, ptr, ptr }, ptr %o280, i32 0, i32 3
  store ptr null, ptr %t188
  %t189 = getelementptr { ptr, ptr, ptr, ptr }, ptr %o280, i32 0, i32 0
  store ptr null, ptr %t189
  %t190 = getelementptr { ptr, ptr, ptr }, ptr %o282, i32 0, i32 2
  store ptr null, ptr %t190
  %t191 = getelementptr { ptr, ptr, ptr, ptr }, ptr %o284, i32 0, i32 3
  store ptr null, ptr %t191
  %t192 = getelementptr { ptr, ptr, ptr, ptr }, ptr %o284, i32 0, i32 0
  store ptr null, ptr %t192
  %t193 = getelementptr { ptr, ptr, ptr }, ptr %o286, i32 0, i32 2
  store ptr null, ptr %t193
  %t194 = getelementptr { ptr, ptr, ptr }, ptr %o288, i32 0, i32 2
  store ptr null, ptr %t194
  %t195 = getelementptr { i32, i32 }, ptr %o290, i32 0, i32 1
  store i32 1, ptr %t195
  %t196 = getelementptr { [128 x float], [8 x float], i32 }, ptr %o44, i32 0, i32 0
  call void @llvm.memcpy.p0.p0.i64(ptr %o45, ptr %t196, i64 ptrtoint (ptr getelementptr ([128 x float], ptr null, i64 1) to i64), i1 false)
  %t197 = getelementptr { ptr, i32, float }, ptr %o75, i32 0, i32 0
  store ptr %o45, ptr %t197
  %t198 = getelementptr { ptr, i32 }, ptr %o67, i32 0, i32 0
  store ptr %o46, ptr %t198
  %t199 = getelementptr { ptr, i32 }, ptr %o94, i32 0, i32 0
  store ptr %o46, ptr %t199
  %t200 = getelementptr { ptr, i32 }, ptr %o121, i32 0, i32 0
  store ptr %o46, ptr %t200
  %t201 = getelementptr { ptr, i32 }, ptr %o148, i32 0, i32 0
  store ptr %o46, ptr %t201
  %t202 = getelementptr { ptr, i32 }, ptr %o175, i32 0, i32 0
  store ptr %o46, ptr %t202
  %t203 = getelementptr { ptr, i32 }, ptr %o202, i32 0, i32 0
  store ptr %o46, ptr %t203
  %t204 = getelementptr { ptr, i32 }, ptr %o229, i32 0, i32 0
  store ptr %o46, ptr %t204
  %t205 = getelementptr { ptr, i32 }, ptr %o256, i32 0, i32 0
  store ptr %o46, ptr %t205
  %t206 = load i32, ptr %o47
  %t207 = getelementptr { i32, i32 }, ptr %o50, i32 0, i32 0
  store i32 %t206, ptr %t207
  %t208 = load i32, ptr %o47
  %t209 = getelementptr { i32, i32 }, ptr %o71, i32 0, i32 0
  store i32 %t208, ptr %t209
  %t210 = load i32, ptr %o47
  %t211 = getelementptr { i32, i32 }, ptr %o77, i32 0, i32 0
  store i32 %t210, ptr %t211
  %t212 = load i32, ptr %o47
  %t213 = getelementptr { i32, i32 }, ptr %o98, i32 0, i32 0
  store i32 %t212, ptr %t213
  %t214 = load i32, ptr %o47
  %t215 = getelementptr { i32, i32 }, ptr %o104, i32 0, i32 0
  store i32 %t214, ptr %t215
  %t216 = load i32, ptr %o47
  %t217 = getelementptr { i32, i32 }, ptr %o125, i32 0, i32 0
  store i32 %t216, ptr %t217
  %t218 = load i32, ptr %o47
  %t219 = getelementptr { i32, i32 }, ptr %o131, i32 0, i32 0
  store i32 %t218, ptr %t219
  %t220 = load i32, ptr %o47
  %t221 = getelementptr { i32, i32 }, ptr %o152, i32 0, i32 0
  store i32 %t220, ptr %t221
  %t222 = load i32, ptr %o47
  %t223 = getelementptr { i32, i32 }, ptr %o158, i32 0, i32 0
  store i32 %t222, ptr %t223
  %t224 = load i32, ptr %o47
  %t225 = getelementptr { i32, i32 }, ptr %o179, i32 0, i32 0
  store i32 %t224, ptr %t225
  %t226 = load i32, ptr %o47
  %t227 = getelementptr { i32, i32 }, ptr %o185, i32 0, i32 0
  store i32 %t226, ptr %t227
  %t228 = load i32, ptr %o47
  %t229 = getelementptr { i32, i32 }, ptr %o206, i32 0, i32 0
  store i32 %t228, ptr %t229
  %t230 = load i32, ptr %o47
  %t231 = getelementptr { i32, i32 }, ptr %o212, i32 0, i32 0
  store i32 %t230, ptr %t231
  %t232 = load i32, ptr %o47
  %t233 = getelementptr { i32, i32 }, ptr %o233, i32 0, i32 0
  store i32 %t232, ptr %t233
  %t234 = load i32, ptr %o47
  %t235 = getelementptr { i32, i32 }, ptr %o239, i32 0, i32 0
  store i32 %t234, ptr %t235
  %t236 = load i32, ptr %o47
  %t237 = getelementptr { i32, i32 }, ptr %o260, i32 0, i32 0
  store i32 %t236, ptr %t237
  %t238 = load i32, ptr %o47
  %t239 = getelementptr { i32, ptr }, ptr %o270, i32 0, i32 0
  store i32 %t238, ptr %t239
  %t240 = load i32, ptr %o47
  %t241 = getelementptr { i32, i32 }, ptr %o290, i32 0, i32 0
  store i32 %t240, ptr %t241
  %t242 = load i1, ptr %o49
  %t243 = getelementptr { { [128 x float], [8 x float], i32 }, i1 }, ptr %o293, i32 0, i32 1
  store i1 %t242, ptr %t243
  %t244 = getelementptr { i32, i32 }, ptr %o50, i32 0, i32 0
  %t245 = load i32, ptr %t244
  %t246 = getelementptr { i32, i32 }, ptr %o50, i32 0, i32 1
  %t247 = load i32, ptr %t246
  %t248 = mul i32 %t245, %t247
  store i32 %t248, ptr %o51
  %t249 = load i32, ptr %o51
  %t250 = getelementptr { i32, i32 }, ptr %o52, i32 0, i32 0
  store i32 %t249, ptr %t250
  %t251 = getelementptr { i32, i32 }, ptr %o52, i32 0, i32 0
  %t252 = load i32, ptr %t251
  %t253 = getelementptr { i32, i32 }, ptr %o52, i32 0, i32 1
  %t254 = load i32, ptr %t253
  %t255 = add i32 %t252, %t254
  store i32 %t255, ptr %o53
  %t256 = load i32, ptr %o53
  %t257 = getelementptr { i32, i32 }, ptr %o54, i32 0, i32 0
  store i32 %t256, ptr %t257
  %t258 = getelementptr { i32, i32 }, ptr %o54, i32 0, i32 0
  %t259 = load i32, ptr %t258
  %t260 = getelementptr { i32, i32 }, ptr %o54, i32 0, i32 1
  %t261 = load i32, ptr %t260
  %t262 = srem i32 %t259, %t261
  store i32 %t262, ptr %o55
  %t263 = load i32, ptr %o55
  %t264 = getelementptr { i32, i32 }, ptr %o56, i32 0, i32 0
  store i32 %t263, ptr %t264
  %t265 = getelementptr { i32, i32 }, ptr %o56, i32 0, i32 0
  %t266 = load i32, ptr %t265
  %t267 = getelementptr { i32, i32 }, ptr %o56, i32 0, i32 1
  %t268 = load i32, ptr %t267
  %t269 = mul i32 %t266, %t268
  store i32 %t269, ptr %o57
  %t270 = load i32, ptr %o57
  %t271 = getelementptr { i32, i32 }, ptr %o58, i32 0, i32 0
  store i32 %t270, ptr %t271
  %t272 = getelementptr { i32, i32 }, ptr %o58, i32 0, i32 0
  %t273 = load i32, ptr %t272
  %t274 = getelementptr { i32, i32 }, ptr %o58, i32 0, i32 1
  %t275 = load i32, ptr %t274
  %t276 = add i32 %t273, %t275
  store i32 %t276, ptr %o59
  %t277 = load i32, ptr %o59
  %t278 = getelementptr { i32, i32 }, ptr %o60, i32 0, i32 0
  store i32 %t277, ptr %t278
  %t279 = getelementptr { i32, i32 }, ptr %o60, i32 0, i32 0
  %t280 = load i32, ptr %t279
  %t281 = getelementptr { i32, i32 }, ptr %o60, i32 0, i32 1
  %t282 = load i32, ptr %t281
  %t283 = srem i32 %t280, %t282
  store i32 %t283, ptr %o61
  %t284 = load i32, ptr %o61
  %t285 = getelementptr { i32, i32 }, ptr %o62, i32 0, i32 0
  store i32 %t284, ptr %t285
  %t286 = getelementptr { i32, i32 }, ptr %o62, i32 0, i32 0
  %t287 = load i32, ptr %t286
  %t288 = getelementptr { i32, i32 }, ptr %o62, i32 0, i32 1
  %t289 = load i32, ptr %t288
  %t290 = sub i32 %t287, %t289
  store i32 %t290, ptr %o63
  %t291 = load i32, ptr %o63
  %t292 = sitofp i32 %t291 to float
  store float %t292, ptr %o64
  %t293 = load float, ptr %o64
  %t294 = getelementptr { float, float }, ptr %o65, i32 0, i32 0
  store float %t293, ptr %t294
  %t295 = getelementptr { float, float }, ptr %o65, i32 0, i32 0
  %t296 = load float, ptr %t295
  %t297 = getelementptr { float, float }, ptr %o65, i32 0, i32 1
  %t298 = load float, ptr %t297
  %t299 = fdiv float %t296, %t298
  store float %t299, ptr %o66
  %t300 = load float, ptr %o66
  %t301 = getelementptr { float, float }, ptr %o69, i32 0, i32 0
  store float %t300, ptr %t301
  %t302 = getelementptr { ptr, i32 }, ptr %o67, i32 0, i32 1
  %t303 = load i32, ptr %t302
  %t304 = sext i32 %t303 to i64
  %t305 = getelementptr [8 x float], ptr %o46, i64 0, i64 %t304
  %t306 = load float, ptr %t305
  store float %t306, ptr %o68
  %t307 = load float, ptr %o68
  %t308 = getelementptr { float, float }, ptr %o69, i32 0, i32 1
  store float %t307, ptr %t308
  %t309 = getelementptr { float, float }, ptr %o69, i32 0, i32 0
  %t310 = load float, ptr %t309
  %t311 = getelementptr { float, float }, ptr %o69, i32 0, i32 1
  %t312 = load float, ptr %t311
  %t313 = fadd float %t310, %t312
  store float %t313, ptr %o70
  %t314 = load float, ptr %o70
  %t315 = getelementptr { ptr, i32, float }, ptr %o75, i32 0, i32 2
  store float %t314, ptr %t315
  %t316 = getelementptr { i32, i32 }, ptr %o71, i32 0, i32 0
  %t317 = load i32, ptr %t316
  %t318 = getelementptr { i32, i32 }, ptr %o71, i32 0, i32 1
  %t319 = load i32, ptr %t318
  %t320 = mul i32 %t317, %t319
  store i32 %t320, ptr %o72
  %t321 = load i32, ptr %o72
  %t322 = getelementptr { i32, i32 }, ptr %o73, i32 0, i32 0
  store i32 %t321, ptr %t322
  %t323 = getelementptr { i32, i32 }, ptr %o73, i32 0, i32 0
  %t324 = load i32, ptr %t323
  %t325 = getelementptr { i32, i32 }, ptr %o73, i32 0, i32 1
  %t326 = load i32, ptr %t325
  %t327 = add i32 %t324, %t326
  store i32 %t327, ptr %o74
  %t328 = load i32, ptr %o74
  %t329 = getelementptr { ptr, i32, float }, ptr %o75, i32 0, i32 1
  store i32 %t328, ptr %t329
  %t330 = getelementptr { ptr, i32, float }, ptr %o75, i32 0, i32 1
  %t331 = load i32, ptr %t330
  %t332 = sext i32 %t331 to i64
  %t333 = icmp slt i64 %t332, 0
  %t334 = icmp sge i64 %t332, 128
  %t335 = or i1 %t333, %t334
  br i1 %t335, label %bb336, label %bb337
bb336:
  call void @mapal_trap(i32 1)
  unreachable
bb337:
  %t338 = getelementptr [128 x float], ptr %o45, i64 0, i64 %t332
  %t339 = getelementptr { ptr, i32, float }, ptr %o75, i32 0, i32 2
  %t340 = load float, ptr %t339
  store float %t340, ptr %t338
  %t341 = getelementptr { ptr, i32, float }, ptr %o102, i32 0, i32 0
  store ptr %o45, ptr %t341
  %t342 = getelementptr { i32, i32 }, ptr %o77, i32 0, i32 0
  %t343 = load i32, ptr %t342
  %t344 = getelementptr { i32, i32 }, ptr %o77, i32 0, i32 1
  %t345 = load i32, ptr %t344
  %t346 = mul i32 %t343, %t345
  store i32 %t346, ptr %o78
  %t347 = load i32, ptr %o78
  %t348 = getelementptr { i32, i32 }, ptr %o79, i32 0, i32 0
  store i32 %t347, ptr %t348
  %t349 = getelementptr { i32, i32 }, ptr %o79, i32 0, i32 0
  %t350 = load i32, ptr %t349
  %t351 = getelementptr { i32, i32 }, ptr %o79, i32 0, i32 1
  %t352 = load i32, ptr %t351
  %t353 = add i32 %t350, %t352
  store i32 %t353, ptr %o80
  %t354 = load i32, ptr %o80
  %t355 = getelementptr { i32, i32 }, ptr %o81, i32 0, i32 0
  store i32 %t354, ptr %t355
  %t356 = getelementptr { i32, i32 }, ptr %o81, i32 0, i32 0
  %t357 = load i32, ptr %t356
  %t358 = getelementptr { i32, i32 }, ptr %o81, i32 0, i32 1
  %t359 = load i32, ptr %t358
  %t360 = srem i32 %t357, %t359
  store i32 %t360, ptr %o82
  %t361 = load i32, ptr %o82
  %t362 = getelementptr { i32, i32 }, ptr %o83, i32 0, i32 0
  store i32 %t361, ptr %t362
  %t363 = getelementptr { i32, i32 }, ptr %o83, i32 0, i32 0
  %t364 = load i32, ptr %t363
  %t365 = getelementptr { i32, i32 }, ptr %o83, i32 0, i32 1
  %t366 = load i32, ptr %t365
  %t367 = mul i32 %t364, %t366
  store i32 %t367, ptr %o84
  %t368 = load i32, ptr %o84
  %t369 = getelementptr { i32, i32 }, ptr %o85, i32 0, i32 0
  store i32 %t368, ptr %t369
  %t370 = getelementptr { i32, i32 }, ptr %o85, i32 0, i32 0
  %t371 = load i32, ptr %t370
  %t372 = getelementptr { i32, i32 }, ptr %o85, i32 0, i32 1
  %t373 = load i32, ptr %t372
  %t374 = add i32 %t371, %t373
  store i32 %t374, ptr %o86
  %t375 = load i32, ptr %o86
  %t376 = getelementptr { i32, i32 }, ptr %o87, i32 0, i32 0
  store i32 %t375, ptr %t376
  %t377 = getelementptr { i32, i32 }, ptr %o87, i32 0, i32 0
  %t378 = load i32, ptr %t377
  %t379 = getelementptr { i32, i32 }, ptr %o87, i32 0, i32 1
  %t380 = load i32, ptr %t379
  %t381 = srem i32 %t378, %t380
  store i32 %t381, ptr %o88
  %t382 = load i32, ptr %o88
  %t383 = getelementptr { i32, i32 }, ptr %o89, i32 0, i32 0
  store i32 %t382, ptr %t383
  %t384 = getelementptr { i32, i32 }, ptr %o89, i32 0, i32 0
  %t385 = load i32, ptr %t384
  %t386 = getelementptr { i32, i32 }, ptr %o89, i32 0, i32 1
  %t387 = load i32, ptr %t386
  %t388 = sub i32 %t385, %t387
  store i32 %t388, ptr %o90
  %t389 = load i32, ptr %o90
  %t390 = sitofp i32 %t389 to float
  store float %t390, ptr %o91
  %t391 = load float, ptr %o91
  %t392 = getelementptr { float, float }, ptr %o92, i32 0, i32 0
  store float %t391, ptr %t392
  %t393 = getelementptr { float, float }, ptr %o92, i32 0, i32 0
  %t394 = load float, ptr %t393
  %t395 = getelementptr { float, float }, ptr %o92, i32 0, i32 1
  %t396 = load float, ptr %t395
  %t397 = fdiv float %t394, %t396
  store float %t397, ptr %o93
  %t398 = load float, ptr %o93
  %t399 = getelementptr { float, float }, ptr %o96, i32 0, i32 0
  store float %t398, ptr %t399
  %t400 = getelementptr { ptr, i32 }, ptr %o94, i32 0, i32 1
  %t401 = load i32, ptr %t400
  %t402 = sext i32 %t401 to i64
  %t403 = getelementptr [8 x float], ptr %o46, i64 0, i64 %t402
  %t404 = load float, ptr %t403
  store float %t404, ptr %o95
  %t405 = load float, ptr %o95
  %t406 = getelementptr { float, float }, ptr %o96, i32 0, i32 1
  store float %t405, ptr %t406
  %t407 = getelementptr { float, float }, ptr %o96, i32 0, i32 0
  %t408 = load float, ptr %t407
  %t409 = getelementptr { float, float }, ptr %o96, i32 0, i32 1
  %t410 = load float, ptr %t409
  %t411 = fadd float %t408, %t410
  store float %t411, ptr %o97
  %t412 = load float, ptr %o97
  %t413 = getelementptr { ptr, i32, float }, ptr %o102, i32 0, i32 2
  store float %t412, ptr %t413
  %t414 = getelementptr { i32, i32 }, ptr %o98, i32 0, i32 0
  %t415 = load i32, ptr %t414
  %t416 = getelementptr { i32, i32 }, ptr %o98, i32 0, i32 1
  %t417 = load i32, ptr %t416
  %t418 = mul i32 %t415, %t417
  store i32 %t418, ptr %o99
  %t419 = load i32, ptr %o99
  %t420 = getelementptr { i32, i32 }, ptr %o100, i32 0, i32 0
  store i32 %t419, ptr %t420
  %t421 = getelementptr { i32, i32 }, ptr %o100, i32 0, i32 0
  %t422 = load i32, ptr %t421
  %t423 = getelementptr { i32, i32 }, ptr %o100, i32 0, i32 1
  %t424 = load i32, ptr %t423
  %t425 = add i32 %t422, %t424
  store i32 %t425, ptr %o101
  %t426 = load i32, ptr %o101
  %t427 = getelementptr { ptr, i32, float }, ptr %o102, i32 0, i32 1
  store i32 %t426, ptr %t427
  %t428 = getelementptr { ptr, i32, float }, ptr %o102, i32 0, i32 1
  %t429 = load i32, ptr %t428
  %t430 = sext i32 %t429 to i64
  %t431 = icmp slt i64 %t430, 0
  %t432 = icmp sge i64 %t430, 128
  %t433 = or i1 %t431, %t432
  br i1 %t433, label %bb434, label %bb435
bb434:
  call void @mapal_trap(i32 1)
  unreachable
bb435:
  %t436 = getelementptr [128 x float], ptr %o45, i64 0, i64 %t430
  %t437 = getelementptr { ptr, i32, float }, ptr %o102, i32 0, i32 2
  %t438 = load float, ptr %t437
  store float %t438, ptr %t436
  %t439 = getelementptr { ptr, i32, float }, ptr %o129, i32 0, i32 0
  store ptr %o45, ptr %t439
  %t440 = getelementptr { i32, i32 }, ptr %o104, i32 0, i32 0
  %t441 = load i32, ptr %t440
  %t442 = getelementptr { i32, i32 }, ptr %o104, i32 0, i32 1
  %t443 = load i32, ptr %t442
  %t444 = mul i32 %t441, %t443
  store i32 %t444, ptr %o105
  %t445 = load i32, ptr %o105
  %t446 = getelementptr { i32, i32 }, ptr %o106, i32 0, i32 0
  store i32 %t445, ptr %t446
  %t447 = getelementptr { i32, i32 }, ptr %o106, i32 0, i32 0
  %t448 = load i32, ptr %t447
  %t449 = getelementptr { i32, i32 }, ptr %o106, i32 0, i32 1
  %t450 = load i32, ptr %t449
  %t451 = add i32 %t448, %t450
  store i32 %t451, ptr %o107
  %t452 = load i32, ptr %o107
  %t453 = getelementptr { i32, i32 }, ptr %o108, i32 0, i32 0
  store i32 %t452, ptr %t453
  %t454 = getelementptr { i32, i32 }, ptr %o108, i32 0, i32 0
  %t455 = load i32, ptr %t454
  %t456 = getelementptr { i32, i32 }, ptr %o108, i32 0, i32 1
  %t457 = load i32, ptr %t456
  %t458 = srem i32 %t455, %t457
  store i32 %t458, ptr %o109
  %t459 = load i32, ptr %o109
  %t460 = getelementptr { i32, i32 }, ptr %o110, i32 0, i32 0
  store i32 %t459, ptr %t460
  %t461 = getelementptr { i32, i32 }, ptr %o110, i32 0, i32 0
  %t462 = load i32, ptr %t461
  %t463 = getelementptr { i32, i32 }, ptr %o110, i32 0, i32 1
  %t464 = load i32, ptr %t463
  %t465 = mul i32 %t462, %t464
  store i32 %t465, ptr %o111
  %t466 = load i32, ptr %o111
  %t467 = getelementptr { i32, i32 }, ptr %o112, i32 0, i32 0
  store i32 %t466, ptr %t467
  %t468 = getelementptr { i32, i32 }, ptr %o112, i32 0, i32 0
  %t469 = load i32, ptr %t468
  %t470 = getelementptr { i32, i32 }, ptr %o112, i32 0, i32 1
  %t471 = load i32, ptr %t470
  %t472 = add i32 %t469, %t471
  store i32 %t472, ptr %o113
  %t473 = load i32, ptr %o113
  %t474 = getelementptr { i32, i32 }, ptr %o114, i32 0, i32 0
  store i32 %t473, ptr %t474
  %t475 = getelementptr { i32, i32 }, ptr %o114, i32 0, i32 0
  %t476 = load i32, ptr %t475
  %t477 = getelementptr { i32, i32 }, ptr %o114, i32 0, i32 1
  %t478 = load i32, ptr %t477
  %t479 = srem i32 %t476, %t478
  store i32 %t479, ptr %o115
  %t480 = load i32, ptr %o115
  %t481 = getelementptr { i32, i32 }, ptr %o116, i32 0, i32 0
  store i32 %t480, ptr %t481
  %t482 = getelementptr { i32, i32 }, ptr %o116, i32 0, i32 0
  %t483 = load i32, ptr %t482
  %t484 = getelementptr { i32, i32 }, ptr %o116, i32 0, i32 1
  %t485 = load i32, ptr %t484
  %t486 = sub i32 %t483, %t485
  store i32 %t486, ptr %o117
  %t487 = load i32, ptr %o117
  %t488 = sitofp i32 %t487 to float
  store float %t488, ptr %o118
  %t489 = load float, ptr %o118
  %t490 = getelementptr { float, float }, ptr %o119, i32 0, i32 0
  store float %t489, ptr %t490
  %t491 = getelementptr { float, float }, ptr %o119, i32 0, i32 0
  %t492 = load float, ptr %t491
  %t493 = getelementptr { float, float }, ptr %o119, i32 0, i32 1
  %t494 = load float, ptr %t493
  %t495 = fdiv float %t492, %t494
  store float %t495, ptr %o120
  %t496 = load float, ptr %o120
  %t497 = getelementptr { float, float }, ptr %o123, i32 0, i32 0
  store float %t496, ptr %t497
  %t498 = getelementptr { ptr, i32 }, ptr %o121, i32 0, i32 1
  %t499 = load i32, ptr %t498
  %t500 = sext i32 %t499 to i64
  %t501 = getelementptr [8 x float], ptr %o46, i64 0, i64 %t500
  %t502 = load float, ptr %t501
  store float %t502, ptr %o122
  %t503 = load float, ptr %o122
  %t504 = getelementptr { float, float }, ptr %o123, i32 0, i32 1
  store float %t503, ptr %t504
  %t505 = getelementptr { float, float }, ptr %o123, i32 0, i32 0
  %t506 = load float, ptr %t505
  %t507 = getelementptr { float, float }, ptr %o123, i32 0, i32 1
  %t508 = load float, ptr %t507
  %t509 = fadd float %t506, %t508
  store float %t509, ptr %o124
  %t510 = load float, ptr %o124
  %t511 = getelementptr { ptr, i32, float }, ptr %o129, i32 0, i32 2
  store float %t510, ptr %t511
  %t512 = getelementptr { i32, i32 }, ptr %o125, i32 0, i32 0
  %t513 = load i32, ptr %t512
  %t514 = getelementptr { i32, i32 }, ptr %o125, i32 0, i32 1
  %t515 = load i32, ptr %t514
  %t516 = mul i32 %t513, %t515
  store i32 %t516, ptr %o126
  %t517 = load i32, ptr %o126
  %t518 = getelementptr { i32, i32 }, ptr %o127, i32 0, i32 0
  store i32 %t517, ptr %t518
  %t519 = getelementptr { i32, i32 }, ptr %o127, i32 0, i32 0
  %t520 = load i32, ptr %t519
  %t521 = getelementptr { i32, i32 }, ptr %o127, i32 0, i32 1
  %t522 = load i32, ptr %t521
  %t523 = add i32 %t520, %t522
  store i32 %t523, ptr %o128
  %t524 = load i32, ptr %o128
  %t525 = getelementptr { ptr, i32, float }, ptr %o129, i32 0, i32 1
  store i32 %t524, ptr %t525
  %t526 = getelementptr { ptr, i32, float }, ptr %o129, i32 0, i32 1
  %t527 = load i32, ptr %t526
  %t528 = sext i32 %t527 to i64
  %t529 = icmp slt i64 %t528, 0
  %t530 = icmp sge i64 %t528, 128
  %t531 = or i1 %t529, %t530
  br i1 %t531, label %bb532, label %bb533
bb532:
  call void @mapal_trap(i32 1)
  unreachable
bb533:
  %t534 = getelementptr [128 x float], ptr %o45, i64 0, i64 %t528
  %t535 = getelementptr { ptr, i32, float }, ptr %o129, i32 0, i32 2
  %t536 = load float, ptr %t535
  store float %t536, ptr %t534
  %t537 = getelementptr { ptr, i32, float }, ptr %o156, i32 0, i32 0
  store ptr %o45, ptr %t537
  %t538 = getelementptr { i32, i32 }, ptr %o131, i32 0, i32 0
  %t539 = load i32, ptr %t538
  %t540 = getelementptr { i32, i32 }, ptr %o131, i32 0, i32 1
  %t541 = load i32, ptr %t540
  %t542 = mul i32 %t539, %t541
  store i32 %t542, ptr %o132
  %t543 = load i32, ptr %o132
  %t544 = getelementptr { i32, i32 }, ptr %o133, i32 0, i32 0
  store i32 %t543, ptr %t544
  %t545 = getelementptr { i32, i32 }, ptr %o133, i32 0, i32 0
  %t546 = load i32, ptr %t545
  %t547 = getelementptr { i32, i32 }, ptr %o133, i32 0, i32 1
  %t548 = load i32, ptr %t547
  %t549 = add i32 %t546, %t548
  store i32 %t549, ptr %o134
  %t550 = load i32, ptr %o134
  %t551 = getelementptr { i32, i32 }, ptr %o135, i32 0, i32 0
  store i32 %t550, ptr %t551
  %t552 = getelementptr { i32, i32 }, ptr %o135, i32 0, i32 0
  %t553 = load i32, ptr %t552
  %t554 = getelementptr { i32, i32 }, ptr %o135, i32 0, i32 1
  %t555 = load i32, ptr %t554
  %t556 = srem i32 %t553, %t555
  store i32 %t556, ptr %o136
  %t557 = load i32, ptr %o136
  %t558 = getelementptr { i32, i32 }, ptr %o137, i32 0, i32 0
  store i32 %t557, ptr %t558
  %t559 = getelementptr { i32, i32 }, ptr %o137, i32 0, i32 0
  %t560 = load i32, ptr %t559
  %t561 = getelementptr { i32, i32 }, ptr %o137, i32 0, i32 1
  %t562 = load i32, ptr %t561
  %t563 = mul i32 %t560, %t562
  store i32 %t563, ptr %o138
  %t564 = load i32, ptr %o138
  %t565 = getelementptr { i32, i32 }, ptr %o139, i32 0, i32 0
  store i32 %t564, ptr %t565
  %t566 = getelementptr { i32, i32 }, ptr %o139, i32 0, i32 0
  %t567 = load i32, ptr %t566
  %t568 = getelementptr { i32, i32 }, ptr %o139, i32 0, i32 1
  %t569 = load i32, ptr %t568
  %t570 = add i32 %t567, %t569
  store i32 %t570, ptr %o140
  %t571 = load i32, ptr %o140
  %t572 = getelementptr { i32, i32 }, ptr %o141, i32 0, i32 0
  store i32 %t571, ptr %t572
  %t573 = getelementptr { i32, i32 }, ptr %o141, i32 0, i32 0
  %t574 = load i32, ptr %t573
  %t575 = getelementptr { i32, i32 }, ptr %o141, i32 0, i32 1
  %t576 = load i32, ptr %t575
  %t577 = srem i32 %t574, %t576
  store i32 %t577, ptr %o142
  %t578 = load i32, ptr %o142
  %t579 = getelementptr { i32, i32 }, ptr %o143, i32 0, i32 0
  store i32 %t578, ptr %t579
  %t580 = getelementptr { i32, i32 }, ptr %o143, i32 0, i32 0
  %t581 = load i32, ptr %t580
  %t582 = getelementptr { i32, i32 }, ptr %o143, i32 0, i32 1
  %t583 = load i32, ptr %t582
  %t584 = sub i32 %t581, %t583
  store i32 %t584, ptr %o144
  %t585 = load i32, ptr %o144
  %t586 = sitofp i32 %t585 to float
  store float %t586, ptr %o145
  %t587 = load float, ptr %o145
  %t588 = getelementptr { float, float }, ptr %o146, i32 0, i32 0
  store float %t587, ptr %t588
  %t589 = getelementptr { float, float }, ptr %o146, i32 0, i32 0
  %t590 = load float, ptr %t589
  %t591 = getelementptr { float, float }, ptr %o146, i32 0, i32 1
  %t592 = load float, ptr %t591
  %t593 = fdiv float %t590, %t592
  store float %t593, ptr %o147
  %t594 = load float, ptr %o147
  %t595 = getelementptr { float, float }, ptr %o150, i32 0, i32 0
  store float %t594, ptr %t595
  %t596 = getelementptr { ptr, i32 }, ptr %o148, i32 0, i32 1
  %t597 = load i32, ptr %t596
  %t598 = sext i32 %t597 to i64
  %t599 = getelementptr [8 x float], ptr %o46, i64 0, i64 %t598
  %t600 = load float, ptr %t599
  store float %t600, ptr %o149
  %t601 = load float, ptr %o149
  %t602 = getelementptr { float, float }, ptr %o150, i32 0, i32 1
  store float %t601, ptr %t602
  %t603 = getelementptr { float, float }, ptr %o150, i32 0, i32 0
  %t604 = load float, ptr %t603
  %t605 = getelementptr { float, float }, ptr %o150, i32 0, i32 1
  %t606 = load float, ptr %t605
  %t607 = fadd float %t604, %t606
  store float %t607, ptr %o151
  %t608 = load float, ptr %o151
  %t609 = getelementptr { ptr, i32, float }, ptr %o156, i32 0, i32 2
  store float %t608, ptr %t609
  %t610 = getelementptr { i32, i32 }, ptr %o152, i32 0, i32 0
  %t611 = load i32, ptr %t610
  %t612 = getelementptr { i32, i32 }, ptr %o152, i32 0, i32 1
  %t613 = load i32, ptr %t612
  %t614 = mul i32 %t611, %t613
  store i32 %t614, ptr %o153
  %t615 = load i32, ptr %o153
  %t616 = getelementptr { i32, i32 }, ptr %o154, i32 0, i32 0
  store i32 %t615, ptr %t616
  %t617 = getelementptr { i32, i32 }, ptr %o154, i32 0, i32 0
  %t618 = load i32, ptr %t617
  %t619 = getelementptr { i32, i32 }, ptr %o154, i32 0, i32 1
  %t620 = load i32, ptr %t619
  %t621 = add i32 %t618, %t620
  store i32 %t621, ptr %o155
  %t622 = load i32, ptr %o155
  %t623 = getelementptr { ptr, i32, float }, ptr %o156, i32 0, i32 1
  store i32 %t622, ptr %t623
  %t624 = getelementptr { ptr, i32, float }, ptr %o156, i32 0, i32 1
  %t625 = load i32, ptr %t624
  %t626 = sext i32 %t625 to i64
  %t627 = icmp slt i64 %t626, 0
  %t628 = icmp sge i64 %t626, 128
  %t629 = or i1 %t627, %t628
  br i1 %t629, label %bb630, label %bb631
bb630:
  call void @mapal_trap(i32 1)
  unreachable
bb631:
  %t632 = getelementptr [128 x float], ptr %o45, i64 0, i64 %t626
  %t633 = getelementptr { ptr, i32, float }, ptr %o156, i32 0, i32 2
  %t634 = load float, ptr %t633
  store float %t634, ptr %t632
  %t635 = getelementptr { ptr, i32, float }, ptr %o183, i32 0, i32 0
  store ptr %o45, ptr %t635
  %t636 = getelementptr { i32, i32 }, ptr %o158, i32 0, i32 0
  %t637 = load i32, ptr %t636
  %t638 = getelementptr { i32, i32 }, ptr %o158, i32 0, i32 1
  %t639 = load i32, ptr %t638
  %t640 = mul i32 %t637, %t639
  store i32 %t640, ptr %o159
  %t641 = load i32, ptr %o159
  %t642 = getelementptr { i32, i32 }, ptr %o160, i32 0, i32 0
  store i32 %t641, ptr %t642
  %t643 = getelementptr { i32, i32 }, ptr %o160, i32 0, i32 0
  %t644 = load i32, ptr %t643
  %t645 = getelementptr { i32, i32 }, ptr %o160, i32 0, i32 1
  %t646 = load i32, ptr %t645
  %t647 = add i32 %t644, %t646
  store i32 %t647, ptr %o161
  %t648 = load i32, ptr %o161
  %t649 = getelementptr { i32, i32 }, ptr %o162, i32 0, i32 0
  store i32 %t648, ptr %t649
  %t650 = getelementptr { i32, i32 }, ptr %o162, i32 0, i32 0
  %t651 = load i32, ptr %t650
  %t652 = getelementptr { i32, i32 }, ptr %o162, i32 0, i32 1
  %t653 = load i32, ptr %t652
  %t654 = srem i32 %t651, %t653
  store i32 %t654, ptr %o163
  %t655 = load i32, ptr %o163
  %t656 = getelementptr { i32, i32 }, ptr %o164, i32 0, i32 0
  store i32 %t655, ptr %t656
  %t657 = getelementptr { i32, i32 }, ptr %o164, i32 0, i32 0
  %t658 = load i32, ptr %t657
  %t659 = getelementptr { i32, i32 }, ptr %o164, i32 0, i32 1
  %t660 = load i32, ptr %t659
  %t661 = mul i32 %t658, %t660
  store i32 %t661, ptr %o165
  %t662 = load i32, ptr %o165
  %t663 = getelementptr { i32, i32 }, ptr %o166, i32 0, i32 0
  store i32 %t662, ptr %t663
  %t664 = getelementptr { i32, i32 }, ptr %o166, i32 0, i32 0
  %t665 = load i32, ptr %t664
  %t666 = getelementptr { i32, i32 }, ptr %o166, i32 0, i32 1
  %t667 = load i32, ptr %t666
  %t668 = add i32 %t665, %t667
  store i32 %t668, ptr %o167
  %t669 = load i32, ptr %o167
  %t670 = getelementptr { i32, i32 }, ptr %o168, i32 0, i32 0
  store i32 %t669, ptr %t670
  %t671 = getelementptr { i32, i32 }, ptr %o168, i32 0, i32 0
  %t672 = load i32, ptr %t671
  %t673 = getelementptr { i32, i32 }, ptr %o168, i32 0, i32 1
  %t674 = load i32, ptr %t673
  %t675 = srem i32 %t672, %t674
  store i32 %t675, ptr %o169
  %t676 = load i32, ptr %o169
  %t677 = getelementptr { i32, i32 }, ptr %o170, i32 0, i32 0
  store i32 %t676, ptr %t677
  %t678 = getelementptr { i32, i32 }, ptr %o170, i32 0, i32 0
  %t679 = load i32, ptr %t678
  %t680 = getelementptr { i32, i32 }, ptr %o170, i32 0, i32 1
  %t681 = load i32, ptr %t680
  %t682 = sub i32 %t679, %t681
  store i32 %t682, ptr %o171
  %t683 = load i32, ptr %o171
  %t684 = sitofp i32 %t683 to float
  store float %t684, ptr %o172
  %t685 = load float, ptr %o172
  %t686 = getelementptr { float, float }, ptr %o173, i32 0, i32 0
  store float %t685, ptr %t686
  %t687 = getelementptr { float, float }, ptr %o173, i32 0, i32 0
  %t688 = load float, ptr %t687
  %t689 = getelementptr { float, float }, ptr %o173, i32 0, i32 1
  %t690 = load float, ptr %t689
  %t691 = fdiv float %t688, %t690
  store float %t691, ptr %o174
  %t692 = load float, ptr %o174
  %t693 = getelementptr { float, float }, ptr %o177, i32 0, i32 0
  store float %t692, ptr %t693
  %t694 = getelementptr { ptr, i32 }, ptr %o175, i32 0, i32 1
  %t695 = load i32, ptr %t694
  %t696 = sext i32 %t695 to i64
  %t697 = getelementptr [8 x float], ptr %o46, i64 0, i64 %t696
  %t698 = load float, ptr %t697
  store float %t698, ptr %o176
  %t699 = load float, ptr %o176
  %t700 = getelementptr { float, float }, ptr %o177, i32 0, i32 1
  store float %t699, ptr %t700
  %t701 = getelementptr { float, float }, ptr %o177, i32 0, i32 0
  %t702 = load float, ptr %t701
  %t703 = getelementptr { float, float }, ptr %o177, i32 0, i32 1
  %t704 = load float, ptr %t703
  %t705 = fadd float %t702, %t704
  store float %t705, ptr %o178
  %t706 = load float, ptr %o178
  %t707 = getelementptr { ptr, i32, float }, ptr %o183, i32 0, i32 2
  store float %t706, ptr %t707
  %t708 = getelementptr { i32, i32 }, ptr %o179, i32 0, i32 0
  %t709 = load i32, ptr %t708
  %t710 = getelementptr { i32, i32 }, ptr %o179, i32 0, i32 1
  %t711 = load i32, ptr %t710
  %t712 = mul i32 %t709, %t711
  store i32 %t712, ptr %o180
  %t713 = load i32, ptr %o180
  %t714 = getelementptr { i32, i32 }, ptr %o181, i32 0, i32 0
  store i32 %t713, ptr %t714
  %t715 = getelementptr { i32, i32 }, ptr %o181, i32 0, i32 0
  %t716 = load i32, ptr %t715
  %t717 = getelementptr { i32, i32 }, ptr %o181, i32 0, i32 1
  %t718 = load i32, ptr %t717
  %t719 = add i32 %t716, %t718
  store i32 %t719, ptr %o182
  %t720 = load i32, ptr %o182
  %t721 = getelementptr { ptr, i32, float }, ptr %o183, i32 0, i32 1
  store i32 %t720, ptr %t721
  %t722 = getelementptr { ptr, i32, float }, ptr %o183, i32 0, i32 1
  %t723 = load i32, ptr %t722
  %t724 = sext i32 %t723 to i64
  %t725 = icmp slt i64 %t724, 0
  %t726 = icmp sge i64 %t724, 128
  %t727 = or i1 %t725, %t726
  br i1 %t727, label %bb728, label %bb729
bb728:
  call void @mapal_trap(i32 1)
  unreachable
bb729:
  %t730 = getelementptr [128 x float], ptr %o45, i64 0, i64 %t724
  %t731 = getelementptr { ptr, i32, float }, ptr %o183, i32 0, i32 2
  %t732 = load float, ptr %t731
  store float %t732, ptr %t730
  %t733 = getelementptr { ptr, i32, float }, ptr %o210, i32 0, i32 0
  store ptr %o45, ptr %t733
  %t734 = getelementptr { i32, i32 }, ptr %o185, i32 0, i32 0
  %t735 = load i32, ptr %t734
  %t736 = getelementptr { i32, i32 }, ptr %o185, i32 0, i32 1
  %t737 = load i32, ptr %t736
  %t738 = mul i32 %t735, %t737
  store i32 %t738, ptr %o186
  %t739 = load i32, ptr %o186
  %t740 = getelementptr { i32, i32 }, ptr %o187, i32 0, i32 0
  store i32 %t739, ptr %t740
  %t741 = getelementptr { i32, i32 }, ptr %o187, i32 0, i32 0
  %t742 = load i32, ptr %t741
  %t743 = getelementptr { i32, i32 }, ptr %o187, i32 0, i32 1
  %t744 = load i32, ptr %t743
  %t745 = add i32 %t742, %t744
  store i32 %t745, ptr %o188
  %t746 = load i32, ptr %o188
  %t747 = getelementptr { i32, i32 }, ptr %o189, i32 0, i32 0
  store i32 %t746, ptr %t747
  %t748 = getelementptr { i32, i32 }, ptr %o189, i32 0, i32 0
  %t749 = load i32, ptr %t748
  %t750 = getelementptr { i32, i32 }, ptr %o189, i32 0, i32 1
  %t751 = load i32, ptr %t750
  %t752 = srem i32 %t749, %t751
  store i32 %t752, ptr %o190
  %t753 = load i32, ptr %o190
  %t754 = getelementptr { i32, i32 }, ptr %o191, i32 0, i32 0
  store i32 %t753, ptr %t754
  %t755 = getelementptr { i32, i32 }, ptr %o191, i32 0, i32 0
  %t756 = load i32, ptr %t755
  %t757 = getelementptr { i32, i32 }, ptr %o191, i32 0, i32 1
  %t758 = load i32, ptr %t757
  %t759 = mul i32 %t756, %t758
  store i32 %t759, ptr %o192
  %t760 = load i32, ptr %o192
  %t761 = getelementptr { i32, i32 }, ptr %o193, i32 0, i32 0
  store i32 %t760, ptr %t761
  %t762 = getelementptr { i32, i32 }, ptr %o193, i32 0, i32 0
  %t763 = load i32, ptr %t762
  %t764 = getelementptr { i32, i32 }, ptr %o193, i32 0, i32 1
  %t765 = load i32, ptr %t764
  %t766 = add i32 %t763, %t765
  store i32 %t766, ptr %o194
  %t767 = load i32, ptr %o194
  %t768 = getelementptr { i32, i32 }, ptr %o195, i32 0, i32 0
  store i32 %t767, ptr %t768
  %t769 = getelementptr { i32, i32 }, ptr %o195, i32 0, i32 0
  %t770 = load i32, ptr %t769
  %t771 = getelementptr { i32, i32 }, ptr %o195, i32 0, i32 1
  %t772 = load i32, ptr %t771
  %t773 = srem i32 %t770, %t772
  store i32 %t773, ptr %o196
  %t774 = load i32, ptr %o196
  %t775 = getelementptr { i32, i32 }, ptr %o197, i32 0, i32 0
  store i32 %t774, ptr %t775
  %t776 = getelementptr { i32, i32 }, ptr %o197, i32 0, i32 0
  %t777 = load i32, ptr %t776
  %t778 = getelementptr { i32, i32 }, ptr %o197, i32 0, i32 1
  %t779 = load i32, ptr %t778
  %t780 = sub i32 %t777, %t779
  store i32 %t780, ptr %o198
  %t781 = load i32, ptr %o198
  %t782 = sitofp i32 %t781 to float
  store float %t782, ptr %o199
  %t783 = load float, ptr %o199
  %t784 = getelementptr { float, float }, ptr %o200, i32 0, i32 0
  store float %t783, ptr %t784
  %t785 = getelementptr { float, float }, ptr %o200, i32 0, i32 0
  %t786 = load float, ptr %t785
  %t787 = getelementptr { float, float }, ptr %o200, i32 0, i32 1
  %t788 = load float, ptr %t787
  %t789 = fdiv float %t786, %t788
  store float %t789, ptr %o201
  %t790 = load float, ptr %o201
  %t791 = getelementptr { float, float }, ptr %o204, i32 0, i32 0
  store float %t790, ptr %t791
  %t792 = getelementptr { ptr, i32 }, ptr %o202, i32 0, i32 1
  %t793 = load i32, ptr %t792
  %t794 = sext i32 %t793 to i64
  %t795 = getelementptr [8 x float], ptr %o46, i64 0, i64 %t794
  %t796 = load float, ptr %t795
  store float %t796, ptr %o203
  %t797 = load float, ptr %o203
  %t798 = getelementptr { float, float }, ptr %o204, i32 0, i32 1
  store float %t797, ptr %t798
  %t799 = getelementptr { float, float }, ptr %o204, i32 0, i32 0
  %t800 = load float, ptr %t799
  %t801 = getelementptr { float, float }, ptr %o204, i32 0, i32 1
  %t802 = load float, ptr %t801
  %t803 = fadd float %t800, %t802
  store float %t803, ptr %o205
  %t804 = load float, ptr %o205
  %t805 = getelementptr { ptr, i32, float }, ptr %o210, i32 0, i32 2
  store float %t804, ptr %t805
  %t806 = getelementptr { i32, i32 }, ptr %o206, i32 0, i32 0
  %t807 = load i32, ptr %t806
  %t808 = getelementptr { i32, i32 }, ptr %o206, i32 0, i32 1
  %t809 = load i32, ptr %t808
  %t810 = mul i32 %t807, %t809
  store i32 %t810, ptr %o207
  %t811 = load i32, ptr %o207
  %t812 = getelementptr { i32, i32 }, ptr %o208, i32 0, i32 0
  store i32 %t811, ptr %t812
  %t813 = getelementptr { i32, i32 }, ptr %o208, i32 0, i32 0
  %t814 = load i32, ptr %t813
  %t815 = getelementptr { i32, i32 }, ptr %o208, i32 0, i32 1
  %t816 = load i32, ptr %t815
  %t817 = add i32 %t814, %t816
  store i32 %t817, ptr %o209
  %t818 = load i32, ptr %o209
  %t819 = getelementptr { ptr, i32, float }, ptr %o210, i32 0, i32 1
  store i32 %t818, ptr %t819
  %t820 = getelementptr { ptr, i32, float }, ptr %o210, i32 0, i32 1
  %t821 = load i32, ptr %t820
  %t822 = sext i32 %t821 to i64
  %t823 = icmp slt i64 %t822, 0
  %t824 = icmp sge i64 %t822, 128
  %t825 = or i1 %t823, %t824
  br i1 %t825, label %bb826, label %bb827
bb826:
  call void @mapal_trap(i32 1)
  unreachable
bb827:
  %t828 = getelementptr [128 x float], ptr %o45, i64 0, i64 %t822
  %t829 = getelementptr { ptr, i32, float }, ptr %o210, i32 0, i32 2
  %t830 = load float, ptr %t829
  store float %t830, ptr %t828
  %t831 = getelementptr { ptr, i32, float }, ptr %o237, i32 0, i32 0
  store ptr %o45, ptr %t831
  %t832 = getelementptr { i32, i32 }, ptr %o212, i32 0, i32 0
  %t833 = load i32, ptr %t832
  %t834 = getelementptr { i32, i32 }, ptr %o212, i32 0, i32 1
  %t835 = load i32, ptr %t834
  %t836 = mul i32 %t833, %t835
  store i32 %t836, ptr %o213
  %t837 = load i32, ptr %o213
  %t838 = getelementptr { i32, i32 }, ptr %o214, i32 0, i32 0
  store i32 %t837, ptr %t838
  %t839 = getelementptr { i32, i32 }, ptr %o214, i32 0, i32 0
  %t840 = load i32, ptr %t839
  %t841 = getelementptr { i32, i32 }, ptr %o214, i32 0, i32 1
  %t842 = load i32, ptr %t841
  %t843 = add i32 %t840, %t842
  store i32 %t843, ptr %o215
  %t844 = load i32, ptr %o215
  %t845 = getelementptr { i32, i32 }, ptr %o216, i32 0, i32 0
  store i32 %t844, ptr %t845
  %t846 = getelementptr { i32, i32 }, ptr %o216, i32 0, i32 0
  %t847 = load i32, ptr %t846
  %t848 = getelementptr { i32, i32 }, ptr %o216, i32 0, i32 1
  %t849 = load i32, ptr %t848
  %t850 = srem i32 %t847, %t849
  store i32 %t850, ptr %o217
  %t851 = load i32, ptr %o217
  %t852 = getelementptr { i32, i32 }, ptr %o218, i32 0, i32 0
  store i32 %t851, ptr %t852
  %t853 = getelementptr { i32, i32 }, ptr %o218, i32 0, i32 0
  %t854 = load i32, ptr %t853
  %t855 = getelementptr { i32, i32 }, ptr %o218, i32 0, i32 1
  %t856 = load i32, ptr %t855
  %t857 = mul i32 %t854, %t856
  store i32 %t857, ptr %o219
  %t858 = load i32, ptr %o219
  %t859 = getelementptr { i32, i32 }, ptr %o220, i32 0, i32 0
  store i32 %t858, ptr %t859
  %t860 = getelementptr { i32, i32 }, ptr %o220, i32 0, i32 0
  %t861 = load i32, ptr %t860
  %t862 = getelementptr { i32, i32 }, ptr %o220, i32 0, i32 1
  %t863 = load i32, ptr %t862
  %t864 = add i32 %t861, %t863
  store i32 %t864, ptr %o221
  %t865 = load i32, ptr %o221
  %t866 = getelementptr { i32, i32 }, ptr %o222, i32 0, i32 0
  store i32 %t865, ptr %t866
  %t867 = getelementptr { i32, i32 }, ptr %o222, i32 0, i32 0
  %t868 = load i32, ptr %t867
  %t869 = getelementptr { i32, i32 }, ptr %o222, i32 0, i32 1
  %t870 = load i32, ptr %t869
  %t871 = srem i32 %t868, %t870
  store i32 %t871, ptr %o223
  %t872 = load i32, ptr %o223
  %t873 = getelementptr { i32, i32 }, ptr %o224, i32 0, i32 0
  store i32 %t872, ptr %t873
  %t874 = getelementptr { i32, i32 }, ptr %o224, i32 0, i32 0
  %t875 = load i32, ptr %t874
  %t876 = getelementptr { i32, i32 }, ptr %o224, i32 0, i32 1
  %t877 = load i32, ptr %t876
  %t878 = sub i32 %t875, %t877
  store i32 %t878, ptr %o225
  %t879 = load i32, ptr %o225
  %t880 = sitofp i32 %t879 to float
  store float %t880, ptr %o226
  %t881 = load float, ptr %o226
  %t882 = getelementptr { float, float }, ptr %o227, i32 0, i32 0
  store float %t881, ptr %t882
  %t883 = getelementptr { float, float }, ptr %o227, i32 0, i32 0
  %t884 = load float, ptr %t883
  %t885 = getelementptr { float, float }, ptr %o227, i32 0, i32 1
  %t886 = load float, ptr %t885
  %t887 = fdiv float %t884, %t886
  store float %t887, ptr %o228
  %t888 = load float, ptr %o228
  %t889 = getelementptr { float, float }, ptr %o231, i32 0, i32 0
  store float %t888, ptr %t889
  %t890 = getelementptr { ptr, i32 }, ptr %o229, i32 0, i32 1
  %t891 = load i32, ptr %t890
  %t892 = sext i32 %t891 to i64
  %t893 = getelementptr [8 x float], ptr %o46, i64 0, i64 %t892
  %t894 = load float, ptr %t893
  store float %t894, ptr %o230
  %t895 = load float, ptr %o230
  %t896 = getelementptr { float, float }, ptr %o231, i32 0, i32 1
  store float %t895, ptr %t896
  %t897 = getelementptr { float, float }, ptr %o231, i32 0, i32 0
  %t898 = load float, ptr %t897
  %t899 = getelementptr { float, float }, ptr %o231, i32 0, i32 1
  %t900 = load float, ptr %t899
  %t901 = fadd float %t898, %t900
  store float %t901, ptr %o232
  %t902 = load float, ptr %o232
  %t903 = getelementptr { ptr, i32, float }, ptr %o237, i32 0, i32 2
  store float %t902, ptr %t903
  %t904 = getelementptr { i32, i32 }, ptr %o233, i32 0, i32 0
  %t905 = load i32, ptr %t904
  %t906 = getelementptr { i32, i32 }, ptr %o233, i32 0, i32 1
  %t907 = load i32, ptr %t906
  %t908 = mul i32 %t905, %t907
  store i32 %t908, ptr %o234
  %t909 = load i32, ptr %o234
  %t910 = getelementptr { i32, i32 }, ptr %o235, i32 0, i32 0
  store i32 %t909, ptr %t910
  %t911 = getelementptr { i32, i32 }, ptr %o235, i32 0, i32 0
  %t912 = load i32, ptr %t911
  %t913 = getelementptr { i32, i32 }, ptr %o235, i32 0, i32 1
  %t914 = load i32, ptr %t913
  %t915 = add i32 %t912, %t914
  store i32 %t915, ptr %o236
  %t916 = load i32, ptr %o236
  %t917 = getelementptr { ptr, i32, float }, ptr %o237, i32 0, i32 1
  store i32 %t916, ptr %t917
  %t918 = getelementptr { ptr, i32, float }, ptr %o237, i32 0, i32 1
  %t919 = load i32, ptr %t918
  %t920 = sext i32 %t919 to i64
  %t921 = icmp slt i64 %t920, 0
  %t922 = icmp sge i64 %t920, 128
  %t923 = or i1 %t921, %t922
  br i1 %t923, label %bb924, label %bb925
bb924:
  call void @mapal_trap(i32 1)
  unreachable
bb925:
  %t926 = getelementptr [128 x float], ptr %o45, i64 0, i64 %t920
  %t927 = getelementptr { ptr, i32, float }, ptr %o237, i32 0, i32 2
  %t928 = load float, ptr %t927
  store float %t928, ptr %t926
  %t929 = getelementptr { ptr, i32, float }, ptr %o264, i32 0, i32 0
  store ptr %o45, ptr %t929
  %t930 = getelementptr { i32, i32 }, ptr %o239, i32 0, i32 0
  %t931 = load i32, ptr %t930
  %t932 = getelementptr { i32, i32 }, ptr %o239, i32 0, i32 1
  %t933 = load i32, ptr %t932
  %t934 = mul i32 %t931, %t933
  store i32 %t934, ptr %o240
  %t935 = load i32, ptr %o240
  %t936 = getelementptr { i32, i32 }, ptr %o241, i32 0, i32 0
  store i32 %t935, ptr %t936
  %t937 = getelementptr { i32, i32 }, ptr %o241, i32 0, i32 0
  %t938 = load i32, ptr %t937
  %t939 = getelementptr { i32, i32 }, ptr %o241, i32 0, i32 1
  %t940 = load i32, ptr %t939
  %t941 = add i32 %t938, %t940
  store i32 %t941, ptr %o242
  %t942 = load i32, ptr %o242
  %t943 = getelementptr { i32, i32 }, ptr %o243, i32 0, i32 0
  store i32 %t942, ptr %t943
  %t944 = getelementptr { i32, i32 }, ptr %o243, i32 0, i32 0
  %t945 = load i32, ptr %t944
  %t946 = getelementptr { i32, i32 }, ptr %o243, i32 0, i32 1
  %t947 = load i32, ptr %t946
  %t948 = srem i32 %t945, %t947
  store i32 %t948, ptr %o244
  %t949 = load i32, ptr %o244
  %t950 = getelementptr { i32, i32 }, ptr %o245, i32 0, i32 0
  store i32 %t949, ptr %t950
  %t951 = getelementptr { i32, i32 }, ptr %o245, i32 0, i32 0
  %t952 = load i32, ptr %t951
  %t953 = getelementptr { i32, i32 }, ptr %o245, i32 0, i32 1
  %t954 = load i32, ptr %t953
  %t955 = mul i32 %t952, %t954
  store i32 %t955, ptr %o246
  %t956 = load i32, ptr %o246
  %t957 = getelementptr { i32, i32 }, ptr %o247, i32 0, i32 0
  store i32 %t956, ptr %t957
  %t958 = getelementptr { i32, i32 }, ptr %o247, i32 0, i32 0
  %t959 = load i32, ptr %t958
  %t960 = getelementptr { i32, i32 }, ptr %o247, i32 0, i32 1
  %t961 = load i32, ptr %t960
  %t962 = add i32 %t959, %t961
  store i32 %t962, ptr %o248
  %t963 = load i32, ptr %o248
  %t964 = getelementptr { i32, i32 }, ptr %o249, i32 0, i32 0
  store i32 %t963, ptr %t964
  %t965 = getelementptr { i32, i32 }, ptr %o249, i32 0, i32 0
  %t966 = load i32, ptr %t965
  %t967 = getelementptr { i32, i32 }, ptr %o249, i32 0, i32 1
  %t968 = load i32, ptr %t967
  %t969 = srem i32 %t966, %t968
  store i32 %t969, ptr %o250
  %t970 = load i32, ptr %o250
  %t971 = getelementptr { i32, i32 }, ptr %o251, i32 0, i32 0
  store i32 %t970, ptr %t971
  %t972 = getelementptr { i32, i32 }, ptr %o251, i32 0, i32 0
  %t973 = load i32, ptr %t972
  %t974 = getelementptr { i32, i32 }, ptr %o251, i32 0, i32 1
  %t975 = load i32, ptr %t974
  %t976 = sub i32 %t973, %t975
  store i32 %t976, ptr %o252
  %t977 = load i32, ptr %o252
  %t978 = sitofp i32 %t977 to float
  store float %t978, ptr %o253
  %t979 = load float, ptr %o253
  %t980 = getelementptr { float, float }, ptr %o254, i32 0, i32 0
  store float %t979, ptr %t980
  %t981 = getelementptr { float, float }, ptr %o254, i32 0, i32 0
  %t982 = load float, ptr %t981
  %t983 = getelementptr { float, float }, ptr %o254, i32 0, i32 1
  %t984 = load float, ptr %t983
  %t985 = fdiv float %t982, %t984
  store float %t985, ptr %o255
  %t986 = load float, ptr %o255
  %t987 = getelementptr { float, float }, ptr %o258, i32 0, i32 0
  store float %t986, ptr %t987
  %t988 = getelementptr { ptr, i32 }, ptr %o256, i32 0, i32 1
  %t989 = load i32, ptr %t988
  %t990 = sext i32 %t989 to i64
  %t991 = getelementptr [8 x float], ptr %o46, i64 0, i64 %t990
  %t992 = load float, ptr %t991
  store float %t992, ptr %o257
  %t993 = load float, ptr %o257
  %t994 = getelementptr { float, float }, ptr %o258, i32 0, i32 1
  store float %t993, ptr %t994
  %t995 = getelementptr { float, float }, ptr %o258, i32 0, i32 0
  %t996 = load float, ptr %t995
  %t997 = getelementptr { float, float }, ptr %o258, i32 0, i32 1
  %t998 = load float, ptr %t997
  %t999 = fadd float %t996, %t998
  store float %t999, ptr %o259
  %t1000 = load float, ptr %o259
  %t1001 = getelementptr { ptr, i32, float }, ptr %o264, i32 0, i32 2
  store float %t1000, ptr %t1001
  %t1002 = getelementptr { i32, i32 }, ptr %o260, i32 0, i32 0
  %t1003 = load i32, ptr %t1002
  %t1004 = getelementptr { i32, i32 }, ptr %o260, i32 0, i32 1
  %t1005 = load i32, ptr %t1004
  %t1006 = mul i32 %t1003, %t1005
  store i32 %t1006, ptr %o261
  %t1007 = load i32, ptr %o261
  %t1008 = getelementptr { i32, i32 }, ptr %o262, i32 0, i32 0
  store i32 %t1007, ptr %t1008
  %t1009 = getelementptr { i32, i32 }, ptr %o262, i32 0, i32 0
  %t1010 = load i32, ptr %t1009
  %t1011 = getelementptr { i32, i32 }, ptr %o262, i32 0, i32 1
  %t1012 = load i32, ptr %t1011
  %t1013 = add i32 %t1010, %t1012
  store i32 %t1013, ptr %o263
  %t1014 = load i32, ptr %o263
  %t1015 = getelementptr { ptr, i32, float }, ptr %o264, i32 0, i32 1
  store i32 %t1014, ptr %t1015
  %t1016 = getelementptr { ptr, i32, float }, ptr %o264, i32 0, i32 1
  %t1017 = load i32, ptr %t1016
  %t1018 = sext i32 %t1017 to i64
  %t1019 = icmp slt i64 %t1018, 0
  %t1020 = icmp sge i64 %t1018, 128
  %t1021 = or i1 %t1019, %t1020
  br i1 %t1021, label %bb1022, label %bb1023
bb1022:
  call void @mapal_trap(i32 1)
  unreachable
bb1023:
  %t1024 = getelementptr [128 x float], ptr %o45, i64 0, i64 %t1018
  %t1025 = getelementptr { ptr, i32, float }, ptr %o264, i32 0, i32 2
  %t1026 = load float, ptr %t1025
  store float %t1026, ptr %t1024
  %t1028 = call i64 @mapal_gpu_alloc(ptr %d1027, i64 512)
  %t1029 = load i32, ptr %o47
  %t1030 = load i1, ptr %s60
  %t1031 = mul i32 %t1029, 8
  %t1032 = add i32 %t1031, 0
  %t1033 = sext i32 %t1032 to i64
  br i1 %t1030, label %bb1034, label %bb1035
bb1035:
  call void @mapal_gpu_h2d(i64 %t1028, ptr %o45, i64 512)
  store i1 true, ptr %s60
  br label %bb1036
bb1034:
  %t1037 = mul i64 %t1033, 4
  %t1038 = add i64 %t1028, %t1037
  %t1039 = getelementptr i8, ptr %o45, i64 %t1037
  call void @mapal_gpu_h2d(i64 %t1038, ptr %t1039, i64 32)
  br label %bb1036
bb1036:
  %t1040 = getelementptr { ptr, ptr, ptr, ptr }, ptr %o266, i32 0, i32 1
  store ptr %o45, ptr %t1040
  %t1041 = getelementptr { ptr, ptr, ptr, ptr }, ptr %o268, i32 0, i32 1
  store ptr %o45, ptr %t1041
  %t1042 = getelementptr { [128 x float], [8 x float], i32 }, ptr %o292, i32 0, i32 0
  call void @llvm.memcpy.p0.p0.i64(ptr %t1042, ptr %o45, i64 ptrtoint (ptr getelementptr ([128 x float], ptr null, i64 1) to i64), i1 false)
  %t1043 = load i64, ptr %d1027
  %t1044 = load i64, ptr %d6
  %t1046 = call i64 @mapal_gpu_alloc(ptr %d1045, i64 512)
  call void @mapal_gpu_launch(i64 -6055507678851317309, i64 %t1043, i64 %t1044, i64 %t1046, i32 1, i32 1, i32 256, i32 0)
  %t1047 = getelementptr { ptr, ptr, ptr, ptr }, ptr %o272, i32 0, i32 2
  store ptr null, ptr %t1047
  %t1048 = load i64, ptr %d1027
  %t1049 = load i64, ptr %d8
  %t1051 = call i64 @mapal_gpu_alloc(ptr %d1050, i64 512)
  call void @mapal_gpu_launch(i64 -6055507678851317309, i64 %t1048, i64 %t1049, i64 %t1051, i32 1, i32 1, i32 256, i32 0)
  %t1052 = getelementptr { ptr, ptr, ptr, ptr }, ptr %o276, i32 0, i32 2
  store ptr null, ptr %t1052
  store i64 0, ptr %s1053
  br label %bb1054
bb1054:
  %t1057 = load i64, ptr %s1053
  %t1058 = icmp uge i64 %t1057, 8
  br i1 %t1058, label %bb1056, label %bb1055
bb1055:
  %t1059 = trunc i64 %t1057 to i32
  %t1061 = getelementptr { i32, ptr }, ptr %o270, i32 0, i32 0
  %t1062 = load i32, ptr %t1061
  %t1063 = getelementptr { i32, i32 }, ptr %s1060, i32 0, i32 0
  store i32 %t1062, ptr %t1063
  %t1064 = getelementptr { i32, i32 }, ptr %s1060, i32 0, i32 1
  store i32 %t1059, ptr %t1064
  %t1065 = load { i32, i32 }, ptr %s1060
  %t1066 = call float @fn12({ i32, i32 } %t1065)
  %t1067 = getelementptr [8 x float], ptr %o271, i64 0, i64 %t1057
  store float %t1066, ptr %t1067
  %t1068 = add i64 %t1057, 1
  store i64 %t1068, ptr %s1053
  br label %bb1054
bb1056:
  %t1070 = call i64 @mapal_gpu_alloc(ptr %d1069, i64 32)
  call void @mapal_gpu_h2d(i64 %t1070, ptr %o271, i64 32)
  %t1071 = getelementptr { ptr, ptr, ptr, ptr }, ptr %o272, i32 0, i32 1
  store ptr %o271, ptr %t1071
  %t1072 = load i64, ptr %d1069
  %t1073 = load i64, ptr %d1045
  %t1075 = call i64 @mapal_gpu_alloc(ptr %d1074, i64 64)
  %t1077 = getelementptr [4 x i64], ptr %p1076, i64 0, i64 0
  store i64 %t1072, ptr %t1077
  %t1078 = getelementptr [4 x i64], ptr %p1076, i64 0, i64 1
  store i64 %t1073, ptr %t1078
  %t1079 = getelementptr [4 x i64], ptr %p1076, i64 0, i64 2
  store i64 0, ptr %t1079
  %t1080 = getelementptr [4 x i64], ptr %p1076, i64 0, i64 3
  store i64 %t1075, ptr %t1080
  call void @mapal_gpu_launchv(i64 -7490935162623979227, i32 0, i32 4, ptr %p1076, i32 1, i32 1, i32 256, i32 0)
  %t1081 = getelementptr { ptr, ptr }, ptr %o274, i32 0, i32 0
  store ptr null, ptr %t1081
  %t1082 = getelementptr { ptr, ptr, ptr, ptr }, ptr %o276, i32 0, i32 1
  store ptr null, ptr %t1082
  %t1083 = load i64, ptr %d1074
  %t1084 = load i64, ptr %d1050
  %t1086 = call i64 @mapal_gpu_alloc(ptr %d1085, i64 32)
  call void @mapal_gpu_launch(i64 -5971682293135266196, i64 %t1083, i64 %t1084, i64 %t1086, i32 1, i32 1, i32 256, i32 0)
  %t1087 = getelementptr { ptr, ptr, ptr, ptr }, ptr %o278, i32 0, i32 1
  store ptr null, ptr %t1087
  %t1088 = load i64, ptr %d1085
  %t1089 = load i64, ptr %d0
  %t1091 = call i64 @mapal_gpu_alloc(ptr %d1090, i64 32)
  call void @mapal_gpu_launch(i64 -5746171913144188873, i64 %t1088, i64 %t1089, i64 %t1091, i32 1, i32 1, i32 256, i32 0)
  %t1092 = getelementptr { ptr, ptr, ptr, ptr }, ptr %o280, i32 0, i32 1
  store ptr null, ptr %t1092
  %t1093 = getelementptr { ptr, ptr, ptr }, ptr %o288, i32 0, i32 0
  store ptr null, ptr %t1093
  %t1094 = load i64, ptr %d1090
  %t1095 = load i64, ptr %d2
  %t1096 = load i64, ptr %d10
  %t1098 = call i64 @mapal_gpu_alloc(ptr %d1097, i64 64)
  %t1100 = getelementptr [5 x i64], ptr %p1099, i64 0, i64 0
  store i64 %t1094, ptr %t1100
  %t1101 = getelementptr [5 x i64], ptr %p1099, i64 0, i64 1
  store i64 %t1095, ptr %t1101
  %t1102 = getelementptr [5 x i64], ptr %p1099, i64 0, i64 2
  store i64 0, ptr %t1102
  %t1103 = getelementptr [5 x i64], ptr %p1099, i64 0, i64 3
  store i64 %t1096, ptr %t1103
  %t1104 = getelementptr [5 x i64], ptr %p1099, i64 0, i64 4
  store i64 %t1098, ptr %t1104
  call void @mapal_gpu_launchv(i64 4644736598714039774, i32 0, i32 5, ptr %p1099, i32 1, i32 1, i32 256, i32 0)
  %t1105 = getelementptr { ptr, ptr, ptr }, ptr %o282, i32 0, i32 0
  store ptr null, ptr %t1105
  %t1106 = getelementptr { ptr, ptr, ptr, ptr }, ptr %o284, i32 0, i32 1
  store ptr null, ptr %t1106
  %t1107 = load i64, ptr %d1097
  %t1108 = load i64, ptr %d4
  %t1109 = load i64, ptr %d12
  %t1110 = load i64, ptr %d1090
  %t1112 = call i64 @mapal_gpu_alloc(ptr %d1111, i64 32)
  %t1114 = getelementptr [6 x i64], ptr %p1113, i64 0, i64 0
  store i64 %t1107, ptr %t1114
  %t1115 = getelementptr [6 x i64], ptr %p1113, i64 0, i64 1
  store i64 %t1108, ptr %t1115
  %t1116 = getelementptr [6 x i64], ptr %p1113, i64 0, i64 2
  store i64 0, ptr %t1116
  %t1117 = getelementptr [6 x i64], ptr %p1113, i64 0, i64 3
  store i64 %t1109, ptr %t1117
  %t1118 = getelementptr [6 x i64], ptr %p1113, i64 0, i64 4
  store i64 %t1110, ptr %t1118
  %t1119 = getelementptr [6 x i64], ptr %p1113, i64 0, i64 5
  store i64 %t1112, ptr %t1119
  call void @mapal_gpu_launchv(i64 -4366650180424198423, i32 0, i32 6, ptr %p1113, i32 1, i32 1, i32 256, i32 0)
  call void @mapal_gpu_d2h(ptr %o289, i64 %t1112, i64 0, i64 32)
  %t1120 = getelementptr { ptr, ptr, ptr }, ptr %o286, i32 0, i32 0
  store ptr null, ptr %t1120
  %t1121 = getelementptr { ptr, ptr, ptr }, ptr %o288, i32 0, i32 1
  store ptr null, ptr %t1121
  %t1122 = getelementptr { [128 x float], [8 x float], i32 }, ptr %o292, i32 0, i32 1
  call void @llvm.memcpy.p0.p0.i64(ptr %t1122, ptr %o289, i64 ptrtoint (ptr getelementptr ([8 x float], ptr null, i64 1) to i64), i1 false)
  %t1123 = getelementptr { i32, i32 }, ptr %o290, i32 0, i32 0
  %t1124 = load i32, ptr %t1123
  %t1125 = getelementptr { i32, i32 }, ptr %o290, i32 0, i32 1
  %t1126 = load i32, ptr %t1125
  %t1127 = add i32 %t1124, %t1126
  store i32 %t1127, ptr %o291
  %t1128 = load i32, ptr %o291
  %t1129 = getelementptr { [128 x float], [8 x float], i32 }, ptr %o292, i32 0, i32 2
  store i32 %t1128, ptr %t1129
  %t1130 = load { [128 x float], [8 x float], i32 }, ptr %o292
  %t1131 = getelementptr { { [128 x float], [8 x float], i32 }, i1 }, ptr %o293, i32 0, i32 0
  store { [128 x float], [8 x float], i32 } %t1130, ptr %t1131
  %t1132 = getelementptr { { [128 x float], [8 x float], i32 }, i1 }, ptr %o293, i32 0, i32 0
  %t1133 = load { [128 x float], [8 x float], i32 }, ptr %t1132
  store { [128 x float], [8 x float], i32 } %t1133, ptr %o44
  br label %bb61
bb63:
  call void @llvm.memcpy.p0.p0.i64(ptr %o295, ptr %o46, i64 ptrtoint (ptr getelementptr ([8 x float], ptr null, i64 1) to i64), i1 false)
  br label %bb64
bb64:
  %t1134 = getelementptr { ptr, i32 }, ptr %o296, i32 0, i32 0
  store ptr %o295, ptr %t1134
  %t1135 = getelementptr { ptr, i32 }, ptr %o300, i32 0, i32 0
  store ptr %o295, ptr %t1135
  %t1136 = getelementptr { ptr, i32 }, ptr %o304, i32 0, i32 0
  store ptr %o295, ptr %t1136
  %t1137 = getelementptr { ptr, i32 }, ptr %o308, i32 0, i32 0
  store ptr %o295, ptr %t1137
  %t1138 = getelementptr { ptr, i32 }, ptr %o312, i32 0, i32 0
  store ptr %o295, ptr %t1138
  %t1139 = getelementptr { ptr, i32 }, ptr %o316, i32 0, i32 0
  store ptr %o295, ptr %t1139
  %t1140 = getelementptr { ptr, i32 }, ptr %o320, i32 0, i32 0
  store ptr %o295, ptr %t1140
  %t1141 = getelementptr { ptr, i32 }, ptr %o324, i32 0, i32 0
  store ptr %o295, ptr %t1141
  %t1142 = getelementptr { ptr, i32 }, ptr %o296, i32 0, i32 1
  %t1143 = load i32, ptr %t1142
  %t1144 = sext i32 %t1143 to i64
  %t1145 = getelementptr [8 x float], ptr %o295, i64 0, i64 %t1144
  %t1146 = load float, ptr %t1145
  store float %t1146, ptr %o297
  %t1147 = load float, ptr %o297
  store float %t1147, ptr %o298
  %t1148 = load float, ptr %o298
  call void @mapal_print_f32(float %t1148, i1 zeroext true)
  %t1149 = getelementptr { ptr, i32 }, ptr %o300, i32 0, i32 1
  %t1150 = load i32, ptr %t1149
  %t1151 = sext i32 %t1150 to i64
  %t1152 = getelementptr [8 x float], ptr %o295, i64 0, i64 %t1151
  %t1153 = load float, ptr %t1152
  store float %t1153, ptr %o301
  %t1154 = load float, ptr %o301
  store float %t1154, ptr %o302
  %t1155 = load float, ptr %o302
  call void @mapal_print_f32(float %t1155, i1 zeroext true)
  %t1156 = getelementptr { ptr, i32 }, ptr %o304, i32 0, i32 1
  %t1157 = load i32, ptr %t1156
  %t1158 = sext i32 %t1157 to i64
  %t1159 = getelementptr [8 x float], ptr %o295, i64 0, i64 %t1158
  %t1160 = load float, ptr %t1159
  store float %t1160, ptr %o305
  %t1161 = load float, ptr %o305
  store float %t1161, ptr %o306
  %t1162 = load float, ptr %o306
  call void @mapal_print_f32(float %t1162, i1 zeroext true)
  %t1163 = getelementptr { ptr, i32 }, ptr %o308, i32 0, i32 1
  %t1164 = load i32, ptr %t1163
  %t1165 = sext i32 %t1164 to i64
  %t1166 = getelementptr [8 x float], ptr %o295, i64 0, i64 %t1165
  %t1167 = load float, ptr %t1166
  store float %t1167, ptr %o309
  %t1168 = load float, ptr %o309
  store float %t1168, ptr %o310
  %t1169 = load float, ptr %o310
  call void @mapal_print_f32(float %t1169, i1 zeroext true)
  %t1170 = getelementptr { ptr, i32 }, ptr %o312, i32 0, i32 1
  %t1171 = load i32, ptr %t1170
  %t1172 = sext i32 %t1171 to i64
  %t1173 = getelementptr [8 x float], ptr %o295, i64 0, i64 %t1172
  %t1174 = load float, ptr %t1173
  store float %t1174, ptr %o313
  %t1175 = load float, ptr %o313
  store float %t1175, ptr %o314
  %t1176 = load float, ptr %o314
  call void @mapal_print_f32(float %t1176, i1 zeroext true)
  %t1177 = getelementptr { ptr, i32 }, ptr %o316, i32 0, i32 1
  %t1178 = load i32, ptr %t1177
  %t1179 = sext i32 %t1178 to i64
  %t1180 = getelementptr [8 x float], ptr %o295, i64 0, i64 %t1179
  %t1181 = load float, ptr %t1180
  store float %t1181, ptr %o317
  %t1182 = load float, ptr %o317
  store float %t1182, ptr %o318
  %t1183 = load float, ptr %o318
  call void @mapal_print_f32(float %t1183, i1 zeroext true)
  %t1184 = getelementptr { ptr, i32 }, ptr %o320, i32 0, i32 1
  %t1185 = load i32, ptr %t1184
  %t1186 = sext i32 %t1185 to i64
  %t1187 = getelementptr [8 x float], ptr %o295, i64 0, i64 %t1186
  %t1188 = load float, ptr %t1187
  store float %t1188, ptr %o321
  %t1189 = load float, ptr %o321
  store float %t1189, ptr %o322
  %t1190 = load float, ptr %o322
  call void @mapal_print_f32(float %t1190, i1 zeroext true)
  %t1191 = getelementptr { ptr, i32 }, ptr %o324, i32 0, i32 1
  %t1192 = load i32, ptr %t1191
  %t1193 = sext i32 %t1192 to i64
  %t1194 = getelementptr [8 x float], ptr %o295, i64 0, i64 %t1193
  %t1195 = load float, ptr %t1194
  store float %t1195, ptr %o325
  %t1196 = load float, ptr %o325
  store float %t1196, ptr %o326
  %t1197 = load float, ptr %o326
  call void @mapal_print_f32(float %t1197, i1 zeroext true)
  call void @mapal_gpu_free(ptr %d0)
  call void @mapal_gpu_free(ptr %d2)
  call void @mapal_gpu_free(ptr %d4)
  call void @mapal_gpu_free(ptr %d6)
  call void @mapal_gpu_free(ptr %d8)
  call void @mapal_gpu_free(ptr %d10)
  call void @mapal_gpu_free(ptr %d12)
  call void @mapal_gpu_free(ptr %d1027)
  call void @mapal_gpu_free(ptr %d1045)
  call void @mapal_gpu_free(ptr %d1050)
  call void @mapal_gpu_free(ptr %d1069)
  call void @mapal_gpu_free(ptr %d1074)
  call void @mapal_gpu_free(ptr %d1085)
  call void @mapal_gpu_free(ptr %d1090)
  call void @mapal_gpu_free(ptr %d1097)
  call void @mapal_gpu_free(ptr %d1111)
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
  %o6 = alloca { i32, i32 }
  %o7 = alloca i32
  %o8 = alloca { i32, i32 }
  %o9 = alloca i32
  %o10 = alloca { i32, i32 }
  %o11 = alloca i32
  %o12 = alloca float
  %o13 = alloca { float, float }
  store i32 %arg, ptr %o0
  %t0 = load i32, ptr %o0
  %t1 = getelementptr { i32, i32 }, ptr %o2, i32 0, i32 0
  store i32 %t0, ptr %t1
  %t2 = getelementptr { i32, i32 }, ptr %o2, i32 0, i32 1
  store i32 17, ptr %t2
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
  store i32 7, ptr %t10
  %t11 = getelementptr { i32, i32 }, ptr %o4, i32 0, i32 0
  %t12 = load i32, ptr %t11
  %t13 = getelementptr { i32, i32 }, ptr %o4, i32 0, i32 1
  %t14 = load i32, ptr %t13
  %t15 = mul i32 %t12, %t14
  store i32 %t15, ptr %o5
  %t16 = load i32, ptr %o5
  %t17 = getelementptr { i32, i32 }, ptr %o6, i32 0, i32 0
  store i32 %t16, ptr %t17
  %t18 = getelementptr { i32, i32 }, ptr %o6, i32 0, i32 1
  store i32 5, ptr %t18
  %t19 = getelementptr { i32, i32 }, ptr %o6, i32 0, i32 0
  %t20 = load i32, ptr %t19
  %t21 = getelementptr { i32, i32 }, ptr %o6, i32 0, i32 1
  %t22 = load i32, ptr %t21
  %t23 = add i32 %t20, %t22
  store i32 %t23, ptr %o7
  %t24 = load i32, ptr %o7
  %t25 = getelementptr { i32, i32 }, ptr %o8, i32 0, i32 0
  store i32 %t24, ptr %t25
  %t26 = getelementptr { i32, i32 }, ptr %o8, i32 0, i32 1
  store i32 17, ptr %t26
  %t27 = getelementptr { i32, i32 }, ptr %o8, i32 0, i32 0
  %t28 = load i32, ptr %t27
  %t29 = getelementptr { i32, i32 }, ptr %o8, i32 0, i32 1
  %t30 = load i32, ptr %t29
  %t31 = srem i32 %t28, %t30
  store i32 %t31, ptr %o9
  %t32 = load i32, ptr %o9
  %t33 = getelementptr { i32, i32 }, ptr %o10, i32 0, i32 0
  store i32 %t32, ptr %t33
  %t34 = getelementptr { i32, i32 }, ptr %o10, i32 0, i32 1
  store i32 8, ptr %t34
  %t35 = getelementptr { i32, i32 }, ptr %o10, i32 0, i32 0
  %t36 = load i32, ptr %t35
  %t37 = getelementptr { i32, i32 }, ptr %o10, i32 0, i32 1
  %t38 = load i32, ptr %t37
  %t39 = sub i32 %t36, %t38
  store i32 %t39, ptr %o11
  %t40 = load i32, ptr %o11
  %t41 = sitofp i32 %t40 to float
  store float %t41, ptr %o12
  %t42 = load float, ptr %o12
  %t43 = getelementptr { float, float }, ptr %o13, i32 0, i32 0
  store float %t42, ptr %t43
  %t44 = getelementptr { float, float }, ptr %o13, i32 0, i32 1
  store float 0x4070000000000000, ptr %t44
  %t45 = getelementptr { float, float }, ptr %o13, i32 0, i32 0
  %t46 = load float, ptr %t45
  %t47 = getelementptr { float, float }, ptr %o13, i32 0, i32 1
  %t48 = load float, ptr %t47
  %t49 = fdiv float %t46, %t48
  store float %t49, ptr %o1
  %t50 = load float, ptr %o1
  ret float %t50
}

define internal float @fn2(i32 %arg) readonly nounwind willreturn {
entry:
  %o0 = alloca i32
  %o1 = alloca float
  %o2 = alloca { i32, i32 }
  %o3 = alloca i32
  %o4 = alloca { i32, i32 }
  %o5 = alloca i32
  %o6 = alloca { i32, i32 }
  %o7 = alloca i32
  %o8 = alloca { i32, i32 }
  %o9 = alloca i32
  %o10 = alloca { i32, i32 }
  %o11 = alloca i32
  %o12 = alloca float
  %o13 = alloca { float, float }
  store i32 %arg, ptr %o0
  %t0 = load i32, ptr %o0
  %t1 = getelementptr { i32, i32 }, ptr %o2, i32 0, i32 0
  store i32 %t0, ptr %t1
  %t2 = getelementptr { i32, i32 }, ptr %o2, i32 0, i32 1
  store i32 17, ptr %t2
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
  store i32 11, ptr %t10
  %t11 = getelementptr { i32, i32 }, ptr %o4, i32 0, i32 0
  %t12 = load i32, ptr %t11
  %t13 = getelementptr { i32, i32 }, ptr %o4, i32 0, i32 1
  %t14 = load i32, ptr %t13
  %t15 = mul i32 %t12, %t14
  store i32 %t15, ptr %o5
  %t16 = load i32, ptr %o5
  %t17 = getelementptr { i32, i32 }, ptr %o6, i32 0, i32 0
  store i32 %t16, ptr %t17
  %t18 = getelementptr { i32, i32 }, ptr %o6, i32 0, i32 1
  store i32 7, ptr %t18
  %t19 = getelementptr { i32, i32 }, ptr %o6, i32 0, i32 0
  %t20 = load i32, ptr %t19
  %t21 = getelementptr { i32, i32 }, ptr %o6, i32 0, i32 1
  %t22 = load i32, ptr %t21
  %t23 = add i32 %t20, %t22
  store i32 %t23, ptr %o7
  %t24 = load i32, ptr %o7
  %t25 = getelementptr { i32, i32 }, ptr %o8, i32 0, i32 0
  store i32 %t24, ptr %t25
  %t26 = getelementptr { i32, i32 }, ptr %o8, i32 0, i32 1
  store i32 17, ptr %t26
  %t27 = getelementptr { i32, i32 }, ptr %o8, i32 0, i32 0
  %t28 = load i32, ptr %t27
  %t29 = getelementptr { i32, i32 }, ptr %o8, i32 0, i32 1
  %t30 = load i32, ptr %t29
  %t31 = srem i32 %t28, %t30
  store i32 %t31, ptr %o9
  %t32 = load i32, ptr %o9
  %t33 = getelementptr { i32, i32 }, ptr %o10, i32 0, i32 0
  store i32 %t32, ptr %t33
  %t34 = getelementptr { i32, i32 }, ptr %o10, i32 0, i32 1
  store i32 8, ptr %t34
  %t35 = getelementptr { i32, i32 }, ptr %o10, i32 0, i32 0
  %t36 = load i32, ptr %t35
  %t37 = getelementptr { i32, i32 }, ptr %o10, i32 0, i32 1
  %t38 = load i32, ptr %t37
  %t39 = sub i32 %t36, %t38
  store i32 %t39, ptr %o11
  %t40 = load i32, ptr %o11
  %t41 = sitofp i32 %t40 to float
  store float %t41, ptr %o12
  %t42 = load float, ptr %o12
  %t43 = getelementptr { float, float }, ptr %o13, i32 0, i32 0
  store float %t42, ptr %t43
  %t44 = getelementptr { float, float }, ptr %o13, i32 0, i32 1
  store float 0x4070000000000000, ptr %t44
  %t45 = getelementptr { float, float }, ptr %o13, i32 0, i32 0
  %t46 = load float, ptr %t45
  %t47 = getelementptr { float, float }, ptr %o13, i32 0, i32 1
  %t48 = load float, ptr %t47
  %t49 = fdiv float %t46, %t48
  store float %t49, ptr %o1
  %t50 = load float, ptr %o1
  ret float %t50
}

define internal float @fn3(i32 %arg) readonly nounwind willreturn {
entry:
  %o0 = alloca i32
  %o1 = alloca float
  %o2 = alloca { i32, i32 }
  %o3 = alloca i32
  %o4 = alloca { i32, i32 }
  %o5 = alloca i32
  %o6 = alloca { i32, i32 }
  %o7 = alloca i32
  %o8 = alloca { i32, i32 }
  %o9 = alloca i32
  %o10 = alloca { i32, i32 }
  %o11 = alloca i32
  %o12 = alloca float
  %o13 = alloca { float, float }
  store i32 %arg, ptr %o0
  %t0 = load i32, ptr %o0
  %t1 = getelementptr { i32, i32 }, ptr %o2, i32 0, i32 0
  store i32 %t0, ptr %t1
  %t2 = getelementptr { i32, i32 }, ptr %o2, i32 0, i32 1
  store i32 17, ptr %t2
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
  store i32 13, ptr %t10
  %t11 = getelementptr { i32, i32 }, ptr %o4, i32 0, i32 0
  %t12 = load i32, ptr %t11
  %t13 = getelementptr { i32, i32 }, ptr %o4, i32 0, i32 1
  %t14 = load i32, ptr %t13
  %t15 = mul i32 %t12, %t14
  store i32 %t15, ptr %o5
  %t16 = load i32, ptr %o5
  %t17 = getelementptr { i32, i32 }, ptr %o6, i32 0, i32 0
  store i32 %t16, ptr %t17
  %t18 = getelementptr { i32, i32 }, ptr %o6, i32 0, i32 1
  store i32 9, ptr %t18
  %t19 = getelementptr { i32, i32 }, ptr %o6, i32 0, i32 0
  %t20 = load i32, ptr %t19
  %t21 = getelementptr { i32, i32 }, ptr %o6, i32 0, i32 1
  %t22 = load i32, ptr %t21
  %t23 = add i32 %t20, %t22
  store i32 %t23, ptr %o7
  %t24 = load i32, ptr %o7
  %t25 = getelementptr { i32, i32 }, ptr %o8, i32 0, i32 0
  store i32 %t24, ptr %t25
  %t26 = getelementptr { i32, i32 }, ptr %o8, i32 0, i32 1
  store i32 17, ptr %t26
  %t27 = getelementptr { i32, i32 }, ptr %o8, i32 0, i32 0
  %t28 = load i32, ptr %t27
  %t29 = getelementptr { i32, i32 }, ptr %o8, i32 0, i32 1
  %t30 = load i32, ptr %t29
  %t31 = srem i32 %t28, %t30
  store i32 %t31, ptr %o9
  %t32 = load i32, ptr %o9
  %t33 = getelementptr { i32, i32 }, ptr %o10, i32 0, i32 0
  store i32 %t32, ptr %t33
  %t34 = getelementptr { i32, i32 }, ptr %o10, i32 0, i32 1
  store i32 8, ptr %t34
  %t35 = getelementptr { i32, i32 }, ptr %o10, i32 0, i32 0
  %t36 = load i32, ptr %t35
  %t37 = getelementptr { i32, i32 }, ptr %o10, i32 0, i32 1
  %t38 = load i32, ptr %t37
  %t39 = sub i32 %t36, %t38
  store i32 %t39, ptr %o11
  %t40 = load i32, ptr %o11
  %t41 = sitofp i32 %t40 to float
  store float %t41, ptr %o12
  %t42 = load float, ptr %o12
  %t43 = getelementptr { float, float }, ptr %o13, i32 0, i32 0
  store float %t42, ptr %t43
  %t44 = getelementptr { float, float }, ptr %o13, i32 0, i32 1
  store float 0x4080000000000000, ptr %t44
  %t45 = getelementptr { float, float }, ptr %o13, i32 0, i32 0
  %t46 = load float, ptr %t45
  %t47 = getelementptr { float, float }, ptr %o13, i32 0, i32 1
  %t48 = load float, ptr %t47
  %t49 = fdiv float %t46, %t48
  store float %t49, ptr %o1
  %t50 = load float, ptr %o1
  ret float %t50
}

define internal float @fn4(i32 %arg) readonly nounwind willreturn {
entry:
  %o0 = alloca i32
  %o1 = alloca float
  %o2 = alloca { i32, i32 }
  %o3 = alloca i32
  %o4 = alloca { i32, i32 }
  %o5 = alloca i32
  %o6 = alloca { i32, i32 }
  %o7 = alloca i32
  %o8 = alloca { i32, i32 }
  %o9 = alloca i32
  %o10 = alloca { i32, i32 }
  %o11 = alloca i32
  %o12 = alloca float
  %o13 = alloca { float, float }
  store i32 %arg, ptr %o0
  %t0 = load i32, ptr %o0
  %t1 = getelementptr { i32, i32 }, ptr %o2, i32 0, i32 0
  store i32 %t0, ptr %t1
  %t2 = getelementptr { i32, i32 }, ptr %o2, i32 0, i32 1
  store i32 17, ptr %t2
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
  store i32 3, ptr %t10
  %t11 = getelementptr { i32, i32 }, ptr %o4, i32 0, i32 0
  %t12 = load i32, ptr %t11
  %t13 = getelementptr { i32, i32 }, ptr %o4, i32 0, i32 1
  %t14 = load i32, ptr %t13
  %t15 = mul i32 %t12, %t14
  store i32 %t15, ptr %o5
  %t16 = load i32, ptr %o5
  %t17 = getelementptr { i32, i32 }, ptr %o6, i32 0, i32 0
  store i32 %t16, ptr %t17
  %t18 = getelementptr { i32, i32 }, ptr %o6, i32 0, i32 1
  store i32 1, ptr %t18
  %t19 = getelementptr { i32, i32 }, ptr %o6, i32 0, i32 0
  %t20 = load i32, ptr %t19
  %t21 = getelementptr { i32, i32 }, ptr %o6, i32 0, i32 1
  %t22 = load i32, ptr %t21
  %t23 = add i32 %t20, %t22
  store i32 %t23, ptr %o7
  %t24 = load i32, ptr %o7
  %t25 = getelementptr { i32, i32 }, ptr %o8, i32 0, i32 0
  store i32 %t24, ptr %t25
  %t26 = getelementptr { i32, i32 }, ptr %o8, i32 0, i32 1
  store i32 17, ptr %t26
  %t27 = getelementptr { i32, i32 }, ptr %o8, i32 0, i32 0
  %t28 = load i32, ptr %t27
  %t29 = getelementptr { i32, i32 }, ptr %o8, i32 0, i32 1
  %t30 = load i32, ptr %t29
  %t31 = srem i32 %t28, %t30
  store i32 %t31, ptr %o9
  %t32 = load i32, ptr %o9
  %t33 = getelementptr { i32, i32 }, ptr %o10, i32 0, i32 0
  store i32 %t32, ptr %t33
  %t34 = getelementptr { i32, i32 }, ptr %o10, i32 0, i32 1
  store i32 8, ptr %t34
  %t35 = getelementptr { i32, i32 }, ptr %o10, i32 0, i32 0
  %t36 = load i32, ptr %t35
  %t37 = getelementptr { i32, i32 }, ptr %o10, i32 0, i32 1
  %t38 = load i32, ptr %t37
  %t39 = sub i32 %t36, %t38
  store i32 %t39, ptr %o11
  %t40 = load i32, ptr %o11
  %t41 = sitofp i32 %t40 to float
  store float %t41, ptr %o12
  %t42 = load float, ptr %o12
  %t43 = getelementptr { float, float }, ptr %o13, i32 0, i32 0
  store float %t42, ptr %t43
  %t44 = getelementptr { float, float }, ptr %o13, i32 0, i32 1
  store float 0x4070000000000000, ptr %t44
  %t45 = getelementptr { float, float }, ptr %o13, i32 0, i32 0
  %t46 = load float, ptr %t45
  %t47 = getelementptr { float, float }, ptr %o13, i32 0, i32 1
  %t48 = load float, ptr %t47
  %t49 = fdiv float %t46, %t48
  store float %t49, ptr %o1
  %t50 = load float, ptr %o1
  ret float %t50
}

define internal float @fn5(i32 %arg) readonly nounwind willreturn {
entry:
  %o0 = alloca i32
  %o1 = alloca float
  %o2 = alloca { i32, i32 }
  %o3 = alloca i32
  %o4 = alloca { i32, i32 }
  %o5 = alloca i32
  %o6 = alloca { i32, i32 }
  %o7 = alloca i32
  %o8 = alloca { i32, i32 }
  %o9 = alloca i32
  %o10 = alloca { i32, i32 }
  %o11 = alloca i32
  %o12 = alloca float
  %o13 = alloca { float, float }
  store i32 %arg, ptr %o0
  %t0 = load i32, ptr %o0
  %t1 = getelementptr { i32, i32 }, ptr %o2, i32 0, i32 0
  store i32 %t0, ptr %t1
  %t2 = getelementptr { i32, i32 }, ptr %o2, i32 0, i32 1
  store i32 17, ptr %t2
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
  store i32 5, ptr %t10
  %t11 = getelementptr { i32, i32 }, ptr %o4, i32 0, i32 0
  %t12 = load i32, ptr %t11
  %t13 = getelementptr { i32, i32 }, ptr %o4, i32 0, i32 1
  %t14 = load i32, ptr %t13
  %t15 = mul i32 %t12, %t14
  store i32 %t15, ptr %o5
  %t16 = load i32, ptr %o5
  %t17 = getelementptr { i32, i32 }, ptr %o6, i32 0, i32 0
  store i32 %t16, ptr %t17
  %t18 = getelementptr { i32, i32 }, ptr %o6, i32 0, i32 1
  store i32 3, ptr %t18
  %t19 = getelementptr { i32, i32 }, ptr %o6, i32 0, i32 0
  %t20 = load i32, ptr %t19
  %t21 = getelementptr { i32, i32 }, ptr %o6, i32 0, i32 1
  %t22 = load i32, ptr %t21
  %t23 = add i32 %t20, %t22
  store i32 %t23, ptr %o7
  %t24 = load i32, ptr %o7
  %t25 = getelementptr { i32, i32 }, ptr %o8, i32 0, i32 0
  store i32 %t24, ptr %t25
  %t26 = getelementptr { i32, i32 }, ptr %o8, i32 0, i32 1
  store i32 17, ptr %t26
  %t27 = getelementptr { i32, i32 }, ptr %o8, i32 0, i32 0
  %t28 = load i32, ptr %t27
  %t29 = getelementptr { i32, i32 }, ptr %o8, i32 0, i32 1
  %t30 = load i32, ptr %t29
  %t31 = srem i32 %t28, %t30
  store i32 %t31, ptr %o9
  %t32 = load i32, ptr %o9
  %t33 = getelementptr { i32, i32 }, ptr %o10, i32 0, i32 0
  store i32 %t32, ptr %t33
  %t34 = getelementptr { i32, i32 }, ptr %o10, i32 0, i32 1
  store i32 8, ptr %t34
  %t35 = getelementptr { i32, i32 }, ptr %o10, i32 0, i32 0
  %t36 = load i32, ptr %t35
  %t37 = getelementptr { i32, i32 }, ptr %o10, i32 0, i32 1
  %t38 = load i32, ptr %t37
  %t39 = sub i32 %t36, %t38
  store i32 %t39, ptr %o11
  %t40 = load i32, ptr %o11
  %t41 = sitofp i32 %t40 to float
  store float %t41, ptr %o12
  %t42 = load float, ptr %o12
  %t43 = getelementptr { float, float }, ptr %o13, i32 0, i32 0
  store float %t42, ptr %t43
  %t44 = getelementptr { float, float }, ptr %o13, i32 0, i32 1
  store float 0x4070000000000000, ptr %t44
  %t45 = getelementptr { float, float }, ptr %o13, i32 0, i32 0
  %t46 = load float, ptr %t45
  %t47 = getelementptr { float, float }, ptr %o13, i32 0, i32 1
  %t48 = load float, ptr %t47
  %t49 = fdiv float %t46, %t48
  store float %t49, ptr %o1
  %t50 = load float, ptr %o1
  ret float %t50
}

define internal float @fn6(i32 %arg) readonly nounwind willreturn {
entry:
  %o0 = alloca i32
  %o1 = alloca float
  %o2 = alloca { i32, i32 }
  %o3 = alloca i32
  %o4 = alloca { i32, i32 }
  %o5 = alloca i32
  %o6 = alloca { i32, i32 }
  %o7 = alloca i32
  %o8 = alloca { i32, i32 }
  %o9 = alloca i32
  %o10 = alloca { i32, i32 }
  %o11 = alloca i32
  %o12 = alloca float
  %o13 = alloca { float, float }
  store i32 %arg, ptr %o0
  %t0 = load i32, ptr %o0
  %t1 = getelementptr { i32, i32 }, ptr %o2, i32 0, i32 0
  store i32 %t0, ptr %t1
  %t2 = getelementptr { i32, i32 }, ptr %o2, i32 0, i32 1
  store i32 17, ptr %t2
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
  store i32 3, ptr %t10
  %t11 = getelementptr { i32, i32 }, ptr %o4, i32 0, i32 0
  %t12 = load i32, ptr %t11
  %t13 = getelementptr { i32, i32 }, ptr %o4, i32 0, i32 1
  %t14 = load i32, ptr %t13
  %t15 = mul i32 %t12, %t14
  store i32 %t15, ptr %o5
  %t16 = load i32, ptr %o5
  %t17 = getelementptr { i32, i32 }, ptr %o6, i32 0, i32 0
  store i32 %t16, ptr %t17
  %t18 = getelementptr { i32, i32 }, ptr %o6, i32 0, i32 1
  store i32 1, ptr %t18
  %t19 = getelementptr { i32, i32 }, ptr %o6, i32 0, i32 0
  %t20 = load i32, ptr %t19
  %t21 = getelementptr { i32, i32 }, ptr %o6, i32 0, i32 1
  %t22 = load i32, ptr %t21
  %t23 = add i32 %t20, %t22
  store i32 %t23, ptr %o7
  %t24 = load i32, ptr %o7
  %t25 = getelementptr { i32, i32 }, ptr %o8, i32 0, i32 0
  store i32 %t24, ptr %t25
  %t26 = getelementptr { i32, i32 }, ptr %o8, i32 0, i32 1
  store i32 17, ptr %t26
  %t27 = getelementptr { i32, i32 }, ptr %o8, i32 0, i32 0
  %t28 = load i32, ptr %t27
  %t29 = getelementptr { i32, i32 }, ptr %o8, i32 0, i32 1
  %t30 = load i32, ptr %t29
  %t31 = srem i32 %t28, %t30
  store i32 %t31, ptr %o9
  %t32 = load i32, ptr %o9
  %t33 = getelementptr { i32, i32 }, ptr %o10, i32 0, i32 0
  store i32 %t32, ptr %t33
  %t34 = getelementptr { i32, i32 }, ptr %o10, i32 0, i32 1
  store i32 8, ptr %t34
  %t35 = getelementptr { i32, i32 }, ptr %o10, i32 0, i32 0
  %t36 = load i32, ptr %t35
  %t37 = getelementptr { i32, i32 }, ptr %o10, i32 0, i32 1
  %t38 = load i32, ptr %t37
  %t39 = sub i32 %t36, %t38
  store i32 %t39, ptr %o11
  %t40 = load i32, ptr %o11
  %t41 = sitofp i32 %t40 to float
  store float %t41, ptr %o12
  %t42 = load float, ptr %o12
  %t43 = getelementptr { float, float }, ptr %o13, i32 0, i32 0
  store float %t42, ptr %t43
  %t44 = getelementptr { float, float }, ptr %o13, i32 0, i32 1
  store float 0x4060000000000000, ptr %t44
  %t45 = getelementptr { float, float }, ptr %o13, i32 0, i32 0
  %t46 = load float, ptr %t45
  %t47 = getelementptr { float, float }, ptr %o13, i32 0, i32 1
  %t48 = load float, ptr %t47
  %t49 = fdiv float %t46, %t48
  store float %t49, ptr %o1
  %t50 = load float, ptr %o1
  ret float %t50
}

define internal float @fn7(i32 %arg) readonly nounwind willreturn {
entry:
  %o0 = alloca i32
  %o1 = alloca float
  %o2 = alloca { i32, i32 }
  %o3 = alloca i32
  %o4 = alloca { i32, i32 }
  %o5 = alloca i32
  %o6 = alloca { i32, i32 }
  %o7 = alloca i32
  %o8 = alloca { i32, i32 }
  %o9 = alloca i32
  %o10 = alloca { i32, i32 }
  %o11 = alloca i32
  %o12 = alloca float
  %o13 = alloca { float, float }
  store i32 %arg, ptr %o0
  %t0 = load i32, ptr %o0
  %t1 = getelementptr { i32, i32 }, ptr %o2, i32 0, i32 0
  store i32 %t0, ptr %t1
  %t2 = getelementptr { i32, i32 }, ptr %o2, i32 0, i32 1
  store i32 17, ptr %t2
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
  store i32 7, ptr %t10
  %t11 = getelementptr { i32, i32 }, ptr %o4, i32 0, i32 0
  %t12 = load i32, ptr %t11
  %t13 = getelementptr { i32, i32 }, ptr %o4, i32 0, i32 1
  %t14 = load i32, ptr %t13
  %t15 = mul i32 %t12, %t14
  store i32 %t15, ptr %o5
  %t16 = load i32, ptr %o5
  %t17 = getelementptr { i32, i32 }, ptr %o6, i32 0, i32 0
  store i32 %t16, ptr %t17
  %t18 = getelementptr { i32, i32 }, ptr %o6, i32 0, i32 1
  store i32 2, ptr %t18
  %t19 = getelementptr { i32, i32 }, ptr %o6, i32 0, i32 0
  %t20 = load i32, ptr %t19
  %t21 = getelementptr { i32, i32 }, ptr %o6, i32 0, i32 1
  %t22 = load i32, ptr %t21
  %t23 = add i32 %t20, %t22
  store i32 %t23, ptr %o7
  %t24 = load i32, ptr %o7
  %t25 = getelementptr { i32, i32 }, ptr %o8, i32 0, i32 0
  store i32 %t24, ptr %t25
  %t26 = getelementptr { i32, i32 }, ptr %o8, i32 0, i32 1
  store i32 17, ptr %t26
  %t27 = getelementptr { i32, i32 }, ptr %o8, i32 0, i32 0
  %t28 = load i32, ptr %t27
  %t29 = getelementptr { i32, i32 }, ptr %o8, i32 0, i32 1
  %t30 = load i32, ptr %t29
  %t31 = srem i32 %t28, %t30
  store i32 %t31, ptr %o9
  %t32 = load i32, ptr %o9
  %t33 = getelementptr { i32, i32 }, ptr %o10, i32 0, i32 0
  store i32 %t32, ptr %t33
  %t34 = getelementptr { i32, i32 }, ptr %o10, i32 0, i32 1
  store i32 8, ptr %t34
  %t35 = getelementptr { i32, i32 }, ptr %o10, i32 0, i32 0
  %t36 = load i32, ptr %t35
  %t37 = getelementptr { i32, i32 }, ptr %o10, i32 0, i32 1
  %t38 = load i32, ptr %t37
  %t39 = sub i32 %t36, %t38
  store i32 %t39, ptr %o11
  %t40 = load i32, ptr %o11
  %t41 = sitofp i32 %t40 to float
  store float %t41, ptr %o12
  %t42 = load float, ptr %o12
  %t43 = getelementptr { float, float }, ptr %o13, i32 0, i32 0
  store float %t42, ptr %t43
  %t44 = getelementptr { float, float }, ptr %o13, i32 0, i32 1
  store float 0x4070000000000000, ptr %t44
  %t45 = getelementptr { float, float }, ptr %o13, i32 0, i32 0
  %t46 = load float, ptr %t45
  %t47 = getelementptr { float, float }, ptr %o13, i32 0, i32 1
  %t48 = load float, ptr %t47
  %t49 = fdiv float %t46, %t48
  store float %t49, ptr %o1
  %t50 = load float, ptr %o1
  ret float %t50
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
  store i32 8, ptr %t26
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
  %t45 = getelementptr [128 x float], ptr %t41, i64 0, i64 %t44
  %t46 = load float, ptr %t45
  store float %t46, ptr %o13
  %t47 = load float, ptr %o13
  %t48 = getelementptr { float, float }, ptr %o20, i32 0, i32 0
  store float %t47, ptr %t48
  %t49 = getelementptr { i32, i32 }, ptr %o14, i32 0, i32 1
  store i32 8, ptr %t49
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
  %t68 = getelementptr [64 x float], ptr %t64, i64 0, i64 %t67
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
  store i32 8, ptr %t18
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
  store i32 8, ptr %t26
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
  %t44 = icmp uge i64 %t43, 8
  br i1 %t44, label %bb42, label %bb41
bb41:
  %t45 = getelementptr [8 x i32], ptr %t37, i64 0, i64 %t43
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

define internal float @fn10({ ptr, i32, ptr, i32, float, i32 } %arg) readonly nounwind willreturn {
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
  store i32 8, ptr %t26
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
  %t45 = getelementptr [128 x float], ptr %t41, i64 0, i64 %t44
  %t46 = load float, ptr %t45
  store float %t46, ptr %o13
  %t47 = load float, ptr %o13
  %t48 = getelementptr { float, float }, ptr %o20, i32 0, i32 0
  store float %t47, ptr %t48
  %t49 = getelementptr { i32, i32 }, ptr %o14, i32 0, i32 1
  store i32 8, ptr %t49
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
  %t68 = getelementptr [64 x float], ptr %t64, i64 0, i64 %t67
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

define internal float @fn11({ ptr, ptr, ptr, i32 } %arg) readonly nounwind willreturn {
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
  store i32 8, ptr %t18
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
  store i32 8, ptr %t26
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
  %t44 = icmp uge i64 %t43, 8
  br i1 %t44, label %bb42, label %bb41
bb41:
  %t45 = getelementptr [8 x i32], ptr %t37, i64 0, i64 %t43
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
  %t62 = call float @fn10({ ptr, i32, ptr, i32, float, i32 } %t61)
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

define internal float @fn12({ i32, i32 } %arg) readonly nounwind willreturn {
entry:
  %o0 = alloca { i32, i32 }
  %o1 = alloca float
  %o2 = alloca i32
  %o3 = alloca i32
  %o4 = alloca { i32, i32 }
  %o5 = alloca i32
  %o6 = alloca { i32, i32 }
  %o7 = alloca i32
  %o8 = alloca { i32, i32 }
  %o9 = alloca i32
  %o10 = alloca { i32, i32 }
  %o11 = alloca i32
  %o12 = alloca { i32, i32 }
  %o13 = alloca i32
  %o14 = alloca { i32, i32 }
  %o15 = alloca i32
  %o16 = alloca { i32, i32 }
  %o17 = alloca i32
  %o18 = alloca float
  %o19 = alloca { float, float }
  store { i32, i32 } %arg, ptr %o0
  %t0 = getelementptr { i32, i32 }, ptr %o0, i32 0, i32 0
  %t1 = load i32, ptr %t0
  store i32 %t1, ptr %o2
  %t2 = getelementptr { i32, i32 }, ptr %o0, i32 0, i32 1
  %t3 = load i32, ptr %t2
  store i32 %t3, ptr %o3
  %t4 = load i32, ptr %o2
  %t5 = getelementptr { i32, i32 }, ptr %o4, i32 0, i32 0
  store i32 %t4, ptr %t5
  %t6 = load i32, ptr %o3
  %t7 = getelementptr { i32, i32 }, ptr %o6, i32 0, i32 1
  store i32 %t6, ptr %t7
  %t8 = getelementptr { i32, i32 }, ptr %o4, i32 0, i32 1
  store i32 8, ptr %t8
  %t9 = getelementptr { i32, i32 }, ptr %o4, i32 0, i32 0
  %t10 = load i32, ptr %t9
  %t11 = getelementptr { i32, i32 }, ptr %o4, i32 0, i32 1
  %t12 = load i32, ptr %t11
  %t13 = mul i32 %t10, %t12
  store i32 %t13, ptr %o5
  %t14 = load i32, ptr %o5
  %t15 = getelementptr { i32, i32 }, ptr %o6, i32 0, i32 0
  store i32 %t14, ptr %t15
  %t16 = getelementptr { i32, i32 }, ptr %o6, i32 0, i32 0
  %t17 = load i32, ptr %t16
  %t18 = getelementptr { i32, i32 }, ptr %o6, i32 0, i32 1
  %t19 = load i32, ptr %t18
  %t20 = add i32 %t17, %t19
  store i32 %t20, ptr %o7
  %t21 = load i32, ptr %o7
  %t22 = getelementptr { i32, i32 }, ptr %o8, i32 0, i32 0
  store i32 %t21, ptr %t22
  %t23 = getelementptr { i32, i32 }, ptr %o8, i32 0, i32 1
  store i32 17, ptr %t23
  %t24 = getelementptr { i32, i32 }, ptr %o8, i32 0, i32 0
  %t25 = load i32, ptr %t24
  %t26 = getelementptr { i32, i32 }, ptr %o8, i32 0, i32 1
  %t27 = load i32, ptr %t26
  %t28 = srem i32 %t25, %t27
  store i32 %t28, ptr %o9
  %t29 = load i32, ptr %o9
  %t30 = getelementptr { i32, i32 }, ptr %o10, i32 0, i32 0
  store i32 %t29, ptr %t30
  %t31 = getelementptr { i32, i32 }, ptr %o10, i32 0, i32 1
  store i32 5, ptr %t31
  %t32 = getelementptr { i32, i32 }, ptr %o10, i32 0, i32 0
  %t33 = load i32, ptr %t32
  %t34 = getelementptr { i32, i32 }, ptr %o10, i32 0, i32 1
  %t35 = load i32, ptr %t34
  %t36 = mul i32 %t33, %t35
  store i32 %t36, ptr %o11
  %t37 = load i32, ptr %o11
  %t38 = getelementptr { i32, i32 }, ptr %o12, i32 0, i32 0
  store i32 %t37, ptr %t38
  %t39 = getelementptr { i32, i32 }, ptr %o12, i32 0, i32 1
  store i32 3, ptr %t39
  %t40 = getelementptr { i32, i32 }, ptr %o12, i32 0, i32 0
  %t41 = load i32, ptr %t40
  %t42 = getelementptr { i32, i32 }, ptr %o12, i32 0, i32 1
  %t43 = load i32, ptr %t42
  %t44 = add i32 %t41, %t43
  store i32 %t44, ptr %o13
  %t45 = load i32, ptr %o13
  %t46 = getelementptr { i32, i32 }, ptr %o14, i32 0, i32 0
  store i32 %t45, ptr %t46
  %t47 = getelementptr { i32, i32 }, ptr %o14, i32 0, i32 1
  store i32 17, ptr %t47
  %t48 = getelementptr { i32, i32 }, ptr %o14, i32 0, i32 0
  %t49 = load i32, ptr %t48
  %t50 = getelementptr { i32, i32 }, ptr %o14, i32 0, i32 1
  %t51 = load i32, ptr %t50
  %t52 = srem i32 %t49, %t51
  store i32 %t52, ptr %o15
  %t53 = load i32, ptr %o15
  %t54 = getelementptr { i32, i32 }, ptr %o16, i32 0, i32 0
  store i32 %t53, ptr %t54
  %t55 = getelementptr { i32, i32 }, ptr %o16, i32 0, i32 1
  store i32 8, ptr %t55
  %t56 = getelementptr { i32, i32 }, ptr %o16, i32 0, i32 0
  %t57 = load i32, ptr %t56
  %t58 = getelementptr { i32, i32 }, ptr %o16, i32 0, i32 1
  %t59 = load i32, ptr %t58
  %t60 = sub i32 %t57, %t59
  store i32 %t60, ptr %o17
  %t61 = load i32, ptr %o17
  %t62 = sitofp i32 %t61 to float
  store float %t62, ptr %o18
  %t63 = load float, ptr %o18
  %t64 = getelementptr { float, float }, ptr %o19, i32 0, i32 0
  store float %t63, ptr %t64
  %t65 = getelementptr { float, float }, ptr %o19, i32 0, i32 1
  store float 0x4070000000000000, ptr %t65
  %t66 = getelementptr { float, float }, ptr %o19, i32 0, i32 0
  %t67 = load float, ptr %t66
  %t68 = getelementptr { float, float }, ptr %o19, i32 0, i32 1
  %t69 = load float, ptr %t68
  %t70 = fdiv float %t67, %t69
  store float %t70, ptr %o1
  %t71 = load float, ptr %o1
  ret float %t71
}

define internal float @fn13({ ptr, ptr, i32, float, i32 } %arg) readonly nounwind willreturn {
entry:
  %o0 = alloca { ptr, ptr, i32, float, i32 }
  %o1 = alloca float
  %o2 = alloca ptr
  %o3 = alloca ptr
  %o4 = alloca i32
  %o5 = alloca float
  %o6 = alloca i32
  %o7 = alloca { ptr, i32 }
  %o8 = alloca float
  %o9 = alloca { i32, i32 }
  %o10 = alloca i32
  %o11 = alloca { i32, i32 }
  %o12 = alloca i32
  %o13 = alloca { ptr, i32 }
  %o14 = alloca float
  %o15 = alloca { float, float }
  %o16 = alloca float
  %o17 = alloca { float, float }
  store { ptr, ptr, i32, float, i32 } %arg, ptr %o0
  %t0 = getelementptr { ptr, ptr, i32, float, i32 }, ptr %o0, i32 0, i32 0
  %t1 = load ptr, ptr %t0
  store ptr %t1, ptr %o2
  %t2 = getelementptr { ptr, ptr, i32, float, i32 }, ptr %o0, i32 0, i32 1
  %t3 = load ptr, ptr %t2
  store ptr %t3, ptr %o3
  %t4 = getelementptr { ptr, ptr, i32, float, i32 }, ptr %o0, i32 0, i32 2
  %t5 = load i32, ptr %t4
  store i32 %t5, ptr %o4
  %t6 = getelementptr { ptr, ptr, i32, float, i32 }, ptr %o0, i32 0, i32 3
  %t7 = load float, ptr %t6
  store float %t7, ptr %o5
  %t8 = getelementptr { ptr, ptr, i32, float, i32 }, ptr %o0, i32 0, i32 4
  %t9 = load i32, ptr %t8
  store i32 %t9, ptr %o6
  %t10 = load ptr, ptr %o2
  %t11 = getelementptr { ptr, i32 }, ptr %o7, i32 0, i32 0
  store ptr %t10, ptr %t11
  %t12 = load ptr, ptr %o3
  %t13 = getelementptr { ptr, i32 }, ptr %o13, i32 0, i32 0
  store ptr %t12, ptr %t13
  %t14 = load i32, ptr %o4
  %t15 = getelementptr { i32, i32 }, ptr %o9, i32 0, i32 0
  store i32 %t14, ptr %t15
  %t16 = load float, ptr %o5
  %t17 = getelementptr { float, float }, ptr %o17, i32 0, i32 0
  store float %t16, ptr %t17
  %t18 = load i32, ptr %o6
  %t19 = getelementptr { ptr, i32 }, ptr %o7, i32 0, i32 1
  store i32 %t18, ptr %t19
  %t20 = load i32, ptr %o6
  %t21 = getelementptr { i32, i32 }, ptr %o11, i32 0, i32 1
  store i32 %t20, ptr %t21
  %t22 = load ptr, ptr %o2
  %t23 = getelementptr { ptr, i32 }, ptr %o7, i32 0, i32 1
  %t24 = load i32, ptr %t23
  %t25 = sext i32 %t24 to i64
  %t26 = getelementptr [8 x float], ptr %t22, i64 0, i64 %t25
  %t27 = load float, ptr %t26
  store float %t27, ptr %o8
  %t28 = load float, ptr %o8
  %t29 = getelementptr { float, float }, ptr %o15, i32 0, i32 0
  store float %t28, ptr %t29
  %t30 = getelementptr { i32, i32 }, ptr %o9, i32 0, i32 1
  store i32 8, ptr %t30
  %t31 = getelementptr { i32, i32 }, ptr %o9, i32 0, i32 0
  %t32 = load i32, ptr %t31
  %t33 = getelementptr { i32, i32 }, ptr %o9, i32 0, i32 1
  %t34 = load i32, ptr %t33
  %t35 = mul i32 %t32, %t34
  store i32 %t35, ptr %o10
  %t36 = load i32, ptr %o10
  %t37 = getelementptr { i32, i32 }, ptr %o11, i32 0, i32 0
  store i32 %t36, ptr %t37
  %t38 = getelementptr { i32, i32 }, ptr %o11, i32 0, i32 0
  %t39 = load i32, ptr %t38
  %t40 = getelementptr { i32, i32 }, ptr %o11, i32 0, i32 1
  %t41 = load i32, ptr %t40
  %t42 = add i32 %t39, %t41
  store i32 %t42, ptr %o12
  %t43 = load i32, ptr %o12
  %t44 = getelementptr { ptr, i32 }, ptr %o13, i32 0, i32 1
  store i32 %t43, ptr %t44
  %t45 = load ptr, ptr %o3
  %t46 = getelementptr { ptr, i32 }, ptr %o13, i32 0, i32 1
  %t47 = load i32, ptr %t46
  %t48 = sext i32 %t47 to i64
  %t49 = getelementptr [128 x float], ptr %t45, i64 0, i64 %t48
  %t50 = load float, ptr %t49
  store float %t50, ptr %o14
  %t51 = load float, ptr %o14
  %t52 = getelementptr { float, float }, ptr %o15, i32 0, i32 1
  store float %t51, ptr %t52
  %t53 = getelementptr { float, float }, ptr %o15, i32 0, i32 0
  %t54 = load float, ptr %t53
  %t55 = getelementptr { float, float }, ptr %o15, i32 0, i32 1
  %t56 = load float, ptr %t55
  %t57 = fmul float %t54, %t56
  store float %t57, ptr %o16
  %t58 = load float, ptr %o16
  %t59 = getelementptr { float, float }, ptr %o17, i32 0, i32 1
  store float %t58, ptr %t59
  %t60 = getelementptr { float, float }, ptr %o17, i32 0, i32 0
  %t61 = load float, ptr %t60
  %t62 = getelementptr { float, float }, ptr %o17, i32 0, i32 1
  %t63 = load float, ptr %t62
  %t64 = fadd float %t61, %t63
  store float %t64, ptr %o1
  %t65 = load float, ptr %o1
  ret float %t65
}

define internal float @fn14({ ptr, ptr, ptr, i32 } %arg) readonly nounwind willreturn {
entry:
  %o0 = alloca { ptr, ptr, ptr, i32 }
  %o1 = alloca float
  %o2 = alloca ptr
  %o3 = alloca ptr
  %o4 = alloca ptr
  %o5 = alloca i32
  %o6 = alloca { ptr, ptr, i32, float, ptr }
  %s20 = alloca float
  %s21 = alloca i64
  %s30 = alloca { ptr, ptr, i32, float, i32 }
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
  %t9 = getelementptr { ptr, ptr, i32, float, ptr }, ptr %o6, i32 0, i32 4
  store ptr %t8, ptr %t9
  %t10 = load ptr, ptr %o3
  %t11 = getelementptr { ptr, ptr, i32, float, ptr }, ptr %o6, i32 0, i32 0
  store ptr %t10, ptr %t11
  %t12 = load ptr, ptr %o4
  %t13 = getelementptr { ptr, ptr, i32, float, ptr }, ptr %o6, i32 0, i32 1
  store ptr %t12, ptr %t13
  %t14 = load i32, ptr %o5
  %t15 = getelementptr { ptr, ptr, i32, float, ptr }, ptr %o6, i32 0, i32 2
  store i32 %t14, ptr %t15
  %t16 = getelementptr { ptr, ptr, i32, float, ptr }, ptr %o6, i32 0, i32 3
  store float 0x0000000000000000, ptr %t16
  %t17 = getelementptr { ptr, ptr, i32, float, ptr }, ptr %o6, i32 0, i32 3
  %t18 = load float, ptr %t17
  %t19 = load ptr, ptr %o2
  store float %t18, ptr %s20
  store i64 0, ptr %s21
  br label %bb22
bb22:
  %t25 = load i64, ptr %s21
  %t26 = icmp uge i64 %t25, 8
  br i1 %t26, label %bb24, label %bb23
bb23:
  %t27 = getelementptr [8 x i32], ptr %t19, i64 0, i64 %t25
  %t28 = load i32, ptr %t27
  %t29 = load float, ptr %s20
  %t31 = load ptr, ptr %o3
  %t32 = getelementptr { ptr, ptr, i32, float, i32 }, ptr %s30, i32 0, i32 0
  store ptr %t31, ptr %t32
  %t33 = load ptr, ptr %o4
  %t34 = getelementptr { ptr, ptr, i32, float, i32 }, ptr %s30, i32 0, i32 1
  store ptr %t33, ptr %t34
  %t35 = getelementptr { ptr, ptr, i32, float, ptr }, ptr %o6, i32 0, i32 2
  %t36 = load i32, ptr %t35
  %t37 = getelementptr { ptr, ptr, i32, float, i32 }, ptr %s30, i32 0, i32 2
  store i32 %t36, ptr %t37
  %t38 = getelementptr { ptr, ptr, i32, float, i32 }, ptr %s30, i32 0, i32 3
  store float %t29, ptr %t38
  %t39 = getelementptr { ptr, ptr, i32, float, i32 }, ptr %s30, i32 0, i32 4
  store i32 %t28, ptr %t39
  %t40 = load { ptr, ptr, i32, float, i32 }, ptr %s30
  %t41 = call float @fn13({ ptr, ptr, i32, float, i32 } %t40)
  store float %t41, ptr %s20
  %t42 = add i64 %t25, 1
  store i64 %t42, ptr %s21
  br label %bb22
bb24:
  %t43 = load float, ptr %s20
  store float %t43, ptr %o1
  %t44 = load float, ptr %o1
  ret float %t44
}

define internal float @fn15({ ptr, i32 } %arg) readonly nounwind willreturn {
entry:
  %o0 = alloca { ptr, i32 }
  %o1 = alloca float
  %o2 = alloca ptr
  %o3 = alloca i32
  %o4 = alloca { ptr, i32 }
  %o5 = alloca float
  %o6 = alloca { float, float }
  store { ptr, i32 } %arg, ptr %o0
  %t0 = getelementptr { ptr, i32 }, ptr %o0, i32 0, i32 0
  %t1 = load ptr, ptr %t0
  store ptr %t1, ptr %o2
  %t2 = getelementptr { ptr, i32 }, ptr %o0, i32 0, i32 1
  %t3 = load i32, ptr %t2
  store i32 %t3, ptr %o3
  %t4 = load ptr, ptr %o2
  %t5 = getelementptr { ptr, i32 }, ptr %o4, i32 0, i32 0
  store ptr %t4, ptr %t5
  %t6 = load i32, ptr %o3
  %t7 = getelementptr { ptr, i32 }, ptr %o4, i32 0, i32 1
  store i32 %t6, ptr %t7
  %t8 = load ptr, ptr %o2
  %t9 = getelementptr { ptr, i32 }, ptr %o4, i32 0, i32 1
  %t10 = load i32, ptr %t9
  %t11 = sext i32 %t10 to i64
  %t12 = getelementptr [16 x float], ptr %t8, i64 0, i64 %t11
  %t13 = load float, ptr %t12
  store float %t13, ptr %o5
  %t14 = load float, ptr %o5
  %t15 = getelementptr { float, float }, ptr %o6, i32 0, i32 0
  store float %t14, ptr %t15
  %t16 = getelementptr { float, float }, ptr %o6, i32 0, i32 1
  store float 0x4030000000000000, ptr %t16
  %t17 = getelementptr { float, float }, ptr %o6, i32 0, i32 0
  %t18 = load float, ptr %t17
  %t19 = getelementptr { float, float }, ptr %o6, i32 0, i32 1
  %t20 = load float, ptr %t19
  %t21 = fdiv float %t18, %t20
  store float %t21, ptr %o1
  %t22 = load float, ptr %o1
  ret float %t22
}

define internal float @fn16({ ptr, ptr, i32, float, i32 } %arg) readonly nounwind willreturn {
entry:
  %o0 = alloca { ptr, ptr, i32, float, i32 }
  %o1 = alloca float
  %o2 = alloca ptr
  %o3 = alloca ptr
  %o4 = alloca i32
  %o5 = alloca float
  %o6 = alloca i32
  %o7 = alloca { ptr, i32 }
  %o8 = alloca float
  %o9 = alloca { i32, i32 }
  %o10 = alloca i32
  %o11 = alloca { i32, i32 }
  %o12 = alloca i32
  %o13 = alloca { ptr, i32 }
  %o14 = alloca float
  %o15 = alloca { float, float }
  %o16 = alloca float
  %o17 = alloca { float, float }
  store { ptr, ptr, i32, float, i32 } %arg, ptr %o0
  %t0 = getelementptr { ptr, ptr, i32, float, i32 }, ptr %o0, i32 0, i32 0
  %t1 = load ptr, ptr %t0
  store ptr %t1, ptr %o2
  %t2 = getelementptr { ptr, ptr, i32, float, i32 }, ptr %o0, i32 0, i32 1
  %t3 = load ptr, ptr %t2
  store ptr %t3, ptr %o3
  %t4 = getelementptr { ptr, ptr, i32, float, i32 }, ptr %o0, i32 0, i32 2
  %t5 = load i32, ptr %t4
  store i32 %t5, ptr %o4
  %t6 = getelementptr { ptr, ptr, i32, float, i32 }, ptr %o0, i32 0, i32 3
  %t7 = load float, ptr %t6
  store float %t7, ptr %o5
  %t8 = getelementptr { ptr, ptr, i32, float, i32 }, ptr %o0, i32 0, i32 4
  %t9 = load i32, ptr %t8
  store i32 %t9, ptr %o6
  %t10 = load ptr, ptr %o2
  %t11 = getelementptr { ptr, i32 }, ptr %o7, i32 0, i32 0
  store ptr %t10, ptr %t11
  %t12 = load ptr, ptr %o3
  %t13 = getelementptr { ptr, i32 }, ptr %o13, i32 0, i32 0
  store ptr %t12, ptr %t13
  %t14 = load i32, ptr %o4
  %t15 = getelementptr { i32, i32 }, ptr %o11, i32 0, i32 1
  store i32 %t14, ptr %t15
  %t16 = load float, ptr %o5
  %t17 = getelementptr { float, float }, ptr %o17, i32 0, i32 0
  store float %t16, ptr %t17
  %t18 = load i32, ptr %o6
  %t19 = getelementptr { ptr, i32 }, ptr %o7, i32 0, i32 1
  store i32 %t18, ptr %t19
  %t20 = load i32, ptr %o6
  %t21 = getelementptr { i32, i32 }, ptr %o9, i32 0, i32 0
  store i32 %t20, ptr %t21
  %t22 = load ptr, ptr %o2
  %t23 = getelementptr { ptr, i32 }, ptr %o7, i32 0, i32 1
  %t24 = load i32, ptr %t23
  %t25 = sext i32 %t24 to i64
  %t26 = getelementptr [16 x float], ptr %t22, i64 0, i64 %t25
  %t27 = load float, ptr %t26
  store float %t27, ptr %o8
  %t28 = load float, ptr %o8
  %t29 = getelementptr { float, float }, ptr %o15, i32 0, i32 0
  store float %t28, ptr %t29
  %t30 = getelementptr { i32, i32 }, ptr %o9, i32 0, i32 1
  store i32 8, ptr %t30
  %t31 = getelementptr { i32, i32 }, ptr %o9, i32 0, i32 0
  %t32 = load i32, ptr %t31
  %t33 = getelementptr { i32, i32 }, ptr %o9, i32 0, i32 1
  %t34 = load i32, ptr %t33
  %t35 = mul i32 %t32, %t34
  store i32 %t35, ptr %o10
  %t36 = load i32, ptr %o10
  %t37 = getelementptr { i32, i32 }, ptr %o11, i32 0, i32 0
  store i32 %t36, ptr %t37
  %t38 = getelementptr { i32, i32 }, ptr %o11, i32 0, i32 0
  %t39 = load i32, ptr %t38
  %t40 = getelementptr { i32, i32 }, ptr %o11, i32 0, i32 1
  %t41 = load i32, ptr %t40
  %t42 = add i32 %t39, %t41
  store i32 %t42, ptr %o12
  %t43 = load i32, ptr %o12
  %t44 = getelementptr { ptr, i32 }, ptr %o13, i32 0, i32 1
  store i32 %t43, ptr %t44
  %t45 = load ptr, ptr %o3
  %t46 = getelementptr { ptr, i32 }, ptr %o13, i32 0, i32 1
  %t47 = load i32, ptr %t46
  %t48 = sext i32 %t47 to i64
  %t49 = getelementptr [128 x float], ptr %t45, i64 0, i64 %t48
  %t50 = load float, ptr %t49
  store float %t50, ptr %o14
  %t51 = load float, ptr %o14
  %t52 = getelementptr { float, float }, ptr %o15, i32 0, i32 1
  store float %t51, ptr %t52
  %t53 = getelementptr { float, float }, ptr %o15, i32 0, i32 0
  %t54 = load float, ptr %t53
  %t55 = getelementptr { float, float }, ptr %o15, i32 0, i32 1
  %t56 = load float, ptr %t55
  %t57 = fmul float %t54, %t56
  store float %t57, ptr %o16
  %t58 = load float, ptr %o16
  %t59 = getelementptr { float, float }, ptr %o17, i32 0, i32 1
  store float %t58, ptr %t59
  %t60 = getelementptr { float, float }, ptr %o17, i32 0, i32 0
  %t61 = load float, ptr %t60
  %t62 = getelementptr { float, float }, ptr %o17, i32 0, i32 1
  %t63 = load float, ptr %t62
  %t64 = fadd float %t61, %t63
  store float %t64, ptr %o1
  %t65 = load float, ptr %o1
  ret float %t65
}

define internal float @fn17({ ptr, ptr, ptr, i32 } %arg) readonly nounwind willreturn {
entry:
  %o0 = alloca { ptr, ptr, ptr, i32 }
  %o1 = alloca float
  %o2 = alloca ptr
  %o3 = alloca ptr
  %o4 = alloca ptr
  %o5 = alloca i32
  %o6 = alloca { ptr, ptr, i32, float, ptr }
  %s20 = alloca float
  %s21 = alloca i64
  %s30 = alloca { ptr, ptr, i32, float, i32 }
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
  %t9 = getelementptr { ptr, ptr, i32, float, ptr }, ptr %o6, i32 0, i32 4
  store ptr %t8, ptr %t9
  %t10 = load ptr, ptr %o3
  %t11 = getelementptr { ptr, ptr, i32, float, ptr }, ptr %o6, i32 0, i32 0
  store ptr %t10, ptr %t11
  %t12 = load ptr, ptr %o4
  %t13 = getelementptr { ptr, ptr, i32, float, ptr }, ptr %o6, i32 0, i32 1
  store ptr %t12, ptr %t13
  %t14 = load i32, ptr %o5
  %t15 = getelementptr { ptr, ptr, i32, float, ptr }, ptr %o6, i32 0, i32 2
  store i32 %t14, ptr %t15
  %t16 = getelementptr { ptr, ptr, i32, float, ptr }, ptr %o6, i32 0, i32 3
  store float 0x0000000000000000, ptr %t16
  %t17 = getelementptr { ptr, ptr, i32, float, ptr }, ptr %o6, i32 0, i32 3
  %t18 = load float, ptr %t17
  %t19 = load ptr, ptr %o2
  store float %t18, ptr %s20
  store i64 0, ptr %s21
  br label %bb22
bb22:
  %t25 = load i64, ptr %s21
  %t26 = icmp uge i64 %t25, 16
  br i1 %t26, label %bb24, label %bb23
bb23:
  %t27 = getelementptr [16 x i32], ptr %t19, i64 0, i64 %t25
  %t28 = load i32, ptr %t27
  %t29 = load float, ptr %s20
  %t31 = load ptr, ptr %o3
  %t32 = getelementptr { ptr, ptr, i32, float, i32 }, ptr %s30, i32 0, i32 0
  store ptr %t31, ptr %t32
  %t33 = load ptr, ptr %o4
  %t34 = getelementptr { ptr, ptr, i32, float, i32 }, ptr %s30, i32 0, i32 1
  store ptr %t33, ptr %t34
  %t35 = getelementptr { ptr, ptr, i32, float, ptr }, ptr %o6, i32 0, i32 2
  %t36 = load i32, ptr %t35
  %t37 = getelementptr { ptr, ptr, i32, float, i32 }, ptr %s30, i32 0, i32 2
  store i32 %t36, ptr %t37
  %t38 = getelementptr { ptr, ptr, i32, float, i32 }, ptr %s30, i32 0, i32 3
  store float %t29, ptr %t38
  %t39 = getelementptr { ptr, ptr, i32, float, i32 }, ptr %s30, i32 0, i32 4
  store i32 %t28, ptr %t39
  %t40 = load { ptr, ptr, i32, float, i32 }, ptr %s30
  %t41 = call float @fn16({ ptr, ptr, i32, float, i32 } %t40)
  store float %t41, ptr %s20
  %t42 = add i64 %t25, 1
  store i64 %t42, ptr %s21
  br label %bb22
bb24:
  %t43 = load float, ptr %s20
  store float %t43, ptr %o1
  %t44 = load float, ptr %o1
  ret float %t44
}

define internal float @fn18({ ptr, i32, ptr, i32, float, i32 } %arg) readonly nounwind willreturn {
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
  store i32 8, ptr %t26
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
  %t45 = getelementptr [8 x float], ptr %t41, i64 0, i64 %t44
  %t46 = load float, ptr %t45
  store float %t46, ptr %o13
  %t47 = load float, ptr %o13
  %t48 = getelementptr { float, float }, ptr %o20, i32 0, i32 0
  store float %t47, ptr %t48
  %t49 = getelementptr { i32, i32 }, ptr %o14, i32 0, i32 1
  store i32 8, ptr %t49
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
  %t68 = getelementptr [64 x float], ptr %t64, i64 0, i64 %t67
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

define internal float @fn19({ ptr, ptr, ptr, i32 } %arg) readonly nounwind willreturn {
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
  store i32 8, ptr %t18
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
  store i32 8, ptr %t26
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
  %t44 = icmp uge i64 %t43, 8
  br i1 %t44, label %bb42, label %bb41
bb41:
  %t45 = getelementptr [8 x i32], ptr %t37, i64 0, i64 %t43
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
  %t62 = call float @fn18({ ptr, i32, ptr, i32, float, i32 } %t61)
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

define internal float @fn20({ ptr, i32, ptr, i32, float, i32 } %arg) readonly nounwind willreturn {
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
  store i32 8, ptr %t26
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
  %t45 = getelementptr [8 x float], ptr %t41, i64 0, i64 %t44
  %t46 = load float, ptr %t45
  store float %t46, ptr %o13
  %t47 = load float, ptr %o13
  %t48 = getelementptr { float, float }, ptr %o20, i32 0, i32 0
  store float %t47, ptr %t48
  %t49 = getelementptr { i32, i32 }, ptr %o14, i32 0, i32 1
  store i32 16, ptr %t49
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
  %t68 = getelementptr [128 x float], ptr %t64, i64 0, i64 %t67
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

define internal float @fn21({ ptr, ptr, ptr, i32 } %arg) readonly nounwind willreturn {
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
  store i32 16, ptr %t18
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
  store i32 16, ptr %t26
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
  %t44 = icmp uge i64 %t43, 8
  br i1 %t44, label %bb42, label %bb41
bb41:
  %t45 = getelementptr [8 x i32], ptr %t37, i64 0, i64 %t43
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
  %t62 = call float @fn20({ ptr, i32, ptr, i32, float, i32 } %t61)
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

define internal float @fn22({ ptr, ptr, i32 } %arg) readonly nounwind willreturn {
entry:
  %o0 = alloca { ptr, ptr, i32 }
  %o1 = alloca float
  %o2 = alloca ptr
  %o3 = alloca ptr
  %o4 = alloca i32
  %o5 = alloca { ptr, i32 }
  %o6 = alloca float
  %o7 = alloca { i32, i32 }
  %o8 = alloca i32
  %o9 = alloca { ptr, i32 }
  %o10 = alloca float
  %o11 = alloca { float, float }
  %o12 = alloca float
  %o13 = alloca { float, float }
  %o14 = alloca i1
  %o15 = alloca { float, float, i1 }
  %o16 = alloca float
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
  %t7 = getelementptr { ptr, i32 }, ptr %o5, i32 0, i32 0
  store ptr %t6, ptr %t7
  %t8 = load ptr, ptr %o3
  %t9 = getelementptr { ptr, i32 }, ptr %o9, i32 0, i32 0
  store ptr %t8, ptr %t9
  %t10 = load i32, ptr %o4
  %t11 = getelementptr { ptr, i32 }, ptr %o5, i32 0, i32 1
  store i32 %t10, ptr %t11
  %t12 = load i32, ptr %o4
  %t13 = getelementptr { i32, i32 }, ptr %o7, i32 0, i32 0
  store i32 %t12, ptr %t13
  %t14 = load ptr, ptr %o2
  %t15 = getelementptr { ptr, i32 }, ptr %o5, i32 0, i32 1
  %t16 = load i32, ptr %t15
  %t17 = sext i32 %t16 to i64
  %t18 = getelementptr [16 x float], ptr %t14, i64 0, i64 %t17
  %t19 = load float, ptr %t18
  store float %t19, ptr %o6
  %t20 = load float, ptr %o6
  %t21 = getelementptr { float, float }, ptr %o11, i32 0, i32 0
  store float %t20, ptr %t21
  %t22 = getelementptr { i32, i32 }, ptr %o7, i32 0, i32 1
  store i32 16, ptr %t22
  %t23 = getelementptr { i32, i32 }, ptr %o7, i32 0, i32 0
  %t24 = load i32, ptr %t23
  %t25 = getelementptr { i32, i32 }, ptr %o7, i32 0, i32 1
  %t26 = load i32, ptr %t25
  %t27 = srem i32 %t24, %t26
  store i32 %t27, ptr %o8
  %t28 = load i32, ptr %o8
  %t29 = getelementptr { ptr, i32 }, ptr %o9, i32 0, i32 1
  store i32 %t28, ptr %t29
  %t30 = load ptr, ptr %o3
  %t31 = getelementptr { ptr, i32 }, ptr %o9, i32 0, i32 1
  %t32 = load i32, ptr %t31
  %t33 = sext i32 %t32 to i64
  %t34 = getelementptr [16 x float], ptr %t30, i64 0, i64 %t33
  %t35 = load float, ptr %t34
  store float %t35, ptr %o10
  %t36 = load float, ptr %o10
  %t37 = getelementptr { float, float }, ptr %o11, i32 0, i32 1
  store float %t36, ptr %t37
  %t38 = getelementptr { float, float }, ptr %o11, i32 0, i32 0
  %t39 = load float, ptr %t38
  %t40 = getelementptr { float, float }, ptr %o11, i32 0, i32 1
  %t41 = load float, ptr %t40
  %t42 = fadd float %t39, %t41
  store float %t42, ptr %o12
  %t43 = load float, ptr %o12
  %t44 = getelementptr { float, float }, ptr %o13, i32 0, i32 0
  store float %t43, ptr %t44
  %t45 = load float, ptr %o12
  %t46 = getelementptr { float, float, i1 }, ptr %o15, i32 0, i32 0
  store float %t45, ptr %t46
  %t47 = getelementptr { float, float }, ptr %o13, i32 0, i32 1
  store float 0x0000000000000000, ptr %t47
  %t48 = getelementptr { float, float }, ptr %o13, i32 0, i32 0
  %t49 = load float, ptr %t48
  %t50 = getelementptr { float, float }, ptr %o13, i32 0, i32 1
  %t51 = load float, ptr %t50
  %t52 = fcmp ogt float %t49, %t51
  store i1 %t52, ptr %o14
  %t53 = load i1, ptr %o14
  %t54 = getelementptr { float, float, i1 }, ptr %o15, i32 0, i32 2
  store i1 %t53, ptr %t54
  %t55 = getelementptr { float, float, i1 }, ptr %o15, i32 0, i32 1
  store float 0x0000000000000000, ptr %t55
  %t56 = getelementptr { float, float, i1 }, ptr %o15, i32 0, i32 0
  %t57 = load float, ptr %t56
  %t58 = getelementptr { float, float, i1 }, ptr %o15, i32 0, i32 1
  %t59 = load float, ptr %t58
  %t60 = getelementptr { float, float, i1 }, ptr %o15, i32 0, i32 2
  %t61 = load i1, ptr %t60
  %t62 = select i1 %t61, float %t57, float %t59
  store float %t62, ptr %o16
  %t63 = load float, ptr %o16
  store float %t63, ptr %o1
  %t64 = load float, ptr %o1
  ret float %t64
}

define internal float @fn23({ ptr, i32, ptr, i32, float, i32 } %arg) readonly nounwind willreturn {
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
  store i32 16, ptr %t26
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
  %t45 = getelementptr [16 x float], ptr %t41, i64 0, i64 %t44
  %t46 = load float, ptr %t45
  store float %t46, ptr %o13
  %t47 = load float, ptr %o13
  %t48 = getelementptr { float, float }, ptr %o20, i32 0, i32 0
  store float %t47, ptr %t48
  %t49 = getelementptr { i32, i32 }, ptr %o14, i32 0, i32 1
  store i32 8, ptr %t49
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
  %t68 = getelementptr [128 x float], ptr %t64, i64 0, i64 %t67
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

define internal float @fn24({ ptr, ptr, ptr, i32 } %arg) readonly nounwind willreturn {
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
  store i32 8, ptr %t18
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
  store i32 8, ptr %t26
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
  %t44 = icmp uge i64 %t43, 16
  br i1 %t44, label %bb42, label %bb41
bb41:
  %t45 = getelementptr [16 x i32], ptr %t37, i64 0, i64 %t43
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
  %t62 = call float @fn23({ ptr, i32, ptr, i32, float, i32 } %t61)
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

define internal float @fn25({ ptr, ptr, i32 } %arg) readonly nounwind willreturn {
entry:
  %o0 = alloca { ptr, ptr, i32 }
  %o1 = alloca float
  %o2 = alloca ptr
  %o3 = alloca ptr
  %o4 = alloca i32
  %o5 = alloca { ptr, i32 }
  %o6 = alloca float
  %o7 = alloca { i32, i32 }
  %o8 = alloca i32
  %o9 = alloca { ptr, i32 }
  %o10 = alloca float
  %o11 = alloca { float, float }
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
  %t7 = getelementptr { ptr, i32 }, ptr %o5, i32 0, i32 0
  store ptr %t6, ptr %t7
  %t8 = load ptr, ptr %o3
  %t9 = getelementptr { ptr, i32 }, ptr %o9, i32 0, i32 0
  store ptr %t8, ptr %t9
  %t10 = load i32, ptr %o4
  %t11 = getelementptr { ptr, i32 }, ptr %o5, i32 0, i32 1
  store i32 %t10, ptr %t11
  %t12 = load i32, ptr %o4
  %t13 = getelementptr { i32, i32 }, ptr %o7, i32 0, i32 0
  store i32 %t12, ptr %t13
  %t14 = load ptr, ptr %o2
  %t15 = getelementptr { ptr, i32 }, ptr %o5, i32 0, i32 1
  %t16 = load i32, ptr %t15
  %t17 = sext i32 %t16 to i64
  %t18 = getelementptr [8 x float], ptr %t14, i64 0, i64 %t17
  %t19 = load float, ptr %t18
  store float %t19, ptr %o6
  %t20 = load float, ptr %o6
  %t21 = getelementptr { float, float }, ptr %o11, i32 0, i32 0
  store float %t20, ptr %t21
  %t22 = getelementptr { i32, i32 }, ptr %o7, i32 0, i32 1
  store i32 8, ptr %t22
  %t23 = getelementptr { i32, i32 }, ptr %o7, i32 0, i32 0
  %t24 = load i32, ptr %t23
  %t25 = getelementptr { i32, i32 }, ptr %o7, i32 0, i32 1
  %t26 = load i32, ptr %t25
  %t27 = srem i32 %t24, %t26
  store i32 %t27, ptr %o8
  %t28 = load i32, ptr %o8
  %t29 = getelementptr { ptr, i32 }, ptr %o9, i32 0, i32 1
  store i32 %t28, ptr %t29
  %t30 = load ptr, ptr %o3
  %t31 = getelementptr { ptr, i32 }, ptr %o9, i32 0, i32 1
  %t32 = load i32, ptr %t31
  %t33 = sext i32 %t32 to i64
  %t34 = getelementptr [8 x float], ptr %t30, i64 0, i64 %t33
  %t35 = load float, ptr %t34
  store float %t35, ptr %o10
  %t36 = load float, ptr %o10
  %t37 = getelementptr { float, float }, ptr %o11, i32 0, i32 1
  store float %t36, ptr %t37
  %t38 = getelementptr { float, float }, ptr %o11, i32 0, i32 0
  %t39 = load float, ptr %t38
  %t40 = getelementptr { float, float }, ptr %o11, i32 0, i32 1
  %t41 = load float, ptr %t40
  %t42 = fadd float %t39, %t41
  store float %t42, ptr %o1
  %t43 = load float, ptr %o1
  ret float %t43
}

define internal float @fn26({ ptr, ptr, i32 } %arg) readonly nounwind willreturn {
entry:
  %o0 = alloca { ptr, ptr, i32 }
  %o1 = alloca float
  %o2 = alloca ptr
  %o3 = alloca ptr
  %o4 = alloca i32
  %o5 = alloca { ptr, i32 }
  %o6 = alloca float
  %o7 = alloca { ptr, i32 }
  %o8 = alloca float
  %o9 = alloca { float, float }
  %o10 = alloca float
  %o11 = alloca { float, float }
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
  %t7 = getelementptr { ptr, i32 }, ptr %o5, i32 0, i32 0
  store ptr %t6, ptr %t7
  %t8 = load ptr, ptr %o3
  %t9 = getelementptr { ptr, i32 }, ptr %o7, i32 0, i32 0
  store ptr %t8, ptr %t9
  %t10 = load i32, ptr %o4
  %t11 = getelementptr { ptr, i32 }, ptr %o5, i32 0, i32 1
  store i32 %t10, ptr %t11
  %t12 = load i32, ptr %o4
  %t13 = getelementptr { ptr, i32 }, ptr %o7, i32 0, i32 1
  store i32 %t12, ptr %t13
  %t14 = load ptr, ptr %o2
  %t15 = getelementptr { ptr, i32 }, ptr %o5, i32 0, i32 1
  %t16 = load i32, ptr %t15
  %t17 = sext i32 %t16 to i64
  %t18 = getelementptr [8 x float], ptr %t14, i64 0, i64 %t17
  %t19 = load float, ptr %t18
  store float %t19, ptr %o6
  %t20 = load float, ptr %o6
  %t21 = getelementptr { float, float }, ptr %o9, i32 0, i32 0
  store float %t20, ptr %t21
  %t22 = load ptr, ptr %o3
  %t23 = getelementptr { ptr, i32 }, ptr %o7, i32 0, i32 1
  %t24 = load i32, ptr %t23
  %t25 = sext i32 %t24 to i64
  %t26 = getelementptr [8 x float], ptr %t22, i64 0, i64 %t25
  %t27 = load float, ptr %t26
  store float %t27, ptr %o8
  %t28 = load float, ptr %o8
  %t29 = getelementptr { float, float }, ptr %o9, i32 0, i32 1
  store float %t28, ptr %t29
  %t30 = getelementptr { float, float }, ptr %o9, i32 0, i32 0
  %t31 = load float, ptr %t30
  %t32 = getelementptr { float, float }, ptr %o9, i32 0, i32 1
  %t33 = load float, ptr %t32
  %t34 = fadd float %t31, %t33
  store float %t34, ptr %o10
  %t35 = load float, ptr %o10
  %t36 = getelementptr { float, float }, ptr %o11, i32 0, i32 0
  store float %t35, ptr %t36
  %t37 = getelementptr { float, float }, ptr %o11, i32 0, i32 1
  store float 0x3FD0000000000000, ptr %t37
  %t38 = getelementptr { float, float }, ptr %o11, i32 0, i32 0
  %t39 = load float, ptr %t38
  %t40 = getelementptr { float, float }, ptr %o11, i32 0, i32 1
  %t41 = load float, ptr %t40
  %t42 = fmul float %t39, %t41
  store float %t42, ptr %o1
  %t43 = load float, ptr %o1
  ret float %t43
}

define i32 @main() {
entry:
  call void @mapal_gpu_init(i32 1)
  call void @mapal_main()
  ret i32 0
}
