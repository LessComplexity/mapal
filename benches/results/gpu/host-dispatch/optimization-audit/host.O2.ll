; ModuleID = '/private/tmp/mapal-s56-gpu/large-composed/host-optimization-evidence/host.emitted.ll'
source_filename = "/private/tmp/mapal-s56-gpu/large-composed/host-optimization-evidence/host.emitted.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@.str0 = private unnamed_addr constant [5 x i8] c"step="
@.str1 = private unnamed_addr constant [1 x i8] c"y"

declare void @mapal_print_i32(i32, i1 zeroext) local_unnamed_addr

declare void @mapal_print_f32(float, i1 zeroext) local_unnamed_addr

declare void @mapal_print_str(ptr, i64, i1 zeroext) local_unnamed_addr

declare void @mapal_gpu_init(i32) local_unnamed_addr

declare i64 @mapal_gpu_alloc(ptr, i64) local_unnamed_addr

declare void @mapal_gpu_h2d(i64, ptr, i64) local_unnamed_addr

declare void @mapal_gpu_d2h(ptr, i64, i64, i64) local_unnamed_addr

declare void @mapal_gpu_launch(i64, i64, i64, i64, i32, i32, i32, i32) local_unnamed_addr

declare void @mapal_gpu_gen(i64, i64, i64) local_unnamed_addr

declare void @mapal_gpu_free(ptr) local_unnamed_addr

declare void @mapal_gpu_launchv(i64, i32, i32, ptr, i32, i32, i32, i32) local_unnamed_addr

define noundef i32 @main() local_unnamed_addr {
entry:
  %o45.i = alloca [128 x float], align 16
  %o271.i = alloca [8 x float], align 16
  %o289.i = alloca [8 x float], align 16
  %d0.i = alloca i64, align 8
  %d2.i = alloca i64, align 8
  %d4.i = alloca i64, align 8
  %d6.i = alloca i64, align 8
  %d8.i = alloca i64, align 8
  %d10.i = alloca i64, align 8
  %d12.i = alloca i64, align 8
  %d1027.i = alloca i64, align 8
  %d1045.i = alloca i64, align 8
  %d1050.i = alloca i64, align 8
  %d1069.i = alloca i64, align 8
  %d1074.i = alloca i64, align 8
  %p1076.i = alloca [4 x i64], align 8
  %d1085.i = alloca i64, align 8
  %d1090.i = alloca i64, align 8
  %d1097.i = alloca i64, align 8
  %p1099.i = alloca [5 x i64], align 8
  %d1111.i = alloca i64, align 8
  %p1113.i = alloca [6 x i64], align 8
  tail call void @mapal_gpu_init(i32 1)
  call void @llvm.lifetime.start.p0(ptr nonnull %o45.i)
  call void @llvm.lifetime.start.p0(ptr nonnull %o271.i)
  call void @llvm.lifetime.start.p0(ptr nonnull %o289.i)
  call void @llvm.lifetime.start.p0(ptr nonnull %d0.i)
  call void @llvm.lifetime.start.p0(ptr nonnull %d2.i)
  call void @llvm.lifetime.start.p0(ptr nonnull %d4.i)
  call void @llvm.lifetime.start.p0(ptr nonnull %d6.i)
  call void @llvm.lifetime.start.p0(ptr nonnull %d8.i)
  call void @llvm.lifetime.start.p0(ptr nonnull %d10.i)
  call void @llvm.lifetime.start.p0(ptr nonnull %d12.i)
  call void @llvm.lifetime.start.p0(ptr nonnull %d1027.i)
  call void @llvm.lifetime.start.p0(ptr nonnull %d1045.i)
  call void @llvm.lifetime.start.p0(ptr nonnull %d1050.i)
  call void @llvm.lifetime.start.p0(ptr nonnull %d1069.i)
  call void @llvm.lifetime.start.p0(ptr nonnull %d1074.i)
  call void @llvm.lifetime.start.p0(ptr nonnull %p1076.i)
  call void @llvm.lifetime.start.p0(ptr nonnull %d1085.i)
  call void @llvm.lifetime.start.p0(ptr nonnull %d1090.i)
  call void @llvm.lifetime.start.p0(ptr nonnull %d1097.i)
  call void @llvm.lifetime.start.p0(ptr nonnull %p1099.i)
  call void @llvm.lifetime.start.p0(ptr nonnull %d1111.i)
  call void @llvm.lifetime.start.p0(ptr nonnull %p1113.i)
  store i64 0, ptr %d0.i, align 8
  store i64 0, ptr %d2.i, align 8
  store i64 0, ptr %d4.i, align 8
  store i64 0, ptr %d6.i, align 8
  store i64 0, ptr %d8.i, align 8
  store i64 0, ptr %d10.i, align 8
  store i64 0, ptr %d12.i, align 8
  store i64 0, ptr %d1027.i, align 8
  store i64 0, ptr %d1045.i, align 8
  store i64 0, ptr %d1050.i, align 8
  store i64 0, ptr %d1069.i, align 8
  store i64 0, ptr %d1074.i, align 8
  store i64 0, ptr %d1085.i, align 8
  store i64 0, ptr %d1090.i, align 8
  store i64 0, ptr %d1097.i, align 8
  store i64 0, ptr %d1111.i, align 8
  %t1.i = call i64 @mapal_gpu_alloc(ptr nonnull %d0.i, i64 256)
  call void @mapal_gpu_gen(i64 570417511890179569, i64 %t1.i, i64 64)
  %t3.i = call i64 @mapal_gpu_alloc(ptr nonnull %d2.i, i64 512)
  call void @mapal_gpu_gen(i64 -1791989716614967237, i64 %t3.i, i64 128)
  %t5.i = call i64 @mapal_gpu_alloc(ptr nonnull %d4.i, i64 512)
  call void @mapal_gpu_gen(i64 -4223766215135587374, i64 %t5.i, i64 128)
  %t7.i = call i64 @mapal_gpu_alloc(ptr nonnull %d6.i, i64 256)
  call void @mapal_gpu_gen(i64 -9196861651627557674, i64 %t7.i, i64 64)
  %t9.i = call i64 @mapal_gpu_alloc(ptr nonnull %d8.i, i64 256)
  call void @mapal_gpu_gen(i64 5761537033769727445, i64 %t9.i, i64 64)
  %t11.i = call i64 @mapal_gpu_alloc(ptr nonnull %d10.i, i64 64)
  call void @mapal_gpu_gen(i64 1396038385954328191, i64 %t11.i, i64 16)
  %t13.i = call i64 @mapal_gpu_alloc(ptr nonnull %d12.i, i64 32)
  call void @mapal_gpu_gen(i64 -3200766962839905091, i64 %t13.i, i64 8)
  call void @mapal_print_str(ptr nonnull @.str0, i64 5, i1 zeroext false)
  call void @mapal_print_i32(i32 -1, i1 zeroext true)
  call void @mapal_print_str(ptr nonnull @.str1, i64 1, i1 zeroext true)
  %o44.sroa.9.0.o45.sroa_idx.i = getelementptr inbounds nuw i8, ptr %o45.i, i64 16
  %o44.sroa.17.0.o45.sroa_idx.i = getelementptr inbounds nuw i8, ptr %o45.i, i64 32
  %o44.sroa.25.0.o45.sroa_idx.i = getelementptr inbounds nuw i8, ptr %o45.i, i64 48
  %o44.sroa.33.0.o45.sroa_idx.i = getelementptr inbounds nuw i8, ptr %o45.i, i64 64
  %o44.sroa.41.0.o45.sroa_idx.i = getelementptr inbounds nuw i8, ptr %o45.i, i64 80
  %o44.sroa.49.0.o45.sroa_idx.i = getelementptr inbounds nuw i8, ptr %o45.i, i64 96
  %o44.sroa.57.0.o45.sroa_idx.i = getelementptr inbounds nuw i8, ptr %o45.i, i64 112
  %o44.sroa.65.0.o45.sroa_idx.i = getelementptr inbounds nuw i8, ptr %o45.i, i64 128
  %o44.sroa.73.0.o45.sroa_idx.i = getelementptr inbounds nuw i8, ptr %o45.i, i64 144
  %o44.sroa.81.0.o45.sroa_idx.i = getelementptr inbounds nuw i8, ptr %o45.i, i64 160
  %o44.sroa.89.0.o45.sroa_idx.i = getelementptr inbounds nuw i8, ptr %o45.i, i64 176
  %o44.sroa.97.0.o45.sroa_idx.i = getelementptr inbounds nuw i8, ptr %o45.i, i64 192
  %o44.sroa.105.0.o45.sroa_idx.i = getelementptr inbounds nuw i8, ptr %o45.i, i64 208
  %o44.sroa.113.0.o45.sroa_idx.i = getelementptr inbounds nuw i8, ptr %o45.i, i64 224
  %o44.sroa.121.0.o45.sroa_idx.i = getelementptr inbounds nuw i8, ptr %o45.i, i64 240
  %o44.sroa.129.0.o45.sroa_idx.i = getelementptr inbounds nuw i8, ptr %o45.i, i64 256
  %o44.sroa.137.0.o45.sroa_idx.i = getelementptr inbounds nuw i8, ptr %o45.i, i64 272
  %o44.sroa.145.0.o45.sroa_idx.i = getelementptr inbounds nuw i8, ptr %o45.i, i64 288
  %o44.sroa.153.0.o45.sroa_idx.i = getelementptr inbounds nuw i8, ptr %o45.i, i64 304
  %o44.sroa.161.0.o45.sroa_idx.i = getelementptr inbounds nuw i8, ptr %o45.i, i64 320
  %o44.sroa.169.0.o45.sroa_idx.i = getelementptr inbounds nuw i8, ptr %o45.i, i64 336
  %o44.sroa.177.0.o45.sroa_idx.i = getelementptr inbounds nuw i8, ptr %o45.i, i64 352
  %o44.sroa.185.0.o45.sroa_idx.i = getelementptr inbounds nuw i8, ptr %o45.i, i64 368
  %o44.sroa.193.0.o45.sroa_idx.i = getelementptr inbounds nuw i8, ptr %o45.i, i64 384
  %o44.sroa.201.0.o45.sroa_idx.i = getelementptr inbounds nuw i8, ptr %o45.i, i64 400
  %o44.sroa.209.0.o45.sroa_idx.i = getelementptr inbounds nuw i8, ptr %o45.i, i64 416
  %o44.sroa.217.0.o45.sroa_idx.i = getelementptr inbounds nuw i8, ptr %o45.i, i64 432
  %o44.sroa.225.0.o45.sroa_idx.i = getelementptr inbounds nuw i8, ptr %o45.i, i64 448
  %o44.sroa.233.0.o45.sroa_idx.i = getelementptr inbounds nuw i8, ptr %o45.i, i64 464
  %o44.sroa.241.0.o45.sroa_idx.i = getelementptr inbounds nuw i8, ptr %o45.i, i64 480
  %o44.sroa.249.0.o45.sroa_idx.i = getelementptr inbounds nuw i8, ptr %o45.i, i64 496
  %t1078.i = getelementptr inbounds nuw i8, ptr %p1076.i, i64 8
  %t1079.i = getelementptr inbounds nuw i8, ptr %p1076.i, i64 16
  %t1080.i = getelementptr inbounds nuw i8, ptr %p1076.i, i64 24
  %t1101.i = getelementptr inbounds nuw i8, ptr %p1099.i, i64 8
  %t1102.i = getelementptr inbounds nuw i8, ptr %p1099.i, i64 16
  %t1103.i = getelementptr inbounds nuw i8, ptr %p1099.i, i64 24
  %t1104.i = getelementptr inbounds nuw i8, ptr %p1099.i, i64 32
  %t1115.i = getelementptr inbounds nuw i8, ptr %p1113.i, i64 8
  %t1116.i = getelementptr inbounds nuw i8, ptr %p1113.i, i64 16
  %t1117.i = getelementptr inbounds nuw i8, ptr %p1113.i, i64 24
  %t1118.i = getelementptr inbounds nuw i8, ptr %p1113.i, i64 32
  %t1119.i = getelementptr inbounds nuw i8, ptr %p1113.i, i64 40
  %o292.sroa.134.512.o289.sroa_idx.i = getelementptr inbounds nuw i8, ptr %o289.i, i64 16
  %o292.sroa.135.512.o289.sroa_idx.i = getelementptr inbounds nuw i8, ptr %o289.i, i64 20
  %o292.sroa.136.512.o289.sroa_idx.i = getelementptr inbounds nuw i8, ptr %o289.i, i64 24
  %o292.sroa.137.512.o289.sroa_idx.i = getelementptr inbounds nuw i8, ptr %o289.i, i64 28
  %t1067.4.i = getelementptr inbounds nuw i8, ptr %o271.i, i64 16
  br label %bb62.i

bb62.i:                                           ; preds = %bb1036.i, %entry
  %indvars.iv.i = phi i64 [ 0, %entry ], [ %indvars.iv.next.i, %bb1036.i ]
  %s60.0161.i = phi i1 [ false, %entry ], [ true, %bb1036.i ]
  %storemerge160.i = phi i32 [ 0, %entry ], [ %t1127.i, %bb1036.i ]
  %o44.sroa.278.0159.i = phi float [ 0.000000e+00, %entry ], [ %o292.sroa.137.512.copyload.i, %bb1036.i ]
  %o44.sroa.275.0158.i = phi float [ 0.000000e+00, %entry ], [ %o292.sroa.136.512.copyload.i, %bb1036.i ]
  %o44.sroa.272.0157.i = phi float [ 0.000000e+00, %entry ], [ %o292.sroa.135.512.copyload.i, %bb1036.i ]
  %o44.sroa.269.0156.i = phi float [ 0.000000e+00, %entry ], [ %o292.sroa.134.512.copyload.i, %bb1036.i ]
  %0 = phi <4 x float> [ zeroinitializer, %entry ], [ %118, %bb1036.i ]
  %1 = phi <4 x float> [ zeroinitializer, %entry ], [ %70, %bb1036.i ]
  %2 = phi <4 x float> [ zeroinitializer, %entry ], [ %71, %bb1036.i ]
  %3 = phi <4 x float> [ zeroinitializer, %entry ], [ %72, %bb1036.i ]
  %4 = phi <4 x float> [ zeroinitializer, %entry ], [ %73, %bb1036.i ]
  %5 = phi <4 x float> [ zeroinitializer, %entry ], [ %74, %bb1036.i ]
  %6 = phi <4 x float> [ zeroinitializer, %entry ], [ %75, %bb1036.i ]
  %7 = phi <4 x float> [ zeroinitializer, %entry ], [ %76, %bb1036.i ]
  %8 = phi <4 x float> [ zeroinitializer, %entry ], [ %77, %bb1036.i ]
  %9 = phi <4 x float> [ zeroinitializer, %entry ], [ %78, %bb1036.i ]
  %10 = phi <4 x float> [ zeroinitializer, %entry ], [ %79, %bb1036.i ]
  %11 = phi <4 x float> [ zeroinitializer, %entry ], [ %80, %bb1036.i ]
  %12 = phi <4 x float> [ zeroinitializer, %entry ], [ %81, %bb1036.i ]
  %13 = phi <4 x float> [ zeroinitializer, %entry ], [ %82, %bb1036.i ]
  %14 = phi <4 x float> [ zeroinitializer, %entry ], [ %83, %bb1036.i ]
  %15 = phi <4 x float> [ zeroinitializer, %entry ], [ %84, %bb1036.i ]
  %16 = phi <4 x float> [ zeroinitializer, %entry ], [ %85, %bb1036.i ]
  %17 = phi <4 x float> [ zeroinitializer, %entry ], [ %86, %bb1036.i ]
  %18 = phi <4 x float> [ zeroinitializer, %entry ], [ %87, %bb1036.i ]
  %19 = phi <4 x float> [ zeroinitializer, %entry ], [ %88, %bb1036.i ]
  %20 = phi <4 x float> [ zeroinitializer, %entry ], [ %89, %bb1036.i ]
  %21 = phi <4 x float> [ zeroinitializer, %entry ], [ %90, %bb1036.i ]
  %22 = phi <4 x float> [ zeroinitializer, %entry ], [ %91, %bb1036.i ]
  %23 = phi <4 x float> [ zeroinitializer, %entry ], [ %92, %bb1036.i ]
  %24 = phi <4 x float> [ zeroinitializer, %entry ], [ %93, %bb1036.i ]
  %25 = phi <4 x float> [ zeroinitializer, %entry ], [ %94, %bb1036.i ]
  %26 = phi <4 x float> [ zeroinitializer, %entry ], [ %95, %bb1036.i ]
  %27 = phi <4 x float> [ zeroinitializer, %entry ], [ %96, %bb1036.i ]
  %28 = phi <4 x float> [ zeroinitializer, %entry ], [ %97, %bb1036.i ]
  %29 = phi <4 x float> [ zeroinitializer, %entry ], [ %98, %bb1036.i ]
  %30 = phi <4 x float> [ zeroinitializer, %entry ], [ %99, %bb1036.i ]
  %31 = phi <4 x float> [ zeroinitializer, %entry ], [ %100, %bb1036.i ]
  %32 = phi <4 x float> [ zeroinitializer, %entry ], [ %101, %bb1036.i ]
  store <4 x float> %1, ptr %o45.i, align 16
  store <4 x float> %2, ptr %o44.sroa.9.0.o45.sroa_idx.i, align 16
  store <4 x float> %3, ptr %o44.sroa.17.0.o45.sroa_idx.i, align 16
  store <4 x float> %4, ptr %o44.sroa.25.0.o45.sroa_idx.i, align 16
  store <4 x float> %5, ptr %o44.sroa.33.0.o45.sroa_idx.i, align 16
  store <4 x float> %6, ptr %o44.sroa.41.0.o45.sroa_idx.i, align 16
  store <4 x float> %7, ptr %o44.sroa.49.0.o45.sroa_idx.i, align 16
  store <4 x float> %8, ptr %o44.sroa.57.0.o45.sroa_idx.i, align 16
  store <4 x float> %9, ptr %o44.sroa.65.0.o45.sroa_idx.i, align 16
  store <4 x float> %10, ptr %o44.sroa.73.0.o45.sroa_idx.i, align 16
  store <4 x float> %11, ptr %o44.sroa.81.0.o45.sroa_idx.i, align 16
  store <4 x float> %12, ptr %o44.sroa.89.0.o45.sroa_idx.i, align 16
  store <4 x float> %13, ptr %o44.sroa.97.0.o45.sroa_idx.i, align 16
  store <4 x float> %14, ptr %o44.sroa.105.0.o45.sroa_idx.i, align 16
  store <4 x float> %15, ptr %o44.sroa.113.0.o45.sroa_idx.i, align 16
  store <4 x float> %16, ptr %o44.sroa.121.0.o45.sroa_idx.i, align 16
  store <4 x float> %17, ptr %o44.sroa.129.0.o45.sroa_idx.i, align 16
  store <4 x float> %18, ptr %o44.sroa.137.0.o45.sroa_idx.i, align 16
  store <4 x float> %19, ptr %o44.sroa.145.0.o45.sroa_idx.i, align 16
  store <4 x float> %20, ptr %o44.sroa.153.0.o45.sroa_idx.i, align 16
  store <4 x float> %21, ptr %o44.sroa.161.0.o45.sroa_idx.i, align 16
  store <4 x float> %22, ptr %o44.sroa.169.0.o45.sroa_idx.i, align 16
  store <4 x float> %23, ptr %o44.sroa.177.0.o45.sroa_idx.i, align 16
  store <4 x float> %24, ptr %o44.sroa.185.0.o45.sroa_idx.i, align 16
  store <4 x float> %25, ptr %o44.sroa.193.0.o45.sroa_idx.i, align 16
  store <4 x float> %26, ptr %o44.sroa.201.0.o45.sroa_idx.i, align 16
  store <4 x float> %27, ptr %o44.sroa.209.0.o45.sroa_idx.i, align 16
  store <4 x float> %28, ptr %o44.sroa.217.0.o45.sroa_idx.i, align 16
  store <4 x float> %29, ptr %o44.sroa.225.0.o45.sroa_idx.i, align 16
  store <4 x float> %30, ptr %o44.sroa.233.0.o45.sroa_idx.i, align 16
  store <4 x float> %31, ptr %o44.sroa.241.0.o45.sroa_idx.i, align 16
  store <4 x float> %32, ptr %o44.sroa.249.0.o45.sroa_idx.i, align 16
  %t248.i = shl nuw nsw i32 %storemerge160.i, 3
  %t26220.urem.i = add nsw i32 %t248.i, -101
  %t26220.cmp.i = icmp samesign ult i64 %indvars.iv.i, 13
  %t360.urem.i = add nsw i32 %t248.i, -100
  %33 = shl nuw nsw i64 %indvars.iv.i, 3
  %34 = or disjoint i64 %33, 1
  %35 = trunc i64 %33 to i32
  %36 = insertelement <4 x i32> poison, i32 %35, i64 0
  %37 = trunc i64 %34 to i32
  %38 = insertelement <4 x i32> %36, i32 %37, i64 1
  %39 = trunc nuw nsw i64 %34 to i32
  %40 = trunc nuw nsw i64 %33 to i32
  %t26220.i = select i1 %t26220.cmp.i, i32 %40, i32 %t26220.urem.i
  %t269.i = mul nsw i32 %t26220.i, 7
  %t276.i = add nsw i32 %t269.i, 13
  %t283.i = srem i32 %t276.i, 101
  %t338.i = getelementptr float, ptr %o45.i, i64 %33
  %t360.cmp.i = icmp samesign ult i64 %34, 101
  %t360.i = select i1 %t360.cmp.i, i32 %39, i32 %t360.urem.i
  %t367.i = mul nuw nsw i32 %t360.i, 7
  %t374.i = add nuw nsw i32 %t367.i, 13
  %t381.i = urem i32 %t374.i, 101
  %t556.urem.i = add nsw i32 %t248.i, -98
  %t458.urem.i = add nsw i32 %t248.i, -99
  %41 = or disjoint i64 %33, 3
  %42 = or disjoint i64 %33, 2
  %43 = trunc i64 %42 to i32
  %44 = insertelement <4 x i32> %38, i32 %43, i64 2
  %45 = trunc i64 %41 to i32
  %46 = insertelement <4 x i32> %44, i32 %45, i64 3
  %47 = trunc nuw nsw i64 %41 to i32
  %48 = trunc nuw nsw i64 %42 to i32
  %t556.cmp.i = icmp samesign ult i64 %41, 101
  %t458.cmp.i = icmp samesign ult i64 %42, 101
  %t556.i = select i1 %t556.cmp.i, i32 %47, i32 %t556.urem.i
  %t458.i = select i1 %t458.cmp.i, i32 %48, i32 %t458.urem.i
  %t563.i = mul nuw nsw i32 %t556.i, 7
  %t465.i = mul nuw nsw i32 %t458.i, 7
  %t570.i = add nuw nsw i32 %t563.i, 13
  %t472.i = add nuw nsw i32 %t465.i, 13
  %t577.i = urem i32 %t570.i, 101
  %t479.i = urem i32 %t472.i, 101
  %t584.i = add nsw i32 %t577.i, -50
  %t486.i = add nsw i32 %t479.i, -50
  %49 = insertelement <2 x i32> poison, i32 %t283.i, i64 0
  %50 = insertelement <2 x i32> %49, i32 %t381.i, i64 1
  %51 = add nsw <2 x i32> %50, splat (i32 -50)
  %52 = shufflevector <2 x i32> %51, <2 x i32> poison, <4 x i32> <i32 0, i32 1, i32 poison, i32 poison>
  %53 = insertelement <4 x i32> %52, i32 %t486.i, i64 2
  %54 = insertelement <4 x i32> %53, i32 %t584.i, i64 3
  %55 = sitofp <4 x i32> %54 to <4 x float>
  %56 = fmul <4 x float> %55, splat (float 1.562500e-02)
  %57 = fadd <4 x float> %0, %56
  store <4 x float> %57, ptr %t338.i, align 16
  %t654.urem.i = add nsw i32 %t248.i, -97
  %t752.urem.i = add nsw i32 %t248.i, -96
  %t850.urem.i = add nsw i32 %t248.i, -95
  %t948.urem.i = add nsw i32 %t248.i, -94
  %58 = insertelement <4 x i64> poison, i64 %33, i64 0
  %59 = shufflevector <4 x i64> %58, <4 x i64> poison, <4 x i32> zeroinitializer
  %60 = or disjoint <4 x i64> %59, <i64 4, i64 5, i64 6, i64 7>
  %61 = extractelement <4 x i64> %60, i64 3
  %62 = extractelement <4 x i64> %60, i64 2
  %63 = extractelement <4 x i64> %60, i64 1
  %64 = extractelement <4 x i64> %60, i64 0
  %t654.cmp.i = icmp samesign ult i64 %64, 101
  %65 = trunc nuw nsw i64 %61 to i32
  %66 = trunc <4 x i64> %60 to <4 x i32>
  %67 = trunc nuw nsw i64 %62 to i32
  %68 = trunc nuw nsw i64 %63 to i32
  %69 = trunc nuw nsw i64 %64 to i32
  %t654.i = select i1 %t654.cmp.i, i32 %69, i32 %t654.urem.i
  %t661.i = mul nuw nsw i32 %t654.i, 7
  %t668.i = add nuw nsw i32 %t661.i, 13
  %t675.i = urem i32 %t668.i, 101
  %t682.i = add nsw i32 %t675.i, -50
  %t684.i = sitofp i32 %t682.i to float
  %t691.i = fmul float %t684.i, 1.562500e-02
  %t705.i = fadd float %o44.sroa.269.0156.i, %t691.i
  %t730.i = getelementptr float, ptr %o45.i, i64 %64
  store float %t705.i, ptr %t730.i, align 16
  %t752.cmp.i = icmp samesign ult i64 %63, 101
  %t752.i = select i1 %t752.cmp.i, i32 %68, i32 %t752.urem.i
  %t759.i = mul nuw nsw i32 %t752.i, 7
  %t766.i = add nuw nsw i32 %t759.i, 13
  %t773.i = urem i32 %t766.i, 101
  %t780.i = add nsw i32 %t773.i, -50
  %t782.i = sitofp i32 %t780.i to float
  %t789.i = fmul float %t782.i, 1.562500e-02
  %t803.i = fadd float %o44.sroa.272.0157.i, %t789.i
  %t828.i = getelementptr float, ptr %o45.i, i64 %63
  store float %t803.i, ptr %t828.i, align 4
  %t850.cmp.i = icmp samesign ult i64 %62, 101
  %t850.i = select i1 %t850.cmp.i, i32 %67, i32 %t850.urem.i
  %t857.i = mul nuw nsw i32 %t850.i, 7
  %t864.i = add nuw nsw i32 %t857.i, 13
  %t871.i = urem i32 %t864.i, 101
  %t878.i = add nsw i32 %t871.i, -50
  %t880.i = sitofp i32 %t878.i to float
  %t887.i = fmul float %t880.i, 1.562500e-02
  %t901.i = fadd float %o44.sroa.275.0158.i, %t887.i
  %t926.i = getelementptr float, ptr %o45.i, i64 %62
  store float %t901.i, ptr %t926.i, align 8
  %t948.cmp.i = icmp samesign ult i64 %61, 101
  %t948.i = select i1 %t948.cmp.i, i32 %65, i32 %t948.urem.i
  %t955.i = mul nuw nsw i32 %t948.i, 7
  %t962.i = add nuw nsw i32 %t955.i, 13
  %t969.i = urem i32 %t962.i, 101
  %t976.i = add nsw i32 %t969.i, -50
  %t978.i = sitofp i32 %t976.i to float
  %t985.i = fmul float %t978.i, 1.562500e-02
  %t999.i = fadd float %o44.sroa.278.0159.i, %t985.i
  %t1024.i = getelementptr float, ptr %o45.i, i64 %61
  store float %t999.i, ptr %t1024.i, align 4
  %t1028.i = call i64 @mapal_gpu_alloc(ptr nonnull %d1027.i, i64 512)
  br i1 %s60.0161.i, label %bb1034.i, label %bb1035.i

bb1035.i:                                         ; preds = %bb62.i
  call void @mapal_gpu_h2d(i64 %t1028.i, ptr nonnull %o45.i, i64 512)
  br label %bb1036.i

bb1034.i:                                         ; preds = %bb62.i
  %t1037.i = shl nuw nsw i64 %indvars.iv.i, 5
  %t1038.i = add i64 %t1028.i, %t1037.i
  %t1039.i = getelementptr i8, ptr %o45.i, i64 %t1037.i
  call void @mapal_gpu_h2d(i64 %t1038.i, ptr %t1039.i, i64 32)
  br label %bb1036.i

bb1036.i:                                         ; preds = %bb1034.i, %bb1035.i
  %70 = load <4 x float>, ptr %o45.i, align 16
  %71 = load <4 x float>, ptr %o44.sroa.9.0.o45.sroa_idx.i, align 16
  %72 = load <4 x float>, ptr %o44.sroa.17.0.o45.sroa_idx.i, align 16
  %73 = load <4 x float>, ptr %o44.sroa.25.0.o45.sroa_idx.i, align 16
  %74 = load <4 x float>, ptr %o44.sroa.33.0.o45.sroa_idx.i, align 16
  %75 = load <4 x float>, ptr %o44.sroa.41.0.o45.sroa_idx.i, align 16
  %76 = load <4 x float>, ptr %o44.sroa.49.0.o45.sroa_idx.i, align 16
  %77 = load <4 x float>, ptr %o44.sroa.57.0.o45.sroa_idx.i, align 16
  %78 = load <4 x float>, ptr %o44.sroa.65.0.o45.sroa_idx.i, align 16
  %79 = load <4 x float>, ptr %o44.sroa.73.0.o45.sroa_idx.i, align 16
  %80 = load <4 x float>, ptr %o44.sroa.81.0.o45.sroa_idx.i, align 16
  %81 = load <4 x float>, ptr %o44.sroa.89.0.o45.sroa_idx.i, align 16
  %82 = load <4 x float>, ptr %o44.sroa.97.0.o45.sroa_idx.i, align 16
  %83 = load <4 x float>, ptr %o44.sroa.105.0.o45.sroa_idx.i, align 16
  %84 = load <4 x float>, ptr %o44.sroa.113.0.o45.sroa_idx.i, align 16
  %85 = load <4 x float>, ptr %o44.sroa.121.0.o45.sroa_idx.i, align 16
  %86 = load <4 x float>, ptr %o44.sroa.129.0.o45.sroa_idx.i, align 16
  %87 = load <4 x float>, ptr %o44.sroa.137.0.o45.sroa_idx.i, align 16
  %88 = load <4 x float>, ptr %o44.sroa.145.0.o45.sroa_idx.i, align 16
  %89 = load <4 x float>, ptr %o44.sroa.153.0.o45.sroa_idx.i, align 16
  %90 = load <4 x float>, ptr %o44.sroa.161.0.o45.sroa_idx.i, align 16
  %91 = load <4 x float>, ptr %o44.sroa.169.0.o45.sroa_idx.i, align 16
  %92 = load <4 x float>, ptr %o44.sroa.177.0.o45.sroa_idx.i, align 16
  %93 = load <4 x float>, ptr %o44.sroa.185.0.o45.sroa_idx.i, align 16
  %94 = load <4 x float>, ptr %o44.sroa.193.0.o45.sroa_idx.i, align 16
  %95 = load <4 x float>, ptr %o44.sroa.201.0.o45.sroa_idx.i, align 16
  %96 = load <4 x float>, ptr %o44.sroa.209.0.o45.sroa_idx.i, align 16
  %97 = load <4 x float>, ptr %o44.sroa.217.0.o45.sroa_idx.i, align 16
  %98 = load <4 x float>, ptr %o44.sroa.225.0.o45.sroa_idx.i, align 16
  %99 = load <4 x float>, ptr %o44.sroa.233.0.o45.sroa_idx.i, align 16
  %100 = load <4 x float>, ptr %o44.sroa.241.0.o45.sroa_idx.i, align 16
  %101 = load <4 x float>, ptr %o44.sroa.249.0.o45.sroa_idx.i, align 16
  %t1043.i = load i64, ptr %d1027.i, align 8
  %t1044.i = load i64, ptr %d6.i, align 8
  %t1046.i = call i64 @mapal_gpu_alloc(ptr nonnull %d1045.i, i64 512)
  call void @mapal_gpu_launch(i64 -6055507678851317309, i64 %t1043.i, i64 %t1044.i, i64 %t1046.i, i32 1, i32 1, i32 256, i32 0)
  %t1048.i = load i64, ptr %d1027.i, align 8
  %t1049.i = load i64, ptr %d8.i, align 8
  %t1051.i = call i64 @mapal_gpu_alloc(ptr nonnull %d1050.i, i64 512)
  call void @mapal_gpu_launch(i64 -6055507678851317309, i64 %t1048.i, i64 %t1049.i, i64 %t1051.i, i32 1, i32 1, i32 256, i32 0)
  %102 = urem <4 x i32> %46, splat (i32 17)
  %103 = trunc nuw nsw <4 x i32> %102 to <4 x i8>
  %104 = mul nuw nsw <4 x i8> %103, splat (i8 5)
  %105 = add nuw nsw <4 x i8> %104, splat (i8 3)
  %106 = urem <4 x i8> %105, splat (i8 17)
  %107 = add nsw <4 x i8> %106, splat (i8 -8)
  %108 = sitofp <4 x i8> %107 to <4 x float>
  %109 = fmul <4 x float> %108, splat (float 3.906250e-03)
  store <4 x float> %109, ptr %o271.i, align 16
  %110 = urem <4 x i32> %66, splat (i32 17)
  %111 = trunc nuw nsw <4 x i32> %110 to <4 x i8>
  %112 = mul nuw nsw <4 x i8> %111, splat (i8 5)
  %113 = add nuw nsw <4 x i8> %112, splat (i8 3)
  %114 = urem <4 x i8> %113, splat (i8 17)
  %115 = add nsw <4 x i8> %114, splat (i8 -8)
  %116 = sitofp <4 x i8> %115 to <4 x float>
  %117 = fmul <4 x float> %116, splat (float 3.906250e-03)
  store <4 x float> %117, ptr %t1067.4.i, align 16
  %t1070.i = call i64 @mapal_gpu_alloc(ptr nonnull %d1069.i, i64 32)
  call void @mapal_gpu_h2d(i64 %t1070.i, ptr nonnull %o271.i, i64 32)
  %t1072.i = load i64, ptr %d1069.i, align 8
  %t1073.i = load i64, ptr %d1045.i, align 8
  %t1075.i = call i64 @mapal_gpu_alloc(ptr nonnull %d1074.i, i64 64)
  store i64 %t1072.i, ptr %p1076.i, align 8
  store i64 %t1073.i, ptr %t1078.i, align 8
  store i64 0, ptr %t1079.i, align 8
  store i64 %t1075.i, ptr %t1080.i, align 8
  call void @mapal_gpu_launchv(i64 -7490935162623979227, i32 0, i32 4, ptr nonnull %p1076.i, i32 1, i32 1, i32 256, i32 0)
  %t1083.i = load i64, ptr %d1074.i, align 8
  %t1084.i = load i64, ptr %d1050.i, align 8
  %t1086.i = call i64 @mapal_gpu_alloc(ptr nonnull %d1085.i, i64 32)
  call void @mapal_gpu_launch(i64 -5971682293135266196, i64 %t1083.i, i64 %t1084.i, i64 %t1086.i, i32 1, i32 1, i32 256, i32 0)
  %t1088.i = load i64, ptr %d1085.i, align 8
  %t1089.i = load i64, ptr %d0.i, align 8
  %t1091.i = call i64 @mapal_gpu_alloc(ptr nonnull %d1090.i, i64 32)
  call void @mapal_gpu_launch(i64 -5746171913144188873, i64 %t1088.i, i64 %t1089.i, i64 %t1091.i, i32 1, i32 1, i32 256, i32 0)
  %t1094.i = load i64, ptr %d1090.i, align 8
  %t1095.i = load i64, ptr %d2.i, align 8
  %t1096.i = load i64, ptr %d10.i, align 8
  %t1098.i = call i64 @mapal_gpu_alloc(ptr nonnull %d1097.i, i64 64)
  store i64 %t1094.i, ptr %p1099.i, align 8
  store i64 %t1095.i, ptr %t1101.i, align 8
  store i64 0, ptr %t1102.i, align 8
  store i64 %t1096.i, ptr %t1103.i, align 8
  store i64 %t1098.i, ptr %t1104.i, align 8
  call void @mapal_gpu_launchv(i64 4644736598714039774, i32 0, i32 5, ptr nonnull %p1099.i, i32 1, i32 1, i32 256, i32 0)
  %t1107.i = load i64, ptr %d1097.i, align 8
  %t1108.i = load i64, ptr %d4.i, align 8
  %t1109.i = load i64, ptr %d12.i, align 8
  %t1110.i = load i64, ptr %d1090.i, align 8
  %t1112.i = call i64 @mapal_gpu_alloc(ptr nonnull %d1111.i, i64 32)
  store i64 %t1107.i, ptr %p1113.i, align 8
  store i64 %t1108.i, ptr %t1115.i, align 8
  store i64 0, ptr %t1116.i, align 8
  store i64 %t1109.i, ptr %t1117.i, align 8
  store i64 %t1110.i, ptr %t1118.i, align 8
  store i64 %t1112.i, ptr %t1119.i, align 8
  call void @mapal_gpu_launchv(i64 -4366650180424198423, i32 0, i32 6, ptr nonnull %p1113.i, i32 1, i32 1, i32 256, i32 0)
  call void @mapal_gpu_d2h(ptr nonnull %o289.i, i64 %t1112.i, i64 0, i64 32)
  %118 = load <4 x float>, ptr %o289.i, align 16
  %o292.sroa.134.512.copyload.i = load float, ptr %o292.sroa.134.512.o289.sroa_idx.i, align 16
  %o292.sroa.135.512.copyload.i = load float, ptr %o292.sroa.135.512.o289.sroa_idx.i, align 4
  %o292.sroa.136.512.copyload.i = load float, ptr %o292.sroa.136.512.o289.sroa_idx.i, align 8
  %o292.sroa.137.512.copyload.i = load float, ptr %o292.sroa.137.512.o289.sroa_idx.i, align 4
  %indvars.iv.next.i = add nuw nsw i64 %indvars.iv.i, 1
  %t1127.i = add nuw nsw i32 %storemerge160.i, 1
  %exitcond.not.i = icmp eq i64 %indvars.iv.next.i, 16
  br i1 %exitcond.not.i, label %mapal_main.exit, label %bb62.i

mapal_main.exit:                                  ; preds = %bb1036.i
  %119 = extractelement <4 x float> %118, i64 0
  call void @mapal_print_f32(float %119, i1 zeroext true)
  %120 = extractelement <4 x float> %118, i64 1
  call void @mapal_print_f32(float %120, i1 zeroext true)
  %121 = extractelement <4 x float> %118, i64 2
  call void @mapal_print_f32(float %121, i1 zeroext true)
  %122 = extractelement <4 x float> %118, i64 3
  call void @mapal_print_f32(float %122, i1 zeroext true)
  call void @mapal_print_f32(float %o292.sroa.134.512.copyload.i, i1 zeroext true)
  call void @mapal_print_f32(float %o292.sroa.135.512.copyload.i, i1 zeroext true)
  call void @mapal_print_f32(float %o292.sroa.136.512.copyload.i, i1 zeroext true)
  call void @mapal_print_f32(float %o292.sroa.137.512.copyload.i, i1 zeroext true)
  call void @mapal_gpu_free(ptr nonnull %d0.i)
  call void @mapal_gpu_free(ptr nonnull %d2.i)
  call void @mapal_gpu_free(ptr nonnull %d4.i)
  call void @mapal_gpu_free(ptr nonnull %d6.i)
  call void @mapal_gpu_free(ptr nonnull %d8.i)
  call void @mapal_gpu_free(ptr nonnull %d10.i)
  call void @mapal_gpu_free(ptr nonnull %d12.i)
  call void @mapal_gpu_free(ptr nonnull %d1027.i)
  call void @mapal_gpu_free(ptr nonnull %d1045.i)
  call void @mapal_gpu_free(ptr nonnull %d1050.i)
  call void @mapal_gpu_free(ptr nonnull %d1069.i)
  call void @mapal_gpu_free(ptr nonnull %d1074.i)
  call void @mapal_gpu_free(ptr nonnull %d1085.i)
  call void @mapal_gpu_free(ptr nonnull %d1090.i)
  call void @mapal_gpu_free(ptr nonnull %d1097.i)
  call void @mapal_gpu_free(ptr nonnull %d1111.i)
  call void @llvm.lifetime.end.p0(ptr nonnull %o45.i)
  call void @llvm.lifetime.end.p0(ptr nonnull %o271.i)
  call void @llvm.lifetime.end.p0(ptr nonnull %o289.i)
  call void @llvm.lifetime.end.p0(ptr nonnull %d0.i)
  call void @llvm.lifetime.end.p0(ptr nonnull %d2.i)
  call void @llvm.lifetime.end.p0(ptr nonnull %d4.i)
  call void @llvm.lifetime.end.p0(ptr nonnull %d6.i)
  call void @llvm.lifetime.end.p0(ptr nonnull %d8.i)
  call void @llvm.lifetime.end.p0(ptr nonnull %d10.i)
  call void @llvm.lifetime.end.p0(ptr nonnull %d12.i)
  call void @llvm.lifetime.end.p0(ptr nonnull %d1027.i)
  call void @llvm.lifetime.end.p0(ptr nonnull %d1045.i)
  call void @llvm.lifetime.end.p0(ptr nonnull %d1050.i)
  call void @llvm.lifetime.end.p0(ptr nonnull %d1069.i)
  call void @llvm.lifetime.end.p0(ptr nonnull %d1074.i)
  call void @llvm.lifetime.end.p0(ptr nonnull %p1076.i)
  call void @llvm.lifetime.end.p0(ptr nonnull %d1085.i)
  call void @llvm.lifetime.end.p0(ptr nonnull %d1090.i)
  call void @llvm.lifetime.end.p0(ptr nonnull %d1097.i)
  call void @llvm.lifetime.end.p0(ptr nonnull %p1099.i)
  call void @llvm.lifetime.end.p0(ptr nonnull %d1111.i)
  call void @llvm.lifetime.end.p0(ptr nonnull %p1113.i)
  ret i32 0
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #0

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #0

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
