## rtx-4090
### attn_1024_64
mapal: 9 launches, 9 distinct, median gap 28.4 us
  mapal_tile_7e5df615868c20ca x1 med 23.7 us grid 16x16x1 blk 256x1 regs 69 smem 38.9 KB
  mapal_tile_cc5bd59a6b05b4c5 x1 med 19.2 us grid 32x1x1 blk 32x1 regs 52 smem 0.0 KB
  mapal_tile_ebb9a4363605b99f x1 med 17.2 us grid 32x1x1 blk 32x1 regs 78 smem 0.0 KB
lib: 78 launches, 9 distinct, median gap 0.7 us
  ampere_sgemm_32x32_sliced1x4_nn x12 med 9.8 us grid 2x32x4 blk 128x1 regs 86 smem 33.8 KB
  ampere_sgemm_32x128_tn x12 med 8.5 us grid 32x8x1 blk 256x1 regs 57 smem 16.4 KB
  softmax_row(float *, const float *, int, float) x6 med 8.6 us grid 128x1x1 blk 256x1 regs 39 smem 0.0 KB
torch: 1803 launches, 22 distinct, median gap 58.8 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x1030 med 68.4 us grid 18432x1x1 blk 128x1 regs 16 smem 0.0 KB
  triton_per_fused_add_clamp_div_max_mul_sub_sum_0 x539 med 4.3 us grid 1024x1x1 blk 256x1 regs 19 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x6 med 260.2 us grid 65536x1x1 blk 128x1 regs 16 smem 0.0 KB
### causal_1024_256
mapal: 2050 launches, 4 distinct, median gap 321.5 us
  mapal_tile_fee5d054184d5db4 x2046 med 4.5 us grid 1x1x1 blk 256x1 regs 38 smem 0.0 KB
  mapal_tile_939b71f99698b7c0 x2 med 7.6 us grid 8x16x1 blk 128x1 regs 80 smem 81.9 KB
  mapal_gen_c0626337c402c009 x1 med 1.4 us grid 256x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: no trace
torch: 354529 launches, 32 distinct, median gap 17.9 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x8854 med 77.1 us grid 18432x1x1 blk 128x1 regs 16 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x73728 med 1.0 us grid 1x1x1 blk 128x1 regs 38 smem 0.0 KB
  std::enable_if<!T7, void>::type internal::gemvx::kernel<int, x18432 med 1.9 us grid 16x1x1 blk 16x32 regs 128 smem 2.0 KB
### convmc_8_64_56_3
mapal: no trace
lib: no trace
torch: 11168 launches, 42 distinct, median gap 3.2 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x7210 med 77.1 us grid 18432x1x1 blk 128x1 regs 16 smem 0.0 KB
  triton_poi_fused_view_2 x1175 med 23.3 us grid 1568x1x1 blk 128x1 regs 25 smem 0.0 KB
  triton_poi_fused_convolution_0 x1281 med 7.4 us grid 53x8x1 blk 256x1 regs 33 smem 16.4 KB
### decode_1024_256
mapal: 4098 launches, 6 distinct, median gap 14.4 us
  mapal_tile_fee5d054184d5db4 x2046 med 4.5 us grid 1x1x1 blk 256x1 regs 38 smem 0.0 KB
  mapal_tile_55fab8ffe34e679c x1024 med 5.4 us grid 1x512x1 blk 256x1 regs 16 smem 0.0 KB
  mapal_tile_31e028a1fa0dc070 x1024 med 3.6 us grid 4x128x1 blk 256x1 regs 16 smem 0.0 KB
lib: 49152 launches, 6 distinct, median gap 5.3 us
  ampere_sgemm_32x32_sliced1x4_nn x8192 med 7.1 us grid 8x32x1 blk 128x1 regs 86 smem 33.8 KB
  std::enable_if<!T7, void>::type internal::gemvx::kernel<int, x8192 med 2.1 us grid 32x1x1 blk 8x32 regs 165 smem 1.0 KB
  std::enable_if<!T7, void>::type internal::gemvx::kernel<int, x8192 med 1.9 us grid 64x1x1 blk 16x16 regs 159 smem 1.0 KB
torch: 148976 launches, 23 distinct, median gap 20.9 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x7004 med 77.1 us grid 18432x1x1 blk 128x1 regs 16 smem 0.0 KB
  std::enable_if<!T7, void>::type internal::gemvx::kernel<int, x18432 med 1.9 us grid 16x1x1 blk 16x32 regs 128 smem 2.0 KB
  void at::native::<unnamed>::indexSelectSmallIndex<float, lon x18432 med 1.6 us grid 2x1x1 blk 128x1 regs 32 smem 0.0 KB
### diamond_2048
mapal: 7 launches, 6 distinct, median gap 21.2 us
  mapal_tile_59027ed417ac8177 x2 med 300.0 us grid 16x16x1 blk 128x1 regs 218 smem 17.4 KB
  mapal_tile_60426c42be5d8b82 x1 med 312.9 us grid 16x16x1 blk 128x1 regs 224 smem 17.4 KB
  mapal_gen_0c2c9285a1cff51b x1 med 11.0 us grid 16384x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: 93 launches, 4 distinct, median gap 2.2 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x54 med 306.2 us grid 128x1x4 blk 256x1 regs 202 smem 50.2 KB
  void cutlass::Kernel<cutlass::gemm::kernel::Gemm<cutlass::ge x18 med 306.5 us grid 16x16x1 blk 256x1 regs 122 smem 17.4 KB
  cell(int, float *, const float *, const float *, long long,  x12 med 18.3 us grid 16384x1x1 blk 256x1 regs 22 smem 0.0 KB
torch: 115 launches, 12 distinct, median gap 2.2 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x54 med 305.2 us grid 128x1x4 blk 256x1 regs 202 smem 50.2 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x6 med 262.3 us grid 65536x1x1 blk 128x1 regs 16 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x15 med 32.0 us grid 4096x1x1 blk 128x1 regs 36 smem 0.0 KB
### ew_4096
mapal: 1 launches, 1 distinct, median gap 0.0 us
  mapal_tile_2728d927486a4a1e x1 med 51.6 us grid 65536x1x1 blk 256x1 regs 17 smem 0.0 KB
lib: 33 launches, 2 distinct, median gap 0.7 us
  cell(int, float *, const float *, const float *, long long,  x24 med 167.8 us grid 65536x1x1 blk 256x1 regs 22 smem 0.0 KB
  gen_op(float *, long long, int, int, int, int, float) x9 med 67.5 us grid 65536x1x1 blk 256x1 regs 18 smem 0.0 KB
torch: 88 launches, 14 distinct, median gap 0.7 us
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x15 med 292.7 us grid 16384x1x1 blk 128x1 regs 36 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x9 med 220.8 us grid 16384x1x1 blk 128x1 regs 38 smem 0.0 KB
  triton_poi_fused_add_mul_0 x9 med 216.6 us grid 32768x1x1 blk 128x1 regs 22 smem 0.0 KB
### fanout_2048
mapal: 5 launches, 5 distinct, median gap 25.1 us
  mapal_tile_4ea5bf1ce04a09bd x1 med 293.5 us grid 16x16x1 blk 128x1 regs 218 smem 17.4 KB
  mapal_tile_59027ed417ac8177 x1 med 292.8 us grid 16x16x1 blk 128x1 regs 218 smem 17.4 KB
  mapal_gen_0c2c9285a1cff51b x1 med 11.0 us grid 16384x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: 45 launches, 3 distinct, median gap 2.3 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x24 med 308.7 us grid 128x1x4 blk 256x1 regs 202 smem 50.2 KB
  cell(int, float *, const float *, const float *, long long,  x12 med 11.2 us grid 16384x1x1 blk 256x1 regs 22 smem 0.0 KB
  gen_op(float *, long long, int, int, int, int, float) x9 med 12.3 us grid 16384x1x1 blk 256x1 regs 18 smem 0.0 KB
torch: 118 launches, 14 distinct, median gap 2.1 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x27 med 306.2 us grid 128x1x4 blk 256x1 regs 202 smem 50.2 KB
  triton_tem_fused_add_mm_mul_0 x9 med 327.0 us grid 512x1x1 blk 128x1 regs 168 smem 50.2 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x6 med 259.9 us grid 65536x1x1 blk 128x1 regs 16 smem 0.0 KB
### firew_rowsum_2048x2048_16
mapal: 4 launches, 4 distinct, median gap 41.2 us
  mapal_tile_65631a69f617c224 x1 med 33.5 us grid 64x1x1 blk 32x1 regs 78 smem 0.0 KB
  mapal_tile_c4cf34b620b8f0ab x1 med 29.3 us grid 16384x1x1 blk 256x1 regs 40 smem 0.0 KB
  mapal_gen_8bd171c603b5edba x1 med 10.9 us grid 16385x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: no trace
torch: 1984 launches, 27 distinct, median gap 3.1 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x1236 med 69.6 us grid 18432x1x1 blk 128x1 regs 16 smem 0.0 KB
  triton_red_fused_mul_relu_sum_view_0 x645 med 27.5 us grid 1024x1x1 blk 512x1 regs 18 smem 0.0 KB
  conv1D_NCHW_general x2 med 4806.9 us grid 1x1x1 blk 256x1 regs 30 smem 0.0 KB
### loopmm_1024_8
mapal: 10 launches, 3 distinct, median gap 0.6 us
  mapal_tile_565ffde502536326 x8 med 57.4 us grid 16x8x1 blk 256x1 regs 95 smem 78.8 KB
  mapal_gen_7a2ade86c804ef09 x1 med 3.8 us grid 4096x1x1 blk 256x1 regs 16 smem 0.0 KB
  mapal_gen_1aeae5b5ba14d87e x1 med 3.4 us grid 4096x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: no trace
torch: 172 launches, 12 distinct, median gap 0.7 us
  triton_tem_fused_mm_0 x72 med 47.9 us grid 128x1x1 blk 256x1 regs 108 smem 75.8 KB
  void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_al x72 med 42.7 us grid 128x1x1 blk 128x1 regs 126 smem 31.7 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x6 med 260.1 us grid 65536x1x1 blk 128x1 regs 16 smem 0.0 KB
### mlp2_2048
mapal: 12 launches, 11 distinct, median gap 8.8 us
  mapal_tile_59027ed417ac8177 x2 med 290.9 us grid 16x16x1 blk 128x1 regs 218 smem 17.4 KB
  mapal_tile_cb8a431d29c1dd2d x1 med 318.8 us grid 16x16x1 blk 128x1 regs 222 smem 17.4 KB
  mapal_tile_4a874e6551089bac x1 med 297.7 us grid 16x16x1 blk 128x1 regs 222 smem 17.4 KB
lib: 153 launches, 5 distinct, median gap 2.1 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x96 med 308.0 us grid 128x1x4 blk 256x1 regs 202 smem 50.2 KB
  void cutlass::Kernel<cutlass::gemm::kernel::Gemm<cutlass::ge x12 med 305.2 us grid 16x16x1 blk 256x1 regs 122 smem 17.4 KB
  void cutlass::Kernel<cutlass::gemm::kernel::Gemm<cutlass::ge x12 med 302.7 us grid 16x16x1 blk 256x1 regs 122 smem 17.4 KB
torch: 40 launches, 6 distinct, median gap 0.7 us
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x15 med 32.5 us grid 4096x1x1 blk 128x1 regs 36 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x5 med 32.6 us grid 4096x1x1 blk 128x1 regs 40 smem 0.0 KB
  void <unnamed>::elementwise_kernel_with_index<int, at::nativ x5 med 32.5 us grid 65536x1x1 blk 64x1 regs 16 smem 0.0 KB
### mlp_2048
mapal: 5 launches, 5 distinct, median gap 27.7 us
  mapal_tile_17e3127bb006f8f8 x1 med 305.7 us grid 16x16x1 blk 128x1 regs 224 smem 17.4 KB
  mapal_tile_59027ed417ac8177 x1 med 287.8 us grid 16x16x1 blk 128x1 regs 218 smem 17.4 KB
  mapal_gen_0c2c9285a1cff51b x1 med 11.0 us grid 16384x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: 51 launches, 4 distinct, median gap 2.2 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x24 med 305.8 us grid 128x1x4 blk 256x1 regs 202 smem 50.2 KB
  void cutlass::Kernel<cutlass::gemm::kernel::Gemm<cutlass::ge x12 med 302.9 us grid 16x16x1 blk 256x1 regs 122 smem 17.4 KB
  gen_op(float *, long long, int, int, int, int, float) x9 med 12.2 us grid 16384x1x1 blk 256x1 regs 18 smem 0.0 KB
torch: 40 launches, 6 distinct, median gap 0.7 us
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x15 med 31.6 us grid 4096x1x1 blk 128x1 regs 36 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x5 med 33.6 us grid 4096x1x1 blk 128x1 regs 40 smem 0.0 KB
  void <unnamed>::elementwise_kernel_with_index<int, at::nativ x5 med 32.5 us grid 65536x1x1 blk 64x1 regs 16 smem 0.0 KB
### pj_m16_n2048_k7168
mapal: 3 launches, 3 distinct, median gap 90.2 us
  mapal_tile_6e4b9bcdd0843cf4 x1 med 68.2 us grid 64x2x1 blk 64x1 regs 251 smem 46.1 KB
  mapal_gen_8e6c109a3b29db10 x1 med 34.8 us grid 57344x1x1 blk 256x1 regs 16 smem 0.0 KB
  mapal_gen_606fb06d76aabeb6 x1 med 1.6 us grid 448x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: no trace
torch: 55 launches, 13 distinct, median gap 3.4 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x6 med 253.1 us grid 65536x1x1 blk 128x1 regs 16 smem 0.0 KB
  triton_tem_fused_mm_0 x9 med 84.4 us grid 32x1x1 blk 128x1 regs 48 smem 42.0 KB
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x6 med 127.1 us grid 112x1x1 blk 128x1 regs 36 smem 0.0 KB
### prefill_1024_512_256
mapal: 1028 launches, 4 distinct, median gap 317.3 us
  mapal_tile_fee5d054184d5db4 x1022 med 4.5 us grid 1x1x1 blk 256x1 regs 38 smem 0.0 KB
  mapal_tile_939b71f99698b7c0 x4 med 7.6 us grid 8x16x1 blk 128x1 regs 80 smem 81.9 KB
  mapal_gen_c0626337c402c009 x1 med 1.4 us grid 256x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: no trace
torch: 172071 launches, 36 distinct, median gap 18.1 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x618 med 77.6 us grid 18432x1x1 blk 128x1 regs 16 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x37008 med 1.0 us grid 512x1x1 blk 128x1 regs 38 smem 0.0 KB
  std::enable_if<!T7, void>::type internal::gemvx::kernel<int, x9216 med 1.9 us grid 16x1x1 blk 16x32 regs 128 smem 2.0 KB
### rowsum_4096
mapal: 4 launches, 4 distinct, median gap 5.1 us
  mapal_tile_86cfda25b6d53a54 x1 med 2388.4 us grid 32x32x1 blk 128x1 regs 242 smem 17.4 KB
  mapal_tile_c63583fc9e8c05c9 x1 med 67.7 us grid 128x1x1 blk 32x1 regs 78 smem 0.0 KB
  mapal_gen_ee8f5451f4dd91de x1 med 62.4 us grid 65536x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: 33 launches, 4 distinct, median gap 3.0 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x12 med 2338.3 us grid 256x2x3 blk 256x1 regs 202 smem 50.2 KB
  gen_op(float *, long long, int, int, int, int, float) x9 med 67.1 us grid 65536x1x1 blk 256x1 regs 18 smem 0.0 KB
  row_sum(float *, const float *, int) x6 med 59.6 us grid 512x1x1 blk 256x1 regs 39 smem 0.0 KB
torch: 1960 launches, 14 distinct, median gap 3.1 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x1236 med 59.6 us grid 18432x1x1 blk 128x1 regs 16 smem 0.0 KB
  triton_red_fused_sum_0 x645 med 92.8 us grid 4096x1x1 blk 512x1 regs 18 smem 0.0 KB
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x18 med 2331.3 us grid 256x2x3 blk 256x1 regs 202 smem 50.2 KB
### rt_mlp2_2048
mapal: 38 launches, 10 distinct, median gap 0.7 us
  mapal_tile_cb8a431d29c1dd2d x8 med 312.6 us grid 16x16x1 blk 128x1 regs 222 smem 17.4 KB
  mapal_tile_c56cd5d3a2524dfa x8 med 313.0 us grid 16x16x1 blk 128x1 regs 224 smem 17.4 KB
  mapal_tile_59027ed417ac8177 x8 med 295.9 us grid 16x16x1 blk 128x1 regs 218 smem 17.4 KB
lib: no trace
torch: 34029 launches, 35 distinct, median gap 3.0 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x20806 med 77.3 us grid 18432x1x1 blk 128x1 regs 16 smem 0.0 KB
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x576 med 328.5 us grid 128x1x4 blk 256x1 regs 202 smem 50.2 KB
  triton_poi_fused__to_copy_add_arange_div_mul_remainder_sub_2 x1175 med 15.3 us grid 4096x1x1 blk 128x1 regs 32 smem 0.0 KB
### smlp_4096_256_1024
mapal: 6 launches, 6 distinct, median gap 41.5 us
  mapal_tile_a1d600c3510aec4a x1 med 57.8 us grid 4x32x1 blk 256x1 regs 95 smem 78.8 KB
  mapal_tile_52fc2d71439d6096 x1 med 44.0 us grid 8x32x1 blk 128x1 regs 222 smem 17.4 KB
  mapal_gen_7a2ade86c804ef09 x1 med 3.9 us grid 4096x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: 60 launches, 7 distinct, median gap 0.7 us
  ampere_sgemm_128x64_nn x12 med 51.5 us grid 8x64x1 blk 128x1 regs 122 smem 13.3 KB
  void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_al x12 med 43.9 us grid 128x1x1 blk 128x1 regs 126 smem 31.7 KB
  void cutlass::Kernel<cutlass::gemm::kernel::Gemm<cutlass::ge x6 med 79.3 us grid 32x2x1 blk 256x1 regs 122 smem 17.4 KB
torch: 95 launches, 16 distinct, median gap 4.7 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x6 med 262.7 us grid 65536x1x1 blk 128x1 regs 16 smem 0.0 KB
  triton_tem_fused_add_mm_relu_0 x9 med 49.4 us grid 512x1x1 blk 128x1 regs 168 smem 50.2 KB
  ampere_sgemm_128x64_nn x9 med 48.6 us grid 8x64x1 blk 128x1 regs 122 smem 13.3 KB
### softmax_rows_4096
mapal: 7 launches, 7 distinct, median gap 3.8 us
  mapal_tile_ef9b0c1ddb04347e x1 med 2356.0 us grid 32x32x1 blk 128x1 regs 218 smem 17.4 KB
  mapal_tile_eb3955d2a8e3de3d x1 med 131.5 us grid 128x1x1 blk 32x1 regs 52 smem 0.0 KB
  mapal_tile_db1dbea1bcb161f7 x1 med 113.9 us grid 128x1x1 blk 32x1 regs 78 smem 0.0 KB
lib: 51 launches, 7 distinct, median gap 0.7 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x12 med 2333.9 us grid 256x2x3 blk 256x1 regs 202 smem 50.2 KB
  softmax_row(float *, const float *, int, float) x6 med 217.7 us grid 512x1x1 blk 256x1 regs 39 smem 0.0 KB
  sm_exp(float *, const float *, const float *, long long, int x6 med 111.8 us grid 65536x1x1 blk 256x1 regs 18 smem 0.0 KB
torch: 223 launches, 19 distinct, median gap 0.7 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x18 med 2331.9 us grid 256x2x3 blk 256x1 regs 202 smem 50.2 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x72 med 146.5 us grid 16384x1x1 blk 128x1 regs 38 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x15 med 294.2 us grid 16384x1x1 blk 128x1 regs 36 smem 0.0 KB
### twin_2048
mapal: 6 launches, 5 distinct, median gap 19.9 us
  mapal_tile_59027ed417ac8177 x2 med 285.0 us grid 16x16x1 blk 128x1 regs 218 smem 17.4 KB
  mapal_gen_0c2c9285a1cff51b x1 med 11.0 us grid 16384x1x1 blk 256x1 regs 16 smem 0.0 KB
  mapal_gen_d62743ecf7801ae8 x1 med 11.0 us grid 16384x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: no trace
torch: 94 launches, 11 distinct, median gap 2.2 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x36 med 306.9 us grid 128x1x4 blk 256x1 regs 202 smem 50.2 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x6 med 259.1 us grid 65536x1x1 blk 128x1 regs 16 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x15 med 32.0 us grid 4096x1x1 blk 128x1 regs 36 smem 0.0 KB
### epi_4096
mapal: 3 launches, 3 distinct, median gap 81.2 us
  mapal_tile_040047611af8cafc x1 med 2374.8 us grid 32x32x1 blk 128x1 regs 226 smem 17.4 KB
  mapal_gen_ee8f5451f4dd91de x1 med 62.7 us grid 65536x1x1 blk 256x1 regs 16 smem 0.0 KB
  mapal_gen_25d49d3d39dbf03f x1 med 39.9 us grid 65536x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: 33 launches, 4 distinct, median gap 33.1 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x12 med 2337.6 us grid 256x2x3 blk 256x1 regs 202 smem 50.2 KB
  void cutlass::Kernel<cutlass::gemm::kernel::Gemm<cutlass::ge x6 med 2428.9 us grid 32x32x1 blk 256x1 regs 122 smem 17.4 KB
  cell(int, float *, const float *, const float *, long long,  x6 med 119.1 us grid 65536x1x1 blk 256x1 regs 22 smem 0.0 KB
torch: 97 launches, 14 distinct, median gap 0.7 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x18 med 2329.4 us grid 256x2x3 blk 256x1 regs 202 smem 50.2 KB
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x15 med 294.2 us grid 16384x1x1 blk 128x1 regs 36 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x6 med 275.3 us grid 65536x1x1 blk 128x1 regs 16 smem 0.0 KB
## rtx-5090
### attn_1024_64
mapal: 9 launches, 9 distinct, median gap 40.0 us
  mapal_tile_2c8b55a0ea082eb4 x1 med 18.0 us grid 32x1x1 blk 32x1 regs 46 smem 0.0 KB
  mapal_tile_a0a6a43f2907890e x1 med 15.8 us grid 32x1x1 blk 32x1 regs 78 smem 0.0 KB
  mapal_tile_ecbd63feb16cc161 x1 med 15.1 us grid 4x32x1 blk 32x1 regs 192 smem 13.3 KB
lib: 78 launches, 9 distinct, median gap 5.3 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_128x32_8x5_tn_al x12 med 8.6 us grid 256x1x1 blk 128x1 regs 80 smem 26.6 KB
  void cutlass::Kernel2<cutlass_80_simt_sgemm_64x64_8x5_nn_ali x12 med 6.0 us grid 16x1x8 blk 128x1 regs 78 smem 20.5 KB
  softmax_row(float *, const float *, int, float) x6 med 7.4 us grid 128x1x1 blk 256x1 regs 38 smem 0.0 KB
torch: 1339 launches, 23 distinct, median gap 101.6 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x748 med 56.4 us grid 24576x1x1 blk 128x1 regs 16 smem 0.0 KB
  triton_per_fused_add_clamp_div_max_mul_sub_sum_1 x357 med 2.9 us grid 1024x1x1 blk 256x1 regs 18 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x6 med 142.0 us grid 65536x1x1 blk 128x1 regs 16 smem 0.0 KB
### causal_1024_256
mapal: 2050 launches, 4 distinct, median gap 76.5 us
  mapal_tile_e2f8ae468c059815 x2046 med 4.3 us grid 1x1x1 blk 256x1 regs 38 smem 0.0 KB
  mapal_tile_022eacf5c2c33e41 x2 med 7.7 us grid 8x32x1 blk 64x1 regs 197 smem 35.8 KB
  mapal_gen_2091e33e430539f8 x1 med 1.2 us grid 256x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: no trace
torch: 348243 launches, 32 distinct, median gap 23.2 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x4711 med 56.3 us grid 24576x1x1 blk 128x1 regs 16 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x73728 med 0.9 us grid 1x1x1 blk 128x1 regs 40 smem 0.0 KB
  void gemvNSP_kernel<float, float, float, float, (int)1, (int x18432 med 2.1 us grid 8x1x1 blk 32x24 regs 28 smem 3.1 KB
### convmc_8_64_56_3
mapal: no trace
lib: no trace
torch: 2325 launches, 36 distinct, median gap 25.3 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x1442 med 56.9 us grid 24576x1x1 blk 128x1 regs 16 smem 0.0 KB
  void cudnn::cnn::conv2d_grouped_direct_kernel<(bool)0, (bool x2 med 676.6 us grid 1568x1x1 blk 1024x1 regs 56 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x6 med 143.3 us grid 65536x1x1 blk 128x1 regs 16 smem 0.0 KB
### decode_1024_256
mapal: 4098 launches, 6 distinct, median gap 15.6 us
  mapal_tile_e2f8ae468c059815 x2046 med 4.2 us grid 1x1x1 blk 256x1 regs 38 smem 0.0 KB
  mapal_tile_a0714d3d20b8101f x1024 med 4.9 us grid 1x512x1 blk 256x1 regs 16 smem 0.0 KB
  mapal_tile_647b5e2c7fca3543 x1024 med 2.8 us grid 4x128x1 blk 256x1 regs 16 smem 0.0 KB
lib: 57344 launches, 7 distinct, median gap 10.2 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_al x8192 med 5.1 us grid 32x1x4 blk 128x1 regs 128 smem 31.7 KB
  std::enable_if<!T7, void>::type internal::gemvx::kernel<int, x8192 med 2.3 us grid 32x1x1 blk 8x32 regs 230 smem 1.0 KB
  void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, fl x8192 med 2.3 us grid 8x64x1 blk 32x16 regs 46 smem 0.0 KB
torch: 147148 launches, 23 distinct, median gap 27.3 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x5840 med 56.2 us grid 24576x1x1 blk 128x1 regs 16 smem 0.0 KB
  void gemvNSP_kernel<float, float, float, float, (int)1, (int x18432 med 2.1 us grid 8x1x1 blk 32x24 regs 28 smem 3.1 KB
  void at::native::index_elementwise_kernel<(int)128, (int)4,  x18432 med 1.4 us grid 1x1x1 blk 128x1 regs 32 smem 0.0 KB
### diamond_2048
mapal: 7 launches, 6 distinct, median gap 36.7 us
  mapal_tile_c59d5a05a86bc84f x2 med 276.3 us grid 338x1x1 blk 256x1 regs 126 smem 17.4 KB
  mapal_tile_cbb11ec5aec4ae21 x1 med 286.9 us grid 16x16x1 blk 256x1 regs 124 smem 17.4 KB
  mapal_gen_2f1707000731833f x1 med 8.8 us grid 16384x1x1 blk 256x1 regs 17 smem 0.0 KB
lib: 93 launches, 4 distinct, median gap 2.9 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x54 med 256.3 us grid 128x1x5 blk 256x1 regs 210 smem 50.2 KB
  void cutlass::Kernel<cutlass::gemm::kernel::Gemm<cutlass::ge x18 med 288.7 us grid 16x16x1 blk 256x1 regs 124 smem 17.4 KB
  cell(int, float *, const float *, const float *, long long,  x12 med 10.0 us grid 16384x1x1 blk 256x1 regs 24 smem 0.0 KB
torch: 115 launches, 12 distinct, median gap 5.0 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x54 med 254.6 us grid 128x1x5 blk 256x1 regs 210 smem 50.2 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x6 med 143.9 us grid 65536x1x1 blk 128x1 regs 16 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x15 med 13.2 us grid 4096x1x1 blk 128x1 regs 40 smem 0.0 KB
### ew_4096
mapal: 1 launches, 1 distinct, median gap 0.0 us
  mapal_tile_7088d91e53f6e62f x1 med 33.2 us grid 65536x1x1 blk 256x1 regs 17 smem 0.0 KB
lib: 33 launches, 2 distinct, median gap 2.8 us
  cell(int, float *, const float *, const float *, long long,  x24 med 90.6 us grid 65536x1x1 blk 256x1 regs 24 smem 0.0 KB
  gen_op(float *, long long, int, int, int, int, float) x9 med 32.4 us grid 65536x1x1 blk 256x1 regs 18 smem 0.0 KB
torch: 1648 launches, 15 distinct, median gap 3.1 us
  triton_poi_fused_add_mul_0 x539 med 129.0 us grid 16384x1x1 blk 128x1 regs 30 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x1030 med 56.7 us grid 24576x1x1 blk 128x1 regs 16 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x15 med 176.3 us grid 16384x1x1 blk 128x1 regs 40 smem 0.0 KB
### fanout_2048
mapal: 5 launches, 5 distinct, median gap 44.6 us
  mapal_tile_e637050b79563630 x1 med 311.5 us grid 16x16x1 blk 256x1 regs 128 smem 17.4 KB
  mapal_tile_c59d5a05a86bc84f x1 med 264.9 us grid 338x1x1 blk 256x1 regs 126 smem 17.4 KB
  mapal_gen_6f89aa19c188a660 x1 med 8.7 us grid 16384x1x1 blk 256x1 regs 17 smem 0.0 KB
lib: 45 launches, 3 distinct, median gap 3.1 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x24 med 255.1 us grid 128x1x5 blk 256x1 regs 210 smem 50.2 KB
  cell(int, float *, const float *, const float *, long long,  x12 med 9.9 us grid 16384x1x1 blk 256x1 regs 24 smem 0.0 KB
  gen_op(float *, long long, int, int, int, int, float) x9 med 8.9 us grid 16384x1x1 blk 256x1 regs 18 smem 0.0 KB
torch: 127 launches, 14 distinct, median gap 7.5 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x36 med 254.8 us grid 128x1x5 blk 256x1 regs 210 smem 50.2 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x6 med 143.1 us grid 65536x1x1 blk 128x1 regs 16 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x15 med 13.2 us grid 4096x1x1 blk 128x1 regs 40 smem 0.0 KB
### firew_rowsum_2048x2048_16
mapal: 4 launches, 4 distinct, median gap 34.5 us
  mapal_tile_b0769a610316cc35 x1 med 30.6 us grid 64x1x1 blk 32x1 regs 78 smem 0.0 KB
  mapal_tile_c0dee36444ab4aec x1 med 19.8 us grid 16384x1x1 blk 256x1 regs 40 smem 0.0 KB
  mapal_gen_eb3a91bdacfb962b x1 med 8.6 us grid 16385x1x1 blk 256x1 regs 17 smem 0.0 KB
lib: no trace
torch: 1976 launches, 24 distinct, median gap 32.9 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x1236 med 50.7 us grid 24576x1x1 blk 128x1 regs 16 smem 0.0 KB
  triton_red_fused_mul_relu_sum_view_0 x645 med 12.8 us grid 2048x1x1 blk 512x1 regs 16 smem 0.0 KB
  void implicit_convolve_sgemm<float, float, (int)512, (int)6, x2 med 845.7 us grid 65536x1x1 blk 8x32 regs 128 smem 10.2 KB
### loopmm_1024_8
mapal: 10 launches, 3 distinct, median gap 2.3 us
  mapal_tile_b3430eaa07e662e7 x8 med 54.4 us grid 16x16x1 blk 256x1 regs 79 smem 8.2 KB
  mapal_gen_da5a5e8d470768f8 x1 med 3.0 us grid 4096x1x1 blk 256x1 regs 17 smem 0.0 KB
  mapal_gen_65fe65acc713e28f x1 med 2.8 us grid 4096x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: no trace
torch: 172 launches, 12 distinct, median gap 3.0 us
  triton_tem_fused_mm_0 x72 med 42.7 us grid 128x1x1 blk 128x1 regs 168 smem 50.2 KB
  void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_al x72 med 41.7 us grid 128x1x5 blk 128x1 regs 128 smem 31.7 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x6 med 142.4 us grid 65536x1x1 blk 128x1 regs 16 smem 0.0 KB
### mlp2_2048
mapal: 12 launches, 11 distinct, median gap 16.7 us
  mapal_tile_c59d5a05a86bc84f x2 med 261.2 us grid 338x1x1 blk 256x1 regs 126 smem 17.4 KB
  mapal_tile_997700f4fb01290b x1 med 315.2 us grid 16x16x1 blk 256x1 regs 125 smem 17.4 KB
  mapal_tile_d2d7996f7d9568ae x1 med 298.4 us grid 16x16x1 blk 256x1 regs 125 smem 17.4 KB
lib: 153 launches, 5 distinct, median gap 2.6 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x96 med 258.5 us grid 128x1x5 blk 256x1 regs 210 smem 50.2 KB
  void cutlass::Kernel<cutlass::gemm::kernel::Gemm<cutlass::ge x12 med 288.2 us grid 16x16x1 blk 256x1 regs 124 smem 17.4 KB
  void cutlass::Kernel<cutlass::gemm::kernel::Gemm<cutlass::ge x12 med 283.7 us grid 16x16x1 blk 256x1 regs 124 smem 17.4 KB
torch: 40 launches, 6 distinct, median gap 28.0 us
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x15 med 13.2 us grid 4096x1x1 blk 128x1 regs 40 smem 0.0 KB
  void <unnamed>::elementwise_kernel_with_index<int, at::nativ x5 med 31.5 us grid 65536x1x1 blk 64x1 regs 16 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x5 med 13.4 us grid 4096x1x1 blk 128x1 regs 40 smem 0.0 KB
### mlp_2048
mapal: 5 launches, 5 distinct, median gap 21.9 us
  mapal_tile_78b03e0e414beaef x1 med 302.9 us grid 16x16x1 blk 256x1 regs 124 smem 17.4 KB
  mapal_tile_c59d5a05a86bc84f x1 med 265.3 us grid 338x1x1 blk 256x1 regs 126 smem 17.4 KB
  mapal_gen_acc3728df88a4caa x1 med 9.2 us grid 16384x1x1 blk 256x1 regs 17 smem 0.0 KB
lib: 51 launches, 4 distinct, median gap 3.1 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x24 med 255.1 us grid 128x1x5 blk 256x1 regs 210 smem 50.2 KB
  void cutlass::Kernel<cutlass::gemm::kernel::Gemm<cutlass::ge x12 med 288.6 us grid 16x16x1 blk 256x1 regs 124 smem 17.4 KB
  gen_op(float *, long long, int, int, int, int, float) x9 med 8.9 us grid 16384x1x1 blk 256x1 regs 18 smem 0.0 KB
torch: 40 launches, 6 distinct, median gap 25.2 us
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x15 med 13.2 us grid 4096x1x1 blk 128x1 regs 40 smem 0.0 KB
  void <unnamed>::elementwise_kernel_with_index<int, at::nativ x5 med 31.6 us grid 65536x1x1 blk 64x1 regs 16 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x5 med 13.5 us grid 4096x1x1 blk 128x1 regs 40 smem 0.0 KB
### pj_m16_n2048_k7168
mapal: 3 launches, 3 distinct, median gap 150.6 us
  mapal_tile_37c5043169cbcc55 x1 med 67.2 us grid 128x2x1 blk 32x1 regs 255 smem 29.7 KB
  mapal_gen_d97f309148284201 x1 med 28.0 us grid 57344x1x1 blk 256x1 regs 17 smem 0.0 KB
  mapal_gen_ab82d06483a925a7 x1 med 1.2 us grid 448x1x1 blk 256x1 regs 17 smem 0.0 KB
lib: no trace
torch: 55 launches, 13 distinct, median gap 30.5 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x6 med 143.2 us grid 65536x1x1 blk 128x1 regs 16 smem 0.0 KB
  triton_tem_fused_mm_0 x9 med 63.7 us grid 64x1x1 blk 64x1 regs 56 smem 100.4 KB
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x6 med 72.3 us grid 112x1x1 blk 128x1 regs 40 smem 0.0 KB
### prefill_1024_512_256
mapal: 1028 launches, 4 distinct, median gap 70.7 us
  mapal_tile_e2f8ae468c059815 x1022 med 4.3 us grid 1x1x1 blk 256x1 regs 38 smem 0.0 KB
  mapal_tile_022eacf5c2c33e41 x4 med 7.8 us grid 8x32x1 blk 64x1 regs 197 smem 35.8 KB
  mapal_gen_2091e33e430539f8 x1 med 1.2 us grid 256x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: no trace
torch: 177585 launches, 36 distinct, median gap 25.8 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x4248 med 56.3 us grid 24576x1x1 blk 128x1 regs 16 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x37008 med 0.9 us grid 512x1x1 blk 128x1 regs 40 smem 0.0 KB
  void gemvNSP_kernel<float, float, float, float, (int)1, (int x9216 med 2.1 us grid 8x1x1 blk 32x24 regs 28 smem 3.1 KB
### rowsum_4096
mapal: 4 launches, 4 distinct, median gap 5.3 us
  mapal_tile_6eab42dba437b31e x1 med 2154.0 us grid 340x1x1 blk 256x1 regs 126 smem 17.4 KB
  mapal_tile_900a9eaaafb2fb06 x1 med 87.8 us grid 128x1x1 blk 32x1 regs 78 smem 0.0 KB
  mapal_gen_86041d43b8de6a2e x1 med 34.7 us grid 65536x1x1 blk 256x1 regs 17 smem 0.0 KB
lib: 33 launches, 4 distinct, median gap 4.2 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x12 med 1956.6 us grid 256x2x4 blk 256x1 regs 210 smem 50.2 KB
  gen_op(float *, long long, int, int, int, int, float) x9 med 32.3 us grid 65536x1x1 blk 256x1 regs 18 smem 0.0 KB
  std::enable_if<!T7, void>::type internal::gemvx::kernel<int, x6 med 16.3 us grid 128x1x1 blk 16x32 regs 128 smem 2.0 KB
torch: 88 launches, 13 distinct, median gap 2.1 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x18 med 1975.2 us grid 256x2x4 blk 256x1 regs 210 smem 50.2 KB
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x15 med 175.7 us grid 16384x1x1 blk 128x1 regs 40 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x5 med 174.8 us grid 16384x1x1 blk 128x1 regs 40 smem 0.0 KB
### rt_mlp2_2048
mapal: 38 launches, 10 distinct, median gap 7.1 us
  mapal_tile_997700f4fb01290b x8 med 326.6 us grid 16x16x1 blk 256x1 regs 125 smem 17.4 KB
  mapal_tile_d2d7996f7d9568ae x8 med 301.4 us grid 16x16x1 blk 256x1 regs 125 smem 17.4 KB
  mapal_tile_ded6aef2041e4d09 x8 med 285.0 us grid 16x16x1 blk 256x1 regs 124 smem 17.4 KB
lib: no trace
torch: 30207 launches, 35 distinct, median gap 18.7 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x18345 med 57.9 us grid 24576x1x1 blk 128x1 regs 16 smem 0.0 KB
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x576 med 253.4 us grid 128x1x5 blk 256x1 regs 210 smem 50.2 KB
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x446 med 13.3 us grid 4096x1x1 blk 128x1 regs 40 smem 0.0 KB
### smlp_4096_256_1024
mapal: 6 launches, 6 distinct, median gap 39.0 us
  mapal_tile_d3ae82cbd020cefb x1 med 58.1 us grid 4x64x1 blk 256x1 regs 79 smem 8.2 KB
  mapal_tile_dfe0c12ef39d46f7 x1 med 37.0 us grid 8x32x1 blk 256x1 regs 125 smem 17.4 KB
  mapal_gen_da5a5e8d470768f8 x1 med 3.1 us grid 4096x1x1 blk 256x1 regs 17 smem 0.0 KB
lib: 54 launches, 6 distinct, median gap 5.4 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_128x128_8x4_nn_a x12 med 48.7 us grid 256x1x2 blk 128x1 regs 200 smem 33.8 KB
  void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_al x12 med 43.3 us grid 128x1x5 blk 128x1 regs 128 smem 31.7 KB
  void cutlass::Kernel<cutlass::gemm::kernel::Gemm<cutlass::ge x6 med 78.1 us grid 32x2x1 blk 256x1 regs 124 smem 17.4 KB
torch: 95 launches, 16 distinct, median gap 31.3 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x6 med 143.9 us grid 65536x1x1 blk 128x1 regs 16 smem 0.0 KB
  void cutlass::Kernel2<cutlass_80_simt_sgemm_128x128_8x4_nn_a x9 med 46.0 us grid 256x1x2 blk 128x1 regs 200 smem 33.8 KB
  triton_tem_fused_add_mm_relu_1 x9 med 46.2 us grid 128x1x1 blk 128x1 regs 168 smem 50.2 KB
### softmax_rows_4096
mapal: 7 launches, 7 distinct, median gap 6.8 us
  mapal_tile_6eab42dba437b31e x1 med 2165.3 us grid 340x1x1 blk 256x1 regs 126 smem 17.4 KB
  mapal_tile_a1d935db9d570bcc x1 med 103.4 us grid 128x1x1 blk 32x1 regs 46 smem 0.0 KB
  mapal_tile_900a9eaaafb2fb06 x1 med 85.9 us grid 128x1x1 blk 32x1 regs 78 smem 0.0 KB
lib: 51 launches, 7 distinct, median gap 2.6 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x12 med 1952.5 us grid 256x2x4 blk 256x1 regs 210 smem 50.2 KB
  softmax_row(float *, const float *, int, float) x6 med 107.8 us grid 512x1x1 blk 256x1 regs 38 smem 0.0 KB
  sm_exp(float *, const float *, const float *, long long, int x6 med 62.0 us grid 65536x1x1 blk 256x1 regs 20 smem 0.0 KB
torch: 1557 launches, 20 distinct, median gap 86.2 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x967 med 57.0 us grid 24576x1x1 blk 128x1 regs 16 smem 0.0 KB
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x18 med 1976.4 us grid 256x2x4 blk 256x1 regs 210 smem 50.2 KB
  triton_red_fused_add_clamp_div_max_mul_sub_sum_0 x376 med 25.7 us grid 4096x1x1 blk 512x1 regs 26 smem 0.0 KB
### twin_2048
mapal: 6 launches, 5 distinct, median gap 29.5 us
  mapal_tile_c59d5a05a86bc84f x2 med 273.5 us grid 338x1x1 blk 256x1 regs 126 smem 17.4 KB
  mapal_gen_2f1707000731833f x1 med 8.8 us grid 16384x1x1 blk 256x1 regs 17 smem 0.0 KB
  mapal_gen_6f89aa19c188a660 x1 med 8.8 us grid 16384x1x1 blk 256x1 regs 17 smem 0.0 KB
lib: no trace
torch: 94 launches, 11 distinct, median gap 22.0 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x36 med 257.5 us grid 128x1x5 blk 256x1 regs 210 smem 50.2 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x6 med 141.9 us grid 65536x1x1 blk 128x1 regs 16 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x15 med 13.2 us grid 4096x1x1 blk 128x1 regs 40 smem 0.0 KB
### epi_4096
mapal: 3 launches, 3 distinct, median gap 79.6 us
  mapal_tile_f342093f1da7ca3b x1 med 2158.8 us grid 32x32x1 blk 256x1 regs 127 smem 17.4 KB
  mapal_gen_37ef5449006a2def x1 med 31.9 us grid 65536x1x1 blk 256x1 regs 17 smem 0.0 KB
  mapal_gen_86041d43b8de6a2e x1 med 31.8 us grid 65536x1x1 blk 256x1 regs 17 smem 0.0 KB
lib: 33 launches, 4 distinct, median gap 58.8 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x12 med 1959.5 us grid 256x2x4 blk 256x1 regs 210 smem 50.2 KB
  void cutlass::Kernel<cutlass::gemm::kernel::Gemm<cutlass::ge x6 med 2156.8 us grid 32x32x1 blk 256x1 regs 124 smem 17.4 KB
  cell(int, float *, const float *, const float *, long long,  x6 med 54.9 us grid 65536x1x1 blk 256x1 regs 24 smem 0.0 KB
torch: 1459 launches, 15 distinct, median gap 80.0 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x931 med 56.3 us grid 24576x1x1 blk 128x1 regs 16 smem 0.0 KB
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x18 med 1972.7 us grid 256x2x4 blk 256x1 regs 210 smem 50.2 KB
  triton_poi_fused_add_mul_0 x440 med 26.1 us grid 32768x1x1 blk 256x1 regs 16 smem 0.0 KB
## h100-pcie
### attn_1024_64
mapal: 9 launches, 9 distinct, median gap 18.4 us
  mapal_tile_2c9ea216f11f7b72 x1 med 30.1 us grid 32x1x1 blk 32x1 regs 59 smem 0.0 KB
  mapal_tile_506b9cd2fd51f140 x1 med 27.7 us grid 32x1x1 blk 32x1 regs 62 smem 0.0 KB
  mapal_tile_1003646a5fb73ea4 x1 med 18.6 us grid 228x1x1 blk 32x1 regs 165 smem 13.3 KB
lib: 78 launches, 9 distinct, median gap 1.3 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_64x64_8x5_nn_ali x12 med 9.9 us grid 16x1x7 blk 128x1 regs 76 smem 20.5 KB
  sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize64x64x8_stage3 x12 med 9.9 us grid 16x16x1 blk 128x1 regs 128 smem 13.3 KB
  softmax_row(float *, const float *, int, float) x6 med 12.6 us grid 128x1x1 blk 256x1 regs 30 smem 0.0 KB
torch: 1803 launches, 22 distinct, median gap 35.0 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x1030 med 26.1 us grid 12800x1x1 blk 128x1 regs 16 smem 0.0 KB
  triton_per_fused_add_clamp_div_max_mul_sub_sum_0 x539 med 4.1 us grid 1024x1x1 blk 256x1 regs 20 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x6 med 132.0 us grid 65536x1x1 blk 128x1 regs 16 smem 0.0 KB
### causal_1024_256
mapal: 2050 launches, 3 distinct, median gap 140.5 us
  mapal_tile_8065b046aa4e3a84 x2048 med 11.5 us grid 228x1x1 blk 128x1 regs 47 smem 6.1 KB
  mapal_gen_398f346f86899636 x1 med 1.5 us grid 256x1x1 blk 256x1 regs 16 smem 0.0 KB
  mapal_gen_ecea1bee62ae950f x1 med 1.2 us grid 256x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: no trace
torch: 347985 launches, 31 distinct, median gap 8.2 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x4532 med 27.1 us grid 12800x1x1 blk 128x1 regs 16 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x73728 med 1.2 us grid 1x1x1 blk 128x1 regs 32 smem 0.0 KB
  std::enable_if<!T7, void>::type internal::gemvx::kernel<int, x27648 med 2.9 us grid 64x1x1 blk 4x32 regs 164 smem 1.0 KB
### convmc_8_64_56_3
mapal: no trace
lib: no trace
torch: 3316 launches, 32 distinct, median gap 4.9 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x2060 med 27.2 us grid 12800x1x1 blk 128x1 regs 16 smem 0.0 KB
  void cudnn::cnn::conv2d_grouped_direct_kernel<(bool)0, (bool x4 med 5099.3 us grid 1568x1x1 blk 1024x1 regs 40 smem 0.0 KB
  void cudnn::cnn::conv2d_grouped_direct_kernel<(bool)0, (bool x4 med 1320.8 us grid 1568x1x1 blk 1024x1 regs 40 smem 0.0 KB
### decode_1024_256
mapal: 5122 launches, 6 distinct, median gap 3.5 us
  mapal_tile_8065b046aa4e3a84 x2048 med 11.5 us grid 228x1x1 blk 128x1 regs 47 smem 6.1 KB
  mapal_tile_a21805fea156cb5e x1024 med 7.8 us grid 1x1x1 blk 256x1 regs 31 smem 0.0 KB
  mapal_tile_e5ca811c8e525f6d x1024 med 3.6 us grid 4x128x1 blk 256x1 regs 16 smem 0.0 KB
lib: 49152 launches, 5 distinct, median gap 1.1 us
  sm80_xmma_gemm_f32f32_f32f32_f32_nn_n_tilesize32x32x8_stage3 x8192 med 11.6 us grid 32x8x1 blk 64x1 regs 80 smem 7.2 KB
  std::enable_if<!T7, void>::type internal::gemvx::kernel<int, x16384 med 2.9 us grid 64x1x1 blk 4x32 regs 164 smem 1.0 KB
  std::enable_if<!T7, void>::type internal::gemvx::kernel<int, x8192 med 3.2 us grid 256x1x1 blk 32x4 regs 159 smem 1.0 KB
torch: 145232 launches, 22 distinct, median gap 8.9 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x4532 med 27.0 us grid 12800x1x1 blk 128x1 regs 16 smem 0.0 KB
  std::enable_if<!T7, void>::type internal::gemvx::kernel<int, x27648 med 2.8 us grid 64x1x1 blk 4x32 regs 164 smem 1.0 KB
  void at::native::<unnamed>::indexSelectSmallIndex<float, lon x18432 med 1.8 us grid 2x1x1 blk 128x1 regs 24 smem 0.0 KB
### diamond_2048
mapal: 7 launches, 6 distinct, median gap 8.0 us
  mapal_tile_844dbc85f2c69fb4 x2 med 619.8 us grid 16x16x1 blk 256x1 regs 127 smem 17.4 KB
  mapal_tile_02e938c50c928e3f x1 med 639.2 us grid 16x16x1 blk 256x1 regs 122 smem 17.4 KB
  mapal_gen_a01fd5f79f4fe5ff x1 med 12.6 us grid 16384x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: 93 launches, 4 distinct, median gap 4.2 us
  sm80_xmma_gemm_f32f32_f32f32_f32_nn_n_tilesize128x128x8_stag x54 med 508.9 us grid 16x16x4 blk 128x1 regs 255 smem 34.8 KB
  void cutlass::Kernel<cutlass::gemm::kernel::Gemm<cutlass::ge x18 med 682.2 us grid 16x16x1 blk 256x1 regs 137 smem 17.4 KB
  cell(int, float *, const float *, const float *, long long,  x12 med 26.1 us grid 16384x1x1 blk 256x1 regs 22 smem 0.0 KB
torch: 115 launches, 12 distinct, median gap 3.9 us
  sm80_xmma_gemm_f32f32_f32f32_f32_nn_n_tilesize128x128x8_stag x54 med 509.2 us grid 16x16x4 blk 128x1 regs 255 smem 34.8 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x6 med 129.0 us grid 65536x1x1 blk 128x1 regs 16 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x15 med 33.0 us grid 4096x1x1 blk 128x1 regs 38 smem 0.0 KB
### ew_4096
mapal: 1 launches, 1 distinct, median gap 0.0 us
  mapal_tile_07af1aed9895b0ef x1 med 72.8 us grid 65536x1x1 blk 256x1 regs 17 smem 0.0 KB
lib: 33 launches, 2 distinct, median gap 1.6 us
  cell(int, float *, const float *, const float *, long long,  x24 med 101.2 us grid 65536x1x1 blk 256x1 regs 22 smem 0.0 KB
  gen_op(float *, long long, int, int, int, int, float) x9 med 62.0 us grid 65536x1x1 blk 256x1 regs 18 smem 0.0 KB
torch: 88 launches, 14 distinct, median gap 1.5 us
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x15 med 147.4 us grid 16384x1x1 blk 128x1 regs 38 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x9 med 111.4 us grid 16384x1x1 blk 128x1 regs 32 smem 0.0 KB
  triton_poi_fused_add_mul_0 x9 med 109.2 us grid 16384x1x1 blk 512x1 regs 16 smem 0.0 KB
### fanout_2048
mapal: 5 launches, 5 distinct, median gap 24.3 us
  mapal_tile_67c37c5c7e9ac22e x1 med 630.3 us grid 16x16x1 blk 256x1 regs 127 smem 17.4 KB
  mapal_tile_844dbc85f2c69fb4 x1 med 617.1 us grid 16x16x1 blk 256x1 regs 127 smem 17.4 KB
  mapal_gen_504d11cf835d20ee x1 med 12.4 us grid 16384x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: 45 launches, 3 distinct, median gap 4.2 us
  sm80_xmma_gemm_f32f32_f32f32_f32_nn_n_tilesize128x128x8_stag x24 med 508.2 us grid 16x16x4 blk 128x1 regs 255 smem 34.8 KB
  cell(int, float *, const float *, const float *, long long,  x12 med 17.9 us grid 16384x1x1 blk 256x1 regs 22 smem 0.0 KB
  gen_op(float *, long long, int, int, int, int, float) x9 med 16.1 us grid 16384x1x1 blk 256x1 regs 18 smem 0.0 KB
torch: 118 launches, 14 distinct, median gap 1.7 us
  sm80_xmma_gemm_f32f32_f32f32_f32_nn_n_tilesize128x128x8_stag x27 med 507.4 us grid 16x16x4 blk 128x1 regs 255 smem 34.8 KB
  triton_tem_fused_add_mm_mul_0 x9 med 528.2 us grid 1024x1x1 blk 128x1 regs 96 smem 16.4 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x6 med 128.4 us grid 65536x1x1 blk 128x1 regs 16 smem 0.0 KB
### firew_rowsum_2048x2048_16
mapal: 4 launches, 4 distinct, median gap 27.0 us
  mapal_tile_0f65c9fe39cf8321 x1 med 61.3 us grid 64x1x1 blk 32x1 regs 63 smem 0.0 KB
  mapal_tile_096f5e1d92228176 x1 med 31.1 us grid 16384x1x1 blk 256x1 regs 31 smem 0.0 KB
  mapal_gen_a6e76937c1e8c33b x1 med 12.1 us grid 16385x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: no trace
torch: 104 launches, 24 distinct, median gap 5.2 us
  conv1D_NCHW_general x2 med 9419.2 us grid 1x1x1 blk 256x1 regs 31 smem 0.0 KB
  void implicit_convolve_sgemm<float, float, (int)128, (int)6, x2 med 1309.2 us grid 65536x1x1 blk 8x32 regs 77 smem 6.1 KB
  void precomputed_convolve_sgemm<float, (int)128, (int)6, (in x2 med 1128.4 us grid 65536x1x1 blk 8x32 regs 64 smem 6.1 KB
### loopmm_1024_8
mapal: 10 launches, 3 distinct, median gap 4.6 us
  mapal_tile_3591423d9b98b88a x8 med 77.2 us grid 228x1x1 blk 128x1 regs 128 smem 6.1 KB
  mapal_gen_eb67bbb3563f7336 x1 med 4.0 us grid 4096x1x1 blk 256x1 regs 16 smem 0.0 KB
  mapal_gen_093825ecb41500cf x1 med 3.7 us grid 4096x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: no trace
torch: 172 launches, 11 distinct, median gap 4.1 us
  sm80_xmma_gemm_f32f32_f32f32_f32_nn_n_tilesize64x64x8_stage3 x144 med 72.6 us grid 16x16x4 blk 128x1 regs 121 smem 13.3 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x6 med 130.0 us grid 65536x1x1 blk 128x1 regs 16 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x6 med 4.4 us grid 1024x1x1 blk 128x1 regs 38 smem 0.0 KB
### mlp2_2048
mapal: 12 launches, 11 distinct, median gap 7.4 us
  mapal_tile_844dbc85f2c69fb4 x2 med 618.8 us grid 16x16x1 blk 256x1 regs 127 smem 17.4 KB
  mapal_tile_33e054128d1b6cf1 x1 med 633.1 us grid 16x16x1 blk 256x1 regs 123 smem 17.4 KB
  mapal_tile_9bd2216916c37eb8 x1 med 626.9 us grid 16x16x1 blk 256x1 regs 123 smem 17.4 KB
lib: 153 launches, 5 distinct, median gap 1.7 us
  sm80_xmma_gemm_f32f32_f32f32_f32_nn_n_tilesize128x128x8_stag x96 med 507.8 us grid 16x16x4 blk 128x1 regs 255 smem 34.8 KB
  void cutlass::Kernel<cutlass::gemm::kernel::Gemm<cutlass::ge x12 med 692.3 us grid 16x16x1 blk 256x1 regs 137 smem 17.4 KB
  void cutlass::Kernel<cutlass::gemm::kernel::Gemm<cutlass::ge x12 med 678.2 us grid 16x16x1 blk 256x1 regs 137 smem 17.4 KB
torch: 40 launches, 6 distinct, median gap 1.2 us
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x15 med 32.4 us grid 4096x1x1 blk 128x1 regs 38 smem 0.0 KB
  void <unnamed>::elementwise_kernel_with_index<int, at::nativ x5 med 43.7 us grid 65536x1x1 blk 64x1 regs 16 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x5 med 33.2 us grid 4096x1x1 blk 128x1 regs 38 smem 0.0 KB
### mlp_2048
mapal: 5 launches, 5 distinct, median gap 24.7 us
  mapal_tile_b393c3ba2e05664d x1 med 621.9 us grid 16x16x1 blk 256x1 regs 122 smem 17.4 KB
  mapal_tile_844dbc85f2c69fb4 x1 med 615.3 us grid 16x16x1 blk 256x1 regs 127 smem 17.4 KB
  mapal_gen_504d11cf835d20ee x1 med 12.4 us grid 16384x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: 51 launches, 4 distinct, median gap 4.1 us
  sm80_xmma_gemm_f32f32_f32f32_f32_nn_n_tilesize128x128x8_stag x24 med 508.7 us grid 16x16x4 blk 128x1 regs 255 smem 34.8 KB
  void cutlass::Kernel<cutlass::gemm::kernel::Gemm<cutlass::ge x12 med 680.7 us grid 16x16x1 blk 256x1 regs 137 smem 17.4 KB
  gen_op(float *, long long, int, int, int, int, float) x9 med 16.2 us grid 16384x1x1 blk 256x1 regs 18 smem 0.0 KB
torch: 40 launches, 6 distinct, median gap 1.2 us
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x15 med 32.8 us grid 4096x1x1 blk 128x1 regs 38 smem 0.0 KB
  void <unnamed>::elementwise_kernel_with_index<int, at::nativ x5 med 43.7 us grid 65536x1x1 blk 64x1 regs 16 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x5 med 33.1 us grid 4096x1x1 blk 128x1 regs 38 smem 0.0 KB
### pj_m16_n2048_k7168
mapal: 4 launches, 4 distinct, median gap 27.7 us
  mapal_tile_62db87fb1364f56e x1 med 80.2 us grid 64x14x1 blk 64x1 regs 127 smem 23.6 KB
  mapal_gen_f51b15635cf3d84d x1 med 40.4 us grid 57344x1x1 blk 256x1 regs 16 smem 0.0 KB
  mapal_tile_7a30627f0e98010d x1 med 1.8 us grid 128x1x1 blk 256x1 regs 22 smem 0.0 KB
lib: no trace
torch: 64 launches, 12 distinct, median gap 1.4 us
  sm80_xmma_gemm_f32f32_f32f32_f32_nn_n_tilesize32x64x8_stage3 x18 med 52.0 us grid 1x32x16 blk 64x1 regs 130 smem 10.2 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x6 med 127.2 us grid 65536x1x1 blk 128x1 regs 16 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x6 med 65.3 us grid 112x1x1 blk 128x1 regs 38 smem 0.0 KB
### prefill_1024_512_256
mapal: 1028 launches, 3 distinct, median gap 154.2 us
  mapal_tile_8065b046aa4e3a84 x1026 med 11.5 us grid 228x1x1 blk 128x1 regs 47 smem 6.1 KB
  mapal_gen_398f346f86899636 x1 med 1.3 us grid 256x1x1 blk 256x1 regs 16 smem 0.0 KB
  mapal_gen_ecea1bee62ae950f x1 med 1.2 us grid 256x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: no trace
torch: 173943 launches, 35 distinct, median gap 8.1 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x1854 med 27.1 us grid 12800x1x1 blk 128x1 regs 16 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x37008 med 1.2 us grid 512x1x1 blk 128x1 regs 32 smem 0.0 KB
  std::enable_if<!T7, void>::type internal::gemvx::kernel<int, x13824 med 2.7 us grid 64x1x1 blk 4x32 regs 164 smem 1.0 KB
### rowsum_4096
mapal: 4 launches, 4 distinct, median gap 5.3 us
  mapal_tile_eb50b87a44f46aa3 x1 med 3544.0 us grid 32x32x1 blk 256x1 regs 127 smem 17.4 KB
  mapal_tile_2c3cef9f76e9b0e8 x1 med 191.2 us grid 64x1x1 blk 64x1 regs 31 smem 0.0 KB
  mapal_gen_7b59dfa488c6072f x1 med 46.9 us grid 65536x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: 33 launches, 4 distinct, median gap 2.0 us
  sm80_xmma_gemm_f32f32_f32f32_f32_nn_n_tilesize128x128x8_stag x12 med 3649.6 us grid 32x32x1 blk 128x1 regs 255 smem 34.8 KB
  gen_op(float *, long long, int, int, int, int, float) x9 med 62.7 us grid 65536x1x1 blk 256x1 regs 18 smem 0.0 KB
  row_sum(float *, const float *, int) x6 med 41.5 us grid 512x1x1 blk 256x1 regs 31 smem 0.0 KB
torch: 88 launches, 13 distinct, median gap 1.6 us
  sm80_xmma_gemm_f32f32_f32f32_f32_nn_n_tilesize128x128x8_stag x18 med 3657.8 us grid 32x32x1 blk 128x1 regs 255 smem 34.8 KB
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x15 med 147.4 us grid 16384x1x1 blk 128x1 regs 38 smem 0.0 KB
  void <unnamed>::elementwise_kernel_with_index<int, at::nativ x5 med 171.2 us grid 262144x1x1 blk 64x1 regs 16 smem 0.0 KB
### rt_mlp2_2048
mapal: 38 launches, 10 distinct, median gap 1.5 us
  mapal_tile_5c213072c31419c7 x8 med 639.6 us grid 16x16x1 blk 256x1 regs 122 smem 17.4 KB
  mapal_tile_33e054128d1b6cf1 x8 med 626.5 us grid 16x16x1 blk 256x1 regs 123 smem 17.4 KB
  mapal_tile_9bd2216916c37eb8 x8 med 622.3 us grid 16x16x1 blk 256x1 regs 123 smem 17.4 KB
lib: no trace
torch: 15549 launches, 36 distinct, median gap 4.5 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x8652 med 27.0 us grid 12800x1x1 blk 128x1 regs 16 smem 0.0 KB
  sm80_xmma_gemm_f32f32_f32f32_f32_nn_n_tilesize128x128x8_stag x432 med 513.2 us grid 16x16x4 blk 128x1 regs 255 smem 34.8 KB
  triton_tem_fused_add_relu_3 x72 med 1061.5 us grid 1024x1x1 blk 128x1 regs 128 smem 16.4 KB
### smlp_4096_256_1024
mapal: 6 launches, 6 distinct, median gap 24.2 us
  mapal_tile_ec31e1a8256dece7 x1 med 89.1 us grid 8x32x1 blk 256x1 regs 123 smem 17.4 KB
  mapal_tile_b05526af18bdc1ae x1 med 79.6 us grid 228x1x1 blk 128x1 regs 128 smem 6.1 KB
  mapal_gen_eb67bbb3563f7336 x1 med 4.0 us grid 4096x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: 54 launches, 5 distinct, median gap 4.2 us
  sm80_xmma_gemm_f32f32_f32f32_f32_nn_n_tilesize64x64x8_stage3 x24 med 75.8 us grid 64x16x1 blk 128x1 regs 121 smem 13.3 KB
  void cutlass::Kernel<cutlass::gemm::kernel::Gemm<cutlass::ge x6 med 115.8 us grid 32x2x1 blk 256x1 regs 137 smem 17.4 KB
  void cutlass::Kernel<cutlass::gemm::kernel::Gemm<cutlass::ge x6 med 97.4 us grid 32x8x1 blk 256x1 regs 137 smem 17.4 KB
torch: 95 launches, 14 distinct, median gap 5.1 us
  sm80_xmma_gemm_f32f32_f32f32_f32_nn_n_tilesize64x64x8_stage3 x27 med 78.9 us grid 64x16x1 blk 128x1 regs 121 smem 13.3 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x6 med 131.6 us grid 65536x1x1 blk 128x1 regs 16 smem 0.0 KB
  triton_tem_fused_add_mm_relu_0 x9 med 74.3 us grid 1024x1x1 blk 128x1 regs 96 smem 8.2 KB
### softmax_rows_4096
mapal: 7 launches, 7 distinct, median gap 5.1 us
  mapal_tile_eb50b87a44f46aa3 x1 med 3547.1 us grid 32x32x1 blk 256x1 regs 127 smem 17.4 KB
  mapal_tile_0ebf3e09053e243a x1 med 213.2 us grid 64x1x1 blk 64x1 regs 31 smem 0.0 KB
  mapal_tile_2c3cef9f76e9b0e8 x1 med 199.0 us grid 64x1x1 blk 64x1 regs 31 smem 0.0 KB
lib: 51 launches, 7 distinct, median gap 1.6 us
  sm80_xmma_gemm_f32f32_f32f32_f32_nn_n_tilesize128x128x8_stag x12 med 3650.9 us grid 32x32x1 blk 128x1 regs 255 smem 34.8 KB
  softmax_row(float *, const float *, int, float) x6 med 212.6 us grid 512x1x1 blk 256x1 regs 30 smem 0.0 KB
  sm_exp(float *, const float *, const float *, long long, int x6 med 92.5 us grid 65536x1x1 blk 256x1 regs 20 smem 0.0 KB
torch: 2407 launches, 20 distinct, median gap 4.7 us
  sm80_xmma_gemm_f32f32_f32f32_f32_nn_n_tilesize128x128x8_stag x18 med 3658.2 us grid 32x32x1 blk 128x1 regs 255 smem 34.8 KB
  triton_red_fused_add_clamp_div_max_mul_sub_sum_0 x751 med 74.9 us grid 4096x1x1 blk 512x1 regs 24 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x1442 med 26.6 us grid 12800x1x1 blk 128x1 regs 16 smem 0.0 KB
### twin_2048
mapal: 6 launches, 5 distinct, median gap 12.9 us
  mapal_tile_844dbc85f2c69fb4 x2 med 617.3 us grid 16x16x1 blk 256x1 regs 127 smem 17.4 KB
  mapal_gen_a01fd5f79f4fe5ff x1 med 12.5 us grid 16384x1x1 blk 256x1 regs 16 smem 0.0 KB
  mapal_gen_320e89c051337355 x1 med 12.4 us grid 16384x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: no trace
torch: 94 launches, 11 distinct, median gap 4.1 us
  sm80_xmma_gemm_f32f32_f32f32_f32_nn_n_tilesize128x128x8_stag x36 med 508.4 us grid 16x16x4 blk 128x1 regs 255 smem 34.8 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x6 med 128.2 us grid 65536x1x1 blk 128x1 regs 16 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x15 med 32.8 us grid 4096x1x1 blk 128x1 regs 38 smem 0.0 KB
### epi_4096
mapal: 3 launches, 3 distinct, median gap 52.3 us
  mapal_tile_8eca1ca76ec71fdf x1 med 3556.5 us grid 32x32x1 blk 256x1 regs 127 smem 17.4 KB
  mapal_gen_7b59dfa488c6072f x1 med 47.5 us grid 65536x1x1 blk 256x1 regs 16 smem 0.0 KB
  mapal_gen_02743d924bf99060 x1 med 47.1 us grid 65536x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: 33 launches, 4 distinct, median gap 19.9 us
  sm80_xmma_gemm_f32f32_f32f32_f32_nn_n_tilesize128x128x8_stag x12 med 3652.8 us grid 32x32x1 blk 128x1 regs 255 smem 34.8 KB
  void cutlass::Kernel<cutlass::gemm::kernel::Gemm<cutlass::ge x6 med 4145.4 us grid 32x32x1 blk 256x1 regs 137 smem 17.4 KB
  cell(int, float *, const float *, const float *, long long,  x6 med 80.8 us grid 65536x1x1 blk 256x1 regs 22 smem 0.0 KB
torch: 97 launches, 14 distinct, median gap 1.3 us
  sm80_xmma_gemm_f32f32_f32f32_f32_nn_n_tilesize128x128x8_stag x18 med 3650.6 us grid 32x32x1 blk 128x1 regs 255 smem 34.8 KB
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x15 med 147.6 us grid 16384x1x1 blk 128x1 regs 38 smem 0.0 KB
  void <unnamed>::elementwise_kernel_with_index<int, at::nativ x5 med 171.3 us grid 262144x1x1 blk 64x1 regs 16 smem 0.0 KB
## rtx-3090
### attn_1024_64
mapal: 9 launches, 9 distinct, median gap 28.4 us
  mapal_tile_3f060d722a067910 x1 med 54.6 us grid 2x512x1 blk 32x1 regs 128 smem 2.0 KB
  mapal_tile_cc03fc0c8721ad68 x1 med 29.7 us grid 32x1x1 blk 32x1 regs 128 smem 0.0 KB
  mapal_tile_17615d9177f88f12 x1 med 27.4 us grid 32x1x1 blk 32x1 regs 102 smem 0.0 KB
lib: 78 launches, 9 distinct, median gap 0.8 us
  ampere_sgemm_32x128_tn x12 med 16.9 us grid 32x8x1 blk 256x1 regs 57 smem 16.4 KB
  ampere_sgemm_64x32_sliced1x4_nn x12 med 16.5 us grid 1x32x4 blk 256x1 regs 82 smem 26.6 KB
  softmax_row(float *, const float *, int, float) x6 med 17.6 us grid 128x1x1 blk 256x1 regs 39 smem 0.0 KB
torch: 2418 launches, 24 distinct, median gap 68.4 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x1442 med 8.3 us grid 1536x1x1 blk 128x1 regs 16 smem 0.0 KB
  triton_per_fused_add_clamp_div_max_mul_sub_sum_1 x751 med 7.7 us grid 1024x1x1 blk 256x1 regs 19 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x6 med 301.8 us grid 65536x1x1 blk 128x1 regs 16 smem 0.0 KB
### causal_1024_256
mapal: 2050 launches, 4 distinct, median gap 351.6 us
  mapal_tile_0fe288810648dc17 x2046 med 6.4 us grid 1x1x1 blk 256x1 regs 38 smem 0.0 KB
  mapal_tile_ab7361387c9b0f70 x2 med 50.7 us grid 8x128x1 blk 32x1 regs 128 smem 2.0 KB
  mapal_gen_f6e78b70e65caa1e x1 med 1.8 us grid 256x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: no trace
torch: 345801 launches, 32 distinct, median gap 17.4 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x73728 med 1.4 us grid 1x1x1 blk 128x1 regs 38 smem 0.0 KB
  std::enable_if<!T7, void>::type internal::gemvx::kernel<int, x18432 med 3.0 us grid 16x1x1 blk 16x32 regs 128 smem 2.0 KB
  triton_red_fused_add_index_select_mm_unsqueeze_view_0 x9216 med 4.7 us grid 64x1x1 blk 256x1 regs 40 smem 1.0 KB
### convmc_8_64_56_3
mapal: no trace
lib: no trace
torch: 10566 launches, 39 distinct, median gap 45.9 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x6798 med 8.3 us grid 1536x1x1 blk 128x1 regs 16 smem 0.0 KB
  triton_poi_fused_convolution_0 x1599 med 18.0 us grid 53x8x1 blk 256x1 regs 33 smem 16.4 KB
  triton_poi_fused_view_2 x1175 med 24.6 us grid 1568x1x1 blk 128x1 regs 25 smem 0.0 KB
### decode_1024_256
mapal: 4098 launches, 6 distinct, median gap 12.1 us
  mapal_tile_0fe288810648dc17 x2046 med 6.4 us grid 1x1x1 blk 256x1 regs 38 smem 0.0 KB
  mapal_tile_16bba5419f8b9499 x1024 med 4.4 us grid 1x256x1 blk 256x1 regs 18 smem 0.0 KB
  mapal_tile_e6560cb0b2958805 x1024 med 4.1 us grid 4x64x1 blk 256x1 regs 17 smem 0.0 KB
lib: 49152 launches, 6 distinct, median gap 2.0 us
  ampere_sgemm_64x32_sliced1x4_nn x8192 med 14.8 us grid 4x32x1 blk 256x1 regs 82 smem 26.6 KB
  std::enable_if<!T7, void>::type internal::gemvx::kernel<int, x8192 med 3.1 us grid 32x1x1 blk 8x32 regs 165 smem 1.0 KB
  std::enable_if<!T7, void>::type internal::gemvx::kernel<int, x8192 med 2.9 us grid 32x1x1 blk 8x32 regs 159 smem 1.0 KB
torch: 148664 launches, 23 distinct, median gap 21.0 us
  std::enable_if<!T7, void>::type internal::gemvx::kernel<int, x18432 med 3.0 us grid 16x1x1 blk 16x32 regs 128 smem 2.0 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x6798 med 8.0 us grid 1536x1x1 blk 128x1 regs 16 smem 0.0 KB
  triton_red_fused_add_index_select_mm_unsqueeze_view_0 x10064 med 4.8 us grid 128x1x1 blk 128x1 regs 40 smem 1.0 KB
### diamond_2048
mapal: 7 launches, 6 distinct, median gap 10.8 us
  mapal_tile_1a22fa58c50c7414 x2 med 827.8 us grid 16x16x1 blk 128x1 regs 218 smem 17.4 KB
  mapal_tile_77021cafc874c2df x1 med 866.5 us grid 16x16x1 blk 128x1 regs 224 smem 17.4 KB
  mapal_gen_f5dcec16a664271f x1 med 21.1 us grid 16384x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: 93 launches, 4 distinct, median gap 2.8 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x54 med 756.3 us grid 128x1x5 blk 256x1 regs 202 smem 50.2 KB
  void cutlass::Kernel<cutlass::gemm::kernel::Gemm<cutlass::ge x18 med 883.5 us grid 16x16x1 blk 256x1 regs 122 smem 17.4 KB
  cell(int, float *, const float *, const float *, long long,  x12 med 61.9 us grid 16384x1x1 blk 256x1 regs 22 smem 0.0 KB
torch: 115 launches, 12 distinct, median gap 2.8 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x54 med 757.0 us grid 128x1x5 blk 256x1 regs 202 smem 50.2 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x6 med 302.4 us grid 65536x1x1 blk 128x1 regs 16 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x15 med 81.4 us grid 4096x1x1 blk 128x1 regs 36 smem 0.0 KB
### ew_4096
mapal: 1 launches, 1 distinct, median gap 0.0 us
  mapal_tile_f0a3b0ee26106009 x1 med 107.6 us grid 65536x1x1 blk 256x1 regs 17 smem 0.0 KB
lib: 33 launches, 2 distinct, median gap 0.9 us
  cell(int, float *, const float *, const float *, long long,  x24 med 198.3 us grid 65536x1x1 blk 256x1 regs 22 smem 0.0 KB
  gen_op(float *, long long, int, int, int, int, float) x9 med 93.3 us grid 65536x1x1 blk 256x1 regs 18 smem 0.0 KB
torch: 1648 launches, 15 distinct, median gap 4.2 us
  triton_poi_fused_add_mul_0 x539 med 238.6 us grid 16384x1x1 blk 128x1 regs 28 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x1030 med 8.2 us grid 1536x1x1 blk 128x1 regs 16 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x15 med 319.0 us grid 16384x1x1 blk 128x1 regs 36 smem 0.0 KB
### fanout_2048
mapal: 5 launches, 5 distinct, median gap 16.4 us
  mapal_tile_cfbe12e68f30e00e x1 med 848.4 us grid 16x16x1 blk 128x1 regs 218 smem 17.4 KB
  mapal_tile_1a22fa58c50c7414 x1 med 823.3 us grid 16x16x1 blk 128x1 regs 218 smem 17.4 KB
  mapal_gen_f5dcec16a664271f x1 med 20.9 us grid 16384x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: 45 launches, 3 distinct, median gap 2.8 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x24 med 756.2 us grid 128x1x5 blk 256x1 regs 202 smem 50.2 KB
  cell(int, float *, const float *, const float *, long long,  x12 med 42.0 us grid 16384x1x1 blk 256x1 regs 22 smem 0.0 KB
  gen_op(float *, long long, int, int, int, int, float) x9 med 25.1 us grid 16384x1x1 blk 256x1 regs 18 smem 0.0 KB
torch: 118 launches, 14 distinct, median gap 2.2 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x27 med 756.6 us grid 128x1x5 blk 256x1 regs 202 smem 50.2 KB
  triton_tem_fused_add_mm_mul_0 x9 med 831.7 us grid 512x1x1 blk 128x1 regs 168 smem 50.2 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x6 med 302.1 us grid 65536x1x1 blk 128x1 regs 16 smem 0.0 KB
### firew_rowsum_2048x2048_16
mapal: 4 launches, 4 distinct, median gap 38.6 us
  mapal_tile_c923147d58a96ae6 x1 med 68.2 us grid 16384x1x1 blk 256x1 regs 40 smem 0.0 KB
  mapal_tile_9b4ca25191072c93 x1 med 58.8 us grid 64x1x1 blk 32x1 regs 104 smem 0.0 KB
  mapal_gen_40cbe9cef6c3136d x1 med 20.9 us grid 16385x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: no trace
torch: 108 launches, 24 distinct, median gap 0.9 us
  conv1D_NCHW_general x2 med 9430.3 us grid 1x1x1 blk 256x1 regs 30 smem 0.0 KB
  void implicit_convolve_sgemm<float, float, (int)128, (int)6, x2 med 1712.7 us grid 65536x1x1 blk 8x32 regs 68 smem 6.1 KB
  void precomputed_convolve_sgemm<float, (int)128, (int)6, (in x2 med 1615.3 us grid 65536x1x1 blk 8x32 regs 66 smem 6.1 KB
### loopmm_1024_8
mapal: 10 launches, 3 distinct, median gap 7.3 us
  mapal_tile_ce5a973dfee9769e x8 med 173.7 us grid 16x16x1 blk 128x1 regs 127 smem 6.1 KB
  mapal_gen_b0b006bfea5ed91e x1 med 6.6 us grid 4096x1x1 blk 256x1 regs 16 smem 0.0 KB
  mapal_gen_915bbd7a252e06e9 x1 med 5.4 us grid 4096x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: no trace
torch: 172 launches, 12 distinct, median gap 2.7 us
  triton_tem_fused_mm_0 x72 med 123.6 us grid 128x1x1 blk 128x1 regs 168 smem 50.2 KB
  ampere_sgemm_128x64_nn x72 med 114.5 us grid 8x16x5 blk 128x1 regs 122 smem 13.3 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x6 med 302.6 us grid 65536x1x1 blk 128x1 regs 16 smem 0.0 KB
### mlp2_2048
mapal: 12 launches, 11 distinct, median gap 11.0 us
  mapal_tile_1a22fa58c50c7414 x2 med 827.1 us grid 16x16x1 blk 128x1 regs 218 smem 17.4 KB
  mapal_tile_7213fe47248d9158 x1 med 808.2 us grid 16x16x1 blk 128x1 regs 222 smem 17.4 KB
  mapal_tile_122b171f1e6d1511 x1 med 802.2 us grid 16x16x1 blk 128x1 regs 222 smem 17.4 KB
lib: 153 launches, 5 distinct, median gap 2.7 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x96 med 760.6 us grid 128x1x5 blk 256x1 regs 202 smem 50.2 KB
  void cutlass::Kernel<cutlass::gemm::kernel::Gemm<cutlass::ge x12 med 881.2 us grid 16x16x1 blk 256x1 regs 122 smem 17.4 KB
  void cutlass::Kernel<cutlass::gemm::kernel::Gemm<cutlass::ge x12 med 880.7 us grid 16x16x1 blk 256x1 regs 122 smem 17.4 KB
torch: 40 launches, 6 distinct, median gap 0.8 us
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x15 med 81.3 us grid 4096x1x1 blk 128x1 regs 36 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x5 med 81.7 us grid 4096x1x1 blk 128x1 regs 40 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x5 med 81.2 us grid 4096x1x1 blk 128x1 regs 34 smem 0.0 KB
### mlp_2048
mapal: 5 launches, 5 distinct, median gap 15.6 us
  mapal_tile_6ca6383d78fdd7ad x1 med 839.8 us grid 16x16x1 blk 128x1 regs 224 smem 17.4 KB
  mapal_tile_1a22fa58c50c7414 x1 med 818.3 us grid 16x16x1 blk 128x1 regs 218 smem 17.4 KB
  mapal_gen_f5dcec16a664271f x1 med 21.0 us grid 16384x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: 51 launches, 4 distinct, median gap 2.8 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x24 med 754.8 us grid 128x1x5 blk 256x1 regs 202 smem 50.2 KB
  void cutlass::Kernel<cutlass::gemm::kernel::Gemm<cutlass::ge x12 med 879.9 us grid 16x16x1 blk 256x1 regs 122 smem 17.4 KB
  cell(int, float *, const float *, const float *, long long,  x6 med 41.4 us grid 16384x1x1 blk 256x1 regs 22 smem 0.0 KB
torch: 40 launches, 6 distinct, median gap 0.8 us
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x15 med 81.2 us grid 4096x1x1 blk 128x1 regs 36 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x5 med 81.6 us grid 4096x1x1 blk 128x1 regs 40 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x5 med 81.4 us grid 4096x1x1 blk 128x1 regs 34 smem 0.0 KB
### pj_m16_n2048_k7168
mapal: 4 launches, 4 distinct, median gap 38.9 us
  mapal_tile_aabc3f35b5cd6bd4 x1 med 115.7 us grid 64x8x1 blk 64x1 regs 99 smem 6.1 KB
  mapal_gen_afd4d8c3eb7fb227 x1 med 67.8 us grid 57344x1x1 blk 256x1 regs 16 smem 0.0 KB
  mapal_tile_164c6cf40f1b1523 x1 med 2.2 us grid 128x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: no trace
torch: 46 launches, 12 distinct, median gap 5.6 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x6 med 300.2 us grid 65536x1x1 blk 128x1 regs 16 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x6 med 141.0 us grid 112x1x1 blk 128x1 regs 36 smem 0.0 KB
  triton_tem_fused_mm_0 x9 med 85.1 us grid 64x1x1 blk 64x1 regs 66 smem 100.4 KB
### prefill_1024_512_256
mapal: 1028 launches, 4 distinct, median gap 328.8 us
  mapal_tile_0fe288810648dc17 x1022 med 6.4 us grid 1x1x1 blk 256x1 regs 38 smem 0.0 KB
  mapal_tile_ab7361387c9b0f70 x4 med 51.1 us grid 8x128x1 blk 32x1 regs 128 smem 2.0 KB
  mapal_gen_f6e78b70e65caa1e x1 med 1.9 us grid 256x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: no trace
torch: 173319 launches, 36 distinct, median gap 17.3 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x37008 med 1.4 us grid 512x1x1 blk 128x1 regs 38 smem 0.0 KB
  std::enable_if<!T7, void>::type internal::gemvx::kernel<int, x9216 med 3.0 us grid 16x1x1 blk 16x32 regs 128 smem 2.0 KB
  triton_red_fused_add_index_select_mm_unsqueeze_view_0 x4608 med 4.8 us grid 256x1x1 blk 128x1 regs 34 smem 1.0 KB
### rowsum_4096
mapal: 4 launches, 4 distinct, median gap 4.2 us
  mapal_tile_2a54d22fbd9beb3d x1 med 5717.8 us grid 32x32x1 blk 128x1 regs 218 smem 17.4 KB
  mapal_tile_bb67f6780dcd1f60 x1 med 179.7 us grid 64x1x1 blk 64x1 regs 62 smem 0.0 KB
  mapal_gen_b80a2c18d283a7c9 x1 med 77.6 us grid 65536x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: 33 launches, 4 distinct, median gap 1.0 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x12 med 5738.3 us grid 256x2x1 blk 256x1 regs 202 smem 50.2 KB
  gen_op(float *, long long, int, int, int, int, float) x9 med 93.6 us grid 65536x1x1 blk 256x1 regs 18 smem 0.0 KB
  std::enable_if<!T7, void>::type internal::gemvx::kernel<int, x6 med 83.8 us grid 1024x1x1 blk 16x4 regs 166 smem 0.0 KB
torch: 88 launches, 13 distinct, median gap 0.8 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x18 med 5522.5 us grid 256x2x1 blk 256x1 regs 202 smem 50.2 KB
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x15 med 318.8 us grid 16384x1x1 blk 128x1 regs 36 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x6 med 301.0 us grid 65536x1x1 blk 128x1 regs 16 smem 0.0 KB
### rt_mlp2_2048
mapal: 38 launches, 10 distinct, median gap 0.8 us
  mapal_tile_dc2dca6488eca567 x8 med 865.8 us grid 16x16x1 blk 128x1 regs 224 smem 17.4 KB
  mapal_tile_1a22fa58c50c7414 x8 med 825.6 us grid 16x16x1 blk 128x1 regs 218 smem 17.4 KB
  mapal_tile_122b171f1e6d1511 x8 med 803.0 us grid 16x16x1 blk 128x1 regs 222 smem 17.4 KB
lib: no trace
torch: 20301 launches, 35 distinct, median gap 42.2 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x576 med 916.7 us grid 128x1x5 blk 256x1 regs 202 smem 50.2 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x11742 med 8.4 us grid 1536x1x1 blk 128x1 regs 16 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x446 med 81.5 us grid 4096x1x1 blk 128x1 regs 36 smem 0.0 KB
### smlp_4096_256_1024
mapal: 6 launches, 6 distinct, median gap 33.0 us
  mapal_tile_c849b908d9a22392 x1 med 171.9 us grid 4x64x1 blk 128x1 regs 127 smem 6.1 KB
  mapal_tile_bf5981fa348d40c1 x1 med 118.2 us grid 8x32x1 blk 128x1 regs 222 smem 17.4 KB
  mapal_gen_b0b006bfea5ed91e x1 med 6.7 us grid 4096x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: 60 launches, 7 distinct, median gap 2.6 us
  ampere_sgemm_128x64_nn x12 med 118.4 us grid 8x64x1 blk 128x1 regs 122 smem 13.3 KB
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x12 med 115.4 us grid 32x1x2 blk 256x1 regs 202 smem 50.2 KB
  void cutlass::Kernel<cutlass::gemm::kernel::Gemm<cutlass::ge x6 med 129.2 us grid 32x8x1 blk 256x1 regs 122 smem 17.4 KB
torch: 95 launches, 15 distinct, median gap 3.8 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x18 med 115.5 us grid 32x1x2 blk 256x1 regs 202 smem 50.2 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x6 med 302.2 us grid 65536x1x1 blk 128x1 regs 16 smem 0.0 KB
  triton_tem_fused_add_mm_relu_0 x9 med 124.6 us grid 512x1x1 blk 128x1 regs 168 smem 50.2 KB
### softmax_rows_4096
mapal: 7 launches, 7 distinct, median gap 4.1 us
  mapal_tile_2a54d22fbd9beb3d x1 med 5700.7 us grid 32x32x1 blk 128x1 regs 218 smem 17.4 KB
  mapal_tile_363eddc9b5d6b88a x1 med 194.4 us grid 64x1x1 blk 64x1 regs 59 smem 0.0 KB
  mapal_tile_bb67f6780dcd1f60 x1 med 189.1 us grid 64x1x1 blk 64x1 regs 62 smem 0.0 KB
lib: 51 launches, 7 distinct, median gap 0.9 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x12 med 5514.6 us grid 256x2x1 blk 256x1 regs 202 smem 50.2 KB
  softmax_row(float *, const float *, int, float) x6 med 402.8 us grid 512x1x1 blk 256x1 regs 39 smem 0.0 KB
  sm_exp(float *, const float *, const float *, long long, int x6 med 182.8 us grid 65536x1x1 blk 256x1 regs 18 smem 0.0 KB
torch: 223 launches, 19 distinct, median gap 0.8 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x18 med 5931.6 us grid 256x2x1 blk 256x1 regs 202 smem 50.2 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x72 med 160.9 us grid 16384x1x1 blk 128x1 regs 38 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x15 med 319.0 us grid 16384x1x1 blk 128x1 regs 36 smem 0.0 KB
### twin_2048
mapal: 6 launches, 5 distinct, median gap 10.9 us
  mapal_tile_1a22fa58c50c7414 x2 med 820.0 us grid 16x16x1 blk 128x1 regs 218 smem 17.4 KB
  mapal_gen_58c15ecd63da1319 x1 med 21.0 us grid 16384x1x1 blk 256x1 regs 16 smem 0.0 KB
  mapal_gen_f5dcec16a664271f x1 med 20.9 us grid 16384x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: no trace
torch: 94 launches, 11 distinct, median gap 2.7 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x36 med 755.5 us grid 128x1x5 blk 256x1 regs 202 smem 50.2 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x6 med 302.3 us grid 65536x1x1 blk 128x1 regs 16 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x15 med 81.3 us grid 4096x1x1 blk 128x1 regs 36 smem 0.0 KB
### epi_4096
mapal: 3 launches, 3 distinct, median gap 69.5 us
  mapal_tile_be0c76d8c52dc76d x1 med 5488.7 us grid 32x32x1 blk 128x1 regs 226 smem 17.4 KB
  mapal_gen_b80a2c18d283a7c9 x1 med 77.6 us grid 65536x1x1 blk 256x1 regs 16 smem 0.0 KB
  mapal_gen_061ef5138af7e408 x1 med 77.2 us grid 65536x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: 33 launches, 4 distinct, median gap 17.0 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x12 med 5533.6 us grid 256x2x1 blk 256x1 regs 202 smem 50.2 KB
  void cutlass::Kernel<cutlass::gemm::kernel::Gemm<cutlass::ge x6 med 5698.0 us grid 32x32x1 blk 256x1 regs 122 smem 17.4 KB
  cell(int, float *, const float *, const float *, long long,  x6 med 160.0 us grid 65536x1x1 blk 256x1 regs 22 smem 0.0 KB
torch: 97 launches, 14 distinct, median gap 0.8 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x18 med 5614.8 us grid 256x2x1 blk 256x1 regs 202 smem 50.2 KB
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x15 med 318.9 us grid 16384x1x1 blk 128x1 regs 36 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x6 med 302.6 us grid 65536x1x1 blk 128x1 regs 16 smem 0.0 KB
