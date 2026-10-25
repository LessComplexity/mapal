## rtx-4090
### attn_1024_64
mapal: 6 launches, 6 distinct, median gap 5.9 us
  mapal_tile_7e5df615868c20ca x1 med 25.5 us grid 16x16x1 blk 256x1 regs 69 smem 38.9 KB
  mapal_tile_e594df4e10ab6f60 x1 med 17.8 us grid 4x32x1 blk 32x1 regs 117 smem 88.1 KB
  mapal_tile_f431e28500e5a286 x1 med 6.4 us grid 1024x1x1 blk 256x1 regs 24 smem 6.1 KB
lib: 78 launches, 9 distinct, median gap 0.9 us
  ampere_sgemm_32x32_sliced1x4_nn x12 med 10.2 us grid 2x32x4 blk 128x1 regs 86 smem 33.8 KB
  ampere_sgemm_32x128_tn x12 med 8.9 us grid 32x8x1 blk 256x1 regs 57 smem 16.4 KB
  softmax_row(float *, const float *, int, float) x6 med 8.7 us grid 128x1x1 blk 256x1 regs 39 smem 0.0 KB
torch: 1803 launches, 23 distinct, median gap 49.5 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x1030 med 68.4 us grid 18432x1x1 blk 128x1 regs 16 smem 0.0 KB
  triton_per_fused_add_clamp_div_max_mul_sub_sum_1 x539 med 4.4 us grid 1024x1x1 blk 256x1 regs 19 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x6 med 260.1 us grid 65536x1x1 blk 128x1 regs 16 smem 0.0 KB
### causal_1024_256
mapal: 6146 launches, 8 distinct, median gap 11.7 us
  mapal_tile_6a3313e9537fe447 x1024 med 32.9 us grid 1x1x1 blk 256x1 regs 40 smem 0.0 KB
  mapal_tile_a5a5b5b477bc038b x1024 med 29.5 us grid 4x1x1 blk 256x1 regs 40 smem 0.0 KB
  mapal_tile_fee5d054184d5db4 x2046 med 4.6 us grid 1x1x1 blk 256x1 regs 38 smem 0.0 KB
lib: no trace
torch: 352977 launches, 32 distinct, median gap 17.4 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x7828 med 77.2 us grid 18432x1x1 blk 128x1 regs 16 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x73728 med 1.1 us grid 1x1x1 blk 128x1 regs 38 smem 0.0 KB
  std::enable_if<!T7, void>::type internal::gemvx::kernel<int, x18432 med 2.0 us grid 16x1x1 blk 16x32 regs 128 smem 2.0 KB
### convmc_8_64_56_3
mapal: 3 launches, 3 distinct, median gap 57.3 us
  mapal_tile_0efeae3016ec8f32 x1 med 958.0 us grid 6272x1x1 blk 256x1 regs 40 smem 0.0 KB
  mapal_gen_49b4bca503d6adb4 x1 med 5.7 us grid 6728x1x1 blk 256x1 regs 16 smem 0.0 KB
  mapal_gen_eeadc0fd8a0f6131 x1 med 1.5 us grid 144x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: no trace
torch: 5864 launches, 42 distinct, median gap 3.8 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x3708 med 77.2 us grid 18432x1x1 blk 128x1 regs 16 smem 0.0 KB
  triton_poi_fused_view_2 x1175 med 24.4 us grid 1568x1x1 blk 128x1 regs 25 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x6 med 260.0 us grid 65536x1x1 blk 128x1 regs 16 smem 0.0 KB
### decode_1024_256
mapal: 5122 launches, 7 distinct, median gap 9.0 us
  mapal_tile_fee5d054184d5db4 x2046 med 4.6 us grid 1x1x1 blk 256x1 regs 38 smem 0.0 KB
  mapal_tile_55fab8ffe34e679c x1024 med 5.7 us grid 1x512x1 blk 256x1 regs 16 smem 0.0 KB
  mapal_tile_31e028a1fa0dc070 x1024 med 3.7 us grid 4x128x1 blk 256x1 regs 16 smem 0.0 KB
lib: 49152 launches, 6 distinct, median gap 5.5 us
  ampere_sgemm_32x32_sliced1x4_nn x8192 med 7.4 us grid 8x32x1 blk 128x1 regs 86 smem 33.8 KB
  std::enable_if<!T7, void>::type internal::gemvx::kernel<int, x8192 med 2.2 us grid 32x1x1 blk 8x32 regs 165 smem 1.0 KB
  std::enable_if<!T7, void>::type internal::gemvx::kernel<int, x8192 med 2.0 us grid 16x1x1 blk 16x32 regs 128 smem 2.0 KB
torch: 146792 launches, 23 distinct, median gap 19.8 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x5562 med 77.2 us grid 18432x1x1 blk 128x1 regs 16 smem 0.0 KB
  std::enable_if<!T7, void>::type internal::gemvx::kernel<int, x18432 med 2.0 us grid 16x1x1 blk 16x32 regs 128 smem 2.0 KB
  void at::native::<unnamed>::indexSelectSmallIndex<float, lon x18432 med 1.7 us grid 2x1x1 blk 128x1 regs 32 smem 0.0 KB
### prefill_1024_512_256
mapal: 3081 launches, 13 distinct, median gap 14.2 us
  mapal_tile_e8003a0b8efe2aa4 x512 med 32.7 us grid 1x1x1 blk 256x1 regs 40 smem 0.0 KB
  mapal_tile_50510469a81382d5 x512 med 29.6 us grid 4x1x1 blk 256x1 regs 40 smem 0.0 KB
  mapal_tile_fee5d054184d5db4 x1022 med 4.6 us grid 1x1x1 blk 256x1 regs 38 smem 0.0 KB
lib: no trace
torch: 178935 launches, 36 distinct, median gap 17.5 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x5150 med 77.3 us grid 18432x1x1 blk 128x1 regs 16 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x37008 med 1.1 us grid 512x1x1 blk 128x1 regs 38 smem 0.0 KB
  triton_red_fused_add_index_select_mm_unsqueeze_view_0 x6834 med 2.5 us grid 128x1x1 blk 128x1 regs 40 smem 2.0 KB
### rt_mlp2_2048
mapal: 54 launches, 12 distinct, median gap 0.7 us
  mapal_tile_c56cd5d3a2524dfa x8 med 328.2 us grid 16x16x1 blk 128x1 regs 224 smem 17.4 KB
  mapal_tile_cb8a431d29c1dd2d x8 med 323.0 us grid 16x16x1 blk 128x1 regs 222 smem 17.4 KB
  mapal_tile_4a874e6551089bac x8 med 321.3 us grid 16x16x1 blk 128x1 regs 222 smem 17.4 KB
lib: no trace
torch: 35277 launches, 35 distinct, median gap 3.6 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x21630 med 77.8 us grid 18432x1x1 blk 128x1 regs 16 smem 0.0 KB
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x576 med 326.4 us grid 128x1x4 blk 256x1 regs 202 smem 50.2 KB
  triton_poi_fused__to_copy_add_arange_div_mul_remainder_sub_0 x1069 med 15.5 us grid 4096x1x1 blk 128x1 regs 24 smem 0.0 KB
### pj_m16_n2048_k7168
mapal: 3 launches, 3 distinct, median gap 28.5 us
  mapal_tile_6e4b9bcdd0843cf4 x1 med 71.4 us grid 64x2x1 blk 64x1 regs 251 smem 46.1 KB
  mapal_gen_8e6c109a3b29db10 x1 med 37.1 us grid 57344x1x1 blk 256x1 regs 16 smem 0.0 KB
  mapal_gen_606fb06d76aabeb6 x1 med 1.5 us grid 448x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: no trace
torch: 55 launches, 13 distinct, median gap 3.8 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x6 med 253.2 us grid 65536x1x1 blk 128x1 regs 16 smem 0.0 KB
  triton_tem_fused_mm_0 x9 med 85.8 us grid 32x1x1 blk 128x1 regs 48 smem 42.0 KB
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x6 med 125.5 us grid 112x1x1 blk 128x1 regs 36 smem 0.0 KB
### loopmm_1024_8
mapal: 10 launches, 3 distinct, median gap 0.7 us
  mapal_tile_565ffde502536326 x8 med 58.7 us grid 16x8x1 blk 256x1 regs 95 smem 78.8 KB
  mapal_gen_7a2ade86c804ef09 x1 med 4.0 us grid 4096x1x1 blk 256x1 regs 16 smem 0.0 KB
  mapal_gen_1aeae5b5ba14d87e x1 med 3.5 us grid 4096x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: no trace
torch: 172 launches, 12 distinct, median gap 0.7 us
  triton_tem_fused_mm_0 x72 med 48.9 us grid 128x1x1 blk 128x1 regs 168 smem 50.2 KB
  void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_al x72 med 43.7 us grid 128x1x1 blk 128x1 regs 126 smem 31.7 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x6 med 260.4 us grid 65536x1x1 blk 128x1 regs 16 smem 0.0 KB
### smlp_4096_256_1024
mapal: 6 launches, 6 distinct, median gap 10.9 us
  mapal_tile_a1d600c3510aec4a x1 med 60.4 us grid 4x32x1 blk 256x1 regs 95 smem 78.8 KB
  mapal_tile_52fc2d71439d6096 x1 med 47.6 us grid 8x32x1 blk 128x1 regs 222 smem 17.4 KB
  mapal_gen_7a2ade86c804ef09 x1 med 4.1 us grid 4096x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: 60 launches, 7 distinct, median gap 0.8 us
  ampere_sgemm_128x64_nn x12 med 54.8 us grid 8x64x1 blk 128x1 regs 122 smem 13.3 KB
  void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_al x12 med 47.3 us grid 128x1x1 blk 128x1 regs 126 smem 31.7 KB
  void cutlass::Kernel<cutlass::gemm::kernel::Gemm<cutlass::ge x6 med 81.0 us grid 32x2x1 blk 256x1 regs 122 smem 17.4 KB
torch: 95 launches, 16 distinct, median gap 4.0 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x6 med 262.8 us grid 65536x1x1 blk 128x1 regs 16 smem 0.0 KB
  ampere_sgemm_128x64_nn x9 med 52.8 us grid 8x64x1 blk 128x1 regs 122 smem 13.3 KB
  triton_tem_fused_add_mm_relu_0 x9 med 53.5 us grid 512x1x1 blk 128x1 regs 168 smem 50.2 KB
### mlp2_2048
mapal: 12 launches, 11 distinct, median gap 0.7 us
  mapal_tile_59027ed417ac8177 x2 med 310.5 us grid 16x16x1 blk 128x1 regs 218 smem 17.4 KB
  mapal_tile_4a874e6551089bac x1 med 344.9 us grid 16x16x1 blk 128x1 regs 222 smem 17.4 KB
  mapal_tile_cb8a431d29c1dd2d x1 med 319.7 us grid 16x16x1 blk 128x1 regs 222 smem 17.4 KB
lib: 153 launches, 5 distinct, median gap 2.3 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x96 med 325.2 us grid 128x1x4 blk 256x1 regs 202 smem 50.2 KB
  void cutlass::Kernel<cutlass::gemm::kernel::Gemm<cutlass::ge x12 med 327.6 us grid 16x16x1 blk 256x1 regs 122 smem 17.4 KB
  void cutlass::Kernel<cutlass::gemm::kernel::Gemm<cutlass::ge x12 med 317.5 us grid 16x16x1 blk 256x1 regs 122 smem 17.4 KB
torch: 40 launches, 6 distinct, median gap 0.7 us
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x15 med 33.6 us grid 4096x1x1 blk 128x1 regs 36 smem 0.0 KB
  void <unnamed>::elementwise_kernel_with_index<int, at::nativ x5 med 34.0 us grid 65536x1x1 blk 64x1 regs 16 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x5 med 31.6 us grid 4096x1x1 blk 128x1 regs 40 smem 0.0 KB
### diamond_2048
mapal: 7 launches, 6 distinct, median gap 0.7 us
  mapal_tile_59027ed417ac8177 x2 med 310.3 us grid 16x16x1 blk 128x1 regs 218 smem 17.4 KB
  mapal_tile_60426c42be5d8b82 x1 med 326.5 us grid 16x16x1 blk 128x1 regs 224 smem 17.4 KB
  mapal_gen_0c2c9285a1cff51b x1 med 11.7 us grid 16384x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: 93 launches, 4 distinct, median gap 2.4 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x54 med 319.4 us grid 128x1x4 blk 256x1 regs 202 smem 50.2 KB
  void cutlass::Kernel<cutlass::gemm::kernel::Gemm<cutlass::ge x18 med 321.2 us grid 16x16x1 blk 256x1 regs 122 smem 17.4 KB
  cell(int, float *, const float *, const float *, long long,  x12 med 19.0 us grid 16384x1x1 blk 256x1 regs 22 smem 0.0 KB
torch: 115 launches, 12 distinct, median gap 2.6 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x54 med 318.8 us grid 128x1x4 blk 256x1 regs 202 smem 50.2 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x6 med 261.9 us grid 65536x1x1 blk 128x1 regs 16 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x15 med 34.3 us grid 4096x1x1 blk 128x1 regs 36 smem 0.0 KB
### fanout_2048
mapal: 5 launches, 5 distinct, median gap 6.4 us
  mapal_tile_4ea5bf1ce04a09bd x1 med 323.8 us grid 16x16x1 blk 128x1 regs 218 smem 17.4 KB
  mapal_tile_59027ed417ac8177 x1 med 312.0 us grid 16x16x1 blk 128x1 regs 218 smem 17.4 KB
  mapal_gen_0c2c9285a1cff51b x1 med 11.5 us grid 16384x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: 45 launches, 3 distinct, median gap 2.4 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x24 med 322.5 us grid 128x1x4 blk 256x1 regs 202 smem 50.2 KB
  cell(int, float *, const float *, const float *, long long,  x12 med 12.0 us grid 16384x1x1 blk 256x1 regs 22 smem 0.0 KB
  gen_op(float *, long long, int, int, int, int, float) x9 med 12.9 us grid 16384x1x1 blk 256x1 regs 18 smem 0.0 KB
torch: 118 launches, 14 distinct, median gap 2.3 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x27 med 320.8 us grid 128x1x4 blk 256x1 regs 202 smem 50.2 KB
  triton_tem_fused_add_mm_mul_0 x9 med 351.0 us grid 512x1x1 blk 128x1 regs 168 smem 50.2 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x6 med 260.0 us grid 65536x1x1 blk 128x1 regs 16 smem 0.0 KB
### indep_2048
mapal: 6 launches, 6 distinct, median gap 0.8 us
  mapal_tile_21820f15e18c3a45 x1 med 327.1 us grid 16x16x1 blk 128x1 regs 224 smem 17.4 KB
  mapal_tile_59027ed417ac8177 x1 med 312.8 us grid 16x16x1 blk 128x1 regs 218 smem 17.4 KB
  mapal_gen_0c2c9285a1cff51b x1 med 11.7 us grid 16384x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: 69 launches, 4 distinct, median gap 2.4 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x36 med 327.5 us grid 128x1x4 blk 256x1 regs 202 smem 50.2 KB
  void cutlass::Kernel<cutlass::gemm::kernel::Gemm<cutlass::ge x12 med 327.7 us grid 16x16x1 blk 256x1 regs 122 smem 17.4 KB
  cell(int, float *, const float *, const float *, long long,  x12 med 15.7 us grid 16384x1x1 blk 256x1 regs 22 smem 0.0 KB
torch: 97 launches, 12 distinct, median gap 2.4 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x36 med 323.8 us grid 128x1x4 blk 256x1 regs 202 smem 50.2 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x6 med 256.5 us grid 65536x1x1 blk 128x1 regs 16 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x15 med 34.2 us grid 4096x1x1 blk 128x1 regs 36 smem 0.0 KB
### twin_2048
mapal: 6 launches, 5 distinct, median gap 0.7 us
  mapal_tile_59027ed417ac8177 x2 med 312.1 us grid 16x16x1 blk 128x1 regs 218 smem 17.4 KB
  mapal_gen_0c2c9285a1cff51b x1 med 11.6 us grid 16384x1x1 blk 256x1 regs 16 smem 0.0 KB
  mapal_gen_d62743ecf7801ae8 x1 med 11.6 us grid 16384x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: no trace
torch: 94 launches, 11 distinct, median gap 2.4 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x36 med 321.0 us grid 128x1x4 blk 256x1 regs 202 smem 50.2 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x6 med 260.0 us grid 65536x1x1 blk 128x1 regs 16 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x15 med 33.8 us grid 4096x1x1 blk 128x1 regs 36 smem 0.0 KB
### mlp_2048
mapal: 5 launches, 5 distinct, median gap 6.2 us
  mapal_tile_17e3127bb006f8f8 x1 med 319.3 us grid 16x16x1 blk 128x1 regs 224 smem 17.4 KB
  mapal_tile_59027ed417ac8177 x1 med 306.6 us grid 16x16x1 blk 128x1 regs 218 smem 17.4 KB
  mapal_gen_d62743ecf7801ae8 x1 med 11.7 us grid 16384x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: 51 launches, 4 distinct, median gap 2.4 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x24 med 321.7 us grid 128x1x4 blk 256x1 regs 202 smem 50.2 KB
  void cutlass::Kernel<cutlass::gemm::kernel::Gemm<cutlass::ge x12 med 321.3 us grid 16x16x1 blk 256x1 regs 122 smem 17.4 KB
  gen_op(float *, long long, int, int, int, int, float) x9 med 13.0 us grid 16384x1x1 blk 256x1 regs 18 smem 0.0 KB
torch: 40 launches, 6 distinct, median gap 0.8 us
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x15 med 33.9 us grid 4096x1x1 blk 128x1 regs 36 smem 0.0 KB
  void <unnamed>::elementwise_kernel_with_index<int, at::nativ x5 med 34.0 us grid 65536x1x1 blk 64x1 regs 16 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x5 med 31.6 us grid 4096x1x1 blk 128x1 regs 40 smem 0.0 KB
## rtx-5090
### attn_1024_64
mapal: 6 launches, 6 distinct, median gap 6.0 us
  mapal_tile_ecbd63feb16cc161 x1 med 14.5 us grid 4x32x1 blk 32x1 regs 190 smem 13.3 KB
  mapal_tile_c2d7bc077897f338 x1 med 8.7 us grid 16x16x1 blk 256x1 regs 72 smem 8.2 KB
  mapal_tile_3f45027c0de40977 x1 med 5.0 us grid 1024x1x1 blk 256x1 regs 27 smem 6.1 KB
lib: 78 launches, 9 distinct, median gap 2.4 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_128x32_8x5_tn_al x12 med 8.7 us grid 256x1x1 blk 128x1 regs 80 smem 26.6 KB
  void cutlass::Kernel2<cutlass_80_simt_sgemm_64x64_8x5_nn_ali x12 med 5.8 us grid 16x1x8 blk 128x1 regs 78 smem 20.5 KB
  softmax_row(float *, const float *, int, float) x6 med 7.2 us grid 128x1x1 blk 256x1 regs 38 smem 0.0 KB
torch: 2427 launches, 22 distinct, median gap 2.9 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x1442 med 52.8 us grid 24576x1x1 blk 128x1 regs 16 smem 0.0 KB
  triton_per_fused_add_clamp_div_max_mul_sub_sum_0 x751 med 2.9 us grid 1024x1x1 blk 256x1 regs 18 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x6 med 143.2 us grid 65536x1x1 blk 128x1 regs 16 smem 0.0 KB
### causal_1024_256
mapal: 6146 launches, 8 distinct, median gap 6.8 us
  mapal_tile_1f1ff3f246817d56 x1024 med 33.4 us grid 1x1x1 blk 256x1 regs 40 smem 0.0 KB
  mapal_tile_463d15bcce77349a x1024 med 25.7 us grid 4x1x1 blk 256x1 regs 40 smem 0.0 KB
  mapal_tile_e2f8ae468c059815 x2046 med 4.1 us grid 1x1x1 blk 256x1 regs 38 smem 0.0 KB
lib: no trace
torch: 347361 launches, 32 distinct, median gap 6.5 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x4120 med 56.2 us grid 24576x1x1 blk 128x1 regs 16 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x73728 med 0.8 us grid 1x1x1 blk 128x1 regs 40 smem 0.0 KB
  void gemvNSP_kernel<float, float, float, float, (int)1, (int x18432 med 2.0 us grid 8x1x1 blk 32x24 regs 28 smem 3.1 KB
### convmc_8_64_56_3
mapal: 3 launches, 3 distinct, median gap 16.9 us
  mapal_tile_aecfae2997eaeec3 x1 med 725.2 us grid 6272x1x1 blk 256x1 regs 40 smem 0.0 KB
  mapal_gen_e9853c9e84d433c5 x1 med 4.2 us grid 6728x1x1 blk 256x1 regs 17 smem 0.0 KB
  mapal_gen_a39aa1067d10fa40 x1 med 0.9 us grid 144x1x1 blk 256x1 regs 17 smem 0.0 KB
lib: no trace
torch: 6815 launches, 43 distinct, median gap 3.0 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x4326 med 56.2 us grid 24576x1x1 blk 128x1 regs 16 smem 0.0 KB
  triton_poi_fused_view_2 x645 med 16.2 us grid 3136x1x1 blk 256x1 regs 16 smem 0.0 KB
  void cudnn::cnn::conv2d_grouped_direct_kernel<(bool)0, (bool x2 med 2212.1 us grid 1568x1x1 blk 1024x1 regs 56 smem 0.0 KB
### decode_1024_256
mapal: 5122 launches, 7 distinct, median gap 2.5 us
  mapal_tile_e2f8ae468c059815 x2046 med 4.1 us grid 1x1x1 blk 256x1 regs 38 smem 0.0 KB
  mapal_tile_a0714d3d20b8101f x1024 med 4.8 us grid 1x512x1 blk 256x1 regs 16 smem 0.0 KB
  mapal_tile_647b5e2c7fca3543 x1024 med 2.7 us grid 4x128x1 blk 256x1 regs 16 smem 0.0 KB
lib: 57344 launches, 7 distinct, median gap 2.3 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_al x8192 med 5.0 us grid 32x1x4 blk 128x1 regs 128 smem 31.7 KB
  std::enable_if<!T7, void>::type internal::gemvx::kernel<int, x8192 med 2.2 us grid 32x1x1 blk 8x32 regs 230 smem 1.0 KB
  void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, fl x8192 med 2.1 us grid 8x64x1 blk 32x16 regs 46 smem 0.0 KB
torch: 145856 launches, 23 distinct, median gap 6.8 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x4944 med 56.2 us grid 24576x1x1 blk 128x1 regs 16 smem 0.0 KB
  void gemvNSP_kernel<float, float, float, float, (int)1, (int x18432 med 2.0 us grid 8x1x1 blk 32x24 regs 28 smem 3.1 KB
  triton_red_fused_add_index_select_mm_unsqueeze_view_0 x10170 med 2.1 us grid 256x1x1 blk 64x1 regs 35 smem 1.0 KB
### prefill_1024_512_256
mapal: 3081 launches, 13 distinct, median gap 6.3 us
  mapal_tile_3313ba029bfd34b5 x512 med 33.4 us grid 1x1x1 blk 256x1 regs 40 smem 0.0 KB
  mapal_tile_053e04729b155244 x512 med 25.9 us grid 4x1x1 blk 256x1 regs 40 smem 0.0 KB
  mapal_tile_e2f8ae468c059815 x1022 med 4.2 us grid 1x1x1 blk 256x1 regs 38 smem 0.0 KB
lib: no trace
torch: 172437 launches, 36 distinct, median gap 6.5 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x824 med 57.3 us grid 24576x1x1 blk 128x1 regs 16 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x37008 med 0.8 us grid 512x1x1 blk 128x1 regs 40 smem 0.0 KB
  void gemvNSP_kernel<float, float, float, float, (int)1, (int x9216 med 2.0 us grid 8x1x1 blk 32x24 regs 28 smem 3.1 KB
### rt_mlp2_2048
mapal: 54 launches, 12 distinct, median gap 1.5 us
  mapal_tile_d2d7996f7d9568ae x8 med 314.9 us grid 16x16x1 blk 256x1 regs 125 smem 17.4 KB
  mapal_tile_ded6aef2041e4d09 x8 med 297.5 us grid 16x16x1 blk 256x1 regs 124 smem 17.4 KB
  mapal_tile_997700f4fb01290b x8 med 295.7 us grid 16x16x1 blk 256x1 regs 125 smem 17.4 KB
lib: no trace
torch: 19989 launches, 35 distinct, median gap 2.3 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x11536 med 58.0 us grid 24576x1x1 blk 128x1 regs 16 smem 0.0 KB
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x576 med 263.8 us grid 128x1x5 blk 256x1 regs 210 smem 50.2 KB
  triton_poi_fused__to_copy_add_arange_div_mul_remainder_sub_1 x1069 med 5.6 us grid 4096x1x1 blk 128x1 regs 24 smem 0.0 KB
### pj_m16_n2048_k7168
mapal: 3 launches, 3 distinct, median gap 11.1 us
  mapal_tile_37c5043169cbcc55 x1 med 61.9 us grid 128x2x1 blk 32x1 regs 255 smem 29.7 KB
  mapal_gen_d97f309148284201 x1 med 29.6 us grid 57344x1x1 blk 256x1 regs 17 smem 0.0 KB
  mapal_gen_ab82d06483a925a7 x1 med 1.0 us grid 448x1x1 blk 256x1 regs 17 smem 0.0 KB
lib: no trace
torch: 64 launches, 12 distinct, median gap 2.4 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x6 med 144.7 us grid 65536x1x1 blk 128x1 regs 16 smem 0.0 KB
  void cutlass::Kernel2<cutlass_80_simt_sgemm_128x32_8x5_nn_al x18 med 27.4 us grid 8x2x32 blk 128x1 regs 80 smem 26.6 KB
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x6 med 72.6 us grid 112x1x1 blk 128x1 regs 40 smem 0.0 KB
### loopmm_1024_8
mapal: 10 launches, 3 distinct, median gap 1.8 us
  mapal_tile_b3430eaa07e662e7 x8 med 56.2 us grid 16x16x1 blk 256x1 regs 79 smem 8.2 KB
  mapal_gen_da5a5e8d470768f8 x1 med 2.9 us grid 4096x1x1 blk 256x1 regs 17 smem 0.0 KB
  mapal_gen_65fe65acc713e28f x1 med 2.8 us grid 4096x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: no trace
torch: 172 launches, 11 distinct, median gap 2.5 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_al x144 med 43.0 us grid 128x1x5 blk 128x1 regs 128 smem 31.7 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x6 med 144.0 us grid 65536x1x1 blk 128x1 regs 16 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x6 med 3.7 us grid 1024x1x1 blk 128x1 regs 40 smem 0.0 KB
### smlp_4096_256_1024
mapal: 6 launches, 6 distinct, median gap 2.8 us
  mapal_tile_d3ae82cbd020cefb x1 med 55.5 us grid 4x64x1 blk 256x1 regs 79 smem 8.2 KB
  mapal_tile_dfe0c12ef39d46f7 x1 med 37.4 us grid 8x32x1 blk 256x1 regs 125 smem 17.4 KB
  mapal_gen_da5a5e8d470768f8 x1 med 2.9 us grid 4096x1x1 blk 256x1 regs 17 smem 0.0 KB
lib: 54 launches, 6 distinct, median gap 3.4 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_128x128_8x4_nn_a x12 med 48.9 us grid 256x1x2 blk 128x1 regs 200 smem 33.8 KB
  void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_al x12 med 42.6 us grid 128x1x5 blk 128x1 regs 128 smem 31.7 KB
  void cutlass::Kernel<cutlass::gemm::kernel::Gemm<cutlass::ge x6 med 77.0 us grid 32x2x1 blk 256x1 regs 124 smem 17.4 KB
torch: 95 launches, 16 distinct, median gap 3.1 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x6 med 144.7 us grid 65536x1x1 blk 128x1 regs 16 smem 0.0 KB
  void cutlass::Kernel2<cutlass_80_simt_sgemm_128x128_8x4_nn_a x9 med 48.4 us grid 256x1x2 blk 128x1 regs 200 smem 33.8 KB
  triton_tem_fused_add_mm_relu_1 x9 med 44.0 us grid 128x1x1 blk 256x1 regs 126 smem 75.8 KB
### mlp2_2048
mapal: 12 launches, 11 distinct, median gap 1.6 us
  mapal_tile_c59d5a05a86bc84f x2 med 269.6 us grid 338x1x1 blk 256x1 regs 128 smem 17.4 KB
  mapal_tile_d2d7996f7d9568ae x1 med 310.6 us grid 16x16x1 blk 256x1 regs 125 smem 17.4 KB
  mapal_tile_997700f4fb01290b x1 med 306.2 us grid 16x16x1 blk 256x1 regs 125 smem 17.4 KB
lib: 153 launches, 5 distinct, median gap 2.4 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x96 med 268.9 us grid 128x1x5 blk 256x1 regs 210 smem 50.2 KB
  void cutlass::Kernel<cutlass::gemm::kernel::Gemm<cutlass::ge x12 med 300.1 us grid 16x16x1 blk 256x1 regs 124 smem 17.4 KB
  void cutlass::Kernel<cutlass::gemm::kernel::Gemm<cutlass::ge x12 med 296.1 us grid 16x16x1 blk 256x1 regs 124 smem 17.4 KB
torch: 40 launches, 6 distinct, median gap 1.6 us
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x15 med 13.1 us grid 4096x1x1 blk 128x1 regs 40 smem 0.0 KB
  void <unnamed>::elementwise_kernel_with_index<int, at::nativ x5 med 31.2 us grid 65536x1x1 blk 64x1 regs 16 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x5 med 13.7 us grid 4096x1x1 blk 128x1 regs 40 smem 0.0 KB
### diamond_2048
mapal: 7 launches, 6 distinct, median gap 1.9 us
  mapal_tile_c59d5a05a86bc84f x2 med 277.4 us grid 338x1x1 blk 256x1 regs 128 smem 17.4 KB
  mapal_tile_cbb11ec5aec4ae21 x1 med 297.6 us grid 16x16x1 blk 256x1 regs 124 smem 17.4 KB
  mapal_gen_213a43e4047e4b79 x1 med 9.2 us grid 16384x1x1 blk 256x1 regs 17 smem 0.0 KB
lib: 93 launches, 4 distinct, median gap 2.6 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x54 med 266.0 us grid 128x1x5 blk 256x1 regs 210 smem 50.2 KB
  void cutlass::Kernel<cutlass::gemm::kernel::Gemm<cutlass::ge x18 med 298.2 us grid 16x16x1 blk 256x1 regs 124 smem 17.4 KB
  cell(int, float *, const float *, const float *, long long,  x12 med 10.5 us grid 16384x1x1 blk 256x1 regs 24 smem 0.0 KB
torch: 115 launches, 12 distinct, median gap 2.5 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x54 med 265.1 us grid 128x1x5 blk 256x1 regs 210 smem 50.2 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x6 med 144.4 us grid 65536x1x1 blk 128x1 regs 16 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x15 med 13.1 us grid 4096x1x1 blk 128x1 regs 40 smem 0.0 KB
### fanout_2048
mapal: 5 launches, 5 distinct, median gap 5.1 us
  mapal_tile_e637050b79563630 x1 med 299.6 us grid 16x16x1 blk 256x1 regs 128 smem 17.4 KB
  mapal_tile_c59d5a05a86bc84f x1 med 279.8 us grid 338x1x1 blk 256x1 regs 128 smem 17.4 KB
  mapal_gen_acc3728df88a4caa x1 med 9.1 us grid 16384x1x1 blk 256x1 regs 17 smem 0.0 KB
lib: 45 launches, 3 distinct, median gap 3.4 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x24 med 263.1 us grid 128x1x5 blk 256x1 regs 210 smem 50.2 KB
  cell(int, float *, const float *, const float *, long long,  x12 med 10.1 us grid 16384x1x1 blk 256x1 regs 24 smem 0.0 KB
  gen_op(float *, long long, int, int, int, int, float) x9 med 8.9 us grid 16384x1x1 blk 256x1 regs 18 smem 0.0 KB
torch: 127 launches, 14 distinct, median gap 2.0 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x36 med 263.5 us grid 128x1x5 blk 256x1 regs 210 smem 50.2 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x6 med 145.2 us grid 65536x1x1 blk 128x1 regs 16 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x15 med 13.0 us grid 4096x1x1 blk 128x1 regs 40 smem 0.0 KB
### indep_2048
mapal: 6 launches, 6 distinct, median gap 1.2 us
  mapal_tile_518f3b62aa8ff818 x1 med 310.4 us grid 16x16x1 blk 256x1 regs 124 smem 17.4 KB
  mapal_tile_c59d5a05a86bc84f x1 med 271.2 us grid 338x1x1 blk 256x1 regs 128 smem 17.4 KB
  mapal_gen_6f89aa19c188a660 x1 med 9.2 us grid 16384x1x1 blk 256x1 regs 17 smem 0.0 KB
lib: 69 launches, 4 distinct, median gap 2.7 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x36 med 266.7 us grid 128x1x5 blk 256x1 regs 210 smem 50.2 KB
  void cutlass::Kernel<cutlass::gemm::kernel::Gemm<cutlass::ge x12 med 294.2 us grid 16x16x1 blk 256x1 regs 124 smem 17.4 KB
  cell(int, float *, const float *, const float *, long long,  x12 med 10.5 us grid 16384x1x1 blk 256x1 regs 24 smem 0.0 KB
torch: 97 launches, 12 distinct, median gap 2.2 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x36 med 264.5 us grid 128x1x5 blk 256x1 regs 210 smem 50.2 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x6 med 142.3 us grid 65536x1x1 blk 128x1 regs 16 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x15 med 13.2 us grid 4096x1x1 blk 128x1 regs 40 smem 0.0 KB
### twin_2048
mapal: 6 launches, 5 distinct, median gap 2.4 us
  mapal_tile_c59d5a05a86bc84f x2 med 277.0 us grid 338x1x1 blk 256x1 regs 128 smem 17.4 KB
  mapal_gen_acc3728df88a4caa x1 med 9.1 us grid 16384x1x1 blk 256x1 regs 17 smem 0.0 KB
  mapal_gen_213a43e4047e4b79 x1 med 9.1 us grid 16384x1x1 blk 256x1 regs 17 smem 0.0 KB
lib: no trace
torch: 94 launches, 11 distinct, median gap 2.8 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x36 med 264.8 us grid 128x1x5 blk 256x1 regs 210 smem 50.2 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x6 med 143.7 us grid 65536x1x1 blk 128x1 regs 16 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x15 med 13.1 us grid 4096x1x1 blk 128x1 regs 40 smem 0.0 KB
### mlp_2048
mapal: 5 launches, 5 distinct, median gap 5.1 us
  mapal_tile_78b03e0e414beaef x1 med 295.4 us grid 16x16x1 blk 256x1 regs 124 smem 17.4 KB
  mapal_tile_c59d5a05a86bc84f x1 med 284.1 us grid 338x1x1 blk 256x1 regs 128 smem 17.4 KB
  mapal_gen_6f89aa19c188a660 x1 med 9.2 us grid 16384x1x1 blk 256x1 regs 17 smem 0.0 KB
lib: 51 launches, 4 distinct, median gap 2.7 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x24 med 262.8 us grid 128x1x5 blk 256x1 regs 210 smem 50.2 KB
  void cutlass::Kernel<cutlass::gemm::kernel::Gemm<cutlass::ge x12 med 293.0 us grid 16x16x1 blk 256x1 regs 124 smem 17.4 KB
  gen_op(float *, long long, int, int, int, int, float) x9 med 8.9 us grid 16384x1x1 blk 256x1 regs 18 smem 0.0 KB
torch: 40 launches, 6 distinct, median gap 1.6 us
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x15 med 13.2 us grid 4096x1x1 blk 128x1 regs 40 smem 0.0 KB
  void <unnamed>::elementwise_kernel_with_index<int, at::nativ x5 med 31.2 us grid 65536x1x1 blk 64x1 regs 16 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x5 med 13.7 us grid 4096x1x1 blk 128x1 regs 40 smem 0.0 KB
## h100-pcie
### attn_1024_64
mapal: 6 launches, 6 distinct, median gap 5.7 us
  mapal_tile_1003646a5fb73ea4 x1 med 19.0 us grid 228x1x1 blk 32x1 regs 165 smem 13.3 KB
  mapal_tile_49448003a1b8421a x1 med 17.6 us grid 16x8x1 blk 256x1 regs 108 smem 13.3 KB
  mapal_tile_3abe00e78b10f1a7 x1 med 7.1 us grid 1024x1x1 blk 256x1 regs 26 smem 6.1 KB
lib: 78 launches, 9 distinct, median gap 1.5 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_64x64_8x5_nn_ali x12 med 9.9 us grid 16x1x7 blk 128x1 regs 76 smem 20.5 KB
  sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize64x64x8_stage3 x12 med 9.9 us grid 16x16x1 blk 128x1 regs 128 smem 13.3 KB
  softmax_row(float *, const float *, int, float) x6 med 12.6 us grid 128x1x1 blk 256x1 regs 30 smem 0.0 KB
torch: 2115 launches, 22 distinct, median gap 33.3 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x1236 med 26.1 us grid 12800x1x1 blk 128x1 regs 16 smem 0.0 KB
  triton_per_fused_add_clamp_div_max_mul_sub_sum_0 x645 med 4.0 us grid 1024x1x1 blk 256x1 regs 20 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x6 med 131.8 us grid 65536x1x1 blk 128x1 regs 16 smem 0.0 KB
### causal_1024_256
mapal: 6146 launches, 7 distinct, median gap 12.2 us
  mapal_tile_303544cceea92378 x1024 med 66.9 us grid 1x1x1 blk 256x1 regs 32 smem 0.0 KB
  mapal_tile_b9dafab7586f1a0c x1024 med 42.1 us grid 4x1x1 blk 256x1 regs 38 smem 0.0 KB
  mapal_tile_8065b046aa4e3a84 x2048 med 11.6 us grid 228x1x1 blk 128x1 regs 47 smem 6.1 KB
lib: no trace
torch: 349857 launches, 31 distinct, median gap 8.4 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x5768 med 27.0 us grid 12800x1x1 blk 128x1 regs 16 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x73728 med 1.2 us grid 1x1x1 blk 128x1 regs 32 smem 0.0 KB
  std::enable_if<!T7, void>::type internal::gemvx::kernel<int, x27648 med 2.8 us grid 64x1x1 blk 4x32 regs 164 smem 1.0 KB
### convmc_8_64_56_3
mapal: 3 launches, 3 distinct, median gap 42.1 us
  mapal_tile_6e6a3c3fd5d8e1a3 x1 med 1597.2 us grid 6272x1x1 blk 256x1 regs 32 smem 0.0 KB
  mapal_gen_872cc1405acca671 x1 med 5.8 us grid 6728x1x1 blk 256x1 regs 16 smem 0.0 KB
  mapal_gen_8143fc663fe67f4e x1 med 1.4 us grid 144x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: no trace
torch: 6124 launches, 32 distinct, median gap 5.3 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x3914 med 27.2 us grid 12800x1x1 blk 128x1 regs 16 smem 0.0 KB
  void cudnn::cnn::conv2d_grouped_direct_kernel<(bool)0, (bool x4 med 5149.6 us grid 1568x1x1 blk 1024x1 regs 40 smem 0.0 KB
  void cudnn::cnn::conv2d_grouped_direct_kernel<(bool)0, (bool x4 med 1328.7 us grid 1568x1x1 blk 1024x1 regs 40 smem 0.0 KB
### decode_1024_256
mapal: 6146 launches, 7 distinct, median gap 3.4 us
  mapal_tile_8065b046aa4e3a84 x2048 med 11.6 us grid 228x1x1 blk 128x1 regs 47 smem 6.1 KB
  mapal_tile_a21805fea156cb5e x1024 med 7.9 us grid 1x1x1 blk 256x1 regs 31 smem 0.0 KB
  mapal_tile_e5ca811c8e525f6d x1024 med 3.5 us grid 4x128x1 blk 256x1 regs 16 smem 0.0 KB
lib: 49152 launches, 5 distinct, median gap 1.1 us
  sm80_xmma_gemm_f32f32_f32f32_f32_nn_n_tilesize32x32x8_stage3 x8192 med 11.8 us grid 32x8x1 blk 64x1 regs 80 smem 7.2 KB
  std::enable_if<!T7, void>::type internal::gemvx::kernel<int, x16384 med 2.9 us grid 64x1x1 blk 4x32 regs 164 smem 1.0 KB
  std::enable_if<!T7, void>::type internal::gemvx::kernel<int, x8192 med 3.2 us grid 256x1x1 blk 32x4 regs 159 smem 1.0 KB
torch: 148040 launches, 22 distinct, median gap 9.2 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x6386 med 27.0 us grid 12800x1x1 blk 128x1 regs 16 smem 0.0 KB
  std::enable_if<!T7, void>::type internal::gemvx::kernel<int, x27648 med 2.8 us grid 64x1x1 blk 4x32 regs 164 smem 1.0 KB
  void at::native::<unnamed>::indexSelectSmallIndex<float, lon x18432 med 1.8 us grid 2x1x1 blk 128x1 regs 24 smem 0.0 KB
### prefill_1024_512_256
mapal: 3081 launches, 12 distinct, median gap 9.2 us
  mapal_tile_e6cb0995095314a1 x512 med 66.2 us grid 1x1x1 blk 256x1 regs 32 smem 0.0 KB
  mapal_tile_8551217df5d7e482 x512 med 43.1 us grid 4x1x1 blk 256x1 regs 38 smem 0.0 KB
  mapal_tile_8065b046aa4e3a84 x1026 med 11.6 us grid 228x1x1 blk 128x1 regs 47 smem 6.1 KB
lib: no trace
torch: 172383 launches, 35 distinct, median gap 8.2 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x37008 med 1.2 us grid 512x1x1 blk 128x1 regs 32 smem 0.0 KB
  std::enable_if<!T7, void>::type internal::gemvx::kernel<int, x13824 med 2.8 us grid 64x1x1 blk 4x32 regs 164 smem 1.0 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x824 med 27.2 us grid 12800x1x1 blk 128x1 regs 16 smem 0.0 KB
### rt_mlp2_2048
mapal: 54 launches, 12 distinct, median gap 1.5 us
  mapal_tile_5c213072c31419c7 x8 med 647.0 us grid 16x16x1 blk 256x1 regs 122 smem 17.4 KB
  mapal_tile_33e054128d1b6cf1 x8 med 630.7 us grid 16x16x1 blk 256x1 regs 123 smem 17.4 KB
  mapal_tile_9bd2216916c37eb8 x8 med 630.2 us grid 16x16x1 blk 256x1 regs 123 smem 17.4 KB
lib: no trace
torch: 23347 launches, 36 distinct, median gap 6.8 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x13801 med 27.0 us grid 12800x1x1 blk 128x1 regs 16 smem 0.0 KB
  sm80_xmma_gemm_f32f32_f32f32_f32_nn_n_tilesize128x128x8_stag x432 med 512.5 us grid 16x16x4 blk 128x1 regs 255 smem 34.8 KB
  triton_tem_fused_add_relu_3 x72 med 1034.0 us grid 1024x1x1 blk 128x1 regs 96 smem 8.2 KB
### pj_m16_n2048_k7168
mapal: 4 launches, 4 distinct, median gap 22.8 us
  mapal_tile_62db87fb1364f56e x1 med 76.6 us grid 64x14x1 blk 64x1 regs 127 smem 23.6 KB
  mapal_gen_f51b15635cf3d84d x1 med 40.2 us grid 57344x1x1 blk 256x1 regs 16 smem 0.0 KB
  mapal_tile_7a30627f0e98010d x1 med 1.7 us grid 128x1x1 blk 256x1 regs 22 smem 0.0 KB
lib: no trace
torch: 64 launches, 12 distinct, median gap 3.7 us
  sm80_xmma_gemm_f32f32_f32f32_f32_nn_n_tilesize32x64x8_stage3 x18 med 52.5 us grid 1x32x16 blk 64x1 regs 130 smem 10.2 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x6 med 127.0 us grid 65536x1x1 blk 128x1 regs 16 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x6 med 65.3 us grid 112x1x1 blk 128x1 regs 38 smem 0.0 KB
### loopmm_1024_8
mapal: 10 launches, 3 distinct, median gap 4.3 us
  mapal_tile_3591423d9b98b88a x8 med 77.5 us grid 228x1x1 blk 128x1 regs 128 smem 6.1 KB
  mapal_gen_eb67bbb3563f7336 x1 med 4.1 us grid 4096x1x1 blk 256x1 regs 16 smem 0.0 KB
  mapal_gen_093825ecb41500cf x1 med 3.7 us grid 4096x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: no trace
torch: 172 launches, 11 distinct, median gap 4.2 us
  sm80_xmma_gemm_f32f32_f32f32_f32_nn_n_tilesize64x64x8_stage3 x144 med 72.7 us grid 16x16x4 blk 128x1 regs 121 smem 13.3 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x6 med 130.2 us grid 65536x1x1 blk 128x1 regs 16 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x6 med 4.3 us grid 1024x1x1 blk 128x1 regs 38 smem 0.0 KB
### smlp_4096_256_1024
mapal: 6 launches, 6 distinct, median gap 11.3 us
  mapal_tile_ec31e1a8256dece7 x1 med 86.7 us grid 8x32x1 blk 256x1 regs 123 smem 17.4 KB
  mapal_tile_b05526af18bdc1ae x1 med 77.9 us grid 228x1x1 blk 128x1 regs 128 smem 6.1 KB
  mapal_gen_eb67bbb3563f7336 x1 med 4.0 us grid 4096x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: 54 launches, 5 distinct, median gap 4.4 us
  sm80_xmma_gemm_f32f32_f32f32_f32_nn_n_tilesize64x64x8_stage3 x24 med 73.3 us grid 64x16x1 blk 128x1 regs 121 smem 13.3 KB
  void cutlass::Kernel<cutlass::gemm::kernel::Gemm<cutlass::ge x6 med 116.4 us grid 32x2x1 blk 256x1 regs 137 smem 17.4 KB
  void cutlass::Kernel<cutlass::gemm::kernel::Gemm<cutlass::ge x6 med 96.8 us grid 32x8x1 blk 256x1 regs 137 smem 17.4 KB
torch: 95 launches, 14 distinct, median gap 5.2 us
  sm80_xmma_gemm_f32f32_f32f32_f32_nn_n_tilesize64x64x8_stage3 x27 med 74.0 us grid 64x16x1 blk 128x1 regs 121 smem 13.3 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x6 med 131.8 us grid 65536x1x1 blk 128x1 regs 16 smem 0.0 KB
  triton_tem_fused_add_mm_relu_0 x9 med 73.2 us grid 1024x1x1 blk 128x1 regs 96 smem 8.2 KB
### mlp2_2048
mapal: 12 launches, 11 distinct, median gap 1.5 us
  mapal_tile_844dbc85f2c69fb4 x2 med 616.6 us grid 16x16x1 blk 256x1 regs 127 smem 17.4 KB
  mapal_tile_33e054128d1b6cf1 x1 med 621.9 us grid 16x16x1 blk 256x1 regs 123 smem 17.4 KB
  mapal_tile_9bd2216916c37eb8 x1 med 621.4 us grid 16x16x1 blk 256x1 regs 123 smem 17.4 KB
lib: 153 launches, 5 distinct, median gap 2.0 us
  sm80_xmma_gemm_f32f32_f32f32_f32_nn_n_tilesize128x128x8_stag x96 med 512.7 us grid 16x16x4 blk 128x1 regs 255 smem 34.8 KB
  void cutlass::Kernel<cutlass::gemm::kernel::Gemm<cutlass::ge x12 med 694.8 us grid 16x16x1 blk 256x1 regs 137 smem 17.4 KB
  void cutlass::Kernel<cutlass::gemm::kernel::Gemm<cutlass::ge x12 med 683.3 us grid 16x16x1 blk 256x1 regs 137 smem 17.4 KB
torch: 40 launches, 6 distinct, median gap 1.2 us
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x15 med 32.8 us grid 4096x1x1 blk 128x1 regs 38 smem 0.0 KB
  void <unnamed>::elementwise_kernel_with_index<int, at::nativ x5 med 43.3 us grid 65536x1x1 blk 64x1 regs 16 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x5 med 33.2 us grid 4096x1x1 blk 128x1 regs 38 smem 0.0 KB
### diamond_2048
mapal: 7 launches, 6 distinct, median gap 6.3 us
  mapal_tile_844dbc85f2c69fb4 x2 med 622.4 us grid 16x16x1 blk 256x1 regs 127 smem 17.4 KB
  mapal_tile_02e938c50c928e3f x1 med 644.0 us grid 16x16x1 blk 256x1 regs 122 smem 17.4 KB
  mapal_gen_a01fd5f79f4fe5ff x1 med 12.7 us grid 16384x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: 93 launches, 4 distinct, median gap 4.3 us
  sm80_xmma_gemm_f32f32_f32f32_f32_nn_n_tilesize128x128x8_stag x54 med 513.1 us grid 16x16x4 blk 128x1 regs 255 smem 34.8 KB
  void cutlass::Kernel<cutlass::gemm::kernel::Gemm<cutlass::ge x18 med 685.1 us grid 16x16x1 blk 256x1 regs 137 smem 17.4 KB
  cell(int, float *, const float *, const float *, long long,  x12 med 26.4 us grid 16384x1x1 blk 256x1 regs 22 smem 0.0 KB
torch: 115 launches, 12 distinct, median gap 3.9 us
  sm80_xmma_gemm_f32f32_f32f32_f32_nn_n_tilesize128x128x8_stag x54 med 512.7 us grid 16x16x4 blk 128x1 regs 255 smem 34.8 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x6 med 128.6 us grid 65536x1x1 blk 128x1 regs 16 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x15 med 32.7 us grid 4096x1x1 blk 128x1 regs 38 smem 0.0 KB
### fanout_2048
mapal: 5 launches, 5 distinct, median gap 3.6 us
  mapal_tile_67c37c5c7e9ac22e x1 med 628.2 us grid 16x16x1 blk 256x1 regs 127 smem 17.4 KB
  mapal_tile_844dbc85f2c69fb4 x1 med 620.7 us grid 16x16x1 blk 256x1 regs 127 smem 17.4 KB
  mapal_gen_504d11cf835d20ee x1 med 12.6 us grid 16384x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: 45 launches, 3 distinct, median gap 4.6 us
  sm80_xmma_gemm_f32f32_f32f32_f32_nn_n_tilesize128x128x8_stag x24 med 507.4 us grid 16x16x4 blk 128x1 regs 255 smem 34.8 KB
  cell(int, float *, const float *, const float *, long long,  x12 med 17.9 us grid 16384x1x1 blk 256x1 regs 22 smem 0.0 KB
  gen_op(float *, long long, int, int, int, int, float) x9 med 16.2 us grid 16384x1x1 blk 256x1 regs 18 smem 0.0 KB
torch: 118 launches, 14 distinct, median gap 1.9 us
  sm80_xmma_gemm_f32f32_f32f32_f32_nn_n_tilesize128x128x8_stag x27 med 507.4 us grid 16x16x4 blk 128x1 regs 255 smem 34.8 KB
  triton_tem_fused_add_mm_mul_0 x9 med 527.0 us grid 1024x1x1 blk 128x1 regs 96 smem 16.4 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x6 med 128.6 us grid 65536x1x1 blk 128x1 regs 16 smem 0.0 KB
### indep_2048
mapal: 6 launches, 6 distinct, median gap 1.5 us
  mapal_tile_55e1464af006c502 x1 med 640.6 us grid 16x16x1 blk 256x1 regs 122 smem 17.4 KB
  mapal_tile_844dbc85f2c69fb4 x1 med 615.3 us grid 16x16x1 blk 256x1 regs 127 smem 17.4 KB
  mapal_gen_a01fd5f79f4fe5ff x1 med 12.8 us grid 16384x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: 69 launches, 4 distinct, median gap 4.3 us
  sm80_xmma_gemm_f32f32_f32f32_f32_nn_n_tilesize128x128x8_stag x36 med 511.3 us grid 16x16x4 blk 128x1 regs 255 smem 34.8 KB
  void cutlass::Kernel<cutlass::gemm::kernel::Gemm<cutlass::ge x12 med 690.2 us grid 16x16x1 blk 256x1 regs 137 smem 17.4 KB
  cell(int, float *, const float *, const float *, long long,  x12 med 26.0 us grid 16384x1x1 blk 256x1 regs 22 smem 0.0 KB
torch: 97 launches, 12 distinct, median gap 4.0 us
  sm80_xmma_gemm_f32f32_f32f32_f32_nn_n_tilesize128x128x8_stag x36 med 512.7 us grid 16x16x4 blk 128x1 regs 255 smem 34.8 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x6 med 128.9 us grid 65536x1x1 blk 128x1 regs 16 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x15 med 32.9 us grid 4096x1x1 blk 128x1 regs 38 smem 0.0 KB
### twin_2048
mapal: 6 launches, 5 distinct, median gap 1.3 us
  mapal_tile_844dbc85f2c69fb4 x2 med 613.7 us grid 16x16x1 blk 256x1 regs 127 smem 17.4 KB
  mapal_gen_a01fd5f79f4fe5ff x1 med 13.4 us grid 16384x1x1 blk 256x1 regs 16 smem 0.0 KB
  mapal_gen_504d11cf835d20ee x1 med 12.6 us grid 16384x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: no trace
torch: 94 launches, 11 distinct, median gap 4.6 us
  sm80_xmma_gemm_f32f32_f32f32_f32_nn_n_tilesize128x128x8_stag x36 med 506.6 us grid 16x16x4 blk 128x1 regs 255 smem 34.8 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x6 med 128.1 us grid 65536x1x1 blk 128x1 regs 16 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x15 med 32.9 us grid 4096x1x1 blk 128x1 regs 38 smem 0.0 KB
### mlp_2048
mapal: 5 launches, 5 distinct, median gap 1.2 us
  mapal_tile_b393c3ba2e05664d x1 med 620.5 us grid 16x16x1 blk 256x1 regs 122 smem 17.4 KB
  mapal_tile_844dbc85f2c69fb4 x1 med 619.8 us grid 16x16x1 blk 256x1 regs 127 smem 17.4 KB
  mapal_gen_504d11cf835d20ee x1 med 12.8 us grid 16384x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: 51 launches, 4 distinct, median gap 4.3 us
  sm80_xmma_gemm_f32f32_f32f32_f32_nn_n_tilesize128x128x8_stag x24 med 513.0 us grid 16x16x4 blk 128x1 regs 255 smem 34.8 KB
  void cutlass::Kernel<cutlass::gemm::kernel::Gemm<cutlass::ge x12 med 683.6 us grid 16x16x1 blk 256x1 regs 137 smem 17.4 KB
  gen_op(float *, long long, int, int, int, int, float) x9 med 16.2 us grid 16384x1x1 blk 256x1 regs 18 smem 0.0 KB
torch: 40 launches, 6 distinct, median gap 1.3 us
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x15 med 32.8 us grid 4096x1x1 blk 128x1 regs 38 smem 0.0 KB
  void <unnamed>::elementwise_kernel_with_index<int, at::nativ x5 med 43.2 us grid 65536x1x1 blk 64x1 regs 16 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x5 med 33.4 us grid 4096x1x1 blk 128x1 regs 38 smem 0.0 KB
## rtx-3090
### attn_1024_64
mapal: 6 launches, 6 distinct, median gap 14.7 us
  mapal_tile_3f060d722a067910 x1 med 55.6 us grid 2x512x1 blk 32x1 regs 128 smem 2.0 KB
  mapal_tile_cb2d651b92e252b2 x1 med 20.2 us grid 16x8x1 blk 256x1 regs 108 smem 13.3 KB
  mapal_tile_6aa25a496bfe2dd1 x1 med 13.0 us grid 1024x1x1 blk 256x1 regs 24 smem 6.1 KB
lib: 78 launches, 9 distinct, median gap 0.8 us
  ampere_sgemm_32x128_tn x12 med 16.8 us grid 32x8x1 blk 256x1 regs 57 smem 16.4 KB
  ampere_sgemm_64x32_sliced1x4_nn x12 med 16.5 us grid 1x32x4 blk 256x1 regs 82 smem 26.6 KB
  softmax_row(float *, const float *, int, float) x6 med 17.6 us grid 128x1x1 blk 256x1 regs 39 smem 0.0 KB
torch: 2418 launches, 24 distinct, median gap 64.7 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x1442 med 8.3 us grid 1536x1x1 blk 128x1 regs 16 smem 0.0 KB
  triton_per_fused_add_clamp_div_max_mul_sub_sum_1 x751 med 7.6 us grid 1024x1x1 blk 256x1 regs 19 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x6 med 302.0 us grid 65536x1x1 blk 128x1 regs 16 smem 0.0 KB
### causal_1024_256
mapal: 6146 launches, 8 distinct, median gap 13.0 us
  mapal_tile_4a7d4bbfa49ba1b0 x1024 med 45.0 us grid 1x1x1 blk 256x1 regs 40 smem 0.0 KB
  mapal_tile_5b66edcc40869ddc x1024 med 43.8 us grid 4x1x1 blk 256x1 regs 40 smem 0.0 KB
  mapal_tile_0fe288810648dc17 x2046 med 6.4 us grid 1x1x1 blk 256x1 regs 38 smem 0.0 KB
lib: no trace
torch: 351417 launches, 32 distinct, median gap 17.1 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x73728 med 1.4 us grid 1x1x1 blk 128x1 regs 38 smem 0.0 KB
  std::enable_if<!T7, void>::type internal::gemvx::kernel<int, x18432 med 3.0 us grid 16x1x1 blk 16x32 regs 128 smem 2.0 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x6798 med 8.1 us grid 1536x1x1 blk 128x1 regs 16 smem 0.0 KB
### convmc_8_64_56_3
mapal: 3 launches, 3 distinct, median gap 78.3 us
  mapal_tile_c3f9263909f9b4e5 x1 med 2200.5 us grid 6272x1x1 blk 256x1 regs 40 smem 0.0 KB
  mapal_gen_7f9dc48c9ec53ea3 x1 med 9.7 us grid 6728x1x1 blk 256x1 regs 16 smem 0.0 KB
  mapal_gen_237fc936aaf78066 x1 med 1.8 us grid 144x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: no trace
torch: 7758 launches, 39 distinct, median gap 46.4 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x4944 med 8.2 us grid 1536x1x1 blk 128x1 regs 16 smem 0.0 KB
  void cudnn::cnn::conv2d_grouped_direct_kernel<(bool)0, (bool x4 med 7380.7 us grid 1568x1x1 blk 1024x1 regs 43 smem 0.0 KB
  triton_poi_fused_view_2 x857 med 24.3 us grid 3136x1x1 blk 256x1 regs 16 smem 0.0 KB
### decode_1024_256
mapal: 5122 launches, 7 distinct, median gap 3.3 us
  mapal_tile_0fe288810648dc17 x2046 med 6.4 us grid 1x1x1 blk 256x1 regs 38 smem 0.0 KB
  mapal_tile_16bba5419f8b9499 x1024 med 4.4 us grid 1x256x1 blk 256x1 regs 18 smem 0.0 KB
  mapal_tile_e6560cb0b2958805 x1024 med 4.1 us grid 4x64x1 blk 256x1 regs 17 smem 0.0 KB
lib: 49152 launches, 6 distinct, median gap 2.0 us
  ampere_sgemm_64x32_sliced1x4_nn x8192 med 14.7 us grid 4x32x1 blk 256x1 regs 82 smem 26.6 KB
  std::enable_if<!T7, void>::type internal::gemvx::kernel<int, x8192 med 3.1 us grid 32x1x1 blk 8x32 regs 165 smem 1.0 KB
  std::enable_if<!T7, void>::type internal::gemvx::kernel<int, x8192 med 2.9 us grid 32x1x1 blk 8x32 regs 159 smem 1.0 KB
torch: 149600 launches, 23 distinct, median gap 19.2 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x7416 med 8.0 us grid 1536x1x1 blk 128x1 regs 16 smem 0.0 KB
  std::enable_if<!T7, void>::type internal::gemvx::kernel<int, x18432 med 3.0 us grid 16x1x1 blk 16x32 regs 128 smem 2.0 KB
  void at::native::<unnamed>::indexSelectSmallIndex<float, lon x18432 med 2.2 us grid 2x1x1 blk 128x1 regs 24 smem 0.0 KB
### prefill_1024_512_256
mapal: 3081 launches, 13 distinct, median gap 14.0 us
  mapal_tile_1de9c1f329ed9513 x512 med 45.1 us grid 1x1x1 blk 256x1 regs 40 smem 0.0 KB
  mapal_tile_99a38c60b394c8a2 x512 med 44.0 us grid 4x1x1 blk 256x1 regs 40 smem 0.0 KB
  mapal_tile_0fe288810648dc17 x1022 med 6.4 us grid 1x1x1 blk 256x1 regs 38 smem 0.0 KB
lib: no trace
torch: 172383 launches, 36 distinct, median gap 17.2 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x37008 med 1.4 us grid 512x1x1 blk 128x1 regs 38 smem 0.0 KB
  std::enable_if<!T7, void>::type internal::gemvx::kernel<int, x9216 med 3.1 us grid 16x1x1 blk 16x32 regs 128 smem 2.0 KB
  void at::native::<unnamed>::indexSelectSmallIndex<float, lon x9216 med 2.4 us grid 2x1x1 blk 128x1 regs 24 smem 0.0 KB
### rt_mlp2_2048
mapal: 54 launches, 12 distinct, median gap 0.8 us
  mapal_tile_dc2dca6488eca567 x8 med 865.4 us grid 16x16x1 blk 128x1 regs 224 smem 17.4 KB
  mapal_tile_1a22fa58c50c7414 x8 med 825.5 us grid 16x16x1 blk 128x1 regs 218 smem 17.4 KB
  mapal_tile_7213fe47248d9158 x8 med 806.3 us grid 16x16x1 blk 128x1 regs 222 smem 17.4 KB
lib: no trace
torch: 31845 launches, 35 distinct, median gap 41.5 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x576 med 802.8 us grid 128x1x5 blk 256x1 regs 202 smem 50.2 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x19364 med 8.4 us grid 1536x1x1 blk 128x1 regs 16 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x446 med 81.3 us grid 4096x1x1 blk 128x1 regs 36 smem 0.0 KB
### pj_m16_n2048_k7168
mapal: 4 launches, 4 distinct, median gap 14.3 us
  mapal_tile_aabc3f35b5cd6bd4 x1 med 115.5 us grid 64x8x1 blk 64x1 regs 99 smem 6.1 KB
  mapal_gen_afd4d8c3eb7fb227 x1 med 67.6 us grid 57344x1x1 blk 256x1 regs 16 smem 0.0 KB
  mapal_tile_164c6cf40f1b1523 x1 med 2.3 us grid 128x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: no trace
torch: 46 launches, 12 distinct, median gap 5.7 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x6 med 300.2 us grid 65536x1x1 blk 128x1 regs 16 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x6 med 140.7 us grid 112x1x1 blk 128x1 regs 36 smem 0.0 KB
  triton_tem_fused_mm_0 x9 med 85.2 us grid 64x1x1 blk 64x1 regs 66 smem 100.4 KB
### loopmm_1024_8
mapal: 10 launches, 3 distinct, median gap 7.3 us
  mapal_tile_ce5a973dfee9769e x8 med 173.5 us grid 16x16x1 blk 128x1 regs 127 smem 6.1 KB
  mapal_gen_b0b006bfea5ed91e x1 med 6.7 us grid 4096x1x1 blk 256x1 regs 16 smem 0.0 KB
  mapal_gen_915bbd7a252e06e9 x1 med 5.2 us grid 4096x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: no trace
torch: 172 launches, 12 distinct, median gap 2.6 us
  triton_tem_fused_mm_0 x72 med 122.8 us grid 128x1x1 blk 128x1 regs 168 smem 50.2 KB
  ampere_sgemm_128x64_nn x72 med 114.1 us grid 8x16x5 blk 128x1 regs 122 smem 13.3 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x6 med 302.6 us grid 65536x1x1 blk 128x1 regs 16 smem 0.0 KB
### smlp_4096_256_1024
mapal: 6 launches, 6 distinct, median gap 8.5 us
  mapal_tile_c849b908d9a22392 x1 med 170.6 us grid 4x64x1 blk 128x1 regs 127 smem 6.1 KB
  mapal_tile_bf5981fa348d40c1 x1 med 117.8 us grid 8x32x1 blk 128x1 regs 222 smem 17.4 KB
  mapal_gen_b0b006bfea5ed91e x1 med 6.6 us grid 4096x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: 60 launches, 7 distinct, median gap 2.7 us
  ampere_sgemm_128x64_nn x12 med 117.6 us grid 8x64x1 blk 128x1 regs 122 smem 13.3 KB
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x12 med 115.0 us grid 32x1x2 blk 256x1 regs 202 smem 50.2 KB
  void cutlass::Kernel<cutlass::gemm::kernel::Gemm<cutlass::ge x6 med 129.0 us grid 32x8x1 blk 256x1 regs 122 smem 17.4 KB
torch: 95 launches, 15 distinct, median gap 3.8 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x18 med 115.2 us grid 32x1x2 blk 256x1 regs 202 smem 50.2 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x6 med 302.2 us grid 65536x1x1 blk 128x1 regs 16 smem 0.0 KB
  triton_tem_fused_add_mm_relu_0 x9 med 124.3 us grid 512x1x1 blk 128x1 regs 168 smem 50.2 KB
### mlp2_2048
mapal: 12 launches, 11 distinct, median gap 0.8 us
  mapal_tile_1a22fa58c50c7414 x2 med 825.6 us grid 16x16x1 blk 128x1 regs 218 smem 17.4 KB
  mapal_tile_7213fe47248d9158 x1 med 808.2 us grid 16x16x1 blk 128x1 regs 222 smem 17.4 KB
  mapal_tile_122b171f1e6d1511 x1 med 803.7 us grid 16x16x1 blk 128x1 regs 222 smem 17.4 KB
lib: 153 launches, 5 distinct, median gap 2.7 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x96 med 759.7 us grid 128x1x5 blk 256x1 regs 202 smem 50.2 KB
  void cutlass::Kernel<cutlass::gemm::kernel::Gemm<cutlass::ge x12 med 886.4 us grid 16x16x1 blk 256x1 regs 122 smem 17.4 KB
  void cutlass::Kernel<cutlass::gemm::kernel::Gemm<cutlass::ge x12 med 881.5 us grid 16x16x1 blk 256x1 regs 122 smem 17.4 KB
torch: 40 launches, 6 distinct, median gap 0.8 us
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x15 med 81.3 us grid 4096x1x1 blk 128x1 regs 36 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x5 med 81.9 us grid 4096x1x1 blk 128x1 regs 40 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x5 med 81.7 us grid 4096x1x1 blk 128x1 regs 34 smem 0.0 KB
### diamond_2048
mapal: 7 launches, 6 distinct, median gap 0.8 us
  mapal_tile_1a22fa58c50c7414 x2 med 829.5 us grid 16x16x1 blk 128x1 regs 218 smem 17.4 KB
  mapal_tile_77021cafc874c2df x1 med 866.7 us grid 16x16x1 blk 128x1 regs 224 smem 17.4 KB
  mapal_gen_58c15ecd63da1319 x1 med 20.8 us grid 16384x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: 93 launches, 4 distinct, median gap 2.7 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x54 med 756.1 us grid 128x1x5 blk 256x1 regs 202 smem 50.2 KB
  void cutlass::Kernel<cutlass::gemm::kernel::Gemm<cutlass::ge x18 med 884.1 us grid 16x16x1 blk 256x1 regs 122 smem 17.4 KB
  cell(int, float *, const float *, const float *, long long,  x12 med 62.0 us grid 16384x1x1 blk 256x1 regs 22 smem 0.0 KB
torch: 115 launches, 12 distinct, median gap 2.8 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x54 med 756.3 us grid 128x1x5 blk 256x1 regs 202 smem 50.2 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x6 med 302.2 us grid 65536x1x1 blk 128x1 regs 16 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x15 med 81.2 us grid 4096x1x1 blk 128x1 regs 36 smem 0.0 KB
### fanout_2048
mapal: 5 launches, 5 distinct, median gap 1.0 us
  mapal_tile_cfbe12e68f30e00e x1 med 846.6 us grid 16x16x1 blk 128x1 regs 218 smem 17.4 KB
  mapal_tile_1a22fa58c50c7414 x1 med 822.6 us grid 16x16x1 blk 128x1 regs 218 smem 17.4 KB
  mapal_gen_c1ed4a9d6a99b5ec x1 med 20.5 us grid 16384x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: 45 launches, 3 distinct, median gap 2.8 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x24 med 756.3 us grid 128x1x5 blk 256x1 regs 202 smem 50.2 KB
  cell(int, float *, const float *, const float *, long long,  x12 med 42.1 us grid 16384x1x1 blk 256x1 regs 22 smem 0.0 KB
  gen_op(float *, long long, int, int, int, int, float) x9 med 24.8 us grid 16384x1x1 blk 256x1 regs 18 smem 0.0 KB
torch: 127 launches, 14 distinct, median gap 1.6 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x36 med 755.9 us grid 128x1x5 blk 256x1 regs 202 smem 50.2 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x6 med 302.3 us grid 65536x1x1 blk 128x1 regs 16 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x15 med 81.1 us grid 4096x1x1 blk 128x1 regs 36 smem 0.0 KB
### indep_2048
mapal: 6 launches, 6 distinct, median gap 0.7 us
  mapal_tile_524d5ba29f761422 x1 med 877.6 us grid 16x16x1 blk 128x1 regs 224 smem 17.4 KB
  mapal_tile_1a22fa58c50c7414 x1 med 823.5 us grid 16x16x1 blk 128x1 regs 218 smem 17.4 KB
  mapal_gen_58c15ecd63da1319 x1 med 20.8 us grid 16384x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: 69 launches, 4 distinct, median gap 2.7 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x36 med 757.1 us grid 128x1x5 blk 256x1 regs 202 smem 50.2 KB
  void cutlass::Kernel<cutlass::gemm::kernel::Gemm<cutlass::ge x12 med 885.8 us grid 16x16x1 blk 256x1 regs 122 smem 17.4 KB
  cell(int, float *, const float *, const float *, long long,  x12 med 62.3 us grid 16384x1x1 blk 256x1 regs 22 smem 0.0 KB
torch: 97 launches, 12 distinct, median gap 2.7 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x36 med 756.5 us grid 128x1x5 blk 256x1 regs 202 smem 50.2 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x6 med 302.2 us grid 65536x1x1 blk 128x1 regs 16 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x15 med 81.3 us grid 4096x1x1 blk 128x1 regs 36 smem 0.0 KB
### twin_2048
mapal: 6 launches, 5 distinct, median gap 1.0 us
  mapal_tile_1a22fa58c50c7414 x2 med 827.9 us grid 16x16x1 blk 128x1 regs 218 smem 17.4 KB
  mapal_gen_58c15ecd63da1319 x1 med 20.9 us grid 16384x1x1 blk 256x1 regs 16 smem 0.0 KB
  mapal_gen_c1ed4a9d6a99b5ec x1 med 20.7 us grid 16384x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: no trace
torch: 94 launches, 11 distinct, median gap 2.7 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x36 med 754.9 us grid 128x1x5 blk 256x1 regs 202 smem 50.2 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x6 med 302.2 us grid 65536x1x1 blk 128x1 regs 16 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x15 med 81.0 us grid 4096x1x1 blk 128x1 regs 36 smem 0.0 KB
### mlp_2048
mapal: 5 launches, 5 distinct, median gap 0.8 us
  mapal_tile_6ca6383d78fdd7ad x1 med 859.9 us grid 16x16x1 blk 128x1 regs 224 smem 17.4 KB
  mapal_tile_1a22fa58c50c7414 x1 med 830.1 us grid 16x16x1 blk 128x1 regs 218 smem 17.4 KB
  mapal_gen_c1ed4a9d6a99b5ec x1 med 20.6 us grid 16384x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: 51 launches, 4 distinct, median gap 2.8 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x24 med 755.8 us grid 128x1x5 blk 256x1 regs 202 smem 50.2 KB
  void cutlass::Kernel<cutlass::gemm::kernel::Gemm<cutlass::ge x12 med 883.4 us grid 16x16x1 blk 256x1 regs 122 smem 17.4 KB
  cell(int, float *, const float *, const float *, long long,  x6 med 41.5 us grid 16384x1x1 blk 256x1 regs 22 smem 0.0 KB
torch: 40 launches, 6 distinct, median gap 0.8 us
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x15 med 81.3 us grid 4096x1x1 blk 128x1 regs 36 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x5 med 81.9 us grid 4096x1x1 blk 128x1 regs 40 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x5 med 81.0 us grid 4096x1x1 blk 128x1 regs 34 smem 0.0 KB
