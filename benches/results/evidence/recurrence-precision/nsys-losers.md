## rtx-4090
### decode_1024_256
mapal: 6146 launches, 8 distinct, median gap 5.2 us
  mapal_tile_fee5d054184d5db4 x2046 med 4.6 us grid 1x1x1 blk 256x1 regs 38 smem 0.0 KB
  mapal_tile_55fab8ffe34e679c x1024 med 5.7 us grid 1x512x1 blk 256x1 regs 16 smem 0.0 KB
  mapal_tile_31e028a1fa0dc070 x1024 med 3.7 us grid 4x128x1 blk 256x1 regs 16 smem 0.0 KB
lib: 49152 launches, 6 distinct, median gap 5.7 us
  ampere_sgemm_32x32_sliced1x4_nn x8192 med 7.4 us grid 8x32x1 blk 128x1 regs 86 smem 33.8 KB
  std::enable_if<!T7, void>::type internal::gemvx::kernel<int, x8192 med 2.2 us grid 32x1x1 blk 8x32 regs 165 smem 1.0 KB
  std::enable_if<!T7, void>::type internal::gemvx::kernel<int, x8192 med 2.0 us grid 64x1x1 blk 16x16 regs 159 smem 1.0 KB
torch: 150848 launches, 23 distinct, median gap 19.9 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x8240 med 75.3 us grid 18432x1x1 blk 128x1 regs 16 smem 0.0 KB
  std::enable_if<!T7, void>::type internal::gemvx::kernel<int, x18432 med 2.0 us grid 16x1x1 blk 16x32 regs 128 smem 2.0 KB
  void at::native::<unnamed>::indexSelectSmallIndex<float, lon x18432 med 1.6 us grid 2x1x1 blk 128x1 regs 32 smem 0.0 KB
### convmc_8_64_56_3
mapal: 3 launches, 3 distinct, median gap 38.5 us
  mapal_tile_898f4ee4efe2e75f x1 med 100.7 us grid 784x1x1 blk 64x1 regs 128 smem 13.3 KB
  mapal_gen_49b4bca503d6adb4 x1 med 5.6 us grid 6728x1x1 blk 256x1 regs 16 smem 0.0 KB
  mapal_gen_eeadc0fd8a0f6131 x1 med 1.4 us grid 144x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: no trace
torch: 14490 launches, 58 distinct, median gap 3.3 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x9270 med 75.0 us grid 18432x1x1 blk 128x1 regs 16 smem 0.0 KB
  triton_poi_fused_view_2 x2032 med 21.8 us grid 1568x1x1 blk 128x1 regs 25 smem 0.0 KB
  triton_poi_fused_convolution_0 x1714 med 7.7 us grid 53x8x1 blk 256x1 regs 33 smem 16.4 KB
### rowsum_2048
mapal: 4 launches, 4 distinct, median gap 7.0 us
  mapal_tile_5cb606ed22540675 x1 med 291.0 us grid 16x16x1 blk 128x1 regs 242 smem 17.4 KB
  mapal_tile_96c6576a8cae5dc3 x1 med 15.0 us grid 256x1x1 blk 256x1 regs 17 smem 0.0 KB
  mapal_gen_0c2c9285a1cff51b x1 med 11.6 us grid 16384x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: 33 launches, 4 distinct, median gap 4.3 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x12 med 320.3 us grid 128x1x4 blk 256x1 regs 202 smem 50.2 KB
  gen_op(float *, long long, int, int, int, int, float) x9 med 12.9 us grid 16384x1x1 blk 256x1 regs 18 smem 0.0 KB
  std::enable_if<!T7, void>::type internal::gemvx::kernel<int, x6 med 6.4 us grid 64x1x1 blk 16x32 regs 128 smem 2.0 KB
torch: 2896 launches, 14 distinct, median gap 3.4 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x1854 med 67.9 us grid 18432x1x1 blk 128x1 regs 16 smem 0.0 KB
  triton_red_fused_sum_0 x963 med 26.7 us grid 1024x1x1 blk 512x1 regs 18 smem 0.0 KB
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x18 med 318.3 us grid 128x1x4 blk 256x1 regs 202 smem 50.2 KB
### rowsum_4096
mapal: 4 launches, 4 distinct, median gap 0.7 us
  mapal_tile_86cfda25b6d53a54 x1 med 2458.1 us grid 32x32x1 blk 128x1 regs 242 smem 17.4 KB
  mapal_tile_21b430b4fdc7cba8 x1 med 78.2 us grid 512x1x1 blk 256x1 regs 17 smem 0.0 KB
  mapal_gen_ee8f5451f4dd91de x1 med 63.3 us grid 65536x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: 33 launches, 4 distinct, median gap 3.2 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x12 med 2379.7 us grid 256x2x3 blk 256x1 regs 202 smem 50.2 KB
  gen_op(float *, long long, int, int, int, int, float) x9 med 65.5 us grid 65536x1x1 blk 256x1 regs 18 smem 0.0 KB
  row_sum(float *, const float *, int) x6 med 57.6 us grid 512x1x1 blk 256x1 regs 39 smem 0.0 KB
torch: 5076 launches, 16 distinct, median gap 1.2 us
  triton_mm x159 med 4274.7 us grid 16384x1x1 blk 64x1 regs 64 smem 4.1 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x3735 med 75.9 us grid 18432x1x1 blk 128x1 regs 16 smem 0.0 KB
  triton_red_fused_sum_0 x1069 med 90.6 us grid 4096x1x1 blk 512x1 regs 18 smem 0.0 KB
### rt_mlp2_2048
mapal: 54 launches, 12 distinct, median gap 0.7 us
  mapal_tile_c56cd5d3a2524dfa x8 med 322.2 us grid 16x16x1 blk 128x1 regs 224 smem 17.4 KB
  mapal_tile_4a874e6551089bac x8 med 316.3 us grid 16x16x1 blk 128x1 regs 222 smem 17.4 KB
  mapal_tile_cb8a431d29c1dd2d x8 med 313.3 us grid 16x16x1 blk 128x1 regs 222 smem 17.4 KB
lib: no trace
torch: 95045 launches, 49 distinct, median gap 3.3 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x59947 med 75.5 us grid 18432x1x1 blk 128x1 regs 16 smem 0.0 KB
  triton_mm_plus_mm x755 med 734.3 us grid 1024x1x1 blk 128x1 regs 128 smem 16.4 KB
  triton_mm x1304 med 244.5 us grid 4096x1x1 blk 64x1 regs 62 smem 4.1 KB
### rt_mlp2_1024
mapal: 54 launches, 12 distinct, median gap 0.7 us
  mapal_tile_e73cd2f736c7b83b x8 med 59.2 us grid 16x8x1 blk 256x1 regs 95 smem 78.8 KB
  mapal_tile_f9c30a534cfa34b9 x8 med 59.5 us grid 16x8x1 blk 256x1 regs 95 smem 78.8 KB
  mapal_tile_2cfa420c9fd353de x8 med 59.0 us grid 16x8x1 blk 256x1 regs 95 smem 78.8 KB
lib: no trace
torch: 71751 launches, 39 distinct, median gap 3.3 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x44084 med 75.5 us grid 18432x1x1 blk 128x1 regs 16 smem 0.0 KB
  void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_al x432 med 44.9 us grid 128x1x1 blk 128x1 regs 126 smem 31.7 KB
  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x64_1 x432 med 28.6 us grid 128x1x1 blk 128x1 regs 136 smem 75.8 KB
### mlp_2048_tf32
mapal: 5 launches, 5 distinct, median gap 4.2 us
  mapal_tile_6ad78253aeca5239 x1 med 251.2 us grid 16x16x1 blk 128x1 regs 198 smem 38.9 KB
  mapal_tile_99245f07d9410790 x1 med 248.0 us grid 16x16x1 blk 128x1 regs 198 smem 38.9 KB
  mapal_gen_d62743ecf7801ae8 x1 med 11.5 us grid 16384x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: no trace
torch: no trace
### mlp2_2048_tf32
mapal: 12 launches, 11 distinct, median gap 0.7 us
  mapal_tile_99245f07d9410790 x2 med 248.2 us grid 16x16x1 blk 128x1 regs 198 smem 38.9 KB
  mapal_tile_52c74331c9fc4a4d x1 med 257.9 us grid 16x16x1 blk 128x1 regs 198 smem 38.9 KB
  mapal_tile_05f6ca2f39f4a64c x1 med 251.7 us grid 16x16x1 blk 128x1 regs 198 smem 38.9 KB
lib: no trace
torch: no trace
### convmc_8_64_56_3_tf32
mapal: 3 launches, 3 distinct, median gap 40.4 us
  mapal_tile_ba7993a34747b588 x1 med 100.1 us grid 784x1x1 blk 64x1 regs 128 smem 13.3 KB
  mapal_gen_49b4bca503d6adb4 x1 med 5.6 us grid 6728x1x1 blk 256x1 regs 16 smem 0.0 KB
  mapal_gen_eeadc0fd8a0f6131 x1 med 1.4 us grid 144x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: no trace
torch: no trace
## rtx-5090
### decode_1024_256
mapal: 6146 launches, 8 distinct, median gap 2.0 us
  mapal_tile_e2f8ae468c059815 x2046 med 4.2 us grid 1x1x1 blk 256x1 regs 38 smem 0.0 KB
  mapal_tile_a0714d3d20b8101f x1024 med 4.8 us grid 1x512x1 blk 256x1 regs 16 smem 0.0 KB
  mapal_tile_647b5e2c7fca3543 x1024 med 2.7 us grid 4x128x1 blk 256x1 regs 16 smem 0.0 KB
lib: 57344 launches, 7 distinct, median gap 2.2 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_al x8192 med 4.9 us grid 32x1x4 blk 128x1 regs 128 smem 31.7 KB
  std::enable_if<!T7, void>::type internal::gemvx::kernel<int, x8192 med 2.2 us grid 32x1x1 blk 8x32 regs 230 smem 1.0 KB
  void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, fl x8192 med 2.1 us grid 8x64x1 blk 32x16 regs 46 smem 0.0 KB
torch: 146480 launches, 23 distinct, median gap 6.8 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x5356 med 55.8 us grid 24576x1x1 blk 128x1 regs 16 smem 0.0 KB
  void gemvNSP_kernel<float, float, float, float, (int)1, (int x18432 med 2.0 us grid 8x1x1 blk 32x24 regs 28 smem 3.1 KB
  void at::native::index_elementwise_kernel<(int)128, (int)4,  x18432 med 1.2 us grid 1x1x1 blk 128x1 regs 32 smem 0.0 KB
### convmc_8_64_56_3
mapal: 3 launches, 3 distinct, median gap 8.9 us
  mapal_tile_d68ad1f305d1b23e x1 med 56.4 us grid 784x1x1 blk 64x1 regs 109 smem 3.1 KB
  mapal_gen_e9853c9e84d433c5 x1 med 3.8 us grid 6728x1x1 blk 256x1 regs 17 smem 0.0 KB
  mapal_gen_a39aa1067d10fa40 x1 med 0.9 us grid 144x1x1 blk 256x1 regs 17 smem 0.0 KB
lib: no trace
torch: 7081 launches, 55 distinct, median gap 2.9 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x4326 med 57.0 us grid 24576x1x1 blk 128x1 regs 16 smem 0.0 KB
  triton_poi_fused_view_2 x548 med 15.7 us grid 3136x1x1 blk 128x1 regs 18 smem 0.0 KB
  void cudnn::cnn::conv2d_grouped_direct_kernel<(bool)0, (bool x2 med 2194.6 us grid 1568x1x1 blk 1024x1 regs 56 smem 0.0 KB
### rowsum_2048
mapal: 4 launches, 4 distinct, median gap 1.5 us
  mapal_tile_c59d5a05a86bc84f x1 med 261.3 us grid 338x1x1 blk 256x1 regs 128 smem 17.4 KB
  mapal_gen_213a43e4047e4b79 x1 med 8.3 us grid 16384x1x1 blk 256x1 regs 17 smem 0.0 KB
  mapal_gen_acc3728df88a4caa x1 med 8.3 us grid 16384x1x1 blk 256x1 regs 17 smem 0.0 KB
lib: 33 launches, 4 distinct, median gap 3.7 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x12 med 249.5 us grid 128x1x5 blk 256x1 regs 210 smem 50.2 KB
  gen_op(float *, long long, int, int, int, int, float) x9 med 8.5 us grid 16384x1x1 blk 256x1 regs 18 smem 0.0 KB
  std::enable_if<!T7, void>::type internal::gemvx::kernel<int, x6 med 5.8 us grid 512x1x1 blk 32x4 regs 228 smem 1.0 KB
torch: 88 launches, 13 distinct, median gap 2.1 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x18 med 249.8 us grid 128x1x5 blk 256x1 regs 210 smem 50.2 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x6 med 143.3 us grid 65536x1x1 blk 128x1 regs 16 smem 0.0 KB
  void at::native::vectorized_elementwise_kernel<(int)2, at::n x15 med 12.7 us grid 4096x1x1 blk 128x1 regs 40 smem 0.0 KB
### rowsum_4096
mapal: 4 launches, 4 distinct, median gap 1.4 us
  mapal_tile_6eab42dba437b31e x1 med 2011.8 us grid 340x1x1 blk 256x1 regs 128 smem 17.4 KB
  mapal_tile_3c18479013e9e014 x1 med 36.5 us grid 4096x1x1 blk 256x1 regs 16 smem 1.0 KB
  mapal_gen_37ef5449006a2def x1 med 31.6 us grid 65536x1x1 blk 256x1 regs 17 smem 0.0 KB
lib: 33 launches, 4 distinct, median gap 2.6 us
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x12 med 1903.1 us grid 256x2x4 blk 256x1 regs 210 smem 50.2 KB
  gen_op(float *, long long, int, int, int, int, float) x9 med 32.7 us grid 65536x1x1 blk 256x1 regs 18 smem 0.0 KB
  std::enable_if<!T7, void>::type internal::gemvx::kernel<int, x6 med 15.6 us grid 128x1x1 blk 16x32 regs 128 smem 2.0 KB
torch: 5158 launches, 16 distinct, median gap 2.2 us
  triton_mm x198 med 2670.1 us grid 16384x1x1 blk 64x1 regs 64 smem 4.1 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x3776 med 57.7 us grid 24576x1x1 blk 128x1 regs 16 smem 0.0 KB
  void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a x36 med 1916.7 us grid 256x2x4 blk 256x1 regs 210 smem 50.2 KB
### rt_mlp2_2048
mapal: 54 launches, 12 distinct, median gap 1.9 us
  mapal_tile_d2d7996f7d9568ae x8 med 290.1 us grid 16x16x1 blk 256x1 regs 125 smem 17.4 KB
  mapal_tile_ded6aef2041e4d09 x8 med 279.0 us grid 16x16x1 blk 256x1 regs 124 smem 17.4 KB
  mapal_tile_997700f4fb01290b x8 med 275.7 us grid 16x16x1 blk 256x1 regs 125 smem 17.4 KB
lib: no trace
torch: 96362 launches, 49 distinct, median gap 2.4 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x60729 med 57.8 us grid 24576x1x1 blk 128x1 regs 16 smem 0.0 KB
  triton_mm_plus_mm x971 med 520.4 us grid 1024x1x1 blk 128x1 regs 128 smem 16.4 KB
  triton_mm x1442 med 214.1 us grid 4096x1x1 blk 64x1 regs 60 smem 4.1 KB
### rt_mlp2_1024
mapal: 54 launches, 12 distinct, median gap 1.6 us
  mapal_tile_1abf6322f3a55baa x8 med 53.7 us grid 16x16x1 blk 256x1 regs 79 smem 8.2 KB
  mapal_tile_02f67bac8399ecdc x8 med 53.5 us grid 16x16x1 blk 256x1 regs 79 smem 8.2 KB
  mapal_tile_0d148ea39fc588dd x8 med 53.3 us grid 16x16x1 blk 256x1 regs 79 smem 8.2 KB
lib: no trace
torch: 55518 launches, 37 distinct, median gap 2.4 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x33372 med 56.4 us grid 24576x1x1 blk 128x1 regs 16 smem 0.0 KB
  void cutlass::Kernel2<cutlass_80_simt_sgemm_128x64_8x5_nn_al x567 med 41.0 us grid 128x1x5 blk 128x1 regs 128 smem 31.7 KB
  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_64x128_3 x567 med 25.9 us grid 64x2x1 blk 128x1 regs 168 smem 75.8 KB
### mlp_2048_tf32
mapal: 5 launches, 5 distinct, median gap 1.9 us
  mapal_tile_57d267aa93b75216 x1 med 219.0 us grid 16x16x1 blk 256x1 regs 122 smem 38.9 KB
  mapal_tile_794728a1e1bcc3a5 x1 med 217.4 us grid 16x16x1 blk 256x1 regs 122 smem 38.9 KB
  mapal_gen_6f89aa19c188a660 x1 med 8.3 us grid 16384x1x1 blk 256x1 regs 17 smem 0.0 KB
lib: no trace
torch: no trace
### mlp2_2048_tf32
mapal: 12 launches, 11 distinct, median gap 1.9 us
  mapal_tile_794728a1e1bcc3a5 x2 med 216.2 us grid 16x16x1 blk 256x1 regs 122 smem 38.9 KB
  mapal_tile_6f78db313013e467 x1 med 218.1 us grid 16x16x1 blk 256x1 regs 122 smem 38.9 KB
  mapal_tile_a0d5f350c1f6a692 x1 med 217.4 us grid 16x16x1 blk 256x1 regs 122 smem 38.9 KB
lib: no trace
torch: no trace
### convmc_8_64_56_3_tf32
mapal: 3 launches, 3 distinct, median gap 9.7 us
  mapal_tile_5271b54b9cb9511d x1 med 56.1 us grid 784x1x1 blk 64x1 regs 109 smem 3.1 KB
  mapal_gen_e9853c9e84d433c5 x1 med 3.8 us grid 6728x1x1 blk 256x1 regs 17 smem 0.0 KB
  mapal_gen_a39aa1067d10fa40 x1 med 0.9 us grid 144x1x1 blk 256x1 regs 17 smem 0.0 KB
lib: no trace
torch: no trace
## h100-pcie
### decode_1024_256
mapal: 7170 launches, 9 distinct, median gap 1.3 us
  mapal_tile_3881eb4bbf2d9cb5 x2046 med 12.9 us grid 1x1x1 blk 256x1 regs 20 smem 0.0 KB
  mapal_tile_a21805fea156cb5e x1024 med 7.9 us grid 1x1x1 blk 256x1 regs 31 smem 0.0 KB
  mapal_tile_e5ca811c8e525f6d x1024 med 3.5 us grid 4x128x1 blk 256x1 regs 16 smem 0.0 KB
lib: 49152 launches, 5 distinct, median gap 1.4 us
  sm80_xmma_gemm_f32f32_f32f32_f32_nn_n_tilesize32x32x8_stage3 x8192 med 11.8 us grid 32x8x1 blk 64x1 regs 80 smem 7.2 KB
  std::enable_if<!T7, void>::type internal::gemvx::kernel<int, x16384 med 2.9 us grid 64x1x1 blk 4x32 regs 164 smem 1.0 KB
  std::enable_if<!T7, void>::type internal::gemvx::kernel<int, x8192 med 3.2 us grid 256x1x1 blk 32x4 regs 159 smem 1.0 KB
torch: 146168 launches, 22 distinct, median gap 8.7 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x5150 med 27.0 us grid 12800x1x1 blk 128x1 regs 16 smem 0.0 KB
  std::enable_if<!T7, void>::type internal::gemvx::kernel<int, x27648 med 2.8 us grid 64x1x1 blk 4x32 regs 164 smem 1.0 KB
  void at::native::<unnamed>::indexSelectSmallIndex<float, lon x18432 med 1.9 us grid 2x1x1 blk 128x1 regs 24 smem 0.0 KB
### convmc_8_64_56_3
mapal: 3 launches, 3 distinct, median gap 22.6 us
  mapal_tile_fd559abd135110ac x1 med 86.6 us grid 392x1x1 blk 64x1 regs 161 smem 4.1 KB
  mapal_gen_872cc1405acca671 x1 med 5.8 us grid 6728x1x1 blk 256x1 regs 16 smem 0.0 KB
  mapal_gen_8143fc663fe67f4e x1 med 1.4 us grid 144x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: no trace
torch: 8798 launches, 47 distinct, median gap 4.9 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x5562 med 27.2 us grid 12800x1x1 blk 128x1 regs 16 smem 0.0 KB
  void cudnn::cnn::conv2d_grouped_direct_kernel<(bool)0, (bool x4 med 5160.7 us grid 1568x1x1 blk 1024x1 regs 40 smem 0.0 KB
  triton_poi_fused_view_2 x866 med 16.1 us grid 1568x1x1 blk 256x1 regs 18 smem 0.0 KB
### rowsum_2048
mapal: 4 launches, 4 distinct, median gap 1.8 us
  mapal_tile_844dbc85f2c69fb4 x1 med 613.0 us grid 16x16x1 blk 256x1 regs 127 smem 17.4 KB
  mapal_gen_320e89c051337355 x1 med 12.2 us grid 16384x1x1 blk 256x1 regs 16 smem 0.0 KB
  mapal_gen_bd1ee8056f37ef5c x1 med 12.1 us grid 16384x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: 33 launches, 4 distinct, median gap 5.8 us
  sm80_xmma_gemm_f32f32_f32f32_f32_nn_n_tilesize128x128x8_stag x12 med 510.0 us grid 16x16x4 blk 128x1 regs 255 smem 34.8 KB
  gen_op(float *, long long, int, int, int, int, float) x9 med 16.3 us grid 16384x1x1 blk 256x1 regs 18 smem 0.0 KB
  std::enable_if<!T7, void>::type internal::gemvx::kernel<int, x6 med 9.5 us grid 512x1x1 blk 16x4 regs 165 smem 0.0 KB
torch: 2272 launches, 14 distinct, median gap 4.9 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x1442 med 26.1 us grid 12800x1x1 blk 128x1 regs 16 smem 0.0 KB
  triton_red_fused_sum_0 x751 med 12.9 us grid 2048x1x1 blk 512x1 regs 16 smem 0.0 KB
  sm80_xmma_gemm_f32f32_f32f32_f32_nn_n_tilesize128x128x8_stag x18 med 523.1 us grid 16x16x4 blk 128x1 regs 255 smem 34.8 KB
### rowsum_4096
mapal: 4 launches, 4 distinct, median gap 1.3 us
  mapal_tile_eb50b87a44f46aa3 x1 med 3564.5 us grid 32x32x1 blk 256x1 regs 127 smem 17.4 KB
  mapal_gen_7b59dfa488c6072f x1 med 49.9 us grid 65536x1x1 blk 256x1 regs 16 smem 0.0 KB
  mapal_gen_02743d924bf99060 x1 med 45.9 us grid 65536x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: 33 launches, 4 distinct, median gap 1.9 us
  sm80_xmma_gemm_f32f32_f32f32_f32_nn_n_tilesize128x128x8_stag x12 med 3664.2 us grid 32x32x1 blk 128x1 regs 255 smem 34.8 KB
  gen_op(float *, long long, int, int, int, int, float) x9 med 61.5 us grid 65536x1x1 blk 256x1 regs 18 smem 0.0 KB
  row_sum(float *, const float *, int) x6 med 42.0 us grid 512x1x1 blk 256x1 regs 31 smem 0.0 KB
torch: 6528 launches, 16 distinct, median gap 4.4 us
  triton_mm x164 med 4843.6 us grid 16384x1x1 blk 64x1 regs 64 smem 4.1 KB
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x4760 med 27.2 us grid 12800x1x1 blk 128x1 regs 16 smem 0.0 KB
  sm80_xmma_gemm_f32f32_f32f32_f32_nn_n_tilesize128x128x8_stag x30 med 3670.6 us grid 32x32x1 blk 128x1 regs 255 smem 34.8 KB
### rt_mlp2_2048
mapal: 54 launches, 12 distinct, median gap 1.6 us
  mapal_tile_5c213072c31419c7 x8 med 649.2 us grid 16x16x1 blk 256x1 regs 122 smem 17.4 KB
  mapal_tile_9bd2216916c37eb8 x8 med 632.7 us grid 16x16x1 blk 256x1 regs 123 smem 17.4 KB
  mapal_tile_33e054128d1b6cf1 x8 med 632.0 us grid 16x16x1 blk 256x1 regs 123 smem 17.4 KB
lib: no trace
torch: 102854 launches, 45 distinct, median gap 4.9 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x65323 med 27.0 us grid 12800x1x1 blk 128x1 regs 16 smem 0.0 KB
  triton_mm_plus_mm x565 med 1044.9 us grid 1024x1x1 blk 128x1 regs 128 smem 16.4 KB
  triton_mm x977 med 386.2 us grid 4096x1x1 blk 64x1 regs 54 smem 4.1 KB
### rt_mlp2_1024
mapal: 54 launches, 12 distinct, median gap 1.9 us
  mapal_tile_c6b617093d2e536e x8 med 133.0 us grid 16x8x1 blk 128x1 regs 127 smem 6.1 KB
  mapal_tile_97a601e6c5b1d580 x8 med 124.7 us grid 16x8x1 blk 128x1 regs 127 smem 6.1 KB
  mapal_tile_1b5dcdf3169ff659 x8 med 124.5 us grid 16x8x1 blk 128x1 regs 127 smem 6.1 KB
lib: no trace
torch: 46488 launches, 36 distinct, median gap 4.9 us
  void at::native::vectorized_elementwise_kernel<(int)4, at::n x27398 med 27.1 us grid 12800x1x1 blk 128x1 regs 16 smem 0.0 KB
  sm80_xmma_gemm_f32f32_f32f32_f32_nn_n_tilesize64x64x8_stage3 x576 med 79.6 us grid 16x16x4 blk 128x1 regs 121 smem 13.3 KB
  sm90_xmma_gemm_f32f32_tf32f32_f32_nn_n_tilesize128x128x32_wa x576 med 19.6 us grid 64x1x1 blk 384x1 regs 168 smem 236.5 KB
### mlp_2048_tf32
mapal: 5 launches, 5 distinct, median gap 1.6 us
  mapal_tile_f1d5c6e29ab8fa35 x1 med 175.8 us grid 16x8x1 blk 256x1 regs 212 smem 205.8 KB
  mapal_tile_75337305197b2ea6 x1 med 172.9 us grid 16x8x1 blk 256x1 regs 254 smem 205.8 KB
  mapal_gen_504d11cf835d20ee x1 med 12.8 us grid 16384x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: no trace
torch: no trace
### mlp2_2048_tf32
mapal: 12 launches, 11 distinct, median gap 1.4 us
  mapal_tile_f1d5c6e29ab8fa35 x2 med 173.6 us grid 16x8x1 blk 256x1 regs 212 smem 205.8 KB
  mapal_tile_33288dfc9436bbf7 x1 med 181.0 us grid 16x8x1 blk 256x1 regs 210 smem 205.8 KB
  mapal_tile_9f6ec26b1607e7a2 x1 med 180.9 us grid 16x8x1 blk 256x1 regs 210 smem 205.8 KB
lib: no trace
torch: no trace
### convmc_8_64_56_3_tf32
mapal: 3 launches, 3 distinct, median gap 27.9 us
  mapal_tile_790082359021fedd x1 med 87.4 us grid 392x1x1 blk 64x1 regs 161 smem 4.1 KB
  mapal_gen_872cc1405acca671 x1 med 5.8 us grid 6728x1x1 blk 256x1 regs 16 smem 0.0 KB
  mapal_gen_8143fc663fe67f4e x1 med 1.4 us grid 144x1x1 blk 256x1 regs 16 smem 0.0 KB
lib: no trace
torch: no trace
## rtx-3090: Compressed file ended before the end-of-stream marker was reached
