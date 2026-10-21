	.file	"host.emitted.ll"
	.section	.rodata.cst16,"aM",@progbits,16
	.p2align	4, 0x0                          # -- Begin function main
.LCPI0_0:
	.long	4294967246                      # 0xffffffce
	.long	4294967246                      # 0xffffffce
	.zero	4
	.zero	4
.LCPI0_1:
	.long	0x3c800000                      # float 0.015625
	.long	0x3c800000                      # float 0.015625
	.long	0x3c800000                      # float 0.015625
	.long	0x3c800000                      # float 0.015625
.LCPI0_2:
	.quad	4                               # 0x4
	.quad	5                               # 0x5
.LCPI0_3:
	.quad	6                               # 0x6
	.quad	7                               # 0x7
.LCPI0_5:
	.long	4042322161                      # 0xf0f0f0f1
	.long	4042322161                      # 0xf0f0f0f1
	.long	4042322161                      # 0xf0f0f0f1
	.long	4042322161                      # 0xf0f0f0f1
.LCPI0_6:
	.long	4294967280                      # 0xfffffff0
	.long	4294967280                      # 0xfffffff0
	.long	4294967280                      # 0xfffffff0
	.long	4294967280                      # 0xfffffff0
.LCPI0_7:
	.zero	16,252
.LCPI0_8:
	.byte	3                               # 0x3
	.byte	3                               # 0x3
	.byte	3                               # 0x3
	.byte	3                               # 0x3
	.zero	1
	.zero	1
	.zero	1
	.zero	1
	.zero	1
	.zero	1
	.zero	1
	.zero	1
	.zero	1
	.zero	1
	.zero	1
	.zero	1
.LCPI0_9:
	.byte	241                             # 0xf1
	.byte	0                               # 0x0
	.byte	241                             # 0xf1
	.byte	0                               # 0x0
	.byte	241                             # 0xf1
	.byte	0                               # 0x0
	.byte	241                             # 0xf1
	.byte	0                               # 0x0
	.byte	241                             # 0xf1
	.byte	0                               # 0x0
	.byte	241                             # 0xf1
	.byte	0                               # 0x0
	.byte	241                             # 0xf1
	.byte	0                               # 0x0
	.byte	241                             # 0xf1
	.byte	0                               # 0x0
.LCPI0_10:
	.zero	16,15
.LCPI0_11:
	.byte	248                             # 0xf8
	.byte	248                             # 0xf8
	.byte	248                             # 0xf8
	.byte	248                             # 0xf8
	.zero	1
	.zero	1
	.zero	1
	.zero	1
	.zero	1
	.zero	1
	.zero	1
	.zero	1
	.zero	1
	.zero	1
	.zero	1
	.zero	1
.LCPI0_12:
	.long	0x3b800000                      # float 0.00390625
	.long	0x3b800000                      # float 0.00390625
	.long	0x3b800000                      # float 0.00390625
	.long	0x3b800000                      # float 0.00390625
	.section	.rodata.cst4,"aM",@progbits,4
	.p2align	2, 0x0
.LCPI0_4:
	.long	0x3c800000                      # float 0.015625
	.text
	.globl	main
	.p2align	4
	.type	main,@function
main:                                   # @main
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$1432, %rsp                     # imm = 0x598
	.cfi_def_cfa_offset 1488
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movl	$1, %edi
	callq	mapal_gpu_init@PLT
	movq	$0, 136(%rsp)
	movq	$0, 128(%rsp)
	movq	$0, 120(%rsp)
	movq	$0, 112(%rsp)
	movq	$0, 104(%rsp)
	movq	$0, 96(%rsp)
	movq	$0, 88(%rsp)
	movq	$0, 32(%rsp)
	movq	$0, 80(%rsp)
	movq	$0, 72(%rsp)
	movq	$0, 64(%rsp)
	movq	$0, 56(%rsp)
	movq	$0, 48(%rsp)
	movq	$0, 24(%rsp)
	movq	$0, 40(%rsp)
	movq	$0, 168(%rsp)
	leaq	136(%rsp), %rdi
	movl	$256, %esi                      # imm = 0x100
	callq	mapal_gpu_alloc@PLT
	movabsq	$570417511890179569, %rdi       # imm = 0x7EA87B471E055F1
	movl	$64, %edx
	movq	%rax, %rsi
	callq	mapal_gpu_gen@PLT
	leaq	128(%rsp), %rdi
	movl	$512, %esi                      # imm = 0x200
	callq	mapal_gpu_alloc@PLT
	movabsq	$-1791989716614967237, %rdi     # imm = 0xE72192D8DC03103B
	movl	$128, %edx
	movq	%rax, %rsi
	callq	mapal_gpu_gen@PLT
	leaq	120(%rsp), %rdi
	movl	$512, %esi                      # imm = 0x200
	callq	mapal_gpu_alloc@PLT
	movabsq	$-4223766215135587374, %rdi     # imm = 0xC5622B017C8C13D2
	movl	$128, %edx
	movq	%rax, %rsi
	callq	mapal_gpu_gen@PLT
	leaq	112(%rsp), %rdi
	movl	$256, %esi                      # imm = 0x100
	callq	mapal_gpu_alloc@PLT
	movabsq	$-9196861651627557674, %rdi     # imm = 0x805E2F0E0E53BCD6
	movl	$64, %edx
	movq	%rax, %rsi
	callq	mapal_gpu_gen@PLT
	leaq	104(%rsp), %rdi
	movl	$256, %esi                      # imm = 0x100
	callq	mapal_gpu_alloc@PLT
	movabsq	$5761537033769727445, %rdi      # imm = 0x4FF51768021A35D5
	movl	$64, %edx
	movq	%rax, %rsi
	callq	mapal_gpu_gen@PLT
	leaq	96(%rsp), %rdi
	movl	$64, %esi
	callq	mapal_gpu_alloc@PLT
	movabsq	$1396038385954328191, %rdi      # imm = 0x135FB983F79E0E7F
	movl	$16, %edx
	movq	%rax, %rsi
	callq	mapal_gpu_gen@PLT
	leaq	88(%rsp), %rdi
	movl	$32, %esi
	callq	mapal_gpu_alloc@PLT
	movabsq	$-3200766962839905091, %rdi     # imm = 0xD3949767DB1F1CBD
	movl	$8, %edx
	movq	%rax, %rsi
	callq	mapal_gpu_gen@PLT
	leaq	.L.str0(%rip), %rdi
	xorl	%r15d, %r15d
	movl	$5, %esi
	xorl	%edx, %edx
	callq	mapal_print_str@PLT
	movl	$-1, %edi
	movl	$1, %esi
	callq	mapal_print_i32@PLT
	leaq	.L.str1(%rip), %rdi
	movl	$1, %esi
	movl	$1, %edx
	callq	mapal_print_str@PLT
	xorps	%xmm0, %xmm0
	movss	%xmm0, 12(%rsp)                 # 4-byte Spill
	movss	%xmm0, 8(%rsp)                  # 4-byte Spill
	movl	$0, %ebx
	movl	$0, %ecx
	movl	$0, %r14d
	movss	%xmm0, 20(%rsp)                 # 4-byte Spill
	movss	%xmm0, 16(%rsp)                 # 4-byte Spill
	xorps	%xmm11, %xmm11
	xorps	%xmm0, %xmm0
	xorps	%xmm1, %xmm1
	xorps	%xmm2, %xmm2
	xorps	%xmm4, %xmm4
	xorps	%xmm5, %xmm5
	xorps	%xmm6, %xmm6
	xorps	%xmm7, %xmm7
	xorps	%xmm8, %xmm8
	xorps	%xmm9, %xmm9
	xorps	%xmm10, %xmm10
	xorps	%xmm12, %xmm12
	xorps	%xmm13, %xmm13
	xorps	%xmm14, %xmm14
	xorps	%xmm15, %xmm15
	movaps	%xmm15, 464(%rsp)               # 16-byte Spill
	movaps	%xmm15, 448(%rsp)               # 16-byte Spill
	movaps	%xmm15, 432(%rsp)               # 16-byte Spill
	movaps	%xmm15, 416(%rsp)               # 16-byte Spill
	movaps	%xmm15, 400(%rsp)               # 16-byte Spill
	movaps	%xmm15, 384(%rsp)               # 16-byte Spill
	movaps	%xmm15, 368(%rsp)               # 16-byte Spill
	movaps	%xmm15, 352(%rsp)               # 16-byte Spill
	movaps	%xmm15, 336(%rsp)               # 16-byte Spill
	movaps	%xmm15, 320(%rsp)               # 16-byte Spill
	movaps	%xmm15, 304(%rsp)               # 16-byte Spill
	movaps	%xmm15, 288(%rsp)               # 16-byte Spill
	movaps	%xmm15, 272(%rsp)               # 16-byte Spill
	movaps	%xmm15, 256(%rsp)               # 16-byte Spill
	movaps	%xmm15, 240(%rsp)               # 16-byte Spill
	movaps	%xmm15, 224(%rsp)               # 16-byte Spill
	movaps	%xmm15, 208(%rsp)               # 16-byte Spill
	movaps	%xmm15, 192(%rsp)               # 16-byte Spill
	jmp	.LBB0_1
	.p2align	4
.LBB0_3:                                # %bb1034.i
                                        #   in Loop: Header=BB0_1 Depth=1
	addq	%rbx, %rax
	leaq	(%rsp,%rbx), %rsi
	addq	$640, %rsi                      # imm = 0x280
	movl	$32, %edx
	movq	%rax, %rdi
.LBB0_4:                                # %bb1036.i
                                        #   in Loop: Header=BB0_1 Depth=1
	callq	mapal_gpu_h2d@PLT
	movaps	640(%rsp), %xmm0
	movaps	%xmm0, 1360(%rsp)               # 16-byte Spill
	movaps	656(%rsp), %xmm0
	movaps	%xmm0, 1344(%rsp)               # 16-byte Spill
	movaps	672(%rsp), %xmm0
	movaps	%xmm0, 1328(%rsp)               # 16-byte Spill
	movaps	688(%rsp), %xmm0
	movaps	%xmm0, 1312(%rsp)               # 16-byte Spill
	movaps	704(%rsp), %xmm0
	movaps	%xmm0, 1296(%rsp)               # 16-byte Spill
	movaps	720(%rsp), %xmm0
	movaps	%xmm0, 1280(%rsp)               # 16-byte Spill
	movaps	736(%rsp), %xmm0
	movaps	%xmm0, 1264(%rsp)               # 16-byte Spill
	movaps	752(%rsp), %xmm0
	movaps	%xmm0, 1248(%rsp)               # 16-byte Spill
	movaps	768(%rsp), %xmm0
	movaps	%xmm0, 1232(%rsp)               # 16-byte Spill
	movaps	784(%rsp), %xmm0
	movaps	%xmm0, 1216(%rsp)               # 16-byte Spill
	movaps	800(%rsp), %xmm0
	movaps	%xmm0, 1200(%rsp)               # 16-byte Spill
	movaps	816(%rsp), %xmm0
	movaps	%xmm0, 1184(%rsp)               # 16-byte Spill
	movaps	832(%rsp), %xmm0
	movaps	%xmm0, 1168(%rsp)               # 16-byte Spill
	movaps	848(%rsp), %xmm0
	movaps	%xmm0, 464(%rsp)                # 16-byte Spill
	movaps	864(%rsp), %xmm0
	movaps	%xmm0, 448(%rsp)                # 16-byte Spill
	movaps	880(%rsp), %xmm0
	movaps	%xmm0, 432(%rsp)                # 16-byte Spill
	movq	32(%rsp), %rbp
	movq	112(%rsp), %r14
	movaps	896(%rsp), %xmm0
	movaps	%xmm0, 416(%rsp)                # 16-byte Spill
	movaps	912(%rsp), %xmm0
	movaps	%xmm0, 400(%rsp)                # 16-byte Spill
	movaps	928(%rsp), %xmm0
	movaps	%xmm0, 384(%rsp)                # 16-byte Spill
	movaps	944(%rsp), %xmm0
	movaps	%xmm0, 368(%rsp)                # 16-byte Spill
	movaps	960(%rsp), %xmm0
	movaps	%xmm0, 352(%rsp)                # 16-byte Spill
	movaps	976(%rsp), %xmm0
	movaps	%xmm0, 336(%rsp)                # 16-byte Spill
	movaps	992(%rsp), %xmm0
	movaps	%xmm0, 320(%rsp)                # 16-byte Spill
	movaps	1008(%rsp), %xmm0
	movaps	%xmm0, 304(%rsp)                # 16-byte Spill
	movaps	1024(%rsp), %xmm0
	movaps	%xmm0, 288(%rsp)                # 16-byte Spill
	movaps	1040(%rsp), %xmm0
	movaps	%xmm0, 272(%rsp)                # 16-byte Spill
	movaps	1056(%rsp), %xmm0
	movaps	%xmm0, 256(%rsp)                # 16-byte Spill
	movaps	1072(%rsp), %xmm0
	movaps	%xmm0, 240(%rsp)                # 16-byte Spill
	movaps	1088(%rsp), %xmm0
	movaps	%xmm0, 224(%rsp)                # 16-byte Spill
	movaps	1104(%rsp), %xmm0
	movaps	%xmm0, 208(%rsp)                # 16-byte Spill
	movaps	1120(%rsp), %xmm0
	movaps	%xmm0, 192(%rsp)                # 16-byte Spill
	movaps	1136(%rsp), %xmm0
	movaps	%xmm0, 1152(%rsp)               # 16-byte Spill
	movl	$512, %esi                      # imm = 0x200
	leaq	80(%rsp), %rdi
	callq	mapal_gpu_alloc@PLT
	movabsq	$-6055507678851317309, %rbx     # imm = 0xABF683D97B6AC1C3
	movq	%rbx, %rdi
	movq	%rbp, %rsi
	movq	%r14, %rdx
	movq	%rax, %rcx
	movl	$1, %r8d
	movl	$1, %r9d
	pushq	$0
	.cfi_adjust_cfa_offset 8
	pushq	$256                            # imm = 0x100
	.cfi_adjust_cfa_offset 8
	callq	mapal_gpu_launch@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
	movq	32(%rsp), %r14
	movq	104(%rsp), %rbp
	movl	$512, %esi                      # imm = 0x200
	leaq	72(%rsp), %rdi
	callq	mapal_gpu_alloc@PLT
	movq	%rbx, %rdi
	movq	%r14, %rsi
	movq	%rbp, %rdx
	movq	%rax, %rcx
	movl	$1, %r8d
	movl	$1, %r9d
	pushq	$0
	.cfi_adjust_cfa_offset 8
	pushq	$256                            # imm = 0x100
	.cfi_adjust_cfa_offset 8
	callq	mapal_gpu_launch@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
	movdqa	144(%rsp), %xmm10               # 16-byte Reload
	movdqa	%xmm10, %xmm0
	movdqa	.LCPI0_5(%rip), %xmm3           # xmm3 = [4042322161,4042322161,4042322161,4042322161]
	pmuludq	%xmm3, %xmm0
	pshufd	$237, %xmm0, %xmm0              # xmm0 = xmm0[1,3,2,3]
	pshufd	$245, %xmm10, %xmm1             # xmm1 = xmm10[1,1,3,3]
	pmuludq	%xmm3, %xmm1
	pshufd	$237, %xmm1, %xmm1              # xmm1 = xmm1[1,3,2,3]
	punpckldq	%xmm1, %xmm0            # xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1]
	movdqa	%xmm0, %xmm1
	movdqa	.LCPI0_6(%rip), %xmm4           # xmm4 = [4294967280,4294967280,4294967280,4294967280]
	pand	%xmm4, %xmm1
	psrld	$4, %xmm0
	paddd	%xmm1, %xmm0
	psubd	%xmm0, %xmm10
	packuswb	%xmm10, %xmm10
	packuswb	%xmm10, %xmm10
	movdqa	%xmm10, %xmm0
	psllw	$2, %xmm0
	movdqa	.LCPI0_7(%rip), %xmm5           # xmm5 = [252,252,252,252,252,252,252,252,252,252,252,252,252,252,252,252]
	pand	%xmm5, %xmm0
	movdqa	.LCPI0_8(%rip), %xmm6           # xmm6 = [3,3,3,3,u,u,u,u,u,u,u,u,u,u,u,u]
	paddb	%xmm6, %xmm10
	paddb	%xmm0, %xmm10
	movdqa	%xmm10, %xmm0
	xorps	%xmm2, %xmm2
	punpckhbw	%xmm2, %xmm0            # xmm0 = xmm0[8],xmm2[8],xmm0[9],xmm2[9],xmm0[10],xmm2[10],xmm0[11],xmm2[11],xmm0[12],xmm2[12],xmm0[13],xmm2[13],xmm0[14],xmm2[14],xmm0[15],xmm2[15]
	movdqa	.LCPI0_9(%rip), %xmm7           # xmm7 = [241,0,241,0,241,0,241,0,241,0,241,0,241,0,241,0]
	pmullw	%xmm7, %xmm0
	psrlw	$8, %xmm0
	movdqa	%xmm10, %xmm1
	punpcklbw	%xmm2, %xmm1            # xmm1 = xmm1[0],xmm2[0],xmm1[1],xmm2[1],xmm1[2],xmm2[2],xmm1[3],xmm2[3],xmm1[4],xmm2[4],xmm1[5],xmm2[5],xmm1[6],xmm2[6],xmm1[7],xmm2[7]
	pmullw	%xmm7, %xmm1
	psrlw	$8, %xmm1
	packuswb	%xmm0, %xmm1
	psrlw	$4, %xmm1
	movdqa	.LCPI0_10(%rip), %xmm8          # xmm8 = [15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15]
	pand	%xmm8, %xmm1
	movdqa	%xmm1, %xmm0
	psllw	$4, %xmm0
	por	%xmm1, %xmm0
	psubb	%xmm0, %xmm10
	movdqa	.LCPI0_11(%rip), %xmm9          # xmm9 = [248,248,248,248,u,u,u,u,u,u,u,u,u,u,u,u]
	paddb	%xmm9, %xmm10
	punpcklbw	%xmm10, %xmm0           # xmm0 = xmm0[0],xmm10[0],xmm0[1],xmm10[1],xmm0[2],xmm10[2],xmm0[3],xmm10[3],xmm0[4],xmm10[4],xmm0[5],xmm10[5],xmm0[6],xmm10[6],xmm0[7],xmm10[7]
	punpcklwd	%xmm0, %xmm0            # xmm0 = xmm0[0,0,1,1,2,2,3,3]
	psrad	$24, %xmm0
	cvtdq2ps	%xmm0, %xmm0
	movaps	.LCPI0_12(%rip), %xmm10         # xmm10 = [3.90625E-3,3.90625E-3,3.90625E-3,3.90625E-3]
	mulps	%xmm10, %xmm0
	movaps	%xmm0, 1392(%rsp)
	movdqa	1376(%rsp), %xmm11              # 16-byte Reload
	movdqa	%xmm11, %xmm0
	pmuludq	%xmm3, %xmm0
	pshufd	$237, %xmm0, %xmm0              # xmm0 = xmm0[1,3,2,3]
	pshufd	$245, %xmm11, %xmm1             # xmm1 = xmm11[1,1,3,3]
	pmuludq	%xmm3, %xmm1
	pshufd	$237, %xmm1, %xmm1              # xmm1 = xmm1[1,3,2,3]
	punpckldq	%xmm1, %xmm0            # xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1]
	movdqa	%xmm0, %xmm1
	pand	%xmm4, %xmm1
	psrld	$4, %xmm0
	paddd	%xmm1, %xmm0
	psubd	%xmm0, %xmm11
	packuswb	%xmm11, %xmm11
	packuswb	%xmm11, %xmm11
	movdqa	%xmm11, %xmm0
	psllw	$2, %xmm0
	pand	%xmm5, %xmm0
	paddb	%xmm6, %xmm11
	paddb	%xmm0, %xmm11
	movdqa	%xmm11, %xmm0
	punpckhbw	%xmm2, %xmm0            # xmm0 = xmm0[8],xmm2[8],xmm0[9],xmm2[9],xmm0[10],xmm2[10],xmm0[11],xmm2[11],xmm0[12],xmm2[12],xmm0[13],xmm2[13],xmm0[14],xmm2[14],xmm0[15],xmm2[15]
	pmullw	%xmm7, %xmm0
	psrlw	$8, %xmm0
	movdqa	%xmm11, %xmm1
	punpcklbw	%xmm2, %xmm1            # xmm1 = xmm1[0],xmm2[0],xmm1[1],xmm2[1],xmm1[2],xmm2[2],xmm1[3],xmm2[3],xmm1[4],xmm2[4],xmm1[5],xmm2[5],xmm1[6],xmm2[6],xmm1[7],xmm2[7]
	pmullw	%xmm7, %xmm1
	psrlw	$8, %xmm1
	packuswb	%xmm0, %xmm1
	psrlw	$4, %xmm1
	pand	%xmm8, %xmm1
	movdqa	%xmm1, %xmm0
	psllw	$4, %xmm0
	por	%xmm1, %xmm0
	psubb	%xmm0, %xmm11
	paddb	%xmm9, %xmm11
	punpcklbw	%xmm11, %xmm0           # xmm0 = xmm0[0],xmm11[0],xmm0[1],xmm11[1],xmm0[2],xmm11[2],xmm0[3],xmm11[3],xmm0[4],xmm11[4],xmm0[5],xmm11[5],xmm0[6],xmm11[6],xmm0[7],xmm11[7]
	punpcklwd	%xmm0, %xmm0            # xmm0 = xmm0[0,0,1,1,2,2,3,3]
	psrad	$24, %xmm0
	cvtdq2ps	%xmm0, %xmm0
	mulps	%xmm10, %xmm0
	movaps	%xmm0, 1408(%rsp)
	movl	$32, %esi
	leaq	64(%rsp), %rdi
	callq	mapal_gpu_alloc@PLT
	movl	$32, %edx
	movq	%rax, %rdi
	leaq	1392(%rsp), %rsi
	callq	mapal_gpu_h2d@PLT
	movq	64(%rsp), %r14
	movq	80(%rsp), %rbp
	movl	$64, %esi
	leaq	56(%rsp), %rdi
	callq	mapal_gpu_alloc@PLT
	movq	%r14, 520(%rsp)
	movq	%rbp, 528(%rsp)
	movq	$0, 536(%rsp)
	movq	%rax, 544(%rsp)
	movabsq	$-7490935162623979227, %rdi     # imm = 0x980ADA2884509125
	xorl	%esi, %esi
	movl	$4, %edx
	leaq	520(%rsp), %rcx
	movl	$1, %r8d
	movl	$1, %r9d
	pushq	$0
	.cfi_adjust_cfa_offset 8
	pushq	$256                            # imm = 0x100
	.cfi_adjust_cfa_offset 8
	callq	mapal_gpu_launchv@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
	movq	56(%rsp), %r14
	movq	72(%rsp), %rbp
	movl	$32, %esi
	leaq	48(%rsp), %rdi
	callq	mapal_gpu_alloc@PLT
	movabsq	$-5971682293135266196, %rdi     # imm = 0xAD205297FE2EAE6C
	movq	%r14, %rsi
	movq	%rbp, %rdx
	movq	%rax, %rcx
	movl	$1, %r8d
	movl	$1, %r9d
	pushq	$0
	.cfi_adjust_cfa_offset 8
	pushq	$256                            # imm = 0x100
	.cfi_adjust_cfa_offset 8
	callq	mapal_gpu_launch@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
	movq	48(%rsp), %r14
	movq	136(%rsp), %rbp
	movl	$32, %esi
	leaq	24(%rsp), %rdi
	callq	mapal_gpu_alloc@PLT
	movabsq	$-5746171913144188873, %rdi     # imm = 0xB0417F16EAB6C037
	movq	%r14, %rsi
	movq	%rbp, %rdx
	movq	%rax, %rcx
	movl	$1, %r8d
	movl	$1, %r9d
	pushq	$0
	.cfi_adjust_cfa_offset 8
	pushq	$256                            # imm = 0x100
	.cfi_adjust_cfa_offset 8
	callq	mapal_gpu_launch@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
	movq	24(%rsp), %r14
	movq	128(%rsp), %rbp
	movq	96(%rsp), %rbx
	movl	$64, %esi
	leaq	40(%rsp), %rdi
	callq	mapal_gpu_alloc@PLT
	movq	%r14, 552(%rsp)
	movq	%rbp, 560(%rsp)
	movq	$0, 568(%rsp)
	movq	%rbx, 576(%rsp)
	movq	%rax, 584(%rsp)
	movabsq	$4644736598714039774, %rdi      # imm = 0x40756B53E19B3DDE
	xorl	%esi, %esi
	movl	$5, %edx
	leaq	552(%rsp), %rcx
	movl	$1, %r8d
	movl	$1, %r9d
	pushq	$0
	.cfi_adjust_cfa_offset 8
	pushq	$256                            # imm = 0x100
	.cfi_adjust_cfa_offset 8
	callq	mapal_gpu_launchv@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
	movq	40(%rsp), %rbx
	movq	120(%rsp), %rbp
	movq	88(%rsp), %r12
	movq	24(%rsp), %r13
	movl	$32, %esi
	leaq	168(%rsp), %rdi
	callq	mapal_gpu_alloc@PLT
	movq	%rax, %r14
	movq	%rbx, 592(%rsp)
	movq	%rbp, 600(%rsp)
	movq	$0, 608(%rsp)
	movq	%r12, 616(%rsp)
	movq	%r13, 624(%rsp)
	movq	%rax, 632(%rsp)
	movabsq	$-4366650180424198423, %rdi     # imm = 0xC3668ACBE16B96E9
	xorl	%esi, %esi
	movl	$6, %edx
	leaq	592(%rsp), %rcx
	movl	$1, %r8d
	movl	$1, %r9d
	pushq	$0
	.cfi_adjust_cfa_offset 8
	pushq	$256                            # imm = 0x100
	.cfi_adjust_cfa_offset 8
	callq	mapal_gpu_launchv@PLT
	addq	$16, %rsp
	.cfi_adjust_cfa_offset -16
	movl	$32, %ecx
	leaq	480(%rsp), %rdi
	movq	%r14, %rsi
	xorl	%edx, %edx
	callq	mapal_gpu_d2h@PLT
	movaps	1152(%rsp), %xmm15              # 16-byte Reload
	movaps	1168(%rsp), %xmm14              # 16-byte Reload
	movaps	1184(%rsp), %xmm13              # 16-byte Reload
	movaps	1200(%rsp), %xmm12              # 16-byte Reload
	movaps	1216(%rsp), %xmm10              # 16-byte Reload
	movaps	1232(%rsp), %xmm9               # 16-byte Reload
	movaps	1248(%rsp), %xmm8               # 16-byte Reload
	movaps	1264(%rsp), %xmm7               # 16-byte Reload
	movaps	1280(%rsp), %xmm6               # 16-byte Reload
	movaps	1296(%rsp), %xmm5               # 16-byte Reload
	movaps	1312(%rsp), %xmm4               # 16-byte Reload
	movaps	1328(%rsp), %xmm2               # 16-byte Reload
	movaps	1344(%rsp), %xmm1               # 16-byte Reload
	movaps	1360(%rsp), %xmm0               # 16-byte Reload
	movaps	480(%rsp), %xmm11
	movss	496(%rsp), %xmm3                # xmm3 = mem[0],zero,zero,zero
	movss	%xmm3, 16(%rsp)                 # 4-byte Spill
	movss	500(%rsp), %xmm3                # xmm3 = mem[0],zero,zero,zero
	movss	%xmm3, 20(%rsp)                 # 4-byte Spill
	movss	504(%rsp), %xmm3                # xmm3 = mem[0],zero,zero,zero
	movss	%xmm3, 8(%rsp)                  # 4-byte Spill
	movss	508(%rsp), %xmm3                # xmm3 = mem[0],zero,zero,zero
	movss	%xmm3, 12(%rsp)                 # 4-byte Spill
	movq	184(%rsp), %rcx                 # 8-byte Reload
	incq	%rcx
	movq	176(%rsp), %rbx                 # 8-byte Reload
	addq	$32, %rbx
	addq	$8, %r15
	movb	$1, %r14b
	cmpq	$128, %r15
	je	.LBB0_5
.LBB0_1:                                # %bb62.i
                                        # =>This Inner Loop Header: Depth=1
	movaps	%xmm0, 640(%rsp)
	movaps	%xmm1, 656(%rsp)
	movaps	%xmm2, 672(%rsp)
	movaps	%xmm4, 688(%rsp)
	movaps	%xmm5, 704(%rsp)
	movaps	%xmm6, 720(%rsp)
	movaps	%xmm7, 736(%rsp)
	movaps	%xmm8, 752(%rsp)
	movaps	%xmm9, 768(%rsp)
	movaps	%xmm10, 784(%rsp)
	movaps	%xmm12, 800(%rsp)
	movaps	%xmm13, 816(%rsp)
	movaps	%xmm14, 832(%rsp)
	movaps	464(%rsp), %xmm0                # 16-byte Reload
	movaps	%xmm0, 848(%rsp)
	movaps	448(%rsp), %xmm0                # 16-byte Reload
	movaps	%xmm0, 864(%rsp)
	movaps	432(%rsp), %xmm0                # 16-byte Reload
	movaps	%xmm0, 880(%rsp)
	movaps	416(%rsp), %xmm0                # 16-byte Reload
	movaps	%xmm0, 896(%rsp)
	movaps	400(%rsp), %xmm0                # 16-byte Reload
	movaps	%xmm0, 912(%rsp)
	movaps	384(%rsp), %xmm0                # 16-byte Reload
	movaps	%xmm0, 928(%rsp)
	movaps	368(%rsp), %xmm0                # 16-byte Reload
	movaps	%xmm0, 944(%rsp)
	movaps	352(%rsp), %xmm0                # 16-byte Reload
	movaps	%xmm0, 960(%rsp)
	movaps	336(%rsp), %xmm0                # 16-byte Reload
	movaps	%xmm0, 976(%rsp)
	movaps	320(%rsp), %xmm0                # 16-byte Reload
	movaps	%xmm0, 992(%rsp)
	movaps	304(%rsp), %xmm0                # 16-byte Reload
	movaps	%xmm0, 1008(%rsp)
	movaps	288(%rsp), %xmm0                # 16-byte Reload
	movaps	%xmm0, 1024(%rsp)
	movaps	272(%rsp), %xmm0                # 16-byte Reload
	movaps	%xmm0, 1040(%rsp)
	movaps	256(%rsp), %xmm0                # 16-byte Reload
	movaps	%xmm0, 1056(%rsp)
	movaps	240(%rsp), %xmm0                # 16-byte Reload
	movaps	%xmm0, 1072(%rsp)
	movaps	224(%rsp), %xmm0                # 16-byte Reload
	movaps	%xmm0, 1088(%rsp)
	movaps	208(%rsp), %xmm0                # 16-byte Reload
	movaps	%xmm0, 1104(%rsp)
	movaps	192(%rsp), %xmm0                # 16-byte Reload
	movaps	%xmm0, 1120(%rsp)
	leal	-101(%r15), %eax
	leal	-100(%r15), %esi
	movq	%rcx, 184(%rsp)                 # 8-byte Spill
	cmpq	$13, %rcx
	cmovbl	%r15d, %eax
	leaq	1(%r15), %rdx
	leal	(,%rax,8), %ecx
	subl	%eax, %ecx
	addl	$13, %ecx
	movslq	%ecx, %rax
	imulq	$680390859, %rax, %rcx          # imm = 0x288DF0CB
	movq	%rcx, %rdi
	shrq	$63, %rdi
	shrq	$32, %rcx
	sarl	$4, %ecx
	addl	%edi, %ecx
	imull	$101, %ecx, %ecx
	subl	%ecx, %eax
	cmpq	$101, %rdx
	cmovbl	%edx, %esi
	movaps	%xmm15, 1136(%rsp)
	leal	(,%rsi,8), %ecx
	subl	%esi, %ecx
	addl	$13, %ecx
	imulq	$1148159575, %rcx, %rsi         # imm = 0x446F8657
	shrq	$32, %rsi
	movl	%ecx, %edi
	subl	%esi, %edi
	shrl	%edi
	addl	%esi, %edi
	shrl	$6, %edi
	imull	$101, %edi, %esi
	subl	%esi, %ecx
	leal	-98(%r15), %esi
	leal	-99(%r15), %edi
	leaq	3(%r15), %r8
	leaq	2(%r15), %r9
	movd	%r8d, %xmm0
	movd	%r9d, %xmm1
	punpckldq	%xmm0, %xmm1            # xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1]
	movd	%r15d, %xmm2
	movd	%edx, %xmm0
	punpckldq	%xmm0, %xmm2            # xmm2 = xmm2[0],xmm0[0],xmm2[1],xmm0[1]
	punpcklqdq	%xmm1, %xmm2            # xmm2 = xmm2[0],xmm1[0]
	movdqa	%xmm2, 144(%rsp)                # 16-byte Spill
	cmpq	$101, %r8
	cmovbl	%r8d, %esi
	cmpq	$101, %r9
	cmovbl	%r9d, %edi
	leal	(,%rsi,8), %edx
	subl	%esi, %edx
	leal	(,%rdi,8), %esi
	subl	%edi, %esi
	addl	$13, %edx
	addl	$13, %esi
	imulq	$1148159575, %rdx, %rdi         # imm = 0x446F8657
	shrq	$32, %rdi
	movl	%edx, %r8d
	subl	%edi, %r8d
	shrl	%r8d
	addl	%edi, %r8d
	shrl	$6, %r8d
	imull	$101, %r8d, %edi
	subl	%edi, %edx
	imulq	$1148159575, %rsi, %rdi         # imm = 0x446F8657
	shrq	$32, %rdi
	movl	%esi, %r8d
	subl	%edi, %r8d
	shrl	%r8d
	addl	%edi, %r8d
	shrl	$6, %r8d
	imull	$101, %r8d, %edi
	subl	%edi, %esi
	addl	$-50, %edx
	addl	$-50, %esi
	movd	%eax, %xmm0
	movd	%ecx, %xmm1
	punpckldq	%xmm1, %xmm0            # xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1]
	movd	%esi, %xmm1
	movd	%edx, %xmm2
	punpcklqdq	%xmm1, %xmm2            # xmm2 = xmm2[0],xmm1[0]
	paddd	.LCPI0_0(%rip), %xmm0
	shufps	$36, %xmm2, %xmm0               # xmm0 = xmm0[0,1],xmm2[2,0]
	cvtdq2ps	%xmm0, %xmm0
	mulps	.LCPI0_1(%rip), %xmm0
	addps	%xmm11, %xmm0
	movaps	%xmm0, 640(%rsp,%r15,4)
	leal	-97(%r15), %edi
	movq	%r15, %xmm0
	pshufd	$68, %xmm0, %xmm0               # xmm0 = xmm0[0,1,0,1]
	movdqa	%xmm0, %xmm2
	por	.LCPI0_2(%rip), %xmm2
	por	.LCPI0_3(%rip), %xmm0
	pshufd	$238, %xmm0, %xmm1              # xmm1 = xmm0[2,3,2,3]
	movq	%xmm1, %rax
	movq	%xmm0, %rcx
	pshufd	$238, %xmm2, %xmm1              # xmm1 = xmm2[2,3,2,3]
	movq	%xmm1, %rdx
	movq	%xmm2, %rsi
	cmpq	$101, %rsi
	movl	%esi, %r8d
	cmovael	%edi, %r8d
	leal	(,%r8,8), %edi
	subl	%r8d, %edi
	addl	$13, %edi
	imulq	$1148159575, %rdi, %r8          # imm = 0x446F8657
	shrq	$32, %r8
	movl	%edi, %r9d
	subl	%r8d, %r9d
	shrl	%r9d
	addl	%r8d, %r9d
	shrl	$6, %r9d
	imull	$101, %r9d, %r8d
	subl	%r8d, %edi
	addl	$-50, %edi
	xorps	%xmm1, %xmm1
	cvtsi2ss	%edi, %xmm1
	shufps	$136, %xmm0, %xmm2              # xmm2 = xmm2[0,2],xmm0[0,2]
	movaps	%xmm2, 1376(%rsp)               # 16-byte Spill
	leal	-96(%r15), %edi
	movss	.LCPI0_4(%rip), %xmm2           # xmm2 = [1.5625E-2,0.0E+0,0.0E+0,0.0E+0]
	mulss	%xmm2, %xmm1
	addss	16(%rsp), %xmm1                 # 4-byte Folded Reload
	cmpq	$101, %rdx
	movl	%edx, %r8d
	cmovael	%edi, %r8d
	leal	(,%r8,8), %edi
	subl	%r8d, %edi
	addl	$13, %edi
	imulq	$1148159575, %rdi, %r8          # imm = 0x446F8657
	shrq	$32, %r8
	movl	%edi, %r9d
	subl	%r8d, %r9d
	shrl	%r9d
	addl	%r8d, %r9d
	shrl	$6, %r9d
	imull	$101, %r9d, %r8d
	subl	%r8d, %edi
	addl	$-50, %edi
	xorps	%xmm0, %xmm0
	cvtsi2ss	%edi, %xmm0
	movss	%xmm1, 640(%rsp,%rsi,4)
	leal	-95(%r15), %esi
	mulss	%xmm2, %xmm0
	addss	20(%rsp), %xmm0                 # 4-byte Folded Reload
	cmpq	$101, %rcx
	movl	%ecx, %edi
	cmovael	%esi, %edi
	leal	(,%rdi,8), %esi
	subl	%edi, %esi
	addl	$13, %esi
	imulq	$1148159575, %rsi, %rdi         # imm = 0x446F8657
	shrq	$32, %rdi
	movl	%esi, %r8d
	subl	%edi, %r8d
	shrl	%r8d
	addl	%edi, %r8d
	shrl	$6, %r8d
	imull	$101, %r8d, %edi
	subl	%edi, %esi
	addl	$-50, %esi
	xorps	%xmm1, %xmm1
	cvtsi2ss	%esi, %xmm1
	movss	%xmm0, 640(%rsp,%rdx,4)
	leal	-94(%r15), %edx
	mulss	%xmm2, %xmm1
	addss	8(%rsp), %xmm1                  # 4-byte Folded Reload
	cmpq	$101, %rax
	movl	%eax, %esi
	cmovael	%edx, %esi
	leal	(,%rsi,8), %edx
	subl	%esi, %edx
	addl	$13, %edx
	imulq	$1148159575, %rdx, %rsi         # imm = 0x446F8657
	shrq	$32, %rsi
	movl	%edx, %edi
	subl	%esi, %edi
	shrl	%edi
	addl	%esi, %edi
	shrl	$6, %edi
	imull	$101, %edi, %esi
	subl	%esi, %edx
	addl	$-50, %edx
	xorps	%xmm0, %xmm0
	cvtsi2ss	%edx, %xmm0
	movss	%xmm1, 640(%rsp,%rcx,4)
	mulss	%xmm2, %xmm0
	addss	12(%rsp), %xmm0                 # 4-byte Folded Reload
	movss	%xmm0, 640(%rsp,%rax,4)
	movl	$512, %esi                      # imm = 0x200
	leaq	32(%rsp), %rdi
	callq	mapal_gpu_alloc@PLT
	testb	$1, %r14b
	movq	%rbx, 176(%rsp)                 # 8-byte Spill
	jne	.LBB0_3
# %bb.2:                                # %bb1035.i
                                        #   in Loop: Header=BB0_1 Depth=1
	movl	$512, %edx                      # imm = 0x200
	movq	%rax, %rdi
	leaq	640(%rsp), %rsi
	jmp	.LBB0_4
.LBB0_5:                                # %mapal_main.exit
	movaps	%xmm11, 144(%rsp)               # 16-byte Spill
	movaps	%xmm11, %xmm0
	movl	$1, %edi
	callq	mapal_print_f32@PLT
	movaps	144(%rsp), %xmm0                # 16-byte Reload
	shufps	$85, %xmm0, %xmm0               # xmm0 = xmm0[1,1,1,1]
	movl	$1, %edi
	callq	mapal_print_f32@PLT
	movaps	144(%rsp), %xmm0                # 16-byte Reload
	movhlps	%xmm0, %xmm0                    # xmm0 = xmm0[1,1]
	movl	$1, %edi
	callq	mapal_print_f32@PLT
	movaps	144(%rsp), %xmm0                # 16-byte Reload
	shufps	$255, %xmm0, %xmm0              # xmm0 = xmm0[3,3,3,3]
	movl	$1, %edi
	callq	mapal_print_f32@PLT
	movss	16(%rsp), %xmm0                 # 4-byte Reload
                                        # xmm0 = mem[0],zero,zero,zero
	movl	$1, %edi
	callq	mapal_print_f32@PLT
	movss	20(%rsp), %xmm0                 # 4-byte Reload
                                        # xmm0 = mem[0],zero,zero,zero
	movl	$1, %edi
	callq	mapal_print_f32@PLT
	movss	8(%rsp), %xmm0                  # 4-byte Reload
                                        # xmm0 = mem[0],zero,zero,zero
	movl	$1, %edi
	callq	mapal_print_f32@PLT
	movss	12(%rsp), %xmm0                 # 4-byte Reload
                                        # xmm0 = mem[0],zero,zero,zero
	movl	$1, %edi
	callq	mapal_print_f32@PLT
	leaq	136(%rsp), %rdi
	callq	mapal_gpu_free@PLT
	leaq	128(%rsp), %rdi
	callq	mapal_gpu_free@PLT
	leaq	120(%rsp), %rdi
	callq	mapal_gpu_free@PLT
	leaq	112(%rsp), %rdi
	callq	mapal_gpu_free@PLT
	leaq	104(%rsp), %rdi
	callq	mapal_gpu_free@PLT
	leaq	96(%rsp), %rdi
	callq	mapal_gpu_free@PLT
	leaq	88(%rsp), %rdi
	callq	mapal_gpu_free@PLT
	leaq	32(%rsp), %rdi
	callq	mapal_gpu_free@PLT
	leaq	80(%rsp), %rdi
	callq	mapal_gpu_free@PLT
	leaq	72(%rsp), %rdi
	callq	mapal_gpu_free@PLT
	leaq	64(%rsp), %rdi
	callq	mapal_gpu_free@PLT
	leaq	56(%rsp), %rdi
	callq	mapal_gpu_free@PLT
	leaq	48(%rsp), %rdi
	callq	mapal_gpu_free@PLT
	leaq	24(%rsp), %rdi
	callq	mapal_gpu_free@PLT
	leaq	40(%rsp), %rdi
	callq	mapal_gpu_free@PLT
	leaq	168(%rsp), %rdi
	callq	mapal_gpu_free@PLT
	xorl	%eax, %eax
	addq	$1432, %rsp                     # imm = 0x598
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
	.cfi_endproc
                                        # -- End function
	.type	.L.str0,@object                 # @.str0
	.section	.rodata,"a",@progbits
.L.str0:
	.ascii	"step="
	.size	.L.str0, 5

	.type	.L.str1,@object                 # @.str1
.L.str1:
	.byte	121
	.size	.L.str1, 1

	.section	".note.GNU-stack","",@progbits
	.addrsig
