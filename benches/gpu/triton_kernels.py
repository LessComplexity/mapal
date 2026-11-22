"""Hand-written Triton competitor kernels (comp_bench.py's `triton*` legs): Triton v3.6.0's tutorial kernels
(python/tutorials/02-fused-softmax.py, 03-matrix-multiplication.py, 06-fused-attention.py; MIT, (c) the Triton
authors), forward paths only, autotuned per GPU by the tutorials' own `triton.autotune` config lists.

Changes against the tutorials, each what a user does to run them on these workloads:
- every kernel: the element type follows the operands (the tutorials store fp16); f32 operands take
  `input_precision="ieee"` (the f32 legs: no TF32) or `"tf32"` (the `-tf32` legs).
- matmul: the tutorial's fused-activation slot carries this suite's epilogues (`affine`: c·alpha + beta, `bias_relu`:
  relu(c + bias[col])); the autotune key adds the precision and epilogue.
- softmax: a `scale` applied to the row before the max (softmax(x·scale): the programs' `(s − max) / n`).
- attention: grouped-query attention (query head h reads K/V head h // (H / H_kv)), the dtype a constexpr (bf16 / f32
  besides fp16), no fp8 / backward; warp specialization as the tutorial's benchmark sets it (only where the device's
  compute capability major is 10).
"""
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

TL_DTYPE = {torch.float16: tl.float16, torch.bfloat16: tl.bfloat16, torch.float32: tl.float32}


def _cc():
    return torch.cuda.get_device_capability()


# --- 03-matrix-multiplication.py ----------------------------------------------------------------------------------

def _mm_configs():
    c = triton.Config
    return [
        c({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        c({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        c({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        c({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        c({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        c({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        c({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        c({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        c({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        c({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        c({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        c({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        c({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        c({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        c({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        c({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    ]


@triton.autotune(configs=_mm_configs(), key=['M', 'N', 'K', 'PREC', 'EPILOGUE'])
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, bias_ptr, alpha, beta, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
                  GROUP_SIZE_M: tl.constexpr, PREC: tl.constexpr, EPILOGUE: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator, input_precision=PREC)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    if EPILOGUE == "affine":
        accumulator = accumulator * alpha + beta
    elif EPILOGUE == "bias_relu":
        bias = tl.load(bias_ptr + offs_cn, mask=offs_cn < N, other=0.0)
        accumulator = tl.maximum(accumulator + bias[None, :].to(tl.float32), 0.0)
    c = accumulator.to(c_ptr.dtype.element_ty)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a, b, prec='ieee', epilogue='none', alpha=1.0, beta=0.0, bias=None):
    """c = epilogue(a @ b); `prec`: the f32 dot precision ('ieee' | 'tf32'; ignored for 16-bit operands)."""
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )  # noqa: E731
    matmul_kernel[grid](a, b, c, bias if bias is not None else c, alpha, beta, M, N, K, a.stride(0), a.stride(1),
                        b.stride(0), b.stride(1), c.stride(0), c.stride(1),
                        PREC=prec if a.dtype == torch.float32 else 'tf32', EPILOGUE=epilogue)
    return c


# --- 02-fused-softmax.py ------------------------------------------------------------------------------------------

@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, scale,
                   BLOCK_SIZE: tl.constexpr, num_stages: tl.constexpr):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf')) * scale
        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)


_SOFTMAX = {}


def softmax(x, scale=1.0):
    """The tutorial's host side: one row per program, programs = SMs × occupancy (from the compiled kernel's registers
    and shared memory), cached per (block size, dtype)."""
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 8
    if 'props' not in _SOFTMAX:  # the tutorial reads the device properties once, at import
        _SOFTMAX['props'] = triton.runtime.driver.active.utils.get_device_properties(x.device.index or 0)
    props = _SOFTMAX['props']
    num_stages = 4 if props['max_shared_mem'] > 200000 else 2
    y = torch.empty_like(x)
    key = (BLOCK_SIZE, x.dtype)
    if key not in _SOFTMAX:
        kernel = softmax_kernel.warmup(y, x, x.stride(0), y.stride(0), n_rows, n_cols, scale, BLOCK_SIZE=BLOCK_SIZE,
                                       num_stages=num_stages, num_warps=num_warps, grid=(1, ))
        kernel._init_handles()
        occupancy = props['max_num_regs'] // max(kernel.n_regs * props['warpSize'] * num_warps, 1)
        if kernel.metadata.shared:
            occupancy = min(occupancy, props['max_shared_mem'] // kernel.metadata.shared)
        _SOFTMAX[key] = (kernel, props['multiprocessor_count'] * max(occupancy, 1))
    kernel, programs = _SOFTMAX[key]
    kernel[(min(programs, n_rows), 1, 1)](y, x, x.stride(0), y.stride(0), n_rows, n_cols, scale, BLOCK_SIZE,
                                          num_stages)
    return y


# --- 06-fused-attention.py (forward) ------------------------------------------------------------------------------

@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q, desc_k, desc_v, offset_y, dtype: tl.constexpr, start_m, qk_scale,
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr, STAGE: tl.constexpr,
                    offs_m: tl.constexpr, offs_n: tl.constexpr, N_CTX: tl.constexpr, warp_specialize: tl.constexpr,
                    PREC: tl.constexpr):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    else:  # causal = False
        lo, hi = 0, N_CTX
    offsetk_y = offset_y + lo
    offsetv_y = offset_y + lo
    for start_n in tl.range(lo, hi, BLOCK_N, warp_specialize=warp_specialize):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = desc_k.load([offsetk_y, 0]).T
        qk = tl.dot(q, k, input_precision=PREC)
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = desc_v.load([offsetv_y, 0])
        p = p.to(dtype)
        acc = tl.dot(p, v, acc, input_precision=PREC)
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        offsetk_y += BLOCK_N
        offsetv_y += BLOCK_N
    return acc, l_i, m_i


def _host_descriptor_pre_hook(nargs):
    BLOCK_M, BLOCK_N, HEAD_DIM = nargs['BLOCK_M'], nargs['BLOCK_N'], nargs['HEAD_DIM']
    if not isinstance(nargs['desc_q'], TensorDescriptor):
        return
    nargs['desc_q'].block_shape = [BLOCK_M, HEAD_DIM]
    nargs['desc_v'].block_shape = [BLOCK_N, HEAD_DIM]
    nargs['desc_k'].block_shape = [BLOCK_N, HEAD_DIM]
    nargs['desc_o'].block_shape = [BLOCK_M, HEAD_DIM]


_ATTN_CONFIGS = [triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w,
                               pre_hook=_host_descriptor_pre_hook)
                 for BM in [64, 128] for BN in [32, 64, 128] for s in [2, 3, 4] for w in [4, 8]]


def _keep(conf):
    return not (torch.cuda.get_device_capability()[0] == 9 and conf.kwargs['BLOCK_M'] * conf.kwargs['BLOCK_N'] <
                128 * 128 and conf.num_warps == 8)


def _prune_invalid_configs(configs, named_args, **kwargs):
    return [conf for conf in configs if conf.kwargs.get('BLOCK_M', 0) <= kwargs['N_CTX']]


@triton.jit
def _maybe_make_tensor_desc(desc_or_ptr, shape, strides, block_shape):
    if isinstance(desc_or_ptr, tl.tensor_descriptor):
        return desc_or_ptr
    else:
        return tl.make_tensor_descriptor(desc_or_ptr, shape, strides, block_shape)


@triton.autotune(configs=list(filter(_keep, _ATTN_CONFIGS)),
                 key=['N_CTX', 'HEAD_DIM', 'STAGE', 'DTYPE', 'PREC', 'warp_specialize'],
                 prune_configs_by={'early_config_prune': _prune_invalid_configs})
@triton.jit
def _attn_fwd(sm_scale, M, Z, H, H_KV, desc_q, desc_k, desc_v, desc_o, N_CTX, HEAD_DIM: tl.constexpr,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, STAGE: tl.constexpr, DTYPE: tl.constexpr,
              PREC: tl.constexpr, warp_specialize: tl.constexpr):
    dtype: tl.constexpr = DTYPE
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    y_dim = Z * H * N_CTX
    kv_dim = Z * H_KV * N_CTX
    desc_q = _maybe_make_tensor_desc(desc_q, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_M, HEAD_DIM])
    desc_v = _maybe_make_tensor_desc(desc_v, shape=[kv_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_N, HEAD_DIM])
    desc_k = _maybe_make_tensor_desc(desc_k, shape=[kv_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_N, HEAD_DIM])
    desc_o = _maybe_make_tensor_desc(desc_o, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_M, HEAD_DIM])
    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    offset_kv = off_z * (N_CTX * H_KV) + (off_h // (H // H_KV)) * N_CTX  # GQA: the query head's K/V head
    qo_offset_y = offset_y + start_m * BLOCK_M
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    q = desc_q.load([qo_offset_y, 0])
    # stage 1: off-band; stage 2: on-band (causal)
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, desc_k, desc_v, offset_kv, dtype, start_m, qk_scale,
                                        BLOCK_M, HEAD_DIM, BLOCK_N, 4 - STAGE, offs_m, offs_n, N_CTX,
                                        warp_specialize, PREC)
    if STAGE & 2:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, desc_k, desc_v, offset_kv, dtype, start_m, qk_scale,
                                        BLOCK_M, HEAD_DIM, BLOCK_N, 2, offs_m, offs_n, N_CTX, warp_specialize, PREC)
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    desc_o.store([qo_offset_y, 0], acc.to(dtype))


def attention(q, k, v, causal, sm_scale, prec='ieee'):
    """o = softmax(q·kᵀ·sm_scale [causal]) · v; q [Z, H, N, D], k / v [Z, H_kv, N, D], contiguous; N a multiple of
    the block sizes (the tutorial's contract). `prec`: the f32 dot precision."""
    Z, H, N, D = q.shape
    assert D in {16, 32, 64, 128, 256} and k.shape[1] == v.shape[1] and H % k.shape[1] == 0
    o = torch.empty_like(q)
    M = torch.empty((Z, H, N), device=q.device, dtype=torch.float32)
    if _cc()[0] >= 9:  # the tutorial's `supports_host_descriptor()` (warp specialization off: its pre-hook path)
        dummy = [1, 1]
        desc_q = TensorDescriptor(q, shape=[Z * H * N, D], strides=[D, 1], block_shape=dummy)
        desc_k = TensorDescriptor(k, shape=[Z * k.shape[1] * N, D], strides=[D, 1], block_shape=dummy)
        desc_v = TensorDescriptor(v, shape=[Z * k.shape[1] * N, D], strides=[D, 1], block_shape=dummy)
        desc_o = TensorDescriptor(o, shape=[Z * H * N, D], strides=[D, 1], block_shape=dummy)
    else:
        desc_q, desc_k, desc_v, desc_o = q, k, v, o

    def alloc_fn(size, align, _):
        return torch.empty(size, dtype=torch.int8, device='cuda')
    triton.set_allocator(alloc_fn)
    ws = _cc()[0] == 10  # the tutorial's benchmark: warp specialization on compute capability 10 only
    extra = {'maxnreg': 168 if D == 128 and q.dtype == torch.float16 else 80} if ws else {}
    grid = lambda META: (triton.cdiv(N, META['BLOCK_M']), Z * H, 1)  # noqa: E731
    _attn_fwd[grid](sm_scale, M, Z, H, k.shape[1], desc_q, desc_k, desc_v, desc_o, N_CTX=N, HEAD_DIM=D,
                    STAGE=3 if causal else 1, DTYPE=TL_DTYPE[q.dtype], PREC=prec if q.dtype == torch.float32 else 'tf32',
                    warp_specialize=ws, **extra)
    return o
