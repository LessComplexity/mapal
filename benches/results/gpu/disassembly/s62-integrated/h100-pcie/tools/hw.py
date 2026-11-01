"""Hardware usage from time + known bytes/FLOPs (no counters: vast refuses them, ERR_NVGPUCTRPERM).

bytes = A + B + C (each operand read once from DRAM, C written once: the compulsory traffic; L2 is
flushed before every timed launch). Peaks from the box's device attributes: FP32 = SMs x 128 lanes
x 2 x max SM clock; DRAM = 2 x memory clock x bus width / 8.
Class: `memory` / `compute` when the kernel reaches >= 50% of the roof its arithmetic intensity
sits under; `latency` when it reaches neither (the roof is not what limits it); `launch` when the
measured time is under 10 us.
"""


def peaks(facts):
    sms = int(facts['sms']); clk = int(facts['sm_clock_khz']) * 1e3
    mem = int(facts['mem_clock_khz']) * 1e3; bus = int(facts.get('bus_width_bits', 5120))
    return dict(sms=sms, flops=sms * 128 * 2 * clk, bw=2 * mem * bus / 8)


def usage(M, N, K, ms, pk):
    flops = 2.0 * M * N * K
    byts = 4.0 * (M * K + K * N + M * N)
    t = ms * 1e-3
    ai = flops / byts
    ridge = pk['flops'] / pk['bw']
    gbs, fl = byts / t, flops / t
    roof_t = max(byts / pk['bw'], flops / pk['flops'])
    under = 'memory' if ai < ridge else 'compute'
    frac = (gbs / pk['bw']) if under == 'memory' else (fl / pk['flops'])
    cls = 'launch' if ms < 0.010 else under if frac >= 0.5 else 'latency'
    return dict(bytes=byts, flops=flops, ai=round(ai, 2), ridge=round(ridge, 2), gbs=gbs / 1e9,
                bw_pct=100 * gbs / pk['bw'], tflops=fl / 1e12, fp32_pct=100 * fl / pk['flops'],
                roof_ms=roof_t * 1e3, roof_frac=roof_t / t, roof=under, cls=cls)


if __name__ == '__main__':
    pk = peaks(dict(sms=114, sm_clock_khz=1755000, mem_clock_khz=1593000, bus_width_bits=5120))
    u = usage(16, 2048, 7168, 0.0689, pk)
    assert abs(pk['flops'] / 1e12 - 51.2) < 0.1 and abs(pk['bw'] / 1e9 - 2039) < 1
    assert u['roof'] == 'memory' and 40 < u['bw_pct'] < 45, u
    print('ok', round(u['gbs']), round(u['bw_pct'], 1), u['cls'])
