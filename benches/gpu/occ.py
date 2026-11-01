"""Theoretical occupancy on sm_90 (CUDA occupancy-calculator rules; H100 limits)."""
import math

SM90 = dict(threads=2048, blocks=32, regs=65536, reg_unit=256, max_warps=64, smem=233472, smem_reserved=1024)


def occupancy(threads, regs, smem_bytes, lim=SM90):
    """-> (blocks/SM, warps/SM, occupancy fraction, limiter)."""
    warps = math.ceil(threads / 32)
    regs_warp = math.ceil(max(regs, 1) * 32 / lim['reg_unit']) * lim['reg_unit']
    by = {
        'blocks': lim['blocks'],
        'threads': lim['max_warps'] // warps,
        'regs': (lim['regs'] // regs_warp) // warps,
        'smem': lim['smem'] // (smem_bytes + lim['smem_reserved']),
    }
    b = min(by.values())
    lim_name = min(by, key=lambda k: by[k])
    return b, b * warps, b * warps / lim['max_warps'], lim_name


def waves(grid_blocks, blocks_per_sm, sms):
    return grid_blocks / (blocks_per_sm * sms) if blocks_per_sm else float('inf')


if __name__ == '__main__':
    assert occupancy(256, 32, 0) == (8, 64, 1.0, 'blocks') or occupancy(256, 32, 0)[2] == 1.0
    assert occupancy(64, 128, 22528)[0] == 8  # regs: 65536/(128*32)=16 warps -> 8 blocks of 2 warps
    print('ok', occupancy(64, 128, 22528), occupancy(256, 40, 0))
