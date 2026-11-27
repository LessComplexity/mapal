//! The host target profiles (generic, Apple M, Zen 3, Raptor Lake, x86-64-v3) and the CUDA Ada base.

use super::*;

/// **S62 `hostopt`: a GPU build's one profile** — the device's facts
/// (`name`, `machine`) and the host's CPU facts for everything outside an
/// offloaded site. Not one field with two readings (the S45 lesson): every
/// CPU-shaped field of a GPU profile always described the host, and this
/// fills them from a host profile instead of [`GENERIC`]. `name` stays the
/// device's — the rung table and kernel names key on it.
pub fn with_host(dev: &TargetProfile, host: &TargetProfile) -> TargetProfile {
    TargetProfile {
        name: dev.name,
        machine: dev.machine,
        ..*host
    }
}

/// Today's six literals, exactly: 128-bit vectors, 32 registers, a 512 KB
/// per-core L2. Emission under this profile is byte-identical to the
/// pre-profile emitter for every shape — the plan's rule 1, and the reason it
/// is the default.
pub const GENERIC: TargetProfile = TargetProfile {
    name: "generic",
    machine: Machine::Cpu,
    vec_bytes: 16,
    vec_regs: 32,
    acc_vecs_per_row: 4,
    nc_tiles: 32,
    l2_bytes: 512 * 1024,
    l2_cores: 1,
    heap_min_bytes: 256 * 1024,
    sme: None,
    // No L1D geometry ⇒ the S45 move rung cannot fire under the DEFAULT
    // profile, so `--target=generic` stays byte-identical for all 171 swept
    // cells. Firing needs a profile that describes a real part.
    l1d: None,
    // No measured rates ⇒ every task is dispatched, which is what every profile
    // did before this field existed. `apple-m` and `zen3` inherit it because
    // nobody has measured them either; `raptorlake`'s numbers are OWED and were
    // not taken because the box was unreachable this session.
    dispatch: None,
    math: HOST_MATH,
};

/// Apple M-series: NEON at 16 B × 32 registers, but a 16 MB L2 **shared by five
/// P-cores** (`hw.perflevel0.l2cachesize` / `hw.perflevel0.cpusperl2` on this M4
/// Pro). Same tile widths as `generic`; the L2 is what differs, and it closes the
/// KC gate by derivation.
pub const APPLE_M: TargetProfile = TargetProfile {
    name: "apple-m",
    l2_bytes: 16 * 1024 * 1024,
    l2_cores: 5,
    ..GENERIC
};

/// AVX2 zen3: 32 B vectors, 16 registers, 512 KB per-core L2.
///
/// **UNTESTED — read off documentation, never measured.** Note it moves two
/// constants at once relative to `generic`: `tile_j(f32)` 16 → 32 and `tile_i`
/// 4 → 2. That is self-consistent arithmetic, not a validated configuration;
/// the box leg is what settles it.
pub const ZEN3: TargetProfile = TargetProfile {
    name: "zen3",
    vec_bytes: 32,
    vec_regs: 16,
    ..GENERIC
};

/// Intel Raptor Lake — the i9-14900F box (24C/32T, 8 P + 16 E, governor
/// `performance`). **The cross-compilation case, and the reason named profiles
/// still exist:** the box has gcc and no clang, so every leg is emitted on the
/// Mac, where nothing about the i9 can be detected. Every number here was read
/// off `/sys/devices/system/cpu/cpu0/cache/` on the box itself and is recorded
/// with its source:
///
/// | field | source | value |
/// | --- | --- | --- |
/// | `l1d.bytes` | `index0/size` | 48K (P-core; the E-core is 32K) |
/// | `l1d.line_bytes` | `index0/coherency_line_size` | 64 |
/// | `l1d.page_bytes` | `getconf PAGESIZE` | 4096 |
/// | `l2_bytes` | `index2/size` | 2048K |
/// | `l2_cores` | `index2/shared_cpu_list` = `0-1` | **1** physical core (two SMT threads) |
///
/// `sets`/`ways` are **not** recorded: `page/line = 64` and
/// `48K/(64·64) = 12` reproduce `number_of_sets=64` and
/// `ways_of_associativity=12` exactly (`l1d_derivation_reproduces_the_i9`).
///
/// It describes the **P-core**, which is what the 1-thread legs are pinned to
/// (`taskset -c 4`). The E-core's 32K/8-way L1D derives 8 ways from the same
/// rule and would give the same block here; the P-core is the conservative and
/// the measured one.
pub const RAPTORLAKE: TargetProfile = TargetProfile {
    name: "raptorlake",
    vec_bytes: 32,
    vec_regs: 16,
    l2_bytes: 2 * 1024 * 1024,
    l2_cores: 1,
    l1d: Some(L1d {
        bytes: 48 * 1024,
        line_bytes: 64,
        sets: 64,
    }),
    ..GENERIC
};

/// **The rented GPU boxes' host (S62 `hostopt`): x86-64-v3** — the ISA level
/// (AVX2 + FMA: 32 B vectors, 16 registers) every vast.ai host CPU met so far
/// implements, and the `-march` `benches/gpu/bundle.sh` compiles its host
/// objects for. A level, not a family: the boxes' CPUs vary per rental
/// (`lscpu` is recorded in every results file), so only the ISA facts are
/// written down; the cache facts stay [`GENERIC`]'s lower bounds (512 KB L2
/// per core, no L1D geometry ⇒ move-panel declines), which no cell of the
/// S62 set reads.
pub const X86_64_V3: TargetProfile = TargetProfile {
    name: "x86-64-v3",
    vec_bytes: 32,
    vec_regs: 16,
    // MEASURED ON THE BOXES (S62, `benches/results/gpu/s62/cpugpu/`), each the
    // most conservative of the boxes it was read on:
    //
    // `flops_per_ns` — one lane of this build, `run1/results-*.txt` `lane` rows
    //   (the CPU build at MAPAL_PAR=1): pt_pc_128, 3 × 2·128³ / iter ms — 47
    //   (EPYC 7402), 30 (EPYC 7402), 26 (Xeon 8480+), 24 (EPYC 7B12). A slower
    //   lane only ever keeps a site on the device, and dispatches a task.
    // `bytes_per_ns` — pt_pj_m1_n2048_k7168, its site's 58.76 MB / iter ms on
    //   the same rows: 5, 5, 7, 6.
    // `dispatch_ns` — the S51 difference method, `run2/results-*.txt`:
    //   pt_pc_32's CPU build minus its every-task-inline build (a scratch
    //   profile, deleted after the run), over the 3 tile tasks between its
    //   clocks — 28.1 us (EPYC 7742, 128 threads), 34.5 us (EPYC 7452, 128),
    //   16.7 us (Xeon SPR, 32). The smallest dispatches the most tasks: the
    //   behaviour before this was measured is dispatch_ns = 0.
    dispatch: Some(DispatchCost {
        flops_per_ns: 24,
        bytes_per_ns: 5,
        dispatch_ns: 16_700,
        lanes: 0,
    }),
    ..GENERIC
};

/// **MEASUREMENT SCRATCH — S51b, DELETE AFTER USE. Not a machine.**
///
/// `raptorlake` with one field changed: a `dispatch_ns` so large that
/// [`FnEmit::under_dispatch_cost`]'s roofline answers "inline" for every tile
/// site in the program. It exists to make the `dispatch_ns` A/B possible at all,
/// because that constant can only be measured as a DIFFERENCE — dispatch
/// everything (`raptorlake`, `dispatch: None`) against inline everything (this)
/// — divided by the number of dispatches that flipped.
///
/// `flops_per_ns` is the real measured 144; `bytes_per_ns` is a placeholder and
/// is not read for any decision here, because `ns < dispatch_ns` holds for every
/// site at this `dispatch_ns` whatever the rate. Only the huge value matters.
///
/// **Do not ship a program built against this.** It is the `kc_nest` mistake in
/// miniature — a measurement lever that must not outlive its measurement — and
/// the repo already carries a P1 to delete that one.
pub const RAPTORLAKE_INLINE: TargetProfile = TargetProfile {
    name: "raptorlake-inline",
    dispatch: Some(DispatchCost {
        flops_per_ns: 144,
        bytes_per_ns: 100,
        dispatch_ns: 1_000_000_000,
        lanes: 0,
    }),
    ..RAPTORLAKE
};

/// NVIDIA Ada (`sm_89`) — the box GPU, an RTX 4070 Ti (compute capability 8.9,
/// driver 610.43.03, 12 GB). plan-s41 step 1; S52 gave it the facts the tile
/// geometry reads ([`TargetProfile::gpu_tile`]).
///
/// The CPU-shaped fields are inherited from [`GENERIC`]: on a GPU profile they
/// describe the **host** that runs everything outside an offloaded site, and
/// no GPU realization reads them. S52 deliberately did NOT give them GPU
/// readings (plan-s41 §2.1 proposed `vec_bytes` → per-thread registers,
/// `l2_bytes` → shared memory): one field with two readings is the `l2_bytes`
/// per-core/shared ambiguity that cost S45 a session, so every GPU quantity is
/// its own [`Gpu`] field.
///
/// `sm_89` has 4th-generation tensor cores; `mma` is the precision-face rung
/// (ADR-0032 D1/D3), not this one.
pub const CUDA_ADA: TargetProfile = TargetProfile {
    name: "cuda-ada",
    machine: Machine::Gpu(ADA_SM),
    ..GENERIC
};
