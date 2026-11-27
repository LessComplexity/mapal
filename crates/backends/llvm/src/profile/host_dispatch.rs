//! The measured host dispatch cost (Linux): probe, cache and parse.

use super::DispatchCost;

pub(crate) fn get() -> Option<DispatchCost> {
    let env = std::env::var("MAPAL_NATIVE_DISPATCH").ok();
    match env.as_deref() {
        Some("off") => return None,
        Some(s) if s != "measure" => return parse(s),
        _ => {}
    }
    let key = key();
    let path = cache_path();
    if env.is_none()
        && let Some(hit) = path
            .as_ref()
            .and_then(|p| std::fs::read_to_string(p).ok())
            .and_then(|text| lookup(&text, &key))
    {
        return Some(hit);
    }
    // Only a measurement that describes this machine is cached. One taken
    // on a busy machine (another build, a test suite) times lanes queued
    // behind someone else's threads; one taken by an unoptimized build
    // (`cargo test`) times its own debug code — 10 flops and 1 byte a ns
    // on the 9970X against 282 and 116 optimized. Either serves its
    // process only, and a later quiet, optimized run measures again.
    // `measure` is an explicit request and always writes.
    let cache = env.is_some() || !(cfg!(debug_assertions) || machine_busy());
    let d = measure()?;
    if cache && let Some(p) = path {
        let old = std::fs::read_to_string(&p).unwrap_or_default();
        let mut text: String = old
            .lines()
            .filter(|l| l.split('\t').next() != Some(key.as_str()))
            .map(|l| format!("{l}\n"))
            .collect();
        text.push_str(&line(&key, &d));
        let _ = std::fs::create_dir_all(p.parent().unwrap_or(std::path::Path::new(".")));
        let _ = std::fs::write(&p, text);
    }
    Some(d)
}

/// `d,l,f,b` -> the economics; `None` on anything else.
pub(crate) fn parse(s: &str) -> Option<DispatchCost> {
    let v: Vec<u64> = s
        .split(',')
        .map(|x| x.trim().parse().ok())
        .collect::<Option<_>>()?;
    let [dispatch_ns, lanes, flops_per_ns, bytes_per_ns] = v[..] else {
        return None;
    };
    (dispatch_ns > 0 && flops_per_ns > 0 && bytes_per_ns > 0).then_some(DispatchCost {
        flops_per_ns,
        bytes_per_ns,
        dispatch_ns,
        lanes,
    })
}

pub(crate) fn line(key: &str, d: &DispatchCost) -> String {
    format!(
        "{key}\t{},{},{},{}\n",
        d.dispatch_ns, d.lanes, d.flops_per_ns, d.bytes_per_ns
    )
}

pub(crate) fn lookup(text: &str, key: &str) -> Option<DispatchCost> {
    text.lines()
        .filter_map(|l| l.split_once('\t'))
        .find(|(k, _)| *k == key)
        .and_then(|(_, v)| parse(v))
}

/// `/proc/loadavg` against `online` CPUs: busy when the 1-minute load or
/// the tasks runnable right now (less the reader) exceed a quarter of
/// them. Unreadable is quiet: the cache behaves as it did before.
pub(crate) fn busy(loadavg: &str, online: u64) -> bool {
    let mut f = loadavg.split_whitespace();
    let load1 = f.next().and_then(|x| x.parse::<f64>().ok());
    let running = f
        .nth(2)
        .and_then(|x| x.split_once('/'))
        .and_then(|(r, _)| r.parse::<u64>().ok());
    let quarter = online as f64 / 4.0;
    load1.is_some_and(|l| l > quarter)
        || running.is_some_and(|r| r.saturating_sub(1) as f64 > quarter)
}

fn machine_busy() -> bool {
    let online = std::fs::read_to_string("/sys/devices/system/cpu/online")
        .map_or(1, |s| super::cpu_list_len(&s));
    std::fs::read_to_string("/proc/loadavg").is_ok_and(|t| busy(&t, online))
}

fn key() -> String {
    let model = std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|t| {
            t.lines()
                .find(|l| l.starts_with("model name"))
                .and_then(|l| l.split_once(':'))
                .map(|(_, m)| m.trim().to_string())
        })
        .unwrap_or_default();
    let logical = std::thread::available_parallelism().map_or(1, usize::from);
    let par = std::env::var("MAPAL_PAR").unwrap_or_default();
    format!(
        "v1|{model}|{logical}|par={par}|{}",
        env!("CARGO_PKG_VERSION")
    )
}

fn cache_path() -> Option<std::path::PathBuf> {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| std::path::Path::new(&h).join(".cache")))?;
    Some(base.join("mapal").join("native-dispatch.tsv"))
}

fn measure() -> Option<DispatchCost> {
    let wave = mapal_rt::calibrate::wave(64)?;
    Some(DispatchCost {
        flops_per_ns: lane_flops_per_ns(),
        bytes_per_ns: lane_bytes_per_ns(),
        dispatch_ns: wave.wave_ns,
        lanes: wave.lanes,
    })
}

/// Best of 5 timings of `f`, which reports the work it did.
fn best_rate(mut f: impl FnMut() -> u64) -> u64 {
    (0..5)
        .map(|_| {
            let t = std::time::Instant::now();
            let work = f();
            work / (t.elapsed().as_nanos() as u64).max(1)
        })
        .max()
        .unwrap_or(1)
        .max(1)
}

fn lane_flops_per_ns() -> u64 {
    const ITERS: u64 = 200_000;
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            // 12 chains x 16 lanes x 2 flops per step.
            return best_rate(|| {
                std::hint::black_box(unsafe { fma512(ITERS) });
                ITERS * 12 * 16 * 2
            });
        }
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            return best_rate(|| {
                std::hint::black_box(unsafe { fma256(ITERS) });
                ITERS * 12 * 8 * 2
            });
        }
    }
    best_rate(|| {
        let mut a = [1.0f32; 12];
        for _ in 0..ITERS {
            for x in a.iter_mut() {
                *x = std::hint::black_box(*x) * 0.999_999 + 1e-7;
            }
        }
        std::hint::black_box(a);
        ITERS * 12 * 2
    })
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn fma512(iters: u64) -> f32 {
    use std::arch::x86_64::*;
    let m = _mm512_set1_ps(0.999_999);
    let c = _mm512_set1_ps(1e-7);
    let mut a = [_mm512_set1_ps(1.0); 12];
    // `black_box` every 64 steps: the chains escape, so the loop cannot be
    // deleted or folded, and 64 register-resident steps sit between escapes.
    for _ in 0..std::hint::black_box(iters / 64) {
        for _ in 0..64 {
            for x in a.iter_mut() {
                *x = _mm512_fmadd_ps(*x, m, c);
            }
        }
        a = std::hint::black_box(a);
    }
    a.iter().map(|&x| _mm512_reduce_add_ps(x)).sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn fma256(iters: u64) -> f32 {
    use std::arch::x86_64::*;
    let m = _mm256_set1_ps(0.999_999);
    let c = _mm256_set1_ps(1e-7);
    let mut a = [_mm256_set1_ps(1.0); 12];
    // `black_box` every 64 steps: the chains escape, so the loop cannot be
    // deleted or folded, and 64 register-resident steps sit between escapes.
    for _ in 0..std::hint::black_box(iters / 64) {
        for _ in 0..64 {
            for x in a.iter_mut() {
                *x = _mm256_fmadd_ps(*x, m, c);
            }
        }
        a = std::hint::black_box(a);
    }
    let mut out = [0f32; 8];
    let mut s = _mm256_setzero_ps();
    for x in a {
        s = _mm256_add_ps(s, x);
    }
    unsafe { _mm256_storeu_ps(out.as_mut_ptr(), s) };
    out.iter().sum()
}

fn lane_bytes_per_ns() -> u64 {
    let buf = vec![1.0f32; 2 << 20];
    best_rate(|| {
        let mut acc = [0f32; 16];
        for _ in 0..4 {
            for chunk in std::hint::black_box(&buf).chunks_exact(16) {
                for (a, x) in acc.iter_mut().zip(chunk) {
                    *a += x;
                }
            }
        }
        std::hint::black_box(acc);
        4 * buf.len() as u64 * 4
    })
}
