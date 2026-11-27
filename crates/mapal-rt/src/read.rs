//! `mapal_read` / `mapal_read_task`: the `"file" -> read` builtin (ADR-0045).
use super::*;

pub const READ_DATA_DIR_ENV: &str = "MAPAL_DATA_DIR";

/// ADR-0045 D4: a failed read's exit code (`mapal_ir::read::EXIT_CODE`).
pub const READ_EXIT_CODE: i32 = 103;

/// The `read` path rule (ADR-0045 D5, `mapal_ir::read::resolve_with`): an
/// absolute literal as written; a relative one against `env` when set and
/// non-empty, else against the recorded `data_dir`, else the working
/// directory.
pub fn read_resolve(path: &str, env: Option<&str>, data_dir: Option<&str>) -> std::path::PathBuf {
    let p = std::path::Path::new(path);
    if p.is_absolute() {
        return p.to_path_buf();
    }
    match env.filter(|e| !e.is_empty()).or(data_dir) {
        Some(base) => std::path::Path::new(base).join(p),
        None => p.to_path_buf(),
    }
}

/// Read `path`'s exact `bytes` into `dst` (`None`: check the file only).
/// `Err` is the failure's message, `mapal_ir::read::{size,open}_message`'s
/// text exactly.
pub fn read_file_into(
    path: &std::path::Path,
    dst: Option<&mut [u8]>,
    bytes: u64,
) -> Result<(), String> {
    use std::io::Read as _;
    let open = |e: io::Error| format!("mapal: read failed: {}: {e}", path.display());
    let size = |actual: u64| {
        format!(
            "mapal: read failed: {}: expected {bytes} bytes, found {actual}",
            path.display()
        )
    };
    let meta = std::fs::metadata(path).map_err(open)?;
    if meta.is_dir() {
        return Err(open(io::Error::from(io::ErrorKind::IsADirectory)));
    }
    if meta.len() != bytes {
        return Err(size(meta.len()));
    }
    let mut f = std::fs::File::open(path).map_err(open)?;
    if let Some(dst) = dst {
        // One pass from the page cache into the program's storage: the
        // file's bytes move once (the evidence rule's host side).
        f.read_exact(dst).map_err(open)?;
    }
    Ok(())
}

/// `read` (ADR-0045): the file at the resolved `path` — exactly `bytes` long
/// — read into `dst` (null: checked only), or the defined failure: stdout
/// flushed, the message on stderr, exit 103. Under `MAPAL_GPU_TRACE` (or
/// `MAPAL_READ_TRACE`) one `mapal_read_trace` line reports the bytes moved.
///
/// # Safety
/// `path` (`dir`) must point to `path_len` (`dir_len`) bytes of UTF-8 (the
/// emitter passes private string globals); `dst`, when not null, must be
/// valid for writes of `bytes`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mapal_read(
    path: *const u8,
    path_len: usize,
    dir: *const u8,
    dir_len: usize,
    dst: *mut u8,
    bytes: u64,
) {
    let text = |p: *const u8, n: usize| -> Option<&str> {
        (!p.is_null() && n > 0)
            .then(|| unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(p, n)) })
    };
    let lit = text(path, path_len).unwrap_or("");
    let env = std::env::var(READ_DATA_DIR_ENV).ok();
    let full = read_resolve(lit, env.as_deref(), text(dir, dir_len));
    let t0 = std::time::Instant::now();
    let out =
        (!dst.is_null()).then(|| unsafe { std::slice::from_raw_parts_mut(dst, bytes as usize) });
    let moved = if out.is_some() { bytes } else { 0 };
    if let Err(msg) = read_file_into(&full, out, bytes) {
        read_failed(&msg);
    }
    read_trace(&full, moved, t0);
}

/// Exit with a read failure's message (stdout flushed first).
pub(crate) fn read_failed(msg: &str) -> ! {
    let _ = io::stdout().flush();
    eprintln!("{msg}");
    std::process::exit(READ_EXIT_CODE);
}

/// The failures `mapal_read_task` recorded, by topo position: the one a
/// checkpoint delivers is the run's minimum trap.
pub(crate) static READ_FAILURES: Mutex<Vec<(i64, String)>> = Mutex::new(Vec::new());

/// The trap kind a recorded read failure carries (0 and 1 are the language
/// traps).
pub(crate) const READ_TRAP_KIND: u32 = 2;

/// `read` in a task of the parallel entry (ADR-0045 D8): as [`mapal_read`],
/// but a failure zero-fills `dst`, keeps its message and records a trap at
/// `topo` ([`mapal_par_trap`]) — the first checkpoint past it delivers the
/// run's earliest trap, so the oracle's order holds (every output and trap
/// before the read first).
///
/// # Safety
/// As [`mapal_read`]; called from inside a task of a launched run.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn mapal_read_task(
    path: *const u8,
    path_len: usize,
    dir: *const u8,
    dir_len: usize,
    dst: *mut u8,
    bytes: u64,
    topo: i64,
) {
    let text = |p: *const u8, n: usize| -> Option<&str> {
        (!p.is_null() && n > 0)
            .then(|| unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(p, n)) })
    };
    let lit = text(path, path_len).unwrap_or("");
    let env = std::env::var(READ_DATA_DIR_ENV).ok();
    let full = read_resolve(lit, env.as_deref(), text(dir, dir_len));
    let t0 = std::time::Instant::now();
    let mut out =
        (!dst.is_null()).then(|| unsafe { std::slice::from_raw_parts_mut(dst, bytes as usize) });
    let moved = if out.is_some() { bytes } else { 0 };
    match read_file_into(&full, out.as_deref_mut(), bytes) {
        Ok(()) => read_trace(&full, moved, t0),
        Err(msg) => {
            if let Some(out) = out {
                out.fill(0);
            }
            lock(&READ_FAILURES).push((topo, msg));
            mapal_par_trap(topo, READ_TRAP_KIND);
        }
    }
}

/// Deliver the read failure recorded at `topo` (exit 103).
pub(crate) fn read_failure_at(topo: i64) -> ! {
    let msg = lock(&READ_FAILURES)
        .iter()
        .find(|(t, _)| *t == topo)
        .map(|(_, m)| m.clone())
        .unwrap_or_else(|| "mapal: read failed".into());
    read_failed(&msg)
}

/// The `mapal_read_trace` line, under `MAPAL_GPU_TRACE` or `MAPAL_READ_TRACE`.
pub(crate) fn read_trace(full: &std::path::Path, moved: u64, t0: std::time::Instant) {
    let on = |v: &str| std::env::var(v).is_ok_and(|t| !t.is_empty() && t != "0");
    if on("MAPAL_READ_TRACE") || on("MAPAL_GPU_TRACE") {
        eprintln!(
            "mapal_read_trace path={} bytes={moved} ms={:.3}",
            full.display(),
            t0.elapsed().as_secs_f64() * 1e3
        );
    }
}
