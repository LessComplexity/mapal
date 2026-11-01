//! Source formatting effects. The syntax crate owns every layout decision.

use std::fs::{self, OpenOptions};
use std::io::{self, Read, Write};
use std::path::Path;

use mapal_syntax::{LineIndex, Severity, format_source};

use crate::Res;

const HELP: &str = "usage: mapal fmt [--check | --stdout] <files... | ->
  <files...>  format explicit files in place
  --check     report files needing formatting; write nothing (exit 1 on differences)
  --stdout    preview one file on stdout; write nothing
  -           read stdin; write formatted source to stdout (or check with --check)
  --          treat following arguments as file paths
  -h, --help  show this help

Uses four spaces, multiline braces, and consistent spacing. Preserves comments
and literal spelling. Invalid input is never replaced. No LLVM toolchain needed.";

#[derive(Default)]
struct Options {
    check: bool,
    stdout: bool,
    files: Vec<String>,
}

fn usage(message: &str) -> u8 {
    eprintln!("mapal fmt: {message}\n{HELP}");
    2
}

fn options(args: impl Iterator<Item = String>) -> Res<Option<Options>> {
    let mut options = Options::default();
    let mut paths_only = false;
    for arg in args {
        if paths_only {
            options.files.push(arg);
            continue;
        }
        match arg.as_str() {
            "-h" | "--help" => {
                println!("{HELP}");
                return Ok(None);
            }
            "--check" => options.check = true,
            "--stdout" => options.stdout = true,
            "--" => paths_only = true,
            "-" => options.files.push(arg),
            _ if arg.starts_with('-') => {
                return Err(usage(&format!("unknown argument `{arg}`")));
            }
            _ => options.files.push(arg),
        }
    }
    if options.files.is_empty() {
        return Err(usage("provide explicit files or `-` for stdin"));
    }
    if options.check && options.stdout {
        return Err(usage("--check and --stdout cannot be combined"));
    }
    if options.stdout && options.files.len() != 1 {
        return Err(usage("--stdout requires exactly one input"));
    }
    if options.files.iter().any(|file| file == "-") && options.files.len() != 1 {
        return Err(usage("stdin (`-`) cannot be combined with other inputs"));
    }
    Ok(Some(options))
}

struct Input {
    name: String,
    source: String,
    formatted: String,
}

pub(super) fn run(args: impl Iterator<Item = String>) -> Res<u8> {
    let Some(options) = options(args)? else {
        return Ok(0);
    };

    // Collect and validate the whole explicit batch before performing any write.
    let mut inputs = Vec::with_capacity(options.files.len());
    let mut failed = false;
    for name in options.files {
        let source = if name == "-" {
            let mut source = String::new();
            io::stdin().read_to_string(&mut source).map(|_| source)
        } else {
            fs::read_to_string(&name)
        };
        let source = match source {
            Ok(source) => source,
            Err(error) => {
                eprintln!("mapal fmt: cannot read {name}: {error}");
                failed = true;
                continue;
            }
        };
        let formatted = match format_source(&source) {
            Ok(formatted) => formatted,
            Err(diagnostics) => {
                let lines = LineIndex::new(&source);
                for diagnostic in diagnostics {
                    let severity = match diagnostic.severity {
                        Severity::Error => "error",
                        Severity::Warning => "warning",
                    };
                    eprintln!(
                        "{} {severity}[{}] (fmt) {}",
                        crate::at(&name, &lines, diagnostic.span.start),
                        diagnostic.code.0,
                        diagnostic.message
                    );
                }
                failed = true;
                continue;
            }
        };
        inputs.push(Input {
            name,
            source,
            formatted,
        });
    }
    if failed {
        return Err(1);
    }

    if options.check {
        let mut different = false;
        for input in inputs {
            if input.source != input.formatted {
                eprintln!("{}: needs formatting", input.name);
                different = true;
            }
        }
        return Ok(u8::from(different));
    }
    if options.stdout || inputs[0].name == "-" {
        io::stdout()
            .lock()
            .write_all(inputs[0].formatted.as_bytes())
            .map_err(|error| {
                eprintln!("mapal fmt: cannot write stdout: {error}");
                1
            })?;
        return Ok(0);
    }

    for input in inputs {
        if input.source == input.formatted {
            continue;
        }
        replace_file(Path::new(&input.name), &input.source, &input.formatted).map_err(|error| {
            eprintln!("mapal fmt: cannot write {}: {error}", input.name);
            1
        })?;
    }
    Ok(0)
}

/// Write beside the destination, then rename: failed writes never truncate source.
/// Resolve symlinks so formatting a symlink preserves the link itself.
fn replace_file(path: &Path, original: &str, formatted: &str) -> io::Result<()> {
    let path = fs::canonicalize(path)?;
    // Multiple explicit paths may name the same file (including through symlinks).
    if fs::read_to_string(&path)? == formatted {
        return Ok(());
    }
    let permissions = fs::metadata(&path)?.permissions();
    if permissions.readonly() {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "file is read-only",
        ));
    }
    let parent = path.parent().expect("canonical file has a parent");
    for attempt in 0..100 {
        let temporary = parent.join(format!(".mapal-fmt-{}-{attempt}.tmp", std::process::id()));
        let mut file = match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
        {
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            result => result?,
        };
        let result = (|| {
            file.set_permissions(permissions)?;
            file.write_all(formatted.as_bytes())?;
            file.sync_all()?;
            drop(file);
            if fs::read_to_string(&path)? != original {
                return Err(io::Error::other("file changed while formatting; retry"));
            }
            fs::rename(&temporary, &path)
        })();
        if result.is_err() {
            let _ = fs::remove_file(&temporary);
        }
        return result;
    }
    Err(io::Error::new(
        io::ErrorKind::AlreadyExists,
        "cannot create a temporary output file",
    ))
}
