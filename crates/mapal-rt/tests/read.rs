//! ADR-0045: the runtime's copy of the `read` rule is the IR's. `mapal-rt`
//! links into every emitted program and depends on nothing, so it carries
//! its own path resolution and failure messages; these replay
//! `mapal_ir::read`'s against them case by case.

use mapal_rt::{READ_DATA_DIR_ENV, READ_EXIT_CODE, read_file_into, read_resolve};

#[test]
fn constants_agree() {
    assert_eq!(READ_DATA_DIR_ENV, mapal_ir::read::DATA_DIR_ENV);
    assert_eq!(READ_EXIT_CODE, mapal_ir::read::EXIT_CODE);
}

#[test]
fn resolution_agrees() {
    let paths = [
        "w.bin",
        "weights/embedding.bin",
        "/abs/w.bin",
        "../up.bin",
        "./here.bin",
    ];
    let envs = [None, Some(""), Some("/env"), Some("rel/env")];
    let dirs = [None, Some("/src/dir"), Some("")];
    for p in paths {
        for e in envs {
            for d in dirs {
                assert_eq!(
                    read_resolve(p, e, d),
                    mapal_ir::read::resolve_with(p, e, d),
                    "{p:?} {e:?} {d:?}"
                );
            }
        }
    }
}

#[test]
fn messages_agree() {
    let dir = tempfile::tempdir().unwrap();
    let short = dir.path().join("short.bin");
    std::fs::write(&short, [7u8; 10]).unwrap();
    // Wrong length.
    assert_eq!(
        read_file_into(&short, None, 12),
        Err(mapal_ir::read::size_message(&short, 12, 10))
    );
    assert_eq!(
        mapal_ir::read::load(short.to_str().unwrap(), None, 12),
        Err(mapal_ir::read::size_message(&short, 12, 10))
    );
    // Missing, and a directory.
    for p in [dir.path().join("missing.bin"), dir.path().to_path_buf()] {
        let want = mapal_ir::read::load(p.to_str().unwrap(), None, 4).unwrap_err();
        assert_eq!(read_file_into(&p, None, 4), Err(want), "{}", p.display());
    }
    // The right length: the bytes, once.
    let mut buf = [0u8; 10];
    assert_eq!(read_file_into(&short, Some(&mut buf), 10), Ok(()));
    assert_eq!(buf, [7u8; 10]);
    assert_eq!(
        mapal_ir::read::load(short.to_str().unwrap(), None, 10),
        Ok(vec![7u8; 10])
    );
}
