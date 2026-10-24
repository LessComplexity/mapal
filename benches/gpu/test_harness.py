#!/usr/bin/env python3
"""Exercise GPU smoke status and rental cleanup without CUDA, SSH or a rental."""
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import unittest


HERE = Path(__file__).resolve().parent


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        (self.root / "test").mkdir()
        (self.root / "tiles").mkdir()
        self.calls = self.root / "calls"
        self.calls.touch()
        self.env = {
            **os.environ,
            "PATH": f"{self.bin}:{os.environ['PATH']}",
            "STUB_LOG": str(self.calls),
            "KEEP": "0",
            "PROFILE": "1",
            "START_TRIES": "1",
        }
        self.stub("nvidia-smi", "echo test-device")
        self.stub("nvcc", 'echo nvcc >> "$STUB_LOG"; exit 99')
        self.stub("timeout", 'shift; exec "$@"')
        self.stub("cc", '''
src=$1
case "$src" in *link_fail*) exit 2 ;; esac
while [ "$1" != -o ]; do shift; done
cp "$src" "$2"
chmod +x "$2"
''')

    def executable(self, path, body):
        path.write_text("#!/bin/bash\nset -eu\n" + body + "\n")
        path.chmod(0o755)

    def stub(self, name, body):
        self.executable(self.bin / name, body)

    def value(self, name="px_value", output="42", expected="42", rc=0):
        self.executable(self.root / "test" / f"{name}.host.o",
                        f"printf '%s\\n' {shlex.quote(output)}\nexit {rc}")
        if expected is not None:
            (self.root / "tiles" / f"{name.split('.')[0]}.expect").write_text(expected + "\n")

    def trap(self, rc=101):
        self.executable(self.root / "test/e2e_trap.host.o", f"exit {rc}")

    def run_box(self, success, mode="1", nvcc=False):
        result = subprocess.run(
            ["bash", str(HERE / "box_run.sh"), "test", "sm_89", "1"],
            cwd=self.root, env={**self.env, "DIFFERENTIAL_ONLY": mode},
            # Bound hangs; local process startup latency is not a performance assertion.
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode == 0, success, result.stdout + result.stderr)
        log = (self.root / "results-test.txt").read_text()
        self.assertEqual("== done" in log, success, log)
        self.assertEqual("nvcc" in self.calls.read_text(), nvcc)
        return log

    def test_passing_smoke(self):
        self.value()
        self.value("px_value_c")
        self.trap()
        log = self.run_box(True)
        self.assertIn("differential: 2 pass, 0 fail", log)
        self.assertIn("traps: 1 pass, 0 fail", log)

    def test_mismatch_aggregates_and_stops_full_mode_too(self):
        self.value("px_a", output="wrong")
        self.value("px_z")
        self.trap()
        for mode in ("1", "0"):
            with self.subTest(mode=mode):
                log = self.run_box(False, mode)
                self.assertIn("differential: 1 pass, 1 fail", log)
                self.assertIn("TRAP-PASS", log)

    def test_wrong_trap(self):
        self.value()
        self.trap(rc=0)
        self.assertIn("TRAP-FAIL", self.run_box(False))

    def test_empty_value_scope_even_with_passing_trap(self):
        self.trap()
        log = self.run_box(False)
        self.assertIn("differential: 0 pass, 0 fail", log)
        self.assertIn("TRAP-PASS", log)

    def test_empty_full_mode_reaches_kernel_harness(self):
        log = self.run_box(False, mode="0", nvcc=True)
        self.assertIn("differential: 0 pass, 0 fail", log)
        self.assertIn("HARNESS BUILD FAILED", log)
        self.assertNotIn("VALIDATION FAILED", log)

    def test_nonzero_value_exit(self):
        self.value(rc=7)
        self.assertIn("rc=7", self.run_box(False))

    def test_missing_oracle_cannot_pass_empty_output(self):
        self.value(output="", expected=None)
        self.assertIn("missing oracle", self.run_box(False))

    def test_link_failure_cannot_run_a_stale_binary(self):
        self.value("px_link_fail")
        self.executable(self.root / "prog_px_link_fail", "echo 42")
        self.assertIn("link failed", self.run_box(False))

    def test_failed_tf32_reference_cannot_be_an_oracle(self):
        self.value("px_value_tf", output="wrong", expected=None)
        self.value("px_value_tf.off-epilogue", output="wrong", expected=None, rc=7)
        self.assertIn("TF32 reference failed", self.run_box(False))

    def timing_stubs(self):
        self.stub("date", 'case "$1" in +%s%N) echo 1000000000 ;; *) echo fixture-time ;; esac')
        self.stub("nvcc", '''
echo nvcc >> "$STUB_LOG"
[ "$1" = --version ] && { echo stub-compiler; exit 0; }
while [ "$1" != -o ]; do shift; done
printf '#!/bin/bash\\nexit 0\\n' > "$2"
chmod +x "$2"
''')
        self.env.update({"E2E_REPS": "1", "PROBE_REPS": "1", "E2E_ARMS": "on", "PROBE_ARMS": "on"})

    def test_timed_process_status_after_plausible_output(self):
        # The real pipeline must preserve the program's status through `tr`,
        # and a failed traced run must not pass because its output looks valid.
        self.timing_stubs()
        # The probe's default face answers to its exact oracle within D2's bound (value_gate.py, K).
        (self.root / "value_gate.py").write_text((HERE / "value_gate.py").read_text())
        (self.root / "tiles/pt_mlp_2048_x.expect").write_text("42\n")
        (self.root / "tiles/pt_mlp_2048.kmax").write_text("4096\n")
        for name, tag in (("e2e_1024", "e2e"), ("loop_mm_1024_8", "e2e"), ("pt_mlp_2048", "probe")):
            for phase in ("none", "wall", "trace"):
                with self.subTest(program=name, failure=phase):
                    self.env["FAIL_PHASE"] = phase
                    self.executable(self.root / "test" / f"{name}.host.o", '''
printf '42\\niter ms=1\\n'
if [ "${MAPAL_GPU_TRACE:-0}" = 1 ]; then
    echo 'mapal_gpu_trace_total launches=1 kernel_ms=1' >&2
    [ "$FAIL_PHASE" = trace ] && exit 9
else
    [ "$FAIL_PHASE" = wall ] && exit 7
fi
exit 0
''')
                    log = self.run_box(phase == "none", mode="0", nvcc=True)
                    face = "_default" if tag == "probe" else "_exact"
                    self.assertIn(f"{tag} prog={name} face={face} arm=on rep=1", log)
                    self.assertIn(f"rc={7 if phase == 'wall' else 0} trace_rc={9 if phase == 'trace' else 0}", log)
                    self.assertIn("out=[42 iter ms=1 ]", log)
                    self.assertEqual("VALIDATION FAILED" in log, phase != "none")
                    (self.root / "test" / f"{name}.host.o").unlink()
                    for binary in self.root.glob(f"*_{name}"):
                        binary.unlink()

    def test_decode_status_after_differential_passes(self):
        self.timing_stubs()
        (self.root / "tiles/decode_1024_256.expect").write_text("42\n")
        self.executable(self.root / "test/decode_1024_256.host.o", '''
echo 42
if [ "${MAPAL_GPU_TRACE:-0}" = 1 ]; then
    echo 'mapal_gpu_trace_total launches=1 kernel_ms=1' >&2
    [ "$FAIL_PHASE" = trace ] && exit 9
else
    count=$(cat decode-calls 2>/dev/null || echo 0)
    echo $((count + 1)) > decode-calls
    [ "$count" -gt 0 ] && [ "$FAIL_PHASE" = wall ] && exit 7
fi
exit 0
''')
        for phase in ("none", "wall", "trace"):
            with self.subTest(failure=phase):
                self.env["FAIL_PHASE"] = phase
                (self.root / "decode-calls").write_text("0\n")
                log = self.run_box(phase == "none", mode="0", nvcc=True)
                self.assertIn("differential: 1 pass, 0 fail", log)
                self.assertIn(f"decode prog=decode_1024_256 rep=1 rc={9 if phase == 'trace' else 0} wall_s=", log)
                self.assertIn(f"wall_rc={7 if phase == 'wall' else 0}", log)
                self.assertEqual("VALIDATION FAILED" in log, phase != "none")

    def test_wrapper_retrieves_then_cleans_up_on_success_and_failure(self):
        self.stub("vastai", '''
echo "vastai $*" >> "$STUB_LOG"
case "$1 $2" in
    'create instance') echo '{"new_contract":123}' ;;
    'show instance') echo '{"actual_status":"running"}' ;;
    'show instances') echo '[]' ;;
    'ssh-url 123') echo 'ssh://root@stub:1234' ;;
    'destroy instance') ;;
    *) exit 99 ;;
esac
''')
        self.stub("ssh", '''
command=${!#}
echo "ssh $command" >> "$STUB_LOG"
case "$command" in
    *'bash box_run.sh '*) echo remote-output; exit "$REMOTE_RC" ;;
    *'tar -xzf -'*) cat >/dev/null ;;
    'cat /w/results-test.txt') echo retained-result ;;
    'true') ;;
    *) exit 1 ;;
esac
''')
        bundle = self.root / "bundle"
        bundle.mkdir()
        (bundle / "payload").write_text("stub bundle")
        for rc, mode in ((0, "1"), (7, "1"), (7, "0")):
            with self.subTest(remote_rc=rc, differential_only=mode):
                self.calls.write_text("")
                res = self.root / f"res-{rc}-{mode}"
                result = subprocess.run(
                    ["bash", str(HERE / "vast_run.sh"), "456", "test", "sm_89",
                     str(bundle), str(res), "1"],
                    env={**self.env, "DIFFERENTIAL_ONLY": mode, "REMOTE_RC": str(rc)},
                    capture_output=True, text=True, timeout=30,
                )
                self.assertEqual(result.returncode, rc, result.stdout + result.stderr)
                self.assertEqual((res / "results-test.txt").read_text(), "retained-result\n")
                self.assertIn("remote-output", (res / "run-test.out").read_text())
                calls = self.calls.read_text()
                self.assertIn(f"DIFFERENTIAL_ONLY='{mode}'", calls)
                self.assertLess(calls.index("cat /w/results-test.txt"),
                                calls.index("vastai destroy instance 123 -y"))
                self.assertNotIn("box_profile.sh", calls)


class RuntimeProgramTests(unittest.TestCase):
    """S62: the request clock and the retention loops are where gen_tiles.py puts them."""

    def test_request_clock_and_retention_placement(self):
        with tempfile.TemporaryDirectory() as d:
            subprocess.run(["python3", str(HERE / "gen_tiles.py"), d], check=True, capture_output=True)
            rq = sorted(Path(d).glob("pt_rq_*.mapal"))
            self.assertEqual(len(rq), 12)
            for p in rq:
                s = p.read_text()
                inside = s[s.index("() -> time -> t0;"):s.index("() -> time -> t1;")]
                after = s[s.index("() -> time -> t1;"):]
                base = (Path(d) / p.name.replace("pt_rq_", "pt_")).read_text()
                outs = re.findall(r"    (\w+)\[0\] -> println;", base)
                self.assertTrue(outs, p.name)
                for o in outs:
                    self.assertIn(f"{o}[0] -> r0_{o};", inside, p.name)
                    self.assertIn(f"-> r1_{o};", inside, p.name)
                    self.assertIn(f"r0_{o} -> println;", after, p.name)
                self.assertNotIn("println", inside, p.name)
            for tag in ("rt", "rg"):
                progs = sorted(Path(d).glob(f"pt_{tag}_*.mapal"))
                self.assertEqual(len(progs), 4)
                for p in progs:
                    s = p.read_text()
                    pre, body = s.split("    loop {", 1)
                    immutable = re.findall(r"\(t \* \d+ \+ \d+\) % \d+ - \d+ -> widen_f32 -> v; v / [\d.]+ \} -> (\w+);", s)
                    immutable += re.findall(r"krange -> map \{ t -> t % \d+ - \d+ -> widen_f32 -> v; v / [\d.]+ \} -> (\w+);", s)
                    request = re.findall(r"\(\(t \+ it\) \* \d+ \+ \d+\) % [^}]*\} -> (\w+);", s)
                    self.assertTrue(immutable and request, p.name)
                    for x in request:
                        self.assertIn(f"-> {x};", body, p.name)
                    for x in immutable:
                        self.assertIn(f"-> {x};", body if tag == "rg" else pre, p.name)
                        self.assertNotIn(f"-> {x};", pre if tag == "rg" else body, p.name)


class DefaultFaceBoundTests(unittest.TestCase):
    """flow: every timed program's default face is gated by ADR-0040 D2's bound, never bit for bit."""

    @classmethod
    def setUpClass(cls):
        import runpy
        import sys
        cls.tmp = tempfile.TemporaryDirectory()
        argv, sys.argv = sys.argv, ["gen_tiles.py", cls.tmp.name]
        try:
            cls.g = runpy.run_path(str(HERE / "gen_tiles.py"), run_name="gen_tiles")
        finally:
            sys.argv = argv
        cls.d = Path(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_every_timed_program_has_k_over_its_longest_fold(self):
        for f in sorted(self.d.glob("pt_*.mapal")):
            k = self.d / f"{f.stem}.kmax"
            self.assertTrue(k.exists(), f.stem)
            src = f.read_text()
            lens = {b: int(a) for a, b in re.findall(r"    (\d+) -> iota -> (\w+);", src)}
            folds = [lens[r] for r in re.findall(r", (\w+)\) -> fold", src) if r in lens]
            self.assertGreaterEqual(int(k.read_text()), max(folds, default=0), f.stem)

    def test_facts_on_graph_sweep_outputs_are_inside_the_bound(self):
        import sys
        sys.path.insert(0, str(HERE))
        from value_gate import within
        # (stem, the exact face's CPU oracle, the box's default-face GPU output): split-K (H100, 3090),
        # stream-K (5090) — order free, so the bits differ from the default face's CPU build.
        cases = [("pt_mlp2_2048", "2977.8862 1891.8232 476.01288 754.7543", "2977.8848 1891.8242 476.0133 754.7556"),
                 ("pt_twin_2048", "107.69852 -188.7298 25.11887 -7.992958", "107.69845 -188.7301 25.118898 -7.9929657"),
                 ("pt_pj_m16_n2048_k7168", "48.000435 28.635628", "48.000366 28.634766"),
                 ("pt_pj_m16_n2048_k7168", "48.000435 28.635628", "48.00029 28.63475")]
        for stem, exact, got in cases:
            k = int((self.d / f"{stem}.kmax").read_text())
            self.assertTrue(within(exact, got, k)[0], stem)
            self.assertFalse(within(exact, " ".join(str(float(x) * 1.01) for x in got.split()), k)[0], stem)

    def test_fold_chain_bounds_generated_chains(self):
        import random
        rng = random.Random(7)
        g = self.g
        for _ in range(200):
            ops, body, true = [("x0", 16, 'a'), ("x1", 16, 'b')], "", {"x0": 0, "x1": 0}
            for i in range(rng.randint(1, 6)):
                a, b = rng.choice(list(true)), rng.choice(list(true))
                K = rng.choice((1, 3, 64, 512, 2048))
                body += g["gsite"](f"s{i}", a, b, 4, 4, K)
                true[f"s{i}"] = K + max(true[a], true[b])
            outs = [(o, 16) for o in true if o.startswith("s")]
            k = g["fold_chain"](g["composed"](ops, body, outs))
            longest = max(true.values())
            self.assertGreaterEqual(k, longest)
            self.assertLessEqual(k, longest + 16 * len(outs) + 16)


if __name__ == "__main__":
    unittest.main()
