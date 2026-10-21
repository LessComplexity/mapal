#!/usr/bin/env python3
"""Exercise GPU smoke status and rental cleanup without CUDA, SSH or a rental."""
import os
from pathlib import Path
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
            capture_output=True, text=True, timeout=10,
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
                    capture_output=True, text=True, timeout=10,
                )
                self.assertEqual(result.returncode, rc, result.stdout + result.stderr)
                self.assertEqual((res / "results-test.txt").read_text(), "retained-result\n")
                self.assertIn("remote-output", (res / "run-test.out").read_text())
                calls = self.calls.read_text()
                self.assertIn(f"DIFFERENTIAL_ONLY='{mode}'", calls)
                self.assertLess(calls.index("cat /w/results-test.txt"),
                                calls.index("vastai destroy instance 123 -y"))
                self.assertNotIn("box_profile.sh", calls)


if __name__ == "__main__":
    unittest.main()
