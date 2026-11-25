#!/usr/bin/env python3
"""Exercise GPU smoke status and rental cleanup without CUDA, SSH or a rental."""
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
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
            "PREFLIGHT": "0",  # the stub bundles hold no harness (MeasureTests checks preflight.py)
            "CUDA_CACHE_PATH": str(self.root / "jit"),  # box_run.sh checks the driver's JIT cache is writable
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

    def test_box_tunes_gates_every_candidate_and_carries_rules_only_programs(self):
        """per-site-choice B5 on stub binaries: the value gate runs every candidate index; one traced tune run per
        tabled stem writes the wisdom; a program whose picks are all candidate 0 is carried; the others time AUTO
        (`on`, reading the wisdom) against RULES; a failing candidate drops AUTO with it."""
        self.timing_stubs()
        self.env.pop("PROBE_ARMS")
        for name, key, win, bad in (("pt_a_64", "aa", 1, 0), ("pt_b_64", "bb", 0, 0), ("pt_c_64", "cc", 1, 1)):
            (self.root / "tiles" / f"{name}.expect").write_text("42\n")
            (self.root / "test" / f"{name}.sites").write_text(
                f"mapal_tile_{key:0>16} 0 2 k0 1 1 32 0\nmapal_tile_{key:0>16} 1 2 k1 1 1 32 0\n")
            self.executable(self.root / "test" / f"{name}.host.o", f'''
p=${{MAPAL_GPU_PICK:-auto}}; pick=0; src=rules; w="$MAPAL_WISDOM/wisdom-{key}.tsv"
if [ "$p" = rules ]; then src=forced
elif [ "$p" != auto ] && [ "$p" != tune ]; then pick=$p; src=forced
elif [ "${{MAPAL_GPU_TUNE:-0}}" = 1 ]; then
    pick={win}; src=trial; mkdir -p "$MAPAL_WISDOM"; printf '{key:0>16}\\t1\\t2\\t%s\\t5\\t100\\t50\\tk0\\tk1\\n' $pick >> "$w"
elif [ -s "$w" ]; then pick={win}; src=wisdom
fi
if [ "$pick" = 1 ] && [ {bad} = 1 ]; then echo 41; else echo 42; fi
echo 'iter ms=1'
if [ "${{MAPAL_GPU_TRACE:-0}}" = 1 ]; then
    echo "mapal_gpu_trace op=site site={key:0>16} key={key:0>16} n=2 pick=$pick src=$src" >&2
    echo 'mapal_gpu_trace_total module_ms=3.5 launches=1' >&2
fi
exit 0
''')
        log = self.run_box(True, mode="0", nvcc=True)
        self.assertIn("value-gate prog=pt_a_64 arms=4 pass", log)  # on, rules, pick0, pick1
        self.assertIn("tune prog=pt_a_64 face=_default rc=0 sites=1 trials=0 picks=[00000000000000aa:1:trial ] "
                      "module_ms=3.5", log)
        self.assertIn("carried prog=pt_b_64 (pick = rules everywhere)", log)
        self.assertNotIn("probe prog=pt_b_64", log)
        self.assertRegex(log, r"probe prog=pt_a_64 face=_default arm=on rep=1 .* sites=\[00000000000000aa:1:wisdom \]")
        self.assertIn("probe prog=pt_a_64 face=_default arm=rules rep=1", log)
        self.assertIn("VALUE-FAIL prog=pt_c_64 face=_exact arm=pick1", log)  # (an exact oracle's label)
        self.assertNotIn("probe prog=pt_c_64 face=_default arm=on", log)
        self.assertIn("probe prog=pt_c_64 face=_default arm=rules", log)
        self.assertIn("wisdom: 3 decisions, 2 not candidate 0; carried 1 programs", log)

    def test_choice_rows_judge_the_wisdom_and_auto_against_rules(self):
        import sys
        sys.path.insert(0, str(HERE))
        import evidence
        part = self.root / "rtx-4090"
        part.mkdir()
        ids = [5, -7]
        (part / "pt_a_64.host.ll").write_text(
            "  call void @mapal_gpu_site(i64 5, i32 1, i32 0, i32 2, ptr @m.cands1, i32 3, ptr %p, i32 2, ptr null)\n"
            "@m.cands1 = private unnamed_addr constant [2 x %mapal_cand] ["
            + ", ".join(f"%mapal_cand {{ i64 {i}, i64 0, i64 0, i64 0, i32 1, i32 1, i32 32, i32 0, i32 0, i32 0 }}"
                        for i in ids) + "]\n")
        t = evidence.site_tables((part / "pt_a_64.host.ll").read_text())
        self.assertEqual([c["kernel"] for c in t[0]["cands"]], [evidence.kname(i) for i in ids])
        k = "\t".join(evidence.kname(i) for i in ids)
        (self.root / "wisdom").mkdir()
        (self.root / "wisdom" / "wisdom-x.tsv").write_text(
            f"00000000000000aa\t0000000000000005\t2\t1\t5\t100000\t90000\t{k}\n"  # 10 % faster: pick 1
            f"00000000000000bb\t0000000000000005\t2\t1\t5\t100000\t98000\t{k}\n")  # 2 %: within the margin
        res = self.root / "results.txt"
        res.write_text("".join(
            f"probe prog=pt_a_64 face=_default arm={a} rep={r} wall_us=1 out=[1 iter ms={ms} ] rc=0{x}\n"
            for r in (1, 2) for a, ms, x in (("on", 9.0, " sites=[00000000000000aa:1:wisdom ]"), ("rules", 10.0, ""),
                                             ("null", 9.05, ""))))
        rows = evidence.choice_rows("rtx-4090", str(part), {"pt_a_64": {}}, str(res))
        got = {(r[4], r[5]): r[7] for r in rows}
        self.assertEqual(got[("stored pick = measured argmin beyond max(3 %, 1 us)", "pick=1")], "ok")
        self.assertEqual(got[("stored pick = measured argmin beyond max(3 %, 1 us)", "pick=0")], "MISMATCH")
        self.assertEqual(got[("AUTO launches the stored pick (src=wisdom)", "pick=1 src=wisdom")], "ok")
        self.assertEqual(got[("AUTO <= RULES x (1 + null)", "auto <= rules*(1+null)")], "ok")

    def test_choice_rows_read_a_confirm_revert_as_the_runtime_does(self):
        # bf16-integrate: a trial's pick that confirm_pick.py reverts (`key site n 0 0`, no minima) is the key's
        # decision — the runtime launches candidate 0 from it; the checker kept the trial's pick and flagged 18
        # launches. AUTO then launches what RULES does: its time against RULES is null by construction.
        import sys
        sys.path.insert(0, str(HERE))
        import evidence
        part = self.root / "rtx-5080"
        part.mkdir()
        ids = [5, -7]
        (part / "pt_a_64.host.ll").write_text(
            "  call void @mapal_gpu_site(i64 5, i32 1, i32 0, i32 2, ptr @m.cands1, i32 3, ptr %p, i32 2, ptr null)\n"
            "@m.cands1 = private unnamed_addr constant [2 x %mapal_cand] ["
            + ", ".join(f"%mapal_cand {{ i64 {i}, i64 0, i64 0, i64 0, i32 1, i32 1, i32 32, i32 0, i32 0, i32 0 }}"
                        for i in ids) + "]\n")
        k = "\t".join(evidence.kname(i) for i in ids)
        (self.root / "wisdom").mkdir()
        (self.root / "wisdom" / "wisdom-x.tsv").write_text(
            f"00000000000000aa\t0000000000000005\t2\t1\t5\t100000\t90000\t{k}\n"  # the trial: pick 1
            "00000000000000aa\t0000000000000005\t2\t0\t0\n")  # confirm_pick.py: reverted to 0
        w = evidence.wisdom_lines(str(self.root / "wisdom" / "wisdom-x.tsv"))["00000000000000aa"]
        self.assertEqual((w["pick"], w["trial"], w["ns"]), (0, 1, [100000, 90000]))
        res = self.root / "results.txt"
        res.write_text("".join(
            f"probe prog=pt_a_64 face=_default arm={a} rep={r} wall_us=1 out=[1 iter ms={ms} ] rc=0{x}\n"
            for r in (1, 2, 3) for a, ms, x in (("on", 10.2, " sites=[00000000000000aa:0:wisdom ]"),
                                                ("rules", 10.0, ""), ("null", 10.21, ""))))
        rows = evidence.choice_rows("rtx-5080", str(part), {"pt_a_64": {}}, str(res))
        got = {(r[4], r[5]): (r[6], r[7]) for r in rows}
        self.assertEqual(got[("AUTO launches the stored pick (src=wisdom)", "pick=0 src=wisdom")][1], "ok")
        argmin = got[("stored pick = measured argmin beyond max(3 %, 1 us)", "pick=1")]
        self.assertEqual(argmin, ("pick=1 ns=100000,90000 confirm=0", "ok"))
        auto = got[("AUTO <= RULES x (1 + null)", "auto <= rules*(1+null)")]
        self.assertIn("same launches", auto[0])
        self.assertEqual(auto[1], "ok")
        # A trial whose head-to-head pairs kept candidate 0 (cross-arch) stores 0 although its minima's argmin is
        # the challenger (the pairs' runs enter the minima): candidate 0 is the trial's decision.
        (self.root / "wisdom" / "wisdom-x.tsv").write_text(
            f"00000000000000cc\t0000000000000005\t2\t0\t5\t15519\t13216\t{k}\n")
        (self.root / "tune").mkdir()
        (self.root / "tune" / "pt_a_64.trace").write_text("".join(
            f"mapal_gpu_trace op=trial site=0000000000000005 cand={c} kernel=x rep={5 + i // 2} ns={ns} confirm=1\n"
            for i, (c, ns) in enumerate([(0, 15807), (1, 19999), (1, 13216), (0, 15744), (0, 15519), (1, 13344)]))
            + "mapal_gpu_trace op=site site=0000000000000005 key=00000000000000cc n=2 pick=0 src=trial reps=5\n")
        rows = evidence.choice_rows("rtx-5080", str(part), {"pt_a_64": {}}, str(res))
        got = {(r[4], r[5]): (r[6], r[7]) for r in rows}
        self.assertEqual(got[("stored pick = measured argmin beyond max(3 %, 1 us)", "pick=0")],
                         ("pick=0 ns=15519,13216 pairs=veto", "ok"))

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
[ "$1" = --profile ] && { echo "profile $2" >> "$STUB_LOG"; shift 2; }
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
                # Every rental call goes through the itur account, never the default profile.
                self.assertEqual(calls.count("profile itur\n"), calls.count("vastai "))


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


class MeasureTests(unittest.TestCase):
    """site-geometry A: changed.py env / ingest, preflight.py, search.py plan + pick on a stub knobs.json."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.env = {**os.environ, "MAPAL_RESULT_CACHE": str(self.root / "cache")}

    def py(self, *args, ok=True, env=None):
        r = subprocess.run(["python3", *map(str, args)], capture_output=True, text=True,
                           env={**self.env, **(env or {})}, timeout=600)
        if ok:
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        return r

    def tsv(self, name, rows):
        (self.root / name).write_text("\n".join("\t".join(r) for r in rows) + "\n")

    def test_env_selects_changed_rows_and_cause_arms(self):
        self.tsv("emission.tsv", [["program", "arch", "face", "status", "b", "h"],
                                  ["pt_a_64", "rtx-4090", "default", "changed", "1", "2"],
                                  ["pt_a_64", "rtx-4090", "contract", "changed", "1", "3"],
                                  ["pt_b_64", "rtx-4090", "default", "same", "1", "1"],
                                  ["pt_c_64", "x86-64-v3", "default", "changed", "1", "4"],
                                  ["mm_16x32x64", "rtx-4090", "exact", "new", "-", "5"],
                                  ["cell_8", "rtx-4090", "exact", "changed", "1", "6"],
                                  ["cell_9", "rtx-4090", "default", "changed", "1", "7"],
                                  ["pt_d_64", "h100-pcie", "default", "changed", "1", "8"],
                                  ["pt_e_64", "rtx-4090", "default", "cached", "1", "9"]])
        self.tsv("arms.tsv", [["program", "arch", "face", "arm", "hash", "tag"],
                              ["pt_a_64", "rtx-4090", "default", "off-x", "a", "cause"],
                              ["pt_a_64", "rtx-4090", "contract", "off-y", "b", "carried"],
                              ["cell_8", "rtx-4090", "exact", "off-z", "c", "new"]])
        out = self.py(HERE / "changed.py", "env", self.root, "rtx-4090").stdout
        got = dict(re.findall(r"^(\w+)='(.*)'$", out, re.M))
        self.assertEqual(got["SUITE"], "pt_a_64 pt_c_64")
        self.assertEqual(got["SUITE_C"], "pt_a_64")
        self.assertEqual(got["CELLS"], "16x32x64")
        self.assertEqual(got["ONLY"], "cell_8 cell_9")  # cell_9's default face: D2's bound, given a K
        self.assertEqual((got["DEFAULT"], got["UNCHECKED"]), ("", "cell_9"))  # no tiles/cell_9.kmax
        self.assertEqual(got["ARMS"], "rtx-4090:pt_a_64::x:off")  # the carried arm stays out
        self.assertEqual(got["HOST_RUNGS"], "z")
        every = self.py(HERE / "changed.py", "env", self.root, "rtx-4090", env={"ALL_ARMS": "1"}).stdout
        self.assertIn("rtx-4090:pt_a_64:_c:y:off", every)

    def test_ingest_files_mapal_and_library_rows(self):
        (self.root / "tiles").mkdir()
        (self.root / "tiles" / "pt_a_64.mapal").write_text("prog")
        self.tsv("emission.tsv", [["program", "arch", "face", "status", "b", "h"],
                                  ["pt_a_64", "rtx-4090", "default", "changed", "1", "h0"],
                                  ["pt_a_64", "x86-64-v3", "default", "changed", "1", "hc"]])
        self.tsv("arms.tsv", [["program", "arch", "face", "arm", "hash", "tag"],
                              ["pt_a_64", "rtx-4090", "default", "off-x", "hx", "cause"]])
        res = self.root / "results.txt"
        res.write_text(
            "name, driver_version, clocks\nNVIDIA X, 570.1, 3000\n"
            "Cuda compilation tools, release 12.8, V12.8.93\ntorch version=2.11.0 cuda=12.8\n"
            + "".join(f"probe prog=pt_a_64 face=_default arm={a} rep={r} wall_us=1 out=[1 iter ms={t + r} ] rc=0\n"
                      for a, t in (("on", 10), ("off-x", 20), ("null", 10), ("cpu", 90)) for r in (1, 2))
            + "chain probe=a n=64 leg=cublas launches=1 reps=3 min_ms=1 med_ms=2.5 max_ms=3\n"
            + "torch prog=pt_a_64 mode=eager n=3 min_ms=1 med_ms=3.5 max_ms=4 within=yes\n")
        self.py(HERE / "changed.py", "ingest", self.root, "rtx-4090", res)
        rows = [l.split("\t") for l in (self.root / "cache" / "rtx-4090.tsv").read_text().splitlines()[1:]]
        got = {(r[0], r[1], r[2]): r[3].split()[0] for r in rows}
        self.assertEqual(got[("h0", "mapal", "-|12.8|570.1")], "med_ms=11.5")
        self.assertEqual(got[("hx", "mapal", "-|12.8|570.1")], "med_ms=21.5")
        self.assertEqual(got[("h0", "mapal-null", "-|12.8|570.1")], "med_ms=11.5")
        self.assertEqual(got[("hc", "mapal", "-|12.8|570.1")], "med_ms=91.5")
        src = hashlib.sha256(b"prog").hexdigest()[:16]
        self.assertEqual(got[(src, "chain-cublas", "12.8")], "med_ms=2.5")
        self.assertEqual(got[(src, "torch-eager", "2.11.0")], "med_ms=3.5")
        self.assertTrue(all(r[4] == "NVIDIA X|570.1" for r in rows))

    def test_ingest_keys_a_tabled_row_by_its_selection(self):
        self.tsv("emission.tsv", [["program", "arch", "face", "status", "b", "h"],
                                  ["pt_a_64", "rtx-4090", "default", "changed", "1", "h0"],
                                  ["pt_b_64", "rtx-4090", "default", "changed", "1", "h1"]])
        self.tsv("arms.tsv", [["program", "arch", "face", "arm", "hash", "tag"]])
        self.tsv("tables.tsv", [["program", "arch", "face", "head_hash", "tables", "max_n"],
                                ["pt_a_64", "rtx-4090", "default", "h0", "1", "2"],
                                ["pt_b_64", "rtx-4090", "default", "h1", "1", "2"]])
        res = self.root / "results.txt"
        res.write_text("name, driver_version, clocks\nNVIDIA X, 570.1, 3000\n"
                       + "".join(f"probe prog=pt_a_64 face=_default arm={a} rep=1 wall_us=1 out=[1 iter ms={t} ] rc=0\n"
                                 for a, t in (("on", 9), ("rules", 10), ("pick1", 8)))
                       + "carried prog=pt_b_64 (pick = rules everywhere)\n")
        self.py(HERE / "changed.py", "ingest", self.root, "rtx-4090", res)
        rows = [l.split("\t") for l in (self.root / "cache" / "rtx-4090.tsv").read_text().splitlines()[1:]]
        got = {r[0]: r[3].split()[0] for r in rows}
        self.assertEqual((got["h0@auto"], got["h0@rules"], got["h0@pick1"]), ("med_ms=9", "med_ms=10", "med_ms=8"))
        self.assertEqual(got["h1@auto"], "carried")

    def bundle(self):
        b = self.root / "bundle"
        (b / "p").mkdir(parents=True)
        (b / "tiles").mkdir()
        (b / "cutlass/include/cutlass").mkdir(parents=True)
        (b / "libmapal_rt.a").write_bytes(b"!<arch>\n" + b"\0" * 60 + b"\x7fELF" + b"\0" * 14 + b"\x3e\x00")
        for f in ("mapal_gpu.c", "mapal_choice.h", "tile_bench.cu", "chain_bench.cu", "value_gate.py"):
            (b / f).touch()
        (b / "box_run.sh").write_text("case $probe in epi|decode_*) ;; *) continue ;; esac\n")
        (b / "torch_bench.py").write_text(
            "import torch\ndef workload(stem):\n    if stem.startswith('pt_fir_'): return None, ()\n"
            "    raise LookupError(stem)\n")
        for s in ("cell_8", "cell_8_c.off-x", "pt_epi_64", "pt_fir_9_4", "pt_odd_64", "pt_odd_64_c"):
            (b / "p" / f"{s}.host.o").touch()
        for s in ("cell_8", "cell_8_c", "pt_epi_64", "pt_fir_9_4", "pt_odd_64"):
            (b / "tiles" / f"{s}.expect").write_text("1\n")
        return b

    def test_preflight_passes_a_whole_bundle(self):
        b = self.bundle()
        r = self.py(HERE / "preflight.py", b, "p", ok=False, env={"NO_LEG": "pt_odd_*"})
        self.assertIn("pt_odd_64_c: no oracle", r.stdout)
        (b / "tiles" / "pt_odd_64_c.expect").write_text("1\n")
        r = self.py(HERE / "preflight.py", b, "p", env={"NO_LEG": "pt_odd_*"})
        self.assertIn("preflight p: OK", r.stdout)

    def test_preflight_fails_what_box_run_would_stop_on(self):
        b = self.bundle()
        (b / "tiles" / "cell_8_c.expect").unlink()
        shutil.rmtree(b / "cutlass")
        (b / "libmapal_rt.a").write_bytes(b"\xcf\xfa\xed\xfe")  # Mach-O
        r = self.py(HERE / "preflight.py", b, "p", ok=False)
        self.assertEqual(r.returncode, 1)
        for want in ("cell_8_c.off-x: missing oracle tiles/cell_8_c.expect", "cutlass/include/cutlass missing",
                     "not x86-64 ELF", "pt_odd_64: no library leg", "pt_odd_64_c: no oracle"):
            self.assertIn(want, r.stdout)
        self.assertNotIn("pt_epi_64: no library leg", r.stdout)  # chain_bench's
        self.assertNotIn("pt_fir_9_4: no library leg", r.stdout)  # torch_bench's

    def test_targets_maps_touched_files_to_test_targets(self):
        t = HERE.parent.parent / "tests/targets.py"
        src = self.py(t, "crates/backends/llvm/src/func/splitk.rs").stdout.splitlines()
        self.assertEqual(src[:2], ["cargo test --release -p mapal-backend-llvm --test splitk",
                                   "cargo test --release -p mapal-backend-llvm --lib"])
        self.assertIn("cargo test --release -p mapal-backend-llvm --test differential", src)
        shared = self.py(t, "crates/mapal-rewrite/tests/testgen/mod.rs").stdout.splitlines()
        self.assertIn("cargo test --release -p mapal-backend-llvm --test differential", shared)
        self.assertIn("cargo test --release -p mapal-rewrite --test property", shared)
        self.assertEqual(self.py(t, "benches/gpu/changed.py").stdout.strip(),
                         "python3 -m unittest benches/gpu/test_harness.py")

    def test_search_plan_and_pick_on_a_stub_table(self):
        real = HERE.parent.parent / "target/release/examples/emit"
        if not real.exists():
            self.skipTest("no release emitter (cargo build --release --example emit -p mapal-backend-llvm)")
        stub = self.root / "geom_emit.py"
        # The stub table's knobs are the emitter's measurement overrides; `sk` a and b are one emission.
        stub.write_text(
            "#!/usr/bin/env python3\nimport os, sys\nargs = []\nfor a in sys.argv[1:]:\n"
            "    if a.startswith('--geom='):\n        k, v = a[7:].split('=', 1)\n"
            "        a = {'splitk': '--splitk=' + v, 'kc': '--kc' if v == '1' else None,\n"
            "             'sk': '--splitk=2'}[k]\n    if a:\n        args.append(a)\n"
            f"os.execv({str(real)!r}, [{str(real)!r}, *args])\n")
        stub.chmod(0o755)
        knobs = self.root / "knobs.json"
        knobs.write_text(json.dumps({"rtx-4090": {"splitk": [0, 2, 4, 8], "kc": [0, 1], "sk": [0, "a", "b"]}}))
        out = self.root / "s"
        progs = ["mm_16x2048x7168", "pt_pj_m1_n2048_k7168", "pt_rowsum_2048"]
        env = {"EMIT": str(stub)}
        arms = self.py(HERE / "search.py", "plan", out, knobs, "rtx-4090", *progs, env=env).stdout.split()
        plan = {(r[0], r[2], r[3]): (r[5], r[6]) for r in
                (l.split("\t") for l in (out / "plan.tsv").read_text().splitlines()[1:])}
        self.assertEqual(plan[("pt_rowsum_2048", "splitk", "2")][1], "drop")  # no site the knob reaches
        self.assertEqual(plan[("mm_16x2048x7168", "kc", "1")][1], "drop")
        # (fill-wave ships S=4 on this cell: splitk=4 is the shipped build.)
        self.assertEqual(plan[("mm_16x2048x7168", "splitk", "4")][1], "drop")
        self.assertEqual(plan[("mm_16x2048x7168", "splitk", "2")][1], "build")
        self.assertEqual(plan[("mm_16x2048x7168", "sk", "a")], ("on-g-splitk-2", "alias"))
        self.assertIn("rtx-4090:pt_pj_m1_n2048_k7168::splitk=2:geom", arms)
        self.assertFalse(any(":sk=" in a or ":kc=" in a for a in arms))
        # The box run: splitk=2 is 20% (probe) and 10% (cell) faster; nulls 0.5% and 1%.
        probe = {"on": 10.0, "null": 10.05, "on-g-splitk-2": 8.0, "on-g-splitk-4": 9.0, "on-g-splitk-8": 9.5}
        cell = {"": 1.0, ".null": 1.01, ".on-g-splitk-2": 0.9}
        lines = ["name, driver_version, x", "NVIDIA X, 570.1, 1"]
        for rep in (1, 2):
            lines += [f"probe prog=pt_pj_m1_n2048_k7168 face=_default arm={a} rep={rep} wall_us=1 "
                      f"out=[1 iter ms={t} ] rc=0" for a, t in probe.items()]
            lines += [f"pass={rep} cell=16x2048x7168 time leg={l} M=16 N=2048 K=7168 n=30 min_ms=1 med_ms={t} max_ms=1"
                      for l, t in [*((f"mm_16x2048x7168{a}", t) for a, t in cell.items()), ("cublas", 2.0)]]
        res = self.root / "results-rtx-4090.txt"
        res.write_text("\n".join(lines) + "\n")
        best = self.py(HERE / "search.py", "pick", out, "rtx-4090", res, "--knobs", knobs).stdout
        rows = {r[1]: r for r in (l.split("\t") for l in (out / "best-rtx-4090.tsv").read_text().splitlines()[1:])}
        self.assertEqual((rows["splitk"][3], rows["splitk"][9]), ("2", "ship"), best)
        self.assertEqual(rows["splitk"][6], "0.010")  # the largest null over the programs it reaches
        self.assertEqual(rows["kc"][9], "no-site")
        self.assertEqual((rows["sk"][3], rows["sk"][9]), ("a", "ship"))
        self.assertTrue(rows["splitk"][10].endswith("results-rtx-4090.txt") and rows["splitk"][11] == "NVIDIA X|570.1")
        # Measured verdicts are cached by (arm, shipped): the same plan builds nothing again.
        self.py(HERE / "search.py", "plan", out, knobs, "rtx-4090", *progs, env=env)
        st = [l.split("\t")[6] for l in (out / "plan.tsv").read_text().splitlines()[1:]]
        self.assertNotIn("build", st)
        self.assertIn("cached", st)


class PlanCostFoldTests(unittest.TestCase):
    def test_folds_split_by_program_across_rung_classes(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("plancost", HERE / "plancost.py")
        pc = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pc)
        # a program's site-geometry row and its `@pc` row in different rung classes (the class-hashed split's leak)
        rows = [(f"p{i}{tag}", cls, 0.02, []) for i in range(12) for tag, cls in (("", "splitk"), ("@pc", "g-tile"))]
        fs = pc.folds(rows)
        self.assertEqual(len(fs), 2 * len(pc.SALTS))
        for fit, hold in fs:
            base = lambda part: {x[0].split("@")[0] for x in part}
            self.assertFalse(base(fit) & base(hold))
            self.assertEqual(sorted(fit + hold), sorted(rows))
        self.assertEqual(fs[0], fs[1][::-1])

    def test_scale_report_on_synthetic_results(self):
        # per-site-choice ladders: AUTO / RULES / legs / null / crossover / a failed torch leg / a cell's site pick.
        r = subprocess.run(["python3", str(HERE / "scale_report.py"), "demo"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


class CompetitorTests(unittest.TestCase):
    """comp_bench.py's legs per stem and matrix.py's competitor families, fairness and every-leg tables (no CUDA)."""

    def test_comp_legs_apply_at_the_row_precision(self):
        import sys
        from unittest import mock
        sys.path.insert(0, str(HERE))
        with mock.patch.dict(sys.modules, {'torch': mock.MagicMock()}):
            sys.modules.pop('comp_bench', None)
            sys.modules.pop('torch_bench', None)
            import comp_bench as cb
            legs = cb.legs_for
            bf = ['liger-bf16', 'fa2-bf16', 'fa3-bf16', 'flashinfer-bf16', 'trtllm-bf16']
            self.assertEqual(legs('pt_llm_dec_b8_l2048'), ['liger'] + bf)            # + the bf16 reference legs
            self.assertEqual(legs('pt_llm_dec_b8_l2048_bf16w'), bf)                  # the fair legs only
            self.assertEqual(legs('pt_llm_pre_t2048_tf32'), ['liger-tf32', 'triton-tf32'])
            self.assertIn('triton-bf16', legs('pt_llm_blkpre_t2048'))
            self.assertEqual(legs('pt_pj_m1_n4096_k4096_tf32'), ['triton-tf32'])
            self.assertEqual(legs('mm_4096x4096x4096'), ['triton'])
            for s in ('pt_attn_1024_64', 'pt_softmax_rows_4096', 'pt_lsm_1_131072', 'pt_mlp2_2048', 'pt_epi_4096'):
                self.assertEqual(legs(s), ['triton'], s)
            for s in ('pt_ew_4096', 'pt_conv_2048_5', 'pt_decode_1024_256', 'pt_llm_kvx_b1_l2048', 'pt_fanout_2048'):
                self.assertEqual(legs(s), [], s)                                     # no library leg applies
            for leg, (fam, prec) in cb.LEGS.items():                                 # names carry the precision
                self.assertEqual(prec, 'bf16' if leg.endswith('-bf16') else 'tf32' if leg.endswith('-tf32') else 'f32')
            sys.modules.pop('comp_bench', None)
            sys.modules.pop('torch_bench', None)

    def test_matrix_lists_every_leg_by_family_and_keeps_precision(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('matrix', HERE.parent / 'local' / 'matrix.py')
        mx = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mx)
        fam = {'torch-compile-cg': 'cg', 'torch-compile-ro-tf32': 'ro', 'torch-sdpa-compile-cg': 'cg',
               'torch-bf16-cudnn': 'sdpa', 'torch-sdpa-efficient': 'sdpa', 'torch-bf16-compile': 'compile',
               'torch-bf16': 'eager', 'fa2-bf16': 'fa', 'fa3-bf16': 'fa', 'flashinfer-bf16': 'flashinfer',
               'triton-tf32': 'triton', 'liger': 'liger', 'chain-fused': 'cuda', 'chain-lt': 'vendor',
               'lt-tf32-fresh': 'vendor', 'cutlass-tuned': 'vendor'}
        for leg, f in fam.items():
            self.assertEqual(mx.cfam(leg), f, leg)
        self.assertEqual([mx.leg_prec(x) for x in ('fa2-bf16', 'torch-bf16-compile-cg', 'triton-tf32', 'liger')],
                         ['bf16', 'bf16', 'tf32', 'f32'])
        self.assertFalse(mx.is_new('torch-compile') or mx.is_new('cublas-fresh') or mx.is_new('torch-bf16'))
        self.assertTrue(mx.is_new('torch-compile-ro') and mx.is_new('triton') and mx.is_new('torch-bf16-cudnn'))
        res = self.root_file('a.txt', [
            'probe prog=pt_attn_1024_64 face=_default arm=on rep=1 out=[1 iter ms=0.05]',
            'probe prog=pt_attn_1024_64 face=_default arm=null rep=1 out=[1 iter ms=0.05]',
            'torch prog=pt_attn_1024_64 mode=compile n=3 min_ms=0.04 med_ms=0.04 max_ms=0.05 within=yes maxdev=0',
            'torch prog=pt_attn_1024_64 mode=compile-cg n=3 min_ms=0.02 med_ms=0.02 max_ms=0.03 within=yes maxdev=0 '
            'cg=skipped(1)',
            'torch prog=pt_attn_1024_64 mode=compile-ro n=3 min_ms=0.03 med_ms=0.03 max_ms=0.03 within=yes maxdev=0 '
            'cg=on',
            'comp prog=pt_attn_1024_64 leg=triton family=triton prec=f32 n=3 min_ms=0.06 med_ms=0.06 max_ms=0.06 '
            'within=no maxdev=1 tol=0.001',
            'comp prog=pt_attn_1024_64 leg=fa2-bf16 family=fa2 prec=bf16 n=3 min_ms=0.01 med_ms=0.01 max_ms=0.01 '
            'within=yes maxdev=0 tol=0.05'])
        other = self.root_file('b.txt', [
            'probe prog=pt_attn_1024_64 face=_default arm=on rep=1 out=[1 iter ms=0.05]',
            'torch prog=pt_attn_1024_64 mode=compile n=3 min_ms=0.04 med_ms=0.04 max_ms=0.05 within=yes maxdev=0',
            'comp family=triton unavailable reason="x"'])
        out = Path(self.tmp.name)
        r = subprocess.run(['python3', str(HERE.parent / 'local' / 'matrix.py'), '--out', str(out), f'p1={res}@aaaaaaaa',
                            f'p2={other}@bbbbbbbb'], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        rows = {x['part']: x for x in json.loads((out / 'matrix.json').read_text())['rows']}
        p1 = rows['p1']
        # the CUDA-graph leg Inductor skipped and the gate failure never count; the bf16 leg is not an f32 row's best
        self.assertEqual((p1['best_leg'], p1['best_ms']), ('torch-compile-ro', 0.03))
        self.assertEqual((p1['base_best_leg'], p1['base_ratio']), ('torch-compile', 1.25))
        self.assertEqual(p1['every_leg']['torch-compile-cg'][1], 'cg-skipped')
        self.assertEqual(p1['every_leg']['triton'][1], 'gate')
        self.assertNotIn('fa2-bf16', p1['every_leg'])
        md = (out / 'matrix.md').read_text()
        self.assertIn('**torch-compile-ro 0.03 (1.67)**', md)
        line = next(x for x in md.splitlines() if x.startswith('| attn_1024_64 | p2 |'))
        self.assertIn('not run', line)                                   # p2: the CUDA-graph families
        self.assertIn('unavailable', line)                               # p2: Triton reported missing

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def root_file(self, name, lines):
        p = Path(self.tmp.name) / name
        p.write_text('\n'.join(lines) + '\n')
        return p

    def test_confirm_pick_keeps_a_program_win_and_reverts_a_loss(self):
        # sweep-runtime-2: a fake program whose `iter ms` reads the wisdom's last line for each key — site a's pick
        # makes the program faster (keep), site b's slower (revert: candidate 0 appended, the last line wins).
        root = Path(self.tmp.name)
        wdir = root / 'wisdom'
        wdir.mkdir()
        w = wdir / 'wisdom-dev-sm120.tsv'
        w.write_text('00000000000000aa\t01\t4\t2\t5\t1\n00000000000000bb\t02\t4\t3\t5\t1\n')
        exe = root / 'prog'
        exe.write_text('#!/usr/bin/env python3\nimport os, pathlib\n'
                       'last = {}\n'
                       'for f in pathlib.Path(os.environ["MAPAL_WISDOM"]).glob("wisdom-*.tsv"):\n'
                       '    for l in f.read_text().splitlines():\n'
                       '        x = l.split("\\t"); last[x[0]] = int(x[3])\n'
                       'ms = 10.0 - (1.0 if last["00000000000000aa"] else 0) + (1.0 if last["00000000000000bb"] else 0)\n'
                       'print(f"out=[1 2 iter ms={ms} ]")\n')
        exe.chmod(0o755)
        out = subprocess.run(['python3', str(HERE / 'confirm_pick.py'), str(exe), str(wdir),
                              '00000000000000aa:2:trial', '00000000000000bb:3:trial', '00000000000000cc:1:wisdom'],
                             capture_output=True, text=True, check=True).stdout.splitlines()
        self.assertEqual(len(out), 2, out)
        self.assertTrue(out[0].startswith('confirm key=00000000000000aa pick=2') and out[0].endswith('keep'), out)
        self.assertTrue(out[1].startswith('confirm key=00000000000000bb pick=3') and out[1].endswith('revert'), out)
        lines = w.read_text().splitlines()
        self.assertEqual(lines[-1].split('\t')[:4], ['00000000000000bb', '02', '4', '0'])
        self.assertEqual(sum(l.startswith('00000000000000aa') for l in lines), 1)


class BoundaryTests(unittest.TestCase):
    """gaps-disasm H: every competitor leg the matrix reads is timed at the program's boundary, and every reader takes
    that time, never the event bracket printed beside it; a rung A/B reads its OFF arm at the ON text's own picks."""

    # One line of each harness the matrix ingests: the boundary median 2.5, the event bracket 9.75 beside it.
    LINES = (
        "pass=1 cell=mm_16x16x16 time leg=cublas M=16 N=16 K=16 n=30 min_ms=2 med_ms=2.5 max_ms=3 dev_med_ms=9.75",
        "time leg=cublas M=16 N=16 K=16 n=30 min_ms=2 med_ms=2.5 max_ms=3 dev_med_ms=9.75",
        "gemm prog=pt_pj_m1_n64_k64 leg=lt algo=0/8 min_ms=2 med_ms=2.5 max_ms=3 within=yes maxdev=1e-07 "
        "values=1,2 dev_med_ms=9.75",
        "chain probe=attn_1024 n=64 leg=fused launches=3 reps=30 min_ms=2 med_ms=2.5 max_ms=3 dev_med_ms=9.75",
        "torch prog=pt_attn_1024_64 mode=compile n=30 min_ms=2 med_ms=2.5 max_ms=3 within=yes maxdev=1e-07 "
        "compile_s=0.7 peak_mb=72.0 values=1,2 dev_med_ms=9.75",
        "comp prog=pt_attn_1024_64 leg=triton family=triton prec=f32 n=30 min_ms=2 med_ms=2.5 max_ms=3 within=yes "
        "maxdev=1e-07 values=1,2 dev_med_ms=9.75",
    )

    def test_harnesses_time_every_matrix_leg_at_the_program_boundary(self):
        # matrix.py's TORCH / COMP / GEMM / CHAIN / CELL lines come from these; each times a repetition at the
        # program's boundary unless BENCH_BOUNDARY=event (a measurement flag) and prints the bracket beside.
        for src, line in (("torch_bench.py", "torch prog="), ("gemm_lib.cu", "gemm prog="),
                          ("chain_bench.cu", "chain probe=%s n=%d leg="), ("tile_bench.cu", "time leg=")):
            text = (HERE / src).read_text()
            self.assertIn("BENCH_BOUNDARY", text, src)
            fmt = [l for l in text.splitlines() if line in l and "med_ms=" in l]
            self.assertTrue(fmt and all("dev_med_ms=" in l or "dev_med" in text for l in fmt), src)
        chain = (HERE / "chain_bench.cu").read_text()
        timing = chain[chain.index("flush_bytes = 256u << 20"):chain.index("chain_samples probe=")]
        self.assertIn("steady_clock", timing)  # the host clock around the leg's calls ...
        self.assertIn("cudaDeviceSynchronize", timing)  # ... from an idle device to its synchronize
        self.assertIn("dev_med_ms=", timing)

    def test_every_reader_takes_the_boundary_time(self):
        import ast
        readers = sorted(HERE.glob("*.py")) + sorted((HERE.parent / "local").glob("*.py"))
        checked = 0
        for f in readers:
            if f.name == Path(__file__).name:
                continue
            for node in ast.walk(ast.parse(f.read_text(), str(f))):
                if not (isinstance(node, ast.Constant) and isinstance(node.value, str) and "med_ms=" in node.value):
                    continue
                try:
                    rx = re.compile(node.value)
                except re.error:
                    continue
                for line in self.LINES:
                    m = rx.search(line)
                    if m:
                        checked += 1
                        self.assertNotIn("9.75", m.groups(), f"{f.name}:{node.lineno} reads the event bracket")
        self.assertGreater(checked, 20)
        import importlib.util
        spec = importlib.util.spec_from_file_location('matrix', HERE.parent / 'local' / 'matrix.py')
        mx = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mx)
        with tempfile.TemporaryDirectory() as d:
            res = Path(d) / "results.txt"
            res.write_text("\n".join(self.LINES) + "\n")
            _, legs, cells, _ = mx.parse(str(res))
        self.assertEqual(cells["mm_16x16x16"]["cublas"], [2.5])
        self.assertEqual(legs["pt_attn_1024_64"]["chain-fused"][0], 2.5)
        self.assertEqual(legs["pt_attn_1024_64"]["torch-compile"][0], 2.5)
        self.assertEqual(legs["pt_attn_1024_64"]["triton"][0], 2.5)
        self.assertEqual(legs["pt_pj_m1_n64_k64"]["lt"][0], 2.5)

    def test_rung_ab_reads_the_off_arm_at_its_own_picks(self):
        # A tuned box: `on` launches the wisdom's candidate 3, the OFF text the rules' candidate 0 (its keys are
        # not in the wisdom); `rules` is the ON text at candidate 0. The rung is null at one pick (10.05 vs 10.0),
        # a 12 % "win" against `on` (the pick's, not the rung's).
        rows = []
        for r in (1, 2, 3):
            for arm, ms, sites in (("on", 9.0, "600b5866ec11871a:3:wisdom"), ("rules", 10.0, "0000000000000000:0:forced"),
                                   ("null", 9.1, "600b5866ec11871a:3:wisdom"), ("off-x", 10.05, "53f1e1c0d837131c:0:rules")):
                sp = f" sites=[{sites} ]" if r == 1 else ""
                rows.append(f"probe prog=pt_a_64 face=_default arm={arm} rep={r} wall_us=1 out=[1 iter ms={ms} ] "
                            f"rc=0 trace_rc=0{sp}\n")
                if r == 1:
                    rows.append(f"probe_kernels prog=pt_a_64 face=_default arm={arm} rep=1 k ms={ms / 2}\n")
        res = self.root / "results.txt"
        res.write_text("".join(rows))
        out = subprocess.run(["python3", HERE / "probe_rungs.py", res], capture_output=True, text=True,
                             check=True).stdout
        row = next(l for l in out.splitlines() if l.startswith("| x |"))
        self.assertIn("| OFF | OFF |", row)  # 1.005: no effect at one pick
        self.assertIn("1.005", row)
        self.assertIn("(vs rules)", row)
        ab = subprocess.run(["python3", HERE / "arm_ratio.py", "--pair", "on", "off-x", "--pair", "rules", "off-x", res],
                            capture_output=True, text=True).stdout
        self.assertIn("pick-confounded ([3]/[0])", next(l for l in ab.splitlines() if l.startswith("| on | off-x |")))
        self.assertIn("| none |", next(l for l in ab.splitlines() if l.startswith("| rules | off-x |")))
        # No ON-text arm at the OFF arm's picks: the pair judges nothing.
        res.write_text("".join(l for l in rows if "arm=rules" not in l))
        out = subprocess.run(["python3", HERE / "probe_rungs.py", res], capture_output=True, text=True,
                             check=True).stdout
        self.assertIn("a_64: pick≠", next(l for l in out.splitlines() if l.startswith("| x |")))

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)


if __name__ == "__main__":
    unittest.main()
