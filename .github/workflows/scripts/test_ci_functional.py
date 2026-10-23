# SPDX-License-Identifier: BSD-3-Clause

import ast
from collections import Counter
from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from functional_cases import Case, STATE_DTYPE_CASES, StateDtypeSubset, cases, v_last_lists
import run_functional as runner


class FunctionalCITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_inventory_matches_checkin_counts_and_assets(self):
        selected = cases()
        self.assertEqual(Counter(case.phase for case in selected), {"decode": 33, "prefill": 17})
        self.assertEqual(len({case.name for case in selected}), 50)
        runner.validate_inventory(selected)
        self.assertEqual(sum("--kernel=cuda" in case.argv for case in selected), 4)
        self.assertFalse(any("--max-cases" in arg for case in selected for arg in case.argv))

    @staticmethod
    def v_last_line(batch, seqlen=1, heads=16):
        return (f"[GDN]:[D][{batch}]gdn_attn:seqlen_q:{seqlen},head_dim_qk:128,head_dim_v:128,"
                f"num_heads_qk:{heads},num_heads_v:32,batch_size:{batch},dtype:bf16.")

    @staticmethod
    def line_params(line):
        return dict(re.findall(r"(\w+):([^,]+)", line.split("gdn_attn:", 1)[1].rstrip(".")))

    def test_v_last_selects_two_tp_endpoints_and_five_batches_per_model(self):
        for case in (case for case in cases() if "--kernel=cuda" in case.argv):
            with self.subTest(model=case.name):
                output = self.root / case.name
                output.mkdir()
                source = runner.ROOT / next(arg.split("=", 1)[1] for arg in case.argv
                                            if arg.startswith("--caselist="))
                original = source.read_bytes()
                command, cwd, expected = runner.prepare(case, output)
                self.assertEqual(expected, 10)
                self.assertEqual(cwd, runner.ROOT)
                self.assertIn(f"--caselist={output / 'lists'}", command)
                generated = sorted((output / "lists").iterdir())
                self.assertEqual({int(re.search(r"_tp(\d+)_", path.name)[1]) for path in generated}, {1, 16})
                baseline = self.line_params(source.read_text().splitlines()[1])
                batches = []
                for path in generated:
                    tp = int(re.search(r"tp(\d+)", str(path))[1])
                    params = [self.line_params(line) for line in path.read_text().splitlines()]
                    batch = [int(row["batch_size"]) for row in params]
                    self.assertEqual(len(set(batch)), 5)
                    self.assertTrue(all(0 < value < 256 for value in batch))
                    self.assertEqual(batch[0], 1)
                    batches.append(batch)
                    for row in params:
                        self.assertEqual(int(row["seqlen_q"]), 1)
                        for key in ("num_heads_qk", "num_heads_v"):
                            self.assertEqual(int(row[key]) * tp, int(baseline[key]))
                        for key in ("head_dim_qk", "head_dim_v", "dtype"):
                            self.assertEqual(row[key], baseline[key])
                self.assertEqual(batches[0], batches[1])
                self.assertEqual(source.read_bytes(), original)

    def test_v_last_deduplicates_batches_and_ignores_unused_seqlens(self):
        text = "\n".join(self.v_last_line(batch, seqlen) for batch, seqlen in
                         ((1, 1), (1, 1), (4, 1), (16, 3), (64, 4), (255, 5), (256, 1)))
        selected = v_last_lists(text)
        self.assertEqual(set(selected), {1, 16})
        for lines in selected.values():
            self.assertEqual([int(self.line_params(line)["batch_size"]) for line in lines], [1, 4])

    def test_v_last_rejects_empty_or_invalid_inventory(self):
        for text in ("", "# empty", self.v_last_line(256), self.v_last_line(1, 3),
                     self.v_last_line(1, heads=24), self.v_last_line(1, heads=0)):
            with self.subTest(text=text), self.assertRaises(ValueError):
                v_last_lists(text)

    def test_v_last_incomplete_tp_inventory_cannot_pass(self):
        case = next(case for case in cases() if "--kernel=cuda" in case.argv)

        def execute(command, cwd, env, log, timeout):
            log.write_text("*--- start test case 1/1 ---*\n--- case pass!!! ---\nTotal cases: 1\n")
            return 0

        with patch.object(runner, "execute", execute), redirect_stdout(io.StringIO()):
            result = runner.run_case(case, self.root, 5)
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["children"][-1]["name"], "inventory")

    def test_v_last_complete_two_file_summary_passes(self):
        case = next(case for case in cases() if "--kernel=cuda" in case.argv)

        def execute(command, cwd, env, log, timeout):
            lines = [f"*--- start test case {i}/5 ---*\n--- case pass!!! ---"
                     for tp in (1, 16) for i in range(1, 6)]
            log.write_text("\n".join(lines) + "\nTotal cases: 10\n")
            return 0

        with patch.object(runner, "execute", execute), redirect_stdout(io.StringIO()):
            result = runner.run_case(case, self.root, 5)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(len(result["children"]), 10)

    def test_list_reports_actual_reduced_scope(self):
        with patch.object(sys, "argv", ["runner", "--list"]), redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(runner.main(), 0)
        text = stream.getvalue()
        self.assertEqual(text.count("v-last: TP=(1, 16)"), 4)
        self.assertEqual(text.count("cases=10"), 4)
        self.assertIn("state-dtype: 38 precision cases", text)
        self.assertIn("Total groups: 50", text)

    def test_other_kernel_and_prefill_commands_are_unchanged(self):
        for case in cases():
            if "--kernel=cuda" in case.argv or case.report in ("pytest", "jsonl"):
                continue
            command, cwd, expected = runner.prepare(case, self.root)
            self.assertEqual(command[2:], list(case.argv))
            self.assertEqual(cwd, runner.ROOT)
            self.assertIsNone(expected)

    def test_state_dtype_uses_ci_selection_from_tests_directory(self):
        case = next(case for case in cases() if case.name == "k-last-state-dtype")
        command, cwd, expected = runner.prepare(case, self.root)
        self.assertEqual(cwd, runner.ROOT / "tests")
        self.assertEqual(command[2], str(Path(runner.__file__).resolve()))
        self.assertEqual(command[3:], ["--state-dtype-junit", str(self.root / "pytest.xml")])
        self.assertIsNone(expected)

    def test_state_dtype_ci_entry_runs_original_pytest_with_selection(self):
        calls = []
        fake_pytest = SimpleNamespace(main=lambda argv, plugins: calls.append((argv, plugins)) or 0)
        with patch.dict(sys.modules, {"pytest": fake_pytest}):
            with patch.object(sys, "argv", ["runner", "--state-dtype-junit", "result.xml"]):
                self.assertEqual(runner.main(), 0)
        argv, plugins = calls[0]
        self.assertEqual(argv, ["test_k_last_state_dtype.py", "-v", "-o",
                                "console_output_style=classic", "--junitxml=result.xml"])
        self.assertIsInstance(plugins[0], StateDtypeSubset)

    def test_state_dtype_empty_collection_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "Expected 38 state-dtype cases"):
            StateDtypeSubset().pytest_collection_modifyitems(None, [])

    @unittest.skipUnless(importlib.util.find_spec("pytest"), "requires pytest for collection")
    def test_state_dtype_subset_with_actual_product_parametrization(self):
        import pytest

        source = ast.parse((runner.ROOT / "tests/test_k_last_state_dtype.py").read_text())
        module = ast.parse("import pytest\nfrom types import SimpleNamespace\n"
                           "torch = SimpleNamespace(float32='torch.float32', "
                           "float16='torch.float16', bfloat16='torch.bfloat16')\n")
        for node in source.body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                node.body = [ast.Pass()]
                module.body.append(node)
        path = self.root / "test_state_dtype_collection.py"
        path.write_text(ast.unparse(ast.fix_missing_locations(module)))

        class Capture(StateDtypeSubset):
            def pytest_collection_modifyitems(self, config, items):
                self.total = len(items)
                super().pytest_collection_modifyitems(config, items)
                self.selected = [(item.originalname, item.callspec.params) for item in items]

        selection = Capture()
        with redirect_stdout(io.StringIO()):
            rc = pytest.main([str(path), "--collect-only", "-q", "--noconftest"], plugins=[selection])
        self.assertEqual(rc, 0)
        self.assertEqual(selection.total, 745)
        self.assertEqual(len(selection.selected), STATE_DTYPE_CASES)
        self.assertEqual(Counter(name for name, _ in selection.selected),
                         {"test_k_last_state_dtype": 26, "test_nonpow2_entry_modes": 8,
                          "test_k_last_kda_lower_bound": 2, "test_k_last_vllm_slot_table": 2})
        core = [params for name, params in selection.selected
                if name == "test_k_last_state_dtype" and params["mode"] in ("decode", "verify")]
        self.assertEqual(len(core), 16)
        self.assertEqual({(p["state_dtype"], p["activation_dtype"], p["is_kda"], p["mode"]) for p in core},
                         {(state, activation, kda, mode) for state in ("torch.float32", "torch.bfloat16")
                          for activation in ("torch.float16", "torch.bfloat16")
                          for kda in (False, True) for mode in ("decode", "verify")})
        self.assertEqual({p["mode"] for name, p in selection.selected if name == "test_k_last_state_dtype"},
                         {"decode", "verify", "writeback", "tree", "ragged", "graph", "tunable"})

    def make_wheel(self, directories=None, source=b"# source\n", manifest=True):
        package = self.root / "pla"
        package.mkdir(exist_ok=True)
        (package / "__init__.py").write_bytes(b"# source\n")
        wheel = self.root / "pla.whl"
        with ZipFile(wheel, "w") as archive:
            for directory in runner.EXTENSION_DIRS if directories is None else directories:
                archive.writestr(f"{directory}/_C.test.so", b"current-binary")
            if manifest:
                archive.writestr(runner.BUILD_MANIFEST, b"{}")
            archive.writestr("pla/__init__.py", source)
        return wheel

    def test_stage_same_wheel_artifacts_without_modifying_source(self):
        wheel = self.make_wheel()
        with redirect_stdout(io.StringIO()):
            runner.stage_wheel(wheel, self.root)
        self.assertEqual(len(list(self.root.rglob("*.so"))), 6)
        self.assertEqual((self.root / runner.BUILD_MANIFEST).read_bytes(), b"{}")
        self.assertEqual((self.root / "pla/__init__.py").read_bytes(), b"# source\n")

    def test_stage_rejects_incomplete_artifacts(self):
        for kwargs in ({"directories": ["pla/decode"]}, {"manifest": False}):
            with self.assertRaises(ValueError):
                runner.stage_wheel(self.make_wheel(**kwargs), self.root)
            self.assertFalse(list(self.root.rglob("*.so")))

    def test_stage_rejects_mismatched_source_before_writing(self):
        with self.assertRaisesRegex(ValueError, "does not match the current source"):
            runner.stage_wheel(self.make_wheel(source=b"# other\n"), self.root)
        self.assertFalse(list(self.root.rglob("*.so")))

    def test_both_marker_formats(self):
        for start in ("1", "1/1", "1 (shape)", "1/1 (shape)"):
            rows = runner.parse_markers(
                f"*--- start test case {start} ---*\n--- case pass!!! ---\nTotal cases: 1\n")
            self.assertEqual([row["status"] for row in rows], ["PASS"])

    def test_missing_marker_summary_and_zero_tests_fail(self):
        for text in ("", "Total cases: 0\n", "*--- start test case 1/1 ---*\n",
                     "*--- start test case 1/2 ---*\n--- case pass!!! ---\nTotal cases: 1\n",
                     "*--- start test case 1 ---*\n--- case pass!!! ---\n"):
            self.assertIn("FAIL", [row["status"] for row in runner.parse_markers(text)])

    def test_failed_subcase_cannot_pass(self):
        rows = runner.parse_markers(
            "*--- start test case 1 ---*\n--- case fail!!! ---\nTotal cases: 1\n")
        self.assertEqual(rows[0]["status"], "FAIL")

    def test_pytest_skip_and_failure(self):
        path = self.root / "pytest.xml"
        path.write_text('<testsuites><testsuite><testcase name="a"/>'
                        '<testcase name="b"><skipped message="two devices required"/></testcase>'
                        '<testcase name="c"><failure message="wrong"/></testcase>'
                        '</testsuite></testsuites>')
        self.assertEqual([row["status"] for row in runner.parse_pytest(path)], ["PASS", "SKIP", "FAIL"])

    def test_jsonl_requires_every_selected_record(self):
        path = self.root / "replay.jsonl"
        path.write_text(json.dumps({"source": "a", "line": 1, "pass": True}) + "\n")
        self.assertEqual(runner.parse_jsonl(path, 1)[0]["status"], "PASS")
        self.assertEqual(runner.parse_jsonl(path, 2)[-1]["status"], "FAIL")
        path.write_text("")
        self.assertEqual(runner.parse_jsonl(path, 0)[-1]["status"], "FAIL")

    def test_replay_selects_full_lists(self):
        case = next(case for case in cases() if case.name == "kimi-kda-full")
        command, cwd, expected = runner.prepare(case, self.root)
        self.assertEqual(expected, 2688)
        self.assertEqual(len(list((self.root / "lists").iterdir())), 2)
        self.assertIn("--reference=serial", command)
        self.assertNotIn("--index-file", command)
        self.assertEqual(cwd, runner.ROOT)

    def test_pytest_does_not_run_from_source_package_root(self):
        case = next(case for case in cases() if case.report == "pytest")
        command, cwd, _ = runner.prepare(case, self.root)
        self.assertEqual(cwd, runner.ROOT / "tests")
        self.assertEqual(command[1:3], ["-m", "pytest"])

    def test_failure_does_not_prevent_next_group(self):
        with redirect_stdout(io.StringIO()):
            failed = runner.run_case(Case("bad", "decode", ("-c", "raise SystemExit(3)")), self.root, 5)
            passed = runner.run_case(Case("good", "decode", ("-c", "print('ok')")), self.root, 5)
        self.assertEqual(failed["status"], "FAIL")
        self.assertEqual(passed["status"], "PASS")

    def test_timeout_and_nonzero_exit_are_failures(self):
        with redirect_stdout(io.StringIO()):
            result = runner.run_case(Case("slow", "decode", ("-c", "import time; time.sleep(5)")),
                                     self.root, 0.1)
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["returncode"], 124)

    def test_legacy_environment_does_not_leak_to_other_groups(self):
        with patch.dict(os.environ, {"PLA_FLASHKDA_TC01_IMPL": "wrong", "PYTHONPATH": "wrong"}):
            with redirect_stdout(io.StringIO()):
                case = Case("env", "decode", ("-c", "import os; assert 'PYTHONPATH' not in os.environ; "
                            "assert 'PLA_FLASHKDA_TC01_IMPL' not in os.environ"))
                result = runner.run_case(case, self.root, 5)
                legacy = Case("legacy", "prefill", ("-c", "import os; "
                              "assert os.environ['PLA_FLASHKDA_TC01_IMPL'] == 'legacy'"), legacy=True)
                other = runner.run_case(legacy, self.root, 5)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(other["status"], "PASS")

    def make_reports(self, boards=runner.BOARDS, phases=runner.PHASES):
        for board in boards:
            for phase in phases:
                output = self.root / board / phase
                output.mkdir(parents=True)
                records = [{"name": "environment", "status": "PASS",
                            "children": [runner.item("environment", "PASS")]}]
                records += [{"name": case.name, "status": "PASS",
                             "children": [runner.item(case.name, "PASS")]} for case in cases()
                            if case.phase == phase]
                runner.save_report({"board": board, "phase": phase, "results": records}, output)

    def test_aggregate_complete_matrix(self):
        self.make_reports()
        with patch.dict(os.environ, {"GITHUB_STEP_SUMMARY": str(self.root / "github.md")}):
            with redirect_stdout(io.StringIO()):
                self.assertEqual(runner.summarize(self.root), 0)
        self.assertIn("OAM-810E", (self.root / "github.md").read_text())
        self.assertIn("ZW-M890P", (self.root / "github.md").read_text())

    def test_board_summary_does_not_require_other_board(self):
        board = runner.BOARDS[0]
        self.make_reports(boards=(board,))
        with patch.dict(os.environ, {}, clear=True), redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(runner.summarize(self.root, (board,)), 0)
        self.assertNotIn(runner.BOARDS[1], stream.getvalue())
        with patch.dict(os.environ, {}, clear=True), redirect_stdout(io.StringIO()):
            self.assertEqual(runner.summarize(self.root), 1)

    def test_board_summary_requires_both_phases(self):
        board = runner.BOARDS[0]
        self.make_reports(boards=(board,), phases=("decode",))
        with patch.dict(os.environ, {}, clear=True), redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(runner.summarize(self.root, (board,)), 1)
        self.assertIn(f"{board} / prefill", stream.getvalue())

    def test_board_summary_cli_selects_requested_board(self):
        board = runner.BOARDS[1]
        self.make_reports(boards=(board,))
        args = ["runner", "--summary-root", str(self.root), "--board", board]
        with patch.object(sys, "argv", args), patch.dict(os.environ, {}, clear=True):
            with redirect_stdout(io.StringIO()):
                self.assertEqual(runner.main(), 0)

    def test_phase_outcomes_cannot_be_hidden_by_passed_reports(self):
        self.make_reports()
        for name in ("PLA_DECODE_RESULT", "PLA_PREFILL_RESULT", "PLA_BUILD_RESULT"):
            for outcome in ("failure", "skipped", "cancelled"):
                with self.subTest(name=name, outcome=outcome):
                    with patch.dict(os.environ, {name: outcome}, clear=True):
                        with redirect_stdout(io.StringIO()):
                            self.assertEqual(runner.summarize(self.root, runner.BOARDS[:1]), 1)
        successful = {name: "success" for name in
                      ("PLA_BUILD_RESULT", "PLA_DECODE_RESULT", "PLA_PREFILL_RESULT")}
        with patch.dict(os.environ, successful, clear=True), redirect_stdout(io.StringIO()):
            self.assertEqual(runner.summarize(self.root, runner.BOARDS[:1]), 0)

    def test_workflow_has_independent_board_pipelines(self):
        workflow = (runner.ROOT / ".github/workflows/pla-basic-test-ppu.yml").read_text()
        header = workflow.split("  build-test:\n", 1)[1].split("    steps:\n", 1)[0]
        self.assertIn("name: Build & Test (${{ matrix.board }})", header)
        self.assertIn("board: [OAM-810E, ZW-M890P]", header)
        self.assertIn("fail-fast: false", header)
        self.assertIn("runs-on: k8s-runner-group-cpu-thead", header)
        self.assertNotIn("needs:", header)
        self.assertNotIn("phase:", header)
        self.assertIn("image: ghcr.io/actions/actions-runner:latest", header)
        self.assertNotIn("sdk2.2.0", header)
        self.assertIn("timeout-minutes: 900", header)
        self.assertEqual(workflow.count("uses: t-head/ppu-scheduler-action@main"), 1)
        worker = workflow.split("          command: |\n", 1)[1].split("      - name:", 1)[0]
        self.assertEqual(worker.count("python setup.py "), 1)
        self.assertIn("python setup.py bdist_wheel", worker)
        self.assertNotIn("python setup.py build", worker)
        self.assertNotIn("--build-base", worker)
        self.assertNotIn("--skip-build", worker)
        self.assertIn("--kill-after=30s 180m", worker)
        self.assertIn("for phase in decode prefill; do", worker)
        self.assertIn("--budget-seconds 18000 || result=1", worker)
        self.assertIn('exit "$result"', worker)
        self.assertIn("timeout_minutes: 840", workflow)
        self.assertIn("PLA_TEST_RESULT: ${{ steps.ppu-test.outcome }}", workflow)
        self.assertIn("name: pla-results-${{ matrix.board }}", workflow)
        self.assertIn("repository: t-head/actlize", workflow)
        self.assertIn("ref: 7c08602a4f4a3ba1ebee7ce16e58eb496e183115", workflow)
        self.assertIn("path: 3rdparty/actlize", workflow)
        self.assertNotIn("scripts/ci/", workflow)
        self.assertNotIn("tests/test_ci_functional.py", workflow)
        self.assertNotIn("  summary:", workflow)
        self.assertNotIn("continue-on-error:", workflow)

    def test_entry_requires_serial_review_gates(self):
        workflow = (runner.ROOT / ".github/workflows/ci.yml").read_text()
        self.assertIn("branches: [master]", workflow)
        self.assertIn("workflow_dispatch:", workflow)
        self.assertIn("github.event.pull_request.number || github.ref", workflow)
        dependencies = {"ai-code-review": "precheck", "human-review": "ai-code-review",
                        "ci": "human-review"}
        for name, dependency in dependencies.items():
            job = workflow.split(f"  {name}:\n", 1)[1].split("\n\n", 1)[0]
            self.assertEqual([line.strip() for line in job.splitlines() if "needs:" in line],
                             [f"needs: {dependency}"])
            self.assertNotIn("if:", job)
            self.assertNotIn("continue-on-error:", job)
        self.assertNotIn("pull_request_review:", workflow)
        self.assertNotIn("scripts/ci/", workflow)
        self.assertNotIn("tests/test_ci_functional.py", workflow)

    def test_aggregate_missing_matrix_is_failure(self):
        with patch.dict(os.environ, {}, clear=True), redirect_stdout(io.StringIO()):
            self.assertEqual(runner.summarize(self.root), 1)

    def test_aggregate_incomplete_groups_is_failure(self):
        self.make_reports()
        path = self.root / runner.BOARDS[0] / "decode" / "results.json"
        report = json.loads(path.read_text())
        report["results"].pop()
        path.write_text(json.dumps(report))
        with patch.dict(os.environ, {}, clear=True), redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(runner.summarize(self.root), 1)
        self.assertIn("Incomplete test group inventory", stream.getvalue())
        self.assertTrue(stream.getvalue().isascii())

    def test_corrupt_report_still_produces_summary(self):
        (self.root / "results.json").write_text("incomplete json")
        with patch.dict(os.environ, {}, clear=True), redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(runner.summarize(self.root), 1)
        self.assertIn("ZW-M890P", stream.getvalue())
        self.assertIn("Cannot parse report", stream.getvalue())
        self.assertTrue(stream.getvalue().isascii())

    def test_summary_output_is_english_for_complete_and_missing_reports(self):
        self.make_reports()
        for complete in (True, False):
            for boards in (runner.BOARDS, runner.BOARDS[:1]):
                with self.subTest(complete=complete, boards=boards):
                    root = self.root if complete else self.root / "missing"
                    summary = self.root / f"step-{complete}-{len(boards)}.md"
                    with patch.dict(os.environ, {"GITHUB_STEP_SUMMARY": str(summary)}, clear=True):
                        with redirect_stdout(io.StringIO()) as stream:
                            self.assertEqual(runner.summarize(root, boards), int(not complete))
                    text = summary.read_text()
                    self.assertTrue(text.isascii())
                    self.assertIn("Functional Test Summary", text)
                    self.assertEqual(stream.getvalue(), text + "\n")
                    if complete:
                        self.assertIn("| Test group | Status | Passed | Failed | Skipped | Seconds |", text)
                        self.assertIn("PASS excludes SKIP; NOT_RUN means not executed.", text)
                    else:
                        self.assertIn("No report received", text)

    def test_report_artifacts_use_english_and_preserve_status_counts(self):
        records = [{"name": status.lower(), "status": status,
                    "children": [runner.item(status.lower(), status)]}
                   for status in ("PASS", "FAIL", "SKIP")]
        records.append(runner.item("pending", "NOT_RUN"))
        report = {"board": "OAM-810E", "phase": "decode", "results": records}
        runner.save_report(report, self.root)
        for name in ("summary.md", "results.json", "junit.xml"):
            self.assertTrue((self.root / name).read_text().isascii())
        text = (self.root / "summary.md").read_text()
        for status in ("PASS", "FAIL", "SKIP", "NOT_RUN"):
            self.assertIn(f"'{status}': 1", text)
        suite = ET.parse(self.root / "junit.xml").getroot()
        self.assertEqual((suite.get("tests"), suite.get("failures"), suite.get("skipped")),
                         ("4", "2", "1"))
        self.assertEqual([node.get("message") for node in suite.iter("failure")],
                         ["See group log", "Not executed"])

    def test_scheduler_failure_cannot_be_hidden_by_passed_reports(self):
        self.make_reports()
        with patch.dict(os.environ, {"PLA_TEST_RESULT": "failure"}, clear=True):
            with redirect_stdout(io.StringIO()):
                self.assertEqual(runner.summarize(self.root), 1)

    def test_pending_groups_are_not_reported_as_passed(self):
        report = {"board": "OAM-810E", "phase": "decode",
                  "results": [runner.item("not-started", "NOT_RUN")]}
        runner.save_report(report, self.root)
        self.assertEqual(ET.parse(self.root / "junit.xml").getroot().get("failures"), "1")
        self.assertIn("NOT_RUN", (self.root / "summary.md").read_text())


if __name__ == "__main__":
    unittest.main(verbosity=2)
