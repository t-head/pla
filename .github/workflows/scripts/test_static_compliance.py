#!/usr/bin/env python3
"""CPU-only scanner and workflow regressions; no product code is executed."""

import ast
from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = Path(__file__).with_name("run_static_compliance.py")
spec = importlib.util.spec_from_file_location("run_static_compliance", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
scanner = runner.scanner
WORKFLOW = ROOT / ".github/workflows/static-compliance-scan.yml"


class StaticComplianceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.previous = Path.cwd()
        os.chdir(self.root)
        self.git("init", "-q")
        self.env = dict(os.environ, GIT_AUTHOR_NAME="CI", GIT_COMMITTER_NAME="CI",
                        GIT_AUTHOR_EMAIL="ci@example.org", GIT_COMMITTER_EMAIL="ci@example.org")
        self.write("source.py", "value = 1\n")
        self.base = self.commit("test: initial fixture")
        scanner.repo_root.cache_clear()
        scanner.head_exists.cache_clear()

    def tearDown(self):
        os.chdir(self.previous)
        scanner.repo_root.cache_clear()
        scanner.head_exists.cache_clear()
        self.temp.cleanup()

    def git(self, *args):
        return subprocess.run(["git", *args], cwd=self.root, check=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, env=getattr(self, "env", None)).stdout.strip()

    def write(self, path, text):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")

    def commit(self, message):
        self.git("add", ".")
        self.git("commit", "-q", "-m", message)
        return self.git("rev-parse", "HEAD")

    def scan(self, head, base=None):
        with redirect_stdout(io.StringIO()):
            return runner.scan(base or self.base, head)

    def rules(self, result):
        return {item["rule"] for item in result["findings"]}

    def test_clean_increment_passes(self):
        self.write("source.py", "value = 2\n")
        result = self.scan(self.commit("test: clean change"))
        self.assertEqual(result["status"], "PASS")
        self.assertEqual((result["files"], result["added_lines"], result["commits"]), (1, 1, 1))

    def test_existing_violations_do_not_enter_increment(self):
        self.write("source.py", "# \u4e2d\u6587\nvalue = 1\n")
        base = self.commit("test: historical fixture")
        self.write("source.py", "# \u4e2d\u6587\nvalue = 2\n")
        self.assertEqual(self.scan(self.commit("test: clean addition"), base)["status"], "PASS")

    def test_added_vendor_names_pass_normal(self):
        self.write("source.py", "cu" + "daMalloc();\na" + "compute_task();\n")
        result = self.scan(self.commit("test: vendor fixture"))
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["findings"], [])

    def test_vendor_names_in_commit_messages_pass_normal(self):
        self.write("source.py", "value = 2\n")
        result = self.scan(self.commit("test: cu" + "da fixture\n\na" + "compute task"))
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["commits"], 1)
        self.assertEqual(result["findings"], [])

    def test_normal_retains_all_core_line_rules(self):
        fixtures = {
            "next-gen-chip-term": "ppu" + "0017",
            "restricted-eula-component": "cute" + "DSL",
            "tensor-core-term": "tensor" + " core",
            "internal-host-name": "git" + "lab",
            "chinese-text": "\u4e2d\u6587",
            "illegal-char": "\u00a0",
        }
        for rule, text in fixtures.items():
            with self.subTest(rule=rule):
                self.write("source.py", f"# {text}\n")
                result = self.scan(self.commit("test: core rule fixture"))
                self.assertEqual(result["status"], "FAIL")
                self.assertIn(rule, self.rules(result))

    def test_chinese_and_internal_terms_are_core_findings(self):
        self.write("source.py", "# \u4e2d\u6587\n# git" + "lab\n")
        self.assertTrue({"chinese-text", "internal-host-name"} <=
                        self.rules(self.scan(self.commit("test: content fixture"))))

    def test_only_changed_commit_messages_are_scanned(self):
        self.write("source.py", "value = 2\n")
        result = self.scan(self.commit("test: fixture\n\n\u4e2d\u6587"))
        self.assertIn("chinese-text", self.rules(result))
        self.assertTrue(any(item["path"].startswith("commit ") for item in result["findings"]))

    def test_merge_base_excludes_target_branch_changes(self):
        self.git("branch", "target")
        self.write("source.py", "value = 2\n")
        head = self.commit("test: branch fixture")
        self.git("checkout", "-q", "target")
        self.write("target.py", "# \u4e2d\u6587\n")
        base = self.commit("test: target fixture")
        self.git("checkout", "-q", "--detach", head)
        result = self.scan(head, base)
        self.assertEqual(result["merge_base"], self.base)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["commits"], 1)

    def test_same_base_head_is_error(self):
        with self.assertRaisesRegex(RuntimeError, "resolves to head"):
            self.scan(self.base)

    def test_invalid_and_missing_sha_are_errors(self):
        for base in ("master", "--all", "0" * 40):
            with self.subTest(base=base), self.assertRaises(RuntimeError):
                self.scan(self.base, base)

    def test_checkout_mismatch_is_error(self):
        self.write("source.py", "value = 2\n")
        self.commit("test: new checkout")
        with self.assertRaisesRegex(RuntimeError, "Checkout does not match"):
            self.scan(self.base)

    def test_unrelated_history_is_error_without_fallback(self):
        self.git("checkout", "-q", "--orphan", "unrelated")
        self.write("source.py", "value = 3\n")
        head = self.commit("test: unrelated fixture")
        with self.assertRaises(RuntimeError):
            self.scan(head)

    def test_deletion_only_change_is_not_false_empty_scan(self):
        self.write("source.py", "")
        result = self.scan(self.commit("test: remove content"))
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["added_lines"], 0)
        self.assertEqual(result["commits"], 1)

    def test_empty_scan_inventory_is_error(self):
        self.write("source.py", "value = 2\n")
        head = self.commit("test: fixture")
        with patch.object(scanner, "scan_diff", return_value=([], 0, 0, 0)):
            with self.assertRaisesRegex(RuntimeError, "Empty scan"):
                self.scan(head)

    def test_exclusions_are_narrow_and_visible(self):
        for path in runner.IGNORED_PATHS[1:]:
            self.write(path, "# \u4e2d\u6587\n")
        self.write(".aoneci/legacy.yml", "# \u4e2d\u6587\n")
        self.write(".github/other.py", "# \u4e2d\u6587\n")
        result = self.scan(self.commit("test: exclusion fixture"))
        paths = {item["path"] for item in result["findings"]}
        self.assertEqual(paths, {".github/other.py"})
        self.assertFalse(scanner.is_ignored(".aoneci-extra/other.py", runner.IGNORED_PATHS))
        self.assertEqual(result["files"], 1)

    def test_changed_vendor_path_passes_normal(self):
        self.write("n" + "v_helper.py", "value = 1\n")
        result = self.scan(self.commit("test: path fixture"))
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["findings"], [])
        self.assertEqual(result["files"], 1)

    def test_inline_asm_uses_whole_file_context(self):
        self.write("source.cpp", 'asm volatile(\n    "ppu.test;"\n);\n')
        base = self.commit("test: assembly fixture")
        self.write("source.cpp", 'asm volatile(\n    "ppu.test; trap;"\n);\n')
        result = self.scan(self.commit("test: assembly change"), base)
        self.assertIn("asm-ppu-prefix", self.rules(result))
        self.assertTrue(any(item["token"] == "trap" for item in result["findings"]))

    def test_assembly_prose_does_not_count_as_executable_asm(self):
        self.write("source.cpp", '// asm("trap;")\nconst char *s = "asm(trap)";\n')
        self.assertEqual(self.scan(self.commit("test: prose fixture"))["status"], "PASS")

    def test_payload_prefixes_and_no_newline_markers_preserve_line_numbers(self):
        diff = "--- a/source.cpp\n+++ b/source.cpp\n@@ -1 +1,2 @@\n-old\n\\ No newline at end of file\n+++counter;\n+value;\n"
        self.assertEqual([(item.line, item.text) for item in scanner.parse_added_lines(diff)],
                         [(1, "++counter;"), (2, "value;")])

    def test_normal_and_strong_rules_remain_distinct(self):
        text = "cu" + "daMalloc();"
        self.assertFalse(scanner.scan_text("source.cpp", 1, text, scanner.NORMAL_MODE))
        self.assertTrue(scanner.scan_text("source.cpp", 1, text, scanner.STRONG_MODE))

    def test_report_escapes_control_characters(self):
        self.assertNotIn("\0", scanner.printable("token\0value"))
        self.assertIn("\\u202e", scanner.printable("token\u202evalue"))

    def test_product_code_is_only_read_not_executed(self):
        marker = self.root / "executed"
        self.write("source.py", f"from pathlib import Path\nPath({str(marker)!r}).touch()\n")
        self.assertEqual(self.scan(self.commit("test: inert source"))["status"], "PASS")
        self.assertFalse(marker.exists())

    def test_reports_preserve_pass_fail_and_error_exit_codes(self):
        for text, expected in (("value = 2\n", 0), ("# \u4e2d\u6587\n", 1)):
            self.write("source.py", text)
            head = self.commit("test: report fixture")
            output = self.root / f"report-{expected}"
            with redirect_stdout(io.StringIO()):
                code = runner.main(["--base", self.base, "--head", head, "--output", str(output)])
            self.assertEqual(code, expected)
            result = json.loads((output / "results.json").read_text())
            self.assertEqual(result["status"], "FAIL" if expected else "PASS")
            self.assertEqual(result["mode"], "normal")
            self.assertEqual(len(result["scanner_sha256"]), 64)
            self.assertIn("Rule mode: normal", (output / "scan.log").read_text())
            summary = (output / "summary.md").read_text()
            self.assertIn("Static Compliance Scan (normal)", summary)
            self.assertIn("Explicit exclusions", summary)
        output = self.root / "report-error"
        with redirect_stdout(io.StringIO()):
            code = runner.main(["--base", "0" * 40, "--head", head, "--output", str(output)])
        self.assertEqual(code, 2)
        result = json.loads((output / "results.json").read_text())
        self.assertEqual(result["status"], "ERROR")
        self.assertEqual(result["mode"], "normal")
        self.assertIn("Static Compliance Scan (normal)", (output / "summary.md").read_text())

    def test_source_text_cannot_inject_step_summary_or_console_commands(self):
        self.write("source.py", "# \u4e2d\u6587\n::error::fixture\n```\n")
        head = self.commit("test: log fixture")
        output = self.root / "report"
        summary = self.root / "step.md"
        with patch.dict(os.environ, {"GITHUB_STEP_SUMMARY": str(summary)}):
            with redirect_stdout(io.StringIO()) as console:
                self.assertEqual(runner.main(["--base", self.base, "--head", head,
                                              "--output", str(output)]), 1)
        self.assertIn("Static Compliance Scan (normal): FAIL", console.getvalue())
        self.assertNotIn("::error::", console.getvalue())
        self.assertNotIn("\u4e2d\u6587", summary.read_text())
        self.assertNotIn("```", summary.read_text())

    def workflow_shell(self, step):
        block = WORKFLOW.read_text().split(f"      - name: {step}\n", 1)[1]
        block = block.split("      - name:", 1)[0]
        return textwrap.dedent(block.split("        run: |\n", 1)[1])

    def test_invalid_ref_is_rejected_before_checkout(self):
        code = self.workflow_shell("Validate scan refs")
        for value in ("", "--all", "$(touch injected)", "branch; touch injected"):
            env = dict(os.environ, SCAN_BASE=value, SCAN_HEAD=self.base)
            result = subprocess.run(["bash", "-c", code], env=env, cwd=self.root)
            self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / "injected").exists())

    def test_fetch_failure_stops_before_scan(self):
        env = dict(os.environ, SCAN_BASE="0" * 40, SCAN_HEAD=self.base, RUNNER_TEMP=str(self.root))
        result = subprocess.run(["bash", "-c", self.workflow_shell("Scan incremental changes")],
                                cwd=self.root, env=env, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("python", result.stderr)
        self.assertTrue((self.root / "static-compliance/fetch.log").exists())
        self.assertFalse((self.root / "static-compliance/results.json").exists())

    def test_workflow_is_isolated_read_only_and_has_explicit_inventory(self):
        workflow = WORKFLOW.read_text()
        self.assertIs(runner.SCAN_MODE, scanner.NORMAL_MODE)
        self.assertEqual(workflow.count("Static Compliance Scan (normal)"), 2)
        self.assertNotIn("(strong)", workflow)
        self.assertIn("runs-on: ubuntu-latest", workflow)
        self.assertIn("contents: read", workflow)
        self.assertNotIn("secrets:", workflow)
        self.assertNotIn("pull_request_target:", workflow)
        self.assertNotIn("continue-on-error:", workflow)
        self.assertNotIn("ppu-scheduler-action", workflow)
        self.assertIn("fetch-depth: 0", workflow)
        self.assertIn("sparse-checkout: .github/workflows\n", workflow)
        self.assertEqual(workflow.count("persist-credentials: false"), 2)
        self.assertIn("ref: ${{ github.workflow_sha }}", workflow)
        self.assertIn("ref: ${{ env.SCAN_HEAD }}", workflow)
        self.assertIn("python3 -I", workflow)
        self.assertIn("FETCH_HEAD^{commit}", workflow)
        self.assertIn("if-no-files-found: error", workflow)
        self.assertIn("always()", workflow)
        self.assertLess(workflow.index("Validate scan refs"), workflow.index("Checkout workflow tools"))
        for step in ("Validate scan refs", "Scan incremental changes"):
            subprocess.run(["bash", "-n"], input=self.workflow_shell(step), text=True, check=True)

    @unittest.skipUnless(importlib.util.find_spec("yaml"), "optional YAML parser check")
    def test_workflow_yaml_and_reusable_inputs_are_valid(self):
        import yaml
        workflow = yaml.load(WORKFLOW.read_text(), Loader=yaml.BaseLoader)
        entry = yaml.load((ROOT / ".github/workflows/ci.yml").read_text(), Loader=yaml.BaseLoader)
        self.assertEqual(set(workflow["on"]), {"workflow_call", "workflow_dispatch"})
        self.assertEqual(set(workflow["on"]["workflow_call"]["inputs"]), {"base", "head"})
        self.assertEqual(workflow["permissions"], {"contents": "read"})
        self.assertEqual(entry["jobs"]["ci"]["needs"], ["human-review", "static-compliance"])
        self.assertEqual(entry["jobs"]["static-compliance"]["needs"], "precheck")
        for step in workflow["jobs"]["scan"]["steps"]:
            if "run" in step:
                subprocess.run(["bash", "-n"], input=step["run"], text=True, check=True)

    @unittest.skipUnless(os.environ.get("COMPLIANCE_WORKTREE"), "optional working-tree scan")
    def test_current_worktree_changes_pass_normal_without_staging(self):
        os.chdir(os.environ["COMPLIANCE_WORKTREE"])
        scanner.repo_root.cache_clear()
        scanner.head_exists.cache_clear()
        diff = scanner.run_git(["diff", "--unified=0", "HEAD"])
        paths = scanner.run_git(["diff", "--name-only", "HEAD"]).splitlines()
        untracked = scanner.run_git(["ls-files", "--others", "--exclude-standard", "-z"])
        for path in untracked.split("\0"):
            if path.startswith(".github/") and Path(path).suffix in (".py", ".yml"):
                paths.append(path)
                content = Path(path).read_text()
                self.assertTrue(all(line == line.rstrip() for line in content.splitlines()))
                if Path(path).suffix == ".py":
                    ast.parse(content)
                proc = subprocess.run(["git", "diff", "--no-index", "--unified=0", "/dev/null", path],
                                      capture_output=True, text=True)
                self.assertIn(proc.returncode, (0, 1))
                diff += proc.stdout
        findings, files, lines, _ = scanner.scan_diff(None, diff, scanner.NORMAL_MODE, runner.IGNORED_PATHS)
        paths = [path for path in paths if not scanner.is_ignored(path, runner.IGNORED_PATHS)]
        findings.extend(scanner.scan_paths(paths, scanner.NORMAL_MODE))
        self.assertGreater(files, 0)
        self.assertGreater(lines, 0)
        self.assertEqual(findings, [])

    @unittest.skipUnless(os.environ.get("COMPLIANCE_REFERENCE"), "optional upstream parity check")
    def test_vendored_snapshot_matches_original_executable_ast(self):
        class StripDocs(ast.NodeTransformer):
            def strip(self, node):
                self.generic_visit(node)
                if node.body and isinstance(node.body[0], ast.Expr):
                    value = node.body[0].value
                    if isinstance(value, ast.Constant) and isinstance(value.value, str):
                        node.body.pop(0)
                return node
            visit_Module = strip
            visit_ClassDef = strip
            visit_FunctionDef = strip

        def executable(path):
            return ast.dump(StripDocs().visit(ast.parse(Path(path).read_text())), include_attributes=False)

        self.assertEqual(executable(os.environ["COMPLIANCE_REFERENCE"]), executable(runner.SCANNER_PATH))


if __name__ == "__main__":
    unittest.main(verbosity=2)
