#!/usr/bin/env python3
"""Run the shared normal rules against an explicitly identified PR change set."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import sys

SCANNER_PATH = Path(__file__).with_name("static_compliance_scan.py")
spec = importlib.util.spec_from_file_location("static_compliance_scan", SCANNER_PATH)
scanner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = scanner
spec.loader.exec_module(scanner)

SCAN_MODE = scanner.NORMAL_MODE
IGNORED_PATHS = (
    ".aoneci",
    ".github/workflows/scripts/static_compliance_scan.py",
    ".github/workflows/scripts/test_static_compliance.py",
)


def scan(base, head):
    for name, value in (("base", base), ("head", head)):
        if not re.fullmatch(r"[0-9a-f]{40}", value):
            raise RuntimeError(f"{name} must be a full commit SHA")
        scanner.run_git(["rev-parse", "--verify", value + "^{commit}"])
    actual_head = scanner.run_git(["rev-parse", "HEAD"]).strip()
    if actual_head != head:
        raise RuntimeError("Checkout does not match the requested head SHA")
    merge_base = scanner.run_git(["merge-base", base, head]).strip()
    if not merge_base or merge_base == head:
        raise RuntimeError("Incremental base resolves to head; nothing would be checked")
    scanner.repo_root.cache_clear()
    scanner.head_exists.cache_clear()
    findings, files, lines, commits = scanner.scan_diff(
        merge_base, None, SCAN_MODE, IGNORED_PATHS,
    )
    if not files and not lines and not commits:
        raise RuntimeError("Empty scan inventory")
    print(f"Base SHA: {base}")
    print(f"Head SHA: {head}")
    print(f"Merge base: {merge_base}")
    print(f"Rule mode: {SCAN_MODE.name}")
    print(f"Ignored paths: {' '.join(IGNORED_PATHS)}")
    print("Only rule definitions, rule fixtures and legacy pipeline files are excluded.")
    print(f"Scanned changed files: {files}")
    print(f"Scanned added lines: {lines}")
    print(f"Scanned commit messages: {commits}")
    scanner.print_findings(findings)
    return {
        "status": "FAIL" if findings else "PASS",
        "base": base, "head": head, "merge_base": merge_base,
        "files": files, "added_lines": lines, "commits": commits,
        "findings": [item._asdict() for item in scanner.sort_findings(findings)],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    log = io.StringIO()
    with redirect_stdout(log), redirect_stderr(log):
        try:
            result = scan(args.base, args.head)
            code = int(result["status"] == "FAIL")
        except (RuntimeError, OSError) as exc:
            result = {"status": "ERROR", "error": str(exc)}
            print(f"Static compliance scan ERROR: {exc}")
            code = 2
    result["mode"] = SCAN_MODE.name
    result["ignored_paths"] = list(IGNORED_PATHS)
    result["scanner_sha256"] = hashlib.sha256(SCANNER_PATH.read_bytes()).hexdigest()
    (args.output / "scan.log").write_text(log.getvalue(), encoding="utf-8")
    (args.output / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8",
    )
    summary = f"## Static Compliance Scan ({SCAN_MODE.name})\n\nStatus: {result['status']}\n\n"
    if result["status"] != "ERROR":
        summary += (f"Changed files: {result['files']}; added lines: {result['added_lines']}; "
                    f"commit messages: {result['commits']}; findings: {len(result['findings'])}.\n\n")
    summary += "Detailed results: static-compliance-results artifact.\n\n"
    summary += "Explicit exclusions:\n" + "".join(f"- `{path}`\n" for path in IGNORED_PATHS)
    (args.output / "summary.md").write_text(summary, encoding="utf-8")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as handle:
            handle.write(summary)
    print(f"Static Compliance Scan ({SCAN_MODE.name}): {result['status']}")
    return code


if __name__ == "__main__":
    sys.exit(main())
