# SPDX-License-Identifier: BSD-3-Clause

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from zipfile import ZipFile

from functional_cases import BASELINE, STATE_DTYPE_CASES, StateDtypeSubset, cases, v_last_lists

ROOT = Path(__file__).resolve().parents[3]
BOARDS = ("OAM-810E", "ZW-M890P")
PHASES = ("decode", "prefill")
EXTENSION_DIRS = {
    "pla/decode", "pla/decode/gdn/v_last", "pla/decode/kda_mtp_sglang",
    "pla/prefill/flashqla", "pla/prefill/flashkdapro", "pla/prefill/flashkdapro/_tc01_fla",
}
BUILD_MANIFEST = "pla/prefill/flashkdapro/_tc01_fla/_build_manifest.json"
START = re.compile(r"\*---\s*start test case (\d+)(?:/(\d+))?(?: \((.*?)\))?\s*---\*")
VERDICT = re.compile(r"---\s*case (pass|fail)\s*!+\s*---")


def item(name, status, detail=""):
    return {"name": name, "status": status, "detail": detail}


def parse_markers(text):
    rows, active, planned = [], None, 0
    for line in text.splitlines():
        start = START.search(line)
        if start:
            if active is not None:
                rows.append(item(active, "FAIL", "Missing end marker"))
            active = f"{start[1]}: {start[3] or ''}"
            planned = max(planned, int(start[2] or 0))
        verdict = VERDICT.search(line)
        if verdict:
            if active is None:
                rows.append(item("protocol", "FAIL", "End marker without a start marker"))
            else:
                rows.append(item(active, "PASS" if verdict[1] == "pass" else "FAIL"))
            active = None
    if active is not None:
        rows.append(item(active, "FAIL", "Missing result when the process ended"))
    totals = re.findall(r"^Total cases:\s*(\d+)", text, re.MULTILINE)
    planned_headers = re.findall(r"^Planned cases:\s*(\d+)", text, re.MULTILINE)
    planned = max([planned] + [int(value) for value in planned_headers])
    if not rows or not totals or len(rows) < planned or len(rows) != sum(map(int, totals)):
        rows.append(item("incomplete", "FAIL", f"Planned {planned}; received {len(rows)} results or missing a complete summary"))
    return rows


def parse_pytest(path):
    rows = []
    for node in ET.parse(path).iter("testcase"):
        status = "PASS"
        detail = ""
        for tag, value in (("skipped", "SKIP"), ("failure", "FAIL"), ("error", "FAIL")):
            child = node.find(tag)
            if child is not None:
                status, detail = value, child.get("message", "")
        rows.append(item(node.get("name", "unknown"), status, detail))
    return rows or [item("empty-pytest", "FAIL", "No tests collected")]


def parse_jsonl(path, expected):
    rows = []
    for raw in path.read_text().splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        rows.append(item(f"{row.get('source')}:{row.get('line')}",
                         "PASS" if row.get("pass") is True else "FAIL",
                         str(row.get("error", ""))))
    if len(rows) != expected or expected == 0:
        rows.append(item("inventory", "FAIL", f"Expected {expected}, received {len(rows)}"))
    return rows


def execute(command, cwd, env, log, timeout):
    with log.open("w") as stream:
        stream.write(shlex.join(command) + "\n")
        stream.flush()
        proc = subprocess.Popen(command, cwd=cwd, env=env, stdout=stream,
                                stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            stream.write(f"\nTIMEOUT after {timeout}s\n")
            return 124
        except BaseException:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            raise


def stage_wheel(wheel, root=ROOT):
    with ZipFile(wheel) as archive:
        extensions = [name for name in archive.namelist()
                      if Path(name).name.startswith("_C.") and name.endswith(".so")]
        if (len(extensions) != len(EXTENSION_DIRS)
                or {str(Path(name).parent) for name in extensions} != EXTENSION_DIRS):
            raise ValueError("Wheel must contain all six PLA extensions")
        if BUILD_MANIFEST not in archive.namelist():
            raise ValueError("Wheel is missing the FLA build manifest")
        for name in archive.namelist():
            if name.startswith("pla/") and name.endswith(".py"):
                if ".." in Path(name).parts or (root / name).read_bytes() != archive.read(name):
                    raise ValueError(f"Wheel does not match the current source: {name}")
        for name in extensions + [BUILD_MANIFEST]:
            destination = root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(name))
            print(f"staged: {name}")


def validate_inventory(selected):
    if not selected:
        raise ValueError("Test matrix is empty")
    for case in selected:
        script = ROOT / ("tests" if case.report == "pytest" else "") / case.argv[0]
        if not script.is_file():
            raise FileNotFoundError(script)
        lists = list(case.lists) + [arg.split("=", 1)[1] for arg in case.argv
                                   if arg.startswith("--caselist=")]
        for name in lists:
            path = ROOT / name
            if not path.is_file() or not any(
                line.strip() and not line.startswith("#") for line in path.read_text().splitlines()
            ):
                raise ValueError(f"Missing or empty caselist: {path}")
            if "--kernel=cuda" in case.argv:
                v_last_lists(path.read_text())


def prepare(case, directory):
    command = [sys.executable, "-u", *case.argv]
    cwd = ROOT
    expected = None
    if case.report == "pytest":
        cwd = ROOT / "tests"
        command = [sys.executable, "-m", "pytest", *case.argv, "-v",
                   "-o", "console_output_style=classic", f"--junitxml={directory / 'pytest.xml'}"]
        if case.name == "k-last-state-dtype":
            command = [sys.executable, "-u", str(Path(__file__).resolve()),
                       "--state-dtype-junit", str(directory / "pytest.xml")]
    elif case.report == "markers" and "--kernel=cuda" in case.argv:
        arg = next(arg for arg in case.argv if arg.startswith("--caselist="))
        source = ROOT / arg.split("=", 1)[1]
        selected = v_last_lists(source.read_text())
        lists = directory / "lists"
        lists.mkdir()
        for tp, lines in selected.items():
            name = source.name.replace("_tp1_", f"_tp{tp}_")
            if f"_tp{tp}_" not in name:
                raise ValueError("v-last input must be a TP1 caselist")
            (lists / name).write_text("\n".join(lines) + "\n")
        expected = sum(map(len, selected.values()))
        command[command.index(arg)] = f"--caselist={lists}"
    elif case.report == "jsonl":
        lists = directory / "lists"
        lists.mkdir()
        expected = 0
        for name in case.lists:
            source = ROOT / name
            shutil.copyfile(source, lists / source.name)
            expected += sum(line.startswith("[KDA]") for line in source.read_text().splitlines())
        command += [f"--list-dir={lists}", f"--output={directory / 'replay.jsonl'}",
                    f"--expected-list-files={len(case.lists)}", f"--expected-records={expected}"]
    return command, cwd, expected


def run_case(case, output, timeout):
    directory = output / case.name
    directory.mkdir()
    started = time.monotonic()
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PLA_FLASHKDA_TC01_IMPL", None)
    env["SAIL_PLA_CUDA"] = "1"
    if case.legacy:
        env["PLA_FLASHKDA_TC01_IMPL"] = "legacy"
    log = directory / "stdout.log"
    command, cwd, expected = prepare(case, directory)
    rc = execute(command, cwd, env, log, timeout)
    text = log.read_text(errors="replace")
    try:
        if case.report == "markers":
            children = parse_markers(text)
            if expected is not None and len(children) != expected:
                children.append(item("inventory", "FAIL", f"Expected {expected}, received {len(children)}"))
        elif case.report == "pytest":
            children = parse_pytest(directory / "pytest.xml")
        elif case.report == "jsonl":
            children = parse_jsonl(directory / "replay.jsonl", expected)
        else:
            skipped = case.name == "flashkda-tc01-auto-cp" and "SKIP: tc01 test requires" in text
            children = [item(case.name, "SKIP" if skipped else "PASS")]
    except (OSError, ValueError, ET.ParseError) as exc:
        children = [item("report", "FAIL", str(exc))]
    if rc:
        children.append(item("process", "FAIL", f"exit={rc}; log: {case.name}/stdout.log"))
    counts = Counter(child["status"] for child in children)
    status = "FAIL" if counts["FAIL"] else ("SKIP" if not counts["PASS"] else "PASS")
    result = {"name": case.name, "status": status, "returncode": rc,
              "seconds": round(time.monotonic() - started, 2), "children": children,
              "command": command, "log": f"{case.name}/stdout.log"}
    print(f"{case.name}: {status} {dict(counts)} ({result['seconds']}s)", flush=True)
    if status == "FAIL":
        print("\n".join(text.splitlines()[-30:]), flush=True)
    return result


def markdown(report):
    records = report["results"]
    counts = Counter(record["status"] for record in records)
    lines = [f"## PLA {report['board']} / {report['phase']}",
             f"Source revision: `{report.get('revision', 'local')}`; functional matrix baseline: `{BASELINE}`.",
             f"Test groups: {dict(counts)}. PASS excludes SKIP; NOT_RUN means not executed.",
             "Scripts without per-case reporting are counted at process level; see logs for detailed assertions.", "",
             "| Test group | Status | Passed | Failed | Skipped | Seconds |",
             "|---|---|---:|---:|---:|---:|"]
    for record in records:
        child = Counter(row["status"] for row in record.get("children", []))
        lines.append(f"| {record['name']} | {record['status']} | {child['PASS']} | "
                     f"{child['FAIL']} | {child['SKIP']} | {record.get('seconds', 0)} |")
    return "\n".join(lines) + "\n"


def save_report(report, output):
    text = json.dumps(report, ensure_ascii=False, indent=2)
    pending = output / "results.json.tmp"
    pending.write_text(text)
    pending.replace(output / "results.json")
    (output / "summary.md").write_text(markdown(report))
    suite = ET.Element("testsuite", name=f"{report['board']}-{report['phase']}")
    totals = Counter()
    for record in report["results"]:
        children = record.get("children") or [item(record["name"], "FAIL", "Not executed")]
        for child in children:
            node = ET.SubElement(suite, "testcase", classname=record["name"], name=child["name"])
            totals["tests"] += 1
            if child["status"] == "FAIL":
                totals["failures"] += 1
                ET.SubElement(node, "failure", message=child.get("detail", "") or "See group log")
            elif child["status"] == "SKIP":
                totals["skipped"] += 1
                ET.SubElement(node, "skipped", message=child.get("detail", ""))
    for key in ("tests", "failures", "skipped"):
        suite.set(key, str(totals[key]))
    ET.ElementTree(suite).write(output / "junit.xml", encoding="utf-8", xml_declaration=True)


def summarize(root, boards=BOARDS):
    reports = {}
    failed = False
    title = "Both Platforms" if len(boards) == 2 else ", ".join(boards)
    texts = [f"# PLA {title} Functional Test Summary\n"]
    for name in ("PLA_BUILD_RESULT", "PLA_TEST_RESULT", "PLA_DECODE_RESULT", "PLA_PREFILL_RESULT"):
        if os.environ.get(name):
            texts.append(f"{name}: `{os.environ[name]}`\n")
            failed |= os.environ[name] != "success"
    for path in root.rglob("results.json"):
        try:
            report = json.loads(path.read_text())
            key = (report["board"], report["phase"])
            if key in reports or not isinstance(report["results"], list):
                raise ValueError("Duplicate report or invalid results format")
            reports[key] = report
        except (OSError, ValueError, KeyError, TypeError) as exc:
            texts.append(f"FAIL: Cannot parse report {path.name}: {exc}\n")
            failed = True
    for board in boards:
        for phase in PHASES:
            report = reports.get((board, phase))
            if report is None:
                texts.append(f"## {board} / {phase}\nFAIL: No report received. Check build, scheduling, or environment setup logs.\n")
                failed = True
                continue
            expected = {case.name for case in cases() if case.phase == phase} | {"environment"}
            actual = {row["name"] for row in report["results"]}
            if actual != expected or len(actual) != len(report["results"]):
                texts.append(f"FAIL: Incomplete test group inventory for {board}/{phase}.\n")
                failed = True
            failed |= any(row["status"] not in ("PASS", "SKIP") for row in report["results"])
            texts.append(markdown(report))
    text = "\n".join(texts)
    print(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as stream:
            stream.write(text)
    return int(failed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=PHASES)
    parser.add_argument("--board", choices=BOARDS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--budget-seconds", type=int, default=18000)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--summary-root", type=Path)
    parser.add_argument("--stage-wheel", type=Path)
    parser.add_argument("--state-dtype-junit", type=Path)
    args = parser.parse_args()
    if args.state_dtype_junit is not None:
        import pytest
        return pytest.main(["test_k_last_state_dtype.py", "-v", "-o", "console_output_style=classic",
                            f"--junitxml={args.state_dtype_junit}"], plugins=[StateDtypeSubset()])
    if args.stage_wheel is not None:
        stage_wheel(args.stage_wheel)
        return 0
    if args.summary_root is not None:
        return summarize(args.summary_root, (args.board,) if args.board else BOARDS)
    selected = [case for case in cases() if not args.phase or case.phase == args.phase]
    validate_inventory(selected)
    if args.list:
        for case in selected:
            print(case.phase, case.name, shlex.join(case.argv), case.lists)
            if "--kernel=cuda" in case.argv:
                source = ROOT / next(arg.split("=", 1)[1] for arg in case.argv if arg.startswith("--caselist="))
                text = source.read_text()
                subset = v_last_lists(text)
                batches = [int(re.search(r"batch_size:(\d+)", line)[1]) for line in subset[1]]
                original = sum(bool(line.strip()) and not line.strip().startswith("#") for line in text.splitlines())
                print(f"  v-last: TP={tuple(subset)}; batch={batches}; cases={sum(map(len, subset.values()))}"
                      f" (from {original} TP1 records)")
            elif case.name == "k-last-state-dtype":
                print(f"  state-dtype: {STATE_DTYPE_CASES} precision cases selected by CI")
        print(f"Total groups: {len(selected)}")
        return 0
    if not args.output or not args.board or not args.phase or args.budget_seconds <= 0:
        parser.error("Running tests requires --output, --board, --phase, and a positive time budget")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"board": args.board, "phase": args.phase, "baseline": BASELINE,
              "revision": os.environ.get("PLA_TEST_REVISION", "local"),
              "results": [item("environment", "NOT_RUN")] + [item(case.name, "NOT_RUN") for case in selected]}
    save_report(report, output)
    deadline = time.monotonic() + args.budget_seconds

    def terminate(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, terminate)
    try:
        probe = (
            "import importlib, torch; "
            "assert torch.cuda.is_available(), 'PPU unavailable'; "
            "print('torch=', torch.__version__, 'devices=', torch.cuda.device_count()); "
            "print(torch.cuda.get_device_name(0)); "
            "[print(importlib.import_module(n).__file__) for n in "
            "('pla.decode._C','pla.decode.gdn.v_last._C','pla.decode.kda_mtp_sglang._C',"
            "'pla.prefill.flashqla._C','pla.prefill.flashkdapro._C','pla.prefill.flashkdapro._tc01_fla._C')]; "
            "torch.cuda.synchronize()"
        )
        probe_env = os.environ.copy()
        probe_env.pop("PYTHONPATH", None)
        rc = execute([sys.executable, "-c", probe], output, probe_env, output / "environment.log", 120)
        report["results"][0] = {"name": "environment", "status": "FAIL" if rc else "PASS",
                                "children": [item("environment", "FAIL" if rc else "PASS", f"exit={rc}")]}
        save_report(report, output)
        if rc:
            return 1
        for index, case in enumerate(selected, 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            print(f"[{index}/{len(selected)}] {case.name}", flush=True)
            try:
                report["results"][index] = run_case(case, output, min(case.timeout, remaining))
            except Exception as exc:
                report["results"][index] = {"name": case.name, "status": "FAIL",
                    "children": [item("runner", "FAIL", f"{type(exc).__name__}: {exc}")]}
            save_report(report, output)
    finally:
        save_report(report, output)
        print(markdown(report), flush=True)
    return int(any(row["status"] not in ("PASS", "SKIP") for row in report["results"]))


if __name__ == "__main__":
    raise SystemExit(main())
