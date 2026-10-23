#!/usr/bin/env python3
"""Static compliance scanner shared by the PPU libraries.

Vendored snapshot with explanatory comments/docstrings omitted. Rule definitions,
scan logic, CLI and exit codes are unchanged. Incremental scans require --base;
--all audits tracked content; --diff-file reads a supplied diff.
"""

import argparse
import functools
import os
import re
import subprocess
import sys
from typing import (Dict, Iterable, Iterator, List, NamedTuple, Optional,
                    Sequence, Set, Tuple)

NEXT_GEN_CHIP_RE = re.compile(
    r"(?:ppu|b" r"tv)[\s_\-]*v?(?:1\.7|2\.0|0017|0020|17|20)(?![0-9.])",
    re.IGNORECASE,
)
EULA_COMPONENT_RE = re.compile(r"\b(?:cuteDSL|pycute)\b", re.IGNORECASE)
TENSOR_CORE_RE = re.compile(r"tensor[\s_\-]*cores?", re.IGNORECASE)
INTERNAL_HOST_RE = re.compile(r"alibaba|gitlab", re.IGNORECASE)
CHINESE_TEXT_RE = re.compile(
    r"[\u3000-\u303f\u4e00-\u9fff\uff01-\uff60\uffe0-\uffe6]+"
)
ILLEGAL_CHAR_RE = re.compile(
    r"[\u00a0\u00ad\u061c\u180e\u2000-\u200f\u2018-\u201f\u2028-\u202f"
    r"\u205f\u2060-\u2064\u2066-\u2069\ufeff\ufff9-\ufffb]+"
)
CORE_LINE_RULES = (
    ("next-gen-chip-term", NEXT_GEN_CHIP_RE),
    ("restricted-eula-component", EULA_COMPONENT_RE),
    ("tensor-core-term", TENSOR_CORE_RE),
    ("internal-host-name", INTERNAL_HOST_RE),
    ("chinese-text", CHINESE_TEXT_RE),
    ("illegal-char", ILLEGAL_CHAR_RE),
)
ACOMPUTE_RE = re.compile(r"\bacompute[A-Za-z0-9_]*\b", re.IGNORECASE)
VENDOR_LINE_RULES = (("acompute-name", ACOMPUTE_RE),)
VENDOR_LIBRARY_RE = re.compile(
    r"\b(?:cuda|cudnn|cublas|curand|cusolver|cusparse|cufft|cupti|cutensor)"
    r"[A-Za-z0-9_]*\b",
    re.IGNORECASE,
)
VENDOR_NAME_RE = re.compile(
    r"\b(?:"
    r"[nN][vV][A-Za-z0-9_]*|"
    r"[sS][mM][0-9][A-Za-z0-9_]*|"
    r"[nN](?:[cC][uU]|[sS][yY][sS]|[sS][iI][gG][hH][tT])[A-Za-z0-9_\-]*|"
    r"(?:cu[A-Z0-9_]|CU[A-Za-z0-9_]|Cu[A-Za-z0-9_])[A-Za-z0-9_]*"
    r")\b"
)
ALLOWED_VENDOR_NAME_PREFIXES = (
    "cute", "cutlass", "cub", "current", "cus",
    "cuid", "cuarch", "cu39",
    "cu_count", "cu_idx", "cu_num", "cu_var", "cu_occupancy",
    "cu_seqlens", "cu_seq_len",
)
VENDOR_EXEMPT_URL_RE = re.compile(r"(?:https?://|git@|ssh://)")
VENDOR_EXEMPT_COPYRIGHT_RE = re.compile(r"copyright", re.IGNORECASE)
VENDOR_ATTRIBUTION_PREFIX = "originally developed by nvidia"
ASM_START_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:__asm__|__asm|asm)(?![A-Za-z0-9_])"
    r"(?:\s+(?:__volatile__|volatile))?\s*\("
)
ASM_OPERAND_RE = re.compile(r"[A-Za-z]\"\s*\(")
ASM_STARTS_WITH_LETTER_RE = re.compile(r"[A-Za-z]")
ASM_LABEL_RE = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*:(?!:)\s*")
ASM_STATEMENT_LEAD_RE = re.compile(r"(?:\s|\\[ntr]|[\"'])*")
ASM_MNEMONIC_SPLIT_RE = re.compile(r"[\s;\"']+")
ASM_STRING_PREFIXES = ('u8"', 'u"', 'U"', 'L"', 'R"', '"', "'")
ASM_COMMENT_PREFIXES = ("//", "/*", "*")


class Finding(NamedTuple):
    rule: str
    path: str
    line: Optional[int]
    token: str
    text: str


class DiffLine(NamedTuple):
    path: str
    line: int
    text: str


class Commit(NamedTuple):
    sha: str
    message: str


class Mode(NamedTuple):
    name: str
    line_rules: Sequence[Tuple[str, "re.Pattern"]]
    check_vendor_names: bool
    check_paths: bool


STRONG_MODE = Mode(
    name="strong",
    line_rules=CORE_LINE_RULES + VENDOR_LINE_RULES,
    check_vendor_names=True,
    check_paths=True,
)
NORMAL_MODE = Mode(
    name="normal",
    line_rules=CORE_LINE_RULES,
    check_vendor_names=False,
    check_paths=False,
)
MODES = {mode.name: mode for mode in (STRONG_MODE, NORMAL_MODE)}
DEFAULT_MODE_NAME = STRONG_MODE.name


def run_git(args: Sequence[str], check: bool = True) -> str:
    proc = subprocess.run(
        ["git", *args], check=False,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        encoding="utf-8", errors="replace",
    )
    if check and proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "git command failed")
    return proc.stdout


@functools.lru_cache(maxsize=None)
def repo_root() -> str:
    try:
        top = run_git(["rev-parse", "--show-toplevel"]).strip()
    except RuntimeError:
        top = ""
    return top or os.getcwd()


@functools.lru_cache(maxsize=None)
def head_exists() -> bool:
    try:
        run_git(["rev-parse", "--verify", "HEAD^{commit}"], check=True)
    except RuntimeError:
        return False
    return True


def normalize_path(path: str) -> str:
    return path.replace(os.sep, "/")


def normalize_ignore(value: str) -> str:
    cleaned = normalize_path(value.strip())
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    return cleaned.strip("/")


def is_ignored(path: str, ignored: Sequence[str]) -> bool:
    return any(path == prefix or path.startswith(prefix + "/") for prefix in ignored)


def resolve_base(base_ref: str) -> str:
    try:
        run_git(["rev-parse", "--verify", base_ref + "^{commit}"], check=True)
    except RuntimeError:
        raise RuntimeError(f"base ref {base_ref!r} cannot be resolved by git")
    try:
        merge_base = run_git(["merge-base", base_ref, "HEAD"], check=True).strip()
    except RuntimeError:
        print(f"WARNING: {base_ref} shares no history with HEAD, using it directly. "
              f"In a shallow clone, fetch more depth for a meaningful diff.",
              file=sys.stderr)
        return base_ref
    return merge_base or base_ref


def changed_paths(base: Optional[str]) -> List[str]:
    if base:
        output = run_git(["diff", "--name-only", "--diff-filter=ACMR", base, "HEAD"])
    else:
        output = run_git(["ls-files"])
    return [normalize_path(line.strip()) for line in output.splitlines() if line.strip()]


def diff_text(base: Optional[str]) -> str:
    if base:
        return run_git(["diff", "--unified=0", "--no-ext-diff", "--no-color", base, "HEAD"])
    return run_git(["show", "--format=", "--unified=0", "--no-ext-diff", "--no-color", "HEAD"])


HUNK_HEADER_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def parse_added_lines(diff: str) -> Iterator[DiffLine]:
    current_path: Optional[str] = None
    new_line: Optional[int] = None
    for raw_line in diff.splitlines():
        if raw_line.startswith("+++ "):
            marker_path = raw_line[4:].strip()
            if marker_path == "/dev/null":
                current_path = None
            elif marker_path.startswith("b/"):
                current_path = normalize_path(marker_path[2:])
            else:
                current_path = normalize_path(marker_path)
            continue
        if raw_line.startswith("--- "):
            continue
        hunk = HUNK_HEADER_RE.match(raw_line)
        if hunk:
            new_line = int(hunk.group(1))
            continue
        if current_path is None or new_line is None:
            continue
        if raw_line.startswith("+"):
            yield DiffLine(current_path, new_line, raw_line[1:])
            new_line += 1
        elif raw_line.startswith("-"):
            continue
        elif raw_line.startswith("\\"):
            continue
        else:
            new_line += 1


COMMIT_LOG_FORMAT = "%h%n%B"


def commit_messages(base: Optional[str]) -> List[Commit]:
    if not head_exists():
        return []
    revisions = f"{base}..HEAD" if base else "HEAD"
    output = run_git(["log", "-z", f"--format={COMMIT_LOG_FORMAT}", revisions])
    commits: List[Commit] = []
    for entry in output.split("\0"):
        if not entry.strip():
            continue
        sha, _, message = entry.partition("\n")
        commits.append(Commit(sha.strip(), message))
    return commits


def is_vendor_exempt_line(text: str) -> bool:
    return bool(
        VENDOR_EXEMPT_URL_RE.search(text)
        or VENDOR_EXEMPT_COPYRIGHT_RE.search(text)
        or text.strip().lower().startswith(VENDOR_ATTRIBUTION_PREFIX)
    )


def is_allowed_vendor_name(token: str, text: str, start: int) -> bool:
    lowered = token.lower()
    if start > 0 and text[start - 1] == "\\" and lowered.startswith("nv"):
        return True
    return lowered.startswith(ALLOWED_VENDOR_NAME_PREFIXES)


def scan_vendor_names(path: str, line: Optional[int], text: str) -> List[Finding]:
    if is_vendor_exempt_line(text):
        return []
    findings: List[Finding] = []
    for match in VENDOR_LIBRARY_RE.finditer(text):
        findings.append(Finding("nv-style-name", path, line, match.group(0), text))
    for match in VENDOR_NAME_RE.finditer(text):
        token = match.group(0)
        if is_allowed_vendor_name(token, text, match.start()):
            continue
        findings.append(Finding("nv-style-name", path, line, token, text))
    return findings


def scan_text(path: str, line: Optional[int], text: str, mode: Mode) -> List[Finding]:
    findings: List[Finding] = []
    for rule, pattern in mode.line_rules:
        for match in pattern.finditer(text):
            findings.append(Finding(rule, path, line, match.group(0), text))
    if mode.check_vendor_names:
        findings.extend(scan_vendor_names(path, line, text))
    return findings


def scan_paths(paths: Iterable[str], mode: Mode) -> List[Finding]:
    if not mode.check_paths:
        return []
    findings: List[Finding] = []
    for path in paths:
        findings.extend(scan_text(path, None, path, mode))
    return findings


def scan_commit_messages(commits: Sequence[Commit], mode: Mode) -> List[Finding]:
    findings: List[Finding] = []
    for commit in commits:
        location = f"commit {commit.sha}"
        for number, text in enumerate(commit.message.splitlines(), start=1):
            findings.extend(scan_text(location, number, text, mode))
    return findings


def normalize_asm_payload(text: str) -> str:
    stripped = text.strip()
    for prefix in ASM_STRING_PREFIXES:
        if stripped.startswith(prefix):
            return stripped[len(prefix):].lstrip()
    return stripped


def strip_asm_statement_lead(statement: str) -> str:
    lead = ASM_STATEMENT_LEAD_RE.match(statement)
    return statement[lead.end():] if lead else statement.strip()


def scan_asm_line(item: DiffLine, text: str) -> List[Finding]:
    findings: List[Finding] = []
    for statement in normalize_asm_payload(text).split(";"):
        payload = strip_asm_statement_lead(statement)
        if not payload or payload.startswith(ASM_COMMENT_PREFIXES):
            continue
        if ASM_OPERAND_RE.match(payload):
            continue
        label = ASM_LABEL_RE.match(payload)
        if label:
            payload = payload[label.end():]
        if ASM_STARTS_WITH_LETTER_RE.match(payload) and not payload.startswith("ppu."):
            mnemonic = ASM_MNEMONIC_SPLIT_RE.split(payload, maxsplit=1)[0]
            findings.append(
                Finding("asm-ppu-prefix", item.path, item.line, mnemonic, item.text))
    return findings


def blank_comments_and_string_bodies(text: str, in_block_comment: bool) -> Tuple[str, bool]:
    out: List[str] = []
    index = 0
    length = len(text)
    quote = ""
    while index < length:
        char = text[index]
        if in_block_comment:
            if text.startswith("*/", index):
                in_block_comment = False
                out.append("  ")
                index += 2
            else:
                out.append(" ")
                index += 1
            continue
        if quote:
            if char == "\\" and index + 1 < length:
                out.append("  ")
                index += 2
            elif char == quote:
                quote = ""
                out.append(char)
                index += 1
            else:
                out.append(" ")
                index += 1
            continue
        if text.startswith("//", index):
            out.append(" " * (length - index))
            break
        if text.startswith("/*", index):
            in_block_comment = True
            out.append("  ")
            index += 2
            continue
        if char in "\"'":
            quote = char
        out.append(char)
        index += 1
    return "".join(out), in_block_comment


def asm_block_line_numbers(lines: Sequence[str]) -> Set[int]:
    inside: Set[int] = set()
    in_block_comment = False
    open_block = False
    depth = 0
    for number, raw in enumerate(lines, start=1):
        clean, in_block_comment = blank_comments_and_string_bodies(raw, in_block_comment)
        if open_block:
            rest = clean
        else:
            start = ASM_START_RE.search(clean)
            if start is None:
                continue
            open_block = True
            depth = 0
            rest = clean[start.end() - 1:]
        inside.add(number)
        for char in rest:
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    open_block = False
                    break
    return inside


def read_current_file_lines(
    path: str, cache: Dict[str, Optional[List[str]]],
) -> Optional[List[str]]:
    if path in cache:
        return cache[path]
    actual_path = path if os.path.isabs(path) else os.path.join(repo_root(), path)
    try:
        with open(actual_path, "r", encoding="utf-8", errors="replace") as handle:
            lines: Optional[List[str]] = handle.read().splitlines()
    except OSError:
        lines = None
    cache[path] = lines
    return lines


def current_asm_line_numbers(
    path: str, file_cache: Dict[str, Optional[List[str]]],
    asm_cache: Dict[str, Optional[Set[int]]],
) -> Optional[Set[int]]:
    if path in asm_cache:
        return asm_cache[path]
    lines = read_current_file_lines(path, file_cache)
    asm_cache[path] = None if lines is None else asm_block_line_numbers(lines)
    return asm_cache[path]


def asm_payload(text: str) -> str:
    clean, _ = blank_comments_and_string_bodies(text, False)
    start = ASM_START_RE.search(clean)
    return text[start.end():] if start else text


def scan_asm_blocks(added_lines: Sequence[DiffLine]) -> List[Finding]:
    findings: List[Finding] = []
    file_cache: Dict[str, Optional[List[str]]] = {}
    asm_cache: Dict[str, Optional[Set[int]]] = {}
    current_path: Optional[str] = None
    previous_line: Optional[int] = None
    in_asm = False
    depth = 0
    for item in added_lines:
        inside = current_asm_line_numbers(item.path, file_cache, asm_cache)
        if inside is not None:
            if item.line in inside:
                findings.extend(scan_asm_line(item, asm_payload(item.text)))
            continue
        if item.path != current_path:
            current_path = item.path
            previous_line = None
            in_asm = False
            depth = 0
        if previous_line is not None and item.line != previous_line + 1:
            in_asm = False
            depth = 0
        previous_line = item.line
        text = item.text
        clean, _ = blank_comments_and_string_bodies(text, False)
        start = ASM_START_RE.search(clean)
        if start:
            in_asm = True
            depth = 0
            rest = clean[start.end() - 1:]
            findings.extend(scan_asm_line(item, text[start.end():]))
        elif in_asm:
            rest = clean
            findings.extend(scan_asm_line(item, text))
        else:
            rest = ""
        for char in rest:
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    in_asm = False
                    break
    return findings


def apply_rules(added_lines: Sequence[DiffLine], mode: Mode) -> List[Finding]:
    findings: List[Finding] = []
    for item in added_lines:
        findings.extend(scan_text(item.path, item.line, item.text, mode))
    findings.extend(scan_asm_blocks(added_lines))
    return findings


def scan_diff(
    base: Optional[str], diff_override: Optional[str], mode: Mode,
    ignored: Sequence[str] = (), check_commit_messages: bool = True,
) -> Tuple[List[Finding], int, int, int]:
    paths = changed_paths(base) if diff_override is None else []
    paths = [path for path in paths if not is_ignored(path, ignored)]
    text = diff_text(base) if diff_override is None else diff_override
    scanned_files = set(paths)
    added_lines: List[DiffLine] = []
    for item in parse_added_lines(text):
        if is_ignored(item.path, ignored):
            continue
        added_lines.append(item)
        scanned_files.add(item.path)
    commits = (
        commit_messages(base) if check_commit_messages and diff_override is None else []
    )
    findings = apply_rules(added_lines, mode)
    findings.extend(scan_paths(paths, mode))
    findings.extend(scan_commit_messages(commits, mode))
    return dedupe_findings(findings), len(scanned_files), len(added_lines), len(commits)


def tracked_paths() -> List[str]:
    paths = [normalize_path(line) for line in run_git(["ls-files"]).split("\n") if line.strip()]
    return sorted(paths)


def scan_whole_tree(
    mode: Mode, ignored: Sequence[str] = (), check_commit_messages: bool = True,
) -> Tuple[List[Finding], int, int, int]:
    added_lines: List[DiffLine] = []
    scanned_paths: List[str] = []
    for path in tracked_paths():
        if is_ignored(path, ignored):
            continue
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                lines = handle.read().split("\n")
        except OSError:
            continue
        if lines and lines[-1] == "":
            lines.pop()
        scanned_paths.append(path)
        for number, text in enumerate(lines, start=1):
            added_lines.append(DiffLine(path, number, text))
    commits = commit_messages(None) if check_commit_messages else []
    findings = apply_rules(added_lines, mode)
    findings.extend(scan_paths(scanned_paths, mode))
    findings.extend(scan_commit_messages(commits, mode))
    return dedupe_findings(findings), len(scanned_paths), len(added_lines), len(commits)


def dedupe_findings(findings: Sequence[Finding]) -> List[Finding]:
    seen = set()
    unique: List[Finding] = []
    for finding in findings:
        key = (finding.rule, finding.path, finding.line, finding.token)
        if key in seen:
            continue
        seen.add(key)
        unique.append(finding)
    return unique


def sort_findings(findings: Sequence[Finding]) -> List[Finding]:
    return sorted(
        findings,
        key=lambda f: (f.path, -1 if f.line is None else f.line, f.rule, f.token),
    )


UNPRINTABLE_RE = re.compile(r"[^\t\x20-\x7e\u00a0-\U0010ffff]")
MAX_REPORTED_TEXT = 200


def escaped(chars: str) -> str:
    return "".join(f"\\u{ord(char):04x}" for char in chars)


def printable(text: str) -> str:
    cleaned = UNPRINTABLE_RE.sub("\ufffd", text)
    cleaned = ILLEGAL_CHAR_RE.sub(lambda m: escaped(m.group()), cleaned).strip()
    if len(cleaned) > MAX_REPORTED_TEXT:
        return cleaned[:MAX_REPORTED_TEXT] + " ..."
    return cleaned


def print_finding_lines(findings: Sequence[Finding]) -> None:
    for finding in findings:
        location = finding.path if finding.line is None else f"{finding.path}:{finding.line}"
        print(f"- [{finding.rule}] {location}: token={printable(finding.token)!r}")
        text = printable(finding.text)
        if text:
            print(f"  {text}")


def print_findings(findings: Sequence[Finding]) -> None:
    if not findings:
        print("Static compliance scan PASS")
        return
    ordered = sort_findings(findings)
    print("Static compliance scan FAILED")
    print(f"Findings: {len(ordered)}")
    print_finding_lines(ordered)
    counts: Dict[str, int] = {}
    for finding in ordered:
        counts[finding.rule] = counts.get(finding.rule, 0) + 1
    print("Summary by rule:")
    for rule in sorted(counts):
        print(f"  {rule}: {counts[rule]}")


def read_diff_override(source: str) -> str:
    if source == "-":
        return sys.stdin.read()
    with open(source, "r", encoding="utf-8", errors="replace") as handle:
        return handle.read()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan incremental code for static compliance rules."
    )
    parser.add_argument(
        "--base",
        help="Base git ref for the incremental scan. Required unless --all or "
             "--diff-file is given. There is no default and nothing is guessed: "
             "a base the caller did not choose scans less than it appears to "
             "and passes for the wrong reason.",
    )
    parser.add_argument(
        "--diff-file",
        help="Read unified diff from a file. Use '-' for stdin.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Scan every line of every file git tracks instead of a diff. "
             "No --base can do this: a line unchanged since the first commit "
             "never shows up as added. Ignores --base and --diff-file.",
    )
    parser.add_argument(
        "--ignore", action="append", metavar="PATH",
        help="Skip this file, or this directory and everything under it. "
             "Repeatable. The path is relative to the repository root. Nothing "
             "else is excluded by path or extension, so this is the one way to "
             "leave a tree out, and the run prints back what it skipped.",
    )
    parser.add_argument(
        "--mode", choices=sorted(MODES), default=DEFAULT_MODE_NAME,
        help="Which rules to apply. strong, the default, adds vendor style "
             "naming to the core rules, which is an audit rather than a gate: "
             "expect findings in the hundreds. normal keeps only licensed "
             "material and unreleased hardware information, so that every "
             "finding is actionable on its own, which is what a merge request "
             "gate needs.",
    )
    parser.add_argument(
        "--no-commit-messages", action="store_true",
        help="Skip the commit message pass. Messages are scanned by default, "
             "base..HEAD or every commit under --all, because they ship with "
             "the history. Use this for a repository whose legacy messages "
             "cannot be reworded; the run prints back that it skipped them.",
    )
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    mode = MODES[args.mode]
    ignored = tuple(
        prefix for prefix in (normalize_ignore(value) for value in args.ignore or ())
        if prefix
    )
    check_commits = not args.no_commit_messages
    diff_override = None
    base = None
    try:
        if args.all:
            if args.base or args.diff_file:
                print("WARNING: --all scans the whole tree, --base and --diff-file are ignored.",
                      file=sys.stderr)
            findings, file_count, line_count, commit_count = scan_whole_tree(
                mode, ignored, check_commits)
        else:
            if args.diff_file:
                diff_override = read_diff_override(args.diff_file)
            else:
                if not args.base:
                    print("ERROR: --base is required for an incremental scan. There is "
                          "no default: a guessed base scans less than you think and "
                          "passes for the wrong reason. Use --all to scan everything, "
                          "or --diff-file to supply a diff.", file=sys.stderr)
                    return 2
                base = resolve_base(args.base)
                print(f"Using git base: {base}")
                if base == run_git(["rev-parse", "HEAD"]).strip():
                    print("WARNING: base resolves to HEAD, the incremental diff is empty.",
                          file=sys.stderr)
            findings, file_count, line_count, commit_count = scan_diff(
                base, diff_override, mode, ignored, check_commits)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"ERROR: cannot read diff input: {exc}", file=sys.stderr)
        return 2
    print(f"Rule mode: {mode.name}")
    if ignored:
        print(f"Ignored paths: {' '.join(ignored)}")
    if args.all:
        print(f"Scanned tracked files: {file_count}")
        print(f"Scanned lines: {line_count}")
    else:
        print(f"Scanned changed files: {file_count}")
        print(f"Scanned added lines: {line_count}")
    if not check_commits:
        print("Commit messages: not scanned (--no-commit-messages)")
    elif diff_override is not None:
        print("Commit messages: not scanned, a supplied diff carries none")
    else:
        print(f"Scanned commit messages: {commit_count}")
    if not file_count and not line_count and not commit_count:
        print("WARNING: nothing was scanned, check the base ref.", file=sys.stderr)
    print_findings(findings)
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
