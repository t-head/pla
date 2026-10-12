# Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Unified output protocol of the PLA test scripts (parsed per sub-case by xtest's PLAResultParser).

The ACTest-side ``acompute.PLAResultParser`` reads neither JSON nor exit-code details; it only scans
stdout line by line, splitting one run into sub-cases by the markers below and turning each into an ES record and a JUnit
fragment. Any test script that wants to plug into xtest must print these markers verbatim, otherwise the whole stdout degrades
into "one record judged by the exit code", leaving a single row in the CI table with no way to locate the specific shape:

    Kernel type: <kernel>              <- sub-case name prefix, applied to every following case
    Case file: <caselist path>          <- same as above; printed once here before the cases so the parser
                                          can put the caselist name into the sub-case names (the one in
                                          SUMMARY is for closing and has no effect on cases already printed)
    *--- start test case <i>/<N> (<label>) ---*
    <any info lines; lines starting with [FAIL] or containing Max diff go into the ES detail field>
    --- case pass!!! ---   or   --- case fail!!! ---
    ...
    ============================================================
    SUMMARY
    ============================================================
    Total cases: N
    Passed: X
    Failed: Y
    Case file: <caselist path>

One important parser behavior: if a start marker is followed by neither a pass nor a fail marker (the process was
killed or the kernel crashed), that sub-case is judged FAILED and the crash reason is extracted from the info lines. So
every case must print start/marker as a pair; an explicit fail is better than a silent skip.
"""
import re
import traceback

_LABEL_BAD = re.compile(r"[()\r\n]")

# Error signatures judged as "the device is no longer usable": once the CUDA context is polluted by an illegal access, every
# following case fails in a chain; continuing only floods the screen with meaningless FAILs and hits the timeout, so aborting is more informative.
_FATAL_HINTS = (
    "illegal memory access",
    "invalid page",
    "umd error",
    "device-side assert",
    "cuda error",
    "out of memory",
)


def sanitize_label(label):
    """The label is put into the sub-case name by the parser; strip brackets and newlines so the start marker is not broken."""
    return _LABEL_BAD.sub("_", str(label)).strip()


class CaseReporter:
    """Sub-case counter that prints according to the PLAResultParser protocol.

    Typical usage::

        rep = CaseReporter("kda_decode_mega", caselist_path, total=len(cases))
        rep.header()
        for c in cases:
            rep.start(f"H{c.H}_B{c.B}")
            rep.info(f"batch_size:{c.B}")
            with rep.guard():          # exceptions/crashes become case fail automatically
                ok, detail = run(c)
                rep.ok() if ok else rep.fail(detail)
        return rep.summary()
    """

    def __init__(self, kernel_type, case_file, total=0):
        self.kernel_type = kernel_type
        self.case_file = case_file
        self.total = total
        self.index = 0
        self.passed = 0
        self.failed = 0
        self.aborted = False
        self._open = False

    # ---------------- output ----------------
    def header(self):
        print(f"Kernel type: {self.kernel_type}")
        print(f"Case file: {self.case_file}")
        print(f"Planned cases: {self.total}", flush=True)

    def start(self, label=""):
        self.index += 1
        label = sanitize_label(label)
        suffix = f" ({label})" if label else ""
        total = self.total or self.index
        print(f"*--- start test case {self.index}/{total}{suffix} ---*", flush=True)
        self._open = True

    def info(self, text):
        print(text, flush=True)

    def ok(self, note=""):
        if note:
            print(note)
        self.passed += 1
        self._open = False
        print("--- case pass!!! ---", flush=True)

    def fail(self, detail=""):
        # detail uses the [FAIL] prefix: the parser's _extract_fail_detail only picks lines starting with [FAIL] or
        # containing Max diff; other lines never reach the ES detail field.
        if detail:
            print(f"[FAIL] {detail}")
        self.failed += 1
        self._open = False
        print("--- case fail!!! ---", flush=True)

    def summary(self):
        """Print the SUMMARY section and return the process exit code (0 all passed / 1 failures / 2 aborted)."""
        if self._open:
            # Catch the "started but forgot to close" branch so the parser is not left with an incomplete case
            self.fail("case ended without a verdict (script bug)")
        print()
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total cases: {self.passed + self.failed}")
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        print(f"Case file: {self.case_file}")
        if self.aborted:
            print("ABORTED: device is no longer usable, remaining cases were not run")
            return 2
        return 1 if self.failed else 0

    # ---------------- exception fallback ----------------
    def catch(self, exc):
        """Record an exception as a failure of the current case; set aborted if the device is judged dead."""
        text = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        self.fail(text[:300])
        if any(h in text.lower() for h in _FATAL_HINTS):
            self.aborted = True

    def guard(self):
        """``with rep.guard():`` -- exceptions inside the block become case fail instead of crashing the whole script."""
        return _Guard(self)


class _Guard:
    def __init__(self, reporter):
        self._rep = reporter

    def __enter__(self):
        return self._rep

    def __exit__(self, exc_type, exc, tb):
        if exc is None:
            return False
        if isinstance(exc, KeyboardInterrupt):
            return False
        self._rep.catch(exc)
        return True  # recorded; swallow the exception and continue with the next case
