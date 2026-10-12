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

"""Render cuda_free-vs-master comparison sections into both perf reports.

The 2026-09-12 experiment: on the same ZW-M890P host, same container, same
measurement methodology, the exact case sets that showed PLA losses on
master (51aefe7) were re-measured against a cuda_free (343274e) build, to
answer whether those losses are a cuda_free-branch regression.

Inputs (all produced on the measurement host):
  * reports/perf_cudafree_recheck_prefill.csv — 261 prefill cases
    (237 previously-slow + 24 winning controls), master-recheck and
    cuda_free speedups side by side.
  * reports/perf_cudafree_recheck_decode.csv — 139 decode (shape, kernel)
    pairs, original-worst / master-recheck / cuda_free ratios.
  * master + cuda_free curated-37 JSONs for the standard prefill set.

Usage:
    python3 scripts/prefill/generate_cudafree_compare_section.py --date 2026-09-12
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SUBTITLE = '<p class="subtitle">'

PREFILL_H2 = "cuda_free branch comparison: flashkdapro regresses across the board vs master ({date})"
DECODE_H2 = "cuda_free branch comparison: lagging decode cases reproduce consistently ({date})"


def _stats(values):
    return statistics.median(values), min(values), max(values)


def _fmt_x(v, nd=3):
    return f"{v:.{nd}f}x"


def load_prefill_rows(path):
    return list(csv.DictReader(path.open()))


def load_decode_rows(path):
    return list(csv.DictReader(path.open()))


def load_curated(path):
    return json.load(path.open())["records"]


def curated_key(r):
    if r["family"] == "kda":
        return ("kda", r["tp"], r["num_heads"], r["batch"], r["total_tokens"])
    return ("gdn", r["batch"], r["seq_len"], r["num_k_heads"], r["num_v_heads"])


def build_prefill_section(rows, master_cur, cf_cur, date):
    slow = [r for r in rows if r["group"] == "slow"]
    ctrl = [r for r in rows if r["group"] == "control"]

    def col(recs, name):
        return [float(r[name]) for r in recs]

    pla_ratio = [
        float(r["cuda_free_recheck_pla_ms"]) / float(r["master_recheck_pla_ms"])
        for r in rows
    ]
    tri_ratio = [
        float(r["triton_ms"]) / (float(r["master_recheck_pla_ms"]) * float(r["master_recheck_speedup"]))
        for r in rows
    ]
    # the triton_ms column comes from the cuda_free re-test; master's triton time is recovered as pla_ms*speedup
    pla_med = statistics.median(pla_ratio)
    tri_med = statistics.median(tri_ratio)

    s_m, s_mi, s_ma = _stats(col(slow, "master_recheck_speedup"))
    s_c, s_ci, s_ca = _stats(col(slow, "cuda_free_recheck_speedup"))
    c_m, c_mi, c_ma = _stats(col(ctrl, "master_recheck_speedup"))
    c_c, c_ci, c_ca = _stats(col(ctrl, "cuda_free_recheck_speedup"))

    mm = {curated_key(r): r for r in master_cur if "speedup" in r}
    cc = {curated_key(r): r for r in cf_cur if "speedup" in r}
    common = sorted(set(mm) & set(cc))
    fam_rows = []
    for fam, label in (("kda", "KDA (flashkdapro)"), ("gdn", "GDN (flashqla)")):
        ks = [k for k in common if k[0] == fam]
        if not ks:
            continue
        ms = [mm[k]["speedup"] for k in ks]
        cs = [cc[k]["speedup"] for k in ks]
        tot_m = sum(mm[k]["pla_ms"] for k in ks)
        tot_c = sum(cc[k]["pla_ms"] for k in ks)
        fam_rows.append((label, len(ks), ms, cs, tot_c / tot_m))

    kda_detail = []
    for k in [k for k in common if k[0] == "kda"]:
        r, s = mm[k], cc[k]
        kda_detail.append((r, s))

    parts = []
    parts.append("<section>")
    parts.append(f"<h2>{PREFILL_H2.format(date=date)}</h2>")
    parts.append(
        "<p>To answer \"are the lagging cases a regression of the cuda_free branch\": on the same ZW-M890P, "
        "in the same container, builds of master@51aefe7 and cuda_free@343274e were used in turn "
        "to re-test the <strong>exact same</strong> case set with exactly the same methodology "
        "(prefill: 8 blocks x 10 alternating block timings; decode: iters=50/warmup=10). "
        "The cuda_free build uses its required actlize@v1.0.0 (12965118). "
        f"The median Triton baseline drift between the two sides on the re-test set is <strong>{tri_med:.3f}x</strong>, "
        "so machine and environment factors are ruled out; both rounds passed every accuracy gate, and the difference is purely performance.</p>"
    )
    parts.append(
        '<div class="summary-grid">\n'
        f'  <div class="metric">flashkdapro PLA time (cf/master)<span class="value loss">{pla_med:.2f}x</span></div>\n'
        f'  <div class="metric">Lagging-group speedup (master → cf)<span class="value loss">{s_m:.3f} → {s_c:.3f}</span></div>\n'
        f'  <div class="metric">Control-group speedup (master → cf)<span class="value loss">{c_m:.3f} → {c_c:.3f}</span></div>\n'
        f'  <div class="metric">Triton baseline drift<span class="value">{tri_med:.3f}x</span></div>\n'
        "</div>"
    )
    parts.append(f"<h3>Re-test set ({len(rows)} cases: {len(slow)} originally lagging + {len(ctrl)} control)</h3>")
    parts.append(
        "<table><tr><th>Group</th><th>n</th><th>master median [range]</th>"
        "<th>cuda_free median [range]</th><th>Conclusion</th></tr>"
        f"<tr><td>Originally lagging</td><td>{len(slow)}</td>"
        f"<td>{s_m:.3f} [{s_mi:.3f}, {s_ma:.3f}]</td>"
        f'<td class="loss">{s_c:.3f} [{s_ci:.3f}, {s_ca:.3f}]</td>'
        f"<td>all slower still on cf</td></tr>"
        f"<tr><td>Control (master leads)</td><td>{len(ctrl)}</td>"
        f"<td>{c_m:.3f} [{c_mi:.3f}, {c_ma:.3f}]</td>"
        f'<td class="loss">{c_c:.3f} [{c_ci:.3f}, {c_ca:.3f}]</td>'
        f"<td>24/24 turn to lagging on cf</td></tr></table>"
    )
    parts.append("<h3>curated 37 standard set comparison</h3>")
    fam_html = ["<table><tr><th>family</th><th>n</th><th>master median speedup</th>"
                "<th>cuda_free median speedup</th><th>PLA total time cf/master</th></tr>"]
    for label, n, ms, cs, ratio in fam_rows:
        cls = ' class="loss"' if statistics.median(cs) < statistics.median(ms) else ""
        fam_html.append(
            f"<tr><td>{label}</td><td>{n}</td>"
            f"<td>{statistics.median(ms):.3f} [{min(ms):.3f}, {max(ms):.3f}]</td>"
            f"<td{cls}>{statistics.median(cs):.3f} [{min(cs):.3f}, {max(cs):.3f}]</td>"
            f"<td>{ratio:.2f}x</td></tr>"
        )
    fam_html.append("</table>")
    parts.append("".join(fam_html))
    parts.append("<h3>curated KDA 24 case details</h3>")
    det = ["<table><tr><th>tp</th><th>H</th><th>B</th><th>T</th>"
           "<th>master speedup</th><th>cuda_free speedup</th>"
           "<th>PLA ms (master → cf)</th></tr>"]
    for r, s in kda_detail:
        det.append(
            f"<tr><td>{r['tp']}</td><td>{r['num_heads']}</td><td>{r['batch']}</td>"
            f"<td>{r['total_tokens']}</td><td>{r['speedup']:.3f}</td>"
            f'<td class="loss">{s["speedup"]:.3f}</td>'
            f"<td>{r['pla_ms']:.3f} → {s['pla_ms']:.3f}</td></tr>"
        )
    det.append("</table>")
    parts.append("".join(det))
    parts.append(
        '<p class="note"><strong>Conclusion:</strong> the lagging list on master (224 confirmed lagging cases) '
        "was <strong>not</strong> introduced by cuda_free -- master itself already loses on the "
        "<code>N*H&lt;40</code> baseline branch. But the control experiment exposed a more serious, independent problem: "
        "<strong>flashkdapro on the cuda_free branch regresses 3–4.5x across the board vs master</strong> "
        "(261/261 re-tested cases without exception, covering all BV16/BV32/nows/baseline dispatch branches, "
        "and even cases that lead by 1.6–2.2x on master lag Triton by more than 2x on cuda_free); "
        "flashqla is only slightly slower (about 11%), and decode is unchanged (see the section of the same name in the decode report). "
        "A torch.profiler breakdown (B=1/H=96/T=4096, nows path, PLA total 6.77ms/call) shows the regression "
        "concentrated in two CuTe device kernels: K2 recurrence-nows 5.24ms/call (77.5%) and "
        "K1 prepare 1.42ms/call (21.0%); the host side (aten/cudaLaunch) is under 2%, "
        "ruling out dispatch branch changes and host instrumentation; "
        "the regression matches cuda_free migrating the prefill CuTe traits from cutlass3 to actlize "
        "and the changes to the three flashkdapro kernel files; kernel owners need to narrow it down to the specific operator. "
        "Per-case comparison data is in <code>reports/perf_cudafree_recheck_prefill.csv</code>. "
        f"Raw curated JSON: <code>reports/prefill_bench_cudafree_{date.replace('-', '')}.json</code> (cuda_free side).</p>"
    )
    parts.append("</section>")
    return "\n".join(parts)


def build_decode_section(rows, date):
    conf = [r for r in rows if float(r["master_recheck_ratio"]) < 1.0]
    flip = [r for r in rows if float(r["master_recheck_ratio"]) >= 1.0]

    def col(recs, name):
        return [float(r[name]) for r in recs]

    m_m, m_mi, m_ma = _stats(col(conf, "master_recheck_ratio"))
    c_m, c_mi, c_ma = _stats(col(conf, "cuda_free_recheck_ratio"))
    deltas = [
        float(r["cuda_free_recheck_ratio"]) - float(r["master_recheck_ratio"])
        for r in conf
    ]
    d_med = statistics.median(deltas)
    big = [d for d in deltas if abs(d) > 0.02]
    both = sum(1 for r in conf if float(r["cuda_free_recheck_ratio"]) < 1.0)
    flip_win = sum(1 for r in flip if float(r["cuda_free_recheck_ratio"]) >= 1.0)
    f_m = statistics.median(col(flip, "cuda_free_recheck_ratio")) if flip else float("nan")

    return "\n".join([
        "<section>",
        f"<h2>{DECODE_H2.format(date=date)}</h2>",
        "<p>The 139 deduplicated (shape, kernel) pairs where Triton won in the 2026-09-12 sweep were "
        "re-tested on the same ZW-M890P with the same method (iters=50 / warmup=10 / exclusive machine) on both the "
        "master@51aefe7 and cuda_free@343274e builds. Per-pair data is in "
        "<code>reports/perf_cudafree_recheck_decode.csv</code>.</p>",
        "<table><tr><th>Group</th><th>n</th><th>master re-test median [range]</th>"
        "<th>cuda_free re-test median [range]</th><th>cf − master</th></tr>"
        f"<tr><td>Stably lagging (confirmed in two rounds)</td><td>{len(conf)}</td>"
        f"<td>{m_m:.3f} [{m_mi:.3f}, {m_ma:.3f}]</td>"
        f"<td>{c_m:.3f} [{c_mi:.3f}, {c_ma:.3f}]</td>"
        f"<td>median {d_med:+.3f}, {len(big)} with |Δ|&gt;0.02</td></tr>"
        f"<tr><td>First-round transient (flipped back to leading on re-test)</td><td>{len(flip)}</td>"
        f"<td>≥1.000</td><td>{flip_win}/{len(flip)} leading, median {f_m:.3f}</td>"
        f"<td>—</td></tr></table>",
        f"<p><strong>Conclusion:</strong> the {len(conf)} stably lagging shapes "
        f"<strong>reproduce {both}/{len(conf)} consistently</strong> on cuda_free, with every two-round ratio difference inside the noise band -- "
        "this slight lag (packed large-batch GDN, 0.944–0.991x) is behavior <strong>shared</strong> by both branches, "
        "<strong>not</strong> a cuda_free regression; cuda_free also introduces no new lag in decode. "
        "(The prefill-side conclusion is the opposite, see the section of the same name in the prefill report: flashkdapro regresses severely on cuda_free.)</p>",
        "</section>",
    ])


def upsert(page, section, h2_marker):
    if h2_marker in page:
        start = page.rindex("<section>", 0, page.index(h2_marker))
        end = page.index("</section>", start) + len("</section>")
        return page[:start] + section + page[end:]
    anchor_end = page.index("</p>", page.index(SUBTITLE)) + len("</p>")
    return page[:anchor_end] + "\n\n" + section + "\n" + page[anchor_end:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=str, required=True)
    parser.add_argument("--prefill-csv", type=Path,
                        default=REPO_ROOT / "reports" / "perf_cudafree_recheck_prefill.csv")
    parser.add_argument("--decode-csv", type=Path,
                        default=REPO_ROOT / "reports" / "perf_cudafree_recheck_decode.csv")
    parser.add_argument("--master-curated", type=Path,
                        default=REPO_ROOT / "reports" / "prefill_bench_20260911.json")
    parser.add_argument("--cf-curated", type=Path,
                        default=REPO_ROOT / "reports" / "prefill_bench_cudafree_20260912.json")
    parser.add_argument("--prefill-html", type=Path,
                        default=REPO_ROOT / "reports" / "perf_report_prefill.html")
    parser.add_argument("--decode-html", type=Path,
                        default=REPO_ROOT / "reports" / "perf_report_890p.html")
    args = parser.parse_args()

    prefill_rows = load_prefill_rows(args.prefill_csv)
    decode_rows = load_decode_rows(args.decode_csv)
    master_cur = load_curated(args.master_curated)
    cf_cur = load_curated(args.cf_curated)

    prefill_section = build_prefill_section(prefill_rows, master_cur, cf_cur, args.date)
    decode_section = build_decode_section(decode_rows, args.date)

    page = args.prefill_html.read_text()
    page = upsert(page, prefill_section, PREFILL_H2.format(date=args.date))
    args.prefill_html.write_text(page)
    print(f"updated {args.prefill_html}")

    page = args.decode_html.read_text()
    page = upsert(page, decode_section, DECODE_H2.format(date=args.date))
    args.decode_html.write_text(page)
    print(f"updated {args.decode_html}")


if __name__ == "__main__":
    main()
