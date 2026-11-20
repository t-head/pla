# SPDX-License-Identifier: Apache-2.0
"""Summarize only completed, validated cycle shards; never count partial output as a pass."""
import argparse
from collections import defaultdict
import glob
import json
import math
from pathlib import Path


def geometric_mean(values):
    values = list(values)
    return math.exp(sum(math.log(x) for x in values) / len(values))


def load_rows(patterns):
    rows = {}
    for path in sorted({p for pattern in patterns for p in glob.glob(pattern)}):
        records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        if not records or records[-1].get("type") != "summary" or records[-1].get("status") != "PASS":
            raise ValueError(f"incomplete or failed shard: {path}")
        cases = [r for r in records if r.get("type") == "case"]
        if len(cases) != records[-1]["selected_cases"]:
            raise ValueError(f"row count mismatch: {path}")
        for row in cases:
            if row["status"] != "PASS":
                raise ValueError(f"unvalidated case: {row['key']}")
            if row["key"] in rows:
                raise ValueError(f"duplicate case: {row['key']}")
            for backend in row["backends"].values():
                if not isinstance(backend["validation"], dict):
                    raise ValueError(f"missing numerical validation: {row['key']}")
                if backend["timing"] == "graph" and backend["graph_validation"] != "PASS":
                    raise ValueError(f"graph was not validated: {row['key']}")
            rows[row["key"]] = row
    if not rows:
        raise ValueError("no completed shards matched")
    return rows


def summarize(rows):
    groups = defaultdict(list)
    for row in rows.values():
        c = row["case"]
        for tag in (c["mode"], f"{c['mode']}/T{c['T']}", f"{c['mode']}/B{c['B']}"):
            groups[tag].append(row)
    output = {}
    for tag, data in sorted(groups.items()):
        common = set.intersection(*(set(r["backends"]) for r in data))
        point = {"cases": len(data)}
        for backend in sorted(common):
            point[backend + "_round_us_geomean"] = geometric_mean(r["backends"][backend]["round_us"] for r in data)
            point[backend + "_max_output_l2"] = max(r["backends"][backend]["validation"]["max_output_relative_l2"] for r in data)
            point[backend + "_max_state_l2"] = max(r["backends"][backend]["validation"]["max_state_relative_l2"] for r in data)
        if {"pla", "triton"} <= common:
            ratios = [r["backends"]["pla"]["round_us"] / r["backends"]["triton"]["round_us"] for r in data]
            point["pla_over_triton_geomean"] = geometric_mean(ratios)
            point["pla_slower_over_5pct"] = sum(ratio > 1.05 for ratio in ratios)
        output[tag] = point
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("paths", nargs="+")
    p.add_argument("--expected", type=int, required=True)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    rows = load_rows(args.paths)
    result = {"cases": len(rows), "expected": args.expected,
              "complete": len(rows) == args.expected, "groups": summarize(rows),
              "measurement": "complete-cycle CUDA graph elapsed; not OPF Active Time"}
    text = json.dumps(result, indent=2)
    if args.output:
        with args.output.open("x") as f:
            f.write(text + "\n")
    print(text)
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
