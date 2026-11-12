#!/usr/bin/env python3
# Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause
"""Portable near-collinear GDN accuracy reproducer for 810E/890P.

  python scripts/prefill/flashqla/repro_collinear_accuracy.py generate --output /tmp/gdn-fixtures
  python scripts/prefill/flashqla/repro_collinear_accuracy.py run --fixtures /tmp/gdn-fixtures --output /tmp/gdn-results
  python scripts/prefill/flashqla/repro_collinear_accuracy.py selftest

Generate once and reuse the same files on both boards. CPU seed=101 preserves
the original constructor's formulas/RNG order, but CPU RNG is NOT guaranteed
bitwise identical to the original H200-generated fixtures. The stress jointly
changes K and beta; it does not establish a K-only cause or full-model damage.
All references run on CPU in FP64; run imports PLA lazily and needs its built
extension. Run exit codes: 0 pass, 1 accuracy/empty-state failure, 2 exception.
"""

import argparse
import hashlib
import importlib
import json
import math
import platform
import subprocess
import sys
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F


TOLERANCE = .01


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    return value


def dump(path, value):
    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def make_case(lengths, hv, variant, seed=101):
    rng, tokens, hq, dim = torch.Generator(device="cpu").manual_seed(seed), sum(lengths), 2, 128
    def randn(*shape):
        return torch.randn(*shape, generator=rng, dtype=torch.float32)
    q = F.normalize(randn(tokens, hq, dim), dim=-1).bfloat16()
    k = randn(tokens, hq, dim)
    if variant == "collinear":
        k = randn(1, hq, dim) + .003 * k
    k = F.normalize(k, dim=-1).bfloat16()
    v = randn(tokens, hv, dim).bfloat16()
    g = F.logsigmoid(randn(tokens, hv)) * .0625
    beta = randn(tokens, hv).sigmoid()
    if variant == "collinear":
        beta = .85 + .15 * beta
    h0 = randn(len(lengths), hv, dim, dim) * .125
    return dict(q=q, k=k, v=v, g=g, beta=beta, h0=h0, lengths=list(lengths), scale=dim ** -.5)


def offsets(case):
    result = [0]
    for length in case["lengths"]:
        result.append(result[-1] + int(length))
    return result


def reference(case, chunked=False):
    """Independent token recurrence or unit-lower-triangular 64-token solve."""
    q, k, v, g, beta = (case[key].double() for key in ("q", "k", "v", "g", "beta"))
    heads = torch.arange(v.shape[1]) // (v.shape[1] // k.shape[1])
    state, out = case["h0"].double().clone(), torch.empty_like(v)
    for seq, (start, stop) in enumerate(zip(offsets(case), offsets(case)[1:])):
        s = state[seq]
        if not chunked:
            for t in range(start, stop):
                kt, qt = k[t, heads], q[t, heads]
                s.mul_(g[t].exp()[:, None, None])
                residual = (v[t] - (s @ kt.unsqueeze(-1)).squeeze(-1)) * beta[t, :, None]
                s.add_(residual.unsqueeze(-1) * kt.unsqueeze(-2))
                out[t] = (s @ qt.unsqueeze(-1)).squeeze(-1) * case["scale"]
            continue
        for left in range(start, stop, 64):
            right = min(left + 64, stop)
            query, key = (x[left:right, heads].transpose(0, 1) for x in (q, k))
            value, b = v[left:right].transpose(0, 1), beta[left:right].transpose(0, 1)
            prefix = g[left:right].transpose(0, 1).cumsum(-1)
            eye = torch.eye(right - left, dtype=torch.float64)
            relative = (prefix[:, :, None] - prefix[:, None, :]).masked_fill(~eye.cumsum(-2).bool(), -torch.inf).exp()
            gram = ((key @ key.transpose(-1, -2)) * relative * b[:, :, None]).tril(-1)
            decay, st = prefix.exp().unsqueeze(-1), s.transpose(-1, -2)
            rhs = b[:, :, None] * (value - (key @ st) * decay)
            residual = torch.linalg.solve_triangular(eye + gram, rhs, upper=False, unitriangular=True)
            result = (query @ st) * decay + ((query @ key.transpose(-1, -2)) * relative) @ residual
            out[left:right] = (result * case["scale"]).transpose(0, 1)
            end_decay = (prefix[:, -1:] - prefix).exp().unsqueeze(-1)
            s.mul_(prefix[:, -1].exp()[:, None, None])
            s.add_((residual * end_decay).transpose(-1, -2) @ key)
    return out, state


def tensor_identity(tensor):
    data = tensor.contiguous().view(torch.uint8).numpy().tobytes()
    return dict(shape=list(tensor.shape), dtype=str(tensor.dtype), sha256=hashlib.sha256(data).hexdigest())


def generate(args):
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(seed=args.seed, suite=args.suite, rng_device="cpu", torch_version=str(torch.__version__),
                    python=sys.version, byteorder=sys.byteorder, script_sha256=file_sha(__file__),
                    rng_warning="CPU RNG does not guarantee bitwise identity with original H200 fixtures",
                    state_layout="[B,H,V,K]", cases=[])
    configurations = [("collinear", [128]), ("normal", [128])]
    if args.suite == "full":
        configurations = [("collinear", [128]), ("collinear", [8192]),
                          ("collinear", [1, 65, 127, 0]), ("normal", [128])]
    for model, hv in (("flash-next", 6), ("max", 16)):
        for variant, lengths in configurations:
            case = make_case(lengths, hv, variant, args.seed)
            chunked = max(lengths) > 512
            o, h = reference(case, chunked=chunked)
            name = f"{model}-{variant}-" + "_".join(map(str, lengths))
            case.update(name=name, reference_o=o, reference_h=h,
                        reference_kind="FP64 chunk triangular64" if chunked else "FP64 token recurrence",
                        metadata=dict(model=model, variant=variant, hq=2, hv=hv, seed=args.seed, rng_device="cpu"))
            path = args.output / f"{name}.pt"
            torch.save(case, path)
            entry = dict(name=name, filename=path.name, bytes=path.stat().st_size, sha256=file_sha(path),
                         reference_kind=case["reference_kind"],
                         tensors={key: tensor_identity(value) for key, value in case.items() if isinstance(value, torch.Tensor)})
            manifest["cases"].append(entry)
            dump(args.output / "manifest.json", manifest)
            print(json.dumps({key: entry[key] for key in ("name", "sha256", "reference_kind")}), flush=True)
    return 0


def errors(actual, expected, axes):
    a, b = actual.double(), expected.double()
    finite = torch.isfinite(a).all(dim=axes) & torch.isfinite(b).all(dim=axes)
    rms, base = (a - b).square().mean(dim=axes).sqrt(), b.square().mean(dim=axes).sqrt()
    relative, maximum = rms / base.clamp_min(1e-8), (a - b).abs().amax(dim=axes)
    return relative.masked_fill(~finite, torch.nan), maximum.masked_fill(~finite, torch.nan), finite


def peak(values):
    flat = torch.as_tensor(values, dtype=torch.float64).flatten()
    finite = flat[torch.isfinite(flat)]
    return finite.max().item() if finite.numel() else None


def metrics(result, case):
    o, h = result
    if o.shape != case["reference_o"].shape or h.shape != case["reference_h"].shape:
        raise ValueError("Output/state shape mismatch")
    if o.dtype != torch.bfloat16 or h.dtype != torch.float32:
        raise ValueError("Expected BF16 output and FP32 state")
    output, absolute, ofinite = [], [], []
    for a, b in zip(o.split(case["lengths"]), case["reference_o"].split(case["lengths"])):
        r, m, f = errors(a, b, (0, 2)) if a.numel() else (torch.empty(0), torch.empty(0), torch.ones(0, dtype=torch.bool))
        output.append(r.tolist()); absolute.append(m.tolist()); ofinite.append(f.tolist())
    hr, hm, hf = errors(h, case["reference_h"], (2, 3))
    nonempty = torch.tensor([length > 0 for length in case["lengths"]])
    omax, hmax = peak([v for row in output for v in row]), peak(hr[nonempty])
    finite_output, finite_nonempty = all(all(row) for row in ofinite), bool(hf[nonempty].all())
    empty_unchanged = all(torch.equal(h[index], case["h0"][index]) for index in (~nonempty).nonzero().flatten().tolist())
    passed_nonempty = finite_output and finite_nonempty and omax is not None and hmax is not None and max(omax, hmax) <= TOLERANCE
    return dict(passed=passed_nonempty and empty_unchanged and bool(hf.all()), passed_nonempty=passed_nonempty,
                finite=finite_output and bool(hf.all()), finite_output=finite_output,
                finite_state_nonempty=finite_nonempty, finite_state_empty=bool(hf[~nonempty].all()),
                output_relative_rms_max=omax, state_relative_rms_nonempty_max=hmax,
                state_relative_rms_empty_max=peak(hr[~nonempty]),
                output_relative_rms_by_sequence_head=output, output_abs_max_by_sequence_head=absolute,
                output_finite_by_sequence_head=ofinite, state_relative_rms_by_sequence_head=hr.tolist(),
                state_abs_max_by_sequence_head=hm.tolist(), state_finite_by_sequence_head=hf.tolist(),
                empty_state_bitwise_unchanged=empty_unchanged, empty_sequences=(~nonempty).nonzero().flatten().tolist())


def forward(function, case, device, auto_cp):
    cu = offsets(case)
    result = function(*(case[key].to(device).unsqueeze(0).contiguous() for key in ("q", "k", "v", "g", "beta")),
                      scale=float(case["scale"]), initial_state=case["h0"].transpose(-1, -2).contiguous().to(device),
                      cu_seqlens=torch.tensor(cu, dtype=torch.int32, device=device), raw_cu_seqlens=tuple(cu),
                      output_final_state=True, auto_cp=auto_cp, normalize_qk=False, final_state_dtype=torch.float32)
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    return result[2].squeeze(0).detach().cpu(), result[4].transpose(-1, -2).contiguous().detach().cpu()


def git(repo, *args):
    try:
        result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    except OSError as error:
        return {"error": f"git unavailable: {error}"}
    return result.stdout.strip() if not result.returncode else {"error": result.stderr.strip()}


def identity(repo):
    roots = ("pla/prefill/flashqla", "src/prefill/flashqla")
    sources = {str(p.relative_to(repo)): file_sha(p) for root in roots for p in sorted((repo / root).rglob("*"))
               if p.is_file() and p.suffix in (".py", ".cc", ".cu", ".cuh", ".h", ".inc")}
    return dict(repo=str(repo), head=git(repo, "rev-parse", "HEAD"), status=git(repo, "status", "--short"),
                source_sha256=sources, diff=git(repo, "diff", "--", *roots), script_sha256=file_sha(__file__),
                torch_version=str(torch.__version__), torch_file=torch.__file__, python=sys.version,
                platform=platform.platform(), argv=sys.argv)


def validate_case(case):
    t, hv, dim = case["v"].shape
    hq = case["q"].shape[1]
    if dim != 128 or hq != 2 or hv not in (6, 16) or t != sum(case["lengths"]):
        raise ValueError("Expected packed D=128, Hq=2, Hv=6/16 fixture")
    shapes = dict(q=(t,hq,dim), k=(t,hq,dim), v=(t,hv,dim), g=(t,hv), beta=(t,hv),
                  h0=(len(case["lengths"]),hv,dim,dim), reference_o=(t,hv,dim), reference_h=(len(case["lengths"]),hv,dim,dim))
    for key, shape in shapes.items():
        tensor = case[key]
        if tuple(tensor.shape) != shape or tensor.device.type != "cpu" or not torch.isfinite(tensor).all():
            raise ValueError(f"Invalid shape/device/finiteness: {key}")
        dtype = torch.bfloat16 if key in ("q","k","v") else torch.float32
        if tensor.dtype not in ((torch.float32, torch.float64) if key.startswith("reference_") else (dtype,)):
            raise ValueError(f"Invalid dtype: {key}")
    if any(int(length) < 0 for length in case["lengths"]) or not t:
        raise ValueError("Expected nonnegative sequence lengths with nonempty total")


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    repo, ident, records = args.repo.resolve(), identity(args.repo.resolve()), []
    dump(args.output / "identity.json", ident)
    try:
        sys.path.insert(0, str(repo))
        ops = importlib.import_module("pla.prefill.flashqla.ops")
        extension = importlib.import_module("pla.prefill.flashqla._C")
        for module in (ops, extension):
            if not Path(module.__file__).resolve().is_relative_to(repo):
                raise RuntimeError(f"Imported PLA module outside --repo: {module.__file__}")
        torch.cuda.set_device(torch.device(args.device))
        ident.update(device=str(torch.cuda.get_device_properties(args.device)),
                     device_capability=list(torch.cuda.get_device_capability(args.device)),
                     torch_cuda_version=torch.version.cuda,
                     modules=[dict(name=m.__name__, path=str(Path(m.__file__).resolve()), sha256=file_sha(m.__file__)) for m in (ops, extension)])
    except Exception:
        ident["setup_exception"] = traceback.format_exc()
        dump(args.output / "identity.json", ident)
        dump(args.output / "summary.json", dict(exit_code=2, exception=ident["setup_exception"]))
        return 2
    dump(args.output / "identity.json", ident)
    def append(record, stream):
        records.append(record); stream.write(json.dumps(clean(record), allow_nan=False) + "\n"); stream.flush()
    with (args.output / "results.jsonl").open("x") as stream:
        files = sorted(args.fixtures.glob("*.pt")) if args.fixtures.is_dir() else [args.fixtures]
        if not files:
            append(dict(passed=False, exception="No .pt fixtures found"), stream)
        for path in files:
            try:
                payload = torch.load(path, map_location="cpu", weights_only=True)
                if not isinstance(payload, dict):
                    raise ValueError("Fixture must be a dictionary")
                cases = payload["cases"] if "cases" in payload else [payload]
                if not isinstance(cases, list) or not cases or any(not isinstance(case, dict) for case in cases):
                    raise ValueError("cases must be a nonempty list of dictionaries")
                digest = file_sha(path)
            except Exception:
                append(dict(fixture=str(path), passed=False, exception=traceback.format_exc()), stream); continue
            for index, case in enumerate(cases):
                for auto_cp in (False, True):
                    record = dict(name=case.get("name", path.stem), fixture=str(path), fixture_sha256=digest,
                                  case_index=index, auto_cp=auto_cp, actual_use_cp=False, planner_called=False,
                                  reference_kind=case.get("reference_kind"), metadata=case.get("metadata"),
                                  lengths=case.get("lengths"), tolerance=TOLERANCE, state_layout="VK adapted to legacy KV",
                                  normalize_qk=False)
                    original = ops.calc_cp_seqs
                    def observed(*a, **kw):
                        plan = original(*a, **kw)
                        record.update(planner_called=True, actual_use_cp=bool(plan[0]),
                                      cp_segments=plan[1].numel()-1 if plan[1] is not None else None)
                        return plan
                    try:
                        validate_case(case); ops.calc_cp_seqs = observed
                        result = forward(ops.chunk_gated_delta_rule_fwd, case, args.device, auto_cp)
                        record.update(metrics(result, case))
                        if args.save_outputs:
                            artifact = f"{path.stem}-{index}-cp{int(auto_cp)}.pt"
                            torch.save(dict(o=result[0], h=result[1]), args.output / artifact)
                            record["output_artifact"] = artifact
                    except Exception:
                        record.update(passed=False, exception=traceback.format_exc())
                    finally:
                        ops.calc_cp_seqs = original
                    append(record, stream)
                    print(json.dumps({key: clean(record.get(key)) for key in ("name", "auto_cp", "actual_use_cp", "passed", "output_relative_rms_max", "state_relative_rms_nonempty_max", "empty_state_bitwise_unchanged")}), flush=True)
    exceptions = sum("exception" in r for r in records)
    failed = sum(not r["passed"] for r in records)
    code = 2 if exceptions else 1 if failed else 0
    dump(args.output / "summary.json", dict(exit_code=code, runs=len(records), passed=len(records)-failed,
         accuracy_failures=failed-exceptions, exceptions=exceptions,
         nonempty_failures=sum(r.get("passed_nonempty") is False for r in records),
         empty_state_failures=sum(r.get("empty_state_bitwise_unchanged") is False for r in records),
         actual_cp_runs=sum(r.get("actual_use_cp", False) for r in records),
         interpretation="Per-sequence/head 1% RMS is an operator diagnostic; empty state is reported separately, not as collinear error"))
    return code


def selftest():
    for hv, variant in ((6, "collinear"), (16, "normal")):
        case = make_case([1,65,127,0], hv, variant)
        token, chunk = reference(case), reference(case, chunked=True)
        for a, b in zip(token, chunk):
            torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-10)
        case.update(reference_o=token[0], reference_h=token[1])
        def fake(q,k,v,g,beta,**kwargs):
            assert torch.equal(kwargs["initial_state"], case["h0"].transpose(-1,-2).contiguous())
            assert kwargs["normalize_qk"] is False and tuple(kwargs["raw_cu_seqlens"]) == tuple(offsets(case))
            return None,None,token[0].bfloat16().unsqueeze(0),None,token[1].float().transpose(-1,-2).contiguous()
        result = forward(fake, case, "cpu", False)
        assert metrics(result, case)["passed"]
        bad = result[1].clone(); bad[-1].fill_(torch.nan)
        measured = metrics((result[0],bad), case)
        assert not measured["passed"] and measured["passed_nonempty"] and not measured["finite_state_empty"]
        json.dumps(clean(measured), allow_nan=False)
    print("PASS: FP64 token/chunk, GVA3/8, legacy KV/VK adapter, empty/nonfinite separation, strict JSON; CPU only")
    return 0


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--threads", type=int, default=8)
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate"); gen.add_argument("--output", type=Path, required=True)
    gen.add_argument("--suite", choices=("quick","full"), default="quick"); gen.add_argument("--seed", type=int, default=101)
    execute = sub.add_parser("run"); execute.add_argument("--fixtures", type=Path, required=True)
    execute.add_argument("--output", type=Path, required=True); execute.add_argument("--device", default="cuda:0")
    default_repo = next((parent for parent in Path(__file__).resolve().parents
                         if (parent / "pla/prefill").is_dir()), Path.cwd())
    execute.add_argument("--repo", type=Path, default=default_repo)
    execute.add_argument("--save-outputs", action="store_true"); sub.add_parser("selftest")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    try:
        return generate(args) if args.command == "generate" else run(args) if args.command == "run" else selftest()
    except Exception:
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
