#!/usr/bin/env python3
"""Compare full-vocabulary logits on exactly the same teacher-forced tokens.

Calibration selects a candidate; heldout reports it without changing the fit.
Thresholds are project diagnostics, not universal guarantees of model quality.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file

GATES = {"mean_kl_max": .15, "case_mean_kl_max": .30, "perplexity_ratio_max": 1.10,
         "case_perplexity_ratio_max": 1.25, "top1_agreement_min": .80}


def log_softmax(x):
    x = x.astype(np.float64)
    shifted = x - x.max(-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(-1, keepdims=True))


def read(folder):
    receipt = json.loads((folder / "receipt.json").read_text())
    for row in receipt["cases"]:
        path = folder / row["logitsFile"]
        if path.name != row["logitsFile"]: raise ValueError("Unsafe logits filename")
        if "logitsSHA256" in row and hashlib.sha256(path.read_bytes()).hexdigest() != row["logitsSHA256"]:
            raise ValueError("Logits checksum mismatch")
    return receipt


def compare(reference: Path, candidate: Path):
    ref, cand = read(reference), read(candidate)
    if not ref.get("conversionSHA256") or not cand.get("conversionSHA256"):
        raise ValueError("Paired qualification requires verified conversion provenance")
    if ref["suiteSHA256"] != cand["suiteSHA256"]: raise ValueError("Different token suites")
    if [r["sequence"] for r in ref["cases"]] != [r["sequence"] for r in cand["cases"]]:
        raise ValueError("Different token ids, boundaries, ordering, or split")
    rows = []
    for r, c in zip(ref["cases"], cand["cases"]):
        seq = r["sequence"]
        a = load_file(str(reference / r["logitsFile"]))["logits"]
        b = load_file(str(candidate / c["logitsFile"]))["logits"]
        target = np.array(seq["tokens"][seq["scoreStart"]:])
        if a.shape != b.shape or a.shape[0] != len(target) or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("Logits shapes or finite values differ")
        lp, lq = log_softmax(a), log_softmax(b)
        delta = lq[np.arange(len(target)), target] - lp[np.arange(len(target)), target]
        kl = (np.exp(lp) * (lp - lq)).sum(-1)
        top1 = (a.argmax(-1) == b.argmax(-1)).astype(float)
        rows.append(dict(id=seq["id"], language=seq["language"], task=seq["task"], split=seq["split"],
                         tokens=len(target), mean_kl=float(kl.mean()), p95_kl=float(np.quantile(kl, .95)),
                         top1_agreement=float(top1.mean()), delta_nll=float(-delta.mean()),
                         reference_nll=float(-lp[np.arange(len(target)), target].mean()),
                         candidate_nll=float(-lq[np.arange(len(target)), target].mean()),
                         perplexity_ratio=float(np.exp(-delta.mean())),
                         token_logprob_rmse=float(np.sqrt(np.mean(delta ** 2))),
                         p95_abs_token_logprob_delta=float(np.quantile(abs(delta), .95))))
    def aggregate(selected):
        n = sum(r["tokens"] for r in selected)
        result = {"cases": len(selected), "tokens": n}
        for key in ["mean_kl", "top1_agreement", "delta_nll", "reference_nll", "candidate_nll"]:
            result[key] = sum(r[key] * r["tokens"] for r in selected) / n
        result["perplexity_ratio"] = float(np.exp(result["delta_nll"]))
        result["worst_case_mean_kl"] = max(r["mean_kl"] for r in selected)
        result["worst_case_perplexity_ratio"] = max(r["perplexity_ratio"] for r in selected)
        result["passes_diagnostic_gates"] = (result["mean_kl"] <= GATES["mean_kl_max"] and
            result["worst_case_mean_kl"] <= GATES["case_mean_kl_max"] and
            result["perplexity_ratio"] <= GATES["perplexity_ratio_max"] and
            result["worst_case_perplexity_ratio"] <= GATES["case_perplexity_ratio_max"] and
            result["top1_agreement"] >= GATES["top1_agreement_min"])
        return result
    heldout = aggregate([r for r in rows if r["split"] == "heldout"])
    languages = {lang: aggregate([r for r in rows if r["split"] == "heldout" and r["language"] == lang])
                 for lang in sorted({r["language"] for r in rows if r["split"] == "heldout"})}
    return dict(schema_version=1, reference=str(reference), candidate=str(candidate),
                suite_sha256=ref["suiteSHA256"], gates=GATES, cases=rows,
                calibration=aggregate([r for r in rows if r["split"] == "calibration"]),
                heldout=heldout, by_language=languages,
                passes_all_heldout_gates=heldout["passes_diagnostic_gates"] and
                    all(row["passes_diagnostic_gates"] for row in languages.values()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = [compare(args.reference, candidate) for candidate in args.candidates]
    selected = min(reports, key=lambda r: r["calibration"]["mean_kl"])
    result = dict(reports=reports, selected_by_calibration=str(selected["candidate"]),
                  selected_passes_heldout=selected["passes_all_heldout_gates"])
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for report in reports: print(json.dumps({"candidate": report["candidate"], "heldout": report["heldout"]}))
