#!/usr/bin/env python3
"""Compare pinned Laya CPU reference predictions with a source-built native CLI.

Requires local checkpoint assets. This script never downloads a model.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import torch
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--native-bin", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip()
    if revision != "573e5b62696ba441230cd6be71d593331b5d23af":
        raise ValueError("The Laya SDK checkout is not at the pinned revision")
    sys.path.insert(0, str(args.upstream))
    import laya
    from laya.common import build_sequence, collate_items, QTYPES
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    binary_sha256 = hashlib.sha256(args.native_bin.read_bytes()).hexdigest()
    report = {"sdk_revision": revision, "native_executable": str(args.native_bin),
              "native_binary_sha256": binary_sha256, "cases": []}
    core_questions = [
        {"id": "department", "type": "choice", "instructions": "Which department handles this request?", "criteria": ["billing", "technical support", "sales"]},
        {"id": "urgency", "type": "score", "instructions": "Rate urgency.", "criteria": ["routine", "soon", "immediate"]},
        {"id": "refund", "type": "noul", "instructions": "The customer requests a refund."},
    ]
    cases = [
        ("english", "I was charged twice. Please refund the duplicate subscription charge.", core_questions),
        ("multilingual", "ខ្ញុំត្រូវបានគិតប្រាក់ពីរដង។ Quiero un reembolso. मुझे धनवापसी चाहिए। 🌍", core_questions),
        ("single", "Everything is ready.", [{"id": "only", "type": "choice", "instructions": "Pick the only label.", "criteria": ["ready"]}]),
        ("calibration", "Choose category seven. [MASK] <mask>", [{"id": "category", "type": "choice", "instructions": "Choose the category.", "criteria": [f"category {n}" for n in range(12)]}]),
        ("truncation", "Status is pending. " * 900, core_questions),
    ]
    for folder in ["", "multilingual", "typed-decisions"]:
        root = args.checkpoints / folder
        name = folder or "english"
        agent = laya.load(str(root), device="cpu")
        for case_name, state, questions in cases:
            request = {"state": state, "questions": questions}
            input_path = args.output / f"{name}-{case_name}-request.json"
            input_path.write_text(json.dumps(request, ensure_ascii=False))
            qdefs = {}
            items = []
            token_fixtures = []
            for question in questions:
                qid = question["id"]
                qdef = {k: v for k, v in question.items() if k != "id"}
                qdefs[qid] = qdef
                internal = agent._to_internal(qdef)
                ids, markers = build_sequence(agent.tok, state, internal, agent.cfg["max_len"], agent.cfg["head_max_len"])
                items.append({"ids": ids, "markers": markers, "qtype": QTYPES[question["type"]]})
                token_fixtures.append({"question": question, "state": state, "ids": ids, "markers": markers})
            (args.output / f"{name}-{case_name}-tokens.json").write_text(json.dumps(token_fixtures, ensure_ascii=False))
            reference = agent.predict(state, qdefs)
            start = time.monotonic()
            proc = subprocess.run([str(args.native_bin), "text", "decide", "--model", str(root), "--input", str(input_path)], capture_output=True, text=True, timeout=180)
            if proc.returncode:
                raise RuntimeError(proc.stderr)
            native = json.loads(proc.stdout)
            wall = time.monotonic() - start
            (args.output / f"{name}-{case_name}-native.json").write_text(json.dumps(native, indent=2, ensure_ascii=False))
            (args.output / f"{name}-{case_name}-reference.json").write_text(json.dumps(reference, indent=2, ensure_ascii=False))
            error = 0.0
            for question in questions:
                qid = question["id"]
                actual, expected = native["answers"][qid], reference["answers"][qid]
                if question["type"] == "choice":
                    assert actual["choice"] == expected["choice"], (name, case_name, qid)
                for field in ["score", "noul", "confidence"]:
                    if field in expected:
                        error = max(error, abs(actual[field] - expected[field]))
                for key, value in expected.get("probabilities", {}).items():
                    error = max(error, abs(actual["probabilities"][key] - value))
                error = max(error, abs(actual["actProbability"] - expected["action"]["act_probability"]))
            assert native["inputTokens"] == reference["usage"]["input_tokens"], (name, case_name, "token count")
            entry = {"checkpoint": name, "case": case_name, "maximum_output_error": error, "cold_cli_seconds": wall,
                     "input_tokens": native["inputTokens"], "passed": error <= 0.0002}
            report["cases"].append(entry)
            print(json.dumps(entry), flush=True)
            (args.output / "report.json").write_text(json.dumps(report, indent=2))
            if error > 0.0002:
                raise AssertionError(entry)
            if case_name == "english":
                batch = collate_items([items], agent.tok.pad_token_id)
                with torch.no_grad():
                    logits, actions = agent.model(batch["input_ids"], batch["attention_mask"], batch["marker_pos"], batch["marker_mask"], batch["qtype"])
                save_file({"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"],
                           "marker_positions": batch["marker_pos"], "marker_mask": batch["marker_mask"],
                           "question_types": batch["qtype"], "logits": logits, "action_logits": actions}, args.output / f"{name}-real-reference.safetensors")
        del agent
    if hashlib.sha256(args.native_bin.read_bytes()).hexdigest() != binary_sha256:
        raise RuntimeError("The native binary changed during qualification; rerun against a stable build")
    report["passed"] = all(row["passed"] for row in report["cases"])
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
