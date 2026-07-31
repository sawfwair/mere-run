#!/usr/bin/env python3
"""Gate an Inkling LoRA on held-out behavioral paraphrases."""

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
from typing import Optional


def load_cases(path: pathlib.Path) -> list[dict]:
    cases = []
    for raw_line in path.read_text().splitlines():
        if raw_line.strip():
            cases.append(json.loads(raw_line))
    return cases


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_case(
    cli: pathlib.Path,
    model_root: pathlib.Path,
    case: dict,
    adapter: Optional[pathlib.Path],
    reasoning_effort: float,
    max_tokens: int,
) -> dict:
    messages = case["messages"]
    system = next(message["content"] for message in messages if message["role"] == "system")
    prompt = next(message["content"] for message in messages if message["role"] == "user")
    expected = next(message["content"] for message in messages if message["role"] == "assistant")
    command = [
        str(cli),
        "text",
        "chat",
        "--model",
        "text-chat-inkling-small",
        "--model-root",
        str(model_root),
        "--system",
        system,
        "--prompt",
        prompt,
        "--reasoning-effort",
        str(reasoning_effort),
        "--temperature",
        "0",
        "--top-p",
        "1",
        "--max-tokens",
        str(max_tokens),
        "--quiet",
    ]
    if adapter is not None:
        command.extend(["--lora", str(adapter)])
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{case['id']} failed with exit {completed.returncode}: {completed.stderr.strip()}"
        )
    observed = completed.stdout.strip()
    return {
        "id": case["id"],
        "expected": expected,
        "observed": observed,
        "exact": observed == expected,
    }


def main() -> int:
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=pathlib.Path, required=True)
    parser.add_argument("--model-root", type=pathlib.Path, required=True)
    parser.add_argument(
        "--eval",
        type=pathlib.Path,
        default=repo_root / "Tests/MereRunCLITests/Fixtures/Inkling/receptivity-eval.jsonl",
    )
    parser.add_argument("--cli", type=pathlib.Path, default=repo_root / ".build/debug/mere.run")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--reasoning-effort", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--minimum-adapter-score", type=int, default=4)
    parser.add_argument("--minimum-improvement", type=int, default=3)
    args = parser.parse_args()

    cases = load_cases(args.eval.resolve())
    if not cases:
        raise RuntimeError("evaluation dataset is empty")
    if not args.cli.is_file():
        raise RuntimeError(f"CLI binary not found: {args.cli}")
    if not args.adapter.is_file():
        raise RuntimeError(f"adapter not found: {args.adapter}")
    if not args.model_root.is_dir():
        raise RuntimeError(f"model root not found: {args.model_root}")

    base = []
    for case in cases:
        result = run_case(
            args.cli,
            args.model_root,
            case,
            None,
            args.reasoning_effort,
            args.max_tokens,
        )
        base.append(result)
        print(
            f"[inkling-receptivity] base {result['id']} exact={result['exact']}",
            file=sys.stderr,
            flush=True,
        )
    adapted = []
    for case in cases:
        result = run_case(
            args.cli,
            args.model_root,
            case,
            args.adapter.resolve(),
            args.reasoning_effort,
            args.max_tokens,
        )
        adapted.append(result)
        print(
            f"[inkling-receptivity] adapter {result['id']} exact={result['exact']}",
            file=sys.stderr,
            flush=True,
        )
    base_score = sum(item["exact"] for item in base)
    adapter_score = sum(item["exact"] for item in adapted)
    passed = (
        adapter_score >= args.minimum_adapter_score
        and adapter_score - base_score >= args.minimum_improvement
    )
    report = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "model": "text-chat-inkling-small",
        "model_root": str(args.model_root.resolve()),
        "model_config_sha256": sha256(args.model_root / "config.json"),
        "adapter": str(args.adapter.resolve()),
        "adapter_sha256": sha256(args.adapter),
        "cli": str(args.cli.resolve()),
        "cli_sha256": sha256(args.cli),
        "eval_dataset": str(args.eval.resolve()),
        "reasoning_effort": args.reasoning_effort,
        "max_tokens": args.max_tokens,
        "case_count": len(cases),
        "base_exact": base_score,
        "adapter_exact": adapter_score,
        "improvement": adapter_score - base_score,
        "required_adapter_exact": args.minimum_adapter_score,
        "required_improvement": args.minimum_improvement,
        "base_cases": base,
        "adapter_cases": adapted,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(args.output.resolve())
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
