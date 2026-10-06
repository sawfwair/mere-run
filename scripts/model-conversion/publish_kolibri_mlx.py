#!/usr/bin/env python3
"""Publish the two measured Kolibri artifacts, verifying Hub hashes before public release.

Requires huggingface-hub==1.28.0. Run beside the remote checkpoint assets.
Large uploads resume through the SDK's upload cache. No local weight copy is made.
"""
import argparse
import hashlib
import json
from pathlib import Path

REPOSITORIES = {"mixed2": "Sawfwair/Kolibri-1-MLX-Mixed-2bit", "q8": "Sawfwair/Kolibri-1-MLX-8bit"}
SOURCE_REVISION = "7a8f290e7858825c3cf5e4c447ba68345de9f1d3"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def prepare(args):
    root = args.artifact
    evidence = json.loads(args.evidence.read_text())
    measured = evidence["artifacts"][args.profile]
    conversion_path = root / "KOLIBRI_CONVERSION.json"
    conversion = json.loads(conversion_path.read_text())
    if (digest(conversion_path.read_bytes()) != measured["conversion_sha256"] or
            conversion["source_revision"] != SOURCE_REVISION or
            conversion["source_repository"] != "Aleph-Alpha/Kolibri-1-BF16" or
            conversion["profile"] != args.profile):
        raise ValueError("Artifact does not match the measured pinned conversion")
    result = (next(row for row in evidence["q2_selection"]["reports"] if row["candidate"].endswith("/mixed2"))
              if args.profile == "mixed2" else evidence["controls"]["q8"])
    if not result["passes_all_heldout_gates"]:
        raise ValueError("Artifact did not pass overall and per-language diagnostic gates")
    if args.profile == "mixed2" and not evidence["q2_selection"]["selected_by_calibration"].endswith("/mixed2"):
        raise ValueError("Baseline Q2 was not selected by calibration")
    for name, receipt_key in [("config.json", "configSHA256"), ("model.safetensors.index.json", "indexSHA256")]:
        if digest((root / name).read_bytes()) != measured["scoring_receipt"][receipt_key]:
            raise ValueError(f"Measured metadata changed: {name}")
    index = json.loads((root / "model.safetensors.index.json").read_text())
    if set(index["weight_map"].values()) != set(conversion["output_files"]):
        raise ValueError("Index and conversion shard closures differ")
    # The SDK computes payload hashes during upload. Verify every remote digest
    # against the measured manifest before making the repository public.
    files = dict(conversion["output_files"])
    for name, metadata in files.items():
        if Path(name).name != name or (root / name).stat().st_size != metadata["bytes"]:
            raise ValueError(f"Invalid shard path or size: {name}")
    card_path = Path(__file__).parent / "model-cards" / f"kolibri-1-mlx-{'mixed-2bit' if args.profile == 'mixed2' else '8bit'}.md"
    (root / "README.md").write_bytes(card_path.read_bytes())
    (root / "LICENSE").write_bytes(args.upstream_license.read_bytes())
    (root / "UPSTREAM_MODEL_CARD.md").write_bytes(args.upstream_card.read_bytes())
    (root / "KOLIBRI_NATIVE_DIAGNOSTIC.json").write_bytes(args.evidence.read_bytes())
    qualification = dict(schema_version=1, profile=args.profile,
                         scope="Small paired-logprob diagnostic; general capability and Apple full-checkpoint memory fit unverified.",
                         suite_sha256=evidence["suite"]["sha256"], comparison=result,
                         conversion_sha256=measured["conversion_sha256"], scoring_receipt=measured["scoring_receipt"])
    (root / "KOLIBRI_QUALIFICATION.json").write_text(json.dumps(qualification, indent=2) + "\n")
    (root / "MODIFICATIONS.md").write_text(
        "# Modifications by Sawfwair\n\n"
        f"Source: Aleph-Alpha/Kolibri-1-BF16@{SOURCE_REVISION}.\n\n"
        f"Native runtime source snapshot: {args.source_commit}.\n\n"
        "Routed experts are stacked in numeric order. Router correction biases are\n"
        "converted losslessly from BF16 to FP32. Projection weights use the explicit\n"
        "MLX affine policies in config.json; embeddings, norms, router weights, and\n"
        "the vocabulary head retain source precision. The source chat template and\n"
        "tokenizer are preserved. No retraining or instruction fine-tuning occurred.\n\n"
        "KOLIBRI_CONVERSION.json retains the original pinned source/shard provenance.\n"
        "KOLIBRI_NATIVE_DIAGNOSTIC.json records all measured variants and test scope.\n"
        "Public availability does not change quality_qualified: false in the\n"
        "conversion manifest or establish general quality and Apple memory fit.\n"
    )
    for name in ["README.md", "LICENSE", "UPSTREAM_MODEL_CARD.md", "MODIFICATIONS.md",
                 "config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json",
                 "generation_config.json", "KOLIBRI_CONVERSION.json", "KOLIBRI_QUALIFICATION.json",
                 "KOLIBRI_NATIVE_DIAGNOSTIC.json"]:
        data = (root / name).read_bytes()
        files[name] = dict(sha256=digest(data), bytes=len(data))
    sums = "".join(f"{metadata['sha256']}  {name}\n" for name, metadata in sorted(files.items()))
    (root / "SHA256SUMS").write_text(sums)
    data = sums.encode()
    files["SHA256SUMS"] = dict(sha256=digest(data), bytes=len(data))
    return files


def verify(api, repository, revision, files):
    info = api.model_info(repository, revision=revision, files_metadata=True)
    remote = {row.rfilename: row for row in info.siblings}
    if not set(files).issubset(remote):
        raise ValueError("Hub repository is missing publication files")
    for name, expected in files.items():
        row = remote[name]
        if row.size != expected["bytes"]:
            raise ValueError(f"Hub size mismatch: {name}")
        if row.lfs is not None:
            sha = row.lfs.sha256
        else:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(repository, name, revision=info.sha, token=api.token)
            sha = digest(Path(path).read_bytes())
        if sha != expected["sha256"]:
            raise ValueError(f"Hub SHA-256 mismatch: {name}")
    return info.sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=REPOSITORIES, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--upstream-license", type=Path, required=True)
    parser.add_argument("--upstream-card", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    files = prepare(args)
    repository = REPOSITORIES[args.profile]
    print(json.dumps(dict(repository=repository, files=len(files), bytes=sum(r["bytes"] for r in files.values()))), flush=True)
    if args.prepare_only:
        return
    from huggingface_hub import HfApi
    token = args.token_file.read_text().strip() if args.token_file else None
    api = HfApi(token=token)
    if not args.verify_only:
        api.create_repo(repository, repo_type="model", private=True, exist_ok=True)
        if not api.model_info(repository).private:
            raise ValueError("Upload requires a private staging repository; use --verify-only for an existing release")
        api.upload_large_folder(repo_id=repository, repo_type="model", folder_path=args.artifact,
                                allow_patterns=list(files), num_workers=args.workers, print_report=True, print_report_every=60)
    revision = verify(api, repository, "main", files)
    if not args.verify_only:
        api.update_repo_settings(repository, repo_type="model", private=False)
    public_api = HfApi(token=False)
    public_revision = verify(public_api, repository, revision, files)
    receipt = dict(schema_version=1, repository=repository, revision=public_revision,
                   public=True, files=files, conversion_sha256=files["KOLIBRI_CONVERSION.json"]["sha256"])
    if args.receipt:
        args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(dict(repository=repository, revision=public_revision, public=True, verified_files=len(files))), flush=True)


if __name__ == "__main__":
    main()
