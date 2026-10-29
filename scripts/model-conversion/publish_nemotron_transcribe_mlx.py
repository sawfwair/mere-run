#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13,<3.15"
# dependencies = ["huggingface-hub==2.3.0"]
# ///
"""Publish the verified Nemotron Transcribe routed-expert INT4 artifact to Sawfwair."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

REPOSITORY = "Sawfwair/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-MLX-4bit"
SOURCE_REVISION = "ef905575a106fdad904f3b0ce661dccad654607d"
MODEL_ID = "speech-asr-nemotron-35-transcribe"
FORMAT = {"version": 1, "bits": 4, "groupSize": 64, "mode": "affine",
          "scope": "routed-experts", "layout": "expert-output-input"}


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def prepare(args):
    root = Path(__file__).resolve().parents[2]
    source, stage = args.artifact.resolve(), args.staging.resolve()
    if stage.exists():
        raise FileExistsError(f"Publication staging already exists: {stage}")
    conversion = json.loads((source / "MERERUN_CONVERSION.json").read_text())
    if (conversion["source"]["revision"] != SOURCE_REVISION or conversion["quantization"] != FORMAT
            or conversion["routed_matrices"] != 46 or conversion["output_tensors"] != 1361):
        raise ValueError("Artifact does not match the qualified conversion")
    index = json.loads((source / "model.safetensors.index.json").read_text())
    recorded = {row["filename"]: row for row in conversion["artifacts"]}
    if set(index["weight_map"].values()) != set(recorded) or len(recorded) != 29:
        raise ValueError("Converted shard closure changed")
    for name, row in sorted(recorded.items()):
        path = source / name
        if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            raise ValueError("Converted artifact checksum mismatch: " + name)
    stage.mkdir(parents=True)
    for name in recorded:
        os.link(source / name, stage / name)
    for name in ["config.json", "tokenizer.json", "tokenizer_config.json", "expert-quantization.json",
                 "model.safetensors.index.json", "MERERUN_CONVERSION.json", "LICENSE", "THIRD_PARTY_NOTICES.md"]:
        shutil.copyfile(source / name, stage / name)
    shutil.copytree(source / "licenses", stage / "licenses")
    manifest = json.loads((source / "mererun_model.json").read_text())
    manifest.pop("usageTermsAcknowledged", None)
    manifest.pop("createdAt", None)
    manifest["id"] = MODEL_ID
    (stage / "mererun_model.json").write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.copyfile(args.upstream / "README.md", stage / "UPSTREAM_MODEL_CARD.md")
    shutil.copyfile(Path(__file__).parent / "model-cards/nemotron-transcribe-mlx-4bit.md", stage / "README.md")
    qualification = root / "docs/qualification/nemotron-transcribe-int4-20261028"
    for original, name in [("README.md", "QUALIFICATION.md"), ("report.json", "QUALIFICATION.json")]:
        shutil.copyfile(qualification / original, stage / name)
    (stage / "MODIFICATIONS.md").write_text(
        "# Modifications by Sawfwair\n\n"
        f"Source: nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B@{SOURCE_REVISION}.\n\n"
        "The 46 main-decoder routed expert matrices were transposed from [expert,input,output] "
        "to [expert,output,input] and packed as MLX affine W4A16/group-64 with BF16 scales/biases. "
        "All other inference tensors retain their original values/dtypes, including unused MTP tensors. "
        "CTC and extra training state are omitted from the inference artifact. No training occurred.\n\n"
        "Weights are byte-identical to the qualified local conversion. Publication adds a model card, "
        "qualification receipts, this notice, and checksums; it removes per-install consent/timestamp "
        "state from mererun_model.json. The original model card is UPSTREAM_MODEL_CARD.md. "
        "Original OpenMDW-1.1 licenses and notices are retained.\n")
    expected = {str(p.relative_to(stage)): {"bytes": p.stat().st_size, "sha256": digest(p)}
                for p in sorted(stage.rglob("*")) if p.is_file()}
    sums = stage / "SHA256SUMS"
    sums.write_text("".join(f"{row['sha256']}  {name}\n" for name, row in sorted(expected.items())))
    expected[sums.name] = {"bytes": sums.stat().st_size, "sha256": digest(sums)}
    return expected


def verify(api, revision, expected):
    from huggingface_hub import hf_hub_download
    info = api.model_info(REPOSITORY, revision=revision, files_metadata=True)
    remote = {row.rfilename: row for row in info.siblings}
    if set(remote) - set(expected) - {".gitattributes"} or not set(expected).issubset(remote):
        raise ValueError("Published file closure differs from the prepared artifact")
    for name, row in expected.items():
        entry = remote[name]
        if entry.size != row["bytes"]:
            raise ValueError("Published size mismatch: " + name)
        if entry.lfs:
            checksum = entry.lfs.sha256
        else:
            if entry.size > 32 * 1024 * 1024:
                raise ValueError("Large file has no server-side SHA-256 metadata: " + name)
            checksum = digest(Path(hf_hub_download(REPOSITORY, name, revision=info.sha, token=api.token)))
        if checksum != row["sha256"]:
            raise ValueError("Published checksum mismatch: " + name)
    return info.sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--prepared-receipt", type=Path)
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    expected = (json.loads(args.prepared_receipt.read_text())["files"]
                if args.prepared_receipt else prepare(args))
    result = {"repository": REPOSITORY, "source_revision": SOURCE_REVISION,
              "bytes": sum(row["bytes"] for row in expected.values()), "files": expected}
    if args.prepare_only:
        args.receipt.write_text(json.dumps(dict(result, uploaded=False), indent=2) + "\n")
        print(json.dumps({"prepared_files": len(expected), "bytes": result["bytes"]}), flush=True)
        return
    for name, row in expected.items():
        path = args.staging / name
        if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            raise ValueError("Prepared publication file changed: " + name)
    from huggingface_hub import HfApi
    token = args.token_file.read_text().strip() if args.token_file else os.environ.get("HF_WRITE_TOKEN") or os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    identity = api.whoami()
    if not any(org["name"].lower() == "sawfwair" for org in identity.get("orgs", [])):
        raise ValueError("Authenticated account is not a Sawfwair organization member")
    if not args.verify_only:
        api.create_repo(REPOSITORY, repo_type="model", private=True, exist_ok=True)
        if not api.model_info(REPOSITORY).private:
            raise ValueError("Refusing to overwrite a public release; use --verify-only")
        api.upload_folder(repo_id=REPOSITORY, repo_type="model", folder_path=args.staging,
                          allow_patterns=list(expected), commit_message="Publish qualified native MLX 4-bit routed experts")
    revision = verify(api, "main", expected)
    if not args.verify_only:
        api.update_repo_settings(REPOSITORY, repo_type="model", private=False)
    revision = verify(HfApi(token=False), revision, expected)
    result.update(revision=revision, public=True, uploaded=True)
    args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "files"}), flush=True)


if __name__ == "__main__":
    main()
