#!/usr/bin/env python3
"""Publish the measured DRAMA-3 expert-only Q4 export to Sawfwair.

Requires huggingface-hub. Authenticate with hf auth login before publishing.
Stages privately, verifies file hashes, then verifies the public immutable revision.
Hard links preserve the measured export without duplicating its 27.4 GB payload.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

REPOSITORY = "Sawfwair/DRAMA-3-MLX-4bit"
SOURCE_REVISION = "dd64a85cd82e9521f7202610ede97b367fc83946"
FORMAT = "drama3-experts-q4-g64-v1"


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def prepare(args):
    source, stage = args.artifact.resolve(), args.staging.resolve()
    if source == stage or source in stage.parents:
        raise ValueError("Publication staging must be outside the measured export")
    conversion = json.loads((source / "drama3-quantization.json").read_text())
    if (conversion["format"] != FORMAT or conversion["expertBits"] != 4 or
            conversion["groupSize"] != 64 or
            conversion["upstreamRepository"] != "fishaudio/drama-3" or
            conversion["upstreamRevision"] != SOURCE_REVISION):
        raise ValueError("Export differs from the qualified source and Q4 policy")
    recorded = {row["name"]: row for row in conversion["files"]}
    if len(recorded) != len(conversion["files"]):
        raise ValueError("Duplicate export filenames")
    actual = {p.name for p in source.iterdir() if p.is_file()}
    if actual != set(recorded) | {"drama3-quantization.json"}:
        raise ValueError("Measured export file closure differs from its receipt")
    index = json.loads((source / "model.safetensors.index.json").read_text())
    shards = set(index["weight_map"].values())
    if shards != {name for name in recorded if name.endswith(".safetensors")} or len(shards) != 88:
        raise ValueError("Incomplete 88-shard index closure")
    expected = {}
    for position, (name, row) in enumerate(sorted(recorded.items()), 1):
        if Path(name).name != name or (source / name).is_symlink():
            raise ValueError("Unsafe export filename: " + name)
        path = source / name
        if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            raise ValueError("Measured export SHA-256 mismatch: " + name)
        if position % 10 == 0:
            print(f"Verified export files: {position}/{len(recorded)}", flush=True)
        expected[name] = dict(bytes=row["bytes"], sha256=row["sha256"])
    receipt_path = source / "drama3-quantization.json"
    expected[receipt_path.name] = dict(bytes=receipt_path.stat().st_size, sha256=digest(receipt_path))
    qualification_root = Path(__file__).resolve().parents[2] / "docs/qualification/drama3-native-20261015"
    extended = json.loads((qualification_root / "extended-results.json").read_text())
    if (extended["checkpoint_revision"] != SOURCE_REVISION or extended["quantization"] != FORMAT or
            len(extended["cases"]) != 7 or
            any(row["status"] != "passed" for row in extended["cases"].values())):
        raise ValueError("Bounded qualification receipt differs from published scope")
    additions = {
        "README.md": Path(__file__).parent / "model-cards/drama-3-mlx-4bit.md",
        "DRAMA3_EXTENDED_RESULTS.json": qualification_root / "extended-results.json",
        "DRAMA3_MAIN_INTEGRATION.json": qualification_root / "main-integration-20261023.json",
    }
    allowed = set(expected) | set(additions) | {"UPSTREAM_MODEL_CARD.md", "MODIFICATIONS.md", "SHA256SUMS"}
    stage.mkdir(parents=True, exist_ok=True)
    if any(p.name not in allowed | {".cache"} for p in stage.iterdir()):
        raise ValueError("Unexpected publication staging file")
    # Link weights and runtime metadata; never edit the measured source files.
    for name in sorted(expected):
        destination = stage / ("UPSTREAM_MODEL_CARD.md" if name == "README.md" else name)
        if destination.exists():
            if not destination.samefile(source / name):
                raise ValueError("Staged export is not the measured source: " + name)
        else:
            os.link(source / name, destination)
    expected["UPSTREAM_MODEL_CARD.md"] = expected.pop("README.md")
    for name, path in additions.items():
        destination = stage / name
        if destination.exists():
            destination.unlink()
        shutil.copyfile(path, destination)
        expected[name] = dict(bytes=destination.stat().st_size, sha256=digest(destination))
    modifications = stage / "MODIFICATIONS.md"
    modifications.write_text(
        "# Modifications by Sawfwair\n\n"
        f"Source: fishaudio/drama-3@{SOURCE_REVISION}.\n\n"
        "Built with Fish Audio. Routed experts were decoded from FP8 using the source block scales, "
        "then packed as MLX affine Q4/group-64. Other tensors retain checkpoint precision; "
        "the original DAC codec is unchanged. No training or fine-tuning occurred.\n\n"
        "The original export README is UPSTREAM_MODEL_CARD.md. The original export receipt "
        "drama3-quantization.json describes that README under its original name. The published "
        "README adds runtime instructions, attribution, and measured limitations. "
        "Publication adds the two bounded qualification receipts and SHA256SUMS; "
        "runtime files are byte-identical to the measured export.\n"
    )
    expected[modifications.name] = dict(bytes=modifications.stat().st_size, sha256=digest(modifications))
    sums = stage / "SHA256SUMS"
    sums.write_text("".join(f"{row['sha256']}  {name}\n" for name, row in sorted(expected.items())))
    expected[sums.name] = dict(bytes=sums.stat().st_size, sha256=digest(sums))
    return expected


def verify(api, revision, expected):
    from huggingface_hub import hf_hub_download
    info = api.model_info(REPOSITORY, revision=revision, files_metadata=True)
    remote = {row.rfilename: row for row in info.siblings}
    if set(remote) - set(expected) - {".gitattributes"} or not set(expected).issubset(remote):
        raise ValueError("Hub publication file closure differs")
    for name, metadata in expected.items():
        row = remote[name]
        if row.size != metadata["bytes"]:
            raise ValueError("Hub size mismatch: " + name)
        sha = row.lfs.sha256 if row.lfs else digest(Path(hf_hub_download(
            REPOSITORY, name, revision=info.sha, token=api.token)))
        if sha != metadata["sha256"]:
            raise ValueError("Hub SHA-256 mismatch: " + name)
    return info.sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    expected = prepare(args)
    summary = dict(repository=REPOSITORY, file_count=len(expected), bytes=sum(r["bytes"] for r in expected.values()))
    print(json.dumps(summary), flush=True)
    if args.prepare_only:
        args.receipt.write_text(json.dumps(dict(**summary, uploaded=False, files=expected), indent=2) + "\n")
        return
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_WRITE_TOKEN") or os.environ.get("HF_TOKEN"))
    identity = api.whoami()
    if not any(org["name"].lower() == "sawfwair" for org in identity.get("orgs", [])):
        raise ValueError("Authenticated account is not a Sawfwair organization member")
    if not args.verify_only:
        api.create_repo(REPOSITORY, repo_type="model", private=True, exist_ok=True)
        if not api.model_info(REPOSITORY).private:
            raise ValueError("Refusing to overwrite a public release; use --verify-only")
        api.upload_large_folder(repo_id=REPOSITORY, repo_type="model", folder_path=args.staging,
                                allow_patterns=list(expected), num_workers=4,
                                print_report=True, print_report_every=60)
    revision = verify(api, "main", expected)
    if not args.verify_only:
        api.update_repo_settings(REPOSITORY, repo_type="model", private=False)
    public_api = HfApi(token=False)
    revision = verify(public_api, revision, expected)
    if public_api.model_info(REPOSITORY, revision=revision).private:
        raise ValueError("Published repository is not publicly accessible")
    result = dict(repository=REPOSITORY, revision=revision, public=True, uploaded=True,
                  bytes=summary["bytes"], source_revision=SOURCE_REVISION, files=expected)
    args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k:v for k,v in result.items() if k != "files"}), flush=True)


if __name__ == "__main__":
    main()
