#!/usr/bin/env python3
"""Verify and publish a local Turbo mixed-Q4/Q8 bundle to Sawfwair.

Uses the local HF_WRITE_TOKEN credential, falling back to HF_TOKEN. No credentials
are sent to a conversion host.
Stages privately, verifies every file, then publishes and verifies anonymously.
"""
import argparse
import json
import os
from pathlib import Path

from convert_qwen_image21_turbo_mlx import ARTIFACT, REPOSITORY, REVISION, digest


def verify(api, revision, expected):
    from huggingface_hub import hf_hub_download
    info = api.model_info(ARTIFACT, revision=revision, files_metadata=True)
    remote = {entry.rfilename: entry for entry in info.siblings}
    for name, metadata in expected.items():
        entry = remote[name]
        if entry.size != metadata["bytes"]:
            raise ValueError("Remote size mismatch: " + name)
        sha = entry.lfs.sha256 if entry.lfs else digest(Path(hf_hub_download(
            ARTIFACT, name, revision=info.sha, token=api.token)))
        if sha != metadata["sha256"]:
            raise ValueError("Remote SHA-256 mismatch: " + name)
    return info.sha


def publish(root, receipt):
    from huggingface_hub import HfApi
    token = os.environ.get("HF_WRITE_TOKEN") or os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    if api.whoami()["auth"]["accessToken"]["role"] == "read":
        raise ValueError("The selected token is read-only; Sawfwair repository creation and upload require write access.")
    conversion = json.loads((root / "QWEN21_CONVERSION.json").read_text())
    if conversion["source_repository"] != REPOSITORY or conversion["source_revision"] != REVISION:
        raise ValueError("Source provenance differs from the pinned Turbo checkpoint")
    expected = {}
    for line in (root / "SHA256SUMS").read_text().splitlines():
        sha, name = line.split("  ", 1)
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Unsafe bundle filename")
        if "__pycache__" in Path(name).parts or name.endswith(".pyc"):
            raise ValueError("Generated Python cache is not a model artifact: " + name)
        path = root / name
        if digest(path) != sha:
            raise ValueError("Local SHA-256 mismatch: " + name)
        expected[name] = {"bytes": path.stat().st_size, "sha256": sha}
    sums = root / "SHA256SUMS"
    expected["SHA256SUMS"] = {"bytes": sums.stat().st_size, "sha256": digest(sums)}
    if not {"LICENSE", "Notice", "MODIFICATIONS.md", "UPSTREAM_MODEL_CARD.md", "README.md", "mererun_model.json"}.issubset(expected):
        raise ValueError("Incomplete provenance/license bundle")
    api.create_repo(ARTIFACT, private=True, exist_ok=True)
    if not api.model_info(ARTIFACT).private:
        raise ValueError("Refusing to overwrite a public release")
    api.upload_large_folder(repo_id=ARTIFACT, repo_type="model", folder_path=root,
                            allow_patterns=list(expected), num_workers=4,
                            print_report=True, print_report_every=60)
    revision = verify(api, "main", expected)
    api.update_repo_settings(ARTIFACT, private=False)
    public_revision = verify(HfApi(token=False), revision, expected)
    result = {"repository": ARTIFACT, "revision": public_revision, "public": True,
              "files": expected, "source_revision": REVISION,
              "bytes": sum(item["bytes"] for item in expected.values())}
    receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "files"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    publish(args.artifact, args.receipt)
