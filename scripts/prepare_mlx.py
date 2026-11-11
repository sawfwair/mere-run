#!/usr/bin/env python3
"""Apply or verify the audited host-only patch for the pinned MLX dependency."""

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = "Source/Cmlx/mlx"
PATCH = ROOT / "scripts/patches/mlx-m5-affine-q8-nax.patch"
MANIFEST = PATCH.with_suffix(".json")


@dataclass(frozen=True)
class PatchManifest:
    swift_revision: str
    core_revision: str
    source_path: str
    base_sha256: str
    patched_sha256: str
    patch_sha256: str


def digest(data):
    return hashlib.sha256(data).hexdigest()


def git(checkout, *args):
    return subprocess.check_output(
        ["git", "-C", str(checkout), *args], text=True
    ).strip()


def changed_paths(checkout):
    paths = set(git(checkout, "diff", "HEAD", "--ignore-submodules=none", "--name-only").splitlines())
    paths.update(git(checkout, "diff", "--cached", "--ignore-submodules=none", "--name-only").splitlines())
    return paths


def validate_state(manifest, swift_revision, core_revision, swift_changes,
                   core_changes, source_sha256):
    if swift_revision != manifest.swift_revision or core_revision != manifest.core_revision:
        raise ValueError("MLX revisions differ from the audited patch manifest.")
    if source_sha256 == manifest.base_sha256 and not swift_changes and not core_changes:
        return "original"
    if (source_sha256 == manifest.patched_sha256
            and swift_changes == {CORE_PATH}
            and core_changes == {manifest.source_path}):
        return "patched"
    raise ValueError("Unverified MLX edits: only the exact audited host patch is allowed.")


def verify_checkout(checkout, manifest):
    core = checkout / CORE_PATH
    return validate_state(
        manifest, git(checkout, "rev-parse", "HEAD"), git(core, "rev-parse", "HEAD"),
        changed_paths(checkout), changed_paths(core),
        digest((core / manifest.source_path).read_bytes())
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, default=Path(os.environ.get(
        "MERERUN_MLX_SWIFT_CHECKOUT", ROOT / ".build/checkouts/mlx-swift")))
    parser.add_argument("--verify-only", action="store_true",
                        help="Accept the pristine pin or the exact host patch; change no files.")
    args = parser.parse_args()
    manifest = PatchManifest(**json.loads(MANIFEST.read_text()))
    if digest(PATCH.read_bytes()) != manifest.patch_sha256:
        raise ValueError("MLX patch differs from its audited manifest.")
    if not args.checkout.exists() and not args.verify_only:
        subprocess.run(["swift", "package", "resolve"], cwd=ROOT, check=True)
    state = verify_checkout(args.checkout, manifest)
    if state == "original" and not args.verify_only:
        core = args.checkout / CORE_PATH
        source = core / manifest.source_path
        subprocess.run(["git", "-C", str(core), "apply", "--check", str(PATCH)], check=True)
        source.chmod(source.stat().st_mode | stat.S_IWUSR)
        subprocess.run(["git", "-C", str(core), "apply", str(PATCH)], check=True)
        state = verify_checkout(args.checkout, manifest)
    print(f"[mlx] verified {state} host dispatch; shader sources remain unchanged.", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"[mlx] {error}", file=sys.stderr)
        sys.exit(1)
