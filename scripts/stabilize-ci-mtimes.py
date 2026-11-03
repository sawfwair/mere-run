#!/usr/bin/env python3
"""Give every tracked file a modification time derived from its content.

A fresh checkout stamps every file with the current time, which makes SwiftPM
recompile the whole package even when a restored build directory is current.
Build tools rebuild when a time differs, not only when it is newer, so equal
content keeps its objects and any edit still recompiles.
"""

import os
import subprocess
import sys
from pathlib import Path

EPOCH = 1_000_000_000
SPAN = 500_000_000
REGULAR_FILE_MODES = {"100644", "100755"}


def main() -> None:
    # A container job's checkout belongs to a different user than the job.
    listing = subprocess.run(
        ["git", "-c", "safe.directory=*", "ls-files", "--stage", "-z"],
        check=True,
        capture_output=True,
    ).stdout
    stamped = 0
    for entry in listing.split(b"\0"):
        if not entry:
            continue
        metadata, path = entry.split(b"\t", 1)
        mode, blob, _ = metadata.decode().split(" ")
        if mode not in REGULAR_FILE_MODES:
            continue
        stamp = EPOCH + int(blob[:8], 16) % SPAN
        os.utime(path, (stamp, stamp))
        stamped += 1
    print(f"Stamped {stamped} tracked files.")

    # XcodeGen recreates its untracked project on each runner. Give those files
    # stable times too, so an identical project can reuse restored DerivedData.
    for directory in sys.argv[1:]:
        for path in sorted(Path(directory).rglob("*")):
            if path.is_file() and not path.is_symlink():
                blob = subprocess.check_output(["git", "hash-object", "--", str(path)]).strip()
                stamp = EPOCH + int(blob[:8], 16) % SPAN
                os.utime(path, (stamp, stamp))


if __name__ == "__main__":
    main()
