#!/usr/bin/env python3
"""Summarize one Actions attempt using job timings and cache restore outputs."""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path


def elapsed(start, end):
    if not start or not end:
        return None
    return max(0, (datetime.fromisoformat(end.replace("Z", "+00:00"))
                   - datetime.fromisoformat(start.replace("Z", "+00:00"))).total_seconds())


def phase(name):
    if name.startswith(("Restore ", "Save ", "Post Restore ")):
        return "cache"
    if name == "Run iOS unit tests":
        return "tests"
    if name.startswith(("Build ", "Warm default-branch ", "Run package checks",
                        "Run app-scoped package checks", "Run Linux CLI compatibility gate")):
        return "build_gate"
    if name.startswith("Verify "):
        return "validation"
    return "setup_other"


def summarize_jobs(pages):
    rows = []
    for page in pages:
        for job in page["jobs"]:
            if job["name"] == "ci-timings":
                continue
            total = elapsed(job.get("started_at"), job.get("completed_at"))
            phases = dict.fromkeys(("cache", "tests", "build_gate", "validation"), 0)
            steps = []
            for step in job.get("steps", []):
                seconds = (elapsed(step.get("started_at"), step.get("completed_at"))
                           if step["conclusion"] != "skipped" else 0)
                category = phase(step["name"])
                if category in phases:
                    phases[category] += seconds or 0
                steps.append({"name": step["name"], "phase": category,
                              "seconds": seconds, "conclusion": step["conclusion"]})
            phases["setup_other"] = (max(0, total - sum(phases.values()))
                                     if total is not None else None)
            rows.append({"name": job["name"], "conclusion": job["conclusion"],
                         "queue_seconds": elapsed(job.get("created_at"), job.get("started_at")),
                         "total_seconds": total, "phases": phases, "steps": steps})
    return rows


def summarize_caches(needs, pages):
    # Prefer the default-branch entry if an older PR cache shares its key.
    inventory = {}
    for page in pages:
        for cache in page["actions_caches"]:
            if cache["key"] not in inventory or cache["ref"] == "refs/heads/main":
                inventory[cache["key"]] = cache
    rows = []
    for job in ("swift", "linux-cli", "ios"):
        outputs = needs[job]["outputs"]
        kinds = ("dependencies", "native", "build") if job == "linux-cli" else ("dependencies", "build")
        for kind in kinds:
            primary = outputs.get(f"{kind}-key", "")
            matched = outputs.get(f"{kind}-match", "")
            outcome = outputs.get(f"{kind}-result", "skipped")
            if outcome == "failure":
                result = "restore failed"
            elif not primary:
                result = "not run"
            elif not matched:
                result = "miss"
            else:
                result = "exact hit" if primary == matched else "fallback hit"
            entry = inventory.get(matched)
            rows.append({"job": job, "cache": kind, "result": result,
                         "primary_key": primary, "matched_key": matched,
                         "archive_bytes": entry["size_in_bytes"] if entry else None})
    return rows


def duration(seconds):
    if seconds is None:
        return "—"
    minutes, seconds = divmod(round(seconds), 60)
    return f"{minutes}m{seconds:02d}s" if minutes else f"{seconds}s"


def markdown(report):
    lines = ["## CI timings and cache reuse", "",
             f"Event: `{report['event']}` · Lane: `{report['lane']}` · "
             f"Commit: `{report['sha']}` · Attempt: {report['attempt']}", "",
             "| Job | Result | Queue | Setup/other | Cache | Build/gate | Tests | Validation | Total |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in report["jobs"]:
        phases = row["phases"]
        values = [row["queue_seconds"], phases["setup_other"], phases["cache"],
                  phases["build_gate"], phases["tests"], phases["validation"], row["total_seconds"]]
        lines.append(f"| {row['name']} | {row['conclusion']} | "
                     + " | ".join(duration(value) for value in values) + " |")
    lines += ["", "Queue measures job creation to runner start; dependency waits are excluded. "
              "Total measures runner time, excluding queue. Setup/other includes checkout, installs, "
              "cleanup, and gaps between steps. Build/gate includes tests inside the macOS package gate "
              "and smoke checks inside the Linux gate. The Tests column measures standalone iOS tests.", "",
              "| Job | Cache | Restore | Archive size |",
              "| --- | --- | --- | ---: |"]
    for row in report["caches"]:
        size = f"{row['archive_bytes'] / 1024 ** 2:.1f} MiB" if row["archive_bytes"] is not None else "—"
        lines.append(f"| {row['job']} | {row['cache']} | {row['result']} | {size} |")
    lines += ["", "Archive sizes come from the cache inventory; an evicted entry has no size. "
              "A fallback hit restores a previous commit's compiled output. "
              "Main pushes warm caches; compare PRs with PRs and queue runs with queue runs.", "",
              "<details><summary>Cache keys</summary>", ""]
    for row in report["caches"]:
        lines += [f"- {row['job']} / {row['cache']}: requested `{row['primary_key'] or '(none)'}`; "
                  f"restored `{row['matched_key'] or '(none)'}`."]
    lines += ["", "</details>", "", "The `ci-timings-<attempt>` artifact contains seconds and per-step timings as JSON."]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--caches", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()
    needs = json.loads(os.environ["CI_NEEDS"])
    report = {"event": os.environ["GITHUB_EVENT_NAME"], "sha": os.environ["GITHUB_SHA"],
              "lane": needs["changes"]["outputs"].get("lane", "unknown"),
              "attempt": int(os.environ["GITHUB_RUN_ATTEMPT"]),
              "jobs": summarize_jobs(json.loads(args.jobs.read_text())),
              "caches": summarize_caches(needs, json.loads(args.caches.read_text()))}
    args.json_output.write_text(json.dumps(report, indent=2) + "\n")
    print(markdown(report))


if __name__ == "__main__":
    main()
