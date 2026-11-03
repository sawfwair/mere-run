"""Regression tests for CI measurements, cache timestamps, and warm-up scope."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("ci_summary", ROOT / "scripts/ci-summary.py")
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def init_repo(root):
    git(root, "init", "-q")
    git(root, "config", "user.name", "CI fixture")
    git(root, "config", "user.email", "ci@example.invalid")


class TimingTests(unittest.TestCase):
    def test_queue_excludes_dependency_wait_and_skipped_steps_cost_zero(self):
        row = SUMMARY.summarize_jobs([{"jobs": [{
            "name": "ios", "conclusion": "success",
            "created_at": "2026-11-03T01:00:00Z",
            "started_at": "2026-11-03T01:00:05Z",
            "completed_at": "2026-11-03T01:02:05Z",
            "steps": [
                {"name": "Restore iOS build directory", "conclusion": "success",
                 "started_at": "2026-11-03T01:00:10Z", "completed_at": "2026-11-03T01:00:20Z"},
                {"name": "Build iOS app and tests for release", "conclusion": "success",
                 "started_at": "2026-11-03T01:00:20Z", "completed_at": "2026-11-03T01:01:50Z"},
                {"name": "Run iOS unit tests", "conclusion": "skipped",
                 "started_at": "0001-01-01T00:00:00Z", "completed_at": "2026-11-03T01:02:00Z"},
            ],
        }]}])[0]
        self.assertEqual(row["queue_seconds"], 5)
        self.assertEqual(row["total_seconds"], 120)
        self.assertEqual(row["phases"], {
            "cache": 10, "build_gate": 90, "tests": 0, "validation": 0, "setup_other": 20,
        })

    def test_missing_timestamps_and_multiple_pages(self):
        rows = SUMMARY.summarize_jobs([
            {"jobs": [{"name": "ios", "conclusion": "skipped"}]},
            {"jobs": [{"name": "ci-timings"}]},
        ])
        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0]["total_seconds"])
        self.assertEqual(SUMMARY.duration(None), "—")

    def test_cache_fallback_is_a_hit_and_failed_restores_are_visible(self):
        needs = {job: {"outputs": {}} for job in ("swift", "linux-cli", "ios")}
        needs["linux-cli"]["outputs"] = {
            "dependencies-key": "deps", "dependencies-match": "deps", "dependencies-result": "success",
            "build-key": "build-new", "build-match": "build-old", "build-result": "success",
        }
        needs["ios"]["outputs"] = {
            "dependencies-key": "ios-deps", "dependencies-result": "success",
            "build-result": "failure",
        }
        caches = SUMMARY.summarize_caches(needs, [{"actions_caches": [
            {"key": "build-old", "ref": "refs/heads/main", "size_in_bytes": 1024},
        ]}])
        self.assertEqual([row["result"] for row in caches], [
            "not run", "not run", "exact hit", "fallback hit", "miss", "restore failed",
        ])
        self.assertEqual(caches[3]["archive_bytes"], 1024)
        report = {"event": "merge_group", "lane": "full", "sha": "abc123",
                  "attempt": 2, "jobs": [], "caches": caches}
        self.assertIn("fallback hit", SUMMARY.markdown(report))
        self.assertIn("Attempt: 2", SUMMARY.markdown(report))


class CacheTimestampTests(unittest.TestCase):
    def test_identical_generated_project_keeps_time_and_edits_change_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            init_repo(root)
            tracked = root / "source.swift"
            tracked.write_text("let value = 1\n")
            git(root, "add", ".")
            project = root / "app.xcodeproj"
            project.mkdir()
            generated = project / "project.pbxproj"
            generated.write_text("project version 1\n")
            stamp = [sys.executable, str(ROOT / "scripts/stabilize-ci-mtimes.py"), str(project)]
            subprocess.run(stamp, cwd=root, check=True, capture_output=True)
            original = generated.stat().st_mtime_ns
            tracked_time = tracked.stat().st_mtime_ns
            generated.write_text("project version 1\n")
            subprocess.run(stamp, cwd=root, check=True, capture_output=True)
            self.assertEqual(generated.stat().st_mtime_ns, original)
            self.assertEqual(tracked.stat().st_mtime_ns, tracked_time)
            generated.write_text("project version 2\n")
            subprocess.run(stamp, cwd=root, check=True, capture_output=True)
            self.assertNotEqual(generated.stat().st_mtime_ns, original)

    def test_linux_numerics_patch_does_not_rewrite_a_restored_checkout(self):
        source = (ROOT / "scripts/prepare-linux-native.sh").read_text()
        function = source.split("patch_mlx_swift_for_linux() {", 1)[1].split("\n}\n", 1)[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Package.swift").write_text("// fixture package\n")
            numerics = root / "Sources/RealModule"
            numerics.mkdir(parents=True)
            float16 = numerics / "Float16+Real.swift"
            float16.write_text("#if !arch(wasm32)\n// fixture\n#endif\n")
            script = """
                set -euo pipefail
                require_tool() { :; }
                swift() { :; }
                patch_mlx_cpu_jit_f16c_probe() { :; }
                if [[ "$(uname -s)" == Darwin ]]; then
                  sed() { shift; command sed -i '' "$@"; }
                fi
                mlx_swift_checkout="$PWD"
                swift_numerics_checkout="$PWD"
                linux_accel=cpu
            """
            script = textwrap.dedent(script) + "\npatch_mlx_swift_for_linux() {" + function + "\n}\npatch_mlx_swift_for_linux\n"
            subprocess.run(["bash", "-c", script], cwd=root, check=True, capture_output=True)
            patched = float16.read_bytes()
            timestamp = float16.stat().st_mtime_ns
            self.assertEqual(patched.count(b"os(Linux)"), 1)
            subprocess.run(["bash", "-c", script], cwd=root, check=True, capture_output=True)
            self.assertEqual(float16.read_bytes(), patched)
            self.assertEqual(float16.stat().st_mtime_ns, timestamp)


class WarmScopeTests(unittest.TestCase):
    def test_main_warms_all_platforms_and_pr_scopes_stay_narrow(self):
        workflow = (ROOT / ".github/workflows/ci.yml").read_text()
        block = workflow.split("      - name: Classify the changed paths\n", 1)[1]
        lines = block.split("        run: |\n", 1)[1].splitlines(keepends=True)
        script_lines = []
        for line in lines:
            if line.strip() and not line.startswith("          "):
                break
            script_lines.append(line)
        script = textwrap.dedent("".join(script_lines))
        cases = [
            ("README.md", "pull_request", "false", "fast", "false", "false"),
            ("apps/macos/view.swift", "merge_group", "true", "fast", "false", "false"),
            ("Sources/MereRunCLI/command.swift", "pull_request", "true", "full", "true", "false"),
            ("Sources/MereRunCore/runtime.swift", "merge_group", "true", "full", "true", "true"),
            (".github/workflows/ci.yml", "pull_request", "true", "full", "true", "true"),
            ("scripts/stabilize-ci-mtimes.py", "pull_request", "true", "full", "true", "true"),
            ("unknown/file", "merge_group", "true", "full", "true", "true"),
            ("README.md", "push", "true", "warm", "true", "true"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            init_repo(root)
            git(root, "commit", "--allow-empty", "-qm", "base")
            base = git(root, "rev-parse", "HEAD")
            for path, event, code, lane, linux, ios in cases:
                with self.subTest(path=path, event=event):
                    git(root, "reset", "--hard", base)
                    changed = root / path
                    changed.parent.mkdir(parents=True, exist_ok=True)
                    changed.write_text("fixture\n")
                    git(root, "add", path)
                    git(root, "commit", "-qm", "change")
                    output = root / "output"
                    output.write_text("")
                    env = dict(os.environ, BASE_SHA=base, HEAD_SHA=git(root, "rev-parse", "HEAD"),
                               EVENT_NAME=event, GITHUB_OUTPUT=str(output),
                               GITHUB_STEP_SUMMARY=str(root / "summary"))
                    subprocess.run(["bash", "-c", script], cwd=root, env=env, check=True, capture_output=True)
                    actual = dict(line.split("=", 1) for line in output.read_text().splitlines())
                    self.assertEqual(actual, {"code": code, "lane": lane, "linux": linux, "ios": ios})


if __name__ == "__main__":
    unittest.main()
