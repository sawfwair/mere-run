"""Regression tests for CI measurements, cache timestamps, and warm-up scope."""

import importlib.util
import os
from pathlib import Path
import platform
import re
import shutil
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


def workflow_script(name):
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    block = workflow.split(f"      - name: {name}\n", 1)[1]
    lines = block.split("        run: |\n", 1)[1].splitlines(keepends=True)
    script_lines = []
    for line in lines:
        if line.strip() and not line.startswith("          "):
            break
        script_lines.append(line)
    return textwrap.dedent("".join(script_lines))


def shell_function(path, name):
    body = (ROOT / path).read_text().split(f"{name}() {{", 1)[1].split("\n}\n", 1)[0]
    return f"{name}() {{" + body + "\n}\n"


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
            "native-key": "native", "native-match": "native", "native-result": "success",
        }
        needs["ios"]["outputs"] = {
            "dependencies-key": "ios-deps", "dependencies-result": "success",
            "build-result": "failure",
        }
        caches = SUMMARY.summarize_caches(needs, [{"actions_caches": [
            {"key": "build-old", "ref": "refs/heads/main", "size_in_bytes": 1024},
        ]}])
        self.assertEqual([row["result"] for row in caches], [
            "not run", "not run", "exact hit", "exact hit", "fallback hit", "miss", "restore failed",
        ])
        self.assertEqual(caches[3]["cache"], "native")
        self.assertEqual(caches[4]["archive_bytes"], 1024)
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
        script = workflow_script("Classify the changed paths")
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


class LinuxCacheKeyTests(unittest.TestCase):
    def keys(self, compiler="gcc 11.4", cpu="runner-a", target="x86-64-v3", **overrides):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            for name in ("swift", "cc", "c++", "clang", "cmake", "awk"):
                path = fake_bin / name
                path.write_text('#!/bin/sh\ncase "$(basename "$0")" in\n'
                                'cc|c++) echo "$COMPILER_VERSION" ;;\n'
                                'awk) echo "$FAKE_CPU_FLAGS" ;;\n'
                                '*) echo "fixed version" ;;\nesac\n')
                path.chmod(0o755)
            env = dict(os.environ, PATH=f"{fake_bin}:{os.environ['PATH']}",
                       RUNNER_OS="Linux", RUNNER_ARCH="X64", MERERUN_LINUX_CPU_TARGET=target,
                       DEPENDENCY_HASH="downloads", NATIVE_HASH="native", BUILD_HASH="build",
                       COMPILER_VERSION=compiler, FAKE_CPU_FLAGS=cpu, GITHUB_OUTPUT=str(root / "outputs"))
            env.update(overrides)
            result = subprocess.run(["bash", "-c", workflow_script("Identify Linux cache keys")],
                                    env=env, text=True, capture_output=True)
            if result.returncode:
                return result.returncode
            return dict(line.split("=", 1) for line in (root / "outputs").read_text().splitlines())

    def test_host_features_do_not_partition_caches(self):
        self.assertEqual(self.keys(cpu="avx2 fma"), self.keys(cpu="avx2 fma avx512f amx_tile"))

    def test_compiler_changes_keep_downloads_but_invalidate_compiled_output(self):
        old, new = self.keys(), self.keys(compiler="gcc 12")
        self.assertEqual(old["dependencies"], new["dependencies"])
        self.assertNotEqual(old["native"], new["native"])
        self.assertNotEqual(old["build"], new["build"])

    def test_build_policy_changes_do_not_invalidate_downloads_or_native_libraries(self):
        old, new = self.keys(), self.keys(BUILD_HASH="new-build")
        self.assertEqual(old["dependencies"], new["dependencies"])
        self.assertEqual(old["native"], new["native"])
        self.assertNotEqual(old["build"], new["build"])

    def test_native_policy_changes_invalidate_native_and_swift_output(self):
        old, new = self.keys(), self.keys(NATIVE_HASH="new-native")
        self.assertEqual(old["dependencies"], new["dependencies"])
        self.assertNotEqual(old["native"], new["native"])
        self.assertNotEqual(old["build"], new["build"])

    def test_shared_caches_reject_host_native_target(self):
        self.assertEqual(self.keys(target="native"), 1)


class LinuxCPUTargetTests(unittest.TestCase):
    def configure(self, target="x86-64-v3", arch="x86_64", accel="cpu"):
        script = 'source "$HELPER"\nconfigure_linux_cpu_target "$ARCH" "$ACCEL"\n'
        script += 'printf "%s\\0" "${linux_cpu_cmake_flags[@]}" -- "${linux_cpu_swift_flags[@]}"\n'
        env = dict(os.environ, HELPER=str(ROOT / "scripts/linux-cpu-target.sh"),
                   MERERUN_LINUX_CPU_TARGET=target, ARCH=arch, ACCEL=accel)
        return subprocess.run(["bash", "-ec", script], env=env, capture_output=True)

    def test_target_controls_all_compilers_and_disables_host_extensions(self):
        result = self.configure()
        self.assertEqual(result.returncode, 0, result.stderr)
        cmake, swift = result.stdout.decode().strip("\0").split("\0--\0")
        self.assertIn("-DGGML_NATIVE=OFF", cmake.split("\0"))
        self.assertIn("-DCMAKE_CXX_FLAGS=-march=x86-64-v3 -mtune=generic", cmake.split("\0"))
        for feature in ("AVX_VNNI", "AVX512", "AVX512_VBMI", "AVX512_VNNI", "AVX512_BF16",
                        "AMX_TILE", "AMX_INT8", "AMX_BF16"):
            self.assertIn(f"-DGGML_{feature}=OFF", cmake.split("\0"))
        self.assertNotIn("native", swift)

    def test_target_is_explicit_and_only_supported_for_x86_cpu_builds(self):
        for args in ({"arch": "arm64"}, {"accel": "cuda"}, {"target": "unknown"}):
            with self.subTest(args=args):
                self.assertEqual(self.configure(**args).returncode, 64)
        self.assertEqual(self.configure(target="native", arch="arm64", accel="cuda").returncode, 0)

    @unittest.skipUnless(shutil.which("clang"), "requires Clang")
    def test_c_and_cxx_baseline_has_avx2_without_avx512_or_amx(self):
        source = """
            #if !defined(__AVX2__) || !defined(__FMA__) || !defined(__F16C__) || !defined(__BMI2__)
            #error missing v3 features
            #endif
            #if defined(__AVX512F__) || defined(__AVXVNNI__) || defined(__AMX_TILE__)
            #error host-only extensions enabled
            #endif
            int main(void) { return 0; }
        """
        for language in ("c", "c++"):
            result = subprocess.run(["clang", "--target=x86_64-unknown-linux-gnu", "-march=x86-64-v3",
                                     "-mtune=generic", "-fsyntax-only", "-x", language, "-"],
                                    input=source, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_both_llama_configurations_receive_the_fixed_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = '''
                set -euo pipefail
                source "$HELPER"
                configure_linux_cpu_target x86_64 cpu
                require_tool() { :; }
                git() { :; }
                cp() { :; }
                write_llama_pc() { :; }
                native_build_jobs() { echo 2; }
                cmake() { printf '%s\\0' "$@" >> "$RECORD"; printf '\\n' >> "$RECORD"; }
                linux_accel=cpu
                llama_src="$PWD/src"
                llama_build="$PWD/build"
                llama_prefix="$PWD/install"
                llama_commit=fixture
                llama_url=https://example.invalid/fixture.git
            '''
            script = textwrap.dedent(script) + shell_function("scripts/prepare-linux-native.sh", "build_llama")
            script += "build_llama\n"
            env = dict(os.environ, HELPER=str(ROOT / "scripts/linux-cpu-target.sh"),
                       MERERUN_LINUX_CPU_TARGET="x86-64-v3", RECORD=str(root / "cmake-args"))
            subprocess.run(["bash", "-c", script], cwd=root, env=env, check=True, capture_output=True)
            calls = [line.split("\0") for line in (root / "cmake-args").read_text().splitlines()]
            configurations = [args for args in calls if "-S" in args]
            self.assertEqual(len(configurations), 2)
            for args in configurations:
                self.assertIn("-DGGML_NATIVE=OFF", args)
                self.assertIn("-DCMAKE_CXX_FLAGS=-march=x86-64-v3 -mtune=generic", args)

    def test_swift_build_wrapper_forwards_the_target_to_all_compilers(self):
        script = 'source "$HELPER"\nconfigure_linux_cpu_target x86_64 cpu\n'
        script += 'swift() { printf "%s\\0" "$@"; }\n'
        script += shell_function("scripts/check-linux.sh", "linux_swift")
        script += "linux_swift build --product mere.run\n"
        env = dict(os.environ, HELPER=str(ROOT / "scripts/linux-cpu-target.sh"),
                   MERERUN_LINUX_CPU_TARGET="x86-64-v3")
        result = subprocess.run(["bash", "-ec", script], env=env, capture_output=True, check=True)
        args = result.stdout.decode().strip("\0").split("\0")
        for compiler in ("-Xcc", "-Xcxx"):
            self.assertTrue(any(args[i:i+2] == [compiler, "-march=x86-64-v3"]
                                for i in range(len(args) - 1)))
        self.assertEqual(args[-2:], ["--product", "mere.run"])

    @unittest.skipUnless(shutil.which("swiftc") and
                         (sys.platform == "darwin" or platform.machine() == "x86_64"),
                         "requires an x86_64 Swift toolchain or Xcode SDK")
    def test_swift_emits_the_declared_cpu_target(self):
        result = self.configure()
        swift_flags = result.stdout.decode().strip("\0").split("\0--\0")[1].split("\0")
        compiler_flags = []
        for wrapper, flag in zip(swift_flags[::2], swift_flags[1::2]):
            if wrapper == "-Xcc":
                compiler_flags.extend([wrapper, flag])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "probe.swift"
            source.write_text("public func cpuProbe(_ x: Int) -> Int { x + 1 }\n")
            target = "x86_64-apple-macosx15.0" if sys.platform == "darwin" else "x86_64-unknown-linux-gnu"
            subprocess.run(["swiftc", "-target", target, *compiler_flags,
                            "-module-cache-path", str(root / "modules"),
                            "-emit-ir", str(source), "-o", str(root / "probe.ll")],
                           text=True, capture_output=True, check=True)
            ir = (root / "probe.ll").read_text()
            self.assertIn('"target-cpu"="x86-64-v3"', ir)
            features = re.findall(r'"target-features"="([^"]+)"', ir)
            self.assertTrue(features)
            for feature_set in features:
                self.assertIn("+avx2", feature_set)
                self.assertNotIn("+avx512", feature_set)
                self.assertNotIn("+amx", feature_set)

    def test_runner_probe_uses_generic_code_and_rejects_an_incompatible_cpu(self):
        for supported, status in (("0", 0), ("1", 65)):
            with self.subTest(supported=supported):
                script = '''
                    source "$HELPER"
                    configure_linux_cpu_target x86_64 cpu
                    cc() {
                      [[ "$1" == -march=x86-64 ]] || return 1
                      local executable="${@: -1}"
                      printf '#!/bin/sh\\nexit %s\\n' "$PROBE_RESULT" > "$executable"
                      chmod +x "$executable"
                    }
                    verify_linux_cpu_target
                '''
                env = dict(os.environ, HELPER=str(ROOT / "scripts/linux-cpu-target.sh"),
                           MERERUN_LINUX_CPU_TARGET="x86-64-v3", PROBE_RESULT=supported)
                result = subprocess.run(["bash", "-ec", textwrap.dedent(script)], env=env, capture_output=True)
                self.assertEqual(result.returncode, status, result.stderr)


if __name__ == "__main__":
    unittest.main()
