#!/usr/bin/env python3
"""Build isolated native and upstream qualification executables. Never edits vendor/."""
import argparse
import json
import os
from pathlib import Path
import subprocess

UPSTREAM = "49b9033e44ab0fbb935973eca3213757644d1bbb"
REPO = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path, help="external scratch directory (builds can exceed 2 GB)")
args = parser.parse_args()
root = args.output.resolve()
root.mkdir(parents=True, exist_ok=True)

def run(argv, **kwargs):
    subprocess.run([str(x) for x in argv], check=True, **kwargs)

mlx = REPO / ".build/checkouts/mlx-swift"
if not mlx.is_dir():
    run(["swift", "package", "--disable-keychain", "resolve"], cwd=REPO)
native = root / "native-harness"
(native / "Sources").mkdir(parents=True, exist_ok=True)
model_link = native / "Sources/AudioPhononModel"
if not model_link.is_symlink():
    model_link.symlink_to(REPO / "Sources/AudioPhononModel", target_is_directory=True)
executable_sources = native / "Sources/PhononQualification"
if executable_sources.is_symlink():
    executable_sources.unlink()
executable_sources.mkdir(exist_ok=True)
main_link = executable_sources / "main.swift"
if not main_link.is_symlink():
    main_link.symlink_to(REPO / "scripts/phonon-qualification/main.swift")
manifest = '''// swift-tools-version: 6.1
import PackageDescription
let package = Package(
    name: "PhononQualification", platforms: [.macOS(.v15)],
    products: [.executable(name: "PhononQualification", targets: ["PhononQualification"])],
    dependencies: [.package(path: MLX_PATH)],
    targets: [
        .target(name: "AudioPhononModel", dependencies: [
            .product(name: "MLX", package: "mlx-swift"), .product(name: "MLXFast", package: "mlx-swift"),
            .product(name: "MLXNN", package: "mlx-swift"), .product(name: "MLXRandom", package: "mlx-swift")
        ], exclude: ["README.md"], swiftSettings: [.interoperabilityMode(.Cxx), .unsafeFlags(["-enable-testing"])]),
        .executableTarget(name: "PhononQualification", dependencies: ["AudioPhononModel"],
            sources: ["main.swift"], swiftSettings: [.interoperabilityMode(.Cxx)])
    ], cxxLanguageStandard: .cxx17
)
'''.replace("MLX_PATH", json.dumps(str(mlx)))
(native / "Package.swift").write_text(manifest)
run(["swift", "build", "--package-path", native, "--disable-keychain", "--disable-index-store",
     "-c", "release", "--product", "PhononQualification"])
metallib = native / ".build/release/mlx.metallib"
if not metallib.exists():
    metallib.symlink_to(REPO / "vendor/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib")
upstream = root / "upstream-source"
if not upstream.exists():
    run(["git", "clone", "https://github.com/gradium-ai/xn-ptts.git", upstream])
    run(["git", "checkout", "--detach", UPSTREAM], cwd=upstream)
head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=upstream, text=True).strip()
if head != UPSTREAM:
    raise SystemExit(f"Expected upstream {UPSTREAM}; found {head}")
# Register an additional example only. The upstream library and official bench stay unmodified.
manifest_path = upstream / "ptts/Cargo.toml"
manifest = manifest_path.read_text()
if 'name = "mere_qualify"' not in manifest:
    manifest_path.write_text(manifest + '\n[[example]]\nname = "mere_qualify"\nrequired-features = ["hf"]\n')
(upstream / "ptts/examples/mere_qualify.rs").write_text((REPO / "scripts/phonon-qualification/upstream.rs").read_text())
build_env = dict(os.environ, CARGO_TARGET_DIR=str(root / "upstream-target"))
run(["cargo", "build", "--locked", "--release", "-p", "ptts", "--features", "hf,accelerate,kai",
     "--example", "bench", "--example", "mere_qualify"], cwd=upstream, env=build_env)
print(json.dumps({"native": str(native / ".build/release/PhononQualification"),
                  "upstream": str(root / "upstream-target/release/examples/mere_qualify"),
                  "bench": str(root / "upstream-target/release/examples/bench")}, indent=2))
