import Foundation
import XCTest

final class LinuxNativeBridgeTests: XCTestCase {
    func testPackageManifestCanConsumePrebuiltMlxSwiftCudaArtifacts() throws {
        let package = try readRepositoryFile("Package.swift")

        XCTAssertTrue(package.contains("MERERUN_MLX_SWIFT_LINKAGE"))
        XCTAssertTrue(package.contains("cuda-prebuilt"))
        XCTAssertTrue(package.contains("MERERUN_MLX_SWIFT_BUILD_DIR"))
        XCTAssertTrue(package.contains("MERERUN_MLX_SWIFT_SOURCE_DIR"))
        XCTAssertTrue(package.contains("MERERUN_MLX_SWIFT_LINK_FLAGS"))
        XCTAssertTrue(package.contains("func mlxDependency(_ name: String) -> [Target.Dependency]"))
        XCTAssertTrue(package.contains("useLinuxPrebuiltMLX ? [] : [.product(name: name, package: \"mlx-swift\")]"))
        XCTAssertTrue(package.contains("prebuiltMLXLinkerSettings"))
        XCTAssertTrue(package.contains("-lMLXFast"))
        XCTAssertTrue(package.contains("$ORIGIN/lib"))
        XCTAssertTrue(package.contains("linuxNativeLlamaLibraryPath"))
        XCTAssertTrue(package.contains("linuxPackageSwiftSettings"))
        XCTAssertTrue(package.contains("-strict-concurrency=targeted"))
        XCTAssertFalse(
            package.contains(".product(name: \"MLXFast\", package: \"mlx-swift\")"),
            "Direct MLX product dependencies must go through mlxDependency so cuda-prebuilt mode can import CMake-built modules."
        )
    }

    func testLinuxNativePreparationExportsSwiftPMCudaBridgeEnvironment() throws {
        let script = try readRepositoryFile("scripts/prepare-linux-native.sh")

        XCTAssertTrue(script.contains("verify_mlx_swift_cuda_bridge"))
        XCTAssertTrue(script.contains("mlx_swift_cuda_link_flags"))
        XCTAssertTrue(script.contains("export MERERUN_MLX_SWIFT_LINKAGE=\"cuda-prebuilt\""))
        XCTAssertTrue(script.contains("export MERERUN_MLX_SWIFT_BUILD_DIR=\"$native_root/build/mlx-swift-cuda-smoke\""))
        XCTAssertTrue(script.contains("export MERERUN_MLX_SWIFT_SOURCE_DIR=\"$repo_root/.build/native/src/mlx-swift\""))
        XCTAssertTrue(script.contains("export MERERUN_MLX_SWIFT_LINK_FLAGS=\"$mlx_link_flags\""))
        XCTAssertTrue(script.contains("$mlx_cmake_build/MLX.swiftmodule"))
        XCTAssertTrue(script.contains("$mlx_cmake_build/libMLX.a"))
        XCTAssertTrue(script.contains("$mlx_cmake_build/_deps/mlx-c-build/libmlxc.a"))
        XCTAssertTrue(script.contains("$mlx_cmake_build/_deps/mlx-build/libmlx.a"))
        XCTAssertTrue(script.contains("$mlx_cmake_build/lib/libNumerics.a"))
        XCTAssertTrue(script.contains("detect_cuda_dependency_defaults"))
        XCTAssertTrue(script.contains("patch_mlx_cuda_jit_include_path"))
        XCTAssertTrue(script.contains("MERERUN_CUDA_CCCL_INCLUDE"))
        XCTAssertTrue(script.contains("MERERUN_CUDA_CCCL_INCLUDE_PATH"))
        XCTAssertTrue(script.contains("MERERUN_NATIVE_BUILD_JOBS"))
        XCTAssertTrue(script.contains("MERERUN_CUDA_ARCHITECTURES"))
        XCTAssertTrue(script.contains("require_cmake_at_least 3 25 0"))
        XCTAssertTrue(script.contains("MERERUN_LINUX_ACCEL=cuda requires cmake >= "))
        XCTAssertTrue(script.contains("SwiftPM Linux package fixes applied."))
        XCTAssertTrue(script.contains("CUDA builds use the CMake prebuilt bridge."))
        XCTAssertTrue(script.contains("-DCMAKE_CUDA_ARCHITECTURES=$MERERUN_CUDA_ARCHITECTURES"))
        XCTAssertTrue(
            script.contains("cmake --build \"$llama_build\" --config Release --parallel \"$build_jobs\"")
        )
        XCTAssertTrue(
            script.contains("cmake --build \"$mlx_cmake_build\" --parallel \"$build_jobs\"")
        )
        XCTAssertTrue(script.contains(#"return std::filesystem::path(\"/usr/local/cuda\");"#))
        XCTAssertFalse(script.contains("return default_cuda_toolkit_path();"))
        XCTAssertTrue(script.contains("detect_cuda_cccl_include_path"))
        XCTAssertTrue(script.contains("cuda_toolkit_root_candidates"))
        XCTAssertTrue(script.contains("Source/Cmlx/mlx/mlx/backend/cuda/jit_module.cpp"))
        XCTAssertTrue(script.contains("$mlx_cmake_build/_deps/mlx-src/mlx/backend/cuda/jit_module.cpp"))
        XCTAssertTrue(script.contains("19e92c33ef974661e4b1e43dd48be231d07be5ed"))
        XCTAssertTrue(script.contains("/usr/include/$deb_multiarch"))
        XCTAssertTrue(script.contains("/usr/lib/$deb_multiarch"))
        XCTAssertTrue(script.contains("CUDA_LIBRARY_PATH"))
        XCTAssertTrue(script.contains("$cuda_root/targets/$cuda_target/lib"))
        XCTAssertTrue(script.contains("aarch64-linux"))
        XCTAssertFalse(script.contains("whose Linux path remains CPU-oriented until a package bridge is added"))
    }

    func testLinuxArm64MLXBuildsSelectBF16CapableToolchain() throws {
        let packageScript = try readRepositoryFile("scripts/package-linux.sh")
        let checkScript = try readRepositoryFile("scripts/check-linux.sh")
        let prepareScript = try readRepositoryFile("scripts/prepare-linux-native.sh")
        let toolchainScript = try readRepositoryFile("scripts/linux-arm64-bf16-toolchain.sh")

        XCTAssertTrue(packageScript.contains("platform_arch=\"arm64\""))
        XCTAssertTrue(packageScript.contains("deb_arch=\"arm64\""))
        XCTAssertTrue(packageScript.contains("deb_multiarch=\"aarch64-linux-gnu\""))
        XCTAssertTrue(packageScript.contains("MERERUN_LINUX_ALLOW_ARM64_CPU_PACKAGE"))
        XCTAssertTrue(packageScript.contains("Linux arm64 release packages must use MERERUN_LINUX_ACCEL=cuda"))
        XCTAssertTrue(packageScript.contains("CUDA_LIBRARY_PATH"))
        XCTAssertTrue(packageScript.contains("$cuda_root/targets/$cuda_target/lib"))
        XCTAssertTrue(packageScript.contains("aarch64-linux"))
        XCTAssertTrue(packageScript.contains("/usr/local/cuda-*"))
        XCTAssertTrue(packageScript.contains("cuda_cccl_include"))
        XCTAssertTrue(packageScript.contains("MERERUN_CUDA_CCCL_INCLUDE_PATH"))
        XCTAssertTrue(packageScript.contains("CPATH=\"$cuda_cccl_include"))
        XCTAssertTrue(packageScript.contains(".mererun-linux-cuda"))
        XCTAssertTrue(packageScript.contains("MERERUN_LLAMA_CLI"))
        XCTAssertTrue(packageScript.contains("$llama_prefix/bin/llama-cli"))
        XCTAssertTrue(packageScript.contains("cuda_root/include/cccl"))
        XCTAssertTrue(packageScript.contains("cuda_root/targets/$cuda_target/include/cccl"))
        XCTAssertTrue(packageScript.contains("cuda-cccl-13-0"))
        XCTAssertTrue(packageScript.contains("cuda-cudart-13-0"))
        XCTAssertTrue(packageScript.contains("cuda-nvrtc-13-0"))
        XCTAssertTrue(packageScript.contains("libcublas-13-0"))
        XCTAssertTrue(packageScript.contains("libcudnn9-cuda-13"))
        XCTAssertTrue(packageScript.contains("libnccl2"))
        XCTAssertTrue(
            packageScript.contains("configure_linux_arm64_bf16_toolchain \"$platform_arch\" \"package-linux\"")
        )
        XCTAssertTrue(
            checkScript.contains("configure_linux_arm64_bf16_toolchain \"$arch\" \"check-linux\"")
        )
        XCTAssertTrue(
            prepareScript.contains("configure_linux_arm64_bf16_toolchain \"$arch\" \"prepare-linux-native\"")
        )
        XCTAssertTrue(toolchainScript.contains("#include <arm_bf16.h>"))
        XCTAssertTrue(toolchainScript.contains("#include <string>"))
        XCTAssertTrue(toolchainScript.contains(".build/toolchains/linux-arm64-bf16"))
        XCTAssertTrue(toolchainScript.contains("clang++"))
        XCTAssertTrue(toolchainScript.contains("clang-17"))
    }

    func testLinuxReleaseWorkflowKeepsArm64OnCudaRunner() throws {
        let workflow = try readRepositoryFile(".github/workflows/linux-release.yml")
        let packageTest = try readRepositoryFile("scripts/test-package-linux.sh")

        XCTAssertTrue(workflow.contains("runs-on: ubuntu-22.04"))
        XCTAssertFalse(workflow.contains("ubuntu-22.04-arm"))
        XCTAssertTrue(workflow.contains("Build Linux x86_64"))
        XCTAssertTrue(workflow.contains("Build Linux arm64 CUDA"))
        XCTAssertTrue(workflow.contains("runs-on: [self-hosted, linux, arm64, cuda]"))
        XCTAssertTrue(workflow.contains("MERERUN_LINUX_ACCEL: cuda"))
        XCTAssertTrue(workflow.contains("cuda-cccl-13-0"))
        XCTAssertTrue(workflow.contains("libcudnn9-dev-cuda-13"))
        XCTAssertTrue(workflow.contains("libnccl-dev"))
        XCTAssertTrue(workflow.contains("python3-pip"))
        XCTAssertTrue(workflow.contains("cmake>=3.25,<4"))
        XCTAssertTrue(workflow.contains("MERERUN_RELEASE_ARM64_CUDA == '1'"))
        XCTAssertTrue(workflow.contains("build_arm64_cuda"))
        XCTAssertTrue(workflow.contains("pattern: mere-run-linux-*-packages"))
        XCTAssertTrue(workflow.contains("Build combined checksum manifest"))
        XCTAssertTrue(workflow.contains("files: dist/linux/*"))
        XCTAssertTrue(packageTest.contains("platform_arch=\"arm64\""))
        XCTAssertTrue(packageTest.contains("linux-${platform_arch}.tar.gz"))
        XCTAssertTrue(packageTest.contains("MERERUN_LINUX_ALLOW_ARM64_CPU_PACKAGE=1"))
    }

    private func readRepositoryFile(_ relativePath: String) throws -> String {
        let url = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(relativePath, isDirectory: false)
        return try String(contentsOf: url, encoding: .utf8)
    }
}
