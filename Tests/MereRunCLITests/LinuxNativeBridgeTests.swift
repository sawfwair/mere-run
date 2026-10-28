import Foundation
import XCTest

final class LinuxNativeBridgeTests: XCTestCase {
    func testQuantizedKernelPatchIsIdempotentAndRejectsSourceDrift() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let kernel = root.appendingPathComponent("device/qmm_sm80.cuh")
        let launcher = root.appendingPathComponent("quantized/qmm/qmm_sm80.cu")
        let qmv = root.appendingPathComponent("quantized/qmm/qmv.cu")
        try FileManager.default.createDirectory(at: kernel.deletingLastPathComponent(), withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: launcher.deletingLastPathComponent(), withIntermediateDirectories: true)
        try "  // Epilogue.\n  copy(r2s_copy_c, r2s_tCrC, r2s_tCsC);\n".write(to: kernel, atomically: true, encoding: .utf8)
        try "      \"qmm_sm80_tn_{}_m{}_b{}_g{}_{}\",\n".write(to: launcher, atomically: true, encoding: .utf8)
        try "  cuda::std::conditional_t<(bits == 1 || bits >= 8), float, T>\n      sums[elems_per_thread] = {};\n"
            .write(to: qmv, atomically: true, encoding: .utf8)
        // SwiftPM dependency files are read-only even when their directories are writable.
        for file in [kernel, launcher, qmv] {
            try FileManager.default.setAttributes([.posixPermissions: 0o444], ofItemAtPath: file.path)
        }
        let script = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("scripts/patch-mlx-cuda-quantized.sh")
        let apply = { () throws -> Int32 in
            let process = Process()
            process.executableURL = URL(fileURLWithPath: "/bin/bash")
            process.arguments = [script.path, root.path]
            process.standardOutput = FileHandle.nullDevice
            process.standardError = FileHandle.nullDevice
            try process.run()
            process.waitUntilExit()
            return process.terminationStatus
        }
        XCTAssertEqual(try apply(), 0)
        let patched = try String(contentsOf: kernel, encoding: .utf8)
        let wait = try XCTUnwrap(patched.range(of: "cp_async_wait<0>();"))
        let barrier = try XCTUnwrap(patched.range(of: "__syncthreads();"))
        let output = try XCTUnwrap(patched.range(of: "copy(r2s_copy_c"))
        XCTAssertLessThan(wait.lowerBound, barrier.lowerBound)
        XCTAssertLessThan(barrier.lowerBound, output.lowerBound)
        let patchedLauncher = try String(contentsOf: launcher, encoding: .utf8)
        let patchedQMV = try String(contentsOf: qmv, encoding: .utf8)
        XCTAssertTrue(patchedQMV.contains("cuda::std::is_same_v<T, cutlass::bfloat16_t>"))
        XCTAssertTrue(patchedQMV.contains("MERERUN_CUDA_BF16_QMV_FLOAT_ACCUMULATION"))
        XCTAssertTrue(patchedLauncher.contains("mr_epilogue1"))
        XCTAssertEqual(try apply(), 0)
        XCTAssertEqual(try String(contentsOf: kernel, encoding: .utf8), patched)
        XCTAssertEqual(try String(contentsOf: launcher, encoding: .utf8), patchedLauncher)
        XCTAssertEqual(try String(contentsOf: qmv, encoding: .utf8), patchedQMV)

        let changed = "  // Upstream layout changed.\n"
        try changed.write(to: kernel, atomically: true, encoding: .utf8)
        XCTAssertEqual(try apply(), 69)
        XCTAssertEqual(try String(contentsOf: kernel, encoding: .utf8), changed)
        XCTAssertEqual(try String(contentsOf: launcher, encoding: .utf8), patchedLauncher)
        XCTAssertEqual(try String(contentsOf: qmv, encoding: .utf8), patchedQMV)
    }

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
        XCTAssertTrue(script.contains("patch_mlx_cuda_subbyte_qmm_pointer_step"))
        XCTAssertTrue(script.contains("patch_mlx_cpu_jit_f16c_probe"))
        XCTAssertTrue(script.contains("MERERUN_MLX_X86_AVX2_IMPLIES_F16C"))
        XCTAssertTrue(script.contains("MERERUN_MLX_CUDA_JIT_INCLUDE"))
        XCTAssertTrue(script.contains("MERERUN_MLX_CUDA_JIT_INCLUDE_PATH"))
        XCTAssertTrue(script.contains("MERERUN_CUDA_CCCL_INCLUDE"))
        XCTAssertTrue(script.contains("MERERUN_CUDA_CCCL_INCLUDE_PATH"))
        XCTAssertTrue(script.contains("MERERUN_CUDA_SUBBYTE_QMM_POINTER_STEP"))
        XCTAssertTrue(script.contains("sizeof_bits_v<Quant> < 8 ? sizeof_bits_v<Quant> : 8"))
        XCTAssertTrue(script.contains("MERERUN_NATIVE_BUILD_JOBS"))
        XCTAssertTrue(script.contains("MERERUN_CUDA_ARCHITECTURES"))
        XCTAssertTrue(script.contains("MLX_SWIFT_CUDA_URL"))
        XCTAssertTrue(script.contains("https://github.com/sawfwair/mlx-swift.git"))
        XCTAssertTrue(script.contains("submodule update --init --depth 1"))
        XCTAssertTrue(script.contains("Source/Cmlx/mlx Source/Cmlx/mlx-c"))
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
        XCTAssertTrue(script.contains("Source/Cmlx/mlx/mlx/backend/cuda/device/qmm_naive.cuh"))
        XCTAssertTrue(script.contains("Source/Cmlx/mlx/mlx/backend/cpu/jit_compiler.cpp"))
        XCTAssertTrue(script.contains("$mlx_cmake_build/_deps/mlx-src/mlx/backend/cuda/jit_module.cpp"))
        XCTAssertTrue(script.contains("4988f6e866057afd130c1515ecef0c9bab9a15f8"))
        XCTAssertTrue(script.contains("-DLLAMA_BUILD_COMMON=OFF"))
        XCTAssertTrue(script.contains("-DLLAMA_BUILD_TOOLS=OFF"))
        XCTAssertTrue(script.contains("-DLLAMA_BUILD_SERVER=OFF"))
        XCTAssertTrue(script.contains("-DLLAMA_BUILD_APP=OFF"))
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
        XCTAssertTrue(packageScript.contains("MERERUN_MLX_CUDA_JIT_INCLUDE_DIR"))
        XCTAssertTrue(packageScript.contains("MERERUN_MLX_CUDA_JIT_INCLUDE_PATH"))
        XCTAssertTrue(packageScript.contains("$mlx_cuda_jit_include_dir/cute"))
        XCTAssertTrue(packageScript.contains("$mlx_cuda_jit_include_dir/cutlass"))
        XCTAssertTrue(packageScript.contains("CUTLASS-LICENSE.txt"))
        XCTAssertTrue(packageScript.contains("CPATH=\"$cuda_cccl_include"))
        XCTAssertTrue(packageScript.contains(".mererun-linux-cuda"))
        XCTAssertTrue(packageScript.contains("cuda_runtime_root"))
        XCTAssertTrue(packageScript.contains("export CUDA_HOME=\"$cuda_runtime_root\""))
        XCTAssertTrue(packageScript.contains("export CUDA_PATH=\"$CUDA_HOME\""))
        XCTAssertTrue(packageScript.contains("cuda_runtime_library_candidates"))
        XCTAssertTrue(packageScript.contains("$cuda_root/targets/$cuda_target/lib"))
        XCTAssertTrue(packageScript.contains("LD_LIBRARY_PATH=\"$LD_LIBRARY_PATH:$cuda_runtime_library\""))
        XCTAssertTrue(packageScript.contains("MERERUN_LLAMA_CLI"))
        XCTAssertTrue(packageScript.contains("$llama_prefix/bin/llama-cli"))
        XCTAssertTrue(packageScript.contains(".mererun-package-id"))
        XCTAssertTrue(packageScript.contains("if [[ -z \"${MLX_PTX_CACHE_DIR:-}\" ]]; then"))
        XCTAssertTrue(
            packageScript.contains(
                "export MLX_PTX_CACHE_DIR=\"${XDG_CACHE_HOME:-${HOME:-/tmp}/.cache}/mere.run/mlx-ptx/$mererun_package_id\""
            )
        )
        XCTAssertTrue(packageScript.contains("cuda_root/include/cccl"))
        XCTAssertTrue(packageScript.contains("cuda_root/targets/$cuda_target/include/cccl"))
        XCTAssertTrue(packageScript.contains("cuda-cccl-12-8 | libcu++-dev"))
        XCTAssertTrue(packageScript.contains("cuda-cudart-dev-12-8"))
        XCTAssertTrue(packageScript.contains("cuda-cudart-12-8 | libcudart12"))
        XCTAssertTrue(packageScript.contains("cuda-nvrtc-12-8 | libnvrtc12"))
        XCTAssertTrue(packageScript.contains("libcublas-12-8 | libcublas12"))
        XCTAssertTrue(packageScript.contains("libcufft-12-8 | libcufft11"))
        XCTAssertTrue(packageScript.contains("libcudnn9-cuda-12 | python3-torch-cuda"))
        XCTAssertTrue(packageScript.contains("cuda-cccl-13-0"))
        XCTAssertTrue(packageScript.contains("cuda-cudart-dev-13-0"))
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

    func testReleaseArtifactsAreNotBuiltByPublicGitHubActions() throws {
        let packageTest = try readRepositoryFile("scripts/test-package-linux.sh")
        let packageScript = try readRepositoryFile("scripts/package-linux.sh")
        let macOSPackageScript = try readRepositoryFile("scripts/package-macos.sh")

        XCTAssertFalse(repositoryFileExists(".github/workflows/linux-release.yml"))
        XCTAssertFalse(repositoryFileExists(".github/workflows/macos-release.yml"))
        XCTAssertTrue(packageScript.contains("Build Linux release artifacts for the headless mere.run CLI."))
        XCTAssertTrue(packageScript.contains("--artifact-suffix NAME"))
        XCTAssertTrue(packageScript.contains("MERERUN_LINUX_ACCEL"))
        XCTAssertTrue(packageScript.contains("MERERUN_LINUX_ACCEL=cuda"))
        XCTAssertTrue(packageScript.contains("MERERUN_LINUX_ALLOW_ARM64_CPU_PACKAGE"))
        XCTAssertTrue(macOSPackageScript.contains("scripts/build_mere_run_app.sh"))
        XCTAssertTrue(packageTest.contains("platform_arch=\"arm64\""))
        XCTAssertTrue(packageTest.contains("linux-${platform_arch}.tar.gz"))
        XCTAssertTrue(packageTest.contains("MERERUN_LINUX_ALLOW_ARM64_CPU_PACKAGE=1"))
        XCTAssertTrue(packageTest.contains("mere.run/mlx-ptx/mere-run-0.0.0+cuda-deps-fixture-linux-${platform_arch}-cuda"))
        XCTAssertTrue(packageTest.contains("MLX_PTX_CACHE_DIR=/custom/ptx"))
    }

    func testMacOSPackageEmbedsTheStapledAppBeforeCreatingTheDMG() throws {
        let script = try readRepositoryFile("scripts/package-macos.sh")
        let appNotarize = try XCTUnwrap(script.range(of: "notarize \"$app_zip_path\""))
        let appStaple = try XCTUnwrap(script.range(of: "xcrun stapler staple \"$bundle\""))
        let appValidate = try XCTUnwrap(script.range(of: "xcrun stapler validate \"$bundle\""))
        let bundleCopy = try XCTUnwrap(script.range(of: "cp -R \"$bundle\" \"$staging/\""))
        let imageCreate = try XCTUnwrap(script.range(of: "hdiutil create"))
        let imageNotarize = try XCTUnwrap(script.range(of: "notarize \"$dmg_path\""))
        let imageValidate = try XCTUnwrap(script.range(of: "xcrun stapler validate \"$dmg_path\""))

        XCTAssertLessThan(appNotarize.lowerBound, appStaple.lowerBound)
        XCTAssertLessThan(appStaple.lowerBound, appValidate.lowerBound)
        XCTAssertLessThan(appValidate.lowerBound, bundleCopy.lowerBound)
        XCTAssertLessThan(bundleCopy.lowerBound, imageCreate.lowerBound)
        XCTAssertLessThan(imageCreate.lowerBound, imageNotarize.lowerBound)
        XCTAssertLessThan(imageNotarize.lowerBound, imageValidate.lowerBound)
        XCTAssertFalse(script.contains("if notarize \"$app_zip_path\""))
        XCTAssertFalse(script.contains("if notarize \"$dmg_path\""))
    }

    func testMacOSAppBundlesAndSecurelyConfiguresSparkle() throws {
        let package = try readRepositoryFile("Package.swift")
        let script = try readRepositoryFile("scripts/build_mere_run_app.sh")

        XCTAssertTrue(package.contains(#".product(name: "Sparkle", package: "Sparkle")"#))
        XCTAssertTrue(package.contains(#"sparkle-project/Sparkle", exact: "2.9.5""#))
        XCTAssertTrue(script.contains(#"sparkle_expected_version="2.9.5""#))
        XCTAssertTrue(script.contains("CFBundleShortVersionString raw"))
        XCTAssertTrue(script.contains("Expected Sparkle ${sparkle_expected_version}"))
        XCTAssertTrue(script.contains("ditto \"$sparkle_framework\" \"${frameworks}/Sparkle.framework\""))
        XCTAssertTrue(script.contains("SUFeedURL"))
        XCTAssertTrue(script.contains("https://mere.run/releases/appcast.xml"))
        XCTAssertTrue(script.contains("git describe --tags --exact-match"))
        XCTAssertTrue(script.contains(#"sparkle_public_ed_key="6sFs+7UqYcE7rThPAovzMDsZtKyf/h4/d8rUmPSH2rw=""#))
        XCTAssertTrue(script.contains("SUEnableAutomaticChecks"))
        XCTAssertTrue(script.contains("SUVerifyUpdateBeforeExtraction"))
        XCTAssertTrue(script.contains("SURequireSignedFeed"))
        XCTAssertTrue(script.contains("NSMicrophoneUsageDescription"))
        XCTAssertTrue(script.contains("local voice references and transcription input"))
        XCTAssertTrue(script.contains("CFBundleURLTypes"))
        XCTAssertTrue(script.contains(#""CFBundleURLSchemes":["mererun"]"#))
        XCTAssertTrue(script.contains("MereRunDebug.entitlements"))
        XCTAssertTrue(script.contains(#"if [[ "$identity" == "-" ]]"#))
        XCTAssertTrue(script.contains(#"-ffile-prefix-map=${source_path_map}"#))
        XCTAssertTrue(script.contains(#"-Xswiftc -file-prefix-map"#))
        let releaseBuildGuard = try XCTUnwrap(
            script.range(
                of: #"""
                if [[ "$configuration" == "release" ]]; then
                  # Only distributable builds need checkout-path sanitization.
                """#
            )
        )
        let releaseBuildEnd = try XCTUnwrap(
            script.range(
                of: #"""
                fi

                swift "${swift_app_args[@]}"
                """#,
                range: releaseBuildGuard.lowerBound..<script.endIndex
            )
        )
        let releaseBuildBlock = script[releaseBuildGuard.lowerBound..<releaseBuildEnd.upperBound]
        XCTAssertTrue(releaseBuildBlock.contains(#"-ffile-prefix-map=${source_path_map}"#))
        XCTAssertTrue(releaseBuildBlock.contains(#"-Xswiftc -file-prefix-map"#))
        XCTAssertTrue(releaseBuildBlock.contains(#"swift_app_args+=(--configuration release)"#))
        XCTAssertEqual(script.components(separatedBy: #"swift_app_args+=("${swift_path_map_args[@]}")"#).count, 2)
        XCTAssertTrue(script.contains("MERERUN_SWIFT_SCRATCH_PATH"))
        XCTAssertTrue(script.contains(#"${swift_scratch_path}=/src/mere-run/.build"#))
        XCTAssertTrue(script.contains("verify_private_build_path_absent"))
        XCTAssertTrue(script.contains("Private source path leaked"))
        XCTAssertTrue(script.contains(#"if [[ "$configuration" == "release" ]]; then"#))
        XCTAssertTrue(script.contains("--preserve-metadata=entitlements"))
        XCTAssertTrue(script.contains("XPCServices/Installer.xpc"))
        XCTAssertTrue(script.contains("XPCServices/Downloader.xpc"))
        XCTAssertTrue(script.contains(#""${sparkle_version_root}/Autoupdate""#))
        XCTAssertTrue(script.contains(#""${sparkle_version_root}/Updater.app""#))

        for entitlementsPath in [
            "scripts/MereRun.entitlements",
            "scripts/MereRunDebug.entitlements",
            "scripts/MereRunCLI.entitlements",
        ] {
            let entitlements = try readRepositoryFile(entitlementsPath)
            XCTAssertTrue(
                entitlements.contains("com.apple.security.device.audio-input"),
                "\(entitlementsPath) must authorize microphone capture"
            )
        }
    }

    private func readRepositoryFile(_ relativePath: String) throws -> String {
        let url = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(relativePath, isDirectory: false)
        return try String(contentsOf: url, encoding: .utf8)
    }

    private func repositoryFileExists(_ relativePath: String) -> Bool {
        let url = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(relativePath, isDirectory: false)
        return FileManager.default.fileExists(atPath: url.path)
    }
}
