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
        XCTAssertFalse(script.contains("whose Linux path remains CPU-oriented until a package bridge is added"))
    }

    private func readRepositoryFile(_ relativePath: String) throws -> String {
        let url = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(relativePath, isDirectory: false)
        return try String(contentsOf: url, encoding: .utf8)
    }
}
