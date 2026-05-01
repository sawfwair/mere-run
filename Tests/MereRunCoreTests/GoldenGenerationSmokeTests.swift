import CryptoKit
import Foundation
import XCTest
@testable import MereRunCore

final class GoldenGenerationSmokeTests: XCTestCase {

    private func sha256Hex(_ data: Data) -> String {
        let digest = SHA256.hash(data: data)
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    private func shouldRunGoldens(_ env: [String: String]) -> Bool {
        let raw = (env["MERERUN_TEST_RUN_GOLDENS"] ?? "").lowercased()
        return raw == "1" || raw == "true" || raw == "yes"
    }

    private func isGPUEnabled(_ env: [String: String]) -> Bool {
        (env["MERERUN_TEST_MLX_DEVICE"] ?? "").lowercased() == "gpu"
    }

    func testZetaGenerateSmoke() async throws {
        let env = ProcessInfo.processInfo.environment
        guard shouldRunGoldens(env) else { throw XCTSkip("Set MERERUN_TEST_RUN_GOLDENS=1 to enable golden smoke tests.") }
        guard isGPUEnabled(env) else { throw XCTSkip("Set MERERUN_TEST_MLX_DEVICE=gpu to run golden smoke tests.") }
        guard let modelRoot = env["MERERUN_TEST_ZETA_MODEL_ROOT"], !modelRoot.isEmpty else {
            throw XCTSkip("Set MERERUN_TEST_ZETA_MODEL_ROOT to a local Zeta model root to run this test.")
        }

        MLXTestSupport.ensureMetalLibraryAvailable()

        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let outputURL = temp.appendingPathComponent("zeta-golden.png")
        let request = GenerationRequest(
            prompt: "a red apple on a wooden table, studio lighting",
            width: 1024,
            height: 1024,
            steps: 4,
            guidanceScale: 4.0,
            seed: 0,
            outputURL: outputURL,
            model: modelRoot
        )

        let generator = ZImageTurboGenerator()
        let result = try await generator.generate(request)

        XCTAssertTrue(FileManager.default.fileExists(atPath: result.outputURL.path))
        let data = try Data(contentsOf: result.outputURL)
        XCTAssertFalse(data.isEmpty)

        if let expected = env["MERERUN_TEST_ZETA_SHA256"], !expected.isEmpty {
            XCTAssertEqual(sha256Hex(data), expected.lowercased())
        }
    }

    func testFlux2GenerateSmoke() async throws {
        let env = ProcessInfo.processInfo.environment
        guard shouldRunGoldens(env) else { throw XCTSkip("Set MERERUN_TEST_RUN_GOLDENS=1 to enable golden smoke tests.") }
        guard isGPUEnabled(env) else { throw XCTSkip("Set MERERUN_TEST_MLX_DEVICE=gpu to run golden smoke tests.") }
        guard let modelRoot = env["MERERUN_TEST_MODEL_ROOT"], !modelRoot.isEmpty else {
            throw XCTSkip("Set MERERUN_TEST_MODEL_ROOT to a local mere.run model root to run this test.")
        }

        MLXTestSupport.ensureMetalLibraryAvailable()

        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let outputURL = temp.appendingPathComponent("mererun-golden.png")
        let request = GenerationRequest(
            prompt: "a red apple on a wooden table, studio lighting",
            width: 1024,
            height: 1024,
            steps: 4,
            guidanceScale: 1.0,
            seed: 0,
            outputURL: outputURL,
            model: modelRoot
        )

        let generator = Flux2KleinGenerator()
        let result = try await generator.generate(request)

        XCTAssertTrue(FileManager.default.fileExists(atPath: result.outputURL.path))
        let data = try Data(contentsOf: result.outputURL)
        XCTAssertFalse(data.isEmpty)

        if let expected = env["MERERUN_TEST_FLUX2_SHA256"], !expected.isEmpty {
            XCTAssertEqual(sha256Hex(data), expected.lowercased())
        }
    }
}
