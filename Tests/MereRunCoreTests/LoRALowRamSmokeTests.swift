import Foundation
import XCTest
@testable import MereRunCore

final class LoRALowRamSmokeTests: MereRunCoreTestCase {

    private func shouldRunLowRamSmoke(_ env: [String: String]) -> Bool {
        let raw = (env["MERERUN_TEST_RUN_LORA_SMOKE"] ?? "").lowercased()
        return raw == "1" || raw == "true" || raw == "yes"
    }

    private func isGPUEnabled(_ env: [String: String]) -> Bool {
        (env["MERERUN_TEST_MLX_DEVICE"] ?? "").lowercased() == "gpu"
    }

    private func writeTinyPNG(to url: URL) throws {
        // 1x1 PNG (white pixel)
        let base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+Xn6cAAAAASUVORK5CYII="
        guard let data = Data(base64Encoded: base64) else {
            XCTFail("Failed to decode embedded PNG fixture.")
            return
        }
        try TestFileSystem.writeFile(url, contents: data)
    }

    private func loadManifest(nextTo outputURL: URL) throws -> LoRATrainingManifest {
        let manifestURL = LoRATrainingManifest.url(nextTo: outputURL)
        let data = try Data(contentsOf: manifestURL)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try decoder.decode(LoRATrainingManifest.self, from: data)
    }

    func testFlux2LowRamTrainingSmoke() async throws {
        let env = ProcessInfo.processInfo.environment
        guard shouldRunLowRamSmoke(env) else { throw XCTSkip("Set MERERUN_TEST_RUN_LORA_SMOKE=1 to enable low-ram LoRA smoke tests.") }
        guard isGPUEnabled(env) else { throw XCTSkip("Set MERERUN_TEST_MLX_DEVICE=gpu to run low-ram LoRA smoke tests.") }
        guard let modelRoot = env["MERERUN_TEST_MODEL_ROOT"], !modelRoot.isEmpty else {
            throw XCTSkip("Set MERERUN_TEST_MODEL_ROOT to a local mere.run model root to run this test.")
        }

        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let imageURL = temp.appendingPathComponent("flux-smoke.png")
        try writeTinyPNG(to: imageURL)

        let outputURL = temp.appendingPathComponent("flux-low-ram.safetensors")
        let examples = [
            Flux2KleinLoRATrainingExample(imageURL: imageURL, caption: "test subject")
        ]

        var config = Flux2KleinLoRATrainingConfig()
        config.width = 256
        config.height = 256
        config.maxTextLength = 64
        config.schedulerSteps = 16
        config.trainingSteps = 1
        config.batchSize = 1
        config.learningRate = 2e-4
        config.loraRank = 4
        config.loraAlpha = 1.0
        config.captionDropout = 0
        config.logEvery = 1
        config.progressive = false
        config.useCompile = false
        config.lowRam = true

        try await Flux2KleinLoRATrainer.train(
            modelPath: modelRoot,
            examples: examples,
            outputURL: outputURL,
            config: config
        )

        XCTAssertTrue(FileManager.default.fileExists(atPath: outputURL.path))
        let manifest = try loadManifest(nextTo: outputURL)
        XCTAssertEqual(manifest.format, "mererun.flux2.lora")
        XCTAssertEqual(manifest.training.datasetCount, 1)
        XCTAssertEqual(manifest.extras?["low_ram"], "true")
    }

    func testZImageLowRamTrainingSmoke() async throws {
        let env = ProcessInfo.processInfo.environment
        guard shouldRunLowRamSmoke(env) else { throw XCTSkip("Set MERERUN_TEST_RUN_LORA_SMOKE=1 to enable low-ram LoRA smoke tests.") }
        guard isGPUEnabled(env) else { throw XCTSkip("Set MERERUN_TEST_MLX_DEVICE=gpu to run low-ram LoRA smoke tests.") }
        guard let modelRoot = env["MERERUN_TEST_ZETA_MODEL_ROOT"], !modelRoot.isEmpty else {
            throw XCTSkip("Set MERERUN_TEST_ZETA_MODEL_ROOT to a local Zeta model root to run this test.")
        }

        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let imageURL = temp.appendingPathComponent("zimage-smoke.png")
        try writeTinyPNG(to: imageURL)

        let outputURL = temp.appendingPathComponent("zimage-low-ram.safetensors")
        let examples = [
            ZImageTurboLoRATrainingExample(imageURL: imageURL, caption: "test subject")
        ]

        var config = ZImageTurboLoRATrainingConfig()
        config.width = 256
        config.height = 256
        config.maxTextLength = 64
        config.schedulerSteps = 16
        config.trainingSteps = 1
        config.batchSize = 1
        config.learningRate = 1e-4
        config.loraRank = 4
        config.captionDropout = 0
        config.logEvery = 1
        config.lowRam = true
        config.useTrainingAdapter = false
        config.syntheticSampleCount = nil

        try await ZImageTurboLoRATrainer.train(
            modelPath: modelRoot,
            examples: examples,
            outputURL: outputURL,
            config: config
        )

        XCTAssertTrue(FileManager.default.fileExists(atPath: outputURL.path))
        let manifest = try loadManifest(nextTo: outputURL)
        XCTAssertEqual(manifest.format, "mererun.zimage.lora")
        XCTAssertEqual(manifest.training.datasetCount, 1)
        XCTAssertEqual(manifest.extras?["low_ram"], "true")
    }
}
