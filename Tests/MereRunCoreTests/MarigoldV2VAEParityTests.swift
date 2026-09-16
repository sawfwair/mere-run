import Foundation
import MLX
import XCTest
@testable import MereRunCore

final class MarigoldV2VAEParityTests: MereRunCoreTestCase {
    func testInstalledVAEReferenceTensorsWhenRequested() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let root = environment["MERERUN_TEST_MARIGOLD_ROOT"],
              let fixture = environment["MERERUN_TEST_MARIGOLD_VAE_FIXTURE"],
              let output = environment["MERERUN_TEST_MARIGOLD_VAE_OUTPUT"] else {
            throw XCTSkip("Set MERERUN_TEST_MARIGOLD_ROOT, MERERUN_TEST_MARIGOLD_VAE_FIXTURE, and MERERUN_TEST_MARIGOLD_VAE_OUTPUT.")
        }
        let dtype: DType = environment["MERERUN_TEST_MARIGOLD_VAE_BF16"] == "1" ? .bfloat16 : .float32
        let resources = MarigoldV2Resources(rootURL: URL(fileURLWithPath: root))
        let configs = try MarigoldV2ModelConfigs.load(from: resources)
        let vae = QwenImageEditVAE(config: configs.vae)
        try HFSafetensorsWeightsLoader.applyWeights(
            url: resources.vaeWeightsURL, to: vae.underlyingVAE, dtype: dtype,
            verify: [.shapeMismatch], mapper: QwenImageEditVAE.weightMapper
        )
        MLX.eval(vae)
        let arrays = try MLX.loadArrays(url: URL(fileURLWithPath: fixture))
        let input = try XCTUnwrap(arrays["input"]).asType(dtype)
        let raw = vae.underlyingVAE.encodeImageUnscaled(input)
        let normalized = QwenImageEditVAE.normalizeLatents(raw, config: configs.vae)
        let decoded = vae.underlyingVAE.decodeImageUnscaled(raw)
        MLX.eval(raw, normalized, decoded)
        var captured = ["raw_mode": raw, "normalized": normalized, "base_decoded": decoded]
        try MarigoldV2VAEDecoder.applyIfPresent(url: resources.trainablesURL, to: vae)
        if dtype == .float32 {
            vae.update(parameters: vae.parameters().mapValues { $0.asType(.float32) })
        }
        let tuned = vae.underlyingVAE.decodeImageUnscaled(raw)
        MLX.eval(tuned)
        captured["tuned_decoded"] = tuned
        try MLX.save(arrays: captured.mapValues { $0.asType(.float32) }, url: URL(fileURLWithPath: output))
        for key in captured.keys.sorted() {
            let actual = try XCTUnwrap(captured[key]).asType(.float32)
            if let expected = arrays[key] {
                let error = MLX.mean(MLX.abs(actual - expected)).item(Float.self)
                let maximum = MLX.max(MLX.abs(actual - expected)).item(Float.self)
                FileHandle.standardError.write(Data("marigold_vae \(key) mae=\(error) max=\(maximum)\n".utf8))
                XCTAssertLessThan(error, dtype == .float32 ? 0.00002 : 0.008, key)
            }
            XCTAssertTrue(MLX.all(MLX.isFinite(actual)).item(Bool.self))
        }
    }
}
