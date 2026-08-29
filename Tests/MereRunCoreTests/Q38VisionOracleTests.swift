import Foundation
import MediaIO
import MLX
import MLXNN
import XCTest
@testable import MereRunCore

/// Exports only the installed vision tower, never the language model. The
/// exact preprocessed pixels allow an independent backend to inspect errors.
final class Q38VisionOracleTests: MereRunCoreTestCase {
    private enum Precision: String { case bf16, fp32 }

    func testInstalledVisionEmbeddingExport() throws {
        let environment = ProcessInfo.processInfo.environment
        try XCTSkipUnless(environment["MERERUN_TEST_Q38_VISION_ORACLE"] == "1",
                          "Installed vision reference export is opt-in.")
        let root = URL(fileURLWithPath: try XCTUnwrap(environment["MERERUN_TEST_Q38_FLASH_NEXT_MODEL_ROOT"]))
        let source = URL(fileURLWithPath: try XCTUnwrap(environment["MERERUN_TEST_Q38_VISION_ORACLE_IMAGE"]))
        let output = URL(fileURLWithPath: try XCTUnwrap(environment["MERERUN_TEST_Q38_VISION_ORACLE_OUTPUT"]))
        try XCTSkipIf(FileManager.default.fileExists(atPath: output.path), "Do not overwrite reference evidence")
        let config = try JSONDecoder().decode(
            Q35Config.self, from: Data(contentsOf: root.appendingPathComponent("config.json"))
        )
        let tower = Q35VisionTower(config: config)
        try tower.loadWeights(from: Q35Resources(rootURL: root))
        let precision = try XCTUnwrap(Precision(rawValue: environment["MERERUN_TEST_Q38_VISION_ORACLE_DTYPE"] ?? "bf16"))
        if precision == .fp32 {
            tower.update(parameters: tower.parameters().mapValues { $0.asType(.float32) })
        }
        let image = try MediaImageIO.decode(source)
        let bounds = Q35Resources.visionPixelBounds(forModelId: Q35Resources.q38FlashNextMixedModelId)
        let size = try Q35Generator.qwen3VLTargetSize(
            originalWidth: image.width, originalHeight: image.height,
            patchSize: tower.patchSize, spatialMergeSize: tower.spatialMergeSize,
            minPixels: bounds.minimum, maxPixels: bounds.maximum
        )
        let resized = try MediaImageIO.resized(image, width: size.width, height: size.height)
        let pixels = MLXArray(
            MediaImageIO.rgbCHWFloat(resized, normalizedToMinusOneToOne: true),
            [1, 3, resized.height, resized.width]
        )
        let grid = (1, resized.height / tower.patchSize, resized.width / tower.patchSize)
        let embeddings = try tower.encodeImage(pixelValues: pixels, gridTHW: grid)
        MLX.eval(embeddings)
        XCTAssertTrue(embeddings.asArray(Float.self).allSatisfy(\.isFinite))
        try MLX.save(arrays: [
            "pixels_chw": pixels,
            "grid_thw": MLXArray([Int32(grid.0), Int32(grid.1), Int32(grid.2)], [1, 3]),
            "native_embeddings": embeddings,
        ], url: output)
        print("[q38-vision-oracle] shape=\(embeddings.shape) dtype=\(embeddings.dtype) output=\(output.path)")
    }
}
