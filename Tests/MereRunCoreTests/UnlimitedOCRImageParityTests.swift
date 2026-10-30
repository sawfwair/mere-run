import Foundation
import MediaIO
import MLX
import XCTest
import MereRunMLXTestSupport
@testable import MereRunCore

final class UnlimitedOCRImageParityTests: MLXTestCase {
    func testDecodedAndResizedPixelsAgainstUpstream() throws {
        let env = ProcessInfo.processInfo.environment
        guard let path = env["MERERUN_TEST_UNLIMITED_OCR_TRACE"],
              let image = env["MERERUN_TEST_UNLIMITED_OCR_IMAGE"] else {
            throw XCTSkip("Provide an upstream capture and its source image for pixel parity.")
        }
        let reference = try MLX.loadArrays(url: URL(fileURLWithPath: path))
        let source = try MediaImageIO.decodeWithPillowJPEG(URL(fileURLWithPath: image))
        let rgb = MLXArray(MediaImageIO.rgbCHWFloat(source, normalizedToMinusOneToOne: false))
            .reshaped(3, source.height, source.width).transposed(1, 2, 0) * 255
        let decodedDelta = abs(rgb - (try XCTUnwrap(reference["decoded_rgb"])).asType(.float32))
        let actual = try UnlimitedOCRImage.prepare(URL(fileURLWithPath: image), crop: false).global
        let delta = abs(actual.asType(.float32) - (try XCTUnwrap(reference["pixels"])).asType(.float32))
        let statistics = Statistics(image: URL(fileURLWithPath: image).lastPathComponent,
            decodedMaximum: decodedDelta.max().item(Float.self), decodedMean: mean(decodedDelta).item(Float.self),
            normalizedMaximum: delta.max().item(Float.self), normalizedMean: mean(delta).item(Float.self))
        FileHandle.standardError.write(Data("[parity] decoded RGB: max=\(statistics.decodedMaximum), mean=\(statistics.decodedMean); normalized pixels: max=\(statistics.normalizedMaximum), mean=\(statistics.normalizedMean)\n".utf8))
        if env["MERERUN_TEST_UNLIMITED_OCR_EXPECT_IDENTICAL_PIXELS"] == "1" {
            XCTAssertEqual(statistics.decodedMaximum, 0)
            XCTAssertEqual(statistics.normalizedMaximum, 0)
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(statistics).write(to: URL(fileURLWithPath: path).deletingLastPathComponent()
            .appendingPathComponent(statistics.image + ".pixels.json"))
        try MLX.save(arrays: ["pixels": actual], url: URL(fileURLWithPath: path).deletingPathExtension().appendingPathExtension("native-pixels.safetensors"))
    }

    private struct Statistics: Encodable {
        let image: String
        let decodedMaximum: Float
        let decodedMean: Float
        let normalizedMaximum: Float
        let normalizedMean: Float
    }
}
