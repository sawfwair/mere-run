import MLX
import MediaIO
import MereRunCore
import XCTest

final class VideoDepthAnythingPreprocessorTests: MereRunCoreTestCase {
    func testCreatesNormalizedBTHWCVideoAtReferenceDimensions() throws {
        let frame = try MediaImage(
            width: 4,
            height: 2,
            rgba8: [UInt8](repeating: 128, count: 4 * 2 * 4)
        )
        let result = try VideoDepthAnythingPreprocessor.normalizedVideo(
            frames: [frame, frame],
            inputSize: 14
        )
        MLX.eval(result.video)
        XCTAssertEqual(result.plan.networkWidth, 28)
        XCTAssertEqual(result.plan.networkHeight, 14)
        XCTAssertEqual(result.video.shape, [1, 2, 14, 28, 3])
        XCTAssertTrue(result.video.asArray(Float.self).allSatisfy(\.isFinite))
    }

    func testRejectsDimensionChangesAcrossFrames() throws {
        let first = try MediaImage(width: 2, height: 2, rgba8: [UInt8](repeating: 0, count: 16))
        let second = try MediaImage(width: 3, height: 2, rgba8: [UInt8](repeating: 0, count: 24))
        XCTAssertThrowsError(
            try VideoDepthAnythingPreprocessor.normalizedVideo(frames: [first, second], inputSize: 14)
        ) { error in
            XCTAssertEqual(
                error as? VideoDepthAnythingPreprocessingError,
                .inconsistentFrameDimensions(
                    expectedWidth: 2,
                    expectedHeight: 2,
                    actualWidth: 3,
                    actualHeight: 2
                )
            )
        }
    }

    func testDepthResizePreservesBatchAndFrameAxes() throws {
        let depth = MLXArray((0..<16).map(Float.init)).reshaped(1, 2, 2, 4)
        let resized = try VideoDepthAnythingPreprocessor.resizeDepth(
            depth,
            sourceWidth: 8,
            sourceHeight: 6
        )
        MLX.eval(resized)
        XCTAssertEqual(resized.shape, [1, 2, 6, 8])
        XCTAssertTrue(resized.asArray(Float.self).allSatisfy(\.isFinite))
    }

    func testCubicResizeAgainstPinnedOpenCVFixture() throws {
        let root = ProcessInfo.processInfo.environment["MERERUN_TEST_VDA_PREPROCESS"] ?? ""
        try XCTSkipIf(root.isEmpty, "Set MERERUN_TEST_VDA_PREPROCESS to the frozen upstream fixture.")
        let rootURL = URL(fileURLWithPath: root, isDirectory: true)
        let source = [UInt8](try Data(contentsOf: rootURL.appendingPathComponent("source.rgb8")))
        let expected = try readFloat32(rootURL.appendingPathComponent("normalized.f32"))
        let frameCount = 2
        let width = 61
        let height = 37
        var frames: [MediaImage] = []
        for frameIndex in 0..<frameCount {
            var rgba = [UInt8](repeating: 255, count: width * height * 4)
            for pixel in 0..<(width * height) {
                let sourceOffset = (frameIndex * width * height + pixel) * 3
                let destination = pixel * 4
                rgba[destination] = source[sourceOffset]
                rgba[destination + 1] = source[sourceOffset + 1]
                rgba[destination + 2] = source[sourceOffset + 2]
            }
            frames.append(try MediaImage(width: width, height: height, rgba8: rgba))
        }
        let result = try VideoDepthAnythingPreprocessor.normalizedVideo(frames: frames, inputSize: 56)
        MLX.eval(result.video)
        let actual = result.video.asArray(Float.self)
        XCTAssertEqual(result.video.shape, [1, 2, 56, 98, 3])
        XCTAssertEqual(actual.count, expected.count)
        let differences = zip(actual, expected).map { abs($0 - $1) }
        let mean = differences.reduce(0, +) / Float(differences.count)
        let maximum = differences.max() ?? 0
        XCTAssertLessThanOrEqual(mean, 2e-5, "preprocess MAE \(mean), max \(maximum)")
        XCTAssertLessThanOrEqual(maximum, 2e-4, "preprocess MAE \(mean), max \(maximum)")
    }

    private func readFloat32(_ url: URL) throws -> [Float] {
        let data = try Data(contentsOf: url)
        return stride(from: 0, to: data.count, by: 4).map { offset in
            let bits = UInt32(data[offset])
                | (UInt32(data[offset + 1]) << 8)
                | (UInt32(data[offset + 2]) << 16)
                | (UInt32(data[offset + 3]) << 24)
            return Float(bitPattern: bits)
        }
    }
}
