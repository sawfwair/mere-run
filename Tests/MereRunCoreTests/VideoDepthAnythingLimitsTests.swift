import Foundation
import MediaIO
import MereRunCore
import XCTest

final class VideoDepthAnythingLimitsTests: XCTestCase {
    func testRequestControlsUseBoundedDefaultAndRejectOutOfRangeValues() throws {
        XCTAssertEqual(
            try VideoDepthAnythingLimits.validateRequest(
                inputSize: VideoDepthAnythingLimits.defaultInputSize,
                maximumFrameCount: nil
            ),
            VideoDepthAnythingLimits.defaultMaximumFrameCount
        )

        XCTAssertThrowsError(
            try VideoDepthAnythingLimits.validateRequest(
                inputSize: VideoDepthAnythingLimits.maximumInputSize + 1,
                maximumFrameCount: 1
            )
        ) { error in
            XCTAssertEqual(
                error as? VideoDepthAnythingLimitError,
                .inputSizeOutOfRange(
                    actual: VideoDepthAnythingLimits.maximumInputSize + 1,
                    minimum: VideoDepthAnythingLimits.minimumInputSize,
                    maximum: VideoDepthAnythingLimits.maximumInputSize
                )
            )
        }

        XCTAssertThrowsError(
            try VideoDepthAnythingLimits.validateRequest(
                inputSize: VideoDepthAnythingLimits.defaultInputSize,
                maximumFrameCount: VideoDepthAnythingLimits.maximumFrameCount + 1
            )
        )
    }

    func testDecodedFrameAndAggregatePixelBudgetsUseOverflowSafeArithmetic() throws {
        try VideoDepthAnythingLimits.validateDecodedSequence(
            width: 1_920,
            height: 1_080,
            frameCount: VideoDepthAnythingLimits.defaultMaximumFrameCount
        )

        XCTAssertThrowsError(
            try VideoDepthAnythingLimits.validateDecodedSequence(
                width: VideoDepthAnythingLimits.maximumDecodedFrameDimension + 1,
                height: 2,
                frameCount: 1
            )
        ) { error in
            guard case .decodedFrameDimensionLimitExceeded = error as? VideoDepthAnythingLimitError else {
                return XCTFail("Unexpected error: \(error)")
            }
        }

        XCTAssertThrowsError(
            try VideoDepthAnythingLimits.validateDecodedSequence(
                width: 8_192,
                height: 8_192,
                frameCount: 1
            )
        ) { error in
            guard case .decodedFramePixelLimitExceeded = error as? VideoDepthAnythingLimitError else {
                return XCTFail("Unexpected error: \(error)")
            }
        }

        XCTAssertThrowsError(
            try VideoDepthAnythingLimits.validateDecodedSequence(
                width: 1_920,
                height: 1_080,
                frameCount: Int.max
            )
        ) { error in
            XCTAssertEqual(
                error as? VideoDepthAnythingLimitError,
                .aggregateDecodedPixelLimitExceeded(
                    actual: Int.max,
                    maximum: VideoDepthAnythingLimits.maximumAggregateDecodedPixelCount
                )
            )
        }
    }

    func testPreprocessingPlanRejectsUnsafeDimensionsWithoutTrapping() throws {
        XCTAssertThrowsError(
            try VideoDepthAnythingPreprocessingPlan(
                sourceWidth: Int.max,
                sourceHeight: 2,
                requestedInputSize: VideoDepthAnythingLimits.maximumInputSize
            )
        )
        XCTAssertThrowsError(
            try VideoDepthAnythingPreprocessingPlan(
                sourceWidth: 8_192,
                sourceHeight: 2,
                requestedInputSize: VideoDepthAnythingLimits.maximumInputSize
            )
        ) { error in
            guard case .networkDimensionLimitExceeded = error as? VideoDepthAnythingLimitError else {
                return XCTFail("Unexpected error: \(error)")
            }
        }
    }

    func testGeneratorRejectsControlsBeforeLookingAtInputOrCheckpoint() async {
        let generator = VideoDepthAnythingGenerator()
        do {
            _ = try await generator.generate(
                videoURL: URL(fileURLWithPath: "/definitely/missing/video.mp4"),
                outputDirectory: URL(fileURLWithPath: "/definitely/missing/output"),
                model: "/definitely/missing/model.pth",
                inputSize: VideoDepthAnythingLimits.maximumInputSize + 1
            )
            XCTFail("Expected request validation to fail")
        } catch {
            guard case .inputSizeOutOfRange = error as? VideoDepthAnythingLimitError else {
                return XCTFail("Request did not fail at the early limit check: \(error)")
            }
        }
    }

    func testPreflightRunsSharedBoundedAdmissionWithoutInference() async throws {
        let modelPath = ProcessInfo.processInfo.environment["MERERUN_TEST_VDA_CONVERTED_RELATIVE"] ?? ""
        try XCTSkipIf(
            modelPath.isEmpty || !FileManager.default.fileExists(atPath: modelPath),
            "Set MERERUN_TEST_VDA_CONVERTED_RELATIVE to exercise the complete dry-run admission path"
        )
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("vda-preflight-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let videoURL = root.appendingPathComponent("input.mp4")
        let width = 64
        let height = 64
        let sourceFrameCount = 4
        let rgb24 = [UInt8](repeating: 127, count: width * height * sourceFrameCount * 3)
        try MediaVideoIO.writeMP4(
            rgb24: rgb24,
            width: width,
            height: height,
            frameCount: sourceFrameCount,
            fps: 4,
            to: videoURL
        )

        let result = try await VideoDepthAnythingGenerator.preflight(
            videoURL: videoURL,
            model: modelPath,
            inputSize: 14,
            maximumFrameCount: 2
        )

        XCTAssertEqual(result.input.path, videoURL.path)
        XCTAssertEqual(result.input.byteCount, try ModelArtifactPin.fileByteCount(videoURL))
        XCTAssertEqual(result.input.sha256, try ModelArtifactPin.fileSHA256(videoURL))
        XCTAssertEqual(result.checkpoint.variant, .relative)
        XCTAssertEqual(result.sourceWidth, width)
        XCTAssertEqual(result.sourceHeight, height)
        XCTAssertEqual(result.frameCount, 2)
        XCTAssertGreaterThan(result.sourceFPS, 0)
        XCTAssertEqual(result.effectiveInputSize, 14)
        XCTAssertEqual(result.networkWidth, 14)
        XCTAssertEqual(result.networkHeight, 14)
        XCTAssertEqual(result.windowCount, 1)
    }
}
