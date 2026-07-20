import Foundation
import MediaIO
import MLX
import XCTest
@testable import MereRunCore

final class SAM31ImageSegmenterTests: MereRunCoreTestCase {
    func testDefaultOutputPathDerivation() {
        let imageURL = URL(fileURLWithPath: "/tmp/photo.png")
        XCTAssertEqual(SAM31ImageSegmenter.defaultAnnotatedOutputURL(for: imageURL).path, "/tmp/photo_segmented.png")
        XCTAssertEqual(SAM31ImageSegmenter.defaultJSONOutputURL(for: imageURL).path, "/tmp/photo_segmented.json")
    }

    func testResizeMaskUpsamplesBilinearly() {
        let source: [Float] = [
            -1, 1,
            -1, 1,
        ]

        let resized = SAM31ImageSegmenter.resizeMask(
            source,
            sourceWidth: 2,
            sourceHeight: 2,
            targetWidth: 4,
            targetHeight: 4
        )

        XCTAssertEqual(resized.count, 16)
        XCTAssertLessThan(resized[0], 0)
        XCTAssertGreaterThan(resized[3], 0)
    }

    func testPostprocessAppliesThresholdAndNMS() {
        let predLogits = MLXArray([5.0, 4.8, -4.0], [1, 3]).asType(.float32)
        let predBoxes = MLXArray([
            0.10, 0.10, 0.40, 0.40,
            0.12, 0.12, 0.41, 0.41,
            0.60, 0.60, 0.80, 0.80,
        ], [1, 3, 4]).asType(.float32)
        let predMasks = MLXArray([
            1, 1, 1, 1,
            1, 1, 1, 1,
            -1, -1, -1, -1,
        ], [1, 3, 2, 2]).asType(.float32)
        let presenceLogits = MLXArray([4.0], [1, 1]).asType(.float32)
        let output = SAM31DetectorOutput(
            predLogits: predLogits,
            predBoxes: predBoxes,
            predMasks: predMasks,
            presenceLogits: presenceLogits
        )

        let detections = SAM31ImageSegmenter.postprocess(
            output: output,
            prompt: "a person",
            imageWidth: 8,
            imageHeight: 8,
            threshold: 0.5,
            nmsThreshold: 0.3
        )

        XCTAssertEqual(detections.count, 1)
        XCTAssertEqual(detections[0].label, "a person")
        XCTAssertGreaterThan(detections[0].score, 0.5)
        XCTAssertGreaterThan(detections[0].maskAreaPixels, 0)
    }

    func testBinaryMaskPromptAcceptsEveryPaletteColor() throws {
        let image = try MediaImage(
            width: 4,
            height: 1,
            rgba8: [
                0, 0, 255, 255,
                255, 0, 0, 255,
                0, 255, 0, 255,
                0, 0, 0, 255,
            ]
        )

        XCTAssertEqual(
            SAM31ImageSegmenter.binaryMaskPromptValues(from: image),
            [1, 1, 1, 0]
        )
    }

    func testDenseMaskPromptIncludesTheUpstreamNotAPointToken() {
        let encoder = SAM31InteractivePromptEncoder(
            config: SAM31PromptEncoderConfig(
                hiddenSize: 8,
                imageSize: 8,
                patchSize: 4,
                maskInputChannels: 4
            )
        )
        let output = encoder(
            masks: MLX.ones([1, 8, 8, 1], dtype: .float32),
            targetHeight: 2,
            targetWidth: 2
        )

        XCTAssertEqual(output.sparseEmbeddings.shape, [1, 1, 8])
        XCTAssertEqual(output.denseEmbeddings.shape, [1, 4, 8])
    }

    func testSmallEnclosedMaskHolesAreFilledWithoutClosingBackground() {
        let mask: [UInt8] = [
            0, 0, 0, 0, 0,
            0, 1, 1, 1, 0,
            0, 1, 0, 1, 0,
            0, 1, 1, 1, 0,
            0, 0, 0, 0, 0,
        ]

        let filled = SAM31ImageSegmenter.fillSmallHoles(
            binaryMask: mask,
            width: 5,
            height: 5,
            maximumArea: 1
        )

        XCTAssertEqual(filled[12], 1)
        XCTAssertEqual(filled[0], 0)
    }

    func testInteractiveDecoderMLPCheckpointKeysMapToNativeLayers() {
        let prefix = "tracker_model.interactive_sam_mask_decoder.output_hypernetworks_mlps.2"

        XCTAssertEqual(
            SAM31ImageSegmenter.mapCheckpointKey("\(prefix).proj_in.weight"),
            "\(prefix).layer1.weight"
        )
        XCTAssertEqual(
            SAM31ImageSegmenter.mapCheckpointKey("\(prefix).layers.0.bias"),
            "\(prefix).layer2.bias"
        )
        XCTAssertEqual(
            SAM31ImageSegmenter.mapCheckpointKey("\(prefix).proj_out.weight"),
            "\(prefix).layer3.weight"
        )
    }

    func testJSONMetadataEncodesExpectedShape() throws {
        let metadata = SAM31SegmentationMetadata(
            modelID: "vision-segment-sam31",
            inputImagePath: "/tmp/input.png",
            annotatedImagePath: "/tmp/input_segmented.png",
            jsonOutputPath: "/tmp/input_segmented.json",
            prompts: ["a cat"],
            threshold: 0.3,
            resolution: 1008,
            detections: [
                SAM31SegmentationDetection(
                    label: "a cat",
                    score: 0.91,
                    box: SAM31SegmentationBox(x1: 10, y1: 20, x2: 30, y2: 40),
                    maskAreaPixels: 512
                )
            ]
        )

        let data = try JSONEncoder().encode(metadata)
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])

        XCTAssertEqual(json["modelID"] as? String, "vision-segment-sam31")
        XCTAssertEqual(json["resolution"] as? Int, 1008)
        XCTAssertEqual((json["prompts"] as? [String])?.first, "a cat")
        XCTAssertEqual((json["detections"] as? [[String: Any]])?.count, 1)
    }
}
