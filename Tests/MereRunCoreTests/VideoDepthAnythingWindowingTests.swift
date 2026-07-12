import MereRunCore
import XCTest

final class VideoDepthAnythingWindowingTests: XCTestCase {
    func testPreprocessingPlanMatchesReferenceLowerBoundSizing() throws {
        let landscape = try VideoDepthAnythingPreprocessingPlan(
            sourceWidth: 1_920,
            sourceHeight: 1_080
        )
        XCTAssertEqual(landscape.effectiveInputSize, 518)
        XCTAssertEqual(landscape.networkWidth, 924)
        XCTAssertEqual(landscape.networkHeight, 518)

        let ultrawide = try VideoDepthAnythingPreprocessingPlan(
            sourceWidth: 3_840,
            sourceHeight: 1_080
        )
        XCTAssertEqual(ultrawide.effectiveInputSize, 252)
        XCTAssertEqual(ultrawide.networkWidth, 896)
        XCTAssertEqual(ultrawide.networkHeight, 252)

        let portrait = try VideoDepthAnythingPreprocessingPlan(
            sourceWidth: 1_080,
            sourceHeight: 1_920
        )
        XCTAssertEqual(portrait.networkWidth, 518)
        XCTAssertEqual(portrait.networkHeight, 924)
    }

    func testPlansMatchReferencePaddingAndRecursiveKeyframeSubstitution() throws {
        let plans = try VideoDepthAnythingWindowing.plans(originalFrameCount: 45)
        XCTAssertEqual(plans.count, 3)
        XCTAssertEqual(plans[0].sourceFrameIndices, Array(0..<32))
        XCTAssertEqual(Array(plans[1].sourceFrameIndices.prefix(10)), [0, 12, 24, 25, 26, 27, 28, 29, 30, 31])
        XCTAssertEqual(Array(plans[1].sourceFrameIndices.dropFirst(10).prefix(3)), [32, 33, 34])
        XCTAssertEqual(Array(plans[2].sourceFrameIndices.prefix(10)), [0, 34, 44, 44, 44, 44, 44, 44, 44, 44])
        XCTAssertTrue(plans[2].sourceFrameIndices.dropFirst(10).allSatisfy { $0 == 44 })
    }

    func testRelativeWindowsAreAffineAlignedBlendedAndTruncated() throws {
        let first = (0..<32).map { frame in [Float(frame), Float(frame + 10)] }
        // The second window is in a different affine coordinate system. Its
        // first two frames correspond to reference keyframes 0 and 12.
        var second: [[Float]] = []
        for frame in 0..<32 {
            let firstValue = Float(frame * 6 - 4)
            let secondValue = Float(frame * 6 + 16)
            second.append([firstValue, secondValue])
        }
        second[0] = [-4, 16]     // maps to [0, 10]
        second[1] = [20, 40]     // maps to [12, 22]

        let result = try VideoDepthAnythingWindowing.align(
            windows: [first, second],
            originalFrameCount: 45,
            semantics: .affineRelative
        )
        XCTAssertEqual(result.frames.count, 45)
        XCTAssertEqual(result.alignments.count, 2)
        XCTAssertEqual(result.alignments[1].scale, 0.5, accuracy: 1e-5)
        XCTAssertEqual(result.alignments[1].shift, 2, accuracy: 1e-5)
        // Blend endpoints are exactly the old and newly aligned frames.
        XCTAssertEqual(result.frames[24], first[24])
        XCTAssertEqual(result.frames[31], second[9].map { max(0, $0 * 0.5 + 2) })
        XCTAssertEqual(result.frames[32], second[10].map { max(0, $0 * 0.5 + 2) })
    }

    func testMetricWindowsNeverReceiveAffineRescaling() throws {
        let first = (0..<32).map { _ in [Float(1)] }
        let second = (0..<32).map { _ in [Float(7)] }
        let result = try VideoDepthAnythingWindowing.align(
            windows: [first, second],
            originalFrameCount: 44,
            semantics: .metricMeters
        )
        XCTAssertEqual(result.alignments[1], VideoDepthAnythingAffineAlignment(scale: 1, shift: 0))
        XCTAssertEqual(result.frames[31], [7])
        XCTAssertEqual(result.frames[32], [7])
    }

    func testSolveAffineRejectsMismatchedFramesWithoutTrapping() {
        XCTAssertThrowsError(
            try VideoDepthAnythingWindowing.solveAffine(
                prediction: [[1]],
                target: [[1], [2]]
            )
        ) { error in
            XCTAssertEqual(
                error as? VideoDepthAnythingWindowingError,
                .affineFrameCountMismatch(prediction: 1, target: 2)
            )
        }

        XCTAssertThrowsError(
            try VideoDepthAnythingWindowing.solveAffine(
                prediction: [[1, 2]],
                target: [[1]]
            )
        ) { error in
            XCTAssertEqual(
                error as? VideoDepthAnythingWindowingError,
                .affineDepthElementCountMismatch(frame: 0, prediction: 2, target: 1)
            )
        }
    }

    func testStreamingAlignmentMatchesBatchAndRetainsOnlyMutableTail() throws {
        let first = (0..<32).map { frame in [Float(frame), Float(frame + 10)] }
        var second: [[Float]] = []
        for frame in 0..<32 {
            let left = Float(frame * 6 - 4)
            let right = Float(frame * 6 + 16)
            second.append([left, right])
        }
        second[0] = [-4, 16]
        second[1] = [20, 40]
        let expected = try VideoDepthAnythingWindowing.align(
            windows: [first, second],
            originalFrameCount: 45,
            semantics: .affineRelative
        )

        var aligner = try VideoDepthAnythingWindowing.StreamingAligner(
            originalFrameCount: 45,
            semantics: .affineRelative
        )
        let firstChunk = try aligner.append(window: first, isFinal: false)
        XCTAssertEqual(firstChunk.finalizedFrames.count, 24)
        XCTAssertEqual(aligner.retainedFrameCount, 8)
        let secondChunk = try aligner.append(window: second, isFinal: true)

        XCTAssertEqual(
            firstChunk.finalizedFrames + secondChunk.finalizedFrames,
            expected.frames
        )
        XCTAssertEqual(
            [firstChunk.alignment, secondChunk.alignment],
            expected.alignments
        )
        XCTAssertEqual(aligner.retainedFrameCount, 0)
        XCTAssertThrowsError(try aligner.append(window: second, isFinal: true)) { error in
            XCTAssertEqual(
                error as? VideoDepthAnythingWindowingError,
                .streamingAlignmentAlreadyFinished
            )
        }
    }

    func testStreamingAlignmentDiscardsPaddedFramesWithoutLosingMutableTail() throws {
        let first = (0..<32).map { [Float($0)] }
        let second = (0..<32).map { [Float(100 + $0)] }
        var aligner = try VideoDepthAnythingWindowing.StreamingAligner(
            originalFrameCount: 23,
            semantics: .metricMeters
        )

        let firstChunk = try aligner.append(window: first, isFinal: false)
        let secondChunk = try aligner.append(window: second, isFinal: true)

        XCTAssertEqual(firstChunk.finalizedFrames, Array(first.prefix(23)))
        XCTAssertTrue(secondChunk.finalizedFrames.isEmpty)
    }
}
