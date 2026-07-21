import XCTest
@testable import MereRunCore

final class SAM31VideoTrackerTests: MereRunCoreTestCase {
    func testFallbackRejectsCollapsedMaskArea() {
        let box = SAM31SegmentationBox(x1: 100, y1: 50, x2: 500, y2: 450)
        let seed = trackingResult(area: 60_000, box: box)
        let previous = trackingResult(area: 45_000, box: box)
        let collapsed = SAM31SegmentationDetection(
            objectID: "performer",
            label: "performer",
            score: 0.7,
            box: SAM31SegmentationBox(x1: 450, y1: 100, x2: 500, y2: 430),
            maskAreaPixels: 5_000
        )

        XCTAssertTrue(
            SAM31VideoTracker.needsFallback(
                detection: collapsed,
                previousResult: previous,
                seedResult: seed,
                threshold: 0.05
            )
        )
    }

    func testFallbackAcceptsStableMaskArea() {
        let box = SAM31SegmentationBox(x1: 100, y1: 50, x2: 500, y2: 450)
        let seed = trackingResult(area: 60_000, box: box)
        let previous = trackingResult(area: 50_000, box: box)
        let stable = SAM31SegmentationDetection(
            objectID: "performer",
            label: "performer",
            score: 0.7,
            box: SAM31SegmentationBox(x1: 105, y1: 55, x2: 505, y2: 455),
            maskAreaPixels: 48_000
        )

        XCTAssertFalse(
            SAM31VideoTracker.needsFallback(
                detection: stable,
                previousResult: previous,
                seedResult: seed,
                threshold: 0.05
            )
        )
    }

    func testTrackingCandidatePrefersGeometryOverBackgroundScore() throws {
        let seedBox = SAM31SegmentationBox(x1: 286, y1: 100, x2: 646, y2: 480)
        let seed = trackingResult(area: 59_000, box: seedBox)
        let background = SAM31SegmentationDetection(
            objectID: "performer",
            label: "performer",
            score: 0.11,
            box: SAM31SegmentationBox(x1: 0, y1: 0, x2: 832, y2: 480),
            maskAreaPixels: 372_000,
            candidateIndex: 0
        )
        let performer = SAM31SegmentationDetection(
            objectID: "performer",
            label: "performer",
            score: 0.10,
            box: SAM31SegmentationBox(x1: 268, y1: 124, x2: 640, y2: 480),
            maskAreaPixels: 61_779,
            candidateIndex: 1
        )

        let selected = try XCTUnwrap(
            SAM31VideoTracker.preferredTrackingDetection(
                [background, performer],
                previousResult: seed,
                seedResult: seed,
                threshold: 0.05
            )
        )

        XCTAssertEqual(selected.candidateIndex, 1)
    }

    func testTrackingCandidatePrefersDefaultMaskWhenGeometryIsComparable() throws {
        let box = SAM31SegmentationBox(x1: 270, y1: 90, x2: 640, y2: 480)
        let previous = trackingResult(area: 61_000, box: box)
        let defaultMask = SAM31SegmentationDetection(
            objectID: "performer",
            label: "performer",
            score: 1,
            box: box,
            maskAreaPixels: 62_000,
            candidateIndex: 0
        )
        let alternate = SAM31SegmentationDetection(
            objectID: "performer",
            label: "performer",
            score: 1,
            box: box,
            maskAreaPixels: 61_000,
            candidateIndex: 3
        )

        let selected = try XCTUnwrap(
            SAM31VideoTracker.preferredTrackingDetection(
                [alternate, defaultMask],
                previousResult: previous,
                seedResult: previous,
                threshold: 0.05
            )
        )

        XCTAssertEqual(selected.candidateIndex, 0)
    }

    func testTrackingCandidateRejectsCollapsedOnlyResultSoTheGapCanReanchor() {
        let box = SAM31SegmentationBox(x1: 100, y1: 50, x2: 500, y2: 450)
        let seed = trackingResult(area: 60_000, box: box)
        let collapsed = SAM31SegmentationDetection(
            objectID: "performer",
            label: "performer",
            score: 0.7,
            box: SAM31SegmentationBox(x1: 450, y1: 100, x2: 500, y2: 430),
            maskAreaPixels: 5_000,
            candidateIndex: 0
        )

        XCTAssertNil(
            SAM31VideoTracker.preferredTrackingDetection(
                [collapsed],
                previousResult: seed,
                seedResult: seed,
                threshold: 0.05
            )
        )
    }

    func testReappearingObjectCanRecoverFromItsSeedAfterAVisibilityGap() {
        let seedBox = SAM31SegmentationBox(x1: 100, y1: 50, x2: 300, y2: 450)
        let seed = trackingResult(area: 48_000, box: seedBox)
        let missing = SAM31TrackingObjectResult(
            objectID: "performer",
            label: "performer",
            score: 0,
            visible: false,
            box: seedBox,
            maskAreaPixels: 0
        )
        let reappeared = SAM31SegmentationDetection(
            objectID: "performer",
            label: "performer",
            score: 0.72,
            box: SAM31SegmentationBox(x1: 510, y1: 55, x2: 710, y2: 455),
            maskAreaPixels: 46_000,
            candidateIndex: 0
        )

        XCTAssertFalse(
            SAM31VideoTracker.needsFallback(
                detection: reappeared,
                previousResult: missing,
                seedResult: seed,
                threshold: 0.05
            )
        )
    }

    func testMultiSubjectCandidateSelectionKeepsStableObjectIDs() throws {
        let leftBox = SAM31SegmentationBox(x1: 40, y1: 60, x2: 260, y2: 450)
        let rightBox = SAM31SegmentationBox(x1: 560, y1: 60, x2: 790, y2: 450)
        let left = trackingResult(objectID: "lead", area: 52_000, box: leftBox)
        let right = trackingResult(objectID: "partner", area: 54_000, box: rightBox)
        let detections = [
            SAM31SegmentationDetection(
                objectID: "partner",
                label: "partner",
                score: 0.8,
                box: rightBox,
                maskAreaPixels: 53_000,
                candidateIndex: 0
            ),
            SAM31SegmentationDetection(
                objectID: "lead",
                label: "lead",
                score: 0.8,
                box: leftBox,
                maskAreaPixels: 51_000,
                candidateIndex: 0
            ),
        ]

        let selected = SAM31VideoTracker.trackingDetectionsByObjectID(
            detections,
            previousResultsByObject: ["lead": left, "partner": right],
            seedResultsByObject: ["lead": left, "partner": right],
            threshold: 0.05
        )

        XCTAssertEqual(try XCTUnwrap(selected["lead"]).objectID, "lead")
        XCTAssertEqual(try XCTUnwrap(selected["lead"]).box, leftBox)
        XCTAssertEqual(try XCTUnwrap(selected["partner"]).objectID, "partner")
        XCTAssertEqual(try XCTUnwrap(selected["partner"]).box, rightBox)
    }

    private func trackingResult(
        objectID: String = "performer",
        area: Int,
        box: SAM31SegmentationBox
    ) -> SAM31TrackingObjectResult {
        SAM31TrackingObjectResult(
            objectID: objectID,
            label: objectID,
            score: 0.8,
            visible: true,
            box: box,
            maskAreaPixels: area
        )
    }
}
