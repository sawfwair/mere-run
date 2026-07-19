import Foundation
import MediaIO
import XCTest
@testable import MereRunCore

final class SCAIL2MaskPlanTests: XCTestCase {
    func testDecodesSnakeCasePlanWithCompatibilityDefaults() throws {
        let data = Data(
            """
            {
              "schema_version": 1,
              "driving_video": "driver.mp4",
              "width": 896,
              "height": 512,
              "fps": 16,
              "subjects": [{
                "id": "dancer",
                "color": "blue",
                "reference_image": "reference.png",
                "reference_selector": {"text": "the dancer"},
                "driving_selector": {
                  "positive_points": [{"x": 120, "y": 80}],
                  "negative_points": [{"x": 12, "y": 8}]
                }
              }]
            }
            """.utf8
        )

        let plan = try JSONDecoder().decode(SCAIL2MaskPlan.self, from: data)

        try plan.validate()
        XCTAssertEqual(plan.schemaVersion, 1)
        XCTAssertEqual(plan.subjects.count, 1)
        XCTAssertEqual(plan.subjects[0].color, .blue)
        XCTAssertEqual(plan.threshold, 0.05)
        XCTAssertEqual(plan.resolution, 1008)
        XCTAssertEqual(plan.seedFrameSearchLimit, 48)
        XCTAssertEqual(plan.paletteTolerance, SCAIL2Palette.codecTolerance)
    }

    func testRejectsMoreThanSixSubjects() {
        let subjects = (0..<7).map { index in
            SCAIL2MaskSubject(
                id: "subject-\(index)",
                color: SCAIL2SubjectColor.assignmentOrder[index % 6],
                referenceImage: "reference-\(index).png",
                referenceSelector: SCAIL2MaskSelector(text: "person"),
                drivingSelector: SCAIL2MaskSelector(text: "person")
            )
        }
        let plan = SCAIL2MaskPlan(
            drivingVideo: "driver.mp4",
            width: 896,
            height: 512,
            fps: 16,
            subjects: subjects
        )

        XCTAssertThrowsError(try plan.validate()) { error in
            XCTAssertEqual(
                error as? SCAIL2MaskPlan.ValidationError,
                .invalidSubjectCount(7)
            )
        }
    }

    func testCorrectionPropagationUsesNearestImmutableBoundary() {
        let corrections = [
            SCAIL2MaskCorrection(
                subjectID: "dancer",
                frameIndex: 10,
                positivePoints: [SCAIL2MaskPoint(x: 10, y: 10)]
            ),
            SCAIL2MaskCorrection(
                subjectID: "dancer",
                frameIndex: 30,
                positivePoints: [SCAIL2MaskPoint(x: 30, y: 30)]
            ),
            SCAIL2MaskCorrection(
                subjectID: "dancer",
                frameIndex: 50,
                positivePoints: [SCAIL2MaskPoint(x: 50, y: 50)]
            ),
        ]

        XCTAssertEqual(SCAIL2MaskPreparer.correctionIndex(for: 20, corrections: corrections), 0)
        XCTAssertEqual(SCAIL2MaskPreparer.correctionIndex(for: 21, corrections: corrections), 1)
        XCTAssertEqual(SCAIL2MaskPreparer.correctionIndex(for: 41, corrections: corrections), 2)
        XCTAssertEqual(
            SCAIL2MaskPreparer.correctionPropagationRange(at: 0, corrections: corrections),
            0...20
        )
        XCTAssertEqual(
            SCAIL2MaskPreparer.correctionPropagationRange(at: 1, corrections: corrections),
            21...40
        )
        XCTAssertEqual(
            SCAIL2MaskPreparer.correctionPropagationRange(at: 2, corrections: corrections),
            41...Int.max
        )
    }

    func testPaletteCompositionUsesStablePriorityAndReportsOverlap() throws {
        let first: [UInt8] = [1, 1, 0, 0]
        let second: [UInt8] = [0, 1, 1, 0]

        let result = try SCAIL2Palette.compose(
            width: 2,
            height: 2,
            subjectMasks: [(.blue, first), (.red, second)]
        )

        XCTAssertEqual(result.overlapPixelCount, 1)
        XCTAssertEqual(Array(result.image.rgba8[4..<7]), [0, 0, 255])
        XCTAssertEqual(Array(result.image.rgba8[8..<11]), [255, 0, 0])
    }

    func testPaletteSnappingAcceptsCodecDriftAndRejectsAmbiguity() throws {
        let nearBlue = try MediaImage(
            width: 1,
            height: 1,
            rgba8: [3, 2, 249, 255]
        )
        let snapped = try SCAIL2Palette.snapped(nearBlue, tolerance: 8)
        XCTAssertEqual(snapped.rgba8, [0, 0, 255, 255])

        let ambiguous = try MediaImage(
            width: 1,
            height: 1,
            rgba8: [127, 127, 0, 255]
        )
        XCTAssertThrowsError(try SCAIL2Palette.snapped(ambiguous, tolerance: 128)) { error in
            guard case SCAIL2PaletteError.ambiguousColor = error else {
                return XCTFail("Unexpected error: \(error)")
            }
        }
    }

    func testPaletteSnappingAcceptsUnambiguousProResEdgeColor() throws {
        let blackBackground = try MediaImage(
            width: 3,
            height: 1,
            rgba8: [
                3, 2, 4, 255,
                0, 5, 62, 255,
                30, 40, 65, 255,
            ]
        )
        let snappedBackground = try SCAIL2Palette.snapped(
            blackBackground,
            tolerance: SCAIL2Palette.codecTolerance
        )
        XCTAssertEqual(
            snappedBackground.rgba8,
            [
                0, 0, 0, 255,
                0, 0, 0, 255,
                0, 0, 0, 255,
            ]
        )

        let decodedEdge = try MediaImage(
            width: 4,
            height: 1,
            rgba8: [
                14, 12, 86, 255,
                82, 84, 115, 255,
                51, 61, 83, 255,
                50, 49, 68, 255,
            ]
        )
        let snapped = try SCAIL2Palette.snapped(
            decodedEdge,
            tolerance: SCAIL2Palette.codecTolerance
        )
        XCTAssertEqual(
            snapped.rgba8,
            [
                0, 0, 255, 255,
                0, 0, 255, 255,
                0, 0, 255, 255,
                0, 0, 255, 255,
            ]
        )
    }

    func testManifestRequiresAndEncodesTheExactModelRevision() throws {
        let manifest = SCAIL2MaskManifest(
            status: "preview_ready",
            previewFrame: 0,
            modelID: "vision-segment-sam31",
            modelRevision: "a992e302ea9b0f03f41dfd93414a4fd0e818f65b",
            drivingSourcePath: "driver.mp4",
            drivingProxyPath: nil,
            drivingMaskPath: nil,
            overlayPreviewPath: "preview.png",
            contactSheetPath: "contact-sheet.png",
            trackingPath: nil,
            qualityPath: "quality.json",
            frameCount: 1,
            fps: 16,
            width: 32,
            height: 32,
            subjects: [],
            corrections: [],
            artifacts: []
        )

        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(manifest))
                as? [String: Any]
        )
        XCTAssertEqual(
            object["model_revision"] as? String,
            "a992e302ea9b0f03f41dfd93414a4fd0e818f65b"
        )
    }

    func testPaletteVideoRoundTripPreservesLegalLabels() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("scail-palette-test-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        var frameURLs: [URL] = []
        var expected: [MediaImage] = []
        for (frameIndex, color) in SCAIL2SubjectColor.assignmentOrder.prefix(3).enumerated() {
            let mask = [UInt8](repeating: 1, count: 32 * 32)
            let image = try SCAIL2Palette.compose(
                width: 32,
                height: 32,
                subjectMasks: [(color, mask)]
            ).image
            let url = root.appendingPathComponent("source-\(frameIndex).png")
            try MediaImageIO.writePNG(image, to: url)
            frameURLs.append(url)
            expected.append(image)
        }
        let videoURL = root.appendingPathComponent("mask.mov")
        try MediaVideoIO.writePaletteVideo(frameURLs: frameURLs, fps: 16, to: videoURL)

        let decoded = try MediaVideoIO.extractFrames(
            from: videoURL,
            into: root.appendingPathComponent("decoded", isDirectory: true)
        )

        XCTAssertEqual(decoded.frameURLs.count, 3)
        XCTAssertEqual(decoded.fps, 16, accuracy: 0.1)
        for (index, url) in decoded.frameURLs.enumerated() {
            let snapped = try SCAIL2Palette.snapped(
                try MediaImageIO.decode(url),
                tolerance: 96
            )
            XCTAssertEqual(snapped, expected[index])
        }
    }
}
