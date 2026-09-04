@testable import StudioKit
@testable import StudioUI
import CoreGraphics
import Foundation
import XCTest

/// The Analyze archetype: decoding the documents the CLI really writes, putting their coordinates
/// on screen, and the per-task view switch and next steps the canvas renders from.
final class StudioAnalyzeTests: XCTestCase {
    // MARK: - vision ground

    /// `FalconPerceptionGroundingMetadata` as the grounder encodes it: camelCase keys, escaped
    /// slashes, `queries` rather than `prompts`, and boxes normalized to 0…1.
    private let groundJSON = """
    {
      "annotatedImagePath" : "\\/tmp\\/mug_grounded.png",
      "detections" : [
        {
          "box" : { "x1" : 0.240234375, "x2" : 0.700195312, "y1" : 0.299804687, "y2" : 0.740234375 },
          "hw" : { "h" : 0.440429687, "w" : 0.459960937 },
          "label" : "coffee cup",
          "maskPath" : "\\/tmp\\/masks\\/coffee-cup_mask_0.png",
          "score" : 0.94,
          "xy" : { "x" : 0.470214843, "y" : 0.520019531 }
        },
        {
          "box" : { "x1" : 0.080078125, "x2" : 0.259765625, "y1" : 0.739257812, "y2" : 0.899414062 },
          "hw" : { "h" : 0.16015625, "w" : 0.1796875 },
          "label" : "saucer",
          "score" : 0.81,
          "xy" : { "x" : 0.169921875, "y" : 0.819335937 }
        }
      ],
      "inputImagePath" : "\\/tmp\\/mug.png",
      "jsonOutputPath" : "\\/tmp\\/mug_grounded.json",
      "modelID" : "vision-ground-falcon-perception",
      "queries" : [ "every coffee cup and what it sits on" ],
      "schemaVersion" : 1
    }
    """

    func testGroundDocumentDecodesAndScalesNormalizedBoxesToPixels() throws {
        let document = try XCTUnwrap(StudioAnalyzeDocument.decode(Data(groundJSON.utf8)))
        guard case .ground(let ground) = document else {
            return XCTFail("Expected a ground document, got \(document)")
        }
        XCTAssertEqual(ground.schemaVersion, 1)
        XCTAssertEqual(ground.modelID, "vision-ground-falcon-perception")
        XCTAssertEqual(ground.queries, ["every coffee cup and what it sits on"])
        XCTAssertEqual(document.modelID, "vision-ground-falcon-perception")

        let detections = document.detections(imageSize: CGSize(width: 1_024, height: 1_024))
        XCTAssertEqual(detections.map(\.label), ["coffee cup", "saucer"])
        XCTAssertEqual(detections[0].boxDescription, "[246, 307, 717, 758]")
        XCTAssertEqual(detections[1].boxDescription, "[82, 757, 266, 921]")
        XCTAssertEqual(detections[0].confidenceDescription, "0.94")
        XCTAssertEqual(detections[0].tabLabel, "coffee cup 0.94")
        XCTAssertEqual(detections[0].maskURL?.lastPathComponent, "coffee-cup_mask_0.png")
        XCTAssertNil(detections[1].maskURL)
        XCTAssertEqual(document.summary(detectionCount: detections.count), "2 objects found")
    }

    /// Today's grounder never sets `score`, so the confidence column has to be optional.
    func testGroundDetectionWithoutScoreHasNoConfidence() throws {
        let json = """
        {
          "detections" : [
            { "box" : { "x1" : 0, "x2" : 0.5, "y1" : 0, "y2" : 0.5 }, "label" : "person" }
          ],
          "inputImagePath" : "/tmp/a.png",
          "annotatedImagePath" : "/tmp/a_grounded.png",
          "modelID" : "vision-ground-falcon-perception",
          "queries" : [ "person" ],
          "schemaVersion" : 1
        }
        """
        let document = try XCTUnwrap(StudioAnalyzeDocument.decode(Data(json.utf8)))
        let detections = document.detections(imageSize: CGSize(width: 800, height: 600))
        XCTAssertNil(detections[0].confidenceDescription)
        XCTAssertEqual(detections[0].tabLabel, "person")
        XCTAssertEqual(detections[0].boxDescription, "[0, 0, 400, 300]")
        XCTAssertEqual(document.summary(detectionCount: 1), "1 object found")
    }

    // MARK: - vision segment

    /// `SAM31SegmentationMetadata`: schema 2, `prompts`, absolute pixel xyxy, optional `objectID`,
    /// `promptKind`, `maskPath`.
    func testSegmentDocumentDecodesPixelBoxesAndMasks() throws {
        let json = """
        {
          "annotatedImagePath" : "\\/tmp\\/crowd_segmented.png",
          "detections" : [
            {
              "box" : { "x1" : 412, "x2" : 638, "y1" : 96, "y2" : 701 },
              "label" : "person",
              "maskAreaPixels" : 84213,
              "maskPath" : "\\/tmp\\/masks\\/person_mask_0.png",
              "objectID" : "person",
              "promptKind" : "text",
              "score" : 0.7431640625
            },
            {
              "box" : { "x1" : 21, "x2" : 179, "y1" : 41, "y2" : 298 },
              "label" : "phone",
              "maskAreaPixels" : 30112,
              "score" : 0.9091796875
            }
          ],
          "inputImagePath" : "\\/tmp\\/crowd.png",
          "jsonOutputPath" : "\\/tmp\\/crowd_segmented.json",
          "modelID" : "vision-segment-sam31",
          "prompts" : [ "person", "phone" ],
          "resolution" : 1008,
          "schemaVersion" : 2,
          "threshold" : 0.05000000074505806
        }
        """
        let document = try XCTUnwrap(StudioAnalyzeDocument.decode(Data(json.utf8)))
        guard case .segmentation(let segmentation) = document else {
            return XCTFail("Expected a segmentation document, got \(document)")
        }
        XCTAssertEqual(segmentation.schemaVersion, 2)
        XCTAssertEqual(segmentation.prompts, ["person", "phone"])
        XCTAssertEqual(segmentation.detections[1].objectID, nil)

        // Pixel documents ignore the image size they are handed.
        let detections = document.detections(imageSize: CGSize(width: 1_280, height: 960))
        XCTAssertEqual(detections[0].boxDescription, "[412, 96, 638, 701]")
        XCTAssertEqual(detections[1].boxDescription, "[21, 41, 179, 298]")
        XCTAssertEqual(detections[0].confidenceDescription, "0.74")
        XCTAssertEqual(detections[0].maskURL?.path, "/tmp/masks/person_mask_0.png")
        XCTAssertNil(detections[1].maskURL)
    }

    // MARK: - vision track

    /// `SAM31TrackingRun`: every frame carries an entry for every object, including the ones the
    /// tracker lost (`visible: false`), which must not be drawn.
    func testTrackDocumentReadsFramesAndDropsInvisibleDetections() throws {
        let json = """
        {
          "annotatedVideoPath" : "\\/tmp\\/street_tracked.mp4",
          "droppedFrameCount" : 0,
          "fps" : 30,
          "frameHeight" : 720,
          "frameWidth" : 1280,
          "frames" : [
            {
              "detections" : [
                {
                  "box" : { "x1" : 402, "x2" : 611, "y1" : 288, "y2" : 431 },
                  "label" : "red car", "maskAreaPixels" : 21874, "objectID" : "red-car",
                  "score" : 0.86328125, "visible" : true
                }
              ],
              "frameIndex" : 0, "timestampSeconds" : 0
            },
            {
              "detections" : [
                {
                  "box" : { "x1" : 409, "x2" : 618, "y1" : 289, "y2" : 433 },
                  "label" : "red car", "maskAreaPixels" : 0, "objectID" : "red-car",
                  "score" : 0, "visible" : false
                }
              ],
              "frameIndex" : 1, "timestampSeconds" : 0.03333333333333333
            }
          ],
          "initFrameIndex" : 0,
          "inputVideoPath" : "\\/tmp\\/street.mp4",
          "modelID" : "vision-segment-sam31",
          "objects" : [
            { "label" : "red car", "objectID" : "red-car", "promptKind" : "text", "seedFrameIndex" : 0,
              "seedPoints" : [] }
          ],
          "schemaVersion" : 1
        }
        """
        let document = try XCTUnwrap(StudioAnalyzeDocument.decode(Data(json.utf8)))
        guard case .tracking(let tracking) = document else {
            return XCTFail("Expected a tracking document, got \(document)")
        }
        XCTAssertEqual(tracking.frameWidth, 1_280)
        XCTAssertEqual(document.reportedInputSize, CGSize(width: 1_280, height: 720))
        XCTAssertEqual(tracking.duration, 2.0 / 30, accuracy: 0.0001)
        XCTAssertEqual(document.summary(detectionCount: 1), "1 object across 2 frames")

        let first = document.detections(imageSize: CGSize(width: 1_280, height: 720), frame: 0)
        XCTAssertEqual(first.map(\.boxDescription), ["[402, 288, 611, 431]"])
        XCTAssertTrue(document.detections(imageSize: CGSize(width: 1_280, height: 720), frame: 1).isEmpty)
    }

    // MARK: - speech

    func testDiarizationDocumentDecodesSnakeCase() throws {
        let json = """
        {
          "duration_seconds" : 12.5,
          "model" : "speech-diarize-pyannote",
          "schema_version" : 1,
          "segments" : [
            { "duration_seconds" : 4.0, "end_seconds" : 4.0, "speaker" : "speaker_0",
              "speaker_index" : 0, "start_seconds" : 0.0 },
            { "duration_seconds" : 3.5, "end_seconds" : 8.0, "speaker" : "speaker_1",
              "speaker_index" : 1, "start_seconds" : 4.5 }
          ],
          "speaker_count" : 2
        }
        """
        let document = try XCTUnwrap(StudioAnalyzeDocument.decode(Data(json.utf8)))
        guard case .diarization = document else {
            return XCTFail("Expected a diarization document, got \(document)")
        }
        XCTAssertEqual(document.modelID, "speech-diarize-pyannote")
        XCTAssertEqual(document.summary(detectionCount: 0), "2 speakers · 2 turns")
        let segments = document.speechSegments
        XCTAssertEqual(segments.map(\.speaker), ["Speaker 1", "Speaker 2"])
        XCTAssertEqual(segments[1].start, 4.5)
    }

    /// `speech transcribe` writes text, not JSON: the transcript, a blank line, then one
    /// `[MM:SS.mmm --> MM:SS.mmm]` line per alignment.
    func testTranscriptTextParsesTimestampedSegments() throws {
        let text = """
        Welcome aboard. Everything you make here stays on this Mac.

        [00:00.000 --> 00:02.480] Welcome aboard.
        [00:02.480 --> 01:05.900] Everything you make here stays on this Mac.
        """
        let document = try XCTUnwrap(StudioAnalyzeDocument.decode(Data(text.utf8)))
        guard case .transcript(let transcript) = document else {
            return XCTFail("Expected a transcript, got \(document)")
        }
        XCTAssertEqual(transcript.text, "Welcome aboard. Everything you make here stays on this Mac.")
        XCTAssertEqual(transcript.segments.count, 2)
        XCTAssertEqual(transcript.segments[0].text, "Welcome aboard.")
        XCTAssertEqual(transcript.segments[1].start, 2.48, accuracy: 0.001)
        XCTAssertEqual(transcript.duration, 65.9, accuracy: 0.001)
        XCTAssertEqual(document.summary(detectionCount: 0), "2 segments")
        XCTAssertEqual(document.speechSegments.map(\.startDescription), ["0:00", "0:02"])
    }

    func testTranscriptTimestampsAcceptAnHoursField() {
        XCTAssertEqual(StudioTranscriptDocument.seconds("00:02.480"), 2.48)
        XCTAssertEqual(StudioTranscriptDocument.seconds("01:02:03.500"), 3_723.5)
        XCTAssertNil(StudioTranscriptDocument.seconds("nonsense"))
    }

    func testUnreadableDataDecodesToNothing() {
        XCTAssertNil(StudioAnalyzeDocument.decode(Data()))
        XCTAssertNil(StudioAnalyzeDocument.decode(Data("   \n  ".utf8)))
    }

    // MARK: - Pixels to points

    func testViewRectScalesPixelBoxesOntoTheDisplayedImage() {
        let box = CGRect(x: 246, y: 307, width: 471, height: 451)
        let mapped = StudioAnalyzeGeometry.viewRect(
            for: box,
            imageSize: CGSize(width: 1_024, height: 1_024),
            displaySize: CGSize(width: 512, height: 512)
        )
        XCTAssertEqual(mapped.minX, 123, accuracy: 0.001)
        XCTAssertEqual(mapped.minY, 153.5, accuracy: 0.001)
        XCTAssertEqual(mapped.width, 235.5, accuracy: 0.001)
        XCTAssertEqual(mapped.height, 225.5, accuracy: 0.001)
    }

    /// A non-square image scales each axis by its own ratio, because the container is sized to the
    /// image's aspect ratio before the overlay is drawn.
    func testViewRectScalesEachAxisIndependently() {
        let mapped = StudioAnalyzeGeometry.viewRect(
            for: CGRect(x: 640, y: 180, width: 320, height: 90),
            imageSize: CGSize(width: 1_280, height: 720),
            displaySize: CGSize(width: 512, height: 288)
        )
        XCTAssertEqual(mapped, CGRect(x: 256, y: 72, width: 128, height: 36))
    }

    func testViewRectOfAnUnmeasuredImageIsEmpty() {
        XCTAssertEqual(
            StudioAnalyzeGeometry.viewRect(
                for: CGRect(x: 1, y: 1, width: 1, height: 1),
                imageSize: .zero,
                displaySize: CGSize(width: 100, height: 100)
            ),
            .zero
        )
    }

    func testFittedRectCentersALetterboxedImage() {
        let fitted = StudioAnalyzeGeometry.fittedRect(
            imageSize: CGSize(width: 1_600, height: 800),
            in: CGSize(width: 400, height: 400)
        )
        XCTAssertEqual(fitted, CGRect(x: 0, y: 100, width: 400, height: 200))
    }

    // MARK: - Per-task views

    func testViewSegmentsMatchWhatEachTaskProduces() {
        XCTAssertEqual(StudioTask.visionFind.analyzeArchetype?.views, [.boxes, .masks, .json])
        XCTAssertEqual(StudioTask.visionSegment.analyzeArchetype?.views, [.boxes, .masks, .json])
        XCTAssertEqual(StudioTask.visionTrack.analyzeArchetype?.views, [.video, .json])
        XCTAssertEqual(StudioTask.audioTranscribe.analyzeArchetype?.views, [.transcript, .timeline, .json])
        XCTAssertEqual(StudioTask.audioWhoSpoke.analyzeArchetype?.views, [.transcript, .timeline, .json])
        XCTAssertEqual(StudioTask.visionRead.analyzeArchetype?.views, [.text])
        XCTAssertEqual(
            StudioTask.visionFind.analyzeArchetype?.views.map(\.title),
            ["Boxes", "Masks", "JSON"]
        )
    }

    func testEveryInputFirstTaskDeclaresAnArchetypeAndTheOthersDoNot() {
        let expected: Set<StudioTask> = [
            .visionRead, .visionFind, .visionSegment, .visionTrack, .visionDepth, .visionPose,
            .visionFaces, .visionFlow, .visionGeometry, .visionLive,
            .audioTranscribe, .audioWhoSpoke, .audioEnhance, .audioSeparate,
            .textEmbeddings, .textAnonymize,
            .earthFlood, .earthFire, .earthTessera, .earthOlmoEarth,
            .soundScore, .soundCondition
        ]
        XCTAssertEqual(Set(StudioTask.allCases.filter(\.isAnalyzeTask)), expected)
        for task in expected {
            let archetype = StudioAnalyzeArchetype.archetypes[task]
            XCTAssertEqual(archetype?.task, task, "\(task) archetype records the wrong task")
            XCTAssertFalse(archetype?.views.isEmpty ?? true, "\(task) declares no result view")
            XCTAssertEqual(archetype?.defaultView, archetype?.views.first)
        }
        XCTAssertFalse(StudioTask.imageGenerate.isAnalyzeTask)
        XCTAssertFalse(StudioTask.chatChat.isAnalyzeTask)
        XCTAssertFalse(StudioTask.musicRealtime.isAnalyzeTask)
    }

    func testInputKindsMatchWhatEachTaskTakes() {
        XCTAssertEqual(StudioTask.visionFind.analyzeArchetype?.inputKind, .image)
        XCTAssertEqual(StudioTask.visionTrack.analyzeArchetype?.inputKind, .video)
        XCTAssertEqual(StudioTask.audioTranscribe.analyzeArchetype?.inputKind, .audio)
        XCTAssertEqual(StudioTask.textEmbeddings.analyzeArchetype?.inputKind, .file)
        XCTAssertEqual(StudioAnalyzeInputKind.audio.noun, "audio file")
    }

    // MARK: - Next actions

    func testFindOffersTheBoardsNextSteps() throws {
        let archetype = try XCTUnwrap(StudioTask.visionFind.analyzeArchetype)
        XCTAssertEqual(
            archetype.nextActions.map(\.title),
            ["Segment these", "Track in video", "Save JSON"]
        )
        XCTAssertEqual(archetype.siblingTasks, [.visionSegment, .visionTrack])
        XCTAssertEqual(archetype.nextActions.last?.kind, .save(.json))
    }

    func testEveryNextStepTargetsARealSiblingTask() {
        for task in StudioTask.allCases.filter(\.isAnalyzeTask) {
            guard let archetype = task.analyzeArchetype else { continue }
            for sibling in archetype.siblingTasks {
                XCTAssertNotEqual(sibling, task, "\(task) offers itself as a next step")
                XCTAssertTrue(sibling.isAnalyzeTask, "\(task) hands off to \(sibling), which is not input-first")
            }
        }
    }

    /// The image carries into Segment, which takes images; Track needs a clip, so it opens with
    /// the prompt and an empty well rather than an input it cannot use.
    func testHandoffCarriesTheInputOnlyWhereTheTargetAcceptsIt() {
        let image = URL(fileURLWithPath: "/tmp/mug.png")
        XCTAssertTrue(StudioAnalyzeHandoff.carriesInput(image, to: .visionSegment))
        XCTAssertTrue(StudioAnalyzeHandoff.carriesInput(image, to: .visionRead))
        XCTAssertFalse(StudioAnalyzeHandoff.carriesInput(image, to: .visionTrack))
        XCTAssertTrue(
            StudioAnalyzeHandoff.carriesInput(URL(fileURLWithPath: "/tmp/clip.mp4"), to: .visionTrack)
        )

        let carried = StudioAnalyzeHandoff.make(
            to: .visionSegment, inputPath: image.path, prompt: "every coffee cup"
        )
        XCTAssertEqual(carried.inputPath, image.path)
        XCTAssertEqual(carried.prompt, "every coffee cup")

        let dropped = StudioAnalyzeHandoff.make(
            to: .visionTrack, inputPath: image.path, prompt: "every coffee cup"
        )
        XCTAssertEqual(dropped.inputPath, "")
        XCTAssertEqual(dropped.prompt, "every coffee cup")
    }

    func testHandoffAppliesToTheTargetTasksDraft() {
        var draft = StudioDraft()
        draft.reset(for: .segment)
        StudioAnalyzeHandoff
            .make(to: .visionSegment, inputPath: "/tmp/mug.png", prompt: "the saucer")
            .apply(to: &draft)
        XCTAssertEqual(draft.inputPath, "/tmp/mug.png")
        XCTAssertEqual(draft.prompt, "the saucer")

        var trackDraft = StudioDraft()
        trackDraft.reset(for: .track)
        StudioAnalyzeHandoff
            .make(to: .visionTrack, inputPath: "/tmp/mug.png", prompt: "the saucer")
            .apply(to: &trackDraft)
        XCTAssertEqual(trackDraft.inputPath, "")
        XCTAssertEqual(trackDraft.prompt, "the saucer")
    }

    // MARK: - Where the result document lives

    func testDocumentSourcePrefersTheJSONSidecar() {
        let item = StudioLibraryItem(
            id: UUID(),
            mode: .findObjects,
            prompt: "every coffee cup",
            inputURL: URL(fileURLWithPath: "/tmp/mug.png"),
            outputURL: URL(fileURLWithPath: "/tmp/out/vision.ground-1.png"),
            createdAt: Date(),
            updatedAt: Date(),
            status: .completed,
            exitCode: 0,
            commandPreview: "mere.run vision ground",
            outputText: nil,
            artifactURLs: [
                URL(fileURLWithPath: "/tmp/out/vision.ground-1.png"),
                URL(fileURLWithPath: "/tmp/out/vision.ground-1.json")
            ]
        )
        XCTAssertEqual(
            StudioAnalyzeDocumentSource.url(for: item)?.lastPathComponent,
            "vision.ground-1.json"
        )
    }

    func testDocumentSourceFallsBackToATranscriptFile() {
        let item = StudioLibraryItem(
            id: UUID(),
            mode: .listen,
            prompt: "",
            inputURL: URL(fileURLWithPath: "/tmp/talk.wav"),
            outputURL: URL(fileURLWithPath: "/tmp/out/talk.txt"),
            createdAt: Date(),
            updatedAt: Date(),
            status: .completed,
            exitCode: 0,
            commandPreview: "mere.run speech transcribe",
            outputText: "Welcome aboard."
        )
        XCTAssertEqual(StudioAnalyzeDocumentSource.url(for: item)?.lastPathComponent, "talk.txt")
    }

    /// The canvas can only render what the run was asked to write, so Studio's vision runs always
    /// request the structured document beside the annotated output.
    func testStudioVisionRunsRequestTheirResultDocument() throws {
        var draft = StudioDraft()
        draft.reset(for: .findObjects)
        draft.inputPath = "/tmp/mug.png"
        draft.prompt = "every coffee cup"
        let request = try StudioCommandAdapter.makeRequest(mode: .findObjects, draft: draft)
        XCTAssertTrue(request.draft.visionJSONOutputPath.hasSuffix(".json"))
        XCTAssertEqual(
            URL(fileURLWithPath: request.draft.visionJSONOutputPath).deletingPathExtension(),
            URL(fileURLWithPath: request.draft.outputPath).deletingPathExtension()
        )
        XCTAssertTrue(request.draft.visionMaskOutputDirectory.hasSuffix("-masks"))

        var trackDraft = StudioDraft()
        trackDraft.reset(for: .track)
        trackDraft.inputPath = "/tmp/clip.mp4"
        trackDraft.prompt = "the red car"
        let trackRequest = try StudioCommandAdapter.makeRequest(mode: .track, draft: trackDraft)
        XCTAssertTrue(trackRequest.draft.visionJSONOutputPath.hasSuffix(".json"))
        // A tracked clip writes one mask set per frame; Studio does not ask for thousands of PNGs.
        XCTAssertTrue(trackRequest.draft.visionMaskOutputDirectory.isEmpty)
    }

    func testResultPathsLeaveExplicitChoicesAlone() {
        var draft = CommandDraft()
        draft.outputPath = "/tmp/out/vision.ground-1.png"
        draft.visionJSONOutputPath = "/elsewhere/mine.json"
        StudioVisionResultPaths.apply(to: &draft, wantsMasks: true)
        XCTAssertEqual(draft.visionJSONOutputPath, "/elsewhere/mine.json")
        XCTAssertEqual(draft.visionMaskOutputDirectory, "/tmp/out/vision.ground-1-masks")

        var blank = CommandDraft()
        StudioVisionResultPaths.apply(to: &blank, wantsMasks: true)
        XCTAssertTrue(blank.visionJSONOutputPath.isEmpty)
        XCTAssertTrue(blank.visionMaskOutputDirectory.isEmpty)
    }
}
