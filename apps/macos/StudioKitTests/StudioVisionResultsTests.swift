@testable import StudioKit
import CoreGraphics
import Foundation
import XCTest

/// The Vision documents the Analyze canvas draws for Depth, Pose, Faces, Flow, and Geometry,
/// written the way the CLI's own Codable types encode them; the face picker's lookup; what a
/// directory output holds; and the argv the task drafts build against what the Vision Lab page
/// built for the same settings.
final class StudioVisionResultsTests: XCTestCase {
    // MARK: - Documents

    func testPoseDocumentDecodesAndPlacesLandmarksInStoredPixels() throws {
        let json = """
        {"imageWidth":640,"imageHeight":480,"coordinateSpace":"normalized-bottom-left",
         "subjects":[{"kind":"body","index":0,"points":[{"name":"nose","x":0.5,"y":0.75,"confidence":0.9}]},
                     {"kind":"hand","index":0,"points":[{"name":"wrist","x":0.25,"y":0.5,"confidence":0.4},{"name":"thumbTip","x":0.3,"y":0.55,"confidence":0.2}]}]}
        """
        let document = try XCTUnwrap(StudioAnalyzeDocument.decode(Data(json.utf8)))
        guard case .pose(let pose) = document else { return XCTFail("decoded as \(document)") }
        XCTAssertEqual(pose.summary, "2 subjects · 3 landmarks")
        XCTAssertEqual(document.summary(detectionCount: 0), "2 subjects · 3 landmarks")
        XCTAssertEqual(document.reportedInputSize, CGSize(width: 640, height: 480))
        let nose = try XCTUnwrap(pose.subjects.first?.points.first)
        XCTAssertEqual(pose.storedPoint(nose), CGPoint(x: 320, y: 120), "bottom-left space flips y into the stored pixels")
        XCTAssertTrue(document.detections(imageSize: CGSize(width: 640, height: 480)).isEmpty, "landmarks are not boxes")
    }

    func testFlowFieldDecodesAndMeasuresItsMotion() throws {
        var data = Data()
        func append(_ value: UInt32) { withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) } }
        append(Float(202_021.25).bitPattern)
        append(2)
        append(2)
        for vector in [(3.0 as Float, 4.0 as Float), (0, 0), (0, 0), (0, 1)] {
            append(vector.0.bitPattern)
            append(vector.1.bitPattern)
        }
        let document = try XCTUnwrap(StudioAnalyzeDocument.decode(data))
        guard case .flow(let field) = document else { return XCTFail("decoded as \(document)") }
        XCTAssertEqual(field.width, 2)
        XCTAssertEqual(field.height, 2)
        XCTAssertEqual(field.summary, "2×2 dense flow")
        let statistics = field.statistics
        XCTAssertEqual(statistics.maximumMagnitude, 5, accuracy: 0.0001)
        XCTAssertEqual(statistics.meanMagnitude, 1.5, accuracy: 0.0001)
        XCTAssertEqual(statistics.movingFraction, 0.5, accuracy: 0.0001)
        XCTAssertThrowsError(try StudioFlowField.decode(Data(repeating: 0, count: 8)))
    }

    /// `vision face embed --json-output` (`FaceEmbeddingOutput`): one record with its vector.
    func testFaceEmbeddingDocumentDecodesToOneBox() throws {
        let json = """
        {"image":"/tmp/portrait.png","modelID":"vision-face-buffalo-l",
         "face":{"index":1,"detection":{"score":0.91,"boundingBox":{"x":10,"y":20,"width":100,"height":120},"landmarks":[]},
                 "embedding":[0.6,0.8]}}
        """
        let document = try XCTUnwrap(StudioAnalyzeDocument.decode(Data(json.utf8)))
        guard case .faceEmbedding(let embedding) = document else { return XCTFail("decoded as \(document)") }
        XCTAssertEqual(embedding.face.embedding?.count, 2)
        XCTAssertEqual(document.summary(detectionCount: 1), "Face 2 embedded · 2 dimensions")
        XCTAssertEqual(document.modelID, "vision-face-buffalo-l")
        let boxes = document.detections(imageSize: CGSize(width: 1, height: 1))
        XCTAssertEqual(boxes.map(\.label), ["Face 2"])
        XCTAssertEqual(boxes.first?.box, CGRect(x: 10, y: 20, width: 100, height: 120))
        XCTAssertEqual(boxes.first?.confidence, 0.91)
    }

    /// `vision face compare --json-output` (`FaceComparisonOutput`).
    func testFaceComparisonDocumentDecodesTheSimilarity() throws {
        let json = """
        {"modelID":"vision-face-buffalo-l","referenceImage":"/tmp/a.png","referenceFaceIndex":0,
         "candidateImage":"/tmp/b.png","candidateFaceIndex":2,"cosineSimilarity":0.8312}
        """
        let document = try XCTUnwrap(StudioAnalyzeDocument.decode(Data(json.utf8)))
        guard case .faceComparison(let comparison) = document else { return XCTFail("decoded as \(document)") }
        XCTAssertEqual(comparison.candidateFaceIndex, 2)
        XCTAssertEqual(document.summary(detectionCount: 0), "Similarity 0.83")
        XCTAssertTrue(document.detections(imageSize: CGSize(width: 1, height: 1)).isEmpty)
    }

    /// `vision face batch --jsonl-output`: one `FaceBatchOutput` per line, failures included.
    func testFaceBatchDocumentReadsEveryLine() throws {
        let faces = """
        {"image":"/tmp/a.png","modelID":"m","width":64,"height":64,"elapsedMilliseconds":1,\
        "faces":[{"index":0,"detection":{"score":0.9,"boundingBox":{"x":1,"y":1,"width":2,"height":2},"landmarks":[]}},\
        {"index":1,"detection":{"score":0.8,"boundingBox":{"x":5,"y":5,"width":2,"height":2},"landmarks":[]}}]}
        """
        let jsonl = """
        {"ok":true,"image":"/tmp/a.png","result":\(faces)}

        {"ok":false,"image":"/tmp/b.png","error":"could not decode image"}
        """
        let document = try XCTUnwrap(StudioAnalyzeDocument.decode(Data(jsonl.utf8)))
        guard case .faceBatch(let batch) = document else { return XCTFail("decoded as \(document)") }
        XCTAssertEqual(batch.entries.count, 2)
        XCTAssertEqual(batch.faceCount, 2)
        XCTAssertEqual(batch.failureCount, 1)
        XCTAssertEqual(document.summary(detectionCount: 0), "2 images · 2 faces · 1 failed")
        XCTAssertNil(StudioFaceBatchDocument.decode(Data("{\"width\":1}\n{\"ok\":true,\"image\":\"x\"}".utf8)),
                     "a line that is not a batch entry makes it another document")
        XCTAssertNil(StudioFaceBatchDocument.decode(Data()))
    }

    func testFaceDetectionRecordKeepsAnOptionalEmbedding() throws {
        let json = """
        {"width":64,"height":64,"faces":[{"index":0,"detection":{"score":0.9,"boundingBox":{"x":1,"y":1,"width":2,"height":2},"landmarks":[]},"embedding":[1,0]}]}
        """
        let result = try JSONDecoder().decode(StudioFaceOverlayResult.self, from: Data(json.utf8))
        XCTAssertEqual(result.faces.first?.embedding, [1, 0])
        guard case .faces = try XCTUnwrap(StudioAnalyzeDocument.decode(Data(json.utf8))) else {
            return XCTFail("a detection result with embeddings is still a face document")
        }
    }

    /// `vision depth`'s `<stem>-depth.json` (`MarigoldV2DepthManifest`), as the exporter encodes it.
    func testDepthManifestDecodesForThePanel() throws {
        let json = """
        {"schemaVersion":1,"createdAt":"2026-09-24T10:00:00Z","inputPath":"/tmp/portrait.png","inputByteCount":1024,
         "inputSHA256":"ab","outputDirectory":"/tmp/portrait-depth","width":960,"height":720,"inferenceWidth":1024,"inferenceHeight":768,
         "semantics":"affine-relative","parameterization":"log","checkpoint":"log-stage2","seeThrough":false,
         "depthStatistics":{"rawMinimum":0.012,"rawMaximum":0.981,"normalizationNear":0.01,"normalizationFar":0.99},
         "model":{"modelID":"vision-depth-marigold-v2","upstreamRepository":"r","upstreamRevision":"v","license":"l","inferenceBackend":"mlx"},
         "artifacts":[]}
        """
        let document = try XCTUnwrap(StudioAnalyzeDocument.decode(Data(json.utf8)))
        guard case .depthManifest(let manifest) = document else { return XCTFail("decoded as \(document)") }
        XCTAssertEqual(manifest.summary, "960 × 720 · inferred at 1024 × 768 · log-stage2")
        XCTAssertEqual(document.summary(detectionCount: 0), manifest.summary)
        XCTAssertEqual(document.modelID, "vision-depth-marigold-v2")
        XCTAssertTrue(document.detections(imageSize: CGSize(width: 1, height: 1)).isEmpty)
    }

    /// A JSON object none of the writers' shapes match (a camera file, a scene manifest) is not a
    /// transcript, so the panel lists the run's files instead of saying nothing matched.
    func testUnknownJSONIsNotATranscript() {
        XCTAssertNil(StudioAnalyzeDocument.decode(Data("{\"cameras\":[{\"fx\":1}]}".utf8)))
        XCTAssertNil(StudioAnalyzeDocument.decode(Data("[1, 2, 3]".utf8)))
        guard case .transcript = StudioAnalyzeDocument.decode(Data("Plain prose the run printed.".utf8)) else {
            return XCTFail("prose is still a transcript")
        }
    }

    // MARK: - Picking a face

    func testFacePickMapsFlagsToPicturesAndFindsTheNewestDetection() throws {
        XCTAssertEqual(StudioFacePick.argumentIndex(forFlag: "--face-index"), 0)
        XCTAssertEqual(StudioFacePick.argumentIndex(forFlag: "--reference-face-index"), 0)
        XCTAssertEqual(StudioFacePick.argumentIndex(forFlag: "--candidate-face-index"), 1)
        XCTAssertNil(StudioFacePick.argumentIndex(forFlag: "--score-threshold"))

        let root = FileManager.default.temporaryDirectory.appendingPathComponent("face-pick-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let picture = root.appendingPathComponent("portrait.png")
        try Data([0x89, 0x50]).write(to: picture)
        let older = root.appendingPathComponent("older.json")
        let newer = root.appendingPathComponent("newer.json")
        try Data("{}".utf8).write(to: older)
        try Data("{}".utf8).write(to: newer)
        func row(_ document: URL, at date: Date, template: CommandTemplateID = .visionFaceDetect, status: StudioLibraryStatus = .completed) -> StudioLibraryItem {
            StudioLibraryItem(
                id: UUID(), mode: .readImage, prompt: "", inputURL: picture, outputURL: nil, createdAt: date, updatedAt: date,
                status: status, exitCode: 0, commandPreview: "", outputText: nil, templateID: template, artifactURLs: [document]
            )
        }
        let items = [
            row(older, at: Date(timeIntervalSince1970: 100)),
            row(newer, at: Date(timeIntervalSince1970: 200)),
            row(root.appendingPathComponent("pose.json"), at: Date(timeIntervalSince1970: 300), template: .visionPose),
            row(root.appendingPathComponent("failed.json"), at: Date(timeIntervalSince1970: 400), status: .failed),
        ]
        XCTAssertEqual(StudioFacePick.detectionDocumentURL(for: picture.path, in: items), newer)
        XCTAssertNil(StudioFacePick.detectionDocumentURL(for: root.appendingPathComponent("other.png").path, in: items))
        XCTAssertNil(StudioFacePick.detectionDocumentURL(for: "", in: items))
    }

    // MARK: - Directory outputs

    func testRunArtifactsSortADepthAndAGeometryDirectory() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("vision-artifacts-\(UUID().uuidString)")
        let output = root.appendingPathComponent("portrait-a1b2", isDirectory: true)
        let views = output.appendingPathComponent("views", isDirectory: true)
        try FileManager.default.createDirectory(at: views, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        for name in ["scene.ply", "scene.glb", "cameras.json", "scene-manifest.json", "portrait-depth.exr", "portrait-depth.png", "review.mp4"] {
            try Data([0]).write(to: output.appendingPathComponent(name))
        }
        try Data([0]).write(to: views.appendingPathComponent("000000-depth.png"))
        try FileManager.default.createDirectory(at: output.appendingPathComponent("other"), withIntermediateDirectories: true)
        try Data([0]).write(to: output.appendingPathComponent("other/ignored.png"))

        let item = StudioLibraryItem(
            id: UUID(), mode: .readImage, prompt: "", inputURL: nil, outputURL: output, createdAt: Date(), updatedAt: Date(),
            status: .completed, exitCode: 0, commandPreview: "", outputText: nil, templateID: .visionGeometry
        )
        let artifacts = StudioVisionRunArtifacts.read(item: item)
        XCTAssertEqual(artifacts.scenes.map(\.lastPathComponent), ["scene.glb", "scene.ply"], "the GLB leads")
        XCTAssertEqual(artifacts.previews.map(\.lastPathComponent), ["portrait-depth.png", "000000-depth.png"])
        XCTAssertEqual(artifacts.clips.map(\.lastPathComponent), ["review.mp4"])
        XCTAssertEqual(artifacts.documents.map(\.lastPathComponent), ["cameras.json", "scene-manifest.json"])
        XCTAssertFalse(artifacts.isEmpty)
        XCTAssertTrue(StudioVisionRunArtifacts(documents: artifacts.documents).isEmpty, "documents alone give the canvas nothing to show")

        let missing = StudioLibraryItem(
            id: UUID(), mode: .readImage, prompt: "", inputURL: nil, outputURL: root.appendingPathComponent("gone"),
            createdAt: Date(), updatedAt: Date(), status: .completed, exitCode: 0, commandPreview: "", outputText: nil
        )
        XCTAssertTrue(StudioVisionRunArtifacts.read(item: missing).isEmpty)
    }

    // MARK: - Argv parity with the Vision Lab page

    /// The page built a `CommandDraft` per variant from its own fields; the workspace's draft is
    /// the console form seeded from that same command, so the argv for the page's settings is
    /// the page's argv, with the destination routing fills where the page had a path field.
    func testTaskDraftsBuildThePagesArgvForItsSettings() throws {
        let root = "/tmp/vision-out"
        let cases: [(CommandTemplateID, (inout CommandDraft) -> Void)] = [
            (.visionFaceDetect, { draft in
                draft.inputPath = "/tmp/a.png"
                draft.visionFaceScoreThreshold = 0.65
                draft.visionExecutionProvider = "auto"
                draft.visionMaxFaces = 3
                draft.visionIncludeEmbeddings = true
                draft.json = true
                draft.visionJSONOutputPath = root + "/result.json"
            }),
            (.visionFaceEmbed, { draft in
                draft.inputPath = "/tmp/a.png"
                draft.visionFaceScoreThreshold = 0.5
                draft.visionExecutionProvider = "coreml"
                draft.visionFaceIndex = "2"
                draft.json = true
                draft.visionJSONOutputPath = root + "/result.json"
            }),
            (.visionFaceCompare, { draft in
                draft.inputPath = "/tmp/a.png"
                draft.visionSecondInputPath = "/tmp/b.png"
                draft.visionFaceScoreThreshold = 0.65
                draft.visionExecutionProvider = "auto"
                draft.visionReferenceFaceIndex = "1"
                draft.visionCandidateFaceIndex = "0"
                draft.json = true
                draft.visionJSONOutputPath = root + "/result.json"
            }),
            (.visionFaceBatch, { draft in
                draft.inputPath = "/tmp/a.png"
                draft.visionAdditionalInputs = "/tmp/b.png\n/tmp/c.png"
                draft.visionInputList = "/tmp/list.txt"
                draft.visionFaceScoreThreshold = 0.65
                draft.visionExecutionProvider = "cpu"
                draft.visionMaxFaces = 0
                draft.visionIncludeEmbeddings = false
                draft.visionFailFast = true
                draft.json = true
                draft.visionJSONLOutput = root + "/faces.jsonl"
            }),
            (.visionPose, { draft in
                draft.inputPath = "/tmp/a.png"
                draft.visionPoseBody = true
                draft.visionPoseHands = false
                draft.visionPoseFace = true
                draft.visionMaxHands = 2
                draft.visionMinimumConfidence = 0.1
                draft.json = true
                draft.visionJSONOutputPath = root + "/result.json"
            }),
            (.visionFlow, { draft in
                draft.inputPath = "/tmp/a.png"
                draft.visionSecondInputPath = "/tmp/b.png"
                draft.visionFlowAccuracy = "very-high"
                draft.json = true
                draft.outputPath = root + "/motion.flo"
                draft.visionJSONOutputPath = root + "/motion.json"
            }),
            (.visionDepth, { draft in
                draft.inputPath = "/tmp/a.png"
                draft.visionMaxEdge = 1_024
                draft.visionNative = false
                draft.visionCheckpoint = "log-stage2"
                draft.dryRun = false
                draft.json = true
                draft.outputPath = root
            }),
            (.visionDepthVideo, { draft in
                draft.inputPath = "/tmp/a.mp4"
                draft.visionInputSize = 518
                draft.visionMaxFrames = 240
                draft.dryRun = false
                draft.json = true
                draft.outputPath = root
            }),
            (.visionGeometry, { draft in
                draft.inputPath = "/tmp/a.png"
                draft.visionResolutionLevel = 9
                draft.visionTokenCount = 0
                draft.visionMaxPoints = 0
                draft.dryRun = false
                draft.json = true
                draft.outputPath = root
            }),
            (.visionGeometryMultiview, { draft in
                draft.inputPath = "/tmp/a.png"
                draft.visionAdditionalInputs = "/tmp/b.png"
                draft.camerasPath = "/tmp/cameras.json"
                draft.visionProcessResolution = 504
                draft.visionReferenceView = "saddle-balanced"
                draft.visionConfidencePercentile = 40
                draft.visionMaxPoints = 0
                draft.dryRun = false
                draft.json = true
                draft.outputPath = root
            }),
            (.visionTrackLive, { draft in
                draft.prompt = "a person\nthe red mug"
                draft.visionCamera = 1
                draft.durationSeconds = 10
                draft.visionInitFrame = 0
                draft.visionSeedSearchFrames = 30
                draft.visionThreshold = 0.05
                draft.visionResolution = 1_008
                draft.force = true
                draft.visionShowLabels = true
                draft.json = true
                draft.outputPath = root + "/live-tracking.mp4"
                draft.visionJSONOutputPath = root + "/live-tracking.json"
            }),
        ]
        for (templateID, configure) in cases {
            let template = try XCTUnwrap(CommandCatalog.template(id: templateID))
            var page = template.defaultDraft()
            configure(&page)
            let pageArgv = template.arguments(from: page)
            let draft = StudioTaskDraft(templateID: templateID, form: StudioConsoleCommand.seed(template: template, draft: page))
            XCTAssertEqual(draft.arguments, pageArgv, "\(templateID)")
            XCTAssertEqual(draft.request()?.execution?.arguments, pageArgv, "\(templateID) request")
            XCTAssertEqual(draft.request()?.mode, template.libraryMode, "\(templateID) attribution")
        }
    }

    /// Faces ▸ Compare reached from Detect keeps the picture as the reference and the second
    /// picture goes to the candidate positional; Batch from Detect keeps it as the first image.
    func testFaceVariantsKeepThePictureAcrossTheSwitch() {
        var draft = StudioTaskDraft(templateID: .visionFaceDetect)
        StudioTaskSchema.slots(for: .visionFaceDetect)[0].attach([URL(fileURLWithPath: "/tmp/a.png")], to: &draft)
        draft.switchTemplate(to: .visionFaceCompare)
        XCTAssertEqual(draft.argument(0), "/tmp/a.png")
        StudioTaskSchema.slots(for: .visionFaceCompare)[1].attach([URL(fileURLWithPath: "/tmp/b.png")], to: &draft)
        XCTAssertEqual(Array(draft.arguments.prefix(5)), ["vision", "face", "compare", "/tmp/a.png", "/tmp/b.png"])
        draft.form["--reference-face-index"] = .integer(1)
        XCTAssertEqual(draft.arguments.firstIndex(of: "--reference-face-index").map { draft.arguments[$0 + 1] }, "1")
        XCTAssertEqual(StudioTaskSchema.overrideID(forFlag: "--reference-face-index", templateID: .visionFaceCompare), .faceIndex)
        XCTAssertEqual(StudioTaskSchema.overrideID(forFlag: "--cameras", templateID: .visionGeometryMultiview), .cameras)
    }
}
