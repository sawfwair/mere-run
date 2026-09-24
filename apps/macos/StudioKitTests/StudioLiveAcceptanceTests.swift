@testable import StudioKit
import AVFoundation
import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers
import XCTest

/// Headless live acceptance of the Studio flows against the real `mere.run` CLI and the models
/// installed on this Mac.
///
/// Each test builds the command exactly the way its Studio page does (`StudioCommandAdapter` for
/// composer tasks, the page's own `CommandDraft` for specialist pages), runs the CLI, and decodes
/// the real output with the Studio decoder the page uses. Inputs are drawn or synthesized here, or
/// generated with the CLI (a portrait for Faces and Find, a photo of an apple for Segment), so
/// nothing binary is committed. The CLI's argv, streams, and exit code for every step are kept
/// under `<dir>/<flow>/` and one line per flow is appended to `<dir>/summary.log`.
///
/// Skipped unless `MERERUN_LIVE_ACCEPTANCE_DIR` names a directory, mirroring `StudioSnapshotTests`,
/// and each test skips on its own when a model it needs is not installed. The CLI is
/// `MERERUN_LIVE_CLI`, or the `mere.run` beside this test bundle in the build products
/// (`swift build --show-bin-path`), so `swift build` first. A full pass loads a dozen models and
/// takes several minutes; one flow at a time is `--filter StudioLiveAcceptanceTests/test04`.
///
///     MERERUN_LIVE_ACCEPTANCE_DIR=/tmp/live swift test --filter StudioLiveAcceptanceTests
final class StudioLiveAcceptanceTests: XCTestCase {
    private var live: URL!
    private var cli: URL!
    private var stepCounter = 0

    override func setUpWithError() throws {
        try super.setUpWithError()
        guard let directory = Self.liveDirectory() else {
            throw XCTSkip("Set MERERUN_LIVE_ACCEPTANCE_DIR to a directory to run the live Studio acceptance pass.")
        }
        live = directory
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        cli = Self.cliURL()
        guard FileManager.default.isExecutableFile(atPath: cli.path) else {
            throw XCTSkip("No mere.run binary at \(cli.path); build first or set MERERUN_LIVE_CLI.")
        }
        // Every Studio destination (composer and specialist pages alike) files under the configured
        // root, so the run's outputs land in the live directory instead of ~/Pictures etc. The root
        // lives in this process only: a suite named once per process, set through its registration
        // domain, which cfprefsd never persists — so two `swift test --filter …/testNN` processes
        // running at once cannot overwrite or clear each other's root.
        let suite = try XCTUnwrap(UserDefaults(suiteName: "run.mere.studio.live-acceptance.\(UUID().uuidString)"))
        suite.register(defaults: [StudioOutputLocation.rootDefaultsKey: directory.path])
        StudioOutputLocation.defaults = suite
        let probe = StudioOutputLocation.outputDirectoryURL(domain: .text, prompt: "probe", fallbackStem: "probe").path
        XCTAssertTrue(probe.hasPrefix(directory.path), "The configured root did not take: destinations would file under \(probe)")
        continueAfterFailure = true
    }

    override func tearDownWithError() throws {
        StudioOutputLocation.defaults = .standard
        try super.tearDownWithError()
    }

    // MARK: - Text ▸ Decisions

    func test01DecisionsExampleDecidesAndChecksFit() throws {
        try requireModels(["text-decide-laya"])
        let flow = "01-decisions"
        let document = StudioDecisionDocument.example
        XCTAssertEqual(document.problems, [])

        // The page writes the request beside the output, in the Text domain's folder.
        let directory = StudioOutputLocation.outputDirectoryURL(
            domain: .text,
            prompt: document.questions.first?.prompt ?? document.text,
            fallbackStem: "decisions"
        )
        let requestURL = directory.appendingPathComponent("request.json")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try document.requestJSON().write(to: requestURL, options: .atomic)
        XCTAssertTrue(directory.path.hasPrefix(live.path), "The Decisions request should be filed under the configured root: \(directory.path)")

        let template = try XCTUnwrap(CommandCatalog.template(id: .textDecide))
        func command(preflight: Bool) -> CommandDraft {
            var command = CommandDraft()
            command.inputPath = requestURL.path
            command.model = "text-decide-laya"
            command.outputPath = directory.appendingPathComponent(preflight ? "fit.json" : "decisions.json").path
            command.preflight = preflight
            command.force = true
            return command
        }

        // Decide.
        let decide = command(preflight: false)
        let decided = try runCLI(flow, template.arguments(from: decide), timeout: 900)
        XCTAssertEqual(decided.exitCode, 0, decided.failureDescription)
        let fromFile = (try? Data(contentsOf: URL(fileURLWithPath: decide.outputPath))).flatMap(StudioDecisionOutput.init(data:))
        let fromText = StudioDecisionOutput(outputText: decided.libraryOutputText)
        let output = try XCTUnwrap(fromFile ?? fromText, "Neither the -o file nor stdout decoded as a decision document")
        XCTAssertNotNil(fromFile, "The page reads the -o file first; it did not decode")
        XCTAssertNotNil(fromText, "The page falls back to the captured stdout; it did not decode")
        let result = try XCTUnwrap(output.result, "Decide should produce answers, not only a plan")
        XCTAssertEqual(Set(result.answers.keys), Set(document.resolvedKeys))
        XCTAssertEqual(result.plan.questions.map(\.id), document.resolvedKeys)
        let department = try XCTUnwrap(result.answers["department"])
        XCTAssertEqual(department.type, "choice")
        XCTAssertEqual(department.choice, "billing", "The handbook example routes a duplicate charge to billing")
        XCTAssertEqual(Set(department.probabilities.keys), ["billing", "technical support", "sales"])
        let urgency = try XCTUnwrap(result.answers["urgency"])
        XCTAssertEqual(urgency.type, "score")
        XCTAssertNotNil(urgency.score)
        XCTAssertEqual(Set(urgency.probabilities.keys), ["0", "1", "2"], "Score probabilities are keyed by level index, which the answer card relies on")
        let refund = try XCTUnwrap(result.answers["refund"])
        XCTAssertEqual(refund.type, "noul")
        XCTAssertGreaterThanOrEqual(refund.noul ?? 0, 0.5, "The customer asks for a refund")
        XCTAssertEqual(Set(refund.probabilities.keys), ["false", "true"])

        // Check fit.
        let fit = command(preflight: true)
        let checked = try runCLI(flow, template.arguments(from: fit), timeout: 300)
        XCTAssertEqual(checked.exitCode, 0, checked.failureDescription)
        let fitOutput = try XCTUnwrap(
            (try? Data(contentsOf: URL(fileURLWithPath: fit.outputPath))).flatMap(StudioDecisionOutput.init(data:))
                ?? StudioDecisionOutput(outputText: checked.libraryOutputText)
        )
        XCTAssertNil(fitOutput.result, "--preflight writes only the plan")
        XCTAssertEqual(fitOutput.plan.questions.count, 3)
        XCTAssertEqual(fitOutput.plan.questions.map(\.id), document.resolvedKeys)
        XCTAssertGreaterThan(fitOutput.plan.maxTokens, 0)
        XCTAssertFalse(fitOutput.plan.questions.contains { $0.wasTruncated }, "The short example should fit without cuts")
        conclude(flow, "department=\(department.choice ?? "-") urgency=\(urgency.score.map { String(format: "%.2f", $0) } ?? "-") refund=\(refund.noul.map { String(format: "%.2f", $0) } ?? "-") fit=\(fitOutput.plan.questions.map(\.inputTokens))")
    }

    // MARK: - Vision ▸ Segment (drawn prompts)

    /// A box with a positive and a negative point drawn on a photo is one object: Studio labels the
    /// points after the box (`StudioRegionPromptText.pointLines`) and the CLI groups them
    /// (`SAM31PromptSet.normalized`), so the result is one mask on the apple.
    func test02SegmentWithDrawnBoxAndPointsFindsTheApple() throws {
        try requireModels(["vision-segment-sam31", "image-klein-nano"])
        let flow = "02-segment-drawn"
        let photo = try applePhoto(flow: flow)
        let apple = try appleRect(flow: flow)
        let box = apple.insetBy(dx: -20, dy: -20)
        // The negative point sits inside the box but on the table, the correction the box needs.
        let table = CGPoint(x: box.minX + 8, y: box.maxY - 8)

        var draft = StudioDraft()
        draft.reset(for: .segment)
        draft.prompt = ""
        draft.inputPath = photo.path
        draft.visionRegionPrompts = [
            .box(box, label: "apple"),
            .point(CGPoint(x: apple.midX, y: apple.midY), isPositive: true),
            .point(table, isPositive: false),
        ]
        let (request, argv) = try composerRequest(mode: .segment, draft: draft)
        XCTAssertEqual(argv.first, "vision")
        XCTAssertEqual(argv.filter { $0 == "--box" }.count, 1)
        XCTAssertTrue(argv.contains("\(Self.pixels(box.minX)),\(Self.pixels(box.minY)),\(Self.pixels(box.maxX)),\(Self.pixels(box.maxY)),apple"), "Drawn box encodes as whole pixels with its label: \(argv)")
        XCTAssertTrue(argv.contains("\(Self.pixels(apple.midX)),\(Self.pixels(apple.midY)),positive,apple"), "The positive point carries the box's label: \(argv)")
        XCTAssertTrue(argv.contains("\(Self.pixels(table.x)),\(Self.pixels(table.y)),negative,apple"), "The negative point carries the box's label: \(argv)")
        XCTAssertFalse(argv.contains("--prompt"), "No text prompt was typed")
        XCTAssertTrue(request.draft.outputPath.hasPrefix(live.path))
        XCTAssertEqual(URL(fileURLWithPath: request.draft.visionJSONOutputPath).deletingPathExtension().path, URL(fileURLWithPath: request.draft.outputPath).deletingPathExtension().path)

        let run = try runCLI(flow, argv, timeout: 900)
        XCTAssertEqual(run.exitCode, 0, run.failureDescription)
        let document = try decodeAnalyzeDocument(at: request.draft.visionJSONOutputPath)
        guard case .segmentation(let segmentation) = document else {
            return XCTFail("Expected a segmentation document, decoded \(document)")
        }
        // The CLI records `standardizedFileURL.path`, which drops a `/private` prefix; the page compares standardized URLs too.
        XCTAssertEqual(URL(fileURLWithPath: segmentation.inputImagePath).standardizedFileURL.path, photo.standardizedFileURL.path)
        let detections = document.detections(imageSize: CGSize(width: 640, height: 480))
        XCTAssertEqual(detections.count, 1, "One box with its points is one object; got \(detections.map(\.boxDescription))")
        let best = detections.map { Self.iou($0.box, apple) }.max() ?? 0
        XCTAssertGreaterThanOrEqual(best, 0.6, "The drawn prompts' mask should agree with the text prompt's; apple=\(Self.description(of: apple)) boxes: \(detections.map(\.boxDescription))")
        XCTAssertEqual(detections.first?.label, "apple")
        XCTAssertTrue(FileManager.default.fileExists(atPath: request.draft.outputPath), "The annotated PNG should exist at \(request.draft.outputPath)")
        XCTAssertTrue(FileManager.default.fileExists(atPath: request.draft.visionMaskOutputDirectory), "The mask directory should exist at \(request.draft.visionMaskOutputDirectory)")
        for detection in detections {
            if let mask = detection.maskURL {
                XCTAssertTrue(FileManager.default.fileExists(atPath: mask.path), "maskPath does not exist: \(mask.path)")
            }
        }
        XCTAssertEqual(document.summary(detectionCount: detections.count), "1 object found")
        conclude(flow, "apple=\(Self.description(of: apple)) detections=\(detections.count) bestIoU=\(String(format: "%.2f", best)) boxes=\(detections.map(\.boxDescription)) masks=\(detections.compactMap(\.maskURL).count)")
    }

    // MARK: - Vision ▸ Segment (EXIF orientation 6)

    func test03SegmentOnRotatedJPEGMapsDisplayPromptsToStoredPixels() throws {
        try requireModels(["vision-segment-sam31", "image-klein-nano"])
        let flow = "03-segment-rotated"
        let photo = try applePhoto(flow: flow)
        let apple = try appleRect(flow: flow)
        let image = try Self.rotatedJPEG(of: photo, in: fixtures(), name: "apple-orientation6.jpg", exifOrientation: 6)

        let metadata = try XCTUnwrap(StudioImageMetadata.read(image))
        XCTAssertEqual(metadata.storedSize, CGSize(width: 640, height: 480))
        XCTAssertEqual(metadata.orientation, .right, "ImageIO should report EXIF orientation 6")
        XCTAssertEqual(metadata.displaySize, CGSize(width: 480, height: 640))
        let orientation = metadata.orientation
        let storedSize = metadata.storedSize

        // The page shows the upright picture; the box is drawn there and written back stored.
        let displayApple = orientation.displayRect(fromStored: apple, storedSize: storedSize)
        XCTAssertEqual(displayApple.width, apple.height, accuracy: 0.001)
        XCTAssertEqual(displayApple.height, apple.width, accuracy: 0.001)
        let drawn: [StudioRegionPrompt] = [
            .box(displayApple.insetBy(dx: -20, dy: -20), label: "apple"),
            .point(CGPoint(x: displayApple.midX, y: displayApple.midY), isPositive: true),
        ]
        let stored = drawn.inStoredSpace(orientation, storedSize: storedSize)
        let storedBox = try XCTUnwrap(stored.first?.rect)
        XCTAssertEqual(storedBox.minX, apple.minX - 20, accuracy: 0.001)
        XCTAssertEqual(storedBox.minY, apple.minY - 20, accuracy: 0.001)
        XCTAssertEqual(storedBox.maxX, apple.maxX + 20, accuracy: 0.001)
        XCTAssertEqual(storedBox.maxY, apple.maxY + 20, accuracy: 0.001)
        let storedPoint = try XCTUnwrap(stored.last?.point)
        XCTAssertEqual(storedPoint.x, apple.midX, accuracy: 0.001)
        XCTAssertEqual(storedPoint.y, apple.midY, accuracy: 0.001)
        // Round trip, the way the layer's binding reads them back.
        XCTAssertEqual(stored.inDisplaySpace(orientation, storedSize: storedSize).first?.rect?.minX ?? -1, displayApple.minX - 20, accuracy: 0.001)

        var draft = StudioDraft()
        draft.reset(for: .segment)
        draft.prompt = ""
        draft.inputPath = image.path
        draft.visionRegionPrompts = stored
        let (request, argv) = try composerRequest(mode: .segment, draft: draft)
        XCTAssertTrue(argv.contains("\(Self.pixels(storedBox.minX)),\(Self.pixels(storedBox.minY)),\(Self.pixels(storedBox.maxX)),\(Self.pixels(storedBox.maxY)),apple"), "The box must reach the CLI in stored pixels: \(argv)")

        let run = try runCLI(flow, argv, timeout: 900)
        XCTAssertEqual(run.exitCode, 0, run.failureDescription)
        let document = try decodeAnalyzeDocument(at: request.draft.visionJSONOutputPath)
        guard case .segmentation = document else {
            return XCTFail("Expected a segmentation document, decoded \(document)")
        }
        let detections = document.detections(imageSize: storedSize)
        XCTAssertEqual(detections.count, 1, "SAM should find the one prompted object on the rotated JPEG; got \(detections.map(\.boxDescription))")
        let best = detections.map { Self.iou($0.box, apple) }.max() ?? 0
        XCTAssertGreaterThanOrEqual(best, 0.6, "The CLI's box should be in stored space on the apple; boxes: \(detections.map(\.boxDescription))")
        for detection in detections {
            XCTAssertLessThanOrEqual(detection.box.maxX, storedSize.width + 1, "A result past the stored width means the CLI applied the EXIF transform")
            XCTAssertLessThanOrEqual(detection.box.maxY, storedSize.height + 1)
        }
        // And the page maps the result back onto the upright picture.
        if let first = detections.first {
            let shown = orientation.displayRect(fromStored: first.box, storedSize: storedSize)
            XCTAssertGreaterThanOrEqual(Self.iou(shown, displayApple), 0.5)
        }
        conclude(flow, "orientation=\(orientation) stored=\(Int(storedSize.width))x\(Int(storedSize.height)) detections=\(detections.count) bestIoU=\(String(format: "%.2f", best)) boxes=\(detections.map(\.boxDescription))")
    }

    // MARK: - Vision ▸ Track (prompt frame and end frame)

    /// `vision track --init-frame 10 --end-frame 30` seeds on frame 10 and tracks frames 0–30
    /// (`SAM31VideoTracker.track` propagates both ways from the seed), which is what Studio's frame
    /// scrubber now says.
    func test04TrackSeedsOnThePromptFrameAndTracksFromFrameZero() throws {
        try requireModels(["vision-segment-sam31"])
        let flow = "04-track"
        let video = try Self.movingSquareVideo(in: fixtures())

        // The page measures the clip and builds the frame grid the scrubber counts in.
        let asset = AVURLAsset(url: video)
        let track = try XCTUnwrap(asset.tracks(withMediaType: .video).first)
        let grid = StudioVideoFrameGrid(duration: CMTimeGetSeconds(asset.duration), frameRate: Double(track.nominalFrameRate))
        XCTAssertEqual(grid.frameCount, Self.videoFrameCount, "The grid should count the clip's \(Self.videoFrameCount) frames")
        let seed = 10
        let end = 30
        let seedSquare = Self.videoSquare(atFrame: seed)

        var draft = StudioDraft()
        draft.reset(for: .track)
        draft.prompt = ""
        draft.inputPath = video.path
        draft.visionRegionPrompts = [.box(seedSquare.insetBy(dx: -10, dy: -10), label: "square")]
        draft.visionInitFrame = seed
        draft.visionEndFrame = end
        let (request, argv) = try composerRequest(mode: .track, draft: draft)
        XCTAssertEqual(argv.firstIndex(of: "--init-frame").map { argv[$0 + 1] }, "10")
        XCTAssertEqual(argv.firstIndex(of: "--end-frame").map { argv[$0 + 1] }, "30")
        XCTAssertTrue(request.draft.visionMaskOutputDirectory.isBlank, "Track does not ask for per-frame masks")
        let promised = grid.trackRangeDescription(promptFrame: seed, endFrame: end)
        XCTAssertEqual(promised, "Prompts on frame 10 · tracks frames 0–30")

        let run = try runCLI(flow, argv, timeout: 1_800)
        XCTAssertEqual(run.exitCode, 0, run.failureDescription)
        let document = try decodeAnalyzeDocument(at: request.draft.visionJSONOutputPath)
        guard case .tracking(let tracking) = document else {
            return XCTFail("Expected a tracking document, decoded \(document)")
        }
        XCTAssertEqual(tracking.frameWidth, 640)
        XCTAssertEqual(tracking.frameHeight, 360)
        XCTAssertEqual(tracking.objects.count, 1, "One drawn box seeds one object")
        XCTAssertEqual(tracking.objects.first?.seedFrameIndex, seed)
        let indices = tracking.frames.map(\.frameIndex)
        XCTAssertEqual(indices.first, 0, "The CLI tracks from the first frame whatever the prompt frame; got \(indices.prefix(3))")
        XCTAssertEqual(indices.last, end, "Tracking should stop on the inclusive end frame; got \(indices.suffix(3))")
        XCTAssertEqual(indices.count, end + 1, "Frame range should be 0…end inclusive, as Studio promises: \(promised)")
        let visibleFrames = tracking.frames.filter { $0.detections.contains { $0.visible } }
        XCTAssertGreaterThanOrEqual(visibleFrames.count, end / 2, "The square should stay visible for most of the range")
        var worstDrift = 0.0
        for frame in tracking.frames {
            guard let detection = frame.detections.first, detection.visible else { continue }
            let expected = Self.videoSquare(atFrame: frame.frameIndex)
            let box = document.detections(imageSize: CGSize(width: 640, height: 360), frame: frame.frameIndex).first?.box ?? .zero
            worstDrift = max(worstDrift, abs(box.midX - expected.midX))
        }
        XCTAssertLessThanOrEqual(worstDrift, 40, "The tracked box should follow the moving square; worst horizontal drift \(worstDrift)px")
        XCTAssertTrue(FileManager.default.fileExists(atPath: request.draft.outputPath), "The annotated clip should exist at \(request.draft.outputPath)")
        conclude(flow, "\(promised) → frames=\(indices.first ?? -1)…\(indices.last ?? -1) (\(indices.count)) seed=\(tracking.objects.first?.seedFrameIndex ?? -1) visible=\(visibleFrames.count) worstDrift=\(String(format: "%.1f", worstDrift))px fps=\(tracking.fps)")
    }

    // MARK: - Vision ▸ Find → Segment handoff

    func test05FindDetectionsHandOffToSegmentAsBoxPrompts() throws {
        try requireModels(["vision-ground-falcon-perception", "vision-segment-sam31"])
        let flow = "05-find-to-segment"
        let squareImage = try Self.squareImage(in: fixtures(), name: "square-640x480.png")

        // Find on the synthetic square; a real photo of a person is the fallback subject.
        var candidates: [(URL, String, CGSize)] = [(squareImage, "red square", CGSize(width: 640, height: 480))]
        if let face = try? faceImage(flow: flow) {
            candidates.append((face, "a person", CGSize(width: 512, height: 512)))
        }
        var found: (image: URL, prompt: String, size: CGSize, document: StudioAnalyzeDocument, detections: [StudioAnalyzeDetection])?
        for (image, prompt, size) in candidates {
            var find = StudioDraft()
            find.reset(for: .findObjects)
            find.prompt = prompt
            find.inputPath = image.path
            let (request, argv) = try composerRequest(mode: .findObjects, draft: find)
            XCTAssertTrue(argv.contains("--query") || argv.contains("--prompt"), "Find sends the query: \(argv)")
            XCTAssertFalse(argv.contains("--box"), "Find takes no box prompts")
            let run = try runCLI(flow, argv, timeout: 900)
            XCTAssertEqual(run.exitCode, 0, run.failureDescription)
            let document = try decodeAnalyzeDocument(at: request.draft.visionJSONOutputPath)
            guard case .ground = document else {
                XCTFail("Expected a ground document, decoded \(document)")
                continue
            }
            let detections = document.detections(imageSize: size)
            if !detections.isEmpty {
                found = (image, prompt, size, document, detections)
                break
            }
        }
        guard let found else {
            return XCTFail("Find returned no detections for either the synthetic square or the generated portrait")
        }
        // Ground boxes are normalized; the pixel boxes must sit inside the picture.
        for detection in found.detections {
            XCTAssertGreaterThanOrEqual(detection.box.minX, -1)
            XCTAssertLessThanOrEqual(detection.box.maxX, found.size.width + 1, "Ground detection outside the image: \(detection.boxDescription)")
            XCTAssertLessThanOrEqual(detection.box.maxY, found.size.height + 1)
        }
        if found.image == squareImage, let first = found.detections.first {
            XCTAssertGreaterThanOrEqual(Self.iou(first.box, Self.squareRect), 0.3, "Find's box should land on the square: \(first.boxDescription)")
        }

        // "Segment these" carries the input and the found boxes as drawn prompts.
        let handoff = StudioAnalyzeHandoff.make(to: .visionSegment, inputPath: found.image.path, prompt: found.prompt, detections: found.detections)
        XCTAssertEqual(handoff.inputPath, found.image.path)
        XCTAssertEqual(handoff.regionPrompts.count, found.detections.count)
        XCTAssertEqual(handoff.regionPrompts.first?.label, found.detections.first?.label)
        var segment = StudioDraft()
        segment.reset(for: .segment)
        segment.visionRegionPrompts = [.point(CGPoint(x: 1, y: 1), isPositive: true, label: "stale")]
        handoff.apply(to: &segment)
        XCTAssertEqual(segment.inputPath, found.image.path)
        XCTAssertEqual(segment.prompt, found.prompt)
        XCTAssertEqual(segment.visionRegionPrompts?.count, found.detections.count, "The previous picture's prompts must be replaced by the found boxes")
        let (request, argv) = try composerRequest(mode: .segment, draft: segment)
        XCTAssertEqual(argv.filter { $0 == "--box" }.count, found.detections.count)
        let run = try runCLI(flow, argv, timeout: 900)
        XCTAssertEqual(run.exitCode, 0, run.failureDescription)
        let document = try decodeAnalyzeDocument(at: request.draft.visionJSONOutputPath)
        guard case .segmentation = document else {
            return XCTFail("Expected a segmentation document, decoded \(document)")
        }
        let segmented = document.detections(imageSize: found.size)
        XCTAssertFalse(segmented.isEmpty, "Segment found nothing for the handed-off boxes")
        let overlap = segmented.map { seg in found.detections.map { Self.iou(seg.box, $0.box) }.max() ?? 0 }.max() ?? 0
        XCTAssertGreaterThanOrEqual(overlap, 0.4, "Segment's mask box should overlap the box Find handed over; segment=\(segmented.map(\.boxDescription)) find=\(found.detections.map(\.boxDescription))")
        conclude(flow, "subject=\(found.image.lastPathComponent) query=\"\(found.prompt)\" found=\(found.detections.count) \(found.detections.map(\.boxDescription)) segmented=\(segmented.count) overlap=\(String(format: "%.2f", overlap))")
    }

    // MARK: - Vision ▸ Faces

    func test06FaceDetectionResultDecodesForTheClickToPickOverlay() throws {
        try requireModels(["vision-face-buffalo-l"])
        let flow = "06-faces"
        let face = try? faceImage(flow: flow)
        let subject = try face ?? Self.squareImage(in: fixtures(), name: "square-640x480.png")
        let expectedSize = face == nil ? (640, 480) : (512, 512)

        // The Vision Lab page's draft for Face detection.
        let template = try XCTUnwrap(CommandCatalog.template(id: .visionFaceDetect))
        let root = StudioOutputLocation.specialistDirectory(domain: .vision, name: "vision", configuredRoot: live.path)
        var draft = template.defaultDraft()
        draft.inputPath = subject.path
        draft.model = template.defaultDraft().model
        draft.visionFaceScoreThreshold = 0.65
        draft.visionExecutionProvider = "auto"
        draft.visionMaxFaces = 0
        draft.visionIncludeEmbeddings = false
        draft.json = true
        draft.visionJSONOutputPath = root.appendingPathComponent("result.json").path
        let (request, argv) = try specialistRequest(templateID: .visionFaceDetect, mode: .readImage, draft: draft)
        XCTAssertEqual(Array(argv.prefix(3)), ["vision", "face", "detect"])
        XCTAssertTrue(argv.contains("--json-output"))

        let run = try runCLI(flow, argv, timeout: 900)
        XCTAssertEqual(run.exitCode, 0, run.failureDescription)
        let result = try XCTUnwrap(StudioFaceOverlayResult.load(from: URL(fileURLWithPath: request.draft.visionJSONOutputPath)), "The overlay's decoder rejected the face document")
        XCTAssertEqual(result.width, expectedSize.0)
        XCTAssertEqual(result.height, expectedSize.1)
        for record in result.faces {
            XCTAssertEqual(record.detection.landmarks.count, 5, "Buffalo-L reports five landmarks")
            XCTAssertGreaterThanOrEqual(record.detection.score, 0.65)
            XCTAssertGreaterThanOrEqual(record.detection.boundingBox.x, -1)
            XCTAssertLessThanOrEqual(record.detection.boundingBox.x + record.detection.boundingBox.width, Double(result.width) + 1)
        }
        XCTAssertEqual(result.faces.map(\.index), Array(0..<result.faces.count), "Faces are numbered the way --face-index counts them")
        if face != nil {
            XCTAssertGreaterThanOrEqual(result.faces.count, 1, "The generated portrait should contain a detectable face")
        }
        // The stdout JSON the page ignores should also be the same document.
        XCTAssertEqual(try JSONDecoder().decode(StudioFaceOverlayResult.self, from: Data(run.stdout.utf8)), result)
        conclude(flow, "subject=\(subject.lastPathComponent) faces=\(result.faces.count) scores=\(result.faces.map { String(format: "%.2f", $0.detection.score) }) json=\(request.draft.visionJSONOutputPath)")
    }

    // MARK: - Vision ▸ Depth (still)

    func test07DepthStillWritesIntoTheSpecialistDirectory() throws {
        try requireModels(["vision-depth-marigold-v2"])
        let flow = "07-depth"
        let subject = try (try? faceImage(flow: flow)) ?? Self.squareImage(in: fixtures(), name: "square-640x480.png")
        let template = try XCTUnwrap(CommandCatalog.template(id: .visionDepth))
        let root = StudioOutputLocation.specialistDirectory(domain: .vision, name: "vision", configuredRoot: live.path)
        XCTAssertEqual(root.deletingLastPathComponent().path, live.appendingPathComponent("Vision").path)
        XCTAssertTrue(root.lastPathComponent.hasPrefix("vision-"))

        var draft = template.defaultDraft()
        draft.inputPath = subject.path
        draft.model = template.defaultDraft().model
        draft.visionMaxEdge = 1_024
        draft.visionNative = false
        draft.visionCheckpoint = nil
        draft.dryRun = false
        draft.json = true
        draft.outputPath = root.path
        let (request, argv) = try specialistRequest(templateID: .visionDepth, mode: .readImage, draft: draft)
        XCTAssertEqual(Array(argv.prefix(2)), ["vision", "depth"])
        XCTAssertTrue(argv.contains("--json"))
        XCTAssertEqual(argv.firstIndex(of: "--max-edge").map { argv[$0 + 1] }, "1024")

        let run = try runCLI(flow, argv, timeout: 1_800)
        XCTAssertEqual(run.exitCode, 0, run.failureDescription)
        let object = try XCTUnwrap(StudioStructuredOutput.objectData(in: run.stdout).flatMap { try? JSONSerialization.jsonObject(with: $0) } as? [String: Any], "--json should print one JSON object")
        let files = (try? FileManager.default.contentsOfDirectory(atPath: request.draft.outputPath)) ?? []
        XCTAssertFalse(files.isEmpty, "The CLI should have written into \(request.draft.outputPath)")
        XCTAssertTrue(files.contains { $0.lowercased().hasSuffix(".png") }, "Expected a depth preview PNG in \(files)")
        conclude(flow, "output=\(request.draft.outputPath) files=\(files.sorted()) jsonKeys=\(object.keys.sorted())")
    }

    // MARK: - Vision ▸ Multi-view geometry (cameras)

    func test08GeometryMultiviewAcceptsStudioCamerasAndRejectsWrongSizes() throws {
        try requireModels(["vision-geometry-da3-small"])
        let flow = "08-geometry-multiview"
        let viewA = try Self.squareImage(in: fixtures(), name: "view-a-320x240.png", size: CGSize(width: 320, height: 240), square: CGRect(x: 100, y: 60, width: 80, height: 80))
        let viewB = try Self.squareImage(in: fixtures(), name: "view-b-320x240.png", size: CGSize(width: 320, height: 240), square: CGRect(x: 124, y: 60, width: 80, height: 80))
        let views = [viewA, viewB].map { StudioCameraView(name: $0.lastPathComponent, pixelSize: StudioPixelSize.of($0)) }
        XCTAssertEqual(views.map(\.pixelSize), [StudioPixelSize(width: 320, height: 240), StudioPixelSize(width: 320, height: 240)])

        var second = StudioGeometryCamera.identity(size: views[1].pixelSize)
        second.translation = [0.1, 0, 0]
        let cameras = StudioGeometryCameraDocument(cameras: [.identity(size: views[0].pixelSize), second])
        XCTAssertEqual(cameras.problems(views: views), [])
        let template = try XCTUnwrap(CommandCatalog.template(id: .visionGeometryMultiview))

        func makeDraft(root: URL, camerasPath: String, dryRun: Bool) -> CommandDraft {
            var draft = template.defaultDraft()
            draft.inputPath = viewA.path
            draft.visionAdditionalInputs = viewB.path
            draft.model = template.defaultDraft().model
            draft.camerasPath = camerasPath
            draft.visionProcessResolution = 504
            draft.visionReferenceView = "saddle-balanced"
            draft.visionConfidencePercentile = 40
            draft.visionMaxPoints = 0
            draft.dryRun = dryRun
            draft.json = true
            draft.outputPath = root.path
            return draft
        }

        // Correct cameras: dry run, then the real solve.
        let root = StudioOutputLocation.specialistDirectory(domain: .vision, name: "vision", configuredRoot: live.path)
        let camerasURL = StudioCameraDocuments.url(besideOutputDirectory: root.path)
        try FileManager.default.createDirectory(at: camerasURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try cameras.json().write(to: camerasURL, options: .atomic)
        XCTAssertEqual(camerasURL.lastPathComponent, "\(root.lastPathComponent).cameras.json")
        XCTAssertEqual(try StudioGeometryCameraDocument.importing(Data(contentsOf: camerasURL)).cameras.map(\.imageWidth), [320, 320])

        let (_, dryArgv) = try specialistRequest(templateID: .visionGeometryMultiview, mode: .readImage, draft: makeDraft(root: root, camerasPath: camerasURL.path, dryRun: true))
        XCTAssertEqual(dryArgv.firstIndex(of: "--cameras").map { dryArgv[$0 + 1] }, camerasURL.path)
        let dry = try runCLI(flow, dryArgv, timeout: 600)
        XCTAssertEqual(dry.exitCode, 0, dry.failureDescription)
        XCTAssertNotNil(StudioStructuredOutput.objectData(in: dry.stdout), "--dry-run --json should print JSON")

        let (request, argv) = try specialistRequest(templateID: .visionGeometryMultiview, mode: .readImage, draft: makeDraft(root: root, camerasPath: camerasURL.path, dryRun: false))
        let run = try runCLI(flow, argv, timeout: 1_800)
        XCTAssertEqual(run.exitCode, 0, run.failureDescription)
        let files = (try? FileManager.default.contentsOfDirectory(atPath: request.draft.outputPath)) ?? []
        XCTAssertFalse(files.isEmpty, "The scene directory should have been written at \(request.draft.outputPath)")

        // A camera sized for another image: Studio blocks it, and the CLI must too.
        let wrong = StudioGeometryCameraDocument(cameras: [.identity(), second])
        let wrongProblems = wrong.problems(views: views)
        XCTAssertEqual(wrongProblems, ["Camera 1 is sized 1920 × 1080 but view-a-320x240.png is 320 × 240."])
        let wrongRoot = StudioOutputLocation.specialistDirectory(domain: .vision, name: "vision-wrong", configuredRoot: live.path)
        let wrongURL = StudioCameraDocuments.url(besideOutputDirectory: wrongRoot.path)
        try wrong.json().write(to: wrongURL, options: .atomic)
        let (_, wrongDryArgv) = try specialistRequest(templateID: .visionGeometryMultiview, mode: .readImage, draft: makeDraft(root: wrongRoot, camerasPath: wrongURL.path, dryRun: true))
        let wrongDry = try runCLI(flow, wrongDryArgv, timeout: 600)
        let (_, wrongArgv) = try specialistRequest(templateID: .visionGeometryMultiview, mode: .readImage, draft: makeDraft(root: wrongRoot, camerasPath: wrongURL.path, dryRun: false))
        let wrongRun = try runCLI(flow, wrongArgv, timeout: 1_800)
        XCTAssertNotEqual(wrongRun.exitCode, 0, "The CLI accepted a camera whose image size is not the image's")
        let message = (wrongRun.stderr + wrongRun.stdout).lowercased()
        XCTAssertTrue(message.contains("1920") || message.contains("dimension") || message.contains("size"), "The rejection should name the size mismatch; got: \(wrongRun.stderr.suffix(400))")
        conclude(flow, "dryRun=\(dry.exitCode) run=\(run.exitCode) files=\(files.sorted()) wrongSize: dryRun exit=\(wrongDry.exitCode) run exit=\(wrongRun.exitCode) \(StudioFailureSummary.lastMeaningfulLine(in: wrongRun.stderr) ?? "")")
    }

    // MARK: - 3D ▸ InstantMesh (cameras)

    func test09InstantMeshDryRunAcceptsStudioCameras() throws {
        try requireModels(["image-3d-instantmesh-base"])
        let flow = "09-instantmesh"
        let views = try (0..<4).map { index in
            try Self.squareImage(in: fixtures(), name: "mesh-view-\(index).png", size: CGSize(width: 256, height: 256), square: CGRect(x: 60 + index * 10, y: 70, width: 100, height: 100))
        }
        let cameras = StudioInstantMeshCameraDocument(cameras: (0..<4).map { _ in .example })
        XCTAssertEqual(cameras.problems(viewCount: 4), [])
        let template = try XCTUnwrap(CommandCatalog.template(id: .imageReconstruct3DMultiview))
        let root = StudioOutputLocation.specialistDirectory(domain: .threeD, name: "3d-asset", configuredRoot: live.path)
        let camerasURL = StudioCameraDocuments.url(besideOutputDirectory: root.path)
        try FileManager.default.createDirectory(at: camerasURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try cameras.json().write(to: camerasURL, options: .atomic)

        // Studio3DCreationView.commandDraft for the InstantMesh engine with Preflight only on.
        func makeDraft(camerasPath: String) -> CommandDraft {
            var draft = template.defaultDraft()
            draft.inputPath = ""
            draft.referenceImagePaths = views.map(\.path).joined(separator: "\n")
            draft.outputPath = root.path
            draft.model = ""
            draft.reconstructionResolution = 256
            draft.noVertexColors = false
            draft.camerasPath = camerasPath
            draft.dryRun = true
            draft.json = true
            return draft
        }
        let (_, argv) = try specialistRequest(templateID: .imageReconstruct3DMultiview, mode: .createImage, draft: makeDraft(camerasPath: camerasURL.path))
        XCTAssertEqual(argv.filter { $0 == "--view" }.count, 4)
        XCTAssertTrue(argv.contains("--dry-run"))
        let run = try runCLI(flow, argv, timeout: 600)
        XCTAssertEqual(run.exitCode, 0, run.failureDescription)
        let object = StudioStructuredOutput.objectData(in: run.stdout).flatMap { try? JSONSerialization.jsonObject(with: $0) } as? [String: Any]
        XCTAssertNotNil(object, "--dry-run --json should print a JSON plan")

        // Three cameras for four views: Studio blocks it; the CLI must too.
        let short = StudioInstantMeshCameraDocument(cameras: (0..<3).map { _ in .example })
        XCTAssertEqual(short.problems(viewCount: 4), ["Add one camera per view: 4 views, 3 cameras."])
        let shortURL = live.appendingPathComponent("\(flow)/short.cameras.json")
        try FileManager.default.createDirectory(at: shortURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try short.json().write(to: shortURL, options: .atomic)
        let (_, shortArgv) = try specialistRequest(templateID: .imageReconstruct3DMultiview, mode: .createImage, draft: makeDraft(camerasPath: shortURL.path))
        let shortRun = try runCLI(flow, shortArgv, timeout: 600)
        XCTAssertNotEqual(shortRun.exitCode, 0, "The CLI accepted 3 cameras for 4 views")
        conclude(flow, "dryRun exit=\(run.exitCode) keys=\(object?.keys.sorted() ?? []) shortCameras exit=\(shortRun.exitCode) \(StudioFailureSummary.lastMeaningfulLine(in: shortRun.stderr) ?? "")")
    }

    // MARK: - Audio ▸ Who Spoke

    func test10WhoSpokeFindsTwoSpeakersInTheSynthesizedConversation() throws {
        try requireModels(["speech-diarization-sortformer", "speech-tts-qwen3-nano"])
        let flow = "10-who-spoke"
        let conversation = try twoSpeakerClip(flow: flow)

        // StudioVoiceView's diarization draft: the template's default with the input attached.
        let template = try XCTUnwrap(CommandCatalog.template(id: .speechDiarize))
        var draft = template.defaultDraft()
        draft.inputPath = conversation.path
        draft.outputPath = StudioOutputLocation.specialistFile(domain: .audio, name: "speakers", fileExtension: "json", configuredRoot: live.path).path
        let (request, argv) = try specialistRequest(templateID: .speechDiarize, mode: .listen, draft: draft)
        XCTAssertEqual(Array(argv.prefix(2)), ["speech", "diarize"])
        XCTAssertEqual(argv.firstIndex(of: "--format").map { argv[$0 + 1] }, "json", "The template default asks for JSON explicitly")
        XCTAssertEqual(argv.firstIndex(of: "--output").map { argv[$0 + 1] } ?? argv.firstIndex(of: "-o").map { argv[$0 + 1] }, request.draft.outputPath)

        let run = try runCLI(flow, argv, timeout: 900)
        XCTAssertEqual(run.exitCode, 0, run.failureDescription)
        let outputURL = URL(fileURLWithPath: request.draft.outputPath)
        let document = try XCTUnwrap(StudioDiarizationDocument.load(from: outputURL), "The speaker timeline decoder could not read \(outputURL.path)")
        XCTAssertGreaterThanOrEqual(document.speakerCount, 2, "Two voices were synthesized; summary: \(document.summary)")
        XCTAssertGreaterThanOrEqual(document.segments.count, 2)
        let speakers = document.speakers
        XCTAssertGreaterThanOrEqual(speakers.count, 2)
        XCTAssertEqual(speakers.map(\.id), speakers.map(\.id).sorted(), "Lanes are ordered by speaker index")
        XCTAssertEqual(speakers.map(\.name).prefix(2), ["Speaker 1", "Speaker 2"])
        XCTAssertGreaterThan(document.durationSeconds, 4)
        for segment in document.segments {
            XCTAssertGreaterThanOrEqual(segment.startSeconds, 0)
            XCTAssertLessThanOrEqual(segment.endSeconds, document.durationSeconds + 0.5)
        }
        // The Analyze canvas decodes the same file through the shared document switch.
        let analyze = try decodeAnalyzeDocument(at: outputURL.path)
        guard case .diarization = analyze else { return XCTFail("The shared decoder read the timeline as \(analyze)") }
        XCTAssertEqual(analyze.speechSegments.count, document.segments.count)
        conclude(flow, "speakers=\(document.speakerCount) turns=\(document.segments.count) duration=\(String(format: "%.1f", document.durationSeconds))s lanes=\(speakers.map { "\($0.name):\($0.talkTimeDescription)/\($0.turnCount)" })")
    }

    // MARK: - Music ▸ Analyze

    func test11MusicAnalyzeDecodesIntoTheAnalysisDocument() throws {
        try requireModels(["music-acestep"])
        let flow = "11-music-analyze"
        let music = try Self.toneMusic(in: fixtures())
        let template = try XCTUnwrap(CommandCatalog.template(id: .musicAnalyze))

        var draft = template.defaultDraft()
        draft.model = draft.model.isBlank ? "music-acestep" : draft.model
        draft.inputPath = music.path
        draft.useDuration = true
        draft.durationSeconds = 10
        let (_, argv) = try specialistRequest(templateID: .musicAnalyze, mode: .music, draft: draft)
        XCTAssertEqual(Array(argv.prefix(2)), ["music", "analyze"])
        XCTAssertEqual(argv.firstIndex(of: "--duration").map { argv[$0 + 1] }, "10")

        var run = try runCLI(flow, argv, timeout: 2_400)
        var modelUsed = draft.model
        if run.exitCode != 0, try Self.installedModels.get().contains("music-acestep-xl-turbo-lm4b") {
            // The page's default checkpoint may lack a language model; the LM-bearing checkpoint is the fallback.
            var retry = draft
            retry.model = "music-acestep-xl-turbo-lm4b"
            modelUsed = retry.model
            let (_, retryArgv) = try specialistRequest(templateID: .musicAnalyze, mode: .music, draft: retry)
            run = try runCLI(flow, retryArgv, timeout: 2_400)
        }
        XCTAssertEqual(run.exitCode, 0, run.failureDescription)
        let analysis = try XCTUnwrap(StudioMusicAnalysisDocument.decode(run.libraryOutputText), "stdout did not decode as MusicAnalyzeOutput: \(run.stdout.prefix(600))")
        XCTAssertEqual(URL(fileURLWithPath: analysis.audio).standardizedFileURL.path, music.standardizedFileURL.path)
        XCTAssertFalse(analysis.model.isEmpty)
        XCTAssertEqual(analysis.inputDurationSeconds, 12, accuracy: 1.0)
        XCTAssertLessThanOrEqual(analysis.analyzedDurationSeconds, analysis.inputDurationSeconds + 0.5)
        XCTAssertEqual(analysis.analyzedDescription, "0:10 of 0:12")
        conclude(flow, "model=\(modelUsed) analyzed=\(analysis.analyzedDescription) bpm=\(analysis.tempoDescription ?? "-") key=\(analysis.metadata.keyscale ?? "-") meter=\(analysis.metadata.timesignature ?? "-") language=\(analysis.metadata.language ?? "-") caption=\(analysis.caption?.prefix(80) ?? "-")")
    }

    // MARK: - Music ▸ Transcribe instruments

    func test12TranscribeInstrumentListFeedsThePicker() throws {
        let flow = "12-instruments"
        let run = try runCLI(flow, StudioInstrumentList.listArguments, timeout: 300)
        XCTAssertEqual(run.exitCode, 0, run.failureDescription)
        let names = StudioInstrumentList.parse(run.stdout)
        XCTAssertGreaterThanOrEqual(names.count, 5, "Expected the instrument groups on stdout; got stdout=\(run.stdout.prefix(300)) stderr=\(run.stderr.prefix(300))")
        for name in names {
            XCTAssertTrue(name.allSatisfy { $0.isLetter || $0.isNumber || $0 == "_" }, "Unexpected instrument name token: \(name)")
        }
        XCTAssertEqual(StudioInstrumentList.decode(StudioInstrumentList.encode(Array(names.prefix(3)))), Array(names.prefix(3)))
        XCTAssertEqual(StudioInstrumentList.parse(run.stdout + "\n" + run.stdout), names, "Repeated names must not duplicate")
        conclude(flow, "instruments=\(names.count) \(names)")
    }

    // MARK: - Music ▸ Train manifest

    /// The page's default draft must parse: `--factor` is LoKr's and stays off a LoRA command line
    /// (a separate `-1` used to read as the next option), so the trainer gets as far as loading
    /// both clips before the missing checkpoints stop it.
    func test13MusicTrainManifestIsAcceptedByTheTrainer() throws {
        let flow = "13-music-train-manifest"
        let clipA = try Self.toneMusic(in: fixtures())
        let clipB = try Self.toneMusic(in: fixtures(), name: "tone-b.wav", seconds: 6)
        let manifest = StudioMusicTrainingManifest(clips: [
            StudioMusicTrainingClip(audioPath: clipA.path, caption: "bright arpeggiated synth, 120 bpm, four on the floor", lyrics: "la la la\nla la"),
            StudioMusicTrainingClip(audioPath: clipB.path, caption: "short synth loop, steady kick"),
        ])
        XCTAssertEqual(manifest.problems(), [])
        XCTAssertEqual(manifest.readyClipCount(), 2)

        let template = try XCTUnwrap(CommandCatalog.template(id: .musicTrainAdapter))
        var draft = template.defaultDraft()
        draft.outputPath = live.appendingPathComponent("\(flow)/music-adapter.safetensors").path
        draft.seed = "42"
        draft.steps = 1
        // Point the trainer at no ACE-Step so it fails after it has parsed the manifest.
        draft.model = live.appendingPathComponent("\(flow)/no-acestep-root").path
        draft.musicCheckpointsRoot = live.appendingPathComponent("\(flow)/no-checkpoints").path
        let manifestURL = StudioMusicTrainingManifest.manifestURL(besideOutput: draft.outputPath)
        try FileManager.default.createDirectory(at: manifestURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try manifest.jsonl().write(to: manifestURL, options: .atomic)
        XCTAssertTrue(manifestURL.lastPathComponent.hasPrefix("music-adapter.dataset") && manifestURL.pathExtension == "jsonl", manifestURL.lastPathComponent)
        draft.inputPath = manifestURL.path
        let imported = try StudioMusicTrainingManifest.importing(Data(contentsOf: manifestURL), from: manifestURL)
        XCTAssertEqual(imported.clips.map { $0.audioURL.path }, manifest.clips.map { $0.audioURL.path }, "Import round-trips the (standardized) clip paths")
        XCTAssertEqual(imported.clips.first?.lyrics, "la la la\nla la")

        let (_, argv) = try specialistRequest(templateID: .musicTrainAdapter, mode: .music, draft: draft)
        XCTAssertEqual(argv.firstIndex(of: "--dataset").map { argv[$0 + 1] }, manifestURL.path)
        XCTAssertFalse(argv.contains { $0.hasPrefix("--factor") }, "LoRA training has no factor: \(argv)")
        XCTAssertNil(argv.firstIndex { $0.hasPrefix("-") && Int($0) != nil }, "A negative value must be joined to its flag: \(argv)")
        let run = try runCLI(flow, argv, timeout: 300)
        let text = run.stdout + "\n" + run.stderr
        XCTAssertFalse(text.contains("Missing value for"), "The command line did not parse: \(text.prefix(300))")
        XCTAssertTrue(text.contains("Loading ACE-Step for lora training with 2 example(s)"), "The trainer should have loaded both clips before failing on the missing checkpoints; got: \(text.suffix(600))")
        XCTAssertFalse(text.lowercased().contains("dataset record"), "The manifest itself must not be rejected: \(text.suffix(600))")
        XCTAssertNotEqual(run.exitCode, 0, "Without ACE-Step the run should stop; it did not (timedOut=\(run.timedOut))")

        // Studio blocks a clip without a caption; the CLI rejects the same manifest the same way.
        var broken = manifest
        broken.clips[1].caption = "   "
        XCTAssertEqual(broken.problems(), ["Clip 2 needs a caption."])
        let brokenURL = live.appendingPathComponent("\(flow)/broken.dataset.jsonl")
        try broken.jsonl().write(to: brokenURL, options: .atomic)
        var brokenDraft = draft
        brokenDraft.inputPath = brokenURL.path
        let (_, brokenArgv) = try specialistRequest(templateID: .musicTrainAdapter, mode: .music, draft: brokenDraft)
        let brokenRun = try runCLI(flow, brokenArgv, timeout: 120)
        XCTAssertNotEqual(brokenRun.exitCode, 0)
        XCTAssertTrue((brokenRun.stderr + brokenRun.stdout).lowercased().contains("empty caption"), "Expected the CLI's empty-caption error: \(brokenRun.stderr.suffix(300))")
        conclude(flow, "valid: exit=\(run.exitCode) '\(StudioFailureSummary.lastMeaningfulLine(in: run.stderr) ?? "")' broken: exit=\(brokenRun.exitCode) '\(StudioFailureSummary.lastMeaningfulLine(in: brokenRun.stderr) ?? "")'")
    }

    // MARK: - Image ▸ Datasets ▸ Run plan

    func test14RunPlanPreflightAndMaterializeDecodeIntoTheReport() throws {
        let flow = "14-run-plan"
        let plan = try runPlanFile(flow: flow)

        // Image ▸ Datasets ▸ Run plan: the task draft's fresh state is Preflight with JSON on.
        var preflight = StudioTaskDraft(templateID: .imageRunPlan)
        preflight.setArgument(0, plan.path)
        let (_, argv) = try taskRequest(preflight, task: .imageDatasets)
        XCTAssertEqual(argv, ["image", "run-plan", plan.path, "--preflight", "--json"])
        let run = try runCLI(flow, argv, timeout: 600)
        XCTAssertEqual(run.exitCode, 0, run.failureDescription)
        guard case .runPlan = StudioAnalyzeDocument.decode(Data(run.libraryOutputText.utf8)) else {
            return XCTFail("The Analyze canvas did not read the preflight as a run plan report: \(run.stdout.prefix(400))")
        }
        let report = try XCTUnwrap(StudioRunPlanReport.decode(outputText: run.libraryOutputText), "The preflight envelope did not decode: \(run.stdout.prefix(800))")
        XCTAssertEqual(report.title, "Training plan")
        XCTAssertEqual(report.command, ["image", "train-lora"])
        XCTAssertTrue(["ok", "warning", "blocked"].contains(report.status), report.status)
        guard case .training(let training) = report.result else { return XCTFail("Expected a training preflight") }
        XCTAssertEqual(training.dataset.usablePairCount, 2)
        XCTAssertEqual(training.dataset.imageCount, 2)
        XCTAssertEqual(training.runPlan.arguments.model, "image-klein-nano")
        XCTAssertEqual(training.runPlan.resolved.trainingSteps, 2)
        let titles = report.sections.map(\.title)
        XCTAssertEqual(titles.filter { ["Training", "Schedule", "Dataset", "Model", "Output"].contains($0) }.count, 5, "Sections: \(titles)")
        XCTAssertEqual(report.sections.first { $0.title == "Dataset" }?.rows.first { $0.label == "Usable pairs" }?.value, "2 of 2 images")
        XCTAssertNotNil(report.sections.first { $0.title == "Training" }?.rows.first { $0.label == "Steps" })

        // Run plan ▸ Materialize: Preflight off, a run directory chosen in the inspector's Output
        // section (the page's default run-plan folder).
        var materialize = preflight
        materialize.form["--preflight"] = .flag(false)
        let runDirectory = StudioOutputLocation.specialistDirectory(domain: .image, name: "run-plan", configuredRoot: live.path).path
        materialize.form["--materialize"] = .text(runDirectory)
        let (_, materializeArgv) = try taskRequest(materialize, task: .imageDatasets)
        XCTAssertEqual(materializeArgv, ["image", "run-plan", plan.path, "--json", "--materialize", runDirectory])
        let materialized = try runCLI(flow, materializeArgv, timeout: 600)
        XCTAssertEqual(materialized.exitCode, 0, materialized.failureDescription)
        let materializedReport = try XCTUnwrap(StudioRunPlanReport.decode(outputText: materialized.libraryOutputText), "The materialize envelope did not decode: \(materialized.stdout.prefix(800))")
        XCTAssertEqual(materializedReport.title, "Materialized run")
        guard case .materialized(let run) = materializedReport.result else { return XCTFail("Expected a materialization") }
        XCTAssertEqual(URL(fileURLWithPath: run.runDirectory).standardizedFileURL.path, URL(fileURLWithPath: runDirectory).standardizedFileURL.path)
        XCTAssertTrue(FileManager.default.fileExists(atPath: run.planPath), "plan.json should exist at \(run.planPath)")
        XCTAssertTrue(FileManager.default.fileExists(atPath: run.runManifestPath), "The run manifest should exist at \(run.runManifestPath)")
        XCTAssertEqual(materializedReport.sections.map(\.title), ["Files", "Before"])
        Self.materializedRunDirectory = URL(fileURLWithPath: run.runDirectory)
        conclude(flow, "preflight status=\(report.status) diagnostics=\(report.diagnostics.map { "\($0.severity):\($0.id)" }) sections=\(titles) materialized=\(run.runDirectory)")
    }

    // MARK: - Runs ▸ inspect

    func test15RunInspectDecodesADirectoryAPlanAndAReport() throws {
        let flow = "15-run-inspect"
        let plan = try runPlanFile(flow: flow)
        let template = try XCTUnwrap(CommandCatalog.template(id: .runInspect))
        func inspect(_ reference: String) throws -> (CLIResult, StudioRunInspection?) {
            // StudioOperationsView runs `run inspect <reference> --json` as a utility command; the
            // Operations catalog row builds the same argv from its draft.
            var draft = template.defaultDraft()
            draft.operationsReference = reference
            draft.json = true
            let argv = template.arguments(from: draft)
            XCTAssertEqual(argv, ["run", "inspect", reference, "--json"])
            let run = try runCLI(flow, argv, timeout: 300)
            return (run, StudioRunInspection.decode(run.stdout))
        }

        // The plan file itself.
        let (planRun, planInspection) = try inspect(plan.path)
        XCTAssertEqual(planRun.exitCode, 0, planRun.failureDescription)
        guard case .local(let planEnvelope)? = planInspection else {
            return XCTFail("run inspect on a plan did not decode as a local envelope: \(planRun.stdout.prefix(600))")
        }
        XCTAssertEqual(planEnvelope.result.plan?.kind, "image.train_lora")
        XCTAssertEqual(planEnvelope.result.plan?.command, ["image", "train-lora"])
        let planPresentation = planInspection!.presentation
        XCTAssertEqual(planPresentation.facts.first { $0.label == "Plan" }?.value, "image.train_lora")

        // A materialized run directory (made by test14 or here).
        let directory = try Self.materializedRunDirectory ?? {
            let target = StudioOutputLocation.specialistDirectory(domain: .image, name: "run-plan", configuredRoot: live.path)
            var materialize = CommandDraft()
            materialize.inputPath = plan.path
            materialize.materializePath = target.path
            materialize.json = true
            let (_, argv) = try specialistRequest(templateID: .imageRunPlan, mode: .createImage, draft: materialize)
            let run = try runCLI(flow, argv, timeout: 600)
            XCTAssertEqual(run.exitCode, 0, run.failureDescription)
            return target
        }()
        let (directoryRun, directoryInspection) = try inspect(directory.path)
        XCTAssertEqual(directoryRun.exitCode, 0, directoryRun.failureDescription)
        guard case .local(let directoryEnvelope)? = directoryInspection else {
            return XCTFail("run inspect on a run directory did not decode: \(directoryRun.stdout.prefix(600))")
        }
        let runDirectory = try XCTUnwrap(directoryEnvelope.result.runDirectory, "Expected a run_directory result, got kind \(directoryEnvelope.result.kind)")
        XCTAssertEqual(runDirectory.manifest?.model, "image-klein-nano")
        XCTAssertEqual(runDirectory.manifest?.totalSteps, 2)
        let presentation = directoryInspection!.presentation
        XCTAssertEqual(presentation.facts.first { $0.label == "Model" }?.value, "image-klein-nano")
        XCTAssertEqual(presentation.facts.first { $0.label == "Progress" }?.value, "0 of 2 steps")
        XCTAssertFalse(presentation.outputs.isEmpty, "The Runs page lists the run's files")
        XCTAssertTrue(presentation.outputs.contains { $0.name == "plan.json" }, "Outputs: \(presentation.outputs.map(\.name))")

        // A structured report saved to disk: the train-lora preflight envelope.
        let reportURL = live.appendingPathComponent("\(flow)/preflight-report.json")
        try Data(Self.runPlanPreflightStdout.utf8).write(to: reportURL, options: .atomic)
        let (reportRun, reportInspection) = try inspect(reportURL.path)
        XCTAssertEqual(reportRun.exitCode, 0, reportRun.failureDescription)
        guard case .local(let reportEnvelope)? = reportInspection else {
            return XCTFail("run inspect on a report did not decode: \(reportRun.stdout.prefix(600))")
        }
        XCTAssertEqual(reportEnvelope.result.report?.command, ["image", "train-lora"])
        XCTAssertEqual(reportInspection!.presentation.facts.first { $0.label == "Mode" }?.value, "preflight")
        conclude(flow, "plan=\(planEnvelope.result.plan?.kind ?? "-") directory state=\(presentation.state) facts=\(presentation.facts.map { "\($0.label)=\($0.value)" }) outputs=\(presentation.outputs.map(\.name)) report=\(reportEnvelope.result.report?.status ?? "-")")
    }

    // MARK: - Sound ▸ Generate (renoise)

    func test16SoundGenerateRenoiseArgumentIsLocaleSafeAndAccepted() throws {
        try requireModels(["sfx-woosh-dflow"])
        let flow = "16-sfx-renoise"
        let template = try XCTUnwrap(CommandCatalog.template(id: .sfxGenerate))

        // The page's slider writes the amount through StudioRenoise, never through a locale formatter.
        let german = Locale(identifier: "de_DE")
        XCTAssertEqual(0.5.formatted(.number.locale(german)), "0,5", "de_DE formats decimals with a comma")
        let amount = StudioRenoise.amount(0.5)
        XCTAssertEqual(amount.argument, "0.5")
        XCTAssertEqual(StudioRenoise.amount(0.35).argument, "0.35")
        XCTAssertEqual(amount.problems(steps: 2), [])
        XCTAssertEqual(StudioRenoise.inferredMode(argument: amount.argument), .amount)
        let schedule = StudioRenoise.schedule("0.25, 0.75")
        XCTAssertEqual(schedule.argument, "0.25,0.75")
        XCTAssertEqual(schedule.problems(steps: 2), [])
        let tooLong = StudioRenoise.schedule("0.1,0.2,0.3")
        XCTAssertEqual(tooLong.problems(steps: 2), ["The renoise schedule has 3 values but the run has 2 steps."])
        XCTAssertEqual(StudioRenoise.amount("0,5").problems(steps: 2), ["Renoise must be a number between 0 and 1, with a point for decimals."])

        func makeDraft(renoise: String, name: String) -> CommandDraft {
            var draft = template.defaultDraft()
            draft.prompt = "a short whoosh"
            draft.secondaryText = ""
            draft.model = "sfx-woosh-dflow"
            draft.durationSeconds = 1
            draft.steps = 2
            draft.cfgScale = 4.5
            draft.seed = "7"
            draft.sfxRenoise = renoise
            draft.outputPath = StudioOutputLocation.specialistFile(domain: .sound, name: name, fileExtension: "wav", configuredRoot: live.path).path
            return draft
        }
        for (renoise, name) in [(amount.argument, "sfx-amount"), (schedule.argument, "sfx-schedule")] {
            let draft = makeDraft(renoise: renoise, name: name)
            let (request, argv) = try specialistRequest(templateID: .sfxGenerate, mode: .sfx, draft: draft)
            XCTAssertEqual(argv.firstIndex(of: "--renoise").map { argv[$0 + 1] }, renoise)
            XCTAssertEqual(argv.firstIndex(of: "--steps").map { argv[$0 + 1] } ?? argv.firstIndex(of: "-s").map { argv[$0 + 1] }, "2")
            XCTAssertTrue(request.draft.outputPath.hasPrefix(live.appendingPathComponent("Sound").path), request.draft.outputPath)
            let run = try runCLI(flow, argv, timeout: 900)
            XCTAssertEqual(run.exitCode, 0, run.failureDescription)
            XCTAssertTrue(FileManager.default.fileExists(atPath: request.draft.outputPath), "No WAV at \(request.draft.outputPath)")
            if let file = try? AVAudioFile(forReading: URL(fileURLWithPath: request.draft.outputPath)) {
                XCTAssertEqual(Double(file.length) / file.fileFormat.sampleRate, 1, accuracy: 0.25)
            }
        }
        // The page blocks a schedule that does not match the step count; the CLI does too.
        let (_, badArgv) = try specialistRequest(templateID: .sfxGenerate, mode: .sfx, draft: makeDraft(renoise: tooLong.argument, name: "sfx-bad"))
        let bad = try runCLI(flow, badArgv, timeout: 300)
        XCTAssertNotEqual(bad.exitCode, 0)
        XCTAssertTrue(bad.stderr.contains("--renoise must contain one value or exactly --steps values"), bad.stderr.suffix(300).description)
        conclude(flow, "amount='\(amount.argument)' schedule='\(schedule.argument)' both generated; bad schedule exit=\(bad.exitCode)")
    }

    // MARK: - Chat thinking

    /// With thinking shown the CLI streams the model's reasoning markup raw — `<think>…</think>`
    /// for most families, `<|channel>thought … <channel|>` for Gemma 4 — and
    /// `ConversationTranscript.splitThinking` keeps it beside the answer, never in it.
    func test17ChatWithThinkingShownSplitsReasoningFromTheAnswer() throws {
        let flow = "17-chat-thinking"
        let installed = try Self.installedModels.get()
        let candidates = ["text-chat-gemma4-turbo", "text-chat-gemma4-12b-4bit", "text-chat-nemotron-35-lightning"].filter(installed.contains)
        guard !candidates.isEmpty else { throw XCTSkip("No thinking-capable chat model installed") }
        var outcome: (model: String, reply: ConversationTranscript.Reply, run: CLIResult)?
        for model in candidates {
            var draft = StudioDraft()
            draft.reset(for: .chat)
            draft.prompt = "What is 17 multiplied by 23? Think it through step by step first, then give the final number in one short sentence."
            draft.model = model
            draft.thinkingMode = .show
            // Room for the whole thought: Gemma 4 works a short multiplication four ways first.
            draft.maxTokens = 3_072
            draft.temperature = 0.2
            // A conversation turn: the canvas streams, so the request carries a conversation id.
            let request = try StudioCommandAdapter.makeRequest(mode: .chat, draft: draft, conversationID: UUID())
            let argv = request.template.arguments(from: request.draft)
            XCTAssertTrue(argv.contains("--stream"), "Conversation turns stream: \(argv)")
            XCTAssertTrue(argv.contains("--thinking"), "Thinking shown passes --thinking: \(argv)")
            XCTAssertFalse(argv.contains("--no-thinking"))
            let run = try runCLI(flow, argv, timeout: 1_800)
            guard run.exitCode == 0 else {
                XCTFail("\(model): \(run.failureDescription)")
                continue
            }
            let reply = ConversationTranscript.splitThinking(run.stdout)
            outcome = (model, reply, run)
            if reply.reasoning != nil { break }
        }
        let result = try XCTUnwrap(outcome, "No candidate model produced a reply")
        XCTAssertFalse(result.reply.answer.isEmpty, "The answer must not be empty")
        XCTAssertTrue(result.reply.answer.contains("391"), "17 × 23 = 391; answer was: \(result.reply.answer.prefix(300))")
        XCTAssertNotNil(result.reply.reasoning, "With --thinking the stream should carry a reasoning block; stdout began: \(result.run.stdout.prefix(300))")
        XCTAssertGreaterThan(result.reply.reasoning?.count ?? 0, 20)
        for marker in ["<think>", "</think>", "<|channel>", "<channel|>"] {
            XCTAssertFalse(result.reply.answer.contains(marker), "\(marker) leaked into the answer: \(result.reply.answer.prefix(200))")
            XCTAssertFalse(result.reply.reasoning?.contains(marker) ?? false, "\(marker) leaked into the reasoning")
        }
        XCTAssertFalse(result.reply.isThinking)
        // The streaming split hides the still-open block the way the live bubble does.
        let opener = ["<think>", "<|channel>thought"].compactMap { result.run.stdout.range(of: $0) }.min { $0.lowerBound < $1.lowerBound }
        let open = try XCTUnwrap(opener, "The stream should open a reasoning block: \(result.run.stdout.prefix(200))")
        let partial = String(result.run.stdout[..<result.run.stdout.index(open.upperBound, offsetBy: min(20, result.run.stdout.distance(from: open.upperBound, to: result.run.stdout.endIndex)))])
        XCTAssertTrue(ConversationTranscript.splitThinking(partial, streaming: true).isThinking)
        conclude(flow, "model=\(result.model) marker=\(result.run.stdout[open]) reasoningChars=\(result.reply.reasoning?.count ?? 0) answer=\"\(result.reply.answer.prefix(120))\" tps=\(ConversationTranscript.decodeTokensPerSecond(in: result.run.stderr.components(separatedBy: .newlines)).map { String(format: "%.1f", $0) } ?? "-")")
    }

    /// The CLI prints its error and then ArgumentParser's usage trailer; the turn's one-line reason
    /// is the error, not "See 'mere.run --help' for more information."
    func test18ChatFailedTurnSummarizesToAPlainReason() throws {
        let flow = "18-chat-failed-turn"
        var draft = StudioDraft()
        draft.reset(for: .chat)
        draft.prompt = "Hello"
        draft.model = "text-chat-does-not-exist-xyz"
        draft.requireInstalled = true
        draft.maxTokens = 16
        let request = try StudioCommandAdapter.makeRequest(mode: .chat, draft: draft, conversationID: UUID())
        let argv = request.template.arguments(from: request.draft)
        let run = try runCLI(flow, argv, timeout: 300)
        XCTAssertNotEqual(run.exitCode, 0, "A nonexistent model must fail")
        // The Library summarizes a failed conversation turn from the job's stderr log and exit code.
        let summary = StudioFailureSummary.summary(outputText: nil, logLines: run.stderr.components(separatedBy: .newlines), exitCode: run.exitCode)
        XCTAssertFalse(summary.hasPrefix("The run exited with code"), "No meaningful stderr line was found; stderr: \(run.stderr.suffix(400))")
        XCTAssertFalse(summary.hasPrefix("Error:") || summary.hasPrefix("error:"), "CLI framing should be stripped: \(summary)")
        XCTAssertTrue(summary.hasPrefix("Model 'text-chat-does-not-exist-xyz' is not installed"), "The reason should be the CLI's model line, not its usage trailer: \(summary)")
        XCTAssertFalse(summary.localizedCaseInsensitiveContains("--help"), summary)
        XCTAssertTrue(summary.first?.isUppercase ?? false, "Summaries start with a capital: \(summary)")
        conclude(flow, "exit=\(run.exitCode) summary=\"\(summary)\"")
    }

    // MARK: - Specialist output routing

    func test19SpecialistDestinationsFileUnderTheConfiguredRootAndArePrepared() throws {
        let flow = "19-specialist-routing"
        let stamp = Date(timeIntervalSince1970: 1_800_000_000)
        let expectedStamp = DateFormatter.mereRunTimestamp.string(from: stamp)
        let directory = StudioOutputLocation.specialistDirectory(domain: .vision, name: "vision", now: stamp, configuredRoot: live.path)
        XCTAssertEqual(directory.path, live.appendingPathComponent("Vision/vision-\(expectedStamp)").path)
        let file = StudioOutputLocation.specialistFile(domain: .sound, name: "sfx", fileExtension: "wav", now: stamp, configuredRoot: live.path)
        XCTAssertEqual(file.path, live.appendingPathComponent("Sound/sfx-\(expectedStamp).wav").path)
        // Without a configured root the same call files under the media folder for the extension.
        let home = URL(fileURLWithPath: "/Users/example", isDirectory: true)
        XCTAssertEqual(StudioOutputLocation.specialistFile(domain: .sound, name: "sfx", fileExtension: "wav", now: stamp, configuredRoot: "", home: home).path, "/Users/example/Music/mere.run/Sound/sfx-\(expectedStamp).wav")
        XCTAssertEqual(StudioOutputLocation.specialistDirectory(domain: .vision, name: "vision", now: stamp, configuredRoot: "", home: home).path, "/Users/example/Documents/mere.run/Vision/vision-\(expectedStamp)")

        // What StudioSpecialistRunner does before launching: the destination's folder is created.
        var directoryDraft = CommandDraft()
        directoryDraft.outputPath = directory.path
        let preparedDirectory = StudioOutputLocation.preparingDestination(of: directoryDraft)
        XCTAssertNil(preparedDirectory.fallbackReason)
        XCTAssertTrue(FileManager.default.fileExists(atPath: directory.deletingLastPathComponent().path), "The domain folder should exist")
        let leafExists = FileManager.default.fileExists(atPath: directory.path)
        var fileDraft = CommandDraft()
        fileDraft.outputPath = file.path
        let preparedFile = StudioOutputLocation.preparingDestination(of: fileDraft)
        XCTAssertNil(preparedFile.fallbackReason)
        XCTAssertTrue(FileManager.default.fileExists(atPath: file.deletingLastPathComponent().path))
        // A directory destination without a trailing slash only gets its parent created; the CLI
        // must create the leaf itself (the depth, geometry, and 3D flows above prove whether it does).
        conclude(flow, "directory=\(directory.path) leafPreCreated=\(leafExists) file=\(file.path)")
    }

    // MARK: - Text ▸ Embeddings and Anonymize

    /// Both text utilities from their task drafts: Embeddings writes its vectors under the Text
    /// folder and the canvas reads them as a cosine matrix; Anonymize keeps the paste as one
    /// text, prints JSON, and its spans decode for the renderer.
    func test25EmbeddingsAndAnonymizeDecodeFromTheTaskDrafts() throws {
        let flow = "25-text-utilities"
        try requireModels(["text-embed-qwen3-0.6b", "text-anonymize-privacy-filter"])

        var embed = StudioTaskDraft(templateID: .textEmbed)
        embed.prompt = "semantic search query\nrelated document"
        embed.form["--max-tokens"] = .integer(256)
        let (_, embedArgv) = try taskRequest(embed, task: .textEmbeddings)
        XCTAssertEqual(Array(embedArgv.prefix(4)), ["text", "embed", "semantic search query", "related document"])
        let vectorsPath = try XCTUnwrap(Self.value(of: "--output", in: embedArgv))
        XCTAssertTrue(vectorsPath.hasPrefix(live.appendingPathComponent("Text").path), "Embeddings should file under Text: \(vectorsPath)")
        let embedded = try runCLI(flow, embedArgv, timeout: 600)
        XCTAssertEqual(embedded.exitCode, 0, embedded.failureDescription)
        guard case .embeddings(let vectors) = try decodeAnalyzeDocument(at: vectorsPath) else {
            return XCTFail("The vectors file did not decode as embeddings")
        }
        XCTAssertEqual(vectors.vectors.count, 2)
        XCTAssertGreaterThan(vectors.dimensions, 0)
        let similarity = vectors.cosineSimilarity(vectors.vectors[0], vectors.vectors[1])
        XCTAssertTrue((-1...1).contains(similarity), "cosine \(similarity)")
        guard case .embeddings = StudioAnalyzeDocument.decode(Data(embedded.libraryOutputText.utf8)) else {
            return XCTFail("The printed response did not decode as embeddings either")
        }

        var anonymize = StudioTaskDraft(templateID: .textAnonymize)
        let paste = "My name is Alice Smith and my email is alice@example.com."
        anonymize.prompt = paste
        let (_, anonymizeArgv) = try taskRequest(anonymize, task: .textAnonymize)
        XCTAssertEqual(Array(anonymizeArgv.prefix(3)), ["text", "anonymize", paste])
        XCTAssertTrue(anonymizeArgv.contains("--json"))
        let protectedPath = try XCTUnwrap(Self.value(of: "--output", in: anonymizeArgv))
        XCTAssertTrue(protectedPath.hasPrefix(live.appendingPathComponent("Text").path), "Anonymize should file under Text: \(protectedPath)")
        let anonymized = try runCLI(flow, anonymizeArgv, timeout: 600)
        XCTAssertEqual(anonymized.exitCode, 0, anonymized.failureDescription)
        guard case .anonymization(let spans) = try decodeAnalyzeDocument(at: protectedPath) else {
            return XCTFail("The protected-text file did not decode as an anonymization")
        }
        XCTAssertEqual(spans.results.count, 1, "the paste is one text")
        XCTAssertGreaterThan(spans.spanCount, 0, "the filter should mark the name or the email")
        XCTAssertNotEqual(spans.protectedText, paste)
        conclude(flow, "embed dims=\(vectors.dimensions) cosine=\(String(format: "%.3f", similarity)) file=\(vectorsPath) anonymize spans=\(spans.results[0].spans.map(\.label)) protected='\(spans.protectedText)'")
    }

    // MARK: - Image ▸ Datasets ▸ Discover

    /// Discover from its task draft over a folder holding one two-pair dataset: the well's
    /// folder becomes `--root`, the envelope decodes into candidates, and the leaf is trainable.
    func test26DatasetDiscoverFindsTheFixtureDataset() throws {
        let flow = "26-dataset-discover"
        let root = fixtures().appendingPathComponent("discover-root", isDirectory: true)
        let leaf = root.appendingPathComponent("squares", isDirectory: true)
        try FileManager.default.createDirectory(at: leaf, withIntermediateDirectories: true)
        for (index, color) in [(0, CGColor(red: 0.9, green: 0.2, blue: 0.2, alpha: 1)), (1, CGColor(red: 0.2, green: 0.4, blue: 0.9, alpha: 1))] {
            let image = leaf.appendingPathComponent("sample-\(index).png")
            if !FileManager.default.fileExists(atPath: image.path) {
                try Self.writeImage(to: image, size: CGSize(width: 256, height: 256), square: CGRect(x: 64, y: 64, width: 128, height: 128), color: color, type: .png, exifOrientation: nil)
            }
            try "a \(index == 0 ? "red" : "blue") square on a white background".write(to: leaf.appendingPathComponent("sample-\(index).txt"), atomically: true, encoding: .utf8)
        }

        var discover = StudioTaskDraft(templateID: .imageDatasetDiscover)
        XCTAssertTrue(discover.attach(dropped: [root], slots: discover.slots), "the folder lands in the --root slot")
        let (_, argv) = try taskRequest(discover, task: .imageDatasets)
        XCTAssertEqual(argv, ["image", "dataset", "discover", "--root", root.path, "--max-depth", "4", "--min-usable-pairs", "1", "--json"])
        let run = try runCLI(flow, argv, timeout: 120)
        XCTAssertEqual(run.exitCode, 0, run.failureDescription)
        guard case .datasetDiscovery(let document) = try XCTUnwrap(StudioAnalyzeDocument.decode(Data(run.libraryOutputText.utf8))) else {
            return XCTFail("The discover envelope did not decode: \(run.stdout.prefix(600))")
        }
        let squares = try XCTUnwrap(document.candidates.first { URL(fileURLWithPath: $0.path).standardizedFileURL.path == leaf.standardizedFileURL.path }, "candidates: \(document.candidates.map(\.path))")
        XCTAssertEqual(squares.usablePairs, 2)
        XCTAssertEqual(squares.images, 2)
        XCTAssertTrue(squares.trainable, "status \(squares.status): \(squares.problems)")
        conclude(flow, "scanned=\(document.scannedDirectories) candidates=\(document.candidates.count) trainable=\(document.trainableCount) headline='\(document.headline)'")
    }

    // MARK: - Studio request builders

    /// A task on the shared task workspace, prepared the way `StudioTaskRunner.request(for:task:)`
    /// prepares it: the destination named by routing, then Command edits, validation, and the
    /// folder made real. Every destination the argv carries must sit under the configured root;
    /// a throwaway defaults suite that lost its root would otherwise send a live run into the
    /// user's own folders, so the check comes before anything runs.
    private func taskRequest(_ draft: StudioTaskDraft, task: StudioTask) throws -> (request: StudioRunRequest, argv: [String]) {
        let base = try XCTUnwrap(StudioOutputLocation.destination(for: draft).request(), "\(task) cannot run from Studio")
        let prepared = try MainActor.assumeIsolated { try StudioTaskRunner.prepare(base, sessions: StudioTaskSessions()) }
        XCTAssertNil(prepared.fallbackReason, "The run fell back to App Outputs: \(prepared.fallbackReason ?? "")")
        let argv = try XCTUnwrap(prepared.request.execution).arguments
        assertDestinations(of: draft, in: argv, stayUnder: live)
        return (prepared.request, argv)
    }

    /// Every destination the argv names — the output routing filled, the sidecars beside it, and
    /// a folder the draft chose (`--materialize`) — is under the live directory, so a run can
    /// never write into the user's own folders.
    private func assertDestinations(of draft: StudioTaskDraft, in argv: [String], stayUnder root: URL, file: StaticString = #filePath, line: UInt = #line) {
        guard let capability = draft.capability else { return XCTFail("\(draft.templateID) has no contract", file: file, line: line) }
        let destinationFlags = StudioTaskSchema.outputFlags(for: capability).union(StudioTaskSchema.chosenOutputFlags)
        for (index, word) in argv.enumerated() where destinationFlags.contains(word) && index + 1 < argv.count {
            XCTAssertTrue(argv[index + 1].hasPrefix(root.path), "\(word) escaped the configured root: \(argv[index + 1])", file: file, line: line)
        }
    }

    /// The user's own media folders. A CLI launch naming a file under one of them, outside the
    /// live directory, is a harness fault (the root did not take) and never runs.
    private var fencedFolders: [URL] {
        let home = FileManager.default.homeDirectoryForCurrentUser
        return ["Music", "Pictures", "Documents", "Movies", "Desktop", "Downloads"].map {
            home.appendingPathComponent($0, isDirectory: true)
        }
    }

    /// The word after `flag` in an argv, or nil when the flag is absent.
    private static func value(of flag: String, in argv: [String]) -> String? {
        guard let index = argv.firstIndex(of: flag), index + 1 < argv.count else { return nil }
        return argv[index + 1]
    }

    /// A composer task's request, prepared the way `StudioPromptTaskController` prepares it: the
    /// destination folder is created (under the configured root) before the CLI launches.
    private func composerRequest(mode: StudioMode, draft: StudioDraft) throws -> (request: StudioRunRequest, argv: [String]) {
        let request = try StudioCommandAdapter.makeRequest(mode: mode, draft: draft)
        let prepared = StudioOutputLocation.preparing(request)
        XCTAssertNil(prepared.fallbackReason, "The run fell back to App Outputs: \(prepared.fallbackReason ?? "")")
        XCTAssertTrue(prepared.request.draft.outputPath.hasPrefix(live.path), "Output escaped the configured root: \(prepared.request.draft.outputPath)")
        return (prepared.request, prepared.request.template.arguments(from: prepared.request.draft))
    }

    /// A specialist page's request, prepared the way `StudioTaskRunner` prepares every run
    /// (Command edits, validation, destination), so the tests build exactly what the app runs.
    private func specialistRequest(templateID: CommandTemplateID, mode: StudioMode, draft: CommandDraft) throws -> (request: StudioRunRequest, argv: [String]) {
        let template = try XCTUnwrap(CommandCatalog.template(id: templateID))
        let base = StudioRunRequest(mode: mode, templateID: templateID, template: template, draft: draft)
        // XCTest runs these on the main thread; the runner and the session store are main-actor.
        let prepared = try MainActor.assumeIsolated { try StudioTaskRunner.prepare(base, sessions: StudioTaskSessions()) }
        XCTAssertNil(prepared.fallbackReason, "The run fell back to App Outputs: \(prepared.fallbackReason ?? "")")
        return (prepared.request, template.arguments(from: prepared.request.draft))
    }

    private func decodeAnalyzeDocument(at path: String) throws -> StudioAnalyzeDocument {
        let url = URL(fileURLWithPath: path)
        XCTAssertTrue(FileManager.default.fileExists(atPath: url.path), "The result document was not written at \(url.path)")
        let data = try Data(contentsOf: url)
        return try XCTUnwrap(StudioAnalyzeDocument.decode(data), "StudioAnalyzeDocument.decode rejected \(url.lastPathComponent): \(String(decoding: data.prefix(600), as: UTF8.self))")
    }

    // MARK: - Shared fixtures made with the CLI

    // XCTest runs these methods serially; the caches only cross test methods in this process.
    nonisolated(unsafe) private static var faceImageURL: URL?
    nonisolated(unsafe) private static var faceImageFailed = false
    nonisolated(unsafe) private static var applePhotoURL: URL?
    nonisolated(unsafe) private static var appleRect: CGRect?
    nonisolated(unsafe) private static var twoSpeakerURL: URL?
    nonisolated(unsafe) private static var runPlanURL: URL?
    nonisolated(unsafe) private static var runPlanPreflightStdout = ""
    nonisolated(unsafe) private static var materializedRunDirectory: URL?

    /// A portrait from Klein nano, so Faces and Find have a real subject. Cached across tests.
    private func faceImage(flow: String) throws -> URL {
        if let url = Self.faceImageURL { return url }
        struct Unavailable: Error {}
        guard !Self.faceImageFailed, (try? Self.installedModels.get())?.contains("image-klein-nano") == true else { throw Unavailable() }
        let url = fixtures().appendingPathComponent("portrait-512.png")
        if !FileManager.default.fileExists(atPath: url.path) {
            var draft = StudioDraft()
            draft.reset(for: .createImage)
            draft.prompt = "close-up portrait photograph of a smiling woman facing the camera, soft studio light, sharp focus"
            draft.model = "image-klein-nano"
            draft.width = 512
            draft.height = 512
            draft.steps = 4
            draft.seed = "7"
            let (request, _) = try composerRequest(mode: .createImage, draft: draft)
            var command = request.draft
            command.outputPath = url.path
            let run = try runCLI(flow, request.template.arguments(from: command), timeout: 900)
            guard run.exitCode == 0, FileManager.default.fileExists(atPath: url.path) else {
                Self.faceImageFailed = true
                throw Unavailable()
            }
        }
        Self.faceImageURL = url
        return url
    }

    /// A 640×480 photo of a red apple on a table from Klein nano, the real subject Segment's box and
    /// point prompts are drawn on (SAM has no texture to work with on a flat synthetic square).
    /// Cached across tests.
    private func applePhoto(flow: String) throws -> URL {
        if let url = Self.applePhotoURL { return url }
        let url = fixtures().appendingPathComponent("apple-640x480.png")
        if !FileManager.default.fileExists(atPath: url.path) {
            var draft = StudioDraft()
            draft.reset(for: .createImage)
            draft.prompt = "a red apple on a wooden table, photo"
            draft.model = "image-klein-nano"
            draft.width = 640
            draft.height = 480
            draft.steps = 4
            draft.seed = "3"
            let (request, _) = try composerRequest(mode: .createImage, draft: draft)
            var command = request.draft
            command.outputPath = url.path
            let run = try runCLI(flow, request.template.arguments(from: command), timeout: 900)
            XCTAssertEqual(run.exitCode, 0, run.failureDescription)
        }
        Self.applePhotoURL = url
        return url
    }

    /// Where the apple is, in stored pixels: the box of the mask the text prompt "apple" gives on the
    /// photo, run the way Segment runs a typed prompt. The detector path is the one the live pass
    /// found accurate, so the drawn box is placed from it and the drawn prompts' mask is scored
    /// against it. Cached across tests.
    private func appleRect(flow: String) throws -> CGRect {
        if let rect = Self.appleRect { return rect }
        var draft = StudioDraft()
        draft.reset(for: .segment)
        draft.prompt = "apple"
        draft.inputPath = try applePhoto(flow: flow).path
        let (request, argv) = try composerRequest(mode: .segment, draft: draft)
        XCTAssertTrue(argv.contains("--prompt"))
        let run = try runCLI(flow, argv, timeout: 900)
        XCTAssertEqual(run.exitCode, 0, run.failureDescription)
        let document = try decodeAnalyzeDocument(at: request.draft.visionJSONOutputPath)
        let detections = document.detections(imageSize: CGSize(width: 640, height: 480))
        let best = try XCTUnwrap(detections.max { ($0.confidence ?? 0) < ($1.confidence ?? 0) }, "The text prompt found no apple in the generated photo")
        guard best.box.width >= 80, best.box.height >= 80 else {
            throw XCTSkip("The text prompt's apple is too small to prompt on: \(best.boxDescription)")
        }
        Self.appleRect = best.box
        return best.box
    }

    /// Two voices, three turns (A, B, A), synthesized with the Speak composer and joined with gaps.
    private func twoSpeakerClip(flow: String) throws -> URL {
        if let url = Self.twoSpeakerURL { return url }
        let target = fixtures().appendingPathComponent("two-speakers.wav")
        if !FileManager.default.fileExists(atPath: target.path) {
            let turns: [(voice: String, text: String, name: String)] = [
                ("A calm female voice with clear pronunciation", "Good morning everyone, and thank you for joining the quarterly review. Today we will walk through the roadmap and the numbers behind it.", "turn-a1"),
                ("A deep, slow male voice with a low pitch", "Thanks. Before we start, I want to flag that the shipping dates moved by two weeks because of the supplier delay we discussed.", "turn-b1"),
                ("A calm female voice with clear pronunciation", "That is right, and the plan already accounts for it. Let us begin with the first milestone and the budget attached to it.", "turn-a2"),
            ]
            var clips: [URL] = []
            for turn in turns {
                let clip = fixtures().appendingPathComponent("\(turn.name).wav")
                if !FileManager.default.fileExists(atPath: clip.path) {
                    var draft = StudioDraft()
                    draft.reset(for: .speak)
                    draft.prompt = turn.text
                    draft.secondaryText = turn.voice
                    draft.model = "speech-tts-qwen3-nano"
                    let (request, _) = try composerRequest(mode: .speak, draft: draft)
                    var command = request.draft
                    command.outputPath = clip.path
                    let argv = request.template.arguments(from: command)
                    XCTAssertEqual(argv.firstIndex(of: "--voice").map { argv[$0 + 1] } ?? argv.firstIndex(of: "-v").map { argv[$0 + 1] }, turn.voice)
                    let run = try runCLI(flow, argv, timeout: 900)
                    XCTAssertEqual(run.exitCode, 0, run.failureDescription)
                }
                clips.append(clip)
            }
            try Self.concatenate(clips, gapSeconds: 0.6, to: target)
        }
        Self.twoSpeakerURL = target
        return target
    }

    /// A saved `image.train_lora` plan, extracted from `image train-lora --preflight --json` on a
    /// two-image dataset (the CLI writes plans only through run-plan materialization, so the
    /// preflight's embedded `run_plan` is the seed).
    private func runPlanFile(flow: String) throws -> URL {
        if let url = Self.runPlanURL { return url }
        let dataset = fixtures().appendingPathComponent("lora-dataset", isDirectory: true)
        try FileManager.default.createDirectory(at: dataset, withIntermediateDirectories: true)
        for (index, color) in [(0, CGColor(red: 0.9, green: 0.2, blue: 0.2, alpha: 1)), (1, CGColor(red: 0.2, green: 0.4, blue: 0.9, alpha: 1))] {
            let image = dataset.appendingPathComponent("sample-\(index).png")
            if !FileManager.default.fileExists(atPath: image.path) {
                try Self.writeImage(to: image, size: CGSize(width: 256, height: 256), square: CGRect(x: 64, y: 64, width: 128, height: 128), color: color, type: .png, exifOrientation: nil)
            }
            try "a \(index == 0 ? "red" : "blue") square on a white background".write(to: dataset.appendingPathComponent("sample-\(index).txt"), atomically: true, encoding: .utf8)
        }
        let planFolder = live.appendingPathComponent("run-plan-fixture", isDirectory: true)
        try FileManager.default.createDirectory(at: planFolder, withIntermediateDirectories: true)
        let preflight = try runCLI(flow, [
            "image", "train-lora", "--data", dataset.path, "--output", planFolder.appendingPathComponent("adapter.safetensors").path,
            "--model", "image-klein-nano", "--training-steps", "2", "--width", "256", "--height", "256", "--preflight", "--json",
        ], timeout: 600)
        XCTAssertEqual(preflight.exitCode, 0, preflight.failureDescription)
        Self.runPlanPreflightStdout = preflight.stdout
        let envelope = try XCTUnwrap(StudioStructuredOutput.objectData(in: preflight.stdout).flatMap { try? JSONSerialization.jsonObject(with: $0) } as? [String: Any])
        var runPlan = try XCTUnwrap((envelope["result"] as? [String: Any])?["run_plan"] as? [String: Any], "The train-lora preflight result carries run_plan")
        // The plan file needs its header; the preflight's copy carries it already when the CLI writes it.
        var patched: [String] = []
        if runPlan["schema_version"] == nil { runPlan["schema_version"] = 1; patched.append("schema_version") }
        if runPlan["kind"] == nil { runPlan["kind"] = "image.train_lora"; patched.append("kind") }
        if runPlan["command"] == nil { runPlan["command"] = ["image", "train-lora"]; patched.append("command") }
        if runPlan["created_at"] == nil { runPlan["created_at"] = ISO8601DateFormatter().string(from: Date()); patched.append("created_at") }
        XCTAssertEqual(patched, [], "The preflight's run_plan is missing plan-file fields the CLI's own decoder requires: \(patched)")
        let url = planFolder.appendingPathComponent("plan.json")
        try JSONSerialization.data(withJSONObject: runPlan, options: [.prettyPrinted, .sortedKeys]).write(to: url, options: .atomic)
        Self.runPlanURL = url
        return url
    }

    // MARK: - Synthetic fixtures

    static let squareRect = CGRect(x: 200, y: 120, width: 160, height: 160)
    static let videoFrameCount = 60

    /// Where the square sits on frame `frame` of the synthetic clip: it slides left to right.
    static func videoSquare(atFrame frame: Int) -> CGRect {
        let progress = Double(frame) / Double(videoFrameCount - 1)
        return CGRect(x: 40 + 400 * progress, y: 100, width: 100, height: 100)
    }

    private func fixtures() -> URL {
        let url = live.appendingPathComponent("fixtures", isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    /// A red square on white at `square` (top-left pixel coordinates), as PNG or, with an EXIF
    /// orientation, as JPEG carrying that tag.
    static func squareImage(in folder: URL, name: String, size: CGSize = CGSize(width: 640, height: 480), square: CGRect = squareRect, exifOrientation: Int? = nil) throws -> URL {
        let url = folder.appendingPathComponent(name)
        if FileManager.default.fileExists(atPath: url.path) { return url }
        try writeImage(to: url, size: size, square: square, color: CGColor(red: 0.85, green: 0.1, blue: 0.1, alpha: 1), type: exifOrientation == nil ? .png : .jpeg, exifOrientation: exifOrientation)
        return url
    }

    static func writeImage(to url: URL, size: CGSize, square: CGRect, color: CGColor, type: UTType, exifOrientation: Int?) throws {
        let width = Int(size.width)
        let height = Int(size.height)
        guard let context = CGContext(
            data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { throw XCTSkip("Could not make a drawing context") }
        // Draw in top-left coordinates, the way the CLI and Studio count pixels.
        context.translateBy(x: 0, y: size.height)
        context.scaleBy(x: 1, y: -1)
        context.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
        context.fill(CGRect(origin: .zero, size: size))
        context.setFillColor(color)
        context.fill(square)
        guard let image = context.makeImage(),
              let destination = CGImageDestinationCreateWithURL(url as CFURL, type.identifier as CFString, 1, nil) else {
            throw XCTSkip("Could not write \(url.lastPathComponent)")
        }
        var properties: [CFString: Any] = [:]
        if let exifOrientation { properties[kCGImagePropertyOrientation] = exifOrientation }
        if type == .jpeg { properties[kCGImageDestinationLossyCompressionQuality] = 0.95 }
        CGImageDestinationAddImage(destination, image, properties as CFDictionary)
        guard CGImageDestinationFinalize(destination) else { throw XCTSkip("Could not finalize \(url.lastPathComponent)") }
    }

    /// `photo`'s stored pixels written again as a JPEG carrying EXIF orientation `exifOrientation`,
    /// so the picture displays turned while the CLI decodes the same stored pixels.
    static func rotatedJPEG(of photo: URL, in folder: URL, name: String, exifOrientation: Int) throws -> URL {
        let url = folder.appendingPathComponent(name)
        if FileManager.default.fileExists(atPath: url.path) { return url }
        guard let source = CGImageSourceCreateWithURL(photo as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil),
              let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.jpeg.identifier as CFString, 1, nil) else {
            throw XCTSkip("Could not rewrite \(photo.lastPathComponent)")
        }
        let properties: [CFString: Any] = [kCGImagePropertyOrientation: exifOrientation, kCGImageDestinationLossyCompressionQuality: 0.95]
        CGImageDestinationAddImage(destination, image, properties as CFDictionary)
        guard CGImageDestinationFinalize(destination) else { throw XCTSkip("Could not finalize \(name)") }
        return url
    }

    static func pixels(_ value: Double) -> String {
        String(Int(value.rounded()))
    }

    /// "[195, 85, 445, 380]", the way `StudioAnalyzeDetection.boxDescription` prints a box.
    static func description(of rect: CGRect) -> String {
        "[\(pixels(rect.minX)), \(pixels(rect.minY)), \(pixels(rect.maxX)), \(pixels(rect.maxY))]"
    }

    /// A 640×360, 30 fps, 60-frame H.264 clip of a square sliding across a white frame.
    static func movingSquareVideo(in folder: URL) throws -> URL {
        let url = folder.appendingPathComponent("moving-square.mp4")
        if FileManager.default.fileExists(atPath: url.path) { return url }
        let width = 640
        let height = 360
        let writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
            AVVideoCompressionPropertiesKey: [AVVideoAverageBitRateKey: 6_000_000, AVVideoMaxKeyFrameIntervalKey: 15],
        ])
        input.expectsMediaDataInRealTime = false
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
        ])
        writer.add(input)
        guard writer.startWriting() else { throw writer.error ?? CocoaError(.fileWriteUnknown) }
        writer.startSession(atSourceTime: .zero)
        for frame in 0..<videoFrameCount {
            while !input.isReadyForMoreMediaData { Thread.sleep(forTimeInterval: 0.005) }
            guard let pool = adaptor.pixelBufferPool else { throw CocoaError(.fileWriteUnknown) }
            var buffer: CVPixelBuffer?
            CVPixelBufferPoolCreatePixelBuffer(nil, pool, &buffer)
            guard let pixelBuffer = buffer else { throw CocoaError(.fileWriteUnknown) }
            CVPixelBufferLockBaseAddress(pixelBuffer, [])
            if let context = CGContext(
                data: CVPixelBufferGetBaseAddress(pixelBuffer), width: width, height: height, bitsPerComponent: 8,
                bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer), space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
            ) {
                context.translateBy(x: 0, y: CGFloat(height))
                context.scaleBy(x: 1, y: -1)
                context.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
                context.fill(CGRect(x: 0, y: 0, width: width, height: height))
                context.setFillColor(CGColor(red: 0.1, green: 0.3, blue: 0.85, alpha: 1))
                context.fill(videoSquare(atFrame: frame))
            }
            CVPixelBufferUnlockBaseAddress(pixelBuffer, [])
            guard adaptor.append(pixelBuffer, withPresentationTime: CMTime(value: CMTimeValue(frame), timescale: 30)) else {
                throw writer.error ?? CocoaError(.fileWriteUnknown)
            }
        }
        input.markAsFinished()
        let done = DispatchSemaphore(value: 0)
        writer.finishWriting { done.signal() }
        done.wait()
        guard writer.status == .completed else { throw writer.error ?? CocoaError(.fileWriteUnknown) }
        return url
    }

    /// Twelve seconds of synthetic music: a kick on every beat at 120 BPM under an arpeggio.
    static func toneMusic(in folder: URL, name: String = "tone-music.wav", seconds: Double = 12) throws -> URL {
        let url = folder.appendingPathComponent(name)
        if FileManager.default.fileExists(atPath: url.path) { return url }
        let sampleRate = 48_000.0
        guard let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 2) else { throw CocoaError(.fileWriteUnknown) }
        let frames = AVAudioFrameCount(seconds * sampleRate)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames), let channels = buffer.floatChannelData else {
            throw CocoaError(.fileWriteUnknown)
        }
        buffer.frameLength = frames
        let beat = 0.5
        let chord = [261.63, 329.63, 392.0, 523.25]
        for index in 0..<Int(frames) {
            let time = Double(index) / sampleRate
            let beatPhase = time.truncatingRemainder(dividingBy: beat)
            let kick = sin(2 * .pi * 55 * beatPhase) * exp(-beatPhase * 18) * 0.8
            let notePhase = time.truncatingRemainder(dividingBy: beat / 2)
            let note = chord[Int(time / (beat / 2)) % chord.count]
            let tone = sin(2 * .pi * note * time) * exp(-notePhase * 4) * 0.25
            let pad = (sin(2 * .pi * 130.81 * time) + sin(2 * .pi * 196.0 * time)) * 0.06
            let sample = Float(kick + tone + pad)
            channels[0][index] = sample
            channels[1][index] = sample * 0.9
        }
        let file = try AVAudioFile(forWriting: url, settings: [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: sampleRate,
            AVNumberOfChannelsKey: 2,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsFloatKey: false,
            AVLinearPCMIsBigEndianKey: false,
            AVLinearPCMIsNonInterleaved: false,
        ], commonFormat: .pcmFormatFloat32, interleaved: false)
        try file.write(from: buffer)
        return url
    }

    /// Joins clips of one format into a mono WAV with `gapSeconds` of silence between them.
    static func concatenate(_ urls: [URL], gapSeconds: Double, to target: URL) throws {
        let files = try urls.map { try AVAudioFile(forReading: $0) }
        guard let first = files.first else { return }
        let sampleRate = first.processingFormat.sampleRate
        guard let mono = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1) else { throw CocoaError(.fileWriteUnknown) }
        let gap = AVAudioFrameCount(gapSeconds * sampleRate)
        let total = files.reduce(AVAudioFrameCount(0)) { $0 + AVAudioFrameCount($1.length) } + gap * AVAudioFrameCount(files.count + 1)
        guard let output = AVAudioPCMBuffer(pcmFormat: mono, frameCapacity: total), let destination = output.floatChannelData?[0] else {
            throw CocoaError(.fileWriteUnknown)
        }
        var cursor = Int(gap)
        for file in files {
            guard file.processingFormat.sampleRate == sampleRate else {
                throw XCTSkip("TTS clips differ in sample rate (\(file.processingFormat.sampleRate) vs \(sampleRate)); cannot join without resampling")
            }
            guard let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length)) else {
                throw CocoaError(.fileReadUnknown)
            }
            try file.read(into: buffer)
            guard let source = buffer.floatChannelData?[0] else { throw CocoaError(.fileReadUnknown) }
            for index in 0..<Int(buffer.frameLength) {
                destination[cursor + index] = source[index]
            }
            cursor += Int(buffer.frameLength) + Int(gap)
        }
        output.frameLength = AVAudioFrameCount(cursor)
        try? FileManager.default.removeItem(at: target)
        let file = try AVAudioFile(forWriting: target, settings: [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: sampleRate,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsFloatKey: false,
            AVLinearPCMIsBigEndianKey: false,
            AVLinearPCMIsNonInterleaved: false,
        ], commonFormat: .pcmFormatFloat32, interleaved: false)
        try file.write(from: output)
    }

    static func iou(_ lhs: CGRect, _ rhs: CGRect) -> Double {
        let intersection = lhs.intersection(rhs)
        guard !intersection.isNull, intersection.width > 0, intersection.height > 0 else { return 0 }
        let overlap = intersection.width * intersection.height
        let union = lhs.width * lhs.height + rhs.width * rhs.height - overlap
        return union > 0 ? overlap / union : 0
    }

    // MARK: - The CLI

    struct CLIResult {
        let argv: [String]
        let exitCode: Int32
        let stdout: String
        let stderr: String
        let duration: TimeInterval
        let timedOut: Bool

        /// The text a Library row keeps: stdout, then stderr behind the `STDERR` line.
        var libraryOutputText: String {
            let out = stdout.trimmingCharacters(in: .whitespacesAndNewlines)
            let err = stderr.trimmingCharacters(in: .whitespacesAndNewlines)
            return err.isEmpty ? out : "\(out)\n\nSTDERR\n\(err)"
        }

        var failureDescription: String {
            "exit \(exitCode)\(timedOut ? " (timed out)" : "") after \(String(format: "%.0f", duration))s: \(argv.joined(separator: " "))\n--- stderr tail ---\n\(stderr.suffix(1_200))\n--- stdout tail ---\n\(stdout.suffix(600))"
        }
    }

    private final class DataBox: @unchecked Sendable {
        var data = Data()
    }

    /// Runs the CLI, capturing both streams to files under `live/<flow>/` as evidence.
    @discardableResult
    private func runCLI(_ flow: String, _ argv: [String], timeout: TimeInterval) throws -> CLIResult {
        struct EscapedRoot: LocalizedError {
            let path: String
            var errorDescription: String? { "The CLI would write outside the live directory: \(path)" }
        }
        let livePath = live.standardizedFileURL.path
        for token in argv where token.hasPrefix("/") {
            let path = URL(fileURLWithPath: token).standardizedFileURL.path
            guard !path.hasPrefix(livePath + "/"), fencedFolders.contains(where: { path.hasPrefix($0.standardizedFileURL.path + "/") }) else {
                continue
            }
            XCTFail("Refusing to launch the CLI with \(token): it is under the user's own folders, not \(live.path)")
            throw EscapedRoot(path: token)
        }
        stepCounter += 1
        let folder = live.appendingPathComponent(flow, isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let stem = String(format: "%02d-%@", stepCounter, argv.prefix(3).filter { !$0.hasPrefix("-") && !$0.hasPrefix("/") }.joined(separator: "-"))

        let process = Process()
        process.executableURL = cli
        process.arguments = argv
        process.standardInput = FileHandle.nullDevice
        let stdout = Pipe()
        let stderr = Pipe()
        process.standardOutput = stdout
        process.standardError = stderr
        let outBox = DataBox()
        let errBox = DataBox()
        let readers = DispatchGroup()
        readers.enter()
        DispatchQueue.global(qos: .userInitiated).async {
            outBox.data = stdout.fileHandleForReading.readDataToEndOfFile()
            readers.leave()
        }
        readers.enter()
        DispatchQueue.global(qos: .userInitiated).async {
            errBox.data = stderr.fileHandleForReading.readDataToEndOfFile()
            readers.leave()
        }
        let finished = DispatchSemaphore(value: 0)
        process.terminationHandler = { _ in finished.signal() }
        let start = Date()
        try "\(argv.map(Self.shellQuoted).joined(separator: " "))\n".write(to: folder.appendingPathComponent("\(stem).argv.txt"), atomically: true, encoding: .utf8)
        try process.run()
        var timedOut = false
        if finished.wait(timeout: .now() + timeout) == .timedOut {
            timedOut = true
            process.terminate()
            if finished.wait(timeout: .now() + 30) == .timedOut {
                kill(process.processIdentifier, SIGKILL)
                _ = finished.wait(timeout: .now() + 10)
            }
        }
        readers.wait()
        let duration = Date().timeIntervalSince(start)
        let result = CLIResult(
            argv: argv,
            exitCode: process.terminationStatus,
            stdout: String(decoding: outBox.data, as: UTF8.self),
            stderr: String(decoding: errBox.data, as: UTF8.self),
            duration: duration,
            timedOut: timedOut
        )
        try? result.stdout.write(to: folder.appendingPathComponent("\(stem).stdout.txt"), atomically: true, encoding: .utf8)
        try? result.stderr.write(to: folder.appendingPathComponent("\(stem).stderr.txt"), atomically: true, encoding: .utf8)
        try? "exit=\(result.exitCode) timedOut=\(timedOut) seconds=\(String(format: "%.1f", duration))\n".write(to: folder.appendingPathComponent("\(stem).exit.txt"), atomically: true, encoding: .utf8)
        print("LIVE[\(flow)] exit=\(result.exitCode)\(timedOut ? " TIMEOUT" : "") \(String(format: "%.1f", duration))s :: \(argv.joined(separator: " "))")
        return result
    }

    /// One line per flow in `live/summary.log`, for the report.
    private func conclude(_ flow: String, _ evidence: String) {
        let line = "\(flow): \(evidence)\n"
        let url = live.appendingPathComponent("summary.log")
        if let handle = try? FileHandle(forWritingTo: url) {
            handle.seekToEndOfFile()
            handle.write(Data(line.utf8))
            try? handle.close()
        } else {
            try? line.write(to: url, atomically: true, encoding: .utf8)
        }
        print("LIVE-SUMMARY \(line)", terminator: "")
    }

    private func requireModels(_ ids: [String]) throws {
        let installed = try Self.installedModels.get()
        let missing = ids.filter { !installed.contains($0) }
        if !missing.isEmpty { throw XCTSkip("Not installed: \(missing.joined(separator: ", "))") }
    }

    /// The installed managed models per `mere.run model list --json`, read once. A failure to
    /// read the inventory is kept as the error, so a test that needs a model skips with the reason
    /// rather than silently as "not installed".
    private static let installedModels: Result<Set<String>, Error> = Result {
        struct InventoryUnavailable: LocalizedError {
            let reason: String
            var errorDescription: String? { "mere.run model list --json failed: \(reason)" }
        }
        struct Inventory: Decodable {
            struct Rows: Decodable { let rows: [Row] }
            struct Row: Decodable {
                let id: String
                let status: String
            }
            let inventory: Rows
        }
        let cli = cliURL()
        guard FileManager.default.isExecutableFile(atPath: cli.path) else {
            throw InventoryUnavailable(reason: "no executable at \(cli.path)")
        }
        let process = Process()
        process.executableURL = cli
        process.arguments = ["model", "list", "--json"]
        let stdout = Pipe()
        let stderr = Pipe()
        process.standardOutput = stdout
        process.standardError = stderr
        try process.run()
        let data = stdout.fileHandleForReading.readDataToEndOfFile()
        let diagnostics = String(decoding: stderr.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            throw InventoryUnavailable(reason: "exit \(process.terminationStatus): \(diagnostics.suffix(300))")
        }
        let inventory = try JSONDecoder().decode(Inventory.self, from: data)
        return Set(inventory.inventory.rows.filter { $0.status == "installed" }.map(\.id))
    }

    private static func liveDirectory() -> URL? {
        guard let path = ProcessInfo.processInfo.environment["MERERUN_LIVE_ACCEPTANCE_DIR"],
              !path.trimmingCharacters(in: .whitespaces).isEmpty else { return nil }
        return URL(fileURLWithPath: NSString(string: path).expandingTildeInPath, isDirectory: true)
    }

    /// `MERERUN_LIVE_CLI`, or the `mere.run` built beside this test bundle: the bundle sits in the
    /// build products directory (`swift build --show-bin-path`), whatever the triple and
    /// configuration.
    private static func cliURL() -> URL {
        if let path = ProcessInfo.processInfo.environment["MERERUN_LIVE_CLI"], !path.isEmpty {
            return URL(fileURLWithPath: NSString(string: path).expandingTildeInPath)
        }
        return Bundle(for: StudioLiveAcceptanceTests.self).bundleURL
            .deletingLastPathComponent()
            .appendingPathComponent("mere.run")
    }

    private static func shellQuoted(_ word: String) -> String {
        word.rangeOfCharacter(from: CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "-_./=:,")).inverted) == nil
            ? word
            : "'" + word.replacingOccurrences(of: "'", with: "'\\''") + "'"
    }
}
