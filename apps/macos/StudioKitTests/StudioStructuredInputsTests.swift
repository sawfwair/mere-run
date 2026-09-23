import ImageIO
@testable import StudioKit
import UniformTypeIdentifiers
import XCTest

/// The manifests, documents, and argument syntaxes Studio builds for the CLI, checked against the
/// decoders the CLI uses for them.
final class StudioStructuredInputsTests: XCTestCase {
    private var root: URL!

    override func setUpWithError() throws {
        try super.setUpWithError()
        root = FileManager.default.temporaryDirectory.appendingPathComponent("StudioStructuredInputs-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
        try super.tearDownWithError()
    }

    // MARK: - Music training manifest

    /// `ACEStepAdapterTrainingPlan.ManifestRecord` in MereRunCore: `audio` and `caption` required,
    /// `lyrics` optional, decoded one line at a time when the file is not a JSON array.
    private struct CLIManifestRecord: Codable, Equatable {
        var audio: String
        var caption: String
        var lyrics: String?
    }

    private func cliLoadManifest(_ data: Data) throws -> [CLIManifestRecord] {
        let decoder = JSONDecoder()
        if let records = try? decoder.decode([CLIManifestRecord].self, from: data) { return records }
        return try String(decoding: data, as: UTF8.self).split(whereSeparator: \.isNewline).map {
            try decoder.decode(CLIManifestRecord.self, from: Data($0.utf8))
        }
    }

    func testTheManifestWritesOneRecordPerLineThatTheCLIDecoderReads() throws {
        let manifest = StudioMusicTrainingManifest(clips: [
            .init(audioPath: "/Music/clips/intro.wav", caption: "warm analog synth pad, slow", lyrics: "  la la la \n"),
            .init(audioPath: "~/Music/clips/drums.wav", caption: " tight funk drums "),
        ])

        let records = try cliLoadManifest(try manifest.jsonl())

        XCTAssertEqual(records, [
            .init(audio: "/Music/clips/intro.wav", caption: "warm analog synth pad, slow", lyrics: "la la la"),
            .init(audio: NSHomeDirectory() + "/Music/clips/drums.wav", caption: "tight funk drums", lyrics: nil),
        ])
        XCTAssertEqual(String(decoding: try manifest.jsonl(), as: UTF8.self).split(separator: "\n").count, 2)
    }

    func testImportingReadsJSONLAndJSONArraysAndResolvesRelativePathsAgainstTheManifest() throws {
        let manifestURL = root.appendingPathComponent("dataset.jsonl")
        let jsonl = Data("""
        {"audio": "clips/a.wav", "caption": "a", "lyrics": "verse"}
        {"audio": "/abs/b.wav", "caption": "b"}
        """.utf8)
        let array = Data(#"[{"audio": "c.wav", "caption": "c"}]"#.utf8)

        let fromLines = try StudioMusicTrainingManifest.importing(jsonl, from: manifestURL)
        let fromArray = try StudioMusicTrainingManifest.importing(array, from: manifestURL)

        XCTAssertEqual(fromLines.clips.map(\.audioPath), [root.appendingPathComponent("clips/a.wav").path, "/abs/b.wav"])
        XCTAssertEqual(fromLines.clips.map(\.lyrics), ["verse", ""])
        XCTAssertEqual(fromArray.clips.map(\.audioPath), [root.appendingPathComponent("c.wav").path])
        XCTAssertThrowsError(try StudioMusicTrainingManifest.importing(Data("{\"audio\": \"x\"}\n".utf8), from: manifestURL)) {
            XCTAssertEqual($0 as? StudioMusicManifestImportError, .invalidLine(1))
        }
        XCTAssertThrowsError(try StudioMusicTrainingManifest.importing(Data("[]".utf8), from: manifestURL)) {
            XCTAssertEqual($0 as? StudioMusicManifestImportError, .empty)
        }
    }

    func testManifestProblemsAreTheTrainersChecksInThePagesWords() throws {
        let present = root.appendingPathComponent("present.wav")
        try Data([0]).write(to: present)
        let manifest = StudioMusicTrainingManifest(clips: [
            .init(audioPath: present.path, caption: "fine"),
            .init(audioPath: "", caption: "no file"),
            .init(audioPath: root.appendingPathComponent("gone.wav").path, caption: " "),
        ])

        XCTAssertEqual(StudioMusicTrainingManifest().problems(), ["Add at least one audio clip."])
        XCTAssertEqual(manifest.problems(), [
            "Clip 2 has no audio file.",
            "Clip 3 is missing its audio file, gone.wav.",
            "Clip 3 needs a caption.",
        ])
        XCTAssertEqual(manifest.readyClipCount(), 1)
    }

    func testScanningAFolderPairsAudioWithSiblingCaptions() throws {
        try Data([0]).write(to: root.appendingPathComponent("b.mp3"))
        try Data([0]).write(to: root.appendingPathComponent("a.wav"))
        try Data("bright piano\n".utf8).write(to: root.appendingPathComponent("a.txt"))
        try Data([0]).write(to: root.appendingPathComponent("notes.txt"))

        let clips = StudioMusicTrainingManifest.clips(scanning: root)

        XCTAssertEqual(clips.map(\.fileName), ["a.wav", "b.mp3"])
        XCTAssertEqual(clips.map(\.caption), ["bright piano", ""])
        XCTAssertEqual(
            StudioMusicTrainingManifest.manifestURL(besideOutput: "/Runs/music-adapter-1.safetensors").path,
            "/Runs/music-adapter-1.dataset.jsonl"
        )
    }

    // MARK: - Run plan report

    func testATrainingPreflightDecodesToReadableSections() throws {
        let report = try XCTUnwrap(StudioRunPlanReport.decode(outputText: Self.trainingPreflight + "\n\nSTDERR\nLoading…\n"))

        XCTAssertEqual(report.status, "warning")
        XCTAssertEqual(report.title, "Training plan")
        XCTAssertEqual(report.diagnostics.map(\.severity), [.warning])
        guard case .training(let preflight) = report.result else { return XCTFail("expected a training result") }
        XCTAssertEqual(preflight.runPlan.resolved.trainingSteps, 1_200)
        XCTAssertEqual(preflight.dataset.usablePairCount, 48)

        let sections = Dictionary(uniqueKeysWithValues: report.sections.map { ($0.title, $0.rows) })
        XCTAssertEqual(report.sections.map(\.title), ["Training", "Schedule", "Memory and targets", "Dataset", "Model", "Output"])
        XCTAssertEqual(sections["Training"]?.map(\.label), ["Recipe", "Steps", "Resolution", "Batch size", "Rank", "Learning rate", "Caption dropout", "Seed", "Checkpoints", "Previews"])
        XCTAssertEqual(sections["Training"]?.map(\.value), [
            "krea-fast-style", "1,200", "1024 × 1024", "1", "16 · alpha 16", "0.0001", "10%", "42",
            "every 250 steps · 4 expected", "every 250 steps",
        ])
        XCTAssertEqual(sections["Memory and targets"]?.map(\.value), ["1536", "Progressive resolution, Compiled train step"])
        XCTAssertEqual(sections["Dataset"]?.map(\.value), ["/data/mugs", "48 of 50 images", "2 missing captions"])
        XCTAssertEqual(sections["Model"]?.map(\.value), ["image-krea2-raw", "krea2", "/Models/krea2"])
        XCTAssertEqual(sections["Output"]?.first?.path, "/Runs/mugs.safetensors")
    }

    func testAGenerationPreflightAndAMaterializedRunDecodeByModeAndCommand() throws {
        let generation = try XCTUnwrap(StudioRunPlanReport.decode(Data(Self.generationPreflight.utf8)))
        let materialized = try XCTUnwrap(StudioRunPlanReport.decode(Data(Self.materialized.utf8)))

        XCTAssertEqual(generation.title, "Generation plan")
        XCTAssertEqual(generation.sections.map(\.title), ["Generation", "Inputs", "Model", "Output"])
        XCTAssertEqual(generation.sections[0].rows.map(\.value), [
            "a ceramic mug", "1024 × 768", "28", "3.5", "3", "512", "7", "text to image",
        ])
        XCTAssertEqual(generation.sections[1].rows.map(\.value), ["/refs/mug.png", "style.safetensors · scale 0.8 · missing"])
        XCTAssertEqual(generation.sections[2].rows.map(\.value), ["image-zimage-turbo", "No · about 6.2 GB to download"])

        XCTAssertEqual(materialized.title, "Materialized run")
        XCTAssertEqual(materialized.sections[0].rows.map(\.label), ["Run folder", "Plan", "Actions", "Run manifest", "Events", "Output"])
        XCTAssertEqual(materialized.sections[0].rows.map(\.path), materialized.sections[0].rows.map(\.value))
    }

    // MARK: - Instruments

    func testTheInstrumentListIsOneNamePerLine() {
        XCTAssertEqual(StudioInstrumentList.parse("acoustic_piano\n\n drums \nvoice\ndrums\n"), ["acoustic_piano", "drums", "voice"])
        XCTAssertEqual(StudioInstrumentList.displayName("soprano_and_alto_sax"), "Soprano and alto sax")
        XCTAssertEqual(StudioInstrumentList.encode(["voice", "drums"]), "voice,drums")
        XCTAssertEqual(StudioInstrumentList.decode(" voice, drums ,"), ["voice", "drums"])
        XCTAssertEqual(StudioInstrumentList.listArguments, ["music", "transcribe", "--list-instruments"])
    }

    // MARK: - Target ranks

    func testTargetRanksRoundTripTheKleinSuffixMap() {
        let ranks = StudioTargetRank.decode(".attn.to_q=128, .ff.linear_in=64,broken")

        XCTAssertEqual(ranks.map(\.suffix), [".attn.to_q", ".ff.linear_in", "broken"])
        XCTAssertEqual(ranks.map(\.rank), [128, 64, 0])
        XCTAssertEqual(StudioTargetRank.encode(Array(ranks.prefix(2))), ".attn.to_q=128,.ff.linear_in=64")
        XCTAssertEqual(StudioTargetRank.problems(ranks + [.init(suffix: " ", rank: 8)]), [
            "Target 3 needs a rank of at least 1.",
            "Target 4 needs a module suffix, like .attn.to_q.",
        ])
        XCTAssertEqual(StudioTargetRank.encode([]), "")
        // A row still waiting for its suffix is not sent as "=64", which the CLI rejects.
        XCTAssertEqual(StudioTargetRank.encode([.init(suffix: "", rank: 64), .init(suffix: ".attn.to_k", rank: 32)]), ".attn.to_k=32")
    }

    // MARK: - Renoise

    func testRenoiseReadsAndWritesTheCLIsArgument() {
        XCTAssertEqual(StudioRenoise.inferredMode(argument: ""), .automatic)
        XCTAssertEqual(StudioRenoise.inferredMode(argument: "0.35"), .amount)
        XCTAssertEqual(StudioRenoise.inferredMode(argument: "0.1, 0.2,0.3"), .schedule)
        XCTAssertEqual(StudioRenoise(mode: .amount, argument: "0.35"), .amount(0.35))
        XCTAssertEqual(StudioRenoise(mode: .amount, argument: "junk"), .amount(0.5))
        XCTAssertEqual(StudioRenoise.amount(0.35).argument, "0.35")
        XCTAssertEqual(StudioRenoise.schedule(" 0.1, 0.25,1 ").argument, "0.1,0.25,1")
        XCTAssertEqual(StudioRenoise.amount(1.5).problems(steps: 3), ["Renoise must be between 0 and 1."])
        XCTAssertEqual(StudioRenoise.schedule("0.1, 0.2").problems(steps: 3), ["The renoise schedule has 2 values but the run has 3 steps."])
        XCTAssertEqual(StudioRenoise.schedule("0.1, 0.2, 0.3").problems(steps: 3), [])
        XCTAssertEqual(StudioRenoise.schedule("0.3").problems(steps: 3), ["The renoise schedule has 1 values but the run has 3 steps."])
        XCTAssertEqual(StudioRenoise.schedule("").problems(steps: 3), ["Enter one renoise amount per step, separated by commas."])
    }

    /// The argument is what `parseRenoiseSchedule` splits on commas and reads with `Float(_:)`, so a
    /// locale that writes decimals with a comma must not reach it; and a token the CLI would reject
    /// is reported rather than dropped.
    func testRenoiseWritesAPointDecimalWhateverTheLocaleAndKeepsBadTokens() {
        let german = 0.35.formatted(.number.locale(Locale(identifier: "de_DE")))
        XCTAssertEqual(german, "0,35", "the hazard this test guards against")

        XCTAssertEqual(StudioRenoise.amount(0.35).argument, "0.35")
        XCTAssertEqual(Float(StudioRenoise.amount(0.35).argument), 0.35)
        // A comma decimal typed into a schedule is two tokens to the CLI, so the step count catches it.
        XCTAssertEqual(StudioRenoise.schedule("0.5,\(german)").scheduleValues, [0.5, 0, 35])
        XCTAssertEqual(
            StudioRenoise.schedule("0.5,\(german)").problems(steps: 2),
            ["The renoise schedule has 3 values but the run has 2 steps.", "Renoise values must be between 0 and 1."]
        )
        XCTAssertNil(StudioRenoise.schedule("0.5,half").scheduleValues)
        XCTAssertEqual(
            StudioRenoise.schedule("0.5,half").problems(steps: 2),
            ["Renoise amounts must be numbers separated by commas, with a point for decimals."]
        )
    }

    /// Line separators `JSONEncoder` does not escape would split a record when the trainer splits
    /// the file on newlines; they are written as `\n`.
    func testLyricsWithUnicodeLineSeparatorsStayOneRecord() throws {
        let manifest = StudioMusicTrainingManifest(clips: [
            .init(audioPath: "/clips/a.wav", caption: "verse\u{2028}chorus", lyrics: "one\u{2029}two\u{85}three\r\nfour\rfive"),
        ])

        let records = try cliLoadManifest(try manifest.jsonl())

        XCTAssertEqual(records.count, 1)
        XCTAssertEqual(records[0].caption, "verse\nchorus")
        XCTAssertEqual(records[0].lyrics, "one\ntwo\nthree\nfour\nfive")
    }

    func testDraftNamesCarryTheirContentAndRunFilesNeverOverwrite() throws {
        let a = try StudioMusicTrainingManifest(clips: [.init(audioPath: "/a.wav", caption: "a")]).jsonl()
        let b = try StudioMusicTrainingManifest(clips: [.init(audioPath: "/a.wav", caption: "b")]).jsonl()

        XCTAssertEqual(StudioMusicTrainingManifest.draftManifestURL(content: a), StudioMusicTrainingManifest.draftManifestURL(content: a))
        XCTAssertNotEqual(StudioMusicTrainingManifest.draftManifestURL(content: a), StudioMusicTrainingManifest.draftManifestURL(content: b))
        XCTAssertTrue(StudioMusicTrainingManifest.isDraftURL(StudioMusicTrainingManifest.draftManifestURL(content: a)))
        XCTAssertFalse(StudioMusicTrainingManifest.isDraftURL(root.appendingPathComponent("dataset.jsonl")))

        try Data([0]).write(to: root.appendingPathComponent("adapter.dataset.jsonl"))
        XCTAssertEqual(
            StudioMusicTrainingManifest.manifestURL(besideOutput: root.appendingPathComponent("adapter.safetensors").path).lastPathComponent,
            "adapter.dataset-2.jsonl"
        )
        try Data([0]).write(to: root.appendingPathComponent("run.cameras.json"))
        XCTAssertEqual(
            StudioCameraDocuments.url(besideOutputDirectory: root.appendingPathComponent("run").path).lastPathComponent,
            "run.cameras-2.json"
        )
    }

    func testAnUnknownDiagnosticSeverityDoesNotDropTheReport() throws {
        let json = Self.materialized.replacingOccurrences(of: "\"severity\": \"note\"", with: "\"severity\": \"hint\"")

        let report = try XCTUnwrap(StudioRunPlanReport.decode(Data(json.utf8)))

        XCTAssertEqual(report.diagnostics.map(\.severity), [.unknown])
    }

    // MARK: - Cameras

    /// `DepthAnything3CameraDocument` as `vision geometry-multiview --cameras` decodes it.
    private struct CLIGeometryCameraDocument: Decodable {
        struct Intrinsics: Decodable {
            let imageWidth: Int
            let imageHeight: Int
            let normalizedFX: Double
            let normalizedFY: Double
            let normalizedCX: Double
            let normalizedCY: Double
        }

        struct Extrinsics: Decodable {
            let rotation: [Double]
            let translation: [Double]
        }

        struct Camera: Decodable {
            let intrinsics: Intrinsics
            let extrinsics: Extrinsics
        }

        let schemaVersion: Int
        let cameras: [Camera]
    }

    func testGeometryCamerasEncodeToTheDocumentTheCLIDecodes() throws {
        let camera = StudioGeometryCamera(imageWidth: 1_920, imageHeight: 1_080, normalizedFX: 1.2, normalizedFY: 2.1, translation: [0, 0, 4])
        let document = StudioGeometryCameraDocument(cameras: [camera, .identity()])

        let decoded = try JSONDecoder().decode(CLIGeometryCameraDocument.self, from: try document.json())
        let reimported = try StudioGeometryCameraDocument.importing(try document.json())

        XCTAssertEqual(decoded.schemaVersion, 1)
        XCTAssertEqual(decoded.cameras.count, 2)
        XCTAssertEqual(decoded.cameras[0].intrinsics.normalizedFY, 2.1)
        XCTAssertEqual(decoded.cameras[0].extrinsics.rotation, [1, 0, 0, 0, 1, 0, 0, 0, 1])
        XCTAssertEqual(decoded.cameras[0].extrinsics.translation, [0, 0, 4])
        XCTAssertEqual(reimported.cameras.map(\.imageWidth), [1_920, 1_920])
        XCTAssertEqual(reimported.cameras[0].translation, [0, 0, 4])
    }

    func testGeometryCameraProblemsAreTheCLIsValidationRules() {
        var mirrored = StudioGeometryCamera.identity()
        mirrored.rotation = [-1, 0, 0, 0, 1, 0, 0, 0, 1]
        var skewed = StudioGeometryCamera.identity()
        skewed.rotation = [1, 0.5, 0, 0, 1, 0, 0, 0, 1]
        var flat = StudioGeometryCamera.identity(width: 0)
        flat.normalizedFX = 0
        let views = [StudioCameraView(name: "a.png"), StudioCameraView(name: "b.png"), StudioCameraView(name: "c.png")]

        XCTAssertEqual(StudioGeometryCamera.identity().problems(), [])
        XCTAssertEqual(mirrored.problems(), ["has a rotation that mirrors (its determinant is not +1)."])
        XCTAssertEqual(skewed.problems(), ["has a rotation that is not a pure rotation (a row or column is not unit length)."])
        XCTAssertEqual(flat.problems(), ["needs a positive image size.", "needs positive focal lengths."])
        XCTAssertEqual(
            StudioGeometryCameraDocument(cameras: [mirrored]).problems(views: views),
            ["Add one camera per view: 3 views, 1 camera.", "Camera 1 has a rotation that mirrors (its determinant is not +1)."]
        )
    }

    /// `DepthAnything3CameraValidation.validate(_:sourceDimensions:)` rejects a camera whose image
    /// size is not the decoded image's, so the page reads each view's size the way the CLI does
    /// (`kCGImagePropertyPixelWidth` and `Height`) and sizes new cameras to it with square pixels.
    func testGeometryCamerasAreSizedToTheirImagesAndReportAMismatch() throws {
        let image = root.appendingPathComponent("view.png")
        try Self.writePNG(to: image, width: 640, height: 360)

        let size = try XCTUnwrap(StudioPixelSize.of(image))
        let sized = StudioGeometryCamera.identity(size: size)
        let view = StudioCameraView(name: "view.png", pixelSize: size)

        XCTAssertEqual(size, StudioPixelSize(width: 640, height: 360))
        XCTAssertEqual(sized.imageWidth, 640)
        XCTAssertEqual(sized.imageHeight, 360)
        XCTAssertEqual(sized.normalizedFX * Double(sized.imageWidth), sized.normalizedFY * Double(sized.imageHeight), accuracy: 1e-9)
        XCTAssertEqual(sized.problems(view: view), [])
        XCTAssertEqual(StudioGeometryCamera.identity().problems(view: view), ["is sized 1920 × 1080 but view.png is 640 × 360."])
        XCTAssertEqual(StudioGeometryCamera.identity().problems(view: StudioCameraView(name: "unreadable.png")), [])
        XCTAssertNil(StudioPixelSize.of(root.appendingPathComponent("missing.png")))
    }

    private static func writePNG(to url: URL, width: Int, height: Int) throws {
        guard let context = CGContext(
            data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ), let image = context.makeImage(),
              let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
            throw XCTSkip("Could not draw a PNG")
        }
        CGImageDestinationAddImage(destination, image, nil)
        CGImageDestinationFinalize(destination)
    }

    func testInstantMeshCamerasEncodeSixteenValuesPerView() throws {
        let document = StudioInstantMeshCameraDocument(cameras: [.example, .example, .example, .example])

        let object = try JSONSerialization.jsonObject(with: try document.json()) as? [String: Any]
        let reimported = try StudioInstantMeshCameraDocument.importing(try document.json())

        XCTAssertEqual(object?["schemaVersion"] as? Int, 1)
        XCTAssertEqual((object?["cameras"] as? [[Double]])?.count, 4)
        XCTAssertEqual((object?["cameras"] as? [[Double]])?.first, [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 4, 1.866, 1.866, 0.5, 0.5])
        XCTAssertEqual(reimported.cameras.count, 4)
        XCTAssertEqual(Set(document.cameras.map(\.id)).count, 4, "each starting camera is its own row")
        XCTAssertEqual(document.problems(viewCount: 4), [])
        XCTAssertEqual(document.problems(viewCount: 6), ["Add one camera per view: 6 views, 4 cameras."])
        XCTAssertEqual(StudioInstantMeshCamera(values: [1, 2]).problems, ["needs 16 values: a 3 × 4 pose and fx, fy, cx, cy."])
        XCTAssertThrowsError(try StudioInstantMeshCameraDocument.importing(Data(#"{"schemaVersion": 2, "cameras": []}"#.utf8)))
        XCTAssertEqual(
            StudioCameraDocuments.url(besideOutputDirectory: "/Movies/MereRun/3D/20260923-101010").path,
            "/Movies/MereRun/3D/20260923-101010.cameras.json"
        )
    }

    // MARK: - Samples

    /// `image run-plan plan.json --preflight --json` for a training plan: `StructuredRunEnvelope`
    /// with a `LoRATrainingPreflightResult`, keys as the CLI's CodingKeys spell them.
    private static let trainingPreflight = """
    {"schema_version": 1, "mere_run_version": "0.55.0", "command": ["image", "train-lora"], "mode": "preflight",
     "status": "warning", "created_at": "2026-09-23T10:00:00Z", "cwd": "/Runs",
     "summary": "Ready to train with 48 usable pairs; 2 images are missing captions.",
     "request": {"data": "/data/mugs", "output": "/Runs/mugs.safetensors", "model": "image-krea2-raw", "recipe": "krea-fast-style",
       "training_steps": 1200, "width": 1024, "height": 1024, "rank": 16, "alpha": 16, "learning_rate": 0.0001, "caption_dropout": 0.1},
     "result": {
       "dataset": {"directory": "/data/mugs", "mode": "directory", "image_count": 50, "caption_count": 48, "usable_pair_count": 48,
         "missing_caption_count": 2, "empty_caption_count": 0, "duplicate_caption_group_count": 0, "duplicate_caption_count": 0,
         "excluded_preview_image_count": 0, "placeholder_caption_count": 0},
       "model": {"requested": "image-krea2-raw", "kind": "managed", "installed": true, "path": "/Models/krea2", "family": "krea2",
         "upstream_repo_id": "krea/krea-2-raw"},
       "output": {"path": "/Runs/mugs.safetensors", "parent_directory": "/Runs", "parent_exists": true, "parent_will_be_created": false,
         "exists": false, "extension_valid": true},
       "plan": {"recipe": "krea-fast-style", "training_steps": 1200, "width": 1024, "height": 1024, "rank": 16, "alpha": 16,
         "learning_rate": 0.0001, "caption_dropout": 0.1, "checkpoint_interval": 250, "expected_checkpoint_count": 4,
         "max_resolution": 1536, "low_ram": false, "no_compile": false, "lr_warmup_steps": 100, "use_cosine_scheduler": true, "lr_min_factor": 0.1},
       "run_plan": {"schema_version": 1, "kind": "image.train_lora", "command": ["image", "train-lora"],
         "created_at": "2026-09-23T10:00:00Z", "cwd": "/Runs",
         "arguments": {"data": "/data/mugs", "output": "/Runs/mugs.safetensors", "model": "image-krea2-raw", "source_recipe": "krea-fast-style",
           "width": 1024, "height": 1024, "training_steps": 1200, "batch_size": 1, "learning_rate": 0.0001, "rank": 16, "alpha": 16,
           "max_text_length": 512, "scheduler_steps": 1000, "caption_dropout": 0.1, "seed": 42, "lite": false,
           "exclude_preview_images": false, "checkpoint_interval": 250, "max_resolution": 1536, "progressive": true, "low_ram": false,
           "no_compile": false, "gradient_checkpointing": false, "benchmark_warmup_steps": 0, "sample_interval": 250,
           "sample_steps": 20, "sample_cfg": 3.5, "sample_lora_scale": 1, "visualize": false, "visualize_port": 8765,
           "lr_warmup_steps": 100, "no_cosine_scheduler": false, "lr_min_factor": 0.1, "quiet": false},
         "resolved": {"recipe": "krea-fast-style", "training_steps": 1200, "width": 1024, "height": 1024, "rank": 16, "alpha": 16,
           "learning_rate": 0.0001, "caption_dropout": 0.1, "checkpoint_interval": 250, "expected_checkpoint_count": 4,
           "max_resolution": 1536, "low_ram": false, "no_compile": false, "lr_warmup_steps": 100, "use_cosine_scheduler": true, "lr_min_factor": 0.1}}},
     "diagnostics": [{"id": "missing_captions", "severity": "warning", "title": "Missing captions",
       "message": "2 images have no caption and will be skipped.", "locations": [], "suggested_action_ids": []}],
     "actions": []}
    """

    private static let generationPreflight = """
    {"schema_version": 1, "mere_run_version": "0.55.0", "command": ["image", "generate"], "mode": "preflight",
     "status": "blocked", "created_at": "2026-09-23T10:00:00Z", "cwd": "/Runs", "summary": "The model is not installed.",
     "request": {"prompt": "a ceramic mug", "output": "/Runs/mug.png", "model": "image-zimage-turbo", "width": 1024, "height": 768},
     "result": {
       "model": {"requested": "image-zimage-turbo", "kind": "managed", "installed": false, "estimated_download_bytes": 6200000000},
       "output": {"path": "/Runs/mug.png", "parent_directory": "/Runs", "parent_exists": true, "parent_will_be_created": false,
         "exists": false, "expected_extension": "png", "extension_valid": true},
       "inputs": {"reference_images": [{"requested": "refs/mug.png", "path": "/refs/mug.png", "exists": true, "is_directory": false}],
         "mask_feather": 0, "missing_count": 0},
       "loras": [{"requested": "style.safetensors", "path": "/Runs/style.safetensors", "exists": false, "is_directory": false, "scale": 0.8}],
       "structured_prompt": {"enabled": false, "model": "text-chat-gemma4-12b-4bit", "backend": "mlx", "max_tokens": 512, "fallback_available": true},
       "plan": {"family": "zimage", "width": 1024, "height": 768, "requested_steps": 28, "effective_steps": 28,
         "effective_cfg_scale": 3.5, "effective_sigma_shift": 3, "max_sequence_length": 512, "effective_max_sequence_length": 512,
         "input_mode": "text_to_image"},
       "run_plan": {"schema_version": 1, "kind": "image.generate", "command": ["image", "generate"], "created_at": "2026-09-23T10:00:00Z",
         "cwd": "/Runs", "arguments": {"prompt": "a ceramic mug", "output": "/Runs/mug.png", "model": "image-zimage-turbo",
           "width": 1024, "height": 768, "seed": 7, "reference_images": ["/refs/mug.png"], "keep_original_aspect": false,
           "max_sequence_length": 512, "structured_prompt": false, "structured_prompt_model": "text-chat-gemma4-12b-4bit",
           "structured_prompt_max_tokens": 512, "loras": ["style.safetensors"], "lora_scale": 0.8, "quiet": false},
         "resolved": {"family": "zimage", "width": 1024, "height": 768, "max_sequence_length": 512,
           "effective_max_sequence_length": 512, "input_mode": "text_to_image"}}},
     "diagnostics": [{"id": "model_missing", "severity": "blocker", "title": "Model not installed", "message": "Pull image-zimage-turbo first.",
       "locations": [], "suggested_action_ids": ["pull-model"]}],
     "actions": []}
    """

    private static let materialized = """
    {"schema_version": 1, "mere_run_version": "0.55.0", "command": ["image", "run-plan"], "mode": "materialize", "status": "ok",
     "created_at": "2026-09-23T10:00:00Z", "cwd": "/Runs", "summary": "Materialized image.train_lora run at /Runs/mugs-run.",
     "request": {"plan_file": "/Runs/plan.json", "run_directory": "/Runs/mugs-run"},
     "result": {"run_directory": "/Runs/mugs-run", "plan_path": "/Runs/mugs-run/plan.json", "actions_path": "/Runs/mugs-run/actions.json",
       "run_manifest_path": "/Runs/mugs-run/run.json", "events_path": "/Runs/mugs-run/mugs.events.jsonl",
       "output_path": "/Runs/mugs-run/mugs.safetensors", "original_output_path": "/Runs/mugs.safetensors"},
     "diagnostics": [{"id": "output_relocated", "severity": "note", "title": "Output relocated",
       "message": "The materialized plan writes its output inside the run directory.", "locations": [], "suggested_action_ids": []}],
     "actions": []}
    """
}
