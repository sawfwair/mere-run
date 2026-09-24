@testable import StudioKit
import Foundation
import XCTest

/// Text ▸ Embeddings, Text ▸ Anonymize, and Image ▸ Datasets on the shared task workspace: a task
/// draft builds the argv the Utility Lab page built for the same settings, the typed input lands
/// in the positional the way each command reads it, the discover root is a well slot, and every
/// result the CLI prints decodes into the Analyze document the renderers draw from.
final class StudioTextDatasetsWorkspaceTests: XCTestCase {
    private var root: URL!
    private var suite: UserDefaults!
    private let suiteName = "run.mere.studio.text-datasets-tests"

    override func setUpWithError() throws {
        try super.setUpWithError()
        root = FileManager.default.temporaryDirectory.appendingPathComponent("text-datasets-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        suite = try XCTUnwrap(UserDefaults(suiteName: suiteName))
        suite.removePersistentDomain(forName: suiteName)
        suite.set(root.path, forKey: StudioOutputLocation.rootDefaultsKey)
        StudioOutputLocation.defaults = suite
    }

    override func tearDownWithError() throws {
        StudioOutputLocation.defaults = .standard
        suite.removePersistentDomain(forName: suiteName)
        try? FileManager.default.removeItem(at: root)
        try super.tearDownWithError()
    }

    // MARK: - Argv parity with the Utility Lab page

    /// Embeddings: one line per text, the page's `--pretty`, and the destination routing names
    /// under Text — the same argv `StudioUtilityLabView` built from its `CommandDraft`.
    func testEmbeddingsDraftBuildsThePagesArgv() throws {
        var draft = StudioTaskDraft(templateID: .textEmbed)
        // The template's defaults: its example text, the default model, and the page's --pretty.
        XCTAssertEqual(draft.prompt, "semantic search query")
        XCTAssertEqual(Array(draft.arguments.prefix(5)), ["text", "embed", "semantic search query", "--model", "text-embed-qwen3-0.6b"])
        XCTAssertEqual(draft.arguments.last, "--pretty")
        draft.prompt = "semantic search query\nrelated document"
        draft.form["--max-tokens"] = .integer(2_048)
        XCTAssertEqual(draft.prompt, "semantic search query\nrelated document")
        let named = StudioOutputLocation.destination(for: draft)
        let output = named.text("--output")
        XCTAssertTrue(output.hasPrefix(root.appendingPathComponent("Text").path), output)
        XCTAssertEqual(URL(fileURLWithPath: output).pathExtension, "json")

        let template = try XCTUnwrap(CommandCatalog.template(id: .textEmbed))
        var page = template.defaultDraft()
        page.prompt = "semantic search query\nrelated document"
        page.maxTokens = 2_048
        page.force = true
        page.outputPath = output
        XCTAssertEqual(named.arguments, template.arguments(from: page))
        XCTAssertEqual(named.arguments, [
            "text", "embed", "semantic search query", "related document", "--model", "text-embed-qwen3-0.6b",
            "--max-tokens", "2048", "--output", output, "--pretty",
        ])
    }

    /// Anonymize: the paste stays one text (the page sent it as one argument, so a paragraph
    /// keeps its lines), and `--json --pretty` are on so the spans renderer has a document.
    func testAnonymizeDraftKeepsThePasteAsOneTextAndAsksForJSON() throws {
        var draft = StudioTaskDraft(templateID: .textAnonymize)
        let paste = "My name is Alice Smith.\nMy email is alice@example.com."
        draft.prompt = paste
        XCTAssertEqual(draft.prompt, paste)
        XCTAssertEqual(draft.form.arguments, [paste])
        draft.form["--max-tokens"] = .integer(2_048)
        let named = StudioOutputLocation.destination(for: draft)
        let output = named.text("--output")
        XCTAssertTrue(output.hasPrefix(root.appendingPathComponent("Text").path), output)

        let template = try XCTUnwrap(CommandCatalog.template(id: .textAnonymize))
        var page = template.defaultDraft()
        page.prompt = paste
        page.maxTokens = 2_048
        page.all = true
        page.force = true
        page.outputPath = output
        // The page emitted its options in its builder's order; the form emits the contract's.
        XCTAssertEqual(Set(named.arguments), Set(template.arguments(from: page)))
        XCTAssertEqual(Array(named.arguments.prefix(3)), ["text", "anonymize", paste])
        XCTAssertEqual(named.arguments.count, template.arguments(from: page).count)
    }

    /// Discover's `--root` is required but filed under no group; it is the well slot all the
    /// same, and the draft builds the page's argv.
    func testDiscoverRootIsTheWellSlotAndTheDraftBuildsThePagesArgv() throws {
        let slots = StudioTaskSchema.slots(for: .imageDatasetDiscover)
        XCTAssertEqual(slots.map(\.id), ["--root"])
        XCTAssertEqual(slots.first?.acceptedTypes, [.folder])
        XCTAssertEqual(StudioTask.imageDatasets.presentation.attaching(slots.first).attachLabel, "Choose folder…")

        let folder = root.appendingPathComponent("datasets", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        var draft = StudioTaskDraft(templateID: .imageDatasetDiscover)
        XCTAssertTrue(draft.attach(dropped: [folder], slots: draft.slots))
        XCTAssertEqual(draft.primaryInputPath, folder.path)

        let template = try XCTUnwrap(CommandCatalog.template(id: .imageDatasetDiscover))
        var page = template.defaultDraft()
        page.inputPath = folder.path
        XCTAssertEqual(draft.arguments, template.arguments(from: page))
        XCTAssertEqual(draft.arguments, [
            "image", "dataset", "discover", "--root", folder.path, "--max-depth", "4", "--min-usable-pairs", "1", "--json",
        ])
        XCTAssertEqual(StudioOutputLocation.destination(for: draft).arguments, draft.arguments, "discover prints; nothing to route")
    }

    /// Run plan starts as the page did, on Preflight; turning it off and naming a run directory
    /// is Materialize, and both are one argv apart from the page's.
    func testRunPlanDraftPreflightsByDefaultAndMaterializesWhenAsked() throws {
        let plan = root.appendingPathComponent("plan.json").path
        var draft = StudioTaskDraft(templateID: .imageRunPlan)
        draft.setArgument(0, plan)
        XCTAssertEqual(draft.primaryInputPath, plan)
        XCTAssertEqual(draft.arguments, ["image", "run-plan", plan, "--preflight", "--json"])

        let template = try XCTUnwrap(CommandCatalog.template(id: .imageRunPlan))
        var page = template.defaultDraft()
        page.inputPath = plan
        page.preflight = true
        page.materializePath = ""
        page.json = true
        XCTAssertEqual(draft.arguments, template.arguments(from: page))

        let runDirectory = root.appendingPathComponent("run-plan", isDirectory: true).path
        draft.form["--preflight"] = .flag(false)
        draft.form["--materialize"] = .text(runDirectory)
        page.preflight = false
        page.materializePath = runDirectory
        XCTAssertEqual(Set(draft.arguments), Set(template.arguments(from: page)))
        XCTAssertEqual(draft.arguments, ["image", "run-plan", plan, "--json", "--materialize", runDirectory])
        XCTAssertEqual(StudioOutputLocation.destination(for: draft).arguments, draft.arguments, "the run directory is the user's choice")
    }

    /// Validate takes no input; the suite and family chips carry the page's defaults and the
    /// artifact folder is routed under Image.
    func testValidateDraftBuildsThePagesArgvWithARoutedArtifactFolder() throws {
        XCTAssertEqual(StudioTaskSchema.slots(for: .imageValidate), [])
        let draft = StudioTaskDraft(templateID: .imageValidate)
        let named = StudioOutputLocation.destination(for: draft)
        let output = named.text("--output")
        XCTAssertTrue(output.hasPrefix(root.appendingPathComponent("Image").path), output)

        let template = try XCTUnwrap(CommandCatalog.template(id: .imageValidate))
        var page = template.defaultDraft()
        page.outputPath = output
        XCTAssertEqual(named.arguments, template.arguments(from: page))
        XCTAssertEqual(named.arguments, ["image", "validate", "--test", "all", "--family", "zimage", "--output", output])
    }

    /// Image ▸ Datasets opens on Discover, and its three variants keep the input each takes.
    func testDatasetsVariantsSwitchTheirInput() throws {
        XCTAssertEqual(StudioTask.imageDatasets.variantTemplates.map(\.id), [.imageDatasetDiscover, .imageValidate, .imageRunPlan])
        var draft = try XCTUnwrap(StudioTaskDraft(task: .imageDatasets))
        XCTAssertEqual(draft.templateID, .imageDatasetDiscover)
        draft.switchTemplate(to: .imageRunPlan)
        XCTAssertEqual(StudioTaskSchema.slots(for: draft.templateID).map(\.id), ["file"])
        XCTAssertEqual(StudioTaskSchema.slots(for: draft.templateID).first?.acceptedTypes, [.json])
    }

    /// Library ▸ "Use these settings" on an Embeddings row lands its texts back in the editor.
    func testUseTheseSettingsRestoresTheEmbeddingTexts() throws {
        let template = try XCTUnwrap(CommandCatalog.template(id: .textEmbed))
        var page = template.defaultDraft()
        page.prompt = "first\nsecond"
        page.force = true
        page.outputPath = root.appendingPathComponent("Text/first-second-ab12cd.json").path
        let item = StudioLibraryItem(
            id: UUID(), mode: .chat, prompt: "first\nsecond", inputURL: nil, outputURL: nil,
            createdAt: Date(), updatedAt: Date(), status: .completed, exitCode: 0,
            commandPreview: "mere.run text embed first second --pretty", outputText: nil,
            templateID: .textEmbed, commandDraft: page, commandArguments: template.arguments(from: page)
        )
        XCTAssertTrue(StudioLibraryDraftRestoration.canRestore(item))
        let restored = try XCTUnwrap(StudioLibraryDraftRestoration.taskDraft(from: item))
        XCTAssertEqual(restored.templateID, .textEmbed)
        XCTAssertEqual(restored.prompt, "first\nsecond")
        // The recorded run's destination was that run's, not a setting: routing names a new one.
        XCTAssertFalse(restored.arguments.contains("--output"))
        page.outputPath = ""
        XCTAssertEqual(restored.arguments, template.arguments(from: page))
    }

    // MARK: - Documents

    func testEmbeddingsJSONDecodesIntoTheAnalyzeDocument() throws {
        let data = Data(
            """
            {"object": "list", "model": "text-embed-qwen3-0.6b",
             "data": [{"object": "embedding", "index": 0, "embedding": [1.0, 0.0, 0.0]},
                      {"object": "embedding", "index": 1, "embedding": [0.8, 0.6, 0.0]}],
             "usage": {"prompt_tokens": 7, "total_tokens": 7}}
            """.utf8
        )
        guard case .embeddings(let document) = try XCTUnwrap(StudioAnalyzeDocument.decode(data)) else {
            return XCTFail("Expected an embeddings document")
        }
        XCTAssertEqual(document.model, "text-embed-qwen3-0.6b")
        XCTAssertEqual(document.promptTokens, 7)
        XCTAssertEqual(document.dimensions, 3)
        XCTAssertEqual(document.cosineSimilarity(document.vectors[0], document.vectors[1]), 0.8, accuracy: 0.0001)
        XCTAssertEqual(document.summary, "2 vectors · 3 dimensions")
        XCTAssertEqual(StudioAnalyzeDocument.embeddings(document).modelID, "text-embed-qwen3-0.6b")
        XCTAssertEqual(StudioAnalyzeDocument.embeddings(document).summary(detectionCount: 0), "2 vectors · 3 dimensions")
        XCTAssertNil(StudioEmbeddingDocument.decode(Data("{\"model\": \"x\", \"data\": []}".utf8)), "usage is part of the response")
    }

    /// The row's captured output keeps stderr after the JSON; the document is still found.
    func testAnonymizationOutputDecodesWithStderrBehindIt() throws {
        let output = """
        {"object": "list", "model": "text-anonymize-privacy-filter",
         "data": [{"text": "Call Alice at 555-1234", "anonymized_text": "Call [NAME] at [PHONE]", "token_count": 8,
                   "spans": [{"label": "NAME", "text": "Alice", "startToken": 1, "endToken": 2},
                             {"label": "PHONE", "text": "555-1234", "startToken": 3, "endToken": 6}]}]}

        STDERR
        Loading model from /Users/example/Library/Application Support/MereRun/models/text-anonymize-privacy-filter
        """
        guard case .anonymization(let document) = try XCTUnwrap(StudioAnalyzeDocument.decode(Data(output.utf8))) else {
            return XCTFail("Expected an anonymization document")
        }
        XCTAssertEqual(document.results.map(\.id), [0])
        XCTAssertEqual(document.results[0].spans.map(\.id), [0, 1])
        XCTAssertEqual(document.results[0].spans.map(\.label), ["NAME", "PHONE"])
        XCTAssertEqual(document.spanCount, 2)
        XCTAssertEqual(document.tokenCount, 8)
        XCTAssertEqual(document.protectedText, "Call [NAME] at [PHONE]")
        XCTAssertEqual(document.summary, "2 PII spans in 1 document")
    }

    func testDiscoveryEnvelopeDecodesIntoCandidates() throws {
        let output = """
        {"schema_version": 1, "mere_run_version": "0.55.0", "command": ["image", "dataset", "discover"], "mode": "inspection",
         "status": "warning", "created_at": "2026-09-23T10:00:00Z", "cwd": "/tmp", "summary": "Found 2 dataset candidates under /tmp/datasets.",
         "request": {"root": "/tmp/datasets", "max_depth": 4, "min_usable_pairs": 1, "exclude_preview_images": false},
         "result": {"root": "/tmp/datasets", "scanned_directory_count": 8, "candidate_count": 2, "trainable_candidate_count": 1,
           "candidates": [
             {"id": "portraits", "name": "Portraits", "path": "/tmp/datasets/portraits", "relative_path": "portraits", "depth": 1,
              "status": "ok", "trainable": true, "image_count": 12, "caption_count": 12, "usable_pair_count": 12,
              "missing_caption_count": 0, "empty_caption_count": 0, "duplicate_caption_group_count": 0, "placeholder_caption_count": 0,
              "diagnostics": []},
             {"id": "sketches", "name": "Sketches", "path": "/tmp/datasets/sketches", "relative_path": "sketches", "depth": 1,
              "status": "blocked", "trainable": false, "image_count": 3, "caption_count": 0, "usable_pair_count": 0,
              "missing_caption_count": 3, "empty_caption_count": 0, "duplicate_caption_group_count": 0, "placeholder_caption_count": 0,
              "diagnostics": [{"id": "missing_captions", "severity": "blocker", "title": "Missing captions",
                               "message": "3 images have no caption.", "locations": [], "suggested_action_ids": []}]}]},
         "diagnostics": [{"id": "preview_images", "severity": "note", "title": "Preview images",
                          "message": "2 preview images were counted.", "locations": [], "suggested_action_ids": []}],
         "actions": []}
        """
        guard case .datasetDiscovery(let document) = try XCTUnwrap(StudioAnalyzeDocument.decode(Data(output.utf8))) else {
            return XCTFail("Expected a discovery document")
        }
        XCTAssertEqual(document.scannedDirectories, 8)
        XCTAssertEqual(document.candidates.map(\.id), ["portraits", "sketches"])
        XCTAssertEqual(document.trainableCount, 1)
        XCTAssertEqual(document.candidates[1].problems, ["Missing captions: 3 images have no caption."])
        XCTAssertEqual(document.diagnostics, ["Preview images: 2 preview images were counted."])
        XCTAssertEqual(document.headline, "2 candidates · 1 trainable")
        XCTAssertNil(StudioAnalyzeDocument.datasetDiscovery(document).modelID)
    }

    func testRunPlanEnvelopeDecodesIntoTheReport() throws {
        let output = """
        {"schema_version": 1, "mere_run_version": "0.55.0", "command": ["image", "run-plan"], "mode": "materialize",
         "status": "ok", "created_at": "2026-09-23T10:00:00Z", "cwd": "/tmp", "summary": "Materialized image.train_lora run at /tmp/run.",
         "request": {"plan_file": "/tmp/plan.json", "run_directory": "/tmp/run"},
         "result": {"run_directory": "/tmp/run", "plan_path": "/tmp/run/plan.json", "actions_path": "/tmp/run/actions.json",
           "run_manifest_path": "/tmp/run/run.json", "events_path": "/tmp/run/events.jsonl",
           "output_path": "/tmp/run/adapter.safetensors", "original_output_path": "/tmp/adapter.safetensors"},
         "diagnostics": [], "actions": []}
        """
        guard case .runPlan(let report) = try XCTUnwrap(StudioAnalyzeDocument.decode(Data(output.utf8))) else {
            return XCTFail("Expected a run plan report")
        }
        XCTAssertEqual(report.title, "Materialized run")
        XCTAssertEqual(StudioAnalyzeDocument.runPlan(report).summary(detectionCount: 0), "Materialized run")
    }

    /// `image validate` reports on stderr; the folder it names is the result.
    func testValidationOutputDecodesIntoTheArtifactFolder() throws {
        let folder = root.appendingPathComponent("validation", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        try Data([0x89, 0x50]).write(to: folder.appendingPathComponent("vae-roundtrip.png"))
        try "ok".write(to: folder.appendingPathComponent("report.txt"), atomically: true, encoding: .utf8)
        let output = """
        Image validation
          family: zimage
          suite: all
          output: \(folder.path)
          Using model: /Users/example/Library/Application Support/MereRun/models/image-zimage-turbo
          Source: managed

        Validation complete. Artifacts written to \(folder.path).
        """
        guard case .validation(let report) = try XCTUnwrap(StudioAnalyzeDocument.decode(Data(output.utf8))) else {
            return XCTFail("Expected a validation report")
        }
        XCTAssertEqual(report.family, "zimage")
        XCTAssertEqual(report.suite, "all")
        XCTAssertEqual(report.artifactDirectory.standardizedFileURL.path, folder.standardizedFileURL.path)
        XCTAssertEqual(report.artifacts().map(\.lastPathComponent), ["report.txt", "vae-roundtrip.png"])
        XCTAssertEqual(report.summary, "Z-Image · every suite")

        let unfinished = "Image validation\n  family: klein\n  suite: vae\n  output: /tmp/x\nError: model missing"
        XCTAssertNil(StudioImageValidationReport.decode(outputText: unfinished), "a run that stopped has no folder to show")
        guard case .transcript = try XCTUnwrap(StudioAnalyzeDocument.decode(Data(unfinished.utf8))) else {
            return XCTFail("The CLI's words stand in for the report")
        }
    }
}
