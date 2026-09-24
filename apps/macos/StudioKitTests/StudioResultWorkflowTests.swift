import Foundation
import MereRunContract
import StudioTestSupport
@testable import StudioKit
import XCTest

@MainActor
final class StudioResultWorkflowTests: XCTestCase {
    private func library() -> StudioLibraryStore {
        StudioLibraryStore(libraryURL: FileManager.default.temporaryDirectory.appendingPathComponent("result-workflow-\(UUID()).json"))
    }

    private func image(_ library: StudioLibraryStore, seed: String = "42", output: String = "/tmp/source.png") throws -> StudioLibraryItem {
        let template = try XCTUnwrap(CommandCatalog.template(id: .imageGenerate))
        var draft = template.defaultDraft()
        draft.prompt = "A ceramic bowl"
        draft.seed = seed
        draft.outputPath = output
        return library.start(request: StudioRunRequest(mode: .createImage, templateID: template.id,
            template: template, draft: draft), commandPreview: "fixture")
    }

    func testReplayReservesDistinctDestinationsBeforeFilesExistAndKeepsLineage() throws {
        let item = try image(library())
        let first = try XCTUnwrap(StudioLibraryReplay.request(for: item))
        let second = try XCTUnwrap(StudioLibraryReplay.request(for: item))
        XCTAssertNotEqual(first.draft.outputPath, second.draft.outputPath)
        XCTAssertNotEqual(first.draft.outputPath, item.commandDraft?.outputPath)
        XCTAssertEqual(first.parentID, item.id)
        XCTAssertEqual(first.draft.seed, "42")
    }

    func testReplaySeparatesSidecarsWhenThePrimaryOutputIsImplicit() throws {
        let command = StudioExecution(templateID: .musicGenerate,
            arguments: ["music", "generate", "A quiet piano", "--lrc-output", "/tmp/lyrics.lrc"])
        let first = try XCTUnwrap(command.replay(outputPath: "").form).text("--lrc-output")
        let second = try XCTUnwrap(command.replay(outputPath: "").form).text("--lrc-output")
        XCTAssertNotEqual(first, "/tmp/lyrics.lrc")
        XCTAssertNotEqual(first, second)
        XCTAssertEqual(URL(fileURLWithPath: first).pathExtension, "lrc")
    }

    /// Run again on a transcribe row renames the MIDI and the context document it named beside
    /// it, so the replay never writes over the original's files; the input and options stay.
    func testReplayRenamesEveryDerivedSidecarOfATranscribeRow() throws {
        let template = try XCTUnwrap(CommandCatalog.template(id: .musicTranscribe))
        var draft = template.defaultDraft()
        draft.inputPath = "/tmp/harbor-lights.wav"
        draft.outputPath = "/tmp/out/harbor-lights-a1b2c3.mid"
        draft.musicContextOutput = "/tmp/out/harbor-lights-a1b2c3-context.json"
        draft.musicInstruments = "piano"
        let row = library().start(request: StudioRunRequest(mode: .music, templateID: .musicTranscribe,
            template: template, draft: draft), commandPreview: "fixture")

        let replay = try XCTUnwrap(StudioLibraryReplay.request(for: row))
        let form = try XCTUnwrap(replay.execution?.form)

        XCTAssertNotEqual(replay.draft.outputPath, draft.outputPath)
        XCTAssertEqual(URL(fileURLWithPath: replay.draft.outputPath).pathExtension, "mid")
        let context = form.text("--context-output")
        XCTAssertNotEqual(context, draft.musicContextOutput, "the sidecar is renamed too")
        XCTAssertTrue(context.hasSuffix("-context.json"), context)
        XCTAssertEqual(URL(fileURLWithPath: context).deletingLastPathComponent().path, "/tmp/out")
        XCTAssertEqual(form.arguments.first, "/tmp/harbor-lights.wav")
        XCTAssertEqual(form.text("--instruments"), "piano")

        let batchTemplate = try XCTUnwrap(CommandCatalog.template(id: .visionFaceBatch))
        let batch = StudioExecution(templateID: .visionFaceBatch, arguments: batchTemplate.arguments(from: batchTemplate.defaultDraft())
            + ["/tmp/1.png", "--jsonl-output", "/tmp/out/faces.jsonl"])
        XCTAssertNotEqual(try XCTUnwrap(batch.replay(outputPath: "/tmp/out/faces-2.jsonl").form).text("--jsonl-output"), "/tmp/out/faces.jsonl")
    }

    /// A command whose file follows its `--format` replays into a file of that format: a JSON
    /// transcription stays JSON and an RTTM diarization stays RTTM, never the template's default
    /// `.mid` or `.json`.
    func testReplayKeepsTheExtensionTheRunsFormatWrote() throws {
        let store = library()
        let transcribe = try XCTUnwrap(CommandCatalog.template(id: .musicTranscribe))
        var transcription = transcribe.defaultDraft()
        transcription.inputPath = "/tmp/harbor-lights.wav"
        transcription.musicTranscribeFormat = "json"
        transcription.outputPath = "/tmp/out/harbor-lights-a1b2c3.json"
        let transcribed = store.start(request: StudioRunRequest(mode: .music, templateID: .musicTranscribe,
            template: transcribe, draft: transcription), commandPreview: "fixture")
        let rerun = try XCTUnwrap(StudioLibraryReplay.request(for: transcribed))
        XCTAssertEqual(URL(fileURLWithPath: rerun.draft.outputPath).pathExtension, "json", rerun.draft.outputPath)
        XCTAssertEqual(try XCTUnwrap(rerun.execution?.form).text("--output"), rerun.draft.outputPath)

        let diarize = try XCTUnwrap(CommandCatalog.template(id: .speechDiarize))
        var speakers = diarize.defaultDraft()
        speakers.inputPath = "/tmp/standup.wav"
        speakers.speechDiarizationFormat = "rttm"
        speakers.outputPath = "/tmp/out/standup-d4e5f6.rttm"
        let diarized = store.start(request: StudioRunRequest(mode: .listen, templateID: .speechDiarize,
            template: diarize, draft: speakers), commandPreview: "fixture")
        let again = try XCTUnwrap(StudioLibraryReplay.request(for: diarized))
        XCTAssertEqual(URL(fileURLWithPath: again.draft.outputPath).pathExtension, "rttm", again.draft.outputPath)
    }

    func testComparisonShowsSeedChangesAndOmitsDestinationNoise() throws {
        let store = library()
        let first = try image(store)
        let second = try image(store, seed: "99", output: "/tmp/new.png")
        let changes = StudioResultComparison.differences(first, second)
        XCTAssertEqual(changes.map(\.id), ["--seed"])
        XCTAssertEqual(changes.first?.first, "42")
        XCTAssertEqual(changes.first?.second, "99")
    }

    func testComparisonIncludesPositionalPromptsAndAdditionalOptionsWithoutSecrets() throws {
        var first = try image(library())
        first.templateID = .musicGenerate
        first.commandArguments = ["music", "generate", "A quiet piano", "--future-option", "one", "--hf-token", "first-secret"]
        var second = first
        second.commandArguments = ["music", "generate", "A brass band", "--future-option", "two", "--hf-token", "second-secret"]
        let changes = StudioResultComparison.differences(first, second)
        XCTAssertEqual(changes.map(\.id), ["argument.0", "extraArguments"])
        XCTAssertEqual(changes.first?.first, "A quiet piano")
        XCTAssertEqual(changes.first?.second, "A brass band")
        XCTAssertTrue(changes.last?.second.contains("two") == true)
        XCTAssertFalse(changes.description.contains("first-secret"))
        XCTAssertFalse(changes.description.contains("second-secret"))
    }

    func testComparisonIdentifiesCommandsAndDoesNotInventSettingsForLegacyResults() throws {
        let first = try image(library())
        var second = first
        second.templateID = .musicGenerate
        second.commandArguments = ["music", "generate", "A quiet piano"]
        XCTAssertEqual(StudioResultComparison.differences(first, second).first?.id, "command")
        second.commandArguments = nil
        second.commandDraft = nil
        XCTAssertFalse(StudioResultComparison.hasSettings(second))
        XCTAssertTrue(StudioResultComparison.differences(first, second).isEmpty)
    }

    func testImagePanningStaysInsideTheViewportAfterZoomingOutOrResizing() {
        let pan = CGSize(width: 400, height: -600)
        XCTAssertEqual(StudioResultViewport.clampedPan(pan, zoom: 3, size: CGSize(width: 800, height: 600)), pan)
        XCTAssertEqual(StudioResultViewport.clampedPan(pan, zoom: 2, size: CGSize(width: 800, height: 600)),
                       CGSize(width: 400, height: -300))
        XCTAssertEqual(StudioResultViewport.clampedPan(pan, zoom: 2, size: CGSize(width: 400, height: 300)),
                       CGSize(width: 200, height: -150))
        XCTAssertEqual(StudioResultViewport.clampedPan(pan, zoom: 1, size: CGSize(width: 800, height: 600)), .zero)
    }

    func testEveryContinuationCarriesSourceAndLineageWithoutChangingOriginal() throws {
        let item = try image(library())
        let url = URL(fileURLWithPath: "/tmp/source.png")
        for action in StudioResultContinuation.allCases {
            let draft = try XCTUnwrap(action.draft(from: item, url: url, baseline: StudioDraft()))
            XCTAssertEqual(draft.parentID, item.id)
            XCTAssertEqual(action == .reference ? draft.referenceImagePaths : draft.inputPath, url.path)
        }
        XCTAssertEqual(item.commandDraft?.outputPath, url.path)
        XCTAssertNil(StudioResultContinuation.edit.draft(from: item, url: URL(fileURLWithPath: "/tmp/sound.wav"), baseline: StudioDraft()))
    }

    func testSavingOntoSourceAndReplacingDestinationPreserveTheOriginal() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("export-\(UUID())")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let source = directory.appendingPathComponent("source.png")
        let target = directory.appendingPathComponent("target.png")
        try Data("original".utf8).write(to: source)
        try Data("old copy".utf8).write(to: target)
        try StudioFileExport.copy(source, to: source)
        try StudioFileExport.copy(source, to: target)
        XCTAssertEqual(try Data(contentsOf: source), Data("original".utf8))
        XCTAssertEqual(try Data(contentsOf: target), Data("original".utf8))
        let missing = directory.appendingPathComponent("missing")
        XCTAssertThrowsError(try StudioFileExport.copy(missing, to: target))
        XCTAssertThrowsError(try StudioFileExport.copy(missing, to: missing))
        XCTAssertEqual(try Data(contentsOf: target), Data("original".utf8))
    }

    func testFolderExportPreservesCollisionsDeduplicatesSharedSourcesAndReportsMissingFiles() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("batch-export-\(UUID())")
        let first = root.appendingPathComponent("a/result.png")
        let second = root.appendingPathComponent("b/result.png")
        let folder = root.appendingPathComponent("saved")
        for directory in [first.deletingLastPathComponent(), second.deletingLastPathComponent(), folder] {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        }
        defer { try? FileManager.default.removeItem(at: root) }
        try Data("first".utf8).write(to: first)
        try Data("second".utf8).write(to: second)
        let existing = folder.appendingPathComponent("result.png")
        try Data("keep".utf8).write(to: existing)
        let missing = root.appendingPathComponent("missing.png")
        let report = StudioFileExport.copy([first, first, second, missing], into: folder)
        XCTAssertEqual(report.destinations.map(\.lastPathComponent), ["result-2.png", "result-3.png"])
        XCTAssertEqual(report.failures, [missing])
        XCTAssertEqual(try Data(contentsOf: existing), Data("keep".utf8))
        XCTAssertEqual(try Data(contentsOf: first), Data("first".utf8))
        XCTAssertEqual(try Data(contentsOf: report.destinations[1]), Data("second".utf8))
        let failed = StudioFileExport.copy([first], into: root.appendingPathComponent("not-a-folder"))
        XCTAssertEqual(failed.failures, [first])
        XCTAssertTrue(failed.destinations.isEmpty)
    }

    func testResultFocusFollowsSelectionAndSurvivesRestartUntilItsRowIsDeleted() throws {
        let store = library()
        var first = try image(store)
        first.outputURL = URL(fileURLWithPath: "/tmp/first.png")
        var second = try image(store)
        second.outputURL = URL(fileURLWithPath: "/tmp/second.png")
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("focus-\(UUID()).json")
        defer { try? FileManager.default.removeItem(at: url) }
        let sessions = StudioTaskSessions(url: url)
        let selection = StudioResultSelection(itemID: first.id, url: try XCTUnwrap(first.outputURL))
        sessions.rememberSelection(first.id, for: .createImage)
        sessions.setFocus(selection, for: .imageGenerate)
        sessions.rememberSelection(UUID(), for: .chat)
        XCTAssertEqual(sessions.focusedResult(for: .imageGenerate, items: [first, second]), selection)
        sessions.flush()
        let restored = StudioTaskSessions(url: url)
        XCTAssertEqual(restored.focusedResult(for: .imageGenerate, items: [first, second]), selection)
        restored.rememberSelection(second.id, for: .createImage)
        XCTAssertNil(restored.focusedResult(for: .imageGenerate, items: [first, second]))
        restored.setFocus(selection, for: .imageGenerate)
        restored.forgetLibraryItems([first.id, second.id])
        XCTAssertNil(restored.focusedResult(for: .imageGenerate, items: [first, second]))
        XCTAssertNil(restored.selection(for: .createImage, items: [first, second], preferredID: nil).item)
    }

    func testHistoryRecordsBothCompletionsWithoutAnyViewAndObservingTwiceIsIdempotent() async throws {
        let runner = RecordingProcessRunner()
        let controller = MereRunController(secretStore: InMemorySecretStore(), processRunner: runner, resolvesCLIOnInit: false)
        defer { controller.terminateAllProcesses() }
        let store = library()
        store.observe(controller: controller)
        store.observe(controller: controller)
        let first = try image(store)
        let second = try image(store, seed: "99")
        for item in [first, second] {
            let template = try XCTUnwrap(CommandCatalog.template(id: .imageGenerate))
            controller.run(studio: StudioRunRequest(id: item.id, mode: .createImage, templateID: .imageGenerate,
                                                   template: template, draft: try XCTUnwrap(item.commandDraft)))
        }
        runner.starts[1].termination(0)
        runner.starts[0].termination(3)
        try await Task.sleep(for: .milliseconds(100))
        XCTAssertEqual(store.items.count, 2)
        XCTAssertEqual(store.items.first(where: { $0.id == first.id })?.status, .failed)
        XCTAssertEqual(store.items.first(where: { $0.id == second.id })?.status, .completed)
        let restored = StudioLibraryStore(libraryURL: store.libraryURL)
        XCTAssertEqual(restored.items.map(\.id), store.items.map(\.id))
        XCTAssertEqual(restored.items.map(\.status), store.items.map(\.status))
        XCTAssertEqual(restored.items.map(\.commandArguments), store.items.map(\.commandArguments))
    }

    func testAnalyzeRejectsAReplacedInputAtTheSamePath() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("input-\(UUID()).png")
        defer { try? FileManager.default.removeItem(at: url) }
        try Data("first".utf8).write(to: url)
        var item = try image(library())
        item.inputURL = url
        item.inputIdentity = StudioInputIdentity.read(url)
        XCTAssertTrue(StudioInputIdentity.matches(item: item, input: url))
        try Data("replaced content".utf8).write(to: url)
        XCTAssertFalse(StudioInputIdentity.matches(item: item, input: url))
        XCTAssertFalse(StudioInputIdentity.matches(item: item, input: nil))
    }

    func testCommandSettingsDiscardSecretsEvenWithoutAnEnvironmentTransport() throws {
        var form = StudioConsoleDraft()
        form["--infinity-api-key"] = .text("fixture-secret")
        let state = StudioTaskCommandState(templateID: .visionServe, sourceArguments: [], form: form)
        XCTAssertNil(state.withoutSecrets.form.values["--infinity-api-key"])
        XCTAssertEqual(state.form.text("--infinity-api-key"), "fixture-secret")
    }

    func testOptionalAndDictionaryTaskStateNeverPersistLaunchSecrets() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("secret-state-\(UUID()).json")
        defer { try? FileManager.default.removeItem(at: url) }
        let sessions = StudioTaskSessions(url: url)
        var draft = CommandDraft()
        draft.apiKey = "fixture-secret"
        sessions.set(["a": draft], for: "dictionary")
        sessions.set(Optional(draft), for: "optional")
        sessions.flush()
        let restored = StudioTaskSessions(url: url)
        XCTAssertEqual(restored.value(for: "dictionary", default: [String: CommandDraft]())["a"]?.apiKey, "")
        XCTAssertEqual(restored.value(for: "optional", default: Optional<CommandDraft>.none)?.apiKey, "")
    }

    func testEveryCommandHasTaskOwnershipAndPresentationMetadata() {
        for template in CommandCatalog.templates where template.externalURL == nil {
            XCTAssertTrue(template.id.studioTask.commandTemplates.contains(where: { $0.id == template.id }), template.id.rawValue)
            for option in template.id.capability?.options ?? [] {
                XCTAssertNotNil(option.group, "\(template.id.rawValue) \(option.flag)")
                XCTAssertNotNil(option.tier, "\(template.id.rawValue) \(option.flag)")
            }
        }
    }

    func testNarrowLayoutGivesTheResultRoomAndOverlaysAuxiliaryPanels() {
        let narrow = StudioLayoutPolicy.presentation(width: 600, library: true, inspector: false, command: true)
        XCTAssertFalse(narrow.showsLibrary)
        XCTAssertFalse(narrow.panelIsInline)
        XCTAssertLessThanOrEqual(narrow.panelWidth, 600)
        let wide = StudioLayoutPolicy.presentation(width: 1300, library: true, inspector: false, command: true)
        XCTAssertTrue(wide.showsLibrary)
        XCTAssertTrue(wide.panelIsInline)
    }
}
