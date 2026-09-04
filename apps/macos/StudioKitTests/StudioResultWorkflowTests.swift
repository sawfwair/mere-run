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

    func testComparisonShowsSeedChangesAndOmitsDestinationNoise() throws {
        let store = library()
        let first = try image(store)
        let second = try image(store, seed: "99", output: "/tmp/new.png")
        let changes = StudioResultComparison.differences(first, second)
        XCTAssertEqual(changes.map(\.id), ["--seed"])
        XCTAssertEqual(changes.first?.first, "42")
        XCTAssertEqual(changes.first?.second, "99")
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
        XCTAssertThrowsError(try StudioFileExport.copy(directory.appendingPathComponent("missing"), to: target))
        XCTAssertEqual(try Data(contentsOf: target), Data("original".utf8))
    }

    func testHistoryRecordsBothCompletionsWithoutAnyViewAndObservingTwiceIsIdempotent() async throws {
        let runner = RecordingProcessRunner()
        let controller = MereRunController(processRunner: runner, resolvesCLIOnInit: false)
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
