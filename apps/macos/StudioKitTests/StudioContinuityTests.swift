import Foundation
import MereRunContract
import StudioTestSupport
@testable import StudioKit
import XCTest

@MainActor
final class StudioContinuityTests: XCTestCase {
    private func temporaryLibrary() -> StudioLibraryStore {
        StudioLibraryStore(libraryURL: FileManager.default.temporaryDirectory
            .appendingPathComponent("studio-continuity-\(UUID()).json"))
    }

    func testConsoleEditsLaunchFromAnEmptyComposerAndHistoryReplaysExactly() throws {
        let runner = RecordingProcessRunner()
        let controller = MereRunController(processRunner: runner, resolvesCLIOnInit: false)
        defer { controller.terminateAllProcesses() }
        let template = try XCTUnwrap(CommandCatalog.template(id: .imageGenerate))
        var seed = template.defaultDraft()
        seed.prompt = ""
        seed.outputPath = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID()).png").path
        var form = StudioConsoleCommand.seed(template: template, draft: seed)
        form["--prompt"] = .text("a green ceramic bowl")
        form["--width"] = .text("768")
        form["--ref-image"] = .text("/tmp/first reference.png\n/tmp/second.png")
        form.extraArguments = "--future-option 'two words'"
        let launch = try XCTUnwrap(StudioConsoleRun(template: template, draft: form, seed: seed))
        XCTAssertNil(launch.validationMessage)
        XCTAssertTrue(controller.runConsole(template: template, draft: launch.commandDraft,
                                            arguments: launch.arguments, requestID: UUID()))
        XCTAssertEqual(runner.starts.count, 1)
        let library = temporaryLibrary()
        let item = library.start(
            request: StudioRunRequest(mode: .createImage, templateID: template.id, template: template, draft: launch.commandDraft),
            commandPreview: "fixture", arguments: launch.arguments
        )
        XCTAssertEqual(item.prompt, "a green ceramic bowl")
        let replay = try XCTUnwrap(StudioLibraryReplay.request(for: item))
        XCTAssertEqual(replay.draft.width, 768)
        let execution = try XCTUnwrap(replay.execution)
        XCTAssertEqual(execution.replacing("--output", with: seed.outputPath).arguments, launch.arguments)
        XCTAssertNotEqual(replay.draft.outputPath, seed.outputPath)
        XCTAssertTrue(execution.arguments.contains("two words"))
    }

    func testConsolePreservesServerAuthenticationChecksAfterEditingTheHost() throws {
        let template = try XCTUnwrap(CommandCatalog.template(id: .apiServe))
        var form = StudioConsoleCommand.seed(template: template, draft: template.defaultDraft())
        form["--host"] = .text("0.0.0.0")
        var launch = try XCTUnwrap(StudioConsoleRun(template: template, draft: form, seed: template.defaultDraft()))
        XCTAssertNotNil(launch.validationMessage)
        let runner = RecordingProcessRunner()
        let controller = MereRunController(processRunner: runner, resolvesCLIOnInit: false)
        defer { controller.terminateAllProcesses() }
        XCTAssertFalse(controller.runConsole(template: template, draft: launch.commandDraft,
            arguments: launch.arguments, requestID: UUID()))
        XCTAssertTrue(runner.starts.isEmpty)
        form["--api-key"] = .text("fixture-key")
        launch = try XCTUnwrap(StudioConsoleRun(template: template, draft: form, seed: template.defaultDraft()))
        XCTAssertNil(launch.validationMessage)
        XCTAssertFalse(launch.arguments.contains("fixture-key"))
        XCTAssertTrue(controller.runConsole(template: template, draft: launch.commandDraft,
            arguments: launch.arguments, requestID: UUID()))
        XCTAssertEqual(runner.starts.first?.configuration.environment["MERERUN_API_KEY"], "fixture-key")
    }

    func testPositionalMusicPromptEditsUpdateTheComposer() throws {
        let capability = try XCTUnwrap(CommandTemplateID.musicGenerate.capability)
        let before = StudioConsoleCommand.seed(capability: capability, arguments: ["music", "generate", "First"])
        let after = StudioConsoleCommand.seed(capability: capability, arguments: ["music", "generate", "Second"])
        var draft = StudioDraft()
        after.applyingChanges(from: before, to: &draft, mode: .music, templateID: .musicGenerate)
        XCTAssertEqual(draft.prompt, "Second")
    }

    func testBranchRestoresTheSettingsAtTheBranchPointIncludingAnEmptyInstruction() throws {
        let library = temporaryLibrary()
        let id = UUID()
        library.appendUser(conversationID: id, mode: .chat, model: "model-a", systemPrompt: nil, content: "First")
        library.appendAssistant(conversationID: id, content: "Answer", exitCode: 0, model: "model-a")
        let point = try XCTUnwrap(library.items.first?.messages?.last?.id)
        library.appendUser(conversationID: id, mode: .code, model: "model-b", systemPrompt: "Later", content: "Second")
        let original = try XCTUnwrap(library.items.first)
        let branch = try XCTUnwrap(library.branch(conversationID: id, at: point, inclusive: true))
        XCTAssertEqual(branch.mode, .chat)
        XCTAssertEqual(branch.model, "model-a")
        XCTAssertNil(branch.systemPrompt)
        XCTAssertEqual(library.items.first(where: { $0.id == id }), original)
    }

    func testRelaunchReconcilesUnownedRunsAndKeepsPartialArtifacts() throws {
        let library = temporaryLibrary()
        let template = try XCTUnwrap(CommandCatalog.template(id: .imageGenerate))
        let request = StudioRunRequest(mode: .createImage, templateID: template.id,
                                       template: template, draft: template.defaultDraft())
        library.start(request: request, commandPreview: "fixture", status: .running)
        let partial = URL(fileURLWithPath: "/tmp/partial.png")
        library.updateOutput(id: request.id, outputURL: partial)
        let restored = StudioLibraryStore(libraryURL: library.libraryURL)
        XCTAssertEqual(restored.items.first?.status, .interrupted)
        XCTAssertEqual(restored.items.first?.outputURL, partial)
        let cards = StudioFeedCardBuilder.cards(items: restored.items, mode: .createImage, job: { _ in nil })
        XCTAssertNotEqual(cards.first?.kind, .running)
    }

    func testStoppingAConversationDoesNotStopTheNewerImageJob() throws {
        let runner = RecordingProcessRunner()
        let controller = MereRunController(processRunner: runner, resolvesCLIOnInit: false)
        defer { controller.terminateAllProcesses() }
        let chat = try XCTUnwrap(CommandCatalog.template(id: .textChat))
        let image = try XCTUnwrap(CommandCatalog.template(id: .imageGenerate))
        let conversationID = UUID()
        controller.run(studio: StudioRunRequest(mode: .chat, templateID: chat.id, template: chat,
                                               draft: chat.defaultDraft(), conversationID: conversationID))
        controller.run(studio: StudioRunRequest(mode: .createImage, templateID: image.id, template: image,
                                               draft: image.defaultDraft()))
        XCTAssertEqual(runner.processes.count, 2)
        controller.cancelConversation(conversationID)
        XCTAssertEqual(runner.processes.first?.terminateCallCount, 1)
        XCTAssertEqual(runner.processes.last?.terminateCallCount, 0)
    }

    func testTaskSelectionRestoresTheDraftsThreadAndPreservesAnExplicitNewChat() throws {
        let library = temporaryLibrary()
        let olderID = UUID()
        let newerID = UUID()
        library.appendUser(conversationID: olderID, mode: .chat, model: "first", systemPrompt: nil, content: "Older")
        library.appendUser(conversationID: newerID, mode: .chat, model: "second", systemPrompt: nil, content: "Newer")
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("selection-\(UUID()).json")
        defer { try? FileManager.default.removeItem(at: url) }
        let sessions = StudioTaskSessions(url: url)
        sessions.rememberSelection(olderID, for: .chat)
        sessions.flush()
        let restored = StudioTaskSessions(url: url)
        let selection = restored.selection(for: .chat, items: library.items, preferredID: nil)
        XCTAssertEqual(selection.item?.id, olderID)
        XCTAssertTrue(selection.hasMemory)
        XCTAssertFalse(selection.isExplicit)
        let picked = restored.selection(for: .chat, items: library.items, preferredID: newerID)
        XCTAssertEqual(picked.item?.id, newerID)
        XCTAssertTrue(picked.isExplicit)
        restored.rememberSelection(nil, for: .chat)
        let fresh = restored.selection(for: .chat, items: library.items, preferredID: nil)
        XCTAssertNil(fresh.item)
        XCTAssertTrue(fresh.hasMemory)
    }

    func testFullTaskSettingsSurviveRelaunchAndSecretsOnlyStayInMemory() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("studio-tasks-\(UUID()).json")
        let sessions = StudioTaskSessions(url: url)
        var draft = StudioDraft()
        draft.prompt = ""
        draft.model = "chosen-model"
        draft.width = 768
        draft.seed = "42"
        sessions.set(draft, for: "image.generate.draft")
        var command = CommandDraft()
        command.apiKey = "test-only-secret"
        command.openWebUIAdminPassword = "test-only-password"
        sessions.set(command, for: "server.command")
        XCTAssertEqual(sessions.value(for: "server.command", default: CommandDraft()).apiKey, command.apiKey)
        sessions.flush()
        let restored = StudioTaskSessions(url: url)
        XCTAssertEqual(restored.value(for: "image.generate.draft", default: StudioDraft()), draft)
        XCTAssertEqual(restored.value(for: "server.command", default: CommandDraft()).apiKey, "")
        let text = try String(contentsOf: url, encoding: .utf8)
        XCTAssertFalse(text.contains("test-only-secret"))
        XCTAssertFalse(text.contains("test-only-password"))
    }

    func testUnsentConversationDraftsKeepTheirOwnTextAttachmentAndPresetAfterRelaunch() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("thread-drafts-\(UUID()).json")
        defer { try? FileManager.default.removeItem(at: url) }
        let sessions = StudioTaskSessions(url: url)
        let older = UUID()
        let newer = UUID()
        var draft = StudioDraft()
        draft.prompt = "Unsent follow-up"
        draft.inputPath = "/tmp/attached.png"
        draft.model = "chosen-model"
        draft.secondaryText = "Custom instruction"
        draft.temperature = 0.7
        sessions.rememberConversationDraft(draft, conversationID: older, mode: .chat)
        var other = StudioDraft()
        other.prompt = "Different thread"
        sessions.rememberConversationDraft(other, conversationID: newer, mode: .chat)
        other.prompt = "A new thread"
        sessions.rememberConversationDraft(other, conversationID: nil, mode: .chat)
        other.prompt = "Code preset"
        sessions.rememberConversationDraft(other, conversationID: older, mode: .code)
        sessions.flush()
        let restored = StudioTaskSessions(url: url)
        XCTAssertEqual(restored.conversationDraft(conversationID: older, mode: .chat), draft)
        XCTAssertEqual(restored.conversationDraft(conversationID: newer, mode: .chat)?.prompt, "Different thread")
        XCTAssertEqual(restored.conversationDraft(conversationID: nil, mode: .chat)?.prompt, "A new thread")
        XCTAssertEqual(restored.conversationDraft(conversationID: older, mode: .code), other)
        restored.forgetConversationDrafts([older])
        restored.flush()
        let afterDeletion = StudioTaskSessions(url: url)
        XCTAssertNil(afterDeletion.conversationDraft(conversationID: older, mode: .chat))
        XCTAssertNil(afterDeletion.conversationDraft(conversationID: older, mode: .code))
        XCTAssertEqual(afterDeletion.conversationDraft(conversationID: newer, mode: .chat)?.prompt, "Different thread")
    }
}
