import Foundation
import StudioTestSupport
@testable import StudioKit
import XCTest

@MainActor
final class StudioPromptTaskControllerTests: XCTestCase {
    private var root: URL!
    private var runner: RecordingProcessRunner!
    private var controller: MereRunController!
    private var library: StudioLibraryStore!
    private var prompt: StudioPromptTaskController!

    override func setUp() async throws {
        try await MainActor.run {
            root = FileManager.default.temporaryDirectory.appendingPathComponent("prompt-task-\(UUID())")
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            runner = RecordingProcessRunner()
            controller = MereRunController(processRunner: runner, resolvesCLIOnInit: false,
                taskSessions: StudioTaskSessions(url: root.appendingPathComponent("sessions.json")))
            library = StudioLibraryStore(libraryURL: root.appendingPathComponent("library.json"))
            library.observe(controller: controller)
            prompt = StudioPromptTaskController(controller: controller, library: library)
        }
    }

    override func tearDown() async throws {
        try await MainActor.run {
            controller.terminateAllProcesses()
            controller.taskSessions.flush()
            prompt = nil
            library = nil
            controller = nil
            runner = nil
            try FileManager.default.removeItem(at: root)
        }
    }

    private func activate(_ mode: StudioMode, selected: UUID? = nil) {
        _ = prompt.activate(mode, preferredID: selected)
        controller.readinessByMode[mode] = .ready
    }

    private func thread(mode: StudioMode = .chat, text: String = "First question", reply: String? = nil) -> UUID {
        let id = UUID()
        library.appendUser(conversationID: id, mode: mode, model: "chosen-model", systemPrompt: "Be precise", content: text)
        if let reply { library.appendAssistant(conversationID: id, content: reply, exitCode: 0, model: "chosen-model", systemPrompt: "Be precise") }
        return id
    }

    private func override(_ request: StudioRunRequest, flag: String, value: String) {
        var form = StudioConsoleCommand.seed(template: request.template, draft: request.draft)
        form[flag] = .text(value)
        controller.taskSessions.set(StudioTaskCommandState(templateID: request.templateID,
            sourceArguments: request.template.arguments(from: request.draft), form: form),
            for: request.templateID.studioTask.rawValue + ".commandOverride")
    }

    private func temporaryOutput(_ draft: CommandDraft) -> StudioOutputLocation.Preparation {
        var moved = draft
        moved.outputPath = root.appendingPathComponent("output-\(UUID()).png").path
        return StudioOutputLocation.Preparation(draft: moved)
    }

    func testEditsPersistSynchronouslyAndEachPromptTaskRestoresItsFullDraft() throws {
        var drafts: [StudioMode: StudioDraft] = [:]
        for (index, mode) in StudioMode.allCases.enumerated() {
            activate(mode)
            prompt.draft.prompt = "Unsent \(mode.rawValue)"
            prompt.draft.seed = String(index)
            prompt.draft.width = 768 + index
            prompt.draft.inputPath = root.appendingPathComponent("\(mode.rawValue).png").path
            drafts[mode] = prompt.draft
            let saved = controller.taskSessions.value(for: mode.task.rawValue + ".draft", default: StudioDraft())
            XCTAssertEqual(saved, prompt.draft, "No view observer or task switch is needed to save \(mode).")
        }
        for mode in StudioMode.allCases.reversed() {
            activate(mode)
            XCTAssertEqual(prompt.draft, drafts[mode])
        }
        controller.taskSessions.flush()
        let restoredHost = MereRunController(processRunner: RecordingProcessRunner(), resolvesCLIOnInit: false,
            taskSessions: StudioTaskSessions(url: root.appendingPathComponent("sessions.json")))
        defer { restoredHost.terminateAllProcesses() }
        let restored = StudioPromptTaskController(controller: restoredHost, library: library)
        _ = restored.activate(.video, preferredID: nil)
        XCTAssertEqual(restored.draft, drafts[.video])
        XCTAssertTrue(runner.starts.isEmpty)
    }

    func testLegacyImportKeepsUnvisitedTasksAndFullSessionDraftsTakePrecedence() throws {
        activate(.createImage)
        prompt.draft.prompt = "Current full draft"
        prompt.draft.seed = "73"
        let current = prompt.draft
        var oldImage = StudioDraft()
        oldImage.prompt = "Old image"
        var oldVideo = StudioDraft()
        oldVideo.prompt = "Unvisited legacy video"
        oldVideo.secondaryText = "Keep this too"
        let encoded = StudioDraftMemory.encode([
            .imageGenerate: StudioDraftMemory.entry(for: oldImage),
            .videoGenerate: StudioDraftMemory.entry(for: oldVideo)
        ])
        prompt.importLegacyDrafts(encoded)
        controller.taskSessions.flush()
        let restored = StudioTaskSessions(url: root.appendingPathComponent("sessions.json"))
        XCTAssertEqual(restored.value(for: StudioTask.imageGenerate.rawValue + ".draft", default: StudioDraft()), current)
        XCTAssertEqual(restored.value(for: StudioTask.videoGenerate.rawValue + ".draft", default: StudioDraft()).prompt,
                       oldVideo.prompt)
        activate(.video)
        XCTAssertEqual(prompt.draft.secondaryText, oldVideo.secondaryText)
        prompt.draft.prompt = "Edited after import"
        prompt.importLegacyDrafts(encoded)
        XCTAssertEqual(prompt.draft.prompt, "Edited after import")
    }

    func testThreadSwitchesAndTaskDetoursPreserveUnsentMessagesAndExplicitNewChat() throws {
        let first = thread(text: "First thread")
        let second = thread(text: "Second thread")
        activate(.chat, selected: first)
        prompt.draft.prompt = "Unsent first follow-up"
        prompt.draft.inputPath = "/tmp/first.png"
        let firstDraft = prompt.draft
        prompt.restoreConversation(try XCTUnwrap(library.items.first { $0.id == second }))
        prompt.draft.prompt = "Unsent second follow-up"
        let secondDraft = prompt.draft
        prompt.startNewConversation()
        prompt.draft.prompt = "Unsent new thread"
        let newDraft = prompt.draft
        activate(.createImage)
        activate(.chat)
        XCTAssertNil(prompt.activeConversationID)
        XCTAssertEqual(prompt.draft, newDraft)
        prompt.restoreConversation(try XCTUnwrap(library.items.first { $0.id == first }))
        XCTAssertEqual(prompt.draft, firstDraft)
        prompt.restoreConversation(try XCTUnwrap(library.items.first { $0.id == second }))
        XCTAssertEqual(prompt.draft, secondDraft)
        XCTAssertEqual(library.items.count, 2)
    }

    func testChatCodePresetSwitchKeepsTheOpenThreadAndSeparateDrafts() {
        let id = thread()
        activate(.chat, selected: id)
        prompt.draft.prompt = "Chat follow-up"
        prompt.draft.temperature = 0.3
        let chat = prompt.draft
        activate(.code, selected: id)
        XCTAssertEqual(prompt.activeConversationID, id)
        XCTAssertEqual(prompt.draft.prompt, "")
        prompt.draft.prompt = "Code follow-up"
        let code = prompt.draft
        activate(.chat, selected: id)
        XCTAssertEqual(prompt.activeConversationID, id)
        XCTAssertEqual(prompt.draft, chat)
        activate(.code, selected: id)
        XCTAssertEqual(prompt.draft, code)
        XCTAssertEqual(library.items.count, 1)
    }

    func testOpeningTheMostRecentOtherPresetDoesNotSaveTheDepartingDraftIntoChat() {
        let id = thread(mode: .code)
        activate(.createImage)
        prompt.draft.prompt = "Image must stay an image"
        let image = prompt.draft
        let activation = prompt.activate(.chat, preferredID: nil)
        XCTAssertEqual(activation.mode, .code)
        XCTAssertEqual(activation.selectedLibraryID, id)
        XCTAssertEqual(prompt.draft.prompt, "")
        XCTAssertFalse(controller.taskSessions.contains(StudioTask.chatChat.rawValue + ".draft"))
        activate(.createImage)
        XCTAssertEqual(prompt.draft, image)
    }

    func testAnalyzeHandoffKeepsSourceAndTargetSettingsSeparate() {
        activate(.segment)
        prompt.draft.model = "segment-model"
        activate(.findObjects)
        prompt.draft.inputPath = "/tmp/objects.png"
        prompt.draft.prompt = "Every red chair"
        let source = prompt.draft
        prompt.prepareAnalyzeHandoff(to: StudioMode.segment.task)
        let activation = prompt.activate(.segment, preferredID: nil)
        XCTAssertNil(activation.selectedLibraryID)
        XCTAssertEqual(prompt.draft.inputPath, source.inputPath)
        XCTAssertEqual(prompt.draft.prompt, source.prompt)
        XCTAssertEqual(prompt.draft.model, "segment-model")
        activate(.findObjects)
        XCTAssertEqual(prompt.draft, source)
    }

    func testGenerationUsesCommandOverridesAndRecordsTheLaunchedCommand() throws {
        activate(.createImage)
        prompt.draft.prompt = "A green ceramic bowl"
        let base = try StudioCommandAdapter.makeRequest(mode: .createImage, draft: prompt.draft)
        override(base, flag: "--width", value: "768")
        let submission = try XCTUnwrap(prompt.runPrompt(inventory: [], prepareOutput: temporaryOutput))
        let request = submission.request
        XCTAssertEqual(request.draft.width, 768)
        XCTAssertEqual(runner.starts.count, 1)
        let row = try XCTUnwrap(library.items.first { $0.id == request.id })
        XCTAssertEqual(row.commandArguments, request.execution?.arguments)
        XCTAssertEqual(row.commandDraft?.outputPath, request.draft.outputPath)
        XCTAssertTrue(runner.starts[0].configuration.arguments.contains(request.draft.outputPath))
        XCTAssertEqual(prompt.draft.prompt, "A green ceramic bowl")
    }

    func testRejectedGenerationLeavesDraftHistoryAndOutputDirectoriesUntouched() throws {
        activate(.createImage)
        prompt.draft.prompt = "Keep my work"
        let original = prompt.draft
        let base = try StudioCommandAdapter.makeRequest(mode: .createImage, draft: original)
        override(base, flag: "--width", value: "not-an-integer")
        var preparedOutput = false
        XCTAssertThrowsError(try prompt.runPrompt(inventory: [], prepareOutput: { draft in
            preparedOutput = true
            return self.temporaryOutput(draft)
        }))
        XCTAssertFalse(preparedOutput)
        XCTAssertEqual(prompt.draft, original)
        XCTAssertTrue(library.items.isEmpty)
        XCTAssertTrue(controller.jobs.all.isEmpty)
        XCTAssertTrue(runner.starts.isEmpty)
    }

    func testBlockedReadinessLeavesTheDraftAndLibraryUntouched() {
        activate(.createImage)
        prompt.draft.prompt = "Still editable"
        controller.readinessByMode[.createImage] = .missingModel("missing-model")
        XCTAssertThrowsError(try prompt.runPrompt(inventory: [], prepareOutput: temporaryOutput))
        XCTAssertEqual(prompt.draft.prompt, "Still editable")
        XCTAssertTrue(library.items.isEmpty)
        XCTAssertTrue(runner.starts.isEmpty)
    }

    func testSendingATurnUsesCommandOverridesAndPreventsDuplicateQueuedTurns() throws {
        activate(.chat)
        prompt.draft.prompt = "Explain the image"
        prompt.draft.inputPath = "/tmp/turn.png"
        let base = try StudioCommandAdapter.makeRequest(mode: .chat, draft: prompt.draft)
        override(base, flag: "--max-tokens", value: "128")
        let sent = try XCTUnwrap(prompt.runPrompt(inventory: []))
        let id = try XCTUnwrap(sent.request.conversationID)
        XCTAssertEqual(prompt.activeConversationID, id)
        XCTAssertEqual(prompt.draft.prompt, "")
        XCTAssertEqual(prompt.draft.inputPath, "")
        XCTAssertEqual(sent.request.draft.maxTokens, 128)
        XCTAssertEqual(library.items.first?.messages?.last?.imagePath, "/tmp/turn.png")
        XCTAssertTrue(controller.runningConversationIDs.contains(id))
        prompt.draft.prompt = "Wait for the first turn"
        XCTAssertNil(try prompt.runPrompt(inventory: []))
        XCTAssertEqual(prompt.draft.prompt, "Wait for the first turn")
        XCTAssertEqual(library.items.first?.messages?.count, 1)
        XCTAssertEqual(runner.starts.count, 1)
    }

    func testRejectedRetryPreservesReplyDraftSelectionAndStoredTranscript() throws {
        let id = thread(reply: "Keep this answer")
        activate(.chat, selected: id)
        prompt.draft.prompt = "Unsent follow-up"
        let base = try StudioCommandAdapter.makeRequest(mode: .chat, draft: prompt.draft)
        override(base, flag: "--max-tokens", value: "invalid")
        let original = try XCTUnwrap(library.items.first { $0.id == id })
        let stored = try Data(contentsOf: library.libraryURL)
        let draft = prompt.draft
        XCTAssertThrowsError(try prompt.retryLastTurn(inventory: []))
        XCTAssertEqual(library.items.first { $0.id == id }, original)
        XCTAssertEqual(try Data(contentsOf: library.libraryURL), stored)
        XCTAssertEqual(prompt.draft, draft)
        XCTAssertEqual(prompt.activeConversationID, id)
        XCTAssertTrue(controller.jobs.all.isEmpty)
        XCTAssertTrue(runner.starts.isEmpty)
    }

    func testRetryUsesTheSameCommandResolutionAsSendWithoutConsumingUnsentText() throws {
        let id = thread(reply: "Replace this answer")
        activate(.chat, selected: id)
        prompt.draft.prompt = "Do not send this follow-up yet"
        let base = try StudioCommandAdapter.makeRequest(mode: .chat, draft: prompt.draft)
        override(base, flag: "--max-tokens", value: "128")
        let originalDraft = prompt.draft
        let request = try XCTUnwrap(prompt.retryLastTurn(inventory: []))
        XCTAssertEqual(request.draft.maxTokens, 128)
        XCTAssertEqual(request.conversationID, id)
        XCTAssertTrue(request.draft.prompt.contains("First question"))
        XCTAssertFalse(request.draft.prompt.contains(originalDraft.prompt))
        XCTAssertEqual(prompt.draft, originalDraft)
        XCTAssertEqual(library.items.first?.messages?.count, 1)
        XCTAssertEqual(runner.starts.count, 1)
    }

    func testStopTargetsTheOpenConversationWhileAnotherTaskIsRunning() throws {
        activate(.chat)
        prompt.draft.prompt = "Chat is first"
        let chat = try XCTUnwrap(prompt.runPrompt(inventory: []))
        activate(.createImage)
        prompt.draft.prompt = "Image is newer"
        _ = try prompt.runPrompt(inventory: [], prepareOutput: temporaryOutput)
        activate(.chat, selected: chat.request.conversationID)
        XCTAssertEqual(runner.processes.count, 2)
        prompt.stop(task: .chatChat)
        XCTAssertEqual(runner.processes[0].terminateCallCount, 1)
        XCTAssertEqual(runner.processes[1].terminateCallCount, 0)
    }

    func testEditingTheFirstUserTurnKeepsItsTextAsANewUnsentConversation() throws {
        let id = thread(reply: "An answer")
        activate(.chat, selected: id)
        let messageID = try XCTUnwrap(library.items.first?.messages?.first?.id)
        XCTAssertTrue(prompt.editMessage(messageID))
        XCTAssertNil(prompt.activeConversationID)
        XCTAssertTrue(library.items.isEmpty)
        XCTAssertEqual(prompt.draft.prompt, "First question")
        XCTAssertEqual(controller.taskSessions.conversationDraft(conversationID: nil, mode: .chat), prompt.draft)
        XCTAssertNil(controller.taskSessions.conversationDraft(conversationID: id, mode: .chat))
    }

    func testRejectedSendKeepsNewConversationTextAndAttachmentWithoutCreatingHistory() throws {
        activate(.chat)
        prompt.draft.prompt = "A new question"
        prompt.draft.inputPath = "/tmp/unsent-image.png"
        let original = prompt.draft
        let base = try StudioCommandAdapter.makeRequest(mode: .chat, draft: original)
        override(base, flag: "--max-tokens", value: "invalid")
        XCTAssertThrowsError(try prompt.runPrompt(inventory: []))
        XCTAssertEqual(prompt.draft, original)
        XCTAssertNil(prompt.activeConversationID)
        XCTAssertTrue(library.items.isEmpty)
        XCTAssertTrue(controller.runningConversationIDs.isEmpty)
        XCTAssertTrue(runner.starts.isEmpty)
    }

    func testReplayKeepsRecordedArgumentsWhenTheCurrentCommandPanelHasDifferentValues() throws {
        activate(.chat)
        prompt.draft.prompt = "Original request"
        let base = try StudioCommandAdapter.makeRequest(mode: .chat, draft: prompt.draft)
        override(base, flag: "--max-tokens", value: "128")
        let original = controller.taskSessions.resolving(base)
        let row = library.start(request: original, commandPreview: "fixture")
        override(base, flag: "--max-tokens", value: "256")
        let replay = try prompt.replay(row)
        XCTAssertEqual(replay.parentID, row.id)
        XCTAssertEqual(replay.draft.maxTokens, 128)
        XCTAssertEqual(replay.execution?.arguments, original.execution?.arguments)
        XCTAssertEqual(runner.starts.count, 1)
    }

    func testBranchUsesThePresetAtTheBranchPointAndPreservesTheOriginalUnsentDraft() throws {
        let id = thread(mode: .chat, reply: "First answer")
        let point = try XCTUnwrap(library.items.first?.messages?.last?.id)
        library.appendUser(conversationID: id, mode: .code, model: "code-model", systemPrompt: "Write code", content: "Later turn")
        activate(.code, selected: id)
        prompt.draft.prompt = "Unsent original follow-up"
        let original = try XCTUnwrap(library.items.first { $0.id == id })
        let originalDraft = prompt.draft
        let branch = try XCTUnwrap(prompt.branchFromMessage(point))
        XCTAssertEqual(branch.mode, .chat)
        XCTAssertNotEqual(branch.selectedLibraryID, id)
        _ = prompt.activate(branch.mode, preferredID: branch.selectedLibraryID)
        XCTAssertEqual(prompt.activeConversationID, branch.selectedLibraryID)
        XCTAssertEqual(prompt.draft.model, "chosen-model")
        XCTAssertEqual(prompt.draft.secondaryText, "Be precise")
        XCTAssertEqual(prompt.draft.prompt, "")
        XCTAssertEqual(library.items.first { $0.id == id }, original)
        _ = prompt.activate(.code, preferredID: id)
        XCTAssertEqual(prompt.draft, originalDraft)
    }

    func testDeletingAnOpenThreadRestoresTheUnsentNewConversation() throws {
        activate(.chat)
        prompt.draft.prompt = "A separate new thread"
        let newDraft = prompt.draft
        let id = thread()
        prompt.restoreConversation(try XCTUnwrap(library.items.first { $0.id == id }))
        prompt.draft.prompt = "Deleted thread's follow-up"
        library.delete(id: id)
        prompt.forgetConversations([id])
        XCTAssertNil(prompt.activeConversationID)
        XCTAssertEqual(prompt.draft, newDraft)
        XCTAssertNil(controller.taskSessions.conversationDraft(conversationID: id, mode: .chat))
    }

    func testOutputFallbackMovesTheLaunchedAndRecordedDestinationTogether() throws {
        activate(.createImage)
        prompt.draft.prompt = "Fallback output"
        let base = try StudioCommandAdapter.makeRequest(mode: .createImage, draft: prompt.draft)
        override(base, flag: "--width", value: "768")
        let fallback = root.appendingPathComponent("fallback.png").path
        let submission = try XCTUnwrap(prompt.runPrompt(inventory: [], prepareOutput: { draft in
            var moved = draft
            moved.outputPath = fallback
            return StudioOutputLocation.Preparation(draft: moved, fallbackReason: "Fixture destination is read-only.")
        }))
        XCTAssertEqual(submission.outputFallbackReason, "Fixture destination is read-only.")
        XCTAssertEqual(submission.request.draft.outputPath, fallback)
        XCTAssertTrue(try XCTUnwrap(submission.request.execution).arguments.contains(fallback))
        XCTAssertEqual(library.items.first?.commandDraft?.outputPath, fallback)
        XCTAssertTrue(try XCTUnwrap(runner.starts.first).configuration.arguments.contains(fallback))
    }
}
