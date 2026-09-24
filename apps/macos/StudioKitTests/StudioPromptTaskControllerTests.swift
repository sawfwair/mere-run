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
            controller = MereRunController(secretStore: InMemorySecretStore(), processRunner: runner, resolvesCLIOnInit: false,
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
        let restoredHost = MereRunController(secretStore: InMemorySecretStore(), processRunner: RecordingProcessRunner(), resolvesCLIOnInit: false,
            taskSessions: StudioTaskSessions(url: root.appendingPathComponent("sessions.json")))
        defer { restoredHost.terminateAllProcesses() }
        let restored = StudioPromptTaskController(controller: restoredHost, library: library)
        _ = restored.activate(.video, preferredID: nil)
        XCTAssertEqual(restored.draft, drafts[.video])
        XCTAssertTrue(runner.starts.isEmpty)
    }

    /// Library ▸ "Use these settings" reads the recorded command back through the contract
    /// bindings, so the prompt, model, and every option land in the draft fields that emit
    /// them — for the task that ran it, whether or not it is the open one.
    func testUseTheseSettingsRestoresARunsPromptModelAndOptionsIntoItsTaskDraft() throws {
        activate(.chat)
        let template = try XCTUnwrap(CommandCatalog.template(id: .imageGenerate))
        var recorded = template.defaultDraft()
        recorded.prompt = "a green ceramic bowl"
        recorded.model = "image-zimage-turbo"
        recorded.width = 768
        recorded.height = 512
        recorded.steps = 9
        recorded.seed = "4242"
        recorded.cfgScale = 3.5
        recorded.outputPath = root.appendingPathComponent("bowl.png").path
        let request = StudioRunRequest(mode: .createImage, templateID: .imageGenerate, template: template, draft: recorded)
        let item = library.start(request: request, commandPreview: "fixture")

        XCTAssertTrue(prompt.useSettings(from: item))
        let parked = controller.taskSessions.value(for: StudioTask.imageGenerate.rawValue + ".draft", default: StudioDraft())
        XCTAssertEqual(parked.prompt, "a green ceramic bowl")
        XCTAssertEqual(parked.model, "image-zimage-turbo")
        XCTAssertEqual(parked.width, 768)
        XCTAssertEqual(parked.height, 512)
        XCTAssertEqual(parked.steps, 9)
        XCTAssertEqual(parked.seed, "4242")
        XCTAssertEqual(parked.cfgScale, 3.5)
        XCTAssertEqual(parked.parentID, item.id)
        XCTAssertEqual(prompt.draft.prompt, "", "the open Chat draft is untouched")

        activate(.createImage)
        XCTAssertEqual(prompt.draft, parked, "opening the task lands on the restored draft")
        let replayed = try StudioCommandAdapter.makeRequest(mode: .createImage, draft: prompt.draft)
        XCTAssertEqual(replayed.draft.width, 768)
        XCTAssertEqual(replayed.draft.seed, "4242")
        XCTAssertEqual(replayed.draft.model, "image-zimage-turbo")

        prompt.draft.prompt = "something else"
        XCTAssertTrue(prompt.useSettings(from: item))
        XCTAssertEqual(prompt.draft.prompt, "a green ceramic bowl", "the open task's composer updates in place")

        var legacy = item
        legacy.commandDraft = nil
        XCTAssertFalse(prompt.useSettings(from: legacy), "a row from before commands were recorded restores nothing")
        let threadID = thread()
        let threadItem = try XCTUnwrap(library.items.first { $0.id == threadID })
        XCTAssertFalse(prompt.useSettings(from: threadItem), "threads are reopened, not restored")
        XCTAssertTrue(runner.starts.isEmpty)
    }

    /// Models ▸ "Use for … by default" is what a task starts from: a composer or parked draft
    /// that was following the previous default moves onto the new one at once, fresh drafts
    /// follow, nil restores the built-in default, and a model the user picked by hand — in a
    /// parked draft or an open thread — stays.
    func testModelsPageDefaultMovesTheTaskOntoTheModelNowAndForFreshDrafts() throws {
        activate(.createImage)
        let builtIn = prompt.draft.model
        prompt.setPreferredModel("image-zimage-turbo", for: .createImage)
        XCTAssertEqual(prompt.draft.model, "image-zimage-turbo", "the composer was on the built-in default, so it follows")
        XCTAssertEqual(controller.taskSessions.preferredModel(for: .createImage), "image-zimage-turbo")
        XCTAssertEqual(prompt.freshDraft(for: .createImage).model, "image-zimage-turbo")

        activate(.video)
        prompt.setPreferredModel("image-other", for: .createImage)
        let followed = controller.taskSessions.value(for: StudioTask.imageGenerate.rawValue + ".draft", default: StudioDraft())
        XCTAssertEqual(followed.model, "image-other", "a parked draft on the old default moves too, so the change shows on return")

        activate(.createImage)
        prompt.draft.model = "hand-picked"
        activate(.video)
        prompt.setPreferredModel("image-third", for: .createImage)
        let kept = controller.taskSessions.value(for: StudioTask.imageGenerate.rawValue + ".draft", default: StudioDraft())
        XCTAssertEqual(kept.model, "hand-picked", "a model the user chose is not overwritten")
        XCTAssertEqual(prompt.freshDraft(for: .createImage).model, "image-third", "but a fresh draft starts on the new default")

        activate(.createImage)
        prompt.draft.model = "image-third"
        prompt.setPreferredModel(nil, for: .createImage)
        XCTAssertEqual(prompt.draft.model, builtIn, "nil returns a following draft to the built-in default")

        let threadID = thread()
        activate(.chat, selected: threadID)
        XCTAssertEqual(prompt.draft.model, "chosen-model")
        prompt.setPreferredModel("text-chat-qwen3.6-4b", for: .chat)
        XCTAssertEqual(prompt.draft.model, "chosen-model", "an open thread keeps its own model")
        prompt.startNewConversation()
        XCTAssertEqual(prompt.draft.model, "text-chat-qwen3.6-4b", "a new thread starts on the default")
        XCTAssertTrue(runner.starts.isEmpty)
    }

    /// Segment and Track take their picture as a positional argument and their prompts as `--box`
    /// and `--point`. Restoring a recorded run lands the input first and the drawing after it, so
    /// the boxes and points survive the input change that clears prompts drawn on another
    /// picture, and Track's `--init-frame` / `--end-frame` come back onto its scrubber — whether
    /// the run's task is opened afterwards or is the one already open.
    func testUseTheseSettingsRestoresDrawnPromptsAndFramesWithTheirInput() throws {
        let box = StudioRegionPrompt.box(CGRect(x: 40, y: 30, width: 120, height: 80))
        let point = StudioRegionPrompt.point(CGPoint(x: 400, y: 260), isPositive: true)
        let negative = StudioRegionPrompt.point(CGPoint(x: 12, y: 18), isPositive: false)
        let expectedShapes: [StudioRegionPrompt.Shape] = [
            .box(x1: 40, y1: 30, x2: 160, y2: 110),
            .point(x: 400, y: 260, isPositive: true),
            .point(x: 12, y: 18, isPositive: false)
        ]

        var segment = StudioDraft()
        segment.reset(for: .segment)
        segment.inputPath = root.appendingPathComponent("mug.png").path
        segment.visionRegionPrompts = [box, point, negative]
        let segmentRequest = try StudioCommandAdapter.makeRequest(mode: .segment, draft: segment)
        let segmentItem = library.start(request: segmentRequest, commandPreview: "fixture")

        var track = StudioDraft()
        track.reset(for: .track)
        track.inputPath = root.appendingPathComponent("clip.mp4").path
        track.visionRegionPrompts = [box, point]
        track.visionInitFrame = 12
        track.visionEndFrame = 40
        let trackRequest = try StudioCommandAdapter.makeRequest(mode: .track, draft: track)
        let trackItem = library.start(request: trackRequest, commandPreview: "fixture")

        // Restored from another task, then opened: the route through `activate`.
        activate(.chat)
        XCTAssertTrue(prompt.useSettings(from: segmentItem))
        activate(.segment, selected: segmentItem.id)
        XCTAssertEqual(prompt.draft.inputPath, segment.inputPath)
        XCTAssertEqual(prompt.draft.visionRegionPrompts?.map(\.shape), expectedShapes)

        XCTAssertTrue(prompt.useSettings(from: trackItem))
        activate(.track, selected: trackItem.id)
        XCTAssertEqual(prompt.draft.inputPath, track.inputPath)
        XCTAssertEqual(prompt.draft.visionRegionPrompts?.map(\.shape), Array(expectedShapes.prefix(2)))
        XCTAssertEqual(prompt.draft.visionInitFrame, 12)
        XCTAssertEqual(prompt.draft.visionEndFrame, 40)

        // Restored into the open task, after the picture was replaced and the drawing lost.
        prompt.draft.replaceInput(root.appendingPathComponent("other.mp4").path)
        XCTAssertNil(prompt.draft.visionRegionPrompts)
        XCTAssertTrue(prompt.useSettings(from: trackItem))
        XCTAssertEqual(prompt.draft.inputPath, track.inputPath)
        XCTAssertEqual(prompt.draft.visionRegionPrompts?.map(\.shape), Array(expectedShapes.prefix(2)))
        XCTAssertEqual(prompt.draft.visionInitFrame, 12)
        XCTAssertEqual(prompt.draft.visionEndFrame, 40)
        XCTAssertTrue(runner.starts.isEmpty)
    }

    /// A Console run of a template the composer does not build (an upscale, an edit) records a
    /// command but has no composer to land in, so the action is not offered rather than failing.
    func testUseTheseSettingsIsOnlyOfferedForCommandsTheComposerBuilds() throws {
        let generate = try XCTUnwrap(CommandCatalog.template(id: .imageGenerate))
        let generated = library.start(
            request: StudioRunRequest(mode: .createImage, templateID: .imageGenerate, template: generate, draft: generate.defaultDraft()),
            commandPreview: "fixture"
        )
        XCTAssertTrue(StudioLibraryDraftRestoration.canRestore(generated))

        let otherTemplate = try XCTUnwrap(CommandCatalog.templates.first { $0.libraryMode == .createImage && $0.id != .imageGenerate })
        let other = library.start(
            request: StudioRunRequest(mode: .createImage, templateID: otherTemplate.id, template: otherTemplate, draft: otherTemplate.defaultDraft()),
            commandPreview: "fixture"
        )
        XCTAssertFalse(StudioLibraryDraftRestoration.canRestore(other), otherTemplate.id.rawValue)
        XCTAssertNil(StudioLibraryDraftRestoration.draft(from: other, baseline: StudioDraft()))
        XCTAssertFalse(prompt.useSettings(from: other))
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
        let original = try XCTUnwrap(library.items.first { $0.id == id })
        let stored = try Data(contentsOf: library.libraryURL)
        let draft = prompt.draft
        for value in ["invalid", "0", "-1", "2147483648"] {
            override(base, flag: "--max-tokens", value: value)
            XCTAssertThrowsError(try prompt.retryLastTurn(inventory: []), value)
            XCTAssertEqual(library.items.first { $0.id == id }, original)
            XCTAssertEqual(try Data(contentsOf: library.libraryURL), stored)
            XCTAssertEqual(prompt.draft, draft)
            XCTAssertEqual(prompt.activeConversationID, id)
            XCTAssertTrue(controller.jobs.all.isEmpty)
            XCTAssertTrue(runner.starts.isEmpty)
        }
    }

    func testRejectedCodeRetryPreservesReplyDraftSelectionAndStoredTranscript() throws {
        let id = thread(mode: .code, reply: "Keep this function")
        activate(.code, selected: id)
        prompt.draft.prompt = "Unsent follow-up"
        let base = try StudioCommandAdapter.makeRequest(mode: .code, draft: prompt.draft)
        XCTAssertEqual(base.templateID, .textCode)
        let original = try XCTUnwrap(library.items.first { $0.id == id })
        let stored = try Data(contentsOf: library.libraryURL)
        let draft = prompt.draft
        for value in ["invalid", "0", "-1", "2147483648"] {
            override(base, flag: "--max-tokens", value: value)
            XCTAssertThrowsError(try prompt.retryLastTurn(inventory: []), value)
            XCTAssertEqual(library.items.first { $0.id == id }, original)
            XCTAssertEqual(try Data(contentsOf: library.libraryURL), stored)
            XCTAssertEqual(prompt.draft, draft)
            XCTAssertEqual(prompt.activeConversationID, id)
            XCTAssertTrue(controller.jobs.all.isEmpty)
            XCTAssertTrue(runner.starts.isEmpty)
        }
    }

    func testCodeRetryUsesTheSameCommandResolutionAsSendWithoutConsumingUnsentText() throws {
        let id = thread(mode: .code, reply: "Replace this function")
        activate(.code, selected: id)
        prompt.draft.prompt = "Do not send this follow-up yet"
        let base = try StudioCommandAdapter.makeRequest(mode: .code, draft: prompt.draft)
        override(base, flag: "--max-tokens", value: "128")
        let originalDraft = prompt.draft
        let request = try XCTUnwrap(prompt.retryLastTurn(inventory: []))
        XCTAssertEqual(request.templateID, .textCode)
        XCTAssertEqual(request.draft.maxTokens, 128)
        XCTAssertEqual(request.conversationID, id)
        XCTAssertTrue(request.draft.prompt.contains("First question"))
        XCTAssertFalse(request.draft.prompt.contains(originalDraft.prompt))
        XCTAssertEqual(prompt.draft, originalDraft)
        XCTAssertEqual(library.items.first?.messages?.count, 1)
        XCTAssertEqual(runner.starts.count, 1)
    }

    func testRetryBlockedByReadinessPreservesReplyAndDraftLikeSend() throws {
        for mode in [StudioMode.chat, .code] {
            let id = thread(mode: mode, reply: "Keep this answer")
            activate(mode, selected: id)
            prompt.draft.prompt = "Unsent follow-up"
            let original = try XCTUnwrap(library.items.first { $0.id == id })
            let draft = prompt.draft
            for state in [ModelReadinessState.missingModel("Install the model first."),
                          .unsupported("This model cannot run here."), .checking] {
                controller.readinessByMode[mode] = state
                XCTAssertThrowsError(try prompt.retryLastTurn(inventory: []), "\(mode) \(state)") { error in
                    XCTAssertEqual(error.localizedDescription, state.message(titles: .none))
                }
                XCTAssertEqual(library.items.first { $0.id == id }, original)
                XCTAssertEqual(prompt.draft, draft)
                XCTAssertEqual(prompt.activeConversationID, id)
                XCTAssertTrue(runner.starts.isEmpty)
            }
        }
    }

    func testSendRejectsInvalidTokenBudgetBeforeCreatingHistory() throws {
        activate(.chat)
        prompt.draft.prompt = "Keep this unsent question"
        for tokens in [0, -1, 129] {
            prompt.draft.maxTokens = tokens
            prompt.draft.contextSize = 128
            let original = prompt.draft
            XCTAssertThrowsError(try prompt.runPrompt(inventory: []))
            XCTAssertEqual(prompt.draft, original)
            XCTAssertTrue(library.items.isEmpty)
            XCTAssertTrue(runner.starts.isEmpty)
        }
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

    func testStoppedReplyRetainsCancellationAfterReloadAndBranching() async throws {
        activate(.chat)
        prompt.draft.prompt = "Keep the partial reply"
        let submission = try XCTUnwrap(prompt.runPrompt(inventory: []))
        let conversationID = try XCTUnwrap(submission.request.conversationID)
        runner.starts[0].stdout("Partial reply")
        prompt.stop(task: .chatChat)
        runner.starts[0].termination(15)
        for _ in 0..<100 where controller.runningConversationIDs.contains(conversationID) {
            try await Task.sleep(for: .milliseconds(5))
        }

        let restored = StudioLibraryStore(libraryURL: library.libraryURL)
        let item = try XCTUnwrap(restored.items.first { $0.id == conversationID })
        let reply = try XCTUnwrap(item.messages?.last)
        XCTAssertEqual(item.status, .cancelled)
        XCTAssertEqual(reply.content, "Partial reply")
        XCTAssertEqual(reply.cancelled, true)
        XCTAssertTrue(reply.failed, "Keep the legacy nonzero-exit flag for older readers.")
        let branch = try XCTUnwrap(restored.branch(conversationID: conversationID, at: reply.id, inclusive: true))
        XCTAssertEqual(branch.status, .cancelled)
        XCTAssertEqual(branch.messages?.last?.cancelled, true)
        XCTAssertEqual(branch.messages?.last?.content, "Partial reply")
        XCTAssertEqual(restored.items.first { $0.id == conversationID }, item)
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
