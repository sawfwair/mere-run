@testable import StudioKit
import XCTest

@MainActor
final class StudioLibraryStoreTests: XCTestCase {
    func testLibraryPersistsAndReloadsItems() throws {
        let url = try temporaryLibraryURL()
        let store = StudioLibraryStore(libraryURL: url)
        let item = StudioLibraryItem(
            id: UUID(),
            mode: .createImage,
            prompt: "a small brass lamp",
            inputURL: nil,
            outputURL: URL(fileURLWithPath: "/tmp/lamp.png"),
            createdAt: Date(),
            updatedAt: Date(),
            status: .completed,
            exitCode: 0,
            commandPreview: "mere.run image generate",
            outputText: nil
        )

        store.upsert(item)

        let reloaded = StudioLibraryStore(libraryURL: url)
        XCTAssertEqual(reloaded.items.count, 1)
        XCTAssertEqual(reloaded.items.first?.id, item.id)
        XCTAssertEqual(reloaded.items.first?.mode, .createImage)
        XCTAssertEqual(reloaded.items.first?.prompt, "a small brass lamp")
        XCTAssertEqual(reloaded.items.first?.outputURL?.path, "/tmp/lamp.png")
        XCTAssertEqual(reloaded.items.first?.status, .completed)
        XCTAssertEqual(reloaded.items.first?.exitCode, 0)
        XCTAssertEqual(reloaded.items.first?.commandPreview, "mere.run image generate")
    }

    func testLibraryCompletionUpdatesRunningItem() throws {
        let url = try temporaryLibraryURL()
        let store = StudioLibraryStore(libraryURL: url)
        let request = try StudioCommandAdapter.makeRequest(
            mode: .createImage,
            draft: {
                var draft = StudioDraft()
                draft.reset(for: .createImage)
                draft.prompt = "a blue plate"
                return draft
            }()
        )

        store.start(request: request, commandPreview: "preview")
        store.complete(
            id: request.id,
            exitCode: 0,
            outputURL: URL(fileURLWithPath: "/tmp/plate.png"),
            outputText: nil,
            commandPreview: "preview"
        )

        XCTAssertEqual(store.items.first?.status, .completed)
        XCTAssertEqual(store.items.first?.exitCode, 0)
        XCTAssertEqual(store.items.first?.outputURL?.path, "/tmp/plate.png")
    }

    func testAdvancedRunPersistsTypedDraftAndAllArtifacts() throws {
        let url = try temporaryLibraryURL()
        let store = StudioLibraryStore(libraryURL: url)
        let template = try XCTUnwrap(CommandCatalog.template(id: .musicTrainAdapter))
        var draft = template.defaultDraft()
        draft.inputPath = "/tmp/music-dataset.jsonl"
        draft.outputPath = "/tmp/my-style.safetensors"
        let request = StudioRunRequest(
            mode: template.libraryMode,
            templateID: template.id,
            template: template,
            draft: draft
        )

        store.start(request: request, commandPreview: "mere.run music train-adapter")
        store.complete(
            id: request.id,
            exitCode: 0,
            outputURL: URL(fileURLWithPath: draft.outputPath),
            outputText: nil,
            commandPreview: "mere.run music train-adapter",
            artifactURLs: [
                URL(fileURLWithPath: draft.outputPath),
                URL(fileURLWithPath: "/tmp/my-style.loss.csv"),
            ]
        )

        let reloaded = StudioLibraryStore(libraryURL: url)
        let item = try XCTUnwrap(reloaded.items.first)
        XCTAssertEqual(item.templateID, .musicTrainAdapter)
        XCTAssertEqual(item.commandDraft, draft.withoutSecrets)
        XCTAssertEqual(item.displayKindTitle, "Train music adapter")
        XCTAssertEqual(item.allArtifactURLs.count, 2)
    }

    func testRunningLibraryItemStartsWithoutOutputAndCanPublishOutput() throws {
        let url = try temporaryLibraryURL()
        let store = StudioLibraryStore(libraryURL: url)
        let request = try StudioCommandAdapter.makeRequest(
            mode: .createImage,
            draft: {
                var draft = StudioDraft()
                draft.reset(for: .createImage)
                draft.prompt = "a blue plate"
                return draft
            }()
        )

        store.start(request: request, commandPreview: "preview")
        XCTAssertNil(store.items.first?.outputURL)

        store.updateOutput(id: request.id, outputURL: URL(fileURLWithPath: "/tmp/plate.png"))

        XCTAssertEqual(store.items.first?.status, .running)
        XCTAssertEqual(store.items.first?.outputURL?.path, "/tmp/plate.png")
    }

    func testLibraryCompletionPersistsStdoutOnlyRuns() throws {
        let url = try temporaryLibraryURL()
        let store = StudioLibraryStore(libraryURL: url)
        let request = try StudioCommandAdapter.makeRequest(
            mode: .chat,
            draft: {
                var draft = StudioDraft()
                draft.reset(for: .chat)
                draft.prompt = "Hello"
                return draft
            }()
        )

        store.start(request: request, commandPreview: "preview")
        store.complete(
            id: request.id,
            exitCode: 0,
            outputURL: nil,
            outputText: "Hi from local chat.",
            commandPreview: "preview"
        )

        let item = try XCTUnwrap(store.items.first)
        XCTAssertEqual(item.status, .completed)
        XCTAssertNil(item.outputURL)
        XCTAssertEqual(item.outputText, "Hi from local chat.")

        let reloaded = StudioLibraryStore(libraryURL: url)
        XCTAssertEqual(reloaded.items.first?.outputText, "Hi from local chat.")
    }

    func testCorruptLibraryRecoversToEmptyList() throws {
        let url = try temporaryLibraryURL()
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try Data("not json".utf8).write(to: url)

        let store = StudioLibraryStore(libraryURL: url)

        XCTAssertTrue(store.items.isEmpty)
        XCTAssertFalse(FileManager.default.fileExists(atPath: url.path))
    }

    func testConversationAppendsBuildOrderedThreadAndPersist() throws {
        let url = try temporaryLibraryURL()
        let store = StudioLibraryStore(libraryURL: url)
        let conversationID = UUID()

        store.appendUser(
            conversationID: conversationID, mode: .chat, model: "text-chat-gemma4",
            systemPrompt: "You are helpful.", content: "hi"
        )
        store.appendAssistant(conversationID: conversationID, content: "hello", exitCode: 0)
        store.appendUser(
            conversationID: conversationID, mode: .chat, model: "text-chat-gemma4",
            systemPrompt: "You are helpful.", content: "and now?"
        )

        let reloaded = StudioLibraryStore(libraryURL: url)
        XCTAssertEqual(reloaded.items.count, 1)
        let item = try XCTUnwrap(reloaded.items.first)
        XCTAssertEqual(item.messages?.map(\.role), [.user, .assistant, .user])
        XCTAssertEqual(item.messages?.map(\.content), ["hi", "hello", "and now?"])
        XCTAssertEqual(item.systemPrompt, "You are helpful.")
        XCTAssertEqual(item.model, "text-chat-gemma4")
        XCTAssertEqual(item.displayTitle, "hi")
    }

    func testFailedAssistantTurnIsMarkedButThreadKept() throws {
        let url = try temporaryLibraryURL()
        let store = StudioLibraryStore(libraryURL: url)
        let conversationID = UUID()
        store.appendUser(
            conversationID: conversationID, mode: .chat, model: nil, systemPrompt: nil, content: "hi"
        )
        store.appendAssistant(conversationID: conversationID, content: "boom", exitCode: 1)

        XCTAssertEqual(store.items.count, 1)
        let item = try XCTUnwrap(store.items.first)
        XCTAssertEqual(item.status, .failed)
        XCTAssertEqual(item.messages?.last?.failed, true)
    }

    func testAppendAssistantOnUnknownConversationIsNoOp() throws {
        let url = try temporaryLibraryURL()
        let store = StudioLibraryStore(libraryURL: url)
        store.appendAssistant(conversationID: UUID(), content: "orphan", exitCode: 0)
        XCTAssertTrue(store.items.isEmpty)
    }

    func testDropLastAssistantEnablesRetry() throws {
        let url = try temporaryLibraryURL()
        let store = StudioLibraryStore(libraryURL: url)
        let conversationID = UUID()
        store.appendUser(
            conversationID: conversationID, mode: .chat, model: nil, systemPrompt: nil, content: "hi"
        )
        store.appendAssistant(conversationID: conversationID, content: "wrong", exitCode: 0)
        store.dropLastAssistant(conversationID: conversationID)

        XCTAssertEqual(store.items.first?.messages?.map(\.role), [.user])
    }

    func testTruncateRemovesFromMessageAndReturnsUserContent() throws {
        let url = try temporaryLibraryURL()
        let store = StudioLibraryStore(libraryURL: url)
        let conversationID = UUID()
        store.appendUser(conversationID: conversationID, mode: .chat, model: nil, systemPrompt: nil, content: "one")
        store.appendAssistant(conversationID: conversationID, content: "a1", exitCode: 0)
        store.appendUser(conversationID: conversationID, mode: .chat, model: nil, systemPrompt: nil, content: "two")
        store.appendAssistant(conversationID: conversationID, content: "a2", exitCode: 0)

        let messages = try XCTUnwrap(store.items.first?.messages)
        let secondTurnID = try XCTUnwrap(messages.first { $0.content == "two" }?.id)
        let removed = store.truncate(conversationID: conversationID, removingFrom: secondTurnID)

        XCTAssertEqual(removed?.content, "two")
        XCTAssertEqual(store.items.first?.messages?.map(\.content), ["one", "a1"])
    }

    func testAppendUserStoresImageForVisionTurnAndTruncateReturnsIt() throws {
        let url = try temporaryLibraryURL()
        let store = StudioLibraryStore(libraryURL: url)
        let conversationID = UUID()
        store.appendUser(
            conversationID: conversationID, mode: .chat, model: nil, systemPrompt: nil,
            content: "what is this?", imagePath: "/tmp/photo.png"
        )
        let messageID = try XCTUnwrap(store.items.first?.messages?.first?.id)
        let removed = store.truncate(conversationID: conversationID, removingFrom: messageID)
        XCTAssertEqual(removed?.imagePath, "/tmp/photo.png")
        XCTAssertEqual(store.items.first?.messages?.isEmpty, true)
    }

    func testAssistantTurnsRecordTheModelAndSystemPromptTheyRanWith() throws {
        let url = try temporaryLibraryURL()
        let store = StudioLibraryStore(libraryURL: url)
        let conversationID = UUID()

        store.appendUser(
            conversationID: conversationID, mode: .chat, model: "text-chat-qwen3.6-4b",
            systemPrompt: nil, content: "first"
        )
        store.appendAssistant(
            conversationID: conversationID, content: "a1", exitCode: 0,
            model: "text-chat-qwen3.6-4b", systemPrompt: nil, tokensPerSecond: 41.2
        )
        // The model and system prompt change for the NEXT turn; the earlier turn keeps its own.
        store.appendUser(
            conversationID: conversationID, mode: .chat, model: "text-chat-gemma4-12b-4bit",
            systemPrompt: "Be terse.", content: "second"
        )
        store.appendAssistant(
            conversationID: conversationID, content: "a2", exitCode: 0,
            model: "text-chat-gemma4-12b-4bit", systemPrompt: "Be terse."
        )

        let reloaded = StudioLibraryStore(libraryURL: url)
        let item = try XCTUnwrap(reloaded.items.first)
        let assistants = try XCTUnwrap(item.messages?.filter { $0.role == .assistant })
        XCTAssertEqual(assistants.map(\.model), ["text-chat-qwen3.6-4b", "text-chat-gemma4-12b-4bit"])
        XCTAssertEqual(assistants.map(\.systemPrompt), [nil, "Be terse."])
        XCTAssertEqual(assistants.map(\.tokensPerSecond), [41.2, nil])
        // Thread-level values follow the latest turn for compatibility with retry.
        XCTAssertEqual(item.model, "text-chat-gemma4-12b-4bit")
        XCTAssertEqual(item.systemPrompt, "Be terse.")
        XCTAssertEqual(item.messages?.first?.model, "text-chat-qwen3.6-4b")
    }

    func testPresetChangeOnAThreadIsRecordedWhenTheNextTurnIsSent() throws {
        let url = try temporaryLibraryURL()
        let store = StudioLibraryStore(libraryURL: url)
        let conversationID = UUID()
        store.appendUser(conversationID: conversationID, mode: .chat, model: nil, systemPrompt: nil, content: "hi")
        store.appendAssistant(conversationID: conversationID, content: "hello", exitCode: 0)
        XCTAssertEqual(store.items.first?.mode, .chat)

        store.appendUser(conversationID: conversationID, mode: .code, model: "text-code-north-mini", systemPrompt: nil, content: "now code")

        let item = try XCTUnwrap(store.items.first)
        XCTAssertEqual(item.mode, .code)
        XCTAssertEqual(item.commandPreview, "mere.run text code")
        XCTAssertEqual(item.messages?.count, 3)
    }

    func testBranchBeforeUserTurnKeepsEarlierTurnsAndLeavesSourceIntact() throws {
        let url = try temporaryLibraryURL()
        let store = StudioLibraryStore(libraryURL: url)
        let conversationID = UUID()
        store.appendUser(conversationID: conversationID, mode: .code, model: "m", systemPrompt: "s", content: "one")
        store.appendAssistant(conversationID: conversationID, content: "a1", exitCode: 0, model: "m", tokensPerSecond: 12)
        store.appendUser(conversationID: conversationID, mode: .code, model: "m", systemPrompt: "s", content: "two")
        store.appendAssistant(conversationID: conversationID, content: "a2", exitCode: 0, model: "m")

        let source = try XCTUnwrap(store.items.first)
        let secondTurnID = try XCTUnwrap(source.messages?.first { $0.content == "two" }?.id)
        let branch = try XCTUnwrap(store.branch(conversationID: conversationID, at: secondTurnID, inclusive: false))

        XCTAssertNotEqual(branch.id, source.id)
        XCTAssertEqual(branch.messages?.map(\.content), ["one", "a1"])
        XCTAssertEqual(branch.mode, .code)
        XCTAssertEqual(branch.model, "m")
        XCTAssertEqual(branch.systemPrompt, "s")
        XCTAssertEqual(branch.messages?.last?.tokensPerSecond, 12)
        XCTAssertEqual(branch.status, .completed)
        // Copies get their own identities so both threads can be edited independently.
        XCTAssertNotEqual(branch.messages?.first?.id, source.messages?.first?.id)

        let reloaded = StudioLibraryStore(libraryURL: url)
        XCTAssertEqual(reloaded.items.count, 2)
        XCTAssertEqual(reloaded.items.first?.id, branch.id)
        let original = try XCTUnwrap(reloaded.items.first { $0.id == conversationID })
        XCTAssertEqual(original.messages?.map(\.content), ["one", "a1", "two", "a2"])
    }

    func testBranchAfterAssistantTurnIncludesThatReply() throws {
        let url = try temporaryLibraryURL()
        let store = StudioLibraryStore(libraryURL: url)
        let conversationID = UUID()
        store.appendUser(conversationID: conversationID, mode: .chat, model: nil, systemPrompt: nil, content: "one")
        store.appendAssistant(conversationID: conversationID, content: "a1", exitCode: 0)
        store.appendUser(conversationID: conversationID, mode: .chat, model: nil, systemPrompt: nil, content: "two")
        store.appendAssistant(conversationID: conversationID, content: "a2", exitCode: 0)

        let firstReplyID = try XCTUnwrap(store.items.first?.messages?.first { $0.content == "a1" }?.id)
        let branch = try XCTUnwrap(store.branch(conversationID: conversationID, at: firstReplyID, inclusive: true))
        XCTAssertEqual(branch.messages?.map(\.content), ["one", "a1"])

        XCTAssertNil(store.branch(conversationID: UUID(), at: firstReplyID, inclusive: true))
        XCTAssertNil(store.branch(conversationID: conversationID, at: UUID(), inclusive: true))
    }

    func testLegacyThreadRowsDecodeWithoutPerTurnFields() throws {
        let url = try temporaryLibraryURL()
        let json = """
        [{"id":"\(UUID().uuidString)","mode":"chat","prompt":"",\
        "createdAt":"2026-01-01T00:00:00Z","updatedAt":"2026-01-01T00:00:00Z",\
        "status":"completed","commandPreview":"mere.run text chat",\
        "messages":[{"id":"\(UUID().uuidString)","role":"user","content":"hi",\
        "createdAt":"2026-01-01T00:00:00Z","failed":false},\
        {"id":"\(UUID().uuidString)","role":"assistant","content":"hello",\
        "createdAt":"2026-01-01T00:00:01Z","failed":false}],"model":"text-chat-gemma4"}]
        """
        try XCTUnwrap(json.data(using: .utf8)).write(to: url)

        let store = StudioLibraryStore(libraryURL: url)
        let item = try XCTUnwrap(store.items.first)
        XCTAssertEqual(item.messages?.count, 2)
        XCTAssertNil(item.messages?.last?.model)
        XCTAssertNil(item.messages?.last?.tokensPerSecond)
        XCTAssertEqual(item.model, "text-chat-gemma4")
    }

    func testLoadKeepsValidRowsWhenOneIsCorrupt() throws {
        let url = try temporaryLibraryURL()
        let good = """
        {"id":"\(UUID().uuidString)","mode":"chat","prompt":"keep me",\
        "createdAt":"2026-01-01T00:00:00Z","updatedAt":"2026-01-01T00:00:00Z",\
        "status":"completed","commandPreview":"mere.run text chat"}
        """
        let bad = #"{"mode":"chat","prompt":"missing required fields"}"#
        let json = "[\(good),\(bad)]"
        try XCTUnwrap(json.data(using: .utf8)).write(to: url)

        let store = StudioLibraryStore(libraryURL: url)
        XCTAssertEqual(store.items.count, 1)
        XCTAssertEqual(store.items.first?.prompt, "keep me")
        // The file is intact (not moved to corrupt-recovery) since the array itself parsed.
        XCTAssertTrue(FileManager.default.fileExists(atPath: url.path))
    }

    func testRaycastReceiptImportsPersistsAndDeduplicatesArtifact() throws {
        let libraryURL = try temporaryLibraryURL()
        let artifactURL = libraryURL.deletingLastPathComponent().appendingPathComponent("result.png")
        try Data("image".utf8).write(to: artifactURL)
        let receipt = StudioLibraryImportReceipt(
            version: StudioLibraryImportReceipt.currentVersion,
            id: UUID(),
            source: .raycast,
            kind: .image,
            prompt: "a hand-painted lunar greenhouse",
            artifactPath: artifactURL.path,
            createdAt: Date(timeIntervalSince1970: 1_785_593_400)
        )
        let receiptURL = try writeReceipt(receipt, beside: libraryURL)
        let store = StudioLibraryStore(libraryURL: libraryURL)

        let imported = try store.importReceipt(at: receiptURL)
        let duplicate = try store.importReceipt(at: receiptURL)

        XCTAssertEqual(imported, duplicate)
        XCTAssertEqual(store.items.count, 1)
        XCTAssertEqual(imported.id, receipt.id)
        XCTAssertEqual(imported.mode, .createImage)
        XCTAssertEqual(imported.prompt, receipt.prompt)
        XCTAssertEqual(imported.outputURL, artifactURL.standardizedFileURL)
        XCTAssertEqual(imported.status, .completed)
        XCTAssertEqual(imported.exitCode, 0)
        XCTAssertEqual(imported.templateID, .imageGenerate)
        XCTAssertEqual(imported.source, .raycast)
        XCTAssertEqual(imported.commandPreview, "mere.run image generate")

        let reloaded = StudioLibraryStore(libraryURL: libraryURL)
        XCTAssertEqual(reloaded.items.first?.id, imported.id)
        XCTAssertEqual(reloaded.items.first?.outputURL, imported.outputURL)
        XCTAssertEqual(reloaded.items.first?.source, .raycast)
    }

    func testReceiptRejectsArtifactKindMismatch() throws {
        let libraryURL = try temporaryLibraryURL()
        let artifactURL = libraryURL.deletingLastPathComponent().appendingPathComponent("result.wav")
        try Data("audio".utf8).write(to: artifactURL)
        let receipt = StudioLibraryImportReceipt(
            version: StudioLibraryImportReceipt.currentVersion,
            id: UUID(),
            source: .raycast,
            kind: .image,
            prompt: "this claims to be an image",
            artifactPath: artifactURL.path,
            createdAt: Date()
        )
        let receiptURL = try writeReceipt(receipt, beside: libraryURL)
        let store = StudioLibraryStore(libraryURL: libraryURL)

        XCTAssertThrowsError(try store.importReceipt(at: receiptURL)) { error in
            XCTAssertEqual(error as? StudioLibraryImportError, .artifactKindMismatch("image"))
        }
        XCTAssertTrue(store.items.isEmpty)
    }

    func testReceiptIDCannotReplaceAnotherArtifact() throws {
        let libraryURL = try temporaryLibraryURL()
        let root = libraryURL.deletingLastPathComponent()
        let existingURL = root.appendingPathComponent("existing.png")
        let importedURL = root.appendingPathComponent("imported.png")
        try Data("one".utf8).write(to: existingURL)
        try Data("two".utf8).write(to: importedURL)
        let id = UUID()
        let store = StudioLibraryStore(libraryURL: libraryURL)
        store.upsert(StudioLibraryItem(
            id: id,
            mode: .createImage,
            prompt: "existing",
            inputURL: nil,
            outputURL: existingURL,
            createdAt: Date(),
            updatedAt: Date(),
            status: .completed,
            exitCode: 0,
            commandPreview: "mere.run image generate",
            outputText: nil
        ))
        let receipt = StudioLibraryImportReceipt(
            version: StudioLibraryImportReceipt.currentVersion,
            id: id,
            source: .raycast,
            kind: .image,
            prompt: "collision",
            artifactPath: importedURL.path,
            createdAt: Date()
        )
        let receiptURL = try writeReceipt(receipt, beside: libraryURL)

        XCTAssertThrowsError(try store.importReceipt(at: receiptURL)) { error in
            XCTAssertEqual(error as? StudioLibraryImportError, .receiptIDConflict(id))
        }
        XCTAssertEqual(store.items.count, 1)
        XCTAssertEqual(store.items.first?.outputURL, existingURL)
    }

    func testReceiptRejectsUnsupportedVersionAndEmptyPrompt() throws {
        let libraryURL = try temporaryLibraryURL()
        let artifactURL = libraryURL.deletingLastPathComponent().appendingPathComponent("result.png")
        try Data("image".utf8).write(to: artifactURL)
        let store = StudioLibraryStore(libraryURL: libraryURL)

        for (receipt, expected) in [
            (
                StudioLibraryImportReceipt(
                    version: 99, id: UUID(), source: .raycast, kind: .image,
                    prompt: "valid", artifactPath: artifactURL.path, createdAt: Date()
                ),
                StudioLibraryImportError.unsupportedVersion(99)
            ),
            (
                StudioLibraryImportReceipt(
                    version: 1, id: UUID(), source: .raycast, kind: .image,
                    prompt: "   ", artifactPath: artifactURL.path, createdAt: Date()
                ),
                StudioLibraryImportError.emptyPrompt
            ),
        ] {
            let receiptURL = try writeReceipt(receipt, beside: libraryURL)
            XCTAssertThrowsError(try store.importReceipt(at: receiptURL)) { error in
                XCTAssertEqual(error as? StudioLibraryImportError, expected)
            }
        }
        XCTAssertTrue(store.items.isEmpty)
    }

    func testRaycastJSONContractImportsWithoutSwiftEncoder() throws {
        let libraryURL = try temporaryLibraryURL()
        let artifactURL = libraryURL.deletingLastPathComponent().appendingPathComponent("result.wav")
        try Data("audio".utf8).write(to: artifactURL)
        let id = UUID()
        let json = """
        {
          "version": 1,
          "id": "\(id.uuidString.lowercased())",
          "source": "raycast",
          "kind": "speech",
          "prompt": "Welcome aboard.",
          "artifactPath": "\(artifactURL.path)",
          "createdAt": "2026-08-01T18:30:00Z"
        }
        """
        let receiptURL = libraryURL.deletingLastPathComponent().appendingPathComponent("raycast-contract.json")
        try XCTUnwrap(json.data(using: .utf8)).write(to: receiptURL)

        let imported = try StudioLibraryStore(libraryURL: libraryURL).importReceipt(at: receiptURL)

        XCTAssertEqual(imported.id, id)
        XCTAssertEqual(imported.mode, .speak)
        XCTAssertEqual(imported.templateID, .speechSynthesize)
        XCTAssertEqual(imported.source, .raycast)
    }

    func testReceiptBoundsAndArtifactPathAreEnforced() throws {
        let libraryURL = try temporaryLibraryURL()
        let root = libraryURL.deletingLastPathComponent()
        let oversizedURL = root.appendingPathComponent("oversized.json")
        try Data(repeating: 0x20, count: StudioLibraryImportReceipt.maximumByteCount + 1).write(to: oversizedURL)
        let store = StudioLibraryStore(libraryURL: libraryURL)

        XCTAssertThrowsError(try store.importReceipt(at: oversizedURL)) { error in
            XCTAssertEqual(error as? StudioLibraryImportError, .receiptTooLarge)
        }

        let relativeReceipt = StudioLibraryImportReceipt(
            version: 1,
            id: UUID(),
            source: .raycast,
            kind: .image,
            prompt: "relative output",
            artifactPath: "result.png",
            createdAt: Date()
        )
        let relativeReceiptURL = try writeReceipt(relativeReceipt, beside: libraryURL)
        XCTAssertThrowsError(try store.importReceipt(at: relativeReceiptURL)) { error in
            XCTAssertEqual(error as? StudioLibraryImportError, .artifactPathMustBeAbsolute)
        }
    }

    // MARK: - Favorites

    func testFavoritesSurviveAReloadAndUnstarringLeavesTheKeyOut() throws {
        let url = try temporaryLibraryURL()
        let store = StudioLibraryStore(libraryURL: url)
        let item = makeItem(prompt: "a ceramic coffee mug")
        store.upsert(item)

        store.setFavorite(id: item.id, isFavorite: true)
        XCTAssertTrue(StudioLibraryStore(libraryURL: url).items.first?.isStarred == true)

        store.setFavorite(id: item.id, isFavorite: false)
        let reloaded = StudioLibraryStore(libraryURL: url)
        XCTAssertFalse(reloaded.items.first?.isStarred == true)
        // Unstarred rows keep the shape a build without favorites writes and reads.
        let text = try String(contentsOf: url, encoding: .utf8)
        XCTAssertFalse(text.contains("isFavorite"))
    }

    func testFavoritingAnUnknownRowIsANoOp() throws {
        let store = StudioLibraryStore(libraryURL: try temporaryLibraryURL())
        store.setFavorite(id: UUID(), isFavorite: true)
        XCTAssertTrue(store.items.isEmpty)
    }

    // MARK: - Delete

    func testDeleteMovesEveryArtifactToTheTrashOnlyWhenAsked() throws {
        let url = try temporaryLibraryURL()
        let directory = url.deletingLastPathComponent()
        let primary = directory.appendingPathComponent("mug.png")
        let sidecar = directory.appendingPathComponent("mug.json")
        try Data([0]).write(to: primary)
        try Data([0]).write(to: sidecar)

        var trashed: [URL] = []
        let store = StudioLibraryStore(libraryURL: url, trashItem: { trashed.append($0) })
        var kept = makeItem(prompt: "kept")
        kept.outputURL = primary
        var item = makeItem(prompt: "a ceramic coffee mug")
        item.outputURL = primary
        item.artifactURLs = [primary, sidecar]
        store.upsert(kept)
        store.upsert(item)

        store.delete(ids: [kept.id], trashingFiles: false)
        XCTAssertTrue(trashed.isEmpty, "forgetting a row must not touch its files")

        let failures = store.delete(ids: [item.id], trashingFiles: true)
        XCTAssertTrue(failures.isEmpty)
        XCTAssertEqual(Set(trashed), [primary, sidecar], "the primary output and its sidecars go together")
        XCTAssertTrue(store.items.isEmpty)
    }

    func testDeleteReportsFilesItCouldNotTrashButStillForgetsTheRow() throws {
        let url = try temporaryLibraryURL()
        let file = url.deletingLastPathComponent().appendingPathComponent("locked.png")
        try Data([0]).write(to: file)

        struct TrashFailure: Error {}
        let store = StudioLibraryStore(libraryURL: url, trashItem: { _ in throw TrashFailure() })
        var item = makeItem(prompt: "locked")
        item.outputURL = file
        store.upsert(item)

        let failures = store.delete(ids: [item.id], trashingFiles: true)
        XCTAssertEqual(failures, [file])
        XCTAssertTrue(store.items.isEmpty)
    }

    func testDeleteSkipsFilesThatAreAlreadyGone() throws {
        let url = try temporaryLibraryURL()
        var trashed: [URL] = []
        let store = StudioLibraryStore(libraryURL: url, trashItem: { trashed.append($0) })
        var item = makeItem(prompt: "already gone")
        item.outputURL = url.deletingLastPathComponent().appendingPathComponent("missing.png")
        store.upsert(item)

        XCTAssertTrue(store.delete(ids: [item.id], trashingFiles: true).isEmpty)
        XCTAssertTrue(trashed.isEmpty)
    }

    func testBatchDeleteRemovesEveryNamedRowAndLeavesTheRest() throws {
        let store = StudioLibraryStore(libraryURL: try temporaryLibraryURL())
        let first = makeItem(prompt: "one")
        let second = makeItem(prompt: "two")
        let survivor = makeItem(prompt: "three")
        for row in [first, second, survivor] { store.upsert(row) }

        store.delete(ids: [first.id, second.id], trashingFiles: false)
        XCTAssertEqual(store.items.map(\.prompt), ["three"])
    }

    // MARK: - library.json compatibility

    /// A library.json in exactly the shape shipped before favorites existed. It must keep
    /// decoding, row for row, with `isFavorite` reading as nil.
    func testRowsWrittenBeforeFavoritesStillDecode() throws {
        let url = try temporaryLibraryURL()
        try Self.legacyLibraryFixture.write(to: url, atomically: true, encoding: .utf8)

        let store = StudioLibraryStore(libraryURL: url)

        XCTAssertEqual(store.items.count, 3)
        let image = try XCTUnwrap(store.items.first { $0.prompt == "A lighthouse on a basalt shore at dusk" })
        XCTAssertEqual(image.mode, .createImage)
        XCTAssertEqual(image.templateID, .imageGenerate)
        XCTAssertEqual(image.outputURL?.path, "/Users/example/Library/Application Support/MereRun/App Outputs/imageGenerate-20260101-120000.png")
        XCTAssertEqual(image.artifactURLs?.count, 1)
        XCTAssertEqual(image.status, .completed)
        XCTAssertNil(image.isFavorite)
        XCTAssertFalse(image.isStarred)

        let thread = try XCTUnwrap(store.items.first { $0.isConversation })
        XCTAssertEqual(thread.messages?.count, 2)
        XCTAssertEqual(thread.model, "gemma4-e4b")

        let imported = try XCTUnwrap(store.items.first { $0.source == .raycast })
        XCTAssertEqual(imported.customTitle, "Raycast mug")

        // Rewriting must not disturb the rows that were already there.
        store.setFavorite(id: image.id, isFavorite: true)
        let reloaded = StudioLibraryStore(libraryURL: url)
        XCTAssertEqual(reloaded.items.count, 3)
        XCTAssertTrue(reloaded.items.first { $0.id == image.id }?.isStarred == true)
        XCTAssertEqual(reloaded.items.first { $0.isConversation }?.messages?.count, 2)
    }

    /// The reverse direction: a row this build wrote, read by a decoder that does not know the
    /// key, still decodes every field it does know.
    func testFavoriteRowsDecodeWithoutTheNewKey() throws {
        var item = makeItem(prompt: "starred")
        item.isFavorite = true
        let data = try JSONEncoder.mereRunApp.encode([item])

        struct LegacyRow: Decodable {
            let id: UUID
            let mode: String
            let prompt: String
            let status: String
        }
        let rows = try JSONDecoder.mereRunApp.decode([LegacyRow].self, from: data)
        XCTAssertEqual(rows.count, 1)
        XCTAssertEqual(rows.first?.prompt, "starred")
        XCTAssertEqual(rows.first?.mode, "createImage")
        XCTAssertEqual(rows.first?.status, "completed")
    }

    private static let legacyLibraryFixture = """
    [
      {
        "artifactURLs" : [
          "file:///Users/example/Library/Application%20Support/MereRun/App%20Outputs/imageGenerate-20260101-120000.png"
        ],
        "commandPreview" : "mere.run image generate --model image-zimage-nano",
        "createdAt" : "2026-01-01T12:00:00Z",
        "exitCode" : 0,
        "id" : "6C4F2B22-2C1B-4B3E-9A61-9C2E1F5B7A01",
        "mode" : "createImage",
        "outputURL" : "file:///Users/example/Library/Application%20Support/MereRun/App%20Outputs/imageGenerate-20260101-120000.png",
        "prompt" : "A lighthouse on a basalt shore at dusk",
        "status" : "completed",
        "templateID" : "imageGenerate",
        "updatedAt" : "2026-01-01T12:00:04Z"
      },
      {
        "commandPreview" : "mere.run text chat --model gemma4-e4b",
        "createdAt" : "2026-01-01T11:00:00Z",
        "exitCode" : 0,
        "id" : "6C4F2B22-2C1B-4B3E-9A61-9C2E1F5B7A02",
        "messages" : [
          {
            "content" : "Explain a Fresnel lens.",
            "createdAt" : "2026-01-01T11:00:00Z",
            "failed" : false,
            "id" : "6C4F2B22-2C1B-4B3E-9A61-9C2E1F5B7A03",
            "role" : "user"
          },
          {
            "content" : "It folds a thick lens into rings.",
            "createdAt" : "2026-01-01T11:00:09Z",
            "failed" : false,
            "id" : "6C4F2B22-2C1B-4B3E-9A61-9C2E1F5B7A04",
            "model" : "gemma4-e4b",
            "role" : "assistant"
          }
        ],
        "mode" : "chat",
        "model" : "gemma4-e4b",
        "prompt" : "",
        "status" : "completed",
        "updatedAt" : "2026-01-01T11:00:09Z"
      },
      {
        "commandPreview" : "mere.run image generate",
        "createdAt" : "2026-01-01T10:00:00Z",
        "customTitle" : "Raycast mug",
        "exitCode" : 0,
        "id" : "6C4F2B22-2C1B-4B3E-9A61-9C2E1F5B7A05",
        "mode" : "createImage",
        "outputURL" : "file:///Users/example/Pictures/mug.png",
        "prompt" : "a ceramic coffee mug",
        "source" : "raycast",
        "status" : "completed",
        "templateID" : "imageGenerate",
        "updatedAt" : "2026-01-01T10:00:03Z"
      }
    ]
    """

    private func makeItem(prompt: String) -> StudioLibraryItem {
        StudioLibraryItem(
            id: UUID(),
            mode: .createImage,
            prompt: prompt,
            inputURL: nil,
            outputURL: nil,
            createdAt: Date(),
            updatedAt: Date(),
            status: .completed,
            exitCode: 0,
            commandPreview: "mere.run image generate",
            outputText: nil,
            templateID: .imageGenerate
        )
    }

    private func writeReceipt(_ receipt: StudioLibraryImportReceipt, beside libraryURL: URL) throws -> URL {
        let receiptURL = libraryURL.deletingLastPathComponent()
            .appendingPathComponent("receipt-\(UUID().uuidString).json")
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        try encoder.encode(receipt).write(to: receiptURL)
        return receiptURL
    }

    private func temporaryLibraryURL() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("mere-run-app-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root.appendingPathComponent("library.json")
    }
}
