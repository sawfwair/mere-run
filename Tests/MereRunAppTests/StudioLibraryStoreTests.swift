@testable import MereRunApp
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
        XCTAssertEqual(item.commandDraft, draft)
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
