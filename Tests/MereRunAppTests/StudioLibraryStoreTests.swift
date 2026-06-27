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

    private func temporaryLibraryURL() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("mere-run-app-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root.appendingPathComponent("library.json")
    }
}
