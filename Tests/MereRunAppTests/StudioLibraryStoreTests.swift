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
            commandPreview: "mere.run image generate"
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
            commandPreview: "preview"
        )

        XCTAssertEqual(store.items.first?.status, .completed)
        XCTAssertEqual(store.items.first?.exitCode, 0)
        XCTAssertEqual(store.items.first?.outputURL?.path, "/tmp/plate.png")
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

    private func temporaryLibraryURL() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("mere-run-app-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root.appendingPathComponent("library.json")
    }
}
