@testable import MereRunApp
import XCTest

final class StudioThreadListPresenterTests: XCTestCase {
    private let calendar = Calendar(identifier: .gregorian)
    private let noon = Date(timeIntervalSince1970: 1_756_900_800) // 2025-09-03 12:00:00 UTC

    func testThreadsAreTheConversationRowsNewestActivityFirst() {
        let older = thread(title: "older", updatedAgo: 3_600)
        let newer = thread(title: "newer", updatedAgo: 60)
        let image = mediaRow()

        let threads = StudioThreadListPresenter.threads(in: [older, image, newer])

        XCTAssertEqual(threads.map(\.displayTitle), ["newer", "older"])
    }

    func testMediaItemsExcludeThreadsSoTheyNeverFileIntoTheLibrary() {
        let image = mediaRow()
        let items = [thread(title: "chat", updatedAgo: 0), image, thread(title: "code", mode: .code, updatedAgo: 0)]

        XCTAssertEqual(StudioThreadListPresenter.mediaItems(in: items).map(\.id), [image.id])
    }

    func testChatAndCodeThreadsShareOneList() {
        let chat = thread(title: "chat", mode: .chat, updatedAgo: 10)
        let code = thread(title: "code", mode: .code, updatedAgo: 5)

        XCTAssertEqual(StudioThreadListPresenter.threads(in: [chat, code]).map(\.mode), [.code, .chat])
    }

    func testFilterMatchesTitleOrAnyTurn() {
        let diffusion = thread(title: "Summarize diffusion models", updatedAgo: 0, reply: "Noise is predicted.")
        let bytes = thread(title: "Swift function that formats byte counts", updatedAgo: 0)

        XCTAssertEqual(StudioThreadListPresenter.filter([diffusion, bytes], query: "byte").map(\.id), [bytes.id])
        XCTAssertEqual(StudioThreadListPresenter.filter([diffusion, bytes], query: "NOISE").map(\.id), [diffusion.id])
        XCTAssertEqual(StudioThreadListPresenter.filter([diffusion, bytes], query: "  ").count, 2)
    }

    func testSectionsSplitTodayFromEarlierAndOmitEmptyGroups() {
        let today = thread(title: "today", updatedAgo: 3_600, now: noon)
        let yesterday = thread(title: "yesterday", updatedAgo: 86_400, now: noon)
        let lastWeek = thread(title: "week", updatedAgo: 86_400 * 6, now: noon)

        let sections = StudioThreadListPresenter.sections([today, yesterday, lastWeek], now: noon, calendar: calendar)
        XCTAssertEqual(sections.map(\.title), ["Today", "Earlier"])
        XCTAssertEqual(sections[0].threads.map(\.displayTitle), ["today"])
        XCTAssertEqual(sections[1].threads.map(\.displayTitle), ["yesterday", "week"])

        let onlyEarlier = StudioThreadListPresenter.sections([lastWeek], now: noon, calendar: calendar)
        XCTAssertEqual(onlyEarlier.map(\.title), ["Earlier"])
    }

    func testMetaNamesPresetModelAndActivity() {
        let chat = thread(title: "chat", updatedAgo: 3_600, now: noon, model: "text-chat-qwen3.6-4b")
        let chatMeta = StudioThreadListPresenter.meta(for: chat, now: noon, calendar: calendar)
        XCTAssertTrue(chatMeta.hasPrefix("Qwen3.6 4B · "), chatMeta)
        XCTAssertFalse(chatMeta.hasPrefix("Code"))

        let code = thread(title: "code", mode: .code, updatedAgo: 86_400, now: noon, model: "text-code-gemma-4")
        XCTAssertEqual(
            StudioThreadListPresenter.meta(for: code, now: noon, calendar: calendar),
            "Code · Gemma 4 · Yesterday"
        )

        let old = thread(title: "old", updatedAgo: 86_400 * 4, now: noon, model: "text-chat-qwen3.6-4b")
        let oldMeta = StudioThreadListPresenter.meta(for: old, now: noon, calendar: calendar)
        XCTAssertFalse(oldMeta.hasSuffix("Yesterday"), oldMeta)
        XCTAssertTrue(oldMeta.hasPrefix("Qwen3.6 4B · "), oldMeta)
    }

    func testModelLabelFallsBackToThePresetDefault() {
        let untitled = thread(title: "x", mode: .code, updatedAgo: 0, model: nil)
        let expected = StudioModelPicker.resolvedModelID("", mode: .code)
        XCTAssertFalse(expected.isEmpty)
        XCTAssertEqual(
            StudioThreadListPresenter.modelLabel(for: untitled),
            StudioModelPicker.displayModelName(expected)
        )
    }

    func testCodePresetMapsToItsTaskAndBack() {
        XCTAssertEqual(StudioTask.chatCode.mode, .code)
        XCTAssertEqual(StudioMode.code.task, .chatCode)
        XCTAssertEqual(StudioTask.chatCode.domain, .chat)
        XCTAssertEqual(StudioDomain.chat.tasks, [.chatChat, .chatCode, .chatTrain])
    }

    // MARK: Fixtures

    private func thread(
        title: String,
        mode: StudioMode = .chat,
        updatedAgo: TimeInterval,
        now: Date = Date(),
        model: String? = "text-chat-qwen3.6-4b",
        reply: String = "reply"
    ) -> StudioLibraryItem {
        let updated = now.addingTimeInterval(-updatedAgo)
        return StudioLibraryItem(
            id: UUID(),
            mode: mode,
            prompt: "",
            inputURL: nil,
            outputURL: nil,
            createdAt: updated.addingTimeInterval(-30),
            updatedAt: updated,
            status: .completed,
            exitCode: 0,
            commandPreview: "mere.run text \(mode == .code ? "code" : "chat")",
            outputText: nil,
            messages: [
                StudioMessage(role: .user, content: title, createdAt: updated.addingTimeInterval(-30)),
                StudioMessage(role: .assistant, content: reply, createdAt: updated)
            ],
            systemPrompt: nil,
            model: model
        )
    }

    private func mediaRow() -> StudioLibraryItem {
        StudioLibraryItem(
            id: UUID(),
            mode: .createImage,
            prompt: "a mug",
            inputURL: nil,
            outputURL: nil,
            createdAt: Date(),
            updatedAt: Date(),
            status: .completed,
            exitCode: 0,
            commandPreview: "mere.run image generate",
            outputText: nil
        )
    }
}
