@testable import StudioKit
@testable import StudioUI
import XCTest

final class StudioLibraryPresentationTests: XCTestCase {
    // MARK: - Filtering

    func testScopeNarrowsToTheCurrentDomainAndAllShowsEverything() {
        let items = [
            item(mode: .createImage, templateID: .imageGenerate, output: "mug.png"),
            item(mode: .speak, templateID: .speechSynthesize, output: "line.wav"),
        ]

        XCTAssertEqual(StudioLibraryPresenter.scoped(items, scope: .all, domain: .image).count, 2)
        let scoped = StudioLibraryPresenter.scoped(items, scope: .domain, domain: .image)
        XCTAssertEqual(scoped.map(\.mode), [.createImage])
    }

    func testKindFilterSplitsImagesVideoAudioAndText() {
        let image = item(mode: .createImage, templateID: .imageGenerate, output: "mug.png")
        let clip = item(mode: .video, templateID: .videoGenerate, output: "clip.mp4")
        let song = item(mode: .music, templateID: .musicGenerate, output: "song.wav")
        let document = item(mode: .findObjects, templateID: .visionGround, output: "boxes.json")
        let transcript = item(mode: .listen, templateID: .speechTranscribe, output: nil, outputText: "Welcome aboard.")
        let items = [image, clip, song, document, transcript]

        func matching(_ kind: StudioLibraryKind) -> [UUID] {
            StudioLibraryPresenter.filter(
                items, with: StudioLibraryFilter(scope: .all, kind: kind)
            ).map(\.id)
        }

        XCTAssertEqual(matching(.all).count, 5)
        XCTAssertEqual(matching(.images), [image.id])
        XCTAssertEqual(matching(.video), [clip.id])
        XCTAssertEqual(matching(.audio), [song.id])
        XCTAssertEqual(matching(.text), [document.id, transcript.id])
    }

    func testFavoritesFilterKeepsOnlyStarredRows() {
        var starred = item(mode: .createImage, templateID: .imageGenerate, output: "mug.png")
        starred.isFavorite = true
        let plain = item(mode: .createImage, templateID: .imageGenerate, output: "plate.png")

        let filtered = StudioLibraryPresenter.filter(
            [starred, plain], with: StudioLibraryFilter(scope: .all, favoritesOnly: true)
        )
        XCTAssertEqual(filtered.map(\.id), [starred.id])
        XCTAssertTrue(starred.isStarred)
        XCTAssertFalse(plain.isStarred, "a row with no isFavorite key is not a favorite")
    }

    func testSearchMatchesTitleKindAndPromptAndComposesWithTheOtherFilters() {
        var starred = item(
            mode: .createImage, templateID: .imageGenerate, output: "mug.png",
            prompt: "a ceramic coffee mug"
        )
        starred.isFavorite = true
        let other = item(
            mode: .createImage, templateID: .imageGenerate, output: "diner.png",
            prompt: "a rainy diner window"
        )

        let filter = StudioLibraryFilter(
            scope: .domain, domain: .image, kind: .images, favoritesOnly: true, query: "COFFEE"
        )
        XCTAssertEqual(StudioLibraryPresenter.filter([starred, other], with: filter).map(\.id), [starred.id])
        XCTAssertTrue(
            StudioLibraryPresenter.filter(
                [starred, other],
                with: StudioLibraryFilter(scope: .all, query: "nothing here")
            ).isEmpty
        )
    }

    // MARK: - Multi-selection

    func testPlainClickReplacesTheBatchAndOpensTheRow() {
        let ids = [UUID(), UUID(), UUID()]
        let result = StudioLibrarySelection.click(
            on: ids[2], visible: ids, selection: [ids[0]], anchor: ids[0], modifiers: .none
        )
        XCTAssertEqual(result.selection, [ids[2]])
        XCTAssertEqual(result.anchor, ids[2])
        XCTAssertEqual(result.opened, ids[2])
    }

    func testCommandClickAddsAndRemovesWithoutChangingWhatIsOpen() {
        let ids = [UUID(), UUID(), UUID()]
        let added = StudioLibrarySelection.click(
            on: ids[2], visible: ids, selection: [ids[0]], anchor: ids[0], modifiers: .command
        )
        XCTAssertEqual(added.selection, [ids[0], ids[2]])
        XCTAssertEqual(added.opened, ids[2])

        let removed = StudioLibrarySelection.click(
            on: ids[2], visible: ids, selection: added.selection, anchor: ids[2], modifiers: .command
        )
        XCTAssertEqual(removed.selection, [ids[0]])
        XCTAssertNil(removed.opened, "deselecting a row must not yank the canvas to it")
    }

    func testShiftClickSelectsTheRunOfVisibleRowsInEitherDirection() {
        let ids = (0..<5).map { _ in UUID() }
        let down = StudioLibrarySelection.click(
            on: ids[3], visible: ids, selection: [ids[1]], anchor: ids[1], modifiers: .shift
        )
        XCTAssertEqual(down.selection, Set(ids[1...3]))
        XCTAssertEqual(down.anchor, ids[1], "the anchor stays put so the range can be re-dragged")

        let up = StudioLibrarySelection.click(
            on: ids[0], visible: ids, selection: down.selection, anchor: ids[3], modifiers: .shift
        )
        XCTAssertEqual(up.selection, Set(ids[0...3]))
    }

    func testShiftClickWithNoAnchorBehavesLikeAPlainClick() {
        let ids = [UUID(), UUID()]
        let result = StudioLibrarySelection.click(
            on: ids[1], visible: ids, selection: [], anchor: nil, modifiers: .shift
        )
        XCTAssertEqual(result.selection, [ids[1]])
        XCTAssertEqual(result.opened, ids[1])
    }

    func testModifiersReadTheEventFlags() {
        XCTAssertEqual(StudioLibrarySelection.Modifiers(event: [.command]), .command)
        XCTAssertEqual(StudioLibrarySelection.Modifiers(event: [.shift]), .shift)
        XCTAssertEqual(StudioLibrarySelection.Modifiers(event: [.option]), .none)
    }

    // MARK: - Thumbnail cache

    func testThumbnailCacheKeysSeparatePathSizeAndModificationDate() {
        let url = URL(fileURLWithPath: "/tmp/mere/clip.mp4")
        let stamp = Date(timeIntervalSince1970: 1_700_000_000)
        let key = StudioThumbnailCache.key(url: url, maxPixelSize: 80, modified: stamp)

        XCTAssertEqual(key, "/tmp/mere/clip.mp4|80|1700000000")
        XCTAssertEqual(key, StudioThumbnailCache.key(url: url, maxPixelSize: 80, modified: stamp))
        XCTAssertNotEqual(key, StudioThumbnailCache.key(url: url, maxPixelSize: 160, modified: stamp))
        XCTAssertNotEqual(
            key,
            StudioThumbnailCache.key(url: url, maxPixelSize: 80, modified: stamp.addingTimeInterval(1)),
            "a file rewritten in place must not keep its old thumbnail"
        )
        XCTAssertEqual(
            StudioThumbnailCache.key(url: url, maxPixelSize: 80, modified: nil),
            "/tmp/mere/clip.mp4|80|missing"
        )
    }

    func testThumbnailCacheKeysStandardizeThePath() {
        XCTAssertEqual(
            StudioThumbnailCache.key(url: URL(fileURLWithPath: "/tmp/mere/../mere/clip.mp4"), maxPixelSize: 40, modified: nil),
            StudioThumbnailCache.key(url: URL(fileURLWithPath: "/tmp/mere/clip.mp4"), maxPixelSize: 40, modified: nil)
        )
    }

    @MainActor
    func testThumbnailCacheStoresAndReturnsByKey() {
        let cache = StudioThumbnailCache()
        let image = NSImage(size: NSSize(width: 4, height: 4))
        cache.store(image, forKey: "a")
        cache.store([0.1, 0.9], forKey: "b")

        XCTAssertIdentical(cache.image(forKey: "a"), image)
        XCTAssertNil(cache.image(forKey: "b"))
        XCTAssertEqual(cache.peaks(forKey: "b"), [0.1, 0.9])
    }

    // MARK: - Per-task drafts

    func testDraftMemoryKeepsEachTaskSeparate() {
        var image = StudioDraft()
        image.reset(for: .createImage)
        image.prompt = "a ceramic coffee mug"
        var video = StudioDraft()
        video.reset(for: .video)
        video.prompt = "a slow pan over rooftops"
        video.inputPath = "/tmp/frame.png"

        let encoded = StudioDraftMemory.encode([
            .imageGenerate: StudioDraftMemory.entry(for: image),
            .videoGenerate: StudioDraftMemory.entry(for: video),
        ])
        let decoded = StudioDraftMemory.decode(encoded)

        XCTAssertEqual(decoded[.imageGenerate]?.prompt, "a ceramic coffee mug")
        XCTAssertEqual(decoded[.videoGenerate]?.prompt, "a slow pan over rooftops")
        XCTAssertEqual(decoded[.videoGenerate]?.inputPath, "/tmp/frame.png")
        XCTAssertEqual(decoded[.imageGenerate]?.inputPath, "")
        XCTAssertNil(decoded[.chatChat])
    }

    func testDraftMemoryDropsEmptyEntriesAndUnknownTasks() {
        var empty = StudioDraft()
        empty.reset(for: .findObjects)
        empty.prompt = ""
        XCTAssertEqual(StudioDraftMemory.encode([.visionFind: StudioDraftMemory.entry(for: empty)]), "")
        XCTAssertTrue(StudioDraftMemory.decode("").isEmpty)
        XCTAssertTrue(StudioDraftMemory.decode("not json").isEmpty)
        XCTAssertTrue(
            StudioDraftMemory.decode(#"{"nope.task":{"prompt":"x","secondaryText":"","inputPath":""}}"#).isEmpty
        )
        XCTAssertTrue(
            StudioDraftMemory.decode(#"{"models.installed":{"prompt":"x","secondaryText":"","inputPath":""}}"#).isEmpty,
            "only prompt tasks have a draft to restore"
        )
    }

    func testRestoringADraftPutsBackTheWordsButNotAMissingAttachment() throws {
        let file = FileManager.default.temporaryDirectory
            .appendingPathComponent("StudioDraftMemoryTests-\(UUID().uuidString).png")
        try Data([0]).write(to: file)
        addTeardownBlock { try? FileManager.default.removeItem(at: file) }

        var restored = StudioDraft()
        restored.reset(for: .createImage)
        StudioDraftMemory.apply(
            StudioDraftMemoryEntry(prompt: "a rainy diner", secondaryText: "no text", inputPath: file.path),
            to: &restored
        )
        XCTAssertEqual(restored.prompt, "a rainy diner")
        XCTAssertEqual(restored.secondaryText, "no text")
        XCTAssertEqual(restored.inputPath, file.path)

        var dangling = StudioDraft()
        dangling.reset(for: .createImage)
        StudioDraftMemory.apply(
            StudioDraftMemoryEntry(prompt: "a rainy diner", inputPath: "/tmp/gone-\(UUID().uuidString).png"),
            to: &dangling
        )
        XCTAssertEqual(dangling.inputPath, "", "a file that has since gone is not restored as a chip")
    }

    func testRestoringDoesNotClobberSettingsTheTaskDefaults() {
        var draft = StudioDraft()
        draft.reset(for: .video)
        let defaultWidth = draft.width
        StudioDraftMemory.apply(StudioDraftMemoryEntry(prompt: "rooftops"), to: &draft)
        XCTAssertEqual(draft.width, defaultWidth)
        XCTAssertEqual(draft.prompt, "rooftops")
    }

    // MARK: - Helpers

    private func item(
        mode: StudioMode,
        templateID: CommandTemplateID,
        output: String?,
        outputText: String? = nil,
        prompt: String = "prompt"
    ) -> StudioLibraryItem {
        StudioLibraryItem(
            id: UUID(),
            mode: mode,
            prompt: prompt,
            inputURL: nil,
            outputURL: output.map { URL(fileURLWithPath: "/tmp/\($0)") },
            createdAt: Date(),
            updatedAt: Date(),
            status: .completed,
            exitCode: 0,
            commandPreview: "mere.run",
            outputText: outputText,
            templateID: templateID
        )
    }
}
