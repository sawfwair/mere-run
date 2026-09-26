import Foundation
import StudioTestSupport
@testable import StudioKit
import XCTest

/// Compare's rules without a view: which rows compare together, which file each pane shows, what
/// each pane lists from its recorded argv, where a stored comparison lives, and how the shared
/// audio and video transport hands playback between panes.
@MainActor
final class StudioCompareTests: XCTestCase {
    private var root: URL!
    private var library: StudioLibraryStore!

    override func setUp() async throws {
        try await MainActor.run {
            root = FileManager.default.temporaryDirectory.appendingPathComponent("compare-\(UUID())")
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            library = StudioLibraryStore(libraryURL: root.appendingPathComponent("library.json"))
        }
    }

    override func tearDown() async throws {
        try await MainActor.run {
            library = nil
            try FileManager.default.removeItem(at: root)
        }
    }

    /// A finished Image ▸ Generate row whose recorded argv carries `seed`, `steps`, and `model`.
    private func image(
        seed: String = "42", steps: Int = 8, model: String = "", prompt: String = "A ceramic bowl",
        file: String = "bowl.png", group: UUID? = nil
    ) throws -> StudioLibraryItem {
        let template = try XCTUnwrap(CommandCatalog.template(id: .imageGenerate))
        var draft = template.defaultDraft()
        draft.prompt = prompt
        draft.seed = seed
        draft.steps = steps
        draft.model = model
        draft.outputPath = root.appendingPathComponent(file).path
        let row = library.start(request: StudioRunRequest(mode: .createImage, templateID: template.id,
            template: template, draft: draft), commandPreview: "fixture", source: .contract)
        library.complete(id: row.id, exitCode: 0, outputURL: URL(fileURLWithPath: draft.outputPath),
                         outputText: nil, commandPreview: "fixture")
        if let group { library.assignVariationGroup(group, to: [row.id]) }
        return try XCTUnwrap(library.items.first { $0.id == row.id })
    }

    private func sound(file: String = "hit.wav") throws -> StudioLibraryItem {
        let template = try XCTUnwrap(CommandCatalog.template(id: .sfxGenerate))
        var draft = template.defaultDraft()
        draft.prompt = "A door slam"
        draft.outputPath = root.appendingPathComponent(file).path
        let row = library.start(request: StudioRunRequest(mode: .sfx, templateID: template.id,
            template: template, draft: draft), commandPreview: "fixture", source: .contract)
        library.complete(id: row.id, exitCode: 0, outputURL: URL(fileURLWithPath: draft.outputPath),
                         outputText: nil, commandPreview: "fixture")
        return try XCTUnwrap(library.items.first { $0.id == row.id })
    }

    // MARK: Picking

    func testTwoToFourFinishedResultsOfOneKindCompare() throws {
        let a = try image(file: "a.png"), b = try image(file: "b.png"), c = try image(file: "c.png")
        let d = try image(file: "d.png"), e = try image(file: "e.png")
        XCTAssertNil(StudioCompare.comparableMedia([a]))
        XCTAssertEqual(StudioCompare.comparableMedia([a, b]), .image)
        XCTAssertEqual(StudioCompare.comparableMedia([a, b, c, d]), .image)
        XCTAssertNil(StudioCompare.comparableMedia([a, b, c, d, e]), "a hand-picked set stops at four")
        XCTAssertEqual(StudioCompare.unavailableReason([a, b, c, d, e]), "Compare takes up to 4 results")

        let hit = try sound()
        XCTAssertNil(StudioCompare.comparableMedia([a, hit]), "a picture never compares with a sound")
        XCTAssertEqual(StudioCompare.unavailableReason([a, hit]), "Compare results of one kind: images, sounds, or videos")
        XCTAssertEqual(StudioCompare.comparableMedia([hit, try sound(file: "slam.wav")]), .audio)

        var running = try image(file: "f.png")
        running.status = .running
        XCTAssertNil(StudioCompare.comparableMedia([a, running]), "only finished runs compare")
        XCTAssertNil(StudioCompare.media(of: running))
    }

    func testAPaneShowsTheRowsFileOfTheComparedKind() throws {
        var item = try image(file: "bowl.png")
        let sidecar = root.appendingPathComponent("bowl.json")
        item.artifactURLs = [sidecar, item.outputURL].compactMap { $0 }
        item.outputURL = sidecar
        XCTAssertEqual(StudioCompare.media(of: item), .image, "a structured-prompt sidecar is not what is compared")
        XCTAssertEqual(StudioCompare.url(of: item, as: .image)?.lastPathComponent, "bowl.png")
    }

    func testAVariationGroupOpensWithItsFinishedRunsInSubmissionOrder() throws {
        let group = UUID()
        let first = try image(seed: "1", file: "1.png", group: group)
        let second = try image(seed: "2", file: "2.png", group: group)
        XCTAssertEqual(StudioCompare.groupItems(group, in: library.items).map(\.id), [first.id, second.id])

        let third = try image(seed: "3", file: "3.png", group: group)
        library.setStatus(.running, id: third.id)
        XCTAssertEqual(StudioCompare.groupItems(group, in: library.items).count, 2, "a run still in flight waits")
        _ = try image(file: "other.png")
        XCTAssertEqual(StudioCompare.groupItems(group, in: library.items).map(\.id), [first.id, second.id])

        let lone = UUID()
        _ = try image(file: "lone.png", group: lone)
        XCTAssertTrue(StudioCompare.groupItems(lone, in: library.items).isEmpty, "one finished run is nothing to compare")
    }

    func testAStoredComparisonLivesPerTaskAndClosesWhenARowGoes() throws {
        let sessions = StudioTaskSessions(url: root.appendingPathComponent("sessions.json"))
        let a = try image(file: "a.png"), b = try image(file: "b.png"), c = try image(file: "c.png")
        sessions.setFocus(StudioResultSelection(itemID: a.id, url: try XCTUnwrap(a.outputURL)), for: .imageGenerate)

        sessions.setComparison([c, a, b], for: .imageGenerate)

        XCTAssertEqual(sessions.comparison(for: .imageGenerate, items: library.items)?.map(\.id), [c.id, a.id, b.id])
        XCTAssertNil(sessions.focusedResult(for: .imageGenerate, items: library.items), "Compare replaces a focused result")
        XCTAssertNil(sessions.comparison(for: .videoGenerate, items: library.items))
        sessions.setFocus(nil, for: .imageGenerate)
        XCTAssertNil(sessions.comparison(for: .imageGenerate, items: library.items),
                     "returning the page to its composer (Use these settings, Send to) closes Compare")
        sessions.setComparison([c, a, b], for: .imageGenerate)
        library.delete(ids: [a.id, b.id], trashingFiles: false)
        XCTAssertNil(sessions.comparison(for: .imageGenerate, items: library.items), "one row left is not a comparison")
        sessions.flush()
    }

    func testCompareOpensOnAPageThatDrawsItAndNamesAFileThatIsGone() throws {
        let a = try image(file: "a.png"), b = try image(file: "b.png")
        XCTAssertEqual(StudioCompare.hostTask(for: a), .imageGenerate)
        var trained = a
        trained.templateID = .imageTrainLoRA
        XCTAssertEqual(StudioCompare.hostTask(for: trained), trained.mode.task, "a Train page draws no canvas to replace")

        XCTAssertEqual(StudioCompare.missingFile(in: [a, b])?.lastPathComponent, "a.png")
        try Data().write(to: try XCTUnwrap(a.outputURL))
        XCTAssertEqual(StudioCompare.missingFile(in: [a, b])?.lastPathComponent, "b.png")
        try Data().write(to: try XCTUnwrap(b.outputURL))
        XCTAssertNil(StudioCompare.missingFile(in: [a, b]))
    }

    // MARK: Settings

    func testEachPaneListsItsSeedAndModelAndOnlyTheOptionsThatDiffer() throws {
        let group = UUID()
        let a = try image(seed: "101", steps: 8, file: "a.png", group: group)
        let b = try image(seed: "202", steps: 8, file: "b.png", group: group)
        let c = try image(seed: "303", steps: 12, file: "c.png", group: group)

        let panes = StudioCompare.panes(for: [a, b, c], source: .contract)

        XCTAssertEqual(panes.map(\.letter), ["A", "B", "C"])
        XCTAssertEqual(panes.map(\.url.lastPathComponent), ["a.png", "b.png", "c.png"])
        XCTAssertEqual(panes[0].settings.map(\.id), ["--seed", "--model", "--steps"],
                       "seed and model always; the prompt, size, and destination match or never show")
        XCTAssertEqual(panes.map { $0.settings.first { $0.kind == .seed }?.value }, ["101", "202", "303"])
        XCTAssertEqual(panes.map { $0.settings.first { $0.id == "--steps" }?.value }, ["8", "8", "12"])
        XCTAssertEqual(panes[0].settings.first { $0.kind == .model }?.value, a.recordedModelID ?? "Default")
    }

    func testDifferentPromptsAndModelsAreListedFromTheRecordedArgv() throws {
        let a = try image(model: "z-image-turbo", prompt: "A ceramic bowl", file: "a.png")
        var b = try image(model: "flux2-klein-4b", prompt: "A glass bowl", file: "b.png")
        // The row's recorded argv wins over its draft: what ran is what is compared.
        b.commandDraft?.seed = "999"

        let settings = StudioCompare.settings(for: [a, b], source: .contract)

        XCTAssertEqual(settings[0].first { $0.kind == .model }?.value, "z-image-turbo")
        XCTAssertEqual(settings[1].first { $0.kind == .model }?.value, "flux2-klein-4b")
        XCTAssertEqual(settings[1].first { $0.kind == .seed }?.value, "42")
        XCTAssertEqual(settings[1].map(\.id), ["--seed", "--model", "--prompt"], "only the prompt differs besides the model")
        XCTAssertEqual(settings[0].last?.value, "A ceramic bowl")
        XCTAssertEqual(settings[1].last?.value, "A glass bowl")
        XCTAssertFalse(settings.joined().contains { $0.id == "--output" }, "where a run wrote is never a setting")
    }

    func testARunWithoutASeedReadsAsRandom() throws {
        let a = try image(seed: "", file: "a.png"), b = try image(seed: "7", file: "b.png")
        let settings = StudioCompare.settings(for: [a, b], source: .contract)
        XCTAssertEqual(settings.map { $0.first { $0.kind == .seed }?.value }, ["Random", "7"])
    }

    // MARK: Layout

    func testTheGridKeepsPanesSideBySideWhileEachStaysUsable() {
        XCTAssertEqual(StudioCompare.columns(count: 2, width: 1_200), 2)
        XCTAssertEqual(StudioCompare.columns(count: 3, width: 1_200), 3)
        XCTAssertEqual(StudioCompare.columns(count: 4, width: 1_400), 4)
        XCTAssertEqual(StudioCompare.columns(count: 4, width: 900), 2)
        XCTAssertEqual(StudioCompare.columns(count: 8, width: 1_400), 4)
        XCTAssertEqual(StudioCompare.columns(count: 3, width: 500), 1)
    }

    // MARK: Transport

    func testSwitchingSidesKeepsThePositionAndClampsToAShorterPane() {
        var transport = StudioCompareTransport(durations: [8, 5, 10])
        transport.seek(to: 3.5)
        transport.select(2)
        XCTAssertEqual(transport.active, 2)
        XCTAssertEqual(transport.position, 3.5, "A/B listening compares the same moment")
        transport.seek(to: 7)
        transport.select(1)
        XCTAssertEqual(transport.position, 5, "a shorter side picks up at its end")
        XCTAssertEqual(transport.progress(of: 1), 1)
        XCTAssertEqual(transport.progress(of: 2), 0.5)
        XCTAssertEqual(transport.span, 10)
    }

    func testSeekingOnAPanesWaveformMakesItTheOneHeard() {
        var transport = StudioCompareTransport(durations: [8, 4])
        transport.seek(fraction: 0.5, in: 1)
        XCTAssertEqual(transport.active, 1)
        XCTAssertEqual(transport.position, 2)
        transport.isPlaying = true
        transport.advance(to: 3, stillPlaying: true)
        XCTAssertEqual(transport.position, 3)
        transport.advance(to: 4, stillPlaying: false)
        XCTAssertFalse(transport.isPlaying)
        XCTAssertEqual(transport.position, 0, "the end of the heard side stops at the start")
    }
}
