import Foundation
import StudioTestSupport
@testable import StudioKit
import XCTest

/// Collections, the model and task filters, and provenance between Library rows. Every Library
/// here lives in a temporary folder, `collections.json` included, and undo runs through a real
/// `UndoManager` grouped by event the way the window's is.
@MainActor
final class StudioLibraryCollectionsTests: XCTestCase {
    private var root: URL!
    private var manager: UndoManager!
    private var library: StudioLibraryStore!

    override func setUp() async throws {
        try await MainActor.run {
            root = FileManager.default.temporaryDirectory.appendingPathComponent("studio-collections-\(UUID())")
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            manager = UndoManager()
            library = StudioLibraryStore(libraryURL: libraryURL)
            library.undo.manager = manager
        }
    }

    override func tearDown() async throws {
        try await MainActor.run {
            library = nil
            manager = nil
            try FileManager.default.removeItem(at: root)
        }
    }

    private var libraryURL: URL { root.appendingPathComponent("library.json") }

    /// One user event: everything `body` registers is one undo step.
    private func event(_ body: () -> Void) {
        body()
        RunLoop.current.run(until: Date())
        XCTAssertEqual(manager.groupingLevel, 0)
    }

    private func finished(
        _ prompt: String,
        output: String,
        mode: StudioMode = .createImage,
        templateID: CommandTemplateID = .imageGenerate,
        model: String? = nil,
        at createdAt: Date = Date(),
        input: String? = nil,
        arguments: [String]? = nil
    ) -> StudioLibraryItem {
        var item = StudioLibraryItem(
            id: UUID(), mode: mode, prompt: prompt, inputURL: input.map { URL(fileURLWithPath: $0) },
            outputURL: URL(fileURLWithPath: output), createdAt: createdAt, updatedAt: createdAt,
            status: .completed, exitCode: 0, commandPreview: "mere.run", outputText: nil,
            templateID: templateID, commandArguments: arguments
        )
        item.artifactURLs = [URL(fileURLWithPath: output)]
        item.model = model
        return item
    }

    // MARK: - Collections

    func testCollectionsAreCreatedRenamedAndDeletedAndSurviveAReload() throws {
        let lamp = finished("a brass lamp", output: "/tmp/lamp.png")
        let plate = finished("a blue plate", output: "/tmp/plate.png")
        library.upsert(lamp)
        library.upsert(plate)

        let covers = library.createCollection(named: "  Album covers ", adding: [lamp.id])
        let drafts = library.createCollection(named: "", adding: [lamp.id, plate.id, lamp.id])
        XCTAssertEqual(covers.name, "Album covers")
        XCTAssertEqual(drafts.name, "New collection")
        XCTAssertEqual(library.suggestedCollectionName, "New collection 2")
        XCTAssertEqual(drafts.itemIDs, [lamp.id, plate.id])

        library.renameCollection(id: drafts.id, to: "Kitchen")
        library.renameCollection(id: drafts.id, to: "   ")

        let reloaded = StudioLibraryStore(libraryURL: libraryURL)
        XCTAssertEqual(reloaded.collections.map(\.name), ["Album covers", "Kitchen"])
        XCTAssertEqual(reloaded.collections.last?.itemIDs, [lamp.id, plate.id])
        XCTAssertTrue(FileManager.default.fileExists(atPath: root.appendingPathComponent("collections.json").path))

        reloaded.deleteCollection(id: covers.id)
        XCTAssertEqual(StudioLibraryStore(libraryURL: libraryURL).collections.map(\.name), ["Kitchen"])
    }

    func testARowCanBeInSeveralCollectionsAndDeletingOneKeepsTheRows() {
        let lamp = finished("a brass lamp", output: "/tmp/lamp.png")
        let plate = finished("a blue plate", output: "/tmp/plate.png")
        library.upsert(lamp)
        library.upsert(plate)
        let covers = library.createCollection(named: "Covers", adding: [lamp.id])
        let props = library.createCollection(named: "Props", adding: [lamp.id, plate.id])

        XCTAssertEqual(library.collections(containing: lamp.id).map(\.name), ["Covers", "Props"])
        XCTAssertEqual(library.collections(containing: plate.id).map(\.name), ["Props"])

        library.deleteCollection(id: props.id)

        XCTAssertEqual(library.items.count, 2)
        XCTAssertEqual(library.collections.map(\.id), [covers.id])
        XCTAssertEqual(library.collections(containing: lamp.id).map(\.name), ["Covers"])
        XCTAssertTrue(library.collections(containing: plate.id).isEmpty)
    }

    func testAddingRemovingAndTogglingMembership() {
        let lamp = finished("a brass lamp", output: "/tmp/lamp.png")
        let plate = finished("a blue plate", output: "/tmp/plate.png")
        library.upsert(lamp)
        library.upsert(plate)
        let props = library.createCollection(named: "Props")

        library.addToCollection(id: props.id, itemIDs: [lamp.id])
        library.toggleMembership(collectionID: props.id, itemIDs: [lamp.id, plate.id])
        XCTAssertEqual(library.collections[0].itemIDs, [lamp.id, plate.id])

        library.toggleMembership(collectionID: props.id, itemIDs: [lamp.id, plate.id])
        XCTAssertTrue(library.collections[0].itemIDs.isEmpty)

        library.addToCollection(id: props.id, itemIDs: [plate.id])
        library.removeFromCollection(id: props.id, itemIDs: [lamp.id])
        XCTAssertEqual(library.collections[0].itemIDs, [plate.id])
    }

    func testADeletedRowLeavesTheCountAndComesBackIntoItsCollectionOnUndo() {
        let lamp = finished("a brass lamp", output: "/tmp/lamp.png")
        let plate = finished("a blue plate", output: "/tmp/plate.png")
        library.upsert(lamp)
        library.upsert(plate)
        var props: StudioLibraryCollection!
        event { props = library.createCollection(named: "Props", adding: [lamp.id, plate.id]) }

        event { library.delete(ids: [lamp.id], trashingFiles: false) }
        XCTAssertEqual(library.memberCount(of: library.collections[0]), 1)

        manager.undo()
        XCTAssertEqual(library.memberCount(of: library.collections[0]), 2)
        XCTAssertEqual(library.collections(containing: lamp.id).map(\.id), [props.id])
    }

    func testDroppedFilesJoinTheRunsThatMadeThem() {
        let lamp = finished("a brass lamp", output: "/tmp/stems/lamp.png")
        library.upsert(lamp)

        XCTAssertEqual(library.itemIDs(producing: [URL(fileURLWithPath: "/tmp/stems/../stems/lamp.png")]), [lamp.id])
        XCTAssertTrue(library.itemIDs(producing: [URL(fileURLWithPath: "/tmp/elsewhere.png")]).isEmpty)
    }

    func testAnUnreadableCollectionsFileIsMovedAsideNotOverwritten() throws {
        let url = root.appendingPathComponent("collections.json")
        try Data("{\"version\": \"one\"}".utf8).write(to: url)

        let store = StudioLibraryStore(libraryURL: libraryURL)

        XCTAssertTrue(store.collections.isEmpty)
        XCTAssertFalse(FileManager.default.fileExists(atPath: url.path))
        let aside = try FileManager.default.contentsOfDirectory(atPath: root.path).filter { $0.hasPrefix("collections.corrupt-") }
        XCTAssertEqual(aside.count, 1)
    }

    // MARK: - Undo

    func testCollectionEditsUndoAndRedoAsNamedSteps() throws {
        let lamp = finished("a brass lamp", output: "/tmp/lamp.png")
        let plate = finished("a blue plate", output: "/tmp/plate.png")
        library.upsert(lamp)
        library.upsert(plate)

        var props: StudioLibraryCollection!
        event { props = library.createCollection(named: "Props", adding: [lamp.id]) }
        XCTAssertEqual(manager.undoActionName, "New Collection")
        event { library.addToCollection(id: props.id, itemIDs: [plate.id]) }
        XCTAssertEqual(manager.undoActionName, "Add to Collection")
        event { library.renameCollection(id: props.id, to: "Set dressing") }
        XCTAssertEqual(manager.undoActionName, "Rename Collection")
        event { library.removeFromCollection(id: props.id, itemIDs: [lamp.id]) }
        XCTAssertEqual(manager.undoActionName, "Remove from Collection")
        event { library.deleteCollection(id: props.id) }
        XCTAssertEqual(manager.undoActionName, "Delete Collection")
        XCTAssertTrue(library.collections.isEmpty)

        manager.undo()
        XCTAssertEqual(library.collections.first?.name, "Set dressing")
        XCTAssertEqual(library.collections.first?.itemIDs, [plate.id])
        manager.undo()
        XCTAssertEqual(library.collections.first?.itemIDs, [plate.id, lamp.id])
        manager.undo()
        XCTAssertEqual(library.collections.first?.name, "Props")
        manager.undo()
        XCTAssertEqual(library.collections.first?.itemIDs, [lamp.id])
        manager.undo()
        XCTAssertTrue(library.collections.isEmpty)
        XCTAssertEqual(library.items.count, 2)

        manager.redo()
        XCTAssertEqual(library.collections.first?.name, "Props")
        XCTAssertEqual(manager.redoActionName, "Add to Collection")
        manager.redo()
        XCTAssertEqual(library.collections.first?.itemIDs, [lamp.id, plate.id])
        // The undone steps were written, so a reload sees what the window sees.
        XCTAssertEqual(StudioLibraryStore(libraryURL: libraryURL).collections.first?.itemIDs, [lamp.id, plate.id])
    }

    func testAddingWhatIsAlreadyThereIsNotAStep() {
        let lamp = finished("a brass lamp", output: "/tmp/lamp.png")
        library.upsert(lamp)
        var props: StudioLibraryCollection!
        event { props = library.createCollection(named: "Props", adding: [lamp.id]) }
        event { library.addToCollection(id: props.id, itemIDs: [lamp.id]) }
        event { library.renameCollection(id: props.id, to: "Props") }

        manager.undo()
        XCTAssertTrue(library.collections.isEmpty)
        XCTAssertFalse(manager.canUndo)
    }

    // MARK: - Filters

    func testCollectionModelAndTaskFiltersNarrowTheColumn() {
        let now = Date()
        let lamp = finished("a brass lamp", output: "/tmp/lamp.png", model: "image-zimage-nano", at: now)
        let plate = finished("a blue plate", output: "/tmp/plate.png", model: "image-flux2-klein", at: now.addingTimeInterval(-60))
        let song = finished("a sea shanty", output: "/tmp/shanty.wav", mode: .music, templateID: .musicGenerate,
                            model: "music-ace-step", at: now.addingTimeInterval(-120))
        let items = [lamp, plate, song]
        let props = StudioLibraryCollection(name: "Props", itemIDs: [plate.id, song.id])

        func ids(_ filter: StudioLibraryFilter) -> [UUID] {
            StudioLibraryPresenter.filter(items, with: filter, titles: .none).map(\.id)
        }

        XCTAssertEqual(ids(StudioLibraryFilter(scope: .all, collection: props)), [plate.id, song.id])
        XCTAssertEqual(ids(StudioLibraryFilter(scope: .domain, domain: .image, collection: props)), [plate.id])
        XCTAssertEqual(ids(StudioLibraryFilter(scope: .all, modelID: "image-zimage-nano")), [lamp.id])
        XCTAssertEqual(ids(StudioLibraryFilter(scope: .all, task: .musicCompose)), [song.id])
        XCTAssertEqual(ids(StudioLibraryFilter(scope: .all, collection: props, task: .imageGenerate)), [plate.id])
        XCTAssertTrue(ids(StudioLibraryFilter(scope: .all, collection: props, modelID: "image-zimage-nano")).isEmpty)
    }

    func testFilterOptionsListOnlyWhatTheRowsHold() {
        let lamp = finished("a brass lamp", output: "/tmp/lamp.png", model: "image-zimage-nano")
        let plate = finished("a blue plate", output: "/tmp/plate.png", model: "image-zimage-nano")
        let song = finished("a sea shanty", output: "/tmp/shanty.wav", mode: .music, templateID: .musicGenerate, model: "music-ace-step")

        let models = StudioLibraryPresenter.modelOptions(in: [lamp, plate, song], titles: .none)
        XCTAssertEqual(models.map(\.value).sorted(), ["image-zimage-nano", "music-ace-step"])

        let tasks = StudioLibraryPresenter.taskOptions(in: [lamp, song], scope: .all)
        XCTAssertEqual(tasks.map(\.value), [.imageGenerate, .musicCompose])
        XCTAssertEqual(tasks.map(\.title), ["Image · Generate", "Music · Compose"])
        XCTAssertEqual(StudioLibraryPresenter.taskOptions(in: [lamp], scope: .domain).map(\.title), ["Generate"])
    }

    // MARK: - Provenance

    func testASubmittedRunRecordsTheRowWhoseOutputItReads() throws {
        let picture = finished("a ceramic mug", output: root.appendingPathComponent("mug.png").path,
                               at: Date().addingTimeInterval(-60))
        library.upsert(picture)
        let template = try XCTUnwrap(CommandCatalog.template(id: .visionSegment))
        let request = StudioRunRequest(
            mode: .segment, templateID: .visionSegment, template: template, draft: template.defaultDraft(),
            execution: StudioExecution(templateID: .visionSegment, arguments: [
                "vision", "segment", picture.outputURL!.path, "--prompt", "cup",
                "--output", root.appendingPathComponent("mug-cup.png").path,
            ])
        )

        let recorded = library.start(request: request, commandPreview: "mere.run vision segment", source: .contract)

        XCTAssertEqual(recorded.sourceItemIDs, [picture.id])
        let reloaded = StudioLibraryStore(libraryURL: libraryURL)
        XCTAssertEqual(reloaded.items.first { $0.id == request.id }?.sourceItemIDs, [picture.id])
        XCTAssertEqual(reloaded.lineage.madeFrom(request.id).map(\.id), [picture.id])
        XCTAssertEqual(reloaded.lineage.usedIn(picture.id).map(\.id), [request.id])
    }

    func testAPromptRunRecordsItsAttachedInputAndARunWithoutOneRecordsNothing() throws {
        let picture = finished("a ceramic mug", output: root.appendingPathComponent("mug.png").path,
                               at: Date().addingTimeInterval(-60))
        library.upsert(picture)
        var draft = StudioDraft()
        draft.reset(for: .readImage)
        draft.inputPath = picture.outputURL!.path
        draft.prompt = "What is on the table?"
        let read = try StudioCommandAdapter.makeRequest(mode: .readImage, draft: draft, source: .contract)

        XCTAssertEqual(library.start(request: read, commandPreview: "preview", source: .contract).sourceItemIDs, [picture.id])

        var fresh = StudioDraft()
        fresh.reset(for: .createImage)
        fresh.prompt = "a new mug"
        let generate = try StudioCommandAdapter.makeRequest(mode: .createImage, draft: fresh, source: .contract)
        XCTAssertNil(library.start(request: generate, commandPreview: "preview", source: .contract).sourceItemIDs)
        let data = try String(contentsOf: libraryURL, encoding: .utf8)
        XCTAssertEqual(data.components(separatedBy: "\"sourceItemIDs\"").count - 1, 1)
    }

    func testAnOutputDestinationIsNeverASource() throws {
        let path = root.appendingPathComponent("mug-cup.png").path
        let earlier = finished("the first cut", output: path, mode: .segment, templateID: .visionSegment,
                               at: Date().addingTimeInterval(-60))
        library.upsert(earlier)
        let template = try XCTUnwrap(CommandCatalog.template(id: .visionSegment))
        let request = StudioRunRequest(
            mode: .segment, templateID: .visionSegment, template: template, draft: template.defaultDraft(),
            execution: StudioExecution(templateID: .visionSegment, arguments: [
                "vision", "segment", "/elsewhere/mug.png", "--prompt", "cup", "--output", path,
            ])
        )

        XCTAssertNil(library.start(request: request, commandPreview: "preview", source: .contract).sourceItemIDs)
    }

    func testLinksAreInferredForRowsThatRecordedNone() {
        let start = Date().addingTimeInterval(-600)
        let mug = finished("a ceramic mug", output: "/tmp/lineage/mug.png", at: start)
        let cup = finished("cup", output: "/tmp/lineage/cup.png", mode: .segment, templateID: .visionSegment,
                           at: start.addingTimeInterval(60),
                           arguments: ["vision", "segment", "/tmp/lineage/mug.png", "--output", "/tmp/lineage/cup.png"])
        let caption = finished("caption", output: "/tmp/lineage/caption.txt", mode: .readImage, templateID: .visionInspect,
                               at: start.addingTimeInterval(120), input: "/tmp/lineage/./mug.png")
        // Made after the caption read the path, so it cannot be what the caption read.
        let remake = finished("a second mug", output: "/tmp/lineage/mug.png", at: start.addingTimeInterval(180))

        let lineage = StudioLibraryLineage(items: [remake, caption, cup, mug])

        XCTAssertEqual(lineage.madeFrom(cup.id).map(\.id), [mug.id])
        XCTAssertEqual(lineage.madeFrom(caption.id).map(\.id), [mug.id])
        XCTAssertEqual(lineage.usedIn(mug.id).map(\.id), [caption.id, cup.id])
        XCTAssertTrue(lineage.usedIn(remake.id).isEmpty)
        XCTAssertFalse(lineage.hasLinks(remake.id))
    }

    func testRecordedLinksWinAndARemovedLinkStaysRemovedThroughUndo() {
        let start = Date().addingTimeInterval(-600)
        let mug = finished("a ceramic mug", output: "/tmp/lineage/mug.png", at: start)
        var cup = finished("cup", output: "/tmp/lineage/cup.png", mode: .segment, templateID: .visionSegment,
                           at: start.addingTimeInterval(60), input: "/tmp/lineage/mug.png")
        library.upsert(mug)
        library.upsert(cup)
        XCTAssertEqual(library.lineage.madeFrom(cup.id).map(\.id), [mug.id])

        event { library.removeSource(mug.id, from: cup.id) }
        XCTAssertEqual(manager.undoActionName, "Remove Link")
        XCTAssertTrue(library.lineage.madeFrom(cup.id).isEmpty)
        XCTAssertEqual(StudioLibraryStore(libraryURL: libraryURL).items.first { $0.id == cup.id }?.sourceItemIDs, [])

        manager.undo()
        XCTAssertNil(library.items.first { $0.id == cup.id }?.sourceItemIDs)
        XCTAssertEqual(library.lineage.madeFrom(cup.id).map(\.id), [mug.id])
        manager.redo()
        XCTAssertTrue(library.lineage.madeFrom(cup.id).isEmpty)

        // A recorded source is taken as recorded, whatever the paths say.
        cup.sourceItemIDs = [mug.id]
        let other = finished("unrelated", output: "/tmp/lineage/other.png", at: start.addingTimeInterval(30))
        XCTAssertEqual(StudioLibraryLineage(items: [cup, other, mug]).madeFrom(cup.id).map(\.id), [mug.id])
    }

    // MARK: - library.json compatibility

    /// A library.json in the shape shipped before collections and provenance: it decodes row for
    /// row, infers the link its paths imply, and a rewrite adds neither new key to a row without one.
    func testALibraryWrittenBeforeCollectionsAndProvenanceStillDecodes() throws {
        try Self.legacyLibrary.write(to: libraryURL, atomically: true, encoding: .utf8)

        let store = StudioLibraryStore(libraryURL: libraryURL)

        XCTAssertEqual(store.items.count, 2)
        XCTAssertTrue(store.collections.isEmpty)
        XCTAssertTrue(store.items.allSatisfy { $0.sourceItemIDs == nil })
        let mug = try XCTUnwrap(store.items.first { $0.mode == .createImage })
        let read = try XCTUnwrap(store.items.first { $0.mode == .readImage })
        XCTAssertEqual(store.lineage.madeFrom(read.id).map(\.id), [mug.id])

        store.setFavorite(id: mug.id, isFavorite: true)
        let written = try String(contentsOf: libraryURL, encoding: .utf8)
        XCTAssertFalse(written.contains("sourceItemIDs"))
        XCTAssertFalse(written.contains("collection"))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("collections.json").path))
        XCTAssertEqual(StudioLibraryStore(libraryURL: libraryURL).items.count, 2)
    }

    private static let legacyLibrary = """
    [
      {
        "commandPreview" : "mere.run vision read /Users/example/mug.png",
        "createdAt" : "2026-01-01T12:05:00Z",
        "exitCode" : 0,
        "id" : "8D0F4A10-5E7B-4C6D-9F11-2B3C4D5E6F01",
        "inputURL" : "file:///Users/example/mug.png",
        "mode" : "readImage",
        "outputText" : "A mug on a table.",
        "prompt" : "What is on the table?",
        "status" : "completed",
        "templateID" : "visionInspect",
        "updatedAt" : "2026-01-01T12:05:04Z"
      },
      {
        "artifactURLs" : [
          "file:///Users/example/mug.png"
        ],
        "commandPreview" : "mere.run image generate --model image-zimage-nano",
        "createdAt" : "2026-01-01T12:00:00Z",
        "exitCode" : 0,
        "id" : "8D0F4A10-5E7B-4C6D-9F11-2B3C4D5E6F02",
        "mode" : "createImage",
        "outputURL" : "file:///Users/example/mug.png",
        "prompt" : "a ceramic mug",
        "status" : "completed",
        "templateID" : "imageGenerate",
        "updatedAt" : "2026-01-01T12:00:04Z"
      }
    ]
    """
}
