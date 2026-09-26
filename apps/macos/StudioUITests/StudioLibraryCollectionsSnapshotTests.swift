@testable import StudioKit
@testable import StudioUI
import StudioTestSupport
import AppKit
import SwiftUI
import XCTest

/// Offscreen snapshots of collections, the Library filter, and lineage links, light and dark.
/// Skipped unless `MERERUN_STUDIO_SNAPSHOT_DIR` names a directory, like `StudioSnapshotTests`:
///
/// ```
/// MERERUN_STUDIO_SNAPSHOT_DIR=/tmp/shots swift test --filter StudioLibraryCollectionsSnapshotTests
/// ```
///
/// The Library, its collections, and the pictures are written to a temporary folder; no CLI
/// process starts and nothing is presented on screen.
@MainActor
final class StudioLibraryCollectionsSnapshotTests: XCTestCase {
    private var outputDirectory: URL!
    private var root: URL!
    private var library: StudioLibraryStore!
    private var controller: MereRunController!
    /// The picture a Segment and a Read run were made from, and the cut Segment made.
    private var mug: StudioLibraryItem!
    private var cutout: StudioLibraryItem!
    private var covers: StudioLibraryCollection!

    override func setUp() async throws {
        try await MainActor.run { try prepare() }
    }

    override func tearDown() async throws {
        await MainActor.run {
            controller?.terminateAllProcesses()
            controller?.taskSessions.flush()
            if let root { try? FileManager.default.removeItem(at: root) }
        }
    }

    private func prepare() throws {
        guard let path = ProcessInfo.processInfo.environment["MERERUN_STUDIO_SNAPSHOT_DIR"],
              !path.trimmingCharacters(in: .whitespaces).isEmpty else {
            throw XCTSkip("Set MERERUN_STUDIO_SNAPSHOT_DIR to a directory to write Studio snapshots.")
        }
        outputDirectory = URL(fileURLWithPath: NSString(string: path).expandingTildeInPath, isDirectory: true)
        try FileManager.default.createDirectory(at: outputDirectory, withIntermediateDirectories: true)
        root = FileManager.default.temporaryDirectory.appendingPathComponent("StudioCollectionsSnapshots-\(UUID())", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        controller = MereRunController(
            secretStore: InMemorySecretStore(), processRunner: RecordingProcessRunner(), resolvesCLIOnInit: false,
            taskSessions: StudioTaskSessions(url: root.appendingPathComponent("sessions.json"))
        )
        library = StudioLibraryStore(libraryURL: root.appendingPathComponent("library.json"))
        try seed()
    }

    /// The Library column with two collections as chips under the search field: the first render
    /// shows every run with no collection chosen, the second is filtered to "Album covers".
    func testLibraryCollectionChipsSnapshots() throws {
        let renders: [(String, StudioSnapshotAppearance, UUID?)] = [
            ("collections-library-light", .light, nil),
            ("collections-library-dark", .dark, nil),
            ("collections-library-filtered-light", .light, covers.id),
            ("collections-library-filtered-dark", .dark, covers.id),
        ]
        for (name, appearance, collectionID) in renders {
            let view = libraryPanel
                .environment(\.studioLibrarySeed, StudioLibrarySeed(viewMode: .list, scope: .all, collectionID: collectionID))
            try write(view, size: CGSize(width: 300, height: 560), appearance: appearance, name: name)
        }
    }

    /// The filter popover's content with a task and a model chosen, as it opens from the funnel
    /// beside the search field.
    func testLibraryFilterPanelSnapshots() throws {
        let items = library.items
        for (name, appearance) in [("collections-filter-light", StudioSnapshotAppearance.light), ("collections-filter-dark", .dark)] {
            let view = StudioLibraryFilterPanel(
                kind: .constant(.images),
                favoritesOnly: .constant(false),
                task: .constant(.visionSegment),
                modelID: .constant("vision-sam3"),
                taskOptions: StudioLibraryPresenter.taskOptions(in: items, scope: .all),
                modelOptions: StudioLibraryPresenter.modelOptions(in: items, titles: .none)
            )
            .background(MereRunTheme.surface)
            try write(view, size: CGSize(width: StudioLibraryFilterPanel.width, height: 380), appearance: appearance, name: name)
        }
    }

    /// A run's detail with its lineage: the Segment cut is made from the mug and used in a Read
    /// run; the mug's detail lists both runs it was used in.
    func testRunDetailLineageSnapshots() throws {
        for (name, appearance) in [("collections-lineage-detail-light", StudioSnapshotAppearance.light), ("collections-lineage-detail-dark", .dark)] {
            let view = VStack(spacing: 16) {
                StudioRunDetailView(item: cutout, preferredKinds: [.image])
                StudioRunDetailView(item: mug, preferredKinds: [.image])
            }
            .padding(16)
            .frame(width: 620)
            .background(MereRunTheme.background)
            .environmentObject(controller)
            .environment(\.studioLibraryLinks, links)
            try write(view, size: CGSize(width: 620, height: 820), appearance: appearance, name: name)
        }
    }

    /// The focused result's footer and a feed card, each with the cut's lineage.
    func testResultAndCardLineageSnapshots() throws {
        let url = try XCTUnwrap(cutout.outputURL)
        for (name, appearance) in [("collections-lineage-result-light", StudioSnapshotAppearance.light), ("collections-lineage-result-dark", .dark)] {
            let view = StudioResultWorkspaceView(
                item: cutout, url: url, items: library.items,
                onClose: {}, onVary: { _ in }, onSave: { _ in }, onContinue: { _, _, _ in }
            )
            .environment(\.studioLibraryLinks, links)
            try write(view, size: CGSize(width: 900, height: 560), appearance: appearance, name: name)
        }
        for (name, appearance) in [("collections-lineage-card-light", StudioSnapshotAppearance.light), ("collections-lineage-card-dark", .dark)] {
            let view = StudioGenerationCard(item: cutout, isHighlighted: false, actions: Self.inertActions)
                .padding(16)
                .frame(width: 620)
                .background(MereRunTheme.background)
                .environment(\.studioLibraryLinks, links)
            try write(view, size: CGSize(width: 620, height: 440), appearance: appearance, name: name)
        }
    }

    // MARK: - Fixture

    private var links: StudioLibraryLinks {
        StudioLibraryLinks(library: library, open: { _ in }, removeLink: { _, _ in })
    }

    private var libraryPanel: some View {
        StudioLibraryPanel(
            items: library.items,
            domain: .image,
            scope: .constant(.all),
            viewMode: .constant(.list),
            kind: .constant(.all),
            favoritesOnly: .constant(false),
            progressByID: [:],
            selectedID: .constant(cutout.id),
            onSelect: { _ in }, onDelete: { _, _ in }, onRename: { _, _ in }, onToggleFavorite: { _ in },
            onQuickLook: { _ in }, onReveal: { _ in }, onExport: { _ in }, onRetry: { _ in }, onEdit: { _ in },
            onUseSettings: { _ in }, onCompare: { _ in }, onRunVariations: { _, _ in }
        )
        .frame(width: 300, height: 560)
        .environmentObject(library)
    }

    private static let inertActions = StudioFeedActions(
        vary: { _ in }, rerun: { _ in }, saveTo: { _ in }, cancel: { _ in }, remove: { _ in }, retry: { _ in },
        delete: { _ in }, useSettings: { _ in }, pullModel: { _ in }, useExample: { _ in }, attach: nil
    )

    /// Six finished runs across Image, Vision, and Music: a generated mug, its Segment cut, a
    /// Read of the cut, a second generation, a song, and a caption of the mug, with two
    /// collections holding some of them.
    private func seed() throws {
        let now = StudioSnapshotRenderer.referenceDate
        let mugURL = root.appendingPathComponent("mug.png")
        let cutURL = root.appendingPathComponent("mug-cup.png")
        let lampURL = root.appendingPathComponent("lamp.png")
        try Self.writePNG(to: mugURL, hue: 0.08)
        try Self.writePNG(to: cutURL, hue: 0.55)
        try Self.writePNG(to: lampURL, hue: 0.12)

        mug = Self.run("A ceramic mug in soft morning light", output: mugURL, mode: .createImage, templateID: .imageGenerate,
                       model: "image-zimage-nano", at: now.addingTimeInterval(-3_600))
        cutout = Self.run("cup", output: cutURL, mode: .segment, templateID: .visionSegment, model: "vision-sam3",
                          at: now.addingTimeInterval(-2_400))
        cutout.customTitle = "Mug, cup only"
        cutout.sourceItemIDs = [mug.id]
        var read = Self.run("What is printed on the cup?", output: nil, mode: .readImage, templateID: .visionInspect,
                            model: "vision-qwen3-vl", at: now.addingTimeInterval(-1_200))
        read.inputURL = cutURL
        read.outputText = "A small lighthouse and the word HARBOR."
        var caption = Self.run("Describe the mug", output: nil, mode: .readImage, templateID: .visionInspect,
                               model: "vision-qwen3-vl", at: now.addingTimeInterval(-1_800))
        caption.inputURL = mugURL
        caption.outputText = "A cream mug on a linen cloth."
        let lamp = Self.run("A brass desk lamp, studio lit", output: lampURL, mode: .createImage, templateID: .imageGenerate,
                            model: "image-flux2-klein", at: now.addingTimeInterval(-600))
        let song = Self.run("A sea shanty for a harbor morning", output: nil, mode: .music, templateID: .musicGenerate,
                            model: "music-ace-step", at: now.addingTimeInterval(-300))
        for item in [mug!, cutout!, caption, read, lamp, song] { library.upsert(item) }

        covers = library.createCollection(named: "Album covers", adding: [mug.id, lamp.id, song.id])
        library.createCollection(named: "Harbor mug", adding: [mug.id, cutout.id, read.id])
    }

    private static func run(
        _ prompt: String, output: URL?, mode: StudioMode, templateID: CommandTemplateID, model: String, at date: Date
    ) -> StudioLibraryItem {
        var item = StudioLibraryItem(
            id: UUID(), mode: mode, prompt: prompt, inputURL: nil, outputURL: output, createdAt: date, updatedAt: date,
            status: .completed, exitCode: 0, commandPreview: "mere.run", outputText: nil, templateID: templateID
        )
        item.artifactURLs = output.map { [$0] }
        item.model = model
        return item
    }

    private func write<Content: View>(_ view: Content, size: CGSize, appearance: StudioSnapshotAppearance, name: String) throws {
        let rep = try StudioSnapshotRenderer.render(view, size: size, appearance: appearance)
        let coverage = StudioSnapshotRenderer.nonBlankCoverage(of: rep)
        XCTAssertGreaterThan(coverage, 0.05, "\(name) rendered blank (non-blank coverage \(coverage))")
        let data = try XCTUnwrap(rep.representation(using: .png, properties: [:]))
        try data.write(to: outputDirectory.appendingPathComponent("\(name).png"), options: .atomic)
    }

    /// A 512×384 picture: a soft two-tone gradient around `hue` with a round shape in the middle.
    private static func writePNG(to url: URL, hue: CGFloat) throws {
        let rep = try XCTUnwrap(NSBitmapImageRep(
            bitmapDataPlanes: nil, pixelsWide: 512, pixelsHigh: 384, bitsPerSample: 8, samplesPerPixel: 4,
            hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0
        ))
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
        let bounds = NSRect(x: 0, y: 0, width: 512, height: 384)
        NSGradient(
            starting: NSColor(hue: hue, saturation: 0.35, brightness: 0.95, alpha: 1),
            ending: NSColor(hue: hue + 0.05, saturation: 0.6, brightness: 0.55, alpha: 1)
        )?.draw(in: bounds, angle: -60)
        NSColor(hue: hue, saturation: 0.15, brightness: 0.98, alpha: 1).setFill()
        NSBezierPath(ovalIn: NSRect(x: 176, y: 112, width: 160, height: 160)).fill()
        NSGraphicsContext.restoreGraphicsState()
        try XCTUnwrap(rep.representation(using: .png, properties: [:])).write(to: url)
    }
}
