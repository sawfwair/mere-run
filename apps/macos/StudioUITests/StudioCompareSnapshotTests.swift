@testable import StudioKit
@testable import StudioUI
import StudioTestSupport
import AVFoundation
import AppKit
import SwiftUI
import XCTest

/// Offscreen snapshots of Compare and of the feed's variation cards, written only when
/// `MERERUN_STUDIO_SNAPSHOT_DIR` names a directory (see `StudioSnapshotTests`):
///
/// ```
/// MERERUN_STUDIO_SNAPSHOT_DIR=/tmp/shell-shots swift test --filter StudioCompareSnapshotTests
/// ```
///
/// The pictures and sounds are drawn and synthesized into a temporary folder, and the Library is
/// a temporary `library.json`; nothing touches the user's Library, Preferences, or media folders.
@MainActor
final class StudioCompareSnapshotTests: XCTestCase {
    private var outputDirectory: URL!
    private var root: URL!
    private var library: StudioLibraryStore!

    override func setUpWithError() throws {
        try super.setUpWithError()
        guard let path = ProcessInfo.processInfo.environment["MERERUN_STUDIO_SNAPSHOT_DIR"],
              !path.trimmingCharacters(in: .whitespaces).isEmpty else {
            throw XCTSkip("Set MERERUN_STUDIO_SNAPSHOT_DIR to a directory to write Compare snapshots.")
        }
        outputDirectory = URL(fileURLWithPath: NSString(string: path).expandingTildeInPath, isDirectory: true)
        root = FileManager.default.temporaryDirectory.appendingPathComponent("StudioCompareSnapshots-\(UUID())", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: outputDirectory, withIntermediateDirectories: true)
        library = StudioLibraryStore(libraryURL: root.appendingPathComponent("library.json"))
    }

    override func tearDownWithError() throws {
        library = nil
        if let root { try? FileManager.default.removeItem(at: root) }
        try super.tearDownWithError()
    }

    /// Four variations of one Image ▸ Generate prompt: side by side and as an A/B slider, light
    /// and dark, at the default window width; two picked results at a compact width.
    func testImageCompareSnapshots() throws {
        let group = UUID()
        let hues: [CGFloat] = [0.08, 0.55, 0.33, 0.8]
        let items = try hues.enumerated().map { offset, hue in
            try imageRow(seed: String(1_000 + offset * 3_517), steps: offset == 3 ? 12 : 8, hue: hue,
                         file: "harbor-\(offset).png", group: group, favorite: offset == 1)
        }
        let size = CGSize(width: 1_440, height: 860)
        for appearance in StudioSnapshotAppearance.allCases {
            try write(compare(items), size: size, appearance: appearance, name: "compare-images-grid-\(appearance.rawValue)")
            try write(compare(items, layout: .slider), size: size, appearance: appearance,
                      name: "compare-images-slider-\(appearance.rawValue)")
        }
        try write(compare(Array(items.prefix(2))), size: CGSize(width: 820, height: 760), appearance: .light,
                  name: "compare-images-pair-compact-light")
    }

    /// Three sounds from one Sound ▸ Generate prompt: stacked waveforms on one playhead, A
    /// heard, light and dark.
    func testAudioCompareSnapshots() throws {
        let group = UUID()
        let items = try [(220.0, 2.4), (330.0, 3.2), (440.0, 1.8)].enumerated().map { offset, tone in
            try soundRow(seed: String(77 + offset), frequency: tone.0, seconds: tone.1, file: "door-\(offset).wav", group: group)
        }
        for appearance in StudioSnapshotAppearance.allCases {
            try write(compare(items), size: CGSize(width: 1_200, height: 760), appearance: appearance,
                      name: "compare-audio-\(appearance.rawValue)", settle: 2.5)
        }
    }

    /// A finished variation's card: "2 of 4" beside its time, Vary with its variations menu, and
    /// "Compare 4"; one card picked for Compare, so the feed's selection bar shows.
    func testVariationCardSnapshots() throws {
        let group = UUID()
        let items = try (0..<4).map { offset in
            try imageRow(seed: String(40 + offset), steps: 8, hue: 0.1 + CGFloat(offset) * 0.2,
                         file: "kite-\(offset).png", group: group, favorite: false)
        }
        let cards = items.map { StudioFeedCard(item: $0, kind: .generation, job: nil) }
        let actions = StudioFeedActions(
            vary: { _ in }, rerun: { _ in }, saveTo: { _ in }, cancel: { _ in }, remove: { _ in }, retry: { _ in },
            delete: { _ in }, useSettings: { _ in }, pullModel: { _ in }, useExample: { _ in }, attach: nil,
            runVariations: { _, _ in }, compare: { _ in }
        )
        for appearance in StudioSnapshotAppearance.allCases {
            let card = StudioGenerationCard(
                item: items[1], isHighlighted: false, actions: actions,
                variation: StudioVariations.positions(in: items)[items[1].id], variationGroup: items,
                isPickedForCompare: true, onPickForCompare: {}
            )
            .padding(24)
            .frame(width: 860)
            .background(MereRunTheme.background)
            try write(card, size: CGSize(width: 860, height: 420), appearance: appearance,
                      name: "variation-card-\(appearance.rawValue)")
            let bar = StudioCompareSelectionBar(items: Array(cards.prefix(3).map(\.item)), onCompare: { _ in }, onClear: {})
                .padding(24)
                .frame(width: 520)
                .background(MereRunTheme.background)
            try write(bar, size: CGSize(width: 520, height: 110), appearance: appearance,
                      name: "compare-selection-bar-\(appearance.rawValue)")
        }
    }

    /// The Run area where the three controls meet: the Image composer with earlier prompts
    /// (the history clock), Run variations, and Run; over it Enhance with twelve files in its
    /// well, where Run reads "Run 12" and variations are not offered, since a batch already runs
    /// once per file.
    func testComposerRunAreaSnapshots() throws {
        let controller = MereRunController(secretStore: InMemorySecretStore(), processRunner: RecordingProcessRunner(),
            resolvesCLIOnInit: false, taskSessions: StudioTaskSessions(url: root.appendingPathComponent("sessions.json")))
        controller.modelIdentities.use(nil)
        defer { controller.terminateAllProcesses() }
        let history = try (0..<3).map { offset in
            try imageRow(seed: String(offset), steps: 8, hue: 0.3, file: "history-\(offset).png", group: UUID(), favorite: false)
        }
        let clips = try (1...12).map { index -> URL in
            let url = root.appendingPathComponent(String(format: "interview-%02d.wav", index))
            try Self.writeToneWAV(to: url, frequency: 220, seconds: 0.2)
            return url
        }
        var enhance = StudioTaskDraft(templateID: .audioEnhance)
        enhance.model = "audio-enhance-universr-audio"
        enhance.attach(dropped: clips, slots: enhance.slots(source: .contract))
        XCTAssertTrue(StudioVariations.applies(to: enhance, source: .contract), "UniverSR takes a seed")

        let size = CGSize(width: 760, height: 380)
        for appearance in StudioSnapshotAppearance.allCases {
            let view = VStack(spacing: 0) {
                RunAreaTaskHost(draft: enhance)
                RunAreaImageHost()
            }
            .environmentObject(controller)
            .environment(\.studioLibraryItems, history)
            .background(MereRunTheme.background)
            try write(view, size: size, appearance: appearance, name: "composer-run-area-\(appearance.rawValue)")
        }
    }

    // MARK: Rendering

    private func compare(_ items: [StudioLibraryItem], layout: StudioCompareImageLayout = .grid) -> some View {
        StudioCompareView(items: items, onClose: {}, onKeep: { _ in }, onUseSettings: { _ in }, initialImageLayout: layout)
    }

    private func write<Content: View>(
        _ view: Content, size: CGSize, appearance: StudioSnapshotAppearance, name: String, settle: TimeInterval = 2
    ) throws {
        let rep = try StudioSnapshotRenderer.render(view.frame(width: size.width, height: size.height),
                                                    size: size, appearance: appearance, settle: settle)
        let coverage = StudioSnapshotRenderer.nonBlankCoverage(of: rep)
        XCTAssertGreaterThan(coverage, 0.05, "\(name) rendered blank (non-blank coverage \(coverage))")
        guard let data = rep.representation(using: .png, properties: [:]) else { throw StudioSnapshotError.pngEncodingFailed }
        try data.write(to: outputDirectory.appendingPathComponent("\(name).png"), options: .atomic)
    }

    // MARK: Fixtures

    private func imageRow(
        seed: String, steps: Int, hue: CGFloat, file: String, group: UUID, favorite: Bool
    ) throws -> StudioLibraryItem {
        let url = root.appendingPathComponent(file)
        try Self.writeHarborPNG(to: url, hue: hue)
        let template = try XCTUnwrap(CommandCatalog.template(id: .imageGenerate))
        var draft = template.defaultDraft()
        draft.prompt = "A harbor at dusk, fishing boats, long exposure"
        draft.seed = seed
        draft.steps = steps
        draft.width = 768
        draft.height = 512
        draft.outputPath = url.path
        return try finish(StudioRunRequest(mode: .createImage, templateID: template.id, template: template, draft: draft),
                          output: url, group: group, favorite: favorite)
    }

    private func soundRow(seed: String, frequency: Double, seconds: Double, file: String, group: UUID) throws -> StudioLibraryItem {
        let url = root.appendingPathComponent(file)
        try Self.writeToneWAV(to: url, frequency: frequency, seconds: seconds)
        let template = try XCTUnwrap(CommandCatalog.template(id: .sfxGenerate))
        var draft = template.defaultDraft()
        draft.prompt = "A heavy wooden door slams in a stone hallway"
        draft.seed = seed
        draft.outputPath = url.path
        return try finish(StudioRunRequest(mode: .sfx, templateID: template.id, template: template, draft: draft),
                          output: url, group: group, favorite: false)
    }

    private func finish(_ request: StudioRunRequest, output: URL, group: UUID, favorite: Bool) throws -> StudioLibraryItem {
        let row = library.start(request: request, commandPreview: "fixture", source: .contract)
        library.complete(id: row.id, exitCode: 0, outputURL: output, outputText: nil, commandPreview: "fixture")
        library.assignVariationGroup(group, to: [row.id])
        if favorite { library.setFavorite(id: row.id, isFavorite: true) }
        return try XCTUnwrap(library.items.first { $0.id == row.id })
    }

    /// A sky in `hue`, a sea, and a boat, so each variation reads as its own picture.
    private static func writeHarborPNG(to url: URL, hue: CGFloat) throws {
        let size = CGSize(width: 768, height: 512)
        guard let rep = NSBitmapImageRep(
            bitmapDataPlanes: nil, pixelsWide: Int(size.width), pixelsHigh: Int(size.height), bitsPerSample: 8,
            samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0
        ), let context = NSGraphicsContext(bitmapImageRep: rep) else { throw StudioSnapshotError.noBitmap }
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = context
        let sky = NSGradient(starting: NSColor(hue: hue, saturation: 0.55, brightness: 0.95, alpha: 1),
                             ending: NSColor(hue: hue + 0.08, saturation: 0.7, brightness: 0.55, alpha: 1))
        sky?.draw(in: CGRect(x: 0, y: size.height * 0.38, width: size.width, height: size.height * 0.62), angle: -90)
        NSColor(hue: 0.58, saturation: 0.55, brightness: 0.35, alpha: 1).setFill()
        CGRect(x: 0, y: 0, width: size.width, height: size.height * 0.38).fill()
        NSColor(hue: hue, saturation: 0.3, brightness: 1, alpha: 0.9).setFill()
        NSBezierPath(ovalIn: CGRect(x: size.width * (0.2 + hue * 0.5), y: size.height * 0.55, width: 70, height: 70)).fill()
        NSColor(white: 0.12, alpha: 1).setFill()
        let hull = NSBezierPath()
        hull.move(to: CGPoint(x: 250, y: 190))
        hull.line(to: CGPoint(x: 470, y: 190))
        hull.line(to: CGPoint(x: 430, y: 150))
        hull.line(to: CGPoint(x: 290, y: 150))
        hull.close()
        hull.fill()
        CGRect(x: 355, y: 190, width: 6, height: 130).fill()
        NSGraphicsContext.restoreGraphicsState()
        guard let data = rep.representation(using: .png, properties: [:]) else { throw StudioSnapshotError.pngEncodingFailed }
        try data.write(to: url, options: .atomic)
    }

    /// A decaying tone with a little tremolo, so each waveform has its own shape.
    private static func writeToneWAV(to url: URL, frequency: Double, seconds: Double) throws {
        let rate = 22_050.0
        let format = try XCTUnwrap(AVAudioFormat(standardFormatWithSampleRate: rate, channels: 1))
        let frames = AVAudioFrameCount(rate * seconds)
        let buffer = try XCTUnwrap(AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames))
        buffer.frameLength = frames
        let samples = try XCTUnwrap(buffer.floatChannelData?[0])
        for frame in 0..<Int(frames) {
            let time = Double(frame) / rate
            let envelope = exp(-time * 1.6) * (0.65 + 0.35 * sin(2 * .pi * time * frequency / 110))
            samples[frame] = Float(envelope * sin(2 * .pi * frequency * time) * 0.8)
        }
        let file = try AVAudioFile(forWriting: url, settings: format.settings)
        try file.write(from: buffer)
    }
}

/// The Image composer with variations offered, the way the prompt workspace hosts it.
private struct RunAreaImageHost: View {
    @State private var draft: StudioDraft = {
        var draft = StudioDraft()
        draft.reset(for: .createImage)
        draft.prompt = "A harbor at dusk, fishing boats, long exposure"
        return draft
    }()
    @FocusState private var focused: Bool

    var body: some View {
        StudioComposer(
            mode: .createImage, draft: $draft, isRunning: false, queuedCount: 0, readiness: .ready,
            modelInventory: [], promptFocus: $focused, onRun: {}, onStop: {}, onShowModels: {},
            onRecallPrompt: { draft.prompt = $0 }, onRunVariations: { _ in }
        )
    }
}

/// A task composer over a batched well, handed variations the way the task workspace would.
private struct RunAreaTaskHost: View {
    @State var draft: StudioTaskDraft
    @FocusState private var focused: Bool

    var body: some View {
        StudioTaskComposer(
            task: .audioEnhance, draft: $draft, isRunning: false, queuedCount: 0, readiness: .ready, modelInventory: [],
            promptFocus: $focused, onRun: {}, onStop: {}, onShowModels: {}, onRecallPrompt: { _ in },
            onRunVariations: { _ in }
        )
    }
}
