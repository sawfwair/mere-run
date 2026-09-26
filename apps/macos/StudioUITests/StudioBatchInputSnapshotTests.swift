@testable import StudioKit
@testable import StudioUI
import AppKit
import SwiftUI
import XCTest

/// Offscreen snapshots of batch inputs, light and dark: the composer's stacked well with Run 12
/// (an audio batch and a picture batch), the batch's file list, the bar that follows a batch on
/// its page and in the run queue, and the Library picker's checks for a slot that takes several.
///
/// Skipped unless `MERERUN_STUDIO_SNAPSHOT_DIR` names a directory, like `StudioSnapshotTests`.
/// Everything renders through `StudioSnapshotRenderer` in a window that is never ordered on
/// screen; the files are written to a temporary folder and nothing reaches the user's Library.
@MainActor
final class StudioBatchInputSnapshotTests: XCTestCase {
    private var outputDirectory: URL!
    private var root: URL!

    override func setUpWithError() throws {
        try super.setUpWithError()
        guard let path = ProcessInfo.processInfo.environment["MERERUN_STUDIO_SNAPSHOT_DIR"],
              !path.trimmingCharacters(in: .whitespaces).isEmpty else {
            throw XCTSkip("Set MERERUN_STUDIO_SNAPSHOT_DIR to a directory to write Studio snapshots.")
        }
        outputDirectory = URL(fileURLWithPath: NSString(string: path).expandingTildeInPath, isDirectory: true)
        root = FileManager.default.temporaryDirectory.appendingPathComponent("batch-snapshots-\(UUID())", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        if let root { try? FileManager.default.removeItem(at: root) }
        try super.tearDownWithError()
    }

    /// Transcribe-style audio: twelve files on Audio ▸ Enhance. Pictures: four on Vision ▸ Depth.
    func testBatchedWellAndRunCountSnapshots() throws {
        let clips = try (1...12).map { try file(String(format: "interview-%02d.wav", $0)) }
        var enhance = StudioTaskDraft(templateID: .audioEnhance)
        enhance.attach(dropped: clips, slots: enhance.slots(source: .contract))
        XCTAssertEqual(enhance.batchInputPaths.count, 12)

        let pictures = try [("harbor", NSColor.systemTeal), ("orchard", NSColor.systemGreen),
                            ("canyon", NSColor.systemOrange), ("attic", NSColor.systemPurple)]
            .map { try picture($0.0, color: $0.1) }
        var depth = StudioTaskDraft(templateID: .visionDepth)
        depth.attach(dropped: pictures, slots: depth.slots(source: .contract))

        let size = CGSize(width: 760, height: 190)
        for appearance in StudioSnapshotAppearance.allCases {
            try write(ComposerHost(task: .audioEnhance, draft: enhance), size: size,
                      name: "batch-well-run12-\(appearance.rawValue)", appearance: appearance)
            try write(ComposerHost(task: .visionDepth, draft: depth), size: size,
                      name: "batch-well-pictures-\(appearance.rawValue)", appearance: appearance)
        }
        try write(ComposerHost(task: .audioEnhance, draft: enhance, readiness: .missingModel("audio-enhance-ap-bwe-16kto48k")),
                  size: size, name: "batch-well-run12-blocked-light", appearance: .light)
    }

    func testBatchListStatusAndQueueSnapshots() throws {
        let clips = try (1...12).map { try file(String(format: "interview-%02d.wav", $0)) }
        var draft = StudioTaskDraft(templateID: .audioEnhance)
        draft.attach(dropped: clips, slots: draft.slots(source: .contract))
        let slot = try XCTUnwrap(draft.slots(source: .contract).first)
        let progress = StudioBatchProgress(group: UUID(), task: .audioEnhance, total: 12, completed: 4, failed: 1, active: 7,
                                           startedAt: Date())

        for appearance in StudioSnapshotAppearance.allCases {
            let list = StudioBatchFileList(slot: slot, draft: .constant(draft), onAddFromDisk: {})
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
                .background(MereRunTheme.background)
            try write(list, size: CGSize(width: 340, height: 400), name: "batch-list-\(appearance.rawValue)", appearance: appearance)

            let bars = VStack(spacing: 12) {
                StudioBatchStatusBar(progress: progress, onStop: {})
                StudioBatchQueueRow(progress: progress, onStop: {})
                    .frame(width: 400)
                    .background(RoundedRectangle(cornerRadius: MereRunTheme.Radius.popover).fill(MereRunTheme.surface))
            }
            .padding(.vertical, 16)
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
            .background(MereRunTheme.background)
            try write(bars, size: CGSize(width: 720, height: 190), name: "batch-progress-\(appearance.rawValue)", appearance: appearance)
        }
    }

    func testLibraryPickerChecksForASlotThatTakesSeveralSnapshots() throws {
        let items = try ["Harbor at dusk", "Orchard rows", "Canyon light"].enumerated().map { index, title in
            let url = try picture("result-\(index)", color: [NSColor.systemTeal, .systemGreen, .systemOrange][index])
            var item = StudioLibraryItem(
                id: UUID(), mode: .createImage, prompt: title, inputURL: nil, outputURL: url,
                createdAt: StudioSnapshotRenderer.referenceDate.addingTimeInterval(Double(-index) * 600),
                updatedAt: StudioSnapshotRenderer.referenceDate, status: .completed, exitCode: 0,
                commandPreview: "mere.run image generate", outputText: nil
            )
            item.artifactURLs = [url]
            return item
        }
        let slot = try XCTUnwrap(StudioTaskSchema.primarySlot(for: .visionDepth))
        for appearance in StudioSnapshotAppearance.allCases {
            let picker = StudioLibraryInputPicker(requirement: StudioAttachmentRequirement(slot: slot), items: items,
                                                  onPick: { _ in }, onChooseFromDisk: {})
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(MereRunTheme.surface)
            try write(picker, size: CGSize(width: 400, height: 520), name: "batch-library-picker-\(appearance.rawValue)",
                      appearance: appearance)
        }
    }

    // MARK: - Helpers

    private func write<Content: View>(_ view: Content, size: CGSize, name: String, appearance: StudioSnapshotAppearance) throws {
        let url = outputDirectory.appendingPathComponent("\(name).png")
        try StudioSnapshotRenderer.writePNG(view, size: size, appearance: appearance, to: url, settle: 1.2)
        XCTAssertTrue(FileManager.default.fileExists(atPath: url.path))
    }

    private func file(_ name: String) throws -> URL {
        let url = root.appendingPathComponent(name)
        try Data([0]).write(to: url)
        return url
    }

    /// A small picture of one colour with a lighter band, so a thumbnail reads as a picture.
    private func picture(_ name: String, color: NSColor) throws -> URL {
        let side = 96
        let rep = try XCTUnwrap(NSBitmapImageRep(
            bitmapDataPlanes: nil, pixelsWide: side, pixelsHigh: side, bitsPerSample: 8, samplesPerPixel: 4,
            hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0
        ))
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
        color.setFill()
        NSRect(x: 0, y: 0, width: side, height: side).fill()
        color.blended(withFraction: 0.5, of: .white)?.setFill()
        NSRect(x: 0, y: side / 3, width: side, height: side / 4).fill()
        NSGraphicsContext.restoreGraphicsState()
        let url = root.appendingPathComponent("\(name).png")
        try XCTUnwrap(rep.representation(using: .png, properties: [:])).write(to: url)
        return url
    }
}

/// The task composer on its own, as the workspace hosts it, with the prompt unfocused.
private struct ComposerHost: View {
    let task: StudioTask
    @State var draft: StudioTaskDraft
    var readiness: ModelReadinessState = .ready
    @FocusState private var focused: Bool

    init(task: StudioTask, draft: StudioTaskDraft, readiness: ModelReadinessState = .ready) {
        self.task = task
        _draft = State(initialValue: draft)
        self.readiness = readiness
    }

    var body: some View {
        VStack {
            Spacer(minLength: 0)
            StudioTaskComposer(
                task: task, draft: $draft, isRunning: false, queuedCount: 0, readiness: readiness, modelInventory: [],
                promptFocus: $focused, onRun: {}, onStop: {}, onShowModels: {}, onRecallPrompt: { _ in }
            )
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(MereRunTheme.background)
    }
}
