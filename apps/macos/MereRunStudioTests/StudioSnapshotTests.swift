@testable import MereRunApp
import AppKit
import Foundation
import SwiftUI
import XCTest

/// Offscreen snapshots of the Studio shell for visual review.
///
/// These tests are skipped unless the `MERERUN_STUDIO_SNAPSHOT_DIR` environment variable names a
/// directory, so CI and the default `swift test` run never render anything. When it is set, each
/// test writes PNGs into that directory:
///
/// ```
/// MERERUN_STUDIO_SNAPSHOT_DIR=/tmp/shell-shots swift test --filter StudioSnapshotTests
/// ```
///
/// Everything is rendered through `StudioSnapshotRenderer`, which hosts the view in a window that
/// is never ordered on screen. The controller uses a process runner that fails every launch, so no
/// CLI process starts, and the Library is a temporary `library.json` seeded with fixture rows; the
/// user's real Library, `UserDefaults` domain, and Application Support are never touched.
@MainActor
final class StudioSnapshotTests: XCTestCase {
    static let shellSize = CGSize(
        width: StudioLayoutPolicy.defaultWindowWidth,
        height: StudioLayoutPolicy.defaultWindowHeight
    )
    /// The size the v2 mockups are drawn at.
    static let fidelitySize = CGSize(width: 1_440, height: 900)
    static let consoleSize = CGSize(width: 1_260, height: 780)
    static let settingsSize = CGSize(width: 560, height: 640)

    private var fixture: SnapshotFixture!

    override func setUpWithError() throws {
        try super.setUpWithError()
        guard let directory = Self.snapshotDirectory() else {
            throw XCTSkip("Set MERERUN_STUDIO_SNAPSHOT_DIR to a directory to write Studio shell snapshots.")
        }
        fixture = try SnapshotFixture(outputDirectory: directory)
    }

    override func tearDownWithError() throws {
        fixture?.tearDown()
        fixture = nil
        try super.tearDownWithError()
    }

    /// The full `StudioRootView` shell for every domain at its default task, light and dark.
    ///
    /// The shell restores `studio.destination` from `@SceneStorage` on appear, and outside a real
    /// scene that is always the default (Image), so each domain is reached the way a user reaches
    /// it: through the shared `NavigationModel` after the view has appeared.
    func testShellSnapshotsForEveryDomain() throws {
        for domain in StudioDomain.allCases {
            for appearance in StudioSnapshotAppearance.allCases {
                let navigation = NavigationModel()
                let view = StudioRootView()
                    .environmentObject(fixture.controller)
                    .environmentObject(fixture.library)
                    .environmentObject(navigation)
                    .frame(width: Self.shellSize.width, height: Self.shellSize.height)
                try fixture.write(
                    view,
                    size: Self.shellSize,
                    appearance: appearance,
                    name: "shell-\(domain.rawValue)-\(appearance.rawValue)",
                    settle: 2.0,
                    afterAppear: { navigation.open(destination: domain.defaultDestination) }
                )
            }
        }
    }

    /// Video ▸ Subjects in the Track stage: a three-subject plan whose masks were tracked (the
    /// manifest, tracking, and quality reports the CLI would have written, with a synthesized
    /// overlay frame), while a re-track job is held open by the process seam so the job bar
    /// shows it running. No CLI starts. Rendered at the mockup size, light and dark, plus the
    /// default window size.
    func testVideoSubjectsProjectSnapshots() throws {
        var seed = try fixture.seedTrackedSubjectsProject()
        let renders: [(name: String, size: CGSize, appearance: StudioSnapshotAppearance, stage: StudioSubjectsStage)] = [
            ("f8-subjects-light", Self.fidelitySize, .light, .track),
            ("f8-subjects-dark", Self.fidelitySize, .dark, .track),
            ("f8-subjects-compact-light", Self.shellSize, .light, .track),
            ("f8-subjects-plan-light", Self.fidelitySize, .light, .plan),
            ("f8-subjects-animate-light", Self.fidelitySize, .light, .animate)
        ]
        for render in renders {
            seed.stage = render.stage
            let navigation = NavigationModel()
            let view = StudioRootView()
                .environmentObject(fixture.controller)
                .environmentObject(fixture.library)
                .environmentObject(navigation)
                .environment(\.studioSubjectsProjectSeed, seed)
                .frame(width: render.size.width, height: render.size.height)
            try fixture.write(
                view,
                size: render.size,
                appearance: render.appearance,
                name: render.name,
                settle: 2.5,
                afterAppear: { navigation.open(task: .videoSubjects) }
            )
        }
        let maskRow = fixture.library.items.first { $0.id == seed.maskRequestID }
        XCTAssertEqual(maskRow?.status, .running)
    }

    /// The Settings scene content at the width the app gives it.
    func testSettingsSnapshots() throws {
        for appearance in StudioSnapshotAppearance.allCases {
            let view = MereRunSettingsView()
                .environmentObject(fixture.controller)
                .environmentObject(fixture.crashReporter)
                .frame(width: Self.settingsSize.width)
            try fixture.write(
                view,
                size: Self.settingsSize,
                appearance: appearance,
                name: "settings-\(appearance.rawValue)"
            )
        }
    }

    /// The Command Console window content at its default window size.
    func testCommandConsoleSnapshots() throws {
        for appearance in StudioSnapshotAppearance.allCases {
            let view = AdvancedControlSurface()
                .environmentObject(fixture.controller)
                .environmentObject(fixture.library)
                .environmentObject(NavigationModel())
                .frame(width: Self.consoleSize.width, height: Self.consoleSize.height)
            try fixture.write(
                view,
                size: Self.consoleSize,
                appearance: appearance,
                name: "console-\(appearance.rawValue)"
            )
        }
    }

    private static func snapshotDirectory() -> URL? {
        guard let path = ProcessInfo.processInfo.environment["MERERUN_STUDIO_SNAPSHOT_DIR"],
              !path.trimmingCharacters(in: .whitespaces).isEmpty else {
            return nil
        }
        return URL(fileURLWithPath: NSString(string: path).expandingTildeInPath, isDirectory: true)
    }
}

// MARK: - Fixture

/// A controller, Library, and on-disk artifacts that live only for one test.
@MainActor
private final class SnapshotFixture {
    let outputDirectory: URL
    let root: URL
    let controller: MereRunController
    let library: StudioLibraryStore
    let crashReporter = StudioCrashReporter()
    private let processRunner = SnapshotProcessRunner()

    init(outputDirectory: URL) throws {
        self.outputDirectory = outputDirectory
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("StudioSnapshotTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: outputDirectory, withIntermediateDirectories: true)

        // The first-run banner is dismissed state in the volatile registration domain only, so the
        // shell renders as a returning user sees it without writing to any persistent defaults.
        UserDefaults.standard.register(defaults: ["mererun.app.hasCompletedWelcome": true])

        controller = MereRunController(
            processRunner: processRunner,
            cliResolver: { _ in .executable(URL(fileURLWithPath: "/usr/local/bin/mere.run")) },
            resolvesCLIOnInit: true
        )
        library = StudioLibraryStore(libraryURL: root.appendingPathComponent("library.json"))
        try seedLibrary()
    }

    func tearDown() {
        controller.terminateAllProcesses()
        try? FileManager.default.removeItem(at: root)
    }

    func write<Content: View>(
        _ view: Content,
        size: CGSize,
        appearance: StudioSnapshotAppearance,
        name: String,
        settle: TimeInterval = 1.5,
        afterAppear: (() -> Void)? = nil,
        file: StaticString = #filePath,
        line: UInt = #line
    ) throws {
        let url = outputDirectory.appendingPathComponent("\(name).png", isDirectory: false)
        let rep = try StudioSnapshotRenderer.render(
            view, size: size, appearance: appearance, settle: settle, afterAppear: afterAppear
        )
        let coverage = StudioSnapshotRenderer.nonBlankCoverage(of: rep)
        XCTAssertGreaterThan(
            coverage, 0.05,
            "\(name) rendered blank (non-blank coverage \(coverage))",
            file: file, line: line
        )
        guard let data = rep.representation(using: .png, properties: [:]) else {
            throw StudioSnapshotError.pngEncodingFailed
        }
        try data.write(to: url, options: .atomic)
    }

    // MARK: Seed data

    private func seedLibrary() throws {
        let imageURL = root.appendingPathComponent("lighthouse.png", isDirectory: false)
        try Self.writeFixturePNG(to: imageURL, size: CGSize(width: 768, height: 512))
        let audioURL = root.appendingPathComponent("narration.wav", isDirectory: false)
        try Self.writeSilentWAV(to: audioURL, seconds: 4)

        let now = Date()
        var rows: [StudioLibraryItem] = []

        rows.append(StudioLibraryItem(
            id: UUID(),
            mode: .createImage,
            prompt: "A lighthouse on a basalt shore at dusk, long exposure, film grain",
            inputURL: nil,
            outputURL: imageURL,
            createdAt: now.addingTimeInterval(-60 * 5),
            updatedAt: now.addingTimeInterval(-60 * 4),
            status: .completed,
            exitCode: 0,
            commandPreview: "mere.run image generate --model z-image-turbo --size 768x512 --steps 8",
            outputText: nil,
            artifactURLs: [imageURL]
        ))

        rows.append(StudioLibraryItem(
            id: UUID(),
            mode: .createImage,
            prompt: "Isometric cutaway of a lighthouse lamp room",
            inputURL: nil,
            outputURL: nil,
            createdAt: now.addingTimeInterval(-60 * 40),
            updatedAt: now.addingTimeInterval(-60 * 39),
            status: .failed,
            exitCode: 1,
            commandPreview: "mere.run image generate --model z-image-turbo --size 1024x1024",
            outputText: "error: model z-image-turbo is not installed. Run `mere.run model pull z-image-turbo`."
        ))

        rows.append(StudioLibraryItem(
            id: UUID(),
            mode: .speak,
            prompt: "Welcome aboard. Everything you make here stays on this Mac.",
            inputURL: nil,
            outputURL: audioURL,
            createdAt: now.addingTimeInterval(-60 * 22),
            updatedAt: now.addingTimeInterval(-60 * 21),
            status: .completed,
            exitCode: 0,
            commandPreview: "mere.run speech speak --voice nova --out narration.wav",
            outputText: nil,
            artifactURLs: [audioURL]
        ))

        rows.append(StudioLibraryItem(
            id: UUID(),
            mode: .listen,
            prompt: "",
            inputURL: audioURL,
            outputURL: nil,
            createdAt: now.addingTimeInterval(-60 * 90),
            updatedAt: now.addingTimeInterval(-60 * 89),
            status: .completed,
            exitCode: 0,
            commandPreview: "mere.run speech transcribe narration.wav",
            outputText: "Welcome aboard. Everything you make here stays on this Mac."
        ))

        rows.append(StudioLibraryItem(
            id: UUID(),
            mode: .chat,
            prompt: "",
            inputURL: nil,
            outputURL: nil,
            createdAt: now.addingTimeInterval(-60 * 12),
            updatedAt: now.addingTimeInterval(-60 * 11),
            status: .completed,
            exitCode: 0,
            commandPreview: "mere.run text chat --model gemma4-e4b",
            outputText: nil,
            messages: [
                StudioMessage(
                    role: .user,
                    content: "Explain what a lighthouse Fresnel lens does in two sentences.",
                    createdAt: now.addingTimeInterval(-60 * 12)
                ),
                StudioMessage(
                    role: .assistant,
                    content: """
                    A Fresnel lens folds a thick glass lens into concentric rings so it can bend \
                    light from the lamp into a tight horizontal beam while using far less glass. \
                    That beam is what lets a modest lamp be seen twenty miles offshore.
                    """,
                    createdAt: now.addingTimeInterval(-60 * 11)
                )
            ],
            systemPrompt: nil,
            model: "gemma4-e4b"
        ))

        rows.append(StudioLibraryItem(
            id: UUID(),
            mode: .code,
            prompt: "",
            inputURL: nil,
            outputURL: nil,
            createdAt: now.addingTimeInterval(-60 * 200),
            updatedAt: now.addingTimeInterval(-60 * 199),
            status: .completed,
            exitCode: 0,
            commandPreview: "mere.run text code --model qwen3-coder",
            outputText: nil,
            messages: [
                StudioMessage(role: .user, content: "Write a Swift function that reverses a string.",
                              createdAt: now.addingTimeInterval(-60 * 200)),
                StudioMessage(
                    role: .assistant,
                    content: "```swift\nfunc reversed(_ text: String) -> String {\n    String(text.reversed())\n}\n```",
                    createdAt: now.addingTimeInterval(-60 * 199)
                )
            ],
            systemPrompt: nil,
            model: "qwen3-coder"
        ))

        // Oldest first so upsert's insert-at-front leaves the newest row on top.
        for row in rows.sorted(by: { $0.createdAt < $1.createdAt }) {
            library.upsert(row)
        }
    }

    // MARK: Tracked subjects project

    /// A `skate-clip-01` project with three subjects whose masks were tracked through 240 frames,
    /// laid out the way `video prepare-masks` leaves an output directory: `manifest.json`,
    /// `tracking.json`, `quality.json`, prepared reference images, and an overlay frame. The
    /// re-track job is started through the real controller and Library paths and held open by
    /// the process seam so the job bar shows it running.
    func seedTrackedSubjectsProject() throws -> StudioSubjectsProjectSeed {
        let project = root.appendingPathComponent("skate-clip-01", isDirectory: true)
        let prepared = project.appendingPathComponent("tracked", isDirectory: true)
        try FileManager.default.createDirectory(at: prepared, withIntermediateDirectories: true)
        let drivingVideo = project.appendingPathComponent("skate-clip-01.mp4", isDirectory: false)
        try Data().write(to: drivingVideo)

        let frameCount = 240
        let subjects = [
            StudioSCAILSubject(
                name: "Skater", color: "blue",
                referenceImage: project.appendingPathComponent("skater-front.png").path,
                referencePrompt: "skater", drivingPrompt: "skater"
            ),
            StudioSCAILSubject(
                name: "Board", color: "red",
                referenceImage: project.appendingPathComponent("board.png").path,
                referencePrompt: "skateboard", drivingPositivePoints: "612,480"
            ),
            StudioSCAILSubject(
                name: "Backpack", color: "green",
                referenceImage: project.appendingPathComponent("backpack.png").path,
                referencePrompt: "backpack", drivingBox: "500,200,600,330"
            ),
        ]
        let corrections = [
            StudioSCAILCorrection(subjectID: "Backpack", frameIndex: 88, positivePoints: "540,250"),
            StudioSCAILCorrection(subjectID: "Backpack", frameIndex: 142, box: "505,205,605,335"),
        ]
        let swatches: [(name: String, color: NSColor)] = [
            ("Skater", NSColor(calibratedRed: 0.61, green: 0.48, blue: 0.18, alpha: 1)),
            ("Board", NSColor(calibratedRed: 0.37, green: 0.48, blue: 0.27, alpha: 1)),
            ("Backpack", NSColor(calibratedRed: 0.61, green: 0.46, blue: 0.13, alpha: 1)),
        ]
        for (subject, swatch) in zip(subjects, swatches) {
            try Self.writeSwatchPNG(to: URL(fileURLWithPath: subject.referenceImage), color: swatch.color)
            try Self.writeSwatchPNG(
                to: prepared.appendingPathComponent("reference-\(subject.name)-prepared.png"),
                color: swatch.color
            )
        }
        try Self.writeOverlayFramePNG(
            to: prepared.appendingPathComponent("overlay-frame-88.png"),
            size: CGSize(width: 832, height: 468)
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let manifest = StudioSCAILManifest(
            status: "ready",
            previewFrame: 88,
            drivingSourcePath: drivingVideo.path,
            drivingProxyPath: "driving-proxy.mp4",
            drivingMaskPath: "driving-mask.mov",
            overlayPreviewPath: "overlay-frame-88.png",
            contactSheetPath: "contact-sheet.png",
            trackingPath: "tracking.json",
            qualityPath: "quality.json",
            frameCount: frameCount,
            fps: 24,
            subjects: subjects.map {
                StudioSCAILManifest.Subject(
                    id: $0.name,
                    color: $0.color,
                    preparedReferenceImagePath: "reference-\($0.name)-prepared.png",
                    referenceMaskPath: "reference-\($0.name)-mask.png"
                )
            },
            corrections: corrections.map {
                StudioSCAILManifest.Correction(subjectID: $0.subjectID, frameIndex: $0.frameIndex)
            }
        )
        try encoder.encode(manifest).write(to: prepared.appendingPathComponent("manifest.json"))
        let tracking = StudioSCAILTrackingReport(
            frameCount: frameCount,
            fps: 24,
            subjects: zip(subjects, [240, 231, 240]).map { subject, visible in
                StudioSCAILTrackingReport.Subject(
                    id: subject.name,
                    frames: (0..<frameCount).map { index in
                        StudioSCAILTrackingReport.Frame(
                            frameIndex: index,
                            detections: [
                                StudioSCAILTrackingReport.Detection(visible: index < visible, score: 0.92)
                            ]
                        )
                    }
                )
            }
        )
        try encoder.encode(tracking).write(to: prepared.appendingPathComponent("tracking.json"))
        let quality = StudioSCAILQualityReport(
            blockingErrors: [],
            warnings: [
                StudioSCAILQualityReport.Warning(
                    code: "weak_score", subjectID: "Board", frameIndex: 233,
                    message: "Subject Board has weak mask confidence at frame 233."
                ),
            ]
        )
        try encoder.encode(quality).write(to: prepared.appendingPathComponent("quality.json"))

        // The re-track the user just asked for: a real Library row and controller session.
        guard let template = CommandCatalog.template(id: .videoPrepareMasks) else {
            throw StudioSnapshotError.noContentView
        }
        var draft = template.defaultDraft()
        draft.inputPath = project.appendingPathComponent("plan.json").path
        draft.outputPath = prepared.path
        draft.model = "vision-segment-sam31"
        let request = StudioRunRequest(
            mode: .video,
            templateID: .videoPrepareMasks,
            template: template,
            draft: draft,
            createdAt: Date().addingTimeInterval(-95)
        )
        let preview = controller.commandPreview(template: template, draft: draft, masksSecrets: true)
        library.start(request: request, commandPreview: preview, status: .running)
        processRunner.liveSessionMarker = "prepare-masks"
        guard controller.run(studio: request), let live = processRunner.liveStarts.last else {
            throw StudioSnapshotError.noContentView
        }
        live.stderr("Preparing SCAIL-2 masks from \(draft.inputPath)\n")
        live.stderr("Segmenting 3 reference images with vision-segment-sam31\n")

        var components = Calendar.current.dateComponents([.year, .month, .day], from: Date())
        components.hour = 13
        components.minute = 26
        return StudioSubjectsProjectSeed(
            drivingVideo: drivingVideo.path,
            fps: 24,
            subjects: subjects,
            corrections: corrections,
            preparedDirectory: prepared,
            planSavedAt: Calendar.current.date(from: components),
            stage: .track,
            maskRequestID: request.id
        )
    }

    /// A flat tinted tile standing in for a subject's reference image.
    private static func writeSwatchPNG(to url: URL, color: NSColor) throws {
        try writePNG(to: url, size: CGSize(width: 96, height: 96)) { bounds in
            color.setFill()
            bounds.fill()
        }
    }

    /// A driving frame with each subject's mask tinted over it, as the CLI's overlay preview
    /// shows: a dark studio gradient, the skater as a tall rounded shape, the board under the
    /// feet, the backpack at the shoulder.
    private static func writeOverlayFramePNG(to url: URL, size: CGSize) throws {
        try writePNG(to: url, size: size) { bounds in
            let ground = NSGradient(colors: [
                NSColor(calibratedRed: 0.34, green: 0.34, blue: 0.36, alpha: 1),
                NSColor(calibratedRed: 0.23, green: 0.23, blue: 0.25, alpha: 1),
                NSColor(calibratedRed: 0.16, green: 0.16, blue: 0.18, alpha: 1),
            ])
            ground?.draw(in: bounds, angle: 90)
            let width = bounds.width
            let height = bounds.height
            func rect(_ x: CGFloat, _ y: CGFloat, _ w: CGFloat, _ h: CGFloat) -> CGRect {
                // Fractions of the frame, y from the top as the mockup lays them out.
                CGRect(x: width * x, y: height * (1 - y - h), width: width * w, height: height * h)
            }
            NSColor(calibratedRed: 0.61, green: 0.48, blue: 0.18, alpha: 0.45).setFill()
            NSBezierPath(roundedRect: rect(0.38, 0.22, 0.22, 0.60), xRadius: width * 0.09, yRadius: height * 0.18).fill()
            NSColor(calibratedRed: 0.37, green: 0.48, blue: 0.27, alpha: 0.5).setFill()
            NSBezierPath(roundedRect: rect(0.34, 0.78, 0.32, 0.08), xRadius: 6, yRadius: 6).fill()
            NSColor(calibratedRed: 0.61, green: 0.46, blue: 0.13, alpha: 0.5).setFill()
            NSBezierPath(roundedRect: rect(0.52, 0.32, 0.09, 0.18), xRadius: 6, yRadius: 6).fill()
        }
    }

    private static func writePNG(to url: URL, size: CGSize, draw: (CGRect) -> Void) throws {
        let width = Int(size.width)
        let height = Int(size.height)
        guard let rep = NSBitmapImageRep(
            bitmapDataPlanes: nil,
            pixelsWide: width,
            pixelsHigh: height,
            bitsPerSample: 8,
            samplesPerPixel: 4,
            hasAlpha: true,
            isPlanar: false,
            colorSpaceName: .deviceRGB,
            bytesPerRow: 0,
            bitsPerPixel: 0
        ), let context = NSGraphicsContext(bitmapImageRep: rep) else {
            throw StudioSnapshotError.noBitmap
        }
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = context
        draw(CGRect(x: 0, y: 0, width: width, height: height))
        NSGraphicsContext.restoreGraphicsState()
        guard let data = rep.representation(using: .png, properties: [:]) else {
            throw StudioSnapshotError.pngEncodingFailed
        }
        try data.write(to: url, options: .atomic)
    }

    /// A soft two-tone gradient with a horizon line, so the canvas visibly shows an image.
    private static func writeFixturePNG(to url: URL, size: CGSize) throws {
        let width = Int(size.width)
        let height = Int(size.height)
        guard let rep = NSBitmapImageRep(
            bitmapDataPlanes: nil,
            pixelsWide: width,
            pixelsHigh: height,
            bitsPerSample: 8,
            samplesPerPixel: 4,
            hasAlpha: true,
            isPlanar: false,
            colorSpaceName: .deviceRGB,
            bytesPerRow: 0,
            bitsPerPixel: 0
        ), let context = NSGraphicsContext(bitmapImageRep: rep) else {
            throw StudioSnapshotError.noBitmap
        }
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = context
        let sky = NSGradient(
            starting: NSColor(calibratedRed: 0.98, green: 0.62, blue: 0.36, alpha: 1),
            ending: NSColor(calibratedRed: 0.22, green: 0.20, blue: 0.42, alpha: 1)
        )
        sky?.draw(in: CGRect(x: 0, y: 0, width: width, height: height), angle: 90)
        NSColor(calibratedRed: 0.10, green: 0.11, blue: 0.16, alpha: 1).setFill()
        CGRect(x: 0, y: 0, width: width, height: height / 3).fill()
        NSColor(calibratedWhite: 0.96, alpha: 1).setFill()
        CGRect(x: width * 2 / 3, y: height / 3, width: 18, height: height / 3).fill()
        NSColor(calibratedRed: 1.0, green: 0.93, blue: 0.62, alpha: 1).setFill()
        NSBezierPath(ovalIn: CGRect(x: width * 2 / 3 - 6, y: height * 2 / 3 - 6, width: 30, height: 30)).fill()
        NSGraphicsContext.restoreGraphicsState()
        guard let data = rep.representation(using: .png, properties: [:]) else {
            throw StudioSnapshotError.pngEncodingFailed
        }
        try data.write(to: url, options: .atomic)
    }

    /// A valid 16 kHz mono 16-bit PCM WAV of near-silence with a quiet tone so a waveform draws.
    private static func writeSilentWAV(to url: URL, seconds: Int) throws {
        let sampleRate = 16_000
        let frames = sampleRate * seconds
        var samples = Data(capacity: frames * 2)
        for index in 0..<frames {
            let time = Double(index) / Double(sampleRate)
            let envelope = 0.5 + 0.5 * sin(time * 1.7)
            let value = Int16(sin(time * 2 * .pi * 220) * 6_000 * envelope)
            samples.append(UInt8(truncatingIfNeeded: value & 0xFF))
            samples.append(UInt8(truncatingIfNeeded: (value >> 8) & 0xFF))
        }
        var data = Data()
        func appendLE32(_ value: UInt32) { withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) } }
        func appendLE16(_ value: UInt16) { withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) } }
        data.append(contentsOf: Array("RIFF".utf8))
        appendLE32(UInt32(36 + samples.count))
        data.append(contentsOf: Array("WAVE".utf8))
        data.append(contentsOf: Array("fmt ".utf8))
        appendLE32(16)
        appendLE16(1)
        appendLE16(1)
        appendLE32(UInt32(sampleRate))
        appendLE32(UInt32(sampleRate * 2))
        appendLE16(2)
        appendLE16(16)
        data.append(contentsOf: Array("data".utf8))
        appendLE32(UInt32(samples.count))
        data.append(samples)
        try data.write(to: url, options: .atomic)
    }
}

// MARK: - Process seam

/// Refuses every launch: reports one stderr line and a non-zero exit asynchronously so readiness
/// and status probes settle to their "could not check" states without a CLI ever running.
///
/// The one exception is a launch whose arguments contain `liveSessionMarker`: that is held open
/// as a live session (never terminated, stdin recorded) so Session-archetype views can be
/// rendered mid-run. The test feeds it the lines the CLI would have written.
private final class SnapshotProcessRunner: MereRunProcessRunning, @unchecked Sendable {
    struct LiveStart {
        let configuration: MereRunProcessConfiguration
        let stdout: @Sendable (String) -> Void
        let stderr: @Sendable (String) -> Void
        let process: SnapshotLiveProcess
    }

    private let lock = NSLock()
    private var refusedLaunches = 0
    private var _liveStarts: [LiveStart] = []
    private var _liveSessionMarker: String?

    var refusedLaunchCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return refusedLaunches
    }

    var liveSessionMarker: String? {
        get {
            lock.lock()
            defer { lock.unlock() }
            return _liveSessionMarker
        }
        set {
            lock.lock()
            _liveSessionMarker = newValue
            lock.unlock()
        }
    }

    var liveStarts: [LiveStart] {
        lock.lock()
        defer { lock.unlock() }
        return _liveStarts
    }

    func start(
        configuration: MereRunProcessConfiguration,
        stdout: @escaping @Sendable (String) -> Void,
        stderr: @escaping @Sendable (String) -> Void,
        termination: @escaping @Sendable (Int32) -> Void
    ) throws -> MereRunRunningProcess {
        lock.lock()
        if let marker = _liveSessionMarker, configuration.arguments.contains(marker) {
            let process = SnapshotLiveProcess()
            _liveStarts.append(LiveStart(configuration: configuration, stdout: stdout, stderr: stderr, process: process))
            lock.unlock()
            return process
        }
        refusedLaunches += 1
        lock.unlock()
        DispatchQueue.global().asyncAfter(deadline: .now() + .milliseconds(30)) {
            stderr("Snapshot harness: the CLI is not launched while rendering.\n")
            termination(1)
        }
        return SnapshotRefusedProcess()
    }
}

private final class SnapshotRefusedProcess: MereRunRunningProcess {
    func terminate() {}
}

/// A session the harness holds open: nothing to terminate, stdin lines kept for assertions.
private final class SnapshotLiveProcess: MereRunRunningProcess, @unchecked Sendable {
    private let lock = NSLock()
    private var _standardInputs: [String] = []

    var standardInputs: [String] {
        lock.lock()
        defer { lock.unlock() }
        return _standardInputs
    }

    func terminate() {}

    func sendStandardInput(_ text: String) throws {
        lock.lock()
        _standardInputs.append(text)
        lock.unlock()
    }
}
