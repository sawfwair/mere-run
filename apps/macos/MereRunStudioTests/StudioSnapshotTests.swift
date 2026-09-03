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
private final class SnapshotProcessRunner: MereRunProcessRunning, @unchecked Sendable {
    private let lock = NSLock()
    private var refusedLaunches = 0

    var refusedLaunchCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return refusedLaunches
    }

    func start(
        configuration: MereRunProcessConfiguration,
        stdout: @escaping @Sendable (String) -> Void,
        stderr: @escaping @Sendable (String) -> Void,
        termination: @escaping @Sendable (Int32) -> Void
    ) throws -> MereRunRunningProcess {
        lock.lock()
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
