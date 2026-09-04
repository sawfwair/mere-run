@testable import MereRunApp
import AVFoundation
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
    /// The size the v2 mockups were drawn at; fidelity renders use it so they overlay 1:1.
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

    /// The Main board: Image ▸ Generate at the mockup's 1440×900 with the boards' feed — a
    /// finished generation with two outputs, a run in flight (held open by the process seam at
    /// "Denoising 15/24 · 0:41"), and a queued run behind a concurrent model pull — the inspector
    /// open with two settings changed from the defaults, light and dark; then the same feed with
    /// the Command view column in the inspector's place and the Library hidden, as the Command
    /// board shows it. Readiness is answered from a scripted `model capabilities` and
    /// `model list`, so the composer is live and no readiness card appears.
    func testMainBoardFidelitySnapshots() throws {
        let fidelity = try SnapshotFixture(
            outputDirectory: fixture.outputDirectory,
            seed: .mockup,
            processRunner: SnapshotProcessRunner(script: ModelsInventoryScript.readinessResponses)
        )
        defer { fidelity.tearDown() }
        try fidelity.startMainBoardJobs()

        var draft = StudioDraft()
        draft.reset(for: .createImage)
        draft.prompt = "a ceramic coffee mug in soft morning light"
        draft.cfgScale = 3.5
        draft.sigmaShift = 3.0

        let renders: [(name: String, appearance: StudioSnapshotAppearance, command: Bool)] = [
            ("f3-main-light", .light, false),
            ("f3-main-dark", .dark, false),
            ("f3-command-light", .light, true),
            ("f3-command-dark", .dark, true),
        ]
        for render in renders {
            let navigation = NavigationModel()
            let view = StudioRootView(seededDrafts: [.createImage: draft])
                .environmentObject(fidelity.controller)
                .environmentObject(fidelity.library)
                .environmentObject(navigation)
                .frame(width: Self.fidelitySize.width, height: Self.fidelitySize.height)
            try fidelity.write(
                view,
                size: Self.fidelitySize,
                appearance: render.appearance,
                name: render.name,
                settle: 3.0,
                afterAppear: {
                    if render.command {
                        navigation.showLibrary = false
                        navigation.toggleCommandColumn(for: .imageGenerate)
                    } else {
                        navigation.toggleInspector(for: .imageGenerate)
                    }
                }
            )
        }
    }

    /// The Library column at the mockup's 1440×900 on Image ▸ Generate: list mode light and dark
    /// (which must still read as `Main.png`'s column), grid mode light and dark with the same rows
    /// as three-across thumbnails, and one render with three rows selected so the batch bar shows.
    /// The rows come from the mockup fixture, so the thumbnails are real decoded pictures.
    func testLibraryColumnFidelitySnapshots() throws {
        let fidelity = try SnapshotFixture(
            outputDirectory: fixture.outputDirectory,
            seed: .mockup,
            processRunner: SnapshotProcessRunner(script: ModelsInventoryScript.readinessResponses)
        )
        defer { fidelity.tearDown() }
        // The same board the Main render shows, so the list column can be laid over `Main.png`:
        // two finished generations, one running, one queued.
        try fidelity.startMainBoardJobs()
        try fidelity.seedLibraryColumnVariety()

        var draft = StudioDraft()
        draft.reset(for: .createImage)
        draft.prompt = "a ceramic coffee mug in soft morning light"

        // List mode keeps the mockup's own scope (this domain) so the column reads as `Main.png`;
        // grid and the batch render widen it to All, where every kind of thumbnail is on show.
        let renders: [(name: String, appearance: StudioSnapshotAppearance, seed: StudioLibrarySeed)] = [
            ("b4-library-list-light", .light, StudioLibrarySeed(viewMode: .list)),
            ("b4-library-list-dark", .dark, StudioLibrarySeed(viewMode: .list)),
            ("b4-library-grid-light", .light, StudioLibrarySeed(viewMode: .grid, scope: .all)),
            ("b4-library-grid-dark", .dark, StudioLibrarySeed(viewMode: .grid, scope: .all)),
            ("b4-library-kinds-light", .light, StudioLibrarySeed(viewMode: .list, scope: .all)),
            ("b4-library-multiselect-light", .light, StudioLibrarySeed(viewMode: .list, scope: .all, batchCount: 3)),
        ]
        for render in renders {
            let navigation = NavigationModel()
            let view = StudioRootView(seededDrafts: [.createImage: draft])
                .environmentObject(fidelity.controller)
                .environmentObject(fidelity.library)
                .environmentObject(navigation)
                .environment(\.studioLibrarySeed, render.seed)
                .frame(width: Self.fidelitySize.width, height: Self.fidelitySize.height)
            try fidelity.write(
                view,
                size: Self.fidelitySize,
                appearance: render.appearance,
                name: render.name,
                settle: 3.0,
                afterAppear: { navigation.toggleInspector(for: .imageGenerate) }
            )
        }
    }

    /// The Activity popover over the Main board, as `ActivityDark.png` shows it: the sidebar
    /// footer's popover open on the same three jobs the board runs — the fox generation denoising,
    /// the model pull a quarter of the way through, and the diner generation queued behind them.
    /// Light and dark; the jobs come from the harness's scripted process runner, so no CLI runs.
    func testActivityPopoverFidelitySnapshots() throws {
        let fidelity = try SnapshotFixture(
            outputDirectory: fixture.outputDirectory,
            seed: .mockup,
            processRunner: SnapshotProcessRunner(script: ModelsInventoryScript.readinessResponses)
        )
        defer { fidelity.tearDown() }
        try fidelity.startMainBoardJobs()
        // The popover's footer reports the version handshake, which the shell never probes itself.
        fidelity.controller.refreshCLIVersion()

        var draft = StudioDraft()
        draft.reset(for: .createImage)
        draft.prompt = "a ceramic coffee mug in soft morning light"
        draft.cfgScale = 3.5
        draft.sigmaShift = 3.0

        for appearance in StudioSnapshotAppearance.allCases {
            let navigation = NavigationModel()
            let view = StudioRootView(seededDrafts: [.createImage: draft])
                .environmentObject(fidelity.controller)
                .environmentObject(fidelity.library)
                .environmentObject(navigation)
                .frame(width: Self.fidelitySize.width, height: Self.fidelitySize.height)
            try fidelity.write(
                view,
                size: Self.fidelitySize,
                appearance: appearance,
                name: "f11-activity-\(appearance.rawValue)",
                settle: 3.0,
                afterAppear: {
                    navigation.toggleInspector(for: .imageGenerate)
                    navigation.showActivity = true
                }
            )
        }

        // With nothing in flight the same popover carries the machine's own state instead.
        let idleNavigation = NavigationModel()
        let idle = StudioRootView()
            .environmentObject(fixture.controller)
            .environmentObject(fixture.library)
            .environmentObject(idleNavigation)
            .frame(width: Self.fidelitySize.width, height: Self.fidelitySize.height)
        try fixture.write(
            idle,
            size: Self.fidelitySize,
            appearance: .dark,
            name: "f11-activity-idle-dark",
            settle: 3.0,
            afterAppear: { idleNavigation.showActivity = true }
        )
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

    /// The composer at mockup size with the mockup's sample content: Image ▸ Generate with an
    /// empty well and the sample prompt, and Vision ▸ Find with `mug.png` attached, so the
    /// renders line up with the v2 mockups for review.
    func testComposerFidelitySnapshots() throws {
        var image = StudioDraft()
        image.reset(for: .createImage)
        image.prompt = "a ceramic coffee mug in soft morning light"

        var find = StudioDraft()
        find.reset(for: .findObjects)
        find.prompt = "every coffee cup and what it sits on"
        find.inputPath = fixture.mugURL.path

        let scenes: [(name: String, destination: StudioDestination, drafts: [StudioMode: StudioDraft])] = [
            ("composer-image-generate", StudioTask.imageGenerate.destination, [.createImage: image]),
            ("composer-vision-find", StudioTask.visionFind.destination, [.findObjects: find]),
        ]
        for scene in scenes {
            for appearance in StudioSnapshotAppearance.allCases {
                let navigation = NavigationModel()
                let view = StudioRootView(seededDrafts: scene.drafts)
                    .environmentObject(fixture.controller)
                    .environmentObject(fixture.library)
                    .environmentObject(navigation)
                    .frame(width: Self.fidelitySize.width, height: Self.fidelitySize.height)
                try fixture.write(
                    view,
                    size: Self.fidelitySize,
                    appearance: appearance,
                    name: "\(scene.name)-\(appearance.rawValue)",
                    settle: 2.0,
                    afterAppear: { navigation.open(destination: scene.destination) }
                )
            }
        }
    }

    /// Music ▸ Realtime mid-session: a Magenta RT2 run held open by the process seam (no CLI
    /// starts), the CLI's own frame progress and steering echoes fed back as stderr, and the
    /// recording it would be writing synthesized on disk so the waveform has peaks. Rendered at
    /// the mockup size, light and dark, plus the default window size.
    func testMusicRealtimeSessionSnapshots() throws {
        let requestID = try fixture.seedLiveRealtimeSession()
        let seed = StudioRealtimeSteeringSeed(
            promptA: "slow-burn synthwave, hopeful bridge",
            promptB: "brushed drums, dusty piano",
            blend: 0.35
        )
        let renders: [(name: String, size: CGSize, appearance: StudioSnapshotAppearance)] = [
            ("f7-realtime-light", Self.fidelitySize, .light),
            ("f7-realtime-dark", Self.fidelitySize, .dark),
            ("f7-realtime-compact-light", Self.shellSize, .light)
        ]
        for render in renders {
            let navigation = NavigationModel()
            let view = StudioRootView()
                .environmentObject(fixture.controller)
                .environmentObject(fixture.library)
                .environmentObject(navigation)
                .environment(\.studioRealtimeSteeringSeed, seed)
                .frame(width: render.size.width, height: render.size.height)
            try fixture.write(
                view,
                size: render.size,
                appearance: render.appearance,
                name: render.name,
                settle: 2.0,
                afterAppear: {
                    navigation.open(task: .musicRealtime)
                    self.fixture.steerLiveRealtimeSession(requestID: requestID)
                }
            )
        }
        XCTAssertTrue(fixture.controller.canSteerRealtimeMusic(requestID: requestID))
    }

    /// The Analyze board: Vision ▸ Find at the mockup's 1440×900, with a 1024×1024 image in the
    /// composer's well and a finished `vision ground` run whose `--json-output` document carries
    /// the board's two detections, so the boxes, the label tabs, the detection rows, and the
    /// contextual next steps are all rendered from a real result document. Then Audio ▸ Transcribe
    /// once, to show the same archetype carrying a waveform and a transcript.
    func testAnalyzeBoardFidelitySnapshots() throws {
        let analyze = try SnapshotFixture(
            outputDirectory: fixture.outputDirectory,
            seed: .analyze,
            processRunner: SnapshotProcessRunner(script: ModelsInventoryScript.analyzeReadinessResponses)
        )
        defer { analyze.tearDown() }

        var find = StudioDraft()
        find.reset(for: .findObjects)
        find.prompt = SnapshotFixture.analyzePrompt
        find.inputPath = analyze.largeMugURL.path
        find.visionThreshold = 0.3

        var listen = StudioDraft()
        listen.reset(for: .listen)
        listen.inputPath = analyze.narrationURL.path

        let renders: [(name: String, task: StudioTask, appearance: StudioSnapshotAppearance)] = [
            ("f6-analyze-find-light", .visionFind, .light),
            ("f6-analyze-find-dark", .visionFind, .dark),
            ("f6-analyze-transcribe-light", .audioTranscribe, .light)
        ]
        for render in renders {
            let navigation = NavigationModel()
            let view = StudioRootView(seededDrafts: [.findObjects: find, .listen: listen])
                .environmentObject(analyze.controller)
                .environmentObject(analyze.library)
                .environmentObject(navigation)
                .frame(width: Self.fidelitySize.width, height: Self.fidelitySize.height)
            try analyze.write(
                view,
                size: Self.fidelitySize,
                appearance: render.appearance,
                name: render.name,
                settle: 3.0,
                afterAppear: { navigation.open(task: render.task) }
            )
        }
    }

    /// Chat at the mockup size with the Converse board's threads: the thread list with four
    /// rows, the diffusion thread open (two user turns, a reply with a Python block, and a reply
    /// streaming in), the model and system chips, and the composer's Stop circle. `model list`
    /// and `model capabilities` are scripted so readiness is real; the turn's `text chat` is
    /// held open by the process seam and fed its first words. Light, dark, and the default
    /// window size.
    func testConverseFidelitySnapshots() throws {
        let directory = try XCTUnwrap(Self.snapshotDirectory())
        fixture.tearDown()
        fixture = try SnapshotFixture(
            outputDirectory: directory,
            seed: .converse,
            processRunner: SnapshotProcessRunner(script: ConverseScript.responses)
        )

        var chat = StudioDraft()
        chat.reset(for: .chat)
        chat.model = SnapshotFixture.converseChatModelID
        chat.thinkingMode = .hide

        let renders: [(name: String, size: CGSize, appearance: StudioSnapshotAppearance)] = [
            ("f5-converse-light", Self.fidelitySize, .light),
            ("f5-converse-dark", Self.fidelitySize, .dark),
            ("f5-converse-compact-light", Self.shellSize, .light)
        ]
        for render in renders {
            let navigation = NavigationModel()
            let view = StudioRootView(seededDrafts: [.chat: chat])
                .environmentObject(fixture.controller)
                .environmentObject(fixture.library)
                .environmentObject(navigation)
                .frame(width: render.size.width, height: render.size.height)
            try fixture.write(
                view,
                size: render.size,
                appearance: render.appearance,
                name: render.name,
                settle: 2.5,
                afterAppear: {
                    navigation.open(task: .chatChat)
                    if !self.fixture.controller.runningConversationIDs.contains(SnapshotFixture.converseThreadID) {
                        try? self.fixture.seedLiveChatTurn()
                    }
                }
            )
        }
        XCTAssertTrue(fixture.controller.runningConversationIDs.contains(SnapshotFixture.converseThreadID))
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

    /// The Command Console: the catalog, `image generate` rendered from the contract, and the run
    /// pane. At the mockup size so it can be read against the Command board, whose grouped rows,
    /// monospaced flag column and "Will run" block it follows.
    func testCommandConsoleSnapshots() throws {
        for appearance in StudioSnapshotAppearance.allCases {
            let view = StudioConsoleView()
                .environmentObject(fixture.controller)
                .environmentObject(fixture.library)
                .environmentObject(NavigationModel())
                .frame(width: Self.fidelitySize.width, height: Self.fidelitySize.height)
            try fixture.write(
                view,
                size: Self.fidelitySize,
                appearance: appearance,
                name: "console-\(appearance.rawValue)",
                settle: 2.0,
                afterAppear: {
                    guard let template = CommandCatalog.template(id: .imageGenerate) else { return }
                    self.fixture.controller.select(template)
                    self.fixture.controller.draft.prompt = "a ceramic coffee mug in soft morning light"
                    self.fixture.controller.draft.model = "image-zimage-nano-q4"
                    self.fixture.controller.draft.outputPath =
                        "~/Pictures/mere.run/Image/ceramic-coffee-mug-8813.png"
                }
            )
        }
    }

    /// Models ▸ Installed at the mockup size with a seeded inventory: `model list`,
    /// `model capabilities`, `model storage`, `model info`, `model runtime get`, and
    /// `adapter list` are answered by a scripted runner, and the Library carries the runs the
    /// detail column reads (usage, a quality gate, a benchmark) plus a running composer pull so
    /// the job bar renders. No CLI is launched.
    func testModelsInstalledFidelitySnapshots() throws {
        let directory = try XCTUnwrap(Self.snapshotDirectory())
        fixture.tearDown()
        fixture = try SnapshotFixture(
            outputDirectory: directory,
            processRunner: SnapshotProcessRunner(script: ModelsInventoryScript.responses)
        )
        try fixture.seedModelsLibrary()

        for appearance in StudioSnapshotAppearance.allCases {
            let navigation = NavigationModel()
            let view = StudioRootView()
                .environmentObject(fixture.controller)
                .environmentObject(fixture.library)
                .environmentObject(navigation)
                .frame(width: Self.fidelitySize.width, height: Self.fidelitySize.height)
            try fixture.write(
                view,
                size: Self.fidelitySize,
                appearance: appearance,
                name: "models-installed-\(appearance.rawValue)",
                settle: 3.0,
                afterAppear: { navigation.open(task: .modelsInstalled) }
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
    /// A placeholder "mug.png" for the attachment well, drawn in-test.
    let mugURL: URL
    /// The same picture at 1024×1024, the size the Analyze board's result document is in.
    private(set) var largeMugURL: URL!
    /// A short recording for the Transcribe board's waveform.
    private(set) var narrationURL: URL!
    private let processRunner: MereRunProcessRunning
    /// The default runner's live-session seam; nil when the fixture was given a scripted runner.
    private var liveSessionRunner: SnapshotProcessRunner? { processRunner as? SnapshotProcessRunner }

    /// Which rows the temporary Library holds.
    enum Seed {
        /// One row per kind of output (image, audio, transcript, chat, code) across several domains.
        case fixture
        /// The finished Image rows the design boards show; `startMainBoardJobs` adds the live ones.
        case mockup
        /// The Analyze board's rows: a finished `vision ground` with its result document, an
        /// earlier Read and Segment, and a transcript.
        case analyze
        /// The four Converse threads the design mockups show, the newest with a reply in flight.
        case converse
    }

    init(
        outputDirectory: URL,
        seed: Seed = .fixture,
        processRunner: MereRunProcessRunning = SnapshotProcessRunner()
    ) throws {
        self.outputDirectory = outputDirectory
        self.processRunner = processRunner
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("StudioSnapshotTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: outputDirectory, withIntermediateDirectories: true)
        mugURL = root.appendingPathComponent("mug.png", isDirectory: false)
        try Self.writeMugPNG(to: mugURL, side: 512)

        // The first-run banner is dismissed state in the volatile registration domain only, so the
        // shell renders as a returning user sees it without writing to any persistent defaults.
        UserDefaults.standard.register(defaults: ["mererun.app.hasCompletedWelcome": true])

        controller = MereRunController(
            processRunner: processRunner,
            cliResolver: { _ in .executable(URL(fileURLWithPath: "/usr/local/bin/mere.run")) },
            resolvesCLIOnInit: true
        )
        library = StudioLibraryStore(libraryURL: root.appendingPathComponent("library.json"))
        switch seed {
        case .fixture: try seedLibrary()
        case .mockup: try seedMockupLibrary()
        case .analyze: try seedAnalyzeLibrary()
        case .converse: seedConverseLibrary()
        }
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

    /// The finished Image rows of the Studio v2 design boards: the astronaut generation with two
    /// outputs at 12:43 and the older mug, each with the command it ran so the cards show chips.
    private func seedMockupLibrary() throws {
        guard let template = CommandCatalog.template(id: .imageGenerate) else {
            throw StudioSnapshotError.noContentView
        }
        var rows: [StudioLibraryItem] = []
        for (index, seedRow) in Self.mockupImageRows.enumerated() {
            var outputs: [URL] = []
            for (outputIndex, hue) in seedRow.hues.enumerated() {
                let url = root.appendingPathComponent("mockup-\(index)-\(outputIndex).png", isDirectory: false)
                try Self.writeFixturePNG(to: url, size: CGSize(width: 512, height: 512), hueOffset: hue)
                outputs.append(url)
            }
            var draft = template.defaultDraft()
            draft.prompt = seedRow.prompt
            draft.seed = seedRow.seed
            draft.model = "image-zimage-nano"
            let createdAt = Self.mockupTime(hour: seedRow.hour, minute: seedRow.minute)
            rows.append(StudioLibraryItem(
                id: UUID(),
                mode: .createImage,
                prompt: seedRow.prompt,
                inputURL: nil,
                outputURL: outputs.first,
                createdAt: createdAt,
                updatedAt: createdAt.addingTimeInterval(30),
                status: .completed,
                exitCode: 0,
                commandPreview: "mere.run image generate --model image-zimage-nano --width 1024 --height 1024 --steps 4",
                outputText: nil,
                templateID: .imageGenerate,
                commandDraft: draft,
                artifactURLs: outputs
            ))
        }

        for row in rows.sorted(by: { $0.createdAt < $1.createdAt }) {
            library.upsert(row)
        }
    }

    /// The prompt the Analyze board asks Vision ▸ Find.
    static let analyzePrompt = "every coffee cup and what it sits on"

    /// The Analyze board's rows: a finished `vision ground` whose `--json-output` document holds
    /// the board's two detections in the shape `FalconPerceptionGrounder` writes (normalized
    /// boxes, camelCase keys), the earlier Read and Segment the Library column lists, and a
    /// transcript so Audio ▸ Transcribe renders the same archetype.
    private func seedAnalyzeLibrary() throws {
        guard let groundTemplate = CommandCatalog.template(id: .visionGround) else {
            throw StudioSnapshotError.noContentView
        }
        let directory = root.appendingPathComponent("analyze", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        let mug = directory.appendingPathComponent("mug.png", isDirectory: false)
        try Self.writeMugPNG(to: mug, side: 1_024)
        largeMugURL = mug
        let annotated = directory.appendingPathComponent("mug_grounded.png", isDirectory: false)
        try Self.writeMugPNG(to: annotated, side: 1_024)
        let document = directory.appendingPathComponent("mug_grounded.json", isDirectory: false)
        try Self.groundDocument(input: mug, annotated: annotated, document: document)
            .write(to: document, atomically: true, encoding: .utf8)
        let narration = directory.appendingPathComponent("narration.wav", isDirectory: false)
        try Self.writeSilentWAV(to: narration, seconds: 6)
        narrationURL = narration

        var findDraft = groundTemplate.defaultDraft()
        findDraft.prompt = Self.analyzePrompt
        findDraft.inputPath = mug.path
        findDraft.outputPath = annotated.path
        findDraft.visionJSONOutputPath = document.path
        findDraft.visionThreshold = 0.3

        let findAt = Self.mockupTime(hour: 13, minute: 31)
        var rows: [StudioLibraryItem] = [
            StudioLibraryItem(
                id: UUID(),
                mode: .findObjects,
                prompt: Self.analyzePrompt,
                inputURL: mug,
                outputURL: annotated,
                createdAt: findAt,
                updatedAt: findAt.addingTimeInterval(1.8),
                status: .completed,
                exitCode: 0,
                commandPreview: "mere.run vision ground mug.png --query \"\(Self.analyzePrompt)\"",
                outputText: nil,
                templateID: .visionGround,
                commandDraft: findDraft,
                artifactURLs: [annotated, document]
            ),
            StudioLibraryItem(
                id: UUID(),
                mode: .readImage,
                prompt: "Describe this scene in one paragraph",
                inputURL: mug,
                outputURL: nil,
                createdAt: Self.mockupTime(hour: 13, minute: 31).addingTimeInterval(-60 * 60 * 24),
                updatedAt: Self.mockupTime(hour: 13, minute: 32).addingTimeInterval(-60 * 60 * 24),
                status: .completed,
                exitCode: 0,
                commandPreview: "mere.run vision inspect mug.png",
                outputText: """
                A white ceramic mug sits just left of centre on a warm grey table, lit from the \
                upper right so a soft shadow falls across the saucer beneath it.
                """
            ),
            StudioLibraryItem(
                id: UUID(),
                mode: .segment,
                prompt: "the neon sign",
                inputURL: mug,
                outputURL: nil,
                createdAt: findAt.addingTimeInterval(-60 * 60 * 24 * 5),
                updatedAt: findAt.addingTimeInterval(-60 * 60 * 24 * 5 + 3),
                status: .completed,
                exitCode: 0,
                commandPreview: "mere.run vision segment diner.png --prompt \"the neon sign\""
            ),
            StudioLibraryItem(
                id: UUID(),
                mode: .listen,
                prompt: "",
                inputURL: narration,
                outputURL: nil,
                createdAt: Self.mockupTime(hour: 11, minute: 8),
                updatedAt: Self.mockupTime(hour: 11, minute: 8).addingTimeInterval(2.4),
                status: .completed,
                exitCode: 0,
                commandPreview: "mere.run speech transcribe narration.wav --timestamps",
                outputText: Self.transcriptText,
                templateID: .speechTranscribe
            )
        ]
        rows.sort { $0.createdAt < $1.createdAt }
        for row in rows { library.upsert(row) }
    }

    /// The board's two detections as `vision ground --json-output` writes them: normalized 0…1
    /// boxes, camelCase keys, sorted. `[246, 307, 717, 758]` and `[82, 757, 266, 921]` of a
    /// 1024×1024 image are what the app must show once it scales them back to pixels.
    private static func groundDocument(input: URL, annotated: URL, document: URL) -> String {
        func normalized(_ box: [Int]) -> String {
            let values = box.map { Double($0) / 1_024 }
            return """
            { "x1" : \(values[0]), "y1" : \(values[1]), "x2" : \(values[2]), "y2" : \(values[3]) }
            """
        }
        return """
        {
          "annotatedImagePath" : "\(annotated.path)",
          "detections" : [
            {
              "box" : \(normalized([246, 307, 717, 758])),
              "hw" : { "h" : 0.4404296875, "w" : 0.4599609375 },
              "label" : "coffee cup",
              "score" : 0.94,
              "xy" : { "x" : 0.4702148437, "y" : 0.5200195312 }
            },
            {
              "box" : \(normalized([82, 757, 266, 921])),
              "hw" : { "h" : 0.16015625, "w" : 0.1796875 },
              "label" : "saucer",
              "score" : 0.81,
              "xy" : { "x" : 0.169921875, "y" : 0.8193359375 }
            }
          ],
          "inputImagePath" : "\(input.path)",
          "jsonOutputPath" : "\(document.path)",
          "modelID" : "vision-ground-falcon-perception",
          "queries" : [ "\(analyzePrompt)" ],
          "schemaVersion" : 1
        }
        """
    }

    private static let transcriptText = """
    Welcome aboard. Everything you make here stays on this Mac, and nothing is uploaded.

    [00:00.000 --> 00:02.480] Welcome aboard.
    [00:02.480 --> 00:04.960] Everything you make here stays on this Mac,
    [00:04.960 --> 00:06.000] and nothing is uploaded.
    """

    private static func mockupTime(hour: Int, minute: Int) -> Date {
        let calendar = Calendar.current
        let today = calendar.startOfDay(for: Date())
        return calendar.date(byAdding: DateComponents(hour: hour, minute: minute), to: today) ?? today
    }

    /// The live part of the Main board, through the real controller and Library paths: the fox
    /// generation running (held open by the process seam, fed the CLI's denoising progress and
    /// backdated 41 s), a concurrent model pull taking the second inference slot, and the diner
    /// generation queued behind them. Returns nothing the render needs; the feed reads the store.
    func startMainBoardJobs() throws {
        guard let runner = liveSessionRunner, let pullTemplate = CommandCatalog.template(id: .modelPull) else {
            throw StudioSnapshotError.noContentView
        }
        runner.liveSessionMarkers = ["generate", "pull"]

        var runningDraft = StudioDraft()
        runningDraft.reset(for: .createImage)
        runningDraft.prompt = "A polished obsidian fox figurine on a cobalt plinth, studio light"
        let running = try Self.mockupRequest(mode: .createImage, draft: runningDraft, hour: 12, minute: 44)
        library.start(
            request: running,
            commandPreview: controller.commandPreview(template: running.template, draft: running.draft, masksSecrets: true),
            status: .running
        )
        guard controller.run(studio: running), let live = runner.liveStarts.last,
              let job = controller.jobs.job(requestID: running.id) else {
            throw StudioSnapshotError.noContentView
        }
        live.stderr("Loading image-zimage-nano\n")
        live.stderr("{\"event\":\"progress\",\"stage\":\"denoising\",\"step\":14,\"total_steps\":24}\n")
        job.markRunning(status: job.status, at: Date().addingTimeInterval(-41))

        var pullDraft = pullTemplate.defaultDraft()
        pullDraft.model = ModelsInventoryScript.pullingModelID
        let pull = StudioRunRequest(mode: .readImage, templateID: .modelPull, template: pullTemplate, draft: pullDraft)
        guard controller.run(studio: pull), let pullLive = runner.liveStarts.last else {
            throw StudioSnapshotError.noContentView
        }
        pullLive.stderr("[\(ModelsInventoryScript.pullingModelID)] 25%  1.2 GB / 4.8 GB  9.7 MB/s  ETA 3m 20s\n")

        var queuedDraft = StudioDraft()
        queuedDraft.reset(for: .createImage)
        queuedDraft.prompt = "a rainy diner window at dusk, warm neon"
        let queued = try Self.mockupRequest(mode: .createImage, draft: queuedDraft, hour: 12, minute: 45)
        library.start(
            request: queued,
            commandPreview: controller.commandPreview(template: queued.template, draft: queued.draft, masksSecrets: true),
            status: .queued
        )
        guard controller.run(studio: queued) else { throw StudioSnapshotError.noContentView }
    }

    private static func mockupRequest(mode: StudioMode, draft: StudioDraft, hour: Int, minute: Int) throws -> StudioRunRequest {
        let request = try StudioCommandAdapter.makeRequest(mode: mode, draft: draft)
        return StudioRunRequest(
            id: request.id,
            mode: request.mode,
            templateID: request.templateID,
            template: request.template,
            draft: request.draft,
            createdAt: mockupTime(hour: hour, minute: minute)
        )
    }

    /// One finished run per kind of thumbnail the column draws — a clip (poster frame), a spoken
    /// line (peak silhouette), and a transcript (first line) — plus a star on the mockup's mug, so
    /// the Library renders exercise every branch with real files rather than glyphs.
    func seedLibraryColumnVariety() throws {
        let clip = root.appendingPathComponent("rooftops.mp4", isDirectory: false)
        try Self.writeFixtureMP4(to: clip, size: CGSize(width: 256, height: 144), frames: 12)
        let line = root.appendingPathComponent("welcome.wav", isDirectory: false)
        try Self.writeSilentWAV(to: line, seconds: 3)

        let rows: [StudioLibraryItem] = [
            StudioLibraryItem(
                id: UUID(),
                mode: .video,
                prompt: "A slow pan over wet rooftops at first light",
                inputURL: nil,
                outputURL: clip,
                createdAt: Self.mockupTime(hour: 12, minute: 12),
                updatedAt: Self.mockupTime(hour: 12, minute: 13),
                status: .completed,
                exitCode: 0,
                commandPreview: "mere.run video generate --model video-ltx2 --frames 121",
                outputText: nil,
                templateID: .videoGenerate,
                artifactURLs: [clip]
            ),
            StudioLibraryItem(
                id: UUID(),
                mode: .speak,
                prompt: "Welcome aboard. Everything you make here stays on this Mac.",
                inputURL: nil,
                outputURL: line,
                createdAt: Self.mockupTime(hour: 11, minute: 40),
                updatedAt: Self.mockupTime(hour: 11, minute: 40),
                status: .completed,
                exitCode: 0,
                commandPreview: "mere.run speech synthesize --voice nova",
                outputText: nil,
                templateID: .speechSynthesize,
                artifactURLs: [line]
            ),
            StudioLibraryItem(
                id: UUID(),
                mode: .listen,
                prompt: "",
                inputURL: line,
                outputURL: nil,
                createdAt: Self.mockupTime(hour: 11, minute: 8),
                updatedAt: Self.mockupTime(hour: 11, minute: 8),
                status: .completed,
                exitCode: 0,
                commandPreview: "mere.run speech transcribe welcome.wav --timestamps",
                outputText: Self.transcriptText,
                templateID: .speechTranscribe
            )
        ]
        for row in rows.sorted(by: { $0.createdAt < $1.createdAt }) { library.upsert(row) }

        if let mug = library.items.last(where: { $0.mode == .createImage }) {
            library.setFavorite(id: mug.id, isFavorite: true)
        }
    }

    /// Runs the Models detail column reads: image generations with the seeded default model
    /// (usage and last-run duration), a passed quality gate, a Lite benchmark, and a running
    /// composer-initiated pull that the job bar and list report.
    func seedModelsLibrary() throws {
        let now = Date()
        var rows: [StudioLibraryItem] = []

        for (index, seconds) in [3.4, 3.6, 3.1].enumerated() {
            let started = now.addingTimeInterval(-60 * Double(8 + index * 30))
            rows.append(StudioLibraryItem(
                id: UUID(),
                mode: .createImage,
                prompt: "Product shot of a linen-wrapped ceramic mug, soft window light",
                inputURL: nil,
                outputURL: nil,
                createdAt: started,
                updatedAt: started.addingTimeInterval(seconds),
                status: .completed,
                exitCode: 0,
                commandPreview: "mere.run image generate --model image-zimage-nano --size 1024x1024 --steps 4",
                outputText: nil
            ))
        }

        let gateDate = now.addingTimeInterval(-60 * 60 * 24 * 4)
        rows.append(StudioLibraryItem(
            id: UUID(),
            mode: .chat,
            prompt: "",
            inputURL: nil,
            outputURL: nil,
            createdAt: gateDate,
            updatedAt: gateDate.addingTimeInterval(240),
            status: .completed,
            exitCode: 0,
            commandPreview: "mere.run gate --suite all",
            outputText: "gate: 5 suites passed",
            templateID: .qualityGate
        ))

        var liteDraft = CommandDraft()
        liteDraft.benchmarkSuite = "lite"
        rows.append(StudioLibraryItem(
            id: UUID(),
            mode: .chat,
            prompt: "",
            inputURL: nil,
            outputURL: nil,
            createdAt: gateDate.addingTimeInterval(600),
            updatedAt: gateDate.addingTimeInterval(900),
            status: .completed,
            exitCode: 0,
            commandPreview: "mere.run model benchmark fused --suite lite --json",
            outputText: nil,
            templateID: .modelBenchmarkFused,
            commandDraft: liteDraft
        ))

        var pullDraft = CommandDraft()
        pullDraft.model = ModelsInventoryScript.pullingModelID
        rows.append(StudioLibraryItem(
            id: UUID(),
            mode: .readImage,
            prompt: "",
            inputURL: nil,
            outputURL: nil,
            createdAt: now.addingTimeInterval(-60 * 3),
            updatedAt: now.addingTimeInterval(-60 * 3),
            status: .running,
            exitCode: nil,
            commandPreview: "mere.run model pull \(ModelsInventoryScript.pullingModelID)",
            outputText: nil,
            templateID: .modelPull,
            commandDraft: pullDraft
        ))

        for row in rows.sorted(by: { $0.createdAt < $1.createdAt }) {
            library.upsert(row)
        }
    }

    private struct MockupImageRow {
        let prompt: String
        let seed: String
        let hour: Int
        let minute: Int
        /// One output per hue, so the astronaut card shows two pictures.
        let hues: [CGFloat]
    }

    private static let mockupImageRows: [MockupImageRow] = [
        MockupImageRow(
            prompt: "A tiny brass astronaut watering a bonsai tree, cinematic macro",
            seed: "8812", hour: 12, minute: 43, hues: [0.10, 0.62]
        ),
        MockupImageRow(
            prompt: "a ceramic coffee mug in soft morning light",
            seed: "", hour: 9, minute: 5, hues: [0.55]
        )
    ]

    // MARK: Converse threads

    /// The id of the mockup's open thread, whose last user turn is answered live.
    static let converseThreadID = UUID()
    static let converseChatModelID = "text-chat-qwen3.6-4b"

    /// The Converse board's threads: today's diffusion thread (with a Python block and a turn
    /// awaiting its reply), yesterday's Code thread, and two older chats.
    private func seedConverseLibrary() {
        let calendar = Calendar.current
        let today = calendar.startOfDay(for: Date())
        func at(daysAgo: Int, hour: Int, minute: Int) -> Date {
            calendar.date(byAdding: DateComponents(day: -daysAgo, hour: hour, minute: minute), to: today) ?? today
        }
        let replyAt = at(daysAgo: 0, hour: 13, minute: 20)
        let diffusion = StudioLibraryItem(
            id: Self.converseThreadID,
            mode: .chat,
            prompt: "",
            inputURL: nil,
            outputURL: nil,
            createdAt: at(daysAgo: 0, hour: 13, minute: 19),
            updatedAt: replyAt,
            status: .running,
            exitCode: nil,
            commandPreview: "mere.run text chat",
            outputText: nil,
            customTitle: "Summarize diffusion models in one paragraph",
            messages: [
                StudioMessage(
                    role: .user,
                    content: "Summarize diffusion models in one paragraph, for someone who knows what a neural net is.",
                    createdAt: at(daysAgo: 0, hour: 13, minute: 19)
                ),
                StudioMessage(
                    role: .assistant,
                    content: """
                    A diffusion model learns to reverse a gradual noising process. During training, \
                    images are corrupted with increasing Gaussian noise and a network is taught to \
                    predict that noise at each step. At sampling time it starts from pure noise and \
                    repeatedly subtracts its predicted noise, so structure emerges over a few dozen \
                    steps. Guidance from a text encoder steers each step toward the prompt.

                    ```python
                    x = torch.randn(1, 4, 64, 64)
                    for t_ in scheduler.timesteps:
                        eps = unet(x, t_, cond).sample
                        x = scheduler.step(eps, t_, x).prev_sample
                    ```
                    """,
                    createdAt: replyAt,
                    model: Self.converseChatModelID,
                    tokensPerSecond: 41
                ),
                StudioMessage(
                    role: .user,
                    content: "Why predict the noise instead of the image?",
                    createdAt: replyAt
                )
            ],
            systemPrompt: nil,
            model: Self.converseChatModelID
        )

        func thread(
            title: String,
            reply: String,
            mode: StudioMode,
            model: String,
            daysAgo: Int,
            hour: Int
        ) -> StudioLibraryItem {
            let asked = at(daysAgo: daysAgo, hour: hour, minute: 5)
            return StudioLibraryItem(
                id: UUID(),
                mode: mode,
                prompt: "",
                inputURL: nil,
                outputURL: nil,
                createdAt: asked,
                updatedAt: asked.addingTimeInterval(40),
                status: .completed,
                exitCode: 0,
                commandPreview: mode == .code ? "mere.run text code" : "mere.run text chat",
                outputText: nil,
                messages: [
                    StudioMessage(role: .user, content: title, createdAt: asked),
                    StudioMessage(role: .assistant, content: reply, createdAt: asked.addingTimeInterval(40), model: model)
                ],
                systemPrompt: nil,
                model: model
            )
        }
        let rows = [
            diffusion,
            thread(
                title: "Swift function that formats byte counts",
                reply: "```swift\nfunc formatted(bytes: Int64) -> String {\n    ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)\n}\n```",
                mode: .code, model: "text-code-gemma-4", daysAgo: 1, hour: 16
            ),
            thread(
                title: "Draft a friendly reply declining a meeting",
                reply: "Thanks for the invitation — I can't make Thursday, but I'd be glad to catch up next week.",
                mode: .chat, model: Self.converseChatModelID, daysAgo: 4, hour: 10
            ),
            thread(
                title: "What can I cook with mushrooms, eggs, spinach?",
                reply: "A quick frittata: sauté the mushrooms, wilt the spinach, pour over beaten eggs, and finish under the broiler.",
                mode: .chat, model: Self.converseChatModelID, daysAgo: 6, hour: 19
            )
        ]
        for row in rows.sorted(by: { $0.createdAt < $1.createdAt }) {
            library.upsert(row)
        }
    }

    /// Answers the open thread's last turn live: the turn runs through the real controller and
    /// transcript paths (the process seam holds `text chat` open), and the first words of the
    /// reply arrive on stdout so the transcript shows a streaming turn with its caret.
    func seedLiveChatTurn() throws {
        guard let thread = library.items.first(where: { $0.id == Self.converseThreadID }),
              let runner = liveSessionRunner else {
            throw StudioSnapshotError.noContentView
        }
        var draft = StudioDraft()
        draft.reset(for: .chat)
        draft.model = Self.converseChatModelID
        draft.thinkingMode = .hide
        draft.prompt = ConversationTranscript.render(messages: thread.messages ?? []).prompt
        let request = try StudioCommandAdapter.makeRequest(mode: .chat, draft: draft, conversationID: thread.id)
        runner.liveSessionMarkers = ["chat"]
        guard controller.run(studio: request), let live = runner.liveStarts.last else {
            throw StudioSnapshotError.noContentView
        }
        live.stdout("Sure. The key idea is that noise is easier to predict than")
    }

    // MARK: Live realtime session

    /// Starts a Magenta RT2 session through the real controller and Library paths. The process
    /// seam holds the launch open instead of refusing it, so `canSteerRealtimeMusic` is true and
    /// the view re-attaches on appear exactly as it would to a session the user started. The run
    /// is backdated so the transport clock reads minutes in, and the recording the CLI would be
    /// streaming to disk is synthesized with the writer's unpatched (zero-length) header.
    func seedLiveRealtimeSession() throws -> UUID {
        guard let template = CommandCatalog.template(id: .musicRealtime) else {
            throw StudioSnapshotError.noContentView
        }
        let recordingURL = root.appendingPathComponent("realtime-session.wav", isDirectory: false)
        try Self.writeGrowingFloatWAV(to: recordingURL, seconds: 254)

        var draft = template.defaultDraft()
        draft.prompt = "slow-burn synthwave, hopeful bridge, brushed drums, dusty piano"
        draft.model = "music-magenta-rt2-medium"
        draft.durationSeconds = 300
        draft.outputPath = recordingURL.path
        draft.musicPlay = true
        draft.musicInteractive = true
        draft.musicTemperature = 1.1
        draft.musicTopK = 40
        draft.musicCFGMusicCoCa = 4

        let request = StudioRunRequest(
            mode: .music,
            templateID: .musicRealtime,
            template: template,
            draft: draft,
            createdAt: Date().addingTimeInterval(-249)
        )
        let preview = controller.commandPreview(template: template, draft: draft, masksSecrets: true)
        library.start(request: request, commandPreview: preview, status: .running)

        guard let runner = liveSessionRunner else {
            throw StudioSnapshotError.noContentView
        }
        runner.liveSessionMarkers = ["--interactive"]
        guard controller.run(studio: request), let live = runner.liveStarts.last else {
            throw StudioSnapshotError.noContentView
        }
        live.stderr("Starting Magenta RT2 realtime model music-magenta-rt2-medium\n")
        live.stderr("Interactive steering enabled. Commands: prompt <text> | temp | topk | mc | quit\n")
        live.stderr("Realtime frame 6226/7500\n")
        live.stderr("Realtime frame 6251/7500\n")
        return request.id
    }

    /// What the user has done so far in the session: one prompt steer and the frames since.
    func steerLiveRealtimeSession(requestID: UUID) {
        guard let live = liveSessionRunner?.liveStarts.last else { return }
        controller.submitRealtimeMusicCommand(
            "prompt slow-burn synthwave, hopeful bridge, brushed drums, dusty piano",
            requestID: requestID
        )
        live.stderr("queued prompt\n")
        live.stderr("Realtime frame 6301/7500\n")
        live.stderr("Realtime frame 6326/7500\n")
        live.stderr("Realtime frame 6353/7500\n")
    }

    /// A float32 mono WAV whose RIFF and data sizes are still zero, as `StreamingWAVWriter`
    /// leaves them until it closes. The envelope changes slower than one waveform bar
    /// (~2.6 s of a 96-bar view) so the bars swell and breathe instead of all peaking.
    private static func writeGrowingFloatWAV(to url: URL, seconds: Int) throws {
        let sampleRate = 8_000
        let frames = sampleRate * seconds
        let secondsPerBar = Float(seconds) / 96
        var data = Data(capacity: 44 + frames * 4)
        func appendLE32(_ value: UInt32) { withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) } }
        func appendLE16(_ value: UInt16) { withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) } }
        data.append(contentsOf: Array("RIFF".utf8))
        appendLE32(36)
        data.append(contentsOf: Array("WAVE".utf8))
        data.append(contentsOf: Array("fmt ".utf8))
        appendLE32(16)
        appendLE16(3)
        appendLE16(1)
        appendLE32(UInt32(sampleRate))
        appendLE32(UInt32(sampleRate * 4))
        appendLE16(4)
        appendLE16(32)
        data.append(contentsOf: Array("data".utf8))
        appendLE32(0)
        for index in 0..<frames {
            let time = Float(index) / Float(sampleRate)
            let bar = time / secondsPerBar
            let envelope = 0.15 + 0.85 * abs(sin(bar * 0.37) * 0.7 + sin(bar * 1.3) * 0.3)
            let value = sin(time * 2 * .pi * 220) * envelope
            withUnsafeBytes(of: value.bitPattern.littleEndian) { data.append(contentsOf: $0) }
        }
        try data.write(to: url, options: .atomic)
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
        guard let runner = liveSessionRunner else { throw StudioSnapshotError.noContentView }
        runner.liveSessionMarkers = ["prepare-masks"]
        guard controller.run(studio: request), let live = runner.liveStarts.last else {
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

    /// A real, playable H.264 file: a handful of frames whose hue drifts, so the Library's poster
    /// frame comes from `AVAssetImageGenerator` decoding an actual movie rather than a stub.
    private static func writeFixtureMP4(to url: URL, size: CGSize, frames: Int) throws {
        let writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: Int(size.width),
            AVVideoHeightKey: Int(size.height)
        ])
        input.expectsMediaDataInRealTime = false
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: input,
            sourcePixelBufferAttributes: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32ARGB]
        )
        writer.add(input)
        guard writer.startWriting() else { throw StudioSnapshotError.noBitmap }
        writer.startSession(atSourceTime: .zero)

        for frame in 0..<frames {
            var buffer: CVPixelBuffer?
            guard let pool = adaptor.pixelBufferPool,
                  CVPixelBufferPoolCreatePixelBuffer(nil, pool, &buffer) == kCVReturnSuccess,
                  let pixelBuffer = buffer else {
                throw StudioSnapshotError.noBitmap
            }
            CVPixelBufferLockBaseAddress(pixelBuffer, [])
            if let base = CVPixelBufferGetBaseAddress(pixelBuffer),
               let context = CGContext(
                   data: base,
                   width: Int(size.width),
                   height: Int(size.height),
                   bitsPerComponent: 8,
                   bytesPerRow: CVPixelBufferGetBytesPerRow(pixelBuffer),
                   space: CGColorSpaceCreateDeviceRGB(),
                   bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
               ) {
                let hue = CGFloat(frame) / CGFloat(max(1, frames)) * 0.2 + 0.55
                context.setFillColor(NSColor(calibratedHue: hue, saturation: 0.5, brightness: 0.7, alpha: 1).cgColor)
                context.fill(CGRect(origin: .zero, size: size))
                context.setFillColor(NSColor(calibratedWhite: 0.12, alpha: 1).cgColor)
                context.fill(CGRect(x: 0, y: 0, width: size.width, height: size.height * 0.32))
            }
            CVPixelBufferUnlockBaseAddress(pixelBuffer, [])
            while !input.isReadyForMoreMediaData { usleep(2_000) }
            adaptor.append(pixelBuffer, withPresentationTime: CMTime(value: CMTimeValue(frame), timescale: 12))
        }

        input.markAsFinished()
        let finished = DispatchSemaphore(value: 0)
        writer.finishWriting { finished.signal() }
        finished.wait()
        guard writer.status == .completed else { throw StudioSnapshotError.noBitmap }
    }

    /// A soft two-tone gradient with a horizon line, so the canvas visibly shows an image.
    /// `hueOffset` shifts the palette so several fixtures read as different pictures.
    private static func writeFixturePNG(to url: URL, size: CGSize, hueOffset: CGFloat = 0) throws {
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
        let skyTop = NSColor(calibratedRed: 0.98, green: 0.62, blue: 0.36, alpha: 1)
        let skyBottom = NSColor(calibratedRed: 0.22, green: 0.20, blue: 0.42, alpha: 1)
        let sky = NSGradient(
            starting: NSColor(
                calibratedHue: (skyTop.hueComponent + hueOffset).truncatingRemainder(dividingBy: 1),
                saturation: skyTop.saturationComponent, brightness: skyTop.brightnessComponent, alpha: 1
            ),
            ending: NSColor(
                calibratedHue: (skyBottom.hueComponent + hueOffset).truncatingRemainder(dividingBy: 1),
                saturation: skyBottom.saturationComponent, brightness: skyBottom.brightnessComponent, alpha: 1
            )
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

    /// A white mug with a handle on a warm grey ground, so the well's thumbnail reads as a photo.
    private static func writeMugPNG(to url: URL, side: Int) throws {
        guard let rep = NSBitmapImageRep(
            bitmapDataPlanes: nil,
            pixelsWide: side,
            pixelsHigh: side,
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
        let unit = CGFloat(side)
        let ground = NSGradient(
            starting: NSColor(calibratedRed: 0.86, green: 0.84, blue: 0.80, alpha: 1),
            ending: NSColor(calibratedRed: 0.62, green: 0.60, blue: 0.57, alpha: 1)
        )
        ground?.draw(in: CGRect(x: 0, y: 0, width: unit, height: unit), angle: 75)
        NSColor(calibratedWhite: 0.3, alpha: 0.18).setFill()
        NSBezierPath(ovalIn: CGRect(x: unit * 0.18, y: unit * 0.14, width: unit * 0.62, height: unit * 0.12)).fill()
        NSColor(calibratedWhite: 0.97, alpha: 1).setFill()
        NSBezierPath(
            roundedRect: CGRect(x: unit * 0.26, y: unit * 0.2, width: unit * 0.42, height: unit * 0.5),
            xRadius: unit * 0.05,
            yRadius: unit * 0.05
        ).fill()
        let handle = NSBezierPath(ovalIn: CGRect(x: unit * 0.6, y: unit * 0.3, width: unit * 0.22, height: unit * 0.26))
        handle.lineWidth = unit * 0.05
        NSColor(calibratedWhite: 0.95, alpha: 1).setStroke()
        handle.stroke()
        NSColor(calibratedRed: 0.80, green: 0.70, blue: 0.55, alpha: 1).setFill()
        CGRect(x: unit * 0.26, y: unit * 0.2, width: unit * 0.42, height: unit * 0.1).fill()
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

/// The harness's process runner. In order: a launch whose arguments match a scripted `Response`
/// is answered from the script (asynchronously, so a page's `.task` work sees the ordering a
/// real process would give it); the sidebar's `status --json` probe gets a canned snapshot
/// (server idle, 92 models installed) so the footer renders its resting "Ready" state; a launch
/// carrying one of `liveSessionMarkers` is held open as a live session (never terminated, stdin
/// recorded) so Session views and running feed cards can be rendered mid-run and fed the lines
/// the CLI would have written; everything else is refused with one stderr line and a non-zero
/// exit, so no CLI ever runs while rendering.
private final class SnapshotProcessRunner: MereRunProcessRunning, @unchecked Sendable {
    struct Response {
        let matches: ([String]) -> Bool
        let stdout: String
        let exitCode: Int32
    }

    struct LiveStart {
        let configuration: MereRunProcessConfiguration
        let stdout: @Sendable (String) -> Void
        let stderr: @Sendable (String) -> Void
        let process: SnapshotLiveProcess
    }

    private let script: [Response]
    private let lock = NSLock()
    private var refusedLaunches = 0
    private var _liveStarts: [LiveStart] = []
    private var _liveSessionMarkers: Set<String> = []

    init(script: [Response] = []) {
        self.script = script
    }

    var refusedLaunchCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return refusedLaunches
    }

    static let installedModelCount = 92

    /// What `status --json` prints for an idle server with `installedModelCount` models, so the
    /// sidebar footer settles to "Ready · N models".
    static func statusSnapshot(installedModelCount: Int) -> String {
        let models = (1...installedModelCount).map { "{\"id\":\"model-\($0)\"}" }.joined(separator: ",")
        return "{\"server\":{\"health\":\"down\",\"loadedModels\":[]},\"installedModels\":[\(models)]}\n"
    }

    private static let statusSnapshot = statusSnapshot(installedModelCount: installedModelCount)

    /// Argument tokens that mark a launch to hold open as a live session.
    var liveSessionMarkers: Set<String> {
        get {
            lock.lock()
            defer { lock.unlock() }
            return _liveSessionMarkers
        }
        set {
            lock.lock()
            _liveSessionMarkers = newValue
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
        // The controller prepends `--models-root <path>` when a root is configured; match on the
        // subcommand arguments that follow it.
        var arguments = configuration.arguments
        if arguments.first == "--models-root", arguments.count >= 2 {
            arguments.removeFirst(2)
        }
        if let response = script.first(where: { $0.matches(arguments) }) {
            DispatchQueue.global().asyncAfter(deadline: .now() + .milliseconds(30)) {
                if !response.stdout.isEmpty { stdout(response.stdout) }
                termination(response.exitCode)
            }
            return SnapshotRefusedProcess()
        }
        if arguments.first == "status", arguments.contains("--json") {
            let snapshot = Self.statusSnapshot
            DispatchQueue.global().asyncAfter(deadline: .now() + .milliseconds(30)) {
                stdout(snapshot)
                termination(0)
            }
            return SnapshotRefusedProcess()
        }
        lock.lock()
        if !_liveSessionMarkers.isEmpty, arguments.contains(where: { _liveSessionMarkers.contains($0) }) {
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

/// What the Converse render's CLI reads answer: the mockup's status footer, and the Models
/// board's inventory and capabilities (which list the chat model installed, so Chat is ready).
private enum ConverseScript {
    static var responses: [SnapshotProcessRunner.Response] {
        [
            .init(
                matches: { $0.first == "status" && $0.contains("--json") },
                stdout: SnapshotProcessRunner.statusSnapshot(installedModelCount: SnapshotProcessRunner.installedModelCount),
                exitCode: 0
            ),
            .init(matches: { $0 == ["model", "list"] }, stdout: ModelsInventoryScript.modelList, exitCode: 0),
            .init(
                matches: { $0 == ["model", "capabilities", "--all", "--json"] },
                stdout: ModelsInventoryScript.capabilities,
                exitCode: 0
            ),
        ]
    }
}

/// The model inventory the Models ▸ Installed fidelity render shows: the mockup's sample
/// lineup, expressed the way `mere.run model list`, `model capabilities`, `model storage`,
/// `model info`, `model runtime get`, and `adapter list` print it.
private enum ModelsInventoryScript {
    static let defaultModelID = "image-zimage-nano"
    static let pullingModelID = "vision-chat-qwen3.6-vl-4b"
    /// The rows of `modelList` whose status is installed; the sidebar footer counts the same.
    static let installedModelCount = 7

    static let modelList = """
    ID                         Category     Status     Referenced
    ---------------------------------------------------------------
    image-zimage-nano          image        installed  2.1 GB
    text-chat-qwen3.6-4b       text-chat    installed  2.6 GB
    vision-chat-qwen3.6-vl-4b  vision-chat  missing    —
    video-ltx2-fast            video        installed  9.4 GB
    music-ace-step-1.5         music        installed  3.3 GB
    music-magenta-rt2-medium   music        installed  1.9 GB
    speech-tts-kokoro-82m      speech-tts   installed  330 MB
    speech-asr-parakeet-tdt    speech-asr   installed  2.4 GB

    """

    static let capabilities = """
    {"models": [
      {"id": "image-zimage-nano", "title": "Zimage Nano", "summary": "", "minimumUnifiedMemoryGB": 8, "recommendedUnifiedMemoryGB": 16, "supported": true, "reasons": [], "estimatedDownloadBytes": 2100000000, "sourceRepository": "mere-run/zimage-nano-q4", "publisher": "mere.run"},
      {"id": "text-chat-qwen3.6-4b", "title": "Qwen3.6 4B", "summary": "", "minimumUnifiedMemoryGB": 8, "recommendedUnifiedMemoryGB": 16, "supported": true, "reasons": [], "estimatedDownloadBytes": 2600000000, "sourceRepository": "mere-run/qwen3.6-4b-q4", "publisher": "mere.run"},
      {"id": "vision-chat-qwen3.6-vl-4b", "title": "Qwen3.6-VL 4B", "summary": "", "minimumUnifiedMemoryGB": 8, "recommendedUnifiedMemoryGB": 16, "supported": true, "reasons": [], "estimatedDownloadBytes": 4800000000, "sourceRepository": "mere-run/qwen3.6-vl-4b-q4", "publisher": "mere.run"},
      {"id": "video-ltx2-fast", "title": "LTX-2 Fast", "summary": "", "minimumUnifiedMemoryGB": 16, "recommendedUnifiedMemoryGB": 32, "supported": true, "reasons": [], "estimatedDownloadBytes": 9400000000, "sourceRepository": "mere-run/ltx2-fast", "publisher": "mere.run"},
      {"id": "music-ace-step-1.5", "title": "ACE-Step 1.5", "summary": "", "minimumUnifiedMemoryGB": 8, "recommendedUnifiedMemoryGB": 16, "supported": true, "reasons": [], "estimatedDownloadBytes": 3300000000, "sourceRepository": "mere-run/ace-step-1.5", "publisher": "mere.run"},
      {"id": "music-magenta-rt2-medium", "title": "Magenta RT2 medium", "summary": "", "minimumUnifiedMemoryGB": 8, "recommendedUnifiedMemoryGB": 16, "supported": true, "reasons": [], "estimatedDownloadBytes": 1900000000, "sourceRepository": "mere-run/magenta-rt2-medium", "publisher": "mere.run"},
      {"id": "speech-tts-kokoro-82m", "title": "Kokoro 82M", "summary": "", "minimumUnifiedMemoryGB": 4, "recommendedUnifiedMemoryGB": 8, "supported": true, "reasons": [], "estimatedDownloadBytes": 330000000, "sourceRepository": "mere-run/kokoro-82m", "publisher": "mere.run"},
      {"id": "speech-asr-parakeet-tdt", "title": "Parakeet TDT", "summary": "", "minimumUnifiedMemoryGB": 16, "recommendedUnifiedMemoryGB": 16, "supported": false, "reasons": ["Needs 16 GB of unified memory; this Mac has 8 GB."], "estimatedDownloadBytes": 2400000000, "sourceRepository": "mere-run/parakeet-tdt", "publisher": "mere.run"}
    ]}
    """

    static let storage = """
    {"applicationSupportBytes": 48000000000, "garbageCollectableBytes": 0, "models": []}
    """

    static var modelInfo: String {
        let root = NSHomeDirectory() + "/Library/Application Support/MereRun/models/image-zimage-nano"
        return """
        Model Root: \(root)
        Model ID: image-zimage-nano
        Source: primary
        Ownership: primary-managed

        Manifest (local)
          schemaVersion: 2
          id: image-zimage-nano
          engine: mlx
          precision: bf16
          quantization: bits=4 groupSize=64 scheme=affine
          upstreamRepoId: mere-run/zimage-nano-q4

        Validation
          isValid: true

        """
    }

    static let adapters = """
    {"schemaVersion": 1, "adapterStore": "/tmp/adapters", "adapters": [
      {"id": "linen-still-life-v2", "title": "Linen still life", "version": "2", "summary": "", "baseModelID": "image-zimage-nano", "format": "lora", "license": "MIT", "byteCount": 48000000, "installed": true, "path": "/tmp/adapters/linen-still-life-v2"},
      {"id": "bronze-product-shots", "title": "Bronze product shots", "version": "1", "summary": "", "baseModelID": "image-zimage-nano", "format": "lora", "license": "MIT", "byteCount": 44000000, "installed": true, "path": "/tmp/adapters/bronze-product-shots"}
    ]}
    """

    /// Only what a prompt mode's readiness check and model chip read, leaving the status probe
    /// to the runner's default so the footer keeps the boards' "Ready · 92 models".
    static var readinessResponses: [SnapshotProcessRunner.Response] {
        [
            .init(matches: { $0 == ["model", "list"] }, stdout: modelList, exitCode: 0),
            .init(matches: { $0 == ["model", "capabilities", "--all", "--json"] }, stdout: capabilities, exitCode: 0),
            version,
        ]
    }

    /// `model list` and `model capabilities` with Vision ▸ Find's model installed, so the Analyze
    /// board renders its result rather than a readiness card.
    static var analyzeReadinessResponses: [SnapshotProcessRunner.Response] {
        let extraModels = [
            (id: "vision-ground-falcon-perception", category: "vision-ground", title: "Falcon Perception"),
            (id: "speech-asr-parakeet", category: "speech-asr", title: "Parakeet")
        ]
        let list = modelList.replacingOccurrences(
            of: "image-zimage-nano          image        installed  2.1 GB",
            with: (["image-zimage-nano          image        installed  2.1 GB"]
                + extraModels.map { "\($0.id)  \($0.category)  installed  1.8 GB" })
                .joined(separator: "\n")
        )
        let extraCapabilities = extraModels.map { model in
            """
            ,{"id": "\(model.id)", "title": "\(model.title)", "summary": "", \
            "minimumUnifiedMemoryGB": 8, "recommendedUnifiedMemoryGB": 16, "supported": true, \
            "reasons": [], "estimatedDownloadBytes": 1800000000, \
            "sourceRepository": "mere-run/\(model.id)", "publisher": "mere.run"}
            """
        }.joined(separator: "\n")
        let capabilityJSON = capabilities.replacingOccurrences(
            of: "\n]}", with: "\n\(extraCapabilities)\n]}"
        )
        return [
            .init(matches: { $0 == ["model", "list"] }, stdout: list, exitCode: 0),
            .init(matches: { $0 == ["model", "capabilities", "--all", "--json"] }, stdout: capabilityJSON, exitCode: 0),
        ]
    }

    /// The CLI's own version, answered with the app's, so the Activity popover's footer shows the
    /// matched handshake a bundled CLI produces.
    static var version: SnapshotProcessRunner.Response {
        let bundled = Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String
        return .init(matches: { $0 == ["--version"] }, stdout: (bundled ?? "dev") + "\n", exitCode: 0)
    }

    static var responses: [SnapshotProcessRunner.Response] {
        [
            .init(
                matches: { $0.first == "status" && $0.contains("--json") },
                stdout: SnapshotProcessRunner.statusSnapshot(installedModelCount: installedModelCount),
                exitCode: 0
            ),
            .init(matches: { $0 == ["model", "list"] }, stdout: modelList, exitCode: 0),
            .init(matches: { $0 == ["model", "capabilities", "--all", "--json"] }, stdout: capabilities, exitCode: 0),
            .init(matches: { $0 == ["model", "storage", "--json"] }, stdout: storage, exitCode: 0),
            .init(matches: { $0.starts(with: ["model", "info", defaultModelID]) }, stdout: modelInfo, exitCode: 0),
            .init(matches: { $0.starts(with: ["model", "runtime", "get"]) }, stdout: "{\"pinned\": false}\n", exitCode: 0),
            .init(matches: { $0 == ["adapter", "list", "--json"] }, stdout: adapters, exitCode: 0),
        ]
    }
}
