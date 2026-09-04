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
    private let processRunner: MereRunProcessRunning
    /// The default runner's live-session seam; nil when the fixture was given a scripted runner.
    private var liveSessionRunner: SnapshotProcessRunner? { processRunner as? SnapshotProcessRunner }

    /// Which rows the temporary Library holds.
    enum Seed {
        /// One row per kind of output (image, audio, transcript, chat, code) across several domains.
        case fixture
        /// The finished Image rows the design boards show; `startMainBoardJobs` adds the live ones.
        case mockup
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
