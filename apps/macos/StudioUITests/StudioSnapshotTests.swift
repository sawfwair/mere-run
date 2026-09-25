@testable import StudioKit
@testable import StudioUI
import StudioTestSupport
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
                        navigation.toggleCommandColumn()
                    } else {
                        navigation.toggleInspector(for: .imageGenerate)
                    }
                }
            )
        }
    }

    /// Model scope across the prompt modes, over the shipped contract with no CLI to ask: each
    /// draft holds values its model does not use or runs its own way, so the inspector opens on
    /// the model's own options with the note at its top, light and dark. FastH3 with an end
    /// image, timings, and 30 steps (and the Command view listing them under "Not sent"); FL2VA
    /// with source audio and timings; YuE2 set to cover a source song; Magenta with ACE-Step's
    /// steps and seed; Krea 2 with references and CFG; Qwen-Image-Edit with an input image;
    /// text-only Gemma 4 with an image and top-k; Qwen3-ASR on French; and, in the Command
    /// Console, GLM-OCR with LightOnOCR's model and budget. Then the note's states on their own, light and dark.
    func testModelScopeSnapshots() throws {
        let installed: [(id: String, category: String, title: String)] = [
            ("video-minimax-h3-fasth3-vsa-datafree-mlx", "video", "FastH3"),
            ("video-minimax-h3-fl2va-mlx", "video", "MiniMax-H3 FL2VA"),
            ("music-yue2", "music", "YuE2"),
            ("music-magenta-rt2-small", "music", "Magenta RealTime 2 Small"),
            ("image-krea2-raw", "image", "Krea 2 Raw"),
            ("image-qwen-edit-2511", "image", "Qwen-Image-Edit"),
            ("text-chat-gemma4-12b-4bit", "text-chat", "Gemma 4 12B"),
            ("vision-ocr-lighton", "vision-ocr", "LightOnOCR"),
            ("speech-asr-qwen3", "speech-asr", "Qwen3-ASR"),
        ]
        // The mockup Library holds Image rows only, so Chat opens on a new thread with the draft's model.
        let scoped = try SnapshotFixture(
            outputDirectory: fixture.outputDirectory,
            seed: .mockup,
            processRunner: SnapshotProcessRunner(script: ModelsInventoryScript.readinessResponses(installing: installed))
        )
        defer { scoped.tearDown() }
        let source = StudioScopeSource(identities: StudioFixedModelIdentities())

        func draft(_ mode: StudioMode, _ edit: (inout StudioDraft) -> Void) -> StudioDraft {
            var draft = StudioDraft()
            draft.reset(for: mode)
            edit(&draft)
            return draft
        }
        let cases: [(name: String, mode: StudioMode, task: StudioTask, draft: StudioDraft)] = [
            ("scope-video-fasth3", .video, .videoGenerate, draft(.video) {
                $0.prompt = "A lighthouse at dusk, waves breaking on the rocks"
                $0.model = "video-minimax-h3-fasth3-vsa-datafree-mlx"
                $0.endImagePath = "/tmp/lighthouse-end.png"
                $0.timings = true
                $0.h3Steps = 30
            }),
            ("scope-video-fl2va", .video, .videoGenerate, draft(.video) {
                $0.prompt = "A brass robot walking through fog"
                $0.model = "video-minimax-h3-fl2va-mlx"
                $0.inputPath = "/tmp/robot-start.png"
                $0.audioPath = "/tmp/footsteps.wav"
                $0.timings = true
            }),
            ("scope-music-yue2", .music, .musicCompose, draft(.music) {
                $0.prompt = "warm indie folk with a whistled hook"
                $0.model = "music-yue2"
                $0.musicTask = "cover"
                $0.musicSourceAudio = "/tmp/demo.wav"
            }),
            ("scope-music-magenta", .music, .musicCompose, draft(.music) {
                $0.prompt = "slow ambient pads"
                $0.model = "music-magenta-rt2-small"
                $0.musicOverrideSteps = true
                $0.steps = 50
                $0.seed = "7"
            }),
            ("scope-image-krea", .createImage, .imageGenerate, draft(.createImage) {
                $0.prompt = "editorial portrait, window light"
                $0.model = "image-krea2-raw"
                $0.referenceImagePaths = "/tmp/person.png"
                $0.cfgScale = 4
            }),
            ("scope-image-qwen-edit", .createImage, .imageGenerate, draft(.createImage) {
                $0.prompt = "replace the sky with a storm front"
                $0.model = "image-qwen-edit-2511"
                $0.inputPath = "/tmp/field.png"
            }),
            ("scope-chat-gemma", .chat, .chatChat, draft(.chat) {
                $0.model = "text-chat-gemma4-12b-4bit"
                $0.inputPath = "/tmp/receipt.png"
                $0.topK = 20
            }),
            ("scope-transcribe-qwen", .listen, .audioTranscribe, draft(.listen) {
                $0.model = "speech-asr-qwen3"
                $0.inputPath = "/tmp/interview.wav"
                $0.language = "fr"
            }),
        ]
        for item in cases {
            for appearance in StudioSnapshotAppearance.allCases {
                try render(item.draft, mode: item.mode, task: item.task, command: false,
                           name: "\(item.name)-\(appearance.rawValue)", appearance: appearance)
            }
        }
        try render(cases[0].draft, mode: .video, task: .videoGenerate, command: true,
                   name: "scope-video-fasth3-command-light", appearance: .light)

        // A local folder while the CLI identifies it, and once it could not: every option shows,
        // none locked to another model's value, and the note says why.
        let folder = "/tmp/checkpoints/my-ltx-folder"
        for (state, identity) in [("pending", StudioModelIdentity.pending), ("failed", .unidentified)] {
            let folderSource = StudioScopeSource(identities: StudioFixedModelIdentities([folder: identity]))
            let folderDraft = draft(.video) {
                $0.prompt = "A lighthouse at dusk, waves breaking on the rocks"
                $0.model = folder
            }
            for appearance in StudioSnapshotAppearance.allCases {
                try render(folderDraft, mode: .video, task: .videoGenerate, command: false,
                           name: "scope-video-folder-\(state)-\(appearance.rawValue)", appearance: appearance, source: folderSource)
            }
        }

        // OCR picks its backend in the Command Console: GLM-OCR reads neither LightOnOCR's model
        // nor its token budget, which the console lists under the note.
        if let ocr = CommandCatalog.template(id: .visionOCR) {
            scoped.controller.select(ocr)
            scoped.controller.consoleSeedArguments = [
                "vision", "ocr", "/tmp/receipt.png", "--backend", "glm", "--model", "vision-ocr-lighton", "--max-tokens", "4096",
            ]
        }
        for appearance in StudioSnapshotAppearance.allCases {
            let view = StudioConsoleView()
                .environment(\.studioScopeSource, source)
                .environmentObject(scoped.controller)
                .environmentObject(scoped.library)
                .environmentObject(NavigationModel())
                .frame(width: Self.fidelitySize.width, height: Self.fidelitySize.height)
            try scoped.write(view, size: Self.fidelitySize, appearance: appearance,
                             name: "scope-ocr-glm-\(appearance.rawValue)", settle: 2.0)
        }

        func render(
            _ draft: StudioDraft, mode: StudioMode, task: StudioTask, command: Bool,
            name: String, appearance: StudioSnapshotAppearance, source overriding: StudioScopeSource? = nil
        ) throws {
            let navigation = NavigationModel()
            let view = StudioRootView(seededDrafts: [mode: draft])
                .environment(\.studioScopeSource, overriding ?? source)
                .environmentObject(scoped.controller)
                .environmentObject(scoped.library)
                .environmentObject(navigation)
                .frame(width: Self.fidelitySize.width, height: Self.fidelitySize.height)
            try scoped.write(
                view, size: Self.fidelitySize, appearance: appearance, name: name, settle: 2.5,
                afterAppear: {
                    navigation.open(task: task)
                    if command {
                        navigation.showLibrary = false
                        navigation.toggleCommandColumn()
                    } else {
                        navigation.toggleInspector(for: task)
                    }
                }
            )
        }

        let notes = VStack(alignment: .leading, spacing: 14) {
            StudioScopeNote(notice: StudioScopeNotice(
                kind: .identifying, title: "Identifying h3-ref2va-local…",
                details: ["Every option shows until mere.run knows which model this is."]
            ))
            StudioScopeNote(notice: StudioScopeNotice(
                kind: .unidentified, title: "mere.run couldn't identify my-checkpoint",
                details: ["Every option is shown, and the CLI checks them when it runs."]
            ))
            StudioScopeNote(notice: StudioScopeNotice(
                kind: .unused, title: "Not used by FastH3: End image, Timings.",
                details: ["Denoising steps: FastH3 runs 5; your 30 is kept.", "Your values are kept for when you switch back."]
            ))
            StudioScopeNote(
                notice: StudioScopeNotice(kind: .unused, title: "Not used by YuE2: Source audio, Task type.", details: []),
                eyebrow: "Not sent"
            )
        }
        .padding(16)
        .frame(width: 320, alignment: .topLeading)
        .background(MereRunTheme.background)
        for (name, appearance) in [("scope-note-light", StudioSnapshotAppearance.light), ("scope-note-dark", .dark)] {
            try scoped.write(notes, size: CGSize(width: 320, height: 380), appearance: appearance, name: name)
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

    /// The idle Activity popover when the inventory skipped a model drive that did not answer.
    func testActivityPopoverSkippedModelLocationSnapshots() throws {
        let status = StudioMachineStatus.ready(
            installedModels: 88,
            skippedLocations: [StudioSkippedModelLocation(path: "/Volumes/MODELS", problem: .unresponsive)]
        )
        let size = CGSize(width: StudioActivityPopover.width + 40, height: 360)
        for appearance in StudioSnapshotAppearance.allCases {
            let view = StudioActivityPopover(
                jobs: fixture.controller.jobs,
                status: status,
                appVersion: "1.0",
                cliVersion: "0.55.0",
                modelsRoot: "~/Library/Application Support/MereRun/models",
                resolvedCLI: "/Applications/MereRun.app/Contents/MacOS/mere.run",
                onOpenServer: {},
                onOpenModels: {}
            )
            .padding(20)
            .frame(width: size.width, height: size.height, alignment: .top)
            .background(MereRunTheme.background)
            try fixture.write(view, size: size, appearance: appearance, name: "activity-skipped-location-\(appearance.rawValue)")
        }
    }

    /// The menu bar extra's panel in each server state, light and dark: stopped; running with two
    /// resident text models, a speech sidecar, live traffic, and Studio work in flight; running
    /// outside Studio; and stopped unexpectedly. `/runtime/status` is answered by
    /// `SnapshotRuntimeEndpoint`, so a server running on this machine never shows up.
    func testMenuBarPanelSnapshots() throws {
        SnapshotRuntimeEndpoint.install()
        defer { SnapshotRuntimeEndpoint.uninstall() }
        // The panel as it drops from the menu bar: a card over the desktop.
        let size = CGSize(width: StudioMenuBarPanel.width + 40, height: 600)

        // Every render polls the stub, whose token count never moves, so each render sets the
        // decode history it shows after that poll rather than inheriting the last render's.
        func render(
            _ name: String,
            fixture: SnapshotFixture,
            answer: SnapshotRuntimeEndpoint.Answer,
            throughput: [Double] = []
        ) throws {
            let controller = fixture.controller
            XCTAssertEqual(controller.runtimeHost, "127.0.0.1")
            XCTAssertEqual(controller.runtimePort, 8_080)
            let panel = StudioMenuBarPanel(controller: controller, onOpenStudio: {}, onOpenServer: {})
                .clipShape(RoundedRectangle(cornerRadius: MereRunTheme.Radius.lg))
                .overlay {
                    RoundedRectangle(cornerRadius: MereRunTheme.Radius.lg)
                        .strokeBorder(MereRunTheme.border, lineWidth: 1)
                }
                .mereShadow(radius: 12, y: 8)
                .padding(20)
                .frame(width: size.width, height: size.height, alignment: .top)
                .background(MereRunTheme.background)
            for appearance in [StudioSnapshotAppearance.light, .dark] {
                SnapshotRuntimeEndpoint.answer = answer
                try fixture.write(
                    panel,
                    size: size,
                    appearance: appearance,
                    name: "menu-bar-\(name)-\(appearance.rawValue)",
                    settle: 1.5,
                    afterAppear: {
                        Task {
                            await controller.servingMonitor.refreshRuntimeNow(controller: controller)
                            controller.servingMonitor.throughputHistory = throughput
                        }
                    }
                )
            }
        }

        // The status item glyph itself, at 8× so its period can be judged: lit and idle.
        for isServing in [true, false] {
            let image = StudioMenuBarIcon.image(isServing: isServing)
            let rep = try XCTUnwrap(NSBitmapImageRep(
                bitmapDataPlanes: nil, pixelsWide: 176, pixelsHigh: 176, bitsPerSample: 8, samplesPerPixel: 4,
                hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0
            ))
            NSGraphicsContext.saveGraphicsState()
            NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
            NSColor.white.setFill()
            NSRect(x: 0, y: 0, width: 176, height: 176).fill()
            image.draw(in: NSRect(x: 0, y: 0, width: 176, height: 176))
            NSGraphicsContext.restoreGraphicsState()
            let data = try XCTUnwrap(rep.representation(using: .png, properties: [:]))
            try data.write(to: fixture.outputDirectory.appendingPathComponent("menu-bar-icon-\(isServing ? "serving" : "idle").png"))
        }

        let stopped = try SnapshotFixture(outputDirectory: fixture.outputDirectory, machineMonitor: Self.scriptedMachine())
        defer { stopped.tearDown() }
        try render("stopped", fixture: stopped, answer: .unreachable)

        let runningRunner = SnapshotProcessRunner(script: ModelsInventoryScript.readinessResponses)
        let running = try SnapshotFixture(
            outputDirectory: fixture.outputDirectory,
            seed: .mockup,
            processRunner: runningRunner,
            machineMonitor: Self.scriptedMachine(busy: true)
        )
        // Two minutes of decode traffic: quiet, a burst of chat turns, quiet, then a steady client.
        let traffic: [Double] = (0..<StudioMachineMonitor.historyLength).map { index in
            let t = Double(index)
            switch index {
            case 8..<20: return 32 + 9 * sin(t / 1.7)
            case 38...: return 41 + 5 * sin(t / 2.3)
            default: return 0
            }
        }
        defer { running.tearDown() }
        try running.startMainBoardJobs()
        runningRunner.liveSessionMarkers.insert("serve")
        XCTAssertNil(running.controller.localServer.start())
        let vision = try XCTUnwrap(CommandCatalog.template(id: .visionServe)).defaultDraft()
        running.controller.visionServer.start(draft: vision)
        try render("running", fixture: running, answer: .runtime(SnapshotRuntimeEndpoint.busyRuntime), throughput: traffic)

        let external = try SnapshotFixture(
            outputDirectory: fixture.outputDirectory,
            machineMonitor: Self.scriptedMachine(thermal: .serious)
        )
        defer { external.tearDown() }
        try render(
            "external",
            fixture: external,
            answer: .runtime(SnapshotRuntimeEndpoint.busyRuntime),
            throughput: Array(repeating: 0, count: 12)
        )

        let failedRunner = SnapshotProcessRunner()
        failedRunner.liveSessionMarkers = ["serve"]
        let failed = try SnapshotFixture(outputDirectory: fixture.outputDirectory, processRunner: failedRunner)
        defer { failed.tearDown() }
        XCTAssertNil(failed.controller.localServer.start())
        let serve = try XCTUnwrap(failedRunner.liveStarts.last)
        serve.stderr("Error: 127.0.0.1:8080 is already in use by another process.\n")
        serve.termination(1)
        try render("failed", fixture: failed, answer: .unreachable)
    }

    /// Two minutes of a 64 GB Mac's load, the same on every render: CPU idling around 15–30%, with
    /// a sustained climb near the end when `busy`.
    private static func scriptedMachine(
        busy: Bool = false,
        thermal: ProcessInfo.ThermalState = .nominal
    ) -> StudioMachineMonitor {
        var index = 0
        let monitor = StudioMachineMonitor(sampler: {
            defer { index += 1 }
            let t = Double(index)
            let load = busy && index > 38 ? 0.32 + 0.1 * sin(t / 2) : 0
            return StudioMachineMonitor.Sample(
                cpu: 0.2 + 0.07 * sin(t / 4.5) + 0.03 * sin(t * 1.3) + load,
                memoryUsedBytes: busy ? 44_023_414_784 : 29_527_900_160,
                memoryTotalBytes: 68_719_476_736,
                thermalState: thermal
            )
        })
        for _ in 0..<StudioMachineMonitor.historyLength { monitor.sampleNow() }
        return monitor
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
    /// The Server domain with Studio's API server running against a busy runtime: Serving on its
    /// Overview, Telemetry, and Configuration sections, then the Vision server and Music server
    /// tasks with their servers running and a few log lines.
    func testServerDomainSnapshots() throws {
        SnapshotRuntimeEndpoint.install()
        defer { SnapshotRuntimeEndpoint.uninstall() }
        SnapshotRuntimeEndpoint.answer = .runtime(SnapshotRuntimeEndpoint.busyRuntime)

        let runner = SnapshotProcessRunner()
        runner.liveSessionMarkers = ["serve"]
        let server = try SnapshotFixture(outputDirectory: fixture.outputDirectory, processRunner: runner)
        defer { server.tearDown() }
        let controller = server.controller
        XCTAssertNil(controller.localServer.start())
        controller.visionServer.start(draft: try XCTUnwrap(CommandCatalog.template(id: .visionServe)).defaultDraft())
        controller.musicServer.start(draft: try XCTUnwrap(CommandCatalog.template(id: .musicServe)).defaultDraft())
        for start in runner.liveStarts.dropFirst() {
            start.stderr("Loading model…\n")
            start.stdout("Listening on 127.0.0.1 — ready for requests\n")
        }

        let renders: [(name: String, task: StudioTask, section: StudioServingSection?)] = [
            ("server-serving-overview", .serverServing, .overview),
            ("server-serving-telemetry", .serverServing, .telemetry),
            ("server-serving-configuration", .serverServing, .configuration),
            ("server-vision", .serverVision, nil),
            ("server-music", .serverMusic, nil),
        ]
        for render in renders {
            if let section = render.section {
                controller.taskSessions.set(section, for: StudioTask.serverServing.rawValue + ".ServingConsole.section")
            }
            let navigation = NavigationModel()
            let view = StudioRootView()
                .environmentObject(controller)
                .environmentObject(server.library)
                .environmentObject(navigation)
                .frame(width: Self.shellSize.width, height: Self.shellSize.height)
            try server.write(
                view,
                size: Self.shellSize,
                appearance: .light,
                name: render.name,
                settle: 2.0,
                afterAppear: {
                    navigation.open(destination: render.task.destination)
                    Task { await controller.servingMonitor.refreshRuntimeNow(controller: controller) }
                }
            )
        }
    }

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

    /// Drawing prompts instead of typing them: the region editor over the 1024×1024 mug with two
    /// boxes (the labeled one selected, showing its handles), a positive and a negative point,
    /// light and dark; then Segment on the Analyze board with the same prompts drawn over a
    /// seeded `vision segment` result, and Track's seed-frame scrubber over the in-test clip with
    /// a box on frame 12 and tracking set to end at frame 40.
    func testRegionPromptEditorSnapshots() throws {
        let analyze = try SnapshotFixture(
            outputDirectory: fixture.outputDirectory,
            seed: .analyze,
            processRunner: SnapshotProcessRunner(script: ModelsInventoryScript.analyzeReadinessResponses)
        )
        defer { analyze.tearDown() }

        let prompts = SnapshotFixture.regionPrompts
        let image = try XCTUnwrap(
            StudioImagePreviewLoader.downsampledImage(from: analyze.largeMugURL, maxPixelSize: 1_600)?.image
        )
        let editorSize = CGSize(width: 640, height: 620)
        for appearance in StudioSnapshotAppearance.allCases {
            let view = RegionEditorPreview(image: image, prompts: prompts, selection: prompts[0].id)
                .padding(24)
                .frame(width: editorSize.width, height: editorSize.height)
                .background(MereRunTheme.background)
            try analyze.write(view, size: editorSize, appearance: appearance, name: "f6-region-editor-\(appearance.rawValue)")
        }
        // The positive point selected (its ring and halo), with two more points hard against the
        // picture's top-right corner and left edge, whose tags flip and slide to stay on it.
        let edgePrompts = prompts + [
            .point(CGPoint(x: 1_010, y: 12), isPositive: true, label: "rim"),
            .point(CGPoint(x: 8, y: 1_015), isPositive: false)
        ]
        let pointSelected = RegionEditorPreview(image: image, prompts: edgePrompts, selection: prompts[2].id)
            .padding(24)
            .frame(width: editorSize.width, height: editorSize.height)
            .background(MereRunTheme.background)
        try analyze.write(pointSelected, size: editorSize, appearance: .light, name: "f6-region-editor-point-selected")

        var segment = StudioDraft()
        segment.reset(for: .segment)
        segment.prompt = ""
        segment.inputPath = analyze.largeMugURL.path
        segment.visionRegionPrompts = prompts

        var track = StudioDraft()
        track.reset(for: .track)
        track.prompt = "the bright band"
        track.inputPath = analyze.clipURL.path
        track.visionRegionPrompts = [
            .box(CGRect(x: 60, y: 40, width: 220, height: 150), label: "band"),
            .point(CGPoint(x: 420, y: 300), isPositive: false)
        ]
        track.visionInitFrame = 12
        track.visionEndFrame = 40

        let renders: [(name: String, task: StudioTask, appearance: StudioSnapshotAppearance)] = [
            ("f6-analyze-segment-light", .visionSegment, .light),
            ("f6-analyze-segment-dark", .visionSegment, .dark),
            ("f6-analyze-track-light", .visionTrack, .light),
            ("f6-analyze-track-dark", .visionTrack, .dark)
        ]
        for render in renders {
            let navigation = NavigationModel()
            let view = StudioRootView(seededDrafts: [.segment: segment, .track: track])
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

    /// The Analyze board at the default window (1440×820, Library showing) and narrower: Segment
    /// on a portrait picture, which has to fit the column above the composer rather than run
    /// under it, and Track's frame editor, whose range line and mark buttons have to fit their
    /// rows; each with a box and a point drawn.
    func testAnalyzeBoardWindowFitSnapshots() throws {
        let analyze = try SnapshotFixture(
            outputDirectory: fixture.outputDirectory,
            seed: .analyze,
            processRunner: SnapshotProcessRunner(script: ModelsInventoryScript.analyzeReadinessResponses)
        )
        defer { analyze.tearDown() }

        var segment = StudioDraft()
        segment.reset(for: .segment)
        segment.inputPath = analyze.portraitURL.path
        segment.visionRegionPrompts = [
            .box(CGRect(x: 180, y: 300, width: 360, height: 520), label: "figure"),
            .point(CGPoint(x: 360, y: 560), isPositive: true),
            .point(CGPoint(x: 700, y: 40), isPositive: false)
        ]

        var track = StudioDraft()
        track.reset(for: .track)
        track.inputPath = analyze.clipURL.path
        track.visionRegionPrompts = [
            .box(CGRect(x: 60, y: 40, width: 220, height: 150), label: "band"),
            .point(CGPoint(x: 170, y: 115), isPositive: true)
        ]
        track.visionInitFrame = 21

        let sizes: [(name: String, size: CGSize)] = [
            ("1440", Self.shellSize),
            ("1100", CGSize(width: 1_100, height: 760))
        ]
        for (label, size) in sizes {
            for (name, task) in [("segment-portrait", StudioTask.visionSegment), ("track", .visionTrack)] {
                let navigation = NavigationModel()
                let view = StudioRootView(seededDrafts: [.segment: segment, .track: track])
                    .environmentObject(analyze.controller)
                    .environmentObject(analyze.library)
                    .environmentObject(navigation)
                    .frame(width: size.width, height: size.height)
                try analyze.write(
                    view,
                    size: size,
                    appearance: .light,
                    name: "f6-analyze-\(name)-\(label)",
                    settle: 3.0,
                    afterAppear: { navigation.open(task: task) }
                )
            }
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

    /// The transcript's turn states on their own, light and dark: a reply that thought first
    /// (its "Thinking" disclosure collapsed above the answer), a reply the person stopped (its
    /// note and regenerate icon, no reason row), then a turn that failed with its one-line
    /// reason, Retry, and the run's log behind a collapsed "Show log"; and a second render of a
    /// reply streaming in while the model is still inside its reasoning block.
    func testConverseTurnStatesSnapshots() throws {
        let asked = StudioSnapshotRenderer.referenceDate.addingTimeInterval(-240)
        let model = SnapshotFixture.converseChatModelID
        let thread = StudioLibraryItem(
            id: UUID(), mode: .chat, prompt: "", inputURL: nil, outputURL: nil,
            createdAt: asked, updatedAt: asked.addingTimeInterval(200), status: .failed, exitCode: 1,
            commandPreview: "mere.run text chat", outputText: nil,
            messages: [
                StudioMessage(role: .user, content: "Why predict the noise instead of the image?", createdAt: asked),
                StudioMessage(
                    role: .assistant,
                    content: """
                    Predicting the noise gives the network a target with the same scale at every \
                    step, so one set of weights serves the whole schedule. Predicting the clean \
                    image makes the early, very noisy steps an almost impossible regression.
                    """,
                    createdAt: asked.addingTimeInterval(30), model: model, tokensPerSecond: 39,
                    reasoning: """
                    The question is about the parameterization. Two things matter: the target's \
                    variance across timesteps, and how the loss weights the steps. Keep it short \
                    and concrete.
                    """
                ),
                StudioMessage(role: .user, content: "And the one-line version?", createdAt: asked.addingTimeInterval(90)),
                StudioMessage(
                    role: .assistant, content: "The noise target keeps the same variance at every step, so",
                    createdAt: asked.addingTimeInterval(100), failed: true, cancelled: true, model: model
                ),
                StudioMessage(role: .user, content: "Show the sampling loop in Swift with MLX.", createdAt: asked.addingTimeInterval(180)),
                StudioMessage(
                    role: .assistant, content: "", createdAt: asked.addingTimeInterval(200), failed: true, model: model,
                    failureReason: "Model 'text-chat-qwen3.6-4b' is not installed.",
                    logTail: [
                        "Loading text-chat-qwen3.6-4b…",
                        "error: model 'text-chat-qwen3.6-4b' is not installed",
                        "Exited with code 1.",
                    ]
                ),
            ],
            systemPrompt: nil, model: model
        )
        let size = CGSize(width: 900, height: 760)
        for appearance in StudioSnapshotAppearance.allCases {
            let states = StudioConversationView(
                item: thread, liveReply: nil, isRunning: false, mode: .chat,
                onNewChat: {}, onCopy: { _ in }, onRetry: {}, onEdit: { _ in }, onBranch: { _ in }
            )
            .background(MereRunTheme.background)
            try fixture.write(states, size: size, appearance: appearance, name: "chat-turn-states-\(appearance.rawValue)", settle: 1.5)

            var streaming = thread
            streaming.status = .running
            streaming.messages = Array(thread.messages?.prefix(5) ?? [])
            let live = StudioConversationView(
                item: streaming,
                liveReply: ConversationTranscript.Reply(
                    answer: "",
                    reasoning: "The user wants MLX, so the tensors are MLXArray and the scheduler step is explicit.",
                    isThinking: true
                ),
                isRunning: true, mode: .chat,
                onNewChat: {}, onCopy: { _ in }, onRetry: {}, onEdit: { _ in }, onBranch: { _ in }
            )
            .background(MereRunTheme.background)
            try fixture.write(live, size: size, appearance: appearance, name: "chat-thinking-live-\(appearance.rawValue)", settle: 1.5)
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
            for (tab, suffix) in [(MereRunSettingsView.Tab.general, ""), (.server, "-server")] {
                let view = MereRunSettingsView(tab: tab)
                    .environmentObject(fixture.controller)
                    .environmentObject(fixture.crashReporter)
                    .frame(width: Self.settingsSize.width)
                try fixture.write(
                    view,
                    size: Self.settingsSize,
                    appearance: appearance,
                    name: "settings\(suffix)-\(appearance.rawValue)"
                )
            }
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
    /// detail column reads (usage, a quality gate, a benchmark). A real fixture job
    /// supplies the running download in the job bar. No CLI is launched.
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
        for appearance in StudioSnapshotAppearance.allCases {
            let navigation = NavigationModel()
            let view = StudioRootView()
                .environmentObject(fixture.controller)
                .environmentObject(fixture.library)
                .environmentObject(navigation)
            try fixture.write(view, size: CGSize(width: 768, height: 820), appearance: appearance,
                name: "models-installed-compact-\(appearance.rawValue)", settle: 2,
                afterAppear: { navigation.open(task: .modelsInstalled) })
        }

    }

    func testLayaDecisionWorkspaceSnapshots() throws {
        for width in [768.0, 1440.0] {
            for appearance in StudioSnapshotAppearance.allCases {
                let navigation = NavigationModel()
                let view = StudioRootView()
                    .environmentObject(fixture.controller)
                    .environmentObject(fixture.library)
                    .environmentObject(navigation)
                try fixture.write(view, size: CGSize(width: width, height: 820), appearance: appearance,
                                  name: "laya-decisions-\(Int(width))-\(appearance.rawValue)", settle: 1,
                                  afterAppear: { navigation.open(task: .textDecide) })
            }
        }
    }

    /// Music ▸ Train with three clips in the manifest editor — two captioned, one with lyrics, and
    /// one still needing a caption so the row and the problem list show — light and dark.
    func testMusicTrainingManifestEditorSnapshots() throws {
        let training = try SnapshotFixture(
            outputDirectory: fixture.outputDirectory,
            processRunner: SnapshotProcessRunner(script: ModelsInventoryScript.trainingReadinessResponses)
        )
        defer { training.tearDown() }
        let clips = training.root.appendingPathComponent("clips", isDirectory: true)
        try FileManager.default.createDirectory(at: clips, withIntermediateDirectories: true)
        var manifest = StudioMusicTrainingManifest()
        let rows: [(name: String, caption: String, lyrics: String)] = [
            ("late-night-drive.wav", "warm analog synth pad over a slow four-on-the-floor kick, 92 bpm",
             "city lights blur past the window\nwe don't say a word"),
            ("brass-hit.wav", "short brass section stab, bright and dry", ""),
            ("vocal-take.wav", "", ""),
        ]
        for row in rows {
            let url = clips.appendingPathComponent(row.name)
            try SnapshotFixture.writeSilentWAV(to: url, seconds: 2)
            manifest.clips.append(.init(audioPath: url.path, caption: row.caption, lyrics: row.lyrics))
        }
        training.controller.taskSessions.set(manifest, for: StudioTask.musicTrain.rawValue + ".Training.musicManifest")

        let renders: [(name: String, appearance: StudioSnapshotAppearance, size: CGSize)] = [
            ("music-train-manifest-light", .light, Self.fidelitySize),
            ("music-train-manifest-dark", .dark, Self.fidelitySize),
            ("music-train-manifest-narrow-light", .light, CGSize(width: 1_140, height: 820)),
        ]
        for render in renders {
            let navigation = NavigationModel()
            let view = StudioRootView()
                .environmentObject(training.controller)
                .environmentObject(training.library)
                .environmentObject(navigation)
            try training.write(view, size: render.size, appearance: render.appearance, name: render.name, settle: 2.5,
                               afterAppear: { navigation.open(task: .musicTrain) })
        }
    }

    /// Image ▸ Train over a finished run: the dataset folder in the well with its inspection,
    /// the Krea 2 model ready, the page's sections, and the dashboard following the run — loss
    /// curve, samples, checkpoints, A/B against an earlier run, and history — light and dark at
    /// the mockup size and at a narrower width; then Chat ▸ Train before anything is attached,
    /// light and dark.
    func testTrainingProjectSnapshots() throws {
        let training = try SnapshotFixture(
            outputDirectory: fixture.outputDirectory,
            processRunner: SnapshotProcessRunner(script: ModelsInventoryScript.trainingReadinessResponses)
        )
        defer { training.tearDown() }
        try training.seedTrainingRuns()

        func render(_ task: StudioTask, name: String, appearance: StudioSnapshotAppearance, size: CGSize) throws {
            let navigation = NavigationModel()
            let view = StudioRootView()
                .environmentObject(training.controller)
                .environmentObject(training.library)
                .environmentObject(navigation)
            try training.write(view, size: size, appearance: appearance, name: name, settle: 3.0,
                               afterAppear: { navigation.open(task: task) })
        }

        for appearance in StudioSnapshotAppearance.allCases {
            try render(.imageTrain, name: "train-image-\(appearance.rawValue)", appearance: appearance, size: Self.fidelitySize)
        }
        try render(.imageTrain, name: "train-image-narrow-light", appearance: .light, size: CGSize(width: 1_140, height: 820))
        for appearance in StudioSnapshotAppearance.allCases {
            try render(.chatTrain, name: "train-text-\(appearance.rawValue)", appearance: appearance, size: Self.fidelitySize)
        }
    }

    /// Image ▸ Datasets ▸ Run plan on the shared task workspace, through the root, after a
    /// training-plan preflight with one warning: the plan file in the input column, the report's
    /// status, warning, and sections as the result panel's rows, the variant and Preflight chips
    /// in the composer, light and dark.
    func testImageRunPlanReportSnapshots() throws {
        let plan = fixture.root.appendingPathComponent("plan.json")
        try Data("{\"schema_version\": 1, \"kind\": \"image.train_lora\"}".utf8).write(to: plan)
        let startedAt = Self.boardTime(hour: 10, minute: 0)
        let item = StudioLibraryItem(
            id: UUID(),
            mode: .createImage,
            prompt: "",
            inputURL: plan,
            outputURL: nil,
            createdAt: startedAt,
            updatedAt: startedAt.addingTimeInterval(0.8),
            status: .completed,
            exitCode: 0,
            commandPreview: "mere.run image run-plan plan.json --preflight --json",
            outputText: Self.trainingPlanPreflight,
            templateID: .imageRunPlan,
            artifactURLs: []
        )
        fixture.library.upsert(item)
        var draft = StudioTaskDraft(templateID: .imageRunPlan)
        draft.setArgument(0, plan.path)
        fixture.controller.taskSessions.setTaskDraft(draft, for: .imageDatasets)
        fixture.controller.taskSessions.set(Optional(item.id), for: StudioTask.imageDatasets.rawValue + ".requestID")

        for appearance in StudioSnapshotAppearance.allCases {
            let navigation = NavigationModel()
            let view = StudioRootView()
                .environmentObject(fixture.controller)
                .environmentObject(fixture.library)
                .environmentObject(navigation)
                .frame(width: 1_200, height: 820)
            try fixture.write(view, size: CGSize(width: 1_200, height: 820), appearance: appearance,
                              name: "image-run-plan-\(appearance.rawValue)", settle: 2.0,
                              afterAppear: { navigation.open(task: .imageDatasets) })
        }
    }

    /// Text ▸ Embeddings, Text ▸ Anonymize, and Image ▸ Datasets ▸ Discover and Validate on the
    /// shared task workspace, through the root. Embeddings with three texts in the editor and
    /// the cosine matrix beside them (light and dark, and at a narrower window); Anonymize with
    /// a paste and its protected text and spans; Discover with a scanned folder and three
    /// candidates, one blocked; Validate before any run, with no input column to fill.
    func testTextAndDatasetsWorkspaceSnapshots() throws {
        let sessions = fixture.controller.taskSessions

        // Embeddings: the run wrote its vectors beside the Text folder; the row keeps the file.
        let texts = ["semantic search query", "related document", "an unrelated recipe for soup"]
        let vectors = fixture.root.appendingPathComponent("embeddings.json")
        try Data(Self.embeddingsOutput.utf8).write(to: vectors)
        let embeddingsStart = Self.boardTime(hour: 9, minute: 12)
        fixture.library.upsert(StudioLibraryItem(
            id: UUID(), mode: .chat, prompt: texts.joined(separator: "\n"), inputURL: nil, outputURL: vectors,
            createdAt: embeddingsStart, updatedAt: embeddingsStart.addingTimeInterval(2.4), status: .completed, exitCode: 0,
            commandPreview: "mere.run text embed … --output embeddings.json --pretty", outputText: Self.embeddingsOutput,
            templateID: .textEmbed, artifactURLs: [vectors]
        ))
        var embeddings = StudioTaskDraft(templateID: .textEmbed)
        embeddings.prompt = texts.joined(separator: "\n")
        sessions.setTaskDraft(embeddings, for: .textEmbeddings)

        // Anonymize: the paste as one text, three spans found.
        let paste = "My name is Alice Smith and my email is alice@example.com. Call me on 555-0134."
        let anonymizeStart = Self.boardTime(hour: 9, minute: 20)
        fixture.library.upsert(StudioLibraryItem(
            id: UUID(), mode: .chat, prompt: paste, inputURL: nil, outputURL: nil,
            createdAt: anonymizeStart, updatedAt: anonymizeStart.addingTimeInterval(1.1), status: .completed, exitCode: 0,
            commandPreview: "mere.run text anonymize … --json --pretty", outputText: Self.anonymizationOutput(paste),
            templateID: .textAnonymize
        ))
        var anonymize = StudioTaskDraft(templateID: .textAnonymize)
        anonymize.prompt = paste
        sessions.setTaskDraft(anonymize, for: .textAnonymize)

        // Discover: a scanned folder with three candidate leaves.
        let datasets = fixture.root.appendingPathComponent("datasets", isDirectory: true)
        try FileManager.default.createDirectory(at: datasets, withIntermediateDirectories: true)
        let discoverStart = Self.boardTime(hour: 11, minute: 5)
        fixture.library.upsert(StudioLibraryItem(
            id: UUID(), mode: .createImage, prompt: "", inputURL: datasets, outputURL: nil,
            createdAt: discoverStart, updatedAt: discoverStart.addingTimeInterval(0.6), status: .completed, exitCode: 0,
            commandPreview: "mere.run image dataset discover --root datasets --max-depth 4 --min-usable-pairs 1 --json",
            outputText: Self.discoveryOutput(root: datasets), templateID: .imageDatasetDiscover
        ))
        var discover = StudioTaskDraft(templateID: .imageDatasetDiscover)
        discover.form["--root"] = .text(datasets.path)
        sessions.setTaskDraft(discover, for: .imageDatasets)

        func render(_ task: StudioTask, name: String, appearance: StudioSnapshotAppearance, size: CGSize) throws {
            let navigation = NavigationModel()
            let view = StudioRootView()
                .environmentObject(fixture.controller)
                .environmentObject(fixture.library)
                .environmentObject(navigation)
                .frame(width: size.width, height: size.height)
            try fixture.write(view, size: size, appearance: appearance, name: name, settle: 2.0,
                              afterAppear: { navigation.open(task: task) })
        }

        let board = CGSize(width: 1_440, height: 820)
        for appearance in StudioSnapshotAppearance.allCases {
            try render(.textEmbeddings, name: "text-embeddings-\(appearance.rawValue)", appearance: appearance, size: board)
            try render(.textAnonymize, name: "text-anonymize-\(appearance.rawValue)", appearance: appearance, size: board)
            try render(.imageDatasets, name: "image-datasets-discover-\(appearance.rawValue)", appearance: appearance, size: board)
        }
        try render(.textEmbeddings, name: "text-embeddings-narrow-light", appearance: .light, size: CGSize(width: 1_024, height: 760))

        // Validate: no input, nothing run yet.
        var validate = discover
        validate.switchTemplate(to: .imageValidate)
        sessions.setTaskDraft(validate, for: .imageDatasets)
        try render(.imageDatasets, name: "image-datasets-validate-empty-light", appearance: .light, size: CGSize(width: 1_200, height: 820))
    }

    /// A clock time on the reference day, for the rows the text and dataset boards seed.
    private static func boardTime(hour: Int, minute: Int) -> Date {
        let calendar = Calendar.current
        let today = calendar.startOfDay(for: StudioSnapshotRenderer.referenceDate)
        return calendar.date(byAdding: DateComponents(hour: hour, minute: minute), to: today) ?? today
    }

    /// `text embed` for three texts at eight dimensions: the first two close, the third apart.
    private static let embeddingsOutput = """
    {
      "object": "list",
      "model": "text-embed-qwen3-0.6b",
      "data": [
        {"object": "embedding", "index": 0, "embedding": [0.61, 0.42, -0.18, 0.33, 0.07, -0.29, 0.44, 0.12]},
        {"object": "embedding", "index": 1, "embedding": [0.58, 0.39, -0.22, 0.30, 0.11, -0.25, 0.47, 0.09]},
        {"object": "embedding", "index": 2, "embedding": [-0.21, 0.08, 0.66, -0.12, 0.51, 0.34, -0.19, 0.27]}
      ],
      "usage": {"prompt_tokens": 19, "total_tokens": 19}
    }
    """

    /// `text anonymize --json --pretty` for one paste with a name, an email, and a phone number.
    private static func anonymizationOutput(_ text: String) -> String {
        """
        {
          "object": "list",
          "model": "text-anonymize-privacy-filter",
          "data": [
            {
              "text": "\(text)",
              "anonymized_text": "My name is [NAME] and my email is [EMAIL]. Call me on [PHONE].",
              "token_count": 24,
              "spans": [
                {"label": "NAME", "text": "Alice Smith", "startToken": 3, "endToken": 5},
                {"label": "EMAIL", "text": "alice@example.com", "startToken": 10, "endToken": 16},
                {"label": "PHONE", "text": "555-0134", "startToken": 21, "endToken": 24}
              ]
            }
          ]
        }
        """
    }

    /// `image dataset discover --json` over a folder with a ready leaf, one with warnings, and one
    /// blocked for want of captions, plus one note about preview images.
    private static func discoveryOutput(root: URL) -> String {
        func candidate(_ id: String, _ name: String, status: String, trainable: Bool, images: Int, captions: Int, usable: Int, diagnostics: String) -> String {
            """
            {"id": "\(id)", "name": "\(name)", "path": "\(root.appendingPathComponent(id).path)", "relative_path": "\(id)", "depth": 1,
             "status": "\(status)", "trainable": \(trainable), "image_count": \(images), "caption_count": \(captions), "usable_pair_count": \(usable),
             "missing_caption_count": \(images - captions), "empty_caption_count": 0, "duplicate_caption_group_count": 0, "placeholder_caption_count": 0,
             "diagnostics": [\(diagnostics)]}
            """
        }
        let warning = """
        {"id": "missing_captions", "severity": "warning", "title": "Missing captions", "message": "2 images have no caption and will be skipped.", "locations": [], "suggested_action_ids": []}
        """
        let blocker = """
        {"id": "missing_captions", "severity": "blocker", "title": "Missing captions", "message": "No image has a caption.", "locations": [], "suggested_action_ids": []}
        """
        return """
        {"schema_version": 1, "mere_run_version": "0.55.0", "command": ["image", "dataset", "discover"], "mode": "inspection",
         "status": "warning", "created_at": "2026-09-23T11:05:00Z", "cwd": "\(root.path)",
         "summary": "Found 3 dataset candidates under \(root.lastPathComponent); 2 are trainable.",
         "request": {"root": "\(root.path)", "max_depth": 4, "min_usable_pairs": 1, "exclude_preview_images": false},
         "result": {"root": "\(root.path)", "scanned_directory_count": 14, "candidate_count": 3, "trainable_candidate_count": 2,
           "candidates": [
             \(candidate("ceramic-mugs", "Ceramic mugs", status: "ok", trainable: true, images: 48, captions: 48, usable: 48, diagnostics: "")),
             \(candidate("portraits", "Portraits", status: "warning", trainable: true, images: 32, captions: 30, usable: 30, diagnostics: warning)),
             \(candidate("sketches", "Sketches", status: "blocked", trainable: false, images: 6, captions: 0, usable: 0, diagnostics: blocker))
           ]},
         "diagnostics": [{"id": "preview_images", "severity": "note", "title": "Preview images",
                          "message": "3 preview images were counted; exclude them to train on originals only.", "locations": [], "suggested_action_ids": []}],
         "actions": []}
        """
    }

    /// The camera editors at the width of the 3D and Vision columns: InstantMesh with four views and
    /// one camera short, and multi-view geometry with two views and one mirrored rotation, so the
    /// per-view fields, the Match views button, and the CLI's checks all show, light and dark.
    func testCameraEditorSnapshots() throws {
        let instantMesh = StudioInstantMeshCameraDocument(cameras: (0..<3).map { _ in .example })
        var mirrored = StudioGeometryCamera.identity()
        mirrored.rotation = [-1, 0, 0, 0, 1, 0, 0, 0, 1]
        let geometry = StudioGeometryCameraDocument(cameras: [.identity(width: 4_032, height: 3_024), mirrored])

        for appearance in StudioSnapshotAppearance.allCases {
            let view = VStack(alignment: .leading, spacing: 24) {
                StudioInstantMeshCameraEditor(
                    enabled: .constant(true), document: .constant(instantMesh),
                    viewNames: ["front.png", "right.png", "back.png", "left.png"], message: .constant(nil)
                )
                Divider()
                StudioGeometryCameraEditor(
                    enabled: .constant(true), document: .constant(geometry),
                    views: [
                        StudioCameraView(name: "IMG_0412.heic", pixelSize: StudioPixelSize(width: 4_032, height: 3_024)),
                        StudioCameraView(name: "IMG_0413.heic", pixelSize: StudioPixelSize(width: 4_032, height: 3_024)),
                    ],
                    message: .constant(nil)
                )
            }
            .padding(18)
            .frame(width: 520, alignment: .topLeading)
            .background(MereRunTheme.background)
            .foregroundStyle(MereRunTheme.textPrimary)
            .environmentObject(fixture.controller)
            .environmentObject(fixture.library)
            try fixture.write(view, size: CGSize(width: 520, height: 1_180), appearance: appearance,
                              name: "camera-editors-\(appearance.rawValue)", settle: 1)
        }
    }

    /// `image run-plan --preflight --json` for a training plan, as `LoRATrainingPreflightEnvelope`
    /// prints it.
    private static let trainingPlanPreflight = """
    {"schema_version": 1, "mere_run_version": "0.55.0", "command": ["image", "train-lora"], "mode": "preflight",
     "status": "warning", "created_at": "2026-09-23T10:00:00Z", "cwd": "/Users/nerd/Pictures/mere.run/Image",
     "summary": "Ready to train on 48 usable pairs; 2 images have no caption and will be skipped.",
     "request": {"data": "/Users/nerd/Pictures/datasets/ceramic-mugs", "output": "/Users/nerd/Pictures/mere.run/Image/ceramic-mugs.safetensors",
       "model": "image-krea2-raw", "recipe": "krea-fast-style", "training_steps": 1200, "width": 1024, "height": 1024, "rank": 16,
       "alpha": 16, "learning_rate": 0.0001, "caption_dropout": 0.1},
     "result": {
       "dataset": {"directory": "/Users/nerd/Pictures/datasets/ceramic-mugs", "mode": "directory", "image_count": 50, "caption_count": 48,
         "usable_pair_count": 48, "missing_caption_count": 2, "empty_caption_count": 0, "duplicate_caption_group_count": 0,
         "duplicate_caption_count": 0, "excluded_preview_image_count": 0, "placeholder_caption_count": 0},
       "model": {"requested": "image-krea2-raw", "kind": "managed", "installed": true,
         "path": "/Users/nerd/Library/Application Support/MereRun/models/image-krea2-raw", "family": "krea2", "upstream_repo_id": "krea/krea-2-raw"},
       "output": {"path": "/Users/nerd/Pictures/mere.run/Image/ceramic-mugs.safetensors", "parent_directory": "/Users/nerd/Pictures/mere.run/Image",
         "parent_exists": true, "parent_will_be_created": false, "exists": false, "extension_valid": true},
       "plan": {"recipe": "krea-fast-style", "training_steps": 1200, "width": 1024, "height": 1024, "rank": 16, "alpha": 16,
         "learning_rate": 0.0001, "caption_dropout": 0.1, "checkpoint_interval": 250, "expected_checkpoint_count": 4,
         "max_resolution": 1536, "low_ram": false, "no_compile": false, "lr_warmup_steps": 100, "use_cosine_scheduler": true, "lr_min_factor": 0.1},
       "run_plan": {"schema_version": 1, "kind": "image.train_lora", "command": ["image", "train-lora"],
         "created_at": "2026-09-23T10:00:00Z", "cwd": "/Users/nerd/Pictures/mere.run/Image",
         "arguments": {"data": "/Users/nerd/Pictures/datasets/ceramic-mugs", "output": "/Users/nerd/Pictures/mere.run/Image/ceramic-mugs.safetensors",
           "model": "image-krea2-raw", "source_recipe": "krea-fast-style", "width": 1024, "height": 1024, "training_steps": 1200,
           "batch_size": 1, "learning_rate": 0.0001, "rank": 16, "alpha": 16, "max_text_length": 512, "scheduler_steps": 1000,
           "caption_dropout": 0.1, "seed": 42, "lite": false, "exclude_preview_images": false, "checkpoint_interval": 250,
           "max_resolution": 1536, "progressive": true, "low_ram": false, "no_compile": false, "gradient_checkpointing": false,
           "benchmark_warmup_steps": 0, "sample_interval": 250, "sample_prompt": "a ceramic coffee mug in soft morning light",
           "sample_steps": 20, "sample_cfg": 3.5, "sample_lora_scale": 1, "visualize": false, "visualize_port": 8765,
           "lr_warmup_steps": 100, "no_cosine_scheduler": false, "lr_min_factor": 0.1, "quiet": false},
         "resolved": {"recipe": "krea-fast-style", "training_steps": 1200, "width": 1024, "height": 1024, "rank": 16, "alpha": 16,
           "learning_rate": 0.0001, "caption_dropout": 0.1, "checkpoint_interval": 250, "expected_checkpoint_count": 4,
           "max_resolution": 1536, "low_ram": false, "no_compile": false, "lr_warmup_steps": 100, "use_cosine_scheduler": true, "lr_min_factor": 0.1}}},
     "diagnostics": [{"id": "missing_captions", "severity": "warning", "title": "Missing captions",
       "message": "2 images have no caption and will be skipped.", "locations": [], "suggested_action_ids": []}],
     "actions": []}
    """

    /// Text ▸ Decisions with the handbook example in the editor and a finished run beside it:
    /// a choice, a score, and a yes-or-no answer, one of them cut to fit.
    func testLayaDecisionAnswersSnapshots() throws {
        let run = fixture.root.appendingPathComponent("decisions-run", isDirectory: true)
        try FileManager.default.createDirectory(at: run, withIntermediateDirectories: true)
        let requestURL = run.appendingPathComponent("request.json")
        let outputURL = run.appendingPathComponent("decisions.json")
        try StudioDecisionDocument.example.requestJSON().write(to: requestURL)
        try Data(Self.layaResult.utf8).write(to: outputURL)

        var draft = CommandDraft()
        draft.inputPath = requestURL.path
        draft.outputPath = outputURL.path
        draft.model = "text-decide-laya"
        var item = StudioLibraryItem(
            id: UUID(),
            mode: .chat,
            prompt: "Decisions",
            inputURL: requestURL,
            outputURL: outputURL,
            createdAt: StudioSnapshotRenderer.referenceDate,
            updatedAt: StudioSnapshotRenderer.referenceDate,
            status: .completed,
            exitCode: 0,
            commandPreview: "mere.run text decide --input request.json",
            outputText: nil,
            artifactURLs: [outputURL]
        )
        item.templateID = .textDecide
        item.commandDraft = draft
        fixture.library.upsert(item)
        let sessions = fixture.controller.taskSessions
        sessions.set(StudioDecisionDocument.example, for: StudioTask.textDecide.rawValue + ".Laya.document")
        sessions.set(Optional(item.id), for: StudioTask.textDecide.rawValue + ".requestID")

        for appearance in StudioSnapshotAppearance.allCases {
            let navigation = NavigationModel()
            let view = StudioRootView()
                .environmentObject(fixture.controller)
                .environmentObject(fixture.library)
                .environmentObject(navigation)
            try fixture.write(view, size: CGSize(width: 1_440, height: 900), appearance: appearance,
                              name: "laya-answers-\(appearance.rawValue)", settle: 1.5,
                              afterAppear: { navigation.open(task: .textDecide) })
        }
    }

    private static let layaResult = """
    {"model": "text-decide-laya", "runtime": "mlx", "inputTokens": 131, "outputTokens": 0,
     "plan": {"model": "text-decide-laya", "maxTokens": 512, "headMaxTokens": 192, "questions": [
       {"id": "department", "inputTokens": 44, "stateTokens": 18, "stateTokensDropped": 0, "instructionTokensDropped": 0, "optionTokensDropped": [0, 0, 0], "optionCount": 3},
       {"id": "urgency", "inputTokens": 45, "stateTokens": 18, "stateTokensDropped": 0, "instructionTokensDropped": 0, "optionTokensDropped": [0, 0, 0], "optionCount": 3},
       {"id": "refund", "inputTokens": 42, "stateTokens": 18, "stateTokensDropped": 6, "instructionTokensDropped": 0, "optionTokensDropped": [0, 0], "optionCount": 2}]},
     "answers": {
       "department": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.91, "technical support": 0.06, "sales": 0.03}, "confidence": 0.78, "actProbability": 0.2, "rawTemperature": 0.9, "appliedTemperature": 0.9, "temperatureClamped": false},
       "urgency": {"type": "score", "score": 1.24, "probabilities": {"0": 0.18, "1": 0.4, "2": 0.42}, "confidence": 0.31, "actProbability": 0.3, "rawTemperature": 1.1, "appliedTemperature": 1.1, "temperatureClamped": false},
       "refund": {"type": "noul", "noul": 0.94, "probabilities": {"false": 0.06, "true": 0.94}, "confidence": 0.94, "actProbability": 0.7, "rawTemperature": 1.0, "appliedTemperature": 1.0, "temperatureClamped": false}
     }}
    """

    func testResultWorkspaceFocusAndComparisonSnapshots() throws {
        let fidelity = try SnapshotFixture(outputDirectory: fixture.outputDirectory, seed: .mockup)
        defer { fidelity.tearDown() }
        let images = fidelity.library.items.flatMap { item in
            item.allArtifactURLs.filter { StudioOutputFileKind.classify($0) == .image }.map { (item, $0) }
        }
        let first = try XCTUnwrap(images.first)
        let second = try XCTUnwrap(images.dropFirst().first)
        for width in [768.0, 1440.0] {
            for appearance in [StudioSnapshotAppearance.light, .dark] {
                let view = StudioResultWorkspaceView(item: first.0, url: first.1, items: fidelity.library.items,
                    initialComparison: StudioResultSelection(itemID: second.0.id, url: second.1),
                    onClose: {}, onVary: { _ in }, onSave: { _ in }, onContinue: { _, _, _ in })
                try fidelity.write(view, size: CGSize(width: width, height: 820), appearance: appearance,
                    name: "completion-compare-\(Int(width))-\(appearance)", settle: 2)
            }
        }
        let view = StudioResultWorkspaceView(item: first.0, url: first.1, items: fidelity.library.items,
            onClose: {}, onVary: { _ in }, onSave: { _ in }, onContinue: { _, _, _ in })
        try fidelity.write(view, size: Self.fidelitySize, appearance: .dark, name: "completion-focus-dark", settle: 2)
    }

    func testCompactSpecialistAndCommandSnapshots() throws {
        for task in [StudioTask.visionFaces, .audioEnhance, .threeDFromImage, .musicAnalyze, .imageDatasets, .voiceVoices] {
            let navigation = NavigationModel()
            let view = StudioRootView().environmentObject(fixture.controller)
                .environmentObject(fixture.library).environmentObject(navigation)
            try fixture.write(view, size: CGSize(width: 960, height: 760), appearance: .light,
                name: "completion-compact-\(task.rawValue)", settle: 2, afterAppear: { navigation.open(task: task) })
        }
        let navigation = NavigationModel()
        let view = StudioRootView().environmentObject(fixture.controller)
            .environmentObject(fixture.library).environmentObject(navigation)
        try fixture.write(view, size: CGSize(width: 768, height: 760), appearance: .light,
            name: "completion-compact-command", settle: 2, afterAppear: { navigation.toggleCommandColumn() })
    }

    /// Compare, "Use these settings", and the readiness card's next steps, light and dark. First
    /// the Library column with its two newest image rows batched, so the bar offers Compare. Then
    /// the feed with a finished mockup run (its card carries the settings icon beside Vary), a
    /// run that failed for want of its model (a plain reason and Get the model beside "Use these
    /// settings" and Retry), and the readiness card for that model — Get the model and Choose
    /// another model — followed by the same card before any check, when the Mac cannot run the
    /// model, and when the check failed with the CLI's line kept as muted detail.
    func testLibraryReuseAndReadinessSnapshots() throws {
        let fidelity = try SnapshotFixture(
            outputDirectory: fixture.outputDirectory,
            seed: .mockup,
            processRunner: SnapshotProcessRunner(script: ModelsInventoryScript.readinessResponses)
        )
        defer { fidelity.tearDown() }

        // The column first, while its two newest rows are both finished pictures.
        var draft = StudioDraft()
        draft.reset(for: .createImage)
        draft.prompt = "a ceramic coffee mug in soft morning light"
        for appearance in StudioSnapshotAppearance.allCases {
            let navigation = NavigationModel()
            let view = StudioRootView(seededDrafts: [.createImage: draft])
                .environmentObject(fidelity.controller)
                .environmentObject(fidelity.library)
                .environmentObject(navigation)
                .environment(\.studioLibrarySeed, StudioLibrarySeed(viewMode: .list, batchCount: 2))
                .frame(width: Self.fidelitySize.width, height: Self.fidelitySize.height)
            try fidelity.write(
                view, size: Self.fidelitySize, appearance: appearance,
                name: "library-compare-batch-\(appearance.rawValue)", settle: 3.0
            )
        }

        try fidelity.seedFailedImageRun()
        // The failed run's model is in the inventory but not on this Mac, with a title, so the
        // failed card and the readiness card both name it "Z-Image Turbo" and offer the pull.
        let inventory = StudioModelInventoryParser.rows(from: ModelsInventoryScript.modelList) + [
            StudioModelInventoryRow(id: "image-zimage-turbo", category: "image", status: "missing", size: "—",
                                    usageTerms: nil, title: "Z-Image Turbo", estimatedDownloadBytes: 6_300_000_000)
        ]
        let titles = StudioModelTitles(rows: inventory)

        func feed(_ readiness: ModelReadinessState) -> some View {
            let cards = StudioFeedCardBuilder.cards(items: fidelity.library.items, mode: .createImage) { _ in nil }
            let shown = [cards.last { $0.kind == .generation }, cards.first { $0.item.status == .failed }].compactMap { $0 }
            let noop: (StudioLibraryItem) -> Void = { _ in }
            return StudioFeedCanvas(
                presentation: StudioTaskPresentation(mode: .createImage),
                slots: StudioMode.createImage.attachmentSlots,
                cards: shown,
                readiness: readiness,
                pullJob: nil,
                highlightedID: nil,
                newResultID: .constant(nil),
                actions: StudioFeedActions(
                    vary: noop, rerun: noop, useAsInput: { _ in }, saveTo: { _ in }, cancel: { _ in },
                    remove: { _ in }, retry: noop, delete: noop, useSettings: noop, pullModel: { _ in },
                    useExample: { _ in }, attach: {}
                ),
                readinessActions: StudioReadinessActions(
                    scope: StudioModelScope(mode: .createImage, source: .contract), model: .constant("image-zimage-turbo"), modelInventory: inventory,
                    pullModel: {}, openModels: {}, recheck: {}
                )
            )
            .environment(\.studioModelTitles, titles)
            .frame(width: 900, height: 640)
            .background(MereRunTheme.background)
        }

        for appearance in StudioSnapshotAppearance.allCases {
            try fidelity.write(
                feed(.missingModel("image-zimage-turbo")), size: CGSize(width: 900, height: 640),
                appearance: appearance, name: "library-reuse-readiness-missing-\(appearance.rawValue)", settle: 2
            )
        }
        let tooLarge = StudioModelCapability(
            modelID: "image-zimage-turbo", isSupported: false, minimumUnifiedMemoryGB: 32,
            recommendedUnifiedMemoryGB: 64, download: nil, reason: nil
        )
        let variants: [(name: String, readiness: ModelReadinessState)] = [
            ("unchecked", .notChecked),
            ("unsupported", .unsupported(try XCTUnwrap(tooLarge.unavailableMessage(titles: titles)))),
            ("unknown", .unknown(MereRunController.modelListUnavailableMessage,
                                 detail: "Models root /Volumes/Models is not mounted")),
        ]
        for variant in variants {
            try fidelity.write(
                feed(variant.readiness), size: CGSize(width: 900, height: 640),
                appearance: .light, name: "library-reuse-readiness-\(variant.name)-light", settle: 2
            )
        }
    }

    /// Audio ▸ Who Spoke on the shared task workspace with a finished diarization: the recording
    /// in the well and on the canvas, one lane per speaker over its length, and every turn as the
    /// Analyze panel's rows with Save timeline…, light and dark. The page's own draft key from
    /// before the move is what the seed parks, so the board also proves the one-time import.
    func testWhoSpokeTimelineSnapshots() throws {
        let audio = try SnapshotFixture(
            outputDirectory: fixture.outputDirectory,
            processRunner: SnapshotProcessRunner(script: AudioVoiceScript.responses)
        )
        defer { audio.tearDown() }
        try audio.seedDiarizationRun()
        for appearance in StudioSnapshotAppearance.allCases {
            let navigation = NavigationModel()
            let view = StudioRootView()
                .environmentObject(audio.controller)
                .environmentObject(audio.library)
                .environmentObject(navigation)
                .frame(width: Self.fidelitySize.width, height: Self.fidelitySize.height)
            try audio.write(
                view,
                size: Self.fidelitySize,
                appearance: appearance,
                name: "who-spoke-\(appearance.rawValue)",
                settle: 2.5,
                afterAppear: { navigation.open(task: .audioWhoSpoke) }
            )
        }
    }

    /// The Audio and Voice pages through the root, light and dark at 1440×820 and once at
    /// 960×760: Who Spoke, Enhance, and Separate on the shared task workspace with a finished
    /// run each (the stems list plays from the manifest), Voice ▸ Voices as the Manage page with
    /// two saved voices and its New voice form, and Audio ▸ Live idle and then mid-session — a
    /// `speech listen` job held open by the process seam, adopted by the page, with three
    /// events streamed into its transcript.
    // swiftlint:disable:next function_body_length
    func testAudioAndVoicePageSnapshots() throws {
        let pages = try SnapshotFixture(
            outputDirectory: fixture.outputDirectory,
            processRunner: SnapshotProcessRunner(script: AudioVoiceScript.responses)
        )
        defer { pages.tearDown() }
        try pages.seedDiarizationRun()
        try pages.seedEnhanceRun()
        try pages.seedSeparateRun(task: .audioSeparate)
        let voices = try pages.voiceProfileSeed()
        let wide = CGSize(width: 1_440, height: 820)
        let narrow = CGSize(width: 960, height: 760)

        func render(
            _ task: StudioTask, name: String, size: CGSize, appearance: StudioSnapshotAppearance,
            profiles: [StudioVoiceProfileRecord] = [], afterAppear: (() -> Void)? = nil
        ) throws {
            let navigation = NavigationModel()
            let view = StudioRootView()
                .environmentObject(pages.controller)
                .environmentObject(pages.library)
                .environmentObject(navigation)
                .environment(\.studioVoiceProfileSeed, profiles)
                .frame(width: size.width, height: size.height)
            try pages.write(view, size: size, appearance: appearance, name: name, settle: 2.5, afterAppear: {
                navigation.open(task: task)
                afterAppear?()
            })
        }

        for appearance in StudioSnapshotAppearance.allCases {
            let suffix = appearance.rawValue
            try render(.audioWhoSpoke, name: "audio-who-spoke-\(suffix)", size: wide, appearance: appearance)
            try render(.audioEnhance, name: "audio-enhance-\(suffix)", size: wide, appearance: appearance)
            try render(.audioSeparate, name: "audio-separate-\(suffix)", size: wide, appearance: appearance)
            try render(.voiceVoices, name: "voice-voices-\(suffix)", size: wide, appearance: appearance, profiles: voices)
            try render(.audioLive, name: "audio-live-idle-\(suffix)", size: wide, appearance: appearance)
        }
        try render(.audioWhoSpoke, name: "audio-who-spoke-compact-light", size: narrow, appearance: .light)
        try render(.audioEnhance, name: "audio-enhance-compact-light", size: narrow, appearance: .light)
        try render(.audioSeparate, name: "audio-separate-compact-light", size: narrow, appearance: .light)
        try render(.voiceVoices, name: "voice-voices-new-compact-light", size: narrow, appearance: .light)
        try render(.audioLive, name: "audio-live-compact-light", size: narrow, appearance: .light)

        let requestID = try pages.seedLiveListenSession()
        for appearance in StudioSnapshotAppearance.allCases {
            try render(.audioLive, name: "audio-live-running-\(appearance.rawValue)", size: wide, appearance: appearance) {
                pages.speakIntoLiveListenSession()
            }
        }
        XCTAssertTrue(pages.controller.jobs.job(requestID: requestID)?.state.isRunning ?? false, "the seam holds the session open")
    }

    /// Music ▸ Analyze with a finished ACE-Step analysis: tempo, key, meter, language, and how
    /// much was analyzed as tiles, the caption and lyrics as prose, and the model's reply folded
    /// away, light and dark.
    /// Music ▸ Analyze and Music ▸ Transcribe on the shared task workspace, through the root.
    /// Analyze with a finished run seeded (the song in the well and the input strip with its
    /// player, the Analysis panel's tiles, caption, lyrics, and folded model reply) beside its
    /// inspector, where the checkpoint root files under Model as a folder chooser, light and
    /// dark; Transcribe with a seeded MIDI transcription (the piano roll under Notes, with Quick
    /// Look and Reveal) beside its inspector, whose instruments editor shows the chips picked
    /// from the CLI's list, light and dark; and Transcribe at the compact width without the
    /// inspector. The models the tasks default to are installed in the scripted inventory, so
    /// the boards show results rather than readiness cards.
    func testMusicAnalysisSnapshots() throws {
        let music = try SnapshotFixture(
            outputDirectory: fixture.outputDirectory,
            processRunner: SnapshotProcessRunner(script: ModelsInventoryScript.musicReadinessResponses)
        )
        defer { music.tearDown() }
        try music.seedMusicAnalysisRun()
        try music.seedTranscribeRun()

        func render(_ task: StudioTask, name: String, appearance: StudioSnapshotAppearance, size: CGSize, inspector: Bool) throws {
            let navigation = NavigationModel()
            let view = StudioRootView()
                .environmentObject(music.controller)
                .environmentObject(music.library)
                .environmentObject(navigation)
                .frame(width: size.width, height: size.height)
            try music.write(view, size: size, appearance: appearance, name: name, settle: 2.5, afterAppear: {
                navigation.open(task: task)
                if inspector { navigation.toggleInspector(for: task) }
            })
        }

        for appearance in StudioSnapshotAppearance.allCases {
            try render(.musicAnalyze, name: "music-analyze-\(appearance.rawValue)", appearance: appearance,
                       size: Self.fidelitySize, inspector: true)
            try render(.musicTranscribe, name: "music-transcribe-\(appearance.rawValue)", appearance: appearance,
                       size: Self.fidelitySize, inspector: true)
        }
        try render(.musicTranscribe, name: "music-transcribe-compact-light", appearance: .light,
                   size: CGSize(width: 960, height: 760), inspector: false)
    }

    /// The shared task workspace, rendered directly: Audio ▸ Enhance as an Analyze
    /// task with an audio well, once with a finished enhance run seeded so the input strip, the
    /// player, and the result column draw, light and dark; its inspector column beside it; and
    /// Vision ▸ Pose empty, so the serif empty state and the well's attach button show.
    func testTaskWorkspaceSnapshots() throws {
        let workspace = try SnapshotFixture(
            outputDirectory: fixture.outputDirectory,
            processRunner: SnapshotProcessRunner(script: ModelsInventoryScript.analyzeReadinessResponses)
        )
        defer { workspace.tearDown() }
        try workspace.seedEnhanceRun()
        let sessions = workspace.controller.taskSessions
        let runner = StudioTaskRunner(controller: workspace.controller, library: workspace.library)

        func render(_ task: StudioTask, name: String, appearance: StudioSnapshotAppearance, size: CGSize) throws {
            let navigation = NavigationModel(destination: task.destination)
            let view = StudioTaskWorkspace(task: task, models: workspace.controller.modelStore)
                .environmentObject(workspace.controller)
                .environmentObject(workspace.library)
                .environmentObject(navigation)
                .environment(\.studioTaskSessions, sessions)
                .environment(\.studioTaskScope, task.rawValue)
                .environment(\.studioTaskRunner, runner)
                .frame(width: size.width, height: size.height)
            try workspace.write(view, size: size, appearance: appearance, name: name, settle: 2.5)
        }

        for appearance in StudioSnapshotAppearance.allCases {
            try render(.audioEnhance, name: "task-workspace-enhance-\(appearance.rawValue)", appearance: appearance,
                       size: CGSize(width: 1_140, height: 820))
        }
        try render(.visionPose, name: "task-workspace-pose-empty-light", appearance: .light, size: CGSize(width: 960, height: 760))

        let draft = try XCTUnwrap(sessions.taskDraft(for: .audioEnhance))
        let inspector = StudioTaskInspector(
            task: .audioEnhance, draft: .constant(draft), modelInventory: workspace.controller.modelStore.rows,
            readiness: .ready, onShowModels: {}, onClose: {}
        )
        .environmentObject(workspace.controller)
        .frame(width: StudioLayoutPolicy.inspectorWidth, height: 820)
        try workspace.write(inspector, size: CGSize(width: StudioLayoutPolicy.inspectorWidth, height: 820),
                            appearance: .light, name: "task-workspace-enhance-inspector-light", settle: 1.5)
    }

    /// The Sound tasks on the shared workspace, each over a seeded finished run: Video Foley's
    /// feed card reviews the clip against its waveform (light and dark, and at a narrower width),
    /// Condition's card shows the conditioning tensors' header, Encode's Analyze board the
    /// `.npy` header, Decode's the decoded audio, Score's the CLAP gauge; then Foley's inspector
    /// with the renoise editor on Fixed amount. Readiness is answered from a scripted inventory
    /// with the Woosh models installed.
    func testSoundWorkspaceSnapshots() throws {
        let sound = try SnapshotFixture(
            outputDirectory: fixture.outputDirectory,
            processRunner: SnapshotProcessRunner(script: ModelsInventoryScript.soundReadinessResponses)
        )
        defer { sound.tearDown() }
        try sound.seedSoundRuns()
        let sessions = sound.controller.taskSessions
        let runner = StudioTaskRunner(controller: sound.controller, library: sound.library)

        func render(_ task: StudioTask, name: String, appearance: StudioSnapshotAppearance, size: CGSize) throws {
            let navigation = NavigationModel(destination: task.destination)
            let view = StudioTaskWorkspace(task: task, models: sound.controller.modelStore)
                .environmentObject(sound.controller)
                .environmentObject(sound.library)
                .environmentObject(navigation)
                .environment(\.studioTaskSessions, sessions)
                .environment(\.studioTaskScope, task.rawValue)
                .environment(\.studioTaskRunner, runner)
                .frame(width: size.width, height: size.height)
            try sound.write(view, size: size, appearance: appearance, name: name, settle: 2.5)
        }

        let wide = CGSize(width: 1_140, height: 820)
        for appearance in StudioSnapshotAppearance.allCases {
            try render(.soundFoley, name: "task-workspace-sound-foley-\(appearance.rawValue)", appearance: appearance, size: wide)
            try render(.soundScore, name: "task-workspace-sound-score-\(appearance.rawValue)", appearance: appearance, size: wide)
        }
        try render(.soundFoley, name: "task-workspace-sound-foley-narrow-light", appearance: .light, size: CGSize(width: 820, height: 760))
        try render(.soundCondition, name: "task-workspace-sound-condition-light", appearance: .light, size: wide)
        try render(.soundEncode, name: "task-workspace-sound-encode-light", appearance: .light, size: wide)
        try render(.soundDecode, name: "task-workspace-sound-decode-light", appearance: .light, size: wide)

        let draft = try XCTUnwrap(sessions.taskDraft(for: .soundFoley))
        let inspector = StudioTaskInspector(
            task: .soundFoley, draft: .constant(draft), modelInventory: sound.controller.modelStore.rows,
            readiness: .ready, onShowModels: {}, onClose: {}
        )
        .environmentObject(sound.controller)
        .environment(\.studioTaskSessions, sessions)
        .environment(\.studioTaskScope, StudioTask.soundFoley.rawValue)
        .frame(width: StudioLayoutPolicy.inspectorWidth, height: 820)
        try sound.write(inspector, size: CGSize(width: StudioLayoutPolicy.inspectorWidth, height: 820),
                        appearance: .light, name: "task-workspace-sound-foley-inspector-light", settle: 1.5)
    }

    /// The Vision tasks on the task workspace, each with a finished run on a drawn picture so
    /// the input strip, the result view, the panel rows, and the canvas renderer draw: Faces
    /// (boxes, then the Points overlay and the Compare inspector with its click-to-pick face
    /// picker), Pose, Flow, Depth (the preview PNG in the input column), Geometry (the scene
    /// strip, and the multi-view inspector's camera editor), light and dark, and the Live session
    /// idle and after a capture. Nothing runs; the rows and files are seeded.
    func testVisionWorkspaceSnapshots() throws {
        let vision = try SnapshotFixture(
            outputDirectory: fixture.outputDirectory,
            processRunner: SnapshotProcessRunner(script: ModelsInventoryScript.analyzeReadinessResponses)
        )
        defer { vision.tearDown() }
        let seeded = try vision.seedVisionRuns()
        let sessions = vision.controller.taskSessions
        let runner = StudioTaskRunner(controller: vision.controller, library: vision.library)
        let wide = CGSize(width: 1_440, height: 820)
        let narrow = CGSize(width: 960, height: 760)

        func render(_ task: StudioTask, name: String, appearance: StudioSnapshotAppearance, size: CGSize, view: StudioAnalyzeResultView? = nil) throws {
            let navigation = NavigationModel(destination: task.destination)
            navigation.selectedLibraryID = seeded[task]
            let workspace = StudioTaskWorkspace(task: task, models: vision.controller.modelStore)
                .environmentObject(vision.controller)
                .environmentObject(vision.library)
                .environmentObject(navigation)
                .environment(\.studioTaskSessions, sessions)
                .environment(\.studioTaskScope, task.rawValue)
                .environment(\.studioTaskRunner, runner)
                .frame(width: size.width, height: size.height)
            try vision.write(workspace, size: size, appearance: appearance, name: name, settle: 3)
        }

        for appearance in StudioSnapshotAppearance.allCases {
            let suffix = appearance.rawValue
            try render(.visionFaces, name: "vision-faces-\(suffix)", appearance: appearance, size: wide)
            try render(.visionPose, name: "vision-pose-\(suffix)", appearance: appearance, size: wide)
            try render(.visionFlow, name: "vision-flow-\(suffix)", appearance: appearance, size: wide)
            try render(.visionDepth, name: "vision-depth-\(suffix)", appearance: appearance, size: wide)
            try render(.visionGeometry, name: "vision-geometry-\(suffix)", appearance: appearance, size: wide)
        }
        try render(.visionFaces, name: "vision-faces-narrow-light", appearance: .light, size: narrow)
        try render(.visionGeometry, name: "vision-geometry-narrow-light", appearance: .light, size: narrow)

        // Faces ▸ Compare in the inspector: the reference picker shows the Detect run's boxes on
        // the portrait; the candidate has no detection yet and keeps the plain field.
        var compare = try XCTUnwrap(sessions.taskDraft(for: .visionFaces))
        compare.switchTemplate(to: .visionFaceCompare)
        StudioTaskSchema.slots(for: .visionFaceCompare)[1].attach([vision.secondPortraitURL], to: &compare)
        compare.form["--reference-face-index"] = .integer(1)
        func inspector(_ task: StudioTask, draft: StudioTaskDraft, name: String) throws {
            let view = StudioTaskInspector(
                task: task, draft: .constant(draft), modelInventory: vision.controller.modelStore.rows,
                readiness: .ready, onShowModels: {}, onClose: {}
            )
            .environmentObject(vision.controller)
            .environmentObject(vision.library)
            .environment(\.studioTaskSessions, sessions)
            .environment(\.studioTaskScope, task.rawValue)
            .frame(width: StudioLayoutPolicy.inspectorWidth, height: 820)
            try vision.write(view, size: CGSize(width: StudioLayoutPolicy.inspectorWidth, height: 820),
                             appearance: .light, name: name, settle: 2.5)
        }
        try inspector(.visionFaces, draft: compare, name: "vision-faces-compare-inspector-light")

        // Geometry ▸ Multi-view in the inspector: two ordered views and a camera per view. The
        // cameras are sized for another picture, so the editor shows the CLI's checks and writes
        // no draft file while rendering.
        var multiview = try XCTUnwrap(sessions.taskDraft(for: .visionGeometry))
        multiview.switchTemplate(to: .visionGeometryMultiview)
        StudioTaskSchema.slots(for: .visionGeometryMultiview)[0]
            .attach([vision.portraitURL, vision.secondPortraitURL], to: &multiview)
        let cameras = StudioGeometryCameraDocument(cameras: [.identity(), .identity()])
        sessions.set(cameras, for: StudioTask.visionGeometry.rawValue + ".geometryCameras")
        sessions.set(true, for: StudioTask.visionGeometry.rawValue + ".suppliesCameras")
        try inspector(.visionGeometry, draft: multiview, name: "vision-geometry-multiview-inspector-light")

        // Live: idle with the example prompts, then the finished capture with its clip.
        let liveKey = StudioTask.visionLive.rawValue + ".requestID"
        let liveRun = sessions.value(for: liveKey, default: Optional<UUID>.none)
        func renderLive(name: String, appearance: StudioSnapshotAppearance) throws {
            let navigation = NavigationModel(destination: StudioTask.visionLive.destination)
            let view = StudioLiveTrackSession(models: vision.controller.modelStore)
                .environmentObject(vision.controller)
                .environmentObject(vision.library)
                .environmentObject(navigation)
                .environment(\.studioTaskSessions, sessions)
                .environment(\.studioTaskScope, StudioTask.visionLive.rawValue)
                .environment(\.studioTaskRunner, runner)
                .frame(width: wide.width, height: wide.height)
            try vision.write(view, size: wide, appearance: appearance, name: name, settle: 3)
        }
        sessions.set(Optional<UUID>.none, for: liveKey)
        try renderLive(name: "vision-live-idle-light", appearance: .light)
        sessions.set(liveRun, for: liveKey)
        for appearance in StudioSnapshotAppearance.allCases {
            try renderLive(name: "vision-live-ended-\(appearance.rawValue)", appearance: appearance)
        }
    }

    /// 3D ▸ From image on the shared task workspace: the feed with a finished TripoSR run (its
    /// mesh through Quick Look and the manifest's counts under the tile), the well holding the
    /// picture, and the Engine chip, light and dark at 1440×820 and at a narrower width; then the
    /// inspector on TRELLIS.2 with its remesh controls, and on InstantMesh with four views and
    /// cameras one short, so the ordered-view rows, the camera cards, and the CLI's check show.
    func testThreeDWorkspaceSnapshots() throws {
        let workspace = try SnapshotFixture(
            outputDirectory: fixture.outputDirectory,
            processRunner: SnapshotProcessRunner(script: ModelsInventoryScript.readinessResponses(installing: [
                (id: "image-3d-triposr", category: "image", title: "TripoSR"),
                (id: "image-3d-trellis2-4b", category: "image", title: "TRELLIS.2 4B"),
                (id: "image-3d-instantmesh-base", category: "image", title: "InstantMesh"),
            ]))
        )
        defer { workspace.tearDown() }
        // The camera editor saves its document as a draft file while cameras are on; keep that
        // in the fixture's folder rather than the user's Application Support.
        StudioTestDefaults.register([StudioOutputLocation.supportRootDefaultsKey: workspace.root.path])
        defer { StudioTestDefaults.restore() }
        let views = try workspace.seedMeshRun()
        let task = StudioTask.threeDFromImage
        let sessions = workspace.controller.taskSessions
        let runner = StudioTaskRunner(controller: workspace.controller, library: workspace.library)

        func render(name: String, appearance: StudioSnapshotAppearance, size: CGSize) throws {
            let navigation = NavigationModel(destination: task.destination)
            let view = StudioTaskWorkspace(task: task, models: workspace.controller.modelStore)
                .environmentObject(workspace.controller)
                .environmentObject(workspace.library)
                .environmentObject(navigation)
                .environment(\.studioTaskSessions, sessions)
                .environment(\.studioTaskScope, task.rawValue)
                .environment(\.studioTaskRunner, runner)
                .frame(width: size.width, height: size.height)
            try workspace.write(view, size: size, appearance: appearance, name: name, settle: 3)
        }

        for appearance in StudioSnapshotAppearance.allCases {
            try render(name: "three-d-workspace-\(appearance.rawValue)", appearance: appearance, size: CGSize(width: 1_440, height: 820))
        }
        try render(name: "three-d-workspace-narrow-light", appearance: .light, size: CGSize(width: 1_140, height: 820))

        func inspector(_ draft: StudioTaskDraft, height: CGFloat) -> some View {
            StudioTaskInspector(
                task: task, draft: .constant(draft), modelInventory: workspace.controller.modelStore.rows,
                readiness: .ready, onShowModels: {}, onClose: {}
            )
            .environmentObject(workspace.controller)
            .environmentObject(workspace.library)
            .environment(\.studioTaskSessions, sessions)
            .environment(\.studioTaskScope, task.rawValue)
            .frame(width: StudioLayoutPolicy.inspectorWidth, height: height)
        }

        var trellis = StudioTaskDraft(templateID: .imageReconstruct3DTrellis2)
        trellis.setArgument(0, workspace.mugURL.path)
        try workspace.write(inspector(trellis, height: 820), size: CGSize(width: StudioLayoutPolicy.inspectorWidth, height: 820),
                            appearance: .light, name: "three-d-inspector-trellis-light", settle: 1.5)

        var instantMesh = StudioTaskDraft(templateID: .imageReconstruct3DMultiview)
        StudioTaskSchema.slots(for: .imageReconstruct3DMultiview)[0].attach(views, to: &instantMesh)
        sessions.set(StudioInstantMeshCameraDocument(cameras: (0..<3).map { _ in .example }), for: task.rawValue + ".3DCreation.cameras")
        sessions.set(true, for: task.rawValue + ".3DCreation.suppliesCameras")
        for appearance in StudioSnapshotAppearance.allCases {
            try workspace.write(inspector(instantMesh, height: 1_400), size: CGSize(width: StudioLayoutPolicy.inspectorWidth, height: 1_400),
                                appearance: appearance, name: "three-d-inspector-instantmesh-\(appearance.rawValue)", settle: 1.5)
        }
    }

    /// Earth on the shared task workspace, through the root: TESSERA with a finished run — the
    /// attached bundle read against the tensors the command needs on the left, the embedding's
    /// header in the result panel on the right, the inspector open with the constrained
    /// dimensions picker — light and dark at the board size and at a narrow width; Flood with a
    /// bundle missing its DEM, so the checklist warns before any run; OlmoEarth with nothing
    /// attached, the serif empty state; and TESSERA's inspector column on its own.
    func testEarthWorkspaceSnapshots() throws {
        let earth = try SnapshotFixture(
            outputDirectory: fixture.outputDirectory,
            processRunner: SnapshotProcessRunner(script: ModelsInventoryScript.analyzeReadinessResponses)
        )
        defer { earth.tearDown() }
        try earth.seedEarthRuns()

        func render(_ task: StudioTask, name: String, appearance: StudioSnapshotAppearance, size: CGSize) throws {
            let navigation = NavigationModel()
            let view = StudioRootView()
                .environmentObject(earth.controller)
                .environmentObject(earth.library)
                .environmentObject(navigation)
            try earth.write(view, size: size, appearance: appearance, name: name, settle: 2.5,
                            afterAppear: { navigation.open(task: task) })
        }

        let board = CGSize(width: 1_440, height: 820)
        for appearance in StudioSnapshotAppearance.allCases {
            try render(.earthTessera, name: "earth-tessera-result-\(appearance.rawValue)", appearance: appearance, size: board)
        }
        try render(.earthTessera, name: "earth-tessera-result-narrow-light", appearance: .light, size: CGSize(width: 960, height: 760))
        try render(.earthFlood, name: "earth-flood-missing-dem-light", appearance: .light, size: board)
        try render(.earthFlood, name: "earth-flood-missing-dem-dark", appearance: .dark, size: board)
        try render(.earthOlmoEarth, name: "earth-olmoearth-empty-light", appearance: .light, size: board)

        let draft = try XCTUnwrap(earth.controller.taskSessions.taskDraft(for: .earthTessera))
        let inspector = StudioTaskInspector(
            task: .earthTessera, draft: .constant(draft), modelInventory: earth.controller.modelStore.rows,
            readiness: .ready, onShowModels: {}, onClose: {}
        )
        .environmentObject(earth.controller)
        .frame(width: StudioLayoutPolicy.inspectorWidth, height: 820)
        try earth.write(inspector, size: CGSize(width: StudioLayoutPolicy.inspectorWidth, height: 820),
                        appearance: .light, name: "earth-tessera-inspector-light", settle: 1.5)
        let olmoInspector = StudioTaskInspector(
            task: .earthOlmoEarth, draft: .constant(StudioTaskDraft(templateID: .geoOlmoEarth)),
            modelInventory: earth.controller.modelStore.rows, readiness: .ready, onShowModels: {}, onClose: {}
        )
        .environmentObject(earth.controller)
        .frame(width: StudioLayoutPolicy.inspectorWidth, height: 820)
        try earth.write(olmoInspector, size: CGSize(width: StudioLayoutPolicy.inspectorWidth, height: 820),
                        appearance: .light, name: "earth-olmoearth-inspector-light", settle: 1.5)
    }

    /// Runs opened on a failed graph run: its state and what went wrong, the facts, each step
    /// with its own state, the outputs with Reveal, and the raw report folded away. `executor
    /// list`, `run list`, and `run inspect` are answered by a scripted runner; no CLI runs.
    /// A Project or Manage page's run whose result is what the CLI printed — a benchmark table —
    /// shown as printed: the `|` columns, `*fused*`, and `__baseline__` stay characters rather
    /// than turning into Markdown emphasis, light and dark.
    func testRunDetailPrintedReportSnapshots() throws {
        let report = """
        model                     | prefill tok/s | decode tok/s
        --------------------------|---------------|-------------
        *fused* text-chat-gemma4  |        1841.2 |         61.2
        __baseline__ pipeline     |         912.7 |          9.8
        """
        let item = StudioLibraryItem(
            id: UUID(), mode: .chat, prompt: "", inputURL: nil, outputURL: nil,
            createdAt: Self.boardTime(hour: 10, minute: 0), updatedAt: Self.boardTime(hour: 10, minute: 2),
            status: .completed, exitCode: 0, commandPreview: "mere.run model benchmark text-chat-gemma4",
            outputText: report, templateID: .modelBenchmark, artifactURLs: []
        )
        for appearance in StudioSnapshotAppearance.allCases {
            let view = StudioRunDetailView(item: item, preferredKinds: [.text])
                .environmentObject(fixture.controller)
                .padding(16)
                .frame(width: 640, height: 280)
                .background(MereRunTheme.background)
            try fixture.write(view, size: CGSize(width: 640, height: 280), appearance: appearance,
                              name: "run-detail-printed-report-\(appearance.rawValue)")
        }
    }

    func testRunsInspectionSnapshots() throws {
        let runs = try SnapshotFixture(
            outputDirectory: fixture.outputDirectory,
            processRunner: SnapshotProcessRunner(script: RunsScript.responses)
        )
        defer { runs.tearDown() }
        for appearance in StudioSnapshotAppearance.allCases {
            let view = StudioOperationsView(initialSelection: RunsScript.runPath)
                .environmentObject(runs.controller)
                .environmentObject(runs.library)
                .frame(width: Self.fidelitySize.width, height: Self.fidelitySize.height)
            try runs.write(
                view,
                size: Self.fidelitySize,
                appearance: appearance,
                name: "runs-inspect-\(appearance.rawValue)",
                settle: 2.5
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
    /// A 640×360 clip of 60 frames at 12 fps for Track's seed-frame scrubber.
    private(set) var clipURL: URL!
    /// A 720×1280 portrait picture, the shape that has to fit the column above the composer.
    private(set) var portraitURL: URL!
    /// A second picture for the Vision boards' two-image tasks (Compare's candidate, Flow's
    /// target, the second multi-view frame).
    private(set) var secondPortraitURL: URL!

    /// The prompts the region-editor renders draw on the 1024×1024 mug: the cup's box (labeled,
    /// selected in the shot), the saucer's, a positive point on the handle, a negative one on
    /// the shadow.
    static let regionPrompts: [StudioRegionPrompt] = [
        .box(CGRect(x: 246, y: 307, width: 471, height: 451), label: "coffee cup"),
        .box(CGRect(x: 82, y: 757, width: 184, height: 164)),
        .point(CGPoint(x: 690, y: 520), isPositive: true),
        .point(CGPoint(x: 880, y: 900), isPositive: false)
    ]
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
        processRunner: MereRunProcessRunning = SnapshotProcessRunner(),
        machineMonitor: StudioMachineMonitor? = nil
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
            secretStore: InMemorySecretStore(),
            processRunner: processRunner,
            cliResolver: { _ in .executable(URL(fileURLWithPath: "/usr/local/bin/mere.run")) },
            resolvesCLIOnInit: true,
            machineMonitor: machineMonitor,
            initialRuntimeHost: "127.0.0.1",
            initialRuntimePort: 8_080
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

        let now = StudioSnapshotRenderer.referenceDate
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
        let clip = directory.appendingPathComponent("band.mp4", isDirectory: false)
        try Self.writeFixtureMP4(to: clip, size: CGSize(width: 640, height: 360), frames: 60)
        clipURL = clip
        let portrait = directory.appendingPathComponent("portrait.png", isDirectory: false)
        try Self.writeFixturePNG(to: portrait, size: CGSize(width: 720, height: 1_280), hueOffset: 0.6)
        portraitURL = portrait
        let segmented = directory.appendingPathComponent("mug_segmented.png", isDirectory: false)
        try Self.writeMugPNG(to: segmented, side: 1_024)
        let segmentDocument = directory.appendingPathComponent("mug_segmented.json", isDirectory: false)
        try Self.segmentDocument(input: mug, annotated: segmented, document: segmentDocument)
            .write(to: segmentDocument, atomically: true, encoding: .utf8)

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
                prompt: "the cup",
                inputURL: mug,
                outputURL: segmented,
                createdAt: findAt.addingTimeInterval(-60 * 60 * 24 * 5),
                updatedAt: findAt.addingTimeInterval(-60 * 60 * 24 * 5 + 3),
                status: .completed,
                exitCode: 0,
                commandPreview: "mere.run vision segment mug.png --prompt \"the cup\"",
                templateID: .visionSegment,
                artifactURLs: [segmented, segmentDocument]
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

    /// One detection as `vision segment --json-output` writes it (`SAM31SegmentationMetadata`):
    /// pixel xyxy boxes, camelCase keys, no mask sidecar so the Masks view shows its fallback.
    private static func segmentDocument(input: URL, annotated: URL, document: URL) -> String {
        """
        {
          "annotatedImagePath" : "\(annotated.path)",
          "detections" : [
            {
              "box" : { "x1" : 250, "y1" : 312, "x2" : 712, "y2" : 752 },
              "label" : "the cup",
              "maskAreaPixels" : 148220,
              "objectID" : "obj-1",
              "promptKind" : "text",
              "score" : 0.91
            }
          ],
          "inputImagePath" : "\(input.path)",
          "jsonOutputPath" : "\(document.path)",
          "modelID" : "vision-segment-sam31",
          "prompts" : [ "the cup" ],
          "schemaVersion" : 1,
          "threshold" : 0.05
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
        let today = calendar.startOfDay(for: StudioSnapshotRenderer.referenceDate)
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
            status: .running, source: .contract
        )
        guard controller.run(studio: running), let live = runner.liveStarts.last,
              let job = controller.jobs.job(requestID: running.id) else {
            throw StudioSnapshotError.noContentView
        }
        live.stderr("Loading image-zimage-nano\n")
        live.stderr("{\"event\":\"progress\",\"stage\":\"denoising\",\"step\":14,\"total_steps\":24}\n")
        job.markRunning(status: job.status, at: StudioSnapshotRenderer.referenceDate.addingTimeInterval(-41))

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
            status: .queued, source: .contract
        )
        guard controller.run(studio: queued) else { throw StudioSnapshotError.noContentView }
    }

    private static func mockupRequest(mode: StudioMode, draft: StudioDraft, hour: Int, minute: Int) throws -> StudioRunRequest {
        let request = try StudioCommandAdapter.makeRequest(mode: mode, draft: draft, source: .contract)
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

    /// One Image ▸ Generate run that failed a few minutes ago with its command recorded, so the
    /// feed's failed card offers "Use these settings" beside Retry.
    func seedFailedImageRun() throws {
        guard let template = CommandCatalog.template(id: .imageGenerate) else {
            throw StudioSnapshotError.noContentView
        }
        var draft = template.defaultDraft()
        draft.prompt = "a lighthouse keeper's desk, brass instruments, late afternoon"
        draft.model = "image-zimage-turbo"
        draft.width = 1_536
        draft.height = 1_024
        draft.steps = 9
        draft.seed = "8181"
        // Earlier than the mockup's finished runs, so the column's first two rows stay pictures.
        library.upsert(StudioLibraryItem(
            id: UUID(),
            mode: .createImage,
            prompt: draft.prompt,
            inputURL: nil,
            outputURL: nil,
            createdAt: Self.mockupTime(hour: 8, minute: 40),
            updatedAt: Self.mockupTime(hour: 8, minute: 41),
            status: .failed,
            exitCode: 1,
            commandPreview: "mere.run " + template.arguments(from: draft, source: .contract).joined(separator: " "),
            outputText: "error: image-zimage-turbo is not installed; run `mere.run model pull image-zimage-turbo`",
            templateID: .imageGenerate,
            commandDraft: draft
        ))
    }

    /// Runs the Models detail column reads: image generations with the seeded default model
    /// (usage and last-run duration), a passed quality gate, a Lite benchmark, and a running
    /// composer-initiated pull that the job bar and list report.
    func seedModelsLibrary() throws {
        let now = StudioSnapshotRenderer.referenceDate
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

        guard let runner = liveSessionRunner, let template = CommandCatalog.template(id: .modelPull) else {
            throw StudioSnapshotError.noContentView
        }
        runner.liveSessionMarkers = ["pull"]
        var pullDraft = template.defaultDraft()
        pullDraft.model = ModelsInventoryScript.pullingModelID
        let pull = StudioRunRequest(mode: .readImage, templateID: .modelPull, template: template, draft: pullDraft)
        guard controller.modelStore.startPull(pull) else { throw StudioSnapshotError.noContentView }

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
        let today = calendar.startOfDay(for: StudioSnapshotRenderer.referenceDate)
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
        let request = try StudioCommandAdapter.makeRequest(mode: .chat, draft: draft, conversationID: thread.id, source: .contract)
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
            createdAt: StudioSnapshotRenderer.referenceDate.addingTimeInterval(-249)
        )
        let preview = controller.commandPreview(template: template, draft: draft, masksSecrets: true)
        library.start(request: request, commandPreview: preview, status: .running, source: .contract)

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
            createdAt: StudioSnapshotRenderer.referenceDate.addingTimeInterval(-95)
        )
        let preview = controller.commandPreview(template: template, draft: draft, masksSecrets: true)
        library.start(request: request, commandPreview: preview, status: .running, source: .contract)
        guard let runner = liveSessionRunner else { throw StudioSnapshotError.noContentView }
        runner.liveSessionMarkers = ["prepare-masks"]
        guard controller.run(studio: request), let live = runner.liveStarts.last else {
            throw StudioSnapshotError.noContentView
        }
        live.stderr("Preparing SCAIL-2 masks from \(draft.inputPath)\n")
        live.stderr("Segmenting 3 reference images with vision-segment-sam31\n")

        var components = Calendar.current.dateComponents([.year, .month, .day], from: StudioSnapshotRenderer.referenceDate)
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

    // MARK: Specialist results

    /// A finished Who Spoke run: a stand-up recording and the timeline `speech diarize --format
    /// json` wrote for it, made the page's current run the way a real run leaves it.
    func seedDiarizationRun() throws {
        guard let template = CommandCatalog.template(id: .speechDiarize) else {
            throw StudioSnapshotError.noContentView
        }
        let recording = root.appendingPathComponent("standup.wav", isDirectory: false)
        try Self.writeSilentWAV(to: recording, seconds: 20)
        let timeline = root.appendingPathComponent("standup-speakers.json", isDirectory: false)
        try Self.diarizationDocument(source: recording).write(to: timeline, atomically: true, encoding: .utf8)

        var draft = template.defaultDraft()
        draft.inputPath = recording.path
        draft.outputPath = timeline.path
        draft.speechDiarizationFormat = "json"
        let startedAt = Self.mockupTime(hour: 14, minute: 2)
        let row = StudioLibraryItem(
            id: UUID(),
            mode: .listen,
            prompt: "",
            inputURL: recording,
            outputURL: timeline,
            createdAt: startedAt,
            updatedAt: startedAt.addingTimeInterval(6.1),
            status: .completed,
            exitCode: 0,
            commandPreview: "mere.run speech diarize standup.wav --format json --output standup-speakers.json",
            outputText: nil,
            templateID: .speechDiarize,
            commandDraft: draft,
            artifactURLs: [timeline]
        )
        library.upsert(row)
        let scope = StudioTask.audioWhoSpoke.rawValue
        controller.taskSessions.set(Optional(row.id), for: scope + ".requestID")
        controller.taskSessions.set(draft, for: scope + ".Voice.diarizationDraft")
    }

    /// A finished Audio ▸ Enhance run for the task workspace: a narrow-band memo and the 48 kHz
    /// file `audio enhance` wrote beside it, with the task draft pointed at the memo the way the
    /// workspace leaves it after a run.
    func seedEnhanceRun() throws {
        guard let template = CommandCatalog.template(id: .audioEnhance) else {
            throw StudioSnapshotError.noContentView
        }
        let memo = root.appendingPathComponent("voice-memo.wav", isDirectory: false)
        try Self.writeSilentWAV(to: memo, seconds: 12)
        let enhanced = root.appendingPathComponent("voice-memo-48k.wav", isDirectory: false)
        try Self.writeSilentWAV(to: enhanced, seconds: 12)

        var draft = template.defaultDraft()
        draft.inputPath = memo.path
        draft.outputPath = enhanced.path
        let startedAt = Self.mockupTime(hour: 9, minute: 41)
        let request = StudioRunRequest(mode: .listen, templateID: .audioEnhance, template: template, draft: draft)
        var row = StudioLibraryItem(
            id: UUID(),
            mode: .listen,
            prompt: "",
            inputURL: memo,
            outputURL: enhanced,
            createdAt: startedAt,
            updatedAt: startedAt.addingTimeInterval(8.4),
            status: .completed,
            exitCode: 0,
            commandPreview: "mere.run audio enhance voice-memo.wav --output voice-memo-48k.wav",
            outputText: nil,
            templateID: .audioEnhance,
            commandDraft: draft,
            commandArguments: template.arguments(from: request.draft, source: .contract),
            artifactURLs: [enhanced]
        )
        row.inputIdentity = StudioInputIdentity.read(memo)
        library.upsert(row)
        var taskDraft = StudioTaskDraft(templateID: .audioEnhance)
        taskDraft.setArgument(0, memo.path)
        controller.taskSessions.setTaskDraft(taskDraft, for: .audioEnhance)
        controller.taskSessions.set(Optional(row.id), for: StudioTask.audioEnhance.rawValue + ".requestID")
    }

    /// One finished run per Sound task, each with its task draft parked: Video Foley's clip and
    /// the WAV made for it, Condition's safetensors, Encode's `.npy`, Decode's WAV from it, and
    /// Score's printed CLAP result. Rows file under Sound ▸ Generate's mode, as the SFX Lab
    /// page filed them.
    func seedSoundRuns() throws {
        let sound = root.appendingPathComponent("sound", isDirectory: true)
        try FileManager.default.createDirectory(at: sound, withIntermediateDirectories: true)
        let clip = sound.appendingPathComponent("walk.mp4", isDirectory: false)
        try Self.writeFixtureMP4(to: clip, size: CGSize(width: 640, height: 360), frames: 36)
        let foley = sound.appendingPathComponent("walk.wav", isDirectory: false)
        try Self.writeSilentWAV(to: foley, seconds: 3)
        let conditioning = sound.appendingPathComponent("heavy-wooden-door.safetensors", isDirectory: false)
        try TensorFixtures.safetensors(
            [("text_embeddings", [1, 77, 1_024]), ("pooled_embedding", [1, 1_024]), ("attention_mask", [1, 77])],
            metadata: ["model": "sfx-woosh-dflow"]
        ).write(to: conditioning, options: .atomic)
        let hit = sound.appendingPathComponent("hit.wav", isDirectory: false)
        try Self.writeSilentWAV(to: hit, seconds: 2)
        let latents = sound.appendingPathComponent("hit.npy", isDirectory: false)
        try TensorFixtures.npy(descriptor: "<f4", shape: "(1, 128, 87)").write(to: latents, options: .atomic)
        let decoded = sound.appendingPathComponent("hit-decoded.wav", isDirectory: false)
        try Self.writeSilentWAV(to: decoded, seconds: 2)
        let bottle = sound.appendingPathComponent("bottle.wav", isDirectory: false)
        try Self.writeSilentWAV(to: bottle, seconds: 4)

        struct Run {
            let task: StudioTask
            let templateID: CommandTemplateID
            let prompt: String
            let input: URL?
            let output: URL?
            let outputText: String?
            let startedAt: Date
            let elapsed: TimeInterval
            let edit: (inout CommandDraft) -> Void
        }
        let runs: [Run] = [
            Run(task: .soundFoley, templateID: .sfxVideo, prompt: "Footsteps on wet gravel, close perspective", input: clip,
                output: foley, outputText: nil, startedAt: Self.mockupTime(hour: 14, minute: 2), elapsed: 46) {
                $0.sfxRenoise = "0.35"
                $0.seed = "11"
            },
            Run(task: .soundCondition, templateID: .sfxConditionText, prompt: "Heavy wooden door creaking open", input: nil,
                output: conditioning, outputText: nil, startedAt: Self.mockupTime(hour: 13, minute: 48), elapsed: 3.2) { _ in },
            Run(task: .soundEncode, templateID: .sfxAEEncode, prompt: "", input: hit, output: latents, outputText: nil,
                startedAt: Self.mockupTime(hour: 13, minute: 40), elapsed: 2.1) { _ in },
            Run(task: .soundDecode, templateID: .sfxAEDecode, prompt: "", input: latents, output: decoded, outputText: nil,
                startedAt: Self.mockupTime(hour: 13, minute: 42), elapsed: 1.7) { _ in },
            Run(task: .soundScore, templateID: .sfxClapScore, prompt: "A glass bottle breaking on concrete", input: bottle,
                output: nil, outputText: """
                Loading sfx-woosh-clap
                {"prompt":"A glass bottle breaking on concrete","score":0.634,"audio":"\(bottle.path)","model":"sfx-woosh-clap"}
                """, startedAt: Self.mockupTime(hour: 13, minute: 55), elapsed: 1.4) { _ in },
        ]
        for run in runs {
            guard let template = CommandCatalog.template(id: run.templateID) else { throw StudioSnapshotError.noContentView }
            var draft = template.defaultDraft()
            draft.prompt = run.prompt
            draft.inputPath = run.input?.path ?? ""
            draft.outputPath = run.output?.path ?? ""
            run.edit(&draft)
            let arguments = template.arguments(from: draft, source: .contract)
            var row = StudioLibraryItem(
                id: UUID(),
                mode: .sfx,
                prompt: run.prompt,
                inputURL: run.input,
                outputURL: run.output,
                createdAt: run.startedAt,
                updatedAt: run.startedAt.addingTimeInterval(run.elapsed),
                status: .completed,
                exitCode: 0,
                commandPreview: (["mere.run"] + arguments).joined(separator: " "),
                outputText: run.outputText,
                templateID: run.templateID,
                commandDraft: draft,
                commandArguments: arguments,
                artifactURLs: run.output.map { [$0] } ?? []
            )
            if let input = run.input { row.inputIdentity = StudioInputIdentity.read(input) }
            library.upsert(row)
            // The parked draft is the run's settings with the destination left to routing.
            var parked = draft
            parked.outputPath = ""
            let taskDraft = StudioTaskDraft(templateID: run.templateID, form: StudioConsoleCommand.seed(template: template, draft: parked, source: .contract))
            controller.taskSessions.setTaskDraft(taskDraft, for: run.task)
            controller.taskSessions.set(Optional(row.id), for: run.task.rawValue + ".requestID")
        }
        controller.taskSessions.set(StudioRenoise.Mode.amount, for: StudioTask.soundFoley.rawValue + ".renoiseMode")
    }

    /// One finished run per Vision task on the task workspace — Detect faces, Pose, Flow, Depth,
    /// Geometry, and a Live capture — on a drawn 960×720 picture, with each task's draft pointed
    /// at its input and its run remembered the way the workspace leaves them. Returns the row id
    /// per task so a board can select it.
    @discardableResult
    func seedVisionRuns() throws -> [StudioTask: UUID] {
        let directory = root.appendingPathComponent("vision", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let size = CGSize(width: 960, height: 720)
        let portrait = directory.appendingPathComponent("portrait.png", isDirectory: false)
        try Self.writeFixturePNG(to: portrait, size: size, hueOffset: 0.15)
        portraitURL = portrait
        let second = directory.appendingPathComponent("portrait-later.png", isDirectory: false)
        try Self.writeFixturePNG(to: second, size: size, hueOffset: 0.32)
        secondPortraitURL = second
        var seeded: [StudioTask: UUID] = [:]

        func seed(
            _ templateID: CommandTemplateID, task: StudioTask, inputs: [URL], output: URL?, artifacts: [URL],
            prompt: String = "", minute: Int
        ) throws {
            guard let template = CommandCatalog.template(id: templateID) else { throw StudioSnapshotError.noContentView }
            var draft = StudioTaskDraft(templateID: templateID)
            draft.form["--dry-run"] = .unset
            let slots = StudioTaskSchema.slots(for: templateID)
            if let first = slots.first {
                if first.allowsMultiple {
                    first.attach(inputs, to: &draft)
                } else {
                    for (slot, input) in zip(slots, inputs) { slot.attach([input], to: &draft) }
                }
            }
            if !prompt.isEmpty { draft.prompt = prompt }
            if let output, let flag = draft.capability?.output.flag { draft.form[flag] = .text(output.path) }
            if let document = artifacts.first(where: { $0.pathExtension == "json" }),
               draft.capability?.options.contains(where: { $0.flag == "--json-output" }) == true,
               draft.text("--json-output").isEmpty {
                draft.form["--json-output"] = .text(document.path)
            }
            let startedAt = Self.mockupTime(hour: 14, minute: minute)
            var row = StudioLibraryItem(
                id: UUID(),
                mode: template.libraryMode,
                prompt: prompt,
                inputURL: inputs.first,
                outputURL: output,
                createdAt: startedAt,
                updatedAt: startedAt.addingTimeInterval(2.6),
                status: .completed,
                exitCode: 0,
                commandPreview: "mere.run " + draft.arguments(source: .contract).joined(separator: " "),
                outputText: nil,
                templateID: templateID,
                commandDraft: draft.run(source: .contract)?.commandDraft,
                commandArguments: draft.arguments(source: .contract),
                artifactURLs: artifacts
            )
            if let input = inputs.first { row.inputIdentity = StudioInputIdentity.read(input) }
            library.upsert(row)
            controller.taskSessions.setTaskDraft(draft, for: task)
            controller.taskSessions.set(Optional(row.id), for: task.rawValue + ".requestID")
            seeded[task] = row.id
        }

        let faces = directory.appendingPathComponent("portrait-faces.json", isDirectory: false)
        try Self.faceDocument(size: size).write(to: faces, atomically: true, encoding: .utf8)
        try seed(.visionFaceDetect, task: .visionFaces, inputs: [portrait], output: nil, artifacts: [faces], minute: 2)

        let pose = directory.appendingPathComponent("portrait-pose.json", isDirectory: false)
        try Self.poseDocument(size: size).write(to: pose, atomically: true, encoding: .utf8)
        try seed(.visionPose, task: .visionPose, inputs: [portrait], output: pose, artifacts: [pose], minute: 4)

        let flow = directory.appendingPathComponent("portrait-motion.flo", isDirectory: false)
        try Self.flowField(width: 96, height: 72).write(to: flow, options: .atomic)
        let flowDocument = directory.appendingPathComponent("portrait-motion.json", isDirectory: false)
        try "{\"width\":96,\"height\":72}".write(to: flowDocument, atomically: true, encoding: .utf8)
        try seed(.visionFlow, task: .visionFlow, inputs: [portrait, second], output: flow, artifacts: [flow, flowDocument], minute: 6)

        let depthDirectory = directory.appendingPathComponent("portrait-depth", isDirectory: true)
        try FileManager.default.createDirectory(at: depthDirectory, withIntermediateDirectories: true)
        let depthPreview = depthDirectory.appendingPathComponent("portrait-depth.png", isDirectory: false)
        try Self.writeDepthPNG(to: depthPreview, size: size)
        let depthManifest = depthDirectory.appendingPathComponent("portrait-depth.json", isDirectory: false)
        try Self.depthManifest(size: size, outputDirectory: depthDirectory).write(to: depthManifest, atomically: true, encoding: .utf8)
        try seed(.visionDepth, task: .visionDepth, inputs: [portrait], output: depthDirectory,
                 artifacts: [depthPreview, depthManifest], minute: 8)

        let sceneDirectory = directory.appendingPathComponent("portrait-scene", isDirectory: true)
        try FileManager.default.createDirectory(at: sceneDirectory, withIntermediateDirectories: true)
        let points = sceneDirectory.appendingPathComponent("portrait-points.ply", isDirectory: false)
        try Self.pointCloudPLY().write(to: points, atomically: true, encoding: .utf8)
        let sceneDepth = sceneDirectory.appendingPathComponent("portrait-depth.png", isDirectory: false)
        try Self.writeDepthPNG(to: sceneDepth, size: size)
        let sceneNormal = sceneDirectory.appendingPathComponent("portrait-normal.png", isDirectory: false)
        try Self.writeFixturePNG(to: sceneNormal, size: size, hueOffset: 0.55)
        try seed(.visionGeometry, task: .visionGeometry, inputs: [portrait], output: sceneDirectory,
                 artifacts: [points, sceneDepth, sceneNormal], minute: 10)

        let clip = directory.appendingPathComponent("live-tracking.mp4", isDirectory: false)
        try Self.writeFixtureMP4(to: clip, size: CGSize(width: 640, height: 360), frames: 24)
        let tracking = directory.appendingPathComponent("live-tracking.json", isDirectory: false)
        try Self.trackingDocument(clip: clip).write(to: tracking, atomically: true, encoding: .utf8)
        try seed(.visionTrackLive, task: .visionLive, inputs: [], output: clip, artifacts: [clip, tracking],
                 prompt: "the person", minute: 12)
        return seeded
    }

    /// `vision face detect --json-output` for the drawn portrait: two faces with five landmarks
    /// each, in stored pixels, numbered the way `--face-index` counts them.
    private static func faceDocument(size: CGSize) -> String {
        func face(_ index: Int, x: Double, y: Double, width: Double, height: Double, score: Double) -> String {
            let landmarks = [(0.3, 0.38), (0.7, 0.38), (0.5, 0.58), (0.35, 0.78), (0.65, 0.78)]
                .map { "{\"x\":\(x + $0.0 * width),\"y\":\(y + $0.1 * height)}" }
                .joined(separator: ",")
            return """
            {"index":\(index),"detection":{"score":\(score),"boundingBox":{"x":\(x),"y":\(y),"width":\(width),"height":\(height)},"landmarks":[\(landmarks)]}}
            """
        }
        return """
        {"elapsedMilliseconds":412.5,"height":\(Int(size.height)),"image":"portrait.png","modelID":"vision-face-buffalo-l","width":\(Int(size.width)),
         "faces":[\(face(0, x: 268, y: 150, width: 196, height: 236, score: 0.93)),\(face(1, x: 560, y: 210, width: 150, height: 184, score: 0.81))]}
        """
    }

    /// `vision pose --json-output`: a body and a hand, normalized with the origin top-left.
    private static func poseDocument(size: CGSize) -> String {
        let body = [
            ("nose", 0.38, 0.28), ("leftShoulder", 0.30, 0.42), ("rightShoulder", 0.47, 0.42), ("leftElbow", 0.24, 0.56),
            ("rightElbow", 0.53, 0.55), ("leftWrist", 0.22, 0.70), ("rightWrist", 0.58, 0.66), ("leftHip", 0.33, 0.72),
            ("rightHip", 0.44, 0.72), ("leftKnee", 0.32, 0.88), ("rightKnee", 0.45, 0.88),
        ]
        let hand = [("wrist", 0.58, 0.66), ("thumbTip", 0.62, 0.62), ("indexTip", 0.63, 0.66), ("middleTip", 0.63, 0.69), ("littleTip", 0.61, 0.72)]
        func points(_ list: [(String, Double, Double)]) -> String {
            list.map { "{\"name\":\"\($0.0)\",\"x\":\($0.1),\"y\":\($0.2),\"confidence\":0.86}" }.joined(separator: ",")
        }
        return """
        {"imageWidth":\(Int(size.width)),"imageHeight":\(Int(size.height)),"coordinateSpace":"normalized",
         "subjects":[{"kind":"body","index":0,"points":[\(points(body))]},{"kind":"hand","index":0,"points":[\(points(hand))]}]}
        """
    }

    /// A Middlebury `.flo` with a gentle swirl, so the vectors read as motion.
    private static func flowField(width: Int, height: Int) -> Data {
        var data = Data()
        func append(_ value: UInt32) { withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) } }
        append(Float(202_021.25).bitPattern)
        append(UInt32(width))
        append(UInt32(height))
        for y in 0..<height {
            for x in 0..<width {
                let dx = Double(x) / Double(width) - 0.5
                let dy = Double(y) / Double(height) - 0.5
                append(Float(-dy * 6).bitPattern)
                append(Float(dx * 6).bitPattern)
            }
        }
        return data
    }

    /// A grayscale ramp with a brighter oval, the shape a depth preview has.
    private static func writeDepthPNG(to url: URL, size: CGSize) throws {
        let width = Int(size.width)
        let height = Int(size.height)
        guard let rep = NSBitmapImageRep(
            bitmapDataPlanes: nil, pixelsWide: width, pixelsHigh: height, bitsPerSample: 8, samplesPerPixel: 4,
            hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0
        ), let context = NSGraphicsContext(bitmapImageRep: rep) else {
            throw StudioSnapshotError.noBitmap
        }
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = context
        NSGradient(starting: NSColor(calibratedWhite: 0.08, alpha: 1), ending: NSColor(calibratedWhite: 0.82, alpha: 1))?
            .draw(in: CGRect(x: 0, y: 0, width: width, height: height), angle: 90)
        NSColor(calibratedWhite: 0.96, alpha: 1).setFill()
        NSBezierPath(ovalIn: CGRect(x: width / 4, y: height / 5, width: width / 3, height: height / 2)).fill()
        NSGraphicsContext.restoreGraphicsState()
        guard let data = rep.representation(using: .png, properties: [:]) else {
            throw StudioSnapshotError.pngEncodingFailed
        }
        try data.write(to: url, options: .atomic)
    }

    /// `vision depth`'s `<stem>-depth.json` (`MarigoldV2DepthManifest`) for the seeded run.
    private static func depthManifest(size: CGSize, outputDirectory: URL) -> String {
        """
        {"schemaVersion":1,"createdAt":"2026-09-04T14:08:02Z","inputPath":"portrait.png","inputByteCount":40960,"inputSHA256":"0",
         "outputDirectory":"\(outputDirectory.path)","width":\(Int(size.width)),"height":\(Int(size.height)),
         "inferenceWidth":1024,"inferenceHeight":768,"semantics":"affine-relative","parameterization":"log",
         "checkpoint":"log-stage2","seeThrough":false,
         "depthStatistics":{"rawMinimum":0.018,"rawMaximum":0.974,"normalizationNear":0.01,"normalizationFar":0.99},
         "model":{"modelID":"vision-depth-marigold-v2","upstreamRepository":"prs-eth/marigold-depth-v2","upstreamRevision":"main",
                  "license":"Apache-2.0","inferenceBackend":"mlx"},
         "artifacts":[]}
        """
    }

    /// A small colored point cloud, as `vision geometry` writes its PLY.
    private static func pointCloudPLY() -> String {
        var lines = ["ply", "format ascii 1.0", "element vertex 64", "property float x", "property float y", "property float z",
                     "property uchar red", "property uchar green", "property uchar blue", "end_header"]
        for index in 0..<64 {
            let x = Double(index % 8) / 7 - 0.5
            let y = Double(index / 8) / 7 - 0.5
            let z = (x * x + y * y) * 0.6
            lines.append(String(format: "%.3f %.3f %.3f %d %d %d", x, y, z, 90 + index * 2, 120, 200 - index))
        }
        return lines.joined(separator: "\n") + "\n"
    }

    /// `vision track-live --json-output` (`SAM31TrackingRun`) for the seeded clip: one object,
    /// visible for the first three quarters of the capture.
    private static func trackingDocument(clip: URL) -> String {
        let frames = (0..<24).map { index -> String in
            let visible = index < 18
            let x = 200 + index * 8
            return """
            {"frameIndex":\(index),"timestampSeconds":\(Double(index) / 12),"detections":[{"objectID":"obj-1","label":"the person","score":\(visible ? 0.91 : 0),"visible":\(visible),"box":{"x1":\(x),"y1":80,"x2":\(x + 140),"y2":330}}]}
            """
        }.joined(separator: ",")
        return """
        {"schemaVersion":1,"modelID":"vision-segment-sam31","inputVideoPath":"camera:0","annotatedVideoPath":"\(clip.path)",
         "fps":12,"frameWidth":640,"frameHeight":360,"objects":[{"objectID":"obj-1","label":"the person","seedFrameIndex":0}],"frames":[\(frames)]}
        """
    }

    /// A finished Separate run for `task` (Audio ▸ Separate or Music ▸ Separate): a track, the
    /// two stems `music separate` wrote beside it, and the manifest it printed and saved, with
    /// the task draft pointed at the track.
    func seedSeparateRun(task: StudioTask) throws {
        guard let template = CommandCatalog.template(id: .musicSeparate) else {
            throw StudioSnapshotError.noContentView
        }
        let track = root.appendingPathComponent("late-set.wav", isDirectory: false)
        try Self.writeSilentWAV(to: track, seconds: 12)
        let stems = root.appendingPathComponent("late-set-stems", isDirectory: true)
        try FileManager.default.createDirectory(at: stems, withIntermediateDirectories: true)
        var stemURLs: [URL] = []
        for name in ["vocals", "instrumental"] {
            let url = stems.appendingPathComponent("\(name).wav", isDirectory: false)
            try Self.writeSilentWAV(to: url, seconds: 12)
            stemURLs.append(url)
        }
        let manifestURL = stems.appendingPathComponent("separation.json", isDirectory: false)
        let manifest = Self.separationManifest(source: track, stems: stemURLs, manifest: manifestURL)
        try manifest.write(to: manifestURL, atomically: true, encoding: .utf8)

        var draft = template.defaultDraft()
        draft.inputPath = track.path
        draft.outputPath = stems.path
        let startedAt = Self.mockupTime(hour: 16, minute: 12)
        var row = StudioLibraryItem(
            id: UUID(),
            mode: .music,
            prompt: "",
            inputURL: track,
            outputURL: stems,
            createdAt: startedAt,
            updatedAt: startedAt.addingTimeInterval(21.5),
            status: .completed,
            exitCode: 0,
            commandPreview: "mere.run music separate late-set.wav --output-dir late-set-stems",
            outputText: manifest,
            templateID: .musicSeparate,
            commandDraft: draft,
            commandArguments: template.arguments(from: draft, source: .contract),
            artifactURLs: [stems] + stemURLs + [manifestURL]
        )
        row.inputIdentity = StudioInputIdentity.read(track)
        library.upsert(row)
        var taskDraft = StudioTaskDraft(templateID: .musicSeparate)
        taskDraft.setArgument(0, track.path)
        controller.taskSessions.setTaskDraft(taskDraft, for: task)
        controller.taskSessions.set(Optional(row.id), for: task.rawValue + ".requestID")
    }

    /// Two saved voices for Voice ▸ Voices, their references written here so the detail's
    /// player has a file to load.
    func voiceProfileSeed() throws -> [StudioVoiceProfileRecord] {
        let made = Self.mockupTime(hour: 10, minute: 5)
        var records: [StudioVoiceProfileRecord] = []
        let voices: [(id: String, name: String, language: String?, transcript: String)] = [
            ("6F9B2C1E-0D44-4C1B-9A7E-3B2C4D5E6F70", "Narrator", "en",
             "Harbour lights are blinking slow on the water where the old boats go. I left my coat on the ferry rail and watched the evening turn to pale."),
            ("A1B2C3D4-E5F6-4A7B-8C9D-0E1F2A3B4C5D", "Field host", nil,
             "Good morning everyone, and thank you for joining the quarterly review."),
        ]
        for voice in voices {
            let reference = root.appendingPathComponent("\(voice.name.lowercased().replacingOccurrences(of: " ", with: "-"))-reference.wav")
            try Self.writeSilentWAV(to: reference, seconds: 6)
            records.append(StudioVoiceProfileRecord(
                id: UUID(uuidString: voice.id)!, name: voice.name, createdAt: made, updatedAt: made,
                transcript: voice.transcript, language: voice.language,
                referenceAudioRelativePath: reference.path, modelFingerprint: nil
            ))
        }
        return records
    }

    /// A `speech listen` session started through the task runner and held open by the process
    /// seam, with its ready event already on stdout, the way the page finds one when it appears.
    func seedLiveListenSession() throws -> UUID {
        guard let runner = liveSessionRunner else {
            throw StudioSnapshotError.noContentView
        }
        runner.liveSessionMarkers = ["listen"]
        var draft = StudioTaskDraft(templateID: .speechListen)
        draft.form["--language"] = .text("en")
        controller.taskSessions.setTaskDraft(draft, for: .audioLive)
        controller.checkReadiness(for: .audioLive, modelID: StudioTaskSchema.modelID(for: draft, source: .contract))
        let request = try StudioTaskRunner(controller: controller, library: library).run(draft.liveListenLaunch(), task: .audioLive)
        guard let live = runner.liveStarts.last else {
            throw StudioSnapshotError.noContentView
        }
        live.stdout(#"{"protocol":1,"type":"ready"}"# + "\n")
        live.stderr("Listening. Press Ctrl-C to stop.\n")
        return request.id
    }

    /// Two committed utterances and a partial one, as `speech listen --jsonl` streams them.
    func speakIntoLiveListenSession() {
        guard let live = liveSessionRunner?.liveStarts.last else { return }
        live.stdout(#"{"protocol":1,"type":"commit","utteranceId":"u1","revision":4,"text":"Good morning everyone, and thank you for joining the quarterly review."}"# + "\n")
        live.stdout(#"{"protocol":1,"type":"commit","utteranceId":"u2","revision":3,"text":"Today we will walk through the roadmap and the numbers behind it."}"# + "\n")
        live.stdout(#"{"protocol":1,"type":"partial","utteranceId":"u3","revision":2,"text":"Before we start, I want to flag that the shipping dates"}"# + "\n")
        live.stderr("Committed utterance u2\n")
    }

    /// A finished 3D ▸ TripoSR run of the mug: its output folder holding a small OBJ Quick Look
    /// can draw and the two manifests the CLI writes beside a mesh, with the task draft pointed
    /// at the picture on the TripoSR engine. Returns four view pictures for an InstantMesh draft.
    func seedMeshRun() throws -> [URL] {
        guard let template = CommandCatalog.template(id: .imageReconstruct3D) else {
            throw StudioSnapshotError.noContentView
        }
        let folder = root.appendingPathComponent("3D/mug", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let mesh = folder.appendingPathComponent("mug.glb", isDirectory: false)
        try Self.cubeGLB().write(to: mesh, options: .atomic)
        let manifest = folder.appendingPathComponent("mug-manifest.json", isDirectory: false)
        try """
        {"schemaVersion": 1, "inputPaths": ["\(mugURL.path)"], "outputDirectory": "\(folder.path)",
         "coordinateSystem": "x-right-y-up-z-forward", "units": "normalized-object-space", "inferredUnseenGeometry": true,
         "vertexCount": 12480, "triangleCount": 24956, "bounds": {"min": [-0.5, -0.5, -0.5], "max": [0.5, 0.5, 0.5]}, "artifacts": []}
        """.write(to: manifest, atomically: true, encoding: .utf8)
        let runManifest = folder.appendingPathComponent("mug-run-manifest.json", isDirectory: false)
        try """
        {"schemaVersion": 1, "outputDirectory": "\(folder.path)",
         "mesh": {"coordinateSystem": "x-right-y-up-z-forward", "units": "normalized-object-space", "inferredUnseenGeometry": true,
                  "vertexCount": 12480, "triangleCount": 24956, "bounds": {"min": [-0.5, -0.5, -0.5], "max": [0.5, 0.5, 0.5]}},
         "artifacts": []}
        """.write(to: runManifest, atomically: true, encoding: .utf8)

        var draft = template.defaultDraft()
        draft.inputPath = mugURL.path
        draft.outputPath = folder.path
        let startedAt = Self.mockupTime(hour: 10, minute: 12)
        let request = StudioRunRequest(mode: .createImage, templateID: .imageReconstruct3D, template: template, draft: draft)
        var row = StudioLibraryItem(
            id: UUID(),
            mode: .createImage,
            prompt: "",
            inputURL: mugURL,
            outputURL: mesh,
            createdAt: startedAt,
            updatedAt: startedAt.addingTimeInterval(41),
            status: .completed,
            exitCode: 0,
            commandPreview: "mere.run image reconstruct-3d mug.png --output 3D/mug --resolution 256",
            outputText: nil,
            templateID: .imageReconstruct3D,
            commandDraft: draft,
            commandArguments: template.arguments(from: request.draft, source: .contract),
            artifactURLs: [mesh, manifest, runManifest],
            artifactRoles: [
                manifest.standardizedFileURL.path: "mesh-manifest-json",
                runManifest.standardizedFileURL.path: "triposr-run-manifest-json",
            ]
        )
        row.inputIdentity = StudioInputIdentity.read(mugURL)
        library.upsert(row)
        var taskDraft = StudioTaskDraft(templateID: .imageReconstruct3D)
        taskDraft.setArgument(0, mugURL.path)
        controller.taskSessions.setTaskDraft(taskDraft, for: .threeDFromImage)
        controller.taskSessions.set(Optional(row.id), for: StudioTask.threeDFromImage.rawValue + ".requestID")

        return try ["front", "right", "back", "left"].map { name in
            let url = root.appendingPathComponent("\(name).png", isDirectory: false)
            try Self.writeMugPNG(to: url, side: 256)
            return url
        }
    }

    /// A unit cube as binary glTF: eight corners and twelve triangles in one buffer, the shape the
    /// 3D commands' GLB exports take.
    private static func cubeGLB() -> Data {
        let corners: [Float] = [
            -0.5, -0.5, -0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, -0.5,
            -0.5, -0.5, 0.5, 0.5, -0.5, 0.5, 0.5, 0.5, 0.5, -0.5, 0.5, 0.5,
        ]
        let triangles: [UInt16] = [
            0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4,
            1, 2, 6, 1, 6, 5, 2, 3, 7, 2, 7, 6, 3, 0, 4, 3, 4, 7,
        ]
        var binary = Data()
        triangles.forEach { binary.append(contentsOf: withUnsafeBytes(of: $0.littleEndian, Array.init)) }
        corners.forEach { binary.append(contentsOf: withUnsafeBytes(of: $0.bitPattern.littleEndian, Array.init)) }
        let indexBytes = triangles.count * 2
        var json = Data("""
        {"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0]}],"nodes":[{"mesh":0}],
        "meshes":[{"primitives":[{"attributes":{"POSITION":1},"indices":0}]}],
        "buffers":[{"byteLength":\(binary.count)}],
        "bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":\(indexBytes),"target":34963},
        {"buffer":0,"byteOffset":\(indexBytes),"byteLength":\(corners.count * 4),"target":34962}],
        "accessors":[{"bufferView":0,"componentType":5123,"count":\(triangles.count),"type":"SCALAR"},
        {"bufferView":1,"componentType":5126,"count":\(corners.count / 3),"type":"VEC3","min":[-0.5,-0.5,-0.5],"max":[0.5,0.5,0.5]}]}
        """.utf8)
        while json.count % 4 != 0 { json.append(0x20) }
        func word(_ value: UInt32) -> [UInt8] { withUnsafeBytes(of: value.littleEndian, Array.init) }
        var glb = Data()
        glb.append(contentsOf: word(0x4654_6C67))
        glb.append(contentsOf: word(2))
        glb.append(contentsOf: word(UInt32(12 + 8 + json.count + 8 + binary.count)))
        glb.append(contentsOf: word(UInt32(json.count)))
        glb.append(contentsOf: word(0x4E4F_534A))
        glb.append(json)
        glb.append(contentsOf: word(UInt32(binary.count)))
        glb.append(contentsOf: word(0x004E_4942))
        glb.append(binary)
        return glb
    }

    /// The Earth board's state: a finished TESSERA run over a four-observation bundle written
    /// here (its 64-wide embedding beside it under Earth, and the JSON the command printed as
    /// the row's output text), the bundle parked in TESSERA's task draft; and a Flood task draft
    /// pointed at a bundle without its DEM, so the checklist has something to warn about.
    func seedEarthRuns() throws {
        guard let template = CommandCatalog.template(id: .geoTessera) else {
            throw StudioSnapshotError.noContentView
        }
        let bundle = root.appendingPathComponent("valley-2024.safetensors", isDirectory: false)
        try TensorFixtures.write(to: bundle, tensors: [
            .float32("S2", shape: [1, 4, 10], value: 1_200),
            .float32("S2_DOY", shape: [1, 4], value: 120),
            .float32("S1_ASC", shape: [1, 4, 2], value: -12),
            .float32("S1_ASC_DOY", shape: [1, 4], value: 118),
        ])
        let embedding = root.appendingPathComponent("Earth/valley-2024-a1b2c3.safetensors", isDirectory: false)
        try TensorFixtures.write(
            to: embedding,
            tensors: [.float32("embeddings", shape: [1, 64], value: 0.031)],
            metadata: [
                "format": "mere.run/tessera-v2-embeddings-v1", "model_id": "vision-embed-tessera-v2-large",
                "source_revision": "4f1c2e9", "dimensions": "64",
            ]
        )

        var draft = template.defaultDraft()
        draft.inputPath = bundle.path
        draft.outputPath = embedding.path
        draft.model = "vision-embed-tessera-v2-large"
        draft.geoDimensions = "64"
        let startedAt = Self.mockupTime(hour: 10, minute: 12)
        let request = StudioRunRequest(mode: template.libraryMode, templateID: .geoTessera, template: template, draft: draft)
        var row = StudioLibraryItem(
            id: UUID(),
            mode: template.libraryMode,
            prompt: "",
            inputURL: bundle,
            outputURL: embedding,
            createdAt: startedAt,
            updatedAt: startedAt.addingTimeInterval(6.2),
            status: .completed,
            exitCode: 0,
            commandPreview: "mere.run geo tessera valley-2024.safetensors --output valley-2024-a1b2c3.safetensors --dimensions 64 --json",
            outputText: """
            {
              "batch_size" : 1,
              "device" : "metal",
              "inference_seconds" : 0.41,
              "input_path" : "\(bundle.path)",
              "model_id" : "vision-embed-tessera-v2-large",
              "model_load_seconds" : 5.8,
              "operation" : "time-series-embedding",
              "output_path" : "\(embedding.path)",
              "schema_version" : 1,
              "status" : "completed",
              "variant" : "large"
            }
            """,
            templateID: .geoTessera,
            commandDraft: draft,
            commandArguments: template.arguments(from: request.draft, source: .contract),
            artifactURLs: [embedding]
        )
        row.inputIdentity = StudioInputIdentity.read(bundle)
        library.upsert(row)
        var taskDraft = StudioTaskDraft(templateID: .geoTessera)
        taskDraft.setArgument(0, bundle.path)
        // No model in the draft: the scripted inventory does not list the TESSERA checkpoints, so
        // a named one would only draw the "couldn't check" card over the result.
        taskDraft.form["--dimensions"] = .integer(64)
        controller.taskSessions.setTaskDraft(taskDraft, for: .earthTessera)
        controller.taskSessions.set(Optional(row.id), for: StudioTask.earthTessera.rawValue + ".requestID")

        let incomplete = root.appendingPathComponent("delta-tiles.safetensors", isDirectory: false)
        try TensorFixtures.write(to: incomplete, tensors: [
            .float32("S2L2A", shape: [1, 12, 4, 8, 8], value: 0.2),
            .float32("S1RTC", shape: [1, 2, 4, 8, 8], value: -0.4),
        ])
        var flood = StudioTaskDraft(templateID: .geoFlood)
        flood.setArgument(0, incomplete.path)
        controller.taskSessions.setTaskDraft(flood, for: .earthFlood)
    }

    /// Two finished Image ▸ Train runs the dashboard can follow and compare: a dataset folder of
    /// six captioned pictures, an adapter for it with a 40-step loss log, three preview samples,
    /// and two checkpoints beside it (the run the page follows), and an earlier, shorter run for
    /// the B side of the comparison. The task draft is parked on the dataset with a recipe.
    func seedTrainingRuns() throws {
        guard let template = CommandCatalog.template(id: .imageTrainLoRA) else { throw StudioSnapshotError.noContentView }
        let dataset = root.appendingPathComponent("datasets/warm-still-life", isDirectory: true)
        try FileManager.default.createDirectory(at: dataset, withIntermediateDirectories: true)
        let captions = [
            "a ceramic mug on linen in soft morning light", "a pear on a wooden board, warm window light",
            "a stack of letters tied with twine", "a brass candlestick beside a folded napkin",
            "dried flowers in a glass bottle on a sill", "a bowl of walnuts on a dark table",
        ]
        for (index, caption) in captions.enumerated() {
            try Self.writeFixturePNG(to: dataset.appendingPathComponent("still-\(index + 1).png"), size: CGSize(width: 512, height: 512),
                                     hueOffset: CGFloat(index) * 0.13)
            try caption.write(to: dataset.appendingPathComponent("still-\(index + 1).txt"), atomically: true, encoding: .utf8)
        }

        let folder = root.appendingPathComponent("training/Image", isDirectory: true)
        func seedRun(stem: String, steps: Int, loss: (Int) -> Double, samples: Int, hour: Int, minute: Int) throws -> StudioLibraryItem {
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            let adapter = folder.appendingPathComponent("\(stem).safetensors")
            try Data(repeating: 0, count: 64).write(to: adapter)
            var events: [String] = []
            for index in 0..<40 {
                let step = max(1, (index + 1) * steps / 40)
                events.append(
                    "{\"sequence\": \(index + 1), \"type\": \"step\", \"stage\": \"train\", \"step\": \(step), " +
                    "\"total_steps\": \(steps), \"loss\": \(String(format: "%.5f", loss(step))), \"fraction\": \(Double(step) / Double(steps))}"
                )
            }
            events.append("{\"sequence\": 41, \"type\": \"run_finished\", \"stage\": \"finished\", \"step\": \(steps), \"total_steps\": \(steps), \"fraction\": 1, \"path\": \"\(adapter.path)\"}")
            try events.joined(separator: "\n").write(to: folder.appendingPathComponent("\(stem).events.jsonl"), atomically: true, encoding: .utf8)
            let sampleFolder = folder.appendingPathComponent("samples", isDirectory: true)
            try FileManager.default.createDirectory(at: sampleFolder, withIntermediateDirectories: true)
            for index in 0..<samples {
                try Self.writeFixturePNG(to: sampleFolder.appendingPathComponent("\(stem)-step-\(String(format: "%04d", (index + 1) * 250)).png"),
                                         size: CGSize(width: 512, height: 512), hueOffset: 0.6 + CGFloat(index) * 0.1)
            }
            let checkpointFolder = folder.appendingPathComponent("checkpoints", isDirectory: true)
            try FileManager.default.createDirectory(at: checkpointFolder, withIntermediateDirectories: true)
            for step in stride(from: 250, through: steps, by: 250) where step < steps {
                try Data(repeating: 0, count: 16).write(to: checkpointFolder.appendingPathComponent("\(stem)-checkpoint-step\(step).safetensors"))
            }

            var draft = template.defaultDraft()
            draft.inputPath = dataset.path
            draft.outputPath = adapter.path
            draft.trainingRecipe = "krea-fast-style"
            draft.seed = "42"
            draft.checkpointInterval = 250
            draft.sampleInterval = 250
            draft.steps = steps
            let startedAt = Self.mockupTime(hour: hour, minute: minute)
            var row = StudioLibraryItem(
                id: UUID(),
                mode: .createImage,
                prompt: "",
                inputURL: dataset,
                outputURL: adapter,
                createdAt: startedAt,
                updatedAt: startedAt.addingTimeInterval(Double(steps) * 2.3),
                status: .completed,
                exitCode: 0,
                commandPreview: "mere.run image train-lora --data warm-still-life --output \(stem).safetensors --recipe krea-fast-style",
                outputText: nil,
                templateID: .imageTrainLoRA,
                commandDraft: draft,
                commandArguments: template.arguments(from: draft, source: .contract),
                artifactURLs: [adapter]
            )
            row.inputIdentity = StudioInputIdentity.read(dataset)
            library.upsert(row)
            return row
        }

        let earlier = try seedRun(stem: "warm-still-life-7", steps: 600, loss: { 0.42 * exp(-Double($0) / 260) + 0.11 },
                                  samples: 2, hour: 8, minute: 5)
        let latest = try seedRun(stem: "warm-still-life-42", steps: 1_000, loss: { 0.39 * exp(-Double($0) / 340) + 0.08 + 0.012 * sin(Double($0) / 37) },
                                 samples: 3, hour: 10, minute: 12)

        var taskDraft = StudioTrainingRun.applyingPageDefaults(StudioTaskDraft(templateID: .imageTrainLoRA))
        taskDraft.form["--data"] = .text(dataset.path)
        taskDraft.form["--recipe"] = .text("krea-fast-style")
        // Choosing the recipe on the page clears the seeded options it decides; the seed does too.
        controller.taskSessions.setTaskDraft(StudioTrainingRun.applyingRecipe(taskDraft), for: .imageTrain)
        let scope = StudioTask.imageTrain.rawValue
        controller.taskSessions.set(Optional(latest.id), for: scope + ".requestID")
        controller.taskSessions.set(Optional(latest.id), for: scope + ".Training.compareA")
        controller.taskSessions.set(Optional(earlier.id), for: scope + ".Training.compareB")
    }

    /// A finished Music ▸ Analyze run: the song and the JSON `music analyze` printed for it, kept
    /// as the row's output text the way the Library keeps stdout.
    func seedMusicAnalysisRun() throws {
        guard let template = CommandCatalog.template(id: .musicAnalyze) else {
            throw StudioSnapshotError.noContentView
        }
        let song = root.appendingPathComponent("harbor-lights.wav", isDirectory: false)
        try Self.writeSilentWAV(to: song, seconds: 20)

        var draft = template.defaultDraft()
        draft.inputPath = song.path
        draft.useDuration = true
        draft.durationSeconds = 30
        let startedAt = Self.mockupTime(hour: 11, minute: 48)
        var row = StudioLibraryItem(
            id: UUID(),
            mode: .music,
            prompt: "",
            inputURL: song,
            outputURL: nil,
            createdAt: startedAt,
            updatedAt: startedAt.addingTimeInterval(14),
            status: .completed,
            exitCode: 0,
            commandPreview: "mere.run music analyze harbor-lights.wav --duration 30",
            outputText: Self.musicAnalysisOutput(audio: song),
            templateID: .musicAnalyze,
            commandDraft: draft,
            commandArguments: template.arguments(from: draft, source: .contract)
        )
        row.inputIdentity = StudioInputIdentity.read(song)
        library.upsert(row)
        let scope = StudioTask.musicAnalyze.rawValue
        controller.taskSessions.set(Optional(row.id), for: scope + ".requestID")
        // The page's draft, under its own key: the workspace imports it into the task draft once.
        controller.taskSessions.set(draft, for: scope + ".MusicTools.analyzeDraft")
    }

    /// A finished Music ▸ Transcribe run: the song, the MIDI `music transcribe` wrote for it
    /// (eight bars of chords, a bass line, and a melody on three channels), and the musical
    /// context document beside it, with the instrument list the inspector's picker reads already
    /// cached on the controller.
    func seedTranscribeRun() throws {
        let song = root.appendingPathComponent("harbor-lights.wav", isDirectory: false)
        if !FileManager.default.fileExists(atPath: song.path) {
            try Self.writeSilentWAV(to: song, seconds: 20)
        }
        let midi = root.appendingPathComponent("harbor-lights-7c1e2a.mid", isDirectory: false)
        try Self.writeDemoMIDI(to: midi)
        let context = root.appendingPathComponent("harbor-lights-7c1e2a-context.json", isDirectory: false)
        try Self.musicalContextDocument.write(to: context, atomically: true, encoding: .utf8)

        var taskDraft = StudioTaskDraft(templateID: .musicTranscribe)
        taskDraft.setArgument(0, song.path)
        taskDraft.form["--instruments"] = .text("voice,drums,electric_bass,piano")
        var ran = taskDraft
        ran.form["--output"] = .text(midi.path)
        ran.form["--context-output"] = .text(context.path)
        guard let request = ran.request(source: .contract) else { throw StudioSnapshotError.noContentView }
        let startedAt = Self.mockupTime(hour: 11, minute: 52)
        var row = StudioLibraryItem(
            id: UUID(),
            mode: request.mode,
            prompt: "",
            inputURL: song,
            outputURL: midi,
            createdAt: startedAt,
            updatedAt: startedAt.addingTimeInterval(41),
            status: .completed,
            exitCode: 0,
            commandPreview: "mere.run music transcribe harbor-lights.wav --instruments voice,drums,electric_bass,piano --output harbor-lights-7c1e2a.mid",
            outputText: nil,
            templateID: .musicTranscribe,
            commandDraft: request.draft,
            commandArguments: request.execution?.arguments,
            artifactURLs: [midi, context]
        )
        row.inputIdentity = StudioInputIdentity.read(song)
        library.upsert(row)
        controller.taskSessions.setTaskDraft(taskDraft, for: .musicTranscribe)
        controller.taskSessions.set(Optional(row.id), for: StudioTask.musicTranscribe.rawValue + ".requestID")
        controller.cachedInstrumentNames = [
            "voice", "drums", "electric_bass", "piano", "acoustic_guitar", "electric_guitar", "strings", "brass",
            "soprano_and_alto_sax", "synth_lead", "synth_pad", "organ",
        ]
    }

    /// `MuScriptorMusicalContext` for the harbor-lights demo.
    private static let musicalContextDocument = """
    {
      "tempo" : { "bpm" : 96.02, "confidence" : 0.91 },
      "timeSignature" : { "name" : "4/4", "numerator" : 4, "denominator" : 4, "confidence" : 0.84 },
      "keySignature" : { "name" : "D major", "tonic" : "D", "mode" : "major", "confidence" : 0.77 },
      "beats" : []
    }
    """

    /// A type-0 Standard MIDI File at 480 PPQ and 96 BPM: eight bars of a D–Bm–G–A progression
    /// as held chords on channel 0, a walking bass on channel 1, and an eighth-note melody on
    /// channel 2 — 120 notes across three channels, so the piano roll shows its hues and
    /// velocities.
    static func writeDemoMIDI(to url: URL) throws {
        let ppq = 480
        var events: [(tick: Int, bytes: [UInt8])] = [(0, [0xFF, 0x51, 0x03, 0x09, 0x89, 0x68])]
        func note(_ pitch: Int, at start: Int, for length: Int, velocity: Int, channel: Int) {
            events.append((start, [UInt8(0x90 | channel), UInt8(pitch), UInt8(velocity)]))
            events.append((start + length, [UInt8(0x80 | channel), UInt8(pitch), 0]))
        }
        let chords = [[62, 66, 69], [59, 62, 66], [67, 71, 74], [57, 61, 64]]
        let bass = [50, 47, 43, 45]
        let melody = [74, 76, 78, 81, 78, 76, 74, 73, 71, 69, 71, 73, 74, 78, 76, 74]
        for bar in 0..<8 {
            let barStart = bar * 4 * ppq
            for (index, pitch) in chords[bar % 4].enumerated() {
                note(pitch, at: barStart, for: 4 * ppq - 40, velocity: 64 + index * 6, channel: 0)
            }
            for beat in 0..<4 {
                note(bass[bar % 4] + (beat == 2 ? 7 : 0), at: barStart + beat * ppq, for: ppq - 60, velocity: 88, channel: 1)
            }
            for eighth in 0..<8 {
                let pitch = melody[(bar * 8 + eighth) % melody.count]
                note(pitch, at: barStart + eighth * ppq / 2, for: ppq / 2 - 30, velocity: 70 + (eighth % 3) * 12, channel: 2)
            }
        }
        // Note-offs before note-ons at the same tick, so a repeated pitch closes before it reopens.
        events.sort { $0.tick == $1.tick ? $0.bytes[0] < $1.bytes[0] : $0.tick < $1.tick }

        var track = Data()
        var last = 0
        for event in events {
            track.append(contentsOf: variableLength(event.tick - last))
            track.append(contentsOf: event.bytes)
            last = event.tick
        }
        track.append(contentsOf: [0x00, 0xFF, 0x2F, 0x00])

        var data = Data()
        func appendBE32(_ value: UInt32) { withUnsafeBytes(of: value.bigEndian) { data.append(contentsOf: $0) } }
        func appendBE16(_ value: UInt16) { withUnsafeBytes(of: value.bigEndian) { data.append(contentsOf: $0) } }
        data.append(contentsOf: Array("MThd".utf8))
        appendBE32(6)
        appendBE16(0)
        appendBE16(1)
        appendBE16(UInt16(ppq))
        data.append(contentsOf: Array("MTrk".utf8))
        appendBE32(UInt32(track.count))
        data.append(track)
        try data.write(to: url, options: .atomic)
    }

    /// A MIDI variable-length quantity: seven bits per byte, high bit set on all but the last.
    private static func variableLength(_ value: Int) -> [UInt8] {
        var bytes = [UInt8(value & 0x7F)]
        var remaining = value >> 7
        while remaining > 0 {
            bytes.insert(UInt8(remaining & 0x7F) | 0x80, at: 0)
            remaining >>= 7
        }
        return bytes
    }

    /// `SpeechDiarizationPayload` for a three-minute stand-up: three voices, sixteen turns.
    private static func diarizationDocument(source: URL) -> String {
        let turns: [(speaker: Int, start: Double, end: Double)] = [
            (0, 0.4, 9.8), (1, 10.3, 24.1), (0, 24.6, 27.9), (1, 28.2, 41.0), (2, 41.7, 58.3),
            (0, 58.9, 63.2), (2, 63.4, 79.8), (1, 80.5, 96.2), (0, 96.4, 99.1), (1, 99.3, 112.7),
            (2, 113.5, 130.0), (0, 130.6, 148.9), (1, 149.2, 152.4), (0, 152.6, 171.3),
            (2, 171.9, 183.0), (0, 183.4, 190.8),
        ]
        let segments = turns.map { turn in
            String(
                format: "    { \"speaker\" : \"speaker_%d\", \"speaker_index\" : %d, \"start_seconds\" : %.1f, \"end_seconds\" : %.1f, \"duration_seconds\" : %.1f }",
                turn.speaker, turn.speaker, turn.start, turn.end, turn.end - turn.start
            )
        }.joined(separator: ",\n")
        return """
        {
          "schema_version" : 1,
          "model" : "speech-diarization-sortformer",
          "source" : "\(source.path)",
          "runtime" : "mlx",
          "device" : "gpu",
          "duration_seconds" : 192.4,
          "speaker_count" : 3,
          "processing_seconds" : 6.1,
          "segments" : [
        \(segments)
          ]
        }
        """
    }

    /// The `separation.json` `music separate` writes for two stems, as `MusicSeparationManifest`
    /// encodes it (snake_case, sorted keys).
    private static func separationManifest(source: URL, stems: [URL], manifest: URL) -> String {
        let stemEntries = stems.map { url in
            """
              {
                "name" : "\(url.deletingPathExtension().lastPathComponent)",
                "path" : "\(url.path)",
                "sha256" : "0000000000000000000000000000000000000000000000000000000000000000"
              }
            """
        }.joined(separator: ",\n")
        return """
        {
          "chunk_size" : 352800,
          "chunks" : 3,
          "created_at" : "2026-09-24T16:12:21Z",
          "elapsed_seconds" : 21.5,
          "manifest_path" : "\(manifest.path)",
          "model" : {
            "compute_type" : "float16",
            "id" : "music-separate-bs-roformer-viperx-1297",
            "license" : "MIT",
            "repository" : "mere-run/bs-roformer",
            "revision" : "main",
            "weights_sha256" : "0000000000000000000000000000000000000000000000000000000000000000"
          },
          "overlap" : 2,
          "schema_version" : 1,
          "source" : {
            "channels" : 1,
            "frames" : 192000,
            "path" : "\(source.path)",
            "sample_rate" : 16000,
            "sha256" : "0000000000000000000000000000000000000000000000000000000000000000"
          },
          "stems" : [
        \(stemEntries)
          ]
        }

        """
    }

    /// `MusicAnalyzeOutput` for the harbor-lights demo, with the model's reply kept.
    private static func musicAnalysisOutput(audio: URL) -> String {
        """
        {
          "analyzedDurationSeconds" : 30,
          "audio" : "\(audio.path)",
          "checkpointsRoot" : "/Users/example/Library/Application Support/MereRun/models/music-acestep",
          "inputDurationSeconds" : 214.6,
          "languageModelRoot" : "/Users/example/Library/Application Support/MereRun/models/music-acestep/lm",
          "languageModelSource" : "bundled",
          "lmSubdirectory" : "acestep-5Hz-lm-1.7B",
          "metadata" : {
            "bpm" : 96,
            "caption" : "Warm, unhurried indie folk: fingerpicked nylon guitar over a soft brushed kit, an upright bass walking underneath, and a close, breathy lead vocal with light harmonies on the chorus. Wide, roomy reverb; a late-evening, harbourside mood.",
            "durationSeconds" : 214.6,
            "keyscale" : "D major",
            "language" : "en",
            "lyrics" : "[verse]\\nHarbour lights are blinking slow\\nOn the water where the old boats go\\nI left my coat on the ferry rail\\nAnd watched the evening turn to pale\\n\\n[chorus]\\nStay a while, the tide is low\\nThere's nowhere else we have to go",
            "timesignature" : "4/4"
          },
          "model" : "music-acestep",
          "rawLMOutput" : "<bpm>96</bpm><keyscale>D major</keyscale><timesignature>4/4</timesignature><language>en</language><caption>Warm, unhurried indie folk…</caption>",
          "turboSubdirectory" : "acestep-v15-turbo"
        }
        """
    }

    /// A valid 16 kHz mono 16-bit PCM WAV of near-silence with a quiet tone so a waveform draws.
    static func writeSilentWAV(to url: URL, seconds: Int) throws {
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
        let termination: @Sendable (Int32) -> Void
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
            _liveStarts.append(LiveStart(
                configuration: configuration,
                stdout: stdout,
                stderr: stderr,
                termination: termination,
                process: process
            ))
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
            .init(matches: { $0 == ["model", "list", "--json"] }, stdout: ModelsInventoryScript.inventoryJSON(from: ModelsInventoryScript.modelList), exitCode: 0),
            .init(
                matches: { $0 == ["model", "capabilities", "--all", "--json"] },
                stdout: ModelsInventoryScript.capabilities,
                exitCode: 0
            ),
        ]
    }
}

/// What the Runs render's CLI reads answer: no Relay executors, one durable graph run under
/// `~/runs`, and that run's manifest as `run inspect --json` prints it — failed at its render
/// node on the second attempt, with the poster it managed to write.
private enum RunsScript {
    static let runPath = "/Users/example/runs/poster-2026-09-03"

    static let list = """
    {
      "summary" : "1 durable run under /Users/example/runs",
      "result" : {
        "root" : "/Users/example/runs",
        "scanned_directory_count" : 3,
        "entries" : [
          {
            "id" : "poster-2026-09-03",
            "kind" : "graph_run",
            "path" : "\(runPath)",
            "relative_path" : "poster-2026-09-03",
            "status" : "failed",
            "state" : "failed",
            "summary" : "Graph poster failed at render",
            "created_at" : "2026-09-03T12:00:00Z",
            "updated_at" : "2026-09-03T12:00:41Z",
            "event_count" : 12,
            "artifact_count" : 1,
            "diagnostic_count" : 1,
            "blocker_count" : 1
          }
        ]
      }
    }

    """

    static let inspection = """
    {
      "attempt" : 1,
      "contract_version" : "mere.run/graph-run.v1",
      "created_at" : "2026-09-03T12:00:00Z",
      "error" : "render: image generate exited with status 1 (model image-zimage-nano is not installed)",
      "executor" : { "kind" : "local", "profile" : null, "job_reference" : null },
      "graph_fingerprint" : "3f9c21",
      "graph_name" : "poster",
      "job_id" : "poster-2026-09-03",
      "nodes" : [
        { "artifacts" : [], "attempt" : 1, "completed_at" : "2026-09-03T12:00:01Z", "fingerprint" : "n0", "id" : "fetch-references", "kind" : "files.copy", "max_attempts" : 1, "models" : [], "outputs" : [], "started_at" : "2026-09-03T12:00:00Z", "state" : "finished" },
        { "artifacts" : [], "attempt" : 1, "completed_at" : "2026-09-03T12:00:20Z", "fingerprint" : "n1", "id" : "write-tagline", "kind" : "text.chat", "max_attempts" : 1, "models" : [], "outputs" : [], "started_at" : "2026-09-03T12:00:02Z", "state" : "finished" },
        { "artifacts" : [ { "content_type" : "image/png", "kind" : "image", "name" : "poster-draft", "path" : "\(runPath)/poster-draft.png", "sha256" : "cd", "size_bytes" : 1843200 } ], "attempt" : 2, "error" : "image generate exited with status 1", "fingerprint" : "n2", "id" : "render", "kind" : "image.generate", "max_attempts" : 3, "models" : [], "outputs" : [], "started_at" : "2026-09-03T12:00:20Z", "state" : "failed" },
        { "artifacts" : [], "attempt" : 0, "fingerprint" : "n3", "id" : "publish", "kind" : "files.export", "max_attempts" : 1, "models" : [], "outputs" : [], "state" : "planned" }
      ],
      "outputs" : [
        { "content_type" : "image/png", "kind" : "image", "name" : "poster-draft", "path" : "\(runPath)/poster-draft.png", "sha256" : "cd", "size_bytes" : 1843200 }
      ],
      "state" : "failed",
      "updated_at" : "2026-09-03T12:00:41Z"
    }

    """

    static var responses: [SnapshotProcessRunner.Response] {
        [
            .init(matches: { $0 == ["executor", "list", "--json"] }, stdout: "{\"profiles\": []}\n", exitCode: 0),
            .init(matches: { $0.starts(with: ["run", "list"]) }, stdout: list, exitCode: 0),
            .init(matches: { $0.starts(with: ["run", "inspect"]) }, stdout: inspection, exitCode: 0),
        ]
    }
}

/// The model inventory the Models ▸ Installed fidelity render shows: the mockup's sample
/// lineup, expressed the way `mere.run model list`, `model capabilities`, `model storage`,
/// `model info`, `model runtime get`, and `adapter list` print it.
/// What the Audio and Voice boards' CLI reads answer: the inventory with the diarization,
/// enhancement, and separation models installed, so the composers are live, and the two
/// microphones `speech listen --list-devices` lists for Audio ▸ Live's device chip.
private enum AudioVoiceScript {
    static let models = [
        (id: "speech-diarization-sortformer", category: "speech-diarization", title: "Sortformer"),
        (id: "speech-diarization-nemotron3", category: "speech-diarization", title: "Nemotron 3 Diarization"),
        (id: "audio-enhance-ap-bwe-16kto48k", category: "audio", title: "AP-BWE 16k to 48k"),
        (id: "music-separate-bs-roformer-viperx-1297", category: "music", title: "BS-RoFormer ViperX"),
    ]

    static var responses: [SnapshotProcessRunner.Response] {
        ModelsInventoryScript.analyzeReadinessResponses(alsoInstalling: models) + [
            .init(
                matches: { $0.starts(with: ["speech", "listen"]) && $0.contains("--list-devices") },
                stdout: "* BuiltInMicrophoneDevice\tMacBook Pro Microphone\n  AppleUSBAudioEngine:0001\tStudio USB Mic\n",
                exitCode: 0
            ),
        ]
    }
}

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

    static func inventoryJSON(from text: String) -> String {
        struct Row: Encodable { let id: String; let category: String; let status: String; let size: String }
        struct Inventory: Encodable { let rows: [Row] }
        struct Document: Encodable { let inventory: Inventory; let usageTerms: [String] }
        let rows = StudioModelInventoryParser.rows(from: text).map {
            Row(id: $0.id, category: $0.category, status: $0.status, size: $0.size)
        }
        do {
            return String(decoding: try JSONEncoder().encode(Document(inventory: Inventory(rows: rows), usageTerms: [])), as: UTF8.self)
        } catch { preconditionFailure("Could not encode snapshot inventory: \(error)") }
    }

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
            .init(matches: { $0 == ["model", "list", "--json"] }, stdout: ModelsInventoryScript.inventoryJSON(from: modelList), exitCode: 0),
            .init(matches: { $0 == ["model", "capabilities", "--all", "--json"] }, stdout: capabilities, exitCode: 0),
            version,
        ]
    }

    /// `model list` and `model capabilities` with the Analyze tasks' models installed (Vision ▸
    /// Find's, the Vision specialists', Transcribe's), so the Analyze boards render their results
    /// rather than a readiness card.
    static var analyzeReadinessResponses: [SnapshotProcessRunner.Response] {
        analyzeReadinessResponses(alsoInstalling: [])
    }

    /// The same answers with more models installed, for the boards of other Analyze tasks.
    static func analyzeReadinessResponses(
        alsoInstalling more: [(id: String, category: String, title: String)]
    ) -> [SnapshotProcessRunner.Response] {
        readinessResponses(installing: [
            (id: "vision-ground-falcon-perception", category: "vision-ground", title: "Falcon Perception"),
            (id: "speech-asr-parakeet", category: "speech-asr", title: "Parakeet"),
            (id: "vision-face-buffalo-l", category: "vision-face", title: "Face Buffalo L"),
            (id: "vision-depth-marigold-v2", category: "vision-depth", title: "Depth Marigold V2"),
            (id: "vision-geometry-moge2-small", category: "vision-geometry", title: "Geometry MoGe2 Small"),
            (id: "vision-geometry-da3-small", category: "vision-geometry", title: "Geometry DA3 Small"),
            (id: "vision-segment-sam31", category: "vision-segment", title: "SAM 3.1")
        ] + more)
    }

    /// The same with the Sound tasks' Woosh models installed, so the Sound boards render their
    /// results and composers rather than readiness cards.
    static var soundReadinessResponses: [SnapshotProcessRunner.Response] {
        readinessResponses(installing: [
            (id: "sfx-woosh-dvflow-8s", category: "sfx", title: "Woosh DVFlow 8s"),
            (id: "sfx-woosh-dflow", category: "sfx", title: "Woosh DFlow"),
            (id: "sfx-woosh-clap", category: "sfx", title: "Woosh CLAP")
        ])
    }

    /// The same with the Music tasks' default models installed, so their boards render results.
    static var musicReadinessResponses: [SnapshotProcessRunner.Response] {
        readinessResponses(installing: [
            (id: "music-acestep", category: "music", title: "ACE-Step 1.5"),
            (id: "music-muscriptor-medium", category: "music", title: "MuScriptor medium")
        ])
    }

    /// `model list` and `model capabilities` with `extraModels` installed beside the fixture's.
    static func readinessResponses(
        installing extraModels: [(id: String, category: String, title: String)]
    ) -> [SnapshotProcessRunner.Response] {
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
            .init(matches: { $0 == ["model", "list", "--json"] }, stdout: ModelsInventoryScript.inventoryJSON(from: list), exitCode: 0),
            .init(matches: { $0 == ["model", "capabilities", "--all", "--json"] }, stdout: capabilityJSON, exitCode: 0),
        ]
    }

    /// `model list` and `model capabilities` with the three trainers' default models installed,
    /// so the Train pages render with Start available rather than a readiness message.
    static var trainingReadinessResponses: [SnapshotProcessRunner.Response] {
        let extraModels = [
            (id: "image-krea2-raw", category: "image", title: "Krea 2 raw"),
            (id: "text-chat-gemma4-12b-4bit", category: "text-chat", title: "Gemma 4 12B"),
            (id: "music-acestep", category: "music", title: "ACE-Step"),
        ]
        let list = modelList.replacingOccurrences(
            of: "image-zimage-nano          image        installed  2.1 GB",
            with: (["image-zimage-nano          image        installed  2.1 GB"]
                + extraModels.map { "\($0.id)  \($0.category)  installed  6.2 GB" })
                .joined(separator: "\n")
        )
        let extraCapabilities = extraModels.map { model in
            """
            ,{"id": "\(model.id)", "title": "\(model.title)", "summary": "", \
            "minimumUnifiedMemoryGB": 16, "recommendedUnifiedMemoryGB": 32, "supported": true, \
            "reasons": [], "estimatedDownloadBytes": 6200000000, \
            "sourceRepository": "mere-run/\(model.id)", "publisher": "mere.run"}
            """
        }.joined(separator: "\n")
        let capabilityJSON = capabilities.replacingOccurrences(of: "\n]}", with: "\n\(extraCapabilities)\n]}")
        return [
            .init(matches: { $0 == ["model", "list"] }, stdout: list, exitCode: 0),
            .init(matches: { $0 == ["model", "list", "--json"] }, stdout: ModelsInventoryScript.inventoryJSON(from: list), exitCode: 0),
            .init(matches: { $0 == ["model", "capabilities", "--all", "--json"] }, stdout: capabilityJSON, exitCode: 0),
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
            .init(matches: { $0 == ["model", "list", "--json"] }, stdout: ModelsInventoryScript.inventoryJSON(from: modelList), exitCode: 0),
            .init(matches: { $0 == ["model", "capabilities", "--all", "--json"] }, stdout: capabilities, exitCode: 0),
            .init(matches: { $0 == ["model", "storage", "--json"] }, stdout: storage, exitCode: 0),
            .init(matches: { $0.starts(with: ["model", "info", defaultModelID]) }, stdout: modelInfo, exitCode: 0),
            .init(matches: { $0.starts(with: ["model", "runtime", "get"]) }, stdout: "{\"pinned\": false}\n", exitCode: 0),
            .init(matches: { $0 == ["adapter", "list", "--json"] }, stdout: adapters, exitCode: 0),
        ]
    }
}

/// Answers the runtime's `/runtime/status` for the menu bar renders, in place of the network: a
/// `URLProtocol` registered for `URLSession.shared`, which is what `StudioServingMonitor` polls.
private final class SnapshotRuntimeEndpoint: URLProtocol {
    enum Answer {
        case unreachable
        case runtime(String)
    }

    nonisolated(unsafe) static var answer = Answer.unreachable

    static func install() { URLProtocol.registerClass(SnapshotRuntimeEndpoint.self) }
    static func uninstall() { URLProtocol.unregisterClass(SnapshotRuntimeEndpoint.self) }

    override class func canInit(with request: URLRequest) -> Bool {
        request.url?.path.hasPrefix("/runtime/") == true
    }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        switch Self.answer {
        case .unreachable:
            client?.urlProtocol(self, didFailWithError: URLError(.cannotConnectToHost))
        case .runtime(let json):
            let response = HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: "HTTP/1.1", headerFields: nil)!
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: Data(json.utf8))
            client?.urlProtocolDidFinishLoading(self)
        }
    }

    override func stopLoading() {}

    /// Two resident text models (one serving two requests, one queued behind them), a resident
    /// speech sidecar, about 12 GB in use, up a little over two hours.
    static let busyRuntime = """
    {
      "object": "runtime.model_pool",
      "defaultModel": "text-chat-gemma4",
      "activeRequests": 2,
      "admission": {
        "maxActiveRequests": 4, "activeRequests": 2, "queuedRequests": 1,
        "totalAdmittedRequests": 41, "totalCompletedRequests": 38, "totalCancelledRequests": 0
      },
      "memory": { "currentBytes": 13314398618, "pressure": "nominal" },
      "process": { "processID": 48213, "uptimeSeconds": 8100 },
      "models": [
        { "id": "text-chat-gemma4", "loaded": true, "ready": true, "activeRequests": 2, "pinned": false },
        { "id": "text-code-north-mini", "loaded": true, "ready": true, "activeRequests": 0, "pinned": false },
        { "id": "text-chat-laguna", "loaded": false, "activeRequests": 0, "pinned": false }
      ],
      "sidecars": {
        "defaultIdleTTLSeconds": 300, "pressure": "nominal", "loadedCount": 1,
        "activeRequests": 0, "queuedRequests": 0,
        "residents": [{
          "kind": "speech", "modelID": "speech-kokoro", "loaded": true, "ready": true,
          "activeRequests": 0, "queuedRequests": 0, "pinned": false, "ttlSeconds": 300,
          "loadCount": 1, "replacementCount": 0, "evictionCount": 0,
          "completedRequests": 3, "failedRequests": 0
        }]
      }
    }
    """
}

// MARK: - Region editor preview

/// The region-prompt editor's building blocks with a chosen prompt already selected, so the
/// render shows the handles a click reveals; `StudioRegionPromptEditor` itself starts with no
/// selection.
private struct RegionEditorPreview: View {
    let image: NSImage
    @State var prompts: [StudioRegionPrompt]
    @State var selection: UUID?
    @State private var tool = StudioRegionTool.box

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            StudioRegionToolbar(tool: $tool, prompts: $prompts, selection: $selection)
            Image(nsImage: image)
                .resizable()
                .aspectRatio(contentMode: .fit)
                .overlay {
                    GeometryReader { geometry in
                        StudioRegionPromptLayer(
                            prompts: $prompts,
                            imageSize: CGSize(width: 1_024, height: 1_024),
                            fitted: CGRect(origin: .zero, size: geometry.size),
                            tool: $tool,
                            selection: $selection
                        )
                    }
                }
                .mereMediaFrame()
        }
    }
}
