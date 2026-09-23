@testable import StudioKit
import StudioTestSupport
import XCTest

@MainActor
final class StudioLocalServerTests: XCTestCase {
    // MARK: - Phase

    func testPhaseCombinesTheServerProcessWithTheEndpointsAnswer() {
        let exited = Date(timeIntervalSince1970: 1_000)
        let before = exited.addingTimeInterval(-1)
        let after = exited.addingTimeInterval(1)
        let phase = StudioLocalServer.phase(process:answeredAt:)

        XCTAssertEqual(phase(.none(exitedAt: nil), nil), .stopped)
        XCTAssertEqual(phase(.none(exitedAt: nil), after), .external)
        XCTAssertEqual(phase(.running(stopRequested: false, since: exited), nil), .starting)
        XCTAssertEqual(phase(.running(stopRequested: false, since: exited), after), .running)
        // An answer to a poll sent before this server launched was the previous server's.
        XCTAssertEqual(phase(.running(stopRequested: false, since: exited), before), .starting)
        XCTAssertEqual(phase(.running(stopRequested: true, since: exited), after), .stopping)
        // An answer from before Studio's server exited was that server's, not someone else's.
        XCTAssertEqual(phase(.none(exitedAt: exited), before), .stopped)
        XCTAssertEqual(phase(.none(exitedAt: exited), after), .external)
        XCTAssertEqual(phase(.failed("Address already in use", exitedAt: exited), nil), .failed("Address already in use"))
        // A server that failed because another already holds the port hands over to that one.
        XCTAssertEqual(phase(.failed("Address already in use", exitedAt: exited), after), .external)
    }

    func testOnlyAServerStudioStartedCanBeStopped() {
        XCTAssertTrue(StudioLocalServer.Phase.starting.isOwned)
        XCTAssertTrue(StudioLocalServer.Phase.running.isOwned)
        XCTAssertTrue(StudioLocalServer.Phase.stopping.isOwned)
        XCTAssertFalse(StudioLocalServer.Phase.external.isOwned)
        XCTAssertFalse(StudioLocalServer.Phase.failed("x").isOwned)
        XCTAssertTrue(StudioLocalServer.Phase.external.isServing)
        XCTAssertFalse(StudioLocalServer.Phase.starting.isServing)
    }

    // MARK: - Lifecycle

    func testStartRunsTheServerInTheServiceLaneWithoutTakingARunSlotOrTheConsole() throws {
        let (controller, runner, restore) = makeController()
        defer { restore() }
        controller.runtimeAPIKey = "local-secret"

        XCTAssertNil(controller.localServer.start())

        XCTAssertEqual(runner.starts.count, 1)
        let configuration = runner.starts[0].configuration
        let arguments = configuration.arguments
        XCTAssertTrue(arguments.contains("serve"))
        XCTAssertEqual(arguments.firstIndex(of: "--port").map { arguments[$0 + 1] }, "9")
        // The key travels in the environment, never in argv.
        XCTAssertFalse(arguments.contains("local-secret"))
        XCTAssertEqual(configuration.environment[CommandLaunchEnvironment.apiKeyEnvironmentKey], "local-secret")

        XCTAssertEqual(controller.jobs.running(in: .service).count, 1)
        XCTAssertTrue(controller.jobs.running(in: .inference).isEmpty)
        XCTAssertTrue(controller.jobs.hasCapacity(in: .inference))
        // A server is not a run: it takes neither the console nor the composer's Stop.
        XCTAssertFalse(controller.isRunning)
        XCTAssertNil(controller.activeRunRequestID)
        XCTAssertEqual(controller.localServer.phase, .starting)
    }

    func testStopTerminatesStudiosServerAndSettlesStopped() async {
        let (controller, runner, restore) = makeController()
        defer { restore() }
        let server = controller.localServer
        server.start()

        XCTAssertTrue(server.stop())
        XCTAssertEqual(runner.processes[0].terminateCallCount, 1)
        XCTAssertEqual(server.phase, .stopping)

        runner.starts[0].termination(15)
        await settle()
        XCTAssertEqual(server.phase, .stopped)
        XCTAssertFalse(server.stop(), "nothing left to stop")
    }

    func testAServerThatExitsOnItsOwnReportsItsLastError() async {
        let (controller, runner, restore) = makeController()
        defer { restore() }
        let server = controller.localServer
        server.start()

        runner.starts[0].stderr("Loading runtime settings\nError: port 9 is already in use\n")
        runner.starts[0].termination(1)
        await settle()

        XCTAssertEqual(server.phase, .failed("Error: port 9 is already in use"))
    }

    func testRestartWaitsForTheOldServerToExitBeforeLaunchingTheNext() async {
        let (controller, runner, restore) = makeController()
        defer { restore() }
        let server = controller.localServer
        server.start()

        let restart = Task { await server.restart() }
        await settle()
        XCTAssertEqual(runner.processes[0].terminateCallCount, 1)
        XCTAssertEqual(runner.starts.count, 1, "the port is still held until the old process exits")

        runner.starts[0].termination(15)
        let refusal = await restart.value
        XCTAssertNil(refusal)
        XCTAssertEqual(runner.starts.count, 2)
        XCTAssertEqual(server.phase, .starting)
    }

    func testStartRefusesToServeBeyondThisMacWithoutAKey() {
        let (controller, runner, restore) = makeController()
        defer { restore() }
        controller.runtimeHost = "0.0.0.0"
        controller.runtimeAPIKey = ""

        XCTAssertNotNil(controller.localServer.start())
        XCTAssertTrue(runner.starts.isEmpty)
        XCTAssertEqual(controller.localServer.phase, .stopped)
    }

    func testAdoptsAServerStartedFromTheCommandConsoleSoStopReachesIt() throws {
        let (controller, runner, restore) = makeController()
        defer { restore() }
        let template = try XCTUnwrap(CommandCatalog.template(id: .apiServe))
        let server = controller.localServer
        var draft = template.defaultDraft()
        draft.port = 9

        controller.runConsole(template: template, draft: draft, arguments: template.arguments(from: draft), requestID: nil)

        XCTAssertEqual(server.phase, .starting)
        XCTAssertTrue(server.stop())
        XCTAssertEqual(runner.processes[0].terminateCallCount, 1)
    }

    func testOptionsCarryOverFromTheServerPageAndPersistUnderTheirOwnKey() throws {
        let sessions = StudioTaskSessions()
        var legacy = try XCTUnwrap(CommandCatalog.template(id: .apiServe)).defaultDraft()
        legacy.model = "text-chat-gemma4-e4b"
        sessions.set(legacy, for: StudioLocalServer.legacyOptionsKey)
        let controller = MereRunController(
            secretStore: InMemorySecretStore(),
            processRunner: RecordingProcessRunner(),
            resolvesCLIOnInit: false,
            taskSessions: sessions
        )

        XCTAssertEqual(controller.localServer.options.model, "text-chat-gemma4-e4b")
        controller.localServer.options.model = "text-chat-q36"
        XCTAssertEqual(
            sessions.value(for: StudioLocalServer.optionsKey, default: CommandDraft()).model,
            "text-chat-q36"
        )
    }

    func testCommandViewEditsCannotMoveTheServerOffTheEndpointStudioWatches() throws {
        let (controller, runner, restore) = makeController()
        defer { restore() }
        let template = try XCTUnwrap(CommandCatalog.template(id: .apiServe))
        let source = controller.localServer.launchDraft
        var edited = source
        edited.port = 9_999
        edited.contextSize = 4_096
        controller.taskSessions.set(
            Optional(StudioTaskCommandState(
                templateID: .apiServe,
                sourceArguments: template.arguments(from: source),
                form: StudioConsoleCommand.seed(template: template, draft: edited)
            )),
            for: StudioTask.serverServing.rawValue + ".commandOverride"
        )

        controller.localServer.start()

        let arguments = runner.starts[0].configuration.arguments
        XCTAssertEqual(arguments.firstIndex(of: "--port").map { arguments[$0 + 1] }, "9", "the endpoint stays pinned")
        XCTAssertEqual(arguments.firstIndex(of: "--context-size").map { arguments[$0 + 1] }, "4096", "other edits apply")
    }

    func testABlankHostServesOnLoopback() {
        let (controller, runner, restore) = makeController()
        defer { restore() }
        controller.runtimeHost = "  "

        XCTAssertEqual(controller.localServer.safety, .loopback)
        XCTAssertNil(controller.localServer.start())
        let arguments = runner.starts[0].configuration.arguments
        XCTAssertEqual(arguments.firstIndex(of: "--host").map { arguments[$0 + 1] }, "127.0.0.1")
    }

    func testAConsolePreflightIsNotAdoptedAsTheServer() throws {
        let (controller, _, restore) = makeController()
        defer { restore() }
        let template = try XCTUnwrap(CommandCatalog.template(id: .apiServe))
        var preflight = template.defaultDraft()
        preflight.port = 9
        preflight.preflight = true

        controller.runConsole(template: template, draft: preflight, arguments: template.arguments(from: preflight), requestID: nil)

        XCTAssertEqual(controller.localServer.phase, .stopped)
    }

    // MARK: - Resident servers

    func testResidentServersRunInTheServiceLaneAndStopOnlyThemselves() async throws {
        let (controller, runner, restore) = makeController()
        defer { restore() }
        // A generation holds the console, as one does whenever the music page is open mid-run.
        let custom = try XCTUnwrap(CommandCatalog.template(id: .custom))
        var generation = custom.defaultDraft()
        generation.extraArguments = "image generate"
        controller.jobs.submit(JobRequest(
            lane: .inference,
            template: custom,
            draft: generation,
            requestID: UUID(),
            configuration: MereRunProcessConfiguration(
                executableURL: URL(fileURLWithPath: "/usr/bin/true"),
                arguments: ["image", "generate"],
                currentDirectoryURL: FileManager.default.temporaryDirectory,
                environment: [:],
                keepsStandardInputOpen: false
            ),
            displayCommand: "mere.run image generate"
        ))
        XCTAssertTrue(controller.isRunning)

        let music = controller.musicServer
        music.start(draft: try XCTUnwrap(CommandCatalog.template(id: .musicServe)).defaultDraft())
        XCTAssertTrue(music.state.isRunning)
        XCTAssertFalse(music.state.isStopping)
        XCTAssertEqual(controller.jobs.running(in: .service).count, 1)
        XCTAssertEqual(controller.jobs.running(in: .inference).count, 1, "the server took no generation slot")

        XCTAssertTrue(music.stop())
        XCTAssertEqual(runner.processes[1].terminateCallCount, 1)
        XCTAssertEqual(runner.processes[0].terminateCallCount, 0, "the generation keeps running")
        runner.starts[1].termination(15)
        await settle()
        XCTAssertEqual(music.state, .none(exitedAt: music.job?.result?.completedAt))
        XCTAssertTrue(controller.isRunning)
    }

    func testResidentServerThatFailsToStartSaysWhy() async throws {
        let (controller, runner, restore) = makeController()
        defer { restore() }
        let vision = controller.visionServer
        vision.start(draft: try XCTUnwrap(CommandCatalog.template(id: .visionServe)).defaultDraft())

        runner.starts[0].stderr("Error: no vision grounding model is installed\n")
        runner.starts[0].termination(1)
        await settle()

        guard case .failed(let message, _) = vision.state else {
            return XCTFail("expected a failure, got \(vision.state)")
        }
        XCTAssertEqual(message, "Error: no vision grounding model is installed")
        XCTAssertEqual(StudioQuitWarning.message(for: controller), nil, "a stopped server holds nothing back")
    }

    // MARK: - Quit warning

    func testQuitWarnsOnlyWhenItWouldStopAServerOrTheUsersWork() {
        let message = StudioQuitWarning.message(servers:runningJobs:queuedJobs:)
        XCTAssertNil(message([], 0, 0))
        XCTAssertEqual(message(["API server"], 0, 0), "Quitting stops the API server Studio started.")
        XCTAssertEqual(
            message(["API server"], 2, 0),
            "Quitting stops the API server Studio started and 2 running jobs."
        )
        XCTAssertEqual(
            message(["API server", "vision server"], 1, 0),
            "Quitting stops the API server and vision server Studio started, and 1 running job."
        )
        XCTAssertEqual(
            message(["API server", "vision server", "music server"], 0, 0),
            "Quitting stops the API server, vision server, and music server Studio started."
        )
        XCTAssertEqual(message([], 1, 3), "Quitting stops 1 running job. 3 queued jobs will not start.")
        XCTAssertEqual(message([], 0, 1), "Quitting drops 1 queued job.")
    }

    func testQuitWarningCountsTheServerButNotUtilityReads() async throws {
        let (controller, _, restore) = makeController()
        defer { restore() }
        XCTAssertNil(StudioQuitWarning.message(for: controller))

        controller.localServer.start()
        _ = Task { await controller.utilityCommandResult(args: ["model", "list"]) }
        await settle()

        XCTAssertEqual(StudioQuitWarning.message(for: controller), "Quitting stops the API server Studio started.")
    }
}

private extension StudioLocalServerTests {
    /// A controller whose server would listen on the discard port, so the endpoint poll a stopped
    /// server triggers never reaches a server running on this machine. Runtime settings persist
    /// through UserDefaults; the returned closure puts them back.
    func makeController() -> (MereRunController, RecordingProcessRunner, @MainActor () -> Void) {
        let runner = RecordingProcessRunner()
        let controller = MereRunController(
            secretStore: InMemorySecretStore(),
            processRunner: runner,
            resolvesCLIOnInit: false,
            taskSessions: StudioTaskSessions()
        )
        controller.cliPath = "/usr/bin/true"
        let host = controller.runtimeHost
        let port = controller.runtimePort
        controller.runtimeHost = "127.0.0.1"
        controller.runtimePort = 9
        return (controller, runner, {
            controller.runtimeHost = host
            controller.runtimePort = port
        })
    }

    func settle() async {
        for _ in 0..<10 { await Task.yield() }
    }
}
