import Foundation
import MereRunContract
@testable import StudioKit
import StudioTestSupport
import XCTest

/// `catalog resolve` answers: whose they are, when Studio asks, and how long an answer stands.
@MainActor
final class StudioModelIdentityStoreTests: XCTestCase {
    private let video = MereRunCapabilityCatalog.videoGenerate

    /// M-e: each controller asks its own CLI. An answer one controller's resolver gives never
    /// scopes another controller's surfaces, and nothing reads a process-wide store.
    func testEachControllerKeepsItsOwnAnswers() async throws {
        let first = MereRunController(secretStore: InMemorySecretStore(), processRunner: RecordingProcessRunner(), resolvesCLIOnInit: false)
        let second = MereRunController(secretStore: InMemorySecretStore(), processRunner: RecordingProcessRunner(), resolvesCLIOnInit: false)
        XCTAssertFalse(first.modelIdentities === second.modelIdentities)
        let folder = "/tmp/model-scope/each-controller"
        first.modelIdentities.use { _ in
            MereRunFamilyResolutionReport(
                capability: "video.generate", family: "h3-ref2va", familyTitle: "MiniMax-H3 Ref2VA",
                model: folder, source: .identified, violations: [], warnings: []
            )
        }
        second.modelIdentities.use { _ in nil }
        let commandLine = ["video", "generate", "a lighthouse", "--model", folder]
        for _ in 0..<200 where first.scopeSource.scope(capability: video, commandLine: commandLine).family == nil {
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTAssertEqual(first.scopeSource.scope(capability: video, commandLine: commandLine).family?.id, "h3-ref2va")
        XCTAssertNil(second.scopeSource.scope(capability: video, commandLine: commandLine).family)
    }

    // MARK: A question in flight

    /// H-c: Transcribe on Parakeet, then `--task translate` added. While the CLI is asked about
    /// the new command line, the last answer for the same model (Parakeet) must not stand in:
    /// the contract itself already routes a translation to Qwen3-ASR, and a run launched on
    /// Parakeet's scope would drop `--task translate`. The launch keeps it, answer or not.
    func testAPendingAnswerNeverReplacesTheFamilyTheContractNames() async throws {
        let gate = Gate()
        let store = StudioModelIdentityStore(debounce: .zero)
        store.use { commandLine in
            if commandLine.contains("translate") { await gate.wait() }
            let translate = commandLine.contains("translate")
            return MereRunFamilyResolutionReport(
                capability: "speech.transcribe", family: translate ? "qwen3-asr" : "parakeet",
                familyTitle: translate ? "Qwen3-ASR" : "Parakeet",
                model: translate ? "speech-asr-qwen3" : "speech-asr-parakeet", source: translate ? .defaultModel : .model,
                violations: [], warnings: []
            )
        }
        let source = StudioScopeSource(identities: store)
        let transcribe = MereRunCapabilityCatalog.speechTranscribe
        let parakeet = ["speech", "transcribe", "/tmp/talk.wav", "--model", "speech-asr-parakeet"]
        try await settle(store) {
            if case .resolved = store.identity(of: Array(parakeet.dropFirst(2)), model: "speech-asr-parakeet", for: transcribe) {
                return true
            }
            return false
        }
        XCTAssertEqual(source.scope(capability: transcribe, commandLine: parakeet).family?.id, "parakeet")

        let translate = parakeet + ["--task", "translate"]
        let pending = source.scope(capability: transcribe, commandLine: translate)
        XCTAssertTrue(isPending(store.identity(of: Array(translate.dropFirst(2)), model: "speech-asr-parakeet", for: transcribe)))
        XCTAssertEqual(pending.family?.id, "qwen3-asr", "the contract's own family, not Parakeet's last answer")
        XCTAssertFalse(pending.awaitsCLI, "a family is known, so a launch need not wait")
        XCTAssertEqual(StudioOptionScopes.filtered(translate, scope: pending).suffix(2), ["--task", "translate"])

        var draft = try XCTUnwrap(CommandCatalog.template(id: .speechTranscribe)).defaultDraft()
        draft.inputPath = "/tmp/talk.wav"
        draft.model = "speech-asr-parakeet"
        draft.task = "translate"
        let template = try XCTUnwrap(CommandCatalog.template(id: .speechTranscribe))
        XCTAssertNil(template.validationMessage(for: draft, source: source))
        let launched = template.arguments(from: draft, source: source)
        XCTAssertEqual(launched.firstIndex(of: "--task").map { launched[$0 + 1] }, "translate", "\(launched)")

        await gate.open()
        try await settle(store) {
            if case .resolved = store.identity(of: Array(translate.dropFirst(2)), model: "speech-asr-parakeet", for: transcribe) {
                return true
            }
            return false
        }
        XCTAssertEqual(source.scope(capability: transcribe, commandLine: translate).family?.id, "qwen3-asr")
    }

    /// A folder only the CLI places: a launch waits until the CLI says what it is. Then the folder
    /// keeps that answer while a choice the family resolver does not read is asked about again,
    /// so the surface does not flash back to every option and a launch need not wait.
    func testAFolderKeepsItsLastAnswerWhileAChoiceIsAskedAbout() async throws {
        let gate = Gate()
        let store = StudioModelIdentityStore(debounce: .zero)
        let folder = "/tmp/model-scope/ref2va"
        let first = ["video", "generate", "x", "--model", folder]
        let changed = first + ["--output-mode", "video-only"]
        store.use { commandLine in
            if commandLine == changed { await gate.wait() }
            return MereRunFamilyResolutionReport(
                capability: "video.generate", family: "h3-ref2va", familyTitle: "MiniMax-H3 Ref2VA",
                model: folder, source: .identified, violations: [], warnings: []
            )
        }
        let source = StudioScopeSource(identities: store)
        // Nothing is known about the folder yet: every option shows, and a launch waits.
        XCTAssertTrue(source.scope(capability: video, commandLine: first).awaitsCLI)
        var draft = try XCTUnwrap(CommandCatalog.template(id: .videoGenerate)).defaultDraft()
        draft.prompt = "x"
        draft.model = folder
        let template = try XCTUnwrap(CommandCatalog.template(id: .videoGenerate))
        XCTAssertEqual(template.validationMessage(for: draft, source: source), StudioOptionScope.awaitingCLIMessage)
        try await settle(store) { source.scope(capability: self.video, commandLine: first).awaitsCLI == false }
        // Now the folder's own checks apply (Ref2VA needs its ordered references), not the wait.
        try await settle(store) { template.validationMessage(for: draft, source: source) != StudioOptionScope.awaitingCLIMessage }
        let pending = source.scope(capability: video, commandLine: changed)
        XCTAssertEqual(pending.family?.id, "h3-ref2va")
        XCTAssertEqual(pending.identity, .notNeeded)
        XCTAssertFalse(pending.awaitsCLI, "the resolver reads the same model and routing flags")
        await gate.open()
    }

    // MARK: When Studio asks

    /// M-a: a free-text flag that can change the family (`--language`), typed a letter at a
    /// time, and a model typed into the console, each ask once, after the typing stops.
    func testTypingAsksOnceAfterTheLastKeystroke() async throws {
        let asked = AskedLines()
        let store = StudioModelIdentityStore(debounce: .milliseconds(80))
        store.use { commandLine in
            await asked.append(commandLine)
            return nil
        }
        let transcribe = MereRunCapabilityCatalog.speechTranscribe
        for language in ["f", "fr", "fre", "fren", "french"] {
            _ = store.identity(of: ["/tmp/talk.wav", "--language", language], model: nil, for: transcribe)
            try await Task.sleep(for: .milliseconds(5))
        }
        for model in ["/tmp/m", "/tmp/my", "/tmp/my-ltx"] {
            _ = store.identity(of: ["x", "--model", model], model: model, for: video)
            try await Task.sleep(for: .milliseconds(5))
        }
        try await Task.sleep(for: .milliseconds(400))
        let lines = await asked.lines
        XCTAssertEqual(lines.count, 2, "\(lines)")
        XCTAssertTrue(lines.contains(["speech", "transcribe", "/tmp/talk.wav", "--language", "french"]), "\(lines)")
        XCTAssertTrue(lines.contains(["video", "generate", "x", "--model", "/tmp/my-ltx"]), "\(lines)")
    }

    /// M-b: reading an answer never touches the file system, and the folder's date is read off
    /// the main thread when the CLI is asked, and again later to notice an edited folder.
    func testFolderDatesAreReadOffTheMainThreadAndAnEditedFolderIsAskedAgain() async throws {
        let dates = FolderDates()
        let asked = AskedLines()
        let store = StudioModelIdentityStore(debounce: .zero, recheckFoldersAfter: .milliseconds(30)) { url in
            dates.read(url)
        }
        let folder = "/tmp/model-scope/edited"
        store.use { commandLine in
            await asked.append(commandLine)
            return MereRunFamilyResolutionReport(
                capability: "video.generate", family: "h3-ref2va", familyTitle: "MiniMax-H3 Ref2VA",
                model: folder, source: .identified, violations: [], warnings: []
            )
        }
        let arguments = ["x", "--model", folder]
        try await settle(store) { !self.isPending(store.identity(of: arguments, model: folder, for: self.video)) }
        for _ in 0..<50 { _ = store.identity(of: arguments, model: folder, for: video) }
        XCTAssertFalse(dates.readOnMain, "a read on the main thread can block on a volume prompt")
        XCTAssertGreaterThanOrEqual(dates.count, 1)

        dates.touch()
        try await Task.sleep(for: .milliseconds(60))
        try await settle(store) {
            _ = store.identity(of: arguments, model: folder, for: self.video)
            return await asked.lines.count == 2
        }
        XCTAssertFalse(dates.readOnMain)
    }

    // MARK: How long an answer stands

    /// M-c: a failed `catalog resolve` (nonzero exit, unreadable output) is asked again, after
    /// a backoff, rather than standing as "couldn't identify" for the session.
    func testAFailedAnswerIsAskedAgainAfterABackoff() async throws {
        let attempts = AskedLines()
        let store = StudioModelIdentityStore(debounce: .zero, retryDelays: [.milliseconds(40)])
        let folder = "/tmp/model-scope/flaky"
        store.use { commandLine in
            await attempts.append(commandLine)
            guard await attempts.lines.count > 1 else { return nil }
            return MereRunFamilyResolutionReport(
                capability: "video.generate", family: "h3-ref2va", familyTitle: "MiniMax-H3 Ref2VA",
                model: folder, source: .identified, violations: [], warnings: []
            )
        }
        let arguments = ["x", "--model", folder]
        try await settle(store) { store.identity(of: arguments, model: folder, for: self.video) == .unidentified }
        try await Task.sleep(for: .milliseconds(80))
        try await settle(store) {
            if case .resolved = store.identity(of: arguments, model: folder, for: self.video) { return true }
            return false
        }
        let count = await attempts.lines.count
        XCTAssertEqual(count, 2)
    }

    /// M-c: installing or removing a model, or refreshing the inventory, changes what the CLI
    /// would answer, so every answer is asked again.
    func testInstallsAndInventoryRefreshesForgetTheAnswers() async throws {
        let runner = RecordingProcessRunner()
        let controller = MereRunController(
            secretStore: InMemorySecretStore(), processRunner: runner,
            cliResolver: { _ in .executable(URL(fileURLWithPath: "/stub/mere.run")) }, resolvesCLIOnInit: false
        )
        defer { controller.terminateAllProcesses() }
        let folder = "/tmp/model-scope/installed"
        controller.modelIdentities.use { _ in
            MereRunFamilyResolutionReport(
                capability: "video.generate", family: "h3-ref2va", familyTitle: "MiniMax-H3 Ref2VA",
                model: folder, source: .identified, violations: [], warnings: []
            )
        }
        let commandLine = ["video", "generate", "x", "--model", folder]
        let answered: @MainActor () async -> Bool = {
            controller.scopeSource.scope(capability: self.video, commandLine: commandLine).family != nil
        }
        try await settle(controller.modelIdentities, answered)

        let remove = try XCTUnwrap(CommandCatalog.template(id: .modelRemove))
        var draft = remove.defaultDraft()
        draft.model = "video-minimax-h3-ref2va-mlx"
        XCTAssertTrue(controller.runConsole(template: remove, draft: draft, arguments: ["model", "remove", draft.model], requestID: nil))
        let removal = try XCTUnwrap(runner.starts.last)
        removal.termination(0)
        try await settle(controller.modelIdentities) {
            self.isPending(controller.modelIdentities.identity(of: Array(commandLine.dropFirst(2)), model: folder, for: self.video))
        }
        try await settle(controller.modelIdentities, answered)

        let refresh = Task { await controller.modelStore.refresh() }
        try await settle(controller.modelIdentities) { runner.starts.count > 1 }
        let inventory = try XCTUnwrap(runner.starts.last)
        inventory.stdout("{\"inventory\":{\"rows\":[]},\"usageTerms\":[]}")
        inventory.termination(0)
        try await settle(controller.modelIdentities) { runner.starts.count > 2 }
        let metadata = try XCTUnwrap(runner.starts.last)
        metadata.stdout("{\"models\":[]}")
        metadata.termination(0)
        await refresh.value
        XCTAssertTrue(isPending(controller.modelIdentities.identity(of: Array(commandLine.dropFirst(2)), model: folder, for: video)))
    }

    // MARK: What the CLI hears

    /// A key typed into the command line never reaches `catalog resolve`'s argv, which any
    /// process on the Mac can read.
    func testTheCLIHearsNoSecrets() async throws {
        let asked = AskedLines()
        let store = StudioModelIdentityStore(debounce: .zero)
        store.use { commandLine in
            await asked.append(commandLine)
            return nil
        }
        let ocr = MereRunCapabilityCatalog.visionOCR
        _ = store.identity(
            of: ["/tmp/receipt.png", "--backend", "infinity", "--infinity-model", "/tmp/model-scope/ocr",
                 "--infinity-api-key", "sk-live-1", "--api-key=sk-live-2"],
            model: "/tmp/model-scope/ocr", for: ocr
        )
        try await settle(store) { await asked.lines.count == 1 }
        let lines = await asked.lines
        let line = try XCTUnwrap(lines.first)
        XCTAssertFalse(line.contains { $0.contains("sk-live") }, "\(line)")
        XCTAssertFalse(line.contains { $0.hasPrefix("--infinity-api-key") || $0.hasPrefix("--api-key") }, "\(line)")
        XCTAssertTrue(line.contains("/tmp/model-scope/ocr"))
    }

    // MARK: Helpers

    private func isPending(_ identity: StudioModelIdentity) -> Bool {
        switch identity {
        case .pending, .pendingAfter: return true
        case .resolved, .identified, .unidentified: return false
        }
    }

    private func settle(
        _ store: StudioModelIdentityStore,
        _ condition: @MainActor () async -> Bool,
        file: StaticString = #filePath,
        line: UInt = #line
    ) async throws {
        let deadline = ContinuousClock.now.advanced(by: .seconds(5))
        while await !condition(), ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(5))
        }
        let settled = await condition()
        XCTAssertTrue(settled, "timed out", file: file, line: line)
    }
}

/// Holds a resolver until the test lets it answer.
private actor Gate {
    private var isOpen = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    func wait() async {
        if isOpen { return }
        await withCheckedContinuation { waiters.append($0) }
    }

    func open() {
        isOpen = true
        for waiter in waiters { waiter.resume() }
        waiters = []
    }
}

private actor AskedLines {
    private(set) var lines: [[String]] = []

    func append(_ line: [String]) {
        lines.append(line)
    }
}

/// A folder's modification date that a test can bump, recording whether it was ever read on
/// the main thread.
private final class FolderDates: @unchecked Sendable {
    private let lock = NSLock()
    private var date = Date(timeIntervalSince1970: 1_000)
    private var reads = 0
    private var onMain = false

    func read(_ url: URL) -> Date? {
        lock.withLock {
            reads += 1
            if Thread.isMainThread { onMain = true }
            return date
        }
    }

    func touch() {
        lock.withLock { date = date.addingTimeInterval(60) }
    }

    var count: Int { lock.withLock { reads } }
    var readOnMain: Bool { lock.withLock { onMain } }
}
