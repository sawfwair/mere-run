import AppKit
import AVFoundation
import Combine
import Foundation
import UserNotifications

enum MereRunLaunch: Equatable {
    case executable(URL)
    case swiftRun(packagePath: URL)

    var executableURL: URL {
        switch self {
        case .executable(let url):
            return url
        case .swiftRun:
            return URL(fileURLWithPath: "/usr/bin/swift")
        }
    }

    var sourceDescription: String {
        switch self {
        case .executable(let url):
            if CLIResolver.isBundledExecutable(url) {
                return "Bundled CLI"
            }
            return url.path
        case .swiftRun(let packagePath):
            return "swift run --package-path \(packagePath.path)"
        }
    }

    func processArguments(for cliArguments: [String]) -> [String] {
        switch self {
        case .executable:
            return cliArguments
        case .swiftRun(let packagePath):
            return ["run", "--package-path", packagePath.path, "mere.run"] + cliArguments
        }
    }

    func displayCommand(for cliArguments: [String]) -> String {
        switch self {
        case .executable(let url):
            if url.path == "/usr/bin/env" {
                return (["/usr/bin/env", "mere.run"] + cliArguments).shellQuoted()
            }
            return ([url.path] + cliArguments).shellQuoted()
        case .swiftRun(let packagePath):
            return (["swift", "run", "--package-path", packagePath.path, "mere.run"] + cliArguments).shellQuoted()
        }
    }
}

enum CLIResolver {
    static func resolve(customPath: String) -> MereRunLaunch {
        let fm = FileManager.default
        let expandedCustomPath = NSString(string: customPath).expandingTildeInPath
        if !expandedCustomPath.isBlank, fm.isExecutableFile(atPath: expandedCustomPath) {
            return .executable(URL(fileURLWithPath: expandedCustomPath))
        }

        let roots = packageRootCandidates(fileManager: fm)
        let candidates = bundledCandidates()
            + siblingCandidates()
            + buildProductCandidates(packageRoots: roots)
            + installedCandidates(fileManager: fm)

        for candidate in candidates {
            if fm.isExecutableFile(atPath: candidate.path) {
                return .executable(candidate)
            }
        }

        if let packageRoot = roots.first {
            return .swiftRun(packagePath: packageRoot)
        }

        return .executable(URL(fileURLWithPath: "/usr/bin/env"))
    }

    static func existingInstalledCLI(fileManager fm: FileManager = .default) -> URL? {
        installedCandidates(fileManager: fm).first { candidate in
            fm.isExecutableFile(atPath: candidate.path)
        }
    }

    private static func bundledCandidates() -> [URL] {
        let resourceURL = Bundle.main.resourceURL
        let helpersURL = Bundle.main.bundleURL
            .appendingPathComponent("Contents/Helpers", isDirectory: true)
        return [
            helpersURL.appendingPathComponent("mere.run"),
            resourceURL?.appendingPathComponent("mere.run/mere.run"),
            resourceURL?.appendingPathComponent("mere.run"),
        ].compactMap { $0 }
    }

    /// True when `url` points inside the app bundle's helper or resource payload, used to
    /// label the resolved CLI as "Bundled CLI" regardless of the (Helpers vs Resources) layout.
    static func isBundledExecutable(_ url: URL) -> Bool {
        let roots = [
            Bundle.main.bundleURL.appendingPathComponent("Contents/Helpers", isDirectory: true).path,
            Bundle.main.resourceURL?.path,
        ].compactMap { $0 }
        return roots.contains { url.path.hasPrefix($0) }
    }

    private static func siblingCandidates() -> [URL] {
        guard let executableURL = Bundle.main.executableURL else {
            return []
        }
        return [
            executableURL.deletingLastPathComponent().appendingPathComponent("mere.run")
        ]
    }

    static func installedCandidates(fileManager fm: FileManager) -> [URL] {
        [
            "/usr/local/bin/mere.run",
            "/opt/homebrew/bin/mere.run",
            fm.homeDirectoryForCurrentUser
                .appendingPathComponent(".local", isDirectory: true)
                .appendingPathComponent("bin", isDirectory: true)
                .appendingPathComponent("mere.run", isDirectory: false)
                .path,
        ].map(URL.init(fileURLWithPath:))
    }

    private static func buildProductCandidates(packageRoots: [URL]) -> [URL] {
        packageRoots.flatMap { root in
            [
                ".build/arm64-apple-macosx/debug/mere.run",
                ".build/arm64-apple-macosx/release/mere.run",
                ".build/debug/mere.run",
                ".build/release/mere.run",
            ].map { root.appendingPathComponent($0) }
        }
    }

    private static func packageRootCandidates(fileManager fm: FileManager) -> [URL] {
        let anchors: [URL] = [
            URL(fileURLWithPath: fm.currentDirectoryPath, isDirectory: true),
            Bundle.main.executableURL?.deletingLastPathComponent(),
            Bundle.main.resourceURL,
        ].compactMap { $0 }

        var roots: [URL] = []
        var seen: Set<String> = []
        for anchor in anchors {
            guard let root = nearestPackageRoot(from: anchor, fileManager: fm) else {
                continue
            }
            let path = root.standardizedFileURL.path
            if seen.insert(path).inserted {
                roots.append(root)
            }
        }
        return roots
    }

    static func nearestPackageRoot(from anchor: URL, fileManager fm: FileManager) -> URL? {
        var cursor = startingDirectoryPath(from: anchor, fileManager: fm)
        var seen: Set<String> = []
        while true {
            guard seen.insert(cursor).inserted else {
                return nil
            }
            let packagePath = (cursor as NSString).appendingPathComponent("Package.swift")
            if fm.fileExists(atPath: packagePath) {
                return URL(fileURLWithPath: cursor, isDirectory: true)
            }

            let parent = (cursor as NSString).deletingLastPathComponent
            if parent == cursor || parent.isEmpty {
                return nil
            }
            cursor = parent
        }
    }

    private static func startingDirectoryPath(from anchor: URL, fileManager fm: FileManager) -> String {
        var path = (anchor.standardizedFileURL.path as NSString).standardizingPath
        var isDirectory: ObjCBool = false
        if fm.fileExists(atPath: path, isDirectory: &isDirectory), !isDirectory.boolValue {
            path = (path as NSString).deletingLastPathComponent
        }
        return path.isEmpty ? "/" : path
    }
}

private final class ReadinessOutputBuffer: @unchecked Sendable {
    private let lock = NSLock()
    private var value = ""

    func append(_ text: String) {
        lock.lock()
        value += text
        lock.unlock()
    }

    func text() -> String {
        lock.lock()
        defer { lock.unlock() }
        return value
    }
}

struct MereRunUtilityCommandResult: Equatable {
    let commandPreview: String
    let exitCode: Int32
    let stdout: String
    let stderr: String

    var outputText: String {
        let trimmedStdout = stdout.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedStderr = stderr.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmedStdout.isEmpty { return trimmedStderr }
        if trimmedStderr.isEmpty { return trimmedStdout }
        return "\(trimmedStdout)\n\nSTDERR\n\(trimmedStderr)"
    }
}

@MainActor
final class MereRunController: ObservableObject {
    @Published var selectedTemplate: CommandTemplate
    @Published var draft: CommandDraft
    @Published var logs: [LogLine] = []
    @Published var isRunning = false
    @Published var status = "Idle"
    @Published var lastExitCode: Int32?
    @Published var resolvedCLI = ""
    @Published var lastOutputURL: URL?
    @Published var lastRunResult: MereRunRunResult?
    /// Lossless completion stream. `lastRunResult` coalesces — two runs finishing in the same
    /// runloop turn would deliver only the latest via onChange — so must-deliver side effects
    /// (appending a chat reply, completing a library row) subscribe here and get every result.
    let runCompletions = PassthroughSubject<MereRunRunResult, Never>()
    @Published private(set) var activeRunRequestID: UUID?
    /// Conversation ids with a turn currently in flight. Tracked across ALL sessions (not just
    /// the foreground one) so the UI can disable a thread's composer and stream into the right
    /// thread even when a different conversation is in the foreground.
    @Published private(set) var runningConversationIDs: Set<UUID> = []
    /// Live, think-stripped assistant text per in-flight conversation, so a streaming bubble can
    /// render even for a background conversation (the foreground-only liveOutputText cannot).
    @Published private(set) var conversationLiveText: [UUID: String] = [:]
    /// Latest parsed `status --json` snapshot for the Studio status pill (nil until first probe).
    @Published private(set) var serverStatus: StudioServerStatus?
    @Published private(set) var queuedRunCount = 0
    @Published var readinessByMode: [StudioMode: ModelReadinessState] = [:]
    @Published var modelCapabilitiesByID: [String: StudioModelCapability] = [:]
    @Published private(set) var recommendedChatModelID: String? = nil {
        didSet {
            guard oldValue != recommendedChatModelID else { return }
            applyRecommendedChatDefaultsToCurrentDraft(replacing: oldValue)
        }
    }
    @Published private(set) var recommendedCodeModelID: String? = nil {
        didSet {
            guard oldValue != recommendedCodeModelID else { return }
            applyRecommendedCodeDefaultsToCurrentDraft(replacing: oldValue)
        }
    }
    @Published var cliPath: String {
        didSet { UserDefaults.standard.set(cliPath, forKey: Keys.cliPath) }
    }
    @Published var modelsRoot: String {
        didSet { UserDefaults.standard.set(modelsRoot, forKey: Keys.modelsRoot) }
    }
    @Published var hubCache: String {
        didSet { UserDefaults.standard.set(hubCache, forKey: Keys.hubCache) }
    }
    @Published var workingDirectory: String {
        didSet { UserDefaults.standard.set(workingDirectory, forKey: Keys.workingDirectory) }
    }
    // The runtime server the Studio talks to for model load/unload. Owned here — and persisted —
    // rather than derived from a transient command draft, so requests target the actual runtime.
    @Published var runtimeHost: String {
        didSet { UserDefaults.standard.set(runtimeHost, forKey: Keys.runtimeHost) }
    }
    @Published var runtimePort: Int {
        didSet { UserDefaults.standard.set(runtimePort, forKey: Keys.runtimePort) }
    }
    @Published var runtimeAPIKey: String {
        didSet { UserDefaults.standard.set(runtimeAPIKey, forKey: Keys.runtimeAPIKey) }
    }
    @Published private(set) var liveOutputText = ""
    @Published private(set) var currentProgress: StudioRunProgress?
    /// Live progress keyed by durable Studio request id. Unlike `currentProgress`, this covers
    /// background runs too, so Library rows keep reporting useful work while another run owns
    /// the foreground canvas.
    @Published private(set) var progressByRequestID: [UUID: StudioRunProgress] = [:]
    @Published private(set) var cliVersion: String?
    @Published private(set) var cliInstallationStatus = CLIInstallationStatus.checking

    /// The app's own version, read from the bundle for the version handshake display.
    var appVersion: String {
        let short = Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String
        return short ?? "dev"
    }

    private enum Keys {
        static let cliPath = "mererun.app.cliPath"
        static let modelsRoot = "mererun.app.modelsRoot"
        static let hubCache = "mererun.app.hubCache"
        static let workingDirectory = "mererun.app.workingDirectory"
        static let runtimeHost = "mererun.app.runtimeHost"
        static let runtimePort = "mererun.app.runtimePort"
        static let runtimeAPIKey = "mererun.app.runtimeAPIKey"
    }

    private let processRunner: MereRunProcessRunning
    private let fileSystem: MereRunFileProbing
    private let cliResolve: (String) -> MereRunLaunch
    private var didSynchronizeCLIInstallationAfterLaunch = false
    private var utilityProcesses: [UUID: MereRunRunningProcess] = [:]
    private var readinessProcesses: [StudioMode: MereRunRunningProcess] = [:]
    private var readinessRequests: [StudioMode: ReadinessRequest] = [:]
    /// Owns every run's lifecycle: lanes, FIFO queues, child processes, buffers, progress and
    /// artifact detection. The controller mirrors the foreground job into its published console
    /// fields and re-broadcasts completions; views may also observe a `Job` directly.
    let jobs: JobStore
    /// The job whose live state mirrors into the published console fields (the run the
    /// single-pane console/canvas currently shows). Background jobs still complete into the
    /// library by request id. Persists past completion so the last run's result stays visible.
    private var foregroundJob: Job?
    private var jobEventSubscription: AnyCancellable?

    /// Conservative cap on simultaneous inference runs. ML inference is memory-heavy, so this
    /// stays small; `JobLane.inference.capacity` is the single knob.
    static let maxConcurrentRuns = JobLane.inference.capacity

    private struct ReadinessRequest: Equatable {
        let modelID: String
        let cliPath: String
        let modelsRoot: String
        let hubCache: String
    }

    var commandArguments: [String] {
        commandArguments(template: selectedTemplate, draft: draft)
    }

    var commandPreview: String {
        commandPreview(template: selectedTemplate, draft: draft, masksSecrets: true)
    }

    var advancedCommandPreview: String {
        commandPreview(template: selectedTemplate, draft: draft, masksSecrets: false)
    }

    var canSubmitVideoSessionRequest: Bool {
        guard let job = foregroundJob else { return false }
        return job.request.template.id == .videoSession && job.state.isRunning
    }

    func canSteerRealtimeMusic(requestID: UUID) -> Bool {
        jobs.running.contains {
            $0.request.requestID == requestID && $0.request.template.id == .musicRealtime
        }
    }

    func isRequestRunning(_ requestID: UUID) -> Bool {
        jobs.running.contains { $0.request.requestID == requestID }
    }

    func runningRequestID(for templateID: CommandTemplateID) -> UUID? {
        jobs.running.first {
            $0.request.template.id == templateID && $0.request.requestID != nil
        }?.request.requestID
    }

    func logs(for requestID: UUID) -> [LogLine] {
        jobs.job(requestID: requestID)?.log.lines ?? []
    }

    init(
        processRunner: MereRunProcessRunning = FoundationMereRunProcessRunner(),
        fileSystem: MereRunFileProbing = FileManager.default,
        cliResolver: @escaping (String) -> MereRunLaunch = { CLIResolver.resolve(customPath: $0) },
        resolvesCLIOnInit: Bool = true
    ) {
        self.processRunner = processRunner
        self.fileSystem = fileSystem
        self.cliResolve = cliResolver
        jobs = JobStore(processRunner: processRunner, fileSystem: fileSystem)
        let initial = CommandCatalog.templates.first!
        selectedTemplate = initial
        draft = initial.defaultDraft()
        cliPath = UserDefaults.standard.string(forKey: Keys.cliPath) ?? ""
        modelsRoot = UserDefaults.standard.string(forKey: Keys.modelsRoot) ?? ""
        hubCache = UserDefaults.standard.string(forKey: Keys.hubCache) ?? ""
        workingDirectory = UserDefaults.standard.string(forKey: Keys.workingDirectory)
            ?? FileManager.default.homeDirectoryForCurrentUser.path
        runtimeHost = UserDefaults.standard.string(forKey: Keys.runtimeHost) ?? "127.0.0.1"
        runtimePort = (UserDefaults.standard.object(forKey: Keys.runtimePort) as? Int) ?? 8080
        runtimeAPIKey = UserDefaults.standard.string(forKey: Keys.runtimeAPIKey) ?? ""
        jobEventSubscription = jobs.events.sink { [weak self] event in
            self?.handle(event)
        }
        if resolvesCLIOnInit {
            refreshResolvedCLI()
        }
    }

    func select(_ template: CommandTemplate) {
        selectedTemplate = template
        draft = defaultDraft(for: template)
        lastOutputURL = nil
        lastExitCode = nil
        status = "Idle"
    }

    func defaultDraft(for template: CommandTemplate) -> CommandDraft {
        var nextDraft = template.defaultDraft()
        applyRecommendedChatDefaults(to: &nextDraft, templateID: template.id, replacing: nil)
        applyRecommendedCodeDefaults(to: &nextDraft, templateID: template.id, replacing: nil)
        return nextDraft
    }

    func recommendedChatModelForStudioDefault() -> String {
        recommendedChatModelID ?? StudioChatDefaults.fallbackModelID
    }

    func recommendedCodeModelForStudioDefault() -> String {
        recommendedCodeModelID ?? StudioCodeDefaults.fallbackModelID
    }

    func applyRecommendedDefaults(to studioDraft: inout StudioDraft, for mode: StudioMode) {
        switch mode {
        case .chat:
            let recommended = recommendedChatModelForStudioDefault()
            if StudioChatDefaults.shouldReplaceModelDefault(studioDraft.model) {
                studioDraft.model = recommended
            }
        case .code:
            let recommended = recommendedCodeModelForStudioDefault()
            if StudioCodeDefaults.shouldReplaceModelDefault(studioDraft.model) {
                studioDraft.model = recommended
            }
        default:
            break
        }
    }

    /// Selects the active Studio mode's canonical command and carries the quick composer's typed
    /// values into Advanced. Keeping this transition in the controller makes opening Advanced and
    /// switching modes while it is already open follow the same tested path.
    func syncAdvanced(to mode: StudioMode, from studioDraft: StudioDraft) {
        guard let template = CommandCatalog.template(id: mode.defaultTemplateID) else { return }
        select(template)
        draft.prompt = studioDraft.prompt
        if !studioDraft.model.isBlank { draft.model = studioDraft.model }
        if !studioDraft.inputPath.isBlank { draft.inputPath = studioDraft.inputPath }
        draft.temperature = studioDraft.temperature
        draft.topP = studioDraft.topP
        draft.minP = studioDraft.minP
        draft.maxTokens = studioDraft.maxTokens
        draft.contextSize = studioDraft.contextSize
        draft.topK = studioDraft.topK
        draft.kvBits = studioDraft.kvBits
        draft.kvQuantScheme = studioDraft.kvQuantScheme
        draft.kvGroupSize = studioDraft.kvGroupSize
        draft.quantizedKVStart = studioDraft.quantizedKVStart
        draft.responseFormat = studioDraft.responseFormat
        draft.thinkingMode = studioDraft.thinkingMode
        draft.loraPath = studioDraft.loraPath
        draft.loraScale = studioDraft.loraScale
        draft.force = studioDraft.stats
        draft.tools = studioDraft.tools
        draft.toolLoop = studioDraft.toolLoop
        draft.allowShellExec = studioDraft.allowShellExec
        draft.allowAbsoluteToolPaths = studioDraft.allowAbsoluteToolPaths
        draft.autoApproveTools = studioDraft.autoApproveTools
        draft.sandboxDir = studioDraft.sandboxDir
        draft.requireInstalled = studioDraft.requireInstalled
        draft.json = studioDraft.preflightJSON
        draft.cfgScale = studioDraft.cfgScale
        draft.strength = studioDraft.strength
        draft.sigmaShift = studioDraft.sigmaShift
        draft.referenceImagePaths = studioDraft.referenceImagePaths
        draft.keepOriginalAspect = studioDraft.keepOriginalAspect
        draft.structuredPrompt = studioDraft.structuredPrompt
        draft.structuredPromptModel = studioDraft.structuredPromptModel
        draft.structuredPromptMaxTokens = studioDraft.structuredPromptMaxTokens
        draft.maxSequenceLength = studioDraft.imageMaxSequenceLength
        draft.kreaConditioningMultiplier = studioDraft.kreaConditioningMultiplier
        draft.kreaConditioningLayerWeights = studioDraft.kreaConditioningLayerWeights
        draft.kreaBaseQuantizationBits = studioDraft.kreaBaseQuantizationBits
        draft.progressJSON = studioDraft.progressJSON
        draft.language = studioDraft.language
        draft.backend = studioDraft.backend
        draft.timestamps = studioDraft.timestamps
        draft.fps = studioDraft.fps
        draft.numFrames = studioDraft.numFrames
        draft.useDuration = studioDraft.useDuration
        draft.durationSeconds = studioDraft.durationSeconds
        draft.videoQuality = studioDraft.videoQuality
        draft.videoOutputMode = studioDraft.videoOutputMode
        draft.audioPath = studioDraft.audioPath
        draft.audioStartTime = studioDraft.audioStartTime
        draft.audioMaxDuration = studioDraft.audioMaxDuration
        draft.endImagePath = studioDraft.endImagePath
        draft.endImageStrength = studioDraft.endImageStrength
        draft.scheduleShift = studioDraft.scheduleShift
        draft.a2vGuidanceScale = studioDraft.a2vGuidanceScale
        draft.videoCFGGuidanceScale = studioDraft.videoCFGGuidanceScale
        draft.audioCFGGuidanceScale = studioDraft.audioCFGGuidanceScale
        draft.v2aGuidanceScale = studioDraft.v2aGuidanceScale
        draft.a2vSteps = studioDraft.a2vSteps
        draft.preflight = studioDraft.preflight
        draft.timings = studioDraft.timings
        draft.timingsOutputPath = studioDraft.timingsOutputPath
    }

    private func applyRecommendedChatDefaultsToCurrentDraft(replacing oldRecommendation: String?) {
        applyRecommendedChatDefaults(
            to: &draft,
            templateID: selectedTemplate.id,
            replacing: oldRecommendation
        )
    }

    private func applyRecommendedCodeDefaultsToCurrentDraft(replacing oldRecommendation: String?) {
        applyRecommendedCodeDefaults(
            to: &draft,
            templateID: selectedTemplate.id,
            replacing: oldRecommendation
        )
    }

    private func applyRecommendedChatDefaults(
        to draft: inout CommandDraft,
        templateID: CommandTemplateID,
        replacing oldRecommendation: String?
    ) {
        let recommended = recommendedChatModelForStudioDefault()
        switch templateID {
        case .textChat, .openWebui:
            if StudioChatDefaults.shouldReplaceModelDefault(draft.model, oldRecommendation: oldRecommendation) {
                draft.model = recommended
            }
        case .apiServe:
            if StudioChatDefaults.shouldReplaceModelDefault(draft.model, oldRecommendation: oldRecommendation) {
                draft.model = recommended
            }
            if StudioChatDefaults.shouldReplaceServingEngineDefault(
                draft.engine,
                oldRecommendation: oldRecommendation
            ) {
                draft.engine = StudioChatDefaults.servingEngine(for: recommended)
            }
        default:
            break
        }
    }

    private func applyRecommendedCodeDefaults(
        to draft: inout CommandDraft,
        templateID: CommandTemplateID,
        replacing oldRecommendation: String?
    ) {
        let recommended = recommendedCodeModelForStudioDefault()
        switch templateID {
        case .textCode, .agentOnboard:
            if StudioCodeDefaults.shouldReplaceModelDefault(draft.model, oldRecommendation: oldRecommendation) {
                draft.model = recommended
            }
        default:
            break
        }
    }

    /// URL on the runtime server (model load/unload) for `path`, built from the owned
    /// host/port rather than a transient command draft.
    func runtimeURL(path: String) -> URL {
        let host = runtimeHost.trimmingCharacters(in: .whitespacesAndNewlines)
        let safeHost = host.isEmpty ? "127.0.0.1" : host
        var components = URLComponents()
        components.scheme = "http"
        components.host = safeHost
        components.port = runtimePort
        components.path = path
        return components.url ?? URL(string: "http://127.0.0.1:8080\(path)")!
    }

    /// The `Authorization` header value for runtime-server requests, or nil when no key is set.
    var runtimeAuthorizationHeader: String? {
        let key = runtimeAPIKey.trimmingCharacters(in: .whitespacesAndNewlines)
        return key.isEmpty ? nil : "Bearer \(key)"
    }

    func refreshResolvedCLI() {
        let launch = cliResolve(cliPath)
        resolvedCLI = displayDescription(for: launch)
    }

    func installTerminalCLI() {
        guard cliInstallationStatus.phase != .synchronizing else { return }
        let context = CLIInstallationContext.current(customCLIPath: cliPath)
        cliInstallationStatus = synchronizingCLIStatus(from: cliInstallationStatus)
        Task { @MainActor in
            let nextStatus = await Task.detached(priority: .utility) {
                CLIInstallationSynchronizer.performManualAction(context: context)
            }.value
            publishCLIInstallationStatus(nextStatus, automatic: false)
        }
    }

    func synchronizeCLIInstallationAfterLaunch() async {
        guard !didSynchronizeCLIInstallationAfterLaunch else { return }
        didSynchronizeCLIInstallationAfterLaunch = true
        let context = CLIInstallationContext.current(customCLIPath: cliPath)
        cliInstallationStatus = synchronizingCLIStatus(from: cliInstallationStatus)
        let nextStatus = await Task.detached(priority: .utility) {
            CLIInstallationSynchronizer.synchronizeAfterLaunch(context: context)
        }.value
        publishCLIInstallationStatus(nextStatus, automatic: true)
    }

    func installCodexSkills() {
        let outcome = CodexSkillInstaller.installBundledSkillsIfAvailable()
        handleSkillInstall(outcome)
    }

    /// Stores (or clears) the Hugging Face token via the CLI's `config` store so gated/private
    /// model pulls can authenticate. Returns whether the command succeeded.
    @discardableResult
    func saveHuggingFaceToken(_ token: String) async -> Bool {
        let trimmed = token.trimmingCharacters(in: .whitespacesAndNewlines)
        let args = trimmed.isEmpty
            ? ["config", "unset", "hf-token"]
            : ["config", "set", "hf-token", "--from-env", "MERERUN_CONFIG_VALUE"]
        let result = await utilityCommandResult(
            args: args,
            environmentOverrides: trimmed.isEmpty ? [:] : ["MERERUN_CONFIG_VALUE": trimmed]
        )
        return result.exitCode == 0
    }

    /// Probes the configured local API server and publishes the parsed snapshot for the status pill.
    func refreshServerStatus() async {
        var args = ["status", "--json", "--host", runtimeHost, "--port", String(runtimePort)]
        if !runtimeAPIKey.isBlank { args += ["--api-key", runtimeAPIKey] }
        let result = await utilityCommandResult(args: args)
        serverStatus = StudioServerStatus.parse(jsonStdout: result.stdout)
    }

    /// The persisted Hugging Face endpoint (config key hf-endpoint), or "" if unset.
    func loadHuggingFaceEndpoint() async -> String {
        let result = await utilityCommandResult(args: ["config", "get", "hf-endpoint", "--reveal"], masksSecrets: false)
        guard result.exitCode == 0 else { return "" }
        let value = result.stdout.trimmingCharacters(in: .whitespacesAndNewlines)
        // `config get` prints "(unset)" style text when missing; treat any non-URL as empty.
        return value.hasPrefix("http") ? value : ""
    }

    func saveHuggingFaceEndpoint(_ endpoint: String) async -> Bool {
        let trimmed = endpoint.trimmingCharacters(in: .whitespacesAndNewlines)
        let args = trimmed.isEmpty
            ? ["config", "unset", "hf-endpoint"]
            : ["config", "set", "hf-endpoint", trimmed]
        let result = await utilityCommandResult(args: args)
        return result.exitCode == 0
    }

    /// Builds the Export Diagnostics report from live app state without copying console
    /// text or command arguments, both of which can contain private user content.
    func diagnosticsReport(libraryItems: [StudioLibraryItem]) -> String {
        let recentLog = logs.suffix(400).reduce(into: StudioDiagnostics.LogSummary()) { summary, line in
            switch line.stream {
            case .system: summary.systemCount += 1
            case .stdout: summary.stdoutCount += 1
            case .stderr: summary.stderrCount += 1
            }
        }
        return StudioDiagnostics.report(
            appVersion: appVersion,
            cliVersion: cliVersion,
            resolvedCLI: resolvedCLI,
            serverStatus: serverStatus,
            libraryItems: libraryItems,
            recentLog: recentLog
        )
    }

    /// Reads every persisted configuration value with secrets masked, for the Settings summary.
    func loadConfigurationSummary() async -> String {
        let result = await utilityCommandResult(args: ["config", "list"], masksSecrets: false)
        guard result.exitCode == 0 else { return "" }
        return result.stdout.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Resolves the on-disk configuration file path so Settings can reveal it in Finder.
    func loadConfigurationPath() async -> String {
        let result = await utilityCommandResult(args: ["config", "path"], masksSecrets: false)
        guard result.exitCode == 0 else { return "" }
        return result.stdout.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Lists saved voice-clone profiles via `speech profile list` (tab-separated id/name/updated).
    func loadVoiceProfiles() async -> [StudioVoiceProfile] {
        let result = await utilityCommandResult(args: ["speech", "profile", "list"])
        guard result.exitCode == 0 else { return [] }
        return StudioVoiceProfile.parse(listOutput: result.stdout)
    }

    func loadMIDIInputs() async -> [String] {
        let result = await utilityCommandResult(args: ["music", "realtime", "--list-midi-inputs"])
        guard result.exitCode == 0 else { return [] }
        return result.stdout
            .components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty && $0 != "No MIDI input sources found." }
    }

    /// Offline guide topics for the in-app help panel (`guide --list --json`).
    func loadGuideTopics() async -> [StudioGuideTopic] {
        let result = await utilityCommandResult(args: ["guide", "--list", "--json"])
        guard result.exitCode == 0 else { return [] }
        return StudioGuideTopic.parse(listJSON: result.stdout)
    }

    /// Markdown content for one guide topic (`guide <command-path> --json`).
    func loadGuideContent(commandPath: [String]) async -> String {
        let result = await utilityCommandResult(args: ["guide"] + commandPath + ["--json"])
        guard result.exitCode == 0 else {
            return result.stderr.isBlank ? "No guide is available for this topic." : result.stderr
        }
        return StudioGuideTopic.parseContent(payloadJSON: result.stdout) ?? result.stdout
    }

    func commandArguments(template: CommandTemplate, draft: CommandDraft) -> [String] {
        var args: [String] = []
        if !modelsRoot.isBlank {
            args += ["--models-root", NSString(string: modelsRoot).expandingTildeInPath]
        }
        args += template.arguments(from: draft)
        return args
    }

    func commandPreview(template: CommandTemplate, draft: CommandDraft, masksSecrets: Bool) -> String {
        let launch = cliResolve(cliPath)
        let args = commandArguments(template: template, draft: draft)
        return launch.displayCommand(for: masksSecrets ? args.maskingSecrets() : args)
    }

    func utilityCommandResult(
        args: [String],
        commandID: UUID = UUID(),
        masksSecrets: Bool = true,
        environmentOverrides: [String: String] = [:],
        onOutput: (@MainActor @Sendable (String) -> Void)? = nil
    ) async -> MereRunUtilityCommandResult {
        await utilityCommandResult(
            args: args,
            commandID: commandID,
            masksSecrets: masksSecrets,
            environmentOverrides: environmentOverrides,
            callbacks: .init(onOutput: onOutput)
        )
    }

    func utilityCommandResult(
        args: [String],
        commandID: UUID = UUID(),
        masksSecrets: Bool = true,
        environmentOverrides: [String: String] = [:],
        onStandardOutput: @escaping @MainActor @Sendable (String) -> Void
    ) async -> MereRunUtilityCommandResult {
        await utilityCommandResult(
            args: args,
            commandID: commandID,
            masksSecrets: masksSecrets,
            environmentOverrides: environmentOverrides,
            callbacks: .init(onStandardOutput: onStandardOutput)
        )
    }

    private struct UtilityOutputCallbacks {
        let onOutput: (@MainActor @Sendable (String) -> Void)?
        let onStandardOutput: (@MainActor @Sendable (String) -> Void)?

        init(
            onOutput: (@MainActor @Sendable (String) -> Void)? = nil,
            onStandardOutput: (@MainActor @Sendable (String) -> Void)? = nil
        ) {
            self.onOutput = onOutput
            self.onStandardOutput = onStandardOutput
        }
    }

    private func utilityCommandResult(
        args: [String],
        commandID: UUID,
        masksSecrets: Bool,
        environmentOverrides: [String: String],
        callbacks: UtilityOutputCallbacks
    ) async -> MereRunUtilityCommandResult {
        let launch = cliResolve(cliPath)
        let cliArgs: [String]
        if !modelsRoot.isBlank {
            cliArgs = ["--models-root", NSString(string: modelsRoot).expandingTildeInPath] + args
        } else {
            cliArgs = args
        }
        let display = launch.displayCommand(for: masksSecrets ? cliArgs.maskingSecrets() : cliArgs)
        let output = ReadinessOutputBuffer()
        let errors = ReadinessOutputBuffer()

        return await withCheckedContinuation { continuation in
            do {
                let process = try processRunner.start(
                    configuration: processConfiguration(
                        launch: launch,
                        args: cliArgs,
                        environmentTemplateID: .custom,
                        environmentDraft: CommandDraft(),
                        environmentOverrides: environmentOverrides
                    ),
                    stdout: { text in
                        output.append(text)
                        if callbacks.onOutput != nil || callbacks.onStandardOutput != nil {
                            Task { @MainActor in
                                callbacks.onOutput?(text)
                                callbacks.onStandardOutput?(text)
                            }
                        }
                    },
                    stderr: { text in
                        errors.append(text)
                        if let onOutput = callbacks.onOutput {
                            Task { @MainActor in onOutput(text) }
                        }
                    },
                    termination: { [weak self] code in
                        let result = MereRunUtilityCommandResult(
                            commandPreview: display,
                            exitCode: code,
                            stdout: output.text(),
                            stderr: errors.text()
                        )
                        Task { @MainActor in
                            self?.utilityProcesses[commandID] = nil
                            continuation.resume(returning: result)
                        }
                    }
                )
                utilityProcesses[commandID] = process
            } catch {
                continuation.resume(
                    returning: MereRunUtilityCommandResult(
                        commandPreview: display,
                        exitCode: -1,
                        stdout: "",
                        stderr: error.localizedDescription
                    )
                )
            }
        }
    }

    @discardableResult
    func cancelUtilityCommand(_ commandID: UUID) -> Bool {
        guard let process = utilityProcesses[commandID] else { return false }
        process.terminate()
        return true
    }

    @discardableResult
    func interruptUtilityCommand(_ commandID: UUID) -> Bool {
        guard let process = utilityProcesses[commandID] else { return false }
        process.interrupt()
        return true
    }


    @discardableResult
    func run(studio request: StudioRunRequest) -> Bool {
        // Track the conversation as in-flight at SUBMISSION time, not at start, so a turn that
        // queues behind the concurrency cap still blocks a second send into the same thread and
        // shows a pending bubble. Cleared on every exit.
        if let conversationID = request.conversationID {
            runningConversationIDs.insert(conversationID)
        }
        if let shortCircuit = ensureCameraAccess(
            for: request.template.id,
            retry: { _ = $0.run(studio: request) }
        ) {
            return shortCircuit
        }
        return submitInferenceJob(
            template: request.template,
            draft: request.draft,
            requestID: request.id,
            conversationID: request.conversationID,
            queueNotice: "Queued \(request.mode.title.lowercased()) job."
        )
    }

    func checkReadiness(for mode: StudioMode, draft studioDraft: StudioDraft) {
        let requirement = StudioCommandAdapter.capabilityRequirement(for: mode, draft: studioDraft)
        guard let requirement else {
            cancelReadinessCheck(for: mode)
            readinessRequests[mode] = nil
            readinessByMode[mode] = .ready
            return
        }

        let modelID: String
        switch requirement {
        case .managedModel(let id):
            modelID = id
        case .unavailable(let message):
            cancelReadinessCheck(for: mode)
            readinessRequests[mode] = nil
            readinessByMode[mode] = .unsupported(message)
            return
        }

        let request = ReadinessRequest(
            modelID: modelID,
            cliPath: cliPath,
            modelsRoot: modelsRoot,
            hubCache: hubCache
        )
        if readinessRequests[mode] == request, readinessProcesses[mode] != nil {
            return
        }

        cancelReadinessCheck(for: mode)
        readinessRequests[mode] = request
        readinessByMode[mode] = .checking

        if let message = modelCapabilitiesByID[modelID]?.unavailableMessage {
            readinessByMode[mode] = .unsupported(message)
            return
        }

        if modelCapabilitiesByID[modelID] != nil {
            startModelListReadinessCheck(for: mode, modelID: modelID, request: request)
            return
        }

        startCapabilityReadinessCheck(for: mode, modelID: modelID, request: request)
    }

    private func startCapabilityReadinessCheck(
        for mode: StudioMode,
        modelID: String,
        request: ReadinessRequest
    ) {
        let launch = cliResolve(cliPath)
        let capabilityTemplate = CommandCatalog.template(id: .modelCapabilities) ?? selectedTemplate
        var capabilityDraft = capabilityTemplate.defaultDraft()
        capabilityDraft.all = true
        capabilityDraft.json = true
        let args = commandArguments(
            template: capabilityTemplate,
            draft: capabilityDraft
        )
        let output = ReadinessOutputBuffer()
        let errors = ReadinessOutputBuffer()

        do {
            readinessProcesses[mode] = try processRunner.start(
                configuration: processConfiguration(
                    launch: launch,
                    args: args,
                    environmentTemplateID: capabilityTemplate.id,
                    environmentDraft: capabilityDraft
                ),
                stdout: { text in output.append(text) },
                stderr: { text in errors.append(text) },
                termination: { [weak self] code in
                    Task { @MainActor in
                        guard self?.readinessRequests[mode] == request else { return }
                        guard let self else { return }
                        let report = ModelCapabilitiesParser.report(from: output.text())
                        if !report.capabilitiesByID.isEmpty {
                            self.modelCapabilitiesByID = report.capabilitiesByID
                        }
                        if let recommendedChatModelID = report.recommendedChatModelID,
                           !recommendedChatModelID.isBlank {
                            self.recommendedChatModelID = recommendedChatModelID
                        }
                        if let recommendedCodeModelID = report.recommendedCodeModelID,
                           !recommendedCodeModelID.isBlank {
                            self.recommendedCodeModelID = recommendedCodeModelID
                        }

                        if let message = self.modelCapabilitiesByID[modelID]?.unavailableMessage {
                            self.readinessProcesses[mode] = nil
                            self.readinessByMode[mode] = .unsupported(message)
                            return
                        }

                        if code != 0, report.capabilitiesByID.isEmpty {
                            self.readinessProcesses[mode] = nil
                            let detail = errors.text().trimmingCharacters(in: .whitespacesAndNewlines)
                            self.readinessByMode[mode] = .unknown(
                                detail.isEmpty ? "Could not check model capabilities." : detail
                            )
                            return
                        }

                        self.startModelListReadinessCheck(for: mode, modelID: modelID, request: request)
                    }
                }
            )
        } catch {
            readinessRequests[mode] = nil
            readinessProcesses[mode] = nil
            readinessByMode[mode] = .unknown(error.localizedDescription)
        }
    }

    private func startModelListReadinessCheck(
        for mode: StudioMode,
        modelID: String,
        request: ReadinessRequest
    ) {
        let launch = cliResolve(cliPath)
        let modelListTemplate = CommandCatalog.template(id: .modelList) ?? selectedTemplate
        let modelListDraft = modelListTemplate.defaultDraft()
        let args = commandArguments(
            template: modelListTemplate,
            draft: modelListDraft
        )
        let output = ReadinessOutputBuffer()

        do {
            readinessProcesses[mode] = try processRunner.start(
                configuration: processConfiguration(
                    launch: launch,
                    args: args,
                    environmentTemplateID: modelListTemplate.id,
                    environmentDraft: modelListDraft
                ),
                stdout: { text in output.append(text) },
                stderr: { _ in },
                termination: { [weak self] _ in
                    Task { @MainActor in
                        guard self?.readinessRequests[mode] == request else { return }
                        self?.readinessProcesses[mode] = nil
                        if let message = self?.modelCapabilitiesByID[modelID]?.unavailableMessage {
                            self?.readinessByMode[mode] = .unsupported(message)
                            return
                        }
                        self?.readinessByMode[mode] = ModelReadinessParser.state(
                            for: modelID,
                            modelListOutput: output.text()
                        )
                    }
                }
            )
        } catch {
            readinessRequests[mode] = nil
            readinessProcesses[mode] = nil
            readinessByMode[mode] = .unknown(error.localizedDescription)
        }
    }

    @discardableResult
    func run() -> Bool {
        guard jobs.hasCapacity(in: .inference) else { return false }
        if let shortCircuit = ensureCameraAccess(for: selectedTemplate.id, retry: { _ = $0.run() }) {
            return shortCircuit
        }
        return submitInferenceJob(
            template: selectedTemplate,
            draft: draft,
            requestID: nil,
            conversationID: nil,
            queueNotice: nil
        )
    }

    /// Snapshots the launch (resolved CLI, argv, environment, working directory) into a
    /// `JobRequest` and hands it to the inference lane. Returns false only when the job failed
    /// preflight or could not launch; a queued job returns true.
    private func submitInferenceJob(
        template: CommandTemplate,
        draft: CommandDraft,
        requestID: UUID?,
        conversationID: UUID?,
        queueNotice: String?
    ) -> Bool {
        refreshResolvedCLI()
        let launch = cliResolve(cliPath)
        let args = commandArguments(template: template, draft: draft)
        let request = JobRequest(
            lane: .inference,
            template: template,
            draft: draft,
            requestID: requestID,
            conversationID: conversationID,
            configuration: processConfiguration(
                launch: launch,
                args: args,
                environmentTemplateID: template.id,
                environmentDraft: draft
            ),
            displayCommand: launch.displayCommand(for: args)
        )
        let id = jobs.submit(request)
        refreshQueuedRunCount()
        guard let job = jobs.job(id) else { return false }
        if job.state.isQueued, let queueNotice {
            append(queueNotice, stream: .system)
        }
        return !job.state.isTerminal
    }

    func cancel() {
        guard let job = foregroundJob, job.state.isRunning else { return }
        jobs.cancel(job.id)
    }

    @discardableResult
    func cancel(requestID: UUID) -> Bool {
        guard let job = jobs.job(requestID: requestID), job.state.isRunning else { return false }
        return jobs.cancel(job.id)
    }

    @discardableResult
    func submitRealtimeMusicCommand(_ line: String, requestID: UUID) -> Bool {
        guard let job = jobs.job(requestID: requestID),
              job.request.template.id == .musicRealtime,
              job.state.isRunning else {
            return false
        }
        return jobs.sendLiveControl(line, to: job.id)
    }

    @discardableResult
    func submitVideoSessionRequest() -> Bool {
        guard let job = foregroundJob,
              job.request.template.id == .videoSession,
              job.state.isRunning else {
            status = "Start the resident session first"
            append("Start the resident LTX session before submitting a render.", stream: .stderr)
            return false
        }

        let prompt = draft.prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty else {
            status = "Prompt required"
            jobs.annotate(job.id, "Enter a prompt for the resident render.", stream: .stderr)
            return false
        }
        guard !draft.outputPath.isBlank else {
            status = "Output required"
            jobs.annotate(job.id, "Choose an output MP4 path for the resident render.", stream: .stderr)
            return false
        }

        let request = StudioVideoSessionRequest(
            id: UUID().uuidString,
            prompt: prompt,
            output: NSString(string: draft.outputPath).expandingTildeInPath,
            width: draft.width,
            height: draft.height,
            numFrames: draft.numFrames,
            fps: draft.fps,
            seed: Int(draft.seed),
            image: expandedOptionalPath(draft.imagePath),
            imageStrength: draft.imagePath.isBlank ? nil : draft.strength,
            endImage: expandedOptionalPath(draft.endImagePath),
            endImageStrength: draft.endImagePath.isBlank ? nil : draft.endImageStrength
        )
        return jobs.sendVideoSessionRequest(request, to: job.id)
    }

    // MARK: Foreground mirror

    private func handle(_ event: JobStore.Event) {
        switch event {
        case .started(let job):
            setForeground(job)
        case .changed(let job):
            mirrorForeground(job)
            mirrorCrossRunState(job)
        case .finished(let job, let result):
            finish(job, result: result)
        }
        refreshQueuedRunCount()
    }

    /// Resets the published console fields to reflect `job` as the run the single-pane
    /// console/canvas shows. Background jobs keep their own state and complete into the library.
    private func setForeground(_ job: Job) {
        foregroundJob = job
        logs = job.log.lines
        liveOutputText = job.liveText
        currentProgress = job.progress
        status = job.status
        activeRunRequestID = job.request.requestID
        lastOutputURL = job.primaryArtifactURL
        lastExitCode = job.exitCode
        isRunning = job.state.isRunning
    }

    /// Mirrors `job`'s live state into the published console fields while it is the foreground.
    private func mirrorForeground(_ job: Job) {
        guard job === foregroundJob else { return }
        logs = job.log.lines
        liveOutputText = job.liveText
        currentProgress = job.progress
        status = job.status
        if let url = job.primaryArtifactURL { lastOutputURL = url }
    }

    /// Publishes the per-request and per-conversation state every run reports, foreground or not.
    private func mirrorCrossRunState(_ job: Job) {
        if let requestID = job.request.requestID, progressByRequestID[requestID] != job.progress {
            progressByRequestID[requestID] = job.progress
        }
        if let conversationID = job.request.conversationID,
           conversationLiveText[conversationID] != job.conversationLiveText {
            conversationLiveText[conversationID] = job.conversationLiveText
        }
    }

    private func finish(_ job: Job, result: JobResult) {
        if case .preflightFailed(let failure) = job.state {
            // Preflight failures never became the foreground: report them on the console the
            // user is looking at, as the run would have.
            switch failure {
            case .invalidRequest(let message):
                append(message, stream: .system)
                status = message
            case .outputLocationUnavailable(let message):
                append(message, stream: .stderr)
                status = "Output path unavailable"
            }
            lastExitCode = result.exitCode
        } else if job === foregroundJob {
            logs = job.log.lines
            liveOutputText = job.liveText
            currentProgress = nil
            status = job.status
            lastExitCode = result.exitCode
            if let outputURL = result.outputURL { lastOutputURL = outputURL }
            isRunning = false
            activeRunRequestID = nil
        }

        if let requestID = job.request.requestID {
            progressByRequestID[requestID] = nil
        }
        if let conversationID = job.request.conversationID {
            conversationLiveText[conversationID] = nil
            runningConversationIDs.remove(conversationID)
        }
        if job.startedAt != nil {
            notifyCompletionIfNeeded(
                success: result.exitCode == 0,
                summary: result.exitCode == 0
                    ? (result.outputURL?.lastPathComponent ?? "Completed")
                    : "Exited with code \(result.exitCode)"
            )
        }

        // Every run — foreground or background — publishes its result so the library (keyed by
        // request id) records it.
        lastRunResult = result
        runCompletions.send(result)
    }

    private func refreshQueuedRunCount() {
        let count = jobs.queued(in: .inference).count
        if queuedRunCount != count {
            queuedRunCount = count
        }
    }

    /// Terminates every child process the app launched (runs, queued utility commands, and
    /// readiness probes, including a long-lived `api serve`). Called on app termination so
    /// child CLIs are never orphaned.
    func terminateAllProcesses() {
        jobs.terminateAll()
        for process in utilityProcesses.values {
            process.terminate()
        }
        for process in readinessProcesses.values {
            process.terminate()
        }
    }

    private func requiresCameraAccess(_ templateID: CommandTemplateID) -> Bool {
        templateID == .visionTrackLive
    }

    /// Returns `nil` when the run may proceed (no camera needed, or already authorized).
    /// Returns a `Bool` to short-circuit submission when access is pending (`retry` runs once
    /// the user grants it) or denied. The CLI captures the camera as a child of this app bundle,
    /// so TCC attributes access here and the usage string lives in the app's Info.plist.
    private func ensureCameraAccess(
        for templateID: CommandTemplateID,
        retry: @escaping @MainActor (MereRunController) -> Void
    ) -> Bool? {
        guard requiresCameraAccess(templateID) else { return nil }
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            return nil
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                Task { @MainActor in
                    guard let self else { return }
                    if granted {
                        retry(self)
                    } else {
                        self.reportCameraDenied()
                    }
                }
            }
            return false
        default:
            reportCameraDenied()
            return false
        }
    }

    private func reportCameraDenied() {
        status = "Camera access denied"
        append(
            "Camera access is required for live tracking. Enable it in "
                + "System Settings → Privacy & Security → Camera.",
            stream: .stderr
        )
    }

    /// Posts a local notification when a run finishes while the app is backgrounded, so
    /// users can leave a multi-minute pull or render running. No-op for non-bundle (dev) launches.
    private func notifyCompletionIfNeeded(success: Bool, summary: String) {
        // NSApp is nil outside a running NSApplication (e.g. unit tests); guard before use.
        guard Bundle.main.bundleIdentifier != nil, NSApp != nil, !NSApp.isActive else { return }
        UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound]) { granted, _ in
            guard granted else { return }
            let content = UNMutableNotificationContent()
            content.title = success ? "Run complete" : "Run failed"
            content.body = summary
            content.sound = .default
            UNUserNotificationCenter.current().add(
                UNNotificationRequest(identifier: UUID().uuidString, content: content, trigger: nil)
            )
        }
    }

    /// Reads the bundled/resolved CLI version for the app↔CLI version handshake display.
    func refreshCLIVersion() {
        Task { @MainActor in
            let result = await utilityCommandResult(args: ["--version"])
            guard result.exitCode == 0 else { return }
            let version = result.stdout.trimmingCharacters(in: .whitespacesAndNewlines)
            cliVersion = version.isEmpty ? nil : version
        }
    }

    func openLastOutput() {
        guard let lastOutputURL else { return }
        NSWorkspace.shared.open(lastOutputURL)
    }

    func revealLastOutput() {
        guard let lastOutputURL else { return }
        NSWorkspace.shared.activateFileViewerSelecting([lastOutputURL])
    }

    private func synchronizingCLIStatus(from current: CLIInstallationStatus) -> CLIInstallationStatus {
        CLIInstallationStatus(
            phase: .synchronizing,
            kind: current.kind,
            resolvedPath: current.resolvedPath,
            installedVersion: current.installedVersion,
            bundledVersion: current.bundledVersion,
            detail: "Validating and staging the complete CLI payload…",
            lastSynchronizationError: nil,
            allowsManualAction: false
        )
    }

    private func publishCLIInstallationStatus(_ nextStatus: CLIInstallationStatus, automatic: Bool) {
        cliInstallationStatus = nextStatus
        refreshResolvedCLI()
        refreshCLIVersion()

        if let message = nextStatus.lastSynchronizationError {
            if !automatic {
                status = "CLI synchronization needs attention"
            }
            append("Terminal CLI synchronization: \(message)", stream: .stderr)
        } else if !automatic, nextStatus.phase == .upToDate {
            status = "CLI installed"
            if let path = nextStatus.resolvedPath {
                append("Installed Terminal CLI at \(URL(fileURLWithPath: path).abbreviatedForDisplay).", stream: .system)
            }
        }
    }

    private func handleSkillInstall(_ outcome: CodexSkillInstallOutcome) {
        switch outcome {
        case .installed(let names, let destination):
            status = "Skills installed"
            let noun = names.count == 1 ? "skill" : "skills"
            append(
                "Installed Codex \(noun) \(names.joined(separator: ", ")) at \(destination.abbreviatedForDisplay).",
                stream: .system
            )
        case .skippedNoBundledSkills:
            status = "No bundled skills"
            append("Codex skill install skipped: bundled skills were not found.", stream: .stderr)
        case .failed(let message):
            status = "Skill install failed"
            append("Codex skill install failed: \(message)", stream: .stderr)
        }
    }

    private func displayDescription(for launch: MereRunLaunch) -> String {
        let description = launch.sourceDescription
        guard case .executable(let url) = launch,
              CLIResolver.isBundledExecutable(url),
              let installedURL = CLIResolver.existingInstalledCLI() else {
            return description
        }

        return "\(description) + \(installedURL.abbreviatedForDisplay)"
    }

    private func cancelReadinessCheck(for mode: StudioMode) {
        readinessProcesses[mode]?.terminate()
        readinessProcesses[mode] = nil
    }

    private func expandedOptionalPath(_ path: String) -> String? {
        guard !path.isBlank else { return nil }
        return NSString(string: path).expandingTildeInPath
    }

    /// Appends a controller-level message (install, camera, queue notices) directly to the
    /// foreground console; these are not tied to any one run's session.
    private func append(_ text: String, stream: LogStream) {
        for line in text.replacingOccurrences(of: "\r", with: "\n").components(separatedBy: .newlines) {
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }
            logs.append(LogLine(stream: stream, text: trimmed))
        }
        if logs.count > LogRing.defaultCapacity {
            logs.removeFirst(logs.count - LogRing.defaultCapacity)
        }
    }

    /// Output detection for one run; see `ArtifactResolver`. Kept on the controller so callers
    /// and tests have a single entry point that shares the injected file probe.
    func detectOutputURL(expected: URL?, stdout: String) -> URL? {
        ArtifactResolver(fileSystem: fileSystem).primaryOutput(expected: expected, stdout: stdout)
    }

    private func processEnvironment(templateID: CommandTemplateID, draft: CommandDraft) -> [String: String] {
        var env = ProcessInfo.processInfo.environment
        env["PATH"] = [
            "/opt/homebrew/bin",
            "/usr/local/bin",
            "/usr/bin",
            "/bin",
            "/usr/sbin",
            "/sbin",
            env["PATH"] ?? "",
        ].joined(separator: ":")
        if !modelsRoot.isBlank {
            env["MERERUN_MODELS_DIR"] = NSString(string: modelsRoot).expandingTildeInPath
        }
        if !hubCache.isBlank {
            env["MERERUN_HUB_CACHE"] = NSString(string: hubCache).expandingTildeInPath
        }
        for (key, value) in CommandLaunchEnvironment.overrides(templateID: templateID, draft: draft) {
            env[key] = value
        }
        return env
    }

    private func processConfiguration(
        launch: MereRunLaunch,
        args: [String],
        environmentTemplateID: CommandTemplateID? = nil,
        environmentDraft: CommandDraft? = nil,
        environmentOverrides: [String: String] = [:]
    ) -> MereRunProcessConfiguration {
        let processArgs: [String]
        if case .executable(let url) = launch, url.path == "/usr/bin/env" {
            processArgs = ["mere.run"] + args
        } else {
            processArgs = launch.processArguments(for: args)
        }
        let templateID = environmentTemplateID ?? selectedTemplate.id
        let environmentDraft = environmentDraft ?? draft

        var environment = processEnvironment(templateID: templateID, draft: environmentDraft)
        for (key, value) in environmentOverrides {
            environment[key] = value
        }
        return MereRunProcessConfiguration(
            executableURL: launch.executableURL,
            arguments: processArgs,
            currentDirectoryURL: workingDirectoryURL(),
            environment: environment,
            keepsStandardInputOpen: templateID == .videoSession || templateID == .musicRealtime
        )
    }

    private func workingDirectoryURL() -> URL {
        let expanded = NSString(string: workingDirectory).expandingTildeInPath
        var isDirectory: ObjCBool = false
        if FileManager.default.fileExists(atPath: expanded, isDirectory: &isDirectory), isDirectory.boolValue {
            return URL(fileURLWithPath: expanded, isDirectory: true)
        }
        return FileManager.default.homeDirectoryForCurrentUser
    }
}

private extension URL {
    var abbreviatedForDisplay: String {
        let homePath = FileManager.default.homeDirectoryForCurrentUser.path
        if path == homePath {
            return "~"
        }
        if path.hasPrefix(homePath + "/") {
            return "~" + path.dropFirst(homePath.count)
        }
        return path
    }
}

extension Array where Element == String {
    func maskingSecrets() -> [String] {
        var masked = self
        var index = masked.startIndex
        while index < masked.endIndex {
            if masked[index] == "--api-key"
                || masked[index] == "--infinity-api-key"
                || masked[index] == "hf-token" {
                let valueIndex = masked.index(after: index)
                if valueIndex < masked.endIndex {
                    masked[valueIndex] = "••••••••"
                }
            }
            index = masked.index(after: index)
        }
        return masked
    }

    func shellQuoted() -> String {
        map { arg in
            if arg.isEmpty { return "''" }
            let safe = CharacterSet(charactersIn: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_+-=.,/:@%")
            if arg.unicodeScalars.allSatisfy({ safe.contains($0) }) {
                return arg
            }
            return "'" + arg.replacingOccurrences(of: "'", with: "'\\''") + "'"
        }
        .joined(separator: " ")
    }
}
