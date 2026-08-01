import AppKit
import AVFoundation
import Combine
import Foundation
import UserNotifications

/// Decodes a byte stream into UTF-8 incrementally, retaining any incomplete trailing
/// multibyte sequence until the next read so codepoints split across pipe reads are
/// never dropped. Each stream owns its own decoder; the readability queue is serial
/// per file handle, so no locking is required within a single stream.
private final class IncrementalUTF8Decoder: @unchecked Sendable {
    private var buffer = Data()
    private static let lossyFlushThreshold = 1 << 20

    func push(_ data: Data) -> String? {
        buffer.append(data)
        guard !buffer.isEmpty else { return nil }

        // A well-formed UTF-8 stream only fails to decode when the final codepoint is
        // truncated (at most 3 trailing bytes). Trim up to 3 bytes to find the boundary.
        let maxBackoff = min(3, buffer.count)
        for back in 0...maxBackoff {
            let length = buffer.count - back
            guard length > 0 else { break }
            if let decoded = String(data: buffer.prefix(length), encoding: .utf8) {
                buffer.removeFirst(length)
                return decoded.isEmpty ? nil : decoded
            }
        }

        // Genuinely malformed mid-stream bytes (should not happen from the CLI): avoid
        // unbounded buffering by flushing lossily once the backlog grows too large.
        if buffer.count > Self.lossyFlushThreshold {
            return flush()
        }
        return nil
    }

    func flush() -> String? {
        guard !buffer.isEmpty else { return nil }
        let decoded = String(decoding: buffer, as: UTF8.self)
        buffer.removeAll(keepingCapacity: false)
        return decoded.isEmpty ? nil : decoded
    }
}

enum LogStream {
    case system
    case stdout
    case stderr

    var label: String {
        switch self {
        case .system: return "mere"
        case .stdout: return "out"
        case .stderr: return "err"
        }
    }
}

struct LogLine: Identifiable, Equatable {
    let id = UUID()
    let date = Date()
    let stream: LogStream
    let text: String
}

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

    private static func installedCandidates(fileManager fm: FileManager) -> [URL] {
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

struct MereRunProcessConfiguration: Equatable {
    let executableURL: URL
    let arguments: [String]
    let currentDirectoryURL: URL
    let environment: [String: String]
    let keepsStandardInputOpen: Bool
}

enum MereRunProcessInputError: LocalizedError {
    case unavailable

    var errorDescription: String? {
        "This command does not accept interactive input."
    }
}

protocol MereRunRunningProcess: AnyObject {
    func terminate()
    func sendStandardInput(_ text: String) throws
}

extension MereRunRunningProcess {
    func sendStandardInput(_ text: String) throws {
        throw MereRunProcessInputError.unavailable
    }
}

protocol MereRunProcessRunning: AnyObject {
    func start(
        configuration: MereRunProcessConfiguration,
        stdout: @escaping @Sendable (String) -> Void,
        stderr: @escaping @Sendable (String) -> Void,
        termination: @escaping @Sendable (Int32) -> Void
    ) throws -> MereRunRunningProcess
}

/// A seam over filesystem existence checks so run-output detection can be unit-tested without
/// touching the real disk. `FileManager` is the production implementation.
protocol MereRunFileProbing {
    func fileExists(atPath path: String) -> Bool
}

extension FileManager: MereRunFileProbing {}

private final class FoundationRunningProcess: MereRunRunningProcess, @unchecked Sendable {
    private let process: Process
    private let stdinPipe: Pipe?
    private let stdoutPipe: Pipe
    private let stderrPipe: Pipe

    init(process: Process, stdinPipe: Pipe?, stdoutPipe: Pipe, stderrPipe: Pipe) {
        self.process = process
        self.stdinPipe = stdinPipe
        self.stdoutPipe = stdoutPipe
        self.stderrPipe = stderrPipe
    }

    func terminate() {
        try? stdinPipe?.fileHandleForWriting.close()
        process.terminate()
    }

    func sendStandardInput(_ text: String) throws {
        guard let stdinPipe else {
            throw MereRunProcessInputError.unavailable
        }
        try stdinPipe.fileHandleForWriting.write(contentsOf: Data(text.utf8))
    }

    func cleanup() {
        try? stdinPipe?.fileHandleForWriting.close()
        stdoutPipe.fileHandleForReading.readabilityHandler = nil
        stderrPipe.fileHandleForReading.readabilityHandler = nil
    }

    func waitUntilExit() -> Int32 {
        process.waitUntilExit()
        cleanup()
        return process.terminationStatus
    }
}

private final class FoundationMereRunProcessRunner: MereRunProcessRunning {
    func start(
        configuration: MereRunProcessConfiguration,
        stdout: @escaping @Sendable (String) -> Void,
        stderr: @escaping @Sendable (String) -> Void,
        termination: @escaping @Sendable (Int32) -> Void
    ) throws -> MereRunRunningProcess {
        let process = Process()
        process.executableURL = configuration.executableURL
        process.arguments = configuration.arguments
        process.currentDirectoryURL = configuration.currentDirectoryURL
        process.environment = configuration.environment

        let stdinPipe = configuration.keepsStandardInputOpen ? Pipe() : nil
        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        process.standardInput = stdinPipe
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe

        let runningProcess = FoundationRunningProcess(
            process: process,
            stdinPipe: stdinPipe,
            stdoutPipe: stdoutPipe,
            stderrPipe: stderrPipe
        )

        let stdoutDecoder = IncrementalUTF8Decoder()
        let stderrDecoder = IncrementalUTF8Decoder()

        stdoutPipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            if data.isEmpty {
                if let tail = stdoutDecoder.flush() { stdout(tail) }
                return
            }
            if let text = stdoutDecoder.push(data) { stdout(text) }
        }

        stderrPipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            if data.isEmpty {
                if let tail = stderrDecoder.flush() { stderr(tail) }
                return
            }
            if let text = stderrDecoder.push(data) { stderr(text) }
        }

        try process.run()

        DispatchQueue.global(qos: .userInitiated).async {
            termination(runningProcess.waitUntilExit())
        }

        return runningProcess
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

struct MereRunRunResult: Identifiable, Equatable {
    let id: UUID
    let requestID: UUID?
    let templateID: CommandTemplateID
    let commandPreview: String
    let exitCode: Int32
    let outputURL: URL?
    let artifactURLs: [URL]
    let outputText: String?
    let completedAt: Date
    /// When this run was a chat/code turn, the conversation it belongs to (so completion routes
    /// to the thread rather than the legacy single-result path).
    let conversationID: UUID?

    init(
        id: UUID = UUID(),
        requestID: UUID? = nil,
        templateID: CommandTemplateID,
        commandPreview: String,
        exitCode: Int32,
        outputURL: URL?,
        artifactURLs: [URL] = [],
        outputText: String?,
        completedAt: Date = Date(),
        conversationID: UUID? = nil
    ) {
        self.id = id
        self.requestID = requestID
        self.templateID = templateID
        self.commandPreview = commandPreview
        self.exitCode = exitCode
        self.outputURL = outputURL
        self.artifactURLs = artifactURLs
        self.outputText = outputText
        self.completedAt = completedAt
        self.conversationID = conversationID
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

private struct StudioVideoSessionRequest: Encodable {
    let id: String
    let prompt: String
    let output: String
    let width: Int
    let height: Int
    let numFrames: Int
    let fps: Int
    let seed: Int?
    let image: String?
    let imageStrength: Double?
    let endImage: String?
    let endImageStrength: Double?
}

private struct StudioVideoSessionResponse: Decodable {
    let status: String
    let output: String?
    let error: String?
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
    private var utilityProcesses: [UUID: MereRunRunningProcess] = [:]
    private var readinessProcesses: [StudioMode: MereRunRunningProcess] = [:]
    private var readinessRequests: [StudioMode: ReadinessRequest] = [:]
    /// Runs currently in flight, each with its own isolated process, buffers, logs, progress
    /// and output. Up to `maxConcurrentRuns` run at once; the rest wait in `queuedRuns`.
    private var sessions: [RunSession] = []
    /// The session whose live state mirrors into the published console fields (the run the
    /// single-pane console/canvas currently shows). Background sessions still complete into the
    /// library by request id. Persists past completion so the last run's result stays visible.
    private var foregroundSessionID: UUID?
    private var queuedRuns: [StudioRunRequest] = []

    private var foregroundSession: RunSession? {
        sessions.first { $0.id == foregroundSessionID }
    }

    private static let stdoutBufferByteLimit = 32 * 1024
    private static let stderrBufferByteLimit = 32 * 1024
    private static let outputDetectionLineLimit = 40
    private static let logLineLimit = 1200
    /// Min growth in a conversation reply before the live think-stripped text is re-published,
    /// bounding per-chunk re-strip cost. The final reply is always published in full on finish.
    private static let liveStripGranularity = 80
    /// Conservative cap on simultaneous runs. ML inference is memory-heavy, so this stays small;
    /// it is the single knob for how many runs may execute at once before further ones queue.
    static let maxConcurrentRuns = 2

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
        guard let session = foregroundSession else { return false }
        return session.spec.template.id == .videoSession
            && session.process != nil
            && session.exitCode == nil
    }

    func canSteerRealtimeMusic(requestID: UUID) -> Bool {
        sessions.contains {
            $0.spec.requestID == requestID
                && $0.spec.template.id == .musicRealtime
                && $0.process != nil
                && $0.exitCode == nil
        }
    }

    func isRequestRunning(_ requestID: UUID) -> Bool {
        sessions.contains {
            $0.spec.requestID == requestID
                && $0.process != nil
                && $0.exitCode == nil
        }
    }

    func runningRequestID(for templateID: CommandTemplateID) -> UUID? {
        sessions.first {
            $0.spec.template.id == templateID
                && $0.spec.requestID != nil
                && $0.process != nil
                && $0.exitCode == nil
        }?.spec.requestID
    }

    func logs(for requestID: UUID) -> [LogLine] {
        sessions.first { $0.spec.requestID == requestID }?.logs ?? []
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
        let outcome = CLIBootstrapInstaller.installBundledCLIIfNeeded()
        handleCLIInstall(outcome)
        refreshResolvedCLI()
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
                        if let onOutput {
                            Task { @MainActor in onOutput(text) }
                        }
                    },
                    stderr: { text in
                        errors.append(text)
                        if let onOutput {
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

    /// A captured snapshot of what to run, decoupled from the live editing state
    /// (`selectedTemplate`/`draft`). A queued or Studio-initiated run executes from its own
    /// snapshot, so it never overwrites whatever the user is currently editing in either surface.
    private struct RunSpec {
        let template: CommandTemplate
        let draft: CommandDraft
        let requestID: UUID?
        var conversationID: UUID? = nil
    }

    /// All mutable state for a single in-flight run, isolated so concurrent runs never clobber
    /// each other's process, buffers, logs, progress or output. `@MainActor` (so it is Sendable
    /// and can be captured by the process callbacks); every field is touched only on the main actor.
    @MainActor
    private final class RunSession: Identifiable {
        let id = UUID()
        let spec: RunSpec
        let preview: String
        let expectedOutput: URL?
        var process: MereRunRunningProcess?
        var stdoutBuffer = ""
        var stderrBuffer = ""
        /// Unbounded stdout accumulator for conversation turns only (chat/code), so a long reply
        /// is captured in full — `stdoutBuffer` is capped at 32 KB for the console.
        var fullOutput = ""
        /// Length of `fullOutput` at the last live think-strip; used to throttle re-stripping the
        /// whole accumulator on every chunk (it would otherwise be O(n²) over a long reply).
        var lastLiveStripLength = 0
        var logs: [LogLine] = []
        var liveOutputText = ""
        var currentProgress: StudioRunProgress?
        var lastOutputURL: URL?
        var exitCode: Int32?
        var status: String
        var interactiveOutputBuffer = ""
        var outputWatchTask: Task<Void, Never>?

        init(spec: RunSpec, preview: String, expectedOutput: URL?, status: String) {
            self.spec = spec
            self.preview = preview
            self.expectedOutput = expectedOutput
            self.status = status
        }
    }

    @discardableResult
    func run(studio request: StudioRunRequest) -> Bool {
        // Track the conversation as in-flight at SUBMISSION time, not at start, so a turn that
        // queues behind the concurrency cap still blocks a second send into the same thread and
        // shows a pending bubble. Idempotent with the insert in startRun; cleared on every exit.
        if let conversationID = request.conversationID {
            runningConversationIDs.insert(conversationID)
        }
        guard sessions.count < Self.maxConcurrentRuns, queuedRuns.isEmpty else {
            enqueue(request)
            return true
        }

        return startRun(RunSpec(
            template: request.template,
            draft: request.draft,
            requestID: request.id,
            conversationID: request.conversationID
        ))
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
        guard sessions.count < Self.maxConcurrentRuns else { return false }
        return startRun(RunSpec(template: selectedTemplate, draft: draft, requestID: nil))
    }

    @discardableResult
    private func startRun(_ spec: RunSpec) -> Bool {
        guard sessions.count < Self.maxConcurrentRuns else { return false }
        if let shortCircuit = ensureCameraAccess(spec: spec) {
            return shortCircuit
        }
        refreshResolvedCLI()

        let launch = cliResolve(cliPath)
        let args = commandArguments(template: spec.template, draft: spec.draft)
        let display = launch.displayCommand(for: args)
        let expectedOutput = expectedOutputURL(template: spec.template, draft: spec.draft)
        let initialStatus = spec.template.id == .modelPull ? "Downloading model" : "Running"
        let session = RunSession(spec: spec, preview: display, expectedOutput: expectedOutput, status: initialStatus)

        if let message = spec.template.validationMessage(for: spec.draft) {
            append(message, stream: .system)
            status = message
            finishPreflightFailure(session: session, exitCode: 64, outputText: message)
            return false
        }

        if let prepError = prepareOutputLocation(template: spec.template, draft: spec.draft) {
            append(prepError, stream: .stderr)
            status = "Output path unavailable"
            finishPreflightFailure(session: session, exitCode: -1, outputText: prepError)
            return false
        }

        sessions.append(session)
        if let conversationID = spec.conversationID {
            runningConversationIDs.insert(conversationID)
        }
        setForeground(session)
        append(display, stream: .system, to: session)
        if spec.conversationID == nil {
            startOutputWatch(for: session)
        }

        do {
            session.process = try processRunner.start(
                configuration: processConfiguration(
                    launch: launch,
                    args: args,
                    environmentTemplateID: spec.template.id,
                    environmentDraft: spec.draft
                ),
                stdout: { [weak self, weak session] text in
                    Task { @MainActor in
                        guard let self, let session else { return }
                        self.append(text, stream: .stdout, to: session)
                    }
                },
                stderr: { [weak self, weak session] text in
                    Task { @MainActor in
                        guard let self, let session else { return }
                        self.append(text, stream: .stderr, to: session)
                    }
                },
                termination: { [weak self, weak session] code in
                    Task { @MainActor in
                        guard let self, let session else { return }
                        self.finishRun(session: session, exitCode: code)
                    }
                }
            )
        } catch {
            append(error.localizedDescription, stream: .stderr, to: session)
            finishRun(session: session, exitCode: -1)
            return false
        }
        return true
    }

    func cancel() {
        guard let session = foregroundSession, session.process != nil else { return }
        session.process?.terminate()
        append("Termination requested.", stream: .system, to: session)
    }

    @discardableResult
    func cancel(requestID: UUID) -> Bool {
        guard let session = sessions.first(where: {
            $0.spec.requestID == requestID && $0.process != nil && $0.exitCode == nil
        }) else {
            return false
        }
        session.process?.terminate()
        append("Termination requested.", stream: .system, to: session)
        mirrorForeground(session)
        return true
    }

    @discardableResult
    func submitRealtimeMusicCommand(_ line: String, requestID: UUID) -> Bool {
        guard let session = sessions.first(where: {
            $0.spec.requestID == requestID && $0.spec.template.id == .musicRealtime
        }),
        let process = session.process,
        session.exitCode == nil else {
            return false
        }
        do {
            try process.sendStandardInput(line + "\n")
            append("Live control → \(line)", stream: .system, to: session)
            mirrorForeground(session)
            return true
        } catch {
            append(error.localizedDescription, stream: .stderr, to: session)
            mirrorForeground(session)
            return false
        }
    }

    @discardableResult
    func submitVideoSessionRequest() -> Bool {
        guard let session = foregroundSession,
              session.spec.template.id == .videoSession,
              let process = session.process,
              session.exitCode == nil else {
            status = "Start the resident session first"
            append("Start the resident LTX session before submitting a render.", stream: .stderr)
            return false
        }

        let prompt = draft.prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty else {
            status = "Prompt required"
            append("Enter a prompt for the resident render.", stream: .stderr, to: session)
            return false
        }
        guard !draft.outputPath.isBlank else {
            status = "Output required"
            append("Choose an output MP4 path for the resident render.", stream: .stderr, to: session)
            return false
        }

        let output = NSString(string: draft.outputPath).expandingTildeInPath
        do {
            try FileManager.default.createDirectory(
                at: URL(fileURLWithPath: output).deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            let request = StudioVideoSessionRequest(
                id: UUID().uuidString,
                prompt: prompt,
                output: output,
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
            let encoder = JSONEncoder()
            encoder.keyEncodingStrategy = .convertToSnakeCase
            encoder.outputFormatting = [.sortedKeys]
            let data = try encoder.encode(request)
            guard let line = String(data: data, encoding: .utf8) else {
                throw CocoaError(.fileWriteInapplicableStringEncoding)
            }
            try process.sendStandardInput(line + "\n")
            session.status = "Rendering resident request"
            session.lastOutputURL = nil
            append("Submitted resident render → \(output)", stream: .system, to: session)
            mirrorForeground(session)
            return true
        } catch {
            session.status = "Session submission failed"
            append(error.localizedDescription, stream: .stderr, to: session)
            mirrorForeground(session)
            return false
        }
    }

    /// Resets the published console fields to reflect `session` as the run the single-pane
    /// console/canvas shows. Background sessions keep their own state and complete into the library.
    private func setForeground(_ session: RunSession) {
        foregroundSessionID = session.id
        logs = session.logs
        liveOutputText = session.liveOutputText
        currentProgress = session.currentProgress
        status = session.status
        activeRunRequestID = session.spec.requestID
        lastOutputURL = session.lastOutputURL
        lastExitCode = session.exitCode
        isRunning = session.exitCode == nil
    }

    /// Mirrors `session`'s live state into the published console fields while it is the foreground.
    private func mirrorForeground(_ session: RunSession) {
        guard session.id == foregroundSessionID else { return }
        logs = session.logs
        liveOutputText = session.liveOutputText
        currentProgress = session.currentProgress
        status = session.status
        if let url = session.lastOutputURL { lastOutputURL = url }
    }

    /// Terminates every child process the app launched (active run, queued utility, and
    /// readiness probes, including a long-lived `api serve`). Called on app termination so
    /// child CLIs are never orphaned.
    func terminateAllProcesses() {
        for session in sessions {
            session.process?.terminate()
        }
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
    /// Returns a `Bool` to short-circuit `startRun` when access is pending (async request
    /// will retry) or denied. The CLI captures the camera as a child of this app bundle,
    /// so TCC attributes access here and the usage string lives in the app's Info.plist.
    private func ensureCameraAccess(spec: RunSpec) -> Bool? {
        guard requiresCameraAccess(spec.template.id) else { return nil }
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            return nil
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                Task { @MainActor in
                    guard let self else { return }
                    if granted {
                        _ = self.startRun(spec)
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

    private func finishRun(session: RunSession, exitCode: Int32) {
        session.outputWatchTask?.cancel()
        session.outputWatchTask = nil
        session.process = nil
        session.exitCode = exitCode
        session.currentProgress = nil
        if let requestID = session.spec.requestID {
            progressByRequestID[requestID] = nil
        }

        // Conversation replies are prose, not artifacts — never run output-file detection on them
        // (a path-like substring in a reply must not become a bogus lastOutputURL/status).
        let detectedOutput = session.spec.conversationID == nil
            ? detectOutputURL(expected: session.expectedOutput, stdout: session.stdoutBuffer)
            : nil
        let reportedOutputs = session.spec.conversationID == nil
            ? detectedOutputURLs(stdout: session.stdoutBuffer)
            : []
        let artifactURLs = StudioArtifactDiscovery.urls(
            templateID: session.spec.template.id,
            draft: session.spec.draft,
            primaryOutput: detectedOutput,
            reportedOutputs: reportedOutputs
        )
        // Conversation turns finalize from the unbounded, think-stripped accumulator so long
        // replies are not clipped by the 32 KB console buffer; other modes use the buffer.
        let outputText: String?
        if session.spec.conversationID != nil {
            // Strip reasoning for conversation turns on EVERY exit path (not just success) so
            // <think> blocks and STDERR never leak into the next turn's replayed prompt; use the
            // full accumulator, not the capped console buffer.
            outputText = ConversationTranscript.stripThinkTags(session.fullOutput.replacingOccurrences(of: "\0", with: ""))
        } else {
            outputText = capturedResultText(for: session, exitCode: exitCode)
        }
        if let detectedOutput { session.lastOutputURL = detectedOutput }
        if let conversationID = session.spec.conversationID {
            conversationLiveText[conversationID] = nil
        }

        if exitCode == 0 {
            session.status = detectedOutput == nil ? "Completed" : "Completed: \(detectedOutput!.lastPathComponent)"
            session.logs.append(LogLine(stream: .system, text: "Completed with exit code 0."))
        } else {
            session.status = "Exited \(exitCode)"
            session.logs.append(LogLine(stream: .system, text: "Exited with code \(exitCode)."))
        }

        notifyCompletionIfNeeded(
            success: exitCode == 0,
            summary: exitCode == 0
                ? (detectedOutput?.lastPathComponent ?? "Completed")
                : "Exited with code \(exitCode)"
        )

        // Every run — foreground or background — publishes its result so the library (keyed by
        // request id) records it. Only the foreground updates the shared console fields.
        if session.id == foregroundSessionID {
            logs = session.logs
            liveOutputText = session.liveOutputText
            currentProgress = nil
            status = session.status
            lastExitCode = exitCode
            if let detectedOutput { lastOutputURL = detectedOutput }
            isRunning = false
            activeRunRequestID = nil
        }

        if let conversationID = session.spec.conversationID {
            runningConversationIDs.remove(conversationID)
        }

        let result = MereRunRunResult(
            requestID: session.spec.requestID,
            templateID: session.spec.template.id,
            commandPreview: session.preview,
            exitCode: exitCode,
            outputURL: detectedOutput,
            artifactURLs: artifactURLs,
            outputText: outputText,
            conversationID: session.spec.conversationID
        )
        lastRunResult = result
        runCompletions.send(result)

        sessions.removeAll { $0.id == session.id }
        startNextQueuedRun()
    }

    private func finishPreflightFailure(session: RunSession, exitCode: Int32, outputText: String?) {
        lastExitCode = exitCode
        if let requestID = session.spec.requestID {
            progressByRequestID[requestID] = nil
        }
        if let conversationID = session.spec.conversationID {
            runningConversationIDs.remove(conversationID)
        }
        let result = MereRunRunResult(
            requestID: session.spec.requestID,
            templateID: session.spec.template.id,
            commandPreview: session.preview,
            exitCode: exitCode,
            outputURL: nil,
            artifactURLs: [],
            outputText: outputText,
            conversationID: session.spec.conversationID
        )
        lastRunResult = result
        runCompletions.send(result)
        startNextQueuedRun()
    }

    private func enqueue(_ request: StudioRunRequest) {
        queuedRuns.append(request)
        queuedRunCount = queuedRuns.count
        append("Queued \(request.mode.title.lowercased()) job.", stream: .system)
    }

    private func startNextQueuedRun() {
        guard sessions.count < Self.maxConcurrentRuns, !queuedRuns.isEmpty else { return }
        let next = queuedRuns.removeFirst()
        queuedRunCount = queuedRuns.count
        _ = startRun(RunSpec(
            template: next.template,
            draft: next.draft,
            requestID: next.id,
            conversationID: next.conversationID
        ))
    }

    private func startOutputWatch(for session: RunSession) {
        session.outputWatchTask = Task { [weak self, weak session] in
            while !Task.isCancelled {
                await MainActor.run {
                    guard let self, let session else { return }
                    self.publishDetectedOutputIfNeeded(for: session)
                }
                try? await Task.sleep(for: .milliseconds(350))
            }
        }
    }

    private func publishDetectedOutputIfNeeded(for session: RunSession) {
        guard session.exitCode == nil else { return }
        let detectedOutput = detectOutputURL(expected: session.expectedOutput, stdout: session.stdoutBuffer)
        guard let detectedOutput, detectedOutput != session.lastOutputURL else { return }
        session.lastOutputURL = detectedOutput
        session.status = "Generated: \(detectedOutput.lastPathComponent)"
        if session.id == foregroundSessionID {
            lastOutputURL = detectedOutput
            status = session.status
        }
    }

    private func handleCLIInstall(_ outcome: CLIBootstrapInstallOutcome) {
        switch outcome {
        case .installed(let url):
            status = "CLI installed"
            append("Installed Terminal CLI at \(url.abbreviatedForDisplay).", stream: .system)
        case .failed(let message):
            status = "CLI install failed"
            append("Terminal CLI install failed: \(message)", stream: .stderr)
        case .alreadyInstalled(let url):
            status = "CLI already installed"
            append("Terminal CLI already installed at \(url.abbreviatedForDisplay).", stream: .system)
        case .skippedNoBundledCLI:
            status = "No bundled CLI"
            append("Terminal CLI install skipped: bundled CLI payload was not found.", stream: .stderr)
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

    /// Appends a run's output to its session, mirroring into the published console fields when
    /// that session is the foreground one.
    private func append(_ text: String, stream: LogStream, to session: RunSession) {
        if stream == .stdout {
            if session.spec.template.id == .videoSession {
                consumeVideoSessionOutput(text, session: session)
            }
            session.stdoutBuffer += text
            session.stdoutBuffer = Self.trimmed(session.stdoutBuffer, toByteLimit: Self.stdoutBufferByteLimit)
            session.liveOutputText = session.stdoutBuffer.replacingOccurrences(of: "\0", with: "")
            if let conversationID = session.spec.conversationID {
                // Conversation turns accumulate the full (untrimmed) reply and publish a live,
                // think-stripped view so a streaming bubble renders even when backgrounded. The
                // streaming variant hides an in-progress (unclosed) reasoning block. Re-stripping
                // the whole accumulator on every chunk is O(n²), so throttle to ~every 80 chars of
                // growth; finishRun always publishes the complete, fully-stripped reply.
                session.fullOutput += text
                // Publish immediately on the first chunk (fast first render), then throttle.
                if session.lastLiveStripLength == 0
                    || session.fullOutput.count - session.lastLiveStripLength >= Self.liveStripGranularity {
                    session.lastLiveStripLength = session.fullOutput.count
                    conversationLiveText[conversationID] = ConversationTranscript.stripThinkTags(
                        session.fullOutput.replacingOccurrences(of: "\0", with: ""),
                        streaming: true
                    )
                }
            } else {
                publishDetectedOutputIfNeeded(for: session)
            }
        } else if stream == .stderr {
            session.stderrBuffer += text
            session.stderrBuffer = Self.trimmed(session.stderrBuffer, toByteLimit: Self.stderrBufferByteLimit)
            if session.spec.template.id == .videoSession,
               text.localizedCaseInsensitiveContains("session ready") {
                session.status = "Resident session ready"
            }
        }

        let normalized = text
            .replacingOccurrences(of: "\r", with: "\n")
            .components(separatedBy: .newlines)

        for line in normalized {
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }
            // Collapse repeated carriage-return progress updates into one structured value
            // instead of flooding the log with hundreds of lines.
            if let progress = StudioProgressParser.parse(trimmed) {
                session.currentProgress = progress
                if let requestID = session.spec.requestID {
                    progressByRequestID[requestID] = progress
                }
                continue
            }
            session.logs.append(LogLine(stream: stream, text: trimmed))
        }

        if session.logs.count > Self.logLineLimit {
            session.logs.removeFirst(session.logs.count - Self.logLineLimit)
        }
        mirrorForeground(session)
    }

    private func consumeVideoSessionOutput(_ text: String, session: RunSession) {
        session.interactiveOutputBuffer += text
        let lines = session.interactiveOutputBuffer.components(separatedBy: .newlines)
        session.interactiveOutputBuffer = lines.last ?? ""

        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        for line in lines.dropLast() {
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty,
                  let data = trimmed.data(using: .utf8),
                  let response = try? decoder.decode(StudioVideoSessionResponse.self, from: data) else {
                continue
            }

            if response.status == "result", let output = response.output {
                let url = URL(fileURLWithPath: output)
                session.lastOutputURL = url
                session.status = "Generated: \(url.lastPathComponent)"
                if session.id == foregroundSessionID {
                    lastOutputURL = url
                }
            } else if response.status == "error" {
                session.status = "Resident render failed"
                if let error = response.error, !error.isBlank {
                    session.logs.append(LogLine(stream: .stderr, text: error))
                }
            }
        }
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
        if logs.count > Self.logLineLimit {
            logs.removeFirst(logs.count - Self.logLineLimit)
        }
    }

    private static func trimmed(_ buffer: String, toByteLimit limit: Int) -> String {
        guard buffer.utf8.count > limit else { return buffer }
        return String(decoding: buffer.utf8.suffix(limit), as: UTF8.self)
    }

    private func nonEmptyTrimmed(_ buffer: String) -> String? {
        let trimmed = buffer
            .replacingOccurrences(of: "\0", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    private func capturedResultText(for session: RunSession, exitCode: Int32) -> String? {
        let stdout = nonEmptyTrimmed(session.stdoutBuffer)
        guard exitCode != 0, let stderr = nonEmptyTrimmed(session.stderrBuffer) else {
            return stdout
        }

        if let stdout {
            return "\(stdout)\n\nSTDERR\n\(stderr)"
        }
        return stderr
    }

    private func expectedOutputURL(template: CommandTemplate, draft: CommandDraft) -> URL? {
        guard template.outputKind != .none, !draft.outputPath.isBlank else {
            return nil
        }
        return URL(fileURLWithPath: NSString(string: draft.outputPath).expandingTildeInPath)
    }

    /// Ensures the run's output directory exists. Returns `nil` on success, or a diagnostic
    /// message when the directory could not be created (so the caller can surface it per run).
    private func prepareOutputLocation(template: CommandTemplate, draft: CommandDraft) -> String? {
        guard !draft.outputPath.isBlank else {
            return nil
        }

        let outputURL = URL(fileURLWithPath: NSString(string: draft.outputPath).expandingTildeInPath)
        let directoryURL: URL
        switch template.outputKind {
        case .file:
            directoryURL = outputURL.deletingLastPathComponent()
        case .directory:
            directoryURL = outputURL
        case .none:
            return nil
        }

        do {
            try FileManager.default.createDirectory(at: directoryURL, withIntermediateDirectories: true)
            return nil
        } catch {
            return "Could not create output directory \(directoryURL.path): \(error.localizedDescription)"
        }
    }

    func detectOutputURL(expected: URL?, stdout: String) -> URL? {
        let fm = fileSystem

        // 1. The explicit `--output` path the request asked for, once it has landed.
        if let expected, fm.fileExists(atPath: expected.path) {
            return expected
        }

        // 2. Honor the CLI's stdout contract: media commands print the artifact path as a bare
        //    line and directory/OCR commands as `input -> output` pairs, most-recent first.
        //    This is the only path that detects `input -> output` outputs at all (a whole pair
        //    line never resolves as a file), and it targets the result line rather than guessing.
        for candidate in StudioResultParser.outputPaths(fromStdout: stdout) {
            let expanded = NSString(string: candidate).expandingTildeInPath
            if fm.fileExists(atPath: expanded) {
                return URL(fileURLWithPath: expanded)
            }
        }

        // 3. Fallback for commands without a clean path contract: the last trailing stdout
        //    line that happens to resolve to an existing file (e.g. a relative path).
        let candidates = stdout
            .components(separatedBy: .newlines)
            .compactMap { line -> String? in
                let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
                return trimmed.isEmpty ? nil : trimmed
            }
            .suffix(Self.outputDetectionLineLimit)
            .reversed()

        for candidate in candidates {
            let expanded = NSString(string: candidate).expandingTildeInPath
            if fm.fileExists(atPath: expanded) {
                return URL(fileURLWithPath: expanded)
            }
        }

        return nil
    }

    private func detectedOutputURLs(stdout: String) -> [URL] {
        StudioResultParser.outputPaths(fromStdout: stdout).compactMap { candidate in
            let expanded = NSString(string: candidate).expandingTildeInPath
            guard fileSystem.fileExists(atPath: expanded) else { return nil }
            return URL(fileURLWithPath: expanded)
        }
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
