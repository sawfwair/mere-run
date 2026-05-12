import AppKit
import Foundation

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
            if Bundle.main.resourceURL.map({ url.path.hasPrefix($0.path) }) == true {
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
        return [
            resourceURL?.appendingPathComponent("mere.run/mere.run"),
            resourceURL?.appendingPathComponent("mere.run"),
        ].compactMap { $0 }
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
}

protocol MereRunRunningProcess: AnyObject {
    func terminate()
}

protocol MereRunProcessRunning: AnyObject {
    func start(
        configuration: MereRunProcessConfiguration,
        stdout: @escaping @Sendable (String) -> Void,
        stderr: @escaping @Sendable (String) -> Void,
        termination: @escaping @Sendable (Int32) -> Void
    ) throws -> MereRunRunningProcess
}

private final class FoundationRunningProcess: MereRunRunningProcess, @unchecked Sendable {
    private let process: Process
    private let stdoutPipe: Pipe
    private let stderrPipe: Pipe

    init(process: Process, stdoutPipe: Pipe, stderrPipe: Pipe) {
        self.process = process
        self.stdoutPipe = stdoutPipe
        self.stderrPipe = stderrPipe
    }

    func terminate() {
        process.terminate()
    }

    func cleanup() {
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

        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe

        let runningProcess = FoundationRunningProcess(
            process: process,
            stdoutPipe: stdoutPipe,
            stderrPipe: stderrPipe
        )

        stdoutPipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else { return }
            stdout(text)
        }

        stderrPipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else { return }
            stderr(text)
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
    let outputText: String?
    let completedAt: Date

    init(
        id: UUID = UUID(),
        requestID: UUID? = nil,
        templateID: CommandTemplateID,
        commandPreview: String,
        exitCode: Int32,
        outputURL: URL?,
        outputText: String?,
        completedAt: Date = Date()
    ) {
        self.id = id
        self.requestID = requestID
        self.templateID = templateID
        self.commandPreview = commandPreview
        self.exitCode = exitCode
        self.outputURL = outputURL
        self.outputText = outputText
        self.completedAt = completedAt
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
    @Published private(set) var activeRunRequestID: UUID?
    @Published private(set) var queuedRunCount = 0
    @Published var readinessByMode: [StudioMode: ModelReadinessState] = [:]
    @Published var modelCapabilitiesByID: [String: StudioModelCapability] = [:]
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
    @Published private(set) var liveOutputText = ""

    private enum Keys {
        static let cliPath = "mererun.app.cliPath"
        static let modelsRoot = "mererun.app.modelsRoot"
        static let hubCache = "mererun.app.hubCache"
        static let workingDirectory = "mererun.app.workingDirectory"
    }

    private let processRunner: MereRunProcessRunning
    private var currentProcess: MereRunRunningProcess?
    private var utilityProcesses: [UUID: MereRunRunningProcess] = [:]
    private var readinessProcesses: [StudioMode: MereRunRunningProcess] = [:]
    private var readinessRequests: [StudioMode: ReadinessRequest] = [:]
    private var stdoutBuffer = ""
    private var stderrBuffer = ""
    private var activeRunTemplateID: CommandTemplateID?
    private var activeRunPreview = ""
    private var outputWatchTask: Task<Void, Never>?
    private var queuedRuns: [StudioRunRequest] = []
    private var didAttemptAutomaticCLIInstall = false

    private static let stdoutBufferByteLimit = 32 * 1024
    private static let stderrBufferByteLimit = 32 * 1024
    private static let outputDetectionLineLimit = 40

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

    init(
        processRunner: MereRunProcessRunning = FoundationMereRunProcessRunner(),
        resolvesCLIOnInit: Bool = true
    ) {
        self.processRunner = processRunner
        let initial = CommandCatalog.templates.first!
        selectedTemplate = initial
        draft = initial.defaultDraft()
        cliPath = UserDefaults.standard.string(forKey: Keys.cliPath) ?? ""
        modelsRoot = UserDefaults.standard.string(forKey: Keys.modelsRoot) ?? ""
        hubCache = UserDefaults.standard.string(forKey: Keys.hubCache) ?? ""
        workingDirectory = UserDefaults.standard.string(forKey: Keys.workingDirectory)
            ?? FileManager.default.homeDirectoryForCurrentUser.path
        if resolvesCLIOnInit {
            refreshResolvedCLI()
        }
    }

    func select(_ template: CommandTemplate) {
        selectedTemplate = template
        draft = template.defaultDraft()
        lastOutputURL = nil
        lastExitCode = nil
        status = "Idle"
    }

    func refreshResolvedCLI() {
        if cliPath.isBlank, !didAttemptAutomaticCLIInstall {
            didAttemptAutomaticCLIInstall = true
            handleAutomaticCLIInstall(CLIBootstrapInstaller.installBundledCLIIfNeeded())
        }

        let launch = CLIResolver.resolve(customPath: cliPath)
        resolvedCLI = displayDescription(for: launch)
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
        let launch = CLIResolver.resolve(customPath: cliPath)
        let args = commandArguments(template: template, draft: draft)
        return launch.displayCommand(for: masksSecrets ? args.maskingSecrets() : args)
    }

    func utilityCommandResult(args: [String], masksSecrets: Bool = true) async -> MereRunUtilityCommandResult {
        let launch = CLIResolver.resolve(customPath: cliPath)
        let cliArgs: [String]
        if !modelsRoot.isBlank {
            cliArgs = ["--models-root", NSString(string: modelsRoot).expandingTildeInPath] + args
        } else {
            cliArgs = args
        }
        let display = launch.displayCommand(for: masksSecrets ? cliArgs.maskingSecrets() : cliArgs)
        let output = ReadinessOutputBuffer()
        let errors = ReadinessOutputBuffer()
        let id = UUID()

        return await withCheckedContinuation { continuation in
            do {
                let process = try processRunner.start(
                    configuration: processConfiguration(
                        launch: launch,
                        args: cliArgs,
                        environmentTemplateID: .custom,
                        environmentDraft: CommandDraft()
                    ),
                    stdout: { text in output.append(text) },
                    stderr: { text in errors.append(text) },
                    termination: { [weak self] code in
                        let result = MereRunUtilityCommandResult(
                            commandPreview: display,
                            exitCode: code,
                            stdout: output.text(),
                            stderr: errors.text()
                        )
                        Task { @MainActor in
                            self?.utilityProcesses[id] = nil
                            continuation.resume(returning: result)
                        }
                    }
                )
                utilityProcesses[id] = process
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
    func run(studio request: StudioRunRequest) -> Bool {
        if isRunning || !queuedRuns.isEmpty {
            enqueue(request)
            return true
        }

        selectedTemplate = request.template
        draft = request.draft
        return startRun(requestID: request.id)
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
        let launch = CLIResolver.resolve(customPath: cliPath)
        let capabilityTemplate = CommandCatalog.template(id: .modelCapabilities) ?? selectedTemplate
        var capabilityDraft = capabilityTemplate.defaultDraft()
        capabilityDraft.all = true
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
                        let capabilities = ModelCapabilitiesParser.capabilities(from: output.text())
                        if !capabilities.isEmpty {
                            self.modelCapabilitiesByID = capabilities
                        }

                        if let message = self.modelCapabilitiesByID[modelID]?.unavailableMessage {
                            self.readinessProcesses[mode] = nil
                            self.readinessByMode[mode] = .unsupported(message)
                            return
                        }

                        if code != 0, capabilities.isEmpty {
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
        let launch = CLIResolver.resolve(customPath: cliPath)
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
        startRun(requestID: nil)
    }

    @discardableResult
    private func startRun(requestID: UUID?) -> Bool {
        guard !isRunning else { return false }
        refreshResolvedCLI()

        let launch = CLIResolver.resolve(customPath: cliPath)
        let args = commandArguments
        let display = launch.displayCommand(for: args)
        let expectedOutput = expectedOutputURL()
        activeRunTemplateID = selectedTemplate.id
        activeRunPreview = display
        activeRunRequestID = requestID

        if let message = selectedTemplate.validationMessage(for: draft) {
            append(message, stream: .system)
            status = message
            lastExitCode = 64
            finishPreflightFailure(exitCode: 64, outputText: message)
            return false
        }

        logs.removeAll()
        stdoutBuffer.removeAll(keepingCapacity: true)
        stderrBuffer.removeAll(keepingCapacity: true)
        liveOutputText = ""
        lastOutputURL = nil
        lastExitCode = nil

        guard prepareOutputLocation() else {
            lastExitCode = -1
            finishPreflightFailure(exitCode: -1, outputText: capturedResultText(exitCode: -1) ?? status)
            return false
        }

        status = selectedTemplate.id == .modelPull ? "Downloading model" : "Running"
        isRunning = true

        append(display, stream: .system)
        startOutputWatch(expectedOutput: expectedOutput)

        do {
            currentProcess = try processRunner.start(
                configuration: processConfiguration(launch: launch, args: args),
                stdout: { [weak self] text in
                    Task { @MainActor in
                        self?.append(text, stream: .stdout)
                    }
                },
                stderr: { [weak self] text in
                    Task { @MainActor in
                        self?.append(text, stream: .stderr)
                    }
                },
                termination: { [weak self] code in
                    Task { @MainActor in
                        self?.finishRun(exitCode: code, expectedOutput: expectedOutput)
                    }
                }
            )
        } catch {
            isRunning = false
            status = "Failed to start"
            append(error.localizedDescription, stream: .stderr)
            if let activeRunTemplateID {
                lastRunResult = MereRunRunResult(
                    requestID: activeRunRequestID,
                    templateID: activeRunTemplateID,
                    commandPreview: activeRunPreview,
                    exitCode: -1,
                    outputURL: nil,
                    outputText: capturedResultText(exitCode: -1)
                )
            }
            activeRunTemplateID = nil
            activeRunPreview = ""
            activeRunRequestID = nil
            stopOutputWatch()
            startNextQueuedRun()
            return false
        }
        return true
    }

    func cancel() {
        guard isRunning else { return }
        currentProcess?.terminate()
        append("Termination requested.", stream: .system)
    }

    func openLastOutput() {
        guard let lastOutputURL else { return }
        NSWorkspace.shared.open(lastOutputURL)
    }

    func revealLastOutput() {
        guard let lastOutputURL else { return }
        NSWorkspace.shared.activateFileViewerSelecting([lastOutputURL])
    }

    private func finishRun(exitCode: Int32, expectedOutput: URL?) {
        currentProcess = nil
        isRunning = false
        stopOutputWatch()
        lastExitCode = exitCode

        let detectedOutput = detectOutputURL(expected: expectedOutput, stdout: stdoutBuffer)
        let outputText = capturedResultText(exitCode: exitCode)
        lastOutputURL = detectedOutput

        if exitCode == 0 {
            status = detectedOutput == nil ? "Completed" : "Completed: \(detectedOutput!.lastPathComponent)"
            append("Completed with exit code 0.", stream: .system)
        } else {
            status = "Exited \(exitCode)"
            append("Exited with code \(exitCode).", stream: .system)
        }

        if let activeRunTemplateID {
            lastRunResult = MereRunRunResult(
                requestID: activeRunRequestID,
                templateID: activeRunTemplateID,
                commandPreview: activeRunPreview,
                exitCode: exitCode,
                outputURL: detectedOutput,
                outputText: outputText
            )
        }
        activeRunTemplateID = nil
        activeRunPreview = ""
        activeRunRequestID = nil
        startNextQueuedRun()
    }

    private func finishPreflightFailure(exitCode: Int32, outputText: String?) {
        guard let templateID = activeRunTemplateID else { return }
        lastRunResult = MereRunRunResult(
            requestID: activeRunRequestID,
            templateID: templateID,
            commandPreview: activeRunPreview,
            exitCode: exitCode,
            outputURL: nil,
            outputText: outputText
        )
        activeRunTemplateID = nil
        activeRunPreview = ""
        activeRunRequestID = nil
        stopOutputWatch()
        startNextQueuedRun()
    }

    private func enqueue(_ request: StudioRunRequest) {
        queuedRuns.append(request)
        queuedRunCount = queuedRuns.count
        append("Queued \(request.mode.title.lowercased()) job.", stream: .system)
    }

    private func startNextQueuedRun() {
        guard !isRunning, currentProcess == nil, !queuedRuns.isEmpty else { return }
        let next = queuedRuns.removeFirst()
        queuedRunCount = queuedRuns.count
        selectedTemplate = next.template
        draft = next.draft
        _ = startRun(requestID: next.id)
    }

    private func startOutputWatch(expectedOutput: URL?) {
        stopOutputWatch()
        outputWatchTask = Task { [weak self] in
            while !Task.isCancelled {
                await MainActor.run {
                    self?.publishDetectedOutputIfNeeded(expectedOutput: expectedOutput)
                }
                try? await Task.sleep(for: .milliseconds(350))
            }
        }
    }

    private func stopOutputWatch() {
        outputWatchTask?.cancel()
        outputWatchTask = nil
    }

    private func publishDetectedOutputIfNeeded(expectedOutput: URL?) {
        guard isRunning else { return }
        let detectedOutput = detectOutputURL(expected: expectedOutput, stdout: stdoutBuffer)
        guard let detectedOutput, detectedOutput != lastOutputURL else { return }
        lastOutputURL = detectedOutput
        status = "Generated: \(detectedOutput.lastPathComponent)"
    }

    private func handleAutomaticCLIInstall(_ outcome: CLIBootstrapInstallOutcome) {
        switch outcome {
        case .installed(let url):
            status = "CLI installed"
            append("Installed Terminal CLI at \(url.abbreviatedForDisplay).", stream: .system)
        case .failed(let message):
            append("Terminal CLI install skipped: \(message)", stream: .stderr)
        case .alreadyInstalled, .skippedNoBundledCLI:
            break
        }
    }

    private func displayDescription(for launch: MereRunLaunch) -> String {
        let description = launch.sourceDescription
        guard case .executable(let url) = launch,
              Bundle.main.resourceURL.map({ url.path.hasPrefix($0.path) }) == true,
              let installedURL = CLIResolver.existingInstalledCLI() else {
            return description
        }

        return "\(description) + \(installedURL.abbreviatedForDisplay)"
    }

    private func cancelReadinessCheck(for mode: StudioMode) {
        readinessProcesses[mode]?.terminate()
        readinessProcesses[mode] = nil
    }

    private func append(_ text: String, stream: LogStream) {
        if stream == .stdout {
            stdoutBuffer += text
            trimStdoutBuffer()
            liveOutputText = stdoutBuffer.replacingOccurrences(of: "\0", with: "")
            publishDetectedOutputIfNeeded(expectedOutput: expectedOutputURL())
        } else if stream == .stderr {
            stderrBuffer += text
            trimStderrBuffer()
        }

        let normalized = text
            .replacingOccurrences(of: "\r", with: "\n")
            .components(separatedBy: .newlines)

        for line in normalized {
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }
            logs.append(LogLine(stream: stream, text: trimmed))
        }

        if logs.count > 1200 {
            logs.removeFirst(logs.count - 1200)
        }
    }

    private func trimStdoutBuffer() {
        guard stdoutBuffer.utf8.count > Self.stdoutBufferByteLimit else { return }
        stdoutBuffer = String(decoding: stdoutBuffer.utf8.suffix(Self.stdoutBufferByteLimit), as: UTF8.self)
    }

    private func trimStderrBuffer() {
        guard stderrBuffer.utf8.count > Self.stderrBufferByteLimit else { return }
        stderrBuffer = String(decoding: stderrBuffer.utf8.suffix(Self.stderrBufferByteLimit), as: UTF8.self)
    }

    private func capturedStdoutText() -> String? {
        let trimmed = stdoutBuffer
            .replacingOccurrences(of: "\0", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    private func capturedStderrText() -> String? {
        let trimmed = stderrBuffer
            .replacingOccurrences(of: "\0", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    private func capturedResultText(exitCode: Int32) -> String? {
        let stdout = capturedStdoutText()
        guard exitCode != 0, let stderr = capturedStderrText() else {
            return stdout
        }

        if let stdout {
            return "\(stdout)\n\nSTDERR\n\(stderr)"
        }
        return stderr
    }

    private func expectedOutputURL() -> URL? {
        guard selectedTemplate.outputKind != .none, !draft.outputPath.isBlank else {
            return nil
        }
        return URL(fileURLWithPath: NSString(string: draft.outputPath).expandingTildeInPath)
    }

    private func prepareOutputLocation() -> Bool {
        guard !draft.outputPath.isBlank else {
            return true
        }

        let outputURL = URL(fileURLWithPath: NSString(string: draft.outputPath).expandingTildeInPath)
        let directoryURL: URL
        switch selectedTemplate.outputKind {
        case .file:
            directoryURL = outputURL.deletingLastPathComponent()
        case .directory:
            directoryURL = outputURL
        case .none:
            return true
        }

        do {
            try FileManager.default.createDirectory(at: directoryURL, withIntermediateDirectories: true)
            return true
        } catch {
            status = "Output path unavailable"
            append("Could not create output directory \(directoryURL.path): \(error.localizedDescription)", stream: .stderr)
            return false
        }
    }

    private func detectOutputURL(expected: URL?, stdout: String) -> URL? {
        let fm = FileManager.default
        if let expected, fm.fileExists(atPath: expected.path) {
            return expected
        }

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
        environmentDraft: CommandDraft? = nil
    ) -> MereRunProcessConfiguration {
        let processArgs: [String]
        if case .executable(let url) = launch, url.path == "/usr/bin/env" {
            processArgs = ["mere.run"] + args
        } else {
            processArgs = launch.processArguments(for: args)
        }
        let templateID = environmentTemplateID ?? selectedTemplate.id
        let environmentDraft = environmentDraft ?? draft

        return MereRunProcessConfiguration(
            executableURL: launch.executableURL,
            arguments: processArgs,
            currentDirectoryURL: workingDirectoryURL(),
            environment: processEnvironment(templateID: templateID, draft: environmentDraft)
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
            if masked[index] == "--api-key" {
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
