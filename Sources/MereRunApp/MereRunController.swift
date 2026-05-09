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

    private static func nearestPackageRoot(from anchor: URL, fileManager fm: FileManager) -> URL? {
        var cursor = anchor.standardizedFileURL
        while true {
            if fm.fileExists(atPath: cursor.appendingPathComponent("Package.swift").path) {
                return cursor
            }
            let parent = cursor.deletingLastPathComponent()
            if parent.path == cursor.path {
                return nil
            }
            cursor = parent
        }
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
    let id = UUID()
    let templateID: CommandTemplateID
    let commandPreview: String
    let exitCode: Int32
    let outputURL: URL?
    let outputText: String?
    let completedAt = Date()
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
    @Published var readinessByMode: [StudioMode: ModelReadinessState] = [:]
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
    private var stdoutBuffer = ""
    private var activeRunTemplateID: CommandTemplateID?
    private var activeRunPreview = ""
    private var didAttemptAutomaticCLIInstall = false

    private static let stdoutBufferByteLimit = 32 * 1024
    private static let outputDetectionLineLimit = 40

    var commandArguments: [String] {
        commandArguments(template: selectedTemplate, draft: draft)
    }

    var commandPreview: String {
        commandPreview(template: selectedTemplate, draft: draft, masksSecrets: true)
    }

    var advancedCommandPreview: String {
        commandPreview(template: selectedTemplate, draft: draft, masksSecrets: false)
    }

    init(processRunner: MereRunProcessRunning = FoundationMereRunProcessRunner()) {
        self.processRunner = processRunner
        let initial = CommandCatalog.templates.first!
        selectedTemplate = initial
        draft = initial.defaultDraft()
        cliPath = UserDefaults.standard.string(forKey: Keys.cliPath) ?? ""
        modelsRoot = UserDefaults.standard.string(forKey: Keys.modelsRoot) ?? ""
        hubCache = UserDefaults.standard.string(forKey: Keys.hubCache) ?? ""
        workingDirectory = UserDefaults.standard.string(forKey: Keys.workingDirectory) ?? FileManager.default.homeDirectoryForCurrentUser.path
        refreshResolvedCLI()
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

    func run(studio request: StudioRunRequest) {
        selectedTemplate = request.template
        draft = request.draft
        run()
    }

    func checkReadiness(for mode: StudioMode, draft studioDraft: StudioDraft) {
        let modelID = StudioCommandAdapter.requiredModel(for: mode, draft: studioDraft)
        guard !modelID.isBlank else {
            readinessByMode[mode] = .ready
            return
        }

        readinessByMode[mode] = .checking
        let launch = CLIResolver.resolve(customPath: cliPath)
        let args = commandArguments(
            template: CommandCatalog.template(id: .modelList) ?? selectedTemplate,
            draft: CommandCatalog.template(id: .modelList)?.defaultDraft() ?? CommandDraft()
        )
        let output = ReadinessOutputBuffer()

        do {
            _ = try processRunner.start(
                configuration: processConfiguration(launch: launch, args: args),
                stdout: { text in output.append(text) },
                stderr: { _ in },
                termination: { [weak self] _ in
                    Task { @MainActor in
                        self?.readinessByMode[mode] = ModelReadinessParser.state(
                            for: modelID,
                            modelListOutput: output.text()
                        )
                    }
                }
            )
        } catch {
            readinessByMode[mode] = .unknown(error.localizedDescription)
        }
    }

    func run() {
        guard !isRunning else { return }
        refreshResolvedCLI()

        if let message = selectedTemplate.validationMessage(for: draft) {
            append(message, stream: .system)
            status = message
            lastExitCode = nil
            return
        }

        let launch = CLIResolver.resolve(customPath: cliPath)
        let args = commandArguments
        let display = launch.displayCommand(for: args)
        let expectedOutput = expectedOutputURL()

        logs.removeAll()
        stdoutBuffer.removeAll(keepingCapacity: true)
        liveOutputText = ""
        lastOutputURL = nil
        lastExitCode = nil

        guard prepareOutputLocation() else { return }

        status = "Running"
        isRunning = true
        activeRunTemplateID = selectedTemplate.id
        activeRunPreview = display

        append(display, stream: .system)

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
                    templateID: activeRunTemplateID,
                    commandPreview: activeRunPreview,
                    exitCode: -1,
                    outputURL: nil,
                    outputText: capturedStdoutText()
                )
            }
            activeRunTemplateID = nil
            activeRunPreview = ""
            return
        }
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
        lastExitCode = exitCode

        let detectedOutput = detectOutputURL(expected: expectedOutput, stdout: stdoutBuffer)
        let outputText = capturedStdoutText()
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
                templateID: activeRunTemplateID,
                commandPreview: activeRunPreview,
                exitCode: exitCode,
                outputURL: detectedOutput,
                outputText: outputText
            )
        }
        activeRunTemplateID = nil
        activeRunPreview = ""
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

    private func append(_ text: String, stream: LogStream) {
        if stream == .stdout {
            stdoutBuffer += text
            trimStdoutBuffer()
            liveOutputText = stdoutBuffer.replacingOccurrences(of: "\0", with: "")
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

    private func capturedStdoutText() -> String? {
        let trimmed = stdoutBuffer
            .replacingOccurrences(of: "\0", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    private func expectedOutputURL() -> URL? {
        guard selectedTemplate.outputKind.isFile, !draft.outputPath.isBlank else {
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

    private func processEnvironment() -> [String: String] {
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
        for (key, value) in CommandLaunchEnvironment.overrides(templateID: selectedTemplate.id, draft: draft) {
            env[key] = value
        }
        return env
    }

    private func processConfiguration(launch: MereRunLaunch, args: [String]) -> MereRunProcessConfiguration {
        let processArgs: [String]
        if case .executable(let url) = launch, url.path == "/usr/bin/env" {
            processArgs = ["mere.run"] + args
        } else {
            processArgs = launch.processArguments(for: args)
        }

        return MereRunProcessConfiguration(
            executableURL: launch.executableURL,
            arguments: processArgs,
            currentDirectoryURL: workingDirectoryURL(),
            environment: processEnvironment()
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
