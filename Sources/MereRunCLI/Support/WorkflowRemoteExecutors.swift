import ArgumentParser
import MereRunRelayKit
import Crypto
import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif
import MereRunCore

struct SSHWorkflowExecutor {
    let profile: WorkflowExecutorProfile
    let executableRunner: ([String], FileHandle?) throws -> WorkflowProcessResult

    init(
        profile: WorkflowExecutorProfile,
        executableRunner: @escaping ([String], FileHandle?) throws -> WorkflowProcessResult = runExecutable
    ) {
        self.profile = profile
        self.executableRunner = executableRunner
    }

    func probe() throws -> WorkflowExecutorProbe {
        let result = try runSSH(remoteCommand: "\(remoteExecutable) graph worker probe --json")
        guard result.status == 0 else {
            throw ValidationError("SSH executor probe failed with status \(result.status).")
        }
        return try JSONDecoder().decode(WorkflowExecutorProbe.self, from: Data(result.stdout.utf8))
    }

    func submit(bundleDirectory: URL, localRunDirectory: URL) throws -> WorkflowRemoteJob {
        let worker = try probe()
        let job = try WorkflowBundleCodec.decoder().decode(
            WorkflowJobManifest.self,
            from: Data(contentsOf: bundleDirectory.appendingPathComponent(WorkflowJobManifest.filename))
        )
        try validateWorker(worker, for: job, executor: "ssh:\(profile.name)")
        let assets = try WorkflowBundleCodec.decoder().decode(
            WorkflowAssetManifest.self,
            from: Data(contentsOf: bundleDirectory.appendingPathComponent(WorkflowAssetManifest.filename))
        )
        let assetEntries = Dictionary(
            assets.groups.flatMap(\.entries).map { ($0.digest, $0) },
            uniquingKeysWith: { first, _ in first }
        )
        let remoteRoot = try resolvedRemoteRoot()
        let remoteJob = "\(remoteRoot)/jobs/\(job.jobID)"
        let prepare = [
            "set -eu",
            "root=\(shellQuote(remoteRoot))",
            "job=\(shellQuote(remoteJob))",
            "test ! -e \"$job\"",
            "mkdir -p \"$root/assets/sha256\" \"$root/assets/incoming\" \"$job/assets/sha256\"",
        ].joined(separator: "; ")
        guard try runSSH(remoteCommand: prepare).status == 0 else {
            throw ValidationError("Could not prepare SSH workflow job directory.")
        }

        let missing = try missingRemoteAssets(entries: assetEntries, remoteRoot: remoteRoot)
        for digest in missing {
            guard let entry = assetEntries[digest] else {
                throw ValidationError("Remote requested an unknown workflow asset: \(digest)")
            }
            let source = bundleDirectory.appendingPathComponent("assets/sha256/\(digest)")
            let incoming = "\(remoteRoot)/assets/incoming/\(job.jobID)-\(digest)"
            try uploadFile(source, remotePath: incoming)
            let finalize = [
                "set -eu",
                "incoming=\(shellQuote(incoming))",
                "destination=\(shellQuote("\(remoteRoot)/assets/sha256/\(digest)"))",
                "test \"$(wc -c <\"$incoming\" | tr -d ' ')\" = \(shellQuote(String(entry.sizeBytes)))",
                "if command -v sha256sum >/dev/null 2>&1; then actual=$(sha256sum \"$incoming\" | awk '{print $1}'); else actual=$(shasum -a 256 \"$incoming\" | awk '{print $1}'); fi",
                "test \"$actual\" = \(shellQuote(digest))",
                "mv \"$incoming\" \"$destination\"",
            ].joined(separator: "; ")
            guard try runSSH(remoteCommand: finalize).status == 0 else {
                throw ValidationError("SSH workflow asset verification failed: \(digest)")
            }
        }

        let temporaryArchive = FileManager.default.temporaryDirectory
            .appendingPathComponent("mere-run-job-\(job.jobID).tar.gz")
        let temporaryLauncher = FileManager.default.temporaryDirectory
            .appendingPathComponent("mere-run-graph-launcher-\(job.jobID).sh")
        defer {
            try? FileManager.default.removeItem(at: temporaryArchive)
            try? FileManager.default.removeItem(at: temporaryLauncher)
        }
        try createArchive(bundleDirectory: bundleDirectory, destination: temporaryArchive)
        try Data(Self.launcherScript.utf8).write(to: temporaryLauncher, options: .atomic)
        try uploadFile(temporaryLauncher, remotePath: "\(remoteJob)/launcher.sh")

        var commandParts = [
            "set -eu",
            "root=\(shellQuote(remoteRoot))",
            "job=\(shellQuote(remoteJob))",
            "chmod 700 \"$job/launcher.sh\"",
            "tar -xzf - -C \"$job\"",
        ]
        commandParts.append(contentsOf: assetEntries.keys.sorted().map { digest in
            "ln \"$root/assets/sha256/\(digest)\" \"$job/assets/sha256/\(digest)\" 2>/dev/null || cp \"$root/assets/sha256/\(digest)\" \"$job/assets/sha256/\(digest)\""
        })
        commandParts.append(
            "nohup \"$job/launcher.sh\" \"$job\" \(shellQuote(profile.mereRunPath ?? "mere.run")) "
                + ">\"$job/worker.events.jsonl\" 2>\"$job/worker.stderr.log\" </dev/null "
                + "& pid=$!; echo \"$pid\" >\"$job/worker.pid\"; printf '%s\\n' \"$job\""
        )
        let command = commandParts.joined(separator: "; ")
        let result = try runSSH(remoteCommand: command, standardInput: try FileHandle(forReadingFrom: temporaryArchive))
        guard result.status == 0 else {
            throw ValidationError("SSH workflow submission failed with status \(result.status).")
        }
        let reference = "ssh://\(profile.name)/\(job.jobID)"
        try initializeRemoteRunRecord(
            bundleDirectory: bundleDirectory,
            runDirectory: localRunDirectory,
            executor: .init(kind: "ssh", profile: profile.name, jobReference: reference),
            state: .queued,
            pluginNodes: WorkflowGraphProviderRegistry.discoveredCatalog().nodes
        )
        return WorkflowRemoteJob(
            jobID: job.jobID,
            jobReference: reference,
            state: .queued,
            executor: "ssh:\(profile.name)",
            runDirectory: localRunDirectory.path,
            createdAt: job.createdAt,
            updatedAt: job.createdAt,
            artifacts: [],
            error: nil,
            placement: nil,
            metrics: nil
        )
    }

    func inspect(jobID: String) throws -> WorkflowRemoteJob {
        let remoteJobExpression = remotePathExpression("\(profile.remoteRoot!)/jobs/\(jobID)")
        let command = "job=\(remoteJobExpression); \(remoteExecutable) graph worker inspect --run-dir \"$job\" --json"
        let result = try runSSH(remoteCommand: command)
        guard result.status == 0 else {
            throw ValidationError("SSH workflow inspection failed with status \(result.status).")
        }
        let manifest = try WorkflowBundleCodec.decoder().decode(GraphRunManifest.self, from: Data(result.stdout.utf8))
        return WorkflowRemoteJob(
            jobID: jobID,
            jobReference: "ssh://\(profile.name)/\(jobID)",
            state: manifest.state,
            executor: "ssh:\(profile.name)",
            runDirectory: nil,
            createdAt: manifest.createdAt,
            updatedAt: manifest.updatedAt,
            artifacts: manifest.outputs,
            error: manifest.error,
            placement: nil,
            metrics: nil
        )
    }

    func events(jobID: String) throws -> String {
        let remoteJobExpression = remotePathExpression("\(profile.remoteRoot!)/jobs/\(jobID)")
        let command = "job=\(remoteJobExpression); test ! -f \"$job/worker.events.jsonl\" || cat \"$job/worker.events.jsonl\""
        let result = try runSSH(remoteCommand: command)
        guard result.status == 0 else {
            throw ValidationError("SSH workflow event read failed with status \(result.status).")
        }
        return result.stdout
    }

    func cancel(jobID: String) throws -> WorkflowRemoteJob {
        let existing = try? inspect(jobID: jobID)
        let remoteJobExpression = remotePathExpression("\(profile.remoteRoot!)/jobs/\(jobID)")
        let command = [
            "set -eu",
            "job=\(remoteJobExpression)",
            "\(remoteExecutable) graph worker cancel --run-dir \"$job\" --json",
        ].joined(separator: "; ")
        let result = try runSSH(remoteCommand: command)
        guard result.status == 0 else {
            throw ValidationError("SSH workflow cancellation failed with status \(result.status).")
        }
        return WorkflowRemoteJob(
            jobID: jobID,
            jobReference: "ssh://\(profile.name)/\(jobID)",
            state: .cancelled,
            executor: "ssh:\(profile.name)",
            runDirectory: nil,
            createdAt: existing?.createdAt,
            updatedAt: Date(),
            artifacts: existing?.artifacts ?? [],
            error: "Cancellation requested.",
            placement: nil,
            metrics: nil
        )
    }

    func fetch(
        jobID: String,
        into destination: URL,
        allArtifacts: Bool,
        artifactNames: Set<String> = []
    ) throws -> WorkflowRemoteJob {
        try prepareFetchDestination(destination, expectedJobID: jobID)
        let remotePath = try resolvedRemoteJobPath(jobID: jobID)
        let existing = try inspect(jobID: jobID)
        let selected = try selectedArtifacts(
            existing.artifacts,
            allArtifacts: allArtifacts,
            artifactNames: artifactNames
        )
        var arguments = sshCommonArguments(executable: "scp")
        arguments.append("-r")
        arguments.append(contentsOf: Self.fetchRelativePaths(
            allArtifacts: allArtifacts && artifactNames.isEmpty,
            includeOutputs: artifactNames.isEmpty
        ).map {
            scpRemotePath("\(remotePath)/\($0)")
        })
        arguments.append(destination.path)
        let result = try executableRunner(arguments, nil)
        guard result.status == 0 else {
            throw ValidationError("SSH workflow fetch failed with status \(result.status).")
        }
        if !artifactNames.isEmpty {
            for artifact in selected {
                let localURL = try confinedFetchURL(root: destination, relativePath: artifact.path)
                if try verifiedExistingArtifact(artifact, at: localURL) { continue }
                try FileManager.default.createDirectory(
                    at: localURL.deletingLastPathComponent(),
                    withIntermediateDirectories: true
                )
                var artifactArguments = sshCommonArguments(executable: "scp")
                artifactArguments.append(scpRemotePath("\(remotePath)/\(artifact.path)"))
                artifactArguments.append(localURL.path)
                let artifactResult = try executableRunner(artifactArguments, nil)
                guard artifactResult.status == 0 else {
                    throw ValidationError("SSH artifact fetch failed with status \(artifactResult.status): \(artifact.name)")
                }
            }
        }
        var manifest = try WorkflowBundleCodec.decoder().decode(
            GraphRunManifest.self,
            from: Data(contentsOf: destination.appendingPathComponent(GraphRunManifest.filename))
        )
        manifest.executor = .init(
            kind: "ssh",
            profile: profile.name,
            jobReference: "ssh://\(profile.name)/\(jobID)"
        )
        try WorkflowBundleCodec.write(manifest, to: destination.appendingPathComponent(GraphRunManifest.filename))
        try verifyFetchedArtifacts(artifactNames.isEmpty ? manifest.outputs : selected, root: destination)
        return WorkflowRemoteJob(
            jobID: jobID,
            jobReference: "ssh://\(profile.name)/\(jobID)",
            state: manifest.state,
            executor: "ssh:\(profile.name)",
            runDirectory: destination.path,
            createdAt: manifest.createdAt,
            updatedAt: manifest.updatedAt,
            artifacts: artifactNames.isEmpty ? manifest.outputs : selected,
            error: manifest.error,
            placement: nil,
            metrics: nil
        )
    }

    private var remoteExecutable: String {
        shellQuote(profile.mereRunPath ?? "mere.run")
    }

    private func createArchive(bundleDirectory: URL, destination: URL) throws {
        let result = try executableRunner([
            "tar", "--no-xattrs", "-czf", destination.path,
            "-C", bundleDirectory.path,
            "graph.json", "inputs.json", WorkflowAssetManifest.filename, WorkflowJobManifest.filename,
        ], nil)
        guard result.status == 0 else {
            throw ValidationError("Could not create workflow transport archive.")
        }
    }

    private func runSSH(remoteCommand: String, standardInput: FileHandle? = nil) throws -> WorkflowProcessResult {
        var arguments = sshCommonArguments(executable: "ssh")
        arguments += [profile.destination!, remoteCommand]
        return try executableRunner(arguments, standardInput)
    }

    private func sshCommonArguments(executable: String) -> [String] {
        var arguments = [executable, "-o", "BatchMode=yes"]
        if executable == "scp" {
            // Force the legacy SCP transport because its remote shell honors
            // the POSIX quoting applied by scpRemotePath. Modern SFTP mode
            // treats those quotes as literal filename characters.
            arguments.append("-O")
        }
        if let port = profile.port {
            arguments += [executable == "scp" ? "-P" : "-p", String(port)]
        }
        if let identityFile = profile.identityFile {
            arguments += ["-i", identityFile]
        }
        return arguments
    }

    private func remotePathExpression(_ value: String) -> String {
        if value.hasPrefix("~/") {
            return "$HOME/\(shellQuote(String(value.dropFirst(2))))"
        }
        return shellQuote(value)
    }

    private static let launcherScript = """
    #!/bin/sh
    set -eu
    job=$1
    runner=$2
    case "$runner" in
      '~/'*) runner="$HOME/${runner#~/}" ;;
    esac
    exec "$runner" graph worker execute --bundle "$job" --run-dir "$job" --json-stream
    """

    private func resolvedRemoteRoot() throws -> String {
        let expression = remotePathExpression(profile.remoteRoot!)
        let result = try runSSH(remoteCommand: "set -eu; root=\(expression); mkdir -p \"$root\"; cd \"$root\"; pwd -P")
        guard result.status == 0,
              result.stdout.hasPrefix("/"),
              !result.stdout.contains("\n") else {
            throw ValidationError("SSH executor returned an invalid remote root.")
        }
        return result.stdout
    }

    private func missingRemoteAssets(
        entries: [String: WorkflowAssetEntry],
        remoteRoot: String
    ) throws -> [String] {
        guard !entries.isEmpty else { return [] }
        let checks = entries.keys.sorted().map { digest in
            "test -f \"$root/assets/sha256/\(digest)\" || printf '%s\\n' \(shellQuote(digest))"
        }.joined(separator: "; ")
        let result = try runSSH(remoteCommand: "set -eu; root=\(shellQuote(remoteRoot)); \(checks)")
        guard result.status == 0 else {
            throw ValidationError("Could not query the SSH workflow asset cache.")
        }
        return result.stdout.split(whereSeparator: \Character.isNewline).map(String.init)
    }

    private func uploadFile(_ localURL: URL, remotePath: String) throws {
        var arguments = sshCommonArguments(executable: "scp")
        arguments += [localURL.path, scpRemotePath(remotePath)]
        guard try executableRunner(arguments, nil).status == 0 else {
            throw ValidationError("SSH file upload failed: \(localURL.lastPathComponent)")
        }
    }

    private func resolvedRemoteJobPath(jobID: String) throws -> String {
        "\(try resolvedRemoteRoot())/jobs/\(jobID)"
    }

    private func scpRemotePath(_ path: String) -> String {
        "\(profile.destination!):\(shellQuote(path))"
    }

    static func fetchRelativePaths(allArtifacts: Bool, includeOutputs: Bool = true) -> [String] {
        var paths = [
            GraphRunManifest.filename,
            "graph.json",
            "inputs.json",
            WorkflowJobManifest.filename,
            WorkflowAssetManifest.filename,
            "events.jsonl",
        ]
        if includeOutputs { paths.append("outputs") }
        if allArtifacts {
            paths += ["actions.json", "nodes"]
        }
        return paths
    }
}

// Submission logic lives in MereRunRelayKit; the CLI supplies discovered
// plugin nodes for the local run record's validation pass.
extension RelayWorkflowExecutor {
    func submit(bundleDirectory: URL, localRunDirectory: URL) async throws -> WorkflowRemoteJob {
        try await submit(
            bundleDirectory: bundleDirectory,
            localRunDirectory: localRunDirectory,
            pluginNodes: WorkflowGraphProviderRegistry.discoveredCatalog().nodes
        )
    }
}

func runExecutable(
    arguments: [String],
    standardInput: FileHandle? = nil
) throws -> WorkflowProcessResult {
    guard let executable = arguments.first else {
        throw ValidationError("Missing executable.")
    }
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
    process.arguments = [executable] + arguments.dropFirst()
    process.standardInput = standardInput ?? FileHandle.nullDevice
    let stdout = Pipe()
    process.standardOutput = stdout
    process.standardError = FileHandle.standardError
    try process.run()
    process.waitUntilExit()
    if let standardInput { try? standardInput.close() }
    return WorkflowProcessResult(
        status: process.terminationStatus,
        stdout: String(decoding: stdout.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    )
}

func shellQuote(_ value: String) -> String {
    "'\(value.replacingOccurrences(of: "'", with: "'\\''"))'"
}

