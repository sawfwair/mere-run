import ArgumentParser
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
            state: .queued
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

struct RelayWorkflowExecutor {
    let profile: WorkflowExecutorProfile

    func probe() async throws -> WorkflowExecutorProbe {
        let data = try await request(path: "/api/graph-jobs/capabilities", method: "GET")
        return try JSONDecoder().decode(WorkflowExecutorProbe.self, from: data)
    }

    func fleet() async throws -> RelayFleetSnapshot {
        let data = try await request(path: "/api/fleet", method: "GET")
        return try WorkflowBundleCodec.decoder().decode(RelayFleetSnapshot.self, from: data)
    }

    func refreshNode(deviceID: String) async throws -> RelayFleetRefreshResult {
        let encodedDeviceID = try encodedPathSegment(deviceID)
        let data = try await request(
            path: "/api/fleet/nodes/\(encodedDeviceID)/refresh",
            method: "POST",
            body: Data("{}".utf8)
        )
        return try WorkflowBundleCodec.decoder().decode(RelayFleetRefreshResult.self, from: data)
    }

    func configureNode(deviceID: String, patch: RelayFleetNodePolicyPatch) async throws -> RelayFleetNode {
        let encodedDeviceID = try encodedPathSegment(deviceID)
        let data = try await request(
            path: "/api/fleet/nodes/\(encodedDeviceID)",
            method: "PATCH",
            body: try WorkflowBundleCodec.encoder().encode(patch)
        )
        return try WorkflowBundleCodec.decoder().decode(RelayFleetNode.self, from: data)
    }

    func submit(bundleDirectory: URL, localRunDirectory: URL) async throws -> WorkflowRemoteJob {
        let graph = try WorkflowGraphDocument.load(from: bundleDirectory.appendingPathComponent("graph.json"))
        let inputs = try WorkflowInputsDocument.load(from: bundleDirectory.appendingPathComponent("inputs.json"))
        let assets = try WorkflowBundleCodec.decoder().decode(
            WorkflowAssetManifest.self,
            from: Data(contentsOf: bundleDirectory.appendingPathComponent(WorkflowAssetManifest.filename))
        )
        let job = try WorkflowBundleCodec.decoder().decode(
            WorkflowJobManifest.self,
            from: Data(contentsOf: bundleDirectory.appendingPathComponent(WorkflowJobManifest.filename))
        )
        try validateWorker(try await probe(), for: job, executor: "relay:\(profile.name)", allowMixedBackend: true)
        let createBody = RelayGraphCreateRequest(
            job: job,
            graph: graph,
            inputs: inputs,
            assets: assets,
            bundleDocuments: try Dictionary(uniqueKeysWithValues: [
                WorkflowJobManifest.filename,
                "graph.json",
                "inputs.json",
                WorkflowAssetManifest.filename,
            ].map { path in
                (path, try Data(contentsOf: bundleDirectory.appendingPathComponent(path)))
            })
        )
        let createData = try await request(
            path: "/api/graph-jobs",
            method: "POST",
            body: try WorkflowBundleCodec.encoder().encode(createBody)
        )
        let created = try JSONDecoder().decode(RelayGraphCreateResponse.self, from: createData)
        let assetRoot = bundleDirectory.appendingPathComponent("assets/sha256", isDirectory: true)
        for digest in created.missingAssetDigests {
            let url = assetRoot.appendingPathComponent(digest)
            let data = try Data(contentsOf: url)
            guard ModelArtifactPinDigest.sha256(data) == digest else {
                throw ValidationError("Job bundle asset digest mismatch before relay upload: \(digest)")
            }
            _ = try await request(
                path: "/api/graph-jobs/\(created.jobID)/assets/\(digest)",
                method: "PUT",
                contentType: "application/octet-stream",
                body: data,
                transientRetryAttempts: 3
            )
        }
        let commitData = try await request(
            path: "/api/graph-jobs/\(created.jobID)/commit",
            method: "POST",
            body: Data("{}".utf8)
        )
        let committed = try WorkflowBundleCodec.decoder().decode(RelayGraphJobResponse.self, from: commitData)
        let reference = "relay://\(profile.name)/\(committed.jobID)"
        try initializeRemoteRunRecord(
            bundleDirectory: bundleDirectory,
            runDirectory: localRunDirectory,
            executor: .init(kind: "relay", profile: profile.name, jobReference: reference),
            state: committed.state
        )
        return committed.remoteJob(profile: profile.name, localRunDirectory: localRunDirectory.path)
    }

    func inspect(jobID: String) async throws -> WorkflowRemoteJob {
        let data = try await request(path: "/api/graph-jobs/\(jobID)", method: "GET")
        return try WorkflowBundleCodec.decoder()
            .decode(RelayGraphJobResponse.self, from: data)
            .remoteJob(profile: profile.name, localRunDirectory: nil)
    }

    func events(jobID: String) async throws -> String {
        let data = try await request(path: "/api/graph-jobs/\(jobID)/events", method: "GET")
        return String(decoding: data, as: UTF8.self)
    }

    func cancel(jobID: String) async throws -> WorkflowRemoteJob {
        let data = try await request(path: "/api/graph-jobs/\(jobID)", method: "DELETE")
        return try WorkflowBundleCodec.decoder()
            .decode(RelayGraphJobResponse.self, from: data)
            .remoteJob(profile: profile.name, localRunDirectory: nil)
    }

    func retry(jobID: String) async throws -> WorkflowRemoteJob {
        let data = try await request(
            path: "/api/graph-jobs/\(jobID)/retry",
            method: "POST",
            body: Data("{}".utf8)
        )
        return try WorkflowBundleCodec.decoder()
            .decode(RelayGraphJobResponse.self, from: data)
            .remoteJob(profile: profile.name, localRunDirectory: nil)
    }

    func list(limit: Int) async throws -> [WorkflowRemoteJob] {
        let data = try await request(path: "/api/graph-jobs?limit=\(limit)", method: "GET")
        let response = try WorkflowBundleCodec.decoder().decode(RelayGraphJobListResponse.self, from: data)
        return response.jobs.map { $0.remoteJob(profile: profile.name, localRunDirectory: nil) }
    }

    func fetch(
        jobID: String,
        into destination: URL,
        allArtifacts: Bool,
        artifactNames: Set<String> = []
    ) async throws -> WorkflowRemoteJob {
        let job = try await inspect(jobID: jobID)
        guard job.state == .finished else {
            throw ValidationError("Relay job \(jobID) is not finished.")
        }
        try prepareFetchDestination(destination, expectedJobID: jobID)
        let manifestData = try await request(path: "/api/graph-jobs/\(jobID)/run-manifest", method: "GET")
        var manifest = try WorkflowBundleCodec.decoder().decode(GraphRunManifest.self, from: manifestData)
        guard manifest.jobID == jobID else {
            throw ValidationError("Relay returned a run manifest for a different job.")
        }
        manifest.executor = .init(
            kind: "relay",
            profile: profile.name,
            jobReference: "relay://\(profile.name)/\(jobID)"
        )
        let selected = try selectedArtifacts(
            job.artifacts,
            allArtifacts: allArtifacts,
            artifactNames: artifactNames
        )
        for artifact in selected {
            let url = try confinedFetchURL(root: destination, relativePath: artifact.path)
            if try verifiedExistingArtifact(artifact, at: url) { continue }
            let encodedName = try encodedPathSegment(artifact.name)
            let data = try await request(
                path: "/api/graph-jobs/\(jobID)/artifacts/\(encodedName)",
                method: "GET",
                authorize: true
            )
            guard Int64(data.count) == artifact.sizeBytes,
                  ModelArtifactPinDigest.sha256(data) == artifact.sha256 else {
                throw ValidationError("Fetched relay artifact failed verification: \(artifact.name)")
            }
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try data.write(to: url, options: .atomic)
        }
        try WorkflowBundleCodec.write(manifest, to: destination.appendingPathComponent(GraphRunManifest.filename))
        let fetchedOutputs = manifest.outputs.filter { output in
            selected.contains { $0.name == output.name }
        }
        try verifyFetchedArtifacts(fetchedOutputs, root: destination)
        return WorkflowRemoteJob(
            jobID: job.jobID,
            jobReference: job.jobReference,
            state: job.state,
            executor: job.executor,
            runDirectory: destination.path,
            createdAt: job.createdAt,
            updatedAt: job.updatedAt,
            artifacts: selected,
            error: job.error,
            placement: job.placement,
            metrics: job.metrics
        )
    }

    private func request(
        path: String,
        method: String,
        contentType: String = "application/json",
        body: Data? = nil,
        authorize: Bool = true,
        transientRetryAttempts: Int = 1
    ) async throws -> Data {
        guard let baseURL = profile.url, let url = URL(string: "\(baseURL)\(path)") else {
            throw ValidationError("Relay executor profile has an invalid URL.")
        }
        let credential = authorize ? try await RelayAuthentication.resolveCredential(profile: profile) : nil
        var result = try await Self.performWithTransientRetries(maximumAttempts: transientRetryAttempts) {
            try await performRequest(
                url: url,
                method: method,
                contentType: contentType,
                body: body,
                credential: credential
            )
        }
        if result.response.statusCode == 401, credential?.refreshable == true {
            let refreshed = try await RelayAuthentication.resolveCredential(profile: profile, forceRefresh: true)
            result = try await Self.performWithTransientRetries(maximumAttempts: transientRetryAttempts) {
                try await performRequest(
                    url: url,
                    method: method,
                    contentType: contentType,
                    body: body,
                    credential: refreshed
                )
            }
        }
        guard (200..<300).contains(result.response.statusCode) else {
            let status = result.response.statusCode
            let data = result.data
            let detail = String(decoding: data, as: UTF8.self)
            throw ValidationError("Relay request failed with HTTP \(status): \(detail)")
        }
        return result.data
    }

    static func performWithTransientRetries(
        maximumAttempts: Int,
        delayNanoseconds: UInt64 = 250_000_000,
        operation: () async throws -> (data: Data, response: HTTPURLResponse)
    ) async throws -> (data: Data, response: HTTPURLResponse) {
        precondition(maximumAttempts > 0)
        for attempt in 1...maximumAttempts {
            do {
                let result = try await operation()
                guard isTransientHTTPStatus(result.response.statusCode), attempt < maximumAttempts else {
                    return result
                }
            } catch let error as URLError {
                guard attempt < maximumAttempts else { throw error }
            }
            try await Task.sleep(nanoseconds: UInt64(attempt) * delayNanoseconds)
        }
        preconditionFailure("Transient request retry loop exhausted without returning.")
    }

    private static func isTransientHTTPStatus(_ statusCode: Int) -> Bool {
        statusCode == 408 || statusCode == 429 || (500...599).contains(statusCode)
    }

    private func performRequest(
        url: URL,
        method: String,
        contentType: String,
        body: Data?,
        credential: RelayResolvedCredential?
    ) async throws -> (data: Data, response: HTTPURLResponse) {
        var request = URLRequest(url: url)
        request.httpMethod = method
        request.timeoutInterval = 60
        request.setValue(contentType, forHTTPHeaderField: "Content-Type")
        if let credential {
            request.setValue("Bearer \(credential.accessToken)", forHTTPHeaderField: "Authorization")
        }
        request.httpBody = body
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw ValidationError("Relay returned a non-HTTP response.")
        }
        return (data, http)
    }
}

private struct RelayGraphCreateRequest: Codable {
    let job: WorkflowJobManifest
    let graph: WorkflowGraphDocument
    let inputs: WorkflowInputsDocument
    let assets: WorkflowAssetManifest
    let bundleDocuments: [String: Data]

    enum CodingKeys: String, CodingKey {
        case job
        case graph
        case inputs
        case assets
        case bundleDocuments = "bundle_documents"
    }
}

private struct RelayGraphCreateResponse: Codable {
    let jobID: String
    let state: GraphRunState
    let missingAssetDigests: [String]

    enum CodingKeys: String, CodingKey {
        case jobID = "job_id"
        case state
        case missingAssetDigests = "missing_asset_digests"
    }
}

private struct RelayGraphJobResponse: Codable {
    let jobID: String
    let state: GraphRunState
    let createdAt: Date?
    let updatedAt: Date?
    let artifacts: [GraphRunArtifact]
    let error: String?
    let placement: WorkflowGraphPlacement?
    let metrics: WorkflowGraphExecutionMetrics?

    enum CodingKeys: String, CodingKey {
        case jobID = "job_id"
        case state
        case createdAt = "created_at"
        case updatedAt = "updated_at"
        case artifacts
        case error
        case placement
        case metrics
    }

    func remoteJob(profile: String, localRunDirectory: String?) -> WorkflowRemoteJob {
        WorkflowRemoteJob(
            jobID: jobID,
            jobReference: "relay://\(profile)/\(jobID)",
            state: state,
            executor: "relay:\(profile)",
            runDirectory: localRunDirectory,
            createdAt: createdAt,
            updatedAt: updatedAt,
            artifacts: artifacts,
            error: error,
            placement: placement,
            metrics: metrics
        )
    }
}

private struct RelayGraphJobListResponse: Codable {
    let jobs: [RelayGraphJobResponse]
}

enum ModelArtifactPinDigest {
    static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
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

private func initializeRemoteRunRecord(
    bundleDirectory: URL,
    runDirectory: URL,
    executor: GraphRunExecutorRecord,
    state: GraphRunState
) throws {
    let graph = try WorkflowGraphDocument.load(from: bundleDirectory.appendingPathComponent("graph.json"))
    let inputs = try WorkflowInputsDocument.load(from: bundleDirectory.appendingPathComponent("inputs.json"))
    let job = try WorkflowBundleCodec.decoder().decode(
        WorkflowJobManifest.self,
        from: Data(contentsOf: bundleDirectory.appendingPathComponent(WorkflowJobManifest.filename))
    )
    let validation = WorkflowGraphValidator.validate(graph: graph, inputs: inputs)
    let nodesByID = Dictionary(uniqueKeysWithValues: graph.nodes.map { ($0.id, $0) })
    let now = Date()
    let manifest = GraphRunManifest(
        contractVersion: GraphRunManifest.contractVersion,
        jobID: job.jobID,
        graphName: graph.name,
        graphFingerprint: job.graphFingerprint,
        state: state,
        createdAt: now,
        updatedAt: now,
        attempt: 1,
        executor: executor,
        nodes: validation.order.compactMap { id in
            nodesByID[id].map {
                GraphRunNodeRecord(
                    id: id,
                    kind: $0.kind,
                    state: .planned,
                    startedAt: nil,
                    completedAt: nil,
                    exitStatus: nil,
                    attempt: 0,
                    maxAttempts: $0.execution?.resolvedMaxAttempts ?? 1,
                    fingerprint: "",
                    artifacts: [],
                    error: nil
                )
            }
        },
        outputs: [],
        error: nil
    )
    try WorkflowBundleCodec.write(manifest, to: runDirectory.appendingPathComponent(GraphRunManifest.filename))
    try WorkflowBundleCodec.write([DeclarativeAction](), to: runDirectory.appendingPathComponent("actions.json"))
}

private func verifyFetchedArtifacts(_ artifacts: [GraphRunArtifact], root: URL) throws {
    for artifact in artifacts {
        let localURL = try confinedFetchURL(root: root, relativePath: artifact.path)
        guard FileManager.default.fileExists(atPath: localURL.path),
              try ModelArtifactPin.fileByteCount(localURL) == artifact.sizeBytes,
              try ModelArtifactPin.fileSHA256(localURL) == artifact.sha256 else {
            throw ValidationError("Fetched artifact failed verification: \(artifact.name)")
        }
    }
}

private func selectedArtifacts(
    _ artifacts: [GraphRunArtifact],
    allArtifacts: Bool,
    artifactNames: Set<String>
) throws -> [GraphRunArtifact] {
    if artifactNames.isEmpty {
        return artifacts.filter {
            allArtifacts || $0.kind == "graph.output" || $0.kind == "graph.report" || $0.kind == "graph.manifest"
        }
    }
    let selected = artifacts.filter { artifactNames.contains($0.name) }
    let found = Set(selected.map(\.name))
    let missing = artifactNames.subtracting(found).sorted()
    guard missing.isEmpty else {
        throw ValidationError("Remote run does not contain artifacts: \(missing.joined(separator: ", ")).")
    }
    return selected
}

private func verifiedExistingArtifact(_ artifact: GraphRunArtifact, at url: URL) throws -> Bool {
    guard FileManager.default.fileExists(atPath: url.path) else { return false }
    return try ModelArtifactPin.fileByteCount(url) == artifact.sizeBytes
        && ModelArtifactPin.fileSHA256(url) == artifact.sha256
}

func validateWorker(
    _ worker: WorkflowExecutorProbe,
    for job: WorkflowJobManifest,
    executor: String,
    allowMixedBackend: Bool = false
) throws {
    guard worker.contractVersions.contains(job.contractVersion) else {
        throw ValidationError("Executor '\(executor)' does not support \(job.contractVersion).")
    }
    guard workflowVersion(worker.workerVersion, satisfiesMinimum: job.requirements.minimumMereRunVersion) else {
        throw ValidationError(
            "Executor '\(executor)' reports mere.run \(worker.workerVersion); job requires \(job.requirements.minimumMereRunVersion) or newer."
        )
    }
    let missingKinds = job.requirements.nodeKinds.filter { !worker.nodeKinds.contains($0) }
    guard missingKinds.isEmpty else {
        throw ValidationError("Executor '\(executor)' is missing node kinds: \(missingKinds.joined(separator: ", ")).")
    }
    let missingProviders = job.requirements.providers.filter { !worker.providers.contains($0) }
    guard missingProviders.isEmpty else {
        let descriptions = missingProviders.map { "\($0.id)@\($0.version) [\($0.catalogSHA256)]" }
        throw ValidationError(
            "Executor '\(executor)' is missing exact graph providers: \(descriptions.joined(separator: ", "))."
        )
    }
    let missingModels = job.requirements.modelIDs.filter { !worker.installedModelIDs.contains($0) }
    guard missingModels.isEmpty else {
        throw ValidationError("Executor '\(executor)' is missing models: \(missingModels.joined(separator: ", ")).")
    }
    let missingSecrets = job.requirements.secretNames.filter { !worker.availableSecretNames.contains($0) }
    guard missingSecrets.isEmpty else {
        throw ValidationError(
            "Executor '\(executor)' is missing configured secrets: \(missingSecrets.joined(separator: ", "))."
        )
    }
    let backendAccepted = job.requirements.acceleratorBackends.contains(worker.acceleratorBackend)
        || (allowMixedBackend && worker.acceleratorBackend == "mixed")
    guard backendAccepted else {
        throw ValidationError(
            "Executor '\(executor)' uses \(worker.acceleratorBackend); accepted backends are \(job.requirements.acceleratorBackends.joined(separator: ", "))."
        )
    }
    if let minimum = job.requirements.minimumAcceleratorMemoryBytes,
       worker.memoryBytes < UInt64(minimum) {
        throw ValidationError("Executor '\(executor)' does not have the required accelerator memory.")
    }
    if let minimum = job.requirements.minimumSystemMemoryBytes,
       worker.systemMemoryBytes < UInt64(minimum) {
        throw ValidationError("Executor '\(executor)' does not have the required system memory.")
    }
    if let minimum = job.requirements.minimumCPUCores,
       worker.logicalCPUCores < minimum {
        throw ValidationError("Executor '\(executor)' does not have the required CPU cores.")
    }
    if let minimum = job.requirements.minimumDiskBytes,
       worker.availableDiskBytes.map({ $0 < minimum }) != false {
        throw ValidationError("Executor '\(executor)' does not report the required free disk space.")
    }
    if job.requirements.networkAccess, !worker.networkAccess {
        throw ValidationError("Executor '\(executor)' does not allow required network access.")
    }
}

private func prepareFetchDestination(_ destination: URL, expectedJobID: String) throws {
    var isDirectory: ObjCBool = false
    if FileManager.default.fileExists(atPath: destination.path, isDirectory: &isDirectory) {
        guard isDirectory.boolValue else {
            throw ValidationError("Fetch destination is not a directory: \(destination.path)")
        }
        let jobURL = destination.appendingPathComponent(WorkflowJobManifest.filename)
        guard FileManager.default.fileExists(atPath: jobURL.path) else {
            throw ValidationError("Existing fetch destination is not the local bundle for job \(expectedJobID).")
        }
        let job = try WorkflowBundleCodec.decoder().decode(WorkflowJobManifest.self, from: Data(contentsOf: jobURL))
        guard job.jobID == expectedJobID else {
            throw ValidationError("Existing fetch destination belongs to job \(job.jobID), not \(expectedJobID).")
        }
    } else {
        try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
    }
}

private func confinedFetchURL(root: URL, relativePath: String) throws -> URL {
    guard !relativePath.isEmpty,
          !relativePath.hasPrefix("/"),
          relativePath.split(separator: "/", omittingEmptySubsequences: false).allSatisfy({
              !$0.isEmpty && $0 != "." && $0 != ".."
          }) else {
        throw ValidationError("Remote artifact path is not confined: \(relativePath)")
    }
    return root.appendingPathComponent(relativePath).standardizedFileURL
}

private func encodedPathSegment(_ value: String) throws -> String {
    var allowed = CharacterSet.alphanumerics
    allowed.insert(charactersIn: "-._~")
    guard let encoded = value.addingPercentEncoding(withAllowedCharacters: allowed) else {
        throw ValidationError("Remote artifact name cannot be URL encoded.")
    }
    return encoded
}
