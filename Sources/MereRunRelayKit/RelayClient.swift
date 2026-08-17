import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

/// HTTP client for the relay graph-job and fleet API.
///
/// Submission stays with the CLI, which owns bundle materialization; every
/// other operation — probe, fleet, inspect, events, cancel, retry, list, and
/// verified artifact fetch — lives here and needs no local runtime.
public struct RelayWorkflowExecutor: Sendable {
    public let profile: WorkflowExecutorProfile

    public init(profile: WorkflowExecutorProfile) {
        self.profile = profile
    }

    public func probe() async throws -> WorkflowExecutorProbe {
        let data = try await request(path: "/api/graph-jobs/capabilities", method: "GET")
        return try JSONDecoder().decode(WorkflowExecutorProbe.self, from: data)
    }

    public func fleet() async throws -> RelayFleetSnapshot {
        let data = try await request(path: "/api/fleet", method: "GET")
        return try WorkflowBundleCodec.decoder().decode(RelayFleetSnapshot.self, from: data)
    }

    public func refreshNode(deviceID: String) async throws -> RelayFleetRefreshResult {
        let encodedDeviceID = try encodedPathSegment(deviceID)
        let data = try await request(
            path: "/api/fleet/nodes/\(encodedDeviceID)/refresh",
            method: "POST",
            body: Data("{}".utf8)
        )
        return try WorkflowBundleCodec.decoder().decode(RelayFleetRefreshResult.self, from: data)
    }

    public func configureNode(deviceID: String, patch: RelayFleetNodePolicyPatch) async throws -> RelayFleetNode {
        let encodedDeviceID = try encodedPathSegment(deviceID)
        let data = try await request(
            path: "/api/fleet/nodes/\(encodedDeviceID)",
            method: "PATCH",
            body: try WorkflowBundleCodec.encoder().encode(patch)
        )
        return try WorkflowBundleCodec.decoder().decode(RelayFleetNode.self, from: data)
    }

    public func inspect(jobID: String) async throws -> WorkflowRemoteJob {
        let data = try await request(path: "/api/graph-jobs/\(jobID)", method: "GET")
        return try WorkflowBundleCodec.decoder()
            .decode(RelayGraphJobResponse.self, from: data)
            .remoteJob(profile: profile.name, localRunDirectory: nil)
    }

    public func events(jobID: String) async throws -> String {
        let data = try await request(path: "/api/graph-jobs/\(jobID)/events", method: "GET")
        return String(decoding: data, as: UTF8.self)
    }

    public func cancel(jobID: String) async throws -> WorkflowRemoteJob {
        let data = try await request(path: "/api/graph-jobs/\(jobID)", method: "DELETE")
        return try WorkflowBundleCodec.decoder()
            .decode(RelayGraphJobResponse.self, from: data)
            .remoteJob(profile: profile.name, localRunDirectory: nil)
    }

    public func retry(jobID: String) async throws -> WorkflowRemoteJob {
        let data = try await request(
            path: "/api/graph-jobs/\(jobID)/retry",
            method: "POST",
            body: Data("{}".utf8)
        )
        return try WorkflowBundleCodec.decoder()
            .decode(RelayGraphJobResponse.self, from: data)
            .remoteJob(profile: profile.name, localRunDirectory: nil)
    }

    public func list(limit: Int) async throws -> [WorkflowRemoteJob] {
        let data = try await request(path: "/api/graph-jobs?limit=\(limit)", method: "GET")
        let response = try WorkflowBundleCodec.decoder().decode(RelayGraphJobListResponse.self, from: data)
        return response.jobs.map { $0.remoteJob(profile: profile.name, localRunDirectory: nil) }
    }

    public func fetch(
        jobID: String,
        into destination: URL,
        allArtifacts: Bool,
        artifactNames: Set<String> = []
    ) async throws -> WorkflowRemoteJob {
        let job = try await inspect(jobID: jobID)
        guard job.state == .finished else {
            throw RelayClientError("Relay job \(jobID) is not finished.")
        }
        try prepareFetchDestination(destination, expectedJobID: jobID)
        let manifestData = try await request(path: "/api/graph-jobs/\(jobID)/run-manifest", method: "GET")
        var manifest = try WorkflowBundleCodec.decoder().decode(GraphRunManifest.self, from: manifestData)
        guard manifest.jobID == jobID else {
            throw RelayClientError("Relay returned a run manifest for a different job.")
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
                throw RelayClientError("Fetched relay artifact failed verification: \(artifact.name)")
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

    package func request(
        path: String,
        method: String,
        contentType: String = "application/json",
        body: Data? = nil,
        authorize: Bool = true,
        transientRetryAttempts: Int = 1
    ) async throws -> Data {
        guard let baseURL = profile.url, let url = URL(string: "\(baseURL)\(path)") else {
            throw RelayClientError("Relay executor profile has an invalid URL.")
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
            throw RelayClientError("Relay request failed with HTTP \(status): \(detail)")
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
            throw RelayClientError("Relay returned a non-HTTP response.")
        }
        return (data, http)
    }
}

package struct RelayGraphCreateResponse: Codable {
    package let jobID: String
    package let state: GraphRunState
    package let missingAssetDigests: [String]

    enum CodingKeys: String, CodingKey {
        case jobID = "job_id"
        case state
        case missingAssetDigests = "missing_asset_digests"
    }
}

package struct RelayGraphJobResponse: Codable {
    package let jobID: String
    package let state: GraphRunState
    package let createdAt: Date?
    package let updatedAt: Date?
    package let artifacts: [GraphRunArtifact]
    package let error: String?
    package let placement: WorkflowGraphPlacement?
    package let metrics: WorkflowGraphExecutionMetrics?

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

    package func remoteJob(profile: String, localRunDirectory: String?) -> WorkflowRemoteJob {
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

package struct RelayGraphJobListResponse: Codable {
    package let jobs: [RelayGraphJobResponse]
}

package func prepareFetchDestination(_ destination: URL, expectedJobID: String) throws {
    var isDirectory: ObjCBool = false
    if FileManager.default.fileExists(atPath: destination.path, isDirectory: &isDirectory) {
        guard isDirectory.boolValue else {
            throw RelayClientError("Fetch destination is not a directory: \(destination.path)")
        }
        let jobURL = destination.appendingPathComponent(WorkflowJobManifest.filename)
        guard FileManager.default.fileExists(atPath: jobURL.path) else {
            throw RelayClientError("Existing fetch destination is not the local bundle for job \(expectedJobID).")
        }
        let job = try WorkflowBundleCodec.decoder().decode(WorkflowJobManifest.self, from: Data(contentsOf: jobURL))
        guard job.jobID == expectedJobID else {
            throw RelayClientError("Existing fetch destination belongs to job \(job.jobID), not \(expectedJobID).")
        }
    } else {
        try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
    }
}

package func selectedArtifacts(
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
        throw RelayClientError("Remote run does not contain artifacts: \(missing.joined(separator: ", ")).")
    }
    return selected
}

package func verifiedExistingArtifact(_ artifact: GraphRunArtifact, at url: URL) throws -> Bool {
    guard FileManager.default.fileExists(atPath: url.path) else { return false }
    return try ModelArtifactPinDigest.fileByteCount(url) == artifact.sizeBytes
        && ModelArtifactPinDigest.fileSHA256(url) == artifact.sha256
}

package func verifyFetchedArtifacts(_ artifacts: [GraphRunArtifact], root: URL) throws {
    for artifact in artifacts {
        let localURL = try confinedFetchURL(root: root, relativePath: artifact.path)
        guard FileManager.default.fileExists(atPath: localURL.path),
              try ModelArtifactPinDigest.fileByteCount(localURL) == artifact.sizeBytes,
              try ModelArtifactPinDigest.fileSHA256(localURL) == artifact.sha256 else {
            throw RelayClientError("Fetched artifact failed verification: \(artifact.name)")
        }
    }
}

package func confinedFetchURL(root: URL, relativePath: String) throws -> URL {
    guard !relativePath.isEmpty,
          !relativePath.hasPrefix("/"),
          relativePath.split(separator: "/", omittingEmptySubsequences: false).allSatisfy({
              !$0.isEmpty && $0 != "." && $0 != ".."
          }) else {
        throw RelayClientError("Remote artifact path is not confined: \(relativePath)")
    }
    return root.appendingPathComponent(relativePath).standardizedFileURL
}

package func encodedPathSegment(_ value: String) throws -> String {
    var allowed = CharacterSet.alphanumerics
    allowed.insert(charactersIn: "-._~")
    guard let encoded = value.addingPercentEncoding(withAllowedCharacters: allowed) else {
        throw RelayClientError("Remote artifact name cannot be URL encoded.")
    }
    return encoded
}
