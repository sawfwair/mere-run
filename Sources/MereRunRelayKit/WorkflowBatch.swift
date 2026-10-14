import Foundation

public struct WorkflowBatchInput: Codable, Equatable, Sendable {
    public let key: String
    public let inputs: WorkflowInputsDocument

    public init(key: String, inputs: WorkflowInputsDocument) {
        self.key = key
        self.inputs = inputs
    }
}

public struct WorkflowBatchManifest: Codable, Equatable, Sendable {
    public static let filename = "batch.json"
    public let schemaVersion: Int
    public let batchID: String
    public let idempotencyKey: String
    public let sourceFingerprint: String
    public let createdAt: Date
    public let maxInFlight: Int
    public let maxAttempts: Int
    public let items: [Item]

    public struct Item: Codable, Equatable, Sendable {
        public let key: String
        public let jobID: String
        public let seed: Int64
        public let bundleDirectory: String

        enum CodingKeys: String, CodingKey {
            case key, seed
            case jobID = "job_id"
            case bundleDirectory = "bundle_directory"
        }
    }

    enum CodingKeys: String, CodingKey {
        case items
        case schemaVersion = "schema_version"
        case batchID = "batch_id"
        case idempotencyKey = "idempotency_key"
        case sourceFingerprint = "source_fingerprint"
        case createdAt = "created_at"
        case maxInFlight = "max_in_flight"
        case maxAttempts = "max_attempts"
    }
}

/// Pins child identities and seeds before creating any bundle. Replays never resample.
public enum WorkflowBatchMaterializer {
    public static func materialize(
        graph: WorkflowGraphDocument,
        items: [WorkflowBatchInput],
        destination: URL,
        environment: WorkflowMaterializationEnvironment,
        maxInFlight: Int = 8,
        maxAttempts: Int = 1
    ) throws -> WorkflowBatchManifest {
        try validate(items: items, maxInFlight: maxInFlight, maxAttempts: maxAttempts)
        let fingerprint = try WorkflowBundleCodec.hash(Source(graph: graph, items: items))
        let manifestURL = destination.appendingPathComponent(WorkflowBatchManifest.filename)
        let manifest: WorkflowBatchManifest
        if FileManager.default.fileExists(atPath: manifestURL.path) {
            manifest = try WorkflowBundleCodec.decoder().decode(WorkflowBatchManifest.self, from: Data(contentsOf: manifestURL))
            guard manifest.schemaVersion == 1, manifest.sourceFingerprint == fingerprint,
                  manifest.maxInFlight == maxInFlight, manifest.maxAttempts == maxAttempts,
                  manifest.items.map(\.key) == items.map(\.key) else {
                throw RelayClientError("Batch directory belongs to a different immutable batch request.")
            }
        } else {
            if FileManager.default.fileExists(atPath: destination.path),
               !(try FileManager.default.contentsOfDirectory(atPath: destination.path)).isEmpty {
                throw RelayClientError("Batch destination must be empty or contain this batch's manifest.")
            }
            try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
            let batchID = UUID().uuidString.lowercased()
            manifest = WorkflowBatchManifest(
                schemaVersion: 1, batchID: batchID, idempotencyKey: batchID,
                sourceFingerprint: fingerprint, createdAt: Date(timeIntervalSince1970: floor(Date().timeIntervalSince1970)),
                maxInFlight: maxInFlight, maxAttempts: maxAttempts,
                items: items.enumerated().map { ordinal, item in
                    .init(key: item.key, jobID: UUID().uuidString.lowercased(),
                          seed: Int64.random(in: 0...Int64.max), bundleDirectory: "items/\(ordinal)")
                }
            )
            try WorkflowBundleCodec.write(manifest, to: manifestURL)
        }
        for (input, item) in zip(items, manifest.items) {
            let directory = try confinedFetchURL(root: destination, relativePath: item.bundleDirectory)
            if try reuseBundle(directory, item: item, pluginNodes: environment.pluginNodes) { continue }
            guard let jobID = UUID(uuidString: item.jobID) else { throw RelayClientError("Batch child job ID is invalid.") }
            _ = try WorkflowBundleMaterializer(
                graph: graph, suppliedInputs: input.inputs, destination: directory,
                environment: environment, now: { manifest.createdAt }, jobID: { jobID }, seed: { item.seed }
            ).materialize()
        }
        return manifest
    }

    private struct Source: Encodable {
        let graph: WorkflowGraphDocument
        let items: [WorkflowBatchInput]
    }

    private static func validate(items: [WorkflowBatchInput], maxInFlight: Int, maxAttempts: Int) throws {
        guard (1...1000).contains(items.count), (1...20).contains(maxInFlight), (1...4).contains(maxAttempts) else {
            throw RelayClientError("Batch requires 1–1000 items, 1–20 in-flight jobs, and 1–4 attempts.")
        }
        guard Set(items.map(\.key)).count == items.count,
              items.allSatisfy({ !$0.key.isEmpty && $0.key.count <= 160 && $0.key.range(of: "^[A-Za-z0-9_.:-]+$", options: .regularExpression) != nil }) else {
            throw RelayClientError("Batch item keys must be unique and contain only letters, numbers, dots, underscores, colons, or hyphens.")
        }
    }

    private static func reuseBundle(_ directory: URL, item: WorkflowBatchManifest.Item, pluginNodes: [WorkflowNodeCatalogEntry]) throws -> Bool {
        guard FileManager.default.fileExists(atPath: directory.path) else { return false }
        let manifestURL = directory.appendingPathComponent(WorkflowJobManifest.filename)
        guard FileManager.default.fileExists(atPath: manifestURL.path) else {
            // Preserve interrupted preparation before reconstructing the owned child directory.
            try FileManager.default.moveItem(at: directory, to: directory.appendingPathExtension("incomplete-\(UUID().uuidString)"))
            return false
        }
        let bundle = try verifiedPortableWorkflowBundle(at: directory, pluginNodes: pluginNodes)
        guard bundle.job.jobID == item.jobID else { throw RelayClientError("Batch child directory has a different job ID.") }
        return true
    }
}

public struct RelayGraphBatch: Codable, Equatable, Sendable {
    public let schemaVersion: Int
    public let batchID: String
    public let state: State
    public let requestSHA256: String
    public let maxInFlight: Int
    public let maxAttempts: Int
    public let itemCount: Int
    public let createdAt: Date
    public let updatedAt: Date
    public let error: String?
    public let counts: Counts
    public let items: [Item]
    public let missingAssetDigests: [String]?
    public let nextCursor: Int?

    public enum State: String, Codable, Sendable {
        case planned, running, finished, failed, cancelled
        public var isTerminal: Bool { self == .finished || self == .failed || self == .cancelled }
    }
    public struct Counts: Codable, Equatable, Sendable {
        public let pending, queued, running, finished, failed, cancelled: Int
    }
    public enum ItemState: String, Codable, Sendable {
        case pending, planned, preflighting, queued, assigned, running, finished, failed, cancelled
    }
    public struct Item: Codable, Equatable, Sendable {
        public let key: String
        public let ordinal: Int
        public let jobID: String
        public let state: ItemState
        public let attempt: Int
        public let error: String?
        public let artifacts: [GraphRunArtifact]
        public let placement: WorkflowGraphPlacement?

        enum CodingKeys: String, CodingKey {
            case key, ordinal, state, attempt, error, artifacts, placement
            case jobID = "job_id"
        }
    }
    enum CodingKeys: String, CodingKey {
        case state, counts, items, error
        case schemaVersion = "schema_version"
        case batchID = "batch_id"
        case requestSHA256 = "request_sha256"
        case maxInFlight = "max_in_flight"
        case maxAttempts = "max_attempts"
        case itemCount = "item_count"
        case createdAt = "created_at"
        case updatedAt = "updated_at"
        case missingAssetDigests = "missing_asset_digests"
        case nextCursor = "next_cursor"
    }
}

package struct RelayBatchCreateRequest: Encodable {
    let schemaVersion = 1
    let batchID: String
    let idempotencyKey: String
    let maxInFlight: Int
    let maxAttempts: Int
    let items: [Item]
    struct Item: Encodable {
        let key: String
        let submission: RelayGraphCreateRequest
    }
    enum CodingKeys: String, CodingKey {
        case items
        case schemaVersion = "schema_version"
        case batchID = "batch_id"
        case idempotencyKey = "idempotency_key"
        case maxInFlight = "max_in_flight"
        case maxAttempts = "max_attempts"
    }
}

public extension RelayWorkflowExecutor {
    func batch(batchID: String) async throws -> RelayGraphBatch {
        try await batchRequest(batchID: batchID, method: "GET")
    }
    func cancelBatch(batchID: String) async throws -> RelayGraphBatch {
        try await batchRequest(batchID: batchID, method: "DELETE")
    }
    func retryBatch(batchID: String, retryKey: UUID = UUID()) async throws -> RelayGraphBatch {
        try await batchRequest(batchID: batchID, action: "/retry", method: "POST",
                               body: WorkflowBundleCodec.encoder().encode(BatchRetryRequest(retryKey: retryKey.uuidString.lowercased())))
    }

    func submitBatch(directory: URL, pluginNodes: [WorkflowNodeCatalogEntry] = []) async throws -> RelayGraphBatch {
        let manifest = try WorkflowBundleCodec.decoder().decode(
            WorkflowBatchManifest.self, from: Data(contentsOf: directory.appendingPathComponent(WorkflowBatchManifest.filename))
        )
        guard manifest.schemaVersion == 1, UUID(uuidString: manifest.batchID) != nil,
              (1...1000).contains(manifest.items.count), (1...20).contains(manifest.maxInFlight),
              (1...4).contains(manifest.maxAttempts), Set(manifest.items.map(\.key)).count == manifest.items.count,
              Set(manifest.items.map(\.jobID)).count == manifest.items.count else {
            throw RelayClientError("Invalid batch manifest version, bounds, or child identities.")
        }
        let items = try manifest.items.map { item in
            let root = try confinedFetchURL(root: directory, relativePath: item.bundleDirectory)
            let bundle = try verifiedPortableWorkflowBundle(at: root, pluginNodes: pluginNodes)
            guard bundle.job.jobID == item.jobID else { throw RelayClientError("Batch child identity mismatch.") }
            return RelayBatchCreateRequest.Item(key: item.key, submission: try RelayGraphCreateRequest(bundleDirectory: root))
        }
        let body = RelayBatchCreateRequest(batchID: manifest.batchID, idempotencyKey: manifest.idempotencyKey,
                                          maxInFlight: manifest.maxInFlight, maxAttempts: manifest.maxAttempts, items: items)
        let created = try WorkflowBundleCodec.decoder().decode(RelayGraphBatch.self, from: await request(
            path: "/api/graph-batches", method: "POST", body: WorkflowBundleCodec.encoder().encode(body), transientRetryAttempts: 3
        ))
        guard created.batchID == manifest.batchID else { throw RelayClientError("Relay returned a different batch identity.") }
        for digest in created.missingAssetDigests ?? [] {
            guard let index = items.firstIndex(where: { $0.submission.assets.groups.contains(where: { $0.entries.contains(where: { $0.digest == digest }) }) }) else {
                throw RelayClientError("Relay requested an undeclared batch asset.")
            }
            let root = try confinedFetchURL(root: directory, relativePath: manifest.items[index].bundleDirectory)
            let data = try Data(contentsOf: root.appendingPathComponent("assets/sha256/\(digest)"))
            guard ModelArtifactPinDigest.sha256(data) == digest else { throw RelayClientError("Batch asset digest mismatch.") }
            _ = try await request(path: "/api/graph-batches/\(manifest.batchID)/assets/\(digest)", method: "PUT",
                                  contentType: "application/octet-stream", body: data, transientRetryAttempts: 3)
        }
        return try await batchRequest(batchID: manifest.batchID, action: "/commit", method: "POST")
    }

    private func batchRequest(batchID: String, action: String = "", method: String, body: Data? = nil) async throws -> RelayGraphBatch {
        guard UUID(uuidString: batchID) != nil else { throw RelayClientError("Invalid batch ID.") }
        let data = try await request(path: "/api/graph-batches/\(batchID)\(action)", method: method, body: body, transientRetryAttempts: 3)
        var result = try WorkflowBundleCodec.decoder().decode(RelayGraphBatch.self, from: data)
        let fingerprint = result.requestSHA256
        let count = result.itemCount
        guard result.schemaVersion == 1, result.batchID == batchID, (1...1000).contains(count) else { throw RelayClientError("Invalid batch identity or size.") }
        var items = result.items
        while let cursor = result.nextCursor {
            guard cursor == items.count, cursor < count else { throw RelayClientError("Invalid batch page cursor.") }
            let page = try await request(path: "/api/graph-batches/\(batchID)?cursor=\(cursor)", method: "GET", transientRetryAttempts: 3)
            result = try WorkflowBundleCodec.decoder().decode(RelayGraphBatch.self, from: page)
            guard result.batchID == batchID, result.requestSHA256 == fingerprint, result.itemCount == count,
                  !result.items.isEmpty else { throw RelayClientError("Invalid batch result page.") }
            items.append(contentsOf: result.items)
        }
        guard items.count == count, Set(items.map(\.jobID)).count == count,
              Set(items.map(\.key)).count == count, items.enumerated().allSatisfy({ $0.offset == $0.element.ordinal }) else {
            throw RelayClientError("Batch result index is incomplete or duplicated.")
        }
        return RelayGraphBatch(schemaVersion: result.schemaVersion, batchID: result.batchID, state: result.state,
                               requestSHA256: result.requestSHA256, maxInFlight: result.maxInFlight, maxAttempts: result.maxAttempts,
                               itemCount: count, createdAt: result.createdAt, updatedAt: result.updatedAt, error: result.error,
                               counts: result.counts, items: items, missingAssetDigests: result.missingAssetDigests, nextCursor: nil)
    }
}

package extension RelayGraphCreateRequest {
    init(bundleDirectory: URL) throws {
        let decoder = WorkflowBundleCodec.decoder()
        self.init(
            job: try decoder.decode(WorkflowJobManifest.self, from: Data(contentsOf: bundleDirectory.appendingPathComponent("job.json"))),
            graph: try WorkflowGraphDocument.load(from: bundleDirectory.appendingPathComponent("graph.json")),
            inputs: try WorkflowInputsDocument.load(from: bundleDirectory.appendingPathComponent("inputs.json")),
            assets: try decoder.decode(WorkflowAssetManifest.self, from: Data(contentsOf: bundleDirectory.appendingPathComponent("assets.json"))),
            bundleDocuments: try Dictionary(uniqueKeysWithValues: ["job.json", "graph.json", "inputs.json", "assets.json"].map {
                ($0, try Data(contentsOf: bundleDirectory.appendingPathComponent($0)))
            })
        )
    }
}

private struct BatchRetryRequest: Encodable {
    let retryKey: String
    enum CodingKeys: String, CodingKey { case retryKey = "retry_key" }
}

public struct RelayGraphBatchSummary: Codable, Equatable, Sendable {
    public let schemaVersion: Int
    public let batchID: String
    public let requestSHA256: String
    public let state: RelayGraphBatch.State
    public let maxInFlight: Int
    public let maxAttempts: Int
    public let itemCount: Int
    public let createdAt: Date
    public let updatedAt: Date
    public let error: String?
    enum CodingKeys: String, CodingKey {
        case state, error
        case schemaVersion = "schema_version"
        case batchID = "batch_id"
        case requestSHA256 = "request_sha256"
        case maxInFlight = "max_in_flight"
        case maxAttempts = "max_attempts"
        case itemCount = "item_count"
        case createdAt = "created_at"
        case updatedAt = "updated_at"
    }
}

private struct RelayGraphBatchList: Decodable { let batches: [RelayGraphBatchSummary] }

public extension RelayWorkflowExecutor {
    func listBatches() async throws -> [RelayGraphBatchSummary] {
        let data = try await request(path: "/api/graph-batches", method: "GET")
        return try WorkflowBundleCodec.decoder().decode(RelayGraphBatchList.self, from: data).batches
    }
}
