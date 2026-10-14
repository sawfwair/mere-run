import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif
import XCTest
@testable import MereRunRelayKit

final class WorkflowBatchHTTPTests: XCTestCase {
    func testClientCreatesCommitsAndResumesExactBundlesThroughTheBatchHTTPContract() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root); BatchHTTPProtocol.clear() }
        let graph = try JSONDecoder().decode(WorkflowGraphDocument.self, from: Data(#"""
        {"schema_version":1,"kind":"mere.run/workflow-graph","name":"http-fixture",
         "inputs":{"prompt":{"type":"string"}},"nodes":[{"id":"text","kind":"text.value",
         "arguments":{"value":{"$ref":"inputs.prompt"}}}],"outputs":{"text":{"$ref":"nodes.text.outputs.text"}}}
        """#.utf8))
        let manifest = try WorkflowBatchMaterializer.materialize(graph: graph,
            items: [.init(key: "row", inputs: .init(values: ["prompt": .string("fixture")]))],
            destination: root, environment: .portable)
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [BatchHTTPProtocol.self]
        let session = URLSession(configuration: config)
        defer { session.invalidateAndCancel() }
        let profile = WorkflowExecutorProfile(name: "batch", kind: .relay, destination: nil, remoteRoot: nil,
            port: nil, identityFile: nil, mereRunPath: nil, url: "https://batch.example", tokenFile: nil)
        let executor = RelayWorkflowExecutor(profile: profile, credentialStorage: BatchCredentials(), session: session)
        BatchHTTPProtocol.configure(batchID: manifest.batchID, jobID: manifest.items[0].jobID)
        let committed = try await executor.submitBatch(directory: root)
        XCTAssertEqual(committed.batchID, manifest.batchID)
        XCTAssertEqual(committed.state, .running)
        let replay = try await executor.submitBatch(directory: root)
        XCTAssertEqual(replay.items[0].jobID, committed.items[0].jobID)
        let calls = BatchHTTPProtocol.requests()
        XCTAssertEqual(calls.map(\.path), ["/api/graph-batches", "/api/graph-batches/\(manifest.batchID)/commit",
                                           "/api/graph-batches", "/api/graph-batches/\(manifest.batchID)/commit"])
        XCTAssertEqual(calls[0].body, calls[2].body)
        let submitted = try JSONDecoder().decode(Submitted.self, from: XCTUnwrap(calls[0].body))
        let exact = try Data(contentsOf: root.appendingPathComponent(manifest.items[0].bundleDirectory).appendingPathComponent("graph.json"))
        XCTAssertEqual(submitted.items[0].submission.bundleDocuments["graph.json"], exact)
    }

    private struct Submitted: Decodable {
        let items: [Item]
        struct Item: Decodable { let submission: Submission }
        struct Submission: Decodable {
            let bundleDocuments: [String: Data]
            enum CodingKeys: String, CodingKey { case bundleDocuments = "bundle_documents" }
        }
    }
}

private struct BatchCredentials: RelayCredentialStorage {
    func load() throws -> RelayOAuthTokenSet? {
        .init(accessToken: "batch-fixture", refreshToken: nil, tokenType: "Bearer", expiresIn: nil, obtainedAtEpochSeconds: nil)
    }
    func save(_ tokenSet: RelayOAuthTokenSet) throws {}
    func clear() throws {}
}

private final class BatchHTTPProtocol: URLProtocol, @unchecked Sendable {
    struct Call { let path: String; let body: Data? }
    private static let lock = NSLock()
    nonisolated(unsafe) private static var calls: [Call] = []
    nonisolated(unsafe) private static var batchID = ""
    nonisolated(unsafe) private static var jobID = ""

    static func configure(batchID: String, jobID: String) {
        lock.lock(); defer { lock.unlock() }
        self.batchID = batchID; self.jobID = jobID; calls = []
    }
    static func requests() -> [Call] { lock.lock(); defer { lock.unlock() }; return calls }
    static func clear() { configure(batchID: "", jobID: "") }
    override class func canInit(with request: URLRequest) -> Bool { request.url?.host == "batch.example" }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    private static func bodyData(_ request: URLRequest) -> Data? {
        if let body = request.httpBody { return body }
        guard let stream = request.httpBodyStream else { return nil }
        stream.open(); defer { stream.close() }
        var result = Data()
        var buffer = [UInt8](repeating: 0, count: 4096)
        while true {
            let count = stream.read(&buffer, maxLength: buffer.count)
            if count <= 0 { return result }
            result.append(contentsOf: buffer.prefix(count))
        }
    }
    override func startLoading() {
        Self.lock.lock()
        let path = request.url?.path ?? ""
        Self.calls.append(Call(path: path, body: Self.bodyData(request)))
        let id = Self.batchID; let child = Self.jobID
        Self.lock.unlock()
        let running = path.hasSuffix("/commit")
        let state = running ? "running" : "planned"
        let itemState = running ? "queued" : "pending"
        let pending = running ? 0 : 1
        let queued = running ? 1 : 0
        let payload = """
        {"schema_version":1,"batch_id":"\(id)","request_sha256":"\(String(repeating: "a", count: 64))",
        "state":"\(state)","max_in_flight":8,"max_attempts":1,"item_count":1,
        "created_at":"2026-10-14T12:00:00.123Z","updated_at":"2026-10-14T12:00:00.123Z","error":null,
        "counts":{"pending":\(pending),"queued":\(queued),"running":0,"finished":0,"failed":0,"cancelled":0},
        "missing_asset_digests":[],"next_cursor":null,"items":[{"key":"row","ordinal":0,"job_id":"\(child)",
        "state":"\(itemState)","attempt":0,"error":null,"artifacts":[]}]}
        """
        guard let url = request.url, let response = HTTPURLResponse(url: url, statusCode: 200, httpVersion: nil,
            headerFields: ["Content-Type": "application/json"]) else { return }
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: Data(payload.utf8))
        client?.urlProtocolDidFinishLoading(self)
    }
    override func stopLoading() {}
}
