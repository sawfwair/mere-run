import Foundation
import MereRunRelayKit
import XCTest

@testable import MereRunCLI

/// The direct lane's state machine: pairing, the auth gate, the job spool
/// lifecycle, and restart recovery. The HTTP layer is a thin translation on
/// top of this actor, and the full client round trip is exercised against a
/// live `relay serve` in development.
final class LocalRelayServerTests: XCTestCase {
    private var root: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("local-relay-tests-\(UUID().uuidString)", isDirectory: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
    }

    private func makeState(pairingWindowMinutes: Int = 15) async throws -> LocalRelayState {
        try await LocalRelayState(
            rootDirectory: root,
            relayName: "test-relay",
            pairingCode: "123456",
            pairingWindowMinutes: pairingWindowMinutes,
            probeProvider: {
                WorkflowExecutorProbe(
                    schemaVersion: 1,
                    workerVersion: MereRunCLIVersion.current,
                    contractVersions: [WorkflowJobManifest.contractVersion],
                    platform: "macos",
                    architecture: "arm64",
                    acceleratorBackend: "metal",
                    memoryBytes: 8 << 30,
                    systemMemoryBytes: 8 << 30,
                    logicalCPUCores: 8,
                    availableDiskBytes: 100 << 30,
                    networkAccess: true,
                    nodeKinds: ["boolean.value"],
                    installedModelIDs: [],
                    availableSecretNames: [],
                    providers: []
                )
            }
        )
    }

    // MARK: - Pairing

    func testPairingRejectsWrongCodeAndIssuesTokenForRightCode() async throws {
        let state = try await makeState()

        guard case .rejected = try await state.pair(code: "000000", deviceName: "bad") else {
            return XCTFail("A wrong code must be rejected.")
        }
        let authorizedBefore = await state.authorized(bearerToken: "anything")
        XCTAssertFalse(authorizedBefore)

        guard case .paired(let token) = try await state.pair(code: "123-456", deviceName: "phone") else {
            return XCTFail("The dashed form of the right code must pair.")
        }
        XCTAssertGreaterThanOrEqual(token.count, 40)
        let authorized = await state.authorized(bearerToken: token)
        XCTAssertTrue(authorized)
        let stillRejectsGarbage = await state.authorized(bearerToken: String(token.dropLast()))
        XCTAssertFalse(stillRejectsGarbage)
    }

    func testPairingClosesAfterWindowExpires() async throws {
        let state = try await makeState(pairingWindowMinutes: 0)
        guard case .closed = try await state.pair(code: "123456", deviceName: "late") else {
            return XCTFail("An expired window must refuse pairing.")
        }
    }

    func testPairingClosesAfterAttemptLimit() async throws {
        let state = try await makeState()
        for _ in 0..<LocalRelayState.pairingAttemptLimit {
            _ = try await state.pair(code: "999999", deviceName: "guess")
        }
        guard case .closed = try await state.pair(code: "123456", deviceName: "brute") else {
            return XCTFail("The right code after too many failures must find pairing closed.")
        }
    }

    func testPairedDevicesSurviveRestart() async throws {
        var token = ""
        do {
            let state = try await makeState()
            guard case .paired(let issued) = try await state.pair(code: "123456", deviceName: "phone") else {
                return XCTFail("Pairing must succeed.")
            }
            token = issued
        }
        let restarted = try await makeState()
        let authorized = await restarted.authorized(bearerToken: token)
        XCTAssertTrue(authorized, "Tokens must survive a server restart.")
    }

    // MARK: - Job lifecycle

    private func makeCreateRequest(jobID: String = UUID().uuidString) throws -> RelayGraphCreateRequest {
        let graph = WorkflowGraphDocument(
            schemaVersion: WorkflowGraphDocument.schemaVersion,
            kind: WorkflowGraphDocument.kind,
            name: "test-graph",
            inputs: [:],
            nodes: [WorkflowNode(id: "flag", kind: "boolean.value", arguments: ["value": .boolean(true)], dependsOn: nil)],
            outputs: ["value": .reference("nodes.flag.outputs.value")],
            metadata: nil
        )
        let inputs = WorkflowInputsDocument(values: [:])
        let assets = WorkflowAssetManifest(schemaVersion: 1, groups: [])
        let job = WorkflowJobManifest(
            contractVersion: WorkflowJobManifest.contractVersion,
            jobID: jobID,
            createdAt: Date(),
            graphFingerprint: "test",
            inputFingerprint: "test",
            requirements: WorkflowJobRequirements(
                minimumMereRunVersion: "0.1.0",
                nodeKinds: ["boolean.value"],
                modelIDs: [],
                acceleratorBackends: ["cpu", "metal", "cuda", "rocm"],
                minimumAcceleratorMemoryBytes: nil
            ),
            outputs: []
        )
        let encoder = WorkflowBundleCodec.encoder()
        return RelayGraphCreateRequest(
            job: job,
            graph: graph,
            inputs: inputs,
            assets: assets,
            bundleDocuments: [
                WorkflowJobManifest.filename: try encoder.encode(job),
                "graph.json": try encoder.encode(graph),
                "inputs.json": try encoder.encode(inputs),
                WorkflowAssetManifest.filename: try encoder.encode(assets),
            ]
        )
    }

    func testCreateCommitAndQueueFlow() async throws {
        let state = try await makeState()
        let request = try makeCreateRequest()
        let created = try await state.create(request: request)
        XCTAssertEqual(created.state, .planned)
        XCTAssertTrue(created.missingAssetDigests.isEmpty)

        let bundleDir = await state.bundleDirectory(jobID: created.jobID)
        XCTAssertTrue(FileManager.default.fileExists(
            atPath: bundleDir.appendingPathComponent("graph.json").path
        ))

        let queue = await state.makeQueue()
        let committed = try await state.commit(jobID: created.jobID)
        XCTAssertEqual(committed.state, .queued)

        var iterator = queue.makeAsyncIterator()
        let queuedID = await iterator.next()
        XCTAssertEqual(queuedID, created.jobID)

        let began = await state.beginRun(jobID: created.jobID)
        XCTAssertTrue(began)
        await state.finishRun(jobID: created.jobID, state: .finished, error: nil)
        let record = await state.record(jobID: created.jobID)
        XCTAssertEqual(record?.state, .finished)
    }

    func testCommitRefusesMissingAssets() async throws {
        let state = try await makeState()
        var request = try makeCreateRequest()
        let payload = Data("hello".utf8)
        let digest = ModelArtifactPinDigest.sha256(payload)
        let assets = WorkflowAssetManifest(
            schemaVersion: 1,
            groups: [WorkflowAssetGroup(
                name: "photo",
                kind: .asset,
                entries: [WorkflowAssetEntry(
                    path: "assets/sha256/\(digest)",
                    digest: digest,
                    sizeBytes: Int64(payload.count),
                    contentType: "application/octet-stream"
                )]
            )]
        )
        request = RelayGraphCreateRequest(
            job: request.job,
            graph: request.graph,
            inputs: request.inputs,
            assets: assets,
            bundleDocuments: request.bundleDocuments
        )
        let created = try await state.create(request: request)
        XCTAssertEqual(created.missingAssetDigests, [digest])

        do {
            _ = try await state.commit(jobID: created.jobID)
            XCTFail("Commit must refuse while assets are missing.")
        } catch let error as RelayClientError {
            XCTAssertTrue(error.message.contains("missing asset"))
        }

        do {
            try await state.storeAsset(jobID: created.jobID, digest: digest, data: Data("tampered".utf8))
            XCTFail("A digest mismatch must be refused.")
        } catch let error as RelayClientError {
            XCTAssertTrue(error.message.contains("does not match digest"))
        }

        try await state.storeAsset(jobID: created.jobID, digest: digest, data: payload)
        let committed = try await state.commit(jobID: created.jobID)
        XCTAssertEqual(committed.state, .queued)
    }

    func testCancelQueuedJobAndRetry() async throws {
        let state = try await makeState()
        let created = try await state.create(request: try makeCreateRequest())
        _ = try await state.commit(jobID: created.jobID)

        let cancelled = try await state.cancel(jobID: created.jobID)
        XCTAssertEqual(cancelled.state, .cancelled)

        let retried = try await state.retry(jobID: created.jobID)
        XCTAssertEqual(retried.state, .queued)

        let finishedRecord = await state.record(jobID: created.jobID)
        XCTAssertNil(finishedRecord?.error)
    }

    func testRetryRefusesTerminalSuccess() async throws {
        let state = try await makeState()
        let created = try await state.create(request: try makeCreateRequest())
        _ = try await state.commit(jobID: created.jobID)
        _ = await state.beginRun(jobID: created.jobID)
        await state.finishRun(jobID: created.jobID, state: .finished, error: nil)

        do {
            _ = try await state.retry(jobID: created.jobID)
            XCTFail("A finished job must not retry.")
        } catch let error as RelayClientError {
            XCTAssertTrue(error.message.contains("only failed or cancelled"))
        }
    }

    func testInterruptedJobRecoversAsFailed() async throws {
        var jobID = ""
        do {
            let state = try await makeState()
            let created = try await state.create(request: try makeCreateRequest())
            jobID = created.jobID
            _ = try await state.commit(jobID: jobID)
        }
        let restarted = try await makeState()
        let record = await restarted.record(jobID: jobID)
        XCTAssertEqual(record?.state, .failed)
        XCTAssertTrue(record?.error?.contains("stopped before") == true)
        let retried = try await restarted.retry(jobID: jobID)
        XCTAssertEqual(retried.state, .queued)
    }

    // MARK: - Fleet

    func testFleetSnapshotDescribesThisMachine() async throws {
        let state = try await makeState()
        let snapshot = await state.fleetSnapshot()
        XCTAssertEqual(snapshot.summary.totalNodes, 1)
        XCTAssertEqual(snapshot.summary.onlineNodes, 1)
        XCTAssertEqual(snapshot.nodes.first?.status, "online")
        XCTAssertEqual(snapshot.nodes.first?.policy.displayName, "test-relay")
        XCTAssertFalse(snapshot.nodes.first?.capabilities.graphWorker?.nodeKinds.isEmpty ?? true)
    }

    // MARK: - Wire types

    func testDiscoveryDocumentRoundTrips() throws {
        let document = LocalRelayDiscoveryDocument(
            schemaVersion: 1,
            kind: LocalRelayDiscoveryDocument.kind,
            authMode: LocalRelayDiscoveryDocument.authMode,
            relayName: "lab",
            contractVersions: [WorkflowJobManifest.contractVersion]
        )
        let data = try WorkflowBundleCodec.encoder().encode(document)
        let text = String(decoding: data, as: UTF8.self)
        XCTAssertTrue(text.contains("\"auth_mode\""))
        XCTAssertTrue(text.contains("\"graph_contract_versions\""))
        let decoded = try WorkflowBundleCodec.decoder().decode(LocalRelayDiscoveryDocument.self, from: data)
        XCTAssertEqual(decoded, document)
    }

    func testLocallyIssuedTokenSetNeverExpires() {
        let tokenSet = RelayOAuthTokenSet(
            accessToken: "opaque-not-a-jwt",
            refreshToken: nil,
            tokenType: "Bearer",
            expiresIn: nil,
            obtainedAtEpochSeconds: Int64(Date().timeIntervalSince1970)
        )
        XCTAssertTrue(tokenSet.isFresh(now: Int64(Date().timeIntervalSince1970) + 10_000_000))
    }
}
