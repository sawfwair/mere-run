import Foundation
import XCTest
import MereRunExecution
import MereRunRelayKit
@testable import MereRunCLI

final class WorkflowInterruptionTests: XCTestCase {
    func testAbandonedWorkerRepairsOnlyIncompleteTailAndPreservesCompletedNode() throws {
        let directory = try fixture()
        let before = try read(directory)
        let event = GraphRunEvent(sequence: 0, createdAt: Date(), type: "run_started", state: .running, nodeID: nil, message: nil)
        let tail = Data(#"{"sequence":1,"type":"node_"#.utf8)
        let eventsURL = directory.appendingPathComponent("events.jsonl")
        try (WorkflowBundleCodec.lineEncoder().encode(event) + Data([10]) + tail).write(to: eventsURL)
        let recovered = try WorkflowRunRecovery.inspect(at: directory, now: Date(timeIntervalSince1970: 1_800_000_000))
        XCTAssertEqual(recovered.state, .failed)
        XCTAssertNotNil(recovered.interruptedAt)
        XCTAssertEqual(recovered.nodes[0], before.nodes[0])
        XCTAssertEqual(recovered.nodes[1].state, .failed)
        XCTAssertNotNil(recovered.nodes[1].completedAt)
        XCTAssertEqual(recovered.outputs, before.outputs)
        let fragment = try XCTUnwrap(FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .first { $0.pathExtension == "fragment" })
        XCTAssertEqual(try Data(contentsOf: fragment), tail)
        let events = try String(contentsOf: eventsURL, encoding: .utf8).split(separator: "\n")
            .map { try WorkflowBundleCodec.decoder().decode(GraphRunEvent.self, from: Data($0.utf8)) }
        XCTAssertEqual(events.map(\.sequence), [0, 1])
        XCTAssertEqual(events.last?.type, "run_interrupted")
        let preserved = try Data(contentsOf: eventsURL)
        XCTAssertEqual(try WorkflowRunRecovery.inspect(at: directory), recovered)
        XCTAssertEqual(try Data(contentsOf: eventsURL), preserved)
    }

    func testActiveLeaseLiveChildAndRemoteOwnershipPreventRecovery() throws {
        let directory = try fixture()
        let lease = try XCTUnwrap(RunDirectoryLease.acquire(in: directory, filename: ".workflow-run.lock"))
        XCTAssertEqual(try WorkflowRunRecovery.inspect(at: directory).state, .running)
        lease.release()
        try WorkflowChildProcessRegistry.register(ProcessInfo.processInfo.processIdentifier, in: directory)
        XCTAssertEqual(try WorkflowRunRecovery.inspect(at: directory).state, .running)
        try WorkflowChildProcessRegistry.clear(in: directory)
        var manifest = try read(directory)
        for (kind, state) in [("ssh", GraphRunState.running), ("local", .queued), ("worker", .assigned)] {
            manifest.executor = .init(kind: kind, profile: nil, jobReference: nil)
            manifest.state = state
            try WorkflowBundleCodec.write(manifest, to: directory.appendingPathComponent("run.json"))
            XCTAssertEqual(try WorkflowRunRecovery.inspect(at: directory), manifest)
        }
    }

    func testUnsupportedVersionAndInteriorEventCorruptionAreNotRewritten() throws {
        let directory = try fixture(contract: "mere.run/graph-run.v999")
        let manifestURL = directory.appendingPathComponent("run.json")
        let before = try Data(contentsOf: manifestURL)
        XCTAssertThrowsError(try WorkflowRunRecovery.inspect(at: directory))
        XCTAssertEqual(try Data(contentsOf: manifestURL), before)
        let valid = try fixture()
        let validURL = valid.appendingPathComponent("run.json")
        let validBefore = try Data(contentsOf: validURL)
        let log = Data("incomplete\n{\"sequence\":1".utf8)
        let eventsURL = valid.appendingPathComponent("events.jsonl")
        try log.write(to: eventsURL)
        XCTAssertThrowsError(try WorkflowRunRecovery.inspect(at: valid))
        XCTAssertEqual(try Data(contentsOf: validURL), validBefore)
        XCTAssertEqual(try Data(contentsOf: eventsURL), log)
    }

    func testCompleteFinalEventWithoutNewlineIsRetained() throws {
        let directory = try fixture()
        let event = GraphRunEvent(sequence: 0, createdAt: Date(), type: "run_started", state: .running, nodeID: nil, message: nil)
        let eventsURL = directory.appendingPathComponent("events.jsonl")
        try WorkflowBundleCodec.lineEncoder().encode(event).write(to: eventsURL)
        _ = try WorkflowRunRecovery.inspect(at: directory)
        XCTAssertEqual(try String(contentsOf: eventsURL, encoding: .utf8).split(separator: "\n").count, 2)
        XCTAssertFalse(try FileManager.default.contentsOfDirectory(atPath: directory.path).contains { $0.hasSuffix(".fragment") })
    }

    func testCompleteInvalidFinalEventWithoutNewlineIsNotRepaired() throws {
        let directory = try fixture()
        let url = directory.appendingPathComponent("events.jsonl")
        let invalidEvent = Data(#"{"sequence":0,"type":"run_started"}"#.utf8)
        try invalidEvent.write(to: url)
        XCTAssertThrowsError(try WorkflowRunRecovery.inspect(at: directory))
        XCTAssertEqual(try Data(contentsOf: url), invalidEvent)
        XCTAssertEqual(try read(directory).state, .running)
    }

    private func read(_ directory: URL) throws -> GraphRunManifest {
        try WorkflowBundleCodec.decoder().decode(GraphRunManifest.self, from: Data(contentsOf: directory.appendingPathComponent("run.json")))
    }

    private func fixture(contract: String = GraphRunManifest.contractVersion) throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        let timestamp = Date()
        let manifest = GraphRunManifest(
            contractVersion: contract, jobID: "interruption", graphName: "recovery", graphFingerprint: "fixture",
            state: .running, createdAt: timestamp, updatedAt: timestamp, attempt: 1,
            executor: .init(kind: "worker", profile: nil, jobReference: nil),
            nodes: [
                .init(id: "done", kind: "text.value", state: .finished, startedAt: timestamp, completedAt: timestamp,
                      exitStatus: 0, fingerprint: "done", artifacts: [],
                      outputs: [.init(name: "text", type: .string, value: .string("retained"), path: nil, contentType: nil, sizeBytes: nil, sha256: nil)], error: nil),
                .init(id: "active", kind: "text.value", state: .running, startedAt: timestamp, completedAt: nil,
                      exitStatus: nil, fingerprint: "active", artifacts: [], error: nil)
            ], outputs: [], error: nil
        )
        try WorkflowBundleCodec.write(manifest, to: directory.appendingPathComponent("run.json"))
        return directory
    }
}
