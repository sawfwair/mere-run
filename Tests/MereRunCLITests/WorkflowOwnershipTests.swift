import Foundation
import XCTest
@testable import MereRunCLI
@testable import MereRunRelayKit

final class WorkflowOwnershipTests: XCTestCase {
    func testThrowingStreamCallbackStopsChildGroupAndRemovesRegistration() throws {
        let root = try temporaryDirectory()
        let node = root.appendingPathComponent("run/nodes/000-fixture")
        try FileManager.default.createDirectory(at: node, withIntermediateDirectories: true)
        let started = Date()
        XCTAssertThrowsError(try WorkflowProcessRunner().run(
            executable: URL(fileURLWithPath: "/bin/sh"),
            arguments: ["-c", "(trap '' TERM; sleep 1; printf leaked > escaped.txt) & printf 'ready\\n'; wait"],
            currentDirectory: node, timeoutSeconds: 5,
            stdoutLineHandler: { _ in throw FixtureError.rejected }
        )) { error in XCTAssertTrue(error is FixtureError) }
        XCTAssertLessThan(Date().timeIntervalSince(started), 2)
        XCTAssertTrue(WorkflowChildProcessRegistry.processIDs(in: root.appendingPathComponent("run")).isEmpty)
        Thread.sleep(forTimeInterval: 1.1)
        XCTAssertFalse(FileManager.default.fileExists(atPath: node.appendingPathComponent("escaped.txt").path))
    }

    func testUnterminatedStdoutIsBoundedAndNeverReturnedAsSuccess() throws {
        let root = try temporaryDirectory()
        let node = root.appendingPathComponent("run/nodes/000-fixture")
        try FileManager.default.createDirectory(at: node, withIntermediateDirectories: true)
        XCTAssertThrowsError(try WorkflowProcessRunner(maximumStdoutBytes: 4096).run(
            executable: URL(fileURLWithPath: "/bin/sh"),
            arguments: ["-c", "head -c 1048576 /dev/zero; sleep 30"],
            currentDirectory: node, timeoutSeconds: 5, stdoutLineHandler: { _ in XCTFail("No complete line") }
        )) { error in XCTAssertEqual((error as? WorkflowProcessOutputLimitError)?.limit, 4096) }
        XCTAssertTrue(WorkflowChildProcessRegistry.processIDs(in: root.appendingPathComponent("run")).isEmpty)
    }

    func testCancellationWithoutDeadlineStopsSilentProcess() async throws {
        let root = try temporaryDirectory()
        let run = root.appendingPathComponent("run")
        let node = run.appendingPathComponent("nodes/000-fixture")
        try FileManager.default.createDirectory(at: node, withIntermediateDirectories: true)
        let task = Task.detached {
            try WorkflowProcessRunner().run(
                executable: URL(fileURLWithPath: "/bin/sleep"), arguments: ["30"],
                currentDirectory: node, timeoutSeconds: nil, stdoutLineHandler: nil
            )
        }
        defer { task.cancel() }
        for _ in 0..<200 {
            if !WorkflowChildProcessRegistry.processIDs(in: run).isEmpty { break }
            try await Task.sleep(nanoseconds: 10_000_000)
        }
        XCTAssertFalse(WorkflowChildProcessRegistry.processIDs(in: run).isEmpty)
        task.cancel()
        do {
            _ = try await task.value
            XCTFail("Expected cancellation")
        } catch { XCTAssertTrue(error is WorkflowCancellationError) }
        XCTAssertTrue(WorkflowChildProcessRegistry.processIDs(in: run).isEmpty)
    }

    func testRunLeaseRejectsSecondOwnerWithoutClearingCancellationOrEvents() throws {
        let fixture = try bundleFixture()
        let run = fixture.root.appendingPathComponent("run")
        let first = store(bundle: fixture.bundle, run: run, resume: false)
        let session = try first.prepare(graph: fixture.bundle.graph, job: fixture.bundle.job, order: ["text"])
        defer { session.lease.release() }
        try first.persist(session.manifest)
        let cancel = run.appendingPathComponent("cancel.request")
        try Data("cancel".utf8).write(to: cancel)
        let marker = run.appendingPathComponent("events.jsonl")
        try Data("previous event\n".utf8).write(to: marker)
        let record = run.appendingPathComponent(GraphRunManifest.filename)
        let original = try Data(contentsOf: record)
        let second = store(bundle: fixture.bundle, run: run, resume: true)
        XCTAssertThrowsError(try second.prepare(graph: fixture.bundle.graph, job: fixture.bundle.job, order: ["text"])) { error in
            XCTAssertTrue(String(describing: error).contains("active worker"))
        }
        XCTAssertEqual(try Data(contentsOf: record), original)
        XCTAssertEqual(try String(contentsOf: marker, encoding: .utf8), "previous event\n")
        XCTAssertTrue(FileManager.default.fileExists(atPath: cancel.path))
        session.lease.release()
        let livePID = ProcessInfo.processInfo.processIdentifier
        try WorkflowChildProcessRegistry.register(livePID, in: run)
        XCTAssertThrowsError(try second.prepare(graph: fixture.bundle.graph, job: fixture.bundle.job, order: ["text"])) { error in
            XCTAssertTrue(String(describing: error).contains("active child processes"))
        }
        XCTAssertTrue(FileManager.default.fileExists(atPath: cancel.path))
        WorkflowChildProcessRegistry.unregister(livePID, in: run)
        let resumed = try second.prepare(graph: fixture.bundle.graph, job: fixture.bundle.job, order: ["text"])
        defer { resumed.lease.release() }
        XCTAssertEqual(resumed.manifest.attempt, 2)
    }

    func testResumeRejectsChangedInputBeforeMutatingPreviousRun() throws {
        let fixture = try bundleFixture()
        let run = fixture.root.appendingPathComponent("run")
        let first = store(bundle: fixture.bundle, run: run, resume: false)
        let session = try first.prepare(graph: fixture.bundle.graph, job: fixture.bundle.job, order: ["text"])
        try first.persist(session.manifest)
        session.lease.release()
        let cancel = run.appendingPathComponent("cancel.request")
        try Data().write(to: cancel)
        let record = run.appendingPathComponent(GraphRunManifest.filename)
        let original = try Data(contentsOf: record)
        let different = try WorkflowBundleMaterializer(
            graph: fixture.bundle.graph, suppliedInputs: .init(values: ["prompt": .string("different")]),
            destination: fixture.root.appendingPathComponent("other-bundle"), seed: { 42 }
        ).materialize()
        XCTAssertThrowsError(try store(bundle: different, run: run, resume: true).prepare(
            graph: different.graph, job: different.job, order: ["text"]
        )) { error in XCTAssertTrue(String(describing: error).contains("input fingerprint changed")) }
        XCTAssertEqual(try Data(contentsOf: record), original)
        XCTAssertTrue(FileManager.default.fileExists(atPath: cancel.path))
    }

    func testArtifactOwnershipRejectsSymlinkEscapesBeforeCleanup() throws {
        let root = try temporaryDirectory()
        let run = root.appendingPathComponent("run")
        let node = run.appendingPathComponent("nodes/000-fixture")
        let outside = root.appendingPathComponent("outside")
        try FileManager.default.createDirectory(at: node, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: outside, withIntermediateDirectories: true)
        let keep = outside.appendingPathComponent("keep.txt")
        try Data("keep".utf8).write(to: keep)
        try FileManager.default.createSymbolicLink(at: node.appendingPathComponent("escape"), withDestinationURL: outside)
        let artifacts = WorkflowArtifactStore(bundleDirectory: root, runDirectory: run, cacheDirectory: nil, resume: false, fileManager: .default)
        let outputs = ["file": WorkflowInvocationOutput(type: .asset, path: "escape/keep.txt", optional: false, contentTypes: [])]
        XCTAssertThrowsError(try artifacts.invocationOutputURL("escape/keep.txt", nodeDirectory: node))
        XCTAssertThrowsError(try artifacts.clearAttemptOutputs(outputs, nodeDirectory: node))
        XCTAssertThrowsError(try artifacts.artifactURL(for: "nodes/000-fixture/escape/keep.txt"))
        XCTAssertEqual(try Data(contentsOf: keep), Data("keep".utf8))
    }

    private enum FixtureError: Error { case rejected }

    private func store(bundle: WorkflowBundleMaterialization, run: URL, resume: Bool) -> WorkflowRunStore {
        .init(bundleDirectory: bundle.directory, runDirectory: run, resume: resume,
              executor: .init(kind: "local", profile: nil, jobReference: nil),
              fileManager: .default, now: Date.init, eventHandler: nil)
    }

    private func bundleFixture() throws -> (root: URL, bundle: WorkflowBundleMaterialization) {
        let root = try temporaryDirectory()
        let graph = try JSONDecoder().decode(WorkflowGraphDocument.self, from: Data(#"{"schema_version":1,"kind":"mere.run/workflow-graph","name":"ownership","inputs":{"prompt":{"type":"string"}},"nodes":[{"id":"text","kind":"text.value","arguments":{"value":{"$ref":"inputs.prompt"}}}],"outputs":{"text":{"$ref":"nodes.text.outputs.text"}}}"#.utf8))
        let bundle = try WorkflowBundleMaterializer(
            graph: graph, suppliedInputs: .init(values: ["prompt": .string("hello")]),
            destination: root.appendingPathComponent("bundle"), seed: { 42 }
        ).materialize()
        return (root, bundle)
    }

    private func temporaryDirectory() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        return root
    }
}
