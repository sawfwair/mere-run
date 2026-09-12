import Foundation
import XCTest
import MereRunRelayKit
@testable import MereRunCLI

final class WorkflowRecoveryTests: XCTestCase {
    func testCancellationSettlesActiveNodeAndResumeKeepsCompletedOutputs() throws {
        let root = try temporaryDirectory()
        let graph = try decodeGraph(#"{"schema_version":1,"kind":"mere.run/workflow-graph","name":"recovery","inputs":{},"nodes":[{"id":"first","kind":"text.value","arguments":{"value":"retained"}},{"id":"second","kind":"text.value","arguments":{"value":{"$ref":"nodes.first.outputs.text"}}}],"outputs":{"text":{"$ref":"nodes.second.outputs.text"}}}"#)
        let bundle = try materialize(graph, at: root)
        let run = root.appendingPathComponent("run")
        let outcome = try WorkflowRunner(
            bundleDirectory: bundle.directory, runDirectory: run,
            cacheDirectory: root.appendingPathComponent("cache"),
            eventHandler: { event in
                if event.type == "node_started", event.nodeID == "second" {
                    try? Data("cancel".utf8).write(to: run.appendingPathComponent("cancel.request"))
                }
            }
        ).execute()
        XCTAssertEqual(outcome.state, .cancelled)
        let cancelled = try manifest(at: run)
        XCTAssertEqual(cancelled.nodes[0].state, .finished)
        XCTAssertEqual(cancelled.nodes[1].state, .cancelled)
        XCTAssertNotNil(cancelled.nodes[1].completedAt)
        let priorOutputs = cancelled.nodes[0].outputs
        let cancellationEvents = try events(at: run)
        XCTAssertEqual(cancellationEvents.filter { $0.type == "node_cancelled" && $0.nodeID == "second" }.count, 1)

        let resumed = try WorkflowRunner(
            bundleDirectory: bundle.directory, runDirectory: run, resume: true,
            cacheDirectory: root.appendingPathComponent("cache")
        ).execute()
        XCTAssertEqual(resumed.state, .finished)
        let completed = try manifest(at: run)
        XCTAssertEqual(completed.attempt, 2)
        XCTAssertEqual(completed.nodes[0].outputs, priorOutputs)
        XCTAssertTrue(try events(at: run).contains { $0.type == "node_resumed" && $0.nodeID == "first" })
        XCTAssertTrue(completed.nodes.allSatisfy { $0.state == .finished && $0.error == nil })
    }

    func testCancellingTheCallerStopsParallelChildrenWithoutAnExternalCancelCommand() async throws {
        let root = try temporaryDirectory()
        let graph = try decodeGraph(#"{"schema_version":1,"kind":"mere.run/workflow-graph","name":"parallel-recovery","execution":{"max_parallel_nodes":2},"inputs":{},"nodes":[{"id":"first","kind":"image.generate","arguments":{"prompt":"fixture","model":"image-zimage-nano","width":256,"height":256,"steps":1}},{"id":"second","kind":"image.generate","arguments":{"prompt":"fixture","model":"image-zimage-nano","width":256,"height":256,"steps":1}}],"outputs":{}}"#)
        let bundle = try materialize(graph, at: root)
        let run = root.appendingPathComponent("run")
        let finished = expectation(description: "caller cancellation stops parallel children")
        let runner = WorkflowRunner(
            bundleDirectory: bundle.directory, runDirectory: run,
            processRunner: RecoveryWorkflowProcessRunner()
        )
        let task = Task.detached {
            defer { finished.fulfill() }
            return try runner.execute()
        }
        defer { task.cancel() }
        for _ in 0..<200 {
            if WorkflowChildProcessRegistry.processIDs(in: run).count == 2 { break }
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTAssertEqual(WorkflowChildProcessRegistry.processIDs(in: run).count, 2)
        task.cancel()
        await fulfillment(of: [finished], timeout: 2)
        // A failing baseline still owns its fixture children. Stop only those
        // children so a regression does not leave sleeping processes behind.
        if !WorkflowChildProcessRegistry.processIDs(in: run).isEmpty {
            try Data("fixture cleanup".utf8).write(to: run.appendingPathComponent("cancel.request"))
            WorkflowChildProcessRegistry.terminateAll(in: run)
        }
        let outcome = try await task.value
        XCTAssertEqual(outcome.state, .cancelled)
        XCTAssertTrue(WorkflowChildProcessRegistry.processIDs(in: run).isEmpty)
        XCTAssertTrue(try manifest(at: run).nodes.allSatisfy { $0.state == .cancelled })
    }

    private func decodeGraph(_ json: String) throws -> WorkflowGraphDocument {
        try JSONDecoder().decode(WorkflowGraphDocument.self, from: Data(json.utf8))
    }

    private func materialize(_ graph: WorkflowGraphDocument, at root: URL) throws -> WorkflowBundleMaterialization {
        try WorkflowBundleMaterializer(
            graph: graph, suppliedInputs: .init(values: [:]),
            destination: root.appendingPathComponent("bundle"), seed: { 42 }
        ).materialize()
    }

    private func manifest(at run: URL) throws -> GraphRunManifest {
        try WorkflowBundleCodec.decoder().decode(
            GraphRunManifest.self, from: Data(contentsOf: run.appendingPathComponent(GraphRunManifest.filename))
        )
    }

    private func events(at run: URL) throws -> [GraphRunEvent] {
        try String(contentsOf: run.appendingPathComponent("events.jsonl"), encoding: .utf8)
            .split(separator: "\n").map { try WorkflowBundleCodec.decoder().decode(GraphRunEvent.self, from: Data($0.utf8)) }
    }

    private func temporaryDirectory() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        return root
    }
}

private struct RecoveryWorkflowProcessRunner: WorkflowProcessRunning, WorkflowStreamingProcessRunning, WorkflowCancellableProcessRunning {
    func run(arguments: [String], currentDirectory: URL) throws -> WorkflowProcessResult {
        try run(executable: URL(fileURLWithPath: "/bin/sh"), arguments: arguments,
                currentDirectory: currentDirectory, timeoutSeconds: nil, stdoutLineHandler: nil)
    }

    func run(
        executable: URL, arguments: [String], currentDirectory: URL,
        timeoutSeconds: Int?, stdoutLineHandler: ((String) throws -> Void)?
    ) throws -> WorkflowProcessResult {
        try run(
            executable: executable, arguments: arguments, currentDirectory: currentDirectory,
            timeoutSeconds: timeoutSeconds, stdoutLineHandler: stdoutLineHandler,
            stdoutChunkHandler: nil, isCancelled: { Task.isCancelled }
        )
    }

    func run(
        executable: URL, arguments: [String], currentDirectory: URL,
        timeoutSeconds: Int?, stdoutLineHandler: ((String) throws -> Void)?,
        stdoutChunkHandler: ((Data) throws -> Void)?, isCancelled: () -> Bool
    ) throws -> WorkflowProcessResult {
        if arguments.contains("--preflight") { return .init(status: 0, stdout: "{}") }
        return try WorkflowProcessRunner().run(
            executable: URL(fileURLWithPath: "/bin/sh"),
            arguments: ["-c", "(trap '' TERM; sleep 30) & wait"],
            currentDirectory: currentDirectory, timeoutSeconds: timeoutSeconds,
            stdoutLineHandler: stdoutLineHandler, stdoutChunkHandler: stdoutChunkHandler,
            isCancelled: isCancelled
        )
    }
}
