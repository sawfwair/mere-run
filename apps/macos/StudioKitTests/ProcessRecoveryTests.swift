import Foundation
import XCTest
@testable import StudioKit

final class ProcessRecoveryTests: XCTestCase {
    func testCompletionWaitsForBothOutputStreamsAndTheirFinalCallbacks() async throws {
        let root = try temporaryDirectory()
        let output = ProcessRecoveryOutput()
        let finished = expectation(description: "process and output finished")
        let marker = root.appendingPathComponent("reader-started")
        let process = try FoundationMereRunProcessRunner().start(
            configuration: configuration(
                "printf 'ready\\n'; while [ ! -f reader-started ]; do sleep 0.01; done; printf 'final\\n'; printf 'error\\n' >&2",
                root: root
            ),
            stdout: { text in
                if text.contains("ready") {
                    try? Data().write(to: marker)
                    Thread.sleep(forTimeInterval: 0.1)
                }
                output.append(text, error: false)
            },
            stderr: { output.append($0, error: true) },
            termination: { code in
                output.finish(code)
                finished.fulfill()
            }
        )
        defer { withExtendedLifetime(process) {} }
        await fulfillment(of: [finished], timeout: 5)
        let result = output.completed
        XCTAssertEqual(result?.code, 0)
        XCTAssertEqual(result?.stdout, "ready\nfinal\n")
        XCTAssertEqual(result?.stderr, "error\n")
    }

    func testCancellationStopsOwnedDescendantsBeforeCompletion() async throws {
        let root = try temporaryDirectory()
        let ready = expectation(description: "child ready")
        let finished = expectation(description: "cancelled process finished")
        let process = try FoundationMereRunProcessRunner().start(
            configuration: configuration(
                "(trap '' TERM; sleep 1; printf escaped > escaped.txt) & printf 'ready\\n'; wait",
                root: root
            ),
            stdout: { if $0.contains("ready") { ready.fulfill() } },
            stderr: { _ in },
            termination: { _ in finished.fulfill() }
        )
        await fulfillment(of: [ready], timeout: 3)
        process.terminate()
        await fulfillment(of: [finished], timeout: 3)
        try await Task.sleep(for: .milliseconds(1_200))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("escaped.txt").path))
    }

    func testCancellationCanStopAProcessThatDoesNotReadInteractiveInput() async throws {
        let root = try temporaryDirectory()
        let ready = expectation(description: "interactive child ready")
        let finished = expectation(description: "interactive child stopped")
        let written = expectation(description: "blocked input writer released")
        let process = try XCTUnwrap(FoundationMereRunProcessRunner().start(
            configuration: .init(executableURL: URL(fileURLWithPath: "/bin/sh"),
                                 arguments: ["-c", "trap '' TERM; printf ready; sleep 30"],
                                 currentDirectoryURL: root, environment: ProcessInfo.processInfo.environment,
                                 keepsStandardInputOpen: true),
            stdout: { if $0.contains("ready") { ready.fulfill() } }, stderr: { _ in },
            termination: { _ in finished.fulfill() }
        ) as? FoundationRunningProcess)
        await fulfillment(of: [ready], timeout: 3)
        let writer = Task.detached {
            defer { written.fulfill() }
            do {
                try process.sendStandardInput(String(repeating: "x", count: 1_048_576))
                XCTFail("The silent fixture cannot accept the complete input")
            } catch { }
        }
        try await Task.sleep(for: .milliseconds(50))
        process.terminate()
        await fulfillment(of: [finished, written], timeout: 3)
        await writer.value
    }

    @MainActor
    func testNativeJobQueueDeliversFinalReceiptsAndStartsNextJobAfterCancellation() async throws {
        let root = try temporaryDirectory()
        let store = JobStore()
        let template = try XCTUnwrap(CommandCatalog.template(id: .imageGenerate))
        func request(_ name: String, script: String) -> JobRequest {
            var draft = template.defaultDraft()
            draft.prompt = "fixture"
            draft.outputPath = root.appendingPathComponent("\(name).png").path
            return .init(
                lane: .inference, template: template, draft: draft, requestID: UUID(),
                configuration: configuration(script, root: root), displayCommand: "fixture \(name)", scopeSource: .contract
            )
        }
        let first = store.submit(request("first", script: "trap '' TERM; printf ready; sleep 30"))
        let second = store.submit(request("second", script: "trap '' TERM; printf ready; sleep 30"))
        let output = root.appendingPathComponent("retry.png")
        let third = store.submit(request("retry", script:
            "printf complete > retry.png; printf '{\"event\":\"result\",\"exit\":0,\"outputs\":[{\"kind\":\"image\",\"path\":\"\(output.path)\"}]}\\n'"))
        XCTAssertEqual(store.job(third)?.state, .queued)
        for _ in 0..<200 {
            if store.job(first)?.liveText == "ready", store.job(second)?.liveText == "ready" { break }
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTAssertTrue(store.cancel(first))
        XCTAssertTrue(store.cancel(second))
        for _ in 0..<400 {
            // Either cancelled job can free the slot that starts the retry.
            // The other process may still be draining when the retry finishes.
            if [first, second, third].allSatisfy({ store.job($0)?.state.isTerminal == true }) { break }
            try await Task.sleep(for: .milliseconds(10))
        }
        let result = try XCTUnwrap(store.job(third)?.result)
        XCTAssertEqual(result.exitCode, 0)
        XCTAssertEqual(result.outputURL, output)
        XCTAssertEqual(result.artifactURLs, [output])
        XCTAssertFalse(result.outputText?.contains("\"event\"") == true)
        XCTAssertTrue(store.job(first)?.state.isTerminal == true)
        XCTAssertTrue(store.job(second)?.state.isTerminal == true)
        XCTAssertTrue(store.running(in: .inference).isEmpty)
    }

    func testLargeInterleavedStreamsAndSplitUnicodeAreDeliveredCompletely() async throws {
        let root = try temporaryDirectory()
        let output = ProcessRecoveryOutput()
        let finished = expectation(description: "both pipes drained")
        let process = try FoundationMereRunProcessRunner().start(
            configuration: configuration(
                "head -c 131072 /dev/zero | tr '\\000' x; head -c 131072 /dev/zero | tr '\\000' y >&2; printf '\\360\\237'; sleep 0.03; printf '\\231\\202'",
                root: root
            ), stdout: { output.append($0, error: false) }, stderr: { output.append($0, error: true) },
            termination: { output.finish($0); finished.fulfill() }
        )
        defer { withExtendedLifetime(process) {} }
        await fulfillment(of: [finished], timeout: 5)
        XCTAssertEqual(output.completed?.code, 0)
        XCTAssertEqual(output.completed?.stdout, String(repeating: "x", count: 131_072) + "🙂")
        XCTAssertEqual(output.completed?.stderr, String(repeating: "y", count: 131_072))
    }

    func testRepeatedShortProcessesDeliverFinalOutputAndExitStatus() async throws {
        let root = try temporaryDirectory()
        let runner = FoundationMereRunProcessRunner()
        for index in 0..<32 {
            let output = ProcessRecoveryOutput()
            let finished = expectation(description: "short process \(index) finished")
            let exitCode: Int32 = index.isMultiple(of: 2) ? 0 : 7
            let process = try runner.start(
                configuration: configuration(
                    "printf 'result-\(index)\\n'; printf 'diagnostic-\(index)\\n' >&2; exit \(exitCode)",
                    root: root
                ),
                stdout: { output.append($0, error: false) },
                stderr: { output.append($0, error: true) },
                termination: { output.finish($0); finished.fulfill() }
            )
            defer { process.terminate() }
            await fulfillment(of: [finished], timeout: 5)
            let result = try XCTUnwrap(output.completed)
            XCTAssertEqual(result.code, exitCode)
            XCTAssertEqual(result.stdout, "result-\(index)\n")
            XCTAssertEqual(result.stderr, "diagnostic-\(index)\n")
        }
    }

    private func configuration(_ script: String, root: URL) -> MereRunProcessConfiguration {
        .init(executableURL: URL(fileURLWithPath: "/bin/sh"), arguments: ["-c", script],
              currentDirectoryURL: root, environment: ProcessInfo.processInfo.environment,
              keepsStandardInputOpen: false)
    }

    private func temporaryDirectory() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        return root
    }
}

private final class ProcessRecoveryOutput: @unchecked Sendable {
    struct Result {
        let code: Int32
        let stdout: String
        let stderr: String
    }
    private let lock = NSLock()
    private var stdout = ""
    private var stderr = ""
    private var result: Result?

    func append(_ text: String, error: Bool) {
        lock.withLock {
            if error { stderr += text } else { stdout += text }
        }
    }

    func finish(_ code: Int32) {
        lock.withLock { result = .init(code: code, stdout: stdout, stderr: stderr) }
    }

    var completed: Result? { lock.withLock { result } }
}
