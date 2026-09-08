import Foundation
import XCTest
@testable import MereRunCLI

final class BoundedProcessRunnerTests: XCTestCase {
    func testDrainsBothStreamsAfterCaptureLimitsAreReached() throws {
        let process = Self.shell(
            "head -c 1048576 /dev/zero & head -c 1048576 /dev/zero >&2 & wait"
        )
        let result = try BoundedProcessRunner.run(process, timeout: 5, outputLimitBytes: 4096)

        XCTAssertEqual(result.completion, .exited)
        XCTAssertEqual(result.status, 0)
        XCTAssertTrue(result.stdout.truncated)
        XCTAssertTrue(result.stderr.truncated)
        XCTAssertEqual(result.stdout.text, String(repeating: "\0", count: 4096) + "\n[output truncated]")
        XCTAssertEqual(result.stderr.text, result.stdout.text)
    }

    func testCapturesSplitUTF8AndFinalOutputInOrder() throws {
        let process = Self.shell("printf '\\342'; sleep 0.02; printf '\\230'; sleep 0.02; printf '\\203\\nfinished'")
        let result = try BoundedProcessRunner.run(process, timeout: 5)

        XCTAssertEqual(result.completion, .exited)
        XCTAssertEqual(result.stdout.text, "☃\nfinished")
        XCTAssertFalse(result.stdout.truncated)
        XCTAssertEqual(result.stderr.text, "")
    }

    func testCombinedOutputPreservesOrdinaryToolResponseAndExitStatus() throws {
        let result = try BoundedProcessRunner.run(
            Self.shell("printf first; printf second >&2; printf third; exit 7"),
            timeout: 5, combineOutput: true
        )
        XCTAssertEqual(result.completion, .exited)
        XCTAssertEqual(result.status, 7)
        XCTAssertEqual(result.stdout.text, "firstsecondthird")
        XCTAssertEqual(result.stderr.text, "")
    }

    func testTimeoutStopsChildrenEvenAfterTheShellLeaderExits() throws {
        let root = try temporaryDirectory()
        let process = Self.shell("(trap '' TERM; sleep 1; printf leaked > escaped.txt) & exit 0")
        process.currentDirectoryURL = root
        let result = try BoundedProcessRunner.run(process, timeout: 0.15)

        XCTAssertEqual(result.completion, .timedOut)
        XCTAssertLessThan(result.seconds, 2)
        Thread.sleep(forTimeInterval: 1)
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("escaped.txt").path))
    }

    func testTaskCancellationStopsAnAlreadyRunningChild() async throws {
        let root = try temporaryDirectory()
        let started = root.appendingPathComponent("started.txt")
        let task = Task.detached { () throws -> BoundedProcessRunner.Result in
            let process = Process()
            process.executableURL = URL(fileURLWithPath: "/bin/sh")
            process.arguments = ["-c", "printf started > started.txt; exec sleep 30"]
            process.currentDirectoryURL = root
            return try BoundedProcessRunner.run(process, timeout: 5)
        }
        defer { task.cancel() }
        for _ in 0..<200 {
            if FileManager.default.fileExists(atPath: started.path) { break }
            try await Task.sleep(nanoseconds: 10_000_000)
        }
        XCTAssertTrue(FileManager.default.fileExists(atPath: started.path))
        task.cancel()
        let result = try await task.value
        XCTAssertEqual(result.completion, .cancelled)
        XCTAssertLessThan(result.seconds, 3)
    }

    func testLaunchFailureThrowsWithoutWaitingForOutput() {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/mererun-test-missing-executable")
        XCTAssertThrowsError(try BoundedProcessRunner.run(process, timeout: 1))
    }

    private static func shell(_ command: String) -> Process {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/sh")
        process.arguments = ["-c", command]
        return process
    }

    private func temporaryDirectory() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        return root
    }
}
