import Foundation
import XCTest
@testable import MereRunCLI
@testable import MereRunCore

final class BuiltinToolsTests: XCTestCase {
    func testWriteFileStaysInsideSandbox() throws {
        let sandbox = try makeSandbox()
        let policy = BuiltinTools.ToolExecutionPolicy(
            sandboxDir: sandbox,
            allowShellExec: false,
            allowAbsolutePaths: false
        )

        let result = try BuiltinTools.execute(
            ToolCall(name: "write_file", arguments: ["path": "nested/output.txt", "content": "ok"]),
            policy: policy
        )

        XCTAssertTrue(result.contains("Wrote"))
        let written = sandbox.appendingPathComponent("nested/output.txt")
        XCTAssertEqual(try String(contentsOf: written, encoding: .utf8), "ok")
    }

    func testWriteFileRejectsTraversalOutsideSandbox() throws {
        let sandbox = try makeSandbox()
        let policy = BuiltinTools.ToolExecutionPolicy(
            sandboxDir: sandbox,
            allowShellExec: false,
            allowAbsolutePaths: false
        )

        XCTAssertThrowsError(
            try BuiltinTools.execute(
                ToolCall(name: "write_file", arguments: ["path": "../escape.txt", "content": "nope"]),
                policy: policy
            )
        )
    }

    func testWriteFileRejectsAbsolutePathByDefault() throws {
        let sandbox = try makeSandbox()
        let policy = BuiltinTools.ToolExecutionPolicy(
            sandboxDir: sandbox,
            allowShellExec: false,
            allowAbsolutePaths: false
        )

        XCTAssertThrowsError(
            try BuiltinTools.execute(
                ToolCall(name: "write_file", arguments: ["path": "/tmp/escape.txt", "content": "nope"]),
                policy: policy
            )
        )
    }

    func testWriteFileRejectsSymlinkedParentOutsideSandbox() throws {
        let sandbox = try makeSandbox()
        let outside = try makeSandbox()
        let link = sandbox.appendingPathComponent("outside", isDirectory: true)
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: outside)

        let policy = BuiltinTools.ToolExecutionPolicy(
            sandboxDir: sandbox,
            allowShellExec: false,
            allowAbsolutePaths: false
        )

        XCTAssertThrowsError(
            try BuiltinTools.execute(
                ToolCall(name: "write_file", arguments: ["path": "outside/escape.txt", "content": "nope"]),
                policy: policy
            )
        )
        XCTAssertFalse(FileManager.default.fileExists(atPath: outside.appendingPathComponent("escape.txt").path))
    }

    func testShellExecRequiresExplicitAllowance() throws {
        let sandbox = try makeSandbox()
        let policy = BuiltinTools.ToolExecutionPolicy(
            sandboxDir: sandbox,
            allowShellExec: false,
            allowAbsolutePaths: false
        )

        let result = try BuiltinTools.execute(
            ToolCall(name: "shell_exec", arguments: ["command": "pwd"]),
            policy: policy
        )

        XCTAssertEqual(result, "Denied: 'shell_exec' requires --allow-shell-exec.")
    }

    func testShellExecCannotBeAutoApproved() {
        XCTAssertTrue(
            BuiltinTools.canAutoApprove(
                ToolCall(name: "write_file", arguments: [:]),
                autoApproveTools: true
            )
        )
        XCTAssertFalse(
            BuiltinTools.canAutoApprove(
                ToolCall(name: "shell_exec", arguments: ["command": "pwd"]),
                autoApproveTools: true
            )
        )
        XCTAssertFalse(
            BuiltinTools.canAutoApprove(
                ToolCall(name: "write_file", arguments: [:]),
                autoApproveTools: false
            )
        )
    }

    func testAllowedShellExecDrainsLargeOutputAndReportsTruncation() throws {
        let policy = BuiltinTools.ToolExecutionPolicy(
            sandboxDir: try makeSandbox(), allowShellExec: true, allowAbsolutePaths: false,
            shellTimeout: 5, shellOutputLimitBytes: 128
        )
        let result = try BuiltinTools.execute(
            ToolCall(name: "shell_exec", arguments: ["command": "head -c 1048576 /dev/zero"]),
            policy: policy
        )
        XCTAssertEqual(result, String(repeating: "\0", count: 128) + "\n[output truncated]")
    }

    func testAllowedShellExecReportsTimeout() throws {
        let policy = BuiltinTools.ToolExecutionPolicy(
            sandboxDir: try makeSandbox(), allowShellExec: true, allowAbsolutePaths: false,
            shellTimeout: 0.1
        )
        let result = try BuiltinTools.execute(
            ToolCall(name: "shell_exec", arguments: ["command": "sleep 30"]), policy: policy
        )
        XCTAssertTrue(result.hasPrefix("Timed out after 0.1 seconds."))
    }

    private func makeSandbox() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        addTeardownBlock {
            try? FileManager.default.removeItem(at: root)
        }
        return root
    }
}
