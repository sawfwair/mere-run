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
