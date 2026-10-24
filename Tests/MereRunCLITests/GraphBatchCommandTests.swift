import ArgumentParser
import Foundation
import XCTest
@testable import MereRunCLI

final class GraphBatchCommandTests: XCTestCase {
    func testBatchSubmissionParsesIndependentRowsAndExplicitRetryPolicy() throws {
        let command = try GraphBatchSubmit.parse(["workflow.json", "--items-jsonl", "rows.jsonl", "--executor", "relay:fleet",
                                                 "--run-dir", "batch", "--max-in-flight", "3", "--max-attempts", "2", "--json"])
        XCTAssertEqual(command.maxInFlight, 3)
        XCTAssertEqual(command.maxAttempts, 2)
        XCTAssertEqual(command.executor, "relay:fleet")
    }

    func testJSONLAndJSONArrayProduceTheSameTypedRows() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let row = #"{"key":"scene-1","inputs":{"prompt":"fixture","seed":9223372036854775807}}"#
        let jsonl = root.appendingPathComponent("rows.jsonl")
        let json = root.appendingPathComponent("rows.json")
        try Data((row + "\n").utf8).write(to: jsonl)
        try Data(("[" + row + "]").utf8).write(to: json)
        let rows = try GraphBatchSubmit.loadItems(jsonl)
        XCTAssertEqual(rows, try GraphBatchSubmit.loadItems(json))
        XCTAssertEqual(rows[0].inputs.values["seed"], .integer(Int64.max))
        try Data("{bad json}\n".utf8).write(to: jsonl)
        XCTAssertThrowsError(try GraphBatchSubmit.loadItems(jsonl)) { error in
            XCTAssertTrue(String(describing: error).contains("line 1"))
        }
    }

    func testBatchRemoteCommandsParseStableBatchIdentifiers() throws {
        let id = UUID().uuidString.lowercased()
        XCTAssertEqual(try GraphBatchInspect.parse([id, "--executor", "relay:fleet"]).options.batchID, id)
        XCTAssertEqual(try GraphBatchCancel.parse([id, "--executor", "relay:fleet"]).options.batchID, id)
        XCTAssertEqual(try GraphBatchRetry.parse([id, "--executor", "relay:fleet"]).options.batchID, id)
        XCTAssertThrowsError(try GraphBatchWatch.parse([id, "--executor", "relay:fleet", "--timeout-seconds", "-1"]))
    }
}
