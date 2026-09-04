import Foundation
@testable import MereRunCLI
import XCTest

final class ModelContextWindowTests: XCTestCase {
    func testReadsDeclaredTextContextAndLeavesUnknownConfigurationsUnspecified() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("model-context-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let fixtures: [(String, Int?)] = [
            (#"{"max_position_embeddings":32768}"#, 32768),
            (#"{"text_config":{"max_position_embeddings":131072},"max_position_embeddings":1024}"#, 131072),
            (#"{"context_length":8192}"#, 8192),
            (#"{"max_position_embeddings":0}"#, nil),
            (#"{"max_position_embeddings":"invalid"}"#, nil),
            (#"{"model_type":"unknown"}"#, nil)
        ]
        for (json, expected) in fixtures {
            try Data(json.utf8).write(to: root.appendingPathComponent("config.json"))
            XCTAssertEqual(ModelContextWindow.read(at: root), expected)
        }
    }

    func testModelListJSONIsAnExplicitAdditiveOption() throws {
        let command = try ModelList.parse(["--json", "--measure-sizes"])
        XCTAssertTrue(command.json)
        XCTAssertTrue(command.measureSizes)
    }
}
