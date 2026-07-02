import Foundation
import XCTest
@testable import MereRunCore

final class TextSFTDatasetTests: XCTestCase {
    func testLoadsChatJSONLDatasetAndSummarizes() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let datasetURL = directory.appendingPathComponent("pairs.jsonl")
        try """
        {"id":"local-assistant-0001","sources":["docs:local"],"messages":[{"role":"system","content":"You know Mere."},{"role":"user","content":"What is local mode?"},{"role":"assistant","content":"Local mode is explicit and project-specific."}]}
        {"id":"local-assistant-0002","sources":["docs:run"],"messages":[{"role":"system","content":"You know Mere."},{"role":"user","content":"Which model?"},{"role":"assistant","content":"Use text-chat-gemma4-12b-4bit for the local assistant lane."}]}
        """.write(to: datasetURL, atomically: true, encoding: .utf8)

        let examples = try TextSFTDataset.load(from: datasetURL)
        let summary = TextSFTDataset.summarize(examples)

        XCTAssertEqual(examples.count, 2)
        XCTAssertEqual(summary.exampleCount, 2)
        XCTAssertEqual(summary.messageCount, 6)
        XCTAssertEqual(summary.sourceCount, 2)
        XCTAssertEqual(summary.fingerprint.count, 64)
    }

    func testRejectsDuplicatePrompts() throws {
        let examples = [
            TextSFTExample(
                id: "one",
                sources: ["docs"],
                messages: [
                    ChatMessage(role: .system, content: "System"),
                    ChatMessage(role: .user, content: "Same prompt"),
                    ChatMessage(role: .assistant, content: "First answer"),
                ]
            ),
            TextSFTExample(
                id: "two",
                sources: ["docs"],
                messages: [
                    ChatMessage(role: .system, content: "System"),
                    ChatMessage(role: .user, content: " same PROMPT "),
                    ChatMessage(role: .assistant, content: "Second answer"),
                ]
            ),
        ]

        XCTAssertThrowsError(try TextSFTDataset.validate(examples)) { error in
            XCTAssertTrue(String(describing: error).contains("duplicatePrompt"))
        }
    }
}
