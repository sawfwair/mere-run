import Foundation
import XCTest
@testable import MereRunCore

final class LFM2GenerationStreamTests: XCTestCase {
    private final class Output: @unchecked Sendable {
        private let lock = NSLock()
        private var contents = ""
        func accept(_ progress: ChatProgress) {
            lock.withLock { contents += progress.message ?? "" }
        }
        var text: String { lock.withLock { contents } }
    }

    func testHidesThinkingForEveryTagBoundaryAndPreservesVisibleWhitespace() {
        let raw = "<think>Private analysis</think>\nVisible answer."
        for split in 0...raw.count {
            let output = Output()
            let stream = LFM2GenerationStream(showThinking: false, handler: output.accept)
            stream.accept(ChatProgress(stage: .generating, message: String(raw.prefix(split))))
            stream.accept(ChatProgress(stage: .generating, message: String(raw.dropFirst(split))))
            stream.finish()
            XCTAssertEqual(output.text, "\nVisible answer.", "Boundary \(split)")
        }
    }

    func testTruncatedPrefilledThinkingNeverBecomesVisible() {
        let output = Output()
        let stream = LFM2GenerationStream(showThinking: false, handler: output.accept)
        stream.accept(ChatProgress(stage: .generating, message: "<think>"))
        stream.accept(ChatProgress(stage: .generating, message: "Unfinished reasoning</thi"))
        XCTAssertEqual(output.text, "")
        stream.finish()
        XCTAssertEqual(output.text, "")
    }

    func testReopenedThinkingAndSplitTagsRemainHidden() {
        let output = Output()
        let stream = LFM2GenerationStream(showThinking: false, handler: output.accept)
        for character in "Before<THINK>first</THINK>between<think>second</think>after" {
            stream.accept(ChatProgress(stage: .generating, message: String(character)))
        }
        stream.finish()
        XCTAssertEqual(output.text, "Beforebetweenafter")
    }

    func testOrdinaryTextStreamsBeforeCompletionAndPreservesLiteralTagPrefix() {
        let output = Output()
        let stream = LFM2GenerationStream(showThinking: false, handler: output.accept)
        stream.accept(ChatProgress(stage: .generating, message: "Visible now <thin"))
        XCTAssertEqual(output.text, "Visible now ")
        stream.finish()
        XCTAssertEqual(output.text, "Visible now <thin")
    }

    func testExplicitThinkingOutputPreservesRawStream() {
        let output = Output()
        let stream = LFM2GenerationStream(showThinking: true, handler: output.accept)
        let raw = "<think>Analysis</think>Answer"
        for character in raw { stream.accept(ChatProgress(stage: .generating, message: String(character))) }
        stream.finish()
        XCTAssertEqual(output.text, raw)
    }
}
