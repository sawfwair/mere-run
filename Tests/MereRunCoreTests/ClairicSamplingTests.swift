#if !os(iOS)
import Foundation
import MLX
import MereRunMLXTestSupport
import XCTest
@testable import MereRunCore

final class ClairicSamplingTests: MLXTestCase {
    func testSlotMasksInvalidTokensAndScoresNullIndependentlyOfTheChosenValue() throws {
        let root = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let data = """
        {
          "model":{"type":"BPE","unk_token":"?","vocab":{"n":0,"t":1,"f":2,"null":3,"true":4,"false":5,"?":6},"merges":[]},
          "added_tokens":[{"id":7,"content":"<|im_end|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],
          "decoder":{"type":"ByteLevel"}
        }
        """
        try Data(data.utf8).write(to: root.appending(path: "tokenizer.json"))
        let config = #"{"tokenizer_class":"Qwen2Tokenizer","unk_token":"?","eos_token":"<|im_end|>"}"#
        try Data(config.utf8).write(to: root.appending(path: "tokenizer_config.json"))
        let tokenizer = try ClairicTokenizer(root: root, vocabularySize: 8)
        let spec = ClairicField(type: .boolean)
        let slot = ClairicPrompt.Slot(field: "paid", index: nil, spec: spec, instruction: "")
        let grammar = try ClairicSlotGrammar(spec)
        let row = ClairicDecodeRow(slot: slot, caches: [], logits: MLXArray([0, 0, 0, 1, 2, 0, 100, 100] as [Float]),
                                   grammar: grammar, nextPosition: 0, usesMRoPE: false)
        XCTAssertEqual(try row.step(tokenizer: tokenizer, isFirst: true), 4)
        XCTAssertNil(try row.step(tokenizer: tokenizer, isFirst: false))
        let answer = row.answer(source: "Payment received.", hasMedia: false)
        let sum = exp(Float(2)) + exp(Float(1)) + 4
        XCTAssertEqual(answer.value, .bool(true))
        XCTAssertTrue(answer.valid)
        XCTAssertTrue(answer.complete)
        XCTAssertEqual(answer.confidence, exp(Float(2)) / sum, accuracy: 0.00001)
        XCTAssertEqual(answer.p_null, exp(Float(1)) / sum, accuracy: 0.00001)
        XCTAssertEqual(row.tokens, 2)
    }
}
#endif
