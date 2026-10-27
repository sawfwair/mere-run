import Foundation
import Tokenizers
import XCTest
@testable import MereRunCore

final class LFM2IgnoreMergesEncoderTests: MereRunCoreTestCase {
    private func fixture(ignoreMerges: Bool) throws -> URL {
        let root = FileManager.default.temporaryDirectory.appending(path: "lfm2-tokenizer-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try Data(#"{"tokenizer_class":"GPT2Tokenizer","eos_token":"<|im_end|>"}"#.utf8)
            .write(to: root.appending(path: "tokenizer_config.json"))
        let data = #"""
        {"version":"1.0","normalizer":null,
         "added_tokens":[{"id":5,"content":"<|im_end|>","lstrip":false,"rstrip":false,"single_word":false,"normalized":false,"special":true}],
         "pre_tokenizer":{"type":"Sequence","pretokenizers":[
           {"type":"Split","pattern":{"Regex":" ?[a-z]+|\\s+"},"behavior":"Isolated","invert":false},
           {"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false}]},
         "post_processor":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},
         "decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},
         "model":{"type":"BPE","vocab":{"a":0,"b":1,"ab":2,"Ġab":3,"Ġ":4,"<|im_end|>":5},"merges":[],"ignore_merges":IGNORE}}
        """#.replacingOccurrences(of: "IGNORE", with: ignoreMerges ? "true" : "false")
        try Data(data.utf8).write(to: root.appending(path: "tokenizer.json"))
        return root
    }

    func testWholePieceLookupPrecedesMergeFallbackAndPreservesSpecialBoundaries() async throws {
        let root = try fixture(ignoreMerges: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let fallback = try await AutoTokenizer.from(modelFolder: root)
        XCTAssertEqual(fallback.encode(text: "ab", addSpecialTokens: false), [0, 1])
        let encoder = try XCTUnwrap(LFM2IgnoreMergesEncoder.load(from: root, fallback: fallback))
        XCTAssertEqual(encoder.encode("ab"), [2])
        XCTAssertEqual(encoder.encode(" ab"), [3])
        XCTAssertEqual(encoder.encode("aa"), [0, 0])
        XCTAssertEqual(encoder.encode("ab<|im_end|> ab"), [2, 5, 3])
    }

    func testMergeBasedCheckpointsKeepTheirExistingTokenizer() async throws {
        let root = try fixture(ignoreMerges: false)
        defer { try? FileManager.default.removeItem(at: root) }
        let fallback = try await AutoTokenizer.from(modelFolder: root)
        XCTAssertNil(try LFM2IgnoreMergesEncoder.load(from: root, fallback: fallback))
        XCTAssertEqual(fallback.encode(text: "ab", addSpecialTokens: false), [0, 1])
    }
}
