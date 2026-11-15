import Foundation
import MLX
import MereRunMLXTestSupport
import XCTest
@testable import MereRunGemmaModel

final class SOVGenerationOptimizationTests: MLXTestCase {
    func testSharedAttentionStorageMatchesReferenceAcrossRolloverAndForks() throws {
        for window in [2, 4, 8] {
            let reference = Gemma4SlidingKVCache(maxSize: window)
            let shared = Gemma4SlidingKVCache(maxSize: window)
            shared.reuseAttentionStorage = true
            var position = 0
            for count in [1, 1, 1, 3, 1, 2, 7, 1, 1, 3] {
                // Distinct heads and unequal K/V widths exercise retained strided views.
                let keys = MLXArray((0..<(2 * count * 4)).map { Float($0 + position * 100) }).reshaped(1, 2, count, 4)
                let values = MLXArray((0..<(2 * count * 2)).map { Float($0 - position * 100) }).reshaped(1, 2, count, 2)
                let expected = try XCTUnwrap(reference.attentionState(appending: keys, values: values))
                let actual = try XCTUnwrap(shared.attentionState(appending: keys, values: values))
                XCTAssertEqual(actual.0.asArray(Float.self), expected.0.asArray(Float.self))
                XCTAssertEqual(actual.1.asArray(Float.self), expected.1.asArray(Float.self))
                let expectedState = try XCTUnwrap(reference.currentState())
                let actualState = try XCTUnwrap(shared.currentState())
                XCTAssertEqual(actualState.0.asArray(Float.self), expectedState.0.asArray(Float.self))
                XCTAssertEqual(actualState.1.asArray(Float.self), expectedState.1.asArray(Float.self))
                XCTAssertEqual(shared.offset, reference.offset)
                let trial = try XCTUnwrap(shared.fork() as? Gemma4SlidingKVCache)
                XCTAssertTrue(trial.reuseAttentionStorage)
                trial.append(keys: keys, values: values)
                XCTAssertEqual(shared.currentState()!.0.asArray(Float.self), actualState.0.asArray(Float.self))
                position += count
            }
        }
    }

    func testDraftHeadPackingPreservesTargetParametersAndOutput() throws {
        let config = try JSONDecoder().decode(Gemma4TextConfig.self, from: Data(#"{"model_type":"gemma4_text","hidden_size":64,"num_hidden_layers":2,"intermediate_size":128,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":32,"vocab_size":128,"max_position_embeddings":128,"rms_norm_eps":0.000001,"final_logit_softcapping":30}"#.utf8))
        let target = Gemma4TextCausalLM(config: config)
        let before = Dictionary(uniqueKeysWithValues: target.parameters().flattened())
        let hidden = sin(MLXArray((0..<192).map(Float.init))).reshaped(1, 3, 64)
        let original = target.outputLogits(hidden)
        eval(original)
        let head = target.makeQuantizedDraftHead()
        eval(head.parameters())
        XCTAssertEqual(head.bits, 8)
        XCTAssertEqual(head.groupSize, 64)
        XCTAssertEqual(head.weight.dtype, .uint32)
        let after = Dictionary(uniqueKeysWithValues: target.parameters().flattened())
        XCTAssertEqual(Set(before.keys), Set(after.keys))
        for (name, parameter) in before { XCTAssertTrue(parameter === after[name]) }
        XCTAssertEqual(target.outputLogits(hidden).asArray(Float.self), original.asArray(Float.self))
        let draftLogits = target.softcapOutputLogits(head(hidden))
        XCTAssertEqual(draftLogits.shape, original.shape)
        XCTAssertTrue(draftLogits.asArray(Float.self).allSatisfy(\.isFinite))
        XCTAssertLessThan(abs(draftLogits - original).max().item(Float.self), 0.1)
    }
}
