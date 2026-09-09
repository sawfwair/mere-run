import Foundation
import XCTest
import MLX
import MLXRandom
import MereRunDecode
import MereRunMLXTestSupport
@testable import MereRunGemmaModel

final class Gemma4MTPBoundaryTests: MLXTestCase {
    private let textConfigJSON = """
        {
          "model_type": "gemma4_text", "hidden_size": 8,
          "num_hidden_layers": 2, "intermediate_size": 16,
          "num_attention_heads": 2, "num_key_value_heads": 1,
          "head_dim": 4, "max_position_embeddings": 128,
          "rms_norm_eps": 0.000001, "vocab_size": 32,
          "layer_types": ["full_attention", "full_attention"],
          "num_kv_shared_layers": 1, "sliding_window": 8
        }
        """

    private func makeModels() throws -> (Gemma4TextCausalLM, Gemma4AssistantDraftModel) {
        let decoder = JSONDecoder()
        let targetConfig = try decoder.decode(Gemma4TextConfig.self, from: Data(textConfigJSON.utf8))
        let assistantJSON = """
            {"model_type": "gemma4_assistant", "backbone_hidden_size": 8,
             "text_config": \(textConfigJSON)}
            """
        let assistantConfig = try decoder.decode(Gemma4AssistantConfig.self, from: Data(assistantJSON.utf8))
        return (Gemma4TextCausalLM(config: targetConfig), try Gemma4AssistantDraftModel(config: assistantConfig))
    }

    func testDraftsPreserveSharedTargetCacheAndRepeatDeterministically() throws {
        MLXRandom.seed(73)
        let (target, assistant) = try makeModels()
        let cache = target.makeAttentionCache()
        let prefill = target.forwardForSpeculation(
            inputIds: MLXArray([Int32(3), 4, 5]).reshaped(1, 3), cache: cache
        )
        let shared = try XCTUnwrap(prefill.sharedKVStates["full_attention"])
        MLX.eval(prefill.logits, prefill.hidden, shared.keys, shared.values)
        let originalKeys = shared.keys.asArray(Float.self)
        let originalValues = shared.values.asArray(Float.self)
        let nextToken = argMax(prefill.logits[0, -1, 0...]).item(Int.self)
        let hidden = target.speculativeDraftHidden(prefill.hidden[0..., (-1)..., 0...])
        let config = GenerationConfig(maxTokens: 8, temperature: 0, topP: 1)

        func draft() throws -> [Int] {
            try assistant.draftBlock(
                lastToken: nextToken, hidden: hidden, sharedKVStates: prefill.sharedKVStates,
                positionOffset: 3, blockSize: 4, baseModel: target,
                generationConfig: config, repetitionHistory: [3, 4, 5]
            ).tokens
        }

        let first = try draft()
        XCTAssertEqual(first.count, 3)
        XCTAssertEqual(try draft(), first)
        XCTAssertTrue(first.allSatisfy { (0..<32).contains($0) })
        XCTAssertEqual(cache.map(\.offset), [3])
        XCTAssertEqual(shared.keys.asArray(Float.self), originalKeys)
        XCTAssertEqual(shared.values.asArray(Float.self), originalValues)
    }

    func testTargetVerificationMatchesSerialDecodeAndForkRestoresPrefix() throws {
        MLXRandom.seed(79)
        let (target, assistant) = try makeModels()
        let prefix = target.makeAttentionCache()
        let prefill = target.forwardForSpeculation(
            inputIds: MLXArray([Int32(3), 4, 5]).reshaped(1, 3), cache: prefix
        )
        MLX.eval(prefill.logits, prefill.hidden)
        let firstToken = argMax(prefill.logits[0, -1, 0...]).item(Int.self)
        let draft = try assistant.draftBlock(
            lastToken: firstToken,
            hidden: target.speculativeDraftHidden(prefill.hidden[0..., (-1)..., 0...]),
            sharedKVStates: prefill.sharedKVStates, positionOffset: 3, blockSize: 4,
            baseModel: target, generationConfig: GenerationConfig(maxTokens: 8, temperature: 0, topP: 1),
            repetitionHistory: [3, 4, 5]
        )
        let tokens = [firstToken] + draft.tokens
        let verificationCache = prefix.map { $0.fork() }
        let verification = target.forwardForSpeculation(
            inputIds: MLXArray(tokens.map(Int32.init)).reshaped(1, tokens.count), cache: verificationCache
        )
        let serialCache = prefix.map { $0.fork() }
        let serialLogits = tokens.map { token in
            let logits = target.forward(inputIds: MLXArray([Int32(token)]).reshaped(1, 1), cache: serialCache)
            MLX.eval(logits)
            return logits
        }
        let serial = concatenated(serialLogits, axis: 1)
        MLX.eval(verification.logits, serial)
        XCTAssertLessThan(MLX.max(MLX.abs(verification.logits - serial)).item(Float.self), 1e-4)
        XCTAssertEqual(prefix.map(\.offset), [3])
        XCTAssertEqual(verificationCache.map(\.offset), [7])

        // A rejected draft resumes from the retained prefix plus accepted tokens.
        let restoredCache = prefix.map { $0.fork() }
        let accepted = target.forward(
            inputIds: MLXArray(tokens.prefix(2).map(Int32.init)).reshaped(1, 2), cache: restoredCache
        )
        MLX.eval(accepted)
        XCTAssertEqual(restoredCache.map(\.offset), [5])
        XCTAssertLessThan(
            MLX.max(MLX.abs(accepted - serial[0..., ..<2, 0...])).item(Float.self), 1e-4
        )
    }

    func testSharedSlidingWindowVerificationMatchesSerialDecodeAfterEviction() throws {
        try assertSlidingVerification(quantization: nil)
    }

    func testSharedQuantizedSlidingWindowVerificationMatchesSerialDecodeAfterEviction() throws {
        try assertSlidingVerification(
            quantization: Gemma4KVCacheQuantization(bits: 4, groupSize: 64, quantizedStart: 0)
        )
    }

    private func assertSlidingVerification(quantization: Gemma4KVCacheQuantization?) throws {
        MLXRandom.seed(89)
        var configJSON = textConfigJSON.replacingOccurrences(of: "full_attention", with: "sliding_attention")
        if quantization != nil {
            configJSON = configJSON.replacingOccurrences(of: "\"head_dim\": 4", with: "\"head_dim\": 64")
        }
        let config = try JSONDecoder().decode(Gemma4TextConfig.self, from: Data(configJSON.utf8))
        let target = Gemma4TextCausalLM(config: config)
        let prefix = target.makeAttentionCache(quantization: quantization)
        let prompt = target.forward(
            inputIds: MLXArray((1...8).map(Int32.init)).reshaped(1, 8), cache: prefix
        )
        MLX.eval(prompt)
        let tokens = [9, 10, 11, 12]
        let verificationCache = prefix.map { $0.fork() }
        let verification = target.forwardForSpeculation(
            inputIds: MLXArray(tokens.map(Int32.init)).reshaped(1, 4), cache: verificationCache
        )
        let serialCache = prefix.map { $0.fork() }
        let serialLogits = tokens.map { token in
            let logits = target.forward(inputIds: MLXArray([Int32(token)]).reshaped(1, 1), cache: serialCache)
            MLX.eval(logits)
            return logits
        }
        let serial = concatenated(serialLogits, axis: 1)
        MLX.eval(verification.logits, serial)
        XCTAssertLessThan(MLX.max(MLX.abs(verification.logits - serial)).item(Float.self), 1e-4)
        XCTAssertEqual(prefix.map(\.offset), [8])
        XCTAssertEqual(verificationCache.map(\.offset), [12])
    }

}
