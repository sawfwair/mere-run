import Foundation
import MLX
import XCTest
@testable import MereRunCore
@testable import MereRunQwenModel

final class Q35DFlash2StreamingTests: MereRunCoreTestCase {
    func testConfirmedDFlashTokensStreamReasoningAndSplitUnicodeBeforeCompletion() throws {
        let target = Q35Model(config: try JSONDecoder().decode(Q35Config.self, from: Data(Self.target.utf8)))
        target.update(parameters: target.parameters().mapValues { MLXArray.zeros(like: $0) })
        let draft = Q35DFlash2Model(config: try JSONDecoder().decode(Q35DFlash2Config.self, from: Data(Self.draft.utf8)))
        draft.update(parameters: draft.parameters().mapValues { MLXArray.zeros(like: $0) })
        let caches: [Q35LayerCache?] = [.linear(Q35LinearCache()), .full(KVCacheSimple())]
        let prompt = [1, 2, 3]
        let output = target.forwardPrefill(
            MLXArray(prompt.map(Int32.init)).reshaped(1, prompt.count), cache: caches, captureLayerIndices: [0, 1]
        )
        let history = draft.makeCache()
        draft.appendTargetContext(output.capturedHiddenStates, cache: history)
        let observer = NativeChatOutputObserver(initialReasoning: true, parseTools: false)
        let recorded = RecordedEvents()
        let bytes = Array("work 🧮</think>42".utf8)
        let result = try Q35Generator.decodeDFlash2(
            model: target, draft: draft, draftCache: history, initialLogits: output.logits,
            layerCaches: caches, eosSet: [], tokenBudget: bytes.count,
            generationConfig: GenerationConfig(temperature: 0, repetitionPenalty: 1), promptTokens: prompt,
            decodeToken: { _ in "\u{FFFD}" },
            decodeTokens: { String(decoding: bytes.prefix($0.count), as: UTF8.self) },
            progressHandler: { observer.observe($0, handler: { recorded.append($0) }) }
        )
        XCTAssertEqual(result.generatedTokens.count, bytes.count)
        XCTAssertEqual(result.acceleration?.route, "dflash2")
        // Check callbacks before finish: final reconciliation must not hide a
        // decoder that buffers everything until the response is assembled.
        XCTAssertEqual(recorded.events.compactMap { if case .reasoning(let value) = $0 { return value }; return nil }.joined(), "work 🧮")
        XCTAssertEqual(recorded.events.compactMap { if case .text(let value) = $0 { return value }; return nil }.joined(), "42")
        XCTAssertEqual(observer.finish(String(decoding: bytes, as: UTF8.self), handler: nil), [.reasoning("work 🧮"), .text("42")])
    }

    private final class RecordedEvents: @unchecked Sendable {
        private let lock = NSLock()
        private var values: [ChatGenerationEvent] = []
        var events: [ChatGenerationEvent] { lock.withLock { values } }
        func append(_ progress: ChatProgress) {
            if let event = progress.event { lock.withLock { values.append(event) } }
        }
    }

    private static let target = #"""
    {"model_type":"qwen3_5","text_config":{
      "model_type":"qwen3_5_text","hidden_size":32,"intermediate_size":64,"num_hidden_layers":2,
      "num_attention_heads":4,"num_key_value_heads":2,"head_dim":32,
      "layer_types":["linear_attention","full_attention"],
      "linear_num_key_heads":1,"linear_num_value_heads":2,
      "linear_key_head_dim":8,"linear_value_head_dim":8,"linear_conv_kernel_dim":4,
      "attention_bias":false,"attention_dropout":0,"attn_output_gate":true,"eos_token_id":63,"vocab_size":64,
      "max_position_embeddings":4096,"rms_norm_eps":0.000001,
      "rope_parameters":{"rope_theta":10000,"partial_rotary_factor":1}}}
    """#

    private static let draft = #"""
    {"architectures":["DFlash2DraftModel"],"model_type":"qwen3","is_causal":false,
     "attention_bias":false,"hidden_act":"silu","tie_word_embeddings":false,
     "hidden_size":32,"intermediate_size":64,"num_hidden_layers":1,"num_target_layers":2,
     "num_attention_heads":4,"num_key_value_heads":2,"head_dim":8,"vocab_size":64,
     "rms_norm_eps":0.000001,"sliding_window":4,"layer_types":["sliding_attention"],
     "rope_parameters":{"rope_theta":10000,"rope_type":"default"},
     "dflash_config":{"block_size":4,"conv_group_size":16,"conv_kernel_size":2,
       "mask_token_id":63,"selector_rank":1,"selector_top_k":2,"target_layer_ids":[0,1]}}
    """#
}
