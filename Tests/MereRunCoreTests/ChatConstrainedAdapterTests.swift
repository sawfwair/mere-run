import Foundation
import MLX
import XCTest
@testable import MereRunCore
@testable import MereRunQwenModel
@testable import MereRunGemmaModel

final class ChatConstrainedAdapterTests: MereRunCoreTestCase {
    func testQwenAndGemmaGenerationUseTheSchemaDecoder() async throws {
        let root = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: root) }
        try Data(#"{"model":{"type":"BPE","unk_token":"[UNK]","vocab":{"[UNK]":0,"{":1,"\"ok\"":2,":":3,"true":4,"}":5},"merges":[]},"decoder":{"type":"Fuse"}}"#.utf8)
            .write(to: root.appendingPathComponent("tokenizer.json"))
        try Data(#"{"tokenizer_class":"Qwen2Tokenizer","pad_token":"[UNK]","model_max_length":4096,"chat_template":"x"}"#.utf8)
            .write(to: root.appendingPathComponent("tokenizer_config.json"))
        let schema: [String: ChatJSONValue] = ["type": .string("object"), "additionalProperties": .bool(false),
            "properties": .object(["ok": .object(["type": .string("boolean")])]), "required": .array([.string("ok")])]
        var request = ChatRequest(messages: [.init(role: .user, content: "answer")], maxTokens: 16,
            temperature: 0, showThinking: false, requiresJSON: true, maxContextTokens: 128)
        request.outputConstraint = try ChatOutputConstraint(schema: schema)
        request.preparedPrompt = try ChatPreparedPrompt(tokens: [1], contextLimit: 128)
        let qwen = Q35Generator(modelId: Q35Resources.q38TwentySevenB4BitModelId,
            prefixKVCacheEnabled: false, continuousBatchingEnabled: false)
        let qwenResponse: ChatResponse
        do { qwenResponse = try await qwen.fixtureResponse(request, root: root) }
        catch { XCTFail("Qwen constrained generation failed: \(error)"); return }
        XCTAssertEqual(qwenResponse.response, #"{"ok":true}"#)
        XCTAssertEqual(qwenResponse.generationEvents, [.text(#"{"ok":true}"#)])
        let gemma = Gemma4Generator(modelId: Gemma4Resources.defaultModelId)
        let gemmaResponse: ChatResponse
        do { gemmaResponse = try await gemma.fixtureResponse(request, root: root) }
        catch { XCTFail("Gemma constrained generation failed: \(error)"); return }
        XCTAssertEqual(gemmaResponse.response, #"{"ok":true}"#)
        XCTAssertEqual(gemmaResponse.generationEvents, [.text(#"{"ok":true}"#)])
    }
}

private extension Q35Generator {
    func fixtureResponse(_ request: ChatRequest, root: URL) async throws -> ChatResponse {
        let config = try JSONDecoder().decode(Q35Config.self, from: Data(#"""
        {"model_type":"qwen3_5","text_config":{
          "model_type":"qwen3_5_text","hidden_size":32,"intermediate_size":64,"num_hidden_layers":2,
          "num_attention_heads":4,"num_key_value_heads":2,"head_dim":32,
          "layer_types":["linear_attention","full_attention"],
          "linear_num_key_heads":1,"linear_num_value_heads":2,"linear_key_head_dim":8,"linear_value_head_dim":8,
          "linear_conv_kernel_dim":4,"attention_bias":false,"attention_dropout":0,"attn_output_gate":true,"eos_token_id":63,"vocab_size":64,"max_position_embeddings":4096,"rms_norm_eps":0.000001,
          "rope_parameters":{"rope_theta":10000,"partial_rotary_factor":1}}}
        """#.utf8))
        let fixture = Q35Model(config: config)
        fixture.update(parameters: fixture.parameters().mapValues { MLXArray.zeros(like: $0) })
        model = fixture; loadedConfig = config
        tokenizerAndTemplate = try Q35TokenizerAndTemplate.load(from: root)
        XCTAssertEqual(tokenizerAndTemplate?.decode(tokens: [1, 2, 3, 4, 5]), #"{"ok":true}"#)
        let probe = fixture.forwardPrefill(MLXArray([Int32(1)]).reshaped(1, 1), cache: makeLayerCaches(config: config))
        XCTAssertTrue(probe.logits.asArray(Float.self).allSatisfy(\.isFinite))
        return try await generate(request, progressHandler: nil, maxContextLength: 128)
    }
}

private extension Gemma4Generator {
    func fixtureResponse(_ request: ChatRequest, root: URL) async throws -> ChatResponse {
        let config = try JSONDecoder().decode(Gemma4Config.self, from: Data(#"""
        {"model_type":"gemma4","eos_token_id":[63],"text_config":{
          "model_type":"gemma4_text","hidden_size":32,"num_hidden_layers":2,"intermediate_size":32,
          "num_attention_heads":4,"num_key_value_heads":2,"head_dim":8,"max_position_embeddings":128,
          "rms_norm_eps":0.000001,"vocab_size":64,"vocab_size_per_layer_input":64,"hidden_size_per_layer_input":0,
          "sliding_window":32,"layer_types":["sliding_attention","full_attention"],"tie_word_embeddings":true}}
        """#.utf8))
        let fixture = Gemma4TextCausalLM(config: config.textConfig)
        fixture.update(parameters: fixture.parameters().mapValues { MLXArray.ones(like: $0) * 0.01 })
        model = fixture; loadedConfig = config
        let probe = fixture(MLXArray([Int32(1)]).reshaped(1, 1))
        XCTAssertTrue(probe.asArray(Float.self).allSatisfy(\.isFinite))
        tokenizerAndTemplate = try await Gemma4TokenizerAndTemplate.load(from: root)
        XCTAssertEqual(tokenizerAndTemplate?.decode(tokens: [1, 2, 3, 4, 5]), #"{"ok":true}"#)
        return try await generate(request, progressHandler: nil, maxContextLength: 128)
    }
}
