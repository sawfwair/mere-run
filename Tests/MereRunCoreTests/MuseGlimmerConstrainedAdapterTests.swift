import Foundation
import MLX
import XCTest
@testable import MereRunCore

final class MuseGlimmerConstrainedAdapterTests: MereRunCoreTestCase {
    func testLoadedNativeGeneratorConstrainedAnswerAndStrictTool() async throws {
        let previousDFlash = ProcessInfo.processInfo.environment["MERERUN_MUSE_GLIMMER_DFLASH"]
        setenv("MERERUN_MUSE_GLIMMER_DFLASH", "0", 1)
        defer {
            if let previousDFlash { setenv("MERERUN_MUSE_GLIMMER_DFLASH", previousDFlash, 1) }
            else { unsetenv("MERERUN_MUSE_GLIMMER_DFLASH") }
        }
        let root = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: root) }
        var config = try JSONDecoder().decode([String: ChatJSONValue].self, from: Data(MuseGlimmerTests.configJSON.utf8))
        config.removeValue(forKey: "quantization")
        var textConfig = try XCTUnwrap(config["text_config"]?.objectValue)
        textConfig["max_position_embeddings"] = .number(4096)
        config["text_config"] = .object(textConfig)
        let configData = try JSONEncoder().encode(config)
        try configData.write(to: root.appendingPathComponent("config.json"))
        let fixture = MuseGlimmerModel(config: try JSONDecoder().decode(MuseGlimmerConfig.self, from: configData))
        fixture.update(parameters: fixture.parameters().mapValues { MLXArray.zeros(like: $0) })
        try MLX.save(arrays: Dictionary(uniqueKeysWithValues: fixture.parameters().flattened()), url: root.appendingPathComponent("model.safetensors"))
        try Data(#"{"model":{"type":"BPE","unk_token":"[UNK]","vocab":{"[UNK]":0," to=user<|message|>":1,"<|eot|>":2,"{":3,"\"ok\"":4,":":5,"true":6,"}":7," to=edit<|message|><atem:function_calls><atem:invoke name=\"edit\"><atem:parameter name=\"replace\">":8,"</atem:parameter></atem:invoke></atem:function_calls><|eot|>":9,"<|patch|>":28,"<|image_start|>":29,"<|image_end|>":30},"merges":[]},"decoder":{"type":"Fuse"}}"#.utf8)
            .write(to: root.appendingPathComponent("tokenizer.json"))
        try Data(#"{"tokenizer_class":"Qwen2Tokenizer","pad_token":"[UNK]","eos_token":"<|eot|>","model_max_length":4096,"chat_template":"{ {{ messages[0]['content'] }}{{ tools|default([])|tojson }} }"}"#.utf8)
            .write(to: root.appendingPathComponent("tokenizer_config.json"))
        try Data(#"{"eos_token_id":[2],"max_length":4096}"#.utf8).write(to: root.appendingPathComponent("generation_config.json"))
        try Data("{}".utf8).write(to: root.appendingPathComponent("processor_config.json"))
        let schema: [String: ChatJSONValue] = ["type": .string("object"), "additionalProperties": .bool(false),
            "properties": .object(["ok": .object(["type": .string("boolean")])]), "required": .array([.string("ok")])]
        var request = ChatRequest(messages: [.init(role: .user, content: "answer")], maxTokens: 16,
            temperature: MuseGlimmerResources.recommendedTemperature, topP: MuseGlimmerResources.recommendedTopP,
            requiresJSON: true, maxContextTokens: 4096)
        request.outputConstraint = try ChatOutputConstraint(schema: schema)
        let generator = MuseGlimmerGenerator()
        let prepared = try await generator.preparePrompt(request, modelPath: root.path)
        var ordinary = request
        ordinary.outputConstraint = nil; ordinary.requiresJSON = false
        let ordinaryPrompt = try await generator.preparePrompt(ordinary, modelPath: root.path)
        XCTAssertGreaterThan(prepared.tokens.count, ordinaryPrompt.tokens.count)
        request.preparedPrompt = prepared
        let response = try await generator.chat(request, modelPath: root.path, progressHandler: nil)
        XCTAssertEqual(response.response, #"{"ok":true}"#)
        XCTAssertEqual(response.finishReason, .stop)
        XCTAssertEqual(response.promptTokens, prepared.tokens.count)
        XCTAssertEqual(response.tokensGenerated, 7)
        XCTAssertEqual(response.generationEvents?.compactMap { if case .text(let text) = $0 { text } else { nil } }.joined(), response.response)
        let tool = ToolDefinition(name: "edit", description: "edit", parameterSchema: ["type": .string("object"),
            "additionalProperties": .bool(false), "properties": .object(["replace": .object(["type": .string("boolean")])]),
            "required": .array([.string("replace")])])
        request.tools = [tool]; request.toolChoice = .required
        request.outputConstraint = try ChatOutputConstraint.tools([tool], choice: .required, parallel: false)
        request.preparedPrompt = nil
        let toolPrompt = try await generator.preparePrompt(request, modelPath: root.path)
        XCTAssertNotEqual(toolPrompt.tokens, prepared.tokens)
        request.preparedPrompt = toolPrompt
        let call = try await generator.chat(request, modelPath: root.path, progressHandler: nil)
        XCTAssertEqual(call.promptTokens, toolPrompt.tokens.count)
        XCTAssertEqual(call.toolCalls?.count, 1)
        XCTAssertEqual(call.toolCalls?.first?.name, "edit")
        XCTAssertEqual(call.toolCalls?.first?.jsonArguments?["replace"], .bool(true))
        XCTAssertEqual(call.response, "")
        XCTAssertEqual(call.finishReason, .stop)
        let stats = await generator.dflashStats()
        XCTAssertFalse(stats.active)
        await generator.unload()
    }
}
