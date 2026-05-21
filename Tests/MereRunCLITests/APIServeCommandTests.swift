import Foundation
import XCTest
@testable import MereRunCLI
@testable import MereRunCore

final class APIServeCommandTests: XCTestCase {
    func testAPICommandExposesServeSubcommand() {
        let commandNames = Set(API.configuration.subcommands.map { $0.configuration.commandName })
        XCTAssertEqual(commandNames, Set(["serve"]))
    }

    func testAPIServeParsesDefaults() throws {
        let cmd = try APIServe.parse([])

        XCTAssertEqual(cmd.port, 8080)
        XCTAssertEqual(cmd.host, "127.0.0.1")
        XCTAssertEqual(cmd.engine, .textCode)
        XCTAssertNil(cmd.model)
        XCTAssertNil(cmd.lora)
        XCTAssertNil(cmd.apiKey)
        XCTAssertEqual(cmd.rateLimitPerMinute, 60)
        XCTAssertEqual(cmd.contextSize, 32_768)
        XCTAssertNil(cmd.kvBits)
        XCTAssertNil(cmd.kvQuantScheme)
        XCTAssertNil(cmd.kvGroupSize)
        XCTAssertNil(cmd.quantizedKVStart)
    }

    func testAPIServeParsesOverrides() throws {
        let cmd = try APIServe.parse([
            "--host", "0.0.0.0",
            "--port", "11434",
            "--engine", "text-chat-gemma4",
            "--model-path", "/tmp/gemma4",
            "--lora", "/tmp/adapter.safetensors",
            "--api-key", "secret",
            "--rate-limit-per-minute", "120",
            "--context-size", "8192",
            "--kv-bits", "3.5",
            "--kv-quant-scheme", "turboquant",
            "--kv-group-size", "32",
            "--quantized-kv-start", "2048",
        ])

        XCTAssertEqual(cmd.host, "0.0.0.0")
        XCTAssertEqual(cmd.port, 11_434)
        XCTAssertEqual(cmd.engine, .textChatGemma4)
        XCTAssertEqual(cmd.model, "/tmp/gemma4")
        XCTAssertEqual(cmd.lora, "/tmp/adapter.safetensors")
        XCTAssertEqual(cmd.apiKey, "secret")
        XCTAssertEqual(cmd.rateLimitPerMinute, 120)
        XCTAssertEqual(cmd.contextSize, 8192)
        XCTAssertEqual(cmd.kvBits, 3.5)
        XCTAssertEqual(cmd.kvQuantScheme, "turboquant")
        XCTAssertEqual(cmd.kvGroupSize, 32)
        XCTAssertEqual(cmd.quantizedKVStart, 2048)
    }

    func testAPIServeGemma4TurboModelDefaultsToTurboQuantKVCache() throws {
        let cmd = try APIServe.parse([
            "--engine", "text-chat-gemma4",
            "--model-path", Gemma4Resources.turboModelId,
        ])

        let quantization = try cmd.resolveGemma4KVCacheQuantization()

        XCTAssertEqual(quantization.bits, Gemma4Resources.defaultTurboKVBits)
        XCTAssertEqual(quantization.scheme, .turboquant)
        XCTAssertEqual(quantization.groupSize, Gemma4Resources.defaultKVGroupSize)
        XCTAssertEqual(quantization.quantizedStart, Gemma4Resources.defaultTurboQuantizedKVStart)
    }

    func testAPIServeGemma4TurboKVFlagsOverrideIndependently() throws {
        let cmd = try APIServe.parse([
            "--engine", "text-chat-gemma4",
            "--model-path", Gemma4Resources.turboModelId,
            "--kv-quant-scheme", "uniform",
            "--kv-group-size", "32",
            "--quantized-kv-start", "128",
        ])

        let quantization = try cmd.resolveGemma4KVCacheQuantization()

        XCTAssertEqual(quantization.bits, Gemma4Resources.defaultTurboKVBits)
        XCTAssertEqual(quantization.scheme, .uniform)
        XCTAssertEqual(quantization.groupSize, 32)
        XCTAssertEqual(quantization.quantizedStart, 128)
    }

    func testHealthContractUsesStablePayload() {
        XCTAssertEqual(APIServerContract.healthStatus(), APIHealthStatus(status: "ok"))
    }

    func testModelsContractUsesProvidedModelID() {
        let response = APIServerContract.modelsResponse(
            modelId: "mererun-test-model",
            createdAt: Date(timeIntervalSince1970: 123)
        )

        XCTAssertEqual(response.object, "list")
        XCTAssertEqual(response.data.count, 1)
        XCTAssertEqual(response.data.first?.id, "mererun-test-model")
        XCTAssertEqual(response.data.first?.object, "model")
        XCTAssertEqual(response.data.first?.owned_by, "mere.run")
        XCTAssertEqual(response.data.first?.created, 123)
    }

    func testChatRequestValidationAcceptsBoundedDefaults() throws {
        let request = OpenAIChatRequest(
            model: "mererun-test-model",
            messages: [OpenAIChatMessage(role: "user", content: "hello")]
        )

        let chatRequest = try APIServerContract.chatRequest(
            from: request,
            fallbackLoraPath: " /tmp/default.safetensors ",
            contextSize: 4_096
        )

        XCTAssertEqual(chatRequest.maxTokens, APIServerContract.defaultMaxTokens)
        XCTAssertEqual(chatRequest.temperature, 1.0)
        XCTAssertEqual(chatRequest.topP, 0.95)
        XCTAssertEqual(chatRequest.messages, [ChatMessage(role: .user, content: "hello")])
        guard case .local(let path, let scale) = chatRequest.lora else {
            return XCTFail("Expected server-configured LoRA to be applied")
        }
        XCTAssertEqual(path, "/tmp/default.safetensors")
        XCTAssertEqual(scale, 1.0)
    }

    func testChatRequestDecodesStructuredOpenAIContentParts() throws {
        let data = """
        {
          "model": "mererun-test-model",
          "messages": [
            {
              "role": "user",
              "content": [
                { "type": "text", "text": "hello" },
                { "type": "input_text", "text": "setup mere.run" }
              ]
            },
            {
              "role": "assistant"
            }
          ]
        }
        """.data(using: .utf8)!

        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)
        let chatRequest = try APIServerContract.chatRequest(
            from: request,
            fallbackLoraPath: nil,
            contextSize: 4_096
        )

        XCTAssertEqual(chatRequest.messages[0].content, "hello\nsetup mere.run")
        XCTAssertEqual(chatRequest.messages[1].content, "")
    }

    func testChatRequestDecodesModernOpenAIFieldsAndUnknownFields() throws {
        let data = """
        {
          "model": "mererun-test-model",
          "messages": [
            { "role": "developer", "content": "Follow repo rules." },
            { "role": "user", "content": "hello" }
          ],
          "max_completion_tokens": 77,
          "stream_options": { "include_usage": true },
          "parallel_tool_calls": true,
          "metadata": { "client": "test" },
          "reasoning_effort": "low",
          "x-client-extra": { "kept": true }
        }
        """.data(using: .utf8)!

        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)

        XCTAssertEqual(request.max_completion_tokens, 77)
        XCTAssertEqual(request.stream_options?.include_usage, true)
        XCTAssertEqual(request.parallel_tool_calls, true)
        XCTAssertEqual(request.metadata?["client"], "test")
        XCTAssertEqual(request.reasoning_effort, "low")
        XCTAssertEqual(request.unknownFields["x-client-extra"]?.objectValue?["kept"]?.boolValue, true)
    }

    func testChatRequestMapsDeveloperRoleAndMaxCompletionTokens() throws {
        let request = OpenAIChatRequest(
            model: "mererun-test-model",
            messages: [
                OpenAIChatMessage(role: "developer", content: "Follow repo rules."),
                OpenAIChatMessage(role: "user", content: "hello"),
            ],
            max_completion_tokens: 77
        )

        let chatRequest = try APIServerContract.chatRequest(
            from: request,
            fallbackLoraPath: nil,
            contextSize: 4_096
        )

        XCTAssertEqual(chatRequest.maxTokens, 77)
        XCTAssertEqual(chatRequest.messages[0].role, .system)
        XCTAssertEqual(chatRequest.messages[0].content, "Follow repo rules.")
    }

    func testChatRequestValidatesImageContentPartsAgainstEngineCapability() throws {
        let data = """
        {
          "model": "mererun-test-model",
          "messages": [
            {
              "role": "user",
              "content": [
                { "type": "text", "text": "describe" },
                { "type": "image_url", "image_url": { "url": "file:///tmp/image.png" } }
              ]
            }
          ]
        }
        """.data(using: .utf8)!

        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: data)

        XCTAssertThrowsError(
            try APIServerContract.chatRequest(
                from: request,
                fallbackLoraPath: nil,
                contextSize: 4_096,
                capabilities: .localText
            )
        ) { error in
            XCTAssertTrue(error.localizedDescription.contains("image content"))
        }

        let chatRequest = try APIServerContract.chatRequest(
            from: request,
            fallbackLoraPath: nil,
            contextSize: 4_096,
            capabilities: .localTextWithToolsAndVision
        )
        XCTAssertEqual(chatRequest.messages[0].content, "describe")
        XCTAssertEqual(chatRequest.messages[0].imageUrl, "file:///tmp/image.png")
    }

    func testChatRequestMapsSupportedOpenAITools() throws {
        let tool = OpenAIChatTool(
            function: OpenAIChatToolFunction(
                name: "lookup",
                description: "Look up a value.",
                parameters: .object([
                    "type": .string("object"),
                    "properties": .object([
                        "query": .object([
                            "type": .string("string"),
                            "description": .string("Search query"),
                        ]),
                    ]),
                    "required": .array([.string("query")]),
                ])
            )
        )
        let request = OpenAIChatRequest(
            model: "mererun-test-model",
            messages: [OpenAIChatMessage(role: "user", content: "hello")],
            tools: [tool],
            tool_choice: .mode("auto")
        )

        XCTAssertThrowsError(
            try APIServerContract.chatRequest(
                from: request,
                fallbackLoraPath: nil,
                contextSize: 4_096,
                capabilities: .localText
            )
        ) { error in
            XCTAssertTrue(error.localizedDescription.contains("tools"))
        }

        let chatRequest = try APIServerContract.chatRequest(
            from: request,
            fallbackLoraPath: nil,
            contextSize: 4_096,
            capabilities: .localTextWithTools
        )
        XCTAssertEqual(chatRequest.tools?.first?.name, "lookup")
        XCTAssertEqual(chatRequest.tools?.first?.parameters["query"]?.description, "Search query")
        XCTAssertEqual(chatRequest.tools?.first?.required, ["query"])
    }

    func testChatRequestValidatesStructuredOutputsAgainstEngineCapability() throws {
        let request = OpenAIChatRequest(
            model: "mererun-test-model",
            messages: [OpenAIChatMessage(role: "user", content: "hello")],
            response_format: OpenAIResponseFormat(type: "json_object")
        )

        XCTAssertThrowsError(
            try APIServerContract.chatRequest(
                from: request,
                fallbackLoraPath: nil,
                contextSize: 4_096,
                capabilities: .localText
            )
        ) { error in
            XCTAssertTrue(error.localizedDescription.contains("response_format"))
        }

        let chatRequest = try APIServerContract.chatRequest(
            from: request,
            fallbackLoraPath: nil,
            contextSize: 4_096,
            capabilities: .localTextWithStructuredJSON
        )
        XCTAssertTrue(chatRequest.requiresJSON)
    }

    func testChatRequestRejectsUnsupportedHighImpactFields() {
        let request = OpenAIChatRequest(
            model: "mererun-test-model",
            messages: [OpenAIChatMessage(role: "user", content: "hello")],
            stop: .string("END"),
            seed: 42,
            presence_penalty: 0.5,
            logprobs: true,
            reasoning_effort: "low",
            think: true
        )

        XCTAssertThrowsError(
            try APIServerContract.chatRequest(from: request, fallbackLoraPath: nil, contextSize: 4_096)
        ) { error in
            let message = error.localizedDescription
            XCTAssertTrue(
                message.contains("stop")
                    || message.contains("seed")
                    || message.contains("presence_penalty")
                    || message.contains("logprobs")
                    || message.contains("reasoning_effort")
                    || message.contains("thinking")
            )
        }
    }

    func testStreamingUsageOptionHonorsCapabilities() throws {
        let request = OpenAIChatRequest(
            model: "mererun-test-model",
            messages: [OpenAIChatMessage(role: "user", content: "hello")],
            stream_options: OpenAIStreamOptions(include_usage: true)
        )

        XCTAssertTrue(
            try APIServerContract.includeUsageInStreaming(
                request,
                capabilities: .localText
            )
        )

        XCTAssertThrowsError(
            try APIServerContract.includeUsageInStreaming(
                request,
                capabilities: APIEngineCapabilities(supportsUsageInStreaming: false)
            )
        ) { error in
            XCTAssertTrue(error.localizedDescription.contains("stream_options.include_usage"))
        }
    }

    func testChatRequestValidationRejectsOversizedMaxTokens() {
        let request = OpenAIChatRequest(
            model: "mererun-test-model",
            messages: [OpenAIChatMessage(role: "user", content: "hello")],
            max_tokens: 4_097
        )

        XCTAssertThrowsError(
            try APIServerContract.chatRequest(from: request, fallbackLoraPath: nil, contextSize: 4_096)
        ) { error in
            XCTAssertTrue(error.localizedDescription.contains("max_tokens"))
        }
    }

    func testChatRequestValidationRejectsPerRequestLora() {
        let request = OpenAIChatRequest(
            model: "mererun-test-model",
            messages: [OpenAIChatMessage(role: "user", content: "hello")],
            lora: "/tmp/request-controlled.safetensors"
        )

        XCTAssertThrowsError(
            try APIServerContract.chatRequest(from: request, fallbackLoraPath: nil, contextSize: 4_096)
        ) { error in
            XCTAssertTrue(error.localizedDescription.contains("lora"))
        }
    }

    func testChatRequestValidationRejectsInvalidSamplingValues() {
        let badTemperature = OpenAIChatRequest(
            model: "mererun-test-model",
            messages: [OpenAIChatMessage(role: "user", content: "hello")],
            temperature: 3.0
        )
        let badTopP = OpenAIChatRequest(
            model: "mererun-test-model",
            messages: [OpenAIChatMessage(role: "user", content: "hello")],
            top_p: 1.5
        )

        XCTAssertThrowsError(
            try APIServerContract.chatRequest(from: badTemperature, fallbackLoraPath: nil, contextSize: 4_096)
        ) { error in
            XCTAssertTrue(error.localizedDescription.contains("temperature"))
        }
        XCTAssertThrowsError(
            try APIServerContract.chatRequest(from: badTopP, fallbackLoraPath: nil, contextSize: 4_096)
        ) { error in
            XCTAssertTrue(error.localizedDescription.contains("top_p"))
        }
    }

    func testChatContentTypeValidationRejectsBrowserSimpleTypes() {
        XCTAssertFalse(APIServerContract.acceptsJSONContentType(nil))
        XCTAssertFalse(APIServerContract.acceptsJSONContentType(""))
        XCTAssertFalse(APIServerContract.acceptsJSONContentType("text/plain"))
        XCTAssertFalse(APIServerContract.acceptsJSONContentType("application/x-www-form-urlencoded"))
        XCTAssertFalse(APIServerContract.acceptsJSONContentType("multipart/form-data; boundary=abc"))
    }

    func testChatContentTypeValidationAcceptsJSONTypes() {
        XCTAssertTrue(APIServerContract.acceptsJSONContentType("application/json"))
        XCTAssertTrue(APIServerContract.acceptsJSONContentType("application/json; charset=utf-8"))
        XCTAssertTrue(APIServerContract.acceptsJSONContentType("application/vnd.openai+json"))
    }

    func testStreamingStatusMessagesAreNotTreatedAsTokens() {
        XCTAssertTrue(APIServerContract.isStreamingStatusMessage("Generating..."))
        XCTAssertTrue(APIServerContract.isStreamingStatusMessage("Generating response"))
        XCTAssertTrue(APIServerContract.isStreamingStatusMessage("Retrying generation"))
        XCTAssertTrue(APIServerContract.isStreamingStatusMessage("DS4 chat completion"))
        XCTAssertFalse(APIServerContract.isStreamingStatusMessage("Actual generated text"))
    }
}
