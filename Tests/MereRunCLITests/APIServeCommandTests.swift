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
        XCTAssertEqual(cmd.kvQuantScheme, "uniform")
        XCTAssertEqual(cmd.kvGroupSize, 64)
        XCTAssertEqual(cmd.quantizedKVStart, 5_000)
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
}
