import Foundation
import XCTest
@testable import MereRunCLI

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
}
