import XCTest
@testable import MereRunCLI
@testable import MereRunCore

final class ModelBenchmarkCommandTests: XCTestCase {
    func testModelCommandExposesBenchmarkSubcommand() {
        let commandNames = Set(Model.configuration.subcommands.map { $0.configuration.commandName })
        XCTAssertTrue(commandNames.contains("benchmark"))
    }

    func testBenchmarkCommandExposesVLMSubcommand() {
        let commandNames = Set(ModelBenchmark.configuration.subcommands.map { $0.configuration.commandName })
        XCTAssertTrue(commandNames.contains("vlm"))
    }

    func testGemma4KVBenchmarkParsesDefaults() throws {
        let cmd = try ModelBenchmarkGemma4KV.parse([])

        XCTAssertEqual(cmd.model, Gemma4Resources.turboModelId)
        XCTAssertEqual(cmd.promptRepeat, 220)
        XCTAssertNil(cmd.promptRepeatValues)
        XCTAssertEqual(cmd.decodeTokens, 48)
        XCTAssertNil(cmd.decodeTokenValues)
        XCTAssertEqual(cmd.temperature, 0)
        XCTAssertEqual(cmd.topP, 1)
        XCTAssertFalse(cmd.json)
    }

    func testGemma4KVBenchmarkParsesOverrides() throws {
        let cmd = try ModelBenchmarkGemma4KV.parse([
            "--model", Gemma4Resources.turboModelId,
            "--model-root", "/tmp/gemma4",
            "--prompt", "Benchmark this",
            "--decode-tokens", "32",
            "--decode-token-values", "32,128",
            "--prompt-repeat-values", "64",
            "--temperature", "0.2",
            "--top-p", "0.7",
            "--json",
        ])

        XCTAssertEqual(cmd.model, Gemma4Resources.turboModelId)
        XCTAssertEqual(cmd.modelRoot, "/tmp/gemma4")
        XCTAssertEqual(cmd.prompt, "Benchmark this")
        XCTAssertEqual(cmd.decodeTokens, 32)
        XCTAssertEqual(cmd.decodeTokenValues, "32,128")
        XCTAssertEqual(cmd.promptRepeatValues, "64")
        XCTAssertEqual(cmd.temperature, 0.2)
        XCTAssertEqual(cmd.topP, 0.7)
        XCTAssertTrue(cmd.json)
    }

    func testVLMBenchmarkParsesDefaults() throws {
        let cmd = try ModelBenchmarkVLM.parse([])

        XCTAssertNil(cmd.models)
        XCTAssertEqual(cmd.dataset, .syntheticVQA)
        XCTAssertNil(cmd.lmmsTasks)
        XCTAssertNil(cmd.fixtureDir)
        XCTAssertNil(cmd.outputDir)
        XCTAssertNil(cmd.lmmsEvalRoot)
        XCTAssertEqual(cmd.lmmsEvalPython, "python3")
        XCTAssertFalse(cmd.dryRun)
        XCTAssertFalse(cmd.externalEndpoint)
        XCTAssertNil(cmd.baseURL)
        XCTAssertEqual(cmd.apiKey, "mere-run-local-eval")
        XCTAssertEqual(cmd.host, "127.0.0.1")
        XCTAssertEqual(cmd.port, 11934)
        XCTAssertNil(cmd.limit)
        XCTAssertFalse(cmd.logSamples)
        XCTAssertEqual(cmd.maxTokens, 24)
        XCTAssertEqual(cmd.contextSize, 4096)
        XCTAssertEqual(cmd.temperature, 0)
        XCTAssertEqual(cmd.topP, 1)
        XCTAssertFalse(cmd.json)
    }

    func testVLMBenchmarkParsesOverrides() throws {
        let cmd = try ModelBenchmarkVLM.parse([
            "--models", "vision-chat-gemma4-12b,text-chat-q36-nano",
            "--dataset", "mathvista-testmini",
            "--lmms-eval-root", "/tmp/lmms-eval",
            "--lmms-eval-python", "/tmp/venv/bin/python",
            "--dry-run",
            "--output-dir", "/tmp/vlm-results",
            "--external-endpoint",
            "--base-url", "http://127.0.0.1:11934/v1",
            "--api-key", "test-key",
            "--host", "localhost",
            "--port", "12000",
            "--limit", "4",
            "--log-samples",
            "--fixture-dir", "/tmp/vlm-fixtures",
            "--max-tokens", "12",
            "--context-size", "2048",
            "--temperature", "0.1",
            "--top-p", "0.8",
            "--json",
        ])

        XCTAssertEqual(cmd.models, "vision-chat-gemma4-12b,text-chat-q36-nano")
        XCTAssertEqual(cmd.dataset, .mathvistaTestMini)
        XCTAssertNil(cmd.lmmsTasks)
        XCTAssertEqual(cmd.lmmsEvalRoot, "/tmp/lmms-eval")
        XCTAssertEqual(cmd.lmmsEvalPython, "/tmp/venv/bin/python")
        XCTAssertTrue(cmd.dryRun)
        XCTAssertEqual(cmd.outputDir, "/tmp/vlm-results")
        XCTAssertTrue(cmd.externalEndpoint)
        XCTAssertEqual(cmd.baseURL, "http://127.0.0.1:11934/v1")
        XCTAssertEqual(cmd.apiKey, "test-key")
        XCTAssertEqual(cmd.host, "localhost")
        XCTAssertEqual(cmd.port, 12000)
        XCTAssertEqual(cmd.limit, "4")
        XCTAssertTrue(cmd.logSamples)
        XCTAssertEqual(cmd.fixtureDir, "/tmp/vlm-fixtures")
        XCTAssertEqual(cmd.maxTokens, 12)
        XCTAssertEqual(cmd.contextSize, 2048)
        XCTAssertEqual(cmd.temperature, 0.1)
        XCTAssertEqual(cmd.topP, 0.8)
        XCTAssertTrue(cmd.json)
    }

    func testVLMBenchmarkParsesRawLMMSTasks() throws {
        let cmd = try ModelBenchmarkVLM.parse([
            "--lmms-tasks", "mathvista_testmini,chartqa",
            "--dry-run",
            "--json",
        ])

        XCTAssertEqual(cmd.dataset, .syntheticVQA)
        XCTAssertEqual(cmd.lmmsTasks, "mathvista_testmini,chartqa")
        XCTAssertTrue(cmd.dryRun)
        XCTAssertTrue(cmd.json)
    }
}
