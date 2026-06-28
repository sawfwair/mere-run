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

    func testBenchmarkCommandExposesCodeSubcommand() {
        let commandNames = Set(ModelBenchmark.configuration.subcommands.map { $0.configuration.commandName })
        XCTAssertTrue(commandNames.contains("code"))
    }

    func testCodeBenchmarkRequiresExecutionOptIn() {
        XCTAssertThrowsError(try ModelBenchmarkCode.parse([]))
    }

    func testCodeBenchmarkParsesDryRunDefaults() throws {
        let cmd = try ModelBenchmarkCode.parse(["--dry-run"])

        XCTAssertNil(cmd.models)
        XCTAssertEqual(cmd.suite, .humanEvalSlice)
        XCTAssertNil(cmd.tasks)
        XCTAssertEqual(cmd.maxTokens, 512)
        XCTAssertEqual(cmd.temperature, 0)
        XCTAssertEqual(cmd.topP, 1)
        XCTAssertEqual(cmd.executionTimeout, 5)
        XCTAssertEqual(cmd.python, "python3")
        XCTAssertTrue(cmd.dryRun)
        XCTAssertFalse(cmd.allowCodeExecution)
        XCTAssertFalse(cmd.json)
    }

    func testCodeBenchmarkParsesOverrides() throws {
        let cmd = try ModelBenchmarkCode.parse([
            "--models", "text-agent-ornith-9b,text-code-north-mini",
            "--suite", "humaneval-slice",
            "--tasks", "HumanEval/0,HumanEval/8",
            "--max-tokens", "256",
            "--temperature", "0.2",
            "--top-p", "0.8",
            "--execution-timeout", "3.5",
            "--python", "/tmp/venv/bin/python",
            "--dry-run",
            "--allow-code-execution",
            "--json",
        ])

        XCTAssertEqual(cmd.models, "text-agent-ornith-9b,text-code-north-mini")
        XCTAssertEqual(cmd.suite, .humanEvalSlice)
        XCTAssertEqual(cmd.tasks, "HumanEval/0,HumanEval/8")
        XCTAssertEqual(cmd.maxTokens, 256)
        XCTAssertEqual(cmd.temperature, 0.2)
        XCTAssertEqual(cmd.topP, 0.8)
        XCTAssertEqual(cmd.executionTimeout, 3.5)
        XCTAssertEqual(cmd.python, "/tmp/venv/bin/python")
        XCTAssertTrue(cmd.dryRun)
        XCTAssertTrue(cmd.allowCodeExecution)
        XCTAssertTrue(cmd.json)
    }

    func testBenchmarkCommandExposesGemma4MTPSubcommand() {
        let commandNames = Set(ModelBenchmark.configuration.subcommands.map { $0.configuration.commandName })
        XCTAssertTrue(commandNames.contains("gemma4-mtp"))
    }

    func testBenchmarkCommandExposesQ36MTPSubcommand() {
        let commandNames = Set(ModelBenchmark.configuration.subcommands.map { $0.configuration.commandName })
        XCTAssertTrue(commandNames.contains("q36-mtp"))
    }

    func testQ36MTPBenchmarkParsesDefaults() throws {
        let cmd = try ModelBenchmarkQ36MTP.parse([])

        XCTAssertEqual(cmd.model, Q35Resources.q36NanoModelId)
        XCTAssertNil(cmd.modelRoot)
        XCTAssertNil(cmd.prompt)
        XCTAssertNil(cmd.promptFile)
        XCTAssertEqual(cmd.promptRepeat, 150)
        XCTAssertNil(cmd.promptRepeatValues)
        XCTAssertEqual(cmd.decodeTokens, 32)
        XCTAssertNil(cmd.decodeTokenValues)
        XCTAssertEqual(cmd.temperature, 0)
        XCTAssertNil(cmd.temperatureValues)
        XCTAssertEqual(cmd.topP, 0.9)
        XCTAssertEqual(cmd.contextSize, Q35Resources.defaultContextLength)
        XCTAssertNil(cmd.mtpBlockSize)
        XCTAssertEqual(cmd.forcedMTPMinPromptTokens, 1)
        XCTAssertFalse(cmd.json)
    }

    func testQ36MTPBenchmarkParsesOverrides() throws {
        let cmd = try ModelBenchmarkQ36MTP.parse([
            "--model", Q35Resources.q36NanoModelId,
            "--model-root", "/tmp/q36",
            "--prompt", "Benchmark Q36",
            "--decode-tokens", "24",
            "--decode-token-values", "24,48",
            "--prompt-repeat-values", "64",
            "--temperature", "0.7",
            "--temperature-values", "0,0.7",
            "--top-p", "0.8",
            "--context-size", "8192",
            "--mtp-block-size", "6",
            "--forced-mtp-min-prompt-tokens", "2",
            "--json",
        ])

        XCTAssertEqual(cmd.model, Q35Resources.q36NanoModelId)
        XCTAssertEqual(cmd.modelRoot, "/tmp/q36")
        XCTAssertEqual(cmd.prompt, "Benchmark Q36")
        XCTAssertEqual(cmd.decodeTokens, 24)
        XCTAssertEqual(cmd.decodeTokenValues, "24,48")
        XCTAssertEqual(cmd.promptRepeatValues, "64")
        XCTAssertEqual(cmd.temperature, 0.7)
        XCTAssertEqual(cmd.temperatureValues, "0,0.7")
        XCTAssertEqual(cmd.topP, 0.8)
        XCTAssertEqual(cmd.contextSize, 8192)
        XCTAssertEqual(cmd.mtpBlockSize, 6)
        XCTAssertEqual(cmd.forcedMTPMinPromptTokens, 2)
        XCTAssertTrue(cmd.json)
    }

    func testGemma4MTPBenchmarkParsesDefaults() throws {
        let cmd = try ModelBenchmarkGemma4MTP.parse([])

        XCTAssertEqual(cmd.model, Gemma4Resources.twelveB4BitModelId)
        XCTAssertNil(cmd.modelRoot)
        XCTAssertNil(cmd.prompt)
        XCTAssertNil(cmd.promptFile)
        XCTAssertEqual(cmd.promptRepeat, 220)
        XCTAssertNil(cmd.promptRepeatValues)
        XCTAssertEqual(cmd.decodeTokens, 48)
        XCTAssertNil(cmd.decodeTokenValues)
        XCTAssertNil(cmd.mtpBlockSize)
        XCTAssertNil(cmd.mtpMinPromptTokens)
        XCTAssertFalse(cmd.json)
    }

    func testGemma4MTPBenchmarkParsesOverrides() throws {
        let cmd = try ModelBenchmarkGemma4MTP.parse([
            "--model", Gemma4Resources.twelveB4BitModelId,
            "--model-root", "/tmp/gemma4-4bit",
            "--prompt", "Benchmark MTP",
            "--decode-tokens", "32",
            "--decode-token-values", "32,128",
            "--prompt-repeat-values", "64",
            "--mtp-block-size", "6",
            "--mtp-min-prompt-tokens", "1",
            "--json",
        ])

        XCTAssertEqual(cmd.model, Gemma4Resources.twelveB4BitModelId)
        XCTAssertEqual(cmd.modelRoot, "/tmp/gemma4-4bit")
        XCTAssertEqual(cmd.prompt, "Benchmark MTP")
        XCTAssertEqual(cmd.decodeTokens, 32)
        XCTAssertEqual(cmd.decodeTokenValues, "32,128")
        XCTAssertEqual(cmd.promptRepeatValues, "64")
        XCTAssertEqual(cmd.mtpBlockSize, 6)
        XCTAssertEqual(cmd.mtpMinPromptTokens, 1)
        XCTAssertTrue(cmd.json)
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
