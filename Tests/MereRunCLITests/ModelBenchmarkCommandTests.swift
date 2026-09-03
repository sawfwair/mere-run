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

    func testBenchmarkCommandExposesChatSubcommand() {
        let commandNames = Set(ModelBenchmark.configuration.subcommands.map { $0.configuration.commandName })
        XCTAssertTrue(commandNames.contains("chat"))
    }

    func testBenchmarkCommandExposesFusedSubcommands() {
        let commandNames = Set(ModelBenchmark.configuration.subcommands.map { $0.configuration.commandName })
        XCTAssertTrue(commandNames.contains("fused"))
        XCTAssertTrue(commandNames.contains("fused-fixture"))
    }

    func testFusedBenchmarkParsesQualityDefaultsWithoutGreedyControls() throws {
        let command = try ModelBenchmarkFused.parse(["--dry-run"])

        XCTAssertEqual(command.suite, .lite)
        XCTAssertNil(command.trials)
        XCTAssertEqual(command.logprobs, .summary)
        XCTAssertEqual(command.topLogprobs, 5)
        XCTAssertEqual(command.performanceLane, .none)
        XCTAssertTrue(command.dryRun)
        XCTAssertFalse(command.allowCodeExecution)
    }

    func testFusedBenchmarkParsesCheckpointedBoundedResume() throws {
        let command = try ModelBenchmarkFused.parse([
            "--dry-run",
            "--checkpoint", "/tmp/mere-fused-checkpoint.json",
            "--resume",
            "--case-trial-limit", "3",
        ])

        XCTAssertEqual(command.checkpoint, "/tmp/mere-fused-checkpoint.json")
        XCTAssertTrue(command.resume)
        XCTAssertEqual(command.caseTrialLimit, 3)
    }

    func testFusedBenchmarkRejectsUnboundedResumeControls() {
        XCTAssertThrowsError(try ModelBenchmarkFused.parse(["--dry-run", "--resume"]))
        XCTAssertThrowsError(try ModelBenchmarkFused.parse([
            "--dry-run",
            "--case-trial-limit", "1",
        ]))
        XCTAssertThrowsError(try ModelBenchmarkFused.parse([
            "--dry-run",
            "--checkpoint", "/tmp/mere-fused-checkpoint.json",
            "--case-trial-limit", "0",
        ]))
    }

    func testBenchmarkCommandExposesToolCallsSubcommand() {
        let commandNames = Set(ModelBenchmark.configuration.subcommands.map { $0.configuration.commandName })
        XCTAssertTrue(commandNames.contains("tool-calls"))
    }

    func testBenchmarkCommandExposesToolContinuationsSubcommand() {
        let commandNames = Set(ModelBenchmark.configuration.subcommands.map { $0.configuration.commandName })
        XCTAssertTrue(commandNames.contains("tool-continuations"))
    }

    func testBenchmarkCommandExposesCodeSubcommand() {
        let commandNames = Set(ModelBenchmark.configuration.subcommands.map { $0.configuration.commandName })
        XCTAssertTrue(commandNames.contains("code"))
    }

    func testBenchmarkCommandExposesLagunaDFlashSubcommand() {
        let commandNames = Set(ModelBenchmark.configuration.subcommands.map {
            $0.configuration.commandName
        })
        XCTAssertTrue(commandNames.contains("laguna-dflash"))
    }

    func testLagunaDFlashBenchmarkParsesDefaults() throws {
        let command = try ModelBenchmarkLagunaDFlash.parse([
            "--laguna-path", "/tmp/Laguna-S-2.1-NVFP4-mlx",
            "--laguna-dflash-path", "/tmp/Laguna-S-2.1-DFlash",
        ])

        XCTAssertEqual(command.decodeTokenValues, "8,12,16,24,32,48")
        XCTAssertEqual(command.repetitions, 3)
        XCTAssertEqual(command.lagunaDflashTokens, 12)
        XCTAssertEqual(command.temperature, 0)
        XCTAssertEqual(command.topP, 1)
        XCTAssertEqual(command.topK, 0)
        XCTAssertEqual(command.minP, LagunaResources.recommendedMinP)
        XCTAssertEqual(command.contextSize, 4_096)
        XCTAssertEqual(command.fixture, .deterministicProse)
        XCTAssertNil(command.prompt)
        XCTAssertNil(command.promptFile)
        XCTAssertNil(command.concurrencyValues)
        XCTAssertEqual(command.warmupRepetitions, 1)
        XCTAssertFalse(command.mixedFixtures)
        XCTAssertFalse(command.includeAutomatic)
        XCTAssertFalse(command.logResponses)
        XCTAssertFalse(command.json)
    }

    func testLagunaDFlashBenchmarkRejectsInvalidMatrix() {
        XCTAssertThrowsError(try ModelBenchmarkLagunaDFlash.parse([
            "--laguna-path", "/tmp/Laguna-S-2.1-NVFP4-mlx",
            "--laguna-dflash-path", "/tmp/Laguna-S-2.1-DFlash",
            "--decode-token-values", "8,0,32",
        ]))
        XCTAssertThrowsError(try ModelBenchmarkLagunaDFlash.parse([
            "--laguna-path", "/tmp/Laguna-S-2.1-NVFP4-mlx",
            "--laguna-dflash-path", "/tmp/Laguna-S-2.1-DFlash",
            "--repetitions", "0",
        ]))
    }

    func testLagunaDFlashBenchmarkParsesCodeCompletionFixture() throws {
        let command = try ModelBenchmarkLagunaDFlash.parse([
            "--laguna-path", "/tmp/Laguna-S-2.1-NVFP4-mlx",
            "--laguna-dflash-path", "/tmp/Laguna-S-2.1-DFlash",
            "--fixture", "code-completion",
        ])

        XCTAssertEqual(command.fixture, .codeCompletion)
    }

    func testLagunaDFlashBenchmarkParsesSamplingRecipe() throws {
        let command = try ModelBenchmarkLagunaDFlash.parse([
            "--laguna-path", "/tmp/Laguna-S-2.1-NVFP4-mlx",
            "--laguna-dflash-path", "/tmp/Laguna-S-2.1-DFlash",
            "--temperature", "1",
            "--top-p", "0.95",
            "--top-k", "20",
            "--min-p", "0.05",
        ])

        XCTAssertEqual(command.temperature, 1)
        XCTAssertEqual(command.topP, 0.95)
        XCTAssertEqual(command.topK, 20)
        XCTAssertEqual(command.minP, 0.05)
    }

    func testLagunaDFlashBenchmarkParsesResidentConcurrencyMatrix() throws {
        let command = try ModelBenchmarkLagunaDFlash.parse([
            "--laguna-path", "/tmp/Laguna-S-2.1-NVFP4-mlx",
            "--laguna-dflash-path", "/tmp/Laguna-S-2.1-DFlash",
            "--concurrency-values", "1,2,4",
            "--warmup-repetitions", "2",
            "--mixed-fixtures",
            "--log-responses",
        ])

        XCTAssertEqual(command.concurrencyValues, "1,2,4")
        XCTAssertEqual(command.warmupRepetitions, 2)
        XCTAssertTrue(command.mixedFixtures)
        XCTAssertTrue(command.logResponses)
    }

    func testLagunaDFlashBenchmarkRejectsInvalidConcurrencyMatrix() {
        let prefix = [
            "--laguna-path", "/tmp/Laguna-S-2.1-NVFP4-mlx",
            "--laguna-dflash-path", "/tmp/Laguna-S-2.1-DFlash",
        ]
        XCTAssertThrowsError(try ModelBenchmarkLagunaDFlash.parse(
            prefix + ["--concurrency-values", "1,0,4"]
        ))
        XCTAssertThrowsError(try ModelBenchmarkLagunaDFlash.parse(
            prefix + ["--concurrency-values", "1,2,2"]
        ))
        XCTAssertThrowsError(try ModelBenchmarkLagunaDFlash.parse(
            prefix + ["--mixed-fixtures"]
        ))
        XCTAssertThrowsError(try ModelBenchmarkLagunaDFlash.parse(
            prefix + ["--warmup-repetitions", "-1"]
        ))
    }

    func testCodeBenchmarkRequiresExecutionOptIn() {
        XCTAssertThrowsError(try ModelBenchmarkCode.parse([]))
    }

    func testCodeBenchmarkParsesDryRunDefaults() throws {
        let cmd = try ModelBenchmarkCode.parse(["--dry-run"])

        XCTAssertNil(cmd.models)
        XCTAssertNil(cmd.lagunaPath)
        XCTAssertNil(cmd.lagunaDflashPath)
        XCTAssertEqual(cmd.lagunaDflashTokens, 12)
        XCTAssertEqual(cmd.lagunaDflashMinTokens, 32)
        XCTAssertEqual(cmd.suite, .humanEvalSlice)
        XCTAssertNil(cmd.tasks)
        XCTAssertNil(cmd.humanevalFile)
        XCTAssertEqual(cmd.maxTokens, 1024)
        XCTAssertEqual(cmd.temperature, 0)
        XCTAssertEqual(cmd.topP, 1)
        XCTAssertEqual(cmd.topK, 0)
        XCTAssertNil(cmd.minP)
        XCTAssertEqual(cmd.resolvedMinP, 0)
        XCTAssertEqual(cmd.executionTimeout, 5)
        XCTAssertEqual(cmd.python, "python3")
        XCTAssertEqual(cmd.sandbox, .auto)
        XCTAssertTrue(cmd.dryRun)
        XCTAssertFalse(cmd.allowCodeExecution)
        XCTAssertFalse(cmd.json)
    }

    func testCodeBenchmarkDefaultModelsAdaptToMachineMemory() {
        let thirtyTwoGB = MereRunMachineProfile(
            physicalMemoryBytes: 32 * 1_073_741_824,
            processorName: "M4",
            isAppleSiliconMac: true
        )
        let sixtyFourGB = MereRunMachineProfile(
            physicalMemoryBytes: 64 * 1_073_741_824,
            processorName: "M4 Max",
            isAppleSiliconMac: true
        )

        XCTAssertEqual(ModelBenchmarkCode.defaultModelIDs(on: thirtyTwoGB), [
            Q35Resources.ornith9BModelId,
            Q35Resources.ornith35BMLX4BitModelId,
            NorthMiniCodeResources.modelId,
        ])
        XCTAssertEqual(ModelBenchmarkCode.defaultModelIDs(on: sixtyFourGB), [
            Q35Resources.ornith9BModelId,
            Q35Resources.ornith35BMLX4BitModelId,
            Q35Resources.ornith35BMLX6BitModelId,
            Q35Resources.ornith35BMLX8BitModelId,
            NorthMiniCodeResources.modelId,
            CodeGenResources.defaultModelId,
        ])
    }

    func testCodeBenchmarkAcceptsQ38CodeGenerationLane() throws {
        for modelID in [
            Q35Resources.q38TwentySevenBModelId,
            Q35Resources.q38TwentySevenB4BitModelId,
        ] {
            let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: modelID))
            XCTAssertEqual(spec.category, .visionChat)
            XCTAssertTrue(ModelBenchmarkCode.supportsCodingBenchmark(spec))
        }
    }

    func testCodeBenchmarkAcceptsOrnithVisionAgentLane() throws {
        for modelID in [
            Q35Resources.ornith35BMLX4BitModelId,
            Q35Resources.ornith35BVisionModelId,
        ] {
            let spec = try XCTUnwrap(ManagedModelCatalog.spec(for: modelID))
            XCTAssertEqual(spec.category, .visionChat)
            XCTAssertTrue(ModelBenchmarkCode.supportsCodingBenchmark(spec))
        }
    }

    func testCodeBenchmarkScoresOnlyVisibleCodeAfterImplicitThinkingPrefix() {
        let response = ChatResponse(
            response: """
            from typing? The prompt already imports it. Return only code.
            </think>

            def has_close_elements(numbers, threshold):
                return False
            """,
            tokensGenerated: 32
        )

        let scored = ModelBenchmarkCode.scoredCodeResponse(response)

        XCTAssertFalse(scored.contains("from typing?"))
        XCTAssertTrue(scored.hasPrefix("def has_close_elements"))
    }

    func testCodeBenchmarkParsesOverrides() throws {
        let cmd = try ModelBenchmarkCode.parse([
            "--models", "text-agent-ornith-35b-mlx,text-code-north-mini",
            "--suite", "humaneval-slice",
            "--tasks", "HumanEval/0,HumanEval/8",
            "--max-tokens", "256",
            "--temperature", "0.2",
            "--top-p", "0.8",
            "--top-k", "20",
            "--min-p", "0.05",
            "--execution-timeout", "3.5",
            "--python", "/tmp/venv/bin/python",
            "--sandbox", "none",
            "--dry-run",
            "--allow-code-execution",
            "--json",
        ])

        XCTAssertEqual(cmd.models, "text-agent-ornith-35b-mlx,text-code-north-mini")
        XCTAssertEqual(cmd.suite, .humanEvalSlice)
        XCTAssertEqual(cmd.tasks, "HumanEval/0,HumanEval/8")
        XCTAssertEqual(cmd.maxTokens, 256)
        XCTAssertEqual(cmd.temperature, 0.2)
        XCTAssertEqual(cmd.topP, 0.8)
        XCTAssertEqual(cmd.topK, 20)
        XCTAssertEqual(cmd.minP, 0.05)
        XCTAssertEqual(cmd.executionTimeout, 3.5)
        XCTAssertEqual(cmd.python, "/tmp/venv/bin/python")
        XCTAssertEqual(cmd.sandbox, .none)
        XCTAssertTrue(cmd.dryRun)
        XCTAssertTrue(cmd.allowCodeExecution)
        XCTAssertTrue(cmd.json)
    }

    func testCodeBenchmarkParsesHumanEvalFile() throws {
        let url = try makeHumanEvalJSONLFixture()
        let cmd = try ModelBenchmarkCode.parse([
            "--tasks", "HumanEval/42",
            "--humaneval-file", url.path,
            "--dry-run",
        ])

        XCTAssertEqual(cmd.humanevalFile, url.path)
        XCTAssertEqual(cmd.tasks, "HumanEval/42")
    }

    func testCodeBenchmarkLagunaPathSelectsEvaluationModel() throws {
        let cmd = try ModelBenchmarkCode.parse([
            "--laguna-path", "/tmp/Laguna-S-2.1-NVFP4-mlx",
            "--laguna-dflash-path", "/tmp/Laguna-S-2.1-DFlash",
            "--laguna-dflash-tokens", "5",
            "--laguna-dflash-min-tokens", "24",
            "--dry-run",
        ])

        XCTAssertEqual(cmd.lagunaPath, "/tmp/Laguna-S-2.1-NVFP4-mlx")
        XCTAssertEqual(cmd.lagunaDflashPath, "/tmp/Laguna-S-2.1-DFlash")
        XCTAssertEqual(cmd.lagunaDflashTokens, 5)
        XCTAssertEqual(cmd.lagunaDflashMinTokens, 24)
    }

    func testHumanEvalJSONLLoaderDecodesOfficialShape() throws {
        let url = try makeHumanEvalJSONLFixture()

        let tasks = try HumanEvalJSONLLoader.load(from: url)

        XCTAssertEqual(tasks.count, 1)
        XCTAssertEqual(tasks[0].taskID, "HumanEval/42")
        XCTAssertEqual(tasks[0].entryPoint, "answer")
        XCTAssertTrue(tasks[0].prompt.contains("def answer()"))
        XCTAssertTrue(tasks[0].tests.contains("assert candidate() == 42"))
    }

    private func makeHumanEvalJSONLFixture() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("mere-run-humaneval-loader-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        addTeardownBlock {
            try? FileManager.default.removeItem(at: directory)
        }
        let url = directory.appendingPathComponent("HumanEval.jsonl")
        let line = """
        {"task_id":"HumanEval/42","prompt":"def answer():\\n    pass\\n","entry_point":"answer","canonical_solution":"    return 42\\n","test":"def check(candidate):\\n    assert candidate() == 42\\n"}
        """
        try Data((line + "\n").utf8).write(to: url)
        return url
    }

    func testCodeExecutionSandboxNoneRunsPython() throws {
        let result = try CodeExecutionSandbox.runPython(
            program: "print('ok')",
            python: "python3",
            mode: .none,
            timeout: 2
        )

        XCTAssertTrue(result.passed)
        XCTAssertEqual(result.backend, .none)
        XCTAssertEqual(result.stdout.trimmingCharacters(in: .whitespacesAndNewlines), "ok")
    }

    #if os(macOS)
    func testCodeExecutionSandboxExecDeniesHomeReads() throws {
        try CodeExecutionSandbox.preflight(mode: .macOSSandboxExec)
        let home = FileManager.default.homeDirectoryForCurrentUser
            .resolvingSymlinksInPath()
            .path
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "'", with: "\\'")
        let result = try CodeExecutionSandbox.runPython(
            program: """
            import os
            try:
                os.listdir('\(home)')
                raise SystemExit(7)
            except PermissionError:
                print('denied')
            """,
            python: "python3",
            mode: .macOSSandboxExec,
            timeout: 2
        )

        XCTAssertTrue(result.passed, result.errorSummary ?? "")
        XCTAssertEqual(result.backend, .macOSSandboxExec)
        XCTAssertEqual(result.stdout.trimmingCharacters(in: .whitespacesAndNewlines), "denied")
    }
    #endif

    func testBenchmarkCommandExposesGemma4MTPSubcommand() {
        let commandNames = Set(ModelBenchmark.configuration.subcommands.map { $0.configuration.commandName })
        XCTAssertTrue(commandNames.contains("gemma4-mtp"))
    }

    func testBenchmarkCommandExposesQ36MTPSubcommand() {
        let commandNames = Set(ModelBenchmark.configuration.subcommands.map { $0.configuration.commandName })
        XCTAssertTrue(commandNames.contains("q36-mtp"))
    }

    func testBenchmarkCommandExposesAPIWorkloadSubcommand() {
        let commandNames = Set(ModelBenchmark.configuration.subcommands.map { $0.configuration.commandName })
        XCTAssertTrue(commandNames.contains("api-workload"))
    }

    func testAPIWorkloadBenchmarkParsesDefaults() throws {
        let cmd = try ModelBenchmarkAPIWorkload.parse(["--dry-run"])

        XCTAssertEqual(cmd.host, "127.0.0.1")
        XCTAssertEqual(cmd.port, 8080)
        XCTAssertNil(cmd.apiKey)
        XCTAssertEqual(cmd.model, Gemma4Resources.turboModelId)
        XCTAssertNil(cmd.workloadFile)
        XCTAssertEqual(cmd.turns, 8)
        XCTAssertEqual(cmd.sharedPrefixRepeat, 32)
        XCTAssertEqual(cmd.maxTokens, 64)
        XCTAssertEqual(cmd.temperature, 0)
        XCTAssertEqual(cmd.topP, 1)
        XCTAssertEqual(cmd.concurrency, 1)
        XCTAssertEqual(cmd.timeoutSeconds, 300)
        XCTAssertTrue(cmd.dryRun)
        XCTAssertFalse(cmd.json)
    }

    func testAPIWorkloadBenchmarkParsesOverrides() throws {
        let cmd = try ModelBenchmarkAPIWorkload.parse([
            "--host", "localhost",
            "--port", "11434",
            "--api-key", "secret",
            "--model", Q35Resources.q36NanoModelId,
            "--workload-file", "/tmp/workload.jsonl",
            "--turns", "3",
            "--shared-prefix-repeat", "4",
            "--max-tokens", "16",
            "--temperature", "0.2",
            "--top-p", "0.8",
            "--concurrency", "2",
            "--timeout-seconds", "30",
            "--dry-run",
            "--json",
        ])

        XCTAssertEqual(cmd.host, "localhost")
        XCTAssertEqual(cmd.port, 11434)
        XCTAssertEqual(cmd.apiKey, "secret")
        XCTAssertEqual(cmd.model, Q35Resources.q36NanoModelId)
        XCTAssertEqual(cmd.workloadFile, "/tmp/workload.jsonl")
        XCTAssertEqual(cmd.turns, 3)
        XCTAssertEqual(cmd.sharedPrefixRepeat, 4)
        XCTAssertEqual(cmd.maxTokens, 16)
        XCTAssertEqual(cmd.temperature, 0.2)
        XCTAssertEqual(cmd.topP, 0.8)
        XCTAssertEqual(cmd.concurrency, 2)
        XCTAssertEqual(cmd.timeoutSeconds, 30)
        XCTAssertTrue(cmd.dryRun)
        XCTAssertTrue(cmd.json)
    }

    func testAPIWorkloadBuiltInFixtureUsesStablePrefix() {
        let cases = ModelBenchmarkAPIWorkload.builtInStablePrefixWorkload(turns: 2, sharedPrefixRepeat: 3)

        XCTAssertEqual(cases.count, 2)
        XCTAssertEqual(cases[0].messages.count, 2)
        XCTAssertEqual(cases[1].messages.count, 2)
        XCTAssertEqual(cases[0].messages[0].role, "system")
        XCTAssertEqual(cases[1].messages[0].role, "system")
        XCTAssertEqual(cases[0].messages[0].content, cases[1].messages[0].content)
        XCTAssertNotEqual(cases[0].messages[1].content, cases[1].messages[1].content)
        XCTAssertTrue(cases[0].messages[0].content.contains("Cache note 3"))
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

    func testBenchmarkCommandExposesQ38VerificationSubcommand() {
        let commandNames = Set(ModelBenchmark.configuration.subcommands.map { $0.configuration.commandName })
        XCTAssertTrue(commandNames.contains("q38-verification"))
    }

    func testQ38VerificationBenchmarkParsesDefaults() throws {
        let cmd = try ModelBenchmarkQ38Verification.parse(["--model-root", "/tmp/flash-next"])

        XCTAssertEqual(cmd.model, Q35Resources.q38FlashNext3BitNativePLEModelId)
        XCTAssertEqual(cmd.modelRoot, "/tmp/flash-next")
        XCTAssertEqual(cmd.widths, "1,4,8,16,32")
        XCTAssertEqual(cmd.tokens, 128)
        XCTAssertEqual(cmd.trials, 2)
        XCTAssertFalse(cmd.json)
        XCTAssertNoThrow(try cmd.validate())
    }

    func testQ38VerificationBenchmarkRejectsInvalidWidth() {
        XCTAssertThrowsError(try ModelBenchmarkQ38Verification.parse([
            "--model-root", "/tmp/flash-next",
            "--widths", "4,64",
        ]))
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

    func testQ36MTPBenchmarkAcceptsQ38AndOfficialOrnithQuantizedTargets() throws {
        for modelId in [
            Q35Resources.q38TwentySevenBModelId,
            Q35Resources.q38TwentySevenB4BitModelId,
            Q35Resources.ornith35BMLX4BitModelId,
            Q35Resources.ornith35BMLX6BitModelId,
            Q35Resources.ornith35BMLX8BitModelId,
            Q35Resources.ornith35BVisionModelId,
        ] {
            let cmd = try ModelBenchmarkQ36MTP.parse(["--model", modelId])
            XCTAssertNoThrow(try cmd.validate())
        }
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

    func testChatBenchmarkParsesDefaults() throws {
        let cmd = try ModelBenchmarkChat.parse([])

        XCTAssertNil(cmd.models)
        XCTAssertNil(cmd.lagunaPath)
        XCTAssertNil(cmd.lagunaDflashPath)
        XCTAssertEqual(cmd.lagunaDflashTokens, 12)
        XCTAssertEqual(cmd.lagunaDflashMinTokens, 32)
        XCTAssertEqual(cmd.lagunaDflashRouting, .automatic)
        XCTAssertEqual(cmd.suite, .mereChatSlice)
        XCTAssertNil(cmd.cases)
        XCTAssertEqual(cmd.maxTokens, 96)
        XCTAssertEqual(cmd.temperature, 0)
        XCTAssertEqual(cmd.topP, 1)
        XCTAssertEqual(cmd.topK, 0)
        XCTAssertNil(cmd.minP)
        XCTAssertEqual(cmd.resolvedMinP, 0)
        XCTAssertNil(cmd.contextSize)
        XCTAssertEqual(cmd.concurrency, 1)
        XCTAssertFalse(cmd.dryRun)
        XCTAssertFalse(cmd.logResponses)
        XCTAssertFalse(cmd.json)
    }

    func testChatBenchmarkParsesOverrides() throws {
        let cmd = try ModelBenchmarkChat.parse([
            "--models", "text-chat-lfm25-a1b-8bit,text-chat-gemma4-nano",
            "--suite", "mere-chat-slice",
            "--cases", "MereChat/0,MereChat/3",
            "--max-tokens", "128",
            "--temperature", "0.1",
            "--top-p", "0.8",
            "--top-k", "20",
            "--min-p", "0.05",
            "--context-size", "4096",
            "--concurrency", "2",
            "--laguna-dflash-routing", "target-only",
            "--dry-run",
            "--log-responses",
            "--json",
        ])

        XCTAssertEqual(cmd.models, "text-chat-lfm25-a1b-8bit,text-chat-gemma4-nano")
        XCTAssertEqual(cmd.suite, .mereChatSlice)
        XCTAssertEqual(cmd.cases, "MereChat/0,MereChat/3")
        XCTAssertEqual(cmd.maxTokens, 128)
        XCTAssertEqual(cmd.temperature, 0.1)
        XCTAssertEqual(cmd.topP, 0.8)
        XCTAssertEqual(cmd.topK, 20)
        XCTAssertEqual(cmd.minP, 0.05)
        XCTAssertEqual(cmd.contextSize, 4096)
        XCTAssertEqual(cmd.concurrency, 2)
        XCTAssertEqual(cmd.lagunaDflashRouting, .targetOnly)
        XCTAssertTrue(cmd.dryRun)
        XCTAssertTrue(cmd.logResponses)
        XCTAssertTrue(cmd.json)
    }

    func testChatBenchmarkSuiteHasFiftyTwoCases() {
        XCTAssertEqual(ChatBenchmarkCase.mereChatSlice.count, 52)
    }

    func testChatBenchmarkSelectsCaseSubset() throws {
        let cmd = try ModelBenchmarkChat.parse(["--cases", "MereChat/0,MereChat/39"])
        let selected = try cmd.selectedCases()

        XCTAssertEqual(selected.map(\.caseID), ["MereChat/0", "MereChat/39"])
    }

    func testChatBenchmarkScoresGroundedEmailAnswer() {
        let benchmarkCase = ChatBenchmarkCase.mereChatSlice[0]
        let evaluation = benchmarkCase.evaluate(
            "mail_103 from abenewsoil@gmail.com on 2026-06-27, subject Revised nursery quote."
        )

        XCTAssertTrue(evaluation.passed, evaluation.failedChecks.joined(separator: "\n"))
        XCTAssertEqual(evaluation.passedChecks, evaluation.totalChecks)
    }

    func testChatBenchmarkRejectsHallucinatedMissingSenderAnswer() {
        let benchmarkCase = ChatBenchmarkCase.mereChatSlice[1]
        let evaluation = benchmarkCase.evaluate(
            "The most recent email from abenewsoil@gmail.com is mail_103, Revised nursery quote, on 2026-06-27."
        )

        XCTAssertFalse(evaluation.passed)
        XCTAssertTrue(evaluation.failedChecks.contains { $0.contains("NOT_IN_EVIDENCE") })
    }

    func testChatBenchmarkScoresJSONExtraction() {
        let benchmarkCase = ChatBenchmarkCase.mereChatSlice[3]
        let evaluation = benchmarkCase.evaluate(
            """
            {"id":"mail_301","sender":"orders@nova.example","subject":"PO 1178 accepted","date":"2026-06-22T08:03"}
            """
        )

        XCTAssertTrue(evaluation.passed, evaluation.failedChecks.joined(separator: "\n"))
    }

    func testChatBenchmarkAllowsExplicitBrowserNotOpenedEvidence() {
        let benchmarkCase = ChatBenchmarkCase.mereChatSlice[33]
        let evaluation = benchmarkCase.evaluate(
            "No, the browser did not open (browser_opened=false). The device code is MERE-42AB."
        )

        XCTAssertTrue(evaluation.passed, evaluation.failedChecks.joined(separator: "\n"))
    }

    func testChatFlavorScenariosAcceptGroundedReferenceResponses() throws {
        let responses = [
            "MereChat/40": "The reply was faster because the model was already loaded; generation still ran for this answer.",
            "MereChat/41": "Could you please send the invoice by Friday? Thank you.",
            "MereChat/42": "I'm sorry—that's frustrating. First check autosave, then check the local backup.",
            "MereChat/43": "It did not fail; deployment dep_812 succeeded at 2026-08-17T14:22:00Z.",
            "MereChat/44": "Both estimates are preliminary: one range is 6 to 8 hours and the other is 10 to 12 hours, so timing remains uncertain.",
            "MereChat/45": "Choose local because private offline operation is the priority; the main tradeoff is slower setup.",
            "MereChat/46": "1. Run tests\n2. Build artifacts\n3. Verify signatures\n4. Publish",
            "MereChat/47": "The supported inventory total is 317.",
            "MereChat/48": "Build passed; signature not checked; release not published.",
            "MereChat/49": "No. job_b is the counterexample: 47 seconds exceeds the 30-second target.",
            "MereChat/50": "- Local AI.\n- Private by default.\n- Fast on device.",
            "MereChat/51": "Which object do you mean: project proj_harbor or email mail_901? Please clarify.",
        ]

        for (caseID, response) in responses {
            let benchmarkCase = try XCTUnwrap(
                ChatBenchmarkCase.mereChatSlice.first { $0.caseID == caseID }
            )
            let evaluation = benchmarkCase.evaluate(response)
            XCTAssertTrue(
                evaluation.passed,
                "\(caseID): \(evaluation.failedChecks.joined(separator: ", "))"
            )
        }
    }

    func testToolCallsBenchmarkParsesDefaults() throws {
        let cmd = try ModelBenchmarkToolCalls.parse([])

        XCTAssertNil(cmd.models)
        XCTAssertNil(cmd.lagunaPath)
        XCTAssertNil(cmd.lagunaDflashPath)
        XCTAssertEqual(cmd.lagunaDflashTokens, 12)
        XCTAssertEqual(cmd.lagunaDflashMinTokens, 32)
        XCTAssertNil(cmd.cases)
        XCTAssertEqual(cmd.maxTokens, 192)
        XCTAssertEqual(cmd.temperature, 0)
        XCTAssertEqual(cmd.topP, 1)
        XCTAssertEqual(cmd.topK, 0)
        XCTAssertNil(cmd.minP)
        XCTAssertEqual(cmd.resolvedMinP, 0)
        XCTAssertNil(cmd.contextSize)
        XCTAssertFalse(cmd.dryRun)
        XCTAssertFalse(cmd.logResponses)
        XCTAssertFalse(cmd.json)
    }

    func testChatAndToolCallBenchmarksSelectLagunaEvaluationPath() throws {
        let path = "/Volumes/model-store/Laguna-S-2.1-NVFP4-mlx"
        let draftPath = "/Volumes/model-store/Laguna-S-2.1-DFlash"
        let chat = try ModelBenchmarkChat.parse([
            "--laguna-path", path,
            "--laguna-dflash-path", draftPath,
            "--laguna-dflash-tokens", "5",
            "--laguna-dflash-min-tokens", "24",
            "--laguna-dflash-routing", "dflash",
            "--dry-run",
        ])
        let tools = try ModelBenchmarkToolCalls.parse([
            "--laguna-path", path,
            "--laguna-dflash-path", draftPath,
            "--laguna-dflash-tokens", "5",
            "--laguna-dflash-min-tokens", "24",
            "--dry-run",
        ])

        XCTAssertEqual(chat.lagunaPath, path)
        XCTAssertEqual(chat.lagunaDflashPath, draftPath)
        XCTAssertEqual(chat.lagunaDflashTokens, 5)
        XCTAssertEqual(chat.lagunaDflashMinTokens, 24)
        XCTAssertEqual(chat.lagunaDflashRouting, .dflash)
        XCTAssertEqual(chat.resolvedMinP, LagunaResources.recommendedMinP)
        XCTAssertEqual(try chat.selectedModelIDs(), [LagunaResources.modelID])
        XCTAssertEqual(tools.lagunaPath, path)
        XCTAssertEqual(tools.lagunaDflashPath, draftPath)
        XCTAssertEqual(tools.lagunaDflashTokens, 5)
        XCTAssertEqual(tools.lagunaDflashMinTokens, 24)
        XCTAssertEqual(tools.resolvedMinP, LagunaResources.recommendedMinP)
        XCTAssertEqual(try tools.selectedModelIDs(), [LagunaResources.modelID])
    }

    func testLagunaDFlashBenchmarkFlagsUseInstalledDefaultsAndRejectInvalidCombinations() throws {
        let installed = try ModelBenchmarkChat.parse([
            "--laguna-dflash-path", "/tmp/Laguna-S-2.1-DFlash",
        ])
        XCTAssertEqual(try installed.selectedModelIDs(), [LagunaResources.modelID])
        XCTAssertEqual(installed.lagunaDflashPath, "/tmp/Laguna-S-2.1-DFlash")

        XCTAssertThrowsError(try ModelBenchmarkChat.parse([
            "--laguna-path", "/tmp/Laguna-S-2.1-NVFP4-mlx",
            "--laguna-dflash-tokens", "16",
        ]))
        XCTAssertThrowsError(try ModelBenchmarkChat.parse([
            "--laguna-path", "/tmp/Laguna-S-2.1-NVFP4-mlx",
            "--concurrency", "0",
        ]))
        XCTAssertThrowsError(try ModelBenchmarkChat.parse([
            "--laguna-path", "/tmp/Laguna-S-2.1-NVFP4-mlx",
            "--laguna-dflash-min-tokens", "0",
        ]))
        XCTAssertThrowsError(try ModelBenchmarkChat.parse([
            "--laguna-path", "/tmp/Laguna-S-2.1-NVFP4-mlx",
            "--laguna-dflash-routing", "dflash",
        ]))
    }

    func testToolCallsBenchmarkParsesOverrides() throws {
        let cmd = try ModelBenchmarkToolCalls.parse([
            "--models", "text-chat-q36-nano,text-chat-gemma4-12b-4bit",
            "--cases", "MereTool/0,MereTool/4",
            "--max-tokens", "96",
            "--temperature", "0.1",
            "--top-p", "0.8",
            "--top-k", "20",
            "--min-p", "0.05",
            "--context-size", "4096",
            "--dry-run",
            "--log-responses",
            "--json",
        ])

        XCTAssertEqual(cmd.models, "text-chat-q36-nano,text-chat-gemma4-12b-4bit")
        XCTAssertEqual(cmd.cases, "MereTool/0,MereTool/4")
        XCTAssertEqual(cmd.maxTokens, 96)
        XCTAssertEqual(cmd.temperature, 0.1)
        XCTAssertEqual(cmd.topP, 0.8)
        XCTAssertEqual(cmd.topK, 20)
        XCTAssertEqual(cmd.minP, 0.05)
        XCTAssertEqual(cmd.contextSize, 4096)
        XCTAssertTrue(cmd.dryRun)
        XCTAssertTrue(cmd.logResponses)
        XCTAssertTrue(cmd.json)
    }

    func testToolCallsBenchmarkSuiteHasTenCases() {
        XCTAssertEqual(ToolBenchmarkCase.mereToolSlice.count, 10)
    }

    func testToolCallsBenchmarkSelectsCaseSubset() throws {
        let cmd = try ModelBenchmarkToolCalls.parse(["--cases", "MereTool/0,MereTool/9"])
        let selected = try cmd.selectedCases()

        XCTAssertEqual(selected.map(\.caseID), ["MereTool/0", "MereTool/9"])
    }

    func testToolCallsBenchmarkScoresExpectedToolCall() {
        let benchmarkCase = ToolBenchmarkCase.mereToolSlice[0]
        let evaluation = benchmarkCase.expectation.evaluate([
            ToolBenchmarkObservedCall(
                call: ToolCall(
                    name: "mere_email_search",
                    arguments: [
                        "workspace": "sawfwair",
                        "sender": "abenewsoil@gmail.com",
                        "after": "2026-06-01",
                    ]
                )
            ),
        ])

        XCTAssertTrue(evaluation.passed, evaluation.failedChecks.joined(separator: "\n"))
    }

    func testToolCallsBenchmarkRejectsWrongToolCall() {
        let benchmarkCase = ToolBenchmarkCase.mereToolSlice[0]
        let evaluation = benchmarkCase.expectation.evaluate([
            ToolBenchmarkObservedCall(
                call: ToolCall(
                    name: "mere_project_search",
                    arguments: ["workspace": "sawfwair"]
                )
            ),
        ])

        XCTAssertFalse(evaluation.passed)
        XCTAssertTrue(evaluation.failedChecks.contains { $0.contains("expected tool mere_email_search") })
    }

    func testToolCallsBenchmarkScoresNoToolCase() {
        let benchmarkCase = ToolBenchmarkCase.mereToolSlice[4]
        let evaluation = benchmarkCase.expectation.evaluate([])

        XCTAssertTrue(evaluation.passed, evaluation.failedChecks.joined(separator: "\n"))
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
