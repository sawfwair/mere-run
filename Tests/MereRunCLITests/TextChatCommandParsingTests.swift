import XCTest
@testable import MereRunCLI
@testable import MereRunCore

final class TextChatCommandParsingTests: XCTestCase {
    func testTextChatDefaultsToNonStreamingCLIOutput() throws {
        let cmd = try TextChat.parse([
            "--prompt", "Say hello",
        ])

        XCTAssertFalse(cmd.stream)
        XCTAssertEqual(cmd.prompt, "Say hello")
        XCTAssertEqual(cmd.model, TextChat.defaultChatModelId)
        XCTAssertEqual(cmd.responseFormat, .text)
        XCTAssertEqual(cmd.markdown, .auto)
    }

    func testTextChatDefaultModelSelectionForMemoryBands() {
        let compactAppleSilicon = MereRunMachineProfile(
            physicalMemoryBytes: 24 * 1_073_741_824,
            processorName: "Apple Silicon",
            isAppleSiliconMac: true
        )
        let undersizedAppleSilicon = MereRunMachineProfile(
            physicalMemoryBytes: 8 * 1_073_741_824,
            processorName: "Apple Silicon",
            isAppleSiliconMac: true
        )

        XCTAssertEqual(
            TextChat.defaultChatModelId(on: compactAppleSilicon),
            Gemma4Resources.twelveB4BitModelId
        )
        XCTAssertEqual(
            TextChat.defaultChatModelId(on: undersizedAppleSilicon),
            Gemma4Resources.nanoModelId
        )
    }

    func testTextChatParsesStreamingFlag() throws {
        let cmd = try TextChat.parse([
            "--prompt", "Stream this",
            "--stream",
            "--markdown", "never",
        ])

        XCTAssertTrue(cmd.stream)
        XCTAssertEqual(cmd.markdown, .never)
    }

    func testTextChatParsesDiffusionGemmaModel() throws {
        let cmd = try TextChat.parse([
            "--model", DiffusionGemmaResources.modelID,
            "--max-tokens", "256",
            "--seed", "123",
            "--show-unmasking",
            "--prompt", "Explain block diffusion.",
        ])

        XCTAssertEqual(cmd.model, DiffusionGemmaResources.modelID)
        XCTAssertEqual(cmd.maxTokens, DiffusionGemmaResources.maximumCanvasLength)
        XCTAssertEqual(cmd.seed, 123)
        XCTAssertTrue(cmd.showUnmasking)
        XCTAssertNoThrow(try TextChat.validateDiffusionOptions(
            seed: cmd.seed,
            showUnmasking: cmd.showUnmasking,
            modelID: cmd.model
        ))
    }

    func testTextChatRejectsDiffusionOptionsForOtherModels() {
        XCTAssertThrowsError(try TextChat.validateDiffusionOptions(
            seed: 123,
            showUnmasking: false,
            modelID: Gemma4Resources.nanoModelId
        ))
    }

    func testDiffusionDraftProgressUsesStderrWithoutMarkingFinalOutputStreamed() throws {
        let stdout = TextOutputRecorder()
        let stderr = TextOutputRecorder()
        let streamingOutput = StreamingChatOutput(enabled: true, writer: stdout.write)
        let progressHandler = try XCTUnwrap(TextChatProgressHandler.make(
            quiet: false,
            streamingOutput: streamingOutput,
            diagnosticWriter: stderr.write
        ))

        progressHandler(ChatProgress(
            stage: .generating,
            diffusion: ChatDiffusionProgress(
                draftText: "The [Mask] answer",
                step: 2,
                totalSteps: 48,
                canvasIndex: 1,
                blockComplete: true
            )
        ))

        XCTAssertEqual(stdout.value, "")
        XCTAssertFalse(streamingOutput.hasWritten)
        XCTAssertTrue(stderr.value.contains("canvas=1 step=2/48"))
        XCTAssertTrue(stderr.value.contains("The [Mask] answer"))
    }

    func testQuietStreamingKeepsGeneratingCallbackAndSuppressesDiagnostics() throws {
        let stdout = TextOutputRecorder()
        let stderr = TextOutputRecorder()
        let streamingOutput = StreamingChatOutput(enabled: true, writer: stdout.write)
        let progressHandler = try XCTUnwrap(
            TextChatProgressHandler.make(
                quiet: true,
                streamingOutput: streamingOutput,
                diagnosticWriter: stderr.write
            )
        )

        progressHandler(ChatProgress(stage: .loadingModel, message: "Loading model"))
        progressHandler(ChatProgress(stage: .generating, message: "live token"))

        XCTAssertEqual(stdout.value, "live token")
        XCTAssertEqual(stderr.value, "")
        XCTAssertTrue(streamingOutput.hasWritten)
    }

    func testQuietNonStreamingDisablesProgressCallback() {
        let streamingOutput = StreamingChatOutput(enabled: false)

        XCTAssertNil(
            TextChatProgressHandler.make(
                quiet: true,
                streamingOutput: streamingOutput
            )
        )
    }

    func testTextChatParsesJSONObjectResponseFormatForQ36() throws {
        let cmd = try TextChat.parse([
            "--model", Q35Resources.q36NanoModelId,
            "--response-format", "json_object",
            "--prompt", "Return an object",
        ])

        XCTAssertEqual(cmd.responseFormat, .jsonObject)
        XCTAssertNoThrow(
            try TextChat.validate(
                responseFormat: cmd.responseFormat,
                modelID: cmd.model
            )
        )
    }

    func testTextChatRejectsJSONObjectResponseFormatForGGUF() throws {
        let cmd = try TextChat.parse([
            "--model", ModelResolver.ModelID.q36NanoGGUF.rawValue,
            "--response-format", "json_object",
            "--prompt", "Return an object",
        ])

        XCTAssertThrowsError(
            try TextChat.validate(
                responseFormat: cmd.responseFormat,
                modelID: cmd.model
            )
        ) { error in
            let message = String(describing: error)
            XCTAssertTrue(message.contains("llama.cpp/GGUF"))
            XCTAssertTrue(message.contains(Q35Resources.q36NanoModelId))
        }
    }

    func testTextChatParsesLoRAAdapterOptions() throws {
        let cmd = try TextChat.parse([
            "--prompt", "Use adapter",
            "--lora", "/tmp/adapter.safetensors",
            "--lora-scale", "0.75",
        ])

        XCTAssertEqual(cmd.loraPath, "/tmp/adapter.safetensors")
        XCTAssertEqual(cmd.loraScale, 0.75)
    }

    func testTextChatBackendDescriptionIdentifiesNativeMLXModels() {
        let backend = TextChat.backendDescription(for: Gemma4Resources.twelveB4BitModelId)

        XCTAssertTrue(backend.contains("native MLX"))
    }

    func testDenseLFM2ParsesAsNativeMLX() throws {
        let command = try TextChat.parse([
            "--model", LFM2Resources.denseModelId,
            "--prompt", "Explain local inference",
        ])

        XCTAssertEqual(command.model, LFM2Resources.denseModelId)
        XCTAssertTrue(TextChat.backendDescription(for: command.model).contains("native MLX"))
    }

    func testInklingParsesAsNativeMLXWithOperationalContext() throws {
        let command = try TextChat.parse([
            "--model", InklingResources.modelID,
            "--context-size", "32768",
            "--prompt", "Plan a migration",
        ])

        XCTAssertEqual(command.model, InklingResources.modelID)
        XCTAssertEqual(command.contextSize, InklingResources.defaultContextLength)
        XCTAssertTrue(TextChat.backendDescription(for: command.model).contains("native MLX"))
        XCTAssertThrowsError(
            try TextChat.validate(responseFormat: .jsonObject, modelID: command.model)
        )
    }

    func testInklingParsesAndValidatesReasoningEffort() throws {
        let command = try TextChat.parse([
            "--model", InklingResources.modelID,
            "--reasoning-effort", "0.2",
            "--prompt", "Answer directly",
        ])

        XCTAssertEqual(command.reasoningEffort, 0.2)
        XCTAssertNoThrow(
            try TextChat.validateReasoningEffort(command.reasoningEffort, modelID: command.model)
        )
        XCTAssertThrowsError(
            try TextChat.validateReasoningEffort(1, modelID: command.model)
        )
        XCTAssertThrowsError(
            try TextChat.validateReasoningEffort(
                command.reasoningEffort,
                modelID: Gemma4Resources.twelveB4BitModelId
            )
        )
    }

    func testMuseGlimmerParsesNativeMultimodalReasoningContract() throws {
        let command = try TextChat.parse([
            "--model", MuseGlimmerResources.modelId,
            "--image", "/tmp/muse-glimmer-input.png",
            "--reasoning-effort", "1",
            "--prompt", "Inspect this interface",
        ])

        XCTAssertEqual(command.model, MuseGlimmerResources.modelId)
        XCTAssertEqual(command.image, "/tmp/muse-glimmer-input.png")
        XCTAssertEqual(command.reasoningEffort, 1)
        XCTAssertTrue(TextChat.backendDescription(for: command.model).contains("native MLX"))
        XCTAssertNoThrow(
            try TextChat.validateReasoningEffort(command.reasoningEffort, modelID: command.model)
        )
        XCTAssertThrowsError(
            try TextChat.validate(responseFormat: .jsonObject, modelID: command.model)
        )
    }

    func testQ38AcceptsContinuousReasoningEffort() throws {
        XCTAssertNoThrow(
            try TextChat.validateReasoningEffort(
                0.5,
                modelID: Q35Resources.q38TwentySevenB4BitModelId
            )
        )
        XCTAssertThrowsError(
            try TextChat.validateReasoningEffort(
                1.1,
                modelID: Q35Resources.q38TwentySevenBModelId
            )
        )
    }

    func testMuseGlimmerDFlashStatsAreMachineScannable() {
        let formatted = TextChat.formatMuseDFlashStats(MuseGlimmerDFlashStats(
            enabled: true,
            active: true,
            assistantModelPath: "/tmp/muse-assistant",
            speculativeTokens: 3,
            rounds: 4,
            draftedTokens: 12,
            acceptedTokens: 7,
            rejectedTokens: 2,
            fullAcceptanceRounds: 1,
            targetVerificationForwards: 4
        ))

        XCTAssertTrue(formatted.contains("muse_dflash=active"))
        XCTAssertTrue(formatted.contains("proposals=3"))
        XCTAssertTrue(formatted.contains("acceptance=58.3%"))
        XCTAssertTrue(formatted.contains("verify=4"))
    }

    func testNemotronParsesAndFormatsAdaptiveDSparkStats() throws {
        let command = try TextChat.parse([
            "--model", NemotronHResources.modelID,
            "--stats",
            "--prompt", "Design an actor queue",
        ])

        XCTAssertEqual(command.model, NemotronHResources.modelID)
        XCTAssertTrue(command.stats)
        XCTAssertTrue(TextChat.backendDescription(for: command.model).contains("native MLX"))
        XCTAssertThrowsError(
            try TextChat.validate(responseFormat: .jsonObject, modelID: command.model)
        )

        let formatted = TextChat.formatNemotronDSparkStats(NemotronHDSparkStats(
            enabled: true,
            active: false,
            speculativeTokens: 3,
            rounds: 2,
            draftedTokens: 6,
            acceptedDraftTokens: 2,
            rejectedDraftTokens: 2,
            targetVerificationForwards: 2,
            targetRecoveryForwards: 2,
            targetFallbackForwards: 57,
            adaptiveFallbacks: 1,
            reason: "draft acceptance fell below 67%"
        ))
        XCTAssertTrue(formatted.contains("dspark=fallback"))
        XCTAssertTrue(formatted.contains("block=3"))
        XCTAssertTrue(formatted.contains("acceptance=33.3%"))
        XCTAssertTrue(formatted.contains("fallback_forwards=57"))
        XCTAssertTrue(formatted.contains("adaptive_fallbacks=1"))

        let lfm2Formatted = TextChat.formatLFM2DSparkStats(LFM2DSparkStats(
            enabled: true,
            active: true,
            speculativeTokens: 9,
            rounds: 2,
            draftedTokens: 18,
            acceptedDraftTokens: 15,
            rejectedDraftTokens: 1,
            targetVerificationForwards: 2,
            targetRecoveryForwards: 1
        ))
        XCTAssertTrue(lfm2Formatted.contains("lfm25_dspark=active"))
        XCTAssertTrue(lfm2Formatted.contains("block=9"))
        XCTAssertTrue(lfm2Formatted.contains("acceptance=83.3%"))
        XCTAssertTrue(lfm2Formatted.contains("verification=2"))
    }

    func testTextChatBackendDescriptionIdentifiesGGUFModels() {
        let backend = TextChat.backendDescription(for: ModelResolver.ModelID.q36NanoGGUF.rawValue)

        XCTAssertEqual(backend, "llama.cpp/GGUF")
    }

    func testTextChatTTFTIncludesAllWorkBeforeFirstToken() throws {
        let timing = ChatTiming(
            loadSeconds: 1.0,
            prefillSeconds: 2.0,
            cacheConversionSeconds: 0.25,
            decodeSeconds: 3.0,
            firstTokenSeconds: 0.5
        )

        XCTAssertEqual(try XCTUnwrap(TextChat.ttftSeconds(for: timing)), 3.75, accuracy: 0.000_001)
        XCTAssertNil(TextChat.ttftSeconds(for: ChatTiming(firstTokenSeconds: nil)))
    }

    func testGemma4TurboDefaultsToTurboQuantKVCache() throws {
        let cmd = try TextChat.parse([
            "--prompt", "Say hello",
            "--model", Gemma4Resources.turboModelId,
        ])

        let quantization = try cmd.resolveGemma4KVCacheQuantization(for: Gemma4Resources.turboModelId)

        XCTAssertEqual(quantization.bits, Gemma4Resources.defaultTurboKVBits)
        XCTAssertEqual(quantization.scheme, .turboquant)
        XCTAssertEqual(quantization.groupSize, Gemma4Resources.defaultKVGroupSize)
        XCTAssertEqual(quantization.quantizedStart, Gemma4Resources.defaultTurboQuantizedKVStart)
    }

    func testExplicitGemma4KVCacheOptionsOverrideTurboDefaults() throws {
        let cmd = try TextChat.parse([
            "--prompt", "Say hello",
            "--model", Gemma4Resources.turboModelId,
            "--kv-bits", "3.5",
            "--kv-quant-scheme", "turboquant",
            "--kv-group-size", "32",
            "--quantized-kv-start", "128",
        ])

        let quantization = try cmd.resolveGemma4KVCacheQuantization(for: Gemma4Resources.turboModelId)

        XCTAssertEqual(quantization.bits, 3.5)
        XCTAssertEqual(quantization.scheme, .turboquant)
        XCTAssertEqual(quantization.groupSize, 32)
        XCTAssertEqual(quantization.quantizedStart, 128)
    }

    func testGemma4TurboKVFlagsOverrideIndependently() throws {
        let cmd = try TextChat.parse([
            "--prompt", "Say hello",
            "--model", Gemma4Resources.turboModelId,
            "--kv-quant-scheme", "uniform",
            "--kv-group-size", "32",
            "--quantized-kv-start", "128",
        ])

        let quantization = try cmd.resolveGemma4KVCacheQuantization(for: Gemma4Resources.turboModelId)

        XCTAssertEqual(quantization.bits, Gemma4Resources.defaultTurboKVBits)
        XCTAssertEqual(quantization.scheme, .uniform)
        XCTAssertEqual(quantization.groupSize, 32)
        XCTAssertEqual(quantization.quantizedStart, 128)
    }

    func testGemma4PolarKVFlagsParse() throws {
        let cmd = try TextChat.parse([
            "--prompt", "Say hello",
            "--model", Gemma4Resources.turboModelId,
            "--kv-bits", "2",
            "--kv-quant-scheme", "polar",
        ])

        let quantization = try cmd.resolveGemma4KVCacheQuantization(for: Gemma4Resources.turboModelId)

        XCTAssertEqual(quantization.bits, 2)
        XCTAssertEqual(quantization.scheme, .polar)
    }

    func testCleanResponseRemovesCompletedThinkingBlock() throws {
        let cmd = try TextChat.parse([
            "--prompt", "Say hello",
        ])

        let cleaned = cmd.cleanResponse("<think>\nworking\n</think>\n4", showThinking: false)

        XCTAssertEqual(cleaned, "4")
    }

    func testCleanResponseRemovesDanglingThinkingBlock() throws {
        let cmd = try TextChat.parse([
            "--prompt", "Say hello",
        ])

        let cleaned = cmd.cleanResponse("<think>\nThe user asks for a short answer.", showThinking: false)

        XCTAssertEqual(cleaned, "")
    }

    func testTextChatThinkingFlagParsesAsTriState() throws {
        let unset = try TextChat.parse(["--prompt", "hi"])
        XCTAssertNil(unset.thinking)

        let enabled = try TextChat.parse(["--prompt", "hi", "--thinking"])
        XCTAssertEqual(enabled.thinking, true)

        let disabled = try TextChat.parse(["--prompt", "hi", "--no-thinking"])
        XCTAssertEqual(disabled.thinking, false)
    }

    func testTextChatSamplingOptionsDefaultToNilForModelResolution() throws {
        let cmd = try TextChat.parse(["--prompt", "hi"])
        XCTAssertNil(cmd.temperature)
        XCTAssertNil(cmd.topP)
        XCTAssertNil(cmd.topK)
        XCTAssertNil(cmd.minP)

        let explicit = try TextChat.parse([
            "--prompt", "hi",
            "--temperature", "0.2",
            "--top-p", "0.8",
            "--top-k", "40",
            "--min-p", "0.05",
        ])
        XCTAssertEqual(explicit.temperature, 0.2)
        XCTAssertEqual(explicit.topP, 0.8)
        XCTAssertEqual(explicit.topK, 40)
        XCTAssertEqual(explicit.minP, 0.05)
    }

    func testQwenAgentAndQ38LanesDefaultToThinkingAndRecommendedSampling() {
        XCTAssertTrue(Q35Resources.thinkingDefault(forModelId: Q35Resources.ornith35BMLXModelId))
        XCTAssertTrue(Q35Resources.thinkingDefault(forModelId: Q35Resources.ornith35BMLX4BitModelId))
        XCTAssertTrue(Q35Resources.thinkingDefault(forModelId: Q35Resources.ornith35BVisionModelId))
        XCTAssertTrue(Q35Resources.thinkingDefault(forModelId: Q35Resources.ornith9BModelId))
        XCTAssertTrue(Q35Resources.thinkingDefault(forModelId: Q35Resources.q38TwentySevenBModelId))
        XCTAssertTrue(Q35Resources.thinkingDefault(forModelId: Q35Resources.q38TwentySevenB4BitModelId))
        XCTAssertFalse(Q35Resources.thinkingDefault(forModelId: Q35Resources.q36NanoModelId))
        XCTAssertFalse(Q35Resources.thinkingDefault(forModelId: Gemma4Resources.twelveB4BitModelId))

        let sampling = Q35Resources.recommendedSampling(forModelId: Q35Resources.ornith35BMLXModelId)
        XCTAssertEqual(sampling?.temperature, 1.0)
        XCTAssertEqual(sampling?.topP, 0.95)
        XCTAssertEqual(sampling?.topK, 20)
        XCTAssertEqual(
            Q35Resources.recommendedSampling(forModelId: Q35Resources.ornith35BMLX4BitModelId),
            sampling
        )
        XCTAssertEqual(
            Q35Resources.recommendedSampling(forModelId: Q35Resources.ornith35BVisionModelId),
            sampling
        )
        let q38Sampling = Q35Resources.recommendedSampling(forModelId: Q35Resources.q38TwentySevenBModelId)
        XCTAssertEqual(q38Sampling?.temperature, 1.0)
        XCTAssertEqual(q38Sampling?.topP, 0.95)
        XCTAssertEqual(q38Sampling?.topK, 20)
        XCTAssertEqual(
            Q35Resources.recommendedSampling(forModelId: Q35Resources.q38TwentySevenB4BitModelId),
            q38Sampling
        )
        XCTAssertNil(Q35Resources.recommendedSampling(forModelId: Q35Resources.q36NanoModelId))
    }

    func testBonsaiParsesLongContextAndFourBitKVCache() throws {
        let cmd = try TextChat.parse([
            "--prompt", "Summarize this repository",
            "--model", Q35Resources.bonsai27B1BitModelId,
            "--context-size", "262144",
            "--kv-bits", "4",
        ])

        XCTAssertEqual(cmd.model, Q35Resources.bonsai27B1BitModelId)
        XCTAssertEqual(cmd.contextSize, Q35Resources.bonsai27B1BitContextLength)
        XCTAssertEqual(try cmd.resolveQ35KVCacheMode(for: cmd.model), .affine4)
        XCTAssertTrue(Q35Resources.thinkingDefault(forModelId: cmd.model))
        XCTAssertEqual(Q35Resources.recommendedSampling(forModelId: cmd.model)?.temperature, 0.7)
        XCTAssertEqual(Q35Resources.recommendedSampling(forModelId: cmd.model)?.topP, 0.95)
        XCTAssertEqual(Q35Resources.recommendedSampling(forModelId: cmd.model)?.topK, 20)
        XCTAssertEqual(Q35Resources.defaultContextLength(forModelId: cmd.model), 262_144)
    }

    func testQ35RejectsUnsupportedKVCacheWidth() throws {
        let cmd = try TextChat.parse([
            "--prompt", "hello",
            "--model", Q35Resources.bonsai27B1BitModelId,
            "--kv-bits", "3",
        ])

        XCTAssertThrowsError(try cmd.resolveQ35KVCacheMode(for: cmd.model)) { error in
            XCTAssertTrue(String(describing: error).contains("must be 4 or 8"))
        }
    }
}

private final class TextOutputRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var text = ""

    var value: String {
        lock.lock()
        defer { lock.unlock() }
        return text
    }

    func write(_ value: String) {
        lock.lock()
        defer { lock.unlock() }
        text += value
    }
}
