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
        ])

        XCTAssertTrue(cmd.stream)
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

    func testTextChatBackendDescriptionIdentifiesGGUFModels() {
        let backend = TextChat.backendDescription(for: ModelResolver.ModelID.q36NanoGGUF.rawValue)

        XCTAssertEqual(backend, "llama.cpp/GGUF")
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

        let explicit = try TextChat.parse([
            "--prompt", "hi",
            "--temperature", "0.2",
            "--top-p", "0.8",
            "--top-k", "40",
        ])
        XCTAssertEqual(explicit.temperature, 0.2)
        XCTAssertEqual(explicit.topP, 0.8)
        XCTAssertEqual(explicit.topK, 40)
    }

    func testOrnithLanesDefaultToThinkingAndRecommendedSampling() {
        XCTAssertTrue(Q35Resources.thinkingDefault(forModelId: Q35Resources.ornith35BMLXModelId))
        XCTAssertTrue(Q35Resources.thinkingDefault(forModelId: Q35Resources.ornith9BModelId))
        XCTAssertFalse(Q35Resources.thinkingDefault(forModelId: Q35Resources.q36NanoModelId))
        XCTAssertFalse(Q35Resources.thinkingDefault(forModelId: Gemma4Resources.twelveB4BitModelId))

        let sampling = Q35Resources.recommendedSampling(forModelId: Q35Resources.ornith35BMLXModelId)
        XCTAssertEqual(sampling?.temperature, 1.0)
        XCTAssertEqual(sampling?.topP, 0.95)
        XCTAssertEqual(sampling?.topK, 20)
        XCTAssertNil(Q35Resources.recommendedSampling(forModelId: Q35Resources.q36NanoModelId))
    }
}
