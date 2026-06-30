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
        XCTAssertEqual(cmd.model, Gemma4Resources.twelveB4BitModelId)
    }

    func testTextChatParsesStreamingFlag() throws {
        let cmd = try TextChat.parse([
            "--prompt", "Stream this",
            "--stream",
        ])

        XCTAssertTrue(cmd.stream)
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
}
