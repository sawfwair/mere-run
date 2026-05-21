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
    }

    func testTextChatParsesStreamingFlag() throws {
        let cmd = try TextChat.parse([
            "--prompt", "Stream this",
            "--stream",
        ])

        XCTAssertTrue(cmd.stream)
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
}
