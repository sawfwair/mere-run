import Foundation
import XCTest
@testable import MereRunCore

final class QwenTokenizerCompatibilityTests: MereRunCoreTestCase {

    private func normalizedTokenizerClass(from object: [String: Any]) throws -> String? {
        let data = try JSONSerialization.data(withJSONObject: object, options: [])
        let config = try QwenTokenizer.normalizedTokenizerConfig(
            data: data,
            url: URL(fileURLWithPath: "/tmp/tokenizer_config.json"),
            overrideTokenizerClass: "Qwen2Tokenizer"
        )
        return config["tokenizer_class"].string()
    }

    func testTokenizersBackendIsRemappedToSupportedTokenizer() throws {
        let tokenizerClass = try normalizedTokenizerClass(from: [
            "tokenizer_class": "TokenizersBackend",
            "model_max_length": 262_144,
        ])
        XCTAssertEqual(tokenizerClass, "Qwen2Tokenizer")
    }

    func testSupportedTokenizerClassIsPreserved() throws {
        let tokenizerClass = try normalizedTokenizerClass(from: [
            "tokenizer_class": "Qwen2Tokenizer",
            "model_max_length": 262_144,
        ])
        XCTAssertEqual(tokenizerClass, "Qwen2Tokenizer")
    }

    func testMissingTokenizerClassDoesNotCrashNormalization() throws {
        let tokenizerClass = try normalizedTokenizerClass(from: [
            "model_max_length": 262_144,
        ])
        XCTAssertNil(tokenizerClass)
    }
}
