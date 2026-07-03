import Foundation
import XCTest
@testable import MereRunCore

final class SAM31TokenizerCompatibilityTests: MereRunCoreTestCase {

    private func normalizedTokenizerClass(from object: [String: Any]) throws -> String? {
        let data = try JSONSerialization.data(withJSONObject: object, options: [])
        let config = try SAM31Tokenizer.normalizedTokenizerConfig(
            data: data,
            url: URL(fileURLWithPath: "/tmp/tokenizer_config.json"),
            overrideTokenizerClass: "GPT2Tokenizer"
        )
        return config["tokenizer_class"].string()
    }

    func testClipTokenizerMirrorConfigIsRemappedToSupportedTokenizer() throws {
        let tokenizerClass = try normalizedTokenizerClass(from: [
            "tokenizer_class": "CLIPTokenizer",
            "model_max_length": 32,
        ])
        XCTAssertEqual(tokenizerClass, "GPT2Tokenizer")
    }

    func testTokenizersBackendConfigIsRemappedToSupportedTokenizer() throws {
        let tokenizerClass = try normalizedTokenizerClass(from: [
            "tokenizer_class": "TokenizersBackend",
            "model_max_length": 32,
        ])
        XCTAssertEqual(tokenizerClass, "GPT2Tokenizer")
    }

    func testMissingTokenizerClassIsFilledForSAM() throws {
        let tokenizerClass = try normalizedTokenizerClass(from: [
            "model_max_length": 32,
        ])
        XCTAssertEqual(tokenizerClass, "GPT2Tokenizer")
    }
}
