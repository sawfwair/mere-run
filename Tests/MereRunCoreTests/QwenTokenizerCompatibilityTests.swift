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

    func testSiblingChatTemplateIsLoadedWhenConfigOmitsTemplate() throws {
        let temp = try TestFileSystem.makeTempDir()
        defer { try? FileManager.default.removeItem(at: temp) }

        let configURL = temp.appendingPathComponent("tokenizer_config.json")
        let templateURL = temp.appendingPathComponent("chat_template.jinja")
        try TestFileSystem.writeFile(templateURL, contents: Data("hello {{ messages }}".utf8))

        let data = try JSONSerialization.data(withJSONObject: [
            "tokenizer_class": "TokenizersBackend",
            "model_max_length": 262_144,
        ], options: [])
        let config = try QwenTokenizer.normalizedTokenizerConfig(
            data: data,
            url: configURL,
            overrideTokenizerClass: "Qwen2Tokenizer"
        )

        XCTAssertEqual(config["chat_template"].string(), "hello {{ messages }}")
    }
}
