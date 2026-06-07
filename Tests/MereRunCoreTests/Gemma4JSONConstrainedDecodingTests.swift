import Foundation
import MLX
import XCTest
@testable import MereRunCore

final class Gemma4JSONConstrainedDecodingTests: XCTestCase {
    // MARK: - JSONPrefixScanner

    func testAcceptsValidJSONIncrementally() {
        var scanner = JSONPrefixScanner()
        // Pieces shaped like real BPE output: fragments that split tokens mid-string.
        let pieces = ["{\"short", " desc", "ription\": \"A sign", " in a forest\",", " \"count\": 12.5,", " \"ok\": true}", ""]
        for piece in pieces {
            XCTAssertTrue(scanner.accept(piece), "rejected valid piece: \(piece)")
        }
        XCTAssertTrue(scanner.isComplete)
    }

    func testAcceptsNestedContainersAndEscapes() {
        var scanner = JSONPrefixScanner()
        XCTAssertTrue(scanner.accept(#"{"a": [{"b": "she said \"hi\""}, [1, -2.5e3, null]], "c": false}"#))
        XCTAssertTrue(scanner.isComplete)
    }

    func testAcceptsNonASCIIInsideStringsOnly() {
        var inString = JSONPrefixScanner()
        XCTAssertTrue(inString.accept(#"{"título": "señal — 標識 🌲"#))

        var outside = JSONPrefixScanner()
        XCTAssertFalse(outside.accept(#"{標識"#))
    }

    func testRejectsTokenSalad() {
        var scanner = JSONPrefixScanner()
        XCTAssertFalse(scanner.accept("{ed feetization mas Dod asked or<audio|>"))

        var fresh = JSONPrefixScanner()
        XCTAssertFalse(fresh.accept("<|image>"))
    }

    func testRejectsProseAndFencesBeforeJSONStarts() {
        var scanner = JSONPrefixScanner()
        XCTAssertFalse(scanner.accept("Here is the JSON you asked for: {"))

        var fenced = JSONPrefixScanner()
        XCTAssertFalse(fenced.accept("```json"))

        var leadingWhitespace = JSONPrefixScanner()
        XCTAssertTrue(leadingWhitespace.accept("  \n\t{"))
    }

    func testRejectsMismatchedBrackets() {
        var scanner = JSONPrefixScanner()
        XCTAssertFalse(scanner.accept(#"{"a": [1, 2}"#))

        var underflow = JSONPrefixScanner()
        XCTAssertTrue(underflow.accept("{}"))
        XCTAssertFalse(underflow.accept("}"))
    }

    func testCompleteAllowsOnlyTrailingWhitespace() {
        var scanner = JSONPrefixScanner()
        XCTAssertTrue(scanner.accept(#"{"done": true}"#))
        XCTAssertTrue(scanner.isComplete)
        XCTAssertTrue(scanner.accept(" \n"))
        XCTAssertFalse(scanner.accept("extra"))
    }

    func testAllowsOnlyJSONLiteralsOutsideStrings() {
        var literals = JSONPrefixScanner()
        XCTAssertTrue(literals.accept(#"{"a": true, "b": false, "c": null}"#))
        XCTAssertTrue(literals.isComplete)

        // Split across BPE-style pieces.
        var split = JSONPrefixScanner()
        XCTAssertTrue(split.accept(#"{"a": tr"#))
        XCTAssertTrue(split.accept("ue}"))

        // Arbitrary words made of literal letters must not slip through —
        // this was the context-poisoning vector for mid-generation salad.
        var junk = JSONPrefixScanner()
        XCTAssertFalse(junk.accept(#"{"a": ensure"#))

        var uppercase = JSONPrefixScanner()
        XCTAssertFalse(uppercase.accept(#"{"a": True"#))
    }

    // MARK: - Token ban mask

    func testApplyTokenBanMakesTokensUnsampleable() {
        // Token 3 has by far the highest logit; banning it must reroute argmax.
        let logits = MLXArray([Float](
            [0.1, 0.2, 0.3, 10.0, 0.4]
        ))
        let banned = applyTokenBan(logits: logits, tokens: [3])
        XCTAssertEqual(argMaxSample(logits: banned), 4)
        // Out-of-range ids are ignored rather than crashing.
        let unchanged = applyTokenBan(logits: logits, tokens: [99, -1])
        XCTAssertEqual(argMaxSample(logits: unchanged), 3)
    }

    func testGenerationConfigBannedTokensFlowThroughSampleToken() {
        let logits = MLXArray([Float]([0.0, 8.0, 0.0, 0.0]))
        var config = GenerationConfig(temperature: 0)
        config.bannedTokens = [1]
        let token = sampleToken(logits: logits, config: config, previousTokens: [])
        XCTAssertNotEqual(token, 1)
    }
}
