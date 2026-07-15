import Foundation
import MLX
import XCTest
@testable import MereRunCore

final class JSONConstrainedDecodingTests: MereRunCoreTestCase {
    // MARK: - JSONObjectPrefixGrammar

    func testAcceptsValidJSONIncrementally() {
        var grammar = JSONObjectPrefixGrammar()
        // Pieces shaped like real BPE output: fragments that split tokens mid-string.
        let pieces = ["{\"short", " desc", "ription\": \"A sign", " in a forest\",", " \"count\": 12.5,", " \"ok\": true}", ""]
        for piece in pieces {
            XCTAssertTrue(grammar.accept(piece), "rejected valid piece: \(piece)")
        }
        XCTAssertTrue(grammar.isComplete)
    }

    func testAcceptsNestedContainersAndEscapes() {
        let json = "{\"a\":[{\"b\":\"she said \\\"hi\\\"\"},[1,-2.5e3,null]],\"c\":false}"
        var grammar = JSONObjectPrefixGrammar()
        XCTAssertTrue(grammar.accept(json))
        XCTAssertTrue(grammar.isComplete)
        XCTAssertNoThrow(try JSONSerialization.jsonObject(with: Data(json.utf8)))
    }

    func testAcceptsUnicodeAndEscapedStrings() {
        let json = "{\"título\":\"señal — 標識 🌲\",\"escaped\":\"line\\nquote: \\\" and smile: \\u263A\"}"
        var grammar = JSONObjectPrefixGrammar()
        XCTAssertTrue(grammar.accept(json))
        XCTAssertTrue(grammar.isComplete)
        XCTAssertNoThrow(try JSONSerialization.jsonObject(with: Data(json.utf8)))

        var outside = JSONObjectPrefixGrammar()
        XCTAssertFalse(outside.accept("{標識"))
    }

    func testRejectsTokenSalad() {
        var grammar = JSONObjectPrefixGrammar()
        XCTAssertFalse(grammar.accept("{ed feetization mas Dod asked or<audio|>"))

        var fresh = JSONObjectPrefixGrammar()
        XCTAssertFalse(fresh.accept("<|image>"))
    }

    func testRejectsProseAndFencesBeforeJSONStarts() {
        var grammar = JSONObjectPrefixGrammar()
        XCTAssertFalse(grammar.accept("Here is the JSON you asked for: {"))

        var fenced = JSONObjectPrefixGrammar()
        XCTAssertFalse(fenced.accept("```json"))

        var leadingWhitespace = JSONObjectPrefixGrammar()
        XCTAssertTrue(leadingWhitespace.accept("  \n\t{"))
    }

    func testRequiresObjectRoot() {
        for invalidRoot in ["[]", "true", "null", "\"text\"", "42"] {
            var grammar = JSONObjectPrefixGrammar()
            XCTAssertFalse(grammar.accept(invalidRoot), "accepted non-object root: \(invalidRoot)")
        }
    }

    func testRejectsInvalidObjectAndArrayState() {
        let invalidDocuments = [
            "{\"a\" 1}",
            "{\"a\":1 \"b\":2}",
            "{\"a\":1,}",
            "{\"a\":[1,]}",
            "{\"a\":[1,2}",
            "{\"a\":{]}",
        ]
        for document in invalidDocuments {
            var grammar = JSONObjectPrefixGrammar()
            XCTAssertFalse(grammar.accept(document), "accepted invalid JSON: \(document)")
        }

        var underflow = JSONObjectPrefixGrammar()
        XCTAssertTrue(underflow.accept("{}"))
        XCTAssertFalse(underflow.accept("}"))
    }

    func testRejectsInvalidNumbersLiteralsEscapesAndControlCharacters() {
        let invalidDocuments = [
            "{\"n\":01}",
            "{\"n\":-}",
            "{\"n\":1.}",
            "{\"n\":1e}",
            "{\"n\":1e+}",
            "{\"v\":truth}",
            "{\"v\":True}",
            "{\"s\":\"bad\\xescape\"}",
            "{\"s\":\"line\nbreak\"}",
        ]
        for document in invalidDocuments {
            var grammar = JSONObjectPrefixGrammar()
            XCTAssertFalse(grammar.accept(document), "accepted invalid JSON: \(document)")
        }
    }

    func testCompleteAllowsOnlyTrailingWhitespace() {
        var grammar = JSONObjectPrefixGrammar()
        XCTAssertTrue(grammar.accept("{\"done\":true}"))
        XCTAssertTrue(grammar.isComplete)
        XCTAssertTrue(grammar.accept(" \n"))
        XCTAssertFalse(grammar.accept("extra"))
    }

    func testAllowsOnlyJSONLiteralsOutsideStrings() {
        var literals = JSONObjectPrefixGrammar()
        XCTAssertTrue(literals.accept("{\"a\":true,\"b\":false,\"c\":null}"))
        XCTAssertTrue(literals.isComplete)

        // Split across BPE-style pieces.
        var split = JSONObjectPrefixGrammar()
        XCTAssertTrue(split.accept("{\"a\":tr"))
        XCTAssertTrue(split.accept("ue}"))

        var junk = JSONObjectPrefixGrammar()
        XCTAssertFalse(junk.accept("{\"a\":ensure"))

        var uppercase = JSONObjectPrefixGrammar()
        XCTAssertFalse(uppercase.accept("{\"a\":True"))
    }

    func testEveryPrefixOfValidDocumentIsAccepted() {
        let json = "{\"nested\":{\"array\":[-12,0,3.5,6.02e23,true,false,null,\"🌊\"]}}"
        for end in json.indices {
            let prefix = String(json[...end])
            var grammar = JSONObjectPrefixGrammar()
            XCTAssertTrue(grammar.accept(prefix), "rejected valid prefix: \(prefix)")
        }
    }

    func testStreamingAndNonStreamingOutputAreIdenticalValidJSON() throws {
        let output = "{\"summary\":\"café \\\"ready\\\"\",\"items\":[{\"id\":1},{\"id\":2}]}"
        var nonStreaming = JSONObjectPrefixGrammar()
        XCTAssertTrue(nonStreaming.accept(output))
        XCTAssertTrue(nonStreaming.isComplete)

        var streaming = JSONObjectPrefixGrammar()
        let pieces = ["{\"sum", "mary\":\"caf", "é \\\"ready\\\"\",", "\"items\":[{\"id\":1}", ",", "{\"id\":2}]}" ]
        for piece in pieces {
            XCTAssertTrue(streaming.accept(piece), "rejected streaming piece: \(piece)")
        }
        XCTAssertTrue(streaming.isComplete)
        XCTAssertNoThrow(try JSONSerialization.jsonObject(with: Data(output.utf8)))
    }

    func testConstrainedTokenSearchesBeyondDistributionHead() {
        var logits = (0..<80).map(Float.init)
        logits[0] = -100
        var grammar = JSONObjectPrefixGrammar()
        let token = jsonConstrainedToken(
            initial: 79,
            logits: MLXArray(logits),
            config: GenerationConfig(temperature: 0),
            eosSet: [],
            grammar: &grammar,
            decode: { $0 == 0 ? "{" : "prose" }
        )
        XCTAssertEqual(token, 0)
    }

    func testEOSIsAdmittedOnlyAfterRootObjectCloses() {
        var incomplete = JSONObjectPrefixGrammar()
        XCTAssertNil(jsonConstrainedToken(
            initial: 0,
            logits: MLXArray([Float(1)]),
            config: GenerationConfig(temperature: 0),
            eosSet: [0],
            grammar: &incomplete,
            decode: { _ in "" }
        ))

        var complete = JSONObjectPrefixGrammar()
        XCTAssertTrue(complete.accept("{}"))
        XCTAssertEqual(jsonConstrainedToken(
            initial: 0,
            logits: MLXArray([Float(1)]),
            config: GenerationConfig(temperature: 0),
            eosSet: [0],
            grammar: &complete,
            decode: { _ in "" }
        ), 0)
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

    func testGenerationConfigBannedTokensFlowThroughGreedyTokenArray() {
        let logits = MLXArray([Float]([0.0, 8.0, 0.0, 0.0]))
        var config = GenerationConfig(temperature: 0)
        config.bannedTokens = [1]
        let token = greedySampleTokenArray(logits: logits, config: config, previousTokens: [])
        XCTAssertNotEqual(token.item(Int.self), 1)
    }

    func testGreedyTokenArrayHonorsTensorRepetitionPenalty() {
        let logits = MLXArray([Float]([0.0, 5.0, 4.0]))
        let config = GenerationConfig(temperature: 0, repetitionPenalty: 2.0)
        let previous = MLXArray([Int32(1)])
        let token = greedySampleTokenArray(
            logits: logits,
            config: config,
            previousTokenIndices: previous
        )
        XCTAssertEqual(token.item(Int.self), 2)
    }

    func testGemma4MultimodalDecodeBanIncludesVisionPromptTokens() {
        let banned = Gemma4Generator.multimodalDecodeBannedTokens(
            imageTokenId: 10,
            audioTokenId: 11,
            videoTokenId: 12,
            boiTokenId: 13,
            boaTokenId: 14,
            eoiTokenId: 15,
            eoaTokenId: 16,
            excluding: [15]
        )

        XCTAssertEqual(banned, [10, 11, 12, 13, 14, 16])
    }
}
