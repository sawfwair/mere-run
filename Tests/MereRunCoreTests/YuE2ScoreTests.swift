import Foundation
import XCTest
@testable import MereRunCore

final class YuE2ScoreTests: XCTestCase {
    private func score(vocal: String, instrumental: String, key: String = "C") -> String {
        """
        X:1
        T:
        M:4/4
        L:1/32
        Q:1/4=120
        V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
        V: Ins clef=treble name="Ins Melody" snm="Inst."
        K:\(key)
        % Verse
        V: Vocal
        \(vocal)
        V: Ins
        \(instrumental)
        """
    }

    func testTransferPreservesMelodyAndClipsOverlappingAccompaniment() throws {
        let source = try YuE2Score(abc: score(vocal: "z8 \"G7\" ^F8- F8 z8 |", instrumental: "C32 |", key: "G"))
        let result = try source.instrumentalized()
        let output = try YuE2Score(abc: result.abc)
        XCTAssertTrue(output.vocal.isEmpty)
        XCTAssertEqual(output.instrumental, [.init(onset: 0, pitch: 60, duration: 256),
            .init(onset: 256, pitch: 66, duration: 512), .init(onset: 768, pitch: 60, duration: 256)])
        XCTAssertEqual(output.chords, source.chords)
        XCTAssertEqual(output.sections, source.sections)
        XCTAssertEqual(result.affectedInstrumentalNotes, [0])
        XCTAssertThrowsError(try source.instrumentalized(overlap: .error))
        XCTAssertEqual(try YuE2Score(abc: source.instrumentalized(keepChords: false).abc).chords, [])
    }

    func testAccidentalTieAcrossBarAndCompressedRestsRoundTrip() throws {
        let source = try YuE2Score(abc: score(vocal: "^F32- | F16 z16 |", instrumental: "Z2 |"))
        XCTAssertEqual(source.vocal, [.init(onset: 0, pitch: 66, duration: 1536)])
        let output = try YuE2Score(abc: source.instrumentalized().abc)
        XCTAssertEqual(output.instrumental, source.vocal)
        XCTAssertEqual(output.bars, source.bars)
    }

    func testUnsupportedOrInconsistentNotationFailsBeforeConversion() throws {
        for (vocal, instrumental) in [("C8 |", "Z |"), ("C32- |", "Z |"), ("[CEG]32 |", "Z |"),
                                      ("C32 |", "Z2 |"), ("z32 |", "Z |"), ("C32 |", "\"C\" C32 |") ] {
            XCTAssertThrowsError(try YuE2Score(abc: score(vocal: vocal, instrumental: instrumental)).instrumentalized())
        }
    }

    func testMinimalJSONRequestResolvesDefaultsAndInstrumentalConflicts() throws {
        let plan = try JSONDecoder().decode(YuE2GenerationPlan.self, from: Data(#"{"style":"piano","instrumental":true}"#.utf8))
        XCTAssertEqual(plan.instrumental, true)
        XCTAssertEqual(plan.planning, .full)
        XCTAssertEqual(plan.steps, 32)
        XCTAssertThrowsError(try YuE2GenerationPlan(style: "piano", lyrics: "words", instrumental: true))
        XCTAssertThrowsError(try YuE2GenerationPlan(style: "piano", lyrics: "", planning: .off, instrumental: true))
    }

    func testTokenizerPreflightChecksContextAndConvertsExternalInstrumentalScore() throws {
        let tokens = (0...255).map { Data([UInt8($0)]) }
        let contents = tokens.enumerated().map { "\($0.element.base64EncodedString()) \($0.offset)" }.joined(separator: "\n")
        let tokenizer = try YuE2Tokenizer(contents: contents, vocabularySize: tokens.count)
        let planning = try YuE2GenerationPlan(style: "piano", lyrics: "").preflight(tokenizer: tokenizer)
        XCTAssertEqual(planning.nextStage, .score)
        XCTAssertNil(planning.acousticChunkFrames)
        let abc = score(vocal: "C32 |", instrumental: "Z |")
        let request = try YuE2GenerationPlan(style: "piano", lyrics: "", abc: abc, instrumental: true)
        let supplied = try request.preflight(tokenizer: tokenizer)
        XCTAssertEqual(supplied.nextStage, .semantic)
        XCTAssertEqual(supplied.scoreTokens, try tokenizer.encode(YuE2Score(abc: abc).instrumentalized().abc).count)
        XCTAssertEqual(supplied.acousticChunkFrames?.reduce(0, +), request.semanticSampling.maximumTokens)
        XCTAssertEqual(supplied.maximumDecodedSamples, 9000 * 1920 - 64)
        XCTAssertThrowsError(try YuE2GenerationPlan(style: String(repeating: "x", count: 25000), lyrics: "")
            .preflight(tokenizer: tokenizer))
    }
}
