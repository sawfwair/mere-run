import AudioCore
@testable import AudioSTT
import Foundation
import XCTest

/// Qwen3-ASR transcribes in the spoken language unless the user turn asks for a translation, so
/// `--task translate` has to reach the prompt; before, the task was dropped and the run
/// transcribed.
final class Qwen3ASRTranslationPromptTests: XCTestCase {
    private var root: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try Data(#"{"tokenizer_class": "Qwen2Tokenizer"}"#.utf8).write(to: root.appendingPathComponent("tokenizer_config.json"))
        try Data().write(to: root.appendingPathComponent("merges.txt"))
        try JSONEncoder().encode(Self.byteLevelVocabulary()).write(to: root.appendingPathComponent("vocab.json"))
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
    }

    /// One token per byte, spelled the way a byte-level BPE vocabulary spells bytes, so any text
    /// encodes without merges.
    private static func byteLevelVocabulary() -> [String: Int] {
        var bytes = Array(33...126) + Array(161...172) + Array(174...255)
        var scalars = bytes
        for byte in 0..<256 where !bytes.contains(byte) {
            bytes.append(byte)
            scalars.append(256 + scalars.count - 188)
        }
        return Dictionary(uniqueKeysWithValues: zip(scalars, bytes).map { scalar, byte in
            (String(Character(UnicodeScalar(UInt32(scalar))!)), byte)
        })
    }

    private func prompt(_ task: ASRTask, language: String?) throws -> (tokens: [Int], tokenizer: Qwen3ASRTokenizer) {
        let tokenizer = try Qwen3ASRTokenizer.load(from: root)
        let tokens = tokenizer.createQwen3ASRPrompt(
            audioPlaceholderCount: 3,
            language: Qwen3ASRGenerator.promptLanguage(task: task, language: language),
            supportedLanguages: ["German", "English"],
            instruction: Qwen3ASRGenerator.instruction(for: task)
        )
        return (tokens, tokenizer)
    }

    func testTranslationAsksForEnglishAfterTheAudioAndAnswersInEnglish() throws {
        let (tokens, tokenizer) = try prompt(.translate, language: "german")
        let audioEnd = try XCTUnwrap(tokens.firstIndex(of: tokenizer.audioEndTokenId))
        let instruction = tokenizer.encode("Translate the audio to English.")
        XCTAssertEqual(Array(tokens[(audioEnd + 1)...].prefix(instruction.count)), instruction)
        XCTAssertEqual(tokens[audioEnd + 1 + instruction.count], tokenizer.imEndId)
        let answer = tokenizer.encode("assistant\nlanguage English") + [tokenizer.asrTextTokenId]
        XCTAssertEqual(Array(tokens.suffix(answer.count)), answer)
    }

    func testTranscriptionAsksNothingAndKeepsTheLanguageHint() throws {
        let (tokens, tokenizer) = try prompt(.transcribe, language: "german")
        let audioEnd = try XCTUnwrap(tokens.firstIndex(of: tokenizer.audioEndTokenId))
        XCTAssertEqual(tokens[audioEnd + 1], tokenizer.imEndId)
        let answer = tokenizer.encode("assistant\nlanguage German") + [tokenizer.asrTextTokenId]
        XCTAssertEqual(Array(tokens.suffix(answer.count)), answer)
    }
}
