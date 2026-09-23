import Foundation
import XCTest
@testable import MereRunCore

final class FalconPerceptionStopTokenTests: XCTestCase {
    private func withTokenizer(
        endOfQueryID: Int?,
        run: (FalconPerceptionTokenizer) throws -> Void
    ) throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        struct AddedToken: Encodable {
            let id: Int
            let content: String
            let single_word = false
            let lstrip = false
            let rstrip = false
            let normalized = false
            let special = true
        }
        struct Model: Encodable {
            let type = "BPE"
            let vocab: [String: Int]
            let merges: [String] = []
            let unk_token = "<|pad|>"
        }
        struct DataFile: Encodable {
            let version = "1.0"
            let added_tokens: [AddedToken]
            let model: Model
        }
        var tokens = [AddedToken(id: 0, content: "<|pad|>"), AddedToken(id: 11, content: "<|endoftext|>")]
        if let endOfQueryID { tokens.append(AddedToken(id: endOfQueryID, content: "<|end_of_query|>")) }
        let file = DataFile(added_tokens: tokens, model: Model(vocab: Dictionary(uniqueKeysWithValues: tokens.map { ($0.content, $0.id) })))
        try JSONEncoder().encode(file).write(to: root.appendingPathComponent("tokenizer.json"))
        try Data("""
        {"tokenizer_class":"PreTrainedTokenizerFast","pad_token":"<|pad|>","eos_token":"<|endoftext|>","model_max_length":8192}
        """.utf8).write(to: root.appendingPathComponent("tokenizer_config.json"))
        try run(FalconPerceptionTokenizer.load(from: root))
    }

    func testStopsAtModelEosAndThePinnedQueryTerminator() throws {
        try withTokenizer(endOfQueryID: 263) { tokenizer in
            XCTAssertEqual(tokenizer.endOfQueryTokenID, 263)
            XCTAssertTrue(tokenizer.isGenerationStopToken(11, modelEosTokenID: 11))
            XCTAssertTrue(tokenizer.isGenerationStopToken(263, modelEosTokenID: 11))
            for token in [0, 12, 240, 241, 262, 264] {
                XCTAssertFalse(tokenizer.isGenerationStopToken(token, modelEosTokenID: 11))
            }
        }
    }

    func testResolvesQueryTerminatorFromVocabularyInsteadOfHardcodingAnID() throws {
        try withTokenizer(endOfQueryID: 37) { tokenizer in
            XCTAssertTrue(tokenizer.isGenerationStopToken(37, modelEosTokenID: 99))
            XCTAssertTrue(tokenizer.isGenerationStopToken(99, modelEosTokenID: 99))
            XCTAssertFalse(tokenizer.isGenerationStopToken(263, modelEosTokenID: 99))
            XCTAssertFalse(tokenizer.isGenerationStopToken(11, modelEosTokenID: 99))
        }
    }

    func testTokenizerWithoutQueryTerminatorKeepsModelEosBehavior() throws {
        try withTokenizer(endOfQueryID: nil) { tokenizer in
            XCTAssertNil(tokenizer.endOfQueryTokenID)
            XCTAssertTrue(tokenizer.isGenerationStopToken(11, modelEosTokenID: 11))
            XCTAssertFalse(tokenizer.isGenerationStopToken(263, modelEosTokenID: 11))
            XCTAssertFalse(tokenizer.isGenerationStopToken(0, modelEosTokenID: 11))
        }
    }

    func testPinnedTokenizerWithoutLoadingCheckpointWeights() throws {
        guard let path = ProcessInfo.processInfo.environment["MERERUN_FALCON_STOP_TOKENIZER_ROOT"] else {
            throw XCTSkip("Set MERERUN_FALCON_STOP_TOKENIZER_ROOT to verify the pinned tokenizer without model inference.")
        }
        let root = URL(fileURLWithPath: path)
        let tokenizer = try FalconPerceptionTokenizer.load(from: root)
        let config = try FalconPerceptionModelConfig.load(from: root.appendingPathComponent("config.json"))
        XCTAssertEqual(tokenizer.endOfQueryTokenID, 263)
        XCTAssertEqual(config.eosID, 11)
        XCTAssertEqual(config.vocabSize, 65536)
        let observed = (0..<config.vocabSize).filter {
            tokenizer.isGenerationStopToken($0, modelEosTokenID: config.eosID)
        }
        XCTAssertEqual(observed, [11, 263])
    }
}
