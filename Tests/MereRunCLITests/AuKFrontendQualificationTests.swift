import Foundation
import MLX
import MereRunAudioModels
import MereRunMLXTestSupport
import XCTest
@testable import MereRunCore

final class AuKFrontendQualificationTests: MLXTestCase {
    func testTokenizerAndMelMatchHuggingFaceProcessor() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let outputPath = environment["AUK_QUALIFICATION_ROOT"],
              let thinkerPath = environment["AUK_THINKER_ROOT"] else {
            throw XCTSkip("Set AuK qualification paths to compare the trained model frontend")
        }
        struct Request: Decodable { let instruction: String }
        let root = URL(fileURLWithPath: outputPath)
        let thinker = URL(fileURLWithPath: thinkerPath)
        let request = try JSONDecoder().decode(Request.self, from: Data(contentsOf: root.appendingPathComponent("request.json")))
        let inputs = try loadArrays(url: root.appendingPathComponent("inputs.safetensors"))
        let config = try AuKThinkerConfiguration.load(from: thinker.appendingPathComponent("config.json"))
        for name in ["text", "audio"] {
            let wave = name == "audio" ? try XCTUnwrap(inputs["wave16"]).asArray(Float.self) : nil
            let (tokens, mel) = try AuKGenerator.instructionInputs(request.instruction, audio: wave, root: thinker,
                                                                  audioTokenIndex: config.audioTokenIndex)
            XCTAssertEqual(tokens, try XCTUnwrap(inputs[name + "_ids"]).asArray(Int.self))
            if let mel {
                try save(arrays: ["mel": mel], url: root.appendingPathComponent("native-mel.safetensors"))
                let expected = try XCTUnwrap(inputs["audio_mel"])
                XCTAssertEqual(mel.shape, expected.shape)
                let error = abs(mel - expected).max().item(Float.self)
                print("AuK frontend mel maximum absolute error: \(error)")
                XCTAssertLessThan(error, 0.0001)
            }
        }
    }
}
