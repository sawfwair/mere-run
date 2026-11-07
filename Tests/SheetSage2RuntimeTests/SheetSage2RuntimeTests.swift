import Foundation
import MLX
import MereRunMLXTestSupport
import XCTest
@testable import MereRunSheetSageModel

final class SheetSage2RuntimeTests: XCTestCase {
    private var root: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures") }
    override func setUp() { super.setUp(); MLXTestSupport.ensureMetalLibraryAvailable() }
    private func tensors(_ name: String) throws -> [String: MLXArray] { try MLX.loadArrays(url: root.appendingPathComponent(name)) }
    private func model() throws -> SheetSage2Model {
        try SheetSage2Model(configuration: JSONDecoder().decode(SheetSage2Configuration.self,
            from: Data(contentsOf: root.appendingPathComponent("config.json"))),
            parent: tensors("parent.safetensors"), adapter: tensors("adapter.safetensors"))
    }
    private func close(_ actual: MLXArray, _ expected: MLXArray, _ label: String, tolerance: Float = 3e-5) {
        XCTAssertEqual(actual.shape, expected.shape, label)
        guard actual.shape == expected.shape else { return }
        XCTAssertLessThanOrEqual(max(abs(actual - expected)).item(Float.self), tolerance, label)
    }
    func testFrontendEncoderAndCachedDecoderMatchIndependentUpstream() throws {
        let model = try model()
        let expected = try tensors("expected.safetensors")
        let audio = try XCTUnwrap(expected["audio"])
        let mel = try model.encoder.mel(audio)
        close(mel, try XCTUnwrap(expected["mel"]), "mel", tolerance: 3e-4)
        let subsampled = try model.encoder.subsample(mel)
        close(subsampled, try XCTUnwrap(expected["subsampled"]), "ConvNeXt")
        close(try model.encoder.block(subsampled, index: 0), try XCTUnwrap(expected["block0"]), "Conformer")
        let memory = try model.encode(audio, windowSamples: audio.size)
        close(memory, try XCTUnwrap(expected["encoder"]), "merged adapter and weighted encoder")
        let tokens = try XCTUnwrap(expected["tokens"]).asArray(Int32.self).map(Int.init)
        let cache = SheetSage2DecoderCache(layers: 2)
        close(try model.logits(tokens, memory: memory, cache: cache), try XCTUnwrap(expected["prefill"]), "prefill")
        close(try model.logits([9], memory: memory, cache: cache), try XCTUnwrap(expected["cached"]), "cached")
        close(try model.logits(tokens + [9], memory: memory, cache: SheetSage2DecoderCache(layers: 2)),
              try XCTUnwrap(expected["uncached"]), "uncached")
    }
    func testEventMasksMatchUpstreamGrammarAtEveryStep() throws {
        struct Reference: Decodable { let tokens: [Int]; let allowed: [[Int]] }
        let reference = try JSONDecoder().decode(Reference.self, from: Data(contentsOf: root.appendingPathComponent("grammar.json")))
        var grammar = SheetSage2Vocabulary.Grammar()
        for (index, token) in reference.tokens.enumerated() {
            let allowed = grammar.mask().asArray(Bool.self).enumerated().compactMap { $0.element ? $0.offset : nil }
            XCTAssertEqual(allowed, reference.allowed[index], "Token \(index)")
            try grammar.update(token)
        }
        XCTAssertEqual(SheetSage2Vocabulary.chords.count, SheetSage2Vocabulary.chord.count)
        XCTAssertThrowsError(try grammar.update(0))
    }
    func testMissingAndUnexpectedTensorsAreRejected() throws {
        let config = try JSONDecoder().decode(SheetSage2Configuration.self, from: Data(contentsOf: root.appendingPathComponent("config.json")))
        var parent = try tensors("parent.safetensors")
        parent["extra.weight"] = MLXArray.zeros([1])
        XCTAssertThrowsError(try SheetSage2Model(configuration: config, parent: parent, adapter: tensors("adapter.safetensors")))
        parent.removeValue(forKey: "extra.weight")
        parent.removeValue(forKey: "feature_extractor.mel_std")
        XCTAssertThrowsError(try SheetSage2Model(configuration: config, parent: parent, adapter: tensors("adapter.safetensors")))
    }
}
