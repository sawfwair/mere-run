import Foundation
import MLX
import MereRunMLXTestSupport
import XCTest
@testable import AudioWhistleModel

final class WhistleModelTests: MLXTestCase {
    func testSyntheticEncoderAndFourCachedStepsMatchIndependentGraph() throws {
        let url = try XCTUnwrap(Bundle.module.url(forResource: "synthetic", withExtension: "safetensors", subdirectory: "Fixtures/Whistle"))
        let (reference, metadata) = try MLX.loadArraysAndMetadata(url: url)
        let layout = try JSONDecoder().decode([String: [Int]].self, from: Data(try XCTUnwrap(metadata["layout"]).utf8))
        var arrays: [String: MLXArray] = [:]
        for (index, name) in layout.keys.sorted().enumerated() {
            let shape = try XCTUnwrap(layout[name])
            let count = shape.reduce(1, *)
            let pattern = MLXArray((0..<97).map { Float(sin(Double($0) * 0.1 + Double(index) * 0.3) * 0.02) })
            arrays[name] = take(pattern, MLXArray(0..<count) % 97).reshaped(shape)
        }
        let model = WhistleModel(weights: try WhistleWeights(arrays: arrays))
        let audio = try model.encode(try XCTUnwrap(reference["mel"]))
        assertClose(stacked(audio.keys).reshaped(8, 8, 4, 48), try XCTUnwrap(reference["keys"]), tolerance: 2e-5)
        assertClose(stacked(audio.values).reshaped(8, 8, 4, 64), try XCTUnwrap(reference["values"]), tolerance: 2e-5)
        var cache = WhistleDecoderCache()
        for (step, token) in [2, 8192, 234, 456].enumerated() {
            let logits = try model.decode(token: token, audio: audio, cache: &cache)
            assertClose(logits, try XCTUnwrap(reference["logits"])[step], tolerance: 2e-4)
        }
        XCTAssertEqual(cache.tokens.count, 4)
        XCTAssertEqual(cache.values[7]?.shape, [1, 2, 4, 64])
    }

    func testEngramHashMatchesUnsignedReferenceAndIncludesTableOffsets() {
        XCTAssertEqual(WhistleModel.engramIndices(tokens: [2, 8192, 234, 456], position: 3),
                       [2854, 34798, 40813, 58715])
    }

    func testVocabularyRejectsTruncatedOrWrongContainer() {
        XCTAssertThrowsError(try WhistleVocabulary(cact: Data()))
        XCTAssertThrowsError(try WhistleVocabulary(cact: Data(repeating: 0, count: 320)))
    }

    func testNormUsesZeroCenteredScale() {
        let x = MLXArray([Float(3), 4]).reshaped(1, 2)
        let scale = MLXArray([Float(0), 1])
        let actual = WhistleMath.norm(x, scale).asArray(Float.self)
        XCTAssertEqual(actual[0], 3 / sqrt(12.5 + 1e-6), accuracy: 1e-6)
        XCTAssertEqual(actual[1], 8 / sqrt(12.5 + 1e-6), accuracy: 1e-6)
    }

    /// Optional complete-checkpoint comparison against an independently exported audio graph.
    /// Normal CI remains asset-free; the runtime guide documents fixture generation.
    func testOriginalCheckpointMatchesIndependentEncoderAndCachedDecoder() throws {
        guard let directory = ProcessInfo.processInfo.environment["MERERUN_WHISTLE_PARITY_DIR"] else {
            throw XCTSkip("Set MERERUN_WHISTLE_PARITY_DIR for original-checkpoint parity")
        }
        struct Reference: Decodable {
            let mel: [Float]
            let keys: [Float]
            let values: [Float]
            let logits: [[Float]]
        }
        let root = URL(fileURLWithPath: directory)
        let reference = try JSONDecoder().decode(Reference.self, from: Data(contentsOf: root.appendingPathComponent("whistle-reference.json")))
        let model = WhistleModel(weights: try WhistleWeights.load(from: root.appendingPathComponent("whistle.safetensors")))
        let context = try model.encode(MLXArray(reference.mel, [32, 80]))
        assertClose(stacked(context.keys).reshaped(-1), MLXArray(reference.keys), tolerance: 3e-4)
        assertClose(stacked(context.values).reshaped(-1), MLXArray(reference.values), tolerance: 3e-4)
        var cache = WhistleDecoderCache()
        for (step, token) in [2, 8192, 234, 456].enumerated() {
            let actual = try model.decode(token: token, audio: context, cache: &cache)
            assertClose(actual, MLXArray(reference.logits[step]), tolerance: 3e-3)
        }
        XCTAssertEqual(cache.tokens, [2, 8192, 234, 456])
        XCTAssertEqual(cache.keys[0]?.shape, [1, 2, 4, 48])
        XCTAssertEqual(cache.projections[0].count, 6)
        XCTAssertThrowsError(try model.decode(token: 8199, audio: context, cache: &cache))
    }

    func testPackedArchiveMatchesIndependentDequantizedGraph() throws {
        try Device.withDefaultDevice(Device(.gpu)) {
        guard let directory = ProcessInfo.processInfo.environment["MERERUN_WHISTLE_PARITY_DIR"] else {
            throw XCTSkip("Set MERERUN_WHISTLE_PARITY_DIR for packed-checkpoint parity")
        }
        struct Reference: Decodable { let mel: [Float]; let keys: [Float]; let values: [Float]; let logits: [[Float]] }
        let root = URL(fileURLWithPath: directory)
        let reference = try JSONDecoder().decode(Reference.self, from: Data(contentsOf: root.appendingPathComponent("whistle-cq-reference.json")))
        let weights = try WhistleWeights.load(cact: Data(contentsOf: root.appendingPathComponent("whistle.cact")))
        XCTAssertNil(weights.arrays["engrams_0/embedding"])
        XCTAssertEqual(weights.packed["engrams_0/embedding"]?.first?.packed.size, 2_359_296)
        let vocabulary = try WhistleVocabulary(cact: Data(contentsOf: root.appendingPathComponent("whistle.cact")))
        for text in [" Siobhan", " Grüße Łódź 🚀", " e\u{301}"] {
            XCTAssertEqual(try vocabulary.decode(vocabulary.encode(text)), text)
        }
        let model = WhistleModel(weights: weights)
        let context = try model.encode(MLXArray(reference.mel, [32, 80]))
        assertClose(stacked(context.keys).reshaped(-1), MLXArray(reference.keys), tolerance: 6e-4)
        assertClose(stacked(context.values).reshaped(-1), MLXArray(reference.values), tolerance: 6e-4)
        var cache = WhistleDecoderCache()
        for (step, token) in [2, 8192, 234, 456].enumerated() {
            assertClose(try model.decode(token: token, audio: context, cache: &cache), MLXArray(reference.logits[step]), tolerance: 6e-3)
        }
        }
    }

    func testPackedCQ2CQ4MatrixProductsAndSparseRowGather() throws {
        try Device.withDefaultDevice(Device(.gpu)) {
        struct Fixture: Decodable {
            let bits: Int; let packed: [UInt8]; let norms: [Float]; let codebook: [Float]
            let input: [Float]; let rows: [Float]; let output: [Float]
        }
        let url = try XCTUnwrap(Bundle.module.url(forResource: "cq-kernels", withExtension: "json", subdirectory: "Fixtures/Whistle"))
        for fixture in try JSONDecoder().decode([Fixture].self, from: Data(contentsOf: url)) {
            let matrix = WhistleCactusQuant(packed: MLXArray(fixture.packed), norms: MLXArray(fixture.norms),
                                           codebook: MLXArray(fixture.codebook), columns: 256, rows: 4, bits: fixture.bits)
            let input = MLXArray(fixture.input, [3, 256])
            let rows = MLXArray(fixture.rows, [4, 256])
            assertClose(matrix.project(input), MLXArray(fixture.output, [3, 4]), tolerance: 2e-6)
            assertClose(matrix.portableProject(input), MLXArray(fixture.output, [3, 4]), tolerance: 2e-6)
            assertClose(matrix.portableGather(MLXArray([Int32(3), 1, 3])),
                        take(rows, MLXArray([Int32(3), 1, 3]), axis: 0), tolerance: 2e-7)
            assertClose(matrix.slice(1..<3).portableProject(input),
                        MLXArray(fixture.output, [3, 4])[0..., 1..<3], tolerance: 2e-6)
            assertClose(matrix.gather(MLXArray([Int32(3), 1, 3])), take(rows, MLXArray([Int32(3), 1, 3]), axis: 0), tolerance: 2e-7)
            assertClose(matrix.slice(1..<3).project(input), MLXArray(fixture.output, [3, 4])[0..., 1..<3], tolerance: 2e-6)
        }
        }
    }

    func testLadderAndKeywordBias() {
        XCTAssertEqual(WhistleModel.layers(depth: 2), [0, 7])
        XCTAssertEqual(WhistleModel.layers(depth: 4), [0, 3, 5, 7])
        XCTAssertEqual(WhistleModel.layers(depth: 8), Array(0..<8))
        let bias = WhistleModel.keywordBias(history: [90, 12], sequences: [[12, 34], [12, 34, 56]])
        XCTAssertEqual(bias[12], 2)
        XCTAssertEqual(bias[34], 5)
        XCTAssertEqual(bias[56], 0)
    }

    private func assertClose(_ actual: MLXArray, _ expected: MLXArray, tolerance: Float) {
        let error = max(abs(actual - expected)).item(Float.self)
        XCTAssertLessThan(error, tolerance)
    }
}
