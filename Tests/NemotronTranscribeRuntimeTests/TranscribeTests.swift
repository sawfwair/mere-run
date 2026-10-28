import Foundation
import MLX
import MereRunMLXTestSupport
import XCTest
@testable import AudioNemotronTranscribeModel

final class NemotronTranscribeTests: MLXTestCase {
    func testPublishedConfigurationAndInventory() throws {
        let root = Bundle.module.url(forResource: "Fixtures", withExtension: nil)!
        let config = try NemotronTranscribeConfig.load(at: root)
        XCTAssertEqual(config.llmConfig.layersBlockType.count, 52)
        XCTAssertEqual(config.llmConfig.layersBlockType.filter { $0 == "mamba" }.count, 23)
        XCTAssertEqual(config.llmConfig.layersBlockType.filter { $0 == "moe" }.count, 23)
        XCTAssertEqual(config.llmConfig.numExpertsPerTok, 6)
        XCTAssertEqual(config.eosTokenId, [11])
        let inventory = try TranscribeWeights.inventory()
        XCTAssertEqual(inventory["llm.model.layers.1.mixer.experts.gate_and_up_projs"]?.shape, [128, 2688, 1856])
        XCTAssertEqual(inventory["llm.model.layers.1.mixer.experts.down_projs"]?.shape, [128, 1856, 2688])
        XCTAssertEqual(inventory["llm.model.layers.0.mixer._fp32_params.A_log"]?.dtype, "F32")
        XCTAssertFalse(inventory.keys.contains { $0.hasPrefix("ctc_timestamp.") })
    }

    func testArchitectureDriftIsRejected() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let fixture = Bundle.module.url(forResource: "config", withExtension: "json", subdirectory: "Fixtures")!
        let original = try String(contentsOf: fixture, encoding: .utf8)
        try original.replacingOccurrences(of: "\"hidden_size\": 2688", with: "\"hidden_size\": 2048")
            .write(to: root.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        XCTAssertThrowsError(try NemotronTranscribeConfig.load(at: root))
    }

    func testWaveformChunkingPreservesTinyTails() {
        XCTAssertEqual(NemotronTranscribeModel.audioChunks(sampleCount: 480_100), [0..<480_100])
        XCTAssertEqual(NemotronTranscribeModel.audioChunks(sampleCount: 480_320), [0..<480_000, 480_000..<480_320])
        XCTAssertEqual(NemotronTranscribeModel.audioChunks(sampleCount: 960_001), [0..<480_000, 480_000..<960_001])
        XCTAssertEqual(NemotronTranscribeModel.audioChunks(sampleCount: 0), [])
    }

    func testUnbiasedFeatureNormalizationAndSpeakerKernel() {
        let features = MLXArray([Float(1), 2, 3, 4, 5, 6]).reshaped(1, 3, 2)
        let result = NemotronTranscribeFusion.normalizeFeatures(features)
        assertClose(result, MLXArray([Float(-1), -1, 0, 0, 1, 1]).reshaped(1, 3, 2), tolerance: 1e-5)
        let kernel = NemotronTranscribeFusion.speakerKernel(speakers: 2, dimensions: 4)
        let expected: [Float] = [0, 1, 0, 1, sin(1), cos(1), sin(0.01), cos(0.01)]
        assertClose(kernel, MLXArray(expected).reshaped(2, 4), tolerance: 1e-6)
    }

    func testRouterBiasChangesSelectionButNotMixingProbabilities() {
        let input = MLXArray([Float(1)]).reshaped(1, 1, 1)
        let weight = MLXArray([Float(0), 1, 2]).reshaped(3, 1)
        let result = TranscribeLanguage.route(input, weight: weight, bias: MLXArray([Float(3), 2, -3]), topK: 2, scale: 2.5)
        let indices = result.0.reshaped(-1).asArray(Int32.self).map(Int.init)
        XCTAssertEqual(Set(indices), [0, 1])
        let probabilities = [Float(0.5), 1 / (1 + exp(Float(-1)))]
        let expected = indices.map { probabilities[$0] / probabilities.reduce(0, +) * 2.5 }
        assertClose(result.1.reshaped(-1), MLXArray(expected), tolerance: 1e-6)
    }

    func testEncoderAgainstIndependentScalarReference() throws {
        let fixture = try reference()
        let encoder = TranscribeEncoder(weights: fixture.weights, prefix: "encoder", width: 8, heads: 2,
                                        layers: 1, rotaryDimensions: 2, qkNorm: true)
        let result = try encoder(fixture.input.array)
        assertClose(result, fixture.output.array, tolerance: 2e-5)
    }

    func testHybridPrefillMatchesIndependentReferenceAndCachedDecode() throws {
        let fixture = try reference()
        let language = TranscribeLanguage(weights: fixture.weights, config: fixture.config)
        let input = MLXArray([Int32(1), 2, 3]).reshaped(1, 3)
        let fullCache = NemotronTranscribeCache(count: 3)
        let full = try language(embeddings: language.inputEmbeddings(input), cache: fullCache)
        assertClose(full, fixture.logits.array, tolerance: 3e-5)
        let cache = NemotronTranscribeCache(count: 3)
        let prefix = try language(embeddings: language.inputEmbeddings(input[0..., ..<2]), cache: cache)
        eval(prefix)
        let decoded = try language(embeddings: language.inputEmbeddings(input[0..., 2...]), cache: cache)
        assertClose(decoded, full, tolerance: 3e-5)
        XCTAssertEqual(cache.offset, 3)
        XCTAssertEqual(cache.recurrent[0]?.dtype, .float32)
    }

    func testBF16HybridPrefillAndCachedDecode() throws {
        let fixture = try reference()
        let arrays = Dictionary(uniqueKeysWithValues: fixture.weights.arrays.map { key, value in
            (key, value.asType(key.contains("._fp32_params.") ? .float32 : .bfloat16))
        })
        let language = TranscribeLanguage(weights: TranscribeWeights(arrays: arrays), config: fixture.config)
        let input = MLXArray([Int32(1), 2, 3]).reshaped(1, 3)
        let full = try language(embeddings: language.inputEmbeddings(input), cache: NemotronTranscribeCache(count: 3))
        eval(full)
        XCTAssertEqual(full.dtype, .bfloat16)
        XCTAssertTrue(full.asType(.float32).asArray(Float.self).allSatisfy(\.isFinite))
        let cache = NemotronTranscribeCache(count: 3)
        let prefix = try language(embeddings: language.inputEmbeddings(input[0..., ..<2]), cache: cache)
        eval(prefix)
        let decoded = try language(embeddings: language.inputEmbeddings(input[0..., 2...]), cache: cache)
        assertClose(decoded.asType(.float32), full.asType(.float32), tolerance: 0.01)
        XCTAssertEqual(cache.offset, 3)
        XCTAssertEqual(cache.recurrent[0]?.dtype, .float32)
    }

    func testMambaMetalScanMatchesPortableRecurrence() {
        for dtype in [DType.float32, .bfloat16] {
            let x = (sin(MLXArray(0..<48).asType(.float32)) * Float(0.1)).reshaped(1, 3, 4, 4).asType(dtype)
            let b = (cos(MLXArray(0..<768).asType(.float32)) * Float(0.1)).reshaped(1, 3, 2, 128).asType(dtype)
            let c = (sin(MLXArray(0..<768).asType(.float32)) * Float(0.1)).reshaped(1, 3, 2, 128).asType(dtype)
            let dt = MLXArray.zeros([1, 3, 4], dtype: dtype)
            let aLog = MLXArray.zeros([4])
            let skip = MLXArray.ones([4])
            let bias = MLXArray([-1.0, -0.5, 0.0, 0.5] as [Float])
            let expected = transcribeSSMUpdate(x: x, b: b, c: c, dt: dt, aLog: aLog, d: skip,
                dtBias: bias, state: nil, minimumTimeStep: 0, maximumTimeStep: .greatestFiniteMagnitude, useMetalKernel: false)
            let actual = transcribeSSMUpdate(x: x, b: b, c: c, dt: dt, aLog: aLog, d: skip,
                dtBias: bias, state: nil, minimumTimeStep: 0, maximumTimeStep: .greatestFiniteMagnitude)
            assertClose(actual.0.asType(.float32), expected.0.asType(.float32), tolerance: 0.001)
            assertClose(actual.1, expected.1, tolerance: 1e-5)
            XCTAssertEqual(actual.1.dtype, .float32)
        }
    }

    private func assertClose(_ actual: MLXArray, _ expected: MLXArray, tolerance: Float,
                             file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(actual.shape, expected.shape, file: file, line: line)
        XCTAssertLessThanOrEqual(MLX.max(abs(actual - expected)).item(Float.self), tolerance, file: file, line: line)
    }

    private struct Tensor: Decodable {
        let shape: [Int]
        let values: [Float]
        var array: MLXArray { MLXArray(values, shape) }
    }
    private struct Reference: Decodable {
        let tensors: [String: Tensor]
        let input: Tensor
        let output: Tensor
        let logits: Tensor
        let config: NemotronTranscribeConfig.Language
        var weights: TranscribeWeights { TranscribeWeights(arrays: tensors.mapValues(\.array)) }
    }
    private func reference() throws -> Reference {
        let url = Bundle.module.url(forResource: "reference", withExtension: "json", subdirectory: "Fixtures")!
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(Reference.self, from: Data(contentsOf: url))
    }
}
