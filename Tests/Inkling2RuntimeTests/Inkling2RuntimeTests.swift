import XCTest
import MLX
import MereRunMLXTestSupport
@testable import MereRunInkling2Model

final class Inkling2RuntimeTests: MLXTestCase {
    private var fixtures: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures") }

    private struct ArrayFixture: Decodable { let shape: [Int]; let values: [Float] }
    private struct Reference: Decodable {
        let tokens: [Int32]
        let logits: [Float]
        let pixels: [Float]
        let imageFeatures: [Float]
    }

    private func tiny() throws -> (Inkling2Model, Reference, [String: MLXArray]) {
        let config = try Inkling2Configuration.decode(Data(contentsOf: fixtures.appendingPathComponent("tiny-config.json")))
        let arrays = try JSONDecoder().decode([String: ArrayFixture].self, from: Data(contentsOf: fixtures.appendingPathComponent("weights.json")))
            .mapValues { MLXArray($0.values, $0.shape) }
        let reference = try JSONDecoder().decode(Reference.self, from: Data(contentsOf: fixtures.appendingPathComponent("reference.json")))
        return (try Inkling2Model(config: config, weights: Inkling2Weights(arrays: arrays)), reference, arrays)
    }

    private func assertClose(_ actual: MLXArray, _ expected: [Float], tolerance: Float = 3e-5, file: StaticString = #filePath, line: UInt = #line) {
        let values = actual.asArray(Float.self)
        XCTAssertEqual(values.count, expected.count, file: file, line: line)
        let error = zip(values, expected).map { abs($0 - $1) }.max() ?? 0
        XCTAssertTrue(error.isFinite && error < tolerance, "maximum error \(error)", file: file, line: line)
    }

    func testPublishedConfiguration() throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures")
        let c = try Inkling2Configuration.decode(Data(contentsOf: root.appendingPathComponent("config.json")))
        XCTAssertEqual(c.textConfig.numHiddenLayers, 48)
        XCTAssertEqual(c.textConfig.localLayerIds.count, 40)
        XCTAssertEqual(c.textConfig.nRoutedExperts, 384)
        XCTAssertEqual(c.textConfig.numExpertsPerTok, 8)
    }

    func testIndependentDecoderReferenceAndCacheChunkBoundaries() throws {
        let (model, reference, _) = try tiny()
        let tokens = MLXArray(reference.tokens).reshaped(1, -1)
        let full = try model(tokens, cache: Inkling2Cache(layerCount: 4), lastPositionOnly: false)
        assertClose(full, reference.logits)
        for chunks in [[1, 1, 1, 1, 1, 1, 1], [2, 3, 2], [4, 3]] {
            let cache = Inkling2Cache(layerCount: 4)
            var outputs: [MLXArray] = [], offset = 0
            for size in chunks {
                outputs.append(try model(tokens[0..., offset..<(offset + size)], cache: cache, lastPositionOnly: false))
                offset += size
            }
            XCTAssertEqual(cache.offset, 7)
            XCTAssertEqual(cache.layers[0].keys?.dim(2), 2)
            XCTAssertEqual(cache.layers[2].keys?.dim(2), 7)
            assertClose(concatenated(outputs, axis: 1), reference.logits)
        }
    }

    func testIndependentVisionReferenceWithAsymmetricGrid() throws {
        let (model, reference, _) = try tiny()
        let pixels = MLXArray(reference.pixels).reshaped(6, 4, 4, 3)
        let output = try model.imageFeatures(pixels: pixels, rows: 2, columns: 3)
        XCTAssertEqual(output.shape, [6, 4])
        assertClose(output, reference.imageFeatures)
    }

    func testNVFP4NibblesScalesAndOuterDenominator() {
        let packed: [UInt8] = (0..<8).map { i in
            let low = UInt8(i * 2)
            let high = UInt8(i * 2 + 1) << 4
            return low | high
        }
        let bytes = MLXArray(packed).reshaped(1, 8)
        let decoded = Inkling2Math.decodeNVFP4(bytes, scales: MLXArray([UInt8(0x40)]).reshaped(1, 1), global: MLXArray(Float(0.25)))
        assertClose(decoded, [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, -0.0, -0.25, -0.5, -0.75, -1, -1.5, -2, -3])
    }

    func testMXFP8UnsignedExponentScales() {
        let bytes = MLXArray(Array(repeating: UInt8(0x38), count: 32) + Array(repeating: UInt8(0xb8), count: 32)).reshaped(1, 64)
        let result = Inkling2Math.decodeMXFP8(bytes, scales: MLXArray([UInt8(128), 126]).reshaped(1, 2))
        assertClose(result, Array(repeating: Float(2), count: 32) + Array(repeating: Float(-0.5), count: 32))
    }

    func testImagePaddingAndTransparency() throws {
        let image = try Inkling2ImagePatches.make(rgba: [255, 0, 0, 255, 0, 0, 0, 0], width: 2, height: 1, tokenSize: 4)
        XCTAssertEqual(image.rows, 1)
        XCTAssertEqual(image.columns, 1)
        let values = image.pixels.asArray(Float.self)
        XCTAssertEqual(Array(values.prefix(3)), [1, -1, -1])
        XCTAssertEqual(values[3], 0.6, accuracy: 1e-6)
        XCTAssertEqual(values[6], -1 / 127.5 - 1, accuracy: 1e-6)
    }

    func testRejectsIncompleteAndWrongShapeWeights() throws {
        let (model, _, arrays) = try tiny()
        var missing = arrays
        missing.removeValue(forKey: "model.llm.layers.2.attn.k_sconv.static_w")
        XCTAssertThrowsError(try Inkling2Model(config: model.config, weights: Inkling2Weights(arrays: missing)))
        var wrong = arrays
        wrong["model.llm.layers.1.mlp.experts.w2_weight"] = MLXArray.zeros([3, 3, 4])
        XCTAssertThrowsError(try Inkling2Model(config: model.config, weights: Inkling2Weights(arrays: wrong)))
    }

    func testCompletePublishedCheckpointSchema() throws {
        let config = try Inkling2Configuration.decode(Data(contentsOf: fixtures.appendingPathComponent("config.json")))
        let tensors = try JSONDecoder().decode([String: Inkling2Weights.Tensor].self,
            from: Data(contentsOf: fixtures.appendingPathComponent("published-tensors.json")))
        XCTAssertEqual(tensors.count, 1770)
        _ = try Inkling2Model(config: config, weights: Inkling2Weights(metadata: tensors))
    }

    func testCheckpointReaderAndExpertSlices() throws {
        let (_, _, arrays) = try tiny()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try MLX.save(arrays: arrays, url: root.appendingPathComponent("model.safetensors"))
        struct Index: Encodable { let weight_map: [String: String] }
        try JSONEncoder().encode(Index(weight_map: arrays.mapValues { _ in "model.safetensors" }))
            .write(to: root.appendingPathComponent("model.safetensors.index.json"))
        let weights = try Inkling2Weights(root: root)
        XCTAssertEqual(weights.names, Set(arrays.keys))
        let name = "model.llm.layers.0.mlp.experts.w13_weight"
        assertClose(try weights.expert(name, index: 2), arrays[name]![2].asArray(Float.self))
    }

    func testOriginalPackedArchiveDecodesSelectedExpertWithoutOtherExperts() throws {
        struct Tensor: Encodable { let dtype: String; let shape: [Int]; let data_offsets: [Int] }
        struct Index: Encodable { let weight_map: [String: String] }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        var header: [String: Tensor] = [:], payload = Data()
        func add(_ name: String, _ dtype: String, _ shape: [Int], _ bytes: Data) {
            header[name] = Tensor(dtype: dtype, shape: shape, data_offsets: [payload.count, payload.count + bytes.count])
            payload.append(bytes)
        }
        func floats(_ values: [Float]) -> Data { values.withUnsafeBufferPointer { Data(buffer: $0) } }
        let nv = "experts.nv", mx = "experts.mx"
        add(nv, "U8", [2, 1, 8], Data(Array(repeating: UInt8(0), count: 8) + Array(repeating: UInt8(0x32), count: 8)))
        add(nv + "_scale", "F8_E4M3", [2, 1, 1], Data([0x38, 0x40]))
        add(nv + "_amax", "F32", [2], floats([2688, 1344]))
        add(nv + "_outer_scale_denominator", "F32", [1], floats([2688]))
        add(mx, "F8_E4M3", [2, 1, 32], Data(Array(repeating: UInt8(0x38), count: 32) + Array(repeating: UInt8(0xb8), count: 32)))
        add(mx + "_scale", "U8", [2, 1, 1], Data([127, 128]))
        let json = try JSONEncoder().encode(header)
        var length = UInt64(json.count).littleEndian
        var file = withUnsafeBytes(of: &length) { Data($0) }
        file.append(json); file.append(payload)
        try file.write(to: root.appendingPathComponent("packed.safetensors"))
        try JSONEncoder().encode(Index(weight_map: header.mapValues { _ in "packed.safetensors" }))
            .write(to: root.appendingPathComponent("model.safetensors.index.json"))
        let weights = try Inkling2Weights(root: root)
        assertClose(try weights.expert(nv, index: 1), (0..<16).map { $0.isMultiple(of: 2) ? Float(1) : Float(1.5) })
        assertClose(try weights.expert(mx, index: 1), Array(repeating: -2, count: 32))
    }
}
