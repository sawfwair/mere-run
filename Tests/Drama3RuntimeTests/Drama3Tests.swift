import Foundation
import MLX
import MLXNN
import MereRunMLXTestSupport
import XCTest
@testable import AudioDrama3Model

final class Drama3Tests: MLXTestCase {
    private var fixtures: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures") }
    private func config(_ file: String = "tiny-config.json") throws -> Drama3Config {
        try JSONDecoder().decode(Drama3Config.self, from: Data(contentsOf: fixtures.appendingPathComponent(file)))
    }

    func testFP8DecodeKnownValues() {
        XCTAssertEqual(Drama3TensorArchive.decodeE4M3(0x38), 1)
        XCTAssertEqual(Drama3TensorArchive.decodeE4M3(0x7e), 448)
        XCTAssertEqual(Drama3TensorArchive.decodeE4M3(0x01), 1.0 / 512)
        XCTAssertEqual(Drama3TensorArchive.decodeE4M3(0xb8), -1)
        XCTAssertTrue(Drama3TensorArchive.decodeE4M3(0x7f).isNaN)
    }

    func testTensorReaderSeeksAndReportsTruncatedPayload() throws {
        let file = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".safetensors")
        defer { try? FileManager.default.removeItem(at: file) }
        try MLX.save(arrays: ["first": MLXArray([Float(1), 2]), "second": MLXArray([Float(3), 4])], url: file)
        let reader = try Drama3TensorArchive(url: file)
        XCTAssertEqual(try reader.read("second").asArray(Float.self), [3, 4])
        XCTAssertEqual(try reader.read("first").asArray(Float.self), [1, 2])
        let writer = try FileHandle(forWritingTo: file)
        try writer.truncate(atOffset: 8)
        try writer.close()
        XCTAssertThrowsError(try reader.read("first")) { error in
            XCTAssertTrue(error.localizedDescription.contains("truncated tensor data"))
        }
    }

    func testPublishedNamespacesExactlyCoverNonExpertRuntimeParameters() throws {
        struct Shape: Decodable { let shape: [Int] }
        let c = try config("config.json")
        XCTAssertEqual(c.text.numHiddenLayers, 40)
        XCTAssertEqual(c.audio.codebooks, 10)
        let model = try Drama3Model(config: c)
        let parameters = Dictionary(uniqueKeysWithValues: model.parameters().flattened().filter {
            !$0.0.contains(".switch_mlp.") && !$0.0.hasSuffix(".qkNormWeightBF16")
        }.map { ($0.0, $0.1.shape) })
        let published = try JSONDecoder().decode([String: Shape].self,
            from: Data(contentsOf: fixtures.appendingPathComponent("nonexpert-shapes.json")))
        var mapped: [String: [Int]] = [:]
        for (name, shape) in published {
            let key = try XCTUnwrap(Drama3Model.mappedKey(name), name)
            let expected = key.hasSuffix(".conv1d.weight") ? [shape.shape[0], shape.shape[2], shape.shape[1]] : shape.shape
            XCTAssertNil(mapped.updateValue(expected, forKey: key), "duplicate \(key)")
        }
        XCTAssertEqual(parameters, mapped)
        let packed = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        XCTAssertEqual(packed["backbone.layers.0.mlp.switch_mlp.gate_proj.weight"]?.shape, [256, 512, 256])
        XCTAssertEqual(packed["backbone.layers.0.mlp.switch_mlp.down_proj.scales"]?.shape, [256, 2_048, 8])
    }

    func testPromptSegmentsPreserveReferenceAndGenerationBoundaries() throws {
        let c = try config()
        var encoded: [String] = []
        let codes = (0..<10).map { [Int32($0), Int32($0 + 10)] }
        let rows = try Drama3Prompt.rows(text: "Hello [laughs]", references: [.init(transcript: "  <|speaker:2|> Welcome", codes: codes)], config: c) {
            encoded.append($0)
            return [encoded.count]
        }
        XCTAssertEqual(encoded.first, "<|im_start|>system\n")
        XCTAssertEqual(encoded[1], "convert the provided text to speech reference to the following:")
        XCTAssertEqual(encoded[2], "\n\n[Reference 1]\nText: <|speaker:2|>   Welcome\nSpeech: ")
        XCTAssertEqual(encoded.last, "<|im_start|>assistant\n<|voice|>")
        XCTAssertTrue(encoded.contains("<|speaker:0|> Hello [laughs]"))
        XCTAssertEqual(rows[3], [Int32(c.semanticStart)] + codes.map { $0[0] })
        XCTAssertEqual(rows[4], [Int32(c.semanticStart + 10)] + codes.map { $0[1] })
    }

    func testPromptRejectsMalformedSpeakersAndStructuralTokens() {
        for text in [" ", "<|speaker:16|> Hi", "<|speaker:01|> Hi", "<|speaker|> Hi", "Hi <|im_end|>", "<|semantic:2|>"] {
            XCTAssertThrowsError(try Drama3Prompt.validate(text), text)
        }
        XCTAssertNoThrow(try Drama3Prompt.validate("<|speaker:15|> [sighs] Fine."))
    }

    private struct TowerReference: Decodable {
        let hidden: [Float]
        let extra: [Float]
        let codes: [Int32]
        let semantic: [Float]
        let output: [Float]
    }

    private func towerLayer() throws -> Drama3TowerLayer {
        let layer = Drama3TowerLayer(try config().audio)
        let parameters = layer.parameters().flattened().map { name, value in
            let seed = ("layer." + name).utf8.reduce(0) { $0 + Int($1) } % 97
            let numbers = (0..<value.size).map { Float(sin(Double($0 + seed) * 0.17) * 0.03) }
            return (name, MLXArray(numbers, value.shape))
        }
        try layer.update(parameters: ModuleParameters.unflattened(parameters), verify: .all)
        return layer
    }

    func testTowerMatchesIndependentNumPyAndKeepsTemporalCache() throws {
        let reference = try JSONDecoder().decode(TowerReference.self,
            from: Data(contentsOf: fixtures.appendingPathComponent("tower-reference.json")))
        let layer = try towerLayer()
        let h = MLXArray(reference.hidden).reshaped(1, 3, 16)
        let code = MLXArray(reference.codes).reshaped(1, 3)
        let semantic = MLXArray(reference.semantic).reshaped(1, 3)
        let extra = MLXArray(reference.extra).reshaped(1, 3, 16)
        let cache = Drama3Tower(try config()).makeCache()[0]
        let all = layer(h, code: code, semantic: semantic, extra: extra, cache: cache)
        let actual = all.reshaped(-1).asArray(Float.self)
        for (value, expected) in zip(actual, reference.output) { XCTAssertEqual(value, expected, accuracy: 3e-5) }
        XCTAssertEqual(cache.offset, 3)
        let serialCache = Drama3Tower(try config()).makeCache()[0]
        var serial: [Float] = []
        for row in 0..<3 {
            let result = layer(h[0..., row..<(row + 1), 0...], code: code[0..., row..<(row + 1)],
                semantic: semantic[0..., row..<(row + 1)], extra: extra[0..., row..<(row + 1), 0...], cache: serialCache)
            serial += result.reshaped(-1).asArray(Float.self)
        }
        XCTAssertEqual(serialCache.offset, 3)
        for (value, expected) in zip(serial, reference.output) { XCTAssertEqual(value, expected, accuracy: 3e-5) }
    }

    func testFloatMaskPreservesBF16TowerComputation() throws {
        let layer = try towerLayer()
        try layer.update(parameters: ModuleParameters.unflattened(layer.parameters().flattened().map {
            ($0.0, $0.1.asType(.bfloat16))
        }), verify: .all)
        let hidden = MLXArray.ones([1, 1, 16], dtype: .bfloat16)
        let codes = MLXArray([Int32(3)]).reshaped(1, 1)
        let floatMask = MLXArray.ones([1, 1], dtype: .float32)
        let cache = Drama3Tower(try config()).makeCache()[0]
        let output = layer(hidden, code: codes, semantic: floatMask, cache: cache)
        XCTAssertEqual(output.dtype, .bfloat16)
        XCTAssertTrue(output.asArray(Float.self).allSatisfy(\.isFinite))
        let referenceCache = Drama3Tower(try config()).makeCache()[0]
        let reference = layer(hidden, code: codes, semantic: floatMask.asType(.bfloat16), cache: referenceCache)
        XCTAssertEqual(output.asArray(Float.self), reference.asArray(Float.self))
    }

    func testBF16PromptStaysBF16ThroughBothBackboneAttentionFamilies() throws {
        let data = try String(contentsOf: fixtures.appendingPathComponent("tiny-config.json"), encoding: .utf8)
            .replacingOccurrences(of: "\"hidden_size\": 16", with: "\"hidden_size\": 64")
            .replacingOccurrences(of: "\"dim\": 16", with: "\"dim\": 64")
        for attention in ["full_attention", "linear_attention"] {
            let source = data.replacingOccurrences(of: "\"full_attention\"", with: "\"\(attention)\"")
            let config = try JSONDecoder().decode(Drama3Config.self, from: Data(source.utf8))
            let model = try Drama3Model(config: config)
            try model.update(parameters: ModuleParameters.unflattened(model.parameters().flattened().map {
                ($0.0, $0.1.dtype == .uint32 ? $0.1 : $0.1.asType(.bfloat16))
            }), verify: .all)
            let frames = MLXArray(Array(repeating: [Int32(config.semanticStart)] + Array(repeating: Int32(1), count: 10), count: 8)
                .flatMap { $0 }).reshaped(1, 8, 11)
            let embeddings = model.backbone.embeddings(frames[.ellipsis, 0])
            let input = model.tower.backboneInput(frames, text: embeddings)
            XCTAssertEqual(input.dtype, .bfloat16, attention)
            let output = model.backbone(input, cache: model.backbone.makeCache())
            MLX.eval(output)
            XCTAssertEqual(output.dtype, .bfloat16, attention)
        }
    }

    func testCausalCodecConvolutionAndWeightNormalization() throws {
        let codec = try Drama3Codec(weights: [
            "test.weight_v": MLXArray([Float(3), 4]).reshaped(1, 1, 2),
            "test.weight_g": MLXArray([Float(10)]).reshaped(1, 1, 1),
            "test.bias": MLXArray([Float(1)]),
            "transpose.weight": MLXArray([Float(1), 2, 3, 4]).reshaped(1, 1, 4)
        ])
        // Folded kernel [6,8], one left zero: [1*8+1, 1*6+2*8+1, 2*6+3*8+1].
        let input = MLXArray([Float(1), 2, 3]).reshaped(1, 3, 1)
        XCTAssertEqual(try codec.conv(input, "test").reshaped(-1).asArray(Float.self), [9, 23, 37])
        // Stride two transpose then causal right crop of two samples.
        XCTAssertEqual(try codec.conv(MLXArray([Float(1), 2]).reshaped(1, 2, 1), "transpose",
                                      stride: 2, transpose: true).reshaped(-1).asArray(Float.self), [1, 2, 5, 8])
    }

    func testFP8BlockScaleAndAffinePackingAgainstIndependentDequantization() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        let values = Array(repeating: UInt8(0x38), count: 128 * 128) + Array(repeating: UInt8(0x40), count: 128 * 128)
        struct Tensor: Encodable { let dtype: String; let shape: [Int]; let data_offsets: [Int] }
        let header = try JSONEncoder().encode([
            "expert.weight": Tensor(dtype: "F8_E4M3", shape: [256, 128], data_offsets: [0, values.count]),
            "expert.weight_scale_inv": Tensor(dtype: "F32", shape: [2, 1], data_offsets: [values.count, values.count + 8])
        ])
        var length = UInt64(header.count).littleEndian
        var data = withUnsafeBytes(of: &length) { Data($0) }; data.append(header); data.append(contentsOf: values)
        let scales = [Float(3), 5]; data.append(scales.withUnsafeBytes { Data($0) })
        try data.write(to: url)
        let archive = try Drama3TensorArchive(url: url)
        let decoded = try archive.decodedWeight("expert.weight")
        XCTAssertEqual(decoded[0, 0].item(Float.self), 3)
        XCTAssertEqual(decoded[128, 0].item(Float.self), 10)
        // A varying matrix exercises affine offsets rather than just constant groups.
        let dense = MLXArray((0..<128).map { Float($0) / 64 - 1 }).reshaped(1, 128)
        let q = MLX.quantized(dense, groupSize: 64, bits: 4)
        let unpacked = MLX.dequantized(q.wq, scales: q.scales, biases: q.biases, groupSize: 64, bits: 4)
        let x = MLXArray.ones([1, 128])
        let direct = MLX.matmul(x, unpacked.transposed())
        let packed = MLX.quantizedMM(x, q.wq, scales: q.scales, biases: q.biases, transpose: true, groupSize: 64, bits: 4)
        XCTAssertEqual(packed.item(Float.self), direct.item(Float.self), accuracy: 1e-4)
    }
}
