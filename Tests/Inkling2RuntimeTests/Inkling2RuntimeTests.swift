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
    private struct MiniReference: Decodable {
        let logits: [Float]
        let weights: [String: ArrayFixture]
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

    func testMiniReferenceAcrossLoopPassesAndCacheChunkBoundaries() throws {
        let (_, baseReference, baseWeights) = try tiny()
        let config = try Inkling2Configuration.decode(Data(contentsOf: fixtures.appendingPathComponent("tiny-mini-config.json")))
        let reference = try JSONDecoder().decode(MiniReference.self, from: Data(contentsOf: fixtures.appendingPathComponent("mini-reference.json")))
        let decoderWeights = baseWeights.filter { $0.key.hasPrefix("model.llm.") }
            .merging(reference.weights.mapValues { MLXArray($0.values, $0.shape) }) { _, new in new }
        let model = try Inkling2Model(config: config, weights: Inkling2Weights(arrays: decoderWeights))
        let tokens = MLXArray(baseReference.tokens).reshaped(1, -1)
        let fullAttentionSlots: Set<Int> = [2, 5, 7, 8, 11, 12, 15]
        for chunks in [[7], [1, 1, 1, 1, 1, 1, 1], [2, 3, 2]] {
            let cache = model.makeCache()
            var outputs: [MLXArray] = [], offset = 0
            for size in chunks {
                outputs.append(try model(tokens[0..., offset..<(offset + size)], cache: cache, lastPositionOnly: false))
                offset += size
            }
            assertClose(concatenated(outputs, axis: 1), reference.logits)
            XCTAssertEqual(cache.layers.count, 16)
            for (index, layer) in cache.layers.enumerated() {
                XCTAssertEqual(layer.offset, 7)
                XCTAssertEqual(layer.keys?.dim(2), fullAttentionSlots.contains(index) ? 7 : 2)
                XCTAssertEqual(layer.convolutions.count, 4)
            }
        }
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

    func testPackedGatherMatchesOriginalExpertMath() throws {
        #if os(macOS)
        try XCTSkipUnless(Device.defaultDevice().deviceType == .gpu, "Packed inference uses Metal.")
        let tokens = 3, width = 128, outputs = 64
        let input = MLXArray((0..<(tokens * width)).map { Float($0 % 15 - 7) / 64 }).reshaped(tokens, 1, 1, width)
        let indices = MLXArray([Int32(1), 0, 1, 1, 0, 1]).reshaped(tokens, 2)
        for nvfp4 in [true, false] {
            let packedWidth = nvfp4 ? width / 2 : width
            let groupCount = width / (nvfp4 ? 16 : 32)
            let codes: [UInt8] = nvfp4 ? [0x32, 0xb6, 0xa0, 0xf2] : [0x38, 0xb8, 0x40, 0x30]
            let data: [UInt8] = (0..<(2 * outputs * packedWidth)).map {
                codes[($0 + $0 / packedWidth) % codes.count]
            }
            let bytes = MLXArray(data).reshaped(2, outputs, packedWidth)
            let scaleCount = 2 * outputs * groupCount
            let scaleBytes = (0..<scaleCount).map { position -> UInt8 in
                if nvfp4 {
                    return position.isMultiple(of: 2) ? 0x38 : 0x40
                }
                return UInt8(126 + position % 3)
            }
            let scales = MLXArray(scaleBytes).reshaped(2, outputs, groupCount)
            let global = MLXArray([Float(0.125), 0.375])
            let projection = Inkling2PackedExperts.Projection(
                packed: bytes.view(dtype: .uint32), scales: scales, expertScales: nvfp4 ? global : nil
            )
            let dense = nvfp4
                ? Inkling2Math.decodeNVFP4(bytes, scales: scales, global: global.reshaped(2, 1, 1, 1))
                : Inkling2Math.decodeMXFP8(bytes, scales: scales)
            let expected = matmul(input, take(dense, indices, axis: 0).swappedAxes(-1, -2))
            assertClose(projection(input, indices: indices), expected.asArray(Float.self), tolerance: 1e-4)
        }
        #else
        throw XCTSkip("Packed inference uses Metal.")
        #endif
    }

    func testMetalConvolutionMatchesStaticAndDynamicTapOrder() throws {
        #if os(macOS)
        try XCTSkipUnless(Device.defaultDevice().deviceType == .gpu, "Fused convolution uses Metal.")
        let channels = 4, sequence = 2, batch = 2, taps = 3, heads = 2
        let samples = (0..<(batch * (sequence + taps - 1) * channels)).map { Float($0 - 16) / 8 }
        let coefficients = (0..<(channels * taps)).map { Float($0 - 5) / 8 }
        let projections = (0..<(batch * sequence * heads * taps)).map { Float(7 - $0) / 16 }
        let context = MLXArray(samples).reshaped(batch, sequence + taps - 1, channels)
        let input = context[0..., (taps - 1)..., 0...]
        for dynamic in [false, true] {
            let weight = MLXArray(coefficients).reshaped(dynamic ? [taps, channels] : [channels, 1, taps])
            let projected = dynamic ? MLXArray(projections).reshaped(batch, sequence, heads * taps) : nil
            let expected = (0..<input.size).map { element in
                let channel = element % channels
                let token = element / channels
                let first = element + token / sequence * (taps - 1) * channels
                let current = first + (taps - 1) * channels
                var sum = samples[current]
                for tap in 0..<taps {
                    if dynamic {
                        let head = channel / (channels / heads)
                        let offset = (token * heads + head) * taps + tap
                        sum += samples[current - tap * channels] * (coefficients[tap * channels + channel] + projections[offset])
                    } else {
                        sum += samples[first + tap * channels] * coefficients[channel * taps + tap]
                    }
                }
                return sum / sqrt(Float((dynamic ? 2 : 1) * taps + 1))
            }
            let result = Inkling2ShortConvolution.apply(input, context: context, taps: weight,
                                                       kernelSize: taps, dynamic: projected, heads: heads)
            assertClose(result, expected)
        }
        #else
        throw XCTSkip("Fused convolution uses Metal.")
        #endif
    }

    func testMetalDecodeAttentionMatchesEagerGroupedAttention() throws {
        #if os(macOS)
        try XCTSkipUnless(Device.defaultDevice().deviceType == .gpu, "Fused attention uses Metal.")
        let batch = 2, heads = 6, kvHeads = 2, width = 64, relativeWidth = 8, extent = 16
        func array(_ shape: [Int], pattern: Int) -> MLXArray {
            let count = shape.reduce(1, *)
            return MLXArray((0..<count).map { Float(($0 * pattern) % 53 - 25) / 64 })
                .reshaped(shape).asType(.bfloat16)
        }
        let query = array([batch, 1, heads, width], pattern: 3)
        let relative = array([batch, 1, heads, relativeWidth], pattern: 5)
        let profiles = array([relativeWidth, extent], pattern: 7)
        let sinks = array([batch, 1, heads], pattern: 11)
        let gate = array([batch, 1, heads * width], pattern: 13)
        for length in [1, 13, 77, Inkling2DecodeAttention.maxKeys] {
            let keys = array([batch, kvHeads, length, width], pattern: 17)
            let values = array([batch, kvHeads, length, width], pattern: 19)
            let keyHeads = repeated(keys, count: heads / kvHeads, axis: 1).asType(.float32)
            let valueHeads = repeated(values, count: heads / kvHeads, axis: 1)
            let distances = MLXArray((0..<length).reversed().map(Int32.init))
            let rel = matmul(relative.asType(.float32), profiles.asType(.float32)).transposed(0, 2, 1, 3)
            let bias = MLX.where(distances .< extent, rel[.ellipsis, minimum(distances, MLXArray(Int32(extent - 1)))],
                                 MLXArray(Float(0)))
            let queryHeads = query.transposed(0, 2, 1, 3).asType(.float32)
            let scores = matmul(queryHeads, keyHeads.swappedAxes(-1, -2)) / sqrt(Float(width)) + bias
            let sinkLogits = sinks.transposed(0, 2, 1).expandedDimensions(axis: -1).asType(.float32)
            let probability = softmax(concatenated([scores, sinkLogits], axis: -1), axis: -1)[.ellipsis, ..<length]
            let attended = matmul(probability.asType(.bfloat16), valueHeads)
                .transposed(0, 2, 1, 3).reshaped(batch, 1, heads * width)
            let expected = (attended.asType(.float32) * sigmoid(gate.asType(.float32))).asType(.bfloat16)
            let fused = Inkling2DecodeAttention.apply(query: query, keys: keys, values: values, relative: relative,
                                                      profiles: profiles, sinks: sinks, gate: gate)
            assertClose(fused.asType(.float32), expected.asType(.float32).asArray(Float.self), tolerance: 0.003)
        }
        #else
        throw XCTSkip("Fused attention uses Metal.")
        #endif
    }

    func testMetalDecodeResidualMatchesEagerDepthAttention() throws {
        #if os(macOS)
        try XCTSkipUnless(Device.defaultDevice().deviceType == .gpu, "Fused depth attention uses Metal.")
        let width = 2_304
        func values(rows: Int, step: Int) -> MLXArray {
            MLXArray((0..<(rows * width)).map { Float(($0 * step) % 109 - 53) / 64 }).reshaped(rows, width)
        }
        let partial = values(rows: 6, step: 5).reshaped(2, 3, width)
        let blocks = [7, 9, 13, 17].map { values(rows: 6, step: $0).reshaped(2, 3, width).asType(.bfloat16) }
        for blockCount in [0, 2, 4] {
            let saved = Array(blocks.prefix(blockCount))
            for queryCount in [1, 3, 4] {
                let queries = values(rows: queryCount, step: 23)
                let expected = Inkling2Math.residual(saved, partial: partial, query: queries)
                let actual = Inkling2Math.residual(saved, partial: partial, query: queries, fusedDecode: true)
                XCTAssertEqual(actual.count, queryCount)
                for (left, right) in zip(actual, expected) {
                    assertClose(left, right.asArray(Float.self), tolerance: 1e-4)
                }
            }
        }
        #else
        throw XCTSkip("Fused depth attention uses Metal.")
        #endif
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

    func testCompletePublishedMiniCheckpointSchema() throws {
        let data = try Data(contentsOf: fixtures.appendingPathComponent("mini-config.json"))
        let config = try Inkling2Configuration.decode(data)
        XCTAssertEqual(config.textConfig.numHiddenLayers, 24)
        XCTAssertEqual(config.textConfig.loopConfig?.layers, 6..<18)
        XCTAssertEqual(config.textConfig.cacheLayerCount, 48)
        let draft = try Inkling2DSparkConfiguration.decode(data)
        try draft.validate(against: config.textConfig)
        XCTAssertEqual(draft.targetLayerIds, [5, 23, 41, 48])
        XCTAssertEqual(draft.blockSize, 8)
        var tensors = try JSONDecoder().decode([String: Inkling2Weights.Tensor].self,
            from: Data(contentsOf: fixtures.appendingPathComponent("mini-published-tensors.json")))
        XCTAssertEqual(tensors.count, 1244)
        _ = try Inkling2Model(config: config, weights: Inkling2Weights(metadata: tensors))
        tensors.removeValue(forKey: "model.llm.layers.6.controllers.v.gain_weight")
        XCTAssertThrowsError(try Inkling2Model(config: config, weights: Inkling2Weights(metadata: tensors)))
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
