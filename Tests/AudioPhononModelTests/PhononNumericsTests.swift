import Foundation
import MLX
import MereRunMLXTestSupport
import XCTest
@testable import AudioPhononModel

final class PhononNumericsTests: MLXTestCase {
    func testPublishedConfigurationAndTokenizerAgainstHuggingFaceOracle() throws {
        let directory = try XCTUnwrap(Bundle.module.resourceURL).appendingPathComponent("Fixtures")
        let config = try PhononConfig.load(from: directory.appendingPathComponent("config.json"))
        XCTAssertEqual(config.sampleRate, 24000)
        XCTAssertEqual(config.flowLm.ldim, 32)
        XCTAssertEqual(config.mimi.ratios, [6, 5, 4])
        let tokenizer = try PhononTokenizer(url: directory.appendingPathComponent("tokenizer.json"))
        XCTAssertEqual(tokenizer.vocabularySize, config.flowLm.nBins)
        struct Oracle: Decodable { let text: String; let ids: [Int] }
        let cases = try JSONDecoder().decode([Oracle].self, from: Data(contentsOf: directory.appendingPathComponent("tokenizer-oracle.json")))
        for item in cases { XCTAssertEqual(try tokenizer.encode(item.text), item.ids, item.text) }
        XCTAssertEqual(PhononTokenizer.prepare(" hello\r\nworld ").text, "Hello world.")
        XCTAssertEqual(PhononTokenizer.prepare("one two three four").tailFrames, 3)
        XCTAssertEqual(PhononTokenizer.prepare("one two three four five").tailFrames, 1)
    }

    func testTextPlanningAgainstUpstreamOracle() throws {
        struct Chunk: Decodable { let text: String; let tokens: [Int]; let tailFrames: Int }
        struct Case: Decodable { let input: String; let chunks: [Chunk] }
        struct Oracle: Decodable { let cases: [Case] }
        let directory = try XCTUnwrap(Bundle.module.resourceURL).appendingPathComponent("Fixtures")
        let tokenizer = try PhononTokenizer(url: directory.appendingPathComponent("tokenizer.json"))
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let oracle = try decoder.decode(Oracle.self, from: Data(contentsOf: directory.appendingPathComponent("planner-oracle.json")))
        for item in oracle.cases {
            let actual = try tokenizer.chunks(item.input)
            XCTAssertEqual(actual.map(\.text), item.chunks.map(\.text), item.input)
            XCTAssertEqual(actual.map(\.tokens), item.chunks.map(\.tokens), item.input)
            XCTAssertEqual(actual.map(\.tailFrames), item.chunks.map(\.tailFrames), item.input)
            XCTAssertTrue(actual.allSatisfy { $0.tokens.count <= 200 })
        }
        for text in ["Hello world.", "We stopped at the café.", "Emoji 🐈 and CJK 漢字."] {
            XCTAssertEqual(try tokenizer.decode(tokenizer.encode(text)), text)
        }
        XCTAssertThrowsError(try tokenizer.decode([-1]))
        XCTAssertThrowsError(try tokenizer.decode([tokenizer.vocabularySize]))
    }

    func testCausalConvolutionStreamingMatchesScalarOracle() throws {
        let weights: [Float] = [1, -2, 3, 2, 1, -1]
        let conv = try PhononConv(PhononWeights(arrays: [
            "c.weight": MLXArray(weights, [2, 1, 3]), "c.bias": MLXArray([Float(0.3), -0.2])
        ]), "c", input: 1, output: 2, kernel: 3)
        var state = conv.makeState()
        let input: [Float] = [1, 2, -3, 4, 5]
        let first = conv(MLXArray(Array(input[0..<2]), [1, 2, 1]), state: &state)
        let second = conv(MLXArray(Array(input[2...]), [1, 3, 1]), state: &state)
        let result = concatenated([first, second], axis: 1).asArray(Float.self)
        var expected: [Float] = []
        for time in input.indices {
            for channel in 0..<2 {
                var value: Float = channel == 0 ? 0.3 : -0.2
                for tap in 0..<3 where time + tap >= 2 {
                    value += input[time + tap - 2] * weights[channel * 3 + tap]
                }
                expected.append(value)
            }
        }
        assertClose(result, expected)
    }

    func testTransposedConvolutionOverlapSubtractsBiasAndPreservesChannels() throws {
        for groups in [1, 2] {
            let inputs = 2, outputs = 2, kernel = 4, stride = 2, frames = 3
            let weights = (0..<(inputs * outputs / groups * kernel)).map { Float($0 + 1) * 0.1 }
            let bias: [Float] = [0.3, -0.2]
            let conv = try PhononConv(PhononWeights(arrays: [
                "c.weight": MLXArray(weights, [inputs, outputs / groups, kernel]), "c.bias": MLXArray(bias)
            ]), "c", input: inputs, output: outputs, kernel: kernel, stride: stride, groups: groups, transposed: true)
            let data: [Float] = [1, 2, -1, 3, 2, -2]
            var state = conv.makeState()
            let chunks = (0..<frames).map { frame in
                conv(MLXArray(Array(data[(frame * inputs)..<((frame + 1) * inputs)]), [1, 1, inputs]), state: &state)
            }
            let result = concatenated(chunks, axis: 1).asArray(Float.self)
            var expected = [Float](repeating: 0, count: (frames * stride + kernel - stride) * outputs)
            for time in 0..<frames {
                for channel in 0..<inputs {
                    for out in 0..<(outputs / groups) {
                        let output = groups == 1 ? out : channel
                        for tap in 0..<kernel {
                            let weightIndex = (channel * (outputs / groups) + out) * kernel + tap
                            expected[(time * stride + tap) * outputs + output] += data[time * inputs + channel] * weights[weightIndex]
                        }
                    }
                }
            }
            expected = Array(expected.prefix(frames * stride * outputs))
            for index in expected.indices { expected[index] += bias[index % outputs] }
            assertClose(result, expected)
        }
    }

    func testTransformerFullAndIncrementalCausalityAndWindow() throws {
        let prefix = "t.layers.0", width = 4
        var tensors: [String: MLXArray] = [:]
        for name in ["norm1", "norm2"] {
            tensors[prefix + ".\(name).weight"] = MLXArray.ones([width])
            tensors[prefix + ".\(name).bias"] = MLXArray.zeros([width])
        }
        for (name, output, input) in [("self_attn.in_proj", 12, 4), ("self_attn.out_proj", 4, 4), ("linear1", 8, 4), ("linear2", 4, 8)] {
            tensors[prefix + ".\(name).weight"] = MLXArray((0..<(output * input)).map { sin(Float($0 + 1)) * 0.2 }, [output, input])
        }
        for name in ["layer_scale_1", "layer_scale_2"] { tensors[prefix + ".\(name).scale"] = MLXArray.ones([width]) }
        let transformer = try PhononTransformer(PhononWeights(arrays: tensors), "t", width: width, heads: 2,
                                                count: 1, feedforward: 8, period: 10000, context: 3)
        let input = MLXArray((0..<24).map { cos(Float($0)) }, [1, 6, 4])
        var fullState = transformer.makeState(), incrementalState = transformer.makeState()
        let full = transformer(input, state: &fullState)
        let incremental = (0..<6).map { index in transformer(input[0..., index..<(index + 1), 0...], state: &incrementalState) }
        assertClose(full.asArray(Float.self), concatenated(incremental, axis: 1).asArray(Float.self), tolerance: 1e-5)
        XCTAssertEqual(incrementalState[0].keys?.dim(2), 3)
        XCTAssertEqual(incrementalState[0].offset, 6)
        // Mutating future inputs must not change the first three causal outputs.
        var changedState = transformer.makeState()
        let changed = concatenated([input[0..., ..<3, 0...], input[0..., 3..., 0...] * 3], axis: 1)
        let output = transformer(changed, state: &changedState)
        assertClose(full[0..., ..<3, 0...].asArray(Float.self), output[0..., ..<3, 0...].asArray(Float.self), tolerance: 1e-5)
    }

    func testLongPrefixAndShortCachedChunksMatchFullAttention() throws {
        let width = 128, count = 2, length = 137
        var tensors: [String: MLXArray] = [:]
        for index in 0..<count {
            let prefix = "t.layers.\(index)"
            for name in ["norm1", "norm2"] {
                tensors[prefix + ".\(name).weight"] = MLXArray.ones([width])
                tensors[prefix + ".\(name).bias"] = MLXArray.zeros([width])
            }
            for (name, output, input) in [
                ("self_attn.in_proj", 3 * width, width), ("self_attn.out_proj", width, width),
                ("linear1", 2 * width, width), ("linear2", width, 2 * width)
            ] {
                tensors[prefix + ".\(name).weight"] = MLXArray(
                    (0..<(output * input)).map { sin(Float($0 + index + 1)) * 0.025 }, [output, input]
                )
            }
        }
        let transformer = try PhononTransformer(PhononWeights(arrays: tensors), "t", width: width,
                                                heads: 2, count: count, feedforward: 2 * width, period: 10000)
        let input = MLXArray((0..<(length * width)).map { cos(Float($0) * 0.03) }, [1, length, width])
        var fullState = transformer.makeState(), cachedState = transformer.makeState()
        let full = transformer(input, state: &fullState)
        eval([full] + fullState.flatMap(\.arrays))
        var chunks: [MLXArray] = []
        for range in [0..<125, 125..<136, 136..<137] {
            let chunk = transformer(input[0..., range, 0...], state: &cachedState)
            eval([chunk] + cachedState.flatMap(\.arrays))
            chunks.append(chunk)
        }
        let actual = concatenated(chunks, axis: 1).asArray(Float.self)
        XCTAssertTrue(actual.allSatisfy(\.isFinite))
        assertClose(actual, full.asArray(Float.self), tolerance: 2e-5)
        XCTAssertEqual(cachedState[0].offset, length)
    }

    func testTimeEmbeddingUsesUncenteredRMS() throws {
        var tensors: [String: MLXArray] = [:]
        tensors["t.freqs"] = MLXArray.zeros([128])
        tensors["t.mlp.0.weight"] = MLXArray.zeros([4, 256])
        tensors["t.mlp.0.bias"] = MLXArray.zeros([4])
        tensors["t.mlp.2.weight"] = MLXArray.zeros([4, 4])
        let row: [Float] = [1, 2, 3, 6]
        tensors["t.mlp.2.bias"] = MLXArray(row)
        tensors["t.mlp.3.alpha"] = MLXArray([Float](repeating: 2, count: 4))
        let time = try PhononFlow.TimeEmbedding(PhononWeights(arrays: tensors), "t", width: 4)
        let rms = sqrt(row.reduce(0) { $0 + $1 * $1 } / 4 + 1e-5)
        assertClose(time(0.5).asArray(Float.self), row.map { $0 * 2 / rms })
    }

    func testTensorShapeFailureIsAnError() throws {
        XCTAssertThrowsError(try PhononLinear(PhononWeights(arrays: ["p.weight": MLXArray.zeros([3, 4])]), "p", 4, 2))
    }

    private func assertClose(_ actual: [Float], _ expected: [Float], tolerance: Float = 1e-5,
                             file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(actual.count, expected.count, file: file, line: line)
        for (value, target) in zip(actual, expected) { XCTAssertEqual(value, target, accuracy: tolerance, file: file, line: line) }
    }
}
