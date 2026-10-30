import Foundation
import MLX
import MLXFast
import XCTest
import MereRunMLXTestSupport
@testable import MereRunUnlimitedOCRModel

/// Opt-in same-prefix diagnostics; checkpoints and large captures remain external.
final class UnlimitedOCRPriceDiagnostics: MLXTestCase {
    private struct Input: Decodable {
        let step: Int
        let tokens: [Int]
        let promptTokens: [Int]
        let zeroTokens: [String: [Int]]
    }
    private struct Decision: Encodable {
        let step: Int
        let expected: Int
        let winner: Int
        let expectedScore: Float
        let winnerScore: Float
        let score00: Float
        let score000: Float
    }
    private struct Metric: Encodable {
        let maximum: Float
        let unequal: Int
    }

    func testIsolatedOutputProjection() throws {
        let env = ProcessInfo.processInfo.environment
        guard let directory = env["MERERUN_TEST_UNLIMITED_OCR_PRICE"] else {
            throw XCTSkip("Provide the upstream price projection capture.")
        }
        let output = URL(fileURLWithPath: directory)
        let reference = try MLX.loadArrays(url: output.appendingPathComponent("head.safetensors"))
        let root = URL(fileURLWithPath: try XCTUnwrap(env["MERERUN_TEST_UNLIMITED_OCR_ROOT"]))
        let weights = try MLX.loadArrays(url: root.appendingPathComponent("model-00002-of-00002.safetensors"))
        let head = try XCTUnwrap(weights["lm_head.weight"])
        let hidden = try XCTUnwrap(reference["hidden"])
        let original = try XCTUnwrap(reference["original"])
        let bf16 = matmul(hidden, head.T)
        let fp32 = matmul(hidden.asType(.float32), head.asType(.float32).T).asType(.bfloat16)
        var metrics: [String: Metric] = [:]
        for (name, actual) in [("bf16", bf16), ("fp32", fp32)] {
            metrics[name] = Metric(maximum: abs(actual.asType(.float32) - original.asType(.float32)).max().item(Float.self),
                                   unequal: (actual .!= original).sum().item(Int.self))
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(metrics).write(to: output.appendingPathComponent("native-head.json"))
    }

    func testSamePrefixPriceDecision() throws {
        let env = ProcessInfo.processInfo.environment
        guard let directory = env["MERERUN_TEST_UNLIMITED_OCR_PRICE"] else {
            throw XCTSkip("Provide a teacher-forced upstream price capture.")
        }
        XCTAssertEqual(Device.defaultDevice().deviceType, .gpu)
        let output = URL(fileURLWithPath: directory)
        let input = try JSONDecoder().decode(Input.self, from: Data(contentsOf: output.appendingPathComponent("upstream.json")))
        let reference = try MLX.loadArrays(url: output.appendingPathComponent("upstream.safetensors"))
        let root = URL(fileURLWithPath: try XCTUnwrap(env["MERERUN_TEST_UNLIMITED_OCR_ROOT"]))
        var tensors: [String: MLXArray] = [:]
        for name in ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"] {
            tensors.merge(try MLX.loadArrays(url: root.appendingPathComponent(name))) { first, _ in first }
        }
        let config = try UnlimitedOCRConfiguration.load(Data(contentsOf: root.appendingPathComponent("config.json")))
        let decoder = UnlimitedOCRDecoder(config: config, tensors: tensors)
        let embeddings = try XCTUnwrap(reference["embeddings"])
        var modes: [(String, MLXArray)] = [("upstream-vision", embeddings)]
        if let path = env["MERERUN_TEST_UNLIMITED_OCR_PRICE_NATIVE_FEATURES"] {
            let features = try XCTUnwrap(MLX.loadArrays(url: URL(fileURLWithPath: path))["features.0"])
            let first = try XCTUnwrap(input.promptTokens.firstIndex(of: 128815))
            let last = try XCTUnwrap(input.promptTokens.lastIndex(of: 128815))
            XCTAssertEqual(last - first + 1, features.dim(0))
            modes.append(("native-vision", concatenated([embeddings[..<first], features, embeddings[(last + 1)...]], axis: 0)))
        }
        let id00 = try XCTUnwrap(input.zeroTokens["00"]?.first)
        let id000 = try XCTUnwrap(input.zeroTokens["000"]?.first)
        for (mode, prompt) in modes {
            var state = try UnlimitedOCRPageState(tokens: input.promptTokens)
            let cache = (0..<12).map { _ in UnlimitedOCRCache() }
            var logits = decoder.forward(prompt, state: state, cache: cache)
            var finalNorm: MLXArray?
            var decisions: [Decision] = []
            for step in 0...input.step {
                let winner = argMax(logits).item(Int.self)
                decisions.append(Decision(step: step, expected: input.tokens[step], winner: winner,
                    expectedScore: logits[input.tokens[step]].item(Float.self), winnerScore: logits[winner].item(Float.self),
                    score00: logits[id00].item(Float.self), score000: logits[id000].item(Float.self)))
                if step == input.step {
                    var arrays = ["logits": logits]
                    if let finalNorm {
                        arrays["head.input"] = finalNorm
                        arrays["unrounded.pair"] = matmul(finalNorm.asType(.float32),
                            decoder.weights.tensor("lm_head.weight").take(MLXArray([id00, id000]), axis: 0).asType(.float32).T)
                    }
                    try MLX.save(arrays: arrays, url: output.appendingPathComponent(mode + ".safetensors"))
                    break
                }
                let token = input.tokens[step]
                state.append(token)
                logits = decoder.forward(decoder.embeddings([token]), state: state, cache: cache,
                    trace: step + 1 == input.step ? { name, value in
                        if name == "decoder.norm" { finalNorm = value }
                    } : nil)
            }
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(decisions).write(to: output.appendingPathComponent(mode + ".json"))
            FileHandle.standardError.write(Data("[price] \(mode): \(decisions.last!)\n".utf8))
        }
        // Isolate attention from accumulated vision, projection, and cache drift.
        var state = try UnlimitedOCRPageState(tokens: input.promptTokens)
        for token in input.tokens.prefix(input.step) { state.append(token) }
        var metrics: [String: Metric] = [:]
        func compare(_ name: String, _ actual: MLXArray, _ expected: MLXArray) {
            metrics[name] = Metric(maximum: abs(actual.asType(.float32) - expected.asType(.float32)).max().item(Float.self),
                                   unequal: (actual .!= expected).sum().item(Int.self))
        }
        for layer in 0..<12 {
            func tensor(_ name: String) throws -> MLXArray { try XCTUnwrap(reference["attention.\(layer).\(name)"]) }
            let q = try tensor("q"), k = try tensor("k"), v = try tensor("v")
            let expected = try tensor("output").asType(.bfloat16)
            let sparse = config.sparseAttentionConfig
            let actual = UnlimitedOCRAttention.attend(q: q, k: k, v: v, indexQ: try tensor("iq"), indexK: try tensor("ik"),
                head: try tensor("head").asType(.float32), state: state, offset: state.tokenPositions.count - 1,
                scale: sparse.dashScalePerLayer[layer], floor: sparse.dashFloorPerLayer[layer], sigma: sparse.dashSigma)
            compare("layer.\(layer).gathered", actual, expected)
            let topi = try tensor("topi").squeezed(axis: 0)
            let valid = try tensor("valid").squeezed(axis: 0)
            let blockBias = MLXArray.full([(k.dim(1) + 31) / 32], values: MLXArray(-Float.infinity))
            blockBias[topi] = which(valid, try tensor("bias").squeezed(axis: 0), MLXArray(-Float.infinity))
            let positions = MLXArray(0..<k.dim(1))
            let local = positions .> k.dim(1) - 1 - 128
            let mask = which(local, MLXArray(Float(0)), blockBias.take(positions.floorDivide(32)))
            let scores = matmul(q.asType(.float32), k.asType(.float32).swappedAxes(-1, -2)) * Float(1 / sqrt(Double(128))) + mask
            let explicit = matmul(softmax(scores, axis: -1), v.asType(.float32)).asType(.bfloat16)
            compare("layer.\(layer).explicit", explicit, expected)
            let fused = MLXFast.scaledDotProductAttention(queries: q.asType(.float32).expandedDimensions(axis: 0),
                keys: k.asType(.float32).expandedDimensions(axis: 0), values: v.asType(.float32).expandedDimensions(axis: 0),
                scale: Float(1 / sqrt(Double(128))), mask: mask).asType(.bfloat16).squeezed(axis: 0)
            compare("layer.\(layer).fusedDense", fused, expected)
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(metrics).write(to: output.appendingPathComponent("isolated-attention.json"))
    }
}
