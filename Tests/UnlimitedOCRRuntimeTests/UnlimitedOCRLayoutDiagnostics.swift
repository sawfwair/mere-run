import Foundation
import MLX
import XCTest
import MereRunMLXTestSupport
@testable import MereRunUnlimitedOCRModel

/// Causal activation substitutions and same-input block probes for the first split.
final class UnlimitedOCRLayoutDiagnostics: MLXTestCase {
    private struct Input: Decodable { let step: Int; let tokens: [Int]; let promptTokens: [Int] }
    private struct Boundary: Encodable {
        let vision: String
        let boundary: String
        let winner: Int
        let candidates: [Int]
        let scores: [Float]
        let firstDivergence: Int?
    }
    private struct Metric: Encodable { let maximum: Float; let unequal: Int; let rmse: Float }

    func testFirstLayoutSplit() throws {
        let env = ProcessInfo.processInfo.environment
        guard let path = env["MERERUN_TEST_UNLIMITED_OCR_LAYOUT"] else {
            throw XCTSkip("Provide the first-layout upstream capture.")
        }
        XCTAssertEqual(Device.defaultDevice().deviceType, .gpu)
        let directory = URL(fileURLWithPath: path)
        let input = try JSONDecoder().decode(Input.self, from: Data(contentsOf: directory.appendingPathComponent("upstream.json")))
        let reference = try MLX.loadArrays(url: directory.appendingPathComponent("upstream.safetensors"))
        let root = URL(fileURLWithPath: try XCTUnwrap(env["MERERUN_TEST_UNLIMITED_OCR_ROOT"]))
        var tensors: [String: MLXArray] = [:]
        for name in ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"] {
            tensors.merge(try MLX.loadArrays(url: root.appendingPathComponent(name))) { first, _ in first }
        }
        let config = try UnlimitedOCRConfiguration.load(Data(contentsOf: root.appendingPathComponent("config.json")))
        let model = UnlimitedOCRDecoder(config: config, tensors: tensors)
        let prompt = try XCTUnwrap(reference["embeddings"])
        var modes: [(String, MLXArray)] = [("upstream", prompt)]
        if let featuresPath = env["MERERUN_TEST_UNLIMITED_OCR_PRICE_NATIVE_FEATURES"] {
            let features = try XCTUnwrap(MLX.loadArrays(url: URL(fileURLWithPath: featuresPath))["features.0"])
            let first = try XCTUnwrap(input.promptTokens.firstIndex(of: 128815))
            let last = try XCTUnwrap(input.promptTokens.lastIndex(of: 128815))
            modes.append(("native", concatenated([prompt[..<first], features, prompt[(last + 1)...]], axis: 0)))
        }
        var results: [Boundary] = []
        var captures: [String: MLXArray] = [:]
        for (vision, embeddings) in env["MERERUN_TEST_UNLIMITED_OCR_LAYOUT_PROBES_ONLY"] == "1" ? [] : modes {
            for boundary in -1...12 {
                Memory.clearCache()
                var step = 0
                var state = try UnlimitedOCRPageState(tokens: input.promptTokens)
                let cache = (0..<12).map { _ in UnlimitedOCRCache() }
                let name = boundary == 12 ? "decoder.norm" : "decoder.block.\(boundary)"
                func transform(_ key: String, _ value: MLXArray) -> MLXArray {
                    if key == name {
                        return reference["step.\(step).\(key)"]!.reshaped(value.shape)
                    }
                    return value
                }
                func observe(_ key: String, _ value: MLXArray) {
                    if boundary == -1 && (step == 0 || step == input.step) {
                        captures["\(vision).step.\(step).\(key)"] = value
                    }
                }
                var logits = model.forward(embeddings, state: state, cache: cache, trace: observe, transform: transform)
                var firstDivergence: Int?
                for index in 0..<input.step {
                    if firstDivergence == nil && argMax(logits).item(Int.self) != input.tokens[index] {
                        firstDivergence = index
                    }
                    state.append(input.tokens[index])
                    step = index + 1
                    logits = model.forward(model.embeddings([input.tokens[index]]), state: state, cache: cache,
                                           trace: observe, transform: transform)
                }
                let indices = argSort(-logits)[..<8]
                let winner = argMax(logits).item(Int.self)
                if firstDivergence == nil && winner != input.tokens[input.step] { firstDivergence = input.step }
                if boundary == 12 { XCTAssertEqual(winner, argMax(reference["logits"]!).item(Int.self)) }
                results.append(Boundary(vision: vision, boundary: boundary < 0 ? "baseline" : name,
                    winner: winner, candidates: indices.asArray(Int.self),
                    scores: logits.take(indices).asArray(Float.self), firstDivergence: firstDivergence))
            }
        }
        var metrics: [String: Metric] = [:]
        func compare(_ name: String, _ actual: MLXArray, _ expected: MLXArray) {
            let target = expected.reshaped(actual.shape)
            let delta = actual.asType(.float32) - target.asType(.float32)
            metrics[name] = Metric(maximum: abs(delta).max().item(Float.self), unequal: (actual .!= target).sum().item(Int.self),
                                   rmse: sqrt(mean(square(delta))).item(Float.self))
        }
        var state = try UnlimitedOCRPageState(tokens: input.promptTokens)
        for token in input.tokens.prefix(input.step) { state.append(token) }
        let tok = MLXArray([state.tokenPositions.last!])
        let page = MLXArray([state.pagePositions.last!])
        for layer in 0..<12 {
            let prefix = "layer.\(layer)."
            let cache = UnlimitedOCRCache()
            cache.keys = try XCTUnwrap(reference["attention.\(layer).k"])[0..., ..<(state.tokenPositions.count - 1), 0...]
            cache.values = try XCTUnwrap(reference["attention.\(layer).v"])[0..., ..<(state.tokenPositions.count - 1), 0...]
            cache.indexKeys = try XCTUnwrap(reference["attention.\(layer).ik"])[..<(state.tokenPositions.count - 1)]
            let value = try XCTUnwrap(reference[prefix + "input"])
            let result = model.block(value, layer: layer, tok: tok, page: page, state: state,
                offset: state.tokenPositions.count - 1, cache: cache, trace: { name, actual in
                    if let expected = reference[prefix + name] { compare(prefix + name, actual, expected) }
                })
            compare(prefix + "output", result, try XCTUnwrap(reference[prefix + "output"]))
        }
        let dense = "layer.0.mlp."
        let normalized = reference["layer.0.post_attention_layernorm"]!.reshaped(1, 1280)
        let gate = model.weights.linear(normalized, "model.layers.0.mlp.gate_proj")
        let up = model.weights.linear(normalized, "model.layers.0.mlp.up_proj")
        let activation = UnlimitedOCRMath.silu(reference[dense + "gate_proj"]!.reshaped(1, -1))
        let product = reference[dense + "act_fn"]!.reshaped(1, -1) * reference[dense + "up_proj"]!.reshaped(1, -1)
        let down = model.weights.linear(product, "model.layers.0.mlp.down_proj")
        for (key, value) in [("gate_proj", gate), ("up_proj", up), ("act_fn", activation), ("down_proj", down)] {
            compare(dense + key + ".isolated", value, reference[dense + key]!)
            captures[dense + key] = value
        }
        let activationURL = directory.appendingPathComponent("activations.safetensors")
        if FileManager.default.fileExists(atPath: activationURL.path) {
            let activationReference = try MLX.loadArrays(url: activationURL)
            let x = activationReference["x"]!
            for (key, actual) in [("silu", UnlimitedOCRMath.silu(x)), ("gelu", UnlimitedOCRMath.gelu(x)),
                                  ("sigmoid", sigmoid(x.asType(.float32)).asType(x.dtype))] {
                compare("activation." + key, actual, activationReference[key]!)
                captures["activation." + key] = actual
            }
        }
        if let prefillPath = env["MERERUN_TEST_UNLIMITED_OCR_LAYOUT_PREFILL"] {
            let prefill = try MLX.loadArrays(url: URL(fileURLWithPath: prefillPath))
            let patch = UnlimitedOCRWeights(tensors: tensors).conv(prefill["pixels"]!, "model.sam_model.patch_embed.proj", stride: 16)
            compare("prefill.sam.patch", patch, prefill["sam.patch"]!)
            captures["prefill.sam.patch"] = patch
            let pixels = prefill["pixels"]!
            let patches = pixels.reshaped(1, 64, 16, 64, 16, 3)
            let weight = tensors["model.sam_model.patch_embed.proj.weight"]!.asType(.float32)
            let bias = tensors["model.sam_model.patch_embed.proj.bias"]!.asType(.float32)
            for (order, input, kernel) in [
                ("chw", patches.transposed(0, 1, 3, 5, 2, 4), weight),
                ("hwc", patches.transposed(0, 1, 3, 2, 4, 5), weight.transposed(0, 2, 3, 1))
            ] {
                let actual = addMM(bias, input.reshaped(4096, 768).asType(.float32), kernel.reshaped(768, 768).T)
                    .asType(.bfloat16).reshaped(patch.shape)
                compare("prefill.sam.patch." + order, actual, prefill["sam.patch"]!)
                captures["prefill.sam.patch." + order] = actual
            }
            if env["MERERUN_TEST_UNLIMITED_OCR_LAYOUT_PATCH_SUBSTITUTIONS"] == "1" {
                let weights = UnlimitedOCRWeights(tensors: tensors)
                let vision = UnlimitedOCRVision(weights: weights)
                for (name, substitute) in [("upstream", prefill["sam.patch"]!),
                                            ("chw", captures["prefill.sam.patch.chw"]!),
                                            ("hwc", captures["prefill.sam.patch.hwc"]!)] {
                    let encoded = vision.encode(pixels, transform: { key, value in key == "sam.patch" ? substitute : value })
                    let newline = broadcast(weights.tensor("model.image_newline"), to: [16, 1, 1280])
                    let page = concatenated([concatenated([encoded.reshaped(16, 16, 1280), newline], axis: 1)
                        .reshaped(272, 1280), weights.tensor("model.view_seperator").reshaped(1, 1280)], axis: 0)
                    let first = input.promptTokens.firstIndex(of: 128815)!
                    let last = input.promptTokens.lastIndex(of: 128815)!
                    let embeddings = concatenated([prompt[..<first], page, prompt[(last + 1)...]], axis: 0)
                    var state = try UnlimitedOCRPageState(tokens: input.promptTokens)
                    let cache = (0..<12).map { _ in UnlimitedOCRCache() }
                    var logits = model.forward(embeddings, state: state, cache: cache)
                    var firstDivergence: Int?
                    for index in 0..<input.step {
                        if firstDivergence == nil && argMax(logits).item(Int.self) != input.tokens[index] {
                            firstDivergence = index
                        }
                        state.append(input.tokens[index])
                        logits = model.forward(model.embeddings([input.tokens[index]]), state: state, cache: cache)
                    }
                    let winner = argMax(logits).item(Int.self)
                    if firstDivergence == nil && winner != input.tokens[input.step] { firstDivergence = input.step }
                    let indices = argSort(-logits)[..<8]
                    results.append(Boundary(vision: "patch-" + name, boundary: "sam.patch", winner: winner,
                        candidates: indices.asArray(Int.self), scores: logits.take(indices).asArray(Float.self),
                        firstDivergence: firstDivergence))
                    Memory.clearCache()
                }
            }
            let norm = prefill["decoder.first.norm1"]!
            let prefillState = try UnlimitedOCRPageState(tokens: input.promptTokens)
            let prefillNames = ["input_layernorm": "norm1", "post_attention_layernorm": "norm2",
                                "self_attn.0": "attention", "mlp": "down"]
            _ = model.block(prefill["embeddings"]!, layer: 0, tok: MLXArray(prefillState.tokenPositions),
                page: MLXArray(prefillState.pagePositions), state: prefillState, offset: 0,
                cache: UnlimitedOCRCache(), trace: { name, actual in
                    if let key = prefillNames[name] {
                        compare("prefill.decoder.first." + key, actual, prefill["decoder.first." + key]!)
                    }
                })
            for projection in ["q_proj", "k_proj", "v_proj"] {
                let actual = model.weights.linear(norm, "model.layers.0.self_attn." + projection)
                compare("prefill.decoder.first." + projection, actual, prefill["decoder.first." + projection]!)
                captures["prefill.decoder.first." + projection] = actual
            }
            for (projection, key) in [("gate_proj", "gate"), ("up_proj", "up")] {
                let actual = model.weights.linear(prefill["decoder.first.norm2"]!, "model.layers.0.mlp." + projection)
                compare("prefill.decoder.first." + key, actual, prefill["decoder.first." + key]!)
                captures["prefill.decoder.first." + key] = actual
            }
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        if !results.isEmpty {
            try encoder.encode(results).write(to: directory.appendingPathComponent("boundaries.json"))
        }
        try encoder.encode(metrics).write(to: directory.appendingPathComponent("isolated-blocks.json"))
        try MLX.save(arrays: captures, url: directory.appendingPathComponent("native-states.safetensors"))
    }
}
