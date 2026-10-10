import Foundation
import MLX

package struct WhistleDecoderCache {
    var keys: [MLXArray?] = Array(repeating: nil, count: 8)
    var values: [MLXArray?] = Array(repeating: nil, count: 8)
    var projections: [[MLXArray]] = Array(repeating: [], count: 8)
    var tokens: [Int] = []
    package var alignment: MLXArray?

    package init() {}
}

extension WhistleModel {
    /// One cached autoregressive step. Cross-attention K/V are computed once per clip.
    package func decode(token: Int, audio: WhistleAudioContext, cache: inout WhistleDecoderCache, depth: Int = 8, alignment: Bool = false) throws -> MLXArray {
        guard (0..<8199).contains(token), cache.tokens.count < 320 else {
            throw WhistleError.invalid("token or decoder position exceeds the checkpoint vocabulary/context")
        }
        guard (2...8).contains(depth) else { throw WhistleError.invalid("decoder depth must be 2...8") }
        cache.tokens.append(token)
        cache.alignment = nil
        var attention: [MLXArray] = []
        let x = weights.gather("embedding/embedding", indices: MLXArray([Int32(token)])).reshaped(1, 512) * sqrt(Float(512))
        var stream = broadcast(x.expandedDimensions(axis: 1), to: [1, 4, 512])
        for layer in Self.layers(depth: depth) {
            try Task.checkCancellation()
            stream = WhistleMath.advance(stream, weights: weights, prefix: "stack/", layer: layer) {
                self.decoderBlock($0, audio: audio, cache: &cache, layer: layer, attention: &attention, alignment: alignment)
            }
        }
        let output = WhistleMath.norm(mean(stream, axis: 1), weights("stack/final_norm/scale"))
        let logits = weights.project(output, "embedding/embedding")[0]
        if alignment { cache.alignment = concatenated(attention, axis: 0); eval(cache.alignment!) }
        eval(logits)
        return logits
    }

    private func decoderBlock(
        _ input: MLXArray, audio: WhistleAudioContext, cache: inout WhistleDecoderCache, layer: Int, attention: inout [MLXArray], alignment: Bool
    ) -> MLXArray {
        let prefix = "stack/layers/block/"
        func weight(_ name: String) -> MLXArray { weights(prefix + name, layer: layer) }
        func norm(_ x: MLXArray, _ name: String) -> MLXArray { WhistleMath.norm(x, weight(name + "/scale")) }
        var x = input
        if let site = [3, 7].firstIndex(of: layer) {
            let (key, value) = engram(tokens: cache.tokens, site: site)
            let alpha = sigmoid(sum(WhistleMath.unit(x) * WhistleMath.unit(key), axis: -1, keepDims: true) / sqrt(Float(512)))
            x = x + alpha * value
        }
        let normalized = norm(x, "ZCRMSNorm_0")
        let names = ["q", "k", "v"]
        let current = names.map { weights.project(normalized, prefix + "self_attn/\($0)_proj/kernel", layer: layer) }
        let history = cache.projections[layer]
        var convolved: [MLXArray] = []
        for index in 0..<3 {
            let taps = weight("self_attn/\(names[index])_taps")
            var z = current[index] * taps[0]
            for tap in 1..<3 where history.count >= tap * 3 {
                z = z + history[history.count - tap * 3 + index] * taps[tap]
            }
            convolved.append(z)
        }
        cache.projections[layer] = Array((history + current).suffix(6))
        let position = cache.tokens.count - 1
        let q = WhistleMath.rope(WhistleMath.norm(convolved[0].reshaped(1, 8, 1, 48), weight("self_attn/q_norm/scale")), offset: position)
        let k = WhistleMath.rope(WhistleMath.norm(convolved[1].reshaped(1, 2, 1, 48), weight("self_attn/k_norm/scale")), offset: position)
        let v = convolved[2].reshaped(1, 2, 1, 64)
        let keys = cache.keys[layer].map { concatenated([$0, k], axis: 2) } ?? k
        let values = cache.values[layer].map { concatenated([$0, v], axis: 2) } ?? v
        cache.keys[layer] = keys
        cache.values[layer] = values
        var output = WhistleMath.attention(q, keys, values)
        output = output * sigmoid(weights.project(normalized, prefix + "self_attn/gate_proj/kernel", layer: layer))
        output = weights.project(output, prefix + "self_attn/out_proj/kernel", layer: layer)
        x = x + sigmoid(weight("attn_gate")) * norm(output, "post_attn_norm")
        let crossInput = norm(x, "cross_norm")
        let crossQuery = WhistleMath.norm(
            weights.project(crossInput, prefix + "cross_attn/q_proj/kernel", layer: layer).reshaped(1, 8, 1, 48), weight("cross_attn/q_norm/scale")
        )
        if alignment {
            let probabilities = softmax(matmul(crossQuery, audio.keys[layer].swappedAxes(-1, -2)) / sqrt(Float(48)), axis: -1)
            attention.append(probabilities.reshaped(8, -1))
        }
        output = WhistleMath.attention(crossQuery, audio.keys[layer], audio.values[layer])
        output = output * sigmoid(weights.project(crossInput, prefix + "cross_attn/gate_proj/kernel", layer: layer))
        output = weights.project(output, prefix + "cross_attn/out_proj/kernel", layer: layer)
        x = x + sigmoid(weight("cross_gate")) * norm(output, "post_cross_norm")
        return x + WhistleMath.hadamard(norm(x, "pre_hada_norm"), weights: weights, prefix: prefix + "hadamard_mlp/", layer: layer)
    }

    /// Endpoint-preserving bisection, using physical layer indices for mHC and Engram.
    package static func layers(depth: Int) -> [Int] {
        var selected = [0, 7]
        while selected.count < depth {
            selected.sort()
            let gap = zip(selected, selected.dropFirst()).max { left, right in
                let a = left.1 - left.0, b = right.1 - right.0
                return a == b ? left.0 > right.0 : a < b
            }!
            selected.append((gap.0 + gap.1) / 2)
        }
        return selected.sorted()
    }

    package static func engramIndices(tokens: [Int], position: Int) -> [Int32] {
        var indices: [Int32] = []
        for (orderIndex, order) in [2, 3].enumerated() {
            for head in 0..<2 {
                var hash = UInt32(0x9E3779B9) &* UInt32(orderIndex * 2 + head + 1)
                for offset in 0..<order {
                    let token = position >= offset ? UInt32(tokens[position - offset]) : 0
                    hash = (hash ^ token) &* 0x01000193
                }
                hash = hash ^ (hash >> 15)
                indices.append(Int32(hash % 18432) + Int32((orderIndex * 2 + head) * 18432))
            }
        }
        return indices
    }

    private func engram(tokens: [Int], site: Int) -> (MLXArray, MLXArray) {
        let prefix = "engrams_\(site)/"
        var key = MLXArray.zeros([1, 512])
        var value = MLXArray.zeros([1, 512])
        for tap in 0..<4 {
            let position = tokens.count - 1 - tap * 3
            guard position >= 0 else { continue }
            let rows = weights.gather(prefix + "embedding", indices: MLXArray(Self.engramIndices(tokens: tokens, position: position)))
            let mask = MLXArray([Float(position >= 1 ? 1 : 0), Float(position >= 1 ? 1 : 0),
                                 Float(position >= 2 ? 1 : 0), Float(position >= 2 ? 1 : 0)]).reshaped(4, 1)
            let features = (rows * mask).reshaped(1, 512)
            if tap == 0 { key = weights.project(features, prefix + "key_proj/kernel") }
            value = value + weights.project(features, prefix + "value_proj/kernel") * weights(prefix + "taps")[tap]
        }
        return (key, value)
    }
}
