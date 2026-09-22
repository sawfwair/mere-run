import MLX
import MLXNN

struct LayaEncoderLayer {
    let attentionNorm: LayaNorm?
    let mlpNorm: LayaNorm
    let queryKeyValue: LayaLinear
    let attentionOutput: LayaLinear
    let mlpInput: LayaLinear
    let mlpOutput: LayaLinear
    let heads: Int
    let local: Bool
    let rotaryBase: Float

    init(configuration: LayaEncoderConfiguration, index: Int, weights: inout LayaWeights) throws {
        let prefix = "encoder.layers.\(index)"
        let size = configuration.hiddenSize
        attentionNorm = index == 0 ? nil : try weights.norm(
            prefix + ".attn_norm", size, bias: configuration.normBias, epsilon: configuration.normEps)
        mlpNorm = try weights.norm(prefix + ".mlp_norm", size, bias: configuration.normBias, epsilon: configuration.normEps)
        queryKeyValue = try weights.linear(prefix + ".attn.Wqkv", size, 3 * size, bias: configuration.attentionBias)
        attentionOutput = try weights.linear(prefix + ".attn.Wo", size, size, bias: configuration.attentionBias)
        mlpInput = try weights.linear(prefix + ".mlp.Wi", size, 2 * configuration.intermediateSize, bias: configuration.mlpBias)
        mlpOutput = try weights.linear(prefix + ".mlp.Wo", configuration.intermediateSize, size, bias: configuration.mlpBias)
        heads = configuration.numAttentionHeads
        local = !index.isMultiple(of: configuration.globalAttnEveryNLayers)
        guard let rotary = configuration.ropeParameters[configuration.layerTypes[index]] else {
            throw LayaModelError.invalidConfiguration("Missing rotary parameters.")
        }
        rotaryBase = rotary.ropeTheta
    }

    func callAsFunction(_ input: MLXArray, globalMask: MLXArray, localMask: MLXArray) -> MLXArray {
        let normalized = attentionNorm?(input) ?? input
        let parts = split(queryKeyValue(normalized), parts: 3, axis: -1)
        let shape = [input.dim(0), input.dim(1), heads, -1]
        let query = parts[0].reshaped(shape).transposed(0, 2, 1, 3)
        let key = parts[1].reshaped(shape).transposed(0, 2, 1, 3)
        let value = parts[2].reshaped(shape).transposed(0, 2, 1, 3)
        let rotatedQuery = MLXFast.RoPE(query, dimensions: query.dim(-1), traditional: false, base: rotaryBase, scale: 1, offset: 0)
        let rotatedKey = MLXFast.RoPE(key, dimensions: key.dim(-1), traditional: false, base: rotaryBase, scale: 1, offset: 0)
        let attended = MLXFast.scaledDotProductAttention(
            queries: rotatedQuery, keys: rotatedKey, values: value,
            scale: 1 / Float(query.dim(-1)).squareRoot(), mask: .array(local ? localMask : globalMask))
        let hidden = input + attentionOutput(attended.transposed(0, 2, 1, 3).reshaped(input.shape))
        let gated = split(mlpInput(mlpNorm(hidden)), parts: 2, axis: -1)
        return hidden + mlpOutput(gelu(gated[0]) * gated[1])
    }
}

struct LayaDecisionLayer {
    let norm1: LayaNorm
    let norm2: LayaNorm
    let queryKeyValue: LayaLinear
    let attentionOutput: LayaLinear
    let linear1: LayaLinear
    let linear2: LayaLinear
    let heads: Int

    init(size: Int, index: Int, weights: inout LayaWeights) throws {
        let prefix = "head.layers.\(index)"
        norm1 = try weights.norm(prefix + ".norm1", size)
        norm2 = try weights.norm(prefix + ".norm2", size)
        queryKeyValue = try LayaLinear(weight: weights.take(prefix + ".self_attn.in_proj_weight", [3 * size, size]),
                                      bias: weights.take(prefix + ".self_attn.in_proj_bias", [3 * size]))
        attentionOutput = try weights.linear(prefix + ".self_attn.out_proj", size, size)
        linear1 = try weights.linear(prefix + ".linear1", size, 4 * size)
        linear2 = try weights.linear(prefix + ".linear2", 4 * size, size)
        heads = max(1, size / 64)
    }

    func callAsFunction(_ input: MLXArray, mask: MLXArray) -> MLXArray {
        let parts = split(queryKeyValue(norm1(input)), parts: 3, axis: -1)
        let shaped = parts.map { $0.reshaped([input.dim(0), input.dim(1), heads, -1]).transposed(0, 2, 1, 3) }
        let attended = MLXFast.scaledDotProductAttention(
            queries: shaped[0], keys: shaped[1], values: shaped[2],
            scale: 1 / Float(shaped[0].dim(-1)).squareRoot(), mask: .array(mask))
        let hidden = input + attentionOutput(attended.transposed(0, 2, 1, 3).reshaped(input.shape))
        // PyTorch TransformerEncoderLayer defaults to ReLU; only the scorer uses GELU.
        return hidden + linear2(relu(linear1(norm2(hidden))))
    }
}
