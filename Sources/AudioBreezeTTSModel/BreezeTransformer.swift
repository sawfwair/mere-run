import Foundation
import MLX
import MLXFast
import MLXNN
import MereRunKVCache

struct BreezeTransformerShape {
    let hiddenSize: Int
    let intermediateSize: Int
    let numAttentionHeads: Int
    let numKeyValueHeads: Int
    let headDim: Int
    let rmsNormEps: Float
    let ropeTheta: Float
    let ropeScaling: BreezeRopeScaling?
    let queryKeyNorm: Bool

    init(_ config: BreezeBackboneConfig, ropeScaling: BreezeRopeScaling?) {
        hiddenSize = config.hiddenSize
        intermediateSize = config.intermediateSize
        numAttentionHeads = config.numAttentionHeads
        numKeyValueHeads = config.numKeyValueHeads
        headDim = config.headDim
        rmsNormEps = config.rmsNormEps
        ropeTheta = config.ropeTheta
        self.ropeScaling = ropeScaling
        queryKeyNorm = true
    }

    init(_ config: BreezeDepthDecoderConfig) {
        hiddenSize = config.hiddenSize
        intermediateSize = config.intermediateSize
        numAttentionHeads = config.numAttentionHeads
        numKeyValueHeads = config.numKeyValueHeads
        headDim = config.headDim
        rmsNormEps = config.rmsNormEps
        ropeTheta = config.ropeTheta
        ropeScaling = config.ropeScaling
        queryKeyNorm = false
    }
}

struct BreezeRotary {
    let inverseFrequencies: MLXArray

    init(dimensions: Int, theta: Float, scaling: BreezeRopeScaling?) {
        let values = stride(from: 0, to: dimensions, by: 2).map { index -> Float in
            let frequency = 1 / Foundation.pow(theta, Float(index) / Float(dimensions))
            guard let scaling else { return frequency }
            if scaling.ropeType == "linear" { return frequency / scaling.factor }
            guard scaling.ropeType == "llama3",
                  let original = scaling.originalMaxPositionEmbeddings,
                  let high = scaling.highFreqFactor,
                  let low = scaling.lowFreqFactor else { return frequency }
            let wavelength = 2 * Float.pi / frequency
            let highWavelength = Float(original) / high
            let lowWavelength = Float(original) / low
            if wavelength < highWavelength { return frequency }
            if wavelength > lowWavelength { return frequency / scaling.factor }
            let blend = (Float(original) / wavelength - low) / (high - low)
            return (1 - blend) * frequency / scaling.factor + blend * frequency
        }
        inverseFrequencies = MLXArray(values)
    }

    func apply(_ x: MLXArray, offset: Int) -> MLXArray {
        let positions = MLX.arange(offset, offset + x.dim(2)).asType(.float32)
        let frequencies = positions[0..., .newAxis] * inverseFrequencies[.newAxis, 0...]
        let angles = MLX.concatenated([frequencies, frequencies], axis: -1)
        let cosine = MLX.cos(angles).asType(x.dtype)[.newAxis, .newAxis, 0..., 0...]
        let sine = MLX.sin(angles).asType(x.dtype)[.newAxis, .newAxis, 0..., 0...]
        let half = x.dim(-1) / 2
        let rotated = MLX.concatenated([-x[.ellipsis, half...], x[.ellipsis, 0..<half]], axis: -1)
        return x * cosine + rotated * sine
    }
}

private final class BreezeTransformerAttention: Module {
    private let config: BreezeTransformerShape
    private let rotary: BreezeRotary
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm?
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm?

    init(config: BreezeTransformerShape) {
        self.config = config
        rotary = BreezeRotary(dimensions: config.headDim, theta: config.ropeTheta, scaling: config.ropeScaling)
        _qProj.wrappedValue = Linear(config.hiddenSize, config.numAttentionHeads * config.headDim, bias: false)
        _kProj.wrappedValue = Linear(config.hiddenSize, config.numKeyValueHeads * config.headDim, bias: false)
        _vProj.wrappedValue = Linear(config.hiddenSize, config.numKeyValueHeads * config.headDim, bias: false)
        _oProj.wrappedValue = Linear(config.numAttentionHeads * config.headDim, config.hiddenSize, bias: false)
        _qNorm.wrappedValue = config.queryKeyNorm ? RMSNorm(dimensions: config.headDim, eps: config.rmsNormEps) : nil
        _kNorm.wrappedValue = config.queryKeyNorm ? RMSNorm(dimensions: config.headDim, eps: config.rmsNormEps) : nil
    }

    func callAsFunction(_ x: MLXArray, cache: KVCache?) -> MLXArray {
        let batch = x.dim(0)
        let length = x.dim(1)
        let offset = cache?.offset ?? 0
        var query = qProj(x).reshaped(batch, length, config.numAttentionHeads, config.headDim)
        var key = kProj(x).reshaped(batch, length, config.numKeyValueHeads, config.headDim)
        if let qNorm { query = qNorm(query) }
        if let kNorm { key = kNorm(key) }
        query = rotary.apply(query.transposed(0, 2, 1, 3), offset: offset)
        key = rotary.apply(key.transposed(0, 2, 1, 3), offset: offset)
        var value = vProj(x).reshaped(batch, length, config.numKeyValueHeads, config.headDim).transposed(0, 2, 1, 3)
        if let cache { (key, value) = cache.update(keys: key, values: value) }
        let output = MLXFast.scaledDotProductAttention(
            queries: query,
            keys: key,
            values: value,
            scale: 1 / Float(config.headDim).squareRoot(),
            mask: length == 1 ? .none : .causal
        )
        return oProj(output.transposed(0, 2, 1, 3).reshaped(batch, length, config.hiddenSize))
    }
}

private final class BreezeTransformerMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(config: BreezeTransformerShape) {
        _gateProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(MLXNN.silu(gateProj(x)) * upProj(x))
    }
}

final class BreezeTransformerBlock: Module {
    @ModuleInfo(key: "self_attn") private var attention: BreezeTransformerAttention
    @ModuleInfo(key: "mlp") private var mlp: BreezeTransformerMLP
    @ModuleInfo(key: "input_layernorm") private var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") private var postAttentionNorm: RMSNorm

    init(config: BreezeTransformerShape) {
        _attention.wrappedValue = BreezeTransformerAttention(config: config)
        _mlp.wrappedValue = BreezeTransformerMLP(config: config)
        _inputNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ x: MLXArray, cache: KVCache? = nil) -> MLXArray {
        let attended = x + attention(inputNorm(x), cache: cache)
        return attended + mlp(postAttentionNorm(attended))
    }
}
