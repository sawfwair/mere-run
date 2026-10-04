// Architecture derived from Aleph Alpha's Apache-2.0 Kolibri inference implementation.
import MLX
import MLXFast
import MLXNN
import MereRunTensor

func kolibriRMSNorm(dimensions: Int, eps: Float) -> RMSNorm {
    let norm = RMSNorm(dimensions: dimensions, eps: eps)
    norm.update(parameters: .unflattened(["weight": MLXArray.ones([dimensions], dtype: .bfloat16)]))
    return norm
}

package final class KolibriProjection: Module {
    @ParameterInfo(key: "weight") package var weight: MLXArray
    @ParameterInfo(key: "scales") package var scales: MLXArray?
    @ParameterInfo(key: "biases") package var biases: MLXArray?
    let policy: KolibriConfiguration.Quantization?

    init(input: Int, output: Int, experts: Int? = nil, policy: KolibriConfiguration.Quantization?) {
        self.policy = policy
        let prefix = experts.map { [$0] } ?? []
        _weight.wrappedValue = MLXArray.zeros(
            prefix + [output, policy.map { input * $0.bits / 32 } ?? input],
            dtype: policy == nil ? .bfloat16 : .uint32
        )
        _scales.wrappedValue = policy.map { MLXArray.zeros(prefix + [output, input / $0.groupSize], dtype: .bfloat16) }
        _biases.wrappedValue = policy.map { MLXArray.zeros(prefix + [output, input / $0.groupSize], dtype: .bfloat16) }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if let policy, let scales, let biases {
            return PortableQuantizedLinear(weight: weight, bias: nil, scales: scales, biases: biases,
                                           groupSize: policy.groupSize, bits: policy.bits)(x)
        }
        return matmul(x, weight.asType(x.dtype).T)
    }

    func callAsFunction(_ x: MLXArray, indices: MLXArray) -> MLXArray {
        let expanded = x.expandedDimensions(axis: -2)
        if let policy, let scales, let biases {
            return portableGatherQuantizedMM(expanded, weight, scales: scales, biases: biases,
                                             rhsIndices: indices, transpose: true,
                                             groupSize: policy.groupSize, bits: policy.bits,
                                             mode: .affine, sortedIndices: false).squeezed(axis: -2)
        }
        return matmul(expanded, weight.take(indices, axis: 0).swappedAxes(-1, -2)).squeezed(axis: -2)
    }
}

package final class KolibriMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: KolibriProjection
    @ModuleInfo(key: "up_proj") var upProj: KolibriProjection
    @ModuleInfo(key: "down_proj") var downProj: KolibriProjection

    var observeInput: ((String, MLXArray) -> Void)?
    let path: String

    init(config: KolibriConfiguration, path: String, experts: Bool) {
        self.path = path
        let width = experts ? config.moeIntermediateSize : config.sharedExpertIntermediateSize
        let count = experts ? config.numExperts : nil
        _gateProj.wrappedValue = KolibriProjection(input: config.hiddenSize, output: width, experts: count,
                                                  policy: config.quantization?[path + ".gate_proj"])
        _upProj.wrappedValue = KolibriProjection(input: config.hiddenSize, output: width, experts: count,
                                                policy: config.quantization?[path + ".up_proj"])
        _downProj.wrappedValue = KolibriProjection(input: width, output: config.hiddenSize, experts: count,
                                                  policy: config.quantization?[path + ".down_proj"])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        observeInput?(path + ".gate_proj", x)
        observeInput?(path + ".up_proj", x)
        let activated = silu(gateProj(x)) * upProj(x)
        observeInput?(path + ".down_proj", activated)
        return downProj(activated)
    }

    func callAsFunction(_ x: MLXArray, indices: MLXArray) -> MLXArray {
        let expanded = x.expandedDimensions(axis: -2)
        observeInput?(path + ".gate_proj", x)
        observeInput?(path + ".up_proj", x)
        let activated = silu(gateProj(expanded, indices: indices)) * upProj(expanded, indices: indices)
        observeInput?(path + ".down_proj", activated)
        return downProj(activated, indices: indices)
    }
}

package final class KolibriRouter: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "e_score_correction_bias") var correctionBias: MLXArray
    let topK: Int
    let renormalize: Bool

    init(config: KolibriConfiguration) {
        topK = config.numExpertsPerTok
        renormalize = config.normTopkProb
        _weight.wrappedValue = MLXArray.zeros([config.numExperts, config.hiddenSize], dtype: .bfloat16)
        _correctionBias.wrappedValue = MLXArray.zeros([config.numExperts], dtype: .float32)
        super.init()
    }

    package func callAsFunction(_ x: MLXArray) -> (indices: MLXArray, weights: MLXArray) {
        // Correction bias changes selection only. The unbiased sigmoid supplies route weights.
        let logits = matmul(x.asType(.float32), weight.asType(.float32).T)
        let indices = argPartition(-(logits + correctionBias), kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        var weights = sigmoid(takeAlong(logits, indices, axis: -1))
        if renormalize { weights = weights / (weights.sum(axis: -1, keepDims: true) + 1e-20) }
        return (indices, weights)
    }
}

package final class KolibriMoE: Module {
    @ModuleInfo(key: "gate") package var gate: KolibriRouter
    @ModuleInfo(key: "experts") var experts: KolibriMLP
    @ModuleInfo(key: "shared_experts") var sharedExperts: KolibriMLP

    init(config: KolibriConfiguration, path: String) {
        _gate.wrappedValue = KolibriRouter(config: config)
        _experts.wrappedValue = KolibriMLP(config: config, path: path + ".experts", experts: true)
        _sharedExperts.wrappedValue = KolibriMLP(config: config, path: path + ".shared_experts", experts: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let routing = gate(x)
        let outputs = experts(x, indices: routing.indices)
        let routed = (outputs.asType(.float32) * routing.weights.expandedDimensions(axis: -1)).sum(axis: -2).asType(x.dtype)
        return routed + sharedExperts(x)
    }
}
