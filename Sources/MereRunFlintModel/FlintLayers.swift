import MLX
import MLXFast
import MLXNN
import MereRunTensor

func flintRMSNorm(dimensions: Int, eps: Float) -> RMSNorm {
    let norm = RMSNorm(dimensions: dimensions, eps: eps)
    norm.update(parameters: .unflattened(["weight": MLXArray.ones([dimensions], dtype: .bfloat16)]))
    return norm
}

package class FlintProjection: Module {
    @ParameterInfo(key: "weight") package var weight: MLXArray
    @ParameterInfo(key: "scales") package var scales: MLXArray?
    @ParameterInfo(key: "biases") package var biases: MLXArray?
    let policy: FlintConfiguration.Quantization?

    init(input: Int, output: Int, experts: Int? = nil, policy: FlintConfiguration.Quantization?) {
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

package class FlintFeedForward: Module {
    func callAsFunction(_ x: MLXArray) -> MLXArray { preconditionFailure("Flint feed-forward implementation is required.") }
}

package final class FlintMLP: FlintFeedForward {
    @ModuleInfo(key: "gate_proj") var gateProj: FlintProjection
    @ModuleInfo(key: "up_proj") var upProj: FlintProjection
    @ModuleInfo(key: "down_proj") var downProj: FlintProjection

    init(config: FlintConfiguration, path: String, width: Int, experts: Bool = false) {
        let count = experts ? config.numExperts : nil
        _gateProj.wrappedValue = FlintProjection(input: config.hiddenSize, output: width, experts: count, policy: config.policy(for: path))
        _upProj.wrappedValue = FlintProjection(input: config.hiddenSize, output: width, experts: count, policy: config.policy(for: path))
        _downProj.wrappedValue = FlintProjection(input: width, output: config.hiddenSize, experts: count, policy: config.policy(for: path))
        super.init()
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray { downProj(silu(gateProj(x)) * upProj(x)) }

    func callAsFunction(_ x: MLXArray, indices: MLXArray) -> MLXArray {
        let expanded = x.expandedDimensions(axis: -2)
        return downProj(silu(gateProj(expanded, indices: indices)) * upProj(expanded, indices: indices), indices: indices)
    }
}

package final class FlintRouter: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "expert_bias") var expertBias: MLXArray
    let topK: Int

    init(config: FlintConfiguration) {
        topK = config.numExpertsPerTok
        _weight.wrappedValue = MLXArray.zeros([config.numExperts, config.hiddenSize], dtype: .bfloat16)
        _expertBias.wrappedValue = MLXArray.zeros([config.numExperts], dtype: .float32)
        super.init()
    }

    package func callAsFunction(_ x: MLXArray) -> (indices: MLXArray, weights: MLXArray) {
        let scores = sigmoid(matmul(x, weight.asType(x.dtype).T).asType(.float32))
        // The released router adds expert bias to probabilities for selection only.
        let indices = argPartition(-(scores + expertBias), kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        let selected = takeAlong(scores, indices, axis: -1)
        return (indices, selected / (selected.sum(axis: -1, keepDims: true) + 1e-20))
    }
}

package final class FlintMoE: FlintFeedForward {
    @ModuleInfo(key: "gate") package var gate: FlintRouter
    @ModuleInfo(key: "switch_mlp") var experts: FlintMLP
    @ModuleInfo(key: "shared_expert") var sharedExpert: FlintMLP

    init(config: FlintConfiguration, path: String) {
        _gate.wrappedValue = FlintRouter(config: config)
        _experts.wrappedValue = FlintMLP(config: config, path: path + ".switch_mlp", width: config.moeIntermediateSize, experts: true)
        _sharedExpert.wrappedValue = FlintMLP(config: config, path: path + ".shared_expert", width: config.sharedExpertIntermediateSize)
        super.init()
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let routing = gate(x)
        let routed = (experts(x, indices: routing.indices) * routing.weights.asType(x.dtype).expandedDimensions(axis: -1)).sum(axis: -2)
        return routed + sharedExpert(x)
    }
}

package final class FlintEmbedding: FlintProjection {
    init(config: FlintConfiguration) {
        super.init(input: config.hiddenSize, output: config.vocabSize, policy: config.policy(for: "model.embed_tokens"))
    }

    override func callAsFunction(_ tokens: MLXArray) -> MLXArray {
        let selected = weight.take(tokens, axis: 0)
        if let policy, let scales, let biases {
            return dequantized(selected, scales: scales.take(tokens, axis: 0), biases: biases.take(tokens, axis: 0),
                               groupSize: policy.groupSize, bits: policy.bits)
        }
        return selected
    }
}
