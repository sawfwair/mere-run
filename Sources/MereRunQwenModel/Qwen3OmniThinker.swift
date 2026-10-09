import Foundation
import MLX
import MLXFast
import MLXNN
import MereRunTextEncoder

/// Causal, single-pass Qwen3-Omni MoE states, without generation or the talker/code2wav networks.
package final class Qwen3OmniThinker: Module {
    @ModuleInfo(key: "embed_tokens") package var embedding: Embedding
    @ModuleInfo var layers: [Qwen3OmniLayer]
    @ModuleInfo var norm: RMSNorm
    @ModuleInfo(key: "lm_head") var outputEmbedding: Linear
    package let config: Qwen3OmniConfiguration.Text
    package let quantization: Qwen3OmniConfiguration.ExpertQuantization?

    package init(config: Qwen3OmniConfiguration.Text, quantization: Qwen3OmniConfiguration.ExpertQuantization? = nil) {
        self.config = config
        self.quantization = quantization
        self._embedding.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self._layers.wrappedValue = (0..<config.numHiddenLayers).map { _ in Qwen3OmniLayer(config, quantization: quantization) }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._outputEmbedding.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
    }

    package func callAsFunction(ids: MLXArray, positions: MLXArray, embeddings: MLXArray? = nil,
                               visualIndices: [Int] = [], deepstack: [MLXArray] = []) throws -> MLXArray {
        var hidden = embeddings ?? embedding(ids)
        for (index, layer) in layers.enumerated() {
            try Task.checkCancellation()
            hidden = layer(hidden, positions: positions)
            if index < deepstack.count {
                guard deepstack[index].shape == [visualIndices.count, config.hiddenSize] else {
                    throw ClefError.invalidInput("Clef Omni deep-stack feature count differs from visual placeholders.")
                }
                // Scatter a sparse additive residual; audio/text positions receive zero.
                let residual = MLXArray.zeros(hidden.shape, dtype: hidden.dtype)
                residual[0, MLXArray(visualIndices.map(Int32.init))] = deepstack[index]
                hidden = hidden + residual
            }
            MLX.eval(hidden)
        }
        return norm(hidden)
    }

    package func lexical(_ ids: MLXArray) -> MLXArray { outputEmbedding.weight.take(ids, axis: 0) }
}

final class Qwen3OmniLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: Qwen3OmniAttention
    @ModuleInfo var mlp: Qwen3OmniExperts
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postNorm: RMSNorm

    init(_ config: Qwen3OmniConfiguration.Text, quantization: Qwen3OmniConfiguration.ExpertQuantization?) {
        self._attention.wrappedValue = Qwen3OmniAttention(config)
        self._mlp.wrappedValue = Qwen3OmniExperts(config, quantization: quantization)
        self._inputNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ x: MLXArray, positions: MLXArray) -> MLXArray {
        let hidden = x + attention(inputNorm(x), positions: positions)
        return hidden + mlp(postNorm(hidden))
    }
}

final class Qwen3OmniAttention: Module {
    @ModuleInfo(key: "q_proj") var query: Linear
    @ModuleInfo(key: "k_proj") var key: Linear
    @ModuleInfo(key: "v_proj") var value: Linear
    @ModuleInfo(key: "o_proj") var output: Linear
    @ModuleInfo(key: "q_norm") var queryNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var keyNorm: RMSNorm
    let config: Qwen3OmniConfiguration.Text
    let rotary: Qwen3VLRotaryEmbedding

    init(_ config: Qwen3OmniConfiguration.Text) {
        self.config = config
        self._query.wrappedValue = Linear(config.hiddenSize, config.numAttentionHeads * config.headDim, bias: false)
        self._key.wrappedValue = Linear(config.hiddenSize, config.numKeyValueHeads * config.headDim, bias: false)
        self._value.wrappedValue = Linear(config.hiddenSize, config.numKeyValueHeads * config.headDim, bias: false)
        self._output.wrappedValue = Linear(config.numAttentionHeads * config.headDim, config.hiddenSize, bias: false)
        self._queryNorm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)
        self._keyNorm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)
        rotary = Qwen3VLRotaryEmbedding(dim: config.headDim, base: config.ropeTheta,
                                       mropeSection: config.ropeScaling.mropeSection)
    }

    func callAsFunction(_ x: MLXArray, positions: MLXArray) -> MLXArray {
        let batch = x.dim(0), count = x.dim(1), dim = config.headDim
        let query = queryNorm(self.query(x).reshaped(batch, count, config.numAttentionHeads, dim)).transposed(0, 2, 1, 3)
        let key = keyNorm(self.key(x).reshaped(batch, count, config.numKeyValueHeads, dim)).transposed(0, 2, 1, 3)
        let value = self.value(x).reshaped(batch, count, config.numKeyValueHeads, dim).transposed(0, 2, 1, 3)
        let (cos, sin) = rotary(positionIds: positions, dtype: x.dtype)
        let rotated = applyRotaryPosEmb(query, key, cos: cos, sin: sin)
        let attended = MLXFast.scaledDotProductAttention(queries: rotated.0, keys: rotated.1, values: value,
                                                        scale: 1 / sqrt(Float(dim)), mask: .causal)
        return output(attended.transposed(0, 2, 1, 3).reshaped(batch, count, config.numAttentionHeads * dim))
    }
}

/// BF16 or affine Q4 expert banks, with bounded routed-matrix temporaries.
final class Qwen3OmniExperts: Module {
    @ModuleInfo var gate: Linear
    @ModuleInfo(key: "gate_proj") var gateProjection: Q35SwitchLinear
    @ModuleInfo(key: "up_proj") var upProjection: Q35SwitchLinear
    @ModuleInfo(key: "down_proj") var downProjection: Q35SwitchLinear
    let config: Qwen3OmniConfiguration.Text

    init(_ config: Qwen3OmniConfiguration.Text, quantization: Qwen3OmniConfiguration.ExpertQuantization?) {
        self.config = config
        self._gate.wrappedValue = Linear(config.hiddenSize, config.numExperts, bias: false)
        func projection(_ input: Int, _ output: Int) -> Q35SwitchLinear {
            Q35SwitchLinear(inputDims: input, outputDims: output, numExperts: config.numExperts,
                            groupSize: quantization?.groupSize ?? 64, bits: quantization?.bits ?? 4, quantized: quantization != nil, bias: false)
        }
        self._gateProjection.wrappedValue = projection(config.hiddenSize, config.moeIntermediateSize)
        self._upProjection.wrappedValue = projection(config.hiddenSize, config.moeIntermediateSize)
        self._downProjection.wrappedValue = projection(config.moeIntermediateSize, config.hiddenSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let flat = x.reshaped(-1, config.hiddenSize)
        var outputs: [MLXArray] = []
        for start in stride(from: 0, to: flat.dim(0), by: 8) {
            let chunk = flat[start..<min(start + 8, flat.dim(0))]
            let logits = gate(chunk)
            let probabilities = softmax(logits.asType(.float32), axis: -1)
            let indices = argPartition(-probabilities, kth: config.numExpertsPerTok - 1, axis: -1)[0..., 0..<config.numExpertsPerTok]
            var scores = takeAlong(probabilities, indices, axis: -1)
            if config.normTopkProb { scores = scores / scores.sum(axis: -1, keepDims: true) }
            scores = scores.asType(logits.dtype)
            let input = chunk.expandedDimensions(axes: [1, 2])
            let activated = silu(gateProjection.applyFlat(input, indices: indices, sortedIndices: false))
                * upProjection.applyFlat(input, indices: indices, sortedIndices: false)
            let routed = downProjection.applyFlat(activated, indices: indices, sortedIndices: false).squeezed(axis: -2)
            let result = (routed * scores.expandedDimensions(axis: -1)).sum(axis: 1)
            MLX.eval(result)
            outputs.append(result)
        }
        return concatenated(outputs, axis: 0).reshaped(x.shape)
    }
}
