// Native implementation of the Salesforce SOV-Transcribe-1 audio tower (Apache-2.0).
import MLX
import MLXFast
import MLXNN

package final class SOVAudioAttention: Module {
    @ModuleInfo(key: "q_proj") var query: Linear
    @ModuleInfo(key: "k_proj") var key: Linear
    @ModuleInfo(key: "v_proj") var value: Linear
    @ModuleInfo(key: "out_proj") var output: Linear
    let heads: Int
    let headDim: Int

    init(config: SOVAudioConfig) {
        heads = config.encoderAttentionHeads
        headDim = config.dModel / heads
        _query.wrappedValue = Linear(config.dModel, config.dModel)
        _key.wrappedValue = Linear(config.dModel, config.dModel, bias: false)
        _value.wrappedValue = Linear(config.dModel, config.dModel)
        _output.wrappedValue = Linear(config.dModel, config.dModel)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        func split(_ x: MLXArray) -> MLXArray { x.reshaped(x.dim(0), x.dim(1), heads, headDim).transposed(0, 2, 1, 3) }
        let attended = MLXFast.scaledDotProductAttention(
            queries: split(query(x)), keys: split(key(x)), values: split(value(x)),
            scale: 1 / Float(headDim).squareRoot(), mask: .none
        )
        return output(attended.transposed(0, 2, 1, 3).reshaped(x.shape))
    }
}

package final class SOVAudioLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: SOVAudioAttention
    @ModuleInfo(key: "self_attn_layer_norm") var attentionNorm: LayerNorm
    @ModuleInfo(key: "final_layer_norm") var finalNorm: LayerNorm
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    private var dynamicFirst: SOVDynamicInt8?
    private var dynamicSecond: SOVDynamicInt8?

    func prepareDynamicInt8() throws {
        dynamicFirst = try SOVDynamicInt8(weight: fc1.weight, rotate: false)
        dynamicSecond = try SOVDynamicInt8(weight: fc2.weight, rotate: true)
    }

    init(config: SOVAudioConfig) {
        _attention.wrappedValue = SOVAudioAttention(config: config)
        _attentionNorm.wrappedValue = LayerNorm(dimensions: config.dModel, eps: 1e-5)
        _finalNorm.wrappedValue = LayerNorm(dimensions: config.dModel, eps: 1e-5)
        _fc1.wrappedValue = Linear(config.dModel, config.encoderFFNDim)
        _fc2.wrappedValue = Linear(config.encoderFFNDim, config.dModel)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x + attention(attentionNorm(x))
        let normalized = finalNorm(residual)
        let first = dynamicFirst.map { $0(normalized, bias: fc1.bias!) } ?? fc1(normalized)
        let activated = gelu(first)
        return residual + (dynamicSecond.map { $0(activated, bias: fc2.bias!) } ?? fc2(activated))
    }
}

package final class SOVAudioEncoder: Module {
    @ModuleInfo(key: "conv1") var conv1: Conv1d
    @ModuleInfo(key: "conv2") var conv2: Conv1d
    @ModuleInfo(key: "embed_positions") var positions: Embedding
    @ModuleInfo(key: "layers") var layers: [SOVAudioLayer]
    @ModuleInfo(key: "layer_norm") var norm: LayerNorm
    let config: SOVAudioConfig

    package init(config: SOVAudioConfig) {
        self.config = config
        _conv1.wrappedValue = Conv1d(inputChannels: config.numMelBins, outputChannels: config.dModel, kernelSize: 3, padding: 1)
        _conv2.wrappedValue = Conv1d(inputChannels: config.dModel, outputChannels: config.dModel, kernelSize: 3, stride: 2, padding: 1)
        _positions.wrappedValue = Embedding(embeddingCount: config.maxSourcePositions, dimensions: config.dModel)
        _layers.wrappedValue = (0..<config.encoderLayers).map { _ in SOVAudioLayer(config: config) }
        _norm.wrappedValue = LayerNorm(dimensions: config.dModel, eps: 1e-5)
        super.init()
    }

    /// Pack once, after loading the immutable checkpoint parameters.
    package func prepareDynamicInt8() throws {
        for layer in layers { try layer.prepareDynamicInt8() }
    }

    /// Whisper features: [windows, mel bins, frames], including padded audio.
    package func callAsFunction(_ mel: MLXArray) throws -> MLXArray {
        guard mel.ndim == 3, mel.dim(1) == config.numMelBins, mel.dim(2) == config.maxSourcePositions * 2 else {
            throw SOVTranscribeError.invalid("SOV-Transcribe expects complete 30-second mel windows.")
        }
        var x = gelu(conv2(gelu(conv1(mel.transposed(0, 2, 1).asType(conv1.weight.dtype)))))
        x = (x + positions.weight).asType(x.dtype)
        for layer in layers {
            try Task.checkCancellation()
            x = layer(x)
            eval(x)
        }
        return norm(x)
    }
}

package final class SOVAudioProjector: Module {
    @ModuleInfo(key: "linear_1") var first: Linear
    @ModuleInfo(key: "linear_2") var second: Linear
    let factor: Int

    package init(config: SOVTranscribeConfig) {
        factor = config.downsampleFactor
        _first.wrappedValue = Linear(config.audioConfig.dModel * factor, config.textConfig.hiddenSize, bias: false)
        _second.wrappedValue = Linear(config.textConfig.hiddenSize, config.textConfig.hiddenSize, bias: false)
        super.init()
    }

    package func callAsFunction(_ audio: MLXArray) -> MLXArray {
        let stacked = audio.reshaped(audio.dim(0), audio.dim(1) / factor, audio.dim(2) * factor)
        return second(gelu(first(stacked)))
    }
}
