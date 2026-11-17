import Foundation
import MereRunKVCache
import MereRunQwenModel
import MLX
import MLXNN

private final class HaruAudioProjection: Module {
    let fc1: Linear
    let norm: RMSNorm
    let fc2: Linear

    init(latent: Int, width: Int) {
        fc1 = Linear(latent, width)
        norm = RMSNorm(dimensions: width, eps: 1e-6)
        fc2 = Linear(width, width)
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        fc2(norm(fc1(input)))
    }
}

package final class HaruModel: Module {
    let lm: Q35Model
    let codec: HaruCodec
    let head: HaruMeanFlow
    @ModuleInfo(key: "model_audio_proj") private var modelAudio: HaruAudioProjection
    @ModuleInfo(key: "user_audio_proj") private var userAudio: HaruAudioProjection
    @ParameterInfo(key: "latent_scale") private var scale: MLXArray
    @ParameterInfo(key: "latent_bias") private var bias: MLXArray
    let config: HaruConfiguration

    package init(config: HaruConfiguration) {
        self.config = config
        self.lm = Q35Model(config: config.lm)
        self.codec = HaruCodec(config: config.codec)
        self.head = HaruMeanFlow(config: config.head)
        let width = config.lm.textConfig.hiddenSize
        _modelAudio.wrappedValue = HaruAudioProjection(latent: config.latentDim, width: width)
        _userAudio.wrappedValue = HaruAudioProjection(latent: config.latentDim, width: width)
        _scale.wrappedValue = MLXArray(Float(1))
        _bias.wrappedValue = MLXArray(Float(0))
    }

    func makeCache() -> [Q35LayerCache?] {
        config.lm.textConfig.layerTypes.map { type in
            type == "full_attention" ? .full(KVCacheSimple()) : .linear(Q35LinearCache())
        }
    }

    var dtype: DType { modelAudio.fc1.weight.dtype }

    func embeddings(_ tokens: MLXArray, model: MLXArray? = nil, user: MLXArray? = nil) -> MLXArray {
        var hidden = lm.embeddings(for: tokens)
        if let model { hidden = hidden + modelAudio(((model + bias) * scale).asType(dtype)) }
        if let user { hidden = hidden + userAudio(((user + bias) * scale).asType(dtype)) }
        return hidden
    }

    func rawLatent(_ normalized: MLXArray) -> MLXArray {
        normalized.asType(.float32) / scale - bias
    }

    package static func load(from root: URL) throws -> HaruModel {
        let config = try HaruConfiguration.load(from: root)
        let model = HaruModel(config: config)
        let weights = try MLX.loadArrays(url: root.appending(path: "model.safetensors"))
        try model.install(weights)
        model.train(false)
        return model
    }

    func install(_ weights: [String: MLXArray]) throws {
        let backbone = lm.hybridCheckpointParameters().map { ("lm." + $0.0, $0.1) }
        let audio = parameters().flattened().filter { !$0.0.hasPrefix("lm.") }
        let expected = backbone + audio
        var updates: [(String, MLXArray)] = []
        var unused = Set(weights.keys)
        for (key, parameter) in expected {
            let source = key.hasPrefix("lm.model.") ? "lm." + key.dropFirst("lm.model.".count) : key
            guard let weight = weights[source], weight.dtype.isFloatingPoint else {
                throw HaruError.checkpointParameter(source)
            }
            let value = try Self.map(weight, source: source)
            guard value.shape == parameter.shape else { throw HaruError.checkpointParameter(source) }
            updates.append((key, value))
            unused.remove(source)
        }
        if let extra = unused.sorted().first { throw HaruError.checkpointParameter(extra) }
        // Finish the lazy source reads before submitting layout conversions.
        MLX.eval(Array(weights.values))
        try update(parameters: ModuleParameters.unflattened(updates), verify: [.noUnusedKeys, .shapeMismatch])
        MLX.eval(updates.map(\.1))
    }

    private static func map(_ weight: MLXArray, source: String) throws -> MLXArray {
        if source == "latent_scale" || source == "latent_bias" ||
            source == "lm.norm.weight" || source.hasSuffix("_layernorm.weight") ||
            source.hasSuffix(".q_norm.weight") || source.hasSuffix(".k_norm.weight") {
            return weight.asType(.float32)
        }
        if source.hasSuffix(".linear_attn.conv1d.weight") {
            guard weight.ndim == 3 else { throw HaruError.checkpointParameter(source) }
            return weight.transposed(0, 2, 1)
        }
        guard source.hasPrefix("codec."), source.hasSuffix(".weight"), weight.ndim == 3 else {
            return weight
        }
        let (first, second, third) = weight.shape3
        if source.hasPrefix("codec.decoder.up.") {
            return weight.transposed(1, 2, 0).reshaped(second * third, first)
        }
        return weight.transposed(0, 2, 1).reshaped(first, second * third)
    }
}
