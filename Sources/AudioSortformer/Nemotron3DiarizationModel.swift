import Foundation
import MLX
import MLXNN

/// NVIDIA Nemotron 3 Diarization's feature-stacking RoPE encoder.
///
/// This is a separate graph from the FastConformer-based four-speaker Sortformer.
/// Weight names mirror the pinned NeMo state dict after Conv1d layout conversion.
/// Architecture reference: NVIDIA/NeMo cf724ac337d1ebc7d0dda1e23fb80916f52927a5.
private final class Nemotron3FeatureStacking: Module {
    @ModuleInfo var proj: Linear

    override init() {
        _proj.wrappedValue = Linear(128 * 8, 512, bias: false)
    }

    func callAsFunction(_ features: MLXArray) -> MLXArray {
        var frames = features.transposed(0, 2, 1)
        let remainder = frames.dim(1) % 8
        if remainder != 0 {
            frames = MLX.padded(
                frames,
                widths: [.init((0, 0)), .init((0, 8 - remainder)), .init((0, 0))]
            )
        }
        return proj(frames.reshaped(frames.dim(0), frames.dim(1) / 8, 128 * 8))
    }
}

private final class Nemotron3Attention: Module {
    @ModuleInfo(key: "w_qkv") var wQKV: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear
    private let rope = RoPE(dimensions: 64, traditional: false, base: 10_000)

    override init() {
        _wQKV.wrappedValue = Linear(512, 1_536, bias: false)
        _outProj.wrappedValue = Linear(512, 512)
    }

    func callAsFunction(_ value: MLXArray) -> MLXArray {
        let batch = value.dim(0)
        let frames = value.dim(1)
        let qkv = MLX.split(wQKV(value), parts: 3, axis: -1)
        let query = rope(qkv[0].reshaped(batch, frames, 8, 64).transposed(0, 2, 1, 3), offset: 0)
        let key = rope(qkv[1].reshaped(batch, frames, 8, 64).transposed(0, 2, 1, 3), offset: 0)
        let values = qkv[2].reshaped(batch, frames, 8, 64).transposed(0, 2, 1, 3)
        let scores = MLX.matmul(query * Float(0.125), key.transposed(0, 1, 3, 2))
        let attended = MLX.matmul(softmax(scores.asType(.float32), axis: -1).asType(values.dtype), values)
        return outProj(attended.transposed(0, 2, 1, 3).reshaped(batch, frames, 512))
    }
}

private final class Nemotron3FeedForwardNet: Module {
    @ModuleInfo var first: Linear
    @ModuleInfo var second: Linear

    override init() {
        _first.wrappedValue = Linear(512, 2_048)
        _second.wrappedValue = Linear(2_048, 512)
    }

    func callAsFunction(_ value: MLXArray) -> MLXArray {
        second(gelu(first(value)))
    }
}

private final class Nemotron3FeedForward: Module {
    @ModuleInfo var net: Nemotron3FeedForwardNet

    override init() {
        _net.wrappedValue = Nemotron3FeedForwardNet()
    }

    func callAsFunction(_ value: MLXArray) -> MLXArray { net(value) }
}

private final class Nemotron3TransformerBlock: Module {
    @ModuleInfo var norm1: LayerNorm
    @ModuleInfo var attn: Nemotron3Attention
    @ModuleInfo var norm2: LayerNorm
    @ModuleInfo var ffn: Nemotron3FeedForward

    override init() {
        _norm1.wrappedValue = LayerNorm(dimensions: 512)
        _attn.wrappedValue = Nemotron3Attention()
        _norm2.wrappedValue = LayerNorm(dimensions: 512)
        _ffn.wrappedValue = Nemotron3FeedForward()
    }

    func callAsFunction(_ value: MLXArray) -> MLXArray {
        let attended = value + attn(norm1(value))
        return attended + ffn(norm2(attended))
    }
}

private final class Nemotron3Encoder: Module {
    @ModuleInfo(key: "pre_encode") var preEncode: Nemotron3FeatureStacking
    @ModuleInfo(key: "embed_norm") var embedNorm: LayerNorm
    var layers: [Nemotron3TransformerBlock]
    @ModuleInfo(key: "final_norm") var finalNorm: LayerNorm

    override init() {
        _preEncode.wrappedValue = Nemotron3FeatureStacking()
        _embedNorm.wrappedValue = LayerNorm(dimensions: 512)
        layers = (0..<31).map { _ in Nemotron3TransformerBlock() }
        _finalNorm.wrappedValue = LayerNorm(dimensions: 512)
    }

    func callAsFunction(_ value: MLXArray) -> MLXArray {
        var hidden = embedNorm(value)
        for layer in layers {
            hidden = layer(hidden)
        }
        return finalNorm(hidden)
    }
}

private final class Nemotron3SpeakerHead: Module {
    @ModuleInfo(key: "encoder_proj") var encoderProj: Linear
    @ModuleInfo(key: "subpixel_upsample") var subpixelUpsample: Conv1d
    @ModuleInfo(key: "first_hidden_to_hidden") var firstHiddenToHidden: Linear
    @ModuleInfo(key: "single_hidden_to_spks") var singleHiddenToSpks: Linear
    @ParameterInfo(key: "learnable_sil_emb") var learnableSilEmb: MLXArray

    override init() {
        _encoderProj.wrappedValue = Linear(512, 192)
        _subpixelUpsample.wrappedValue = Conv1d(
            inputChannels: 192, outputChannels: 1_536, kernelSize: 3, padding: 1
        )
        _firstHiddenToHidden.wrappedValue = Linear(192, 192)
        _singleHiddenToSpks.wrappedValue = Linear(192, 8)
        _learnableSilEmb.wrappedValue = MLXArray.zeros([512])
    }

    func probabilities(_ hidden: MLXArray) -> MLXArray {
        let projected = subpixelUpsample(encoderProj(hidden))
        let upsampled = projected.reshaped(projected.dim(0), projected.dim(1) * 8, 192)
        return sigmoid(singleHiddenToSpks(relu(firstHiddenToHidden(relu(upsampled)))))
    }
}

public final class Nemotron3DiarizationModel: Module {
    @ModuleInfo private var encoder: Nemotron3Encoder
    @ModuleInfo(key: "sortformer_modules") private var sortformerModules: Nemotron3SpeakerHead

    public override init() {
        _encoder.wrappedValue = Nemotron3Encoder()
        _sortformerModules.wrappedValue = Nemotron3SpeakerHead()
    }

    public func preEncode(_ features: MLXArray) -> MLXArray {
        encoder.preEncode(features.asType(encoder.preEncode.proj.weight.dtype))
    }

    /// Accepts a sequence of pre-encoded 80 ms frames, including speaker cache and FIFO context.
    public func probabilities(preEncoded: MLXArray) -> MLXArray {
        sortformerModules.probabilities(encoder(preEncoded))
    }

    public var silenceEmbedding: MLXArray { sortformerModules.learnableSilEmb }

    public static func compatibleWeights(_ source: [String: MLXArray]) -> [String: MLXArray] {
        var target: [String: MLXArray] = [:]
        for (key, value) in source {
            guard key.hasPrefix("encoder.") || key.hasPrefix("sortformer_modules.") else { continue }
            guard !key.hasPrefix("sortformer_modules.activity_head.")
                    && !key.hasPrefix("sortformer_modules.hidden_to_spks.") else { continue }
            let mapped = key
                .replacingOccurrences(of: ".ffn.net.0.", with: ".ffn.net.first.")
                .replacingOccurrences(of: ".ffn.net.3.", with: ".ffn.net.second.")
            if key == "sortformer_modules.subpixel_upsample.weight" {
                target[mapped] = value.transposed(0, 2, 1)
            } else {
                target[mapped] = value
            }
        }
        return target
    }
}
