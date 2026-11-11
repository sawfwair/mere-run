import Foundation
import MLX

/// The original (non-Diffusers-remapped) H3-Image checkpoint configuration.
public struct MiniMaxH3ImageConfiguration: Decodable, Sendable {
    public let hiddenSize: Int
    public let layerCount: Int
    public let refinerLayerCount: Int
    public let headCount: Int
    public let headDimension: Int
    public let feedForwardSize: Int
    public let latentChannels: Int
    public let patchSize: [Int]
    public let textDimension: Int
    public let frequencyDimension: Int
    public let timeHiddenSize: Int
    public let timeDimension: Int
    public let ropeFrequencyCount: Int
    public let audioEnabled: Bool

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size", layerCount = "num_layers"
        case refinerLayerCount = "token_refiner_num_layers"
        case headCount = "num_attention_heads", headDimension = "attention_head_dim"
        case feedForwardSize = "ffn_hidden_size", latentChannels = "latents_dim", patchSize = "patch_size"
        case textDimension = "text_dim", frequencyDimension = "timestep_input_dim"
        case timeHiddenSize = "time_embed_hidden_size", timeDimension = "time_embed_dim"
        case ropeFrequencyCount = "rope_inv_freq_len", audioEnabled = "audio_enabled"
    }

    public var isReleasedArchitecture: Bool {
        hiddenSize == 5_376 && layerCount == 50 && refinerLayerCount == 2
            && headCount == 56 && headDimension == 128 && feedForwardSize == 14_336
            && latentChannels == 24 && patchSize == [1, 2, 2] && textDimension == 5_120
            && frequencyDimension == 256 && timeHiddenSize == 5_376 && timeDimension == 2_688
            && ropeFrequencyCount == 16 && !audioEnabled
    }

    public var transformer: MiniMaxH3TransformerConfiguration {
        .init(hiddenSize: hiddenSize, layerCount: layerCount, refinerLayerCount: refinerLayerCount,
              attentionHeadCount: headCount, attentionHeadDimension: headDimension,
              feedForwardSize: feedForwardSize, videoLatentChannels: latentChannels,
              patchSize: patchSize, textDimension: textDimension, timeFrequencyDimension: frequencyDimension,
              timeEmbeddingHiddenSize: timeHiddenSize, timeEmbeddingDimension: timeDimension,
              ropeFrequencyCount: ropeFrequencyCount)
    }
}

public enum MiniMaxH3ImageSampling {
    public static let steps = 6
    public static let referenceTimestep: Float = 0.999
    /// Round through the released FP32 timestep, then back to sigma before Euler integration.
    public static let timesteps: [Float] = [1.0, 0.9675236940383911, 0.8875485062599182,
                                          0.75, 0.5327739119529724, 0.23200887441635132]
        .map { Float(1 - $0) }
    public static let sigmas: [Float] = timesteps.map { 1 - $0 } + [0]

    /// H3 predicts data-ward velocity; decreasing sigma moves along, rather than against, it.
    public static func step(sample: MLXArray, velocity: MLXArray, index: Int) -> MLXArray {
        sample.asType(.float32) + (sigmas[index] - sigmas[index + 1]) * velocity.asType(.float32)
    }
}
