// Native Swift/MLX reimplementation of the pinned LiquidAI D1 references.
// Modified for mere.run; see THIRD_PARTY_NOTICES.md and licenses/d1-LFM-OPEN-LICENSE.txt.
import Foundation

public struct D1VLVisionConfig: Decodable, Sendable, Hashable {
    public let modelType: String
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numChannels: Int
    public let numPatches: Int
    public let patchSize: Int
    public let layerNormEpsilon: Float
    public let visionUseHead: Bool?

    private enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numChannels = "num_channels"
        case numPatches = "num_patches"
        case patchSize = "patch_size"
        case layerNormEpsilon = "layer_norm_eps"
        case visionUseHead = "vision_use_head"
    }
}
