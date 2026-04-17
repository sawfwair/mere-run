import Foundation
import MLX
import MLXNN
import MLXFast

/// Vision/text support types used by the LightOn OCR runtime.
/// These stay outside the actor so the actor file can focus on request flow.

final class VisionProjection: Module {
    @ModuleInfo(key: "linear_1") private var linear1: Linear
    @ModuleInfo(key: "linear_2") private var linear2: Linear
    @ModuleInfo(key: "norm") private var norm: RMSNorm
    @ModuleInfo(key: "patch_merger") private var patchMerger: PatchMerger

    init(visionHiddenSize: Int, textHiddenSize: Int, spatialMergeSize: Int) {
        self._norm.wrappedValue = RMSNorm(dimensions: visionHiddenSize, eps: 1e-6)
        self._patchMerger.wrappedValue = PatchMerger(hiddenSize: visionHiddenSize, spatialMergeSize: spatialMergeSize)
        self._linear1.wrappedValue = Linear(visionHiddenSize, textHiddenSize, bias: false)
        self._linear2.wrappedValue = Linear(textHiddenSize, textHiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray, gridH: Int, gridW: Int, spatialMergeSize: Int) -> MLXArray {
        var h = norm(x)
        h = patchMerger(h, gridH: gridH, gridW: gridW, spatialMergeSize: spatialMergeSize)
        h = linear1(h)
        h = gelu(h)
        h = linear2(h)
        return h
    }
}

final class PatchMerger: Module {
    @ModuleInfo(key: "merging_layer") private var mergingLayer: Linear

    init(hiddenSize: Int, spatialMergeSize: Int) {
        self._mergingLayer.wrappedValue = Linear(hiddenSize * spatialMergeSize * spatialMergeSize, hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray, gridH: Int, gridW: Int, spatialMergeSize: Int) -> MLXArray {
        let batch = x.dim(0)
        let hiddenSize = x.dim(2)
        let mergedH = gridH / spatialMergeSize
        let mergedW = gridW / spatialMergeSize
        if mergedH < 1 || mergedW < 1 {
            return x
        }

        let usableH = mergedH * spatialMergeSize
        let usableW = mergedW * spatialMergeSize
        var h = x.reshaped(batch, gridH, gridW, hiddenSize)
        if usableH < gridH || usableW < gridW {
            h = h[0..., 0..<usableH, 0..<usableW, 0...]
        }
        h = h.reshaped(batch, mergedH, spatialMergeSize, mergedW, spatialMergeSize, hiddenSize)
        h = h.transposed(0, 1, 3, 2, 4, 5)
        h = h.transposed(0, 1, 2, 5, 3, 4)
        h = h.reshaped(batch, mergedH * mergedW, spatialMergeSize * spatialMergeSize * hiddenSize)
        return mergingLayer(h)
    }
}

struct LightOnOCRConfig: Decodable {
    let visionConfig: VisionConfig
    let textConfig: TextConfig
    let spatialMergeSize: Int

    struct VisionConfig: Decodable {
        let hiddenSize: Int
        let numHiddenLayers: Int
        let numAttentionHeads: Int
        let headDim: Int
        let intermediateSize: Int
        let patchSize: Int
        let imageSize: Int
        let numChannels: Int
        let ropeTheta: Float

        enum CodingKeys: String, CodingKey {
            case hiddenSize = "hidden_size"
            case numHiddenLayers = "num_hidden_layers"
            case numAttentionHeads = "num_attention_heads"
            case headDim = "head_dim"
            case intermediateSize = "intermediate_size"
            case patchSize = "patch_size"
            case imageSize = "image_size"
            case numChannels = "num_channels"
            case ropeTheta = "rope_theta"
        }
    }

    struct TextConfig: Decodable {
        let vocabSize: Int
        let hiddenSize: Int
        let numHiddenLayers: Int
        let numAttentionHeads: Int
        let numKeyValueHeads: Int
        let intermediateSize: Int
        let ropeTheta: Float
        let maxPositionEmbeddings: Int
        let rmsNormEps: Float?
        let headDim: Int
        let ropeScaling: RopeScaling?
        let modelType: String?

        enum CodingKeys: String, CodingKey {
            case vocabSize = "vocab_size"
            case hiddenSize = "hidden_size"
            case numHiddenLayers = "num_hidden_layers"
            case numAttentionHeads = "num_attention_heads"
            case numKeyValueHeads = "num_key_value_heads"
            case intermediateSize = "intermediate_size"
            case ropeTheta = "rope_theta"
            case maxPositionEmbeddings = "max_position_embeddings"
            case rmsNormEps = "rms_norm_eps"
            case headDim = "head_dim"
            case ropeScaling = "rope_scaling"
            case modelType = "model_type"
        }
    }

    struct RopeScaling: Decodable {
        let type: String?
        let mropeSection: [Int]?
        let mropeInterleaved: Bool?

        enum CodingKeys: String, CodingKey {
            case type
            case ropeType = "rope_type"
            case mropeSection = "mrope_section"
            case mropeInterleaved = "mrope_interleaved"
        }

        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            let explicitType = try container.decodeIfPresent(String.self, forKey: .type)
            let ropeType = try container.decodeIfPresent(String.self, forKey: .ropeType)
            self.type = explicitType ?? ropeType
            self.mropeSection = try container.decodeIfPresent([Int].self, forKey: .mropeSection)
            self.mropeInterleaved = try container.decodeIfPresent(Bool.self, forKey: .mropeInterleaved)
        }
    }

    enum CodingKeys: String, CodingKey {
        case visionConfig = "vision_config"
        case textConfig = "text_config"
        case spatialMergeSize = "spatial_merge_size"
    }
}

public enum LightOnOCRError: LocalizedError {
    case modelsNotLoaded
    case configNotFound(URL)
    case weightsNotFound(URL)
    case imageLoadFailed(URL)
    case generationFailed

    public var errorDescription: String? {
        switch self {
        case .modelsNotLoaded:
            return "LightOnOCR models not loaded"
        case .configNotFound(let url):
            return "Config not found at \(url.path)"
        case .weightsNotFound(let url):
            return "Weights not found at \(url.path)"
        case .imageLoadFailed(let url):
            return "Failed to load image: \(url.path)"
        case .generationFailed:
            return "Text generation failed"
        }
    }
}

extension QwenEncoder {
    func forwardFromEmbeddings(embeddings: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embeddings.asType(.bfloat16)

        let n = h.dim(1)
        let mask: MLXFast.ScaledDotProductAttentionMaskMode
        if let cache = cache?.first {
            mask = cache.makeMask(n: n)
        } else if n == 1 {
            mask = .none
        } else {
            mask = .causal
        }

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i], debug: false, debugMLP: false)
        }

        h = norm(h)
        return embedTokens.asLinear(h)
    }
}
