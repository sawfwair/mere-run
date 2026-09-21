import Foundation

public struct QwenImage21TransformerConfig: Codable, Sendable {
    public var attentionHeadDim: Int
    public var axesDimsRope: [Int]
    public var contextInDim: Int
    public var inChannels: Int
    public var numAttentionHeads: Int
    public var numLayers: Int
    public var outChannels: Int
    public var patchSize: Int
    public var mlpRatio: Int
    public var eps: Float
    public var causalCondition: Bool
    public var hiddenSize: Int { attentionHeadDim * numAttentionHeads }

    enum CodingKeys: String, CodingKey {
        case attentionHeadDim = "attention_head_dim", axesDimsRope = "axes_dims_rope"
        case contextInDim = "context_in_dim", inChannels = "in_channels"
        case numAttentionHeads = "num_attention_heads", numLayers = "num_layers"
        case outChannels = "out_channels", patchSize = "patch_size", mlpRatio = "mlp_ratio"
        case eps, causalCondition = "causal_condition"
    }
}

public struct QwenImage21VAEConfig: Codable, Sendable {
    public var baseDim: Int
    public var decoderBaseDim: Int
    public var dimMult: [Int]
    public var inChannels: Int
    public var outChannels: Int
    public var zDim: Int
    public var numResBlocks: Int
    public var temporalDownsample: [Bool]
    public var isResidual: Bool
    public var patchSize: Int?
    public var attnScales: [Float]
    public var scaleFactorSpatial: Int
    public var latentsMean: [Float]
    public var latentsStd: [Float]

    enum CodingKeys: String, CodingKey {
        case baseDim = "base_dim", decoderBaseDim = "decoder_base_dim", dimMult = "dim_mult"
        case inChannels = "in_channels", outChannels = "out_channels", zDim = "z_dim"
        case numResBlocks = "num_res_blocks", temporalDownsample = "temperal_downsample"
        case isResidual = "is_residual", patchSize = "patch_size", attnScales = "attn_scales"
        case scaleFactorSpatial = "scale_factor_spatial", latentsMean = "latents_mean", latentsStd = "latents_std"
    }
}

public enum QwenImage21Error: LocalizedError {
    case invalidConfiguration(String)
    case invalidWeights(String)
    case invalidLayout(String)

    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let detail): "Invalid Qwen Image 2.1 configuration: \(detail)"
        case .invalidWeights(let detail): "Invalid Qwen Image 2.1 weights: \(detail)"
        case .invalidLayout(let detail): "Invalid Qwen Image 2.1 token layout: \(detail)"
        }
    }
}

/// Layout is request-owned, including image boundaries even when two images are adjacent.
public struct QwenImage21Layout: Sendable {
    public struct Segment: Sendable {
        public let range: Range<Int>
        public let image: Bool
    }
    public let segments: [Segment]
    public let positions: [[Int]]
    public let textIndices: [Int]
    public let imageIndices: [Int]
    public let prefixCount: Int
    public let targetCount: Int
    public var count: Int { prefixCount + targetCount }

    public init(imageSlots: [Bool], imageShapes: [(height: Int, width: Int)]) throws {
        guard let target = imageShapes.last, imageShapes.allSatisfy({ $0.height > 0 && $0.width > 0 }) else {
            throw QwenImage21Error.invalidLayout("Image shapes must be positive.")
        }
        let lengths = imageShapes.map { $0.height * $0.width }
        guard lengths.allSatisfy({ $0.isMultiple(of: 4) }), imageSlots.filter({ $0 }).count * 4 == lengths.reduce(0, +) else {
            throw QwenImage21Error.invalidLayout("Each vision slot must represent four latent tokens.")
        }
        var textIndices: [Int] = [], imageIndices: [Int] = [], positions: [[Int]] = []
        var imageIDs: [Int] = []
        var imageIndex = 0, offset = 0, position = 0, frame = 0
        for (slot, isImage) in imageSlots.enumerated() {
            if !isImage {
                guard offset == 0 else { throw QwenImage21Error.invalidLayout("Interrupted image block.") }
                textIndices.append(slot)
                imageIndices.append(-1)
                positions.append([position, position, position])
                imageIDs.append(-1)
                position += 1
            } else {
                if offset == 0 { frame = position }
                let shape = imageShapes[imageIndex]
                for _ in 0..<4 {
                    textIndices.append(-1)
                    imageIndices.append(lengths.prefix(imageIndex).reduce(0, +) + offset)
                    positions.append([frame, offset / shape.width - (shape.height - shape.height / 2),
                                      offset % shape.width - (shape.width - shape.width / 2)])
                    imageIDs.append(imageIndex)
                    offset += 1
                }
                if offset == lengths[imageIndex] {
                    position += max(shape.height, shape.width)
                    imageIndex += 1
                    offset = 0
                }
            }
        }
        let targetCount = target.height * target.width
        let prefixCount = positions.count - targetCount
        guard imageIDs.suffix(targetCount).allSatisfy({ $0 == imageShapes.count - 1 }) else {
            throw QwenImage21Error.invalidLayout("Target image must be the final block.")
        }
        var segments: [Segment] = []
        var start = 0
        for end in 1...imageIDs.count where end == imageIDs.count || imageIDs[end] != imageIDs[start] {
            segments.append(Segment(range: start..<end, image: imageIDs[start] >= 0))
            start = end
        }
        self.segments = segments
        self.positions = positions
        self.textIndices = textIndices
        self.imageIndices = imageIndices
        self.prefixCount = prefixCount
        self.targetCount = targetCount
    }
}
