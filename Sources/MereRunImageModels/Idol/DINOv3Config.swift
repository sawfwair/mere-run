import Foundation

// MARK: - DINOv3 Configuration

public struct DINOv3Config: Sendable {
    public var hiddenSize: Int = 4096
    public var intermediateSize: Int = 8192  // SwiGLU intermediate (2x hidden)
    public var numHiddenLayers: Int = 40
    public var numAttentionHeads: Int = 32
    public var patchSize: Int = 16
    public var imageSize: Int = 224  // Matches DiffSynth DINOv3 processor/config
    public var numChannels: Int = 3
    public var layerNormEps: Float = 1e-5
    public var numRegisterTokens: Int = 4
    public var useRoPE: Bool = true
    public var ropeTheta: Float = 100.0
    /// The original Z-Image-i2L checkpoint uses a gated SwiGLU MLP. Standard
    /// DINOv3 ViT checkpoints, including the ViT-L/16 encoder used by
    /// TRELLIS.2, use a single GELU projection instead.
    public var useGatedMLP: Bool = true
    public var queryBias: Bool = false
    public var keyBias: Bool = false
    public var valueBias: Bool = false
    public var projectionBias: Bool = true
    public var mlpBias: Bool = true

    public var headDim: Int { hiddenSize / numAttentionHeads }

    public init() {}
}
