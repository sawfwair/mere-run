import Foundation

// MARK: - SigLIP2 Configuration

public struct SigLIP2Config: Sendable {
    public var hiddenSize: Int = 1536
    public var intermediateSize: Int = 6144
    public var numHiddenLayers: Int = 40
    public var numAttentionHeads: Int = 16
    public var patchSize: Int = 16
    public var imageSize: Int = 384  // Native resolution
    public var numChannels: Int = 3
    public var layerNormEps: Float = 1e-6

    public var headDim: Int { hiddenSize / numAttentionHeads }
    public var numPatches: Int { (imageSize / patchSize) * (imageSize / patchSize) }

    public init() {}
}

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

    public var headDim: Int { hiddenSize / numAttentionHeads }

    public init() {}
}

// MARK: - Z-Image-i2L Configuration

public struct ZImageI2LConfig: Sendable {
    // Input dimensions (SigLIP + DINOv3 concatenated)
    public var siglipDim: Int = 1536
    public var dinov3Dim: Int = 4096
    public var inputDim: Int { siglipDim + dinov3Dim }  // 5632

    // i2L bottleneck (CompressedMLP mid-dim, called `compress_dim` in DiffSynth).
    //
    // The Z-Image-i2L weights use `proj_in: Linear(inputDim -> compressDim)` with weight shape:
    //   [compressDim, inputDim] == [128, 5632]
    public var compressDim: Int = 128

    // Output LoRA rank (r). Z-Image-i2L is trained for rank=4.
    public var loraRank: Int = 4

    // Z-Image Turbo transformer layout.
    // - `layers`: 30 transformer blocks
    // - `context_refiner`: 2 blocks
    // - `noise_refiner`: 2 blocks
    public var numLayers: Int = 30
    public var numRefinerLayers: Int = 2

    // Residual path exists in DiffSynth, but the public Z-Image-i2L weights shipped without it.
    public var useResidual: Bool = false

    public init() {}
}

// MARK: - Combined Model Configs

public struct ZImageI2LModelConfigs: Sendable {
    public var siglip2: SigLIP2Config
    public var dinov3: DINOv3Config
    public var i2l: ZImageI2LConfig

    public init(
        siglip2: SigLIP2Config = SigLIP2Config(),
        dinov3: DINOv3Config = DINOv3Config(),
        i2l: ZImageI2LConfig = ZImageI2LConfig()
    ) {
        self.siglip2 = siglip2
        self.dinov3 = dinov3
        self.i2l = i2l
    }
}
