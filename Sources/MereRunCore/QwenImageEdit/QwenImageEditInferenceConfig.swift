import Foundation

public struct QwenImageEditInferenceConfig: Sendable, Hashable {
    public var width: Int
    public var height: Int
    public var numInferenceSteps: Int
    public var guidanceScale: Float
    public var seed: UInt64?

    public init(
        width: Int = 1024,
        height: Int = 1024,
        numInferenceSteps: Int = 50,
        guidanceScale: Float = 4.0,
        seed: UInt64? = nil
    ) {
        // Round to multiple of 16 (VAE requirement)
        self.width = Self.roundToMultipleOf16(width)
        self.height = Self.roundToMultipleOf16(height)
        self.numInferenceSteps = max(numInferenceSteps, 1)
        self.guidanceScale = guidanceScale
        self.seed = seed
    }

    /// Latent dimensions (after VAE encoding with 8x compression)
    public var latentHeight: Int { height / 8 }
    public var latentWidth: Int { width / 8 }

    /// Image sequence length for dynamic shifting (patches = latent_h * latent_w / patch_size^2)
    /// Assuming patch size of 2 for Qwen-Image
    public var imageSeqLen: Int {
        (latentHeight / 2) * (latentWidth / 2)
    }

    private static func roundToMultipleOf16(_ value: Int) -> Int {
        16 * (max(value, 16) / 16)
    }
}
