import Foundation

/// High level facts pulled from the Hugging Face model card and configs.
public enum ZImageModelMetadata {
  public static let repository = "Tongyi-MAI/Z-Image-Turbo"
  public static let license = "apache-2.0"
  public static let parameters = "6B"

  public static let scheduler = "FlowMatchEulerDiscreteScheduler"
  public static let transformer = "ZImageTransformer2DModel"
  public static let textEncoder = "Qwen3"
  public static let tokenizer = "Qwen2Tokenizer"
  public static let vae = "AutoencoderKL"

  public static let recommendedWidth = 1024
  public static let recommendedHeight = 1024
  public static let recommendedInferenceSteps = 9 // gives 8 DiT forwards
  public static let recommendedGuidanceScale: Float = 0.0 // Turbo is distilled for zero CFG

  public enum VAE {
    public static let latentChannels = 16
    public static let scalingFactor: Float = 0.3611
    public static let shiftFactor: Float = 0.1159
    public static let sampleSize = 1024
    public static let blockOutChannels: [Int] = [128, 256, 512, 512]
  }

  public enum Transformer {
    public static let hiddenSize = 3840
    public static let layers = 30
    public static let refinerLayers = 2
    public static let heads = 30
    public static let kvHeads = 30
    public static let timestepScale: Float = 1000.0
    public static let ropeTheta: Float = 256.0
    public static let inChannels = 16
    public static let axesDims = [32, 48, 48]
    public static let axesLens = [1536, 512, 512]
  }

  public enum TextEncoder {
    public static let hiddenSize = 2560
    public static let layers = 36
    public static let heads = 32
    public static let kvHeads = 8
    public static let vocabSize = 151_936
    public static let maxPositionEmbeddings = 40_960
    public static let intermediateSize = 9_728
    public static let ropeTheta: Float = 1_000_000
  }
}
