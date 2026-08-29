import Foundation

public struct QwenVisionConfiguration {
  public enum MLPStyle {
    case twoLayer
    case gated
  }

  public var depth: Int
  public var embedDim: Int
  public var mlpHiddenDim: Int
  public var hiddenAct: Activation
  public var mlpStyle: MLPStyle
  public var numHeads: Int
  public var eps: Float
  public var patchSize: Int
  public var temporalPatchSize: Int
  public var spatialMergeSize: Int
  public var inChannels: Int
  public var outHiddenDim: Int
  public var windowSize: Int
  public var fullAttentionBlockIndices: [Int]
  public var patchEmbedBias: Bool
  /// Qwen3-VL: number of learned position embeddings (e.g. 2304 = 48*48)
  public var numPositionEmbeddings: Int?
  /// Qwen3-VL: use learned position embeddings + rotary (vs rotary-only for Qwen2-VL)
  public var useLearnedPosEmbed: Bool
  /// Qwen3-VL: layer indices for deepstack visual features (e.g. [5, 11, 17])
  public var deepstackVisualIndexes: [Int]

  public enum Activation {
    case geluApproximate
    case silu
  }

  public init(
    depth: Int = 32,
    embedDim: Int = 1_280,
    mlpHiddenDim: Int = 3_420,
    hiddenAct: Activation = .silu,
    mlpStyle: MLPStyle = .twoLayer,
    numHeads: Int = 16,
    eps: Float = 1e-6,
    patchSize: Int = 14,
    temporalPatchSize: Int = 2,
    spatialMergeSize: Int = 2,
    inChannels: Int = 3,
    outHiddenDim: Int = 3_584,
    windowSize: Int = 112,
    fullAttentionBlockIndices: [Int] = [7, 15, 23, 31],
    patchEmbedBias: Bool = false,
    numPositionEmbeddings: Int? = nil,
    useLearnedPosEmbed: Bool = false,
    deepstackVisualIndexes: [Int] = []
  ) {
    self.depth = depth
    self.embedDim = embedDim
    self.mlpHiddenDim = mlpHiddenDim
    self.hiddenAct = hiddenAct
    self.mlpStyle = mlpStyle
    self.numHeads = numHeads
    self.eps = eps
    self.patchSize = patchSize
    self.temporalPatchSize = temporalPatchSize
    self.spatialMergeSize = spatialMergeSize
    self.inChannels = inChannels
    self.outHiddenDim = outHiddenDim
    self.windowSize = windowSize
    self.fullAttentionBlockIndices = fullAttentionBlockIndices
    self.patchEmbedBias = patchEmbedBias
    self.numPositionEmbeddings = numPositionEmbeddings
    self.useLearnedPosEmbed = useLearnedPosEmbed
    self.deepstackVisualIndexes = deepstackVisualIndexes
  }
}
