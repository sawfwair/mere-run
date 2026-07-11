import Foundation
import MLX
import MLXNN

// Owns the public text encoder entrypoints and multimodal replacement flow.
// Low-level RoPE math and transformer block definitions live in companion
// files so readers can follow the high-level encode path without paging
// through architecture internals.

enum QwenTextEncoderError: Error {
  case visionTowerUnavailable
  case mismatchedVisionTokenCount
}

/// Selects the single-token cached-attention implementation used by the
/// shared Qwen text stack. Native mode keeps grouped key/value heads compact
/// and lets MLX dispatch its GQA kernel. Reference mode expands the heads and
/// uses explicit float32 attention for multimodal checkpoints whose MRoPE
/// decode parity still depends on the reference formulation.
public enum QwenCachedDecodeAttentionMode: String, Sendable {
  case automatic
  case native
  case reference
}

public struct QwenTextEncoderConfiguration {
  public var vocabSize: Int
  public var hiddenSize: Int
  public var numHiddenLayers: Int
  public var numAttentionHeads: Int
  public var numKeyValueHeads: Int
  public var intermediateSize: Int
  public var ropeTheta: Float
  public var maxPositionEmbeddings: Int
  public var rmsNormEps: Float
  public var promptDropIndex: Int
  public var headDim: Int
  public var mropeSection: [Int]?
  public var mropeInterleaved: Bool
  public var useFloat32Activations: Bool
  public var cachedDecodeAttentionMode: QwenCachedDecodeAttentionMode

  public init(
    vocabSize: Int = 151_936,
    hiddenSize: Int = 2560,
    numHiddenLayers: Int = 36,
    numAttentionHeads: Int = 32,
    numKeyValueHeads: Int = 8,
    intermediateSize: Int = 9_728,
    ropeTheta: Float = 1_000_000.0,
    maxPositionEmbeddings: Int = 40_960,
    rmsNormEps: Float = 1e-6,
    promptDropIndex: Int = 0,
    headDim: Int = 128,
    mropeSection: [Int]? = nil,
    mropeInterleaved: Bool = false,
    useFloat32Activations: Bool = false,
    cachedDecodeAttentionMode: QwenCachedDecodeAttentionMode = .automatic
  ) {
    self.vocabSize = vocabSize
    self.hiddenSize = hiddenSize
    self.numHiddenLayers = numHiddenLayers
    self.numAttentionHeads = numAttentionHeads
    self.numKeyValueHeads = numKeyValueHeads
    self.intermediateSize = intermediateSize
    self.ropeTheta = ropeTheta
    self.maxPositionEmbeddings = maxPositionEmbeddings
    self.rmsNormEps = rmsNormEps
    self.promptDropIndex = promptDropIndex
    self.headDim = headDim
    self.mropeSection = mropeSection
    self.mropeInterleaved = mropeInterleaved
    self.useFloat32Activations = useFloat32Activations
    self.cachedDecodeAttentionMode = cachedDecodeAttentionMode
  }
}

public final class QwenTextEncoder: Module {

  public let configuration: QwenTextEncoderConfiguration
  @ModuleInfo(key: "encoder") var encoder: QwenEncoder
  private var visionTower: QwenVisionTower?

  public init(configuration: QwenTextEncoderConfiguration = .init()) {
    self.configuration = configuration
    self._encoder.wrappedValue = QwenEncoder(configuration: configuration)
  }

  func setVisionTower(_ tower: QwenVisionTower) {
    self.visionTower = tower
  }

  public func callAsFunction(
    inputIds: MLXArray,
    attentionMask: MLXArray? = nil
  ) -> (MLXArray, MLXArray) {
    encode(inputIds: inputIds, attentionMask: attentionMask)
  }

  public func encode(
    inputIds: MLXArray,
    attentionMask: MLXArray?,
    keepFullSequence: Bool = false
  ) -> (MLXArray, MLXArray) {
    let result = encoder.forward(
      inputIds: inputIds,
      attentionMask: attentionMask,
      outputHiddenStates: false
    )
    let hiddenStates = result.lastHiddenState
    if keepFullSequence {
      let mask = attentionMask ?? MLX.ones([hiddenStates.dim(0), hiddenStates.dim(1)], dtype: .int32)
      return (hiddenStates, mask.asType(.int32))
    }
    let processed = QwenTextEncoder.processTextEmbeddings(
      hiddenStates: hiddenStates,
      attentionMask: attentionMask,
      dropIndex: configuration.promptDropIndex
    )
    return processed
  }

  public func forwardWithHiddenStates(
    inputIds: MLXArray,
    attentionMask: MLXArray?
  ) -> (lastHiddenState: MLXArray, hiddenStates: [MLXArray]?) {
    return encoder.forward(
      inputIds: inputIds,
      attentionMask: attentionMask,
      outputHiddenStates: true
    )
  }

  public func encodeForZImage(
    inputIds: MLXArray,
    attentionMask: MLXArray?
  ) -> [MLXArray] {
    let result = encoder.forward(
      inputIds: inputIds,
      attentionMask: attentionMask,
      outputHiddenStates: true
    )

    guard let allHiddenStates = result.hiddenStates, allHiddenStates.count >= 2 else {
      return [result.lastHiddenState]
    }

    // Get second-to-last hidden state (before final norm)
    let secondToLast = allHiddenStates[allHiddenStates.count - 2]

    // Extract only the valid (non-padding) tokens for each batch item
    let batchSize = secondToLast.dim(0)
    var embeddingsList: [MLXArray] = []
    embeddingsList.reserveCapacity(batchSize)

    for i in 0..<batchSize {
      let batchEmbeds = secondToLast[i]

      if let mask = attentionMask {
        MLX.eval(mask)
        let batchMask = mask[i].asArray(Int32.self)
        let validCount = batchMask.filter { $0 != 0 }.count

        if validCount > 0 && validCount < batchMask.count {
          let validEmbeds = batchEmbeds[0..<validCount]
          embeddingsList.append(validEmbeds)
        } else {
          embeddingsList.append(batchEmbeds)
        }
      } else {
        embeddingsList.append(batchEmbeds)
      }
    }

    return embeddingsList
  }

  static func processTextEmbeddings(
    hiddenStates: MLXArray,
    attentionMask: MLXArray?,
    dropIndex: Int
  ) -> (MLXArray, MLXArray) {
    let batchSize = hiddenStates.dim(0)
    let seqLen = hiddenStates.dim(1)
    let hiddenDim = hiddenStates.dim(2)

    var mask: MLXArray
    if let attentionMask {
      mask = attentionMask
    } else {
      mask = MLX.ones([batchSize, seqLen], dtype: .int32)
    }
    if mask.dtype != .int32 {
      mask = mask.asType(.int32)
    }

    let trimmedStart = max(0, min(dropIndex, seqLen))

    // Pre-compute valid lengths to avoid .item() calls in loop
    let validLengthsArray = mask.sum(axis: 1)
    MLX.eval(validLengthsArray)
    let validLengths = validLengthsArray.asArray(Int.self)
    let trimmedLengths = validLengths.map { max(0, $0 - trimmedStart) }
    let maxTrimmedLength = trimmedLengths.max() ?? 0

    // Build padded embeddings and masks
    var paddedEmbeds: [MLXArray] = []
    paddedEmbeds.reserveCapacity(batchSize)
    var paddedMasks: [MLXArray] = []
    paddedMasks.reserveCapacity(batchSize)

    for batch in 0..<batchSize {
      let trimmedLength = trimmedLengths[batch]
      let sliceEnd = trimmedStart + trimmedLength

      var sampleEmbeds: MLXArray
      if trimmedLength > 0 {
        sampleEmbeds = hiddenStates[batch, trimmedStart..<sliceEnd, 0...]
      } else {
        sampleEmbeds = MLX.zeros([0, hiddenDim], dtype: hiddenStates.dtype)
      }

      if trimmedLength < maxTrimmedLength {
        let pad = MLX.zeros([maxTrimmedLength - trimmedLength, hiddenDim], dtype: hiddenStates.dtype)
        sampleEmbeds = MLX.concatenated([sampleEmbeds, pad], axis: 0)
      }
      paddedEmbeds.append(sampleEmbeds)

      let sampleMask: MLXArray
      if trimmedLength == 0 {
        sampleMask = MLX.zeros([maxTrimmedLength], dtype: .int32)
      } else if trimmedLength == maxTrimmedLength {
        sampleMask = MLX.ones([maxTrimmedLength], dtype: .int32)
      } else {
        let tailOnes = MLX.ones([trimmedLength], dtype: .int32)
        let leadZeros = MLX.zeros([maxTrimmedLength - trimmedLength], dtype: .int32)
        sampleMask = MLX.concatenated([tailOnes, leadZeros], axis: 0)
      }
      paddedMasks.append(sampleMask)
    }

    let promptEmbeds = MLX.stacked(paddedEmbeds, axis: 0)
    let encoderMask = MLX.stacked(paddedMasks, axis: 0)
    return (promptEmbeds, encoderMask)
  }

}

extension QwenTextEncoder {
  public func encodeJoint(
    inputIds: MLXArray,
    attentionMask: MLXArray?,
    imageTokenId: Int,
    visionStartTokenId: Int,
    placeholderGridTHW: [(Int, Int, Int)],
    spatialMergeSize: Int,
    replacements: [MLXArray],
    dropIndex dropIndexOverride: Int? = nil
  ) -> (MLXArray, MLXArray) {
    var tokenIds = inputIds
    if tokenIds.dtype != .int32 {
      tokenIds = tokenIds.asType(.int32)
    }
    var hiddenStates = encoder.embed(inputIds: tokenIds)
    let dropIndexValue = dropIndexOverride ?? configuration.promptDropIndex

    if !replacements.isEmpty {
      hiddenStates = replaceVisionTokens(
        hiddenStates: hiddenStates,
        inputIds: tokenIds,
        imageTokenId: imageTokenId,
        replacements: replacements
      )
    }

    let attentionMaskUpdated: MLXArray
    if let attentionMask {
      attentionMaskUpdated = attentionMask.asType(.int32)
    } else {
      attentionMaskUpdated = MLX.ones([hiddenStates.dim(0), hiddenStates.dim(1)], dtype: .int32)
    }

    // For joint encoding, use the standard forward path
    let result = encoder.forward(
      embeddings: hiddenStates,
      attentionMask: attentionMaskUpdated,
      outputHiddenStates: false
    )

    let processed = QwenTextEncoder.processTextEmbeddings(
      hiddenStates: result.lastHiddenState,
      attentionMask: attentionMaskUpdated,
      dropIndex: dropIndexValue
    )
    return processed
  }

  private func replaceVisionTokens(
    hiddenStates: MLXArray,
    inputIds: MLXArray,
    imageTokenId: Int,
    replacements: [MLXArray]
  ) -> MLXArray {
    let batch = hiddenStates.dim(0)
    let seqLen = hiddenStates.dim(1)

    guard let first = replacements.first else {
      return hiddenStates
    }
    var replacementTensor = replacements.count == 1 ? first : MLX.concatenated(replacements, axis: 0)
    if replacementTensor.dtype != hiddenStates.dtype {
      replacementTensor = replacementTensor.asType(hiddenStates.dtype)
    }
    let replacementCount = replacementTensor.dim(0)

    let tokenArray = inputIds.asType(.int32)
    MLX.eval(tokenArray)
    let tokenValues = tokenArray.asArray(Int32.self)

    let result = hiddenStates
    for row in 0..<batch {
      let base = row * seqLen
      var positions: [Int] = []
      positions.reserveCapacity(replacementCount)
      for position in 0..<seqLen where tokenValues[base + position] == Int32(imageTokenId) {
        positions.append(position)
      }

      precondition(
        positions.count == replacementCount,
        "[QwenTextEncoder] placeholder mismatch in row \(row): found \(positions.count), expected \(replacementCount)"
      )

      guard let start = positions.first else { continue }
      let end = start + replacementCount

      let isContiguous = positions.enumerated().allSatisfy { offset, value in
        value == start + offset
      }

      if isContiguous {
        result[row, start..<end, 0...] = replacementTensor
      } else {
        let idx = MLXArray(positions.map { Int32($0) })
        result[row, idx, 0...] = replacementTensor
      }
    }
    return result
  }
}
