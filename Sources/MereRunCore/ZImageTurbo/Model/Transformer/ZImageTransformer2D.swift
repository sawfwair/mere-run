import Foundation
import MLX
import MLXNN

final class ZImageModuleTable2_1<T: Module>: Module {
  @ModuleInfo(key: "2-1") var twoOne: T

  init(_ value: T) {
    self._twoOne.wrappedValue = value
    super.init()
  }
}

public final class ZImageTransformer2DModel: Module {
  public let configuration: ZImageTurboTransformerConfig
  @ModuleInfo(key: "t_embedder") var tEmbedder: ZImageTimestepEmbedder
  @ModuleInfo(key: "all_x_embedder") var allXEmbedder: ZImageModuleTable2_1<Linear>
  @ModuleInfo(key: "all_final_layer") var allFinalLayer: ZImageModuleTable2_1<ZImageFinalLayer>
  @ModuleInfo(key: "noise_refiner") public internal(set) var noiseRefiner: [ZImageTransformerBlock]
  @ModuleInfo(key: "context_refiner") public internal(set) var contextRefiner: [ZImageTransformerBlock]
  @ModuleInfo(key: "layers") public internal(set) var layers: [ZImageTransformerBlock]
  @ModuleInfo(key: "cap_embedder") var capEmbedder: (RMSNorm, Linear)

  let ropeEmbedder: ZImageRopeEmbedder
  @ParameterInfo(key: "x_pad_token") private var xPadToken: MLXArray?
  @ParameterInfo(key: "cap_pad_token") private var capPadToken: MLXArray?

  private var cache: TransformerCache?
  private var cacheKey: TransformerCacheKey?

  /// Enable gradient checkpointing to reduce memory during training.
  /// When enabled, intermediate activations are not stored during forward pass
  /// and are recomputed during backward pass.
  public var gradientCheckpointing: Bool = false

  public init(configuration: ZImageTurboTransformerConfig) {
    self.configuration = configuration
    let outSize = min(configuration.dim, 256)
    self._tEmbedder.wrappedValue = ZImageTimestepEmbedder(outSize: outSize, midSize: 1024)

    let patchSize = 2
    let fPatchSize = 1
    let inFeatures = fPatchSize * patchSize * patchSize * configuration.inChannels
    self._allXEmbedder.wrappedValue = ZImageModuleTable2_1(Linear(inFeatures, configuration.dim, bias: true))
    self._allFinalLayer.wrappedValue = ZImageModuleTable2_1(ZImageFinalLayer(
      hiddenSize: configuration.dim,
      outChannels: patchSize * patchSize * fPatchSize * configuration.inChannels
    ))

    self._capEmbedder.wrappedValue = (
      RMSNorm(dimensions: configuration.capFeatDim, eps: configuration.normEps),
      Linear(configuration.capFeatDim, configuration.dim, bias: true)
    )

    var noiseBlocks: [ZImageTransformerBlock] = []
    for layerId in 0..<configuration.nRefinerLayers {
      noiseBlocks.append(
        ZImageTransformerBlock(
          layerId: 1000 + layerId,
          dim: configuration.dim,
          nHeads: configuration.nHeads,
          nKvHeads: configuration.nKVHeads,
          normEps: configuration.normEps,
          qkNorm: configuration.qkNorm,
          modulation: true
        )
      )
    }
    self._noiseRefiner.wrappedValue = noiseBlocks

    var contextBlocks: [ZImageTransformerBlock] = []
    for layerId in 0..<configuration.nRefinerLayers {
      contextBlocks.append(
        ZImageTransformerBlock(
          layerId: layerId,
          dim: configuration.dim,
          nHeads: configuration.nHeads,
          nKvHeads: configuration.nKVHeads,
          normEps: configuration.normEps,
          qkNorm: configuration.qkNorm,
          modulation: false
        )
      )
    }
    self._contextRefiner.wrappedValue = contextBlocks

    var mainLayers: [ZImageTransformerBlock] = []
    for layerId in 0..<configuration.nLayers {
      mainLayers.append(
        ZImageTransformerBlock(
          layerId: layerId,
          dim: configuration.dim,
          nHeads: configuration.nHeads,
          nKvHeads: configuration.nKVHeads,
          normEps: configuration.normEps,
          qkNorm: configuration.qkNorm,
          modulation: true
        )
      )
    }
    self._layers.wrappedValue = mainLayers

    self.ropeEmbedder = ZImageRopeEmbedder(
      theta: configuration.ropeTheta,
      axesDims: configuration.axesDims,
      axesLens: configuration.axesLens
    )

    self._xPadToken.wrappedValue = MLX.zeros([1, configuration.dim], dtype: .bfloat16)
    self._capPadToken.wrappedValue = MLX.zeros([1, configuration.dim], dtype: .bfloat16)
    super.init()
  }

  public func clearCache() {
    cache = nil
    cacheKey = nil
  }

  private func getOrBuildCache(
    batch: Int,
    height: Int,
    width: Int,
    frames: Int,
    capOriLen: Int,
    patchSize: Int,
    fPatchSize: Int
  ) -> TransformerCache {
    let key = TransformerCacheKey(
      batch: batch,
      height: height,
      width: width,
      frames: frames,
      capOriLen: capOriLen
    )

    if let existingKey = cacheKey, let existingCache = cache, existingKey == key {
      return existingCache
    }

    let newCache = TransformerCacheBuilder.build(
      batch: batch,
      height: height,
      width: width,
      frames: frames,
      capOriLen: capOriLen,
      patchSize: patchSize,
      fPatchSize: fPatchSize,
      ropeEmbedder: ropeEmbedder
    )

    self.cache = newCache
    self.cacheKey = key

    return newCache
  }

  public func forward(
    latents: MLXArray,
    timestep: MLXArray,
    promptEmbeds: MLXArray
  ) -> MLXArray {
    let hasFrameDim = latents.ndim == 5
    let batch = latents.dim(0)
    let channels = latents.dim(1)
    let frames = hasFrameDim ? latents.dim(2) : 1
    let height = latents.dim(hasFrameDim ? 3 : 2)
    let width = latents.dim(hasFrameDim ? 4 : 3)

    let patchSize = 2
    let fPatchSize = 1
    let xEmbed = allXEmbedder.twoOne
    let finalLayer = allFinalLayer.twoOne

    let capOriLen = promptEmbeds.dim(1)
    let cached = getOrBuildCache(
      batch: batch,
      height: height,
      width: width,
      frames: frames,
      capOriLen: capOriLen,
      patchSize: patchSize,
      fPatchSize: fPatchSize
    )

    var latentsWithFrame = latents
    if !hasFrameDim {
      latentsWithFrame = MLX.expandedDimensions(latents, axis: 2)
    }

    let tScaled = timestep * MLXArray(configuration.tScale)
    var tEmb = tEmbedder(tScaled)

    var capFeat = promptEmbeds
    if cached.capPad > 0 {
      let last = promptEmbeds[0..., capOriLen - 1, 0...]
      let pad = MLX.broadcast(last, to: [batch, cached.capPad, promptEmbeds.dim(2)])
      capFeat = MLX.concatenated([promptEmbeds, pad], axis: 1)
    }
    capFeat = capEmbedder.1(capEmbedder.0(capFeat))

    if let capPadToken, let capPadMask = cached.capPadMask {
      let padDim = capPadToken.dim(capPadToken.ndim - 1)
      let pad = MLX.broadcast(capPadToken.reshaped(1, 1, padDim), to: [batch, cached.capSeqLen, padDim])
      capFeat = MLX.where(MLX.expandedDimensions(capPadMask, axis: 2), pad, capFeat)
    }

    var image = latentsWithFrame
      .reshaped(batch, channels, cached.fTokens, fPatchSize, cached.hTokens, patchSize, cached.wTokens, patchSize)
      .transposed(0, 2, 4, 6, 3, 5, 7, 1)
      .reshaped(batch, cached.imageTokens, patchSize * patchSize * fPatchSize * channels)

    if cached.imgPad > 0 {
      let last = image[0..., cached.imageTokens - 1, 0...]
      let pad = MLX.broadcast(last, to: [batch, cached.imgPad, image.dim(2)])
      image = MLX.concatenated([image, pad], axis: 1)
    }

    image = xEmbed(image)
    tEmb = tEmb.asType(image.dtype)

    if let xPadToken, let imgPadMask = cached.imgPadMask {
      let padDim = xPadToken.dim(xPadToken.ndim - 1)
      let pad = MLX.broadcast(xPadToken.reshaped(1, 1, padDim), to: [batch, cached.imgSeqLen, padDim])
      image = MLX.where(MLX.expandedDimensions(imgPadMask, axis: 2), pad, image)
    }

    var noiseStream = image
    for block in noiseRefiner {
      if gradientCheckpointing {
        let blockRef = block
        let imgFreqs = cached.imgFreqs
        let tEmbRef = tEmb
        let checkpointedBlock = checkpoint { (inputs: [MLXArray]) -> [MLXArray] in
          [blockRef(inputs[0], attnMask: nil, freqsCis: imgFreqs, adalnInput: tEmbRef)]
        }
        noiseStream = checkpointedBlock([noiseStream])[0]
      } else {
        noiseStream = block(
          noiseStream,
          attnMask: nil,
          freqsCis: cached.imgFreqs,
          adalnInput: tEmb
        )
      }
    }

    var capStream = capFeat
    for block in contextRefiner {
      if gradientCheckpointing {
        let blockRef = block
        let capFreqs = cached.capFreqs
        let checkpointedBlock = checkpoint { (inputs: [MLXArray]) -> [MLXArray] in
          [blockRef(inputs[0], attnMask: nil, freqsCis: capFreqs, adalnInput: nil)]
        }
        capStream = checkpointedBlock([capStream])[0]
      } else {
        capStream = block(
          capStream,
          attnMask: nil,
          freqsCis: cached.capFreqs,
          adalnInput: nil
        )
      }
    }

    var unified = MLX.concatenated([noiseStream, capStream], axis: 1)

    for block in layers {
      if gradientCheckpointing {
        let blockRef = block
        let unifiedFreqs = cached.unifiedFreqsCis
        let tEmbRef = tEmb
        let checkpointedBlock = checkpoint { (inputs: [MLXArray]) -> [MLXArray] in
          [blockRef(inputs[0], attnMask: nil, freqsCis: unifiedFreqs, adalnInput: tEmbRef)]
        }
        unified = checkpointedBlock([unified])[0]
      } else {
        unified = block(unified, attnMask: nil, freqsCis: cached.unifiedFreqsCis, adalnInput: tEmb)
      }
    }

    let imageOut = unified[0..., 0..<cached.imageTokens, 0...]
    let projected = finalLayer(imageOut, conditioning: tEmb)
    let outChannels = configuration.inChannels

    var reshaped = projected
      .reshaped(batch, cached.fTokens, cached.hTokens, cached.wTokens, fPatchSize, patchSize, patchSize, outChannels)
      .transposed(0, 7, 1, 4, 2, 5, 3, 6)
      .reshaped(batch, outChannels, cached.fTokens * fPatchSize, cached.hTokens * patchSize, cached.wTokens * patchSize)

    reshaped = reshaped[0..., 0..., 0, 0..., 0...]
    return reshaped
  }
}
