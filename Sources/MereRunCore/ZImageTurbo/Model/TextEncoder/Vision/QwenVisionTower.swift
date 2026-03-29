import Foundation
import MLX
import MLXNN

enum VisionTowerError: Error {
  case windowIndexOutOfBounds
}

struct QwenVisionTowerOutput {
  let hiddenStates: MLXArray
  let cumulativeWindowSequenceLengths: MLXArray
  let cumulativeSequenceLengths: MLXArray
  let windowIndex: MLXArray
  let rotaryCos: MLXArray
  let rotarySin: MLXArray
  let patchInputs: MLXArray
  /// Qwen3-VL: deepstack visual features from intermediate layers
  let deepstackFeatures: [MLXArray]
}

final class QwenVisionTower: Module {
  let configuration: QwenVisionConfiguration

  @ModuleInfo(key: "patch_embed") private var patchEmbed: QwenVisionPatchEmbed
  @ModuleInfo(key: "patch_merger") private var patchMerger: QwenVisionPatchMerger
  @ModuleInfo(key: "blocks") private var blocks: [QwenVisionBlock]
  /// Qwen3-VL: learned position embeddings
  @ModuleInfo(key: "pos_embed") private var posEmbed: Embedding?
  /// Qwen3-VL: deepstack mergers for intermediate visual features
  @ModuleInfo(key: "deepstack_merger_list") private var deepstackMergerList: [QwenVisionPatchMerger]

  private var mergeUnit: Int { configuration.spatialMergeSize * configuration.spatialMergeSize }
  private let headDim: Int
  private let rotaryInnerDim: Int
  private let rotaryHalfDim: Int
  private let ropeTheta: Float = 10_000.0
  private let invFrequencies: [Float32]
  /// Qwen3-VL: grid size for position embedding interpolation
  private let numGridPerSide: Int
  /// Qwen3-VL: layer indices for deepstack features
  private let deepstackVisualIndexes: [Int]

  init(configuration: QwenVisionConfiguration) {
    self.configuration = configuration
    self.headDim = configuration.embedDim / configuration.numHeads
    self.rotaryInnerDim = max(1, headDim / 2)
    self.rotaryHalfDim = max(1, rotaryInnerDim / 2)
    self.invFrequencies = QwenVisionTower.computeInverseFrequencies(
      dim: rotaryInnerDim,
      theta: ropeTheta
    )
    self.deepstackVisualIndexes = configuration.deepstackVisualIndexes

    // Qwen3-VL: learned position embeddings
    if configuration.useLearnedPosEmbed, let numPosEmbed = configuration.numPositionEmbeddings {
      self._posEmbed.wrappedValue = Embedding(embeddingCount: numPosEmbed, dimensions: configuration.embedDim)
      self.numGridPerSide = Int(sqrt(Double(numPosEmbed)))
    } else {
      self._posEmbed.wrappedValue = nil
      self.numGridPerSide = 0
    }

    self._patchEmbed.wrappedValue = QwenVisionPatchEmbed(
      patchSize: configuration.patchSize,
      temporalPatchSize: configuration.temporalPatchSize,
      inChannels: configuration.inChannels,
      embedDim: configuration.embedDim,
      bias: configuration.patchEmbedBias
    )
    self._patchMerger.wrappedValue = QwenVisionPatchMerger(
      contextDim: configuration.embedDim,
      outputDim: configuration.outHiddenDim,
      spatialMergeSize: configuration.spatialMergeSize
    )

    // Qwen3-VL: deepstack mergers (with postshuffle norm)
    self._deepstackMergerList.wrappedValue = configuration.deepstackVisualIndexes.map { _ in
      QwenVisionPatchMerger(
        contextDim: configuration.embedDim,
        outputDim: configuration.outHiddenDim,
        spatialMergeSize: configuration.spatialMergeSize,
        usePostshuffleNorm: true
      )
    }

    let blocks = (0..<configuration.depth).map { index in
      QwenVisionBlock(configuration: configuration, blockIndex: index)
    }
    self._blocks.wrappedValue = blocks
  }

  func callAsFunction(patchInputs: MLXArray, grid: [QwenVisionGrid]) throws -> QwenVisionTowerOutput {
    // Dispatch to Qwen3-VL path if using learned position embeddings
    if configuration.useLearnedPosEmbed {
      return try forwardQwen3(patchInputs: patchInputs, grid: grid)
    }

    // Qwen2-VL path (windowed attention)
    precondition(patchInputs.ndim == 3, "Expected patch input shape [batch, tokens, patchVolume]")
    precondition(grid.count == patchInputs.dim(0), "Grid metadata must match batch size")

    let batch = patchInputs.dim(0)
    let tokensPerSample = patchInputs.dim(1)
    let patchVolume = patchInputs.dim(2)

    var flattened = patchInputs.reshaped(batch * tokensPerSample, patchVolume)
    flattened = patchEmbed(flattened)
    let computeType = patchInputs.dtype

    var sampleOffset = 0
    var outputs: [MLXArray] = []
    var cuWindowLengths: [Int] = [0]
    var cuFullLengths: [Int] = [0]
    var windowIndicesAll: [Int] = []
    var rotaryCosSlices: [MLXArray] = []
    var rotarySinSlices: [MLXArray] = []

    let maxGridSize = max(1, grid.map { max($0.height, $0.width) }.max() ?? 1)
    let rotaryTable = makeRotaryTable(maxSize: maxGridSize, dtype: computeType)

    var llmCellBase = 0
    for entry in grid {
      let tokenCount = entry.temporal * entry.height * entry.width
      let start = sampleOffset
      let end = sampleOffset + tokenCount
      let sampleTokens = flattened[start..<end, 0...]
      sampleOffset = end

      let llmHeight = entry.height / configuration.spatialMergeSize
      let llmWidth = entry.width / configuration.spatialMergeSize
      precondition(llmHeight > 0 && llmWidth > 0, "Invalid spatial merge dimensions")

      let positions = makeOrderedPatchPositions(
        temporal: entry.temporal,
        height: entry.height,
        width: entry.width
      )
      var rotary = makeRotaryEmbeddings(
        rotaryTable: rotaryTable,
        orderedH: positions.h,
        orderedW: positions.w
      )

      let windowData = computeWindowIndexAndSeqLens(
        temporal: entry.temporal,
        height: entry.height,
        width: entry.width
      )
      let windowIndex = windowData.index
      let cuWindowSeqLens = windowData.cuWindowSeqLens

      // Reorder hidden states and rotary embeddings by window index (group of 4 tokens per LLM cell)
      var hidden = sampleTokens
      hidden = hidden.reshaped(tokenCount / mergeUnit, mergeUnit, configuration.embedDim)
      hidden = MLX.take(hidden, windowIndex, axis: 0)
      hidden = hidden.reshaped(1, tokenCount, configuration.embedDim)
      let rotaryDim = rotary.cos.dim(1)
      rotary.cos = rotary.cos.reshaped(tokenCount / mergeUnit, mergeUnit, rotaryDim)
      rotary.cos = MLX.take(rotary.cos, windowIndex, axis: 0).reshaped(tokenCount, rotaryDim)
      rotary.sin = rotary.sin.reshaped(tokenCount / mergeUnit, mergeUnit, rotaryDim)
      rotary.sin = MLX.take(rotary.sin, windowIndex, axis: 0).reshaped(tokenCount, rotaryDim)

      // Build cuSeqlens for full attention (per temporal frame, unmerged grid)
      let fullPerFrame = Int32(entry.height * entry.width)
      let cuFullSeqLens = MLX.multiply(
        MLXArray(0..<(entry.temporal + 1)).asType(.int32),
        MLXArray(fullPerFrame, dtype: .int32)
      )

      let cosEmbed = rotary.cos.asType(hidden.dtype)
      let sinEmbed = rotary.sin.asType(hidden.dtype)
      let fullMask = buildAttentionMask(
        sequenceLength: tokenCount,
        cuSeqlens: cuFullSeqLens,
        dtype: computeType
      )
      let windowMask = buildAttentionMask(
        sequenceLength: tokenCount,
        cuSeqlens: cuWindowSeqLens,
        dtype: computeType
      )
      for (bidx, block) in blocks.enumerated() {
        let isFull = configuration.fullAttentionBlockIndices.contains(bidx)
        let attnMask = isFull ? fullMask : windowMask
        hidden = block(
          hidden,
          rotaryEmbedding: (cos: cosEmbed, sin: sinEmbed),
          attentionMask: attnMask,
          cuSeqlens: nil
        )
      }
      hidden = hidden.reshaped(tokenCount, configuration.embedDim)
      // Merge 2x2 tokens per LLM cell
      let merged = patchMerger(hidden)

      // Reverse window index to original LLM order
      let reverseIdx = argsortInt32(windowIndex)
      let restored = MLX.take(merged, reverseIdx, axis: 0)

      outputs.append(restored)
      rotaryCosSlices.append(rotary.cos)
      rotarySinSlices.append(rotary.sin)

      // Update cumulative lengths
      let mergedPerFrame = (entry.height / configuration.spatialMergeSize) * (entry.width / configuration.spatialMergeSize)
      var cumulativeWindow = cuWindowLengths.last ?? 0
      for _ in 0..<entry.temporal { cumulativeWindow += mergedPerFrame; cuWindowLengths.append(cumulativeWindow) }

      var cumulativeFull = cuFullLengths.last ?? 0
      let perFrame = entry.height * entry.width
      for _ in 0..<entry.temporal { cumulativeFull += perFrame; cuFullLengths.append(cumulativeFull) }

      do {
        let shifted = MLX.add(windowIndex.asType(.int32), MLXArray([Int32(llmCellBase)].map(Float32.init), [1]).asType(.int32))
        let flat = shifted.asType(.int32)
        MLX.eval(flat)
        let vals = flat.asArray(Int32.self)
        windowIndicesAll.append(contentsOf: vals.map { Int($0) })
      }

      llmCellBase += mergedPerFrame * entry.temporal
    }

    let hiddenStates = outputs.count == 1 ? outputs[0] : MLX.concatenated(outputs, axis: 0)
    let windowSeqLens = makeIndexArray(cuWindowLengths)
    let fullSeqLens = makeIndexArray(cuFullLengths)
    let windowIndicesArray = makeIndexArray(windowIndicesAll)
    let rotaryCos = rotaryCosSlices.count == 1 ? rotaryCosSlices[0] : MLX.concatenated(rotaryCosSlices, axis: 0)
    let rotarySin = rotarySinSlices.count == 1 ? rotarySinSlices[0] : MLX.concatenated(rotarySinSlices, axis: 0)

    return QwenVisionTowerOutput(
      hiddenStates: hiddenStates,
      cumulativeWindowSequenceLengths: windowSeqLens,
      cumulativeSequenceLengths: fullSeqLens,
      windowIndex: windowIndicesArray,
      rotaryCos: rotaryCos,
      rotarySin: rotarySin,
      patchInputs: patchInputs,
      deepstackFeatures: []  // Qwen2-VL doesn't use deepstack
    )
  }

  // MARK: - Qwen3-VL Forward (learned position embeddings, no windowed attention)

  private func forwardQwen3(patchInputs: MLXArray, grid: [QwenVisionGrid]) throws -> QwenVisionTowerOutput {
    precondition(patchInputs.ndim == 3, "Expected patch input shape [batch, tokens, patchVolume]")
    precondition(grid.count == patchInputs.dim(0), "Grid metadata must match batch size")
    guard let posEmbed = posEmbed else {
      fatalError("pos_embed required for Qwen3-VL but not initialized")
    }

    let batch = patchInputs.dim(0)
    let tokensPerSample = patchInputs.dim(1)
    let patchVolume = patchInputs.dim(2)

    // Patch embedding
    let flattened = patchInputs.reshaped(batch * tokensPerSample, patchVolume)
    var hiddenStates = patchEmbed(flattened)

    // Add interpolated position embeddings
    let gridThw = grid.map { (t: $0.temporal, h: $0.height, w: $0.width) }
    let posEmbeds = fastPosEmbedInterpolate(gridThw: gridThw, posEmbed: posEmbed)
    hiddenStates = hiddenStates + posEmbeds

    // Compute rotary position embeddings
    let rotaryPosEmb = computeQwen3RotaryPosEmb(gridThw: gridThw)

    // Build cu_seqlens for variable-length attention
    var cuSeqlens: [Int32] = [0]
    for (t, h, w) in gridThw {
      let seqLenPerFrame = h * w
      for _ in 0..<t {
        cuSeqlens.append(cuSeqlens.last! + Int32(seqLenPerFrame))
      }
    }
    let cuSeqlensArray = MLXArray(cuSeqlens.map(Float32.init), [cuSeqlens.count]).asType(.int32)

    // Reshape for block processing: [1, seqLen, embedDim]
    let seqLen = hiddenStates.dim(0)
    hiddenStates = hiddenStates.reshaped(1, seqLen, configuration.embedDim)

    // Prepare rotary embeddings
    let rotaryEmb = prepareQwen3Rotary(rotaryPosEmb: rotaryPosEmb, dtype: hiddenStates.dtype)

    // Process through all blocks, capturing deepstack features at specified layers
    var deepstackFeatures: [MLXArray] = []
    for (layerIdx, block) in blocks.enumerated() {
      hiddenStates = block(
        hiddenStates,
        rotaryEmbedding: rotaryEmb,
        attentionMask: nil,
        cuSeqlens: cuSeqlensArray
      )

      // Capture deepstack features at intermediate layers
      if let mergerIdx = deepstackVisualIndexes.firstIndex(of: layerIdx) {
        // Reshape to [seqLen, embedDim] for the merger
        let h2d = hiddenStates.reshaped(seqLen, configuration.embedDim)
        let feature = deepstackMergerList[mergerIdx](h2d)
        deepstackFeatures.append(feature)
      }
    }

    // Reshape back to [seqLen, embedDim]
    hiddenStates = hiddenStates.reshaped(seqLen, configuration.embedDim)

    // Merge patches
    let merged = patchMerger(hiddenStates)

    // Build output (simplified - no window reordering needed for Qwen3-VL)
    var cuMergedLens: [Int] = [0]
    for (_, h, w) in gridThw {
      let mergedH = h / configuration.spatialMergeSize
      let mergedW = w / configuration.spatialMergeSize
      let mergedPerFrame = mergedH * mergedW
      for _ in 0..<1 {  // t is always 1 for single images
        cuMergedLens.append(cuMergedLens.last! + mergedPerFrame)
      }
    }

    return QwenVisionTowerOutput(
      hiddenStates: merged,
      cumulativeWindowSequenceLengths: makeIndexArray(cuMergedLens),
      cumulativeSequenceLengths: makeIndexArray(cuMergedLens),
      windowIndex: MLXArray([Int32(0)]),
      rotaryCos: rotaryPosEmb,
      rotarySin: rotaryPosEmb,
      patchInputs: patchInputs,
      deepstackFeatures: deepstackFeatures
    )
  }

  /// Fast position embedding interpolation (Qwen3-VL style)
  private func fastPosEmbedInterpolate(gridThw: [(t: Int, h: Int, w: Int)], posEmbed: Embedding) -> MLXArray {
    var idxList: [[Int32]] = [[], [], [], []]
    var weightList: [[Float32]] = [[], [], [], []]

    for (t, h, w) in gridThw {
      let hIdxs = (0..<h).map { Float32($0) * Float32(numGridPerSide - 1) / Float32(max(1, h - 1)) }
      let wIdxs = (0..<w).map { Float32($0) * Float32(numGridPerSide - 1) / Float32(max(1, w - 1)) }

      for hi in 0..<h {
        let hIdx = hIdxs[hi]
        let hFloor = Int32(hIdx)
        let hCeil = min(hFloor + 1, Int32(numGridPerSide - 1))
        let dh = hIdx - Float32(hFloor)

        for wi in 0..<w {
          let wIdx = wIdxs[wi]
          let wFloor = Int32(wIdx)
          let wCeil = min(wFloor + 1, Int32(numGridPerSide - 1))
          let dw = wIdx - Float32(wFloor)

          let baseH = hFloor * Int32(numGridPerSide)
          let baseHCeil = hCeil * Int32(numGridPerSide)

          idxList[0].append(baseH + wFloor)
          idxList[1].append(baseH + wCeil)
          idxList[2].append(baseHCeil + wFloor)
          idxList[3].append(baseHCeil + wCeil)

          weightList[0].append((1 - dh) * (1 - dw))
          weightList[1].append((1 - dh) * dw)
          weightList[2].append(dh * (1 - dw))
          weightList[3].append(dh * dw)
        }
      }
    }

    // Lookup embeddings and interpolate
    let maxLen = idxList[0].count
    var result: MLXArray? = nil
    for i in 0..<4 {
      let indices = MLXArray(idxList[i].map(Float32.init), [maxLen]).asType(.int32)
      let weights = MLXArray(weightList[i], [maxLen, 1])
      let embeds = posEmbed(indices)  // [maxLen, embedDim]
      let weighted = embeds * weights
      if result == nil {
        result = weighted
      } else {
        result = result! + weighted
      }
    }

    guard var posEmbeds = result else {
      return MLXArray([Float32(0)])
    }

    // Permute for spatial merging
    let mergeSize = configuration.spatialMergeSize
    var permutedList: [MLXArray] = []
    var start = 0
    for (t, h, w) in gridThw {
      let hwCount = h * w
      let end = start + hwCount
      var pe = posEmbeds[start..<end, 0...]  // [h*w, embedDim]
      // Tile for temporal dimension
      pe = MLX.tiled(pe, repetitions: [t, 1])  // [t*h*w, embedDim]
      pe = pe.reshaped(t, h, w, configuration.embedDim)
      pe = pe.reshaped(t, h / mergeSize, mergeSize, w / mergeSize, mergeSize, configuration.embedDim)
      pe = pe.transposed(0, 1, 3, 2, 4, 5)
      pe = pe.reshaped(t * (h / mergeSize) * (w / mergeSize) * mergeSize * mergeSize, configuration.embedDim)
      permutedList.append(pe)
      start = end
    }

    return permutedList.count == 1 ? permutedList[0] : MLX.concatenated(permutedList, axis: 0)
  }

  /// Compute rotary position embeddings (Qwen3-VL style)
  private func computeQwen3RotaryPosEmb(gridThw: [(t: Int, h: Int, w: Int)]) -> MLXArray {
    let mergeSize = configuration.spatialMergeSize
    var posIds: [Float32] = []

    for (t, h, w) in gridThw {
      let mergedH = h / mergeSize
      let mergedW = w / mergeSize

      for _ in 0..<t {
        for bh in 0..<mergedH {
          for bw in 0..<mergedW {
            for ih in 0..<mergeSize {
              for iw in 0..<mergeSize {
                let hPos = Float32(bh * mergeSize + ih)
                let wPos = Float32(bw * mergeSize + iw)
                posIds.append(hPos)
                posIds.append(wPos)
              }
            }
          }
        }
      }
    }

    let totalTokens = posIds.count / 2
    let posArray = MLXArray(posIds, [totalTokens, 2])

    // Build frequency table
    let maxGridSize = gridThw.map { max($0.h, $0.w) }.max() ?? 1
    let freqTable = makeRotaryTable(maxSize: maxGridSize, dtype: .float32)

    // Lookup h and w embeddings
    let hIndices = posArray[0..., 0..<1].squeezed(axis: 1).asType(.int32)
    let wIndices = posArray[0..., 1..<2].squeezed(axis: 1).asType(.int32)
    let hEmb = MLX.take(freqTable, hIndices, axis: 0)
    let wEmb = MLX.take(freqTable, wIndices, axis: 0)

    return MLX.concatenated([hEmb, wEmb], axis: 1)
  }

  /// Prepare rotary embeddings for attention (cos, sin)
  private func prepareQwen3Rotary(rotaryPosEmb: MLXArray, dtype: DType) -> (cos: MLXArray, sin: MLXArray) {
    // rotaryPosEmb is [seq, head_dim/2] (h+w embeddings concatenated, each head_dim/4)
    // Tile by 2 to match full head_dim
    var cos = MLX.cos(rotaryPosEmb)  // [seq, head_dim/2]
    var sin = MLX.sin(rotaryPosEmb)  // [seq, head_dim/2]
    cos = MLX.tiled(cos, repetitions: [1, 2]).asType(dtype)  // [seq, head_dim]
    sin = MLX.tiled(sin, repetitions: [1, 2]).asType(dtype)  // [seq, head_dim]
    return (cos, sin)
  }

  private func buildAttentionMask(
    sequenceLength: Int,
    cuSeqlens: MLXArray,
    dtype: DType
  ) -> MLXArray {
    // This mask is block-diagonal based on `cuSeqlens` boundaries; it can be built cheaply on CPU.
    // The previous implementation materialized an [L, chunkCount] membership matrix and did a matmul
    // to get [L, L], which is extremely expensive for 512x512 inputs (L ~ 260k).
    let lensArr = cuSeqlens.asType(.int32)
    MLX.eval(lensArr)
    let lens = lensArr.asArray(Int32.self).map { Int($0) }
    guard lens.count >= 2 else {
      return MLX.zeros([1, sequenceLength, sequenceLength], dtype: .bool)
    }

    var mask = [UInt8](repeating: 0, count: sequenceLength * sequenceLength)
    for i in 0..<(lens.count - 1) {
      let start = max(0, min(lens[i], sequenceLength))
      let end = max(0, min(lens[i + 1], sequenceLength))
      guard end > start else { continue }
      for row in start..<end {
        let base = row * sequenceLength
        mask.withUnsafeMutableBufferPointer { buf in
          buf.baseAddress!.advanced(by: base + start).assign(repeating: 1, count: end - start)
        }
      }
    }

    // Build as [1, L, L] bool.
    let arr = MLXArray(mask.map(Float32.init), [1, sequenceLength, sequenceLength]).asType(.bool)
    return arr
  }

  // Build window index and cumulative lengths for attention chunking matching diffusers
  private func computeWindowIndexAndSeqLens(
    temporal: Int,
    height: Int,
    width: Int
  ) -> (index: MLXArray, cuWindowSeqLens: MLXArray) {
    let llmH = height / configuration.spatialMergeSize
    let llmW = width / configuration.spatialMergeSize
    let vitWindow = configuration.windowSize / configuration.spatialMergeSize / configuration.patchSize
    let padH = (vitWindow - (llmH % vitWindow)) % vitWindow
    let padW = (vitWindow - (llmW % vitWindow)) % vitWindow
    let numWindowsH = (llmH + padH) / vitWindow
    let numWindowsW = (llmW + padW) / vitWindow

    // Build padded grid for a single frame.
    var padded: [Int32] = Array(repeating: -100, count: (llmH + padH) * (llmW + padW))
    for h in 0..<llmH {
      for w in 0..<llmW {
        padded[h * (llmW + padW) + w] = Int32(h * llmW + w)
      }
    }

    // Precompute window indices and counts for one frame.
    var frameWindowIndices: [Int32] = []
    var frameCuLens: [Int32] = [0]
    for wh in 0..<numWindowsH {
      for ww in 0..<numWindowsW {
        var count: Int32 = 0
        for ih in 0..<vitWindow {
          let h = wh * vitWindow + ih
          let rowBase = h * (llmW + padW)
          for iw in 0..<vitWindow {
            let w = ww * vitWindow + iw
            let val = padded[rowBase + w]
            if val != -100 {
              frameWindowIndices.append(val)
              count += 1
            }
          }
        }
        let last = frameCuLens.last ?? 0
        frameCuLens.append(last + count * Int32(mergeUnit))
      }
    }

    // Replicate per-frame windows across temporal dimension with base offsets.
    var windowIndexAll: [Int32] = []
    windowIndexAll.reserveCapacity(frameWindowIndices.count * temporal)
    var cuLens: [Int32] = [0]
    cuLens.reserveCapacity(frameCuLens.count * temporal)
    let perFrameBase = Int32(llmH * llmW)
    for t in 0..<temporal {
      let base = Int32(t) * perFrameBase
      for idx in frameWindowIndices {
        windowIndexAll.append(idx + base)
      }
      let offset = cuLens.last ?? 0
      for len in frameCuLens.dropFirst() {
        cuLens.append(offset + len)
      }
    }

    let indexArr = MLXArray(windowIndexAll.map(Float32.init), [windowIndexAll.count]).asType(.int32)
    let cuArr = MLXArray(cuLens.map(Float32.init), [cuLens.count]).asType(.int32)
    return (indexArr, cuArr)
  }

  // Argsort for int32 array (CPU path) → MLXArray[int32]
  private func argsortInt32(_ values: MLXArray) -> MLXArray {
    let ints = values.asType(.int32)
    MLX.eval(ints)
    let swiftVals = ints.asArray(Int32.self)
    let indexed = swiftVals.enumerated().map { ($0.offset, $0.element) }
    let sorted = indexed.sorted { $0.1 < $1.1 }
    let result: [Int32] = sorted.map { Int32($0.0) }
    return MLXArray(result.map(Float32.init), [result.count]).asType(.int32)
  }

  private func makeOrderedPatchPositions(
    temporal: Int,
    height: Int,
    width: Int
  ) -> (h: MLXArray, w: MLXArray) {
    let merge = configuration.spatialMergeSize
    let llmHeight = height / merge
    let llmWidth = width / merge
    let llmCount = temporal * llmHeight * llmWidth

    var hValues: [Float32] = []
    var wValues: [Float32] = []
    hValues.reserveCapacity(llmCount * mergeUnit)
    wValues.reserveCapacity(llmCount * mergeUnit)

    for _ in 0..<temporal {
      for hBlock in 0..<llmHeight {
        for wBlock in 0..<llmWidth {
          for innerH in 0..<merge {
            for innerW in 0..<merge {
              hValues.append(Float32(hBlock * merge + innerH))
              wValues.append(Float32(wBlock * merge + innerW))
            }
          }
        }
      }
    }

    let llmShape = [llmCount, mergeUnit]
    let hArray = MLXArray(hValues, llmShape).asType(.int32)
    let wArray = MLXArray(wValues, llmShape).asType(.int32)

    return (
      hArray.reshaped(hArray.dim(0) * hArray.dim(1)),
      wArray.reshaped(wArray.dim(0) * wArray.dim(1))
    )
  }

  private func makeRotaryEmbeddings(
    rotaryTable: MLXArray,
    orderedH: MLXArray,
    orderedW: MLXArray
  ) -> (cos: MLXArray, sin: MLXArray) {
    let hAngles = MLX.take(rotaryTable, orderedH, axis: 0)
    let wAngles = MLX.take(rotaryTable, orderedW, axis: 0)
    let baseAngles = MLX.concatenated([hAngles, wAngles], axis: 1)
    let duplicated = MLX.concatenated([baseAngles, baseAngles], axis: 1)
    return (MLX.cos(duplicated), MLX.sin(duplicated))
  }

  private func makeIndexArray(_ values: [Int]) -> MLXArray {
    let floats = values.map(Float32.init)
    return MLXArray(floats, [values.count]).asType(.int32)
  }

  private func makeRotaryTable(maxSize: Int, dtype: DType) -> MLXArray {
    let count = max(1, rotaryHalfDim)
    var table: [Float32] = []
    table.reserveCapacity(maxSize * count)

    for position in 0..<maxSize {
      let pos = Float32(position)
      for freq in invFrequencies {
        table.append(pos * freq)
      }
    }

    return MLXArray(table, [maxSize, count]).asType(dtype)
  }

  private static func computeInverseFrequencies(dim: Int, theta: Float) -> [Float32] {
    var values: [Float32] = []
    let stepCount = max(1, dim / 2)
    let dimFloat = Float32(dim)
    for i in 0..<stepCount {
      let exponent = Float32(2 * i) / dimFloat
      let freq = powf(theta, -exponent)
      values.append(freq)
    }
    return values
  }

  var blockCount: Int {
    blocks.count
  }

  func updatePatchEmbed(parameters: ModuleParameters) {
    patchEmbed.update(parameters: parameters)
  }

  func updatePatchMerger(parameters: ModuleParameters) {
    patchMerger.update(parameters: parameters)
  }

  fileprivate func patchEmbedModule() -> QwenVisionPatchEmbed {
    patchEmbed
  }

  fileprivate func patchMergerModule() -> QwenVisionPatchMerger {
    patchMerger
  }

  func updateBlock(at index: Int, parameters: ModuleParameters) {
    precondition(index >= 0 && index < blocks.count, "Block index out of range")
    blocks[index].update(parameters: parameters)
  }
}
