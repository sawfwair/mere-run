import Foundation
import MLX
import MLXFast
import MLXNN

@inline(__always)
private func museGlimmerRMSNormNoScale(_ x: MLXArray, eps: Float) -> MLXArray {
    let dtype = x.dtype
    let promoted = x.asType(.float32)
    let variance = (promoted * promoted).mean(axis: -1, keepDims: true)
    return (promoted * MLX.rsqrt(variance + MLXArray(eps))).asType(dtype)
}

final class MuseGlimmerCenteredRMSNorm: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    private let eps: Float

    init(dimensions: Int, eps: Float) {
        self._weight.wrappedValue = MLXArray.zeros([dimensions])
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let dtype = x.dtype
        let promoted = x.asType(.float32)
        let variance = (promoted * promoted).mean(axis: -1, keepDims: true)
        return (promoted * MLX.rsqrt(variance + MLXArray(eps)) * (1 + weight.asType(.float32)))
            .asType(dtype)
    }
}

final class MuseGlimmerRMSNorm: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    private let eps: Float

    init(dimensions: Int, eps: Float) {
        self._weight.wrappedValue = MLXArray.ones([dimensions])
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let dtype = x.dtype
        let promoted = x.asType(.float32)
        let variance = (promoted * promoted).mean(axis: -1, keepDims: true)
        return (promoted * MLX.rsqrt(variance + MLXArray(eps)) * weight.asType(.float32)).asType(dtype)
    }
}

final class MuseGlimmerTextMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(config: MuseGlimmerTextConfig) {
        self._gateProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(MLXNN.silu(gateProj(x)) * upProj(x))
    }
}

final class MuseGlimmerTextAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "gate_proj") var gateProj: Linear

    private let headCount: Int
    private let keyValueHeadCount: Int
    private let headDimension: Int
    private let slidingWindow: Int?
    private let qkScaleFactor: Float
    private let rmsNormEps: Float
    private let attentionScale: Float
    private let rope: RoPE?

    init(config: MuseGlimmerTextConfig, layerIndex: Int) {
        headCount = config.numAttentionHeads
        keyValueHeadCount = config.numKeyValueHeads
        headDimension = config.headDim
        slidingWindow = config.layerTypes[layerIndex] == "sliding_attention" ? config.slidingWindow : nil
        qkScaleFactor = config.qkScaleFactor
        rmsNormEps = config.rmsNormEps
        attentionScale = 1 / sqrt(Float(config.headDim))
        let theta = config.layerRopeTheta[layerIndex]
        rope = theta > 0
            ? RoPE(dimensions: config.headDim, traditional: false, base: theta)
            : nil
        self._qProj.wrappedValue = Linear(
            config.hiddenSize,
            config.numAttentionHeads * config.headDim,
            bias: config.attentionBias
        )
        self._kProj.wrappedValue = Linear(
            config.hiddenSize,
            config.numKeyValueHeads * config.headDim,
            bias: config.attentionBias
        )
        self._vProj.wrappedValue = Linear(
            config.hiddenSize,
            config.numKeyValueHeads * config.headDim,
            bias: config.attentionBias
        )
        self._oProj.wrappedValue = Linear(
            config.numAttentionHeads * config.headDim,
            config.hiddenSize,
            bias: config.attentionBias
        )
        self._gateProj.wrappedValue = Linear(
            config.hiddenSize,
            config.numAttentionHeads * config.headDim,
            bias: false
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cache: Gemma4AttentionCache?) -> MLXArray {
        let batch = x.dim(0)
        let sequence = x.dim(1)
        let offset = cache?.offset ?? 0
        var queries = qProj(x).reshaped(batch, sequence, headCount, headDimension).transposed(0, 2, 1, 3)
        var keys = kProj(x).reshaped(batch, sequence, keyValueHeadCount, headDimension).transposed(0, 2, 1, 3)
        var values = vProj(x).reshaped(batch, sequence, keyValueHeadCount, headDimension).transposed(0, 2, 1, 3)

        queries = museGlimmerRMSNormNoScale(queries, eps: rmsNormEps) * qkScaleFactor
        keys = museGlimmerRMSNormNoScale(keys, eps: rmsNormEps)
        if let rope {
            queries = rope(queries, offset: offset)
            keys = rope(keys, offset: offset)
        }

        if let cache, let state = cache.attentionState(appending: keys, values: values) {
            keys = state.0
            values = state.1
        }
        let mask = attentionMask(
            queryLength: sequence,
            queryOffset: offset,
            keyLength: keys.dim(2),
            windowSize: slidingWindow,
            dtype: x.dtype
        )
        let attended = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: attentionScale,
            mask: mask
        )
        var output = attended.transposed(0, 2, 1, 3)
            .reshaped(batch, sequence, headCount * headDimension)
        output = output * MLX.sigmoid(gateProj(x))
        return oProj(output)
    }

    private func attentionMask(
        queryLength: Int,
        queryOffset: Int,
        keyLength: Int,
        windowSize: Int?,
        dtype: DType
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        guard queryLength > 1 else { return .none }
        let keyStart = max(0, queryOffset + queryLength - keyLength)
        let queryPositions = MLXArray(Int32(queryOffset)..<Int32(queryOffset + queryLength)).reshaped(queryLength, 1)
        let keyPositions = MLXArray(Int32(keyStart)..<Int32(keyStart + keyLength)).reshaped(1, keyLength)
        var allowed = keyPositions .<= queryPositions
        if let windowSize {
            allowed = allowed .&& (keyPositions .> (queryPositions - Int32(windowSize)))
        }
        let typed = allowed.asType(dtype).reshaped(1, 1, queryLength, keyLength)
        let zeros = MLXArray.zeros([1, 1, queryLength, keyLength], dtype: dtype)
        return .array(MLX.where(typed .> MLXArray(0).asType(dtype), zeros, zeros - 1e9))
    }
}

final class MuseGlimmerTextDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: MuseGlimmerTextAttention
    @ModuleInfo(key: "mlp") var mlp: MuseGlimmerTextMLP
    @ModuleInfo(key: "input_layernorm") var inputNorm: MuseGlimmerCenteredRMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionNorm: MuseGlimmerCenteredRMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedForwardNorm: MuseGlimmerCenteredRMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedForwardNorm: MuseGlimmerCenteredRMSNorm

    init(config: MuseGlimmerTextConfig, layerIndex: Int) {
        self._attention.wrappedValue = MuseGlimmerTextAttention(config: config, layerIndex: layerIndex)
        self._mlp.wrappedValue = MuseGlimmerTextMLP(config: config)
        self._inputNorm.wrappedValue = MuseGlimmerCenteredRMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
        self._postAttentionNorm.wrappedValue = MuseGlimmerCenteredRMSNorm(
            dimensions: config.hiddenSize,
            eps: config.postNormEps
        )
        self._preFeedForwardNorm.wrappedValue = MuseGlimmerCenteredRMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
        self._postFeedForwardNorm.wrappedValue = MuseGlimmerCenteredRMSNorm(
            dimensions: config.hiddenSize,
            eps: config.postNormEps
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cache: Gemma4AttentionCache?) -> MLXArray {
        let afterAttention = x + postAttentionNorm(attention(inputNorm(x), cache: cache))
        return afterAttention + postFeedForwardNorm(mlp(preFeedForwardNorm(afterAttention)))
    }
}

struct MuseGlimmerLanguageModelOutput {
    let hidden: MLXArray
    let capturedHiddenStates: [Int: MLXArray]
}

final class MuseGlimmerLanguageModel: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [MuseGlimmerTextDecoderLayer]
    @ModuleInfo(key: "norm") var norm: MuseGlimmerRMSNorm

    private let config: MuseGlimmerTextConfig

    init(config: MuseGlimmerTextConfig) {
        self.config = config
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize,
            dimensions: config.hiddenSize
        )
        self._layers.wrappedValue = (0..<config.numHiddenLayers).map {
            MuseGlimmerTextDecoderLayer(config: config, layerIndex: $0)
        }
        self._norm.wrappedValue = MuseGlimmerRMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func embeddings(_ inputIds: MLXArray) -> MLXArray {
        museGlimmerRMSNormNoScale(embedTokens(inputIds.asType(.int32)), eps: config.rmsNormEps)
    }

    func callAsFunction(_ embeddings: MLXArray, cache: [Gemma4AttentionCache]?) -> MLXArray {
        forward(embeddings, cache: cache).hidden
    }

    func forward(
        _ embeddings: MLXArray,
        cache: [Gemma4AttentionCache]?,
        captureLayerIndices: Set<Int> = []
    ) -> MuseGlimmerLanguageModelOutput {
        var hidden = embeddings
        var capturedHiddenStates: [Int: MLXArray] = [:]
        for (index, layer) in layers.enumerated() {
            hidden = layer(hidden, cache: cache?[index])
            if captureLayerIndices.contains(index) {
                capturedHiddenStates[index] = hidden
            }
        }
        return MuseGlimmerLanguageModelOutput(
            hidden: norm(hidden),
            capturedHiddenStates: capturedHiddenStates
        )
    }

    func makeCache() -> [Gemma4AttentionCache] {
        config.layerTypes.map { type -> Gemma4AttentionCache in
            if type == "sliding_attention" {
                return Gemma4SlidingKVCache(maxSize: config.slidingWindow)
            }
            return Gemma4FullKVCache()
        }
    }
}

struct MuseGlimmerForwardOutput {
    let logits: MLXArray
    let capturedHiddenStates: [Int: MLXArray]
}

final class MuseGlimmerVisionPatchEmbedder: Module {
    @ModuleInfo(key: "patch_embedding") var patchEmbedding: Linear
    @ModuleInfo(key: "position_embedding_table") var positionEmbeddingTable: Embedding

    private let side: Int
    private let hiddenSize: Int

    init(config: MuseGlimmerVisionConfig) {
        side = config.posEmbHeight
        hiddenSize = config.hiddenSize
        self._patchEmbedding.wrappedValue = Linear(
            config.patchTemporal * 3 * config.patchSize * config.patchSize,
            config.hiddenSize,
            bias: false
        )
        self._positionEmbeddingTable.wrappedValue = Embedding(
            embeddingCount: config.posEmbHeight * config.posEmbWidth,
            dimensions: config.hiddenSize
        )
        super.init()
    }

    func callAsFunction(
        _ patches: MLXArray,
        grids: [(temporal: Int, height: Int, width: Int)]
    ) -> MLXArray {
        let geometry = Self.bilinearPositionGeometry(grids: grids, side: side)
        let indices = geometry.indices
        let weights = geometry.weights
        var position = MLXArray.zeros([patches.dim(0), hiddenSize], dtype: patches.dtype)
        for corner in 0..<4 {
            let indexArray = MLXArray(indices[corner])
            let weightArray = MLXArray(weights[corner], [weights[corner].count, 1]).asType(patches.dtype)
            position = position + positionEmbeddingTable(indexArray) * weightArray
        }
        return patchEmbedding(patches) + position
    }

    static func bilinearPositionGeometry(
        grids: [(temporal: Int, height: Int, width: Int)],
        side: Int
    ) -> (indices: [[Int32]], weights: [[Float]]) {
        var indices = [[Int32]](repeating: [], count: 4)
        var weights = [[Float]](repeating: [], count: 4)
        for grid in grids {
            for _ in 0..<grid.temporal {
                for row in 0..<grid.height {
                    let rowValue = (Float(row) + 0.5) * (Float(side) / Float(grid.height)) - 0.5
                    for column in 0..<grid.width {
                        let columnValue = (Float(column) + 0.5) * (Float(side) / Float(grid.width)) - 0.5
                        appendBilinear(
                            row: rowValue,
                            column: columnValue,
                            side: side,
                            indices: &indices,
                            weights: &weights
                        )
                    }
                }
            }
        }
        return (indices, weights)
    }

    private static func appendBilinear(
        row: Float,
        column: Float,
        side: Int,
        indices: inout [[Int32]],
        weights: inout [[Float]]
    ) {
        let floorRow = Int(floor(row))
        let ceilRow = floorRow + 1
        let floorColumn = Int(floor(column))
        let ceilColumn = floorColumn + 1
        let rowFraction = row - Float(floorRow)
        let columnFraction = column - Float(floorColumn)
        let corners = [
            (floorRow, floorColumn, (1 - rowFraction) * (1 - columnFraction)),
            (floorRow, ceilColumn, (1 - rowFraction) * columnFraction),
            (ceilRow, floorColumn, rowFraction * (1 - columnFraction)),
            (ceilRow, ceilColumn, rowFraction * columnFraction),
        ]
        for (index, corner) in corners.enumerated() {
            let valid = corner.0 >= 0 && corner.0 < side && corner.1 >= 0 && corner.1 < side
            let clampedRow = min(max(0, corner.0), side - 1)
            let clampedColumn = min(max(0, corner.1), side - 1)
            indices[index].append(Int32(clampedRow * side + clampedColumn))
            weights[index].append(valid ? corner.2 : 0)
        }
    }
}

final class MuseGlimmerVisionAttention: Module {
    @ModuleInfo(key: "proj") var projection: Linear
    @ModuleInfo(key: "q_proj") var queryProjection: Linear
    @ModuleInfo(key: "k_proj") var keyProjection: Linear
    @ModuleInfo(key: "v_proj") var valueProjection: Linear

    private let headCount: Int
    private let headDimension: Int
    private let scale: Float

    init(config: MuseGlimmerVisionConfig) {
        headCount = config.numAttentionHeads
        headDimension = config.hiddenSize / config.numAttentionHeads
        scale = 1 / sqrt(Float(headDimension))
        self._projection.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
        self._queryProjection.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
        self._keyProjection.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
        self._valueProjection.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        cosine: MLXArray,
        sine: MLXArray,
        segments: [Int]
    ) -> MLXArray {
        let sequence = x.dim(0)
        var queries = queryProjection(x).reshaped(1, sequence, headCount, headDimension)
        var keys = keyProjection(x).reshaped(1, sequence, headCount, headDimension)
        let values = valueProjection(x).reshaped(1, sequence, headCount, headDimension)
        queries = applyRotary(queries, cosine: cosine, sine: sine).transposed(0, 2, 1, 3)
        keys = applyRotary(keys, cosine: cosine, sine: sine).transposed(0, 2, 1, 3)
        let transposedValues = values.transposed(0, 2, 1, 3)
        var pieces: [MLXArray] = []
        for index in 0..<(segments.count - 1) {
            let start = segments[index]
            let end = segments[index + 1]
            guard end > start else { continue }
            pieces.append(MLXFast.scaledDotProductAttention(
                queries: queries[0..., 0..., start..<end, 0...],
                keys: keys[0..., 0..., start..<end, 0...],
                values: transposedValues[0..., 0..., start..<end, 0...],
                scale: scale,
                mask: .none
            ))
        }
        let attended = pieces.count == 1 ? pieces[0] : MLX.concatenated(pieces, axis: 2)
        return projection(attended.transposed(0, 2, 1, 3).reshaped(sequence, headCount * headDimension))
    }

    private func applyRotary(_ x: MLXArray, cosine: MLXArray, sine: MLXArray) -> MLXArray {
        let promoted = x.asType(.float32)
        let half = x.dim(-1) / 2
        let rotated = MLX.concatenated([
            -promoted[0..., 0..., 0..., half...],
            promoted[0..., 0..., 0..., 0..<half],
        ], axis: -1)
        return (promoted * cosine.asType(.float32) + rotated * sine.asType(.float32)).asType(x.dtype)
    }
}

final class MuseGlimmerVisionMLP: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    init(config: MuseGlimmerVisionConfig) {
        self._fc1.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: true)
        self._fc2.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(MLXNN.gelu(fc1(x)))
    }
}

final class MuseGlimmerVisionEncoderLayer: Module {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "attn") var attention: MuseGlimmerVisionAttention
    @ModuleInfo(key: "mlp") var mlp: MuseGlimmerVisionMLP

    init(config: MuseGlimmerVisionConfig) {
        self._norm1.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        self._norm2.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        self._attention.wrappedValue = MuseGlimmerVisionAttention(config: config)
        self._mlp.wrappedValue = MuseGlimmerVisionMLP(config: config)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        cosine: MLXArray,
        sine: MLXArray,
        segments: [Int]
    ) -> MLXArray {
        let afterAttention = x + attention(norm1(x), cosine: cosine, sine: sine, segments: segments)
        return afterAttention + mlp(norm2(afterAttention))
    }
}

final class MuseGlimmerVisionTower: Module {
    @ModuleInfo(key: "patch_embedder") var patchEmbedder: MuseGlimmerVisionPatchEmbedder
    @ModuleInfo(key: "ln_pre") var preNorm: LayerNorm
    @ModuleInfo(key: "layers") var layers: [MuseGlimmerVisionEncoderLayer]
    @ModuleInfo(key: "ln_post") var postNorm: LayerNorm

    private let config: MuseGlimmerVisionConfig

    init(config: MuseGlimmerVisionConfig) {
        self.config = config
        self._patchEmbedder.wrappedValue = MuseGlimmerVisionPatchEmbedder(config: config)
        self._preNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        self._layers.wrappedValue = (0..<config.numHiddenLayers).map { _ in
            MuseGlimmerVisionEncoderLayer(config: config)
        }
        self._postNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        super.init()
    }

    func callAsFunction(
        _ patches: MLXArray,
        grids: [(temporal: Int, height: Int, width: Int)]
    ) -> MLXArray {
        let layout = makeLayout(grids: grids)
        var hidden = preNorm(patchEmbedder(patches, grids: grids))
        let order = MLXArray(layout.order.map(Int32.init))
        hidden = MLX.take(hidden, order, axis: 0)
        let rotary = makeRotary(positionPairs: layout.positions, dtype: hidden.dtype)
        for (index, layer) in layers.enumerated() {
            let segments = config.layerTypes[index] == "full_attention"
                ? layout.fullSegments
                : layout.windowSegments
            hidden = layer(hidden, cosine: rotary.cosine, sine: rotary.sine, segments: segments)
        }
        hidden = MLX.take(hidden, MLXArray(layout.reverseOrder.map(Int32.init)), axis: 0)
        return pixelShuffle(postNorm(hidden), grids: grids)
    }

    private func makeLayout(
        grids: [(temporal: Int, height: Int, width: Int)]
    ) -> (order: [Int], reverseOrder: [Int], positions: [(Int, Int)], fullSegments: [Int], windowSegments: [Int]) {
        let windowSide = max(1, config.posEmbHeight)
        var order: [Int] = []
        var positions: [(Int, Int)] = []
        var fullSegments = [0]
        var windowSegments = [0]
        var base = 0
        for grid in grids {
            for frame in 0..<grid.temporal {
                for windowRow in stride(from: 0, to: grid.height, by: windowSide) {
                    for windowColumn in stride(from: 0, to: grid.width, by: windowSide) {
                        var windowCount = 0
                        for row in windowRow..<min(grid.height, windowRow + windowSide) {
                            for column in windowColumn..<min(grid.width, windowColumn + windowSide) {
                                order.append(base + frame * grid.height * grid.width + row * grid.width + column)
                                positions.append((column + 1, row + 1))
                                windowCount += 1
                            }
                        }
                        windowSegments.append((windowSegments.last ?? 0) + windowCount)
                    }
                }
                fullSegments.append((fullSegments.last ?? 0) + grid.height * grid.width)
            }
            base += grid.temporal * grid.height * grid.width
        }
        var reverseOrder = [Int](repeating: 0, count: order.count)
        for (ordered, original) in order.enumerated() {
            reverseOrder[original] = ordered
        }
        return (order, reverseOrder, positions, fullSegments, windowSegments)
    }

    private func makeRotary(
        positionPairs: [(Int, Int)],
        dtype: DType
    ) -> (cosine: MLXArray, sine: MLXArray) {
        let headDimension = config.hiddenSize / config.numAttentionHeads
        let spatialDimension = headDimension / 2
        var values: [Float] = []
        values.reserveCapacity(positionPairs.count * headDimension)
        for position in positionPairs {
            var widthFrequencies: [Float] = []
            var heightFrequencies: [Float] = []
            for index in stride(from: 0, to: spatialDimension, by: 2) {
                let inverse = pow(config.ropeParameters.ropeTheta, -Float(index) / Float(spatialDimension))
                widthFrequencies.append(Float(position.0) * inverse)
                heightFrequencies.append(Float(position.1) * inverse)
            }
            values.append(contentsOf: widthFrequencies)
            values.append(contentsOf: heightFrequencies)
            values.append(contentsOf: widthFrequencies)
            values.append(contentsOf: heightFrequencies)
        }
        let angles = MLXArray(values, [1, positionPairs.count, 1, headDimension]).asType(dtype)
        return (MLX.cos(angles), MLX.sin(angles))
    }

    private func pixelShuffle(
        _ hidden: MLXArray,
        grids: [(temporal: Int, height: Int, width: Int)]
    ) -> MLXArray {
        let merge = config.mergeSize
        let dimensions = hidden.dim(-1)
        var chunks: [MLXArray] = []
        var offset = 0
        for grid in grids {
            let count = grid.temporal * grid.height * grid.width
            var indices: [Int32] = []
            for frame in 0..<grid.temporal {
                for blockRow in 0..<(grid.height / merge) {
                    for blockColumn in 0..<(grid.width / merge) {
                        for innerRow in 0..<merge {
                            for innerColumn in 0..<merge {
                                indices.append(Int32(
                                    frame * grid.height * grid.width
                                        + (blockRow * merge + innerRow) * grid.width
                                        + blockColumn * merge + innerColumn
                                ))
                            }
                        }
                    }
                }
            }
            let chunk = hidden[offset..<(offset + count), 0...]
            chunks.append(
                MLX.take(chunk, MLXArray(indices), axis: 0)
                    .reshaped(count / (merge * merge), merge * merge, dimensions)
                    .transposed(0, 2, 1)
                    .reshaped(count / (merge * merge), dimensions * merge * merge)
            )
            offset += count
        }
        return chunks.count == 1 ? chunks[0] : MLX.concatenated(chunks, axis: 0)
    }
}

final class MuseGlimmerVisionAdapter: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    init(config: MuseGlimmerConfig) {
        self._fc1.wrappedValue = Linear(config.outHiddenSize, config.projectorHiddenSize, bias: false)
        self._fc2.wrappedValue = Linear(config.projectorHiddenSize, config.projectorHiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXNN.gelu(fc2(MLXNN.gelu(fc1(x))))
    }
}

final class MuseGlimmerBackbone: Module {
    @ModuleInfo(key: "vision_tower") var visionTower: MuseGlimmerVisionTower
    @ModuleInfo(key: "language_model") var languageModel: MuseGlimmerLanguageModel
    @ModuleInfo(key: "vision_adapter") var visionAdapter: MuseGlimmerVisionAdapter
    @ModuleInfo(key: "vision_projection") var visionProjection: Linear

    private let config: MuseGlimmerConfig

    init(config: MuseGlimmerConfig) {
        self.config = config
        self._visionTower.wrappedValue = MuseGlimmerVisionTower(config: config.visionConfig)
        self._languageModel.wrappedValue = MuseGlimmerLanguageModel(config: config.textConfig)
        self._visionAdapter.wrappedValue = MuseGlimmerVisionAdapter(config: config)
        self._visionProjection.wrappedValue = Linear(
            config.projectorHiddenSize,
            config.textConfig.hiddenSize,
            bias: false
        )
        super.init()
    }

    func embeddings(
        inputIds: MLXArray,
        imageBatch: MuseGlimmerImageBatch?
    ) throws -> MLXArray {
        var hidden = languageModel.embeddings(inputIds)
        guard let imageBatch else { return hidden }
        var features = visionTower(imageBatch.pixelValues, grids: imageBatch.grids)
        features = museGlimmerRMSNormNoScale(
            visionProjection(visionAdapter(features)),
            eps: config.textConfig.rmsNormEps
        )
        let ids = inputIds.asType(.int32)
        MLX.eval(ids)
        let rawIds = ids.asArray(Int32.self)
        let positions = rawIds.enumerated().compactMap { index, token in
            token == Int32(config.imageTokenId) ? index : nil
        }
        guard positions.count == features.dim(0) else {
            throw MuseGlimmerError.unsupportedConfiguration(
                "Muse Glimmer prompt contains \(positions.count) image tokens but the vision tower produced \(features.dim(0)) features."
            )
        }
        guard !positions.isEmpty else { return hidden }
        var parts: [MLXArray] = []
        var cursor = 0
        for (featureIndex, position) in positions.enumerated() {
            if position > cursor {
                parts.append(hidden[0..., cursor..<position, 0...])
            }
            parts.append(features[featureIndex..<(featureIndex + 1), 0...].expandedDimensions(axis: 0))
            cursor = position + 1
        }
        if cursor < hidden.dim(1) {
            parts.append(hidden[0..., cursor..., 0...])
        }
        hidden = parts.count == 1 ? parts[0] : MLX.concatenated(parts, axis: 1)
        return hidden
    }
}

public final class MuseGlimmerModel: Module, @unchecked Sendable {
    @ModuleInfo(key: "model") var model: MuseGlimmerBackbone
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public let config: MuseGlimmerConfig

    public init(config: MuseGlimmerConfig) {
        self.config = config
        self._model.wrappedValue = MuseGlimmerBackbone(config: config)
        self._lmHead.wrappedValue = Linear(
            config.textConfig.hiddenSize,
            config.textConfig.vocabSize,
            bias: false
        )
        super.init()
    }

    func forwardPrefill(
        inputIds: MLXArray,
        imageBatch: MuseGlimmerImageBatch?,
        cache: [Gemma4AttentionCache]
    ) throws -> MLXArray {
        try forwardPrefillDetailed(
            inputIds: inputIds,
            imageBatch: imageBatch,
            cache: cache
        ).logits
    }

    func forwardPrefillDetailed(
        inputIds: MLXArray,
        imageBatch: MuseGlimmerImageBatch?,
        cache: [Gemma4AttentionCache],
        captureLayerIndices: Set<Int> = []
    ) throws -> MuseGlimmerForwardOutput {
        let output = model.languageModel.forward(
            try model.embeddings(inputIds: inputIds, imageBatch: imageBatch),
            cache: cache,
            captureLayerIndices: captureLayerIndices
        )
        var hidden = output.hidden
        if hidden.dim(1) > 1 {
            hidden = hidden[0..., (hidden.dim(1) - 1)..., 0...]
        }
        return MuseGlimmerForwardOutput(
            logits: logits(hidden),
            capturedHiddenStates: output.capturedHiddenStates
        )
    }

    func callAsFunction(_ inputIds: MLXArray, cache: [Gemma4AttentionCache]) -> MLXArray {
        var hidden = forward(inputIds, cache: cache).logits
        if hidden.dim(1) > 1 {
            hidden = hidden[0..., (hidden.dim(1) - 1)..., 0...]
        }
        return hidden
    }

    func forward(
        _ inputIds: MLXArray,
        cache: [Gemma4AttentionCache],
        captureLayerIndices: Set<Int> = []
    ) -> MuseGlimmerForwardOutput {
        let output = model.languageModel.forward(
            model.languageModel.embeddings(inputIds),
            cache: cache,
            captureLayerIndices: captureLayerIndices
        )
        return MuseGlimmerForwardOutput(
            logits: logits(output.hidden),
            capturedHiddenStates: output.capturedHiddenStates
        )
    }

    func makeCache() -> [Gemma4AttentionCache] {
        model.languageModel.makeCache()
    }

    func rawInputEmbeddings(for inputIds: MLXArray) -> MLXArray {
        model.languageModel.embedTokens(inputIds.asType(.int32))
    }

    func inputEmbeddings(for inputIds: MLXArray) -> MLXArray {
        model.languageModel.embeddings(inputIds)
    }

    func rawOutputLogits(from hidden: MLXArray) -> MLXArray {
        lmHead(hidden)
    }

    func outputLogits(from hidden: MLXArray) -> MLXArray {
        logits(hidden)
    }

    private func logits(_ hidden: MLXArray) -> MLXArray {
        var output = lmHead(hidden) * config.textConfig.outputMultiplier
        let cap = config.textConfig.finalLogitSoftcapping
        if cap > 0 {
            output = MLX.tanh(output / cap) * cap
        }
        return output
    }
}
