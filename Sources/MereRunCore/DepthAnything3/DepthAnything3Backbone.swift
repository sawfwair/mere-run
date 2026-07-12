import Foundation
@preconcurrency import MLX
import MLXFast
import MLXNN

private final class DA3BackboneAttention: Module {
    let headCount: Int
    let headDimension: Int
    let scale: Float
    let rotaryBaseFrequency: Float
    let usesRotaryEmbedding: Bool

    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "q_norm") var queryNorm: LayerNorm?
    @ModuleInfo(key: "k_norm") var keyNorm: LayerNorm?
    @ModuleInfo(key: "proj") var projection: Linear

    init(
        hiddenSize: Int,
        headCount: Int,
        queryKeyNorm: Bool,
        queryKeyNormEpsilon: Float,
        usesRotaryEmbedding: Bool,
        rotaryBaseFrequency: Float
    ) {
        self.headCount = headCount
        self.headDimension = hiddenSize / headCount
        self.scale = 1 / sqrt(Float(headDimension))
        self.rotaryBaseFrequency = rotaryBaseFrequency
        self.usesRotaryEmbedding = usesRotaryEmbedding
        self._qkv.wrappedValue = Linear(hiddenSize, hiddenSize * 3, bias: true)
        self._queryNorm.wrappedValue = queryKeyNorm
            ? LayerNorm(dimensions: headDimension, eps: queryKeyNormEpsilon)
            : nil
        self._keyNorm.wrappedValue = queryKeyNorm
            ? LayerNorm(dimensions: headDimension, eps: queryKeyNormEpsilon)
            : nil
        self._projection.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
        super.init()
    }

    func callAsFunction(_ input: MLXArray, positions: MLXArray?) -> MLXArray {
        let batch = input.dim(0)
        let sequence = input.dim(1)
        let projected = qkv(input).reshaped(batch, sequence, 3, headCount, headDimension)
        var query = projected[0..., 0..., 0, 0..., 0...].transposed(0, 2, 1, 3)
        var key = projected[0..., 0..., 1, 0..., 0...].transposed(0, 2, 1, 3)
        let value = projected[0..., 0..., 2, 0..., 0...].transposed(0, 2, 1, 3)
        if let queryNorm, let keyNorm {
            query = queryNorm(query)
            key = keyNorm(key)
        }
        if usesRotaryEmbedding, let positions {
            query = applyRotaryEmbedding(query, positions: positions)
            key = applyRotaryEmbedding(key, positions: positions)
        }
        let attended = MLXFast.scaledDotProductAttention(
            queries: query,
            keys: key,
            values: value,
            scale: scale,
            mask: .none
        )
        return projection(
            attended.transposed(0, 2, 1, 3)
                .reshaped(batch, sequence, headCount * headDimension)
        )
    }

    private func applyRotaryEmbedding(_ tokens: MLXArray, positions: MLXArray) -> MLXArray {
        precondition(tokens.ndim == 4 && positions.ndim == 3)
        precondition(tokens.dim(0) == positions.dim(0))
        precondition(tokens.dim(2) == positions.dim(1) && positions.dim(2) == 2)
        precondition(tokens.dim(3).isMultiple(of: 4))
        let directionalDimension = tokens.dim(3) / 2
        let frequencyCount = directionalDimension / 2
        let inverseFrequencies = MLXArray((0..<frequencyCount).map { index -> Float in
            let exponent = Float(index * 2) / Float(directionalDimension)
            return 1 / pow(rotaryBaseFrequency, exponent)
        }).asType(tokens.dtype)

        let vertical = tokens[0..., 0..., 0..., 0..<directionalDimension]
        let horizontal = tokens[0..., 0..., 0..., directionalDimension...]
        let verticalRotated = applyOneDimensionalRotary(
            vertical,
            positions: positions[0..., 0..., 0],
            inverseFrequencies: inverseFrequencies
        )
        let horizontalRotated = applyOneDimensionalRotary(
            horizontal,
            positions: positions[0..., 0..., 1],
            inverseFrequencies: inverseFrequencies
        )
        return MLX.concatenated([verticalRotated, horizontalRotated], axis: -1)
    }

    private func applyOneDimensionalRotary(
        _ tokens: MLXArray,
        positions: MLXArray,
        inverseFrequencies: MLXArray
    ) -> MLXArray {
        let dimension = tokens.dim(3)
        let half = dimension / 2
        let anglesHalf = positions.asType(tokens.dtype).expandedDimensions(axis: -1)
            * inverseFrequencies.reshaped(1, 1, -1)
        let angles = MLX.concatenated([anglesHalf, anglesHalf], axis: -1)
            .expandedDimensions(axis: 1)
        let first = tokens[0..., 0..., 0..., 0..<half]
        let second = tokens[0..., 0..., 0..., half...]
        let rotated = MLX.concatenated([-second, first], axis: -1)
        return tokens * MLX.cos(angles) + rotated * MLX.sin(angles)
    }
}

private final class DA3BackboneMLP: Module {
    @ModuleInfo(key: "fc1") var first: Linear
    @ModuleInfo(key: "fc2") var second: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        self._first.wrappedValue = Linear(hiddenSize, intermediateSize, bias: true)
        self._second.wrappedValue = Linear(intermediateSize, hiddenSize, bias: true)
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        second(gelu(first(input)))
    }
}

private final class DA3LayerScale: Module {
    @ParameterInfo(key: "gamma") var gamma: MLXArray

    init(hiddenSize: Int, initialValue: Float) {
        self._gamma.wrappedValue = MLX.ones([hiddenSize], dtype: .float32) * initialValue
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        input * gamma.asType(input.dtype)
    }
}

private final class DA3BackboneBlock: Module {
    @ModuleInfo(key: "norm1") var firstNorm: LayerNorm
    @ModuleInfo(key: "attn") var attention: DA3BackboneAttention
    @ModuleInfo(key: "ls1") var firstScale: DA3LayerScale
    @ModuleInfo(key: "norm2") var secondNorm: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: DA3BackboneMLP
    @ModuleInfo(key: "ls2") var secondScale: DA3LayerScale

    init(configuration: DepthAnything3Configuration, index: Int) {
        self._firstNorm.wrappedValue = LayerNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.backboneLayerNormEpsilon
        )
        self._attention.wrappedValue = DA3BackboneAttention(
            hiddenSize: configuration.hiddenSize,
            headCount: configuration.headCount,
            queryKeyNorm: index >= configuration.queryKeyNormStart,
            queryKeyNormEpsilon: configuration.headLayerNormEpsilon,
            usesRotaryEmbedding: index >= configuration.rotaryEmbeddingStart,
            rotaryBaseFrequency: configuration.rotaryBaseFrequency
        )
        self._firstScale.wrappedValue = DA3LayerScale(
            hiddenSize: configuration.hiddenSize,
            initialValue: 1
        )
        self._secondNorm.wrappedValue = LayerNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.backboneLayerNormEpsilon
        )
        self._mlp.wrappedValue = DA3BackboneMLP(
            hiddenSize: configuration.hiddenSize,
            intermediateSize: configuration.intermediateSize
        )
        self._secondScale.wrappedValue = DA3LayerScale(
            hiddenSize: configuration.hiddenSize,
            initialValue: 1
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray, positions: MLXArray?) -> MLXArray {
        let attended = input + firstScale(attention(firstNorm(input), positions: positions))
        return attended + secondScale(mlp(secondNorm(attended)))
    }
}

private final class DA3PatchEmbed: Module {
    @ModuleInfo(key: "proj") var projection: Conv2d

    init(configuration: DepthAnything3Configuration) {
        self._projection.wrappedValue = Conv2d(
            inputChannels: 3,
            outputChannels: configuration.hiddenSize,
            kernelSize: IntOrPair(configuration.patchSize),
            stride: IntOrPair(configuration.patchSize),
            padding: IntOrPair(0),
            bias: true
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        projection(input)
    }
}

private final class DA3VisionTransformer: Module {
    let configuration: DepthAnything3Configuration

    @ParameterInfo(key: "cls_token") var classToken: MLXArray
    @ParameterInfo(key: "camera_token") var cameraToken: MLXArray
    @ParameterInfo(key: "pos_embed") var positionEmbedding: MLXArray
    @ModuleInfo(key: "patch_embed") var patchEmbed: DA3PatchEmbed
    @ModuleInfo(key: "blocks") var blocks: [DA3BackboneBlock]
    @ModuleInfo(key: "norm") var norm: LayerNorm

    init(configuration: DepthAnything3Configuration) {
        self.configuration = configuration
        self._classToken.wrappedValue = MLX.zeros([1, 1, configuration.hiddenSize], dtype: .float32)
        self._cameraToken.wrappedValue = MLX.zeros([1, 2, configuration.hiddenSize], dtype: .float32)
        self._positionEmbedding.wrappedValue = MLX.zeros(
            [1, configuration.positionGridSize * configuration.positionGridSize + 1, configuration.hiddenSize],
            dtype: .float32
        )
        self._patchEmbed.wrappedValue = DA3PatchEmbed(configuration: configuration)
        self._blocks.wrappedValue = (0..<configuration.layerCount).map {
            DA3BackboneBlock(configuration: configuration, index: $0)
        }
        self._norm.wrappedValue = LayerNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.headLayerNormEpsilon
        )
        super.init()
    }

    func callAsFunction(
        _ normalizedImages: MLXArray,
        suppliedCameraToken: MLXArray?,
        referenceViewStrategy: DepthAnything3ReferenceViewStrategy
    ) -> DepthAnything3BackboneOutput {
        precondition(normalizedImages.ndim == 5 && normalizedImages.dim(4) == 3)
        let batch = normalizedImages.dim(0)
        let views = normalizedImages.dim(1)
        let height = normalizedImages.dim(2)
        let width = normalizedImages.dim(3)
        precondition(height.isMultiple(of: configuration.patchSize))
        precondition(width.isMultiple(of: configuration.patchSize))
        if let suppliedCameraToken {
            precondition(suppliedCameraToken.shape == [batch, views, configuration.hiddenSize])
        }

        let flattened = normalizedImages.reshaped(batch * views, height, width, 3)
        let patches = patchEmbed(flattened)
        let patchHeight = patches.dim(1)
        let patchWidth = patches.dim(2)
        let patchCount = patchHeight * patchWidth
        let flattenedPatches = patches.reshaped(batch * views, patchCount, configuration.hiddenSize)
        let cls = MLX.broadcast(
            classToken.asType(flattenedPatches.dtype),
            to: [batch * views, 1, configuration.hiddenSize]
        )
        var hidden = MLX.concatenated([cls, flattenedPatches], axis: 1)
        hidden = hidden + interpolatedPosition(
            patchHeight: patchHeight,
            patchWidth: patchWidth,
            dtype: hidden.dtype
        )
        hidden = hidden.reshaped(batch, views, patchCount + 1, configuration.hiddenSize)

        let localPositions = positionGrid(
            batch: batch,
            views: views,
            patchHeight: patchHeight,
            patchWidth: patchWidth,
            noSpatialDifference: false,
            dtype: hidden.dtype
        )
        let globalPositions = positionGrid(
            batch: batch,
            views: views,
            patchHeight: patchHeight,
            patchWidth: patchWidth,
            noSpatialDifference: true,
            dtype: hidden.dtype
        )

        var localHidden = hidden
        var selectedReferences: [Int]?
        var outputFeatures: [MLXArray] = []
        var finalCameraToken: MLXArray?
        for (index, block) in blocks.enumerated() {
            if index == configuration.alternateAttentionStart - 1, views >= 3 {
                let references = selectReferenceViews(hidden, strategy: referenceViewStrategy)
                hidden = reorderViews(hidden, references: references)
                localHidden = reorderViews(localHidden, references: references)
                selectedReferences = references
            }

            if index == configuration.alternateAttentionStart {
                let replacement: MLXArray
                if let suppliedCameraToken {
                    // Matches upstream behavior: supplied camera tokens remain in
                    // caller order even when automatic reference selection has
                    // reordered image features.
                    replacement = suppliedCameraToken.asType(hidden.dtype)
                } else if views == 1 {
                    replacement = MLX.broadcast(
                        cameraToken[0..., 0..<1, 0...],
                        to: [batch, 1, configuration.hiddenSize]
                    )
                } else {
                    let reference = MLX.broadcast(
                        cameraToken[0..., 0..<1, 0...],
                        to: [batch, 1, configuration.hiddenSize]
                    )
                    let sources = MLX.broadcast(
                        cameraToken[0..., 1..<2, 0...],
                        to: [batch, views - 1, configuration.hiddenSize]
                    )
                    replacement = MLX.concatenated([reference, sources], axis: 1)
                }
                hidden = MLX.concatenated(
                    [replacement.expandedDimensions(axis: 2), hidden[0..., 0..., 1..., 0...]],
                    axis: 2
                )
            }

            if index >= configuration.alternateAttentionStart && index % 2 == 1 {
                let flattenedGlobal = hidden.reshaped(
                    batch,
                    views * (patchCount + 1),
                    configuration.hiddenSize
                )
                let position = index >= configuration.rotaryEmbeddingStart
                    ? globalPositions.reshaped(batch, views * (patchCount + 1), 2)
                    : nil
                hidden = block(flattenedGlobal, positions: position)
                    .reshaped(batch, views, patchCount + 1, configuration.hiddenSize)
            } else {
                let flattenedLocal = hidden.reshaped(
                    batch * views,
                    patchCount + 1,
                    configuration.hiddenSize
                )
                let position = index >= configuration.rotaryEmbeddingStart
                    ? localPositions.reshaped(batch * views, patchCount + 1, 2)
                    : nil
                hidden = block(flattenedLocal, positions: position)
                    .reshaped(batch, views, patchCount + 1, configuration.hiddenSize)
                localHidden = hidden
            }

            if configuration.outputLayers.contains(index) {
                var combined = MLX.concatenated([localHidden, hidden], axis: -1)
                if let selectedReferences {
                    combined = restoreViews(combined, references: selectedReferences)
                }
                finalCameraToken = combined[0..., 0..., 0, 0...]
                let local = combined[0..., 0..., 0..., 0..<configuration.hiddenSize]
                let crossView = norm(combined[0..., 0..., 0..., configuration.hiddenSize...])
                let normalized = MLX.concatenated([local, crossView], axis: -1)
                outputFeatures.append(normalized[0..., 0..., 1..., 0...])
            }
        }
        precondition(outputFeatures.count == 4)
        guard let finalCameraToken else {
            preconditionFailure("DA3 output layers did not emit a camera token")
        }
        return DepthAnything3BackboneOutput(
            patchFeatures: outputFeatures,
            cameraToken: finalCameraToken,
            patchHeight: patchHeight,
            patchWidth: patchWidth
        )
    }

    private func interpolatedPosition(
        patchHeight: Int,
        patchWidth: Int,
        dtype: DType
    ) -> MLXArray {
        let classPosition = positionEmbedding[0..., 0..<1, 0...]
        let source = configuration.positionGridSize
        let patchPosition = positionEmbedding[0..., 1..., 0...]
            .reshaped(1, source, source, configuration.hiddenSize)
        let resized: MLXArray
        if patchHeight == source && patchWidth == source {
            resized = patchPosition
        } else {
            resized = dinoV2PyTorchBicubicResize(
                patchPosition.asType(.float32),
                outputHeight: patchHeight,
                outputWidth: patchWidth,
                offset: 0.1
            )
        }
        return MLX.concatenated(
            [classPosition.asType(dtype), resized.reshaped(1, patchHeight * patchWidth, configuration.hiddenSize).asType(dtype)],
            axis: 1
        )
    }

    private func positionGrid(
        batch: Int,
        views: Int,
        patchHeight: Int,
        patchWidth: Int,
        noSpatialDifference: Bool,
        dtype: DType
    ) -> MLXArray {
        var values: [Float] = [0, 0]
        values.reserveCapacity((patchHeight * patchWidth + 1) * 2)
        for y in 0..<patchHeight {
            for x in 0..<patchWidth {
                if noSpatialDifference {
                    values.append(contentsOf: [1, 1])
                } else {
                    values.append(contentsOf: [Float(y + 1), Float(x + 1)])
                }
            }
        }
        let oneView = MLXArray(values).asType(dtype)
            .reshaped(1, 1, patchHeight * patchWidth + 1, 2)
        return MLX.broadcast(
            oneView,
            to: [batch, views, patchHeight * patchWidth + 1, 2]
        )
    }

    private func selectReferenceViews(
        _ input: MLXArray,
        strategy: DepthAnything3ReferenceViewStrategy
    ) -> [Int] {
        let batch = input.dim(0)
        let views = input.dim(1)
        guard views > 1 else { return [Int](repeating: 0, count: batch) }
        switch strategy {
        case .first:
            return [Int](repeating: 0, count: batch)
        case .middle:
            return [Int](repeating: views / 2, count: batch)
        case .saddleBalanced, .saddleSimilarityRange:
            break
        }

        let channels = input.dim(3)
        let classFeatures = input[0..., 0..., 0, 0...].asType(.float32)
        MLX.eval(classFeatures)
        let values = classFeatures.asArray(Float.self)
        var selected: [Int] = []
        selected.reserveCapacity(batch)
        for batchIndex in 0..<batch {
            let base = batchIndex * views * channels
            var normalized = [[Float]]()
            var norms = [Float]()
            var variances = [Float]()
            for view in 0..<views {
                let offset = base + view * channels
                let vector = Array(values[offset..<(offset + channels)])
                let squaredNorm = vector.reduce(Float(0)) { $0 + $1 * $1 }
                let length = sqrt(max(squaredNorm, 1e-20))
                let unit = vector.map { $0 / length }
                normalized.append(unit)
                norms.append(length)
                let mean = unit.reduce(0, +) / Float(channels)
                variances.append(unit.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Float(max(1, channels - 1)))
            }
            var similarity = [Float](repeating: 0, count: views * views)
            for row in 0..<views {
                for column in 0..<views {
                    var dot: Float = 0
                    for channel in 0..<channels {
                        dot += normalized[row][channel] * normalized[column][channel]
                    }
                    similarity[row * views + column] = row == column ? dot - 1 : dot
                }
            }
            switch strategy {
            case .saddleBalanced:
                let similarityScores = (0..<views).map { row in
                    (0..<views).reduce(Float(0)) { $0 + similarity[row * views + $1] }
                        / Float(views - 1)
                }
                let normalizedSimilarity = normalizeMetric(similarityScores)
                let normalizedNorms = normalizeMetric(norms)
                let normalizedVariances = normalizeMetric(variances)
                var scores = [Float]()
                scores.reserveCapacity(views)
                for view in 0..<views {
                    let similarityDistance = abs(normalizedSimilarity[view] - 0.5)
                    let normDistance = abs(normalizedNorms[view] - 0.5)
                    let varianceDistance = abs(normalizedVariances[view] - 0.5)
                    scores.append(similarityDistance + normDistance + varianceDistance)
                }
                let selectedView = scores.enumerated().min { $0.element < $1.element }?.offset ?? 0
                selected.append(selectedView)
            case .saddleSimilarityRange:
                let ranges = (0..<views).map { row -> Float in
                    let rowValues = Array(similarity[(row * views)..<((row + 1) * views)])
                    return (rowValues.max() ?? 0) - (rowValues.min() ?? 0)
                }
                let selectedView = ranges.enumerated().max { $0.element < $1.element }?.offset ?? 0
                selected.append(selectedView)
            case .first, .middle:
                preconditionFailure("handled above")
            }
        }
        return selected
    }

    private func normalizeMetric(_ values: [Float]) -> [Float] {
        let minimum = values.min() ?? 0
        let maximum = values.max() ?? 0
        return values.map { ($0 - minimum) / (maximum - minimum + 1e-8) }
    }

    private func reorderViews(_ input: MLXArray, references: [Int]) -> MLXArray {
        let batch = input.dim(0)
        let views = input.dim(1)
        precondition(references.count == batch)
        let batches = (0..<batch).map { batchIndex -> MLXArray in
            let reference = references[batchIndex]
            let order = [reference] + (0..<views).filter { $0 != reference }
            return MLX.concatenated(order.map {
                input[batchIndex..<(batchIndex + 1), $0..<($0 + 1), 0..., 0...]
            }, axis: 1)
        }
        return MLX.concatenated(batches, axis: 0)
    }

    private func restoreViews(_ input: MLXArray, references: [Int]) -> MLXArray {
        let batch = input.dim(0)
        let views = input.dim(1)
        precondition(references.count == batch)
        let batches = (0..<batch).map { batchIndex -> MLXArray in
            let reference = references[batchIndex]
            let order = (0..<views).map { original -> Int in
                if original == reference { return 0 }
                return original < reference ? original + 1 : original
            }
            return MLX.concatenated(order.map {
                input[batchIndex..<(batchIndex + 1), $0..<($0 + 1), 0..., 0...]
            }, axis: 1)
        }
        return MLX.concatenated(batches, axis: 0)
    }
}

final class DepthAnything3Backbone: Module {
    @ModuleInfo(key: "pretrained") private var pretrained: DA3VisionTransformer

    init(configuration: DepthAnything3Configuration) {
        self._pretrained.wrappedValue = DA3VisionTransformer(configuration: configuration)
        super.init()
    }

    func callAsFunction(
        _ normalizedImages: MLXArray,
        suppliedCameraToken: MLXArray?,
        referenceViewStrategy: DepthAnything3ReferenceViewStrategy
    ) -> DepthAnything3BackboneOutput {
        pretrained(
            normalizedImages,
            suppliedCameraToken: suppliedCameraToken,
            referenceViewStrategy: referenceViewStrategy
        )
    }
}
