import Foundation
import MLX
import MLXNN

/// Audio-free H3 with the released fused PDD output heads. All checkpoint tensors retain original keys.
public final class MiniMaxH3ImageTransformer: Module {
    @ModuleInfo(key: "video_patch_proj") var imageInput: Linear
    @ModuleInfo(key: "condition_proj") var textInput: Linear
    @ModuleInfo(key: "time_embedder") var timeEmbedder: MiniMaxH3TimeEmbedding
    @ModuleInfo(key: "token_refiner") var tokenRefiner: MiniMaxH3TokenRefiner
    @ModuleInfo(key: "blocks") var blocks: [MiniMaxH3TransformerBlock]
    @ModuleInfo(key: "final_layer") var finalLayer: MiniMaxH3ImageOutput
    let configuration: MiniMaxH3TransformerConfiguration
    package var maximumProjectionRows = 8_192
    package var maximumAttentionHeadsPerKernel = 8
    package var maximumAttentionQueryTokens = 640
    package var profile: ((String, Int, Int, Double) -> Void)?
    private var modulationCache: MiniMaxH3ImageModulationCache?

    public init(configuration: MiniMaxH3TransformerConfiguration = .init()) {
        self.configuration = configuration
        _imageInput.wrappedValue = Linear(configuration.videoPatchDimension, configuration.hiddenSize)
        _textInput.wrappedValue = Linear(configuration.textDimension, configuration.hiddenSize)
        _timeEmbedder.wrappedValue = MiniMaxH3TimeEmbedding(configuration: configuration)
        _tokenRefiner.wrappedValue = MiniMaxH3TokenRefiner(configuration: configuration)
        _blocks.wrappedValue = (0..<configuration.layerCount).map { _ in MiniMaxH3TransformerBlock(configuration: configuration, includeAdaLN: true) }
        _finalLayer.wrappedValue = MiniMaxH3ImageOutput(configuration: configuration)
    }

    public func prepareText(_ states: MLXArray) -> MLXArray {
        let text = tokenRefiner(miniMaxH3Linear(textInput, states))
        MLX.eval(text)
        return text
    }

    public func callAsFunction(imageRows: MLXArray, text: MLXArray, layout: MiniMaxH3ImageLayout, step: Int) throws -> MLXArray {
        try self(imageRows: imageRows, text: text, preparedLayout: prepareLayout(layout), step: step)
    }

    private func embedTime(step: Int) -> MLXArray {
        let timestep = MiniMaxH3ImageSampling.timesteps[step]
        let half = configuration.timeFrequencyDimension / 2
        let frequencies = MLXArray((0..<half).map { exp(-log(Float(10_000)) * Float($0) / Float(half)) })
        let arguments = MLXArray([timestep, MiniMaxH3ImageSampling.referenceTimestep]).reshaped(2, 1) * frequencies.reshaped(1, -1)
        return timeEmbedder(MLX.concatenated([MLX.cos(arguments), MLX.sin(arguments)], axis: -1))
    }

    public func prepareLayout(_ layout: MiniMaxH3ImageLayout) -> MiniMaxH3ImagePreparedLayout {
        let inverse = MLXArray((0..<configuration.ropeFrequencyCount).map {
            1 / pow(configuration.ropeTheta, Float($0) / Float(configuration.ropeFrequencyCount))
        })
        let angles = (layout.positions.expandedDimensions(axis: -1) * inverse.reshaped(1, 1, -1))
            .reshaped(-1, 3 * configuration.ropeFrequencyCount)
        let rotary = MLX.concatenated([angles, angles], axis: -1).reshaped(1, -1, 1, 6 * configuration.ropeFrequencyCount)
        // The released attention casts rotary tables to Q/K's dtype before
        // multiplying. Keeping FP32 tables here promotes the entire BF16
        // attention and following residual stream to FP32.
        let attentionType = blocks[0].attention.queryNorm.weight.dtype
        let rope = MiniMaxH3RotaryEmbedding(cosine: MLX.cos(rotary).asType(attentionType),
                                           sine: MLX.sin(rotary).asType(attentionType))
        let indices = layout.adaLNIndices()
        MLX.eval(rope.cosine, rope.sine, indices)
        return MiniMaxH3ImagePreparedLayout(rope: rope, indices: indices, targetRows: layout.targetRows)
    }

    public func callAsFunction(imageRows: MLXArray, text: MLXArray, preparedLayout layout: MiniMaxH3ImagePreparedLayout, step: Int) throws -> MLXArray {
        let time = modulationCache?.timeEmbeddings[step] ?? embedTime(step: step)
        var hidden = MLX.concatenated([text, miniMaxH3Linear(imageInput, imageRows).asType(text.dtype)], axis: 1)
        for (index, block) in blocks.enumerated() {
            try Task.checkCancellation()
            let started = Date.timeIntervalSinceReferenceDate
            let modulation = modulationCache?.blocks[index][step]
            if hidden.dim(1) > maximumProjectionRows {
                hidden = try chunkedBlock(block, hidden: hidden, time: time, indices: layout.indices, rope: layout.rope, cachedModulation: modulation,
                    record: profile.map { callback in { phase, seconds in callback(phase, step, index, seconds) } })
            } else {
                hidden = block(hidden, timeEmbedding: time, adaLNIndices: layout.indices, rope: layout.rope, cachedModulation: modulation)
            }
            MLX.eval(hidden)
            profile?("block", step, index, Date.timeIntervalSinceReferenceDate - started)
        }
        return finalLayer(hidden[0..., layout.targetRows, 0...], timeEmbedding: time, step: step,
                          cachedModulation: modulationCache?.finalModulations[step])
    }

    package func precomputeModulations() -> MiniMaxH3ImageModulationCache {
        let times = (0..<6).map { step in
            let time = embedTime(step: step)
            MLX.eval(time)
            return time
        }
        let modulations = blocks.map { block in
            let values = times.map { time in
                let value = block.precomputeModulation(timeEmbedding: time)
                MLX.eval(value)
                return value
            }
            let value = MLX.stacked(values)
            MLX.eval(value)
            return value
        }
        let final = MLX.stacked(times.map { finalLayer.precomputeModulation($0) })
        MLX.eval(final)
        return MiniMaxH3ImageModulationCache(timeEmbeddings: MLX.stacked(times), blocks: modulations, finalModulations: final)
    }

    package func installModulationCache(_ cache: MiniMaxH3ImageModulationCache) {
        precondition(cache.matches(configuration))
        modulationCache = cache
        update(modules: ModuleChildren.unflattened([("time_embedder", MiniMaxH3TimeEmbedding(discarded: ()))]))
        for block in blocks { block.discardAdaLNWeights() }
        finalLayer.discardModulationWeights()
    }

    /// Materialize one block at a time, releasing its packed Q8 tensors before
    /// advancing, so conversion does not retain two complete transformer copies.
    package func materializeResidentBF16() {
        for block in tokenRefiner.blocks {
            _ = miniMaxH3MaterializeResidentBF16(in: block)
            miniMaxH3EvaluateParameters(in: block)
            Memory.clearCache()
        }
        for block in blocks {
            _ = miniMaxH3MaterializeResidentBF16(in: block)
            miniMaxH3EvaluateParameters(in: block)
            Memory.clearCache()
        }
        if let resident = miniMaxH3ResidentBF16Linear(textInput) {
            update(modules: ModuleChildren.unflattened([("condition_proj", resident.linear)]))
        }
        miniMaxH3EvaluateParameters(in: self)
        Memory.clearCache()
    }

    /// Bound fused QKV and SwiGLU intermediates while retaining full attention over every packed row.
    private func chunkedBlock(_ block: MiniMaxH3TransformerBlock, hidden: MLXArray,
                              time: MLXArray, indices: MLXArray, rope: MiniMaxH3RotaryEmbedding, cachedModulation: MLXArray?,
                              record: ((String, Double) -> Void)?) throws -> MLXArray {
        var started = Date.timeIntervalSinceReferenceDate
        let projectionRows = min(maximumProjectionRows, 4_096)
        let ranges = stride(from: 0, to: hidden.dim(1), by: projectionRows).map {
            $0..<min($0 + projectionRows, hidden.dim(1))
        }
        let modulation = cachedModulation ?? block.precomputeModulation(timeEmbedding: time)
        var projections: [[MLXArray]] = []
        for range in ranges {
            try Task.checkCancellation()
            let slice = MiniMaxH3RotaryEmbedding(cosine: rope.cosine[0..., range, 0..., 0...],
                                                sine: rope.sine[0..., range, 0..., 0...])
            let projected = block.attentionProjection(hidden[0..., range, 0...], timeEmbedding: time,
                adaLNIndices: indices[range], rope: slice, cachedModulation: modulation)
            MLX.eval(projected)
            projections.append(projected)
        }
        let query = MLX.concatenated(projections.map { $0[0] }, axis: 2)
        let key = MLX.concatenated(projections.map { $0[1] }, axis: 2)
        let value = MLX.concatenated(projections.map { $0[2] }, axis: 2)
        MLX.eval(query, key, value)
        record?("qkv", Date.timeIntervalSinceReferenceDate - started)
        started = Date.timeIntervalSinceReferenceDate
        let attended = block.scaledDotProductAttention(queries: query, keys: key, values: value,
            maximumQueryTokens: maximumAttentionQueryTokens,
            maximumHeadsPerKernel: maximumAttentionHeadsPerKernel, maximumKernelsPerEvaluation: 1)
        MLX.eval(attended)
        record?("attention", Date.timeIntervalSinceReferenceDate - started)
        started = Date.timeIntervalSinceReferenceDate
        var outputs: [MLXArray] = []
        for (index, range) in ranges.enumerated() {
            try Task.checkCancellation()
            let residual = block.attentionProjectionResidual(hidden[0..., range, 0...],
                attended: attended[0..., 0..., range, 0...], gate: projections[index][3])
            let output = block.feedForwardResidual(residual, timeEmbedding: time,
                adaLNIndices: indices[range], cachedModulation: modulation)
            MLX.eval(output)
            outputs.append(output)
        }
        let result = MLX.concatenated(outputs, axis: 1)
        MLX.eval(result)
        record?("mlp_and_residual", Date.timeIntervalSinceReferenceDate - started)
        return result
    }
}

final class MiniMaxH3ImageOutput: Module {
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "adaln_proj") var adaLN: MiniMaxH3AdaLNProjection
    @ParameterInfo(key: "pdd_weight") var weights: MLXArray
    @ParameterInfo(key: "pdd_bias") var biases: MLXArray

    init(configuration: MiniMaxH3TransformerConfiguration) {
        _norm.wrappedValue = RMSNorm(dimensions: configuration.hiddenSize, eps: configuration.normEpsilon)
        _adaLN.wrappedValue = MiniMaxH3AdaLNProjection(inputDimension: configuration.timeEmbeddingDimension,
            hiddenSize: configuration.hiddenSize, partCount: 2, modalityCount: 1)
        _weights.wrappedValue = MLXArray.zeros([6, configuration.videoPatchDimension, configuration.hiddenSize])
        _biases.wrappedValue = MLXArray.zeros([6, configuration.videoPatchDimension])
    }

    func precomputeModulation(_ time: MLXArray) -> MLXArray {
        MLX.concatenated(adaLN(time), axis: -1)
    }

    func discardModulationWeights() {
        update(modules: ModuleChildren.unflattened([("adaln_proj", MiniMaxH3AdaLNProjection(discarded: ()))]))
    }

    func callAsFunction(_ value: MLXArray, timeEmbedding: MLXArray, step: Int, cachedModulation: MLXArray?) -> MLXArray {
        let modulation = cachedModulation.map { MLX.split($0, parts: 2, axis: -1) } ?? adaLN(timeEmbedding)
        let hidden = norm(value) * (1 + modulation[1][0]) + modulation[0][0]
        return MLX.matmul(hidden.asType(weights.dtype), weights[step].T) + biases[step]
    }
}

public struct MiniMaxH3ImagePreparedLayout {
    let rope: MiniMaxH3RotaryEmbedding
    let indices: MLXArray
    let targetRows: Range<Int>
}
