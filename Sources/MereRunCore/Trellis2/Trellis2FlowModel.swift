import Foundation
@preconcurrency import MLX
import MLXFast
import MLXNN

public enum Trellis2ModelError: Error, Equatable, LocalizedError, Sendable {
    case missingWeight(String)
    case invalidWeightShape(name: String, expected: [Int], actual: [Int])
    case invalidInputShape(expected: String, actual: [Int])
    case emptySparseStructure
    case emptyExtractedMesh
    case excessiveSparseStructure(actual: Int, maximum: Int)
    case unsupportedCheckpoint(String)

    public var errorDescription: String? {
        switch self {
        case .missingWeight(let name):
            "TRELLIS.2 checkpoint is missing tensor '\(name)'."
        case .invalidWeightShape(let name, let expected, let actual):
            "TRELLIS.2 tensor '\(name)' has shape \(actual); expected \(expected)."
        case .invalidInputShape(let expected, let actual):
            "TRELLIS.2 input has shape \(actual); expected \(expected)."
        case .emptySparseStructure:
            "TRELLIS.2 produced an empty sparse structure."
        case .emptyExtractedMesh:
            "TRELLIS.2 decoded sparse voxels but did not produce a connected mesh."
        case .excessiveSparseStructure(let actual, let maximum):
            "TRELLIS.2 produced \(actual) sparse tokens, exceeding the \(maximum)-token safety limit."
        case .unsupportedCheckpoint(let reason):
            "Unsupported TRELLIS.2 checkpoint: \(reason)"
        }
    }
}

/// Direct safetensors view used by the TRELLIS.2 functional runtime. Keeping
/// PyTorch names intact lets mere.run load Microsoft's BF16 files without a
/// multi-gigabyte conversion or a second on-disk copy.
struct Trellis2WeightStore {
    let values: [String: MLXArray]

    init(url: URL) throws {
        self.values = try MLX.loadArrays(url: url)
    }

    init(values: [String: MLXArray]) {
        self.values = values
    }

    func tensor(_ name: String) throws -> MLXArray {
        guard let value = values[name] else { throw Trellis2ModelError.missingWeight(name) }
        return value
    }

    func checkedTensor(_ name: String, shape: [Int]) throws -> MLXArray {
        let value = try tensor(name)
        guard value.shape == shape else {
            throw Trellis2ModelError.invalidWeightShape(name: name, expected: shape, actual: value.shape)
        }
        return value
    }

    func linear(_ input: MLXArray, prefix: String) throws -> MLXArray {
        let weight = try tensor("\(prefix).weight")
        guard weight.ndim == 2 else {
            throw Trellis2ModelError.invalidInputShape(expected: "rank-2 linear weight", actual: weight.shape)
        }
        var output = MLX.matmul(input.asType(weight.dtype), weight.transposed())
        if let bias = values["\(prefix).bias"] {
            output = output + bias
        }
        return output
    }
}

struct Trellis2RotaryEmbedding {
    let cosine: MLXArray
    let sine: MLXArray
    let headDimension: Int

    init(coordinates: [Trellis2VoxelCoordinate], headDimension: Int) {
        precondition(headDimension.isMultiple(of: 2))
        self.headDimension = headDimension
        let pairCount = headDimension / 2
        let frequencyCount = pairCount / 3
        var frequencies = [Float]()
        frequencies.reserveCapacity(frequencyCount)
        for index in 0..<frequencyCount {
            frequencies.append(1 / pow(10_000, Float(index) / Float(frequencyCount)))
        }

        var angles = [Float]()
        angles.reserveCapacity(coordinates.count * pairCount)
        for coordinate in coordinates {
            for axis in 0..<3 {
                let position = Float(coordinate[axis])
                for frequency in frequencies {
                    angles.append(position * frequency)
                }
            }
            angles.append(contentsOf: repeatElement(0, count: pairCount - 3 * frequencyCount))
        }
        let phase = MLXArray(angles).reshaped(coordinates.count, pairCount)
        self.cosine = MLX.cos(phase)
        self.sine = MLX.sin(phase)
    }

    func apply(_ input: MLXArray) -> MLXArray {
        precondition(input.ndim == 4 && input.dim(3) == headDimension)
        let batch = input.dim(0)
        let length = input.dim(1)
        let heads = input.dim(2)
        let pairs = headDimension / 2
        let paired = input.asType(.float32).reshaped(batch, length, heads, pairs, 2)
        let real = paired[0..., 0..., 0..., 0..., 0..<1]
        let imaginary = paired[0..., 0..., 0..., 0..., 1..<2]
        let cosValue = cosine.expandedDimensions(axes: [0, 2, 4])
        let sinValue = sine.expandedDimensions(axes: [0, 2, 4])
        return MLX.concatenated([
            real * cosValue - imaginary * sinValue,
            real * sinValue + imaginary * cosValue,
        ], axis: -1).reshaped(batch, length, heads, headDimension).asType(input.dtype)
    }
}

/// Native MLX implementation of both the dense sparse-structure DiT and the
/// sparse shape/texture SLat DiTs. The three official checkpoints share this
/// exact 30-block architecture; only input channels and token coordinates
/// differ.
struct Trellis2FlowModel {
    let configuration: Trellis2FlowConfiguration
    private let weights: Trellis2WeightStore
    private let dynamicSparseRuntime: DynamicSparseAttentionRuntime?

    init(configuration: Trellis2FlowConfiguration, checkpointURL: URL) throws {
        self.configuration = configuration
        self.weights = try Trellis2WeightStore(url: checkpointURL)
        self.dynamicSparseRuntime = DynamicSparseAttentionRuntime.configured(model: .trellis2)
        try validateWeights()
    }

    init(configuration: Trellis2FlowConfiguration, weights: [String: MLXArray]) throws {
        self.configuration = configuration
        self.weights = Trellis2WeightStore(values: weights)
        self.dynamicSparseRuntime = DynamicSparseAttentionRuntime.configured(model: .trellis2)
        try validateWeights()
    }

    func beginDenoisingStep(index: Int, count: Int) {
        dynamicSparseRuntime?.beginStep(index: index, count: count)
    }

    private func validateWeights() throws {
        let channels = configuration.modelChannels
        let conditionChannels = configuration.conditionChannels
        let headDimension = configuration.headDimension
        _ = try weights.checkedTensor("input_layer.weight", shape: [channels, configuration.inputChannels])
        _ = try weights.checkedTensor("out_layer.weight", shape: [configuration.outputChannels, channels])
        _ = try weights.checkedTensor("t_embedder.mlp.0.weight", shape: [channels, 256])
        _ = try weights.checkedTensor("t_embedder.mlp.2.weight", shape: [channels, channels])
        _ = try weights.checkedTensor("adaLN_modulation.1.weight", shape: [6 * channels, channels])
        for index in 0..<configuration.blockCount {
            let prefix = "blocks.\(index)"
            _ = try weights.checkedTensor("\(prefix).modulation", shape: [6 * channels])
            _ = try weights.checkedTensor("\(prefix).self_attn.to_qkv.weight", shape: [3 * channels, channels])
            _ = try weights.checkedTensor(
                "\(prefix).cross_attn.to_kv.weight",
                shape: [2 * channels, conditionChannels]
            )
            _ = try weights.checkedTensor(
                "\(prefix).self_attn.q_rms_norm.gamma",
                shape: [configuration.headCount, headDimension]
            )
            _ = try weights.checkedTensor(
                "\(prefix).cross_attn.k_rms_norm.gamma",
                shape: [configuration.headCount, headDimension]
            )
        }
    }

    func predict(
        input: MLXArray,
        coordinates: [Trellis2VoxelCoordinate],
        timestep: Float,
        condition: MLXArray
    ) throws -> MLXArray {
        guard input.ndim == 3,
              input.dim(1) == coordinates.count,
              input.dim(2) == configuration.inputChannels else {
            throw Trellis2ModelError.invalidInputShape(
                expected: "[batch, \(coordinates.count), \(configuration.inputChannels)]",
                actual: input.shape
            )
        }
        guard condition.ndim == 3,
              condition.dim(0) == input.dim(0),
              condition.dim(2) == configuration.conditionChannels else {
            throw Trellis2ModelError.invalidInputShape(
                expected: "[\(input.dim(0)), tokens, \(configuration.conditionChannels)] conditioning",
                actual: condition.shape
            )
        }

        let modelType = try weights.tensor("input_layer.weight").dtype
        var hidden = try weights.linear(input, prefix: "input_layer").asType(modelType)
        let timeEmbedding = try timeEmbedding(timestep: timestep, batch: input.dim(0)).asType(modelType)
        let sharedModulation = try weights.linear(silu(timeEmbedding), prefix: "adaLN_modulation.1")
            .asType(modelType)
        let context = condition.asType(modelType)
        let rotary = Trellis2RotaryEmbedding(
            coordinates: coordinates,
            headDimension: configuration.headDimension
        )

        for index in 0..<configuration.blockCount {
            hidden = try block(
                hidden,
                context: context,
                sharedModulation: sharedModulation,
                rotary: rotary,
                index: index
            )
            MLX.eval(hidden)
            Memory.clearCache()
        }

        hidden = MLXFast.layerNorm(hidden.asType(.float32), eps: 1e-6).asType(modelType)
        return try weights.linear(hidden, prefix: "out_layer").asType(input.dtype)
    }

    private func timeEmbedding(timestep: Float, batch: Int) throws -> MLXArray {
        let half = 128
        var values = [Float]()
        values.reserveCapacity(256)
        for index in 0..<half {
            let frequency = exp(-log(10_000) * Float(index) / Float(half))
            values.append(cos(timestep * frequency))
        }
        for index in 0..<half {
            let frequency = exp(-log(10_000) * Float(index) / Float(half))
            values.append(sin(timestep * frequency))
        }
        let frequencyEmbedding = MLX.broadcast(MLXArray(values).reshaped(1, 256), to: [batch, 256])
        var hidden = try weights.linear(frequencyEmbedding, prefix: "t_embedder.mlp.0")
        hidden = silu(hidden)
        return try weights.linear(hidden, prefix: "t_embedder.mlp.2")
    }

    private func block(
        _ input: MLXArray,
        context: MLXArray,
        sharedModulation: MLXArray,
        rotary: Trellis2RotaryEmbedding,
        index: Int
    ) throws -> MLXArray {
        let prefix = "blocks.\(index)"
        let blockModulation = try weights.tensor("\(prefix).modulation")
        let modulation = sharedModulation + blockModulation
        let parts = MLX.split(modulation, parts: 6, axis: -1)
        let shiftAttention = parts[0].expandedDimensions(axis: 1)
        let scaleAttention = parts[1].expandedDimensions(axis: 1)
        let gateAttention = parts[2].expandedDimensions(axis: 1)
        let shiftMLP = parts[3].expandedDimensions(axis: 1)
        let scaleMLP = parts[4].expandedDimensions(axis: 1)
        let gateMLP = parts[5].expandedDimensions(axis: 1)

        let dtype = input.dtype
        var hidden = MLXFast.layerNorm(input.asType(.float32), eps: 1e-6)
        hidden = (hidden * (1 + scaleAttention.asType(.float32)) + shiftAttention.asType(.float32)).asType(dtype)
        hidden = try selfAttention(
            hidden,
            rotary: rotary,
            prefix: "\(prefix).self_attn",
            layerIndex: index
        )
        var output = (input.asType(.float32) + hidden.asType(.float32) * gateAttention.asType(.float32)).asType(dtype)

        hidden = try affineLayerNorm(output, prefix: "\(prefix).norm2")
        hidden = try crossAttention(hidden, context: context, prefix: "\(prefix).cross_attn")
        output = output + hidden

        hidden = MLXFast.layerNorm(output.asType(.float32), eps: 1e-6)
        hidden = (hidden * (1 + scaleMLP.asType(.float32)) + shiftMLP.asType(.float32)).asType(dtype)
        hidden = try weights.linear(hidden, prefix: "\(prefix).mlp.mlp.0")
        hidden = approximateGELU(hidden)
        hidden = try weights.linear(hidden, prefix: "\(prefix).mlp.mlp.2")
        return (output.asType(.float32) + hidden.asType(.float32) * gateMLP.asType(.float32)).asType(dtype)
    }

    private func selfAttention(
        _ input: MLXArray,
        rotary: Trellis2RotaryEmbedding,
        prefix: String,
        layerIndex: Int
    ) throws -> MLXArray {
        let batch = input.dim(0)
        let length = input.dim(1)
        let heads = configuration.headCount
        let headDimension = configuration.headDimension
        let qkv = try weights.linear(input, prefix: "\(prefix).to_qkv")
            .reshaped(batch, length, 3, heads, headDimension)
        let pieces = MLX.split(qkv, parts: 3, axis: 2).map { $0.squeezed(axis: 2) }
        var query = try multiHeadRMSNorm(pieces[0], prefix: "\(prefix).q_rms_norm")
        var key = try multiHeadRMSNorm(pieces[1], prefix: "\(prefix).k_rms_norm")
        query = rotary.apply(query)
        key = rotary.apply(key)
        let attended = attention(
            query: query,
            key: key,
            value: pieces[2],
            dynamicSparseRuntime: dynamicSparseRuntime,
            layerIndex: layerIndex
        )
            .reshaped(batch, length, configuration.modelChannels)
        return try weights.linear(attended, prefix: "\(prefix).to_out")
    }

    private func crossAttention(
        _ input: MLXArray,
        context: MLXArray,
        prefix: String
    ) throws -> MLXArray {
        let batch = input.dim(0)
        let queryLength = input.dim(1)
        let contextLength = context.dim(1)
        let heads = configuration.headCount
        let headDimension = configuration.headDimension
        var query = try weights.linear(input, prefix: "\(prefix).to_q")
            .reshaped(batch, queryLength, heads, headDimension)
        let keyValue = try weights.linear(context, prefix: "\(prefix).to_kv")
            .reshaped(batch, contextLength, 2, heads, headDimension)
        let pieces = MLX.split(keyValue, parts: 2, axis: 2).map { $0.squeezed(axis: 2) }
        query = try multiHeadRMSNorm(query, prefix: "\(prefix).q_rms_norm")
        let key = try multiHeadRMSNorm(pieces[0], prefix: "\(prefix).k_rms_norm")
        let attended = attention(query: query, key: key, value: pieces[1])
            .reshaped(batch, queryLength, configuration.modelChannels)
        return try weights.linear(attended, prefix: "\(prefix).to_out")
    }

    private func multiHeadRMSNorm(_ input: MLXArray, prefix: String) throws -> MLXArray {
        let gamma = try weights.tensor("\(prefix).gamma").asType(.float32)
        let squaredNorm = MLX.sum(input.asType(.float32) * input.asType(.float32), axis: -1, keepDims: true)
        let normalized = input.asType(.float32) * MLX.rsqrt(MLX.maximum(squaredNorm, MLXArray(1e-24)))
        let scale = MLXArray(sqrt(Float(configuration.headDimension)))
        return (normalized * gamma * scale).asType(input.dtype)
    }

    private func attention(
        query: MLXArray,
        key: MLXArray,
        value: MLXArray,
        dynamicSparseRuntime: DynamicSparseAttentionRuntime? = nil,
        layerIndex: Int = 0
    ) -> MLXArray {
        let q = query.transposed(0, 2, 1, 3)
        let k = key.transposed(0, 2, 1, 3)
        let v = value.transposed(0, 2, 1, 3)
        let scale = 1 / sqrt(Float(configuration.headDimension))
        let sparse = dynamicSparseRuntime?.call(
            queries: q,
            keys: k,
            values: v,
            layerIndex: layerIndex,
            prefixTokenCount: 0,
            scale: scale
        )
        return (sparse ?? MLXFast.scaledDotProductAttention(
            queries: q,
            keys: k,
            values: v,
            scale: scale,
            mask: .none
        )).transposed(0, 2, 1, 3)
    }

    private func affineLayerNorm(_ input: MLXArray, prefix: String) throws -> MLXArray {
        try MLXFast.layerNorm(
            input.asType(.float32),
            weight: weights.tensor("\(prefix).weight").asType(.float32),
            bias: weights.tensor("\(prefix).bias").asType(.float32),
            eps: 1e-6
        ).asType(input.dtype)
    }

    private func approximateGELU(_ input: MLXArray) -> MLXArray {
        let value = input.asType(.float32)
        let cubic = value * value * value
        let output = 0.5 * value * (1 + MLX.tanh(0.797_884_6 * (value + 0.044715 * cubic)))
        return output.asType(input.dtype)
    }
}

enum Trellis2FlowSampler {
    static func sample(
        noise: MLXArray,
        coordinates: [Trellis2VoxelCoordinate],
        condition: MLXArray,
        negativeCondition: MLXArray,
        model: Trellis2FlowModel,
        configuration: Trellis2SamplerConfiguration,
        staticInput: MLXArray? = nil,
        progress: ((Int, Int) -> Void)? = nil
    ) throws -> MLXArray {
        if let staticInput {
            guard staticInput.ndim == 3,
                  staticInput.dim(0) == noise.dim(0),
                  staticInput.dim(1) == noise.dim(1),
                  staticInput.dim(2) + noise.dim(2) == model.configuration.inputChannels else {
                throw Trellis2ModelError.invalidInputShape(
                    expected: "[batch, tokens, \(model.configuration.inputChannels - noise.dim(2))] static input",
                    actual: staticInput.shape
                )
            }
        }

        func modelInput(_ dynamic: MLXArray) -> MLXArray {
            guard let staticInput else { return dynamic }
            return MLX.concatenated([dynamic, staticInput], axis: -1)
        }

        var sample = noise
        let timesteps = configuration.timesteps
        for index in 0..<configuration.steps {
            model.beginDenoisingStep(index: index, count: configuration.steps)
            let timestep = timesteps[index]
            let previous = timesteps[index + 1]
            let strength = configuration.guidanceInterval.contains(timestep)
                ? configuration.guidanceStrength
                : 1
            let velocity: MLXArray
            if strength == 1 {
                velocity = try model.predict(
                    input: modelInput(sample),
                    coordinates: coordinates,
                    timestep: 1_000 * timestep,
                    condition: condition
                )
            } else {
                // The reference sampler evaluates positive and negative CFG
                // branches sequentially. Preserve that order to avoid doubling
                // the peak activation footprint on unified memory.
                let positive = try model.predict(
                    input: modelInput(sample),
                    coordinates: coordinates,
                    timestep: 1_000 * timestep,
                    condition: condition
                )
                MLX.eval(positive)
                Memory.clearCache()
                let negative = try model.predict(
                    input: modelInput(sample),
                    coordinates: coordinates,
                    timestep: 1_000 * timestep,
                    condition: negativeCondition
                )
                MLX.eval(negative)
                Memory.clearCache()
                var guided = strength * positive + (1 - strength) * negative
                if configuration.guidanceRescale > 0 {
                    let sigma = configuration.sigmaMinimum + (1 - configuration.sigmaMinimum) * timestep
                    let positiveStart = (1 - configuration.sigmaMinimum) * sample - sigma * positive
                    let guidedStart = (1 - configuration.sigmaMinimum) * sample - sigma * guided
                    let reductionAxes = Array(1..<positiveStart.ndim)
                    let positiveDeviation = MLX.std(
                        positiveStart,
                        axes: reductionAxes,
                        keepDims: true,
                        ddof: 1
                    )
                    let guidedDeviation = MLX.std(
                        guidedStart,
                        axes: reductionAxes,
                        keepDims: true,
                        ddof: 1
                    )
                    let rescaledStart = guidedStart * (
                        positiveDeviation / MLX.maximum(guidedDeviation, MLXArray(1e-6))
                    )
                    let blendedStart = configuration.guidanceRescale * rescaledStart
                        + (1 - configuration.guidanceRescale) * guidedStart
                    guided = ((1 - configuration.sigmaMinimum) * sample - blendedStart) / sigma
                }
                velocity = guided
            }
            sample = sample - (timestep - previous) * velocity
            MLX.eval(sample)
            Memory.clearCache()
            progress?(index + 1, configuration.steps)
        }
        return sample
    }
}
