import Foundation
@preconcurrency import MLX
import MLXFast
import MLXNN

public struct TESSERAArchitecture: Codable, Equatable, Sendable {
    public let representationDimension: Int
    public let latentDimension: Int
    public let layerCount: Int
    public let headCount: Int
    public let feedForwardDimension: Int
    public let maximumSequenceLength: Int
    public let qkNormalization: Bool?
    public let fusionLayerCount: Int?

    public var hiddenDimension: Int { latentDimension * 4 }

    public init(
        representationDimension: Int,
        latentDimension: Int,
        layerCount: Int,
        headCount: Int,
        feedForwardDimension: Int,
        maximumSequenceLength: Int,
        qkNormalization: Bool? = nil,
        fusionLayerCount: Int? = nil
    ) {
        self.representationDimension = representationDimension
        self.latentDimension = latentDimension
        self.layerCount = layerCount
        self.headCount = headCount
        self.feedForwardDimension = feedForwardDimension
        self.maximumSequenceLength = maximumSequenceLength
        self.qkNormalization = qkNormalization
        self.fusionLayerCount = fusionLayerCount
    }

    private enum CodingKeys: String, CodingKey {
        case representationDimension = "representation_dimension"
        case latentDimension = "latent_dimension"
        case layerCount = "layer_count"
        case headCount = "head_count"
        case feedForwardDimension = "feed_forward_dimension"
        case maximumSequenceLength = "maximum_sequence_length"
        case qkNormalization = "qk_normalization"
        case fusionLayerCount = "fusion_layer_count"
    }
}

/// Native MLX implementation of the deployable TESSERA v2 student encoder.
///
/// `s2` and `s1` follow the exact upstream neural boundary: standardized bands
/// followed by the raw day-of-year value in the final channel. The public CLI
/// accepts raw bands and performs the pinned normalization before calling this
/// model.
public final class TESSERAModel: @unchecked Sendable {
    public static let representationDimension = 128
    public static let supportedOutputDimensions: Set<Int> = [16, 32, 64, 128]

    private let weights: [String: MLXArray]
    public let architecture: TESSERAArchitecture
    private let dtype: DType

    public init(weights: [String: MLXArray], architecture: TESSERAArchitecture) throws {
        let required = Self.requiredTensorNames(architecture: architecture)
        let missing = required.filter { weights[$0] == nil }
        guard missing.isEmpty else {
            throw TESSERAModelError.missingTensors(missing)
        }
        guard [Self.representationDimension, 1_024].contains(architecture.representationDimension),
              architecture.hiddenDimension.isMultiple(of: architecture.headCount),
              (architecture.qkNormalization == true) == (architecture.fusionLayerCount != nil) else {
            throw TESSERAModelError.invalidArchitecture(architecture)
        }
        self.weights = weights
        self.architecture = architecture
        self.dtype = weights["s2_backbone.embedding.0.weight"]!.dtype
    }

    public func callAsFunction(
        s2: MLXArray,
        s1: MLXArray,
        outputDimensions: Int = representationDimension
    ) throws -> MLXArray {
        let supportedDimensions = architecture.qkNormalization == true
            ? [architecture.representationDimension]
            : Array(Self.supportedOutputDimensions)
        guard supportedDimensions.contains(outputDimensions) else {
            throw TESSERAModelError.unsupportedOutputDimensions(outputDimensions)
        }
        try Self.validateInput(s2, channels: 11, name: "S2")
        try Self.validateInput(s1, channels: 3, name: "S1")
        guard s2.dim(0) == s1.dim(0) else {
            throw TESSERAModelError.inconsistentBatchSize
        }
        guard s2.dim(1) <= architecture.maximumSequenceLength,
              s1.dim(1) <= architecture.maximumSequenceLength else {
            throw TESSERAModelError.sequenceTooLong(
                maximum: architecture.maximumSequenceLength,
                s2: s2.dim(1),
                s1: s1.dim(1)
            )
        }

        let s2Features = encodeBackbone(s2.asType(dtype), prefix: "s2_backbone")
        let s1Features = encodeBackbone(s1.asType(dtype), prefix: "s1_backbone")
        var hidden: MLXArray
        if let fusionLayerCount = architecture.fusionLayerCount {
            hidden = MLX.stacked([s2Features, s1Features], axis: 1)
                + weight("fusion_modality_embed")
            for layer in 0..<fusionLayerCount {
                hidden = qkNormalizedTransformerLayer(
                    hidden,
                    prefix: "fusion_transformer.layers.\(layer)",
                    feedForwardDimension: architecture.hiddenDimension * 2
                )
            }
            hidden = hidden.reshaped(hidden.dim(0), 2 * architecture.hiddenDimension)
        } else {
            hidden = MLX.concatenated([s2Features, s1Features], axis: -1)
        }
        hidden = linear(hidden, prefix: "dim_reducer.0")
        hidden = layerNorm(
            hidden,
            weight: weight("dim_reducer.1.weight"),
            bias: weight("dim_reducer.1.bias"),
            epsilon: 1e-5
        )
        hidden = MLXNN.relu(hidden)
        hidden = linear(hidden, prefix: "dim_reducer.4")
        hidden = layerNorm(hidden, epsilon: 1e-5)
        return hidden[0..., 0..<outputDimensions]
    }

    private func encodeBackbone(_ input: MLXArray, prefix: String) -> MLXArray {
        let bands = input[0..., 0..., 0..<(input.dim(2) - 1)]
        let dayOfYear = input[0..., 0..., input.dim(2) - 1]
        var hidden = MLXNN.relu(linear(bands, prefix: "\(prefix).embedding.0"))
        hidden = linear(hidden, prefix: "\(prefix).embedding.2")
            + temporalEncoding(dayOfYear, dimension: architecture.hiddenDimension, dtype: dtype)

        for layer in 0..<architecture.layerCount {
            let layerPrefix = "\(prefix).transformer_encoder.layers.\(layer)"
            hidden = if architecture.qkNormalization == true {
                qkNormalizedTransformerLayer(
                    hidden,
                    prefix: layerPrefix,
                    feedForwardDimension: architecture.feedForwardDimension
                )
            } else {
                transformerLayer(hidden, prefix: layerPrefix)
            }
        }
        let attentionWeights = MLX.softmax(
            linear(hidden, prefix: "\(prefix).attn_pool.query"),
            axis: 1
        )
        return (attentionWeights * hidden).sum(axis: 1)
    }

    /// PyTorch's default TransformerEncoderLayer is post-normalization.
    private func transformerLayer(_ input: MLXArray, prefix: String) -> MLXArray {
        let projection = linear(input, prefix: "\(prefix).self_attn", combinedProjection: true)
        let batch = input.dim(0)
        let sequence = input.dim(1)
        let hidden = architecture.hiddenDimension
        let headDimension = hidden / architecture.headCount
        let qkv = projection
            .reshaped(batch, sequence, 3, architecture.headCount, headDimension)
            .transposed(2, 0, 3, 1, 4)
        let attended = MLXFast.scaledDotProductAttention(
            queries: qkv[0],
            keys: qkv[1],
            values: qkv[2],
            scale: 1 / sqrt(Float(headDimension)),
            mask: .none
        )
        let attentionOutput = linear(
            attended.transposed(0, 2, 1, 3).reshaped(batch, sequence, hidden),
            prefix: "\(prefix).self_attn.out_proj"
        )
        var result = layerNorm(
            input + attentionOutput,
            weight: weight("\(prefix).norm1.weight"),
            bias: weight("\(prefix).norm1.bias"),
            epsilon: 1e-5
        )
        let feedForward = linear(
            MLXNN.relu(linear(result, prefix: "\(prefix).linear1")),
            prefix: "\(prefix).linear2"
        )
        result = layerNorm(
            result + feedForward,
            weight: weight("\(prefix).norm2.weight"),
            bias: weight("\(prefix).norm2.bias"),
            epsilon: 1e-5
        )
        return result
    }

    private func qkNormalizedTransformerLayer(
        _ input: MLXArray,
        prefix: String,
        feedForwardDimension: Int
    ) -> MLXArray {
        let normalizedAttention = layerNorm(
            input,
            weight: weight("\(prefix).norm1.weight"),
            bias: weight("\(prefix).norm1.bias"),
            epsilon: 1e-5
        )
        let batch = input.dim(0)
        let sequence = input.dim(1)
        let headDimension = architecture.hiddenDimension / architecture.headCount
        var queries = linear(normalizedAttention, prefix: "\(prefix).q_proj")
            .reshaped(batch, sequence, architecture.headCount, headDimension)
        var keys = linear(normalizedAttention, prefix: "\(prefix).k_proj")
            .reshaped(batch, sequence, architecture.headCount, headDimension)
        let values = linear(normalizedAttention, prefix: "\(prefix).v_proj")
            .reshaped(batch, sequence, architecture.headCount, headDimension)
        queries = rmsNorm(
            queries,
            weight: weight("\(prefix).q_norm.weight"),
            epsilon: Float.ulpOfOne
        )
        keys = rmsNorm(
            keys,
            weight: weight("\(prefix).k_norm.weight"),
            epsilon: Float.ulpOfOne
        )
        let attended = MLXFast.scaledDotProductAttention(
            queries: queries.transposed(0, 2, 1, 3),
            keys: keys.transposed(0, 2, 1, 3),
            values: values.transposed(0, 2, 1, 3),
            scale: 1 / sqrt(Float(headDimension)),
            mask: .none
        )
        let attention = linear(
            attended.transposed(0, 2, 1, 3)
                .reshaped(batch, sequence, architecture.hiddenDimension),
            prefix: "\(prefix).out_proj"
        )
        let residual = input + attention
        let normalizedFeedForward = layerNorm(
            residual,
            weight: weight("\(prefix).norm2.weight"),
            bias: weight("\(prefix).norm2.bias"),
            epsilon: 1e-5
        )
        precondition(weight("\(prefix).linear1.weight").dim(0) == feedForwardDimension)
        let feedForward = linear(
            MLXNN.relu(linear(normalizedFeedForward, prefix: "\(prefix).linear1")),
            prefix: "\(prefix).linear2"
        )
        return residual + feedForward
    }

    private func temporalEncoding(
        _ dayOfYear: MLXArray,
        dimension: Int,
        dtype: DType
    ) -> MLXArray {
        let indices = MLX.arange(0, dimension, step: 2, dtype: .float32)
        let divisors = MLX.exp(indices * (-log(Float(10_000)) / Float(dimension)))
        let angles = dayOfYear.asType(.float32).expandedDimensions(axis: -1) * divisors
        return MLX.stacked([MLX.sin(angles), MLX.cos(angles)], axis: -1)
            .reshaped(dayOfYear.dim(0), dayOfYear.dim(1), dimension)
            .asType(dtype)
    }

    private func linear(
        _ input: MLXArray,
        prefix: String,
        combinedProjection: Bool = false
    ) -> MLXArray {
        let weightName = combinedProjection
            ? "\(prefix).in_proj_weight"
            : "\(prefix).weight"
        let biasName = combinedProjection
            ? "\(prefix).in_proj_bias"
            : "\(prefix).bias"
        return MLX.matmul(input, weight(weightName).T) + weight(biasName)
    }

    private func layerNorm(
        _ input: MLXArray,
        weight: MLXArray? = nil,
        bias: MLXArray? = nil,
        epsilon: Float
    ) -> MLXArray {
        let promoted = input.asType(.float32)
        let mean = promoted.mean(axis: -1, keepDims: true)
        let centered = promoted - mean
        let variance = (centered * centered).mean(axis: -1, keepDims: true)
        var result = centered / MLX.sqrt(variance + epsilon)
        if let weight { result = result * weight.asType(.float32) }
        if let bias { result = result + bias.asType(.float32) }
        return result.asType(input.dtype)
    }

    private func rmsNorm(
        _ input: MLXArray,
        weight: MLXArray,
        epsilon: Float
    ) -> MLXArray {
        let promoted = input.asType(.float32)
        let scale = MLX.rsqrt((promoted * promoted).mean(axis: -1, keepDims: true) + epsilon)
        return (promoted * scale * weight.asType(.float32)).asType(input.dtype)
    }

    private func weight(_ name: String) -> MLXArray { weights[name]! }

    private static func validateInput(_ input: MLXArray, channels: Int, name: String) throws {
        guard input.ndim == 3, input.dim(0) > 0, input.dim(1) > 0, input.dim(2) == channels else {
            throw TESSERAModelError.invalidInputShape(
                name: name,
                expected: [-1, -1, channels],
                actual: input.shape
            )
        }
    }

    public static func requiredTensorNames(architecture: TESSERAArchitecture) -> [String] {
        var names: [String] = []
        for backbone in ["s2_backbone", "s1_backbone"] {
            names += [
                "\(backbone).embedding.0.weight", "\(backbone).embedding.0.bias",
                "\(backbone).embedding.2.weight", "\(backbone).embedding.2.bias",
            ]
            for layer in 0..<architecture.layerCount {
                let prefix = "\(backbone).transformer_encoder.layers.\(layer)"
                names += transformerTensorNames(
                    prefix: prefix,
                    qkNormalization: architecture.qkNormalization == true
                )
            }
            names += ["\(backbone).attn_pool.query.weight", "\(backbone).attn_pool.query.bias"]
        }
        names += [
            "dim_reducer.0.weight", "dim_reducer.0.bias",
            "dim_reducer.1.weight", "dim_reducer.1.bias",
            "dim_reducer.4.weight", "dim_reducer.4.bias",
        ]
        if let fusionLayerCount = architecture.fusionLayerCount {
            names.append("fusion_modality_embed")
            for layer in 0..<fusionLayerCount {
                names += transformerTensorNames(
                    prefix: "fusion_transformer.layers.\(layer)",
                    qkNormalization: true
                )
            }
        }
        return names
    }

    private static func transformerTensorNames(
        prefix: String,
        qkNormalization: Bool
    ) -> [String] {
        var names: [String]
        if qkNormalization {
            names = [
                "\(prefix).q_proj.weight", "\(prefix).q_proj.bias",
                "\(prefix).k_proj.weight", "\(prefix).k_proj.bias",
                "\(prefix).v_proj.weight", "\(prefix).v_proj.bias",
                "\(prefix).out_proj.weight", "\(prefix).out_proj.bias",
                "\(prefix).q_norm.weight", "\(prefix).k_norm.weight",
            ]
        } else {
            names = [
                "\(prefix).self_attn.in_proj_weight", "\(prefix).self_attn.in_proj_bias",
                "\(prefix).self_attn.out_proj.weight", "\(prefix).self_attn.out_proj.bias",
            ]
        }
        names += [
            "\(prefix).linear1.weight", "\(prefix).linear1.bias",
            "\(prefix).linear2.weight", "\(prefix).linear2.bias",
            "\(prefix).norm1.weight", "\(prefix).norm1.bias",
            "\(prefix).norm2.weight", "\(prefix).norm2.bias",
        ]
        return names
    }
}

public enum TESSERAPreprocessor {
    public static let sentinel2BandOrder = [
        "B04", "B02", "B03", "B08", "B8A", "B05", "B06", "B07", "B11", "B12",
    ]
    public static let sentinel1BandOrder = ["VV", "VH"]

    private static let sentinel2Mean: [Float] = [
        1633.0042, 1341.1090, 1539.5536, 3054.8269, 3117.4658,
        2004.1648, 2694.7275, 2945.1504, 2266.6079, 1657.3094,
    ]
    private static let sentinel2StandardDeviation: [Float] = [
        1999.4603, 2014.7549, 1929.2201, 1754.2493, 1649.9807,
        1936.8988, 1748.6041, 1708.6991, 1207.5250, 1108.6046,
    ]
    private static let sentinel1AscendingMean: [Float] = [5909.3921, 3405.0322]
    private static let sentinel1AscendingStandardDeviation: [Float] = [1507.1750, 1531.2615]
    private static let sentinel1DescendingMean: [Float] = [5816.1382, 3277.7576]
    private static let sentinel1DescendingStandardDeviation: [Float] = [1554.6475, 1546.4733]
    private static let sentinel1TeacherMean: [Float] = [5862.7652, 3341.3949]
    private static let sentinel1TeacherStandardDeviation: [Float] = [1531.8051, 1540.2014]

    public static func prepareSentinel2(bands: MLXArray, dayOfYear: MLXArray) throws -> MLXArray {
        try validate(bands: bands, dayOfYear: dayOfYear, channels: 10, name: "S2")
        let normalized = normalize(
            bands,
            mean: sentinel2Mean,
            standardDeviation: sentinel2StandardDeviation
        )
        return MLX.concatenated(
            [normalized, dayOfYear.asType(.float32).expandedDimensions(axis: -1)],
            axis: -1
        )
    }

    public static func prepareSentinel1(
        ascendingBands: MLXArray?,
        ascendingDayOfYear: MLXArray?,
        descendingBands: MLXArray?,
        descendingDayOfYear: MLXArray?,
        variant: TESSERAVariant = .medium
    ) throws -> MLXArray {
        if variant == .teacher {
            return try prepareTeacherSentinel1(
                ascendingBands: ascendingBands,
                ascendingDayOfYear: ascendingDayOfYear,
                descendingBands: descendingBands,
                descendingDayOfYear: descendingDayOfYear
            )
        }
        var parts: [MLXArray] = []
        if let ascendingBands, let ascendingDayOfYear {
            try validate(bands: ascendingBands, dayOfYear: ascendingDayOfYear, channels: 2, name: "S1_ASC")
            parts.append(MLX.concatenated([
                normalize(
                    ascendingBands,
                    mean: sentinel1AscendingMean,
                    standardDeviation: sentinel1AscendingStandardDeviation
                ),
                ascendingDayOfYear.asType(.float32).expandedDimensions(axis: -1),
            ], axis: -1))
        } else if ascendingBands != nil || ascendingDayOfYear != nil {
            throw TESSERAModelError.incompleteInputPair("S1_ASC", "S1_ASC_DOY")
        }
        if let descendingBands, let descendingDayOfYear {
            try validate(bands: descendingBands, dayOfYear: descendingDayOfYear, channels: 2, name: "S1_DESC")
            parts.append(MLX.concatenated([
                normalize(
                    descendingBands,
                    mean: sentinel1DescendingMean,
                    standardDeviation: sentinel1DescendingStandardDeviation
                ),
                descendingDayOfYear.asType(.float32).expandedDimensions(axis: -1),
            ], axis: -1))
        } else if descendingBands != nil || descendingDayOfYear != nil {
            throw TESSERAModelError.incompleteInputPair("S1_DESC", "S1_DESC_DOY")
        }
        guard let first = parts.first else {
            throw TESSERAModelError.missingSentinel1
        }
        guard parts.allSatisfy({ $0.dim(0) == first.dim(0) }) else {
            throw TESSERAModelError.inconsistentBatchSize
        }
        return parts.count == 1 ? first : MLX.concatenated(parts, axis: 1)
    }

    private static func prepareTeacherSentinel1(
        ascendingBands: MLXArray?,
        ascendingDayOfYear: MLXArray?,
        descendingBands: MLXArray?,
        descendingDayOfYear: MLXArray?
    ) throws -> MLXArray {
        var bands: [MLXArray] = []
        var days: [MLXArray] = []
        for (name, values, dayOfYear) in [
            ("S1_ASC", ascendingBands, ascendingDayOfYear),
            ("S1_DESC", descendingBands, descendingDayOfYear),
        ] {
            if let values, let dayOfYear {
                try validate(bands: values, dayOfYear: dayOfYear, channels: 2, name: name)
                bands.append(values)
                days.append(dayOfYear)
            } else if values != nil || dayOfYear != nil {
                throw TESSERAModelError.incompleteInputPair(name, "\(name)_DOY")
            }
        }
        guard let first = bands.first else { throw TESSERAModelError.missingSentinel1 }
        guard bands.allSatisfy({ $0.dim(0) == first.dim(0) }) else {
            throw TESSERAModelError.inconsistentBatchSize
        }
        let mergedBands = bands.count == 1 ? first : MLX.concatenated(bands, axis: 1)
        let mergedDays = days.count == 1 ? days[0] : MLX.concatenated(days, axis: 1)
        return MLX.concatenated([
            normalize(
                mergedBands,
                mean: sentinel1TeacherMean,
                standardDeviation: sentinel1TeacherStandardDeviation
            ),
            mergedDays.asType(.float32).expandedDimensions(axis: -1),
        ], axis: -1)
    }

    private static func normalize(
        _ bands: MLXArray,
        mean: [Float],
        standardDeviation: [Float]
    ) -> MLXArray {
        let meanArray = MLXArray(mean).asType(.float32)
        let standardDeviationArray = MLXArray(standardDeviation).asType(.float32)
        return (bands.asType(.float32) - meanArray) / standardDeviationArray
    }

    private static func validate(
        bands: MLXArray,
        dayOfYear: MLXArray,
        channels: Int,
        name: String
    ) throws {
        guard bands.ndim == 3, bands.dim(0) > 0, bands.dim(1) > 0, bands.dim(2) == channels else {
            throw TESSERAModelError.invalidInputShape(
                name: name,
                expected: [-1, -1, channels],
                actual: bands.shape
            )
        }
        guard dayOfYear.ndim == 2,
              dayOfYear.dim(0) == bands.dim(0),
              dayOfYear.dim(1) == bands.dim(1) else {
            throw TESSERAModelError.invalidInputShape(
                name: "\(name)_DOY",
                expected: [bands.dim(0), bands.dim(1)],
                actual: dayOfYear.shape
            )
        }
    }
}

public enum TESSERAModelError: Error, Equatable, LocalizedError, Sendable {
    case missingTensors([String])
    case invalidArchitecture(TESSERAArchitecture)
    case invalidInputShape(name: String, expected: [Int], actual: [Int])
    case incompleteInputPair(String, String)
    case missingSentinel1
    case inconsistentBatchSize
    case sequenceTooLong(maximum: Int, s2: Int, s1: Int)
    case unsupportedOutputDimensions(Int)

    public var errorDescription: String? {
        switch self {
        case .missingTensors(let names):
            "TESSERA checkpoint is missing tensors: \(names.joined(separator: ", "))."
        case .invalidArchitecture(let architecture):
            "Unsupported TESSERA architecture: \(architecture)."
        case .invalidInputShape(let name, let expected, let actual):
            "TESSERA \(name) input must have shape \(expected); found \(actual)."
        case .incompleteInputPair(let bands, let dates):
            "TESSERA input must provide \(bands) and \(dates) together."
        case .missingSentinel1:
            "TESSERA requires at least one Sentinel-1 ascending or descending sequence."
        case .inconsistentBatchSize:
            "TESSERA Sentinel-1 and Sentinel-2 inputs must use the same batch size."
        case .sequenceTooLong(let maximum, let s2, let s1):
            "TESSERA sequences may contain at most \(maximum) observations; found S2=\(s2), S1=\(s1)."
        case .unsupportedOutputDimensions(let dimensions):
            "TESSERA Matryoshka output dimensions must be one of 16, 32, 64, or 128; found \(dimensions)."
        }
    }
}
