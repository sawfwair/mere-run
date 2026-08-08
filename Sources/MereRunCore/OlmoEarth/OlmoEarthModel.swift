import Foundation
@preconcurrency import MLX
import MLXFast
import MLXNN

public enum OlmoEarthModality: String, Codable, CaseIterable, Hashable, Sendable {
    case sentinel2L2A = "sentinel2_l2a"
    case sentinel1 = "sentinel1"
    case landsat = "landsat"

    public var inputTensorName: String {
        switch self {
        case .sentinel2L2A: "S2L2A"
        case .sentinel1: "S1RTC"
        case .landsat: "LANDSAT"
        }
    }

    public var outputTensorName: String {
        switch self {
        case .sentinel2L2A: "S2L2A_EMBEDDINGS"
        case .sentinel1: "S1RTC_EMBEDDINGS"
        case .landsat: "LANDSAT_EMBEDDINGS"
        }
    }

    public var channelCount: Int {
        switch self {
        case .sentinel2L2A: 12
        case .sentinel1: 2
        case .landsat: 11
        }
    }

    public var bandOrder: [String] {
        switch self {
        case .sentinel2L2A:
            ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09"]
        case .sentinel1:
            ["vv", "vh"]
        case .landsat:
            ["B8", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B9", "B10", "B11"]
        }
    }
}

public struct OlmoEarthArchitecture: Codable, Equatable, Sendable {
    public let embeddingDimension: Int
    public let headCount: Int
    public let depth: Int
    public let mlpRatio: Float
    public let maximumSequenceLength: Int
    public let maximumPatchSize: Int
    public let patchHiddenDimension: Int
    public let positionEncoding: String
    public let temporalCoordinateScale: Float

    public var headDimension: Int { embeddingDimension / headCount }
    public var feedForwardDimension: Int { Int(Float(embeddingDimension) * mlpRatio) }

    private enum CodingKeys: String, CodingKey {
        case embeddingDimension = "embedding_dimension"
        case headCount = "head_count"
        case depth
        case mlpRatio = "mlp_ratio"
        case maximumSequenceLength = "maximum_sequence_length"
        case maximumPatchSize = "maximum_patch_size"
        case patchHiddenDimension = "patch_hidden_dimension"
        case positionEncoding = "position_encoding"
        case temporalCoordinateScale = "temporal_coordinate_scale"
    }
}

public struct OlmoEarthEmbeddingResult {
    public let tokens: [OlmoEarthModality: MLXArray]
    public let pooled: [OlmoEarthModality: MLXArray]
    public let patchRows: Int
    public let patchColumns: Int
}

/// Native inference-only OlmoEarth v1.2 encoder for the three primary imagery
/// modalities: Sentinel-2 L2A, Sentinel-1 RTC, and Landsat.
public final class OlmoEarthModel: @unchecked Sendable {
    private static let baseGroundSampleDistance: Float = 10

    private let weights: [String: MLXArray]
    public let architecture: OlmoEarthArchitecture
    private let dtype: DType

    public init(weights: [String: MLXArray], architecture: OlmoEarthArchitecture) throws {
        let required = Self.requiredTensorNames(architecture: architecture)
        let missing = required.filter { weights[$0] == nil }
        guard missing.isEmpty else {
            throw OlmoEarthModelError.missingTensors(missing)
        }
        guard architecture.embeddingDimension.isMultiple(of: architecture.headCount),
              architecture.headDimension.isMultiple(of: 4),
              architecture.positionEncoding == "rope_3d_mixed" else {
            throw OlmoEarthModelError.invalidArchitecture(architecture)
        }
        self.weights = weights
        self.architecture = architecture
        self.dtype = weights["blocks.0.attn.q.weight"]!.dtype
    }

    public func callAsFunction(
        modalities: [OlmoEarthModality: MLXArray],
        timestamps: MLXArray,
        patchSize: Int = 4,
        inputResolutionMeters: Float = 10
    ) throws -> OlmoEarthEmbeddingResult {
        guard [1, 2, 4, 8].contains(patchSize), patchSize <= architecture.maximumPatchSize else {
            throw OlmoEarthModelError.unsupportedPatchSize(patchSize)
        }
        guard inputResolutionMeters > 0 else {
            throw OlmoEarthModelError.invalidInputResolution(inputResolutionMeters)
        }
        guard !modalities.isEmpty else {
            throw OlmoEarthModelError.missingModalities
        }
        guard timestamps.ndim == 3, timestamps.dim(0) > 0,
              timestamps.dim(1) > 0, timestamps.dim(2) == 3 else {
            throw OlmoEarthModelError.invalidInputShape(
                name: "TIMESTAMPS",
                expected: [-1, -1, 3],
                actual: timestamps.shape
            )
        }
        guard timestamps.dim(1) <= architecture.maximumSequenceLength else {
            throw OlmoEarthModelError.sequenceTooLong(
                maximum: architecture.maximumSequenceLength,
                actual: timestamps.dim(1)
            )
        }
        try validateTimestamps(timestamps)

        let orderedModalities = modalities.keys.sorted { $0.rawValue < $1.rawValue }
        var encoded: [OlmoEarthModality: MLXArray] = [:]
        var commonRows: Int?
        var commonColumns: Int?
        for modality in orderedModalities {
            let input = modalities[modality]!
            try Self.validateInput(
                input,
                modality: modality,
                timestamps: timestamps,
                patchSize: patchSize
            )
            let tokens = patchEmbed(input.asType(dtype), modality: modality, patchSize: patchSize)
            if let commonRows, let commonColumns {
                guard tokens.dim(1) == commonRows, tokens.dim(2) == commonColumns else {
                    throw OlmoEarthModelError.inconsistentSpatialDimensions
                }
            } else {
                commonRows = tokens.dim(1)
                commonColumns = tokens.dim(2)
            }
            encoded[modality] = addCompositeEncodings(
                tokens,
                modality: modality,
                timestamps: timestamps
            )
        }

        let patchRows = commonRows!
        let patchColumns = commonColumns!
        let positionsPerModality = try buildPositions(
            timestamps: timestamps,
            rows: patchRows,
            columns: patchColumns,
            patchSize: patchSize,
            inputResolutionMeters: inputResolutionMeters
        )
        let positions = MLX.concatenated(
            orderedModalities.map { _ in positionsPerModality },
            axis: 1
        )

        let flattened = orderedModalities.map { modality in
            encoded[modality]!.reshaped(
                timestamps.dim(0),
                patchRows * patchColumns * timestamps.dim(1),
                architecture.embeddingDimension
            )
        }
        var hidden = MLX.concatenated(flattened, axis: 1)
        for block in 0..<architecture.depth {
            hidden = transformerBlock(hidden, positions: positions, index: block)
            MLX.eval(hidden)
        }
        hidden = layerNorm(
            hidden,
            weight: weight("norm.weight"),
            bias: weight("norm.bias"),
            epsilon: 1e-5
        )

        let tokensPerModality = patchRows * patchColumns * timestamps.dim(1)
        var outputs: [OlmoEarthModality: MLXArray] = [:]
        var pooled: [OlmoEarthModality: MLXArray] = [:]
        for (index, modality) in orderedModalities.enumerated() {
            let start = index * tokensPerModality
            let end = start + tokensPerModality
            let value = hidden[0..., start..<end, 0...].reshaped(
                timestamps.dim(0),
                patchRows,
                patchColumns,
                timestamps.dim(1),
                architecture.embeddingDimension
            )
            outputs[modality] = value
            pooled[modality] = value.mean(axis: 3)
        }
        return OlmoEarthEmbeddingResult(
            tokens: outputs,
            pooled: pooled,
            patchRows: patchRows,
            patchColumns: patchColumns
        )
    }

    private func patchEmbed(
        _ input: MLXArray,
        modality: OlmoEarthModality,
        patchSize: Int
    ) -> MLXArray {
        let resized = OlmoEarthBicubicResize.resizeForBasePatch(
            input,
            requestedPatchSize: patchSize,
            basePatchSize: architecture.maximumPatchSize
        )
        let prefix = "patch_embeddings.per_modality_embeddings.\(modality.rawValue).\(modality.rawValue)__0"
        var pixels = MLXNN.relu(linear(resized, prefix: "\(prefix).pixel_proj.0"))
        let batch = pixels.dim(0)
        let rows = pixels.dim(1) / architecture.maximumPatchSize
        let columns = pixels.dim(2) / architecture.maximumPatchSize
        let timestamps = pixels.dim(3)
        pixels = pixels
            .reshaped(
                batch,
                rows,
                architecture.maximumPatchSize,
                columns,
                architecture.maximumPatchSize,
                timestamps,
                architecture.patchHiddenDimension
            )
            .transposed(0, 1, 3, 5, 2, 4, 6)
            .reshaped(
                batch,
                rows,
                columns,
                timestamps,
                architecture.maximumPatchSize * architecture.maximumPatchSize
                    * architecture.patchHiddenDimension
            )
        return linear(pixels, prefix: "\(prefix).proj")
    }

    private func addCompositeEncodings(
        _ input: MLXArray,
        modality: OlmoEarthModality,
        timestamps: MLXArray
    ) -> MLXArray {
        let batch = input.dim(0)
        let rows = input.dim(1)
        let columns = input.dim(2)
        let count = input.dim(3)
        let quarter = architecture.embeddingDimension / 4
        let targetShape = [batch, rows, columns, count, quarter]
        let channel = MLX.broadcast(
            weight("composite_encodings.per_modality_channel_embeddings.\(modality.rawValue)")
                .reshaped(1, 1, 1, 1, quarter),
            to: targetShape
        )
        let month = MLX.broadcast(
            monthEmbeddings(timestamps).reshaped(batch, 1, 1, count, quarter),
            to: targetShape
        )
        let zeros = MLX.zeros(targetShape, dtype: input.dtype)
        return input + MLX.concatenated([channel, zeros, month, zeros], axis: -1)
    }

    private func monthEmbeddings(_ timestamps: MLXArray) -> MLXArray {
        let timestampValues = timestamps.asType(.int32).asArray(Int32.self)
        let table = weight("composite_encodings.month_embed.weight")
            .asType(.float32)
            .asArray(Float.self)
        let dimension = architecture.embeddingDimension / 4
        var values: [Float] = []
        values.reserveCapacity(timestamps.dim(0) * timestamps.dim(1) * dimension)
        for index in stride(from: 1, to: timestampValues.count, by: 3) {
            let month = Int(timestampValues[index])
            let start = month * dimension
            values.append(contentsOf: table[start..<(start + dimension)])
        }
        return MLXArray(values, [timestamps.dim(0), timestamps.dim(1), dimension]).asType(dtype)
    }

    private func validateTimestamps(_ timestamps: MLXArray) throws {
        let values = timestamps.asType(.int32).asArray(Int32.self)
        for index in stride(from: 0, to: values.count, by: 3) {
            let day = Int(values[index])
            let month = Int(values[index + 1])
            let year = Int(values[index + 2])
            guard (1...31).contains(day), (0...11).contains(month) else {
                throw OlmoEarthModelError.invalidTimestamp(day: day, month: month, year: year)
            }
        }
    }

    private func buildPositions(
        timestamps: MLXArray,
        rows: Int,
        columns: Int,
        patchSize: Int,
        inputResolutionMeters: Float
    ) throws -> MLXArray {
        let dateValues = timestamps.asType(.int32).asArray(Int32.self)
        let daysBeforeMonth: [Float] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
        var temporal: [Float] = []
        temporal.reserveCapacity(timestamps.dim(0) * timestamps.dim(1))
        for index in stride(from: 0, to: dateValues.count, by: 3) {
            let day = Int(dateValues[index])
            let month = Int(dateValues[index + 1])
            let year = Int(dateValues[index + 2])
            guard (1...31).contains(day), (0...11).contains(month) else {
                throw OlmoEarthModelError.invalidTimestamp(day: day, month: month, year: year)
            }
            let days = Float(year - 2_000) * 365.25 + daysBeforeMonth[month] + Float(day - 1)
            temporal.append(days * architecture.temporalCoordinateScale)
        }

        let batch = timestamps.dim(0)
        let count = timestamps.dim(1)
        let spatialScale = inputResolutionMeters * Float(patchSize) / Self.baseGroundSampleDistance
        var positions: [Float] = []
        positions.reserveCapacity(batch * rows * columns * count * 3)
        for batchIndex in 0..<batch {
            for row in 0..<rows {
                for column in 0..<columns {
                    for time in 0..<count {
                        positions.append(temporal[batchIndex * count + time])
                        positions.append(Float(row) * spatialScale)
                        positions.append(Float(column) * spatialScale)
                    }
                }
            }
        }
        return MLXArray(positions, [batch, rows * columns * count, 3]).asType(dtype)
    }

    private func transformerBlock(
        _ input: MLXArray,
        positions: MLXArray,
        index: Int
    ) -> MLXArray {
        let prefix = "blocks.\(index)"
        let normalizedAttention = layerNorm(
            input,
            weight: weight("\(prefix).norm1.weight"),
            bias: weight("\(prefix).norm1.bias"),
            epsilon: 1e-5
        )
        let batch = input.dim(0)
        let count = input.dim(1)
        let heads = architecture.headCount
        let headDimension = architecture.headDimension
        var queries = linear(normalizedAttention, prefix: "\(prefix).attn.q")
            .reshaped(batch, count, heads, headDimension)
            .transposed(0, 2, 1, 3)
        var keys = linear(normalizedAttention, prefix: "\(prefix).attn.k")
            .reshaped(batch, count, heads, headDimension)
            .transposed(0, 2, 1, 3)
        let values = linear(normalizedAttention, prefix: "\(prefix).attn.v")
            .reshaped(batch, count, heads, headDimension)
            .transposed(0, 2, 1, 3)
        let frequencies = weight("\(prefix).attn.rope_mixed_freqs")
        queries = applyMixed3DRoPE(queries, positions: positions, frequencies: frequencies)
        keys = applyMixed3DRoPE(keys, positions: positions, frequencies: frequencies)
        let attended = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: 1 / sqrt(Float(headDimension)),
            mask: .none
        )
        let projected = linear(
            attended.transposed(0, 2, 1, 3).reshaped(
                batch,
                count,
                architecture.embeddingDimension
            ),
            prefix: "\(prefix).attn.proj"
        )
        let residual = input + projected
        let normalizedMLP = layerNorm(
            residual,
            weight: weight("\(prefix).norm2.weight"),
            bias: weight("\(prefix).norm2.bias"),
            epsilon: 1e-5
        )
        let feedForward = linear(
            MLXNN.gelu(linear(normalizedMLP, prefix: "\(prefix).mlp.fc1")),
            prefix: "\(prefix).mlp.fc2"
        )
        return residual + feedForward
    }

    private func applyMixed3DRoPE(
        _ input: MLXArray,
        positions: MLXArray,
        frequencies: MLXArray
    ) -> MLXArray {
        let temporal = positions[0..., 0..., 0].reshaped(positions.dim(0), 1, positions.dim(1), 1)
        let row = positions[0..., 0..., 1].reshaped(positions.dim(0), 1, positions.dim(1), 1)
        let column = positions[0..., 0..., 2].reshaped(positions.dim(0), 1, positions.dim(1), 1)
        let temporalFrequencies = frequencies[0].expandedDimensions(axes: [0, 2])
        let rowFrequencies = frequencies[1].expandedDimensions(axes: [0, 2])
        let columnFrequencies = frequencies[2].expandedDimensions(axes: [0, 2])
        let angles = temporal * temporalFrequencies
            + row * rowFrequencies
            + column * columnFrequencies
        let cosine = MLX.repeated(MLX.cos(angles).expandedDimensions(axis: -1), count: 2, axis: -1)
            .reshaped(input.shape)
            .asType(input.dtype)
        let sine = MLX.repeated(MLX.sin(angles).expandedDimensions(axis: -1), count: 2, axis: -1)
            .reshaped(input.shape)
            .asType(input.dtype)
        let pairs = input.reshaped(
            input.dim(0),
            input.dim(1),
            input.dim(2),
            input.dim(3) / 2,
            2
        )
        let rotated = MLX.stacked([-pairs[0..., 0..., 0..., 0..., 1], pairs[0..., 0..., 0..., 0..., 0]], axis: -1)
            .reshaped(input.shape)
        return input * cosine + rotated * sine
    }

    private func linear(_ input: MLXArray, prefix: String) -> MLXArray {
        MLX.matmul(input, weight("\(prefix).weight").T) + weight("\(prefix).bias")
    }

    private func layerNorm(
        _ input: MLXArray,
        weight: MLXArray,
        bias: MLXArray,
        epsilon: Float
    ) -> MLXArray {
        let promoted = input.asType(.float32)
        let mean = promoted.mean(axis: -1, keepDims: true)
        let centered = promoted - mean
        let variance = (centered * centered).mean(axis: -1, keepDims: true)
        return ((centered / MLX.sqrt(variance + epsilon))
            * weight.asType(.float32) + bias.asType(.float32)).asType(input.dtype)
    }

    private func weight(_ name: String) -> MLXArray { weights[name]! }

    private static func validateInput(
        _ input: MLXArray,
        modality: OlmoEarthModality,
        timestamps: MLXArray,
        patchSize: Int
    ) throws {
        guard input.ndim == 5,
              input.dim(0) == timestamps.dim(0),
              input.dim(1) > 0,
              input.dim(2) > 0,
              input.dim(3) == timestamps.dim(1),
              input.dim(4) == modality.channelCount else {
            throw OlmoEarthModelError.invalidInputShape(
                name: modality.inputTensorName,
                expected: [timestamps.dim(0), -1, -1, timestamps.dim(1), modality.channelCount],
                actual: input.shape
            )
        }
        guard input.dim(1).isMultiple(of: patchSize), input.dim(2).isMultiple(of: patchSize) else {
            throw OlmoEarthModelError.spatialDimensionsNotDivisible(
                height: input.dim(1),
                width: input.dim(2),
                patchSize: patchSize
            )
        }
    }

    public static func requiredTensorNames(architecture: OlmoEarthArchitecture) -> [String] {
        var names: [String] = []
        for block in 0..<architecture.depth {
            let prefix = "blocks.\(block)"
            names += [
                "\(prefix).norm1.weight", "\(prefix).norm1.bias",
                "\(prefix).attn.rope_mixed_freqs",
                "\(prefix).attn.q.weight", "\(prefix).attn.q.bias",
                "\(prefix).attn.k.weight", "\(prefix).attn.k.bias",
                "\(prefix).attn.v.weight", "\(prefix).attn.v.bias",
                "\(prefix).attn.proj.weight", "\(prefix).attn.proj.bias",
                "\(prefix).norm2.weight", "\(prefix).norm2.bias",
                "\(prefix).mlp.fc1.weight", "\(prefix).mlp.fc1.bias",
                "\(prefix).mlp.fc2.weight", "\(prefix).mlp.fc2.bias",
            ]
        }
        names.append("composite_encodings.month_embed.weight")
        for modality in OlmoEarthModality.allCases {
            names.append("composite_encodings.per_modality_channel_embeddings.\(modality.rawValue)")
            let prefix = "patch_embeddings.per_modality_embeddings.\(modality.rawValue).\(modality.rawValue)__0"
            names += [
                "\(prefix).pixel_proj.0.weight", "\(prefix).pixel_proj.0.bias",
                "\(prefix).proj.weight", "\(prefix).proj.bias",
            ]
        }
        names += ["norm.weight", "norm.bias"]
        return names
    }
}

public enum OlmoEarthPreprocessor {
    public static func normalize(_ input: MLXArray, modality: OlmoEarthModality) -> MLXArray {
        let statistics = statistics(for: modality)
        let mean = MLXArray(statistics.mean).asType(.float32)
        let standardDeviation = MLXArray(statistics.standardDeviation).asType(.float32)
        return (input.asType(.float32) - (mean - 2 * standardDeviation)) / (4 * standardDeviation)
    }

    private static func statistics(
        for modality: OlmoEarthModality
    ) -> (mean: [Float], standardDeviation: [Float]) {
        switch modality {
        case .sentinel2L2A:
            (
                [
                    1188.9413, 1407.7739, 1513.0574, 2755.4812, 1890.9893, 2483.7812,
                    2722.7283, 2885.5703, 2562.8525, 1914.1383, 1115.8495, 3269.8125,
                ],
                [
                    1859.1924, 1727.7388, 1740.7758, 1612.2566, 1754.7321, 1622.1173,
                    1621.8226, 1611.3588, 1441.5472, 1328.8915, 1955.6992, 2651.0884,
                ]
            )
        case .sentinel1:
            ([-11.648991, -17.745436], [10.84035, 10.216274])
        case .landsat:
            (
                [
                    10132.091, 11000.96, 10493.416, 10146.318, 10236.446, 14427.021,
                    12164.081, 9712.033, 4585.4697, 21347.893, 19686.232,
                ],
                [
                    7788.728, 7857.9204, 7872.392, 7676.2773, 8038.2183, 9302.432,
                    7442.3906, 6037.7124, 2549.7454, 10957.281, 9911.421,
                ]
            )
        }
    }
}

private enum OlmoEarthBicubicResize {
    static func resizeForBasePatch(
        _ input: MLXArray,
        requestedPatchSize: Int,
        basePatchSize: Int
    ) -> MLXArray {
        guard requestedPatchSize != basePatchSize else { return input }
        let targetHeight = input.dim(1) / requestedPatchSize * basePatchSize
        let targetWidth = input.dim(2) / requestedPatchSize * basePatchSize
        let sourceHeight = input.dim(1)
        let sourceWidth = input.dim(2)
        let trailing = input.dim(3) * input.dim(4)
        let batch = input.dim(0)
        let source = input.asType(.float32).asArray(Float.self)

        let horizontal = bicubicResamplingWeights(sourceSize: sourceWidth, targetSize: targetWidth)
        var intermediate = [Float](
            repeating: 0,
            count: batch * sourceHeight * targetWidth * trailing
        )
        for batchIndex in 0..<batch {
            for y in 0..<sourceHeight {
                for targetX in 0..<targetWidth {
                    let targetOffset = (((batchIndex * sourceHeight + y) * targetWidth) + targetX) * trailing
                    for (sourceX, weight) in horizontal[targetX] {
                        let sourceOffset = (((batchIndex * sourceHeight + y) * sourceWidth) + sourceX) * trailing
                        for channel in 0..<trailing {
                            intermediate[targetOffset + channel] += source[sourceOffset + channel] * weight
                        }
                    }
                }
            }
        }

        let vertical = bicubicResamplingWeights(sourceSize: sourceHeight, targetSize: targetHeight)
        var output = [Float](
            repeating: 0,
            count: batch * targetHeight * targetWidth * trailing
        )
        for batchIndex in 0..<batch {
            for targetY in 0..<targetHeight {
                for (sourceY, weight) in vertical[targetY] {
                    for x in 0..<targetWidth {
                        let sourceOffset = (((batchIndex * sourceHeight + sourceY) * targetWidth) + x) * trailing
                        let targetOffset = (((batchIndex * targetHeight + targetY) * targetWidth) + x) * trailing
                        for channel in 0..<trailing {
                            output[targetOffset + channel] += intermediate[sourceOffset + channel] * weight
                        }
                    }
                }
            }
        }
        return MLXArray(
            output,
            [batch, targetHeight, targetWidth, input.dim(3), input.dim(4)]
        ).asType(input.dtype)
    }

    private static func bicubicResamplingWeights(
        sourceSize: Int,
        targetSize: Int
    ) -> [[(index: Int, weight: Float)]] {
        let scale = Float(sourceSize) / Float(targetSize)
        let filterScale = max(scale, 1)
        let support = 2 * filterScale
        var result: [[(index: Int, weight: Float)]] = []
        result.reserveCapacity(targetSize)
        for destination in 0..<targetSize {
            let center = scale * (Float(destination) + 0.5)
            let first = max(0, Int(ceil(center - support - 0.5)))
            let last = min(sourceSize - 1, Int(floor(center + support - 0.5)))
            var weights: [(Int, Float)] = []
            var total: Float = 0
            if first <= last {
                for index in first...last {
                    let distance = (Float(index) + 0.5 - center) / filterScale
                    let weight = cubicKernel(distance)
                    if weight != 0 {
                        weights.append((index, weight))
                        total += weight
                    }
                }
            }
            if total != 0 {
                weights = weights.map { ($0.0, $0.1 / total) }
            }
            result.append(weights)
        }
        return result
    }

    private static func cubicKernel(_ value: Float) -> Float {
        let distance = abs(value)
        let coefficient: Float = -0.5
        if distance < 1 {
            return ((coefficient + 2) * distance - (coefficient + 3)) * distance * distance + 1
        }
        if distance < 2 {
            return (((distance - 5) * distance + 8) * distance - 4) * coefficient
        }
        return 0
    }
}

public enum OlmoEarthModelError: Error, Equatable, LocalizedError, Sendable {
    case missingTensors([String])
    case invalidArchitecture(OlmoEarthArchitecture)
    case missingModalities
    case invalidInputShape(name: String, expected: [Int], actual: [Int])
    case inconsistentSpatialDimensions
    case spatialDimensionsNotDivisible(height: Int, width: Int, patchSize: Int)
    case unsupportedPatchSize(Int)
    case invalidInputResolution(Float)
    case sequenceTooLong(maximum: Int, actual: Int)
    case invalidTimestamp(day: Int, month: Int, year: Int)

    public var errorDescription: String? {
        switch self {
        case .missingTensors(let names):
            "OlmoEarth checkpoint is missing tensors: \(names.joined(separator: ", "))."
        case .invalidArchitecture(let architecture):
            "Unsupported OlmoEarth v1.2 architecture: \(architecture)."
        case .missingModalities:
            "OlmoEarth requires at least one of S2L2A, S1RTC, or LANDSAT."
        case .invalidInputShape(let name, let expected, let actual):
            "OlmoEarth \(name) input must have shape \(expected); found \(actual)."
        case .inconsistentSpatialDimensions:
            "OlmoEarth modalities must resolve to the same patch grid."
        case .spatialDimensionsNotDivisible(let height, let width, let patchSize):
            "OlmoEarth input \(height)x\(width) is not divisible by patch size \(patchSize)."
        case .unsupportedPatchSize(let patchSize):
            "OlmoEarth patch size must be 1, 2, 4, or 8; found \(patchSize)."
        case .invalidInputResolution(let resolution):
            "OlmoEarth input resolution must be positive; found \(resolution)."
        case .sequenceTooLong(let maximum, let actual):
            "OlmoEarth supports at most \(maximum) timestamps; found \(actual)."
        case .invalidTimestamp(let day, let month, let year):
            "OlmoEarth timestamp must be (day 1-31, month 0-11, year); found \(day), \(month), \(year)."
        }
    }
}
