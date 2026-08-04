import Foundation
@preconcurrency import MLX
import MLXFast
import MLXNN

/// Native MLX implementation of the pinned TerraMind-1.0-base ImpactMesh Flood graph.
///
/// Inputs use the TerraTorch layout `[batch, channels, 4, 256, 256]`. The model
/// returns unnormalized logits in `[batch, 2, 256, 256]`; callers must blend
/// tiled logits before applying softmax to preserve TerraTorch parity.
public final class TerraMindFloodModel: @unchecked Sendable {
    public static let tileSize = 256
    public static let timestampCount = 4
    public static let classCount = 2

    private let weights: [String: MLXArray]
    private let dtype: DType

    public init(weights: [String: MLXArray]) throws {
        let missing = Self.requiredTensorNames.filter { weights[$0] == nil }
        guard missing.isEmpty else {
            throw TerraMindFloodModelError.missingTensors(missing)
        }
        self.weights = weights
        self.dtype = weights["encoder.encoder.encoder.0.attn.qkv.weight"]!.dtype
    }

    public func callAsFunction(
        s2l2a: MLXArray,
        s1rtc: MLXArray,
        dem: MLXArray
    ) throws -> MLXArray {
        try Self.validateInput(s2l2a, channels: 12, name: "S2L2A")
        try Self.validateInput(s1rtc, channels: 2, name: "S1RTC")
        try Self.validateInput(dem, channels: 1, name: "DEM")
        guard s2l2a.dim(0) == s1rtc.dim(0), s2l2a.dim(0) == dem.dim(0) else {
            throw TerraMindFloodModelError.inconsistentBatchSize
        }

        var levels = Array(repeating: [MLXArray](), count: 4)
        for timestamp in 0..<Self.timestampCount {
            var hidden = MLX.concatenated([
                embed(s2l2a, timestamp: timestamp, modality: "untok_sen2l2a@224", channels: 12),
                embed(s1rtc, timestamp: timestamp, modality: "untok_sen1rtc@224", channels: 2),
                embed(dem, timestamp: timestamp, modality: "untok_dem@224", channels: 1),
            ], axis: 1)

            var selectedIndex = 0
            for block in 0..<12 {
                hidden = transformerBlock(hidden, index: block)
                // Materialize each block so four temporal passes do not retain one
                // enormous lazy attention graph on the unified-memory heap.
                MLX.eval(hidden)
                guard Self.selectedBlocks.contains(block) else { continue }
                let output = block == 11
                    ? layerNorm(
                        hidden,
                        weight: weight("encoder.encoder.encoder_norm.weight"),
                        bias: weight("encoder.encoder.encoder_norm.bias"),
                        epsilon: 1e-6
                    )
                    : hidden
                let merged = output
                    .reshaped(output.dim(0), 3, 256, 768)
                    .mean(axis: 1)
                MLX.eval(merged)
                levels[selectedIndex].append(merged)
                selectedIndex += 1
            }
        }

        let temporal = levels.map { timestepFeatures in
            MLX.concatenated(timestepFeatures, axis: 2)
                .reshaped(timestepFeatures[0].dim(0), 16, 16, 3_072)
        }
        return decode(temporal).transposed(0, 3, 1, 2)
    }

    private func embed(
        _ input: MLXArray,
        timestamp: Int,
        modality: String,
        channels: Int
    ) -> MLXArray {
        let image = input[0..., 0..., timestamp, 0..., 0...]
            .asType(dtype)
            .transposed(0, 2, 3, 1)
        let batch = image.dim(0)
        let patches = image
            .reshaped(batch, 16, 16, 16, 16, channels)
            .transposed(0, 1, 3, 2, 4, 5)
            .reshaped(batch, 256, 256 * channels)
        let projected = MLX.matmul(
            patches,
            weight("encoder.encoder.encoder_embeddings.\(modality).proj.weight").T
        )
        return projected
            + weight("encoder.encoder.encoder_embeddings.\(modality).pos_emb")
            + weight("encoder.encoder.encoder_embeddings.\(modality).mod_emb")
    }

    private func transformerBlock(_ input: MLXArray, index: Int) -> MLXArray {
        let prefix = "encoder.encoder.encoder.\(index)"
        let normalizedAttention = layerNorm(
            input,
            weight: weight("\(prefix).norm1.weight"),
            bias: weight("\(prefix).norm1.bias"),
            epsilon: 1e-6
        )
        let qkv = MLX.matmul(normalizedAttention, weight("\(prefix).attn.qkv.weight").T)
        let batch = qkv.dim(0)
        let count = qkv.dim(1)
        let split = qkv
            .reshaped(batch, count, 3, 12, 64)
            .transposed(2, 0, 3, 1, 4)
        let attended = MLXFast.scaledDotProductAttention(
            queries: split[0],
            keys: split[1],
            values: split[2],
            scale: 0.125,
            mask: .none
        )
        let attentionOutput = MLX.matmul(
            attended.transposed(0, 2, 1, 3).reshaped(batch, count, 768),
            weight("\(prefix).attn.proj.weight").T
        )
        let residual = input + attentionOutput
        let normalizedMLP = layerNorm(
            residual,
            weight: weight("\(prefix).norm2.weight"),
            bias: weight("\(prefix).norm2.bias"),
            epsilon: 1e-6
        )
        let gated = MLXNN.silu(MLX.matmul(normalizedMLP, weight("\(prefix).mlp.fc1.weight").T))
            * MLX.matmul(normalizedMLP, weight("\(prefix).mlp.fc3.weight").T)
        return residual + MLX.matmul(gated, weight("\(prefix).mlp.fc2.weight").T)
    }

    private func decode(_ features: [MLXArray]) -> MLXArray {
        var pyramid: [MLXArray] = []

        var first = transposedConvolution(
            features[0],
            weight: weight("neck.2.fpn1.0.weight"),
            bias: weight("neck.2.fpn1.0.bias")
        )
        first = batchNorm(first, prefix: "neck.2.fpn1.1")
        first = MLXNN.gelu(first)
        first = transposedConvolution(
            first,
            weight: weight("neck.2.fpn1.3.weight"),
            bias: weight("neck.2.fpn1.3.bias")
        )
        pyramid.append(first)
        pyramid.append(transposedConvolution(
            features[1],
            weight: weight("neck.2.fpn2.0.weight"),
            bias: weight("neck.2.fpn2.0.bias")
        ))
        pyramid.append(features[2])
        pyramid.append(MaxPool2d(kernelSize: 2, stride: 2)(features[3]))

        var hidden = pyramid[3]
        for block in 0..<4 {
            let target = block == 3 ? hidden : upsampleNearest(hidden, factor: 2)
            if block < 3 {
                hidden = MLX.concatenated([target, pyramid[2 - block]], axis: 3)
            } else {
                hidden = target
            }
            hidden = decoderConvolution(hidden, prefix: "decoder.decoder.blocks.\(block).conv1")
            hidden = decoderConvolution(hidden, prefix: "decoder.decoder.blocks.\(block).conv2")
            MLX.eval(hidden)
        }

        var logits = MLX.conv2d(hidden, weight("head.head.2.weight"), padding: 0)
        logits = logits + weight("head.head.2.bias")
        return Upsample(scaleFactor: 4.0, mode: .linear(alignCorners: false))(logits)
    }

    private func decoderConvolution(_ input: MLXArray, prefix: String) -> MLXArray {
        let convolved = MLX.conv2d(input, weight("\(prefix).0.weight"), padding: 1)
        return MLXNN.relu(batchNorm(convolved, prefix: "\(prefix).1"))
    }

    private func transposedConvolution(
        _ input: MLXArray,
        weight: MLXArray,
        bias: MLXArray
    ) -> MLXArray {
        MLX.convTransposed2d(input, weight, stride: 2) + bias
    }

    private func batchNorm(_ input: MLXArray, prefix: String) -> MLXArray {
        let promoted = input.asType(.float32)
        let mean = weight("\(prefix).running_mean").asType(.float32)
        let variance = weight("\(prefix).running_var").asType(.float32)
        let scale = weight("\(prefix).weight").asType(.float32)
        let bias = weight("\(prefix).bias").asType(.float32)
        return (((promoted - mean) / MLX.sqrt(variance + 1e-5)) * scale + bias).asType(input.dtype)
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

    private func upsampleNearest(_ input: MLXArray, factor: Float) -> MLXArray {
        Upsample(scaleFactor: FloatOrArray(factor), mode: .nearest)(input)
    }

    private func weight(_ name: String) -> MLXArray { weights[name]! }

    private static func validateInput(_ input: MLXArray, channels: Int, name: String) throws {
        let expectedTail = [channels, timestampCount, tileSize, tileSize]
        guard input.ndim == 5, Array(input.shape.dropFirst()) == expectedTail else {
            throw TerraMindFloodModelError.invalidInputShape(
                name: name,
                expected: [-1] + expectedTail,
                actual: input.shape
            )
        }
    }

    private static let selectedBlocks: Set<Int> = [2, 5, 8, 11]

    public static let requiredTensorNames: [String] = {
        var names: [String] = []
        for modality in ["untok_sen2l2a@224", "untok_sen1rtc@224", "untok_dem@224"] {
            let prefix = "encoder.encoder.encoder_embeddings.\(modality)"
            names += ["\(prefix).mod_emb", "\(prefix).pos_emb", "\(prefix).proj.weight"]
        }
        for block in 0..<12 {
            let prefix = "encoder.encoder.encoder.\(block)"
            names += [
                "\(prefix).norm1.weight", "\(prefix).norm1.bias",
                "\(prefix).attn.qkv.weight", "\(prefix).attn.proj.weight",
                "\(prefix).norm2.weight", "\(prefix).norm2.bias",
                "\(prefix).mlp.fc1.weight", "\(prefix).mlp.fc2.weight", "\(prefix).mlp.fc3.weight",
            ]
        }
        names += ["encoder.encoder.encoder_norm.weight", "encoder.encoder.encoder_norm.bias"]
        for block in 0..<4 {
            for convolution in ["conv1", "conv2"] {
                let prefix = "decoder.decoder.blocks.\(block).\(convolution)"
                names += [
                    "\(prefix).0.weight", "\(prefix).1.weight", "\(prefix).1.bias",
                    "\(prefix).1.running_mean", "\(prefix).1.running_var",
                ]
            }
        }
        names += [
            "head.head.2.weight", "head.head.2.bias",
            "neck.2.fpn1.0.weight", "neck.2.fpn1.0.bias",
            "neck.2.fpn1.1.weight", "neck.2.fpn1.1.bias",
            "neck.2.fpn1.1.running_mean", "neck.2.fpn1.1.running_var",
            "neck.2.fpn1.3.weight", "neck.2.fpn1.3.bias",
            "neck.2.fpn2.0.weight", "neck.2.fpn2.0.bias",
        ]
        return names
    }()
}

public enum TerraMindFloodModelError: Error, Equatable, LocalizedError, Sendable {
    case missingTensors([String])
    case invalidInputShape(name: String, expected: [Int], actual: [Int])
    case inconsistentBatchSize

    public var errorDescription: String? {
        switch self {
        case .missingTensors(let names):
            "TerraMind Flood checkpoint is missing tensors: \(names.joined(separator: ", "))."
        case .invalidInputShape(let name, let expected, let actual):
            "TerraMind Flood \(name) input must have shape \(expected); found \(actual)."
        case .inconsistentBatchSize:
            "TerraMind Flood modalities must use the same batch size."
        }
    }
}
