import Foundation
@preconcurrency import MLX
import MLXFast
import MLXNN

/// Native MLX port of the dense 16-latent TRELLIS sparse-structure decoder.
/// The official dependency checkpoint is stored as PyTorch NCDHW kernels;
/// kernels are transposed at use time into MLX's NDHWC layout.
struct Trellis2StructureDecoder {
    private let weights: Trellis2WeightStore

    init(checkpointURL: URL) throws {
        self.weights = try Trellis2WeightStore(url: checkpointURL)
        try validateWeights()
    }

    init(weights: [String: MLXArray]) throws {
        self.weights = Trellis2WeightStore(values: weights)
        try validateWeights()
    }

    private func validateWeights() throws {
        _ = try weights.checkedTensor("input_layer.weight", shape: [512, 8, 3, 3, 3])
        _ = try weights.checkedTensor("out_layer.2.weight", shape: [1, 32, 3, 3, 3])
        let residuals: [(String, Int)] = [
            ("middle_block.0", 512),
            ("middle_block.1", 512),
            ("blocks.0", 512),
            ("blocks.1", 512),
            ("blocks.3", 128),
            ("blocks.4", 128),
            ("blocks.6", 32),
            ("blocks.7", 32),
        ]
        for (prefix, channels) in residuals {
            _ = try weights.checkedTensor(
                "\(prefix).conv1.weight",
                shape: [channels, channels, 3, 3, 3]
            )
            _ = try weights.checkedTensor(
                "\(prefix).conv2.weight",
                shape: [channels, channels, 3, 3, 3]
            )
        }
        _ = try weights.checkedTensor("blocks.2.conv.weight", shape: [1_024, 512, 3, 3, 3])
        _ = try weights.checkedTensor("blocks.5.conv.weight", shape: [256, 128, 3, 3, 3])
    }

    func decode(
        latentTokens: MLXArray,
        outputResolution: Int = 32,
        maximumTokens: Int
    ) throws -> [Trellis2VoxelCoordinate] {
        guard latentTokens.shape == [1, 4_096, 8] else {
            throw Trellis2ModelError.invalidInputShape(
                expected: "[1, 4096, 8] sparse-structure latent",
                actual: latentTokens.shape
            )
        }
        guard outputResolution == 32 || outputResolution == 64 else {
            throw Trellis2ModelError.unsupportedCheckpoint(
                "the native structure decoder supports 32 or 64 output coordinates"
            )
        }

        var hidden = latentTokens.reshaped(1, 16, 16, 16, 8)
        hidden = try convolution(hidden, prefix: "input_layer").asType(.float16)
        hidden = try residualBlock(hidden, prefix: "middle_block.0")
        hidden = try residualBlock(hidden, prefix: "middle_block.1")
        hidden = try residualBlock(hidden, prefix: "blocks.0")
        hidden = try residualBlock(hidden, prefix: "blocks.1")
        hidden = try pixelShuffle3D(convolution(hidden, prefix: "blocks.2.conv"))
        hidden = try residualBlock(hidden, prefix: "blocks.3")
        hidden = try residualBlock(hidden, prefix: "blocks.4")
        hidden = try pixelShuffle3D(convolution(hidden, prefix: "blocks.5.conv"))
        hidden = affineLayerNorm(hidden, prefix: "out_layer.0")
        hidden = silu(hidden)
        hidden = try convolution(hidden.asType(.float32), prefix: "out_layer.2")
        MLX.eval(hidden)

        let occupied = (hidden[0, 0..., 0..., 0..., 0] .> 0)
            .asType(.int32)
            .asArray(Int32.self)
        var coordinates = Set<Trellis2VoxelCoordinate>()
        coordinates.reserveCapacity(min(maximumTokens, 49_152))
        for x in 0..<64 {
            for y in 0..<64 {
                for z in 0..<64 where occupied[(x * 64 + y) * 64 + z] != 0 {
                    let divisor = Int32(64 / outputResolution)
                    coordinates.insert(Trellis2VoxelCoordinate(
                        x: Int32(x) / divisor,
                        y: Int32(y) / divisor,
                        z: Int32(z) / divisor
                    ))
                }
            }
        }
        guard !coordinates.isEmpty else { throw Trellis2ModelError.emptySparseStructure }
        guard coordinates.count <= maximumTokens else {
            throw Trellis2ModelError.excessiveSparseStructure(
                actual: coordinates.count,
                maximum: maximumTokens
            )
        }
        return coordinates.sorted {
            ($0.x, $0.y, $0.z) < ($1.x, $1.y, $1.z)
        }
    }

    private func residualBlock(_ input: MLXArray, prefix: String) throws -> MLXArray {
        var hidden = affineLayerNorm(input, prefix: "\(prefix).norm1")
        hidden = silu(hidden)
        hidden = try convolution(hidden, prefix: "\(prefix).conv1")
        hidden = affineLayerNorm(hidden, prefix: "\(prefix).norm2")
        hidden = silu(hidden)
        hidden = try convolution(hidden, prefix: "\(prefix).conv2")
        return hidden + input
    }

    private func affineLayerNorm(_ input: MLXArray, prefix: String) -> MLXArray {
        MLXFast.layerNorm(
            input.asType(.float32),
            weight: weights.values["\(prefix).weight"]!.asType(.float32),
            bias: weights.values["\(prefix).bias"]!.asType(.float32),
            eps: 1e-6
        ).asType(input.dtype)
    }

    private func convolution(_ input: MLXArray, prefix: String) throws -> MLXArray {
        let pytorchWeight = try weights.tensor("\(prefix).weight")
        guard pytorchWeight.ndim == 5, pytorchWeight.dim(1) == input.dim(4) else {
            throw Trellis2ModelError.invalidInputShape(
                expected: "PyTorch [output, \(input.dim(4)), depth, height, width] kernel",
                actual: pytorchWeight.shape
            )
        }
        let weight = pytorchWeight.transposed(0, 2, 3, 4, 1)
        var output = MLX.conv3d(
            input.asType(weight.dtype),
            weight,
            stride: 1,
            padding: 1
        )
        if let bias = weights.values["\(prefix).bias"] {
            output = output + bias
        }
        return output
    }

    private func pixelShuffle3D(_ input: MLXArray) throws -> MLXArray {
        guard input.ndim == 5, input.dim(4).isMultiple(of: 8) else {
            throw Trellis2ModelError.invalidInputShape(
                expected: "NDHWC tensor with channels divisible by 8",
                actual: input.shape
            )
        }
        let batch = input.dim(0)
        let depth = input.dim(1)
        let height = input.dim(2)
        let width = input.dim(3)
        let channels = input.dim(4) / 8
        return input
            .reshaped(batch, depth, height, width, channels, 2, 2, 2)
            .transposed(0, 1, 5, 2, 6, 3, 7, 4)
            .reshaped(batch, depth * 2, height * 2, width * 2, channels)
    }
}
