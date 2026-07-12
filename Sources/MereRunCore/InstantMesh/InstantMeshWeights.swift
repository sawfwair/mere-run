import Foundation
@preconcurrency import MLX
import MLXNN

public enum InstantMeshWeightError: Error, Equatable, LocalizedError, Sendable {
    case invalidInventory(tensorCount: Int, scalarCount: Int)

    public var errorDescription: String? {
        switch self {
        case .invalidInventory(let tensorCount, let scalarCount):
            "InstantMesh checkpoint has \(tensorCount) tensors/\(scalarCount) scalars; expected "
                + "\(InstantMeshWeights.sourceTensorCount)/\(InstantMeshWeights.sourceScalarCount)."
        }
    }
}

/// Exact tensor contract for the reconstruction-only InstantMesh Base weights.
///
/// The published Lightning CKPT has a nested pickle root and is deliberately
/// not interpreted in-process. Audited release tooling verifies its pinned
/// bytes with PyTorch's weights-only loader and emits deterministic
/// safetensors; runtime inference loads only that non-executable format.
public enum InstantMeshWeights {
    public static let sourceTensorCount = 455
    public static let sourceScalarCount = 313_352_516

    /// Filled by the deterministic converter proof for the exact pinned CKPT.
    public static let convertedSafetensorsByteCount: Int64 = 1_253_463_832
    public static let convertedSafetensorsSHA256 =
        "2380601d17f6a817de0bf5328188ccea397af9d75c07b4b3cc476322dcca76af"

    public static func load(
        model: InstantMeshModel,
        safetensorsURL: URL,
        dtype: DType = .float16
    ) throws {
        try HFSafetensorsWeightsLoader.applyWeights(
            url: safetensorsURL,
            to: model,
            dtype: dtype,
            verify: .all,
            mapper: mapSourceTensor
        )
    }

    public static func mapSourceTensor(key: String, value: MLXArray) -> [(String, MLXArray)] {
        switch key {
        case "encoder.model.embeddings.patch_embeddings.projection.weight":
            // PyTorch Conv2d [out, in, height, width] -> MLX [out, height, width, in].
            return [(key, contiguous(value.transposed(0, 2, 3, 1)))]
        case "transformer.deconv.weight":
            // PyTorch ConvTranspose2d [in, out, height, width] ->
            // MLX ConvTransposed2d [out, height, width, in].
            return [(key, contiguous(value.transposed(1, 2, 3, 0)))]
        default:
            return [(key, value)]
        }
    }

    private static func contiguous(_ value: MLXArray) -> MLXArray {
        value.reshaped(-1).reshaped(value.shape)
    }
}
