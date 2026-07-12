import Foundation
@preconcurrency import MLX
import MLXNN

public enum VideoDepthAnythingWeightError: Error, Equatable, LocalizedError, Sendable {
    case duplicateMappedTensor(String)

    public var errorDescription: String? {
        switch self {
        case .duplicateMappedTensor(let name):
            "Video Depth Anything weight mapping produced duplicate tensor '\(name)'."
        }
    }
}

/// Native layout mapping for VDA PyTorch state dicts, loaded either from a
/// deterministically converted safetensors file or the restricted native PTH
/// reader. Neither path executes Python or pickle callables.
public enum VideoDepthAnythingWeights {
    public static let sourceTensorCount = 351
    public static let sourceScalarCount = 29_080_193
    /// DINOv2's training-only mask token is serialized in the source state
    /// dict but is not read anywhere in Video Depth Anything inference.
    public static let excludedTrainingScalarCount = 384
    public static let inferenceScalarCount = sourceScalarCount - excludedTrainingScalarCount

    public static func load(
        model: VideoDepthAnythingModel,
        safetensorsURL: URL,
        dtype: DType = .float32
    ) throws {
        try HFSafetensorsWeightsLoader.applyWeights(
            url: safetensorsURL,
            to: model,
            dtype: dtype,
            verify: .all,
            mapper: mapSourceTensor
        )
    }

    /// Loads the official pinned PyTorch state dict through the restricted,
    /// non-executing archive reader. The archive parser never imports Python,
    /// invokes pickle reducers, or constructs arbitrary objects.
    public static func load(
        model: VideoDepthAnythingModel,
        archive: PyTorchStateDictArchive,
        dtype: DType = .float32
    ) throws {
        var target: [String: MLXArray] = [:]
        for descriptor in archive.tensors {
            let source = try archive.loadArray(for: descriptor, dtype: dtype)
            for (key, value) in mapSourceTensor(key: descriptor.name, value: source) {
                guard target[key] == nil else {
                    throw VideoDepthAnythingWeightError.duplicateMappedTensor(key)
                }
                target[key] = value
            }
        }
        try model.update(parameters: ModuleParameters.unflattened(target), verify: .all)
        MLX.eval(model)
    }

    public static func mapSourceTensor(
        key: String,
        value: MLXArray
    ) -> [(String, MLXArray)] {
        if key == "pretrained.mask_token" {
            // Masked-patch training state is present upstream but unused by
            // Video Depth Anything inference.
            return []
        }
        if key == "pretrained.pos_embed" {
            guard value.ndim == 3, value.dim(1) >= 2 else { return [] }
            return [
                ("pretrained.class_position", value[0..., 0..<1, 0...]),
                ("pretrained.patch_position", value[0..., 1..., 0...]),
            ]
        }
        if key == "head.resize_layers.0.weight" || key == "head.resize_layers.1.weight" {
            return [(key, contiguous(value.transposed(1, 2, 3, 0)))]
        }
        if value.ndim == 4 {
            return [(key, contiguous(value.transposed(0, 2, 3, 1)))]
        }
        return [(key, value)]
    }

    private static func contiguous(_ value: MLXArray) -> MLXArray {
        value.reshaped(-1).reshaped(value.shape)
    }
}
