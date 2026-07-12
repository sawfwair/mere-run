import Foundation
@preconcurrency import MLX
import MLXNN

public enum TripoSRWeightError: Error, Equatable, LocalizedError, Sendable {
    case duplicateMappedTensor(String)
    case invalidInventory(tensorCount: Int, scalarCount: Int)
    case invalidMaterializationBatchByteCount(Int)

    public var errorDescription: String? {
        switch self {
        case .duplicateMappedTensor(let name):
            "TripoSR weight mapping produced duplicate tensor '\(name)'."
        case .invalidInventory(let tensorCount, let scalarCount):
            "TripoSR checkpoint has \(tensorCount) tensors/\(scalarCount) scalars; expected "
                + "\(TripoSRWeights.sourceTensorCount)/\(TripoSRWeights.sourceScalarCount)."
        case .invalidMaterializationBatchByteCount(let value):
            "TripoSR materialization batch byte count must be positive; received \(value)."
        }
    }
}

/// Exact native layout mapping for the pinned upstream TripoSR state dict.
///
/// The CKPT path uses `PyTorchStateDictArchive`, which interprets only the
/// whitelisted tensor-state-dict pickle grammar and never executes Python
/// reducers. The safetensors path is intended for deterministic release
/// packages emitted by `convert_triposr.py`.
public enum TripoSRWeights {
    public static let sourceTensorCount = 549
    public static let sourceScalarCount = 419_275_628
    /// Byte-stable output produced twice by the committed converter from the
    /// exact pinned checkpoint (safetensors 0.8.0, no unordered metadata map).
    public static let convertedSafetensorsByteCount: Int64 = 1_677_170_936
    public static let convertedSafetensorsSHA256 =
        "f72bb520b8b1a5639600ac818496f22d6ccb3b42d3942412bd1e2375ef780a2b"

    public static func load(
        model: TripoSRModel,
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

    public static func load(
        model: TripoSRModel,
        archive: PyTorchStateDictArchive,
        dtype: DType = .float16,
        materializationBatchByteCount: Int = 128 * 1_024 * 1_024
    ) throws {
        guard materializationBatchByteCount > 0 else {
            throw TripoSRWeightError.invalidMaterializationBatchByteCount(
                materializationBatchByteCount
            )
        }
        let scalarCount = archive.tensors.reduce(0) { $0 + $1.elementCount }
        guard archive.tensors.count == sourceTensorCount, scalarCount == sourceScalarCount else {
            throw TripoSRWeightError.invalidInventory(
                tensorCount: archive.tensors.count,
                scalarCount: scalarCount
            )
        }

        var target: [String: MLXArray] = [:]
        target.reserveCapacity(sourceTensorCount)
        var pendingMaterialization: [MLXArray] = []
        var pendingByteCount = 0
        for descriptor in archive.tensors {
            let source = try archive.loadArray(for: descriptor, dtype: dtype)
            for (key, value) in mapSourceTensor(key: descriptor.name, value: source) {
                guard target[key] == nil else {
                    throw TripoSRWeightError.duplicateMappedTensor(key)
                }
                target[key] = value
                pendingMaterialization.append(value)
                pendingByteCount += value.shape.reduce(1, *) * byteCount(of: value.dtype)
                if pendingByteCount >= materializationBatchByteCount {
                    // Materialize bounded groups so float16 casts and the two
                    // convolution transposes do not retain every float32
                    // backing storage until first inference.
                    MLX.eval(pendingMaterialization)
                    pendingMaterialization.removeAll(keepingCapacity: true)
                    pendingByteCount = 0
                }
            }
        }
        if !pendingMaterialization.isEmpty {
            MLX.eval(pendingMaterialization)
        }
        try model.update(parameters: ModuleParameters.unflattened(target), verify: .all)
    }

    public static func mapSourceTensor(key: String, value: MLXArray) -> [(String, MLXArray)] {
        switch key {
        case "image_tokenizer.model.embeddings.patch_embeddings.projection.weight":
            // PyTorch Conv2d [out, in, height, width] -> MLX [out, height, width, in].
            return [(key, contiguous(value.transposed(0, 2, 3, 1)))]
        case "post_processor.upsample.weight":
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

    private static func byteCount(of dtype: DType) -> Int {
        switch dtype {
        case .bool, .int8, .uint8: 1
        case .float16, .bfloat16, .int16, .uint16: 2
        case .float32, .int32, .uint32: 4
        case .float64, .int64, .uint64, .complex64: 8
        }
    }
}
