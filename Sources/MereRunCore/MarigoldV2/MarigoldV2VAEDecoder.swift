import Foundation
import MLX
import MLXNN

/// Applies the fine-tuned VAE decoder that some Marigold V2 checkpoints ship.
///
/// The Stage 2 depth recipe retrains the decoder alongside the transformer
/// adapters, so `VAE.*` holds full replacement weights rather than low-rank
/// deltas. Checkpoints trained without decoder fine-tuning carry no `VAE.*`
/// tensors and decode through the frozen base decoder unchanged.
public enum MarigoldV2VAEDecoder {
    public static let sourcePrefix = "VAE."

    /// Tensor count of the published decoder, used to reject a truncated or
    /// partially rewritten checkpoint before it reaches the module tree.
    public static let expectedTensorCount = 108

    public enum DecoderError: LocalizedError {
        case unexpectedTensorCount(expected: Int, actual: Int)

        public var errorDescription: String? {
            switch self {
            case .unexpectedTensorCount(let expected, let actual):
                return "Marigold V2 VAE decoder tensor count mismatch: expected \(expected), found \(actual)."
            }
        }
    }

    /// Overrides the decoder in `vae` when the checkpoint carries one.
    ///
    /// - Returns: the number of tensors applied, or zero when the checkpoint
    ///   decodes with the frozen base decoder.
    @discardableResult
    public static func applyIfPresent(
        url: URL,
        to vae: QwenImageEditVAE
    ) throws -> Int {
        var weights: [String: MLXArray] = [:]
        let count = try SafetensorsStreamingLoader.forEachTensor(
            url: url,
            where: { $0.hasPrefix(sourcePrefix) },
            dtype: .bfloat16
        ) { key, value in
            let path = String(key.dropFirst(sourcePrefix.count))
            for (mappedKey, mappedValue) in QwenImageEditVAE.weightMapper(key: path, value: value) {
                weights[mappedKey] = mappedValue
            }
        }

        guard count > 0 else {
            return 0
        }
        guard count == expectedTensorCount else {
            throw DecoderError.unexpectedTensorCount(expected: expectedTensorCount, actual: count)
        }

        try vae.underlyingVAE.update(
            parameters: ModuleParameters.unflattened(weights),
            verify: [.shapeMismatch, .noUnusedKeys]
        )
        MLX.eval(vae)
        Memory.clearCache()
        return count
    }
}
