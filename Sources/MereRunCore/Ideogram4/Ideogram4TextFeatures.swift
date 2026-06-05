import Foundation
import MLX

public enum Ideogram4TextFeatures {
    public static let qwen3VLActivationLayers = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35]

    public static func encode(
        inputIds: MLXArray,
        attentionMask: MLXArray?,
        using encoder: QwenEncoder,
        activationLayers: [Int] = qwen3VLActivationLayers
    ) -> MLXArray {
        concatenate(encoder.forwardActivationHiddenStates(
            inputIds: inputIds,
            attentionMask: attentionMask,
            activationLayers: activationLayers
        ))
    }

    public static func encode(
        embeddings: MLXArray,
        attentionMask: MLXArray?,
        positionIds: MLXArray? = nil,
        tokenTypes: MLXArray? = nil,
        using encoder: QwenEncoder,
        activationLayers: [Int] = qwen3VLActivationLayers
    ) -> MLXArray {
        concatenate(encoder.forwardActivationHiddenStates(
            embeddings: embeddings,
            attentionMask: attentionMask,
            positionIds: positionIds,
            tokenTypes: tokenTypes,
            activationLayers: activationLayers
        ))
    }

    public static func concatenate(_ hiddenStates: [MLXArray]) -> MLXArray {
        guard let first = hiddenStates.first else {
            return MLX.zeros([1, 0, 0], dtype: .bfloat16)
        }
        if hiddenStates.count == 1 {
            return first
        }
        // Upstream Ideogram stacks tap activations as (tap, batch, seq, hidden),
        // permutes to (batch, seq, hidden, tap), then flattens. That interleaves
        // tap features per hidden channel instead of appending whole layers.
        return MLX.stacked(hiddenStates, axis: -1)
            .reshaped(first.dim(0), first.dim(1), first.dim(2) * hiddenStates.count)
    }
}
