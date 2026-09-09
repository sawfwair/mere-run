import Foundation
import MLX
import MLXFast
import MLXNN
import Cmlx

@inline(__always)
package func gemma4RMSNormNoScale(_ x: MLXArray, eps: Float) -> MLXArray {
    let noWeight = MLXArray.mlxNone
    let stream = StreamOrDevice.default
    var result = mlx_array_new()
    _ = withExtendedLifetime((noWeight, stream)) {
        mlx_fast_rms_norm(&result, x.ctx, noWeight.ctx, eps, stream.ctx)
    }
    return MLXArray(result)
}

@inline(__always)
package func gemma4RMSNormNoScale(_ x: MLXArray, weight: MLXArray, eps: Float) -> MLXArray {
    MLXFast.rmsNorm(x, weight: weight, eps: eps)
}

package struct Gemma4SharedKVState {
    package let keys: MLXArray
    package let values: MLXArray
    package let offset: Int
    package let maxSize: Int?

    package init(keys: MLXArray, values: MLXArray, offset: Int, maxSize: Int?) {
        self.keys = keys
        self.values = values
        self.offset = offset
        self.maxSize = maxSize
    }
}

package struct Gemma4LanguageModelOutput {
    package let hidden: MLXArray
    package let preNormHidden: MLXArray
    package let sharedKVStates: [String: Gemma4SharedKVState]
    package let hiddenStates: [MLXArray]?
}

package struct Gemma4ForwardOutput {
    package let logits: MLXArray
    package let hidden: MLXArray
    package let sharedKVStates: [String: Gemma4SharedKVState]
}

package protocol Gemma4CausalModel: AnyObject, Sendable {
    func forward(inputIds: MLXArray, cache: [Gemma4AttentionCache]?) -> MLXArray
    /// Forward for a prefill chunk: fills the KV cache but only returns logits
    /// for the chunk's final position.
    func prefillStep(inputIds: MLXArray, cache: [Gemma4AttentionCache]?) -> MLXArray
    func forwardForSpeculation(inputIds: MLXArray, cache: [Gemma4AttentionCache]?) -> Gemma4ForwardOutput
    func inputEmbeddings(for inputIds: MLXArray) -> MLXArray
    func speculativeLogits(fromHidden hidden: MLXArray) -> MLXArray
    func speculativeDraftHidden(_ hidden: MLXArray) -> MLXArray
    func makeAttentionCache(quantization: Gemma4KVCacheQuantization?) -> [Gemma4AttentionCache]
}

package extension Gemma4CausalModel {
    func prefillStep(inputIds: MLXArray, cache: [Gemma4AttentionCache]?) -> MLXArray {
        forward(inputIds: inputIds, cache: cache)
    }
}
