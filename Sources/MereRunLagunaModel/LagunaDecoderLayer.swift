import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom
import MereRunTensor
import MereRunGemmaModel
import MereRunDecode

package final class LagunaDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") package var selfAttention: LagunaAttention
    @ModuleInfo(key: "mlp") package var mlp: LagunaFeedForward
    @ModuleInfo(key: "input_layernorm") package var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") package var postAttentionLayerNorm: RMSNorm

    package init(config: LagunaConfig, layerIndex: Int) {
        self._selfAttention.wrappedValue = LagunaAttention(config: config, layerIndex: layerIndex)
        self._mlp.wrappedValue = config.isSparse(layerIndex: layerIndex)
            ? LagunaSparseMoE(config: config)
            : LagunaDenseMLP(
                inputDimensions: config.hiddenSize,
                hiddenDimensions: config.intermediateSize
            )
        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
        super.init()
    }

    package func callAsFunction(
        _ x: MLXArray,
        cache: Gemma4AttentionCache?,
        precomputedMask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        precomputedRoPEAtlas: MLXArray? = nil,
        useCustomKernels: Bool = true
    ) -> MLXArray {
        let attentionBranch = selfAttention(
            inputLayerNorm(x),
            cache: cache,
            precomputedMask: precomputedMask,
            precomputedRoPEAtlas: precomputedRoPEAtlas,
            residualForFusedQKV: x,
            rmsNormWeight: inputLayerNorm.weight,
            useCustomKernels: useCustomKernels
        )
        let attended: MLXArray
        let normalized: MLXArray
        if useCustomKernels,
           LagunaGraphAccelerationPolicy.prefillFusedResidualRMSNormEnabled,
           let fused = LagunaFusedPrefill.residualRMSNorm(
               residual: x,
               branch: attentionBranch,
               weight: postAttentionLayerNorm.weight
           ) {
            attended = fused.summed
            normalized = fused.normalized
        } else {
            attended = x + attentionBranch
            normalized = postAttentionLayerNorm(attended)
        }
        if normalized.dim(0) == 1,
           normalized.dim(1) == 1,
           let sparse = mlp as? LagunaSparseMoE {
            return sparse(
                normalized,
                residual: attended,
                useCustomKernels: useCustomKernels
            )
        }
        if let sparse = mlp as? LagunaSparseMoE {
            return attended + sparse(
                normalized,
                residual: nil,
                useCustomKernels: useCustomKernels
            )
        }
        return attended + mlp(normalized)
    }

    package func prepareFusedNormAffineQKVWarmUp() -> (rows: Int, output: MLXArray)? {
        selfAttention.prepareFusedNormAffineQKVWarmUp(
            normWeight: inputLayerNorm.weight
        )
    }

    /// Preserve every terminal-layer K/V row while carrying only the consumed
    /// final residual row through attention output and the MLP.
    package func callLastPrefillRow(
        _ x: MLXArray,
        cache: Gemma4AttentionCache?
    ) -> MLXArray {
        let normalized = inputLayerNorm(x)
        let attentionBranch = selfAttention.callLastPrefillRow(
            normalized,
            cache: cache
        )
        let attended = x[0..., (x.dim(1) - 1)..., 0...] + attentionBranch
        return attended + mlp(postAttentionLayerNorm(attended))
    }
}
