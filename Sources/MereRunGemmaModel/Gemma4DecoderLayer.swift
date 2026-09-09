import Foundation
import MLX
import MLXFast
import MLXNN

final class Gemma4DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: Gemma4Attention
    @ModuleInfo(key: "mlp") var mlp: Gemma4MLP
    @ModuleInfo(key: "router") var router: Gemma4Router?
    @ModuleInfo(key: "experts") var experts: Gemma4Experts?
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm_2") var preFeedforwardLayerNorm2: RMSNorm?
    @ModuleInfo(key: "post_feedforward_layernorm_1") var postFeedforwardLayerNorm1: RMSNorm?
    @ModuleInfo(key: "post_feedforward_layernorm_2") var postFeedforwardLayerNorm2: RMSNorm?
    @ModuleInfo(key: "per_layer_input_gate") var perLayerInputGate: Linear
    @ModuleInfo(key: "per_layer_projection") var perLayerProjection: Linear
    @ModuleInfo(key: "post_per_layer_input_norm") var postPerLayerInputNorm: RMSNorm
    @ParameterInfo(key: "layer_scalar") var layerScalar: MLXArray

    private let hasPerLayerInput: Bool
    private let rmsNormEps: Float
    private var compiledPostSegment: Gemma4CompiledSegment?
    private var compiledPostAttempted = false

    init(config: Gemma4TextConfig, layerIndex: Int, forceKVShared: Bool = false) {
        self._selfAttention.wrappedValue = Gemma4Attention(
            config: config,
            layerIndex: layerIndex,
            forceKVShared: forceKVShared
        )
        self._mlp.wrappedValue = Gemma4MLP(config: config, layerIndex: layerIndex, forceKVShared: forceKVShared)
        if config.enableMoEBlock {
            self._router.wrappedValue = Gemma4Router(config: config)
            self._experts.wrappedValue = Gemma4Experts(config: config)
            self._preFeedforwardLayerNorm2.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
            self._postFeedforwardLayerNorm1.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
            self._postFeedforwardLayerNorm2.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        } else {
            self._router.wrappedValue = nil
            self._experts.wrappedValue = nil
            self._preFeedforwardLayerNorm2.wrappedValue = nil
            self._postFeedforwardLayerNorm1.wrappedValue = nil
            self._postFeedforwardLayerNorm2.wrappedValue = nil
        }
        self._inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._preFeedforwardLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postFeedforwardLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self.hasPerLayerInput = config.hiddenSizePerLayerInput > 0
        self._perLayerInputGate.wrappedValue = Linear(config.hiddenSize, max(1, config.hiddenSizePerLayerInput), bias: false)
        self._perLayerProjection.wrappedValue = Linear(max(1, config.hiddenSizePerLayerInput), config.hiddenSize, bias: false)
        self._postPerLayerInputNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._layerScalar.wrappedValue = MLXArray.ones([1])
        self.rmsNormEps = config.rmsNormEps
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        cache: Gemma4AttentionCache?,
        perLayerInput: MLXArray?,
        visionBlockIDs: MLXArray? = nil,
        attentionMask: MLXArray? = nil
    ) -> MLXArray {
        let attentionResidual = x
        var hidden = inputLayerNorm(x)
        hidden = selfAttention(
            hidden,
            cache: cache,
            visionBlockIDs: visionBlockIDs,
            attentionMask: attentionMask
        )

        let perLayerInputActive = hasPerLayerInput && perLayerInput != nil
        if let fusedOutput = fusedDecodeFeedForward(
            attentionOutput: hidden,
            attentionResidual: attentionResidual,
            sequenceLength: x.dim(1),
            perLayerInputActive: perLayerInputActive
        ) {
            return fusedOutput
        }
        if let compiled = resolvedCompiledPostSegment(
            sequenceLength: x.dim(1),
            perLayerInputActive: perLayerInputActive
        ) {
            return compiled.function([hidden, attentionResidual])[0]
        }

        hidden = postAttentionLayerNorm(hidden)
        hidden = attentionResidual + hidden

        let mlpResidual = hidden
        if let router, let experts, let preFeedforwardLayerNorm2, let postFeedforwardLayerNorm1, let postFeedforwardLayerNorm2 {
            var dense = preFeedforwardLayerNorm(hidden)
            dense = mlp(dense)
            dense = postFeedforwardLayerNorm1(dense)

            let route = router(hidden)
            var sparse = preFeedforwardLayerNorm2(hidden)
            sparse = experts(sparse, indices: route.indices, weights: route.weights)
            sparse = postFeedforwardLayerNorm2(sparse)

            hidden = dense + sparse
        } else {
            hidden = preFeedforwardLayerNorm(hidden)
            hidden = mlp(hidden)
        }
        hidden = postFeedforwardLayerNorm(hidden)
        hidden = mlpResidual + hidden

        if hasPerLayerInput, let perLayerInput {
            let gateResidual = hidden
            var gate = perLayerInputGate(hidden)
            gate = geluApproximate(gate)
            gate = gate * perLayerInput
            gate = perLayerProjection(gate)
            gate = postPerLayerInputNorm(gate)
            hidden = gateResidual + gate
        }

        return hidden * layerScalar
    }

    /// Fused-kernel decode path for everything after attention in the dense
    /// case: (post-attn norm + residual + pre-FFN norm) in one kernel, one
    /// fused gate/up matmul, (gelu·up) in one kernel, the down matmul, then
    /// (post-FFN norm + residual + layer scalar) in one kernel.
    private func fusedDecodeFeedForward(
        attentionOutput: MLXArray,
        attentionResidual: MLXArray,
        sequenceLength: Int,
        perLayerInputActive: Bool
    ) -> MLXArray? {
        guard Gemma4FusedProjectionPolicy.fusedDecodeKernelsEnabled,
              sequenceLength == 1,
              router == nil,
              experts == nil,
              !perLayerInputActive else {
            return nil
        }
        guard let gateUpFused = mlp.resolvedFusedGateUp() else { return nil }

        let hiddenSize = attentionOutput.dim(-1)
        let eps = Gemma4DecodeScalarCache.epsilon(rmsNormEps)
        let (mlpResidual, mlpInput) = Gemma4DecodeFusedKernels.residualDoubleNorm(
            attentionOutput: attentionOutput,
            residual: attentionResidual,
            postNormWeight: postAttentionLayerNorm.weight,
            preNormWeight: preFeedforwardLayerNorm.weight,
            eps: eps,
            hidden: hiddenSize
        )
        let gateUp = gateUpFused.callFused(mlpInput)
        let activated = Gemma4DecodeFusedKernels.geluMul(
            gateUp: gateUp,
            intermediate: gateUp.dim(-1) / 2
        )
        let downOutput = mlp.downProj(activated)
        return Gemma4DecodeFusedKernels.ffnResidualScale(
            downOutput: downOutput,
            mlpResidual: mlpResidual,
            postNormWeight: postFeedforwardLayerNorm.weight,
            layerScalar: layerScalar,
            eps: eps,
            hidden: hiddenSize
        )
    }

    /// Compiled decode segment covering everything after attention in the dense
    /// path: post-attention norm, residual, pre/post feed-forward norms, the MLP,
    /// the second residual, and the layer scalar. MoE and per-layer-input layers
    /// keep the interpreted path.
    private func resolvedCompiledPostSegment(
        sequenceLength: Int,
        perLayerInputActive: Bool
    ) -> Gemma4CompiledSegment? {
        guard Gemma4FusedProjectionPolicy.compiledSegmentsEnabled,
              sequenceLength == 1,
              router == nil,
              experts == nil,
              !perLayerInputActive else {
            return nil
        }
        let fingerprint = [
            ObjectIdentifier(postAttentionLayerNorm),
            ObjectIdentifier(preFeedforwardLayerNorm),
            ObjectIdentifier(postFeedforwardLayerNorm),
            ObjectIdentifier(mlp),
            ObjectIdentifier(mlp.gateProj),
            ObjectIdentifier(mlp.upProj),
            ObjectIdentifier(mlp.downProj),
            ObjectIdentifier(layerScalar),
        ]
        if let segment = compiledPostSegment {
            if segment.matches(fingerprint) { return segment }
            compiledPostSegment = nil
            compiledPostAttempted = false
        }
        if !compiledPostAttempted {
            compiledPostAttempted = true
            let postAttentionLayerNorm = self.postAttentionLayerNorm
            let preFeedforwardLayerNorm = self.preFeedforwardLayerNorm
            let postFeedforwardLayerNorm = self.postFeedforwardLayerNorm
            let mlp = self.mlp
            let layerScalar = self.layerScalar
            let function = MLX.compile { (inputs: [MLXArray]) -> [MLXArray] in
                let attentionOutput = inputs[0]
                let attentionResidual = inputs[1]
                var hidden = postAttentionLayerNorm(attentionOutput)
                hidden = attentionResidual + hidden
                let mlpResidual = hidden
                hidden = preFeedforwardLayerNorm(hidden)
                hidden = mlp(hidden)
                hidden = postFeedforwardLayerNorm(hidden)
                hidden = mlpResidual + hidden
                return [hidden * layerScalar]
            }
            compiledPostSegment = Gemma4CompiledSegment(function: function, fingerprint: fingerprint)
        }
        return compiledPostSegment
    }
}
