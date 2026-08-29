import Foundation
import MLX
import MLXNN

/// MMDiT Transformer Block
/// Supports both joint attention (text + image) and single-stream modes.
public final class MMDiTBlock: Module {
    public enum BlockType: Sendable {
        case joint    // Processes both text and image streams with joint attention
        case single   // Processes only image stream (used after joint blocks)
    }

    public let blockType: BlockType
    public let dim: Int
    public let numHeads: Int
    public let numKVHeads: Int
    public let headDim: Int
    public let mlpRatio: Float

    // Layer norms for image stream
    @ModuleInfo(key: "norm1") var norm1: QwenLayerNormNoAffine
    @ModuleInfo(key: "norm2") var norm2: QwenLayerNormNoAffine

    // Layer norms for context stream (joint blocks only)
    @ModuleInfo(key: "norm1_context") var norm1Context: QwenLayerNormNoAffine?
    @ModuleInfo(key: "norm2_context") var norm2Context: QwenLayerNormNoAffine?

    // Attention
    @ModuleInfo(key: "attn") var attn: MMDiTAttention

    // Feed-forward for image stream (GELU, not SwiGLU)
    @ModuleInfo(key: "ff") var ff: MMDiTFeedForwardGELU

    // Feed-forward for context stream (joint blocks only)
    @ModuleInfo(key: "ff_context") var ffContext: MMDiTFeedForwardGELU?

    // AdaLN modulation
    @ModuleInfo(key: "adaLN_modulation") var adaLN: AdaLNZero?

    // Context AdaLN (joint blocks only)
    @ModuleInfo(key: "adaLN_modulation_context") var adaLNContext: AdaLNZero?

    public init(
        blockType: BlockType,
        dim: Int,
        numHeads: Int,
        numKVHeads: Int? = nil,
        headDim: Int? = nil,
        mlpRatio: Float = 4.0,
        qkNorm: Bool = true,
        contextDim: Int? = nil,
        conditioningDim: Int? = nil,
        normEps: Float = 1e-6,
        factory: DenseLayerFactory = .standard,
        basePath: String = ""
    ) {
        self.blockType = blockType
        self.dim = dim
        self.numHeads = numHeads
        self.numKVHeads = numKVHeads ?? numHeads
        self.headDim = headDim ?? (dim / numHeads)
        self.mlpRatio = mlpRatio

        let mlpHiddenDim = Int(Float(dim) * mlpRatio)
        let ctxDim = contextDim ?? dim

        // Image stream norms
        self._norm1.wrappedValue = QwenLayerNormNoAffine(eps: normEps)
        self._norm2.wrappedValue = QwenLayerNormNoAffine(eps: normEps)

        // Image stream MLP (GELU)
        let ffPath = basePath.isEmpty ? "ff" : "\(basePath).ff"
        self._ff.wrappedValue = MMDiTFeedForwardGELU(
            dim: dim,
            hiddenDim: mlpHiddenDim,
            factory: factory,
            basePath: ffPath
        )

        // AdaLN for image stream
        let adaLNPath = basePath.isEmpty ? "adaLN_modulation" : "\(basePath).adaLN_modulation"
        self._adaLN.wrappedValue = AdaLNZero(
            hiddenSize: dim,
            conditioningDim: conditioningDim,
            factory: factory,
            basePath: adaLNPath
        )

        switch blockType {
        case .joint:
            // Context stream components
            self._norm1Context.wrappedValue = QwenLayerNormNoAffine(eps: normEps)
            self._norm2Context.wrappedValue = QwenLayerNormNoAffine(eps: normEps)

            let ctxMlpHidden = Int(Float(ctxDim) * mlpRatio)
            let ffCtxPath = basePath.isEmpty ? "ff_context" : "\(basePath).ff_context"
            self._ffContext.wrappedValue = MMDiTFeedForwardGELU(
                dim: ctxDim,
                hiddenDim: ctxMlpHidden,
                factory: factory,
                basePath: ffCtxPath
            )

            // Context AdaLN (optional - some models use shared conditioning)
            let adaLNCtxPath = basePath.isEmpty ? "adaLN_modulation_context" : "\(basePath).adaLN_modulation_context"
            self._adaLNContext.wrappedValue = AdaLNZero(
                hiddenSize: ctxDim,
                conditioningDim: conditioningDim,
                factory: factory,
                basePath: adaLNCtxPath
            )

            // Joint attention with context projections
            let attnPath = basePath.isEmpty ? "attn" : "\(basePath).attn"
            self._attn.wrappedValue = MMDiTAttention(
                dim: dim,
                numHeads: numHeads,
                numKVHeads: self.numKVHeads,
                headDim: self.headDim,
                qkNorm: qkNorm,
                hasContextProjection: true,
                contextDim: ctxDim,
                normEps: normEps,
                factory: factory,
                basePath: attnPath
            )

        case .single:
            // Single-stream attention
            let attnPath = basePath.isEmpty ? "attn" : "\(basePath).attn"
            self._attn.wrappedValue = MMDiTAttention(
                dim: dim,
                numHeads: numHeads,
                numKVHeads: self.numKVHeads,
                headDim: self.headDim,
                qkNorm: qkNorm,
                hasContextProjection: false,
                normEps: normEps,
                factory: factory,
                basePath: attnPath
            )
        }

        super.init()
    }

    /// Process joint block (text + image streams)
    /// - Parameters:
    ///   - x: Image/latent stream [batch, imgSeqLen, dim]
    ///   - context: Text/context stream [batch, ctxSeqLen, contextDim]
    ///   - conditioning: Timestep conditioning [batch, condDim]
    ///   - xFreqsCis: RoPE frequencies for image
    ///   - contextFreqsCis: RoPE frequencies for context
    ///   - attnMask: Optional attention mask
    /// - Returns: (imageOut, contextOut) tuple
    public func forwardJoint(
        x: MLXArray,
        context: MLXArray,
        imageConditioning: MLXArray,
        textConditioning: MLXArray,
        imageTokenConditionMask: MLXArray? = nil,
        xFreqsCis: MLXArray? = nil,
        contextFreqsCis: MLXArray? = nil,
        attnMask: MLXArray? = nil
    ) -> (image: MLXArray, context: MLXArray) {
        guard blockType == .joint else {
            fatalError("forwardJoint called on single block")
        }

        guard let norm1Context, let norm2Context, let ffContext else {
            fatalError("Missing context components for joint block")
        }

        // Get modulation parameters
        let imgMod = adaLN!(
            imageConditioning,
            tokenConditionMask: imageTokenConditionMask
        )
        let ctxMod: AdaLNModulation
        if let adaLNContext {
            ctxMod = adaLNContext(textConditioning)
        } else {
            ctxMod = imgMod  // Share modulation if no separate context AdaLN
        }

        // Pre-attention normalization with modulation
        let xNormed = imgMod.modulatePreAttn(norm1(x))
        let ctxNormed = ctxMod.modulatePreAttn(norm1Context(context))

        // Joint attention
        let (attnImg, attnCtx) = attn.jointAttention(
            x: xNormed,
            context: ctxNormed,
            xFreqsCis: xFreqsCis,
            contextFreqsCis: contextFreqsCis,
            attnMask: attnMask
        )

        // Residual with gate
        var xOut = x + imgMod.gateAttn(attnImg)
        var ctxOut = context + ctxMod.gateAttn(attnCtx)

        // MLP with modulation
        let xMlpIn = imgMod.modulatePreMLP(norm2(xOut))
        let ctxMlpIn = ctxMod.modulatePreMLP(norm2Context(ctxOut))

        xOut = xOut + imgMod.gateMLP(ff(xMlpIn))
        ctxOut = ctxOut + ctxMod.gateMLP(ffContext(ctxMlpIn))

        return (xOut, ctxOut)
    }

    /// Process single block (image stream only)
    /// - Parameters:
    ///   - x: Image/latent stream [batch, seqLen, dim]
    ///   - conditioning: Timestep conditioning [batch, condDim]
    ///   - freqsCis: RoPE frequencies
    ///   - attnMask: Optional attention mask
    /// - Returns: Output tensor
    public func forwardSingle(
        x: MLXArray,
        conditioning: MLXArray,
        freqsCis: MLXArray? = nil,
        attnMask: MLXArray? = nil
    ) -> MLXArray {
        guard blockType == .single else {
            fatalError("forwardSingle called on joint block")
        }

        // Get modulation
        let mod = adaLN!(conditioning)

        // Pre-attention
        let xNormed = mod.modulatePreAttn(norm1(x))
        let attnOut = attn(xNormed, freqsCis: freqsCis, attnMask: attnMask)
        var out = x + mod.gateAttn(attnOut)

        // MLP
        let mlpIn = mod.modulatePreMLP(norm2(out))
        out = out + mod.gateMLP(ff(mlpIn))

        return out
    }
}

/// Feed-forward network for MMDiT (SwiGLU variant)
public final class MMDiTFeedForward: Module {
    public let dim: Int
    public let hiddenDim: Int

    @ModuleInfo(key: "linear1") var linear1: any DenseLayer
    @ModuleInfo(key: "linear2") var linear2: any DenseLayer
    @ModuleInfo(key: "linear_gate") var linearGate: any DenseLayer

    public init(
        dim: Int,
        hiddenDim: Int,
        factory: DenseLayerFactory = .standard,
        basePath: String = ""
    ) {
        self.dim = dim
        self.hiddenDim = hiddenDim
        let path = basePath.isEmpty ? "" : "\(basePath)."
        self._linear1.wrappedValue = factory.makeDenseLayer(path: "\(path)linear1", inputDim: dim, outputDim: hiddenDim, bias: true)
        self._linear2.wrappedValue = factory.makeDenseLayer(path: "\(path)linear2", inputDim: hiddenDim, outputDim: dim, bias: true)
        self._linearGate.wrappedValue = factory.makeDenseLayer(path: "\(path)linear_gate", inputDim: dim, outputDim: hiddenDim, bias: true)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // SwiGLU: out = W2(SiLU(W1(x)) * W_gate(x))
        linear2(MLXNN.silu(linear1(x)) * linearGate(x))
    }
}

/// GELU feed-forward variant (used in some models)
public final class MMDiTFeedForwardGELU: Module {
    public let dim: Int
    public let hiddenDim: Int

    @ModuleInfo(key: "linear1") var linear1: any DenseLayer
    @ModuleInfo(key: "linear2") var linear2: any DenseLayer

    public init(
        dim: Int,
        hiddenDim: Int,
        factory: DenseLayerFactory = .standard,
        basePath: String = ""
    ) {
        self.dim = dim
        self.hiddenDim = hiddenDim
        let path = basePath.isEmpty ? "" : "\(basePath)."
        self._linear1.wrappedValue = factory.makeDenseLayer(path: "\(path)linear1", inputDim: dim, outputDim: hiddenDim, bias: true)
        self._linear2.wrappedValue = factory.makeDenseLayer(path: "\(path)linear2", inputDim: hiddenDim, outputDim: dim, bias: true)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(MLXNN.geluApproximate(linear1(x)))
    }
}

public final class QwenLayerNormNoAffine: Module {
    public let eps: Float

    public init(eps: Float = 1e-6) {
        self.eps = eps
        super.init()
    }

    public func callAsFunction(_ input: MLXArray) -> MLXArray {
        let compute = input.asType(.float32)
        let mean = MLX.mean(compute, axis: -1, keepDims: true)
        let centered = compute - mean
        let variance = MLX.mean(centered * centered, axis: -1, keepDims: true)
        return (centered * MLX.rsqrt(variance + eps)).asType(input.dtype)
    }
}
