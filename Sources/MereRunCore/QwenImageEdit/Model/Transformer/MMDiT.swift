import Foundation
import MLX
import MLXNN

/// Multi-Modal Diffusion Transformer (MMDiT) for Qwen-Image-Edit
/// This is a unified transformer architecture where all layers perform cross-attention
/// to text encoder embeddings (unlike SD3/FLUX which have joint+single splits).
public final class MMDiT: Module {
    public let config: QwenImageEditTransformerConfig
    private let dynamicSparseRuntime: DynamicSparseAttentionRuntime?

    // Patch embedding - uses DenseLayer to support both Linear and QuantizedLinear
    @ModuleInfo(key: "x_embedder") var xEmbedder: any DenseLayer
    @ModuleInfo(key: "pos_embed") var posEmbed: MLXArray?  // Learnable if not using RoPE

    // Timestep embedding
    @ModuleInfo(key: "t_embedder") var tEmbedder: TimestepEmbedder

    // Context projection (from encoder hidden size to transformer hidden size)
    @ModuleInfo(key: "context_embedder") var contextEmbedder: (any DenseLayer)?

    // Transformer blocks (unified architecture - all blocks do cross-attention)
    @ModuleInfo(key: "transformer_blocks") var transformerBlocks: [MMDiTBlock]

    // Final normalization + modulation
    @ModuleInfo(key: "norm_out") var normOut: FinalAdaLN

    // Final projection to output channels
    @ModuleInfo(key: "proj_out") var projOut: any DenseLayer

    // RoPE
    let rope: MMDiTRoPE

    // Configuration values
    public let patchSize: Int
    public let inChannels: Int
    public let outChannels: Int
    public let hiddenSize: Int
    public let numLayers: Int

    /// Initialize MMDiT with configuration and optional layer factory for quantized models.
    /// - Parameters:
    ///   - config: Model configuration
    ///   - factory: Factory for creating dense layers (default creates standard Linear layers)
    public init(config: QwenImageEditTransformerConfig, factory: DenseLayerFactory = .standard) {
        self.config = config
        self.dynamicSparseRuntime = DynamicSparseAttentionRuntime.configured(model: .qwenImageEdit)

        // Extract configuration
        self.patchSize = config.patchSize
        self.inChannels = config.inChannels
        self.outChannels = config.outChannels
        self.hiddenSize = config.effectiveHiddenSize
        self.numLayers = config.numLayers

        let headDim = config.effectiveHeadDim
        let numKVHeads = config.effectiveNumKVHeads
        let normEps = config.normEps ?? 1e-6
        let mlpRatio = config.mlpRatio ?? 4.0

        // Patch embedding: input is already packed (in_channels=64 for packed latents)
        // Project from packed latent dim to hidden size
        self._xEmbedder.wrappedValue = factory.makeDenseLayer(
            path: "x_embedder",
            inputDim: inChannels,
            outputDim: hiddenSize,
            bias: true
        )

        // Timestep embedding - outputs full hidden size
        self._tEmbedder.wrappedValue = TimestepEmbedder(
            hiddenSize: hiddenSize,
            frequencyDim: 256,
            outputDim: hiddenSize,
            factory: factory,
            basePath: "t_embedder"
        )

        // Context projection from joint_attention_dim (3584) to hidden_size (3072)
        let contextDim = config.jointAttentionDim
        if contextDim != hiddenSize {
            self._contextEmbedder.wrappedValue = factory.makeDenseLayer(
                path: "context_embedder",
                inputDim: contextDim,
                outputDim: hiddenSize,
                bias: true
            )
        }

        // All transformer blocks do cross-attention to context
        var blocksArr: [MMDiTBlock] = []
        for i in 0..<numLayers {
            blocksArr.append(MMDiTBlock(
                blockType: .joint,
                dim: hiddenSize,
                numHeads: config.numAttentionHeads,
                numKVHeads: numKVHeads,
                headDim: headDim,
                mlpRatio: mlpRatio,
                qkNorm: config.qkNorm ?? true,
                contextDim: hiddenSize,  // After projection
                conditioningDim: hiddenSize,  // Timestep embedding output
                normEps: normEps,
                factory: factory,
                basePath: "transformer_blocks.\(i)"
            ))
        }
        self._transformerBlocks.wrappedValue = blocksArr

        // Final normalization + modulation (outputs shift+scale)
        self._normOut.wrappedValue = FinalAdaLN(
            hiddenSize: hiddenSize,
            normEps: normEps,
            factory: factory,
            basePath: "norm_out"
        )

        // Final projection to output channels (packed latents)
        self._projOut.wrappedValue = factory.makeDenseLayer(
            path: "proj_out",
            inputDim: hiddenSize,
            outputDim: inChannels,
            bias: true
        )

        // RoPE with 3D axes from config
        self.rope = MMDiTRoPE(
            theta: config.ropeTheta ?? 10000.0,
            headDim: headDim,
            axesDims: config.axesDimsRope
        )

        super.init()
    }

    func beginDenoisingStep(index: Int, count: Int) {
        dynamicSparseRuntime?.beginStep(index: index, count: count)
    }

    /// Forward pass through MMDiT
    /// - Parameters:
    ///   - latents: Noisy latent tensor [batch, channels, height, width]
    ///   - timestep: Diffusion timestep [batch]
    ///   - contextEmbeds: Text/semantic embeddings from encoder [batch, seqLen, embedDim]
    ///   - imageEmbeds: Optional appearance/reference image embeddings
    /// - Returns: Noise prediction [batch, channels, height, width]
    public func callAsFunction(
        latents: MLXArray,
        timestep: MLXArray,
        contextEmbeds: MLXArray,
        imageEmbeds: MLXArray? = nil
    ) -> MLXArray {
        let height = latents.dim(2)
        let width = latents.dim(3)

        let patchH = height / patchSize
        let patchW = width / patchSize

        // Patchify: [B, C, H, W] -> [B, numPatches, patchDim]
        let patches = patchify(latents)

        // Embed patches
        var x = xEmbedder(patches)

        // Project context if needed
        var ctx = contextEmbeds
        if let contextEmbedder {
            ctx = contextEmbedder(ctx)
        }

        // Combine with appearance embeddings if provided
        if let imageEmbeds {
            ctx = MLX.concatenated([ctx, imageEmbeds], axis: 1)
        }

        // Timestep embedding
        let tEmb = tEmbedder(timestep)

        // Compute RoPE frequencies
        let xFreqsCis = rope.getFreqs2D(height: patchH, width: patchW)
        let ctxFreqsCis = rope.getFreqs(seqLen: ctx.dim(1))

        // Process through all transformer blocks with cross-attention
        for (index, block) in transformerBlocks.enumerated() {
            (x, ctx) = block.forwardJoint(
                x: x,
                context: ctx,
                conditioning: tEmb,
                xFreqsCis: xFreqsCis,
                contextFreqsCis: ctxFreqsCis,
                attnMask: nil,
                dynamicSparseRuntime: dynamicSparseRuntime,
                layerIndex: index
            )
        }

        // Final normalization with modulation
        x = normOut(x, conditioning: tEmb)

        // Final projection to packed latent channels
        x = projOut(x)

        // Unpatchify: [B, numPatches, inChannels] -> [B, inChannels, H, W]
        let output = unpatchify(x, height: height, width: width)

        return output
    }

    /// Forward pass for image editing with dual-path conditioning
    /// - Parameters:
    ///   - latents: Noisy latent tensor [batch, channels, height, width]
    ///   - timestep: Diffusion timestep [batch]
    ///   - semanticEmbeds: Semantic embeddings from Qwen2.5-VL (image + text understanding)
    ///   - appearanceLatents: Encoded input image latents (from VAE encoder)
    /// - Returns: Noise prediction [batch, channels, height, width]
    public func forwardEdit(
        latents: MLXArray,
        timestep: MLXArray,
        semanticEmbeds: MLXArray,
        appearanceLatents: MLXArray
    ) -> MLXArray {
        // Patchify and embed the appearance latents to match hidden dimension
        let patchedAppearance = patchify(appearanceLatents)
        let embeddedAppearance = xEmbedder(patchedAppearance)

        // The appearance embeddings provide fine-grained detail that should be preserved
        // They're concatenated with semantic embeddings to guide the diffusion
        return callAsFunction(
            latents: latents,
            timestep: timestep,
            contextEmbeds: semanticEmbeds,
            imageEmbeds: embeddedAppearance
        )
    }

    // MARK: - Patchification

    private func patchify(_ x: MLXArray) -> MLXArray {
        // [B, C, H, W] -> [B, numPatches, C * patchSize * patchSize]
        let batch = x.dim(0)
        let channels = x.dim(1)
        let height = x.dim(2)
        let width = x.dim(3)

        let patchH = height / patchSize
        let patchW = width / patchSize

        // Reshape: [B, C, pH, pS, pW, pS]
        var out = x.reshaped(batch, channels, patchH, patchSize, patchW, patchSize)

        // Permute to: [B, pH, pW, C, pS, pS]
        out = out.transposed(0, 2, 4, 1, 3, 5)

        // Flatten patches: [B, pH*pW, C*pS*pS]
        out = out.reshaped(batch, patchH * patchW, channels * patchSize * patchSize)

        return out
    }

    private func unpatchify(_ x: MLXArray, height: Int, width: Int) -> MLXArray {
        // [B, numPatches, packedChannels] -> [B, latentChannels, H, W]
        // packedChannels = latentChannels * patchSize * patchSize
        let batch = x.dim(0)
        let patchH = height / patchSize
        let patchW = width / patchSize
        let packedChannels = x.dim(2)
        let latentChannels = packedChannels / (patchSize * patchSize)

        // Reshape: [B, pH, pW, latentChannels, pS, pS]
        var out = x.reshaped(batch, patchH, patchW, latentChannels, patchSize, patchSize)

        // Permute to: [B, latentChannels, pH, pS, pW, pS]
        out = out.transposed(0, 3, 1, 4, 2, 5)

        // Reshape to final: [B, latentChannels, H, W]
        out = out.reshaped(batch, latentChannels, height, width)

        return out
    }
}

// MARK: - Supporting Components

/// Timestep embedder using sinusoidal positional encoding
public final class TimestepEmbedder: Module {
    public let hiddenSize: Int
    public let frequencyDim: Int
    public let outputDim: Int

    @ModuleInfo(key: "mlp") var mlp: (any DenseLayer, any DenseLayer)

    public init(
        hiddenSize: Int,
        frequencyDim: Int = 256,
        outputDim: Int? = nil,
        factory: DenseLayerFactory = .standard,
        basePath: String = ""
    ) {
        self.hiddenSize = hiddenSize
        self.frequencyDim = frequencyDim
        self.outputDim = outputDim ?? hiddenSize

        let path = basePath.isEmpty ? "mlp" : "\(basePath).mlp"
        self._mlp.wrappedValue = (
            factory.makeDenseLayer(path: "\(path).0", inputDim: frequencyDim, outputDim: hiddenSize, bias: true),
            factory.makeDenseLayer(path: "\(path).1", inputDim: hiddenSize, outputDim: self.outputDim, bias: true)
        )
        super.init()
    }

    public func callAsFunction(_ t: MLXArray) -> MLXArray {
        // Sinusoidal embedding
        let halfDim = frequencyDim / 2
        let freqs = MLX.exp(
            -MLXArray(Float.log(10000.0)) * MLXArray(0..<halfDim).asType(.float32) / MLXArray(Float(halfDim))
        )

        let args = t.asType(.float32)[.ellipsis, .newAxis] * freqs[.newAxis]
        let embedding = MLX.concatenated([MLX.cos(args), MLX.sin(args)], axis: -1)

        // MLP projection
        let hidden = MLXNN.silu(mlp.0(embedding))
        return mlp.1(hidden)
    }
}

/// Final AdaLN layer with shift+scale modulation
/// Applied before the final projection in the transformer
/// Note: The model doesn't have a separate norm layer - the norm is implicit/combined
public final class FinalAdaLN: Module {
    public let hiddenSize: Int

    @ModuleInfo(key: "linear") var linear: any DenseLayer

    public init(
        hiddenSize: Int,
        normEps: Float = 1e-6,
        factory: DenseLayerFactory = .standard,
        basePath: String = ""
    ) {
        self.hiddenSize = hiddenSize
        // Outputs 2 * hiddenSize for (shift, scale)
        let path = basePath.isEmpty ? "linear" : "\(basePath).linear"
        self._linear.wrappedValue = factory.makeDenseLayer(
            path: path,
            inputDim: hiddenSize,
            outputDim: 2 * hiddenSize,
            bias: true
        )
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, conditioning: MLXArray) -> MLXArray {
        // Get modulation parameters from conditioning
        let mod = linear(MLXNN.silu(conditioning))

        // Split into shift and scale
        let shift = mod[0..., 0..<hiddenSize].expandedDimensions(axis: 1)
        let scale = mod[0..., hiddenSize...].expandedDimensions(axis: 1)

        // Apply: scale * x + shift (no explicit norm - may be pre-normalized)
        return scale * x + shift
    }
}

// MARK: - Factory Methods

extension MMDiT {
    /// Create MMDiT from Qwen-Image-Edit model configs
    public static func fromConfig(_ config: QwenImageEditTransformerConfig) -> MMDiT {
        MMDiT(config: config)
    }
}
