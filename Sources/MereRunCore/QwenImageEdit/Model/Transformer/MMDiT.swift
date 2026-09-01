import Foundation
import MLX
import MLXNN

/// Qwen's dual-stream diffusion transformer.
public final class MMDiT: Module {
    public let config: QwenImageEditTransformerConfig

    @ModuleInfo(key: "x_embedder") var xEmbedder: any DenseLayer
    @ModuleInfo(key: "t_embedder") var tEmbedder: TimestepEmbedder
    @ModuleInfo(key: "txt_norm") var textNorm: RMSNorm
    @ModuleInfo(key: "context_embedder") var contextEmbedder: (any DenseLayer)?
    @ModuleInfo(key: "transformer_blocks") var transformerBlocks: [MMDiTBlock]
    @ModuleInfo(key: "norm_out") var normOut: FinalAdaLN
    @ModuleInfo(key: "proj_out") var projOut: any DenseLayer

    let rope: MMDiTRoPE

    public let patchSize: Int
    public let inChannels: Int
    public let outChannels: Int
    public let hiddenSize: Int
    public let numLayers: Int

    public init(config: QwenImageEditTransformerConfig, factory: DenseLayerFactory = .standard) {
        self.config = config
        self.patchSize = config.patchSize
        self.inChannels = config.inChannels
        self.outChannels = config.outChannels
        self.hiddenSize = config.effectiveHiddenSize
        self.numLayers = config.numLayers

        let headDim = config.effectiveHeadDim
        let numKVHeads = config.effectiveNumKVHeads
        let normEps = config.normEps ?? 1e-6
        let mlpRatio = config.mlpRatio ?? 4
        let layerCount = config.numLayers
        self.rope = MMDiTRoPE(
            theta: config.ropeTheta ?? 10_000,
            headDim: headDim,
            axesDims: config.axesDimsRope
        )

        self._xEmbedder.wrappedValue = factory.makeDenseLayer(
            path: "x_embedder",
            inputDim: inChannels,
            outputDim: hiddenSize,
            bias: true
        )
        self._tEmbedder.wrappedValue = TimestepEmbedder(
            hiddenSize: hiddenSize,
            frequencyDim: 256,
            outputDim: hiddenSize,
            factory: factory,
            basePath: "t_embedder"
        )
        self._textNorm.wrappedValue = RMSNorm(dimensions: config.jointAttentionDim, eps: normEps)

        if config.jointAttentionDim != hiddenSize {
            self._contextEmbedder.wrappedValue = factory.makeDenseLayer(
                path: "context_embedder",
                inputDim: config.jointAttentionDim,
                outputDim: hiddenSize,
                bias: true
            )
        }

        var blocks: [MMDiTBlock] = []
        blocks.reserveCapacity(layerCount)
        for index in 0..<layerCount {
            blocks.append(MMDiTBlock(
                blockType: .joint,
                dim: hiddenSize,
                numHeads: config.numAttentionHeads,
                numKVHeads: numKVHeads,
                headDim: headDim,
                mlpRatio: mlpRatio,
                qkNorm: config.qkNorm ?? true,
                contextDim: hiddenSize,
                conditioningDim: hiddenSize,
                normEps: normEps,
                factory: factory,
                basePath: "transformer_blocks.\(index)"
            ))
        }
        self._transformerBlocks.wrappedValue = blocks
        self._normOut.wrappedValue = FinalAdaLN(
            hiddenSize: hiddenSize,
            normEps: normEps,
            factory: factory,
            basePath: "norm_out"
        )
        self._projOut.wrappedValue = factory.makeDenseLayer(
            path: "proj_out",
            inputDim: hiddenSize,
            outputDim: patchSize * patchSize * outChannels,
            bias: true
        )
        super.init()
    }

    public func callAsFunction(
        hiddenStates: MLXArray,
        timestep: MLXArray,
        contextEmbeds: MLXArray,
        contextMask: MLXArray,
        imageShapes: [(temporal: Int, height: Int, width: Int)],
        outputTokenCount: Int
    ) -> MLXArray {
        let expectedImageTokens = imageShapes.reduce(0) { partial, shape in
            partial + shape.temporal * shape.height * shape.width
        }
        precondition(hiddenStates.dim(1) == expectedImageTokens)
        precondition(outputTokenCount > 0 && outputTokenCount <= hiddenStates.dim(1))

        var imageStream = xEmbedder(hiddenStates)
        var textStream = textNorm(contextEmbeds)
        if let contextEmbedder {
            textStream = contextEmbedder(textStream)
        }

        let textConditioning = tEmbedder(timestep)
        let imageConditioning: MLXArray
        let imageTokenConditionMask: MLXArray?
        if config.zeroCondT {
            let zeroConditioning = tEmbedder(MLXArray.zeros(timestep.shape).asType(timestep.dtype))
            imageConditioning = MLX.concatenated([textConditioning, zeroConditioning], axis: 0)
            let outputMask = MLXArray.zeros([1, outputTokenCount, 1], dtype: .float32)
            let referenceMask = MLXArray.ones(
                [1, hiddenStates.dim(1) - outputTokenCount, 1],
                dtype: .float32
            )
            imageTokenConditionMask = MLX.concatenated([outputMask, referenceMask], axis: 1)
        } else {
            imageConditioning = textConditioning
            imageTokenConditionMask = nil
        }

        let frequencies = rope.frequencies(
            imageShapes: imageShapes,
            textSequenceLength: textStream.dim(1)
        )
        for block in transformerBlocks {
            (imageStream, textStream) = block.forwardJoint(
                x: imageStream,
                context: textStream,
                imageConditioning: imageConditioning,
                textConditioning: textConditioning,
                imageTokenConditionMask: imageTokenConditionMask,
                xFreqsCis: frequencies.image,
                contextFreqsCis: frequencies.text,
                attnMask: contextMask
            )
        }

        imageStream = normOut(imageStream, conditioning: textConditioning)
        return projOut(imageStream)
    }

    public func forwardEdit(
        latents: MLXArray,
        timestep: MLXArray,
        semanticEmbeds: MLXArray,
        semanticMask: MLXArray,
        appearanceLatents: [MLXArray],
        imageShapes: [(temporal: Int, height: Int, width: Int)]
    ) -> MLXArray {
        let packedReferences = appearanceLatents.map { reference in
            QwenImageEditLatentCreator.packLatents(reference)
        }
        let hiddenStates = MLX.concatenated([latents] + packedReferences, axis: 1)
        let prediction = callAsFunction(
            hiddenStates: hiddenStates,
            timestep: timestep,
            contextEmbeds: semanticEmbeds,
            contextMask: semanticMask,
            imageShapes: imageShapes,
            outputTokenCount: latents.dim(1)
        )
        return prediction[0..., 0..<latents.dim(1), 0...]
    }
}

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
            factory.makeDenseLayer(
                path: "\(path).0",
                inputDim: frequencyDim,
                outputDim: hiddenSize,
                bias: true
            ),
            factory.makeDenseLayer(
                path: "\(path).1",
                inputDim: hiddenSize,
                outputDim: self.outputDim,
                bias: true
            )
        )
        super.init()
    }

    public func callAsFunction(_ timestep: MLXArray) -> MLXArray {
        let embedding = Self.sinusoidalEmbedding(
            timestep,
            frequencyDim: frequencyDim
        )
        return mlp.1(MLXNN.silu(mlp.0(embedding)))
    }

    static func sinusoidalEmbedding(
        _ timestep: MLXArray,
        frequencyDim: Int,
        scale: Float = 1_000
    ) -> MLXArray {
        let halfDimension = frequencyDim / 2
        let exponent = -MLXArray(Float.log(10_000))
            * MLXArray(0..<halfDimension).asType(.float32)
            / MLXArray(Float(halfDimension))
        let frequencies = MLX.exp(exponent)
        let arguments = timestep.asType(.float32)[.ellipsis, .newAxis]
            * MLXArray(scale)
            * frequencies[.newAxis]
        return MLX.concatenated([MLX.cos(arguments), MLX.sin(arguments)], axis: -1)
    }
}

public final class FinalAdaLN: Module {
    public let hiddenSize: Int
    public let norm: QwenLayerNormNoAffine

    @ModuleInfo(key: "linear") var linear: any DenseLayer

    public init(
        hiddenSize: Int,
        normEps: Float = 1e-6,
        factory: DenseLayerFactory = .standard,
        basePath: String = ""
    ) {
        self.hiddenSize = hiddenSize
        self.norm = QwenLayerNormNoAffine(eps: normEps)
        let path = basePath.isEmpty ? "linear" : "\(basePath).linear"
        self._linear.wrappedValue = factory.makeDenseLayer(
            path: path,
            inputDim: hiddenSize,
            outputDim: 2 * hiddenSize,
            bias: true
        )
        super.init()
    }

    public func callAsFunction(_ input: MLXArray, conditioning: MLXArray) -> MLXArray {
        let modulation = linear(MLXNN.silu(conditioning).asType(input.dtype))
        let scale = modulation[0..., 0..<hiddenSize].expandedDimensions(axis: 1)
        let shift = modulation[0..., hiddenSize...].expandedDimensions(axis: 1)
        return norm(input) * (1 + scale) + shift
    }
}

extension MMDiT {
    public static func fromConfig(_ config: QwenImageEditTransformerConfig) -> MMDiT {
        MMDiT(config: config)
    }
}
