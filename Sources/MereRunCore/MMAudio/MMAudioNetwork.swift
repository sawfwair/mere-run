import Foundation
import MLX
import MLXNN
import MLXRandom

public struct MMAudioConditionFeatures {
    public let clip: MLXArray
    public let sync: MLXArray
    public let text: MLXArray

    public init(clip: MLXArray, sync: MLXArray, text: MLXArray) {
        self.clip = clip
        self.sync = sync
        self.text = text
    }
}

struct MMAudioProjectedConditions {
    var clip: MLXArray
    var sync: MLXArray
    var text: MLXArray
    var clipGlobal: MLXArray
    var textGlobal: MLXArray
}

final class MMAudioAudioInputProjection: Module {
    @ModuleInfo(key: "input") var input: MMAudioDenseOrConv1D
    @ModuleInfo(key: "mixer") var mixer: MMAudioConvMLP

    override init() {
        self._input.wrappedValue = MMAudioDenseOrConv1D(
            inputChannels: MMAudioResources.latentDimension,
            outputChannels: MMAudioResources.hiddenDimension,
            kernelSize: 7,
            padding: 3,
            bias: true
        )
        self._mixer.wrappedValue = MMAudioConvMLP(
            dimensions: MMAudioResources.hiddenDimension,
            requestedHiddenDimensions: MMAudioResources.hiddenDimension * 4,
            kernelSize: 7,
            padding: 3
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        mixer(MLXNN.silu(input(x)))
    }
}

final class MMAudioCLIPInputProjection: Module {
    @ModuleInfo(key: "input") var input: Linear
    @ModuleInfo(key: "mixer") var mixer: MMAudioConvMLP

    override init() {
        self._input.wrappedValue = Linear(
            MMAudioResources.clipDimension,
            MMAudioResources.hiddenDimension,
            bias: true
        )
        self._mixer.wrappedValue = MMAudioConvMLP(
            dimensions: MMAudioResources.hiddenDimension,
            requestedHiddenDimensions: MMAudioResources.hiddenDimension * 4,
            kernelSize: 3,
            padding: 1
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        mixer(MLXNN.silu(input(x)))
    }
}

final class MMAudioSyncInputProjection: Module {
    @ModuleInfo(key: "input") var input: MMAudioDenseOrConv1D
    @ModuleInfo(key: "mixer") var mixer: MMAudioConvMLP

    override init() {
        self._input.wrappedValue = MMAudioDenseOrConv1D(
            inputChannels: MMAudioResources.syncDimension,
            outputChannels: MMAudioResources.hiddenDimension,
            kernelSize: 7,
            padding: 3,
            bias: true
        )
        self._mixer.wrappedValue = MMAudioConvMLP(
            dimensions: MMAudioResources.hiddenDimension,
            requestedHiddenDimensions: MMAudioResources.hiddenDimension * 4,
            kernelSize: 3,
            padding: 1
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        mixer(MLXNN.silu(input(x)))
    }
}

final class MMAudioTextInputProjection: Module {
    @ModuleInfo(key: "input") var input: Linear
    @ModuleInfo(key: "mixer") var mixer: MMAudioMLP

    override init() {
        self._input.wrappedValue = Linear(
            MMAudioResources.textDimension,
            MMAudioResources.hiddenDimension,
            bias: true
        )
        self._mixer.wrappedValue = MMAudioMLP(
            dimensions: MMAudioResources.hiddenDimension,
            requestedHiddenDimensions: MMAudioResources.hiddenDimension * 4
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        mixer(MLXNN.silu(input(x)))
    }
}

public final class MMAudioNetwork: Module {
    @ModuleInfo(key: "audio_input_proj") var audioInputProjection: MMAudioAudioInputProjection
    @ModuleInfo(key: "clip_input_proj") var clipInputProjection: MMAudioCLIPInputProjection
    @ModuleInfo(key: "sync_input_proj") var syncInputProjection: MMAudioSyncInputProjection
    @ModuleInfo(key: "text_input_proj") var textInputProjection: MMAudioTextInputProjection
    @ModuleInfo(key: "clip_cond_proj") var clipConditionProjection: Linear
    @ModuleInfo(key: "text_cond_proj") var textConditionProjection: Linear
    @ModuleInfo(key: "global_cond_mlp") var globalConditionMLP: MMAudioMLP
    @ModuleInfo(key: "t_embed") var timestepEmbedder: MMAudioTimestepEmbedder
    @ModuleInfo(key: "joint_blocks") var jointBlocks: [MMAudioJointBlock]
    @ModuleInfo(key: "fused_blocks") var fusedBlocks: [MMAudioSingleBlock]
    @ModuleInfo(key: "final_layer") var finalLayer: MMAudioFinalBlock

    @ParameterInfo(key: "sync_pos_emb") var syncPositionEmbedding: MLXArray
    @ParameterInfo(key: "latent_mean") var latentMean: MLXArray
    @ParameterInfo(key: "latent_std") var latentStandardDeviation: MLXArray
    @ParameterInfo(key: "empty_string_feat") var emptyStringFeatures: MLXArray
    @ParameterInfo(key: "empty_clip_feat") var emptyCLIPFeatures: MLXArray
    @ParameterInfo(key: "empty_sync_feat") var emptySyncFeatures: MLXArray

    public override init() {
        self._audioInputProjection.wrappedValue = MMAudioAudioInputProjection()
        self._clipInputProjection.wrappedValue = MMAudioCLIPInputProjection()
        self._syncInputProjection.wrappedValue = MMAudioSyncInputProjection()
        self._textInputProjection.wrappedValue = MMAudioTextInputProjection()
        self._clipConditionProjection.wrappedValue = Linear(
            MMAudioResources.hiddenDimension,
            MMAudioResources.hiddenDimension,
            bias: true
        )
        self._textConditionProjection.wrappedValue = Linear(
            MMAudioResources.hiddenDimension,
            MMAudioResources.hiddenDimension,
            bias: true
        )
        self._globalConditionMLP.wrappedValue = MMAudioMLP(
            dimensions: MMAudioResources.hiddenDimension,
            requestedHiddenDimensions: MMAudioResources.hiddenDimension * 4
        )
        self._timestepEmbedder.wrappedValue = MMAudioTimestepEmbedder(
            dimensions: MMAudioResources.hiddenDimension
        )
        self._jointBlocks.wrappedValue = (0..<(MMAudioResources.depth - MMAudioResources.fusedDepth)).map { index in
            MMAudioJointBlock(
                dimensions: MMAudioResources.hiddenDimension,
                heads: MMAudioResources.attentionHeads,
                preOnly: index == MMAudioResources.depth - MMAudioResources.fusedDepth - 1
            )
        }
        self._fusedBlocks.wrappedValue = (0..<MMAudioResources.fusedDepth).map { _ in
            MMAudioSingleBlock(
                dimensions: MMAudioResources.hiddenDimension,
                heads: MMAudioResources.attentionHeads,
                preOnly: false,
                kernelSize: 3,
                padding: 1
            )
        }
        self._finalLayer.wrappedValue = MMAudioFinalBlock(
            dimensions: MMAudioResources.hiddenDimension,
            outputDimensions: MMAudioResources.latentDimension
        )
        self._syncPositionEmbedding.wrappedValue = MLXArray.zeros([
            1, 1, 8, MMAudioResources.syncDimension,
        ])
        self._latentMean.wrappedValue = MLXArray.zeros([1, 1, MMAudioResources.latentDimension])
        self._latentStandardDeviation.wrappedValue = MLXArray.ones([1, 1, MMAudioResources.latentDimension])
        self._emptyStringFeatures.wrappedValue = MLXArray.zeros([
            MMAudioResources.textSequenceLength, MMAudioResources.textDimension,
        ])
        self._emptyCLIPFeatures.wrappedValue = MLXArray.zeros([1, MMAudioResources.clipDimension])
        self._emptySyncFeatures.wrappedValue = MLXArray.zeros([1, MMAudioResources.syncDimension])
        super.init()
    }

    public static func load(resources: MMAudioModelResources) throws -> MMAudioNetwork {
        let model = MMAudioNetwork()
        try HFSafetensorsWeightsLoader.applyWeights(
            url: resources.networkWeightsURL,
            to: model,
            dtype: .float16,
            verify: .none,
            mapper: mapWeights
        )
        return model
    }

    public func sampleLatents(
        conditions: MMAudioConditionFeatures,
        negativeText: MLXArray? = nil,
        config: MMAudioGenerationConfig,
        progress: (@Sendable (_ completedStep: Int, _ totalSteps: Int) -> Void)? = nil
    ) -> MLXArray {
        if let seed = config.seed {
            MLXRandom.seed(seed)
        }
        let projected = preprocess(conditions, config: config)
        let empty = emptyConditions(config: config, negativeText: negativeText)
        var latent = MLXRandom.normal([
            1, config.latentSequenceLength, MMAudioResources.latentDimension,
        ]).asType(.float16)
        let stepSize = Float(1) / Float(config.steps)

        for step in 0..<config.steps {
            let timestep = MLXArray([Float(step) * stepSize]).asType(latent.dtype)
            let conditionedFlow = predictFlow(latent, timestep: timestep, conditions: projected, config: config)
            let flow: MLXArray
            if config.guidanceScale < 1 {
                flow = conditionedFlow
            } else {
                let emptyFlow = predictFlow(latent, timestep: timestep, conditions: empty, config: config)
                flow = config.guidanceScale * conditionedFlow + (1 - config.guidanceScale) * emptyFlow
            }
            latent = latent + stepSize * flow
            MLX.eval(latent)
            progress?(step + 1, config.steps)
        }
        return unnormalize(latent)
    }

    public func normalize(_ latent: MLXArray) -> MLXArray {
        (latent - latentMean) / latentStandardDeviation
    }

    public func unnormalize(_ latent: MLXArray) -> MLXArray {
        latent * latentStandardDeviation + latentMean
    }

    func preprocess(
        _ conditions: MMAudioConditionFeatures,
        config: MMAudioGenerationConfig
    ) -> MMAudioProjectedConditions {
        let batch = conditions.clip.dim(0)
        let segments = conditions.sync.dim(1) / 8
        let positionedSync = conditions.sync
            .reshaped(batch, segments, 8, MMAudioResources.syncDimension)
            + syncPositionEmbedding
        var sync = positionedSync.reshaped(batch, segments * 8, MMAudioResources.syncDimension)
        let clip = clipInputProjection(conditions.clip.asType(.float16))
        sync = syncInputProjection(sync.asType(.float16))
        let text = textInputProjection(conditions.text.asType(.float16))
        sync = MMAudioTensorOps.nearestInterpolateSequence(sync, length: config.latentSequenceLength)
        let clipGlobal = clipConditionProjection(clip.mean(axis: 1))
        let textGlobal = textConditionProjection(text.mean(axis: 1))
        MLX.eval(clip, sync, text, clipGlobal, textGlobal)
        return MMAudioProjectedConditions(
            clip: clip,
            sync: sync,
            text: text,
            clipGlobal: clipGlobal,
            textGlobal: textGlobal
        )
    }

    func predictFlow(
        _ latent: MLXArray,
        timestep: MLXArray,
        conditions: MMAudioProjectedConditions,
        config: MMAudioGenerationConfig
    ) -> MLXArray {
        let headDimensions = MMAudioResources.hiddenDimension / MMAudioResources.attentionHeads
        let latentRoPE = MMAudioRoPE.make(
            length: config.latentSequenceLength,
            dimensions: headDimensions,
            frequencyScaling: 1
        )
        let clipRoPE = MMAudioRoPE.make(
            length: config.clipSequenceLength,
            dimensions: headDimensions,
            frequencyScaling: Float(config.latentSequenceLength) / Float(config.clipSequenceLength)
        )
        var latentHidden = audioInputProjection(latent)
        var clip = conditions.clip
        var text = conditions.text
        let globalBase = globalConditionMLP(conditions.clipGlobal + conditions.textGlobal)
        let global = timestepEmbedder(timestep).expandedDimensions(axis: 1) + globalBase.expandedDimensions(axis: 1)
        let extended = global + conditions.sync

        for block in jointBlocks {
            (latentHidden, clip, text) = block(
                latent: latentHidden,
                clip: clip,
                text: text,
                globalCondition: global,
                extendedCondition: extended,
                latentRoPE: latentRoPE,
                clipRoPE: clipRoPE
            )
        }
        for block in fusedBlocks {
            latentHidden = block(latentHidden, condition: extended, rope: latentRoPE)
        }
        return finalLayer(latentHidden, condition: global)
    }

    private func emptyConditions(
        config: MMAudioGenerationConfig,
        negativeText: MLXArray?
    ) -> MMAudioProjectedConditions {
        let text = negativeText ?? emptyStringFeatures.expandedDimensions(axis: 0)
        let conditions = MMAudioConditionFeatures(
            clip: MLX.broadcast(
                emptyCLIPFeatures.expandedDimensions(axis: 0),
                to: [1, config.clipSequenceLength, MMAudioResources.clipDimension]
            ),
            sync: MLX.broadcast(
                emptySyncFeatures.expandedDimensions(axis: 0),
                to: [1, config.syncSequenceLength, MMAudioResources.syncDimension]
            ),
            text: text
        )
        return preprocess(conditions, config: config)
    }

    private static func mapWeights(key: String, value: MLXArray) -> [(String, MLXArray)] {
        if key == "t_embed.freqs" || key == "latent_rot" || key == "clip_rot" {
            return []
        }
        let mappedKey = key
            .replacingOccurrences(of: "_input_proj.0.", with: "_input_proj.input.")
            .replacingOccurrences(of: "_input_proj.2.", with: "_input_proj.mixer.")
            .replacingOccurrences(of: "t_embed.mlp.0.", with: "t_embed.mlp.first.")
            .replacingOccurrences(of: "t_embed.mlp.2.", with: "t_embed.mlp.second.")
            .replacingOccurrences(of: "adaLN_modulation.1.", with: "adaLN_modulation.linear.")
        if mappedKey.hasSuffix(".weight"), value.ndim == 3 {
            let transposed = value.transposed(0, 2, 1)
            return [(mappedKey, transposed.reshaped(-1).reshaped(transposed.shape))]
        }
        return [(mappedKey, value)]
    }
}
