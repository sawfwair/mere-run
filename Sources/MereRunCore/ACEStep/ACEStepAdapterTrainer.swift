import Foundation
import MLX
import MLXNN
import MLXOptimizers
import MLXRandom

public struct ACEStepAdapterTrainingExample: @unchecked Sendable {
    public var audio48kHz: MLXArray
    public var caption: String
    public var lyrics: String

    public init(
        audio48kHz: MLXArray,
        caption: String,
        lyrics: String = ""
    ) {
        self.audio48kHz = audio48kHz
        self.caption = caption
        self.lyrics = lyrics
    }
}

public struct ACEStepAdapterTrainingConfiguration: Sendable {
    public var kind: ACEStepAdapterKind
    public var rank: Int
    public var alpha: Float
    public var factor: Int
    public var trainingSteps: Int
    public var learningRate: Float
    public var weightDecay: Float
    public var seed: UInt64

    public init(
        kind: ACEStepAdapterKind = .lora,
        rank: Int = 8,
        alpha: Float = 16,
        factor: Int = -1,
        trainingSteps: Int = 1_000,
        learningRate: Float = 1e-4,
        weightDecay: Float = 1e-4,
        seed: UInt64 = 42
    ) {
        self.kind = kind
        self.rank = rank
        self.alpha = alpha
        self.factor = factor
        self.trainingSteps = trainingSteps
        self.learningRate = learningRate
        self.weightDecay = weightDecay
        self.seed = seed
    }
}

public struct ACEStepAdapterTrainingProgress: Sendable {
    public var step: Int
    public var totalSteps: Int
    public var loss: Float
}

public struct ACEStepAdapterTrainingReport: Codable, Sendable {
    public var kind: ACEStepAdapterKind
    public var layerCount: Int
    public var trainingSteps: Int
    public var initialLoss: Float
    public var finalLoss: Float
    public var outputSHA256: String
}

public enum ACEStepAdapterTrainingError: LocalizedError {
    case invalidConfiguration(String)
    case noExamples
    case noTrainableLayers
    case invalidAudio(String)

    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let reason):
            return "Invalid ACE-Step adapter training configuration: \(reason)"
        case .noExamples:
            return "ACE-Step adapter training requires at least one audio example."
        case .noTrainableLayers:
            return "No ACE-Step attention projections were available for adapter training."
        case .invalidAudio(let reason):
            return "Invalid ACE-Step adapter training audio: \(reason)"
        }
    }
}

extension ACEStepPipeline {
    /// Fine-tunes the resident decoder with the same flow-matching objective
    /// used by upstream ACE-Step training and writes a directly loadable
    /// PEFT LoRA or LyCORIS LoKr safetensors artifact.
    public func trainAdapter(
        examples: [ACEStepAdapterTrainingExample],
        configuration: ACEStepAdapterTrainingConfiguration,
        outputURL: URL,
        progress: (@Sendable (ACEStepAdapterTrainingProgress) -> Void)? = nil
    ) throws -> ACEStepAdapterTrainingReport {
        try ACEStepAdapterTrainer.train(
            pipeline: self,
            examples: examples,
            configuration: configuration,
            outputURL: outputURL,
            progress: progress
        )
    }
}

private final class ACEStepTrainableLoKRLinear: Linear {
    private let base: Linear
    let rank: Int
    let alphaValue: Float

    @ParameterInfo(key: "lokr_w1") var w1: MLXArray
    @ParameterInfo(key: "lokr_w2_a") var w2A: MLXArray
    @ParameterInfo(key: "lokr_w2_b") var w2B: MLXArray

    init(
        base: Linear,
        rank: Int,
        alpha: Float,
        factor: Int
    ) {
        self.base = base
        self.rank = rank
        self.alphaValue = alpha
        let (outSmall, outLarge) = ACEStepAdapterTrainer.factorization(
            base.shape.0,
            factor: factor
        )
        let (inSmall, inLarge) = ACEStepAdapterTrainer.factorization(
            base.shape.1,
            factor: factor
        )
        let w1Bound = 1 / sqrt(Float(max(1, inSmall)))
        let w2Bound = 1 / sqrt(Float(max(1, inLarge)))
        self._w1.wrappedValue = MLXRandom.uniform(
            low: -w1Bound,
            high: w1Bound,
            [outSmall, inSmall]
        ).asType(.float32)
        self._w2A.wrappedValue = MLXRandom.uniform(
            low: -w2Bound,
            high: w2Bound,
            [outLarge, rank]
        ).asType(.float32)
        self._w2B.wrappedValue = MLXArray.zeros(
            [rank, inLarge],
            dtype: .float32
        )
        super.init(weight: base.weight, bias: base.bias)
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let input = x.asType(.float32)
        let w2 = MLX.matmul(w2A, w2B)
        let prefixShape = Array(input.shape.dropLast())
        let grouped = input.reshaped(
            prefixShape + [w1.dim(1), w2.dim(1)]
        )
        let projectedW2 = MLX.matmul(grouped, w2.T)
        let projectedW1 = MLX.matmul(
            projectedW2.swappedAxes(-1, -2),
            w1.T
        )
            .swappedAxes(-1, -2)
            .reshaped(prefixShape + [w1.dim(0) * w2.dim(0)])
        let contribution = projectedW1 * MLXArray(alphaValue / Float(rank))
        return base(x) + contribution.asType(x.dtype)
    }
}

enum ACEStepAdapterTrainer {
    private struct PreparedExample {
        var cleanLatents: MLXArray
        var encoderHiddenStates: MLXArray
        var encoderAttentionMask: MLXArray
        var contextLatents: MLXArray
    }

    private static let targetSuffixes = [
        ".self_attn.q_proj",
        ".self_attn.k_proj",
        ".self_attn.v_proj",
        ".self_attn.o_proj",
        ".cross_attn.q_proj",
        ".cross_attn.k_proj",
        ".cross_attn.v_proj",
        ".cross_attn.o_proj",
    ]

    static func train(
        pipeline: ACEStepPipeline,
        examples: [ACEStepAdapterTrainingExample],
        configuration: ACEStepAdapterTrainingConfiguration,
        outputURL: URL,
        progress: (@Sendable (ACEStepAdapterTrainingProgress) -> Void)?
    ) throws -> ACEStepAdapterTrainingReport {
        try validate(configuration, examples: examples)
        MLXRandom.seed(configuration.seed)

        let layerCount: Int
        let loraLayers: [String: TrainableLoRALayer]
        let lokrLayers: [String: ACEStepTrainableLoKRLinear]
        switch configuration.kind {
        case .lora:
            loraLayers = try injectLoRA(
                into: pipeline.decoder,
                rank: configuration.rank,
                alpha: configuration.alpha
            )
            lokrLayers = [:]
            layerCount = loraLayers.count
        case .lokr:
            loraLayers = [:]
            lokrLayers = try injectLoKR(
                into: pipeline.decoder,
                rank: configuration.rank,
                alpha: configuration.alpha,
                factor: configuration.factor
            )
            layerCount = lokrLayers.count
        case .auto:
            throw ACEStepAdapterTrainingError.invalidConfiguration(
                "training kind must be lora or lokr, not auto."
            )
        }
        guard layerCount > 0 else {
            throw ACEStepAdapterTrainingError.noTrainableLayers
        }

        try pipeline.decoder.freeze(
            recursive: true,
            keys: nil,
            strict: false
        )
        for layer in loraLayers.values {
            guard let module = layer as? Module else { continue }
            try module.unfreeze(
                recursive: false,
                keys: ["loraDown", "loraUp"],
                strict: true
            )
        }
        for layer in lokrLayers.values {
            try layer.unfreeze(
                recursive: false,
                keys: ["lokr_w1", "lokr_w2_a", "lokr_w2_b"],
                strict: true
            )
        }
        guard !pipeline.decoder.trainableParameters().flattened().isEmpty else {
            throw ACEStepAdapterTrainingError.noTrainableLayers
        }

        let prepared = try examples.map {
            try prepare($0, pipeline: pipeline)
        }
        let optimizer = AdamW(
            learningRate: configuration.learningRate,
            weightDecay: configuration.weightDecay,
            biasCorrection: true
        )
        let lossAndGrad = valueAndGrad(model: pipeline.decoder) {
            model,
            arrays in
            let clean = arrays[0].asType(.float32)
            let noise = arrays[1].asType(.float32)
            let condition = arrays[2]
            let conditionMask = arrays[3]
            let context = arrays[4]
            let timestep = arrays[5].asType(.float32)
            let sample = flowMatchingSample(
                clean: clean,
                noise: noise,
                timestep: timestep
            )
            let prediction = model(
                hiddenStates: sample.noisy.asType(clean.dtype),
                timestep: timestep,
                timestepR: timestep,
                encoderHiddenStates: condition,
                encoderAttentionMask: conditionMask,
                contextLatents: context
            ).asType(.float32)
            return [(prediction - sample.target).square().mean()]
        }

        var initialLoss: Float?
        var finalLoss: Float = .nan
        let schedule = pipeline.checkpointVariant.isTurbo
            ? ACEStepTurboScheduler(fixNFE: 8, shift: 3).timesteps
            : ACEStepContinuousScheduler(
                inferenceSteps: max(8, min(100, configuration.trainingSteps)),
                shift: 1
            ).timesteps

        for step in 0..<configuration.trainingSteps {
            let example = prepared[step % prepared.count]
            let noise = MLXRandom.normal(
                example.cleanLatents.shape,
                key: MLXRandom.key(
                    configuration.seed &+ UInt64(step) &+ 1
                )
            ).asType(.float32)
            let timestepValue = schedule[step % schedule.count]
            let timestep = MLXArray([timestepValue]).asType(.float32)
            let (values, gradients) = lossAndGrad(
                pipeline.decoder,
                [
                    example.cleanLatents,
                    noise,
                    example.encoderHiddenStates,
                    example.encoderAttentionMask,
                    example.contextLatents,
                    timestep,
                ]
            )
            optimizer.update(model: pipeline.decoder, gradients: gradients)
            MLX.eval(values[0], pipeline.decoder, optimizer)
            let loss = values[0].item(Float.self)
            initialLoss = initialLoss ?? loss
            finalLoss = loss
            progress?(
                .init(
                    step: step + 1,
                    totalSteps: configuration.trainingSteps,
                    loss: loss
                )
            )
        }

        try FileManager.default.createDirectory(
            at: outputURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        switch configuration.kind {
        case .lora:
            try LoRASafetensorsWriter.save(
                loraLayers: loraLayers,
                to: outputURL,
                metadata: trainingMetadata(configuration)
            )
        case .lokr:
            try saveLoKR(
                lokrLayers,
                configuration: configuration,
                to: outputURL
            )
        case .auto:
            preconditionFailure("Adapter training kind was validated.")
        }
        return try ACEStepAdapterTrainingReport(
            kind: configuration.kind,
            layerCount: layerCount,
            trainingSteps: configuration.trainingSteps,
            initialLoss: initialLoss ?? .nan,
            finalLoss: finalLoss,
            outputSHA256: ModelArtifactPin.fileSHA256(outputURL)
        )
    }

    static func factorization(
        _ dimension: Int,
        factor requestedFactor: Int
    ) -> (Int, Int) {
        if requestedFactor > 0, dimension.isMultiple(of: requestedFactor) {
            return (
                min(requestedFactor, dimension / requestedFactor),
                max(requestedFactor, dimension / requestedFactor)
            )
        }
        let limit = requestedFactor < 0 ? dimension : requestedFactor
        var best = (1, dimension)
        var candidate = 1
        while candidate * candidate <= dimension {
            if dimension.isMultiple(of: candidate),
               candidate <= limit {
                best = (candidate, dimension / candidate)
            }
            candidate += 1
        }
        return best
    }

    static func flowMatchingSample(
        clean: MLXArray,
        noise: MLXArray,
        timestep: MLXArray
    ) -> (noisy: MLXArray, target: MLXArray) {
        let expandedTimestep = timestep.reshaped(
            timestep.dim(0),
            1,
            1
        )
        return (
            noisy: expandedTimestep * noise
                + (MLXArray(Float(1)) - expandedTimestep) * clean,
            target: noise - clean
        )
    }

    private static func validate(
        _ configuration: ACEStepAdapterTrainingConfiguration,
        examples: [ACEStepAdapterTrainingExample]
    ) throws {
        guard !examples.isEmpty else {
            throw ACEStepAdapterTrainingError.noExamples
        }
        guard configuration.rank > 0 else {
            throw ACEStepAdapterTrainingError.invalidConfiguration(
                "rank must be greater than zero."
            )
        }
        guard configuration.alpha.isFinite, configuration.alpha > 0 else {
            throw ACEStepAdapterTrainingError.invalidConfiguration(
                "alpha must be finite and greater than zero."
            )
        }
        guard configuration.trainingSteps > 0 else {
            throw ACEStepAdapterTrainingError.invalidConfiguration(
                "trainingSteps must be greater than zero."
            )
        }
        guard configuration.learningRate.isFinite,
              configuration.learningRate > 0 else {
            throw ACEStepAdapterTrainingError.invalidConfiguration(
                "learningRate must be finite and greater than zero."
            )
        }
    }

    private static func prepare(
        _ example: ACEStepAdapterTrainingExample,
        pipeline: ACEStepPipeline
    ) throws -> PreparedExample {
        guard example.audio48kHz.ndim == 3,
              example.audio48kHz.dim(0) == 1,
              example.audio48kHz.dim(1) > 0,
              example.audio48kHz.dim(2) == pipeline.vaeConfig.audioChannels else {
            throw ACEStepAdapterTrainingError.invalidAudio(
                "expected [1, samples, \(pipeline.vaeConfig.audioChannels)]."
            )
        }
        let clean = pipeline.vae.tiledEncode(
            example.audio48kHz.asType(.float32),
            sample: false
        ).asType(.float32)
        let source = pipeline.defaultSourceLatents(
            targetFrames: clean.dim(1)
        ).asType(.float32)
        let metadata = ACEStep5HzLMConstrainedSampler.UserMetadata(
            caption: example.caption,
            duration: String(
                Int(
                    (Float(example.audio48kHz.dim(1)) / 48_000)
                        .rounded()
                )
            )
        )
        let inputs = try pipeline.preparePromptConditionInputs(
            caption: example.caption,
            lyrics: example.lyrics,
            srcLatents: source,
            chunkChannels: pipeline.chunkChannelsForPromptConditioning(),
            lmUserMetadata: metadata,
            vocalLanguage: "en",
            instruction: ACEStepTask.textToMusic.instruction(),
            task: .textToMusic
        )
        let prepared = pipeline.prepareCondition(
            textHiddenStates: inputs.textHiddenStates,
            textAttentionMask: inputs.textAttentionMask,
            lyricHiddenStates: inputs.lyricHiddenStates,
            lyricAttentionMask: inputs.lyricAttentionMask,
            referAudioAcousticHiddenStatesPacked:
                inputs.referAudioAcousticHiddenStatesPacked,
            referAudioOrderMask: inputs.referAudioOrderMask,
            hiddenStates: inputs.hiddenStates ?? source,
            attentionMask: inputs.attentionMask
                ?? MLXArray.ones([1, clean.dim(1)], dtype: .int32),
            silenceLatent: inputs.silenceLatent,
            srcLatents: inputs.srcLatents,
            chunkMasks: inputs.chunkMasks,
            isCovers: inputs.isCovers
        )
        MLX.eval(
            clean,
            prepared.encoderHiddenStates,
            prepared.encoderAttentionMask,
            prepared.contextLatents
        )
        return PreparedExample(
            cleanLatents: clean,
            encoderHiddenStates: prepared.encoderHiddenStates,
            encoderAttentionMask: prepared.encoderAttentionMask,
            contextLatents: prepared.contextLatents
        )
    }

    private static func injectLoRA(
        into decoder: Module,
        rank: Int,
        alpha: Float
    ) throws -> [String: TrainableLoRALayer] {
        var replacements: [String: Module] = [:]
        var layers: [String: TrainableLoRALayer] = [:]
        for (path, module) in decoder.namedModules() {
            guard targetSuffixes.contains(where: { path.hasSuffix($0) }),
                  let linear = module as? Linear else {
                continue
            }
            let wrapper = LoRALinear(
                base: linear,
                rank: rank,
                alpha: alpha,
                zeroInitUp: true
            )
            replacements[path] = wrapper
            layers[path] = wrapper
        }
        apply(replacements, to: decoder)
        return layers
    }

    private static func injectLoKR(
        into decoder: Module,
        rank: Int,
        alpha: Float,
        factor: Int
    ) throws -> [String: ACEStepTrainableLoKRLinear] {
        var replacements: [String: Module] = [:]
        var layers: [String: ACEStepTrainableLoKRLinear] = [:]
        for (path, module) in decoder.namedModules() {
            guard targetSuffixes.contains(where: { path.hasSuffix($0) }),
                  let linear = module as? Linear else {
                continue
            }
            let wrapper = ACEStepTrainableLoKRLinear(
                base: linear,
                rank: rank,
                alpha: alpha,
                factor: factor
            )
            replacements[path] = wrapper
            layers[path] = wrapper
        }
        apply(replacements, to: decoder)
        return layers
    }

    private static func apply(
        _ replacements: [String: Module],
        to decoder: Module
    ) {
        decoder.update(
            modules: ModuleChildren.unflattened(
                replacements.map { ($0.key, $0.value) }
            )
        )
    }

    private static func saveLoKR(
        _ layers: [String: ACEStepTrainableLoKRLinear],
        configuration: ACEStepAdapterTrainingConfiguration,
        to url: URL
    ) throws {
        var arrays: [String: MLXArray] = [:]
        for (path, layer) in layers {
            let key = "lycoris_" + path.replacingOccurrences(of: ".", with: "_")
            arrays["\(key).lokr_w1"] = layer.w1.asType(.float16)
            arrays["\(key).lokr_w2_a"] = layer.w2A.asType(.float16)
            arrays["\(key).lokr_w2_b"] = layer.w2B.asType(.float16)
            arrays["\(key).alpha"] = MLXArray(configuration.alpha)
        }
        MLX.eval(Array(arrays.values))
        var metadata = trainingMetadata(configuration)
        metadata["algo"] = "lokr"
        metadata["format"] = "lycoris"
        metadata["linear_dim"] = String(configuration.rank)
        metadata["linear_alpha"] = String(configuration.alpha)
        try MLX.save(arrays: arrays, metadata: metadata, url: url)
    }

    private static func trainingMetadata(
        _ configuration: ACEStepAdapterTrainingConfiguration
    ) -> [String: String] {
        [
            "base_model": "ACE-Step-1.5",
            "format": configuration.kind == .lora ? "peft" : "lycoris",
            "lora_alpha": String(configuration.alpha),
            "lora_rank": String(configuration.rank),
            "mere_run_adapter_kind": configuration.kind.rawValue,
            "mere_run_training_objective": "ace_step_flow_matching",
            "training_steps": String(configuration.trainingSteps),
        ]
    }
}
