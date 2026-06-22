import Foundation
import MLX
import MLXNN
import MLXRandom

public struct WooshGenerationResult: Sendable, Hashable {
    public let samples: [Float]
    public let sampleRate: Int

    public init(samples: [Float], sampleRate: Int = WooshResources.sampleRate) {
        self.samples = samples
        self.sampleRate = sampleRate
    }
}

public final class WooshGenerator {
    private let resources: WooshModelResources
    private let textConditioner: WooshTextConditioner
    private let autoencoder: WooshAudioAutoEncoder
    private let dflow: WooshFlowMapDiT?
    private let flow: WooshLatentDiT?
    private let vflow: WooshVideoLatentModel?
    private let dvflow: WooshVideoFlowMapModel?

    public init(resources: WooshModelResources) throws {
        self.resources = resources
        self.textConditioner = try WooshTextConditioner.load(resources: resources)
        self.autoencoder = try WooshAudioAutoEncoder.load(resources: resources)

        switch resources.variant {
        case .dflow:
            let model = WooshFlowMapDiT()
            try HFSafetensorsWeightsLoader.applyWeights(
                url: resources.generatorWeightsURL,
                to: model,
                dtype: .float32,
                verify: .none,
                mapper: Self.mapDiTWeights
            )
            self.dflow = model
            self.flow = nil
            self.vflow = nil
            self.dvflow = nil
        case .flow:
            let model = WooshLatentDiT()
            try HFSafetensorsWeightsLoader.applyWeights(
                url: resources.generatorWeightsURL,
                to: model,
                dtype: .float32,
                verify: .none,
                mapper: Self.mapDiTWeights
            )
            self.dflow = nil
            self.flow = model
            self.vflow = nil
            self.dvflow = nil
        case .vflow8s:
            let model = WooshVideoLatentModel()
            try HFSafetensorsWeightsLoader.applyWeights(
                url: resources.generatorWeightsURL,
                to: model,
                dtype: .float32,
                verify: .none,
                mapper: Self.mapVideoDiTWeights
            )
            self.dflow = nil
            self.flow = nil
            self.vflow = model
            self.dvflow = nil
        case .dvflow8s:
            let model = WooshVideoFlowMapModel()
            try HFSafetensorsWeightsLoader.applyWeights(
                url: resources.generatorWeightsURL,
                to: model,
                dtype: .float32,
                verify: .none,
                mapper: Self.mapVideoDiTWeights
            )
            self.dflow = nil
            self.flow = nil
            self.vflow = nil
            self.dvflow = model
        }
    }

    public func generate(
        prompt: String,
        config: WooshDenoiseConfig,
        progress: (@Sendable (_ completedStep: Int, _ totalSteps: Int) -> Void)? = nil
    ) throws -> WooshGenerationResult {
        guard config.steps > 0 else {
            throw WooshError.invalidRenoiseSchedule(expected: 1, actual: 0)
        }
        switch resources.variant {
        case .dflow:
            guard let dflow else {
                throw WooshError.unsupportedFlowSampler
            }
            return try generateDFlow(prompt: prompt, config: config, dflow: dflow, progress: progress)
        case .flow:
            guard let flow else {
                throw WooshError.unsupportedFlowSampler
            }
            return try generateFlow(prompt: prompt, config: config, flow: flow, progress: progress)
        case .vflow8s, .dvflow8s:
            throw WooshError.unsupportedFlowSampler
        }
    }

    public func generateVideo(
        prompt: String,
        videoFeatures: MLXArray,
        config: WooshDenoiseConfig,
        progress: (@Sendable (_ completedStep: Int, _ totalSteps: Int) -> Void)? = nil
    ) throws -> WooshGenerationResult {
        guard config.steps > 0 else {
            throw WooshError.invalidRenoiseSchedule(expected: 1, actual: 0)
        }
        guard videoFeatures.ndim == 3, videoFeatures.dim(0) == 1, videoFeatures.dim(2) == 768 else {
            throw WooshError.invalidVideoFeatureShape(videoFeatures.shape)
        }
        switch resources.variant {
        case .dvflow8s:
            guard let dvflow else {
                throw WooshError.unsupportedFlowSampler
            }
            return try generateDVFlow(
                prompt: prompt,
                videoFeatures: videoFeatures,
                config: config,
                dvflow: dvflow,
                progress: progress
            )
        case .vflow8s:
            guard let vflow else {
                throw WooshError.unsupportedFlowSampler
            }
            return try generateVFlow(
                prompt: prompt,
                videoFeatures: videoFeatures,
                config: config,
                vflow: vflow,
                progress: progress
            )
        case .dflow, .flow:
            throw WooshError.unsupportedFlowSampler
        }
    }

    private func generateDFlow(
        prompt: String,
        config: WooshDenoiseConfig,
        dflow: WooshFlowMapDiT,
        progress: (@Sendable (_ completedStep: Int, _ totalSteps: Int) -> Void)?
    ) throws -> WooshGenerationResult {
        let renoise = resolvedRenoiseSchedule(config: config)
        if let seed = config.seed {
            MLXRandom.seed(seed)
        }

        var latents = MLXRandom.normal([
            1,
            WooshResources.latentChannels,
            config.latentFrames(),
        ]).asType(.float32)
        let text = textConditioner.encode(prompts: [prompt])
        let condition = WooshCondition(
            crossAttention: text.embeddings.asType(.float32),
            crossAttentionMask: text.attentionMask,
            cfg: MLXArray([config.guidanceScale]).asType(.float32)
        )

        for step in 0..<config.steps {
            let rawT = 1 - Float(step) / Float(config.steps)
            let rValue = 1 - Float(step + 1) / Float(config.steps)
            let renoised = applyRenoise(latents, t: rawT, r: rValue, strength: renoise[step])
            latents = renoised.latents
            let t = MLXArray([renoised.t]).asType(.float32)
            let r = MLXArray([rValue]).asType(.float32)
            let velocity = dflow(x: latents, t: t, r: r, cond: condition)
            latents = latents - (t - r).reshaped(1, 1, 1) * velocity
            MLX.eval(latents)
            progress?(step + 1, config.steps)
        }

        return WooshGenerationResult(samples: try autoencoder.decode(latents))
    }

    private func generateFlow(
        prompt: String,
        config: WooshDenoiseConfig,
        flow: WooshLatentDiT,
        progress: (@Sendable (_ completedStep: Int, _ totalSteps: Int) -> Void)?
    ) throws -> WooshGenerationResult {
        if let seed = config.seed {
            MLXRandom.seed(seed)
        }

        var latents = MLXRandom.normal([
            1,
            WooshResources.latentChannels,
            config.latentFrames(),
        ]).asType(.float32)
        let positiveText = textConditioner.encode(prompts: [prompt])
        let negativeText = textConditioner.encode(prompts: [""])
        let positive = WooshCondition(
            crossAttention: positiveText.embeddings.asType(.float32),
            crossAttentionMask: positiveText.attentionMask
        )
        let negative = WooshCondition(
            crossAttention: negativeText.embeddings.asType(.float32),
            crossAttentionMask: negativeText.attentionMask
        )
        let stepSize = MLXArray(1 / Float(config.steps)).asType(.float32)
        let guidance = MLXArray(config.guidanceScale).asType(.float32)

        for step in 0..<config.steps {
            let tValue = 1 - Float(step) / Float(config.steps)
            let t = MLXArray([tValue]).asType(.float32)
            let condVelocity = flow(x: latents, t: t, cond: positive)
            let uncondVelocity = flow(x: latents, t: t, cond: negative)
            let velocity = condVelocity + guidance * (condVelocity - uncondVelocity)
            latents = latents + stepSize * velocity
            MLX.eval(latents)
            progress?(step + 1, config.steps)
        }

        return WooshGenerationResult(samples: try autoencoder.decode(latents))
    }

    private func generateDVFlow(
        prompt: String,
        videoFeatures: MLXArray,
        config: WooshDenoiseConfig,
        dvflow: WooshVideoFlowMapModel,
        progress: (@Sendable (_ completedStep: Int, _ totalSteps: Int) -> Void)?
    ) throws -> WooshGenerationResult {
        let renoise = resolvedRenoiseSchedule(config: config)
        if let seed = config.seed {
            MLXRandom.seed(seed)
        }

        var latents = MLXRandom.normal([
            1,
            WooshResources.latentChannels,
            config.latentFrames(),
        ]).asType(.float32)
        let text = textConditioner.encode(prompts: [prompt])
        let condition = WooshCondition(
            crossAttention: text.embeddings.asType(.float32),
            crossAttentionMask: text.attentionMask,
            cfg: MLXArray([config.guidanceScale]).asType(.float32),
            videoFeatures: videoFeatures.asType(.float32)
        )

        for step in 0..<config.steps {
            let rawT = 1 - Float(step) / Float(config.steps)
            let rValue = 1 - Float(step + 1) / Float(config.steps)
            let renoised = applyRenoise(latents, t: rawT, r: rValue, strength: renoise[step])
            latents = renoised.latents
            let t = MLXArray([renoised.t]).asType(.float32)
            let r = MLXArray([rValue]).asType(.float32)
            let velocity = try dvflow(x: latents, t: t, r: r, cond: condition)
            latents = latents - (t - r).reshaped(1, 1, 1) * velocity
            MLX.eval(latents)
            progress?(step + 1, config.steps)
        }

        return WooshGenerationResult(samples: try autoencoder.decode(latents))
    }

    private func generateVFlow(
        prompt: String,
        videoFeatures: MLXArray,
        config: WooshDenoiseConfig,
        vflow: WooshVideoLatentModel,
        progress: (@Sendable (_ completedStep: Int, _ totalSteps: Int) -> Void)?
    ) throws -> WooshGenerationResult {
        if let seed = config.seed {
            MLXRandom.seed(seed)
        }

        var latents = MLXRandom.normal([
            1,
            WooshResources.latentChannels,
            config.latentFrames(),
        ]).asType(.float32)
        let positiveText = textConditioner.encode(prompts: [prompt])
        let negativeText = textConditioner.encode(prompts: [""])
        let video = videoFeatures.asType(.float32)
        let positive = WooshCondition(
            crossAttention: positiveText.embeddings.asType(.float32),
            crossAttentionMask: positiveText.attentionMask,
            videoFeatures: video
        )
        let negative = WooshCondition(
            crossAttention: negativeText.embeddings.asType(.float32),
            crossAttentionMask: negativeText.attentionMask,
            videoFeatures: MLXArray.zeros(video.shape, dtype: video.dtype)
        )
        let stepSize = MLXArray(1 / Float(config.steps)).asType(.float32)
        let guidance = MLXArray(config.guidanceScale).asType(.float32)

        for step in 0..<config.steps {
            let tValue = 1 - Float(step) / Float(config.steps)
            let t = MLXArray([tValue]).asType(.float32)
            let condVelocity = try vflow(x: latents, t: t, cond: positive)
            let uncondVelocity = try vflow(x: latents, t: t, cond: negative)
            let velocity = condVelocity + guidance * (condVelocity - uncondVelocity)
            latents = latents + stepSize * velocity
            MLX.eval(latents)
            progress?(step + 1, config.steps)
        }

        return WooshGenerationResult(samples: try autoencoder.decode(latents))
    }

    private func resolvedRenoiseSchedule(config: WooshDenoiseConfig) -> [Float] {
        if config.renoiseSchedule.isEmpty {
            if resources.variant.defaultRenoiseSchedule.count == config.steps {
                return resources.variant.defaultRenoiseSchedule
            }
            return Array(repeating: 0, count: config.steps)
        }
        if config.renoiseSchedule.count == 1 {
            return Array(repeating: config.renoiseSchedule[0], count: config.steps)
        }
        return config.renoiseSchedule
    }

    private func applyRenoise(
        _ latents: MLXArray,
        t: Float,
        r: Float,
        strength: Float
    ) -> (latents: MLXArray, t: Float) {
        guard strength > 0 else {
            return (latents, t)
        }

        let gamma = strength * (t - r)
        let tHat = min(t + gamma, 1)
        guard tHat > t else {
            return (latents, t)
        }

        let scale = (1 - tHat) / (1 - t + 1e-12)
        let stdSquared = max(0, tHat * tHat - (t * scale) * (t * scale))
        let std = sqrt(stdSquared)
        let noise = MLXRandom.normal(latents.shape).asType(latents.dtype)
        return (latents * MLXArray(scale).asType(latents.dtype) + noise * MLXArray(std).asType(latents.dtype), tHat)
    }

    private static func mapDiTWeights(_ key: String, _ value: MLXArray) -> [(String, MLXArray)] {
        let mappedKey: String
        if key.hasPrefix("dit.") {
            mappedKey = String(key.dropFirst(4))
        } else if key.hasPrefix("preprocessing.")
            || key.hasPrefix("layers.")
            || key.hasPrefix("postprocessing.") {
            mappedKey = key
        } else {
            return []
        }

        return [(mapSequentialMLPKeys(mappedKey), value)]
    }

    private static func mapVideoDiTWeights(_ key: String, _ value: MLXArray) -> [(String, MLXArray)] {
        guard key.hasPrefix("dit.") || key.hasPrefix("conditioners.") else {
            return []
        }
        if key.hasPrefix("dit.preprocessing.to_logvar_ctm.") {
            return []
        }
        return [(mapSequentialMLPKeys(key), value)]
    }

    private static func mapSequentialMLPKeys(_ key: String) -> String {
        var mapped = key
        for moduleName in ["to_timestep_embed", "to_cond_embed", "to_logvar"] {
            mapped = mapped
                .replacingOccurrences(of: ".\(moduleName).0.", with: ".\(moduleName).first.")
                .replacingOccurrences(of: ".\(moduleName).2.", with: ".\(moduleName).second.")
        }
        return mapped
    }
}
