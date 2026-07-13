import Foundation
import MLX

public struct MMAudioGenerationResult: Sendable, Hashable {
    public let samples: [Float]
    public let sampleRate: Int

    public init(samples: [Float], sampleRate: Int = MMAudioResources.sampleRate) {
        self.samples = samples
        self.sampleRate = sampleRate
    }
}

public final class MMAudioGenerator {
    private let resources: MMAudioModelResources

    public init(resources: MMAudioModelResources) throws {
        let missing = resources.validate()
        guard missing.isEmpty else {
            throw MMAudioGeneratorError.missingFiles(missing)
        }
        self.resources = resources
    }

    public func generateText(
        prompt: String,
        negativePrompt: String = "",
        config: MMAudioGenerationConfig,
        progress: (@Sendable (_ completedStep: Int, _ totalSteps: Int) -> Void)? = nil
    ) async throws -> MMAudioGenerationResult {
        try validate(config)
        let conditions = try await makeTextConditions(
            prompt: prompt,
            negativePrompt: negativePrompt,
            config: config
        )
        return try generate(conditions: conditions, config: config, progress: progress)
    }

    public func generateVideo(
        prompt: String,
        negativePrompt: String = "",
        videoURL: URL,
        config: MMAudioGenerationConfig,
        clipBatchSize: Int = 4,
        syncBatchSize: Int = 1,
        progress: (@Sendable (_ completedStep: Int, _ totalSteps: Int) -> Void)? = nil
    ) async throws -> MMAudioGenerationResult {
        try validate(config)
        let conditions = try await makeVideoConditions(
            prompt: prompt,
            negativePrompt: negativePrompt,
            videoURL: videoURL,
            config: config,
            clipBatchSize: clipBatchSize,
            syncBatchSize: syncBatchSize
        )
        return try generate(conditions: conditions, config: config, progress: progress)
    }

    private func makeTextConditions(
        prompt: String,
        negativePrompt: String,
        config: MMAudioGenerationConfig
    ) async throws -> MMAudioConditioningResult {
        let conditioner = try await MMAudioConditioner.load(resources: resources)
        let conditions = conditioner.textConditions(
            prompt: prompt,
            negativePrompt: negativePrompt,
            config: config
        )
        MLX.eval(
            conditions.positive.clip,
            conditions.positive.sync,
            conditions.positive.text,
            conditions.negativeText
        )
        return conditions
    }

    private func makeVideoConditions(
        prompt: String,
        negativePrompt: String,
        videoURL: URL,
        config: MMAudioGenerationConfig,
        clipBatchSize: Int,
        syncBatchSize: Int
    ) async throws -> MMAudioConditioningResult {
        let conditioner = try await MMAudioConditioner.load(resources: resources)
        let conditions = try conditioner.videoConditions(
            prompt: prompt,
            negativePrompt: negativePrompt,
            videoURL: videoURL,
            config: config,
            clipBatchSize: clipBatchSize,
            syncBatchSize: syncBatchSize
        )
        MLX.eval(
            conditions.positive.clip,
            conditions.positive.sync,
            conditions.positive.text,
            conditions.negativeText
        )
        return conditions
    }

    private func generate(
        conditions: MMAudioConditioningResult,
        config: MMAudioGenerationConfig,
        progress: (@Sendable (_ completedStep: Int, _ totalSteps: Int) -> Void)?
    ) throws -> MMAudioGenerationResult {
        Memory.clearCache()
        let latent = try sampleLatents(conditions: conditions, config: config, progress: progress)
        Memory.clearCache()
        let spectrogram = try decodeLatents(latent)
        Memory.clearCache()
        let samples = try vocode(spectrogram)
        Memory.clearCache()
        return MMAudioGenerationResult(samples: samples)
    }

    private func sampleLatents(
        conditions: MMAudioConditioningResult,
        config: MMAudioGenerationConfig,
        progress: (@Sendable (_ completedStep: Int, _ totalSteps: Int) -> Void)?
    ) throws -> MLXArray {
        let network = try MMAudioNetwork.load(resources: resources)
        let latent = network.sampleLatents(
            conditions: conditions.positive,
            negativeText: conditions.negativeText,
            config: config,
            progress: progress
        )
        MLX.eval(latent)
        return latent
    }

    private func decodeLatents(_ latent: MLXArray) throws -> MLXArray {
        let vae = try MMAudioVAE.load(resources: resources)
        let spectrogram = vae.decode(latent)
        MLX.eval(spectrogram)
        return spectrogram
    }

    private func vocode(_ spectrogram: MLXArray) throws -> [Float] {
        let vocoder = try MMAudioBigVGAN.load(resources: resources)
        let waveform = vocoder(spectrogram)
        MLX.eval(waveform)
        return waveform[0, 0..., 0].asType(.float32).asArray(Float.self)
    }

    private func validate(_ config: MMAudioGenerationConfig) throws {
        guard config.durationSeconds > 0 else {
            throw MMAudioGeneratorError.invalidDuration(config.durationSeconds)
        }
        guard config.steps > 0 else {
            throw MMAudioGeneratorError.invalidSteps(config.steps)
        }
        guard config.guidanceScale >= 0 else {
            throw MMAudioGeneratorError.invalidGuidance(config.guidanceScale)
        }
    }
}

public enum MMAudioGeneratorError: LocalizedError {
    case missingFiles([URL])
    case invalidDuration(Float)
    case invalidSteps(Int)
    case invalidGuidance(Float)

    public var errorDescription: String? {
        switch self {
        case .missingFiles(let urls):
            "Missing MMAudio files:\n" + urls.map { "- \($0.path)" }.joined(separator: "\n")
        case .invalidDuration(let value):
            "MMAudio duration must be greater than zero; got \(value)."
        case .invalidSteps(let value):
            "MMAudio steps must be at least one; got \(value)."
        case .invalidGuidance(let value):
            "MMAudio guidance must be non-negative; got \(value)."
        }
    }
}
