import Foundation
import MLX

public enum MiniMaxH3AdaLNCacheError: LocalizedError, Sendable {
    case incompatible(String)

    public var errorDescription: String? {
        switch self {
        case .incompatible(let reason):
            return "MiniMax-H3 AdaLN cache is incompatible: \(reason)"
        }
    }
}

struct MiniMaxH3AdaLNCache {
    static let filename = "adaln_cache.safetensors"
    static let schemaVersion = "2"

    let timeEmbeddings: MLXArray
    let blockModulations: [MLXArray]
    let finalModulations: MLXArray
    let videoSigmas: [Float]
    let audioSigmas: [Float]
    let sourceIdentity: String

    var stepCount: Int { videoSigmas.count - 1 }

    func step(at index: Int) -> MiniMaxH3AdaLNStep {
        precondition(index >= 0 && index < stepCount)
        return MiniMaxH3AdaLNStep(
            timeEmbedding: timeEmbeddings[index],
            blockModulations: blockModulations.map { $0[index] },
            finalModulation: finalModulations[index]
        )
    }

    func isCompatible(
        configuration: MiniMaxH3TransformerConfiguration,
        videoSchedule: MiniMaxH3Schedule,
        audioSchedule: MiniMaxH3Schedule
    ) -> Bool {
        isStructurallyCompatible(configuration: configuration)
            && videoSigmas == videoSchedule.sigmas
            && audioSigmas == audioSchedule.sigmas
    }

    private func isStructurallyCompatible(
        configuration: MiniMaxH3TransformerConfiguration
    ) -> Bool {
        videoSigmas.count >= 2
            && videoSigmas.count == audioSigmas.count
            && blockModulations.count == configuration.layerCount
            && timeEmbeddings.shape == [stepCount, 3, configuration.timeEmbeddingDimension]
            && finalModulations.shape == [stepCount, 3, 2 * configuration.hiddenSize]
            && blockModulations.allSatisfy {
                $0.shape == [stepCount, 3 * 3, 6 * configuration.hiddenSize]
            }
    }

    /// Rebuilds the small inference table for another sampler schedule without
    /// restoring the 13B-parameter AdaLN branch. AdaLN is a smooth function of
    /// timestep, and every stored video/audio point contains all three modality
    /// rows. Combining both source schedules therefore gives up to twice the
    /// sampling density of either schedule alone.
    func resampled(
        configuration: MiniMaxH3TransformerConfiguration,
        videoSchedule: MiniMaxH3Schedule,
        audioSchedule: MiniMaxH3Schedule
    ) throws -> MiniMaxH3AdaLNCache {
        guard isStructurallyCompatible(configuration: configuration) else {
            throw MiniMaxH3AdaLNCacheError.incompatible("source tensor geometry does not match")
        }
        guard videoSchedule.timesteps.count == audioSchedule.timesteps.count else {
            throw MiniMaxH3AdaLNCacheError.incompatible("target video/audio schedules disagree")
        }
        if isCompatible(
            configuration: configuration,
            videoSchedule: videoSchedule,
            audioSchedule: audioSchedule
        ) {
            return self
        }

        let samples = sourceSamples()
        let targetTimesteps = videoSchedule.timesteps.indices.map { index in
            [
                videoSchedule.timesteps[index],
                audioSchedule.timesteps[index],
                max(videoSchedule.timesteps[index], 0.999),
            ]
        }

        let resampledTimeEmbeddings = try MLX.stacked(targetTimesteps.map { step in
            try MLX.stacked(step.map { timestep in
                try interpolatedValue(timestep: timestep, samples: samples) { sample in
                    timeEmbeddings[sample.stepIndex, sample.timestepIndex, 0...]
                }
            }, axis: 0)
        }, axis: 0)
        MLX.eval(resampledTimeEmbeddings)

        var resampledBlocks: [MLXArray] = []
        resampledBlocks.reserveCapacity(blockModulations.count)
        for source in blockModulations {
            let resampled = try MLX.stacked(targetTimesteps.map { step in
                try MLX.concatenated(step.map { timestep in
                    try interpolatedValue(timestep: timestep, samples: samples) { sample in
                        let start = sample.timestepIndex * 3
                        return source[sample.stepIndex, start..<(start + 3), 0...]
                    }
                }, axis: 0)
            }, axis: 0)
            MLX.eval(resampled)
            resampledBlocks.append(resampled)
        }

        let resampledFinal = try MLX.stacked(targetTimesteps.map { step in
            try MLX.stacked(step.map { timestep in
                try interpolatedValue(timestep: timestep, samples: samples) { sample in
                    finalModulations[sample.stepIndex, sample.timestepIndex, 0...]
                }
            }, axis: 0)
        }, axis: 0)
        MLX.eval(resampledFinal)

        return MiniMaxH3AdaLNCache(
            timeEmbeddings: resampledTimeEmbeddings,
            blockModulations: resampledBlocks,
            finalModulations: resampledFinal,
            videoSigmas: videoSchedule.sigmas,
            audioSigmas: audioSchedule.sigmas,
            sourceIdentity: sourceIdentity
        )
    }

    func save(to url: URL, replacing: Bool) throws {
        var arrays = [
            "time_embeddings": timeEmbeddings,
            "final_modulations": finalModulations,
            "video_sigmas": MLXArray(videoSigmas),
            "audio_sigmas": MLXArray(audioSigmas),
        ]
        for (index, modulation) in blockModulations.enumerated() {
            arrays["blocks.\(index).modulations"] = modulation
        }
        let temporaryURL = url.deletingLastPathComponent().appending(
            path: ".\(url.lastPathComponent).\(UUID().uuidString).tmp.safetensors"
        )
        defer { try? FileManager.default.removeItem(at: temporaryURL) }
        try MLX.save(
            arrays: arrays,
            metadata: [
                "schema_version": Self.schemaVersion,
                "format": "mere.run.minimax-h3-adaln-cache",
                "source_identity": sourceIdentity,
            ],
            url: temporaryURL
        )
        if FileManager.default.fileExists(atPath: url.path) {
            guard replacing else {
                throw MiniMaxH3AdaLNCacheError.incompatible("cache already exists at \(url.path)")
            }
            try FileManager.default.removeItem(at: url)
        }
        try FileManager.default.moveItem(at: temporaryURL, to: url)
    }

    static func load(
        from url: URL,
        configuration: MiniMaxH3TransformerConfiguration,
        videoSchedule: MiniMaxH3Schedule,
        audioSchedule: MiniMaxH3Schedule,
        sourceIdentity: String,
        allowScheduleResampling: Bool = false
    ) throws -> MiniMaxH3AdaLNCache {
        let (arrays, metadata) = try MLX.loadArraysAndMetadata(url: url)
        guard metadata["schema_version"] == schemaVersion else {
            throw MiniMaxH3AdaLNCacheError.incompatible("unsupported schema version")
        }
        guard metadata["source_identity"] == sourceIdentity else {
            throw MiniMaxH3AdaLNCacheError.incompatible("transformer artifact changed")
        }
        guard let timeEmbeddings = arrays["time_embeddings"],
              let finalModulations = arrays["final_modulations"],
              let videoSigmaArray = arrays["video_sigmas"],
              let audioSigmaArray = arrays["audio_sigmas"] else {
            throw MiniMaxH3AdaLNCacheError.incompatible("required tensors are missing")
        }
        let blockModulations = try (0..<configuration.layerCount).map { index in
            guard let value = arrays["blocks.\(index).modulations"] else {
                throw MiniMaxH3AdaLNCacheError.incompatible("block \(index) is missing")
            }
            return value
        }
        MLX.eval(videoSigmaArray, audioSigmaArray)
        let cache = MiniMaxH3AdaLNCache(
            timeEmbeddings: timeEmbeddings,
            blockModulations: blockModulations,
            finalModulations: finalModulations,
            videoSigmas: videoSigmaArray.asArray(Float.self),
            audioSigmas: audioSigmaArray.asArray(Float.self),
            sourceIdentity: sourceIdentity
        )
        guard cache.isStructurallyCompatible(configuration: configuration) else {
            throw MiniMaxH3AdaLNCacheError.incompatible("source tensor geometry does not match")
        }
        if cache.isCompatible(
            configuration: configuration,
            videoSchedule: videoSchedule,
            audioSchedule: audioSchedule
        ) {
            return cache
        }
        guard allowScheduleResampling else {
            throw MiniMaxH3AdaLNCacheError.incompatible("schedule does not match")
        }
        return try cache.resampled(
            configuration: configuration,
            videoSchedule: videoSchedule,
            audioSchedule: audioSchedule
        )
    }
}

private extension MiniMaxH3AdaLNCache {
    struct SourceSample {
        let timestep: Float
        let stepIndex: Int
        /// 0 = video, 1 = audio, 2 = condition. Block caches hold three
        /// modality rows per timestep; final/time tables hold one row.
        let timestepIndex: Int
    }

    struct Interpolation {
        let lower: SourceSample
        let upper: SourceSample
        let fraction: Float
    }

    func sourceSamples() -> [SourceSample] {
        var values: [SourceSample] = []
        values.reserveCapacity(stepCount * 3)
        for index in 0..<stepCount {
            let videoTimestep = 1 - videoSigmas[index]
            values.append(.init(timestep: videoTimestep, stepIndex: index, timestepIndex: 0))
            values.append(.init(
                timestep: 1 - audioSigmas[index],
                stepIndex: index,
                timestepIndex: 1
            ))
            values.append(.init(
                timestep: max(videoTimestep, 0.999),
                stepIndex: index,
                timestepIndex: 2
            ))
        }
        values.sort { $0.timestep < $1.timestep }
        return values.reduce(into: []) { unique, sample in
            if unique.last?.timestep != sample.timestep {
                unique.append(sample)
            }
        }
    }

    func interpolation(
        for timestep: Float,
        samples: [SourceSample]
    ) throws -> Interpolation {
        guard let first = samples.first, let last = samples.last,
              timestep >= first.timestep, timestep <= last.timestep else {
            throw MiniMaxH3AdaLNCacheError.incompatible(
                "target timestep \(timestep) is outside the cached curve"
            )
        }
        let upperIndex = samples.firstIndex { $0.timestep >= timestep } ?? (samples.count - 1)
        let upper = samples[upperIndex]
        if upper.timestep == timestep || upperIndex == 0 {
            return Interpolation(lower: upper, upper: upper, fraction: 0)
        }
        let lower = samples[upperIndex - 1]
        let fraction = (timestep - lower.timestep) / (upper.timestep - lower.timestep)
        return Interpolation(lower: lower, upper: upper, fraction: fraction)
    }

    func interpolatedValue(
        timestep: Float,
        samples: [SourceSample],
        value: (SourceSample) -> MLXArray
    ) throws -> MLXArray {
        let interpolation = try interpolation(for: timestep, samples: samples)
        let lower = value(interpolation.lower)
        guard interpolation.fraction != 0 else { return lower }
        let upper = value(interpolation.upper)
        return lower * (1 - interpolation.fraction) + upper * interpolation.fraction
    }
}

public enum MiniMaxH3ModelOptimizer {
    @discardableResult
    public static func optimize(
        resources: MiniMaxH3Resources,
        replacing: Bool = false,
        progressHandler: (@Sendable (Int, Int) -> Void)? = nil
    ) throws -> URL {
        let missing = resources.validate()
        guard missing.isEmpty else { throw MiniMaxH3GeneratorError.missingModelFiles(missing) }
        let configuration = try resources.loadConfiguration()
        let videoSchedule = try MiniMaxH3Schedule(
            pointCount: configuration.sampleSteps,
            shift: configuration.videoFlowShift
        )
        let audioSchedule = try MiniMaxH3Schedule(
            pointCount: configuration.sampleSteps,
            shift: configuration.audioFlowShift
        )
        let sourceIdentity = try resources.adaLNCacheSourceIdentity()
        if FileManager.default.fileExists(atPath: resources.adaLNCacheURL.path), !replacing {
            _ = try MiniMaxH3AdaLNCache.load(
                from: resources.adaLNCacheURL,
                configuration: .init(configuration),
                videoSchedule: videoSchedule,
                audioSchedule: audioSchedule,
                sourceIdentity: sourceIdentity
            )
            return resources.adaLNCacheURL
        }
        let transformer = try MiniMaxH3ModelLoader.loadTransformer(
            resources: resources,
            configuration: configuration
        )
        let cache = transformer.precomputeAdaLN(
            videoSchedule: videoSchedule,
            audioSchedule: audioSchedule,
            sourceIdentity: sourceIdentity,
            progressHandler: progressHandler
        )
        try cache.save(to: resources.adaLNCacheURL, replacing: replacing)
        return resources.adaLNCacheURL
    }
}
