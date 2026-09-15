import AudioCore
import MLX

public struct MiniMaxMusic3AudioResult: Sendable {
    public let waveform: AudioWaveform
    public let nativeSampleRate: Int
    public let frameCount: Int
    public let profile: MiniMaxMusic3GenerationProfile?
    public let audioHealth: MiniMaxMusic3AudioHealthReport
}

/// Owns model loading, serialized generation, evaluation, and resampling.
public actor MiniMaxMusic3GenerationOperation {
    private let resources: MiniMaxMusic3Resources
    private let loadingStrategy: MiniMaxMusic3LoadingStrategy
    private let performanceMode: MiniMaxMusic3PerformanceMode
    private let streams = Stream.Context()
    private var pipeline: MiniMaxMusic3Pipeline?

    public init(
        resources: MiniMaxMusic3Resources, loadingStrategy: MiniMaxMusic3LoadingStrategy,
        performanceMode: MiniMaxMusic3PerformanceMode
    ) {
        self.resources = resources
        self.loadingStrategy = loadingStrategy
        self.performanceMode = performanceMode
    }

    public func load() async throws {
        try Task.checkCancellation()
        try await Stream.withDefaultStream(streams) {
            defer { streams.synchronize() }
            _ = try loadedPipeline()
        }
    }

    public func generate(
        _ plan: MiniMaxMusic3GenerationPlan,
        progress: (@Sendable (MiniMaxMusic3Progress) -> Void)? = nil
    ) async throws -> MiniMaxMusic3AudioResult {
        try Task.checkCancellation()
        return try await Stream.withDefaultStream(streams) {
            defer { streams.synchronize() }
            let result = try loadedPipeline().generate(options: plan.generation, progress: progress)
            try Task.checkCancellation()
            let waveform = try MusicWaveformAdapter.miniMaxMusic3(
                result.waveform, sampleRate: result.sampleRate
            ).resampled(to: plan.sampleRate)
            return MiniMaxMusic3AudioResult(
                waveform: waveform, nativeSampleRate: result.sampleRate, frameCount: result.frameCount,
                profile: result.profile, audioHealth: result.audioHealth
            )
        }
    }

    private func loadedPipeline() throws -> MiniMaxMusic3Pipeline {
        let active = try pipeline ?? MiniMaxMusic3Pipeline(
            resources: resources, loadingStrategy: loadingStrategy, performanceMode: performanceMode
        )
        pipeline = active
        return active
    }
}
