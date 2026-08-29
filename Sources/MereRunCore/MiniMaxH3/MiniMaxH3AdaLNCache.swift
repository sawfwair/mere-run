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

struct MiniMaxH3AdaLNCachePack: Codable, Sendable {
    struct Schedule: Codable, Hashable, Sendable {
        let pointCount: Int
        let videoFlowShift: Float
        let audioFlowShift: Float

        enum CodingKeys: String, CodingKey {
            case pointCount = "point_count"
            case videoFlowShift = "video_flow_shift"
            case audioFlowShift = "audio_flow_shift"
        }

        var filename: String {
            if self == Self(pointCount: 31, videoFlowShift: 12, audioFlowShift: 3) {
                return MiniMaxH3AdaLNCache.filename
            }
            return "adaln_cache-p\(pointCount)-v\(Self.label(videoFlowShift))-a\(Self.label(audioFlowShift)).safetensors"
        }

        func schedules() throws -> (video: MiniMaxH3Schedule, audio: MiniMaxH3Schedule) {
            (
                try MiniMaxH3Schedule(pointCount: pointCount, shift: videoFlowShift),
                try MiniMaxH3Schedule(pointCount: pointCount, shift: audioFlowShift)
            )
        }

        func matches(video: MiniMaxH3Schedule, audio: MiniMaxH3Schedule) -> Bool {
            guard let schedules = try? schedules() else { return false }
            return schedules.video.sigmas == video.sigmas && schedules.audio.sigmas == audio.sigmas
        }

        private static func label(_ value: Float) -> String {
            value.rounded() == value ? String(Int(value)) : String(value)
        }
    }

    struct Entry: Codable, Hashable, Sendable {
        let schedule: Schedule
        let filename: String
        let byteCount: Int64
        let sha256: String

        enum CodingKeys: String, CodingKey {
            case schedule
            case filename
            case byteCount = "byte_count"
            case sha256
        }
    }

    struct Selection {
        let cache: MiniMaxH3AdaLNCache
        let exact: Bool
        let sourceSchedule: Schedule
    }

    static let filename = "adaln_cache.index.json"
    static let schemaVersion = 1
    static let format = "mere.run.minimax-h3-adaln-cache-pack"
    static let productionSchedules = [5, 9, 12, 16, 21, 31].map {
        Schedule(pointCount: $0, videoFlowShift: 12, audioFlowShift: 3)
    } + [5, 9].map {
        Schedule(pointCount: $0, videoFlowShift: 6, audioFlowShift: 3)
    }

    let schemaVersion: Int
    let format: String
    let sourceIdentity: String
    let entries: [Entry]

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case format
        case sourceIdentity = "source_identity"
        case entries
    }

    static func load(
        from rootURL: URL,
        configuration: MiniMaxH3TransformerConfiguration,
        videoSchedule: MiniMaxH3Schedule,
        audioSchedule: MiniMaxH3Schedule,
        sourceIdentity: String,
        fileManager: FileManager = .default
    ) throws -> Selection {
        let indexURL = rootURL.appending(path: filename)
        guard fileManager.fileExists(atPath: indexURL.path) else {
            let legacyURL = rootURL.appending(path: MiniMaxH3AdaLNCache.filename)
            let exactCache = try? MiniMaxH3AdaLNCache.load(
                from: legacyURL,
                configuration: configuration,
                videoSchedule: videoSchedule,
                audioSchedule: audioSchedule,
                sourceIdentity: sourceIdentity
            )
            let cache = try exactCache ?? MiniMaxH3AdaLNCache.load(
                from: legacyURL,
                configuration: configuration,
                videoSchedule: videoSchedule,
                audioSchedule: audioSchedule,
                sourceIdentity: sourceIdentity,
                allowScheduleResampling: true
            )
            return Selection(
                cache: cache,
                exact: exactCache != nil,
                sourceSchedule: Schedule(pointCount: 31, videoFlowShift: 12, audioFlowShift: 3)
            )
        }

        let pack = try loadIndex(from: indexURL, sourceIdentity: sourceIdentity)

        if let entry = pack.entries.first(where: {
            $0.schedule.matches(video: videoSchedule, audio: audioSchedule)
        }) {
            let cache = try load(
                entry: entry,
                rootURL: rootURL,
                configuration: configuration,
                sourceIdentity: sourceIdentity,
                fileManager: fileManager
            )
            guard cache.videoSigmas == videoSchedule.sigmas,
                  cache.audioSigmas == audioSchedule.sigmas else {
                throw MiniMaxH3AdaLNCacheError.incompatible(
                    "exact cache entry does not match its declared schedule"
                )
            }
            return Selection(cache: cache, exact: true, sourceSchedule: entry.schedule)
        }

        guard let entry = pack.entries.max(by: {
            $0.schedule.pointCount < $1.schedule.pointCount
        }) else {
            throw MiniMaxH3AdaLNCacheError.incompatible("cache-pack index has no dense table")
        }
        let source = try load(
            entry: entry,
            rootURL: rootURL,
            configuration: configuration,
            sourceIdentity: sourceIdentity,
            fileManager: fileManager
        )
        return Selection(
            cache: try source.resampled(
                configuration: configuration,
                videoSchedule: videoSchedule,
                audioSchedule: audioSchedule
            ),
            exact: false,
            sourceSchedule: entry.schedule
        )
    }

    /// Validates the immutable production pack without evaluating MLX arrays.
    /// Managed installation uses this path so artifact verification remains a
    /// file-I/O operation; runtime tensor evaluation begins only at inference.
    static func validateClosure(
        from rootURL: URL,
        sourceIdentity: String,
        fileManager: FileManager = .default
    ) throws {
        let indexURL = rootURL.appending(path: filename)
        guard fileManager.fileExists(atPath: indexURL.path) else {
            throw MiniMaxH3AdaLNCacheError.incompatible("cache-pack index is missing")
        }
        let pack = try loadIndex(from: indexURL, sourceIdentity: sourceIdentity)
        let expectedSchedules = Set(productionSchedules)
        guard pack.entries.count == expectedSchedules.count,
              Set(pack.entries.map(\.schedule)) == expectedSchedules else {
            throw MiniMaxH3AdaLNCacheError.incompatible(
                "cache-pack schedule closure does not match the production set"
            )
        }
        guard Set(pack.entries.map(\.filename)).count == pack.entries.count else {
            throw MiniMaxH3AdaLNCacheError.incompatible("cache-pack filenames are not unique")
        }

        for entry in pack.entries {
            guard entry.filename == entry.schedule.filename,
                  entry.filename == URL(fileURLWithPath: entry.filename).lastPathComponent,
                  !entry.filename.contains("/") else {
                throw MiniMaxH3AdaLNCacheError.incompatible(
                    "cache-pack filename is invalid: \(entry.filename)"
                )
            }
            let url = rootURL.appending(path: entry.filename)
            guard fileManager.fileExists(atPath: url.path) else {
                throw MiniMaxH3AdaLNCacheError.incompatible(
                    "cache-pack file is missing: \(entry.filename)"
                )
            }
            guard try ModelArtifactPin.fileByteCount(url, fileManager: fileManager) == entry.byteCount else {
                throw MiniMaxH3AdaLNCacheError.incompatible(
                    "cache-pack byte count changed for \(entry.filename)"
                )
            }
            guard try ModelArtifactPin.fileSHA256(url.resolvingSymlinksInPath())
                    == entry.sha256.lowercased() else {
                throw MiniMaxH3AdaLNCacheError.incompatible(
                    "cache-pack SHA-256 changed for \(entry.filename)"
                )
            }
        }
    }

    static func write(entries: [Entry], sourceIdentity: String, to rootURL: URL) throws -> URL {
        let indexURL = rootURL.appending(path: filename)
        let temporaryURL = rootURL.appending(path: ".\(filename).\(UUID().uuidString).tmp")
        defer { try? FileManager.default.removeItem(at: temporaryURL) }
        let pack = Self(
            schemaVersion: schemaVersion,
            format: format,
            sourceIdentity: sourceIdentity,
            entries: entries.sorted {
                ($0.schedule.pointCount, $0.schedule.videoFlowShift, $0.schedule.audioFlowShift)
                    < ($1.schedule.pointCount, $1.schedule.videoFlowShift, $1.schedule.audioFlowShift)
            }
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        var data = try encoder.encode(pack)
        data.append(0x0A)
        try data.write(to: temporaryURL, options: .atomic)
        if FileManager.default.fileExists(atPath: indexURL.path) {
            try FileManager.default.removeItem(at: indexURL)
        }
        try FileManager.default.moveItem(at: temporaryURL, to: indexURL)
        return indexURL
    }

    static func entry(
        schedule: Schedule,
        url: URL
    ) throws -> Entry {
        Entry(
            schedule: schedule,
            filename: url.lastPathComponent,
            byteCount: try ModelArtifactPin.fileByteCount(url),
            sha256: try ModelArtifactPin.fileSHA256(url)
        )
    }

    private static func loadIndex(
        from indexURL: URL,
        sourceIdentity: String
    ) throws -> Self {
        let pack: Self
        do {
            pack = try JSONDecoder().decode(Self.self, from: Data(contentsOf: indexURL))
        } catch {
            throw MiniMaxH3AdaLNCacheError.incompatible(
                "cache-pack index is invalid: \(error.localizedDescription)"
            )
        }
        guard pack.schemaVersion == schemaVersion, pack.format == format else {
            throw MiniMaxH3AdaLNCacheError.incompatible("unsupported cache-pack index")
        }
        guard pack.sourceIdentity == sourceIdentity else {
            throw MiniMaxH3AdaLNCacheError.incompatible("cache-pack transformer artifact changed")
        }
        guard !pack.entries.isEmpty else {
            throw MiniMaxH3AdaLNCacheError.incompatible("cache-pack index has no entries")
        }
        return pack
    }

    private static func load(
        entry: Entry,
        rootURL: URL,
        configuration: MiniMaxH3TransformerConfiguration,
        sourceIdentity: String,
        fileManager: FileManager
    ) throws -> MiniMaxH3AdaLNCache {
        guard entry.filename == URL(fileURLWithPath: entry.filename).lastPathComponent,
              !entry.filename.contains("/") else {
            throw MiniMaxH3AdaLNCacheError.incompatible("cache-pack filename is unsafe")
        }
        let url = rootURL.appending(path: entry.filename)
        guard fileManager.fileExists(atPath: url.path) else {
            throw MiniMaxH3AdaLNCacheError.incompatible("cache-pack file is missing: \(entry.filename)")
        }
        let actualBytes = try ModelArtifactPin.fileByteCount(url, fileManager: fileManager)
        guard actualBytes == entry.byteCount else {
            throw MiniMaxH3AdaLNCacheError.incompatible(
                "cache-pack byte count changed for \(entry.filename)"
            )
        }
        let actualSHA256 = try ModelArtifactPin.fileSHA256(url.resolvingSymlinksInPath())
        guard actualSHA256 == entry.sha256.lowercased() else {
            throw MiniMaxH3AdaLNCacheError.incompatible(
                "cache-pack SHA-256 changed for \(entry.filename)"
            )
        }
        let schedules = try entry.schedule.schedules()
        return try MiniMaxH3AdaLNCache.load(
            from: url,
            configuration: configuration,
            videoSchedule: schedules.video,
            audioSchedule: schedules.audio,
            sourceIdentity: sourceIdentity
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
    public static func artifactURLs(resources: MiniMaxH3Resources) -> [URL] {
        [resources.adaLNCachePackIndexURL] + MiniMaxH3AdaLNCachePack.productionSchedules.map {
            resources.rootURL.appending(path: $0.filename)
        }
    }

    @discardableResult
    public static func optimize(
        resources: MiniMaxH3Resources,
        replacing: Bool = false,
        progressHandler: (@Sendable (Int, Int) -> Void)? = nil
    ) throws -> URL {
        let missing = resources.validate()
        guard missing.isEmpty else { throw MiniMaxH3GeneratorError.missingModelFiles(missing) }
        let configuration = try resources.loadConfiguration()
        let sourceIdentity = try resources.adaLNCacheSourceIdentity()
        let indexURL = resources.adaLNCachePackIndexURL
        if FileManager.default.fileExists(atPath: indexURL.path) {
            for schedule in MiniMaxH3AdaLNCachePack.productionSchedules {
                let schedules = try schedule.schedules()
                let selection = try MiniMaxH3AdaLNCachePack.load(
                    from: resources.rootURL,
                    configuration: .init(configuration),
                    videoSchedule: schedules.video,
                    audioSchedule: schedules.audio,
                    sourceIdentity: sourceIdentity
                )
                guard selection.exact else {
                    throw MiniMaxH3AdaLNCacheError.incompatible(
                        "exact production schedule is missing: \(schedule.filename)"
                    )
                }
            }
            if !replacing {
                return indexURL
            }
            if try resources.requiresAdaLNCache() {
                return indexURL
            }
        }
        if try resources.requiresAdaLNCache() {
            throw MiniMaxH3AdaLNCacheError.incompatible(
                "a pruned transformer cannot synthesize a missing exact cache pack"
            )
        }
        let transformer = try MiniMaxH3ModelLoader.loadTransformer(
            resources: resources,
            configuration: configuration
        )
        var entries: [MiniMaxH3AdaLNCachePack.Entry] = []
        for schedule in MiniMaxH3AdaLNCachePack.productionSchedules {
            let schedules = try schedule.schedules()
            let cache = transformer.precomputeAdaLN(
                videoSchedule: schedules.video,
                audioSchedule: schedules.audio,
                sourceIdentity: sourceIdentity,
                progressHandler: progressHandler
            )
            let url = resources.rootURL.appending(path: schedule.filename)
            try cache.save(to: url, replacing: replacing)
            entries.append(try MiniMaxH3AdaLNCachePack.entry(schedule: schedule, url: url))
        }
        return try MiniMaxH3AdaLNCachePack.write(
            entries: entries,
            sourceIdentity: sourceIdentity,
            to: resources.rootURL
        )
    }
}
