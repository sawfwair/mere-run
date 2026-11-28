import Foundation
import MLX

extension MiniMaxH3AdaLNCache {
    package func save(to url: URL, replacing: Bool) throws {
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

    package static func load(
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

package struct MiniMaxH3AdaLNCachePack: Codable, Sendable {
    package struct Schedule: Codable, Hashable, Sendable {
        package let pointCount: Int
        package let videoFlowShift: Float
        package let audioFlowShift: Float

        enum CodingKeys: String, CodingKey {
            case pointCount = "point_count"
            case videoFlowShift = "video_flow_shift"
            case audioFlowShift = "audio_flow_shift"
        }

        package var filename: String {
            if self == Self(pointCount: 31, videoFlowShift: 12, audioFlowShift: 3) {
                return MiniMaxH3AdaLNCache.filename
            }
            return "adaln_cache-p\(pointCount)-v\(Self.label(videoFlowShift))-a\(Self.label(audioFlowShift)).safetensors"
        }

        package func schedules() throws -> (video: MiniMaxH3Schedule, audio: MiniMaxH3Schedule) {
            (
                try MiniMaxH3Schedule(pointCount: pointCount, shift: videoFlowShift),
                try MiniMaxH3Schedule(pointCount: pointCount, shift: audioFlowShift)
            )
        }

        package func matches(video: MiniMaxH3Schedule, audio: MiniMaxH3Schedule) -> Bool {
            guard let schedules = try? schedules() else { return false }
            return schedules.video.sigmas == video.sigmas && schedules.audio.sigmas == audio.sigmas
        }

        private static func label(_ value: Float) -> String {
            value.rounded() == value ? String(Int(value)) : String(value)
        }
    }

    package struct Entry: Codable, Hashable, Sendable {
        package let schedule: Schedule
        package let filename: String
        package let byteCount: Int64
        package let sha256: String

        enum CodingKeys: String, CodingKey {
            case schedule
            case filename
            case byteCount = "byte_count"
            case sha256
        }
    }

    package struct Selection {
        package let cache: MiniMaxH3AdaLNCache
        package let exact: Bool
        package let sourceSchedule: Schedule
    }

    package static let filename = "adaln_cache.index.json"
    package static let schemaVersion = 1
    package static let format = "mere.run.minimax-h3-adaln-cache-pack"
    package static let productionSchedules = [5, 9, 12, 16, 21, 31].map {
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

    package static func load(
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

    package static func write(entries: [Entry], sourceIdentity: String, to rootURL: URL) throws -> URL {
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

    package static func entry(
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
