import Foundation
import MLX
import MereRunCore

struct ACEStepRecipeCandidate: Codable {
    var rank: Int
    var index: Int
    var seed: UInt64
    var score: Float
    var metrics: ACEStepCandidateMetrics
    var lmAudioCodeCount: Int?
    var selected: Bool
}

struct ACEStepRecipeConditioningMetadata: Codable, Equatable {
    var bpm: String?
    var duration: String?
    var keyscale: String?
    var language: String?
    var timesignature: String?

    init(_ metadata: ACEStep5HzLMConstrainedSampler.UserMetadata) {
        bpm = metadata.bpm
        duration = metadata.duration
        keyscale = metadata.keyscale
        language = metadata.language
        timesignature = metadata.timesignature
    }
}

struct ACEStepGenerationRecipe: Codable {
    static let currentSchemaVersion = 2

    var schemaVersion: Int
    var createdAt: Date
    var modelID: String
    var checkpointVariant: ACEStepCheckpointVariant
    var decoderSubdirectory: String
    var checkpointSources: [MereRunModelManifest.SourceProvenance]
    var languageModelSubdirectory: String?
    var textEncoderSubdirectory: String
    var adapters: [ACEStepAdapterDescriptor]
    var task: ACEStepTask
    var quality: ACEStepQualityPreset
    var caption: String
    var lyrics: String
    var instruction: String
    var conditioningMetadata: ACEStepRecipeConditioningMetadata
    var inference: ACEStepInferenceConfig
    var repaint: ACEStepRepaintConfiguration?
    var flowEdit: ACEStepFlowEditConfiguration?
    var languageModelUsed: Bool
    var candidates: [ACEStepRecipeCandidate]
    var export: ACEStepAudioExportOptions
    var sourceAudioSHA256: String?
    var outputFilename: String
    var outputSHA256: String
    var lrcFilename: String?
    var lrcTimingIsApproximate: Bool?

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case createdAt = "created_at"
        case modelID = "model_id"
        case checkpointVariant = "checkpoint_variant"
        case decoderSubdirectory = "decoder_subdirectory"
        case checkpointSources = "checkpoint_sources"
        case languageModelSubdirectory = "language_model_subdirectory"
        case textEncoderSubdirectory = "text_encoder_subdirectory"
        case adapters
        case task
        case quality
        case caption
        case lyrics
        case instruction
        case conditioningMetadata = "conditioning_metadata"
        case inference
        case repaint
        case flowEdit = "flow_edit"
        case languageModelUsed = "language_model_used"
        case candidates
        case export
        case sourceAudioSHA256 = "source_audio_sha256"
        case outputFilename = "output_filename"
        case outputSHA256 = "output_sha256"
        case lrcFilename = "lrc_filename"
        case lrcTimingIsApproximate = "lrc_timing_is_approximate"
    }

    func write(to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        try encoder.encode(self).write(to: url, options: .atomic)
    }

    static func checkpointProvenance(
        modelID: String,
        manifest: MereRunModelManifest?
    ) -> [MereRunModelManifest.SourceProvenance] {
        if let sources = manifest?.sources, !sources.isEmpty {
            return sources
        }
        guard let spec = ManagedModelCatalog.spec(for: modelID) else {
            return []
        }
        var sources: [MereRunModelManifest.SourceProvenance] = []
        if let primary = spec.hubFallback {
            sources.append(
                .init(
                    role: "primary",
                    repository: primary.repoId,
                    revision: primary.revision
                )
            )
        }
        sources.append(contentsOf: spec.mountedHubFallbacks.map {
            .init(
                role: "mounted",
                repository: $0.hubFallback.repoId,
                revision: $0.hubFallback.revision,
                destinationPath: $0.destinationPath
            )
        })
        if let repository = spec.upstreamRepoId,
           !sources.contains(where: { $0.repository == repository })
        {
            sources.append(
                .init(
                    role: "upstream",
                    repository: repository,
                    revision: spec.upstreamRevision ?? "unspecified"
                )
            )
        }
        return sources
    }
}

enum ACEStepDAWBundleWriter {
    struct Track {
        var name: String
        var audio: MLXArray
    }

    static func write(
        directory: URL,
        mixURL: URL,
        recipeURL: URL,
        lrcURL: URL?,
        candidates: [Track],
        stems: [Track],
        lrc: ACEStepLRCDocument?,
        exportOptions: ACEStepAudioExportOptions
    ) throws {
        let fileManager = FileManager.default
        try fileManager.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        let audioDirectory = directory.appendingPathComponent(
            "audio",
            isDirectory: true
        )
        try fileManager.createDirectory(
            at: audioDirectory,
            withIntermediateDirectories: true
        )

        let mixDestination = audioDirectory.appendingPathComponent("mix.wav")
        try Data(contentsOf: mixURL).write(to: mixDestination, options: .atomic)
        var projectFiles: [(name: String, audioURL: URL)] = [
            (name: "Mix", audioURL: mixDestination),
        ]

        for (prefix, tracks) in [("candidate", candidates), ("stem", stems)] {
            for (index, track) in tracks.enumerated() {
                let safeName = safeFilename(track.name)
                let filename = "\(prefix)-\(index + 1)-\(safeName).wav"
                let destination = audioDirectory.appendingPathComponent(filename)
                try ACEStepWAVWriter.writeWAV(
                    track.audio,
                    to: destination,
                    sampleRate: 48_000,
                    options: exportOptions
                )
                projectFiles.append((track.name, destination))
            }
        }

        let recipeDestination = directory.appendingPathComponent("recipe.json")
        try Data(contentsOf: recipeURL).write(
            to: recipeDestination,
            options: .atomic
        )
        if let lrcURL {
            try Data(contentsOf: lrcURL).write(
                to: directory.appendingPathComponent("lyrics.lrc"),
                options: .atomic
            )
        }
        try markerCSV(lrc).write(
            to: directory.appendingPathComponent("markers.csv"),
            atomically: true,
            encoding: .utf8
        )
        try reaperProject(
            tracks: projectFiles,
            baseDirectory: directory
        ).write(
            to: directory.appendingPathComponent("mere-run-session.rpp"),
            atomically: true,
            encoding: .utf8
        )
        try readme(trackCount: projectFiles.count).write(
            to: directory.appendingPathComponent("README.md"),
            atomically: true,
            encoding: .utf8
        )
    }

    private static func markerCSV(_ lrc: ACEStepLRCDocument?) -> String {
        var rows = ["Name,Start,End,Length,Color"]
        for (index, line) in (lrc?.lines ?? []).enumerated() {
            let escaped = line.text.replacingOccurrences(of: "\"", with: "\"\"")
            let next = lrc?.lines.dropFirst(index + 1).first?.timestampSeconds
                ?? line.timestampSeconds
            rows.append(
                "\"\(escaped)\",\(line.timestampSeconds),\(next),"
                    + "\(max(0, next - line.timestampSeconds)),"
            )
        }
        return rows.joined(separator: "\n") + "\n"
    }

    private static func reaperProject(
        tracks: [(name: String, audioURL: URL)],
        baseDirectory: URL
    ) -> String {
        var lines = [
            "<REAPER_PROJECT 0.1 \"7.0\" 0",
            "  RIPPLE 0",
            "  GROUPOVERRIDE 0 0 0",
        ]
        for track in tracks {
            let relative = track.audioURL.path.replacingOccurrences(
                of: baseDirectory.path + "/",
                with: ""
            )
            lines.append(contentsOf: [
                "  <TRACK",
                "    NAME \"\(escapedProjectValue(track.name))\"",
                "    <ITEM",
                "      POSITION 0",
                "      <SOURCE WAVE",
                "        FILE \"\(escapedProjectValue(relative))\"",
                "      >",
                "    >",
                "  >",
            ])
        }
        lines.append(">")
        return lines.joined(separator: "\n") + "\n"
    }

    private static func readme(trackCount: Int) -> String {
        """
        # mere.run ACE-Step session

        This portable bundle contains \(trackCount) 48 kHz WAV track(s), the
        exact generation recipe and checkpoint revisions, synchronized lyric
        markers, and a REAPER project. Open `mere-run-session.rpp` in REAPER or
        import `audio/*.wav` and `markers.csv` into another DAW.
        """
            + "\n"
    }

    private static func safeFilename(_ value: String) -> String {
        let mapped = value.lowercased().map { character -> Character in
            character.isLetter || character.isNumber ? character : "-"
        }
        let collapsed = String(mapped)
            .replacingOccurrences(of: "--", with: "-")
            .trimmingCharacters(in: CharacterSet(charactersIn: "-"))
        return collapsed.isEmpty ? "track" : collapsed
    }

    private static func escapedProjectValue(_ value: String) -> String {
        value
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: "\"", with: "\\\"")
    }
}
