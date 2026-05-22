import ArgumentParser
import Foundation
import MLX
import MereRunCore

struct MusicGenerate: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "generate",
        abstract: "Generate audio from caption + optional lyrics using ACE-Step.",
        discussion: """
        Prints the output WAV path to stdout.
        Progress and diagnostics are printed to stderr.

        Example:
          mere.run music generate "upbeat electronic groove" \
            --model music-acestep \
            --lyrics "[verse]\\nwe dance all night" \
            --text-subdirectory Qwen3-Embedding-0.6B \
            -o out.wav
        """
    )

    @Argument(help: "Caption prompt describing the target music.")
    var caption: String

    @Option(name: [.long], help: "Lyrics text (optional).")
    var lyrics: String = ""

    @Option(name: [.customLong("lyrics-file")], help: "Path to lyrics text file. Cannot be used with --lyrics.")
    var lyricsFile: String?

    @Option(name: [.customShort("o"), .long], help: "Output WAV path (default: ./mererun-music-<timestamp>.wav).")
    var output: String?

    @Option(name: [.customShort("m"), .long], help: "Managed model id or local path to the ACE-Step model root/checkpoints root.")
    var model: String = ModelResolver.ModelID.aceStep.rawValue

    @Option(name: [.customLong("checkpoints-root")], help: "Root directory containing ACE-Step checkpoint subdirectories. Auto-discovered if not set.")
    var checkpointsRoot: String?

    @Option(name: [.customLong("turbo-subdirectory")], help: "Turbo decoder subdirectory under checkpoints root.")
    var turboSubdirectory: String = "acestep-v15-turbo"

    @Option(name: [.customLong("vae-subdirectory")], help: "VAE subdirectory under checkpoints root.")
    var vaeSubdirectory: String = "vae"

    @Option(name: [.customLong("lm-subdirectory")], help: "Optional 5Hz LM subdirectory under checkpoints root. Auto-detected as 'music-acestep-5hz-lm-1.7b' when omitted.")
    var lmSubdirectory: String?

    @Option(name: [.customLong("text-subdirectory")], help: "Text encoder subdirectory under checkpoints root. Auto-detected as 'Qwen3-Embedding-0.6B' when omitted.")
    var textSubdirectory: String?

    @Flag(name: [.customLong("use-lm")], help: "Use 5Hz LM constrained decoding.")
    var useLM: Bool = false

    @Option(name: [.customLong("duration")], help: "Output duration in seconds.")
    var durationSeconds: Float = 10.0

    @Option(name: [.customShort("s"), .long], help: "Number of turbo denoise steps.")
    var steps: Int = 8

    @Option(name: [.customLong("shift")], help: "Turbo scheduler shift.")
    var shift: Float = 1.0

    @Option(name: [.long], help: "Seed for deterministic generation.")
    var seed: UInt64?

    @Option(name: [.customLong("audio-cover-strength")], help: "Cover-conditioning strength in [0, 1].")
    var audioCoverStrength: Float = 1.0

    @Option(name: [.customLong("vocal-language")], help: "Language tag used in lyric prompt formatting.")
    var vocalLanguage: String = "en"

    @Option(name: [.customLong("instruction")], help: "Caption instruction prefix.")
    var instruction: String = "Fill the audio semantic mask based on the given conditions:"

    @Option(name: [.customLong("task-type"), .customLong("task")], help: "Task type: text2music, cover, repaint, extract, lego, complete.")
    var taskType: String = "text2music"

    @Option(name: [.customLong("track-name")], help: "Track name for extract/lego tasks.")
    var trackName: String?

    @Option(name: [.customLong("complete-track-classes")], help: "Comma-separated track classes for complete task (e.g. Drums,Bass).")
    var completeTrackClasses: String?

    @Flag(name: [.customLong("non-cover")], help: "Generate as non-cover (isCover=false).")
    var nonCover: Bool = false

    @Option(name: [.customLong("bpm")], help: "Optional BPM metadata for constrained LM / prompt metadata.")
    var bpm: Int?

    @Option(name: [.customLong("keyscale"), .customLong("key")], help: "Optional key metadata (e.g. C major).")
    var keyscale: String?

    @Option(name: [.customLong("timesignature"), .customLong("timesig")], help: "Optional time signature metadata (e.g. 4).")
    var timesignature: String?

    @Option(name: [.customLong("lm-top-k")], help: "LM top-k sampling (0 = disabled).")
    var lmTopK: Int = 0

    @Option(name: [.customLong("lm-top-p")], help: "LM top-p nucleus sampling (1.0 = disabled).")
    var lmTopP: Float = 0.9

    @Option(name: [.customLong("metadata-duration")], help: "Optional metadata duration string for constrained LM.")
    var metadataDuration: String?

    @Option(name: [.customLong("metadata-language")], help: "Optional language metadata for constrained LM.")
    var metadataLanguage: String?

    @Flag(name: [.customLong("no-tiled-vae")], help: "Disable tiled VAE decode.")
    var noTiledVAE: Bool = false

    @Option(name: [.customLong("vae-chunk-size")], help: "VAE tiled decode chunk size.")
    var vaeChunkSize: Int = 512

    @Option(name: [.customLong("vae-overlap")], help: "VAE tiled decode overlap.")
    var vaeOverlap: Int = 64

    @Flag(name: [.short, .long], help: "Quiet mode (suppress stderr diagnostics).")
    var quiet: Bool = false

    func run() async throws {
        try MLXBundleSupport.ensureAvailable(quiet: quiet)

        guard durationSeconds > 0 else {
            throw ValidationError("--duration must be > 0")
        }
        guard steps > 0 else {
            throw ValidationError("--steps must be >= 1")
        }
        guard (0.0...1.0).contains(audioCoverStrength) else {
            throw ValidationError("--audio-cover-strength must be between 0.0 and 1.0")
        }
        guard lmTopK >= 0 else {
            throw ValidationError("--lm-top-k must be >= 0")
        }
        guard (0.0...1.0).contains(lmTopP) else {
            throw ValidationError("--lm-top-p must be between 0.0 and 1.0")
        }
        guard vaeChunkSize > 0 else {
            throw ValidationError("--vae-chunk-size must be > 0")
        }
        guard vaeOverlap >= 0 else {
            throw ValidationError("--vae-overlap must be >= 0")
        }
        if lyricsFile != nil, !lyrics.isEmpty {
            throw ValidationError("Pass either --lyrics or --lyrics-file, not both.")
        }

        let checkpointsRootURL = try await resolveACEStepCheckpointsRoot()
        let resolvedTurboSubdirectory = try resolveACEStepTurboSubdirectory(
            at: checkpointsRootURL,
            explicit: turboSubdirectory
        )
        let resolvedLMSubdirectory = try resolveACEStepLMSubdirectory(
            at: checkpointsRootURL,
            explicit: lmSubdirectory
        )
        let resolvedTextSubdirectory = try resolveACEStepTextSubdirectory(
            at: checkpointsRootURL,
            explicit: textSubdirectory
        )
        if useLM && resolvedLMSubdirectory == nil {
            throw ValidationError("--use-lm requires --lm-subdirectory. Set --lm-subdirectory or keep a default layout like 'music-acestep-5hz-lm-1.7b'.")
        }
        if resolvedTextSubdirectory == nil {
            throw ValidationError("ACE-Step text encoder not found. Set --text-subdirectory or keep a default layout like 'Qwen3-Embedding-0.6B'.")
        }

        let outputURL = CLIOutput.resolveOutputURL(output, defaultPrefix: "mererun-music", defaultExtension: "wav")
        try FileManager.default.createDirectory(at: outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)

        let resolvedLyrics = try loadLyrics()
        let isCover = !nonCover
        let resolvedInstruction = resolveInstruction(
            taskType: taskType,
            explicitInstruction: instruction,
            trackName: trackName,
            completeTrackClasses: completeTrackClasses
        )

        let userMetadata = ACEStep5HzLMConstrainedSampler.UserMetadata(
            bpm: bpm.map(String.init),
            caption: caption,
            duration: resolvedMetadataDuration(),
            keyscale: keyscale,
            language: metadataLanguage,
            timesignature: timesignature
        )

        if !quiet {
            CLIStderr.write("Loading ACE-Step checkpoints from \(checkpointsRootURL.path)\n")
        }

        let container = ACEStepModelContainer(
            checkpointsRootURL: checkpointsRootURL,
            turboSubdirectory: resolvedTurboSubdirectory,
            vaeSubdirectory: vaeSubdirectory,
            lmSubdirectory: useLM ? resolvedLMSubdirectory : nil,
            textEncoderSubdirectory: resolvedTextSubdirectory
        )
        let resources = try await container.resources()
        let pipeline = try ACEStepPipeline(
            decoderResources: resources.decoderResources,
            vaeResources: resources.vaeResources,
            lmResources: resources.lmResources,
            textEncoderResources: resources.textEncoderResources
        )

        let inference = ACEStepInferenceConfig(
            durationSeconds: durationSeconds,
            fixNFE: steps,
            shift: shift,
            timesteps: nil,
            inferMethod: .ode,
            useTiledVaeDecode: !noTiledVAE,
            vaeChunkSize: vaeChunkSize,
            vaeOverlap: vaeOverlap,
            seed: seed
        )

        let audio: MLXArray
        if useLM {
            if !quiet {
                CLIStderr.write("Running constrained 5Hz LM + diffusion\n")
            }
            let result = try pipeline.generatePromptToAudioWithLM(
                caption: caption,
                lyrics: resolvedLyrics,
                config: inference,
                lmConfig: .init(maxNewTokens: 4096, temperature: 0.85, topK: lmTopK, topP: lmTopP),
                lmUserMetadata: userMetadata,
                sourceLatents25Hz: nil,
                referenceTimbreLatents25Hz: nil,
                referenceTimbreAudio48kHz: nil,
                audioCoverStrength: audioCoverStrength,
                vocalLanguage: vocalLanguage,
                instruction: resolvedInstruction,
                isCover: isCover
            )
            if !quiet {
                CLIStderr.write("Generated \(result.lmResult.audioCodeValues.count) audio code tokens\n")
            }
            audio = result.audio
        } else {
            if !quiet {
                CLIStderr.write("Running direct prompt-to-audio diffusion\n")
            }
            audio = try pipeline.generatePromptToAudio(
                caption: caption,
                lyrics: resolvedLyrics,
                config: inference,
                lmUserMetadata: userMetadata,
                sourceLatents25Hz: nil,
                referenceTimbreLatents25Hz: nil,
                referenceTimbreAudio48kHz: nil,
                audioCoverStrength: audioCoverStrength,
                vocalLanguage: vocalLanguage,
                instruction: resolvedInstruction,
                isCover: isCover
            )
        }

        try ACEStepWAVWriter.writeWAV(audio, to: outputURL, sampleRate: 48_000)

        if !quiet {
            CLIStderr.write("Saved audio: \(outputURL.path)\n")
        }
        print(outputURL.path)
    }

    private func loadLyrics() throws -> String {
        if let lyricsFile {
            let lyricsURL = URL(fileURLWithPath: lyricsFile).standardizedFileURL
            guard FileManager.default.fileExists(atPath: lyricsURL.path) else {
                throw ValidationError("Lyrics file not found: \(lyricsURL.path)")
            }
            return try String(contentsOf: lyricsURL, encoding: .utf8)
        }
        return lyrics
    }

    private func resolvedMetadataDuration() -> String {
        if let metadataDuration, !metadataDuration.isEmpty {
            return metadataDuration
        }
        let rounded = max(1, Int(durationSeconds.rounded()))
        return String(rounded)
    }

    private func resolveInstruction(
        taskType: String,
        explicitInstruction: String,
        trackName: String?,
        completeTrackClasses: String?
    ) -> String {
        let defaultInstruction = "Fill the audio semantic mask based on the given conditions:"
        let trimmed = explicitInstruction.trimmingCharacters(in: .whitespacesAndNewlines)
        let normalizedTask = taskType.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()

        let taskInstruction: String
        switch normalizedTask {
        case "text2music", "":
            taskInstruction = "Fill the audio semantic mask based on the given conditions:"
        case "repaint":
            taskInstruction = "Repaint the mask area based on the given conditions:"
        case "cover":
            taskInstruction = "Generate audio semantic tokens based on the given conditions:"
        case "extract":
            if let trackName, !trackName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                taskInstruction = "Extract the \(trackName.trimmingCharacters(in: .whitespacesAndNewlines).uppercased()) track from the audio:"
            } else {
                taskInstruction = "Extract the track from the audio:"
            }
        case "lego":
            if let trackName, !trackName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                taskInstruction = "Generate the \(trackName.trimmingCharacters(in: .whitespacesAndNewlines).uppercased()) track based on the audio context:"
            } else {
                taskInstruction = "Generate the track based on the audio context:"
            }
        case "complete":
            let classes = parseCompleteTrackClasses(completeTrackClasses)
            if !classes.isEmpty {
                taskInstruction = "Complete the input track with \(classes.map { $0.uppercased() }.joined(separator: " | ")):"
            } else {
                taskInstruction = "Complete the input track:"
            }
        default:
            taskInstruction = "Fill the audio semantic mask based on the given conditions:"
        }

        let shouldAutoMapInstruction = trimmed.isEmpty || trimmed == defaultInstruction
        return shouldAutoMapInstruction ? taskInstruction : trimmed
    }

    private func parseCompleteTrackClasses(_ value: String?) -> [String] {
        let rawItems = value?
            .split { $0 == "," || $0 == "|" || $0 == "/" }
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            ?? []
        return rawItems.filter { !$0.isEmpty }
    }

    private func resolveACEStepCheckpointsRoot() async throws -> URL {
        let candidates = buildAcestepCheckpointCandidates()

        for candidate in candidates {
            if isUsableCheckpointsRoot(candidate) {
                return candidate
            }
            let nested = candidate.appendingPathComponent("checkpoints", isDirectory: true)
            if isUsableCheckpointsRoot(nested) {
                return nested
            }
        }

        if let explicit = checkpointsRoot, !explicit.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            throw ValidationError("Checkpoints root not found or incomplete: \(explicit)")
        }

        do {
            let resolved = try await ManagedModelResolver.resolveForRuntime(
                requestedModel: model,
                defaultModelID: ModelResolver.ModelID.aceStep.rawValue
            )
            let root = resolved.url
            if isUsableCheckpointsRoot(root) {
                return root
            }
            let nested = root.appendingPathComponent("checkpoints", isDirectory: true)
            if isUsableCheckpointsRoot(nested) {
                return nested
            }
        } catch let error as ManagedModelResolver.ResolverError {
            throw ValidationError(error.localizedDescription)
        }

        throw ValidationError("Music Acestep checkpoints not found. Add --checkpoints-root or set MERERUN_MUSIC_ACESTEP_ROOT.")
    }

    private func resolveACEStepLMSubdirectory(at root: URL, explicit: String?) throws -> String? {
        let fm = FileManager.default

        func isDirectory(_ url: URL) -> Bool {
            var isDir: ObjCBool = false
            return fm.fileExists(atPath: url.path, isDirectory: &isDir) && isDir.boolValue
        }

        if let explicit, !explicit.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            let explicitNormalized = explicit.trimmingCharacters(in: .whitespacesAndNewlines)
            let explicitRoot = root.appendingPathComponent(explicitNormalized, isDirectory: true)
            guard isDirectory(explicitRoot) else {
                throw ValidationError("--lm-subdirectory not found: \(explicitNormalized)")
            }
            return explicitNormalized
        }

        let preferredCandidates = [
            "music-acestep-5hz-lm-1.7b",
            "music-acestep-5Hz-lm-1.7B",
            "music-acestep-5hz-lm",
            "music-acestep-5Hz-lm",
            "lm",
            "music-acestep-lm"
        ]

        for candidate in preferredCandidates {
            if isDirectory(root.appendingPathComponent(candidate, isDirectory: true)) {
                return candidate
            }
        }

        let discovered = (try? fm.contentsOfDirectory(
            at: root,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ))?.first(where: { directory in
            let isDirectory = (try? directory.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true
            return isDirectory
                && directory.lastPathComponent.lowercased().contains("lm")
                && directory.lastPathComponent.lowercased().contains("5hz")
        })

        return discovered?.lastPathComponent
    }

    private func resolveACEStepTextSubdirectory(at root: URL, explicit: String?) throws -> String? {
        let fm = FileManager.default

        func isDirectory(_ url: URL) -> Bool {
            var isDir: ObjCBool = false
            return fm.fileExists(atPath: url.path, isDirectory: &isDir) && isDir.boolValue
        }

        if let explicit, !explicit.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            let explicitNormalized = explicit.trimmingCharacters(in: .whitespacesAndNewlines)
            let explicitRoot = root.appendingPathComponent(explicitNormalized, isDirectory: true)
            guard isDirectory(explicitRoot) else {
                throw ValidationError("--text-subdirectory not found: \(explicitNormalized)")
            }
            return explicitNormalized
        }

        let preferredCandidates = [
            "Qwen3-Embedding-0.6B",
            "qwen3-embedding-0.6b",
            "Qwen3-Embedding-4B",
            "qwen3-embedding-4b",
            "Qwen3-Embedding",
            "qwen3-embedding",
            "text_encoder",
            "text-encoder"
        ]

        for candidate in preferredCandidates {
            if isDirectory(root.appendingPathComponent(candidate, isDirectory: true)) {
                return candidate
            }
        }

        let discovered = (try? fm.contentsOfDirectory(
            at: root,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ))?.first(where: { directory in
            let isDirectory = (try? directory.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true
            let name = directory.lastPathComponent.lowercased()
            return isDirectory && (
                (name.contains("qwen3") && name.contains("embedding"))
                || name == "text_encoder"
                || name == "text-encoder"
            )
        })

        return discovered?.lastPathComponent
    }

    private func buildAcestepCheckpointCandidates() -> [URL] {
        var candidates: [URL] = []

        if let explicit = checkpointsRoot?.trimmingCharacters(in: .whitespacesAndNewlines), !explicit.isEmpty {
            candidates.append(URL(fileURLWithPath: explicit).standardizedFileURL)
        }

        if let explicitModel = model.trimmingCharacters(in: .whitespacesAndNewlines).nonEmpty {
            let url = URL(fileURLWithPath: explicitModel).standardizedFileURL
            if FileManager.default.fileExists(atPath: url.path) {
                candidates.append(url)
            }
        }

        if let envRoot = ProcessInfo.processInfo.environment["MERERUN_MUSIC_ACESTEP_ROOT"]?
            .trimmingCharacters(in: .whitespacesAndNewlines),
            !envRoot.isEmpty
        {
            candidates.append(URL(fileURLWithPath: envRoot).standardizedFileURL)
        }

        let localModelRoot = MereRunModelPaths.modelsDir
            .appendingPathComponent("music-acestep", isDirectory: true)
            .standardizedFileURL
        candidates.append(localModelRoot)
        candidates.append(localModelRoot.appendingPathComponent("checkpoints", isDirectory: true))
        return candidates
    }

    private func resolveACEStepTurboSubdirectory(at root: URL, explicit: String) throws -> String {
        let fm = FileManager.default
        func isDirectory(_ url: URL) -> Bool {
            var isDir: ObjCBool = false
            return fm.fileExists(atPath: url.path, isDirectory: &isDir) && isDir.boolValue
        }

        let trimmed = explicit.trimmingCharacters(in: .whitespacesAndNewlines)
        let upstreamDefault = "acestep-v15-turbo"
        let compatibilityDefault = "music-acestep-v15-turbo"
        let candidates = trimmed == upstreamDefault
            ? [upstreamDefault, compatibilityDefault]
            : [trimmed]

        for candidate in candidates where isDirectory(root.appendingPathComponent(candidate, isDirectory: true)) {
            return candidate
        }
        throw ValidationError("--turbo-subdirectory not found: \(trimmed)")
    }

    private func isUsableCheckpointsRoot(_ root: URL) -> Bool {
        let fm = FileManager.default

        func isDirectory(_ url: URL) -> Bool {
            var isDir: ObjCBool = false
            return fm.fileExists(atPath: url.path, isDirectory: &isDir) && isDir.boolValue
        }

        guard isDirectory(root) else {
            return false
        }
        let turboCandidates = turboSubdirectory == "acestep-v15-turbo"
            ? ["acestep-v15-turbo", "music-acestep-v15-turbo"]
            : [turboSubdirectory]
        guard turboCandidates.contains(where: { isDirectory(root.appendingPathComponent($0, isDirectory: true)) }) else {
            return false
        }
        guard isDirectory(root.appendingPathComponent(vaeSubdirectory)) else {
            return false
        }

        if useLM, let lmSubdirectory, !lmSubdirectory.isEmpty {
            guard isDirectory(root.appendingPathComponent(lmSubdirectory)) else {
                return false
            }
        }

        if let textSubdirectory, !textSubdirectory.isEmpty {
            guard isDirectory(root.appendingPathComponent(textSubdirectory)) else {
                return false
            }
        }

        return true
    }
}

private extension String {
    var nonEmpty: String? {
        isEmpty ? nil : self
    }
}

private enum ACEStepWAVWriter {
    enum WriterError: LocalizedError {
        case invalidShape([Int])
        case invalidChannels(Int)

        var errorDescription: String? {
            switch self {
            case .invalidShape(let shape):
                return "Unsupported ACE-Step audio tensor shape: \(shape). Expected [1,S,C], [S,C], [1,S], or [S]."
            case .invalidChannels(let channels):
                return "Invalid channel count \(channels). Expected 1...8."
            }
        }
    }

    static func writeWAV(_ audio: MLXArray, to url: URL, sampleRate: Int) throws {
        let (interleaved, channels) = try flattenToInterleaved(audio)
        guard (1...8).contains(channels) else {
            throw WriterError.invalidChannels(channels)
        }

        let int16Samples = interleaved.map { sample -> Int16 in
            let clamped = max(-1.0, min(1.0, sample))
            return Int16(clamped * 32767.0)
        }

        let dataSize = UInt32(int16Samples.count * MemoryLayout<Int16>.size)
        let fileSize = UInt32(36) + dataSize
        let channelsU16 = UInt16(channels)
        let sampleRateU32 = UInt32(sampleRate)
        let bitsPerSample = UInt16(16)
        let blockAlign = UInt16(channels) * (bitsPerSample / 8)
        let byteRate = sampleRateU32 * UInt32(blockAlign)

        var data = Data()
        data.append("RIFF".data(using: .utf8)!)
        append(fileSize, to: &data)
        data.append("WAVE".data(using: .utf8)!)

        data.append("fmt ".data(using: .utf8)!)
        append(UInt32(16), to: &data)  // PCM fmt chunk length
        append(UInt16(1), to: &data)  // PCM format
        append(channelsU16, to: &data)
        append(sampleRateU32, to: &data)
        append(byteRate, to: &data)
        append(blockAlign, to: &data)
        append(bitsPerSample, to: &data)

        data.append("data".data(using: .utf8)!)
        append(dataSize, to: &data)
        for sample in int16Samples {
            append(sample, to: &data)
        }

        try data.write(to: url)
    }

    private static func flattenToInterleaved(_ audio: MLXArray) throws -> ([Float], Int) {
        let sampleChannel: MLXArray
        if audio.ndim == 1 {
            sampleChannel = audio.reshaped(audio.dim(0), 1)
        } else if audio.ndim == 2 {
            if audio.dim(1) <= 8 {
                sampleChannel = audio
            } else if audio.dim(0) <= 8 {
                sampleChannel = audio.transposed(1, 0)
            } else {
                throw WriterError.invalidShape(audio.shape)
            }
        } else if audio.ndim == 3 {
            guard audio.dim(0) == 1 else {
                throw WriterError.invalidShape(audio.shape)
            }
            let squeezed = audio[0, 0..., 0...]
            if squeezed.dim(1) <= 8 {
                sampleChannel = squeezed
            } else if squeezed.dim(0) <= 8 {
                sampleChannel = squeezed.transposed(1, 0)
            } else {
                throw WriterError.invalidShape(audio.shape)
            }
        } else {
            throw WriterError.invalidShape(audio.shape)
        }

        MLX.eval(sampleChannel)
        let channels = sampleChannel.dim(1)
        let interleaved = sampleChannel.asType(.float32).reshaped(-1).asArray(Float.self)
        return (interleaved, channels)
    }

    private static func append<T: FixedWidthInteger>(_ value: T, to data: inout Data) {
        var littleEndian = value.littleEndian
        withUnsafeBytes(of: &littleEndian) { bytes in
            data.append(bytes.bindMemory(to: UInt8.self))
        }
    }
}
