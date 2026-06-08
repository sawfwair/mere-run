import ArgumentParser
import Foundation
import MLX
import MereRunCore

struct MusicAnalyze: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "analyze",
        abstract: "Analyze source audio with ACE-Step audio understanding.",
        discussion: """
        Prints JSON metadata to stdout.
        Progress and diagnostics are printed to stderr.

        Example:
          mere.run music analyze ~/Downloads/song.mp3 \
            --model music-acestep-xl-turbo-lm4b \
            --lm-subdirectory acestep-5Hz-lm-4B
        """
    )

    @Argument(help: "Input audio file to analyze.")
    var audio: String

    @Option(name: [.customShort("m"), .long], help: "Managed ACE-Step model id or local model/checkpoints root.")
    var model: String = ModelResolver.ModelID.aceStep.rawValue

    @Option(name: [.customLong("checkpoints-root")], help: "Root directory containing ACE-Step checkpoint subdirectories. Auto-discovered if not set.")
    var checkpointsRoot: String?

    @Option(name: [.customLong("turbo-subdirectory")], help: "Turbo decoder subdirectory under checkpoints root.")
    var turboSubdirectory: String = "acestep-v15-turbo"

    @Option(name: [.customLong("vae-subdirectory")], help: "VAE subdirectory under checkpoints root.")
    var vaeSubdirectory: String = "vae"

    @Option(name: [.customLong("lm-subdirectory")], help: "5Hz LM subdirectory under checkpoints root. Auto-detected when omitted.")
    var lmSubdirectory: String?

    @Option(name: [.customLong("duration")], help: "Analyze the first N seconds instead of the full decoded input.")
    var durationSeconds: Float?

    @Option(name: [.customLong("max-new-tokens")], help: "Maximum LM tokens to emit during analysis.")
    var maxNewTokens: Int = 2048

    @Option(name: [.customLong("lm-temperature")], help: "LM sampling temperature.")
    var lmTemperature: Float = 0.3

    @Option(name: [.customLong("lm-top-k")], help: "LM top-k sampling (0 = disabled).")
    var lmTopK: Int = 0

    @Option(name: [.customLong("lm-top-p")], help: "LM top-p nucleus sampling (1.0 = disabled).")
    var lmTopP: Float = 0.9

    @Flag(name: [.customLong("include-raw-lm")], help: "Include the raw LM analysis text in stdout JSON.")
    var includeRawLM: Bool = false

    @Flag(name: [.customLong("include-audio-codes")], help: "Include serialized ACE audio-code tokens in stdout JSON.")
    var includeAudioCodes: Bool = false

    @Flag(name: [.short, .long], help: "Quiet mode (suppress stderr diagnostics).")
    var quiet: Bool = false

    func run() async throws {
        try MLXBundleSupport.ensureAvailable(quiet: quiet)

        if let durationSeconds {
            guard durationSeconds > 0 else {
                throw ValidationError("--duration must be > 0")
            }
        }
        guard maxNewTokens > 0 else {
            throw ValidationError("--max-new-tokens must be > 0")
        }
        guard lmTopK >= 0 else {
            throw ValidationError("--lm-top-k must be >= 0")
        }
        guard (0.0...1.0).contains(lmTopP) else {
            throw ValidationError("--lm-top-p must be between 0.0 and 1.0")
        }
        guard lmTemperature >= 0 else {
            throw ValidationError("--lm-temperature must be >= 0")
        }

        let checkpointsRootURL = try await ACEStepCLIHelper.resolveCheckpointsRoot(
            model: model,
            checkpointsRoot: checkpointsRoot,
            turboSubdirectory: turboSubdirectory,
            vaeSubdirectory: vaeSubdirectory,
            lmSubdirectory: lmSubdirectory,
            textSubdirectory: nil
        )
        let resolvedTurboSubdirectory = try ACEStepCLIHelper.resolveTurboSubdirectory(
            at: checkpointsRootURL,
            explicit: turboSubdirectory
        )
        let resolvedLMSubdirectory = try ACEStepCLIHelper.resolveLMSubdirectory(
            at: checkpointsRootURL,
            explicit: lmSubdirectory
        )
        guard let resolvedLMSubdirectory else {
            throw ValidationError(
                "music analyze requires an ACE-Step 5Hz LM subdirectory. "
                    + "Pull `music-acestep` or `music-acestep-xl-turbo-lm4b`, "
                    + "or pass --lm-subdirectory."
            )
        }

        let inputURL = ACEStepCLIHelper.resolveUserPath(audio)
        let audio48kHz = try ACEStepCLIHelper.loadAudio48kHz(audio, label: "audio")
        let inputDurationSeconds = ACEStepCLIHelper.durationSeconds(of: audio48kHz, fallback: 0)
        let analyzedDurationSeconds = durationSeconds ?? inputDurationSeconds

        if !quiet {
            CLIStderr.write("Using ACE-Step source audio: \(inputURL.path)\n")
            CLIStderr.write("Loading ACE-Step checkpoints from \(checkpointsRootURL.path)\n")
            CLIStderr.write("Analyzing ACE-Step source audio with 5Hz LM\n")
        }

        let container = ACEStepModelContainer(
            checkpointsRootURL: checkpointsRootURL,
            turboSubdirectory: resolvedTurboSubdirectory,
            vaeSubdirectory: vaeSubdirectory,
            lmSubdirectory: resolvedLMSubdirectory,
            textEncoderSubdirectory: nil
        )
        let resources = try await container.resources()
        let pipeline = try ACEStepPipeline(
            decoderResources: resources.decoderResources,
            vaeResources: resources.vaeResources,
            lmResources: resources.lmResources,
            textEncoderResources: nil
        )
        let analysis = try pipeline.understandSourceAudio(
            sourceAudio48kHz: audio48kHz,
            durationSeconds: analyzedDurationSeconds,
            lmConfig: .init(
                maxNewTokens: maxNewTokens,
                temperature: lmTemperature,
                topK: lmTopK,
                topP: lmTopP
            )
        )

        if !quiet {
            CLIStderr.write("ACE-Step source analysis: \(analysis.metadata.understandingSummary)\n")
        }

        let output = MusicAnalyzeOutput(
            audio: inputURL.path,
            model: model,
            checkpointsRoot: checkpointsRootURL.path,
            turboSubdirectory: resolvedTurboSubdirectory,
            lmSubdirectory: resolvedLMSubdirectory,
            inputDurationSeconds: inputDurationSeconds,
            analyzedDurationSeconds: analyzedDurationSeconds,
            metadata: analysis.metadata,
            rawLMOutput: includeRawLM ? analysis.lmResult.generatedText : nil,
            audioCodes: includeAudioCodes ? analysis.audioCodes : nil
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(output)
        guard let rendered = String(data: data, encoding: .utf8) else {
            throw ValidationError("Could not encode music analysis JSON as UTF-8.")
        }
        print(rendered)
    }
}

struct MusicAnalyzeOutput: Codable, Equatable {
    let audio: String
    let model: String
    let checkpointsRoot: String
    let turboSubdirectory: String
    let lmSubdirectory: String
    let inputDurationSeconds: Float
    let analyzedDurationSeconds: Float
    let metadata: ACEStepMusicUnderstandingMetadata
    let rawLMOutput: String?
    let audioCodes: String?
}
