import ArgumentParser
import AudioCodecs
import Foundation
import MereRunCore
import MLX

struct SFXCLAP: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "clap",
        abstract: "Embed and score sound effects with Woosh-CLAP.",
        subcommands: [
            SFXCLAPScoreCommand.self,
        ]
    )
}

struct SFXCLAPScoreCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "score",
        abstract: "Score a text prompt against an audio file.",
        discussion: """
        Prints a JSON object to stdout.
        Progress and diagnostics are printed to stderr.
        """
    )

    @Argument(help: "Prompt describing the sound effect.")
    var prompt: String

    @Argument(help: "Input audio path.")
    var audio: String

    @Option(name: [.customShort("m"), .long], help: "Managed model id or local Woosh checkpoints root.")
    var model: String = ModelResolver.ModelID.wooshClap.rawValue

    @Flag(name: [.short, .long], help: "Quiet mode (suppress stderr diagnostics).")
    var quiet: Bool = false

    func run() async throws {
        try MLXBundleSupport.ensureAvailable(quiet: quiet)
        let audioURL = URL(fileURLWithPath: audio).standardizedFileURL
        guard FileManager.default.fileExists(atPath: audioURL.path) else {
            throw ValidationError("Input audio not found: \(audio)")
        }

        let resources = try await SFXCLAPRuntime.resolve(model: model, quiet: quiet)
        if !quiet {
            CLIStderr.write("Loading Woosh-CLAP checkpoints from \(resources.clapURL.path)\n")
        }
        let samples = try AudioReader.readAudioBuffer(
            from: audioURL,
            sampleRate: 32_000,
            channels: 1
        ).samples
        let clap = try WooshCLAP(resources: resources)
        let result = try clap.score(text: prompt, audioSamples: samples)
        let output = SFXCLAPScoreOutput(
            score: result.score,
            prompt: prompt,
            audio: audioURL.path,
            model: model
        )
        let data = try JSONEncoder().encode(output)
        print(String(decoding: data, as: UTF8.self))
    }
}

private struct SFXCLAPScoreOutput: Encodable {
    let score: Float
    let prompt: String
    let audio: String
    let model: String
}

enum SFXCLAPRuntime {
    static func resolve(model: String, quiet: Bool) async throws -> WooshCLAPResources {
        let resolution = try await ManagedModelResolver.resolveForRuntime(
            requestedModel: model,
            defaultModelID: ModelResolver.ModelID.wooshClap.rawValue,
            progress: { event in
                guard !quiet else { return }
                switch event {
                case .downloading(let percent):
                    CLIStderr.write("Downloading Woosh-CLAP assets... \(percent)%\n")
                case .extracting:
                    CLIStderr.write("Extracting Woosh-CLAP assets...\n")
                }
            }
        )
        let checkpointsRoot = WooshResources.normalizeRoot(resolution.url)
        let resources = WooshCLAPResources(checkpointsRootURL: checkpointsRoot)
        let missing = resources.missingFiles()
        guard missing.isEmpty else {
            throw ValidationError(WooshError.missingFiles(missing).localizedDescription)
        }
        return resources
    }
}
