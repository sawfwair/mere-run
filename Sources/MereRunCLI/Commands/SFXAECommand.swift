import ArgumentParser
import AudioCodecs
import Foundation
import MereRunCore
import MLX

struct SFXAE: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "ae",
        abstract: "Encode or decode Woosh audio autoencoder latents.",
        subcommands: [
            SFXAEEncode.self,
            SFXAEDecode.self,
        ]
    )
}

struct SFXAEEncode: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "encode",
        abstract: "Encode audio into normalized Woosh-AE latents.",
        discussion: """
        Prints the output .npy path to stdout.
        Progress and diagnostics are printed to stderr.
        """
    )

    @Argument(help: "Input audio path.")
    var input: String

    @Option(name: [.customShort("o"), .long], help: "Output .npy path (default: ./mererun-sfx-latents-<timestamp>.npy).")
    var output: String?

    @Option(name: [.customShort("m"), .long], help: "Managed model id or local Woosh checkpoints root.")
    var model: String = ModelResolver.ModelID.wooshDFlow.rawValue

    @Flag(name: [.short, .long], help: "Quiet mode (suppress stderr diagnostics).")
    var quiet: Bool = false

    func run() async throws {
        try MLXBundleSupport.ensureAvailable(quiet: quiet)
        let inputURL = URL(fileURLWithPath: input).standardizedFileURL
        guard FileManager.default.fileExists(atPath: inputURL.path) else {
            throw ValidationError("Input audio not found: \(input)")
        }

        let outputURL = CLIOutput.resolveOutputURL(output, defaultPrefix: "mererun-sfx-latents", defaultExtension: "npy")
        try FileManager.default.createDirectory(at: outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        let resources = try await SFXWooshRuntime.resolve(model: model, quiet: quiet)
        if !quiet {
            CLIStderr.write("Loading Woosh-AE checkpoints from \(resources.autoencoderURL.path)\n")
        }

        let audio = try AudioReader.readAudioBuffer(from: inputURL, sampleRate: WooshResources.sampleRate, channels: 1)
        let autoencoder = try WooshAudioAutoEncoder.load(resources: resources)
        let latents = try autoencoder.encode(samples: audio.samples)
        MLX.eval(latents)
        try MLX.save(array: latents, url: outputURL)
        if !quiet {
            CLIStderr.write("Saved latents: \(outputURL.path)\n")
        }
        print(outputURL.path)
    }
}

struct SFXAEDecode: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "decode",
        abstract: "Decode normalized Woosh-AE latents into audio.",
        discussion: """
        Prints the output WAV path to stdout.
        Progress and diagnostics are printed to stderr.
        """
    )

    @Argument(help: "Input .npy latent path with shape [1, 128, frames].")
    var input: String

    @Option(name: [.customShort("o"), .long], help: "Output WAV path (default: ./mererun-sfx-ae-<timestamp>.wav).")
    var output: String?

    @Option(name: [.customShort("m"), .long], help: "Managed model id or local Woosh checkpoints root.")
    var model: String = ModelResolver.ModelID.wooshDFlow.rawValue

    @Flag(name: [.short, .long], help: "Quiet mode (suppress stderr diagnostics).")
    var quiet: Bool = false

    func run() async throws {
        try MLXBundleSupport.ensureAvailable(quiet: quiet)
        let inputURL = URL(fileURLWithPath: input).standardizedFileURL
        guard FileManager.default.fileExists(atPath: inputURL.path) else {
            throw ValidationError("Input latents not found: \(input)")
        }

        let outputURL = CLIOutput.resolveOutputURL(output, defaultPrefix: "mererun-sfx-ae", defaultExtension: "wav")
        try FileManager.default.createDirectory(at: outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        let resources = try await SFXWooshRuntime.resolve(model: model, quiet: quiet)
        if !quiet {
            CLIStderr.write("Loading Woosh-AE checkpoints from \(resources.autoencoderURL.path)\n")
        }

        let latents = try MLX.loadArray(url: inputURL).asType(.float32)
        guard latents.ndim == 3,
              latents.dim(0) == 1,
              latents.dim(1) == WooshResources.latentChannels else {
            throw ValidationError("Expected latents with shape [1, \(WooshResources.latentChannels), frames]; got \(latents.shape)")
        }
        let autoencoder = try WooshAudioAutoEncoder.load(resources: resources)
        let samples = try autoencoder.decode(latents)
        try SFXWAVWriter.writeMonoPCM16(samples: samples, to: outputURL, sampleRate: WooshResources.sampleRate)
        if !quiet {
            CLIStderr.write("Saved audio: \(outputURL.path)\n")
        }
        print(outputURL.path)
    }
}

enum SFXWooshRuntime {
    static func resolve(model: String, quiet: Bool) async throws -> WooshModelResources {
        let resolution = try await ManagedModelResolver.resolveForRuntime(
            requestedModel: model,
            defaultModelID: ModelResolver.ModelID.wooshDFlow.rawValue,
            progress: { event in
                guard !quiet else { return }
                switch event {
                case .downloading(let percent):
                    CLIStderr.write("Downloading Woosh assets... \(percent)%\n")
                case .extracting:
                    CLIStderr.write("Extracting Woosh assets...\n")
                }
            }
        )
        let checkpointsRoot = WooshResources.normalizeRoot(resolution.url)
        guard let variant = WooshVariant.resolve(model: model, rootURL: checkpointsRoot) else {
            throw ValidationError("Woosh checkpoints not found under \(resolution.url.path)")
        }
        let resources = WooshModelResources(checkpointsRootURL: checkpointsRoot, variant: variant)
        let missing = resources.missingFiles()
        guard missing.isEmpty else {
            throw ValidationError(WooshError.missingFiles(missing).localizedDescription)
        }
        return resources
    }
}
