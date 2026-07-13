import ArgumentParser
import Foundation
import MereRunCore
import MLX

struct SFXVideo: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "video",
        abstract: "Generate sound effects from video conditioning.",
        subcommands: [
            SFXVideoGenerate.self,
        ]
    )
}

struct SFXVideoGenerate: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "generate",
        abstract: "Generate an 8-second sound effect from a video or Synchformer features.",
        discussion: """
        The input can be a video file or a .npy Synchformer tensor with shape [frames, 768]
        or [1, frames, 768].
        Prints the output WAV path to stdout.
        """
    )

    @Argument(help: "Prompt describing the video sound.")
    var prompt: String

    @Option(name: [.customLong("negative-prompt")], help: "Negative text conditioning (MMAudio only).")
    var negativePrompt: String = ""

    @Argument(help: "Input video file, or Synchformer .npy features.")
    var input: String

    @Option(name: [.customShort("o"), .long], help: "Output WAV path (default: ./mererun-sfx-video-<timestamp>.wav).")
    var output: String?

    @Option(name: [.customShort("m"), .long], help: "Managed model id or local Woosh VFlow/DVFlow checkpoints root.")
    var model: String = ModelResolver.ModelID.wooshDVFlow8s.rawValue

    @Option(name: [.customLong("synchformer-model")], help: "Managed Synchformer model id, local model root, or safetensors path for raw video input.")
    var synchformerModel: String = ModelResolver.ModelID.wooshSynchformer.rawValue

    @Option(name: [.customLong("duration")], help: "Output duration in seconds.")
    var durationSeconds: Float = 8.0

    @Option(name: [.customShort("s"), .long], help: "Number of denoise steps. DVFlow usually uses 4; VFlow usually uses 32.")
    var steps: Int?

    @Option(name: [.customLong("cfg")], help: "Woosh guidance scale. Defaults to 3.0 for DVFlow and 4.5 for VFlow.")
    var guidanceScale: Float?

    @Option(name: [.long], help: "Seed for deterministic generation.")
    var seed: UInt64?

    @Option(name: [.customLong("renoise")], help: "Renoise amount or comma-separated DVFlow schedule in [0, 1].")
    var renoise: String?

    @Option(name: [.customLong("sync-batch-size")], help: "Synchformer segment batch size for raw video input.")
    var syncBatchSize: Int = 1

    @Option(name: [.customLong("clip-batch-size")], help: "MMAudio CLIP video-frame batch size.")
    var clipBatchSize: Int = 4

    @Flag(name: [.customLong("preflight")], help: "Inspect the SFX video request without loading MLX or generating audio.")
    var preflight: Bool = false

    @Flag(name: [.customLong("json")], help: "With --preflight, emit a structured JSON report.")
    var json: Bool = false

    @Flag(name: [.short, .long], help: "Quiet mode (suppress stderr diagnostics).")
    var quiet: Bool = false

    func run() async throws {
        if json && !preflight {
            throw ValidationError("--json is only supported with --preflight for sfx video generate.")
        }

        let inputURL = URL(fileURLWithPath: input).standardizedFileURL
        let outputURL = CLIOutput.resolveOutputURL(output, defaultPrefix: "mererun-sfx-video", defaultExtension: "wav")
        if preflight {
            try runPreflight(inputURL: inputURL, outputURL: outputURL)
            return
        }

        try MLXBundleSupport.ensureAvailable(quiet: quiet)
        guard durationSeconds > 0 else {
            throw ValidationError("--duration must be > 0")
        }
        if let steps {
            guard steps > 0 else {
                throw ValidationError("--steps must be >= 1")
            }
        }
        if let guidanceScale {
            guard guidanceScale >= 0 else {
                throw ValidationError("--cfg must be >= 0")
            }
        }
        guard syncBatchSize > 0, clipBatchSize > 0 else {
            throw ValidationError("--sync-batch-size and --clip-batch-size must be >= 1")
        }

        guard FileManager.default.fileExists(atPath: inputURL.path) else {
            throw ValidationError("Input not found: \(input)")
        }
        try FileManager.default.createDirectory(at: outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)

        if SFXMMAudioRuntime.isMMAudio(model: model) {
            try await runMMAudio(inputURL: inputURL, outputURL: outputURL)
            return
        }
        guard negativePrompt.isEmpty else {
            throw ValidationError("--negative-prompt is only supported by MMAudio models.")
        }

        let resolved = try await SFXWooshRuntime.resolve(model: model, quiet: quiet)
        guard resolved.variant == .vflow8s || resolved.variant == .dvflow8s else {
            throw ValidationError("sfx video generate requires \(ModelResolver.ModelID.wooshVFlow8s.rawValue) or \(ModelResolver.ModelID.wooshDVFlow8s.rawValue)")
        }
        let stepCount = steps ?? resolved.variant.defaultSteps
        let cfg = guidanceScale ?? (resolved.variant == .dvflow8s ? 3.0 : resolved.variant.defaultGuidanceScale)
        let renoiseSchedule = try parseRenoiseSchedule(steps: stepCount)

        if !quiet {
            CLIStderr.write("Loading Woosh V2A checkpoints from \(resolved.checkpointsRootURL.path)\n")
        }
        let generator = try WooshGenerator(resources: resolved)
        let videoFeatures = try await loadVideoFeaturesOrExtract(from: inputURL)
        let result = try generator.generateVideo(
            prompt: prompt,
            videoFeatures: videoFeatures,
            config: WooshDenoiseConfig(
                durationSeconds: durationSeconds,
                steps: stepCount,
                guidanceScale: cfg,
                seed: seed,
                renoiseSchedule: renoiseSchedule
            ),
            progress: { completed, total in
                guard !quiet else { return }
                CLIStderr.write("Generated Woosh V2A step \(completed)/\(total)\n")
            }
        )

        try SFXWAVWriter.writeMonoPCM16(samples: result.samples, to: outputURL, sampleRate: result.sampleRate)
        if !quiet {
            CLIStderr.write("Saved audio: \(outputURL.path)\n")
        }
        print(outputURL.path)
    }

    private func runMMAudio(inputURL: URL, outputURL: URL) async throws {
        guard inputURL.pathExtension.lowercased() != "npy" else {
            throw ValidationError("MMAudio video generation requires the original video, not Synchformer-only .npy features.")
        }
        guard renoise == nil else {
            throw ValidationError("--renoise is only supported by Woosh models.")
        }
        let resources = try await SFXMMAudioRuntime.resolve(model: model, quiet: quiet)
        let stepCount = steps ?? MMAudioResources.defaultSteps
        let cfg = guidanceScale ?? MMAudioResources.defaultGuidanceScale
        if !quiet {
            CLIStderr.write("Loading native MMAudio assets from \(resources.rootURL.path)\n")
        }
        let generator = try MMAudioGenerator(resources: resources)
        let result = try await generator.generateVideo(
            prompt: prompt,
            negativePrompt: negativePrompt,
            videoURL: inputURL,
            config: MMAudioGenerationConfig(
                durationSeconds: durationSeconds,
                steps: stepCount,
                guidanceScale: cfg,
                seed: seed
            ),
            clipBatchSize: clipBatchSize,
            syncBatchSize: syncBatchSize,
            progress: { completed, total in
                guard !quiet else { return }
                CLIStderr.write("Generated MMAudio step \(completed)/\(total)\n")
            }
        )
        try SFXWAVWriter.writeMonoPCM16(
            samples: result.samples,
            to: outputURL,
            sampleRate: result.sampleRate
        )
        if !quiet {
            CLIStderr.write("Saved audio: \(outputURL.path)\n")
        }
        print(outputURL.path)
    }

    func parseRenoiseSchedule(steps: Int) throws -> [Float] {
        guard let renoise, !renoise.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return []
        }
        let values = renoise
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        guard !values.isEmpty else {
            return []
        }
        let parsed = try values.map { value -> Float in
            guard let parsed = Float(value) else {
                throw ValidationError("--renoise must be a float or comma-separated floats.")
            }
            guard (0...1).contains(parsed) else {
                throw ValidationError("--renoise values must be between 0 and 1.")
            }
            return parsed
        }
        if parsed.count != 1 && parsed.count != steps {
            throw ValidationError("--renoise must contain one value or exactly --steps values.")
        }
        return parsed
    }

    private func loadVideoFeaturesOrExtract(from url: URL) async throws -> MLXArray {
        if url.pathExtension.lowercased() == "npy" {
            return try loadVideoFeatures(url: url)
        }

        let resources = try await resolveSynchformerResources()
        if !quiet {
            CLIStderr.write("Extracting Woosh Synchformer video features from \(url.path)\n")
        }
        let synchformer = try WooshSynchformer(resources: resources)
        return try synchformer.extractFeatures(
            videoURL: url,
            durationSeconds: durationSeconds,
            targetFrameRate: 24,
            segmentBatchSize: syncBatchSize
        )
    }

    private func resolveSynchformerResources() async throws -> WooshSynchformerResources {
        let explicitURL = URL(fileURLWithPath: synchformerModel).standardizedFileURL
        if FileManager.default.fileExists(atPath: explicitURL.path) {
            let resources = WooshSynchformerResources(rootURL: explicitURL)
            let missing = resources.missingFiles()
            guard missing.isEmpty else {
                throw ValidationError(WooshError.missingFiles(missing).localizedDescription)
            }
            return resources
        }

        let resolution = try await ManagedModelResolver.resolveForRuntime(
            requestedModel: synchformerModel,
            defaultModelID: ModelResolver.ModelID.wooshSynchformer.rawValue,
            progress: { event in
                guard !quiet else { return }
                switch event {
                case .downloading(let percent):
                    CLIStderr.write("Downloading Woosh Synchformer assets... \(percent)%\n")
                case .extracting:
                    CLIStderr.write("Extracting Woosh Synchformer assets...\n")
                }
            }
        )
        let resources = WooshSynchformerResources(rootURL: resolution.url)
        let missing = resources.missingFiles()
        guard missing.isEmpty else {
            throw ValidationError(WooshError.missingFiles(missing).localizedDescription)
        }
        return resources
    }

    private func loadVideoFeatures(url: URL) throws -> MLXArray {
        var features = try MLX.loadArray(url: url).asType(.float32)
        if features.ndim == 2, features.dim(1) == 768 {
            features = features.expandedDimensions(axis: 0)
        }
        guard features.ndim == 3, features.dim(0) == 1, features.dim(2) == 768 else {
            throw ValidationError("Expected video features with shape [frames, 768] or [1, frames, 768]; got \(features.shape)")
        }
        return features
    }

    func makePreflightEnvelope(
        inputURL: URL,
        outputURL: URL,
        fileManager: FileManager = .default,
        now: @escaping () -> Date = Date.init
    ) -> SFXVideoPreflightEnvelope {
        let preflightInput = SFXVideoPreflightInput(
            prompt: prompt,
            inputURL: inputURL,
            outputURL: outputURL,
            model: model,
            synchformerModel: synchformerModel,
            durationSeconds: durationSeconds,
            steps: steps,
            guidanceScale: guidanceScale,
            seed: seed,
            renoise: renoise,
            syncBatchSize: syncBatchSize,
            generationArgv: generationActionArguments(inputURL: inputURL, outputURL: outputURL),
            cwd: fileManager.currentDirectoryPath
        )
        return SFXVideoPreflightAnalyzer(
            input: preflightInput,
            fileManager: fileManager,
            now: now
        ).envelope()
    }

    private func runPreflight(inputURL: URL, outputURL: URL) throws {
        let envelope = makePreflightEnvelope(inputURL: inputURL, outputURL: outputURL)
        if json {
            print(try StructuredRunOutput.encode(envelope))
        } else {
            print(envelope.summary)
            for diagnostic in envelope.diagnostics {
                print("[\(diagnostic.severity.rawValue)] \(diagnostic.title): \(diagnostic.message)")
            }
        }
        if envelope.status == .blocked {
            throw ExitCode.failure
        }
    }

    private func generationActionArguments(inputURL: URL, outputURL: URL) -> [String] {
        var args = [
            "mere.run",
            "sfx",
            "video",
            "generate",
            prompt,
            inputURL.path,
            "--output",
            outputURL.path,
            "--model",
            model,
            "--synchformer-model",
            synchformerModel,
            "--duration",
            String(durationSeconds),
            "--sync-batch-size",
            String(syncBatchSize),
        ]
        if let steps {
            args += ["--steps", String(steps)]
        }
        if let guidanceScale {
            args += ["--cfg", String(guidanceScale)]
        }
        if let seed {
            args += ["--seed", String(seed)]
        }
        if let renoise {
            args += ["--renoise", renoise]
        }
        if quiet {
            args.append("--quiet")
        }
        return args
    }
}
