import ArgumentParser
import Foundation
import MereRunCore

struct SFXGenerate: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "generate",
        abstract: "Generate a sound effect from a text prompt.",
        discussion: """
        Prints the output WAV path to stdout.
        Progress and diagnostics are printed to stderr.

        Example:
          mere.run sfx generate "metal clang on concrete" -o clang.wav
        """
    )

    @Argument(help: "Prompt describing the target sound effect.")
    var prompt: String

    @Option(name: [.customLong("negative-prompt")], help: "Negative text conditioning (MMAudio only).")
    var negativePrompt: String = ""

    @Option(name: [.customShort("o"), .long], help: "Output WAV path (default: ./mererun-sfx-<timestamp>.wav).")
    var output: String?

    @Option(name: [.customShort("m"), .long], help: "Managed model id or local Woosh/MMAudio root.")
    var model: String = ModelResolver.ModelID.wooshDFlow.rawValue

    @Option(name: [.customLong("duration")], help: "Output duration in seconds (default: 5 for Woosh, 8 for MMAudio).")
    var durationOverride: Float?

    @Option(name: [.customShort("s"), .customLong("steps")], help: "Denoise steps (default: 4 for Woosh DFlow, 25 for MMAudio).")
    var stepsOverride: Int?

    @Option(name: [.customLong("cfg")], help: "Classifier-free guidance scale.")
    var guidanceScale: Float = 4.5

    @Option(name: [.long], help: "Seed for deterministic generation.")
    var seed: UInt64?

    @Option(name: [.customLong("renoise")], help: "Renoise amount or comma-separated DFlow schedule in [0, 1].")
    var renoise: String?

    @Flag(name: [.short, .long], help: "Quiet mode (suppress stderr diagnostics).")
    var quiet: Bool = false

    var durationSeconds: Float {
        durationOverride ?? (SFXMMAudioRuntime.isMMAudio(model: model)
            ? MMAudioResources.defaultDurationSeconds
            : 5)
    }

    var steps: Int {
        stepsOverride ?? (SFXMMAudioRuntime.isMMAudio(model: model)
            ? MMAudioResources.defaultSteps
            : 4)
    }

    func run() async throws {
        try MLXBundleSupport.ensureAvailable(quiet: quiet)
        guard durationSeconds > 0 else {
            throw ValidationError("--duration must be > 0")
        }
        guard steps > 0 else {
            throw ValidationError("--steps must be >= 1")
        }
        guard guidanceScale >= 0 else {
            throw ValidationError("--cfg must be >= 0")
        }
        if SFXMMAudioRuntime.isMMAudio(model: model) {
            try await runMMAudio()
            return
        }
        guard negativePrompt.isEmpty else {
            throw ValidationError("--negative-prompt is only supported by MMAudio models.")
        }
        let renoiseSchedule = try parseRenoiseSchedule()

        let outputURL = CLIOutput.resolveOutputURL(output, defaultPrefix: "mererun-sfx", defaultExtension: "wav")
        try FileManager.default.createDirectory(at: outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)

        let resolved = try await resolveWooshResources()
        if !quiet {
            CLIStderr.write("Loading Woosh checkpoints from \(resolved.checkpointsRootURL.path)\n")
        }
        let generator = try WooshGenerator(resources: resolved)
        let result = try generator.generate(
            prompt: prompt,
            config: WooshDenoiseConfig(
                durationSeconds: durationSeconds,
                steps: steps,
                guidanceScale: guidanceScale,
                seed: seed,
                renoiseSchedule: renoiseSchedule
            ),
            progress: { completed, total in
                guard !quiet else { return }
                CLIStderr.write("Generated Woosh step \(completed)/\(total)\n")
            }
        )

        try SFXWAVWriter.writeMonoPCM16(samples: result.samples, to: outputURL, sampleRate: result.sampleRate)
        if !quiet {
            CLIStderr.write("Saved audio: \(outputURL.path)\n")
        }
        print(outputURL.path)
    }

    private func runMMAudio() async throws {
        guard renoise == nil else {
            throw ValidationError("--renoise is only supported by Woosh models.")
        }
        let outputURL = CLIOutput.resolveOutputURL(
            output,
            defaultPrefix: "mererun-mmaudio",
            defaultExtension: "wav"
        )
        try FileManager.default.createDirectory(
            at: outputURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let resources = try await SFXMMAudioRuntime.resolve(model: model, quiet: quiet)
        if !quiet {
            CLIStderr.write("Loading native MMAudio assets from \(resources.rootURL.path)\n")
        }
        let generator = try MMAudioGenerator(resources: resources)
        let result = try await generator.generateText(
            prompt: prompt,
            negativePrompt: negativePrompt,
            config: MMAudioGenerationConfig(
                durationSeconds: durationSeconds,
                steps: steps,
                guidanceScale: guidanceScale,
                seed: seed
            ),
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

    func parseRenoiseSchedule() throws -> [Float] {
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

    private func resolveWooshResources() async throws -> WooshModelResources {
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

enum SFXMMAudioRuntime {
    static func isMMAudio(model: String, fileManager: FileManager = .default) -> Bool {
        if model == MMAudioResources.modelID {
            return true
        }
        let root = URL(fileURLWithPath: model).standardizedFileURL
        return fileManager.fileExists(
            atPath: root.appendingPathComponent(MMAudioResources.networkFilename).path
        )
    }

    static func resolve(model: String, quiet: Bool) async throws -> MMAudioModelResources {
        let explicit = URL(fileURLWithPath: model).standardizedFileURL
        if FileManager.default.fileExists(atPath: explicit.path), isMMAudio(model: model) {
            let resources = MMAudioModelResources(rootURL: explicit)
            let missing = resources.validate()
            guard missing.isEmpty else {
                throw ValidationError(MMAudioGeneratorError.missingFiles(missing).localizedDescription)
            }
            return resources
        }
        let resolution = try await ManagedModelResolver.resolveForRuntime(
            requestedModel: model,
            defaultModelID: MMAudioResources.modelID,
            progress: { event in
                guard !quiet else { return }
                switch event {
                case .downloading(let percent):
                    CLIStderr.write("Downloading MMAudio assets... \(percent)%\n")
                case .extracting:
                    CLIStderr.write("Extracting MMAudio assets...\n")
                }
            }
        )
        let resources = MMAudioModelResources(rootURL: resolution.url)
        let missing = resources.validate()
        guard missing.isEmpty else {
            throw ValidationError(MMAudioGeneratorError.missingFiles(missing).localizedDescription)
        }
        return resources
    }
}

enum SFXWAVWriter {
    static func writeMonoPCM16(samples: [Float], to url: URL, sampleRate: Int) throws {
        let normalized = peakNormalized(samples)
        let int16Samples = normalized.map { sample -> Int16 in
            let finiteSample = sample.isFinite ? sample : 0
            let clamped = max(-1, min(1, finiteSample))
            return Int16(clamped * 32767)
        }

        let dataSize = UInt32(int16Samples.count * MemoryLayout<Int16>.size)
        let fileSize = UInt32(36) + dataSize
        let channels = UInt16(1)
        let bitsPerSample = UInt16(16)
        let blockAlign = channels * (bitsPerSample / 8)
        let byteRate = UInt32(sampleRate) * UInt32(blockAlign)

        var data = Data()
        data.append("RIFF".data(using: .utf8)!)
        append(fileSize, to: &data)
        data.append("WAVE".data(using: .utf8)!)
        data.append("fmt ".data(using: .utf8)!)
        append(UInt32(16), to: &data)
        append(UInt16(1), to: &data)
        append(channels, to: &data)
        append(UInt32(sampleRate), to: &data)
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

    private static func peakNormalized(_ samples: [Float]) -> [Float] {
        var peak: Float = 0
        for sample in samples where sample.isFinite {
            peak = max(peak, abs(sample))
        }
        guard peak > 1 else {
            return samples
        }
        return samples.map { sample in
            sample.isFinite ? sample / peak : 0
        }
    }

    private static func append<T: FixedWidthInteger>(_ value: T, to data: inout Data) {
        var littleEndian = value.littleEndian
        withUnsafeBytes(of: &littleEndian) { bytes in
            data.append(bytes.bindMemory(to: UInt8.self))
        }
    }
}
