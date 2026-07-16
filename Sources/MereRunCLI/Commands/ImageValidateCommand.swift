import ArgumentParser
import Foundation
import MLX
import MLXRandom
import MereRunCore

/// Owns CLI parsing and suite orchestration for image validation.
/// The concrete validation families live in companion files so readers can
/// follow the command flow first, then drop into VAE, encoder, transformer,
/// and pipeline details only when they need them.
struct ImageValidate: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "validate",
        abstract: "Run advanced deterministic validation for local image model families.",
        discussion: """
        This command exercises the local image stack with deterministic checks for:
        - VAE decoding and reconstruction
        - text encoder output shape and embedding quality
        - transformer denoising behavior
        - full end-to-end generation with a fixed seed

        Use --save-reference to record artifacts for future comparisons.
        Use --compare to compare a fresh run against an existing reference set.
        """
    )

    enum TestType: String, ExpressibleByArgument, CaseIterable {
        case vae = "vae"
        case encoder = "encoder"
        case transformer = "transformer"
        case pipeline = "pipeline"
        case all = "all"
    }

    @Option(name: [.customShort("t"), .long], help: "Test type: vae, encoder, transformer, pipeline, or all")
    var test: TestType = .all

    @Option(name: [.customShort("m"), .long], help: "Image family: zimage or klein")
    var family: String = "zimage"

    @Option(name: [.customShort("o"), .long], help: "Output directory for test artifacts")
    var output: String = "./validation_output"

    @Flag(name: [.long], help: "Save reference latents and outputs for future comparison")
    var saveReference: Bool = false

    @Flag(name: [.long], help: "Compare against saved reference outputs")
    var compare: Bool = false

    @Option(name: [.long], help: "Reference directory for comparison")
    var referenceDir: String?

    func run() async throws {
        try MLXBundleSupport.ensureAvailable(quiet: false)

        let outputDir = URL(fileURLWithPath: output)
        try FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)

        print("Image validation")
        print("  family: \(family)")
        print("  suite: \(test.rawValue)")
        print("  output: \(output)")

        let modelURL = try resolveModelRootURL(family: family)

        switch test {
        case .vae:
            try await runVAETests(modelURL: modelURL, outputDir: outputDir)
        case .encoder:
            try await runEncoderTests(modelURL: modelURL, outputDir: outputDir)
        case .transformer:
            try await runTransformerTests(modelURL: modelURL, outputDir: outputDir)
        case .pipeline:
            try await runPipelineTests(modelURL: modelURL, outputDir: outputDir)
        case .all:
            try await runVAETests(modelURL: modelURL, outputDir: outputDir)
            try await runEncoderTests(modelURL: modelURL, outputDir: outputDir)
            try await runTransformerTests(modelURL: modelURL, outputDir: outputDir)
            try await runPipelineTests(modelURL: modelURL, outputDir: outputDir)
        }

        print("\nValidation complete. Artifacts written to \(output).")
    }

    func resolveModelRootURL(family: String) throws -> URL {
        let modelId: ModelResolver.ModelID
        switch family.lowercased() {
        case "zimage":
            modelId = .zetaNano
        case "klein":
            modelId = .kleinMax
        default:
            throw ValidationError("Unknown image family: \(family). Use 'zimage' or 'klein'.")
        }

        let resolved: ModelResolver.Resolution
        do {
            resolved = try ModelResolver().resolve(modelId)
        } catch {
            throw ValidationError("Model \(modelId.rawValue) not found. Pull it with `\(CLICommandDisplay.modelPullCommand(for: modelId.rawValue))` before running validation.")
        }

        let report = MereRunModelValidator.validate(modelRoot: resolved.rootURL, expectedModelID: modelId.rawValue)

        print("  Using model: \(resolved.rootURL.path)")
        print("  Source: \(resolved.source.rawValue)")

        if let manifest = report.manifest {
            let engine = manifest.engine?.rawValue ?? "unknown"
            let variant = manifest.variant?.rawValue ?? "unknown"
            let steps = manifest.defaults?.steps.map(String.init) ?? "?"
            let cfg = manifest.defaults?.cfg.map { String(format: "%.2f", $0) } ?? "?"
            let sigmaShift = manifest.defaults?.sigmaShift.map { String(format: "%.2f", $0) } ?? "?"
            print("  Manifest: engine=\(engine) variant=\(variant) defaults=(steps=\(steps) cfg=\(cfg) sigma_shift=\(sigmaShift))")
        }

        if !report.warnings.isEmpty {
            print("  ⚠ Model warnings:")
            for warning in report.warnings {
                print("    - \(warning)")
            }
        }

        if !report.isValid {
            print("  ✗ Model validation failed:")
            for error in report.errors {
                print("    - \(error)")
            }
            throw ImageValidateRuntimeError(
                "Model directory is incomplete. Repair the manifest or pull the model again with `\(CLICommandDisplay.modelPullCommand(for: modelId.rawValue))`."
            )
        }

        return resolved.rootURL
    }

    func saveImage(_ array: MLXArray, to url: URL) throws {
        var pixels = array
        if pixels.ndim == 4 {
            pixels = pixels.squeezed(axis: 0)
        }
        pixels = pixels.transposed(2, 0, 1)
        MLX.eval(pixels)

        let data = try QwenImageIO.imageData(from: pixels)
        try data.write(to: url)
    }

    func print(_ message: String) {
        CLIStderr.write(message + "\n")
    }
}

struct ImageValidateRuntimeError: LocalizedError, Sendable {
    let message: String

    init(_ message: String) {
        self.message = message
    }

    var errorDescription: String? {
        message
    }
}

extension ImageValidate {
    func resolvedZImageResources(modelURL: URL) throws -> (ZImageTurboResources, MereRunModelManifest?) {
        guard let manifest = try MereRunModelManifest.loadIfPresent(from: modelURL) else {
            return (ZImageTurboResources(rootURL: modelURL), nil)
        }

        let componentResolver = ModelComponentResolver(modelRootURL: modelURL, manifest: manifest)
        let tokenizer = try componentResolver.resolveDirectory(for: .tokenizer, fallbackLocalPath: "tokenizer")
        let textEncoder = try componentResolver.resolveDirectory(for: .textEncoder, fallbackLocalPath: "text_encoder")
        let transformer = try componentResolver.resolveDirectory(for: .transformer, fallbackLocalPath: "transformer")
        let vae = try componentResolver.resolveDirectory(for: .vae, fallbackLocalPath: "vae")
        let scheduler = try? componentResolver.resolveDirectory(for: .scheduler, fallbackLocalPath: "scheduler")

        return (
            ZImageTurboResources(
                modelRootURL: modelURL,
                tokenizerDirURL: tokenizer.directoryURL,
                textEncoderDirURL: textEncoder.directoryURL,
                transformerDirURL: transformer.directoryURL,
                vaeDirURL: vae.directoryURL,
                schedulerDirURL: scheduler?.directoryURL ?? modelURL.appendingPathComponent("scheduler", isDirectory: true)
            ),
            manifest
        )
    }
}
