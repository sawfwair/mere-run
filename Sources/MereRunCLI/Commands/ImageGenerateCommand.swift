import ArgumentParser
import Foundation
import MereRunCore

// MARK: - Image Generate Command

struct ImageGenerate: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "generate",
        abstract: "Generate images with local image models.",
        discussion: """
        Prints the output file path to stdout.
        Progress and diagnostics are printed to stderr.
        """
    )

    @Option(name: [.customShort("p"), .long], help: "Text prompt.")
    var prompt: String

    @Option(name: [.customShort("n"), .customLong("negative-prompt")], help: "Negative prompt (used when --cfg > 1.0).")
    var negativePrompt: String?

    @Option(
        name: [.customLong("cfg"), .customLong("cfg-scale")],
        help: "CFG scale (uses negative prompt when > 1.0; default is model-specific)."
    )
    var cfgScale: Double?

    @Option(name: [.customLong("sigma-shift")], help: "Sigma shift for the FlowMatch schedule (i2L recommends 8).")
    var sigmaShift: Double?

    @Option(name: [.customShort("o"), .long], help: "Output PNG path (default: ./mererun-image-<timestamp>.png).")
    var output: String?

    @Option(name: [.customShort("W"), .long], help: "Output width in pixels.")
    var width: Int = 1024

    @Option(name: [.customShort("H"), .long], help: "Output height in pixels.")
    var height: Int = 1024

    @Option(name: [.customShort("s"), .long], help: "Number of inference steps (default is model-specific).")
    var steps: Int?

    @Option(name: [.long], help: "Random seed (UInt64).")
    var seed: UInt64?

    @Option(name: [.customShort("m"), .long], help: "Model path or canonical model id (default: image-zimage-max).")
    var model: String?

    @Option(name: [.customShort("i"), .long], help: "Input image path (enables image-to-image).")
    var input: String?

    @Option(name: [.customLong("ref-image")], help: "Reference image path for HiDream O1 editing/personalization. Repeat for multiple references.")
    var referenceImages: [String] = []

    @Flag(name: [.customLong("keep-original-aspect")], help: "For one HiDream reference image, preserve the original aspect ratio.")
    var keepOriginalAspect: Bool = false

    @Option(name: [.customLong("strength"), .customLong("str")], help: "Image-to-image strength 0.0–1.0 (default: 0.75).")
    var strength: Double = 0.75

    @Option(name: [.customLong("max-sequence-length")], help: "Max text sequence length.")
    var maxSequenceLength: Int = 512

    @Option(name: [.customShort("l"), .long], help: "LoRA safetensors file path.")
    var lora: String?

    @Option(name: [.long], help: "LoRA scale (default: 1.0).")
    var loraScale: Double = 1.0

    @Flag(name: [.short, .long], help: "Print only the output path.")
    var quiet: Bool = false

    func run() async throws {
        try MLXBundleSupport.ensureAvailable(quiet: quiet)

        if let steps, steps <= 0 {
            throw ValidationError("--steps must be >= 1")
        }
        guard width > 0, height > 0 else {
            throw ValidationError("--width/--height must be > 0")
        }
        guard (0.0...1.0).contains(strength) else {
            throw ValidationError("--strength must be between 0.0 and 1.0")
        }

        let outputURL = CLIOutput.resolveOutputURL(output, defaultPrefix: "mererun-image", defaultExtension: "png")
        try FileManager.default.createDirectory(
            at: outputURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )

        let inputURL: URL?
        if let input {
            let url = URL(fileURLWithPath: input).standardizedFileURL
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw ValidationError("Input image not found: \(url.path)")
            }
            inputURL = url
        } else {
            inputURL = nil
        }

        let referenceImageURLs = try referenceImages.map { path in
            let url = URL(fileURLWithPath: path).standardizedFileURL
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw ValidationError("Reference image not found: \(url.path)")
            }
            return url
        }

        let resolvedModel: String?
        let resolver = ModelResolver()

        if let model {
            let url = URL(fileURLWithPath: model).standardizedFileURL
            if FileManager.default.fileExists(atPath: url.path) {
                resolvedModel = url.path
            } else if let id = ModelResolver.ModelID(rawValue: model) {
                do {
                    resolvedModel = try resolver.resolve(id).rootURL.path
                } catch {
                    throw ValidationError("Model \(id.rawValue) not found. Pull it with `mere.run model pull \(id.rawValue)` or point --model at a local path.")
                }
            } else {
                throw ValidationError(
                    "Model path not found: \(model). Pass a local model path or a known model id."
                )
            }
        } else {
            do {
                resolvedModel = try resolver.resolve(.zetaMax).rootURL.path
            } catch {
                throw ValidationError("Image model image-zimage-max not found. Pull it with `mere.run model pull image-zimage-max` or point --model at a local path.")
            }
        }

        let loraConfig: LoRA?
        if let loraPath = lora {
            let loraURL = URL(fileURLWithPath: loraPath).standardizedFileURL
            guard FileManager.default.fileExists(atPath: loraURL.path) else {
                throw ValidationError("LoRA file not found: \(loraPath)")
            }
            loraConfig = .local(path: loraURL.path, scale: loraScale)
        } else {
            loraConfig = nil
        }

        let manifest = try MereRunModelManifest.loadRequired(from: URL(fileURLWithPath: resolvedModel!))
        let effectiveSteps = steps
            ?? (manifest.family == .hidream ? (manifest.defaults?.steps ?? 4) : 4)
        let effectiveCFG = cfgScale
            ?? (manifest.family == .hidream ? (manifest.defaults?.cfg ?? 1.0) : 1.0)

        let request = GenerationRequest(
            prompt: prompt,
            negativePrompt: negativePrompt,
            referenceImages: referenceImageURLs,
            width: width,
            height: height,
            steps: effectiveSteps,
            guidanceScale: effectiveCFG,
            seed: seed,
            outputURL: outputURL,
            model: resolvedModel,
            maxSequenceLength: maxSequenceLength,
            lora: loraConfig,
            enhancePrompt: false,
            inputImage: inputURL,
            strength: strength,
            keepOriginalAspect: keepOriginalAspect,
            useBetaSigmas: false,
            sigmaShift: sigmaShift.map { Float($0) }
        )

        let progressHandler: (@Sendable (GenerationProgress) -> Void)? = quiet ? nil : CLIGenerationProgressPrinter.makeProgressHandler()

        let result: GenerationResult
        switch manifest.family {
        case .klein:
            let generator = Flux2KleinGenerator()
            result = try await generator.generate(request, progressHandler: progressHandler)
        case .zimage:
            let generator = ZImageTurboGenerator()
            result = try await generator.generate(request, progressHandler: progressHandler)
        case .hidream:
            let generator = HiDreamO1Generator()
            result = try await generator.generate(request, progressHandler: progressHandler)
        case .gemma, .qwen, .sam, .falcon, .tts, .asr, .embed, .code, .ocr, .music, .video, .psi, .privacy, nil:
            throw ValidationError("Unsupported image model family for `mere.run image generate`: \(manifest.id)")
        }

        // stdout: machine-readable path (easy for scripts)
        print(result.outputURL.path)
    }
}
