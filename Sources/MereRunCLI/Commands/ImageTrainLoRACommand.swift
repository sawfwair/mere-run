import ArgumentParser
import Foundation
import MereRunCore

struct ImageTrainLoRA: AsyncParsableCommand {
    static let defaultManagedModelID: ModelResolver.ModelID = .krea2Raw

    static let configuration = CommandConfiguration(
        commandName: "train-lora",
        abstract: "Train a local image LoRA adapter.",
        discussion: """
        Krea 2 LoRAs are trained on image-krea2-raw and can be used with image-krea2-turbo via image generate --lora.
        Prints the output LoRA path to stdout. Progress and diagnostics are printed to stderr.
        """
    )

    @Option(name: [.customShort("d"), .long], help: "Dataset directory containing image files with matching .txt captions.")
    var data: String?

    @Option(name: [.customShort("o"), .long], help: "Output .safetensors path.")
    var output: String

    @Option(name: [.customShort("m"), .long], help: "Raw/base model path or canonical model id (default: image-krea2-raw).")
    var model: String?

    @Option(name: [.customShort("W"), .long], help: "Training image width in pixels.")
    var width: Int = 1024

    @Option(name: [.customShort("H"), .long], help: "Training image height in pixels.")
    var height: Int = 1024

    @Option(name: [.customLong("training-steps"), .customLong("steps")], help: "Number of optimizer steps.")
    var trainingSteps: Int = 1000

    @Option(name: [.long], help: "Batch size.")
    var batchSize: Int = 1

    @Option(name: [.customLong("learning-rate"), .customLong("lr")], help: "Learning rate.")
    var learningRate: Float = 1e-4

    @Option(name: [.customLong("rank")], help: "LoRA rank.")
    var rank: Int = 16

    @Option(name: [.customLong("alpha")], help: "LoRA alpha. Defaults to rank.")
    var alpha: Float?

    @Option(name: [.customLong("max-text-length")], help: "Maximum prompt token length.")
    var maxTextLength: Int = 512

    @Option(name: [.customLong("scheduler-steps")], help: "Number of FlowMatch training timesteps.")
    var schedulerSteps: Int = 1000

    @Option(name: [.long], help: "Caption dropout probability between 0.0 and 1.0.")
    var captionDropout: Float = 0.05

    @Option(name: [.long], help: "Random seed. Defaults to wall-clock time when omitted or zero.")
    var seed: UInt64 = 0

    @Flag(name: [.customLong("lite")], help: "Train only attention Q/V LoRA layers to reduce memory.")
    var lite: Bool = false

    @Flag(name: [.customLong("exclude-preview-images")], help: "Ignore preview*.png/jpg/webp images in the dataset folder.")
    var excludePreviewImages: Bool = false

    @Option(name: [.customLong("synthetic-samples")], help: "Use synthetic training samples for runtime smoke tests.")
    var syntheticSamples: Int?

    @Flag(name: [.short, .long], help: "Print only the output path.")
    var quiet: Bool = false

    func run() async throws {
        try MLXBundleSupport.ensureAvailable(quiet: quiet)

        guard width > 0, height > 0, width % 16 == 0, height % 16 == 0 else {
            throw ValidationError("--width/--height must be > 0 and divisible by 16")
        }
        guard trainingSteps >= 1 else {
            throw ValidationError("--training-steps must be >= 1")
        }
        guard batchSize >= 1 else {
            throw ValidationError("--batch-size must be >= 1")
        }
        guard schedulerSteps >= 1 else {
            throw ValidationError("--scheduler-steps must be >= 1")
        }
        guard rank >= 1 else {
            throw ValidationError("--rank must be >= 1")
        }
        guard (0.0...1.0).contains(captionDropout) else {
            throw ValidationError("--caption-dropout must be between 0.0 and 1.0")
        }
        if let syntheticSamples, syntheticSamples < 1 {
            throw ValidationError("--synthetic-samples must be >= 1")
        }

        let outputURL = URL(fileURLWithPath: output).standardizedFileURL
        guard outputURL.pathExtension.lowercased() == "safetensors" else {
            throw ValidationError("--output must end in .safetensors")
        }
        try FileManager.default.createDirectory(
            at: outputURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )

        let modelRoot = try resolveModelRoot()
        let examples: [Krea2LoRATrainingExample]
        let datasetRoot: String?
        if syntheticSamples != nil {
            examples = []
            datasetRoot = nil
        } else {
            guard let data else {
                throw ValidationError("--data is required unless --synthetic-samples is set")
            }
            let dataURL = URL(fileURLWithPath: data).standardizedFileURL
            let pairs = try DatasetLoader.loadImageCaptionPairs(
                from: dataURL,
                excludePreviewImages: excludePreviewImages
            )
            examples = pairs.map { pair in
                Krea2LoRATrainingExample(imageURL: pair.imageURL, caption: pair.caption)
            }
            datasetRoot = dataURL.path
        }

        var config = Krea2LoRATrainingConfig()
        config.width = width
        config.height = height
        config.maxTextLength = maxTextLength
        config.schedulerSteps = schedulerSteps
        config.trainingSteps = trainingSteps
        config.batchSize = batchSize
        config.learningRate = learningRate
        config.seed = seed
        config.loraRank = rank
        config.loraAlpha = alpha
        config.captionDropout = captionDropout
        config.loraTargetSuffixes = lite ? Krea2LoRAInjector.liteTargetSuffixes : nil
        config.syntheticSampleCount = syntheticSamples
        config.datasetRoot = datasetRoot

        let progressHandler = quiet ? nil : Self.makeProgressHandler()
        if !quiet {
            CLIStderr.write("[runtime] image training backend: \(NativeMLXRuntime.backendDescription)\n")
        }

        try await Krea2LoRATrainer.train(
            modelPath: modelRoot.path,
            examples: examples,
            outputURL: outputURL,
            config: config,
            progressHandler: progressHandler
        )

        print(outputURL.path)
    }

    private func resolveModelRoot() throws -> URL {
        let resolver = ModelResolver()
        guard let model else {
            do {
                return try resolver.resolve(Self.defaultManagedModelID).rootURL
            } catch {
                let modelID = Self.defaultManagedModelID.rawValue
                throw ValidationError(
                    "Krea 2 Raw model \(modelID) not found. Pull it with `mere.run model pull \(modelID)` or point --model at a local Raw model path."
                )
            }
        }

        let url = URL(fileURLWithPath: model).standardizedFileURL
        if FileManager.default.fileExists(atPath: url.path) {
            return url
        }
        if let id = ModelResolver.ModelID(rawValue: model) {
            do {
                return try resolver.resolve(id).rootURL
            } catch {
                throw ValidationError(
                    "Model \(id.rawValue) not found. Pull it with `mere.run model pull \(id.rawValue)` or point --model at a local path."
                )
            }
        }
        throw ValidationError("Model path not found: \(model). Pass a local model path or a known model id.")
    }

    private static func makeProgressHandler() -> (@Sendable (Krea2LoRATrainingProgress) -> Void) {
        { progress in
            switch progress.stage {
            case .loadingModels:
                CLIStderr.write("Loading Krea 2 Raw models...\n")
            case .encodingDataset(let current, let total):
                CLIStderr.write("\rEncoding dataset (\(current)/\(total))")
                if current >= total {
                    CLIStderr.write("\n")
                }
            case .injectingLoRA(let layerCount):
                CLIStderr.write("Injected LoRA into \(layerCount) Krea 2 layers.\n")
            case .training(let step, let total, let loss):
                if let loss {
                    CLIStderr.write(String(format: "\rTraining (%d/%d) loss %.6f\n", step, total, loss))
                } else {
                    CLIStderr.write("\rTraining (\(step)/\(total))")
                }
            case .saving:
                CLIStderr.write("Saving LoRA artifacts...\n")
            }
        }
    }
}
