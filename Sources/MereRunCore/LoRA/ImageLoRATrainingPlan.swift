import Foundation

/// Prepared metadata, dataset captions and native trainer configuration.
/// Weight loading and optimizer checkpoint compatibility stay in the trainers.
public struct ImageLoRATrainingPlan: Sendable {
    public enum Training: Sendable {
        case krea(examples: [Krea2LoRATrainingExample], configuration: Krea2LoRATrainingConfig)
        case klein(
            examples: [Flux2KleinLoRATrainingExample], configuration: Flux2KleinLoRATrainingConfig,
            resumeFrom: URL?, sample: Sample?
        )
    }

    public struct Sample: Sendable {
        public let modelPath: String
        public let prompt: String
        public let width: Int
        public let height: Int
        public let steps: Int
        public let guidanceScale: Double
        public let loraScale: Double
        public let seed: UInt64
    }

    public let options: ImageLoRATrainingOptions.Resolved
    public let modelRoot: URL
    public let modelManifest: MereRunModelManifest
    public let outputURL: URL
    public let training: Training

    public static func resolve(_ input: ImageLoRATrainingOptions) throws -> Self {
        try Task.checkCancellation()
        let options = try input.resolve()
        try input.validate(options)
        let outputURL = try outputURL(for: input.output)
        let modelRoot = try input.resolveModelRoot(model: options.model)
        let manifest = try MereRunModelManifest.loadRequired(from: modelRoot)
        let training: Training
        switch manifest.family {
        case .krea:
            training = try input.prepareKrea(options: options)
        case .klein:
            training = try input.prepareKlein(options: options)
        default:
            let family = manifest.family?.rawValue ?? "unknown"
            throw ImageLoRATrainingIssue("Unsupported LoRA training model family: \(family). Use a Krea 2 Raw or FLUX.2 Klein base model.")
        }
        return Self(options: options, modelRoot: modelRoot, modelManifest: manifest,
                    outputURL: outputURL, training: training)
    }

    public static func outputURL(for output: String) throws -> URL {
        let url = URL(fileURLWithPath: output).standardizedFileURL
        guard url.pathExtension.lowercased() == "safetensors" else {
            throw ImageLoRATrainingIssue("--output must end in .safetensors")
        }
        return url
    }

    public var isBenchmark: Bool {
        if case .klein(_, let config, _, _) = training { return config.benchmarkSteps != nil }
        return false
    }
}
