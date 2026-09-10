import Foundation

/// Training inputs shared by command and library callers. Viewer and output
/// presentation options belong to the caller.
public struct TextLoRATrainingOptions: Sendable {
    public let data: String
    public let output: String
    public let model: String
    public let modelPath: String?
    public let eval: String?
    public let adapterName: String
    public let trainingSteps: Int
    public let batchSize: Int
    public let learningRate: Float
    public let rank: Int
    public let alpha: Float?
    public let maxSequenceLength: Int
    public let reasoningEffort: Double
    public let seed: UInt64
    public let resumeFrom: String?
    public let resumeStep: Int?
    public let targetModules: String?
    public let dryRun: Bool

    public init(
        data: String,
        output: String,
        model: String = Gemma4Resources.twelveB4BitModelId,
        modelPath: String? = nil,
        eval: String? = nil,
        adapterName: String = "local-assistant",
        trainingSteps: Int = 600,
        batchSize: Int = 1,
        learningRate: Float = 0.0001,
        rank: Int = 16,
        alpha: Float? = nil,
        maxSequenceLength: Int = 4096,
        reasoningEffort: Double = 0.9,
        seed: UInt64 = 42,
        resumeFrom: String? = nil,
        resumeStep: Int? = nil,
        targetModules: String? = nil,
        dryRun: Bool = false
    ) {
        self.data = data
        self.output = output
        self.model = model
        self.modelPath = modelPath
        self.eval = eval
        self.adapterName = adapterName
        self.trainingSteps = trainingSteps
        self.batchSize = batchSize
        self.learningRate = learningRate
        self.rank = rank
        self.alpha = alpha
        self.maxSequenceLength = maxSequenceLength
        self.reasoningEffort = reasoningEffort
        self.seed = seed
        self.resumeFrom = resumeFrom
        self.resumeStep = resumeStep
        self.targetModules = targetModules
        self.dryRun = dryRun
    }

    public func validate() throws {
        guard trainingSteps >= 1 else {
            throw TextLoRATrainingIssue("--training-steps must be >= 1")
        }
        guard batchSize >= 1 else {
            throw TextLoRATrainingIssue("--batch-size must be >= 1")
        }
        guard learningRate > 0 else {
            throw TextLoRATrainingIssue("--learning-rate must be > 0")
        }
        guard rank >= 1 else {
            throw TextLoRATrainingIssue("--rank must be >= 1")
        }
        if let alpha {
            guard alpha > 0 else {
                throw TextLoRATrainingIssue("--alpha must be > 0")
            }
        }
        guard maxSequenceLength >= 128 else {
            throw TextLoRATrainingIssue("--max-sequence-length must be >= 128")
        }
        guard (0...0.99).contains(reasoningEffort) else {
            throw TextLoRATrainingIssue("--reasoning-effort must be between 0 and 0.99")
        }
        guard !resolvedTargetModules().isEmpty else {
            throw TextLoRATrainingIssue("--target-modules must include at least one target suffix")
        }
        if let resumeStep {
            guard resumeFrom != nil else {
                throw TextLoRATrainingIssue("--resume-step requires --resume-from")
            }
            guard resumeStep > 0, resumeStep < trainingSteps else {
                throw TextLoRATrainingIssue(
                    "--resume-step must be greater than zero and below --training-steps"
                )
            }
        }
        if let resumeFrom {
            guard !dryRun else {
                throw TextLoRATrainingIssue("--resume-from cannot be combined with --dry-run")
            }
            let resumeURL = URL(fileURLWithPath: resumeFrom).standardizedFileURL
            guard FileManager.default.fileExists(atPath: resumeURL.path) else {
                throw TextLoRATrainingIssue("--resume-from checkpoint does not exist: \(resumeURL.path)")
            }
            guard resumeURL.pathExtension.lowercased() == "safetensors"
                    || resumeURL.pathExtension.lowercased() == "zip" else {
                throw TextLoRATrainingIssue("--resume-from must be a .safetensors or .zip checkpoint")
            }
            let outputURL = URL(fileURLWithPath: output).standardizedFileURL
            guard resumeURL != outputURL else {
                throw TextLoRATrainingIssue("--resume-from and --output must be different files")
            }
        }
    }

    func makeManifest(
        family: TextLoRATrainingFamily,
        outputURL: URL,
        datasetSummary: TextSFTDatasetSummary,
        evalPromptCount: Int?,
        status: String
    ) -> TextLoRATrainingManifest {
        TextLoRATrainingManifest(
            format: family.manifestFormat,
            baseModel: model,
            outputFile: outputURL.lastPathComponent,
            adapterName: adapterName,
            modality: family == .gemma4VLM ? "image" : nil,
            trainingScope: family == .gemma4VLM ? "language_attention" : nil,
            training: TextLoRATrainingManifest.Training(
                trainingSteps: trainingSteps,
                batchSize: batchSize,
                learningRate: learningRate,
                maxSequenceLength: maxSequenceLength,
                reasoningEffort: family == .inkling ? reasoningEffort : nil,
                seed: seed,
                dataset: datasetSummary
            ),
            lora: TextLoRATrainingManifest.LoRA(
                rank: rank,
                alpha: alpha ?? Float(rank),
                targetModules: resolvedTargetModules()
            ),
            evalPromptCount: evalPromptCount,
            status: status
        )
    }

    public func resolvedTargetModules() -> [String] {
        let value = targetModules ?? Self.defaultTargetModules(for: model).joined(separator: ",")
        return value
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
    }

    public static func defaultTargetModules(for model: String) -> [String] {
        if InklingResources.handles(modelSpec: model) {
            return InklingTextLoRAInjector.defaultTargetSuffixes
        }
        if LFM2Resources.supportsTextLoRATraining(modelSpec: model) {
            return LFM2TextLoRAInjector.defaultTargetSuffixes
        }
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    }

    public func resolvedTrainingFamily() throws -> TextLoRATrainingFamily {
        if Gemma4Resources.handles(modelSpec: model) {
            let family: TextLoRATrainingFamily = Gemma4Resources.supportsVision(modelSpec: model) ? .gemma4VLM : .gemma4
            if family == .gemma4VLM, batchSize != 1 {
                throw TextLoRATrainingIssue("Gemma 4 VLM LoRA training currently requires --batch-size 1")
            }
            return family
        }
        if LagunaResources.managedModelID(for: model) == LagunaResources.xsModelID {
            return .lagunaXS
        }
        if InklingResources.handles(modelSpec: model) {
            return .inkling
        }
        if LFM2Resources.supportsTextLoRATraining(modelSpec: model) {
            return .lfm2A1B
        }
        throw TextLoRATrainingIssue(
            "--model must be a supported Gemma 4 text or vision model, \(LagunaResources.xsModelID), "
                + "\(InklingResources.modelID), or \(LFM2Resources.defaultModelId)."
        )
    }

}

public enum TextLoRATrainingFamily: Sendable, Equatable {
    case gemma4
    case gemma4VLM
    case lagunaXS
    case inkling
    case lfm2A1B

    public var manifestFormat: String {
        switch self {
        case .gemma4:
            TextLoRATrainingManifest.gemma4Format
        case .gemma4VLM:
            TextLoRATrainingManifest.gemma4VLMFormat
        case .lagunaXS:
            TextLoRATrainingManifest.lagunaFormat
        case .inkling:
            TextLoRATrainingManifest.inklingFormat
        case .lfm2A1B:
            TextLoRATrainingManifest.lfm2Format
        }
    }
}

public struct TextLoRATrainingIssue: Error, LocalizedError, Sendable, CustomStringConvertible {
    public let message: String

    public init(_ message: String) { self.message = message }
    public var errorDescription: String? { message }
    public var description: String { message }
}
