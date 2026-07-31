import Foundation

public struct TextLoRATrainingManifest: Codable, Sendable, Hashable {
    public static let schemaVersion = 1
    public static let gemma4Format = "mererun.gemma4.text-lora"
    public static let lagunaFormat = "mererun.laguna.text-lora"
    public static let inklingFormat = "mererun.inkling.text-lora"
    /// Compatibility alias for the first native text adapter format.
    public static let format = gemma4Format

    public struct Training: Codable, Sendable, Hashable {
        public let trainingSteps: Int
        public let batchSize: Int
        public let learningRate: Float
        public let maxSequenceLength: Int
        public let reasoningEffort: Double?
        public let seed: UInt64
        public let dataset: TextSFTDatasetSummary

        public init(
            trainingSteps: Int,
            batchSize: Int,
            learningRate: Float,
            maxSequenceLength: Int,
            reasoningEffort: Double? = nil,
            seed: UInt64,
            dataset: TextSFTDatasetSummary
        ) {
            self.trainingSteps = trainingSteps
            self.batchSize = batchSize
            self.learningRate = learningRate
            self.maxSequenceLength = maxSequenceLength
            self.reasoningEffort = reasoningEffort
            self.seed = seed
            self.dataset = dataset
        }
    }

    public struct LoRA: Codable, Sendable, Hashable {
        public let rank: Int
        public let alpha: Float
        public let targetModules: [String]

        public init(rank: Int, alpha: Float, targetModules: [String]) {
            self.rank = rank
            self.alpha = alpha
            self.targetModules = targetModules
        }
    }

    public let schemaVersion: Int
    public let createdAt: Date
    public let format: String
    public let baseModel: String
    public let outputFile: String
    public let adapterName: String
    public let training: Training
    public let lora: LoRA
    public let evalPromptCount: Int?
    public let status: String

    public init(
        createdAt: Date = Date(),
        format: String = Self.format,
        baseModel: String,
        outputFile: String,
        adapterName: String,
        training: Training,
        lora: LoRA,
        evalPromptCount: Int?,
        status: String
    ) {
        self.schemaVersion = Self.schemaVersion
        self.createdAt = createdAt
        self.format = format
        self.baseModel = baseModel
        self.outputFile = outputFile
        self.adapterName = adapterName
        self.training = training
        self.lora = lora
        self.evalPromptCount = evalPromptCount
        self.status = status
    }

    public static func url(nextTo outputSafetensorsURL: URL) -> URL {
        outputSafetensorsURL
            .deletingPathExtension()
            .appendingPathExtension("manifest")
            .appendingPathExtension("json")
    }

    public func write(nextTo outputSafetensorsURL: URL) throws {
        let url = Self.url(nextTo: outputSafetensorsURL)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        try encoder.encode(self).write(to: url, options: [.atomic])
    }
}
