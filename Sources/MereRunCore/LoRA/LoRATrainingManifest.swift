import Foundation

public struct LoRATrainingManifest: Codable, Sendable, Hashable {
    public static let schemaVersion: Int = 1

    public struct Training: Codable, Sendable, Hashable {
        public let width: Int
        public let height: Int
        public let trainingSteps: Int
        public let batchSize: Int
        public let learningRate: Float
        public let seed: UInt64
        public let datasetCount: Int
        public let checkpointInterval: Int?
        public let sampleInterval: Int?
        public let samplePrompt: String?
        public let emaDecay: Float
    }

    public struct LoRA: Codable, Sendable, Hashable {
        public let rank: Int
        public let alpha: Float
        public let saveDType: String
        public let includesOptimizerState: Bool
    }

    public let schemaVersion: Int
    public let createdAt: Date

    /// Mirrors the safetensors metadata `format` key (e.g. `mererun.flux2.lora`).
    public let format: String

    /// Mirrors the safetensors metadata `base_model` key (e.g. `flux2-klein`, `z-image-turbo`).
    public let baseModel: String

    /// Filename of the primary LoRA safetensors output.
    public let outputFile: String

    /// Optional filename of the EMA LoRA safetensors output.
    public let emaOutputFile: String?

    public let training: Training
    public let lora: LoRA

    /// Engine-specific metadata as a flat string map.
    public let extras: [String: String]?

    public init(
        createdAt: Date = Date(),
        format: String,
        baseModel: String,
        outputFile: String,
        emaOutputFile: String?,
        training: Training,
        lora: LoRA,
        extras: [String: String]? = nil
    ) {
        self.schemaVersion = Self.schemaVersion
        self.createdAt = createdAt
        self.format = format
        self.baseModel = baseModel
        self.outputFile = outputFile
        self.emaOutputFile = emaOutputFile
        self.training = training
        self.lora = lora
        self.extras = extras
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
        let data = try encoder.encode(self)
        try data.write(to: url, options: [.atomic])
    }
}
