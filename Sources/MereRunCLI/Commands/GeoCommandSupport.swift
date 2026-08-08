import Foundation

enum GeoCommandSupport {
    static func seconds(since start: ContinuousClock.Instant) -> Double {
        let duration = start.duration(to: .now)
        return Double(duration.components.seconds)
            + Double(duration.components.attoseconds) / 1_000_000_000_000_000_000
    }

    static func jsonString<T: Encodable>(_ value: T) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        return String(decoding: try encoder.encode(value), as: UTF8.self)
    }
}

struct GeoModelPayload: Codable, Equatable {
    let schemaVersion: Int
    let status: String
    let operation: String
    let modelID: String
    let variant: String
    let modelPath: String
    let inputPath: String
    let outputPath: String
    let batchSize: Int
    let device: String
    let modelLoadSeconds: Double?
    let inferenceSeconds: Double?

    init(
        schemaVersion: Int = 1,
        status: String,
        operation: String,
        modelID: String,
        variant: String,
        modelPath: String,
        inputPath: String,
        outputPath: String,
        batchSize: Int,
        device: String,
        modelLoadSeconds: Double?,
        inferenceSeconds: Double?
    ) {
        self.schemaVersion = schemaVersion
        self.status = status
        self.operation = operation
        self.modelID = modelID
        self.variant = variant
        self.modelPath = modelPath
        self.inputPath = inputPath
        self.outputPath = outputPath
        self.batchSize = batchSize
        self.device = device
        self.modelLoadSeconds = modelLoadSeconds
        self.inferenceSeconds = inferenceSeconds
    }

    private enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case status
        case operation
        case modelID = "model_id"
        case variant
        case modelPath = "model_path"
        case inputPath = "input_path"
        case outputPath = "output_path"
        case batchSize = "batch_size"
        case device
        case modelLoadSeconds = "model_load_seconds"
        case inferenceSeconds = "inference_seconds"
    }
}
