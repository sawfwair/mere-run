import Foundation
import MereRunCore

struct MiniMaxMusic3GenerationRecipe: Codable {
    static let currentSchemaVersion = 1

    var schemaVersion: Int
    var createdAt: Date
    var modelID: String
    var sourceRepository: String
    var sourceRevision: String
    var caption: String
    var lyrics: String
    var durationSeconds: Float
    var generatedFrameCount: Int
    var inferenceSteps: Int
    var seed: UInt64
    var guidanceScale: Float
    var sampleRate: Int
    var export: ACEStepAudioExportOptions
    var outputFilename: String
    var outputSHA256: String

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case createdAt = "created_at"
        case modelID = "model_id"
        case sourceRepository = "source_repository"
        case sourceRevision = "source_revision"
        case caption
        case lyrics
        case durationSeconds = "duration_seconds"
        case generatedFrameCount = "generated_frame_count"
        case inferenceSteps = "inference_steps"
        case seed
        case guidanceScale = "guidance_scale"
        case sampleRate = "sample_rate"
        case export
        case outputFilename = "output_filename"
        case outputSHA256 = "output_sha256"
    }

    func write(to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        try encoder.encode(self).write(to: url, options: .atomic)
    }
}
