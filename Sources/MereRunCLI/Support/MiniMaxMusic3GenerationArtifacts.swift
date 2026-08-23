import Foundation
import MereRunCore

struct MiniMaxMusic3GenerationRecipe: Codable {
    static let currentSchemaVersion = 7

    var schemaVersion: Int
    var createdAt: Date
    var modelID: String
    var sourceRepository: String
    var sourceRevision: String
    var inputBrief: String
    var caption: String
    var lyrics: String
    var composition: MiniMaxMusic3CompositionReceipt?
    var lyricPreflightPolicy: MiniMaxMusic3LyricPreflightPolicy
    var lyricPreflight: MiniMaxMusic3LyricPreflightReport?
    var durationSeconds: Float
    var requestedMinimumFrames: Int?
    var requestedMaximumFrames: Int?
    var generatedFrameCount: Int
    var samplingTier: MiniMaxMusic3SamplingTier?
    var inferenceSteps: Int
    var seed: UInt64
    var guidanceScale: Float
    var nativeSampleRate: Int
    var outputSampleRate: Int
    var loadingStrategy: MiniMaxMusic3LoadingStrategy
    var performanceMode: MiniMaxMusic3PerformanceMode
    var languageModelPrecision: MiniMaxMusic3WeightPrecision
    var depthDecoderPrecision: MiniMaxMusic3WeightPrecision
    var flowStrategy: MiniMaxMusic3FlowStrategy
    var flowSolver: MiniMaxMusic3FlowSolver
    var autoregressiveGuidanceFrames: Int?
    var flowGuidanceEnd: Float
    var seedStrategy: MiniMaxMusic3SeedStrategy
    var audioHealth: MiniMaxMusic3AudioHealthReport
    var export: ACEStepAudioExportOptions
    var outputFilename: String
    var outputSHA256: String

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case createdAt = "created_at"
        case modelID = "model_id"
        case sourceRepository = "source_repository"
        case sourceRevision = "source_revision"
        case inputBrief = "input_brief"
        case caption
        case lyrics
        case composition
        case lyricPreflightPolicy = "lyric_preflight_policy"
        case lyricPreflight = "lyric_preflight"
        case durationSeconds = "duration_seconds"
        case requestedMinimumFrames = "requested_minimum_frames"
        case requestedMaximumFrames = "requested_maximum_frames"
        case generatedFrameCount = "generated_frame_count"
        case samplingTier = "sampling_tier"
        case inferenceSteps = "inference_steps"
        case seed
        case guidanceScale = "guidance_scale"
        case nativeSampleRate = "native_sample_rate"
        case outputSampleRate = "output_sample_rate"
        case loadingStrategy = "loading_strategy"
        case performanceMode = "performance_mode"
        case languageModelPrecision = "language_model_precision"
        case depthDecoderPrecision = "depth_decoder_precision"
        case flowStrategy = "flow_strategy"
        case flowSolver = "flow_solver"
        case autoregressiveGuidanceFrames = "autoregressive_guidance_frames"
        case flowGuidanceEnd = "flow_guidance_end"
        case seedStrategy = "seed_strategy"
        case audioHealth = "audio_health"
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
