import Foundation

/// Backend options shared by CLI and resident requests. Transport defaults are explicit at the adapter.
public struct ACEStepGenerationOptions: Sendable {
    public var prompt: String
    public var lyrics: String? = nil
    public var instrumental: Bool? = nil
    public var instruction: String? = nil
    public var durationSeconds: Float? = nil
    public var quality: ACEStepQualityPreset? = nil
    public var task: ACEStepTask? = nil
    public var seed: UInt64? = nil
    public var retakeSeed: UInt64? = nil
    public var retakeVariance: Float? = nil
    public var candidates: Int? = nil
    public var steps: Int? = nil
    public var shift: Float? = nil
    public var inferMethod: ACEStepInferenceMethod? = nil
    public var guidanceScale: Float? = nil
    public var guidanceMode: ACEStepGuidanceMode? = nil
    public var cfgIntervalStart: Float? = nil
    public var cfgIntervalEnd: Float? = nil
    public var velocityNormThreshold: Float? = nil
    public var velocityEMAFactor: Float? = nil
    public var sampler: ACEStepSamplerMode? = nil
    public var useLanguageModel: Bool? = nil
    public var lmTopK: Int? = nil
    public var lmTopP: Float? = nil
    public var lmTemperature: Float? = nil
    public var lmRepetitionPenalty: Float? = nil
    public var lmCFGScale: Float? = nil
    public var lmNegativePrompt: String? = nil
    public var bpm: Int? = nil
    public var keyscale: String? = nil
    public var metadataLanguage: String? = nil
    public var timeSignature: String? = nil
    public var vocalLanguage: String? = nil
    public var sourceAudioPath: String? = nil
    public var referenceAudioPaths: [String]? = nil
    public var audioCoverStrength: Float? = nil
    public var coverNoiseStrength: Float? = nil
    public var sourceCaption: String? = nil
    public var sourceLyrics: String? = nil
    public var flowEditNMin: Float? = nil
    public var flowEditNMax: Float? = nil
    public var flowEditNAverage: Int? = nil
    public var trackName: String? = nil
    public var completeTrackClasses: [String]? = nil
    public var repaintStartSeconds: Float? = nil
    public var repaintEndSeconds: Float? = nil
    public var chunkMaskMode: ACEStepChunkMaskMode? = nil
    public var repaintMode: ACEStepRepaintMode? = nil
    public var repaintStrength: Float? = nil
    public var useTiledVAEDecode: Bool? = nil
    public var vaeChunkSize: Int? = nil
    public var vaeOverlap: Int? = nil
    public var rewriteCaption = true
    public var planningSeed: UInt64?
    public var analyzeSourceAudio = false
    public var roundMetadataDuration = true

    public init(prompt: String) { self.prompt = prompt }
}
