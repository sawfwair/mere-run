import Foundation

public struct MiniMaxMusic3GenerationProfile: Codable, Sendable, Equatable {
    public static let currentSchemaVersion = 1

    public let schemaVersion: Int
    public let frameCount: Int
    public let chunkCount: Int
    public let inferenceSteps: Int
    public let totalSeconds: Double
    public let autoregressiveStageSeconds: Double
    public let autoregressiveLoadSeconds: Double
    public let promptPrefillSeconds: Double
    public let semanticSamplingSeconds: Double
    public let residualDepthSeconds: Double
    public let autoregressiveFeedbackSeconds: Double
    public let flowStageSeconds: Double
    public let flowLoadSeconds: Double
    public let conditionEncodingSeconds: Double
    public let flowTransformerSeconds: Double
    public let vocoderStageSeconds: Double
    public let vocoderLoadSeconds: Double
    public let vocoderDecodeSeconds: Double

    init(
        frameCount: Int,
        chunkCount: Int,
        inferenceSteps: Int,
        totalSeconds: Double,
        recorder: MiniMaxMusic3ProfileRecorder
    ) {
        self.schemaVersion = Self.currentSchemaVersion
        self.frameCount = frameCount
        self.chunkCount = chunkCount
        self.inferenceSteps = inferenceSteps
        self.totalSeconds = totalSeconds
        self.autoregressiveStageSeconds = recorder.autoregressiveStageSeconds
        self.autoregressiveLoadSeconds = recorder.autoregressiveLoadSeconds
        self.promptPrefillSeconds = recorder.promptPrefillSeconds
        self.semanticSamplingSeconds = recorder.semanticSamplingSeconds
        self.residualDepthSeconds = recorder.residualDepthSeconds
        self.autoregressiveFeedbackSeconds = recorder.autoregressiveFeedbackSeconds
        self.flowStageSeconds = recorder.flowStageSeconds
        self.flowLoadSeconds = recorder.flowLoadSeconds
        self.conditionEncodingSeconds = recorder.conditionEncodingSeconds
        self.flowTransformerSeconds = recorder.flowTransformerSeconds
        self.vocoderStageSeconds = recorder.vocoderStageSeconds
        self.vocoderLoadSeconds = recorder.vocoderLoadSeconds
        self.vocoderDecodeSeconds = recorder.vocoderDecodeSeconds
    }
}

enum MiniMaxMusic3ProfileStage {
    case autoregressive
    case autoregressiveLoad
    case promptPrefill
    case semanticSampling
    case residualDepth
    case autoregressiveFeedback
    case flow
    case flowLoad
    case conditionEncoding
    case flowTransformer
    case vocoder
    case vocoderLoad
    case vocoderDecode
}

final class MiniMaxMusic3ProfileRecorder {
    private(set) var autoregressiveStageSeconds = 0.0
    private(set) var autoregressiveLoadSeconds = 0.0
    private(set) var promptPrefillSeconds = 0.0
    private(set) var semanticSamplingSeconds = 0.0
    private(set) var residualDepthSeconds = 0.0
    private(set) var autoregressiveFeedbackSeconds = 0.0
    private(set) var flowStageSeconds = 0.0
    private(set) var flowLoadSeconds = 0.0
    private(set) var conditionEncodingSeconds = 0.0
    private(set) var flowTransformerSeconds = 0.0
    private(set) var vocoderStageSeconds = 0.0
    private(set) var vocoderLoadSeconds = 0.0
    private(set) var vocoderDecodeSeconds = 0.0

    func record(_ stage: MiniMaxMusic3ProfileStage, since start: ContinuousClock.Instant) {
        let seconds = Self.seconds(since: start)
        switch stage {
        case .autoregressive:
            autoregressiveStageSeconds += seconds
        case .autoregressiveLoad:
            autoregressiveLoadSeconds += seconds
        case .promptPrefill:
            promptPrefillSeconds += seconds
        case .semanticSampling:
            semanticSamplingSeconds += seconds
        case .residualDepth:
            residualDepthSeconds += seconds
        case .autoregressiveFeedback:
            autoregressiveFeedbackSeconds += seconds
        case .flow:
            flowStageSeconds += seconds
        case .flowLoad:
            flowLoadSeconds += seconds
        case .conditionEncoding:
            conditionEncodingSeconds += seconds
        case .flowTransformer:
            flowTransformerSeconds += seconds
        case .vocoder:
            vocoderStageSeconds += seconds
        case .vocoderLoad:
            vocoderLoadSeconds += seconds
        case .vocoderDecode:
            vocoderDecodeSeconds += seconds
        }
    }

    static func seconds(since start: ContinuousClock.Instant) -> Double {
        let components = start.duration(to: .now).components
        return Double(components.seconds) + Double(components.attoseconds) / 1e18
    }
}
