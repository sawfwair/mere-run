import Foundation

public enum ACEStepQualityPreset: String, CaseIterable, Codable, Hashable, Sendable {
    case draft
    case song
    case final
    case edit

    public func defaults(
        for variant: ACEStepCheckpointVariant,
        task: ACEStepTask
    ) -> ACEStepQualityDefaults {
        switch self {
        case .draft:
            return ACEStepQualityDefaults(
                inferenceSteps: variant.isTurbo ? 4 : 20,
                shift: variant.defaultShift,
                guidanceScale: variant.defaultGuidanceScale,
                samplerMode: .euler,
                velocityNormThreshold: 0,
                velocityEMAFactor: 0,
                usesLanguageModel: false,
                plansMetadata: false,
                automaticDuration: false,
                fallbackDurationSeconds: 10,
                candidateCount: 1
            )
        case .song:
            return ACEStepQualityDefaults(
                inferenceSteps: variant.defaultInferenceSteps,
                shift: variant.defaultShift,
                guidanceScale: variant.defaultGuidanceScale,
                samplerMode: .euler,
                velocityNormThreshold: 0,
                velocityEMAFactor: 0,
                usesLanguageModel: !task.skipsLanguageModel,
                plansMetadata: !task.skipsLanguageModel,
                automaticDuration: task == .textToMusic || task == .complete,
                fallbackDurationSeconds: 30,
                candidateCount: 2
            )
        case .final:
            return ACEStepQualityDefaults(
                inferenceSteps: variant.defaultInferenceSteps,
                shift: variant.defaultShift,
                guidanceScale: variant.defaultGuidanceScale,
                samplerMode: .heun,
                velocityNormThreshold: 2,
                velocityEMAFactor: 0.1,
                usesLanguageModel: !task.skipsLanguageModel,
                plansMetadata: !task.skipsLanguageModel,
                automaticDuration: task == .textToMusic || task == .complete,
                fallbackDurationSeconds: 60,
                candidateCount: 4
            )
        case .edit:
            return ACEStepQualityDefaults(
                inferenceSteps: variant.defaultInferenceSteps,
                shift: variant.defaultShift,
                guidanceScale: variant.defaultGuidanceScale,
                samplerMode: .heun,
                velocityNormThreshold: 2,
                velocityEMAFactor: 0.1,
                usesLanguageModel: !task.skipsLanguageModel,
                plansMetadata: !task.skipsLanguageModel,
                automaticDuration: false,
                fallbackDurationSeconds: 30,
                candidateCount: 2
            )
        }
    }
}

public struct ACEStepQualityDefaults: Hashable, Sendable {
    public var inferenceSteps: Int
    public var shift: Float
    public var guidanceScale: Float
    public var samplerMode: ACEStepSamplerMode
    public var velocityNormThreshold: Float
    public var velocityEMAFactor: Float
    public var usesLanguageModel: Bool
    public var plansMetadata: Bool
    public var automaticDuration: Bool
    public var fallbackDurationSeconds: Float
    public var candidateCount: Int

    public init(
        inferenceSteps: Int,
        shift: Float,
        guidanceScale: Float,
        samplerMode: ACEStepSamplerMode,
        velocityNormThreshold: Float,
        velocityEMAFactor: Float,
        usesLanguageModel: Bool,
        plansMetadata: Bool,
        automaticDuration: Bool,
        fallbackDurationSeconds: Float,
        candidateCount: Int
    ) {
        self.inferenceSteps = inferenceSteps
        self.shift = shift
        self.guidanceScale = guidanceScale
        self.samplerMode = samplerMode
        self.velocityNormThreshold = velocityNormThreshold
        self.velocityEMAFactor = velocityEMAFactor
        self.usesLanguageModel = usesLanguageModel
        self.plansMetadata = plansMetadata
        self.automaticDuration = automaticDuration
        self.fallbackDurationSeconds = fallbackDurationSeconds
        self.candidateCount = candidateCount
    }
}
