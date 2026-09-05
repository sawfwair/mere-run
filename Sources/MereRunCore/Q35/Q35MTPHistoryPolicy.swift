import Foundation

enum Q35MTPHistoryMode: Equatable {
    case none
    case retained
    case streaming
}

/// Uses the same model admission for prefill preparation and token decoding.
extension Q35Generator {
    static func shouldSpeculate(
        modelId: String,
        usesMoE: Bool,
        promptTokenCount: Int,
        maxContextTokens: Int?,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        shouldSpeculate(
            promptTokenCount: promptTokenCount,
            maxContextTokens: maxContextTokens,
            defaultMinimumPromptTokens: defaultMTPMinimumPromptTokens(modelId: modelId, usesMoE: usesMoE),
            enabledByDefault: usesMoE || modelId == Q35Resources.q38TwentySevenB4BitModelId,
            environment: environment
        )
    }

    static func mtpHistoryMode(
        modelId: String,
        isQwen4Exp: Bool,
        usesMoE: Bool,
        speculationEligible: Bool,
        greedy: Bool,
        promptTokenCount: Int,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Q35MTPHistoryMode {
        guard speculationEligible, greedy else { return .none }
        if isQwen4Exp { return .streaming }
        let stream = environment["MERERUN_Q35_MTP_STREAM_HISTORY"]?.lowercased()
        if stream == "none" { return .none }
        let streamEnabled = stream != "0" && stream != "false" && stream != "off"
        let supportsStreaming = Q35Resources.isOrnith35BModelId(modelId)
            || modelId == Q35Resources.q38TwentySevenBModelId
            || modelId == Q35Resources.q38TwentySevenB4BitModelId
        if streamEnabled, supportsStreaming { return .streaming }
        return !usesMoE && promptTokenCount <= 4_096 ? .retained : .none
    }
}
