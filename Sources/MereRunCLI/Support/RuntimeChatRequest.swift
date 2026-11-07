import Foundation
import MereRunCore

struct RuntimeChatConfiguration: Sendable {
    let modelID: String
    let engine: RuntimeServingEngine
    let settings: RuntimeModelSettings
    let profile: ManagedModelAPIProfile
    let contextSize: Int
    let fallbackLoraPath: String?
}

struct RuntimePreparedChat: Sendable {
    var request: ChatRequest
    let includeUsage: Bool
    let scope: APIModelScope
}

/// A protocol-neutral preparation operation. HTTP adapters capture their typed
/// wire request; the serving service only sees model identity and runtime data.
struct RuntimeChatRequest: Sendable {
    let model: String
    let streaming: Bool
    let prepare: @Sendable (RuntimeChatConfiguration) throws -> RuntimePreparedChat

    static func openAI(_ wire: OpenAIChatRequest) -> Self {
        Self(model: wire.model, streaming: wire.stream == true) { configuration in
            var effective = wire
            let settings = configuration.settings
            if effective.max_tokens == nil, effective.max_completion_tokens == nil { effective.max_tokens = settings.maxTokens }
            if effective.temperature == nil { effective.temperature = settings.temperature }
            if effective.top_p == nil { effective.top_p = settings.topP }
            if effective.min_p == nil { effective.min_p = settings.minP }
            let capabilities = APIEngineCapabilities.catalog(configuration.profile)
            let request = try APIServerContract.chatRequest(
                from: effective, fallbackLoraPath: configuration.fallbackLoraPath,
                contextSize: configuration.contextSize, capabilities: capabilities,
                servedModelID: configuration.modelID, apiProfile: configuration.profile
            )
            return RuntimePreparedChat(
                request: request,
                includeUsage: try APIServerContract.includeUsageInStreaming(effective, capabilities: capabilities),
                scope: try APIModelScope.chat(wire, resolved: request, modelID: configuration.modelID)
            )
        }
    }
}
