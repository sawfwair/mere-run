import Foundation
import MereRunCore

/// Adds Messages discovery fields without changing the OpenAI discovery shape.
struct MessagesModelDiscovery: Encodable {
    let model: OpenAIModel
    let local_capabilities: MessagesCapabilities?
    var id: String { model.id }

    init(_ model: OpenAIModel, capabilities: MessagesCapabilities? = nil) {
        self.model = model
        if let capabilities {
            local_capabilities = capabilities
        } else if let profile = ManagedModelCatalog.apiProfile(for: model.id), let engine = profile.servingEngine {
            local_capabilities = .adapter(engine, profile: profile)
        } else { local_capabilities = nil }
    }
    private enum Key: String, CodingKey {
        case type, display_name, created_at, max_input_tokens, max_tokens, local_capabilities
    }
    func encode(to encoder: Encoder) throws {
        try model.encode(to: encoder)
        var c = encoder.container(keyedBy: Key.self)
        try c.encode("model", forKey: .type)
        // This logical model ID can resolve to BF16 or a local conversion.
        // The catalog's default artifact name must not mislabel those weights.
        let displayName = model.id == MuseGlimmerResources.modelId ? "Muse Glimmer 30B" : (model.name ?? model.id)
        try c.encode(displayName, forKey: .display_name)
        try c.encode(ISO8601DateFormatter().string(from: Date(timeIntervalSince1970: Double(model.created))), forKey: .created_at)
        try c.encodeIfPresent(model.limit?.context, forKey: .max_input_tokens)
        try c.encodeIfPresent(model.limit?.output, forKey: .max_tokens)
        try c.encodeIfPresent(local_capabilities, forKey: .local_capabilities)
    }
}

struct MessagesModelPage: Encodable {
    let object = "list"
    let data: [MessagesModelDiscovery]
    let has_more: Bool
    let first_id: String?
    let last_id: String?

    static func page(_ models: [OpenAIModel], limit: Int?, after: String?, before: String?,
                     capabilities: [String: MessagesCapabilities] = [:]) throws -> Self {
        guard after == nil || before == nil else { throw MessagesFailure.invalid("pagination", "use after_id or before_id, not both") }
        let limit = limit ?? 1000
        guard (1...1000).contains(limit) else { throw MessagesFailure.invalid("limit", "must be between 1 and 1000") }
        var models = models.sorted { $0.id < $1.id }
        if let cursor = after ?? before {
            guard let index = models.firstIndex(where: { $0.id == cursor }) else { throw MessagesFailure.invalid("pagination", "unknown cursor") }
            models = after != nil ? Array(models.dropFirst(index + 1)) : Array(models.prefix(index))
        }
        let more = models.count > limit
        let selected = (before == nil ? Array(models.prefix(limit)) : Array(models.suffix(limit))).map {
            MessagesModelDiscovery($0, capabilities: capabilities[$0.id])
        }
        return Self(data: selected, has_more: more, first_id: selected.first?.id, last_id: selected.last?.id)
    }
}
