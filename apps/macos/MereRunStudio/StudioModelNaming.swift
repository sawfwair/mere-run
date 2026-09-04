import Foundation

/// How a model id reads to a person. One place, so the composer chip, the Converse thread
/// header, the inspector, the Models page, and the Activity popover all print the same name for
/// the same id. The exact id stays in tooltips and in the Command view, so the friendly name
/// never hides what the CLI actually expects.
enum StudioModelNaming {
    /// The mode's template default, shown as "Auto".
    static func defaultModelID(for mode: StudioMode) -> String {
        CommandCatalog.template(id: mode.defaultTemplateID)?.defaultModel ?? ""
    }

    /// The model id a draft actually runs with: its explicit model, else the mode's template
    /// default. Empty only when the mode has no default at all.
    static func resolvedModelID(for mode: StudioMode, model: String) -> String {
        let current = model.trimmingCharacters(in: .whitespacesAndNewlines)
        return current.isEmpty ? defaultModelID(for: mode) : current
    }

    /// What a picker shows for a draft: the resolved model's name, or "Auto" when the mode has
    /// no default.
    static func displayLabel(for mode: StudioMode, model: String) -> String {
        let resolved = resolvedModelID(for: mode, model: model)
        return resolved.isEmpty ? "Auto" : displayName(resolved)
    }

    /// A human-facing label for a model id: drop the modality/category prefix and title-case the
    /// distinctive remainder ("text-agent-deepseek-v4-flash" → "Deepseek V4 Flash",
    /// "text-chat-qwen3.6-4b" → "Qwen3.6 4B").
    static func displayName(_ id: String) -> String {
        let leaf = id.components(separatedBy: "/").last ?? id
        var core = leaf
        for prefix in categoryPrefixes where core.hasPrefix(prefix) {
            core = String(core.dropFirst(prefix.count))
            break
        }
        let label = core.split(separator: "-").map(capitalize).joined(separator: " ")
        return label.isEmpty ? leaf : label
    }

    /// The category prefixes `model list` ids carry, longest first so "speech-tts-" wins over
    /// "speech-".
    private static let categoryPrefixes = [
        "text-chat-", "text-agent-", "text-embed-", "text-code-",
        "image-", "video-", "music-", "sfx-", "speech-tts-", "speech-asr-", "speech-",
        "embed-", "vision-ground-", "vision-segment-", "vision-chat-", "vision-ocr-", "vision-", "text-"
    ]

    private static func capitalize(_ token: Substring) -> String {
        if isParameterCount(token) { return token.uppercased() }
        guard let first = token.first, first.isLetter else { return String(token) }
        return first.uppercased() + token.dropFirst()
    }

    /// "4b", "27b", "1.5b": a parameter count, which reads as "4B" the way model cards print it.
    private static func isParameterCount(_ token: Substring) -> Bool {
        guard token.count >= 2, token.last == "b" else { return false }
        let digits = token.dropLast()
        return !digits.isEmpty && digits.allSatisfy { $0.isNumber || $0 == "." } && digits.contains { $0.isNumber }
    }
}
