import SwiftUI

/// The one model control: a chip that opens the `model list` rows filtered to a mode's
/// categories, installed first, with "Auto" for the mode's default. The composer's chip strip
/// and the Converse thread header share it, so a model is picked the same way everywhere.
struct StudioModelPicker: View {
    let mode: StudioMode
    @Binding var model: String
    /// Every row of `model list`, installed or not; the menu filters it to the mode.
    let inventory: [StudioModelInventoryRow]
    let readiness: ModelReadinessState
    let onShowModels: () -> Void

    var body: some View {
        Menu {
            Toggle(isOn: Binding(get: { model.isBlank }, set: { _ in model = "" })) {
                Text(defaultModelID.isEmpty ? "Auto" : "Auto · \(Self.displayModelName(defaultModelID))")
            }
            let choices = mode.modelChoices(from: inventory)
            let installed = choices.filter(\.isInstalled)
            let downloadable = choices.filter { !$0.isInstalled }
            if !installed.isEmpty {
                Section("Installed") {
                    ForEach(installed) { row in modelRow(row) }
                }
            }
            if !downloadable.isEmpty {
                Section("Needs download") {
                    ForEach(downloadable) { row in modelRow(row) }
                }
            }
            if choices.isEmpty {
                Text("No \(mode.title.lowercased()) models listed yet")
            }
            Divider()
            Button("Browse Models…", action: onShowModels)
        } label: {
            StudioComposerChipLabel(title: label, leadingSystemImage: statusGlyph)
        }
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
        .fixedSize()
        .help(help)
        .accessibilityLabel("Model")
        .accessibilityValue(accessibilityValue)
    }

    private func modelRow(_ row: StudioModelInventoryRow) -> some View {
        Toggle(isOn: Binding(get: { row.id == model }, set: { _ in model = row.id })) {
            Label(Self.displayModelName(row.id), systemImage: row.isInstalled ? "internaldrive" : "arrow.down.circle")
        }
    }

    /// The mode's template default, shown as "Auto".
    private var defaultModelID: String {
        CommandCatalog.template(id: mode.defaultTemplateID)?.defaultModel ?? ""
    }

    /// The resolved model id for this run: the explicit model, else the mode's default.
    private var resolvedModelID: String {
        Self.resolvedModelID(model, mode: mode)
    }

    private var label: String {
        resolvedModelID.isEmpty ? "Auto" : Self.displayModelName(resolvedModelID)
    }

    /// A glyph before the model name when the model is not ready: missing locally, or unsupported.
    private var statusGlyph: String? {
        switch readiness {
        case .missingModel: return "arrow.down.circle"
        case .unsupported: return "exclamationmark.triangle"
        case .checking, .ready, .unknown: return nil
        }
    }

    private var help: String {
        let identity = resolvedModelID.isEmpty ? "Auto — the mode's default model" : "Model: \(resolvedModelID)"
        switch readiness {
        case .ready, .unknown: return identity
        default: return "\(identity) · \(readiness.message)"
        }
    }

    private var accessibilityValue: String {
        let identity = resolvedModelID.isEmpty ? "Automatic" : resolvedModelID
        return "\(identity), \(readiness.title)"
    }

    /// The model id a draft actually runs with: its explicit model, else the mode's template
    /// default. Empty only when the mode has no default at all.
    static func resolvedModelID(_ model: String, mode: StudioMode) -> String {
        let current = model.trimmingCharacters(in: .whitespacesAndNewlines)
        if !current.isEmpty { return current }
        return CommandCatalog.template(id: mode.defaultTemplateID)?.defaultModel ?? ""
    }

    /// A human-facing label for a model id: drop the modality/category prefix and
    /// title-case the distinctive remainder ("text-agent-deepseek-v4-flash" →
    /// "Deepseek V4 Flash", "text-chat-qwen3.6-4b" → "Qwen3.6 4B"). The exact id stays in
    /// the chip's tooltip, so the friendly name never hides what the CLI actually expects.
    static func displayModelName(_ id: String) -> String {
        let leaf = id.components(separatedBy: "/").last ?? id
        let prefixes = [
            "text-chat-", "text-agent-", "text-embed-", "text-code-",
            "image-", "video-", "music-", "sfx-", "speech-tts-", "speech-asr-", "speech-",
            "embed-", "vision-ground-", "vision-segment-", "vision-chat-", "vision-ocr-", "vision-", "text-"
        ]
        var core = leaf
        for prefix in prefixes where core.hasPrefix(prefix) {
            core = String(core.dropFirst(prefix.count))
            break
        }
        let words = core.split(separator: "-").map { token -> String in
            if isParameterCount(token) { return token.uppercased() }
            guard let first = token.first, first.isLetter else { return String(token) }
            return first.uppercased() + token.dropFirst()
        }
        let label = words.joined(separator: " ")
        return label.isEmpty ? leaf : label
    }

    /// "4b", "27b", "1.5b": a parameter count, which reads as "4B" the way model cards print it.
    private static func isParameterCount(_ token: Substring) -> Bool {
        guard token.count >= 2, token.last == "b" else { return false }
        let digits = token.dropLast()
        return !digits.isEmpty && digits.allSatisfy { $0.isNumber || $0 == "." } && digits.contains { $0.isNumber }
    }
}
