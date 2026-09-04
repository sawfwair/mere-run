import Foundation

/// How a model id reads to a person. One place, so the composer chip, the Converse thread
/// header, the inspector, the Models page, and the Activity popover all print the same name for
/// the same id. The exact id stays in tooltips and in the Command view, so the friendly name
/// never hides what the CLI actually expects.
package enum StudioModelNaming {
    /// The mode's template default, shown as "Auto".
    package static func defaultModelID(for mode: StudioMode) -> String {
        CommandCatalog.template(id: mode.defaultTemplateID)?.defaultModel ?? ""
    }

    /// The model id a draft actually runs with: its explicit model, else the mode's template
    /// default. Empty only when the mode has no default at all.
    package static func resolvedModelID(for mode: StudioMode, model: String) -> String {
        let current = model.trimmingCharacters(in: .whitespacesAndNewlines)
        return current.isEmpty ? defaultModelID(for: mode) : current
    }

    /// What a picker shows for a draft: the resolved model's name, or "Auto" when the mode has
    /// no default.
    package static func displayLabel(for mode: StudioMode, model: String) -> String {
        let resolved = resolvedModelID(for: mode, model: model)
        return resolved.isEmpty ? "Auto" : displayName(resolved)
    }

    /// A human-facing label for a model id: drop the modality/category prefix and title-case the
    /// distinctive remainder, keeping the casing the model cards print
    /// ("text-agent-deepseek-v4-flash" → "Deepseek V4 Flash", "text-chat-qwen3.6-4b" →
    /// "Qwen3.6 4B", "vision-chat-qwen3.6-vl-4b" → "Qwen3.6-VL 4B").
    package static func displayName(_ id: String) -> String {
        let leaf = id.components(separatedBy: "/").last ?? id
        var core = leaf
        for prefix in categoryPrefixes where core.hasPrefix(prefix) {
            core = String(core.dropFirst(prefix.count))
            break
        }
        var words: [String] = []
        for token in core.split(separator: "-") {
            let word = capitalize(token)
            // A family qualifier hyphenates onto the name it qualifies, the way the model cards
            // print it: "qwen3.6-vl-4b" is "Qwen3.6-VL 4B", not "Qwen3.6 VL 4B".
            if hyphenatedQualifiers.contains(token.lowercased()), let previous = words.popLast() {
                words.append("\(previous)-\(word)")
            } else {
                words.append(word)
            }
        }
        let label = words.joined(separator: " ")
        return label.isEmpty ? leaf : label
    }

    /// The category prefixes `model list` ids carry, longest first so "speech-tts-" wins over
    /// "speech-".
    private static let categoryPrefixes = [
        "text-chat-", "text-agent-", "text-embed-", "text-code-",
        "image-", "video-", "music-", "sfx-", "speech-tts-", "speech-asr-", "speech-",
        "embed-", "vision-ground-", "vision-segment-", "vision-chat-", "vision-ocr-", "vision-", "text-"
    ]

    /// Tokens whose printed casing is not title case: acronyms and product names the model cards
    /// spell a particular way. Everything else title-cases.
    private static let knownCasing: [String: String] = [
        "vl": "VL", "tdt": "TDT", "rt": "RT", "rt2": "RT2", "ltx": "LTX", "ltx2": "LTX-2",
        "mtp": "MTP", "lora": "LoRA", "asr": "ASR", "tts": "TTS", "ocr": "OCR", "vae": "VAE",
        "clip": "CLIP", "t5": "T5", "mmdit": "MMDiT", "dit": "DiT", "sdxl": "SDXL", "sam": "SAM",
        "3d": "3D", "hd": "HD", "sfx": "SFX", "ai": "AI", "gguf": "GGUF", "mlx": "MLX",
        "ace": "ACE", "q4": "Q4", "q8": "Q8", "fp8": "FP8", "bf16": "BF16", "int4": "INT4",
        "int8": "INT8"
    ]

    /// Qualifiers that hyphenate onto the name they qualify, the way the model cards print them:
    /// "qwen3.6-vl-4b" is "Qwen3.6-VL 4B", while "parakeet-tdt" stays "Parakeet TDT".
    private static let hyphenatedQualifiers: Set<String> = ["vl"]

    private static func capitalize(_ token: Substring) -> String {
        if let known = knownCasing[token.lowercased()] { return known }
        if isParameterCount(token) { return token.uppercased() }
        guard let first = token.first, first.isLetter else { return String(token) }
        return first.uppercased() + token.dropFirst()
    }

    /// "4b", "27b", "1.5b", "82m": a parameter count, which reads as "4B" or "82M" the way model
    /// cards print it.
    private static func isParameterCount(_ token: Substring) -> Bool {
        guard token.count >= 2, let last = token.last, last == "b" || last == "m" else { return false }
        let digits = token.dropLast()
        return !digits.isEmpty && digits.allSatisfy { $0.isNumber || $0 == "." } && digits.contains { $0.isNumber }
    }
}

// MARK: - Inventory rows
package struct StudioModelUsageTerms: Equatable {
    package let summary: String
    package let links: [URL]

    package init(summary: String, links: [URL]) {
        self.summary = summary
        self.links = links
    }
}

package struct StudioModelInventoryRow: Identifiable, Equatable {
    package let id: String
    package let category: String
    package let status: String
    package let size: String
    package let usageTerms: StudioModelUsageTerms?
    package let title: String?
    package let summary: String?
    package let estimatedDownloadBytes: Int64?

    package let minimumUnifiedMemoryGB: Int?
    package let recommendedUnifiedMemoryGB: Int?
    package let supported: Bool?
    package let supportReasons: [String]
    package let sourceRepository: String?
    package let publisher: String?
    package let referencedBytes: Int64?
    package let reclaimableBytes: Int64?
    package let sharedBytes: Int64?
    package let externalBytes: Int64?
    /// The model's context window in tokens, when the inventory reports one. Conversation
    /// threads size their transcript budget from it; nil keeps the fixed default budget.
    package let contextWindow: Int?

    package init(
        id: String,
        category: String,
        status: String,
        size: String,
        usageTerms: StudioModelUsageTerms?,
        title: String? = nil,
        summary: String? = nil,
        estimatedDownloadBytes: Int64? = nil,
        minimumUnifiedMemoryGB: Int? = nil,
        recommendedUnifiedMemoryGB: Int? = nil,
        supported: Bool? = nil,
        supportReasons: [String] = [],
        sourceRepository: String? = nil,
        publisher: String? = nil,
        referencedBytes: Int64? = nil,
        reclaimableBytes: Int64? = nil,
        sharedBytes: Int64? = nil,
        externalBytes: Int64? = nil,
        contextWindow: Int? = nil
    ) {
        self.id = id
        self.category = category
        self.status = status
        self.size = size
        self.usageTerms = usageTerms
        self.title = title
        self.summary = summary
        self.estimatedDownloadBytes = estimatedDownloadBytes
        self.minimumUnifiedMemoryGB = minimumUnifiedMemoryGB
        self.recommendedUnifiedMemoryGB = recommendedUnifiedMemoryGB
        self.supported = supported
        self.supportReasons = supportReasons
        self.sourceRepository = sourceRepository
        self.publisher = publisher
        self.referencedBytes = referencedBytes
        self.reclaimableBytes = reclaimableBytes
        self.sharedBytes = sharedBytes
        self.externalBytes = externalBytes
        self.contextWindow = contextWindow
    }

    package var isInstalled: Bool {
        status.lowercased() == "installed"
    }

    package var displayedSize: String {
        guard !isInstalled, let estimatedDownloadBytes else {
            return size
        }
        return ByteCountFormatter.string(fromByteCount: estimatedDownloadBytes, countStyle: .file)
    }
}
