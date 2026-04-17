import Foundation

public enum ASRBackend: String, Sendable, Hashable, Codable, CaseIterable {
    case auto
    case parakeet
    case qwen
}

public enum ASRResolvedBackend: String, Sendable, Hashable, Codable {
    case parakeet
    case qwen
}

public struct ASRBackendAvailability: Sendable, Hashable {
    public let parakeetAvailable: Bool
    public let qwenAvailable: Bool

    public init(parakeetAvailable: Bool, qwenAvailable: Bool) {
        self.parakeetAvailable = parakeetAvailable
        self.qwenAvailable = qwenAvailable
    }

    public var hasAnyBackend: Bool {
        parakeetAvailable || qwenAvailable
    }
}

public struct ASRBackendDecision: Sendable, Hashable {
    public let backend: ASRResolvedBackend
    public let reason: String
    public let normalizedLanguageHint: String?

    public init(
        backend: ASRResolvedBackend,
        reason: String,
        normalizedLanguageHint: String?
    ) {
        self.backend = backend
        self.reason = reason
        self.normalizedLanguageHint = normalizedLanguageHint
    }
}

public enum ASRBackendRouting {
    public static func select(
        task: ASRTask,
        languageHint: String?,
        preferredBackend: ASRBackend,
        availableBackends: ASRBackendAvailability,
        parakeetSupportedLanguageCodes: Set<String>? = nil
    ) -> ASRBackendDecision {
        let normalizedHint = normalizeLanguageHint(languageHint)

        if task == .translate {
            return pick(
                preferred: .qwen,
                fallback: nil,
                availableBackends: availableBackends,
                reason: "translate_requires_qwen",
                normalizedLanguageHint: normalizedHint
            )
        }

        if let raw = languageHint?.trimmingCharacters(in: .whitespacesAndNewlines), !raw.isEmpty {
            guard let normalizedHint else {
                return pick(
                    preferred: .qwen,
                    fallback: nil,
                    availableBackends: availableBackends,
                    reason: "unknown_explicit_language_hint",
                    normalizedLanguageHint: nil
                )
            }

            let parakeetSupports = isParakeetLanguageSupported(
                normalizedHint,
                supportedCodes: parakeetSupportedLanguageCodes
            )
            if !parakeetSupports {
                return pick(
                    preferred: .qwen,
                    fallback: nil,
                    availableBackends: availableBackends,
                    reason: "unsupported_language_for_parakeet",
                    normalizedLanguageHint: normalizedHint
                )
            }
        }

        switch preferredBackend {
        case .qwen:
            return pick(
                preferred: .qwen,
                fallback: .parakeet,
                availableBackends: availableBackends,
                reason: "preferred_qwen",
                normalizedLanguageHint: normalizedHint
            )
        case .parakeet:
            return pick(
                preferred: .parakeet,
                fallback: .qwen,
                availableBackends: availableBackends,
                reason: "preferred_parakeet",
                normalizedLanguageHint: normalizedHint
            )
        case .auto:
            return pick(
                preferred: .parakeet,
                fallback: .qwen,
                availableBackends: availableBackends,
                reason: "auto_prefers_parakeet_for_transcription",
                normalizedLanguageHint: normalizedHint
            )
        }
    }

    public static func normalizeLanguageHint(_ languageHint: String?) -> String? {
        guard let languageHint else { return nil }
        var value = languageHint
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()

        guard !value.isEmpty else { return nil }

        if value == "auto" {
            return nil
        }

        if value.hasPrefix("<|") && value.hasSuffix("|>") {
            value.removeFirst(2)
            value.removeLast(2)
        }

        if let mapped = languageNameToCode[value] {
            return mapped
        }

        if value.contains("-") || value.contains("_") {
            let components = value
                .replacingOccurrences(of: "_", with: "-")
                .split(separator: "-")
            if let head = components.first, head.count >= 2, head.count <= 3 {
                return String(head)
            }
        }

        if value.count >= 2, value.count <= 3,
           value.unicodeScalars.allSatisfy({ $0.properties.isAlphabetic }) {
            return value
        }

        return nil
    }

    public static func isParakeetLanguageSupported(
        _ normalizedCode: String,
        supportedCodes: Set<String>? = nil
    ) -> Bool {
        let normalized = normalizedCode.lowercased()
        if let supportedCodes {
            return supportedCodes.contains(normalized)
        }
        return defaultParakeetLanguageCodes.contains(normalized)
    }

    private static func pick(
        preferred: ASRResolvedBackend,
        fallback: ASRResolvedBackend?,
        availableBackends: ASRBackendAvailability,
        reason: String,
        normalizedLanguageHint: String?
    ) -> ASRBackendDecision {
        if isAvailable(preferred, in: availableBackends) {
            return ASRBackendDecision(
                backend: preferred,
                reason: reason,
                normalizedLanguageHint: normalizedLanguageHint
            )
        }

        if let fallback, isAvailable(fallback, in: availableBackends) {
            return ASRBackendDecision(
                backend: fallback,
                reason: "\(reason)_fallback_to_\(fallback.rawValue)",
                normalizedLanguageHint: normalizedLanguageHint
            )
        }

        return ASRBackendDecision(
            backend: preferred,
            reason: "\(reason)_preferred_unavailable",
            normalizedLanguageHint: normalizedLanguageHint
        )
    }

    private static func isAvailable(
        _ backend: ASRResolvedBackend,
        in availability: ASRBackendAvailability
    ) -> Bool {
        switch backend {
        case .parakeet: return availability.parakeetAvailable
        case .qwen: return availability.qwenAvailable
        }
    }

    private static let languageNameToCode: [String: String] = [
        "afrikaans": "af",
        "arabic": "ar",
        "armenian": "hy",
        "azerbaijani": "az",
        "basque": "eu",
        "belarusian": "be",
        "bengali": "bn",
        "bosnian": "bs",
        "bulgarian": "bg",
        "burmese": "my",
        "catalan": "ca",
        "chinese": "zh",
        "mandarin": "zh",
        "cantonese": "zh",
        "croatian": "hr",
        "czech": "cs",
        "danish": "da",
        "dutch": "nl",
        "english": "en",
        "estonian": "et",
        "finnish": "fi",
        "french": "fr",
        "galician": "gl",
        "german": "de",
        "greek": "el",
        "gujarati": "gu",
        "haitian": "ht",
        "hausa": "ha",
        "hebrew": "he",
        "hindi": "hi",
        "hungarian": "hu",
        "icelandic": "is",
        "indonesian": "id",
        "irish": "ga",
        "italian": "it",
        "japanese": "ja",
        "javanese": "jv",
        "kannada": "kn",
        "kazakh": "kk",
        "khmer": "km",
        "korean": "ko",
        "lao": "lo",
        "latvian": "lv",
        "lithuanian": "lt",
        "macedonian": "mk",
        "malay": "ms",
        "malayalam": "ml",
        "marathi": "mr",
        "mongolian": "mn",
        "nepali": "ne",
        "norwegian": "no",
        "persian": "fa",
        "polish": "pl",
        "portuguese": "pt",
        "punjabi": "pa",
        "romanian": "ro",
        "russian": "ru",
        "serbian": "sr",
        "slovak": "sk",
        "slovenian": "sl",
        "somali": "so",
        "spanish": "es",
        "swahili": "sw",
        "swedish": "sv",
        "tagalog": "tl",
        "tamil": "ta",
        "telugu": "te",
        "thai": "th",
        "turkish": "tr",
        "ukrainian": "uk",
        "urdu": "ur",
        "uzbek": "uz",
        "vietnamese": "vi",
        "welsh": "cy",
        "yiddish": "yi",
        "yoruba": "yo",
        "zulu": "zu",
    ]

    private static let defaultParakeetLanguageCodes: Set<String> = [
        "aa", "ab", "af", "ak", "am", "ar", "as", "av", "ay", "az", "ba", "be", "bg", "bi", "bm", "bn", "bo", "br", "bs", "ca", "ce", "ch", "co", "cr", "cs", "cu", "cv", "cy", "da", "de", "dv", "dz", "ee", "el", "en", "eo", "es", "et", "eu", "fa", "ff", "fi", "fj", "fo", "fr", "fy", "ga", "gd", "gl", "gn", "gu", "gv", "ha", "he", "hi", "ho", "hr", "ht", "hu", "hy", "hz", "ia", "id", "ie", "ig", "ii", "ik", "io", "is", "it", "iu", "ja", "jv", "ka", "kg", "ki", "kj", "kk", "kl", "km", "kn", "ko", "kr", "ks", "ku", "kv", "kw", "ky", "la", "lb", "lg", "li", "ln", "lo", "lt", "lu", "lv", "mg", "mh", "mi", "mk", "ml", "mn", "mr", "ms", "mt", "my", "na", "nb", "nd", "ne", "ng", "nl", "nn", "no", "nr", "nv", "ny", "oc", "oj", "om", "or", "os", "pa", "pi", "pl", "ps", "pt", "qu", "rm", "rn", "ro", "ru", "rw", "sa", "sc", "sd", "se", "sg", "si", "sk", "sl", "sm", "sn", "so", "sq", "sr", "ss", "st", "su", "sv", "sw", "ta", "te", "tg", "th", "ti", "tk", "tl", "tn", "to", "tr", "ts", "tt", "tw", "ty", "ug", "uk", "ur", "uz", "ve", "vi", "vo", "wa", "wo", "xh", "yi", "yo", "za", "zh", "zu",
    ]
}
