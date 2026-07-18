import Crypto
import Foundation

enum Gemma4CanonicalChatTemplate {
    enum Variant: String, Sendable {
        case e4b
        case standard

        var resourceName: String {
            switch self {
            case .e4b:
                "chat_template_e4b_2026-07-15"
            case .standard:
                "chat_template_standard_2026-07-15"
            }
        }

        var expectedSHA256: String {
            switch self {
            case .e4b:
                "0a2c8073c878ab1da004bee933a998606537bbb62016310352c7285c3f01c5b5"
            case .standard:
                "ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4"
            }
        }
    }

    struct Selection: Sendable {
        let variant: Variant
        let template: String
    }

    private static let staleTemplateSHA256s = Set([
        "36e3a42e5cf14cd0020e72d92e1fdd9970f59b82170e421f0cbe1bb42bead3f0",
    ])

    static func override(
        for rootURL: URL,
        fileManager: FileManager = .default,
        recognizedStaleTemplateSHA256s: Set<String> = staleTemplateSHA256s
    ) throws -> Selection? {
        let templateURL = rootURL.appending(path: "chat_template.jinja")
        guard fileManager.fileExists(atPath: templateURL.path) else {
            return nil
        }

        let installedTemplate = try Data(contentsOf: templateURL)
        guard recognizedStaleTemplateSHA256s.contains(sha256(installedTemplate)) else {
            return nil
        }

        let configURL = rootURL.appending(path: "config.json")
        let config = try JSONDecoder().decode(
            Profile.self,
            from: Data(contentsOf: configURL)
        )
        guard let variant = variant(for: config.textConfig) else {
            return nil
        }

        let resourceURL = Bundle.module.url(
            forResource: variant.resourceName,
            withExtension: "jinja",
            subdirectory: "Gemma4"
        ) ?? Bundle.module.url(
            forResource: variant.resourceName,
            withExtension: "jinja"
        )
        guard let resourceURL else {
            throw Error.missingBundledTemplate(variant)
        }

        let resourceData = try Data(contentsOf: resourceURL)
        guard sha256(resourceData) == variant.expectedSHA256 else {
            throw Error.invalidBundledTemplate(variant)
        }
        guard let template = String(data: resourceData, encoding: .utf8) else {
            throw Error.invalidBundledTemplate(variant)
        }
        return Selection(variant: variant, template: template)
    }

    private static func variant(for config: Profile.TextConfig) -> Variant? {
        switch (config.hiddenSize, config.numHiddenLayers) {
        case (2_560, 42):
            .e4b
        case (2_816, 30), (3_840, 48), (5_376, 60):
            .standard
        default:
            nil
        }
    }

    private static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private struct Profile: Decodable {
        let textConfig: TextConfig

        private enum CodingKeys: String, CodingKey {
            case textConfig = "text_config"
        }

        struct TextConfig: Decodable {
            let hiddenSize: Int
            let numHiddenLayers: Int

            private enum CodingKeys: String, CodingKey {
                case hiddenSize = "hidden_size"
                case numHiddenLayers = "num_hidden_layers"
            }
        }
    }

    enum Error: LocalizedError {
        case missingBundledTemplate(Variant)
        case invalidBundledTemplate(Variant)

        var errorDescription: String? {
            switch self {
            case .missingBundledTemplate(let variant):
                "Bundled canonical Gemma 4 chat template is missing for \(variant.rawValue)."
            case .invalidBundledTemplate(let variant):
                "Bundled canonical Gemma 4 chat template failed checksum or UTF-8 validation for \(variant.rawValue)."
            }
        }
    }
}
