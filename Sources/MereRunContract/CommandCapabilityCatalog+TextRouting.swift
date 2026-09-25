import Foundation

extension MereRunCapabilityCatalog {
    enum TextCodeFamily: String, MereRunFamilyID {
        case llamaGGUF = "llama-gguf"
    }

    enum TextEmbedFamily: String, MereRunFamilyID {
        case qwen3Embedding = "qwen3-embedding"
    }

    enum TextAnonymizeFamily: String, MereRunFamilyID {
        case privacyFilter = "privacy-filter"
    }

    enum TextDecideFamily: String, MereRunFamilyID {
        case laya
    }

    static let textCodeRouting = MereRunCapabilityRouting(
        modelFlags: ["--model"],
        defaultModels: [.always("text-code-qwen3")],
        families: [
            .init(
                TextCodeFamily.llamaGGUF,
                title: "llama.cpp GGUF",
                models: ["text-code-qwen3", "text-code-north-mini", "text-agent-ornith-35b", "text-agent-qwen35-9b"]
            )
        ]
    )

    static let textEmbedRouting = MereRunCapabilityRouting(
        modelFlags: ["--model"],
        defaultModels: [.always("text-embed-qwen3-0.6b")],
        families: [.init(TextEmbedFamily.qwen3Embedding, title: "Qwen3 Embedding", models: ["text-embed-qwen3-0.6b"])]
    )

    static let textAnonymizeRouting = MereRunCapabilityRouting(
        modelFlags: ["--model"],
        defaultModels: [.always("text-anonymize-privacy-filter")],
        families: [
            .init(TextAnonymizeFamily.privacyFilter, title: "Privacy Filter", models: ["text-anonymize-privacy-filter"])
        ]
    )

    static let textDecideRouting = MereRunCapabilityRouting(
        modelFlags: ["--model"],
        defaultModels: [.always("text-decide-laya")],
        families: [
            .init(
                TextDecideFamily.laya,
                title: "Laya",
                models: ["text-decide-laya", "text-decide-laya-multilingual", "text-decide-laya-typed-decisions"]
            )
        ]
    )
}
