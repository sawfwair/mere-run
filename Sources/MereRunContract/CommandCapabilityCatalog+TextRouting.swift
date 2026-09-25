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

    /// The code paths `text chat` runs. Where a runtime refuses images on its text-only
    /// checkpoints, the vision-capable ones are a family of their own.
    public enum TextChatFamily: String, CaseIterable, Sendable, MereRunFamilyID {
        case gemma4
        case gemma4Unified = "gemma4-unified"
        case diffusionGemma = "diffusion-gemma"
        case laguna
        case inkling
        case museGlimmer = "muse-glimmer"
        case nemotronH = "nemotron-h"
        case nemotronOmni = "nemotron-omni"
        case lfm2
        /// The 8-bit LFM2.5 A1B mixture of experts: the one LFM2.5 runtime that loads text LoRA
        /// adapters.
        case lfm2A1B = "lfm2-a1b"
        case lfm2VL = "lfm2-vl"
        case q35
        case q35VL = "q35-vl"
        case q38
        case gguf
        case psi

        /// The family that runs a managed model id, from the contract's exact ids; `nil` for an id
        /// the contract does not list.
        public init?(managedModel id: String) {
            guard let family = textChatRouting.families.first(where: { $0.models.contains(id) }) else {
                return nil
            }
            self.init(rawValue: family.id)
        }
    }

    /// The trainers `text train-lora` runs.
    public enum TextTrainLoRAFamily: String, CaseIterable, Sendable, MereRunFamilyID {
        case gemma4
        case gemma4VLM = "gemma4-vlm"
        case lagunaXS = "laguna-xs"
        case inkling
        case lfm2A1B = "lfm2-a1b"
    }

    static let textChatRouting = MereRunCapabilityRouting(
        modelFlags: ["--model"],
        defaultModels: [
            // The CLI picks by unified memory: Gemma 4 12B 4-bit where it fits, else Nano.
            .init(models: ["text-chat-gemma4-12b-4bit", "text-chat-gemma4-nano"], platforms: ["macos"]),
            // Linux tries Qwen3.6 A3B first (GGUF on CUDA, MLX otherwise), then the same Gemma 4 steps.
            .init(
                models: [
                    "text-chat-q36-nano-gguf", "text-chat-q36-nano", "text-chat-gemma4-12b-4bit", "text-chat-gemma4-nano"
                ],
                platforms: ["linux"]
            )
        ],
        families: [
            .init(
                TextChatFamily.gemma4,
                title: "Gemma 4",
                models: [
                    "text-chat-gemma4", "text-chat-gemma4-turbo", "text-chat-gemma4-12b", "text-chat-gemma4-12b-4bit",
                    "text-chat-gemma4-nano", "text-chat-gemma4-max"
                ]
            ),
            .init(TextChatFamily.gemma4Unified, title: "Gemma 4 12B vision", models: ["vision-chat-gemma4-12b"]),
            .init(TextChatFamily.diffusionGemma, title: "DiffusionGemma", models: ["text-chat-diffusiongemma-26b-optiq-4bit"]),
            .init(TextChatFamily.laguna, title: "Laguna", models: ["text-chat-laguna-s-2-1", "text-chat-laguna-xs-2-1"]),
            .init(TextChatFamily.inkling, title: "Inkling-Small", models: ["text-chat-inkling-small"]),
            .init(TextChatFamily.museGlimmer, title: "Muse Glimmer", models: ["vision-chat-muse-glimmer-30b"]),
            .init(TextChatFamily.nemotronH, title: "Nemotron 3.5 Lightning", models: ["text-chat-nemotron-35-lightning"]),
            .init(
                TextChatFamily.nemotronOmni, title: "Nemotron 3 Nano Omni", models: ["omni-chat-nemotron3-nano-30b-a3b-bf16"]
            ),
            .init(
                TextChatFamily.lfm2,
                title: "LFM2.5",
                models: [
                    "text-chat-lfm25-a1b-bf16", "text-chat-lfm25-1.2b-bf16", "text-chat-lfm25-1.2b-qad-4bit",
                    "text-chat-lfm25-2.6b-4bit", "text-chat-lfm25-2.6b-qad-4bit", "text-chat-lfm25-2.6b-bf16"
                ]
            ),
            .init(TextChatFamily.lfm2A1B, title: "LFM2.5 A1B 8-bit", models: ["text-chat-lfm25-a1b-8bit"]),
            .init(TextChatFamily.lfm2VL, title: "LFM2.5-VL", models: ["vision-chat-lfm25-3b-8bit", "vision-chat-lfm25-3b-bf16"]),
            // Checkpoints whose config carries no vision tower.
            .init(
                TextChatFamily.q35,
                title: "Qwen3.6 text",
                models: [
                    "text-chat-q36-nano", "text-agent-ornith-9b", "text-agent-ornith-35b-mlx-6bit",
                    "text-agent-ornith-35b-mlx-8bit", "text-agent-ornith-35b-mlx"
                ]
            ),
            // A vision tower in the checkpoint's config, or in the vision companion it installs.
            .init(
                TextChatFamily.q35VL,
                title: "Qwen3.6 vision",
                models: [
                    "text-chat-bonsai-27b-1bit", "text-chat-bonsai-27b-2bit", "text-chat-bonsai-2-27b-2bit",
                    "text-agent-ornith-35b-mlx-4bit", "vision-chat-ornith-35b"
                ]
            ),
            .init(
                TextChatFamily.q38,
                title: "Qwen3.8",
                models: [
                    "vision-chat-q38-27b", "vision-chat-q38-27b-4bit", "vision-chat-q38-flash-next-mixed",
                    "vision-chat-q38-flash-next-3bit", "vision-chat-q38-flash-next-3bit-native-ple",
                    "vision-chat-q38-flash-next-4bit"
                ]
            ),
            .init(TextChatFamily.gguf, title: "llama.cpp GGUF", models: ["text-chat-q36-nano-gguf"]),
            .init(TextChatFamily.psi, title: "Psi", models: ["text-chat-psi-agent"])
        ],
        excludedModels: .models(
            ["text-agent-deepseek-v4-flash"],
            reason: "DeepSeek V4 Flash runs on its llama.cpp server; use `api serve` or `agent start`."
        )
        .and(["text-chat-mebot"], reason: "MeBot chat runs only through `api serve`.")
        .and(
            ["text-encoder-ltx-gemma3-12b-4bit"],
            reason: "It is the LTX video text encoder that video generation loads, not a chat model."
        )
        .and(
            ["text-chat-gemma4-12b-mtp"],
            reason: "It is the drafter Gemma 4 12B loads itself; pass `--model text-chat-gemma4-12b-4bit`."
        )
        .and(
            ["text-chat-laguna-s-2-1-dflash"],
            reason: "It is the drafter Laguna S 2.1 loads itself; pass `--model text-chat-laguna-s-2-1`."
        )
        .and(
            ["vision-chat-muse-glimmer-30b-assistant", "vision-chat-muse-glimmer-30b-dflash2"],
            reason: "It is a drafter Muse Glimmer loads itself; pass `--model vision-chat-muse-glimmer-30b`."
        )
        .and(
            ["text-chat-nemotron-35-lightning-dspark"],
            reason: "It is the drafter Nemotron 3.5 Lightning loads itself; pass `--model text-chat-nemotron-35-lightning`."
        )
        .and(
            ["text-chat-lfm25-a1b-dspark", "text-chat-lfm25-1.2b-dspark", "text-chat-lfm25-2.6b-dspark", "vision-chat-lfm25-3b-dspark"],
            reason: "It is a drafter its LFM2.5 BF16 model loads itself; pass that model."
        )
        .and(
            ["text-agent-ornith-35b-mtp"],
            reason: "It is the multi-token-prediction head the Ornith 1.5 35B MLX models load themselves; pass one of them."
        )
    )

    static let textTrainLoRARouting = MereRunCapabilityRouting(
        modelFlags: ["--model"],
        defaultModels: [.always("text-chat-gemma4-12b-4bit")],
        families: [
            .init(
                TextTrainLoRAFamily.gemma4,
                title: "Gemma 4",
                models: [
                    "text-chat-gemma4", "text-chat-gemma4-turbo", "text-chat-gemma4-12b", "text-chat-gemma4-12b-4bit",
                    "text-chat-gemma4-nano", "text-chat-gemma4-max"
                ]
            ),
            .init(TextTrainLoRAFamily.gemma4VLM, title: "Gemma 4 12B vision", models: ["vision-chat-gemma4-12b"]),
            .init(TextTrainLoRAFamily.lagunaXS, title: "Laguna XS 2.1", models: ["text-chat-laguna-xs-2-1"]),
            .init(TextTrainLoRAFamily.inkling, title: "Inkling-Small", models: ["text-chat-inkling-small"]),
            .init(TextTrainLoRAFamily.lfm2A1B, title: "LFM2.5 A1B 8-bit", models: ["text-chat-lfm25-a1b-8bit"])
        ],
        excludedModels: .models(
            ["text-chat-laguna-s-2-1"],
            reason: "Only Laguna XS 2.1 has a LoRA trainer; pass `--model text-chat-laguna-xs-2-1`."
        )
        .and(
            ["text-chat-gemma4-12b-mtp"],
            reason: "It is the drafter Gemma 4 12B loads, not a base model; train `text-chat-gemma4-12b-4bit`."
        )
    )

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
