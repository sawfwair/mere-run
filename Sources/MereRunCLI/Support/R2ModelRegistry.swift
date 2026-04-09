import Foundation
import MereRunCore

/// Maps model IDs to R2 archive keys.
enum R2ModelRegistry {
    struct ModelEntry {
        let id: String
        let category: String
        let archiveKey: String
        let packagedArchiveKey: String?
    }

    static let allEntries: [ModelEntry] = [
        // Image generation
        ModelEntry(id: "image-klein-nano",  category: "image", archiveKey: "models/image-klein-nano.tar.gz", packagedArchiveKey: "models/zero-nano.tar.gz"),
        ModelEntry(id: "image-klein-max",   category: "image", archiveKey: "models/image-klein-max.tar.gz", packagedArchiveKey: "models/zero-max.tar.gz"),
        ModelEntry(id: "image-klein-base",  category: "image", archiveKey: "models/image-klein-base.tar.gz", packagedArchiveKey: "models/zero-base.tar.gz"),
        ModelEntry(id: "image-zimage-nano", category: "image", archiveKey: "models/image-zimage-nano.tar.gz", packagedArchiveKey: "models/zeta-nano.tar.gz"),
        ModelEntry(id: "image-zimage-max",  category: "image", archiveKey: "models/image-zimage-max.tar.gz", packagedArchiveKey: "models/zeta-max.tar.gz"),
        ModelEntry(id: "image-zimage-base", category: "image", archiveKey: "models/image-zimage-base.tar.gz", packagedArchiveKey: nil),
        // Chat
        ModelEntry(id: "text-chat-gemma4-nano", category: "text-chat", archiveKey: "models/text-chat-gemma4-nano.tar.gz", packagedArchiveKey: "models/gemma4-nano.tar.gz"),
        ModelEntry(id: "text-chat-gemma4-max",  category: "text-chat", archiveKey: "models/text-chat-gemma4-max.tar.gz", packagedArchiveKey: "models/gemma4-max.tar.gz"),
        ModelEntry(id: "text-chat-mebot",      category: "text-chat", archiveKey: "models/text-chat-mebot.tar.gz", packagedArchiveKey: "models/mebot-instruct.tar.gz"),
        ModelEntry(id: "text-chat-psi-agent",  category: "text-chat", archiveKey: "models/text-chat-psi-agent.tar.gz", packagedArchiveKey: "models/psi-agent.tar.gz"),
        ModelEntry(id: "text-chat-q35",        category: "text-chat", archiveKey: "models/text-chat-q35.tar.gz", packagedArchiveKey: "models/q35.tar.gz"),
        ModelEntry(id: "text-chat-q35-nano",   category: "text-chat", archiveKey: "models/text-chat-q35-nano.tar.gz", packagedArchiveKey: nil),
        // TTS
        ModelEntry(id: "speech-tts-qwen3-nano",         category: "speech-tts", archiveKey: "models/speech-tts-qwen3-nano.tar.gz", packagedArchiveKey: "models/talk-nano.tar.gz"),
        ModelEntry(id: "speech-tts-qwen3-customvoice",  category: "speech-tts", archiveKey: "models/speech-tts-qwen3-customvoice.tar.gz", packagedArchiveKey: nil),
        // OCR
        ModelEntry(id: "vision-ocr-lighton", category: "vision-ocr", archiveKey: "models/vision-ocr-lighton.tar.gz", packagedArchiveKey: "models/ocr.tar.gz"),
        // Segmentation
        ModelEntry(id: "vision-segment-sam31", category: "vision-segment", archiveKey: "models/vision-segment-sam31.tar.gz", packagedArchiveKey: nil),
        // ASR
        ModelEntry(id: "speech-asr-qwen3",      category: "speech-asr", archiveKey: "models/speech-asr-qwen3.tar.gz", packagedArchiveKey: "models/asr.tar.gz"),
        ModelEntry(id: "speech-asr-parakeet",   category: "speech-asr", archiveKey: "models/speech-asr-parakeet.tar.gz", packagedArchiveKey: "models/asr-parakeet.tar.gz"),
        // Music
        ModelEntry(id: "music-acestep", category: "music", archiveKey: "models/music-acestep.tar.gz", packagedArchiveKey: "models/acestep-v15-full.tar.gz"),
        // Video
        ModelEntry(id: "video-ltx-av", category: "video", archiveKey: "models/video-ltx-av.tar.gz", packagedArchiveKey: "models/ltx-video-av.tar.gz"),
        // Code
        ModelEntry(id: "text-code-qwen3", category: "text-code", archiveKey: "models/text-code-qwen3.tar.gz", packagedArchiveKey: "models/qwen3-coder-next.tar.gz"),
        // Embeddings
        ModelEntry(id: "text-embed-qwen3-0.6b", category: "text-embed", archiveKey: "models/text-embed-qwen3-0.6b.tar.gz", packagedArchiveKey: "models/qwen3-embedding-0.6b.tar.gz"),
    ]

    /// Look up an entry by its canonical id string.
    static func entry(for id: String) -> ModelEntry? {
        if id == ModelResolver.ModelID.gemma4.rawValue {
            return ModelEntry(
                id: id,
                category: "text-chat",
                archiveKey: "models/text-chat-gemma4-max.tar.gz",
                packagedArchiveKey: "models/gemma4-max.tar.gz"
            )
        }
        return allEntries.first { $0.id == id }
    }

    /// Backward-compat: R2 object key for a `ModelResolver.ModelID`.
    static func archiveKey(for modelID: ModelResolver.ModelID) -> String? {
        entry(for: modelID.rawValue)?.archiveKey
    }
}
