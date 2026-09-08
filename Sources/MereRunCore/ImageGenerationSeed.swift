import Foundation

/// Preserve each runtime's default seed policy when a run records its seed
/// before inference. Prompt-derived seeds use the existing FNV-1a algorithm.
enum ImageGenerationSeed {
    static func resolve(_ seed: UInt64?, prompt: String, backend: ImageGenerationBackend) -> UInt64 {
        if let seed { return seed }
        switch backend {
        case .flux2Klein, .zImageTurbo, .qwenImageEdit:
            return UInt64.random(in: 0..<UInt64.max)
        case .flux1, .hiDreamO1, .senseNovaU15, .krea2, .ideogram4:
            return prompt.utf8.reduce(UInt64(0xcbf2_9ce4_8422_2325)) { ($0 ^ UInt64($1)) &* 0x100_0000_01b3 }
        }
    }
}
