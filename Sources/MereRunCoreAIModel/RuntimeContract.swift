// Adapted from EverPaper/embeddinggemma-2-coreai@305fc2a96790d9343f38038aeca211dfca56d9a5
// Copyright 2026 Everpaper. Apache-2.0; see LICENSE in this directory.
import Foundation

public struct BundleConfig: Decodable, Sendable {
    public struct Vision: Decodable, Sendable {
        public let asset: String
        public let function: String
        public let max_patches: Int
        public let max_soft_tokens: Int
    }
    public struct Audio: Decodable, Sendable {
        public let asset: String
        public let functions: [String]
        public let frame_buckets: [Int]
        public let max_soft_tokens: Int
    }
    /// The text asset (`mm_<S>` functions: text, images, or both in one sequence).
    public let asset: String
    public let functions: [String]
    public let buckets: [Int]
    public let max_tokens: Int
    public let vision: Vision
    /// The audio asset (`audio_<Tb>` functions); absent in a 1.1 bundle.
    public let audio: Audio?
    public let hidden_size: Int
    public let embedding_dim: Int
    public let matryoshka_dims: [Int]
    public let prompts: [String: String]
    public let bos_token_id: Int32
    public let eos_token_id: Int32
    public let pad_token_id: Int32
    public let boi_token_id: Int32
    public let image_token_id: Int32
    public let eoi_token_id: Int32
    public let boa_token_id: Int32?
    public let audio_token_id: Int32?
    public let eoa_token_id: Int32?
}

public struct EmbedderError: LocalizedError, CustomStringConvertible {
    public let description: String
    public init(description: String) { self.description = description }
    public var errorDescription: String? { description }
}

extension BundleConfig {
    public func validate() throws {
        guard asset == "embeddinggemma-2-text-mm.aimodel", hidden_size == 512, embedding_dim == 768,
              buckets == [32, 64, 128, 288, 512, 1024, 2048], max_tokens == 2048,
              functions == buckets.map({ "mm_\($0)" }), Set(matryoshka_dims) == Set([128, 256, 512, 768]),
              bos_token_id == 2, eos_token_id == 1, pad_token_id == 0,
              boi_token_id == 255999, image_token_id == 258880, eoi_token_id == 258882,
              vision.asset == "embeddinggemma-2-vision.aimodel", vision.function == "vision_2520",
              vision.max_patches == 2520, vision.max_soft_tokens == 280,
              let audio, audio.asset == "embeddinggemma-2-audio.aimodel",
              audio.frame_buckets == [192, 384, 1024, 1536, 3000],
              audio.functions == audio.frame_buckets.map({ "audio_\($0)" }), audio.max_soft_tokens == 750,
              boa_token_id == 256000, audio_token_id == 258881, eoa_token_id == 258883 else {
            throw EmbedderError(description: "Unsupported EmbeddingGemma 2 Core AI bundle configuration; use the pinned managed artifact.")
        }
    }
}

/// A prepared sequence for `mm_<S>`: token ids with placeholder slots, and the soft tokens that fill them.
public struct MultimodalSequence: Sendable {
    public var ids: [Int32]
    public var softTokens: [Float16]   // [slots * 512] in sequence order
    public var slots: [Int]            // positions in `ids` that receive soft tokens (same count as rows)

    public init(ids: [Int32], softTokens: [Float16] = [], slots: [Int] = []) {
        self.ids = ids
        self.softTokens = softTokens
        self.slots = slots
    }
}


extension MultimodalSequence {
    /// Append one complete per-media sequence after stripping its BOS/EOS.
    public mutating func appendMedia(_ media: MultimodalSequence) {
        let offset = ids.count - 1
        ids += media.ids.dropFirst().dropLast()
        slots += media.slots.map { $0 + offset }
        softTokens += media.softTokens
    }
}
