import Foundation

public struct EmbedderError: LocalizedError, CustomStringConvertible {
    public let description: String
    public init(description: String) { self.description = description }
    public var errorDescription: String? { description }
}

/// Published Core ML artifact geometry; no model loading or tensor evaluation.
public struct BundleConfig: Decodable, Sendable {
    public struct TokenTable: Decodable, Sendable { public let file: String; public let dtype: String; public let shape: [Int] }
    public struct Function: Decodable, Sendable { public let tokens: Int; public let slots: Int?; public let patches: Int? }
    public struct Audio: Decodable, Sendable {
        public let package: String
        public let sample_rate: Int
        public let window_seconds: Int
        public let waveform_samples: Int
        public let left_pad_samples: Int
        public let frames: Int
        public let tokens_per_window: Int
        public let boa_token_id: Int32
        public let eoa_token_id: Int32
    }
    public struct Vision: Decodable, Sendable {
        public let package: String
        public let functions: [String: Function]
        public let patch_size: Int
        public let pooling_kernel: Int
        public let position_table: TokenTable
        public let boi_token_id: Int32
        public let eoi_token_id: Int32
    }
    public let base_model: String
    public let format: String
    public let hidden_size: Int
    public let embedding_dim: Int
    public let matryoshka_dims: [Int]
    public let embed_scale: Float
    public let token_table: TokenTable
    public let functions: [String: Function]
    public let bos_token_id: Int32
    public let eos_token_id: Int32
    public let pad_token_id: Int32
    public let audio: Audio
    public let vision: Vision
    public static let lengths = [32, 48, 64, 128, 256, 512]

    public func validate() throws {
        guard base_model == "google/embeddinggemma-2", format == "coreml", hidden_size == 512, embedding_dim == 768,
              Set(matryoshka_dims) == Set([128, 256, 512, 768]), abs(embed_scale - sqrt(Float(512))) < 0.00001,
              token_table.file == "embeddings.bf16", token_table.dtype == "bfloat16", token_table.shape == [262144, 512],
              Set(functions.keys) == Set(Self.lengths.map { "embed_\($0)" } + ["pack_256"]),
              Self.lengths.allSatisfy({ functions["embed_\($0)"]?.tokens == $0 }),
              functions["pack_256"]?.tokens == 256, functions["pack_256"]?.slots == 8,
              bos_token_id == 2, eos_token_id == 1, pad_token_id == 0,
              audio.package == "EmbeddingGemma2Audio.mlpackage", audio.sample_rate == 16000, audio.window_seconds == 10,
              audio.waveform_samples == 160160, audio.left_pad_samples == 160, audio.frames == 1000,
              audio.tokens_per_window == 250, audio.boa_token_id == 256000, audio.eoa_token_id == 258883,
              vision.package == "EmbeddingGemma2Vision.mlpackage", vision.patch_size == 16, vision.pooling_kernel == 3,
              Set(vision.functions.keys) == Set(["vision_70", "vision_140", "vision_280"]),
              [70, 140, 280].allSatisfy({ vision.functions["vision_\($0)"]?.tokens == $0 && vision.functions["vision_\($0)"]?.patches == $0 * 9 }),
              vision.position_table.file == "position_embeddings.f16", vision.position_table.dtype == "float16",
              vision.position_table.shape == [2, 1024, 768], vision.boi_token_id == 255999, vision.eoi_token_id == 258882 else {
            throw EmbedderError(description: "Unsupported EmbeddingGemma 2 Core ML bundle configuration; use the pinned managed artifact.")
        }
    }
}

public struct MultimodalSequence: Sendable {
    public var ids: [Int32]
    public var softTokens: [Float16]
    public var slots: [Int]
    public init(ids: [Int32], softTokens: [Float16] = [], slots: [Int] = []) {
        self.ids = ids; self.softTokens = softTokens; self.slots = slots
    }
    public mutating func appendMedia(_ media: MultimodalSequence) {
        let offset = ids.count - 1
        ids += media.ids.dropFirst().dropLast()
        slots += media.slots.map { $0 + offset }
        softTokens += media.softTokens
    }
}

public struct TokenEmbeddingTable: Sendable {
    private let data: Data
    public let rows: Int
    public let columns: Int
    public let scale: Float
    public init(data: Data, rows: Int, columns: Int, scale: Float) throws {
        guard rows > 0, columns > 0, data.count == rows * columns * 2 else {
            throw EmbedderError(description: "Invalid bfloat16 token embedding table size.")
        }
        self.data = data; self.rows = rows; self.columns = columns; self.scale = scale
    }
    public func embeddings(ids: [Int32]) throws -> [Float16] {
        guard ids.allSatisfy({ $0 >= 0 && Int($0) < rows }) else {
            throw EmbedderError(description: "Token ID is outside the bfloat16 embedding table.")
        }
        return data.withUnsafeBytes { raw in
            let words = raw.bindMemory(to: UInt16.self)
            return ids.flatMap { id in
                (0..<columns).map { column in
                    Float16(Float(bitPattern: UInt32(UInt16(littleEndian: words[Int(id) * columns + column])) << 16) * scale)
                }
            }
        }
    }
}

public enum CoreMLAudioGeometry {
    public static func validFrames(samples: Int) -> Int {
        let count = min(samples, 160000)
        return count < 161 ? 0 : min(1000, (160 + count - 321) / 160 + 1)
    }
    public static func tokenCount(samples: Int) -> Int { (validFrames(samples: samples) + 3) / 4 }
}
