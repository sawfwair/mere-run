import Foundation

public enum EmbeddingGemma2CoreMLCatalog {
    public static let modelID = "text-embed-embeddinggemma2-coreml"
    public static let repository = "FluidInference/embeddinggemma-2-coreml"
    public static let revision = "3f978ddf92cac9aa07eb2dcf6eb36cf450f3dba4"
    public static let hubFallback = HubFallbackConfig(
        repoId: repository, revision: revision,
        patterns: assets.map { $0 + "/*" } + ["config.json", "tokenizer.json", "embeddings.bf16", "position_embeddings.f16", "README.md"]
    )
    public static let assets = ["EmbeddingGemma2Text.mlpackage", "EmbeddingGemma2Vision.mlpackage", "EmbeddingGemma2Audio.mlpackage"]

    public static func bundleRoot(_ root: URL, fileManager: FileManager = .default) -> URL { root }

    public static func missingFiles(root: URL, fileManager: FileManager = .default) -> [URL] {
        let paths = ["config.json", "tokenizer.json", "embeddings.bf16", "position_embeddings.f16"]
            + assets.flatMap { asset in ["Manifest.json", "Data/com.apple.CoreML/model.mlmodel", "Data/com.apple.CoreML/weights/weight.bin"].map { asset + "/" + $0 } }
        return paths.map { root.appending(path: $0) }.filter { !fileManager.fileExists(atPath: $0.path) }
    }

    public static func validateConfiguration(root: URL) throws {
        let bundle = bundleRoot(root)
        let config = try JSONDecoder().decode(
            CoreMLBundleConfig.self,
            from: Data(contentsOf: bundle.appending(path: "config.json"))
        )
        try config.validate()
        if let extended = config.extended {
            for package in [extended.text_package, extended.audio_package] {
                for file in ["Manifest.json", "Data/com.apple.CoreML/model.mlmodel", "Data/com.apple.CoreML/weights/weight.bin"] {
                    guard FileManager.default.fileExists(atPath: bundle.appending(path: package + "/" + file).path) else {
                        throw EmbeddingGemma2Error.invalidConfiguration("Missing expanded Core ML resource: \(package)/\(file).")
                    }
                }
            }
            guard FileManager.default.fileExists(atPath: bundle.appending(path: "conversion.json").path) else {
                throw EmbeddingGemma2Error.invalidConfiguration("Expanded Core ML bundle is missing conversion.json.")
            }
        }
        for (name, bytes) in [("embeddings.bf16", 262_144 * 512 * 2), ("position_embeddings.f16", 2 * 1_024 * 768 * 2)] {
            let size = try FileManager.default.attributesOfItem(atPath: root.appending(path: name).resolvingSymlinksInPath().path)[.size] as? NSNumber
            guard size?.intValue == bytes else {
                throw EmbeddingGemma2Error.invalidConfiguration("Invalid Core ML table size: \(name).")
            }
        }
    }

    private struct CoreMLBundleConfig: Decodable {
        struct TokenTable: Decodable { let file: String; let dtype: String; let shape: [Int] }
        struct Function: Decodable { let tokens: Int; let slots: Int?; let patches: Int? }
        struct Audio: Decodable {
            let package: String; let sample_rate: Int; let window_seconds: Int; let waveform_samples: Int
            let left_pad_samples: Int; let frames: Int; let tokens_per_window: Int
            let boa_token_id: Int32; let eoa_token_id: Int32
        }
        struct Vision: Decodable {
            let package: String; let functions: [String: Function]; let patch_size: Int; let pooling_kernel: Int
            let position_table: TokenTable; let boi_token_id: Int32; let eoi_token_id: Int32
        }
        struct Extended: Decodable {
            let version: Int; let text_package: String; let lengths: [Int]; let audio_package: String
            let audio_frames: Int; let audio_seconds: Int; let source_revision: String; let source_sha256: String
        }
        let base_model: String; let format: String; let hidden_size: Int; let embedding_dim: Int
        let matryoshka_dims: [Int]; let embed_scale: Float; let token_table: TokenTable
        let functions: [String: Function]; let bos_token_id: Int32; let eos_token_id: Int32; let pad_token_id: Int32
        let audio: Audio; let vision: Vision; let extended: Extended?

        static let lengths = [32, 48, 64, 128, 256, 512]
        static let extendedLengths = [1_024, 2_048, 4_096, 8_192]

        func validate() throws {
            guard base_model == "google/embeddinggemma-2", format == "coreml", hidden_size == 512, embedding_dim == 768,
                  Set(matryoshka_dims) == Set([128, 256, 512, 768]), abs(embed_scale - sqrt(Float(512))) < 0.00001,
                  token_table.file == "embeddings.bf16", token_table.dtype == "bfloat16", token_table.shape == [262_144, 512],
                  Set(functions.keys) == Set(Self.lengths.map { "embed_\($0)" } + ["pack_256"]),
                  Self.lengths.allSatisfy({ functions["embed_\($0)"]?.tokens == $0 }),
                  functions["pack_256"]?.tokens == 256, functions["pack_256"]?.slots == 8,
                  bos_token_id == 2, eos_token_id == 1, pad_token_id == 0,
                  audio.package == "EmbeddingGemma2Audio.mlpackage", audio.sample_rate == 16_000, audio.window_seconds == 10,
                  audio.waveform_samples == 160_160, audio.left_pad_samples == 160, audio.frames == 1_000,
                  audio.tokens_per_window == 250, audio.boa_token_id == 256_000, audio.eoa_token_id == 258_883,
                  vision.package == "EmbeddingGemma2Vision.mlpackage", vision.patch_size == 16, vision.pooling_kernel == 3,
                  Set(vision.functions.keys) == Set(["vision_70", "vision_140", "vision_280"]),
                  [70, 140, 280].allSatisfy({ vision.functions["vision_\($0)"]?.tokens == $0 && vision.functions["vision_\($0)"]?.patches == $0 * 9 }),
                  vision.position_table.file == "position_embeddings.f16", vision.position_table.dtype == "float16",
                  vision.position_table.shape == [2, 1_024, 768], vision.boi_token_id == 255_999, vision.eoi_token_id == 258_882 else {
                throw EmbeddingGemma2Error.invalidConfiguration("Unsupported EmbeddingGemma 2 Core ML bundle configuration; use the pinned managed artifact.")
            }
            if let extended {
                guard extended.version == 1, extended.text_package == "EmbeddingGemma2TextExtended.mlpackage",
                      extended.lengths == Self.extendedLengths, extended.audio_package == "EmbeddingGemma2AudioExtended.mlpackage",
                      extended.audio_frames == 3_000, extended.audio_seconds == 30,
                      extended.source_revision == "914f7f89142e33e77833254d9c9b90c3cef7303b",
                      extended.source_sha256 == "197a32965d4b1105faf060417baa899e193fb73cd401f42ec9295234d5553d79" else {
                    throw EmbeddingGemma2Error.invalidConfiguration("Unsupported expanded Core ML bundle; run the pinned EmbeddingGemma conversion tool.")
                }
            }
        }
    }
}
