import Foundation
import MereRunCoreMLModel

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
        let config = try JSONDecoder().decode(BundleConfig.self, from: Data(contentsOf: bundle.appending(path: "config.json")))
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
        for (name, bytes) in [("embeddings.bf16", 262144 * 512 * 2), ("position_embeddings.f16", 2 * 1024 * 768 * 2)] {
            let size = try FileManager.default.attributesOfItem(atPath: root.appending(path: name).resolvingSymlinksInPath().path)[.size] as? NSNumber
            guard size?.intValue == bytes else {
                throw EmbeddingGemma2Error.invalidConfiguration("Invalid Core ML table size: \(name).")
            }
        }
    }
}

public enum EmbeddingGemma2CoreMLRuntime {
    public static var isAvailable: Bool {
        #if canImport(CoreML) && os(macOS) && arch(arm64)
        if #available(macOS 15.0, *) { return true }
        #endif
        return false
    }

    public static func requireAvailable() throws {
        guard isAvailable else {
            throw EmbeddingGemma2Error.invalidConfiguration(
                "EmbeddingGemma 2 Core ML requires Apple Silicon macOS 15 or later and Core ML support in this build."
            )
        }
    }

    public static func validateInputs(_ inputs: [EmbeddingGemma2Input], task: EmbeddingGemma2Task,
                                      title: String?, dimensions: Int, maxTokens: Int?) throws {
        guard !inputs.isEmpty, inputs.allSatisfy({ !$0.content.isEmpty }),
              [128, 256, 512, 768].contains(dimensions), title == nil || task == .document,
              maxTokens == nil || maxTokens! >= 2 else {
            throw EmbeddingGemma2Error.invalidInput("Provide nonempty inputs, 128/256/512/768 dimensions, and at least two tokens. A title requires --task document.")
        }
        for part in inputs.flatMap(\.content) {
            switch part {
            case .video, .videoFrames:
                throw EmbeddingGemma2Error.invalidInput("The pinned Core ML export supports text, images, and audio. Use text-embed-embeddinggemma2 for video.")
            case .image(let url), .audio(let url):
                guard url.isFileURL else { throw EmbeddingGemma2Error.invalidInput("Media inputs require local file URLs.") }
            case .text: break
            }
        }
    }

    public static func embed(root: URL, inputs: [EmbeddingGemma2Input], task: EmbeddingGemma2Task = .raw,
                             title: String? = nil, dimensions: Int = 768, maxTokens: Int? = nil) async throws
        -> (embeddings: [[Float]], tokenCounts: [Int]) {
        try validateInputs(inputs, task: task, title: title, dimensions: dimensions, maxTokens: maxTokens)
        try requireAvailable()
        let missing = EmbeddingGemma2CoreMLCatalog.missingFiles(root: root)
        guard missing.isEmpty else {
            throw EmbeddingGemma2Error.invalidConfiguration("Missing Core ML resources:\n" + missing.map(\.path).joined(separator: "\n"))
        }
        try EmbeddingGemma2CoreMLCatalog.validateConfiguration(root: root)
        #if canImport(CoreML) && os(macOS) && arch(arm64)
        if #available(macOS 15.0, *) {
            return try await execute(root: EmbeddingGemma2CoreMLCatalog.bundleRoot(root), inputs: inputs,
                                     task: task, title: title, dimensions: dimensions, maxTokens: maxTokens)
        }
        #endif
        throw EmbeddingGemma2Error.invalidConfiguration("Core ML is unavailable in this build.")
    }
}
