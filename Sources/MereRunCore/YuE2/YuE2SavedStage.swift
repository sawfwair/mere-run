import Foundation
import MLX

/// Portable, immutable stage receipts. Native RNG identity prevents accidental cross-runtime replay claims.
public struct YuE2SavedStage: Codable, Sendable {
    public enum Stage: String, Codable, CaseIterable, Sendable { case score, semantic, acoustic }
    public let schemaVersion: Int
    public let runtime: String
    public let sourceRevision: String
    public let modelSHA256: String
    public let tokenizerSHA256: String
    public let configurationSHA256: String
    public let stage: Stage
    public let plan: YuE2GenerationPlan
    public let abc: String?
    public let scoreTokens: [Int]
    public let codecTokens: [Int]
    public let scoreTruncated: Bool
    public let musicTruncated: Bool
    public var files: [String: String]

    init(stage: Stage, plan: YuE2GenerationPlan, abc: String?, scoreTokens: [Int], codecTokens: [Int],
         scoreTruncated: Bool, musicTruncated: Bool, resources: YuE2Resources) throws {
        schemaVersion = 1
        runtime = "native-swift-mlx-request-local-v1"
        sourceRevision = YuE2Resources.sourceRevision
        modelSHA256 = try ModelArtifactPin.fileSHA256(resources.rootURL.appendingPathComponent("model.safetensors"))
        tokenizerSHA256 = try ModelArtifactPin.fileSHA256(resources.rootURL.appendingPathComponent("qwen.tiktoken"))
        configurationSHA256 = try ModelArtifactPin.fileSHA256(resources.rootURL.appendingPathComponent("config.json"))
        self.stage = stage
        self.plan = plan
        self.abc = abc
        self.scoreTokens = scoreTokens
        self.codecTokens = codecTokens
        self.scoreTruncated = scoreTruncated
        self.musicTruncated = musicTruncated
        files = [:]
    }

    public func validate() throws {
        guard schemaVersion == 1, runtime == "native-swift-mlx-request-local-v1",
              modelSHA256.count == 64, tokenizerSHA256.count == 64, configurationSHA256.count == 64 else {
            throw YuE2Error.invalidRequest("Unsupported saved stage or incomplete model identity.")
        }
        try plan.validate()
        try YuE2Protocol.validateScore(scoreTokens)
        guard (plan.planning == .off ? abc == nil && scoreTokens.isEmpty : abc != nil),
              codecTokens.count <= plan.semanticSampling.maximumTokens,
              codecTokens.allSatisfy({ (0..<YuE2Protocol.codecSize).contains($0) }),
              stage == .score ? codecTokens.isEmpty : !codecTokens.isEmpty else {
            throw YuE2Error.invalidRequest("Saved stage does not match its score or codec-token contract.")
        }
        let expected = Set((abc == nil ? [] : ["score.abc"]) + (stage == .acoustic ? ["latent.npy"] : []))
        guard Set(files.keys) == expected else {
            throw YuE2Error.invalidRequest("Saved stage has missing or unexpected artifact filenames.")
        }
    }

    /// Inspection reads no models. Hashes establish artifact consistency, not independent provenance.
    public static func load(from directory: URL) throws -> Self {
        let value = try JSONDecoder().decode(Self.self, from: Data(contentsOf: directory.appendingPathComponent("stage.json")))
        try value.validate()
        for (name, hash) in value.files {
            let url = directory.appendingPathComponent(name)
            guard url.standardizedFileURL.resolvingSymlinksInPath().deletingLastPathComponent().path
                    == directory.standardizedFileURL.resolvingSymlinksInPath().path,
                  try ModelArtifactPin.fileSHA256(url) == hash else {
                throw YuE2Error.invalidRequest("Saved artifact integrity check failed for \(name).")
            }
        }
        if let abc = value.abc {
            guard try String(contentsOf: directory.appendingPathComponent("score.abc"), encoding: .utf8) == abc else {
                throw YuE2Error.invalidRequest("Saved score differs from the stage receipt.")
            }
        }
        return value
    }

    func verify(resources: YuE2Resources, tokenizer: YuE2Tokenizer) throws {
        try validate()
        guard try ModelArtifactPin.fileSHA256(resources.rootURL.appendingPathComponent("model.safetensors")) == modelSHA256,
              try ModelArtifactPin.fileSHA256(resources.rootURL.appendingPathComponent("config.json")) == configurationSHA256,
              try ModelArtifactPin.fileSHA256(resources.rootURL.appendingPathComponent("qwen.tiktoken")) == tokenizerSHA256,
              try abc.map(tokenizer.encode) ?? [] == scoreTokens else {
            throw YuE2Error.invalidRequest("Saved stage requires its original model, tokenizer, configuration and exact score.")
        }
    }

    func save(to directory: URL, latents: MLXArray? = nil) throws {
        let manager = FileManager.default
        let parent = directory.deletingLastPathComponent()
        try manager.createDirectory(at: parent, withIntermediateDirectories: true)
        let temporary = parent.appendingPathComponent(".yue2-\(UUID().uuidString)")
        try manager.createDirectory(at: temporary, withIntermediateDirectories: false)
        defer { try? manager.removeItem(at: temporary) }
        var receipt = self
        if let abc {
            try Data(abc.utf8).write(to: temporary.appendingPathComponent("score.abc"))
            receipt.files["score.abc"] = try ModelArtifactPin.fileSHA256(temporary.appendingPathComponent("score.abc"))
        }
        if let latents {
            try Self.validateLatents(latents, frames: codecTokens.count)
            try MLX.save(array: latents.asType(.float32), url: temporary.appendingPathComponent("latent.npy"))
            receipt.files["latent.npy"] = try ModelArtifactPin.fileSHA256(temporary.appendingPathComponent("latent.npy"))
        }
        try receipt.validate()
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(receipt).write(to: temporary.appendingPathComponent("stage.json"))
        // Moving a completed directory refuses an existing destination; failed requests leave no partial stage.
        try manager.moveItem(at: temporary, to: directory)
    }

    public static func validateLatents(_ latents: MLXArray, frames: Int? = nil) throws {
        guard latents.ndim == 2, latents.dim(1) == 64, (1...9000).contains(latents.dim(0)),
              frames == nil || latents.dim(0) == frames,
              latents.dtype == .float32 || latents.dtype == .bfloat16,
              all(isFinite(latents)).item(Bool.self) else {
            throw YuE2Error.invalidRequest("Expected finite floating-point latents [1...9000,64] matching saved music frames.")
        }
    }

    public static func loadLatents(from url: URL) throws -> MLXArray {
        var latents = try MLX.loadArray(url: url)
        if latents.ndim == 3, latents.dim(0) == 1, latents.dim(1) == 64 { latents = latents[0].T }
        try validateLatents(latents)
        return latents
    }
}
