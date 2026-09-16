import Foundation

public struct YuE2Resources: Sendable, Hashable {
    public static let modelID = "music-yue2"
    public static let repository = "m-a-p/YuE2-3B"
    public static let revision = "29b3558dd46954a0cd9021dc76d5c91864a0f1c7"
    public static let vaeRepository = "m-a-p/YuE2-Vae"
    public static let vaeRevision = "9a94e1d0ea9f8087e98f77fa88df4a4068104d2a"
    public static let sourceRevision = "0edaf2f4053ef4731334b8329834b107977f9637"
    public static let estimatedDownloadBytes: Int64 = 7_795_000_000
    public static let snapshotPatterns = [
        "LICENSE", "THIRD_PARTY_NOTICES.md", "licenses/*.txt", "config.json",
        "model.safetensors", "qwen.tiktoken", "yue2_generation_config.json", "weights_manifest.json",
    ]
    public static let vaePatterns = [
        "LICENSE", "THIRD_PARTY_NOTICES.md", "licenses/*.txt", "config.json",
        "model.safetensors", "weights_manifest.json",
    ]

    public let rootURL: URL
    public var vaeURL: URL { rootURL.appendingPathComponent("vae", isDirectory: true) }

    public init(rootURL: URL) { self.rootURL = rootURL }

    public func validate(fileManager: FileManager = .default) -> [URL] {
        ["config.json", "model.safetensors", "qwen.tiktoken", "LICENSE",
         "vae/config.json", "vae/model.safetensors", "vae/LICENSE"].map {
            rootURL.appendingPathComponent($0)
        }.filter { !fileManager.fileExists(atPath: $0.path) }
    }

    public static func looksLikeRoot(_ url: URL) -> Bool {
        struct Identity: Decodable { let model_type: String }
        guard let data = try? Data(contentsOf: url.appendingPathComponent("config.json")),
              let identity = try? JSONDecoder().decode(Identity.self, from: data) else { return false }
        return identity.model_type == "yue2"
    }

    func configuration() throws -> YuE2Configuration {
        let value = try JSONDecoder().decode(
            YuE2Configuration.self, from: Data(contentsOf: rootURL.appendingPathComponent("config.json"))
        )
        try value.validateReleased()
        return value
    }

    func vaeConfiguration() throws -> YuE2VAEConfiguration {
        let value = try JSONDecoder().decode(
            YuE2VAEConfiguration.self, from: Data(contentsOf: vaeURL.appendingPathComponent("config.json"))
        )
        try value.validateReleased()
        return value
    }
}
