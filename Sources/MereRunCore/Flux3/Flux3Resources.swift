import Foundation
import MLX

public struct Flux3Resources: Sendable {
    public static let modelID = "image-flux3-klein"
    public static let repoID = "black-forest-labs/flux-3-image-klein"
    public static let revision = "8454017f9ebaada30c998fccd9f67a80c7580f68"
    public static let referenceRevision = "45e57c9fb06f9751cdced29a65c5d2171003be05"
    public static let patterns = ["LICENSE.md", "NOTICE.md", "README.md", "config.json", "manifest.json",
                                  "model.safetensors", "ae.safetensors", "text_encoder/*"]
    public let rootURL: URL
    public init(rootURL: URL) { self.rootURL = rootURL }

    public func validate(fileManager: FileManager = .default) -> [URL] {
        var paths = ["model.safetensors", "ae.safetensors", "text_encoder/config.json",
                     "text_encoder/tokenizer.json", "text_encoder/tokenizer_config.json"]
        let single = "text_encoder/model.safetensors", indexPath = single + ".index.json"
        if fileManager.fileExists(atPath: rootURL.appending(path: single).path) {
            paths.append(single)
        } else {
            paths.append(indexPath)
            if let data = try? Data(contentsOf: rootURL.appending(path: indexPath)),
               let index = try? JSONDecoder().decode(HFSafetensorsIndex.self, from: data),
               !index.weightMap.isEmpty,
               index.shardFilenames.allSatisfy({ !$0.isEmpty && URL(fileURLWithPath: $0).lastPathComponent == $0 }) {
                paths += index.shardFilenames.map { "text_encoder/" + $0 }
            } else if fileManager.fileExists(atPath: rootURL.appending(path: indexPath).path) {
                return [rootURL.appending(path: indexPath)]
            }
        }
        return paths.map { rootURL.appending(path: $0) }.filter { !fileManager.fileExists(atPath: $0.path) }
    }

    func textArrays() throws -> [String: MLXArray] {
        let root = rootURL.appending(path: "text_encoder")
        let single = root.appending(path: "model.safetensors")
        if FileManager.default.fileExists(atPath: single.path) { return try MLX.loadArrays(url: single) }
        let index = try JSONDecoder().decode(HFSafetensorsIndex.self,
            from: Data(contentsOf: root.appending(path: "model.safetensors.index.json")))
        guard !index.weightMap.isEmpty else { throw Flux3Error.invalidWeights("Empty text encoder index.") }
        var result: [String: MLXArray] = [:]
        for shard in index.shardFilenames {
            guard URL(fileURLWithPath: shard).lastPathComponent == shard else {
                throw Flux3Error.invalidWeights("Text encoder shards must be basenames.")
            }
            for (key, value) in try MLX.loadArrays(url: root.appending(path: shard)) {
                guard result[key] == nil, index.weightMap[key] == shard else {
                    throw Flux3Error.invalidWeights("Text encoder tensor differs from its shard index: \(key).")
                }
                result[key] = value
            }
        }
        guard Set(result.keys) == Set(index.weightMap.keys) else {
            throw Flux3Error.invalidWeights("Incomplete text encoder shards.")
        }
        return result
    }
}
