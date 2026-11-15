import AudioSOVTranscribeModel
import Foundation
import MereRunCore
import MLX
import MLXNN

/// Direct loading of distributed Q8 weights; full payload hashes are verified at download time.
enum SOVPortableBundle {
    static func read(root: URL) throws -> SOVTranscribeBundleManifest? {
        let url = root.appendingPathComponent(SOVTranscribeBundleManifest.filename)
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        let manifest = try JSONDecoder().decode(SOVTranscribeBundleManifest.self, from: Data(contentsOf: url))
        try manifest.validate()
        return manifest
    }

    static func decoderPolicy(requested: SOVDecoderExperts, metalAvailable: Bool) throws -> SOVDecoderExperts {
        switch requested {
        case .bf16: return metalAvailable ? .q8Metal : .q8
        case .q8, .q8Metal: return requested
        default: throw SOVTranscribeError.invalid("Portable SOV bundles contain Q8 group-64 experts; select q8 or q8-metal.")
        }
    }

    static func validateFiles(_ manifest: SOVTranscribeBundleManifest, root: URL, layerCount: Int) throws {
        try manifest.validate()
        guard manifest.layerBits.count == layerCount,
              SOVPackedWeights.digest(try Data(contentsOf: root.appendingPathComponent("config.json"))) == manifest.configSHA256 else {
            throw SOVTranscribeError.invalid("Portable SOV configuration does not match its packed weights.")
        }
        for file in [manifest.audio] + manifest.shards {
            let actual = try SOVPackedWeights.FileRecord(root: root, name: file.name)
            guard actual.bytes == file.bytes, actual.headerSHA256 == file.headerSHA256 else {
                throw SOVTranscribeError.invalid("Portable SOV shard changed: \(file.name). Verify or download the bundle again.")
            }
        }
    }

    static func load(_ manifest: SOVTranscribeBundleManifest, root: URL, into model: Module) throws {
        var loaded: Set<String> = []
        for file in manifest.shards {
            try Task.checkCancellation()
            var duplicate: String?
            try SafetensorsStreamingLoader.applyWeightsLazyMaterialized(url: root.appendingPathComponent(file.name),
                to: model, verify: [.noUnusedKeys, .shapeMismatch], mapper: { name, value in
                    if !loaded.insert(name).inserted { duplicate = name }
                    return [(name, value)]
                }, batchSize: 4)
            if let duplicate { throw SOVTranscribeError.invalid("Duplicate portable SOV tensor: \(duplicate).") }
        }
        try SOVPackedWeights.checkComplete(model: model, loaded: loaded)
    }
}
