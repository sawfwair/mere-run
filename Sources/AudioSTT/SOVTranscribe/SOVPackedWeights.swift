import AudioSOVTranscribeModel
#if canImport(CryptoKit)
import CryptoKit
#else
import Crypto
#endif
import Foundation
import MereRunCore
import MLX
import MLXNN

/// Reusable native target weights. The encoder, tokenizer and draft remain in their source directories.
enum SOVPackedWeights {
    static let manifestName = "sov-packed.json"
    static let formatVersion = 1
    static let packerVersion = "affine-fused-gate-up-v1"

    struct Manifest: Codable {
        let formatVersion: Int
        let packerVersion: String
        let source: Source
        let groupSize: Int
        let layerBits: [Int]
        let shards: [FileRecord]
    }

    struct Source: Codable, Equatable {
        let configSHA256: String
        let indexSHA256: String?
        let files: [FileRecord]
    }

    /// Header hashes plus size/mtime detect stale source and incomplete packed files without
    /// rereading tens of GB on every launch. This is cache validation, not a content-integrity scan.
    struct FileRecord: Codable, Equatable {
        let name: String
        let bytes: Int
        let modified: Date
        let headerSHA256: String

        init(root: URL, name: String) throws {
            guard URL(fileURLWithPath: name).lastPathComponent == name else {
                throw SOVTranscribeError.invalid("Invalid packed checkpoint shard name.")
            }
            let url = root.appendingPathComponent(name)
            let values = try url.resourceValues(forKeys: [.fileSizeKey, .contentModificationDateKey])
            guard let bytes = values.fileSize, let modified = values.contentModificationDate else {
                throw SOVTranscribeError.invalid("Missing SOV checkpoint file metadata: \(name).")
            }
            self.name = name
            self.bytes = bytes
            self.modified = modified
            let metadata = try SafetensorsStreamingLoader.metadata(url: url)
            let description = metadata.sorted { $0.key < $1.key }.map {
                "\($0.key):\($0.value.shape):\($0.value.dtype):\($0.value.startOffset):\($0.value.endOffset)"
            }.joined(separator: "\n")
            headerSHA256 = SOVPackedWeights.digest(Data(description.utf8))
        }
    }

    static func digest(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    static func source(root: URL) throws -> Source {
        let index = root.appendingPathComponent("model.safetensors.index.json")
        let indexData = FileManager.default.fileExists(atPath: index.path) ? try Data(contentsOf: index) : nil
        let names = try indexData.map { try JSONDecoder().decode(HFSafetensorsIndex.self, from: $0).shardFilenames }
            ?? ["model.safetensors"]
        return Source(configSHA256: digest(try Data(contentsOf: root.appendingPathComponent("config.json"))),
            indexSHA256: indexData.map(digest), files: try names.map { try FileRecord(root: root, name: $0) })
    }

    static func mapped(_ key: String, _ value: MLXArray, groupSize: Int, layerBits: [Int]) throws -> [(String, MLXArray)] {
        guard key.hasPrefix("model.language_model.") else { return [] }
        let projection: String
        if key.hasSuffix(".experts.gate_up_proj") { projection = "gate_up_proj" }
        else if key.hasSuffix(".experts.down_proj") { projection = "down_proj" }
        else { return SOVTranscribeWeights.mappedTarget(key, value) }
        // The source layout is validated against the model by the ordinary parameter update.
        let components = key.split(separator: ".")
        guard components.count == 6, let layer = Int(components[3]), layerBits.indices.contains(layer),
              value.ndim == 3, value.dim(-1).isMultiple(of: groupSize), value.dtype == .bfloat16 else {
            throw SOVTranscribeError.invalid("Unsupported SOV source expert tensor: \(key).")
        }
        let prefix = String(key.dropFirst("model.".count).dropLast(projection.count)) + "switch_glu." + projection + "."
        let packed = quantized(value, groupSize: groupSize, bits: layerBits[layer])
        return [(prefix + "weight", packed.wq), (prefix + "scales", packed.scales), (prefix + "biases", packed.biases!)]
    }

    /// Conversion is transactional: only a complete pack is renamed into the requested directory.
    static func prepare(root: URL, destination: URL, model: Gemma4TextCausalLM, groupSize: Int,
                        layerBits: [Int]) throws -> SOVPackedWeightsMeasurement {
        let started = ProcessInfo.processInfo.systemUptime
        let identity = try source(root: root)
        let manifestURL = destination.appendingPathComponent(manifestName)
        if FileManager.default.fileExists(atPath: destination.path) {
            let manifest = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: manifestURL))
            guard manifest.formatVersion == formatVersion, manifest.packerVersion == packerVersion,
                  manifest.source == identity, manifest.groupSize == groupSize, manifest.layerBits == layerBits else {
                throw SOVTranscribeError.invalid("Packed SOV weights do not match the source, precision, or packer version. Use a new packed-weights directory.")
            }
            try load(manifest: manifest, root: destination, into: model)
            return try measurement(manifest, hit: true, started: started)
        }
        let parent = destination.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
        let staging = parent.appendingPathComponent(".sov-pack-\(UUID().uuidString).partial")
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: staging) }
        var records: [FileRecord] = []
        var loaded: Set<String> = []
        for sourceFile in identity.files {
            try Task.checkCancellation()
            var shard: [String: MLXArray] = [:]
            // Materialize a single CPU load node before building any GPU quantization work.
            // Foundation FileHandle reads retain autoreleased Data across a long actor call;
            // native MLX reads avoid that second, checkpoint-sized CPU allocation entirely.
            let sourceURL = root.appendingPathComponent(sourceFile.name)
            let metadata = try SafetensorsStreamingLoader.metadata(url: sourceURL)
            var arrays = try MLX.loadArrays(url: sourceURL, stream: .cpu)
            for key in metadata.sorted(by: { $0.value.startOffset < $1.value.startOffset }).map(\.key)
                where key.hasPrefix("model.language_model.") {
                try Task.checkCancellation()
                guard let value = arrays.removeValue(forKey: key) else {
                    throw SOVTranscribeError.invalid("Missing SOV source tensor: \(key).")
                }
                value.eval()
                let mapped = try mapped(key, value, groupSize: groupSize, layerBits: layerBits)
                eval(mapped.map(\.1))
                try model.update(parameters: ModuleParameters.unflattened(mapped), verify: [.noUnusedKeys, .shapeMismatch])
                for (name, tensor) in mapped {
                    guard loaded.insert(name).inserted else { throw SOVTranscribeError.invalid("Duplicate SOV checkpoint tensor: \(name).") }
                    shard[name] = tensor
                }
                Memory.clearCache()
            }
            if !shard.isEmpty {
                let name = String(format: "target-%05d.safetensors", records.count + 1)
                try MLX.save(arrays: shard, url: staging.appendingPathComponent(name))
                records.append(try FileRecord(root: staging, name: name))
            }
        }
        try checkComplete(model: model, loaded: loaded)
        // Do not publish a conversion if a source file changed while it was being read.
        guard try source(root: root) == identity else { throw SOVTranscribeError.invalid("SOV source changed during packing.") }
        let manifest = Manifest(formatVersion: formatVersion, packerVersion: packerVersion, source: identity,
            groupSize: groupSize, layerBits: layerBits, shards: records)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .prettyPrinted]
        try encoder.encode(manifest).write(to: staging.appendingPathComponent(manifestName), options: .atomic)
        try Task.checkCancellation()
        try FileManager.default.moveItem(at: staging, to: destination)
        return try measurement(manifest, hit: false, started: started)
    }

    static func load(manifest: Manifest, root: URL, into model: Module) throws {
        guard !manifest.shards.isEmpty, Set(manifest.shards.map(\.name)).count == manifest.shards.count else {
            throw SOVTranscribeError.invalid("Packed SOV checkpoint has no shards or duplicate shards.")
        }
        var loaded: Set<String> = []
        for record in manifest.shards {
            try Task.checkCancellation()
            guard try FileRecord(root: root, name: record.name) == record else {
                throw SOVTranscribeError.invalid("Packed SOV shard changed: \(record.name). Use a new packed-weights directory.")
            }
            try SafetensorsStreamingLoader.applyWeightsLazyMaterialized(url: root.appendingPathComponent(record.name),
                to: model, verify: [.noUnusedKeys, .shapeMismatch], mapper: { name, value in
                    loaded.insert(name)
                    return [(name, value)]
                }, batchSize: 4)
        }
        try checkComplete(model: model, loaded: loaded)
    }

    private static func checkComplete(model: Module, loaded: Set<String>) throws {
        let expected = Set(model.parameters().flattened().map(\.0).filter { !$0.contains("per_layer") && !$0.contains(".rope.") })
        let missing = expected.subtracting(loaded).sorted()
        guard missing.isEmpty else { throw SOVTranscribeError.invalid("Packed SOV checkpoint is missing tensors: \(missing.joined(separator: ", ")).") }
    }

    private static func measurement(_ manifest: Manifest, hit: Bool, started: TimeInterval) throws -> SOVPackedWeightsMeasurement {
        SOVPackedWeightsMeasurement(cacheHit: hit, seconds: ProcessInfo.processInfo.systemUptime - started,
            sourceBytes: manifest.source.files.reduce(0) { $0 + $1.bytes }, packedBytes: manifest.shards.reduce(0) { $0 + $1.bytes },
            sourceFingerprint: digest(try JSONEncoder.sorted.encode(manifest.source)))
    }
}

public struct SOVPackedWeightsMeasurement: Codable, Sendable {
    public let cacheHit: Bool
    public let seconds: Double
    public let sourceBytes: Int
    public let packedBytes: Int
    public let sourceFingerprint: String
}

private extension JSONEncoder {
    static var sorted: JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return encoder
    }
}
