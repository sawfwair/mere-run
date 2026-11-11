import Crypto
import Foundation
import MLX
import MLXNN
import MereRunExecution

/// Writes a separate native Q8 checkpoint with exact released-precision timestep tables.
public enum MiniMaxH3ImageOptimizer {
    public static let format = MiniMaxH3ImageMLXFormat.name

    public static func exportQ8(source: URL, destination: URL,
                                progress: ((String, Int, Int) -> Void)? = nil) throws -> [URL] {
        let fm = FileManager.default
        let resources = MiniMaxH3ImageResources(rootURL: source)
        var manifest = try MereRunModelManifest.loadRequired(from: source)
        guard manifest.engine == .miniMaxH3Image, manifest.precision == .bf16,
              try resources.mlxFormat() == nil, resources.validate().isEmpty else {
            throw resources.invalid("Q8 export requires the complete original BF16 H3-Image checkpoint.")
        }
        guard !fm.fileExists(atPath: destination.path) else {
            throw resources.invalid("Output already exists; choose a new Q8 directory.")
        }
        let parent = destination.deletingLastPathComponent()
        try fm.createDirectory(at: parent, withIntermediateDirectories: true)
        let staging = parent.appending(path: ".h3-image-q8-\(UUID().uuidString)")
        try fm.createDirectory(at: staging, withIntermediateDirectories: true)
        var published = false
        defer { if !published { try? fm.removeItem(at: staging) } }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        var identityData = try Data(contentsOf: source.appending(path: MereRunModelManifest.filename))
        identityData.append(try Data(contentsOf: source.appending(path: "transformer/model.safetensors.index.json")))
        let identity = SHA256.hash(data: identityData).map { String(format: "%02x", $0) }.joined()
        let native = MiniMaxH3ImageMLXFormat(format: format, bits: 8, groupSize: 64,
                                           sourceRevision: MiniMaxH3ImageResources.revision, sourceIdentity: identity)
        progress?("modulations", 0, 1)
        try saveModulations(resources, native: native, to: staging)
        progress?("modulations", 1, 1)
        Memory.clearCache()
        var artifacts: [String] = []
        for component in ["transformer", "text_encoder"] {
            let index = try resources.index(component)
            let linearKeys = linearWeightKeys(component: component)
            let directory = staging.appending(path: component)
            try fm.createDirectory(at: directory, withIntermediateDirectories: true)
            var mapping: [String: String] = [:]
            var bytes: Int64 = 0
            for (number, shard) in index.shardFilenames.enumerated() {
                let file = "model-\(number + 1).safetensors"
                let keys = try exportShard(resources, component: component, shard: shard,
                                           linearKeys: linearKeys, to: directory.appending(path: file))
                if !keys.isEmpty {
                    for key in keys { mapping[key] = file }
                    bytes += (try fm.attributesOfItem(atPath: directory.appending(path: file).path)[.size] as? NSNumber)?.int64Value ?? 0
                    artifacts.append("\(component)/\(file)")
                }
                Memory.clearCache()
                progress?(component, number + 1, index.shardFilenames.count)
            }
            let outputIndex = HFSafetensorsIndex(metadata: .init(totalSize: bytes), weightMap: mapping)
            try encoder.encode(outputIndex).write(to: directory.appending(path: "model.safetensors.index.json"))
            try fm.copyItem(at: source.appending(path: "\(component)/config.json"), to: directory.appending(path: "config.json"))
        }
        for path in ["LICENSE", "README.md", "model_index.json", "pdd_config.json", "pdd_fused_heads.safetensors", "vae", "tokenizer", "processor"] {
            try fm.copyItem(at: source.appending(path: path), to: staging.appending(path: path))
        }
        try encoder.encode(native).write(to: staging.appending(path: MiniMaxH3ImageMLXFormat.filename))
        manifest.id = destination.lastPathComponent
        manifest.precision = .int8
        manifest.quantization = .init(bits: 8, groupSize: 64, scheme: format)
        manifest.createdAt = Date()
        try manifest.write(to: staging)
        let outputs = try artifacts.map { name in
            let file = try RunArtifact.read(staging.appending(path: name))
            return FileRecord(path: name, bytes: file.byteCount, sha256: file.sha256)
        }
        try encoder.encode(outputs).write(to: staging.appending(path: "q8-files.json"))
        guard MiniMaxH3ImageResources(rootURL: staging).validate().isEmpty else {
            throw resources.invalid("Exported Q8 checkpoint is incomplete.")
        }
        try fm.moveItem(at: staging, to: destination)
        published = true
        return artifacts.map { destination.appending(path: $0) }
            + [destination.appending(path: MiniMaxH3ImageMLXFormat.cacheFilename), destination.appending(path: MiniMaxH3ImageMLXFormat.filename)]
    }

    private struct FileRecord: Codable { let path: String; let bytes: UInt64; let sha256: String }

    private static func saveModulations(_ resources: MiniMaxH3ImageResources,
                                        native: MiniMaxH3ImageMLXFormat, to root: URL) throws {
        let model = try MiniMaxH3ImageLoader.transformer(resources)
        let cache = model.precomputeModulations()
        try MLX.save(arrays: MiniMaxH3ImageMLXFormat.cacheArrays(cache),
                     metadata: ["format": format, "source_identity": native.sourceIdentity],
                     url: root.appending(path: MiniMaxH3ImageMLXFormat.cacheFilename))
    }

    private static func linearWeightKeys(component: String) -> Set<String> {
        let model: Module = component == "transformer" ? MiniMaxH3ImageTransformer() : MiniMaxH3ModelLoader.makeConditioner()
        return Set(model.leafModules().flattened().compactMap { path, module in
            module is Linear ? "\(path).weight" : nil
        })
    }

    private static func exportShard(_ resources: MiniMaxH3ImageResources, component: String, shard: String,
                                    linearKeys: Set<String>, to url: URL) throws -> [String] {
        let raw = try MLX.loadArrays(url: resources.rootURL.appending(path: "\(component)/\(shard)"))
        let index = try resources.index(component)
        guard Set(raw.keys) == Set(index.weightMap.filter { $0.value == shard }.keys) else {
            throw resources.invalid("Source shard does not match its index: \(shard).")
        }
        let mapped = raw.flatMap { key, value -> [(String, MLXArray)] in
            if component == "transformer" {
                if key == "rope.inv_freq" || key.hasPrefix("final_layer.video_out.") { return [] }
                return MiniMaxH3ModelLoader.releasedBF16TransformerWeight(key: key, value: value, omitCachedAdaLNWeights: true)
            }
            return MiniMaxH3ModelLoader.conditionerWeight(key: MiniMaxH3ModelLoader.conditionerWeightKey(key), value: value)
        }
        let arrays = quantizeArrays(Dictionary(uniqueKeysWithValues: mapped), linearKeys: linearKeys)
        if !arrays.isEmpty { try MLX.save(arrays: arrays, url: url) }
        return Array(arrays.keys)
    }

    static func quantizeArrays(_ arrays: [String: MLXArray], linearKeys: Set<String>) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        for key in arrays.keys.sorted() {
            let value = arrays[key]!
            if linearKeys.contains(key), value.dtype == .bfloat16, value.ndim == 2, value.dim(1).isMultiple(of: 64) {
                let quantized = MLX.quantized(value, groupSize: 64, bits: 8)
                let path = String(key.dropLast(".weight".count))
                MLX.eval(quantized.wq, quantized.scales, quantized.biases!)
                result[key] = quantized.wq
                result[path + ".scales"] = quantized.scales
                result[path + ".biases"] = quantized.biases!
            } else {
                MLX.eval(value)
                result[key] = value
            }
        }
        return result
    }
}
