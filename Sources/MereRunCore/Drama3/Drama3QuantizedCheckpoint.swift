#if !os(iOS)
import AudioDrama3Model
import Crypto
import Foundation
import MLX
import MereRunExecution
import MereRunTensor

/// Export only routed experts to affine Q4/group-64; retain all other model
/// tensors and the original codec. A staging directory is published atomically.
public enum Drama3QuantizedCheckpoint {
    public static let format = "drama3-experts-q4-g64-v1"
    public struct Receipt: Codable, Sendable {
        public let format: String
        public let upstreamRepository: String
        public let upstreamRevision: String
        public let sourceIndexSHA256: String
        public let sourceFiles: [File]
        public let expertBits: Int
        public let groupSize: Int
        public let nonExpertPolicy: String
        public let files: [File]
        public struct File: Codable, Sendable {
            public let name: String
            public let sha256: String
            public let bytes: UInt64
        }
    }

    public static func export(source: URL, destination: URL,
                              progress: ((Int, Int) -> Void)? = nil) throws -> [URL] {
        let fm = FileManager.default
        let root = source.standardizedFileURL.resolvingSymlinksInPath()
        let output = destination.standardizedFileURL
        guard !fm.fileExists(atPath: output.path), output != root else {
            throw Drama3Error.invalid("standalone output already exists; choose a new directory")
        }
        let config = try Drama3Resources.configuration(root: root)
        guard config.quantization == nil else { throw Drama3Error.invalid("checkpoint is already Q4") }
        guard Drama3Resources.missingFiles(root: root, fileManager: fm).isEmpty else {
            throw Drama3Error.invalid("source checkpoint is incomplete")
        }
        let parent = output.deletingLastPathComponent()
        try fm.createDirectory(at: parent, withIntermediateDirectories: true)
        let weights = try Drama3Weights(root: root)
        try Drama3Model(config: config).validateLayout(weights)
        var estimatedBytes: Int64 = 0
        for key in weights.index.weightMap.keys where !key.hasSuffix("_scale_inv") {
            let metadata = try weights.metadata(key)
            if key.contains(".mlp.experts.") {
                let count = Int64(metadata.shape.reduce(1, *))
                estimatedBytes += count / 2 + count / 64 * 4
            } else { estimatedBytes += Int64(metadata.offsets[1] - metadata.offsets[0]) }
        }
        let codecBytes = (try fm.attributesOfItem(atPath: root.appendingPathComponent("codec.pth").path)[.size] as? NSNumber)?.int64Value ?? 0
        estimatedBytes += codecBytes + 512 * 1_024 * 1_024
        let free = (try fm.attributesOfFileSystem(forPath: parent.path)[.systemFreeSize] as? NSNumber)?.int64Value ?? 0
        guard free >= estimatedBytes else {
            throw Drama3Error.invalid("Q4 export needs approximately \(estimatedBytes) free bytes; available: \(free). Source files are preserved.")
        }
        let staging = parent.appendingPathComponent(".drama3-export-" + UUID().uuidString)
        try fm.createDirectory(at: staging, withIntermediateDirectories: false)
        var published = false
        defer { if !published { try? fm.removeItem(at: staging) } }
        var arrays: [String: MLXArray] = [:], mapping: [String: String] = [:]
        var artifacts: [Receipt.File] = [], batchBytes = 0, totalBytes: Int64 = 0, shard = 0
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        func record(_ file: String) throws {
            let artifact = try RunArtifact.read(staging.appendingPathComponent(file))
            artifacts.append(.init(name: file, sha256: artifact.sha256, bytes: artifact.byteCount))
        }
        func flush() throws {
            guard !arrays.isEmpty else { return }
            shard += 1
            let filename = String(format: "drama3-q4-%04d.safetensors", shard)
            try MLX.save(arrays: arrays, url: staging.appendingPathComponent(filename))
            for (key, value) in arrays {
                mapping[key] = filename
                totalBytes += Int64(value.nbytes)
            }
            try record(filename)
            arrays.removeAll(); batchBytes = 0
            Memory.clearCache()
        }
        let keys = weights.index.weightMap.keys.sorted().filter { !$0.hasSuffix("_scale_inv") }
        for (i, key) in keys.enumerated() {
            try Task.checkCancellation()
            if key.contains(".mlp.experts.") {
                guard key.hasSuffix(".weight") else { throw Drama3Error.invalid("unexpected expert tensor: \(key)") }
                let value = try weights.decodedWeight(key)
                let packed = MLX.quantized(value, groupSize: 64, bits: 4)
                guard let bias = packed.biases else { throw Drama3Error.invalid("affine quantizer omitted biases") }
                MLX.eval(packed.wq, packed.scales, bias)
                let base = String(key.dropLast(".weight".count))
                arrays[key] = packed.wq; arrays[base + ".scales"] = packed.scales; arrays[base + ".biases"] = bias
                batchBytes += packed.wq.nbytes + packed.scales.nbytes + bias.nbytes
            } else {
                guard Drama3Model.mappedKey(key) != nil else { throw Drama3Error.invalid("unknown parameter namespace: \(key)") }
                let value = try weights.read(key)
                MLX.eval(value)
                arrays[key] = value; batchBytes += value.nbytes
            }
            if batchBytes >= 256 * 1_024 * 1_024 { try flush() }
            progress?(i + 1, keys.count)
        }
        try flush()
        let index = HFSafetensorsIndex(metadata: .init(totalSize: totalBytes), weightMap: mapping)
        try encoder.encode(index).write(to: staging.appendingPathComponent("model.safetensors.index.json"))
        try record("model.safetensors.index.json")
        var json = try JSONDecoder().decode([String: OpenAIJSONValue].self, from: Data(contentsOf: root.appendingPathComponent("config.json")))
        json.removeValue(forKey: "quantization_config")
        json["mere_run_drama3_quantization"] = .object(["bits": .number(4), "group_size": .number(64), "mode": .string("affine")])
        try encoder.encode(json).write(to: staging.appendingPathComponent("config.json"))
        try record("config.json")
        for file in ["codec.pth", "LICENSE", "README.md", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja"] {
            let from = root.appendingPathComponent(file)
            if fm.fileExists(atPath: from.path) {
                try fm.copyItem(at: from, to: staging.appendingPathComponent(file))
                try record(file)
            }
        }
        let notice = """
        Built with Fish Audio.
        These Fish Audio Materials are licensed under the Fish Audio Community License, Version 1.0. Copyright © 39 AI, INC. All Rights Reserved.
        Modified by mere.run: FP8 routed experts decoded using upstream block scales and packed as affine Q4/group-64. Other weights and codec preserved.
        Source: \(Drama3Resources.repository)@\(Drama3Resources.revision)
        """
        try Data(notice.utf8).write(to: staging.appendingPathComponent("Notice"))
        try record("Notice")
        let sourceIndex = try RunArtifact.read(root.appendingPathComponent("model.safetensors.index.json"))
        let sourceFiles = try (weights.index.shardFilenames + ["config.json", "model.safetensors.index.json", "codec.pth"]).map { name in
            let file = try RunArtifact.read(root.appendingPathComponent(name))
            return Receipt.File(name: name, sha256: file.sha256, bytes: file.byteCount)
        }
        let receipt = Receipt(format: format, upstreamRepository: Drama3Resources.repository,
            upstreamRevision: Drama3Resources.revision, sourceIndexSHA256: sourceIndex.sha256, sourceFiles: sourceFiles,
            expertBits: 4, groupSize: 64, nonExpertPolicy: "preserve checkpoint dtype; original DAC codec",
            files: artifacts)
        try encoder.encode(receipt).write(to: staging.appendingPathComponent("drama3-quantization.json"))
        try fm.moveItem(at: staging, to: output)
        published = true
        return artifacts.map { output.appendingPathComponent($0.name) } + [output.appendingPathComponent("drama3-quantization.json")]
    }
}

#endif
