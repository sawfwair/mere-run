import Foundation
import MLX

public enum LingBotVideoMoEQuantizer {
    public static let defaultOutputModelID = ModelResolver.ModelID.lingBotVideoMoE30BA3B4Bit.rawValue

    public struct Options: Sendable, Hashable {
        public let sourceRoot: URL
        public let outputRoot: URL
        public let bits: Int
        public let groupSize: Int
        public let includeRefiner: Bool
        public let force: Bool

        public init(
            sourceRoot: URL,
            outputRoot: URL,
            bits: Int = 4,
            groupSize: Int = 64,
            includeRefiner: Bool = true,
            force: Bool = false
        ) {
            self.sourceRoot = sourceRoot
            self.outputRoot = outputRoot
            self.bits = bits
            self.groupSize = groupSize
            self.includeRefiner = includeRefiner
            self.force = force
        }
    }

    public struct Progress: Sendable, Hashable {
        public enum State: String, Sendable, Hashable {
            case quantizing
            case reused
        }

        public let component: String
        public let shard: String
        public let completedShards: Int
        public let totalShards: Int
        public let state: State
    }

    public struct Result: Sendable, Hashable {
        public let outputRoot: URL
        public let outputBytes: Int64
        public let quantizedShardCount: Int
        public let reusedShardCount: Int
        public let includesRefiner: Bool
    }

    public enum QuantizerError: LocalizedError, Sendable {
        case sourceAndOutputMatch(URL)
        case sourceAndOutputOverlap(URL, URL)
        case incompatibleOutput(URL)
        case unsupportedBits(Int)
        case invalidGroupSize(Int)
        case missingSource(URL)
        case missingComponent(String, URL)
        case missingIndex(String, URL)
        case notMoE(Int)
        case missingTensor(String, URL)
        case invalidExpertShape(String, [Int], Int)
        case missingAffineBias(String)

        public var errorDescription: String? {
            switch self {
            case .sourceAndOutputMatch(let url):
                return "LingBot MoE quantization requires a separate output directory: \(url.path)"
            case .sourceAndOutputOverlap(let source, let output):
                return "LingBot MoE source and output directories cannot contain one another: \(source.path), \(output.path)"
            case .incompatibleOutput(let url):
                return "Existing LingBot MoE output uses different quantization settings: \(url.path). Pass --force to replace it."
            case .unsupportedBits(let bits):
                return "LingBot MoE conversion currently supports 4-bit output, not \(bits)-bit."
            case .invalidGroupSize(let groupSize):
                return "LingBot MoE quantization group size must be 32, 64, or 128: \(groupSize)"
            case .missingSource(let url):
                return "LingBot MoE source directory not found: \(url.path)"
            case .missingComponent(let component, let url):
                return "LingBot MoE source is missing \(component)/: \(url.path)"
            case .missingIndex(let component, let url):
                return "LingBot MoE \(component) is missing its sharded safetensors index: \(url.path)"
            case .notMoE(let experts):
                return "LingBot MoE conversion requires a transformer with experts; config declares \(experts)."
            case .missingTensor(let key, let url):
                return "Safetensors index references missing tensor '\(key)' in \(url.path)"
            case .invalidExpertShape(let key, let shape, let groupSize):
                return "Expert tensor '\(key)' has shape \(shape); its input dimension must be divisible by group size \(groupSize)."
            case .missingAffineBias(let key):
                return "MLX did not produce affine quantization biases for expert tensor '\(key)'."
            }
        }
    }

    public static func quantize(
        options: Options,
        fileManager: FileManager = .default,
        progress: @Sendable (Progress) -> Void = { _ in }
    ) throws -> Result {
        let sourceRoot = options.sourceRoot.standardizedFileURL
        let outputRoot = options.outputRoot.standardizedFileURL
        guard sourceRoot != outputRoot else {
            throw QuantizerError.sourceAndOutputMatch(sourceRoot)
        }
        let sourcePrefix = sourceRoot.path.hasSuffix("/") ? sourceRoot.path : sourceRoot.path + "/"
        let outputPrefix = outputRoot.path.hasSuffix("/") ? outputRoot.path : outputRoot.path + "/"
        guard !outputRoot.path.hasPrefix(sourcePrefix),
              !sourceRoot.path.hasPrefix(outputPrefix) else {
            throw QuantizerError.sourceAndOutputOverlap(sourceRoot, outputRoot)
        }
        guard options.bits == 4 else {
            throw QuantizerError.unsupportedBits(options.bits)
        }
        guard [32, 64, 128].contains(options.groupSize) else {
            throw QuantizerError.invalidGroupSize(options.groupSize)
        }

        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: sourceRoot.path, isDirectory: &isDirectory),
              isDirectory.boolValue else {
            throw QuantizerError.missingSource(sourceRoot)
        }

        let transformerConfigURL = sourceRoot
            .appendingPathComponent("transformer", isDirectory: true)
            .appendingPathComponent("config.json")
        let probe = try JSONDecoder().decode(
            MoEConfigProbe.self,
            from: Data(contentsOf: transformerConfigURL)
        )
        guard probe.numExperts > 0 else {
            throw QuantizerError.notMoE(probe.numExperts)
        }

        let refinerURL = sourceRoot.appendingPathComponent("refiner", isDirectory: true)
        let hasRefiner = fileManager.fileExists(atPath: refinerURL.path)
        let includesRefiner = options.includeRefiner && hasRefiner
        if options.includeRefiner, !hasRefiner {
            throw QuantizerError.missingComponent("refiner", refinerURL)
        }
        let baseManifest = try MereRunModelManifest.loadRequired(
            from: sourceRoot,
            fileManager: fileManager
        )
        let quantization = LingBotVideoQuantizationConfig(
            bits: options.bits,
            groupSize: options.groupSize,
            includesRefiner: includesRefiner,
            sourceModelID: baseManifest.id
        )

        if options.force, fileManager.fileExists(atPath: outputRoot.path) {
            try fileManager.removeItem(at: outputRoot)
        }
        try fileManager.createDirectory(at: outputRoot, withIntermediateDirectories: true)
        let quantizationURL = outputRoot.appendingPathComponent(LingBotVideoQuantizationConfig.filename)
        if fileManager.fileExists(atPath: quantizationURL.path) {
            let existing = try JSONDecoder().decode(
                LingBotVideoQuantizationConfig.self,
                from: Data(contentsOf: quantizationURL)
            )
            guard existing == quantization else {
                throw QuantizerError.incompatibleOutput(outputRoot)
            }
        }
        try writeJSON(quantization, to: quantizationURL)
        try linkCommonFiles(
            sourceRoot: sourceRoot,
            outputRoot: outputRoot,
            fileManager: fileManager
        )

        var quantizedShardCount = 0
        var reusedShardCount = 0
        let transformerResult = try quantizeComponent(
            named: "transformer",
            sourceRoot: sourceRoot,
            outputRoot: outputRoot,
            bits: options.bits,
            groupSize: options.groupSize,
            fileManager: fileManager,
            progress: progress
        )
        quantizedShardCount += transformerResult.quantized
        reusedShardCount += transformerResult.reused

        if includesRefiner {
            let refinerResult = try quantizeComponent(
                named: "refiner",
                sourceRoot: sourceRoot,
                outputRoot: outputRoot,
                bits: options.bits,
                groupSize: options.groupSize,
                fileManager: fileManager,
                progress: progress
            )
            quantizedShardCount += refinerResult.quantized
            reusedShardCount += refinerResult.reused
        }

        _ = try QuantizedModelManifestWriter.writeQuantizedModelManifest(
            engine: .lingBotVideo,
            inputModelRoot: sourceRoot,
            outputModelRoot: outputRoot,
            bits: options.bits,
            groupSize: options.groupSize,
            scheme: "mlx-affine-routed-experts",
            fileManager: fileManager
        )

        return Result(
            outputRoot: outputRoot,
            outputBytes: try allocatedSize(of: outputRoot, fileManager: fileManager),
            quantizedShardCount: quantizedShardCount,
            reusedShardCount: reusedShardCount,
            includesRefiner: includesRefiner
        )
    }

    static func isExpertTensor(_ key: String) -> Bool {
        key.hasSuffix(".ffn.experts.w1")
            || key.hasSuffix(".ffn.experts.w2")
            || key.hasSuffix(".ffn.experts.w3")
    }

    private static func quantizeComponent(
        named component: String,
        sourceRoot: URL,
        outputRoot: URL,
        bits: Int,
        groupSize: Int,
        fileManager: FileManager,
        progress: @Sendable (Progress) -> Void
    ) throws -> (quantized: Int, reused: Int) {
        let source = sourceRoot
            .appendingPathComponent(component, isDirectory: true)
            .resolvingSymlinksInPath()
        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: source.path, isDirectory: &isDirectory),
              isDirectory.boolValue else {
            throw QuantizerError.missingComponent(component, source)
        }

        let indexName = "diffusion_pytorch_model.safetensors.index.json"
        let sourceIndexURL = source.appendingPathComponent(indexName)
        guard fileManager.fileExists(atPath: sourceIndexURL.path) else {
            throw QuantizerError.missingIndex(component, sourceIndexURL)
        }
        let sourceIndex = try JSONDecoder().decode(
            HFSafetensorsIndex.self,
            from: Data(contentsOf: sourceIndexURL)
        )
        let output = outputRoot.appendingPathComponent(component, isDirectory: true)
        try fileManager.createDirectory(at: output, withIntermediateDirectories: true)
        try linkComponentMetadata(
            source: source,
            output: output,
            fileManager: fileManager
        )

        let outputWeightMap = quantizedWeightMap(sourceIndex.weightMap)
        let sourceKeysByShard = Dictionary(grouping: sourceIndex.weightMap.keys) {
            sourceIndex.weightMap[$0]!
        }
        let outputKeysByShard = Dictionary(grouping: outputWeightMap.keys) {
            outputWeightMap[$0]!
        }

        var quantized = 0
        var reused = 0
        let shards = sourceIndex.shardFilenames
        for (offset, shard) in shards.enumerated() {
            let completed = offset + 1
            let outputShard = output.appendingPathComponent(shard)
            let expectedOutputKeys = Set(outputKeysByShard[shard] ?? [])
            if isCompleteShard(outputShard, expectedKeys: expectedOutputKeys, fileManager: fileManager) {
                reused += 1
                progress(Progress(
                    component: component,
                    shard: shard,
                    completedShards: completed,
                    totalShards: shards.count,
                    state: .reused
                ))
                continue
            }
            if fileManager.fileExists(atPath: outputShard.path) {
                try fileManager.removeItem(at: outputShard)
            }

            progress(Progress(
                component: component,
                shard: shard,
                completedShards: completed,
                totalShards: shards.count,
                state: .quantizing
            ))
            let sourceShard = source.appendingPathComponent(shard)
            let arrays = try SafetensorsStreamingLoader.loadArrays(url: sourceShard)
            var outputArrays: [String: MLXArray] = [:]
            let sourceKeys = sourceKeysByShard[shard] ?? []
            outputArrays.reserveCapacity(sourceKeys.count * 3)
            for key in sourceKeys {
                guard let array = arrays[key] else {
                    throw QuantizerError.missingTensor(key, sourceShard)
                }
                if isExpertTensor(key) {
                    guard let inputDimension = array.shape.last,
                          inputDimension % groupSize == 0 else {
                        throw QuantizerError.invalidExpertShape(key, array.shape, groupSize)
                    }
                    let (weight, scales, biases) = MLX.quantized(
                        array,
                        groupSize: groupSize,
                        bits: bits
                    )
                    guard let biases else {
                        throw QuantizerError.missingAffineBias(key)
                    }
                    MLX.eval(weight, scales, biases)
                    outputArrays[key + ".weight"] = weight
                    outputArrays[key + ".scales"] = scales
                    outputArrays[key + ".biases"] = biases
                } else {
                    outputArrays[key] = array
                }
            }

            let temporary = output.appendingPathComponent(
                ".\(shard).partial-\(UUID().uuidString).safetensors"
            )
            defer { try? fileManager.removeItem(at: temporary) }
            try MLX.save(arrays: outputArrays, url: temporary)
            try fileManager.moveItem(at: temporary, to: outputShard)
            Memory.clearCache()
            quantized += 1
        }

        let totalSize = try shards.reduce(Int64(0)) { partial, shard in
            partial + (try fileSize(
                output.appendingPathComponent(shard),
                fileManager: fileManager
            ))
        }
        let outputIndex = OutputIndex(
            metadata: .init(totalSize: totalSize),
            weightMap: outputWeightMap
        )
        try writeJSON(outputIndex, to: output.appendingPathComponent(indexName))
        return (quantized, reused)
    }

    private static func quantizedWeightMap(_ source: [String: String]) -> [String: String] {
        var output: [String: String] = [:]
        output.reserveCapacity(source.count)
        for (key, shard) in source {
            if isExpertTensor(key) {
                output[key + ".weight"] = shard
                output[key + ".scales"] = shard
                output[key + ".biases"] = shard
            } else {
                output[key] = shard
            }
        }
        return output
    }

    private static func isCompleteShard(
        _ url: URL,
        expectedKeys: Set<String>,
        fileManager: FileManager
    ) -> Bool {
        guard fileManager.fileExists(atPath: url.path),
              let size = try? fileSize(url, fileManager: fileManager),
              size > 0,
              let metadata = try? SafetensorsStreamingLoader.metadata(url: url)
        else {
            return false
        }
        return Set(metadata.keys) == expectedKeys
    }

    private static func linkCommonFiles(
        sourceRoot: URL,
        outputRoot: URL,
        fileManager: FileManager
    ) throws {
        let skipped = Set([
            ".cache",
            "transformer",
            "refiner",
            MereRunModelManifest.filename,
            LingBotVideoQuantizationConfig.filename,
        ])
        for item in try fileManager.contentsOfDirectory(
            at: sourceRoot,
            includingPropertiesForKeys: nil
        ) where !skipped.contains(item.lastPathComponent) {
            try linkTree(
                source: item,
                destination: outputRoot.appendingPathComponent(item.lastPathComponent),
                fileManager: fileManager
            )
        }
    }

    private static func linkComponentMetadata(
        source: URL,
        output: URL,
        fileManager: FileManager
    ) throws {
        for item in try fileManager.contentsOfDirectory(
            at: source,
            includingPropertiesForKeys: nil
        ) {
            let name = item.lastPathComponent
            guard !name.hasSuffix(".safetensors"),
                  !name.hasSuffix(".safetensors.index.json") else {
                continue
            }
            try linkTree(
                source: item,
                destination: output.appendingPathComponent(name),
                fileManager: fileManager
            )
        }
    }

    private static func linkTree(
        source: URL,
        destination: URL,
        fileManager: FileManager
    ) throws {
        if fileManager.fileExists(atPath: destination.path) {
            return
        }
        let resolvedSource = source.resolvingSymlinksInPath()
        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: resolvedSource.path, isDirectory: &isDirectory) else {
            throw QuantizerError.missingSource(resolvedSource)
        }
        if isDirectory.boolValue {
            try fileManager.createDirectory(at: destination, withIntermediateDirectories: true)
            for child in try fileManager.contentsOfDirectory(
                at: resolvedSource,
                includingPropertiesForKeys: nil
            ) {
                try linkTree(
                    source: child,
                    destination: destination.appendingPathComponent(child.lastPathComponent),
                    fileManager: fileManager
                )
            }
            return
        }

        try fileManager.createDirectory(
            at: destination.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        do {
            try fileManager.linkItem(at: resolvedSource, to: destination)
        } catch {
            try fileManager.copyItem(at: resolvedSource, to: destination)
        }
    }

    private static func fileSize(_ url: URL, fileManager: FileManager) throws -> Int64 {
        let attributes = try fileManager.attributesOfItem(atPath: url.path)
        return (attributes[.size] as? NSNumber)?.int64Value ?? 0
    }

    private static func allocatedSize(of root: URL, fileManager: FileManager) throws -> Int64 {
        guard let enumerator = fileManager.enumerator(
            at: root,
            includingPropertiesForKeys: [.isRegularFileKey, .totalFileAllocatedSizeKey],
            options: [.skipsHiddenFiles]
        ) else {
            return 0
        }
        var total: Int64 = 0
        for case let item as URL in enumerator {
            let values = try item.resourceValues(forKeys: [.isRegularFileKey, .totalFileAllocatedSizeKey])
            if values.isRegularFile == true {
                total += Int64(values.totalFileAllocatedSize ?? 0)
            }
        }
        return total
    }

    private static func writeJSON<Value: Encodable>(_ value: Value, to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(value).write(to: url, options: [.atomic])
    }

    private struct MoEConfigProbe: Decodable {
        let numExperts: Int

        private enum CodingKeys: String, CodingKey {
            case numExperts = "num_experts"
        }
    }

    private struct OutputIndex: Encodable {
        struct Metadata: Encodable {
            let totalSize: Int64

            private enum CodingKeys: String, CodingKey {
                case totalSize = "total_size"
            }
        }

        let metadata: Metadata
        let weightMap: [String: String]

        private enum CodingKeys: String, CodingKey {
            case metadata
            case weightMap = "weight_map"
        }
    }
}
