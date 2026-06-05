import Foundation
import MLX
import MLXNN

public enum SDNQWeightsLoader {
    public enum LoaderError: LocalizedError {
        case notSDNQWeights
        case missingWeight(String)
        case unsupportedModule(String)

        public var errorDescription: String? {
            switch self {
            case .notSDNQWeights:
                return "Expected SDNQ weights but none found (missing .scale/.zero_point keys)."
            case .missingWeight(let key):
                return "Missing SDNQ quantized weight for key: \(key)"
            case .unsupportedModule(let path):
                return "SDNQ quantized weights target an unsupported module: \(path)"
            }
        }
    }

    public static func isSDNQQuantized(_ weights: [String: MLXArray]) -> Bool {
        weights.keys.contains { $0.hasSuffix(".scale") }
            && weights.keys.contains { $0.hasSuffix(".zero_point") }
    }

    public static func applyWeights(
        url: URL,
        to model: Module,
        dtype: DType? = .bfloat16,
        keyMapper: ((String) -> String)? = nil,
        allowPlainUpdates: Bool = false,
        mapper: (String, MLXArray) -> [(String, MLXArray)] = { key, value in [(key, value)] }
    ) throws {
        let weights = try MLX.loadArrays(url: url)
        try applyWeightsFromArrays(
            weights,
            to: model,
            dtype: dtype,
            keyMapper: keyMapper,
            allowPlainUpdates: allowPlainUpdates,
            mapper: mapper
        )
    }

    public static func applyShardedWeights(
        indexURL: URL,
        to model: Module,
        dtype: DType? = .bfloat16,
        keyMapper: ((String) -> String)? = nil,
        mapper: (String, MLXArray) -> [(String, MLXArray)] = { key, value in [(key, value)] },
        progressHandler: (@Sendable (HFSafetensorsWeightsLoader.ShardProgress) -> Void)? = nil
    ) throws {
        let fm = FileManager.default
        guard fm.fileExists(atPath: indexURL.path) else {
            throw HFSafetensorsWeightsLoader.LoaderError.indexFileMissing(indexURL)
        }

        let indexData = try Data(contentsOf: indexURL)
        let index = try JSONDecoder().decode(HFSafetensorsIndex.self, from: indexData)
        let baseURL = indexURL.deletingLastPathComponent()
        let shards = index.shardFilenames

        for (shardIndex, shardFilename) in shards.enumerated() {
            let shardURL = baseURL.appending(path: shardFilename)
            guard fm.fileExists(atPath: shardURL.path) else {
                throw HFSafetensorsWeightsLoader.LoaderError.shardFileMissing(shardURL)
            }
            try applyWeights(
                url: shardURL,
                to: model,
                dtype: dtype,
                keyMapper: keyMapper,
                allowPlainUpdates: true,
                mapper: mapper
            )
            progressHandler?(HFSafetensorsWeightsLoader.ShardProgress(
                shardIndex: shardIndex,
                shardCount: shards.count,
                shardFilename: shardFilename
            ))
        }
    }

    public static func applyWeightsFromArrays(
        _ weights: [String: MLXArray],
        to model: Module,
        dtype: DType? = .bfloat16,
        keyMapper: ((String) -> String)? = nil,
        allowPlainUpdates: Bool = false,
        mapper: (String, MLXArray) -> [(String, MLXArray)] = { key, value in [(key, value)] }
    ) throws {
        var mappedWeights = weights
        if let keyMapper {
            var remapped: [String: MLXArray] = [:]
            remapped.reserveCapacity(weights.count)
            for (key, value) in weights {
                remapped[keyMapper(key)] = value
            }
            mappedWeights = remapped
        }

        guard isSDNQQuantized(mappedWeights) else {
            if allowPlainUpdates {
                let updates = mappedWeights.flatMap { key, value in
                    mapper(key, castIfNeeded(value, dtype: dtype))
                }
                if !updates.isEmpty {
                    try model.update(parameters: ModuleParameters.unflattened(updates), verify: .none)
                }
                return
            }
            throw LoaderError.notSDNQWeights
        }

        let leafModules = model.leafModules().flattened()
        var replacements: [String: Module] = [:]
        var parameterUpdates: [(String, MLXArray)] = []
        var usedKeys: Set<String> = []

        for (path, module) in leafModules {
            let scaleKey = "\(path).scale"
            guard let scale = mappedWeights[scaleKey] else { continue }

            let zeroPointKey = "\(path).zero_point"
            guard let zeroPoint = mappedWeights[zeroPointKey] else { continue }

            let weightKey = "\(path).weight"
            guard let quantizedWeight = mappedWeights[weightKey] else {
                throw LoaderError.missingWeight(weightKey)
            }

            let biasKey = "\(path).bias"
            let bias = mappedWeights[biasKey].map { castIfNeeded($0, dtype: dtype) }

            if let linear = module as? Linear {
                let (outputDimensions, inputDimensions) = linear.shape
                let groupCount = max(1, SDNQWeightsLoader.normalizedGroupCount(scale))
                let groupSize = inputDimensions / groupCount

                replacements[path] = SDNQUInt4Linear(
                    weight: quantizedWeight,
                    bias: bias,
                    scales: scale,
                    zeroPoints: zeroPoint,
                    inputDimensions: inputDimensions,
                    outputDimensions: outputDimensions,
                    groupSize: groupSize
                )
            } else if let embedding = module as? Embedding {
                let (embeddingCount, dimensions) = embedding.shape
                let groupCount = max(1, SDNQWeightsLoader.normalizedGroupCount(scale))
                let groupSize = dimensions / groupCount
                replacements[path] = SDNQUInt4Embedding(
                    weight: quantizedWeight,
                    scales: scale,
                    zeroPoints: zeroPoint,
                    embeddingCount: embeddingCount,
                    dimensions: dimensions,
                    groupSize: groupSize,
                    outputDType: dtype ?? .bfloat16
                )
            } else if let conv2d = module as? Conv2d {
                let dequantized = dequantizedConv2dWeight(
                    quantizedWeight,
                    scales: scale,
                    zeroPoints: zeroPoint,
                    conv: conv2d,
                    dtype: dtype ?? scale.dtype
                )
                parameterUpdates.append((weightKey, dequantized))
                if let bias {
                    parameterUpdates.append((biasKey, bias))
                }
            } else {
                throw LoaderError.unsupportedModule(path)
            }

            usedKeys.insert(scaleKey)
            usedKeys.insert(zeroPointKey)
            usedKeys.insert(weightKey)
            if bias != nil { usedKeys.insert(biasKey) }
        }

        if !replacements.isEmpty {
            try applyModuleReplacements(replacements, to: model)
        }

        var remainingUpdates = parameterUpdates
        for (key, value) in mappedWeights where !usedKeys.contains(key) {
            remainingUpdates.append(contentsOf: mapper(key, castIfNeeded(value, dtype: dtype)))
        }
        if !remainingUpdates.isEmpty {
            try model.update(parameters: ModuleParameters.unflattened(remainingUpdates), verify: .none)
        }
    }

    private static func castIfNeeded(_ value: MLXArray, dtype: DType?) -> MLXArray {
        guard let dtype, value.dtype != dtype else { return value }
        guard value.dtype.isFloatingPoint else { return value }
        return value.asType(dtype)
    }

    private static func normalizedGroupCount(_ value: MLXArray) -> Int {
        SDNQUInt4Dequantizer.groupCount(value)
    }

    private static func dequantizedConv2dWeight(
        _ weight: MLXArray,
        scales: MLXArray,
        zeroPoints: MLXArray,
        conv: Conv2d,
        dtype: DType
    ) -> MLXArray {
        let outputChannels = conv.weight.dim(0)
        let kernelHeight = conv.weight.dim(1)
        let kernelWidth = conv.weight.dim(2)
        let inputChannelsPerGroup = conv.weight.dim(3)
        let groupCount = max(1, normalizedGroupCount(scales))
        let groupSize = inputChannelsPerGroup / groupCount
        let quantizedShape: [Int]
        let resultShape: [Int]?

        if scales.ndim >= 5 {
            quantizedShape = [outputChannels, groupCount, groupSize, kernelHeight, kernelWidth]
            resultShape = [outputChannels, inputChannelsPerGroup, kernelHeight, kernelWidth]
        } else {
            quantizedShape = [outputChannels, inputChannelsPerGroup, kernelHeight, kernelWidth]
            resultShape = nil
        }

        let dequantizedOIHW = SDNQUInt4Dequantizer.dequantize(
            weight: weight,
            scales: scales,
            zeroPoints: zeroPoints,
            quantizedShape: quantizedShape,
            resultShape: resultShape,
            dtype: dtype
        )
        return HFSafetensorsWeightsLoader.convWeightOIHWToOHWI(dequantizedOIHW)
    }

    private static func applyModuleReplacements(_ replacements: [String: Module], to model: Module) throws {
        let modulesByPath = Dictionary(uniqueKeysWithValues: model.namedModules().map { ($0.0, $0.1) })
        var grouped: [String: [(String, Module)]] = [:]

        for (path, replacement) in replacements {
            let components = path.split(separator: ".").map(String.init)
            var parentPath: String?
            var childPath: String?
            for splitIndex in stride(from: components.count - 1, through: 0, by: -1) {
                let candidateParent = components[..<splitIndex].joined(separator: ".")
                if modulesByPath[candidateParent] != nil {
                    parentPath = candidateParent
                    childPath = components[splitIndex...].joined(separator: ".")
                    break
                }
            }
            guard let parentPath, let childPath else {
                throw LoaderError.unsupportedModule(path)
            }
            grouped[parentPath, default: []].append((childPath, replacement))
        }

        for (parentPath, updates) in grouped {
            guard let parent = modulesByPath[parentPath] else {
                throw LoaderError.unsupportedModule(parentPath)
            }
            try parent.update(modules: ModuleChildren.unflattened(updates), verify: .none)
        }
    }
}
